"""Step B of docs/llm_agent_tool_protocol.md §6: the agent takes and submits
its task over the harness MCP server.

Covers the server's tool semantics (`HarnessToolServer`), the Claude Code
argv for a tool-protocol call, and the harness side of a tool session: the
runtime is the only completion authority, and the final assistant text
carries no artifact.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent import agent_validators
from finesub.llm.agent.agent_mcp_server import (
    TOOL_NAMES,
    HarnessToolServer,
    request_id_for,
    serve,
)
from finesub.llm.agent.agent_task_runtime import (
    AgentTaskRuntime,
    AgentTaskSpec,
    ValidationResult,
)
from finesub.llm.agent.local_agent import (
    AGENT_TASK_PROMPT_TOOL_SESSION,
    AGENT_TASK_PROMPT_TOOL_SESSION_OPENING,
    ClaudeCodeDriverConfig,
    ClaudeCodeLocalAgentDriver,
    CodexDriverConfig,
    CodexLocalAgentDriver,
    DriverProbe,
    LocalAgentPolicyViolationError,
    LocalAgentEmptyTurnError,
    LocalAgentRefusedCallError,
    mcp_tool_name,
)
from finesub.llm.agent.channels import ChannelLimits
from finesub.llm.agent.local_agent import AgyDriverConfig
from finesub.llm.client import RoleClient
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import load_model_routes

AGY_CHANNEL = AgyDriverConfig().channel


@pytest.fixture(autouse=True)
def _register_reject_bad(monkeypatch):
    def build(_manifest):
        def validate(candidate, _m):
            if str(candidate).startswith("bad"):
                return ValidationResult.repairable("say good, not bad")
            return ValidationResult.accepted(candidate)

        return validate

    monkeypatch.setitem(agent_validators.VALIDATOR_BUILDERS, "test-reject-bad", build)


def _runtime(tmp_path, *, max_repairs=5):
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id="call",
                session_type="correction",
                input_hash="sha256:in",
                goal="answer",
                validator_id="test-reject-bad",
                protocol_key="correction",
                context_key="payload",
                metadata={"max_repair_attempts": max_repairs},
                required_blocks=(
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "payload", "digest": "@context"},
                ),
            )
        ],
        protocol_documents={"correction": "RULES: answer good"},
        context_documents={"payload": "the window text"},
        validators=agent_validators.runtime_validators("test-reject-bad"),
    )


def _server(runtime, session="s1"):
    return HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id=session,
        instance_id="instance-1",
    )


def _call(server, rpc_id, name, **arguments):
    reply = server.handle(
        {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    assert reply is not None
    result = reply["result"]
    return json.loads(result["content"][0]["text"]) if not result["isError"] else result


def test_the_server_walks_an_agent_from_next_task_to_accepted(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    server = _server(runtime)

    init = server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "finesub"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert [tool["name"] for tool in listed] == list(TOOL_NAMES)
    # Honest annotations: nothing that claims a lease or books a pull is
    # read-only or idempotent; nothing is destructive or open-world.
    by_name = {tool["name"]: tool["annotations"] for tool in listed}
    assert by_name["submit"] == {
        "title": "Submit the answer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
    assert by_name["pull_status"]["readOnlyHint"] is True
    unknown = server.handle({"jsonrpc": "2.0", "id": 2, "method": "server/discover"})
    assert unknown["error"]["code"] == -32601

    task = _call(server, 3, "next_task")
    assert task["status"] == "task"
    refs = {block["kind"]: block["ref"] for block in task["required_blocks"]}
    assert set(refs) == {"protocol", "payload"}
    assert task["repair_rounds_remaining"] == 5
    # The small required blocks ride the first answer and are booked as
    # pushed (docs §0-6): no round trip per block for a well-behaved session.
    assert task["blocks"]["protocol"].strip() == "RULES: answer good"
    assert task["blocks"]["payload"].strip() == "the window text"
    assert _call(server, 4, "pull_status")["owed_blocks"] == []

    # Re-reading (after a compact, say) still works and stays inside the task.
    protocol = _call(server, 5, "read_context", ref=refs["protocol"])
    assert protocol["text"].strip() == "RULES: answer good"
    assert protocol["owed_blocks"] == []
    stray = _call(server, 6, "read_context", ref="control/bootstrap.md#sha256:x")
    assert stray["isError"] is True and "not a resource" in stray["content"][0]["text"]
    garbage = _call(server, 7, "read_context", ref="not-a-runtime-ref")
    assert garbage["isError"] is True

    rejected = _call(server, 8, "submit", payload="bad")
    assert rejected["status"] == "repairable"
    assert rejected["validation_errors"] == ["say good, not bad"]
    assert rejected["repair_rounds_remaining"] == 5, "the first rejection opens round 1 of 5"
    # A transport replay of the same JSON-RPC id is the same answer; the
    # repair budget does not move twice.
    assert _call(server, 8, "submit", payload="bad") == rejected
    accepted = _call(server, 9, "submit", payload="good")
    assert accepted["status"] == "accepted"
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "accepted"
    assert record["last_candidate"] == "bad"


def test_a_validators_repair_instruction_reaches_the_model(tmp_path, monkeypatch) -> None:
    """The server is generic: what to send next comes from the validator (a patch, for a window answer)."""

    def build(_manifest):
        return lambda candidate, _m: ValidationResult.repairable(
            "row 4 missing", metadata={"repair_instruction": "send only row 4"})

    monkeypatch.setitem(agent_validators.VALIDATOR_BUILDERS, "test-reject-bad", build)
    server = _server(_runtime(tmp_path))
    _call(server, 1, "next_task")
    rejected = _call(server, 2, "submit", payload="anything")
    assert rejected["validation_errors"] == ["row 4 missing"]
    assert rejected["repair_instruction"] == "send only row 4"


def test_the_gate_still_bites_when_a_block_is_not_pushed(tmp_path) -> None:
    """A required block outside the pushed kinds must be read, and a session
    that still owes it after its one repair is retired by the runtime."""

    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id="call",
                session_type="correction",
                input_hash="sha256:in",
                goal="answer",
                validator_id="test-reject-bad",
                protocol_key="correction",
                context_key="payload",
                required_blocks=(
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "index", "digest": "@context"},
                ),
            )
        ],
        protocol_documents={"correction": "RULES"},
        context_documents={"payload": "INDEX"},
        validators=agent_validators.runtime_validators("test-reject-bad"),
    )
    server = _server(runtime)
    task = _call(server, 1, "next_task")
    assert list(task["blocks"]) == ["protocol"]
    early = _call(server, 2, "submit", payload="good")
    assert early["status"] == "repairable"
    assert [block["kind"] for block in early["owed_blocks"]] == ["index"]
    retired = _call(server, 3, "submit", payload="good")
    assert retired["status"] == "retired"
    assert _call(server, 4, "submit", payload="good")["status"] == "retired"
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "queued" and record["retirements"] == 1


def test_the_server_stops_submits_once_the_repair_budget_is_spent(tmp_path) -> None:
    runtime = _runtime(tmp_path, max_repairs=1)
    server = _server(runtime)
    task = _call(server, 1, "next_task")
    for index, block in enumerate(task["required_blocks"]):
        _call(server, 10 + index, "read_context", ref=block["ref"])

    first = _call(server, 20, "submit", payload="bad")
    assert first["status"] == "repairable", "budget 1: the first answer plus one repair"
    second = _call(server, 21, "submit", payload="bad again")
    assert second["status"] == "repair_exhausted"
    again = _call(server, 22, "submit", payload="good")
    assert again["status"] == "repair_exhausted"
    assert runtime.task_record(assignment_id="assignment-1", task_id="call")["status"] == "repairing"


def test_request_ids_derive_from_session_instance_and_rpc_id(tmp_path) -> None:
    args = {
        "assignment_id": "a",
        "worker_id": "w",
        "session_id": "s",
        "instance_id": "i",
    }
    same = request_id_for(**args, rpc_id=1)
    assert same == request_id_for(**args, rpc_id=1)
    assert same != request_id_for(**args, rpc_id="1")
    assert same != request_id_for(**{**args, "session_id": "s2"}, rpc_id=1)
    assert same != request_id_for(**{**args, "instance_id": "i2"}, rpc_id=1)

    # A transport replay on the same live server is one logical call.
    runtime = _runtime(tmp_path)
    server = _server(runtime)
    task = _call(server, 1, "next_task")
    for index, block in enumerate(task["required_blocks"]):
        _call(server, 10 + index, "read_context", ref=block["ref"])
    first = _call(server, 20, "submit", payload="bad")
    replay = _call(server, 20, "submit", payload="bad")
    assert replay["validation_errors"] == first["validation_errors"]
    assert runtime.task_record(assignment_id="assignment-1", task_id="call")["validation_errors"] == [
        "say good, not bad"
    ]


def test_restarted_mcp_process_can_reuse_rpc_id_for_a_different_call(tmp_path) -> None:
    """Claude reconnects MCP inside one CLI and resets its JSON-RPC counter."""

    runtime = _runtime(tmp_path)
    old = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="same-cli",
        instance_id="old-process",
    )
    _call(old, 2, "next_task")

    restarted = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="same-cli",
        instance_id="new-process",
    )
    _call(restarted, 1, "next_task")
    accepted = _call(restarted, 2, "submit", payload="good")

    assert accepted["status"] == "accepted"
    assert runtime.task_record(assignment_id="assignment-1", task_id="call")["status"] == "accepted"


def test_serve_pumps_newline_delimited_json_and_logs_frames(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    frames = "\n".join(
        json.dumps(message)
        for message in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
    ) + "\n"
    out = io.StringIO()
    log = tmp_path / "frames.jsonl"
    serve(io.BytesIO(frames.encode("utf-8")), out, _server(runtime), log_path=str(log))
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2]
    logged = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [row["dir"] for row in logged] == ["in", "out", "in", "in", "out"]


def test_claude_argv_declares_the_server_inline_and_drops_safe_mode(tmp_path) -> None:
    driver = ClaudeCodeLocalAgentDriver(ClaudeCodeDriverConfig(command=("claude",)))
    driver._resolved_command = ("claude",)
    probe = DriverProbe(
        available=True,
        structured_events=True,
        no_persisted_session=True,
        no_user_config=True,
        no_user_rules=True,
        can_restrict_tools=True,
        has_web_search=True,
        supports_session_reuse=True,
        sandbox_kind="named_tool_allowlist",
        supports_mcp_config=True,
    )
    capsule = SimpleNamespace(root=tmp_path, episode_id="ep", continuation_of="")
    server = {
        "command": "python",
        "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
        "env": {"FINESUB_MCP_SESSION": "s1"},
        "tools": list(TOOL_NAMES),
    }

    plain = driver._argv(capsule, native_search=False, probe=probe)
    tooled = driver._argv(capsule, native_search=False, probe=probe, mcp_server=server)

    assert "--safe-mode" in plain and "--safe-mode" not in tooled
    assert tooled[-1] == AGENT_TASK_PROMPT_TOOL_SESSION
    opened = driver._argv(
        capsule, native_search=False, probe=probe, mcp_server={**server, "opening": True}
    )
    assert opened[-1] == AGENT_TASK_PROMPT_TOOL_SESSION_OPENING
    assert "--strict-mcp-config" in tooled
    config = json.loads(tooled[tooled.index("--mcp-config") + 1])
    assert config["mcpServers"]["finesub"]["env"]["FINESUB_MCP_SESSION"] == "s1"
    allowed = tooled[tooled.index("--allowed-tools") + 1].split(",")
    assert allowed == sorted(mcp_tool_name(name) for name in TOOL_NAMES)
    assert tooled[tooled.index("--tools") + 1] == ""
    assert plain[plain.index("--tools") + 1] == ""
    assert "--disallowed-tools" not in tooled
    assert "--allowed-tools" not in plain
    assert tooled[-1] == AGENT_TASK_PROMPT_TOOL_SESSION
    assert "Do not use shell, filesystem-write, MCP" not in tooled[-1]


def test_codex_tool_session_uses_the_mcp_worker_framing(tmp_path) -> None:
    driver = CodexLocalAgentDriver(CodexDriverConfig(command=("codex",)))
    driver._resolved_command = ("codex",)
    capsule = SimpleNamespace(root=tmp_path, episode_id="ep", continuation_of="")
    probe = SimpleNamespace(no_user_config=True, no_user_rules=True)
    server = {
        "command": "python",
        "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
        "env": {"FINESUB_MCP_SESSION": "s1"},
        "tools": list(TOOL_NAMES),
    }

    argv = driver._argv(
        capsule, native_search=False, probe=probe, mcp_server=server
    )

    assert argv[-1] == AGENT_TASK_PROMPT_TOOL_SESSION
    assert "Do not use shell, filesystem-write, MCP" not in argv[-1]
    opened = driver._argv(
        capsule, native_search=False, probe=probe, mcp_server={**server, "opening": True}
    )
    assert opened[-1] == AGENT_TASK_PROMPT_TOOL_SESSION_OPENING
    assert "next_task" not in opened[-1]


class _ToolUsingFakeDriver:
    """Plays the agent: drives the harness server in-process as the CLI would."""

    conversation_ttl_seconds = 0.0

    def __init__(self, script, *, mcp_opening: str = "", channel: ChannelLimits = ChannelLimits()) -> None:
        self.config = SimpleNamespace(model="", mcp_opening=mcp_opening, channel=channel)
        self.script = script
        self.calls: list = []
        self.servers: list = []

    def probe(self, *, refresh: bool = False):
        return DriverProbe(
            available=True,
            structured_events=True,
            no_persisted_session=True,
            no_user_config=True,
            no_user_rules=True,
            can_restrict_tools=True,
            has_web_search=True,
            supports_session_reuse=True,
            sandbox_kind="named_tool_allowlist",
            supports_mcp_config=True,
        )

    def meets_requirements(self, probe=None, *, native_search: bool = False):
        return True

    # Set to simulate a CLI that keeps talking after its submit: `run` then
    # only returns once the harness's completion predicate says so.
    hang_after_script = False

    def run(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        spec = kwargs["mcp_server"]
        env = spec["env"]
        runtime = AgentTaskRuntime(
            env["FINESUB_MCP_ROOT"],
            validators=agent_validators.runtime_validators("test-reject-bad"),
        )
        server = HarnessToolServer(
            runtime,
            assignment_id=env["FINESUB_MCP_ASSIGNMENT"],
            worker_id=env["FINESUB_MCP_WORKER"],
            session_id=env["FINESUB_MCP_SESSION"],
            instance_id="fake-instance",
            predelivered=[key for key in env.get("FINESUB_MCP_PREDELIVERED", "").split(",") if key],
        )
        self.servers.append(server)
        observer = kwargs.get("observer")
        if observer is not None:
            observer.started({})
        self.script(server)
        if self.hang_after_script:
            completion = kwargs["completion"]
            assert completion(), "the runtime knows the task is accepted; the driver may reclaim"
        return SimpleNamespace(
            content="submitted",
            reported_model="claude",
            execution_attempt={"backend": "local_agent", "capsule_id": "cap-1"},
            episode_id="cap-1",
            conversation_handle="",
            turn_identity="",
            normalized_events=({"event": "result"},),
            usage={"input_tokens": 7, "output_tokens": 1, "total_tokens": 8},
        )


def _client(driver, tmp_path):
    routes = load_model_routes(
        user_config={
            "model_groups": {
                "research-default": {"targets": ["local-claude-sonnet"]}
            }
        }
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    return RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={LLMRole.GENERAL_CAPABLE: role_config_for("research", "quality", routes=routes)},
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
        agent_assignment_root=tmp_path / "assignments",
    )


def _no_api(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )


MESSAGES = [
    {"role": "system", "content": "RULES: answer good"},
    {"role": "user", "content": [{"text": "fix this window"}]},
]


def test_a_tool_session_reads_the_harness_messages_and_submits(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    seen: dict = {}

    def agent(server):
        task = _call(server, 1, "next_task")
        seen["refs"] = {
            block["kind"]: _call(server, 10 + index, "read_context", ref=block["ref"])["text"]
            for index, block in enumerate(task["required_blocks"])
        }
        seen["rejected"] = _call(server, 3, "submit", payload="bad")
        seen["accepted"] = _call(server, 4, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent)
    client = _client(driver, tmp_path)

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=3,
    )

    # The agent saw the harness's own prompt as protocol and payload...
    assert seen["refs"]["protocol"].strip() == "RULES: answer good"
    assert seen["refs"]["payload"].strip() == "fix this window"
    assert seen["rejected"]["status"] == "repairable"
    assert seen["accepted"]["status"] == "accepted"
    # ...and the answer is the accepted artifact, not the chat text.
    assert result.content == "good"
    assert result.repair_exhausted is False
    messages, kwargs = driver.calls[0]
    assert "next_task" in messages[0]["content"]
    assert kwargs["session_scope"] == "task"
    assert kwargs["mcp_server"]["tools"] == list(TOOL_NAMES)
    assert kwargs["mcp_server"]["args"][-1] == "finesub.llm.agent.agent_mcp_server"
    # The call was accepted and no capsule survives it (a real driver prunes
    # its own on a clean run; the fake one never made one), so there is no
    # evidence left to keep and the root goes with it.
    assert list((tmp_path / "assignments").iterdir()) == []


def test_a_tool_session_that_leaves_rejected_reports_the_last_candidate(
    tmp_path, monkeypatch
) -> None:
    _no_api(monkeypatch)

    def agent(server):
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="bad")

    driver = _ToolUsingFakeDriver(agent)
    client = _client(driver, tmp_path)

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=1,
    )

    assert result.content == "bad"
    assert result.repair_exhausted is True
    assert len(list((tmp_path / "assignments").iterdir())) == 1


def _empty_turn_error():
    error = LocalAgentEmptyTurnError("event stream did not contain a final assistant message")
    error._harness_execution_attempts = [{"capsule_id": "cap-0"}]
    return error


def test_an_empty_turn_after_a_rejected_submit_is_the_session_leaving_rejected(
    tmp_path, monkeypatch
) -> None:
    """Codex Luna (2026-11-05): refused, resubmitted unchanged, then ended the
    turn empty. That is the model leaving its repair -- the chain is spent and
    the caller replaces the session -- not a transient fault that walks the
    route chain away from the window. The rounds reported are the ones used,
    so a caller counting them against its own budget still has the rest."""

    _no_api(monkeypatch)

    def agent(server):
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="bad")
        _call(server, 21, "submit", payload="bad again")
        raise _empty_turn_error()

    driver = _ToolUsingFakeDriver(agent)
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=5,
    )

    assert result.content == "bad again"
    assert result.repair_exhausted is True
    assert result.agent_repair_rounds == 1
    assert len(driver.calls) == 1


def test_an_empty_turn_without_a_submit_is_still_the_targets_failure(
    tmp_path, monkeypatch
) -> None:
    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "next_task")
        raise _empty_turn_error()

    driver = _ToolUsingFakeDriver(agent)
    with pytest.raises(LocalAgentEmptyTurnError):
        _client(driver, tmp_path).complete(
            LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
        )
    assert len(driver.calls) == 1


def test_a_tool_session_that_never_submits_is_a_failed_call(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    driver = _ToolUsingFakeDriver(lambda server: _call(server, 1, "next_task"))
    client = _client(driver, tmp_path)

    with pytest.raises(Exception) as excinfo:
        client.complete(
            LLMRole.GENERAL_CAPABLE,
            MESSAGES,
            validator_spec={"id": "test-reject-bad", "params": {}},
        )
    assert "without an accepted submit" in str(excinfo.value)


def _refused_then(error_type):
    """A first session that dies with ``error_type``, then one that answers."""

    seen: list = []

    def agent(server):
        if not seen:
            seen.append("refused")
            error = error_type("agy refused a malformed call")
            error._harness_execution_attempts = [
                {"capsule_id": "cap-0", "refused_tool_calls": ["finesub/call_mcp_tool"]}
            ]
            raise error
        seen.append("answered")
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="good")

    return agent, seen


def test_a_turn_ended_by_a_refused_malformed_call_is_asked_again(tmp_path, monkeypatch) -> None:
    """Owner 2026-10-31: the model's malformed call that agy refused is a slip,
    not a breach -- the premature-stop retry takes it."""

    _no_api(monkeypatch)
    agent, seen = _refused_then(LocalAgentRefusedCallError)
    driver = _ToolUsingFakeDriver(agent)
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    assert seen == ["refused", "answered"]
    first = result.execution_attempts[0]
    assert first["premature_stop"] is True
    assert first["refused_tool_calls"] == ["finesub/call_mcp_tool"]


def test_a_policy_violation_still_ends_the_session_unretried(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    agent, seen = _refused_then(LocalAgentPolicyViolationError)
    driver = _ToolUsingFakeDriver(agent)
    with pytest.raises(LocalAgentPolicyViolationError):
        _client(driver, tmp_path).complete(
            LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
        )
    assert seen == ["refused"]


def test_a_restarted_server_process_reads_the_repair_budget_from_the_task_row(
    tmp_path,
) -> None:
    """docs §7: the budget is durable, so an MCP restart cannot refill it."""

    runtime = _runtime(tmp_path, max_repairs=2)
    first = _server(runtime)
    task = _call(first, 1, "next_task")
    assert task["repair_rounds_remaining"] == 2
    assert _call(first, 2, "submit", payload="bad")["repair_rounds_remaining"] == 2
    assert _call(first, 3, "submit", payload="bad again")["repair_rounds_remaining"] == 1

    # Claude Code restarts the MCP subprocess inside one CLI: new instance,
    # JSON-RPC ids from 1 again, same worker and lease.
    restarted = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-2",
    )
    resumed = _call(restarted, 1, "next_task")
    assert resumed["repair_rounds_remaining"] == 1
    # The same rejected answer again is replayed, not charged.
    replay = _call(restarted, 2, "submit", payload="bad again")
    assert replay["repair_rounds_remaining"] == 1
    assert "already judged" in replay["message"]
    # The one remaining round is still there to be used.
    assert _call(restarted, 3, "submit", payload="good")["status"] == "accepted"


def test_a_server_that_starts_after_the_accept_tells_the_session_to_stop(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    first = _server(runtime)
    _call(first, 1, "next_task")
    assert _call(first, 2, "submit", payload="good")["status"] == "accepted"

    late = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s2",
        instance_id="instance-2",
    )
    reply = _call(late, 1, "next_task")
    assert reply["status"] == "assignment_complete"
    assert "stop" in reply["message"]
    # No task in hand, so a stray submit is refused rather than judged.
    assert _call(late, 2, "submit", payload="good")["isError"] is True


def test_a_session_that_repeats_one_rejected_answer_is_retired_through_the_server(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, max_repairs=1)
    server = _server(runtime)
    _call(server, 1, "next_task")
    cap = runtime.task_record(assignment_id="assignment-1", task_id="call")["max_submits"]
    statuses = [
        _call(server, 10 + index, "submit", payload="bad")["status"] for index in range(cap)
    ]
    # Budget 1: the first rejection opens the one repair round; replays of it are not charged.
    assert set(statuses) == {"repairable"}
    assert runtime.task_record(assignment_id="assignment-1", task_id="call")["repair_rounds_remaining"] == 1
    assert _call(server, 10 + cap, "submit", payload="bad")["status"] == "retired"


def test_claude_raises_its_mcp_reply_limit_only_for_a_tool_session(tmp_path) -> None:
    """`MAX_MCP_OUTPUT_TOKENS` is set in the CLI's own process environment for
    this invocation (the user's settings are never written)."""

    driver = ClaudeCodeLocalAgentDriver(ClaudeCodeDriverConfig(runtime_root=tmp_path / "runtime"))
    env = driver._spawn_environment(mcp_server={"command": "x"})
    assert env["MAX_MCP_OUTPUT_TOKENS"] == "200000"
    assert env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] == "200000"
    assert "MAX_MCP_OUTPUT_TOKENS" not in driver._spawn_environment(mcp_server=None)


# --- exit-on-accept ---------------------------------------------------------


def test_the_completing_accept_is_logged_then_held(tmp_path, monkeypatch) -> None:
    """docs §4 exit-on-accept: the reply that completes the assignment is
    logged at once and held before it reaches the CLI, so the driver ends the
    CLI before it can spend a request on "submitted". Nothing else is held."""

    from finesub.llm.agent import agent_mcp_server

    runtime = _runtime(tmp_path)
    server = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-1",
        accepted_reply_hold_seconds=7.0,
    )
    log = tmp_path / "frames.jsonl"
    slept: list = []

    def fake_sleep(seconds):
        logged = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        slept.append((seconds, logged[-1]["dir"]))

    monkeypatch.setattr(agent_mcp_server.time, "sleep", fake_sleep)
    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "next_task", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "submit", "arguments": {"payload": "bad"}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "submit", "arguments": {"payload": "good"}}},
    ]
    out = io.StringIO()
    serve(
        io.BytesIO("".join(json.dumps(f) + "\n" for f in frames).encode("utf-8")),
        out,
        server,
        log_path=str(log),
    )

    assert slept == [(7.0, "out")]
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    # Still sent when nobody ends the process: a hold costs time, never the call.
    assert json.loads(replies[-1]["result"]["content"][0]["text"])["status"] == "accepted"
    assert server.take_reply_hold() == 0.0


def test_a_tool_session_asks_for_exit_on_accept_from_the_driver_config(
    tmp_path, monkeypatch
) -> None:
    from finesub.llm.agent.agent_mcp_server import ACCEPTED_REPLY_HOLD_ENV
    from finesub.llm.agent.local_agent import ACCEPTED_REPLY_HOLD_SECONDS

    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "next_task")
        _call(server, 2, "submit", payload="good")

    for configured, expected in ((True, True), (False, False), (None, False)):
        driver = _ToolUsingFakeDriver(agent)
        if configured is not None:
            driver.config = SimpleNamespace(model="", exit_on_accept=configured)
        client = _client(driver, tmp_path / str(configured))
        client.complete(
            LLMRole.GENERAL_CAPABLE,
            MESSAGES,
            validator_spec={"id": "test-reject-bad", "params": {}},
            max_repair_attempts=3,
        )
        (_messages, kwargs), = driver.calls
        assert kwargs["exit_on_completion"] is expected
        env = kwargs["mcp_server"]["env"]
        if expected:
            assert float(env[ACCEPTED_REPLY_HOLD_ENV]) == ACCEPTED_REPLY_HOLD_SECONDS
        else:
            assert ACCEPTED_REPLY_HOLD_ENV not in env


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        # owner 2026-09-30: Codex reports usage once, at the end, so `auto`
        # leaves it off; the drivers that can rebuild usage turn it on.
        ("auto", {"codex": False, "claude_code": True, "codebuddy": True, "agy": True, "dsh": False}),
        ("on", {"codex": True, "claude_code": True, "codebuddy": True, "agy": True, "dsh": True}),
        ("off", {"codex": False, "claude_code": False, "codebuddy": False, "agy": False, "dsh": False}),
    ],
)
def test_exit_on_accept_resolves_per_driver(mode, expected) -> None:
    from finesub.llm.routing.execution_policy import ExecutionSettings

    settings = ExecutionSettings(local_agent_exit_on_accept=mode)
    got = {
        name: getattr(settings, f"{name}_driver_config")(model="m").exit_on_accept
        for name in expected
    }
    assert got == expected


def _harness_claims(runtime, *, kinds=("protocol", "payload")):
    """What the opening does before the CLI starts: claim the task as
    `worker-1` and book the blocks it carries as pushed."""

    status = runtime.rehydrate(assignment_id="assignment-1", worker_id="worker-1")
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="opening-claim",
        expected_control_generation=status["control_generation"],
    )
    task = claimed["task"]
    manifest = json.loads(runtime.read_artifact(task["manifest_ref"]))
    blocks = [block for block in manifest["required_blocks"] if block["kind"] in kinds]
    runtime.record_pull(
        assignment_id="assignment-1",
        task_id=task["task_id"],
        worker_id="worker-1",
        lease_generation=int(task["lease_generation"]),
        request_id="opening-push",
        blocks=blocks,
        source="push",
    )
    return [f"{block['kind']}@{block['digest']}" for block in blocks]


def test_after_the_opening_a_submit_needs_no_next_task(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    keys = _harness_claims(runtime)
    server = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-1",
        predelivered=keys,
    )

    assert _call(server, 1, "pull_status")["owed_blocks"] == []
    rejected = _call(server, 2, "submit", payload="bad")
    assert rejected["status"] == "repairable"
    accepted = _call(server, 3, "submit", payload="good")
    assert accepted["status"] == "accepted"
    # Afterwards the session is told what it is told today: there is no task
    # left to submit to, and `next_task` says the work is done.
    again = _call(server, 4, "submit", payload="good")
    assert again["isError"] is True and "call next_task first" in again["content"][0]["text"]
    assert _call(server, 5, "next_task")["status"] == "assignment_complete"
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "accepted" and record["repair_rounds_remaining"] == 5


def test_a_next_task_after_the_opening_says_it_was_sent_and_resends_on_request(tmp_path) -> None:
    """The opening already gave this CLI its task: `next_task` only says so,
    and names `resend` (not in the tool's description) for a session that
    lost it."""

    runtime = _runtime(tmp_path)
    keys = _harness_claims(runtime)
    server = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-1",
        predelivered=keys,
    )
    definition = next(tool for tool in server.tool_definitions if tool["name"] == "next_task")
    assert "resend" not in definition["description"]

    task = _call(server, 1, "next_task")
    reads = {block["kind"]: block.get("read") for block in task["required_blocks"]}
    assert reads == {"protocol": "seen", "payload": "seen"} and task["blocks"] == {}
    assert '"resend": true' in task["note"]
    again = _call(server, 2, "next_task", resend=True)
    assert again["blocks"]["payload"].strip() == "the window text" and "note" not in again
    # The protocol another delivery already booked stays seen.
    assert "protocol" not in again["blocks"]
    assert _call(server, 3, "submit", payload="good")["status"] == "accepted"


def test_an_opening_whose_push_was_not_booked_gets_everything(tmp_path) -> None:
    """The env names a block the ledger does not hold (the booking never
    happened): `seen` needs both, so the protocol is sent."""

    runtime = _runtime(tmp_path)
    keys = _harness_claims(runtime, kinds=("payload",))
    manifest_keys = keys + ["protocol@" + json.loads(
        runtime.read_artifact(
            runtime.rehydrate(assignment_id="assignment-1", worker_id="worker-1")["task"]["manifest_ref"]
        )
    )["required_blocks"][0]["digest"]]
    server = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-1",
        predelivered=manifest_keys,
    )

    task = _call(server, 1, "next_task")
    assert task["blocks"]["protocol"].strip() == "RULES: answer good"


def test_a_restarted_mcp_process_finds_the_task_by_its_lease(tmp_path) -> None:
    """Claude Code restarts the MCP process inside one CLI; the model, which
    already has its task, goes straight to `submit`."""

    runtime = _runtime(tmp_path)
    first = _server(runtime)
    _call(first, 1, "next_task")

    restarted = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="new-process",
    )
    assert _call(restarted, 1, "submit", payload="good")["status"] == "accepted"


def test_without_a_lease_the_server_still_asks_for_next_task(tmp_path) -> None:
    server = _server(_runtime(tmp_path))
    early = _call(server, 1, "submit", payload="good")
    assert early["isError"] is True and "call next_task first" in early["content"][0]["text"]


def _runtime_with_index(tmp_path):
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id="call",
                session_type="correction",
                input_hash="sha256:in",
                goal="answer",
                validator_id="test-reject-bad",
                protocol_key="correction",
                context_key="payload",
                required_blocks=(
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "kb_index", "ref": "kb_index", "digest": "index-digest"},
                    {"kind": "payload", "digest": "@context"},
                ),
            )
        ],
        protocol_documents={"correction": "RULES: answer good"},
        context_documents={"payload": "the window text"},
        validators=agent_validators.runtime_validators("test-reject-bad"),
    )


@pytest.mark.parametrize("opened_with_index", [True, False])
def test_an_index_the_opening_carried_is_seen_and_owes_nothing(tmp_path, opened_with_index) -> None:
    runtime = _runtime_with_index(tmp_path)
    kinds = ("protocol", "kb_index", "payload") if opened_with_index else ("protocol", "payload")
    keys = _harness_claims(runtime, kinds=kinds)
    server = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-1",
        predelivered=keys,
    )

    task = _call(server, 1, "next_task")
    index = next(block for block in task["required_blocks"] if block["kind"] == "kb_index")
    assert index["tool"] == "kb_index"
    assert index["read"] == ("seen" if opened_with_index else "tool")
    owed = [block["kind"] for block in _call(server, 2, "pull_status")["owed_blocks"]]
    assert owed == ([] if opened_with_index else ["kb_index"])



@pytest.fixture
def _opening_on(monkeypatch):
    """The default: no dev switch in the environment."""

    monkeypatch.delenv("FINESUB_TOOL_SESSION_OPENING", raising=False)


def test_an_opening_session_submits_without_next_task(tmp_path, monkeypatch, _opening_on) -> None:
    _no_api(monkeypatch)
    seen: dict = {}

    def agent(server):
        seen["owed"] = _call(server, 1, "pull_status")["owed_blocks"]
        seen["accepted"] = _call(server, 2, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="inline")
    client = _client(driver, tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=3,
    )

    assert result.content == "good"
    assert seen["owed"] == [] and seen["accepted"]["status"] == "accepted"
    messages, kwargs = driver.calls[0]
    opening = messages[0]["content"]
    # The blocks are in the opening, the shared one before the task's own.
    assert opening.index('<block kind="protocol">') < opening.index("The task manifest:")
    assert opening.index("The task manifest:") < opening.index('<block kind="payload">')
    assert "RULES: answer good" in opening and "fix this window" in opening
    assert '"read": "opening"' in opening
    assert kwargs["mcp_server"]["opening"] is True
    assert kwargs["observer"] is not None
    keys = kwargs["mcp_server"]["env"]["FINESUB_MCP_PREDELIVERED"].split(",")
    assert sorted(key.split("@")[0] for key in keys) == ["payload", "protocol"]


def test_the_manifest_says_how_the_task_was_handed_over(tmp_path, monkeypatch, _opening_on) -> None:
    _no_api(monkeypatch)
    seen: dict = {}

    def agent(server):
        _task, manifest = server._require_task()
        seen["opening"] = manifest["metadata"].get("tool_session_opening")
        _call(server, 1, "submit", payload="good")

    client = _client(_ToolUsingFakeDriver(agent, mcp_opening="inline"), tmp_path)
    client.complete(LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}})
    assert seen["opening"] == "inline"


def test_the_switch_off_opens_the_session_as_before(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    monkeypatch.setenv("FINESUB_TOOL_SESSION_OPENING", "0")

    def agent(server):
        _call(server, 1, "next_task")
        _call(server, 2, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="inline")
    _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    messages, kwargs = driver.calls[0]
    assert "Call `next_task`" in messages[0]["content"]
    assert "opening" not in kwargs["mcp_server"]
    assert "FINESUB_MCP_PREDELIVERED" not in kwargs["mcp_server"]["env"]
    assert kwargs["observer"] is None


def test_a_premature_stop_gets_the_same_opening_and_a_fresh_booking(
    tmp_path, monkeypatch, _opening_on
) -> None:
    """The reset clears the ledger; the second CLI is handed the same opening
    and the harness books it again, so it owes nothing either."""

    _no_api(monkeypatch)
    seen: list = []

    def agent(server):
        if not seen:
            seen.append("left")
            return
        seen.append(_call(server, 1, "pull_status")["owed_blocks"])
        _call(server, 2, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="inline")
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    assert seen == ["left", []]
    first, second = (call[0][0]["content"] for call in driver.calls)
    assert first == second


def _single_task_runtime(tmp_path, *, metadata=None, blocks=None, payload="the window text"):
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id="call",
                session_type="correction",
                input_hash="sha256:in",
                goal="answer",
                validator_id="test-reject-bad",
                protocol_key="correction",
                context_key="payload",
                metadata=metadata or {},
                required_blocks=blocks
                or (
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "payload", "digest": "@context"},
                ),
            )
        ],
        protocol_documents={"correction": "RULES: answer good"},
        context_documents={"payload": payload},
        validators=agent_validators.runtime_validators("test-reject-bad"),
    )


def test_the_claim_holds_the_lease_for_the_whole_call(tmp_path) -> None:
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(tmp_path)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="inline", limits=ChannelLimits()
    )
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "leased" and record["lease_owner"] == "worker-1"
    assert int(claimed["task"]["lease_generation"]) == int(record["lease_generation"])


def test_agy_carries_the_protocol_in_its_agent_document(tmp_path) -> None:
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(tmp_path)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="agent_document", limits=AGY_CHANNEL
    )
    reads = {entry["kind"]: entry for entry in claimed["manifest"]["required_blocks"]}
    assert reads["protocol"]["read"] == "system"
    assert claimed["agent_protocol"].strip() == "RULES: answer good"
    assert reads["payload"]["read"] == "file" and "tool" not in reads["payload"]
    paths = [Path(line) for line in claimed["block_files"].splitlines()]
    assert len(paths) == 1 and paths[0].resolve().is_relative_to(runtime.root)
    assert '<block kind="payload">\nthe window text\n</block>' in paths[0].read_text(encoding="utf-8")
    assert claimed["shared_blocks"] == [] and claimed["task_blocks"] == []
    assert sorted(block["kind"] for block in claimed["pushed"]) == ["payload", "protocol"]


def test_agy_names_every_part_of_a_large_payload_and_a_range_past_its_argv(tmp_path) -> None:
    """Each part is one whole `view_file`; the list falls back to a range
    once the opening would pass agy's argv bound."""

    from dataclasses import replace

    from finesub.llm.client import _claim_for_opening

    payload = "".join(f"{index:05d} 台詞「引用」\n" for index in range(5000))
    runtime = _single_task_runtime(tmp_path, payload=payload)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="agent_document", limits=AGY_CHANNEL
    )
    paths = [Path(line) for line in claimed["block_files"].splitlines()]
    assert len(paths) >= 3
    text = ""
    for path in paths:
        part = path.read_text(encoding="utf-8")
        assert len(part.encode("utf-8")) <= AGY_CHANNEL.part_bytes and len(part.splitlines()) <= AGY_CHANNEL.part_lines
        text += part
    assert payload.rstrip() in text
    tight = replace(AGY_CHANNEL, opening_max_chars=4_000)
    ranged = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="agent_document", limits=tight
    )
    assert ranged["block_files"].endswith(f"part-{len(paths):02d}.md ({len(paths)} files, numbered in order)")


def test_the_system_prompt_opening_names_the_protocol_system(tmp_path) -> None:
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(tmp_path)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="system_prompt", limits=ChannelLimits()
    )
    reads = {entry["kind"]: entry for entry in claimed["manifest"]["required_blocks"]}
    assert reads["protocol"]["read"] == "system"
    assert claimed["agent_protocol"].strip() == "RULES: answer good"
    assert reads["payload"]["read"] == "opening"
    assert claimed["shared_blocks"] == []
    assert [kind for kind, _ in claimed["task_blocks"]] == ["payload"]


def test_the_fork_opening_names_the_protocol_earlier(tmp_path) -> None:
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(tmp_path)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="fork", limits=ChannelLimits()
    )
    reads = {entry["kind"]: entry for entry in claimed["manifest"]["required_blocks"]}
    assert reads["protocol"]["read"] == "earlier"
    assert claimed["agent_protocol"].strip() == "RULES: answer good"
    assert reads["payload"]["read"] == "opening"
    assert claimed["shared_blocks"] == []
    assert [kind for kind, _ in claimed["task_blocks"]] == ["payload"]


@pytest.mark.parametrize(
    "mode, declared, large, expected",
    [
        ("inline", True, False, "opening"),
        ("inline", False, False, "tool"),
        ("inline", True, True, "tool"),
        ("agent_document", True, False, "file"),
        ("agent_document", True, True, "file"),
        ("system_prompt", True, False, "opening"),
    ],
)
def test_the_index_rides_the_opening_as_declared_and_while_it_fits(
    tmp_path, monkeypatch, mode, declared, large, expected
) -> None:
    """Carried only as the manifest declared it; inline only while the
    opening stays within the channel's bound, with the payload after it still
    carried when the index is not. agy's parts take any size."""

    import hashlib

    from finesub.llm.client import _claim_for_opening
    from finesub.llm.knowledge import base

    text = "INDEX\n" * (40_000 if large else 1)
    digest = hashlib.sha256((text if declared else "OTHER").encode()).hexdigest()
    runtime = _single_task_runtime(
        tmp_path,
        metadata={"kb_root": str(tmp_path / "kb"), "knowledge_identity": "rev:3"},
        blocks=(
            {"kind": "protocol", "digest": "@protocol"},
            {"kind": "kb_index", "ref": "kb_index", "digest": digest},
            {"kind": "payload", "digest": "@context"},
        ),
    )
    monkeypatch.setattr(base, "kb_index_block_text", lambda root, rev: text)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode=mode,
        limits=AGY_CHANNEL if mode == "agent_document" else ChannelLimits(opening_max_chars=150_000),
    )
    reads = {entry["kind"]: entry for entry in claimed["manifest"]["required_blocks"]}
    index = reads["kb_index"]
    assert index["read"] == expected and index["tool"] == "kb_index"
    pushed = [block["kind"] for block in claimed["pushed"]]
    assert ("kb_index" in pushed) == (expected != "tool")
    assert reads["payload"]["read"] == ("file" if mode == "agent_document" else "opening")
    if expected == "opening":
        kinds = [kind for kind, _text in claimed["shared_blocks"]]
        assert kinds == (["kb_index"] if mode == "system_prompt" else ["protocol", "kb_index"])


def test_a_fork_opening_counts_the_protocol_its_fallback_would_send(tmp_path) -> None:
    """Without a parent the protocol rides as a message before the opening,
    so a fork's bound covers both: the payload needs that much more room
    than under `system_prompt`, whose protocol is not in the input at all."""

    from finesub.llm.agent.local_agent import agent_protocol_earlier_message
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(tmp_path, payload="p" * 5_000)

    def smallest_bound(mode: str) -> int:
        def carried(limit: int) -> bool:
            claimed = _claim_for_opening(runtime, "assignment-1", worker_id="worker-1", mode=mode,
                                         limits=ChannelLimits(opening_max_chars=limit))
            reads = {entry["kind"]: entry["read"] for entry in claimed["manifest"]["required_blocks"]}
            return reads["payload"] == "opening"

        low, high = 1_000, 60_000
        while low < high:
            middle = (low + high) // 2
            low, high = (low, middle) if carried(middle) else (middle + 1, high)
        return low

    protocol = len(agent_protocol_earlier_message("RULES: answer good"))
    assert smallest_bound("fork") - smallest_bound("system_prompt") >= protocol - 50


def test_a_block_past_the_opening_bound_is_left_to_read_context(tmp_path) -> None:
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(tmp_path, payload="x" * 200_000)
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="inline",
        limits=ChannelLimits(opening_max_chars=150_000),
    )
    reads = {entry["kind"]: entry for entry in claimed["manifest"]["required_blocks"]}
    assert reads["protocol"]["read"] == "opening" and "tool" not in reads["protocol"]
    assert reads["payload"]["read"] == "tool" and reads["payload"]["tool"] == "read_context"
    assert [block["kind"] for block in claimed["pushed"]] == ["protocol"]
    assert claimed["task_blocks"] == []


def test_a_lapsed_opening_lease_tells_the_session_to_take_the_task_again(tmp_path) -> None:
    """A CLI that started after the harness's hold ran out (it queued for a
    slot) is told to claim again, and the claim hands it every block."""

    clock = [1000.0]
    runtime = _runtime(tmp_path)
    runtime._clock = lambda: clock[0]
    keys = _harness_claims(runtime)
    clock[0] += 10_000
    server = HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s1",
        instance_id="instance-1",
        predelivered=keys,
    )
    early = _call(server, 1, "submit", payload="good")
    assert early["isError"] is True and "call next_task to take it again" in early["content"][0]["text"]
    task = _call(server, 2, "next_task")
    assert task["blocks"]["protocol"].strip() == "RULES: answer good"
    assert _call(server, 3, "submit", payload="good")["status"] == "accepted"


def _clocked_claim(tmp_path):
    from finesub.llm.client import _claim_for_opening

    clock = [1000.0]
    runtime = _single_task_runtime(tmp_path)
    runtime._clock = lambda: clock[0]
    claimed = _claim_for_opening(
        runtime, "assignment-1", worker_id="worker-1", mode="inline", limits=ChannelLimits()
    )
    return runtime, clock, claimed


def test_the_cli_start_renews_the_lease_the_queue_was_eating(tmp_path) -> None:
    from finesub.llm.client import _opening_observer

    runtime, clock, claimed = _clocked_claim(tmp_path)
    ttl = runtime.lease_ttl_seconds
    clock[0] += ttl - 1  # the driver queued for a slot almost the whole lease
    _opening_observer(
        runtime, claimed, assignment_id="assignment-1", worker_id="worker-1", attempt=0
    ).started({})
    clock[0] += ttl - 1  # past the claim's own deadline, inside the renewed one
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["lease_expires_at"] > clock[0]


def test_a_cli_start_after_the_lease_lapsed_leaves_it_to_the_server(tmp_path) -> None:
    from finesub.llm.client import _opening_observer

    runtime, clock, claimed = _clocked_claim(tmp_path)
    clock[0] += runtime.lease_ttl_seconds + 1
    _opening_observer(
        runtime, claimed, assignment_id="assignment-1", worker_id="worker-1", attempt=0
    ).started({})  # no error: the server says "hold lapsed"


@pytest.mark.parametrize("retaken", [False, True])
def test_a_retry_after_the_lease_moved_books_on_the_lease_it_finds(tmp_path, retaken) -> None:
    """The first CLI stopped early after the claim lapsed -- untouched, or
    taken again by its own `next_task` under a new generation. The retry's
    booking holds or re-claims the lease instead of failing on the old one."""

    from finesub.llm.client import _book_opening, _reset_agent_context

    runtime, clock, claimed = _clocked_claim(tmp_path)
    clock[0] += runtime.lease_ttl_seconds + 1
    if retaken:
        first = HarnessToolServer(
            runtime, assignment_id="assignment-1", worker_id="worker-1",
            session_id="s1", instance_id="i1",
        )
        _call(first, 1, "next_task")
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    _reset_agent_context(runtime, "assignment-1", record, session_id="s1", reason="premature stop")
    server_spec = {"env": {}}
    booked = _book_opening(
        runtime, claimed, assignment_id="assignment-1", worker_id="worker-1", attempt=1,
        mcp_server=server_spec, mode="inline", limits=ChannelLimits(),
        native_search=False,
    )
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["lease_owner"] == "worker-1"
    assert int(booked["task"]["lease_generation"]) == int(record["lease_generation"])
    assert int(record["lease_generation"]) > int(claimed["task"]["lease_generation"])
    second = HarnessToolServer(
        runtime, assignment_id="assignment-1", worker_id="worker-1", session_id="s2",
        instance_id="i2", predelivered=server_spec["env"]["FINESUB_MCP_PREDELIVERED"].split(","),
    )
    assert _call(second, 1, "pull_status")["owed_blocks"] == []
    assert _call(second, 2, "submit", payload="good")["status"] == "accepted"


def test_the_opening_refuses_a_block_it_cannot_place(tmp_path) -> None:
    from finesub.llm.client import _claim_for_opening

    runtime = _single_task_runtime(
        tmp_path,
        blocks=(
            {"kind": "protocol", "digest": "@protocol"},
            {"kind": "payload", "digest": "@context"},
            {"kind": "notes", "ref": "notes", "digest": "sha256:n"},
        ),
    )
    with pytest.raises(ValueError, match="'notes'"):
        _claim_for_opening(
            runtime, "assignment-1", worker_id="worker-1", mode="inline", limits=ChannelLimits()
        )


def test_each_driver_says_what_its_opening_carries() -> None:
    from finesub.llm.agent import local_agent

    def opening(config):
        return config.__dataclass_fields__["mcp_opening"].default

    assert opening(local_agent.CodexDriverConfig) == "inline"
    assert opening(local_agent.ClaudeCodeDriverConfig) == "system_prompt"
    assert opening(local_agent.CodeBuddyDriverConfig) == "inline"
    assert opening(local_agent.AgyDriverConfig) == "agent_document"
    assert opening(local_agent.DshDriverConfig) == ""


def test_a_replayed_claim_and_booking_of_one_attempt_change_nothing(tmp_path) -> None:
    from finesub.llm.client import _book_opening, _claim_for_opening

    runtime = _single_task_runtime(tmp_path)

    def claim():
        return _claim_for_opening(
            runtime, "assignment-1", worker_id="worker-1", mode="inline", limits=ChannelLimits()
        )

    def book(claimed):
        spec = {"env": {}}
        booked = _book_opening(
            runtime, claimed, assignment_id="assignment-1", worker_id="worker-1",
            attempt=0, mcp_server=spec, mode="inline", limits=ChannelLimits(), native_search=False,
        )
        return booked, spec["env"]["FINESUB_MCP_PREDELIVERED"]

    first, again = claim(), claim()
    assert first["task"] == again["task"]
    (booked, keys), (rebooked, rekeys) = book(first), book(again)
    assert booked["task"] == rebooked["task"] and keys == rekeys
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert int(record["lease_generation"]) == int(first["task"]["lease_generation"])
    server = HarnessToolServer(
        runtime, assignment_id="assignment-1", worker_id="worker-1",
        session_id="s1", instance_id="i1", predelivered=keys.split(","),
    )
    assert _call(server, 1, "pull_status")["owed_blocks"] == []


def test_agy_gets_the_protocol_as_its_system_prompt_and_the_payload_as_parts(
    tmp_path, monkeypatch, _opening_on
) -> None:
    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="agent_document", channel=AGY_CHANNEL)
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    messages, kwargs = driver.calls[0]
    opening = messages[0]["content"]
    spec = kwargs["mcp_server"]
    assert "RULES: answer good" in spec["agent_protocol"]
    assert '<block kind="protocol">' not in opening and '<block kind="payload">' not in opening
    assert "RULES: answer good" not in opening
    assert '"read": "system"' in opening and '"read": "file"' in opening
    assert "parallel file-viewing calls" in opening and "control" in opening.rstrip().rsplit("\n", 1)[-1]
    keys = spec["env"]["FINESUB_MCP_PREDELIVERED"].split(",")
    assert sorted(key.split("@")[0] for key in keys) == ["payload", "protocol"]


def test_a_forking_driver_gets_the_protocol_for_its_parent_and_the_payload_inline(
    tmp_path, monkeypatch, _opening_on
) -> None:
    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="fork")
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    messages, kwargs = driver.calls[0]
    opening = messages[0]["content"]
    assert "RULES: answer good" in kwargs["mcp_server"]["agent_protocol"]
    assert '<block kind="protocol">' not in opening and '<block kind="payload">' in opening
    assert '"read": "earlier"' in opening


def test_a_system_prompt_driver_gets_the_protocol_apart_and_the_payload_inline(
    tmp_path, monkeypatch, _opening_on
) -> None:
    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="system_prompt")
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    messages, kwargs = driver.calls[0]
    opening = messages[0]["content"]
    assert "RULES: answer good" in kwargs["mcp_server"]["agent_protocol"]
    assert "RULES: answer good" not in opening
    assert '<block kind="protocol">' not in opening and '<block kind="payload">' in opening
    assert '"read": "system"' in opening


def test_after_a_premature_stop_the_new_cli_finds_the_protocol_seen(
    tmp_path, monkeypatch, _opening_on
) -> None:
    _no_api(monkeypatch)
    reads: list = []

    def agent(server):
        if not reads:
            reads.append("left")
            return
        task = _call(server, 1, "next_task")
        reads.append({block["kind"]: block.get("read") for block in task["required_blocks"]})
        _call(server, 2, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent, mcp_opening="inline")
    result = _client(driver, tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    assert reads[1]["protocol"] == "seen"
