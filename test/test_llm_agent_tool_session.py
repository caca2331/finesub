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
    AgentCapsule,
    ClaudeCodeDriverConfig,
    ClaudeCodeLocalAgentDriver,
    CodexDriverConfig,
    CodexLocalAgentDriver,
    DriverProbe,
    mcp_tool_name,
)
from finesub.llm.client import RoleClient
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import load_model_routes


@pytest.fixture(autouse=True)
def _register_reject_bad(monkeypatch):
    def build(_manifest):
        def validate(candidate, _m):
            if candidate == "bad":
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
    assert rejected["repair_rounds_remaining"] == 4
    # A transport replay of the same JSON-RPC id is the same answer; the
    # repair budget does not move twice.
    assert _call(server, 8, "submit", payload="bad") == rejected
    accepted = _call(server, 9, "submit", payload="good")
    assert accepted["status"] == "accepted"
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "accepted"
    assert record["last_candidate"] == "bad"


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
    assert first["status"] == "repair_exhausted"
    again = _call(server, 21, "submit", payload="good")
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
    capsule = SimpleNamespace(root=tmp_path, episode_id="ep")
    server = {
        "command": "python",
        "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
        "env": {"FINESUB_MCP_SESSION": "s1"},
        "tools": list(TOOL_NAMES),
    }

    plain = driver._argv(capsule, native_search=False, probe=probe)
    tooled = driver._argv(capsule, native_search=False, probe=probe, mcp_server=server)

    assert "--safe-mode" in plain and "--safe-mode" not in tooled
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
    capsule = SimpleNamespace(root=tmp_path, episode_id="ep")
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


class _ToolUsingFakeDriver:
    """Plays the agent: drives the harness server in-process as the CLI would."""

    conversation_ttl_seconds = 0.0

    def __init__(self, script) -> None:
        self.config = SimpleNamespace(model="")
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
        )
        self.servers.append(server)
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
                "research-default": {"targets": ["local-claude-completion-sonnet-5"]}
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


def test_a_restarted_server_process_reads_the_repair_budget_from_the_task_row(
    tmp_path,
) -> None:
    """docs §7: the budget is durable, so an MCP restart cannot refill it."""

    runtime = _runtime(tmp_path, max_repairs=2)
    first = _server(runtime)
    task = _call(first, 1, "next_task")
    assert task["repair_rounds_remaining"] == 2
    assert _call(first, 2, "submit", payload="bad")["repair_rounds_remaining"] == 1

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
    replay = _call(restarted, 2, "submit", payload="bad")
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
    # Budget 1 is spent by the first distinct rejection; replays are told so.
    assert statuses[0] == "repair_exhausted"
    assert set(statuses[1:]) == {"repair_exhausted"}
    assert runtime.task_record(assignment_id="assignment-1", task_id="call")["status"] == "repairing"


def test_claude_raises_its_mcp_reply_limit_only_for_a_tool_session(tmp_path) -> None:
    """`MAX_MCP_OUTPUT_TOKENS` is set in the CLI's own process environment for
    this invocation (the user's settings are never written)."""

    driver = ClaudeCodeLocalAgentDriver(ClaudeCodeDriverConfig(runtime_root=tmp_path / "runtime"))
    env = driver._spawn_environment(mcp_server={"command": "x"})
    assert env["MAX_MCP_OUTPUT_TOKENS"] == "200000"
    assert env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] == "200000"
    assert "MAX_MCP_OUTPUT_TOKENS" not in driver._spawn_environment(mcp_server=None)
