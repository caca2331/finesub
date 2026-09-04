"""Tool-session extras: premature-stop retry, the audit bundle, proxied web
tools and the Codex per-invocation declaration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent import agent_validators
from finesub.llm.agent.agent_mcp_server import TOOL_NAMES, WEB_TOOL_NAMES, HarnessToolServer
from finesub.llm.agent.agent_paths import evidence_locator, resolve_agent_episode_location
from finesub.llm.agent.agent_task_runtime import AgentTaskRuntime, AgentTaskSpec, ValidationResult
from finesub.llm.agent.local_agent import (
    CodexDriverConfig,
    CodexLocalAgentDriver,
    DriverProbe,
    _codex_mcp_server_override,
    _normalize_events,
)
from finesub.llm.client import AGENT_PREMATURE_STOP_RETRIES, RoleClient
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import load_model_routes

from .test_llm_agent_tool_session import _ToolUsingFakeDriver, _call


@pytest.fixture(autouse=True)
def _register_reject_bad(monkeypatch):
    def build(_manifest):
        def validate(candidate, _m):
            if candidate == "bad":
                return ValidationResult.repairable("say good, not bad")
            return ValidationResult.accepted(candidate)

        return validate

    monkeypatch.setitem(agent_validators.VALIDATOR_BUILDERS, "test-reject-bad", build)


def _client(driver, tmp_path, *, target="local-claude-completion-sonnet-5"):
    routes = load_model_routes(
        user_config={"model_groups": {"research-default": {"targets": [target]}}}
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
    {"role": "user", "content": "fix this window"},
]


def test_a_premature_stop_gets_one_fresh_session_on_the_same_worker(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    sessions: list[str] = []

    def agent(server):
        sessions.append(server.session_id)
        task = _call(server, 1, "next_task")
        if len(sessions) == 1:
            return  # left without submitting
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent)
    client = _client(driver, tmp_path)

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )

    assert result.content == "good"
    assert len(driver.calls) == 2 == AGENT_PREMATURE_STOP_RETRIES + 1
    assert len(set(sessions)) == 2
    # Both workers were "worker-1": the second claim renewed the same lease.
    assert all(s.worker_id == "worker-1" for s in driver.servers)
    assert result.execution_attempts[0]["premature_stop"] is True


def test_two_premature_stops_fail_the_call(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    driver = _ToolUsingFakeDriver(lambda server: _call(server, 1, "next_task"))
    client = _client(driver, tmp_path)

    with pytest.raises(Exception) as excinfo:
        client.complete(LLMRole.GENERAL_CAPABLE, MESSAGES)
    assert "without an accepted submit" in str(excinfo.value)
    assert len(driver.calls) == AGENT_PREMATURE_STOP_RETRIES + 1


def test_the_audit_bundle_lands_in_the_capsule(tmp_path, monkeypatch) -> None:
    """What the agent could read, what it submitted and every frame, in one place."""

    _no_api(monkeypatch)
    location = resolve_agent_episode_location(tmp_path / "episodes")
    episode = location.parent / "episode-1"
    episode.mkdir(parents=True)
    locator = evidence_locator(location, "episode-1").as_dict()

    def agent(server):
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="bad")
        _call(server, 21, "submit", payload="good")

    class Driver(_ToolUsingFakeDriver):
        def run(self, messages, **kwargs):
            result = super().run(messages, **kwargs)
            result.execution_attempt["evidence_locator"] = locator
            # The server logs frames only when spawned by a CLI; stand in.
            Path(kwargs["mcp_server"]["env"]["FINESUB_MCP_LOG"]).write_text(
                '{"dir": "in", "frame": "{}"}\n', encoding="utf-8"
            )
            return result

    driver = Driver(agent)
    client = _client(driver, tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=3,
    )

    assert result.content == "good"
    audit = episode / "audit"
    manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validator_id"] == "test-reject-bad"
    assert (audit / "blocks" / "protocol.md").read_text(encoding="utf-8").strip() == "RULES: answer good"
    assert (audit / "blocks" / "payload.md").read_text(encoding="utf-8").strip() == "fix this window"
    outcome = json.loads((audit / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["accepted"] is True
    assert outcome["task"]["last_candidate"] == "bad"
    assert outcome["ledger"]["owed_blocks"] == []
    assert (audit / "artifact.txt").read_text(encoding="utf-8") == "good"
    assert (audit / "mcp-frames.jsonl").exists()
    # The assignment root itself is gone; the bundle is the record.
    assert not any((tmp_path / "assignments").iterdir())


def test_web_tools_are_offered_only_to_a_local_retrieval_task(tmp_path) -> None:
    def runtime(mode):
        return AgentTaskRuntime.start_assignment(
            tmp_path / mode,
            assignment_id="assignment-1",
            worker_goal="answer",
            tasks=[
                AgentTaskSpec(
                    task_id="call",
                    session_type="correction",
                    input_hash="sha256:in",
                    goal="answer",
                    retrieval_mode=mode,
                )
            ],
        )

    plain = HarnessToolServer(
        runtime("none"),
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s",
        instance_id="plain",
    )
    local = HarnessToolServer(
        runtime("local"),
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id="s",
        instance_id="local",
    )
    assert [tool["name"] for tool in plain.tool_definitions] == list(TOOL_NAMES)
    assert [tool["name"] for tool in local.tool_definitions] == [*TOOL_NAMES, *WEB_TOOL_NAMES]
    web = {tool["name"]: tool["annotations"] for tool in local.tool_definitions}["web_search"]
    assert web["openWorldHint"] is True and web["destructiveHint"] is False
    # Unknown to the plain server, and a refusal rather than a crash.
    refused = _call(plain, 1, "web_search", query="x")
    assert refused["isError"] is True


def test_a_tool_session_honours_each_retrieval_state(tmp_path, monkeypatch) -> None:
    """`local` retrieves through the harness; `native` uses the CLI's own tool.

    The switch is three states (routing/profiles.py) and used to reach here as
    a boolean, so every truthy value became `local`: `retrieval=native` ran on
    the harness proxy on every agent call, which is the one thing that switch
    says it never does.
    """

    _no_api(monkeypatch)

    def run(retrieval: str):
        offered: list = []

        def agent(server):
            offered.extend(tool["name"] for tool in server.tool_definitions)
            task = _call(server, 1, "next_task")
            for index, block in enumerate(task["required_blocks"]):
                _call(server, 10 + index, "read_context", ref=block["ref"])
            _call(server, 20, "submit", payload="good")

        driver = _ToolUsingFakeDriver(agent)
        client = _client(driver, tmp_path / retrieval, target="local-claude-native-sonnet-5")
        result = client.complete(
            LLMRole.GENERAL_CAPABLE,
            MESSAGES,
            retrieval=retrieval,
            validator_spec={"id": "test-reject-bad", "params": {}},
        )
        assert result.content == "good"
        return offered, driver.calls[0][1]

    offered, kwargs = run("local")
    # Harness-executed: the proxy is offered and the CLI's own search stays off.
    assert "web_search" in offered and "web_fetch" in offered
    assert kwargs["native_search"] is False
    assert kwargs["mcp_server"]["tools"] == [*TOOL_NAMES, *WEB_TOOL_NAMES]

    offered, kwargs = run("native")
    # Provider-executed: the driver entitles its own search, and the harness
    # offers no web tool to compete with it.
    assert kwargs["native_search"] is True
    assert "web_search" not in offered and "web_fetch" not in offered
    assert kwargs["mcp_server"]["tools"] == list(TOOL_NAMES)


def test_codex_declares_the_server_as_an_inline_config_override(tmp_path) -> None:
    override = _codex_mcp_server_override(
        {
            "command": "C:/py/python.exe",
            "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
            "env": {"FINESUB_MCP_SESSION": "s1", "bad key": "x"},
            "tools": ["next_task", "submit"],
        }
    )
    assert override.startswith("mcp_servers.finesub = {")
    assert 'command = "C:/py/python.exe"' in override
    assert 'env = {FINESUB_MCP_SESSION = "s1"}' in override
    assert 'enabled_tools = ["next_task", "submit"]' in override
    assert 'default_tools_approval_mode = "auto"' in override

    driver = CodexLocalAgentDriver(CodexDriverConfig(command=("codex",)))
    driver._resolved_command = ("codex",)
    probe = DriverProbe(available=True, no_user_config=True, no_user_rules=True, supports_mcp_config=True)
    capsule = SimpleNamespace(root=tmp_path, episode_id="ep")
    argv = driver._argv(capsule, native_search=False, probe=probe, mcp_server={
        "command": "python", "args": [], "env": {}, "tools": ["submit"]
    })
    assert argv[argv.index("--config") + 1].startswith("mcp_servers.finesub = {")
    # The tool-result limit rides the same per-invocation channel.
    assert "tool_output_token_limit=200000" in argv
    assert argv.index("tool_output_token_limit=200000") == argv.index("--config", argv.index("--config") + 1) + 1
    assert "--ignore-user-config" in argv


def test_codex_events_treat_entitled_mcp_calls_as_tool_use(tmp_path) -> None:
    raw = tmp_path / "events.jsonl"
    rows = [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "finesub", "tool": "submit", "status": "completed"}},
        {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "other", "tool": "submit", "status": "completed"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "submitted"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    normalized, _usage, violations, content = _normalize_events(
        raw, native_search=False, max_bytes=1_000_000, extra_entitled=frozenset({"submit"})
    )
    assert content == "submitted"
    assert [row.get("tool") for row in normalized if row.get("event") == "tool_use"] == ["submit", "submit"]
    assert violations == ["target invoked forbidden tool other/submit"]

    _n, _u, plain_violations, _c = _normalize_events(raw, native_search=False, max_bytes=1_000_000)
    assert "target invoked forbidden tool mcp_tool_call" in plain_violations


def test_an_accepted_submit_survives_a_driver_error_after_it(tmp_path, monkeypatch) -> None:
    """agy flips its result to ERROR when the hook denies a stray tool *after*
    the submit; the runtime already holds the artifact, and that is what counts."""

    from finesub.llm.agent.local_agent import LocalAgentPolicyViolationError

    _no_api(monkeypatch)

    def agent(server):
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="good")

    class Driver(_ToolUsingFakeDriver):
        def run(self, messages, **kwargs):
            super().run(messages, **kwargs)
            error = LocalAgentPolicyViolationError("tool call denied by pre-tool hook")
            setattr(error, "_harness_execution_attempts", [{"backend": "local_agent", "capsule_id": "cap-x"}])
            raise error

    driver = Driver(agent)
    client = _client(driver, tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )

    assert result.content == "good"
    assert len(driver.calls) == 1
    assert result.execution_attempts[-1]["capsule_id"] == "cap-x"


def test_a_driver_error_before_any_submit_still_fails_the_call(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.local_agent import LocalAgentTransientError

    _no_api(monkeypatch)

    class Driver(_ToolUsingFakeDriver):
        def run(self, messages, **kwargs):
            super().run(messages, **kwargs)
            raise LocalAgentTransientError("Eligibility check failed: 500")

    driver = Driver(lambda server: _call(server, 1, "next_task"))
    client = _client(driver, tmp_path)
    with pytest.raises(Exception) as excinfo:
        client.complete(LLMRole.GENERAL_CAPABLE, MESSAGES)
    assert "Eligibility" in str(excinfo.value) or "500" in str(excinfo.value)


def test_a_premature_stop_retires_the_task_so_the_next_session_starts_clean(
    tmp_path, monkeypatch
) -> None:
    """The dead CLI's lease and ledger go with it: the fresh session claims a
    new generation and has to be handed its blocks again (docs §0-3)."""

    _no_api(monkeypatch)
    seen: list[dict] = []

    def agent(server):
        task = _call(server, 1, "next_task")
        seen.append({"lease": server._task["lease_generation"], "blocks": sorted(task["blocks"])})
        if len(seen) == 1:
            return
        _call(server, 20, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent)
    client = _client(driver, tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )

    assert result.content == "good"
    # Same lease, same generation (a transport fault is not a replacement);
    # the context was reset, so the blocks were handed over again.
    assert seen[0]["lease"] == 1 and seen[1]["lease"] == 1
    assert seen[1]["blocks"] == ["payload", "protocol"]
    # Accepted in the end, with no capsule left to audit into: nothing is
    # kept (docs §0-4 -- a root is only held back when a bundle that had
    # somewhere to go could not be written).
    assert list((tmp_path / "assignments").iterdir()) == []


def test_the_driver_is_handed_a_completion_predicate_that_turns_true_on_accept(
    tmp_path, monkeypatch
) -> None:
    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "next_task")
        _call(server, 20, "submit", payload="good")

    driver = _ToolUsingFakeDriver(agent)
    driver.hang_after_script = True
    client = _client(driver, tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}}
    )
    assert result.content == "good"
    assert callable(driver.calls[0][1]["completion"])


def test_the_audit_bundle_records_the_state_after_retirement_and_on_errors(
    tmp_path, monkeypatch
) -> None:
    from finesub.llm.agent.agent_paths import evidence_locator, resolve_agent_episode_location
    from finesub.llm.agent.local_agent import LocalAgentTransientError

    _no_api(monkeypatch)
    location = resolve_agent_episode_location(tmp_path / "episodes")
    episode = location.parent / "episode-1"
    episode.mkdir(parents=True)
    locator = evidence_locator(location, "episode-1").as_dict()

    class Driver(_ToolUsingFakeDriver):
        fail = False

        def run(self, messages, **kwargs):
            result = super().run(messages, **kwargs)
            result.execution_attempt["evidence_locator"] = locator
            if self.fail:
                error = LocalAgentTransientError("vendor 500")
                setattr(error, "_harness_execution_attempts", [result.execution_attempt])
                raise error
            return result

    # Exhausted chain: retired, and the bundle says so.
    driver = Driver(lambda server: (_call(server, 1, "next_task"), _call(server, 20, "submit", payload="bad")))
    client = _client(driver, tmp_path)
    client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=1,
    )
    outcome = json.loads((episode / "audit" / "outcome.json").read_text(encoding="utf-8"))
    roots = list((tmp_path / "assignments").iterdir())
    runtime = AgentTaskRuntime(roots[0], validators=agent_validators.runtime_validators("test-reject-bad"))
    final = runtime.task_record(assignment_id=roots[0].name, task_id="call")
    assert outcome["task"] == final
    assert final["status"] == "queued" and final["retirements"] == 1
    assert outcome["conversation"]["resets"][-1]["reason"] == "retired: session ended"

    # Driver error: the bundle is still written, with the error and the
    # retired state.
    for root in roots:
        import shutil
        shutil.rmtree(root)
    shutil.rmtree(episode / "audit")
    driver = Driver(lambda server: _call(server, 1, "next_task"))
    driver.fail = True
    client = _client(driver, tmp_path)
    with pytest.raises(Exception, match="vendor 500"):
        client.complete(LLMRole.GENERAL_CAPABLE, MESSAGES)
    outcome = json.loads((episode / "audit" / "outcome.json").read_text(encoding="utf-8"))
    assert "vendor 500" in outcome["error"]
    assert outcome["task"]["status"] == "queued" and outcome["task"]["retirements"] == 1


def test_a_driver_error_after_next_task_retires_the_lease(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.local_agent import LocalAgentTransientError

    _no_api(monkeypatch)

    class Driver(_ToolUsingFakeDriver):
        def run(self, messages, **kwargs):
            super().run(messages, **kwargs)
            raise LocalAgentTransientError("vendor 500")

    driver = Driver(lambda server: _call(server, 1, "next_task"))
    client = _client(driver, tmp_path)
    with pytest.raises(Exception, match="vendor 500"):
        client.complete(LLMRole.GENERAL_CAPABLE, MESSAGES)
    roots = list((tmp_path / "assignments").iterdir())
    runtime = AgentTaskRuntime(roots[0])
    record = runtime.task_record(assignment_id=roots[0].name, task_id="call")
    assert record["status"] == "queued" and record["lease_owner"] == ""
    assert record["retirements"] == 1


def test_a_replayed_rpc_id_with_different_arguments_is_refused(tmp_path) -> None:
    from .test_llm_agent_tool_session import _runtime, _server

    runtime = _runtime(tmp_path)
    server = _server(runtime)
    _call(server, 1, "next_task")
    first = _call(server, 10, "submit", payload="bad")
    assert first["status"] == "repairable"
    refused = _call(server, 10, "submit", payload="good")
    assert refused["isError"] is True and "already used" in refused["content"][0]["text"]
    assert runtime.task_record(assignment_id="assignment-1", task_id="call")["status"] == "repairing"


def test_an_exhausted_session_leaves_the_task_retired_not_leased(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)

    def agent(server):
        _call(server, 1, "next_task")
        _call(server, 20, "submit", payload="bad")

    driver = _ToolUsingFakeDriver(agent)
    client = _client(driver, tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=1,
    )

    assert result.repair_exhausted is True
    roots = list((tmp_path / "assignments").iterdir())
    runtime = AgentTaskRuntime(roots[0], validators=agent_validators.runtime_validators("test-reject-bad"))
    record = runtime.task_record(assignment_id=roots[0].name, task_id="call")
    assert record["status"] == "queued" and record["lease_owner"] == ""
    assert record["retirements"] == 1


def test_a_media_call_runs_as_a_capsule_session_on_the_same_driver(tmp_path, monkeypatch) -> None:
    """The tool protocol is text-only, so a media call takes the capsule
    transport instead of failing the candidate (docs/llm_local_agent.md
    §12.1, second revision 2026-08-22)."""

    from finesub.llm.media_upload import UploadedFileRef

    _no_api(monkeypatch)
    calls: list = []

    class CapsuleDriver(_ToolUsingFakeDriver):
        def run(self, messages, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content="sub|1|ok",
                reported_model="gemini-3.7-flash",
                execution_attempt={"backend": "local_agent", "capsule_id": "cap-m"},
                episode_id="cap-m",
                conversation_handle="",
                turn_identity="",
                normalized_events=(),
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

    driver = CapsuleDriver(lambda server: None)
    client = _client(driver, tmp_path, target="local-agy-media-gemini-3_8-flash")
    clip = tmp_path / "clip.ogg"
    clip.write_bytes(b"ogg")
    ref = UploadedFileRef(
        file_id="", filename="clip.ogg", mime_type="audio/ogg",
        local_path=str(clip), duration_seconds=1.0,
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}], file_ref=ref
    )
    assert result.content == "sub|1|ok"
    assert result.route_decision["candidates"][0]["agent_transport"] == "capsule"
    assert calls and "mcp_server" not in calls[0]


def test_the_audit_bundle_survives_a_tool_served_block(tmp_path) -> None:
    """A required block whose ref names a tool, not an artifact.

    `kb_index` is fetched through its own tool and carries that tool's name as
    its ref; its identity is the declared digest. Reading it as a stored
    artifact raised, and the bundle is written all-or-nothing -- so every
    knowledge-bound call silently lost its audit and kept its runtime root.
    """

    from finesub.llm.agent.agent_session_host import write_audit_bundle

    location = resolve_agent_episode_location(tmp_path / "episodes")
    episode = location.parent / "episode-1"
    episode.mkdir(parents=True)
    root = tmp_path / "assignments" / "call-1"
    runtime = AgentTaskRuntime.start_assignment(
        root,
        assignment_id="call-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id="call",
                session_type="correction",
                input_hash="sha256:in",
                goal="answer",
                required_blocks=(
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "kb_index", "ref": "kb_index", "digest": "a" * 64},
                ),
            )
        ],
        protocol_documents={"correction": "RULES: answer good"},
    )
    result = SimpleNamespace(
        execution_attempt={
            "evidence_locator": evidence_locator(location, "episode-1").as_dict()
        }
    )

    written = write_audit_bundle(
        runtime,
        result=result,
        root=root,
        assignment_id="call-1",
        worker_id="worker-1",
        record=runtime.task_record(assignment_id="call-1", task_id="call"),
        accepted_text="good",
    )

    assert written is True
    blocks = episode / "audit" / "blocks"
    assert (blocks / "protocol.md").read_text(encoding="utf-8").strip() == "RULES: answer good"
    # No body to copy, so the declaration is what the bundle keeps. (`tool` is
    # the runtime's default; the server rewrites it to `kb_index` in the reply
    # the agent actually sees.)
    declared = json.loads((blocks / "kb_index.json").read_text(encoding="utf-8"))
    assert declared["kind"] == "kb_index"
    assert declared["ref"] == "kb_index"
    assert declared["digest"] == "a" * 64


def test_a_broken_retrieval_ledger_does_not_fail_an_accepted_call(
    tmp_path, monkeypatch
) -> None:
    """Provenance is evidence, not the answer (the audit bundle's rule).

    Folding the proxied searches in must not turn a call the runtime already
    accepted into a failure because its ledger could not be read.
    """

    from finesub.llm.agent.agent_task_runtime import AgentTaskRuntimeError

    _no_api(monkeypatch)

    def boom(self, **_kwargs):
        raise AgentTaskRuntimeError("ledger unreadable")

    monkeypatch.setattr(AgentTaskRuntime, "retrieval_search_events", boom)

    def agent(server):
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="good")

    client = _client(_ToolUsingFakeDriver(agent), tmp_path)
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        MESSAGES,
        retrieval="local",
        validator_spec={"id": "test-reject-bad", "params": {}},
    )

    assert result.content == "good"
