"""Pseudo-conversational sessions (docs/llm_local_agent.md §12.1.3).

One CLI invocation serves a run's tasks in turn: the runtime grows an
unsealed assignment, the MCP server parks `next_task` between tasks, the
host supervises the CLI on a background thread and each `complete()` waits
for its own task only. Covers the runtime additions, the server's long poll,
the driver's parked/timeout hooks and the host end to end with a fake CLI
that plays the agent in-process.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from finesub.llm.agent import agent_validators
from finesub.llm.agent.agent_mcp_server import TOOL_NAMES, WEB_TOOL_NAMES, HarnessToolServer
from finesub.llm.agent.agent_paths import (
    evidence_locator,
    resolve_agent_episode_location,
)
from finesub.llm.agent.agent_session_host import (
    AgentSessionHost,
    agent_session_scope,
    current_registry,
)
from finesub.llm.agent import agent_session_host as host_module
from finesub.llm.agent import agent_task_runtime
from finesub.llm.agent.agent_task_runtime import (
    MAX_REQUEST_RESULTS,
    AgentTaskRuntime,
    AgentTaskSpec,
    AssignmentConflictError,
    ValidationResult,
)
from finesub.llm.agent.agent_validators import VALIDATOR_BUILDERS, runtime_validators
from finesub.llm.agent.local_agent import DriverProbe, LocalAgentTimeoutError
from finesub.llm.client import RoleClient, write_agent_session_usage
from finesub.llm.exchange_metadata import AGENT_SESSION_USAGE_FILENAME
from finesub.llm.task_report import _add_agent_session_usage, render_task_report
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


def _all_validators():
    validators = {}
    for validator_id in sorted(VALIDATOR_BUILDERS):
        validators.update(runtime_validators(validator_id))
    return validators


def _unsealed(tmp_path):
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="serve",
        tasks=[],
        session_scope="assignment",
        validators=_all_validators(),
        sealed=False,
    )


def _spec(task_id, *, validator="accept", payload_key=None):
    return AgentTaskSpec(
        task_id=task_id,
        session_type="correction",
        input_hash=f"sha256:{task_id}",
        goal="answer",
        validator_id=validator,
        protocol_key="correction",
        context_key=payload_key or f"payload-{task_id}",
        required_blocks=(
            {"kind": "protocol", "digest": "@protocol"},
            {"kind": "payload", "digest": "@context"},
        ),
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


def _server(runtime, *, session="s1", instance="i1", wait=0.3):
    return HarnessToolServer(
        runtime,
        assignment_id="assignment-1",
        worker_id="worker-1",
        session_id=session,
        instance_id=instance,
        tools=[*TOOL_NAMES, *WEB_TOOL_NAMES],
        next_task_wait_seconds=wait,
    )


# --- runtime: a growing assignment -------------------------------------------


def test_tasks_are_added_one_at_a_time_and_the_seal_ends_the_assignment(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    assert runtime.status(assignment_id="assignment-1", worker_id="worker-1")["status"] == "waiting"

    runtime.add_task(
        _spec("call-0001"),
        protocol_documents={"correction": "RULES"},
        context_documents={"payload-call-0001": "window one"},
    )
    runtime.add_task(
        _spec("call-0002"),
        protocol_documents={"correction": "RULES"},
        context_documents={"payload-call-0002": "window two"},
    )
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    assert sorted(index["task_manifest_refs"]) == ["call-0001", "call-0002"]
    first = json.loads(runtime.read_artifact(index["task_manifest_refs"]["call-0001"]))
    second = json.loads(runtime.read_artifact(index["task_manifest_refs"]["call-0002"]))
    # One protocol body shared by digest; two payloads, each its own ref.
    assert first["protocol_ref"] == second["protocol_ref"]
    assert first["context_ref"] != second["context_ref"]
    assert runtime.read_artifact(second["context_ref"]) == "window two\n"

    with pytest.raises(AssignmentConflictError, match="already exists"):
        runtime.add_task(_spec("call-0001"), context_documents={"payload-call-0001": "x"})

    runtime.seal()
    with pytest.raises(AssignmentConflictError, match="sealed"):
        runtime.add_task(_spec("call-0003"), context_documents={"payload-call-0003": "x"})
    assert runtime.seal()["status"] in {"ready", "waiting", "task"}


def test_concurrent_adds_keep_their_own_payloads(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    errors: list[BaseException] = []

    def add(index: int) -> None:
        try:
            runtime.add_task(
                _spec(f"call-{index:04d}"),
                protocol_documents={"correction": "RULES"},
                context_documents={f"payload-call-{index:04d}": f"window {index}"},
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(index,)) for index in range(1, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    for task_id, ref in index["task_manifest_refs"].items():
        manifest = json.loads(runtime.read_artifact(ref))
        number = int(task_id.rsplit("-", 1)[-1])
        assert runtime.read_artifact(manifest["context_ref"]) == f"window {number}\n"


def test_a_withdrawn_task_is_terminal_but_does_not_fail_the_assignment(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    runtime.add_task(_spec("call-0001"), protocol_documents={"correction": "R"},
                     context_documents={"payload-call-0001": "w"})
    status = runtime.rehydrate(assignment_id="assignment-1", worker_id="worker-1")
    task = runtime.next_task(
        assignment_id="assignment-1", worker_id="worker-1", request_id="claim-1",
        expected_control_generation=status["control_generation"],
    )["task"]
    runtime.withdraw_task(assignment_id="assignment-1", task_id="call-0001", reason="budget spent")
    record = runtime.task_record(assignment_id="assignment-1", task_id="call-0001")
    assert record["status"] == "withdrawn" and record["lease_owner"] == ""
    # A late submit on the revoked lease lands nowhere.
    from finesub.llm.agent.agent_task_runtime import StaleLeaseError

    with pytest.raises(StaleLeaseError):
        runtime.submit(
            assignment_id="assignment-1", task_id="call-0001", worker_id="worker-1",
            lease_generation=task["lease_generation"], request_id="late",
            input_hash="sha256:call-0001", candidate="anything",
        )
    # Not a failure: the next task is served as usual.
    runtime.add_task(_spec("call-0002"), protocol_documents={"correction": "R"},
                     context_documents={"payload-call-0002": "w2"})
    assert runtime.status(assignment_id="assignment-1", worker_id="worker-1")["status"] == "ready"
    runtime.seal()
    assert runtime.status(assignment_id="assignment-1")["status"] == "ready"


def test_parked_claims_are_not_recorded_and_a_finished_task_prunes_its_rows(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    # More than the table can hold: parked claims must not be durable rows.
    for index in range(MAX_REQUEST_RESULTS + 5):
        status = runtime.rehydrate(assignment_id="assignment-1", worker_id="worker-1")
        if status.get("wait_token"):
            runtime.await_next_task(
                assignment_id="assignment-1", worker_id="worker-1",
                wait_token=status["wait_token"], max_wait_seconds=0.01,
            )
            continue
        runtime.next_task(
            assignment_id="assignment-1", worker_id="worker-1", request_id=f"wait-{index}",
            expected_control_generation=status["control_generation"],
        )
    runtime.add_task(_spec("call-0001"), protocol_documents={"correction": "R"},
                     context_documents={"payload-call-0001": "w"})
    status = runtime.rehydrate(assignment_id="assignment-1", worker_id="worker-1")
    task = runtime.next_task(
        assignment_id="assignment-1", worker_id="worker-1", request_id="claim",
        expected_control_generation=status["control_generation"],
    )["task"]
    runtime.record_pull(
        assignment_id="assignment-1", task_id="call-0001", worker_id="worker-1",
        lease_generation=task["lease_generation"], request_id="pull-1",
        blocks=json.loads(runtime.read_artifact(task["manifest_ref"]))["required_blocks"],
    )
    accepted = runtime.submit(
        assignment_id="assignment-1", task_id="call-0001", worker_id="worker-1",
        lease_generation=task["lease_generation"], request_id="submit-1",
        input_hash="sha256:call-0001", candidate="good",
    )
    assert accepted["accepted_task_id"] == "call-0001"
    state_ref = json.loads(runtime.index_path.read_text(encoding="utf-8"))["state_ref"]
    table = json.loads(runtime.read_artifact(state_ref))["request_results"]
    # A finished task leaves nothing behind: otherwise a run-long session
    # would fill the bounded table with one permanent row per window.
    assert table == {}
    # The one answer it still owes -- the accept -- comes off the task row.
    replayed = runtime.submit(
        assignment_id="assignment-1", task_id="call-0001", worker_id="worker-1",
        lease_generation=task["lease_generation"], request_id="submit-1",
        input_hash="sha256:call-0001", candidate="good",
    )
    assert replayed["accepted_task_id"] == "call-0001"
    with pytest.raises(AssignmentConflictError):
        runtime.submit(
            assignment_id="assignment-1", task_id="call-0001", worker_id="worker-1",
            lease_generation=task["lease_generation"], request_id="submit-1",
            input_hash="sha256:call-0001", candidate="something else",
        )


def test_more_accepted_tasks_than_the_table_holds(tmp_path, monkeypatch) -> None:
    """One session, more accepted tasks than the request table holds (docs §7).

    Every accepted task used to keep its submit row for ever, so a long run
    filled `MAX_REQUEST_RESULTS` and the next claim raised mid-window. The
    cap is patched down rather than the task count up: 500-odd real tasks
    would take minutes of state rewriting to prove the same invariant.
    """

    monkeypatch.setattr(agent_task_runtime, "MAX_REQUEST_RESULTS", 20)
    runtime = _unsealed(tmp_path)
    for index in range(30):
        task_id = f"call-{index:04d}"
        runtime.add_task(
            _spec(task_id),
            protocol_documents={"correction": "R"},
            context_documents={f"payload-{task_id}": f"window {index}"},
        )
        status = runtime.rehydrate(assignment_id="assignment-1", worker_id="worker-1")
        task = runtime.next_task(
            assignment_id="assignment-1", worker_id="worker-1",
            request_id=f"claim-{index}",
            expected_control_generation=status["control_generation"],
        )["task"]
        runtime.record_pull(
            assignment_id="assignment-1", task_id=task_id, worker_id="worker-1",
            lease_generation=task["lease_generation"], request_id=f"pull-{index}",
            blocks=json.loads(runtime.read_artifact(task["manifest_ref"]))["required_blocks"],
        )
        accepted = runtime.submit(
            assignment_id="assignment-1", task_id=task_id, worker_id="worker-1",
            lease_generation=task["lease_generation"], request_id=f"submit-{index}",
            input_hash=f"sha256:{task_id}", candidate="good",
        )
        assert accepted["accepted_task_id"] == task_id
    state_ref = json.loads(runtime.index_path.read_text(encoding="utf-8"))["state_ref"]
    assert json.loads(runtime.read_artifact(state_ref))["request_results"] == {}


# --- server: the long poll and the session-shaped replies --------------------


def test_next_task_parks_then_serves_then_tells_the_session_to_leave(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    server = _server(runtime, wait=0.3)

    started = time.monotonic()
    parked = _call(server, 1, "next_task")
    assert parked["status"] == "still_waiting"
    assert time.monotonic() - started >= 0.25

    runtime.add_task(
        _spec("call-0001", validator="test-reject-bad"),
        protocol_documents={"correction": "RULES"},
        context_documents={"payload-call-0001": "window one"},
    )
    task = _call(server, 2, "next_task")
    assert task["status"] == "task" and task["task_id"] == "call-0001"
    assert task["blocks"]["payload"] == "window one\n"
    accepted = _call(server, 3, "submit", payload="good")
    assert accepted["status"] == "accepted"
    assert "next_task" in accepted["message"]

    # A task added later may name a validator the first one did not.
    runtime.add_task(
        _spec("call-0002", validator="accept"),
        protocol_documents={"correction": "RULES"},
        context_documents={"payload-call-0002": "window two"},
    )
    second = _call(server, 4, "next_task")
    assert second["task_id"] == "call-0002"
    assert _call(server, 5, "submit", payload="anything")["status"] == "accepted"

    runtime.seal()
    done = _call(server, 6, "next_task")
    assert done["status"] == "assignment_complete" and "stop" in done["message"]


def test_a_parked_call_wakes_when_a_task_arrives_or_the_seal_lands(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    server = _server(runtime, wait=5.0)
    other = AgentTaskRuntime(tmp_path / "assignment", validators=_all_validators())

    def add_later() -> None:
        time.sleep(0.3)
        other.add_task(_spec("call-0001"), protocol_documents={"correction": "R"},
                       context_documents={"payload-call-0001": "w"})

    threading.Thread(target=add_later).start()
    started = time.monotonic()
    task = _call(server, 1, "next_task")
    assert task["status"] == "task"
    assert time.monotonic() - started < 4.0
    assert _call(server, 2, "submit", payload="x")["status"] == "accepted"

    threading.Thread(target=lambda: (time.sleep(0.3), other.seal())).start()
    started = time.monotonic()
    assert _call(server, 3, "next_task")["status"] == "assignment_complete"
    assert time.monotonic() - started < 4.0


def test_web_tools_are_exposed_by_the_launch_list_and_admitted_per_task(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    server = _server(runtime)
    assert {tool["name"] for tool in server.tool_definitions} >= set(WEB_TOOL_NAMES)
    runtime.add_task(_spec("call-0001"), protocol_documents={"correction": "R"},
                     context_documents={"payload-call-0001": "w"})
    _call(server, 1, "next_task")
    refused = _call(server, 2, "web_search", query="x")
    assert refused["isError"] is True
    assert "does not retrieve" in refused["content"][0]["text"]


# --- driver: parked silence is not a stall; the call timeout can be overridden


def test_a_parked_session_is_not_reaped_by_the_stall_watchdog(tmp_path, monkeypatch) -> None:
    from .test_llm_local_agent import _driver

    driver = _driver(tmp_path, monkeypatch, timeout_seconds=30, stall_timeout_seconds=0.5)
    started: list[dict] = []

    class Observer:
        @staticmethod
        def started(attempt):
            started.append(dict(attempt))

    with pytest.raises(LocalAgentTimeoutError) as caught:
        driver.run(
            [{"role": "user", "content": "sleep"}],
            task="general",
            parked=lambda: True,
            observer=Observer(),
            timeout_seconds=2.0,
        )
    # The override ended it, not the watchdog: two seconds of silence while
    # parked is waiting, not a stall.
    assert "exceeded 2s" in str(caught.value)
    assert started and started[0]["capsule_id"]


# --- host: one CLI, many tasks ----------------------------------------------


class _SessionFakeDriver:
    """Plays a CLI serving one pseudo-conversational session in-process."""

    driver_id = "claude-code"
    display_name = "Claude Code CLI"
    conversation_ttl_seconds = 0.0

    def __init__(self, answer=None, *, die_on_session: int = 0) -> None:
        # Empty model: the client matches the injected driver to any model.
        self.config = SimpleNamespace(model="", timeout_seconds=20)
        # No capsule unless a test wants one: the bundle needs somewhere to go.
        self.evidence_locator: dict | None = None
        self.answer = answer or (lambda payload, attempt: "good")
        self.die_on_session = die_on_session
        self.calls: list = []
        self.sessions = 0
        self.completion_between_tasks: list[bool] = []
        self.parked_seen: list[bool] = []

    def probe(self, *, refresh: bool = False):
        return DriverProbe(
            available=True, structured_events=True, no_persisted_session=True,
            no_user_config=True, no_user_rules=True, can_restrict_tools=True,
            has_web_search=True, supports_session_reuse=True,
            sandbox_kind="named_tool_allowlist", supports_mcp_config=True,
        )

    def meets_requirements(self, probe=None, *, native_search: bool = False):
        return True

    def _result(self, n: int):
        return SimpleNamespace(
            content="done", reported_model="claude",
            execution_attempt={"backend": "local_agent", "capsule_id": f"cap-{n}"},
            episode_id=f"cap-{n}", conversation_handle="", turn_identity="",
            normalized_events=(),
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )

    def run(self, messages, **kwargs):
        self.calls.append(kwargs)
        self.sessions += 1
        n = self.sessions
        env = kwargs["mcp_server"]["env"]
        runtime = AgentTaskRuntime(env["FINESUB_MCP_ROOT"], validators=_all_validators())
        server = HarnessToolServer(
            runtime,
            assignment_id=env["FINESUB_MCP_ASSIGNMENT"],
            worker_id=env["FINESUB_MCP_WORKER"],
            session_id=env["FINESUB_MCP_SESSION"],
            instance_id=f"instance-{n}",
            tools=env["FINESUB_MCP_TOOLS"].split(","),
            next_task_wait_seconds=0.3,
        )
        # What a real driver hands over right after the spawn.
        started = {"backend": "local_agent", "driver": "claude-code", "capsule_id": f"cap-{n}",
                   "reported_model": "claude", "isolation": {}}
        if self.evidence_locator is not None:
            started["evidence_locator"] = dict(self.evidence_locator)
        kwargs["observer"].started(started)
        completion, parked = kwargs["completion"], kwargs["parked"]
        rpc = 0
        while True:
            rpc += 1
            reply = _call(server, rpc, "next_task")
            if reply["status"] == "still_waiting":
                self.parked_seen.append(parked())
                if completion():
                    break
                continue
            if reply["status"] != "task":
                break
            if self.die_on_session == n:
                return self._result(n)  # left without submitting
            payload = reply["blocks"]["payload"]
            for attempt in range(4):
                rpc += 1
                out = _call(server, rpc, "submit", payload=self.answer(payload, attempt))
                if out["status"] in {"accepted", "repair_exhausted", "retired"}:
                    break
            self.completion_between_tasks.append(completion())
        return self._result(n)


def _client(driver, tmp_path):
    routes = load_model_routes(
        user_config={
            "model_groups": {
                "research-default": {"targets": ["local-claude-completion-sonnet-5"]}
            }
        }
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    config = replace(
        role_config_for("research", "quality", routes=routes),
        agent_session_mode="pseudo-conversational",
    )
    return RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={LLMRole.GENERAL_CAPABLE: config},
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
        agent_assignment_root=tmp_path / "assignments",
    )


def _no_api(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )


REJECT_BAD = {"id": "test-reject-bad", "params": {}}


def _complete(client, text, **kwargs):
    return client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "system", "content": "RULES"}, {"role": "user", "content": text}],
        validator_spec=REJECT_BAD,
        max_repair_attempts=kwargs.pop("max_repair_attempts", 1),
        **kwargs,
    )




def test_every_task_of_a_session_is_audited_into_the_capsule(tmp_path, monkeypatch) -> None:
    """One capsule, one bundle per task (docs §5).

    A session's assignment tree is disposable exactly because these are not:
    they follow the capsule's retention, so a clean run drops both and a
    failing one keeps both for `agent-clean`.
    """

    _no_api(monkeypatch)
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    location = resolve_agent_episode_location(episodes)
    driver = _SessionFakeDriver()
    driver.evidence_locator = evidence_locator(location, "cap-1").as_dict()
    client = _client(driver, tmp_path)

    with agent_session_scope() as registry:
        _complete(client, "window one")
        _complete(client, "window two")
        host = registry.hosts[0]
    bundles = sorted(path.name for path in (episodes / "cap-1").iterdir())
    assert bundles == ["audit-call-0001", "audit-call-0002"]
    first = episodes / "cap-1" / "audit-call-0001"
    assert (first / "artifact.txt").read_text(encoding="utf-8") == "good"
    assert json.loads((first / "manifest.json").read_text(encoding="utf-8"))["task_id"] == "call-0001"
    assert (first / "blocks" / "payload.md").read_text(encoding="utf-8").strip().endswith("window one")
    assert json.loads((first / "outcome.json").read_text(encoding="utf-8"))["accepted"] is True
    # The tree the bundles were copied out of is gone; they are not.
    assert not host.root.exists()



def test_a_lost_audit_bundle_keeps_the_assignment(tmp_path, monkeypatch) -> None:
    """The clean-up rule is "the evidence is elsewhere", not "it succeeded".

    A capsule the bundle could not be written into leaves this tree as the
    only account of the task, so the session stops being disposable.
    """

    _no_api(monkeypatch)
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    driver = _SessionFakeDriver()
    driver.evidence_locator = evidence_locator(
        resolve_agent_episode_location(episodes), "cap-1"
    ).as_dict()
    monkeypatch.setattr(host_module, "write_audit_bundle", lambda *a, **k: False)
    client = _client(driver, tmp_path)
    with agent_session_scope() as registry:
        _complete(client, "window one")
        host = registry.hosts[0]
    assert host.incidents == 1
    assert host.root.exists()


def test_a_session_without_a_capsule_is_still_disposable(tmp_path, monkeypatch) -> None:
    """No capsule means there was never anywhere for a bundle to go."""

    _no_api(monkeypatch)
    client = _client(_SessionFakeDriver(), tmp_path)
    with agent_session_scope() as registry:
        _complete(client, "window one")
        host = registry.hosts[0]
    assert host.incidents == 0 and not host.root.exists()


def test_a_clean_session_leaves_no_assignment_behind(tmp_path, monkeypatch) -> None:
    """Every task accepted: the tree has no reader left (docs §5).

    It holds this run's whole subtitle text and every MCP frame, and a single
    task tool session deletes its own for exactly that reason.
    """

    _no_api(monkeypatch)
    client = _client(_SessionFakeDriver(), tmp_path)
    with agent_session_scope() as registry:
        _complete(client, "window one")
        _complete(client, "window two")
        host = registry.hosts[0]
        assert host.root.exists()
    # Named separately: a session that hiccupped (a premature stop it
    # recovered from) is deliberately kept, so a failure here should say
    # which of the two rules it was.
    assert host.incidents == 0
    assert not host.root.exists()
    # The totals survive the tree: they are the run's, not the session's.
    assert host.usage_totals()["total_tokens"] == 12


def test_a_run_that_raises_keeps_its_sessions_readable(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    client = _client(_SessionFakeDriver(), tmp_path)
    host = None
    with pytest.raises(RuntimeError, match="boom"):
        with agent_session_scope() as registry:
            _complete(client, "window one")
            host = registry.hosts[0]
            raise RuntimeError("boom")
    # The session itself was clean, but the run around it was not: its tree
    # is the readable account of what the agent was given and answered.
    assert host is not None and host.root.exists()
    assert host.runtime.status(assignment_id=host.assignment_id)["status"] == "assignment_complete"


def test_the_run_books_session_usage_where_the_report_reads_it(tmp_path, monkeypatch) -> None:
    """Per-call records of a pseudo session carry no tokens; the run's do."""

    _no_api(monkeypatch)
    client = _client(_SessionFakeDriver(), tmp_path)
    with agent_session_scope() as registry:
        result = _complete(client, "window one")
        _complete(client, "window two")
    # Nothing on the call itself -- usage is metered per CLI invocation.
    assert result.raw_response["usage"]["total_tokens"] == 0

    artifact_dir = tmp_path / "artifacts"
    assert write_agent_session_usage(artifact_dir, registry) is not None
    sessions = json.loads(
        (artifact_dir / AGENT_SESSION_USAGE_FILENAME).read_text(encoding="utf-8")
    )["sessions"]
    assert [row["usage"]["total_input_tokens"] for row in sessions] == [10]
    assert sessions[0]["mode"] == "pseudo-conversational"
    report = render_task_report([], agent_sessions=sessions)
    row = next(line for line in report.splitlines() if sessions[0]["model"] in line)
    # Tokens from the session, calls from the windows: neither derives from
    # the other, and the session never made a "call" of its own.
    assert row.endswith("| 0 | 10 | 0 | 2 | 0 |")
    # One agent, one row: the fold must key exactly as the per-call rows do.
    # Case-folding the tier here split it into a row with the calls and a row
    # with the tokens (seen on a real agy run).
    per_call = {(sessions[0]["provider_tier"], sessions[0]["model"]): Counter({"calls": 2})}
    _add_agent_session_usage(per_call, sessions)
    assert list(per_call) == [(sessions[0]["provider_tier"], sessions[0]["model"])]
    assert per_call[(sessions[0]["provider_tier"], sessions[0]["model"])]["calls"] == 2

    # An artifact directory is reused across runs: a later run that spends
    # nothing on agents must clear this book, or the report keeps reading the
    # previous run's tokens out of it.
    empty = SimpleNamespace(usage_rows=list)
    assert write_agent_session_usage(artifact_dir, empty) is None
    assert not (artifact_dir / AGENT_SESSION_USAGE_FILENAME).exists()


def test_the_session_scope_is_re_entrant(tmp_path, monkeypatch) -> None:
    """A stage that opens its own scope inside a run shares the run's.

    Otherwise research, the windows and the knowledge update would each get
    their own CLI, and only the innermost scope would ever reclaim one.
    """

    _no_api(monkeypatch)
    driver = _SessionFakeDriver()
    client = _client(driver, tmp_path)
    with agent_session_scope() as outer:
        _complete(client, "window one")
        with agent_session_scope() as inner:
            assert inner is outer
            _complete(client, "window two")
        # The inner block leaving must not have ended the run's session.
        assert outer.hosts[0]._session_alive()
    assert driver.sessions == 1


def test_two_calls_ride_one_cli_session_and_the_scope_ends_it(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    driver = _SessionFakeDriver()
    client = _client(driver, tmp_path)

    with agent_session_scope() as registry:
        first = _complete(client, "window one")
        second = _complete(client, "window two")
        host = registry.hosts[0]
        assert host._session_alive()
    # The scope sealed the assignment; the session saw `assignment_complete`
    # and left by itself.
    assert not host._session_alive()

    assert first.content == "good" and second.content == "good"
    assert driver.sessions == 1
    # The first task's accept did not complete the session -- that is the whole
    # point of the tier. Only the first reading says so: the last one is taken
    # after the last submit, which races the scope's seal on the main thread,
    # and `True` there means "the run ended", not "one accept ended it".
    assert driver.completion_between_tasks[0] is False
    # Whenever the session did park between tasks, the driver saw it as parked.
    assert all(driver.parked_seen)
    assert first.resumable is True
    # The second task inherited the first's history: not resumable (gate D, C).
    assert second.resumable is False
    assert first.route_decision["candidates"][0]["agent_transport"] == "tool-session"
    attempt = second.execution_attempts[-1]
    assert attempt["usage_attribution"] == "session" and attempt["task_id"] == "call-0002"
    assert host.session_usage[0]["usage"]["total_tokens"] == 12


def test_a_cli_that_leaves_mid_task_gets_one_fresh_session_on_the_same_task(
    tmp_path, monkeypatch
) -> None:
    _no_api(monkeypatch)
    driver = _SessionFakeDriver(die_on_session=1)
    client = _client(driver, tmp_path)
    with agent_session_scope():
        result = _complete(client, "window one")
        again = _complete(client, "window two")
    assert result.content == "good" and again.content == "good"
    assert driver.sessions == 2
    assert any(item.get("premature_stop") for item in result.execution_attempts)


def test_a_spent_repair_budget_withdraws_the_task_and_a_replacement_starts_fresh(
    tmp_path, monkeypatch
) -> None:
    _no_api(monkeypatch)
    answers = iter(["bad"])
    driver = _SessionFakeDriver(lambda payload, attempt: next(answers, "good"))
    client = _client(driver, tmp_path)
    with agent_session_scope() as registry:
        spent = _complete(client, "window one", max_repair_attempts=0)
        assert spent.repair_exhausted is True and spent.content == "bad"
        host = registry.hosts[0]
        record = host.runtime.task_record(assignment_id=host.assignment_id, task_id="call-0001")
        assert record["status"] == "withdrawn"
        replacement = _complete(client, "window one", fresh_session=True, max_repair_attempts=0)
        assert replacement.content == "good"
    # Tier 2 escaped the conversation: a second CLI answered.
    assert driver.sessions == 2


def test_leaving_the_scope_by_exception_reclaims_a_parked_session(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)
    driver = _SessionFakeDriver()
    client = _client(driver, tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with agent_session_scope() as registry:
            _complete(client, "window one")
            host = registry.hosts[0]
            assert host._session_alive()
            raise RuntimeError("boom")
    assert not host._session_alive()
    assert current_registry() is None
    assert host.runtime.status(assignment_id=host.assignment_id)["status"] == "assignment_complete"


def test_a_task_nobody_finishes_times_out_and_reclaims_the_session(tmp_path, monkeypatch) -> None:
    _no_api(monkeypatch)

    class Stuck(_SessionFakeDriver):
        def run(self, messages, **kwargs):
            self.calls.append(kwargs)
            self.sessions += 1
            completion = kwargs["completion"]
            while not completion():
                time.sleep(0.05)
            return self._result(self.sessions)

    driver = Stuck()
    client = _client(driver, tmp_path)
    with agent_session_scope() as registry:
        def build(driver=driver):
            return AgentSessionHost(
                driver, root=tmp_path / "assignments" / "s", execution_identity={},
                task_timeout_seconds=0.6, label="test",
            )

        host = registry.host_for(("LOCAL_CLAUDE", "sonnet-5", 1, "pseudo-conversational"), build)
        monkeypatch.setattr(client, "_agent_session_host", lambda *a, **k: host)
        with pytest.raises(Exception) as caught:
            _complete(client, "window one")
        assert "did not finish" in str(caught.value)
        record = host.runtime.task_record(assignment_id=host.assignment_id, task_id="call-0001")
        assert record["status"] == "withdrawn"
    assert not host._session_alive()


# --- paging: what the CLI cannot show inline is read page by page ----------


def test_blocks_past_the_inline_limit_are_paged_and_booked_on_the_last_page(tmp_path) -> None:
    """agy replaces a tool reply past ~4-5k characters with a file path and no
    preview (measured 2026-08-22): the server pushes only what fits and
    `read_context` pages the rest, booking the block once its end was read."""

    runtime = _unsealed(tmp_path)
    protocol = "RULES " * 100  # 600 chars: fits
    payload = "".join(f"line {index:04d} " + "x" * 40 + "\n" for index in range(200))  # ~10k
    runtime.add_task(
        _spec("call-0001"),
        protocol_documents={"correction": protocol},
        context_documents={"payload-call-0001": payload},
    )
    server = HarnessToolServer(
        runtime, assignment_id="assignment-1", worker_id="worker-1",
        session_id="s1", instance_id="i1", page_chars=3000,
    )
    task = _call(server, 1, "next_task")
    assert list(task["blocks"]) == ["protocol"]
    paged = {block["kind"]: block for block in task["required_blocks"]}
    assert paged["payload"]["read"] == "paged" and paged["payload"]["chars"] == len(payload)
    assert "read" not in paged["protocol"]

    # Submitting now owes the payload: only its last page books it.
    owed = _call(server, 2, "submit", payload="x")
    assert [block["kind"] for block in owed["owed_blocks"]] == ["payload"]

    ref = paged["payload"]["ref"]
    offset, pages, seen = 0, 0, ""
    while offset is not None:
        page = _call(server, 10 + pages, "read_context", ref=ref, offset=offset)
        assert len(page["text"]) <= 3000
        assert page["text"].endswith("\n") or page["next_offset"] is None  # whole lines
        seen += page["text"]
        pages += 1
        if page["next_offset"] is not None:
            assert "owed_blocks" not in page
        offset = page["next_offset"]
    assert seen == payload and pages >= 4
    assert page["owed_blocks"] == []
    assert _call(server, 40, "submit", payload="good")["status"] == "accepted"


def test_an_unknown_ref_names_the_readable_ones_and_a_bad_offset_is_refused(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    runtime.add_task(_spec("call-0001"), protocol_documents={"correction": "R"},
                     context_documents={"payload-call-0001": "w"})
    server = _server(runtime)
    task = _call(server, 1, "next_task")
    refs = {block["ref"] for block in task["required_blocks"]}
    error = _call(server, 2, "read_context", ref="protocol")["content"][0]["text"]
    assert all(ref in error for ref in refs)
    bad = _call(server, 3, "read_context", ref=next(iter(refs)), offset=99999)
    assert bad["isError"] is True and "past the end" in bad["content"][0]["text"]


def test_without_a_limit_everything_is_pushed_and_read_whole(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    payload = "y" * 20000
    runtime.add_task(_spec("call-0001"), protocol_documents={"correction": "R"},
                     context_documents={"payload-call-0001": payload})
    server = _server(runtime)  # page_chars 0
    task = _call(server, 1, "next_task")
    assert task["blocks"]["payload"] == payload + "\n"
    ref = [b for b in task["required_blocks"] if b["kind"] == "payload"][0]["ref"]
    page = _call(server, 2, "read_context", ref=ref)
    assert page["next_offset"] is None and page["total_chars"] == len(payload) + 1


def test_the_host_hands_the_driver_page_limit_to_the_server(tmp_path) -> None:
    driver = _SessionFakeDriver()
    driver.config = SimpleNamespace(model="", timeout_seconds=20, mcp_page_chars=3000,
                                    conversation_ttl_seconds=300.0, next_task_wait_seconds=240.0)
    host = AgentSessionHost(driver, root=tmp_path / "s", execution_identity={},
                            task_timeout_seconds=10, label="t")
    spec = host._mcp_server_spec("sid")
    assert spec["env"]["FINESUB_MCP_PAGE_CHARS"] == "3000"
    assert spec["env"]["FINESUB_MCP_WAIT_SECONDS"] == "240"
    host.close()


# --- blocks as files: the CLI reads them itself (agy) ------------------------


def test_block_files_mode_hands_over_paths_and_books_the_ledger(tmp_path) -> None:
    runtime = _unsealed(tmp_path)
    payload = "".join(f"line {index:04d}\n" for index in range(2000))  # ~20k
    runtime.add_task(_spec("call-0001"), protocol_documents={"correction": "RULES"},
                     context_documents={"payload-call-0001": payload})
    server = HarnessToolServer(
        runtime, assignment_id="assignment-1", worker_id="worker-1",
        session_id="s1", instance_id="i1", page_chars=2800, block_files=True,
    )
    task = _call(server, 1, "next_task")
    assert task["blocks"] == {}
    blocks = {block["kind"]: block for block in task["required_blocks"]}
    for kind in ("protocol", "payload"):
        assert blocks[kind]["read"] == "file"
        path = Path(blocks[kind]["path"])
        assert path.is_absolute() and path.is_file()
        assert path.resolve().is_relative_to(runtime.root)
    assert Path(blocks["payload"]["path"]).read_text(encoding="utf-8") == payload
    # Handed over as files: nothing is owed, the submit goes to the validator.
    assert _call(server, 2, "submit", payload="good")["status"] == "accepted"


def test_the_host_and_the_tool_call_name_the_assignment_root_as_a_view_root(tmp_path) -> None:
    driver = _SessionFakeDriver()
    driver.config = SimpleNamespace(model="", timeout_seconds=20, mcp_page_chars=2800,
                                    mcp_block_files=True, conversation_ttl_seconds=300.0,
                                    next_task_wait_seconds=240.0)
    host = AgentSessionHost(driver, root=tmp_path / "s", execution_identity={},
                            task_timeout_seconds=10, label="t")
    spec = host._mcp_server_spec("sid")
    assert spec["env"]["FINESUB_MCP_BLOCK_FILES"] == "1"
    assert spec["view_roots"] == [str(host.root)]
    host.close()


def test_a_retrieval_mode_change_does_not_reuse_the_entitled_session(
    tmp_path, monkeypatch
) -> None:
    """One CLI's built-in tools are fixed at launch, so the mode is part of
    the session's identity.

    Reusing the session across a change is wrong in both directions: a later
    `native` task would have no search tool at all, and a later `local`/`none`
    task would keep a search tool it was never granted.
    """

    _no_api(monkeypatch)
    driver = _SessionFakeDriver()
    # The native-declaring target: `retrieval=native` filters on that
    # declaration, so the completion target the other tests use cannot serve
    # the second call at all.
    routes = load_model_routes(
        user_config={
            "model_groups": {
                "research-default": {"targets": ["local-claude-native-sonnet-5"]}
            }
        }
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: replace(
                role_config_for("research", "quality", routes=routes),
                agent_session_mode="pseudo-conversational",
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
        agent_assignment_root=tmp_path / "assignments",
    )

    assert _complete(client, "one", retrieval="local").content == "good"
    assert _complete(client, "two", retrieval="native").content == "good"

    assert [call["native_search"] for call in driver.calls] == [False, True]
    assert driver.sessions == 2


def test_each_task_folds_its_own_proxied_retrieval(tmp_path, monkeypatch) -> None:
    """A session's events belong to the session; a task's sources do not.

    The host returns `normalized_events=()` per task, so without folding the
    ledger in per task, everything `retrieval=local` found in a
    pseudo-conversational run is missing from the evidence downstream reads.
    """

    from finesub.llm.agent import agent_session_host as host_module

    _no_api(monkeypatch)
    folded: list[str] = []
    real = host_module.fold_proxied_retrieval

    def spy(runtime, result, *, assignment_id, task_id):
        folded.append(task_id)
        return real(runtime, result, assignment_id=assignment_id, task_id=task_id)

    monkeypatch.setattr(host_module, "fold_proxied_retrieval", spy)
    client = _client(_SessionFakeDriver(), tmp_path)

    _complete(client, "one", retrieval="local")
    _complete(client, "two", retrieval="local")

    # Per task, with that task's own id -- not once for the whole session.
    assert len(folded) == 2 and len(set(folded)) == 2


def test_an_entitlement_switch_does_not_spend_a_second_driver_slot(
    tmp_path, monkeypatch
) -> None:
    """A long-lived session holds one of the driver's `max_parallel` slots for
    as long as it runs.

    Keying native and proxied sessions apart instead of replacing one with the
    other leaves both holding a slot for the length of the run; at
    `max_parallel=1` the second could never start, and the call that needs it
    waits for a session that only ends when the run does.
    """

    _no_api(monkeypatch)

    class _OneSlotDriver(_SessionFakeDriver):
        """Plays the driver's in-flight gate at `max_parallel=1`."""

        def __init__(self) -> None:
            super().__init__()
            self.slot = threading.BoundedSemaphore(1)
            self.starved = False

        def run(self, messages, **kwargs):
            if not self.slot.acquire(timeout=10):
                self.starved = True
                raise AssertionError("no free driver slot for this session")
            try:
                return super().run(messages, **kwargs)
            finally:
                self.slot.release()

    driver = _OneSlotDriver()
    routes = load_model_routes(
        user_config={
            "model_groups": {
                "research-default": {"targets": ["local-claude-native-sonnet-5"]}
            }
        }
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: replace(
                role_config_for("research", "quality", routes=routes),
                agent_session_mode="pseudo-conversational",
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
        agent_assignment_root=tmp_path / "assignments",
    )

    assert _complete(client, "one", retrieval="local").content == "good"
    assert _complete(client, "two", retrieval="native").content == "good"

    assert driver.starved is False
    # The first session ended before the second started: one slot, in turn.
    assert [call["native_search"] for call in driver.calls] == [False, True]
