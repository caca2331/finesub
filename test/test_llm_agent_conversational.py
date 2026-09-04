"""Conversational workers (docs/llm_local_agent.md §12.1.4): a person's own
agent claims the run's tasks over `finesub agent-task`.

The queue side is the runtime's `claimable_by` dimension and worker kinds;
the harness side is `ConversationalQueue` behind a cell bound to the
conversational target. The scripted host below is a legal worker: it drives
the control CLI exactly as an agent would (`next-task` -> read the files ->
`submit`), so the whole path is covered without a person -- except the
person walking away, which the lease-expiry test covers on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from finesub.llm.agent import agent_validators
from finesub.llm.agent.agent_join import main as join_main
from finesub.llm.agent.agent_session_host import (
    CONVERSATIONAL_JOIN_WAIT_SECONDS,
    ConversationalQueue,
    agent_session_scope,
)
from finesub.llm.agent.agent_task_runtime import (
    REGISTRATION_GRACE_SECONDS,
    AgentTaskRuntime,
    AgentTaskRuntimeError,
    AgentTaskSpec,
    AssignmentConflictError,
    StaleControlGenerationError,
    ValidationResult,
)
from finesub.llm.agent.agent_transports import AgentRuntimeCallError
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
            # `startswith`, so a test can make its rejected draft distinctive
            # enough to grep the whole tree for.
            if str(candidate).startswith("bad"):
                return ValidationResult.repairable("say good, not bad")
            return ValidationResult.accepted(candidate)

        return validate

    monkeypatch.setitem(agent_validators.VALIDATOR_BUILDERS, "test-reject-bad", build)


def _files_containing(root: Path, needle: str) -> list[str]:
    """Every file under `root` whose bytes contain `needle`."""

    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if needle in text:
            hits.append(str(path.relative_to(root)))
    return hits


def _spec(task_id, *, claimable_by=("conversational",), retrieval="none", validator="accept"):
    return AgentTaskSpec(
        task_id=task_id,
        session_type="correction",
        input_hash=f"sha256:{task_id}",
        goal="answer",
        protocol_key="correction",
        context_key=f"payload-{task_id}",
        retrieval_mode=retrieval,
        required_blocks=(
            {"kind": "protocol", "digest": "@protocol"},
            {"kind": "payload", "digest": "@context"},
        ),
        claimable_by=claimable_by,
        validator_id=validator,
    )


def _runtime(tmp_path, *specs, max_workers=2):
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="serve",
        tasks=[],
        session_scope="task",
        max_workers=max_workers,
        sealed=False,
    )
    for spec in specs:
        runtime.add_task(
            spec,
            protocol_documents={"correction": "RULES"},
            context_documents={f"payload-{spec.task_id}": f"text of {spec.task_id}"},
        )
    return runtime


def _request_table(runtime) -> dict:
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    return json.loads(runtime.read_artifact(index["state_ref"]))["request_results"]


def _claim(runtime, worker, kind):
    status = runtime.rehydrate(assignment_id="assignment-1", worker_id=worker)
    return runtime.next_task(
        assignment_id="assignment-1", worker_id=worker,
        request_id=f"claim-{worker}-{status['control_generation']}",
        expected_control_generation=status["control_generation"], worker_kind=kind,
    )


# --- runtime: who may claim what ---------------------------------------------


def test_claimable_by_and_worker_kind_decide_who_gets_a_task(tmp_path) -> None:
    runtime = _runtime(
        tmp_path,
        _spec("for-person"),
        _spec("for-cli", claimable_by=("headless",)),
        _spec("native", retrieval="native"),
        max_workers=3,
    )
    headless = _claim(runtime, "cli-1", "headless")
    assert headless["task"]["task_id"] == "for-cli"
    person = _claim(runtime, "conv-1", "conversational")
    assert person["task"]["task_id"] == "for-person"
    # The native-search task is nobody's: not the person's (no harness search
    # entitlement) and not headless-claimable.
    assert _claim(runtime, "conv-2", "conversational")["status"] == "waiting"
    # A worker keeps its kind.
    with pytest.raises(AssignmentConflictError, match="registered as"):
        _claim(runtime, "conv-1", "headless")


def test_a_conversational_claim_hands_the_blocks_over_as_files(tmp_path) -> None:
    runtime = _runtime(tmp_path, _spec("t1"))
    task = _claim(runtime, "conv-1", "conversational")["task"]
    ledger = runtime.pull_status(assignment_id="assignment-1", task_id="t1", worker_id="conv-1")
    assert ledger["owed_blocks"] == []
    manifest = json.loads(runtime.read_artifact(task["manifest_ref"]))
    assert (runtime.root / manifest["context_ref"].split("#")[0]).read_text(encoding="utf-8") == "text of t1\n"
    # Nothing owed: the submit goes straight to the validator.
    accepted = runtime.submit(
        assignment_id="assignment-1", task_id="t1", worker_id="conv-1",
        lease_generation=task["lease_generation"], request_id="s1",
        input_hash="sha256:t1", candidate="done",
    )
    assert accepted["accepted_task_id"] == "t1"


# --- a scripted host: the control CLI driven the way an agent drives it -----


def _run_cli(args: list[str]) -> dict:
    """The control CLI's parse + dispatch, minus the stdout round trip (this
    runs on a thread, where redirecting the process's stdout is not safe)."""

    from finesub.llm.agent import agent_task_control

    parsed = agent_task_control._parser().parse_args(args)
    return agent_task_control._dispatch(agent_task_control.runtime_for(parsed.root), parsed)


def _scripted_host(root: Path, *, answer=lambda text: "good", stop: threading.Event,
                   worker="conv-1", served=None, think_seconds=0.0):
    """`next-task` -> read the manifest and its files -> `submit`, until told to stop.

    ``think_seconds`` is how long the agent spends between claiming a task and
    submitting it -- a real one takes minutes. Nothing renews the lease during
    it, which is the point: a caller that keeps waiting through it is trusting
    the lease's own deadline, not a wall clock of its own.
    """

    index = json.loads((root / "control" / "index.json").read_text(encoding="utf-8"))
    assignment = index["assignment_id"]
    common = ["--root", str(root), "--assignment", assignment]
    served = [] if served is None else served
    while not stop.is_set():
        status = _run_cli([*common, "status", "--worker", worker])
        if status["status"] == "assignment_complete":
            # What the bootstrap tells a person's agent: this is the only
            # signal that the worker goal is done.
            return served
        if status["status"] == "task":
            task = status["task"]
        else:
            claim = _run_cli([*common, "next-task", "--worker", worker, "--request-id",
                              f"claim-{status['control_generation']}",
                              "--control-generation", str(status["control_generation"])])
            if claim["status"] != "task":
                time.sleep(0.1)
                continue
            task = claim["task"]
        manifest = json.loads((root / task["manifest_ref"].split("#")[0]).read_text(encoding="utf-8"))
        payload = (root / manifest["context_ref"].split("#")[0]).read_text(encoding="utf-8")
        if think_seconds:
            time.sleep(think_seconds)
        body = root / f"answer-{task['task_id']}.json"
        body.write_text(json.dumps(answer(payload)), encoding="utf-8")
        _run_cli([*common, "submit", "--worker", worker, "--request-id", f"submit-{task['task_id']}",
                  "--task", task["task_id"], "--lease-generation", str(task["lease_generation"]),
                  "--input-hash", manifest["input_hash"], "--json-file", str(body)])
        served.append(task["task_id"])
    return served


def _client(tmp_path):
    routes = load_model_routes(
        user_config={
            "presets": {},
            "model_groups": {"research-default": {"targets": ["conversational-agent"]}},
        }
    )
    settings = ExecutionSettings(policy_id="agent-only", local_agent_timeout_seconds=20)
    return RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={LLMRole.GENERAL_CAPABLE: role_config_for("research", "quality", routes=routes)},
        rate_limiter=ModelRateLimiter(enabled=False),
        agent_assignment_root=tmp_path / "assignments",
    )


def _no_api(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )


def test_a_conversational_cell_is_detected_for_forced_serial(tmp_path) -> None:
    """Plan W6: the correction stage asks this before honouring
    `continuity=parallel` -- a conversational chain runs its windows serially
    (one queue, however many agents join), keeping the advice ledger."""

    client = _client(tmp_path)
    assert client.routes_to_conversational(LLMRole.GENERAL_CAPABLE) is True
    plain = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))
    assert plain.routes_to_conversational(LLMRole.GENERAL_CAPABLE) is False


def _complete(client, text):
    return client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "system", "content": "RULES"}, {"role": "user", "content": text}],
        validator_spec={"id": "test-reject-bad", "params": {}},
        max_repair_attempts=1,
    )


def test_a_cell_bound_to_the_conversational_target_is_served_by_a_joined_agent(
    tmp_path, monkeypatch
) -> None:
    _no_api(monkeypatch)
    warnings: list[tuple[str, str]] = []
    from finesub.llm.agent import agent_session_host as host_module

    class Reporter:
        @staticmethod
        def warning(code, message, **kwargs):
            warnings.append((code, message))

        @staticmethod
        def debug(*a, **k):
            pass

    monkeypatch.setattr(host_module, "current_reporter", lambda: Reporter())
    client = _client(tmp_path)
    stop = threading.Event()
    served: list[str] = []

    def run_host() -> None:
        # Wait for the run to announce its assignment root, like a person would.
        while not warnings and not stop.is_set():
            time.sleep(0.05)
        root = Path(warnings[0][1].split('finesub agent-join "')[1].rstrip('"'))
        _scripted_host(root, stop=stop, served=served)

    thread = threading.Thread(target=run_host, daemon=True)
    thread.start()
    with agent_session_scope():
        first = _complete(client, "window one")
        second = _complete(client, "window two")
    stop.set()
    thread.join(timeout=60)

    assert first.content == "good" and second.content == "good"
    assert served == ["call-0001", "call-0002"]
    assert first.route_decision["candidates"][0]["agent_transport"] == "conversational"
    assert first.execution_attempts[-1]["backend"] == "conversational_agent"
    assert first.execution_attempts[-1]["worker_id"] == "conv-1"
    assert warnings[0][0] == "agent-join" and len(warnings) == 1



def test_two_agents_joining_get_their_own_worker_id(tmp_path) -> None:
    """Joining reserves the id (docs §12.1.4).

    Reading the worker list and picking the first free name is not enough:
    two people joining before either has claimed anything both read the same
    list, and the second one would resume the first one's task as its own.
    """

    queue = ConversationalQueue(parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t")
    joined: list[str] = []
    barrier = threading.Barrier(2)

    def join() -> None:
        barrier.wait()
        assert join_main([str(queue.root)]) == 0

    threads = [threading.Thread(target=join) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    state = json.loads(
        queue.runtime.read_artifact(
            json.loads(queue.runtime.index_path.read_text(encoding="utf-8"))["state_ref"]
        )
    )
    joined = list(state["worker_ids"])
    assert sorted(joined) == ["conv-1", "conv-2"]
    assert {name: row["kind"] for name, row in state["workers"].items()} == {
        "conv-1": "conversational", "conv-2": "conversational",
    }
    queue.close()


def test_a_claim_that_lost_the_generation_race_is_re_planned(tmp_path) -> None:
    """The harness adds the next window while the agent is claiming.

    A `next-task` carries the control generation it was planned against, and
    `add_task` moves that. Losing the race is news of work arriving, not an
    error the person's agent should have to know how to read.
    """

    queue = ConversationalQueue(parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t")
    queue.runtime.add_task(_spec("t1"), protocol_documents={"correction": "R"},
                           context_documents={"payload-t1": "p"})
    common = ["--root", str(queue.root), "--assignment", queue.assignment_id]
    stale = _run_cli([*common, "status", "--worker", "conv-1"])["control_generation"]
    # Somebody else writes between the plan and the claim.
    queue.runtime.add_task(_spec("t2"), protocol_documents={"correction": "R"},
                           context_documents={"payload-t2": "p"})
    claim = _run_cli([*common, "next-task", "--worker", "conv-1", "--request-id", "c1",
                      "--control-generation", str(stale)])
    assert claim["status"] == "task" and claim["task"]["task_id"] == "t1"
    queue.close()



def test_every_task_scoped_request_is_filed_under_its_task(tmp_path) -> None:
    """Not just the ones the MCP server issues (docs §7).

    A person's agent drives `finesub agent-task`, which also offers progress
    and conversation checkpoints and `release`. Those rows used to carry no
    task id, so nothing ever pruned them and a long queue filled the bounded
    request table anyway.
    """

    runtime = _runtime(tmp_path, _spec("t1"))
    task = _claim(runtime, "conv-1", "conversational")["task"]
    lease = task["lease_generation"]
    runtime.checkpoint_progress(
        assignment_id="assignment-1", task_id="t1", worker_id="conv-1",
        lease_generation=lease, request_id="p1", progress={"done": 1},
    )
    runtime.checkpoint_conversation(
        assignment_id="assignment-1", task_id="t1", worker_id="conv-1",
        lease_generation=lease, request_id="c1", conversation_epoch=1,
        conversation_handle="h", turn_generation=1, parent_turn_identity="turn-1",
    )
    table = _request_table(runtime)
    # The claim, the progress checkpoint and the conversation checkpoint: all
    # of them are this task's, so all of them say so.
    assert set(table) == {"claim-conv-1-2", "p1", "c1"}
    assert {row["task_id"] for row in table.values()} == {"t1"}

    runtime.submit(
        assignment_id="assignment-1", task_id="t1", worker_id="conv-1",
        lease_generation=lease, request_id="s1",
        input_hash="sha256:t1", candidate="done",
    )
    # Accepted: the task took every one of its rows with it.
    assert _request_table(runtime) == {}


def test_a_registration_counts_against_max_workers_and_expires(tmp_path) -> None:
    """A join nobody followed through on must not hold a slot for ever."""

    clock = {"now": 1000.0}
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment", assignment_id="assignment-1", worker_goal="serve",
        tasks=[], session_scope="task", max_workers=1, sealed=False,
        clock=lambda: clock["now"],
    )
    assert runtime.register_worker(assignment_id="assignment-1")["worker_id"] == "conv-1"
    # Idempotent by id: a rejoin is not a second worker.
    assert runtime.register_worker(
        assignment_id="assignment-1", worker_id="conv-1"
    )["worker_id"] == "conv-1"
    with pytest.raises(AssignmentConflictError, match="takes 1 worker"):
        runtime.register_worker(assignment_id="assignment-1")
    # It never claimed anything: past the grace its slot goes back.
    clock["now"] += REGISTRATION_GRACE_SECONDS + 1
    assert runtime.register_worker(assignment_id="assignment-1")["worker_id"] == "conv-1"



def test_a_media_call_never_reaches_the_queue(tmp_path) -> None:
    """Text-only is enforced at the call, not by the catalog's columns.

    The routing filter already skips this target for a media call, because
    the catalog says it can neither hear nor watch. That is one `false` away
    from being wrong, and the protocol has no media dimension at all: a part
    that got through would be dropped silently on the way to the payload
    document. So the guard sits on the document itself.
    """

    client = _client(tmp_path)
    messages = [
        {"role": "system", "content": "RULES"},
        {"role": "user", "content": [{"text": "window one"}, {"file_id": "files/abc"}]},
    ]
    with pytest.raises(AgentRuntimeCallError, match="text-only"):
        client.complete(
            LLMRole.GENERAL_CAPABLE, messages,
            validator_spec={"id": "accept", "params": {}}, max_repair_attempts=0,
        )
    # Nothing was queued: no assignment was ever created for it.
    assert not (tmp_path / "assignments").exists()


def test_agent_join_prints_the_bootstrap_for_a_waiting_run(tmp_path, capsys) -> None:
    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    from finesub.llm.agent.agent_transports import agent_task_command

    assert join_main([str(queue.root)]) == 0
    text = capsys.readouterr().out
    assert agent_task_command() in text and queue.assignment_id in text
    assert "--worker conv-1" in text
    # The parameters the first live test had to guess at, spelled out.
    for flag in ("--request-id", "--lease-generation", "--input-hash", "--text-file"):
        assert flag in text, flag
    assert "lint" in text
    queue.close()


def test_the_bootstrap_names_a_command_this_machine_can_actually_run(
    tmp_path, monkeypatch, capsys
) -> None:
    """A source checkout has no `finesub` executable.

    The bootstrap hardcoded one, so an agent following the protocol from a
    checkout was told to run something that does not exist -- and only found
    out after being handed the job.
    """

    from finesub.llm.agent import agent_transports

    monkeypatch.setattr(agent_transports.shutil, "which", lambda _name: None)
    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    assert join_main([str(queue.root)]) == 0
    text = capsys.readouterr().out
    assert "finesub agent-task" not in text
    assert "-m finesub.llm.agent.agent_task_control" in text
    queue.close()


def test_nobody_joining_fails_the_call_after_its_wait(tmp_path) -> None:
    queue = ConversationalQueue(parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=1, label="t")
    with pytest.raises(AgentRuntimeCallError, match="agent-join"):
        queue.run_task(
            session_type="correction", input_hash="sha256:x", validator_id="accept",
            metadata={}, protocol_text="R", payload_text="p", retrieval_mode="none", max_repairs=0,
        )
    assert queue.runtime.task_record(assignment_id=queue.assignment_id, task_id="call-0001")["status"] == "withdrawn"
    queue.close()


def test_an_agent_that_walks_away_loses_its_lease_and_the_next_one_finishes(tmp_path) -> None:
    """The person's agent is not a process the harness can reclaim; the lease
    is the only hold on it (docs §12.1.4)."""

    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment", assignment_id="assignment-1", worker_goal="serve",
        tasks=[], session_scope="task", max_workers=2, sealed=False, lease_ttl_seconds=1,
    )
    runtime.add_task(_spec("t1"), protocol_documents={"correction": "R"},
                     context_documents={"payload-t1": "p"})
    gone = _claim(runtime, "conv-1", "conversational")
    assert gone["task"]["task_id"] == "t1"
    time.sleep(1.2)
    # The expired lease is reclaimed on the next read; a second agent gets it.
    again = _claim(runtime, "conv-2", "conversational")
    assert again["task"]["task_id"] == "t1"
    assert again["task"]["lease_generation"] == gone["task"]["lease_generation"] + 1


def test_a_working_agent_is_not_cut_off_by_the_wait_that_was_for_finding_one(
    tmp_path,
) -> None:
    """`local_agent_timeout_seconds` budgets being unclaimed, not the answer.

    It used to be one wall clock started at enqueue, so an agent that had the
    task and was renewing its lease every control command was withdrawn
    mid-answer anyway -- the live test only finished because the knob had been
    raised to its 3600s ceiling.
    """

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=1, label="t"
    )
    stop = threading.Event()
    served: list[str] = []

    def host() -> None:
        # Claims immediately, then spends 2.5s "answering" -- more than twice
        # the budget, and silent throughout. Under the old single wall clock
        # the call is withdrawn while this is still going.
        _scripted_host(queue.root, stop=stop, served=served, think_seconds=2.5)

    thread = threading.Thread(target=host, daemon=True)
    thread.start()
    try:
        result, _, _, _, _ = queue.run_task(
            session_type="correction", input_hash="sha256:x", validator_id="accept",
            metadata={}, protocol_text="R", payload_text="p", retrieval_mode="none",
            max_repairs=0,
        )
    finally:
        stop.set()
        thread.join(timeout=20)
    assert result.content == "good" and served == ["call-0001"]
    queue.close()


def test_the_wait_starts_over_when_an_agent_abandons_the_task(
    tmp_path,
) -> None:
    """A lapsed lease is the same as never having been claimed -- and refills.

    Not draining while somebody holds the task must not become "wait forever
    once anybody touched it": the row keeps naming its worker until some write
    sweeps it, so the lease deadline is what counts. And when it does lapse the
    budget starts from full, because somebody having turned up is evidence the
    run is attended; `MAX_CLAIMS_PER_TASK` is what ends a flapping agent, not
    an accumulated stopwatch.
    """

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=2, label="t"
    )
    queue.runtime.lease_ttl_seconds = 1
    claimed = threading.Event()

    def claim_then_vanish() -> None:
        deadline = time.time() + 20
        while time.time() < deadline:
            status = queue.runtime.rehydrate(
                assignment_id=queue.assignment_id, worker_id="conv-1"
            )
            try:
                got = queue.runtime.next_task(
                    assignment_id=queue.assignment_id, worker_id="conv-1",
                    request_id=f"claim-{status['control_generation']}",
                    expected_control_generation=status["control_generation"],
                    worker_kind="conversational",
                )
            except StaleControlGenerationError:
                # The queue added the task between the read and the claim.
                continue
            if got["status"] == "task":
                claimed.set()
                return
            time.sleep(0.05)

    thread = threading.Thread(target=claim_then_vanish, daemon=True)
    thread.start()
    with pytest.raises(AgentRuntimeCallError, match="agent-join"):
        queue.run_task(
            session_type="correction", input_hash="sha256:x", validator_id="accept",
            metadata={}, protocol_text="R", payload_text="p", retrieval_mode="none",
            max_repairs=0,
        )
    thread.join(timeout=20)
    assert claimed.is_set(), "the task was claimed before it was abandoned"
    queue.close()


def test_the_assignment_id_is_the_directory_name(tmp_path) -> None:
    """Two independent `conv-<hex>` for one queue read as a mismatch."""

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    assert queue.root.name == queue.assignment_id
    queue.close()


def test_agent_join_finds_the_waiting_run_without_being_told_where(
    tmp_path, monkeypatch, capsys
) -> None:
    """The announcement is one reporter line from a worker thread.

    Wherever nothing bound a reporter it goes nowhere, and the directory
    carries a fresh random id every run -- so being unable to find the tree
    again was the difference between joining and not.
    """

    from finesub.llm.agent import agent_join

    monkeypatch.setattr(
        agent_join, "conversational_assignment_parent", lambda: tmp_path
    )
    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )

    assert join_main([]) == 0
    assert queue.assignment_id in capsys.readouterr().out
    queue.close()


def test_agent_join_names_the_candidates_when_more_than_one_run_waits(
    tmp_path, monkeypatch, capsys
) -> None:
    from finesub.llm.agent import agent_join

    monkeypatch.setattr(
        agent_join, "conversational_assignment_parent", lambda: tmp_path
    )
    first = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    second = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )

    assert join_main([]) == 2
    error = capsys.readouterr().err
    assert first.assignment_id in error and second.assignment_id in error
    first.close()
    second.close()


def test_agent_join_says_so_when_nothing_is_waiting(tmp_path, monkeypatch, capsys) -> None:
    """A sealed queue is not a run to join, and neither is an empty parent."""

    from finesub.llm.agent import agent_join

    monkeypatch.setattr(
        agent_join, "conversational_assignment_parent", lambda: tmp_path
    )
    assert join_main([]) == 2
    assert "no run is waiting" in capsys.readouterr().err

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    queue.runtime.seal()
    assert join_main([]) == 2
    assert "no run is waiting" in capsys.readouterr().err


def test_a_live_queue_keeps_agent_clean_from_deleting_it(tmp_path) -> None:
    """The tree now sits where `finesub agent-clean` deletes (docs §12.1.4).

    Every other agent transport publishes an activity lease from the CLI
    invocation it wraps; this one owns no process, so it has to publish its
    own. Without it, a person tidying up old evidence mid-run takes the live
    assignment with it and cuts off their own agent -- and until the tree was
    moved under the episode parent, only the misplacement was hiding that.
    """

    from finesub_bootstrap.locks import LockUnavailable, holding_activity_barrier

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    with pytest.raises(LockUnavailable):
        with holding_activity_barrier(tmp_path, timeout=0):
            pass

    queue.close()

    # Closed: nothing left to protect, and cleanup must not stay blocked.
    with holding_activity_barrier(tmp_path, timeout=0):
        pass


def test_lint_reports_the_same_errors_as_submit_without_spending_anything(
    tmp_path,
) -> None:
    """The one check that does not rely on the other side policing itself.

    A truncated or short-covering answer is invisible to the agent that wrote
    it and obvious to the validator. Finding out at submit time costs a repair
    round and, in a conversation, another full rewrite -- so `lint` runs the
    same validator and changes nothing: no budget, no repair count, no cached
    verdict, no status change.
    """

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    queue.runtime.add_task(_spec("t1", validator="test-reject-bad"),
                           protocol_documents={"correction": "R"},
                           context_documents={"payload-t1": "p"})
    common = ["--root", str(queue.root), "--assignment", queue.assignment_id]
    status = _run_cli([*common, "status", "--worker", "conv-1"])
    task = _run_cli([*common, "next-task", "--worker", "conv-1", "--request-id", "c1",
                     "--control-generation", str(status["control_generation"])])["task"]
    leased = ["--worker", "conv-1", "--task", "t1",
              "--lease-generation", str(task["lease_generation"])]

    body = tmp_path / "draft.txt"
    body.write_text("bad", encoding="utf-8")
    # A lint takes no `--request-id`: nothing is recorded, so nothing can
    # replay, and repeating one is just repeating it.
    verdict = _run_cli([*common, "lint", *leased,
                        "--text-file", str(body)])
    assert verdict["verdict"] == "repairable"
    assert verdict["validation_errors"] == ["say good, not bad"]

    before = queue.runtime.task_record(assignment_id=queue.assignment_id, task_id="t1")
    assert before["status"] == "leased", "a rejected lint is not a repair round"
    assert before["validation_errors"] == []
    # Lint again: still nothing spent.
    _run_cli([*common, "lint", *leased, "--text-file", str(body)])
    after = queue.runtime.task_record(assignment_id=queue.assignment_id, task_id="t1")
    assert after["submit_count"] == before["submit_count"] == 0
    assert after["repair_attempts"] == before["repair_attempts"] == 0

    # Fixed answer lints clean and then submits clean.
    body.write_text("good", encoding="utf-8")
    assert _run_cli([*common, "lint", *leased,
                     "--text-file", str(body)])["verdict"] == "accepted"
    _run_cli([*common, "submit", *leased, "--request-id", "s1",
              "--input-hash", "sha256:t1", "--text-file", str(body)])
    assert queue.runtime.task_record(
        assignment_id=queue.assignment_id, task_id="t1"
    )["status"] == "accepted"
    queue.close()


def test_a_text_file_answer_needs_no_json_quoting(tmp_path) -> None:
    """`--text-file` and a JSON-quoted `--json-file` submit the same answer.

    Hand-quoting tens of KB of CSV into one JSON string is a step that can
    only go wrong, and it did in the first live test.
    """

    from finesub.llm.agent import agent_task_control

    answer = 'a,b\n"x|y",2\n'
    text_file = tmp_path / "answer.csv"
    text_file.write_text(answer, encoding="utf-8")
    json_file = tmp_path / "answer.json"
    json_file.write_text(json.dumps(answer), encoding="utf-8")

    parse = agent_task_control._parser().parse_args
    base = ["--assignment", "a", "submit", "--worker", "w", "--request-id", "r",
            "--task", "t", "--lease-generation", "1", "--input-hash", "h"]
    from_text = agent_task_control._json_input(parse([*base, "--text-file", str(text_file)]))
    from_json = agent_task_control._json_input(parse([*base, "--json-file", str(json_file)]))
    assert from_text == from_json == answer

    with pytest.raises(ValueError, match="not both"):
        agent_task_control._json_input(
            parse([*base, "--text-file", str(text_file), "--json-file", str(json_file)])
        )


def test_a_clean_finish_leaves_a_tombstone_and_the_exchange(tmp_path) -> None:
    """Success used to delete the tree out from under a parked worker.

    `await-next-task` polls for up to 28 minutes and the seal grace is five
    seconds, so a clean session is *certain* to have somebody still waiting:
    they got a vanished working directory instead of `assignment_complete`.
    The run's text still has to go -- what stays is the sealed control plane
    and the exchange, which the run's artifacts then take.
    """

    from finesub.llm.agent.agent_session_host import file_conversational_evidence

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=10, label="t"
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: _scripted_host(queue.root, stop=stop), daemon=True
    )
    thread.start()
    queue.run_task(
        session_type="correction", input_hash="sha256:x", validator_id="accept",
        metadata={}, protocol_text="PROTOCOL", payload_text="SUBTITLE BODY",
        retrieval_mode="none", max_repairs=0,
    )
    root = queue.root
    forgotten: list[str] = []
    original = queue.runtime.forget_drafts
    queue.runtime.forget_drafts = lambda **kwargs: (  # type: ignore[method-assign]
        forgotten.append(kwargs["assignment_id"]) or original(**kwargs)
    )
    queue.close()
    stop.set()
    thread.join(timeout=20)

    # The body is gone -- including any rejected draft still in durable state.
    assert forgotten == [queue.assignment_id]
    assert not (root / "contexts").exists() and not (root / "tasks").exists()
    # ...but a parked worker still gets an answer instead of an empty tree.
    runtime = AgentTaskRuntime(root)
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    assert runtime.status(
        assignment_id=index["assignment_id"], worker_id="conv-1"
    )["status"] == "assignment_complete"

    artifacts = tmp_path / "artifacts"
    moved = file_conversational_evidence(artifacts, _OneQueueRegistry(queue))
    assert moved, "the accepted exchange is filed with the run's other records"
    kept = artifacts / "agent-conversational" / queue.assignment_id / "call-0001"
    assert (kept / "context.md").read_text(encoding="utf-8").strip() == "SUBTITLE BODY"
    assert (kept / "answer.txt").read_text(encoding="utf-8") == "good"
    assert json.loads((kept / "summary.json").read_text(encoding="utf-8"))[
        "accepted_by"
    ] == "conv-1"


class _OneQueueRegistry:
    """What `close()` would have handed the run, for one queue."""

    def __init__(self, queue) -> None:
        self._queue = queue

    def evidence_roots(self):
        return [self._queue.evidence_root]


def test_the_tombstone_does_not_keep_the_window_it_just_dropped(tmp_path) -> None:
    """A repaired task carries the whole rejected window in durable state.

    `last_candidate` is what a session ending without an accepted answer
    reports as its last try, and a tool session's audit bundle reads it as the
    record of the repair round -- so it stays while the tree stands. But
    durable state is exactly what a conversational queue keeps as its
    tombstone, and keeping the control plane must not mean keeping the text
    the clean-up exists to drop.
    """

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=5, label="t"
    )
    queue.runtime.add_task(_spec("t1", validator="test-reject-bad"),
                           protocol_documents={"correction": "R"},
                           context_documents={"payload-t1": "SUBTITLE BODY IN CONTEXT"})
    common = ["--root", str(queue.root), "--assignment", queue.assignment_id]
    status = _run_cli([*common, "status", "--worker", "conv-1"])
    task = _run_cli([*common, "next-task", "--worker", "conv-1", "--request-id", "c1",
                     "--control-generation", str(status["control_generation"])])["task"]
    leased = ["--worker", "conv-1", "--task", "t1",
              "--lease-generation", str(task["lease_generation"])]
    body = tmp_path / "answer.json"
    draft = "bad REJECTED DRAFT SUBTITLE BODY"
    for request_id, answer in (("s1", draft), ("s2", "good")):
        body.write_text(json.dumps(answer), encoding="utf-8")
        _run_cli([*common, "submit", *leased, "--request-id", request_id,
                  "--input-hash", "sha256:t1", "--json-file", str(body)])

    record = queue.runtime.task_record(assignment_id=queue.assignment_id, task_id="t1")
    assert record["status"] == "accepted"
    assert record["last_candidate"] == draft, "kept while the tree stands"

    assert queue.runtime.forget_drafts(assignment_id=queue.assignment_id) == 1
    assert not queue.runtime.task_record(
        assignment_id=queue.assignment_id, task_id="t1"
    )["last_candidate"]
    # Not just the current row. State is one append-only snapshot per change
    # and a tail of the old ones is kept for forensics, so the snapshot from
    # the rejected submit held the draft in full -- and `control/state/` is
    # precisely what the tombstone keeps. Grepping the tree is the assertion
    # that survives any change to how state is stored.
    assert not _files_containing(queue.root, "REJECTED DRAFT")
    # ...and the grep is looking in the right place: the payload is still there.
    assert _files_containing(queue.root, "SUBTITLE BODY IN CONTEXT")
    # Idempotent: a second clean-up has nothing left to drop.
    assert queue.runtime.forget_drafts(assignment_id=queue.assignment_id) == 0
    queue.close()


def test_only_a_correction_task_gets_the_conversational_effort_note(tmp_path) -> None:
    """The note belongs to one session type on one road.

    Not in the shared output contract: every backend answers to that, and the
    same sentence measurably raised a REST model's thinking while buying
    nothing there -- hand-counting characters is a cost only an agent with its
    own tools can incur. Not in the worker bootstrap either: that is
    task-agnostic, it says how to take and submit *a* task.
    """

    from finesub.llm.prompts import conversational_correction_effort

    note = conversational_correction_effort()
    assert "char_count" in note and "差不多就行" in note
    # Not baked into what every backend sees.
    from finesub.llm.prompt_compose import compose_correction_system
    from finesub.llm.routing.profiles import resolve_profile

    assert "差不多就行" not in compose_correction_system(
        resolve_profile("text", "local", "quality")
    )
    # Nor into the protocol handed to any worker regardless of task.
    from finesub.llm.prompts import agent_worker_bootstrap

    assert "差不多就行" not in agent_worker_bootstrap(
        assignment_root="r", assignment_id="a", worker_id="w",
        task_command="finesub agent-task", watch_minutes=28, durable_status="{}",
    )


def test_the_scope_files_the_evidence_even_when_no_front_end_does(
    tmp_path, monkeypatch
) -> None:
    """A stage entered on its own still leaves its exchanges behind.

    Filing used to hang off the two `correction_translation` entrypoints, so
    `run_reference_knowledge_update` and a directly called
    `execute_correction_windows` -- both of which open and close their own
    scope through `within_agent_session_scope` -- dropped the registry with
    the exchanges still staged inside it. The scope owns it now; whoever knows
    the artifact directory only has to name it.
    """

    _no_api(monkeypatch)
    warnings: list[tuple[str, str]] = []
    from finesub.llm.agent import agent_session_host as host_module
    from finesub.llm.agent.agent_session_host import set_run_evidence_destination

    class Reporter:
        @staticmethod
        def warning(code, message, **kwargs):
            warnings.append((code, message))

        @staticmethod
        def debug(*a, **k):
            pass

    monkeypatch.setattr(host_module, "current_reporter", lambda: Reporter())
    client = _client(tmp_path)
    stop = threading.Event()

    def run_host() -> None:
        while not warnings and not stop.is_set():
            time.sleep(0.05)
        root = Path(warnings[0][1].split('finesub agent-join "')[1].rstrip('"'))
        _scripted_host(root, stop=stop)

    thread = threading.Thread(target=run_host, daemon=True)
    thread.start()
    artifacts = tmp_path / "task-artifacts"
    with agent_session_scope():
        set_run_evidence_destination(artifacts)
        _complete(client, "window one")
    stop.set()
    thread.join(timeout=60)

    filed = sorted((artifacts / "agent-conversational").glob("*/call-0001"))
    assert len(filed) == 1, "the accepted exchange is filed when the scope closes"
    assert (filed[0] / "answer.txt").read_text(encoding="utf-8") == "good"


def test_a_queue_that_fails_to_start_gives_its_activity_lease_back(
    tmp_path, monkeypatch
) -> None:
    """The lease is an OS lock: leaked here, it is held until the process ends.

    It is what tells `agent-clean` and `relocate` that a run is live, so a
    queue that never came up must not keep claiming to be one.
    """

    from finesub_bootstrap.locks import activity_is_idle
    from finesub.llm.agent import agent_session_host as host_module

    def refuse(*args, **kwargs):
        raise AgentTaskRuntimeError("no assignment for you")

    monkeypatch.setattr(
        host_module.AgentTaskRuntime, "start_assignment", staticmethod(refuse)
    )
    with pytest.raises(AgentTaskRuntimeError):
        ConversationalQueue(
            parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=10
        )
    assert activity_is_idle(tmp_path), "a queue that never started is not a live run"


def test_closing_gives_the_lease_back_even_when_teardown_is_interrupted(
    tmp_path,
) -> None:
    """`close()` is the only place that releases it, so it may not skip it.

    A Ctrl+C lands where it lands, and teardown -- which sleeps out a seal
    grace -- is a likely place. The lease must not be what survives it.
    """

    from finesub_bootstrap.locks import activity_is_idle

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={}, call_timeout_seconds=60, wait_seconds=10
    )
    assert not activity_is_idle(tmp_path)

    def interrupt() -> None:
        raise KeyboardInterrupt

    queue.runtime.seal = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        queue.close()
    assert activity_is_idle(tmp_path)


def _claim_once(queue, worker: str) -> bool:
    """Take the task and go quiet, the way a dropped agent session does."""

    deadline = time.time() + 20
    while time.time() < deadline:
        status = queue.runtime.rehydrate(
            assignment_id=queue.assignment_id, worker_id=worker
        )
        try:
            got = queue.runtime.next_task(
                assignment_id=queue.assignment_id, worker_id=worker,
                request_id=f"claim-{worker}-{status['control_generation']}",
                expected_control_generation=status["control_generation"],
                worker_kind="conversational",
            )
        except StaleControlGenerationError:
            continue
        if got["status"] == "task":
            return True
        time.sleep(0.05)
    return False


def test_rejoining_after_a_dropped_session_gets_the_full_wait_back(tmp_path) -> None:
    """The coffee-break case: somebody was here, so keep holding the task.

    The budget used to carry across gaps, so a person whose agent session died
    while they were away could come back to a task that had been withdrawn --
    even though an agent had demonstrably been on it minutes earlier. A claim
    is evidence the run is attended, so it refills the wait.
    """

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={},
        call_timeout_seconds=60, wait_seconds=1.5, label="t",
    )
    queue.runtime.lease_ttl_seconds = 1
    stop = threading.Event()
    served: list[str] = []

    def drop_then_rejoin() -> None:
        # Burn most of the first budget, take the task, vanish; the lease
        # lapses, and only then does a real agent turn up. Without the refill
        # the leftovers of that first wait are all it would get.
        time.sleep(1.0)
        _claim_once(queue, "conv-gone")
        time.sleep(1.4)
        _scripted_host(queue.root, stop=stop, served=served, worker="conv-2")

    thread = threading.Thread(target=drop_then_rejoin, daemon=True)
    thread.start()
    try:
        result, _, _, _, _ = queue.run_task(
            session_type="correction", input_hash="sha256:x", validator_id="accept",
            metadata={}, protocol_text="R", payload_text="p", retrieval_mode="none",
            max_repairs=0,
        )
    finally:
        stop.set()
        thread.join(timeout=20)
    assert result.content == "good" and served == ["call-0001"]
    queue.close()


def test_an_agent_that_keeps_taking_the_task_and_leaving_is_given_up_on(
    tmp_path,
) -> None:
    """What refilling the wait leaves uncovered, counted instead of timed.

    Every claim resets the clock, so seconds alone can never end this: the
    agent takes the task, goes quiet, takes it again. `MAX_CLAIMS_PER_TASK`
    is the bound, and it says what actually went wrong.
    """

    from finesub.llm.agent.agent_session_host import MAX_CLAIMS_PER_TASK

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={},
        call_timeout_seconds=60, wait_seconds=30, label="t",
    )
    queue.runtime.lease_ttl_seconds = 1
    stop = threading.Event()

    def flap() -> None:
        while not stop.is_set():
            _claim_once(queue, "conv-1")
            time.sleep(1.2)

    thread = threading.Thread(target=flap, daemon=True)
    thread.start()
    try:
        with pytest.raises(AgentRuntimeCallError, match="never finished"):
            queue.run_task(
                session_type="correction", input_hash="sha256:x", validator_id="accept",
                metadata={}, protocol_text="R", payload_text="p", retrieval_mode="none",
                max_repairs=0,
            )
    finally:
        stop.set()
        thread.join(timeout=20)
    assert MAX_CLAIMS_PER_TASK == 3
    queue.close()


def test_every_assignment_holds_a_task_longer_than_it_lets_one_run(
    tmp_path,
) -> None:
    """One derivation for all three shapes, so the guard can never fire.

    Our deadline has to be strictly later than the agent's, or a punctual
    answer dies at `submit` against a lease that expired on the way there.
    """

    from finesub.llm.agent.agent_task_runtime import (
        LEASE_MARGIN_SECONDS,
        lease_ttl_for,
    )

    from finesub.llm.agent.agent_task_runtime import DEFAULT_LEASE_TTL_SECONDS
    from finesub.llm.routing.execution_policy import ExecutionSettings

    # The module fallback is what the derivation yields at the shipped
    # default; drift between them would make a settings-less runtime disagree
    # with every runtime this process builds.
    assert lease_ttl_for(
        ExecutionSettings().local_agent_timeout_seconds
    ) == DEFAULT_LEASE_TTL_SECONDS
    assert lease_ttl_for(4 * 3600) == 4 * 3600 + LEASE_MARGIN_SECONDS

    queue = ConversationalQueue(
        parent=tmp_path, activity_root=tmp_path, execution_identity={},
        call_timeout_seconds=900, label="t",
    )
    try:
        assert queue.runtime.lease_ttl_seconds == 900 + LEASE_MARGIN_SECONDS
        # And the wait is not what sets it -- that is a fixed hang guard.
        assert queue.wait_seconds == CONVERSATIONAL_JOIN_WAIT_SECONDS
    finally:
        queue.close()
