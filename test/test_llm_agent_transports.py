from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time

import pytest

from finesub.llm.agent.agent_task_runtime import (
    CONVERSATIONAL_WATCH_SECONDS,
    MAX_WATCH_SECONDS,
    AgentTaskRuntime,
    AgentTaskSpec,
    StaleControlGenerationError,
    ValidationResult,
)
from finesub.llm.agent.agent_task_runtime import AssignmentConflictError
from finesub.llm.agent.agent_transports import (
    CONVERSATION_WATCH_MARGIN_SECONDS,
    AssignmentHeadlessWorker,
    HeadlessTaskWorker,
    conversational_bootstrap,
)
from finesub.llm.agent.local_agent import LocalAgentUnavailableError


@dataclass
class _Result:
    content: str
    conversation_handle: str = ""
    turn_identity: str = ""
    execution_attempt: dict = field(default_factory=dict)


class _FakeDriver:
    conversation_ttl_seconds = 0.0

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def run(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _Result(self.outputs.pop(0))


class _ReusableFakeDriver(_FakeDriver):
    def run(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        handle = kwargs.get("conversation_handle") or "conversation-1"
        return _Result(
            self.outputs.pop(0),
            conversation_handle=handle,
            turn_identity=f"sha256:turn-{len(self.calls)}",
        )


def _runtime(
    tmp_path,
    *,
    validator=None,
    task_count=1,
    session_scope="task",
    retrieval_mode="none",
    clock=None,
):
    validators = {"strict": validator} if validator else None
    tasks = [
        AgentTaskSpec(
            task_id=f"task-{index}",
            session_type="research",
            input_hash=f"sha256:{index}",
            goal=f"answer {index}",
            dependencies=(() if index == 1 else (f"task-{index - 1}",)),
            validator_id="strict" if validator else "accept",
            protocol_key="research",
            context_key="run",
            retrieval_mode=retrieval_mode,
        )
        for index in range(1, task_count + 1)
    ]
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="finish every task",
        tasks=tasks,
        session_scope=session_scope,
        bootstrap_text="trusted bootstrap",
        protocol_documents={"research": "stable research protocol"},
        context_documents={"run": "durable run context"},
        validators=validators,
        **({"clock": clock} if clock else {}),
    )


def test_conversational_bootstrap_names_cwd_index_and_completion(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    prompt = conversational_bootstrap(
        runtime, assignment_id="assignment-1", worker_id="worker-1"
    )

    assert str(runtime.root) in prompt
    assert "control/index.json" in prompt
    assert "await-next-task" in prompt
    assert "28 分钟" in prompt
    assert "--worker worker-1" in prompt
    assert "assignment_complete" in prompt


def test_task_scoped_worker_full_replays_and_repairs(tmp_path) -> None:
    def validator(candidate, _manifest):
        if candidate != "good":
            return ValidationResult.repairable("must say good")
        return ValidationResult.accepted({"answer": candidate})

    runtime = _runtime(tmp_path, validator=validator)
    driver = _FakeDriver(["bad", "good"])
    worker = HeadlessTaskWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    result = worker.run_one()
    assert result["status"] == "assignment_complete"
    assert len(driver.calls) == 2
    first_messages, first_kwargs = driver.calls[0]
    assert "trusted bootstrap" in first_messages[0]["content"]
    assert "stable research protocol" in first_messages[0]["content"]
    assert "durable run context" in first_messages[1]["content"]
    assert first_kwargs["previous_output"] == ""
    assert driver.calls[1][1]["previous_output"] == "bad"
    assert driver.calls[1][1]["validation_errors"] == ["must say good"]


def test_task_scoped_worker_runs_dependency_chain_as_fresh_episodes(tmp_path) -> None:
    runtime = _runtime(tmp_path, task_count=2)
    driver = _FakeDriver(["first", "second"])
    worker = HeadlessTaskWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    result = worker.run_until_idle()
    assert result["status"] == "assignment_complete"
    assert len(driver.calls) == 2
    assert all("session_scope=task" in call[0][0]["content"] for call in driver.calls)


def test_assignment_worker_reuses_handle_and_persists_lineage(tmp_path) -> None:
    runtime = _runtime(tmp_path, task_count=2, session_scope="assignment")
    driver = _ReusableFakeDriver(["first", "second"])
    worker = AssignmentHeadlessWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    result = worker.run_until_idle()

    assert result["status"] == "assignment_complete"
    assert len(driver.calls) == 2
    assert driver.calls[0][1]["session_scope"] == "assignment"
    assert driver.calls[0][1]["conversation_handle"] == ""
    assert "trusted bootstrap" in driver.calls[0][0][0]["content"]
    assert driver.calls[1][1]["conversation_handle"] == "conversation-1"
    assert len(driver.calls[1][0]) == 1
    lineage = runtime.conversation_state(
        assignment_id="assignment-1", worker_id="worker-1"
    )
    assert lineage["conversation_handle"] == "conversation-1"
    assert lineage["turn_generation"] == 2
    assert lineage["parent_turn_identity"] == "sha256:turn-2"
    assert lineage["harness_ack_digest"].startswith("sha256:")


def test_assignment_worker_rebuilds_a_lost_conversation_once(tmp_path) -> None:
    class _LosesItsSession(_ReusableFakeDriver):
        def run(self, messages, **kwargs):
            if kwargs.get("conversation_handle") == "conversation-1":
                self.calls.append((messages, kwargs))
                raise LocalAgentUnavailableError("recorded session was pruned")
            return super().run(messages, **kwargs)

    runtime = _runtime(tmp_path, task_count=2, session_scope="assignment")
    driver = _LosesItsSession(["first", "second"])
    worker = AssignmentHeadlessWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    result = worker.run_until_idle()

    assert result["status"] == "assignment_complete"
    assert [call[1]["conversation_handle"] for call in driver.calls] == [
        "",
        "conversation-1",
        "",
    ]
    assert "trusted bootstrap" in driver.calls[2][0][0]["content"]
    lineage = runtime.conversation_state(
        assignment_id="assignment-1", worker_id="worker-1"
    )
    assert lineage["conversation_epoch"] == 2
    assert lineage["turn_generation"] == 1
    assert len(lineage["resets"]) == 1
    assert "pruned" in lineage["resets"][0]["reason"]


def test_assignment_worker_retires_a_conversation_past_its_ttl(tmp_path) -> None:
    now = [1000.0]

    class _ShortLived(_ReusableFakeDriver):
        conversation_ttl_seconds = 60.0

    runtime = _runtime(
        tmp_path, task_count=2, session_scope="assignment", clock=lambda: now[0]
    )
    driver = _ShortLived(["first", "second"])
    worker = AssignmentHeadlessWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    assert worker.run_one()["status"] in {"ready", "task", "accepted"}
    now[0] += 61.0
    assert worker.run_until_idle()["status"] == "assignment_complete"

    assert [call[1]["conversation_handle"] for call in driver.calls] == ["", ""]
    lineage = runtime.conversation_state(
        assignment_id="assignment-1", worker_id="worker-1"
    )
    assert lineage["conversation_epoch"] == 2
    assert "reuse window" in lineage["resets"][0]["reason"]


def test_assignment_worker_does_not_rebuild_twice_for_the_same_turn(tmp_path) -> None:
    class _AlwaysFails(_ReusableFakeDriver):
        def run(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                return _Result(
                    self.outputs.pop(0),
                    conversation_handle="conversation-1",
                    turn_identity="sha256:turn-1",
                )
            raise LocalAgentUnavailableError("the driver itself is down")

    runtime = _runtime(tmp_path, task_count=2, session_scope="assignment")
    driver = _AlwaysFails(["first", "second"])
    worker = AssignmentHeadlessWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    worker.run_one()
    with pytest.raises(LocalAgentUnavailableError):
        worker.run_one()
    assert len(driver.calls) == 3


def test_native_turns_are_booked_against_the_task_even_when_empty(tmp_path) -> None:
    class _NativeDriver(_FakeDriver):
        def run(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return _Result(
                self.outputs.pop(0),
                execution_attempt={
                    "duration_ms": 2000,
                    "search_events": [{"query": "q", "urls": ["https://example.test"]}]
                    if len(self.calls) == 1
                    else [],
                },
            )

    runtime = _runtime(tmp_path, task_count=2, retrieval_mode="native")
    driver = _NativeDriver(["first", "second"])
    worker = HeadlessTaskWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    assert worker.run_until_idle()["status"] == "assignment_complete"
    assert all(call[1]["native_search"] for call in driver.calls)
    for task_id, searches in (("task-1", 1), ("task-2", 0)):
        recorded = sorted((runtime.root / "tasks" / task_id / "retrieval").glob("*.json"))
        assert len(recorded) == 1
        payload = json.loads(recorded[0].read_text(encoding="utf-8"))
        assert len(payload["searches"]) == searches


def test_headless_worker_waits_without_model_calls_until_external_dependency(
    tmp_path,
) -> None:
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="finish",
        tasks=[
            AgentTaskSpec(
                task_id="api",
                session_type="api",
                input_hash="sha256:api",
                goal="external",
                executor="external",
            ),
            AgentTaskSpec(
                task_id="agent",
                session_type="research",
                input_hash="sha256:agent",
                goal="answer",
                dependencies=("api",),
            ),
        ],
    )
    driver = _FakeDriver(["done"])
    worker = HeadlessTaskWorker(
        runtime,
        driver,  # type: ignore[arg-type]
        assignment_id="assignment-1",
        worker_id="worker-1",
    )

    def complete_external() -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = runtime.status(
                assignment_id="assignment-1", worker_id="worker-1"
            )
            if status.get("wait_token"):
                runtime.accept_external_task(
                    assignment_id="assignment-1",
                    task_id="api",
                    request_id="external-result",
                    expected_control_generation=status["control_generation"],
                    input_hash="sha256:api",
                    artifact={"value": 1},
                )
                return
            time.sleep(0.01)
        raise AssertionError("worker did not register its waiter")

    thread = threading.Thread(target=complete_external)
    thread.start()
    result = worker.run_until_complete(max_wait_seconds=0.5)
    thread.join(timeout=2)

    assert result["status"] == "assignment_complete"
    assert len(driver.calls) == 1


def test_agent_session_mode_resolves_per_cell_with_difficulty_fallback() -> None:
    from finesub.llm.routing.config import role_config_for
    from finesub.llm.routing.model_routes import DEFAULT_AGENT_SESSION_MODE

    # Nothing is pinned in the shipped tables yet, so every cell answers with
    # the default rather than raising or returning empty.
    for task_group in ("correction-mm", "research"):
        for difficulty in ("quality", "intermediate", "efficiency"):
            cell = role_config_for(task_group, difficulty)
            assert cell.agent_session_mode == DEFAULT_AGENT_SESSION_MODE


def test_session_scope_for_mode_maps_or_refuses() -> None:
    import pytest as _pytest

    from finesub.llm.agent.agent_transports import session_scope_for_mode

    assert session_scope_for_mode("per-session") == "task"
    assert session_scope_for_mode("resume") == "assignment"
    with _pytest.raises(NotImplementedError, match="pseudo-conversational"):
        session_scope_for_mode("pseudo-conversational")
    with _pytest.raises(ValueError, match="Unknown agent session mode"):
        session_scope_for_mode("nonsense")


def test_a_lost_claim_race_is_retried_not_raised(tmp_path) -> None:
    """Claiming is a compare-and-swap, so losing it is a designed outcome.

    A second worker landing between this one's rehydrate and its claim moves
    the control generation on. Letting that escape killed the worker over a
    race the protocol exists to absorb.
    """

    runtime = _runtime(tmp_path)
    worker = HeadlessTaskWorker(
        runtime,
        _FakeDriver(["done"]),
        assignment_id="assignment-1",
        worker_id="worker-1",
    )
    original = runtime.next_task
    attempts: list[int] = []

    def lose_the_first_race(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise StaleControlGenerationError("someone else moved first")
        return original(**kwargs)

    runtime.next_task = lose_the_first_race  # type: ignore[method-assign]
    result = worker.run_one()

    assert len(attempts) == 2
    assert result["status"] == "assignment_complete"


def test_exhausted_repairs_hand_the_task_back(tmp_path) -> None:
    """Giving up must release the lease, not sit on it.

    Left leased, the task stayed `repairing` until the TTL ran out -- half an
    hour of an assignment with nobody working on it.
    """

    runtime = _runtime(
        tmp_path,
        validator=lambda _candidate, _manifest: ValidationResult.repairable(
            "still wrong"
        ),
    )
    worker = HeadlessTaskWorker(
        runtime,
        _FakeDriver(["a", "b", "c"]),
        assignment_id="assignment-1",
        worker_id="worker-1",
        max_repair_attempts=2,
    )

    result = worker.run_one()

    assert result["status"] == "repair_exhausted"
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    assert index["active_tasks"] == []
    assert index["next_action"] == "claim_ready"


def test_watch_length_follows_the_drivers_conversation_window(tmp_path) -> None:
    """Park no longer than the conversation being protected stays warm.

    Only the reusable worker has anything to protect: the one-shot baseline
    replays in full on the far side of any wait, so it keeps the conversational
    bound. A driver that does not know its window declares 0 and gets the same.
    """

    runtime = _runtime(tmp_path, session_scope="assignment")
    driver = _ReusableFakeDriver(["done"])
    driver.conversation_ttl_seconds = 3600.0
    reusable = AssignmentHeadlessWorker(
        runtime, driver, assignment_id="assignment-1", worker_id="worker-1"
    )
    assert reusable.watch_seconds() == 3600.0 - CONVERSATION_WATCH_MARGIN_SECONDS
    assert reusable.watch_seconds() <= MAX_WATCH_SECONDS

    driver.conversation_ttl_seconds = 0.0
    assert reusable.watch_seconds() == CONVERSATIONAL_WATCH_SECONDS

    driver.conversation_ttl_seconds = 3600.0
    one_shot = HeadlessTaskWorker(
        runtime, driver, assignment_id="assignment-1", worker_id="worker-1"
    )
    assert one_shot.watch_seconds() == CONVERSATIONAL_WATCH_SECONDS


def test_the_bootstrap_prompt_does_not_ask_for_a_keepalive(tmp_path) -> None:
    """An Agent cannot honour a timer, so it is never told to send one.

    Between two `finesub agent-task` calls nothing of the Agent's is running,
    and mid-turn it cannot execute code -- a keepalive it was instructed to
    send would be missed exactly when the task ran long.
    """

    runtime = _runtime(tmp_path)
    prompt = conversational_bootstrap(
        runtime, assignment_id="assignment-1", worker_id="worker-1"
    )

    assert "heartbeat" not in prompt.lower()
    assert "assignment_failed" in prompt


def test_a_requeued_blocked_task_is_picked_back_up(tmp_path) -> None:
    """`blocked` with budget left put the task back on the queue.

    Returning it to the caller meant only a conversational Agent ever made the
    fresh attempt; a headless worker walked away from work still sitting there.
    """

    attempts: list[int] = []

    def blocked_until_the_third_try(_candidate, _manifest):
        attempts.append(1)
        if len(attempts) < 3:
            return ValidationResult.blocked("not yet")
        return ValidationResult.accepted({"ok": True})

    runtime = _runtime(tmp_path, validator=blocked_until_the_third_try)
    worker = HeadlessTaskWorker(
        runtime,
        _FakeDriver(["a", "b", "c"]),
        assignment_id="assignment-1",
        worker_id="worker-1",
        max_repair_attempts=0,
    )

    result = worker.run_until_idle()

    assert len(attempts) == 3
    assert result["status"] == "assignment_complete"


def test_exhausting_the_requeue_budget_stops_the_worker(tmp_path) -> None:
    """`failed` is the state that stops, and the loop must honour it."""

    runtime = _runtime(
        tmp_path,
        validator=lambda _candidate, _manifest: ValidationResult.blocked("never"),
    )
    worker = HeadlessTaskWorker(
        runtime,
        _FakeDriver(["a", "b", "c", "d", "e"]),
        assignment_id="assignment-1",
        worker_id="worker-1",
        max_repair_attempts=0,
    )

    result = worker.run_until_complete()

    assert result["status"] == "failed"
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    assert index["next_action"] == "assignment_failed"


def test_a_call_that_could_outlive_its_lease_is_refused_up_front(tmp_path) -> None:
    """Nothing renews while the driver runs, so the lease must cover a call.

    Configure `local_agent_timeout_seconds` past the assignment TTL and the
    task is reclaimed underneath a worker that is still working -- the output
    it paid for then dies at `submit`. This is the only place both numbers are
    visible, so it is where the pairing is checked.
    """

    from finesub.llm.agent.local_agent import AgentDriverConfig

    class _TimedDriver(_FakeDriver):
        def __init__(self, outputs, timeout_seconds):
            super().__init__(outputs)
            self.config = AgentDriverConfig(timeout_seconds=timeout_seconds)

    runtime = _runtime(tmp_path)
    runtime.lease_ttl_seconds = 900

    with pytest.raises(AssignmentConflictError, match="does not fit inside"):
        HeadlessTaskWorker(
            runtime,
            _TimedDriver(["done"], 1800),
            assignment_id="assignment-1",
            worker_id="worker-1",
        )

    # Comfortably inside the TTL is fine.
    HeadlessTaskWorker(
        runtime,
        _TimedDriver(["done"], 600),
        assignment_id="assignment-1",
        worker_id="worker-1",
    )


def test_a_long_watch_does_not_reap_its_own_waiter(tmp_path) -> None:
    """A headless worker may park longer than the conversational default.

    The waiter row is created with that default, and judging the row by it made
    the worker's own reclaim pass drop its own waiter part way through the
    first watch -- self-healing, but a wasted round trip every time.
    """

    now = [1000.0]
    # An external dependency with one agent task behind it: the worker has
    # nothing it can claim, so it parks.
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="wait for the scheduler",
        tasks=[
            AgentTaskSpec(
                task_id="api",
                session_type="research",
                input_hash="sha256:api",
                goal="external",
                executor="external",
            ),
            AgentTaskSpec(
                task_id="agent",
                session_type="research",
                input_hash="sha256:agent",
                goal="after the api task",
                dependencies=("api",),
            ),
        ],
        clock=lambda: now[0],
    )
    long_watch = float(CONVERSATIONAL_WATCH_SECONDS * 2)
    parked = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-2",
        request_id="park",
        expected_control_generation=1,
    )
    assert parked["status"] == "waiting"

    # Satisfy the dependency first, so the watch below records its bound and
    # then returns on its very first pass. The deadline inside the watch is
    # wall time, not the injected clock, so a long watch that really waited
    # would take as long as it says.
    runtime.accept_external_task(
        assignment_id="assignment-1",
        task_id="api",
        request_id="api-done",
        expected_control_generation=parked["control_generation"],
        input_hash="sha256:api",
        artifact={"result": 1},
    )
    woke = runtime.await_next_task(
        assignment_id="assignment-1",
        worker_id="worker-2",
        wait_token=parked["wait_token"],
        max_wait_seconds=long_watch,
    )
    assert woke["next_action"] == "claim_ready"

    # Past twice the conversational bound -- which is what the row was created
    # with -- but well inside twice the bound this worker actually asked for.
    now[0] += CONVERSATIONAL_WATCH_SECONDS * 2 + 640
    runtime.status(assignment_id="assignment-1")

    # The slot is still worker-2's: judged by the stale default bound it would
    # have been handed back, and a different id could take it.
    with pytest.raises(AssignmentConflictError, match="worker limit"):
        runtime.next_task(
            assignment_id="assignment-1",
            worker_id="worker-9",
            request_id="steal",
            expected_control_generation=runtime.status(
                assignment_id="assignment-1"
            )["control_generation"],
        )
