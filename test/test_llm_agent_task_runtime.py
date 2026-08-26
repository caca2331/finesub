from __future__ import annotations

import json

import pytest

from finesub.llm.agent.agent_task_runtime import (
    CONVERSATIONAL_WATCH_SECONDS,
    DEFAULT_BLOCKED_REQUEUES,
    DEFAULT_LEASE_TTL_SECONDS,
    MAX_WATCH_SECONDS,
    RETAINED_STATE_GENERATIONS,
    WAITER_ABANDON_FACTOR,
    AgentTaskRuntime,
    AgentTaskRuntimeError,
    AgentTaskSpec,
    AssignmentConflictError,
    StaleControlGenerationError,
    StaleLeaseError,
    ValidationResult,
)


def _task(
    task_id: str,
    *,
    dependencies: tuple[str, ...] = (),
    executor: str = "agent",
    validator_id: str = "accept",
) -> AgentTaskSpec:
    return AgentTaskSpec(
        task_id=task_id,
        session_type="correction",
        input_hash=f"sha256:{task_id}",
        goal=f"finish {task_id}",
        dependencies=dependencies,
        executor=executor,
        validator_id=validator_id,
        protocol_key="correction",
        context_key="run-1",
    )


def _start(tmp_path, tasks, **kwargs) -> AgentTaskRuntime:
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="finish the assignment",
        tasks=tasks,
        bootstrap_text="authorized bootstrap",
        protocol_documents={"correction": "stable protocol"},
        context_documents={"run-1": "run context"},
        knowledge_ref="knowledge://git:abc",
        knowledge_snapshot_identity="git:abc:tree:def",
        execution_identity={"driver": "baseline"},
        **kwargs,
    )


def _index(runtime: AgentTaskRuntime) -> dict:
    return json.loads(runtime.index_path.read_text(encoding="utf-8"))


def test_assignment_materializes_digest_refs_and_claims_one_task(tmp_path) -> None:
    runtime = _start(tmp_path, [_task("first"), _task("second")])
    index = _index(runtime)

    assert index["control_generation"] == 1
    assert index["next_action"] == "claim_ready"
    assert index["ready_task"]["task_id"] == "first"
    assert index["refs"]["bootstrap"].startswith("control/bootstrap.md#sha256:")
    assert "timestamp" not in runtime.index_path.read_text(encoding="utf-8")

    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim-1",
        expected_control_generation=1,
    )
    assert claimed["status"] == "task"
    assert claimed["task"]["task_id"] == "first"
    assert claimed["task"]["lease_generation"] == 1
    assert _index(runtime)["active_task"]["task_id"] == "first"
    assert "expires_at" not in runtime.index_path.read_text(encoding="utf-8")

    with pytest.raises(StaleControlGenerationError):
        runtime.next_task(
            assignment_id="assignment-1",
            worker_id="worker-1",
            request_id="claim-stale",
            expected_control_generation=1,
        )


def test_multiworker_assignment_leases_independent_tasks_concurrently(tmp_path) -> None:
    runtime = _start(
        tmp_path,
        [_task("first"), _task("second")],
        max_workers=2,
    )
    first = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim-1",
        expected_control_generation=1,
    )
    second = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-2",
        request_id="claim-2",
        expected_control_generation=first["control_generation"],
    )

    assert first["task"]["task_id"] == "first"
    assert second["task"]["task_id"] == "second"
    assert len(_index(runtime)["active_tasks"]) == 2
    with pytest.raises(AssignmentConflictError, match="worker limit"):
        runtime.next_task(
            assignment_id="assignment-1",
            worker_id="worker-3",
            request_id="claim-3",
            expected_control_generation=second["control_generation"],
        )

    first_done = runtime.submit(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=first["task"]["lease_generation"],
        request_id="submit-1",
        input_hash="sha256:first",
        candidate={"answer": 1},
    )
    assert first_done["status"] == "waiting"
    second_done = runtime.submit(
        assignment_id="assignment-1",
        task_id="second",
        worker_id="worker-2",
        lease_generation=second["task"]["lease_generation"],
        request_id="submit-2",
        input_hash="sha256:second",
        candidate={"answer": 2},
    )
    assert second_done["status"] == "assignment_complete"


def test_multiworker_waiter_survives_unrelated_state_change(tmp_path) -> None:
    runtime = _start(
        tmp_path,
        [
            _task("active"),
            _task("api", executor="external"),
            _task("after-api", dependencies=("api",)),
        ],
        max_workers=2,
    )
    first = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim-1",
        expected_control_generation=1,
    )
    waiting = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-2",
        request_id="wait-2",
        expected_control_generation=first["control_generation"],
    )
    assert waiting["status"] == "waiting"
    external = runtime.accept_external_task(
        assignment_id="assignment-1",
        task_id="api",
        request_id="api-result",
        expected_control_generation=waiting["control_generation"],
        input_hash="sha256:api",
        artifact={"result": 1},
    )
    woke = runtime.await_next_task(
        assignment_id="assignment-1",
        worker_id="worker-2",
        wait_token=waiting["wait_token"],
        # Not a bound this test is about: the result is already accepted, so
        # the watch returns on its first pass. It has to stay above the
        # waiter's freshness window all the same -- at 0.1s the stamp written
        # by `next_task` has aged out by the time the machine gets here under
        # load, the watch re-stamps it, and the commit bumps the very
        # generation compared below. That failed roughly one full-suite run in
        # three.
        max_wait_seconds=5,
    )
    assert woke == {
        "status": "ready",
        "control_generation": external["control_generation"],
        "next_action": "claim_ready",
        "index": "control/index.json",
    }


def test_every_active_task_mutation_is_fenced_and_idempotent(tmp_path) -> None:
    now = [100.0]
    runtime = _start(
        tmp_path, [_task("first")], clock=lambda: now[0], lease_ttl_seconds=300
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    generation = claimed["task"]["lease_generation"]

    checkpoint = runtime.checkpoint_progress(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=generation,
        request_id="checkpoint",
        progress={"stage": 1},
    )
    # A replay is the same id *and* the same input (protocol v4); the same id
    # with new input is refused instead of answered with the old response.
    assert runtime.checkpoint_progress(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=generation,
        request_id="checkpoint",
        progress={"stage": 1},
    ) == checkpoint
    with pytest.raises(AssignmentConflictError, match="different input"):
        runtime.checkpoint_progress(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=generation,
            request_id="checkpoint",
            progress={"stage": 999},
        )
    with pytest.raises(AssignmentConflictError):
        runtime.heartbeat(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=generation,
            request_id="checkpoint",
        )

    # Past the TTL measured from the *checkpoint*, not from the claim: every
    # fenced call renews, which is what stands in for a keepalive.
    now[0] = 1000.0
    reclaimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="reclaim",
        expected_control_generation=checkpoint["control_generation"],
    )
    assert reclaimed["task"]["lease_generation"] == generation + 1
    with pytest.raises(StaleLeaseError):
        runtime.release_task(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=generation,
            request_id="old-release",
        )


def test_conversation_lineage_is_fenced_and_monotonic(tmp_path) -> None:
    runtime = _start(tmp_path, [_task("first")], session_scope="assignment")
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    lease = claimed["task"]["lease_generation"]
    checkpoint = runtime.checkpoint_conversation(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease,
        request_id="turn-1",
        conversation_epoch=1,
        conversation_handle="conversation-1",
        turn_generation=1,
        parent_turn_identity="sha256:turn-1",
    )
    assert checkpoint["status"] == "conversation_checkpointed"
    assert runtime.checkpoint_conversation(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease,
        request_id="turn-1",
        conversation_epoch=1,
        conversation_handle="conversation-1",
        turn_generation=1,
        parent_turn_identity="sha256:turn-1",
    ) == checkpoint
    with pytest.raises(AssignmentConflictError, match="different input"):
        runtime.checkpoint_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease,
            request_id="turn-1",
            conversation_epoch=99,
            conversation_handle="different",
            turn_generation=99,
            parent_turn_identity="different",
        )
    with pytest.raises(AssignmentConflictError, match="monotonic"):
        runtime.checkpoint_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease,
            request_id="turn-skip",
            conversation_epoch=1,
            conversation_handle="conversation-1",
            turn_generation=3,
            parent_turn_identity="sha256:turn-3",
        )


def test_conversation_reset_opens_a_new_epoch_without_stranding_the_task(
    tmp_path,
) -> None:
    runtime = _start(tmp_path, [_task("first")], session_scope="assignment")
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    lease = claimed["task"]["lease_generation"]
    runtime.checkpoint_conversation(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease,
        request_id="turn-1",
        conversation_epoch=1,
        conversation_handle="conversation-1",
        turn_generation=1,
        parent_turn_identity="sha256:turn-1",
    )
    reset = runtime.reset_conversation(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease,
        request_id="reset-1",
        reason="LocalAgentUnavailableError: session was pruned",
    )
    assert reset["status"] == "conversation_reset"
    assert reset["conversation_epoch"] == 2
    assert (
        runtime.reset_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease,
            request_id="reset-1",
            reason="LocalAgentUnavailableError: session was pruned",
        )
        == reset
    )
    with pytest.raises(AssignmentConflictError, match="different input"):
        runtime.reset_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease,
            request_id="reset-1",
            reason="a different reason",
        )

    lineage = runtime.conversation_state(
        assignment_id="assignment-1", worker_id="worker-1"
    )
    assert lineage["conversation_epoch"] == 2
    assert lineage["conversation_handle"] == ""
    assert lineage["turn_generation"] == 0
    assert lineage["resets"] == [
        {
            "conversation_epoch": 1,
            "conversation_handle": "conversation-1",
            "turn_generation": 1,
            "reason": "LocalAgentUnavailableError: session was pruned",
        }
    ]

    with pytest.raises(AssignmentConflictError, match="epoch"):
        runtime.checkpoint_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease,
            request_id="stale-epoch",
            conversation_epoch=1,
            conversation_handle="conversation-2",
            turn_generation=1,
            parent_turn_identity="sha256:turn-2",
        )
    resumed = runtime.checkpoint_conversation(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease,
        request_id="turn-2",
        conversation_epoch=2,
        conversation_handle="conversation-2",
        turn_generation=1,
        parent_turn_identity="sha256:turn-2",
    )
    assert resumed["conversation_epoch"] == 2
    assert (
        runtime.conversation_state(assignment_id="assignment-1", worker_id="worker-1")[
            "resets"
        ]
        == lineage["resets"]
    )


def test_conversation_reset_requires_a_live_lease_and_a_reason(tmp_path) -> None:
    runtime = _start(tmp_path, [_task("first")], session_scope="assignment")
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    lease = claimed["task"]["lease_generation"]
    with pytest.raises(ValueError, match="record why"):
        runtime.reset_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease,
            request_id="reset-empty",
            reason="   ",
        )
    with pytest.raises(StaleLeaseError):
        runtime.reset_conversation(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease + 1,
            request_id="reset-stale",
            reason="stale lease must not retire a live conversation",
        )


def test_submit_repairs_then_accepts_and_piggybacks_dependent_task(tmp_path) -> None:
    def validator(candidate, _manifest):
        if candidate.get("answer") != "ok":
            return ValidationResult.repairable("answer must be ok")
        return ValidationResult.accepted({"normalized": "ok"})

    runtime = _start(
        tmp_path,
        [_task("first", validator_id="strict"), _task("second", dependencies=("first",))],
        validators={"strict": validator},
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    lease_generation = claimed["task"]["lease_generation"]
    repair = runtime.submit(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease_generation,
        request_id="submit-bad",
        input_hash="sha256:first",
        candidate={"answer": "bad"},
    )
    assert repair == {
        "status": "repairable",
        "control_generation": 3,
        "validation_errors": ["answer must be ok"],
        # The durable repair/submit ledger (docs §7) rides every verdict.
        "max_repair_attempts": 5,
        "repair_attempts": 1,
        "repair_rounds_remaining": 4,
        "submit_count": 1,
        "max_submits": 8,
    }

    accepted = runtime.submit(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease_generation,
        request_id="submit-good",
        input_hash="sha256:first",
        candidate={"answer": "ok"},
    )
    assert accepted["status"] == "task"
    assert accepted["accepted_task_id"] == "first"
    assert accepted["task"]["task_id"] == "second"
    assert accepted["task"]["lease_generation"] == 1
    # A replay is the same id *and* the same input; the same id with a
    # different candidate is refused rather than answered with the old
    # response (protocol v4, docs/llm_agent_tool_protocol.md §2).
    assert runtime.submit(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=lease_generation,
        request_id="submit-good",
        input_hash="sha256:first",
        candidate={"answer": "ok"},
    ) == accepted
    with pytest.raises(AssignmentConflictError):
        runtime.submit(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=lease_generation,
            request_id="submit-good",
            input_hash="sha256:first",
            candidate={"different": True},
        )


def test_accepted_submit_wal_recovers_artifact_and_state(tmp_path, monkeypatch) -> None:
    runtime = _start(tmp_path, [_task("first")])
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )

    def crash(_state):
        raise OSError("simulated crash before index commit")

    monkeypatch.setattr(runtime, "_write_prepared_state_locked", crash)
    with pytest.raises(OSError, match="simulated crash"):
        runtime.submit(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=claimed["task"]["lease_generation"],
            request_id="submit",
            input_hash="sha256:first",
            candidate={"answer": "ok"},
        )
    assert runtime.wal_path.exists()

    recovered = AgentTaskRuntime(tmp_path / "assignment")
    status = recovered.status(assignment_id="assignment-1")
    assert status["status"] == "assignment_complete"
    assert not recovered.wal_path.exists()
    assert list((recovered.root / "tasks" / "first" / "submissions").glob("*.json"))


def test_external_dependency_wakes_waiter_without_claiming_task(tmp_path) -> None:
    runtime = _start(
        tmp_path,
        [
            _task("api", executor="external"),
            _task("agent", dependencies=("api",)),
        ],
    )
    waiting = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="wait",
        expected_control_generation=1,
    )
    assert waiting["status"] == "waiting"
    # What the protocol advertises is the conversational bound (a host turn
    # limit); MAX_WATCH_SECONDS is only the ceiling a headless worker may ask
    # for when it is parking against a provider conversation TTL.
    assert waiting["max_wait_seconds"] == CONVERSATIONAL_WATCH_SECONDS
    assert CONVERSATIONAL_WATCH_SECONDS < MAX_WATCH_SECONDS
    assert _index(runtime)["active_task"] is None

    external = runtime.accept_external_task(
        assignment_id="assignment-1",
        task_id="api",
        request_id="api-result",
        expected_control_generation=waiting["control_generation"],
        input_hash="sha256:api",
        artifact={"result": 42},
    )
    woke = runtime.await_next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        wait_token=waiting["wait_token"],
        # Not a bound this test is about: the result is already accepted, so
        # the watch returns on its first pass. It has to stay above the
        # waiter's freshness window all the same -- at 0.1s the stamp written
        # by `next_task` has aged out by the time the machine gets here under
        # load, the watch re-stamps it, and the commit bumps the very
        # generation compared below. That failed roughly one full-suite run in
        # three.
        max_wait_seconds=5,
    )
    assert woke == {
        "status": "ready",
        "control_generation": external["control_generation"],
        "next_action": "claim_ready",
        "index": "control/index.json",
    }
    assert _index(runtime)["active_task"] is None


def test_waiter_returns_still_waiting_at_its_bound(tmp_path) -> None:
    runtime = _start(tmp_path, [_task("api", executor="external")])
    waiting = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="wait",
        expected_control_generation=1,
    )
    result = runtime.await_next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        wait_token=waiting["wait_token"],
        max_wait_seconds=0.01,
    )
    assert result["status"] == "still_waiting"
    assert result["wait_token"] == waiting["wait_token"]


def test_blocked_task_requeues_a_bounded_number_of_times_then_fails(tmp_path) -> None:
    """`blocked` means "repair will not help", not "stop and tell nobody".

    It used to leave the task in a state no operation could move: not queued,
    not accepted, lease already cleared so `release_task` could not reach it.
    The assignment then never completed and never failed, so every worker
    waiting on it parked forever. Now it goes back to the queue for a bounded
    number of fresh attempts and then ends the assignment out loud.
    """

    runtime = _start(
        tmp_path,
        [_task("first", validator_id="blocked")],
        validators={
            "blocked": lambda _candidate, _manifest: ValidationResult.blocked(
                "needs owner"
            )
        },
        blocked_requeues=2,
    )
    statuses: list[str] = []
    for attempt in range(3):
        claimed = runtime.next_task(
            assignment_id="assignment-1",
            worker_id="worker-1",
            request_id=f"claim-{attempt}",
            expected_control_generation=runtime.status(
                assignment_id="assignment-1", worker_id="worker-1"
            )["control_generation"],
        )
        result = runtime.submit(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=claimed["task"]["lease_generation"],
            request_id=f"submit-{attempt}",
            input_hash="sha256:first",
            candidate={},
        )
        statuses.append(result["status"])

    assert statuses == ["blocked", "blocked", "failed"]
    # Two requeues, then the queue stops pretending there is work left.
    assert runtime.status(assignment_id="assignment-1")["status"] == "assignment_failed"
    assert _index(runtime)["next_action"] == "assignment_failed"
    assert _index(runtime)["failed_tasks"] == ["first"]


def test_a_parked_worker_is_told_the_assignment_failed(tmp_path) -> None:
    runtime = _start(
        tmp_path,
        [_task("first", validator_id="blocked"), _task("second", dependencies=("first",))],
        validators={
            "blocked": lambda _candidate, _manifest: ValidationResult.blocked("nope")
        },
        blocked_requeues=0,
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    runtime.submit(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=claimed["task"]["lease_generation"],
        request_id="submit",
        input_hash="sha256:first",
        candidate={},
    )
    woke = runtime.await_next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        wait_token="whatever",
        max_wait_seconds=0.1,
    )
    assert woke["next_action"] == "assignment_failed"


def test_assignment_definition_is_idempotent_but_conflicts_fail(tmp_path) -> None:
    runtime = _start(tmp_path, [_task("first")])
    same = _start(tmp_path, [_task("first")])
    assert same.status(assignment_id="assignment-1")["control_generation"] == 1
    with pytest.raises(AssignmentConflictError):
        _start(tmp_path, [_task("different")])
    with pytest.raises(ValueError, match="cycle"):
        _start(
            tmp_path / "cycle",
            [_task("a", dependencies=("b",)), _task("b", dependencies=("a",))],
        )
    assert runtime.index_path.exists()


def test_tampered_state_is_rejected(tmp_path) -> None:
    runtime = _start(tmp_path, [_task("first")])
    index = _index(runtime)
    relative = index["state_ref"].split("#", 1)[0]
    (runtime.root / relative).write_text("{}\n", encoding="utf-8")
    with pytest.raises(AgentTaskRuntimeError, match="digest mismatch"):
        runtime.status(assignment_id="assignment-1")


def test_a_dead_workers_lease_is_reclaimed_by_a_reader(tmp_path) -> None:
    """Reclamation cannot live only in `next_task`.

    A worker parked in `await_next_task` never calls `next_task` again, so a
    peer that crashed mid-task was never handed on: the survivor waited out the
    whole assignment while the task sat under an expired lease. Reads reclaim
    too, and a read is the one thing every survivor does.
    """

    now = [100.0]
    runtime = _start(
        tmp_path,
        [_task("first"), _task("second")],
        clock=lambda: now[0],
        lease_ttl_seconds=300,
        max_workers=2,
    )
    runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim-1",
        expected_control_generation=1,
    )
    now[0] = 5000.0

    seen = runtime.status(assignment_id="assignment-1", worker_id="worker-2")

    assert seen["status"] == "ready"
    assert _index(runtime)["active_tasks"] == []


def test_a_restarted_worker_is_not_handed_its_own_dead_lease(tmp_path) -> None:
    """Reusing the same worker id after a crash used to cost a whole call.

    `status` returned the expired lease as an active task, the worker ran the
    driver against it, and only `submit` noticed -- so the model call was paid
    for and thrown away, and the assignment ended on an exception instead of a
    fresh claim.
    """

    now = [100.0]
    runtime = _start(
        tmp_path, [_task("first")], clock=lambda: now[0], lease_ttl_seconds=300
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    now[0] = 5000.0

    revived = runtime.status(assignment_id="assignment-1", worker_id="worker-1")
    assert revived["status"] == "ready"

    reclaimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="reclaim",
        expected_control_generation=revived["control_generation"],
    )
    assert reclaimed["status"] == "task"
    assert (
        reclaimed["task"]["lease_generation"] > claimed["task"]["lease_generation"]
    )


def test_any_fenced_call_renews_the_lease(tmp_path) -> None:
    """Liveness comes from work, not from a keepalive.

    A conversational Agent has no process of its own between two control
    commands and cannot run a timer mid-turn, so a heartbeat it was told to
    send would be missed exactly when it mattered.
    """

    now = [100.0]
    runtime = _start(
        tmp_path, [_task("first")], clock=lambda: now[0], lease_ttl_seconds=300
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    generation = claimed["task"]["lease_generation"]

    for step, moment in enumerate((350.0, 600.0, 850.0)):
        now[0] = moment
        runtime.checkpoint_progress(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=generation,
            request_id=f"progress-{step}",
            progress={"step": step},
        )

    now[0] = 900.0
    accepted = runtime.submit(
        assignment_id="assignment-1",
        task_id="first",
        worker_id="worker-1",
        lease_generation=generation,
        request_id="submit",
        input_hash="sha256:first",
        candidate={"ok": True},
    )
    assert accepted["status"] == "assignment_complete"


def test_a_waiter_that_never_comes_back_gives_up_its_worker_slot(tmp_path) -> None:
    """A worker holds no lease while parked, so nothing else notices it die.

    What it does hold is a registration slot, and once `max_workers` is used up
    a replacement under a fresh id is refused outright.
    """

    now = [100.0]
    runtime = _start(
        tmp_path,
        [_task("api", executor="external"), _task("agent", dependencies=("api",))],
        clock=lambda: now[0],
    )
    waiting = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="wait",
        expected_control_generation=1,
    )
    assert waiting["status"] == "waiting"

    now[0] = 100.0 + CONVERSATIONAL_WATCH_SECONDS * 4
    runtime.status(assignment_id="assignment-1")

    replacement = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-2",
        request_id="replace",
        expected_control_generation=runtime.status(assignment_id="assignment-1")[
            "control_generation"
        ],
    )
    assert replacement["status"] == "waiting"


def test_retrieval_settles_against_its_reservation_not_a_live_lease(tmp_path) -> None:
    """A fetch slower than the TTL must still report its own outcome.

    Requiring a live lease here meant a slow call came back as a lease error
    rather than its result, and left the reservation `in_progress` holding a
    `max_parallel` slot that nobody could release.
    """

    now = [100.0]
    runtime = _start(
        tmp_path,
        [
            AgentTaskSpec(
                task_id="first",
                session_type="research",
                input_hash="sha256:first",
                goal="search",
                retrieval_mode="local",
                protocol_key="correction",
                context_key="run-1",
            )
        ],
        clock=lambda: now[0],
        lease_ttl_seconds=300,
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    leased = {
        "assignment_id": "assignment-1",
        "task_id": "first",
        "worker_id": "worker-1",
        "lease_generation": claimed["task"]["lease_generation"],
        "request_id": "search-1",
    }
    reserved = runtime.begin_retrieval_call(
        **leased, operation="search", request={"query": "q"}
    )
    assert reserved["status"] == "in_progress"

    now[0] = 5000.0
    settled = runtime.complete_retrieval_call(
        **leased,
        result={"items": [{"url": "https://example.test/a"}]},
        result_count=1,
        response_tokens=10,
        # Charged wall time stays inside the retrieval budget; what expired is
        # the task lease, which is the thing under test.
        wall_seconds=1.0,
    )
    assert settled["status"] == "completed"
    assert settled["budget"]["in_progress"] == 0


def test_reading_a_directory_with_no_assignment_leaves_nothing_behind(tmp_path) -> None:
    """A mistyped `--root` used to create `control/` wherever the Agent stood.

    Its own bootstrap warns about exactly that case, and the runtime then
    reported "no assignment here" only after littering the directory.
    """

    stray = tmp_path / "somebody-elses-project"
    stray.mkdir()

    with pytest.raises(AssignmentConflictError):
        AgentTaskRuntime(stray).status(assignment_id="assignment-1")

    assert list(stray.iterdir()) == []


def test_state_generations_are_pruned_to_a_bounded_tail(tmp_path) -> None:
    """Only the generation `index.json` names is ever read.

    The rest are forensics, and a long assignment that kept every one of them
    left one file per state change behind for good.
    """

    from finesub.llm.agent.agent_task_runtime import RETAINED_STATE_GENERATIONS

    runtime = _start(tmp_path, [_task("first")])
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    generation = claimed["task"]["lease_generation"]
    for step in range(RETAINED_STATE_GENERATIONS + 15):
        runtime.checkpoint_progress(
            assignment_id="assignment-1",
            task_id="first",
            worker_id="worker-1",
            lease_generation=generation,
            request_id=f"progress-{step}",
            progress={"step": step},
        )

    kept = sorted(runtime.state_root.glob("*.json"))
    assert len(kept) <= RETAINED_STATE_GENERATIONS + 1
    # Whatever else went, the one the index points at is still readable.
    assert runtime.status(assignment_id="assignment-1")["status"] == "task"


def test_the_doc_table_still_matches_the_constants() -> None:
    """The tuned numbers are written down twice; keep them one fact.

    They were briefly written down only *once* -- inside an implementation
    checkpoint that then got archived, taking the retention count and the
    watch-ceiling split out of the tracked docs with it. A table is the right
    home for them, and this is what stops it going stale silently.
    """

    from pathlib import Path

    from finesub.llm.agent import agent_quota
    from finesub.llm.agent.agent_transports import CONVERSATION_WATCH_MARGIN_SECONDS

    doc = (
        Path(__file__).resolve().parents[1] / "docs" / "llm_local_agent.md"
    ).read_text(encoding="utf-8")

    for row, holds in {
        "`DEFAULT_LEASE_TTL_SECONDS` | 30 分钟": DEFAULT_LEASE_TTL_SECONDS == 30 * 60,
        "`CONVERSATIONAL_WATCH_SECONDS` | 28 分钟": CONVERSATIONAL_WATCH_SECONDS == 28 * 60,
        "`MAX_WATCH_SECONDS` | 60 分钟": MAX_WATCH_SECONDS == 60 * 60,
        "`WAITER_ABANDON_FACTOR` | 2×": WAITER_ABANDON_FACTOR == 2.0,
        "`DEFAULT_BLOCKED_REQUEUES` | 2": DEFAULT_BLOCKED_REQUEUES == 2,
        "`RETAINED_STATE_GENERATIONS` | 20": RETAINED_STATE_GENERATIONS == 20,
        "`QUOTA_FREEZE_SECONDS` | 2 小时": agent_quota.QUOTA_FREEZE_SECONDS == 2 * 3600,
    }.items():
        assert holds, f"constant changed without the doc: {row}"
        assert row in doc, f"doc row missing or reworded: {row}"

    # The margin is what makes the ceiling large enough to be worth having.
    assert CONVERSATIONAL_WATCH_SECONDS < MAX_WATCH_SECONDS
    assert CONVERSATION_WATCH_MARGIN_SECONDS == 120.0
