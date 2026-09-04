"""Protocol v4 contract: the required-block ledger, `retire_task`, dedup fingerprints.

docs/llm_agent_tool_protocol.md §3 (ledger + submit gate + one protocol
repair per context), §4 (`retire_task` CAS and who wins a race) and §2
(dedup records carry an input fingerprint; the table is bounded).
"""

from __future__ import annotations

import json

import pytest

from finesub.llm.agent.agent_task_runtime import (
    MAX_RETIREMENTS_PER_TASK,
    MAX_REQUEST_RESULTS,
    AgentTaskRuntime,
    AgentTaskSpec,
    AssignmentConflictError,
    StaleLeaseError,
    ValidationResult,
)

PROTOCOL = {"kind": "protocol", "digest": "sha256:p1", "ref": "control/protocol.md", "tool": "read_context"}
PAYLOAD = {"kind": "payload", "digest": "sha256:w7", "ref": "tasks/call/payload.txt"}


def _runtime(tmp_path, *, session_scope="task", required=(PROTOCOL, PAYLOAD), validator=None):
    validators = {"strict": validator} if validator else None
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
                validator_id="strict" if validator else "accept",
                required_blocks=tuple(required),
            )
        ],
        session_scope=session_scope,
        protocol_documents={"correction": "protocol"},
        validators=validators,
    )


def _claim(runtime, worker="worker-1"):
    status = runtime.rehydrate(assignment_id="assignment-1", worker_id=worker)
    return runtime.next_task(
        assignment_id="assignment-1",
        worker_id=worker,
        request_id=f"claim-{worker}-{status['control_generation']}",
        expected_control_generation=status["control_generation"],
    )["task"]


def _submit(runtime, task, candidate, *, request_id, worker="worker-1"):
    return runtime.submit(
        assignment_id="assignment-1",
        task_id=task["task_id"],
        worker_id=worker,
        lease_generation=task["lease_generation"],
        request_id=request_id,
        input_hash="sha256:in",
        candidate=candidate,
    )


def _pull(runtime, task, blocks, *, request_id, source="pull"):
    return runtime.record_pull(
        assignment_id="assignment-1",
        task_id=task["task_id"],
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id=request_id,
        blocks=blocks,
        source=source,
    )


def test_the_manifest_declares_the_required_blocks(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    task = _claim(runtime)
    manifest = json.loads(runtime.read_artifact(task["manifest_ref"]))
    assert manifest["protocol_version"] == "agent-task-v4"
    assert [block["kind"] for block in manifest["required_blocks"]] == ["protocol", "payload"]
    assert manifest["required_blocks"][1]["tool"] == "read_context"


def test_a_submit_that_owes_blocks_is_rejected_once_naming_every_block(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    task = _claim(runtime)

    first = _submit(runtime, task, "answer", request_id="s1")

    assert first["status"] == "repairable"
    assert first["protocol_violation"] == "missing_required_blocks"
    assert [block["kind"] for block in first["owed_blocks"]] == ["protocol", "payload"]
    assert all("call read_context" in error for error in first["validation_errors"])
    assert "ref=control/protocol.md" in first["validation_errors"][0]
    status = runtime.pull_status(
        assignment_id="assignment-1", task_id="call", worker_id="worker-1"
    )
    assert [block["kind"] for block in status["owed_blocks"]] == ["protocol", "payload"]


def test_pulled_blocks_let_the_submit_through_to_the_validator(tmp_path) -> None:
    seen: list = []

    def validator(candidate, _manifest):
        seen.append(candidate)
        return ValidationResult.accepted(candidate)

    runtime = _runtime(tmp_path, validator=validator)
    task = _claim(runtime)
    _submit(runtime, task, "answer", request_id="s1")
    assert seen == []

    pulled = _pull(runtime, task, [PROTOCOL], request_id="p1")
    assert [block["kind"] for block in pulled["owed_blocks"]] == ["payload"]
    _pull(runtime, task, [PAYLOAD], request_id="p2")

    accepted = _submit(runtime, task, "answer", request_id="s2")
    assert accepted["status"] == "assignment_complete"
    assert seen == ["answer"]


def test_a_second_miss_in_the_same_context_goes_through_the_expensive_loop(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    task = _claim(runtime)
    _submit(runtime, task, "answer", request_id="s1")
    _pull(runtime, task, [PROTOCOL], request_id="p1")

    second = _submit(runtime, task, "answer", request_id="s2")

    # Retired in the same write (docs §0-9): conversation reset, lease
    # revoked, task re-queued, counted as a replacement.
    assert second["status"] == "retired"
    assert second["retirements"] == 1
    assert [block["kind"] for block in second["owed_blocks"]] == ["payload"]
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "queued" and record["lease_owner"] == ""
    assert record["last_candidate"] == "answer"
    with pytest.raises(StaleLeaseError):
        _submit(runtime, task, "late", request_id="s3")
    # The next claim is a new lease, and a new lease is a new context with an
    # empty ledger.
    fresh = _claim(runtime)
    assert fresh["lease_generation"] == task["lease_generation"] + 1
    status = runtime.pull_status(
        assignment_id="assignment-1", task_id="call", worker_id="worker-1"
    )
    assert status["pulled"] == []


def test_a_pushed_block_counts_as_pulled(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    task = _claim(runtime)
    _pull(runtime, task, [PROTOCOL, PAYLOAD], request_id="push-1", source="push")

    assert _submit(runtime, task, "answer", request_id="s1")["status"] == "assignment_complete"


def test_assignment_scope_keeps_the_ledger_across_a_checkpoint_and_clears_it_on_reset(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, session_scope="assignment")
    task = _claim(runtime)
    _pull(runtime, task, [PROTOCOL], request_id="p1")
    runtime.checkpoint_conversation(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="c1",
        conversation_epoch=1,
        conversation_handle="conv-1",
        turn_generation=1,
        parent_turn_identity="sha256:t1",
    )
    status = runtime.pull_status(
        assignment_id="assignment-1", task_id="call", worker_id="worker-1"
    )
    assert status["pulled"] == ["protocol@sha256:p1"]

    runtime.reset_conversation(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="r1",
        reason="compact",
    )
    status = runtime.pull_status(
        assignment_id="assignment-1", task_id="call", worker_id="worker-1"
    )
    assert status["pulled"] == []
    assert [block["kind"] for block in status["owed_blocks"]] == ["protocol", "payload"]


def test_retire_task_resets_the_conversation_and_requeues_under_a_new_generation(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, session_scope="assignment", required=())
    task = _claim(runtime)
    runtime.checkpoint_conversation(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="c1",
        conversation_epoch=1,
        conversation_handle="conv-1",
        turn_generation=1,
        parent_turn_identity="sha256:t1",
    )

    retired = runtime.retire_task(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="retire-1",
        reason="repair budget exhausted",
    )

    assert retired["status"] == "retired"
    assert retired["conversation_epoch"] == 2
    assert retired["retirements"] == 1
    lineage = runtime.conversation_state(assignment_id="assignment-1", worker_id="worker-1")
    assert lineage["conversation_handle"] == ""
    assert lineage["resets"][-1]["reason"] == "retired: repair budget exhausted"
    # A replay of the same id is the same answer, not a second retirement.
    assert (
        runtime.retire_task(
            assignment_id="assignment-1",
            task_id="call",
            worker_id="worker-1",
            lease_generation=task["lease_generation"],
            request_id="retire-1",
            reason="repair budget exhausted",
        )
        == retired
    )
    # Same worker id, fresh session: the next claim is a new generation.
    fresh = _claim(runtime)
    assert fresh["lease_generation"] == task["lease_generation"] + 1
    with pytest.raises(StaleLeaseError):
        _submit(runtime, task, "late", request_id="late-1")


def test_an_accepted_submit_wins_against_a_late_retire(tmp_path) -> None:
    runtime = _runtime(tmp_path, required=())
    task = _claim(runtime)
    assert _submit(runtime, task, "answer", request_id="s1")["status"] == "assignment_complete"

    with pytest.raises(StaleLeaseError):
        runtime.retire_task(
            assignment_id="assignment-1",
            task_id="call",
            worker_id="worker-1",
            lease_generation=task["lease_generation"],
            request_id="retire-1",
            reason="too late",
        )
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    assert index["next_action"] == "assignment_complete"


def test_a_replayed_request_id_returns_the_first_answer_only_for_the_same_input(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, required=(), validator=lambda c, _m: (
        ValidationResult.accepted(c) if c == "good" else ValidationResult.repairable("no")
    ))
    task = _claim(runtime)

    first = _submit(runtime, task, "bad", request_id="s1")
    assert first["status"] == "repairable"
    # Same id, same input: the transport replayed the call.
    assert _submit(runtime, task, "bad", request_id="s1") == first
    # Same id, different input: an id reused for new work, and it is refused
    # rather than answered with the old response.
    with pytest.raises(AssignmentConflictError):
        _submit(runtime, task, "good", request_id="s1")
    # New id, same input: a new logical call, run again.
    again = _submit(runtime, task, "bad", request_id="s2")
    assert again["status"] == "repairable"
    assert _submit(runtime, task, "good", request_id="s3")["status"] == "assignment_complete"


def test_pull_records_keep_only_the_fingerprint(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    task = _claim(runtime)
    _pull(runtime, task, [PROTOCOL], request_id="p1")
    state_path = runtime.state_root / json.loads(
        runtime.index_path.read_text(encoding="utf-8")
    )["state_ref"].split("#", 1)[0].rsplit("/", 1)[-1]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    row = state["request_results"]["p1"]
    assert row["operation"] == "record_pull"
    assert row["response"] is None
    assert row["fingerprint"].startswith("sha256:")
    # Replaying it runs again (idempotently) instead of failing.
    replay = _pull(runtime, task, [PROTOCOL], request_id="p1")
    assert replay["status"] == "pull_recorded"
    with pytest.raises(AssignmentConflictError):
        _pull(runtime, task, [PAYLOAD], request_id="p1")
    # The source is input too: the same id booked as a push is a conflict.
    with pytest.raises(AssignmentConflictError):
        _pull(runtime, task, [PROTOCOL], request_id="p1", source="push")


def test_a_task_scope_reset_clears_the_ledger_but_keeps_the_lease(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    task = _claim(runtime)
    _pull(runtime, task, [PROTOCOL, PAYLOAD], request_id="p1")
    runtime.reset_conversation(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="r1",
        reason="premature stop",
    )
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["lease_owner"] == "worker-1"
    assert record["lease_generation"] == task["lease_generation"]
    status = runtime.pull_status(assignment_id="assignment-1", task_id="call", worker_id="worker-1")
    assert status["pulled"] == []
    assert _submit(runtime, task, "answer", request_id="s1")["status"] == "repairable"


def test_the_dedup_table_is_bounded(tmp_path) -> None:
    runtime = _runtime(tmp_path, required=())
    task = _claim(runtime)
    for index in range(MAX_REQUEST_RESULTS - 1):
        _pull(runtime, task, [PROTOCOL], request_id=f"p{index}")
    with pytest.raises(AssignmentConflictError):
        _pull(runtime, task, [PROTOCOL], request_id="one-too-many")


# --- docs §7: submit is idempotent by content, capped by call count ---------


def _reject_unless_good(candidate, _manifest):
    return (
        ValidationResult.accepted(candidate)
        if candidate == "good"
        else ValidationResult.repairable("no")
    )


def test_a_new_request_id_with_the_same_answer_replays_the_verdict_without_spending_budget(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, required=(), validator=_reject_unless_good)
    task = _claim(runtime)

    first = _submit(runtime, task, "bad", request_id="s1")
    assert first["status"] == "repairable"
    assert (first["repair_attempts"], first["submit_count"]) == (1, 1)

    # Same id: a transport replay, nothing moves.
    assert _submit(runtime, task, "bad", request_id="s1") == first

    # New id, same content: the answer was already judged. The verdict comes
    # back, the submit counts, the repair budget does not.
    again = _submit(runtime, task, "bad", request_id="s2")
    assert again["status"] == "repairable"
    assert again["replayed"] is True
    assert (again["repair_attempts"], again["submit_count"]) == (1, 2)

    # New content: a real new attempt.
    other = _submit(runtime, task, "worse", request_id="s3")
    assert "replayed" not in other
    assert (other["repair_attempts"], other["submit_count"]) == (2, 3)
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert (record["repair_attempts"], record["submit_count"]) == (2, 3)


def test_a_session_that_never_changes_its_answer_is_retired_by_the_submit_cap(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path, required=(), validator=_reject_unless_good)
    task = _claim(runtime)
    cap = runtime.task_record(assignment_id="assignment-1", task_id="call")["max_submits"]

    responses = [
        _submit(runtime, task, "bad", request_id=f"s{index}") for index in range(cap)
    ]
    assert {item["status"] for item in responses} == {"repairable"}
    # Content dedup kept the repair budget at one, so only the cap ends this.
    assert responses[-1]["repair_attempts"] == 1

    retired = _submit(runtime, task, "bad", request_id="s-over")
    assert retired["status"] == "retired"
    assert retired["protocol_violation"] == "submit_cap"
    record = runtime.task_record(assignment_id="assignment-1", task_id="call")
    assert record["status"] == "queued"
    assert record["retirements"] == 1
    assert record["last_candidate"] == "bad"
    # A fresh lease is a fresh context: the ledger starts over.
    assert (record["submit_count"], record["repair_attempts"]) == (0, 0)



def test_a_task_retired_past_its_cap_fails_instead_of_starting_over(tmp_path) -> None:
    """Each retirement hands the task back with a fresh budget.

    Without a cap, a session that will not follow the protocol just starts
    over, and the only thing that ends it is the caller's per-task deadline
    -- a whole model timeout spent on a task nobody was going to finish.
    """

    runtime = _runtime(tmp_path, required=(), validator=_reject_unless_good)
    for retirement in range(1, MAX_RETIREMENTS_PER_TASK + 1):
        task = _claim(runtime)
        cap = runtime.task_record(assignment_id="assignment-1", task_id="call")["max_submits"]
        for index in range(cap):
            _submit(runtime, task, "bad", request_id=f"s{retirement}-{index}")
        out = _submit(runtime, task, "bad", request_id=f"s{retirement}-over")
        record = runtime.task_record(assignment_id="assignment-1", task_id="call")
        assert record["retirements"] == retirement
        if retirement < MAX_RETIREMENTS_PER_TASK:
            assert out["status"] == "retired" and record["status"] == "queued"
    assert out["status"] == "failed"
    assert record["status"] == "failed"
    assert "retired 2 times" in " ".join(record["validation_errors"])
    # Nothing is handed out again: the assignment reports the failure.
    assert runtime.status(assignment_id="assignment-1", worker_id="worker-1")["status"] == (
        "assignment_failed"
    )


def _request_rows(runtime) -> dict:
    index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    return json.loads(runtime.read_artifact(index["state_ref"]))["request_results"]


def test_an_external_accept_leaves_no_request_row_either(tmp_path) -> None:
    """An external accept is an accept (docs §7).

    Tagging its row was not enough: nothing prunes an external task, so a
    scheduler feeding one dependency per window filled the bounded table just
    as the agent's own accepts used to.
    """

    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id=f"ext-{index}", session_type="correction",
                input_hash=f"sha256:ext-{index}", goal="g", validator_id="accept",
                executor="external",
            )
            for index in range(3)
        ],
        session_scope="task",
    )
    generations = {}
    for index in range(3):
        generations[index] = runtime.status(assignment_id="assignment-1")[
            "control_generation"
        ]
        accepted = runtime.accept_external_task(
            assignment_id="assignment-1", task_id=f"ext-{index}",
            request_id=f"r{index}", expected_control_generation=generations[index],
            input_hash=f"sha256:ext-{index}", artifact="done",
        )
        assert accepted["accepted_task_id"] == f"ext-{index}"
    assert _request_rows(runtime) == {}
    # The one answer each still owes -- its accept -- comes off the task row,
    # so re-sending it is still answered rather than refused as "not queued".
    replayed = runtime.accept_external_task(
        assignment_id="assignment-1", task_id="ext-0", request_id="r0",
        expected_control_generation=generations[0],
        input_hash="sha256:ext-0", artifact="done",
    )
    assert replayed["accepted_task_id"] == "ext-0"
    # Same id, different work: still refused, exactly as the request table did.
    with pytest.raises(AssignmentConflictError):
        runtime.accept_external_task(
            assignment_id="assignment-1", task_id="ext-0", request_id="r0",
            expected_control_generation=generations[0],
            input_hash="sha256:ext-0", artifact="something else",
        )


def test_a_conversation_reset_keeps_the_submit_ledger_a_retire_clears_it(tmp_path) -> None:
    runtime = _runtime(tmp_path, required=(), validator=_reject_unless_good)
    task = _claim(runtime)
    _submit(runtime, task, "bad", request_id="s1")
    runtime.reset_conversation(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="reset-1",
        reason="premature stop",
    )
    # Same lease after the reset: the fresh CLI re-sending the old answer is a
    # replay, not a second attempt.
    after_reset = _submit(runtime, task, "bad", request_id="s2")
    assert after_reset["replayed"] is True
    assert after_reset["repair_attempts"] == 1

    runtime.retire_task(
        assignment_id="assignment-1",
        task_id="call",
        worker_id="worker-1",
        lease_generation=task["lease_generation"],
        request_id="retire-1",
        reason="session ended",
    )
    fresh = _claim(runtime)
    assert fresh["lease_generation"] == task["lease_generation"] + 1
    renewed = _submit(runtime, fresh, "bad", request_id="s3")
    assert "replayed" not in renewed
    assert renewed["repair_attempts"] == 1


def test_a_missing_block_rejection_is_neither_budgeted_nor_cached(tmp_path) -> None:
    runtime = _runtime(tmp_path, validator=_reject_unless_good)
    task = _claim(runtime)
    owed = _submit(runtime, task, "good", request_id="s1")
    assert owed["protocol_violation"] == "missing_required_blocks"
    assert owed["repair_attempts"] == 0
    _pull(runtime, task, [PROTOCOL, PAYLOAD], request_id="pull-1")
    # The same content again, now with the blocks read: judged for real, not
    # answered with the stale "you owe blocks" verdict.
    accepted = _submit(runtime, task, "good", request_id="s2")
    assert accepted.get("accepted_task_id") == "call"
