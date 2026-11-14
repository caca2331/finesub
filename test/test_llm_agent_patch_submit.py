"""A tool session repairs a correction window with a `<patch>`, as the harness's own repair round does.

The runtime lays a patch over the draft -- the last rejected answer this lease submitted in this
conversation -- and judges the whole answer that makes (plan: tool session `submit`)."""

from __future__ import annotations

import inspect
import json

import pytest

from finesub.llm.agent.agent_task_runtime import AgentTaskRuntime, AgentTaskSpec
from finesub.llm.agent.agent_validators import runtime_validators, serialize_window
from finesub.llm.chunking import CorrectionBudget, SubtitleSegment, SubtitleWindow

HEADER = "type|position|corrected_text|translation|conf|note"


def _rows(*ids: int) -> str:
    return "\n".join(f"sub|{i}|{i}。|{i}|high|" for i in ids)


def _whole(*ids: int) -> str:
    return f"<translated>\n{HEADER}\n{_rows(*ids)}\n</translated>"


def _patch(*ids: int) -> str:
    return f"<patch>\n{HEADER}\n{_rows(*ids)}\n</patch>"


def _window() -> SubtitleWindow:
    return SubtitleWindow(
        chunk_id="0001",
        segments=[SubtitleSegment(str(i), i * 1.5, i * 1.5 + 1.0, f"{i}。") for i in range(1, 5)],
        overlap_segments=[],
        boundary_reason="end",
        budget=CorrectionBudget(0, 0, 0, ""),
        length_scale=1.0,
    )


def _runtime(tmp_path, *, scope: str = "task", window: SubtitleWindow | None = None) -> tuple[AgentTaskRuntime, dict]:
    spec = AgentTaskSpec(
        task_id="window", session_type="correction", input_hash="sha256:window", goal="correct the window",
        validator_id="correction-window", protocol_key="correction", context_key="run",
        metadata={"validator": {"window": serialize_window(window or _window())}, "variant": "capableC",
                  "capability_tier": "capable"},
    )
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment", assignment_id="a", worker_goal="g", tasks=[spec], bootstrap_text="b",
        protocol_documents={"correction": "p"}, context_documents={"run": "c"}, knowledge_ref="k",
        knowledge_snapshot_identity="k", execution_identity={"driver": "test"},
        validators=runtime_validators("correction-window"), session_scope=scope,
    )
    claimed = runtime.next_task(assignment_id="a", worker_id="w", request_id="claim", expected_control_generation=1)
    lease = dict(assignment_id="a", task_id="window", worker_id="w",
                 lease_generation=claimed["task"]["lease_generation"])
    return runtime, lease


def _submit(runtime, lease, request_id: str, candidate: str) -> dict:
    return runtime.submit(**lease, request_id=request_id, input_hash="sha256:window", candidate=candidate)


def _record(runtime) -> dict:
    return runtime.task_record(assignment_id="a", task_id="window")


def test_a_patch_over_the_rejected_answer_is_accepted_as_the_whole(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    rejected = _submit(runtime, lease, "s1", _whole(1, 2, 3))
    assert rejected["status"] == "repairable"
    assert "<patch>" in rejected["repair_instruction"] and "要改的源序号：4" in rejected["repair_instruction"]

    accepted = _submit(runtime, lease, "s2", _patch(4))
    assert accepted.get("accepted_task_id") == "window"
    artifact = runtime.read_artifact(_record(runtime)["accepted_artifact_ref"])
    assert _rows(1, 2, 3, 4) in json.loads(artifact)["artifact"]


def test_a_rejected_patch_keeps_the_whole_it_made_as_the_draft(tmp_path) -> None:
    """A patch that merges but still fails is the next draft, and what a session ending unaccepted reports:
    the whole answer, never the patch."""

    runtime, lease = _runtime(tmp_path)
    first = _submit(runtime, lease, "s1", _whole(1, 2))
    assert "patch_merged" not in first and "rows_to_fix" not in first
    still_missing = _submit(runtime, lease, "s2", _patch(3))
    assert still_missing["status"] == "repairable" and "4" in " ".join(still_missing["validation_errors"])
    # The errors are about the whole the patch made; source 4 has no row to show.
    assert still_missing["patch_merged"] is True and "rows_to_fix" not in still_missing
    assert _record(runtime)["last_candidate"].count("sub|") == 3
    assert _submit(runtime, lease, "s3", _patch(4)).get("accepted_task_id") == "window"


def test_a_rejected_patch_shows_the_rows_still_to_fix_not_the_whole(tmp_path) -> None:
    """The session has its earlier answer and its patch; what it lacks is how the failing rows now stand."""

    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", _whole(1, 2))
    bad = "sub|3||3|high|"
    verdict = _submit(runtime, lease, "s2", f"<patch>\n{HEADER}\n{bad}\n</patch>")
    assert verdict["status"] == "repairable" and verdict["patch_merged"] is True
    assert verdict["rows_to_fix"] == bad and "要改的源序号：3、4" in verdict["repair_instruction"]
    assert _submit(runtime, lease, "s3", _patch(3, 4)).get("accepted_task_id") == "window"


def test_an_answer_judged_before_is_the_draft_again(tmp_path) -> None:
    """A replayed verdict is about the answer sent again, so that answer is what the next patch goes over."""

    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", _whole(1, 2, 3))
    _submit(runtime, lease, "s2", _whole(1, 2, 4))
    replayed = _submit(runtime, lease, "s3", _whole(1, 2, 3))
    assert replayed.get("replayed") is True and "patch_merged" not in replayed
    assert _record(runtime)["last_candidate"] == _whole(1, 2, 3)
    assert _submit(runtime, lease, "s4", _patch(4)).get("accepted_task_id") == "window"


def test_a_patch_making_a_whole_judged_before_is_replayed_without_rows(tmp_path) -> None:
    """The stored verdict holds no answer text, and the session already saw those rows when it was first given."""

    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", _whole(1, 2))
    bad = f"<patch>\n{HEADER}\nsub|3||3|high|\n</patch>"
    first = _submit(runtime, lease, "s2", bad)
    assert first["rows_to_fix"] and not first.get("replayed")
    again = _submit(runtime, lease, "s3", bad)
    assert again.get("replayed") is True and again["patch_merged"] is True and "rows_to_fix" not in again
    assert _record(runtime)["last_candidate"].count("sub|") == 3
    assert _submit(runtime, lease, "s4", _patch(3, 4)).get("accepted_task_id") == "window"


def test_forget_drafts_leaves_no_row_a_patch_made(tmp_path) -> None:
    """The rejected whole a patch made, and the rows shown of it, are answer text like any draft: the content
    cache keeps none of it, and `forget_drafts` leaves none behind."""

    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", _whole(1, 2))
    marked = f"<patch>\n{HEADER}\nsub|3|x|0.0|3。|MERGED-THEN-REJECTED|high|1|\n</patch>"
    assert "MERGED-THEN-REJECTED" in _submit(runtime, lease, "s2", marked)["rows_to_fix"]
    assert _submit(runtime, lease, "s3", _patch(3, 4)).get("accepted_task_id") == "window"

    runtime.forget_drafts(assignment_id="a")
    assert not [path for path in tmp_path.rglob("*") if path.is_file()
                and "MERGED-THEN-REJECTED" in path.read_text(encoding="utf-8", errors="ignore")]


def test_the_submit_reply_carries_the_rows_and_names_no_whole(tmp_path) -> None:
    """The MCP reply passes `patch_merged` and `rows_to_fix` through; there is nothing of the whole to read."""

    from finesub.llm.agent.agent_mcp_server import HarnessToolServer

    runtime, lease = _runtime(tmp_path)
    server = HarnessToolServer(runtime, assignment_id="a", worker_id="w", session_id="s", instance_id="i")
    # The task `_runtime` already leased to this worker, as the opening claim hands it to the server.
    server._task = {"task_id": "window", "lease_generation": lease["lease_generation"]}
    server._manifest = json.loads(runtime.read_artifact(_record(runtime)["manifest_ref"]))

    def call(rpc_id, name, **arguments):
        reply = server.handle({"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call",
                               "params": {"name": name, "arguments": arguments}})
        return json.loads(reply["result"]["content"][0]["text"])

    call(1, "submit", payload=_whole(1, 2))
    bad = "sub|3||3|high|"
    rejected = call(2, "submit", payload=f"<patch>\n{HEADER}\n{bad}\n</patch>")
    assert rejected["patch_merged"] is True and rejected["rows_to_fix"] == bad
    assert _rows(1, 2) not in json.dumps(rejected, ensure_ascii=False)


def test_the_draft_helpers_keep_their_decorators() -> None:
    """A classmethod stacked on a staticmethod works on 3.12, the version CI runs, and fails every rejected
    submit on 3.13."""

    assert isinstance(inspect.getattr_static(AgentTaskRuntime, "_keep_draft"), staticmethod)
    assert isinstance(inspect.getattr_static(AgentTaskRuntime, "_owed_blocks"), classmethod)


def test_an_unusable_patch_leaves_the_draft_and_spends_a_repair(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", _whole(1, 2, 3))
    bad = _submit(runtime, lease, "s2", f"<patch>\n{HEADER}\nsub|9|x|x|high|\n</patch>")
    assert bad["status"] == "repairable" and "outside this window" in bad["validation_errors"][0]
    assert bad["repair_attempts"] == 2
    assert _record(runtime)["last_candidate"] == _whole(1, 2, 3)
    assert _submit(runtime, lease, "s3", _patch(4)).get("accepted_task_id") == "window"


def test_a_patch_with_no_draft_is_told_to_send_the_whole_answer_for_free(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    verdict = _submit(runtime, lease, "s1", _patch(4))
    assert verdict["status"] == "repairable" and "submit the whole answer" in verdict["validation_errors"][0]
    assert verdict["repair_attempts"] == 0
    assert not _record(runtime).get("last_candidate")


@pytest.mark.parametrize("scope", ["task", "assignment"])
def test_a_draft_from_before_a_conversation_reset_is_not_a_base(tmp_path, scope) -> None:
    """The model that answers after a reset never saw the old draft; a patch then has nothing to go over."""

    runtime, lease = _runtime(tmp_path, scope=scope)
    _submit(runtime, lease, "s1", _whole(1, 2, 3))
    runtime.reset_conversation(**lease, request_id="reset", reason="lost the handle")
    verdict = _submit(runtime, lease, "s2", _patch(4))
    assert "submit the whole answer" in verdict["validation_errors"][0]


def test_lint_judges_the_whole_a_patch_makes(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", _whole(1, 2, 3))
    assert runtime.lint(**lease, candidate=_patch(4))["verdict"] == "accepted"
    assert runtime.lint(**lease, candidate=_patch(3))["verdict"] == "repairable"


@pytest.mark.parametrize("scale", [0.8, 1.0])
def test_submit_holds_the_window_to_the_absolute_threshold_of_its_ratio(tmp_path, scale) -> None:
    """The manifest carries the window's length ratio to the runtime's validator: a merge spanning 6.0 s passes
    the standard tier's 7 s and goes back at the short tier's 5.6 s, where a patch splitting it is accepted."""

    from dataclasses import replace

    window = replace(_window(), segments=[SubtitleSegment("1", 0.0, 3.0, "1。"), SubtitleSegment("2", 3.1, 6.0, "2。"),
                                          SubtitleSegment("3", 7.5, 8.5, "3。"), SubtitleSegment("4", 9.0, 10.0, "4。")],
                     length_scale=scale)
    runtime, lease = _runtime(tmp_path, window=window)
    merged = f"<translated>\n{HEADER}\nsub|1,2|1。2。|12|high|同一句\n{_rows(3, 4)}\n</translated>"
    first = _submit(runtime, lease, "s1", merged)
    if scale == 1.0:
        assert first.get("accepted_task_id") == "window"
        return
    assert first["status"] == "repairable" and "5.6s" in first["repair_instruction"]
    assert _submit(runtime, lease, "s2", _patch(1, 2)).get("accepted_task_id") == "window"
