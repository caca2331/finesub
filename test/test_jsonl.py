"""JSON Lines ledgers: what is appended is what a resume reads back.

Two ways a record used to be lost: `str.splitlines()` cut a line at U+2028,
U+2029 or U+0085 -- characters `json.dumps(ensure_ascii=False)` leaves as they
are -- and an append after a crash's half line fused the new record onto it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from finesub import jsonl
from finesub.reporting import NullReporter, reporting_to

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
SEPARATORS = "a b c\x85d"


class _Warnings(NullReporter):
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.entries.append((code, message))


def test_a_record_with_line_separators_round_trips(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    jsonl.append(path, {"text": SEPARATORS})
    jsonl.append(path, {"text": "plain"})
    assert [row["text"] for row in jsonl.iter_records(path)] == [SEPARATORS, "plain"]
    # Escaped on disk too, so a reader that does split on them is safe.
    assert " " not in path.read_text(encoding="utf-8")
    assert " " not in path.read_text(encoding="utf-8")


def test_lines_cuts_at_newline_only_and_counts_like_splitlines() -> None:
    assert jsonl.lines("") == []
    assert jsonl.lines("a\nb") == ["a", "b"]
    assert jsonl.lines("a\r\nb\n") == ["a", "b"]
    assert jsonl.lines(f"{SEPARATORS}\n") == [SEPARATORS]


def test_an_append_after_a_torn_line_stays_readable(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    jsonl.append(path, {"n": 1})
    with path.open("ab") as handle:
        handle.write(b'{"n": 2, "te')  # the crash
    recorder = _Warnings()
    with reporting_to(recorder):
        # A torn last line is an interrupted write: skipped, not reported.
        assert [row["n"] for row in jsonl.iter_records(path)] == [1]
        assert recorder.entries == []
        jsonl.append(path, {"n": 3}, fsync=True)
        assert [row["n"] for row in jsonl.iter_records(path)] == [1, 3]
        # Now a complete bad line: reported, once per file and count.
        list(jsonl.iter_records(path))
    assert [code for code, _ in recorder.entries] == ["jsonl-bad-lines"]


def test_a_torn_multibyte_tail_does_not_break_the_read(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    jsonl.append(path, {"text": "一"})
    with path.open("ab") as handle:
        handle.write('{"text": "二'.encode("utf-8")[:-1])
    assert [row["text"] for row in jsonl.iter_records(path)] == ["一"]


def test_the_session_checkpoint_ledger(tmp_path) -> None:
    from finesub.llm.session_checkpoint import SESSION_CHECKPOINT_FILENAME, SessionCheckpointStore

    SessionCheckpointStore(tmp_path).commit(session="s", key="k1", input_hash="h", content=SEPARATORS)
    with (tmp_path / SESSION_CHECKPOINT_FILENAME).open("ab") as handle:
        handle.write(b'{"schema_version": 1, "sta')
    SessionCheckpointStore(tmp_path).commit(session="s", key="k2", input_hash="h", content="after")
    reloaded = SessionCheckpointStore(tmp_path)
    assert reloaded.get("s", "k1", "h").content == SEPARATORS
    assert reloaded.get("s", "k2", "h").content == "after"


def test_the_correction_window_ledger(tmp_path) -> None:
    from finesub.llm.stages.correction.commit import _append_window_cache, _load_window_cache

    path = tmp_path / "correction-windows.jsonl"
    _append_window_cache(path, {"task_fingerprint": "f", "chunk_id": "w1", "content": SEPARATORS})
    path.write_bytes(path.read_bytes() + b'{"task_fingerprint": "f", "chu')
    _append_window_cache(path, {"task_fingerprint": "f", "chunk_id": "w2", "content": "after"})
    cache = _load_window_cache(path, "f")
    assert cache["w1"]["content"] == SEPARATORS
    assert cache["w2"]["content"] == "after"


def test_the_task_artifact_ledger(tmp_path) -> None:
    from finesub.llm.knowledge.base import append_task_artifact
    from finesub.llm.knowledge.feedback import iter_task_artifacts

    append_task_artifact(tmp_path, kind="note", payload={"text": SEPARATORS})
    assert [row["payload"]["text"] for row in iter_task_artifacts([tmp_path])] == [SEPARATORS]


def test_the_knowledge_update_chunk_ledger(tmp_path) -> None:
    from finesub.llm.knowledge.update import _append_chunk_ledger, _load_chunk_ledger

    path = tmp_path / "knowledge-update-chunks.jsonl"
    _append_chunk_ledger(path, {"status": "applied", "input_hash": "h", "note": SEPARATORS})
    assert _load_chunk_ledger(path)["h"]["note"] == SEPARATORS


def test_the_share_conflict_ledger(tmp_path) -> None:
    from finesub.llm.knowledge.share import conflicts

    conflicts._append(tmp_path, {"conflict_id": "c1", "theirs": SEPARATORS})
    assert conflicts._rows(tmp_path)[0]["theirs"] == SEPARATORS


def test_the_published_queue_resumes(tmp_path) -> None:
    """`--resume-batch` reads `queue.jsonl` back as a manifest; a row holding
    U+2028 (pasted into `--extra-info-file`) made it `not valid JSON`."""

    from finesub.batch_state import write_queue_view
    from finesub.pipeline import manifest_snapshot
    from finesub.scheduler import BatchItem, ItemResult

    view = tmp_path / "queue.jsonl"
    item = BatchItem(label="a.wav", stages={}, payload="a.wav", row={"source": "a.wav", "extra_info": SEPARATORS})
    write_queue_view(view, [item], [ItemResult(label="a.wav")])
    rows, consumed, _ = manifest_snapshot(view)
    assert [row["extra_info"] for row in rows] == [SEPARATORS] and consumed == 1


def test_the_control_channel(tmp_path) -> None:
    from finesub.batch_state import control_intake
    from finesub.scheduler import BatchItem

    control = tmp_path / "control.jsonl"
    poll = control_intake(control, admit=lambda opts: BatchItem(label=opts["source"], stages={}, payload=None))
    control.write_text(json.dumps({"source": SEPARATORS}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert [item.label for item in poll().items] == [SEPARATORS]


def _splitlines_with_json_loads(source: Path) -> list[int]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    hits: list[int] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        if any(call.func.attr == "loads" for call in calls):
            hits.extend(call.lineno for call in calls if call.func.attr == "splitlines")
    return hits


def test_json_lines_are_not_split_with_splitlines() -> None:
    """A function under `src/` that parses JSON may not cut text with `splitlines()`.

    Scan surface: every `.py` under `src/`, function by function (methods and
    nested functions included; a nested one counts toward its outer function
    too). A function is flagged when it both calls `.loads(...)` and
    `.splitlines()` -- the shape of every hand-written JSONL read this
    replaced, from ledgers on disk to an agent CLI's stdout and a model's
    JSON-per-line answer. Not seen: lines split in one function and parsed in
    another. Read a file with `jsonl.iter_records`, cut text with `jsonl.lines`.
    """

    repository_root = SOURCE_ROOT.parent
    offenders = [
        f"{source.relative_to(repository_root).as_posix()}:{line}"
        for source in sorted(SOURCE_ROOT.rglob("*.py"))
        for line in _splitlines_with_json_loads(source)
    ]
    assert offenders == [], "use finesub.jsonl.iter_records / jsonl.lines instead of splitlines()"
