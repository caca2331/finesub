"""A batch's state on disk: the registry, the lock, the published queue view
and the user's control channel.

Split from `test_pipeline_items.py` when `batch_state` was split from
`pipeline`, on the same line: the subject here is what a batch KNOWS about
itself between runs, not what a task is. Several of these drive `main` --
resume and the batch lock have no smaller door -- but the thing under test is
still the state, not the CLI.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
import threading
import time
import types

import pytest

from finesub import batch_state, pipeline
from finesub.batch_state import (
    batch_is_live,
    batch_lock_path,
    control_intake,
    record_batch,
    resolve_resume_batch,
    strip_view_keys,
    write_queue_view,
)
from finesub.pipeline import build_item, merge_item_options
from finesub.scheduler import BatchItem, IntakePoll, ItemResult, run_batch


def _item(label: str, stages: dict) -> BatchItem:
    return BatchItem(label=label, stages=stages, payload=label)


def _registry(tmp_path, monkeypatch):
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(batch_state, "resolve_logs_dir", lambda: logs)
    return logs.parent / "batches.json"


def _resumable_queue(tmp_path, monkeypatch, batch_id: str, rows: list[dict]):
    """A published queue for `batch_id`, registered as unfinished."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME

    _registry(tmp_path, monkeypatch)
    queue = tmp_path / "batch" / batch_id / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pipeline.record_batch(batch_id, queue, state="unfinished", items=len(rows))
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    return queue


def test_a_run_publishes_its_queue_and_the_view_is_a_manifest_again(tmp_path, monkeypatch) -> None:
    """The view is also the record: its rows are the rows that were asked for,
    so `--manifest` on it is how a batch is continued -- and doing nothing is
    how it is not (there is no standing queue to unwind)."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME, strip_view_keys

    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(sys, "argv", ["finesub", str(a), str(b), "--batch-id", "x", "--language", "ja"])

    def fake_run_batch(items, **kwargs):
        results = [ItemResult(label=item.label, status="done") for item in items]
        kwargs["publish"](items, results)  # as the runner does on every change
        return results

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    assert pipeline.main() == 0

    view = tmp_path / "batch" / "x" / QUEUE_VIEW_FILENAME
    rows = [json.loads(line) for line in view.read_text(encoding="utf-8").splitlines()]
    assert [row["_state"] for row in rows] == ["done", "done"]
    assert [row["source"] for row in rows] == [str(a), str(b)]
    # ...and it reads back as a manifest: the runner's own keys are stripped.
    assert "_state" not in strip_view_keys(rows[0])
    assert pipeline.merge_item_options(strip_view_keys(rows[0]), {})["language"] == "ja"
def test_the_control_channel_adds_reprioritises_and_drops(tmp_path) -> None:
    """One append-only channel the user owns, polled on the runner's own tick:
    a line with `item` is an instruction, anything else is a task."""

    from finesub.pipeline import control_intake

    source = tmp_path / "late.wav"
    source.write_bytes(b"")
    control = tmp_path / "control.jsonl"
    poll = control_intake(
        control, admit=lambda opts: _item(opts["source"], {})
    )
    assert poll().items == ()  # the file need not exist

    control.write_text(
        json.dumps({"source": str(source)}, ensure_ascii=False) + "\n"
        + json.dumps({"item": "a.wav", "priority": 9}) + "\n"
        + json.dumps({"item": "b.wav", "drop": True}) + "\n"
        + "not json\n",
        encoding="utf-8",
    )
    result = poll()
    assert [item.label for item in result.items] == [str(source)]
    assert [dict(action) for action in result.actions] == [
        {"item": "a.wav", "priority": 9},
        {"item": "b.wav", "drop": True},
    ]
    assert result.settled  # the bad line was skipped, not waited on
    assert poll().items == () and poll().actions == ()  # each line acted on once
def test_resume_batch_without_an_id_takes_the_last_unfinished_one(tmp_path, monkeypatch) -> None:
    """`queue.jsonl` lives with the outputs it describes, which is right for
    the file and useless for finding it again -- hence the pointer."""

    registry = _registry(tmp_path, monkeypatch)
    old_queue = tmp_path / "old" / "queue.jsonl"
    new_queue = tmp_path / "new" / "queue.jsonl"
    for path in (old_queue, new_queue):
        path.parent.mkdir()
        path.write_text("", encoding="utf-8")

    pipeline.record_batch("aaa", old_queue, state="unfinished", items=2)
    pipeline.record_batch("bbb", new_queue, state="finished", items=1)
    pipeline.record_batch("ccc", new_queue, state="running", items=3)
    assert registry.is_file()

    resolved, batch_id, refusal = pipeline.resolve_resume_batch("")
    assert refusal == "" and resolved == new_queue  # `running` counts: it was killed
    # A finished batch is never the id-less answer, but naming it still works.
    assert pipeline.resolve_resume_batch("bbb")[0] == new_queue
    assert pipeline.resolve_resume_batch("nope")[0] is None
def test_a_week_old_batch_is_not_resumed_by_surprise(tmp_path, monkeypatch) -> None:
    """Not a prompt (it would hang a script) and not a silent pick-up: it says
    so and prints the command that names the id."""

    registry = _registry(tmp_path, monkeypatch)
    queue = tmp_path / "queue.jsonl"
    queue.write_text("", encoding="utf-8")
    pipeline.record_batch("stale-1", queue, state="unfinished", items=1)
    rows = json.loads(registry.read_text(encoding="utf-8"))
    rows[0]["updated_at"] -= (pipeline.STALE_RESUME_DAYS + 1) * 86400
    registry.write_text(json.dumps(rows), encoding="utf-8")

    resolved, batch_id, refusal = pipeline.resolve_resume_batch("")
    assert resolved is None
    assert "--resume-batch stale-1" in refusal and "days old" in refusal
    # Naming it is a deliberate choice, and is never refused for age.
    assert pipeline.resolve_resume_batch("stale-1")[0] == queue
def test_resume_batch_feeds_the_published_queue_back(tmp_path, monkeypatch) -> None:
    from finesub.batch_state import QUEUE_VIEW_FILENAME

    _registry(tmp_path, monkeypatch)
    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    queue = tmp_path / "batch" / "x" / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps({"source": str(source), "language": "ja", "_state": "failed",
                    "_error": "RuntimeError: boom"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pipeline.record_batch("x", queue, state="unfinished", items=1)
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")

    seen: dict = {}

    def fake_run_batch(items, **kwargs):
        seen["rows"] = [dict(item.row or {}) for item in items]
        return [ItemResult(label=item.label, status="done") for item in items]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])
    assert pipeline.main() == 0
    # The runner's own keys did not come back as manifest options.
    assert seen["rows"][0]["source"] == str(source)
    assert seen["rows"][0]["language"] == "ja"
    assert not [key for key in seen["rows"][0] if key.startswith("_state")]
def test_a_running_batch_is_not_resumed_alongside_itself(tmp_path, monkeypatch) -> None:
    """Two runs on one batch directory would drive two sets of workers over
    one set of outputs. The lock the run holds is what says it is live."""

    from finesub_bootstrap.locks import holding_lock

    _registry(tmp_path, monkeypatch)
    queue = tmp_path / "batch" / "live" / "queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text("", encoding="utf-8")
    pipeline.record_batch("live", queue, state="running", items=1)

    assert pipeline.resolve_resume_batch("")[0] == queue  # nobody holds it
    with holding_lock(pipeline.batch_lock_path(queue)):
        assert batch_state.batch_is_live(queue)
        for requested in ("", "live"):  # naming it does not make it safe
            resolved, _, refusal = pipeline.resolve_resume_batch(requested)
            assert resolved is None
            assert "still running" in refusal and "control.jsonl" in refusal
    assert pipeline.resolve_resume_batch("live")[0] == queue  # released with it
def test_a_batch_from_another_directory_is_not_resumed_here(tmp_path, monkeypatch) -> None:
    """`out/` and every relative source are CWD-relative: resumed elsewhere it
    would find none of the first half's work and land the second half in a
    different tree."""

    registry = _registry(tmp_path, monkeypatch)
    queue = tmp_path / "batch" / "x" / "queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text("", encoding="utf-8")
    pipeline.record_batch("x", queue, state="unfinished", items=1)
    rows = json.loads(registry.read_text(encoding="utf-8"))
    rows[0]["cwd"] = str(tmp_path / "elsewhere")
    registry.write_text(json.dumps(rows), encoding="utf-8")

    resolved, batch_id, refusal = pipeline.resolve_resume_batch("")
    assert resolved is None
    assert "cd " in refusal and "elsewhere" in refusal
    assert pipeline.resolve_resume_batch("x")[0] is None  # not a matter of intent
def test_a_dropped_item_does_not_come_back_on_resume(tmp_path, monkeypatch) -> None:
    """Interrupted and failed rows are resumed; a withdrawn one is the whole
    point of having withdrawn it."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME

    _registry(tmp_path, monkeypatch)
    kept, gone = tmp_path / "kept.wav", tmp_path / "gone.wav"
    kept.write_bytes(b"")
    gone.write_bytes(b"")
    queue = tmp_path / "batch" / "x" / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps({"source": str(kept), "_state": "queued"}) + "\n"
        + json.dumps({"source": str(gone), "_state": "dropped"}) + "\n",
        encoding="utf-8",
    )
    pipeline.record_batch("x", queue, state="unfinished", items=2)
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")

    seen: dict = {}

    def fake_run_batch(items, **kwargs):
        seen["labels"] = [item.label for item in items]
        return [ItemResult(label=item.label, status="done") for item in items]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])
    assert pipeline.main() == 0
    assert seen["labels"] == ["kept.wav"]
def test_the_view_shows_queued_and_running_and_survives_concurrent_publishes(tmp_path) -> None:
    """Every worker publishes, so the view is written concurrently -- and it is
    what a resume reads, so it does not get to be approximately right."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME, write_queue_view

    view = tmp_path / QUEUE_VIEW_FILENAME
    items = [_item(f"i{n}", {}) for n in range(20)]
    results = [ItemResult(label=item.label) for item in items]
    results[0].stage = "asr"
    results[1].status = "done"
    results[2].dropped = True
    results[2].status = "skipped"

    barrier = threading.Barrier(8)

    def publish() -> None:
        barrier.wait(30)
        for _ in range(30):
            write_queue_view(view, items, results)

    threads = [threading.Thread(target=publish) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    rows = [json.loads(line) for line in view.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(items)
    assert [row["_state"] for row in rows[:4]] == ["running", "done", "dropped", "queued"]
    assert rows[0]["_stage"] == "asr"
    assert not list(tmp_path.glob("*.tmp"))
def test_resuming_keeps_the_batch_s_identity(tmp_path, monkeypatch, capsys) -> None:
    """A resume continues the batch; it does not start a second one over its
    outputs. Same id, same directory, same queue view, same registry row --
    which is also what makes the live-batch lock and the "did it finish?"
    bookkeeping refer to the batch being resumed."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME

    _registry(tmp_path, monkeypatch)
    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    queue = tmp_path / "batch" / "b1" / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps({"source": str(source), "_state": "queued"}) + "\n", encoding="utf-8")
    pipeline.record_batch("b1", queue, state="unfinished", items=1)
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")

    def fake_run_batch(items, **kwargs):
        results = [ItemResult(label=item.label, status="done") for item in items]
        kwargs["publish"](items, results)
        return results

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])
    assert pipeline.main() == 0
    assert "Run b1:" in capsys.readouterr().out  # not a new timestamped id
    # No second batch directory, and the run finished the row it resumed...
    assert [path.name for path in (tmp_path / "batch").iterdir() if path.is_dir()] == ["b1"]
    rows = json.loads((tmp_path / "data" / "batches.json").read_text(encoding="utf-8"))
    assert [(row["batch_id"], row["state"]) for row in rows] == [("b1", "finished")]
    # ...so the next id-less resume does not offer it again.
    assert pipeline.resolve_resume_batch("")[0] is None
def test_a_live_batch_cannot_be_resumed_through_main(tmp_path, monkeypatch, capsys) -> None:
    from finesub.batch_state import QUEUE_VIEW_FILENAME
    from finesub_bootstrap.locks import holding_lock

    _registry(tmp_path, monkeypatch)
    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    queue = tmp_path / "batch" / "b2" / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(json.dumps({"source": str(source)}) + "\n", encoding="utf-8")
    pipeline.record_batch("b2", queue, state="running", items=1)
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])

    # The lock the first run holds is the one on the batch being resumed.
    with holding_lock(pipeline.batch_lock_path(queue)):
        assert pipeline.main() == 2
    assert "still running" in capsys.readouterr().err
def test_a_second_run_colliding_with_a_live_batch_is_refused(
    tmp_path, monkeypatch, capsys
) -> None:
    """Both forms are refused on the held lock, with advice that fits the
    form: a batch has a control channel to add to, while a single run polls
    none -- pointing it at one was a NameError (reviewer 2026-08-31 P2)."""

    from finesub.batch_state import CONTROL_FILENAME, QUEUE_VIEW_FILENAME
    from finesub_bootstrap.locks import holding_lock

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    other = tmp_path / "b.wav"
    other.write_bytes(b"")
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    view = tmp_path / "batch" / "nightly" / QUEUE_VIEW_FILENAME
    view.parent.mkdir(parents=True)
    with holding_lock(pipeline.batch_lock_path(view)):
        monkeypatch.setattr(
            sys, "argv", ["finesub", str(source), "--batch-id", "nightly"]
        )
        assert pipeline.main() == 2
        err = capsys.readouterr().err
        assert "already running" in err and "--batch-id" in err
        assert CONTROL_FILENAME not in err  # a single run polls no channel
        monkeypatch.setattr(
            sys, "argv", ["finesub", str(source), str(other), "--batch-id", "nightly"]
        )
        assert pipeline.main() == 2
        assert CONTROL_FILENAME in capsys.readouterr().err
def test_a_priority_lowered_back_to_the_default_does_not_persist(tmp_path) -> None:
    """The view publishes the item's priority now, not the row's -- otherwise
    9 -> 0 leaves the 9 standing and a resume reads it back."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME, strip_view_keys, write_queue_view

    view = tmp_path / QUEUE_VIEW_FILENAME
    item = BatchItem(
        label="a.wav",
        stages={"asr": lambda payload: payload},
        priority=9,
        row={"source": "a.wav", "priority": 9},
    )
    polls = {"n": 0}

    def intake():
        polls["n"] += 1
        if polls["n"] == 1:
            return IntakePoll(actions=({"item": "a.wav", "priority": 0},))
        return IntakePoll()

    run_batch(
        [item],
        workers={"asr": 1},
        intake=intake,
        intake_poll_seconds=0.05,
        publish=lambda batch_items, results: write_queue_view(view, batch_items, results),
    )
    row = json.loads(view.read_text(encoding="utf-8").splitlines()[0])
    assert "priority" not in row
    assert "priority" not in strip_view_keys(row)
def test_two_batches_registering_at_once_both_stay_findable(tmp_path, monkeypatch) -> None:
    """One file, every batch on the machine: an unlocked read-modify-write
    loses whichever run wrote first, and that run is then unresumable."""

    registry = _registry(tmp_path, monkeypatch)
    queues = []
    for name in range(6):
        queue = tmp_path / f"b{name}" / "queue.jsonl"
        queue.parent.mkdir()
        queue.write_text("", encoding="utf-8")
        queues.append(queue)

    barrier = threading.Barrier(len(queues))

    def record(index: int) -> None:
        barrier.wait(30)
        pipeline.record_batch(f"b{index}", queues[index], state="running", items=1)

    threads = [threading.Thread(target=record, args=(n,)) for n in range(len(queues))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    rows = json.loads(registry.read_text(encoding="utf-8"))
    assert sorted(row["batch_id"] for row in rows) == [f"b{n}" for n in range(len(queues))]
def test_a_control_line_written_before_the_kill_is_not_lost(tmp_path) -> None:
    """The cursor is durable and advances only once the runner has acted: a
    line on disk is not evidence that anything was done about it, so a run
    killed between the append and the next poll replays it."""

    from finesub.pipeline import control_intake

    control = tmp_path / "control.jsonl"
    cursor = tmp_path / ".control-cursor"
    control.write_text(json.dumps({"source": str(tmp_path / "a.wav")}) + "\n", encoding="utf-8")

    def fresh():
        return control_intake(
            control,
            admit=lambda opts: _item(opts["source"], {}),
            cursor_path=cursor,
        )

    # A run that reads the line and dies before committing.
    killed = fresh()()
    assert [item.label for item in killed.items] == [str(tmp_path / "a.wav")]
    assert not cursor.exists()

    # The next run does the work, and commits.
    resumed = fresh()()
    assert [item.label for item in resumed.items] == [str(tmp_path / "a.wav")]
    resumed.commit()
    assert cursor.read_text(encoding="utf-8") == "1"

    # Only now is it behind us.
    assert fresh()().items == ()
def test_one_id_in_two_directories_keeps_both_batches(tmp_path, monkeypatch) -> None:
    """`--batch-id nightly` in two checkouts, or two runs starting in the same
    second: keyed on the id alone, the second silently erased the first from
    the only place `--resume-batch` looks."""

    _registry(tmp_path, monkeypatch)
    homes = []
    for name in ("one", "two"):
        home = tmp_path / name
        (home / "out").mkdir(parents=True)
        queue = home / "out" / "queue.jsonl"
        queue.write_text("", encoding="utf-8")
        homes.append((home, queue))

    for home, queue in homes:
        monkeypatch.chdir(home)
        pipeline.record_batch("nightly", queue, state="unfinished", items=1)

    rows = json.loads((tmp_path / "data" / "batches.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    # ...and each directory resumes its own, by id or without one.
    for home, queue in homes:
        monkeypatch.chdir(home)
        assert pipeline.resolve_resume_batch("")[0] == queue
        assert pipeline.resolve_resume_batch("nightly")[0] == queue
def test_a_batch_left_with_only_dropped_items_settles(tmp_path, monkeypatch, capsys) -> None:
    """Killed after the last item was withdrawn: there is nothing to run, and
    "no input" (exit 2) left it `running` forever, to be picked again by the
    next id-less resume."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME

    _registry(tmp_path, monkeypatch)
    queue = tmp_path / "batch" / "b3" / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(
        json.dumps({"source": str(tmp_path / "gone.wav"), "_state": "dropped"}) + "\n",
        encoding="utf-8",
    )
    pipeline.record_batch("b3", queue, state="running", items=1)
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])

    assert pipeline.main() == 0
    assert "nothing left in the queue" in capsys.readouterr().out
    # Closed by the ordinary end-of-run bookkeeping, under the batch lock.
    assert pipeline.resolve_resume_batch("")[0] is None
def test_a_batch_whose_queue_is_gone_says_so(tmp_path, monkeypatch) -> None:
    _registry(tmp_path, monkeypatch)
    queue = tmp_path / "batch" / "b4" / "queue.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text("", encoding="utf-8")
    pipeline.record_batch("b4", queue, state="unfinished", items=1)
    queue.unlink()

    resolved, _, refusal = pipeline.resolve_resume_batch("b4")
    assert resolved is None and "no queue left" in refusal
    # The id-less form never offers one whose queue is gone in the first place.
    assert pipeline.resolve_resume_batch("")[0] is None
def test_the_view_survives_a_row_that_is_not_json(tmp_path, monkeypatch, capsys) -> None:
    """A row is the merged options, and those are not all JSON: `--name` puts
    a Path in `output`. The view is a rendering; it must not be the thing that
    fails -- and the rendered Path is that option's manifest spelling."""

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(sys, "argv", ["finesub", str(source), "--batch-id", "n1", "--name", "myrun"])

    def fake_run_batch(items, **kwargs):
        results = [ItemResult(label=item.label, status="done") for item in items]
        kwargs["publish"](items, results)
        return results

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    assert pipeline.main() == 0
    assert "could not publish" not in capsys.readouterr().err

    row = json.loads((tmp_path / "batch" / "n1" / "queue.jsonl").read_text(encoding="utf-8"))
    assert isinstance(row["output"], str) and row["output"].endswith("myrun.srt")
    assert pipeline.merge_item_options(pipeline.strip_view_keys(row), {})["output"] == row["output"]
def test_the_view_is_rendered_inside_the_write_lock(tmp_path) -> None:
    """Serialising only the write let a stale body land last: a worker renders
    its rows, the intake thread then admits a task, publishes and commits its
    cursor, and the stale body finally takes the lock and erases the task --
    leaving a cursor that says the line was consumed and a view with no sign of
    it. Rendering under the same lock makes the last writer the last reader."""

    from finesub.batch_state import (
        _VIEW_WRITE_LOCK, QUEUE_VIEW_FILENAME, write_queue_view)

    rendered = threading.Event()

    class _WatchedRow(Mapping):
        """A row that says when the view started reading it."""

        def __init__(self, value: dict) -> None:
            self._value = value

        def keys(self):
            rendered.set()
            return self._value.keys()

        def __getitem__(self, key):
            return self._value[key]

        def __iter__(self):
            rendered.set()
            return iter(self._value)

        def __len__(self):
            return len(self._value)

    view = tmp_path / QUEUE_VIEW_FILENAME
    items = [
        BatchItem(label="a.wav", stages={}, row=_WatchedRow({"source": "a.wav"}))
    ]
    results = [ItemResult(label="a.wav")]
    publisher = threading.Thread(
        target=lambda: write_queue_view(view, items, results)
    )

    with _VIEW_WRITE_LOCK:
        publisher.start()
        publisher.join(0.3)
        assert not rendered.is_set(), "the rows were read outside the lock"
    publisher.join(10)
    assert rendered.is_set() and view.is_file()
def test_a_dropped_out_batch_still_runs_a_control_task_it_never_admitted(
    tmp_path, monkeypatch
) -> None:
    """Closing such a batch on the spot skipped the one place work could still
    be waiting: a control line the killed run never got to act on. It goes
    through the ordinary (locked, polled) path with zero starting items."""

    from finesub.batch_state import CONTROL_FILENAME

    gone, late = tmp_path / "gone.wav", tmp_path / "late.wav"
    gone.write_bytes(b"")
    late.write_bytes(b"")
    queue = _resumable_queue(
        tmp_path, monkeypatch, "b8", [{"source": str(gone), "_state": "dropped"}]
    )
    queue.with_name(CONTROL_FILENAME).write_text(
        json.dumps({"source": str(late)}) + "\n", encoding="utf-8"
    )

    ran: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "build_item",
        lambda opts, claims=None: BatchItem(
            label=Path(str(opts["source"])).name,
            stages={"asr": lambda payload: ran.append(str(opts["source"])) or payload},
            row=dict(opts),
        ),
    )
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])
    assert pipeline.main() == 0
    assert ran == [str(late)]
    # It ended as this batch's own run, so the cursor is now past that line...
    assert queue.with_name(".control-cursor").read_text(encoding="utf-8") == "1"
    assert pipeline.resolve_resume_batch("")[0] is None
    # ...and the task it picked up was published into this batch's own view.
    published = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert [row["_state"] for row in published] == ["done"]
    assert published[0]["source"] == str(late)
def test_settling_an_empty_resume_needs_the_batch_lock_like_any_other_run(
    tmp_path, monkeypatch, capsys
) -> None:
    """The old fast path wrote the registry without holding `.batch.lock`, so
    it could close a batch another process had just started."""

    from finesub_bootstrap.locks import holding_lock

    gone = tmp_path / "gone.wav"
    gone.write_bytes(b"")
    queue = _resumable_queue(
        tmp_path, monkeypatch, "b9", [{"source": str(gone), "_state": "dropped"}]
    )
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])

    with holding_lock(pipeline.batch_lock_path(queue)):
        assert pipeline.main() == 2
        assert "still running" in capsys.readouterr().err
    rows = json.loads((tmp_path / "data" / "batches.json").read_text(encoding="utf-8"))
    assert [row["state"] for row in rows] == ["unfinished"]  # nobody settled it
def test_a_failed_publish_holds_the_control_cursor_until_it_succeeds(tmp_path) -> None:
    """End to end through the real control channel: the read cursor moves when
    a line is handed over, the durable one only when its effect is on disk.

    The gap therefore outlives the poll that opened it -- the polls that must
    close it are the later, empty ones -- so "did THIS poll bring anything"
    was the wrong question to ask before committing.
    """

    from finesub.pipeline import (
        CONTROL_FILENAME,
        QUEUE_VIEW_FILENAME,
        control_intake,
        write_queue_view,
    )

    source = tmp_path / "late.wav"
    source.write_bytes(b"")
    control = tmp_path / CONTROL_FILENAME
    cursor = tmp_path / ".control-cursor"
    view = tmp_path / QUEUE_VIEW_FILENAME
    control.write_text(json.dumps({"source": str(source)}) + "\n", encoding="utf-8")

    ran: list[str] = []
    working = threading.Event()
    release = threading.Event()
    polls = {"n": 0}

    def stage(payload):
        ran.append(str(source))
        release.wait(10)  # keeps the batch from draining before poll 3
        return payload

    intake = control_intake(
        control,
        admit=lambda opts: BatchItem(
            label=Path(str(opts["source"])).name,
            stages={"asr": stage},
            row=dict(opts),
        ),
        cursor_path=cursor,
    )

    def counting_intake():
        polls["n"] += 1
        if polls["n"] == 3:
            # Poll 1 read the line and could not publish it. Poll 2 brought
            # nothing of its own -- and used to retire the line regardless.
            assert not cursor.exists(), "retired a task the view never recorded"
            working.set()
            release.set()
        return intake()

    def publish(items, results):
        if not working.is_set():
            raise OSError("no space left on device")
        write_queue_view(view, items, results)

    run_batch(
        [],
        workers={"asr": 1},
        intake=counting_intake,
        intake_poll_seconds=0.05,
        publish=publish,
    )
    assert ran == [str(source)]
    assert cursor.read_text(encoding="utf-8") == "1"
    assert json.loads(view.read_text(encoding="utf-8").splitlines()[0])["source"] == str(source)
