"""The naming and placement rules of a task's outputs.

The rule itself, apart from any front end's adapter over it: the desktop had
one (`TaskRequest`) and the CLI has `_RunPlan`, and keeping the rule here is
what stops a front-end change from quietly redefining it.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

from finesub_bootstrap import task_output


def test_an_invalid_index_is_backed_up_before_a_new_one_replaces_it(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    invalid = b'\xff{not-json'
    index.write_bytes(invalid)

    task_index.merge_write(
        index,
        [{"task_id": "new", "state": "completed", "updated_at": 1.0}],
    )

    assert (tmp_path / "tasks.json.invalid").read_bytes() == invalid
    assert [item["task_id"] for item in task_index.read(index)] == ["new"]


def test_invalid_index_backups_never_overwrite_an_earlier_recovery_copy(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    first_backup = tmp_path / "tasks.json.invalid"
    first_backup.write_text("first", encoding="utf-8")
    index.write_text("{}", encoding="utf-8")

    task_index.merge_write(
        index,
        [{"task_id": "new", "state": "completed", "updated_at": 1.0}],
    )

    assert first_backup.read_text(encoding="utf-8") == "first"
    assert (tmp_path / "tasks.json.invalid.1").read_text(encoding="utf-8") == "{}"


def test_a_failed_invalid_index_backup_leaves_the_original_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    index.write_text("{broken", encoding="utf-8")

    def fail_copy(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        task_index.shutil,
        "copy2",
        fail_copy,
    )

    with pytest.raises(OSError, match="disk full"):
        task_index.merge_write(
            index,
            [{"task_id": "new", "state": "completed", "updated_at": 1.0}],
        )

    assert index.read_text(encoding="utf-8") == "{broken"


def test_one_unusable_timestamp_does_not_stop_the_index(tmp_path: Path) -> None:
    """Every entry is compared against every other on each write.

    A record carrying a string where a number belongs -- a hand-edit, a writer
    we have not met -- would raise `TypeError` inside the merge and stop *all*
    history from being written, on both front ends, for as long as it stayed in
    the file. Ordering the strange entry early is a far smaller wrong.
    """

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    index.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "tasks": [
                    {"task_id": "bad", "state": "completed", "updated_at": "yesterday"},
                    {"task_id": "older", "state": "completed", "updated_at": 5.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    task_index.merge_write(
        index,
        [{"task_id": "new", "state": "completed", "updated_at": 9.0}],
        tmp_path / "tasks",
    )

    assert [entry["task_id"] for entry in task_index.read(index)] == [
        "bad",
        "older",
        "new",
    ]


def test_a_second_writer_does_not_drop_the_first(tmp_path: Path) -> None:
    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    entry = {
        "task_id": "desktop-task",
        "state": "completed",
        "request": {"input": "a.wav"},
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    task_index.merge_write(index, [entry], tmp_path / "tasks")
    task_index.merge_write(
        index,
        [{**entry, "task_id": "cli-task", "updated_at": 2.0}],
        tmp_path / "tasks",
    )

    assert [item["task_id"] for item in task_index.read(index, tmp_path / "tasks")] == [
        "desktop-task",
        "cli-task",
    ]


def test_equal_timestamp_keeps_the_disk_record(tmp_path: Path) -> None:
    """A path-only migration preserves updated_at, so disk wins a tie."""

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    current_root = tmp_path / "current" / "tasks"
    task_index.merge_write(
        index,
        [
            {
                "task_id": "task-a",
                "state": "completed",
                "request": {
                    "input": "a.wav",
                    "output": str(current_root / "task-a" / "clip.srt"),
                },
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ],
        current_root,
    )
    task_index.merge_write(
        index,
        [
            {
                "task_id": "task-a",
                "state": "completed",
                "request": {
                    "input": "a.wav",
                    "output": str(
                        tmp_path / "obsolete" / "tasks" / "task-a" / "clip.srt"
                    ),
                },
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ],
        current_root,
    )

    raw = json.loads(index.read_text(encoding="utf-8"))["tasks"][0]
    assert raw["request"]["output"] == "task-a/clip.srt"


def test_old_terminal_cannot_replace_a_newer_running_generation(
    tmp_path: Path,
) -> None:
    """The generation check and history merge must share one file lock."""

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    tasks = tmp_path / "tasks"
    successor = {
        "task_id": "task-a",
        "state": "running",
        "request": {"input": "a.wav"},
        "created_at": 1.0,
        "updated_at": 3.0,
    }
    task_index.merge_write(index, [successor], tasks)

    task_index.merge_write(
        index,
        [{**successor, "state": "completed", "updated_at": 4.0}],
        tasks,
        preserve_running_after={"task-a": 2.0},
    )

    stored = task_index.read(index, tasks)[0]
    assert stored["state"] == "running"
    assert stored["updated_at"] == 3.0


def test_the_stem_prefers_the_chosen_name_over_the_source() -> None:
    assert task_output.task_stem(name="我的切片", source="D:/media/raw.mp4") == "我的切片"
    assert task_output.task_stem(source="https://example.test/v") == "v"
    assert task_output.task_stem(name="   ", source="D:/media/raw.mp4") == "raw"


def test_a_stem_that_cannot_name_a_file_falls_back() -> None:
    # Not an empty string: that produces a bare ".srt", or a directory entry
    # with no name at all under the task root.
    assert "?" not in task_output.task_stem(source="C:/a/what?.mp4")
    assert task_output.task_stem(source="C:/a/....mp4") == "subtitle"
    assert task_output.task_stem(name="trailing. ") == "trailing"
    assert len(task_output.task_stem(name="x" * 200)) == 80


def test_the_task_id_orders_and_names_and_separates() -> None:
    at = time.mktime((2026, 8, 11, 22, 5, 0, 0, 0, -1))

    first = task_output.new_task_id("clip", now=at)
    second = task_output.new_task_id("clip", now=at)

    assert first.startswith("clip-260811-2205-")
    # Same source, same minute: the stem and timestamp alone cannot tell two
    # runs apart, which is the whole reason for the suffix.
    assert first != second


def test_an_absolute_request_is_honoured_as_given(tmp_path: Path) -> None:
    output = task_output.resolve_task_output(
        tmp_path / "tasks",
        "clip-260811-2205-abc123",
        requested=tmp_path / "elsewhere" / "mine.srt",
        stem="clip",
    )

    assert output == (tmp_path / "elsewhere" / "mine.srt").resolve()


def test_a_relative_request_is_a_file_name_under_the_task(tmp_path: Path) -> None:
    # Not a path to follow: it arrives from a front end whose working directory
    # is not the user's, so `../` would land the output somewhere neither chose.
    output = task_output.resolve_task_output(
        tmp_path / "tasks",
        "clip-260811-2205-abc123",
        requested="../../mine.srt",
        stem="clip",
    )

    assert output == (
        tmp_path / "tasks" / "clip-260811-2205-abc123" / "mine.srt"
    ).resolve()


def test_without_a_request_the_stem_names_the_file(tmp_path: Path) -> None:
    for requested in (None, ""):
        output = task_output.resolve_task_output(
            tmp_path / "tasks",
            "clip-260811-2205-abc123",
            requested=requested,
            stem="clip",
        )

        assert output == (
            tmp_path / "tasks" / "clip-260811-2205-abc123" / "clip.srt"
        ).resolve()


def test_a_busy_index_is_not_mistaken_for_a_corrupt_one(tmp_path: Path) -> None:
    """The failure this guards is losing everyone else's history.

    The index is shared, so a reader can meet it mid-replace and a scanner can
    hold it open. Treating that like invalid content would archive the file and
    write back an index holding only the caller's own entry -- for one unlucky
    moment of I/O.
    """

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    task_index.merge_write(
        index, [{"task_id": "earlier", "updated_at": 1.0}], tmp_path / "tasks"
    )

    original_read_bytes = Path.read_bytes

    def refuse(self, *args, **kwargs):
        if self == index:
            raise PermissionError(32, "being used by another process")
        return original_read_bytes(self, *args, **kwargs)

    Path.read_bytes = refuse
    try:
        with pytest.raises(OSError):
            task_index.merge_write(
                index, [{"task_id": "mine", "updated_at": 2.0}], tmp_path / "tasks"
            )
    finally:
        Path.read_bytes = original_read_bytes

    # Nothing archived, nothing lost: the caller logs and the next write retries.
    assert not list(tmp_path.glob("tasks.json.invalid*"))
    assert [entry["task_id"] for entry in task_index.read(index, tmp_path / "tasks")] == [
        "earlier"
    ]


def test_a_reader_retries_an_index_that_is_being_replaced(tmp_path: Path) -> None:
    """The other front end replacing the file must not read as "no history".

    `os.replace` and an open handle collide on Windows for as long as the
    handle lives. Answering that with an empty list makes the window blink
    empty -- and, where a caller asks "is this output ours", makes it answer no.
    """

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    task_index.merge_write(
        index, [{"task_id": "earlier", "updated_at": 1.0}], tmp_path / "tasks"
    )

    original_read_bytes = Path.read_bytes
    refusals = {"left": 2}

    def refuse_twice(self, *args, **kwargs):
        if self == index and refusals["left"]:
            refusals["left"] -= 1
            raise PermissionError(32, "being used by another process")
        return original_read_bytes(self, *args, **kwargs)

    Path.read_bytes = refuse_twice
    try:
        stored = task_index.read(index, tmp_path / "tasks")
    finally:
        Path.read_bytes = original_read_bytes

    assert refusals["left"] == 0
    assert [entry["task_id"] for entry in stored] == ["earlier"]
