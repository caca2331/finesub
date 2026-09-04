"""Who holds a task lock, for the message a person reads."""

from __future__ import annotations

import json
from pathlib import Path
import time

from finesub_bootstrap import locks


def test_the_lease_lives_and_dies_with_the_lock(tmp_path: Path) -> None:
    lock = locks.task_lock_path(tmp_path, "task-a")

    with locks.holding_lock(lock, lease=locks.lease_record("task-a", "cli")):
        record = locks.read_lease(lock)
        assert record is not None
        assert record["frontend"] == "cli"
        assert record["task_id"] == "task-a"

    assert locks.read_lease(lock) is None


def test_a_lock_without_a_lease_is_ordinary_not_broken(tmp_path: Path) -> None:
    """Older versions, and holders that could not write, leave nothing.

    The lock still answers "busy"; only the name is missing, and the caller
    says so in plainer words rather than refusing to answer.
    """

    lock = locks.task_lock_path(tmp_path, "task-b")

    with locks.holding_lock(lock):
        assert locks.read_lease(lock) is None
        assert locks.describe_lease(locks.read_lease(lock)) == ""


def test_a_lease_that_cannot_be_removed_does_not_fail_the_task(
    tmp_path: Path, monkeypatch
) -> None:
    """Removal is best effort like the write: a denied delete (the file held
    open by an antivirus scan, say) must not turn a finished task into a
    failure or mask its real result."""

    lock = locks.task_lock_path(tmp_path, "task-e")
    original_unlink = Path.unlink

    def deny_json_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".json":
            raise PermissionError("held open by another process")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", deny_json_unlink)

    with locks.holding_lock(lock, lease=locks.lease_record("task-e", "cli")):
        pass

    # The sidecar survived, and that is fine: the lock is free, so readers
    # treat the leftover name as commentary rather than as "busy".
    assert locks.read_lease(lock) is not None
    assert locks.try_lock(lock)


def test_a_damaged_lease_reads_as_absent(tmp_path: Path) -> None:
    lock = locks.task_lock_path(tmp_path, "task-c")
    lock.parent.mkdir(parents=True, exist_ok=True)
    locks.lease_path(lock).write_text("{not json", encoding="utf-8")

    assert locks.read_lease(lock) is None


def test_the_description_names_the_front_end_and_the_process(tmp_path: Path) -> None:
    lock = locks.task_lock_path(tmp_path, "task-d")
    locks.lease_path(lock).parent.mkdir(parents=True, exist_ok=True)
    locks.lease_path(lock).write_text(
        json.dumps(
            {
                "task_id": "task-d",
                "frontend": "cli",
                "pid": 4321,
                "host": "elsewhere",
                "started_at": 1_754_640_000.0,
            }
        ),
        encoding="utf-8",
    )

    said = locks.describe_lease(locks.read_lease(lock))

    assert "命令行" in said
    assert "pid 4321" in said
    # A different machine is worth naming: the pid means nothing here.
    assert "elsewhere" in said


def test_writing_the_lease_is_never_worth_failing_a_task(
    tmp_path: Path, monkeypatch
) -> None:
    lock = locks.task_lock_path(tmp_path, "task-e")

    def _refuse(*_args, **_kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "write_text", _refuse)

    with locks.holding_lock(lock, lease=locks.lease_record("task-e", "cli")):
        assert locks.read_lease(lock) is None


def test_the_survey_names_held_tasks_and_ignores_free_ones(tmp_path: Path) -> None:
    free = locks.task_lock_path(tmp_path, "finished")
    with locks.holding_lock(free, lease=locks.lease_record("finished", "cli")):
        pass

    with locks.holding_lock(
        locks.task_lock_path(tmp_path, "busy"),
        lease=locks.lease_record("busy", "cli"),
    ):
        held = locks.held_task_leases(tmp_path)

    assert [record["task_id"] for _, record in held] == ["busy"]
    # The finished task left its sidecar behind -- locks are never deleted --
    # and must not be mistaken for a holder.
    assert free.is_file()


def test_the_survey_reports_a_held_lock_that_left_no_name(tmp_path: Path) -> None:
    with locks.holding_lock(locks.task_lock_path(tmp_path, "anonymous")):
        held = locks.held_task_leases(tmp_path)

    assert len(held) == 1
    assert held[0][1] is None


def test_the_activity_gate_is_not_a_task(tmp_path: Path) -> None:
    """It shares the `.task-` prefix, and in some layouts the directory too."""

    with locks.holding_lock(locks.activity_gate_path(tmp_path)):
        assert locks.held_task_leases(tmp_path) == []


def test_counting_runs_never_deletes_what_it_counts(tmp_path: Path) -> None:
    """The barrier sweeps free leases; a diagnostic reports and leaves."""

    with locks.holding_activity(tmp_path, lease_id="one"):
        with locks.holding_activity(tmp_path, lease_id="two"):
            assert locks.active_run_count(tmp_path) == 2

    stale = locks.activity_lease_path(tmp_path, "crashed")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"")

    assert locks.active_run_count(tmp_path) == 0
    assert stale.is_file()


def test_a_holder_from_another_day_is_dated(tmp_path: Path) -> None:
    """A hung holder can sit for days; a bare clock time would read as today."""

    lock = locks.task_lock_path(tmp_path, "task-old")
    lock.parent.mkdir(parents=True, exist_ok=True)
    two_days_ago = time.time() - 2 * 24 * 3600
    locks.lease_path(lock).write_text(
        json.dumps({"frontend": "cli", "started_at": two_days_ago}),
        encoding="utf-8",
    )

    said = locks.describe_lease(locks.read_lease(lock))

    assert time.strftime("%m-%d", time.localtime(two_days_ago)) in said


def test_a_holder_from_today_is_just_a_clock_time(tmp_path: Path) -> None:
    lock = locks.task_lock_path(tmp_path, "task-now")

    with locks.holding_lock(lock, lease=locks.lease_record("task-now", "cli")):
        said = locks.describe_lease(locks.read_lease(lock))

    assert "自 " + time.strftime("%H:%M", time.localtime()) in said


def test_a_front_end_this_build_does_not_know_is_named_as_it_named_itself(
    tmp_path: Path,
) -> None:
    """A 0.4.x desktop shares the user-data tree and still writes
    `frontend=desktop`; the label table no longer knows it, and the word
    itself is a better answer than "another process"."""

    lock = locks.task_lock_path(tmp_path, "task-e")
    locks.lease_path(lock).parent.mkdir(parents=True, exist_ok=True)
    locks.lease_path(lock).write_text(
        json.dumps({"task_id": "task-e", "frontend": "desktop", "pid": 7}),
        encoding="utf-8",
    )

    said = locks.describe_lease(locks.read_lease(lock))

    assert said.startswith("desktop")
    assert "pid 7" in said
