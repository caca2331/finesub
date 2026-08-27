"""The platform-neutral half of the `fsops`/`locks` tests.

Split from `desktop/backend/tests/test_fsops.py`: what stays there is what the
Windows runner alone can really execute -- junction and robocopy semantics,
which are the module's reason to exist. What lives here is the behaviour that
holds on any platform: a failed move leaves nothing half-copied, a held lock
blocks a second holder, an atomic write never strands a temp file. This is the
part a pre-commit `pytest -q` should catch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from finesub_bootstrap import fsops
from finesub_bootstrap.locks import (
    LockUnavailable,
    holding_activity,
    holding_activity_barrier,
    holding_lock,
    try_lock,
)


def test_move_tree_leaves_nothing_at_the_destination_when_it_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # The whole point: an interrupted move must not leave a half-copy that the
    # next run mistakes for a real directory.
    source = tmp_path / "source"
    (source / "inner").mkdir(parents=True)
    (source / "inner" / "payload.txt").write_text("data", encoding="utf-8")
    destination = tmp_path / "destination"

    def explode(*_args, **_kwargs):
        (tmp_path / "destination.incoming").mkdir(exist_ok=True)
        raise OSError("disk on fire")

    monkeypatch.setattr(fsops, "copy_tree", explode)

    with pytest.raises(OSError):
        fsops.move_tree(source, destination)

    assert not destination.exists()
    assert not (tmp_path / "destination.incoming").exists()
    assert (source / "inner" / "payload.txt").is_file()


def test_move_tree_moves_everything_then_removes_the_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "inner").mkdir(parents=True)
    (source / "inner" / "payload.txt").write_text("data", encoding="utf-8")
    (source / ".git").mkdir()

    fsops.move_tree(source, tmp_path / "destination")

    assert (tmp_path / "destination" / "inner" / "payload.txt").read_text(
        "utf-8"
    ) == "data"
    assert (tmp_path / "destination" / ".git").is_dir()
    assert not source.exists()


def test_move_tree_refuses_an_occupied_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        fsops.move_tree(source, destination)


def test_a_held_lock_blocks_a_second_holder_until_it_times_out(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "thing.lock"
    with holding_lock(lock_path):
        assert not try_lock(lock_path)
        with pytest.raises(LockUnavailable):
            with holding_lock(lock_path, timeout=0.1):
                pass

    assert try_lock(lock_path)
    # The sidecar is never deleted: a fresh file would let a process that still
    # holds the byte lock coexist with one that just created it.
    assert lock_path.is_file()


def test_activity_barrier_sees_every_concurrent_run(tmp_path: Path) -> None:
    """One task ending must not make a different live task disappear."""

    with holding_activity(tmp_path, lease_id="task-b"):
        with holding_activity(tmp_path, lease_id="task-a"):
            with pytest.raises(LockUnavailable):
                with holding_activity_barrier(tmp_path):
                    pass

        # A has ended; B still keeps the destructive-operation barrier shut.
        with pytest.raises(LockUnavailable):
            with holding_activity_barrier(tmp_path):
                pass

    with holding_activity_barrier(tmp_path):
        with pytest.raises(LockUnavailable):
            with holding_activity(
                tmp_path, lease_id="starts-during-move", timeout=0
            ):
                pass


def test_write_atomic_leaves_no_partial_file_or_temp_behind(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "record.json"

    fsops.write_atomic(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'

    # A failing write must not damage the previous content nor strand a temp.
    class Boom(Exception):
        pass

    original = Path.write_text

    def explode(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.name.endswith(".tmp"):
            raise Boom()
        return original(self, *args, **kwargs)

    Path.write_text = explode  # type: ignore[method-assign]
    try:
        with pytest.raises(Boom):
            fsops.write_atomic(target, "replacement")
    finally:
        Path.write_text = original  # type: ignore[method-assign]

    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    assert list(target.parent.glob("*.tmp")) == []


def _replace_that_frees_up(failures: int, calls: list[Path]):
    """An `os.replace` denied `failures` times, then allowed through.

    The shape of the Windows failure this exists for: a handle on one of the
    two names, held by whoever is reading what was just written, gone again a
    moment later.
    """

    real_replace = os.replace

    def replace(source, destination):
        calls.append(Path(source))
        if len(calls) <= failures:
            raise OSError(5, "Access is denied")
        real_replace(source, destination)

    return replace


def test_replace_path_waits_out_a_handle_and_then_publishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "staging"
    source.write_text("published", encoding="utf-8")
    destination = tmp_path / "final"
    calls: list[Path] = []
    slept: list[float] = []
    monkeypatch.setattr(os, "replace", _replace_that_frees_up(3, calls))
    monkeypatch.setattr(fsops.time, "sleep", slept.append)

    fsops.replace_path(source, destination)

    assert destination.read_text(encoding="utf-8") == "published"
    assert len(calls) == 4
    # Linear backoff, capped. Three waits for three denials -- never one after
    # the attempt that worked.
    assert slept == pytest.approx([0.4, 0.8, 1.2])


def test_replace_path_gives_up_and_reports_the_denial_it_saw(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A budget, not a loop: the caller still learns what happened."""

    source = tmp_path / "staging"
    source.write_text("published", encoding="utf-8")
    calls: list[Path] = []
    monkeypatch.setattr(os, "replace", _replace_that_frees_up(999, calls))
    monkeypatch.setattr(fsops.time, "sleep", lambda _seconds: None)

    with pytest.raises(OSError) as failure:
        fsops.replace_path(source, tmp_path / "final")

    assert len(calls) == fsops.REPLACE_ATTEMPTS
    assert failure.value.errno == 5


def test_write_atomic_waits_less_than_a_tree_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A record is rewritten on every status update.

    A tree is published once and losing it costs a re-download, so it is worth
    seconds. If a record's name stays locked, every later write would pay the
    full budget again -- so this one gives up sooner, and leaves no temp file
    behind when it does.
    """

    calls: list[Path] = []
    monkeypatch.setattr(os, "replace", _replace_that_frees_up(999, calls))
    monkeypatch.setattr(fsops.time, "sleep", lambda _seconds: None)

    with pytest.raises(OSError):
        fsops.write_atomic(tmp_path / "record.json", "{}")

    assert len(calls) == fsops.SMALL_FILE_REPLACE_ATTEMPTS
    assert fsops.SMALL_FILE_REPLACE_ATTEMPTS < fsops.REPLACE_ATTEMPTS
    assert not (tmp_path / "record.json.tmp").exists()


def test_the_cross_volume_probe_still_fails_on_the_first_try(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`move_directory` asks `os.replace` a question, and needs the answer now.

    Its first replace is not a publish: a denial *means* "different volume",
    and is how it decides to copy instead. Waiting eight times for that answer
    would add seconds to the path that was never going to succeed.
    """

    source = tmp_path / "source"
    (source / "inner").mkdir(parents=True)
    (source / "inner" / "payload.txt").write_text("data", encoding="utf-8")
    destination = tmp_path / "destination"
    real_replace = os.replace
    probes: list[Path] = []

    def replace(replace_source, replace_destination):
        if Path(replace_source).name.endswith(".incoming"):
            real_replace(replace_source, replace_destination)
            return
        probes.append(Path(replace_source))
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "replace", replace)

    moved, leftover = fsops.move_directory(source, destination)

    assert moved and leftover == source
    assert len(probes) == 1
    assert (destination / "inner" / "payload.txt").read_text(encoding="utf-8") == "data"
