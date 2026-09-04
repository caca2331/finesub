"""The platform-neutral half of the `fsops`/`locks` tests.

Split from `desktop/backend/tests/test_fsops.py`: what stays there is what the
Windows runner alone can really execute -- junction and robocopy semantics,
which are the module's reason to exist. What lives here is the behaviour that
holds on any platform: a failed move leaves nothing half-copied, a held lock
blocks a second holder, an atomic write never strands a temp file. This is the
part a pre-commit `pytest -q` should catch.
"""

from __future__ import annotations

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


def _failing_replace(failures: int, calls: list[tuple[str, str]]):
    """An `os.replace` that is denied `failures` times, then works."""

    real = fsops.os.replace

    def attempt(source, destination):  # type: ignore[no-untyped-def]
        calls.append((str(source), str(destination)))
        if len(calls) <= failures:
            raise PermissionError(5, "Access is denied")
        return real(source, destination)

    return attempt


def test_a_publishing_rename_waits_out_a_handle_that_lets_go(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # The whole point: the bytes are already written and something is merely
    # reading one of the two names for a moment.
    source = tmp_path / "staging"
    source.write_text("payload", encoding="utf-8")
    destination = tmp_path / "published"
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(fsops.time, "sleep", lambda _: None)
    monkeypatch.setattr(fsops.os, "replace", _failing_replace(2, calls))

    fsops.replace_path(source, destination)

    assert len(calls) == 3
    assert destination.read_text(encoding="utf-8") == "payload"


def test_a_record_rename_gives_up_far_sooner_than_a_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # A record is rewritten on every status update, so a name that stayed
    # locked must not make each later write pay a publish-sized budget.
    monkeypatch.setattr(fsops.time, "sleep", lambda _: None)
    for budget, expected in (
        (fsops.PUBLISH_REPLACE, fsops.PUBLISH_REPLACE.attempts),
        (fsops.RECORD_REPLACE, fsops.RECORD_REPLACE.attempts),
    ):
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(fsops.os, "replace", _failing_replace(999, calls))
        with pytest.raises(PermissionError):
            fsops.replace_path(tmp_path / "a", tmp_path / "b", budget=budget)
        assert len(calls) == expected
    assert fsops.RECORD_REPLACE.attempts < fsops.PUBLISH_REPLACE.attempts


def test_the_denial_itself_survives_a_rename_that_never_clears(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # `[WinError 5]` is what tells a person to look for an antivirus; replacing
    # it with a message of our own would remove the only actionable part.
    monkeypatch.setattr(fsops.time, "sleep", lambda _: None)
    monkeypatch.setattr(fsops.os, "replace", _failing_replace(999, []))

    with pytest.raises(PermissionError) as raised:
        fsops.replace_path(tmp_path / "a", tmp_path / "b")

    assert raised.value.errno == 5
    assert "Access is denied" in str(raised.value)


def test_the_cross_volume_probe_does_not_spend_the_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`move_directory`'s first rename is a question, not a publish.

    Its `OSError` *means* "different volume" and is the signal to fall through
    to copy-verify-then-release. Waiting on it would add the full budget to
    every cross-volume move, which is the normal case it exists to serve.
    """

    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("payload", encoding="utf-8")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(fsops.time, "sleep", lambda _: None)
    monkeypatch.setattr(fsops.os, "replace", _failing_replace(999, calls))

    with pytest.raises(PermissionError):
        fsops.move_directory(source, tmp_path / "destination")

    # One for the probe, then the publish of the verified copy spends its own.
    assert len(calls) == 1 + fsops.PUBLISH_REPLACE.attempts
