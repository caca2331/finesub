from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from finesub_bootstrap import fsops
from finesub_bootstrap.locks import LockUnavailable, holding_lock, try_lock


def _link_directory(link: Path, target: Path) -> None:
    """Point `link` at `target`, however this platform allows it."""

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        if result.returncode == 0:
            return
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform gate
        pytest.skip("this platform will not create directory links")


def test_remove_tree_deletes_a_link_and_not_what_it_points_at(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep-me").write_text("not ours to delete", encoding="utf-8")
    link = tmp_path / "link"
    _link_directory(link, elsewhere)

    fsops.remove_tree(link)

    assert not os.path.lexists(link)
    assert (elsewhere / "keep-me").is_file()


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

    monkeypatch.setattr(fsops.shutil, "copytree", explode)

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
