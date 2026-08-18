"""The link-semantics half of the `fsops` tests.

Only what the Windows runner can really execute stays here: every case is
about directory links -- junctions on Windows, where `remove_tree` and
robocopy's `/XJ` are the behaviour under test. The platform-neutral cases
(move failure atomicity, locks, `write_atomic`) live in
`test/bootstrap/test_fsops.py`, where the pre-commit `pytest -q` runs them.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from finesub_bootstrap import fsops


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


def test_remove_tree_does_not_follow_a_nested_link(tmp_path: Path) -> None:
    """The guarantee must not depend on the interpreter version.

    `shutil.rmtree` only stopped recursing into junctions in CPython 3.12, and
    the CLI wheel declares `requires-python >= 3.10` while running `uninstall`
    on the launcher's own interpreter. Redirecting `models`/`cache`/`tasks` off
    the system drive is a documented setup, so a nested link is exactly what a
    real uninstall meets.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep-me").write_text("not ours to delete", encoding="utf-8")

    ours = tmp_path / "ours"
    (ours / "deep").mkdir(parents=True)
    (ours / "deep" / "own.txt").write_text("ours", encoding="utf-8")
    _link_directory(ours / "deep" / "redirected", elsewhere)

    fsops.remove_tree(ours)

    assert not ours.exists()
    assert (elsewhere / "keep-me").is_file(), "followed the nested link"


def _force_cross_volume(monkeypatch, source: Path) -> None:
    """Take `move_directory`'s copy path without needing a second drive."""

    real = fsops.os.replace

    def replace(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(src) == source:
            raise OSError(18, "Invalid cross-device link")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(fsops.os, "replace", replace)


def test_a_cross_volume_move_keeps_a_nested_link_a_link(
    tmp_path: Path, monkeypatch
) -> None:
    """Relocating a store must not duplicate what the user redirected out of it.

    Within one volume the move is `os.replace`, which moves the link itself, so
    a redirect survives untouched. Across volumes robocopy used to walk through
    the junction: the redirect vanished and its contents were written a second
    time at the destination. The two paths now agree.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "weights.bin").write_bytes(b"W" * 4096)

    source = tmp_path / "store" / "models"
    (source / "own").mkdir(parents=True)
    (source / "own" / "ours.bin").write_bytes(b"O" * 10)
    _link_directory(source / "redirected", elsewhere)

    destination = tmp_path / "moved" / "models"
    _force_cross_volume(monkeypatch, source)

    placed, leftover = fsops.move_directory(source, destination)

    assert placed and leftover == source
    assert (destination / "own" / "ours.bin").read_bytes() == b"O" * 10
    assert fsops.is_directory_link(destination / "redirected"), "materialised it"
    assert (destination / "redirected" / "weights.bin").read_bytes() == b"W" * 4096
    # One physical copy, not two: only the store's own file is counted.
    assert fsops._tree_summary(destination) == (1, 10)

    fsops.remove_tree(leftover)
    assert (elsewhere / "weights.bin").is_file(), "deleted the redirect's target"


def test_a_cross_volume_move_of_a_redirected_directory_moves_the_redirect(
    tmp_path: Path, monkeypatch
) -> None:
    """`/XJ` excludes junctions met on the way down, not the source root."""

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "weights.bin").write_bytes(b"W" * 4096)

    source = tmp_path / "store" / "models"
    source.parent.mkdir(parents=True)
    _link_directory(source, elsewhere)

    destination = tmp_path / "moved" / "models"
    _force_cross_volume(monkeypatch, source)

    placed, leftover = fsops.move_directory(source, destination)

    assert placed and leftover == source
    assert fsops.is_directory_link(destination), "copied someone else's tree"
    assert (destination / "weights.bin").read_bytes() == b"W" * 4096

    fsops.remove_tree(leftover)
    assert not os.path.lexists(source)
    assert (elsewhere / "weights.bin").is_file(), "deleted the redirect's target"
