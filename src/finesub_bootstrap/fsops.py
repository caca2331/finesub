"""Filesystem operations that survive Windows and the users we actually have.

Two things go wrong repeatedly in this project, so they live in one place:
directory links (a `shutil.rmtree` walks straight through a junction into
whatever the user redirected a directory to), and interrupted moves (a
cross-volume `shutil.move` is copy-then-delete, so a crash leaves half a tree
at the destination and turns one bad moment into a state a human has to
untangle).
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import subprocess


def is_directory_link(path: Path) -> bool:
    """Whether `path` is a symlink or a junction (which `is_symlink` misses)."""

    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction(path)) if is_junction is not None else False


def remove_tree(path: Path) -> None:
    """Delete a directory we own, without ever following a link out of it."""

    if not os.path.lexists(path):
        return
    if is_directory_link(path):
        # The link, not the tree it points at: rmtree would walk straight into
        # whatever the user redirected this to.
        try:
            path.rmdir()
        except OSError:
            path.unlink()
        return
    shutil.rmtree(path)


def _tree_summary(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for current, _directories, names in os.walk(path):
        for name in names:
            try:
                total += os.stat(os.path.join(current, name)).st_size
            except OSError:
                continue
            files += 1
    return files, total


def copy_tree(source: Path, destination: Path) -> None:
    """Copy a directory, using robocopy when it is available.

    `robocopy` is on every Windows and is far faster than walking the tree in
    Python for the multi-GB directories this moves. Its exit codes are a
    bitmask where anything below 8 means "copied, possibly with extras"; 8 and
    above are real failures. `/W:1` matters as much as the speed: the default
    is a 30-second wait per retry, so one file held open by a scanner would
    look like a hang.
    """

    if os.name == "nt" and shutil.which("robocopy"):
        result = subprocess.run(
            [
                "robocopy",
                str(source),
                str(destination),
                "/E",
                "/J",
                "/R:1",
                "/W:1",
                "/NFL",
                "/NDL",
                "/NP",
            ],
            capture_output=True,
        )
        if result.returncode < 8:
            return
        raise OSError(
            f"robocopy failed with exit code {result.returncode} copying {source}"
        )
    shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)


def move_directory(source: Path, destination: Path) -> tuple[bool, Path | None]:
    """Place one directory at `destination`.

    Returns whether anything was placed there, and a source still waiting to be
    removed (cross-volume only).

    Within one volume this is a rename, which keeps the hardlinks the download
    cache shares with the runtime, and leaves no source behind. Across volumes
    the copy is verified before anything is released, and the source is
    returned rather than deleted so the caller can record the new location
    first -- a crash between the two should cost a copy, not the data.

    Either way the destination is complete or absent, never half-populated.
    """

    if not source.is_dir() or not any(source.iterdir()):
        return False, None
    if destination.is_dir() and any(destination.iterdir()):
        # Adopting a store that already holds this: merging two of them is the
        # user's decision, not ours.
        return False, None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        remove_tree(destination)
    try:
        os.replace(source, destination)
        return True, None
    except OSError:
        pass  # Different volume: fall through to copy-verify-then-release.
    staging = destination.with_name(f"{destination.name}.incoming")
    remove_tree(staging)
    try:
        copy_tree(source, staging)
        if _tree_summary(source) != _tree_summary(staging):
            raise OSError(
                f"Copy of {source} did not match the source; nothing was moved"
            )
        os.replace(staging, destination)
    except BaseException:
        remove_tree(staging)
        raise
    return True, source


def move_store(
    source_root: Path,
    destination_root: Path,
    names: Sequence[str],
) -> tuple[list[str], list[Path]]:
    """Move the named directories between big-data roots.

    Moves the contents rather than the root itself: by default the big-data
    root *is* the installation directory, and taking that wholesale would drag
    the application and its runtime along with the models. Returns what was
    moved and which sources are still waiting to be removed.
    """

    moved: list[str] = []
    leftovers: list[Path] = []
    for name in names:
        placed, leftover = move_directory(source_root / name, destination_root / name)
        if placed:
            moved.append(name)
        if leftover is not None:
            leftovers.append(leftover)
    return moved, leftovers


def move_tree(source: Path, destination: Path) -> None:
    """Move a directory so that no crash can leave a half-populated destination.

    Copies to a sibling staging directory of the destination, compares the two
    trees, and only then renames into place -- a rename within one volume is
    atomic, so the destination either does not exist or is complete. The source
    is removed last, which is why an interrupted run costs disk rather than
    data.
    """

    if destination.exists():
        raise FileExistsError(f"Move destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.incoming")
    remove_tree(staging)
    try:
        shutil.copytree(source, staging, symlinks=True)
        if _tree_summary(source) != _tree_summary(staging):
            raise OSError(
                f"Copy of {source} did not match the source; nothing was moved"
            )
        os.replace(staging, destination)
    except BaseException:
        remove_tree(staging)
        raise
    remove_tree(source)
