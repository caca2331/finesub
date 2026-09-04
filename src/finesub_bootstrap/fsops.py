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
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import time


@dataclass(frozen=True)
class ReplaceBudget:
    """How long a rename waits out whoever is still holding a name."""

    attempts: int
    backoff_seconds: float
    cap_seconds: float


#: Publishing a tree or a downloaded file. Windows denies a rename while any
#: handle is open on either name, and the holder is almost always something
#: that lets go on its own: an antivirus reading the 2.8GB just written, a sync
#: client, an ordinary reader (Python's `open` does not share deletion). Those
#: windows are short; the budget exists so a sub-second overlap cannot discard
#: minutes of download. Worst case is ~10s, paid once.
PUBLISH_REPLACE = ReplaceBudget(attempts=8, backoff_seconds=0.4, cap_seconds=2.0)

#: Records -- an index, a ledger, a `.env`. Deliberately much shorter: a tree
#: is published once and losing it means downloading again, while a record is
#: rewritten on every status update, so a name that stayed locked would make
#: every later write pay the full budget. Losing one entry until the next write
#: is cheaper than stalling each write for seconds.
RECORD_REPLACE = ReplaceBudget(attempts=4, backoff_seconds=0.05, cap_seconds=0.2)


def replace_path(
    source: Path | str,
    destination: Path | str,
    *,
    budget: ReplaceBudget = PUBLISH_REPLACE,
) -> None:
    """`os.replace`, waiting out a handle that is about to be released.

    Every publish in the installer ends in a rename, and on Windows that is the
    step most likely to fail for a reason unrelated to the work: the bytes are
    already written and verified, and something is merely *reading* one of the
    two names. A resource activation dies with `[WinError 5]` and discards a
    multi-minute download; the desktop then shows that line, which tells a user
    nothing they can act on.

    Not every rename belongs here. Two in this module stay bare on purpose,
    because their failure carries information rather than costing work: the
    same-volume probe in `move_directory`, whose `OSError` *means* "different
    volume" and is the signal to fall through to the copy, and the quarantine
    rename after a digest mismatch. A bounded wait also cannot outlast a reader
    that reopens the name in a loop -- that is a starved writer, a different
    problem, and no caller here polls that way.
    """

    for attempt in range(1, budget.attempts + 1):
        try:
            os.replace(source, destination)
            return
        except OSError:
            if attempt == budget.attempts:
                raise
            time.sleep(
                min(budget.backoff_seconds * attempt, budget.cap_seconds)
            )


def is_directory_link(path: Path) -> bool:
    """Whether `path` is a symlink or a junction (which `is_symlink` misses)."""

    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction(path)) if is_junction is not None else False


def _unlink_directory_link(path: Path) -> None:
    """Remove the link itself, never the tree it points at."""

    try:
        path.rmdir()
    except OSError:
        path.unlink()


def read_directory_link(path: Path) -> str | None:
    """Where a junction or symlink points, or None if `path` is neither."""

    if not is_directory_link(path):
        return None
    try:
        return os.readlink(path)
    except OSError:
        return None


def create_directory_link(path: Path, target: str, *, symlink: bool = False) -> None:
    r"""Recreate a directory link of the kind it came from.

    No stdlib call makes a junction, hence `mklink /J`. Junctions are the kind
    that matters here: a symlink needs administrator rights or developer mode,
    so the redirects users actually manage to create are junctions. The target
    is passed through exactly as `os.readlink` returned it, `\\?\` prefix and
    all -- `mklink` accepts that form and the new link reads back byte-identical
    to the original, so there is no prefix parsing here to get wrong.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if symlink or os.name != "nt":
        os.symlink(target, path, target_is_directory=True)
        return
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(path), target],
        capture_output=True,
    )
    if result.returncode != 0:
        raise OSError(
            f"Could not recreate the directory link {path} -> {target}: "
            f"{result.stdout.decode(errors='replace').strip()}"
        )


def _directory_links(root: Path) -> list[tuple[Path, str, bool]]:
    """Every directory link under `root`: where it sits, points, and its kind.

    Never descends through one, so a link inside a link is not reported -- it
    belongs to whoever owns the target, and recreating the outer link brings it
    along anyway.
    """

    found: list[tuple[Path, str, bool]] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            target = read_directory_link(entry)
            if target is not None:
                found.append((entry.relative_to(root), target, entry.is_symlink()))
            elif entry.is_dir():
                stack.append(entry)
    return found


def remove_tree(path: Path) -> None:
    """Delete a directory we own, without ever following a link out of it.

    The descent is ours rather than `shutil.rmtree`'s because that function
    only stopped recursing into junctions in CPython 3.12. The desktop app
    pins 3.12, but the CLI wheel declares `requires-python >= 3.10` and runs
    `finesub_bootstrap.shell` -- including `uninstall` -- on the *launcher's*
    interpreter, so on a 3.10/3.11 machine a junction nested anywhere under the
    target would be followed and someone else's data deleted. Users do make
    such links (redirecting `models`/`cache`/`tasks` off the system drive is a
    documented setup), so the guarantee has to belong to this code and not to
    whichever interpreter happens to import it.
    """

    if not os.path.lexists(path):
        return
    if is_directory_link(path):
        _unlink_directory_link(path)
        return
    for entry in path.iterdir():
        if is_directory_link(entry):
            _unlink_directory_link(entry)
        elif entry.is_dir():
            remove_tree(entry)
        else:
            entry.unlink(missing_ok=True)
    path.rmdir()


def _tree_summary(path: Path) -> tuple[int, int]:
    """Count the files this tree owns, without counting through a link.

    `os.walk` descends into a junction, since `is_symlink` is False for one, so
    the plain version of this counted whatever the user had redirected part of
    the store to. `copy_tree` deliberately recreates such links instead of
    duplicating what they point at, so counting through them would make the
    verification in `move_directory` fail on precisely the setup it exists to
    preserve.
    """

    if is_directory_link(path):
        return 0, 0
    files = 0
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            entry_path = Path(entry.path)
            if is_directory_link(entry_path):
                continue
            try:
                if entry.is_dir():
                    stack.append(entry_path)
                    continue
                total += entry.stat().st_size
            except OSError:
                continue
            files += 1
    return files, total


def copy_tree(source: Path, destination: Path) -> None:
    """Copy a directory, keeping directory links as links.

    `robocopy` is on every Windows and is far faster than walking the tree in
    Python for the multi-GB directories this moves. Its exit codes are a
    bitmask where anything below 8 means "copied, possibly with extras"; 8 and
    above are real failures. `/W:1` matters as much as the speed: the default
    is a 30-second wait per retry, so one file held open by a scanner would
    look like a hang.

    `/XJ` and the recreation afterwards are what make a relocate across volumes
    agree with one within a volume, where `os.replace` moves the link itself.
    Without them robocopy walks through a junction and writes a second physical
    copy at the destination: the user's redirect is silently gone and the data
    it pointed at is stored twice. Redirecting `models`/`cache`/`tasks` off the
    system drive is a documented setup, so that is a normal store, not an
    exotic one. `shutil.copytree` needs the same treatment for the same reason
    `remove_tree` does -- `symlinks=True` only covers real symlinks, and a
    junction is not one.
    """

    link_target = read_directory_link(source)
    if link_target is not None:
        # The directory *is* the redirect. Copying it would mean copying a tree
        # that belongs to someone else, which is also the one robocopy case
        # `/XJ` does not cover: it excludes junctions it meets on the way down,
        # not the source root it was pointed at.
        remove_tree(destination)
        create_directory_link(destination, link_target, symlink=source.is_symlink())
        return
    nested = _directory_links(source)
    if os.name == "nt" and shutil.which("robocopy"):
        result = subprocess.run(
            [
                "robocopy",
                str(source),
                str(destination),
                "/E",
                "/J",
                "/XJ",
                "/R:1",
                "/W:1",
                "/NFL",
                "/NDL",
                "/NP",
            ],
            capture_output=True,
        )
        if result.returncode >= 8:
            raise OSError(
                f"robocopy failed with exit code {result.returncode} copying {source}"
            )
    else:
        shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
    for relative, target, symlink in nested:
        placed = destination / relative
        # The fallback copies straight through a junction, so a real directory
        # may be sitting where the link belongs.
        remove_tree(placed)
        create_directory_link(placed, target, symlink=symlink)


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
        # Bare on purpose: this failure is the volume probe. `replace_path`
        # would spend its budget waiting for a name that is not held, on every
        # cross-volume move.
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
        replace_path(staging, destination)
    except BaseException:
        remove_tree(staging)
        raise
    return True, source


def write_atomic(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
) -> None:
    """Write a small file so a crash cannot leave it half-written.

    Temp file in the same directory, then `os.replace` -- the rename is atomic
    within a volume, so readers see either the old content or the new one and
    never a truncated middle. This idiom was hand-rolled in seven places and
    skipped in six others; the ones that skipped it are the records whose loss
    is expensive. `pyvenv.cfg` is the sharpest example: a torn write there
    removes the `home` line, the managed interpreter can no longer find its
    stdlib, and the fix is a multi-GB rebuild -- caused by the very routine
    that exists to avoid one.

    For records that must survive a crash *and* concurrent writers, take the
    lock first (see `locks.holding_lock`); this only guarantees atomicity.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(text, encoding=encoding, newline=newline)
        replace_path(temporary, path, budget=RECORD_REPLACE)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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

    Shares `copy_tree` with the store move so both treat directory links the
    same way; a private `shutil.copytree` here would copy through a junction
    while the summary below stopped at it, and the mismatch would abort a
    migration that had nothing wrong with it.
    """

    if destination.exists():
        raise FileExistsError(f"Move destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.incoming")
    remove_tree(staging)
    try:
        copy_tree(source, staging)
        if _tree_summary(source) != _tree_summary(staging):
            raise OSError(
                f"Copy of {source} did not match the source; nothing was moved"
            )
        replace_path(staging, destination)
    except BaseException:
        remove_tree(staging)
        raise
    remove_tree(source)
