"""Pick the installation back up when a full update did not finish.

A full update empties the install root into `.update/backup-<version>/` and
then copies the new tree in. For the seconds that window is open there is no
executable in the install root at all, and nothing used to look at `.update/`
afterwards: a machine that lost power there had a shortcut pointing at a file
that no longer existed, the only copy of the program sitting in `backup-*`, and
the next update attempt deleting that copy before doing anything else.

So the first thing a start-up does now is check whether the last update left
the install root standing, and put it back if not.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil

from finesub_bootstrap.fsops import remove_tree

UPDATE_DIRECTORY_NAME = ".update"
BACKUP_PREFIX = "backup-"
#: Present in every healthy install root; its absence is what "unbootable"
#: means here. Kept in sync with `updates.service.MAIN_EXECUTABLE_NAME`.
MAIN_EXECUTABLE_NAME = "FineSub Desktop.exe"

LogCallback = Callable[[str], None]


def backups(root: Path) -> list[Path]:
    """Every interrupted-update backup, newest first."""

    update_root = root / UPDATE_DIRECTORY_NAME
    if not update_root.is_dir():
        return []
    found = [
        entry
        for entry in update_root.iterdir()
        if entry.is_dir() and entry.name.startswith(BACKUP_PREFIX)
    ]
    return sorted(found, key=lambda entry: entry.stat().st_mtime, reverse=True)


def install_is_bootable(root: Path) -> bool:
    return (root / MAIN_EXECUTABLE_NAME).is_file()


def recover_interrupted_update(
    root: Path,
    *,
    log: LogCallback | None = None,
) -> str | None:
    """Restore the previous version if an update left the root unbootable.

    Returns a message for the user when something was actually done, so the
    app can say what happened rather than silently appearing to downgrade.

    Restoring is automatic because the alternative is not a choice anybody can
    make: with no executable in the install root, there is no interface left to
    offer them one.
    """

    if install_is_bootable(root):
        return None
    available = backups(root)
    if not available:
        return None
    backup = available[0]
    restored: list[str] = []
    for entry in list(backup.iterdir()):
        destination = root / entry.name
        if destination.exists():
            # The new tree already put something here; the backup copy is the
            # stale one. Leave what is in place and drop ours.
            continue
        try:
            shutil.move(str(entry), str(destination))
            restored.append(entry.name)
        except OSError:
            if log is not None:
                log(f"无法恢复 {entry.name}，请手工从 {backup} 取回")
    if not restored:
        return None
    message = (
        f"上次更新未完成，已恢复到更新前的版本（{backup.name}）。"
        "可以重新尝试更新。"
    )
    if log is not None:
        log(message)
    return message


def take_update_error_reports(root: Path) -> list[str]:
    """Read and archive whatever the updater left behind, so it is seen once.

    The updater is a windowed build with no console and no reader: its only way
    to report anything is `<request>.error.txt`, and until now nothing in the
    product ever opened one. A user whose update failed saw the app start on
    the old version with no explanation, indefinitely.
    """

    update_root = root / UPDATE_DIRECTORY_NAME
    if not update_root.is_dir():
        return []
    reports: list[str] = []
    for report in sorted(update_root.glob("*.error.txt")):
        try:
            reports.append(report.read_text(encoding="utf-8", errors="replace"))
            report.replace(report.with_suffix(".txt.seen"))
        except OSError:
            continue
    return reports


def discard_backups(root: Path, *, keep: Path | None = None) -> None:
    """Delete update backups -- only ever called once the root is bootable.

    The old code cleared `source-`/`backup-`/`runner-` at the *start* of the
    next attempt, which is exactly when the previous backup may still be the
    only copy of a working installation.
    """

    if not install_is_bootable(root):
        return
    for backup in backups(root):
        if keep is not None and backup == keep:
            continue
        remove_tree(backup)
