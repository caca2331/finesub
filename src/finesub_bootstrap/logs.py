"""The one log directory, and the one rule for how much of it to keep.

Installs and app sessions both write here. They share the budget on purpose:
"keep the newest hundred files" is a promise about the folder, and two
independent hundreds would be two folders' worth of clutter under one name.
"""

from __future__ import annotations

from pathlib import Path
import sys


LOG_DIR_NAME = "logs"
# Small text files -- a hundred of them is a couple of megabytes and covers
# more history than anyone will look at, without growing forever.
DEFAULT_KEEP = 100


def log_directory(user_data: Path) -> Path:
    return Path(user_data).expanduser() / LOG_DIR_NAME


def prune(directory: Path | None, *, keep: int = DEFAULT_KEEP) -> None:
    """Drop all but the newest `keep` logs, whatever wrote them."""

    if directory is None:
        return
    try:
        # By modification time, not by name: the names are
        # install-<resource>-<stamp> and session-<stamp>, so sorting them puts
        # the prefix ahead of the timestamp and "the newest hundred" would mean
        # the hundred belonging to whichever prefix sorts last.
        logs = sorted(
            Path(directory).glob("*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in logs[keep:]:
        try:
            stale.unlink()
        except OSError:
            # Someone has it open; the next prune will try again.
            pass


def open_log(directory: Path | None, name: str):
    """Open `name` in `directory` for appending, or return None.

    Never fatal: a log that cannot be written is not a reason to fail whatever
    was going to be logged.
    """

    if directory is None:
        return None
    path = Path(directory) / name
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", errors="replace")
    except OSError as error:
        print(f"Warning: cannot write the log {name}: {error}", file=sys.stderr)
        return None
