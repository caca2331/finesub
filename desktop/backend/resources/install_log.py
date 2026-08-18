"""Durable transcript of one resource installation.

The snapshot the interface polls keeps only the last hundred lines and dies
with the window, which is the wrong half of the problem: an install that fails
does so after minutes of output nobody was watching, and "please send the log"
is the first thing anyone asks. So every install also writes a file, in full,
as it goes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import TextIO

# Re-exported: callers reach the shared log directory and its pruning rule
# through this module, and `finesub_bootstrap.logs` is where they live.
from finesub_bootstrap.logs import (  # noqa: F401
    log_directory,
    open_log,
    prune,
)


class InstallLog:
    """A single install's log file. Never fatal: a broken log is not a broken install."""

    def __init__(self, path: Path | None, handle: TextIO | None) -> None:
        self.path = path
        self._handle = handle

    @classmethod
    def open(
        cls,
        directory: Path | None,
        resource_id: str,
        *,
        now: datetime | None = None,
    ) -> "InstallLog":
        if directory is None:
            return cls(None, None)
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        name = f"install-{resource_id}-{stamp}.log"
        handle = open_log(directory, name)
        if handle is None:
            return cls(None, None)
        return cls(Path(directory) / name, handle)

    def write(self, line: str) -> None:
        if self._handle is None:
            return
        try:
            # Flushed per line: the installs worth reading afterwards are the
            # ones that died, and a buffer would take the last minute with it.
            self._handle.write(f"{line}\n")
            self._handle.flush()
        except OSError as error:
            print(f"Warning: cannot write the install log: {error}", file=sys.stderr)
            self.close()

    def finish(self, outcome: str) -> None:
        """Record how the install ended, then close. Called on every path."""

        self.write(f"--- {outcome} ---")
        self.close()

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = None
