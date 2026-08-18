"""A durable record of one run of the app.

The window can go away without anyone learning why: a tray quit, a close, a
crash in the GUI thread and an unhandled exception all end the same way -- no
process, no message, nothing on disk. Asking "why did it exit?" afterwards had
no answer to look at, so this writes the few lines that would have answered it.

Deliberately thin. Not a logging framework, not a place to trace what the app
is doing: start, how it ended, and the traceback when it ended badly. Anything
per-task belongs to the worker's own event log, and anything per-install to
`install_log`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import TextIO

from finesub_bootstrap.logs import log_directory, open_log, prune


class SessionLog:
    """One file per app run. Never fatal: a broken log is not a broken app."""

    def __init__(self, path: Path | None, handle: TextIO | None) -> None:
        self.path = path
        self._handle = handle

    @classmethod
    def disabled(cls) -> "SessionLog":
        return cls(None, None)

    @classmethod
    def open(
        cls,
        user_data: Path | None,
        *,
        now: datetime | None = None,
    ) -> "SessionLog":
        if user_data is None:
            return cls.disabled()
        directory = log_directory(user_data)
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        name = f"session-{stamp}.log"
        handle = open_log(directory, name)
        if handle is None:
            return cls.disabled()
        # Shared with the install logs, on the same "newest hundred" rule.
        prune(directory)
        return cls(directory / name, handle)

    def write(self, line: str) -> None:
        if self._handle is None:
            return
        try:
            stamp = datetime.now().strftime("%H:%M:%S")
            # Flushed per line: the run worth reading afterwards is the one
            # that died, and a buffer would take its last words with it.
            self._handle.write(f"{stamp} {line}\n")
            self._handle.flush()
        except OSError as error:
            print(f"Warning: cannot write the session log: {error}", file=sys.stderr)
            self.close()

    def exception(self, context: str, error: BaseException) -> None:
        self.write(f"{context} raised {type(error).__name__}: {error}")
        if self._handle is None:
            return
        try:
            traceback.print_exception(
                type(error), error, error.__traceback__, file=self._handle
            )
            self._handle.flush()
        except OSError:
            self.close()

    def finish(self, outcome: str) -> None:
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
