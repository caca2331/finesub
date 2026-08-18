"""The per-run `task-log.txt` beside the outputs.

Separate from the manager because it is the one part of a run that outlives
the process without going through the shared index: the drawer's event deque
is bounded and `tasks.json` drops finished events on purpose, so this file is
what a later diagnosis actually reads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO

from desktop.backend.worker.protocol import WorkerEvent


class TaskLog:
    """The run's log on disk, written as it happens.

    It used to be rebuilt at the end from the snapshot's event deque, which is
    bounded: a noisy run pushed the beginning of its own log out of the only
    copy that survives the app closing, and a worker that was killed left no
    file at all. Appending as events arrive decouples the file from the UI's
    event budget -- the drawer can stay small without the log getting shorter.

    The file is `task-log.txt.part` until the task ends, then renamed. A crash
    leaves the part behind on purpose: a partial log of a run that died is the
    case the file exists for.
    """

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self._handle: TextIO | None = None
        self._opened = False
        self._finished = False
        self._failed = False

    def append(self, event: WorkerEvent) -> None:
        # `debug` reaches the file and nothing else: it is the verbose detail
        # the drawer cannot afford but a later diagnosis needs.
        if event.type not in {"log", "debug", "failed"}:
            return
        message = str(event.payload.get("message", "")).rstrip()
        if not message:
            return
        handle = self._open()
        if handle is None:
            return
        try:
            handle.write(message + "\n")
            # Flushed per line: the reason to keep this file is the run that
            # ends by being killed, and buffered tails do not survive that.
            handle.flush()
        except OSError:
            self._failed = True
            self._close()

    def finish(self) -> None:
        """Name the log for good, as soon as the task is reported over.

        At the terminal event rather than at process exit: the UI flips to
        "completed" on that event and offers to open the folder, and a log that
        is still `.part` for another moment is one the user can arrive before.
        Lines that trickle in afterwards are appended to the named file, so
        finishing early costs nothing.
        """

        already_finished = self._finished
        self._finished = True
        self._close()
        if already_finished or self.directory is None or not self._opened:
            return
        try:
            os.replace(self._part_path(), self.directory / "task-log.txt")
        except OSError:
            # A log we could not save must not turn a finished task into a
            # failed one.
            pass

    def _part_path(self) -> Path:
        assert self.directory is not None
        return self.directory / "task-log.txt.part"

    def _open(self) -> TextIO | None:
        if self._handle is not None:
            return self._handle
        if self.directory is None or self._failed:
            return None
        path = (
            self.directory / "task-log.txt" if self._finished else self._part_path()
        )
        # Truncate only when opening the part for the first time; reopening
        # after `finish` must not erase the log it just named.
        mode = "w" if not self._opened and not self._finished else "a"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._handle = open(path, mode, encoding="utf-8", newline="\n")
        except OSError:
            self._failed = True
            return None
        if not self._finished:
            self._opened = True
        return self._handle

    def _close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = None
