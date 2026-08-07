"""Advisory cross-process locks over a sidecar file.

FineSub runs as several independent processes against one shared tree: the
desktop app, the CLI shell, the packaged command line, and worker subprocesses.
Anything that reads a location, decides something, and writes it back needs to
serialize across all of them -- the runtime swap, the data migrations, the
knowledge base's auto-commit, the big-data location record.

Waiting is polled rather than blocking so a pause request still gets through,
and the sidecar file is never deleted: removing it would let a process that
still holds the byte lock coexist with one that just created a fresh file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import time

LogCallback = Callable[[str], None]
PauseCheck = Callable[[], bool]

POLL_SECONDS = 0.5


class LockUnavailable(RuntimeError):
    """Raised by `holding_lock` when `timeout` elapses without the lock."""


def _acquire(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _release(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def holding_lock(
    lock_path: Path,
    *,
    waiting_message: str = "",
    log: LogCallback | None = None,
    should_pause: PauseCheck | None = None,
    timeout: float | None = None,
    on_pause: Callable[[], BaseException] | None = None,
) -> Iterator[None]:
    """Hold the lock at `lock_path`, waiting for it.

    `timeout` of None waits indefinitely; a number raises `LockUnavailable`
    once it elapses, which is how callers that must not block the user
    (the knowledge base, whose failures degrade to warnings) give up.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        announced = False
        while True:
            if _acquire(handle):
                break
            if should_pause is not None and should_pause():
                raise (
                    on_pause() if on_pause is not None else LockUnavailable(
                        f"Paused while waiting for {lock_path.name}"
                    )
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise LockUnavailable(
                    f"Timed out waiting for {lock_path.name}"
                )
            if not announced and log is not None and waiting_message:
                log(waiting_message)
            announced = True
            time.sleep(POLL_SECONDS)
        try:
            yield
        finally:
            _release(handle)
    finally:
        handle.close()


def try_lock(lock_path: Path) -> bool:
    """Whether the lock is free right now, without holding it.

    For preconditions ("nothing is running") rather than mutual exclusion:
    the answer is stale the moment it is returned, so it only rules out the
    obvious cases before a long operation instead of guaranteeing quiet.
    """

    if not lock_path.is_file():
        return True
    try:
        handle = open(lock_path, "a+b")
    except OSError:
        return False
    try:
        if not _acquire(handle):
            return False
        _release(handle)
        return True
    finally:
        handle.close()
