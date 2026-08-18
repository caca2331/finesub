"""Shared access to the machine-local ``.state`` file.

Several subsystems keep counters and cooldowns there, each under its own
top-level key, and each has to read the whole document to preserve the others'
sections. Doing that without a lock loses one side's update whenever two
processes overlap, and rewriting the file in place truncates it if the write is
interrupted -- which reads back as "no state at all", silently discarding
accumulated quota information.

So every mutation goes through :func:`state_section`: one cross-process lock
held across the read and the write, and an atomic replace at the end.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from .paths import resolve_state_file
from .reporting import current_reporter

_LOCK_SUFFIX = ".lock"


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock for the duration of a read-modify-write.

    Blocking, and on a separate file so the state itself can be replaced while
    the lock is held -- an atomic replace swaps the inode, which would drop a
    lock taken on the state file.
    """

    lock_path = path.with_name(path.name + _LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
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
        handle.close()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # Recoverable -- callers rebuild their state - but not silent: this is
        # accumulated quota and cooldown information going away.
        current_reporter().warning(
            "state-unreadable",
            f"discarding unreadable state at {path} ({exc}); starting from empty.",
        )
        return {}
    except OSError as exc:
        current_reporter().warning(
            "state-read-failed", f"could not read state at {path}: {exc}"
        )
        return {}
    return raw if isinstance(raw, dict) else {}


def _store(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_section(namespace: str, path: Optional[Path] = None) -> dict[str, Any]:
    """Return one namespace's contents; unreadable state reads as empty."""

    target = Path(path) if path is not None else resolve_state_file()
    section = _load(target).get(namespace)
    return dict(section) if isinstance(section, dict) else {}


@contextmanager
def state_section(
    namespace: str,
    path: Optional[Path] = None,
) -> Iterator[dict[str, Any]]:
    """Mutate one namespace under a lock, writing the whole document atomically.

    The caller sees only its own section; the surrounding document is preserved
    because the read happens inside the lock that the write also holds.
    """

    target = Path(path) if path is not None else resolve_state_file()
    with _file_lock(target):
        document = _load(target)
        section = document.get(namespace)
        section = dict(section) if isinstance(section, dict) else {}
        yield section
        document[namespace] = section
        try:
            _store(target, document)
        except OSError as exc:
            # Losing an update costs a reset, not the run.
            current_reporter().warning(
                "state-write-failed",
                f"could not write state at {target}: {exc}",
            )
