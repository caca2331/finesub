"""Where one task's output goes, and what that task is called.

Both front ends put finished subtitles under the same `tasks` directory, and
both have to answer the same three questions to do it: what to call this run,
what to call its files, and what an explicitly requested output path means once
a task root exists. The desktop answered them first, inside its job manager;
this module is that logic with the desktop's request object taken out of it, so
the CLI can give the same answers rather than a second set.

Deliberately about strings and paths only. Nothing here reads the environment,
creates a directory or knows what a `TaskRequest` is -- each front end supplies
what it has and applies the result its own way.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from uuid import uuid4


#: Rejected outright by Windows, and a poor idea everywhere else.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Long enough to stay recognisable, short enough that the stem plus a task id
#: plus a suffix stays clear of path length limits on every layer below.
_STEM_LIMIT = 80


def task_stem(*, name: str = "", source: str = "") -> str:
    """Filesystem-safe stem for one task: the chosen name, else the source.

    Falls back to a constant rather than an empty string: a source whose stem
    is all dots or all separators is unusual, but it must not produce a file
    called `.srt` -- or, worse, a directory entry with no name at all.
    """

    raw = name.strip() or Path(source).stem.strip()
    cleaned = _UNSAFE.sub("_", raw)
    return (cleaned.rstrip(" .") or "subtitle")[:_STEM_LIMIT]


def new_task_id(stem: str, *, now: float | None = None) -> str:
    """``<stem>-YYMMDD-HHMM-<6 hex>``.

    A bare uuid told the user nothing about which directory under
    ``user-data/tasks`` belonged to which job. The timestamp orders them and
    the stem names them; the hex suffix keeps two runs of the same source in
    the same minute apart, which a stem and a minute alone cannot.
    """

    stamp = time.strftime("%y%m%d-%H%M", time.localtime(now))
    return f"{stem}-{stamp}-{uuid4().hex[:6]}"


def resolve_task_output(
    tasks_root: Path,
    task_id: str,
    *,
    requested: str | Path | None = None,
    stem: str,
) -> Path:
    """Absolute path of the SRT this task delivers.

    An absolute request is honoured as given -- someone who typed a full path
    meant it. A relative one is taken as a *file name* under the task's own
    directory, not as a path to follow: it arrives from a front end whose
    working directory is not the user's, so honouring `../` would land the
    output somewhere neither of them chose.

    The pipeline derives every other artifact from this path, so it is the only
    location either front end has to decide.
    """

    if requested is not None and str(requested):
        candidate = Path(requested).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (tasks_root / task_id / candidate.name).resolve()
    return (tasks_root / task_id / f"{stem}.srt").resolve()
