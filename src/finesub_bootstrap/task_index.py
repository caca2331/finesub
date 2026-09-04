"""The task record both front ends read and write.

`user-data/tasks.json` indexes what has been run: it is small, irreplaceable,
and -- since personal data became shared -- written by whichever FineSub the
user happens to start. The desktop got here first and already merges rather
than overwrites; this module is that file protocol with the desktop's model
taken out of it, so the CLI can join the same index instead of keeping a second
one beside it.

Entries are plain dicts and `request` is opaque: this layer never interprets
it, which lets the desktop store its typed request while the CLI stores the
TaskRequest-compatible settings it can faithfully replay.

**Do not add fields to an entry, or values a `Literal` does not already
allow.** The desktop validated each entry into a model that forbids extras --
and an installed 0.4.x desktop still reads this index -- so one carrying
something it has not heard of is skipped on read and invisible in
its history. Writing through `merge_write` keeps such an entry -- the merge
starts from what is on disk, not from what the caller could parse -- but a
front end old enough to predate this module writes back only what it loaded,
and there the entry is gone. That is the case to design against: a user can
upgrade one front end and not the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from math import inf
from pathlib import Path
import shutil
import time
from typing import Any

from finesub_bootstrap.fsops import RECORD_REPLACE, replace_path
from finesub_bootstrap.locks import holding_lock

#: Written for the benefit of a reader that might one day need to tell shapes
#: apart. Nothing branches on it yet, and an unknown value is not a reason to
#: refuse a file: entries are validated one by one regardless.
SCHEMA_VERSION = 1

# Nothing is ever dropped from this file: it records what the user has run, and
# a task they might continue is worth finding however long ago it was. It stays
# small because the events are not here -- a finished entry is a few hundred
# bytes, so ten thousand tasks are a few megabytes and one short parse per run.
# Trimming for display is the front end's business (`HISTORY_RENDER_LIMIT`).

#: Long enough to outlast a competing write, short enough that starting a run
#: never appears to hang on one.
LOCK_TIMEOUT_SECONDS = 10


class InvalidTaskIndex(ValueError):
    """An existing index cannot be merged without risking history loss."""


class TaskIndexUnreadable(OSError):
    """The index could not be *opened* -- which says nothing about its contents.

    Deliberately not an `InvalidTaskIndex`: the two are answered differently.
    Invalid content is preserved and replaced; a file that would not open right
    now (the other front end is mid-replace, a scanner has it, the drive
    blinked) must not be, or one unlucky moment turns the whole shared history
    into a `.invalid` backup nobody looks at.
    """


def _ordering_key(entry: Mapping[str, Any]) -> float:
    """When this entry was last touched, for merging and ordering.

    Anything unusable counts as the beginning of time rather than raising. The
    two front ends sort and compare every entry against every other on each
    write, so one record carrying a string where a number belongs -- a
    hand-edit, a writer we have not met -- would raise `TypeError` inside the
    merge and stop *all* history from being written, on both front ends, for as
    long as it stayed in the file. Ordering one strange entry early is a much
    smaller wrong than that.
    """

    value = entry.get("updated_at")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        # JSON integers have no width limit, and Python honours that: an entry
        # carrying `10 ** 10000` converts to no float at all, and the
        # `OverflowError` would stop the merge exactly like the string this
        # guard was written for.
        number = float(value)
    except (OverflowError, ValueError):
        return 0.0
    return number if number == number and number not in (inf, -inf) else 0.0


def _stored_path(value: str, tasks_root: Path | None) -> str:
    """Record a path under the tasks root as relative to it.

    Task outputs move: `finesub relocate` puts them on another disk, and users
    move the folder by hand. Absolute paths in the index would all die at that
    moment even though the files are right there; relative ones resolve against
    wherever the tasks root is now. Paths the user chose themselves live
    outside the root and stay absolute.
    """

    if tasks_root is None or not value:
        return value
    try:
        return Path(value).relative_to(tasks_root).as_posix()
    except ValueError:
        return value


def _loaded_path(value: str, tasks_root: Path | None) -> str:
    if tasks_root is None or not value or Path(value).is_absolute():
        return value
    return str(tasks_root / value)


def _map_paths(entry: dict, tasks_root: Path | None, convert) -> dict:
    request = entry.get("request")
    if isinstance(request, dict) and isinstance(request.get("output"), str):
        # Copied, not mutated: callers pass dicts they still hold, and turning
        # their absolute output into a tasks-relative fragment behind their
        # back is the kind of thing that is only ever found much later.
        entry["request"] = {
            **request,
            "output": convert(request["output"], tasks_root),
        }
    outputs = entry.get("outputs")
    if isinstance(outputs, dict):
        entry["outputs"] = {
            key: convert(value, tasks_root) if isinstance(value, str) else value
            for key, value in outputs.items()
        }
    return entry


def _decode(path: Path, tasks_root: Path | None) -> list[dict]:
    """Decode one existing index, rejecting shapes a write cannot preserve."""

    # Bytes first, decoded second: undecodable content is a property of the
    # file, while failing to open it is a property of the moment.
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TaskIndexUnreadable(f"cannot open {path}") from error
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise InvalidTaskIndex(f"cannot parse {path}") from error
    items = body.get("tasks") if isinstance(body, dict) else body
    if not isinstance(items, list):
        raise InvalidTaskIndex(f"{path} does not contain a task list")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("task_id"), str)
        for item in items
    ):
        raise InvalidTaskIndex(f"{path} contains an invalid task entry")
    return [
        _map_paths(dict(item), tasks_root, _loaded_path)
        for item in items
    ]


#: How many times a reader retries a file it could not open, and how long it
#: waits between attempts. The index is renamed into place while the other
#: front end is possibly reading it, and on Windows that pair collides for as
#: long as a handle is open -- microseconds, but often enough to matter when
#: the answer to "could not open it" is "you have no history". The writer's
#: half of the same collision is `fsops.replace_path`'s record budget, below.
_READ_ATTEMPTS = 3
_READ_RETRY_SECONDS = 0.02


def read(path: Path | None, tasks_root: Path | None = None) -> list[dict]:
    """Every valid stored entry, oldest first, with paths made absolute.

    History is ancillary to running the pipeline, so an unparseable file
    degrades to an empty view rather than raising. A file that would not *open*
    is retried briefly first: another front end replacing it right now is the
    normal case, and answering "no history" for it makes the window blink empty
    while the run it is about is going perfectly well.
    """

    if path is None or not path.is_file():
        return []
    for attempt in range(_READ_ATTEMPTS):
        try:
            return _decode(path, tasks_root)
        except InvalidTaskIndex:
            return []
        except TaskIndexUnreadable:
            if attempt == _READ_ATTEMPTS - 1:
                return []
            time.sleep(_READ_RETRY_SECONDS)
    return []


def _invalid_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.invalid")
    serial = 0
    while candidate.exists():
        serial += 1
        candidate = path.with_name(f"{path.name}.invalid.{serial}")
    return candidate


def _read_for_merge(path: Path, tasks_root: Path | None) -> list[dict]:
    """Read under the write lock, backing up an invalid existing file once.

    `TaskIndexUnreadable` is deliberately not caught. Both callers already log
    and carry on when a history write fails, and the next write retries -- a
    skipped write costs one entry until then, while treating "busy right now"
    as "corrupt" would archive everyone else's tasks and hand back an index
    holding only ours.
    """

    if not path.exists():
        return []
    try:
        return _decode(path, tasks_root)
    except InvalidTaskIndex:
        # copy2 deliberately happens before the replacement write. If the copy
        # fails, propagate and leave the invalid original untouched; recovering
        # the new task record is less important than not destroying the only
        # copy of the old history.
        shutil.copy2(path, _invalid_backup_path(path))
        return []


def merge_write(
    path: Path | None,
    entries: Iterable[Mapping[str, Any]],
    tasks_root: Path | None = None,
    *,
    preserve_running_after: Mapping[str, float] | None = None,
) -> None:
    """Write `entries` into the index, keeping everyone else's.

    Under the lock, and re-reading first: another front end may have appended
    since we last looked, and writing our own view straight out would silently
    drop its tasks. Where both have an entry for one id, the more recently
    updated wins.
    """

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with holding_lock(
        path.with_suffix(f"{path.suffix}.lock"), timeout=LOCK_TIMEOUT_SECONDS
    ):
        by_id = {
            entry["task_id"]: entry
            for entry in _read_for_merge(path, tasks_root)
        }
        for entry in entries:
            existing = by_id.get(entry["task_id"])
            generation_start = (preserve_running_after or {}).get(
                entry["task_id"]
            )
            if (
                existing is not None
                and existing.get("state") == "running"
                and generation_start is not None
                and _ordering_key(existing) > generation_start
            ):
                # A later lifecycle acquired this task id after the caller's
                # worker emitted its terminal event. The caller may only now
                # be draining that old pipe; never let its delayed terminal
                # overwrite the successor's running mark.
                continue
            # Equal timestamps mean the caller has no evidence that its copy
            # is newer. Keep the disk entry: a migration may have rewritten
            # only path representation while deliberately preserving
            # `updated_at`, and an idle desktop can still hold the pre-move
            # absolute form in memory.
            if existing is None or _ordering_key(entry) > _ordering_key(existing):
                by_id[entry["task_id"]] = dict(entry)
        ordered = sorted(by_id.values(), key=_ordering_key)
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "tasks": [
                _map_paths(dict(entry), tasks_root, _stored_path)
                for entry in ordered
            ],
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        # The reader retry above only covers `read()`; this is the writer
        # the same comment describes colliding with it.
        replace_path(temporary, path, budget=RECORD_REPLACE)


def canonical_source(value: str) -> str:
    """One name for one input, whatever the caller typed.

    A local path is resolved, so the same file reached from two working
    directories is one task -- and, far more importantly, so two *different*
    files that happen to share a relative name are not. `clip.wav` in one
    directory and `clip.wav` in another are the same nine characters and
    nothing else; matching on the string would hand the second run the first
    one's ASR result and subtitle its audio with the wrong transcript.

    A URL is kept verbatim: it has no filesystem identity, and resolving it
    would invent one relative to whatever directory the shell happens to be in.
    """

    if "://" in value:
        return value
    try:
        return str(Path(value).expanduser().resolve())
    except (OSError, ValueError):
        return value


def _same_source(entry: Mapping[str, Any], source: str) -> bool:
    """Whether `entry` was run on the same input."""

    request = entry.get("request")
    recorded = request.get("input") if isinstance(request, dict) else None
    if not isinstance(recorded, str) or not recorded:
        return False
    return canonical_source(recorded) == canonical_source(source)


def find_latest(
    entries: Iterable[Mapping[str, Any]],
    *,
    source: str,
    include_running: bool = False,
) -> dict | None:
    """The most recent task for this input, or None to start a new one.

    The input alone decides. Where the subtitle was asked to go deliberately
    does not: a run's own directory holds the ASR result that makes a rerun
    cheap, and that result depends on the source, not on where the finished
    file was copied afterwards. Matching on the destination too would miss the
    common case -- same source, new destination -- and redo the expensive half
    for nothing.

    A task marked running is not returned: continuing one means writing into
    the directory another front end is writing to right now, and the two would
    interleave `-vocal.ogg`, `-aligned.json` and `-stable.json`.

    The mark outlives the process, though. ``include_running`` only exposes
    those candidates to a caller that will establish ownership separately; it
    is not itself evidence that the mark is stale. The CLI, for example, still
    has to acquire that task id's durable sidecar before it may reuse one.
    """

    matches = [
        dict(entry)
        for entry in entries
        if (include_running or entry.get("state") != "running")
        and _same_source(entry, source)
    ]
    return matches[-1] if matches else None
