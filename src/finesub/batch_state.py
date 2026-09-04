"""A batch's own state on disk: what is queued, what the user asked for next.

Four files inside `out/batch/<id>/`, plus one registry beside the runtime data.
None of it knows what a task *is* -- no audio, no model, no stage. That is why
it is here and not in `pipeline`: the same test `scheduler` was extracted on
(the engine is domain-agnostic) applies to this half of what was left behind.

    pipeline  ->  batch_state  ->  scheduler
    (the CLI)     (this file)      (the engine, and the types below)

The direction is what keeps the batch directory's layout describable in one
place. `DEFAULT_BATCH_ROOT` and `STATUS_FILENAME` stay in `scheduler`: the
status log is the engine's own output, and importing them from here rather
than the other way round is what avoids a cycle -- `control_intake` builds a
`BatchItem` and every poll returns an `IntakePoll`.

Two writers, one file each: the runner owns `queue.jsonl` (its published view)
and the user owns `control.jsonl` (append-only instructions). Nothing here
writes both.
"""


from __future__ import annotations


import json


import os


import sys


import threading


import time


from datetime import datetime, timedelta


from pathlib import Path


from typing import Any, Callable, Mapping, Sequence


from .paths import resolve_logs_dir


from .scheduler import BatchItem, IntakePoll, ItemResult


BATCH_REGISTRY_FILENAME = "batches.json"


#: Beyond this, resuming without naming an id refuses and prints the command
#: that names it. Not a prompt (that would hang a script) and not a silent
#: warning (a week-old batch resumed by surprise is exactly the surprise worth
#: preventing): the id-bearing command is the deliberate way to say "yes, that
#: one" (owner 2026-08-31).
STALE_RESUME_DAYS = 7


#: How many entries the registry keeps. It exists to answer "the last one", so
#: a long tail buys nothing and costs a growing file.
BATCH_REGISTRY_LIMIT = 50


def batch_registry_path() -> Path | None:
    directory = resolve_logs_dir()
    return None if directory is None else directory.parent / BATCH_REGISTRY_FILENAME


def _read_batch_registry(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Ancillary: an unreadable pointer file must never fail a run.
        return []
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def record_batch(batch_id: str, queue_path: Path, *, state: str, items: int) -> None:
    """File where this batch's queue is and how it is doing.

    Best effort in both directions: a registry that cannot be written changes
    nothing about the run, and one that cannot be read just means `--resume-batch`
    asks for an id.
    """

    path = batch_registry_path()
    if path is None:
        return
    try:
        from finesub_bootstrap.locks import holding_lock

        path.parent.mkdir(parents=True, exist_ok=True)
        # One file, every batch on this machine: read-modify-write without a
        # lock loses whichever concurrent run wrote first, and that run then
        # cannot be found by `--resume-batch` at all (reviewer 2026-08-31 P2).
        # Read INSIDE the lock, or the merge is against a stale snapshot.
        with holding_lock(path.with_name(f"{path.name}.lock"), timeout=30):
            # Keyed on (cwd, batch_id), not the id alone: the default id is a
            # one-second timestamp and an explicit one is often a word like
            # `nightly`, so two directories collide easily -- and the row that
            # lost was a batch `--resume-batch` could no longer find at all
            # (reviewer 2026-08-31 P2).
            rows = [
                row
                for row in _read_batch_registry(path)
                if not _is_batch(row, batch_id, Path.cwd())
            ]
            rows.append(
                {
                    "batch_id": batch_id,
                    "queue": str(Path(queue_path).resolve()),
                    "cwd": str(Path.cwd()),
                    "state": state,
                    "items": int(items),
                    "updated_at": time.time(),
                }
            )
            rows = rows[-BATCH_REGISTRY_LIMIT:]
            temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temp.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temp, path)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"[batch] could not record this batch: {exc}", file=sys.stderr)


def _is_batch(row: Mapping[str, Any], batch_id: str, cwd: Path) -> bool:
    """Whether a registry row is *this* batch: same id AND same directory."""

    if str(row.get("batch_id") or "") != batch_id:
        return False
    return _ran_here(row, cwd)


def _ran_here(row: Mapping[str, Any], cwd: Path) -> bool:
    """Whether a row belongs to this directory. A row without one is nobody's,
    so it answers yes rather than lingering forever unmatched."""

    recorded = str(row.get("cwd") or "")
    return not recorded or _same_directory(recorded, cwd)


def unfinished_batches() -> list[dict[str, Any]]:
    """Batches that did not reach a clean end, newest first.

    `running` counts as unfinished: a run killed outright never got to write
    its ending, and that is precisely the batch someone comes back for.
    """

    rows = [
        row
        for row in _read_batch_registry(batch_registry_path())
        if row.get("state") not in ("finished",) and Path(str(row.get("queue") or "")).is_file()
    ]
    return sorted(rows, key=lambda row: float(row.get("updated_at") or 0), reverse=True)


def resolve_resume_batch(requested: str) -> tuple[Path | None, str, str]:
    """The queue file to resume and the id it belongs to, else (None, "", why not).

    ``requested`` is a batch id, or ``""`` for "the last unfinished one". An
    id is a deliberate choice and is never refused for age; the id-less form is
    a convenience, and a convenience that silently picks up week-old work is a
    trap.

    The id comes back with the path because resuming keeps it: a run that
    continues a batch IS that batch, and wants its directory, its lock and its
    registry row rather than a fresh set (see `main`).
    """

    rows = unfinished_batches()
    here = Path.cwd()
    if requested:
        named = [
            row
            for row in rows + _read_batch_registry(batch_registry_path())
            if str(row.get("batch_id")) == requested
        ]
        # One id can name a batch in each of several directories. The one here
        # is the one meant; a foreign one still resolves, so that `_resumable`
        # can say where it belongs instead of "no such batch".
        for row in [row for row in named if _ran_here(row, here)] or named:
            return _resumable(row)
        return None, "", f"no batch {requested!r} in the registry"
    local = [row for row in rows if _ran_here(row, here)]
    if not local:
        if rows:
            row = rows[0]
            return None, "", (
                "no unfinished batch in this directory; the most recent one "
                f"({row['batch_id']}) ran in {row.get('cwd')}:\n"
                f"    cd {row.get('cwd')}\n"
                f"    finesub --resume-batch {row['batch_id']}"
            )
        return None, "", "no unfinished batch on record"
    row = local[0]
    age_days = (time.time() - float(row.get("updated_at") or 0)) / 86400
    if age_days > STALE_RESUME_DAYS:
        return None, "", (
            f"the last unfinished batch ({row['batch_id']}) is "
            f"{age_days:.0f} days old, so it is not picked up by default.\n"
            f"To resume it anyway:\n"
            f"    finesub --resume-batch {row['batch_id']}"
        )
    return _resumable(row)


def _resumable(row: Mapping[str, Any]) -> tuple[Path | None, str, str]:
    """A registry row, checked against the two things a queue path cannot say.

    Neither check has anything to do with how deliberate the request was, so
    naming the id does not get past them: a live batch resumed alongside itself
    is two sets of workers on one set of outputs, and a batch resumed from
    another directory is a *different* run -- `out/` and every relative source
    are CWD-relative, so nothing it produced would be found and nothing it
    produces would land where the first half did (reviewer 2026-08-31 P1).
    """

    queue_path = Path(str(row.get("queue") or ""))
    batch_id = str(row.get("batch_id") or "")
    if not queue_path.is_file():
        # Only reachable by id: the id-less form picks from rows whose queue
        # still exists. Say what is missing rather than failing later on a
        # manifest that cannot be opened.
        return None, "", f"batch {batch_id} has no queue left at {queue_path}"
    if batch_is_live(queue_path):
        return None, "", (
            f"batch {batch_id} is still running; add to it by "
            f"appending to {queue_path.with_name(CONTROL_FILENAME)}"
        )
    recorded = str(row.get("cwd") or "")
    if recorded and not _same_directory(recorded, Path.cwd()):
        return None, "", (
            f"batch {batch_id} ran in {recorded}, and its outputs "
            "are relative to it. Resume it from there:\n"
            f"    cd {recorded}\n"
            f"    finesub --resume-batch {batch_id}"
        )
    if not batch_id:
        return None, "", "the registry row names no batch"
    return queue_path, batch_id, ""


def _same_directory(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def batch_lock_path(queue_path: str | Path) -> Path:
    """The lock a running batch holds, beside its queue.

    In the batch directory rather than the registry so that it is the BATCH
    that is locked, not a row about it: a run started with an explicit
    `--batch-id` and one resumed by id both arrive here, and neither reads the
    registry.
    """

    return Path(queue_path).with_name(BATCH_LOCK_FILENAME)


def batch_is_live(queue_path: str | Path) -> bool:
    """Whether some process is running this batch right now.

    Best effort by nature -- the answer is stale as soon as it is given -- so
    it guards against the mistake (resuming what is already running) and not
    against the race. The mutual exclusion is the lock the run itself holds.
    """

    from finesub_bootstrap.locks import try_lock

    try:
        return not try_lock(batch_lock_path(queue_path))
    except OSError:
        return False


QUEUE_VIEW_FILENAME = "queue.jsonl"


CONTROL_FILENAME = "control.jsonl"


CONTROL_CURSOR_FILENAME = ".control-cursor"


BATCH_LOCK_FILENAME = ".batch.lock"


#: Keys the runner writes into the view. They are stripped when the view is
#: read back as a manifest, so `finesub --manifest .../queue.jsonl` resumes a
#: run from its own record. The prefix is the convention: `_x` belongs to the
#: runner, everything else is the row the user asked for.
_VIEW_ONLY_PREFIX = "_"


_VIEW_WRITE_LOCK = threading.Lock()


def write_queue_view(path: Path, items: Sequence[BatchItem], results: Sequence[ItemResult]) -> None:
    """Publish what this run knows, as re-feedable manifest rows.

    Rewritten whole on every state change rather than appended to: this is a
    view, and a view that has to be replayed to be understood is not one. It is
    also the run's record -- kept after the run ends, whatever the ending, so
    "continue this batch tomorrow" is `--manifest` on the file itself and
    "do not continue it" is doing nothing.
    """

    # Rendered INSIDE the lock, not merely written inside it. Serialising only
    # the write let an older snapshot land last: a worker builds its body, the
    # intake thread then admits a task, publishes and commits its cursor, and
    # the older body finally takes the lock and erases the task -- leaving a
    # cursor that says the control line was consumed and a view with no sign of
    # it. Building here makes the last writer the last reader too, and the
    # state it reads only ever moves forward (reviewer 2026-08-31 P1).
    with _VIEW_WRITE_LOCK:
        _write_view_locked(path, items, results)


def _write_view_locked(
    path: Path, items: Sequence[BatchItem], results: Sequence[ItemResult]
) -> None:
    lines = []
    for item, result in zip(items, results):
        row = dict(item.row or {})
        row.setdefault("source", item.label)
        # The item's priority NOW, not the row's: a control action that lowered
        # it back to the default would otherwise leave the old number standing
        # in the view, and a resume would read it back (reviewer 2026-08-31 P2).
        if item.priority:
            row["priority"] = item.priority
        else:
            row.pop("priority", None)
        row["_state"] = result.view_state
        if result.stage:
            row["_stage"] = result.stage
        if result.failed_stage:
            row["_failed_stage"] = result.failed_stage
        if result.error:
            row["_error"] = result.error
        # `default=str` because a row is the merged options, and those are not
        # all JSON to begin with: `--name` puts a Path in `output`. The view is
        # a rendering and must not be the thing that fails -- and a Path
        # rendered as its string IS the manifest spelling of that option.
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    body = "\n".join(lines) + ("\n" if lines else "")
    # The write needs the same lock for its own reason: two threads shared one
    # temp path, so the bytes interleaved and whichever `os.replace` came
    # second raised on a name the first had already moved. A resume reads this
    # file; it does not get to be approximately right (reviewer 2026-08-31 P2).
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(body, encoding="utf-8")
    os.replace(temp, path)


def strip_view_keys(row: Mapping[str, Any]) -> dict[str, Any]:
    """A published row, back to the row it was asked for."""

    return {key: value for key, value in row.items() if not key.startswith(_VIEW_ONLY_PREFIX)}


def is_withdrawn(row: Mapping[str, Any]) -> bool:
    """Whether a published row was dropped by the user rather than left over.

    A view row is the record of what happened to an item, and "you withdrew
    this" has to survive being read back: stripping the state and rebuilding
    the row would resurrect, on the next resume, exactly the task someone took
    the trouble to cancel (reviewer 2026-08-31 P1). Interrupted and failed
    items are NOT withdrawn -- resuming them is the entire point.
    """

    return str(row.get("_state") or "") == "dropped"


def control_intake(
    path: str | Path,
    *,
    admit: Callable[[Mapping[str, Any]], BatchItem],
    cursor_path: str | Path | None = None,
) -> Callable[[], IntakePoll]:
    """Read the control channel: new rows to run, and changes to queued ones.

    ``admit`` takes a control row and returns the item to run, which is where
    the caller applies the run's defaults and whatever validation a row owes
    (`merge_item_options` in the CLI). One hook rather than `defaults` plus a
    builder plus a merge function, because the three are never useful apart --
    and because knowing what a valid option is would put this file back in the
    business of knowing what a task is.

    Append-only and user-owned, polled on the runner's own intake tick. A line
    is a control action when it names an `item`, and a task otherwise -- there
    is no mode flag because there is no ambiguity: a task row must carry a
    `source`, and an action must carry the label of something already here.

    The file need not exist. It is created on first use by whoever writes to
    it, which is the point: nothing has to be prepared in advance for a run
    started from positional arguments to become one you can add to.

    ``cursor_path`` makes the cursor survive the process. It is written by the
    poll's ``commit``, i.e. only once the runner has actually applied what the
    poll returned, so a run killed between "the user appended a line" and "the
    line took effect" replays it instead of skipping it forever. That is also
    why a resume does not simply start past what is in the file: a line on disk
    is not evidence that anything was done about it.
    """

    file_path = Path(path)
    cursor_file = None if cursor_path is None else Path(cursor_path)
    state = {"consumed": _read_control_cursor(cursor_file), "recorded": -1}
    state["recorded"] = state["consumed"]
    if state["consumed"]:
        print(
            f"[batch] control: {state['consumed']} line(s) already acted on; "
            f"append new ones to {file_path}",
            file=sys.stderr,
        )

    def commit() -> None:
        if cursor_file is None or state["consumed"] == state["recorded"]:
            return
        cursor_file.parent.mkdir(parents=True, exist_ok=True)
        temp = cursor_file.with_name(f"{cursor_file.name}.{os.getpid()}.tmp")
        temp.write_text(str(state["consumed"]), encoding="utf-8")
        os.replace(temp, cursor_file)
        state["recorded"] = state["consumed"]

    def poll() -> IntakePoll:
        try:
            text = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return IntakePoll()
        except (OSError, UnicodeDecodeError) as exc:
            return IntakePoll(settled=False, reason=f"{type(exc).__name__}: {exc}")
        complete = text[: text.rfind("\n") + 1]
        lines = complete.splitlines()
        tail = text[len(complete):].strip()
        pending_tail = bool(tail) and len(text.splitlines()) > state["consumed"]
        fresh: list[BatchItem] = []
        actions: list[Mapping[str, Any]] = []
        for line_no in range(state["consumed"], len(lines)):
            line = lines[line_no].strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("a control line must be a JSON object")
                if row.get("item"):
                    actions.append(row)
                elif not is_withdrawn(row):
                    fresh.append(admit(strip_view_keys(row)))
            except Exception as exc:  # noqa: BLE001 -- a typo must not kill the run
                print(
                    f"[batch] control line {line_no + 1} skipped: {exc}",
                    file=sys.stderr,
                )
        state["consumed"] = max(state["consumed"], len(lines))
        return IntakePoll(
            items=tuple(fresh),
            actions=tuple(actions),
            settled=not pending_tail,
            reason="control ends mid-line" if pending_tail else "",
            # Offered whenever ground is still unretired -- not just on the
            # poll that read it. The read cursor moves when a line is HANDED
            # OVER and the durable one only when its effect is on disk, so a
            # poll whose publish failed leaves a gap that later polls, which
            # have nothing new of their own, are the ones who must close
            # (reviewer 2026-08-31 P1). `None` when they are level, so a quiet
            # batch is not republished every tick for nothing.
            commit=commit if state["consumed"] != state["recorded"] else None,
        )

    return poll


def _read_control_cursor(path: Path | None) -> int:
    """How many control lines a previous run of this batch acted on.

    Unreadable or absurd counts as none: replaying a control line costs a
    duplicate that the claims book refuses, while wrongly skipping one loses
    work the user asked for.
    """

    if path is None or not path.is_file():
        return 0
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        return 0


def merged_intake(*polls: Callable[[], IntakePoll]) -> Callable[[], IntakePoll]:
    """One poll out of several sources (the manifest and the control channel).

    Unsettled is sticky across the sources: if either could not be read whole,
    the run must not take "nothing new" for an answer.
    """

    def poll() -> IntakePoll:
        items: list[BatchItem] = []
        actions: list[Mapping[str, Any]] = []
        settled = True
        reasons: list[str] = []
        commits: list[Callable[[], None]] = []
        for one in polls:
            result = one()
            items.extend(result.items)
            actions.extend(result.actions)
            settled = settled and result.settled
            if result.reason:
                reasons.append(result.reason)
            if result.commit is not None:
                commits.append(result.commit)
        return IntakePoll(
            items=tuple(items),
            actions=tuple(actions),
            settled=settled,
            reason="; ".join(reasons),
            # Every source's cursor advances together, because they were all
            # applied together.
            commit=(lambda: [one() for one in commits]) if commits else None,
        )

    return poll
