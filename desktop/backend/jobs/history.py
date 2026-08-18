"""The task record and the rules for reading the shared index.

Everything here is a value or a pure function over one: what a task looks like
once it is written down, and the decisions the manager has to make about a
`tasks.json` it does not own alone. The manager keeps the mutable list, the
lock and the writing; this module is what it consults.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePath
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finesub_bootstrap import task_index

from desktop.backend.common.models import TaskRequest
from desktop.backend.worker.protocol import WorkerEvent


#: How many past tasks the window offers. Not a storage limit -- the shared
#: index keeps everything, and the CLI relies on that to find a task worth
#: continuing however long ago it ran.
HISTORY_RENDER_LIMIT = 100

JobState = Literal[
    "idle",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]


class JobSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    state: JobState
    request: TaskRequest
    events: list[WorkerEvent] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


def without_finished_events(snapshot: JobSnapshot) -> dict:
    """Serialize a snapshot, dropping the event log of anything already over.

    `tasks.json` is described in the launcher as a "small and irreplaceable"
    index, and at the shapes this code itself allows -- 100 tasks x 500 events
    -- events were 100% of an 8.7 MB file, the other fields about 40 KB. They
    are also a second copy: `TaskLog` already puts the full log beside
    the outputs, which is what survives the app closing. Only the running task
    keeps its events, so a restart can still restore a live drawer.
    """

    body = snapshot.model_dump(mode="json")
    if snapshot.state != "running":
        body["events"] = []
    return body


def read_history_file(
    history_path: Path | None,
    output_root: Path | None = None,
) -> list[JobSnapshot]:
    """The shared index, as far as this front end can understand it.

    Entries it cannot validate are skipped rather than fatal: the file is
    shared with the CLI, which writes only the fields it knows, and one
    unreadable record must not cost the user the rest of their history.
    """

    snapshots: list[JobSnapshot] = []
    for entry in task_index.read(history_path, output_root):
        try:
            snapshots.append(JobSnapshot.model_validate(entry))
        except Exception:
            continue
    return snapshots


def validate_task_id(task_id: str) -> None:
    """Reject anything that is not one plain directory name under the root."""

    candidate = PurePath(task_id)
    if (
        len(candidate.parts) != 1
        or candidate.is_absolute()
        or candidate.drive
        or task_id in {".", ".."}
        or any(separator in task_id for separator in ("/", "\\"))
    ):
        raise ValueError(f"Not a task id: {task_id!r}")


def retarget_path(
    value: str | None,
    previous_root: Path,
    current_root: Path,
) -> str | None:
    """Move only paths managed below the previous tasks root."""

    if not value:
        return value
    try:
        relative = Path(value).expanduser().resolve().relative_to(previous_root)
    except ValueError:
        return value
    return str(current_root / relative)


def event_epoch(event: WorkerEvent) -> float:
    """The producer time that orders this worker lifecycle in shared history."""

    try:
        value = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    except (OSError, OverflowError, ValueError):
        return time.time()


def remember_path_alias(
    aliases: dict[Path, Path],
    previous: str | None,
    current: str | None,
) -> None:
    """Record that an output the UI may still name now lives somewhere else."""

    if not previous or not current:
        return
    old = Path(previous).expanduser().resolve()
    new = Path(current).expanduser().resolve()
    if old == new:
        return

    # `new` is authoritative even if an earlier relocation left an alias
    # pointing away from it (A→B followed by B→A). Redirect every alias
    # whose chain reaches `old` straight to `new`, and remove `new` as a
    # key so the graph stays acyclic and one hop deep.
    affected = {old}
    for alias in tuple(aliases):
        cursor = alias
        seen: set[Path] = set()
        while cursor not in seen:
            if cursor == old:
                affected.add(alias)
                break
            seen.add(cursor)
            target = aliases.get(cursor)
            if target is None:
                break
            cursor = target
    aliases.pop(new, None)
    for alias in affected:
        if alias == new:
            aliases.pop(alias, None)
        else:
            aliases[alias] = new
