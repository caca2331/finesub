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

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any
import uuid

LogCallback = Callable[[str], None]
PauseCheck = Callable[[], bool]

POLL_SECONDS = 0.5

#: "A task is writing into the tasks tree right now." Held by the *worker*, for
#: as long as it runs, and consulted by anything that wants to move or delete
#: that tree -- `finesub relocate` and the task-outputs migration.
ACTIVE_LOCK_NAME = ".active.lock"

#: Stable coordination lives in user-data, not under the big-data tree that a
#: relocation renames. Every new run briefly takes the gate while publishing a
#: unique lease; relocation holds it for the whole move, so no run can appear
#: between the liveness check and the rename.
ACTIVITY_GATE_NAME = ".task-activity.lock"
ACTIVITY_LEASE_DIRECTORY = ".task-activity"
TASK_ACTIVITY_ROOT_VARIABLE = "FINESUB_TASK_ACTIVITY_ROOT"

# A managed worker cannot always rediscover its launcher layout from source
# location alone: the thin CLI runs vendored sources from site-packages, and a
# custom FINESUB_HOME is intentionally unrelated to that path. The launcher
# therefore hands the unified agent resolver this one complete location tuple.
AGENT_CAPSULE_ROOT_VARIABLE = "FINESUB_AGENT_CAPSULE_ROOT"
AGENT_ACTIVITY_ROOT_VARIABLE = "FINESUB_AGENT_ACTIVITY_ROOT"
AGENT_IDENTITY_ANCHOR_VARIABLE = "FINESUB_AGENT_IDENTITY_ANCHOR"
AGENT_LOCATOR_KIND_VARIABLE = "FINESUB_AGENT_LOCATOR_KIND"
# `finesub.llm.agent.agent_paths` refuses a locator kind it does not recognise, so the two
# launchers that set the variable must not each spell it out. It is declared
# here because bootstrap may not import `llm`, only the other way round.
AGENT_LOCATOR_KIND_MANAGED = "managed_big_data"


def activity_gate_path(coordination_root: Path) -> Path:
    return Path(coordination_root) / ACTIVITY_GATE_NAME


def activity_lease_path(coordination_root: Path, lease_id: str) -> Path:
    digest = hashlib.sha256(lease_id.encode("utf-8")).hexdigest()
    return (
        Path(coordination_root)
        / ACTIVITY_LEASE_DIRECTORY
        / f"{digest}.lock"
    )


def task_lock_path(tasks_root: Path, task_id: str) -> Path:
    """The durable sidecar that owns one task id.

    The task id is hashed rather than used as a path component because entries
    in the shared index are external input.  Lock files stay beside the task
    directories (and are never deleted, like every other lock in this module),
    so a crashed process leaves enough evidence for the next process to tell a
    stale ``running`` mark from a legacy mark that had no ownership lock.
    """

    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return Path(tasks_root) / f".task-{digest}.lock"


def task_workspace_lock_path(tasks_root: Path, output: str | Path) -> Path:
    """The durable sidecar owning the directory a pipeline writes into.

    A desktop "reuse ASR" run has a fresh task id but deliberately points its
    output at an older task's directory. The task-id locks are therefore
    different even though both pipelines would update the same artifacts.
    Hash the resolved output directory (case-folded on Windows) into a sidecar
    under the shared tasks root so the CLI and every desktop worker agree on
    the second lock. Callers always take the task-id lock first, then this one.
    """

    workspace = Path(output).expanduser().resolve().parent
    identity = os.path.normcase(str(workspace))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path(tasks_root) / f".workspace-{digest}.lock"


def active_lock_path(tasks_root: Path) -> Path:
    """The one path all three sides must agree on.

    They did not: the holder and `finesub relocate` used `<tasks>/.active.lock`
    while the migration checked its *parent*. `try_lock` returns True for a
    file that does not exist, so the migration's "is a desktop app running?"
    guard read a lock nobody ever takes and always passed -- free to move the
    tasks tree out from under a running task.
    """

    return Path(tasks_root) / ACTIVE_LOCK_NAME


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


def _write_lease(lock_path: Path, lease: Mapping[str, Any]) -> None:
    """Best effort: a tasks root nobody can write to keeps today's behaviour."""

    try:
        lease_path(lock_path).write_text(
            json.dumps(dict(lease), ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def lease_path(lock_path: Path) -> Path:
    """The metadata file beside one lock: who is holding it, and since when.

    Beside rather than inside: the lock file is opened and locked by every
    process that so much as asks whether a task is busy, and a reader that has
    to take the lock to learn who holds it learns nothing -- it cannot take it.
    """

    return lock_path.with_suffix(".json")


def read_lease(lock_path: Path) -> dict[str, Any] | None:
    """Whatever the current holder wrote, or None if nobody left anything.

    Absence is ordinary: a lock from an older version, a holder that could not
    write, a process killed between acquiring and writing. Callers therefore
    treat this as *extra* information, never as the answer to "is it busy" --
    that question is the lock's, and only the lock's.
    """

    try:
        record = json.loads(lease_path(lock_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def describe_lease(record: Mapping[str, Any] | None) -> str:
    """One sentence about the holder, for a message a person will read."""

    if not record:
        return ""
    frontend = str(record.get("frontend") or "")
    label = {"desktop": "桌面端", "cli": "命令行"}.get(frontend, frontend or "另一个进程")
    parts = [label]
    pid = record.get("pid")
    if isinstance(pid, int):
        parts.append(f"pid {pid}")
    host = str(record.get("host") or "")
    if host and host != socket.gethostname():
        parts.append(f"主机 {host}")
    started = record.get("started_at")
    if isinstance(started, (int, float)):
        parts.append("自 " + _started_at(float(started)))
    return f"{parts[0]}（{'，'.join(parts[1:])}）" if len(parts) > 1 else parts[0]


def _started_at(stamp: float) -> str:
    """A clock time, dated once it is no longer today's.

    A hung holder can sit there for days, and "自 12:03:20" on such a lease
    reads as *this* noon -- which is the opposite of what the reader needs to
    decide whether to wait for it or kill it.
    """

    when = time.localtime(stamp)
    today = time.localtime()
    same_day = (when.tm_year, when.tm_yday) == (today.tm_year, today.tm_yday)
    return time.strftime("%H:%M:%S" if same_day else "%m-%d %H:%M", when)


def lease_record(task_id: str, frontend: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "frontend": frontend,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": time.time(),
    }


@contextmanager
def holding_lock(
    lock_path: Path,
    *,
    waiting_message: str = "",
    log: LogCallback | None = None,
    should_pause: PauseCheck | None = None,
    timeout: float | None = None,
    on_pause: Callable[[], BaseException] | None = None,
    lease: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Hold the lock at `lock_path`, waiting for it.

    `timeout` of None waits indefinitely; a number raises `LockUnavailable`
    once it elapses, which is how callers that must not block the user
    (the knowledge base, whose failures degrade to warnings) give up.

    `lease` is written beside the lock while it is held and removed on the way
    out, so that whoever finds the lock taken can say *who* has it. It is
    commentary, not control: writing it is best effort, and nothing reads it to
    decide whether work may proceed.
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
        if lease is not None:
            _write_lease(lock_path, lease)
        try:
            yield
        finally:
            if lease is not None:
                # Best effort like the write: a sidecar that cannot be removed
                # must not turn a finished task into a failure, and the lock
                # below has to be released no matter what. A leftover lease
                # beside a free lock is exactly what readers already treat as
                # commentary, not evidence.
                try:
                    lease_path(lock_path).unlink(missing_ok=True)
                except OSError:
                    pass
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


def held_task_leases(tasks_root: Path) -> list[tuple[Path, dict[str, Any] | None]]:
    """Every task lock held right now, paired with whatever name it left.

    For display and for the sentence a refusal prints -- never for a decision.
    The answer is a snapshot: a task can end between the scan and the print,
    and one that starts a moment later is missing from it. Enforcement always
    asks one specific lock instead, which is the only question with an answer
    that stays true long enough to act on.

    Nothing is deleted here, unlike the barrier's sweep of run leases: a
    diagnostic that tidies up is a diagnostic that changes what it reports.
    """

    directory = Path(tasks_root)
    if not directory.is_dir():
        return []
    held: list[tuple[Path, dict[str, Any] | None]] = []
    for lock in sorted(directory.glob(".task-*.lock")):
        # The activity gate shares this prefix and is not a task. It normally
        # lives in user-data, but the two roots coincide in some layouts.
        if lock.name == ACTIVITY_GATE_NAME or try_lock(lock):
            continue
        held.append((lock, read_lease(lock)))
    return held


def active_run_count(coordination_root: Path) -> int:
    """How many runs have published a lease and still hold it.

    This is what makes `relocate` and `uninstall` refuse, so it is worth
    showing even though the leases carry no metadata: their ids are hashed
    into the file name, so a count is genuinely all there is. The names come
    from the task leases, which is why a diagnostic prints both.
    """

    directory = Path(coordination_root) / ACTIVITY_LEASE_DIRECTORY
    if not directory.is_dir():
        return 0
    return sum(1 for lease in directory.glob("*.lock") if not try_lock(lease))


def _discard_stale_activity_leases(coordination_root: Path) -> bool:
    """Remove free leases and report whether at least one is still held.

    The caller holds the activity gate, so no new lease can be published while
    this scan runs. Lease ids are unique and never reused; deleting a free one
    therefore cannot race with a future holder the way deleting a shared
    sidecar would.
    """

    directory = Path(coordination_root) / ACTIVITY_LEASE_DIRECTORY
    if not directory.is_dir():
        return False
    active = False
    for lease in directory.glob("*.lock"):
        if not try_lock(lease):
            active = True
            continue
        try:
            lease.unlink()
        except OSError:
            pass
    return active


@contextmanager
def holding_activity(
    coordination_root: Path,
    *,
    lease_id: str | None = None,
    timeout: float | None = 10,
) -> Iterator[None]:
    """Publish one independently lockable running-process lease.

    Registration is serialized with relocation by the short-lived gate. The
    lease itself is held for the body, so any number of unrelated tasks can be
    represented at once and a crashed process is released by the OS.
    """

    identifier = lease_id or f"{os.getpid()}-{uuid.uuid4().hex}"
    lease = activity_lease_path(coordination_root, identifier)
    lease_stack = ExitStack()
    with holding_lock(activity_gate_path(coordination_root), timeout=timeout):
        lease_stack.enter_context(holding_lock(lease, timeout=0))
    try:
        yield
    finally:
        lease_stack.close()
        try:
            with holding_lock(
                activity_gate_path(coordination_root), timeout=0
            ):
                lease.unlink(missing_ok=True)
        except (OSError, LockUnavailable):
            # A later barrier removes this now-free, uniquely named lease.
            pass


@contextmanager
def holding_activity_barrier(
    coordination_root: Path,
    *,
    legacy_active_lock: Path | None = None,
    timeout: float | None = 0,
) -> Iterator[None]:
    """Exclude task starts and require every published run to be idle.

    Keep this context for the whole destructive operation. The optional legacy
    lock detects an older worker that predates activity leases. It cannot be
    held through the operation because that sidecar lives inside the task tree
    being renamed (Windows refuses to move an open file); the stable gate is
    what excludes all lease-aware starters for the full operation.
    """

    with ExitStack() as stack:
        stack.enter_context(
            holding_lock(activity_gate_path(coordination_root), timeout=timeout)
        )
        if _discard_stale_activity_leases(coordination_root):
            raise LockUnavailable("FineSub tasks are still running")
        if legacy_active_lock is not None and not try_lock(legacy_active_lock):
            raise LockUnavailable("A legacy FineSub task is still running")
        yield


def activity_is_idle(
    coordination_root: Path, *, legacy_active_lock: Path | None = None
) -> bool:
    """A non-owning snapshot for status displays and cheap preconditions."""

    try:
        with holding_activity_barrier(
            coordination_root,
            legacy_active_lock=legacy_active_lock,
            timeout=0,
        ):
            return True
    except (OSError, LockUnavailable):
        return False
