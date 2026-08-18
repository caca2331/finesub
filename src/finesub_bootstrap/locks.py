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
from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import time
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
