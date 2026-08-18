from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
import logging
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from finesub_bootstrap.locks import (
    LockUnavailable,
    holding_activity,
    task_lock_path,
    try_lock,
)
from finesub_bootstrap.processes import terminate_process_tree
from finesub_bootstrap import artifacts, task_index, task_output

from desktop.backend.common.models import TaskRequest
from desktop.backend.jobs.history import (
    HISTORY_RENDER_LIMIT,
    JobSnapshot,
    event_epoch,
    read_history_file,
    remember_path_alias,
    retarget_path,
    validate_task_id,
    without_finished_events,
)
from desktop.backend.jobs.launch import (
    OutputRootResolver,
    ProcessFactory,
    ProcessTerminator,
    WorkerLaunchContext,
    new_task_id,
    spawn_worker,
    task_stem,
)
from desktop.backend.jobs.task_log import TaskLog
from desktop.backend.worker.protocol import WorkerEvent, parse_worker_line


LOGGER = logging.getLogger(__name__)


class JobAlreadyRunning(RuntimeError):
    pass


class JobNotFound(KeyError):
    pass


class JobManager:
    def __init__(
        self,
        *,
        python_executable: str | Path,
        worker_env: Mapping[str, str],
        # Reads the last GPU probe. A callable rather than a list because the
        # probe runs in the background: whatever it has when a task starts is
        # what gets checked, and a task never waits for it.
        available_gpus: Callable[[], Any] = lambda: None,
        working_directory: str | Path | None = None,
        process_factory: ProcessFactory = subprocess.Popen,
        terminate_process_tree: ProcessTerminator = terminate_process_tree,
        event_limit: int = 500,
        history_path: str | Path | None = None,
        output_root: str | Path | None = None,
        output_root_resolver: OutputRootResolver | None = None,
    ) -> None:
        self._worker_context = WorkerLaunchContext(
            python_executable=str(python_executable),
            working_directory=(
                str(working_directory) if working_directory is not None else None
            ),
            environment=dict(worker_env),
        )
        self.available_gpus = available_gpus
        self.process_factory = process_factory
        self.terminate_process_tree = terminate_process_tree
        self.event_limit = max(1, int(event_limit))
        self.history_path = (
            Path(history_path).expanduser().resolve()
            if history_path is not None
            else None
        )
        self.output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else None
        )
        self.output_root_resolver = output_root_resolver
        self._lock = threading.RLock()
        self._process: Any | None = None
        # The start of the worker lifecycle owned by `_process`. Termination
        # paths use it under the history-file lock to avoid overwriting a
        # successor that has already acquired the same task id.
        self._generation_started_at: float | None = None
        self._snapshot: JobSnapshot | None = None
        # Set while spawning, replayed once the task has a snapshot to hold it.
        self._device_notice = ""
        self._events: deque[WorkerEvent] = deque(maxlen=self.event_limit)
        self._event_base_cursor = 0
        self._history: list[JobSnapshot] = []
        # Old UI payloads may still carry a managed output after a relocation.
        # Keep the observed path rewrites so clicking one maps to the current
        # owned file instead of either reopening the old root or being rejected.
        self._path_aliases: dict[Path, Path] = {}
        self._load_history()

    def start(self, request: TaskRequest) -> JobSnapshot:
        with self._lock:
            self._ensure_idle()
            activity, request = self._prepare_task_launch(request)
            try:
                task_id = new_task_id(request)
                request = self._resolve_output(task_id, request)
                now = time.time()
                process = self._spawn_worker(task_id, request)
                self._events.clear()
                self._event_base_cursor = 0
                self._process = process
                self._generation_started_at = now
                self._snapshot = JobSnapshot(
                    task_id=task_id,
                    state="running",
                    request=request,
                    created_at=now,
                    updated_at=now,
                )
                self._history.append(self._snapshot)
                self._replay_device_notice(task_id)
                self._start_reader(task_id, process, activity, now)
            except Exception:
                activity.close()
                raise
            self._persist_history()
            return self._copy_snapshot()

    def _resolve_output(
        self,
        task_id: str,
        request: TaskRequest,
    ) -> TaskRequest:
        if self.output_root is None:
            return request
        output = task_output.resolve_task_output(
            self.output_root,
            task_id,
            requested=request.output,
            stem=task_stem(request),
        )
        return request.model_copy(update={"output": str(output)})

    def cancel(self, task_id: str) -> JobSnapshot:
        with self._lock:
            snapshot = self._require_snapshot(task_id)
            # Same rule as `shutdown`: a task we did not start has no process
            # of ours to terminate, so recording it cancelled would be a claim
            # about a job that carries on regardless.
            if snapshot.state != "running" or self._process is None:
                return self._copy_snapshot()
            snapshot.state = "cancelled"
            snapshot.updated_at = time.time()
            event = WorkerEvent.cancelled(task_id)
            self._record_event(event)
            process = self._process
            generation_started_at = self._generation_started_at
        if process is not None and process.poll() is None:
            self.terminate_process_tree(process)
        with self._lock:
            self._persist_history(
                preserve_running_after=(
                    {task_id: generation_started_at}
                    if generation_started_at is not None
                    else None
                )
            )
            return self._copy_snapshot()

    def shutdown(self) -> None:
        """Stop a running task because the application is quitting.

        Quitting means quitting: the worker is its own process group, so
        without this it survived the window closing and kept consuming the GPU,
        writing into the tasks tree and committing to the knowledge git, with
        no interface left that could see or stop it. Minimising to the tray is
        a different act and does not come through here -- pywebview only fires
        `closed` on a real quit.

        The task is left `interrupted` rather than `cancelled` because the work
        already on disk is reusable: the pipeline skips stages whose artifacts
        exist, so continuing it later costs only what was actually lost.
        """

        with self._lock:
            snapshot = self._snapshot
            process = self._process
            generation_started_at = self._generation_started_at
            # `_process is None` means we did not start it, so stopping it is
            # not ours to report: marking someone else's live task interrupted
            # would put a Continue button in front of a job still writing.
            if snapshot is None or snapshot.state != "running" or process is None:
                return
            snapshot.state = "interrupted"
            snapshot.error = "应用退出时任务被终止，可以继续任务。"
            snapshot.updated_at = time.time()
            # Set before terminating, so the reader thread sees a task that is
            # no longer running and does not overwrite this with a failure.
            try:
                self._persist_history(
                    preserve_running_after=(
                        {snapshot.task_id: generation_started_at}
                        if generation_started_at is not None
                        else None
                    )
                )
            except (OSError, LockUnavailable):
                pass
        if process is not None and process.poll() is None:
            try:
                self.terminate_process_tree(process)
            except OSError:
                pass

    def snapshot(self) -> JobSnapshot | None:
        with self._lock:
            with self._holding_refreshed_task_paths():
                return self._copy_snapshot() if self._snapshot is not None else None

    def history(self) -> list[JobSnapshot]:
        """The task list, newest first, trimmed to what the UI will show.

        The trim is here rather than in the stored index: `tasks.json` is the
        record both front ends keep, and a CLI run must not push somebody's
        older task out of it just because this window can only render so many.
        What a list can usefully display is a property of the list.
        """

        with self._lock:
            with self._holding_refreshed_task_paths():
                recent = self._history[-HISTORY_RENDER_LIMIT:]
                return [
                    snapshot.model_copy(deep=True)
                    for snapshot in reversed(recent)
                ]

    def events_after(
        self,
        after_cursor: int = 0,
    ) -> tuple[list[WorkerEvent], int]:
        with self._lock:
            if self._snapshot is None:
                return [], 0
            cursor = max(0, int(after_cursor))
            start = max(0, cursor - self._event_base_cursor)
            events = [
                event.model_copy(deep=True)
                for event in self._snapshot.events[start:]
            ]
            next_cursor = (
                self._event_base_cursor + len(self._snapshot.events)
            )
            return events, next_cursor

    def retry(self, task_id: str) -> JobSnapshot:
        """Run a finished task again, in place.

        Keeps the original task id rather than minting a new one, which is what
        `resume` already does. A new id used to be paired with the *old*
        request, whose `output` had already been resolved to an absolute path
        under the old directory -- so the retry wrote its subtitle into the
        previous task's folder while "open folder" pointed at a new one holding
        nothing but a log. Reusing the id keeps outputs, log and history entry
        together, and lets the pipeline skip the stages whose artifacts are
        already on disk: a failure at the LLM stage no longer means redoing
        separation and ASR.
        """

        with self._lock:
            self._ensure_idle()
            snapshot = self._require_history(task_id)
            request = snapshot.request.model_copy(deep=True)
            activity, request = self._prepare_task_launch(request)
            try:
                generation_started_at = time.time()
                process = self._spawn_worker(task_id, request)
                self._events.clear()
                self._event_base_cursor = 0
                self._process = process
                self._generation_started_at = generation_started_at
                self._snapshot = snapshot
                self._history.remove(snapshot)
                self._history.append(snapshot)
                snapshot.request = request
                snapshot.state = "running"
                snapshot.events = []
                snapshot.error = None
                snapshot.outputs = {}
                snapshot.updated_at = generation_started_at
                self._replay_device_notice(task_id)
                self._start_reader(
                    task_id, process, activity, generation_started_at
                )
            except Exception:
                activity.close()
                raise
            self._persist_history()
            return self._copy_snapshot()

    def resume(self, task_id: str) -> JobSnapshot:
        with self._lock:
            self._ensure_idle()
            snapshot = self._require_history(task_id)
            if snapshot.state != "interrupted":
                raise ValueError("Only interrupted tasks can be continued")
            request = snapshot.request.model_copy(deep=True)
            activity, request = self._prepare_task_launch(request)
            try:
                generation_started_at = time.time()
                process = self._spawn_worker(task_id, request)
                self._events.clear()
                self._event_base_cursor = 0
                self._process = process
                self._generation_started_at = generation_started_at
                self._snapshot = snapshot
                self._history.remove(snapshot)
                self._history.append(snapshot)
                snapshot.request = request
                snapshot.state = "running"
                snapshot.events = []
                snapshot.error = None
                snapshot.updated_at = generation_started_at
                self._replay_device_notice(task_id)
                self._start_reader(
                    task_id, process, activity, generation_started_at
                )
            except Exception:
                activity.close()
                raise
            self._persist_history()
            return self._copy_snapshot()

    def delete_intermediates(self, task_id: str) -> JobSnapshot:
        """Remove one past task's bulky artifacts, keeping what it still needs.

        Offered for failed and interrupted tasks too, not only finished ones:
        those are precisely the runs that leave a separated vocal track and a
        decoded copy of the input behind, since a run that dies never reaches
        the tidying it would have done itself.

        Which is also why an unfinished task keeps its harness directory: those
        checkpoints are what "continue" resumes from, and a button that frees
        disk space must not quietly turn a resumable task into a re-run of
        every LLM call it already paid for.
        """

        with self._lock:
            if (
                self._snapshot is not None
                and self._snapshot.task_id == task_id
                and self._snapshot.state == "running"
            ):
                raise JobAlreadyRunning("A FineSub task is already running")
            with self._holding_refreshed_task_paths():
                snapshot = self._require_history(task_id)
                # The stored state, not just this process's: the index is
                # shared, so a task the CLI is running right now appears here as
                # `running` with no snapshot of ours behind it. Deleting the
                # artifacts of a run in progress is the one thing this must not
                # do -- the history UI already hides the button for those, so
                # reaching here means something called the bridge directly.
                if snapshot.state == "running":
                    raise JobAlreadyRunning(f"Task {task_id} is still running")
                output = snapshot.request.output
                if not output:
                    raise ValueError("This task recorded no output location")
                delivered = artifacts.deliverable(output, "final-srt")
                if delivered is None:
                    raise ValueError("This task recorded no output location")
                keep = [value for value in snapshot.outputs.values() if value]
                if snapshot.state != "completed":
                    keep.append(artifacts.artifact_directory(delivered))
                artifacts.cleanup_intermediate(delivered, preserve=keep)
                return snapshot.model_copy(deep=True)

    def request_for(self, task_id: str) -> TaskRequest:
        with self._lock:
            with self._holding_refreshed_task_paths():
                return self._require_history(task_id).request.model_copy(deep=True)

    def _ensure_idle(self) -> None:
        if self._snapshot is not None and self._snapshot.state == "running":
            raise JobAlreadyRunning("A FineSub task is already running")

    def _replay_device_notice(self, task_id: str) -> None:
        """Record the spawn-time notice now that a snapshot exists to hold it.

        Every entry point clears the event deque after spawning and only then
        installs the snapshot, so a notice recorded any earlier is either
        dropped outright or attached to the task that just ended.
        """

        if not self._device_notice:
            return
        self._record_event(WorkerEvent.log(task_id, self._device_notice))
        self._device_notice = ""

    @property
    def worker_context(self) -> WorkerLaunchContext:
        return self._worker_context

    def set_worker_context(self, context: WorkerLaunchContext) -> None:
        """Point future workers somewhere else, in one store."""

        self._worker_context = context

    def _spawn_worker(self, task_id: str, request: TaskRequest) -> Any:
        # One read of the context, so the interpreter, cwd and environment the
        # worker gets all belong to the same one even if an install finishes
        # mid-spawn.
        #
        # Cleared first because the notice is now assigned from a return value:
        # it used to be written part-way through the spawn, so every attempt
        # replaced it whether or not the spawn went on to fail. Nothing reads a
        # notice from a spawn that raised -- `_replay_device_notice` only runs
        # after a successful one -- and this keeps it that way.
        self._device_notice = ""
        process, self._device_notice = spawn_worker(
            self._worker_context,
            task_id,
            request,
            available_gpus=self.available_gpus,
            process_factory=self.process_factory,
            output_root=self.output_root,
            history_path=self.history_path,
        )
        return process

    def _prepare_task_launch(
        self, request: TaskRequest
    ) -> tuple[ExitStack, TaskRequest]:
        """Freeze relocation, then resolve the one tasks root both sides use.

        `locations.json` is not enough to resolve an interrupted move: the
        destination is recorded before `tasks/` is moved, and `load_app_paths`
        may deliberately select `migratingFrom/tasks` for this run. The
        launcher owns that complete decision and passes it to the worker. Its
        activity lease also closes the gap between resolving and the worker
        acquiring its independent, orphan-safe lease.
        """

        active = ExitStack()
        try:
            if self.history_path is not None:
                active.enter_context(holding_activity(self.history_path.parent))
            previous = self.output_root
            self._refresh_output_root()
            if previous is not None and self.output_root is not None:
                output = retarget_path(
                    request.output, previous, self.output_root
                )
                if output != request.output:
                    request = request.model_copy(update={"output": output})
        except Exception:
            active.close()
            raise
        return active, request

    def _refresh_output_root(self) -> None:
        if self.output_root_resolver is None:
            return
        resolved = Path(self.output_root_resolver()).expanduser().resolve()
        previous = self.output_root
        if previous is None or resolved == previous:
            self.output_root = resolved
            return

        for snapshot in self._history:
            output = retarget_path(snapshot.request.output, previous, resolved)
            if output != snapshot.request.output:
                self._remember_path_alias(snapshot.request.output, output)
                snapshot.request = snapshot.request.model_copy(
                    update={"output": output}
                )
            updated_outputs: dict[str, str] = {}
            for key, value in snapshot.outputs.items():
                updated = retarget_path(value, previous, resolved) or value
                self._remember_path_alias(value, updated)
                updated_outputs[key] = updated
            snapshot.outputs = updated_outputs
        self.output_root = resolved

    @contextmanager
    def _holding_refreshed_task_paths(self):
        """Refresh idle, user-visible paths while relocation is excluded."""

        if (
            self.output_root_resolver is None
            or self.history_path is None
            or (
                self._snapshot is not None
                and self._snapshot.state == "running"
            )
        ):
            yield self.output_root
            return

        with holding_activity(self.history_path.parent):
            previous = self.output_root
            self._refresh_output_root()
            self._refresh_history_from_disk()
            yield previous

    def _refresh_history_from_disk(self) -> None:
        """Adopt external history changes; disk wins equal timestamps.

        Migration 0003 changes path representation without changing the
        semantic task timestamp. Preserve the current drawer's ephemeral
        events, but otherwise take the disk record on a tie so an idle window
        cannot later resurrect pre-migration absolute paths.
        """

        disk = read_history_file(self.history_path, self.output_root)
        if not disk:
            return
        by_id = {snapshot.task_id: snapshot for snapshot in self._history}
        for snapshot in disk:
            existing = by_id.get(snapshot.task_id)
            if existing is None or snapshot.updated_at >= existing.updated_at:
                if existing is not None:
                    self._remember_path_alias(
                        existing.request.output, snapshot.request.output
                    )
                    for key, value in existing.outputs.items():
                        self._remember_path_alias(value, snapshot.outputs.get(key))
                if existing is not None and existing.events and not snapshot.events:
                    snapshot.events = existing.events
                by_id[snapshot.task_id] = snapshot
        self._history = sorted(by_id.values(), key=lambda item: item.updated_at)
        if self._snapshot is not None:
            current = by_id.get(self._snapshot.task_id)
            if current is not None and not (
                current.state == "running"
                and self._snapshot.state != "running"
            ):
                # `_snapshot` is the task this window can act on, not merely
                # the newest disk record for that id. Another front end may
                # continue a completed task under the same id; keep its
                # `running` record in `_history`, but never turn our drawer,
                # Cancel button and idle gate into owners of its process.
                self._snapshot = current

    def _start_reader(
        self,
        task_id: str,
        process: Any,
        activity: ExitStack,
        generation_started_at: float,
    ) -> None:
        reader = threading.Thread(
            target=self._read_worker,
            args=(task_id, process, activity, generation_started_at),
            name=f"finesub-worker-{task_id[:8]}",
            daemon=True,
        )
        reader.start()

    def _read_worker(
        self,
        task_id: str,
        process: Any,
        activity: ExitStack,
        generation_started_at: float,
    ) -> None:
        assert process.stdout is not None
        task_log = self._open_task_log(task_id)
        try:
            self._read_worker_to_end(
                task_id, process, task_log, generation_started_at
            )
        finally:
            # Each reader owns its own lease. A following task may start as
            # soon as this one reports a terminal event, while this thread is
            # still draining logs; neither lifecycle may close the other's.
            task_log.finish()
            activity.close()
            with self._lock:
                if self._process is process:
                    self._process = None
                    self._generation_started_at = None

    def _read_worker_to_end(
        self,
        task_id: str,
        process: Any,
        task_log: TaskLog,
        generation_started_at: float,
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            event = parse_worker_line(line, task_id=task_id)
            with self._lock:
                if (
                    self._snapshot is None
                    or self._snapshot.task_id != task_id
                    or self._process is not process
                ):
                    continue
                self._record_event(event)
                task_log.append(event)
                event_at = event_epoch(event)
                if event.type == "completed":
                    self._snapshot.state = "completed"
                    self._snapshot.updated_at = max(
                        event_at,
                        math.nextafter(generation_started_at, math.inf),
                    )
                    outputs = event.payload.get("outputs", {})
                    if isinstance(outputs, dict):
                        self._snapshot.outputs = {
                            str(key): str(value)
                            for key, value in outputs.items()
                        }
                elif event.type == "failed":
                    self._snapshot.state = "failed"
                    self._snapshot.error = str(event.payload.get("message", ""))
                    self._snapshot.updated_at = max(
                        event_at,
                        math.nextafter(generation_started_at, math.inf),
                    )
                elif event.type == "cancelled":
                    self._snapshot.state = "cancelled"
                    self._snapshot.updated_at = max(
                        event_at,
                        math.nextafter(generation_started_at, math.inf),
                    )
                elif self._snapshot.state == "running":
                    self._snapshot.updated_at = max(
                        self._snapshot.updated_at, event_at
                    )
                if event.type in {"completed", "failed", "cancelled"}:
                    # The worker's terminal event is not the reader's terminal
                    # operation: history and task-log writes below still touch
                    # managed data. Keep the manager's handoff lease until the
                    # process stream is drained and the reader fully settles.
                    self._persist_history(
                        preserve_running_after={
                            task_id: generation_started_at
                        }
                    )
                    task_log.finish()
        return_code = process.wait()
        with self._lock:
            if (
                self._snapshot is None
                or self._snapshot.task_id != task_id
                or self._process is not process
            ):
                return
            if self._snapshot.state == "running":
                last_log = next(
                    (
                        str(event.payload.get("message", "")).strip()
                        for event in reversed(self._snapshot.events)
                        if event.type == "log"
                        and str(event.payload.get("message", "")).strip()
                    ),
                    "",
                )
                if return_code == 0:
                    message = last_log or "处理进程已退出，但没有返回完成结果。"
                else:
                    message = (
                        last_log
                        or f"FineSub 处理进程异常退出（代码 {return_code}）。"
                    )
                failed_event = WorkerEvent.failed(task_id, message)
                self._record_event(failed_event)
                task_log.append(failed_event)
                self._snapshot.state = "failed"
                self._snapshot.error = message
                self._snapshot.updated_at = max(
                    event_epoch(failed_event),
                    math.nextafter(generation_started_at, math.inf),
                )
            self._persist_history(
                preserve_running_after={task_id: generation_started_at}
            )
            # Inside the lock, so the log is named before anything can observe
            # the state leaving "running" and go looking for the file.
            task_log.finish()

    def _open_task_log(self, task_id: str) -> TaskLog:
        return TaskLog(
            None if self.output_root is None else self.output_root / task_id
        )

    def task_directory(self, task_id: str = "") -> Path:
        """The tasks root, or one task's directory inside it.

        `task_id` must be a single relative name. Without that check `root /
        task_id` would happily accept `..\\..\\somewhere` or an absolute path --
        Windows `Path.__truediv__` discards the left side entirely for the
        latter, so `root / "C:\\Windows"` *is* `C:\\Windows` -- and the bridge
        then created and opened it. The bridge's docstring already promised
        this could not be talked into opening elsewhere; now it cannot.
        """

        with self._lock:
            with self._holding_refreshed_task_paths():
                return self._task_directory(task_id)

    def open_task_directory(
        self, task_id: str, opener: Callable[[Path], None]
    ) -> Path:
        """Resolve, create and reveal a task directory under one activity lease."""

        with self._lock:
            with self._holding_refreshed_task_paths():
                target = self._task_directory(task_id)
                opener(target)
                return target

    def open_owned_output(
        self, output_path: str, opener: Callable[[Path], None]
    ) -> Path:
        """Refresh and reveal only an output still owned by saved history."""

        with self._lock:
            before = {
                (snapshot.task_id, key): value
                for snapshot in self._history
                for key, value in snapshot.outputs.items()
            }
            with self._holding_refreshed_task_paths():
                candidate = Path(output_path).expanduser().resolve()
                owned = {
                    Path(value).expanduser().resolve()
                    for snapshot in self._history
                    for value in snapshot.outputs.values()
                    if value
                }
                seen: set[Path] = set()
                while (
                    candidate not in owned
                    and candidate in self._path_aliases
                    and candidate not in seen
                ):
                    seen.add(candidate)
                    candidate = self._path_aliases[candidate]
                for snapshot in self._history:
                    for key, value in snapshot.outputs.items():
                        previous = before.get((snapshot.task_id, key))
                        if previous and candidate == Path(previous).expanduser().resolve():
                            candidate = Path(value).expanduser().resolve()
                if candidate not in owned:
                    raise ValueError("Output path is not owned by a saved task")
                opener(candidate)
                return candidate

    def _remember_path_alias(
        self, previous: str | None, current: str | None
    ) -> None:
        remember_path_alias(self._path_aliases, previous, current)

    def _task_directory(self, task_id: str) -> Path:
        if self.output_root is None:
            raise ValueError("This build keeps no task directory.")
        if task_id:
            validate_task_id(task_id)
        root = Path(self.output_root)
        target = (root / task_id) if task_id else root
        target.mkdir(parents=True, exist_ok=True)
        return target.resolve()

    def _require_snapshot(self, task_id: str) -> JobSnapshot:
        if self._snapshot is None or self._snapshot.task_id != task_id:
            raise JobNotFound(task_id)
        return self._snapshot

    def _require_history(self, task_id: str) -> JobSnapshot:
        for snapshot in self._history:
            if snapshot.task_id == task_id:
                return snapshot
        raise JobNotFound(task_id)

    def _copy_snapshot(self) -> JobSnapshot:
        if self._snapshot is None:
            raise JobNotFound("No FineSub task has been started")
        data = self._snapshot.model_copy(deep=True)
        data.events = [
            event.model_copy(deep=True)
            for event in self._snapshot.events[-self.event_limit :]
        ]
        return data

    def _record_event(self, event: WorkerEvent) -> None:
        if self._snapshot is None:
            return
        if event.type == "debug":
            # File-only: keeping it here would spend the drawer's whole budget
            # on detail the window does not render.
            return
        self._events.append(event)
        overflow = max(
            len(self._snapshot.events) + 1 - self.event_limit,
            0,
        )
        if overflow:
            self._event_base_cursor += overflow
        self._snapshot.events = [
            *self._snapshot.events,
            event,
        ][-self.event_limit :]

    def _load_history(self) -> None:
        if self.history_path is None or not self.history_path.is_file():
            return
        self._history = read_history_file(self.history_path, self.output_root)
        if not self._history:
            return
        changed = False
        for snapshot in self._history:
            if snapshot.state != "running":
                continue
            stale = self.output_root is None
            if self.output_root is not None:
                lock_path = task_lock_path(self.output_root, snapshot.task_id)
                # No sidecar means an older version wrote this mark. It gives
                # us no safe way to distinguish a crash from a live process,
                # so do not manufacture a Continue action that could start a
                # second writer. New workers create the sidecar before doing
                # any work and leave it behind after exit.
                stale = lock_path.is_file() and try_lock(lock_path)
            if stale:
                snapshot.state = "interrupted"
                snapshot.error = "应用上次退出时任务仍在运行，可以继续任务。"
                snapshot.updated_at = time.time()
                changed = True
        # Never adopt a task we did not start. Anything still marked `running`
        # here is one the lock says another process is working on, and making
        # it this window's current task hands the whole UI -- the drawer, the
        # Cancel button, `shutdown()` -- to a job we cannot see or stop. They
        # would report it stopped and it would keep going.
        ours = [
            snapshot for snapshot in self._history if snapshot.state != "running"
        ]
        self._snapshot = ours[-1] if ours else None
        self._event_base_cursor = 0
        self._events = deque(
            self._snapshot.events[-self.event_limit :] if self._snapshot else [],
            maxlen=self.event_limit,
        )
        if changed:
            self._persist_history()

    def _persist_history(
        self,
        *,
        preserve_running_after: Mapping[str, float] | None = None,
    ) -> None:
        """Write the history back, merged with whatever else has been added.

        Personal data is shared between front ends now, so a second FineSub can
        be appending to this same file. Writing our in-memory snapshot straight
        out would silently drop its tasks; instead we take the lock, re-read,
        and merge by id keeping the more recently updated of each pair.

        Never raises. It used to, and every caller was somewhere that could not
        afford it: from the reader thread an OSError killed the thread, so a
        task that had *finished* was never written down and came back next
        launch as "interrupted", with no task-log.txt either; from `start()` it
        threw after the worker was already live, so the UI said the launch had
        failed while a real transcription ran, and every later Start answered
        "a task is already running" for a task nothing could see. Its sibling
        `TaskLog` had this reasoning in a comment all along -- a record
        we could not save must not change what actually happened.
        """

        try:
            if preserve_running_after is None:
                self._persist_history_locked()
            else:
                self._persist_history_locked(
                    preserve_running_after=preserve_running_after
                )
        except (OSError, LockUnavailable, ValueError):
            LOGGER.exception("could not write the task history")

    def _persist_history_locked(
        self,
        *,
        preserve_running_after: Mapping[str, float] | None = None,
    ) -> None:
        """The write itself. Separated so the guard above has one thing to guard."""

        # Merging with whatever else has been added is `task_index`'s job now:
        # it takes the lock, re-reads and keeps the newer of each pair, which
        # is what lets the CLI append to the same file.
        task_index.merge_write(
            self.history_path,
            [without_finished_events(snapshot) for snapshot in self._history],
            self.output_root,
            preserve_running_after=preserve_running_after,
        )
