from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from desktop.backend.common.models import TaskRequest
from desktop.backend.common.processes import terminate_process_tree
from desktop.backend.worker.protocol import WorkerEvent, parse_worker_line


JobState = Literal[
    "idle",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]


class JobAlreadyRunning(RuntimeError):
    pass


class JobNotFound(KeyError):
    pass


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


ProcessFactory = Callable[..., Any]
ProcessTerminator = Callable[[Any], None]


class JobManager:
    def __init__(
        self,
        *,
        python_executable: str | Path,
        worker_env: Mapping[str, str],
        working_directory: str | Path | None = None,
        process_factory: ProcessFactory = subprocess.Popen,
        terminate_process_tree: ProcessTerminator = terminate_process_tree,
        event_limit: int = 500,
        history_path: str | Path | None = None,
        output_root: str | Path | None = None,
    ) -> None:
        self.python_executable = str(python_executable)
        self.worker_env = dict(worker_env)
        self.working_directory = (
            str(working_directory) if working_directory is not None else None
        )
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
        self._lock = threading.RLock()
        self._process: Any | None = None
        self._snapshot: JobSnapshot | None = None
        self._events: deque[WorkerEvent] = deque(maxlen=self.event_limit)
        self._event_base_cursor = 0
        self._history: list[JobSnapshot] = []
        self._load_history()

    def start(self, request: TaskRequest) -> JobSnapshot:
        with self._lock:
            self._ensure_idle()
            task_id = uuid4().hex
            request = self._resolve_output(task_id, request)
            process = self._spawn_worker(task_id, request)
            self._events.clear()
            self._event_base_cursor = 0
            self._process = process
            now = time.time()
            self._snapshot = JobSnapshot(
                task_id=task_id,
                state="running",
                request=request,
                created_at=now,
                updated_at=now,
            )
            self._history.append(self._snapshot)
            self._start_reader(task_id, process)
            self._persist_history()
            return self._copy_snapshot()

    def _resolve_output(
        self,
        task_id: str,
        request: TaskRequest,
    ) -> TaskRequest:
        if self.output_root is None:
            return request
        if request.output is not None:
            requested = Path(request.output).expanduser()
            output = (
                requested.resolve()
                if requested.is_absolute()
                else (self.output_root / task_id / requested.name).resolve()
            )
            return request.model_copy(update={"output": str(output)})
        stem = Path(request.input).stem.strip()
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem)
        stem = (stem.rstrip(" .") or "subtitle")[:120]
        output = (self.output_root / task_id / f"{stem}.srt").resolve()
        return request.model_copy(update={"output": str(output)})

    def cancel(self, task_id: str) -> JobSnapshot:
        with self._lock:
            snapshot = self._require_snapshot(task_id)
            if snapshot.state != "running":
                return self._copy_snapshot()
            snapshot.state = "cancelled"
            snapshot.updated_at = time.time()
            event = WorkerEvent.cancelled(task_id)
            self._record_event(event)
            process = self._process
        if process is not None and process.poll() is None:
            self.terminate_process_tree(process)
        with self._lock:
            self._persist_history()
            return self._copy_snapshot()

    def snapshot(self) -> JobSnapshot | None:
        with self._lock:
            return self._copy_snapshot() if self._snapshot is not None else None

    def history(self) -> list[JobSnapshot]:
        with self._lock:
            return [
                snapshot.model_copy(deep=True)
                for snapshot in reversed(self._history)
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
        with self._lock:
            request = self._require_history(task_id).request.model_copy(deep=True)
        return self.start(request)

    def resume(self, task_id: str) -> JobSnapshot:
        with self._lock:
            self._ensure_idle()
            snapshot = self._require_history(task_id)
            if snapshot.state != "interrupted":
                raise ValueError("Only interrupted tasks can be continued")
            request = snapshot.request.model_copy(deep=True)
            process = self._spawn_worker(task_id, request)
            self._events.clear()
            self._event_base_cursor = 0
            self._process = process
            self._snapshot = snapshot
            self._history.remove(snapshot)
            self._history.append(snapshot)
            snapshot.state = "running"
            snapshot.events = []
            snapshot.error = None
            snapshot.updated_at = time.time()
            self._start_reader(task_id, process)
            self._persist_history()
            return self._copy_snapshot()

    def request_for(self, task_id: str) -> TaskRequest:
        with self._lock:
            return self._require_history(task_id).request.model_copy(deep=True)

    def _ensure_idle(self) -> None:
        if self._snapshot is not None and self._snapshot.state == "running":
            raise JobAlreadyRunning("A FineSub task is already running")

    def _spawn_worker(self, task_id: str, request: TaskRequest) -> Any:
        command = [
            self.python_executable,
            "-m",
            "desktop.backend.worker.main",
            "--task-id",
            task_id,
        ]
        environment = os.environ.copy()
        environment.update(self.worker_env)
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        process = self.process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            cwd=self.working_directory,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("worker process pipes were not created")
        process.stdin.write(
            json.dumps(
                request.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        process.stdin.flush()
        return process

    def _start_reader(self, task_id: str, process: Any) -> None:
        reader = threading.Thread(
            target=self._read_worker,
            args=(task_id, process),
            name=f"finesub-worker-{task_id[:8]}",
            daemon=True,
        )
        reader.start()

    def _read_worker(self, task_id: str, process: Any) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            event = parse_worker_line(line, task_id=task_id)
            with self._lock:
                if self._snapshot is None or self._snapshot.task_id != task_id:
                    continue
                self._record_event(event)
                self._snapshot.updated_at = time.time()
                if event.type == "completed":
                    self._snapshot.state = "completed"
                    outputs = event.payload.get("outputs", {})
                    if isinstance(outputs, dict):
                        self._snapshot.outputs = {
                            str(key): str(value)
                            for key, value in outputs.items()
                        }
                elif event.type == "failed":
                    self._snapshot.state = "failed"
                    self._snapshot.error = str(event.payload.get("message", ""))
                elif event.type == "cancelled":
                    self._snapshot.state = "cancelled"
                if event.type in {"completed", "failed", "cancelled"}:
                    self._persist_history()
        return_code = process.wait()
        with self._lock:
            if self._snapshot is None or self._snapshot.task_id != task_id:
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
                self._snapshot.state = "failed"
                self._snapshot.error = message
            self._snapshot.updated_at = time.time()
            self._persist_history()

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
        try:
            body = json.loads(self.history_path.read_text(encoding="utf-8"))
            items = body.get("tasks", []) if isinstance(body, dict) else body
            if not isinstance(items, list):
                return
            self._history = [
                JobSnapshot.model_validate(item)
                for item in items
                if isinstance(item, dict)
            ][-100:]
        except (OSError, ValueError):
            self._history = []
            return
        if not self._history:
            return
        changed = False
        for snapshot in self._history:
            if snapshot.state == "running":
                snapshot.state = "interrupted"
                snapshot.error = "应用上次退出时任务仍在运行，可以继续任务。"
                snapshot.updated_at = time.time()
                changed = True
        self._snapshot = self._history[-1]
        self._event_base_cursor = 0
        self._events = deque(
            self._snapshot.events[-self.event_limit :],
            maxlen=self.event_limit,
        )
        if changed:
            self._persist_history()

    def _persist_history(self) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(
            f"{self.history_path.suffix}.tmp"
        )
        payload = {
            "schemaVersion": 1,
            "tasks": [
                snapshot.model_dump(mode="json")
                for snapshot in self._history[-100:]
            ],
        }
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self.history_path)
