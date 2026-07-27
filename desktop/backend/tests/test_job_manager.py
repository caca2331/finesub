from __future__ import annotations

import io
import json
from pathlib import Path
import threading
import time

import pytest

from desktop.backend.common.models import TaskRequest
from desktop.backend.jobs.manager import JobAlreadyRunning, JobManager
from desktop.backend.worker.protocol import WorkerEvent, encode_event


class BlockingStdout:
    def __init__(self) -> None:
        self._closed = threading.Event()

    def __iter__(self):
        self._closed.wait(timeout=5)
        return iter(())

    def close(self) -> None:
        self._closed.set()


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 4321
        self.stdin = io.StringIO()
        self.stdout = BlockingStdout()
        self.stderr = io.StringIO()
        self.returncode: int | None = None
        self.terminated = False

    def wait(self) -> int:
        self.stdout._closed.wait(timeout=5)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


class FinishedProcess:
    def __init__(self, output: str) -> None:
        self.pid = 4322
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(output)
        self.returncode = 0

    def wait(self) -> int:
        return self.returncode

    def poll(self) -> int:
        return self.returncode


def test_job_manager_allows_only_one_active_task() -> None:
    process = FakeProcess()
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *args, **kwargs: process,
        terminate_process_tree=lambda child: None,
    )

    first = manager.start(TaskRequest(input="a.wav"))

    with pytest.raises(JobAlreadyRunning):
        manager.start(TaskRequest(input="b.wav"))
    assert first.state == "running"
    process.stdout.close()


def test_cancel_terminates_process_tree_and_marks_cancelled() -> None:
    process = FakeProcess()

    def terminate(child: FakeProcess) -> None:
        assert child.pid == 4321
        child.terminated = True
        child.returncode = 1
        child.stdout.close()

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *args, **kwargs: process,
        terminate_process_tree=terminate,
    )
    snapshot = manager.start(TaskRequest(input="a.wav"))

    result = manager.cancel(snapshot.task_id)

    assert result.state == "cancelled"
    assert process.terminated is True


def test_start_writes_only_json_request_to_worker_stdin() -> None:
    process = FakeProcess()
    captured: dict[str, object] = {}

    def start_process(*args, **kwargs):
        captured.update(kwargs)
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={"GEMINI_API_KEY": "secret"},
        working_directory="C:/FineSub/app/versions/1.2.0",
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )

    manager.start(TaskRequest(input="C:/media/a.wav", stage="raw-srt"))

    request_line = process.stdin.getvalue()
    assert request_line.endswith("\n")
    assert '"input":"C:/media/a.wav"' in request_line
    assert "secret" not in request_line
    assert captured["cwd"] == "C:/FineSub/app/versions/1.2.0"
    process.stdout.close()


def test_event_cursor_keeps_advancing_when_old_logs_are_trimmed() -> None:
    def process_factory(command, **kwargs):
        task_id = command[-1]
        output = "".join(
            [
                encode_event(WorkerEvent.log(task_id, "one")),
                encode_event(WorkerEvent.log(task_id, "two")),
                encode_event(WorkerEvent.log(task_id, "three")),
                encode_event(WorkerEvent.completed(task_id, {})),
            ]
        )
        return FinishedProcess(output)

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        event_limit=2,
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
    )

    started = manager.start(TaskRequest(input="a.wav"))
    deadline = time.monotonic() + 2
    while manager.snapshot().state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    events, cursor = manager.events_after(0)

    assert cursor == 4
    assert [event.type for event in events] == ["log", "completed"]
    assert manager.events_after(cursor) == ([], cursor)


def test_resume_reuses_interrupted_task_id_and_history_record(
    tmp_path: Path,
) -> None:
    task_id = "interrupted-task"
    request = TaskRequest(input="a.wav", output="a.srt", stage="final-srt")
    history_path = tmp_path / "tasks.json"
    history_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "tasks": [
                    {
                        "task_id": task_id,
                        "state": "running",
                        "request": request.model_dump(mode="json"),
                        "events": [],
                        "outputs": {},
                        "error": None,
                        "created_at": 10,
                        "updated_at": 11,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    process = FakeProcess()

    def process_factory(command, **kwargs):
        captured["command"] = command
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history_path,
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
    )
    assert manager.snapshot().state == "interrupted"

    resumed = manager.resume(task_id)

    assert resumed.task_id == task_id
    assert resumed.state == "running"
    assert resumed.created_at == 10
    assert len(manager.history()) == 1
    assert manager.history()[0].task_id == task_id
    assert captured["command"][-1] == task_id  # type: ignore[index]
    assert json.loads(process.stdin.getvalue())["output"] == "a.srt"
    process.stdout.close()
