from __future__ import annotations

import io
import threading

import pytest

from desktop.backend.common.models import TaskRequest
from desktop.backend.jobs.manager import JobAlreadyRunning, JobManager


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
