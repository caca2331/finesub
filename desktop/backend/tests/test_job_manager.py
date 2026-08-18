from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone

import io
import json
from pathlib import Path
import threading
import time

import pytest

from desktop.backend.common.models import TaskRequest
from desktop.backend.jobs.history import JobSnapshot
from desktop.backend.jobs.launch import WorkerLaunchContext
from desktop.backend.jobs.manager import (
    JobAlreadyRunning,
    JobManager,
    JobNotFound,
)
from desktop.backend.resources.gpus import Gpu, GpuSnapshot
from desktop.backend.worker.protocol import WorkerEvent, encode_event


class BlockingStdout:
    def __init__(self) -> None:
        self._closed = threading.Event()

    def __iter__(self):
        self._closed.wait(timeout=5)
        return iter(())

    def close(self) -> None:
        self._closed.set()


class TerminalThenBlockingStdout(BlockingStdout):
    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line

    def __iter__(self):
        yield self.line
        self._closed.wait(timeout=5)


class DelayedTerminalStdout(BlockingStdout):
    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line
        self.released = threading.Event()

    def __iter__(self):
        self.released.wait(timeout=5)
        yield self.line
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
        worker_env={"GEMINI_FREE": "secret"},
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


def _spawn_environment(request: TaskRequest, **manager_kwargs) -> dict[str, str]:
    process = FakeProcess()
    captured: dict[str, object] = {}

    def start_process(*args, **kwargs):
        captured.update(kwargs)
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={"GEMINI_FREE": "secret"},
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
        **manager_kwargs,
    )
    manager.start(request)
    process.stdout.close()
    return dict(captured["env"])


def _two_cards() -> GpuSnapshot:
    return GpuSnapshot(
        state="ready",
        devices=(
            Gpu(index=0, name="NVIDIA GeForce RTX 5060 Ti", memory_mb=16311),
            Gpu(index=1, name="NVIDIA GeForce RTX 4090", memory_mb=24564),
        ),
    )


def test_the_chosen_card_is_the_only_one_the_worker_can_see() -> None:
    # Making the others invisible rather than passing an index keeps the choice
    # out of the speech stack, where three consumers each take their device
    # differently and two sit in the high-risk core.
    environment = _spawn_environment(
        TaskRequest(input="C:/media/a.wav", gpu_index=1),
        available_gpus=_two_cards,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    # nvidia-smi numbers by PCI bus and CUDA by speed; unpinned, "card 1" can
    # mean different cards in the picker and in the worker.
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_the_device_choice_survives_a_worker_environment_rebuild() -> None:
    # `worker_env` is replaced wholesale every time a resource finishes
    # installing, so the choice cannot live there -- it is merged at spawn.
    process = FakeProcess()
    captured: dict[str, object] = {}
    manager = JobManager(
        python_executable="python.exe",
        worker_env={"GEMINI_FREE": "secret"},
        available_gpus=_two_cards,
        process_factory=lambda *args, **kwargs: (
            captured.update(kwargs) or process
        ),
        terminate_process_tree=lambda child: None,
    )

    manager.set_worker_context(
        replace(manager.worker_context, environment={"GEMINI_FREE": "rebuilt"})
    )
    manager.start(TaskRequest(input="C:/media/a.wav", gpu_index=1))
    process.stdout.close()

    assert dict(captured["env"])["CUDA_VISIBLE_DEVICES"] == "1"


def test_a_card_that_is_no_longer_there_falls_back_to_automatic() -> None:
    # Retry and resume replay a stored request, so a machine that lost a card
    # would otherwise keep pointing at whatever now sits at that index.
    environment = _spawn_environment(
        TaskRequest(input="C:/media/a.wav", gpu_index=3),
        available_gpus=_two_cards,
    )

    assert "CUDA_VISIBLE_DEVICES" not in environment


def test_a_different_card_in_the_same_slot_falls_back_to_automatic() -> None:
    # An index is a position, not an identity: swap the card and the index
    # still resolves, so only the name can tell that the machine changed.
    environment = _spawn_environment(
        TaskRequest(
            input="C:/media/a.wav",
            gpu_index=1,
            gpu_name="NVIDIA GeForce RTX 3090",
        ),
        available_gpus=_two_cards,
    )

    assert "CUDA_VISIBLE_DEVICES" not in environment


def test_an_unfinished_probe_never_delays_or_overrides_the_start() -> None:
    # The real probe returns a snapshot from the first moment, with no devices
    # in it until nvidia-smi answers. Treating that as "the card is gone" would
    # override every choice for the first seconds after launch -- so the state,
    # not the emptiness of the list, is what decides.
    scanning = GpuSnapshot(state="scanning")

    environment = _spawn_environment(
        TaskRequest(input="C:/media/a.wav", gpu_index=1),
        available_gpus=lambda: scanning,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_a_machine_with_no_driver_leaves_the_choice_alone() -> None:
    # "unavailable" means nobody could be asked, which is not evidence that a
    # card is missing -- and it must not produce a "your card is gone" notice.
    environment = _spawn_environment(
        TaskRequest(input="C:/media/a.wav", gpu_index=1),
        available_gpus=lambda: GpuSnapshot(state="unavailable"),
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "1"


def test_the_fallback_is_told_to_the_task_it_affects() -> None:
    # The notice used to be recorded while spawning -- before the snapshot that
    # holds events existed, and before the deque was cleared. It reached nobody.
    process = FakeProcess()
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        available_gpus=_two_cards,
        process_factory=lambda *args, **kwargs: process,
        terminate_process_tree=lambda child: None,
    )

    snapshot = manager.start(TaskRequest(input="C:/media/a.wav", gpu_index=3))
    process.stdout.close()

    messages = [event.payload.get("message", "") for event in snapshot.events]
    assert any("显卡 #3" in message for message in messages)


def test_a_cpu_task_is_not_pinned_to_a_card() -> None:
    environment = _spawn_environment(
        TaskRequest(input="C:/media/a.wav", device="cpu", gpu_index=1),
    )

    assert "CUDA_VISIBLE_DEVICES" not in environment


def test_worker_receives_the_stable_activity_coordination_root(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.locks import TASK_ACTIVITY_ROOT_VARIABLE

    environment = _spawn_environment(
        TaskRequest(input="C:/media/a.wav"),
        history_path=tmp_path / "user-data" / "tasks.json",
        output_root=tmp_path / "big-data" / "tasks",
    )

    assert environment[TASK_ACTIVITY_ROOT_VARIABLE] == str(
        (tmp_path / "user-data").resolve()
    )


def test_interrupted_relocation_uses_the_same_fallback_root_everywhere(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import (
        AppPaths,
        ensure_store,
        load_app_paths,
        record_big_data,
    )

    installation = tmp_path / "install"
    data_root = tmp_path / "data"
    original = AppPaths.for_root(installation, data_root=data_root)
    ensure_store(original)
    original.tasks.mkdir()

    destination = tmp_path / "destination"
    destination_paths = original.with_big_data(destination)
    ensure_store(destination_paths)
    record_big_data(
        data_root,
        destination,
        migrating_from=original.big_data,
    )
    assert load_app_paths(installation, data_root=data_root).tasks == original.tasks

    process = FakeProcess()
    captured: dict[str, object] = {}

    def start_process(*args, **kwargs):
        captured.update(kwargs)
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=original.user_data / "tasks.json",
        # Simulate the incomplete `locations.json`-only choice from the old
        # worker. The full resolver must override it before anything is made.
        output_root=destination_paths.tasks,
        output_root_resolver=lambda: load_app_paths(
            installation, data_root=data_root
        ).tasks,
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )

    started = manager.start(TaskRequest(input="C:/media/a.wav"))

    assert captured["env"]["FINESUB_TASKS_ROOT"] == str(original.tasks)
    assert Path(started.request.output or "").parent.parent == original.tasks
    assert manager.task_directory(started.task_id) == original.tasks / started.task_id
    assert not destination_paths.tasks.exists()
    process.stdout.close()


def test_idle_external_relocation_refreshes_manager_and_worker_together(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import (
        AppPaths,
        ensure_store,
        load_app_paths,
        record_big_data,
    )

    installation = tmp_path / "install"
    data_root = tmp_path / "data"
    original = AppPaths.for_root(installation, data_root=data_root)
    ensure_store(original)

    process = FakeProcess()
    captured: dict[str, object] = {}

    def start_process(*args, **kwargs):
        captured.update(kwargs)
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=original.user_data / "tasks.json",
        output_root=original.tasks,
        output_root_resolver=lambda: load_app_paths(
            installation, data_root=data_root
        ).tasks,
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )

    destination = tmp_path / "destination"
    destination_paths = original.with_big_data(destination)
    ensure_store(destination_paths)
    record_big_data(data_root, destination)

    started = manager.start(TaskRequest(input="C:/media/a.wav"))

    assert captured["env"]["FINESUB_TASKS_ROOT"] == str(destination_paths.tasks)
    assert Path(started.request.output or "").parent.parent == destination_paths.tasks
    assert manager.task_directory(started.task_id) == (
        destination_paths.tasks / started.task_id
    )
    assert not original.tasks.exists()
    process.stdout.close()


def test_idle_path_operations_follow_relocation_without_recreating_old_root(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index
    from finesub_bootstrap.paths import (
        AppPaths,
        ensure_store,
        load_app_paths,
        record_big_data,
    )

    installation = tmp_path / "install"
    data_root = tmp_path / "data"
    original = AppPaths.for_root(installation, data_root=data_root)
    ensure_store(original)
    old_output = original.tasks / "old-task" / "clip.srt"
    task_index.merge_write(
        original.user_data / "tasks.json",
        [
            {
                "task_id": "old-task",
                "state": "completed",
                "request": {"input": "a.wav", "output": str(old_output)},
                "outputs": {"rawSrt": str(old_output)},
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ],
        original.tasks,
    )
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=original.user_data / "tasks.json",
        output_root=original.tasks,
        output_root_resolver=lambda: load_app_paths(
            installation, data_root=data_root
        ).tasks,
    )

    destination = tmp_path / "destination"
    current = original.with_big_data(destination)
    ensure_store(current)
    (current.tasks / "old-task").mkdir(parents=True)
    record_big_data(data_root, destination)

    opened: list[Path] = []
    listed = manager.history()[0]
    revealed_output = manager.open_owned_output(str(old_output), opened.append)
    revealed_directory = manager.open_task_directory("old-task", opened.append)

    expected_output = current.tasks / "old-task" / "clip.srt"
    assert listed.request.output == str(expected_output)
    assert listed.outputs["rawSrt"] == str(expected_output)
    assert revealed_output == expected_output
    assert revealed_directory == current.tasks / "old-task"
    assert opened == [expected_output, current.tasks / "old-task"]
    assert not original.tasks.exists()


def test_idle_manager_cannot_undo_task_output_migration(tmp_path: Path) -> None:
    from finesub_bootstrap.migrations.tasks_location import relocate
    from finesub_bootstrap.paths import AppPaths

    paths = AppPaths.for_root(
        tmp_path / "install", data_root=tmp_path / "data"
    )
    stray = paths.user_data / "tasks"
    old_output = stray / "task-a" / "clip.srt"
    old_output.parent.mkdir(parents=True)
    old_output.write_text("subtitle", encoding="utf-8")
    paths.user_data.mkdir(parents=True, exist_ok=True)
    history = paths.user_data / "tasks.json"
    history.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "tasks": [
                    {
                        "task_id": "task-a",
                        "state": "completed",
                        "request": {
                            "input": "a.wav",
                            "output": str(old_output),
                        },
                        "outputs": {"rawSrt": str(old_output)},
                        "created_at": 1.0,
                        "updated_at": 2.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history,
        output_root=paths.tasks,
    )
    assert manager.history()[0].request.output == str(old_output)

    assert relocate(paths, lambda _message: None)
    assert json.loads(history.read_text(encoding="utf-8"))["tasks"][0][
        "request"
    ]["output"] == "task-a/clip.srt"

    manager._persist_history()

    raw = json.loads(history.read_text(encoding="utf-8"))["tasks"][0]
    assert raw["request"]["output"] == "task-a/clip.srt"
    assert raw["outputs"]["rawSrt"] == "task-a/clip.srt"


def test_disk_refresh_does_not_adopt_an_external_running_task(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    tasks = tmp_path / "tasks"
    history = tmp_path / "user-data" / "tasks.json"
    completed = {
        "task_id": "task-a",
        "state": "completed",
        "request": {"input": "a.wav"},
        "outputs": {},
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    task_index.merge_write(history, [completed], tasks)
    process = FakeProcess()
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history,
        output_root=tasks,
        output_root_resolver=lambda: tasks,
        process_factory=lambda *args, **kwargs: process,
        terminate_process_tree=lambda child: None,
    )
    assert manager.snapshot().state == "completed"

    task_index.merge_write(
        history,
        [{**completed, "state": "running", "updated_at": 3.0}],
        tasks,
    )

    listed = {snapshot.task_id: snapshot for snapshot in manager.history()}
    assert listed["task-a"].state == "running"
    assert manager.snapshot().state == "completed"
    assert manager.request_for("task-a").input == "a.wav"

    started = manager.start(TaskRequest(input="b.wav"))
    assert started.state == "running"
    process.stdout.close()


def test_terminal_reader_does_not_adopt_external_running_generation(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    tasks = tmp_path / "tasks"
    history = tmp_path / "user-data" / "tasks.json"
    process = FakeProcess()
    terminated: list[FakeProcess] = []

    def start_process(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        process.stdout = TerminalThenBlockingStdout(
            encode_event(WorkerEvent.completed(task_id, {}))
        )
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history,
        output_root=tasks,
        output_root_resolver=lambda: tasks,
        process_factory=start_process,
        terminate_process_tree=terminated.append,
    )
    started = manager.start(TaskRequest(input="a.wav"))
    deadline = time.monotonic() + 2
    while manager._snapshot.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._snapshot.state == "completed"
    assert manager._process is process

    external = manager._snapshot.model_dump(mode="json")
    external["state"] = "running"
    external["updated_at"] += 1.0
    task_index.merge_write(history, [external], tasks)

    listed = {snapshot.task_id: snapshot for snapshot in manager.history()}
    assert listed[started.task_id].state == "running"
    assert manager.snapshot().state == "completed"

    cancelled = manager.cancel(started.task_id)
    assert cancelled.state == "completed"
    assert terminated == []
    process.stdout.close()


def test_delayed_old_terminal_cannot_overwrite_new_running_generation(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    tasks = tmp_path / "tasks"
    history = tmp_path / "user-data" / "tasks.json"
    process = FakeProcess()
    terminal_epoch = time.time() + 0.25

    def start_process(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        timestamp = datetime.fromtimestamp(
            terminal_epoch, timezone.utc
        ).isoformat().replace("+00:00", "Z")
        process.stdout = DelayedTerminalStdout(
            encode_event(
                WorkerEvent(
                    type="completed",
                    task_id=task_id,
                    timestamp=timestamp,
                    payload={"outputs": {}},
                )
            )
        )
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history,
        output_root=tasks,
        output_root_resolver=lambda: tasks,
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )
    started = manager.start(TaskRequest(input="a.wav"))

    successor = manager._snapshot.model_dump(mode="json")
    successor["state"] = "running"
    successor["updated_at"] = terminal_epoch + 1.0
    task_index.merge_write(history, [successor], tasks)

    assert isinstance(process.stdout, DelayedTerminalStdout)
    process.stdout.released.set()
    deadline = time.monotonic() + 2
    while manager._snapshot.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    disk = {
        entry["task_id"]: entry
        for entry in task_index.read(history, tasks)
    }
    assert disk[started.task_id]["state"] == "running"
    assert disk[started.task_id]["updated_at"] == terminal_epoch + 1.0
    assert manager.snapshot().state == "completed"


@pytest.mark.parametrize("action", ["cancel", "shutdown"])
def test_local_termination_cannot_overwrite_new_running_generation(
    tmp_path: Path,
    action: str,
) -> None:
    from finesub_bootstrap import task_index

    tasks = tmp_path / "tasks"
    history = tmp_path / "user-data" / "tasks.json"
    process = FakeProcess()
    terminal_epoch = time.time() + 0.25
    terminated: list[FakeProcess] = []

    def start_process(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        timestamp = datetime.fromtimestamp(
            terminal_epoch, timezone.utc
        ).isoformat().replace("+00:00", "Z")
        process.stdout = DelayedTerminalStdout(
            encode_event(
                WorkerEvent(
                    type="completed",
                    task_id=task_id,
                    timestamp=timestamp,
                    payload={"outputs": {}},
                )
            )
        )
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history,
        output_root=tasks,
        process_factory=start_process,
        terminate_process_tree=terminated.append,
    )
    started = manager.start(TaskRequest(input="a.wav"))

    successor = manager._snapshot.model_dump(mode="json")
    successor["state"] = "running"
    successor["updated_at"] = terminal_epoch + 1.0
    task_index.merge_write(history, [successor], tasks)

    if action == "cancel":
        result = manager.cancel(started.task_id)
        assert result.state == "cancelled"
    else:
        manager.shutdown()
        assert manager.snapshot().state == "interrupted"
    assert terminated == [process]

    disk = task_index.read(history, tasks)[0]
    assert disk["state"] == "running"
    assert disk["updated_at"] == terminal_epoch + 1.0

    assert isinstance(process.stdout, DelayedTerminalStdout)
    process.stdout.released.set()
    deadline = time.monotonic() + 2
    while not process.stdout._closed.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    disk = task_index.read(history, tasks)[0]
    assert disk["state"] == "running"
    assert disk["updated_at"] == terminal_epoch + 1.0


def test_retired_reader_cannot_change_same_id_retry(tmp_path: Path) -> None:
    first = FakeProcess()
    second = FakeProcess()
    first_waited = threading.Event()

    def first_wait():
        first_waited.set()
        return first.returncode

    first.wait = first_wait
    processes = iter((first, second))

    def start_process(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        process = next(processes)
        if process is first:
            process.stdout = TerminalThenBlockingStdout(
                encode_event(WorkerEvent.completed(task_id, {}))
            )
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )
    started = manager.start(TaskRequest(input="a.wav"))
    deadline = time.monotonic() + 2
    while manager._snapshot.state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._snapshot.state == "completed"

    retried = manager.retry(started.task_id)
    assert retried.state == "running"
    assert manager._process is second

    assert isinstance(first.stdout, TerminalThenBlockingStdout)
    first.stdout.close()
    assert first_waited.wait(timeout=2)
    assert manager.snapshot().state == "running"
    assert manager._process is second

    second.stdout.close()


def test_output_aliases_stay_current_after_a_to_b_to_a(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    root_a = (tmp_path / "a" / "tasks").resolve()
    root_b = (tmp_path / "b" / "tasks").resolve()
    current_root = [root_a]
    history = tmp_path / "user-data" / "tasks.json"
    output_a = root_a / "task-a" / "clip.srt"
    output_b = root_b / "task-a" / "clip.srt"
    task_index.merge_write(
        history,
        [
            {
                "task_id": "task-a",
                "state": "completed",
                "request": {"input": "a.wav", "output": str(output_a)},
                "outputs": {"rawSrt": str(output_a)},
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ],
        root_a,
    )
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=history,
        output_root=root_a,
        output_root_resolver=lambda: current_root[0],
    )

    current_root[0] = root_b
    assert manager.history()[0].outputs["rawSrt"] == str(output_b)
    current_root[0] = root_a
    assert manager.history()[0].outputs["rawSrt"] == str(output_a)

    opened: list[Path] = []
    revealed = manager.open_owned_output(str(output_b), opened.append)

    assert revealed == output_a
    assert opened == [output_a]
    assert manager._path_aliases.get(output_a) is None
    assert manager._path_aliases[output_b] == output_a


def test_cached_reuse_request_follows_an_idle_external_relocation(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import (
        AppPaths,
        ensure_store,
        load_app_paths,
        record_big_data,
    )

    installation = tmp_path / "install"
    data_root = tmp_path / "data"
    original = AppPaths.for_root(installation, data_root=data_root)
    ensure_store(original)
    stale_output = original.tasks / "old-task" / "clip.srt"

    process = FakeProcess()
    captured: dict[str, object] = {}

    def start_process(*args, **kwargs):
        captured.update(kwargs)
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=original.user_data / "tasks.json",
        output_root=original.tasks,
        output_root_resolver=lambda: load_app_paths(
            installation, data_root=data_root
        ).tasks,
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )

    destination = tmp_path / "destination"
    destination_paths = original.with_big_data(destination)
    ensure_store(destination_paths)
    (destination_paths.tasks / "old-task").mkdir(parents=True)
    record_big_data(data_root, destination)

    started = manager.start(
        TaskRequest(
            input="C:/media/a.wav",
            stage="final-srt",
            output=str(stale_output),
        )
    )

    expected = destination_paths.tasks / "old-task" / "clip.srt"
    worker_request = json.loads(process.stdin.getvalue())
    assert manager.output_root == destination_paths.tasks
    assert captured["env"]["FINESUB_TASKS_ROOT"] == str(destination_paths.tasks)
    assert started.request.output == str(expected)
    assert worker_request["output"] == str(expected)
    assert not original.tasks.exists()
    process.stdout.close()


def test_manager_holds_its_activity_lease_until_reader_finishes(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.locks import activity_is_idle

    process = FakeProcess()

    def start_process(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        process.stdout = TerminalThenBlockingStdout(
            encode_event(WorkerEvent.completed(task_id, {}))
        )
        return process

    coordination_root = tmp_path / "user-data"
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=coordination_root / "tasks.json",
        output_root=tmp_path / "tasks",
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )

    manager.start(TaskRequest(input="C:/media/a.wav"))
    deadline = time.monotonic() + 2
    while manager.snapshot().state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.snapshot().state == "completed"
    assert not activity_is_idle(coordination_root)

    process.stdout.close()
    # Wait for the lease itself, not for poll(): the reader publishes the
    # return code inside `process.wait()` and only releases its lease in the
    # `finally` that follows -- deliberately, per the handoff comment in
    # `_read_worker`. Between those two moments poll() already answers while
    # the lease is still held, so treating poll() as "the reader settled"
    # is exactly the race this test would then fail on.
    deadline = time.monotonic() + 2
    while not activity_is_idle(coordination_root) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert activity_is_idle(coordination_root)


def test_overlapping_readers_keep_independent_activity_leases(
    tmp_path: Path, monkeypatch
) -> None:
    from desktop.backend.jobs import manager as manager_module

    active = 0
    counts_before_entry: list[int] = []

    @contextmanager
    def tracked_activity(_root):
        nonlocal active
        counts_before_entry.append(active)
        active += 1
        try:
            yield
        finally:
            active -= 1

    monkeypatch.setattr(manager_module, "holding_activity", tracked_activity)
    first = FakeProcess()
    second = FakeProcess()
    processes = [first, second]

    def start_process(command, **kwargs):
        process = processes.pop(0)
        if process is first:
            task_id = command[command.index("--task-id") + 1]
            process.stdout = TerminalThenBlockingStdout(
                encode_event(WorkerEvent.completed(task_id, {}))
            )
        return process

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        history_path=tmp_path / "user-data" / "tasks.json",
        output_root=tmp_path / "tasks",
        process_factory=start_process,
        terminate_process_tree=lambda child: None,
    )

    manager.start(TaskRequest(input="C:/media/a.wav"))
    deadline = time.monotonic() + 2
    while manager.snapshot().state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.snapshot().state == "completed"
    assert active == 1

    manager.start(TaskRequest(input="C:/media/b.wav"))

    assert counts_before_entry == [0, 1]
    assert active == 2
    first.stdout.close()
    deadline = time.monotonic() + 2
    while active == 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert active == 1
    second.stdout.close()
    deadline = time.monotonic() + 2
    while active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert active == 0


def test_start_assigns_persistent_output_path(tmp_path: Path) -> None:
    process = FakeProcess()
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        output_root=tmp_path / "user-data" / "tasks",
        process_factory=lambda *args, **kwargs: process,
        terminate_process_tree=lambda child: None,
    )

    started = manager.start(TaskRequest(input="C:/media/my:video.wav"))

    output = Path(started.request.output or "")
    assert output.is_absolute()
    assert output.parent.name == started.task_id
    assert output.name == "my_video.srt"
    assert json.loads(process.stdin.getvalue())["output"] == str(output)
    process.stdout.close()


def test_relative_explicit_output_is_anchored_to_task_storage(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    output_root = tmp_path / "user-data" / "tasks"
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        output_root=output_root,
        process_factory=lambda *args, **kwargs: process,
        terminate_process_tree=lambda child: None,
    )

    started = manager.start(
        TaskRequest(input="a.wav", output="../custom.srt")
    )

    output = Path(started.request.output or "")
    assert output.parent == output_root.resolve() / started.task_id
    assert output.name == "custom.srt"
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

    manager.start(TaskRequest(input="a.wav"))
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


def test_task_ids_name_the_source_and_the_time(tmp_path: Path) -> None:
    # A bare uuid gave the user no way to tell which folder under
    # user-data/tasks belonged to which job.
    from desktop.backend.jobs.launch import new_task_id

    at = time.mktime((2026, 8, 6, 3, 35, 0, 0, 0, -1))
    task_id = new_task_id(TaskRequest(input="D:/media/BV1cJjE6cEt8.mp4"), now=at)

    assert task_id.startswith("BV1cJjE6cEt8-260806-0335-")
    assert len(task_id.rsplit("-", 1)[1]) == 6


def test_the_chosen_name_wins_over_the_source_filename() -> None:
    from desktop.backend.jobs.launch import task_stem

    assert task_stem(TaskRequest(input="D:/media/raw.mp4", name="我的切片")) == "我的切片"
    assert task_stem(TaskRequest(input="https://example.test/v")) == "v"


def test_characters_a_directory_cannot_hold_are_replaced() -> None:
    from desktop.backend.jobs.launch import task_stem

    assert "?" not in task_stem(TaskRequest(input="C:/a/what?.mp4"))
    assert task_stem(TaskRequest(input="C:/a/....mp4")) == "subtitle"


def test_history_survives_the_tasks_folder_moving(tmp_path: Path) -> None:
    # Outputs used to be recorded as absolute paths, so `finesub relocate` --
    # or a user dragging the folder -- left every entry in the task list
    # pointing at nothing while the files sat right there.
    history_path = tmp_path / "tasks.json"
    first_root = tmp_path / "here" / "tasks"
    manager = JobManager(
        python_executable="python",
        worker_env={},
        history_path=history_path,
        output_root=first_root,
    )
    manager._history = [
        JobSnapshot(
            task_id="clip-260806-0335-abc123",
            state="completed",
            request=TaskRequest(
                input="clip.wav",
                output=str(first_root / "clip-260806-0335-abc123" / "clip.srt"),
            ),
            outputs={
                "srt": str(first_root / "clip-260806-0335-abc123" / "clip.srt")
            },
            created_at=1.0,
            updated_at=2.0,
        )
    ]
    manager._persist_history()

    stored = json.loads(history_path.read_text("utf-8"))["tasks"][0]
    assert stored["request"]["output"] == "clip-260806-0335-abc123/clip.srt"
    assert stored["outputs"]["srt"] == "clip-260806-0335-abc123/clip.srt"

    moved_root = tmp_path / "elsewhere" / "tasks"
    reopened = JobManager(
        python_executable="python",
        worker_env={},
        history_path=history_path,
        output_root=moved_root,
    )

    restored = reopened.history()[0]
    assert restored.request.output == str(
        moved_root.resolve() / "clip-260806-0335-abc123" / "clip.srt"
    )
    assert restored.outputs["srt"] == str(
        moved_root.resolve() / "clip-260806-0335-abc123" / "clip.srt"
    )


def test_an_output_the_user_chose_stays_where_they_put_it(tmp_path: Path) -> None:
    history_path = tmp_path / "tasks.json"
    chosen = tmp_path / "Desktop" / "my-subtitle.srt"
    manager = JobManager(
        python_executable="python",
        worker_env={},
        history_path=history_path,
        output_root=tmp_path / "tasks",
    )
    manager._history = [
        JobSnapshot(
            task_id="clip-260806-0335-abc123",
            state="completed",
            request=TaskRequest(input="clip.wav", output=str(chosen)),
            created_at=1.0,
            updated_at=2.0,
        )
    ]
    manager._persist_history()

    stored = json.loads(history_path.read_text("utf-8"))["tasks"][0]
    assert stored["request"]["output"] == str(chosen)


def _completed_manager(tmp_path, output_root=None):
    """A manager whose worker finishes immediately with one log line."""

    def process_factory(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        return FinishedProcess(
            "".join(
                [
                    encode_event(WorkerEvent.log(task_id, "working")),
                    encode_event(WorkerEvent.completed(task_id, {})),
                ]
            )
        )

    return JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
        history_path=tmp_path / "tasks.json",
        output_root=output_root if output_root is not None else tmp_path / "tasks",
    )


def _settle(manager, deadline_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + deadline_seconds
    while manager.snapshot().state == "running" and time.monotonic() < deadline:
        time.sleep(0.01)


def test_retry_keeps_the_task_id_so_outputs_and_log_stay_together(tmp_path) -> None:
    """A new id used to be paired with the old, already-resolved output path.

    The retry then wrote its subtitle into the *previous* task's directory
    while "open folder" pointed at a fresh one holding only `task-log.txt`.
    Reusing the id also lets the pipeline skip stages whose artifacts exist, so
    a failure at the LLM stage no longer redoes separation and ASR.
    """
    manager = _completed_manager(tmp_path)
    first = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))
    _settle(manager)

    again = manager.retry(first.task_id)
    _settle(manager)

    assert again.task_id == first.task_id
    assert again.request.output == first.request.output
    assert Path(again.request.output).parent.name == again.task_id


def test_task_directory_refuses_anything_that_is_not_a_task_id(tmp_path) -> None:
    """On Windows `root / "C:" + sep + "Windows"` discards the root entirely."""
    manager = _completed_manager(tmp_path)

    hostile = [
        "../escaped",
        ".." + chr(92) + "escaped",
        "C:" + chr(92) + "Windows",
        "a/b",
        "..",
        ".",
    ]
    for candidate in hostile:
        with pytest.raises(ValueError):
            manager.task_directory(candidate)

    assert manager.task_directory().is_dir()


def test_finished_tasks_do_not_keep_their_events_on_disk(tmp_path) -> None:
    """Events were the whole weight of a file the launcher calls a small index.

    The full log already lives beside the outputs (`task-log.txt`), so the copy
    here bought nothing and cost the write, the merge and the read on every
    persist -- all under the lock the UI thread waits on.
    """
    manager = _completed_manager(tmp_path)
    snapshot = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))
    _settle(manager)

    stored = json.loads((tmp_path / "tasks.json").read_text(encoding="utf-8"))
    entry = next(
        task for task in stored["tasks"] if task["task_id"] == snapshot.task_id
    )
    assert entry["state"] == "completed"
    assert entry["events"] == []


def test_the_task_log_keeps_lines_the_event_deque_drops(tmp_path) -> None:
    """The UI's event budget must not shorten the log that outlives the app.

    Rebuilding the file from the snapshot at the end meant a noisy run pushed
    its own beginning out of the only copy that survives closing the window --
    exactly the run whose beginning someone later needs.
    """

    def process_factory(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        lines = [
            encode_event(WorkerEvent.log(task_id, f"line {index}"))
            for index in range(50)
        ]
        lines.append(encode_event(WorkerEvent.completed(task_id, {})))
        return FinishedProcess("".join(lines))

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
        event_limit=5,
    )
    snapshot = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))
    _settle(manager)

    written = (
        tmp_path / "tasks" / snapshot.task_id / "task-log.txt"
    ).read_text(encoding="utf-8")
    assert "line 0" in written, "the deque dropped it; the file must not"
    assert "line 49" in written
    assert len(written.splitlines()) == 50


def test_debug_detail_reaches_the_file_but_not_the_drawer(tmp_path) -> None:
    """Diagnosing a desktop run must not mean reproducing it on the CLI.

    The drawer holds a few hundred events; per-group recovery detail would
    spend that whole budget on rows the window does not even render.
    """

    def process_factory(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        return FinishedProcess(
            "".join(
                [
                    encode_event(WorkerEvent.log(task_id, "visible line")),
                    encode_event(WorkerEvent.debug(task_id, "beam rescue group=12")),
                    encode_event(WorkerEvent.completed(task_id, {})),
                ]
            )
        )

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
    )
    snapshot = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))
    _settle(manager)

    written = (
        tmp_path / "tasks" / snapshot.task_id / "task-log.txt"
    ).read_text(encoding="utf-8")
    assert "beam rescue group=12" in written
    assert "visible line" in written

    kinds = {event.type for event in manager.snapshot().events}
    assert "debug" not in kinds
    assert "log" in kinds


def test_a_killed_worker_leaves_its_partial_log_behind(tmp_path) -> None:
    """A run that ends by being killed is the case this file exists for."""

    def process_factory(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        return FinishedProcess(
            encode_event(WorkerEvent.log(task_id, "got this far"))
        )

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
    )
    snapshot = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))
    _settle(manager)

    directory = tmp_path / "tasks" / snapshot.task_id
    # The worker died without a terminal event; the manager still ends the run,
    # so the log is renamed rather than left as a part.
    written = (directory / "task-log.txt").read_text(encoding="utf-8")
    assert "got this far" in written
    assert not (directory / "task-log.txt.part").exists()


def test_a_run_with_no_output_root_does_not_crash_the_reader(tmp_path) -> None:
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda command, **kwargs: FinishedProcess(""),
        terminate_process_tree=lambda child: None,
    )
    manager.start(TaskRequest(input="a.wav"))
    _settle(manager)

    assert manager.snapshot().state in {"completed", "failed"}


def test_a_spawn_sees_one_worker_context_even_if_it_is_swapped(tmp_path) -> None:
    """Interpreter, cwd and environment used to be three separate stores.

    Two callers rewrote them without the manager's lock -- the settings bridge
    and the resource installer's `on_ready`, from its own thread -- while
    `_spawn_worker` read them at three different points. Finishing an install
    just as the user pressed Start could pair the new interpreter with the old
    PYTHONPATH, which surfaces as `ModuleNotFoundError: No module named
    'desktop'` and reads as a corrupt install.
    """
    seen = {}

    def process_factory(command, **kwargs):
        # Swap the context underneath the spawn, between its reads.
        manager.set_worker_context(
            WorkerLaunchContext(
                python_executable="new.exe",
                working_directory="new-cwd",
                environment={"MARKER": "new"},
            )
        )
        seen["executable"] = command[0]
        seen["cwd"] = kwargs["cwd"]
        seen["marker"] = kwargs["env"].get("MARKER")
        return FinishedProcess("")

    manager = JobManager(
        python_executable="old.exe",
        worker_env={"MARKER": "old"},
        working_directory="old-cwd",
        process_factory=process_factory,
        terminate_process_tree=lambda child: None,
    )
    manager.start(TaskRequest(input="a.wav"))

    assert seen == {
        "executable": "old.exe",
        "cwd": "old-cwd",
        "marker": "old",
    }, "the spawn must not be a mix of two contexts"


def test_quitting_terminates_a_running_task(tmp_path) -> None:
    """Quitting means quitting; minimising to the tray is a different act.

    The worker is its own process group, so closing the window used to leave it
    running: still holding the GPU, still writing into the tasks tree, still
    committing to the knowledge git, with no interface left that could see or
    stop it. Left `interrupted` rather than `cancelled` because the artifacts
    already on disk are reusable -- the pipeline skips stages whose outputs
    exist, so continuing later costs only what was actually lost.
    """
    terminated = []
    process = FakeProcess()

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *a, **k: process,
        terminate_process_tree=terminated.append,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
    )
    manager.start(TaskRequest(input="a.wav"))
    assert manager.snapshot().state == "running"

    manager.shutdown()

    assert terminated == [process], "the worker must not outlive the window"
    assert manager.snapshot().state == "interrupted"
    process.stdout.close()


def test_shutdown_is_a_no_op_without_a_running_task(tmp_path) -> None:
    terminated = []
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        terminate_process_tree=terminated.append,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
    )

    manager.shutdown()

    assert terminated == []


def test_a_history_write_that_fails_does_not_change_what_happened(
    tmp_path, monkeypatch
) -> None:
    """It used to raise, and every caller was somewhere that could not afford it.

    From the reader thread an OSError killed the thread, so a task that had
    *finished* was never written down and came back next launch as
    "interrupted", with no task-log.txt either. From `start()` it threw after
    the worker was already live: the UI reported a failed launch while a real
    transcription ran, and every later Start answered "a task is already
    running" for a task nothing could see.
    """
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *a, **k: FinishedProcess(""),
        terminate_process_tree=lambda child: None,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
    )

    def explode(self):
        raise OSError("the history file is held by a backup tool")

    monkeypatch.setattr(JobManager, "_persist_history_locked", explode)

    snapshot = manager.start(TaskRequest(input="a.wav"))

    assert snapshot.state == "running", "the launch succeeded; the record did not"


def test_the_window_shows_a_hundred_tasks_and_the_index_keeps_the_rest(
    tmp_path,
) -> None:
    """Trimming is what a list can render, not what a record may hold.

    The cap used to live in the stored file, which meant a run -- from either
    front end, since they share the index now -- pushed somebody's older task
    out of the record entirely. A task worth continuing is worth finding
    however long ago it ran.
    """

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    task_index.merge_write(
        index,
        [
            {
                "task_id": f"task-{number:03d}",
                "state": "completed",
                "request": {"input": f"{number}.wav"},
                "created_at": float(number),
                "updated_at": float(number),
            }
            for number in range(150)
        ],
        tmp_path / "tasks",
    )

    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *a, **k: FinishedProcess(""),
        terminate_process_tree=lambda child: None,
        history_path=index,
        output_root=tmp_path / "tasks",
    )

    rendered = manager.history()

    assert len(rendered) == 100
    assert rendered[0].task_id == "task-149", "newest first"
    assert len(task_index.read(index, tmp_path / "tasks")) == 150


def _index_with_a_live_foreign_task(tmp_path):
    """A `tasks.json` holding a `running` task this manager did not start."""

    from finesub_bootstrap import task_index

    index = tmp_path / "tasks.json"
    task_index.merge_write(
        index,
        [
            {
                "task_id": "cli-task",
                "state": "running",
                "request": {"input": "a.wav"},
                "created_at": 1.0,
                "updated_at": 1.0,
            }
        ],
        tmp_path / "tasks",
    )
    return index


def test_a_task_this_window_did_not_start_is_not_its_own(tmp_path) -> None:
    """Adopting one hands the whole UI to a job we cannot see or stop.

    The drawer, the Cancel button and `shutdown()` all act on `_snapshot`, and
    none of them can reach another process's worker: they would report the task
    stopped while it carried on writing its artifacts and its knowledge commits.
    """

    from finesub_bootstrap import task_index
    from finesub_bootstrap.locks import holding_lock, task_lock_path

    index = _index_with_a_live_foreign_task(tmp_path)
    lock_path = task_lock_path(tmp_path / "tasks", "cli-task")
    with holding_lock(lock_path):
        manager = JobManager(
            python_executable="python.exe",
            worker_env={},
            process_factory=lambda *a, **k: FinishedProcess(""),
            terminate_process_tree=lambda child: None,
            history_path=index,
            output_root=tmp_path / "tasks",
        )

    assert manager.snapshot() is None, "not ours to show as the current task"

    manager.shutdown()
    with pytest.raises(JobNotFound):
        manager.cancel("cli-task")

    stored = task_index.read(index, tmp_path / "tasks")[-1]
    assert stored["state"] == "running", "we cannot stop it, so we must not say we did"


def test_a_crashed_running_mark_is_still_repaired(tmp_path) -> None:
    # The task sidecar remains after a crash, but the OS has released its lock.
    from finesub_bootstrap import task_index
    from finesub_bootstrap.locks import holding_lock, task_lock_path

    index = _index_with_a_live_foreign_task(tmp_path)
    with holding_lock(task_lock_path(tmp_path / "tasks", "cli-task")):
        pass
    JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *a, **k: FinishedProcess(""),
        terminate_process_tree=lambda child: None,
        history_path=index,
        output_root=tmp_path / "tasks",
    )

    stored = task_index.read(index, tmp_path / "tasks")[-1]
    assert stored["state"] == "interrupted"


def test_a_legacy_running_mark_without_a_task_lock_is_left_alone(tmp_path) -> None:
    """An absent sidecar cannot distinguish an old live process from a crash."""

    from finesub_bootstrap import task_index

    index = _index_with_a_live_foreign_task(tmp_path)
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *a, **k: FinishedProcess(""),
        terminate_process_tree=lambda child: None,
        history_path=index,
        output_root=tmp_path / "tasks",
    )

    assert manager.snapshot() is None
    stored = task_index.read(index, tmp_path / "tasks")[-1]
    assert stored["state"] == "running"


def _finished_task_with_artifacts(tmp_path):
    """A completed task with the files a real run leaves in its directory."""

    manager = _completed_manager(tmp_path)
    snapshot = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))
    _settle(manager)
    output = Path(snapshot.request.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    bulky = stem.with_name(f"{stem.name}-vocal.ogg")
    bulky.write_bytes(b"vocal")
    keeper = stem.with_name(f"{stem.name}-stable.json")
    keeper.write_text("{}", encoding="utf-8")
    checkpoints = stem.with_name(f"{stem.name}.llm-artifacts")
    checkpoints.mkdir()
    (checkpoints / "session-checkpoints.jsonl").write_text("{}", encoding="utf-8")
    return manager, snapshot, bulky, keeper, checkpoints


def test_deleting_intermediates_keeps_what_a_rerun_reads(tmp_path) -> None:
    manager, snapshot, bulky, keeper, checkpoints = _finished_task_with_artifacts(
        tmp_path
    )

    manager.delete_intermediates(snapshot.task_id)

    assert not bulky.exists()
    assert keeper.exists()
    # A finished task has nothing to resume, so its checkpoints go too --
    # they are the largest thing left in the directory.
    assert not checkpoints.exists()


def test_an_unfinished_task_keeps_the_checkpoints_it_would_resume_from(
    tmp_path,
) -> None:
    # The button is offered on failed tasks precisely because those are the
    # ones that leave a vocal track behind. Freeing that space must not cost
    # the user every LLM call they already paid for.
    manager, snapshot, bulky, _keeper, checkpoints = _finished_task_with_artifacts(
        tmp_path
    )
    manager._require_history(snapshot.task_id).state = "interrupted"

    manager.delete_intermediates(snapshot.task_id)

    assert not bulky.exists()
    assert (checkpoints / "session-checkpoints.jsonl").is_file()


def test_deleting_intermediates_refuses_the_running_task(tmp_path) -> None:
    manager = JobManager(
        python_executable="python.exe",
        worker_env={},
        process_factory=lambda *a, **k: FakeProcess(),
        terminate_process_tree=lambda child: None,
        history_path=tmp_path / "tasks.json",
        output_root=tmp_path / "tasks",
    )
    snapshot = manager.start(TaskRequest(input=str(tmp_path / "a.wav"), name="demo"))

    with pytest.raises(JobAlreadyRunning):
        manager.delete_intermediates(snapshot.task_id)


def test_a_task_the_index_says_is_running_is_not_cleaned_up(tmp_path) -> None:
    # The index is shared: a task the CLI is running right now appears in this
    # manager's history as `running` with no process of ours behind it.
    manager, snapshot, bulky, _keeper, _checkpoints = _finished_task_with_artifacts(
        tmp_path
    )
    manager._require_history(snapshot.task_id).state = "running"

    with pytest.raises(JobAlreadyRunning):
        manager.delete_intermediates(snapshot.task_id)

    assert bulky.exists()
