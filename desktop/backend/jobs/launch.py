"""Where a worker runs, and how one is started.

The manager decides *whether* to launch and owns what happens afterwards; this
module answers the questions that are settled before the process exists -- what
the task is called, which card it may see, and which interpreter, directory and
environment it inherits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from finesub_bootstrap.locks import TASK_ACTIVITY_ROOT_VARIABLE
from finesub_bootstrap import task_output

from desktop.backend.common.models import TaskRequest


ProcessFactory = Callable[..., Any]
ProcessTerminator = Callable[[Any], None]
OutputRootResolver = Callable[[], str | Path]


@dataclass(frozen=True)
class WorkerLaunchContext:
    """Everything about *where* a worker runs, swapped as one value.

    The interpreter, the working directory and the environment were three
    separate attributes, rewritten by two callers (the settings bridge and the
    resource installer's `on_ready`, from its own thread) without taking the
    manager's lock, while `spawn_worker` read them at three different points
    of the same function. Finishing an install at the moment the user pressed
    Start could therefore pair the new interpreter with the old PYTHONPATH --
    surfacing as `ModuleNotFoundError: No module named 'desktop'`, which reads
    as a corrupt install. One immutable value replaced atomically removes the
    torn read rather than narrowing it.
    """

    python_executable: str
    working_directory: str | None
    environment: dict[str, str]


# The rules themselves live in `finesub_bootstrap.task_output`, shared with the
# CLI so both front ends name and place a task the same way. What stays here is
# only the adapter from this front end's request object.


def task_stem(request: TaskRequest) -> str:
    """Filesystem-safe stem for one task: the chosen name, else the source."""

    return task_output.task_stem(name=request.name, source=request.input)


def new_task_id(request: TaskRequest, *, now: float | None = None) -> str:
    """``<stem>-YYMMDD-HHMM-<6 hex>``."""

    return task_output.new_task_id(task_stem(request), now=now)


def device_environment(
    request: TaskRequest,
    available_gpus: Callable[[], Any],
) -> tuple[dict[str, str], str]:
    """Point the worker at one GPU, if the task asked for a particular one.

    Returns the environment additions and a notice to show the user, if the
    request had to be overridden. A notice rather than a recorded event because
    this runs before the task's snapshot exists -- anything recorded now is
    dropped or lands on the previous task. `JobManager._replay_device_notice`
    is where it finally reaches the user.

    Making the card invisible rather than threading an index through the
    speech stack is deliberate: separation, faster-whisper and the verifier
    each take their device differently, and two of them sit in the code the
    project treats as high-risk. With one card visible they all see device
    0 and the GPU budget tiers keep meaning what they say.

    Applied at spawn time because `worker_env` is rebuilt wholesale
    whenever a resource finishes installing -- and because retry and resume
    replay a stored request that never passes through the bridge.
    """

    index = request.gpu_index
    if index is None or request.device == "cpu":
        return {}, ""
    pinned = {
        "CUDA_VISIBLE_DEVICES": str(index),
        # nvidia-smi numbers cards by PCI bus; CUDA defaults to ordering by
        # speed. Without pinning this, "card 1" can mean two different
        # cards in the picker and in the worker.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    available = available_gpus()
    # Only a finished probe can say a card is missing. While it is still
    # scanning -- or where there is no driver to ask -- the stored choice
    # stands: a task must never wait for the probe, and "we have not looked
    # yet" is not evidence that the card is gone.
    if getattr(available, "state", None) != "ready":
        return pinned, ""
    present = available.find(index)
    if present is None:
        return {}, f"显卡 #{index} 已不在这台机器上，本次改用自动选择"
    if request.gpu_name and present.name != request.gpu_name:
        # Same slot, different card: the index still resolves, so nothing
        # else would notice that the task is about to run on hardware the
        # user never picked.
        return {}, (
            f"显卡 #{index} 现在是 {present.name}（原为 {request.gpu_name}），"
            "本次改用自动选择"
        )
    return pinned, ""


def spawn_worker(
    context: WorkerLaunchContext,
    task_id: str,
    request: TaskRequest,
    *,
    available_gpus: Callable[[], Any],
    process_factory: ProcessFactory,
    output_root: Path | None,
    history_path: Path | None,
) -> tuple[Any, str]:
    """Start the worker for one task; return it and any device notice.

    `context` is taken as one value so the interpreter, cwd and environment
    below are guaranteed to belong to the same context even if an install
    finishes mid-spawn.
    """

    command = [
        context.python_executable,
        "-m",
        "desktop.backend.worker.main",
        "--task-id",
        task_id,
    ]
    environment = os.environ.copy()
    environment.update(context.environment)
    pinned, notice = device_environment(request, available_gpus)
    environment.update(pinned)
    if output_root is not None:
        # So the worker can take the "something is writing here" lease
        # itself. Held by the launcher it said nothing about an orphan.
        environment["FINESUB_TASKS_ROOT"] = str(output_root)
    if history_path is not None:
        # Stable outside the big-data tree: workers publish independent
        # activity leases here so relocation can see all concurrent runs.
        environment[TASK_ACTIVITY_ROOT_VARIABLE] = str(history_path.parent)
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if os.name == "nt"
        else 0
    )
    process = process_factory(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
        cwd=context.working_directory,
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
    return process, notice
