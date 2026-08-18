"""Run the model prefetch entry point and translate what it prints.

Kept apart from `DesktopResourceService` so the service's `models` branch is a
few lines of policy and this file owns the one thing that is fiddly: a
subprocess whose stdout has to become the stage/log callbacks the resource
install manager already understands.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any, Protocol

from finesub_bootstrap.downloader import DownloadPaused


class _Context(Protocol):
    python_executable: Path
    working_directory: Path
    environment: dict[str, str]


class ModelPrefetchFailed(RuntimeError):
    pass


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_model_prefetch(
    model_ids: Sequence[str],
    *,
    context: _Context,
    stage: Callable[[str, str], None] | None = None,
    log: Callable[[str], None] | None = None,
    should_pause: Callable[[], bool] | None = None,
    process_factory: Callable[..., Any] = subprocess.Popen,
) -> None:
    """Fetch `model_ids` with the managed interpreter, streaming its output."""

    command = [
        str(context.python_executable),
        "-m",
        "desktop.backend.worker.prefetch",
        *model_ids,
    ]
    environment = os.environ.copy()
    environment.update(context.environment)
    process = process_factory(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
        cwd=str(context.working_directory),
        creationflags=_creation_flags(),
    )
    if process.stdout is None:  # pragma: no cover - PIPE is requested above
        process.terminate()
        process.wait()
        raise ModelPrefetchFailed("模型下载进程没有输出通道")

    lines: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        for raw in process.stdout:
            lines.put(raw)

    reader = threading.Thread(
        target=read_output,
        name="finesub-model-prefetch-output",
        daemon=True,
    )
    reader.start()
    tail: list[str] = []

    def consume(raw: str) -> None:
        nonlocal tail
        line = raw.rstrip()
        if not line:
            return
        if line.startswith("STAGE ") and stage is not None:
            # "STAGE 2/3 正在获取 Whisper 识别模型"
            _, _, remainder = line.partition(" ")
            counter, _, message = remainder.partition(" ")
            stage("downloading", f"{message}（{counter}）")
            return
        if log is not None:
            log(line)
        # Only the tail is worth keeping: a failure message is printed last,
        # and the whole transcript already went to the install log.
        tail = [*tail, line][-8:]

    def drain() -> None:
        while True:
            try:
                consume(lines.get_nowait())
            except queue.Empty:
                return

    while process.poll() is None:
        drain()
        if should_pause is not None and should_pause():
            # No resume protocol of our own -- each downloader picks up where it
            # left off the next time it runs, so stopping is the whole of it.
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(timeout=1)
            drain()
            # The install manager reads a plain return as "finished": pausing
            # has to arrive as this exception or the row flips to 已安装.
            raise DownloadPaused("模型下载已暂停")
        time.sleep(0.1)

    # The child has exited, so the reader is at most flushing what already sits
    # in the pipe before it hits EOF -- normally milliseconds. The generous
    # bound is not a budget the failure tail has to fit into; it only caps a
    # wedged pipe (a grandchild that inherited the write end and lives on).
    reader.join(timeout=30)
    drain()
    returncode = process.returncode
    if returncode != 0:
        detail = "；".join(tail[-3:]) if tail else f"退出码 {returncode}"
        raise ModelPrefetchFailed(f"模型下载失败：{detail}")
