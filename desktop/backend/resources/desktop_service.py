from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from desktop.backend.common.models import DownloadProgress, ResourceStatus
from desktop.backend.runtime.environment import RuntimeEnvironment, WorkerContext


class DesktopResourceService:
    def __init__(
        self,
        *,
        bootstrap,
        runtime: RuntimeEnvironment,
    ) -> None:
        self.bootstrap = bootstrap
        self.runtime = runtime

    def check_all(self) -> list[ResourceStatus]:
        return [self.runtime.status(), self.bootstrap.status("ffmpeg")]

    def status(self, resource_id: str) -> ResourceStatus:
        if resource_id == "uv":
            return self.runtime.status()
        if resource_id == "ffmpeg":
            return self.bootstrap.status("ffmpeg")
        raise KeyError(f"Unknown desktop resource: {resource_id}")

    def install(
        self,
        resource_id: str,
        progress: Callable[[DownloadProgress], None],
        *,
        stage: Callable[[str, str], None] | None = None,
        log: Callable[[str], None] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> ResourceStatus:
        background = any(
            callback is not None
            for callback in (stage, log, should_pause)
        )
        if resource_id == "uv":
            if background:
                self.bootstrap.install(
                    "uv",
                    progress,
                    stage=stage,
                    should_pause=should_pause,
                )
                return self.runtime.install(
                    stage=stage,
                    log=log,
                    should_pause=should_pause,
                )
            self.bootstrap.install("uv", progress)
            return self.runtime.install()
        if resource_id == "ffmpeg":
            if not background:
                return self.bootstrap.install("ffmpeg", progress)
            return self.bootstrap.install(
                "ffmpeg",
                progress,
                stage=stage,
                should_pause=should_pause,
            )
        raise KeyError(f"Unknown desktop resource: {resource_id}")

    def locations(self, resource_id: str) -> tuple[Path, Path]:
        if resource_id == "uv":
            return (
                self.bootstrap.cache_path("uv"),
                self.runtime.runtime_root,
            )
        if resource_id == "ffmpeg":
            return (
                self.bootstrap.cache_path("ffmpeg"),
                self.bootstrap.install_path("ffmpeg"),
            )
        raise KeyError(f"Unknown desktop resource: {resource_id}")

    def task_ready(self) -> bool:
        return all(status.state == "ready" for status in self.check_all())

    def worker_context(
        self,
        extra_env: Mapping[str, str],
    ) -> WorkerContext:
        ffmpeg = self.bootstrap.active_file("ffmpeg", "ffmpeg.exe")
        return self.runtime.worker_context(
            ffmpeg_bin=ffmpeg.parent if ffmpeg is not None else None,
            extra_env=extra_env,
        )
