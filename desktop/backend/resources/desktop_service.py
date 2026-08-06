from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from finesub_bootstrap.environment import (
    RuntimeEnvironment,
    WorkerContext,
    shared_environment_overrides,
)
from finesub_bootstrap.capabilities import required_capabilities
from finesub_bootstrap.models import DownloadProgress, ResourceStatus
from finesub_bootstrap.system_tools import (
    SystemTool,
    find_system_ffmpeg,
    find_system_git,
)


# Everything the ResourceManager installs the same way. `uv` is not here: it
# also drives the Python runtime install, so it keeps its own branch.
BOOTSTRAP_RESOURCES = ("ffmpeg", "git", "yt-dlp")

# Needed before any task can run. git and yt-dlp are deliberately absent --
# only `knowledge=update` needs git and only URL input needs yt-dlp, so
# requiring them up front would make every user pay for capabilities most
# never use. `capability_requirements` below is what gates those.
ALWAYS_REQUIRED = ("uv", "ffmpeg")

# Tools the user may already have. Reusing a capable system copy is worth ~150MB
# on a machine that has ffmpeg. yt-dlp is missing on purpose: the pipeline
# imports it from the managed interpreter, which cannot see the user's
# site-packages, so a system install is invisible and there is nothing to reuse.
SYSTEM_TOOL_FINDERS = {
    "ffmpeg": find_system_ffmpeg,
    "git": find_system_git,
}


def capability_requirements(request) -> tuple[str, ...]:
    """Which on-demand resources this particular request needs.

    Thin adapter over the shared rule so the desktop and the CLI cannot drift
    into disagreeing about what a run requires.
    """

    return required_capabilities(
        knowledge=str(getattr(request, "knowledge", "none")),
        source=str(getattr(request, "input", "")),
        stage=str(getattr(request, "stage", "raw-srt")),
    )


class DesktopResourceService:
    def __init__(
        self,
        *,
        bootstrap,
        runtime: RuntimeEnvironment,
        system_tool_finders: Mapping[str, Callable[[], SystemTool | None]]
        | None = None,
    ) -> None:
        self.bootstrap = bootstrap
        self.runtime = runtime
        # Injectable, because otherwise every answer this class gives depends on
        # what happens to be installed on the machine running it -- including in
        # tests and CI.
        self.system_tool_finders = (
            SYSTEM_TOOL_FINDERS if system_tool_finders is None else system_tool_finders
        )
        self._system_tools: dict[str, SystemTool | None] = {}

    def system_tool(self, resource_id: str) -> SystemTool | None:
        """A usable system copy of `resource_id`, probed at most once."""

        finder = self.system_tool_finders.get(resource_id)
        if finder is None:
            return None
        if resource_id not in self._system_tools:
            self._system_tools[resource_id] = finder()
        return self._system_tools[resource_id]

    def check_all(self) -> list[ResourceStatus]:
        """Every resource the user can act on, required ones first.

        The on-demand tools are included so that a task refused for a missing
        git has somewhere to send the user -- but flagged optional, so they do
        not make a perfectly usable install look half-finished.
        """

        required = [self.status(resource_id) for resource_id in ALWAYS_REQUIRED]
        optional = [
            self.status(resource_id).model_copy(update={"optional": True})
            for resource_id in BOOTSTRAP_RESOURCES
            if resource_id not in ALWAYS_REQUIRED
        ]
        return required + optional

    def status(self, resource_id: str) -> ResourceStatus:
        if resource_id == "uv":
            return self.runtime.status()
        if resource_id not in BOOTSTRAP_RESOURCES:
            raise KeyError(f"Unknown desktop resource: {resource_id}")
        found = self.system_tool(resource_id)
        if found is not None:
            return ResourceStatus(
                id=resource_id,
                version=found.version,
                state="ready",
                detail=f"使用系统已安装的版本：{found.path}",
            )
        return self.bootstrap.status(resource_id)

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
        if resource_id not in BOOTSTRAP_RESOURCES:
            raise KeyError(f"Unknown desktop resource: {resource_id}")
        if self.system_tool(resource_id) is not None:
            return self.status(resource_id)
        if not background:
            return self.bootstrap.install(resource_id, progress)
        return self.bootstrap.install(
            resource_id,
            progress,
            stage=stage,
            should_pause=should_pause,
        )

    def ensure(self, resource_ids: tuple[str, ...]) -> list[str]:
        """Names of the given resources that are still missing."""

        return [
            resource_id
            for resource_id in resource_ids
            if self.status(resource_id).state != "ready"
        ]

    def locations(self, resource_id: str) -> tuple[Path, Path]:
        if resource_id == "uv":
            return (
                self.bootstrap.cache_path("uv"),
                self.runtime.runtime_root,
            )
        if resource_id not in BOOTSTRAP_RESOURCES:
            raise KeyError(f"Unknown desktop resource: {resource_id}")
        return (
            self.bootstrap.cache_path(resource_id),
            self.bootstrap.install_path(resource_id),
        )

    def task_ready(self, request=None) -> bool:
        required = ALWAYS_REQUIRED + (
            capability_requirements(request) if request is not None else ()
        )
        return not self.ensure(required)

    def tool_directory(self, resource_id: str, filename: str) -> Path | None:
        """Directory to put on PATH for `resource_id`, system copy first."""

        found = self.system_tool(resource_id)
        if found is not None:
            return found.directory
        active = self.bootstrap.active_file(resource_id, filename)
        return active.parent if active is not None else None

    def worker_context(
        self,
        extra_env: Mapping[str, str],
    ) -> WorkerContext:
        # Same overrides the CLI applies: both launch the same pipeline against
        # the same user-data tree. Settings (API keys) go last so an explicitly
        # configured value always wins.
        environment = {
            **shared_environment_overrides(self.runtime.paths),
            **extra_env,
        }
        git_bin = self.tool_directory("git", "git.exe")
        # yt-dlp is imported, not executed, so it joins PYTHONPATH rather than
        # PATH -- and only when it is actually installed, since a URL task is
        # what pulls it down.
        yt_dlp = (
            [self.bootstrap.install_path("yt-dlp")]
            if self.bootstrap.active_version("yt-dlp") is not None
            else []
        )
        return self.runtime.worker_context(
            ffmpeg_bin=self.tool_directory("ffmpeg", "ffmpeg.exe"),
            extra_env=environment,
            extra_path_dirs=[git_bin] if git_bin is not None else [],
            extra_python_path=yt_dlp,
        )
