from __future__ import annotations

import os
from pathlib import Path

from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.models import ResourceStatus
from desktop.backend.common.models import TaskRequest
from desktop.backend.resources.desktop_service import (
    DesktopResourceService,
    capability_requirements,
)
from finesub_bootstrap.system_tools import SystemTool
from finesub_bootstrap.environment import WorkerContext


class FakeBootstrap:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.installed: list[str] = []
        self.states = {
            resource_id: ResourceStatus(
                id=resource_id, version="1.0", state="missing"
            )
            for resource_id in ("uv", "ffmpeg", "git", "yt-dlp")
        }

    def status(self, resource_id: str) -> ResourceStatus:
        return self.states[resource_id]

    def active_version(self, resource_id: str) -> str | None:
        state = self.states[resource_id]
        return state.version if state.state == "ready" else None

    def install_path(self, resource_id: str) -> Path:
        return self.root / resource_id

    def cache_path(self, resource_id: str) -> Path:
        return self.root / "cache" / resource_id

    def install(self, resource_id: str, progress) -> ResourceStatus:
        self.installed.append(resource_id)
        self.states[resource_id] = self.states[resource_id].model_copy(
            update={"state": "ready"}
        )
        return self.states[resource_id]

    # Mirrors runtime-manifest.json: the required file's path inside the
    # archive is what decides which directory ends up on PATH, so flattening it
    # here would let a wrong parent computation pass.
    LAYOUT = {
        "ffmpeg": Path("bin/ffmpeg.exe"),
        "git": Path("cmd/git.exe"),
        "yt-dlp": Path("yt_dlp/__init__.py"),
    }

    def active_file(self, resource_id: str, filename: str) -> Path | None:
        if self.states[resource_id].state != "ready":
            return None
        relative = self.LAYOUT.get(resource_id, Path(filename))
        if relative.name.casefold() != filename.casefold():
            return None
        target = self.root / resource_id / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"binary")
        return target


class FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = AppPaths.for_root(root)
        self.ready = False
        self.installs = 0

    def status(self) -> ResourceStatus:
        return ResourceStatus(
            id="uv",
            version="Python 3.12",
            state="ready" if self.ready else "missing",
        )

    def install(self) -> ResourceStatus:
        self.installs += 1
        self.ready = True
        return self.status()

    def worker_context(
        self,
        *,
        ffmpeg_bin,
        extra_env,
        extra_path_dirs=(),
        extra_python_path=(),
    ) -> WorkerContext:
        return WorkerContext(
            python_executable=self.root / "python.exe",
            working_directory=self.root / "app",
            environment={
                **extra_env,
                "FFMPEG_BIN": str(ffmpeg_bin) if ffmpeg_bin else "",
                "PATH_DIRS": os.pathsep.join(str(p) for p in extra_path_dirs),
                "PYTHONPATH_EXTRA": os.pathsep.join(
                    str(p) for p in extra_python_path
                ),
            },
        )


def test_python_install_bootstraps_uv_before_activating_runtime(
    tmp_path: Path,
) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(
        bootstrap=bootstrap, runtime=runtime, system_tool_finders={}
    )

    result = service.install("uv", lambda event: None)

    assert result.state == "ready"
    assert bootstrap.installed == ["uv"]
    assert runtime.installs == 1


def test_task_requires_both_python_and_ffmpeg(tmp_path: Path) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(
        bootstrap=bootstrap, runtime=runtime, system_tool_finders={}
    )

    assert service.task_ready() is False
    service.install("uv", lambda event: None)
    assert service.task_ready() is False
    service.install("ffmpeg", lambda event: None)
    assert service.task_ready() is True


def test_worker_context_uses_active_ffmpeg_and_user_settings(
    tmp_path: Path,
) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(
        bootstrap=bootstrap, runtime=runtime, system_tool_finders={}
    )
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    context = service.worker_context({"GEMINI_FREE": "user-key"})

    assert context.environment["GEMINI_FREE"] == "user-key"
    assert context.environment["FFMPEG_BIN"] == str(tmp_path / "ffmpeg" / "bin")


def test_worker_context_pins_the_knowledge_base_to_user_data(
    tmp_path: Path, monkeypatch
) -> None:
    # Left to resolve itself, the knowledge root walks up from the worker's
    # source directory and lands in app/versions/<version>/knowledge -- which
    # the next app update replaces, silently taking the knowledge base with it.
    # The CLI already pins it; the desktop has to agree, or the two disagree
    # about where the same user's knowledge lives.
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
    )

    context = service.worker_context({})

    assert context.environment["FINESUB_KNOWLEDGE_ROOT"] == str(
        AppPaths.for_root(tmp_path).user_data / "knowledge"
    )


def test_explicit_knowledge_root_beats_the_launcher_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_KNOWLEDGE_ROOT", "explicit")
    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
    )

    assert "FINESUB_KNOWLEDGE_ROOT" not in service.worker_context({}).environment


def _service(tmp_path: Path, **finders):
    return DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders=finders,
    )


def test_a_capable_system_ffmpeg_makes_the_download_unnecessary(
    tmp_path: Path,
) -> None:
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    service = _service(tmp_path, ffmpeg=lambda: found)

    status = service.status("ffmpeg")
    service.install("ffmpeg", lambda event: None)

    assert status.state == "ready"
    assert "C:/tools/ffmpeg.exe" in status.detail.replace("\\", "/")
    # install() must be a no-op, not a 146MB second copy.
    assert service.bootstrap.installed == []


def test_the_system_probe_runs_once_per_resource(tmp_path: Path) -> None:
    calls: list[str] = []

    def finder():
        calls.append("ffmpeg")
        return None

    service = _service(tmp_path, ffmpeg=finder)
    for _ in range(4):
        service.status("ffmpeg")

    assert calls == ["ffmpeg"]


def test_git_and_yt_dlp_are_not_required_to_start_an_ordinary_task(
    tmp_path: Path,
) -> None:
    # Requiring them up front would make every user download 42MB for
    # capabilities most never use.
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    assert service.task_ready() is True
    # Listed, so they are reachable -- but flagged optional, so a usable install
    # does not read as half-finished.
    required = [s.id for s in service.check_all() if not s.optional]
    optional = [s.id for s in service.check_all() if s.optional]
    assert required == ["uv", "ffmpeg"]
    assert optional == ["git", "yt-dlp"]


def test_a_knowledge_update_task_needs_git(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)
    request = TaskRequest(input="a.wav", knowledge="update", stage="final-srt")

    assert capability_requirements(request) == ("git",)
    assert service.task_ready(request) is False

    service.install("git", lambda event: None)
    assert service.task_ready(request) is True


def test_a_url_task_needs_yt_dlp(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)
    request = TaskRequest(input="https://example.test/watch?v=1")

    assert capability_requirements(request) == ("yt-dlp",)
    assert service.task_ready(request) is False

    service.install("yt-dlp", lambda event: None)
    assert service.task_ready(request) is True


def test_the_default_settings_do_not_require_git(tmp_path: Path) -> None:
    # knowledge defaults to "update" and stage to raw-srt. The update only runs
    # in the correction stage, so the default task must not demand a download
    # that nothing will use.
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    assert capability_requirements(TaskRequest(input="a.wav")) == ()
    assert service.task_ready(TaskRequest(input="a.wav")) is True


def test_a_system_git_satisfies_the_knowledge_requirement(tmp_path: Path) -> None:
    found = SystemTool(path=Path("C:/Program Files/Git/cmd/git.exe"), version="2.44")
    service = _service(tmp_path, git=lambda: found)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    assert (
        service.task_ready(
            TaskRequest(input="a.wav", knowledge="update", stage="final-srt")
        )
        is True
    )
    assert service.bootstrap.installed == ["uv", "ffmpeg"]


def test_git_goes_on_path_and_yt_dlp_goes_on_pythonpath(tmp_path: Path) -> None:
    # Different injection because they are found differently: git is executed,
    # yt-dlp is imported.
    service = _service(tmp_path)
    service.install("git", lambda event: None)
    service.install("yt-dlp", lambda event: None)

    environment = service.worker_context({}).environment

    assert environment["PATH_DIRS"] == str(tmp_path / "git" / "cmd")
    assert environment["PYTHONPATH_EXTRA"] == str(tmp_path / "yt-dlp")


def test_an_uninstalled_yt_dlp_adds_nothing_to_pythonpath(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.worker_context({}).environment["PYTHONPATH_EXTRA"] == ""
