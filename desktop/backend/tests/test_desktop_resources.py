from __future__ import annotations

from pathlib import Path

from desktop.backend.common.models import ResourceStatus
from desktop.backend.resources.desktop_service import DesktopResourceService
from desktop.backend.runtime.environment import WorkerContext


class FakeBootstrap:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.installed: list[str] = []
        self.states = {
            "uv": ResourceStatus(id="uv", version="0.11", state="missing"),
            "ffmpeg": ResourceStatus(
                id="ffmpeg",
                version="7.1",
                state="missing",
            ),
        }

    def status(self, resource_id: str) -> ResourceStatus:
        return self.states[resource_id]

    def install(self, resource_id: str, progress) -> ResourceStatus:
        self.installed.append(resource_id)
        self.states[resource_id] = self.states[resource_id].model_copy(
            update={"state": "ready"}
        )
        return self.states[resource_id]

    def active_file(self, resource_id: str, filename: str) -> Path | None:
        if self.states[resource_id].state != "ready":
            return None
        target = self.root / resource_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"binary")
        return target


class FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
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

    def worker_context(self, *, ffmpeg_bin, extra_env) -> WorkerContext:
        return WorkerContext(
            python_executable=self.root / "python.exe",
            working_directory=self.root / "app",
            environment={
                **extra_env,
                "FFMPEG_BIN": str(ffmpeg_bin) if ffmpeg_bin else "",
            },
        )


def test_python_install_bootstraps_uv_before_activating_runtime(
    tmp_path: Path,
) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(bootstrap=bootstrap, runtime=runtime)

    result = service.install("uv", lambda event: None)

    assert result.state == "ready"
    assert bootstrap.installed == ["uv"]
    assert runtime.installs == 1


def test_task_requires_both_python_and_ffmpeg(tmp_path: Path) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(bootstrap=bootstrap, runtime=runtime)

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
    service = DesktopResourceService(bootstrap=bootstrap, runtime=runtime)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    context = service.worker_context({"GEMINI_API_KEY": "user-key"})

    assert context.environment["GEMINI_API_KEY"] == "user-key"
    assert context.environment["FFMPEG_BIN"] == str(tmp_path / "ffmpeg")
