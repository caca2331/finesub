from __future__ import annotations

from pathlib import Path
import time

from desktop.backend.common.models import DownloadProgress, ResourceStatus
from desktop.backend.resources.downloader import DownloadPaused
from desktop.backend.resources.install_manager import ResourceInstallManager


class FakeResources:
    def __init__(self, root: Path) -> None:
        self.root = root

    def status(self, resource_id: str) -> ResourceStatus:
        return ResourceStatus(id=resource_id, version="1.0", state="missing")

    def locations(self, resource_id: str) -> tuple[Path, Path]:
        return (
            self.root / "cache" / f"{resource_id}.zip",
            self.root / "runtime" / resource_id,
        )

    def install(self, resource_id, progress, *, stage, log, should_pause):
        stage("downloading", "正在下载资源文件")
        for step in range(1, 21):
            if should_pause():
                raise DownloadPaused()
            progress(
                DownloadProgress(
                    downloaded=step * 5,
                    total=100,
                    bytes_per_second=50,
                )
            )
            time.sleep(0.01)
        return ResourceStatus(id=resource_id, version="1.0", state="ready")


def _wait_for(manager: ResourceInstallManager, resource_id: str, state: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = manager.get(resource_id)
        if snapshot is not None and snapshot.state == state:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"resource did not reach state {state}")


def _wait_for_progress(manager: ResourceInstallManager, resource_id: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snapshot = manager.get(resource_id)
        if snapshot is not None and snapshot.total > 0:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("resource did not report download progress")


def test_resource_install_runs_in_background_and_reports_progress(
    tmp_path: Path,
) -> None:
    manager = ResourceInstallManager(FakeResources(tmp_path))

    started = manager.start("ffmpeg")

    assert started.state in {"queued", "running"}
    running = _wait_for_progress(manager, "ffmpeg")
    assert running.total == 100
    assert running.downloaded > 0
    completed = _wait_for(manager, "ffmpeg", "ready")
    assert completed.phase == "complete"


def test_resource_install_pause_preserves_paths_and_can_resume(
    tmp_path: Path,
) -> None:
    manager = ResourceInstallManager(FakeResources(tmp_path))
    manager.start("uv")
    _wait_for_progress(manager, "uv")

    manager.pause("uv")

    paused = _wait_for(manager, "uv", "paused")
    assert paused.downloaded < paused.total
    assert paused.cache_path.endswith("uv.zip")
    resumed = manager.start("uv")
    assert resumed.state in {"queued", "running"}
    assert _wait_for(manager, "uv", "ready").state == "ready"
