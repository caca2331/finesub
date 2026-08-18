from __future__ import annotations

import os
from pathlib import Path
import time

from finesub_bootstrap.models import ResourceStatus

from desktop.backend.resources import install_log
from desktop.backend.resources.install_manager import ResourceInstallManager


class FakeResources:
    """Resource service that logs a few lines and then either finishes or fails."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def status(self, resource_id: str) -> ResourceStatus:
        return ResourceStatus(id=resource_id, version="1.0", state="missing")

    def locations(self, resource_id: str) -> tuple[Path, Path]:
        return Path("cache"), Path("install")

    def install(self, resource_id, progress, *, stage, log, should_pause):
        for index in range(3):
            log(f"line {index}")
        if self.fail:
            raise RuntimeError("download died")


def _wait_for(manager: ResourceInstallManager, resource_id: str, state: str):
    for _ in range(500):
        snapshot = manager.get(resource_id)
        if snapshot is not None and snapshot.state == state:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"{resource_id} never reached {state}")


def test_the_whole_install_is_written_to_a_file(tmp_path: Path) -> None:
    # The snapshot keeps a hundred lines and dies with the window; an install
    # that failed after twenty minutes is exactly the one someone needs to read
    # afterwards.
    log = install_log.InstallLog.open(tmp_path, "uv")
    for index in range(150):
        log.write(f"line {index}")
    log.finish("安装完成")

    assert log.path is not None
    body = log.path.read_text(encoding="utf-8").splitlines()
    assert len(body) == 151
    assert body[0] == "line 0"
    assert body[-1] == "--- 安装完成 ---"


def test_every_ending_is_recorded_including_the_bad_ones(tmp_path: Path) -> None:
    for outcome in ("安装完成", "失败：download died", "已暂停"):
        log = install_log.InstallLog.open(tmp_path, "uv")
        log.write("some output")
        log.finish(outcome)
        assert log.path is not None
        assert log.path.read_text(encoding="utf-8").endswith(f"--- {outcome} ---\n")


def test_each_line_reaches_the_disk_before_the_next_one(tmp_path: Path) -> None:
    # An install that is killed mid-way must still leave everything it printed.
    log = install_log.InstallLog.open(tmp_path, "uv")
    log.write("first")

    assert log.path is not None
    assert log.path.read_text(encoding="utf-8") == "first\n"


def test_old_logs_are_pruned_to_the_last_hundred(tmp_path: Path) -> None:
    for index in range(120):
        path = tmp_path / f"install-uv-2026080{index:04d}.log"
        path.write_text("x", encoding="utf-8")
        os.utime(path, (index, index))
    (tmp_path / "not-a-log.txt").write_text("x", encoding="utf-8")

    install_log.prune(tmp_path, keep=100)

    remaining = sorted(path.name for path in tmp_path.glob("install-*.log"))
    assert len(remaining) == 100
    # Newest kept, oldest dropped.
    assert remaining[-1] == "install-uv-20260800119.log"
    assert (tmp_path / "not-a-log.txt").exists()


def test_pruning_keeps_the_newest_across_resources(tmp_path: Path) -> None:
    # The file name is install-<resource>-<stamp>, so ordering by name sorts by
    # resource id first: "the newest hundred" would have meant "whatever
    # belongs to the resource that sorts last".
    old_but_late_alphabetically = tmp_path / "install-uv-20260101-000000.log"
    new_but_early_alphabetically = tmp_path / "install-ffmpeg-20260807-000000.log"
    for path, when in (
        (old_but_late_alphabetically, 1_000),
        (new_but_early_alphabetically, 2_000),
    ):
        path.write_text("x", encoding="utf-8")
        os.utime(path, (when, when))

    install_log.prune(tmp_path, keep=1)

    assert new_but_early_alphabetically.exists()
    assert not old_but_late_alphabetically.exists()


def test_a_directory_that_cannot_be_written_does_not_break_the_install(
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory", encoding="utf-8")

    log = install_log.InstallLog.open(blocked, "uv")
    log.write("still fine")
    log.finish("安装完成")

    assert log.path is None


def test_a_finished_install_leaves_its_transcript_behind(tmp_path: Path) -> None:
    manager = ResourceInstallManager(FakeResources(), log_dir=tmp_path / "logs")

    manager.start("uv")
    _wait_for(manager, "uv", "ready")

    logs = list((tmp_path / "logs").glob("install-uv-*.log"))
    assert len(logs) == 1
    body = logs[0].read_text(encoding="utf-8")
    assert "line 0" in body and "line 2" in body
    assert body.strip().endswith("--- 安装完成 ---")
    assert manager.get("uv").log_path == str(logs[0])


def test_a_failed_install_says_so_in_its_transcript(tmp_path: Path) -> None:
    manager = ResourceInstallManager(
        FakeResources(fail=True), log_dir=tmp_path / "logs"
    )

    manager.start("uv")
    _wait_for(manager, "uv", "failed")

    body = next((tmp_path / "logs").glob("install-uv-*.log")).read_text("utf-8")
    assert "download died" in body
