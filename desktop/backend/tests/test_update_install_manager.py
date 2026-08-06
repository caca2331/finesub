from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from finesub_bootstrap.models import DownloadProgress
from desktop.backend.updates.install_manager import (
    UpdateInstallInProgress,
    UpdateInstallManager,
)


class FakeUpdates:
    """Stand-in for GitHubUpdateService.install, driven by the test."""

    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        self.result = result or {"kind": "app", "version": "0.3.2", "restartRequired": True}
        self.error = error
        self.calls: list[tuple[str, ...]] = []
        self.release = threading.Event()
        self.entered = threading.Event()

    def install(self, kind, progress, stage):
        self.calls.append((kind,))
        self.entered.set()
        self.release.wait(timeout=5)
        stage("downloading", "正在下载更新")
        progress(DownloadProgress(downloaded=50, total=100, bytes_per_second=25.0))
        progress(DownloadProgress(downloaded=100, total=100, bytes_per_second=25.0))
        stage("installing", "正在校验并安装更新")
        if self.error is not None:
            raise self.error
        return self.result


def _settled(manager: UpdateInstallManager, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = manager.get()
        if snapshot is not None and snapshot.state in {"ready", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("update install did not settle")


def test_start_returns_immediately_and_reports_progress_as_it_arrives() -> None:
    # The bridge runs on the thread that draws the window, so start() must not
    # block for the length of a multi-hundred-megabyte download.
    updates = FakeUpdates()
    manager = UpdateInstallManager(updates)

    queued = manager.start("app", "0.3.2")

    assert queued.state == "queued"
    assert queued.version == "0.3.2"
    updates.entered.wait(timeout=5)
    updates.release.set()

    settled = _settled(manager)
    assert settled.state == "ready"
    assert settled.phase == "complete"
    assert settled.downloaded == 100
    assert settled.total == 100
    assert settled.restart_required is True
    assert settled.exit_required is False
    assert settled.bytes_per_second == 0


def test_a_full_update_asks_the_user_to_exit_rather_than_restart() -> None:
    updates = FakeUpdates(
        result={"kind": "full", "version": "0.4.0", "exitRequired": True}
    )
    manager = UpdateInstallManager(updates)
    manager.start("full", "0.4.0")
    updates.entered.wait(timeout=5)
    updates.release.set()

    settled = _settled(manager)

    assert settled.exit_required is True
    assert settled.restart_required is False


def test_starting_the_same_install_twice_does_not_download_twice() -> None:
    updates = FakeUpdates()
    manager = UpdateInstallManager(updates)

    manager.start("app", "0.3.2")
    updates.entered.wait(timeout=5)
    again = manager.start("app", "0.3.2")

    assert again.state in {"queued", "running"}
    updates.release.set()
    _settled(manager)
    assert updates.calls == [("app",)]


def test_a_second_different_update_is_refused_while_one_is_running() -> None:
    updates = FakeUpdates()
    manager = UpdateInstallManager(updates)
    manager.start("app", "0.3.2")
    updates.entered.wait(timeout=5)

    with pytest.raises(UpdateInstallInProgress):
        manager.start("full", "0.4.0")

    updates.release.set()
    _settled(manager)


def test_a_finished_install_is_terminal() -> None:
    # "app" staged a pending version that only a restart activates; re-running
    # would fight the install already on disk.
    updates = FakeUpdates()
    manager = UpdateInstallManager(updates)
    manager.start("app", "0.3.2")
    updates.entered.wait(timeout=5)
    updates.release.set()
    _settled(manager)

    again = manager.start("app", "0.3.2")

    assert again.state == "ready"
    assert updates.calls == [("app",)]


def test_a_failed_install_surfaces_the_reason_and_can_be_retried() -> None:
    updates = FakeUpdates(error=ValueError("signature did not verify"))
    manager = UpdateInstallManager(updates)
    manager.start("app", "0.3.2")
    updates.entered.wait(timeout=5)
    updates.release.set()

    settled = _settled(manager)

    assert settled.state == "failed"
    assert "signature did not verify" in settled.error
    assert settled.bytes_per_second == 0

    updates.release.clear()
    updates.entered.clear()
    retried = manager.start("app", "0.3.2")
    assert retried.state == "queued"
    updates.entered.wait(timeout=5)
    updates.release.set()
    _settled(manager)
    assert len(updates.calls) == 2


def test_get_is_none_before_anything_starts() -> None:
    assert UpdateInstallManager(FakeUpdates()).get() is None
