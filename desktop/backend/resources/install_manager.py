from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
import threading
import time

from finesub_bootstrap.downloader import DownloadPaused
from finesub_bootstrap.models import DownloadProgress

from desktop.backend.common.models import ResourceInstallSnapshot


class ResourceInstallConflict(ValueError):
    pass


class ResourceInstallNotFound(KeyError):
    pass


class ResourceInstallManager:
    def __init__(
        self,
        resources,
        *,
        on_ready: Callable[[], None] | None = None,
        log_limit: int = 100,
    ) -> None:
        self.resources = resources
        self.on_ready = on_ready
        self.log_limit = log_limit
        self._lock = threading.RLock()
        self._snapshots: dict[str, ResourceInstallSnapshot] = {}
        self._pause_events: dict[str, threading.Event] = {}

    def start(self, resource_id: str) -> ResourceInstallSnapshot:
        with self._lock:
            active = next(
                (
                    snapshot
                    for snapshot in self._snapshots.values()
                    if snapshot.state in {"queued", "running"}
                    and snapshot.resource_id != resource_id
                ),
                None,
            )
            if active is not None:
                raise ResourceInstallConflict(
                    f"{active.resource_id} 正在处理，请先暂停或等待完成。"
                )
            current = self._snapshots.get(resource_id)
            if current is not None and current.state in {"queued", "running"}:
                return current.model_copy(deep=True)

            status = self.resources.status(resource_id)
            cache_path, install_path = self.resources.locations(resource_id)
            now = time.time()
            if status.state == "ready":
                snapshot = ResourceInstallSnapshot(
                    resource_id=resource_id,
                    resource_version=status.version,
                    state="ready",
                    phase="complete",
                    message="资源已经安装",
                    cache_path=str(cache_path),
                    install_path=str(install_path),
                    started_at=now,
                    updated_at=now,
                )
                self._snapshots[resource_id] = snapshot
                return snapshot.model_copy(deep=True)

            snapshot = ResourceInstallSnapshot(
                resource_id=resource_id,
                resource_version=status.version,
                state="queued",
                message="正在准备后台任务",
                cache_path=str(cache_path),
                install_path=str(install_path),
                started_at=now,
                updated_at=now,
            )
            pause_event = threading.Event()
            self._snapshots[resource_id] = snapshot
            self._pause_events[resource_id] = pause_event
            worker = threading.Thread(
                target=self._run,
                args=(resource_id, pause_event),
                name=f"finesub-resource-{resource_id}",
                daemon=True,
            )
            worker.start()
            return snapshot.model_copy(deep=True)

    def pause(self, resource_id: str) -> ResourceInstallSnapshot:
        with self._lock:
            snapshot = self._require(resource_id)
            if snapshot.state not in {"queued", "running"}:
                return snapshot.model_copy(deep=True)
            self._pause_events[resource_id].set()
            snapshot.message = "正在安全暂停"
            snapshot.updated_at = time.time()
            return snapshot.model_copy(deep=True)

    def get(self, resource_id: str) -> ResourceInstallSnapshot | None:
        with self._lock:
            snapshot = self._snapshots.get(resource_id)
            return snapshot.model_copy(deep=True) if snapshot is not None else None

    def list(self) -> list[ResourceInstallSnapshot]:
        with self._lock:
            return [
                snapshot.model_copy(deep=True)
                for snapshot in self._snapshots.values()
            ]

    def location(self, resource_id: str, kind: str) -> Path:
        if kind not in {"cache", "install"}:
            raise ValueError("资源目录类型只能是 cache 或 install")
        cache_path, install_path = self.resources.locations(resource_id)
        path = cache_path.parent if kind == "cache" else install_path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _run(self, resource_id: str, pause_event: threading.Event) -> None:
        self._update(
            resource_id,
            state="running",
            phase="waiting",
            message="正在连接下载源",
            error="",
        )

        def progress(value: DownloadProgress) -> None:
            message = (
                "下载完成，正在校验"
                if value.total and value.downloaded >= value.total
                else "正在下载资源"
            )
            self._update(
                resource_id,
                state="running",
                phase="downloading",
                message=message,
                downloaded=value.downloaded,
                total=value.total,
                bytes_per_second=value.bytes_per_second,
            )

        def stage(phase: str, message: str) -> None:
            self._update(
                resource_id,
                state="running",
                phase=phase,
                message=message,
            )

        def log(line: str) -> None:
            safe_line = self._sanitize_log(line)
            with self._lock:
                snapshot = self._require(resource_id)
                snapshot.logs = [*snapshot.logs, safe_line][-self.log_limit :]
                snapshot.updated_at = time.time()

        try:
            self.resources.install(
                resource_id,
                progress,
                stage=stage,
                log=log,
                should_pause=pause_event.is_set,
            )
        except DownloadPaused:
            self._update(
                resource_id,
                state="paused",
                message="已暂停，已下载内容会保留",
                bytes_per_second=0,
            )
        except Exception as error:
            self._update(
                resource_id,
                state="failed",
                message="资源安装失败",
                error=str(error) or type(error).__name__,
                bytes_per_second=0,
            )
        else:
            self._update(
                resource_id,
                state="ready",
                phase="complete",
                message="安装完成",
                bytes_per_second=0,
                error="",
            )
            if self.on_ready is not None:
                self.on_ready()

    def _update(self, resource_id: str, **changes: object) -> None:
        with self._lock:
            snapshot = self._require(resource_id)
            for key, value in changes.items():
                setattr(snapshot, key, value)
            snapshot.updated_at = time.time()

    def _require(self, resource_id: str) -> ResourceInstallSnapshot:
        try:
            return self._snapshots[resource_id]
        except KeyError as error:
            raise ResourceInstallNotFound(resource_id) from error

    @staticmethod
    def _sanitize_log(line: str) -> str:
        return re.sub(
            r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD))=\S+",
            r"\1=<redacted>",
            line,
        )
