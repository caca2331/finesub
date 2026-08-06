from __future__ import annotations

from typing import Any, Literal
import threading
import time

from finesub_bootstrap.models import DownloadProgress

from desktop.backend.common.models import UpdateInstallSnapshot


class UpdateInstallInProgress(ValueError):
    pass


class UpdateInstallManager:
    """Run a signed update install off the UI thread and report progress.

    The bridge call has to return immediately: a full package is hundreds of
    megabytes, and pywebview dispatches bridge methods on the thread that draws
    the window. So `start` spawns a worker and the frontend polls `get`, the
    same shape ResourceInstallManager uses for runtime downloads.
    """

    def __init__(self, updates: Any) -> None:
        self.updates = updates
        self._lock = threading.RLock()
        self._snapshot: UpdateInstallSnapshot | None = None

    def start(
        self,
        kind: Literal["app", "full"],
        version: str,
    ) -> UpdateInstallSnapshot:
        with self._lock:
            current = self._snapshot
            if current is not None and current.state in {"queued", "running"}:
                if current.version != version or current.kind != kind:
                    raise UpdateInstallInProgress(
                        f"正在安装 {current.version} 更新，请等待其完成。"
                    )
                return current.model_copy(deep=True)
            # A finished install is terminal: "app" left a pending version that
            # only a restart activates, and "full" handed off to an external
            # updater. Re-running either would fight the install already staged.
            if current is not None and current.state == "ready":
                return current.model_copy(deep=True)

            now = time.time()
            snapshot = UpdateInstallSnapshot(
                version=version,
                kind=kind,
                state="queued",
                message="正在准备下载",
                started_at=now,
                updated_at=now,
            )
            self._snapshot = snapshot
            worker = threading.Thread(
                target=self._run,
                args=(kind,),
                name="finesub-update-install",
                daemon=True,
            )
            worker.start()
            return snapshot.model_copy(deep=True)

    def get(self) -> UpdateInstallSnapshot | None:
        with self._lock:
            snapshot = self._snapshot
            return snapshot.model_copy(deep=True) if snapshot is not None else None

    def _run(self, kind: Literal["app", "full"]) -> None:
        def progress(value: DownloadProgress) -> None:
            self._update(
                state="running",
                phase="downloading",
                message=(
                    "下载完成，正在校验"
                    if value.total and value.downloaded >= value.total
                    else "正在下载更新"
                ),
                downloaded=value.downloaded,
                total=value.total,
                bytes_per_second=value.bytes_per_second,
            )

        def stage(phase: str, message: str) -> None:
            self._update(state="running", phase=phase, message=message)

        self._update(state="running", phase="waiting", message="正在连接下载源")
        try:
            result = self.updates.install(kind, progress, stage)
        except Exception as error:
            self._update(
                state="failed",
                message="更新安装失败",
                error=str(error) or type(error).__name__,
                bytes_per_second=0,
            )
            return
        restart = bool(result.get("restartRequired"))
        exit_required = bool(result.get("exitRequired"))
        self._update(
            state="ready",
            phase="complete",
            message=(
                "更新已就绪，重启 FineSub 后生效"
                if restart
                else "安装程序已启动，请退出 FineSub 以完成更新"
            ),
            version=str(result.get("version") or ""),
            restart_required=restart,
            exit_required=exit_required,
            bytes_per_second=0,
            error="",
        )

    def _update(self, **fields: Any) -> None:
        with self._lock:
            snapshot = self._snapshot
            if snapshot is None:
                return
            for name, value in fields.items():
                if name == "version" and not value:
                    continue
                setattr(snapshot, name, value)
            snapshot.updated_at = time.time()
