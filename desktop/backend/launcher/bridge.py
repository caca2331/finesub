from __future__ import annotations

from collections.abc import Callable
import logging
import os
from pathlib import Path
import webbrowser
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from desktop.backend.common.models import BridgeError, TaskRequest
from desktop.backend.jobs.manager import JobAlreadyRunning, JobNotFound
from desktop.backend.settings.store import SettingsStore


LOGGER = logging.getLogger(__name__)


class ApiKeyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gemini: str | None = None
    exa: str | None = None
    tavily: str | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _success(value: Any = None) -> dict[str, Any]:
    return {"ok": True, "data": _json_safe(value)}


def _failure(error: BridgeError) -> dict[str, Any]:
    return {"ok": False, "error": error.model_dump(mode="json")}


RESOURCE_LABELS = {
    "uv": "Python 运行环境",
    "ffmpeg": "FFmpeg",
    "git": "git（知识库更新需要）",
    "yt-dlp": "yt-dlp（链接下载需要）",
}

def _missing_resource_error(missing: list[str], verb: str) -> BridgeError:
    names = "、".join(RESOURCE_LABELS.get(item, item) for item in missing)
    return BridgeError(
        code="runtime_required",
        message=f"请先安装 {names}，再{verb}字幕任务。",
        action="open_resources",
    )


class DesktopBridge:
    def __init__(
        self,
        *,
        jobs: Any,
        resources: Any,
        resource_installs: Any | None = None,
        settings: SettingsStore,
        updates: Any | None = None,
        update_installs: Any | None = None,
        file_selector: Callable[[], str | None] | None = None,
        output_opener: Callable[[Path], None] | None = None,
        url_opener: Callable[[str], Any] | None = None,
        window: Any | None = None,
        tray: Any | None = None,
        app_version: str = "development",
    ) -> None:
        self.jobs = jobs
        self.resources = resources
        self.resource_installs = resource_installs
        self.settings = settings
        self.updates = updates
        self.update_installs = update_installs
        self.file_selector = file_selector
        self.output_opener = output_opener or self._open_in_explorer
        self.url_opener = url_opener or webbrowser.open
        self.window = window
        self.tray = tray
        self.app_version = app_version

    def get_bootstrap_state(self) -> dict[str, Any]:
        return self._guard(
            lambda: {
                "app_version": self.app_version,
                "resources": self.resources.check_all(),
                "resource_installs": (
                    self.resource_installs.list()
                    if self.resource_installs is not None
                    else []
                ),
                "capabilities": self.settings.get_capabilities(),
                "settings": self.settings.public_settings(),
                "task": self.jobs.snapshot(),
                "tasks": self.jobs.history(),
            }
        )

    def select_input_file(self) -> dict[str, Any]:
        if self.file_selector is None:
            return _failure(
                BridgeError(
                    code="dialog_unavailable",
                    message="当前窗口无法打开文件选择器。",
                )
            )
        return self._guard(lambda: {"path": self.file_selector()})

    def start_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = TaskRequest.model_validate(payload)
        except ValidationError as error:
            return _failure(
                BridgeError(
                    code="invalid_request",
                    message="任务参数无效。",
                    action=str(error.errors(include_url=False)),
                )
            )
        capability_error = self.settings.validate_stage(request.stage)
        if capability_error is not None:
            return _failure(capability_error)
        missing = self._missing_resources(request)
        if missing:
            return _failure(_missing_resource_error(missing, "开始"))
        try:
            return _success(self.jobs.start(request))
        except JobAlreadyRunning:
            return _failure(
                BridgeError(
                    code="task_already_running",
                    message="已有字幕任务正在运行。",
                    action="show_current_task",
                )
            )
        except Exception:
            return self._internal_error("start_task")

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        try:
            return _success(self.jobs.cancel(task_id))
        except JobNotFound:
            return _failure(
                BridgeError(code="task_not_found", message="没有找到该任务。")
            )
        except Exception:
            return self._internal_error("cancel_task")

    def retry_task(self, task_id: str) -> dict[str, Any]:
        try:
            validation = self._validate_saved_task(task_id)
            if validation is not None:
                return _failure(validation)
            return _success(self.jobs.retry(task_id))
        except JobNotFound:
            return _failure(
                BridgeError(code="task_not_found", message="没有找到该任务。")
            )
        except JobAlreadyRunning:
            return _failure(
                BridgeError(
                    code="task_already_running",
                    message="已有字幕任务正在运行。",
                    action="show_current_task",
                )
            )
        except Exception:
            return self._internal_error("retry_task")

    def resume_task(self, task_id: str) -> dict[str, Any]:
        try:
            validation = self._validate_saved_task(task_id)
            if validation is not None:
                return _failure(validation)
            return _success(self.jobs.resume(task_id))
        except JobNotFound:
            return _failure(
                BridgeError(code="task_not_found", message="没有找到该任务。")
            )
        except JobAlreadyRunning:
            return _failure(
                BridgeError(
                    code="task_already_running",
                    message="已有字幕任务正在运行。",
                    action="show_current_task",
                )
            )
        except ValueError as error:
            return _failure(
                BridgeError(code="task_not_resumable", message=str(error))
            )
        except Exception:
            return self._internal_error("resume_task")

    def get_task_snapshot(self) -> dict[str, Any]:
        return self._guard(self.jobs.snapshot)

    def list_tasks(self) -> dict[str, Any]:
        return self._guard(self.jobs.history)

    def _validate_saved_task(self, task_id: str) -> BridgeError | None:
        request = self.jobs.request_for(task_id)
        capability_error = self.settings.validate_stage(request.stage)
        if capability_error is not None:
            return capability_error
        missing = self._missing_resources(request)
        if missing:
            return _missing_resource_error(missing, "继续")
        return None

    def _missing_resources(self, request) -> list[str]:
        """What this specific request still needs.

        Requirements depend on the request: git only for a knowledge update,
        yt-dlp only for a URL. A blanket check would make every user install
        both before their first plain transcription.
        """

        ensure = getattr(self.resources, "ensure", None)
        if not callable(ensure):
            return []
        from desktop.backend.resources.desktop_service import (
            ALWAYS_REQUIRED,
            capability_requirements,
        )

        return ensure(ALWAYS_REQUIRED + capability_requirements(request))

    def poll_events(self, after_cursor: int = 0) -> dict[str, Any]:
        def collect() -> dict[str, Any]:
            events, next_cursor = self.jobs.events_after(after_cursor)
            return {
                "events": events,
                "nextCursor": next_cursor,
            }

        return self._guard(collect)

    def install_resource(self, resource_id: str) -> dict[str, Any]:
        if self.resource_installs is None:
            def install_legacy() -> Any:
                result = self.resources.install(resource_id, lambda event: None)
                self._refresh_worker_environment()
                return result

            return self._guard(install_legacy)
        return self._guard(lambda: self.resource_installs.start(resource_id))

    def get_resource_install(self, resource_id: str) -> dict[str, Any]:
        if self.resource_installs is None:
            return _success(None)
        return self._guard(lambda: self.resource_installs.get(resource_id))

    def list_resource_installs(self) -> dict[str, Any]:
        if self.resource_installs is None:
            return _success([])
        return self._guard(self.resource_installs.list)

    def pause_resource_install(self, resource_id: str) -> dict[str, Any]:
        if self.resource_installs is None:
            return _failure(
                BridgeError(
                    code="resource_manager_unavailable",
                    message="当前构建不支持后台资源任务。",
                )
            )
        return self._guard(lambda: self.resource_installs.pause(resource_id))

    def open_resource_location(
        self,
        resource_id: str,
        kind: Literal["cache", "install"],
    ) -> dict[str, Any]:
        if self.resource_installs is None:
            return _failure(
                BridgeError(
                    code="resource_manager_unavailable",
                    message="当前构建不支持资源目录定位。",
                )
            )

        def open_location() -> dict[str, str]:
            path = self.resource_installs.location(resource_id, kind)
            self.output_opener(path)
            return {"path": str(path)}

        return self._guard(open_location)

    def save_api_keys(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            values = ApiKeyPayload.model_validate(payload)
            self.settings.save_api_keys(**values.model_dump())
            self._refresh_worker_environment()
            return _success(self.settings.public_settings())
        except (ValidationError, ValueError):
            return _failure(
                BridgeError(
                    code="invalid_api_keys",
                    message="API Key 配置无效。",
                    action="check_api_key_values",
                )
            )
        except Exception:
            return self._internal_error("save_api_keys")

    def delete_api_key(
        self,
        provider: Literal["gemini", "exa", "tavily"],
    ) -> dict[str, Any]:
        try:
            self.settings.delete_api_key(provider)
            self._refresh_worker_environment()
            return _success(self.settings.public_settings())
        except ValueError:
            return _failure(
                BridgeError(
                    code="invalid_provider",
                    message="未知的 API 服务商。",
                )
            )
        except Exception:
            return self._internal_error("delete_api_key")

    def check_updates(self) -> dict[str, Any]:
        if self.updates is None:
            return _failure(
                BridgeError(
                    code="updates_unavailable",
                    message="当前构建未配置更新检查。",
                )
            )
        return self._guard(self.updates.check)

    def install_update(self, kind: str, version: str) -> dict[str, Any]:
        if self.updates is None or self.update_installs is None:
            return _failure(
                BridgeError(
                    code="updates_unavailable",
                    message="当前构建未配置更新安装。",
                )
            )
        if kind not in {"app", "full"}:
            return _failure(
                BridgeError(
                    code="invalid_request",
                    message=f"未知的更新类型：{kind}",
                )
            )

        def start() -> Any:
            # The kind comes from what check_updates told the UI. install()
            # re-derives it from the signed manifest and rejects a mismatch, so
            # a stale page cannot talk the backend into the wrong payload.
            return self.update_installs.start(kind, version)

        return self._guard(start)

    def get_update_install(self) -> dict[str, Any]:
        if self.update_installs is None:
            return _success(None)
        return self._guard(self.update_installs.get)

    def open_update_page(self) -> dict[str, Any]:
        if self.updates is None:
            return _failure(
                BridgeError(
                    code="updates_unavailable",
                    message="当前构建未配置更新检查。",
                )
            )

        def open_page() -> dict[str, str]:
            url = self.updates.release_url()
            self.url_opener(url)
            return {"url": url}

        return self._guard(open_page)

    def open_tasks_directory(self, task_id: str = "") -> dict[str, Any]:
        """Reveal user-data/tasks, or one task's folder inside it.

        No path comes from the caller -- only a task id the manager resolves --
        so this cannot be talked into opening somewhere else.
        """

        def open_directory() -> dict[str, str]:
            target = self.jobs.task_directory(task_id)
            self.output_opener(target)
            return {"path": str(target)}

        return self._guard(open_directory)

    def open_output(self, output_path: str) -> dict[str, Any]:
        def open_path() -> dict[str, str]:
            snapshots = self.jobs.history()
            if not snapshots:
                raise ValueError("No task output is available")
            path = Path(output_path).expanduser().resolve()
            safe_outputs = {
                Path(value).expanduser().resolve()
                for snapshot in snapshots
                for value in _json_safe(snapshot).get("outputs", {}).values()
                if isinstance(value, str) and value
            }
            if path not in safe_outputs:
                raise ValueError("Output path is not owned by a saved task")
            self.output_opener(path)
            return {"path": str(path)}

        try:
            return _success(open_path())
        except ValueError:
            return _failure(
                BridgeError(
                    code="invalid_output",
                    message="无法打开不属于当前任务的路径。",
                )
            )
        except Exception:
            return self._internal_error("open_output")

    def minimize_window(self) -> dict[str, Any]:
        return self._window_action("minimize")

    def minimize_to_tray(self) -> dict[str, Any]:
        if self.tray is None:
            return _failure(
                BridgeError(
                    code="tray_unavailable",
                    message="系统托盘尚未初始化。",
                )
            )
        return self._guard(self.tray.hide_window)

    def maximize_window(self) -> dict[str, Any]:
        return self._guard(self._toggle_maximize)

    def close_window(self) -> dict[str, Any]:
        return self._window_action("destroy")

    def _window_action(self, method: str) -> dict[str, Any]:
        if self.window is None:
            return _failure(
                BridgeError(code="window_unavailable", message="窗口尚未初始化。")
            )
        return self._guard(lambda: getattr(self.window, method)())

    def _toggle_maximize(self) -> None:
        if self.window is None:
            raise ValueError("窗口尚未初始化。")
        native = getattr(self.window, "native", None)
        state = getattr(native, "WindowState", "")
        state_name = state.ToString() if hasattr(state, "ToString") else str(state)
        maximized = state_name.rsplit(".", 1)[-1].lower() == "maximized"
        if maximized and native is not None and hasattr(native, "Invoke"):
            import System.Windows.Forms as WinForms

            native.Invoke(
                WinForms.MethodInvoker(
                    lambda: setattr(native, "WindowState", WinForms.FormWindowState.Normal)
                )
            )
            return
        (self.window.restore if maximized else self.window.maximize)()

    def _refresh_worker_environment(self) -> None:
        environment = self.settings.build_worker_env()
        context_builder = getattr(self.resources, "worker_context", None)
        if callable(context_builder):
            context = context_builder(environment)
            self.jobs.python_executable = str(context.python_executable)
            self.jobs.working_directory = str(context.working_directory)
            self.jobs.worker_env = dict(context.environment)
            return
        self.jobs.worker_env = environment

    def _guard(self, action: Callable[[], Any]) -> dict[str, Any]:
        try:
            return _success(action())
        except (ValueError, KeyError) as error:
            return _failure(
                BridgeError(code="invalid_request", message=str(error))
            )
        except Exception:
            return self._internal_error(action.__name__)

    @staticmethod
    def _open_in_explorer(path: Path) -> None:
        target = path if path.is_dir() else path.parent
        if os.name != "nt":
            raise OSError("Explorer integration is only available on Windows")
        os.startfile(str(target))

    @staticmethod
    def _internal_error(operation: str) -> dict[str, Any]:
        LOGGER.exception("Desktop bridge operation failed: %s", operation)
        return _failure(
            BridgeError(
                code="internal_error",
                message="操作失败，请查看应用日志后重试。",
            )
        )
