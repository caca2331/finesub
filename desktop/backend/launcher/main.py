from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

from desktop.backend.common.models import ResourceSpec
from desktop.backend.common.paths import AppPaths
from desktop.backend.common.product import PRODUCT_NAME
from desktop.backend.jobs.manager import JobManager
from desktop.backend.launcher.bridge import DesktopBridge
from desktop.backend.launcher.tray import TrayController
from desktop.backend.resources.desktop_service import DesktopResourceService
from desktop.backend.resources.install_manager import ResourceInstallManager
from desktop.backend.resources.manager import ResourceManager
from desktop.backend.runtime.environment import RuntimeEnvironment
from desktop.backend.settings.store import SettingsStore
from desktop.backend.updates.installer import AppInstaller
from desktop.backend.updates.service import (
    GitHubUpdateService,
    LauncherUpdateConfig,
)


PUBLIC_BRIDGE_METHODS = (
    "get_bootstrap_state",
    "select_input_file",
    "start_task",
    "cancel_task",
    "retry_task",
    "resume_task",
    "get_task_snapshot",
    "list_tasks",
    "poll_events",
    "install_resource",
    "get_resource_install",
    "list_resource_installs",
    "pause_resource_install",
    "open_resource_location",
    "save_api_keys",
    "delete_api_key",
    "check_updates",
    "open_update_page",
    "open_output",
    "minimize_window",
    "minimize_to_tray",
    "maximize_window",
    "close_window",
)


def expose_bridge(window: Any, bridge: DesktopBridge) -> None:
    window.expose(
        *(getattr(bridge, method_name) for method_name in PUBLIC_BRIDGE_METHODS)
    )


def dropped_file_path(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    transfer = event.get("dataTransfer")
    if not isinstance(transfer, dict):
        return None
    files = transfer.get("files")
    if not isinstance(files, list) or not files:
        return None
    first = files[0]
    if not isinstance(first, dict):
        return None
    path = first.get("pywebviewFullPath")
    return path if isinstance(path, str) and path else None


def bind_native_file_drop(window: Any) -> None:
    from webview.dom import DOMEventHandler

    def ignore_drag(_event: Any) -> None:
        return None

    def dispatch_drop(event: Any) -> None:
        path = dropped_file_path(event)
        if path is None:
            return
        encoded = json.dumps({"path": path}, ensure_ascii=False)
        window.evaluate_js(
            "window.dispatchEvent(new CustomEvent("
            f"'finesub:file-drop', {{detail: {encoded}}}"
            "));"
        )

    events = window.dom.document.events
    events.dragenter += DOMEventHandler(ignore_drag, True, False)
    events.dragover += DOMEventHandler(ignore_drag, True, False)
    events.drop += DOMEventHandler(dispatch_drop, True, False)


def install_frozen_pywebview_win32(source_path: Path | None = None) -> None:
    module_name = "webview.platforms.win32"
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ImportError:
        pass

    source_path = source_path or (
        Path(sys.executable).resolve().parent
        / "webview"
        / "platforms"
        / "win32.py"
    )
    if not source_path.is_file():
        raise ImportError(f"Bundled pywebview win32 backend not found: {source_path}")

    platforms = importlib.import_module("webview.platforms")
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pywebview win32 backend: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    setattr(platforms, "win32", module)


def resolve_application_root() -> Path:
    configured = os.environ.get("FINESUB_APP_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def resolve_application_source(paths: AppPaths) -> Path:
    if paths.app_current.is_file():
        try:
            pointer = json.loads(paths.app_current.read_text(encoding="utf-8"))
        except (OSError, ValueError, AttributeError):
            pointer = {}
        current = pointer.get("current")
        if isinstance(current, str) and current:
            source = (paths.app_versions / current).resolve()
            if (
                (source / "src" / "asr_playground" / "pipeline.py").is_file()
                and (source / "pyproject.toml").is_file()
            ):
                return source
    if (
        (paths.root / "src" / "asr_playground" / "pipeline.py").is_file()
        and (paths.root / "pyproject.toml").is_file()
    ):
        return paths.root
    raise FileNotFoundError("The active FineSub application source was not found")


def resolve_frontend_url(
    paths: AppPaths,
    *,
    development_url: str | None = None,
    installer: AppInstaller | None = None,
) -> str:
    if development_url:
        return development_url
    installer = installer or AppInstaller(paths)
    pointer: dict[str, Any] = {}
    if paths.app_current.is_file():
        try:
            pointer = installer.read_pointer()
        except (OSError, ValueError, json.JSONDecodeError):
            pointer = {}
    current = pointer.get("current")
    if isinstance(current, str) and current:
        candidate = (
            paths.app_versions
            / current
            / "desktop"
            / "frontend"
            / "out"
            / "index.html"
        )
        if candidate.is_file():
            return str(candidate.resolve())
        if pointer.get("pendingHealth"):
            restored = installer.rollback_failed_start()
            if restored:
                restored_frontend = (
                    paths.app_versions
                    / restored
                    / "desktop"
                    / "frontend"
                    / "out"
                    / "index.html"
                )
                if restored_frontend.is_file():
                    return str(restored_frontend.resolve())
    development_static = paths.root / "desktop" / "frontend" / "out" / "index.html"
    if development_static.is_file():
        return str(development_static.resolve())
    raise FileNotFoundError("FineSub frontend out/index.html was not found")


def resolve_app_version(paths: AppPaths) -> str:
    candidates = (
        (paths.app_current, "current"),
        (paths.root / "launcher.json", "appVersion"),
        (paths.root / "desktop" / "frontend" / "package.json", "version"),
    )
    for path, key in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, str) and value:
            return value
    return "development"


def _load_resources(paths: AppPaths, app_source: Path) -> ResourceManager:
    manifest_path = (
        app_source / "desktop" / "resources" / "runtime-manifest.json"
    )
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = [
        ResourceSpec.model_validate(resource)
        for resource in body.get("resources", [])
    ]
    return ResourceManager(paths, specs)


def create_backend_services(
    paths: AppPaths,
    *,
    development_python: Path | None = None,
) -> tuple[JobManager, DesktopResourceService, SettingsStore]:
    app_source = resolve_application_source(paths)
    settings = SettingsStore(paths.user_data)
    bootstrap = _load_resources(paths, app_source)

    def active_uv() -> Path:
        executable = bootstrap.active_file("uv", "uv.exe")
        if executable is None:
            raise FileNotFoundError("uv must be installed before Python setup")
        return executable

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        uv_executable=active_uv,
        development_python=development_python,
    )
    resources = DesktopResourceService(
        bootstrap=bootstrap,
        runtime=runtime,
    )
    context = resources.worker_context(settings.build_worker_env())
    jobs = JobManager(
        python_executable=context.python_executable,
        working_directory=context.working_directory,
        worker_env=context.environment,
        history_path=paths.user_data / "tasks.json",
        output_root=paths.user_data / "tasks",
    )

    def refresh_worker_context() -> None:
        updated = resources.worker_context(settings.build_worker_env())
        jobs.python_executable = str(updated.python_executable)
        jobs.working_directory = str(updated.working_directory)
        jobs.worker_env = dict(updated.environment)

    resource_installs = ResourceInstallManager(
        resources,
        on_ready=refresh_worker_context,
    )
    resources.install_manager = resource_installs
    return jobs, resources, settings


def load_update_service(paths: AppPaths) -> GitHubUpdateService | None:
    config_path = paths.root / "launcher.json"
    keys_path = paths.root / "trusted-update-keys.json"
    if not config_path.is_file() or not keys_path.is_file():
        return None
    try:
        config = LauncherUpdateConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        key_document = json.loads(keys_path.read_text(encoding="utf-8"))
        keys = key_document.get("keys")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(keys, dict) or not keys:
        return None
    trusted = {
        str(key_id): str(value)
        for key_id, value in keys.items()
        if isinstance(key_id, str)
        and isinstance(value, str)
        and "replace-with" not in key_id
        and "replace-with" not in value
    }
    if not trusted:
        return None
    return GitHubUpdateService(
        paths=paths,
        config=config,
        trusted_keys=trusted,
    )


def create_application() -> tuple[Any, DesktopBridge, bool]:
    import webview

    root = resolve_application_root()
    paths = AppPaths.for_root(root)
    development_url = os.environ.get("FINESUB_DESKTOP_DEV_URL")
    development = bool(development_url)
    installer = AppInstaller(paths)
    if not development:
        installer.prepare_startup()
    frontend_url = resolve_frontend_url(
        paths,
        development_url=development_url,
        installer=installer,
    )
    jobs, resources, settings = create_backend_services(
        paths,
        development_python=Path(sys.executable) if development else None,
    )
    bridge = DesktopBridge(
        jobs=jobs,
        resources=resources,
        resource_installs=resources.install_manager,
        settings=settings,
        updates=None if development else load_update_service(paths),
        app_version=resolve_app_version(paths),
    )
    window = webview.create_window(
        PRODUCT_NAME,
        frontend_url,
        width=1180,
        height=760,
        min_size=(900, 620),
        frameless=True,
        easy_drag=False,
        background_color="#F7F7F5",
    )
    bridge.window = window
    tray_icon_path = (
        Path(getattr(sys, "_MEIPASS")) / "finesub-desktop.png"
        if getattr(sys, "frozen", False)
        else root / "desktop" / "assets" / "source" / "finesub-desktop.png"
    )
    tray = TrayController(window, tray_icon_path)
    bridge.tray = tray
    expose_bridge(window, bridge)

    def confirm_health(*_args: object) -> None:
        if development or not paths.app_current.is_file():
            return
        pointer = installer.read_pointer()
        current = pointer.get("current")
        if pointer.get("pendingHealth") and isinstance(current, str):
            installer.confirm_health(current)

    window.events.loaded += confirm_health
    window.events.loaded += lambda *_args: tray.start()
    window.events.closed += lambda *_args: tray.stop()

    def select_file() -> str | None:
        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=(
                "媒体文件 (*.wav;*.mp3;*.flac;*.m4a;*.ogg;*.mp4;*.mkv;*.mov;*.webm)",
                "所有文件 (*.*)",
            ),
        )
        return str(result[0]) if result else None

    bridge.file_selector = select_file
    return window, bridge, development


def main() -> int:
    import webview

    install_frozen_pywebview_win32()
    window, _, development = create_application()
    webview.start(
        bind_native_file_drop,
        window,
        gui="edgechromium",
        debug=development,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
