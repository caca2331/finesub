from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from finesub_bootstrap.migrations import apply_pending
from finesub_bootstrap.models import ResourceSpec
from finesub_bootstrap.paths import AppPaths, ensure_store, load_app_paths
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap.environment import RuntimeEnvironment

from desktop.backend.common.product import PRODUCT_NAME
from desktop.backend.jobs.manager import JobManager
from desktop.backend.launcher.bridge import DesktopBridge
from desktop.backend.launcher.tray import TrayController
from desktop.backend.resources.desktop_service import DesktopResourceService
from desktop.backend.resources.install_manager import ResourceInstallManager
from desktop.backend.settings.store import SettingsStore
from desktop.backend.updates.install_manager import UpdateInstallManager
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
    "reveal_api_keys",
    "check_updates",
    "install_update",
    "get_update_install",
    "open_update_page",
    "open_tasks_directory",
    "open_output",
    "minimize_window",
    "minimize_to_tray",
    "maximize_window",
    "close_window",
    "set_window_chrome",
)

# The frameless window draws its own title bar, so the native hit test has to
# be told where it is. These two mirror the CSS -- `--titlebar-height` and the
# three 46px buttons of `.window-actions`; test_window_config.py keeps the two
# sides in step, because a drifting caption band silently steals or leaks the
# drag area.
TITLEBAR_HEIGHT_DP = 40
WINDOW_CONTROLS_WIDTH_DP = 138
# Long enough to cover a cold start on a slow disk, short enough that a window
# which never shows does not hang the callback for the whole session.
WINDOW_READY_TIMEOUT_SECONDS = 10
# The web layer re-applies these from the active theme the moment it paints
# (`useAppearance` -> `set_window_chrome`), so they only decide what the frame
# looks like until then. Following the Windows setting means the default
# "system" theme never flashes the opposite one. Keep in step with the
# `--app-bg` / `--text` pairs in globals.css.
LIGHT_WINDOW_COLORS = ("#F2F3F5", "#1A1A1E")
DARK_WINDOW_COLORS = ("#131316", "#E8E9EC")


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


def prepare_window(window: Any) -> None:
    """Everything that needs a real OS window, once pywebview has shown one.

    pywebview runs this callback before the WinForms window exists, so the
    native work waits for `shown`. A slow start must not take the file drop
    down with it: the window is merely plain without native chrome, but
    useless without drop, so the wait failing is a warning, not an abort.
    """

    if window.events.shown.wait(WINDOW_READY_TIMEOUT_SECONDS):
        background, foreground = system_theme_colors()
        apply_native_window_chrome(window, background, foreground)
        enable_native_window_resize(window)
    else:
        print(
            "Warning: the window was not shown within "
            f"{WINDOW_READY_TIMEOUT_SECONDS}s; "
            "native frame colors and resizing are unavailable",
            file=sys.stderr,
        )
    bind_native_file_drop(window)


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
    # dragenter/dragover exist only for their preventDefault, which pywebview
    # emits synchronously in the injected listener -- the Python callback does
    # nothing. Undebounced, dragover fires tens of times a second and each one
    # serializes a DragEvent (dataTransfer included) across the bridge, which is
    # felt as a stutter while dragging. The debounce collapses that; the drop
    # itself stays immediate.
    events.dragenter += DOMEventHandler(ignore_drag, True, False, debounce=500)
    events.dragover += DOMEventHandler(ignore_drag, True, False, debounce=500)
    events.drop += DOMEventHandler(dispatch_drop, True, False)


def _rgb(value: str) -> tuple[int, int, int]:
    """Parse a CSS color into its channels.

    What arrives is whatever the stylesheet author wrote -- the frontend reads
    these straight out of computed custom properties. Six-digit hex is what the
    themes use today; accepting the short form and rgb() keeps a later theme
    from turning the whole feature into a silently swallowed bridge failure.
    """

    text = value.strip()
    functional = re.fullmatch(r"rgba?\(([^)]*)\)", text, re.IGNORECASE)
    if functional:
        parts = [part for part in re.split(r"[,\s/]+", functional.group(1)) if part]
        if len(parts) < 3:
            raise ValueError(f"unsupported window color: {value!r}")
        channels = tuple(int(round(float(part))) for part in parts[:3])
    else:
        digits = text.lstrip("#")
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        if len(digits) != 6:
            raise ValueError(f"unsupported window color: {value!r}")
        channels = tuple(int(digits[index : index + 2], 16) for index in (0, 2, 4))
    return tuple(min(255, max(0, channel)) for channel in channels)  # type: ignore[return-value]


def _colorref(value: str) -> int:
    red, green, blue = _rgb(value)
    return red | (green << 8) | (blue << 16)


def system_theme_colors() -> tuple[str, str]:
    """The (background, foreground) pair Windows is currently themed for."""

    if os.name != "nt":
        return LIGHT_WINDOW_COLORS
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            apps_use_light, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
    except OSError:
        # The value is absent on some builds; light is the Windows default.
        return LIGHT_WINDOW_COLORS
    return LIGHT_WINDOW_COLORS if apps_use_light else DARK_WINDOW_COLORS


def apply_native_window_chrome(
    window: Any,
    background: str,
    foreground: str,
) -> None:
    """Match the Windows frame and resize border to the active app theme."""
    if os.name != "nt":
        return
    native = getattr(window, "native", None)
    if native is None:
        return

    import ctypes
    from webview.platforms.winforms import Func, Type

    background_ref = _colorref(background)
    foreground_ref = _colorref(foreground)
    handle = native.Handle
    hwnd = handle.ToInt64() if hasattr(handle, "ToInt64") else int(handle)
    hwnd_ptr = ctypes.c_void_p(hwnd)

    def install() -> None:
        try:
            from System.Drawing import ColorTranslator

            native.BackColor = ColorTranslator.FromHtml(
                "#%02X%02X%02X" % _rgb(background)
            )
        except Exception:
            # Cosmetic only: this is the color behind the web view while it
            # paints, and the DWM attributes below matter far more.
            pass
        try:
            dwmapi = ctypes.WinDLL("dwmapi")
            dwmapi.DwmSetWindowAttribute.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
            for attribute, color in (
                (34, background_ref),  # DWMWA_BORDER_COLOR
                (35, background_ref),  # DWMWA_CAPTION_COLOR
                (36, foreground_ref),  # DWMWA_TEXT_COLOR
            ):
                color_value = ctypes.c_uint32(color)
                dwmapi.DwmSetWindowAttribute(
                    hwnd_ptr,
                    attribute,
                    ctypes.byref(color_value),
                    ctypes.sizeof(color_value),
                )
        except (OSError, AttributeError):
            # Older Windows builds may not expose the DWM color attributes.
            pass

    native.Invoke(Func[Type](install))


def enable_native_window_resize(window: Any) -> None:
    """Install frameless resize hit-testing on the WinForms UI thread."""
    if os.name != "nt":
        return
    native = getattr(window, "native", None)
    if native is None:
        return

    import ctypes
    from ctypes import wintypes
    from webview.platforms.winforms import Func, Type

    handle = native.Handle
    hwnd = handle.ToInt64() if hasattr(handle, "ToInt64") else int(handle)
    hwnd_ptr = ctypes.c_void_p(hwnd)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    comctl32 = ctypes.WinDLL("comctl32", use_last_error=True)
    get_style = (
        user32.GetWindowLongPtrW
        if hasattr(user32, "GetWindowLongPtrW")
        else user32.GetWindowLongW
    )
    set_style = (
        user32.SetWindowLongPtrW
        if hasattr(user32, "SetWindowLongPtrW")
        else user32.SetWindowLongW
    )
    get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    set_style.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    set_style.restype = ctypes.c_ssize_t
    user32.SetWindowPos.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = ctypes.c_bool
    user32.IsZoomed.argtypes = [ctypes.c_void_p]
    user32.IsZoomed.restype = ctypes.c_bool

    subclass_proc_type = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    )
    comctl32.SetWindowSubclass.argtypes = [
        ctypes.c_void_p,
        subclass_proc_type,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    comctl32.SetWindowSubclass.restype = ctypes.c_bool
    comctl32.RemoveWindowSubclass.argtypes = [
        ctypes.c_void_p,
        subclass_proc_type,
        ctypes.c_size_t,
    ]
    comctl32.RemoveWindowSubclass.restype = ctypes.c_bool
    comctl32.DefSubclassProc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
    ]
    comctl32.DefSubclassProc.restype = ctypes.c_ssize_t

    @subclass_proc_type
    def hit_test_proc(
        message_hwnd: int,
        message: int,
        wparam: int,
        lparam: int,
        subclass_id: int,
        _reference: int,
    ) -> int:
        if message == 0x0084:  # WM_NCHITTEST
            bounds = wintypes.RECT()
            if user32.GetWindowRect(message_hwnd, ctypes.byref(bounds)):
                packed = lparam
                x = packed & 0xFFFF
                y = (packed >> 16) & 0xFFFF
                x = x - 0x10000 if x & 0x8000 else x
                y = y - 0x10000 if y & 0x8000 else y
                scale = getattr(native, "DeviceDpi", 96) / 96
                border = max(8, round(8 * scale))
                on_left = x < bounds.left + border
                on_right = x >= bounds.right - border
                on_top = y < bounds.top + border
                on_bottom = y >= bounds.bottom - border

                if not user32.IsZoomed(message_hwnd):
                    hit = (
                        13 if on_top and on_left else
                        14 if on_top and on_right else
                        16 if on_bottom and on_left else
                        17 if on_bottom and on_right else
                        10 if on_left else
                        11 if on_right else
                        12 if on_top else
                        15 if on_bottom else
                        0
                    )
                    if hit:
                        return hit

                # Dragging by the title bar has a second implementation in the
                # web layer (`pywebview-drag-region`). Which one is reachable
                # depends on whether the WebView2 child window covers these
                # coordinates -- where it does, the child gets the mouse and
                # this branch is never asked. Both are kept because the
                # unreachable one costs nothing and the region carries no
                # interactive controls either way: window buttons live in the
                # excluded strip on the right.
                caption_height = round(TITLEBAR_HEIGHT_DP * scale)
                controls_width = round(WINDOW_CONTROLS_WIDTH_DP * scale)
                if y < bounds.top + caption_height and x < bounds.right - controls_width:
                    return 2  # HTCAPTION
        elif message == 0x0082:  # WM_NCDESTROY
            comctl32.RemoveWindowSubclass(message_hwnd, hit_test_proc, subclass_id)

        return comctl32.DefSubclassProc(message_hwnd, message, wparam, lparam)

    def install() -> None:
        style = get_style(hwnd_ptr, -16)  # GWL_STYLE
        style |= 0x00040000 | 0x00020000 | 0x00010000  # THICKFRAME, MIN/MAXBOX
        set_style(hwnd_ptr, -16, style)
        user32.SetWindowPos(
            hwnd_ptr,
            None,
            0,
            0,
            0,
            0,
            0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020,
        )  # NOSIZE | NOMOVE | NOZORDER | NOACTIVATE | FRAMECHANGED
        if not comctl32.SetWindowSubclass(hwnd_ptr, hit_test_proc, 1, 0):
            raise ctypes.WinError(ctypes.get_last_error())

    native.Invoke(Func[Type](install))
    window._finesub_hit_test_proc = hit_test_proc


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


def resolve_application_paths(root: Path) -> AppPaths:
    return load_app_paths(root)


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
    # Before anything reads personal data, and never fatal: a failed migration
    # is logged and retried at the next start.
    apply_pending(paths)
    ensure_store(paths)
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
        runtime_lock=app_source / "desktop" / "runtime" / "pylock.win-py312.toml",
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
        # Outputs are big and rebuildable-ish; the history that indexes them is
        # small and irreplaceable. They live on opposite sides of that split.
        output_root=paths.tasks,
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
    paths = resolve_application_paths(root)
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
    updates = None if development else load_update_service(paths)
    bridge = DesktopBridge(
        jobs=jobs,
        resources=resources,
        resource_installs=resources.install_manager,
        settings=settings,
        updates=updates,
        update_installs=(
            UpdateInstallManager(updates) if updates is not None else None
        ),
        app_version=resolve_app_version(paths),
    )
    window = webview.create_window(
        PRODUCT_NAME,
        frontend_url,
        width=1180,
        height=760,
        resizable=True,
        min_size=(720, 520),
        frameless=True,
        easy_drag=False,
        background_color=system_theme_colors()[0],
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
        prepare_window,
        window,
        gui="edgechromium",
        debug=development,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
