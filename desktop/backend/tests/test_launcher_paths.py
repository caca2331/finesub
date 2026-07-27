from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil

from desktop.backend.common.paths import AppPaths
from desktop.backend.launcher.main import (
    create_backend_services,
    dropped_file_path,
    expose_bridge,
    load_update_service,
    resolve_app_version,
    resolve_application_source,
    resolve_frontend_url,
)
from desktop.backend.updates.installer import AppInstaller
from desktop.backend.launcher.bridge import DesktopBridge
from desktop.backend.settings.store import SettingsStore


def test_development_url_takes_precedence(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")

    assert (
        resolve_frontend_url(paths, development_url="http://127.0.0.1:3000")
        == "http://127.0.0.1:3000"
    )


def test_app_version_follows_installed_current_pointer(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    paths.app_current.parent.mkdir(parents=True)
    paths.app_current.write_text(
        '{"current":"2.3.4","previous":null,"pendingHealth":false}',
        encoding="utf-8",
    )

    assert resolve_app_version(paths) == "2.3.4"


def test_native_drop_uses_pywebview_full_path_only() -> None:
    assert (
        dropped_file_path(
            {
                "dataTransfer": {
                    "files": [
                        {
                            "name": "video.mp4",
                            "pywebviewFullPath": "C:/media/video.mp4",
                        }
                    ]
                }
            }
        )
        == "C:/media/video.mp4"
    )
    assert (
        dropped_file_path(
            {"dataTransfer": {"files": [{"name": "video.mp4"}]}}
        )
        is None
    )


def test_bridge_exposes_only_the_public_desktop_api(tmp_path: Path) -> None:
    exposed: list[str] = []

    class FakeWindow:
        def expose(self, *functions: object) -> None:
            exposed.extend(function.__name__ for function in functions)

    bridge = DesktopBridge(
        jobs=object(),
        resources=object(),
        settings=SettingsStore(tmp_path),
    )

    expose_bridge(FakeWindow(), bridge)

    assert exposed == [
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
        "maximize_window",
        "close_window",
    ]


def test_installed_static_frontend_uses_app_current_pointer(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    frontend = (
        paths.app_versions / "1.2.0" / "desktop" / "frontend" / "out" / "index.html"
    )
    frontend.parent.mkdir(parents=True)
    frontend.write_text("<html></html>", encoding="utf-8")
    paths.app_current.parent.mkdir(parents=True, exist_ok=True)
    paths.app_current.write_text(
        '{"current":"1.2.0","previous":"1.1.0","pendingHealth":false}',
        encoding="utf-8",
    )

    resolved = resolve_frontend_url(paths)

    assert resolved == str(frontend.resolve())


def test_pending_broken_app_rolls_back_to_previous_frontend(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    previous = (
        paths.app_versions / "1.1.0" / "desktop" / "frontend" / "out" / "index.html"
    )
    previous.parent.mkdir(parents=True)
    previous.write_text("<html>previous</html>", encoding="utf-8")
    installer = AppInstaller(paths)
    installer.write_pointer(
        current="1.2.0",
        previous="1.1.0",
        pending_health=True,
    )

    resolved = resolve_frontend_url(paths, installer=installer)

    pointer = json.loads(paths.app_current.read_text(encoding="utf-8"))
    assert resolved == str(previous.resolve())
    assert pointer["current"] == "1.1.0"


def test_development_static_frontend_falls_back_to_repo_copy(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    frontend = paths.root / "desktop" / "frontend" / "out" / "index.html"
    frontend.parent.mkdir(parents=True)
    frontend.write_text("<html></html>", encoding="utf-8")

    assert resolve_frontend_url(paths) == str(frontend.resolve())


def test_installed_worker_source_follows_current_app_pointer(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path)
    source = paths.app_versions / "1.2.0"
    (source / "src").mkdir(parents=True)
    (source / "src" / "pipeline.py").write_text("ok", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]", encoding="utf-8")
    paths.app_current.parent.mkdir(parents=True, exist_ok=True)
    paths.app_current.write_text(
        '{"current":"1.2.0","previous":"1.1.0","pendingHealth":false}',
        encoding="utf-8",
    )

    assert resolve_application_source(paths) == source.resolve()


def test_development_services_run_worker_from_repository_source(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    (paths.root / "src").mkdir(parents=True)
    (paths.root / "src" / "pipeline.py").write_text("ok", encoding="utf-8")
    (paths.root / "pyproject.toml").write_text("[project]", encoding="utf-8")
    resources = paths.root / "desktop" / "resources"
    resources.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).parents[2] / "resources" / "runtime-manifest.json",
        resources / "runtime-manifest.json",
    )
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")

    jobs, desktop_resources, _ = create_backend_services(
        paths,
        development_python=python,
    )

    assert jobs.python_executable == str(python.resolve())
    assert jobs.working_directory == str(paths.root)
    assert desktop_resources.check_all()[0].state == "ready"


def test_installed_services_load_resources_from_current_app_version(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    source = paths.app_versions / "1.2.0"
    (source / "src").mkdir(parents=True)
    (source / "src" / "pipeline.py").write_text("ok", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]", encoding="utf-8")
    resources = source / "desktop" / "resources"
    resources.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).parents[2] / "resources" / "runtime-manifest.json",
        resources / "runtime-manifest.json",
    )
    paths.app_current.parent.mkdir(parents=True, exist_ok=True)
    paths.app_current.write_text(
        '{"current":"1.2.0","previous":null,"pendingHealth":false}',
        encoding="utf-8",
    )

    jobs, desktop_resources, _ = create_backend_services(paths)

    assert jobs.working_directory == str(source.resolve())
    assert len(desktop_resources.check_all()) == 2


def test_update_service_loads_only_with_configured_trusted_key(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    paths.root.mkdir(parents=True)
    (paths.root / "launcher.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "appVersion": "1.0.0",
                "launcherVersion": "1.0.0",
                "channel": "stable",
                "platform": "windows-x64",
                "releaseRepository": "caca2331/finesub",
            }
        ),
        encoding="utf-8",
    )
    (paths.root / "trusted-update-keys.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "keys": {
                    "release-key": base64.b64encode(b"k" * 32).decode("ascii")
                },
            }
        ),
        encoding="utf-8",
    )

    service = load_update_service(paths)

    assert service is not None
    assert service.config.release_repository == "caca2331/finesub"
    assert service.trusted_keys == {
        "release-key": base64.b64encode(b"k" * 32).decode("ascii")
    }


def test_update_service_is_disabled_for_placeholder_key(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    paths.root.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).parents[2] / "resources" / "launcher.example.json",
        paths.root / "launcher.json",
    )
    shutil.copy2(
        Path(__file__).parents[2]
        / "resources"
        / "trusted-update-keys.example.json",
        paths.root / "trusted-update-keys.json",
    )

    assert load_update_service(paths) is None
