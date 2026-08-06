from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import ZipFile

import pytest

from finesub_bootstrap.paths import AppPaths
from desktop.backend.updates.installer import AppInstaller
from desktop.backend.updates.manifest import UpdateManifest


REQUIRED_APP_FILES = {
    "src/asr_playground/pipeline.py": b"pipeline",
    "desktop/backend/worker/main.py": b"worker",
    "desktop/frontend/out/index.html": b"<html></html>",
    "pyproject.toml": b"[project]\nname='finesub'\nversion='1.1.0'\n",
}


def _update_archive(
    path: Path,
    *,
    complete: bool = True,
    include_pyproject: bool = True,
) -> bytes:
    files = dict(REQUIRED_APP_FILES)
    if not complete:
        files.pop("desktop/frontend/out/index.html")
    if not include_pyproject:
        files.pop("pyproject.toml")
    files["app-manifest.json"] = b'{"version":"1.1.0","platform":"windows-x64"}'
    with ZipFile(path, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return path.read_bytes()


def _manifest(body: bytes) -> UpdateManifest:
    return UpdateManifest.model_validate(
        {
            "schemaVersion": 1,
            "keyId": "release-key",
            "version": "1.1.0",
            "channel": "stable",
            "platform": "windows-x64",
            "draft": False,
            "prerelease": False,
            "minimumLauncherVersion": "1.0.0",
            "minimumSupportedVersion": "1.0.0",
            "releaseNotes": "",
            "assets": {
                "app": {
                    "url": "https://example.com/app.zip",
                    "size": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "supportedFrom": ["1.0.0"],
                },
                "full": {
                    "url": "https://example.com/full.zip",
                    "size": 1,
                    "sha256": "2" * 64,
                },
            },
        }
    )


def test_app_install_switches_pointer_only_after_validation(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    paths.app_versions.mkdir(parents=True)
    installer = AppInstaller(paths)
    installer.write_pointer(
        current="1.0.0",
        previous=None,
        pending_health=False,
    )
    archive = tmp_path / "app.zip"
    body = _update_archive(archive)

    pending = installer.install(archive, _manifest(body))

    current = json.loads(
        (paths.root / "app" / "current.json").read_text(encoding="utf-8")
    )
    assert pending.version == "1.1.0"
    assert current["current"] == "1.1.0"
    assert current["previous"] == "1.0.0"
    assert current["pendingHealth"] is True


def test_failed_health_check_restores_previous_version(tmp_path: Path) -> None:
    installer = AppInstaller(AppPaths.for_root(tmp_path / "FineSub"))
    installer.write_pointer(
        current="1.1.0",
        previous="1.0.0",
        pending_health=True,
    )

    restored = installer.rollback_failed_start()

    assert restored == "1.0.0"
    assert installer.read_pointer()["current"] == "1.0.0"
    assert installer.read_pointer()["pendingHealth"] is False


def test_pointer_reader_accepts_utf8_bom() -> None:
    root = Path(__file__).resolve().parents[3] / "dist" / f"bom-test-{os.getpid()}"
    try:
        paths = AppPaths.for_root(root / "FineSub")
        installer = AppInstaller(paths)
        installer.pointer_path.parent.mkdir(parents=True)
        installer.pointer_path.write_text(
            '{"current":"1.0.0","previous":null,"pendingHealth":false}',
            encoding="utf-8-sig",
        )

        assert installer.read_pointer()["current"] == "1.0.0"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_invalid_app_archive_does_not_switch_pointer(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    installer = AppInstaller(paths)
    installer.write_pointer(
        current="1.0.0",
        previous=None,
        pending_health=False,
    )
    archive = tmp_path / "app.zip"
    body = _update_archive(archive, complete=False)

    with pytest.raises(FileNotFoundError):
        installer.install(archive, _manifest(body))

    assert installer.read_pointer()["current"] == "1.0.0"


def test_pending_app_gets_one_health_attempt_then_rolls_back(
    tmp_path: Path,
) -> None:
    installer = AppInstaller(AppPaths.for_root(tmp_path / "FineSub"))
    installer.write_pointer(
        current="1.1.0",
        previous="1.0.0",
        pending_health=True,
    )

    first_start = installer.prepare_startup()
    first_pointer = installer.read_pointer()
    second_start = installer.prepare_startup()
    second_pointer = installer.read_pointer()

    assert first_start == "1.1.0"
    assert first_pointer["healthAttempts"] == 1
    assert second_start == "1.0.0"
    assert second_pointer["current"] == "1.0.0"
    assert second_pointer["pendingHealth"] is False


def test_app_archive_without_dependency_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    installer = AppInstaller(paths)
    installer.write_pointer(
        current="1.0.0",
        previous=None,
        pending_health=False,
    )
    archive = tmp_path / "app.zip"
    body = _update_archive(archive, include_pyproject=False)

    with pytest.raises(FileNotFoundError, match="pyproject"):
        installer.install(archive, _manifest(body))

    assert installer.read_pointer()["current"] == "1.0.0"
