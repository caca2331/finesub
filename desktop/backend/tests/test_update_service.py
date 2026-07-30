from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from desktop.backend.common.paths import AppPaths
from desktop.backend.updates.service import (
    GitHubUpdateService,
    LauncherUpdateConfig,
    _read_limited_body,
)


def _zip(path: Path, files: dict[str, bytes]) -> bytes:
    with ZipFile(path, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return path.read_bytes()


def _fixture(
    tmp_path: Path,
    *,
    minimum_launcher: str = "1.0.0",
    include_full_app: bool = True,
) -> tuple[GitHubUpdateService, dict[str, bytes], list[list[str]]]:
    paths = AppPaths.for_root(tmp_path / "FineSub")
    app_body = _zip(
        tmp_path / "app.zip",
        {
            "src/asr_playground/pipeline.py": b"pipeline",
            "desktop/backend/worker/main.py": b"worker",
            "desktop/frontend/out/index.html": b"<html></html>",
            "pyproject.toml": b"[project]\nname='finesub'\nversion='1.1.0'\n",
            "app-manifest.json": (
                b'{"version":"1.1.0","platform":"windows-x64"}'
            ),
        },
    )
    full_files = {
        "FineSub Desktop.exe": b"launcher",
        "updater/FineSub Desktop Updater.exe": b"updater",
    }
    if include_full_app:
        full_files.update(
            {
                "app/current.json": (
                    b'{"current":"1.1.0","previous":null,"pendingHealth":false}'
                ),
                "app/versions/1.1.0/src/asr_playground/pipeline.py": b"pipeline",
                "app/versions/1.1.0/desktop/backend/worker/main.py": b"worker",
                "app/versions/1.1.0/desktop/frontend/out/index.html": (
                    b"<html></html>"
                ),
                "app/versions/1.1.0/pyproject.toml": (
                    b"[project]\nname='finesub'\nversion='1.1.0'\n"
                ),
                "app/versions/1.1.0/app-manifest.json": (
                    b'{"version":"1.1.0","platform":"windows-x64"}'
                ),
            }
        )
    full_body = _zip(tmp_path / "full.zip", full_files)
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    manifest = {
        "schemaVersion": 1,
        "keyId": "release-key",
        "version": "1.1.0",
        "channel": "stable",
        "platform": "windows-x64",
        "draft": False,
        "prerelease": False,
        "minimumLauncherVersion": minimum_launcher,
        "minimumSupportedVersion": "1.0.0",
        "releaseNotes": "修复字幕处理并改进界面",
        "assets": {
            "app": {
                "url": "https://downloads.example/app.zip",
                "size": len(app_body),
                "sha256": hashlib.sha256(app_body).hexdigest(),
                "supportedFrom": ["1.0.0"],
            },
            "full": {
                "url": "https://downloads.example/full.zip",
                "size": len(full_body),
                "sha256": hashlib.sha256(full_body).hexdigest(),
            },
        },
    }
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = base64.b64encode(private.sign(manifest_bytes))
    payloads = {
        "https://downloads.example/manifest": manifest_bytes,
        "https://downloads.example/signature": signature,
        "https://downloads.example/app.zip": app_body,
        "https://downloads.example/full.zip": full_body,
    }
    release = {
        "tag_name": "v1.1.0",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": "update-manifest.json",
                "browser_download_url": "https://downloads.example/manifest",
            },
            {
                "name": "update-manifest.sig",
                "browser_download_url": "https://downloads.example/signature",
            },
        ],
    }
    launched: list[list[str]] = []

    def download(asset, destination, progress):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[asset.url])
        return destination

    service = GitHubUpdateService(
        paths=paths,
        config=LauncherUpdateConfig(
            app_version="1.0.0",
            launcher_version="1.0.0",
            channel="stable",
            platform="windows-x64",
            release_repository="caca2331/finesub",
        ),
        trusted_keys={
            "release-key": base64.b64encode(public).decode("ascii")
        },
        release_fetcher=lambda repository, channel: release,
        bytes_fetcher=lambda url, limit: payloads[url],
        asset_downloader=download,
        process_launcher=lambda command: launched.append(list(command)),
    )
    return service, payloads, launched


def test_check_verifies_release_and_selects_small_app_update(
    tmp_path: Path,
) -> None:
    service, _, _ = _fixture(tmp_path)

    result = service.check()

    assert result["available"] is True
    assert result["version"] == "1.1.0"
    assert result["kind"] == "app"
    assert result["releaseNotes"] == "修复字幕处理并改进界面"
    assert result["mandatory"] is False
    assert result["size"] > 0
    assert result["releaseUrl"] == (
        "https://github.com/caca2331/finesub/releases/tag/v1.1.0"
    )


def test_app_update_downloads_verified_archive_and_switches_pointer(
    tmp_path: Path,
) -> None:
    service, _, _ = _fixture(tmp_path)
    service.paths.app_current.parent.mkdir(parents=True)
    service.paths.app_current.write_text(
        '{"current":"1.0.0","previous":null,"pendingHealth":false}',
        encoding="utf-8",
    )
    service.check()

    result = service.install("app")

    pointer = json.loads(service.paths.app_current.read_text("utf-8"))
    assert result["restartRequired"] is True
    assert result["kind"] == "app"
    assert pointer["current"] == "1.1.0"
    assert pointer["pendingHealth"] is True


def test_check_rejects_release_metadata_that_conflicts_with_manifest(
    tmp_path: Path,
) -> None:
    service, _, _ = _fixture(tmp_path)
    release = service.release_fetcher("caca2331/finesub", "stable")
    service.release_fetcher = lambda repository, channel: {
        **release,
        "draft": True,
    }

    with pytest.raises(ValueError, match="draft"):
        service.check()


def test_full_update_stages_archive_and_launches_isolated_updater(
    tmp_path: Path,
) -> None:
    service, _, launched = _fixture(tmp_path, minimum_launcher="2.0.0")
    installed_updater = service.paths.root / "updater"
    installed_updater.mkdir(parents=True)
    (installed_updater / "FineSub Desktop Updater.exe").write_bytes(
        b"current-updater"
    )
    service.paths.root.mkdir(exist_ok=True)
    (service.paths.root / "FineSub Desktop.exe").write_bytes(
        b"current-launcher"
    )
    service.check()

    result = service.install("full")

    assert result["exitRequired"] is True
    assert result["kind"] == "full"
    assert len(launched) == 1
    runner = Path(launched[0][0])
    request_path = Path(launched[0][2])
    assert runner.is_file()
    request = json.loads(request_path.read_text("utf-8"))
    assert Path(request["source"], "FineSub Desktop.exe").is_file()
    assert Path(request["target"]) == service.paths.root
    assert request["relaunch_path"] == "FineSub Desktop.exe"
    assert "app" in request["preserved"]


def test_full_update_without_versioned_app_is_rejected(tmp_path: Path) -> None:
    service, _, _ = _fixture(
        tmp_path,
        minimum_launcher="2.0.0",
        include_full_app=False,
    )
    installed_updater = service.paths.root / "updater"
    installed_updater.mkdir(parents=True)
    (installed_updater / "FineSub Desktop Updater.exe").write_bytes(b"updater")
    service.check()

    with pytest.raises(FileNotFoundError, match="App"):
        service.install("full")


class _ChunkedResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def iter_bytes(self):
        yield from self._chunks


def test_read_limited_body_rejects_stream_before_exceeding_limit() -> None:
    response = _ChunkedResponse([b"abc", b"def"])
    assert _read_limited_body(response, 6) == b"abcdef"  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exceeds 5 bytes"):
        _read_limited_body(response, 5)  # type: ignore[arg-type]
