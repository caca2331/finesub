from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from desktop.backend.common.models import DownloadAsset, ResourceSpec
from desktop.backend.common.paths import AppPaths
from desktop.backend.resources.downloader import DigestMismatch
from desktop.backend.resources.manager import ResourceManager


def _zip_bytes(path: Path, members: dict[str, bytes]) -> bytes:
    with ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return path.read_bytes()


def _spec(url: str, body: bytes, *, sha256: str | None = None) -> ResourceSpec:
    return ResourceSpec(
        id="ffmpeg",
        version="7.1",
        destination="runtime",
        directory="ffmpeg",
        archive_type="zip",
        required_files=["bin/ffmpeg.exe", "bin/ffprobe.exe"],
        asset=DownloadAsset(
            url=url,
            size=len(body),
            sha256=sha256 or hashlib.sha256(body).hexdigest(),
        ),
    )


def test_resource_install_becomes_ready_only_after_required_files_exist(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"ffmpeg", "bin/ffprobe.exe": b"ffprobe"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    manager = ResourceManager(paths, [_spec(server.url, body)])

    status = manager.install("ffmpeg", lambda event: None)

    assert status.state == "ready"
    assert status.version == "7.1"
    assert (
        paths.runtime / "ffmpeg" / "7.1" / "bin" / "ffmpeg.exe"
    ).is_file()
    pointer = json.loads(
        (paths.runtime / "ffmpeg" / "current.json").read_text("utf-8")
    )
    assert pointer == {"current": "7.1"}


def test_failed_install_leaves_previous_version_active(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"ffmpeg", "bin/ffprobe.exe": b"ffprobe"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    resource_root = paths.runtime / "ffmpeg"
    (resource_root / "7.0" / "bin").mkdir(parents=True)
    (resource_root / "current.json").write_text(
        '{"current":"7.0"}',
        encoding="utf-8",
    )
    manager = ResourceManager(
        paths,
        [_spec(server.url, body, sha256="0" * 64)],
    )

    with pytest.raises(DigestMismatch):
        manager.install("ffmpeg", lambda event: None)

    assert manager.active_version("ffmpeg") == "7.0"


def test_resource_without_required_file_is_not_activated(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = _zip_bytes(tmp_path / "ffmpeg.zip", {"bin/ffmpeg.exe": b"ffmpeg"})
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    manager = ResourceManager(paths, [_spec(server.url, body)])

    with pytest.raises(FileNotFoundError):
        manager.install("ffmpeg", lambda event: None)

    assert manager.active_version("ffmpeg") is None


def test_runtime_manifest_uses_pinned_verified_windows_assets() -> None:
    manifest_path = (
        Path(__file__).parents[2] / "resources" / "runtime-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    resources = [
        ResourceSpec.model_validate(resource)
        for resource in manifest["resources"]
    ]
    assert {resource.id for resource in resources} == {"uv", "ffmpeg"}
    for resource in resources:
        assert "/latest/" not in resource.asset.url
        assert resource.asset.size > 0
        assert resource.asset.sha256 != "0" * 64
