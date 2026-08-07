from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from finesub_bootstrap.models import DownloadAsset, ResourceSpec
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.downloader import DigestMismatch
from finesub_bootstrap.resources import ResourceManager


def test_every_managed_resource_installs_under_the_runtime() -> None:
    """Managed tools must stay private to one installation.

    Their "which version is active" pointer is a single file per resource root.
    Under `models` -- which several installations share -- two apps pinned to
    different manifest versions would flip that pointer back and forth, and a
    packaged command line (which cannot provision) would refuse to run every
    other time. Under `runtime`, which is never shared, the question does not
    arise. Anything moved to `models` needs a different pointer scheme first.
    """

    manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "resources"
            / "runtime-manifest.json"
        ).read_text(encoding="utf-8")
    )

    destinations = {
        resource["id"]: resource.get("destination")
        for resource in manifest["resources"]
    }

    assert set(destinations.values()) == {"runtime"}, destinations


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


def test_install_recovers_complete_version_without_redownloading(
    tmp_path: Path,
) -> None:
    body = b"unused"
    paths = AppPaths.for_root(tmp_path / "app-root")
    manager = ResourceManager(
        paths,
        [_spec("https://invalid.example/ffmpeg.zip", body)],
    )
    version = paths.runtime / "ffmpeg" / "7.1" / "bin"
    version.mkdir(parents=True)
    (version / "ffmpeg.exe").write_bytes(b"ffmpeg")
    (version / "ffprobe.exe").write_bytes(b"ffprobe")

    status = manager.install("ffmpeg", lambda event: None)

    assert status.state == "ready"
    assert manager.active_version("ffmpeg") == "7.1"


def test_install_replaces_an_incomplete_final_version(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"new", "bin/ffprobe.exe": b"new"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    incomplete = paths.runtime / "ffmpeg" / "7.1" / "bin"
    incomplete.mkdir(parents=True)
    (incomplete / "ffmpeg.exe").write_bytes(b"incomplete")
    manager = ResourceManager(paths, [_spec(server.url, body)])

    status = manager.install("ffmpeg", lambda event: None)

    assert status.state == "ready"
    assert (incomplete / "ffmpeg.exe").read_bytes() == b"new"
    assert (incomplete / "ffprobe.exe").read_bytes() == b"new"


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
    # git and yt-dlp join uv and ffmpeg, but they are installed lazily -- only
    # a task that needs them (knowledge=update, URL input) pulls them down.
    assert {resource.id for resource in resources} == {
        "uv",
        "ffmpeg",
        "git",
        "yt-dlp",
    }
    for resource in resources:
        assert "/latest/" not in resource.asset.url
        assert resource.asset.size > 0
        assert resource.asset.sha256 != "0" * 64

    by_id = {resource.id: resource for resource in resources}
    # The required file is what proves an extraction produced the tool. yt-dlp
    # ships as a wheel, which is a zip of the importable package, so the marker
    # is a module rather than an executable.
    assert by_id["git"].required_files == ["cmd/git.exe"]
    assert by_id["yt-dlp"].required_files == ["yt_dlp/__init__.py"]
