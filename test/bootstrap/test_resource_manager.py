from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from finesub_bootstrap.models import DownloadAsset, ResolvableAsset, ResourceSpec
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.downloader import DigestMismatch
from finesub_bootstrap import resources
from finesub_bootstrap.resources import ResourceManager

#: Anchored on the repository rather than counted in `..`s from this file: a
#: test that moves house should not start reading a path that happens to exist
#: somewhere else.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
            REPOSITORY_ROOT
            / "src"
            / "finesub_bootstrap"
            / "runtime-manifest.json"
        ).read_text(encoding="utf-8")
    )

    destinations = {
        resource["id"]: resource.get("destination")
        for resource in manifest["resources"]
    }

    assert set(destinations.values()) == {"runtime"}, destinations


def _dangle(link: Path, target: Path) -> None:
    """Leave `link` owning its name while resolving to nothing.

    A junction where one can be made, because that is the redirect users
    actually leave behind on Windows (moving `models` off the system drive is a
    documented setup) and it needs no privilege -- unlike a symlink, which the
    test user usually may not create there. `CreateJunction` requires the
    target to exist, so it is made and then taken away.
    """

    if os.name == "nt":
        import _winapi

        target.mkdir(parents=True, exist_ok=True)
        _winapi.CreateJunction(str(target), str(link))
        target.rmdir()
        return
    link.symlink_to(target, True)


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


def test_runtime_manifest_pins_every_asset_it_can() -> None:
    manifest_path = (
        REPOSITORY_ROOT / "src" / "finesub_bootstrap" / "runtime-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    resources = [
        ResourceSpec.model_validate(resource)
        for resource in manifest["resources"]
    ]
    # git and yt-dlp join uv and ffmpeg, but they are installed lazily -- only
    # a task that needs them (knowledge=update, URL input) pulls them down.
    # tokcount is lazier still: an LLM run fetches it, and runs anyway if that
    # fails, because the free countTokens endpoint does the same job online.
    assert {resource.id for resource in resources} == {
        "uv",
        "ffmpeg",
        "git",
        "yt-dlp",
        "tokcount",
    }
    # Pinning is the default and the exception is named here, not inferred: an
    # unpinned asset means whatever upstream built today, which nobody tested.
    # ffmpeg earns it because BtbN replaces the bytes behind `latest` on every
    # build -- a pin there fails within a day -- and because the pipeline only
    # ever asks it for `-i`/`-ss`/`-t` and `ffprobe -show_entries`. Adding a
    # second name to this set is a decision about reproducibility, so it should
    # cost a red test first.
    assert {
        resource.id
        for resource in resources
        if isinstance(resource.asset, ResolvableAsset)
    } == {"ffmpeg"}

    for resource in resources:
        if isinstance(resource.asset, ResolvableAsset):
            # Unpinned still means verified: the digest is fetched from the
            # release API at install time and the download is checked against it.
            assert resource.asset.digest_from == "github-release-api"
            continue
        assert "/latest/" not in resource.asset.url
        assert resource.asset.size > 0
        assert resource.asset.sha256 != "0" * 64

    by_id = {resource.id: resource for resource in resources}
    # The required file is what proves an extraction produced the tool. yt-dlp
    # ships as a wheel, which is a zip of the importable package, so the marker
    # is a module rather than an executable.
    assert by_id["git"].required_files == ["cmd/git.exe"]
    assert by_id["yt-dlp"].required_files == ["yt_dlp/__init__.py"]
    # A bare .exe would be simpler to publish, but `archive_type: "file"` keeps
    # the downloaded name -- `tokcount-<version>.bin` -- so nothing could ask
    # for `tokcount.exe` afterwards. Publishing it zipped keeps the lookup.
    assert by_id["tokcount"].required_files == ["tokcount.exe"]


def _install_version(paths: AppPaths, version: str) -> None:
    """Put a complete ffmpeg of `version` on disk and point `current` at it."""

    directory = paths.runtime / "ffmpeg" / version / "bin"
    directory.mkdir(parents=True)
    (directory / "ffmpeg.exe").write_bytes(b"ffmpeg")
    (directory / "ffprobe.exe").write_bytes(b"ffprobe")
    (paths.runtime / "ffmpeg" / "current.json").write_text(
        json.dumps({"current": version}),
        encoding="utf-8",
    )


def test_an_older_installed_version_is_outdated_rather_than_missing(
    tmp_path: Path,
) -> None:
    # Collapsing the two made every manifest bump read as "you never installed
    # this", and -- once these tools became required -- stopped every user's
    # task until they fetched it.
    paths = AppPaths.for_root(tmp_path / "app-root")
    _install_version(paths, "7.0")
    manager = ResourceManager(
        paths, [_spec("https://invalid.example/ffmpeg.zip", b"unused")]
    )

    status = manager.status("ffmpeg")

    assert status.state == "outdated"
    assert status.installed_version == "7.0"
    # `version` stays the target, so the UI can show both sides of the upgrade.
    assert status.version == "7.1"
    assert status.usable is True


def test_a_pointer_to_a_version_that_is_gone_is_missing_not_outdated(
    tmp_path: Path,
) -> None:
    # Nothing to run: "outdated" would promise a working copy that is not there.
    paths = AppPaths.for_root(tmp_path / "app-root")
    _install_version(paths, "7.0")
    (paths.runtime / "ffmpeg" / "7.0" / "bin" / "ffmpeg.exe").unlink()
    manager = ResourceManager(
        paths, [_spec("https://invalid.example/ffmpeg.zip", b"unused")]
    )

    status = manager.status("ffmpeg")

    assert status.state == "missing"
    assert status.usable is False


def test_installing_over_an_outdated_copy_upgrades_it(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"new", "bin/ffprobe.exe": b"new"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    _install_version(paths, "7.0")
    manager = ResourceManager(paths, [_spec(server.url, body)])

    status = manager.install("ffmpeg", lambda event: None)

    assert status.state == "ready"
    assert manager.active_version("ffmpeg") == "7.1"


def test_a_denied_activation_says_what_is_holding_the_directory(
    serve_asset,
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The last step of a multi-minute install, and the one most likely to fail
    # for a reason unrelated to the work. `replace_path` has already waited out
    # the short holds by the time this raises, so the bare `[WinError 5]` that
    # reaches the front end is both final and unactionable -- it names neither
    # the resource nor anything the user can do about it.
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"ffmpeg", "bin/ffprobe.exe": b"ffprobe"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    manager = ResourceManager(paths, [_spec(server.url, body)])
    denial = PermissionError(5, "Access is denied")

    def deny(*_args, **_kwargs):
        raise denial

    monkeypatch.setattr(resources, "replace_path", deny)

    with pytest.raises(RuntimeError) as raised:
        manager.install("ffmpeg", lambda event: None)

    assert "ffmpeg" in str(raised.value)
    # The original stays reachable: the message is for the user, the errno for
    # whoever reads the log.
    assert raised.value.__cause__ is denial
    assert manager.active_version("ffmpeg") is None


def test_a_cleanup_that_also_fails_does_not_replace_the_diagnosis(
    serve_asset,
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The handle that denied the rename denies the delete for the same reason,
    # so the cleanup fails exactly when the interesting failure did. Letting
    # its error escape would hand the user a second, less useful diagnosis in
    # place of the first.
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"ffmpeg", "bin/ffprobe.exe": b"ffprobe"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    manager = ResourceManager(paths, [_spec(server.url, body)])
    activating = False
    real_remove_tree = resources.remove_tree

    def deny(*_args, **_kwargs):
        nonlocal activating
        activating = True
        raise PermissionError(5, "Access is denied")

    def remove_tree(path: Path) -> None:
        # Only once the install is past activation: the earlier calls clear the
        # staging directory, and are how it gets that far at all.
        if activating:
            raise PermissionError(5, "Access is denied")
        real_remove_tree(path)

    monkeypatch.setattr(resources, "replace_path", deny)
    monkeypatch.setattr(resources, "remove_tree", remove_tree)

    with pytest.raises(RuntimeError) as raised:
        manager.install("ffmpeg", lambda event: None)

    assert "无法启用资源" in str(raised.value)


def test_a_name_held_by_a_dangling_link_is_reported_as_taken(
    serve_asset,
    monkeypatch,
    tmp_path: Path,
) -> None:
    # `Path.exists` follows links, so a junction whose target is gone reads as
    # absent while still owning the name -- and a rename onto a taken name is
    # the access denial above, reported as a mystery instead of as the clear
    # "this name is in use" that this check exists to give.
    body = _zip_bytes(
        tmp_path / "ffmpeg.zip",
        {"bin/ffmpeg.exe": b"ffmpeg", "bin/ffprobe.exe": b"ffprobe"},
    )
    server = serve_asset(body)
    paths = AppPaths.for_root(tmp_path / "app-root")
    manager = ResourceManager(paths, [_spec(server.url, body)])
    final = paths.runtime / "ffmpeg" / "7.1"
    final.parent.mkdir(parents=True, exist_ok=True)
    _dangle(final, tmp_path / "gone")

    real_remove_tree = resources.remove_tree

    def remove_tree(path: Path) -> None:
        # Stands in for the name being retaken during the download: the install
        # clears `final` before fetching anything and looks again minutes
        # later, so the two are not the same moment.
        if path == final:
            return
        real_remove_tree(path)

    monkeypatch.setattr(resources, "remove_tree", remove_tree)

    with pytest.raises(FileExistsError):
        manager.install("ffmpeg", lambda event: None)
