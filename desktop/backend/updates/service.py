from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal

import httpx
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field

from desktop.backend.common.models import DownloadProgress
from desktop.backend.common.paths import AppPaths
from desktop.backend.common.product import (
    MAIN_EXECUTABLE_NAME,
    UPDATER_EXECUTABLE_NAME,
)
from desktop.backend.resources.archive import safe_extract_zip
from desktop.backend.resources.downloader import download_asset
from desktop.backend.common.http_client import (
    connection_error,
    create_client,
    is_connection_failure,
    network_routes,
)
from desktop.backend.updater_main import FullUpdateRequest
from desktop.backend.updates.installer import AppInstaller, REQUIRED_APP_FILES
from desktop.backend.updates.manifest import (
    LocalUpdateState,
    UpdateManifest,
    select_asset,
    verify_manifest,
)


ReleaseFetcher = Callable[[str, Literal["stable", "beta"]], dict[str, Any]]
BytesFetcher = Callable[[str, int], bytes]
AssetDownloader = Callable[[Any, Path, Callable[[DownloadProgress], None]], Path]
ProcessLauncher = Callable[[list[str]], Any]


class LauncherUpdateConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_version: int = Field(default=1, alias="schemaVersion")
    app_version: str = Field(alias="appVersion")
    launcher_version: str = Field(alias="launcherVersion")
    channel: Literal["stable", "beta"] = "stable"
    platform: Literal["windows-x64"] = "windows-x64"
    release_repository: str = Field(alias="releaseRepository")
    auto_check_updates: bool = Field(default=True, alias="autoCheckUpdates")
    auto_download_app_updates: bool = Field(
        default=True,
        alias="autoDownloadAppUpdates",
    )


class GitHubUpdateService:
    def __init__(
        self,
        *,
        paths: AppPaths,
        config: LauncherUpdateConfig,
        trusted_keys: dict[str, str],
        release_fetcher: ReleaseFetcher | None = None,
        bytes_fetcher: BytesFetcher | None = None,
        asset_downloader: AssetDownloader = download_asset,
        process_launcher: ProcessLauncher | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.trusted_keys = dict(trusted_keys)
        self.release_fetcher = release_fetcher or _fetch_release
        self.bytes_fetcher = bytes_fetcher or _fetch_bytes
        self.asset_downloader = asset_downloader
        self.process_launcher = process_launcher or _launch_process
        self.app_installer = AppInstaller(paths)
        self._manifest: UpdateManifest | None = None
        self._kind: Literal["app", "full"] | None = None

    def check(self) -> dict[str, Any]:
        release = self.release_fetcher(
            self.config.release_repository,
            self.config.channel,
        )
        if release.get("draft"):
            raise ValueError("GitHub draft releases cannot be installed")
        assets = {
            str(asset.get("name")): str(asset.get("browser_download_url"))
            for asset in release.get("assets", [])
            if isinstance(asset, dict)
        }
        try:
            manifest_url = assets["update-manifest.json"]
            signature_url = assets["update-manifest.sig"]
        except KeyError as error:
            raise ValueError(
                "GitHub Release is missing the signed update manifest"
            ) from error
        manifest_bytes = self.bytes_fetcher(manifest_url, 1024 * 1024)
        signature_bytes = self.bytes_fetcher(signature_url, 4096)
        manifest = verify_manifest(
            manifest_bytes,
            signature_bytes,
            self.trusted_keys,
            expected_channel=self.config.channel,
            expected_platform=self.config.platform,
        )
        if release.get("tag_name") != f"v{manifest.version}":
            raise ValueError("GitHub Release tag does not match the signed manifest")
        if bool(release.get("prerelease")) != manifest.prerelease:
            raise ValueError(
                "GitHub prerelease metadata does not match the signed manifest"
            )
        local = self._local_state()
        if Version(manifest.version) <= Version(local.version):
            self._manifest = None
            self._kind = None
            return {
                "available": False,
                "version": local.version,
            }
        kind = select_asset(manifest, local)
        self._manifest = manifest
        self._kind = kind
        asset = getattr(manifest.assets, kind)
        return {
            "available": True,
            "version": manifest.version,
            "kind": kind,
            "releaseNotes": manifest.release_notes,
            "mandatory": manifest.mandatory,
            "size": asset.size,
        }

    def install(self, kind: Literal["app", "full"]) -> dict[str, Any]:
        if self._manifest is None or self._kind is None:
            status = self.check()
            if not status.get("available"):
                raise ValueError("No newer FineSub release is available")
        assert self._manifest is not None
        assert self._kind is not None
        if kind != self._kind:
            raise ValueError(
                f"Signed update requires {self._kind!r}, not {kind!r}"
            )
        if kind == "app":
            return self._install_app(self._manifest)
        return self._install_full(self._manifest)

    def _install_app(self, manifest: UpdateManifest) -> dict[str, Any]:
        destination = (
            self.paths.root
            / ".update"
            / "downloads"
            / f"finesub-app-{manifest.version}.zip"
        )
        archive = self.asset_downloader(
            manifest.assets.app,
            destination,
            lambda event: None,
        )
        pending = self.app_installer.install(archive, manifest)
        return {
            "kind": "app",
            "version": pending.version,
            "restartRequired": True,
        }

    def _install_full(self, manifest: UpdateManifest) -> dict[str, Any]:
        update_root = self.paths.root / ".update"
        archive = self.asset_downloader(
            manifest.assets.full,
            update_root
            / "downloads"
            / f"finesub-full-{manifest.version}.zip",
            lambda event: None,
        )
        source = update_root / f"source-{manifest.version}"
        backup = update_root / f"backup-{manifest.version}"
        runner = update_root / f"runner-{manifest.version}"
        for directory in (source, backup, runner):
            if directory.exists():
                shutil.rmtree(directory)
        source.mkdir(parents=True, exist_ok=True)
        safe_extract_zip(archive, source)
        if not (source / MAIN_EXECUTABLE_NAME).is_file():
            raise FileNotFoundError(
                f"Full update does not contain {MAIN_EXECUTABLE_NAME}"
            )
        packaged_updater = source / "updater" / UPDATER_EXECUTABLE_NAME
        if not packaged_updater.is_file():
            raise FileNotFoundError(
                "Full update does not contain "
                f"updater/{UPDATER_EXECUTABLE_NAME}"
            )
        full_pointer = source / "app" / "current.json"
        if not full_pointer.is_file():
            raise FileNotFoundError(
                "Full update does not contain a versioned App"
            )
        try:
            full_current = json.loads(
                full_pointer.read_text(encoding="utf-8")
            ).get("current")
        except (OSError, ValueError, AttributeError) as error:
            raise ValueError("Full update App pointer is malformed") from error
        if not isinstance(full_current, str) or not full_current:
            raise ValueError("Full update App pointer has no current version")
        full_app = source / "app" / "versions" / full_current
        missing_app = [
            relative
            for relative in REQUIRED_APP_FILES
            if not (full_app / relative).is_file()
        ]
        if missing_app:
            raise FileNotFoundError(
                f"Full update App is incomplete: {missing_app}"
            )

        installed_updater = self.paths.root / "updater"
        if not (installed_updater / UPDATER_EXECUTABLE_NAME).is_file():
            raise FileNotFoundError("Installed updater runtime is missing")
        shutil.copytree(installed_updater, runner)
        runner_executable = runner / UPDATER_EXECUTABLE_NAME
        request = FullUpdateRequest(
            source=str(source),
            target=str(self.paths.root),
            backup=str(backup),
            parent_pid=os.getpid(),
            relaunch_path=MAIN_EXECUTABLE_NAME,
        )
        request_path = update_root / f"request-{manifest.version}.json"
        request_path.write_text(
            request.model_dump_json(),
            encoding="utf-8",
            newline="\n",
        )
        self.process_launcher(
            [str(runner_executable), "--request", str(request_path)]
        )
        return {
            "kind": "full",
            "version": manifest.version,
            "exitRequired": True,
        }

    def _local_state(self) -> LocalUpdateState:
        version = self.config.app_version
        if self.paths.app_current.is_file():
            try:
                pointer = json.loads(
                    self.paths.app_current.read_text(encoding="utf-8")
                )
                current = pointer.get("current")
                if isinstance(current, str) and current:
                    version = current
            except (OSError, ValueError, AttributeError):
                pass
        return LocalUpdateState(
            version=version,
            launcher_version=self.config.launcher_version,
            channel=self.config.channel,
            platform=self.config.platform,
        )


def _fetch_release(
    repository: str,
    channel: Literal["stable", "beta"],
) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "FineSub-Desktop-Updater",
    }
    timeout = httpx.Timeout(connect=20.0, read=30.0, write=20.0, pool=20.0)
    attempts: list[tuple[str, BaseException]] = []
    for route in network_routes():
        try:
            with create_client(route, timeout=timeout, headers=headers) as client:
                if channel == "stable":
                    response = client.get(
                        f"https://api.github.com/repos/{repository}/releases/latest"
                    )
                    response.raise_for_status()
                    body = response.json()
                    if not isinstance(body, dict):
                        raise ValueError("GitHub latest release response is malformed")
                    return body
                response = client.get(
                    f"https://api.github.com/repos/{repository}/releases",
                    params={"per_page": 20},
                )
                response.raise_for_status()
                releases = response.json()
                if not isinstance(releases, list):
                    raise ValueError("GitHub releases response is malformed")
                for release in releases:
                    if (
                        isinstance(release, dict)
                        and not release.get("draft")
                        and release.get("prerelease")
                    ):
                        return release
        except Exception as error:
            if not is_connection_failure(error):
                raise
            attempts.append((route.label, error))
    if attempts and len(attempts) == len(network_routes()):
        raise connection_error(attempts)
    raise ValueError("No eligible beta release was found")


def _fetch_bytes(url: str, limit: int) -> bytes:
    timeout = httpx.Timeout(connect=20.0, read=30.0, write=20.0, pool=20.0)
    attempts: list[tuple[str, BaseException]] = []
    for route in network_routes():
        try:
            with create_client(route, timeout=timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                body = response.content
            break
        except Exception as error:
            if not is_connection_failure(error):
                raise
            attempts.append((route.label, error))
    else:
        raise connection_error(attempts)
    if len(body) > limit:
        raise ValueError(f"Update metadata exceeds {limit} bytes")
    return body


def _launch_process(command: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=str(Path(command[0]).parent),
        close_fds=True,
    )
