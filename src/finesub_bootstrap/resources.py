from __future__ import annotations

from collections.abc import Callable, Iterable
import json
import os
from pathlib import Path
import shutil

from finesub_bootstrap.asset_resolve import resolve_asset
from finesub_bootstrap.models import (
    DownloadProgress,
    ResourceSpec,
    ResourceStatus,
)
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.archive import safe_extract_zip
from finesub_bootstrap.downloader import DownloadPaused, download_asset
from finesub_bootstrap.fsops import remove_tree, replace_path, write_atomic


StageCallback = Callable[[str, str], None]
PauseCheck = Callable[[], bool]

#: The runtime manifest that ships *inside this package*. It used to live at
#: `desktop/resources/runtime-manifest.json` and be reached by path from three
#: places at once (the launcher, `package_shell`, the CLI's vendored tree),
#: which is what made it hard to move. Callers that provision a specific app
#: snapshot can still pass a path -- the desktop launcher was frozen separately
#: from the app source it installed, so `__file__` there named the wrong tree.
PACKAGED_RUNTIME_MANIFEST = Path(__file__).with_name("runtime-manifest.json")


def read_runtime_manifest(path: Path | None = None) -> dict:
    """The parsed runtime manifest; the packaged one unless told otherwise."""

    manifest = PACKAGED_RUNTIME_MANIFEST if path is None else path
    return json.loads(manifest.read_text(encoding="utf-8"))


def _activation_failure_message(resource_id: str, error: OSError) -> str:
    """Name who is likely holding the directory, and what to do about it.

    This is the last step of a multi-minute install, and the front ends show
    what it raises: the bare `[WinError 5] 拒绝访问` it replaces named neither
    the resource nor anything the user could act on. `replace_path` has already
    waited out the short holds by the time this runs, so what is left is
    something that is not letting go on its own.
    """

    return (
        f"无法启用资源 {resource_id}：目标目录被占用。"
        "常见原因是杀毒软件或网盘同步正在扫描刚解压出来的文件，"
        "或有资源管理器/终端停在该目录里。"
        "请关掉它们后重试；若安装目录在网盘同步目录或网络盘上，"
        f"请改装到本地普通目录。（{error}）"
    )


class ResourceManager:
    def __init__(self, paths: AppPaths, resources: Iterable[ResourceSpec]) -> None:
        self.paths = paths
        self.resources = {resource.id: resource for resource in resources}

    def check_all(self) -> list[ResourceStatus]:
        return [self.status(resource_id) for resource_id in self.resources]

    def status(self, resource_id: str) -> ResourceStatus:
        spec = self._spec(resource_id)
        active = self.active_version(resource_id)
        installed = (
            active
            if active is not None
            and self._required_files_exist(self._version_dir(spec, active), spec)
            else None
        )
        if installed == spec.version:
            return ResourceStatus(
                id=resource_id,
                version=spec.version,
                state="ready",
            )
        if installed is not None:
            # A usable copy of the wrong version. Reported apart from "missing"
            # so consumers can offer the upgrade without blocking on it.
            return ResourceStatus(
                id=resource_id,
                version=spec.version,
                installed_version=installed,
                state="outdated",
            )
        return ResourceStatus(
            id=resource_id,
            version=spec.version,
            state="missing",
        )

    def active_version(self, resource_id: str) -> str | None:
        spec = self._spec(resource_id)
        pointer = self._resource_root(spec) / "current.json"
        if not pointer.is_file():
            return None
        try:
            value = json.loads(pointer.read_text(encoding="utf-8")).get("current")
        except (OSError, ValueError, AttributeError):
            return None
        return value if isinstance(value, str) and value else None

    def active_file(self, resource_id: str, filename: str) -> Path | None:
        spec = self._spec(resource_id)
        active = self.active_version(resource_id)
        if active is None:
            return None
        matching = [
            required
            for required in spec.required_files
            if Path(required).name.casefold() == filename.casefold()
        ]
        if len(matching) != 1:
            return None
        candidate = self._version_dir(spec, active) / Path(matching[0])
        return candidate if candidate.is_file() else None

    def install(
        self,
        resource_id: str,
        progress: Callable[[DownloadProgress], None],
        *,
        stage: StageCallback | None = None,
        should_pause: PauseCheck | None = None,
    ) -> ResourceStatus:
        spec = self._spec(resource_id)
        if self.status(resource_id).state == "ready":
            return self.status(resource_id)

        root = self._resource_root(spec)
        final = self._version_dir(spec, spec.version)
        if self._required_files_exist(final, spec):
            self._write_pointer(root / "current.json", spec.version)
            return ResourceStatus(
                id=spec.id,
                version=spec.version,
                state="ready",
            )
        remove_tree(final)

        downloads = self.paths.cache / "downloads"
        extension = ".zip" if spec.archive_type == "zip" else ".bin"
        archive_path = downloads / f"{spec.id}-{spec.version}{extension}"
        # A resolvable asset becomes an ordinary pinned one here, before anything
        # touches the network for the file itself: everything below -- resume
        # bounds, progress totals, the size and digest checks -- only ever sees a
        # pinned asset. `resolve_asset` is a no-op for the four that are pinned in
        # the manifest, and for the one that is not it is a single sub-second API
        # call, which is why it gets no stage of its own.
        asset = resolve_asset(spec.asset)
        if stage is not None:
            stage("downloading", "正在下载资源文件")
        downloaded = download_asset(
            asset,
            archive_path,
            progress,
            should_pause=should_pause,
        )
        if should_pause is not None and should_pause():
            raise DownloadPaused("Resource installation paused")
        if stage is not None:
            stage("verifying", "正在校验文件完整性")

        staging = root / f"{spec.version}.staging"
        remove_tree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            if spec.archive_type == "zip":
                if stage is not None:
                    stage("extracting", "正在解压资源")
                safe_extract_zip(downloaded, staging)
            else:
                shutil.copy2(downloaded, staging / downloaded.name)
            missing = [
                required
                for required in spec.required_files
                if not (staging / Path(required)).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Resource {spec.id} is missing required files: {missing}"
                )
            root.mkdir(parents=True, exist_ok=True)
            # lexists, not exists: the question is whether the *name* is taken,
            # and `Path.exists` follows links, so a junction whose target is
            # gone reads as absent while still owning the name. The rename
            # below would then fail as an access denial -- a mystery, where
            # this raise says exactly what is in the way. The window is narrow
            # (the download sits between this and the `remove_tree(final)`
            # above, which already unlinks such a junction), which is why the
            # check is worth keeping honest rather than dropping.
            if os.path.lexists(final):
                raise FileExistsError(
                    f"Resource version directory already exists: {final}"
                )
            if should_pause is not None and should_pause():
                raise DownloadPaused("Resource installation paused")
            if stage is not None:
                stage("activating", "正在启用资源")
            try:
                replace_path(staging, final)
            except OSError as error:
                raise RuntimeError(
                    _activation_failure_message(spec.id, error)
                ) from error
            self._write_pointer(root / "current.json", spec.version)
        except Exception:
            # Best effort. What lands here is often a handle someone else holds
            # inside `staging`, and on Windows that denies the delete for the
            # same reason it denied the rename -- so raising from the cleanup
            # would replace the real diagnosis with a second, less useful one.
            # The next install clears this directory before using it, so
            # leaving it behind costs disk, not correctness.
            try:
                remove_tree(staging)
            except OSError:
                pass
            raise

        return ResourceStatus(
            id=spec.id,
            version=spec.version,
            state="ready",
        )

    def cache_path(self, resource_id: str) -> Path:
        spec = self._spec(resource_id)
        extension = ".zip" if spec.archive_type == "zip" else ".bin"
        return self.paths.cache / "downloads" / (
            f"{spec.id}-{spec.version}{extension}"
        )

    def install_path(self, resource_id: str) -> Path:
        spec = self._spec(resource_id)
        return self._version_dir(spec, spec.version)

    def _spec(self, resource_id: str) -> ResourceSpec:
        try:
            return self.resources[resource_id]
        except KeyError as error:
            raise KeyError(f"Unknown resource: {resource_id}") from error

    def _resource_root(self, spec: ResourceSpec) -> Path:
        base = self.paths.runtime if spec.destination == "runtime" else self.paths.models
        return base / spec.directory

    def _version_dir(self, spec: ResourceSpec, version: str) -> Path:
        return self._resource_root(spec) / version

    @staticmethod
    def _required_files_exist(version_dir: Path, spec: ResourceSpec) -> bool:
        return version_dir.is_dir() and all(
            (version_dir / Path(required)).is_file()
            for required in spec.required_files
        )

    @staticmethod
    def _write_pointer(pointer: Path, version: str) -> None:
        write_atomic(
            pointer,
            json.dumps({"current": version}, ensure_ascii=False, separators=(",", ":")),
        )
