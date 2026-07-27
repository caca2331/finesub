from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

from pydantic import BaseModel, ConfigDict

from desktop.backend.common.paths import AppPaths
from desktop.backend.resources.archive import safe_extract_zip
from desktop.backend.updates.manifest import UpdateManifest


REQUIRED_APP_FILES = (
    "src/pipeline.py",
    "desktop/backend/worker/main.py",
    "desktop/frontend/out/index.html",
    "pyproject.toml",
    "app-manifest.json",
)


class PendingSwitch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    previous: str | None
    pending_health: bool


class AppInstaller:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.pointer_path = paths.root / "app" / "current.json"

    def install(
        self,
        archive_path: Path,
        manifest: UpdateManifest,
    ) -> PendingSwitch:
        archive_path = archive_path.expanduser().resolve()
        self._verify_archive(archive_path, manifest)
        self.paths.app_versions.mkdir(parents=True, exist_ok=True)
        staging = self.paths.app_versions / f"{manifest.version}.staging"
        final = self.paths.app_versions / manifest.version
        if staging.exists():
            shutil.rmtree(staging)
        if final.exists():
            raise FileExistsError(f"App version already exists: {final}")
        staging.mkdir(parents=True)
        try:
            safe_extract_zip(archive_path, staging)
            self._validate_app(staging, manifest)
            os.replace(staging, final)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        current = self.read_pointer() if self.pointer_path.is_file() else {}
        previous = current.get("current")
        self.write_pointer(
            current=manifest.version,
            previous=previous if isinstance(previous, str) else None,
            pending_health=True,
        )
        return PendingSwitch(
            version=manifest.version,
            previous=previous if isinstance(previous, str) else None,
            pending_health=True,
        )

    def confirm_health(self, version: str) -> None:
        pointer = self.read_pointer()
        if pointer.get("current") != version:
            raise ValueError("Cannot confirm a version that is not current")
        version_dir = self.paths.app_versions / version
        if not all((version_dir / relative).is_file() for relative in REQUIRED_APP_FILES):
            raise FileNotFoundError(f"App version {version} is incomplete")
        self.write_pointer(
            current=version,
            previous=(
                pointer.get("previous")
                if isinstance(pointer.get("previous"), str)
                else None
            ),
            pending_health=False,
            health_attempts=0,
        )

    def prepare_startup(self) -> str | None:
        if not self.pointer_path.is_file():
            return None
        pointer = self.read_pointer()
        current = pointer.get("current")
        if not isinstance(current, str) or not current:
            return None
        if not pointer.get("pendingHealth"):
            return current
        attempts = pointer.get("healthAttempts", 0)
        if not isinstance(attempts, int) or attempts < 0:
            attempts = 0
        if attempts >= 1:
            return self.rollback_failed_start() or current
        previous = pointer.get("previous")
        self.write_pointer(
            current=current,
            previous=previous if isinstance(previous, str) else None,
            pending_health=True,
            health_attempts=1,
        )
        return current

    def rollback_failed_start(self) -> str | None:
        if not self.pointer_path.is_file():
            return None
        pointer = self.read_pointer()
        if not pointer.get("pendingHealth"):
            return None
        current = pointer.get("current")
        previous = pointer.get("previous")
        if not isinstance(previous, str) or not previous:
            return None
        self.write_pointer(
            current=previous,
            previous=current if isinstance(current, str) else None,
            pending_health=False,
            health_attempts=0,
        )
        return previous

    def read_pointer(self) -> dict[str, object]:
        body = json.loads(self.pointer_path.read_text(encoding="utf-8-sig"))
        if not isinstance(body, dict):
            raise ValueError("App pointer must be a JSON object")
        return body

    def write_pointer(
        self,
        *,
        current: str,
        previous: str | None,
        pending_health: bool,
        health_attempts: int = 0,
    ) -> None:
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.pointer_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "current": current,
                    "previous": previous,
                    "pendingHealth": pending_health,
                    "healthAttempts": health_attempts,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temp, self.pointer_path)

    @staticmethod
    def _verify_archive(archive_path: Path, manifest: UpdateManifest) -> None:
        asset = manifest.assets.app
        if archive_path.stat().st_size != asset.size:
            raise ValueError("App update archive size does not match the manifest")
        digest = hashlib.sha256()
        with archive_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset.sha256:
            raise ValueError("App update archive SHA-256 does not match the manifest")

    @staticmethod
    def _validate_app(staging: Path, manifest: UpdateManifest) -> None:
        missing = [
            relative
            for relative in REQUIRED_APP_FILES
            if not (staging / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"App update is missing required files: {missing}")
        app_manifest = json.loads(
            (staging / "app-manifest.json").read_text(encoding="utf-8")
        )
        if (
            app_manifest.get("version") != manifest.version
            or app_manifest.get("platform") != manifest.platform
        ):
            raise ValueError("App archive metadata does not match update manifest")
