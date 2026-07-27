from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePath
import shutil
import subprocess
import time

from pydantic import BaseModel, ConfigDict, Field, field_validator

from desktop.backend.updates.installer import REQUIRED_APP_FILES


class FullUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    backup: str
    parent_pid: int = Field(ge=0)
    relaunch_path: str
    preserved: list[str] = Field(
        default_factory=lambda: [
            "app",
            "user-data",
            "models",
            "runtime",
            "cache",
        ]
    )

    @field_validator("preserved")
    @classmethod
    def validate_preserved(cls, values: list[str]) -> list[str]:
        for value in values:
            if (
                not value
                or value in {".", "..", ".update"}
                or "/" in value
                or "\\" in value
                or ":" in value
            ):
                raise ValueError("preserved directory names must be simple")
        return values

    @field_validator("relaunch_path")
    @classmethod
    def validate_relaunch_path(cls, value: str) -> str:
        path = PurePath(value.replace("\\", "/"))
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("relaunch path must stay inside the application root")
        return value


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the application root") from error
    return resolved


def wait_for_parent(parent_pid: int, timeout_seconds: float = 120.0) -> None:
    if parent_pid <= 0:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    raise TimeoutError(f"Parent process {parent_pid} did not exit")


def _merge_app_from_full_update(
    source_app: Path,
    target_app: Path,
) -> tuple[Path | None, bytes | None]:
    source_pointer = json.loads(
        (source_app / "current.json").read_text(encoding="utf-8")
    )
    version = source_pointer.get("current")
    if (
        not isinstance(version, str)
        or not version
        or version in {".", ".."}
        or "/" in version
        or "\\" in version
        or ":" in version
    ):
        raise ValueError("Full update App pointer contains an invalid version")
    source_version = source_app / "versions" / version
    missing = [
        relative
        for relative in REQUIRED_APP_FILES
        if not (source_version / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Full update App version is incomplete: {missing}"
        )
    app_manifest = json.loads(
        (source_version / "app-manifest.json").read_text(encoding="utf-8")
    )
    if (
        app_manifest.get("version") != version
        or app_manifest.get("platform") != "windows-x64"
    ):
        raise ValueError("Full update App metadata is inconsistent")

    target_versions = target_app / "versions"
    target_versions.mkdir(parents=True, exist_ok=True)
    target_version = target_versions / version
    created_version: Path | None = None
    if not target_version.exists():
        shutil.copytree(source_version, target_version)
        created_version = target_version

    target_pointer = target_app / "current.json"
    previous_bytes = target_pointer.read_bytes() if target_pointer.is_file() else None
    previous_version: str | None = None
    if previous_bytes is not None:
        try:
            previous_value = json.loads(previous_bytes).get("current")
            if isinstance(previous_value, str) and previous_value:
                previous_version = previous_value
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            previous_version = None
    temporary = target_pointer.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "current": version,
                "previous": previous_version,
                "pendingHealth": True,
                "healthAttempts": 0,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target_pointer)
    return created_version, previous_bytes


def _rollback_app_merge(
    target_app: Path,
    created_version: Path | None,
    previous_pointer: bytes | None,
) -> None:
    pointer = target_app / "current.json"
    if previous_pointer is None:
        if pointer.exists():
            pointer.unlink()
    else:
        temporary = pointer.with_suffix(".rollback.tmp")
        temporary.write_bytes(previous_pointer)
        os.replace(temporary, pointer)
    if created_version is not None and created_version.exists():
        shutil.rmtree(created_version)


def apply_full_update(
    request: FullUpdateRequest,
    *,
    relaunch: bool = True,
) -> None:
    target = Path(request.target).expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Application root does not exist: {target}")
    source = _inside(Path(request.source), target, label="source")
    backup = _inside(Path(request.backup), target, label="backup")
    update_root = target / ".update"
    if source == target or backup == target:
        raise ValueError("source and backup must not be the application root")
    if source == backup or source == backup.parent:
        raise ValueError("source and backup must be separate")
    if source.parent != update_root or backup.parent != update_root:
        raise ValueError("source and backup must be direct children of .update")
    if not source.is_dir():
        raise FileNotFoundError(f"Full update source does not exist: {source}")
    if backup.exists():
        raise FileExistsError(f"Full update backup already exists: {backup}")

    wait_for_parent(request.parent_pid)
    backup.mkdir(parents=True)
    preserved = set(request.preserved)
    moved: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    app_change: tuple[Path | None, bytes | None] | None = None
    try:
        for entry in list(target.iterdir()):
            if entry.name in preserved or entry.name == ".update":
                continue
            destination = backup / entry.name
            shutil.move(str(entry), str(destination))
            moved.append((entry, destination))
        for entry in source.iterdir():
            if entry.name == "app" and entry.name in preserved:
                app_change = _merge_app_from_full_update(
                    entry,
                    target / "app",
                )
                continue
            if entry.name in preserved or entry.name == ".update":
                continue
            destination = target / entry.name
            if entry.is_symlink():
                raise ValueError("Full update archives must not contain symlinks")
            if entry.is_dir():
                shutil.copytree(entry, destination)
            else:
                shutil.copy2(entry, destination)
            installed.append(destination)
    except Exception:
        if app_change is not None:
            _rollback_app_merge(target / "app", *app_change)
        for destination in reversed(installed):
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for original, saved in reversed(moved):
            if saved.exists():
                shutil.move(str(saved), str(original))
        raise

    if relaunch:
        executable = _inside(
            target / request.relaunch_path,
            target,
            label="relaunch path",
        )
        if not executable.is_file():
            raise FileNotFoundError(f"Relaunch executable does not exist: {executable}")
        subprocess.Popen(
            [str(executable)],
            cwd=str(target),
            close_fds=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineSub full application updater")
    parser.add_argument("--request", required=True, help="Path to the update request JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    request_path = Path(args.request).expanduser().resolve()
    request = FullUpdateRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    apply_full_update(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
