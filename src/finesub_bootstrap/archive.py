from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile, ZipInfo


class UnsafeArchivePath(ValueError):
    pass


def _validated_member(info: ZipInfo) -> PurePosixPath:
    normalized = info.filename.replace("\\", "/")
    member = PurePosixPath(normalized)
    first = member.parts[0] if member.parts else ""
    unix_mode = info.external_attr >> 16
    if (
        not member.parts
        or member.is_absolute()
        or normalized.startswith("/")
        or ".." in member.parts
        or ":" in first
        or stat.S_ISLNK(unix_mode)
    ):
        raise UnsafeArchivePath(f"Unsafe ZIP member: {info.filename}")
    return member


def safe_extract_zip(archive_path: Path, destination: Path) -> list[Path]:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with ZipFile(archive_path) as archive:
        validated = [(info, _validated_member(info)) for info in archive.infolist()]
        for info, member in validated:
            target = destination.joinpath(*member.parts)
            try:
                target.resolve().relative_to(destination)
            except ValueError as error:
                raise UnsafeArchivePath(
                    f"ZIP member leaves destination: {info.filename}"
                ) from error
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)

    return extracted
