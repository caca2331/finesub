from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile, ZipInfo


class UnsafeArchivePath(ValueError):
    pass


#: Names Windows reserves for devices, at any directory depth and with any
#: extension. Writing to one goes to the device rather than to a file: `NUL`
#: swallows its content silently, so a member named that way is either a
#: mistake or an attempt to make a file "exist" while holding nothing.
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)


def _is_windows_device_name(part: str) -> bool:
    return part.split(".", 1)[0].strip().lower() in _WINDOWS_DEVICE_NAMES


def _validated_member(info: ZipInfo) -> PurePosixPath:
    normalized = info.filename.replace("\\", "/")
    member = PurePosixPath(normalized)
    unix_mode = info.external_attr >> 16
    # `:` was checked on the first component only, to catch a drive letter.
    # On NTFS it also opens an alternate data stream at *any* depth:
    # `sub/pyproject.toml:evil` writes into a stream of `sub/pyproject.toml`
    # and creates that file as a 0-byte husk on the way -- enough to satisfy
    # every `is_file()` completeness check downstream while carrying nothing.
    if (
        not member.parts
        or member.is_absolute()
        or normalized.startswith("/")
        or ".." in member.parts
        or any(":" in part for part in member.parts)
        or any(_is_windows_device_name(part) for part in member.parts)
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
