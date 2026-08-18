from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from finesub_bootstrap.archive import UnsafeArchivePath, safe_extract_zip


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return path


@pytest.mark.parametrize(
    "member",
    ["../escape.txt", "/absolute.txt", "C:/escape.txt", "bin/../../escape.txt"],
)
def test_zip_rejects_paths_outside_destination(
    member: str,
    tmp_path: Path,
) -> None:
    archive = _make_zip(tmp_path / "bad.zip", {member: b"x"})

    with pytest.raises(UnsafeArchivePath):
        safe_extract_zip(archive, tmp_path / "out")


def test_zip_extracts_normal_members(tmp_path: Path) -> None:
    archive = _make_zip(
        tmp_path / "ok.zip",
        {"bin/tool.exe": b"tool", "share/readme.txt": b"readme"},
    )

    files = safe_extract_zip(archive, tmp_path / "out")

    assert set(files) == {
        tmp_path / "out" / "bin" / "tool.exe",
        tmp_path / "out" / "share" / "readme.txt",
    }
    assert files[0].is_file()


def test_ntfs_stream_and_device_names_are_rejected(tmp_path: Path) -> None:
    """`:` was only checked on the first component, to catch a drive letter.

    At any depth it opens an NTFS alternate data stream instead:
    `sub/pyproject.toml:evil` writes into a stream of `sub/pyproject.toml` and
    creates that file as a 0-byte husk on the way -- which satisfies every
    `is_file()` completeness check downstream while holding nothing. `NUL`
    swallows its content outright.
    """
    import zipfile

    from finesub_bootstrap.archive import UnsafeArchivePath, safe_extract_zip

    for member in ("sub/pyproject.toml:evil", "sub/NUL", "sub/nul.txt", "a/CON"):
        archive = tmp_path / "payload.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, "x")
        with pytest.raises(UnsafeArchivePath):
            safe_extract_zip(archive, tmp_path / "out")
        archive.unlink()


def test_ordinary_members_still_extract(tmp_path: Path) -> None:
    import zipfile

    from finesub_bootstrap.archive import safe_extract_zip

    archive = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sub/pyproject.toml", "[project]")
        handle.writestr("sub/nested/deep.txt", "ok")

    extracted = safe_extract_zip(archive, tmp_path / "out")

    assert (tmp_path / "out" / "sub" / "pyproject.toml").read_text() == "[project]"
    assert (tmp_path / "out" / "sub" / "nested" / "deep.txt").read_text() == "ok"
    assert extracted
