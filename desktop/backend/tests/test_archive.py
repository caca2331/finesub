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
