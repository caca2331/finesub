from __future__ import annotations

import os
from pathlib import Path
import struct
import shutil

from PIL import Image
import pytest

from desktop.scripts.prepare_brand_assets import (
    ICON_SIZES,
    REQUIRED_FONT_FILES,
    prepare_assets,
    prepare_icon,
    resolve_font_sources,
)


def _ico_sizes(path: Path) -> set[tuple[int, int]]:
    body = path.read_bytes()
    reserved, image_type, count = struct.unpack_from("<HHH", body)
    assert reserved == 0
    assert image_type == 1
    sizes: set[tuple[int, int]] = set()
    for index in range(count):
        width, height = struct.unpack_from("<BB", body, 6 + index * 16)
        sizes.add((width or 256, height or 256))
    return sizes


def test_required_font_sources_use_exact_maple_names(tmp_path: Path) -> None:
    for filename in REQUIRED_FONT_FILES.values():
        (tmp_path / filename).write_bytes(b"font")

    sources = resolve_font_sources(tmp_path)

    assert {weight: path.name for weight, path in sources.items()} == {
        "regular": "MapleMonoNL-NF-CN-Regular.ttf",
        "semibold": "MapleMonoNL-NF-CN-SemiBold.ttf",
        "bold": "MapleMonoNL-NF-CN-Bold.ttf",
    }


def test_missing_maple_weight_stops_asset_preparation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MapleMonoNL-NF-CN-Regular.ttf"):
        resolve_font_sources(tmp_path)


def test_icon_contains_every_required_windows_size(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (591, 591), "#173968").save(source)
    output = tmp_path / "app.ico"

    prepare_icon(source, output)

    assert _ico_sizes(output) == {(size, size) for size in ICON_SIZES}


def test_prepare_assets_emits_portable_fonts_icon_and_license(
    tmp_path: Path,
) -> None:
    windows_fonts = Path(os.environ["WINDIR"]) / "Fonts"
    source_font = next(windows_fonts.glob("*.ttf"))
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    for filename in REQUIRED_FONT_FILES.values():
        shutil.copy2(source_font, font_dir / filename)
    source_image = tmp_path / "brand.png"
    Image.new("RGB", (591, 591), "#173968").save(source_image)
    license_source = tmp_path / "OFL.txt"
    license_source.write_text("SIL OPEN FONT LICENSE Version 1.1\n", "utf-8")
    repo_root = tmp_path / "repo"

    result = prepare_assets(
        source_image=source_image,
        font_dir=font_dir,
        repo_root=repo_root,
        license_source=license_source,
    )

    assert result.icon.read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert result.favicon.read_bytes().startswith(b"\x89PNG")
    assert {path.read_bytes()[:4] for path in result.fonts.values()} == {b"wOF2"}
    assert result.license_file.read_text("utf-8").startswith(
        "SIL OPEN FONT LICENSE"
    )
    assert result.source_image.read_bytes() == source_image.read_bytes()
