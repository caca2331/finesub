from __future__ import annotations

from pathlib import Path
import struct

from PIL import Image

from desktop.scripts.prepare_brand_assets import (
    ICON_SIZES,
    prepare_assets,
    prepare_icon,
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


def test_icon_contains_every_required_windows_size(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (591, 591), "#173968").save(source)
    output = tmp_path / "app.ico"

    prepare_icon(source, output)

    assert _ico_sizes(output) == {(size, size) for size in ICON_SIZES}


def test_prepare_assets_emits_icon_and_frontend_image(
    tmp_path: Path,
) -> None:
    source_image = tmp_path / "brand.png"
    Image.new("RGB", (591, 591), "#173968").save(source_image)
    repo_root = tmp_path / "repo"

    result = prepare_assets(
        source_image=source_image,
        repo_root=repo_root,
    )

    assert result.icon.read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert result.favicon.read_bytes().startswith(b"\x89PNG")
    assert result.source_image.read_bytes() == source_image.read_bytes()
