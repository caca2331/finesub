from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

from PIL import Image


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


@dataclass(frozen=True, slots=True)
class PreparedAssets:
    source_image: Path
    icon: Path
    favicon: Path


def prepare_icon(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        square = image.convert("RGBA")
        if square.width != square.height:
            edge = min(square.size)
            left = (square.width - edge) // 2
            top = (square.height - edge) // 2
            square = square.crop((left, top, left + edge, top + edge))
        square.save(
            output,
            format="ICO",
            sizes=[(size, size) for size in ICON_SIZES],
            bitmap_format="png",
        )


def prepare_assets(
    *,
    source_image: Path,
    repo_root: Path,
) -> PreparedAssets:
    source_image = source_image.expanduser().resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Brand image not found: {source_image}")

    root = repo_root.expanduser().resolve()
    source_target = root / "desktop" / "assets" / "source" / "finesub-desktop.png"
    icon_target = root / "desktop" / "assets" / "finesub-desktop.ico"
    public_root = root / "desktop" / "frontend" / "public"
    favicon_target = public_root / "icon.png"

    source_target.parent.mkdir(parents=True, exist_ok=True)
    favicon_target.parent.mkdir(parents=True, exist_ok=True)

    if source_image != source_target:
        shutil.copy2(source_image, source_target)
    shutil.copy2(source_target, favicon_target)
    prepare_icon(source_target, icon_target)

    return PreparedAssets(
        source_image=source_target,
        icon=icon_target,
        favicon=favicon_target,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FineSub Desktop icon assets."
    )
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_assets(
        source_image=args.source_image,
        repo_root=args.repo_root,
    )
    print(f"FineSub Desktop icon: {result.icon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
