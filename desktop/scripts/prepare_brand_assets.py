from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

from fontTools.ttLib import TTFont
from PIL import Image


REQUIRED_FONT_FILES = {
    "regular": "MapleMonoNL-NF-CN-Regular.ttf",
    "semibold": "MapleMonoNL-NF-CN-SemiBold.ttf",
    "bold": "MapleMonoNL-NF-CN-Bold.ttf",
}
OUTPUT_FONT_FILES = {
    weight: f"maple-mono-nl-nf-cn-{weight}.woff2"
    for weight in REQUIRED_FONT_FILES
}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


@dataclass(frozen=True, slots=True)
class PreparedAssets:
    source_image: Path
    icon: Path
    favicon: Path
    fonts: dict[str, Path]
    license_file: Path


def resolve_font_sources(font_dir: Path) -> dict[str, Path]:
    resolved = font_dir.expanduser().resolve()
    sources = {
        weight: resolved / filename
        for weight, filename in REQUIRED_FONT_FILES.items()
    }
    missing = [path.name for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required Maple Mono font files are missing: {', '.join(missing)}"
        )
    return sources


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


def convert_font(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    font = TTFont(source, recalcTimestamp=False)
    try:
        font.flavor = "woff2"
        font.save(output, reorderTables=True)
    finally:
        font.close()


def prepare_assets(
    *,
    source_image: Path,
    font_dir: Path,
    repo_root: Path,
    license_source: Path,
) -> PreparedAssets:
    source_image = source_image.expanduser().resolve()
    license_source = license_source.expanduser().resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Brand image not found: {source_image}")
    if not license_source.is_file():
        raise FileNotFoundError(f"Maple Mono license not found: {license_source}")

    root = repo_root.expanduser().resolve()
    source_target = root / "desktop" / "assets" / "source" / "finesub-desktop.png"
    icon_target = root / "desktop" / "assets" / "finesub-desktop.ico"
    asset_license = (
        root / "desktop" / "assets" / "licenses" / "Maple-Mono-OFL.txt"
    )
    public_root = root / "desktop" / "frontend" / "public"
    favicon_target = public_root / "icon.png"
    font_output = public_root / "fonts"
    public_license = font_output / "OFL.txt"

    source_target.parent.mkdir(parents=True, exist_ok=True)
    favicon_target.parent.mkdir(parents=True, exist_ok=True)
    font_output.mkdir(parents=True, exist_ok=True)
    asset_license.parent.mkdir(parents=True, exist_ok=True)

    if source_image != source_target:
        shutil.copy2(source_image, source_target)
    shutil.copy2(source_target, favicon_target)
    prepare_icon(source_target, icon_target)

    fonts = {}
    for weight, source in resolve_font_sources(font_dir).items():
        target = font_output / OUTPUT_FONT_FILES[weight]
        convert_font(source, target)
        fonts[weight] = target

    if license_source != asset_license:
        shutil.copy2(license_source, asset_license)
    shutil.copy2(asset_license, public_license)
    return PreparedAssets(
        source_image=source_target,
        icon=icon_target,
        favicon=favicon_target,
        fonts=fonts,
        license_file=public_license,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FineSub Desktop icon and embedded Maple Mono fonts."
    )
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--font-directory", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--license-source", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    license_source = args.license_source or (
        args.repo_root
        / "desktop"
        / "assets"
        / "licenses"
        / "Maple-Mono-OFL.txt"
    )
    result = prepare_assets(
        source_image=args.source_image,
        font_dir=args.font_directory,
        repo_root=args.repo_root,
        license_source=license_source,
    )
    print(f"FineSub Desktop icon: {result.icon}")
    for weight, path in result.fonts.items():
        print(f"Maple Mono {weight}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
