from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _AssetReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        for name, value in attrs:
            if name in {"src", "href"} and value:
                self.references.append(value)


def validate_static_export(index_path: Path) -> None:
    index_path = index_path.resolve()
    export_root = index_path.parent
    parser = _AssetReferenceParser()
    parser.feed(index_path.read_text(encoding="utf-8"))

    for reference in parser.references:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if parsed.path.startswith("/"):
            raise ValueError(
                "Static export contains a root-relative asset that cannot "
                f"load from a file WebView: {reference}"
            )

        relative_path = Path(unquote(parsed.path))
        asset_path = (export_root / relative_path).resolve()
        try:
            asset_path.relative_to(export_root)
        except ValueError as error:
            raise ValueError(
                f"Static export asset escapes the export directory: {reference}"
            ) from error
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Static export asset does not exist: {asset_path}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    validate_static_export(args.index)
    print(f"Static WebView export verified: {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
