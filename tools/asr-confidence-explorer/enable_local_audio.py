"""Restore local-file audio access after rendering the standalone explorer."""

from __future__ import annotations

import sys
from pathlib import Path


def enable_local_audio(path: Path) -> None:
    document = path.read_text(encoding="utf-8")
    document = document.replace(
        'sandbox="allow-scripts"',
        'sandbox="allow-scripts allow-same-origin"',
    )
    document = document.replace("media-src blob:", "media-src file: blob:")

    if 'sandbox="allow-scripts allow-same-origin"' not in document:
        raise RuntimeError(f"iframe sandbox marker not found in {path}")
    if document.count("media-src file: blob:") < 2:
        raise RuntimeError(f"outer and inner media CSP markers not found in {path}")

    path.write_text(document, encoding="utf-8")


def main() -> None:
    paths = [Path(value) for value in sys.argv[1:]]
    if not paths:
        paths = [Path(__file__).with_name("index.html")]
    for path in paths:
        enable_local_audio(path)
        print(path)


if __name__ == "__main__":
    main()
