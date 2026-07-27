from __future__ import annotations

from pathlib import Path

import pytest

from desktop.scripts.verify_static_export import validate_static_export


def _write_export(root: Path, html: str) -> Path:
    index = root / "index.html"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(html, encoding="utf-8")
    return index


def test_rejects_root_relative_assets_for_file_webview(tmp_path: Path) -> None:
    index = _write_export(
        tmp_path,
        '<link href="/_next/app.css"><script src="/_next/app.js"></script>',
    )

    with pytest.raises(ValueError, match="root-relative"):
        validate_static_export(index)


def test_accepts_existing_relative_assets(tmp_path: Path) -> None:
    (tmp_path / "_next").mkdir()
    (tmp_path / "_next" / "app.css").write_text("", encoding="utf-8")
    (tmp_path / "_next" / "app.js").write_text("", encoding="utf-8")
    index = _write_export(
        tmp_path,
        (
            '<link href="./_next/app.css">'
            '<script src="./_next/app.js"></script>'
        ),
    )

    validate_static_export(index)


def test_rejects_missing_relative_asset(tmp_path: Path) -> None:
    index = _write_export(
        tmp_path,
        '<script src="./_next/missing.js"></script>',
    )

    with pytest.raises(FileNotFoundError, match="missing.js"):
        validate_static_export(index)
