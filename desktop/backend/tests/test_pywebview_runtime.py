from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from types import ModuleType

from desktop.backend.launcher.main import install_frozen_pywebview_win32


def test_frozen_win32_source_is_registered_as_pywebview_platform() -> None:
    root = Path(__file__).resolve().parents[3] / "dist" / f"win32-load-{os.getpid()}"
    source = root / "win32.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("MARKER = 'loaded-from-source'\n", encoding="utf-8")

    webview = ModuleType("webview")
    webview.__path__ = []  # type: ignore[attr-defined]
    platforms = ModuleType("webview.platforms")
    platforms.__path__ = []  # type: ignore[attr-defined]
    previous = {
        name: sys.modules.get(name)
        for name in ("webview", "webview.platforms", "webview.platforms.win32")
    }
    sys.modules["webview"] = webview
    sys.modules["webview.platforms"] = platforms
    try:
        install_frozen_pywebview_win32(source)

        loaded = sys.modules["webview.platforms.win32"]
        assert loaded.MARKER == "loaded-from-source"  # type: ignore[attr-defined]
        assert platforms.win32 is loaded  # type: ignore[attr-defined]
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        shutil.rmtree(root, ignore_errors=True)
