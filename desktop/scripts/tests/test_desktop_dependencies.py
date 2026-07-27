from __future__ import annotations

from pathlib import Path
import re
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_desktop_extra_declares_every_direct_python_dependency() -> None:
    document = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = document["project"]["optional-dependencies"]["desktop"]
    names = {
        re.split(r"[<>=!~ ;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in dependencies
    }

    assert names == {
        "cryptography",
        "httpx",
        "packaging",
        "pydantic",
        "pywebview",
    }

    development = document["project"]["optional-dependencies"]["dev"]
    development_names = {
        re.split(r"[<>=!~ ;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in development
    }
    assert {"pillow", "pyinstaller", "pyinstaller-hooks-contrib"} <= (
        development_names
    )


def test_windows_ai_runtime_lock_pins_torch_stack() -> None:
    lock_path = REPOSITORY_ROOT / "desktop" / "runtime" / "pylock.win-py312.toml"
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {package["name"]: package["version"] for package in lock["packages"]}

    assert packages["torch"] == "2.9.1+cu128"
    assert packages["torchaudio"] == "2.9.1+cu128"
    assert packages["torchvision"] == "0.24.1+cu128"
