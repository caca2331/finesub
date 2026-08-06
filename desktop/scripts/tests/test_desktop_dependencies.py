from __future__ import annotations

from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


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
        "pillow",
        "pydantic",
        "pystray",
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
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock_path = (
        REPOSITORY_ROOT / "desktop" / "runtime" / "pylock.win-py312.toml"
    )
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {package["name"]: package["version"] for package in lock["packages"]}

    assert packages["torch"] == "2.8.0+cu128"
    assert packages["torchaudio"] == "2.8.0+cu128"
    assert packages["torchvision"] == "0.23.0+cu128"
    assert "audio-separator" in packages
    assert "beartype" in packages
    assert "ml-collections" in packages
    assert "pydantic" in packages

    # The lock deliberately lags [asr], which moved to torch 2.11 in a0c4fb2.
    # Regenerating it is not a version bump: `uv pip compile` over today's [asr]
    # also pulls in 18 packages the desktop runtime has never shipped, among them
    # a *stock* ctranslate2 -- which would install fine and then fail at ASR
    # startup, since fw-refine needs the patched build that is not on any index.
    # So the refresh waits for the desktop runtime work rather than riding along
    # with a CLI release. What this test still guarantees is that the lag is
    # declared here and cannot widen unnoticed.
    requirements = {
        requirement.name.lower(): requirement
        for raw in project["project"]["optional-dependencies"]["asr"]
        for requirement in [Requirement(raw)]
    }
    known_lag = {"torch": "2.11.0", "torchaudio": "2.11.0"}
    for name in ("torch", "torchaudio"):
        public_version = packages[name].split("+", 1)[0]
        if Version(public_version) in requirements[name].specifier:
            continue
        expected = Requirement(f"{name}=={known_lag[name]}")
        assert requirements[name].specifier == expected.specifier, (
            f"{name}: [asr] moved to {requirements[name].specifier} but the "
            f"desktop lock still pins {public_version}. Either refresh "
            f"desktop/runtime/pylock.win-py312.toml or update known_lag here."
        )
