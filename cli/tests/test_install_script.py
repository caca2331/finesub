from __future__ import annotations

from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "install.ps1").read_text(encoding="utf-8")


def test_installer_installs_from_pypi_and_upgrades_in_place() -> None:
    assert "tool install --force finesub" in SCRIPT


def test_installer_provisions_uv_when_missing() -> None:
    assert "astral.sh/uv/install.ps1" in SCRIPT
    assert '$ErrorActionPreference = "Stop"' in SCRIPT
