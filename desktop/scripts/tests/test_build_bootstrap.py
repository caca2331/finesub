from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "build-bootstrap.ps1"
).read_text(encoding="utf-8")
RELEASE_SCRIPT = (
    Path(__file__).parents[1] / "build-release.ps1"
).read_text(encoding="utf-8")


def test_windows_build_uses_pyinstaller_onedir() -> None:
    assert '"-m",' in SCRIPT.lower()
    assert '"pyinstaller",' in SCRIPT.lower()
    assert '"--onedir"' in SCRIPT.lower()
    assert '"--windowed"' in SCRIPT.lower()
    assert "nuitka" not in SCRIPT.lower()
    assert '"pip",' not in SCRIPT.lower()


def test_windows_build_stages_only_release_sources() -> None:
    assert "Copy-PythonTree" in SCRIPT
    assert '"tests"' in SCRIPT
    assert '"__pycache__"' in SCRIPT
    assert '".pyinstaller-stage"' in SCRIPT


def test_windows_build_redacts_environment_for_packaging_tools() -> None:
    assert SCRIPT.count("-RedactSensitiveEnvironment") >= 1


def test_windows_build_applies_product_names_icon_and_version_resource() -> None:
    assert '"--name=FineSub Desktop"' in SCRIPT
    assert '"--icon=$IconPath"' in SCRIPT
    assert '"--version-file=$LauncherVersionFile"' in SCRIPT


def test_windows_build_generates_version_resources_from_release_version() -> None:
    assert "function New-VersionResource" in SCRIPT
    assert "$VersionResourceDirectory" in SCRIPT
    assert "-Version $Version" in SCRIPT


def test_release_build_accepts_ascii_bootstrap_directory() -> None:
    assert "[string]$BootstrapDirectory" in RELEASE_SCRIPT
    assert "-OutputDirectory $BootstrapDirectory" in RELEASE_SCRIPT
    assert '$Bootstrap = Join-Path $BootstrapDirectory "FineSub Desktop.dist"' in (
        RELEASE_SCRIPT
    )
