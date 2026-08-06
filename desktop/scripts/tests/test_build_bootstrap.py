from __future__ import annotations

import json
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
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


def test_windows_build_stages_the_shared_bootstrap_package() -> None:
    # PyInstaller resolves imports from --paths, not the build venv's editable
    # install; without staging src/finesub_bootstrap the frozen launcher would
    # depend on whatever the venv happens to expose.
    assert 'Join-Path $RepoRoot "src\\finesub_bootstrap"' in SCRIPT
    assert 'Join-Path $StageDirectory "finesub_bootstrap"' in SCRIPT


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


def test_desktop_version_sources_match_canonical_version() -> None:
    version = (
        REPOSITORY_ROOT / "desktop" / "VERSION"
    ).read_text(encoding="utf-8").strip()
    launcher = json.loads(
        (
            REPOSITORY_ROOT / "desktop" / "resources" / "launcher.json"
        ).read_text(encoding="utf-8")
    )
    frontend = json.loads(
        (
            REPOSITORY_ROOT / "desktop" / "frontend" / "package.json"
        ).read_text(encoding="utf-8")
    )
    frontend_lock = json.loads(
        (
            REPOSITORY_ROOT / "desktop" / "frontend" / "package-lock.json"
        ).read_text(encoding="utf-8")
    )
    installer = (
        REPOSITORY_ROOT / "desktop" / "installer" / "FineSubDesktop.iss"
    ).read_text(encoding="utf-8")
    launcher_version = (
        REPOSITORY_ROOT
        / "desktop"
        / "assets"
        / "finesub-desktop-version.txt"
    ).read_text(encoding="utf-8")
    updater_version = (
        REPOSITORY_ROOT
        / "desktop"
        / "assets"
        / "finesub-desktop-updater-version.txt"
    ).read_text(encoding="utf-8")

    assert launcher["appVersion"] == version
    assert launcher["launcherVersion"] == version
    assert frontend["version"] == version
    assert frontend_lock["version"] == version
    assert frontend_lock["packages"][""]["version"] == version
    assert re.search(
        rf'#define AppVersion "{re.escape(version)}"',
        installer,
    )
    assert f"StringStruct('ProductVersion', '{version}')" in launcher_version
    assert f"StringStruct('ProductVersion', '{version}')" in updater_version
    assert 'Join-Path $RepoRoot "desktop\\VERSION"' in SCRIPT
