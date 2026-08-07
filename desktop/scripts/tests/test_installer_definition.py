from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = (
    REPO_ROOT / "desktop" / "installer" / "FineSubDesktop.iss"
)
BUILD_SCRIPT = REPO_ROOT / "desktop" / "scripts" / "build-installer.ps1"


def _installer_text() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def _build_script_text() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


def test_installer_has_per_user_configurable_install_directory() -> None:
    script = _installer_text()
    assert "AppName=FineSub Desktop" in script
    assert "DefaultDirName={localappdata}\\Programs\\FineSub Desktop" in script
    assert "PrivilegesRequired=lowest" in script
    assert "DisableDirPage=no" in script


def test_installer_creates_shortcuts_and_can_launch_application() -> None:
    script = _installer_text()
    assert 'Name: "{autoprograms}\\FineSub Desktop"' in script
    assert 'Name: "{autodesktop}\\FineSub Desktop"' in script
    assert "Tasks: desktopicon" in script
    assert '#define AppExeName "FineSub Desktop.exe"' in script
    assert 'Filename: "{app}\\{#AppExeName}"' in script
    assert "postinstall" in script


def test_installer_uses_branding_and_exact_output_name() -> None:
    script = _installer_text()
    assert "SetupIconFile={#SetupIcon}" in script
    assert "UninstallDisplayIcon={app}\\FineSub Desktop.exe" in script
    assert "OutputBaseFilename=FineSub-Desktop-{#AppVersion}-Setup" in script
    assert "Compression=lzma2/ultra64" in script
    assert "SolidCompression=yes" in script


def test_installer_always_uses_bundled_chinese_language() -> None:
    installer = _installer_text()
    build_script = _build_script_text()
    assert '#ifndef ChineseLanguageFile' in installer
    assert 'Name: "chinesesimp"; MessagesFile: "{#ChineseLanguageFile}"' in installer
    assert 'ChineseLanguageFile = Join-Path $RepoRoot "desktop\\installer\\ChineseSimplified.isl"' in build_script
    assert '"/DChineseLanguageFile=$ChineseLanguageFile"' in build_script


def test_installer_writes_the_installed_marker() -> None:
    # The marker is what separates installed copies (personal data in
    # %LOCALAPPDATA%\FineSub) from portable ones; only the installer may
    # create it -- update payloads never ship one and the full updater
    # preserves it.
    script = _installer_text()
    assert (
        "SaveStringToFile(ExpandConstant('{app}\\installed.marker')" in script
    )
    assert "ssPostInstall" in script


def test_uninstall_removes_only_what_can_be_rebuilt_without_asking() -> None:
    # Same split as `finesub uninstall`: rebuildable state goes, and the two
    # kinds that cannot be recreated -- finished subtitles and personal data --
    # are each asked about.
    script = _installer_text()
    for runtime_child in ("runtime", "models", "cache", "app", ".update"):
        assert (
            "DelTree(ExpandConstant('{app}\\" + runtime_child + "')"
        ) in script
    assert "DeleteFile(ExpandConstant('{app}\\installed.marker'))" in script
    assert "{app}\\tasks" in script
    assert "{localappdata}\\FineSub" in script
    assert script.count("MsgBox(") == 2
    assert "usPostUninstall" in script


def test_installer_build_validates_required_application_files() -> None:
    script = _build_script_text()
    for expected in (
        "FineSub Desktop.exe",
        "app\\current.json",
        "runtime-manifest.json",
        "pylock.win-py312.toml",
    ):
        assert expected in script
    assert "ISCC.exe" in script
    assert "FineSub-Desktop-$Version-Setup.exe" in script
