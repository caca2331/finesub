from __future__ import annotations

from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "install.ps1").read_text(encoding="utf-8")


def test_installer_installs_from_pypi_and_upgrades_in_place() -> None:
    assert '"tool", "install", "--force"' in SCRIPT
    assert "& $Uv @InstallArgs finesub" in SCRIPT


def test_the_index_can_be_pointed_elsewhere_but_is_not_guessed() -> None:
    """The automatic route lives in the Python this script is about to install."""

    assert "FINESUB_PYPI_INDEX" in SCRIPT
    assert "--default-index" in SCRIPT


def test_installer_provisions_uv_when_missing() -> None:
    assert "astral.sh/uv/install.ps1" in SCRIPT
    assert '$ErrorActionPreference = "Stop"' in SCRIPT


def test_installer_settles_the_data_directory_without_downloading() -> None:
    """A full setup fetches several GB; this script must stay seconds long."""

    assert "setup --dirs-only" in SCRIPT
    assert "finesub setup\n" not in SCRIPT


def test_the_installer_calls_finesub_by_path_not_by_name() -> None:
    """uv puts its tool directory on the *user* PATH, which the running shell
    does not see. With ErrorActionPreference "Stop" a bare `finesub` on a
    freshly-provisioned machine is a terminating CommandNotFoundException, and
    the install aborts before it ever says how to get started."""

    assert "uv tool dir --bin" in SCRIPT or "tool dir" in SCRIPT
    assert "& $FineSub setup" in SCRIPT
    assert "& finesub " not in SCRIPT


def test_the_installer_does_not_echo_a_possibly_credentialed_index() -> None:
    """An index may be https://user:token@host/simple, and this line lands in
    terminal scrollback and CI logs."""

    assert 'Write-Host "使用指定的 PyPI 源：$env:FINESUB_PYPI_INDEX"' not in SCRIPT
    assert "$IndexHost" in SCRIPT
