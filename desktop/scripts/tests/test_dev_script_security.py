from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPOSITORY_ROOT / "desktop" / "scripts"


def test_run_dev_is_a_download_free_transparent_wrapper() -> None:
    script = (SCRIPTS / "run-dev.ps1").read_text(encoding="utf-8").lower()

    forbidden = {
        "encodedcommand",
        "frombase64string",
        "invoke-expression",
        "invoke-webrequest",
        "start-process",
        "windowstyle",
        "taskkill",
        "pip install",
        "npm ci",
    }
    assert "dev_runner.py" in script
    assert not (forbidden & set(script.split()))
    for token in forbidden:
        assert token not in script


def test_dev_runner_never_invokes_a_shell_or_hidden_window() -> None:
    runner = (SCRIPTS / "dev_runner.py").read_text(encoding="utf-8").lower()

    assert "shell=true" not in runner
    assert "create_no_window" not in runner
    assert "windowstyle" not in runner
    assert "taskkill" not in runner
    assert "invoke-webrequest" not in runner


def test_setup_dev_keeps_installation_explicit_and_foreground() -> None:
    setup = (SCRIPTS / "setup-dev.ps1").read_text(encoding="utf-8").lower()

    assert "pip install" in setup
    assert "npm ci" in setup
    assert "pylock.win-py312.toml" in setup
    assert "desktoponly" in setup
    assert ".[desktop,dev]" in setup
    for token in (
        "start-process",
        "windowstyle",
        "taskkill",
        "invoke-webrequest",
        "encodedcommand",
        "invoke-expression",
    ):
        assert token not in setup
