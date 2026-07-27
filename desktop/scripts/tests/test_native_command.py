from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / "desktop" / "scripts" / "native-command.ps1"


def _read_powershell_redirect(path: Path) -> str:
    content = path.read_bytes()
    return (
        content.decode("utf-16")
        if content.startswith((b"\xff\xfe", b"\xfe\xff"))
        else content.decode("utf-8")
    )


def test_native_command_uses_exit_code_when_stderr_contains_progress() -> None:
    work = REPO_ROOT / "dist" / f"native-command-test-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    stdout_path = work / "stdout.log"
    stderr_path = work / "stderr.log"
    probe_path = work / "probe.ps1"
    probe_path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f". '{HELPER}'",
                "$code = Invoke-NativeCommand `",
                "  -FilePath $env:ComSpec `",
                "  -ArgumentList @('/d', '/c', 'echo normal-progress 1>&2') `",
                f"  -StdoutPath '{stdout_path}' `",
                f"  -StderrPath '{stderr_path}'",
                "if ($code -ne 0) { exit $code }",
            ]
        ),
        encoding="utf-8-sig",
    )
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(probe_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stderr
        assert "normal-progress" in _read_powershell_redirect(stderr_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_native_command_redacts_secrets_and_restores_parent_environment() -> None:
    work = REPO_ROOT / "dist" / f"native-command-env-test-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    stdout_path = work / "stdout.log"
    stderr_path = work / "stderr.log"
    probe_path = work / "probe.ps1"
    probe_path.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f". '{HELPER}'",
                "$env:FINESUB_TEST_API_KEY = 'must-not-leak'",
                "$code = Invoke-NativeCommand `",
                "  -FilePath $env:ComSpec `",
                "  -ArgumentList @('/d', '/c', 'set FINESUB_TEST_API_KEY & exit /b 0') `",
                f"  -StdoutPath '{stdout_path}' `",
                f"  -StderrPath '{stderr_path}' `",
                "  -RedactSensitiveEnvironment",
                "if ($code -ne 0) { exit $code }",
                "if ($env:FINESUB_TEST_API_KEY -ne 'must-not-leak') { exit 90 }",
            ]
        ),
        encoding="utf-8-sig",
    )
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(probe_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stderr
        assert "must-not-leak" not in _read_powershell_redirect(stdout_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
