from __future__ import annotations

from pathlib import Path
import subprocess

from finesub_bootstrap import system_tools


def _fake_run(mapping: dict[str, tuple[int, str]]):
    """Answer probes by the tool name in argv[0], ignoring the rest."""

    def run(command, **_kwargs):
        name = Path(command[0]).stem.lower()
        returncode, output = mapping.get(name, (1, ""))
        return subprocess.CompletedProcess(command, returncode, output, "")

    return run


def test_a_capable_system_ffmpeg_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )
    monkeypatch.setattr(
        system_tools.subprocess,
        "run",
        _fake_run({"ffmpeg": (0, "ffmpeg version 7.1\n libopus aac libmp3lame")}),
    )

    found = system_tools.find_system_ffmpeg()

    assert found is not None
    # Compact: the UI renders this next to the resource name, and a full
    # ffmpeg banner runs to ~90 characters.
    assert found.version == "7.1"
    assert found.directory == Path("C:/tools").resolve()


def test_ffmpeg_without_a_required_encoder_is_refused(monkeypatch) -> None:
    # Presence is not usability. A build missing the encoders the pipeline uses
    # would fail mid-run, long after the user chose to skip the download.
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )
    monkeypatch.setattr(
        system_tools.subprocess,
        "run",
        _fake_run({"ffmpeg": (0, "ffmpeg version 7.1\n libmp3lame")}),
    )

    assert system_tools.find_system_ffmpeg() is None


def test_ffmpeg_without_ffprobe_is_refused(monkeypatch) -> None:
    # The pipeline calls both; half an install is not an install.
    monkeypatch.setattr(
        system_tools.shutil,
        "which",
        lambda name: "C:/tools/ffmpeg.exe" if name == "ffmpeg" else None,
    )

    assert system_tools.find_system_ffmpeg() is None


def test_a_tool_that_fails_to_launch_is_refused(monkeypatch) -> None:
    # A broken shim on PATH must not read as success.
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )

    def explode(*_args, **_kwargs):
        raise OSError("not executable")

    monkeypatch.setattr(system_tools.subprocess, "run", explode)

    assert system_tools.find_system_ffmpeg() is None
    assert system_tools.find_system_git() is None


def test_a_nonzero_exit_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )
    monkeypatch.setattr(
        system_tools.subprocess, "run", _fake_run({"git": (128, "fatal")})
    )

    assert system_tools.find_system_git() is None


def test_a_working_system_git_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )
    monkeypatch.setattr(
        system_tools.subprocess,
        "run",
        _fake_run({"git": (0, "git version 2.44.0.windows.1\n")}),
    )

    found = system_tools.find_system_git()

    assert found is not None
    assert found.version == "2.44.0.windows.1"


def test_missing_from_path_is_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr(system_tools.shutil, "which", lambda _name: None)

    assert system_tools.find_system_ffmpeg() is None
    assert system_tools.find_system_git() is None


def test_an_unrecognised_banner_is_kept_verbatim(monkeypatch) -> None:
    # Better a long string than a wrong one: if the banner is not the shape we
    # expect, report what the tool actually said.
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )
    monkeypatch.setattr(
        system_tools.subprocess, "run", _fake_run({"git": (0, "weird build 9\n")})
    )

    found = system_tools.find_system_git()

    assert found is not None
    assert found.version == "weird build 9"
