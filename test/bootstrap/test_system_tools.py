from __future__ import annotations

from pathlib import Path
import subprocess

from finesub_bootstrap import system_tools, token_counter


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
        _fake_run({"ffmpeg": (0, "ffmpeg version 7.1\n aac libx264 libmp3lame")}),
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


def test_an_lgpl_build_is_refused_for_lacking_libx264(monkeypatch) -> None:
    # The shape this check was missing: a complete, working ffmpeg whose only
    # gap is the GPL-licensed encoder. `libx264` is disabled in the LGPL
    # variants of the common Windows distributions, and a run only finds out
    # when it encodes its first clip -- after the correction stage has already
    # been reached.
    monkeypatch.setattr(
        system_tools.shutil, "which", lambda name: f"C:/tools/{name}.exe"
    )
    monkeypatch.setattr(
        system_tools.subprocess,
        "run",
        _fake_run({"ffmpeg": (0, "ffmpeg version 7.1\n aac flac libopenh264")}),
    )

    assert system_tools.find_system_ffmpeg() is None


def test_the_encoder_gate_matches_what_the_pipeline_requests() -> None:
    # `finesub_bootstrap` may not import the main package, so the gate holds a
    # second copy of the encoder names; this is what keeps the two from
    # drifting, the way `test_llm_video_route.py` pins the clip frame rate.
    # Drift is not hypothetical here: the packaged manifest shipped an LGPL
    # ffmpeg for as long as this gate asked for `libopus` -- which nothing
    # requests -- and not for `libx264`, which every clip does.
    from finesub.media.ffmpeg import AUDIO_CODEC_ARGS, VIDEO_ENCODER_ARGS

    requested = {
        args[args.index(flag) + 1]
        for args, flag in ((AUDIO_CODEC_ARGS, "-c:a"), (VIDEO_ENCODER_ARGS, "-c:v"))
    }

    assert requested <= set(system_tools.REQUIRED_FFMPEG_ENCODERS)


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


def _counter_on_path(monkeypatch, tmp_path: Path, output: tuple[int, str]) -> Path:
    executable = tmp_path / "tokcount.exe"
    executable.write_bytes(b"binary")
    monkeypatch.delenv(token_counter.TOKEN_COUNTER_VARIABLE, raising=False)
    monkeypatch.setattr(
        system_tools.shutil,
        "which",
        lambda name: str(executable) if name == "tokcount" else None,
    )
    monkeypatch.setattr(
        system_tools.subprocess, "run", _fake_run({"tokcount": output})
    )
    return executable


def test_a_system_token_counter_that_counts_is_accepted(
    monkeypatch, tmp_path: Path
) -> None:
    # The binary prints an experimental-tokenizer warning to stderr, which
    # `probe` folds into stdout ahead of the number.
    executable = _counter_on_path(
        monkeypatch, tmp_path, (0, "experimental tokenizer\n2\n")
    )

    found = system_tools.find_system_token_counter()

    assert found is not None
    assert found.path == executable.resolve()


def test_a_token_counter_that_answers_something_else_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    # Worse than nothing: whatever this returns is handed to the pipeline as an
    # exact count, and every budget downstream trusts it absolutely.
    _counter_on_path(monkeypatch, tmp_path, (0, "usage: tokcount [flags]\n"))

    assert system_tools.find_system_token_counter() is None


def test_a_configured_token_counter_is_taken_over_path(
    monkeypatch, tmp_path: Path
) -> None:
    # This finder answers "would the pipeline find a counter without us?", and
    # the pipeline reads the variable first -- so anything it would accept has
    # to stop us downloading a second copy.
    configured = tmp_path / "mine.exe"
    configured.write_bytes(b"binary")
    monkeypatch.setenv(token_counter.TOKEN_COUNTER_VARIABLE, str(configured))
    monkeypatch.setattr(system_tools.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        system_tools.subprocess, "run", _fake_run({"mine": (0, "2\n")})
    )

    found = system_tools.find_system_token_counter()

    assert found is not None
    assert found.path == configured.resolve()


def test_a_configured_token_counter_that_is_not_there_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        token_counter.TOKEN_COUNTER_VARIABLE, str(tmp_path / "gone.exe")
    )
    monkeypatch.setattr(system_tools.shutil, "which", lambda _name: None)

    assert system_tools.find_system_token_counter() is None


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
