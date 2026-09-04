"""What a whole run actually shows, at each level.

Golden-ish: the assertions are on the shape of the output rather than on exact
wording, so rephrasing a label does not fail the suite, but re-introducing a
per-artifact `Wrote`, a Timing block or a second progress channel does.
"""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys

import pytest

from finesub import pipeline, stages
from finesub.reporting import TerminalReporter, reporting_to


def _fakes(monkeypatch) -> None:
    def separate(input_path, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def vad_asr(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def stabilize(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def to_srt(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", vad_asr)
    monkeypatch.setattr(stages.asr_stabilize, "run_asr_stabilize", stabilize)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", to_srt)


def _run(tmp_path, monkeypatch, *, level: str, output=None) -> str:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    _fakes(monkeypatch)
    stream = io.StringIO()
    with reporting_to(TerminalReporter(stream, level=level, isatty=False)):
        pipeline.run_pipeline(
            source,
            output_path=output or (tmp_path / "out" / "final.srt"),
        )
    return stream.getvalue()


def test_a_clean_run_shows_four_stages_and_one_result(tmp_path, monkeypatch) -> None:
    shown = _run(tmp_path, monkeypatch, level="normal")

    stage_lines = [line for line in shown.splitlines() if line.startswith("[")]
    assert [line.split()[0] for line in stage_lines] == [
        "[1/4]",
        "[2/4]",
        "[3/4]",
        "[4/4]",
    ]
    assert shown.count("完成：") == 1
    assert "总耗时：" in shown


def test_normal_output_has_no_per_artifact_or_profiling_noise(
    tmp_path, monkeypatch
) -> None:
    shown = _run(tmp_path, monkeypatch, level="normal")

    assert "Wrote " not in shown
    assert "Timing:" not in shown
    assert "Resource usage" not in shown
    assert "Skipping" not in shown
    assert "Pipeline complete" not in shown, "the result is announced once, as 完成"


def test_a_redirected_run_never_rewrites_a_line(tmp_path, monkeypatch) -> None:
    shown = _run(tmp_path, monkeypatch, level="normal")

    assert "\r" not in shown, "a CI log must stay one line per event"
    assert "\x1b" not in shown


def test_quiet_keeps_only_the_result(tmp_path, monkeypatch) -> None:
    shown = _run(tmp_path, monkeypatch, level="quiet")

    assert "完成：" in shown
    assert "[1/4]" not in shown


def test_verbose_brings_back_the_skipped_artifact_detail(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out" / "final.srt"
    _run(tmp_path, monkeypatch, level="normal", output=output)
    # Second run over the same directory: every stage is now reused.
    shown = _run(tmp_path, monkeypatch, level="verbose", output=output)

    assert "skipping" in shown
    assert "已有结果，跳过" in shown


def test_a_rerun_marks_stages_reused_without_repeating_the_work(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out" / "final.srt"
    _run(tmp_path, monkeypatch, level="normal", output=output)
    shown = _run(tmp_path, monkeypatch, level="normal", output=output)

    assert shown.count("已有结果，跳过") >= 3
    assert shown.count("完成：") == 1


def test_verbose_times_every_stage_not_just_the_one_that_grew_a_timer(
    tmp_path, monkeypatch
) -> None:
    """Verbose promises timing; two of four stages used to report none.

    A reader was left to work out the difference by subtracting from the total.
    """

    shown = _run(tmp_path, monkeypatch, level="verbose")

    timed = {
        line.split("stage=")[1].split()[0]
        for line in shown.splitlines()
        if "stage timing" in line
    }
    assert timed == {"vocal_separation", "asr", "stabilize", "raw_srt"}


def test_a_reused_stage_is_recorded_without_a_duration(tmp_path, monkeypatch) -> None:
    """"Did not run" and "ran instantly" are different facts."""

    output = tmp_path / "out" / "final.srt"
    _run(tmp_path, monkeypatch, level="verbose", output=output)
    shown = _run(tmp_path, monkeypatch, level="verbose", output=output)

    assert "stage timing" not in shown
    metadata = json.loads(
        (output.parent / "final-metadata.json").read_text(encoding="utf-8")
    )
    for record in metadata["timing"]["stages"].values():
        assert record["status"] == "reused"
        assert record.get("elapsed_sec") is None


def test_a_labelled_reporter_names_every_line_it_writes() -> None:
    """Batch shares one terminal; a line that does not say whose it is is noise."""

    stream = io.StringIO()
    reporter = TerminalReporter(
        stream, isatty=False, prefix="[clip-a] ", level="verbose"
    )
    reporter.planned(["vocal"])
    reporter.stage_started("vocal")
    reporter.progress("vocal", completed=1, total=2, unit="blocks")
    reporter.warning("cpu-fallback", "GPU 不可用")
    reporter.completed("out.srt", 12.0)

    written = [line for line in stream.getvalue().splitlines() if line]
    assert all(line.startswith("[clip-a] ") for line in written), written


def test_the_log_file_does_not_drag_tqdm_and_library_logging_along(
    tmp_path, monkeypatch
) -> None:
    """`quieted_libraries` must keep the *terminal* level, not the file's.

    `verbose` means two things: our own debug lines, and un-muting third-party
    logging plus tqdm. The log file wants the first and must never get the
    second -- a progress bar captured into a file is nothing but noise, and it
    is what would turn a tens-of-KB log into tens of MB. The trap is that
    `FanOutReporter.level` is `verbose` (it reports the loudest member), so
    "simplify" this to `reporter.level` and the flood comes back.
    """

    from finesub import pipeline

    seen: list[str] = []

    @contextmanager
    def _record(level):
        seen.append(level)
        yield

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")  # the source is checked before anything is fed
    monkeypatch.setattr(pipeline, "quieted_libraries", _record)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **k: None)
    monkeypatch.setattr(
        sys, "argv", ["finesub", str(clip), "--log-level", "normal"]
    )

    assert pipeline.main() == 0
    assert seen == ["normal"]


def test_a_failed_run_still_gets_its_reason_and_traceback_into_the_log(
    tmp_path, monkeypatch
) -> None:
    """The failure is the line the log exists for.

    It used to reach the terminal only: the exception handler sat outside the
    log's `with`, so the file was already closed by the time it wrote -- and
    `FileReporter` swallows write errors on purpose, so nothing said so.
    """

    from finesub import pipeline

    logs = tmp_path / "logs"
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")  # a missing source fails pre-flight, not mid-run

    def _boom(*args, **kwargs):
        raise RuntimeError("ASR blew up")

    monkeypatch.setattr(pipeline, "resolve_logs_dir", lambda: logs)
    monkeypatch.setattr(pipeline, "run_pipeline", _boom)
    monkeypatch.setattr(sys, "argv", ["finesub", str(clip)])

    assert pipeline.main() == 1

    body = next(logs.glob("*.log")).read_text(encoding="utf-8")
    assert "ASR blew up" in body
    assert "Traceback (most recent call last)" in body
