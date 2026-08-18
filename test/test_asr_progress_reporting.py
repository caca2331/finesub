"""ASR progress throttling and recovery accounting.

Exercises the reporting seams directly rather than through a decode: what is
under test is how often the stage speaks and what it counts, neither of which
needs a model.
"""

from __future__ import annotations

import pytest

from finesub.reporting import TerminalReporter, reporting_to
from finesub.speech.recognition import vad_asr_stage as vad_asr
from finesub.speech.recognition import transcribe


class _Recorder:
    """Records what was reported.

    The stores are named apart from the methods on purpose: calling one
    `progress` would rebind the method on the instance and every later report
    would raise instead of being recorded.
    """

    def __init__(self) -> None:
        self.progress_calls: list[dict] = []
        self.debug_calls: list[str] = []
        self.warnings: list[tuple[str, str]] = []
        self.summaries: list[dict] = []

    # -- Reporter --
    def planned(self, stages) -> None:
        return

    def stage_started(self, stage, *, reused=False, detail="") -> None:
        return

    def progress(self, stage, *, completed, total=None, unit="", detail="") -> None:
        self.progress_calls.append(
            {"stage": stage, "completed": completed, "total": total, "unit": unit}
        )

    def summary(self, stage, metrics) -> None:
        self.summaries.append(dict(metrics))

    def warning(self, code, message, *, impact="", action="") -> None:
        self.warnings.append((code, message))

    def debug(self, message, fields=None) -> None:
        self.debug_calls.append(message)

    def completed(self, output, elapsed_sec) -> None:
        return

    def failed(self, stage, message) -> None:
        return


@pytest.fixture
def recorder():
    recorder = _Recorder()
    with reporting_to(recorder):
        yield recorder


def test_a_hundred_groups_report_at_most_once_per_twentieth(recorder) -> None:
    transcribe._stats_local.progress_step = None
    for completed in range(1, 101):
        transcribe._report_progress(completed, 100, completed)

    assert len(recorder.progress_calls) <= 21
    assert recorder.progress_calls[-1]["completed"] == 100
    assert recorder.progress_calls[0]["unit"] == "intervals"


def test_every_group_still_leaves_a_verbose_trace(recorder) -> None:
    transcribe._stats_local.progress_step = None
    for completed in range(1, 101):
        transcribe._note(f"group ASR (intervals={completed}/100)")

    assert len(recorder.debug_calls) == 100


def test_progress_never_goes_backwards_within_a_step(recorder) -> None:
    transcribe._stats_local.progress_step = None
    transcribe._report_progress(5, 100, 1)
    transcribe._report_progress(6, 100, 2)
    transcribe._report_progress(7, 100, 3)

    assert len(recorder.progress_calls) == 1


def test_a_resumed_run_reports_where_it_actually_starts(recorder) -> None:
    """Reported immediately, or a run resuming at 60% would look like 0%."""

    transcribe._stats_local.progress_step = None
    transcribe._report_progress(60, 100, 12)

    assert recorder.progress_calls[0]["completed"] == 60


def test_recovery_steps_are_counted_while_staying_out_of_normal_output() -> None:
    quiet = _Recorder()
    with reporting_to(quiet), transcribe.collecting_stats() as stats:
        transcribe._note("beam rescue", count="beam_rescue_attempted")
        transcribe._note("beam rescue", count="beam_rescue_attempted")
        transcribe._note("accepted", count="beam_rescue_accepted")
        transcribe._note("isolating", count="isolated_intervals")

    assert stats["beam_rescue_attempted"] == 2
    assert stats["beam_rescue_accepted"] == 1
    assert stats["isolated_intervals"] == 1
    assert quiet.warnings == [], "recovery that worked is not a warning"


def test_a_dropped_group_stays_a_warning() -> None:
    recorder = _Recorder()
    with reporting_to(recorder), transcribe.collecting_stats() as stats:
        transcribe._warn(
            "asr-group-dropped",
            "teacher-force alignment failed; dropping this group",
            count="dropped_groups",
        )

    assert stats["dropped_groups"] == 1
    assert recorder.warnings[0][0] == "asr-group-dropped"


def test_counters_outside_a_collecting_scope_are_harmless(recorder) -> None:
    transcribe._note("no scope here", count="beam_rescue_attempted")

    assert recorder.debug_calls == ["no scope here"]


def test_the_stage_summary_hides_a_clean_run_and_shows_a_rough_one() -> None:
    clean = vad_asr._recovery_summary({}, [{}] * 40)
    rough = vad_asr._recovery_summary(
        {
            "temporary_recalls": 2,
            "isolated_intervals": 1,
            "beam_rescue_attempted": 3,
            "beam_rescue_accepted": 2,
            "dropped_groups": 1,
        },
        [{}] * 40,
    )

    stream = _render(clean)
    assert stream == "      语音识别摘要：区间 40\n"

    rendered = _render(rough)
    assert "临时召回 2" in rendered
    assert "beam 救援 2/3" in rendered
    assert "丢弃 group 1" in rendered
    assert "对齐重试" not in rendered, "zero counters must not reach the line"


def _render(metrics) -> str:
    import io

    stream = io.StringIO()
    TerminalReporter(stream, isatty=False).summary("aligned", metrics)
    return stream.getvalue()
