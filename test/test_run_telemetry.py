"""Two diagnostic fields, and the line between telemetry and a verdict.

`docs/plans/crispasr-followups.md` -> 批次 B asks for both, and is explicit that
neither may become a stall or quality *rule*:

* `wall_cpu_ratio` looks like a hang detector and is not one -- a healthy
  GPU-bound stage blocks inside a CUDA call and accrues almost no CPU time, so
  our ASR stage would trip such a rule on every good run. Deciding a run is
  wedged stays with `speech/runtime/stall_watchdog.py`.
* `audio_coverage` records how much of the source reached the model. It is a
  plain ratio with no threshold attached: the predicate that *judged* it,
  `preprocessing/vad_failover.py`, was deleted on 2026-08-30
  (`docs/bench-baselines.md` 17.12). The field stayed -- it is what every
  analysis after that section reads.

So these tests pin what the fields *contain*, and that they are recorded even
when the value is unremarkable -- a metric that only appears when something is
wrong cannot show a trend.
"""

from __future__ import annotations

from finesub.run_metadata import stage_record
from finesub.speech.recognition.vad_asr_stage import audio_coverage


# --- wall:CPU as telemetry ---------------------------------------------------


def test_a_stage_record_carries_cpu_time_and_the_ratio() -> None:
    record = stage_record(status="executed", elapsed_sec=60.0, cpu_sec=6.0)
    assert record == {
        "status": "executed",
        "elapsed_sec": 60.0,
        "cpu_sec": 6.0,
        "wall_cpu_ratio": 10.0,
    }


def test_a_gpu_bound_stage_is_recorded_not_flagged() -> None:
    """The 100:1 shape is normal here; nothing may treat it as a verdict."""

    record = stage_record(status="executed", elapsed_sec=7825.0, cpu_sec=20.31)
    assert record["wall_cpu_ratio"] > 100
    assert set(record) == {"status", "elapsed_sec", "cpu_sec", "wall_cpu_ratio"}


def test_a_reused_stage_records_neither_time() -> None:
    """"Did not run" and "ran instantly" stay different facts."""

    assert stage_record(status="reused") == {"status": "reused"}


def test_zero_cpu_time_does_not_divide() -> None:
    record = stage_record(status="executed", elapsed_sec=1.0, cpu_sec=0.0)
    assert record["cpu_sec"] == 0.0
    assert "wall_cpu_ratio" not in record


def test_cpu_time_alone_is_recorded_without_a_ratio() -> None:
    record = stage_record(status="executed", cpu_sec=2.5)
    assert record == {"status": "executed", "cpu_sec": 2.5}


# --- audio coverage ----------------------------------------------------------


def _interval(start: float, end: float) -> dict[str, object]:
    return {"start": start, "end": end}


def test_coverage_reports_the_fraction_that_reached_the_model() -> None:
    result = audio_coverage(
        [_interval(0.0, 10.0), _interval(20.0, 30.0)], audio_duration=100.0
    )
    assert result == {
        "audio_sec": 100.0,
        "speech_sec": 20.0,
        "ratio": 0.2,
        "intervals": 2,
    }


def test_a_healthy_ratio_is_still_recorded() -> None:
    """A metric that appears only on failure cannot show a degradation."""

    result = audio_coverage([_interval(0.0, 95.0)], audio_duration=100.0)
    assert result["ratio"] == 0.95


def test_no_speech_is_zero_rather_than_missing() -> None:
    result = audio_coverage([], audio_duration=100.0)
    assert result == {
        "audio_sec": 100.0,
        "speech_sec": 0.0,
        "ratio": 0.0,
        "intervals": 0,
    }


def test_a_zero_length_source_reports_no_ratio_rather_than_dividing() -> None:
    result = audio_coverage([_interval(0.0, 1.0)], audio_duration=0.0)
    assert result["ratio"] is None


def test_reversed_and_unparseable_intervals_are_skipped_not_guessed() -> None:
    """Nothing here may invent speech seconds out of a malformed interval."""

    result = audio_coverage(
        [
            _interval(10.0, 5.0),
            {"start": "x", "end": 3.0},
            {},
            _interval(0.0, 4.0),
        ],
        audio_duration=100.0,
    )
    assert result["speech_sec"] == 4.0
    assert result["intervals"] == 4, "the count is of intervals seen, not of ones used"
