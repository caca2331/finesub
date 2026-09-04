"""The timeline guard, and the SRT validator that had no production caller.

Both are guards, so both are red-verified here: a predicate that cannot fail is
worth nothing, and this suite exists because the tree already contained one
validator nobody called and one that nobody tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.reporting import NullReporter, reporting_to
from finesub.subtitles import rendering, time_order
from finesub.subtitles.model import warn_on_invalid_srt


class _Warnings(NullReporter):
    """Collect `(code, message)` so a test can assert the guard actually fired."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.entries.append((code, message))

    def codes(self) -> list[str]:
        return [code for code, _ in self.entries]


def _segment(start, end, text, words=None):
    segment = {"start": start, "end": end, "text": text}
    if words is not None:
        segment["words"] = [
            {"start": w_start, "end": w_end, "word": word}
            for w_start, w_end, word in words
        ]
    return segment


# --- the defect shape this guard exists for --------------------------------


def test_spans_can_be_monotone_while_the_words_run_backwards() -> None:
    """The #356 shape: a guard on spans passes the list that ships the bug.

    The first segment's span encloses the second's words, so sorting by span
    start says everything is fine while the word writer emits a cue that jumps
    back in time. This is the whole reason the caller names its quantity.
    """

    segments = [
        _segment(0.0, 2.0, "a b", [(0.0, 0.4, "a"), (1.5, 2.0, "b")]),
        _segment(2.0, 3.0, "c", [(0.9, 1.1, "c")]),
    ]

    assert time_order.first_backward(segments, using="spans") is None

    backward = time_order.first_backward(segments, using="words")
    assert backward is not None
    assert backward.index == 1
    assert backward.previous_sec == pytest.approx(1.5)
    assert backward.current_sec == pytest.approx(0.9)


def test_the_message_reads_in_the_units_of_the_file_the_user_is_looking_at() -> None:
    backward = time_order.first_backward(
        [
            _segment(0.0, 2.0, "a", [(90.0, 90.5, "a")]),
            _segment(2.0, 3.0, "b", [(1.25, 1.5, "b")]),
        ],
        using="words",
    )
    assert backward is not None
    described = backward.describe()
    assert "00:01:30,000" in described
    assert "00:00:01,250" in described


# --- what must NOT trip it -------------------------------------------------


def test_empty_text_is_skipped_rather_than_read_as_position_zero() -> None:
    """The writers drop empty segments, so counting one fakes a backward jump."""

    segments = [
        _segment(5.0, 6.0, "x"),
        _segment(0.0, 0.0, ""),
        _segment(7.0, 8.0, "y"),
    ]
    assert time_order.first_backward(segments, using="spans") is None


def test_a_segment_without_words_is_not_an_assertion_about_words() -> None:
    assert time_order.first_backward([_segment(9.0, 9.5, "z")], using="words") is None


def test_unparseable_timestamps_are_skipped_not_guessed() -> None:
    segments = [
        _segment(0.0, 1.0, "a"),
        _segment("later", 2.0, "b"),
        _segment(3.0, 4.0, "c"),
    ]
    assert time_order.first_backward(segments, using="spans") is None


# --- the reporting half ----------------------------------------------------


def test_report_backward_warns_and_names_where() -> None:
    reporter = _Warnings()
    with reporting_to(reporter):
        found = time_order.report_backward(
            [
                _segment(0.0, 2.0, "a", [(1.5, 2.0, "a")]),
                _segment(2.0, 3.0, "b", [(0.9, 1.1, "b")]),
            ],
            using="words",
            where="aligned JSON (clip-aligned.json)",
        )
    assert found is not None
    assert reporter.codes() == ["timeline-out-of-order"]
    assert "clip-aligned.json" in reporter.entries[0][1]


def test_report_backward_is_silent_on_an_ordered_timeline() -> None:
    reporter = _Warnings()
    with reporting_to(reporter):
        assert (
            time_order.report_backward(
                [_segment(0.0, 1.0, "a"), _segment(1.0, 2.0, "b")],
                using="spans",
                where="stable JSON",
            )
            is None
        )
    assert reporter.entries == []


# --- the validator that had no caller --------------------------------------


def test_the_srt_validator_warns_without_blocking_the_write(tmp_path: Path) -> None:
    """Wired warn-only: a run that produced usable subtitles still writes them.

    A three-line cue is a layout complaint, and the validator calls it an
    error. Letting that block the write would throw away the whole file over a
    formatting policy the splitter owns.
    """

    source = tmp_path / "clip-stable.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "one\ntwo\nthree"},
                ]
            }
        ),
        encoding="utf-8",
    )

    reporter = _Warnings()
    with reporting_to(reporter):
        output = rendering.convert_json_to_srt(source, output_path=tmp_path / "clip.srt")

    assert output.is_file()
    assert output.read_text(encoding="utf-8").strip() != ""
    assert "srt-invalid" in reporter.codes()


def test_a_clean_subtitle_produces_no_validator_warning(tmp_path: Path) -> None:
    source = tmp_path / "clip-stable.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "one"},
                    {"start": 1.0, "end": 2.0, "text": "two"},
                ]
            }
        ),
        encoding="utf-8",
    )

    reporter = _Warnings()
    with reporting_to(reporter):
        rendering.convert_json_to_srt(source, output_path=tmp_path / "clip.srt")

    assert [code for code in reporter.codes() if code.startswith("srt-")] == []


def test_warn_on_invalid_srt_reports_every_error_it_finds() -> None:
    reporter = _Warnings()
    with reporting_to(reporter):
        warn_on_invalid_srt("", where="empty.srt")
    assert reporter.codes() == ["srt-invalid"]


# --- the word writer's own guard -------------------------------------------


def test_the_word_writer_warns_when_its_own_quantity_runs_backwards() -> None:
    reporter = _Warnings()
    with reporting_to(reporter):
        rendering.render_word_srt(
            [
                _segment(0.0, 2.0, "a", [(1.5, 2.0, "a")]),
                _segment(2.0, 3.0, "b", [(0.9, 1.1, "b")]),
            ]
        )
    assert "timeline-out-of-order" in reporter.codes()


# --- one finding, one report, under the name the reader can open ------------


def test_an_intermediate_pass_stays_quiet(tmp_path: Path) -> None:
    """`validate=False` is what lets a multi-pass caller report once.

    The raw-SRT export renders into a scratch file and then rewrites it once
    per timeline profile. When every pass validated, one over-budget line was
    reported three times, each naming a `.part` path that does not survive the
    stage -- the shape of warning people learn to filter out.
    """

    source = tmp_path / "clip-stable.json"
    source.write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "one\ntwo\nthree"}]}),
        encoding="utf-8",
    )

    reporter = _Warnings()
    with reporting_to(reporter):
        output = rendering.convert_json_to_srt(
            source, output_path=tmp_path / ".clip.part.srt", validate=False
        )

    assert output.is_file(), "silencing the report must not silence the write"
    assert [code for code in reporter.codes() if code.startswith("srt-")] == []


def test_the_postprocess_pass_can_also_be_silenced(tmp_path: Path) -> None:
    from finesub.subtitles.postprocess import postprocess_srt_file

    target = tmp_path / "clip.srt"
    target.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\none\ntwo\nthree\n\n", encoding="utf-8"
    )

    loud = _Warnings()
    with reporting_to(loud):
        postprocess_srt_file(target, profile=0)
    quiet = _Warnings()
    with reporting_to(quiet):
        postprocess_srt_file(target, profile=0, validate=False)

    assert [code for code in loud.codes() if code.startswith("srt-")] != []
    assert [code for code in quiet.codes() if code.startswith("srt-")] == []


def test_the_raw_srt_export_reports_one_finding_once(tmp_path: Path, monkeypatch) -> None:
    """The end-to-end property, not just the flag that implements it.

    Asserted on the real `_create_raw_srt` composition rather than on a
    hand-rolled imitation of it, because the bug was in the composition: each
    piece was individually correct.
    """

    # `_use_or_create` moved to `stages` with `run_pipeline` (multi-input refactor).
    from finesub import stages as pipeline_module

    stable = tmp_path / "clip-stable.json"
    stable.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341"
                        "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341"
                        "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    final = tmp_path / "clip-raw.srt"
    temporary = final.with_name(f".{final.stem}.part{final.suffix}")

    from finesub.subtitles import rendering as to_srt
    from finesub.subtitles.model import warn_on_invalid_srt as warn
    from finesub.subtitles.postprocess import (
        TIMELINE_POSTPROCESS_PROFILES,
        postprocess_srt_file,
    )

    def create(destination: Path) -> Path:
        produced = to_srt.convert_json_to_srt(
            stable, output_path=destination, word=False, validate=False
        )
        for timeline_profile in TIMELINE_POSTPROCESS_PROFILES:
            postprocess_srt_file(produced, profile=timeline_profile, validate=False)
        produced = Path(produced)
        warn(produced.read_text(encoding="utf-8"), where=str(final))
        return produced

    reporter = _Warnings()
    with reporting_to(reporter):
        pipeline_module._use_or_create(final, "raw SRT export", create)

    budget = [message for code, message in reporter.entries if code == "srt-line-budget"]
    assert len(budget) == 1, f"expected exactly one report, got {budget}"
    assert str(final) in budget[0]
    assert temporary.name not in budget[0], "the report must not name a scratch file"
