"""The alignment-collapse sentinel, red-verified signal by signal.

A sentinel that never fires is indistinguishable from a healthy tree, so every
signal here has a case that trips it *and* a neighbouring case that must not.
The negative controls are the point: this guard sits on the artifact every run
produces, and a false positive on correct Japanese would be worse than no guard
at all.
"""

from __future__ import annotations

from finesub.reporting import NullReporter, reporting_to
from finesub.speech.recognition import align_sentinel


class _Warnings(NullReporter):
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.entries.append((code, message))


def _segment(words):
    return {
        "start": words[0][0] if words else 0.0,
        "end": words[-1][1] if words else 0.0,
        "text": "".join(word for _, _, word in words),
        "words": [
            {"start": start, "end": end, "word": word} for start, end, word in words
        ],
    }


def _healthy():
    """Ten seconds of ordinary speech: nothing here may trip any signal."""

    return [
        _segment([(0.0, 0.4, "and "), (0.4, 0.9, "so "), (0.9, 1.6, "my ")]),
        _segment([(2.0, 2.5, "fellow "), (2.5, 3.4, "americans ")]),
        _segment([(6.0, 6.6, "ask "), (6.6, 7.2, "not "), (7.2, 9.5, "what ")]),
    ]


def test_a_healthy_alignment_trips_nothing() -> None:
    assert align_sentinel.inspect(_healthy(), audio_duration=10.0) is None


# --- signal by signal ------------------------------------------------------


def test_words_pinned_at_the_origin_are_caught() -> None:
    """The shape a vocabulary mismatch produces: success, and every word at 0."""

    segments = [_segment([(0.0, 0.0, "a"), (0.0, 0.0, "b"), (0.0, 0.0, "c")])]
    found = align_sentinel.inspect(segments, audio_duration=11.0)
    assert found is not None
    assert found.signal == "zero-position words"
    assert "3/3" in found.detail


def test_a_single_zero_width_word_is_not_a_collapse() -> None:
    """Rounding puts the odd short token at zero width; that is not the defect."""

    words = [(0.0, 0.5, "a"), (0.5, 0.5, "b")] + [
        (float(i), float(i) + 0.5, "w") for i in range(1, 12)
    ]
    assert align_sentinel.inspect([_segment(words)], audio_duration=20.0) is None


def test_mostly_zero_width_spans_are_caught() -> None:
    words = [(float(i), float(i), "w") for i in range(10)] + [(10.0, 10.5, "x")]
    found = align_sentinel.inspect([_segment(words)], audio_duration=20.0)
    assert found is not None
    assert found.signal == "zero-length spans"


def test_a_whole_transcript_squeezed_into_a_moment_is_caught() -> None:
    words = [(0.0, 0.02, "hello"), (0.02, 0.05, "world"), (0.05, 0.08, "again")]
    found = align_sentinel.inspect([_segment(words)], audio_duration=600.0)
    assert found is not None
    assert found.signal == "characters per second"


def test_a_transcript_covering_almost_none_of_a_long_recording_is_caught() -> None:
    words = [(0.0, 2.0, "hello there"), (2.0, 4.0, "general kenobi")]
    found = align_sentinel.inspect([_segment(words)], audio_duration=600.0)
    assert found is not None
    assert found.signal == "coverage"


# --- the negative controls that keep it usable -----------------------------


def test_short_text_does_not_trip_the_rate_or_coverage_signals() -> None:
    """Under the character floor, both rate-shaped signals stay quiet.

    The span signal is a different question and is checked separately below;
    on a 600s file a 0.05s transcript is genuinely suspicious.
    """

    words = [(0.0, 0.05, "ok")]
    found = align_sentinel.inspect([_segment(words)], audio_duration=600.0)
    assert found is not None
    assert found.signal == "span"


def test_an_ordinary_short_clip_does_not_trip_the_span_signal() -> None:
    """The false positive the absolute audio floor exists for.

    A two-second clip carrying one 0.4s word satisfies `audio >= 4 * span`, so
    without the floor the sentinel would fire on half the fixtures in the tree.
    """

    words = [(0.2, 0.6, "hi")]
    assert align_sentinel.inspect([_segment(words)], audio_duration=2.0) is None


def test_characters_are_counted_as_code_points_not_bytes() -> None:
    """A byte count reads ~3x high on CJK and would fire on correct Japanese.

    20 code points (60 UTF-8 bytes) over a 0.8s span: 25 code points per second,
    comfortably under the 50 limit -- but 75 bytes per second, well over it. So
    this case passes only because the counter counts code points, and it is the
    one that goes red if anyone swaps in `len(text.encode())`.
    """

    text = "日本語のテスト文です"  # 10 code points, 30 bytes
    words = [(0.0, 0.4, text), (0.4, 0.8, text)]  # 20 code points / 60 bytes
    # 0.8s span: 25 code points per second (under 50), 75 bytes per second (over).
    assert align_sentinel.inspect([_segment(words)], audio_duration=2.0) is None


def test_the_duration_signals_are_skipped_when_the_duration_is_unknown() -> None:
    words = [(0.0, 2.0, "hello there"), (2.0, 4.0, "general kenobi")]
    assert align_sentinel.inspect([_segment(words)], audio_duration=None) is None


def test_a_transcript_with_no_words_is_not_an_assertion() -> None:
    assert align_sentinel.inspect([{"start": 0.0, "end": 1.0, "text": "x"}]) is None


def test_unparseable_timestamps_are_skipped() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "a",
            "words": [{"start": None, "end": None, "word": "a"}],
        }
    ]
    assert align_sentinel.inspect(segments, audio_duration=10.0) is None


# --- the reporting half ----------------------------------------------------


def test_report_warns_and_says_it_did_not_repair() -> None:
    reporter = _Warnings()
    with reporting_to(reporter):
        found = align_sentinel.report(
            [_segment([(0.0, 0.0, "a"), (0.0, 0.0, "b")])],
            audio_duration=30.0,
            where="aligned JSON (clip-aligned.json)",
        )
    assert found is not None
    assert reporter.entries[0][0] == "alignment-collapsed"
    assert "clip-aligned.json" in reporter.entries[0][1]


def test_report_is_silent_on_a_healthy_timeline() -> None:
    reporter = _Warnings()
    with reporting_to(reporter):
        assert align_sentinel.report(
            _healthy(), audio_duration=10.0, where="aligned JSON"
        ) is None
    assert reporter.entries == []
