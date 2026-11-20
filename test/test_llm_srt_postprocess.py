from __future__ import annotations

import pytest

from finesub import config as app_config
from finesub.subtitles.postprocess import TimelineParams, postprocess_srt_text
from finesub.subtitles.model import parse_srt


@pytest.fixture(autouse=True)
def _no_config(tmp_path, monkeypatch):
    """A per-test config path that starts absent, so the pads below are the
    defaults and `_configure` has a file to write."""

    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "absent.toml"))
    app_config.clear_config_cache()
    yield
    app_config.clear_config_cache()


def _configure(tmp_path, body: str) -> None:
    (tmp_path / "absent.toml").write_text(body, encoding="utf-8")
    app_config.clear_config_cache()


def _segment_times(text: str) -> list[tuple[float, float]]:
    return [(segment.start, segment.end) for segment in parse_srt(text)]


def test_profile_one_extends_every_end_by_fixed_pad() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:00,300\n中\n\n"
        "2\n00:00:02,000 --> 00:00:02,300\n中中\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\n较长原轴\n\n"
        "4\n00:00:06,000 --> 00:00:06,200\n贴紧\n\n"
        "5\n00:00:06,300 --> 00:00:07,000\n下一条\n"
    )

    rendered, report = postprocess_srt_text(source, profile=1)

    assert _segment_times(rendered) == [
        (0.0, 0.6),
        (2.0, 2.6),
        (4.0, 5.3),
        (6.0, 6.3),
        (6.3, 7.3),
    ]
    assert report.applied_profiles == (1,)
    assert report.duration_extended == 5
    assert report.flash_extended == 0
    assert report.punctuation_replacements == 0


def test_profile_one_caps_extension_at_next_start_and_closes_flash_gap() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:00,300\n贴紧下一条\n\n"
        "2\n00:00:00,500 --> 00:00:02,000\n先延后闪轴\n\n"
        "3\n00:00:02,400 --> 00:00:03,450\n闭合闪轴\n"
    )

    rendered, report = postprocess_srt_text(source, profile=1)
    segments = parse_srt(rendered)

    # Gap to next is 0.2s < 0.3s pad → capped at next start (no leftover flash).
    assert segments[0].end == 0.5
    # +0.3s pad leaves a 0.1s gap (< 0.3s flash threshold) → closed to next start.
    assert segments[1].end == 2.4
    assert report.duration_extended == 3
    assert report.flash_extended == 1


def test_profile_two_cleans_text_without_changing_timeline() -> None:
    source = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "  你好，世界。  \n\n"
        "2\n"
        "00:00:01,100 --> 00:00:01,600\n"
        "第　二句\n"
    )

    rendered, report = postprocess_srt_text(source, profile=2)
    segments = parse_srt(rendered)

    assert _segment_times(rendered) == [(0.0, 0.5), (1.1, 1.6)]
    assert [segment.text for segment in segments] == ["你好 世界", "第 二句"]
    assert report.applied_profiles == (2,)
    assert report.duration_extended == 0
    assert report.flash_extended == 0
    assert report.punctuation_replacements == 3
    assert report.trimmed_lines == 1


def test_profile_zero_is_profile_one_then_profile_two() -> None:
    source = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "  你好，世界。  \n\n"
        "2\n"
        "00:00:01,100 --> 00:00:01,600\n"
        "第　二句\n"
    )

    combined, report = postprocess_srt_text(source, profile=0)
    duration_only, _ = postprocess_srt_text(source, profile=1)
    composed, _ = postprocess_srt_text(duration_only, profile=2)

    assert combined == composed
    assert report.applied_profiles == (3, 4, 1, 2)
    # Seg1: +0.3 → 0.8; leftover gap to 1.1 is 0.3 (not < 0.3) so no flash.
    # Seg2: +0.3 → 1.9.
    assert report.duration_extended == 2
    assert report.flash_extended == 0
    assert report.punctuation_replacements == 3
    assert report.trimmed_lines == 1


def test_profile_minus_one_is_noop_render() -> None:
    source = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "你好，世界。\n"
    )

    rendered, report = postprocess_srt_text(source, profile=-1)

    segment = parse_srt(rendered)[0]
    assert segment.end == 0.5
    assert segment.text == "你好，世界。"
    assert report.profile == -1
    assert report.applied_profiles == ()


def test_unsupported_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected one of -1, 0, 1, 2, 3, 4"):
        postprocess_srt_text(
            "1\n00:00:00,000 --> 00:00:00,500\n字幕\n",
            profile=5,
        )


def test_profile_four_trims_overlaps_and_warns() -> None:
    import io

    from finesub.reporting import TerminalReporter, reporting_to

    source = (
        "1\n00:00:00,000 --> 00:00:02,000\n压到下一句\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n被压\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\n正常贴紧\n"
    )
    stream = io.StringIO()

    with reporting_to(TerminalReporter(stream, isatty=False)):
        rendered, report = postprocess_srt_text(source, profile=4)

    assert _segment_times(rendered) == [(0.0, 1.5), (1.5, 3.0), (3.0, 4.0)]
    assert report.applied_profiles == (4,)
    assert report.overlaps_fixed == 1
    assert report.duration_extended == 0
    warning = stream.getvalue()
    assert "1 overlapping subtitle cue(s)" in warning
    assert "00:00:02,000" in warning


def test_profile_four_is_silent_and_idempotent_without_overlap(capsys) -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n甲\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n乙\n"
    )

    rendered, report = postprocess_srt_text(source, profile=4)

    assert _segment_times(rendered) == [(0.0, 1.0), (1.0, 2.0)]
    assert report.overlaps_fixed == 0
    assert capsys.readouterr().err == ""


def test_profile_four_collapses_a_cue_starting_after_the_next_one() -> None:
    source = (
        "1\n00:00:05,000 --> 00:00:06,000\n乱序\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n更早\n"
    )

    rendered, report = postprocess_srt_text(source, profile=4)

    # The end can never precede its own start, so the cue collapses instead.
    assert _segment_times(rendered) == [(5.0, 5.0), (1.0, 2.0)]
    assert report.overlaps_fixed == 1


def test_profile_zero_resolves_overlap_before_extending_ends() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:02,000\n压到下一句\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n被压\n"
    )

    rendered, report = postprocess_srt_text(source, profile=0)

    # Overlap first (end 2.0 -> 1.5), then the +0.3s pad capped at the next
    # start leaves it there; the last cue keeps the full pad.
    assert _segment_times(rendered) == [(0.0, 1.5), (1.5, 3.3)]
    assert report.applied_profiles == (3, 4, 1, 2)
    assert report.overlaps_fixed == 1
    # Only cue 2 could actually move: cue 1 is already capped at the next start.
    assert report.duration_extended == 1


# --- the cue-end pad's audio bound (P1) ------------------------------------
# The pad has no "next start" to stop at on the last cue, so it used to run
# unconditionally -- which is how a stable JSON clamped exactly to the audio
# end came back out 0.3 s past it. ⚠ The bound is a ceiling, not a rule
# against padding the last cue: both cases are pinned below on purpose,
# because "just stop extending the last cue" would shorten every normal file.


def test_last_cue_with_room_still_gets_the_full_pad() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n前\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\n末条\n"
    )

    rendered, report = postprocess_srt_text(source, profile=1, audio_end=600.0)

    assert _segment_times(rendered)[-1] == (4.0, 5.3)
    assert report.duration_extended == 2


def test_last_cue_against_the_audio_end_is_capped_not_padded() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n前\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\n末条\n"
    )

    rendered, _ = postprocess_srt_text(source, profile=1, audio_end=5.1)

    assert _segment_times(rendered)[-1] == (4.0, 5.1)


def test_a_cue_already_past_the_audio_end_is_clamped() -> None:
    """The final writer must also repair an overrun in translated input."""

    source = "1\n00:00:04,000 --> 00:00:05,000\n越界\n"

    rendered, _ = postprocess_srt_text(source, profile=1, audio_end=4.5)

    assert _segment_times(rendered)[-1] == (4.0, 4.5)


def test_flash_cleanup_and_out_of_bounds_tail_obey_audio_end() -> None:
    source = (
        "1\n00:00:04,000 --> 00:00:04,900\nfirst\n\n"
        "2\n00:00:05,100 --> 00:00:05,300\nsecond\n\n"
        "3\n00:00:05,400 --> 00:00:05,600\nthird\n"
    )
    rendered, _ = postprocess_srt_text(source, profile=0, audio_end=5.0)
    cues = parse_srt(rendered)
    assert [cue.text for cue in cues] == ["first", "second", "third"]
    assert all(cue.start < cue.end <= 5.0 for cue in cues)
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


# ------------------------------------------------------ `[postprocess]` in config


def test_the_report_says_what_the_timeline_ran_with() -> None:
    _, report = postprocess_srt_text("1\n00:00:00,000 --> 00:00:01,000\n中\n", profile=1)

    assert (report.end_extend_seconds, report.flash_gap_seconds) == (0.3, 0.3)


def test_config_changes_the_pad_and_the_report_follows(tmp_path) -> None:
    _configure(tmp_path, "[postprocess]\nend_extend_seconds = 0.5\nflash_gap_seconds = 0.0\n")
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n前\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\n末条\n"
    )

    rendered, report = postprocess_srt_text(source, profile=1)

    assert _segment_times(rendered) == [(0.0, 1.5), (4.0, 5.5)]
    assert (report.end_extend_seconds, report.flash_gap_seconds) == (0.5, 0.0)


def test_a_large_pad_still_stops_at_the_next_start_and_the_audio_end(tmp_path) -> None:
    _configure(tmp_path, "[postprocess]\nend_extend_seconds = 30\n")
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n前\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\n末条\n"
    )

    rendered, _ = postprocess_srt_text(source, profile=1, audio_end=6.0)

    assert _segment_times(rendered) == [(0.0, 4.0), (4.0, 6.0)]


def test_an_explicit_timeline_beats_the_config(tmp_path) -> None:
    _configure(tmp_path, "[postprocess]\nend_extend_seconds = 0.5\n")
    source = "1\n00:00:04,000 --> 00:00:05,000\n末条\n"

    rendered, report = postprocess_srt_text(
        source, profile=1, timeline=TimelineParams(end_extend_seconds=0.1, flash_gap_seconds=0.3)
    )

    assert _segment_times(rendered)[-1] == (4.0, 5.1)
    assert report.end_extend_seconds == 0.1


@pytest.mark.parametrize(
    ("end_extend", "flash_gap", "wording"),
    [
        (-10.0, 0.3, "at least 0.0"),
        (0.3, -1.0, "at least 0.0"),
        (float("nan"), 0.3, "a finite number"),
        ("0.3", 0.3, "a number"),
        (True, 0.3, "a number"),
    ],
)
def test_an_explicit_timeline_is_checked_like_a_configured_one(
    end_extend, flash_gap, wording
) -> None:
    """Passing one skips the config read, not the rules: a negative pad used to
    turn 04:000 --> 05:000 into 04:000 --> 04:000 without a word."""

    with pytest.raises(ValueError, match=wording):
        TimelineParams(end_extend_seconds=end_extend, flash_gap_seconds=flash_gap)


def test_an_explicit_timeline_does_not_read_the_config_it_overrides(tmp_path) -> None:
    _configure(tmp_path, '[postprocess]\nend_extend_seconds = "long"\n')

    rendered, _ = postprocess_srt_text(
        "1\n00:00:04,000 --> 00:00:05,000\n末条\n",
        profile=1,
        timeline=TimelineParams(end_extend_seconds=0.2, flash_gap_seconds=0.3),
    )

    assert _segment_times(rendered)[-1] == (4.0, 5.2)


def test_an_unused_timeline_is_not_reported() -> None:
    _, report = postprocess_srt_text(
        "1\n00:00:00,000 --> 00:00:01,000\n中，\n",
        profile=2,
        timeline=TimelineParams(end_extend_seconds=0.5, flash_gap_seconds=0.3),
    )

    assert (report.end_extend_seconds, report.flash_gap_seconds) == (None, None)


def test_a_profile_without_a_timeline_step_does_not_read_it(tmp_path) -> None:
    """A broken `[postprocess]` must not break a punctuation-only pass."""

    _configure(tmp_path, '[postprocess]\nend_extend_seconds = "long"\n')

    _, report = postprocess_srt_text("1\n00:00:00,000 --> 00:00:01,000\n中，\n", profile=2)

    assert report.end_extend_seconds is None
    with pytest.raises(ValueError, match="postprocess.end_extend_seconds must be a number"):
        postprocess_srt_text("1\n00:00:00,000 --> 00:00:01,000\n中\n", profile=1)
