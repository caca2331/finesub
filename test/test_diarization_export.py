"""Lossless exports: per-speaker SRT tracks and a single ASS.

The invariant is the point: every source line must reach at least one track. A
misassigned line costs one deletion; a dropped line has to be noticed, found and
retyped. So the tests below care far more about "nothing vanished" than about
"the right speaker got it".
"""

from __future__ import annotations

import pytest

from finesub.speech.diarization.export import (
    FALLBACK_TRACK,
    check_completeness,
    group_by_speaker,
    render_ass,
    render_speaker_srts,
)
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel
from finesub.subtitles.model import parse_srt


def line(start, end, text):
    return {"start": start, "end": end, "text": text}


def on_track(track, *, tracks=None, name=None):
    labels = (SpeakerLabel("guest", "high", name=name),) if name else ()
    return SegmentSpeaker(labels=labels, track=track, extras={"tracks": tracks or [track]})


UNPLACED = SegmentSpeaker()


def test_every_line_reaches_a_track():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")]
    speakers = [on_track(0), UNPLACED, on_track(1)]
    groups = group_by_speaker(subtitles, speakers)
    assert check_completeness(subtitles, groups) == []


def test_a_line_with_two_speakers_appears_in_both_tracks():
    subtitles = [line(0, 2, "both")]
    groups = group_by_speaker(subtitles, [on_track(0, tracks=[0, 1])])
    assert sorted(groups) == ["speaker0", "speaker1"]
    assert groups["speaker0"] == groups["speaker1"] == subtitles


def test_an_unplaced_line_lands_in_the_fallback_track():
    subtitles = [line(0, 1, "who?")]
    groups = group_by_speaker(subtitles, [UNPLACED])
    assert list(groups) == [FALLBACK_TRACK]


def test_a_named_speaker_names_the_track():
    groups = group_by_speaker([line(0, 1, "hi")], [on_track(1, name="miyako")])
    assert list(groups) == ["miyako"]


def test_completeness_reports_the_missing_line_rather_than_just_failing():
    subtitles = [line(0, 1, "kept"), line(1, 2, "lost")]
    groups = {"speaker0": [subtitles[0]]}
    assert check_completeness(subtitles, groups) == [1]


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        group_by_speaker([line(0, 1, "a")], [])


def test_srt_tracks_are_independently_numbered_from_one():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")]
    speakers = [on_track(0), on_track(1), on_track(0)]
    rendered = render_speaker_srts(subtitles, speakers, empty_cues=False)
    assert rendered["speaker0"].startswith("1\n")
    assert rendered["speaker1"].startswith("1\n")
    assert "\n2\n" in rendered["speaker0"], "the second line of that track, not of the file"


def test_srt_text_survives_the_round_trip():
    rendered = render_speaker_srts([line(0, 1, "こんにちは")], [on_track(0)], empty_cues=False)
    assert "こんにちは" in rendered["speaker0"]


@pytest.mark.parametrize("empty_cues, texts", [(False, ["a"]), (True, ["a", '""'])])
def test_a_line_without_text_is_in_a_track_only_with_empty_cues(empty_cues, texts):
    subtitles = [line(0, 1, "a"), line(1, 2, " ")]
    rendered = render_speaker_srts(subtitles, [on_track(0), on_track(0)], empty_cues=empty_cues)
    assert [cue.text for cue in parse_srt(rendered["speaker0"])] == texts


def test_ass_keeps_overlapping_speech_as_two_events():
    subtitles = [line(0.0, 2.0, "A talks"), line(0.5, 2.5, "B talks")]
    text = render_ass(subtitles, [on_track(0), on_track(1)])
    assert text.count("Dialogue:") == 2
    assert "A talks" in text and "B talks" in text, "one file expresses both"


def test_ass_declares_a_style_per_speaker():
    text = render_ass([line(0, 1, "a"), line(1, 2, "b")], [on_track(0), on_track(1)])
    assert "Style: speaker0," in text
    assert "Style: speaker1," in text


def test_ass_events_are_ordered_by_time_across_speakers():
    subtitles = [line(5.0, 6.0, "late"), line(0.0, 1.0, "early")]
    text = render_ass(subtitles, [on_track(1), on_track(0)])
    assert text.index("early") < text.index("late")


def test_ass_timestamps_are_centiseconds():
    text = render_ass([line(3661.5, 3662.25, "x")], [on_track(0)])
    assert "1:01:01.50" in text
    assert "1:01:02.25" in text


def test_ass_never_drops_a_line_either():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")]
    text = render_ass(subtitles, [on_track(0), UNPLACED, on_track(0, tracks=[0, 1])])
    assert text.count("Dialogue:") == 4, "three lines, one of them in two tracks"


# --- names a person wrote: safe as file names and ASS styles, never overwriting another export ---------------

def test_a_named_line_on_two_tracks_lands_once_under_its_name():
    groups = group_by_speaker([line(0, 1, "hi")], [on_track(1, tracks=[1, 2], name="miyako")])
    assert groups == {"miyako": [line(0, 1, "hi")]}


def test_a_name_keeps_no_character_a_file_name_or_an_ass_field_cannot_hold():
    name = 'A/B\\C:D*E?F"G<H>I|J,K\x00L\tM'
    (shown,) = group_by_speaker([line(0, 1, "x")], [on_track(0, name=name)])
    assert not set(shown) & set('/\\:*?"<>|,\x00\t'), shown
    assert shown == "A／B＼C：D＊E？F＂G＜H＞I｜J，K L M"
    style = next(row for row in render_ass([line(0, 1, "x")], [on_track(0, name=name)]).splitlines()
                 if row.startswith("Style: "))
    assert len(style.split(",")) == 23, "the name adds no field"


def test_a_long_name_is_cut_to_fit_a_file_name():
    (shown,) = group_by_speaker([line(0, 1, "x")], [on_track(0, name="名" * 300)])
    assert shown == "名" * 64


def test_names_that_would_share_a_file_get_their_own():
    subtitles = [line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c"), line(3, 4, "d"), line(4, 5, "e")]
    speakers = [on_track(1), on_track(2, name="speaker1"), on_track(3, name="Aki"), on_track(4, name="aki"),
                on_track(5, name="other")]
    groups = group_by_speaker(subtitles + [line(5, 6, "f")], speakers + [UNPLACED])
    assert list(groups) == ["speaker1", "speaker1-2", "Aki", "aki-2", "other-2", FALLBACK_TRACK]
    assert [lines[0]["text"] for lines in groups.values()] == ["a", "b", "c", "d", "e", "f"]


def test_a_name_never_passes_for_a_track_or_the_fallback():
    groups = group_by_speaker([line(0, 1, "a"), line(1, 2, "b")], [on_track(0, name="Speaker7"), on_track(1, name="other")])
    assert list(groups) == ["Speaker7-2", "other-2"], "no track 7 and no unplaced line in this export"


def test_the_same_name_written_two_ways_is_one_person():
    groups = group_by_speaker([line(0, 1, "a"), line(1, 2, "b"), line(2, 3, "c")],
                              [on_track(0, name="Aki  Tanaka"), on_track(1, name=" Aki Tanaka\n"),
                               on_track(2, name="Ame\u0301lie")])
    groups.update(group_by_speaker([line(3, 4, "d")], [on_track(3, name="Am\u00e9lie")]))
    assert list(groups) == ["Aki Tanaka", "Amélie"]


def test_the_export_stage_writes_every_name_into_its_own_file(tmp_path):
    import json
    from types import SimpleNamespace

    from finesub.stages import _run_speaker_export_stage

    def named(name, track):
        return {"track": track, "tracks": [track], "block": 0,
                "labels": [{"role": "guest", "conf": "high", "name": name, "src": "human"}]}

    segments = [{"id": str(i), "start": float(i), "end": i + 1.0, "text": f"line {i}", "speaker": speaker}
                for i, speaker in enumerate([named("../A:B", 0), named("..／A：B", 1)], start=1)]
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    _run_speaker_export_stage(SimpleNamespace(stable_json=stable), speaker_tracks="srt", empty_cues=False, announce=lambda *a, **k: None)
    written = sorted(path.name for path in (tmp_path / "clip-speakers").iterdir())
    assert written == ["clip-speakers-..／A：B-2.srt", "clip-speakers-..／A：B.srt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["clip-speakers", "clip-stable.json"]
