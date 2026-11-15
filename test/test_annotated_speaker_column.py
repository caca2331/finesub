"""The `spk` column on `<stem>-annotated.csv`.

Two things worth guarding. `spk` sits before `note` because `note` is the only
cell allowed to hold a raw `|` -- put the new column after it and a piped note
swallows it, silently. And the model never emits this column: it is written long
after the correction round, from labels the correction never saw.
"""

from __future__ import annotations

import json

from finesub.llm.output_protocol import (
    ANNOTATED_CSV_HEADER,
    TranslatedCsvSegment,
    render_annotated_csv,
    split_annotated_row,
)
from finesub.llm.stages.correction.run import _speaker_cells
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel, speaker_cell


def segment(source_ids, note="", translation="你好"):
    return TranslatedCsvSegment(
        source_ids=tuple(source_ids),
        start=0.0,
        end=1.0,
        corrected_text="hello",
        translation=translation,
        conf="high",
        note=note,
    )


def rows(text):
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_the_header_names_spk_before_note():
    assert ANNOTATED_CSV_HEADER.endswith("|spk|note")


def cells(mapping):
    """A stand-in for `_speaker_cells`: a line's cell, keyed by the source ids it covers."""
    return lambda source_ids: mapping.get(tuple(source_ids), "")


def test_a_line_asks_for_the_cell_of_all_its_sources_at_once():
    asked = []
    text = render_annotated_csv([segment(["1", "2"])], lambda ids: asked.append(tuple(ids)) or "s0+s1")
    assert asked == [("1", "2")], "one call per line, so the join happens on speakers, not strings"
    assert rows(text)[0].endswith("|s0+s1|")


def test_a_run_without_diarization_still_has_the_column():
    text = render_annotated_csv([segment(["1"])], None)
    assert rows(text)[0].endswith("|"), "empty cell, not a missing one"
    assert split_annotated_row(rows(text)[0]) is not None, "one layout parses every file"


def test_the_writer_encodes_pipes_inside_a_note():
    text = render_annotated_csv([segment(["1"], note="a|b|c")], cells({("1",): "s0"}))
    fields = split_annotated_row(rows(text)[0])
    assert fields is not None
    assert fields.spk == "s0"
    assert fields.note == "a｜b｜c", "written cells carry the full-width form"


def test_a_row_whose_note_holds_raw_pipes_still_parses():
    # The reason `spk` goes before `note` rather than after it: `note` is the
    # one cell that may arrive with raw pipes, and the split gives it whatever
    # is left. A column after it would be eaten by the first such note.
    fields = split_annotated_row("sub|1|1.0|0.0|hello|你好|high|2|s0|a|b|c")
    assert fields is not None
    assert fields.spk == "s0"
    assert fields.note == "a|b|c"


def test_the_reader_round_trips_what_the_writer_wrote():
    text = render_annotated_csv([segment(["1", "2"], note="n")], cells({("1", "2"): "s0+s1"}))
    fields = split_annotated_row(rows(text)[0])
    assert fields is not None
    assert (fields.kind, fields.position, fields.conf) == ("sub", "1,2", "high")
    assert fields.spk == "s0+s1"
    assert fields.note == "n"


def test_a_nine_column_row_is_refused_rather_than_shifted():
    # An older annotated.csv must fail loudly; the repository regenerates
    # artifacts rather than carrying two layouts.
    assert split_annotated_row("sub|1|1.0|0.0|hello|你好|high|2|note") is None


def test_speaker_cell_prefers_a_name_over_a_track_id():
    named = SegmentSpeaker(labels=(SpeakerLabel("guest", "high", name="miyako"),), track=3)
    assert speaker_cell([named]) == "miyako"


def test_speaker_cell_falls_back_to_track_ids():
    assert speaker_cell([SegmentSpeaker(track=2, extras={"tracks": [2, 0]})]) == "s2+s0"


def test_speaker_cell_is_empty_when_nothing_was_labelled():
    assert speaker_cell([SegmentSpeaker()]) == ""


def test_a_close_call_marks_the_first_track_with_a_tilde():
    # Plan 5l.7: with `review` the first track was not the one whose words these
    # are 44% of the time, without it 11% -- the proofreader's "check me first".
    close = SegmentSpeaker(track=1, review=True, extras={"tracks": [1, 0]})
    assert speaker_cell([close]) == "s1~+s0"


def test_a_merged_line_marks_a_track_once_whichever_segment_flagged_it():
    close = SegmentSpeaker(track=1, review=True, extras={"tracks": [1, 0]})
    plain = SegmentSpeaker(track=1, extras={"tracks": [1]})
    assert speaker_cell([plain, close]) == "s1~+s0"


def test_a_merged_line_leads_with_the_track_that_holds_most_of_it():
    # Audit 5q K6: 2 s led by s0, then 10 s led by s1. First-seen order read `s0+s1`,
    # telling the proofreader s0 is the main speaker of a line s1 holds for 10 of 12 s.
    short = SegmentSpeaker(track=0, extras={"tracks": [0, 1], "coverage": {"0": 0.8, "1": 0.2}})
    long = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 0.9}})
    assert speaker_cell([short, long], [2.0, 10.0]) == "s1+s0"
    assert speaker_cell([short, long], [10.0, 2.0]) == "s0+s1"


def test_one_segment_keeps_its_ranked_order_whatever_the_weight():
    close = SegmentSpeaker(track=1, review=True, extras={"tracks": [1, 0], "coverage": {"1": 0.51, "0": 0.49}})
    assert speaker_cell([close], [3.0]) == "s1~+s0"


def test_a_source_without_coverage_keeps_first_seen_order():
    old = SegmentSpeaker(track=0, extras={"tracks": [0]})
    new = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 1.0}})
    assert speaker_cell([old, new], [1.0, 50.0]) == "s0+s1"


def test_a_merged_line_decides_its_close_flag_on_the_whole_line():
    # Audit 5q: 2 s that were a close call (s0 just ahead) inside 10 s of plain s1. Re-ranked,
    # the source's flag landed on the second id: `s1+s0~`. The line as a whole is not close.
    close = SegmentSpeaker(track=0, review=True, extras={"tracks": [0, 1], "coverage": {"0": 0.52, "1": 0.48}})
    plain = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 0.95}})
    assert speaker_cell([close, plain], [2.0, 10.0], review_margin=0.15) == "s1+s0"


def test_a_merged_line_that_is_close_overall_marks_its_lead():
    first = SegmentSpeaker(track=0, extras={"tracks": [0, 1], "coverage": {"0": 0.9, "1": 0.1}})
    second = SegmentSpeaker(track=1, extras={"tracks": [1, 0], "coverage": {"1": 0.9, "0": 0.1}})
    assert speaker_cell([first, second], [5.0, 5.0], review_margin=0.15) == "s0~+s1"


def test_without_a_margin_a_close_source_marks_the_lead_never_a_later_id():
    close = SegmentSpeaker(track=0, review=True, extras={"tracks": [0, 1], "coverage": {"0": 0.52, "1": 0.48}})
    plain = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 0.95}})
    assert speaker_cell([close, plain], [2.0, 10.0]) == "s1~+s0"
    plain_first = SegmentSpeaker(track=0, extras={"tracks": [0]})
    late_close = SegmentSpeaker(track=1, review=True, extras={"tracks": [1, 0]})
    assert speaker_cell([plain_first, late_close]) == "s0~+s1"


def test_a_track_left_off_one_source_still_counts_its_time_there():
    # 10 s where s1 sits under second_share (not listed), 8 s that are nearly all s1. Summing only
    # the listed ids gave s0 8.4 vs s1 7.6 -> `s0~+s1`; over all coverage it is s1 9.2 vs s0 8.8.
    long_s0 = SegmentSpeaker(track=0, extras={"tracks": [0], "coverage": {"0": 0.84, "1": 0.16}})
    short_s1 = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 0.95, "0": 0.05}})
    assert speaker_cell([long_s0, short_s1], [10.0, 8.0], review_margin=0.15) == "s1~+s0"


def test_the_close_call_compares_against_the_largest_other_share_listed_or_not():
    # s2 holds 4.5 s of the line but no source lists it; the listed second, s1, holds 1 s.
    # Against s1 the line is not close; against s2 (the real runner-up) it is: 5.0 - 4.5 <= 0.75.
    main = SegmentSpeaker(track=0, extras={"tracks": [0], "coverage": {"0": 0.5, "2": 0.45}})
    aside = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 1.0}})
    assert speaker_cell([main, aside], [10.0, 1.0], review_margin=0.15) == "s0~+s1"


def test_a_named_source_beside_a_covered_one_keeps_first_seen_order():
    named = SegmentSpeaker(labels=(SpeakerLabel("guest", "high", name="miyako"),), track=3)
    covered = SegmentSpeaker(track=1, extras={"tracks": [1], "coverage": {"1": 1.0}})
    assert speaker_cell([named, covered], [1.0, 50.0], review_margin=0.15) == "miyako+s1"


def test_the_cell_lists_exactly_the_tracks_stable_holds():
    """No rendering-side rule may drop a track `assign` put in `tracks` (plan closeout 2).

    A `keep_share` thinning rule lived here between 2026-09-12 and 2026-09-18. At the
    theta it shipped with (0.20) it equalled `second_share`, so `assign` could never
    produce a runner-up it would remove -- it was dead code carrying a second rule to
    reason about. This pins the property the deletion leaves behind, over what `assign`
    actually emits rather than over hand-built states.
    """
    from finesub.speech.diarization.assign import assign_subtitle
    from finesub.speech.diarization.segments import TrackSegment

    for leader_end, runner_end in ((10.0, 9.5), (10.0, 5.0), (10.0, 2.1), (10.0, 1.9), (10.0, 0.5)):
        timeline = [TrackSegment(0, 0.0, leader_end), TrackSegment(1, 0.0, runner_end)]
        speaker = assign_subtitle(0.0, 10.0, timeline)
        listed = [f"s{track}" for track in speaker.extras["tracks"]]
        cell = speaker_cell([speaker], [10.0]).replace("~", "")
        assert cell.split("+") == listed, (leader_end, runner_end, cell, listed)


# --- the cell the product CSV writes, from stable -----------------------------

def stable(tmp_path, labelled=True):
    def row(index, tracks, review=False):
        item = {"id": str(index), "start": float(index), "end": index + 1.0, "text": f"line {index}"}
        if labelled:
            item["speaker"] = {"track": tracks[0], "tracks": tracks, "block": 0, "labels": []}
            if review:
                item["speaker"]["review"] = True
        return item

    path = tmp_path / "clip-stable.json"
    segments = [row(1, [0]), row(2, [1]), row(3, [0, 1]), row(4, [1, 0], review=True)]
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return path


def each(cell, *source_ids):
    return [cell([source_id]) for source_id in source_ids]


def test_labelled_input_gets_its_track_ids(tmp_path):
    cell = _speaker_cells(stable(tmp_path))
    assert each(cell, "1", "2", "3") == ["s0", "s1", "s0+s1"]


def test_a_merged_line_names_each_track_once(tmp_path):
    cell = _speaker_cells(stable(tmp_path))
    assert cell(["1", "3"]) == "s0+s1"
    assert cell(["5"]) is None, "a source today's lines do not have: the caller keeps the cell it has"


def test_a_merged_line_keeps_the_close_flag_without_listing_the_track_twice(tmp_path):
    # Plan 5o.1 K1: segment 2 is a plain s1, segment 4 a close call led by s1.
    # Glued as finished strings this read `s1+s1~+s0`.
    cell = _speaker_cells(stable(tmp_path))
    assert cell(["2", "4"]) == "s1~+s0"
    assert cell(["4", "2"]) == "s1~+s0"


def test_the_product_cell_renders_what_stable_holds(tmp_path):
    # Owner 2026-09-15 put theta back to 0.20; 2026-09-18 the rendering rule went away entirely.
    path = tmp_path / "clip-stable.json"
    speaker = {"track": 0, "tracks": [0, 1], "block": 0, "labels": [], "coverage": {"0": 0.7, "1": 0.3}}
    path.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 2.0, "text": "x", "speaker": speaker}]}),
                    encoding="utf-8")
    cell = _speaker_cells(path)
    assert cell(["1"]) == "s0+s1", "s1 holds 0.3 of the line, over 0.2 of s0's 0.7"
    assert json.loads(path.read_text(encoding="utf-8"))["segments"][0]["speaker"]["tracks"] == [0, 1], "stable is untouched"


def test_a_far_behind_runner_up_still_reaches_the_cell(tmp_path):
    """The runner-up here is far under half the leader -- the withdrawn 0.5 rule would have hidden it."""
    path = tmp_path / "clip-stable.json"
    speaker = {"track": 0, "tracks": [0, 1], "block": 0, "labels": [], "coverage": {"0": 0.9, "1": 0.2}}
    path.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 2.0, "text": "x", "speaker": speaker}]}),
                    encoding="utf-8")
    assert _speaker_cells(path)(["1"]) == "s0+s1"
    assert speaker_cell([SegmentSpeaker(track=0, extras={"tracks": [0, 1], "coverage": {"0": 0.9, "1": 0.2}})],
                        [2.0]) == "s0+s1"


def test_a_segment_with_a_broken_time_keeps_its_label(tmp_path):
    # Audit 5q: the duration moved inside the label's try, so `"end": null` dropped the label.
    path = tmp_path / "clip-stable.json"
    segment = {"id": "1", "start": 1.0, "end": None, "text": "x",
               "speaker": {"track": 0, "tracks": [0], "block": 0, "labels": []}}
    path.write_text(json.dumps({"segments": [segment]}), encoding="utf-8")
    cell = _speaker_cells(path)
    assert cell is not None and cell(["1"]).startswith("s0")


def test_unlabelled_input_has_no_cell(tmp_path):
    assert _speaker_cells(stable(tmp_path, labelled=False)) is None
