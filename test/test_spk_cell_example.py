"""The reference parser in `examples/spk_cell.py` reads every shape the production renderer writes.

The `spk` cell is written by `labels.speaker_cell` and `_speaker_cells` and never parsed by finesub
itself, so nothing else notices when a format change leaves downstream parsers behind (a tools/
parser read `(sN)`, `*` and `<` silently wrong for weeks). Each case renders with production code
and asserts what the example makes of it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from finesub.llm.stages.correction.run import _speaker_cells
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization import marks
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel, speaker_cell, speaker_of
from finesub.subtitles.stable_rows import overlay_for, write_overlay

_SPEC = importlib.util.spec_from_file_location("spk_cell", Path(__file__).resolve().parents[1] / "examples" / "spk_cell.py")
spk_cell = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(spk_cell)


def side(tracks, flag=None, alternative=None):
    return {"tracks": tracks, "flag": flag, "alternative": alternative}


def plain(track, *others):
    return SegmentSpeaker(track=track, extras={"tracks": [track, *others]})


def low(track, *others, alternative=None):
    mark = {"tier": "low", **({"alternative": alternative} if alternative is not None else {})}
    return SegmentSpeaker(track=track, review=True, extras={"tracks": [track, *others], "mark": mark})


def close(track, *others):
    return SegmentSpeaker(track=track, review=True, extras={"tracks": [track, *others]})


def filled(track):
    return SegmentSpeaker(extras={"mark": {"tier": "filled", "track": track, "by": "slot"}})


def named(name, track, **fields):
    return SegmentSpeaker(labels=(SpeakerLabel(role="guest", conf="high", name=name, src="human"),), track=track,
                          extras={"tracks": [track]}, **fields)


ACOUSTIC = [
    ([plain(2)], None, "s2", side(["s2"])),
    ([plain(2, 3)], None, "s2+s3", side(["s2", "s3"])),
    ([low(3)], None, "s3?", side(["s3"], "low")),
    ([low(3, alternative=2)], None, "s3?(s2)", side(["s3"], "low", "s2")),
    ([low(3, 1, alternative=5)], None, "s3?(s5)+s1", side(["s3", "s1"], "low", "s5")),
    ([low(3, 5, alternative=5)], None, "s3?+s5", side(["s3", "s5"], "low")),
    ([low(0, alternative=5), low(0, alternative=6)], [1.0, 1.0], "s0?", side(["s0"], "low")),
    ([close(1, 0)], None, "s1~+s0", side(["s1", "s0"], "close")),
    ([plain(0, 2), plain(1), plain(2)], [1.0, 1.0, 1.0], "s0+s2+s1", side(["s0", "s2", "s1"])),
    ([filled(4)], None, "s4*", side(["s4"], "filled")),
    ([filled(2), filled(4), filled(4)], [3.0, 2.0, 2.0], "s4*+s2*", side(["s4", "s2"], "filled")),
    ([SegmentSpeaker()], None, "", side([])),
    ([close(1)], None, "s1~", side(["s1"], "close")),
    ([named("Aki", 3)], None, "Aki", side(["Aki"])),
    ([named("Aki", 3), plain(2)], [1.0, 1.0], "Aki+s2", side(["Aki", "s2"])),
    ([SegmentSpeaker(labels=named("Aki", 3).labels, track=3, review=True,
                     extras={"tracks": [3], "mark": {"tier": "low", "alternative": 5}})], None, "Aki", side(["Aki"])),
    ([named("A+B?(x)<y|z", 3)], None, "A＋B？（x）＜y｜z", side(["A＋B？（x）＜y｜z"])),
    ([named("=1+1", 3)], None, "＝1＋1", side(["＝1＋1"])),
    ([named("s2", 3)], None, "ｓ2", side(["ｓ2"])),
    ([named(" Aki  Tanaka\n\u2028x", 3)], None, "Aki Tanaka x", side(["Aki Tanaka x"])),
    ([named("-1", 3)], None, "－1", side(["－1"])),
    ([named("@a", 3)], None, "＠a", side(["＠a"])),
    ([named(" \t ", 3)], None, "＿", side(["＿"])),
    ([named("A\x00B\x7f", 3)], None, "A B", side(["A B"])),
]


@pytest.mark.parametrize("speakers, weights, rendered, expected", ACOUSTIC, ids=[case[2] or "empty" for case in ACOUSTIC])
def test_the_example_reads_every_acoustic_shape(speakers, weights, rendered, expected):
    cell = speaker_cell(speakers, weights)
    assert cell == rendered, "the renderer changed: update this case and examples/spk_cell.py together"
    assert spk_cell.parse(cell) == {"main": expected["tracks"][0] if expected["tracks"] else None,
                                    "judged": False, "final": expected, "acoustic": expected}


def labelled(track, *others, name=None, **extra):
    coverage = {str(track): 0.9, **{str(other): 0.3 for other in others}}
    label = {"role": "unknown", "conf": "high", **({"name": name, "src": "human"} if name else {})}
    return {"track": track, "tracks": [track, *others], "block": 0, "coverage": coverage, "labels": [label], **extra}


FILLED = {"block": 0, "labels": [], "mark": {"tier": "filled", "track": 4, "by": "slot"}}


def judged(tmp_path, speakers, results, marked=True):
    """The production `spk` cells of a marked stable holding `speakers`, judged as `results` say."""
    segments = []
    for index, speaker in enumerate(speakers, start=1):
        segments.append({"id": str(index), "start": float(index), "end": index + 1.0, "text": f"line {index}",
                         **({"speaker": speaker} if speaker is not None else {})})
    stable = tmp_path / "clip-stable.json"
    lines = [{key: value for key, value in segment.items() if key != "speaker"} for segment in segments]
    stable.write_text(json.dumps({"metadata": {}, "segments": lines}), encoding="utf-8")
    write_overlay(stable, overlay_for(lines, [{"speaker": s["speaker"]} if "speaker" in s else {} for s in segments],
                                      speaker={"marks": marks.metadata() if marked else None}))
    tracks = jd.profile_tracks([speaker_of(s) for s in segments])
    window = {"id": "0001", "first": "1", "last": str(len(segments)), "status": "success",
              "key": jd.window_key(segments, jd.segment_ids(segments), tracks), "results": results}
    jd.sidecar_path(stable).write_text(json.dumps(
        {"version": jd.SIDECAR_VERSION, "route": "separate", "prompt_version": "t",
         "rows_hash": jd.rows_hash(segments), "windows": [window]}), encoding="utf-8")
    return _speaker_cells(stable)


def judged_stable(tmp_path):
    """Four rows: 1 low s2 (+s3, runner-up s5), 2 plain s3 moved to s2, 3 unassigned moved to s3, 4 plain s2."""
    return judged(tmp_path, [labelled(2, 3, review=True, mark={"tier": "low", "alternative": 5}), labelled(3), None,
                             labelled(2)],
                  [{"row": "2", "track": 2, "from": 3}, {"row": "3", "track": 3, "from": None}])


JUDGED = [
    (["2"], "s2<s3", side(["s2"]), side(["s3"])),
    (["3"], "s3<", side(["s3"]), side([])),
    (["2", "4"], "s2<s3+s2", side(["s2"]), side(["s3", "s2"])),
    # Before the judge row 2's s3 led, so row 1's rival s5 stays off the acoustic side; after it, row 1 leads.
    (["1", "2"], "s2?(s5)+s3<s3?+s2", side(["s2", "s3"], "low", "s5"), side(["s3", "s2"], "low")),
    # Row 3, moved onto s3, outweighs row 1: s5 was row 1's rival to s2, so it does not follow the judged s3.
    (["1", "3"], "s3?+s2<s2?(s5)+s3", side(["s3", "s2"], "low"), side(["s2", "s3"], "low", "s5")),
]


@pytest.mark.parametrize("ids, rendered, final, acoustic", JUDGED, ids=[case[1] for case in JUDGED])
def test_the_example_reads_every_judged_shape(tmp_path, ids, rendered, final, acoustic):
    cell = judged_stable(tmp_path)(ids)
    assert cell == rendered, "the renderer changed: update this case and examples/spk_cell.py together"
    assert spk_cell.parse(cell) == {"main": final["tracks"][0], "judged": True, "final": final, "acoustic": acoustic}


def test_the_example_reads_a_close_flag_on_the_judged_side(tmp_path):
    # Without marks a merged line's flag is decided on coverage, on both sides of `<`.
    rows = [{**labelled(0), "coverage": {"0": 1.0}}, {**labelled(2), "coverage": {"2": 1.0}}]
    cell = judged(tmp_path, rows, [{"row": "2", "track": 1, "from": 2}], marked=False)(["1", "2"])
    assert cell == "s0~+s1<s0~+s2"
    assert spk_cell.parse(cell) == {"main": "s0", "judged": True, "final": side(["s0", "s1"], "close"),
                                    "acoustic": side(["s0", "s2"], "close")}


def test_the_example_reads_a_moved_filled_line(tmp_path):
    cell = judged(tmp_path, [labelled(3), FILLED], [{"row": "2", "track": 3, "from": {"filled": 4}}])(["2"])
    assert cell == "s3<s4*"
    assert spk_cell.parse(cell)["acoustic"] == side(["s4"], "filled")


def test_a_moved_named_line_shows_the_name_on_both_sides(tmp_path):
    # Known loss, written into the grammar: a name stands for its source on either side, so the move
    # itself does not show. The track the judge chose is in the sidecar.
    cell = judged(tmp_path, [labelled(3), labelled(2, name="Aki")], [{"row": "2", "track": 3, "from": 2}])(["2"])
    assert cell == "Aki<Aki"
    assert spk_cell.parse(cell)["judged"] is True


@pytest.mark.parametrize("cell", ["s2(s5)", "s2?(s5", "s2<s3<s4", "<s3", "s2*+s3", "s2??", "s2~(s5)", "s2+", "+s2",
                                  "s1?~", "s1+s2?", "s1(s2)", " ", "Aki ", "s2*+s３*", "Aki?", "Aki~+s2"])
def test_the_example_refuses_what_the_grammar_does_not_allow(cell):
    with pytest.raises(ValueError):
        spk_cell.parse(cell)


def test_the_example_reads_a_csv_saved_with_a_bom_and_refuses_a_short_row(tmp_path):
    from finesub.llm.output_protocol import ANNOTATED_CSV_HEADER

    path = tmp_path / "clip-annotated.csv"
    path.write_text("\ufeff" + ANNOTATED_CSV_HEADER + "\nsub|1|1.0|0.0|甲|甲|high|1|s2|\n", encoding="utf-8")
    assert list(spk_cell.rows(str(path))) == [(1, "1", "s2")]
    path.write_text(ANNOTATED_CSV_HEADER + "\nsub|1|1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(spk_cell.rows(str(path)))


def test_the_example_reads_an_annotated_csv(tmp_path):
    from finesub.llm.output_protocol import ANNOTATED_CSV_HEADER

    path = tmp_path / "clip-annotated.csv"
    path.write_text(ANNOTATED_CSV_HEADER + "\nsub|1,2|1.0|0.0|甲|甲|high|1|s1<s1~+s0|n|o\nsub|3|1.0|0.0|乙|乙|high|1||\n",
                    encoding="utf-8")
    assert list(spk_cell.rows(str(path))) == [(1, "1,2", "s1<s1~+s0"), (2, "3", "")]
