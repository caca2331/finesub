"""`subtitles.stable_rows`: the one reader of a stable's lines after diarization -- `-stable.json` joined
with `-diar-rows.json` -- and the guards that keep every consumer on it.

A consumer that reads the stable's `segments` instead of `read_rows` misses every speaker, sees one line
fewer per cut, and every id after the first cut points at the wrong line, silently. Two guards: each known
consumer must give the same answer on a stable with its `-diar-rows.json` as on the same lines written out
plainly (the behaviour guard), and no module in the consumer scope may name a stable's `"segments"` outside
the reader (the text guard).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.subtitles.stable_rows import (
    halves, lines_hash, overlay_for, read_overlay, read_rows, rows, write_overlay,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "finesub"


def word(text, start, end):
    return {"word": text, "start": start, "end": end}


def speaker(track):
    return {"labels": [], "block": 0, "track": track, "tracks": [track], "coverage": {str(track): 1.0}}


def stable():
    """Two lines as stabilization writes them."""
    first = {"start": 0.0, "end": 1.6, "text": "あいうえおかきくけこさし", "lang": "ja",
             "words": [word("あいう", 0.0, 0.4), word("えおか", 0.4, 0.8), word("きくけ", 0.8, 1.2), word("こさし", 1.2, 1.6)],
             "alignment_events": [{"start": 0.1}]}
    second = {"start": 2.0, "end": 3.0, "text": "たちつてと", "lang": "ja", "words": [word("たちつてと", 2.0, 3.0)]}
    return {"metadata": {"asr": {}}, "segments": [first, second]}


def overlay(chosen=True):
    """Its `-diar-rows.json`: the first line is cut before its third word (a cut on the first line shifts
    every later id)."""
    entries = [{"speaker": speaker(5), "split": {"at": 2, "speakers": [speaker(5), speaker(6)]}},
               {"speaker": speaker(6)}]
    return overlay_for(stable()["segments"], entries, speaker={"marks": None},
                       split={"chosen": chosen, "version": 1, "cuts": 1})


def plain():
    """The same lines as `rows(stable(), overlay())`, written out as an ordinary stable with no rows file."""
    return {"metadata": {"asr": {}}, "segments": rows(stable(), overlay())}


# --- the reader ------------------------------------------------------------------------------------------------

def test_chosen_split_lines_read_as_their_halves_with_their_own_speakers():
    lines = rows(stable(), overlay())
    assert [line["text"] for line in lines] == ["あいうえおか", "きくけこさし", "たちつてと"]
    assert [(line["start"], line["end"]) for line in lines[:2]] == [(0.0, 0.8), (0.8, 1.6)]
    assert [line["speaker"]["track"] for line in lines] == [5, 6, 6]
    assert all("split" not in line and "alignment_events" not in line for line in lines[:2])
    assert lines[0]["lang"] == lines[1]["lang"] == "ja"


def test_unchosen_lines_read_whole_with_their_speakers():
    lines = rows(stable(), overlay(chosen=False))
    assert [line["text"] for line in lines] == [line["text"] for line in stable()["segments"]]
    assert [line["speaker"]["track"] for line in lines] == [5, 6]
    assert all("split" not in line for line in lines)


def test_without_rows_made_for_these_lines_the_lines_read_as_they_are():
    assert rows(stable(), None) == stable()["segments"]
    redone = stable()
    redone["segments"][1]["end"] = 3.2
    assert rows(redone, overlay()) == redone["segments"], "made for other lines: not read at all"
    fixed = stable()
    fixed["segments"][1]["text"] = "立ち続けて"
    assert rows(fixed, overlay())[2]["speaker"]["track"] == 6, "a hand-fixed text keeps the labels"
    shorter = overlay()
    shorter["rows"].pop()
    assert rows(stable(), shorter) == stable()["segments"]


def test_an_entry_without_a_speaker_leaves_the_line_unlabelled():
    labelled = stable()
    labelled["segments"][1]["speaker"] = speaker(9)  # whatever a stable carried, the rows decide
    entries = [{"speaker": speaker(5)}, {}]
    lines = rows(labelled, overlay_for(labelled["segments"], entries))
    assert lines[0]["speaker"]["track"] == 5 and "speaker" not in lines[1]


def test_a_half_no_track_claimed_has_no_speaker():
    made = overlay()
    made["rows"][0]["split"]["speakers"][1] = None
    assert "speaker" not in rows(stable(), made)[1]


def test_the_reader_leaves_its_input_alone_and_reads_bare_lists():
    payload, made = stable(), overlay()
    before = json.dumps([payload, made], sort_keys=True)
    rows(payload, made)
    assert json.dumps([payload, made], sort_keys=True) == before
    assert rows([{"text": "x"}], None) == [{"text": "x"}]
    left, right = halves({**stable()["segments"][0], "speaker": speaker(5)}, 2)
    assert left["speaker"]["track"] == right["speaker"]["track"] == 5, "speaker is the caller's to replace"


def test_the_rows_file_is_found_beside_the_stable_and_checked_against_it(tmp_path):
    path = tmp_path / "clip-stable.json"
    path.write_text(json.dumps(stable(), ensure_ascii=False), encoding="utf-8")
    assert read_overlay(path) is None
    assert write_overlay(path, overlay()) == tmp_path / "clip-diar-rows.json"
    assert read_overlay(path)["stable"] == lines_hash(stable()["segments"])
    assert read_rows(path) == rows(stable(), overlay())
    (tmp_path / "clip-diar-rows.json").write_text("{not json", encoding="utf-8")
    assert read_overlay(path) is None and read_rows(path) == stable()["segments"]


def test_an_unreadable_rows_file_says_so_once(tmp_path, capsys):
    path = tmp_path / "clip-stable.json"
    path.write_text(json.dumps(stable(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "clip-diar-rows.json").write_text("{not json", encoding="utf-8")
    read_rows(path)
    read_rows(path)
    assert capsys.readouterr().err.count("Warning: clip-diar-rows.json unreadable") == 1
    redone = overlay()
    redone["stable"] = "0" * 16
    write_overlay(path, redone)
    read_rows(path)
    assert capsys.readouterr().err == "", "a file made for other lines is expected, not a warning"


# --- the behaviour guard: every known consumer reads the same lines either way ------------------------------------

@pytest.fixture
def both(tmp_path):
    a, b = tmp_path / "a" / "clip-stable.json", tmp_path / "b" / "clip-stable.json"
    for path, payload in ((a, stable()), (b, plain())):
        path.parent.mkdir()
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    write_overlay(a, overlay())
    return a, b


def test_the_llm_stage_reads_the_same_lines(both):
    from finesub.llm.chunking import load_segments_from_stable_json

    a, b = (load_segments_from_stable_json(path) for path in both)
    assert [(s.id, s.start, s.end, s.text) for s in a] == [(s.id, s.start, s.end, s.text) for s in b]


def test_the_judge_and_its_checks_read_the_same_lines(both):
    from finesub.speech.diarization import judgments

    a, b = (read_rows(path) for path in both)
    assert a == b
    assert judgments.rows_hash(a) == judgments.rows_hash(b)
    assert judgments.covers(both[0]) == judgments.covers(both[1])


def test_the_spk_column_reads_the_same_lines(both):
    from finesub.llm.stages.correction.run import _speaker_cells

    a, b = (_speaker_cells(path) for path in both)
    for ids in (["1"], ["2"], ["3"], ["1", "2"], ["2", "3"]):
        assert a(ids) == b(ids), ids


def test_the_raw_srt_reads_the_same_lines(both, tmp_path):
    from finesub.subtitles.rendering import convert_json_to_srt

    a, b = (convert_json_to_srt(path, output_path=tmp_path / f"{i}.srt", validate=False).path.read_text(encoding="utf-8")
            for i, path in enumerate(both))
    assert a == b and a.count("-->") == 3


def test_the_speaker_exports_read_the_same_lines(both):
    from finesub.stages import _run_speaker_export_stage

    out = []
    for path in both:
        _run_speaker_export_stage(SimpleNamespace(stable_json=path), speaker_tracks="srt", empty_cues=False, announce=lambda *a, **k: None)
        folder = path.with_name("clip-speakers")
        out.append({p.name: p.read_text(encoding="utf-8") for p in sorted(folder.iterdir())})
    assert out[0] == out[1] and out[0]


# --- the text guard ----------------------------------------------------------------------------------------------

#: Where a stable's lines are read after diarization. A module here naming `"segments"` reads them past the reader,
#: unless it is listed below with what its `"segments"` is.
SCOPE = ("llm", "subtitles", "stages.py", "pipeline.py", "speech/diarization/judgments.py",
         "speech/diarization/labels.py", "speech/diarization/export.py")
ALLOWED = {
    "subtitles/stable_rows.py": None,  # the reader
    "llm/agent/agent_validators.py": 2,  # an agent's window payload, not a stable
    "llm/agent/local_agent.py": 1,  # a recovered agent answer
    "llm/exchange_log.py": 1,  # agy's recovery record
    "llm/token_measure.py": 1,  # a table header
    "subtitles/rendering.py": 1,  # the error message for a payload without lines
    "stages.py": 1,  # the timeline's segment count in a debug record
}


def test_no_consumer_reads_a_stable_s_lines_past_the_reader():
    found = {}
    for entry in SCOPE:
        root = SRC / entry
        for path in ([root] if root.is_file() else sorted(root.rglob("*.py"))):
            count = len(re.findall(r"""["']segments["']""", path.read_text(encoding="utf-8")))
            if count:
                found[path.relative_to(SRC).as_posix()] = count
    unexpected = {name: count for name, count in found.items()
                  if name not in ALLOWED or (ALLOWED[name] is not None and count > ALLOWED[name])}
    assert not unexpected, f"read the lines through subtitles.stable_rows.rows, or list why here: {unexpected}"


def test_explicit_ids_go_positional_when_halves_are_read():
    payload = stable()
    for k, line in enumerate(payload["segments"]):
        line["id"] = f"x{k}"
    made = overlay_for(payload["segments"], overlay()["rows"], split={"chosen": True})
    assert all("id" not in line for line in rows(payload, made)), "a half has no id; the rest must not keep theirs"
    made["split"]["chosen"] = False
    assert [line["id"] for line in rows(payload, made)] == ["x0", "x1"]


def test_a_payload_without_lines_is_not_read_as_an_empty_stable():
    assert rows({"metadata": {}}, None) is None
