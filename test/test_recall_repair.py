"""The recall repair's decisions (docs/asr-stabilize.md, "补救：坍缩行与空档"). The rules themselves were replayed
against the accepted evaluation arms row for row; these pin the contracts around them."""

from __future__ import annotations

import pytest

from finesub.speech.postprocessing import recall_repair as rr
from finesub.speech.postprocessing import squeeze_repair
from finesub.speech.postprocessing.recall_sounds import readings_available
from finesub.speech.postprocessing.segmentation import WHISPER_SEGMENT_WORD_TAG
from finesub.subtitles import line_groups

#: The reading libraries (pykakasi, pypinyin, opencc) are `[asr]` dependencies; the CI lanes install
#: `[harness,dev]` only. The rules import them only when they compare readings, so only the tests that reach a
#: comparison need them -- the rest of the file, and of the files importing this mark, still runs.
needs_readings = pytest.mark.skipif(not readings_available(), reason="needs the reading libraries (the [asr] extra)")


def _row(text: str, start: float, end: float) -> dict:
    step = (end - start) / len(text)
    words = [{"start": start + i * step, "end": start + (i + 1) * step, "word": c, "confidence": 0.9}
             for i, c in enumerate(text)]
    words[0][WHISPER_SEGMENT_WORD_TAG] = True
    return {"start": start, "end": end, "text": text, "words": words, "confidence": 0.9}


def _payload(rows, speech) -> dict:
    return {"segments": rows, "vad_timeline": {"intervals": [{"start": a, "end": b} for a, b in speech]},
            "metadata": {"asr_align": {"model": "test"}}}


def test_whisper_is_asked_only_where_the_referee_heard_content_in_the_language() -> None:
    gap = {"qwen": "今日はいい天気ですね", "language": "Japanese"}
    assert rr.needs_whisper(gap, "ja")
    assert not rr.needs_whisper({**gap, "language": "Chinese"}, "ja")
    assert not rr.needs_whisper({**gap, "qwen": "うん"}, "ja")
    # A gap Whisper was not asked about is never heard: its text is not read.
    assert not rr.heard({"qwen": "うん", "language": "Japanese", "whisper": None}, "ja")


@needs_readings
def test_a_heard_gap_nobody_owns_becomes_a_line_without_words() -> None:
    rows = [_row("おはようございます", 0.0, 1.5)]
    gap = {"start": 10.0, "end": 12.0, "qwen": "今日はいい天気ですね", "language": "Japanese",
           "whisper": "今日はいい天気ですね"}
    out = rr.fill_gaps(rows, [gap], [], "ja", [(0.0, 1.5), (10.0, 12.0)])
    assert out[0] == rows[0]
    assert out[1] == {"start": 10.0, "end": 12.0, "text": "今日はいい天気ですね", "words": [], "gap_fill": True,
                      "whisper": "今日はいい天気ですね"}


@needs_readings
def test_disagreeing_models_fill_nothing() -> None:
    rows = [_row("おはようございます", 0.0, 1.5)]
    gap = {"start": 10.0, "end": 12.0, "qwen": "今日はいい天気ですね", "language": "Japanese",
           "whisper": "全然違う話をしている"}
    assert rr.fill_gaps(rows, [gap], [], "ja", [(10.0, 12.0)]) == rows


def test_any_language_the_referee_names_is_asked_about_and_an_unnamed_one_never() -> None:
    assert rr.needs_whisper({"qwen": "bonjour tout le monde", "language": "French"}, "fr")
    assert not rr.needs_whisper({"qwen": "bonjour tout le monde", "language": None}, "fr")
    assert not rr.needs_whisper({"qwen": "bonjour tout le monde", "language": "French"}, "de")


def test_the_episode_language_is_the_asr_steps_else_the_rows_by_content() -> None:
    rows = [{**_row("うん", 0.0, 1.0), "lang": "ja"}, {**_row("今天天气很好", 1.0, 2.0), "lang": "zh"}]
    assert rr.episode_language({"metadata": {"asr_align": {"language": "ja"}}}, rows) == "ja"
    assert rr.episode_language({"metadata": {"asr_align": {}}}, rows) == "zh"  # うん is a filler: no content
    tie = [{**_row("今日はいい", 0.0, 1.0), "lang": "ja"}, {**_row("今天很好", 1.0, 2.0), "lang": "zh"}]
    assert rr.episode_language({}, tie) == "ja"  # 4 content characters each: the first seen
    assert rr.episode_language({}, [_row("あいう", 0.0, 1.0)]) is None
    # Detected, not told: the ASR step writes the string "None", on the run and on rows it could not tell.
    assert rr.episode_language({"metadata": {"asr_align": {"language": "None"}}}, rows) == "zh"
    assert rr.episode_language({}, [{**_row("今日はいい天気", 0.0, 1.0), "lang": "None"}, rows[1]]) == "zh"


def test_gaps_are_the_uncovered_speech_of_at_least_a_second() -> None:
    payload = _payload([_row("あいうえお", 0.0, 2.0)], [(0.0, 3.5), (5.0, 5.5)])
    assert rr.gaps(payload) == [(2.0, 3.5)]


@needs_readings
def test_without_a_window_heard_a_squeezed_row_stays() -> None:
    squeezed = _row("させていただきましょうそれでは", 10.0, 10.24)
    assert squeeze_repair.is_squeezed(squeezed)
    payload = _payload([squeezed, _row("次です", 20.0, 21.0)], [(10.0, 15.0), (20.0, 21.0)])
    windows = rr.squeeze_windows(payload)
    assert {w["kind"] for w in windows} >= {"gate", "wide"}
    rows, verdicts = rr.repair_squeezed(payload, [{**w, "text": ""} for w in windows], "ja")
    assert [v["branch"] for v in verdicts] == ["suspect"]
    assert rows[0]["end"] == 10.24


@needs_readings
def test_a_squeezed_row_the_referee_hears_after_it_is_stretched() -> None:
    text = "させていただきましょうそれでは"
    payload = _payload([_row(text, 10.0, 10.24), _row("次です", 20.0, 21.0)], [(10.0, 15.0), (20.0, 21.0)])
    windows = rr.squeeze_windows(payload)
    heard = [{**w, "text": text if w["kind"] in ("gate", "wide") else ""} for w in windows]
    rows, verdicts = rr.repair_squeezed(payload, heard, "ja")
    assert verdicts[0]["branch"] == "stretch"
    assert rows[0]["end"] > 11.0 and rows[0][squeeze_repair.SOURCE_FIELD] == {"start": 10.0, "end": 10.24}
    assert line_groups.LANE_FIELD not in rows[0]


@needs_readings
def test_a_row_stretched_over_the_next_one_stays_over_it_on_the_overlap_lane() -> None:
    text = "させていただきましょうそれでは"
    payload = _payload([_row(text, 10.0, 10.24), _row("うんうん", 10.24, 11.0), _row("次です", 20.0, 21.0)],
                       [(10.0, 15.0), (20.0, 21.0)])
    windows = rr.squeeze_windows(payload)
    heard = [{**w, "text": text if w["kind"] in ("overlap", "wide") else ""} for w in windows]
    stretched, verdicts = rr.repair_squeezed(payload, heard, "ja")
    assert verdicts[0]["branch"] == "overlap"
    assert rr.overlap_sources(verdicts) == {(10.0, 10.24)}
    assert not any(line_groups.LANE_FIELD in r for r in stretched)  # the stretch alone: the ladder's `S` reads it
    rows = rr.shown(stretched, rr.overlap_sources(verdicts), payload["vad_timeline"]["intervals"])
    assert rows[0][line_groups.LANE_FIELD] == "10.0-10.24" and rows[0]["end"] > rows[1]["end"]
    assert rows[1:] == payload["segments"][1:]  # the row it lies over keeps its time and stays off the lane


@needs_readings
def test_the_stretch_keeps_the_whole_overlap_for_the_ladder() -> None:
    # a row running past the stretch: `repair_squeezed` leaves both whole (the ladder's `S` is the full stretch,
    # cut only once, by the display after the pick); `shown` is what cuts them
    text = "させていただきましょうそれでは"
    payload = _payload([_row(text, 10.0, 10.24), _row("次の人の長い話です", 10.24, 16.0)], [(10.0, 16.0)])
    windows = rr.squeeze_windows(payload)
    heard = [{**w, "text": text if w["kind"] in ("overlap", "wide") else ""} for w in windows]
    stretched, verdicts = rr.repair_squeezed(payload, heard, "ja")
    assert verdicts[0]["branch"] == "overlap"
    reach = max(r["end"] for r in stretched if r.get(squeeze_repair.SOURCE_FIELD))
    assert reach > 11.5 and stretched[-1] == payload["segments"][1]
    rows = rr.shown(stretched, rr.overlap_sources(verdicts), payload["vad_timeline"]["intervals"])
    assert max(r["end"] for r in rows if r.get(squeeze_repair.SOURCE_FIELD)) < reach


def _moved_piece(start: float, end: float) -> dict:
    return {**_row("させていただきましょうそれでは", start, end), squeeze_repair.SOURCE_FIELD: {"start": 10.0, "end": 10.24}}


def test_a_row_ending_inside_the_moved_piece_stays_under_it_on_the_lane() -> None:
    piece, inside = _moved_piece(10.0, 13.0), _row("うんうん", 11.0, 12.0)
    out = rr.shown([piece, inside], {(10.0, 10.24)}, [{"start": 10.0, "end": 15.0}])
    assert out[0][line_groups.LANE_FIELD] == "10.0-10.24" and (out[0]["start"], out[0]["end"]) == (10.0, 13.0)
    assert out[1] == inside


def test_a_row_running_past_the_moved_piece_is_cut_at_the_midpoint_or_the_pad_start() -> None:
    vad = [{"start": 10.0, "end": 15.0}]
    piece, past = _moved_piece(10.0, 13.0), _row("そうですねほんとに", 11.0, 15.0)
    out = rr.shown([piece, past], {(10.0, 10.24)}, vad)
    assert [(r["start"], r["end"]) for r in out] == [(10.0, 12.0), (12.0, 15.0)]
    assert not any(line_groups.LANE_FIELD in r for r in out)
    assert out[1]["words"][0]["start"] >= 12.0  # the cut row's words are laid over its new span
    out = rr.shown([piece, past], {(10.0, 10.24)}, vad, lambda b, s: 11.6)
    assert [(r["start"], r["end"]) for r in out] == [(10.0, 11.6), (11.6, 15.0)]
    out = rr.shown([piece, past], {(10.0, 10.24)}, vad, lambda b, s: 16.0)  # outside the row: the midpoint
    assert [(r["start"], r["end"]) for r in out] == [(10.0, 12.0), (12.0, 15.0)]
    # Within INSIDE of the piece's end counts as running past it (r1#36's 0.05 s edge).
    out = rr.shown([piece, _row("そうですね", 11.0, 12.96)], {(10.0, 10.24)}, vad)
    assert [(r["start"], r["end"]) for r in out] == [(10.0, 11.98), (11.98, 12.96)]


def test_a_cut_that_would_leave_a_row_flashing_by_keeps_the_two_overlapped() -> None:
    # The pad start falls 0.1 s into the moved row: cut there, its 15 characters would be squeezed again.
    piece, past = _moved_piece(10.0, 13.0), _row("そうですねほんとに", 10.05, 15.0)
    out = rr.shown([piece, past], {(10.0, 10.24)}, [{"start": 10.0, "end": 15.0}], lambda b, s: 10.1)
    assert out[0][line_groups.LANE_FIELD] == "10.0-10.24" and (out[0]["end"], out[1]["start"]) == (13.0, 10.05)
    # The pad start falls 0.1 s before the covered row's end: cut there, its 9 characters would flash by.
    piece, past = _moved_piece(10.0, 13.0), _row("そうですねほんとに", 11.0, 13.2)
    out = rr.shown([piece, past], {(10.0, 10.24)}, [{"start": 10.0, "end": 15.0}], lambda b, s: 13.1)
    assert out[0][line_groups.LANE_FIELD] == "10.0-10.24" and out[1] == past


def test_inside_is_measured_against_the_whole_moved_row_not_its_piece() -> None:
    # B starts in the first piece and ends past it, but well before the row's last piece ends: under the row
    first, last, b = _moved_piece(10.0, 12.0), _moved_piece(12.0, 15.0), _row("うんうん", 11.0, 13.0)
    out = rr.shown([first, b, last], {(10.0, 10.24)}, [{"start": 10.0, "end": 15.0}])
    assert [(r["start"], r["end"]) for r in out] == [(10.0, 12.0), (11.0, 13.0), (12.0, 15.0)]
    assert out[0][line_groups.LANE_FIELD] == "10.0-10.24" and line_groups.LANE_FIELD not in out[1]


def test_two_moved_rows_over_each_other_each_keep_their_span_on_screen() -> None:
    # Two neighbouring collapsed rows moved over the same stretch: one lane each, so the SRT trims neither
    from finesub.subtitles import rendering

    # (Whisper may have overlapped the two: one start, two ends)
    first = {**_row("させていただきましょうそれでは", 10.3, 13.0), squeeze_repair.SOURCE_FIELD: {"start": 10.0, "end": 10.24}}
    second = {**_row("今日はいい天気ですね", 10.4, 12.5), squeeze_repair.SOURCE_FIELD: {"start": 10.0, "end": 10.48}}
    out = rr.shown([first, second, _row("うんうん", 11.0, 12.0)], {(10.0, 10.24), (10.0, 10.48)},
                   [{"start": 10.0, "end": 15.0}])
    assert [r.get(line_groups.LANE_FIELD) for r in out] == ["10.0-10.24", "10.0-10.48", None]
    assert [c[:2] for c in rendering.segment_cues(out)] == [(10.3, 13.0), (10.4, 12.5), (11.0, 12.0)]


def test_with_nothing_moved_the_rows_come_back_as_they_were() -> None:
    rows = [_row("今日はいい天気", 10.0, 12.0), _row("そうですね", 11.0, 13.0)]
    assert rr.shown(rows, set(), [{"start": 10.0, "end": 13.0}]) == rows


@needs_readings
def test_the_ladder_cuts_where_whisper_s_pad_decode_starts_the_row_it_lies_over() -> None:
    from finesub.speech.postprocessing import collapse_ladder

    text = "させていただきましょうそれでは"
    rows = [_row("前の行です", 8.0, 9.5), _row(text, 10.0, 10.24), _row("うんうんそうですね", 11.0, 14.0)]
    payload = _payload(rows, [(8.0, 9.5), (10.0, 16.0)])
    window = {"start": 10.0, "end": 16.0, "segments": [{"words": [
        {"word": text, "start": 10.0, "end": 11.0}, {"word": "うんうんそうですね", "start": 12.6, "end": 14.0}]}]}
    out = collapse_ladder.moved(payload, rows, {1: (10.0, 13.0)}, {("pad", 1): window}, "ja")
    assert [(r["start"], r["end"]) for r in out[1:]] == [(10.0, 12.6), (12.6, 14.0)]
    out = collapse_ladder.moved(payload, rows, {1: (10.0, 13.0)})  # no decode: the midpoint
    assert [(r["start"], r["end"]) for r in out[1:]] == [(10.0, 12.0), (12.0, 14.0)]


def test_every_piece_of_a_moved_row_lying_over_the_next_one_is_on_the_lane() -> None:
    # The ladder's move (`collapse_ladder.moved`) sends the row through the splitter alone, which may cut it inside
    # the row it now lies over: both pieces overlap that row, so both go on the lane and that row keeps its time.
    from finesub.speech.postprocessing import collapse_ladder

    text = "させていただきましょうそれでは。今日はいい天気ですね本当に"
    rows = [_row("前の行です", 8.0, 9.5), _row(text, 10.0, 10.24), _row("うんうん", 11.0, 14.0), _row("次です", 20.0, 21.0)]
    payload = _payload(rows, [(8.0, 9.5), (10.0, 12.0), (13.5, 16.0), (20.0, 21.0)])
    out = collapse_ladder.moved(payload, rows, {1: (10.0, 16.0)})
    moved = [r for r in out if squeeze_repair.SOURCE_FIELD in r]
    assert len(moved) == 2 and all(r[line_groups.LANE_FIELD] for r in moved)
    assert [r for r in out if squeeze_repair.SOURCE_FIELD not in r] == [rows[0], rows[2], rows[3]]
    # A span starting inside the next row (the ladder's `crossed`): the row still lies over it.
    out = collapse_ladder.moved(payload, rows, {1: (11.5, 13.9)})
    moved = [r for r in out if squeeze_repair.SOURCE_FIELD in r]
    assert moved and all(r[line_groups.LANE_FIELD] for r in moved)
    assert [r for r in out if squeeze_repair.SOURCE_FIELD not in r] == [rows[0], rows[2], rows[3]]
