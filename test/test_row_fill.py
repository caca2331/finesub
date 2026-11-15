"""The in-row fill's decisions (`speech/postprocessing/row_fill`): which rows are heard again, which pieces go in and
where. The transcriptions are given; how they are asked for is `test_recall_decode.py`'s."""

from __future__ import annotations

from collections import Counter

import pytest

from finesub.speech.postprocessing import row_fill as rf
from finesub.speech.postprocessing.recall_sounds import sounds, units
from finesub.speech.postprocessing.squeeze_repair import SOURCE_FIELD
from .test_recall_repair import _row, needs_readings

ROW = "おはようございます"
MORE = "きょうもいいてんき"


def _heard(qwen, whisper=None, language="Japanese"):
    return {"qwen": qwen, "language": language, "whisper": whisper}


def _slow_start(row):
    """`row` with its first word drawn out over a second and every confidence low: what the gate keeps."""
    words = [dict(w, confidence=0.3) for w in row["words"]]
    words[0]["end"] = words[1]["start"] = row["start"] + 1.0
    return {**row, "words": words}


def test_the_gate_keeps_a_row_with_a_drawn_out_word_and_skips_squeezed_and_stretched_rows() -> None:
    plain, slow = _row(ROW, 0.0, 1.5), _slow_start(_row(ROW, 3.0, 4.5))
    squeezed = _slow_start(_row("させていただきましょうそれでは", 6.0, 7.2))
    squeezed["words"][0]["end"] = squeezed["words"][1]["start"] = 6.01
    squeezed["end"] = 6.24
    stretched = {**_slow_start(_row(ROW, 9.0, 10.5)), SOURCE_FIELD: {"start": 9.0, "end": 9.2}}
    rows = [plain, slow, squeezed, stretched]
    speech = [{"start": r["start"], "end": r["end"]} for r in rows]
    assert rf.candidates(rows, speech) == [1]


def test_the_gate_s_constants_are_the_fitted_ones() -> None:
    # The score of one feature vector under the constants; the fitted pipeline (StandardScaler + LogisticRegression)
    # gives the same to 1e-15. A change here is a change of the accepted gate.
    row = _slow_start(_row(ROW, 0.0, 1.5))
    score = rf.gate_score(rf.gate_features(row, [(0.0, 1.5)], [(0.0, 1.5)]))
    assert score == pytest.approx(0.6308665546541536, abs=1e-12)
    assert rf.GATE_THRESHOLD == pytest.approx(0.24467164541409855, abs=1e-15)
    assert rf.AUTO_THRESHOLD == 0.5  # `--row-fill auto`'s cut


def test_a_higher_threshold_keeps_only_the_likelier_rows() -> None:
    slow = _slow_start(_row(ROW, 0.0, 1.5))  # 0.63
    middling = _row(ROW, 3.0, 4.5)  # 0.31
    middling["words"] = [dict(w, confidence=0.5) for w in middling["words"]]
    middling["words"][0]["end"] = middling["words"][1]["start"] = 3.6
    rows = [slow, middling]
    speech = [{"start": r["start"], "end": r["end"]} for r in rows]
    assert rf.candidates(rows, speech) == [0, 1] and rf.candidates(rows, speech, rf.AUTO_THRESHOLD) == [0]


@needs_readings
def test_a_piece_both_models_hear_goes_in_as_a_word_of_its_own() -> None:
    row = _row(ROW, 0.0, 1.5)
    heard = _heard(ROW + MORE, ROW + "今日もいい天気")
    assert rf.needs_whisper(row, heard, "ja")
    tally = Counter()
    out = rf.fill_rows([row], {0: heard}, "ja", tally)[0]
    assert out["text"] == ROW + MORE and (out["start"], out["end"]) == (0.0, 1.5)
    assert out["words"][-1] == {"start": 1.5, "end": 1.5, "word": MORE, rf.FIELD: True}
    assert out[rf.FIELD] == [{"text": MORE, "whisper": ROW + "今日もいい天気"}] and tally["pieces"] == 1


@needs_readings
def test_a_piece_whisper_does_not_hear_stays_out() -> None:
    row = _row(ROW, 0.0, 1.5)
    tally = Counter()
    out = rf.fill_rows([row], {0: _heard(ROW + MORE, ROW)}, "ja", tally)[0]
    assert out == row and tally["pieces_no_second_opinion"] == 1


@needs_readings
def test_a_piece_that_is_the_next_row_s_start_stays_out() -> None:
    rows = [_row(ROW, 0.0, 1.5), _row(MORE + "ですね", 1.6, 3.0)]
    tally = Counter()
    out = rf.fill_rows(rows, {0: _heard(ROW + MORE, ROW + MORE)}, "ja", tally)
    assert out == rows and tally["pieces_neighbour"] == 1


@needs_readings
def test_where_the_row_says_something_else_nothing_goes_in() -> None:
    row = _row(ROW + "さようなら", 0.0, 2.0)
    tally = Counter()
    rf.fill_rows([row], {0: _heard(ROW + MORE, ROW + MORE)}, "ja", tally)
    assert tally["pieces_substitution"] == 1
    assert not rf.needs_whisper(row, _heard(ROW + MORE), "ja")


def test_a_referee_text_in_another_language_leaves_the_row_alone() -> None:
    row = _row(ROW, 0.0, 1.5)
    tally = Counter()
    assert rf.fill_rows([row], {0: _heard(ROW + MORE, ROW + MORE, "Chinese")}, "ja", tally)[0] == row
    assert tally["rows_other_language"] == 1


@needs_readings
def test_without_words_that_join_to_the_text_the_piece_goes_into_the_text_alone() -> None:
    row = {**_row(ROW, 0.0, 1.5), "words": []}
    out = rf.fill_rows([row], {0: _heard(MORE + ROW, MORE + ROW)}, "ja")[0]
    assert out["text"] == MORE + ROW and out["words"] == []


@needs_readings
def test_a_piece_that_needs_whisper_s_text_refuses_to_run_without_it() -> None:
    with pytest.raises(ValueError, match="Whisper"):
        rf.fill_rows([_row(ROW, 0.0, 1.5)], {0: _heard(ROW + MORE)}, "ja")


def _english(text, start, end):
    words = text.split()
    step = (end - start) / len(words)
    return {"start": start, "end": end, "text": text, "confidence": 0.9,
            "words": [{"start": start + i * step, "end": start + (i + 1) * step, "word": w, "confidence": 0.9}
                      for i, w in enumerate(words)]}


@needs_readings
def test_units_are_words_where_words_are_spaced_and_sounds_elsewhere() -> None:
    assert units('"Well, I don\'t." OK', "en") == [("well", 1, 5), ("i", 7, 8), ("dont", 9, 14), ("ok", 17, 19)]
    assert [u for u, _, _ in units("Grab stuff.Grab", "en")] == ["grab", "stuff", "grab"]
    assert units("今日", "ja") == [(s, i, i + 1) for s, i in sounds("今日", "ja")]


def test_in_english_a_piece_is_whole_words_and_goes_in_between_spaces() -> None:
    row = _english("I think we should go,", 0.0, 2.0)
    heard = {"qwen": "I think that we should go home.", "language": "English",
             "whisper": " I think that we should go."}
    tally = Counter()
    out = rf.fill_rows([row], {0: heard}, "en", tally)[0]
    # "home" is the referee's alone (no second opinion); "that" goes in as a word, spaced, punctuation kept
    assert out["text"] == "I think that we should go,"
    assert [w["word"] for w in out["words"]] == ["I", "think", "that", "we", "should", "go,"]
    assert out[rf.FIELD] == [{"text": "that", "whisper": heard["whisper"]}]
    assert tally["pieces"] == 1 and tally["pieces_no_second_opinion"] == 1


def test_in_english_a_one_word_row_can_be_matched_and_a_text_only_piece_goes_after_the_punctuation() -> None:
    # one shared word is a block in English (WORD_BLOCK), so a short row is not unmatched by construction
    row = {**_english("Okay.", 0.0, 1.0), "words": []}
    heard = {"qwen": "Okay, now please.", "language": "English", "whisper": " Okay, now please."}
    out = rf.fill_rows([row], {0: heard}, "en")[0]
    assert out["text"] == "Okay. now please" and out[rf.FIELD][0]["text"] == "now please"


@needs_readings
def test_where_words_are_not_spaced_units_and_their_cover_are_the_sounds_and_their_cover() -> None:
    from finesub.speech.postprocessing import recall_sounds as rs

    for lang, a, b in (("ja", "今日はいい天気ですね", "きょうはいいてんきだね"), ("zh", "今天天氣很好", "今天天气不错")):
        assert rs.units(a, lang) == [(s, i, i + 1) for s, i in rs.sounds(a, lang)]
        assert rs.unit_cover(a, b, lang) == rs.cover(a, b, lang)
        ua, ub = ([u for u, _, _ in rs.units(x, lang)] for x in (a, b))
        assert rs.unit_pairs(ua, ub, lang) == rs.pairs(ua, ub)


def test_in_english_fillers_come_off_a_piece_s_ends_by_word_not_by_letter() -> None:
    from finesub.speech.postprocessing.recall_sounds import content_words

    assert [content_words(x, "en") for x in ("look", "eyes", "Yeah, okay", "Oh, my eyes, okay", "uh huh", "go right")] == \
        ["look", "eyes", "", "myeyes", "", "goright"]


def test_in_english_a_head_piece_and_a_middle_piece_go_in_through_the_words() -> None:
    row = _english("we should go now", 1.0, 3.0)
    said = "Well we should go back home now."
    out = rf.fill_rows([row], {0: {"qwen": said, "language": "English", "whisper": " " + said}}, "en")[0]
    assert out["text"] == "Well we should go back home now"
    assert [w["word"] for w in out["words"]] == ["Well", "we", "should", "go", "back home", "now"]
    assert out["words"][0]["start"] == 1.0 and [p["text"] for p in out[rf.FIELD]] == ["Well", "back home"]


def test_a_piece_the_squeezed_row_before_already_says_stays_out_wherever_it_lies() -> None:
    # the row before is squeezed (its speech is past its span, here in this row's): a tail piece that repeats it is
    # that speech written twice; the same row before at its full length leaves the piece in
    said = "Stand up please. Stand up please."
    heard = {"qwen": said, "language": "English", "whisper": " " + said}
    row = _english("Stand up please", 1.0, 3.0)
    tally = Counter()
    squeezed = _english("Stand up please.", 0.9, 0.95)
    assert rf.fill_rows([squeezed, row], {1: heard}, "en", tally)[1] == row and tally["pieces_neighbour"] == 1
    spoken = _english("Stand up please.", -1.0, 0.9)
    assert rf.fill_rows([spoken, row], {1: heard}, "en")[1]["text"] == "Stand up please Stand up please"


def test_a_piece_keeps_its_vowel_and_tone_signs_and_goes_in_after_the_whole_letter() -> None:
    # Thai and Devanagari signs follow their letter and normalising drops them: a cut or an insertion by the bare
    # letters would part them
    for lang, name, row_text, said, want in (
            ("th", "Thai", "สวัสดีที่นี่สนุกมาก", "สวัสดีวันนี้ที่นี่สนุกมาก", "สวัสดีวันนี้ที่นี่สนุกมาก"),
            ("hi", "Hindi", "मैं ठीक हूँ", "मैं ठीक हूँ दोस्तों", "मैं ठीक हूँ दोस्तों")):
        row = {"start": 0.0, "end": 2.0, "text": row_text, "words": [], "confidence": 0.9}
        heard = {"qwen": said, "language": name, "whisper": " " + said}
        assert rf.fill_rows([row], {0: heard}, lang)[0]["text"] == want


def test_a_middle_piece_the_squeezed_row_before_says_stays_out_too() -> None:
    said = "I think stand up please we go."
    heard = {"qwen": said, "language": "English", "whisper": " " + said}
    row = _english("I think we go", 1.0, 3.0)
    tally = Counter()
    assert rf.fill_rows([_english("Stand up please.", 0.9, 0.95), row], {1: heard}, "en", tally)[1] == row
    assert tally["pieces_neighbour"] == 1
