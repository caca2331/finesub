"""Scribe v3's answer as units (`speech/joint/scribe.py`) and the Chinese clause evidence on the word DP
(`speech/joint/lines.py`, `clause_ends`); what they fix: `docs/joint-asr.md`. The collapse's re-ask is tested with
the other repairs (`test_speech_joint_repair.py`), and so is a glued word's."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.speech.joint import lines as L
from finesub.speech.joint import scribe as S
from finesub.speech.postprocessing import segmentation as seg

PARAMS = seg.split_params_for_length_scale(1.0)


def word(text, start, end, kind="word", speaker="speaker_0", chars=None):
    w = {"text": text, "start": start, "end": end, "type": kind, "speaker_id": speaker}
    if chars:
        w["characters"] = chars
    return w


def test_a_mark_glued_to_the_next_word_goes_back_and_the_rest_starts_at_its_own_character():
    glued = word("。26", 335.86, 340.44, chars=[{"text": "。", "start": 335.86, "end": 335.86},
                                              {"text": "2", "start": 339.98, "end": 340.16},
                                              {"text": "6", "start": 340.3, "end": 340.44}])
    units, _ = S.answer_units([word("珂", 335.6, 335.86), glued])
    assert [(u["text"], u["start"], u["end"]) for u in units] == [
        ("珂", 335.6, 335.86), ("。", 335.86, 335.86), ("26", 339.98, 340.44)]
    words = L.dp_words(units, duration=400)
    assert [w["word"] for w in words] == ["珂。", "26"]
    assert words[1]["start"] - words[0]["end"] > 4


def test_an_opening_quote_stays_on_its_word_but_takes_no_time():
    quoted = word('"Hello', 9.0, 10.0, chars=[{"text": '"', "start": 9.0, "end": 9.0},
                                            {"text": "H", "start": 9.6, "end": 9.7},
                                            {"text": "o", "start": 9.9, "end": 10.0}])
    units, _ = S.answer_units([word("said.", 0.5, 1.0), quoted])
    assert [(u["text"], u["start"], u["end"]) for u in units] == [("said.", 0.5, 1.0), ('"Hello', 9.6, 10.0)]
    assert [w["word"] for w in L.dp_words(units, duration=20)] == ["said.", '"Hello']


def test_a_trailing_mark_timed_after_a_pause_does_not_stretch_its_word():
    what = word("What？", 1602.718, 1614.198, chars=[{"text": "W", "start": 1602.718, "end": 1602.758},
                                                    {"text": "t", "start": 1602.998, "end": 1602.998},
                                                    {"text": "？", "start": 1614.138, "end": 1614.198}])
    units, _ = S.answer_units([what])
    assert [(u["text"], u["start"], u["end"]) for u in units] == [("What？", 1602.718, 1602.998)]


def test_a_bracketed_note_is_dropped_and_a_mark_after_it_kept():
    note = [word(ch, 10.0 + k * 0.1, 10.0 + k * 0.1) for k, ch in enumerate("（聞き取り不能")]
    answer = {"words": [word("る", 9.8, 9.9), *note, word("）。", 10.8, 10.8), word("だ", 11.0, 11.1)]}
    words, notes = S.without_notes(answer["words"])
    assert notes == 1
    assert [w["text"] for w in words] == ["る", "。", "だ"]
    units, _ = S.answer_units(answer["words"])
    assert "".join(u["text"] for u in units) == "る。だ"


def test_an_unclosed_bracket_is_kept():
    words = [word("（", 1.0, 1.0)] + [word("あ", 1.0 + k, 1.1 + k) for k in range(20)]
    assert S.without_notes(words) == (words, 0)


def test_a_collapse_is_a_long_zero_length_run_strung_over_many_seconds():
    moved = [word("字", 517.0 + 2.0 * k, 517.0 + 2.0 * k) for k in range(21)]
    burst = [word("え？", 160.0 + 0.3 * k, 160.0 + 0.3 * k) for k in range(17)]
    spaced = [w for pair in zip(moved, [word(" ", 0, 0, kind="spacing")] * 21) for w in pair]
    assert S.collapses(burst) == []
    assert S.collapses(spaced + [word("次", 600.0, 600.2)]) == [(517.0, 557.0)]


def test_clause_evidence_cuts_chinese_at_commas_but_not_a_lone_filler():
    text = "孟老师好，两位老师好，呃，女嘉宾们大家好，我叫卢嘉林，今年27岁，来自哈尔滨。"
    units, t = [], 0.0
    for ch in L.units_of(text):
        units.append({"start": t, "end": t + 0.15, "text": ch, "speaker": "A"})
        t += 0.17
    plain, _ = L.word_lines(units, duration=t + 1, params=PARAMS, language="zh")
    clauses, _ = L.word_lines(units, duration=t + 1, params=PARAMS, language="zh", clauses=L.CLAUSE_BONUS)
    assert len(clauses) > len(plain)
    assert all(line["text"].strip() != "呃，" for line in clauses)
    assert "".join(line["text"] for line in clauses) == "".join(line["text"] for line in plain)
    assert L.clause_bonus("zh-CN") == L.CLAUSE_BONUS and L.clause_bonus("ja") is None and L.clause_bonus(None) is None


# --- glued words --------------------------------------------------------------------------------------------

EXCERPT = json.loads((Path(__file__).parent / "data" / "scribe_glued_excerpt.json").read_text(encoding="utf-8"))


def spelled(text, start, step=0.3, speaker="speaker_0"):
    """One word token of `text`, its characters `step` s apart."""
    chars = [{"text": ch, "start": round(start + step * k, 3), "end": round(start + step * k + step * 0.8, 3)}
             for k, ch in enumerate(text)]
    return word(text, chars[0]["start"], chars[-1]["end"], speaker=speaker, chars=chars)


@pytest.mark.parametrize("text, glued", [
    ("SPRITE" * 31, False),  # a repetition in one token: left as it was
    ("Stratford-on-Avon,", False), ("。Bug。", False), ("！Oh，nice！", False), ("？」", False),
    ("はい。そうです。ありがとう", True),  # two sentence ends between letters, however short
    ("thequickbrownfoxjumpsoverthelazydogagainandagain", True),  # 40+ letters, not one unit said over and over
    ("ははは" * 20, False),
])
def test_a_glued_word_is_told_by_its_shape_not_its_duration(text, glued):
    assert S.is_glued(spelled(text, 0.0, step=0.01)) is glued


def test_the_cached_glued_word_is_found_and_split_and_its_neighbours_left_alone():
    words, found = S.unglued(EXCERPT["words"])
    token = EXCERPT["words"][10]
    assert [w for w in EXCERPT["words"] if S.is_glued(w)] == [token]
    assert found == [S.Glued(996.332, 1979.852, len(token["text"]))]
    assert words[:10] == EXCERPT["words"][:10] and words[-10:] == EXCERPT["words"][-10:]
    units, _ = S.answer_units(words)
    inside = [u for u in units if 996.332 <= u["start"] <= 1979.852]
    texts = [u["text"] for u in inside]
    assert {"CEO", "4.2", "SP"} <= set(texts) and not {"C", "E", "O"} & set(texts)  # Latin and digits stay words
    assert texts[:4] == ["。", "光", "光", "は"]  # 「[聞き取り不能）」 dropped as a note, its 「。」 kept
    assert "笑" not in "".join(texts) and "".join(texts).endswith("戻しておいて。はい、世")  # and 「[笑い]」
    assert {u["speaker"] for u in inside} == {"speaker_3"}
    assert [c for w in words[10:-10] for c in w["characters"]] == token["characters"]  # each in its own time
    lines, _ = L.word_lines(units, duration=2000, params=PARAMS, language="ja")
    assert max(len(line["text"]) for line in lines) < 45 and len(lines) > 8


def test_a_split_shapes_its_words_as_scribe_does():
    token = spelled("Hello world。はい[笑い]", 0.0)
    assert [w["text"] for w in S.split_glued(token)] == ["Hello", "world", "。は", "い", "[笑", "い]"]
    words, notes = S.without_notes(S.split_glued(token))
    assert [w["text"] for w in words] == ["Hello", "world", "。は", "い"] and notes == 1
    untimed = dict(token, characters=[dict(c, start=None) for c in token["characters"]])
    assert S.split_glued(untimed) == [] and S.unglued([untimed])[0] == [untimed]  # kept whole, still reported


# --- a source longer than one call -------------------------------------------------------------------------


def test_a_source_over_the_file_limit_is_cut_at_a_pause_and_its_pieces_stitched(tmp_path, monkeypatch):
    from finesub.speech.joint.mosspro import SpeechSpans

    monkeypatch.setattr(S, "LIMIT", 660.0)  # 15 h in production
    monkeypatch.setattr(S, "LONGEST", 600.0)
    spans, asked = [], []
    monkeypatch.setattr(S, "upload_flac", lambda src, out, span=None: spans.append(span) or Path(out))

    def caller(audio, *, language, seconds, whole):
        lo, hi = spans[-1]
        asked.append((lo, hi, whole))
        # one person throughout, numbered anew in the second piece; a second person only after 900 s
        return {"words": [word(f"w{t}", t - lo, t - lo + 0.5,
                               speaker="speaker_0" if lo == 0 else "speaker_9" if t >= 900 else "speaker_5")
                          for t in range(int(lo) + 1, int(hi) - 1, 2)]}

    source = tmp_path / "long.wav"
    source.write_bytes(b"RIFF")
    heard = SpeechSpans(source, [(0.0, 579.5), (580.5, 1000.0)])  # a pause at 580 s, in the REACH s before 600

    def run(**kw):
        return S.transcribe(source, cache_root=tmp_path / "cloud", language="ja", repair="off", duration=1000.0,
                            speech=heard, **kw)

    t = run(caller=caller)
    assert asked == [(0.0, 580.0, True), (520.0, 1000.0, True)]  # the next starts 60 s before the cut: no pause there
    assert {"pieces": [[0.0, 580.0], [520.0, 1000.0]]} in t.log
    units = t.units()
    texts = [u.text for u in units]
    assert len(texts) == len(set(texts)) and texts == [f"w{x}" for x in range(1, 999, 2)]
    speaker = {u.text: u.speaker for u in units}
    # each piece's speakers kept apart (no linking by the overlap, owner 2026-11-19): only voice merges them
    assert (speaker["w101"], speaker["w701"], speaker["w951"]) == ("w0:speaker_0", "w1:speaker_5", "w1:speaker_9")

    replayed = run(offline=True)
    assert [(u.text, u.speaker) for u in replayed.units()] == [(u.text, u.speaker) for u in units]


@pytest.mark.parametrize("mark, start", [(True, 510.0), (False, 495.0)])
def test_the_next_piece_starts_at_a_pause_the_words_before_break_at(tmp_path, monkeypatch, mark, start):
    from finesub.speech.joint.mosspro import SpeechSpans

    monkeypatch.setattr(S, "LIMIT", 660.0)  # 15 h in production
    monkeypatch.setattr(S, "LONGEST", 600.0)
    spans, asked = [], []
    monkeypatch.setattr(S, "upload_flac", lambda src, out, span=None: spans.append(span) or Path(out))

    def caller(audio, *, language, seconds, whole):
        lo, hi = spans[-1]
        asked.append((lo, hi))
        words = []
        for t in range(int(lo) + 1, int(hi) - 1):
            words.append(word(f"w{t}", t - lo, t - lo + 0.5))
            if mark and t == 509:  # Scribe's mark is a token of its own
                words.append(word("。", t - lo + 0.5, t - lo + 0.5))
        return {"words": words}

    source = tmp_path / "long.wav"
    source.write_bytes(b"RIFF")
    # pauses at 580 s (the end), 495 s (1 s) and 510 s (0.4 s), both 60-90 s before that end
    heard = SpeechSpans(source, [(0.0, 494.5), (495.5, 509.8), (510.2, 579.0), (581.0, 1000.0)])
    S.transcribe(source, cache_root=tmp_path / "cloud", language="ja", repair="off", duration=1000.0, speech=heard,
                 caller=caller)
    assert asked == [(0.0, 580.0), (start, 1000.0)]


def test_a_source_up_to_the_file_limit_is_sent_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "upload_flac", lambda src, out, span=None: Path(out))
    asked = []

    def caller(audio, *, language, seconds, whole):
        asked.append(seconds)
        return {"words": [word("a", 1.0, 1.5)]}

    source = tmp_path / "long.wav"
    source.write_bytes(b"RIFF")
    total = S.LIMIT - 30.0  # longer than one piece, not than a file
    S.transcribe(source, cache_root=tmp_path / "cloud", language="ja", repair="off", duration=total, caller=caller)
    assert asked == [total] and (tmp_path / "cloud" / S.SCRIBE / f"{S.CALL}.raw.json").exists()


def test_shifted_moves_words_and_their_characters():
    got = S.shifted([word("ab", 1.0, 2.0, chars=[{"text": "a", "start": 1.0, "end": 1.5}]), word("c", 3.0, 3.1)], 10.0)
    assert [(w["start"], w["end"]) for w in got] == [(11.0, 12.0), (13.0, 13.1)]
    assert got[0]["characters"][0]["start"] == 11.0
