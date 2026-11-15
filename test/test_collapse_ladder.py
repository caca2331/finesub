"""The collapse ladder's rules (`speech/postprocessing/collapse_ladder`): its Whisper windows, the presence check's
windows, the pin and the pick. Its decisions on whole episodes were checked against the accepted evaluation arm by
replay; how its inputs are asked for is `test_recall_decode.py`'s."""

from __future__ import annotations

from finesub.speech.postprocessing import collapse_ladder as cl
from .test_recall_repair import _payload, _row, needs_readings

SQUEEZED = "させていただきましょうそれでは"


def test_a_window_takes_whole_vad_intervals_grown_over_aligned_segments_and_the_stage_s_tail() -> None:
    rows = [_row("前です", 3.0, 5.0), _row(SQUEEZED, 10.0, 10.24), _row("次です", 16.0, 17.0)]
    payload = _payload(rows, [(3.0, 5.0), (6.0, 9.5), (10.0, 12.0), (12.3, 15.0), (16.0, 17.0)])
    # an aligned segment across 11.5-12.6 joins the interval after the row's own
    payload["segments"] = [{"start": 3.0, "end": 5.0}, {"start": 10.0, "end": 11.5}, {"start": 11.5, "end": 12.6},
                           {"start": 16.0, "end": 17.0}]
    wins = {w["kind"]: w for w in cl.windows(payload, rows)}
    assert [(v["start"], v["end"]) for v in wins["iso"]["intervals"]] == [(10.0, 12.0), (12.3, 15.0)]
    assert wins["iso"]["tail"] == 0.7  # the gap to the next interval (1.0 s), at most the stage's 0.7
    assert [(v["start"], v["end"]) for v in wins["pad"]["intervals"]] == [(6.0, 9.5), (10.0, 12.0), (12.3, 15.0)]
    assert wins["ctx"]["start"] == 3.0 and wins["ctx"]["end"] == 17.0 and wins["ctx"]["tail"] == 0.7


@needs_readings
def test_the_presence_check_needs_the_row_s_text_and_windows_long_enough_to_hear() -> None:
    row = _row(SQUEEZED, 10.0, 10.24)
    assert cl.present(row, (10.0, 12.0), SQUEEZED, "ja")
    assert not cl.present(row, (10.0, 12.0), "させて", "ja")
    # 0.13 s less a quarter at either end is under MIN_CLIP: not heard, whatever the referee said
    assert not cl.present(row, (10.0, 10.13), SQUEEZED, "ja")


@needs_readings
def test_a_drawn_out_first_or_last_word_is_pinned_to_a_typical_length() -> None:
    words = [{"word": "させて", "start": 10.0, "end": 13.0}, {"word": "いただき", "start": 13.0, "end": 13.5}]
    # 3 s over 3 sounds is past LONG per sound (PIN): the head moves to 3 x TYPICAL before the word's end
    assert cl.pinned((10.0, 13.5), words, "ja", "head") == (13.0 - 3 * cl.PIN["ja"][1], 13.5)
    assert cl.pinned((10.0, 13.5), words, "ja", "tail") == (10.0, 13.5)


def _rec(**spans):
    rec = {"index": 0, "echo": False, "late": False}
    for p in cl.STEPS:
        rec[p], rec[f"{p}_guard"] = spans.get(p), None
    for k in cl.KINDS:
        rec[f"{k}_branch"] = spans.get(f"{k}_branch", "whisper" if spans.get(k) else "whisper_miss")
    return rec


def test_the_pick_follows_the_steps_and_takes_a_crossed_span_last() -> None:
    assert cl.pick(_rec(pad=(1, 2), S=(1, 3))) == "pad"
    assert cl.pick(_rec(S=(1, 3), iso=(1, 2))) == "S"
    # iso and ctx are read only when pad gave nothing: here pad's decode moved the row but a guard stopped it
    assert cl.pick(_rec(pad_branch="whisper", iso=(1, 2))) is None
    assert cl.pick(_rec(pad_branch="same", iso=(1, 2))) == "iso"
    # a span past the next row's start is taken only when no proposer stays before it
    assert cl.pick(_rec(pad=(1, 9), pad_branch="crossed", S=(1, 2))) == "S"
    assert cl.pick(_rec(pad=(1, 9), pad_branch="crossed")) == "pad"


@needs_readings
def test_iso_and_ctx_are_decoded_only_for_rows_pad_gave_nothing() -> None:
    rows = [_row(SQUEEZED, 10.0, 10.24), _row("次です", 20.0, 21.0), _row(SQUEEZED, 30.0, 30.24)]
    payload = _payload(rows, [(10.0, 15.0), (20.0, 21.0), (30.0, 35.0)])
    words = [("させて", 10.0, 10.6), ("いただき", 10.6, 11.2), ("ましょう", 11.2, 11.7), ("それでは", 11.7, 12.2)]
    pad = {w["index"]: w for w in cl.windows(payload, rows, ("pad",))}
    assert set(pad) == {0, 2}
    # row 0's pad decode moves it; row 2's heard nothing
    decoded = {("pad", 0): {**pad[0], "segments": [{"words": [{"word": a, "start": s, "end": e} for a, s, e in words]}]},
               ("pad", 2): {**pad[2], "segments": []}}
    assert cl.retried(rows, decoded, "ja") == {2}
    assert {(w["kind"], w["index"]) for w in cl.windows(payload, rows, cl.KINDS[1:], only={2})} == {("iso", 2), ("ctx", 2)}


def test_in_english_whisper_s_words_go_to_a_row_by_words_not_by_letters() -> None:
    rows = [_row("hello there", 8.0, 9.0), _row("we should go now", 10.0, 10.2)]
    said = ["hello", "there", "we", "should", "go", "now"]  # the stage's words carry no spaces

    def window(texts):
        words = [{"word": w, "start": 8.0 + k * 0.5, "end": 8.5 + k * 0.5} for k, w in enumerate(texts)]
        return {"start": 8.0, "end": 12.0, "segments": [{"words": words}]}

    got = cl.assigned_words(rows, 1, window(said), "en")
    assert [w["word"] for w in got] == said[2:]
    # the same letters in other words: a letter pair is shared almost anywhere, a word pair is not
    assert cl.assigned_words(rows, 1, window(["hello", "there", "shoe", "wool", "dog", "snow"]), "en") is None


def test_a_language_without_its_own_pin_takes_its_kind_of_sound_s_and_english_counts_letters() -> None:
    assert cl.pin_of("yue") == cl.pin_of("ko") == cl.PIN["zh"] and cl.pin_of("th") == cl.PIN["en"]
    assert cl.pin_of("ja") == cl.PIN["ja"]
    words = [{"word": "hello", "start": 10.0, "end": 13.5}]
    # 3.5 s over 5 letters is past English's LONG: the head moves to 5 x its TYPICAL before the word's end
    assert cl.pinned((10.0, 14.0), words, "en", "head") == (13.5 - 5 * cl.PIN["en"][1], 14.0)
