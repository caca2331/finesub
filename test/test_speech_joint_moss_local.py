"""Local MOSS (`--asr moss-local`) without a GPU or the model: the rescue's pure rules, the adapter's bookkeeping and
the process state it has to give back, the device criterion, and its wiring into the stage and the batch."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from finesub.speech import joint
from finesub.speech.joint import moss_local as ML
from finesub.speech.joint import relink as RL
from finesub.speech.joint import stage as joint_stage
from finesub.speech.joint.model import Transcript, Unit, segment_turns
from finesub.speech.joint.moss_local import decoder as D
from finesub.speech.joint.moss_local import rescue
from finesub.speech.joint.moss_local import rules as R
from finesub.speech.runtime import device as device_module
from finesub.speech.runtime import hf_weights
from finesub.subtitles.metrics import weighted_char_count


def seg(start, end, text, speaker="S01"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# --- rules ---------------------------------------------------------------------------------------------------

ENGLISH = "so we went out to the park and it was just like the best day of the year for all of us " * 2
GERMAN = ("wir sind heute in den Park gegangen und es war wirklich der schönste Tag des Jahres für uns alle "
          "zusammen mit Freunden und Familie draußen in der Sonne bei gutem Wetter")


def test_english_is_told_from_another_latin_language_by_its_common_words():
    assert R.wrong_language(ENGLISH, "en") is False
    assert R.wrong_language(GERMAN, "en") is True
    assert R.wrong_language("too short to tell", "en") is None


def test_japanese_is_told_from_chinese_by_kana_and_chinese_from_japanese():
    japanese = "今日はみんなで公園に行きました。とても楽しかったですね、また行きたいと思います。" * 2
    chinese = "我们今天一起去了公园，天气非常好，大家都玩得很开心，希望下次还能再一起去那里玩。" * 2
    assert R.wrong_language(japanese, "ja") is False
    assert R.wrong_language(chinese, "ja") is True
    assert R.wrong_language(chinese, "zh") is False
    assert R.wrong_language(japanese, "zh") is True


def test_a_language_it_cannot_test_or_none_passes():
    for language in (None, "", "auto", "sw"):
        assert R.wrong_language(GERMAN, language) is False


def test_a_loop_is_a_short_period_over_the_last_span():
    assert R.periodic([1, 2, 3] * 16)
    assert not R.periodic(list(range(R.LOOP_SPAN)))
    assert not R.periodic([7] * (R.LOOP_SPAN - 1))


def test_gaps_are_the_uncovered_stretches_up_to_the_end():
    segs = [seg(0, 5, "a"), seg(40, 50, "b"), seg(45, 52, "c")]
    assert R.gaps(segs, 100.0) == [(5, 40), (52, 100.0)]
    assert R.gaps(segs, 60.0, least=3.0) == [(5, 40), (52, 60.0)]


AROUND = [seg(20.0, 26.0, "a"), seg(27.0, 33.0, "b"), seg(34.0, 38.0, "c"),  # gap 38-70
          seg(70.0, 75.0, "d"), seg(76.0, 79.0, "e"), seg(79.5, 84.0, "f"), seg(85.0, 90.0, "g")]


def test_each_side_of_a_window_takes_the_lead_and_goes_on_to_the_next_segment_boundary():
    # before: the last segment starting LEAD s or more before the gap (27 <= 28); after: the first ending LEAD s
    # or more after it (84 >= 80) -- whatever the gap's length
    assert R.window(AROUND, 38.0, 70.0, 100.0) == (27.0, 84.0)
    assert R.window([*AROUND[:3], *(dict(s, start=s["start"] - 25, end=s["end"] - 25) for s in AROUND[3:])],
                    38.0, 45.0, 100.0) == (27.0, 59.0)


def test_a_window_side_is_pushed_out_past_a_segment_straddling_it():
    over = seg(25.0, 30.0, "someone talking over b", "S02")  # straddles 27
    late = seg(82.0, 87.0, "someone talking over f", "S02")  # straddles 84
    assert R.window([*AROUND, over, late], 38.0, 70.0, 100.0) == (20.0, 90.0)
    touching = seg(23.0, 27.0, "one stretch of talk cut in two")  # ends where b starts
    assert R.window([*AROUND, touching], 38.0, 70.0, 100.0)[0] == 20.0


def test_a_window_side_with_little_transcript_takes_what_there_is_up_to_the_clip_edge():
    assert R.window(AROUND[1:4], 38.0, 70.0, 100.0) == (27.0, 75.0)
    assert R.window(AROUND[:3], 38.0, 80.0, 80.0) == (27.0, 80.0)  # the gap at the clip's end


def test_a_window_with_nothing_before_it_starts_just_before_the_first_speech():
    assert R.window(AROUND, 0.0, 20.0, 100.0, onset=14.5) == (14.5 - R.ONSET_LEAD, 33.0)
    assert R.window(AROUND, 0.0, 20.0, 100.0) == (0.0, 33.0)
    assert R.window(AROUND, 38.0, 70.0, 100.0, onset=50.0) == (27.0, 84.0)  # transcript before it: unchanged
    assert R.window(AROUND, 0.0, 20.0, 100.0, onset=0.2) == (0.0, 33.0)  # not before the clip
    assert R.window(AROUND, 0.0, 20.0, 100.0, onset=25.0)[0] == 20.0  # not past the gap


def test_a_continuation_starts_where_the_window_it_continues_stopped():
    assert R.window(AROUND, 52.0, 70.0, 100.0, cont=True, onset=53.0) == (52.0, 84.0)
    assert R.window(AROUND, 5.0, 20.0, 100.0, cont=True, onset=9.0)[0] == 5.0  # at the clip head too


def test_a_window_replaces_the_context_it_decoded_again_even_where_an_earlier_window_crosses_its_edge():
    earlier = seg(26.2, 28.0, "an earlier window's line", "c1:S01")  # straddles 27, where the window would start
    segs = [*AROUND, earlier]
    lo, hi = R.window(segs, 38.0, 70.0, 100.0)
    assert (lo, hi) == (26.2, 84.0)
    decoded = [seg(26.3, 33.0, "the earlier line and b"), seg(34.0, 41.0, "c and the gap's first words"),
               seg(60.0, 75.0, "the gap's last words and d"), seg(76.0, 84.0, "e and f")]
    out, gone = R.replace(segs, decoded, lo, hi)
    # the line across the gap's start, centred before it, is kept, and c with it is not there twice
    assert [s["text"] for s in out] == ["a", *(s["text"] for s in decoded), "g"]
    assert sorted(s["text"] for s in gone) == sorted(["an earlier window's line", "b", "c", "d", "e", "f"])


def test_a_window_line_says_again_the_replaced_segment_it_lies_in_with_the_same_words():
    old = [seg(10.0, 13.0, "我觉得小胡这一把非常奇怪", "S11"), seg(13.0, 15.0, "好 你不是的话", "S09")]
    assert R.twin(seg(10.1, 11.4, "我觉得小胡这一把", "c1:S02"), old) == "S11"  # one segment cut in two
    assert R.twin(seg(10.1, 11.4, "我觉得小狐这一把", "c1:S02"), old) == "S11"  # a character heard otherwise
    assert R.twin(seg(10.1, 11.4, "完全不一样的话", "c1:S02"), old) is None
    assert R.twin(seg(12.0, 14.5, "非常奇怪好", "c1:S02"), old) is None  # across two: neither holds 80% of it
    said = [seg(20.0, 23.0, " Oh Jude, you're screwed.", "S01")]
    assert R.twin(seg(20.1, 22.9, " Oh dude, you're screwed.", "c1:S03"), said) == "S01"
    both = [seg(30.0, 31.0, "对", "S01"), seg(30.0, 31.0, "对", "S02")]  # two people, the same word
    assert R.twin(seg(30.0, 31.0, "对", "c1:S01"), both) is None
    earlier = [{**seg(40.0, 42.0, "安宇出局", "c3:S04"), RL.TWIN: "S13"}]  # an earlier window's line said again
    assert R.twin(seg(40.1, 41.9, "安宇出局", "c4:S01"), earlier) == "S13"


def test_a_window_that_stopped_early_keeps_the_context_it_did_not_reach():
    decoded = [seg(27.0, 33.0, "b again"), seg(34.0, 45.0, "c and the gap's first words")]
    out, gone = R.replace(AROUND, decoded, 27.0, 84.0)
    assert [s["text"] for s in gone] == ["b", "c"]
    assert [s["text"] for s in out] == ["a", "b again", "c and the gap's first words", "d", "e", "f", "g"]
    assert R.replace(AROUND, [], 27.0, 84.0) == (AROUND, [])


class _ScriptedPass:
    """`rescue.Pass` without the model: the main pass hands back `main`, each window the next of `windows`
    (segments in the source's time, whether it hit its token budget). The audio is the sample index, so a window
    knows where it starts."""

    main: list = []
    windows: list = []
    seen: list = []
    voiced = rescue.Pass.voiced
    onset = rescue.Pass.onset

    def __init__(self, model, processor, audio, spans, prompt, language, budget_gib, window=False, hole=None):
        self.lo, self.hi = audio[0] / D.SR, (audio[-1] + 1) / D.SR
        self.spans, self.window, self.events, self.acoustic, self.capped, self.dec = spans, window, [], [], False, None

    def run(self):
        if not self.window:
            return [dict(s) for s in self.main]
        type(self).seen.append((round(self.lo, 2), round(self.hi, 2)))
        segs, self.capped = self.windows.pop(0)
        return [{**s, "start": s["start"] - self.lo, "end": s["end"] - self.lo} for s in segs]


def test_a_window_that_hit_its_budget_keeps_the_context_it_did_not_reach_and_its_continuation_ends_as_it_did(
        monkeypatch):
    first = [seg(27.0, 33.0, "b"), seg(34.0, 38.5, "c and the gap's first words"), seg(39.0, 50.0, "x")]
    rest = [seg(51.0, 69.0, "y"), seg(70.0, 75.2, "d again"), seg(76.0, 79.0, "e again"), seg(79.5, 84.0, "f again")]
    monkeypatch.setattr(_ScriptedPass, "main", AROUND)
    monkeypatch.setattr(_ScriptedPass, "windows", [(first, True), (rest, False)])
    monkeypatch.setattr(_ScriptedPass, "seen", [])
    monkeypatch.setattr(rescue, "Pass", _ScriptedPass)
    segs, events = rescue._part(None, None, np.arange(100 * D.SR, dtype=np.float64), np.zeros(0, np.int16),
                                [(20.0, 90.0)], "en", "", 6.5, "")
    # the continuation starts where the first window stopped and ends where it ended: d, e, f were still there
    assert _ScriptedPass.seen == [(27.0, 84.0), (50.0, 84.0)]
    assert [s["text"] for s in segs] == ["a", *(s["text"] for s in first + rest), "g"]
    fills = [e for e in events if e["kind"] == "fill"]
    assert [(e["window"], e["segments"], e["replaced"]) for e in fills] == [([27.0, 84.0], 3, 2),
                                                                           ([50.0, 84.0], 4, 3)]
    assert {s["speaker"] for s in segs if s["text"] == "x"} == {"c1:S01"}
    # a context line said again remembers whose it was, for when its label has too little voice to tell
    assert [(s["text"], s.get(RL.TWIN)) for s in segs if s["speaker"].startswith("c")][:2] == [("b", "S01"),
                                                                                                ("c and the gap's first words", None)]


def test_a_line_the_window_hears_only_the_end_of_is_matched_by_its_end():
    # hina-44's window: the pass's line started before the window, which wrote its tail word three times
    have = [seg(8.0, 10.5, "あ、ハリケーン。")]
    tail, again, new = seg(9.0, 10.4, "ハリケーン"), seg(12.0, 13.0, "ハリケーン"), seg(20.0, 24.0, "新しい話です")
    assert R.clock_off([tail, again, new], have, 9.0, 61.0) == (1, pytest.approx(0.1))


def test_a_context_line_the_window_split_differently_is_matched_by_the_edge_it_kept():
    # the branch's windows (hina-70, hina-72 and two more): the window wrote the line's second half on its own
    have = [seg(789.64, 803.64, "hello there my friend how are you")]
    half = seg(794.79, 803.55, "there my friend how are you")
    assert R.clock_off([half], have, 649.1, 803.64) == (1, pytest.approx(0.09))


def test_a_window_minutes_late_is_caught_however_far_its_anchors_are():
    # hina-14's end: a window whose clock ran 213 s off
    have = [seg(10.0, 12.0, "hello there my friend")]
    late, new = seg(223.0, 225.0, "hello there my friend"), seg(230.0, 234.0, "something nobody said before")
    assert R.clock_off([late, new], have, 9.0, 300.0) == (1, pytest.approx(213.0))


def test_one_anchor_matching_the_same_words_said_again_far_off_does_not_mark_the_clock_off():
    # hina-44's window: `そう。` matched minutes later, the tail of the cut-off line on time
    have = [seg(8.0, 10.5, "あ、ハリケーン。"), seg(10.6, 11.0, "そう。")]
    tail, again, new = seg(9.0, 10.4, "ハリケーン"), seg(200.0, 200.4, "そう"), seg(20.0, 24.0, "新しい話です")
    assert R.clock_off([tail, again, new], have, 9.0, 300.0) == (2, pytest.approx(0.1))


def test_a_window_with_its_clock_off_is_dropped_only_with_the_check_on(monkeypatch):
    main = [seg(0.0, 9.0, "the first line"), seg(10.0, 20.0, "the second line"), seg(60.0, 70.0, "the third line")]
    late = [seg(16.0, 26.0, "the second line"), seg(30.0, 40.0, "something new")]  # the window's clock 6 s late
    got = {}
    for on in (False, True):
        monkeypatch.setattr(R, "CLOCK_CHECK", on)
        monkeypatch.setattr(_ScriptedPass, "main", main)
        monkeypatch.setattr(_ScriptedPass, "windows", [(late, False), ([], False)])  # off: 40-60 s goes on
        monkeypatch.setattr(_ScriptedPass, "seen", [])
        monkeypatch.setattr(rescue, "Pass", _ScriptedPass)
        segs, events = rescue._part(None, None, np.arange(70 * D.SR, dtype=np.float64), np.zeros(0, np.int16),
                                    [(0.0, 70.0)], "en", "", 6.5, "")
        fill = next(e for e in events if e["kind"] == "fill")
        got[on] = ([s["text"] for s in segs], fill["anchors"], fill["anchor_off"])
    assert got[True] == ([s["text"] for s in main], 1, 6.0)  # dropped whole
    assert got[False] == (["the first line", "the second line", "something new", "the third line"], 1, 6.0)


def test_a_late_end_is_pulled_in_but_not_shorter_than_its_text_takes():
    text = "あいうえおかきくけこ"
    w = weighted_char_count(text)
    paced = [seg(float(k), k + 2.0, text) for k in (0, 3, 6)]  # ending on their speech: the source's pace
    spans = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0), (10.0, 11.0)]
    long = seg(10.0, 20.0, text)
    out, trimmed = R.trim_tails([*paced, long], spans)
    floor = 10.0 + R.TAIL_SLACK * w / (w / 2.0)
    assert out[-1]["end"] == pytest.approx(min(20.0, floor))
    assert trimmed == [[10.0, 20.0, round(min(20.0, floor), 2)]]
    assert out[:3] == paced


def test_a_segment_without_speech_keeps_its_end():
    segs = [seg(0.0, 2.0, "abc"), seg(5.0, 9.0, "def")]
    out, trimmed = R.trim_tails(segs, [(0.0, 2.0)])
    assert out == segs and trimmed == []


def _centroids(vectors):
    def centroids(segs):
        return {s["speaker"]: np.asarray(vectors[s["speaker"]], float) for s in segs}

    return centroids


def test_a_long_run_without_a_mark_is_unmarked_and_a_short_one_or_one_with_commas_is_not():
    lost = "キャスターは死の権能を受け継いで生と死を再びる転させるため冥界に留まることを決めました。"
    assert R.unmarked(lost)
    assert not R.unmarked("キャスターは死の権能を受け継いで、生と死を再び流転させるため、冥界に留まることを決めました。")
    assert not R.unmarked("冥界に留まることを決めました")


def test_the_graft_adds_the_re_decodes_marks_without_changing_a_character():
    text, added, _ = R.graft("生と死を再びる転させるため冥界に留まることを決めました。",
                             "生と死を再び流転させるため、冥界に留まることを決めました。")
    assert (text, added) == ("生と死を再びる転させるため、冥界に留まることを決めました。", 1)
    # none after the last character, none where the segment already has one, no sentence end
    assert R.graft("あいう。えお", "あいう、。えお、")[:2] == ("あいう。えお", 0)
    assert R.graft("まあそれかアルレッキーノが用意してる", "まあそれか。アルレッキーノが用意してる")[:2] == (
        "まあそれかアルレッキーノが用意してる", 0)
    # a window written under the Chinese default prompt lends its ， as 、
    assert R.graft("なんか青い光とかが出てギミックが作用する", "なんか青い光とかが出て，ギミックが作用する")[:2] == (
        "なんか青い光とかが出て、ギミックが作用する", 1)
    # none inside a number or Latin text
    assert R.graft("天井まで1000連分の石が必要でそれから", "天井まで1,000連分の石が必要で、それから")[0] == (
        "天井まで1000連分の石が必要で、それから")
    assert R.graft("それでOK so let's goって", "それでOK, so let's goって")[:2] == ("それでOK so let's goって", 0)


def test_a_re_decode_that_heard_something_else_lends_no_marks():
    text, added, similar = R.graft("今回はナドクライの魔人任務について話します", "全然違う話を、しているようです")
    assert (text, added) == ("今回はナドクライの魔人任務について話します", 0) and similar < R.GRAFT_SIMILAR


def test_a_source_longer_than_a_pass_is_cut_into_parts_at_silero_pauses(monkeypatch, tmp_path):
    from finesub.speech.joint.mosspro import SpeechSpans

    parts = []

    def part(model, processor, audio, pcm, spans, language, prompt, budget, main_prompt):
        parts.append(round(len(audio) / D.SR, 1))
        return [], [{"kind": "kv", "format": "fp16", "positions": 0}]

    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: 1000.0)
    monkeypatch.setattr(rescue, "PART_SECONDS", 1000.0)
    monkeypatch.setattr(rescue, "_part", part)
    monkeypatch.setattr(RL, "merge_by_voice", lambda *a, **k: ([], []))
    dur = 2500
    # pauses at 980 s (in the REACH s before 1000), 900 s (60-90 s before that) and 1880 s; none 60-90 s before it
    heard = [(0.0, 899.5), (900.5, 979.0), (981.0, 1879.0), (1881.0, 2500.0)]
    speech = SpeechSpans(tmp_path / "a.wav", heard, np.zeros(dur * D.SR, np.int16))
    result = rescue.transcribe(None, None, np.zeros(dur * D.SR, np.float32), "ja", 1.0, speech)
    assert [p["start"] for p in result.passes] == [0.0, 900.0, 1820.0]
    assert parts == [980.0, 980.0, 680.0]


@pytest.mark.parametrize("holds, part", [(1000.0, 600.0), (400.0, 400.0)])
def test_a_part_is_at_most_ten_minutes_or_what_the_card_holds(monkeypatch, tmp_path, holds, part):
    from finesub.speech.joint.mosspro import SpeechSpans

    parts = []

    def fake_part(model, processor, audio, pcm, spans, language, prompt, budget, main_prompt):
        parts.append(len(audio) / D.SR)
        return [], [{"kind": "kv", "format": "fp16", "positions": 0}]

    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: holds)
    monkeypatch.setattr(rescue, "_part", fake_part)
    monkeypatch.setattr(RL, "merge_by_voice", lambda *a, **k: ([], []))
    dur = 2500
    speech = SpeechSpans(tmp_path / "a.wav", [(0.0, float(dur))], np.zeros(dur * D.SR, np.int16))
    rescue.transcribe(None, None, np.zeros(dur * D.SR, np.float32), "ja", 1.0, speech)
    assert len(parts) > 1 and max(parts) == pytest.approx(part)


@pytest.mark.parametrize("broken, start", [(True, 915.0), (False, 900.0)])
def test_the_next_part_starts_at_a_pause_the_previous_parts_segments_break_at(monkeypatch, tmp_path, broken, start):
    from finesub.speech.joint.mosspro import SpeechSpans

    starts = []

    def part(model, processor, audio, pcm, spans, language, prompt, budget, main_prompt):
        starts.append(len(starts))
        if len(starts) == 1 and broken:
            segs = [{"start": 0.0, "end": 914.7, "text": "あ", "speaker": "S01"},
                    {"start": 915.3, "end": 979.0, "text": "い", "speaker": "S01"}]
        else:
            segs = [{"start": 0.0, "end": len(audio) / D.SR - 1.0, "text": "う", "speaker": "S01"}]
        return segs, [{"kind": "kv", "format": "fp16", "positions": 0}]

    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: 1000.0)
    monkeypatch.setattr(rescue, "PART_SECONDS", 1000.0)
    monkeypatch.setattr(rescue, "_part", part)
    monkeypatch.setattr(RL, "merge_by_voice", lambda *a, **k: ([], []))
    dur = 2500
    # pauses at 980 s (the end), 900 s (1 s) and 915 s (0.4 s), both 60-90 s before that end
    heard = [(0.0, 899.5), (900.5, 914.8), (915.2, 979.0), (981.0, 2500.0)]
    speech = SpeechSpans(tmp_path / "a.wav", heard, np.zeros(dur * D.SR, np.int16))
    result = rescue.transcribe(None, None, np.zeros(dur * D.SR, np.float32), "ja", 1.0, speech)
    assert [p["start"] for p in result.passes][:2] == [0.0, start]


def test_punctuate_re_decodes_only_japanese_segments_that_lost_their_commas(monkeypatch):
    lost = "キャスターは死の権能を受け継いで生と死を再びる転させるため冥界に留まることを決めました。"
    windows = []

    class FakePass:
        def __init__(self, model, processor, audio, spans, prompt, language, budget, window=False):
            assert window
            windows.append(len(audio) / D.SR)
            self.events = []

        def run(self):
            return [seg(10.0, 18.0, lost.replace("継いで", "継いで、"))]

    monkeypatch.setattr(rescue, "Pass", FakePass)
    monkeypatch.setattr(rescue.torch.cuda, "empty_cache", lambda: None)
    segs = [seg(10.0, 18.0, lost), seg(20.0, 22.0, "短いです。"), seg(30.0, 40.0, "这" * 50)]
    out, events = rescue.punctuate(None, None, np.zeros(60 * D.SR, np.float32), [], segs, "ja", "", 1.0)
    assert out[0]["text"] == lost.replace("継いで", "継いで、") and out[1:] == segs[1:]
    assert windows == [28.0]  # LEAD s either side
    rescue.punctuate(None, None, np.zeros(60 * D.SR, np.float32), [(0.0, 30.0), (31.0, 60.0)], segs[:1], "ja", "", 1.0)
    assert windows[1] == 30.5  # out to the pause after it
    assert [(e["kind"], e["added"]) for e in events] == [("punct", 1)]
    assert rescue.punctuate(None, None, np.zeros(60 * D.SR, np.float32), [], segs, "en", "", 1.0) == (segs, [])


# --- one pass, driven by a scripted model --------------------------------------------------------------------


class _Tok:
    """Toy tokenizer: the model's multi-character structural tokens, every other character its own token."""

    MULTI = ("][", " [", "]\n")
    EOS, JUNK = 0, 1  # what a model with nothing to continue writes once EOS is masked

    def __init__(self):
        self.vocab = ["<eos>", "～"]

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab.append(piece)
        return self.vocab.index(piece)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out, i = [], 0
        while i < len(text):
            piece = next((m for m in self.MULTI if text.startswith(m, i)), text[i])
            out.append(self._id(piece))
            i += len(piece)
        return out

    def decode(self, ids) -> str:
        return "".join(self.vocab[i] for i in ids if i != self.EOS)


class _Scripted:
    """Writes the first of `scripts` that what is written so far begins, then EOS; prefers EOS where `ends` (texts
    written so far) say, the script's next token coming second. With no script to continue, EOS (then junk)."""

    VOCAB = 512

    def __init__(self, tok: _Tok, scripts: list[str], ends: tuple[str, ...] = ()):
        self.tok, self.scripts, self.ends = tok, scripts, set(ends)
        self.generation_config = type("G", (), {"eos_token_id": _Tok.EOS})()

    def logits(self, seq: list[int]):
        import torch

        text = self.tok.decode(seq)
        out = torch.full((self.VOCAB,), -1.0)
        out[_Tok.JUNK] = 0.0
        script = next((s for s in self.scripts if s.startswith(text) and len(s) > len(text)), None)
        if script is None:
            out[_Tok.EOS] = 2.0
            return out
        out[self.tok.encode(script[len(text):])[0]] = 1.0
        if text in self.ends:
            out[_Tok.EOS] = 2.0
        return out


class _Decoder:
    """`decoder.Decoder`'s face: the token sequence is rebuilt from the positions `Pass._feed` sets."""

    def __init__(self, model, inputs, budget, budget_gib):
        self.model, self.n, self.seq = model, inputs["input_ids"].shape[1], []
        self.pos, self.max, self.prompt = 0, 1 << 20, [_Tok.EOS]
        self.cache = type("C", (), {"kv": "bf16"})()

    def logits(self, token: int):
        written = self.pos - (self.n - 1)
        self.seq = self.seq[:written - 1] + [token] if written > 0 else []
        return self.model.logits(self.seq)


def run_pass(monkeypatch, scripts, *, seconds, spans, ends=(), window=False, hole=None):
    import torch

    monkeypatch.setattr(D, "encode_inputs", lambda processor, audio, prompt: {"input_ids": torch.zeros((1, 4))})
    monkeypatch.setattr(D, "Decoder", _Decoder)
    tok = _Tok()
    processor = type("P", (), {"tokenizer": tok})()
    p = rescue.Pass(_Scripted(tok, scripts, ends), processor, np.zeros(int(seconds * D.SR), np.float32), spans, "",
                    None, 1.0, window=window, hole=hole)
    segs = p.run()
    return [(s["start"], s["end"], s["speaker"], s["text"]) for s in segs], [e for e in p.events if e["kind"] != "kv"], p


# Times off the whole second: two whole-second timestamps in a row bar a third (`INT_STREAK`).
WELL_FORMED = "[0.12][S01]こんにちは[1.53][1.64][S02]はい[2.05][2.05][音楽][5.07][5.21][S01]じゃあね[6.08]"
PARSED = [(0.12, 1.53, "S01", "こんにちは"), (1.64, 2.05, "S02", "はい"), (5.21, 6.08, "S01", "じゃあね")]


def test_a_well_formed_stream_parses_into_its_segments_and_events(monkeypatch):
    segs, events, p = run_pass(monkeypatch, [WELL_FORMED], seconds=8.0, spans=[(0.1, 6.1)])
    assert segs == PARSED and events == []
    assert p.acoustic == [{"start": 2.05, "end": 5.07, "tag": "音楽"}]


@pytest.mark.parametrize("join", ["] [", "]\n[", "]  ["])
def test_whitespace_between_segments_parses_the_same_with_nothing_decoded_again(monkeypatch, join):
    segs, events, _ = run_pass(monkeypatch, [WELL_FORMED.replace("][", join)], seconds=8.0, spans=[(0.1, 6.1)])
    assert segs == PARSED and events == []


@pytest.mark.parametrize("join", ["][", "] ["])
def test_a_forced_start_opens_its_segment_the_way_the_model_writes_them(monkeypatch, join):
    skipped = f"[0.12][S01]あ[1.13{join}40.14][S01]い[41.15]"
    forced = f"[0.12][S01]あ[1.13{join}5.17][S01]う[39.16{join}40.14][S01]い[41.15]"
    segs, events, p = run_pass(monkeypatch, [skipped, forced], seconds=45.0, spans=[(0.1, 1.13), (5.17, 41.2)])
    assert [e["kind"] for e in events] == ["skip"] and events[0]["forced"] == 5.17
    assert segs == [(0.12, 1.13, "S01", "あ"), (5.17, 39.16, "S01", "う"), (40.14, 41.15, "S01", "い")]
    assert p.tok.decode(p.ids) == forced


def test_a_long_holes_window_that_skips_talk_at_the_holes_start_is_forced_back_to_it(monkeypatch):
    # hina-73's opening: the window wrote only a line after the talk in its hole
    skipped = "[40.14][S01]い[41.15]"
    forced = "[5.17][S01]う[39.16][40.14][S01]い[41.15]"
    segs, events, _ = run_pass(monkeypatch, [skipped, forced], seconds=45.0, spans=[(5.17, 41.2)], window=True,
                               hole=(0.0, 39.5))
    assert [(e["kind"], e["forced"]) for e in events] == [("hole-skip", 5.17)]
    assert segs == [(5.17, 39.16, "S01", "う"), (40.14, 41.15, "S01", "い")]


def test_a_long_holes_window_is_checked_at_its_first_lines_head_before_the_line_can_fail(monkeypatch):
    # the late line never closes (the model wants to end inside it): checked at its head, the talk is not lost
    skipped = "[40.14][S01]い"
    forced = "[5.17][S01]う[39.16][40.14][S01]い[41.15]"
    segs, events, _ = run_pass(monkeypatch, [skipped, forced], seconds=45.0, spans=[(5.17, 41.2)], window=True,
                               hole=(0.0, 39.5))
    assert [(e["kind"], e["forced"]) for e in events] == [("hole-skip", 5.17)]
    assert segs == [(5.17, 39.16, "S01", "う"), (40.14, 41.15, "S01", "い")]


def test_a_windows_lines_in_its_lead_in_or_on_time_force_nothing(monkeypatch):
    lead_in = "[0.50][S01]あ[8.90][12.10][S01]い[44.00]"  # the lead-in's line, then the hole's talk on time
    segs, events, _ = run_pass(monkeypatch, [lead_in], seconds=45.0, spans=[(0.1, 9.0), (12.0, 45.0)], window=True,
                               hole=(10.0, 45.0))
    assert events == [] and len(segs) == 2


def test_an_acoustic_event_in_the_hole_is_not_the_line_checked(monkeypatch):
    skipped = "[0.10][音楽][3.00][40.14][S01]い[41.15]"
    forced = "[0.10][音楽][3.00][5.17][S01]う[39.16][40.14][S01]い[41.15]"
    segs, events, _ = run_pass(monkeypatch, [skipped, forced], seconds=45.0, spans=[(5.17, 41.2)], window=True,
                               hole=(0.0, 39.5))
    assert [(e["kind"], e["forced"]) for e in events] == [("hole-skip", 5.17)]
    assert segs == [(5.17, 39.16, "S01", "う"), (40.14, 41.15, "S01", "い")]


def test_a_window_is_forced_back_to_a_stretch_of_talk_not_a_blip(monkeypatch):
    skipped = "[40.14][S01]い[41.15]"
    forced = "[35.00][S01]う[39.16][40.14][S01]い[41.15]"
    segs, events, _ = run_pass(monkeypatch, [skipped, forced], seconds=45.0, spans=[(2.0, 2.5), (35.0, 44.6)],
                               window=True, hole=(0.0, 39.5))
    assert [(e["kind"], e["forced"]) for e in events] == [("hole-skip", 35.0)]


def test_a_lead_in_line_running_into_the_hole_is_not_written_again(monkeypatch):
    # the window's first line starts in the lead-in and runs 12 s into the hole: that talk is said, the next is on time
    through = "[1.10][S01]あ[14.10][14.50][S01]い[20.30]"
    segs, events, _ = run_pass(monkeypatch, [through], seconds=45.0, spans=[(1.0, 44.0)], window=True,
                               hole=(2.0, 44.0))
    assert events == [] and [s[2:] for s in segs] == [("S01", "あ"), ("S01", "い")]


def test_talk_only_in_the_lead_in_forces_nothing(monkeypatch):
    lead_in_then_after = "[0.50][S01]あ[8.90][44.00][S01]い[44.50]"
    segs, events, _ = run_pass(monkeypatch, [lead_in_then_after], seconds=45.0, spans=[(0.1, 9.0)], window=True,
                               hole=(10.0, 44.0))
    assert events == [] and len(segs) == 2


def test_a_forced_hole_start_drops_a_re_decode_mode_and_skips_talk_a_loop_gave_up(monkeypatch):
    _, _, p = run_pass(monkeypatch, ["[40.14][S01]い[41.15]"], seconds=45.0, spans=[(5.17, 41.2)], window=True)
    o = rescue._Seg(0, 0, 40.14, "S01")
    p.ids, p.segs, p.mode, p.hole = p.tok.encode("[40.14][S01]い[41.15]"), [], ("penalty", 0), (0.0, 39.5)
    assert p._skipped_hole(o) and p.mode is None and p.events[-1]["forced"] == 5.17
    p.ids, p.segs, p.hole, p.floor = p.tok.encode("[40.14][S01]い[41.15]"), [], (0.0, 39.5), 20.0
    assert p._skipped_hole(o) and p.events[-1]["forced"] == 20.0


def test_a_window_is_checked_once_at_its_first_line_in_the_hole(monkeypatch):
    on_time_then_late = "[1.10][S01]あ[2.00][30.00][S01]い[31.00]"
    segs, events, _ = run_pass(monkeypatch, [on_time_then_late], seconds=60.0, spans=[(1.0, 60.0)], window=True,
                               hole=(0.0, 60.0))
    assert events == [] and len(segs) == 2


def test_only_a_long_holes_window_gets_its_hole(monkeypatch):
    holes = []

    class FakePass:
        def __init__(self, model, processor, audio, spans, prompt, language, budget, window=False, hole=None):
            self.spans, self.events, self.capped, self.acoustic, self.window = spans, [], False, [], window
            if window:
                holes.append(hole)

        def voiced(self, a, b):
            return b - a

        onset = rescue.Pass.onset

        def run(self):
            return [] if self.window else [seg(0.0, 20.0, "a"), seg(25.0, 100.0, "b"), seg(160.0, 200.0, "c")]

    monkeypatch.setattr(rescue, "Pass", FakePass)
    monkeypatch.setattr(rescue.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(RL.VOICEPRINTS_CUDA, "release", lambda: None)
    rescue._part(None, None, np.zeros(200 * D.SR, np.float32), None, [(0.0, 200.0)], "en", "", 1.0, "")
    assert holes[0] is None  # 20-25 s, a short gap
    assert holes[1] == pytest.approx((75.0, 135.0))  # 100-160 s, in a window from b's start at 25 s


def test_eos_inside_a_segment_is_masked_and_the_segment_goes_on(monkeypatch):
    script = "[0.12][S01]あいう[3.13][3.14][S02]えお[4.15]"
    segs, events, _ = run_pass(monkeypatch, [script], seconds=5.0, spans=[(0.1, 4.2)], ends=("[0.12][S01]あい",))
    assert segs == [(0.12, 3.13, "S01", "あいう"), (3.14, 4.15, "S02", "えお")]
    assert [e["kind"] for e in events] == ["eos-mid"]


def test_eos_that_keeps_coming_inside_a_segment_ends_the_pass_at_the_last_whole_one(monkeypatch):
    segs, events, _ = run_pass(monkeypatch, ["[0.12][S01]あ[1.13][1.14][S01]い"], seconds=5.0, spans=[(0.1, 4.2)])
    assert segs == [(0.12, 1.13, "S01", "あ")]
    assert [e["kind"] for e in events] == ["eos-mid"] * rescue.EOS_MASKS + ["eos-mid-end"]


def test_an_end_with_talk_after_it_is_forced_on_after_a_closing_bracket_and_whitespace(monkeypatch):
    ended = "[0.12][S01]あ[1.13]\n"
    forced = "[0.12][S01]あ[1.13][40.11][S01]い[49.12]"
    segs, events, _ = run_pass(monkeypatch, [ended, forced], seconds=100.0, spans=[(0.1, 1.13), (40.11, 50.11)])
    assert [(e["kind"], e["forced"]) for e in events] == [("eos", 40.11)]
    assert segs == [(0.12, 1.13, "S01", "あ"), (40.11, 49.12, "S01", "い")]


def test_an_end_forced_on_opens_the_way_the_model_opens_its_segments(monkeypatch):
    ended = "[0.12][S01]あ[1.13] [1.15][S01]い[1.18]"
    forced = "[0.12][S01]あ[1.13] [1.15][S01]い[1.18] [40.11][S01]う[49.12]"
    segs, events, p = run_pass(monkeypatch, [forced], seconds=100.0, spans=[(0.1, 1.2), (40.11, 50.11)], ends=(ended,))
    assert [(e["kind"], e["forced"]) for e in events] == [("eos", 40.11)]
    assert segs[-1] == (40.11, 49.12, "S01", "う") and p.tok.decode(p.ids) == forced


def test_a_run_of_timestamps_outside_a_segment_is_decoded_again_then_given_up(monkeypatch):
    stamps = "[0.12][S01]あ[1.13]" + "".join(f"[{1.13 + 0.14 * k:.2f}]" for k in range(20))
    after = "[0.12][S01]あ[1.13][11.13][S01]い[12.14]"
    segs, events, _ = run_pass(monkeypatch, [stamps, after], seconds=15.0, spans=[(0.1, 13.3)])
    assert [(e["kind"], e["reach"]) for e in events] == [("unparsed", 1.13), ("unparsed-skip", 1.13)]
    assert segs == [(0.12, 1.13, "S01", "あ"), (11.13, 12.14, "S01", "い")]


def test_a_pass_that_parses_nothing_after_a_stretch_given_up_ends_there(monkeypatch):
    stamps = "[0.12][S01]あ[1.13]" + "".join(f"[{1.13 + 0.14 * k:.2f}]" for k in range(20))
    still = "[0.12][S01]あ[1.13][11.13]" + "".join(f"[{11.13 + 0.14 * k:.2f}]" for k in range(20))
    segs, events, _ = run_pass(monkeypatch, [stamps, still], seconds=30.0, spans=[(0.1, 28.3)])
    assert [e["kind"] for e in events] == ["unparsed", "unparsed-skip", "unparsed", "unparsed-end"]
    assert segs == [(0.12, 1.13, "S01", "あ")]


def test_text_with_no_head_after_a_segment_is_decoded_again_from_a_forced_start(monkeypatch):
    # the model opened its last segment with ` [`: the forced start is opened so too
    headless = "[0.12][S01]あ[1.13] [1.15][S01]い[1.18] なんだろうずっとしゃべっているのに時間も話者も書いていない[9.15]"
    forced = "[0.12][S01]あ[1.13] [1.15][S01]い[1.18] [1.21][S02]なんだろう[3.08]"
    segs, events, _ = run_pass(monkeypatch, [headless, forced], seconds=5.0, spans=[(0.1, 0.5), (1.21, 3.1)])
    assert [(e["kind"], e["forced"]) for e in events] == [("unparsed", 1.21)]
    assert segs == [(0.12, 1.13, "S01", "あ"), (1.15, 1.18, "S01", "い"), (1.21, 3.08, "S02", "なんだろう")]


def test_a_window_that_stops_short_of_its_hole_leaves_the_rest_to_another_window(monkeypatch):
    windows = []

    class FakePass:
        def __init__(self, model, processor, audio, spans, prompt, language, budget, window=False, hole=None):
            self.spans, self.events, self.capped, self.acoustic = spans, [], False, []
            self.lo = None if not window else len(windows)
            if window:
                windows.append(len(audio) / D.SR)

        def voiced(self, a, b):
            return b - a

        onset = rescue.Pass.onset

        def run(self):
            if self.lo is None:  # the main pass: up to 20 s of a 200 s source
                return [seg(0.0, 20.0, "main")]
            # the first window, over 0-200 s, stops at 79 s; the next, over 21-200 s, writes the rest
            if self.lo == 0:
                return [seg(0.0, 20.0, "main"), seg(21.0, 79.0, "first window")]
            return [seg(59.0, 178.0, "second window")]

    monkeypatch.setattr(rescue, "Pass", FakePass)
    monkeypatch.setattr(rescue.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(RL.VOICEPRINTS_CUDA, "release", lambda: None)
    segs, events = rescue._part(None, None, np.zeros(200 * D.SR, np.float32), None, [(0.0, 200.0)], "en", "", 1.0, "")
    assert windows == [200.0, 179.0]  # 0-200 s, then from the first window's line at 21 s
    assert [s["text"] for s in segs] == ["main", "first window", "second window"]
    assert [(e["kind"], e["gap"]) for e in events if e["kind"] == "fill"] == [("fill", [20.0, 200.0]),
                                                                               ("fill", [79.0, 200.0])]


def test_a_window_that_hits_its_token_budget_goes_on_from_its_last_segment_without_a_lead_in(monkeypatch):
    # codex acceptance review: the continuation starts at 79 s, the first window's last segment end, though 79 s
    # lies in a pause (76-80 s)
    windows, holes = [], []

    class FakePass:
        def __init__(self, model, processor, audio, spans, prompt, language, budget, window=False, hole=None):
            self.spans, self.events, self.acoustic = spans, [], []
            self.lo = None if not window else len(windows)
            self.capped = self.lo == 0
            if window:
                windows.append(len(audio) / D.SR)
                holes.append(hole)

        def voiced(self, a, b):
            return b - a

        onset = rescue.Pass.onset

        def run(self):
            if self.lo is None:
                return [seg(0.0, 20.0, "main")]
            if self.lo == 0:
                return [seg(0.0, 20.0, "main"), seg(21.0, 79.0, "first window")]
            return [seg(0.0, 121.0, "continued")]

    monkeypatch.setattr(rescue, "Pass", FakePass)
    monkeypatch.setattr(rescue.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(RL.VOICEPRINTS_CUDA, "release", lambda: None)
    spans = [(0.0, 76.0), (80.0, 200.0)]
    segs, events = rescue._part(None, None, np.zeros(200 * D.SR, np.float32), None, spans, "en", "", 1.0, "")
    assert windows == [200.0, 121.0]  # 0-200 s, then 79-200 s
    assert holes == [pytest.approx((20.0, 200.0)), pytest.approx((0.0, 121.0))]  # a continuation's hole starts it
    assert [s["text"] for s in segs] == ["main", "first window", "continued"]


def test_labels_of_one_voice_merge_into_the_one_with_more_speech():
    segs = [seg(0, 10, "a", "S01"), seg(20, 22, "b", "S02"), seg(30, 40, "c", "S03")]
    s = math.sqrt(0.19)
    merges = RL.merge_labels(segs, _centroids({"S01": [1, 0, 0], "S02": [0.9, s, 0], "S03": [0, 0, 1]}), 0.8)
    assert merges == [("S02", "S01", 0.9)]
    assert [x["speaker"] for x in segs] == ["S01", "S01", "S03"]


def test_a_merge_keeps_the_label_heard_first_whatever_talks_more():
    # the judge's profile reads where a label came from (`prompt.origin`): a person heard in the first part is not
    # "only in part 2", nor a main-transcript person "a re-ask's"
    for short, long in [("S01", "w1:S04"), ("w1:S02", "w2:S05"), ("S01", "c1:S03"), ("w1:S02", "c1:S03")]:
        segs = [seg(0, 5, "a", short), seg(10, 90, "b", long)]
        merges = RL.merge_labels(segs, _centroids({short: [1, 0], long: [1, 0]}), 0.8)
        assert merges == [(long, short, 1.0)], (short, long)


def test_two_labels_talking_over_each_other_stay_two_people():
    segs = [seg(0, 10, "a", "S01"), seg(2, 6, "b", "S02")]
    assert RL.merge_labels(segs, _centroids({"S01": [1, 0], "S02": [1, 0]}), 0.8) == []
    assert {x["speaker"] for x in segs} == {"S01", "S02"}


# --- the adapter -----------------------------------------------------------------------------------------------


def test_event_counts_walk_parts_and_windows_and_skip_the_kv_records():
    events = [{"kind": "kv"}, {"kind": "skip"},
              {"kind": "fill", "events": [{"kind": "kv"}, {"kind": "loop-penalty"}]},
              {"kind": "part", "events": [{"kind": "kv"}, {"kind": "skip"}, {"kind": "short", "events": []}]},
              {"kind": "merge"}]
    assert ML.event_counts(events) == {"fill": 1, "loop-penalty": 1, "merge": 1, "short": 1, "skip": 2}


def test_a_later_japanese_parts_main_pass_gets_the_example_and_its_windows_do_not(monkeypatch, tmp_path):
    from finesub.speech.joint.mosspro import SpeechSpans

    seen = []

    def part(model, processor, audio, pcm, spans, language, prompt, budget, main_prompt):
        seen.append((prompt, main_prompt))
        return [], [{"kind": "kv", "format": "fp16", "positions": 0}]

    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: 1000.0)
    monkeypatch.setattr(rescue, "PART_SECONDS", 1000.0)
    monkeypatch.setattr(rescue, "_part", part)
    monkeypatch.setattr(RL, "merge_by_voice", lambda *a, **k: ([], []))
    dur = 2500
    heard = [(0.0, 899.5), (900.5, 979.0), (981.0, 1879.0), (1881.0, 2500.0)]
    speech = SpeechSpans(tmp_path / "a.wav", heard, np.zeros(dur * D.SR, np.int16))
    rescue.transcribe(None, None, np.zeros(dur * D.SR, np.float32), "ja", 1.0, speech)
    later = D.DEFAULT_PROMPT + D.MIDSTART["ja"]
    assert seen == [(D.DEFAULT_PROMPT, D.DEFAULT_PROMPT), (D.DEFAULT_PROMPT, later), (D.DEFAULT_PROMPT, later)]


def test_the_main_pass_takes_the_main_prompt_and_windows_the_plain_one(monkeypatch):
    prompts = []

    class FakePass:
        def __init__(self, model, processor, audio, spans, prompt, language, budget, window=False, hole=None):
            self.spans, self.events, self.capped, self.acoustic, self.window = spans, [], False, [], window
            prompts.append((window, prompt))

        def voiced(self, a, b):
            return b - a

        onset = rescue.Pass.onset

        def run(self):
            return [seg(1.0, 150.0, "window")] if self.window else [seg(0.0, 20.0, "main")]

    monkeypatch.setattr(rescue, "Pass", FakePass)
    monkeypatch.setattr(rescue.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(RL.VOICEPRINTS_CUDA, "release", lambda: None)
    rescue._part(None, None, np.zeros(200 * D.SR, np.float32), None, [(0.0, 200.0)], "en", "plain", 1.0, "main")
    assert prompts[0] == (False, "main") and prompts[1:] and all(p == (True, "plain") for p in prompts[1:])


def test_comma_re_decodes_get_the_later_part_prompt_in_every_part(monkeypatch):
    got = []

    class FakePass:
        def __init__(self, model, processor, audio, spans, prompt, language, budget, window=False, hole=None):
            self.spans, self.events, self.capped, self.acoustic, self.window = spans, [], False, [], window

        def voiced(self, a, b):
            return b - a

        onset = rescue.Pass.onset

        def run(self):
            return [] if self.window else [seg(0.0, 200.0, "main")]

    monkeypatch.setattr(rescue, "Pass", FakePass)
    monkeypatch.setattr(rescue, "punctuate", lambda *a: got.append(a[6]) or (a[4], []))
    monkeypatch.setattr(rescue.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(RL.VOICEPRINTS_CUDA, "release", lambda: None)
    for language in ("ja", None):  # no --language: the Japanese segments still get the Japanese example
        rescue._part(None, None, np.zeros(200 * D.SR, np.float32), None, [(0.0, 200.0)], language, D.DEFAULT_PROMPT,
                     1.0, D.DEFAULT_PROMPT)  # the first part: its main pass has no example
    assert got == [D.prompt_for("ja", later=True)] * 2


def test_prompts_follow_the_language_and_default_to_upstreams():
    assert D.prompt_for("en").startswith("Transcribe the audio.")
    assert D.prompt_for("EN") == D.prompt_for("en")
    for language in (None, "ja", "zh", "de"):
        assert D.prompt_for(language) == D.DEFAULT_PROMPT
    assert D.prompt_for("ja", later=True) == D.prompt_for("JA-jp", later=True) == D.DEFAULT_PROMPT + D.MIDSTART["ja"]
    for language in (None, "zh", "en"):  # no example of their own yet
        assert D.prompt_for(language, later=True) == D.prompt_for(language)


@pytest.fixture
def fake_gpu(monkeypatch, tmp_path):
    """Everything the adapter touches on the card, recorded: the allocator cap, the model load, the decode."""
    import torch

    state = {"fractions": [], "loaded": 0, "prepared": 0, "decode": None}
    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev, backend: None)
    monkeypatch.setattr(device_module, "total_vram_gib", lambda: 16.0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, card=None: state["fractions"].append(round(fraction, 4)))
    monkeypatch.setattr(D, "check_transformers", lambda backend: None)

    def prepare(manifest_id):
        state["prepared"] += 1
        return hf_weights.HfLoad("rev123", True)

    def load(plan):
        state["loaded"] += 1
        return "model", "processor"

    def transcribe(model, processor, audio, language, budget_gib, speech):
        state["budget"] = budget_gib
        if state["decode"] is not None:
            raise state["decode"]
        assert joint.GPU_LOCK.locked()
        return rescue.Result([seg(1.0, 2.0, "hello", "S01"), seg(2.5, 2.5, "", "S02"), seg(3.0, 5.0, "bye", "S02")],
                             [{"kind": "kv"}, {"kind": "eos"}], [{"start": 0.0, "kv": "bf16", "positions": 8192}])

    monkeypatch.setattr(hf_weights, "prepare", prepare)
    monkeypatch.setattr(D, "load", load)
    monkeypatch.setattr(rescue, "transcribe", transcribe)
    monkeypatch.setattr(ML, "_decode", lambda source: np.zeros(16000, np.float32))
    monkeypatch.setattr(ML, "_quiet_transformers", nullcontext)
    monkeypatch.setattr(ML, "_triton_cache_in_checkout", nullcontext)
    return state


def test_the_adapter_hands_back_segments_and_what_the_metadata_records(fake_gpu, tmp_path):
    transcript, route = ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="standard",
                                      device=None)
    assert (transcript.provider, transcript.unit, transcript.duration) == ("moss-local", "segment", 6.0)
    assert [(u.start, u.end, u.speaker) for u in transcript.units()] == [(1.0, 2.0, "S01"), (3.0, 5.0, "S02")]
    assert route == {"revision": "rev123", "kv": [{"start": 0.0, "kv": "bf16", "positions": 8192}],
                     "events": {"eos": 1}}
    from finesub.speech.runtime.resources import get_resource_profile

    assert fake_gpu["budget"] == pytest.approx(get_resource_profile("standard").usable_gib - D.CONTEXT_GIB)
    assert fake_gpu["fractions"][-1] == 1.0 and fake_gpu["fractions"][0] < 1.0
    assert not joint.GPU_LOCK.locked()


def test_a_failed_decode_gives_back_the_cap_and_the_lock(fake_gpu, tmp_path):
    fake_gpu["decode"] = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="standard", device=None)
    assert fake_gpu["fractions"] == [fake_gpu["fractions"][0], 1.0]
    assert not joint.GPU_LOCK.locked()


@pytest.mark.parametrize("chained", [False, True])
def test_a_failed_decode_frees_the_model_before_the_lock_goes(fake_gpu, tmp_path, monkeypatch, chained):
    import weakref

    class Model:
        pass

    box = [Model()]
    gone = weakref.ref(box[0])
    monkeypatch.setattr(D, "load", lambda plan: (box.pop(), "processor"))

    def deeper(model):
        held = [model]  # a frame local, as `Pass` holds the model and its K/V buffers  # noqa: F841
        raise RuntimeError("CUDA out of memory")

    def transcribe(model, processor, audio, language, budget_gib, speech):
        if not chained:
            deeper(model)
        try:
            deeper(model)
        except RuntimeError as e:  # a handler that re-raises: the first one's frames ride along as its cause
            raise RuntimeError("CUDA out of memory, again") from e

    monkeypatch.setattr(rescue, "transcribe", transcribe)
    import torch

    at_release = []  # whether the model was still alive when the step gave the cache back, still holding the lock
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: at_release.append((gone() is not None,
                                                                              joint.GPU_LOCK.locked())))
    with pytest.raises(RuntimeError, match="out of memory"):
        ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="standard", device=None)
    assert at_release == [(False, True)]


def test_a_run_that_cannot_use_it_stops_before_any_download(fake_gpu, tmp_path, monkeypatch):
    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev, backend: "--asr moss-local needs X")
    with pytest.raises(joint_stage.JointAsrError, match="needs X"):
        ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="cpu", device=None)
    assert fake_gpu["prepared"] == 0 and fake_gpu["loaded"] == 0 and fake_gpu["fractions"] == []


def test_without_a_language_it_says_what_that_costs(fake_gpu, tmp_path, capsys):
    ML.transcribe(tmp_path / "a.wav", language=None, duration=6.0, gpu_tier="standard", device=None)
    assert "Warning: --asr moss-local without --language" in capsys.readouterr().err


def test_the_sdpa_swap_lasts_only_while_the_prefill_runs():
    hf_sdpa = pytest.importorskip("transformers.integrations.sdpa_attention")
    original = hf_sdpa.use_gqa_in_sdpa
    with pytest.raises(ValueError):
        with D._expanded_kv_sdpa():
            assert hf_sdpa.use_gqa_in_sdpa(None, None) is False
            raise ValueError("prefill failed")
    assert hf_sdpa.use_gqa_in_sdpa is original


def test_triton_compiles_into_the_checkout_cache_only_during_the_step(monkeypatch, tmp_path):
    import os

    import finesub.paths

    monkeypatch.setattr(finesub.paths, "resolve_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    with ML._triton_cache_in_checkout():
        assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "cache" / "moss-local-triton")
    assert "TRITON_CACHE_DIR" not in os.environ  # Inductor reads it too: the separator's kernels go elsewhere
    monkeypatch.setenv("TRITON_CACHE_DIR", "mine")
    with ML._triton_cache_in_checkout():
        assert os.environ["TRITON_CACHE_DIR"] == "mine"
    assert os.environ["TRITON_CACHE_DIR"] == "mine"


class _Config:
    num_hidden_layers, num_key_value_heads, head_dim, max_position_embeddings = 28, 8, 128, 131072


class _Model:
    class model:
        class language_model:
            config = _Config


@pytest.mark.parametrize("tier, minutes", [("standard", 28), ("high", 57)])
def test_a_pass_is_as_long_as_the_tiers_vram_holds(monkeypatch, tier, minutes):
    import torch

    from finesub.speech.runtime.resources import get_resource_profile

    monkeypatch.delenv("MOSS_FAST_KV", raising=False)
    monkeypatch.setattr(D, "fp8_kernels", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: int(1.69 * 2**30))  # the weights
    budget = get_resource_profile(tier).usable_gib - D.CONTEXT_GIB
    assert D.longest_pass(_Model, rescue.TOKENS_PER_SECOND, budget) == 60.0 * minutes


def test_the_entry_tier_holds_no_minute_so_the_check_refuses_it_up_front(monkeypatch, nvidia):
    import torch

    from finesub.speech.runtime.resources import get_resource_profile

    monkeypatch.setattr(D, "fp8_kernels", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: int(1.69 * 2**30))
    with pytest.raises(RuntimeError, match="holds no minute"):
        D.longest_pass(_Model, rescue.TOKENS_PER_SECOND, get_resource_profile("entry").usable_gib - D.CONTEXT_GIB)
    assert "GPU tier (entry) gives it 3 GiB" in device_module.moss_local_unusable_reason("entry", None)


# --- the device criterion ----------------------------------------------------------------------------------


@pytest.fixture
def nvidia(monkeypatch):
    import torch

    from finesub.speech.runtime import resources

    card = {"capability": (8, 9), "unusable": None, "rocm": False}
    monkeypatch.setattr(resources, "detect_gpu_tier", lambda **k: "standard")
    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: card["unusable"])
    monkeypatch.setattr(device_module, "is_rocm", lambda: card["rocm"])
    monkeypatch.setattr(device_module, "gpu_name", lambda index=None: "Test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: card["capability"])
    return card


def test_an_ampere_or_newer_nvidia_card_runs_it(nvidia):
    assert device_module.moss_local_unusable_reason(None, None) is None
    assert device_module.moss_local_unusable_reason("auto", "cuda") is None
    nvidia["capability"] = (8, 0)
    assert device_module.moss_local_unusable_reason("standard", None) is None


@pytest.mark.parametrize("tier, device, card, says", [
    (None, "cpu", {}, "--device cpu"),
    ("cpu", None, {}, "--gpu-tier cpu"),
    (None, None, {"unusable": "it is unavailable"}, "the GPU cannot be used (it is unavailable)"),
    (None, None, {"rocm": True}, "Test GPU is an AMD card"),
    (None, None, {"capability": (7, 5)}, "Test GPU has compute capability 7.5"),
])
def test_anything_else_is_refused_with_the_reason_and_the_way_out(nvidia, tier, device, card, says):
    nvidia.update(card)
    reason = device_module.moss_local_unusable_reason(tier, device)
    assert says in reason
    assert reason.startswith("--asr moss-local needs an NVIDIA card of compute capability 8.0")
    assert "--asr local" in reason


# --- the stage and the batch ----------------------------------------------------------------------------------


@pytest.fixture
def talk(tmp_path, monkeypatch) -> Path:
    import soundfile as sf

    from finesub.media import ffmpeg

    # CI's Linux job has no ffmpeg; the speech probabilities are stubbed in the test, so silence will do
    monkeypatch.setattr(ffmpeg, "probe_media_duration", lambda src: 8.0)
    monkeypatch.setattr(ffmpeg, "decode_pcm16k", lambda src, **_: bytes(8 * 16000 * 2))

    sr = 16000
    t = np.arange(int(sr * 8.0)) / sr
    path = tmp_path / "talk.wav"
    sf.write(path, (0.1 * np.sin(2 * math.pi * 220 * t)).astype(np.float32), sr)
    return path


def test_the_stage_builds_its_lines_as_moss_pros_and_records_the_run(talk, tmp_path, monkeypatch):
    pytest.importorskip("nagisa")  # Japanese lines split by word group
    from finesub.speech.joint import moss_timing as T

    seen = {}

    def transcribe(source, *, language, duration, gpu_tier, device, speech):
        seen.update(gpu_tier=gpu_tier, device=device, speech=speech)
        units = [Unit(0.5, 3.0, "こんにちは、今日はいい天気ですね。", "S01"), Unit(3.5, 6.0, "そうですね。", "S02")]
        return (Transcript("moss-local", ML.MODEL_NAME, duration, "segment", segment_turns(units)),
                {"revision": "rev", "kv": [{"start": 0.0, "kv": "bf16", "positions": 4096}], "events": {"eos": 1}})

    monkeypatch.setattr(ML, "transcribe", transcribe)
    monkeypatch.setattr(T, "frame_probabilities", lambda pcm, tail=False: np.full(len(pcm) // T.HOP, 0.9, np.float32))
    from finesub.speech.joint import sbd

    monkeypatch.setattr(sbd, "loaded", lambda: None)  # a large download; here as if it could not load
    out = joint_stage.run(talk, backend="moss-local", output_path=tmp_path / "a-aligned.json",
                          cache_root=tmp_path / "cloud", language="ja", repair="on", gpu_tier="high", device="cuda")
    payload = json.loads(out.read_text(encoding="utf-8"))
    record = payload["metadata"][joint_stage.METADATA_KEY]
    assert {k: record[k] for k in ("backend", "model", "revision", "kv", "events")} == {
        "backend": "moss-local", "model": ML.MODEL_NAME, "revision": "rev",
        "kv": [{"start": 0.0, "kv": "bf16", "positions": 4096}], "events": {"eos": 1}}
    # MOSS Pro's line building, aligned by default (`--asr-aligner auto`; no segment here is long enough), no repair
    assert (record["timing"], record["aligned"]) == ("S6b", 0) and "repair" not in record
    assert record["sentence_boundary_model"] is False
    # one Silero pass serves the transcription and the line timing
    assert {k: seen[k] for k in ("gpu_tier", "device")} == {"gpu_tier": "high", "device": "cuda"}
    assert type(seen["speech"]).__name__ == "SpeechSpans" and seen["speech"]._scored is not None
    assert "".join(s["text"] for s in payload["segments"]) == "こんにちは、今日はいい天気ですね。そうですね。"


def test_a_local_moss_task_takes_the_gpu_lane(tmp_path, monkeypatch):
    from finesub.pipeline import build_item
    from finesub.scheduler import GPU_LANE

    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev, backend: None)
    monkeypatch.setattr(D, "check_transformers", lambda backend: None)  # the root CI suite has no [asr]
    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    item = build_item({"source": str(source), "stage": "raw-srt", "output": str(tmp_path / "o" / "talk.srt"),
                       "asr": "moss-local"})
    assert item.asr_lane(item.payload) == GPU_LANE


def test_a_run_that_cannot_decode_with_it_stops_when_the_task_is_built(tmp_path):
    from finesub.pipeline import build_item
    from finesub.stages import default_pipeline_paths

    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    output = tmp_path / "o" / "talk.srt"
    row = {"source": str(source), "stage": "raw-srt", "output": str(output), "asr": "moss-local", "gpu_tier": "cpu"}
    with pytest.raises(ValueError, match="--gpu-tier cpu keeps this run off the GPU"):
        build_item(row)
    build_item({**row, "stage": "vocal"})  # nothing to decode at this stage
    build_item({**row, "asr": "local"})
    aligned = default_pipeline_paths(source, output_path=output).aligned_json
    aligned.parent.mkdir(parents=True)
    aligned.write_text(json.dumps({"segments": [], "metadata": {"joint_asr": {"backend": "moss-local"}}}),
                       encoding="utf-8")
    build_item(row)  # decoded already: an LLM run of it needs no card
    aligned.unlink()
    paths = default_pipeline_paths(source, output_path=output)
    paths.stable_json.write_text(json.dumps({"segments": [], "metadata": {"joint_asr": {"backend": "moss-local"}}}),
                                 encoding="utf-8")
    build_item(row)  # past aligned already
    with pytest.raises(ValueError, match="--gpu-tier cpu"):
        build_item({**row, "stage": "aligned"})  # asked for the aligned again: it decodes


def test_a_linked_source_is_checked_where_the_run_will_look(tmp_path, monkeypatch):
    import os

    from finesub import stages
    from finesub.pipeline import _refuse_local_moss_that_cannot_run

    monkeypatch.setattr(stages, "default_output_path", lambda p: tmp_path / "out" / p.stem / f"{p.stem}.srt")
    target = tmp_path / "talk.wav"
    target.write_bytes(b"")
    link = tmp_path / "alias.wav"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("no symlinks here")
    stable = stages.default_pipeline_paths(target).stable_json
    stable.parent.mkdir(parents=True)
    stable.write_text(json.dumps({"segments": [], "metadata": {"joint_asr": {"backend": "moss-local"}}}),
                      encoding="utf-8")
    _refuse_local_moss_that_cannot_run(str(link), None, {"asr": "moss-local", "gpu_tier": "cpu"}, "raw-srt")


def test_a_url_without_an_output_counts_as_decoding_with_this_runs_backend():
    from finesub.pipeline import _refuse_local_moss_that_cannot_run

    shared = {"asr": "moss-local", "gpu_tier": "cpu"}
    with pytest.raises(ValueError, match="--gpu-tier cpu"):
        _refuse_local_moss_that_cannot_run("https://example.com/v", None, shared, "raw-srt")
    _refuse_local_moss_that_cannot_run("https://example.com/v", None, {**shared, "asr": "moss-pro"}, "raw-srt")


def test_the_aligner_applies_to_local_moss(tmp_path, capsys):
    from finesub.stages import _note_task_backend, default_pipeline_paths

    paths = default_pipeline_paths(tmp_path / "talk.wav", tmp_path / "out" / "talk.srt")
    _note_task_backend(paths, "moss-local", "moss-local", "on")
    assert "--asr-aligner" not in capsys.readouterr().err
