"""The repairs of a cloud transcript (`speech/joint/repair.py`, `live.py`, the adapters' repair steps).

No model, no network: Live, Silero and the voiceprint model are fakes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from finesub.api_keys import ApiKeyEntry
from finesub.speech.joint import gemini35t, live, moss_timing, mosspro
from finesub.speech.joint import repair as R
from finesub.speech.joint.cache import CacheMiss, CallCache


def u(start, end, text, speaker="A"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# --- detection ----------------------------------------------------------------------------------------------


def test_gaps_are_the_surveys_and_only_the_ones_silero_hears_speech_in():
    units = [u(0.0, 1.0, "a"), u(3.4, 4.0, "b"), u(10.0, 11.0, "c")]
    heard = {(1.3, 3.1): 0.0, (4.3, 9.7): 2.0}
    found = R.gaps(units, 12.0, lambda a, b: heard.get((round(a, 1), round(b, 1)), 0.0))
    # 1.0-3.4 widened by PAD is a 1.8 s gap (under GAP_MIN); 11.3-12.0 is under GAP_MIN too
    assert [(round(g.start, 1), round(g.end, 1)) for g in found] == [(4.3, 9.7)]


def test_merged_gaps_stop_at_the_longest_span():
    gaps = [R.Gap(10 + 50 * k, 20 + 50 * k, 5.0) for k in range(4)]
    assert [(a, b) for a, b, _ in R.merged(gaps, longest=100)] == [(10, 70), (110, 170)]


def test_gaps_closer_than_a_minute_merge_for_timed_re_asks():
    gs = [R.Gap(0, 5, 2), R.Gap(50, 55, 2), R.Gap(200, 205, 2)]
    assert [(a, b, len(g)) for a, b, g in R.merged(gs)] == [(0, 55, 2), (200, 205, 1)]


def test_runs_of_ten_or_more_and_which_are_too_fast_to_be_said():
    oh = [u(1.0 + 0.05 * k, 1.05 + 0.05 * k, "Oh") for k in range(30)]
    go = [u(10.0 + 0.3 * k, 10.25 + 0.3 * k, "走") for k in range(10)]
    few = [u(20.0 + 0.3 * k, 20.25 + 0.3 * k, "对") for k in range(9)]
    found = R.runs(oh + go + few)
    assert [(r.unit, r.times, r.fast) for r in found] == [("oh", 30, True), ("走", 10, False)]


def test_a_fast_run_keeps_four_repetitions_a_second_spread_evenly():
    oh = [u(1.0 + 0.05 * k, 1.05 + 0.05 * k, "Oh") for k in range(30)]
    run = R.runs(oh)[0]
    patch = R.patch_fast(oh, run)
    assert len(patch) == 6  # 1.5 s x 4
    assert patch[0]["start"] == 1.0 and patch[-1]["end"] == pytest.approx(2.5)
    assert len({round(p["end"] - p["start"], 3) for p in patch}) == 1


def test_the_longest_run_not_the_total_decides_a_live_replacement():
    assert R.longest_run("哇哦。走走走走走走。吓我一跳。走走走走走。", "走") == 6
    assert R.longest_run("whoa whoa whoa", "woah") == 0  # another spelling is another unit
    assert R.longest_run("always Russian", "oh") == 0


def test_echo_of_the_neighbours_is_trimmed_from_both_edges():
    texts = ["について", "一緒", "に", "新しい", "話", "では"]
    assert R.trim_echo(texts, "これについて一緒に", "ではまた") == (3, 5)
    assert R.trim_echo(["あ", "い"], "あ", "") == (0, 2)  # one character is not an echo


def test_live_text_between_the_anchors_is_new_and_variants_of_them_are_not():
    left, right = R.key("今天天气不错。"), R.key("我们走吧")
    assert R.anchored("今天天气不错。出现了今天的第一个bug。我们走吧", left, right) == ("出现了今天的第一个bug。", None)
    assert R.anchored("这车没见过。车迷见过。我们走吧", R.key("这车没见过"), right) == ("", "echo")
    assert R.anchored("今天天气不错我们走吧", left, right) == ("", "nothing_between")
    assert R.anchored("完全不同的一句话", left, right) == ("", "unheard")  # no telling where the gap is
    assert R.anchored("That's fascinating. That's a fascinating thing", R.key("That's fascinating"), "") == ("", "echo")
    assert R.anchored("我们走吧。下次再说", "", right) == ("", "nothing_between")  # before the first word: nothing new
    assert R.anchored("先说一句。我们走吧", "", right) == ("先说一句。", None)


def test_implausible_text():
    assert R.implausible("。、", 3.0, "zh") == "empty"
    assert R.implausible("あ" * 80, 2.0, "ja") == "too_dense"
    assert R.implausible("これは日本語です", 3.0, "en") == "script"
    assert R.implausible("打球吧", 3.0, "zh") is None


def test_blocks_split_each_speaker_at_pauses_and_drop_short_ones():
    units = [u(0.0, 0.4, "a"), u(0.5, 1.2, "b"), u(2.0, 2.3, "c"), u(0.1, 1.5, "x", "B"), u(5, 6, "n", None)]
    assert R.blocks(units) == [{"start": 0.0, "end": 1.2, "speaker": "A"}, {"start": 0.1, "end": 1.5, "speaker": "B"}]


def test_mode_rejects_unknown_values():
    assert R.mode(" ON ") == "on"
    with pytest.raises(ValueError):
        R.mode("auto")


# --- the Live client ----------------------------------------------------------------------------------------


class FakeSocket:
    def __init__(self, replies):
        self.sent, self.replies, self.header, self.closed = [], list(replies), None, False

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self):
        if not self.replies:
            raise type("WebSocketTimeoutException", (Exception,), {})()
        return self.replies.pop(0)

    def settimeout(self, _):
        pass

    def close(self):
        self.closed = True


def transcription(text):
    return json.dumps({"serverContent": {"inputTranscription": {"text": text}}})


DONE = json.dumps({"serverContent": {"generationComplete": True}})


def test_live_sends_the_key_in_a_header_and_joins_the_text(monkeypatch):
    seen = []

    def connect(url, *, header, **_):
        seen.append((url, header))
        # what comes after generationComplete is not waited for
        return FakeSocket(['{"setupComplete": {}}', transcription("打球"), transcription("吧"), DONE, transcription("多")])

    client = live.Live(entries=[ApiKeyEntry(name="a", key="SECRET")], connect=connect, sleep=lambda _: None)
    assert client(b"\x00" * 3200, language="zh") == "打球吧"
    url, header = seen[0]
    assert "SECRET" not in url and header == ["x-goog-api-key: SECRET"]


def test_live_retries_a_refused_connection_and_never_shows_the_key():
    attempts = []

    def connect(url, **_):
        attempts.append(1)
        raise ConnectionError(f"1011 You exceeded your current quota at {url}?key=SECRET")

    client = live.Live(entries=[ApiKeyEntry(name="a", key="SECRET")], connect=connect, sleep=lambda _: None)
    with pytest.raises(live.LiveError) as error:
        client(b"", language="ja")
    assert len(attempts) == live.ATTEMPTS and "SECRET" not in str(error.value) and "<key>" in str(error.value)


def test_live_setup_names_the_language_only_when_it_has_a_code():
    assert live.setup_message("ja")["setup"]["inputAudioTranscription"]["languageCodes"] == ["ja-JP"]
    assert "languageCodes" not in live.setup_message("ko")["setup"]["inputAudioTranscription"]
    assert live.setup_message(None)["setup"]["realtimeInputConfig"] == {"automaticActivityDetection": {"disabled": True}}


# --- 35t ------------------------------------------------------------------------------------------------------


@pytest.fixture
def source(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "source.wav"
    path.write_bytes(b"RIFF" + bytes(4096))
    monkeypatch.setattr(gemini35t, "extract_upload_mp3", lambda src, a, b, out, **_: Path(out))
    # Silero: speech in every frame (S5b then spreads Live's text over the whole span).
    monkeypatch.setattr(moss_timing, "frame_probabilities", lambda pcm: np.ones(len(pcm) // moss_timing.HOP))
    return path


class FakeVoiceprints:
    """A voice is the PCM's level: every 1000 is another person."""

    error = None

    def embed(self):
        def embed(piece, mask):
            v = np.zeros(8)
            v[int(round(float(np.median(piece)) * 32768 / 1000))] = 1.0
            return v

        return embed


def levels(total, people):
    pcm = np.zeros(int(total * 16000), np.int16)
    for a, b, k in people:
        pcm[int(a * 16000):int(b * 16000)] = 1000 * k
    return pcm


def answer(words):
    return {"status": "completed", "steps": [{"content": [{"type": "text", "annotations": [
        {"type": "word_info", "start_offset": f"{a}s", "end_offset": f"{b}s", "speaker": s, "text": t}
        for a, b, t, s in words]}]}]}


def window_answer(words, name):
    """A 35t window's answer: the `words` (absolute (start, end, text, speaker)) inside it, in window time."""
    _, a, b = name.split("-")
    start, end = float(a), float(b)
    return answer([(x - start, y - start, t, s) for x, y, t, s in words if start <= x < end])


def run_35t(source, tmp_path, words, *, texts=None, voiced=(), pcm=None, repair="on", total=None):
    """35t answering `words` over `total` s (default: just past the last word); Live answers `texts[name]`."""
    asked = []
    total = total if total is not None else max(w[1] for w in words) + 1.0

    def fetch(self, name, call):
        if name.startswith("w-"):
            return window_answer(words, name)
        asked.append(name)
        if (texts or {}).get(name) is live.LiveError:
            raise live.LiveError("35t Live failed 4 times")
        if (texts or {}).get(name) is CacheMiss:
            raise CacheMiss(name)
        if name not in (texts or {}):
            pytest.fail(f"unexpected Live spot {name}")
        return {"text": texts[name]}

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    warned = []
    mp.setattr(gemini35t, "current_reporter", lambda: type("R", (), {
        "warning": lambda self, *a, **k: warned.append(a), "debug": lambda self, *a, **k: None})())
    try:
        t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="zh", caller=object(), duration=total,
                                 repair=repair, live=lambda pcm, language: pytest.fail("Live past the cache"),
                                 speech=mosspro.SpeechSpans(source, list(voiced), pcm if pcm is not None else
                                                            np.zeros(int(total * 16000), np.int16)),
                                 voiceprints=FakeVoiceprints())
    finally:
        mp.undo()
    return t, asked, warned


def talk(a, b, speaker="spk:0", step=0.5):
    return [(a + step * k, a + step * k + 0.4, f"w{a + step * k:.1f}", speaker) for k in range(int((b - a) / step))]


def chars(a, text, speaker="spk:0"):
    """One word per character from `a`, 0.4 s apart."""
    return [(a + 0.4 * k, a + 0.4 * k + 0.3, ch, speaker) for k, ch in enumerate(text)]


def said(words, a, b):
    """What Live writes of the `words` starting in [a, b)."""
    return " ".join(t for x, _, t, _ in words if a <= x < b)


def test_off_leaves_the_transcript_as_it_was(source, tmp_path):
    words = talk(0, 10) + talk(20, 30)
    off, asked, _ = run_35t(source, tmp_path, words, voiced=[(12, 18)], repair="off")
    assert asked == [] and not any("repair" in e for e in off.log)
    assert [(x.start, x.end, x.text, x.speaker) for x in off.units()] == [
        (pytest.approx(a), pytest.approx(b), text, s) for a, b, text, s in words]


def test_a_gap_silero_hears_is_filled_from_live_as_its_own_turn_with_the_nearest_voice(source, tmp_path):
    words = talk(0, 10, "spk:0") + talk(20, 30, "spk:1")
    pcm = levels(31, [(0, 10, 1), (20, 30, 2), (12, 18, 2)])
    # Live hears the gap with 35t's words in the 5 s either side (whole words, 0.3 s more audio)
    wide = said(words, 5, 10) + " 打球吧走吧 " + said(words, 20, 25)
    t, asked, _ = run_35t(source, tmp_path, words, texts={"ctx-4.7-25.2": wide}, voiced=[(12, 18)], pcm=pcm)
    assert asked == ["ctx-4.7-25.2"]
    texts = [x.text for x in t.units()]
    at = texts.index("打")
    assert "".join(texts[at:at + 5]) == "打球吧走吧" and texts[at - 1] == "w9.5"
    filled = [x for x in t.units() if x.text in "打球吧走"]
    assert {x.speaker for x in filled} == {t.units()[-1].speaker}  # the voice of the second speaker
    entry = next(e for e in t.log if e.get("repair") == "miss")
    assert entry["via"] == "live" and entry["heard"] == "wide" and entry["window"] == [4.7, 25.2]
    assert entry["inserted"] == 5


def test_a_wide_answer_with_only_a_fragment_or_a_variant_of_the_neighbours_inserts_nothing(source, tmp_path):
    words = chars(5.0, "今天天气真不错") + chars(20.0, "我们走吧下次再来") + chars(40.0, "这车没见过")
    texts = {"ctx-4.7-23.4": "今天天气真不错。泡了。我们走吧下次再来",  # a fragment
             "ctx-19.7-42.2": "我们走吧下次再来。下次再来吧。这车没见过"}  # a variant of the left side
    t, asked, _ = run_35t(source, tmp_path, words, texts=texts, voiced=[(10, 16), (28, 36)])
    assert sorted(asked) == sorted(texts) and len(t.units()) == len(words)
    assert sorted(e["skipped"] for e in t.log if e.get("repair") == "miss") == ["echo", "short"]


def test_when_the_wide_call_fails_the_gap_alone_is_heard_and_variants_of_its_neighbours_dropped(source, tmp_path):
    words = chars(5.0, "这车没见过") + chars(20.0, "我们走吧")
    texts = {"ctx-4.7-21.8": live.LiveError, "gap-7.2-19.7": "车迷见过"}
    t, asked, _ = run_35t(source, tmp_path, words, texts=texts, voiced=[(10, 16)])
    assert asked == ["ctx-4.7-21.8", "gap-7.2-19.7"] and len(t.units()) == len(words)
    entry = next(e for e in t.log if e.get("repair") == "miss")
    assert entry["heard"] == "alone" and "failed" in entry["wide_failed"] and entry["skipped"] == "echo"
    kept, _, _ = run_35t(source, tmp_path, words, texts=dict(texts, **{"gap-7.2-19.7": "打球吧走吧"}), voiced=[(10, 16)])
    assert "".join(x.text for x in kept.units()) == "这车没见过打球吧走吧我们走吧"
    # a replay reads what the spot alone answered then: the wide window was never stored
    replay, _, _ = run_35t(source, tmp_path, words, texts={"ctx-4.7-21.8": CacheMiss, "gap-7.2-19.7": "打球吧走吧"},
                           voiced=[(10, 16)])
    assert [x.text for x in replay.units()] == [x.text for x in kept.units()]


def test_an_empty_wide_answer_between_two_sides_is_not_heard_and_the_gap_alone_is_asked(source, tmp_path):
    words = chars(5.0, "这车没见过") + chars(20.0, "我们走吧")
    t, asked, _ = run_35t(source, tmp_path, words, texts={"ctx-4.7-21.8": "", "gap-7.2-19.7": "结束了结束了"},
                          voiced=[(10, 16)])
    assert asked == ["ctx-4.7-21.8", "gap-7.2-19.7"] and "结" in [x.text for x in t.units()]
    entry = next(e for e in t.log if e.get("repair") == "miss")
    assert entry["heard"] == "alone" and entry["wide_unused"] == "empty" and entry["window"] == [4.7, 21.8]
    # the gap alone failing gives the spot up with a warning; not cached (a replay of older caches) is no failure
    for alone, warnings in ((live.LiveError, 1), (CacheMiss, 0)):
        t, _, warned = run_35t(source, tmp_path, words, texts={"ctx-4.7-21.8": "", "gap-7.2-19.7": alone},
                               voiced=[(10, 16)])
        entry = next(e for e in t.log if e.get("repair") == "miss")
        assert entry["heard"] == "wide" and entry["skipped"] == "empty" and "alone_failed" in entry
        assert len(warned) == warnings and len(t.units()) == len(words)


def test_an_english_quotation_in_a_chinese_gap_counts_when_the_wide_answer_holds_the_sides(source, tmp_path):
    words = chars(5.0, "我们的会议叫") + chars(20.0, "是一九七一年")
    wide = {"ctx-4.7-22.6": "我们的会议叫Association of Asian Studies，是一九七一年"}
    t, _, _ = run_35t(source, tmp_path, words, texts=wide, voiced=[(10, 16)])
    assert "Association" in [x.text for x in t.units()]
    # heard alone, nothing says the English is the source's: the script check stays
    alone, _, _ = run_35t(source, tmp_path, words, voiced=[(10, 16)],
                          texts={"ctx-4.7-22.6": live.LiveError, "gap-7.6-19.7": "Association of Asian Studies"})
    assert next(e for e in alone.log if e.get("repair") == "miss")["skipped"] == "script"


def test_a_fast_run_is_trimmed_locally_without_asking_live(source, tmp_path):
    # (under 35t's own REPEAT, which would split the window first)
    words = talk(0, 5) + [(6.0 + 0.05 * k, 6.05 + 0.05 * k, "Oh", "spk:0") for k in range(15)] + talk(7.6, 12)
    t, asked, _ = run_35t(source, tmp_path, words)
    assert asked == [] and [x.text for x in t.units()].count("Oh") == 3
    assert next(e for e in t.log if e.get("repair") == "repeat")["via"] == "local"


def test_a_run_at_talking_speed_is_replaced_only_when_live_says_it_far_fewer_times(source, tmp_path):
    run = [(6.0 + 0.3 * k, 6.2 + 0.3 * k, "走", "spk:0") for k in range(14)]
    words = talk(0, 5.6) + run + talk(10.5, 16)
    around = said(words, 1.0, 5.1), said(words, 10.5, 15.1)  # 35t's words in the 5 s either side of the run
    texts = {"ctx-0.7-15.7": f"{around[0]} 哇哦。走走走走走走。吓我一跳。走走走走走。 {around[1]}"}
    t, _, _ = run_35t(source, tmp_path, words, texts=texts, voiced=[(6.0, 10.2)])
    assert "".join(x.text for x in t.units()).count("走") == 11
    entry = next(e for e in t.log if e.get("repair") == "repeat")
    assert entry["longest"] == 6 and entry["replaced"] > 0 and entry["heard"] == "wide"
    # a wide answer without 35t's words either side cannot place the run: the run alone is asked
    alone, asked, _ = run_35t(source, tmp_path, words, voiced=[(6.0, 10.2)], texts={
        "ctx-0.7-15.7": "完全不同的一段话。冲冲冲冲冲冲冲。", "run-6.0-10.1": "哇哦。走走走走走走。吓我一跳。走走走走走。"})
    assert asked == ["ctx-0.7-15.7", "run-6.0-10.1"] and "".join(x.text for x in alone.units()).count("走") == 11
    assert next(e for e in alone.log if e.get("repair") == "repeat")["wide_unused"] == "unheard"
    chant, _, _ = run_35t(source, tmp_path, words, texts={"ctx-0.7-15.7": f"{around[0]} {'走' * 18} {around[1]}"},
                          voiced=[(6.0, 10.2)])
    assert [x.text for x in chant.units()].count("走") == 14  # a real chant: Live writes it long too


@pytest.mark.parametrize("on", [True, False])
def test_a_speakerless_window_takes_the_voices_heard_elsewhere_once_turned_on(source, tmp_path, on):
    # one window, so its "elsewhere" is the speakers of another part: two windows here
    first = talk(0, 540, "spk:0", step=0.5)
    later = talk(470, 999, "spk:0", step=0.5)
    pcm = levels(1000, [(0, 1000, 1)])

    def fetch(self, name, call):
        if name.startswith("w-0.0-"):
            return window_answer(first, name)
        return window_answer([(x, y, t, None) for x, y, t, _ in later], name)

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    mp.setattr(R, "SPEAKERLESS_BY_VOICE", on)
    try:
        t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="zh", caller=object(), duration=1000.0,
                                 repair="on", live=None, speech=mosspro.SpeechSpans(source, [], pcm),
                                 voiceprints=FakeVoiceprints())
    finally:
        mp.undo()
    late = [x for x in t.units() if x.start > 600]
    entry = next(e for e in t.log if e.get("repair") == "speakerless")
    if not on:  # the default: no reader has judged it
        assert late and {x.speaker for x in late} == {None} and entry["skipped"] == "off"
        return
    assert late and {x.speaker for x in late} == {t.units()[0].speaker}
    assert entry["matched"] == entry["blocks"] > 0


def test_a_failed_live_spot_is_given_up_with_a_warning(source, tmp_path, monkeypatch):
    words = talk(0, 10) + talk(20, 30)

    def fetch(self, name, call):
        return window_answer(words, name) if name.startswith("w-") else call()

    def down(pcm, language):
        raise live.LiveError("35t Live failed 4 times")

    monkeypatch.setattr(CallCache, "fetch", fetch)
    warned = []
    monkeypatch.setattr(gemini35t, "current_reporter", lambda: type("R", (), {
        "warning": lambda self, *a, **k: warned.append(a), "debug": lambda self, *a, **k: None})())
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="zh", caller=object(), duration=31.0,
                             repair="on", live=down, speech=mosspro.SpeechSpans(source, [(12, 18)],
                                                                                  np.zeros(31 * 16000, np.int16)),
                             voiceprints=FakeVoiceprints())
    assert [x.text for x in t.units()] == [w[2] for w in words]
    assert "failed" in next(e for e in t.log if e.get("repair") == "miss")
    assert len(warned) == 1 and "1 处" in warned[0][1]


def test_quota_re_asks_35t_up_to_the_cap_and_leaves_the_rest_to_live(source, tmp_path, monkeypatch):
    holes = [(10, 20), (100, 110), (200, 210), (300, 310), (400, 410)]  # far enough apart not to merge
    edges = [0] + [x for h in holes for x in h] + [500]
    words = [w for a, b in zip(edges[::2], edges[1::2]) for w in talk(a, b)]
    voiced = [(a + 2, b - 2) for a, b in holes]
    reasks, lives = [], []

    def fetch(self, name, call):
        if name.startswith("w-"):
            return window_answer(words, name)
        if name.startswith("r-"):
            _, a, b = name.split("-")
            lo, hi = float(a), float(b)
            reasks.append((lo, hi))
            # the re-ask numbers the speaker S0, and hears the hole's speech
            again = [(x, y, t, "S0") for x, y, t, _ in words] + [
                (lo + 10.5 + 0.5 * k, lo + 10.9 + 0.5 * k, f"补{lo:.0f}-{k}", "S0") for k in range(18)]
            return window_answer(again, name.replace("r-", "w-"))
        lives.append(name)
        return {"text": ""}

    monkeypatch.setattr(CallCache, "fetch", fetch)
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="zh", caller=object(), duration=501.0,
                             repair="quota", live=None,
                             speech=mosspro.SpeechSpans(source, voiced, levels(501, [(0, 501, 1)])),
                             voiceprints=FakeVoiceprints())
    # the fifth miss went to Live: nothing over the wide window though 35t has words both sides, so the gap alone
    assert len(reasks) == gemini35t.REASK_CAP and [n.split("-")[0] for n in lives] == ["ctx", "gap"]
    assert next(e for e in t.log if e.get("repair") == "quota_cap")["left_to_live"] == 1
    filled = [x for x in t.units() if x.text.startswith("补")]
    assert len(filled) == 18 * gemini35t.REASK_CAP and {x.speaker for x in filled} == {t.units()[0].speaker}


# --- MAI2 -----------------------------------------------------------------------------------------------------


def run_mai2(source, tmp_path, words, reask, *, voiced, pcm, total):
    from finesub.speech.joint import mai2

    asked = []

    def fetch(self, name, call):
        if name.startswith("w-"):
            return {"words": [{"start": a, "end": b, "word": t, "speaker": s} for a, b, t, s in words]}
        _, a, b = name.split("-")
        asked.append((float(a), float(b)))
        return {"words": [{"start": x - float(a), "end": y - float(a), "word": t, "speaker": s}
                          for x, y, t, s in reask(float(a), float(b))]}

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    mp.setattr(mai2, "extract_upload_mp3", lambda src, a, b, out, **_: Path(out))
    try:
        t = mai2.transcribe(source, cache_root=tmp_path / "cloud", language="en", via="openrouter", repair="on",
                            caller=object(), duration=total, speech=mosspro.SpeechSpans(source, voiced, pcm),
                            voiceprints=FakeVoiceprints())
    finally:
        mp.undo()
    return t, asked


def eng(a, b, speaker):
    return [(a + 0.5 * k, a + 0.5 * k + 0.4, f"s{speaker}w{a}{k}", speaker) for k in range(int((b - a) / 0.5))]


def test_mai2_re_asks_a_gap_and_links_its_speakers_by_the_padding(source, tmp_path):
    words = eng(0, 10, 0) + eng(20, 30, 1)

    def reask(lo, hi):  # numbers its speakers the other way round, and repeats the left neighbour's last word
        return ([(x, y, t, 1 - s) for x, y, t, s in words if lo <= x < hi]
                + [(13.0, 13.3, "s0w019", 0), (13.5, 13.9, "hello", 0), (14.0, 14.4, "there", 0)])

    t, asked = run_mai2(source, tmp_path, words, reask, voiced=[(12, 18)], pcm=np.zeros(31 * 16000, np.int16),
                        total=31.0)
    assert asked == [(0.2, 29.7)]
    got = [(x.text, x.speaker) for x in t.units() if x.text in ("hello", "there")]
    assert got == [("hello", "S1"), ("there", "S1")]  # the re-ask's 0 is the transcript's S1 by the padding
    assert [x.text for x in t.units()].count("s0w019") == 1  # the echo of the left neighbour is dropped
    entry = next(e for e in t.log if e.get("repair") == "miss")
    assert entry["via"] == "mai2" and entry["inserted"] == 2


def test_mai2_failed_re_ask_is_given_up_and_the_rest_goes_on(source, tmp_path):
    from finesub.speech.joint import mai2

    words = eng(0, 10, 0) + eng(20, 100, 1) + eng(110, 130, 0)

    def reask(lo, hi):
        if lo < 50:
            raise mai2.Mai2Error("HTTP 400: bad", status=400)
        return [(x, y, t, s) for x, y, t, s in words if lo <= x < hi] + [(104.0, 104.4, "late", 0)]

    t, asked = run_mai2(source, tmp_path, words, reask, voiced=[(12, 18), (102, 108)],
                        pcm=np.zeros(131 * 16000, np.int16), total=131.0)
    assert len(asked) == 2 and "late" in [x.text for x in t.units()]
    assert "failed" in next(e for e in t.log if e.get("repair") == "miss")


def test_live_session_closed_mid_answer_is_a_failure_not_an_answer(tmp_path):
    def connect(url, **_):
        return FakeSocket(['{"setupComplete": {}}', transcription("半"), ""])  # "" is websocket-client's close

    client = live.Live(entries=[ApiKeyEntry(name="a", key="K")], connect=connect, sleep=lambda _: None)
    with pytest.raises(live.LiveError):
        client(b"", language="zh")
    cache = CallCache(tmp_path, gemini35t.LIVE_CACHE, identity={})
    with pytest.raises(live.LiveError):
        cache.fetch("gap-1.0-4.0", lambda: {"text": client(b"", language="zh")})
    assert not cache.path("gap-1.0-4.0").exists()


def test_live_text_without_generation_complete_is_a_failure_and_silence_is_an_empty_answer(tmp_path):
    replies = {"text": ['{"setupComplete": {}}', transcription("半句")], "silence": ['{"setupComplete": {}}']}

    def client(kind):
        return live.Live(entries=[ApiKeyEntry(name="a", key="K")], connect=lambda url, **_: FakeSocket(replies[kind]),
                         sleep=lambda _: None)

    cache = CallCache(tmp_path, gemini35t.LIVE_CACHE, identity={})
    with pytest.raises(live.LiveError, match="generationComplete"):
        cache.fetch("gap-1.0-4.0", lambda: {"text": client("text")(b"", language="zh")})
    assert not cache.path("gap-1.0-4.0").exists()
    assert client("silence")(b"", language="zh") == ""


def test_live_holds_at_most_its_sessions_across_threads():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    lock, open_now, most, full = threading.Lock(), [0], [0], threading.Event()

    class Slow(FakeSocket):
        def close(self):
            with lock:
                open_now[0] -= 1

    def connect(url, **_):
        with lock:
            open_now[0] += 1
            most[0] = max(most[0], open_now[0])
            if open_now[0] >= live.SESSIONS:
                full.set()
        full.wait(5.0)  # the first sessions stay open until the cap is reached
        time.sleep(0.02)  # and a moment longer: a sixth would get in now if the cap leaked
        return Slow(['{"setupComplete": {}}', transcription("字"), DONE])

    client = live.Live(entries=[ApiKeyEntry(name="a", key="K")], connect=connect, sleep=lambda _: None)
    with ThreadPoolExecutor(live.SESSIONS + 3) as pool:
        assert list(pool.map(lambda _: client(b"", language="zh"), range(live.SESSIONS + 3))) == ["字"] * (live.SESSIONS + 3)
    assert most[0] == live.SESSIONS


def test_one_name_asked_from_two_threads_is_called_once(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    calls, gate = [], threading.Event()

    def call():
        calls.append(1)
        gate.wait(1.0)
        return {"text": f"第{len(calls)}次"}

    cache = CallCache(tmp_path, gemini35t.LIVE_WIDE_CACHE, identity={})
    with ThreadPoolExecutor(2) as pool:
        answers = [pool.submit(cache.fetch, "ctx-0.0-9.0", call) for _ in range(2)]
        gate.set()
        assert [a.result() for a in answers] == [{"text": "第1次"}] * 2
    assert len(calls) == 1


def test_live_error_message_ends_the_attempt():
    def connect(url, **_):
        return FakeSocket(['{"setupComplete": {}}', '{"goAway": {"timeLeft": "0s"}}'])

    client = live.Live(entries=[ApiKeyEntry(name="a", key="K")], connect=connect, sleep=lambda _: None)
    with pytest.raises(live.LiveError, match="goAway"):
        client(b"", language="zh")


def test_a_run_replaced_between_two_gaps_keeps_the_order_and_each_insertion_point(source, tmp_path):
    run = [(16.0 + 0.3 * k, 16.2 + 0.3 * k, "走", "spk:0") for k in range(14)]
    words = talk(0, 10) + talk(12.6, 15.6) + run + talk(20.5, 28) + talk(34, 44)
    texts = {"ctx-4.7-17.7": f"{said(words, 5, 10)} 第一句话 {said(words, 12.6, 17.3)}",
             "ctx-12.3-25.7": f"{said(words, 12.6, 15.6)} 走走走走 {said(words, 20.5, 25.1)}",
             "ctx-22.7-39.2": f"{said(words, 23, 28)} 第二句话 {said(words, 34, 38.7)}"}
    t, _, _ = run_35t(source, tmp_path, words, texts=texts, voiced=[(10.2, 12.3), (16, 20.2), (28.2, 33.7)])
    got = [x.text for x in t.units()]
    assert got.index("第") == got.index("w9.5") + 1
    assert got[got.index("w15.1") + 1:got.index("w15.1") + 6] == ["走", "走", "走", "走", "w20.5"]
    second = got.index("二")
    assert got[second - 1] == "第" and got[second - 2] == "w27.5" and got[second + 3] == "w34.0"


def test_a_transport_failure_in_a_re_ask_gives_up_the_spot_with_a_warning(source, tmp_path, monkeypatch):
    import httpx

    from finesub.speech.joint import mai2

    words = eng(0, 10, 0) + eng(20, 30, 1)

    def reask(lo, hi):
        raise httpx.ConnectError("connection refused")

    warned = []
    monkeypatch.setattr(mai2, "current_reporter", lambda: type("R", (), {
        "warning": lambda self, *a, **k: warned.append(a), "debug": lambda self, *a, **k: None})())
    t, _ = run_mai2(source, tmp_path, words, reask, voiced=[(12, 18)], pcm=np.zeros(31 * 16000, np.int16), total=31.0)
    assert len(t.units()) == len(words) and "ConnectError" in next(e for e in t.log if e.get("repair"))["failed"]
    assert len(warned) == 1


# --- the caches -----------------------------------------------------------------------------------------------


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def quiet():
    return type("R", (), {"warning": lambda self, *a, **k: None, "debug": lambda self, *a, **k: None})()


def asked_nothing(name):
    def call(*_, **__):
        pytest.fail(f"{name} called during an offline replay")
    return call


def repair_then_replay(run, provider, windows, tmp_path):
    """`run(root, offline=...)` three times over a real cache, `windows` being the first pass's answers by name:
    answering every call (fakes), offline from what that stored, and offline from the windows alone. The repairs
    keep out of the provider's own directory; a replay sends nothing and stores nothing."""
    root = tmp_path / "cloud"
    real = CallCache.fetch

    def answered(self, name, call):  # the first pass's windows: the fake's answer, stored for real
        return real(self, name, (lambda: windows[name]) if name.startswith("w-") else call)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CallCache, "fetch", answered)
        first = run(root, offline=False)
    own = root / provider
    assert sorted(p.name for p in own.iterdir()) == ["identity.json", *sorted(f"{n}.raw.json" for n in windows)]
    assert len(list(root.iterdir())) > 1  # the repairs stored somewhere else
    before = files(root)
    assert run(root, offline=True).units() == first.units() and files(root) == before

    windows_only = tmp_path / "windows-only"
    (windows_only / provider).mkdir(parents=True)
    for p in own.iterdir():
        (windows_only / provider / p.name).write_bytes(p.read_bytes())
    before = files(windows_only)
    replay = run(windows_only, offline=True)
    assert files(windows_only) == before
    return replay


@pytest.mark.parametrize("mode", ["on", "quota"])
def test_35t_repairs_keep_out_of_its_cache_and_an_offline_replay_sends_nothing(source, tmp_path, monkeypatch, mode):
    words = talk(0, 10) + talk(20, 30)
    monkeypatch.setattr(gemini35t, "current_reporter", quiet)

    def run(root, *, offline):
        reask = asked_nothing("35t") if offline else (lambda clip, language, seconds: answer([]))
        hear = asked_nothing("Live") if offline else (lambda pcm, language: "补上的话")
        return gemini35t.transcribe(source, cache_root=root, language="zh", caller=reask, duration=31.0, repair=mode,
                                    offline=offline, live=hear,
                                    speech=mosspro.SpeechSpans(source, [(12, 18)], np.zeros(31 * 16000, np.int16)),
                                    voiceprints=FakeVoiceprints())

    replay = repair_then_replay(run, gemini35t.GEMINI_35T, {"w-0.0-31.0": window_answer(words, "w-0.0-31.0")}, tmp_path)
    assert next(e for e in replay.log if e.get("repair") == "miss")["failed"].startswith("not cached")


def test_mai2_repairs_keep_out_of_its_cache_and_an_offline_replay_sends_nothing(source, tmp_path, monkeypatch):
    from finesub.speech.joint import mai2

    words = eng(0, 10, 0) + eng(20, 30, 1)
    monkeypatch.setattr(mai2, "current_reporter", quiet)
    monkeypatch.setattr(mai2, "extract_upload_mp3", lambda src, a, b, out, **_: Path(out))

    def run(root, *, offline):
        reask = asked_nothing("MAI2") if offline else (lambda *a, **k: {"words": [
            {"start": 3.5, "end": 3.9, "word": "hello", "speaker": 0}]})
        return mai2.transcribe(source, cache_root=root, language="en", via="openrouter", repair="on", offline=offline,
                               caller=reask, duration=31.0,
                               speech=mosspro.SpeechSpans(source, [(12, 18)], np.zeros(31 * 16000, np.int16)),
                               voiceprints=FakeVoiceprints())

    windows = {"w-0.0-31.0": {"words": [{"start": a, "end": b, "word": t, "speaker": s} for a, b, t, s in words]}}
    replay = repair_then_replay(run, mai2.MAI2, windows, tmp_path)
    assert "failed" in next(e for e in replay.log if e.get("repair") == "miss")
