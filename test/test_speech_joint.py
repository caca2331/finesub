"""Cloud ASR adapters (`finesub.speech.joint`): caching, keys, 35t / MOSS Pro / MAI2 answers to turns.

No network and no ffmpeg: every provider call is a fake, and the clip cutter is patched out."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from finesub.api_keys import ApiKeyEntry
from finesub.speech import joint
from finesub.speech.joint import gemini35t, mai2, mosspro
from finesub.speech.joint.cache import CacheMiss, CallCache
from finesub.speech.joint.model import UNKNOWN_SPEAKER, Unit, turns_from_units
from finesub.speech.joint.quota import BEIJING, PACIFIC, KeyLedger, QuotaExhausted, pick_key, quota_day
from finesub.speech.joint.stitch import stitch

#: scipy is an `[asr]` dependency; the CI lanes install `[harness,dev]` only. The voice relink's matching imports
#: it inside the function and turns its failure into a warning, so a test without it fails on the speakers, not on
#: the import.
needs_scipy = pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="needs scipy (the [asr] extra)")


@pytest.fixture
def source(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "source.wav"
    path.write_bytes(b"RIFF" + bytes(4096))

    def fake_clip(src, start, end, out, **_):
        Path(out).write_bytes(b"mp3")
        return Path(out)

    for module in (gemini35t, mosspro, mai2):
        monkeypatch.setattr(module, "extract_upload_mp3", fake_clip)
    return path


@pytest.fixture
def ledger_path(tmp_path) -> Path:
    return tmp_path / "state"


def entries(*names: str) -> list[ApiKeyEntry]:
    return [ApiKeyEntry(name=n, key=f"key-{n}") for n in names]


# --- names --------------------------------------------------------------------------------------------------


def test_aliases_resolve_to_full_names():
    assert joint.resolve_backend("gemini") == "gemini-35t"
    assert joint.resolve_backend("MOSS") == "moss-pro"
    assert joint.resolve_backend("mai2") == "mai2"
    with pytest.raises(ValueError):
        joint.resolve_backend("whisper")


def test_turns_group_consecutive_units_and_give_unlabelled_ones_a_key():
    units = [Unit(0, 1, "a", "A"), Unit(1, 2, "b", "A"), Unit(0.5, 0.8, "x", "B"), Unit(2, 3, "c", None)]
    turns = turns_from_units(units)
    assert [(t.speaker, [u.text for u in t.units]) for t in turns] == [
        ("A", ["a", "b"]), ("B", ["x"]), (UNKNOWN_SPEAKER, ["c"])]


# --- cache --------------------------------------------------------------------------------------------------


def test_cache_calls_once_and_replays(tmp_path):
    calls = []
    cache = CallCache(tmp_path, "p", identity={"model": "m"})
    for _ in range(2):
        assert cache.fetch("w-0", lambda: calls.append(1) or {"ok": True}) == {"ok": True}
    assert calls == [1]


def test_offline_cache_refuses_to_call(tmp_path):
    with pytest.raises(CacheMiss):
        CallCache(tmp_path, "p", identity={}, offline=True).fetch("w-0", lambda: {"ok": True})


def test_identity_change_other_than_the_audio_only_warns(tmp_path, monkeypatch):
    CallCache(tmp_path, "p", identity={"model": "m1", "audio": "a"}).fetch("w-0", lambda: {"v": 1})
    warnings = warned(monkeypatch)
    again = CallCache(tmp_path, "p", identity={"model": "m2", "audio": "a"})
    assert again.fetch("w-0", lambda: pytest.fail("called")) == {"v": 1}
    assert warnings == ["cloud-cache-identity"]


def warned(monkeypatch) -> list[str]:
    codes = []

    class Reporter:
        def warning(self, code, message, **_):
            codes.append(code)

    monkeypatch.setattr("finesub.speech.joint.cache.current_reporter", lambda: Reporter())
    return codes


def test_another_audio_discards_the_cached_answers_and_asks_again(tmp_path, monkeypatch):
    old = CallCache(tmp_path, "p", identity={"model": "m", "audio": "a1"})
    old.fetch("w-0", lambda: {"v": 1})
    old.fetch("w-1", lambda: {"v": 1})
    codes = warned(monkeypatch)
    new = CallCache(tmp_path, "p", identity={"model": "m", "audio": "a2"})
    assert new.fetch("w-0", lambda: {"v": 2}) == {"v": 2}
    assert codes == ["cloud-cache-audio"]
    # one recording's answers at a time: w-1 belonged to the old audio and is gone with it
    assert sorted(p.name for p in (tmp_path / "p").iterdir()) == ["identity.json", "w-0.raw.json"]
    assert json.loads((tmp_path / "p" / "identity.json").read_text(encoding="utf-8"))["audio"] == "a2"
    assert CallCache(tmp_path, "p", identity={"model": "m", "audio": "a2"}).fetch("w-0", lambda: pytest.fail("called")) == {"v": 2}


def test_offline_over_another_audio_serves_nothing_and_deletes_nothing(tmp_path):
    CallCache(tmp_path, "p", identity={"audio": "a1"}).fetch("w-0", lambda: {"v": 1})
    with pytest.raises(CacheMiss, match="another recording"):
        CallCache(tmp_path, "p", identity={"audio": "a2"}, offline=True).fetch("w-0", lambda: {"v": 2})
    assert (tmp_path / "p" / "w-0.raw.json").exists()


def test_concurrent_fetches_over_another_audio_discard_once(tmp_path, monkeypatch):
    import threading

    CallCache(tmp_path, "p", identity={"audio": "a1"}).fetch("w-0", lambda: {"v": 1})
    codes = warned(monkeypatch)
    new = CallCache(tmp_path, "p", identity={"audio": "a2"})
    results = []
    threads = [threading.Thread(target=lambda n=n: results.append(new.fetch(f"w-{n}", lambda: {"v": 2})))
               for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [{"v": 2}] * 4 and codes == ["cloud-cache-audio"]


@pytest.mark.parametrize("identity", [
    lambda src: gemini35t.identity(src, "ja"),
    lambda src: mai2.identity(src, "openrouter", "en-US"),
    lambda src: mosspro.identity(src, "ja"),
])
def test_every_cloud_identity_keys_the_audio(source, identity):
    assert identity(source)["audio"] == joint.cache.audio_fingerprint(source)


def test_mai2_rerun_over_another_recording_asks_again(source, tmp_path):
    def run(word):
        answer = {"words": [{"start": 1.0, "end": 1.2, "word": word, "speaker": 0}]}
        t = mai2.transcribe(source, cache_root=tmp_path, language="en", via="openrouter", repair="off",
                            caller=lambda clip, locale: answer, duration=600.0)
        return [u.text for u in t.units()]

    assert run("old") == ["old"]
    assert run("unused") == ["old"]  # same audio: the cached answer, no call
    source.write_bytes(b"RIFF" + bytes([1]) * 4096)
    assert run("new") == ["new"]


# --- keys ---------------------------------------------------------------------------------------------------


def test_one_key_until_spent_then_the_next(ledger_path):
    ledger = KeyLedger("p", "m", PACIFIC, state_path=ledger_path)
    keys = entries("a", "b")
    assert pick_key(keys, ledger, what="x").name == "a"
    ledger.record(keys[0], 10)
    assert pick_key(keys, ledger, what="x").name == "a"  # never alternating
    ledger.mark_exhausted(keys[0])
    assert pick_key(keys, ledger, what="x").name == "b"
    ledger.mark_exhausted(keys[1])
    with pytest.raises(QuotaExhausted):
        pick_key(keys, ledger, what="x")


def test_spent_is_per_model_and_resets_with_the_day(ledger_path):
    now = [datetime(2026, 10, 30, 12, tzinfo=timezone.utc)]
    one = KeyLedger("p", "m1", BEIJING, state_path=ledger_path, clock=lambda: now[0])
    other = KeyLedger("p", "m2", BEIJING, state_path=ledger_path, clock=lambda: now[0])
    key = entries("a")[0]
    one.mark_exhausted(key)
    assert one.usage(key)["exhausted"] and not other.usage(key)["exhausted"]
    now[0] = datetime(2026, 10, 30, 16, 1, tzinfo=timezone.utc)  # 00:01 in Beijing
    assert not one.usage(key)["exhausted"]


def test_quota_days_turn_over_at_their_zones_midnight():
    moment = datetime(2026, 10, 30, 15, 59, tzinfo=timezone.utc)
    assert quota_day(BEIJING, moment) == "2026-10-30"
    assert quota_day(BEIJING, datetime(2026, 10, 30, 16, 0, tzinfo=timezone.utc)) == "2026-10-31"
    assert quota_day(PACIFIC, datetime(2026, 10, 30, 6, 0, tzinfo=timezone.utc)) == "2026-10-29"


def test_35t_caller_moves_to_the_next_key_on_the_daily_wall(ledger_path, tmp_path):
    caller = gemini35t.Caller(ledger=KeyLedger("g", "m", PACIFIC, state_path=ledger_path), entries=entries("a", "b"))
    used = []

    def once(audio, key, *, language):
        used.append(key)
        if key == "key-a":
            raise QuotaExhausted("requests per day")
        return {"status": "completed"}

    caller._once = once
    assert caller(tmp_path / "x.mp3", language=None, seconds=540) == {"status": "completed"}
    assert caller(tmp_path / "x.mp3", language=None, seconds=540) == {"status": "completed"}
    assert used == ["key-a", "key-b", "key-b"]


def test_moss_key_is_skipped_when_the_call_does_not_fit_its_day(ledger_path, tmp_path):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    ledger.record(keys[0], 3000)  # 0.88 h of 1.2, counted x1.06
    caller = mosspro.Caller(ledger=ledger, entries=keys)
    used = []
    caller._once = lambda audio, key: used.append(key) or {"segments": []}
    caller(tmp_path / "x.mp3", seconds=1200)
    assert used == ["key-b"]


@pytest.mark.parametrize("paid, expected", [(True, ["key-b"]), (False, None)])
def test_moss_last_key_goes_past_its_free_hours_only_when_paid(ledger_path, tmp_path, paid, expected):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    for key in keys:
        ledger.record(key, 4100)  # 1.21 h counted (x1.06): past the 1.2 h stop on both
    caller = mosspro.Caller(ledger=ledger, entries=keys, paid=paid)
    used = []
    caller._once = lambda audio, key: used.append(key) or {"segments": []}
    if expected is None:
        with pytest.raises(QuotaExhausted):
            caller(tmp_path / "x.mp3", seconds=60)
    else:
        caller(tmp_path / "x.mp3", seconds=60)
        assert used == expected  # the first key keeps its free-hours stop


def test_moss_paid_key_waits_for_the_free_keys_before_it(ledger_path, tmp_path):
    caller = mosspro.Caller(ledger=KeyLedger("m", "m", BEIJING, state_path=ledger_path), entries=entries("a", "b"), paid=True)
    used = []
    caller._once = lambda audio, key: used.append(key) or {"segments": []}
    caller(tmp_path / "x.mp3", seconds=60)
    assert used == ["key-a"]


def test_moss_single_paid_key_has_no_daily_stop(ledger_path, tmp_path):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    key = entries("a")
    ledger.record(key[0], 4100)
    caller = mosspro.Caller(ledger=ledger, entries=key, paid=True)
    caller._once = lambda audio, k: {"segments": []}
    caller(tmp_path / "x.mp3", seconds=3600)
    assert ledger.usage(key[0])["seconds"] == 7700


def test_moss_paid_key_refused_stops_without_spending_its_day(ledger_path, tmp_path):
    """A 402 on the paid key is an empty balance: stop, give the call back, and leave the key usable after a top-up."""
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    ledger.record(keys[0], 4100)
    caller = mosspro.Caller(ledger=ledger, entries=keys, paid=True)
    caller._once = lambda audio, key: None  # HTTP 402
    with pytest.raises(QuotaExhausted, match="balance"):
        caller(tmp_path / "x.mp3", seconds=60)
    assert ledger.usage(keys[1]) == {"calls": 0, "seconds": 0.0, "exhausted": False}


def test_moss_refund_after_the_day_turned_over_leaves_the_new_day_alone(ledger_path, tmp_path):
    """Reserved before Beijing midnight, refused after it: the new day's account and spent mark stay as they
    are, and the retry goes out on the same key's new day."""
    now = [datetime(2026, 10, 30, 15, 59, tzinfo=timezone.utc)]
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path, clock=lambda: now[0])
    keys = entries("a", "b")
    caller = mosspro.Caller(ledger=ledger, entries=keys, paid=False)
    used = []

    def once(audio, key):
        used.append(key)
        if len(used) > 1:
            return {"segments": []}
        now[0] = datetime(2026, 10, 30, 16, 1, tzinfo=timezone.utc)  # 00:01 in Beijing
        ledger.record(keys[0], 300)  # another batch item, on the new day
        return None  # HTTP 402 for the call reserved yesterday

    caller._once = once
    caller(tmp_path / "x.mp3", seconds=600)
    assert used == ["key-a", "key-a"]
    assert ledger.usage(keys[0]) == {"calls": 2, "seconds": 900.0, "exhausted": False}


def test_moss_paid_reads_the_config_file(ledger_path, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text("[asr]\nmoss_pro_paid = false\n", encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config))
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    key = entries("a")
    ledger.record(key[0], 4100)
    caller = mosspro.Caller(ledger=ledger, entries=key)
    assert caller.paid is False
    with pytest.raises(QuotaExhausted):
        caller(tmp_path / "x.mp3", seconds=60)


def test_moss_refusal_names_the_console(ledger_path, tmp_path):
    caller = mosspro.Caller(ledger=KeyLedger("m", "m", BEIJING, state_path=ledger_path), entries=entries("a"), paid=False)
    caller._once = lambda audio, key: None  # HTTP 402
    with pytest.raises(QuotaExhausted, match="console"):
        caller(tmp_path / "x.mp3", seconds=60)


def test_moss_call_is_on_the_account_while_it_runs(ledger_path, tmp_path):
    """Two batch items at once: the second sees the first one's hour already spent."""
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    used = []

    def first(audio, key):
        inner = mosspro.Caller(ledger=ledger, entries=keys)
        inner._once = lambda audio, key: used.append(key) or {"segments": []}
        inner(audio, seconds=2400)
        used.append(key)
        return {"segments": []}

    outer = mosspro.Caller(ledger=ledger, entries=keys)
    outer._once = first
    outer(tmp_path / "x.mp3", seconds=2400)  # 0.71 h each, 1.2 h a key
    assert used == ["key-b", "key-a"]
    assert ledger.usage(keys[0])["seconds"] == ledger.usage(keys[1])["seconds"] == 2400


@pytest.mark.parametrize("error, refunded", [
    (mosspro.MossProRefused("HTTP 400"), True),  # a 4xx: no task
    (mosspro.MossProError("HTTP 502"), False),  # a gateway error may hide a created task
    (httpx.ConnectError("no route"), True),  # never reached the server
    (httpx.ReadTimeout("answer lost"), False),  # uploaded: the server may have billed it
    (mosspro.MossProError("task FAILED"), False),  # after the task was accepted
])
def test_moss_call_is_refunded_only_when_the_server_certainly_did_not_take_it(ledger_path, tmp_path, error, refunded):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    key = entries("a")
    caller = mosspro.Caller(ledger=ledger, entries=key)

    def once(audio, k):
        raise error

    caller._once = once
    with pytest.raises(type(error)):
        caller(tmp_path / "x.mp3", seconds=60)
    assert ledger.usage(key[0])["seconds"] == (0.0 if refunded else 60.0)


def test_moss_poll_that_cannot_connect_keeps_the_accepted_upload_counted(ledger_path, tmp_path, monkeypatch):
    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return httpx.Response(200, json={"task_id": "t1"})

        def get(self, *args, **kwargs):
            raise httpx.ConnectError("no route")

    monkeypatch.setattr(mosspro, "_client", Client)
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    key = entries("a")
    caller = mosspro.Caller(ledger=ledger, entries=key)
    caller._sleep = lambda seconds: None
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    with pytest.raises(mosspro.MossProError, match="poll failed"):
        caller(audio, seconds=60)
    assert ledger.usage(key[0])["seconds"] == 60.0


def test_a_refused_key_is_refunded_and_spent_in_one_write(ledger_path, tmp_path):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    caller = mosspro.Caller(ledger=ledger, entries=keys)
    caller._once = lambda audio, key: None if key == "key-a" else {"segments": []}
    caller(tmp_path / "x.mp3", seconds=60)
    assert ledger.usage(keys[0]) == {"calls": 0, "seconds": 0.0, "exhausted": True}
    assert ledger.usage(keys[1])["seconds"] == 60.0


# --- stitching ----------------------------------------------------------------------------------------------


def w(start, end, text, speaker="A"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_stitch_keeps_each_word_once_in_output_order_and_links_speakers():
    # B's backchannel comes after A's later word, as 35t emits it; the order survives the seam
    left = [w(0, 1, "a1"), w(5, 6, "a2"), w(2, 3, "b1", "B"), w(40, 41, "a3")]
    right = [w(40, 41, "a3", "X"), w(55, 56, "a4", "X")]
    out = stitch([(0, 50, left), (40, 60, right)], segments=False)
    assert [u["text"] for u in out] == ["a1", "a2", "b1", "a3", "a4"]
    assert [u["speaker"] for u in out] == ["w0:A", "w0:A", "w0:B", "w0:A", "w0:A"]


def test_segment_seams_keep_each_utterance_once_sorted_by_start():
    left = [w(0, 5, "最初の話"), w(3590, 3598, "つなぎ目の文です"), w(3610, 3620, "左だけの文")]
    right = [w(3590.1, 3598.2, "つなぎ目の文です"), w(3610, 3620, "左だけの文"), w(3700, 3705, "後半")]
    out = stitch([(0, 3630, left), (3570, 7200, right)], segments=True)
    texts = [u["text"] for u in out]
    assert texts == ["最初の話", "つなぎ目の文です", "左だけの文", "後半"]
    assert [u["start"] for u in out] == sorted(u["start"] for u in out)


# --- 35t ----------------------------------------------------------------------------------------------------


def answer(words, status="completed"):
    return {"status": status, "steps": [{"content": [{"type": "text", "annotations": [
        {"type": "word_info", "start_offset": f"{a}s", "end_offset": f"{b}s", "speaker": s, "text": t}
        for a, b, t, s in words]}]}]}


def test_35t_repair_keeps_output_order():
    words = [w(0, 0.3, "あ"), w(0.3, 0.6, "い"), w(0.1, 0.3, "うん", "B"), w(0.6, 0.9, "う")]
    fixed, _ = gemini35t.fix_times(words)
    assert [x["text"] for x in fixed] == ["あ", "い", "うん", "う"]


def test_35t_bad_run_is_shifted_back():
    # "b" ends 100 s late and the words after it follow, until the matching jump back
    words = [w(10.0, 10.3, "a"), w(10.3, 110.6, "b"), w(110.6, 110.9, "c"), w(11.0, 11.3, "d")]
    fixed, events = gemini35t.fix_times(words)
    assert events[0]["fix"] == "shift"
    assert [(round(x["start"], 1), round(x["end"], 1)) for x in fixed] == [(10.0, 10.3), (10.3, 10.6), (10.6, 10.9), (11.0, 11.3)]


def transcribe_35t(source, tmp_path, words_for, total):
    """35t over `total` s where each window's answer is `words_for(start, end)` (absolute times)."""
    calls = []

    def fetch(self, name, call):
        _, a, b = name.split("-")
        start, end = float(a), float(b)
        calls.append((start, end))
        words, status = words_for(start, end)
        return answer([(x - start, y - start, text, s) for x, y, text, s in words], status)

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    try:
        t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", caller=object(), duration=total,
                                 repair="off")
    finally:
        mp.undo()
    return t, calls


def speech_every_second(start, end, upto=None):
    stop = min(end, upto) if upto is not None else end
    return [(x, x + 0.5, f"w{int(x)}", "spk:0") for x in range(int(start) + 1, int(stop) - 1)], "completed"


def test_35t_windows_overlap_and_each_word_is_kept_once(source, tmp_path):
    t, calls = transcribe_35t(source, tmp_path, speech_every_second, 1200.0)
    assert calls == [(0.0, 540.0), (480.0, 1020.0), (960.0, 1200.0)]
    texts = [u.text for u in t.units()]
    assert len(texts) == len(set(texts))
    assert {f"w{x}" for x in (1, 500, 1000, 1197)} <= set(texts)


def test_35t_early_stop_continues_at_least_min_advance_on(source, tmp_path):
    def stops_early(start, end):
        return speech_every_second(start, end, upto=start + 60 if start == 0 else None)

    t, calls = transcribe_35t(source, tmp_path, stops_early, 1200.0)
    assert calls[1][0] == 120.0  # last word ~59 s, minus CONTINUE_BACK, but at least MIN_ADVANCE
    assert any("early_stop" in e for e in t.log)


def test_35t_tail_is_continued_once(source, tmp_path):
    def tail_missing(start, end):
        return speech_every_second(start, end, upto=start + 100)

    _, calls = transcribe_35t(source, tmp_path, tail_missing, 500.0)
    assert len(calls) == 2  # the window, then one continuation from its last word; never a third


def test_35t_degenerate_window_is_split_into_overlapping_halves(source, tmp_path):
    def degenerate(start, end):
        if (start, end) == (0.0, 540.0):
            return [(x * 0.2, x * 0.2 + 0.1, "あ", "spk:0") for x in range(1, 40)], "completed"
        return speech_every_second(start, end)

    t, calls = transcribe_35t(source, tmp_path, degenerate, 540.0)
    assert calls == [(0.0, 540.0), (0.0, 300.0), (240.0, 540.0)]
    texts = [u.text for u in t.units()]
    assert len(texts) == len(set(texts)) and "w270" in texts


# --- MOSS Pro -----------------------------------------------------------------------------------------------


def moss_answer(segments, tokens=500):
    return {"status": "SUCCESS", "usage": {"completion_tokens": tokens},
            "result": {"segments": [{"start": a, "end": b, "text": t, "speaker": s} for a, b, t, s in segments]}}


class FakeVoiceprints:
    """A voice is the PCM's level: every 1000 is another person (0 is silence)."""

    def __init__(self, available=True):
        self.available = available
        self.error = None if available else "no model in tests"

    def embed(self):
        if not self.available:
            return None

        def embed(piece, mask):
            v = np.zeros(8)
            v[int(round(float(np.median(piece)) * 32768 / 1000))] = 1.0
            return v

        return embed


def voices(total, people):
    """int16 PCM at 16 kHz: each (start, end, person) span at that person's level."""
    pcm = np.zeros(int(total * 16000), np.int16)
    for a, b, k in people:
        pcm[int(a * 16000):int(b * 16000)] = 1000 * k
    return pcm


def transcribe_moss(source, tmp_path, answers, total, *, voiced=(), language="ja", pcm=None, voiceprints=None):
    calls = []

    def fetch(self, name, call):
        calls.append(name)
        if name not in answers:
            raise AssertionError(f"unexpected call {name}")
        return answers[name]

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    try:
        t = mosspro.transcribe(source, cache_root=tmp_path / "cloud", language=language, caller=object(),
                               duration=total, speech=mosspro.SpeechSpans(source, list(voiced), pcm),
                               voiceprints=voiceprints or FakeVoiceprints(available=False))
    finally:
        mp.undo()
    return t, calls


def test_moss_continues_a_capped_answer_from_its_last_good_segment(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 5, "前半", "S01"), (300, 0, "壊 れ た", "S02")], tokens=16384),
               "s-5.0-600.0": moss_answer([(1, 4, "続き", "S01"), (500, 590, "最後", "S02")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0)
    assert calls == ["s-0.0-600.0", "s-5.0-600.0"]
    assert [u.text for u in t.units()] == ["前半", "続き", "最後"]


def test_moss_continuation_that_makes_no_progress_stops(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 5, "前半", "S01"), (300, 0, "壊 れ た", "S02")], tokens=16384),
               "s-5.0-600.0": moss_answer([(300, 0, "壊 れ た", "S02")], tokens=16384)}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0)
    assert calls == ["s-0.0-600.0", "s-5.0-600.0"]
    assert [u.text for u in t.units()] == ["前半"]
    assert {"continue_stalled": [5.0, 600.0]} in t.log


def test_moss_gap_after_a_stalled_continuation_is_asked_again_for_real(source, tmp_path):
    stalled = moss_answer([(300, 0, "壊 れ た", "S02")], tokens=16384)
    answers = {"s-0.0-600.0": moss_answer([(0, 5, "前半", "S01"), (300, 0, "壊 れ た", "S02")], tokens=16384),
               "s-5.0-600.0": stalled, "s-5.0-600.0-again": moss_answer([(10, 20, "後半", "S01")])}
    for _ in range(2):  # a rerun asks for the same names: the cache answers them
        t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(10, 20)])
        assert calls == ["s-0.0-600.0", "s-5.0-600.0", "s-5.0-600.0-again"]
        assert [u.text for u in t.units()] == ["前半", "後半"]


def test_moss_gap_is_rerequested_only_when_silero_hears_speech(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S01")]),
               "s-100.0-200.0": moss_answer([(10, 20, "拾った", "S02")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(130, 132)])
    assert calls == ["s-0.0-600.0"]  # 2 s of speech: not worth a call
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 120)])
    assert calls == ["s-0.0-600.0", "s-100.0-200.0"]
    assert [u.text for u in t.units()] == ["あ", "拾った", "い"]


@needs_scipy
def test_moss_continuation_speakers_are_relinked_by_voice(source, tmp_path):
    # the main call heard persons 1 (S01) and 2 (S02); the continuation numbers anew: its S01 is person 2,
    # its S02 someone not heard before
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (100, 200, "い", "S02"),
                                           (300, 0, "壊", "S01")], tokens=16384),
               "s-200.0-600.0": moss_answer([(10, 100, "う", "S01"), (100, 390, "え", "S02")])}
    pcm = voices(600, [(0, 100, 1), (100, 200, 2), (210, 300, 2), (300, 590, 3)])
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, pcm=pcm, voiceprints=FakeVoiceprints())
    assert [(u.text, u.speaker) for u in t.units()] == [("あ", "S01"), ("い", "S02"), ("う", "S02"), ("え", "c1:S02")]
    assert t.log[0]["speakers"] == {"S01": "S02", "S02": "c1:S02"}


@needs_scipy
def test_moss_gap_speakers_are_relinked_and_too_little_speech_is_a_new_person(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S02")]),
               "s-100.0-200.0": moss_answer([(10, 50, "う", "S01"), (60, 62, "え", "S02")])}
    pcm = voices(600, [(0, 100, 1), (110, 150, 2), (160, 162, 1), (200, 590, 2)])
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 150)], pcm=pcm,
                           voiceprints=FakeVoiceprints())
    # 2 s of S02 in the gap: not enough to match, though it is person 1's voice
    assert t.log[0]["speakers"] == {"S01": "S02", "S02": "c1:S02"}


def test_moss_relink_without_the_voiceprint_model_makes_everyone_new(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S01")]),
               "s-100.0-200.0": moss_answer([(10, 50, "う", "S01")])}
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 150)])
    assert [u.speaker for u in t.units()] == ["S01", "c1:S01", "S01"]


def test_voiceprint_model_failure_is_said_once_per_transcript(monkeypatch):
    from finesub.speech.joint import relink

    loads, warned = [], []

    def broken():
        loads.append(1)
        raise RuntimeError("offline")

    monkeypatch.setattr(relink, "_wespeaker", broken)
    monkeypatch.setattr(relink, "current_reporter", lambda: type("R", (), {"warning": lambda self, *a, **k: warned.append(a)})())
    prints = relink.Voiceprints()
    for _ in range(2):  # two transcripts, two re-requests each
        link = relink.Relinker(lambda: None, prints)
        link([], [{"start": 0.0, "end": 5.0, "speaker": "S01"}])
        link([], [{"start": 6.0, "end": 9.0, "speaker": "S01"}])
    assert len(loads) == 1 and len(warned) == 2 and "offline" in warned[0][1]


def test_moss_wrong_script_is_asked_once_more_then_stops(source, tmp_path):
    japanese = moss_answer([(0, 590, "これは日本語の訳文です", "S01")])
    english = moss_answer([(0, 590, "this is the English transcript", "S01")])
    t, calls = transcribe_moss(source, tmp_path, {"s-0.0-600.0": japanese, "s-0.0-600.0-retry": english}, 600.0,
                               language="en")
    assert calls == ["s-0.0-600.0", "s-0.0-600.0-retry"] and t.units()[0].text.startswith("this")
    with pytest.raises(mosspro.MossProError):
        transcribe_moss(source, tmp_path, {"s-0.0-600.0": japanese, "s-0.0-600.0-retry": japanese}, 600.0,
                        language="en")


def test_moss_segments_without_a_speaker_get_a_key(source, tmp_path):
    t, _ = transcribe_moss(source, tmp_path, {"s-0.0-60.0": moss_answer([(0, 50, "誰か", None)])}, 60.0)
    assert t.turns[0].speaker == UNKNOWN_SPEAKER


def test_moss_drops_cap_debris_and_music_dots():
    segs = mosspro.segments_of(moss_answer([(0, 2, "はい", "S01"), (3, 0, "壊", "S01"), (5, 5.5, "...", "S02")]), 10.0)
    assert [(s["start"], s["text"]) for s in segs] == [(10.0, "はい")]


# --- MAI2 ---------------------------------------------------------------------------------------------------


def test_mai2_azure_words_take_their_phrase_speaker():
    answer = {"phrases": [{"speaker": 1, "offsetMilliseconds": 0, "durationMilliseconds": 0,
                           "words": [{"text": "こん", "offsetMilliseconds": 1000, "durationMilliseconds": 79}]}]}
    assert mai2.azure_words(answer, 100.0) == [{"start": 101.0, "end": 101.079, "speaker": "S1", "text": "こん"}]


def test_mai2_openrouter_words():
    answer = {"words": [{"start": 1.0, "end": 1.2, "word": "hi", "speaker": 0}]}
    assert mai2.openrouter_words(answer, 0.0) == [{"start": 1.0, "end": 1.2, "speaker": "S0", "text": "hi"}]


def test_mai2_an_answer_cached_on_one_route_replays_on_the_other(source, tmp_path):
    azure = {"phrases": [{"speaker": 1, "words": [{"text": "hi", "offsetMilliseconds": 1000, "durationMilliseconds": 200}]}]}
    openrouter = {"words": [{"start": 1.0, "end": 1.2, "word": "hi", "speaker": 1}]}
    for first, answer, second in (("azure", azure, "openrouter"), ("openrouter", openrouter, "azure")):
        root = tmp_path / first
        asked = mai2.transcribe(source, cache_root=root, language="en", via=first, repair="off", caller=lambda clip, locale: answer,
                                duration=600.0)
        replayed = mai2.transcribe(source, cache_root=root, language="en", via=second, repair="off", offline=True, duration=600.0)
        assert [(u.text, u.speaker) for u in replayed.units()] == [(u.text, u.speaker) for u in asked.units()] == [("hi", "S1")]


def test_mai2_long_source_is_windowed_and_stitched(source, tmp_path):
    def fetch(self, name, call):
        _, a, b = name.split("-")
        a, b = float(a), float(b)
        return {"words": [{"start": x - a, "end": x - a + 0.3, "word": f"w{int(x)}", "speaker": 0}
                          for x in range(int(a) + 1, int(b) - 1, 10)]}

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    try:
        t = mai2.transcribe(source, cache_root=tmp_path, language="en", via="openrouter", repair="off", caller=object(),
                            duration=3000.0)
    finally:
        mp.undo()
    texts = [u.text for u in t.units()]
    assert len(texts) == len(set(texts)) and texts == sorted(texts, key=lambda s: int(s[1:]))
    assert t.log == [{"windows": [[0.0, 1800.0], [1740.0, 3000.0]]}]


def test_mai2_window_failing_is_not_halved_or_cached(source, tmp_path):
    # Audit D-3: a 504 used to be cached as "too large" and the window halved for good. Every window fits the
    # service's documented limit, so a failure after the retries ends the stage and the next run asks again.
    asked = []

    def caller(clip, *, locale):
        raise mai2.Mai2Error("HTTP 504", status=504)

    def clip(src, start, end, out, **_):
        asked.append((start, end))
        Path(out).write_bytes(b"mp3")
        return Path(out)

    mp = pytest.MonkeyPatch()
    mp.setattr(mai2, "extract_upload_mp3", clip)
    try:
        for _ in range(2):
            with pytest.raises(mai2.Mai2Error):
                mai2.transcribe(source, cache_root=tmp_path, language="en", via="openrouter", repair="off",
                                caller=caller, duration=1800.0)
    finally:
        mp.undo()
    assert asked == [(0.0, 1800.0), (0.0, 1800.0)]


# --- transient errors ---------------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status, body=None, text="", headers=None):
        self.status_code, self._body, self.text, self.headers = status, body or {}, text, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        assert self.status_code < 400


class FakeClient:
    """Answers the Files upload, then hands out `answers` for the generation calls in order."""

    def __init__(self, answers):
        self.answers = list(answers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **_):
        if url == gemini35t.UPLOAD:
            return FakeResponse(200, headers={"X-Goog-Upload-URL": "session"})
        if url == "session":
            return FakeResponse(200, {"file": {"uri": "files/1"}})
        return self.answers.pop(0)


PER_MINUTE = ('{"error": {"code": 429, "message": "You exceeded your current quota. Please retry in 7s.", '
              '"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}')
PER_DAY = ('{"error": {"code": 429, "message": "You exceeded your current quota.", '
           '"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}')


def test_35t_per_minute_429_waits_and_keeps_the_key(ledger_path, tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    client = FakeClient([FakeResponse(429, text=PER_MINUTE), FakeResponse(200, {"status": "completed"})])
    monkeypatch.setattr(gemini35t, "_client", lambda: client)
    waits = []
    ledger = KeyLedger("g", "m", PACIFIC, state_path=ledger_path)
    caller = gemini35t.Caller(ledger=ledger, entries=entries("a"), sleep=waits.append)
    assert caller(audio, language=None, seconds=540) == {"status": "completed"}
    assert waits == [12.0] and not ledger.usage(entries("a")[0])["exhausted"]


def test_35t_per_day_429_spends_the_key(ledger_path, tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    monkeypatch.setattr(gemini35t, "_client", lambda: FakeClient([FakeResponse(429, text=PER_DAY)]))
    ledger = KeyLedger("g", "m", PACIFIC, state_path=ledger_path)
    with pytest.raises(QuotaExhausted):
        gemini35t.Caller(ledger=ledger, entries=entries("a"), sleep=lambda s: None)(audio, language=None, seconds=540)
    assert ledger.usage(entries("a")[0])["exhausted"]


def test_mai2_retries_a_transient_error(tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    answers = [FakeResponse(503, text="busy"), FakeResponse(429, text="slow down", headers={"Retry-After": "5"}),
               FakeResponse(200, {"words": []})]
    waits = []
    caller = mai2.Caller("openrouter", "k", sleep=waits.append)
    object.__setattr__(caller, "_once", lambda audio, locale: answers.pop(0))
    assert caller(audio, locale=None) == {"words": []}
    assert waits == [30.0, 5.0]


DAILY_AS_200 = {"error": {"message": "You exceeded your current quota, please check your plan and billing details."}}


def test_35t_daily_wall_as_200_error_moves_to_the_next_key(ledger_path, tmp_path, monkeypatch):
    # docs/gemini35-transcribe.md: the daily wall can come back as HTTP 200 with only this sentence
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    answers = {"key-a": [FakeResponse(200, DAILY_AS_200), FakeResponse(200, DAILY_AS_200)],
               "key-b": [FakeResponse(200, {"status": "completed"})]}

    class ByKey(FakeClient):
        def post(self, url, headers=None, **kw):
            if url in (gemini35t.UPLOAD, "session"):
                return super().post(url)
            return answers[headers["x-goog-api-key"]].pop(0)

    monkeypatch.setattr(gemini35t, "_client", lambda: ByKey([]))
    caller = gemini35t.Caller(ledger=KeyLedger("g", "m", PACIFIC, state_path=ledger_path), entries=entries("a", "b"),
                              sleep=lambda s: None)
    assert caller(audio, language=None, seconds=540) == {"status": "completed"}
    assert caller.ledger.usage(entries("a")[0])["exhausted"]


def test_35t_an_error_answer_is_never_returned_to_be_cached(ledger_path, tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    client = FakeClient([FakeResponse(200, {"error": {"message": "internal"}}), FakeResponse(200, {"status": "completed"})])
    monkeypatch.setattr(gemini35t, "_client", lambda: client)
    caller = gemini35t.Caller(ledger=KeyLedger("g", "m", PACIFIC, state_path=ledger_path), entries=entries("a"),
                              sleep=lambda s: None)
    assert caller(audio, language=None, seconds=540) == {"status": "completed"}


def test_mai2_retries_a_504(tmp_path):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    answers = [FakeResponse(504, text="timeout"), FakeResponse(200, {"words": []})]
    waits = []
    caller = mai2.Caller("openrouter", "k", sleep=waits.append)
    object.__setattr__(caller, "_once", lambda audio, locale: answers.pop(0))
    assert caller(audio, locale=None) == {"words": []}
    assert waits == [mai2.RETRY_WAITS[0]]


def test_mai2_does_not_retry_a_413_or_wait_an_hour(tmp_path):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    for answers in ([FakeResponse(413, text="too large")], [FakeResponse(429, text="later", headers={"Retry-After": "3600"})]):
        caller = mai2.Caller("openrouter", "k", sleep=lambda s: pytest.fail("waited"))
        object.__setattr__(caller, "_once", lambda audio, locale, answers=answers: answers.pop(0))
        with pytest.raises(mai2.Mai2Error):
            caller(audio, locale=None)


def test_mai2_retries_a_dropped_connection(tmp_path):
    # Audit S-4: a reset or a proxy drop has no status; it used to end the whole cloud stage on one window.
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    answers = [httpx.ReadError("[Errno 104] Connection reset by peer"),
               httpx.RemoteProtocolError("Server disconnected without sending a response"),
               FakeResponse(200, {"words": []})]
    waits = []
    caller = mai2.Caller("openrouter", "k", sleep=waits.append)

    def once(audio, locale):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    object.__setattr__(caller, "_once", once)
    assert caller(audio, locale=None) == {"words": []}
    assert waits == list(mai2.RETRY_WAITS[:2])


def test_35t_retries_a_5xx_a_dropped_connection_and_a_body_that_is_not_json(ledger_path, tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")

    class NotJson(FakeResponse):
        def json(self):
            raise json.JSONDecodeError("Expecting value", "<html>", 0)

    class Dropping(FakeClient):
        def post(self, url, **kw):
            answer = super().post(url, **kw)
            if isinstance(answer, Exception):
                raise answer
            return answer

    client = Dropping([FakeResponse(502, text="Bad Gateway"), httpx.ConnectError("[Errno 111] Connection refused"),
                       NotJson(200, text="<html>proxy page</html>"), FakeResponse(500, text="internal"),
                       FakeResponse(200, {"status": "completed"})])
    monkeypatch.setattr(gemini35t, "_client", lambda: client)
    waits = []
    caller = gemini35t.Caller(ledger=KeyLedger("g", "m", PACIFIC, state_path=ledger_path), entries=entries("a"),
                              sleep=waits.append)
    # Four tries per window request: the fifth answer is never asked for.
    with pytest.raises(gemini35t.Gemini35tError, match="no answer after retries"):
        caller(audio, language=None, seconds=540)
    assert waits == [gemini35t.TRANSIENT_WAIT] * 3


def test_35t_does_not_retry_a_bad_request(ledger_path, tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    monkeypatch.setattr(gemini35t, "_client", lambda: FakeClient([FakeResponse(400, text="invalid argument")]))
    caller = gemini35t.Caller(ledger=KeyLedger("g", "m", PACIFIC, state_path=ledger_path), entries=entries("a"),
                              sleep=lambda s: pytest.fail("waited"))
    with pytest.raises(gemini35t.Gemini35tError, match="HTTP 400"):
        caller(audio, language=None, seconds=540)


def test_relink_failure_while_comparing_makes_everyone_new(source, tmp_path, monkeypatch):
    from finesub.speech.joint import relink

    warned = []
    monkeypatch.setattr(relink, "current_reporter", lambda: type("R", (), {"warning": lambda self, *a, **k: warned.append(a)})())

    class Broken(FakeVoiceprints):
        def embed(self):
            def embed(piece, mask):
                raise RuntimeError("bad audio")
            return embed

    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S01")]),
               "s-100.0-200.0": moss_answer([(10, 50, "う", "S01")])}
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 150)], pcm=voices(600, []),
                           voiceprints=Broken())
    assert [u.speaker for u in t.units()] == ["S01", "c1:S01", "S01"] and len(warned) == 1
