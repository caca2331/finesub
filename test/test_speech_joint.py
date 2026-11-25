"""Cloud ASR adapters (`finesub.speech.joint`): caching, keys, 35t / MOSS Pro / MAI2 answers to turns.

No network and no ffmpeg: every provider call is a fake, and the clip cutter is patched out."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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

from .support.fakes import WarningCodeRecorder


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


def silent(source: Path) -> mosspro.SpeechSpans:
    """Silero spans for the fake `source`: none, so every cut stays at its nominal time."""
    return mosspro.SpeechSpans(source, [])


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

    monkeypatch.setattr("finesub.speech.joint.cache.current_reporter", lambda: WarningCodeRecorder(codes))
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


def test_a_reservation_that_waits_for_the_lock_across_midnight_keeps_the_new_days_account(ledger_path, monkeypatch):
    # codex acceptance review: the day was read before the lock, so a job that waited past midnight wrote
    # yesterday's row over the calls another job had recorded on the new day
    from finesub.speech.joint import quota

    now = [datetime(2026, 10, 30, 16, 1, tzinfo=timezone.utc)]  # 00:01 in Beijing
    key = entries("a")
    KeyLedger("m", "m", BEIJING, state_path=ledger_path, clock=lambda: now[0]).record(key[0], 4000)
    now[0] = datetime(2026, 10, 30, 15, 59, tzinfo=timezone.utc)  # the waiting job starts before midnight
    section = quota.state_store.state_section

    def late(*a, **kw):  # the lock is had after midnight
        now[0] = datetime(2026, 10, 30, 16, 2, tzinfo=timezone.utc)
        return section(*a, **kw)

    monkeypatch.setattr(quota.state_store, "state_section", late)
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path, clock=lambda: now[0])
    _, day, _ = ledger.reserve_with(key, lambda usages: (0, 1000, None), what="m")
    assert day == "2026-10-31" and ledger.usage(key[0])["seconds"] == 5000


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
    ledger.record(keys[0], 3200)  # 0.91 h of 1.2, counted x1.02
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
        ledger.record(key, 4300)  # 1.22 h counted (x1.02): past the 1.2 h stop on both
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
    ledger.record(keys[0], 4300)
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
    ledger.record(key[0], 4300)
    caller = mosspro.Caller(ledger=ledger, entries=key)
    assert caller.paid is False
    with pytest.raises(QuotaExhausted):
        caller(tmp_path / "x.mp3", seconds=60)


def test_moss_refusal_says_when_to_rerun(ledger_path, tmp_path):
    caller = mosspro.Caller(ledger=KeyLedger("m", "m", BEIJING, state_path=ledger_path), entries=entries("a"), paid=False)
    caller._once = lambda audio, key: None  # HTTP 402
    with pytest.raises(QuotaExhausted, match="free hours as used up.*Beijing midnight"):
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


# --- filling the keys (owner 2026-11-09; the planner itself: test_speech_joint_fill.py) -----------------------

ROOM = mosspro.DAILY_HOURS * 3600 / mosspro.SERVER_RATIO  # a fresh key's free seconds


class SpanServer:
    """MOSS Pro over the network: each upload is answered with one segment over the span it was cut from."""

    def __init__(self, monkeypatch, refuse=()):
        self.calls, self.refuse = [], set(refuse)

        def clip(src, start, end, out, **_):
            Path(out).write_text(f"{start} {end}", encoding="utf-8")
            return Path(out)

        monkeypatch.setattr(mosspro, "extract_upload_mp3", clip)

    def __call__(self, audio, key):
        a, b = map(float, Path(audio).read_text(encoding="utf-8").split())
        self.calls.append((round(a, 1), round(b, 1), key))
        if key in self.refuse:
            return None
        return moss_answer([(0, b - a - 1, f"{a:.0f}", "S01")])


def fill_caller(ledger, keys, server, *, paid):
    caller = mosspro.Caller(ledger=ledger, entries=keys, paid=paid)
    caller._once = server
    return caller


def moss_run(source, tmp_path, caller, total, **kw):
    return mosspro.transcribe(source, cache_root=tmp_path / "cloud", language="ja", caller=caller, duration=total,
                              speech=mosspro.SpeechSpans(source, [], voices(total, [])),
                              voiceprints=FakeVoiceprints(available=False), **kw)


def test_moss_source_no_key_holds_is_cut_across_keys_and_a_rerun_keeps_the_cuts(source, tmp_path, ledger_path, monkeypatch):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    ledger.record(keys[0], ROOM - 1200)
    ledger.record(keys[1], ROOM - 2000)
    server = SpanServer(monkeypatch)
    t = moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=False), 2800.0)
    assert server.calls == [(0.0, 1200.0, "key-a"), (1140.0, 2800.0, "key-b")]
    assert {"parts": [[0.0, 1200.0], [1140.0, 2800.0]]} in t.log
    assert {"keys": [["s-0.0-1200.0", "a"], ["s-1140.0-2800.0", "b"]]} in t.log
    # other room today, and a replay: the first run's pieces, nothing asked
    fresh = KeyLedger("m", "m", BEIJING, state_path=tmp_path / "other-state")
    for offline in (False, True):
        again = SpanServer(monkeypatch)
        t2 = moss_run(source, tmp_path, fill_caller(fresh, keys, again, paid=False), 2800.0, offline=offline)
        assert again.calls == [] and [u.text for u in t2.units()] == [u.text for u in t.units()]
        assert {"keys": [["s-0.0-1200.0", "a"], ["s-1140.0-2800.0", "b"]]} in t2.log


class ScriptServer(SpanServer):
    """Answers each upload from a script: per span, its answers in the order they are asked."""

    def __init__(self, monkeypatch, script):
        super().__init__(monkeypatch)
        self.script = {span: list(answers) for span, answers in script.items()}

    def __call__(self, audio, key):
        a, b = map(float, Path(audio).read_text(encoding="utf-8").split())
        self.calls.append((round(a, 1), round(b, 1), key))
        return self.script[(round(a, 1), round(b, 1))].pop(0)


def test_moss_rerun_finds_the_main_calls_among_retries_capped_pieces_and_gaps(source, tmp_path, ledger_path, monkeypatch):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b", "c")
    for key, room in zip(keys, (1200, 2500, 2000)):
        ledger.record(key, ROOM - room)
    script = {(0.0, 1200.0): [moss_answer([(0, 1190, "日本語の訳文です", "S01")]),  # wrong script: retried
                              moss_answer([(0, 500, "first part", "S01"), (600, 0, "bro ken", "S02")], tokens=16384)],
              # the output cap at 500 s: the next piece starts there, whole on b
              (500.0, 2800.0): [moss_answer([(0, 300, "second part", "S01"), (700, 2290, "fourth part", "S01")])],
              (800.0, 1200.0): [moss_answer([(100, 150, "third part", "S02")])]}  # the gap, Silero hears speech
    server = ScriptServer(monkeypatch, script)
    speech = mosspro.SpeechSpans(source, [(900.0, 950.0)], voices(2800.0, []))

    def run(caller, **kw):
        return mosspro.transcribe(source, cache_root=tmp_path / "cloud", language="en", caller=caller, duration=2800.0,
                                  speech=speech, voiceprints=FakeVoiceprints(available=False), **kw)

    t = run(fill_caller(ledger, keys, server, paid=False))
    assert server.calls == [(0.0, 1200.0, "key-a"), (0.0, 1200.0, "key-c"), (500.0, 2800.0, "key-b"),
                            (800.0, 1200.0, "key-c")]
    texts = ["first part", "second part", "third part", "fourth part"]
    assert [u.text for u in t.units()] == texts
    keys_log = [["s-0.0-1200.0", "a"], ["s-0.0-1200.0-retry", "c"], ["s-500.0-2800.0", "b"],
                ["s-800.0-1200.0", "c"]]
    assert {"keys": keys_log} in t.log
    # the next day, other room: online and offline alike take every call from the cache and spend nothing
    fresh = KeyLedger("m", "m", BEIJING, state_path=tmp_path / "other-state")
    for offline in (False, True):
        again = SpanServer(monkeypatch)
        t2 = run(fill_caller(fresh, keys, again, paid=False), offline=offline)
        assert again.calls == [] and [u.text for u in t2.units()] == texts and {"keys": keys_log} in t2.log
        assert all(fresh.usage(k)["calls"] == 0 for k in keys)


def test_moss_the_rest_after_a_capped_answer_is_filled_like_any_piece(source, tmp_path, ledger_path, monkeypatch):
    # codex acceptance review: a capped answer's rest (1200 s) fits no key whole; it is cut over the free hours
    # instead of going whole on the paid key
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b", "c")  # c: the paid key
    for key, room in zip(keys, (2100, 1000, 1000)):
        ledger.record(key, ROOM - room)

    class Capped(SpanServer):
        def __call__(self, audio, key):
            a, b = map(float, Path(audio).read_text(encoding="utf-8").split())
            if (a, b) == (0.0, 2000.0):
                self.calls.append((a, b, key))
                return moss_answer([(0, 800, "始め", "S01"), (900, 0, "壊 れ た", "S01")], tokens=16384)
            return super().__call__(audio, key)

    server = Capped(monkeypatch)
    t = moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=True), 2000.0)
    assert server.calls == [(0.0, 2000.0, "key-a"), (800.0, 1800.0, "key-b"), (1740.0, 2000.0, "key-c")]
    assert all(ledger.usage(k)["seconds"] <= ROOM + 1e-6 for k in keys)  # nothing on the balance
    assert [u.text for u in t.units()][0] == "始め" and t.units()[-1].end > 1990


def test_moss_capped_piece_has_its_gaps_asked_and_its_cap_point_is_one_name(source, tmp_path, ledger_path, monkeypatch):
    # codex acceptance review 2: a capped piece skipped the gap pass over what it did transcribe, and a cap point
    # with three decimals was rounded for the next piece's call name and not for its lookup on resume
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a")
    script = {(0.0, 600.0): [moss_answer([(0, 100, "一", "S01"), (200, 323.449, "三", "S01"), (400, 0, "壊 れ た", "S01")],
                                         tokens=16384)],
              (100.0, 200.0): [moss_answer([(20, 30, "二", "S02")])],  # Silero hears 120-130 s
              (323.4, 600.0): [moss_answer([(1, 270, "四", "S01")])]}
    server = ScriptServer(monkeypatch, script)
    speech = mosspro.SpeechSpans(source, [(120.0, 130.0)], voices(600.0, []))

    def run(caller, **kw):
        return mosspro.transcribe(source, cache_root=tmp_path / "cloud", language="ja", caller=caller, duration=600.0,
                                  speech=speech, voiceprints=FakeVoiceprints(available=False), **kw)

    t = run(fill_caller(ledger, keys, server, paid=False))
    assert server.calls == [(0.0, 600.0, "key-a"), (100.0, 200.0, "key-a"), (323.4, 600.0, "key-a")]
    assert [u.text for u in t.units()] == ["一", "二", "三", "四"]
    fresh = KeyLedger("m", "m", BEIJING, state_path=tmp_path / "other-state")
    for offline in (False, True):
        again = SpanServer(monkeypatch)
        t2 = run(fill_caller(fresh, keys, again, paid=False), offline=offline)
        assert again.calls == [] and [u.text for u in t2.units()] == ["一", "二", "三", "四"]


def test_moss_capped_piece_out_of_free_hours_goes_on_from_its_cap_point_the_next_day(source, tmp_path, ledger_path, monkeypatch):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a")
    ledger.record(keys[0], ROOM - 700)
    script = {(0.0, 700.0): [moss_answer([(0, 300, "一", "S01"), (400, 0, "壊 れ た", "S01")], tokens=16384)],
              (300.0, 1200.0): [moss_answer([(1, 890, "二", "S01")])]}
    server = ScriptServer(monkeypatch, script)
    with pytest.raises(QuotaExhausted):
        moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=False), 1200.0)
    assert server.calls == [(0.0, 700.0, "key-a")]
    fresh = KeyLedger("m", "m", BEIJING, state_path=tmp_path / "other-state")
    t = moss_run(source, tmp_path, fill_caller(fresh, keys, server, paid=False), 1200.0)
    assert server.calls[1:] == [(300.0, 1200.0, "key-a")] and [u.text for u in t.units()] == ["一", "二"]


def test_moss_pieces_planned_one_after_another_see_each_others_reservations(ledger_path, tmp_path):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")  # b: the paid key, with today's free hours untouched
    first = mosspro.Caller(ledger=ledger, entries=keys, paid=True).reserve_piece(3000.0, [(0.0, 0.0)], [])
    second = mosspro.Caller(ledger=ledger, entries=keys, paid=True).reserve_piece(3000.0, [(0.0, 0.0)], [])
    assert (first.entry.label, second.entry.label) == ("a", "b")
    assert second.piece.billed == 0  # b's free hours hold it: nothing on the balance
    assert ledger.usage(keys[0])["seconds"] == ledger.usage(keys[1])["seconds"] == 3000


def test_moss_piece_is_planned_inside_the_ledger_lock(ledger_path, monkeypatch):
    # planning outside the lock and recording inside would let two jobs plan on one snapshot (codex)
    from finesub.speech.joint import fill, quota

    inside, seen = [False], []
    section = quota.state_store.state_section

    class Held:
        def __init__(self, *a, **kw):
            self.cm = section(*a, **kw)

        def __enter__(self):
            inside[0] = True
            return self.cm.__enter__()

        def __exit__(self, *exc):
            inside[0] = False
            return self.cm.__exit__(*exc)

    plan = fill.plan
    monkeypatch.setattr(quota.state_store, "state_section", Held)
    monkeypatch.setattr(fill, "plan", lambda *a, **kw: seen.append(inside[0]) or plan(*a, **kw))
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    mosspro.Caller(ledger=ledger, entries=entries("a"), paid=False).reserve_piece(1000.0, [(0.0, 0.0)], [])
    assert seen == [True]


def test_moss_paid_key_refused_is_left_out_and_the_free_hours_fill(source, tmp_path, ledger_path, monkeypatch):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    ledger.record(keys[0], ROOM - 1000)
    server = SpanServer(monkeypatch, refuse={"key-b"})
    with pytest.raises(QuotaExhausted, match="paid last key"):
        moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=True), 1500.0)
    assert server.calls == [(0.0, 1500.0, "key-b"), (0.0, 1000.0, "key-a")]  # whole on b's free hours, then replanned
    assert ledger.usage(keys[1]) == {"calls": 0, "seconds": 0.0, "exhausted": False}  # usable after a top-up


def test_moss_piece_that_fails_before_its_upload_gives_its_reservation_back(source, tmp_path, ledger_path, monkeypatch):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a")
    server = SpanServer(monkeypatch)

    def broken(*_, **__):
        raise OSError("ffmpeg failed")

    monkeypatch.setattr(mosspro, "extract_upload_mp3", broken)
    with pytest.raises(OSError):
        moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=False), 1500.0)
    assert server.calls == [] and ledger.usage(keys[0]) == {"calls": 0, "seconds": 0.0, "exhausted": False}


def test_moss_free_hours_fill_then_stop_and_the_next_day_goes_on_from_the_cache(source, tmp_path, ledger_path, monkeypatch):
    now = [datetime(2026, 11, 9, 4, 0, tzinfo=timezone.utc)]
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path, clock=lambda: now[0])
    keys = entries("a")
    ledger.record(keys[0], ROOM - 1000)
    server = SpanServer(monkeypatch)
    with pytest.raises(QuotaExhausted):
        moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=False), 1500.0)
    assert server.calls == [(0.0, 1000.0, "key-a")]
    now[0] += timedelta(days=1)
    t = moss_run(source, tmp_path, fill_caller(ledger, keys, server, paid=False), 1500.0)
    assert server.calls[1:] == [(940.0, 1500.0, "key-a")]
    assert [u.text for u in t.units()] == ["0", "940"]


# --- stitching ----------------------------------------------------------------------------------------------


def w(start, end, text, speaker="A"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_stitch_keeps_each_word_once_in_output_order_and_each_window_s_speakers_apart():
    # B's backchannel comes after A's later word, as 35t emits it; the order survives the seam. X talks with A in
    # the overlap, yet stays its own label: who is who across windows is the judge's (owner 2026-11-21)
    left = [w(0, 1, "a1"), w(5, 6, "a2"), w(2, 3, "b1", "B"), w(40, 41, "a3")]
    right = [w(40, 41, "a3", "X"), w(55, 56, "a4", "X")]
    out = stitch([(0, 50, left), (40, 60, right)], segments=False)
    assert [u["text"] for u in out] == ["a1", "a2", "b1", "a3", "a4"]
    assert [u["speaker"] for u in out] == ["w0:A", "w0:A", "w0:B", "w1:X", "w1:X"]


def test_stitch_renames_the_label_a_segment_says_again_with_the_speakers():
    left = [w(0, 1, "a1"), {**w(5, 6, "a2", "c1:B"), "twin": "A"}, w(40, 41, "a3")]
    right = [w(40, 41, "a3", "X"), {**w(55, 56, "a4", "c1:Y"), "twin": "X"}]
    out = stitch([(0, 50, left), (40, 60, right)], segments=False)
    assert [u.get("twin") for u in out] == [None, "w0:A", None, "w1:X"]


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


def transcribe_35t(source, tmp_path, words_for, total, spans=()):
    """35t over `total` s where each window's answer is `words_for(start, end)` (absolute times); Silero hears
    `spans` (none: every cut at its nominal time)."""
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
                                 repair="off", speech=mosspro.SpeechSpans(source, list(spans)))
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


def test_35t_windows_are_cut_at_silero_pauses(source, tmp_path):
    # pauses at 520 s (in the REACH s before 540), 445 s (60-90 s before that end) and 970 s
    spans = [(0.0, 444.5), (445.5, 519.0), (521.0, 969.5), (970.5, 1200.0)]
    _, calls = transcribe_35t(source, tmp_path, speech_every_second, 1200.0, spans=spans)
    assert calls == [(0.0, 520.0), (445.0, 970.0), (910.0, 1200.0)]


#: Pauses at 520 s (a window's end), 435 s (1 s) and 450 s (0.4 s, 60-90 s before that end): the short one is taken
#: only where the words before break there.
SHORT_PAUSE_SPANS = [(0.0, 434.5), (435.5, 449.8), (450.2, 519.0), (521.0, 1200.0)]


def marked_at(start, end, mark=True):
    """A word a second; with `mark`, the one at 449 s ends a sentence (the break 449.5-450 s)."""
    words, status = speech_every_second(start, end)
    return [(a, b, f"{text}。" if mark and a == 449 else text, s) for a, b, text, s in words], status


@pytest.mark.parametrize("mark, start", [(True, 450.0), (False, 435.0)])
def test_35t_the_next_window_starts_at_a_pause_the_words_before_break_at(source, tmp_path, mark, start):
    _, calls = transcribe_35t(source, tmp_path, lambda a, b: marked_at(a, b, mark), 1200.0, spans=SHORT_PAUSE_SPANS)
    assert calls[:2] == [(0.0, 520.0), (start, start + 540.0)]


@pytest.mark.parametrize("marked, start", [("whole", 435.0), ("half", 450.0)])
def test_35t_after_a_split_the_next_window_starts_by_the_second_halfs_words(source, tmp_path, marked, start):
    def words_for(a, b):
        words, _ = marked_at(a, b, mark=(a, b) == ((0.0, 520.0) if marked == "whole" else (230.0, 520.0)))
        return words, "incomplete" if (a, b) == (0.0, 520.0) else "completed"  # the whole window: degenerate

    _, calls = transcribe_35t(source, tmp_path, words_for, 1200.0, spans=SHORT_PAUSE_SPANS)
    assert calls[:4] == [(0.0, 520.0), (0.0, 290.0), (230.0, 520.0), (start, start + 540.0)]


def test_35t_answers_cached_at_the_fixed_cuts_replay_though_silero_hears_pauses(source, tmp_path):
    cache = tmp_path / "cloud" / joint.GEMINI_35T
    cache.mkdir(parents=True)
    for a, b in [(0.0, 540.0), (480.0, 1020.0), (960.0, 1200.0)]:
        words = [(x - a, x - a + 0.5, f"w{x}", "spk:0") for x in range(int(a) + 1, int(b) - 1)]
        (cache / f"w-{a:.1f}-{b:.1f}.raw.json").write_text(json.dumps(answer(words)), encoding="utf-8")
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=1200.0, repair="off",
                             offline=True, speech=mosspro.SpeechSpans(source, [(0.0, 499.0), (501.0, 1200.0)]))
    assert {f"w{x}" for x in (1, 500, 1197)} <= {u.text for u in t.units()}


def test_35t_windows_cached_up_to_60_s_before_their_limit_replay(source, tmp_path):
    # codex acceptance review: an end cut 40 s before the limit (by the 60 s reach before 2026-11-22) is found
    # though the reach planned now is 30 s
    cache = tmp_path / "cloud" / joint.GEMINI_35T
    cache.mkdir(parents=True)
    for a, b in [(0.0, 500.0), (440.0, 980.0), (920.0, 1200.0)]:
        words = [(x - a, x - a + 0.5, f"w{x}", "spk:0") for x in range(int(a) + 1, int(b) - 1)]
        (cache / f"w-{a:.1f}-{b:.1f}.raw.json").write_text(json.dumps(answer(words)), encoding="utf-8")
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=1200.0, repair="off",
                             offline=True, speech=mosspro.SpeechSpans(source, [(0.0, 499.0), (501.0, 1200.0)]))
    assert {f"w{x}" for x in (1, 497, 1197)} <= {u.text for u in t.units()}


def test_35t_continues_an_early_stop_at_a_pause_before_its_last_word():
    assert gemini35t._continued(300.0, 120.0, [(285.0, 1.0), (292.0, 0.5), (275.0, 3.0)]) == ((280.0, 290.0), 285.0)
    # the floor is later: a pause just after it
    assert gemini35t._continued(59.5, 120.0, [(125.0, 0.4)]) == ((120.0, 130.0), 125.0)
    assert gemini35t._continued(59.5, 120.0, []) == ((120.0, 130.0), 120.0)


def test_35t_splits_a_degenerate_window_at_pauses_near_its_middle():
    pauses = [(280.0, 1.0), (305.0, 2.0), (215.0, 1.5), (233.0, 0.3)]  # 280, 215: further in or out than allowed
    assert gemini35t._split(0.0, 540.0, pauses, []) == [(0.0, 305.0), (233.0, 540.0)]
    assert gemini35t._split(0.0, 540.0, [], []) == [(0.0, 300.0), (240.0, 540.0)]
    # halves cached at the fixed middle are found
    assert gemini35t._split(0.0, 540.0, pauses, [(0.0, 300.0), (240.0, 540.0)]) == [(0.0, 300.0), (240.0, 540.0)]


def test_35t_short_window_halves_cut_at_fixed_times_replay():
    assert gemini35t._split(0.0, 64.0, [], []) == [(0.0, 62.0), (2.0, 64.0)]  # as fixed halves were cut
    assert gemini35t._split(0.0, 64.0, [(62.6, 1.0), (1.4, 1.0)], [(0.0, 62.0), (2.0, 64.0)]) == [(0.0, 62.0),
                                                                                                (2.0, 64.0)]


@pytest.mark.parametrize("length", [3.0, 30.0, 45.0, 61.0, 64.0, 90.0, 120.0])
def test_35t_halves_of_a_short_window_are_each_shorter_than_it(length):
    pauses = [(0.5, 2.0), (length - 0.5, 2.0), (length / 2, 2.0)]  # pauses at its very edges and middle
    for pauses_heard in (pauses, []):
        (a, b), (c, d) = gemini35t._split(0.0, length, pauses_heard, [])
        assert a == 0.0 and d == length and b < length and c > 0.0 and b - c > 0  # shorter, and overlapping
    assert gemini35t._split(0.0, 90.0, [], []) == [(0.0, 75.0), (15.0, 90.0)]  # as fixed halves were cut


def test_35t_speakerless_halves_are_at_most_reask_half_and_replay_their_fixed_cuts():
    assert gemini35t._split(0.0, 400.0, [], [], gemini35t.REASK_HALF) == [(0.0, 300.0), (100.0, 400.0)]
    pauses = [(296.0, 1.0), (104.0, 1.0), (305.0, 3.0)]  # 305: past REASK_HALF
    assert gemini35t._split(0.0, 400.0, pauses, [], gemini35t.REASK_HALF) == [(0.0, 296.0), (104.0, 400.0)]
    fixed = [(0.0, 300.0), (100.0, 400.0)]
    assert gemini35t._split(0.0, 400.0, pauses, fixed, gemini35t.REASK_HALF) == fixed


def test_35t_reask_of_gaps_merged_to_the_old_limit_replays_its_fixed_pads():
    # two gaps 515 s end to end were one re-ask [a - REASK_PAD, b + REASK_PAD] before the cuts moved to pauses
    assert gemini35t._padded(100.0, 615.0, 2000.0, [(88.0, 1.0), (627.0, 1.0)], [(90.0, 625.0)]) == (90.0, 625.0)
    lo, hi = gemini35t._padded(100.0, 615.0, 2000.0, [(88.0, 1.0), (627.0, 1.0)], [])
    assert hi - lo <= gemini35t.WINDOW and lo <= 90.0 and hi >= 625.0


def put_35t(cache, a, b, words, status="completed"):
    (cache / f"w-{a:.1f}-{b:.1f}.raw.json").write_text(
        json.dumps(answer([(x - a, y - a, text, "spk:0") for x, y, text in words], status)), encoding="utf-8")


def test_35t_a_tail_cached_a_name_apart_from_a_read_half_is_taken(source, tmp_path):
    cache = tmp_path / "cloud" / joint.GEMINI_35T
    cache.mkdir(parents=True)
    put_35t(cache, 0.0, 540.0, [(x * 0.2, x * 0.2 + 0.1, "あ") for x in range(1, 40)])  # degenerate
    put_35t(cache, 0.0, 300.0, [(x, x + 0.5, f"w{x}") for x in range(1, 245)])
    put_35t(cache, 240.0, 540.0, [(x, x + 0.5, f"w{x}") for x in range(241, 250)] + [(249.6, 250.1, "w250")])
    put_35t(cache, 240.1, 540.0, [(x, x + 0.5, f"w{x}") for x in range(251, 539)])  # the tail: last word - 10 s
    # a pause at 235 s, in the tail's range: taking it would ask a window never cached
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=540.0, repair="off",
                             offline=True, speech=mosspro.SpeechSpans(source, [(0.0, 234.5), (235.5, 540.0)]))
    assert "w530" in {u.text for u in t.units()}


def test_35t_a_tail_that_is_a_read_half_still_splits_it_when_that_half_degenerated(source, tmp_path):
    cache = tmp_path / "cloud" / joint.GEMINI_35T
    cache.mkdir(parents=True)
    put_35t(cache, 0.0, 540.0, [(x * 0.2, x * 0.2 + 0.1, "あ") for x in range(1, 40)])  # degenerate
    put_35t(cache, 0.0, 300.0, [(x, x + 0.5, f"w{x}") for x in range(1, 245)])
    # the half repeats from 241 s and stops at 250 s; its own halves hear it right
    put_35t(cache, 240.0, 540.0, [(x, x + 0.5, "また") for x in range(241, 250)] + [(249.5, 250.0, "w250")],
            status="incomplete")
    put_35t(cache, 240.0, 420.0, [(x, x + 0.5, f"w{x}") for x in range(241, 419)])
    put_35t(cache, 360.0, 540.0, [(x, x + 0.5, f"w{x}") for x in range(361, 539)])
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=540.0, repair="off",
                             offline=True, speech=mosspro.SpeechSpans(source, [(0.0, 540.0)]))
    texts = [u.text for u in t.units()]
    assert "また" not in texts and {"w241", "w245", "w530"} <= set(texts)  # the split half is replaced, not kept
    assert len(texts) == len(set(texts))
    assert [e["split"] for e in t.log if "split" in e] == [[0.0, 540.0], [240.0, 540.0]]  # the half, as before


def test_35t_a_tail_that_would_be_a_read_half_again_is_not_asked(source, tmp_path):
    cache = tmp_path / "cloud" / joint.GEMINI_35T
    cache.mkdir(parents=True)
    put_35t(cache, 0.0, 540.0, [(x * 0.2, x * 0.2 + 0.1, "あ") for x in range(1, 40)])  # degenerate
    put_35t(cache, 0.0, 300.0, [(x, x + 0.5, f"w{x}") for x in range(1, 245)])
    put_35t(cache, 240.0, 540.0, [(x, x + 0.5, f"w{x}") for x in range(241, 250)] + [(249.5, 250.0, "w250")])
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=540.0, repair="off",
                             offline=True, speech=mosspro.SpeechSpans(source, [(0.0, 540.0)]))
    assert {"tail_read_already": [240.0, 540.0]} in t.log


def test_35t_tail_after_a_split_window_is_not_its_cached_half_again(source, tmp_path):
    cache = tmp_path / "cloud" / joint.GEMINI_35T
    cache.mkdir(parents=True)

    def put(a, b, words, status="completed"):
        (cache / f"w-{a:.1f}-{b:.1f}.raw.json").write_text(
            json.dumps(answer([(x - a, y - a, text, "spk:0") for x, y, text in words], status)), encoding="utf-8")

    put(0.0, 540.0, [(x * 0.2, x * 0.2 + 0.1, "あ") for x in range(1, 40)])  # degenerate
    put(0.0, 300.0, [(x, x + 0.5, f"w{x}") for x in range(1, 255)])
    put(240.0, 540.0, [(x, x + 0.5, f"w{x}") for x in range(241, 260)])  # both stop by 259.5 s
    put(249.5, 540.0, [(x, x + 0.5, f"w{x}") for x in range(251, 539)])  # the tail, from the last word - 10 s
    t = gemini35t.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=540.0, repair="off",
                             offline=True, speech=mosspro.SpeechSpans(source, [(0.0, 245.5), (246.5, 540.0)]))
    assert {"tail": [259.5, 540.0]} in t.log and "w530" in {u.text for u in t.units()}


def test_mai2_windows_are_cut_at_silero_pauses(source, tmp_path):
    def fetch(self, name, call):
        return {"words": []}

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    try:
        t = mai2.transcribe(source, cache_root=tmp_path, language="en", via="openrouter", repair="off", caller=object(),
                            duration=9000.0, speech=mosspro.SpeechSpans(source, [(0.0, 7119.0), (7121.0, 9000.0)]))
    finally:
        mp.undo()
    assert t.log == [{"windows": [[0.0, 7120.0], [7060.0, 9000.0]]}]


@pytest.mark.parametrize("mark, start", [(True, 7045.0), (False, 7035.0)])
def test_mai2_the_next_window_starts_at_a_pause_the_words_before_break_at(source, tmp_path, mark, start):
    def fetch(self, name, call):
        a = float(name.split("-")[1])
        words = [{"start": x - a, "end": x - a + 0.5, "word": "w" if x != 7044 or not mark else "w.", "speaker": 1}
                 for x in range(int(a) + 1, int(a) + 7139)]
        return {"words": words}

    mp = pytest.MonkeyPatch()
    mp.setattr(CallCache, "fetch", fetch)
    try:
        # pauses at 7120 s (the end), 7035 s (1 s) and 7045 s (0.4 s, after the word ending a sentence at 7044.5 s)
        spans = [(0.0, 7034.5), (7035.5, 7044.8), (7045.2, 7119.0), (7121.0, 9000.0)]
        t = mai2.transcribe(source, cache_root=tmp_path, language="en", via="openrouter", repair="off", caller=object(),
                            duration=9000.0, speech=mosspro.SpeechSpans(source, spans))
    finally:
        mp.undo()
    assert t.log == [{"windows": [[0.0, 7120.0], [start, 9000.0]]}]


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
    mp.setattr(CallCache, "names", lambda self: list(answers))
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


def test_moss_old_cache_capped_twice_is_read_whole(source, tmp_path):
    # codex acceptance review 3: an old task cut its continuations at the unrounded cap point (123.44, then
    # 123.44 + 200.02) and named them at 0.1 s; read from the names' starts, the second cap point is 323.42
    answers = {"s-0.0-600.0": moss_answer([(0, 123.44, "一", "S01"), (300, 0, "壊 れ た", "S01")], tokens=16384),
               "s-123.4-600.0": moss_answer([(1, 200.02, "二", "S01"), (300, 0, "壊 れ た", "S01")], tokens=16384),
               "s-323.5-600.0": moss_answer([(1, 270, "三", "S01")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0)
    assert calls == ["s-0.0-600.0", "s-123.4-600.0", "s-323.5-600.0"]
    assert [u.text for u in t.units()] == ["一", "二", "三"]


def test_moss_piece_capped_a_moment_in_does_not_find_itself_again(source, tmp_path):
    # codex acceptance review 4: with the 0.1 s lookup, a piece capped at 0.1 s matched itself on replay
    answers = {"s-0.0-600.0": moss_answer([(0, 0.1, "一", "S01"), (300, 0, "壊 れ た", "S01")], tokens=16384),
               "s-0.1-600.0": moss_answer([(1, 500, "二", "S01")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0)
    assert calls == ["s-0.0-600.0", "s-0.1-600.0"] and [u.text for u in t.units()] == ["一", "二"]


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


@pytest.mark.parametrize("name", ["s-99.7-200.3", "s-100.0-200.0"], ids=["out-to-pauses", "old-name"])
def test_moss_gap_edges_go_out_to_the_pauses_they_lie_in_and_an_old_call_replays(source, tmp_path, name):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "一", "S01"), (200, 600, "三", "S01")]),
               name: moss_answer([(10, 90, "二", "S02")])}
    voiced = [(0.0, 99.0), (100.4, 150.0), (160.0, 199.6), (201.0, 600.0)]  # the gap's edges in pauses at 99.7, 200.3
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=voiced)
    assert calls == ["s-0.0-600.0", name] and [u.text for u in t.units()] == ["一", "二", "三"]


def test_moss_gap_cached_at_its_own_edges_reads_from_them_unrounded(source, tmp_path):
    # codex acceptance review: the old call s-100.0-200.0 was asked from 100.04 s; read from 100.0 s, what it
    # recovered overlapped "一" by 0.04 s and was dropped
    answers = {"s-0.0-600.0": moss_answer([(0, 100.04, "一", "S01"), (200, 600, "三", "S01")]),
               "s-100.0-200.0": moss_answer([(0, 90, "二", "S02")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(0.0, 600.0)])
    assert calls == ["s-0.0-600.0", "s-100.0-200.0"] and [u.text for u in t.units()] == ["一", "二", "三"]


def test_moss_gap_call_lookup_stays_inside_the_piece(source, tmp_path):
    # codex acceptance review: the tail gap 1140-1200 s of the piece ending at 1200 s must not read the next
    # piece's main call s-1140.0-1200.1 (inside the names' 0.1 s) for its own s-1140.0-1200.0
    answers = {"s-0.0-1200.0": moss_answer([(0, 1140, "一", "S01")]),
               "s-1140.0-1200.0": moss_answer([(1, 50, "二", "S01")]),
               "s-1140.0-1200.1": moss_answer([(1, 50, "二", "S01")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 1200.1, voiced=[(0.0, 1200.1)])
    assert calls[:2] == ["s-0.0-1200.0", "s-1140.0-1200.0"]


@pytest.mark.parametrize("gap", ["s-6.0-600.0", "s-5.0-600.0-again"], ids=["old-name", "new"])
def test_moss_gap_edges_stay_in_the_piece_and_never_find_its_own_main_call(source, tmp_path, gap):
    # the capped first piece's rest [5, 600] has a gap 6-600 s whose left edge lies in a pause (3-6.5 s) with its
    # middle before the piece: it stops at 5 s, the piece's own start, so the span is the main call's again
    answers = {"s-0.0-600.0": moss_answer([(0, 5, "前半", "S01"), (300, 0, "壊 れ た", "S02")], tokens=16384),
               "s-5.0-600.0": moss_answer([(0, 1, "続き", "S01")]), gap: moss_answer([(10, 20, "後半", "S01")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(0.0, 3.0), (6.5, 600.0)])
    assert calls == ["s-0.0-600.0", "s-5.0-600.0", gap]


@pytest.mark.parametrize("first, start", [([(0, 1129.9, "一", "S01"), (1130.3, 1199, "一の続き", "S01")], 1130.0),
                                          ([(0, 1199, "一", "S01")], 1115.0)], ids=["breaks", "no-break"])
def test_moss_next_piece_starts_at_a_pause_the_previous_pieces_segments_break_at(source, tmp_path, ledger_path,
                                                                                monkeypatch, first, start):
    ledger = KeyLedger("m", "m", BEIJING, state_path=ledger_path)
    keys = entries("a", "b")
    ledger.record(keys[0], ROOM - 1200)
    ledger.record(keys[1], ROOM - 2000)
    script = {(0.0, 1200.0): [moss_answer(first)], (start, 2800.0): [moss_answer([(0, 2800 - start, "二", "S01")])]}
    server = ScriptServer(monkeypatch, script)
    # pauses at 1115 s (1 s) and 1130 s (0.4 s), both 60-90 s before the cut at 1200 s
    speech = mosspro.SpeechSpans(source, [(0.0, 1114.5), (1115.5, 1129.8), (1130.2, 2800.0)], voices(2800.0, []))
    mosspro.transcribe(source, cache_root=tmp_path / "cloud", language="ja", duration=2800.0, speech=speech,
                       caller=fill_caller(ledger, keys, server, paid=False), voiceprints=FakeVoiceprints(available=False))
    assert server.calls == [(0.0, 1200.0, "key-a"), (start, 2800.0, "key-b")]


def test_moss_gap_is_rerequested_only_when_silero_hears_speech(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S01")]),
               "s-100.0-200.0": moss_answer([(10, 20, "拾った", "S02")])}
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(130, 132)])
    assert calls == ["s-0.0-600.0"]  # 2 s of speech: not worth a call
    t, calls = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 120)])
    assert calls == ["s-0.0-600.0", "s-100.0-200.0"]
    assert [u.text for u in t.units()] == ["あ", "拾った", "い"]


def test_moss_speakers_after_a_capped_answer_are_merged_by_voice(source, tmp_path):
    # the main call heard persons 1 (S01) and 2 (S02) and was capped at 200 s; the next piece numbers anew: its S01
    # is person 2, its S02 someone not heard before
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (100, 200, "い", "S02"),
                                           (300, 0, "壊", "S01")], tokens=16384),
               "s-200.0-600.0": moss_answer([(10, 100, "う", "S01"), (100, 390, "え", "S02")])}
    pcm = voices(600, [(0, 100, 1), (100, 200, 2), (210, 300, 2), (300, 590, 3)])
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, pcm=pcm, voiceprints=FakeVoiceprints())
    assert [(u.text, u.speaker) for u in t.units()] == [("あ", "w0:S01"), ("い", "w0:S02"), ("う", "w0:S02"),
                                                        ("え", "w1:S02")]
    assert {"merge": [["w1:S01", "w0:S02", 1.0]]} in t.log


def test_moss_gap_speakers_are_merged_and_too_little_voice_is_unknown(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S02")]),
               "s-100.0-200.0": moss_answer([(10, 50, "う", "S01"), (60, 62, "え", "S02")])}
    pcm = voices(600, [(0, 100, 1), (110, 150, 2), (160, 162, 1), (200, 590, 2)])
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 150)], pcm=pcm,
                           voiceprints=FakeVoiceprints())
    # 2 s of S02 in the gap: too little to tell who it is, though it is person 1's voice
    assert [u.speaker for u in t.units()] == ["S01", "S02", None, "S02"]
    assert {"merge": [["c1:S01", "S02", 1.0]]} in t.log and {"unknown": [["c1:S02", 2.0]]} in t.log


def test_a_re_requests_label_under_rescue_min_of_voice_is_unknown_and_only_it():
    from finesub.speech.joint import relink

    segs = [{"start": 0.0, "end": 40.0, "speaker": "S01"}, {"start": 40.0, "end": 44.0, "speaker": "S02"},
            {"start": 50.0, "end": 54.5, "speaker": "c1:S01"},  # 4.5 s: unknown
            {"start": 60.0, "end": 66.0, "speaker": "w1:c2:S01"}]  # 6 s (behind a part's prefix): merged
    pcm = voices(80, [(0, 40, 1), (40, 44, 2), (50, 54.5, 1), (60, 66, 1)])
    merges, unknown = relink.merge_by_voice(segs, lambda: pcm, FakeVoiceprints(), "test")
    assert unknown == [("c1:S01", 4.5)] and merges == [("w1:c2:S01", "S01", 1.0)]
    assert [s["speaker"] for s in segs] == ["S01", "S02", None, "S01"]  # S02, 4 s of the main call: kept


@pytest.mark.parametrize("cos, merged", [(0.88, False), (0.92, True)])
def test_the_moss_voice_merge_takes_only_pairs_from_0_9(cos, merged):
    """`relink.MERGE_THETA` (owner 2026-11-21): below 0.9 about one merge in ten joined two people, so those pairs
    are left to the speaker judge, shown as 高."""
    from types import SimpleNamespace

    from finesub.speech.joint import relink

    vec = {1: np.array([1.0, 0.0]), 2: np.array([cos, (1 - cos * cos) ** 0.5])}
    prints = SimpleNamespace(error=None, embed=lambda: lambda piece, mask: vec[int(round(float(np.median(piece)) * 32.768))])
    segs = [{"start": 0.0, "end": 10.0, "speaker": "S01"}, {"start": 20.0, "end": 30.0, "speaker": "w1:S01"}]
    merges, _ = relink.merge_by_voice(segs, lambda: voices(40, [(0, 10, 1), (20, 30, 2)]), prints, "test")
    assert bool(merges) == merged and len({s["speaker"] for s in segs}) == (1 if merged else 2)


def test_a_short_re_requests_line_that_says_a_replaced_segment_again_takes_its_label():
    from finesub.speech.joint import relink

    segs = [{"start": 0.0, "end": 40.0, "speaker": "S01"}, {"start": 40.0, "end": 44.0, "speaker": "S02"},
            {"start": 50.0, "end": 52.0, "speaker": "c1:S01", relink.TWIN: "S02"},  # says S02's line again
            {"start": 52.0, "end": 54.0, "speaker": "c1:S01"},  # in the gap: nothing to say again
            {"start": 60.0, "end": 62.0, "speaker": "c2:S01", relink.TWIN: "c1:S01"},  # its twin is short too
            {"start": 70.0, "end": 72.0, "speaker": "c3:S01", relink.TWIN: "c9:S02"}]  # its twin was replaced away
    pcm = voices(80, [(0, 40, 1), (40, 44, 2), (50, 52, 2), (52, 54, 1), (60, 62, 1), (70, 72, 1)])
    _, unknown = relink.merge_by_voice(segs, lambda: pcm, FakeVoiceprints(), "test")
    assert unknown == [("c1:S01", 4.0), ("c2:S01", 2.0), ("c3:S01", 2.0)]
    assert [s["speaker"] for s in segs] == ["S01", "S02", "S02", None, None, None]


def test_a_merge_that_fails_keeps_the_labels_already_made_unknown(monkeypatch):
    from finesub.speech.joint import relink

    monkeypatch.setattr(relink, "merge_labels", lambda *a: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(relink, "current_reporter", lambda: type("R", (), {"warning": lambda self, *a: None})())
    segs = [{"start": 0.0, "end": 40.0, "speaker": "S01"}, {"start": 50.0, "end": 52.0, "speaker": "c1:S01"}]
    pcm = voices(60, [(0, 40, 1), (50, 52, 1)])
    assert relink.merge_by_voice(segs, lambda: pcm, FakeVoiceprints(), "test") == ([], [("c1:S01", 2.0)])
    assert segs[1]["speaker"] is None


def test_moss_labels_one_call_split_a_person_into_are_merged(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (100, 200, "い", "S02"), (200, 300, "う", "S03")])}
    pcm = voices(600, [(0, 100, 1), (100, 200, 2), (200, 300, 1)])
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, pcm=pcm, voiceprints=FakeVoiceprints())
    assert [u.speaker for u in t.units()] == ["S01", "S02", "S01"]


def test_moss_relink_without_the_voiceprint_model_makes_everyone_new(source, tmp_path):
    answers = {"s-0.0-600.0": moss_answer([(0, 100, "あ", "S01"), (200, 590, "い", "S01")]),
               "s-100.0-200.0": moss_answer([(10, 50, "う", "S01")])}
    t, _ = transcribe_moss(source, tmp_path, answers, 600.0, voiced=[(110, 150)])
    assert [u.speaker for u in t.units()] == ["S01", "c1:S01", "S01"]


def test_voiceprint_model_failure_is_said_once_per_transcript(monkeypatch):
    from finesub.speech.joint import relink

    loads, warned = [], []

    def broken(device):
        loads.append(1)
        raise RuntimeError("offline")

    monkeypatch.setattr(relink, "_wespeaker", broken)
    monkeypatch.setattr(relink, "current_reporter", lambda: type("R", (), {"warning": lambda self, *a, **k: warned.append(a)})())
    prints = relink.Voiceprints()
    for _ in range(2):  # two transcripts
        segs = [{"start": 0.0, "end": 5.0, "speaker": "S01"}, {"start": 6.0, "end": 9.0, "speaker": "c1:S01"}]
        assert relink.merge_by_voice(segs, lambda: None, prints, "MOSS Pro") == ([], [])
    assert len(loads) == 1 and len(warned) == 2 and "offline" in warned[0][1]



def test_a_gpu_voiceprint_model_that_cannot_load_falls_back_to_the_cpu_one(monkeypatch):
    from finesub.speech.joint import relink

    tried, warned = [], []

    def wespeaker(device):
        tried.append(device)
        if device != "cpu":
            raise RuntimeError("no cuda build here")
        return "cpu-embed"

    monkeypatch.setattr(relink, "_wespeaker", wespeaker)
    monkeypatch.setattr(relink, "current_reporter", lambda: type("R", (), {"warning": lambda self, *a, **k: warned.append(a)})())
    prints = relink.Voiceprints("cuda")
    assert prints.embed() == "cpu-embed" and prints.embed() == "cpu-embed"
    assert tried == ["cuda", "cpu"] and len(warned) == 1 and "no cuda build here" in warned[0][1]


def test_released_voiceprints_load_again_on_next_use(monkeypatch):
    from finesub.speech.joint import relink

    loads = []
    monkeypatch.setattr(relink, "_wespeaker", lambda device: loads.append(device) or object())
    prints = relink.Voiceprints("cuda")
    first = prints.embed()
    prints.release()
    assert prints.embed() is not first and loads == ["cuda", "cuda"]


def test_the_voice_merge_holds_no_model_once_it_is_released(monkeypatch):
    # local MOSS releases its GPU voiceprint model before the GPU lock goes: nothing the merge leaves may hold it
    import gc
    import weakref

    from finesub.speech.joint import relink

    models = []

    def wespeaker(device):
        def embed(piece, mask):
            return np.ones(4) / 2

        models.append(weakref.ref(embed))
        return embed

    monkeypatch.setattr(relink, "_wespeaker", wespeaker)
    prints = relink.Voiceprints("cuda")
    segs = [{"start": 0.0, "end": 5.0, "speaker": "S01"}, {"start": 6.0, "end": 11.0, "speaker": "S02"}]
    merged = relink.merge_by_voice(segs, lambda: np.zeros(16000 * 30, np.int16), prints, "test")
    assert merged == ([("S02", "S01", 1.0)], [])
    prints.release()
    gc.collect()
    assert models[0]() is None


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
    # Vercel caches the Azure part of its answer (`vercel_answer`), so it is the azure shape here.
    for first, answer, second in (("azure", azure, "openrouter"), ("openrouter", openrouter, "azure"),
                                  ("vercel", azure, "openrouter"), ("openrouter", openrouter, "vercel")):
        root = tmp_path / f"{first}-{second}"
        asked = mai2.transcribe(source, cache_root=root, language="en", via=first, repair="off", caller=lambda clip, locale: answer,
                                duration=600.0)
        replayed = mai2.transcribe(source, cache_root=root, language="en", via=second, repair="off", offline=True, duration=600.0)
        assert [(u.text, u.speaker) for u in replayed.units()] == [(u.text, u.speaker) for u in asked.units()] == [("hi", "S1")]


@pytest.mark.parametrize("other", ["openrouter", "azure"])
def test_mai2_routes_share_every_window_of_a_long_source(source, tmp_path, other):
    azure = {"phrases": [{"speaker": 0, "words": [{"text": "hi", "offsetMilliseconds": 1000, "durationMilliseconds": 200}]}]}
    asked = mai2.transcribe(source, cache_root=tmp_path, language="en", via="vercel", repair="off",
                            caller=lambda clip, locale: azure, duration=10000.0, speech=silent(source))
    assert len(list((tmp_path / joint.MAI2).glob("w-*.raw.json"))) == 2
    replayed = mai2.transcribe(source, cache_root=tmp_path, language="en", via=other, repair="off", offline=True,
                               duration=10000.0, speech=silent(source))
    assert [(u.start, u.text) for u in replayed.units()] == [(u.start, u.text) for u in asked.units()]


def test_mai2_keeps_the_window_its_cached_answers_were_cut_with(source, tmp_path, monkeypatch):
    # Answers cached under an older WINDOW are the task's own: a rerun reads them all back and asks nothing.
    azure = {"phrases": [{"speaker": 0, "words": [{"text": "hi", "offsetMilliseconds": 1000, "durationMilliseconds": 200}]}]}
    monkeypatch.setattr(mai2, "WINDOW", 1800.0)
    asked = mai2.transcribe(source, cache_root=tmp_path, language="en", via="openrouter", repair="off",
                            caller=lambda clip, locale: azure, duration=4000.0, speech=silent(source))
    monkeypatch.undo()

    def refuse(clip, locale):
        raise AssertionError("asked again")

    again = mai2.transcribe(source, cache_root=tmp_path, language="en", via="vercel", repair="off", caller=refuse,
                            duration=4000.0, speech=silent(source))
    assert again.log == asked.log == [{"windows": [[0.0, 1800.0], [1740.0, 3540.0], [3480.0, 4000.0]]}]
    assert [(u.start, u.text) for u in again.units()] == [(u.start, u.text) for u in asked.units()]


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
                            duration=9000.0, speech=silent(source))
    finally:
        mp.undo()
    texts = [u.text for u in t.units()]
    assert len(texts) == len(set(texts)) and texts == sorted(texts, key=lambda s: int(s[1:]))
    assert t.log == [{"windows": [[0.0, 7140.0], [7080.0, 9000.0]]}]


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


# The gateway's answer as the 2026-11-19 smoke test got it, trimmed to one phrase.
VERCEL_ANSWER = {
    "text": "The stale smell.", "segments": [{"text": "The stale smell.", "startSecond": 1.36, "endSecond": 2.28}],
    "language": "en", "durationInSeconds": 12, "warnings": [],
    "providerMetadata": {
        "azure": {"phrases": [{"text": "The stale smell.", "offsetMilliseconds": 1360, "durationMilliseconds": 920,
                               "locale": "en", "speaker": 1, "confidence": 0,
                               "words": [{"text": "The", "offsetMilliseconds": 1360, "durationMilliseconds": 119},
                                         {"text": "stale", "offsetMilliseconds": 1640, "durationMilliseconds": 260},
                                         {"text": "smell.", "offsetMilliseconds": 2040, "durationMilliseconds": 240}]}]},
        "gateway": {"cost": "0.00033336"}}}


def test_mai2_vercel_sends_the_azure_options_and_returns_the_azure_answer(tmp_path, monkeypatch):
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    sent = []

    class Gateway(FakeClient):
        def post(self, url, headers=None, json=None, **_):
            sent.append((url, headers, json))
            return FakeResponse(200, VERCEL_ANSWER)

    monkeypatch.setattr(mai2, "_client", lambda: Gateway([]))
    answer = mai2.Caller("vercel", "vck-key")(audio, locale="ja")
    url, headers, body = sent[0]
    assert url == mai2.VERCEL_API
    assert headers == {"Authorization": "Bearer vck-key", "ai-gateway-protocol-version": "0.0.1",
                       "ai-transcription-model-specification-version": "4", "ai-model-id": "microsoft/mai-transcribe-2"}
    assert body["mediaType"] == "audio/mpeg" and body["audio"] == "bXAz"
    assert body["providerOptions"] == {"azure": {"timestamps": "word", "diarization": {"enabled": True},
                                                 "locales": ["ja"]}}
    # What is returned (and cached) is the Azure route's shape, so either route replays the other's windows.
    assert answer == VERCEL_ANSWER["providerMetadata"]["azure"]
    words = mai2.answer_words(answer, 100.0)
    assert [w["text"] for w in words] == ["The", "stale", "smell."]
    assert words[0] == {"start": 101.36, "end": pytest.approx(101.479), "speaker": "S1", "text": "The"}


def test_mai2_vercel_answer_without_azure_phrases_is_an_error(tmp_path, monkeypatch):
    # Without the word-level part (e.g. the options dropped) there is nothing to place; it must not be cached.
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"mp3")
    monkeypatch.setattr(mai2, "_client", lambda: FakeClient([]))
    caller = mai2.Caller("vercel", "k", sleep=lambda s: pytest.fail("waited"))
    no_words = {"providerMetadata": {"azure": {"phrases": [{"text": "hello", "offsetMilliseconds": 0}]}}}
    for answer, error in (({"text": "x", "segments": []}, "Azure phrases"), (no_words, "without word times")):
        object.__setattr__(caller, "_once", lambda audio, locale, answer=answer: FakeResponse(200, answer))
        with pytest.raises(mai2.Mai2Error, match=error):
            caller(audio, locale=None)
    # Silence is a real answer and passes.
    silent = {"providerMetadata": {"azure": {"phrases": []}}}
    object.__setattr__(caller, "_once", lambda audio, locale: FakeResponse(200, silent))
    assert caller(audio, locale=None) == {"phrases": []}


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
