"""The ElevenLabs key pool (`speech/joint/scribe_keys.py`): which key a Scribe call goes on, what freezes a key, how
its credits-per-hour rate is learned, and that a thaw from elsewhere holds at once."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from finesub import key_freeze
from finesub.api_keys import ApiKeyEntry
from finesub.speech.joint import scribe_keys as K
from finesub.speech.joint.scribe import ScribeError, ScribeRefused

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
RESET = datetime(2026, 12, 11, 11, 34, tzinfo=timezone.utc)
HOUR = 3600.0


class Account:
    """Balances by key and a fake Scribe that bills `rate` credits an hour to the key it is called on."""

    def __init__(self, **balances):
        self.balances = dict(balances)  # key -> Balance, or "no_permission"
        self.rates = {}
        self.refuse: set[str] = set()
        self.calls: list[str] = []
        self.fetches: list[str] = []

    def fetch(self, key):
        self.fetches.append(key)
        b = self.balances[key]
        return (b, None) if isinstance(b, str) else ("ok", b)

    def caller(self, key):
        def call(audio, *, language):
            self.calls.append(key)
            if key in self.refuse:
                raise ScribeRefused(401, '{"detail": {"status": "quota_exceeded"}}')
            b = self.balances[key]
            if not isinstance(b, str) and key in self.rates:
                self.balances[key] = K.Balance(b.tier, b.used + self.rates[key], b.limit, b.reset)
            return {"words": [], "key": key}

        return call


def free(used=0, *, reset=RESET):
    return K.Balance("free", used, 10000, reset)


def pool(account, names, tmp_path, monkeypatch):
    monkeypatch.setattr(K, "_REFUSED", set())
    monkeypatch.setattr(K, "_WARNED", set())
    account.warned = []
    monkeypatch.setattr(K, "current_reporter", lambda: type("R", (), {
        "warning": lambda self, code, message, **k: account.warned.append(message)})())
    entries = [ApiKeyEntry(n, n) for n in names]
    return K.KeyPool(entries, fetch=account.fetch, make_caller=account.caller, state_path=tmp_path / ".state",
                     clock=lambda: NOW)


def ask(p, seconds=HOUR, whole=True):
    return p(Path("a.flac"), language="ja", seconds=seconds, whole=whole)["key"]


def frozen(tmp_path):
    return {f.row: f for f in key_freeze.listing(now=NOW, state_path=tmp_path / ".state")}


def test_a_key_short_of_the_source_is_passed_over_not_frozen(tmp_path, monkeypatch, capsys):
    account = Account(a=free(used=9000), b=free())  # a: 1000 credits, about 27 min at 2222/h
    p = pool(account, ["a", "b", "c"], tmp_path, monkeypatch)
    assert ask(p) == "b" and not frozen(tmp_path)
    assert ask(p, seconds=600) == "a"  # ten minutes still fit on a
    err = capsys.readouterr().err
    assert "Note: Scribe: 1.00 h on key b's free credits" in err and "$" not in err


def test_a_spent_key_is_frozen_until_its_refresh_and_not_asked_again(tmp_path, monkeypatch):
    account = Account(a=free(used=9900), b=free())  # 100 credits: under 5 minutes
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    assert ask(p, seconds=60) == "b"
    row = key_freeze.row_key(K.FAMILY, "scribe_v3", "a")
    assert frozen(tmp_path)[row].until == RESET
    account.fetches.clear()
    assert ask(p, seconds=60) == "b" and "a" not in account.fetches

    # thawed by another process: the next decision reads it, and the key is asked again
    key_freeze.thaw([row], state_path=tmp_path / ".state")
    account.balances["a"] = free()
    assert ask(p, seconds=60) == "a"


@pytest.mark.parametrize("reset", [None, NOW - timedelta(days=1)], ids=["no-refresh-time", "refresh-time-past"])
def test_a_spent_key_without_a_refresh_time_ahead_is_frozen_a_fixed_while(tmp_path, monkeypatch, reset):
    account = Account(a=free(used=9990, reset=reset), b=free())
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    assert ask(p, seconds=60) == "b"
    (held,) = frozen(tmp_path).values()
    assert held.until == NOW + K.FREEZE_WITHOUT_RESET


def test_the_last_key_uploads_when_its_free_credits_are_short_and_is_never_frozen(tmp_path, monkeypatch, capsys):
    account = Account(a=free(used=9990), b=free(used=9990))
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    assert ask(p) == "b"
    assert [f.row.rsplit("/", 1)[1] for f in frozen(tmp_path).values()] == ["a"]
    assert "the pool's last, has free credits for about 0.0 h against 1.00 h of audio; uploading anyway" in \
        capsys.readouterr().err


def test_the_last_key_refused_stops_with_its_name_and_refresh_and_freezes_nothing(tmp_path, monkeypatch):
    account = Account(a=free(used=9000), b=free())
    account.refuse.add("b")
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    with pytest.raises(ScribeError, match=r"a: about 0\.45 h left.*b refused \(HTTP 401\).*refresh 2026-12-11 11:34"):
        ask(p)
    assert not frozen(tmp_path)


def test_a_key_refused_although_it_showed_room_is_left_out_for_the_run_not_frozen(tmp_path, monkeypatch):
    account = Account(a=free(), b=free())
    account.refuse.add("a")  # another process spent it in between
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    assert ask(p) == "b" and not frozen(tmp_path)
    assert account.warned[0].startswith("Scribe: key a refused (HTTP 401), left out for this run although its "
                                        "balance showed room: check that it has the Speech to Text")
    account.refuse.clear()
    account.calls.clear()
    assert ask(p) == "b" and account.calls == ["b"]  # a batch does not upload to it first every time
    assert len(account.warned) == 1


def test_a_key_that_cannot_read_its_balance_is_used_until_refused_then_left_out_for_the_run(tmp_path, monkeypatch):
    account = Account(a="no_permission", b=free())
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    assert ask(p) == "a"
    account.refuse.add("a")
    assert ask(p) == "b" and not frozen(tmp_path)
    account.calls.clear()
    assert ask(p) == "b" and account.calls == ["b"]
    account.refuse.add("b")
    with pytest.raises(ScribeError, match="a: refused earlier in this run"):
        ask(p)
    assert "User: Read" in account.warned[0] and len(account.warned) == 2  # once per key


def test_a_call_that_waited_for_the_key_sees_its_refusal(tmp_path, monkeypatch):
    account = Account(a=free(), b=free())
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)

    class Waited:  # the call before this one held the key and was refused meanwhile
        def __enter__(self):
            K._REFUSED.add("a")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(K, "_lock", lambda key_id: Waited() if key_id == "a" else K.threading.Lock())
    assert ask(p) == "b" and "a" not in account.fetches + account.calls


def test_a_key_whose_balance_could_not_be_read_is_left_out_once_refused(tmp_path, monkeypatch):
    account = Account(a="unknown", b=free())  # a revoked key: the subscription call fails as well
    account.refuse.add("a")
    p = pool(account, ["a", "b"], tmp_path, monkeypatch)
    assert ask(p) == "b"
    account.calls.clear()
    assert ask(p) == "b" and account.calls == ["b"]
    assert "check the key" in account.warned[0]


def test_an_unknown_balance_prints_the_pay_as_you_go_estimate(tmp_path, monkeypatch, capsys):
    p = pool(Account(a="unknown"), ["a"], tmp_path, monkeypatch)
    p.note = ", plus a re-ask"
    ask(p, seconds=1800)
    assert "Warning: Scribe bills per audio hour: 0.50 h on key a, about $0.10 at $0.20/h (2026-11 price), plus a " \
           "re-ask" in capsys.readouterr().err
    ask(p, seconds=30, whole=False)
    assert capsys.readouterr().err == ""  # a re-ask says nothing


def rate_of(p, name, balance):
    return p.rate(ApiKeyEntry(name, name), balance)


@pytest.mark.parametrize("billed, seconds, learned", [
    (900, HOUR, 900.0),  # the starter plan's measured 1000/h, within the clamp of 1100
    (5000, HOUR, 2200.0),  # clamped at twice the start
    (0, HOUR, None),  # the balance had not moved yet: nothing to learn
    (-50, HOUR, None),  # refreshed in between
    (100, 120.0, None),  # a short re-ask: whole credits, too coarse
])
def test_the_rate_is_learned_from_a_long_calls_rise_only_and_clamped(tmp_path, monkeypatch, billed, seconds, learned):
    plan = K.Balance("starter", 10630, 40000, RESET)
    account = Account(a=plan)
    account.rates["a"] = billed
    p = pool(account, ["a"], tmp_path, monkeypatch)
    ask(p, seconds=seconds)
    assert rate_of(p, "a", plan) == (learned if learned is not None else K.OTHER_RATE)


def test_a_new_plan_starts_its_rate_over(tmp_path, monkeypatch):
    account = Account(a=K.Balance("starter", 0, 40000, RESET))
    account.rates["a"] = 1000
    p = pool(account, ["a"], tmp_path, monkeypatch)
    ask(p)
    assert rate_of(p, "a", account.balances["a"]) == 1000.0
    assert rate_of(p, "a", K.Balance("creator", 0, 100000, RESET)) == K.OTHER_RATE
    assert rate_of(p, "a", free()) == K.FREE_RATE


def test_a_freeze_holds_until_its_time_and_a_thaw_clears_it(tmp_path):
    path = tmp_path / ".state"
    key_freeze.freeze("elevenlabs/scribe_v3/a", NOW + timedelta(hours=1), "spent", now=NOW, state_path=path)
    assert key_freeze.frozen("elevenlabs/scribe_v3/a", now=NOW, state_path=path).reason == "spent"
    assert key_freeze.frozen("elevenlabs/scribe_v3/a", now=NOW + timedelta(hours=2), state_path=path) is None
    assert key_freeze.thaw(["elevenlabs/scribe_v3/a", "x"], state_path=path) == ["elevenlabs/scribe_v3/a"]
    assert key_freeze.listing(now=NOW, state_path=path) == []
