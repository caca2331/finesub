"""`finesub quota` (`finesub/quota_command.py`): what it lists, and that `thaw` clears a freeze or a spent flag
but not MOSS Pro's hours; the LLM side's three freezes (Gemini's day, an agent subscription, a search key) are rows of
the same ledger, read on every decision, so a thaw holds at once and is never written back."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from finesub import key_freeze, quota_command
from finesub import state as state_store
from finesub.speech.joint import quota


def seed(now: datetime) -> None:
    key_freeze.freeze("elevenlabs/scribe_v3/sp", now + timedelta(days=20), "100 credits left, about 3 min", now=now)
    with state_store.state_section(quota.NAMESPACE) as section:
        section["moss-pro/m/a"] = {"day": quota.quota_day(quota.BEIJING, now), "calls": 3, "seconds": 3672.0,
                                   "exhausted": True}
        section["gemini-35t/t/main"] = {"day": quota.quota_day(quota.PACIFIC, now), "calls": 25, "seconds": 0.0,
                                        "exhausted": True}
        section["gemini-35t/t/old"] = {"day": "2020-01-01", "calls": 25, "seconds": 0.0, "exhausted": True}
        section["elevenlabs/scribe_v3/sp"] = {"tier": "starter", "rate": 1000.0}  # the pool's rate: not an account row


def test_quota_lists_the_frozen_and_todays_accounts_and_thaws_by_name_or_prefix(capsys):
    now = datetime.now(timezone.utc)
    seed(now)
    assert quota_command.main([]) == 0
    listed = capsys.readouterr().out
    assert "elevenlabs/scribe_v3/sp  until" in listed and "100 credits left" in listed
    assert "moss-pro/m/a  3 calls, 1.02 h of audio, spent" in listed and "gemini-35t/t/main  25 calls" in listed
    assert "old" not in listed and "1000" not in listed and listed.count("elevenlabs/scribe_v3/sp") == 1

    assert quota_command.main(["thaw", "moss-pro", "elevenlabs"]) == 0
    assert capsys.readouterr().out.splitlines() == ["Thawed elevenlabs/scribe_v3/sp", "Thawed moss-pro/m/a"]
    account = state_store.read_section(quota.NAMESPACE)
    assert account["moss-pro/m/a"] == {"day": quota.quota_day(quota.BEIJING, now), "calls": 3, "seconds": 3672.0,
                                       "exhausted": False}  # the hours stay counted
    assert key_freeze.listing() == []

    assert quota_command.main(["thaw", "nothing-like-it"]) == 1
    assert "nothing-like-it" in capsys.readouterr().err
    assert quota_command.main(["thaw", "--all"]) == 0
    assert capsys.readouterr().out.splitlines() == ["Thawed gemini-35t/t/main"]
    assert quota_command.main([]) == 0
    assert "spent" not in capsys.readouterr().out


def test_a_thaw_reaches_a_ledger_that_reads_again_and_is_not_written_back():
    """The 35t / MOSS Pro ledger reads its rows inside the lock of every reservation, so a flag cleared by another
    process holds at the next one; a later write by the ledger keeps the cleared flag."""
    now = datetime.now(timezone.utc)
    seed(now)
    ledger = quota.KeyLedger("moss-pro", "m", quota.BEIJING)
    from finesub.api_keys import ApiKeyEntry

    a = ApiKeyEntry("a", "k")
    assert ledger.usage(a)["exhausted"]
    quota_command.thaw(["moss-pro/m/a"], every=False)
    entry, _day = ledger.reserve([a], fits=lambda e, u: True, seconds=60.0, what="t")
    assert entry is a and ledger.usage(a) == {"calls": 4, "seconds": 3732.0, "exhausted": False}


def test_the_llm_freezes_are_listed_thawed_and_not_written_back(tmp_path):
    from finesub.llm.agent.agent_quota import AgentQuotaLedger
    from finesub.llm.rate_limit import ModelRateLimiter, _next_pacific_midnight, daily_freeze_row
    from finesub.llm.routing.config import ModelEndpoint
    from finesub.llm.web_search import ApiKeyPool, SearchApiKey

    limiter = ModelRateLimiter(enabled=True)
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini-3.6-flash")
    agents = AgentQuotaLedger()
    search = ApiKeyPool("exa", [SearchApiKey("e1", "secret")], lock_seconds=3600)
    limiter.mark_daily_exhausted(endpoint, key_id="k1")
    agents.freeze("LOCAL_CODEX", seconds=7200, reason="out of\nquota")
    search.lock(search.entries[0], "HTTP 429")

    rows = {f.row: f for f in key_freeze.listing()}
    assert set(rows) == {"GEMINI_FREE/gemini-3.6-flash/k1", "agent/LOCAL_CODEX", "search/exa/e1"}
    assert rows["GEMINI_FREE/gemini-3.6-flash/k1"].until == _next_pacific_midnight(rows["GEMINI_FREE/gemini-3.6-flash/k1"].since)
    assert rows["agent/LOCAL_CODEX"].reason == "out of quota" and search.available_entries() == []

    back, _ = quota_command.thaw(["GEMINI_FREE", "agent", "search/exa"], every=False)
    assert sorted(back) == sorted(rows)
    assert not limiter.is_daily_exhausted(endpoint, key_id="k1") and not agents.is_frozen("LOCAL_CODEX")
    assert [e.key_id for e in search.available_entries()] == ["e1"]

    # each ledger writing again keeps what was thawed elsewhere
    limiter.note_daily_quota_hit(ModelEndpoint("GEMINI_FREE", "other"), key_id="k1")
    agents.note_failure("LOCAL_CODEX")
    search.mark_used(search.entries[0])
    assert key_freeze.listing() == [] and daily_freeze_row(endpoint, "k1") not in str(state_store.read_section("llm_rate_limit"))


UTC = timezone.utc


def test_a_gemini_day_lock_lasts_to_the_next_pacific_midnight_across_dst(tmp_path):
    from finesub.llm.rate_limit import ModelRateLimiter
    from finesub.llm.routing.config import ModelEndpoint

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini-3.6-flash")
    cases = [  # (marked at, UTC; held until, UTC): PST is UTC-8, PDT UTC-7
        (datetime(2026, 3, 8, 4, 0, tzinfo=UTC), datetime(2026, 3, 8, 8, 0, tzinfo=UTC)),  # Sat 20:00 PST
        (datetime(2026, 3, 8, 19, 0, tzinfo=UTC), datetime(2026, 3, 9, 7, 0, tzinfo=UTC)),  # spring forward day
        (datetime(2026, 10, 31, 19, 0, tzinfo=UTC), datetime(2026, 11, 1, 7, 0, tzinfo=UTC)),
        (datetime(2026, 11, 1, 20, 0, tzinfo=UTC), datetime(2026, 11, 2, 8, 0, tzinfo=UTC)),  # fall back day
        (datetime(2026, 11, 12, 7, 59, tzinfo=UTC), datetime(2026, 11, 12, 8, 0, tzinfo=UTC)),  # a minute before
        (datetime(2026, 11, 12, 8, 1), datetime(2026, 11, 13, 8, 0, tzinfo=UTC)),  # naive: taken as UTC
    ]
    for i, (marked, until) in enumerate(cases):
        key = f"k{i}"
        limiter.mark_daily_exhausted(endpoint, key_id=key, now=marked)
        assert limiter.is_daily_exhausted(endpoint, key_id=key, now=until - timedelta(seconds=1)), marked
        assert not limiter.is_daily_exhausted(endpoint, key_id=key, now=until), marked


def test_an_endpoint_level_day_lock_is_listed_and_thawed(tmp_path):
    from finesub.llm.rate_limit import ModelRateLimiter
    from finesub.llm.routing.config import ModelEndpoint

    path = tmp_path / ".state"
    limiter = ModelRateLimiter(state_path=path, enabled=True)
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini-3.6-flash")
    limiter.mark_daily_exhausted(endpoint)
    assert [f.row for f in key_freeze.listing(state_path=path)] == ["GEMINI_FREE/gemini-3.6-flash/"]
    assert not limiter.is_daily_exhausted(endpoint, key_id="k1")  # the endpoint's lock is not a key's
    key_freeze.thaw(["GEMINI_FREE/gemini-3.6-flash/"], state_path=path)
    assert not limiter.is_daily_exhausted(endpoint)


def test_the_search_pool_skips_frozen_keys_until_the_boundary_and_drops_the_old_lock_fields(tmp_path):
    from finesub.llm.web_search import ApiKeyPool, SearchApiKey

    path = tmp_path / ".state"
    clock = [1_800_000_000.0]
    with state_store.state_section("exa", path) as section:  # a lock in the old shape: ignored, then dropped
        section["keys"] = {"e1": {"last_used_at": 5.0, "locked_until": clock[0] + 999, "lock_reason": "old"}}
    pool = ApiKeyPool("exa", [SearchApiKey(k, k) for k in ("e1", "e2", "e3")], state_path=path, lock_seconds=60,
                      now_func=lambda: clock[0])
    pool.lock(pool.entries[1], "HTTP 402")
    assert [e.key_id for e in pool.available_entries()] == ["e1", "e3"] and pool.locked_key_ids() == ["e2"]
    pool.mark_used(pool.entries[2])
    assert [e.key_id for e in pool.available_entries()] == ["e3", "e1"]  # the most recently successful first
    assert state_store.read_section("exa", path)["keys"]["e1"] == {"last_used_at": 5.0}
    clock[0] += 59
    assert pool.locked_key_ids() == ["e2"]
    clock[0] += 1
    assert pool.locked_key_ids() == [] and len(pool.available_entries()) == 3
