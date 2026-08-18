from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from finesub.llm.agent import agent_quota
from finesub.llm.agent.agent_quota import AgentQuotaLedger
from finesub.llm.agent.local_agent import (
    LocalAgentQuotaError,
    LocalAgentTransientError,
    LocalAgentUnavailableError,
)


@pytest.fixture
def ledger(tmp_path):
    return AgentQuotaLedger(tmp_path / ".state")


def test_one_failure_is_noise_and_two_is_a_question(ledger) -> None:
    """Probing on every failure would double the cost of every network blip.

    Waiting for the second consecutive failure on the same subscription also
    means the vendor's exact wording never has to be guessed -- and guessing is
    the expensive direction, since a freeze takes a working plan out for hours.
    """

    assert ledger.note_failure("LOCAL_CODEX") is False
    assert ledger.note_failure("LOCAL_CODEX") is True

    # A success in between resets the count: this is a *consecutive* streak.
    ledger.note_success("LOCAL_CODEX")
    assert ledger.note_failure("LOCAL_CODEX") is False


def test_the_vendors_wording_is_never_consulted(ledger) -> None:
    """Matching phrases like "usage limit" would only save one call.

    Against that: there is no way to enumerate what a vendor may put in an
    error field, and a false match takes a working subscription out for hours.
    The probe answers the same question with no guessing at all.
    """

    assert ledger.note_failure("LOCAL_CLAUDE") is False
    assert not hasattr(agent_quota, "looks_like_quota")


def test_failure_streaks_are_counted_per_subscription(ledger) -> None:
    assert ledger.note_failure("LOCAL_CODEX") is False
    assert ledger.note_failure("LOCAL_CLAUDE") is False
    assert ledger.note_failure("local_codex") is True


def test_a_freeze_survives_the_process_and_expires_on_time(tmp_path) -> None:
    """Another FineSub must not re-burn a subscription this one found spent."""

    first = AgentQuotaLedger(tmp_path / ".state")
    first.freeze("LOCAL_CODEX", seconds=agent_quota.QUOTA_FREEZE_SECONDS)

    second = AgentQuotaLedger(tmp_path / ".state")
    assert second.is_frozen("LOCAL_CODEX") is True
    assert second.is_frozen("LOCAL_CLAUDE") is False

    past = datetime.now(timezone.utc) + timedelta(hours=6)
    assert second.frozen_until("LOCAL_CODEX", now=past) is None


def test_a_working_call_clears_the_freeze(ledger) -> None:
    ledger.freeze("LOCAL_CODEX", seconds=agent_quota.QUOTA_FREEZE_SECONDS)
    assert ledger.is_frozen("LOCAL_CODEX") is True

    ledger.note_success("LOCAL_CODEX")

    assert ledger.is_frozen("LOCAL_CODEX") is False
    assert AgentQuotaLedger(ledger.state_path).is_frozen("LOCAL_CODEX") is False


def test_a_probe_that_answers_means_it_was_not_the_quota(ledger) -> None:
    ledger.note_failure("LOCAL_CODEX")
    exc = LocalAgentTransientError("Codex CLI exited with status 1")

    returned = agent_quota.evaluate_agent_failure(
        pool="LOCAL_CODEX", exc=exc, ping=lambda: None, ledger=ledger
    )

    assert returned is exc
    assert ledger.is_frozen("LOCAL_CODEX") is False


def test_a_probe_that_fails_too_freezes_the_whole_pool(ledger) -> None:
    ledger.note_failure("LOCAL_CODEX")
    warnings: list[str] = []

    def refuse() -> None:
        raise LocalAgentTransientError("still refusing")

    returned = agent_quota.evaluate_agent_failure(
        pool="LOCAL_CODEX",
        exc=LocalAgentTransientError("Codex CLI exited with status 1"),
        ping=refuse,
        ledger=ledger,
        warn=warnings.append,
    )

    assert isinstance(returned, LocalAgentQuotaError)
    assert returned.route_failure_kind == "quota"
    assert ledger.is_frozen("LOCAL_CODEX") is True
    # The probe reuses this target's own model and config, so it cannot rule
    # out "this target can never work" -- a model name the CLI does not serve
    # fails exactly like a spent plan. Saying only "check your usage" sent
    # people looking in the one place that was fine.
    assert warnings and "logged in" in warnings[0]
    assert "cannot work at all" in warnings[0]
    # And it quotes what actually failed, so the guess can be checked. With no
    # retained capsule to read the vendor's own words from, that is the
    # original exception rather than nothing at all.
    assert "Codex CLI exited with status 1" in warnings[0]


def test_an_expired_login_is_never_mistaken_for_an_empty_wallet(ledger) -> None:
    """The two need opposite responses, and a freeze would hide the real one."""

    ledger.note_failure("LOCAL_CLAUDE")

    def not_logged_in() -> None:
        raise LocalAgentUnavailableError("Claude Code is not authenticated")

    returned = agent_quota.evaluate_agent_failure(
        pool="LOCAL_CLAUDE",
        exc=LocalAgentTransientError("Claude Code exited with status 1"),
        ping=not_logged_in,
        ledger=ledger,
    )

    assert not isinstance(returned, LocalAgentQuotaError)
    assert ledger.is_frozen("LOCAL_CLAUDE") is False


def test_only_transient_failures_are_ever_probed(ledger) -> None:
    """Everything else already has a more specific answer."""

    exc = LocalAgentUnavailableError("no CLI")
    assert (
        agent_quota.evaluate_agent_failure(
            pool="LOCAL_CODEX",
            exc=exc,
            ping=lambda: (_ for _ in ()).throw(AssertionError("must not probe")),
            ledger=ledger,
        )
        is exc
    )
    assert ledger.is_frozen("LOCAL_CODEX") is False


def test_a_quota_error_from_anywhere_is_still_recorded(ledger) -> None:
    """A driver may conclude this on its own, and the ledger still has to hear.

    Otherwise the next candidate on the same subscription gets launched to be
    told the same thing."""

    exc = LocalAgentQuotaError("Codex reports the subscription is out of quota")
    warnings: list[str] = []

    returned = agent_quota.evaluate_agent_failure(
        pool="LOCAL_CODEX",
        exc=exc,
        ping=lambda: (_ for _ in ()).throw(AssertionError("must not probe")),
        ledger=ledger,
        warn=warnings.append,
    )

    assert returned is exc
    assert ledger.is_frozen("LOCAL_CODEX") is True
    assert warnings


def test_a_frozen_pool_disqualifies_every_target_that_shares_it(
    tmp_path, monkeypatch
) -> None:
    """The reason exhaustion is booked per pool rather than per model.

    Without this the chain walked from a spent Codex model straight into its
    sibling on the same plan, paying a full CLI launch to be told the same.
    """

    from finesub.llm.routing.model_router import provider_enabled

    book = AgentQuotaLedger(tmp_path / ".state")
    monkeypatch.setattr(agent_quota, "default_ledger", lambda: book)

    def _candidate(tier: str, pool: str = ""):
        class _Endpoint:
            backend = "local_agent"
            provider_tier = tier
            api_model_id = "gpt-5.6-sol"

        class _Fact:
            effective_quota_pool = pool or tier

        class _Candidate:
            endpoint = _Endpoint()
            fact = _Fact()

        return _Candidate()

    codex = _candidate("LOCAL_CODEX")
    assert provider_enabled(codex, agent_ready=lambda *_: True) is True
    book.freeze("LOCAL_CODEX", seconds=agent_quota.QUOTA_FREEZE_SECONDS)
    assert provider_enabled(codex, agent_ready=lambda *_: True) is False


def test_one_tier_can_meter_two_allowances_separately(tmp_path, monkeypatch) -> None:
    """Antigravity fronts Gemini and Opus on separate allowances.

    Freezing them together would take a working model out of service because
    its neighbour ran dry, which is why the pool is a catalog column rather
    than the provider tier itself.
    """

    from finesub.llm.routing.model_router import provider_enabled
    from finesub.llm.routing.model_routes import default_model_routes

    book = AgentQuotaLedger(tmp_path / ".state")
    monkeypatch.setattr(agent_quota, "default_ledger", lambda: book)
    routes = default_model_routes()

    def _candidate(target_id: str):
        target_fact = routes.target_fact(target_id)

        class _Endpoint:
            backend = "local_agent"
            provider_tier = target_fact.provider_tier
            api_model_id = target_fact.api_model_id

        class _Candidate:
            endpoint = _Endpoint()
            fact = target_fact

        return _Candidate()

    opus = _candidate("local-agy-opus-4_6")
    gemini = _candidate("local-agy-media-gemini-3_7-flash")
    assert opus.fact.effective_quota_pool != gemini.fact.effective_quota_pool

    book.freeze(opus.fact.effective_quota_pool, seconds=agent_quota.QUOTA_FREEZE_SECONDS)
    assert provider_enabled(opus, agent_ready=lambda *_: True) is False
    assert provider_enabled(gemini, agent_ready=lambda *_: True) is True


def test_agent_ping_reports_each_tier_and_its_advice(tmp_path, monkeypatch, capsys) -> None:
    """The one thing a person could not ask before: is it me, my login, or my quota?

    None of the three CLIs exposes a usage query, so a real tiny call is the
    only honest answer, and it has to be reachable without starting a run.
    """

    from finesub.llm.agent import agent_ping, agent_quota
    from finesub.llm.agent.local_agent import LocalAgentQuotaError, LocalAgentUnavailableError

    book = agent_quota.AgentQuotaLedger(tmp_path / ".state")
    monkeypatch.setattr(agent_quota, "default_ledger", lambda: book)

    class _Probe:
        available = True
        version = "1.2.3"
        error = ""

    class _Driver:
        driver_id = "stub"

        def __init__(self, outcome):
            self.outcome = outcome

        def probe(self):
            return _Probe()

        def meets_requirements(self, probe=None, **_):
            return True

        def run(self, messages, **kwargs):
            assert kwargs["task"] == "quota-probe"
            if self.outcome == "ok":
                return type("R", (), {"content": "OK"})()
            if self.outcome == "spent":
                raise LocalAgentQuotaError("out of quota")
            raise LocalAgentUnavailableError("not logged in")

    outcomes = {"LOCAL_CODEX": "ok", "LOCAL_CLAUDE": "auth", "LOCAL_AGY": "spent"}
    monkeypatch.setattr(
        agent_ping,
        "_tier_rows",
        lambda _routes: [
            ("LOCAL_CODEX", "gpt-x", "LOCAL_CODEX"),
            ("LOCAL_CLAUDE", "claude-x", "LOCAL_CLAUDE"),
            # agy meters its two models apart, so the row carries the pool.
            ("LOCAL_AGY", "gemini-x", "AGY_GEMINI"),
        ],
    )
    monkeypatch.setattr(
        "finesub.llm.routing.execution_policy.driver_for_provider_tier",
        lambda _settings, *, provider_tier, model: _Driver(outcomes[provider_tier]),
    )

    # A stale freeze on a subscription that turns out to answer is dropped.
    book.freeze("LOCAL_CODEX", seconds=agent_quota.QUOTA_FREEZE_SECONDS)

    exit_code = agent_ping.main([])

    out = capsys.readouterr().out
    assert "LOCAL_CODEX" in out and "ok" in out
    assert "not_authenticated" in out and "log in again" in out
    # Exit non-zero so a script can act on it.
    assert exit_code == 1
    assert book.is_frozen("LOCAL_CODEX") is False
    # Diagnosing is not enough. A real run against a spent Codex plan reported
    # `out_of_quota` and then let the very next call reach for the same
    # subscription, because nothing had been written down.
    assert "out_of_quota" in out
    # Booked against the *pool*, so agy's Opus is left alone: one CLI, two
    # separately metered allowances.
    assert book.is_frozen("AGY_GEMINI") is True
    assert book.is_frozen("LOCAL_AGY") is False
    # An expired login is not an empty wallet, and must never be frozen.
    assert book.is_frozen("LOCAL_CLAUDE") is False
