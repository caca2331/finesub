"""Ask each local agent CLI whether it can still answer.

None of the three CLIs can be asked how much subscription is left, so the only
honest check is a real, tiny call. This is the same probe the router runs when
it suspects exhaustion, exposed on its own so a person can ask directly --
"is it me, my login, or my quota?" -- without starting a run to find out.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from typing import Any, Sequence

from . import agent_quota, codex_login
from .agent_paths import vendor_error_text


def _tier_rows(routes: Any) -> list[tuple[str, str, str]]:
    """(provider_tier, model, quota_pool) for every routable local-agent target.

    The pool comes along because it is what a freeze is booked against, and on
    Antigravity it is not the tier: one CLI, two separately metered
    allowances. Probing per model already gives one row each.
    """

    seen: dict[tuple[str, str, str], None] = {}
    for target_id in sorted(routes.targets):
        target = routes.targets[target_id]
        if target.backend != "local_agent":
            continue
        fact = routes.target_fact(target_id)
        seen.setdefault(
            (fact.provider_tier, fact.api_model_id, fact.effective_quota_pool), None
        )
    return list(seen)


def _probe_one(
    tier: str, model: str, pool: str, *, timeout_seconds: int
) -> dict[str, Any]:
    from ..routing.execution_policy import (
        driver_for_provider_tier,
        load_execution_settings,
        probe_reasoning_effort,
    )
    from ..routing.model_catalog import get_model_catalog_entry_for_tier
    from .local_agent import probing_driver

    row: dict[str, Any] = {"provider_tier": tier, "model": model, "quota_pool": pool}
    frozen = agent_quota.default_ledger().frozen_until(pool)
    if frozen is not None:
        row["frozen_until"] = frozen.isoformat(timespec="minutes")

    settings = load_execution_settings()
    try:
        # Never through a paid twin: this tool's whole job is to say which of
        # "me, my login, my quota" is at fault for *this* model, and a switch
        # to another allowance answers for the wrong one -- while billing.
        driver = probing_driver(
            driver_for_provider_tier(settings, provider_tier=tier, model=model)
        )
    except ValueError as exc:
        return {**row, "status": "unroutable", "detail": str(exc)}

    # A person waiting on a one-word answer should not sit through the whole
    # production call budget when the CLI has gone unresponsive.
    config = getattr(driver, "config", None)
    if config is not None:
        driver.config = replace(config, timeout_seconds=int(timeout_seconds))
    probe = driver.probe()
    row["driver"] = driver.driver_id
    row["cli_version"] = probe.version
    if not probe.available:
        return {**row, "status": "cli_missing", "detail": probe.error}
    if not driver.meets_requirements(probe):
        return {
            **row,
            "status": "cli_unusable",
            "detail": "the installed CLI lacks flags this driver requires",
        }

    started = time.monotonic()
    try:
        result = driver.run(
            agent_quota.QUOTA_PING_MESSAGES,
            task="quota-probe",
            profile_id="probe=agent-ping",
            reasoning_effort=probe_reasoning_effort(
                settings, get_model_catalog_entry_for_tier(model, tier)
            ),
        )
    except Exception as exc:
        return _failed_row(_with_codex_home(row, exc), exc, pool)
    row = _with_codex_home(row, result)
    row["seconds"] = round(time.monotonic() - started, 1)
    row["reply"] = result.content.strip()[:80]
    # It answered, so whatever the ledger thought is out of date.
    agent_quota.default_ledger().note_success(pool)
    row.pop("frozen_until", None)
    return {**row, "status": "ok"}


def _with_codex_home(row: dict[str, Any], outcome: Any) -> dict[str, Any]:
    """``row`` plus the Codex home the call ran from, from its audit record:
    a one-time warning on stderr is easily lost in a long run, and this is
    the command a person runs to check."""

    attempts = getattr(outcome, "_harness_execution_attempts", None) or [
        getattr(outcome, "execution_attempt", None)
    ]
    for attempt in attempts:
        isolation = attempt.get("isolation") if isinstance(attempt, dict) else None
        if isinstance(isolation, dict) and isolation.get("codex_home"):
            kind = str(isolation["codex_home"])
            if kind == "user" and isolation.get("codex_login") == "unknown":
                kind = "user-unknown"
            return {**row, "codex_home": kind}
    return row


def _failed_row(row: dict[str, Any], exc: Exception, pool: str) -> dict[str, Any]:
    from .local_agent import (
        LocalAgentProviderError,
        LocalAgentQuotaError,
        LocalAgentUnavailableError,
    )

    if isinstance(exc, LocalAgentProviderError):
        # The user's own provider settings, not their login: a proxy that is
        # not running answers "log in again" with the wrong advice.
        return {**row, "status": "provider_unavailable", "detail": str(exc)[:300]}
    if isinstance(exc, LocalAgentUnavailableError):
        return {**row, "status": "not_authenticated", "detail": str(exc)[:300]}
    if isinstance(exc, LocalAgentQuotaError):
        # Diagnosing is not enough: whoever ran this wants the next run to
        # stop reaching for a subscription that just said no.
        deadline = agent_quota.default_ledger().freeze(
            pool, seconds=float(agent_quota.QUOTA_FREEZE_SECONDS), reason=str(exc)
        )
        return {
            **row,
            "status": "out_of_quota",
            "frozen_until": deadline.isoformat(timespec="minutes"),
            "detail": str(exc)[:300],
        }
    # Show what the CLI itself said; nothing here interprets it.
    said = vendor_error_text(exc)
    return {
        **row,
        "status": "failed",
        "detail": (said or f"{type(exc).__name__}: {exc}")[:400],
    }


_ADVICE = {
    "ok": "usable",
    "out_of_quota": "subscription is spent; it will be skipped until it recovers",
    "not_authenticated": "log in again with this CLI",
    "provider_unavailable": (
        "check the provider in this CLI's own config: is the proxy running, "
        "are the variables it names set?"
    ),
    "cli_missing": "the CLI is not installed, or not on PATH",
    "cli_unusable": "update the CLI",
    "unroutable": "no driver is registered for this provider tier",
    "failed": "the call failed for another reason; see the detail",
}


#: What each Codex home means for the person reading `agent-ping`.
def _codex_home_note(kind: str) -> str:
    command = codex_login.login_command()
    return {
        "login": f"Codex home: FineSub's own login (see `{command} --status`)",
        "harness": "Codex home: the run's own; your provider's key comes from the environment, no login needed",
        "user": (
            "Codex home: your own ~/.codex, so your global AGENTS.md reaches FineSub's calls; "
            f"run `{command}` to give FineSub a login of its own"
        ),
        "user-unknown": (
            "Codex home: your own ~/.codex, because FineSub could not read its own login's status "
            f"(`codex login status` did not answer); your global AGENTS.md reaches FineSub's calls. "
            f"`{command} --status` shows why"
        ),
    }.get(kind, kind)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finesub agent-ping",
        description=(
            "Send one tiny call to each installed agent CLI and report whether "
            "it answers. Each probe spends a small amount of that "
            "subscription's quota."
        ),
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="TIER_OR_POOL",
        help=(
            "probe only this provider tier or quota pool (repeatable); "
            "default is every routable one"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="emit one JSON object instead of a table"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="per-probe timeout in seconds"
    )
    args = parser.parse_args(argv)

    from ..routing.model_routes import default_model_routes

    # Matching on either name: a freeze is booked per pool, so "why is this row
    # frozen and its neighbour not" is a pool question, and `--tier LOCAL_AGY`
    # has no way to say "just the Claude allowance".
    wanted = {agent_quota.normalized_pool(item) for item in args.tier}
    rows = [
        _probe_one(tier, model, pool, timeout_seconds=args.timeout)
        for tier, model, pool in _tier_rows(default_model_routes())
        if not wanted
        or {agent_quota.normalized_pool(tier), agent_quota.normalized_pool(pool)}
        & wanted
    ]
    if not rows:
        print("No local-agent targets are declared in the routing tables.")
        return 0

    if args.json:
        print(json.dumps({"probes": rows}, ensure_ascii=False, indent=2))
    else:
        # Name the pool whenever it is not just the tier: two agy rows share a
        # CLI but not an allowance, so one showing `frozen until` and the other
        # not reads as a bug until you can see they are metered apart.
        labels = {
            id(row): (
                f"{row['provider_tier']} / {row['model']}"
                + (
                    f"  [{row['quota_pool']}]"
                    if row["quota_pool"] != row["provider_tier"]
                    else ""
                )
            )
            for row in rows
        }
        width = max(len(value) for value in labels.values())
        for row in rows:
            status = str(row["status"])
            line = f"{labels[id(row)]:<{width}}  {status:<18}"
            if row.get("frozen_until"):
                line += f"  frozen until {row['frozen_until']}"
            print(f"{line}  {_ADVICE.get(status, '')}")
            if row.get("detail"):
                print(f"{'':<{width}}  {row['detail']}")
            if row.get("codex_home"):
                print(f"{'':<{width}}  {_codex_home_note(row['codex_home'])}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
