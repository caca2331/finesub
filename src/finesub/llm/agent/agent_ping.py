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

from . import agent_quota
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
    from ..routing.execution_policy import driver_for_provider_tier, load_execution_settings
    from .local_agent import (
        LocalAgentQuotaError,
        LocalAgentUnavailableError,
    )

    row: dict[str, Any] = {"provider_tier": tier, "model": model, "quota_pool": pool}
    frozen = agent_quota.default_ledger().frozen_until(pool)
    if frozen is not None:
        row["frozen_until"] = frozen.isoformat(timespec="minutes")

    settings = load_execution_settings()
    try:
        driver = driver_for_provider_tier(settings, provider_tier=tier, model=model)
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
            reasoning_effort="low",
        )
    except LocalAgentUnavailableError as exc:
        return {**row, "status": "not_authenticated", "detail": str(exc)[:300]}
    except LocalAgentQuotaError as exc:
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
    except Exception as exc:
        # Show what the CLI itself said; nothing here interprets it.
        said = vendor_error_text(exc)
        return {
            **row,
            "status": "failed",
            "detail": (said or f"{type(exc).__name__}: {exc}")[:400],
        }
    row["seconds"] = round(time.monotonic() - started, 1)
    row["reply"] = result.content.strip()[:80]
    # It answered, so whatever the ledger thought is out of date.
    agent_quota.default_ledger().note_success(pool)
    row.pop("frozen_until", None)
    return {**row, "status": "ok"}


_ADVICE = {
    "ok": "usable",
    "out_of_quota": "subscription is spent; it will be skipped until it recovers",
    "not_authenticated": "log in again with this CLI",
    "cli_missing": "the CLI is not installed, or not on PATH",
    "cli_unusable": "update the CLI",
    "unroutable": "no driver is registered for this provider tier",
    "failed": "the call failed for another reason; see the detail",
}


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
    # has no way to say "just the Opus allowance".
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
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
