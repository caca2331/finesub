"""Which public entry points this machine downloads from.

Resolves to exactly two answers, `cn` or `global`, and records why. The
narrowness is the point: a route is a choice between mirrors of the same
bytes, not a policy surface. Content is still decided by the lock file hashes,
the pinned resource SHA-256 and the model manifest -- a mirror is a faster way
to fetch something we already know the shape of, never a new root of trust.

Three rules worth stating:

* **The probe follows the download's exit, not the machine's location.** It is
  issued through the same preferred route `downloader` uses, so a VPN or a
  corporate proxy is described by where it comes out. This is an approximation
  with a known edge: `network_routes()` is a list -- proxy first, direct as a
  fallback -- and a download that falls through to a different entry than the
  probe used is not evidence about the route, so its outcome never updates the
  cached answer.
* **No address is kept.** The cache holds the verdict, when it was made and
  which endpoint made it. An IP is personal data we have no reason to store.
* **A mirror that keeps failing turns itself off, per machine.** The source
  table ships with the release, so a public entry point going down would
  otherwise need a new version to route around, and the environment override
  only helps someone who reads the documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from finesub_bootstrap.fsops import write_atomic

REGIONS = ("cn", "global")

#: How long an automatic verdict is reused. Long enough that an ordinary
#: session resolves once, short enough that moving countries or turning a VPN
#: off is noticed the same day.
CACHE_TTL_SEC = 24 * 60 * 60

#: The whole probe, across every candidate. A route decision must never be
#: what makes a run feel slow to start, so the budget is small and failing it
#: simply means `global`.
TOTAL_BUDGET_SEC = 3.0
PER_ENDPOINT_TIMEOUT_SEC = 1.5

#: Consecutive failures of one resource class before this machine stops
#: preferring its cn entry point.
FAILURE_LIMIT = 3

STATE_NAME = "download-routes.json"

#: Overrides, in the order they beat each other. Empty string means "disable
#: this class of acceleration and use the official source".
REGION_ENVIRONMENT = "FINESUB_DOWNLOAD_REGION"
OVERRIDES = {
    "pypi": "FINESUB_PYPI_INDEX",
    "huggingface": "FINESUB_HF_ENDPOINT",
    "github": "FINESUB_GITHUB_FILE_PROXY",
}


@dataclass(frozen=True, slots=True)
class RouteDecision:
    region: str
    source: str  # "forced" | "cached" | "probe" | "default"
    endpoint: str = ""

    def describe(self) -> str:
        """A `doctor` line: the verdict and how it was reached, never an IP."""

        detail = {
            "forced": "环境变量",
            "cached": "自动检测，缓存",
            "probe": "自动检测",
            "default": "默认",
        }.get(self.source, self.source)
        return f"{self.region} ({detail})"


def state_path(data_root: Path) -> Path:
    """Beside the small personal data, never in the big-data root.

    The big-data root may not have been chosen yet when the first route is
    resolved -- that is the download this decision is for.
    """

    return data_root / STATE_NAME


def _read_state(data_root: Path) -> dict[str, Any]:
    try:
        body = json.loads(state_path(data_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _write_state(data_root: Path, body: dict[str, Any]) -> None:
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        write_atomic(
            state_path(data_root), json.dumps(body, ensure_ascii=False, indent=2)
        )
    except OSError:
        # A cache we cannot write costs one probe per run; it must not stop one.
        pass


def forced_region() -> str | None:
    value = os.environ.get(REGION_ENVIRONMENT, "").strip().lower()
    return value if value in REGIONS else None


def cached_region(data_root: Path, *, now: float | None = None) -> RouteDecision | None:
    body = _read_state(data_root).get("region")
    if not isinstance(body, dict):
        return None
    region = body.get("region")
    decided_at = body.get("decidedAt")
    if region not in REGIONS or not isinstance(decided_at, (int, float)):
        return None
    if (now if now is not None else time.time()) - decided_at > CACHE_TTL_SEC:
        return None
    return RouteDecision(
        region=region, source="cached", endpoint=str(body.get("endpoint") or "")
    )


def remember_region(
    data_root: Path, decision: RouteDecision, *, now: float | None = None
) -> None:
    body = _read_state(data_root)
    body["schemaVersion"] = 1
    body["region"] = {
        "region": decision.region,
        # No address: only the verdict, its age and which endpoint gave it.
        "decidedAt": now if now is not None else time.time(),
        "endpoint": decision.endpoint,
    }
    _write_state(data_root, body)


def resolve_region(
    data_root: Path,
    *,
    probe=None,
    now: float | None = None,
) -> RouteDecision:
    """`cn` or `global`, by override, then cache, then probe, then default.

    Failing to decide is never an error: an offline machine, a blocked probe
    or a nonsense answer all mean `global`, which is the official source.
    """

    forced = forced_region()
    if forced is not None:
        return RouteDecision(region=forced, source="forced")
    cached = cached_region(data_root, now=now)
    if cached is not None:
        return cached
    decision = (probe or probe_region)()
    if decision is None:
        return RouteDecision(region="global", source="default")
    remember_region(data_root, decision, now=now)
    return decision


def probe_region(endpoints: list[tuple[str, str]] | None = None) -> RouteDecision | None:
    """Ask public country endpoints where this machine's traffic comes out.

    One primary and one spare, both bounded, and the whole thing bounded
    again: a route decision that costs more than a couple of seconds has
    already cost more than it saves.
    """

    import httpx

    from finesub_bootstrap.http_client import create_client, network_routes

    candidates = endpoints if endpoints is not None else public_country_endpoints()
    if not candidates:
        return None
    route = network_routes()[0]
    started = time.monotonic()
    for name, url in candidates:
        if time.monotonic() - started >= TOTAL_BUDGET_SEC:
            break
        try:
            with create_client(
                route, timeout=httpx.Timeout(PER_ENDPOINT_TIMEOUT_SEC)
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                country = _country_from(response)
        except Exception:
            continue
        if country is None:
            continue
        return RouteDecision(
            region="cn" if country == "CN" else "global",
            source="probe",
            endpoint=name,
        )
    return None


def _country_from(response: Any) -> str | None:
    """A two-letter country code from either a JSON body or a plain one."""

    text = (response.text or "").strip()
    if text.startswith("{"):
        try:
            body = response.json()
        except ValueError:
            return None
        for key in ("country_code", "countryCode", "country"):
            value = body.get(key) if isinstance(body, dict) else None
            if isinstance(value, str) and len(value.strip()) == 2:
                return value.strip().upper()
        return None
    return text.upper() if len(text) == 2 and text.isalpha() else None


def public_country_endpoints() -> list[tuple[str, str]]:
    """The probe endpoints, from the shipped source table.

    An empty table would make `auto` resolve to `global` rather than to an
    unverified guess; the shipped entries have passed the availability check
    (the full release drill is still pending, see the plan doc §5).
    """

    from finesub_bootstrap.download_sources import load_sources

    return [
        (str(entry.get("name") or url), str(url))
        for entry in load_sources().get("countryEndpoints", [])
        if (url := entry.get("url"))
    ]


# -- per-class degradation ------------------------------------------------


def failures(data_root: Path, resource_class: str) -> int:
    body = _read_state(data_root).get("failures")
    if not isinstance(body, dict):
        return 0
    value = body.get(resource_class)
    return value if isinstance(value, int) and value > 0 else 0


def record_failure(data_root: Path, resource_class: str) -> None:
    body = _read_state(data_root)
    counters = body.get("failures")
    if not isinstance(counters, dict):
        counters = {}
    counters[resource_class] = failures(data_root, resource_class) + 1
    body["failures"] = counters
    body["schemaVersion"] = 1
    _write_state(data_root, body)


def record_success(data_root: Path, resource_class: str) -> None:
    body = _read_state(data_root)
    counters = body.get("failures")
    if not isinstance(counters, dict) or resource_class not in counters:
        return
    counters.pop(resource_class, None)
    body["failures"] = counters
    _write_state(data_root, body)


def is_degraded(data_root: Path, resource_class: str) -> bool:
    """Whether this machine has given up on the cn entry for this class."""

    return failures(data_root, resource_class) >= FAILURE_LIMIT


def active_mirror(data_root: Path, resource_class: str, region: str) -> str:
    """The entry point to actually use, or "" for the official source.

    The one place that weighs every reason not to use a mirror -- wrong
    region, nothing configured, an override emptied on purpose, or a host this
    machine has given up on. Each caller asking those questions itself is how
    a new resource class ends up honouring three of the four.
    """

    from finesub_bootstrap import download_sources

    if is_degraded(data_root, resource_class):
        return ""
    return download_sources.mirror_for(resource_class, region)
