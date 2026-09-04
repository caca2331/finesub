"""Model-level RPM/TPM(input) rate limiting and daily exhaustion tracking."""

from __future__ import annotations

from finesub.reporting import current_reporter

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import itertools
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

from finesub import state as state_store
from finesub.paths import resolve_state_file

from .routing.config import ModelEndpoint, RateLimitPolicy
from .routing.model_routes import runtime_fact_for

# ---------------------------------------------------------------------------
# 429 Retry-After parser
# ---------------------------------------------------------------------------

# Patterns ordered by specificity. Each captures a numeric wait in seconds.
# Covers: Gemini ("Please retry in 51.98s", "retryDelay": "51.980s"),
# OpenAI/Anthropic ("Retry-After: 30", "retry_after": 30, "retry after 30s"),
# and generic provider text ("wait 60 seconds", "try again in 45s").
_RETRY_AFTER_PATTERNS: List[re.Pattern] = [
    # "retryDelay": "51.980s" or retryDelay: 51.98s (Gemini JSON / text)
    re.compile(r"retryDelay[\"'\s:]+[\"']?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE),
    # "Retry-After: 30" or "retry_after": 30 (HTTP header / OpenAI / Anthropic)
    re.compile(r"retry[_-]after[\"'\s:]+[\"']?(\d+(?:\.\d+)?)\s*s?", re.IGNORECASE),
    # "Please retry in 51.98s" / "retry in 30 seconds" / "try again in 45s"
    re.compile(
        r"(?:please\s+)?(?:retry|try\s+again)\s+in\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?",
        re.IGNORECASE,
    ),
    # "retry after 30 seconds" / "wait 60 seconds" / "wait for 45s"
    re.compile(
        r"(?:retry\s+after|wait(?:\s+for)?)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?",
        re.IGNORECASE,
    ),
    # Bare "in 30s" as a last resort (avoids matching unrelated numbers)
    re.compile(r"\bin\s+(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE),
]


def parse_retry_after_seconds(exc: BaseException) -> float:
    """Extract the provider-suggested wait time from a 429/rate-limit error.

    Structured first: an exception may expose ``retry_after_seconds`` (the
    custom-provider transports do, from the response header -- a header cannot
    appear in the stringified body, so text scanning would miss it entirely).
    Otherwise parses mainstream provider formats (Gemini retryDelay,
    OpenAI/Anthropic Retry-After echoed in the body, generic "retry in Xs"
    text). Returns 0.0 when no wait hint is found. The caller clamps the
    value; this function only extracts.
    """

    structured = getattr(exc, "retry_after_seconds", None)
    if isinstance(structured, (int, float)) and structured > 0:
        return float(structured)
    text = f"{type(exc).__name__}: {exc}"
    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    return 0.0


def key_id_for_secret(secret: str) -> str:
    """Stable non-reversible identifier for an API key (mirrors exa pool).

    Named keys (from ``name:key`` .env syntax) use their human-readable name
    directly; anonymous keys get a ``sha256:<first-12-hex>`` digest so the
    .state file never contains the raw secret.
    """

    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


_PACIFIC = ZoneInfo("America/Los_Angeles")
_STATE_NAMESPACE = "llm_rate_limit"
_DAILY_EXHAUSTED_KEY = "daily_exhausted"
_DAILY_STRIKES_KEY = "daily_strikes"
_COMBO_COOLDOWNS_KEY = "combo_cooldowns"

# After a (tier, model, key) exhausts sticky retries: skip 20m, then probe
# once (retry=0) until 120m; probe success clears, probe failure restarts.
COMBO_COOLDOWN_SKIP_SECONDS = 20 * 60
COMBO_COOLDOWN_TTL_SECONDS = 120 * 60

# Lock a key/endpoint as daily-exhausted only after several consecutive
# per-day 429s — not a lone one. A success clears the streak. (The prior
# ≥5-minute span gate was removed 2026-07-29: with sticky retries already
# shortened because even 5xx burns daily quota, waiting out flicker is
# less valuable than rotating off a repeatedly PerDay-failing key.)
DAILY_STRIKE_COUNT = 3


class ComboCooldownPhase(str, Enum):
    NONE = "none"
    SKIP = "skip"
    PROBE = "probe"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_timestamp(stamp: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _default_state_path() -> Path:
    return resolve_state_file()


def _monotonic_now() -> float:
    import time

    return time.monotonic()


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def endpoint_key(endpoint: ModelEndpoint, key_id: str = "") -> str:
    """Accounting key for rate-limit state.

    RPM/TPM sliding windows use the endpoint-level key (``key_id=""``).
    Strike/daily-exhausted tracking passes a per-key ``key_id`` so one
    exhausted key does not poison its siblings.
    """

    base = f"{endpoint.provider_tier}|{endpoint.api_model_id}"
    return f"{base}|{key_id}" if key_id else base


def _pacific_calendar_day(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(_PACIFIC).date().isoformat()


def _same_pacific_day(earlier: float, later: float) -> bool:
    """Whether two wall-clock stamps fall on one Pacific calendar day.

    Gemini's daily quotas roll over on Pacific midnight, so that is the only
    boundary a "used up for today" streak may span.
    """

    return _pacific_calendar_day(
        datetime.fromtimestamp(earlier, tz=ZoneInfo("UTC"))
    ) == _pacific_calendar_day(datetime.fromtimestamp(later, tz=ZoneInfo("UTC")))


@dataclass
class _TokenEvent:
    """One booked TPM reservation; mutable so ``settle`` can refine it."""

    ts: float
    tokens: int
    event_id: int


@dataclass
class _WindowBucket:
    request_times: Deque[float] = field(default_factory=deque)
    token_events: Deque[_TokenEvent] = field(default_factory=deque)
    # FIFO booking horizon for TPM-bearing reservations: a later one never
    # departs before an earlier one. RPM-only notes and exhausted early-outs
    # bypass it and are inserted in timestamp order instead.
    last_depart: float = 0.0


def _insert_by_order(deq: Deque, item: Any, key: Callable[[Any], float]) -> None:
    """Insert into a timestamp-sorted deque, scanning from the tail.

    Off-horizon bookings land at or near the tail in practice, so the scan is
    effectively O(1); keeping the deques sorted keeps ``_prune`` a popleft
    loop.
    """

    idx = len(deq)
    while idx > 0 and key(deq[idx - 1]) > key(item):
        idx -= 1
    deq.insert(idx, item)


@dataclass(frozen=True)
class RateLimitTicket:
    """A booked departure slot (docs/llm_harness_behavior.md).

    ``reserve`` computes the slot and books the accounting **inside the lock**;
    the caller sleeps outside the lock until ``depart_at`` and then fires.
    ``event_id`` addresses this call's own TPM reservation so ``settle`` can
    refine it even after later calls have booked behind it.
    """

    bucket_key: str
    depart_at: float
    event_id: int | None = None


@dataclass(frozen=True)
class RateLimitLimits:
    effective_rpm: int
    effective_tpm: int


class ModelRateLimiter:
    """Sliding-window RPM + input TPM limiter with observed daily exhaustion."""

    def __init__(
        self,
        *,
        policy: RateLimitPolicy | None = None,
        state_path: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.policy = policy or RateLimitPolicy()
        self.state_path = Path(state_path) if state_path else _default_state_path()
        self.enabled = enabled
        # One lock for every read-modify-write on the in-process view. The
        # limiter instance is `lru_cache`d and shared across threads; without
        # this, concurrent callers see the same headroom and all fire at once
        # (check-then-act), collapsing throttling into a 429 storm.
        self._lock = threading.RLock()
        self._event_ids = itertools.count(1)
        self._windows: Dict[str, _WindowBucket] = {}
        self._daily_exhausted: Dict[str, str] = {}
        self._daily_strikes: Dict[str, List[float]] = {}
        self._combo_cooldowns: Dict[str, str] = {}
        # combo key -> the cooldown window (its start stamp) whose probe this
        # process has already handed out. In memory on purpose: a persisted
        # claim from a process that died mid-probe would block the combo for
        # the rest of the window, and separate front ends probing separately
        # is the pre-existing situation, not a regression this guards.
        self._probe_claims: Dict[str, str] = {}
        # Keys this process removed on purpose. Without them a merge on
        # write would adopt the stale copy straight back off disk.
        self._cleared_exhausted: set[str] = set()
        self._cleared_cooldowns: set[str] = set()
        self._load_state()

    def limits_for(self, endpoint: ModelEndpoint) -> RateLimitLimits:
        # Merged facts, not the packaged catalog alone: a user-declared model
        # has no ``model_catalog.psv`` row, so looking only there turned every
        # real call to a custom provider into a ValueError (the limiter is
        # enabled on every production path).
        entry = runtime_fact_for(endpoint)
        if entry is None:
            raise ValueError(
                f"No runtime fact for {endpoint.provider_tier} + "
                f"{endpoint.api_model_id} (neither model_catalog.psv nor "
                "config.toml [llm.models] declares it)"
            )
        factor = self.policy.safety_factor
        effective_tpm = -1 if entry.tpm < 0 else max(1, int(entry.tpm * factor))
        return RateLimitLimits(
            effective_rpm=max(1, int(entry.rpm * factor)),
            effective_tpm=effective_tpm,
        )

    def is_daily_exhausted(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[datetime] = None,
    ) -> bool:
        now = now or datetime.now(_PACIFIC)
        stamp = self._daily_exhausted.get(endpoint_key(endpoint, key_id))
        if not stamp:
            return False
        try:
            exhausted_at = datetime.fromisoformat(stamp)
        except ValueError:
            return False
        return _pacific_calendar_day(exhausted_at) == _pacific_calendar_day(now)

    def mark_daily_exhausted(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        now = now or datetime.now(_PACIFIC)
        key = endpoint_key(endpoint, key_id)
        with self._lock:
            self._daily_exhausted[key] = now.isoformat(timespec="milliseconds")
            self._cleared_exhausted.discard(key)
            self._daily_strikes.pop(key, None)
            self._persist_state()

    def note_daily_quota_hit(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[float] = None,
        departed_at: Optional[float] = None,
    ) -> bool:
        """Record a per-day 429 and lock only once the strike gate confirms it.

        Locking requires ``DAILY_STRIKE_COUNT`` consecutive hits (a lone PerDay
        429 never poisons the whole day; success clears the streak). Returns
        True iff the endpoint is now marked daily-exhausted.

        Counted **per HTTP attempt**, including the sticky retries of a single
        logical call -- that is deliberate, not an oversight. The wait between
        those attempts is ``max(exponential, provider retryDelay)``, and a real
        PerDay 429 carries a substantial retryDelay, so three of them span a
        genuinely long stretch rather than a burst. (A synthetic error with no
        retryDelay collapses to the bare 5/9/17s backoff, which makes the gate
        look far twitchier in a test than it is against the real API.)

        Strikes are dropped when they belong to an earlier Pacific day: without
        that, two isolated flickers on Monday plus one on Thursday -- a fresh
        daily quota -- added up to a lock, because the streak only ever cleared
        on success or on locking.

        ``departed_at`` (wall clock) is when the failing attempt actually
        fired. A strike only counts if the attempt departed **after the
        previous strike was recorded**: N concurrent in-flight calls hitting
        one quota exhaustion return N 429s within seconds, and without this
        gate they fill the streak instantly and lock the key for a whole
        Pacific day on what is a single observation
        (docs/llm_harness_behavior.md). Serial retries depart after the
        prior 429 by construction, so their behaviour is unchanged.
        """
        with self._lock:
            key = endpoint_key(endpoint, key_id)
            now = now if now is not None else time.time()
            strikes = [
                stamp
                for stamp in self._daily_strikes.setdefault(key, [])
                if _same_pacific_day(stamp, now)
            ]
            self._daily_strikes[key] = strikes
            if departed_at is not None and strikes and departed_at <= strikes[-1]:
                # Departed before the newest strike was known: it confirms the
                # same observation, not a new one.
                self._persist_state()
                return False
            strikes.append(now)
            if len(strikes) > DAILY_STRIKE_COUNT:
                del strikes[:-DAILY_STRIKE_COUNT]
            if len(strikes) >= DAILY_STRIKE_COUNT:
                self.mark_daily_exhausted(endpoint, key_id=key_id)
                return True
            self._persist_state()
            return False

    def reset_daily_strikes(self, endpoint: ModelEndpoint, *, key_id: str = "") -> None:
        """Clear the strike streak (call on a successful response) so only
        *consecutive* per-day 429s accumulate toward a lock."""
        key = endpoint_key(endpoint, key_id)
        with self._lock:
            if self._daily_strikes.pop(key, None) is not None:
                self._persist_state()

    def combo_cooldown_phase(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[datetime] = None,
    ) -> ComboCooldownPhase:
        """Return the transient cooldown phase for a (tier, model, key) combo."""

        if not self.enabled:
            return ComboCooldownPhase.NONE
        key = endpoint_key(endpoint, key_id)
        stamp = self._combo_cooldowns.get(key)
        if not stamp:
            return ComboCooldownPhase.NONE
        started_at = _parse_utc_timestamp(stamp)
        if started_at is None:
            # Under the lock like every other read-modify-write in this class:
            # it was the one mutation that skipped it.
            with self._lock:
                self._combo_cooldowns.pop(key, None)
                self._probe_claims.pop(key, None)
                self._cleared_cooldowns.add(key)
                self._persist_state()
            return ComboCooldownPhase.NONE
        now = now or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = (now - started_at).total_seconds()
        if elapsed >= COMBO_COOLDOWN_TTL_SECONDS:
            self.clear_combo_cooldown(endpoint, key_id=key_id)
            return ComboCooldownPhase.NONE
        if elapsed < COMBO_COOLDOWN_SKIP_SECONDS:
            return ComboCooldownPhase.SKIP
        return ComboCooldownPhase.PROBE

    def combo_cooldown_retry_at(
        self, endpoint: ModelEndpoint, *, key_id: str = ""
    ) -> Optional[datetime]:
        """The moment this combo's skip window ends, if one was ever started.

        Reported to the caller so "everything was skipped" can say how long a
        transient cooldown still has to run -- minutes, which reads nothing
        like the daily lock it used to be lumped in with.

        A pure read, so unlike `combo_cooldown_phase` it neither clears an
        unparseable stamp nor an expired window: the answer is only meaningful
        once that method has said SKIP, and the caller asks in that order.
        """

        stamp = self._combo_cooldowns.get(endpoint_key(endpoint, key_id))
        started_at = _parse_utc_timestamp(stamp) if stamp else None
        if started_at is None:
            return None
        return started_at + timedelta(seconds=COMBO_COOLDOWN_SKIP_SECONDS)

    def claim_combo_probe(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[datetime] = None,
    ) -> bool:
        """Hand the probe to exactly one caller per cooldown window.

        PROBE means "spend one zero-retry call to see whether this combo came
        back". Reading the phase is a pure read, so with ``continuity=parallel``
        every concurrent window read PROBE at once and each fired its own --
        four probes of something meant to be tested with one, four requests off
        a free tier's daily allowance, and four ``note_combo_exhausted`` calls
        restarting the 20-minute skip on top of each other. This is the same
        "N concurrent observations of one fact" the daily-strike gate already
        solved with ``departed_at``; the cooldown never got the treatment.

        Callers that lose the claim must treat the combo as SKIP. The claim is
        scoped to the window's start stamp, so a restarted cooldown re-opens a
        probe on its own. A probe that never settles holds the claim for the
        rest of the window -- deliberately the conservative direction, since
        the alternative is hammering a combo that just failed.
        """

        with self._lock:
            if (
                self.combo_cooldown_phase(endpoint, key_id=key_id, now=now)
                is not ComboCooldownPhase.PROBE
            ):
                return False
            key = endpoint_key(endpoint, key_id)
            window = self._combo_cooldowns.get(key, "")
            if self._probe_claims.get(key) == window:
                return False
            self._probe_claims[key] = window
            return True

    def effective_sticky_retries(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        default_retries: int,
        now: Optional[datetime] = None,
    ) -> int:
        """Sticky retry budget for this combo (0 during PROBE, else default)."""

        if self.combo_cooldown_phase(endpoint, key_id=key_id, now=now) is ComboCooldownPhase.PROBE:
            return 0
        return max(0, int(default_retries))

    def note_combo_exhausted(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        """Start or restart the 20m skip + 100m probe cooldown window."""

        if not self.enabled:
            return
        now = now or _utc_now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        key = endpoint_key(endpoint, key_id)
        with self._lock:
            self._combo_cooldowns[key] = now.isoformat(timespec="milliseconds")
            # New window, new probe: the old claim refers to the window that
            # just failed and must not suppress the next one.
            self._probe_claims.pop(key, None)
            self._cleared_cooldowns.discard(key)
            self._persist_state()

    def clear_combo_cooldown(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
    ) -> None:
        key = endpoint_key(endpoint, key_id)
        with self._lock:
            self._cleared_cooldowns.add(key)
            self._probe_claims.pop(key, None)
            if self._combo_cooldowns.pop(key, None) is not None:
                self._persist_state()

    def wait_seconds(
        self,
        endpoint: ModelEndpoint,
        estimated_input_tokens: int,
        *,
        key_id: str = "",
        now: Optional[float] = None,
    ) -> float:
        """Read-only estimate of the wait a reservation made now would get."""

        if not self.enabled:
            return 0.0
        if self.is_daily_exhausted(endpoint, key_id=key_id):
            return 0.0
        now = now if now is not None else _monotonic_now()
        with self._lock:
            limits = self.limits_for(endpoint)
            bucket = self._bucket(endpoint_key(endpoint, key_id))
            self._prune(bucket, now)
            slot = self._booking_slot(
                bucket, limits, now, max(0, int(estimated_input_tokens))
            )
            return max(0.0, slot - now)

    def _booking_slot(
        self,
        bucket: _WindowBucket,
        limits: RateLimitLimits,
        now: float,
        tokens: int,
    ) -> float:
        """Earliest departure time at which this call fits both windows.

        Booked reservations may carry **future** timestamps, so the windows
        are evaluated at the candidate time rather than at ``now``; the loop
        advances the candidate past whichever constraint still binds until
        neither does (each step lands on an event expiry, so it terminates).
        """

        window = self.policy.window_seconds
        candidate = now
        while True:
            moved = False
            cutoff = candidate - window
            active_requests = [t for t in bucket.request_times if t > cutoff]
            rpm = limits.effective_rpm
            if len(active_requests) >= rpm:
                target = active_requests[len(active_requests) - rpm] + window
                if target > candidate:
                    candidate = target
                    moved = True
                    continue
            tpm = limits.effective_tpm
            if tpm >= 0:
                active_events = [e for e in bucket.token_events if e.ts > cutoff]
                projected = sum(e.tokens for e in active_events) + tokens
                if projected > tpm and active_events:
                    needed = projected - tpm
                    running = 0
                    for event in active_events:
                        running += event.tokens
                        if running >= needed:
                            target = event.ts + window
                            if target > candidate:
                                candidate = target
                                moved = True
                            break
            if not moved:
                return candidate

    def _rpm_only_slot(
        self,
        bucket: _WindowBucket,
        limits: RateLimitLimits,
        now: float,
    ) -> float:
        """Earliest time an RPM-only note can fire (no TPM constraint).

        Unlike :meth:`_booking_slot`, only requests that have actually
        departed **by the candidate time** count toward the window --
        ``_booking_slot`` has no upper bound on the event filter, which is
        correct under FIFO (the candidate never precedes a booking) but would
        charge a zero-token note for sibling windows' future TPM bookings and
        stall a sticky retry for minutes (2026-08-10 review P2).
        """

        window = self.policy.window_seconds
        rpm = limits.effective_rpm
        candidate = now
        while True:
            cutoff = candidate - window
            active = [t for t in bucket.request_times if cutoff < t <= candidate]
            if len(active) < rpm:
                return candidate
            target = active[len(active) - rpm] + window
            if target <= candidate:
                return candidate
            candidate = target

    def reserve(
        self,
        endpoint: ModelEndpoint,
        estimated_input_tokens: int,
        *,
        key_id: str = "",
        token_event: bool = True,
        now_func: Callable[[], float] = _monotonic_now,
    ) -> RateLimitTicket | None:
        """Book a departure slot **inside the lock** and return the ticket.

        Compute-slot and book-accounting are one atomic step, so two threads
        can never both see the same headroom (the check-then-act race the old
        wait/sleep/record loop had). The caller sleeps outside the lock via
        :meth:`wait_for` and fires at ``ticket.depart_at``. TPM-bearing
        bookings are FIFO per bucket: a later reservation never departs before
        an earlier one. RPM-only notes (``token_event=False``) wait only for
        real RPM headroom -- a sticky retry of an in-flight call must not
        queue behind sibling windows' future TPM bookings.
        """

        if not self.enabled:
            return None
        tokens = max(0, int(estimated_input_tokens))
        with self._lock:
            now = now_func()
            bucket_key = endpoint_key(endpoint, key_id)
            bucket = self._bucket(bucket_key)
            self._prune(bucket, now)
            exhausted = self.is_daily_exhausted(endpoint, key_id=key_id)
            if exhausted:
                # Mirrors the old wait_seconds early-out: exhaustion is the
                # rotation layer's concern, not a reason to queue here.
                depart_at = now
            else:
                limits = self.limits_for(endpoint)
                if token_event:
                    depart_at = self._booking_slot(bucket, limits, now, tokens)
                else:
                    # A zero-token note consumes no TPM: only real RPM
                    # headroom (requests departed by the candidate time) may
                    # delay it.
                    depart_at = self._rpm_only_slot(bucket, limits, now)
            if token_event and not exhausted:
                depart_at = max(depart_at, bucket.last_depart)
                bucket.last_depart = depart_at
                bucket.request_times.append(depart_at)
            else:
                # RPM-only notes (the sticky retries of a call already in
                # flight) and exhausted-key early-outs skip the FIFO horizon:
                # queueing them behind other calls' future TPM bookings would
                # stall a retry (or the early-out) for headroom it does not
                # consume. Inserted in timestamp order so pruning stays a
                # popleft scan.
                _insert_by_order(bucket.request_times, depart_at, lambda t: t)
            event_id: int | None = None
            if token_event:
                event_id = next(self._event_ids)
                event = _TokenEvent(ts=depart_at, tokens=tokens, event_id=event_id)
                if exhausted:
                    _insert_by_order(bucket.token_events, event, lambda e: e.ts)
                else:
                    bucket.token_events.append(event)
            return RateLimitTicket(
                bucket_key=bucket_key, depart_at=depart_at, event_id=event_id
            )

    def wait_for(
        self,
        ticket: RateLimitTicket | None,
        *,
        now_func: Callable[[], float] = _monotonic_now,
        sleep_func: Callable[[float], None] = _sleep,
    ) -> None:
        if ticket is None:
            return
        delay = ticket.depart_at - now_func()
        if delay > 0:
            # Reported here rather than at the call sites: the bucket a wait
            # belongs to is only known inside, and both callers -- the ticket
            # and the post-request note -- pass through this one place.
            current_reporter().debug(
                "rate limit wait",
                {"scope": ticket.bucket_key, "seconds": round(delay, 2)},
            )
            sleep_func(delay)

    def acquire(
        self,
        endpoint: ModelEndpoint,
        estimated_input_tokens: int,
        *,
        key_id: str = "",
        now_func: Callable[[], float] = _monotonic_now,
        sleep_func: Callable[[float], None] = _sleep,
    ) -> RateLimitTicket | None:
        """Reserve, then sleep to the booked slot (ticketed acquire)."""

        ticket = self.reserve(
            endpoint, estimated_input_tokens, key_id=key_id, now_func=now_func
        )
        self.wait_for(ticket, now_func=now_func, sleep_func=sleep_func)
        return ticket

    def note_request(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now_func: Callable[[], float] = _monotonic_now,
        sleep_func: Callable[[float], None] = _sleep,
    ) -> RateLimitTicket | None:
        """Wait for RPM headroom and record one request without TPM pre-reserve.

        Sticky retries / failed HTTP attempts still consume provider RPM (and
        even 5xx appears to burn daily quota), so each attempt after the first
        ``acquire`` must call this before the next request.
        """

        # tokens=0 keeps the TPM projection unchanged; only RPM can delay it.
        ticket = self.reserve(
            endpoint, 0, key_id=key_id, token_event=False, now_func=now_func
        )
        self.wait_for(ticket, now_func=now_func, sleep_func=sleep_func)
        return ticket

    def settle(
        self,
        endpoint: ModelEndpoint,
        *,
        actual_input_tokens: int,
        estimated_input_tokens: int,
        key_id: str = "",
        ticket: RateLimitTicket | None = None,
    ) -> None:
        """Refine a reservation with the observed input size.

        With a ticket the adjustment lands on **that call's own** token event
        (found by id in the ticket's bucket); the old last-event fallback hit
        whichever call booked most recently, which under concurrency is
        usually somebody else's. A pruned event is a no-op: the reservation
        already aged out of the window. The legacy fallback remains for
        callers without a ticket.
        """

        if not self.enabled:
            return
        delta = int(actual_input_tokens) - int(estimated_input_tokens)
        if delta == 0:
            return
        with self._lock:
            if ticket is not None:
                if ticket.event_id is None:
                    return
                bucket = self._bucket(ticket.bucket_key)
                for event in bucket.token_events:
                    if event.event_id == ticket.event_id:
                        event.tokens = max(0, event.tokens + delta)
                        return
                return
            bucket = self._bucket(endpoint_key(endpoint, key_id))
            if bucket.token_events:
                last = bucket.token_events[-1]
                last.tokens = max(0, last.tokens + delta)

    def _bucket(self, key: str) -> _WindowBucket:
        if key not in self._windows:
            self._windows[key] = _WindowBucket()
        return self._windows[key]

    def _prune(self, bucket: _WindowBucket, now: float) -> None:
        cutoff = now - self.policy.window_seconds
        while bucket.request_times and bucket.request_times[0] <= cutoff:
            bucket.request_times.popleft()
        while bucket.token_events and bucket.token_events[0].ts <= cutoff:
            bucket.token_events.popleft()

    def _load_state(self) -> None:
        section = state_store.read_section(_STATE_NAMESPACE, self.state_path)
        exhausted = section.get(_DAILY_EXHAUSTED_KEY, {})
        if isinstance(exhausted, Mapping):
            self._daily_exhausted = {str(k): str(v) for k, v in exhausted.items()}
        strikes = section.get(_DAILY_STRIKES_KEY, {})
        if isinstance(strikes, Mapping):
            self._daily_strikes = {
                str(k): [float(t) for t in v]
                for k, v in strikes.items()
                if isinstance(v, list)
            }
        combo = section.get(_COMBO_COOLDOWNS_KEY, {})
        if isinstance(combo, Mapping):
            self._combo_cooldowns = {str(k): str(v) for k, v in combo.items()}

    @staticmethod
    def _adopt_unseen(
        stored: Any,
        mine: Dict[str, str],
        cleared: set[str],
    ) -> None:
        """Take on entries another process wrote, without undoing our own work.

        A plain union would resurrect what we deliberately removed -- a combo
        cooldown cleared by a successful probe would come straight back off
        disk. Only keys this process has neither written nor cleared are
        adopted; for everything else our own view is the newer one.
        """

        if not isinstance(stored, dict):
            return
        for key, value in stored.items():
            name = str(key)
            if name in mine or name in cleared:
                continue
            mine[name] = str(value)

    def _persist_state(self) -> None:
        # Locked and atomic: this file is shared with the search-key limiter,
        # each side rewriting the whole document to keep the other's section.
        #
        # Within our own section we *merge* rather than assign. The in-memory
        # copy was loaded once in __init__ and the limiter is `lru_cache`d for
        # the life of the process, so assigning wholesale erased whatever a
        # concurrently running FineSub had written since -- a genuine daily
        # lock recorded by a batch run vanished the next time the desktop app
        # noted a strike, and both then kept hammering a dead key. Merging
        # costs nothing extra: `state_section` has already re-read the document
        # under the lock, so this doubles as the read-side refresh that keeps
        # the cached snapshot from going stale.
        with state_store.state_section(_STATE_NAMESPACE, self.state_path) as section:
            self._adopt_unseen(
                section.get(_DAILY_EXHAUSTED_KEY),
                self._daily_exhausted,
                self._cleared_exhausted,
            )
            self._adopt_unseen(
                section.get(_COMBO_COOLDOWNS_KEY),
                self._combo_cooldowns,
                self._cleared_cooldowns,
            )
            section[_DAILY_EXHAUSTED_KEY] = dict(self._daily_exhausted)
            section[_DAILY_STRIKES_KEY] = {
                k: list(v) for k, v in self._daily_strikes.items()
            }
            section[_COMBO_COOLDOWNS_KEY] = dict(self._combo_cooldowns)


def estimate_call_input_tokens(
    messages: List[Mapping[str, Any]],
    *,
    file_ref: Any | None = None,
    extra_media_tokens: int = 0,
    execution_settings: Any | None = None,
    high_resolution_video: bool = False,
) -> int:
    """Upper-bound input tokens for rate-limit acquire (text + optional media)."""

    from .token_budget import default_token_counter

    counter = default_token_counter(execution_settings=execution_settings)
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += counter.count_text(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += counter.count_text(str(part.get("text", "")))
    if file_ref is not None:
        local_path = getattr(file_ref, "local_path", "") or ""
        mime_type = getattr(file_ref, "mime_type", "") or ""
        if local_path:
            try:
                # Prefer the duration the caller cut the clip to. Probing is a
                # fallback for media we did not produce: window clips are raw
                # ADTS with no container duration, so ffprobe infers one from
                # the opening frames' bitrate and a window that starts on quiet
                # audio can probe 22x long -- enough to strike every candidate
                # off the chain as ``input_limit``.
                secs = float(getattr(file_ref, "duration_seconds", 0.0) or 0.0)
                if secs <= 0:
                    from finesub.media.clips import probe_audio_duration

                    secs = float(probe_audio_duration(local_path))
                total += counter.count_audio_seconds(secs)
                if mime_type.startswith("video/"):
                    from .routing.profiles import video_tokens_per_second

                    total += int(
                        math.ceil(
                            secs
                            * video_tokens_per_second(
                                high_resolution=high_resolution_video
                            )
                        )
                    )
            except Exception:
                pass
    total += max(0, int(extra_media_tokens))
    return total
