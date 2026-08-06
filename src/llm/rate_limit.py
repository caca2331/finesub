"""Model-level RPM/TPM(input) rate limiting and daily exhaustion tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from asr_playground import state as state_store
from asr_playground.paths import resolve_state_file

from .config import ModelEndpoint, RateLimitPolicy
from .model_catalog import get_model_catalog_entry_for_tier

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

    Parses mainstream provider formats (Gemini retryDelay, OpenAI/Anthropic
    Retry-After, generic "retry in Xs" text). Returns 0.0 when no wait hint
    is found. The caller clamps the value; this function only extracts.
    """

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

    base = f"{endpoint.provider_tier}|{endpoint.litellm_model}"
    return f"{base}|{key_id}" if key_id else base


def _pacific_calendar_day(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(_PACIFIC).date().isoformat()


@dataclass
class _WindowBucket:
    request_times: Deque[float] = field(default_factory=deque)
    token_events: Deque[Tuple[float, int]] = field(default_factory=deque)


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
        self._windows: Dict[str, _WindowBucket] = {}
        self._daily_exhausted: Dict[str, str] = {}
        self._daily_strikes: Dict[str, List[float]] = {}
        self._combo_cooldowns: Dict[str, str] = {}
        self._load_state()

    def limits_for(self, endpoint: ModelEndpoint) -> RateLimitLimits:
        entry = get_model_catalog_entry_for_tier(
            endpoint.litellm_model, endpoint.provider_tier
        )
        if entry is None:
            raise ValueError(
                f"No model_catalog.psv row for {endpoint.provider_tier} + "
                f"{endpoint.litellm_model}"
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
        self._daily_exhausted[key] = now.isoformat(timespec="milliseconds")
        self._daily_strikes.pop(key, None)
        self._persist_state()

    def note_daily_quota_hit(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now: Optional[float] = None,
    ) -> bool:
        """Record a per-day 429 and lock only once the strike gate confirms it.

        Locking requires ``DAILY_STRIKE_COUNT`` consecutive hits (a lone PerDay
        429 never poisons the whole day; success clears the streak). Returns
        True iff the endpoint is now marked daily-exhausted.
        """
        key = endpoint_key(endpoint, key_id)
        now = now if now is not None else time.time()
        strikes = self._daily_strikes.setdefault(key, [])
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
            self._combo_cooldowns.pop(key, None)
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
        self._combo_cooldowns[key] = now.isoformat(timespec="milliseconds")
        self._persist_state()

    def clear_combo_cooldown(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
    ) -> None:
        key = endpoint_key(endpoint, key_id)
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
        if not self.enabled:
            return 0.0
        if self.is_daily_exhausted(endpoint, key_id=key_id):
            return 0.0
        now = now if now is not None else _monotonic_now()
        limits = self.limits_for(endpoint)
        bucket = self._bucket(endpoint_key(endpoint, key_id))
        self._prune(bucket, now)
        waits: List[float] = []

        rpm = limits.effective_rpm
        if len(bucket.request_times) >= rpm:
            oldest = bucket.request_times[0]
            waits.append(max(0.0, self.policy.window_seconds - (now - oldest)))

        tpm = limits.effective_tpm
        if tpm >= 0:
            current_tokens = sum(tokens for _, tokens in bucket.token_events)
            projected = current_tokens + max(0, int(estimated_input_tokens))
            if projected > tpm and bucket.token_events:
                needed = projected - tpm
                running = 0
                for ts, tokens in bucket.token_events:
                    running += tokens
                    if running >= needed:
                        waits.append(max(0.0, self.policy.window_seconds - (now - ts)))
                        break

        return max(waits, default=0.0)

    def acquire(
        self,
        endpoint: ModelEndpoint,
        estimated_input_tokens: int,
        *,
        key_id: str = "",
        now_func: Callable[[], float] = _monotonic_now,
        sleep_func: Callable[[float], None] = _sleep,
    ) -> None:
        if not self.enabled:
            return
        while True:
            now = now_func()
            wait = self.wait_seconds(endpoint, estimated_input_tokens, key_id=key_id, now=now)
            if wait <= 0:
                break
            sleep_func(wait)
        self._record_acquire(endpoint, estimated_input_tokens, key_id=key_id, now=now_func())

    def note_request(
        self,
        endpoint: ModelEndpoint,
        *,
        key_id: str = "",
        now_func: Callable[[], float] = _monotonic_now,
        sleep_func: Callable[[float], None] = _sleep,
    ) -> None:
        """Wait for RPM headroom and record one request without TPM pre-reserve.

        Sticky retries / failed HTTP attempts still consume provider RPM (and
        even 5xx appears to burn daily quota), so each attempt after the first
        ``acquire`` must call this before the next request.
        """

        if not self.enabled:
            return
        while True:
            now = now_func()
            # tokens=0 → TPM projection unchanged; only RPM can force a wait.
            wait = self.wait_seconds(endpoint, 0, key_id=key_id, now=now)
            if wait <= 0:
                break
            sleep_func(wait)
        now = now_func()
        bucket = self._bucket(endpoint_key(endpoint, key_id))
        self._prune(bucket, now)
        bucket.request_times.append(now)

    def settle(
        self,
        endpoint: ModelEndpoint,
        *,
        actual_input_tokens: int,
        estimated_input_tokens: int,
        key_id: str = "",
    ) -> None:
        if not self.enabled:
            return
        bucket = self._bucket(endpoint_key(endpoint, key_id))
        delta = int(actual_input_tokens) - int(estimated_input_tokens)
        if delta != 0 and bucket.token_events:
            last_ts, last_tokens = bucket.token_events[-1]
            bucket.token_events[-1] = (last_ts, max(0, last_tokens + delta))

    def _record_acquire(
        self,
        endpoint: ModelEndpoint,
        estimated_input_tokens: int,
        *,
        key_id: str = "",
        now: float,
    ) -> None:
        bucket = self._bucket(endpoint_key(endpoint, key_id))
        self._prune(bucket, now)
        bucket.request_times.append(now)
        bucket.token_events.append((now, max(0, int(estimated_input_tokens))))

    def _bucket(self, key: str) -> _WindowBucket:
        if key not in self._windows:
            self._windows[key] = _WindowBucket()
        return self._windows[key]

    def _prune(self, bucket: _WindowBucket, now: float) -> None:
        cutoff = now - self.policy.window_seconds
        while bucket.request_times and bucket.request_times[0] <= cutoff:
            bucket.request_times.popleft()
        while bucket.token_events and bucket.token_events[0][0] <= cutoff:
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

    def _persist_state(self) -> None:
        # Locked and atomic: this file is shared with the search-key limiter,
        # each side rewriting the whole document to keep the other's section.
        with state_store.state_section(_STATE_NAMESPACE, self.state_path) as section:
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
) -> int:
    """Upper-bound input tokens for rate-limit acquire (text + optional media)."""

    from .token_budget import default_token_counter

    counter = default_token_counter()
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
                from asr_playground.media.clips import probe_audio_duration

                secs = float(probe_audio_duration(local_path))
                total += counter.count_audio_seconds(secs)
                if mime_type.startswith("video/"):
                    from .profiles import video_tokens_per_second

                    total += int(math.ceil(secs * video_tokens_per_second()))
            except Exception:
                pass
    total += max(0, int(extra_media_tokens))
    return total
