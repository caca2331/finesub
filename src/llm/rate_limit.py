"""Model-level RPM/TPM(input) rate limiting and daily exhaustion tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

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

# Lock a key/endpoint as daily-exhausted only after a *sustained* run of per-day
# 429s — not a lone or bursty one. Gemini's free-tier PerDay signal flickers
# (a key 429s "PerDay" one moment and answers the next), so require several hits
# spread over a couple of minutes before believing the day is genuinely spent.
DAILY_STRIKE_COUNT = 3
DAILY_STRIKE_SPAN_SECONDS = 300.0


def _default_state_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".state"


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

        Locking requires ``DAILY_STRIKE_COUNT`` hits whose oldest-of-the-run to
        newest spans at least ``DAILY_STRIKE_SPAN_SECONDS`` — so a lone/bursty
        PerDay 429 never poisons the whole day. Returns True iff the endpoint is
        now marked daily-exhausted.
        """
        key = endpoint_key(endpoint, key_id)
        now = now if now is not None else time.time()
        strikes = self._daily_strikes.setdefault(key, [])
        strikes.append(now)
        if len(strikes) > DAILY_STRIKE_COUNT:
            del strikes[:-DAILY_STRIKE_COUNT]
        if (
            len(strikes) >= DAILY_STRIKE_COUNT
            and (strikes[-1] - strikes[0]) >= DAILY_STRIKE_SPAN_SECONDS
        ):
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
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, Mapping):
            return
        section = data.get(_STATE_NAMESPACE, {})
        if not isinstance(section, Mapping):
            return
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

    def _persist_state(self) -> None:
        existing: Dict[str, Any] = {}
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = dict(raw)
            except (OSError, json.JSONDecodeError):
                existing = {}
        section = existing.get(_STATE_NAMESPACE, {})
        if not isinstance(section, dict):
            section = {}
        section = dict(section)
        section[_DAILY_EXHAUSTED_KEY] = dict(self._daily_exhausted)
        section[_DAILY_STRIKES_KEY] = {
            k: list(v) for k, v in self._daily_strikes.items()
        }
        existing[_STATE_NAMESPACE] = section
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


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
                from utils.audio import get_audio_duration_sec

                secs = float(get_audio_duration_sec(local_path))
                total += counter.count_audio_seconds(secs)
                if mime_type.startswith("video/"):
                    from .profiles import video_tokens_per_second

                    total += int(math.ceil(secs * video_tokens_per_second()))
            except Exception:
                pass
    total += max(0, int(extra_media_tokens))
    return total
