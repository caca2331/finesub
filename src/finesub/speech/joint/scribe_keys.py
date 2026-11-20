"""Which ElevenLabs key a Scribe call goes out on: the `ELEVENLABS` pool, front to back (owner 2026-11-11).

Up to three keys, in `.env` order (or `[pools] elevenlabs`); the last is taken as the paid one (pay-as-you-go), as
MOSS Pro's is. One call goes whole to one key. A key that may read its subscription (the `user_read` permission)
is asked its balance before each call: credits left / the key's credits per audio hour = the hours it has left.
A key short of the call is passed over for this call; one with less than SPENT_BELOW left is frozen until its
credits refresh (`key_freeze`), and not asked again before then. The last key is never passed over for its
balance, a free one short of the call included (owner: pay-as-you-go may take the rest; if not, a spent free
balance is spent once more, no harm). A key whose balance cannot be read (no user_read, or the subscription
call failed) is used until refused. Any refused key is left out for the rest of this process, not frozen (owner
2026-11-11: a key with User: Read but no Speech to Text shows room and would be tried first for every source of a
batch). Every refusal warns once per key and process, with the hint to enable User: Read where that is why the
balance was not read. "Spent" is read off the balance only: the speech-to-text refusal for an empty balance is
not documented (it may be the same 401 as a bad key).

Credits per hour differ by plan: the free tier's 10k credits buy about 4.5 h (2222/h); `starter` measured 1000/h;
other plans start from 1100 (owner 2026-11-11) and every call of LEARN_MIN_SECONDS or more corrects the key's rate
from its balance before and after (only a rise, within RATE_CLAMP of the start; a new plan starts over). Within
this process a key's checks, balance and call run one at a time, so the change is that call's alone; a call from
another process at the same moment can still land in it, which the clamp bounds. A refusal of a key that showed
room (no Speech to Text permission, or another process spent it first) moves on to the next key; the last key
refused stops the run with its name and refresh time, nothing frozen.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from finesub import api_keys, key_freeze
from finesub import state as state_store
from finesub.api_http import CONNECT_TIMEOUT_SECONDS, RETRYABLE_TRANSPORT, api_http_client
from finesub.reporting import current_reporter

from .quota import NAMESPACE as ACCOUNT
from .scribe import KEY_ENV, MODEL, USD_PER_HOUR, Caller, ScribeError, ScribeRefused

FAMILY = "elevenlabs"
SUBSCRIPTION_API = "https://api.elevenlabs.io/v1/user/subscription"
FREE_TIER = "free"
#: Credits per audio hour to start from: the free tier's (10k credits, about 4.5 h a month), every other plan's.
FREE_RATE = 2222.0
OTHER_RATE = 1100.0
#: A learned rate stays within these multiples of the one it started from.
RATE_CLAMP = (0.5, 2.0)
#: A call this long or longer corrects the rate (the balance counts whole credits; a short re-ask moves it a few).
LEARN_MIN_SECONDS = 300.0
#: A key with less than this left is spent: frozen until its credits refresh.
SPENT_BELOW = 300.0
#: How long a spent key is frozen when its subscription gives no refresh time.
FREEZE_WITHOUT_RESET = timedelta(hours=24)

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
#: Keys refused in this process: left out for the rest of it.
_REFUSED: set[str] = set()
#: Keys whose refusal this process has warned about.
_WARNED: set[str] = set()


@dataclass(frozen=True)
class Balance:
    tier: str
    used: int
    limit: int
    reset: datetime | None

    def start_rate(self) -> float:
        return FREE_RATE if self.tier == FREE_TIER else OTHER_RATE


def subscription(key: str) -> tuple[str, Balance | None]:
    """("ok", the balance) | ("no_permission", None) for a key without `user_read` | ("unknown", None)."""
    import httpx

    try:
        with api_http_client(httpx.Timeout(30.0, connect=CONNECT_TIMEOUT_SECONDS)) as client:
            r = client.get(SUBSCRIPTION_API, headers={"xi-api-key": key})
    except RETRYABLE_TRANSPORT:
        return "unknown", None
    if r.status_code in (401, 403) and "missing_permissions" in r.text:
        return "no_permission", None
    if r.status_code != 200:
        return "unknown", None
    try:
        body = r.json()
        reset = body.get("next_character_count_reset_unix")
        return "ok", Balance(str(body.get("tier") or ""), int(body["character_count"]), int(body["character_limit"]),
                             datetime.fromtimestamp(int(reset), timezone.utc) if reset else None)
    except (ValueError, KeyError, TypeError):
        return "unknown", None


def _lock(key_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key_id, threading.Lock())


class KeyPool:
    """A Scribe caller over the pool: `pool(audio, language=..., seconds=..., whole=...)` is the answer from the
    first key that takes it. `note` is added to the line that says what a whole-source call costs."""

    def __init__(self, entries: Sequence[api_keys.ApiKeyEntry] | None = None, *, note: str = "",
                 fetch: Callable[[str], tuple[str, Balance | None]] = subscription,
                 make_caller: Callable[[str], Callable[..., dict]] = Caller, state_path=None,
                 clock: Callable[[], datetime] | None = None, announce: Callable[[], None] | None = None):
        self._entries = entries
        self.note = note
        self._announce = announce
        self._fetch, self._make_caller = fetch, make_caller
        self._state_path, self._clock = state_path, clock

    def entries(self) -> Sequence[api_keys.ApiKeyEntry]:
        if self._entries is None:
            self._entries = api_keys.resolve_pool(api_keys.ELEVENLABS_POOL, api_keys.read_dotenv())
        return self._entries

    def _now(self) -> datetime:
        return self._clock() if self._clock else datetime.now(timezone.utc)

    def _row(self, entry) -> str:
        return key_freeze.row_key(FAMILY, MODEL, entry.key_id)

    @staticmethod
    def _rate_row(entry) -> str:
        """The key's learned rate in the account (`cloud_asr`): not a day's usage, so `finesub quota` skips it."""
        return f"{FAMILY}/{MODEL}/{entry.key_id}"

    def rate(self, entry, balance: Balance) -> float:
        row = state_store.read_section(ACCOUNT, self._state_path).get(self._rate_row(entry))
        if isinstance(row, dict) and row.get("tier") == balance.tier and isinstance(row.get("rate"), (int, float)):
            return float(row["rate"])
        return balance.start_rate()

    def left_seconds(self, entry, balance: Balance) -> float:
        return max(0, balance.limit - balance.used) / self.rate(entry, balance) * 3600

    def __call__(self, audio: Path, *, language: str | None, seconds: float, whole: bool = False) -> dict:
        entries = self.entries()
        if not entries:
            raise ScribeError(f"--asr scribe needs {KEY_ENV} in .env (and [providers] elevenlabs not false)")
        passed: list[str] = []
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            with _lock(entry.key_id):  # the checks too: a call that waited here sees what the one before it found
                if entry.key_id in _REFUSED:
                    passed.append(f"{entry.label}: refused earlier in this run")
                    continue
                held = None if last else key_freeze.frozen(self._row(entry), now=self._now(),
                                                           state_path=self._state_path)
                if held is not None:
                    passed.append(f"{entry.label}: frozen until {held.until:%Y-%m-%d %H:%M} UTC ({held.reason})")
                    continue
                state, balance = self._fetch(entry.key)
                if balance is not None and not last:
                    left = self.left_seconds(entry, balance)
                    if left < SPENT_BELOW:
                        self._freeze(entry, balance, left)
                        passed.append(f"{entry.label}: spent, frozen until its credits refresh")
                        continue
                    if left < seconds:
                        passed.append(f"{entry.label}: about {left / 3600:.2f} h left, short of {seconds / 3600:.2f} h")
                        continue
                if whole:
                    self._say_cost(entry, balance, seconds, last)
                if self._announce is not None:
                    self._announce()
                try:
                    answer = self._make_caller(entry.key)(audio, language=language)
                except ScribeRefused as error:
                    _REFUSED.add(entry.key_id)
                    said = self._refused(entry, error, state)
                    passed.append(said)
                    if entry.key_id not in _WARNED:
                        _WARNED.add(entry.key_id)
                        current_reporter().warning("scribe-key-refused", f"Scribe: key {said}",
                                                   impact="这次上传换下一把 key（或停下）")
                    if last:
                        raise ScribeError(f"Scribe: every {KEY_ENV} key was passed over or refused: "
                                          + "; ".join(passed) + self._refresh(balance)) from None
                    continue
                if balance is not None and seconds >= LEARN_MIN_SECONDS:
                    self._learn(entry, balance, seconds)
                return answer
        raise ScribeError(f"Scribe: every {KEY_ENV} key was passed over or refused: " + "; ".join(passed))

    def _freeze(self, entry, balance: Balance, left: float) -> None:
        reset = balance.reset if balance.reset and balance.reset > self._now() else None
        until = reset or self._now() + FREEZE_WITHOUT_RESET
        key_freeze.freeze(self._row(entry), until,
                          f"{max(0, balance.limit - balance.used)} credits left, about {left / 60:.0f} min",
                          now=self._now(), state_path=self._state_path)

    @staticmethod
    def _refused(entry, error: ScribeRefused, state: str) -> str:
        said = f"{entry.label} refused (HTTP {error.status}), left out for this run"
        if state == "no_permission":
            said += "; give the key the User: Read permission so its balance can be checked before a call"
        elif state != "ok":
            said += " (its balance could not be read either: check the key)"
        else:
            said += (" although its balance showed room: check that it has the Speech to Text permission (or "
                     "another run spent it first)")
        return said

    @staticmethod
    def _refresh(balance: Balance | None) -> str:
        if balance is None or balance.reset is None:
            return ""
        return f"; the last key's credits refresh {balance.reset:%Y-%m-%d %H:%M} UTC"

    def _say_cost(self, entry, balance: Balance | None, seconds: float, last: bool) -> None:
        hours = seconds / 3600
        if balance is None:
            line = (f"Warning: Scribe bills per audio hour: {hours:.2f} h on key {entry.label}, about "
                    f"${hours * USD_PER_HOUR:.2f} at ${USD_PER_HOUR:.2f}/h (2026-11 price)")
        else:
            left = self.left_seconds(entry, balance) / 3600
            if balance.tier == FREE_TIER and left >= hours:
                line = (f"Note: Scribe: {hours:.2f} h on key {entry.label}'s free credits (about {left:.1f} h left "
                        "this month by estimate)")
            elif balance.tier == FREE_TIER:
                line = (f"Warning: Scribe: key {entry.label}, the pool's last, has free credits for about {left:.1f} h "
                        f"against {hours:.2f} h of audio; uploading anyway (pay-as-you-go, if the account has it, "
                        f"bills the rest at about ${USD_PER_HOUR:.2f}/h)")
            else:
                line = (f"Warning: Scribe: {hours:.2f} h on key {entry.label} ({balance.tier} plan), about "
                        f"{hours * self.rate(entry, balance):.0f} of its credits (about {left:.1f} h left by estimate)")
        print(line + self.note, file=sys.stderr)  # product output

    def _learn(self, entry, before: Balance, seconds: float) -> None:
        state, after = self._fetch(entry.key)
        if state != "ok" or after is None or after.tier != before.tier or after.used <= before.used:
            return
        low, high = (before.start_rate() * k for k in RATE_CLAMP)
        rate = min(high, max(low, (after.used - before.used) / (seconds / 3600)))
        with state_store.state_section(ACCOUNT, self._state_path) as section:
            section[self._rate_row(entry)] = {"tier": after.tier, "rate": round(rate, 1)}
