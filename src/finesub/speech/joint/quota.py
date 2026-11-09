"""Which key a cloud call goes out on, and the local account of each key's day.

One key is used until its day is spent, then the next in file order -- never
alternating, so a failure stays attributable to one key. "Spent" is recorded
per (provider, model, key): 35t's daily requests are not the LLM layer's, and
a key spent on one model may still serve another. The account lives in the
machine-local state file (`finesub.state`, namespace `cloud_asr`) and resets
when the provider's quota day turns over.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from finesub import state as state_store
from finesub.api_keys import ApiKeyEntry

NAMESPACE = "cloud_asr"

#: The zone whose midnight starts a provider's quota day.
PACIFIC = "America/Los_Angeles"  # Gemini
BEIJING = "Asia/Shanghai"  # MOSI
_FIXED = {PACIFIC: -8, BEIJING: 8}


class QuotaExhausted(RuntimeError):
    """Every key of the pool has spent its day (or none fits the call)."""


def quota_day(zone: str, now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        local = moment.astimezone(ZoneInfo(zone))
    except Exception:
        # No tz database (Windows without tzdata): standard time. For Pacific
        # that turns the day over an hour late in summer -- a key waits an hour
        # longer, it is never used before it refreshed.
        local = moment.astimezone(timezone(timedelta(hours=_FIXED[zone])))
    return local.date().isoformat()


class KeyLedger:
    def __init__(self, provider: str, model: str, zone: str, *, state_path=None, clock: Callable[[], datetime] | None = None):
        self.provider, self.model, self.zone = provider, model, zone
        self._state_path = state_path
        self._clock = clock

    def _key(self, entry: ApiKeyEntry) -> str:
        return f"{self.provider}/{self.model}/{entry.key_id}"

    def _today(self) -> str:
        return quota_day(self.zone, self._clock() if self._clock else None)

    def usage(self, entry: ApiKeyEntry) -> dict:
        """{calls, seconds, exhausted} for today; zeros when the day turned over."""

        return _usage(state_store.read_section(NAMESPACE, self._state_path).get(self._key(entry)), self._today())

    def _update(self, entry: ApiKeyEntry, change: Callable[[dict], None]) -> None:
        today = self._today()
        with state_store.state_section(NAMESPACE, self._state_path) as section:
            row = section.get(self._key(entry))
            if not isinstance(row, dict) or row.get("day") != today:
                row = {"day": today, "calls": 0, "seconds": 0.0, "exhausted": False}
            change(row)
            section[self._key(entry)] = row

    def record(self, entry: ApiKeyEntry, seconds: float) -> None:
        """One billed call of `seconds` of audio."""

        self._update(entry, lambda row: _count(row, +1, seconds))

    def reserve(self, entries: Sequence[ApiKeyEntry], *, fits: Callable[[ApiKeyEntry, dict], bool], seconds: float,
                what: str) -> tuple[ApiKeyEntry, str]:
        """`pick_key` and `record` in one locked read-modify-write: the call is on the account before it is
        made, so batch items calling at once cannot each see the same room left on a key. Returns the key
        and the quota day the call went on, which `refund` needs to take back a call the server did not
        take. `fits(entry, usage)` says whether that key has room for the call."""

        if not entries:
            raise QuotaExhausted(f"{what}: no key is configured")
        today = self._today()
        with state_store.state_section(NAMESPACE, self._state_path) as section:
            for entry in entries:
                row = section.get(self._key(entry))
                usage = _usage(row, today)
                if not usage["exhausted"] and fits(entry, usage):
                    row = row if isinstance(row, dict) and row.get("day") == today else {
                        "day": today, "calls": 0, "seconds": 0.0, "exhausted": False}
                    _count(row, +1, seconds)
                    section[self._key(entry)] = row
                    return entry, today
        raise QuotaExhausted(f"{what}: every key has spent today's quota")

    def refund(self, entry: ApiKeyEntry, seconds: float, *, day: str, exhausted: bool = False) -> None:
        """Take back a call reserved on quota `day`; `exhausted` also marks the key spent, in the same write.
        Once that day has turned over there is nothing to take back: the new day's account holds other calls,
        and a refusal of yesterday's call says nothing about today's quota."""

        with state_store.state_section(NAMESPACE, self._state_path) as section:
            row = section.get(self._key(entry))
            if not isinstance(row, dict) or row.get("day") != day:
                return
            _count(row, -1, seconds)
            row["calls"], row["seconds"] = max(0, row["calls"]), max(0.0, row["seconds"])
            if exhausted:
                row["exhausted"] = True
            section[self._key(entry)] = row

    def mark_exhausted(self, entry: ApiKeyEntry) -> None:
        self._update(entry, lambda row: row.__setitem__("exhausted", True))


def _usage(row, today: str) -> dict:
    if not isinstance(row, dict) or row.get("day") != today:
        return {"calls": 0, "seconds": 0.0, "exhausted": False}
    return {"calls": int(row.get("calls", 0)), "seconds": float(row.get("seconds", 0.0)),
            "exhausted": bool(row.get("exhausted", False))}


def _count(row: dict, calls: int, seconds: float) -> None:
    row["calls"] = int(row.get("calls", 0)) + calls
    row["seconds"] = round(float(row.get("seconds", 0.0)) + calls * seconds, 3)


def pick_key(
    entries: Sequence[ApiKeyEntry],
    ledger: KeyLedger,
    *,
    what: str,
) -> ApiKeyEntry:
    """The first key (file order) not spent today."""

    if not entries:
        raise QuotaExhausted(f"{what}: no key is configured")
    for entry in entries:
        usage = ledger.usage(entry)
        if not usage["exhausted"]:
            return entry
    raise QuotaExhausted(f"{what}: every key has spent today's quota")
