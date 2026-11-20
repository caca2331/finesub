"""Keys set aside until a known time: one ledger in `.state` (namespace `key_freeze`).

A row is `<family>/<model>/<key id>` -> `{until, reason, since}` (UTC ISO). The ElevenLabs pool writes it when a key's
balance is spent (until the provider's refresh time); `finesub quota thaw` clears rows early. Every decision reads
the file (`state.read_section` has no cache), and nothing keeps a copy across a call, so a thaw from another process
takes effect at the next decision and is never written back. The family and model are in the key because one key
may be spent on one model and not on another.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, NamedTuple

from . import state as state_store

NAMESPACE = "key_freeze"


class Frozen(NamedTuple):
    row: str
    until: datetime
    reason: str
    since: datetime | None


def row_key(family: str, model: str, key_id: str) -> str:
    return f"{family}/{model}/{key_id}"


def _now(now: datetime | None) -> datetime:
    """A naive moment is taken as UTC, as the ledgers that write here always meant it."""
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _when(value) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _frozen(row: str, value, now: datetime) -> Frozen | None:
    if not isinstance(value, dict):
        return None
    until = _when(value.get("until"))
    if until is None or until <= now:
        return None
    return Frozen(row, until, str(value.get("reason") or ""), _when(value.get("since")))


def frozen(row: str, *, now: datetime | None = None, state_path=None) -> Frozen | None:
    """The row's freeze if it still holds; None when it never was, has passed, or was thawed."""
    return _frozen(row, state_store.read_section(NAMESPACE, state_path).get(row), _now(now))


def freeze(row: str, until: datetime, reason: str, *, now: datetime | None = None, state_path=None) -> None:
    moment = _now(now)
    with state_store.state_section(NAMESPACE, state_path) as section:
        for stale in [k for k, v in section.items() if _frozen(k, v, moment) is None]:
            del section[stale]
        section[row] = {"until": until.astimezone(timezone.utc).isoformat(timespec="seconds"), "reason": reason,
                        "since": moment.isoformat(timespec="seconds")}


def listing(*, now: datetime | None = None, state_path=None) -> list[Frozen]:
    """Every freeze that still holds, by row."""
    moment = _now(now)
    found = (_frozen(k, v, moment) for k, v in sorted(state_store.read_section(NAMESPACE, state_path).items()))
    return [f for f in found if f is not None]


def thaw(rows: Iterable[str], *, state_path=None) -> list[str]:
    """Clear these rows; returns the ones that were there."""
    wanted = set(rows)
    with state_store.state_section(NAMESPACE, state_path) as section:
        gone = sorted(k for k in section if k in wanted)
        for k in gone:
            del section[k]
    return gone
