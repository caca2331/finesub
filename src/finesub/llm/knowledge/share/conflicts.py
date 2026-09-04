"""The pull conflicts a pull leaves behind, and what was decided about them.

``sync`` reports what it could not settle; this module is where that report
stops being a line on a terminal. One append-only ledger beside the push log
in the knowledge root, last row per conflict wins -- the same idiom
``share status`` already uses to find a push record, for the same reason: a
ledger that is rewritten whole loses whichever concurrent writer went first,
and these rows are written by a CLI a user may well run twice.

Identity is the disagreement itself: ``(remote, canonical_id, field, local,
incoming)``. That is deliberate rather than a node/field pair --

* the pull re-reports an unresolved conflict on EVERY pull (``sync``'s module
  docstring: the remote payload never lands locally, so the base never
  matches and the conflict comes back). Keying on the values means those
  repeats collapse onto one open row instead of growing the file once per
  pull;
* if either side then moves, it is a genuinely different question and gets
  its own row rather than silently inheriting the old verdict.

Which is also why ``dismissed`` is sticky and ``resolved`` is not. A dismissal
says "I looked, local is right, stop asking" -- and since the same conflict is
re-reported forever, re-opening it would make the verdict worthless. A
resolution says "I changed something so this goes away"; if the identical
conflict comes back, it did not go away, and the ledger should say so.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .sync import FieldConflict

CONFLICT_LOG_FILENAME = "share-conflicts.jsonl"

OPEN = "open"
RESOLVED = "resolved"
DISMISSED = "dismissed"

#: Verdicts a human or a repair round may record. ``open`` is not among them:
#: it is what the pull writes, never what a decision produces.
VERDICTS = (RESOLVED, DISMISSED)


def conflict_id(record: Mapping[str, Any]) -> str:
    """Stable identity for one disagreement (see the module docstring)."""

    material = json.dumps(
        [
            str(record.get("remote") or ""),
            str(record.get("canonical_id") or ""),
            str(record.get("field") or ""),
            record.get("local"),
            record.get("incoming"),
        ],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _log_path(root: str | Path) -> Path:
    return Path(root) / CONFLICT_LOG_FILENAME


def _rows(root: str | Path) -> list[dict[str, Any]]:
    path = _log_path(root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            # A hand-edited or truncated line must not make the whole ledger
            # unreadable: the rest of it is still every other conflict.
            continue
        if isinstance(row, dict) and row.get("conflict_id"):
            rows.append(row)
    return rows


def _append(root: str | Path, row: Mapping[str, Any]) -> None:
    path = _log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def latest(root: str | Path) -> dict[str, dict[str, Any]]:
    """Every conflict this ledger knows, at its most recent row."""

    state: dict[str, dict[str, Any]] = {}
    for row in _rows(root):
        state[str(row["conflict_id"])] = row
    return state


def open_conflicts(root: str | Path, *, remote: str = "") -> list[dict[str, Any]]:
    """Unresolved conflicts, oldest first (the order they were first seen)."""

    rows = [row for row in latest(root).values() if row.get("status") == OPEN]
    if remote:
        rows = [row for row in rows if row.get("remote") == remote]
    return sorted(rows, key=lambda row: (row.get("first_seen_rev") or 0, row.get("conflict_id")))


def record_conflicts(
    root: str | Path, conflicts: Iterable[FieldConflict]
) -> tuple[list[dict[str, Any]], int]:
    """Write down what a pull could not settle.

    Returns ``(newly opened rows, count already open)`` so the caller can say
    "3 new, 5 still open" instead of implying every pull found everything
    afresh. A dismissed conflict is left alone -- that verdict is sticky.
    """

    known = latest(root)
    opened: list[dict[str, Any]] = []
    still_open = 0
    for conflict in conflicts:
        record = conflict.to_dict()
        identity = conflict_id(record)
        prior = known.get(identity)
        if prior is not None:
            if prior.get("status") == DISMISSED:
                continue
            if prior.get("status") == OPEN:
                still_open += 1
                continue
        row = {
            **record,
            "conflict_id": identity,
            "status": OPEN,
            "first_seen_rev": record.get("pulled_rev"),
            # Only set when this is a re-open, so the plain case stays plain.
            **({"reopened_after": prior.get("status")} if prior is not None else {}),
        }
        _append(root, row)
        known[identity] = row
        opened.append(row)
    return opened, still_open


def record_verdict(
    root: str | Path,
    conflict_id_value: str,
    *,
    status: str,
    reason: str,
    task_id: str = "",
    applied_rev: int | None = None,
) -> dict[str, Any]:
    """Close one conflict. The row keeps the conflict's own fields so the
    ledger stays readable on its own -- someone reading it should not have to
    join two rows to learn what was decided about what."""

    if status not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, not {status!r}")
    prior = latest(root).get(conflict_id_value)
    if prior is None:
        raise KeyError(f"no conflict {conflict_id_value!r} in the ledger")
    row = {
        **{key: value for key, value in prior.items() if key != "reopened_after"},
        "status": status,
        "reason": reason,
        "task_id": task_id,
        "applied_rev": applied_rev,
    }
    _append(root, row)
    return row


def describe(row: Mapping[str, Any]) -> str:
    """One line for a listing: what disagrees, and with what."""

    label = f"{row.get('label') or row.get('canonical_id')}.{row.get('field')}"
    if row.get("kind") == "prose":
        return f"{row.get('conflict_id')}  {label}  [prose] pending manual merge"
    return (
        f"{row.get('conflict_id')}  {label}  local={row.get('local')!r}"
        f" remote={row.get('incoming')!r}"
        + ("" if row.get("had_base") else "  (no common base)")
    )
