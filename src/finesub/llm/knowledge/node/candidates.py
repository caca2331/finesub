"""Candidate decision ledger (kb-followups plan A6).

Second-pass judgment candidates used to be recomputed by every ``build_plan``
call with no memory: a candidate the model had already dismissed — or flagged
as needing a human — reappeared every round. This ledger gives each candidate
a durable decision keyed by a STABLE identity:

* ``candidate_key`` — kind + subject/node/item identity, no content and no
  task_id: the same underlying question always maps to the same key;
* ``content_digest`` — the candidate's content fields: when the content
  changes, standing decisions are superseded and the candidate is visible
  again.

States: ``pending_human`` (the model could not decide and verification has no
path — waits for a person), ``resolved`` (with a resolution: ``dismissed`` /
``applied`` / ``human``), ``superseded`` (content changed after the decision).
This is deliberately a SEPARATE ledger from claim ``evidence``: "no external
source can verify this claim" (``unverifiable``) and "the model cannot decide
this edit" are different verdicts with different lifecycles.

Dry-run discipline: nothing here is written unless the repair CLI ran with
``--apply`` (or a human resolves via the ``candidates`` CLI).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .model import digest
from .store import KnowledgeStore

#: identity fields — everything else on a candidate is content
_KEY_FIELDS = ("kind", "subject_id", "node", "item_id")
_DISPLAY_FIELDS = ("hint", "subject")  # advisory text: not identity, not content

CANDIDATE_STATUSES = ("pending_human", "resolved", "superseded")
CANDIDATE_RESOLUTIONS = ("dismissed", "applied", "human")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _candidate_snapshot(candidate: Mapping[str, Any], *, limit: int = 2000) -> str:
    """A bounded, ALWAYS-valid JSON snapshot of one candidate.

    Slicing the serialized text (the first cut) could truncate mid-string and
    break the "the column holds JSON" contract (review 2026-08-28 P2-1);
    bounding happens by shrinking FIELDS instead — long values first, then
    everything but the identity/label fields — and re-serializing whole."""

    import json

    data = dict(candidate)
    text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    trimmed = {
        key: (value[:200] + "…" if isinstance(value, str) and len(value) > 200 else value)
        for key, value in data.items()
    }
    text = json.dumps(trimmed, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    keep = {
        key: trimmed[key]
        for key in ("kind", "subject", "subject_id", "node", "item_id",
                    "term", "label", "value", "target")
        if key in trimmed
    }
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


def candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, str]:
    """``(candidate_key, content_digest)`` for one candidate-scan row."""

    key = digest({name: str(candidate.get(name, "") or "") for name in _KEY_FIELDS})
    content = digest({
        name: value
        for name, value in sorted(candidate.items())
        if name not in _KEY_FIELDS and name not in _DISPLAY_FIELDS
    })
    return key, content


def record_candidate_decision(
    store: KnowledgeStore,
    *,
    candidate_key: str,
    content_digest: str,
    status: str,
    resolution: str = "",
    reason: str = "",
    task_id: str = "",
    candidate: Mapping[str, Any] | None = None,
    missing: str = "",
) -> None:
    """Book one decision; standing rows of the same key are superseded when
    their digest differs (content moved on), or updated in place when it
    matches (a human resolving a pending row, a re-run refreshing a reason).
    ``candidate`` is a JSON snapshot so a pending row is legible without
    re-running the scan; ``missing`` is the session's stated evidence gap.

    The statements run in the caller's transaction when one is open (the
    repair booking wraps all its verdicts in one — review 2026-08-28 P1-3:
    a half-written ledger must not survive a crash), else autocommit."""

    if status not in ("pending_human", "resolved"):
        raise ValueError(f"unknown candidate status {status!r}")
    if status == "resolved" and resolution not in CANDIDATE_RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r}")
    snapshot = _candidate_snapshot(candidate) if candidate else ""
    now = _now()
    store.conn.execute(
        "UPDATE candidate_decisions SET status='superseded'"
        " WHERE candidate_key=? AND status != 'superseded' AND content_digest != ?",
        (candidate_key, content_digest),
    )
    row = store.conn.execute(
        "SELECT decision_id FROM candidate_decisions"
        " WHERE candidate_key=? AND content_digest=? AND status != 'superseded'",
        (candidate_key, content_digest),
    ).fetchone()
    if row is not None:
        store.conn.execute(
            "UPDATE candidate_decisions SET status=?, resolution=?, reason=?, task_id=?,"
            " resolved_at=?, candidate=CASE WHEN ?='' THEN candidate ELSE ? END,"
            " missing=? WHERE decision_id=?",
            (status, resolution, reason, task_id,
             now if status == "resolved" else "",
             snapshot, snapshot, missing, row["decision_id"]),
        )
        return
    store.conn.execute(
        "INSERT INTO candidate_decisions(candidate_key, content_digest, status, resolution,"
        " reason, task_id, created_at, resolved_at, candidate, missing)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (candidate_key, content_digest, status, resolution, reason, task_id, now,
         now if status == "resolved" else "", snapshot, missing),
    )


def resolve_candidate(
    store: KnowledgeStore,
    candidate_key: str,
    *,
    reason: str = "",
    candidates: Iterable[Mapping[str, Any]] | None = None,
) -> bool:
    """Human resolution of a pending row (CLI). Returns False when no
    non-superseded row exists for the key.

    ``candidates`` is the CURRENT scan. It matters because the usual way a
    person settles a pending row is to go fix the line — which changes the
    content digest, so re-stamping the standing row would book the decision
    against content that no longer exists and ``filter_undecided`` would keep
    surfacing the candidate forever. With the live scan in hand the decision
    is booked against what the person actually looked at; without it (the
    candidate is gone entirely, e.g. the row was deleted) the standing row is
    closed as before."""

    row = store.conn.execute(
        "SELECT decision_id, content_digest, candidate FROM candidate_decisions"
        " WHERE candidate_key=? AND status != 'superseded'"
        " ORDER BY decision_id DESC LIMIT 1",
        (candidate_key,),
    ).fetchone()
    if row is None:
        return False
    live = next(
        (c for c in (candidates or []) if candidate_identity(c)[0] == candidate_key),
        None,
    )
    if live is not None:
        record_candidate_decision(
            store,
            candidate_key=candidate_key,
            content_digest=candidate_identity(live)[1],
            status="resolved",
            resolution="human",
            reason=reason,
            candidate=live,
        )
        return True
    store.conn.execute(
        "UPDATE candidate_decisions SET status='resolved', resolution='human', reason=?,"
        " resolved_at=? WHERE decision_id=?",
        (reason, _now(), row["decision_id"]),
    )
    return True


def standing_decisions(store: KnowledgeStore) -> dict[str, dict[str, Any]]:
    """Latest non-superseded row per candidate_key."""

    rows = store.conn.execute(
        "SELECT * FROM candidate_decisions WHERE status != 'superseded'"
        " ORDER BY decision_id",
    ).fetchall()
    return {row["candidate_key"]: dict(row) for row in rows}


def pending_human(store: KnowledgeStore) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in store.conn.execute(
            "SELECT * FROM candidate_decisions WHERE status='pending_human' ORDER BY decision_id",
        )
    ]


def pending_human_reconciled(store: KnowledgeStore) -> list[dict[str, Any]]:
    """Pending rows annotated against the CURRENT scan (review 2026-08-28
    P2-4: a pending row can go stale when the candidate's content moved on or
    the underlying condition was fixed by other edits). Each row gains
    ``freshness``: ``current`` (scan still produces this exact content),
    ``content-changed`` (same question, new content — the next repair round
    re-surfaces it and this row will be superseded at booking), or ``gone``
    (the scan no longer finds the candidate; safe to resolve)."""

    from .scan import scan_candidates

    current: dict[str, str] = {}
    for cand in scan_candidates(store).candidates:
        key, digest_now = candidate_identity(cand)
        current[key] = digest_now
    rows = pending_human(store)
    for row in rows:
        digest_now = current.get(row["candidate_key"])
        if digest_now is None:
            row["freshness"] = "gone"
        elif digest_now != row["content_digest"]:
            row["freshness"] = "content-changed"
        else:
            row["freshness"] = "current"
    return rows


def filter_undecided(
    store: KnowledgeStore, candidates: Iterable[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Drop candidates whose CURRENT content already carries a standing
    decision (pending_human or resolved). A decision on older content does not
    suppress: the digest differs, the candidate is visible again and the stale
    row is superseded at the next booking."""

    standing = standing_decisions(store)
    kept: list[Mapping[str, Any]] = []
    for candidate in candidates:
        key, content = candidate_identity(candidate)
        row = standing.get(key)
        if row is not None and row["content_digest"] == content:
            continue
        kept.append(candidate)
    return kept
