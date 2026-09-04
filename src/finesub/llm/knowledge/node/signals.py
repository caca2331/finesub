"""Field-level events and claim evidence (plan §5, step 5).

The four-level ladder ``matched → exposed → landed → confirmed/refuted``:
``matched`` is booked by the shadow scan (`matching.log_matched`); this module
books the other three and the claim-level evidence rows they feed.

Everything here is telemetry with the same discipline as the shadow scan —
idempotent inserts keyed by canonical dedupe hashes, never read back into any
prompt (plan §5.4), and never worth failing a run over (callers wrap in
fail-soft guards).

Level semantics (plan §5.1):

- ``exposed``: the node's content actually reached the model — for the REST
  front end that is "rendered into this window's prompt", for the agent front
  end "returned by a kb tool". Exposures of a node whose ``misheard`` item
  matched the window's raw text are ``opportunity=correction`` (the only rows
  admitted to the false-injection denominator, §5.2); the rest are
  ``opportunity=context``.
- ``landed``: the corrected text carries the node's canonical name where the
  raw text did not — the output is *consistent with* the node, deliberately
  not "the model used it" (§1.2: no causality measurement).
- ``confirmed``/``refuted``: claim-level evidence rows, not events. The
  deterministic producer here is the refined-SRT alignment (§5.4): a kept
  correction confirms exactly the misheard-item claim that attributed it, an
  overturned one refutes it. The mistake-ledger half of that feedback stays
  with the model's ``add_mistake`` proposals (refined mode already applies
  them); this module owns the evidence half.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

from .matching import ExactIndex, scan_normalize
from .model import digest
from .store import KnowledgeStore

EXPOSED_ALGO_VERSION = "exposure-1"
LANDED_ALGO_VERSION = "landed-1"
REFINED_ALGO_VERSION = "refined-align-1"
EVIDENCE_KIND_REFINED = "refined_srt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def book_event(store: KnowledgeStore, payload: Mapping[str, object]) -> bool:
    """Idempotent event insert; the dedupe key hashes every key column with
    explicit nulls (plan §2.1). Returns True when the row is new."""

    key_columns = {
        "kind": payload["kind"],
        "opportunity": payload.get("opportunity", ""),
        "task_id": payload.get("task_id", ""),
        "window_id": payload.get("window_id") or None,
        "subject_id": payload.get("subject_id", ""),
        "node_id": payload.get("node_id") or None,
        "item_id": payload.get("item_id") or None,
        "matcher": payload.get("matcher") or None,
        "rev": payload["rev"],
        "span": payload.get("span") or None,
        "algo_version": payload["algo_version"],
    }
    cursor = store.conn.execute(
        "INSERT OR IGNORE INTO events(dedupe_key, trace_id, parent_event_id, kind, opportunity,"
        " task_id, window_id, subject_id, node_id, item_id, matcher, rev, span, algo_version)"
        " VALUES (?, '', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            digest(key_columns),
            key_columns["kind"],
            key_columns["opportunity"],
            key_columns["task_id"],
            key_columns["window_id"],
            key_columns["subject_id"],
            key_columns["node_id"],
            key_columns["item_id"],
            key_columns["matcher"],
            key_columns["rev"],
            key_columns["span"],
            key_columns["algo_version"],
        ),
    )
    return cursor.rowcount > 0


def record_evidence(
    store: KnowledgeStore,
    *,
    node_id: str,
    field_path: str,
    value_hash: str,
    verdict: str,
    evidence_kind: str,
    task_id: str,
    source_ref: str | None = None,
    span: str | None = None,
    algo_version: str = REFINED_ALGO_VERSION,
) -> bool:
    """One claim-level evidence row (plan §5.3): attached to
    ``(node_id, field_path, value_hash)``, deduped without the timestamp so a
    replayed run books nothing new. Returns True when the row is new."""

    if verdict not in ("confirmed", "refuted", "unverifiable"):
        # ``unverifiable`` is the verification task's terminal state (plan
        # §11.4): booked once so the claim is not re-tried every sweep.
        raise ValueError(f"unknown verdict {verdict!r}")
    dedupe = digest(
        {
            "task_id": task_id,
            "node_id": node_id,
            "field_path": field_path,
            "value_hash": value_hash,
            "verdict": verdict,
            "span": span or None,
            "algo_version": algo_version,
            # kind and source are identity too: two URLs corroborating the
            # same claim are two pieces of evidence, not a replay of one
            # (review 2026-08-27 round 7). Replays of the same source still
            # deduplicate.
            "evidence_kind": evidence_kind,
            "source_ref": source_ref or None,
        }
    )
    cursor = store.conn.execute(
        "INSERT OR IGNORE INTO evidence(dedupe_key, node_id, field_path, value_hash, verdict,"
        " evidence_kind, source_ref, task_id, span, algo_version, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dedupe,
            node_id,
            field_path,
            value_hash,
            verdict,
            evidence_kind,
            source_ref,
            task_id,
            span,
            algo_version,
            _now(),
        ),
    )
    return cursor.rowcount > 0


def book_revision_evidence(
    store: KnowledgeStore,
    rev: int,
    *,
    evidence_kind: str,
    task_id: str = "",
    algo_version: str = "revision-provenance-1",
) -> int:
    """Book ``confirmed`` provenance evidence for every claim value one
    revision set (plan §11.4 / O8-O9): node payload string fields born at
    ``rev`` (semantic keys only) plus item values born at ``rev``.

    ``evidence_kind`` names the source class — ``user`` for human applies
    (the owner's edit IS the local endorsement), ``transcript`` for harness
    applies (the value came out of this run's material; revision-level
    granularity, the per-hint source ids stay in the artifacts). Returns the
    number of new rows."""

    from .model import NON_SEMANTIC_PAYLOAD_KEYS

    count = 0
    for row in store.conn.execute(
        "SELECT local_id FROM node_versions WHERE valid_from_rev=?", (rev,)
    ).fetchall():
        node = store.node(row["local_id"], rev)
        if node is None:
            continue  # closed at the same rev (tombstone)
        for key, value in node.payload.items():
            if key in NON_SEMANTIC_PAYLOAD_KEYS or not isinstance(value, str) or not value:
                continue
            if record_evidence(
                store, node_id=node.local_id, field_path=f"payload.{key}",
                value_hash=digest(value), verdict="confirmed",
                evidence_kind=evidence_kind, task_id=task_id, algo_version=algo_version,
            ):
                count += 1
    for row in store.conn.execute(
        "SELECT v.item_id, i.local_id, v.value FROM item_versions v JOIN items i USING(item_id)"
        " WHERE v.valid_from_rev=? AND v.valid_to_rev IS NULL",
        (rev,),
    ).fetchall():
        if record_evidence(
            store, node_id=row["local_id"], field_path=f"items/{row['item_id']}",
            value_hash=digest(row["value"]), verdict="confirmed",
            evidence_kind=evidence_kind, task_id=task_id, algo_version=algo_version,
        ):
            count += 1
    return count


# ---------------------------------------------------------------------------
# exposed


def subject_pack_node_ids(
    store: KnowledgeStore, subject_id: str, rev: int, *, sections: Iterable[str] | None = None
) -> list[str]:
    """The nodes a subject pack renders: the subject plus every descendant
    reachable through live memberships at ``rev``. ``sections`` (the agent's
    ``kb_read`` filter) prunes at the subject's own level only — deeper
    memberships belong to child terms whose section names are unrelated."""

    wanted = set(sections) if sections is not None else None
    out = [subject_id]
    seen = {subject_id}
    frontier = [
        membership.child_id
        for membership in store.children(subject_id, rev)
        if wanted is None or membership.section in wanted
    ]
    while frontier:
        node_id = frontier.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(node_id)
        frontier.extend(membership.child_id for membership in store.children(node_id, rev))
    return out


def misheard_owner_ids(index: ExactIndex, window_text: str) -> set[str]:
    """Nodes whose ``misheard`` item fires in this window's raw text — the
    exposures that count toward the false-injection denominator (§5.2)."""

    return {
        match.key.node_id
        for match in index.scan(window_text)
        if match.key.kind == "misheard"
    }


def log_exposed_nodes(
    store: KnowledgeStore,
    exposures: Iterable[tuple[str, str]],
    *,
    task_id: str,
    window_id: str = "",
    correction_nodes: set[str] | frozenset[str] = frozenset(),
    rev: int | None = None,
) -> int:
    """Book ``exposed`` events for ``(subject_id, node_id)`` pairs. Nodes in
    ``correction_nodes`` are ``opportunity=correction``, the rest ``context``.
    Returns how many rows were new."""

    at = store.current_rev() if rev is None else rev
    inserted = 0
    for subject_id, node_id in exposures:
        inserted += int(
            book_event(
                store,
                {
                    "kind": "exposed",
                    "opportunity": "correction" if node_id in correction_nodes else "context",
                    "task_id": task_id,
                    "window_id": window_id,
                    "subject_id": subject_id,
                    "node_id": node_id,
                    "rev": at,
                    "algo_version": EXPOSED_ALGO_VERSION,
                },
            )
        )
    return inserted


def log_exposed_entries(
    repo,  # KnowledgeRepo; untyped to keep the import edge one-way
    entry_keys: Iterable[str],
    *,
    task_id: str,
    window_id: str = "",
    window_text: str = "",
    rev: int | None = None,
) -> int:
    """REST whole-pack exposure (plan §4.2 item 5): every node the injected
    entries render, correction-opportunity where a misheard item of that node
    fired in the window's raw text."""

    store = repo.store
    at = store.current_rev() if rev is None else rev
    correction: set[str] = set()
    if window_text.strip():
        correction = misheard_owner_ids(ExactIndex.build(store, at, include_tentative=True), window_text)
    exposures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in entry_keys:
        resolved = repo.resolve(key, at)
        if resolved is None:
            continue
        for node_id in subject_pack_node_ids(store, resolved.subject_id, at):
            if node_id not in seen:
                seen.add(node_id)
                exposures.append((resolved.subject_id, node_id))
    return log_exposed_nodes(
        store,
        exposures,
        task_id=task_id,
        window_id=window_id,
        correction_nodes=correction,
        rev=at,
    )


# ---------------------------------------------------------------------------
# landed


@dataclass(frozen=True)
class _NameKey:
    text: str  # normalized
    raw: str
    node_id: str
    subject_id: str
    field_path: str  # payload.surface | payload.zh


def canonical_name_keys(store: KnowledgeStore, rev: int) -> list[_NameKey]:
    """Per-node canonical names — subject/term surfaces plus term ``zh``.
    Deliberately not the item corpus: a misheard variant appearing in the
    corrected text is not a landing."""

    subject_of: dict[str, str] = {}
    for subject in store.subjects(rev):
        subject_of[subject.local_id] = subject.local_id
        stack = [subject.local_id]
        while stack:
            parent = stack.pop()
            for membership in store.children(parent, rev):
                if membership.child_id not in subject_of:
                    subject_of[membership.child_id] = subject.local_id
                    stack.append(membership.child_id)

    keys: list[_NameKey] = []

    def add(node_id: str, field_path: str, raw: str) -> None:
        subject_id = subject_of.get(node_id)
        normalized = scan_normalize(raw)
        if subject_id is not None and len(normalized) >= 2:
            keys.append(_NameKey(normalized, raw, node_id, subject_id, field_path))

    for subject in store.subjects(rev):
        add(subject.local_id, "payload.surface", subject.payload.get("surface", ""))
    for term in store.nodes_of_kind("term", rev):
        add(term.local_id, "payload.surface", term.payload.get("surface", ""))
        add(term.local_id, "payload.zh", term.payload.get("zh", ""))
    return keys


def log_landed_windows(
    store: KnowledgeStore,
    windows: Iterable[tuple[str, str, str]],
    *,
    task_id: str,
    rev: int | None = None,
) -> int:
    """Book ``landed`` events: a canonical name present in the corrected text
    of a window whose raw text does not contain it (plan §5.1 — consistency,
    not usage). ``windows`` yields ``(window_id, raw_text, corrected_text)``."""

    at = store.current_rev() if rev is None else rev
    names = canonical_name_keys(store, at)
    inserted = 0
    for window_id, raw_text, corrected_text in windows:
        raw_norm = scan_normalize(raw_text)
        corrected_norm = scan_normalize(corrected_text)
        booked: set[str] = set()
        for name in names:
            if name.node_id in booked:
                continue
            position = corrected_norm.find(name.text)
            if position == -1 or name.text in raw_norm:
                continue
            booked.add(name.node_id)
            inserted += int(
                book_event(
                    store,
                    {
                        "kind": "landed",
                        "opportunity": "correction",
                        "task_id": task_id,
                        "window_id": window_id,
                        "subject_id": name.subject_id,
                        "node_id": name.node_id,
                        "rev": at,
                        "span": f"{position}-{position + len(name.text)}",
                        "algo_version": LANDED_ALGO_VERSION,
                    },
                )
            )
    return inserted


# ---------------------------------------------------------------------------
# refined-SRT alignment evidence (plan §5.4)


def refined_alignment_evidence(
    store: KnowledgeStore,
    windows: Iterable[tuple[str, str, str, str]],
    *,
    task_id: str,
    rev: int | None = None,
) -> tuple[int, int]:
    """Deterministic confirmed/refuted from the refined-SRT comparison.

    ``windows`` yields ``(window_id, raw_text, final_text, refined_text)``:
    raw is the source-language ASR, final the run's corrected+translated
    output, refined the user's refined subtitle. For every misheard item that
    fired in raw and whose owning term's ``zh`` made it into the final output
    (the correction the item claims to catch), the refined text keeping that
    ``zh`` confirms exactly that item's claim; the refined text dropping it is
    the strongest negative signal and refutes it (§5.3: one claim, never the
    whole node). Returns ``(confirmed, refuted)`` new-row counts."""

    at = store.current_rev() if rev is None else rev
    index = ExactIndex.build(store, at, include_tentative=True)
    zh_of: dict[str, str] = {
        term.local_id: scan_normalize(term.payload.get("zh", ""))
        for term in store.nodes_of_kind("term", at)
    }
    item_value: dict[str, str] = {
        item.item_id: item.value for item in store.all_items(at) if item.field == "misheard"
    }
    confirmed = refuted = 0
    for window_id, raw_text, final_text, refined_text in windows:
        final_norm = scan_normalize(final_text)
        refined_norm = scan_normalize(refined_text)
        claims: set[tuple[str, str]] = set()  # (node_id, item_id)
        for match in index.scan(raw_text):
            if match.key.kind == "misheard" and match.key.item_id in item_value:
                claims.add((match.key.node_id, match.key.item_id))
        for node_id, item_id in sorted(claims):
            zh = zh_of.get(node_id, "")
            if len(zh) < 2 or zh not in final_norm:
                continue  # no correction consistent with this claim to judge
            verdict = "confirmed" if zh in refined_norm else "refuted"
            new = record_evidence(
                store,
                node_id=node_id,
                field_path=f"items/{item_id}",
                value_hash=digest(item_value[item_id]),
                verdict=verdict,
                evidence_kind=EVIDENCE_KIND_REFINED,
                task_id=task_id,
                span=window_id,
            )
            if new:
                if verdict == "confirmed":
                    confirmed += 1
                else:
                    refuted += 1
    return confirmed, refuted
