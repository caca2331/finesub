"""Pull merge (plan §6.1/§6.2): snapshot → local store, one ``pull`` revision.

Field strategies (§6.2): scalars three-way against the last-pulled base
(kept findable forever because version rows are never deleted — the base is
the newest local version whose payload hash equals
``sync_state.last_pulled_payload_hash``); prose keeps local and marks pending;
sets (items / memberships) union by canonical sub-entity id with monotone
tombstones; links merge on their natural key ``(source, rel, target)``.
Redirects are followed before anything else. Conflicts never block the pull —
they land in the report for the human, remote loses (§6.2: default keep
local, mark for merge).

Anti-rollback (hardening 2026-08-26): the client anchors on the last accepted
``(server_rev, chain_hash)`` per remote (store ``meta``); a snapshot whose
history does not extend that anchor is refused before any write.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..node.model import digest
from ..node.store import KnowledgeStore, Transaction
from .exchange import (
    ExchangeError,
    sanitize_payload,
    sanitize_text,
    snapshot_content_problems,
    verify_snapshot,
)

PROSE_FIELDS = ("intro", "body", "text", "description")


#: Conflict kinds. ``scalar`` is a three-way field disagreement (local kept,
#: remote dropped); ``prose`` is a body/intro field, which is never merged
#: automatically at all. They differ in what a resolution MEANS -- one picks a
#: side, the other writes new text -- so the kind travels with the record.
SCALAR_CONFLICT = "scalar"
PROSE_CONFLICT = "prose"


@dataclass
class FieldConflict:
    """One field the pull could not settle, in a form that outlives the pull.

    It used to be a formatted string, which is exactly as much as a human
    could read off the terminal before it scrolled away. A record is what
    lets the conflict be written down, listed later and handed back to a model
    against the entry's CURRENT contents: without ``canonical_id``/``local_id``
    nothing can find the node again, and without ``base``/``had_base`` nobody
    can tell "both sides moved" (a real disagreement) from "there was no common
    ancestor to compare against" (first contact with a locally pre-existing
    node, where EVERY difference is reported).
    """

    remote: str
    canonical_id: str
    local_id: str
    label: str
    field: str
    kind: str
    local: Any = None
    incoming: Any = None
    base: Any = None
    had_base: bool = False
    server_rev: int = 0
    pulled_rev: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def describe(self) -> str:
        """The one-line rendering the CLI prints. Kept next to the record so
        the terminal wording cannot drift away from the stored fields."""

        if self.kind == PROSE_CONFLICT:
            return f"{self.label}.{self.field}"
        return f"{self.label}.{self.field}: local={self.local!r} remote={self.incoming!r}"


@dataclass
class PullReport:
    remote: str
    server_rev: int = 0
    created_nodes: int = 0
    updated_nodes: int = 0
    retired_nodes: int = 0
    created_items: int = 0
    retired_items: int = 0
    created_memberships: int = 0
    retired_memberships: int = 0
    created_links: int = 0
    conflicts: list[FieldConflict] = field(default_factory=list)
    pending_prose: list[FieldConflict] = field(default_factory=list)
    rev: int | None = None

    def unsettled(self) -> list[FieldConflict]:
        """Both lists in one, which is what the conflict ledger stores: a
        prose field pending a manual merge is no less unresolved than a scalar
        one, and splitting them is a presentation choice, not a data one."""

        return [*self.conflicts, *self.pending_prose]

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["conflicts"] = [c.to_dict() for c in self.conflicts]
        data["pending_prose"] = [c.to_dict() for c in self.pending_prose]
        return data


def _anchor_keys(remote: str) -> tuple[str, str]:
    return (f"share:{remote}:server_rev", f"share:{remote}:chain_hash")


def check_anchor(store: KnowledgeStore, remote: str, snapshot: Mapping[str, Any]) -> None:
    rev_key, hash_key = _anchor_keys(remote)
    stored_rev = store.meta(rev_key)
    if stored_rev is None:
        return  # first pull from this remote: trust on first use
    anchor_rev = int(stored_rev)
    anchor_hash = store.meta(hash_key) or ""
    history = snapshot.get("history") or []
    entry = next((h for h in history if h.get("rev") == anchor_rev), None)
    if entry is None or entry.get("chain_hash") != anchor_hash:
        raise ExchangeError(
            f"server history no longer contains the trusted anchor rev {anchor_rev} — refusing"
        )
    if int(snapshot.get("server_rev") or 0) < anchor_rev:
        raise ExchangeError("server_rev went backwards — refusing rollback")


def _base_payload(store: KnowledgeStore, local_id: str, base_hash: str) -> dict[str, Any] | None:
    rows = store.conn.execute(
        "SELECT payload FROM node_versions WHERE local_id=? ORDER BY valid_from_rev DESC",
        (local_id,),
    ).fetchall()
    for row in rows:
        payload = json.loads(row["payload"])
        if digest(payload) == base_hash:
            return payload
    return None


def _merge_scalars(
    local: Mapping[str, Any],
    remote: Mapping[str, Any],
    base: Mapping[str, Any] | None,
    *,
    label: str,
    canonical_id: str,
    local_id: str,
    report: PullReport,
) -> dict[str, Any]:
    """Three-way per field; without a base (first contact with a locally
    pre-existing node) every difference is a conflict and local wins."""

    merged = dict(local)

    def _conflict(key: str, kind: str, **extra: Any) -> FieldConflict:
        return FieldConflict(
            remote=report.remote,
            canonical_id=canonical_id,
            local_id=local_id,
            label=label,
            field=key,
            kind=kind,
            **extra,
        )

    for key in sorted(set(local) | set(remote)):
        local_value = local.get(key)
        remote_value = remote.get(key)
        if local_value == remote_value:
            continue
        if key in PROSE_FIELDS:
            report.pending_prose.append(
                _conflict(
                    key,
                    PROSE_CONFLICT,
                    local=local_value,
                    incoming=remote_value,
                    base=(base or {}).get(key),
                    had_base=base is not None,
                )
            )
            continue
        base_value = (base or {}).get(key)
        if base is not None and local_value == base_value:
            merged[key] = remote_value  # only remote moved
        elif base is not None and remote_value == base_value:
            pass  # only local moved
        else:
            report.conflicts.append(
                _conflict(
                    key,
                    SCALAR_CONFLICT,
                    local=local_value,
                    incoming=remote_value,
                    base=base_value,
                    had_base=base is not None,
                )
            )
    return merged


def apply_snapshot(
    store: KnowledgeStore,
    snapshot: Mapping[str, Any],
    *,
    remote: str,
    task_id: str = "share-pull",
) -> PullReport:
    """Verify, then fold one snapshot into the local store in one revision."""

    verify_snapshot(snapshot)
    # The chain proves authorship, not safety (round 13): a self-consistent
    # malicious snapshot passes verification, so its CONTENT faces the same
    # admission boundary a pushed bundle does before anything is written.
    problems = snapshot_content_problems(snapshot["content"])
    if problems:
        raise ExchangeError("snapshot content refused: " + "; ".join(problems[:5]))
    check_anchor(store, remote, snapshot)
    content = snapshot["content"]
    report = PullReport(remote=remote, server_rev=int(snapshot.get("server_rev") or 0))

    redirects: dict[str, str] = dict(content.get("redirects") or {})

    def follow(canonical_id: str) -> str:
        seen = set()
        while canonical_id in redirects and canonical_id not in seen:
            seen.add(canonical_id)
            canonical_id = redirects[canonical_id]
        return canonical_id

    # What the server said each node looks like *now*: recorded as the next
    # three-way base. On a conflicted field the local value wins and the
    # remote payload never lands locally, so `_base_payload` will not find
    # this hash next time — the conflict is re-reported (never silently
    # resolved toward the remote) until a human converges the two.
    remote_hash: dict[str, str] = {}
    with store.begin("pull", task_id=task_id, note=f"pull from {remote}") as txn:
        # canonical -> local mapping, built INSIDE the transaction: BEGIN
        # IMMEDIATE serializes concurrent pulls, so the second one sees the
        # first one's rows instead of creating a duplicate local node for the
        # same canonical id (review 2026-08-27).
        local_of: dict[str, str] = {}
        for row in store.conn.execute(
            "SELECT canonical_id, local_id FROM sync_state WHERE remote=?", (remote,)
        ):
            local_of[row["canonical_id"]] = row["local_id"]
        for row in store.conn.execute(
            "SELECT canonical_id, local_id FROM node_versions"
            " WHERE canonical_id IS NOT NULL AND valid_to_rev IS NULL"
        ):
            local_of.setdefault(row["canonical_id"], row["local_id"])
        _apply_redirects(store, txn, redirects, report)
        for old, new in redirects.items():
            if old in local_of:
                local_of.setdefault(follow(old), local_of[old])
        for entry in content.get("nodes") or []:
            canonical = follow(str(entry.get("canonical_id") or ""))
            if canonical and not entry.get("retired"):
                remote_hash[canonical] = digest(sanitize_payload(entry.get("payload") or {}))
            _apply_node(store, txn, entry, follow, local_of, remote, report)
        for entry in content.get("items") or []:
            _apply_item(store, txn, entry, local_of, report)
        for entry in content.get("memberships") or []:
            _apply_membership(store, txn, entry, local_of, report)
        for entry in content.get("links") or []:
            _apply_link(store, txn, entry, local_of, report)
        for canonical_id, local_id in sorted(local_of.items()):
            payload_hash = remote_hash.get(canonical_id, "")
            if not payload_hash:
                node = store.conn.execute(
                    "SELECT payload FROM node_versions WHERE local_id=? AND valid_to_rev IS NULL",
                    (local_id,),
                ).fetchone()
                if node is not None:
                    payload_hash = digest(json.loads(node["payload"]))
            txn.conn.execute(
                "INSERT OR REPLACE INTO sync_state(remote, canonical_id, local_id,"
                " last_server_rev, last_pulled_payload_hash) VALUES (?, ?, ?, ?, ?)",
                (remote, canonical_id, local_id, report.server_rev, payload_hash),
            )
        rev_key, hash_key = _anchor_keys(remote)
        txn.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (rev_key, str(report.server_rev)),
        )
        txn.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (hash_key, str(snapshot.get("chain_hash") or "")),
        )
        report.rev = txn.rev
    # Stamped after the fact rather than at the call site: the revision this
    # pull got is only known once the transaction closes, and a conflict that
    # cannot say WHEN it was seen is one nobody can re-check against a later
    # state of the entry.
    for conflict in report.unsettled():
        conflict.server_rev = report.server_rev
        conflict.pulled_rev = report.rev
    return report


def _apply_redirects(
    store: KnowledgeStore, txn: Transaction, redirects: Mapping[str, str], report: PullReport
) -> None:
    for old, new in sorted(redirects.items()):
        txn.conn.execute(
            "INSERT OR REPLACE INTO redirects(old_canonical_id, new_canonical_id, learned_rev)"
            " VALUES (?, ?, ?)",
            (old, new, txn.rev),
        )
        # In-place: canonical_id is server-issued addressing metadata, not
        # user content, and a version row here would collide with the same
        # node's field merge later in this transaction.
        txn.conn.execute(
            "UPDATE node_versions SET canonical_id=? WHERE canonical_id=? AND valid_to_rev IS NULL",
            (new, old),
        )


def _apply_node(
    store: KnowledgeStore,
    txn: Transaction,
    entry: Mapping[str, Any],
    follow,
    local_of: dict[str, str],
    remote: str,
    report: PullReport,
) -> None:
    canonical_id = follow(str(entry.get("canonical_id") or ""))
    if not canonical_id:
        return
    payload = sanitize_payload(entry.get("payload") or {})
    local_id = local_of.get(canonical_id)
    current = store.node(local_id) if local_id else None
    if entry.get("retired"):
        # canonical retire is monotone (§6.2): close the local counterpart.
        if current is not None:
            txn.tombstone_node(local_id)
            report.retired_nodes += 1
        return
    maturity = str(entry.get("maturity") or "normal")
    if maturity not in ("normal", "tentative"):
        maturity = "normal"
    if local_id is None:
        local_id = str(uuid.uuid4())
        local_of[canonical_id] = local_id
        txn.create_node(
            local_id, str(entry.get("kind") or "note"), payload,
            canonical_id=canonical_id, maturity=maturity,
        )
        report.created_nodes += 1
        return
    if current is None:
        return  # locally retired stays retired: no automatic revival (plan §9)
    base_row = store.conn.execute(
        "SELECT last_pulled_payload_hash FROM sync_state WHERE remote=? AND canonical_id=?",
        (remote, canonical_id),
    ).fetchone()
    base = (
        _base_payload(store, local_id, base_row["last_pulled_payload_hash"])
        if base_row and base_row["last_pulled_payload_hash"]
        else None
    )
    label = str(current.payload.get("surface") or current.payload.get("field") or local_id)
    merged = _merge_scalars(
        current.payload,
        payload,
        base,
        label=label,
        canonical_id=canonical_id,
        local_id=local_id,
        report=report,
    )
    # ONE new version per pulled node: a payload merge and a maturity flip
    # (lifecycle follows the server, plan §11.5) land together — two
    # update_node calls in the same revision would trip the same-rev guard.
    if merged != current.payload or current.maturity != maturity:
        txn.update_node(
            local_id,
            payload=merged if merged != current.payload else None,
            canonical_id=canonical_id,
            maturity=maturity if current.maturity != maturity else None,
        )
        report.updated_nodes += 1


def _apply_item(
    store: KnowledgeStore,
    txn: Transaction,
    entry: Mapping[str, Any],
    local_of: dict[str, str],
    report: PullReport,
) -> None:
    canonical_item_id = str(entry.get("canonical_item_id") or "")
    owner = local_of.get(str(entry.get("node") or ""))
    if not canonical_item_id or owner is None:
        return
    existing = store.conn.execute(
        "SELECT v.item_id, v.valid_to_rev IS NULL AS live, v.maturity AS maturity"
        " FROM item_versions v"
        " WHERE v.canonical_item_id=? ORDER BY v.valid_from_rev DESC LIMIT 1",
        (canonical_item_id,),
    ).fetchone()
    if entry.get("retired"):
        if existing is not None and existing["live"]:
            txn.tombstone_item(existing["item_id"])
            report.retired_items += 1
        return
    if existing is not None:
        wanted = str(entry.get("maturity") or "normal")
        if existing["live"] and wanted in ("normal", "tentative") and existing["maturity"] != wanted:
            # lifecycle follows the server (tentative -> normal, plan §11.5)
            txn.update_item(existing["item_id"], maturity=wanted)
            txn.conn.execute(
                "UPDATE item_versions SET canonical_item_id=? WHERE item_id=? AND valid_to_rev IS NULL",
                (canonical_item_id, existing["item_id"]),
            )
        return  # present (or locally tombstoned: local delete wins until server retires)
    value = sanitize_text(str(entry.get("value") or ""))
    normalized = {i.value for i in store.items_of(owner) if i.field == entry.get("field")}
    if value in normalized:
        # same value already there locally without a canonical id: adopt it
        for item in store.items_of(owner):
            if item.field == entry.get("field") and item.value == value and not item.canonical_item_id:
                txn.update_item(item.item_id)  # reopen as a new version...
                txn.conn.execute(
                    "UPDATE item_versions SET canonical_item_id=? WHERE item_id=? AND valid_to_rev IS NULL",
                    (canonical_item_id, item.item_id),
                )
                return
        return
    item_maturity = str(entry.get("maturity") or "normal")
    item_id = str(uuid.uuid4())
    txn.create_item(
        item_id, owner, str(entry.get("field")), value,
        maturity=item_maturity if item_maturity in ("normal", "tentative") else "normal",
    )
    txn.conn.execute(
        "UPDATE item_versions SET canonical_item_id=? WHERE item_id=? AND valid_to_rev IS NULL",
        (canonical_item_id, item_id),
    )
    report.created_items += 1


def _apply_membership(
    store: KnowledgeStore,
    txn: Transaction,
    entry: Mapping[str, Any],
    local_of: dict[str, str],
    report: PullReport,
) -> None:
    canonical = str(entry.get("canonical_membership_id") or "")
    parent = local_of.get(str(entry.get("parent") or ""))
    child = local_of.get(str(entry.get("child") or ""))
    if not canonical or parent is None or child is None:
        return
    existing = store.conn.execute(
        "SELECT membership_id, valid_to_rev IS NULL AS live FROM membership_versions"
        " WHERE canonical_membership_id=? ORDER BY valid_from_rev DESC LIMIT 1",
        (canonical,),
    ).fetchone()
    if entry.get("retired"):
        if existing is not None and existing["live"]:
            txn.tombstone_membership(existing["membership_id"])
            report.retired_memberships += 1
        return
    if existing is not None:
        return
    for membership in store.children(parent):
        if membership.child_id == child and membership.section == entry.get("section"):
            txn.conn.execute(
                "UPDATE membership_versions SET canonical_membership_id=?"
                " WHERE membership_id=? AND valid_to_rev IS NULL",
                (canonical, membership.membership_id),
            )
            return
    membership_id = str(uuid.uuid4())
    txn.create_membership(
        membership_id, parent, child,
        sanitize_text(str(entry.get("section") or ""), max_chars=100),
        int(entry.get("order_key") or 0),
    )
    txn.conn.execute(
        "UPDATE membership_versions SET canonical_membership_id=?"
        " WHERE membership_id=? AND valid_to_rev IS NULL",
        (canonical, membership_id),
    )
    report.created_memberships += 1


def _apply_link(
    store: KnowledgeStore,
    txn: Transaction,
    entry: Mapping[str, Any],
    local_of: dict[str, str],
    report: PullReport,
) -> None:
    source = local_of.get(str(entry.get("source") or ""))
    target = local_of.get(str(entry.get("target") or ""))
    if source is None or target is None:
        return  # dangling reference stays dangling (§6.1) until the node arrives
    rel = str(entry.get("rel") or "")
    existing = [
        link for link in store.links_from(source)
        if link.rel == rel and link.target_id == target
    ]
    if entry.get("retired"):
        for link in existing:
            txn.tombstone_link(link.link_id)
        return
    if existing:
        return
    txn.create_link(str(uuid.uuid4()), source, rel, target)
    report.created_links += 1
