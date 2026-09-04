"""Compensating transactions over the version tables (plan §2.5).

``revert_revision`` writes a new revision that undoes revision ``rev`` —
version rows are only ever added, never rewritten. It is all-or-nothing: if
any entity touched at ``rev`` was touched again later, the revert refuses and
lists the blockers (revert the later revisions first).

``restore_node`` is the user-only undo of a retire: the same ``local_id``
gets a new live version (plus the memberships, items and links that were
closed by the same retire revision), so evidence and history stay attached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .store import KnowledgeStore, NotFoundError, Transaction


@dataclass(frozen=True)
class RevertBlocker:
    entity: str  # node | item | membership | link
    entity_id: str
    reason: str


class RevertError(RuntimeError):
    def __init__(self, blockers: list[RevertBlocker]) -> None:
        lines = ", ".join(f"{b.entity} {b.entity_id} ({b.reason})" for b in blockers)
        super().__init__(f"revert blocked by later changes: {lines}")
        self.blockers = blockers


_TABLES = (
    ("node", "node_versions", "local_id"),
    ("item", "item_versions", "item_id"),
    ("membership", "membership_versions", "membership_id"),
    ("link", "link_versions", "link_id"),
)


def _touched_after(store: KnowledgeStore, table: str, id_col: str, entity_id: str, rev: int) -> bool:
    row = store.conn.execute(
        f"SELECT 1 FROM {table} WHERE {id_col}=? AND"
        " (valid_from_rev > ? OR (valid_to_rev IS NOT NULL AND valid_to_rev > ?)) LIMIT 1",
        (entity_id, rev, rev),
    ).fetchone()
    return row is not None


def _rows_at(store: KnowledgeStore, table: str, id_col: str, rev: int) -> dict[str, dict[str, Any]]:
    """Per entity: the row started at ``rev`` and/or the row closed at ``rev``."""

    out: dict[str, dict[str, Any]] = {}
    for row in store.conn.execute(f"SELECT * FROM {table} WHERE valid_from_rev=?", (rev,)):
        out.setdefault(row[id_col], {})["started"] = dict(row)
    for row in store.conn.execute(f"SELECT * FROM {table} WHERE valid_to_rev=?", (rev,)):
        out.setdefault(row[id_col], {})["closed"] = dict(row)
    return out


def revert_revision(store: KnowledgeStore, rev: int) -> int:
    """Undo revision ``rev`` as a new ``kind=revert`` revision; returns it."""

    store.revision(rev)  # raises NotFoundError for an unknown rev
    plans: list[tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]] = []
    blockers: list[RevertBlocker] = []
    for entity, table, id_col in _TABLES:
        for entity_id, rows in _rows_at(store, table, id_col, rev).items():
            if _touched_after(store, table, id_col, entity_id, rev):
                blockers.append(RevertBlocker(entity, entity_id, "modified after this revision"))
                continue
            plans.append((entity, entity_id, rows.get("started"), rows.get("closed")))
    if blockers:
        raise RevertError(blockers)
    if not plans:
        raise NotFoundError(f"revision {rev} changed nothing")

    with store.begin("revert", base_rev=rev, note=f"revert:{rev}") as txn:
        for entity, entity_id, started, closed in plans:
            if started is not None and closed is None:
                # created at rev -> tombstone
                _tombstone(txn, entity, entity_id, rev)
            elif started is not None and closed is not None:
                # updated at rev -> put the prior content back
                _restore_row(txn, entity, entity_id, closed, expected_from_rev=rev)
            else:
                # tombstoned at rev -> revive the closed row
                assert closed is not None
                _revive_row(txn, entity, entity_id, closed)
        return txn.rev


def restore_node(store: KnowledgeStore, local_id: str) -> int:
    """User-explicit resurrection of a retired node (plan §2.5): same
    ``local_id``, new ``kind=restore`` revision; brings back the memberships,
    items and links the retire closed with it."""

    if store.node(local_id) is not None:
        raise ValueError(f"{local_id} is live; nothing to restore")
    last = store.conn.execute(
        "SELECT * FROM node_versions WHERE local_id=? ORDER BY valid_to_rev DESC LIMIT 1",
        (local_id,),
    ).fetchone()
    if last is None:
        raise NotFoundError(local_id)
    retire_rev = last["valid_to_rev"]

    with store.begin("restore", note=f"restore:{local_id}") as txn:
        _revive_row(txn, "node", local_id, dict(last))
        for row in store.conn.execute(
            "SELECT m.* FROM membership_versions m WHERE m.valid_to_rev=? AND (m.child_id=? OR m.parent_id=?)",
            (retire_rev, local_id, local_id),
        ).fetchall():
            other = row["parent_id"] if row["child_id"] == local_id else row["child_id"]
            if store.node(other) is None:
                continue  # the other end is still retired; its own restore brings the edge back
            live = store.conn.execute(
                "SELECT 1 FROM membership_versions WHERE membership_id=? AND valid_to_rev IS NULL",
                (row["membership_id"],),
            ).fetchone()
            if live is None:
                _revive_row(txn, "membership", row["membership_id"], dict(row))
        for row in store.conn.execute(
            "SELECT v.* FROM item_versions v JOIN items i USING(item_id)"
            " WHERE v.valid_to_rev=? AND i.local_id=?",
            (retire_rev, local_id),
        ).fetchall():
            _revive_row(txn, "item", row["item_id"], dict(row))
        for row in store.conn.execute(
            "SELECT * FROM link_versions WHERE valid_to_rev=? AND (source_id=? OR target_id=?)",
            (retire_rev, local_id, local_id),
        ).fetchall():
            _revive_row(txn, "link", row["link_id"], dict(row))
        return txn.rev


# ---- row-level helpers -----------------------------------------------------------


def _tombstone(txn: Transaction, entity: str, entity_id: str, expected_from_rev: int) -> None:
    if entity == "node":
        txn.tombstone_node(entity_id, expected_from_rev=expected_from_rev)
    elif entity == "item":
        txn.tombstone_item(entity_id, expected_from_rev=expected_from_rev)
    elif entity == "membership":
        txn.tombstone_membership(entity_id, expected_from_rev=expected_from_rev)
    else:
        txn.tombstone_link(entity_id, expected_from_rev=expected_from_rev)


def _restore_row(txn: Transaction, entity: str, entity_id: str, row: dict[str, Any], *, expected_from_rev: int) -> None:
    if entity == "node":
        txn.update_node(
            entity_id,
            payload=json.loads(row["payload"]),
            visibility=row["visibility"],
            canonical_id=row["canonical_id"],
            accepted_rev=row["accepted_rev"],
            expected_from_rev=expected_from_rev,
        )
    elif entity == "item":
        txn.update_item(
            entity_id,
            value=row["value"],
            exact_enabled=bool(row["exact_enabled"]),
            fuzzy_enabled=bool(row["fuzzy_enabled"]),
            requires_subject_context=bool(row["requires_subject_context"]),
            min_mora=row["min_mora"],
            accepted_rev=row["accepted_rev"],
            expected_from_rev=expected_from_rev,
        )
    elif entity == "membership":
        txn.move_membership(
            entity_id,
            parent_id=row["parent_id"],
            section=row["section"],
            order_key=row["order_key"],
            expected_from_rev=expected_from_rev,
        )
    else:
        # link versions are immutable: a link is only ever created or tombstoned
        raise AssertionError("links have no update path")


def _revive_row(txn: Transaction, entity: str, entity_id: str, row: dict[str, Any]) -> None:
    if entity == "node":
        txn.revive_node(
            entity_id,
            json.loads(row["payload"]),
            canonical_id=row["canonical_id"],
            visibility=row["visibility"],
            accepted_rev=row["accepted_rev"],
        )
    elif entity == "item":
        txn.revive_item(
            entity_id,
            row["value"],
            exact_enabled=bool(row["exact_enabled"]),
            fuzzy_enabled=bool(row["fuzzy_enabled"]),
            requires_subject_context=bool(row["requires_subject_context"]),
            min_mora=row["min_mora"],
            canonical_item_id=row["canonical_item_id"],
            accepted_rev=row["accepted_rev"],
        )
    elif entity == "membership":
        txn.revive_membership(
            entity_id,
            row["parent_id"],
            row["child_id"],
            row["section"],
            row["order_key"],
            canonical_membership_id=row["canonical_membership_id"],
            accepted_rev=row["accepted_rev"],
        )
    else:
        txn.revive_link(entity_id, row["source_id"], row["rel"], row["target_id"], accepted_rev=row["accepted_rev"])
