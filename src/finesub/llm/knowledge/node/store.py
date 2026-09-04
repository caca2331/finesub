"""Versioned node store with pinned reads and per-entity CAS (plan §2.1, §2.5).

Writes happen inside one ``revision`` context = one SQLite transaction = one
new ``rev``. Reads take a pinned ``rev`` (default: current) and only see rows
with ``valid_from_rev <= rev < valid_to_rev``; the version rows make that
answerable without holding a read transaction open.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .model import (
    ITEM_FIELDS,
    KINDS,
    LEGACY_KINDS,
    LINK_RELS,
    MATURITIES,
    REVISION_KINDS,
    VISIBILITIES,
    ItemVersion,
    LinkVersion,
    MembershipVersion,
    MigrationAux,
    NodeVersion,
    Revision,
    validate_payload,
)
from .schema import connect, init_schema


class StaleWriteError(RuntimeError):
    """CAS failure: the entity's current version is not the one the writer saw."""


class NotFoundError(KeyError):
    pass


_PINNED = "valid_from_rev <= ? AND (valid_to_rev IS NULL OR valid_to_rev > ?)"


def _row_node(row: sqlite3.Row, kind: str) -> NodeVersion:
    return NodeVersion(
        local_id=row["local_id"],
        kind=kind,
        valid_from_rev=row["valid_from_rev"],
        valid_to_rev=row["valid_to_rev"],
        payload=json.loads(row["payload"]),
        canonical_id=row["canonical_id"],
        visibility=row["visibility"],
        accepted_rev=row["accepted_rev"],
        maturity=row["maturity"],
    )


def _row_item(row: sqlite3.Row) -> ItemVersion:
    return ItemVersion(
        item_id=row["item_id"],
        local_id=row["local_id"],
        field=row["field"],
        valid_from_rev=row["valid_from_rev"],
        valid_to_rev=row["valid_to_rev"],
        value=row["value"],
        exact_enabled=bool(row["exact_enabled"]),
        fuzzy_enabled=bool(row["fuzzy_enabled"]),
        requires_subject_context=bool(row["requires_subject_context"]),
        min_mora=row["min_mora"],
        canonical_item_id=row["canonical_item_id"],
        accepted_rev=row["accepted_rev"],
        maturity=row["maturity"],
    )


def _row_membership(row: sqlite3.Row) -> MembershipVersion:
    return MembershipVersion(
        membership_id=row["membership_id"],
        valid_from_rev=row["valid_from_rev"],
        valid_to_rev=row["valid_to_rev"],
        parent_id=row["parent_id"],
        child_id=row["child_id"],
        section=row["section"],
        order_key=row["order_key"],
        canonical_membership_id=row["canonical_membership_id"],
        accepted_rev=row["accepted_rev"],
    )


def _row_link(row: sqlite3.Row) -> LinkVersion:
    return LinkVersion(
        link_id=row["link_id"],
        valid_from_rev=row["valid_from_rev"],
        valid_to_rev=row["valid_to_rev"],
        source_id=row["source_id"],
        rel=row["rel"],
        target_id=row["target_id"],
        accepted_rev=row["accepted_rev"],
    )


class KnowledgeStore:
    def __init__(self, path: str | Path, *, cross_thread: bool = False) -> None:
        self.path = Path(path)
        self.conn = connect(self.path, cross_thread=cross_thread)
        init_schema(self.conn)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- revisions -----------------------------------------------------

    def current_rev(self) -> int:
        row = self.conn.execute("SELECT MAX(rev) AS rev FROM revisions").fetchone()
        return int(row["rev"] or 0)

    def revision(self, rev: int) -> Revision:
        row = self.conn.execute("SELECT * FROM revisions WHERE rev=?", (rev,)).fetchone()
        if row is None:
            raise NotFoundError(rev)
        return Revision(
            rev=row["rev"],
            created_at=row["created_at"],
            kind=row["kind"],
            task_id=row["task_id"],
            proposal_hash=row["proposal_hash"],
            base_rev=row["base_rev"],
            note=row["note"],
        )

    @contextmanager
    def begin(
        self,
        kind: str,
        *,
        task_id: str = "",
        proposal_hash: str = "",
        base_rev: int | None = None,
        note: str = "",
    ) -> Iterator["Transaction"]:
        """One transaction = one new revision; rolled back entirely on error."""

        if kind not in REVISION_KINDS:
            raise ValueError(f"unknown revision kind {kind!r}")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.conn.execute(
                "INSERT INTO revisions(created_at, kind, task_id, proposal_hash, base_rev, note)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    kind,
                    task_id,
                    proposal_hash,
                    base_rev,
                    note,
                ),
            )
            txn = Transaction(self, int(cursor.lastrowid))
            yield txn
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # ---- pinned reads ----------------------------------------------------

    def _rev(self, rev: int | None) -> int:
        return self.current_rev() if rev is None else int(rev)

    def node(self, local_id: str, rev: int | None = None) -> NodeVersion | None:
        at = self._rev(rev)
        row = self.conn.execute(
            "SELECT v.*, n.kind AS kind FROM node_versions v JOIN nodes n USING(local_id)"
            f" WHERE v.local_id=? AND {_PINNED}",
            (local_id, at, at),
        ).fetchone()
        return _row_node(row, row["kind"]) if row else None

    def node_current_from_rev(self, local_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT valid_from_rev FROM node_versions WHERE local_id=? AND valid_to_rev IS NULL",
            (local_id,),
        ).fetchone()
        return int(row["valid_from_rev"]) if row else None

    def items_of(self, local_id: str, rev: int | None = None) -> list[ItemVersion]:
        at = self._rev(rev)
        rows = self.conn.execute(
            "SELECT v.*, i.local_id AS local_id, i.field AS field FROM item_versions v"
            f" JOIN items i USING(item_id) WHERE i.local_id=? AND {_PINNED}"
            " ORDER BY i.rowid",
            (local_id, at, at),
        ).fetchall()
        return [_row_item(row) for row in rows]

    def all_items(self, rev: int | None = None) -> list[ItemVersion]:
        at = self._rev(rev)
        rows = self.conn.execute(
            "SELECT v.*, i.local_id AS local_id, i.field AS field FROM item_versions v"
            f" JOIN items i USING(item_id) WHERE {_PINNED} ORDER BY i.rowid",
            (at, at),
        ).fetchall()
        return [_row_item(row) for row in rows]

    def children(self, parent_id: str, rev: int | None = None) -> list[MembershipVersion]:
        at = self._rev(rev)
        rows = self.conn.execute(
            f"SELECT * FROM membership_versions WHERE parent_id=? AND {_PINNED}"
            " ORDER BY order_key",
            (parent_id, at, at),
        ).fetchall()
        return [_row_membership(row) for row in rows]

    def parents(self, child_id: str, rev: int | None = None) -> list[MembershipVersion]:
        at = self._rev(rev)
        rows = self.conn.execute(
            f"SELECT * FROM membership_versions WHERE child_id=? AND {_PINNED}",
            (child_id, at, at),
        ).fetchall()
        return [_row_membership(row) for row in rows]

    def links_from(self, source_id: str, rev: int | None = None) -> list[LinkVersion]:
        at = self._rev(rev)
        rows = self.conn.execute(
            f"SELECT * FROM link_versions WHERE source_id=? AND {_PINNED}",
            (source_id, at, at),
        ).fetchall()
        return [_row_link(row) for row in rows]

    def subjects(self, rev: int | None = None) -> list[NodeVersion]:
        """Top-level subjects: kind=subject with no parent membership at ``rev``."""

        at = self._rev(rev)
        rows = self.conn.execute(
            "SELECT v.*, n.kind AS kind FROM node_versions v JOIN nodes n USING(local_id)"
            " WHERE n.kind='subject'"
            " AND v.valid_from_rev <= ? AND (v.valid_to_rev IS NULL OR v.valid_to_rev > ?)"
            " AND NOT EXISTS (SELECT 1 FROM membership_versions m WHERE m.child_id=v.local_id"
            "   AND m.valid_from_rev <= ? AND (m.valid_to_rev IS NULL OR m.valid_to_rev > ?))"
            " ORDER BY n.rowid",
            (at, at, at, at),
        ).fetchall()
        return [_row_node(row, row["kind"]) for row in rows]

    def nodes_of_kind(self, kind: str, rev: int | None = None) -> list[NodeVersion]:
        at = self._rev(rev)
        rows = self.conn.execute(
            "SELECT v.*, n.kind AS kind FROM node_versions v JOIN nodes n USING(local_id)"
            f" WHERE n.kind=? AND {_PINNED} ORDER BY n.rowid",
            (kind, at, at),
        ).fetchall()
        return [_row_node(row, row["kind"]) for row in rows]

    def migration_aux(self, local_id: str) -> MigrationAux | None:
        row = self.conn.execute(
            "SELECT * FROM migration_aux WHERE local_id=?", (local_id,)
        ).fetchone()
        if row is None:
            return None
        return MigrationAux(
            local_id=row["local_id"],
            legacy_raw=row["legacy_raw"],
            source_path=row["source_path"],
            source_line=row["source_line"],
            layout=json.loads(row["layout"] or "{}"),
        )

    def meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


class Transaction:
    """Write handle bound to one revision. Every mutation closes the current
    version row (if any) and inserts a new one; CAS is enforced when the
    caller supplies ``expected_from_rev`` (plan §2.5)."""

    def __init__(self, store: KnowledgeStore, rev: int) -> None:
        self.store = store
        self.conn = store.conn
        self.rev = rev

    # ---- nodes -----------------------------------------------------------

    def create_node(
        self,
        local_id: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        visibility: str = "local",
        canonical_id: str | None = None,
        maturity: str = "normal",
    ) -> NodeVersion:
        # LEGACY_KINDS are still storable: Phase A of the re-import writes
        # them and Phase B converts them (plan §8). The write path rejects
        # them through validate_payload(strict=True).
        if kind not in KINDS and kind not in LEGACY_KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if visibility not in VISIBILITIES:
            raise ValueError(f"unknown visibility {visibility!r}")
        if maturity not in MATURITIES:
            raise ValueError(f"unknown maturity {maturity!r}")
        validate_payload(kind, payload, strict=False)
        self.conn.execute(
            "INSERT INTO nodes(local_id, kind, created_rev) VALUES (?, ?, ?)",
            (local_id, kind, self.rev),
        )
        self.conn.execute(
            "INSERT INTO node_versions(local_id, valid_from_rev, payload, canonical_id, visibility, maturity)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (local_id, self.rev, json.dumps(payload, ensure_ascii=False), canonical_id, visibility, maturity),
        )
        return NodeVersion(local_id, kind, self.rev, None, dict(payload), canonical_id, visibility, maturity=maturity)

    def _close_node(self, local_id: str, expected_from_rev: int | None) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT v.*, n.kind AS kind FROM node_versions v JOIN nodes n USING(local_id)"
            " WHERE v.local_id=? AND v.valid_to_rev IS NULL",
            (local_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(local_id)
        if expected_from_rev is not None and row["valid_from_rev"] != expected_from_rev:
            raise StaleWriteError(
                f"{local_id}: expected version {expected_from_rev}, current is {row['valid_from_rev']}"
            )
        if row["valid_from_rev"] == self.rev:
            raise StaleWriteError(f"{local_id}: already written in revision {self.rev}")
        self.conn.execute(
            "UPDATE node_versions SET valid_to_rev=? WHERE local_id=? AND valid_to_rev IS NULL",
            (self.rev, local_id),
        )
        return row

    def update_node(
        self,
        local_id: str,
        *,
        payload: Mapping[str, Any] | None = None,
        visibility: str | None = None,
        canonical_id: str | None = None,
        accepted_rev: int | None = None,
        expected_from_rev: int | None = None,
        maturity: str | None = None,
    ) -> NodeVersion:
        row = self._close_node(local_id, expected_from_rev)
        kind = row["kind"]
        new_payload = dict(payload) if payload is not None else json.loads(row["payload"])
        validate_payload(kind, new_payload, strict=False)
        new_visibility = visibility if visibility is not None else row["visibility"]
        new_canonical = canonical_id if canonical_id is not None else row["canonical_id"]
        new_maturity = maturity if maturity is not None else row["maturity"]
        if new_maturity not in MATURITIES:
            raise ValueError(f"unknown maturity {new_maturity!r}")
        self.conn.execute(
            "INSERT INTO node_versions(local_id, valid_from_rev, payload, canonical_id, visibility, maturity, accepted_rev)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                local_id,
                self.rev,
                json.dumps(new_payload, ensure_ascii=False),
                new_canonical,
                new_visibility,
                new_maturity,
                accepted_rev,
            ),
        )
        return NodeVersion(local_id, kind, self.rev, None, new_payload, new_canonical, new_visibility, accepted_rev, maturity=new_maturity)

    def tombstone_node(self, local_id: str, *, expected_from_rev: int | None = None) -> None:
        self._close_node(local_id, expected_from_rev)

    def revive_node(
        self,
        local_id: str,
        payload: Mapping[str, Any],
        *,
        canonical_id: str | None = None,
        visibility: str = "local",
        accepted_rev: int | None = None,
        maturity: str = "normal",
    ) -> NodeVersion:
        """New current version for an existing, currently tombstoned identity
        (plan §2.5 restore / revert)."""

        row = self.conn.execute("SELECT kind FROM nodes WHERE local_id=?", (local_id,)).fetchone()
        if row is None:
            raise NotFoundError(local_id)
        live = self.conn.execute(
            "SELECT 1 FROM node_versions WHERE local_id=? AND valid_to_rev IS NULL", (local_id,)
        ).fetchone()
        if live is not None:
            raise StaleWriteError(f"{local_id}: already live")
        kind = row["kind"]
        validate_payload(kind, payload, strict=False)
        if maturity not in MATURITIES:
            raise ValueError(f"unknown maturity {maturity!r}")
        self.conn.execute(
            "INSERT INTO node_versions(local_id, valid_from_rev, payload, canonical_id, visibility, maturity, accepted_rev)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (local_id, self.rev, json.dumps(payload, ensure_ascii=False), canonical_id, visibility, maturity, accepted_rev),
        )
        return NodeVersion(local_id, kind, self.rev, None, dict(payload), canonical_id, visibility, accepted_rev, maturity=maturity)

    # ---- items -----------------------------------------------------------

    def create_item(
        self,
        item_id: str,
        local_id: str,
        field: str,
        value: str,
        *,
        exact_enabled: bool = True,
        fuzzy_enabled: bool = False,
        requires_subject_context: bool = False,
        min_mora: int = 3,
        maturity: str = "normal",
    ) -> ItemVersion:
        if field not in ITEM_FIELDS:
            raise ValueError(f"unknown item field {field!r}")
        if maturity not in MATURITIES:
            raise ValueError(f"unknown maturity {maturity!r}")
        self.conn.execute(
            "INSERT INTO items(item_id, local_id, field, created_rev) VALUES (?, ?, ?, ?)",
            (item_id, local_id, field, self.rev),
        )
        self.conn.execute(
            "INSERT INTO item_versions(item_id, valid_from_rev, value, exact_enabled, fuzzy_enabled,"
            " requires_subject_context, min_mora, maturity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item_id, self.rev, value, int(exact_enabled), int(fuzzy_enabled), int(requires_subject_context), min_mora, maturity),
        )
        return ItemVersion(item_id, local_id, field, self.rev, None, value, exact_enabled, fuzzy_enabled, requires_subject_context, min_mora, maturity=maturity)

    def _close_item(self, item_id: str, expected_from_rev: int | None) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT v.*, i.local_id AS local_id, i.field AS field FROM item_versions v"
            " JOIN items i USING(item_id) WHERE v.item_id=? AND v.valid_to_rev IS NULL",
            (item_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(item_id)
        if expected_from_rev is not None and row["valid_from_rev"] != expected_from_rev:
            raise StaleWriteError(f"{item_id}: expected {expected_from_rev}, current {row['valid_from_rev']}")
        self.conn.execute(
            "UPDATE item_versions SET valid_to_rev=? WHERE item_id=? AND valid_to_rev IS NULL",
            (self.rev, item_id),
        )
        return row

    def update_item(
        self,
        item_id: str,
        *,
        value: str | None = None,
        exact_enabled: bool | None = None,
        fuzzy_enabled: bool | None = None,
        requires_subject_context: bool | None = None,
        min_mora: int | None = None,
        accepted_rev: int | None = None,
        expected_from_rev: int | None = None,
        maturity: str | None = None,
    ) -> ItemVersion:
        row = self._close_item(item_id, expected_from_rev)
        merged = {
            "value": row["value"] if value is None else value,
            "exact_enabled": bool(row["exact_enabled"]) if exact_enabled is None else exact_enabled,
            "fuzzy_enabled": bool(row["fuzzy_enabled"]) if fuzzy_enabled is None else fuzzy_enabled,
            "requires_subject_context": bool(row["requires_subject_context"])
            if requires_subject_context is None
            else requires_subject_context,
            "min_mora": row["min_mora"] if min_mora is None else min_mora,
            "maturity": row["maturity"] if maturity is None else maturity,
        }
        if merged["maturity"] not in MATURITIES:
            raise ValueError(f"unknown maturity {merged['maturity']!r}")
        self.conn.execute(
            "INSERT INTO item_versions(item_id, valid_from_rev, value, exact_enabled, fuzzy_enabled,"
            " requires_subject_context, min_mora, canonical_item_id, maturity, accepted_rev)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                self.rev,
                merged["value"],
                int(merged["exact_enabled"]),
                int(merged["fuzzy_enabled"]),
                int(merged["requires_subject_context"]),
                merged["min_mora"],
                row["canonical_item_id"],
                merged["maturity"],
                accepted_rev,
            ),
        )
        return ItemVersion(
            item_id,
            row["local_id"],
            row["field"],
            self.rev,
            None,
            merged["value"],
            merged["exact_enabled"],
            merged["fuzzy_enabled"],
            merged["requires_subject_context"],
            merged["min_mora"],
            row["canonical_item_id"],
            accepted_rev,
            maturity=merged["maturity"],
        )

    def tombstone_item(self, item_id: str, *, expected_from_rev: int | None = None) -> None:
        self._close_item(item_id, expected_from_rev)

    def revive_item(
        self,
        item_id: str,
        value: str,
        *,
        exact_enabled: bool = True,
        fuzzy_enabled: bool = False,
        requires_subject_context: bool = False,
        min_mora: int = 3,
        canonical_item_id: str | None = None,
        accepted_rev: int | None = None,
        maturity: str = "normal",
    ) -> None:
        row = self.conn.execute(
            "SELECT local_id, field FROM items WHERE item_id=?", (item_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(item_id)
        live = self.conn.execute(
            "SELECT 1 FROM item_versions WHERE item_id=? AND valid_to_rev IS NULL", (item_id,)
        ).fetchone()
        if live is not None:
            raise StaleWriteError(f"{item_id}: already live")
        if maturity not in MATURITIES:
            raise ValueError(f"unknown maturity {maturity!r}")
        self.conn.execute(
            "INSERT INTO item_versions(item_id, valid_from_rev, value, exact_enabled, fuzzy_enabled,"
            " requires_subject_context, min_mora, canonical_item_id, maturity, accepted_rev)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                self.rev,
                value,
                int(exact_enabled),
                int(fuzzy_enabled),
                int(requires_subject_context),
                min_mora,
                canonical_item_id,
                maturity,
                accepted_rev,
            ),
        )

    # ---- memberships -----------------------------------------------------

    def create_membership(
        self, membership_id: str, parent_id: str, child_id: str, section: str, order_key: int
    ) -> MembershipVersion:
        self.conn.execute(
            "INSERT INTO memberships(membership_id, created_rev) VALUES (?, ?)",
            (membership_id, self.rev),
        )
        self.conn.execute(
            "INSERT INTO membership_versions(membership_id, valid_from_rev, parent_id, child_id, section, order_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (membership_id, self.rev, parent_id, child_id, section, order_key),
        )
        return MembershipVersion(membership_id, self.rev, None, parent_id, child_id, section, order_key)

    def _close_membership(self, membership_id: str, expected_from_rev: int | None) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM membership_versions WHERE membership_id=? AND valid_to_rev IS NULL",
            (membership_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(membership_id)
        if expected_from_rev is not None and row["valid_from_rev"] != expected_from_rev:
            raise StaleWriteError(f"{membership_id}: expected {expected_from_rev}, current {row['valid_from_rev']}")
        self.conn.execute(
            "UPDATE membership_versions SET valid_to_rev=? WHERE membership_id=? AND valid_to_rev IS NULL",
            (self.rev, membership_id),
        )
        return row

    def move_membership(
        self,
        membership_id: str,
        *,
        parent_id: str | None = None,
        section: str | None = None,
        order_key: int | None = None,
        expected_from_rev: int | None = None,
    ) -> MembershipVersion:
        row = self._close_membership(membership_id, expected_from_rev)
        new_parent = parent_id or row["parent_id"]
        new_section = section if section is not None else row["section"]
        new_order = row["order_key"] if order_key is None else order_key
        self.conn.execute(
            "INSERT INTO membership_versions(membership_id, valid_from_rev, parent_id, child_id, section,"
            " order_key, canonical_membership_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (membership_id, self.rev, new_parent, row["child_id"], new_section, new_order, row["canonical_membership_id"]),
        )
        return MembershipVersion(membership_id, self.rev, None, new_parent, row["child_id"], new_section, new_order, row["canonical_membership_id"])

    def tombstone_membership(self, membership_id: str, *, expected_from_rev: int | None = None) -> None:
        self._close_membership(membership_id, expected_from_rev)

    def revive_membership(
        self,
        membership_id: str,
        parent_id: str,
        child_id: str,
        section: str,
        order_key: int,
        *,
        canonical_membership_id: str | None = None,
        accepted_rev: int | None = None,
    ) -> None:
        row = self.conn.execute(
            "SELECT 1 FROM memberships WHERE membership_id=?", (membership_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(membership_id)
        live = self.conn.execute(
            "SELECT 1 FROM membership_versions WHERE membership_id=? AND valid_to_rev IS NULL",
            (membership_id,),
        ).fetchone()
        if live is not None:
            raise StaleWriteError(f"{membership_id}: already live")
        self.conn.execute(
            "INSERT INTO membership_versions(membership_id, valid_from_rev, parent_id, child_id, section,"
            " order_key, canonical_membership_id, accepted_rev) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (membership_id, self.rev, parent_id, child_id, section, order_key, canonical_membership_id, accepted_rev),
        )

    # ---- links -------------------------------------------------------------

    def create_link(self, link_id: str, source_id: str, rel: str, target_id: str) -> LinkVersion:
        if rel not in LINK_RELS:
            raise ValueError(f"unknown link rel {rel!r}")
        self.conn.execute("INSERT INTO links(link_id, created_rev) VALUES (?, ?)", (link_id, self.rev))
        self.conn.execute(
            "INSERT INTO link_versions(link_id, valid_from_rev, source_id, rel, target_id) VALUES (?, ?, ?, ?, ?)",
            (link_id, self.rev, source_id, rel, target_id),
        )
        return LinkVersion(link_id, self.rev, None, source_id, rel, target_id)

    def tombstone_link(self, link_id: str, *, expected_from_rev: int | None = None) -> None:
        row = self.conn.execute(
            "SELECT valid_from_rev FROM link_versions WHERE link_id=? AND valid_to_rev IS NULL", (link_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(link_id)
        if expected_from_rev is not None and row["valid_from_rev"] != expected_from_rev:
            raise StaleWriteError(link_id)
        self.conn.execute(
            "UPDATE link_versions SET valid_to_rev=? WHERE link_id=? AND valid_to_rev IS NULL",
            (self.rev, link_id),
        )

    def revive_link(self, link_id: str, source_id: str, rel: str, target_id: str, *, accepted_rev: int | None = None) -> None:
        row = self.conn.execute("SELECT 1 FROM links WHERE link_id=?", (link_id,)).fetchone()
        if row is None:
            raise NotFoundError(link_id)
        live = self.conn.execute(
            "SELECT 1 FROM link_versions WHERE link_id=? AND valid_to_rev IS NULL", (link_id,)
        ).fetchone()
        if live is not None:
            raise StaleWriteError(f"{link_id}: already live")
        self.conn.execute(
            "INSERT INTO link_versions(link_id, valid_from_rev, source_id, rel, target_id, accepted_rev)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (link_id, self.rev, source_id, rel, target_id, accepted_rev),
        )

    # ---- migration sidecar ---------------------------------------------------

    def set_migration_aux(self, aux: MigrationAux) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO migration_aux(local_id, legacy_raw, source_path, source_line, layout)"
            " VALUES (?, ?, ?, ?, ?)",
            (aux.local_id, aux.legacy_raw, aux.source_path, aux.source_line, json.dumps(aux.layout, ensure_ascii=False)),
        )

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
