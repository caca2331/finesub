"""SQLite DDL for the node store (plan §2.1).

Every versioned entity is split into an identity table (immutable columns
only) and a version table keyed by ``(id, valid_from_rev)``; a row is
current while ``valid_to_rev IS NULL``. Nothing is ever deleted (plan §2.1:
no GC in v1), so pinned reads at any past revision stay answerable.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 5

# Older stores are upgraded in place: each entry maps a stored version to the
# statements that bring it one step forward. Version 2 (plan §5.5) adds
# ``evidence.created_at`` — the report's 最近印证日期 needs a timestamp, and the
# dedupe key stays timestamp-free so replays still deduplicate. Version 3
# (plan §11.5) adds ``maturity`` to node AND item versions: the shared
# corpus's distribution lifecycle (normal | tentative), claim-adjacent
# granularity because a verified term can gain one low-confidence item.
# Version 4 (kb-followups plan A6) adds ``candidate_decisions`` — the table
# itself comes from the shared DDL (CREATE IF NOT EXISTS), so the step just
# advances the version. Version 5 (review 2026-08-28 P2-4) adds the human-
# facing columns a pending decision needs: ``candidate`` (JSON snapshot of the
# candidate as booked — the key alone is an opaque hash) and ``missing`` (what
# evidence the session said it lacked); a store already stamped 4 gets the
# ALTERs, a fresh one gets them from the DDL and the ladder tolerates the
# duplicate-column no-op.
MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: ("ALTER TABLE evidence ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",),
    2: (
        "ALTER TABLE node_versions ADD COLUMN maturity TEXT NOT NULL DEFAULT 'normal'",
        "ALTER TABLE item_versions ADD COLUMN maturity TEXT NOT NULL DEFAULT 'normal'",
    ),
    3: (),
    4: (
        "ALTER TABLE candidate_decisions ADD COLUMN candidate TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE candidate_decisions ADD COLUMN missing TEXT NOT NULL DEFAULT ''",
    ),
}

DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS revisions (
        rev INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        kind TEXT NOT NULL,
        task_id TEXT NOT NULL DEFAULT '',
        proposal_hash TEXT NOT NULL DEFAULT '',
        base_rev INTEGER,
        note TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS nodes (
        local_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        created_rev INTEGER NOT NULL REFERENCES revisions(rev)
    )""",
    """CREATE TABLE IF NOT EXISTS node_versions (
        local_id TEXT NOT NULL REFERENCES nodes(local_id),
        valid_from_rev INTEGER NOT NULL REFERENCES revisions(rev),
        valid_to_rev INTEGER REFERENCES revisions(rev),
        payload TEXT NOT NULL,
        canonical_id TEXT,
        visibility TEXT NOT NULL DEFAULT 'local',
        maturity TEXT NOT NULL DEFAULT 'normal',
        accepted_rev INTEGER,
        PRIMARY KEY (local_id, valid_from_rev)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_node_versions_current
        ON node_versions(local_id) WHERE valid_to_rev IS NULL""",
    """CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        local_id TEXT NOT NULL REFERENCES nodes(local_id),
        field TEXT NOT NULL,
        created_rev INTEGER NOT NULL REFERENCES revisions(rev)
    )""",
    """CREATE TABLE IF NOT EXISTS item_versions (
        item_id TEXT NOT NULL REFERENCES items(item_id),
        valid_from_rev INTEGER NOT NULL REFERENCES revisions(rev),
        valid_to_rev INTEGER REFERENCES revisions(rev),
        value TEXT NOT NULL,
        exact_enabled INTEGER NOT NULL DEFAULT 1,
        fuzzy_enabled INTEGER NOT NULL DEFAULT 0,
        requires_subject_context INTEGER NOT NULL DEFAULT 0,
        min_mora INTEGER NOT NULL DEFAULT 3,
        canonical_item_id TEXT,
        maturity TEXT NOT NULL DEFAULT 'normal',
        accepted_rev INTEGER,
        PRIMARY KEY (item_id, valid_from_rev)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_items_node ON items(local_id)""",
    """CREATE TABLE IF NOT EXISTS memberships (
        membership_id TEXT PRIMARY KEY,
        created_rev INTEGER NOT NULL REFERENCES revisions(rev)
    )""",
    """CREATE TABLE IF NOT EXISTS membership_versions (
        membership_id TEXT NOT NULL REFERENCES memberships(membership_id),
        valid_from_rev INTEGER NOT NULL REFERENCES revisions(rev),
        valid_to_rev INTEGER REFERENCES revisions(rev),
        parent_id TEXT NOT NULL REFERENCES nodes(local_id),
        child_id TEXT NOT NULL REFERENCES nodes(local_id),
        section TEXT NOT NULL,
        order_key INTEGER NOT NULL,
        canonical_membership_id TEXT,
        accepted_rev INTEGER,
        PRIMARY KEY (membership_id, valid_from_rev)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_membership_parent ON membership_versions(parent_id)""",
    """CREATE INDEX IF NOT EXISTS idx_membership_child ON membership_versions(child_id)""",
    """CREATE TABLE IF NOT EXISTS links (
        link_id TEXT PRIMARY KEY,
        created_rev INTEGER NOT NULL REFERENCES revisions(rev)
    )""",
    """CREATE TABLE IF NOT EXISTS link_versions (
        link_id TEXT NOT NULL REFERENCES links(link_id),
        valid_from_rev INTEGER NOT NULL REFERENCES revisions(rev),
        valid_to_rev INTEGER REFERENCES revisions(rev),
        source_id TEXT NOT NULL REFERENCES nodes(local_id),
        rel TEXT NOT NULL,
        target_id TEXT NOT NULL REFERENCES nodes(local_id),
        accepted_rev INTEGER,
        PRIMARY KEY (link_id, valid_from_rev)
    )""",
    """CREATE TABLE IF NOT EXISTS redirects (
        old_canonical_id TEXT PRIMARY KEY,
        new_canonical_id TEXT NOT NULL,
        learned_rev INTEGER NOT NULL REFERENCES revisions(rev)
    )""",
    """CREATE TABLE IF NOT EXISTS sync_state (
        remote TEXT NOT NULL,
        canonical_id TEXT NOT NULL,
        local_id TEXT NOT NULL,
        last_server_rev INTEGER,
        last_pulled_payload_hash TEXT,
        PRIMARY KEY (remote, canonical_id)
    )""",
    """CREATE TABLE IF NOT EXISTS evidence (
        evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dedupe_key TEXT NOT NULL UNIQUE,
        node_id TEXT NOT NULL,
        field_path TEXT NOT NULL,
        value_hash TEXT NOT NULL,
        verdict TEXT NOT NULL,
        evidence_kind TEXT NOT NULL,
        source_ref TEXT,
        task_id TEXT NOT NULL,
        span TEXT,
        algo_version TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dedupe_key TEXT NOT NULL UNIQUE,
        trace_id TEXT NOT NULL,
        parent_event_id INTEGER,
        kind TEXT NOT NULL,
        opportunity TEXT NOT NULL,
        task_id TEXT NOT NULL,
        window_id TEXT,
        subject_id TEXT NOT NULL,
        node_id TEXT,
        item_id TEXT,
        matcher TEXT,
        rev INTEGER NOT NULL,
        span TEXT,
        algo_version TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS candidate_decisions (
        decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_key TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        status TEXT NOT NULL,
        resolution TEXT NOT NULL DEFAULT '',
        reason TEXT NOT NULL DEFAULT '',
        task_id TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        resolved_at TEXT NOT NULL DEFAULT '',
        candidate TEXT NOT NULL DEFAULT '',
        missing TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE INDEX IF NOT EXISTS idx_candidate_decisions_key
        ON candidate_decisions(candidate_key)""",
    """CREATE TABLE IF NOT EXISTS migration_aux (
        local_id TEXT PRIMARY KEY REFERENCES nodes(local_id),
        legacy_raw TEXT NOT NULL,
        source_path TEXT NOT NULL,
        source_line INTEGER NOT NULL,
        layout TEXT NOT NULL DEFAULT '{}'
    )""",
)


def connect(path: str | Path, *, cross_thread: bool = False) -> sqlite3.Connection:
    """Open (creating if needed) a store file with the pragmas we rely on.

    ``cross_thread`` is for the share server only: its HTTP handlers run on
    request threads and serialize every store access behind one lock, so the
    same connection may legally cross threads there. Everything else keeps
    sqlite3's per-thread check (the repo registry is per-thread on purpose).
    """

    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=not cross_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    # Telemetry (events/evidence) is written from the harness and the MCP tool
    # server concurrently; a short queue beats an immediate SQLITE_BUSY.
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    # IMMEDIATE, not a deferred BEGIN: every caller here writes (DDL, and the
    # migration ladder), and a deferred transaction takes the read lock first
    # and asks to upgrade later. Two connections opening the same store at once
    # then hold read locks and each wait for the other to drop -- an upgrade
    # deadlock, which SQLite refuses IMMEDIATELY with "database is locked"
    # instead of honouring busy_timeout. Concurrent opens are the normal case
    # since task-level parallelism (每线程一个连接, see repo.py).
    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in DDL:
            conn.execute(statement)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        else:
            stored = int(row["value"])
            while stored in MIGRATIONS:
                for statement in MIGRATIONS[stored]:
                    try:
                        conn.execute(statement)
                    except sqlite3.OperationalError as exc:
                        # A table the old store never had was just created by
                        # the current DDL — already in its final shape, so the
                        # ladder's ALTER is a no-op, not an error.
                        if "duplicate column name" not in str(exc):
                            raise
                stored += 1
            if stored != SCHEMA_VERSION:
                raise RuntimeError(
                    f"knowledge store schema {row['value']} != supported {SCHEMA_VERSION}"
                )
            conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),)
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
