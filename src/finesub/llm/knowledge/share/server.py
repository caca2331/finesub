"""Stdlib share server (plan §6: 服务端形态; hardening 2026-08-26).

``python -m finesub.llm.knowledge.share.server --root <dir> --port <n>``
serves three endpoint families over plain HTTP (TLS/domain belongs to
Caddy or a tunnel, not here):

- ``POST /register`` → an anonymous contributor token (no GitHub/email).
- ``GET /snapshot`` → the whole corpus keyed by canonical ids, with the
  integrity chain (``share_chain`` rows; head hash commits to every prior
  content digest, so a client anchored anywhere detects rewrites).
- ``POST /push`` (contributor token) → review queue, deduplicated by the
  client-generated idempotency key: a retry returns the same queue item.
  ``GET /push/<queue_id>`` returns status + the handle→canonical assignment
  once approved (the client backfills its ``canonical_*`` columns from it).
- ``GET /queue`` + ``POST /verdict`` (maintainer token): queue items are
  leased (token + expiry) and verdicts CAS on ``verdict_version`` — two
  maintainer sessions cannot silently overwrite each other. LLM review does
  NOT run here: the maintainer pulls the queue, reviews locally, posts the
  verdict — no API key ever lives on the server.

Queue, contributors and chain live in the *same* SQLite as the server corpus,
so an approval (verdict CAS + bundle apply + chain append) is one transaction.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from ..base import _match_normalize
from ..node.store import KnowledgeStore, Transaction
from .exchange import (
    bundle_claims,
    bundle_content_digest,
    chain_hash,
    content_digest,
    sanitize_bundle,
    sanitize_payload,
    sanitize_text,
    snapshot_content,
    threshold_report,
    unsatisfied_claims,
    validate_bundle,
)

SERVER_STORE_FILENAME = "share-server.sqlite"
DEFAULT_LEASE_SECONDS = 900

# Abuse bounds for anonymous, public-facing deployments (review 2026-08-27
# round 7): the app layer is the backstop — proxy-level IP rate limiting is
# additional (see deploy/share-server/Caddyfile).
MAX_CONTRIBUTORS = 500
MAX_PENDING_PER_CONTRIBUTOR = 5
MAX_PENDING_TOTAL = 200
PENDING_EXPIRY_DAYS = 30
TENTATIVE_REVIEW_DAYS = 90  # digest NOMINATES uncorroborated tentative for retire (plan §11.5)

_STRICT_URL_RE = re.compile(r"https?://\S+")

_EXTRA_DDL = (
    """CREATE TABLE IF NOT EXISTS share_queue (
        queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key TEXT NOT NULL UNIQUE,
        contributor TEXT NOT NULL,
        bundle TEXT NOT NULL,
        received_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        verdict TEXT NOT NULL DEFAULT '',
        verdict_version INTEGER NOT NULL DEFAULT 0,
        lease_token TEXT,
        lease_until REAL,
        assigned TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS contributors (
        token TEXT PRIMARY KEY,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS share_chain (
        rev INTEGER PRIMARY KEY,
        content_digest TEXT NOT NULL,
        chain_hash TEXT NOT NULL
    )""",
)


class ShareError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ShareService:
    """Storage + protocol logic, HTTP-free so tests drive it directly."""

    def __init__(
        self, root: str | Path, *, maintainer_token: str, auto_tentative: bool = False
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # One connection crossing HTTP handler threads; every request holds
        # ``lock`` for its whole store interaction (see _Handler._dispatch).
        self.lock = threading.RLock()
        self.store = KnowledgeStore(self.root / SERVER_STORE_FILENAME, cross_thread=True)
        for statement in _EXTRA_DDL:
            self.store.conn.execute(statement)
        self.maintainer_token = maintainer_token
        # O12: the reputable-contributor auto-tentative path is WIRED but
        # ships OFF — with it off, an approve_tentative verdict from the
        # maintainer's review session is the only route into tentative.
        self.auto_tentative = auto_tentative
        if self.store.conn.execute("SELECT COUNT(*) AS n FROM share_chain").fetchone()["n"] == 0:
            rev = self.store.current_rev()
            digest_value = content_digest(snapshot_content(self.store, rev))
            self.store.conn.execute(
                "INSERT INTO share_chain(rev, content_digest, chain_hash) VALUES (?, ?, ?)",
                (rev, digest_value, chain_hash("", digest_value)),
            )

    def close(self) -> None:
        self.store.close()

    # ---- contributors -------------------------------------------------

    def register(self) -> dict[str, Any]:
        count = self.store.conn.execute("SELECT COUNT(*) AS n FROM contributors").fetchone()["n"]
        if count >= MAX_CONTRIBUTORS:
            raise ShareError(429, "contributor registrations are full; contact the maintainer")
        token = secrets.token_urlsafe(24)
        self.store.conn.execute(
            "INSERT INTO contributors(token, created_at) VALUES (?, ?)",
            (token, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        return {"token": token}

    def _expire_stale(self) -> None:
        """Lazy sweep: pending items nobody reviewed within the window expire
        (they stop counting toward quotas and stop surfacing in leases). ISO
        UTC strings compare lexicographically, so this is one UPDATE."""

        cutoff = datetime.now(timezone.utc).timestamp() - PENDING_EXPIRY_DAYS * 86_400
        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(timespec="seconds")
        self.store.conn.execute(
            "UPDATE share_queue SET status='expired', lease_token=NULL, lease_until=NULL"
            " WHERE status='pending' AND received_at < ?",
            (cutoff_iso,),
        )

    def _check_quotas(self, contributor: str) -> None:
        pending_mine = self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM share_queue WHERE status='pending' AND contributor=?",
            (contributor,),
        ).fetchone()["n"]
        if pending_mine >= MAX_PENDING_PER_CONTRIBUTOR:
            raise ShareError(429, f"you already have {pending_mine} pending item(s); wait for review")
        pending_total = self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM share_queue WHERE status='pending'"
        ).fetchone()["n"]
        if pending_total >= MAX_PENDING_TOTAL:
            raise ShareError(429, "the review queue is full; try again later")

    def _require_contributor(self, token: str) -> str:
        row = self.store.conn.execute(
            "SELECT token FROM contributors WHERE token=?", (token or "",)
        ).fetchone()
        if row is None:
            raise ShareError(401, "unknown contributor token; POST /register first")
        return row["token"]

    def _require_maintainer(self, token: str) -> None:
        if not token or token != self.maintainer_token:
            raise ShareError(401, "maintainer token required")

    # ---- snapshot -----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        history = [
            {
                "rev": row["rev"],
                "chain_hash": row["chain_hash"],
                "content_digest": row["content_digest"],
            }
            for row in self.store.conn.execute(
                "SELECT rev, chain_hash, content_digest FROM share_chain ORDER BY rev"
            ).fetchall()
        ]
        head = history[-1]
        content = snapshot_content(self.store, head["rev"])
        return {
            "server_rev": head["rev"],
            "history": history,
            "content": content,
            "content_digest": content_digest(content),
            "chain_hash": head["chain_hash"],
        }

    # ---- push ---------------------------------------------------------

    def _check_anchors(self, bundle: Mapping[str, Any]) -> None:
        """Every canonical id the bundle leans on must exist at the current
        server revision (review 2026-08-27 round 7): an invented
        ``canonical_id`` or ``c:`` parent would otherwise pass the closure
        check and materialize as an orphan on approval. New entities never
        name their own ids — the server assigns them."""

        rev = self.store.current_rev()
        referenced: set[str] = set()
        anchored: set[str] = set()
        for node in bundle.get("nodes") or []:
            canonical = str(node.get("canonical_id") or "")
            if not canonical:
                continue
            current = self.store.node(canonical, rev)
            if current is None:
                raise ShareError(400, f"unknown canonical anchor {canonical!r}")
            # The §6.3 gate slots by the DECLARED kind; letting a bundle
            # declare an existing subject as a term would review under the
            # laxer slot while updating the subject (round 11).
            if current.kind != str(node.get("kind")):
                raise ShareError(
                    400,
                    f"node {node.get('handle')!r} declares {canonical!r} as"
                    f" {node.get('kind')!r} but it is a {current.kind}",
                )
            if canonical in anchored:
                raise ShareError(400, f"multiple bundle nodes anchor {canonical!r}")
            anchored.add(canonical)
        for membership in bundle.get("memberships") or []:
            for key in ("parent", "child"):
                ref = str(membership.get(key) or "")
                if ref.startswith("c:"):
                    referenced.add(ref[2:])
        for link in bundle.get("links") or []:
            for key in ("source", "target"):
                ref = str(link.get(key) or "")
                if ref.startswith("c:"):
                    referenced.add(ref[2:])
        for canonical in sorted(referenced):
            if self.store.node(canonical, rev) is None:
                raise ShareError(400, f"unknown canonical anchor {canonical!r}")

    def push(self, token: str, bundle: Mapping[str, Any]) -> dict[str, Any]:
        contributor = self._require_contributor(token)
        problems = validate_bundle(bundle)
        if problems:
            raise ShareError(400, "; ".join(problems[:5]))
        self._check_anchors(bundle)
        self._expire_stale()
        # Normalize BEFORE anything durable: the queue feeds the maintainer's
        # review (an LLM on their machine) — injection must die at the door,
        # not at apply, and a contributor who bypassed the CLI gets the same
        # boundary the CLI applies (§6.4).
        clean = sanitize_bundle(bundle)
        key = str(clean["idempotency_key"])
        if not key:
            raise ShareError(400, "idempotency_key did not survive sanitization")
        existing = self.store.conn.execute(
            "SELECT queue_id, status, contributor, bundle FROM share_queue WHERE idempotency_key=?",
            (key,),
        ).fetchone()
        if existing is not None:
            # Idempotent retry: the same key never creates a second queue item
            # — but only for the same contributor pushing the same content.
            if existing["contributor"] != contributor or bundle_content_digest(
                json.loads(existing["bundle"])
            ) != bundle_content_digest(clean):
                raise ShareError(409, "idempotency key already used by a different contributor or content")
            if existing["status"] != "expired":
                return {"queue_id": existing["queue_id"], "status": existing["status"], "duplicate": True}
            # Expired + same content: revive instead of parroting the dead row
            # forever (round 8 — the CLI reuses the key by content digest, so
            # without this an expired push could never re-enter the queue).
            # Falls through to the quota checks: a revival re-enters pending.
            self._check_quotas(contributor)
            self.store.conn.execute(
                "UPDATE share_queue SET status='pending', received_at=?,"
                " lease_token=NULL, lease_until=NULL WHERE queue_id=?",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), existing["queue_id"]),
            )
            return {"queue_id": existing["queue_id"], "status": "pending", "revived": True}
        # Quotas AFTER the idempotency lookup: a legitimate retry of an
        # already-queued push must return the duplicate, not a 429.
        self._check_quotas(contributor)
        cursor = self.store.conn.execute(
            "INSERT INTO share_queue(idempotency_key, contributor, bundle, received_at)"
            " VALUES (?, ?, ?, ?)",
            (
                key,
                contributor,
                json.dumps(clean, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return {"queue_id": int(cursor.lastrowid), "status": "pending", "duplicate": False}

    def push_status(self, queue_id: int, *, token: str = "", maintainer: str = "") -> dict[str, Any]:
        row = self.store.conn.execute(
            "SELECT queue_id, status, verdict, assigned, contributor FROM share_queue WHERE queue_id=?",
            (queue_id,),
        ).fetchone()
        if row is None:
            raise ShareError(404, f"unknown queue item {queue_id}")
        # Queue ids are guessable; the status (verdict note, assignment map)
        # belongs to the contributor who pushed it — or the maintainer.
        if maintainer != self.maintainer_token and (
            not token or token != row["contributor"]
        ):
            raise ShareError(403, "this queue item belongs to a different contributor")
        return {
            "queue_id": row["queue_id"],
            "status": row["status"],
            "verdict": row["verdict"],
            "assigned": json.loads(row["assigned"] or "{}"),
        }

    # ---- review queue (maintainer) ------------------------------------

    def peek(self, token: str, *, queue_id: int | None = None) -> dict[str, Any]:
        """Read-only queue view: nothing is leased. The dry-run half of the
        review CLI uses this — looking must not lock (round 7)."""

        self._require_maintainer(token)
        self._expire_stale()
        rows = self.store.conn.execute(
            "SELECT queue_id, contributor, bundle, received_at, verdict_version, lease_until"
            " FROM share_queue WHERE status='pending'"
            + (" AND queue_id=?" if queue_id is not None else "")
            + " ORDER BY queue_id LIMIT 50",
            ((queue_id,) if queue_id is not None else ()),
        ).fetchall()
        now = time.time()
        items = [
            {
                "queue_id": row["queue_id"],
                "contributor": row["contributor"],
                "received_at": row["received_at"],
                "verdict_version": row["verdict_version"],
                "leased": bool(row["lease_until"] and row["lease_until"] >= now),
                "bundle": json.loads(row["bundle"]),
                "merge_hints": self._merge_hints(json.loads(row["bundle"])),
            }
            for row in rows
        ]
        return {"items": items}

    def release(self, token: str, *, queue_id: int, lease_token: str) -> dict[str, Any]:
        self._require_maintainer(token)
        updated = self.store.conn.execute(
            "UPDATE share_queue SET lease_token=NULL, lease_until=NULL"
            " WHERE queue_id=? AND status='pending' AND lease_token=?",
            (queue_id, lease_token),
        )
        if updated.rowcount == 0:
            raise ShareError(409, "nothing to release: wrong lease token or item not pending")
        return {"queue_id": queue_id, "released": True}

    def lease(
        self,
        token: str,
        *,
        seconds: int = DEFAULT_LEASE_SECONDS,
        limit: int = 10,
        queue_id: int | None = None,
    ) -> dict[str, Any]:
        self._require_maintainer(token)
        self._expire_stale()
        now = time.time()
        rows = self.store.conn.execute(
            "SELECT queue_id, contributor, bundle, received_at, verdict_version FROM share_queue"
            " WHERE status='pending' AND (lease_until IS NULL OR lease_until < ?)"
            + (" AND queue_id=?" if queue_id is not None else "")
            + " ORDER BY queue_id LIMIT ?",
            ((now, queue_id, max(1, limit)) if queue_id is not None else (now, max(1, limit))),
        ).fetchall()
        items = []
        for row in rows:
            lease_token = secrets.token_urlsafe(12)
            claimed = self.store.conn.execute(
                "UPDATE share_queue SET lease_token=?, lease_until=? WHERE queue_id=?"
                " AND status='pending' AND (lease_until IS NULL OR lease_until < ?)",
                (lease_token, now + seconds, row["queue_id"], now),
            )
            if claimed.rowcount == 0:
                continue  # another maintainer session got it between SELECT and UPDATE
            bundle = json.loads(row["bundle"])
            items.append(
                {
                    "queue_id": row["queue_id"],
                    "contributor": row["contributor"],
                    "received_at": row["received_at"],
                    "verdict_version": row["verdict_version"],
                    "lease_token": lease_token,
                    "bundle": bundle,
                    "merge_hints": self._merge_hints(bundle),
                }
            )
        return {"items": items, "lease_seconds": seconds}

    def _merge_hints(self, bundle: Mapping[str, Any]) -> dict[str, list[str]]:
        """Fingerprint-based merge candidates (§6.2): for each incoming node
        without a canonical id, existing server nodes with the same normalized
        surface. Hints only — the maintainer decides ``merge`` in the verdict;
        nothing merges automatically (plan §9)."""

        from ..node.matching import scan_normalize

        surface_of: dict[str, list[str]] = {}
        rev = self.store.current_rev()
        for kind in ("subject", "term"):
            for node in self.store.nodes_of_kind(kind, rev):
                key = scan_normalize(str(node.payload.get("surface") or ""))
                if key:
                    surface_of.setdefault(key, []).append(node.local_id)
        hints: dict[str, list[str]] = {}
        for node in bundle.get("nodes") or []:
            if node.get("canonical_id"):
                continue
            key = scan_normalize(str((node.get("payload") or {}).get("surface") or ""))
            if key and key in surface_of:
                hints[str(node.get("handle"))] = list(surface_of[key])
        return hints

    def _check_merge(
        self, bundle: Mapping[str, Any], merge: Mapping[str, str], *, override: str
    ) -> None:
        """Structural admission of the verdict's ``merge`` map AND the bundle's
        projected membership graph (reviews 2026-08-27 rounds 8–9). Merge map:
        keys are bundle nodes, kinds match, targets are distinct, a target
        outside the fingerprint hints needs an explicit override reason (the
        hints are the only server-side signal the merge is plausible). Graph:
        the DAG check runs on the FULL projection — server edges plus bundle
        edges mapped through c:/merge with symbolic new nodes — and runs even
        with an empty merge map, because a cycle can be built entirely from
        new nodes or routed through one."""

        rev = self.store.current_rev()
        nodes = {str(node.get("handle")): node for node in bundle.get("nodes") or []}
        hints = self._merge_hints(bundle)
        seen_targets: dict[str, str] = {}
        for handle, canonical in merge.items():
            node = nodes.get(handle)
            if node is None:
                raise ShareError(400, f"merge key {handle!r} is not a bundle node")
            if str(node.get("canonical_id") or ""):
                raise ShareError(
                    400,
                    f"merge key {handle!r} is already anchored to a canonical node —"
                    " merge is for NEW duplicates only",
                )
            target = self.store.node(canonical, rev)
            if target is None:
                raise ShareError(400, f"merge target {canonical!r} does not exist")
            if target.kind != str(node.get("kind") or ""):
                raise ShareError(
                    400,
                    f"merge {handle!r} ({node.get('kind')}) into {canonical!r} ({target.kind}):"
                    " kinds must match",
                )
            if canonical in seen_targets:
                raise ShareError(
                    400,
                    f"merge targets must be distinct: {handle!r} and {seen_targets[canonical]!r}"
                    f" both map to {canonical!r}",
                )
            seen_targets[canonical] = handle
            if canonical not in (hints.get(handle) or []) and not str(override).strip():
                raise ShareError(
                    409,
                    f"merge target {canonical!r} for {handle!r} is not among the fingerprint"
                    " hints; supply an explicit override reason",
                )

        # Projected membership graph (rounds 9–10): server edges + every
        # bundle edge mapped through the FINAL assignment — ``c:`` anchors,
        # the verdict merge map AND each node's own ``canonical_id`` (a node
        # anchored to B whose edge reads ``n1 → c:A`` actually lands as
        # ``B → A``; leaving n1 symbolic let that cycle through, round 10).
        # Unmapped new nodes stay as their own symbolic vertices, and the DAG
        # check runs even with no merge map at all.

        canonical_of = {
            str(node.get("handle")): str(node.get("canonical_id") or "")
            for node in bundle.get("nodes") or []
            if node.get("canonical_id")
        }

        def mapped(ref: str) -> str:
            text = str(ref or "")
            if text.startswith("c:"):
                return text[2:]
            if text in merge:
                return merge[text]
            return canonical_of.get(text, text)

        edges: list[tuple[str, str]] = []
        adjacency: dict[str, set[str]] = {}
        for membership in bundle.get("memberships") or []:
            parent = mapped(membership.get("parent"))
            child = mapped(membership.get("child"))
            if parent == child:
                raise ShareError(400, f"membership makes {parent!r} its own parent")
            edges.append((parent, child))
            adjacency.setdefault(parent, set()).add(child)

        def descendants_step(node_id: str) -> set[str]:
            out = set(adjacency.get(node_id, ()))
            if self.store.node(node_id, rev) is not None:
                out.update(m.child_id for m in self.store.children(node_id, rev))
            return out

        for parent, child in edges:
            seen = {child}
            frontier = [child]
            while frontier:
                for nxt in descendants_step(frontier.pop()):
                    if nxt == parent:
                        raise ShareError(
                            400,
                            f"membership graph closes a cycle through {parent!r} and {child!r}",
                        )
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)

    def verdict(
        self,
        token: str,
        *,
        queue_id: int,
        lease_token: str,
        expected_version: int,
        verdict: str,
        reason: str = "",
        merge: Mapping[str, str] | None = None,
        evidence: list[Mapping[str, Any]] | None = None,
        override: str = "",
    ) -> dict[str, Any]:
        """``merge`` is the §6.2 ``merge_into`` half of an approval: bundle
        handle → existing canonical id. The mapped node attaches its items /
        memberships / links to that node instead of forking a duplicate; its
        payload is NOT taken (the maintainer reviewed the duplicate, content
        differences go back through the normal update path).

        ``evidence`` carries the review session's URL-backed corroborations
        (§6.3): rows reference bundle handles and are booked into the server
        store as ``evidence_kind=external`` in the same approval transaction,
        re-keyed through the handle→canonical assignment. Every row must match
        a claim **frozen at enqueue** — ``bundle_claims`` over the stored
        bundle plus its ``claim_summaries`` — with the actual value hash; a
        row referencing anything else rejects the whole verdict (a review
        model must not be able to invent confirmed evidence, round 7).

        Approval is gated on the §6.3 thresholds: unsatisfied non-prose
        claims (counting this verdict's evidence) refuse the approve unless
        the maintainer passes an explicit ``override`` reason, which is
        recorded in the verdict note."""

        self._require_maintainer(token)
        if verdict not in ("approve", "approve_tentative", "reject"):
            raise ShareError(400, f"unknown verdict {verdict!r}")
        tentative = verdict == "approve_tentative"
        if verdict == "reject":
            updated = self.store.conn.execute(
                "UPDATE share_queue SET status='rejected', verdict=?,"
                " verdict_version=verdict_version+1, lease_token=NULL, lease_until=NULL"
                " WHERE queue_id=? AND status='pending' AND verdict_version=? AND lease_token=?"
                " AND lease_until >= ?",
                (sanitize_text(reason, max_chars=500), queue_id, expected_version, lease_token, time.time()),
            )
            if updated.rowcount == 0:
                raise ShareError(409, "verdict CAS failed: stale version, lost lease, or already decided")
            return {"queue_id": queue_id, "status": "rejected"}

        row = self.store.conn.execute(
            "SELECT bundle, status, verdict_version, lease_token, lease_until FROM share_queue"
            " WHERE queue_id=?",
            (queue_id,),
        ).fetchone()
        if row is None:
            raise ShareError(404, f"unknown queue item {queue_id}")
        bundle = json.loads(row["bundle"])
        self._check_merge(bundle, merge or {}, override=override)
        if tentative:
            problem = tentative_ineligible_reason(self.store, bundle)
            if problem:
                raise ShareError(400, f"approve_tentative refused: {problem}")

        # Evidence must reference the frozen claim set exactly (round 7).
        # ``bundle_claims`` ONLY: its hashes come from the bundle's actual
        # values. The contributor's self-reported claim_summaries are not a
        # reference set — an arbitrary hash there must not become bookable
        # "confirmed external" evidence.
        allowed = {
            (claim["node"], claim["field_path"], claim["value_hash"])
            for claim in bundle_claims(bundle)
        }
        checked_evidence: list[dict[str, Any]] = []
        for entry in evidence or []:
            key = (
                str(entry.get("node") or ""),
                str(entry.get("field_path") or ""),
                str(entry.get("value_hash") or ""),
            )
            url = sanitize_text(str(entry.get("url") or ""), max_chars=500)
            if key not in allowed or not _STRICT_URL_RE.fullmatch(url):
                raise ShareError(
                    400,
                    f"external evidence does not match a frozen claim (or has a bad URL): {key}",
                )
            checked_evidence.append({"node": key[0], "field_path": key[1], "value_hash": key[2], "url": url})

        # §6.3 gate: unsatisfied non-prose claims block the approve unless the
        # maintainer explicitly overrides with a reason. Merge-covered scalar
        # claims are exempt (their payload never lands).
        pending = unsatisfied_claims(
            threshold_report(bundle, external_evidence=checked_evidence), merge=merge
        )
        if pending and not tentative and not str(override).strip():
            summary = "; ".join(f"{p['node']} {p['field_path']} [{p['slot']}]" for p in pending[:5])
            raise ShareError(
                409,
                f"{len(pending)} claim(s) below the §6.3 threshold ({summary}); "
                "supply matching external evidence or an explicit override reason",
            )
        note = sanitize_text(reason, max_chars=500)
        if tentative:
            # sub-threshold admission IS the point (plan §11.5): the content
            # enters shadow-only until corroborated, no override needed
            note = f"{note} [tentative]".strip()
        elif pending:
            note = f"{note} [override: {sanitize_text(str(override), max_chars=200)}]".strip()

        # One transaction: CAS + apply + chain append (see module docstring).
        with self.store.begin("share", task_id=f"share-queue-{queue_id}", note="share approve") as txn:
            claimed = txn.conn.execute(
                "UPDATE share_queue SET status='approved', verdict=?,"
                " verdict_version=verdict_version+1, lease_token=NULL, lease_until=NULL"
                " WHERE queue_id=? AND status='pending' AND verdict_version=? AND lease_token=?"
                " AND lease_until >= ?",
                (note, queue_id, expected_version, lease_token, time.time()),
            )
            if claimed.rowcount == 0:
                raise ShareError(409, "verdict CAS failed: stale version, lost lease, or already decided")
            assigned = _apply_bundle(
                self.store, txn, bundle, merge=merge or {},
                maturity="tentative" if tentative else "normal",
            )
            txn.conn.execute(
                "UPDATE share_queue SET assigned=? WHERE queue_id=?",
                (json.dumps(assigned, ensure_ascii=False), queue_id),
            )
            from ..node.model import digest as value_digest
            from ..node.model import payload_group_hash
            from ..node.signals import record_evidence

            for row in checked_evidence:  # pre-validated against the frozen claim set
                node_id = assigned.get(row["node"])
                field_path = row["field_path"]
                if field_path.startswith("items/"):
                    item_id = assigned.get(field_path.split("/", 1)[1])
                    field_path = f"items/{item_id}" if item_id else ""
                if not (node_id and field_path):
                    continue  # e.g. an item deduped away during apply: nothing to attach to
                # The assignment may have landed elsewhere than the reviewed
                # bundle said (merge, dedupe onto a variant spelling): the
                # evidence hash must equal the CURRENT claim of what it lands
                # on, or the row would be born stale and could later promote
                # a value nobody verified (round 11).
                if field_path.startswith("items/"):
                    landed = next(
                        (i for i in self.store.items_of(node_id, txn.rev)
                         if i.item_id == field_path.split("/", 1)[1]),
                        None,
                    )
                    if landed is None or value_digest(landed.value) != row["value_hash"]:
                        continue
                else:  # payload:<group> — the only other shape bundle_claims emits
                    landed_node = self.store.node(node_id, txn.rev)
                    group = field_path.split(":", 1)[1]
                    if landed_node is None or payload_group_hash(
                        landed_node.kind, landed_node.payload, group
                    ) != row["value_hash"]:
                        continue
                record_evidence(
                    self.store,
                    node_id=node_id,
                    field_path=field_path,
                    value_hash=row["value_hash"],
                    verdict="confirmed",
                    evidence_kind="external",
                    source_ref=row["url"],
                    task_id=f"share-queue-{queue_id}",
                    algo_version="review-external-1",
                )
            content = snapshot_content(self.store, txn.rev)
            digest_value = content_digest(content)
            prev = self.store.conn.execute(
                "SELECT chain_hash FROM share_chain ORDER BY rev DESC LIMIT 1"
            ).fetchone()["chain_hash"]
            txn.conn.execute(
                "INSERT INTO share_chain(rev, content_digest, chain_hash) VALUES (?, ?, ?)",
                (txn.rev, digest_value, chain_hash(prev, digest_value)),
            )
        return {
            "queue_id": queue_id,
            "status": "approved",
            "assigned": assigned,
            **({"tentative": True} if tentative else {}),
        }


    # ---- digest (plan §11.5) -------------------------------------------

    def digest(self, token: str) -> dict[str, Any]:
        """One digest pass: expire stale queue items, aggregate cross-
        contributor votes per claim (ADVISORY ONLY — anonymous tokens are
        write credentials, not evidence identity, so counts satisfy no
        threshold until an anti-Sybil identity exists), assemble the
        maintainer worklist, run the gated auto-tentative path, and nominate
        long-uncorroborated tentative entities for retirement (nomination
        only: nothing is deleted automatically, plan §9)."""

        self._require_maintainer(token)
        self._expire_stale()
        rows = self.store.conn.execute(
            "SELECT queue_id, contributor, bundle FROM share_queue WHERE status='pending'"
            " ORDER BY queue_id"
        ).fetchall()
        reputable = {
            r["contributor"]
            for r in self.store.conn.execute(
                "SELECT DISTINCT contributor FROM share_queue WHERE status='approved'"
            )
        }
        votes: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            bundle = json.loads(row["bundle"])
            for claim in bundle_claims(bundle):
                votes.setdefault((claim["field_path"], claim["value_hash"]), set()).add(
                    row["contributor"]
                )
        auto: list[dict[str, Any]] = []
        worklist: list[dict[str, Any]] = []
        for row in rows:
            bundle = json.loads(row["bundle"])
            ineligible = tentative_ineligible_reason(self.store, bundle)
            eligible = ineligible is None and row["contributor"] in reputable
            if self.auto_tentative and eligible:
                leased = self.lease(token, limit=1, queue_id=row["queue_id"]).get("items") or []
                if not leased:
                    continue  # someone is reviewing it right now: leave it to them
                reply = self.verdict(
                    token,
                    queue_id=row["queue_id"],
                    lease_token=leased[0]["lease_token"],
                    expected_version=leased[0]["verdict_version"],
                    verdict="approve_tentative",
                    reason="digest auto-tentative (reputable contributor, new terms only)",
                )
                auto.append({"queue_id": row["queue_id"], "assigned": reply["assigned"]})
                continue
            pending_claims = unsatisfied_claims(threshold_report(bundle))
            worklist.append(
                {
                    "queue_id": row["queue_id"],
                    "contributor": row["contributor"][:8],
                    "eligible_tentative": eligible,
                    **({"tentative_blocker": ineligible} if ineligible else {}),
                    "pending_claims": [
                        {
                            "node": c["node"],
                            "field_path": c["field_path"],
                            "slot": c["slot"],
                            "label": c["label"],
                            # advisory context only, never a gate input:
                            "contributor_votes": len(
                                votes.get((c["field_path"], c["value_hash"]), set()) & reputable
                            ),
                        }
                        for c in pending_claims
                    ],
                }
            )
        promoted = self._promote_corroborated()
        nominations = self._retire_nominations()
        return {
            "auto_tentative": auto,
            "worklist": worklist,
            "promoted": promoted,
            "tentative_retire_candidates": nominations,
        }

    def _claim_confirmed(self, node_id: str, field_path: str, value_hash: str) -> bool:
        """Corroboration for exactly the CURRENT claim (round 10): an old
        value hash, an unrelated field or another item must not stand in."""

        return (
            self.store.conn.execute(
                "SELECT 1 FROM evidence WHERE node_id=? AND field_path=? AND value_hash=?"
                " AND verdict='confirmed' LIMIT 1",
                (node_id, field_path, value_hash),
            ).fetchone()
            is not None
        )

    def _tentative_entities(self) -> tuple[list[Any], list[sqlite3.Row]]:
        rev = self.store.current_rev()
        nodes = [
            node
            for row in self.store.conn.execute(
                "SELECT local_id FROM node_versions WHERE valid_to_rev IS NULL AND maturity='tentative'"
            )
            if (node := self.store.node(row["local_id"], rev)) is not None
        ]
        items = self.store.conn.execute(
            "SELECT v.item_id, i.local_id, i.field, v.value, r.created_at FROM item_versions v"
            " JOIN items i USING(item_id) JOIN revisions r ON r.rev = v.valid_from_rev"
            " WHERE v.valid_to_rev IS NULL AND v.maturity='tentative'"
        ).fetchall()
        return nodes, items

    def _promote_corroborated(self) -> list[dict[str, Any]]:
        """Digest promotion (round 10): a tentative node whose CURRENT core
        claim has confirmed evidence — or a tentative item whose current
        value does — flips to normal, node and item independently, in one
        revision with its own chain entry."""

        from ..node.model import digest as value_digest
        from ..node.model import payload_group_hash

        nodes, items = self._tentative_entities()
        node_ids = [
            node.local_id
            for node in nodes
            if self._claim_confirmed(
                node.local_id, "payload:core", payload_group_hash(node.kind, node.payload, "core")
            )
        ]
        item_rows = [
            row
            for row in items
            if self._claim_confirmed(
                row["local_id"], f"items/{row['item_id']}", value_digest(row["value"])
            )
        ]
        if not node_ids and not item_rows:
            return []
        promoted: list[dict[str, Any]] = []
        with self.store.begin("share", task_id="digest", note="digest promotion") as txn:
            for node_id in node_ids:
                txn.update_node(node_id, maturity="normal")
                promoted.append({"node": node_id})
            for row in item_rows:
                txn.update_item(row["item_id"], maturity="normal")
                promoted.append({"item": row["item_id"], "node": row["local_id"]})
            content = snapshot_content(self.store, txn.rev)
            digest_value = content_digest(content)
            prev = self.store.conn.execute(
                "SELECT chain_hash FROM share_chain ORDER BY rev DESC LIMIT 1"
            ).fetchone()["chain_hash"]
            txn.conn.execute(
                "INSERT INTO share_chain(rev, content_digest, chain_hash) VALUES (?, ?, ?)",
                (txn.rev, digest_value, chain_hash(prev, digest_value)),
            )
        return promoted

    def _retire_nominations(self) -> list[dict[str, Any]]:
        """Nomination only (plan §9: nothing auto-deletes) — tentative nodes
        AND items past the review window whose current claim never got
        confirmed. Runs after promotion, so freshly promoted entities are
        already out of scope."""

        from ..node.model import digest as value_digest
        from ..node.model import payload_group_hash

        cutoff = datetime.now(timezone.utc).timestamp() - TENTATIVE_REVIEW_DAYS * 86_400
        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(timespec="seconds")
        nominations: list[dict[str, Any]] = []
        rev = self.store.current_rev()
        for row in self.store.conn.execute(
            "SELECT v.local_id, r.created_at FROM node_versions v"
            " JOIN revisions r ON r.rev = v.valid_from_rev"
            " WHERE v.valid_to_rev IS NULL AND v.maturity='tentative' AND r.created_at < ?",
            (cutoff_iso,),
        ).fetchall():
            node = self.store.node(row["local_id"], rev)
            if node is None or self._claim_confirmed(
                node.local_id, "payload:core", payload_group_hash(node.kind, node.payload, "core")
            ):
                continue
            nominations.append(
                {
                    "node": node.local_id,
                    "label": node.payload.get("surface") or node.payload.get("field") or "",
                    "tentative_since": row["created_at"],
                }
            )
        for row in self.store.conn.execute(
            "SELECT v.item_id, i.local_id, i.field, v.value, r.created_at FROM item_versions v"
            " JOIN items i USING(item_id) JOIN revisions r ON r.rev = v.valid_from_rev"
            " WHERE v.valid_to_rev IS NULL AND v.maturity='tentative' AND r.created_at < ?",
            (cutoff_iso,),
        ).fetchall():
            if self._claim_confirmed(
                row["local_id"], f"items/{row['item_id']}", value_digest(row["value"])
            ):
                continue
            nominations.append(
                {
                    "item": row["item_id"],
                    "node": row["local_id"],
                    "label": f"{row['field']}:{row['value']}",
                    "tentative_since": row["created_at"],
                }
            )
        return nominations


def tentative_ineligible_reason(store: KnowledgeStore, bundle: Mapping[str, Any]) -> str | None:
    """Why a bundle may NOT enter as tentative, or ``None`` when it may.

    Tentative is for NEW entities only (plan §11.5): new nodes must be terms
    (the only slot whose distribution risk is a bad-but-shadowed suggestion),
    and a node anchored to an existing canonical id must carry the identical
    payload — a payload update smuggled through the sub-threshold door would
    change verified content without evidence. Items on existing nodes are
    fine: they land item-level tentative."""

    rev = store.current_rev()
    for node in bundle.get("nodes") or []:
        canonical = str(node.get("canonical_id") or "")
        if canonical:
            current = store.node(canonical, rev)
            if current is None:
                return f"unknown canonical anchor {canonical!r}"
            if sanitize_payload(node.get("payload") or {}) != current.payload:
                return f"node {node.get('handle')!r} updates existing {canonical!r} — not tentative material"
            continue
        if str(node.get("kind")) != "term":
            return f"new node {node.get('handle')!r} is {node.get('kind')!r} — tentative admits terms only"
    return None


def _apply_bundle(
    store: KnowledgeStore,
    txn: Transaction,
    bundle: Mapping[str, Any],
    *,
    merge: Mapping[str, str] = {},
    maturity: str = "normal",
) -> dict[str, str]:
    """Fold an approved bundle into the server corpus. On the server the
    store's local_id *is* the canonical id; new entities get fresh ones and
    the handle→canonical assignment is returned for the contributor to
    backfill. Known-value items/memberships dedupe by normalized identity so
    two contributors adding the same alias converge on one canonical row."""

    assigned: dict[str, str] = {}

    def resolve(ref: str) -> str | None:
        text = str(ref or "")
        if text.startswith("c:"):
            canonical = text[2:]
            return canonical if store.node(canonical, txn.rev) is not None else None
        return assigned.get(text)

    for node in bundle.get("nodes") or []:
        payload = sanitize_payload(node.get("payload") or {})
        handle = str(node.get("handle"))
        merged_into = str(merge.get(handle) or "")
        if merged_into:
            if store.node(merged_into, txn.rev) is None:
                raise ShareError(400, f"merge target {merged_into!r} does not exist")
            assigned[handle] = merged_into
            continue
        canonical = str(node.get("canonical_id") or "")
        if canonical:
            current = store.node(canonical, txn.rev)
            if current is None:
                # push admission checks anchors too; this is defense in depth —
                # a client-named id must never materialize as a new entity
                raise ShareError(400, f"unknown canonical anchor {canonical!r}")
            if current.kind != str(node.get("kind")):
                raise ShareError(400, f"kind mismatch on canonical anchor {canonical!r}")
            # A full-threshold (normal) approval covering an existing
            # TENTATIVE node is its promotion (round 10: tentative needs a
            # real path back to normal) — one version for both changes.
            promote = maturity == "normal" and current.maturity == "tentative"
            if current.payload != payload or promote:
                txn.update_node(
                    canonical,
                    payload=payload if current.payload != payload else None,
                    maturity="normal" if promote else None,
                )
            assigned[handle] = canonical
            continue
        canonical = uuid.uuid4().hex  # new entity ids are the server's to assign
        txn.create_node(canonical, str(node.get("kind")), payload, canonical_id=canonical,
                        visibility="shareable", maturity=maturity)
        assigned[handle] = canonical

    for item in bundle.get("items") or []:
        owner = resolve(item.get("node"))
        if owner is None:
            continue
        value = sanitize_text(str(item.get("value") or ""))
        field = str(item.get("field"))
        existing = next(
            (
                candidate
                for candidate in store.items_of(owner, txn.rev)
                if candidate.field == field
                and _match_normalize(candidate.value) == _match_normalize(value)
            ),
            None,
        )
        if existing is not None:
            if maturity == "normal" and existing.maturity == "tentative":
                # A second, fully-reviewed submission of the same NORMALIZED
                # value promotes the tentative item (round 10) — and adopts
                # the reviewed exact representation in the same version: the
                # gate and any external evidence hashed the incoming value,
                # so promoting while keeping a variant spelling would leave
                # the promoted item's current claim unconfirmed (round 11).
                txn.update_item(
                    existing.item_id,
                    value=value if existing.value != value else None,
                    maturity="normal",
                )
            assigned[str(item.get("handle"))] = existing.item_id
            continue
        item_id = uuid.uuid4().hex
        txn.create_item(item_id, owner, field, value, maturity=maturity)
        assigned[str(item.get("handle"))] = item_id

    order = 0
    for membership in bundle.get("memberships") or []:
        parent = resolve(membership.get("parent"))
        child = resolve(membership.get("child"))
        if parent is None or child is None:
            continue
        section = sanitize_text(str(membership.get("section") or ""), max_chars=100)
        if any(
            m.child_id == child and m.section == section
            for m in store.children(parent, txn.rev)
        ):
            continue
        order += 1
        txn.create_membership(
            uuid.uuid4().hex, parent, child, section, int(membership.get("order_key") or order)
        )

    for link in bundle.get("links") or []:
        source = resolve(link.get("source"))
        target = resolve(link.get("target"))
        rel = str(link.get("rel") or "")
        if source is None or target is None or rel not in ("see_also", "supersedes"):
            continue
        if any(
            l.rel == rel and l.target_id == target for l in store.links_from(source, txn.rev)
        ):
            continue
        txn.create_link(uuid.uuid4().hex, source, rel, target)

    return assigned


# ---------------------------------------------------------------------------
# HTTP layer


class _Handler(BaseHTTPRequestHandler):
    service: ShareService  # set by serve()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # quiet; the CLI front end owns stdout

    def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    MAX_BODY_BYTES = 2_000_000

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > self.MAX_BODY_BYTES:
            raise ShareError(413, f"request body over {self.MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ShareError(400, f"invalid JSON body: {exc}")
        return parsed if isinstance(parsed, dict) else {}

    def _dispatch(self) -> None:
        service = self.service
        # The body is read OUTSIDE the service lock: a slow or oversized
        # upload must stall its own connection, not every other request.
        try:
            body = self._body() if self.command == "POST" else {}
        except ShareError as exc:
            self._reply(exc.status, {"error": str(exc)})
            return
        with service.lock:
            self._routed(service, body)

    def _routed(self, service: "ShareService", body: dict[str, Any]) -> None:
        try:
            path = self.path.split("?", 1)[0]
            token = self.headers.get("X-Share-Token", "")
            maintainer = self.headers.get("X-Maintainer-Token", "")
            if self.command == "POST" and path == "/register":
                self._reply(200, service.register())
            elif self.command == "GET" and path == "/snapshot":
                self._reply(200, service.snapshot())
            elif self.command == "POST" and path == "/push":
                self._reply(200, service.push(token, body))
            elif self.command == "GET" and re.fullmatch(r"/push/\d+", path):
                self._reply(
                    200,
                    service.push_status(
                        int(path.rsplit("/", 1)[1]), token=token, maintainer=maintainer
                    ),
                )
            elif self.command == "GET" and path in ("/queue", "/queue/peek"):
                query = self.path.partition("?")[2]
                params: dict[str, str] = {}
                for pair in query.split("&"):
                    if "=" in pair:
                        key, _, value = pair.partition("=")
                        params[key] = value
                wanted = int(params["queue_id"]) if params.get("queue_id") else None
                if path == "/queue/peek":
                    self._reply(200, service.peek(maintainer, queue_id=wanted))
                else:
                    self._reply(
                        200,
                        service.lease(
                            maintainer,
                            seconds=int(params.get("lease_seconds") or DEFAULT_LEASE_SECONDS),
                            limit=int(params.get("limit") or 10),
                            queue_id=wanted,
                        ),
                    )
            elif self.command == "POST" and path == "/release":
                self._reply(
                    200,
                    service.release(
                        maintainer,
                        queue_id=int(body.get("queue_id") or 0),
                        lease_token=str(body.get("lease_token") or ""),
                    ),
                )
            elif self.command == "POST" and path == "/digest":
                self._reply(200, service.digest(maintainer))
            elif self.command == "POST" and path == "/verdict":
                self._reply(
                    200,
                    service.verdict(
                        maintainer,
                        queue_id=int(body.get("queue_id") or 0),
                        lease_token=str(body.get("lease_token") or ""),
                        expected_version=int(body.get("expected_version") or 0),
                        verdict=str(body.get("verdict") or ""),
                        reason=str(body.get("reason") or ""),
                        merge=body.get("merge") or {},
                        evidence=body.get("evidence") or [],
                        override=str(body.get("override") or ""),
                    ),
                )
            else:
                self._reply(404, {"error": f"no route {self.command} {path}"})
        except ShareError as exc:
            self._reply(exc.status, {"error": str(exc)})
        except Exception as exc:  # a broken request must not kill the server
            self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    do_GET = _dispatch
    do_POST = _dispatch


def serve(
    root: str | Path, *, port: int, maintainer_token: str, auto_tentative: bool = False
) -> ThreadingHTTPServer:
    """Bind and return the server (caller decides threading / serve_forever)."""

    service = ShareService(root, maintainer_token=maintainer_token, auto_tentative=auto_tentative)
    handler = type("BoundHandler", (_Handler,), {"service": service})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.share_service = service  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finesub.llm.knowledge.share.server",
        description="Local knowledge share server (plan §6; TLS belongs to Caddy/tunnel).",
    )
    parser.add_argument("--root", required=True, help="data directory OUTSIDE the repo")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--maintainer-token",
        default=None,
        help="review-endpoint token (default: generated and printed once)",
    )
    parser.add_argument(
        "--auto-tentative",
        action="store_true",
        help="digest auto-admits reputable contributors' new-term bundles as tentative"
        " (plan §11.5; DEFAULT OFF per O12 — the review session stays the only route)",
    )
    args = parser.parse_args(argv)
    token = args.maintainer_token or secrets.token_urlsafe(24)
    httpd = serve(args.root, port=args.port, maintainer_token=token, auto_tentative=args.auto_tentative)
    if not args.maintainer_token:
        print(f"maintainer token: {token}")
    print(f"share server on http://127.0.0.1:{args.port} (root: {args.root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        httpd.share_service.close()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    sys.exit(main())
