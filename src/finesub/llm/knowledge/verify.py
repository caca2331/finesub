"""KB verification task (plan §11.4 / O9): the share-review session
localized. Deterministic half: enumerate claims that lack any evidence for
their CURRENT value on the externally-verifiable slots (term / fact — the
only routes a web search can settle). LLM half (dry-run by default, repo
convention): one native-search session over the batch, whose conclusions are
booked as ``evidence_kind=external`` rows — ``confirmed`` needs a URL,
``unverifiable`` is a terminal state so the claim is not re-tried every
sweep. Relation/event/prose claims are never sent out: nothing on the web
can settle them (§6.3), so they would only burn quota."""

from __future__ import annotations

import re
from typing import Any, Mapping

import json

from ..output_tags import find_tag_blocks
from ..prompt_compose import load_prompt_template
from .node.model import payload_group_hash
from .node.render import format_line, node_aliases
from .node.signals import record_evidence
from .node.store import KnowledgeStore

VERIFY_MAX_TOKENS = 8_192
VERIFY_ALGO_VERSION = "kb-verify-1"
_URL_RE = re.compile(r"https?://\S+")

#: Which lines may be sent to an external search is a DATA decision, not a
#: kind decision (review 2026-08-29 P1-2): `[中之人]` is an unregistered label
#: and 人际关系 holds real people, so both stay home. `verify` resolves
#: label -> section -> `unknown_label_verify`, fail-closed at "none".
_VERIFIABLE_KINDS = ("term", "note")


def _placements(store: KnowledgeStore, rev: int) -> dict[str, set[tuple[str, str]]]:
    """``node -> {(section, owning subject's category)}`` — EVERY placement,
    not the first one seen. A node can hang under two subjects, and taking
    whichever membership came first would let a permissive section speak for
    a restrictive one (review 2026-08-29 P1-2)."""

    placements: dict[str, set[tuple[str, str]]] = {}
    for subject in store.subjects(rev):
        category = str(subject.payload.get("category") or "")
        stack = [subject.local_id]
        seen = {subject.local_id}
        while stack:
            parent = stack.pop()
            for membership in store.children(parent, rev):
                placements.setdefault(membership.child_id, set()).add(
                    (membership.section, category)
                )
                if membership.child_id not in seen:
                    seen.add(membership.child_id)
                    stack.append(membership.child_id)
    return placements


def _externally_verifiable(node: Any, placements: dict[str, set[tuple[str, str]]]) -> bool:
    """A line goes out only if EVERY placement says it may. Fail-closed: an
    orphan, an unknown category, or one restrictive parent keeps it home."""

    from .node.presets import preset_for_category

    where = placements.get(node.local_id)
    if not where:
        return False  # orphan: fail closed
    label = str(node.payload.get("label") or "") or None
    for section, category in where:
        if not category:
            return False
        try:
            preset = preset_for_category(category)
        except ValueError:
            return False
        if preset.verify_for(section, label) != "external":
            return False
    return True


def unverified_claims(
    store: KnowledgeStore, rev: int | None = None, *, limit: int = 20
) -> list[dict[str, Any]]:
    """Externally-verifiable claims with zero evidence for their current
    group hash, oldest nodes first, capped at ``limit`` per sweep."""

    at = store.current_rev() if rev is None else rev
    seen = {
        (row["node_id"], row["field_path"], row["value_hash"])
        for row in store.conn.execute(
            "SELECT node_id, field_path, value_hash FROM evidence"
            " WHERE verdict IN ('confirmed', 'refuted', 'unverifiable')"
        )
    }  # refuted settles THIS hash too — fixing the value changes the hash and re-qualifies it
    subject_of: dict[str, str] = {}
    for subject in store.subjects(at):
        stack = [subject.local_id]
        while stack:
            parent = stack.pop()
            for membership in store.children(parent, at):
                if membership.child_id not in subject_of:
                    subject_of[membership.child_id] = subject.payload.get("surface", "")
                    stack.append(membership.child_id)
    claims: list[dict[str, Any]] = []
    placements = _placements(store, at)
    for kind in _VERIFIABLE_KINDS:
        for node in store.nodes_of_kind(kind, at):
            if node.maturity == "tentative":
                continue  # tentative earns its way in via shadow evidence, not quota
            if not _externally_verifiable(node, placements):
                continue
            value_hash = payload_group_hash(kind, node.payload, "core")
            if (node.local_id, "payload:core", value_hash) in seen:
                continue
            claims.append(
                {
                    "claim_id": f"c{len(claims) + 1}",
                    "node_id": node.local_id,
                    "kind": kind,
                    "field_path": "payload:core",
                    "value_hash": value_hash,
                    "line": format_line(node, aliases=node_aliases(store, node, at)),
                    "subject": subject_of.get(node.local_id, ""),
                }
            )
            if len(claims) >= limit:
                return claims
    return claims


def render_verify_prompt(claims: list[Mapping[str, Any]]) -> str:
    rows = [
        f"- {claim['claim_id']}（条目「{claim['subject']}」的 {claim['kind']} 行）：{claim['line']}"
        for claim in claims
    ]
    return load_prompt_template(
        "kb_verify_v1.md",
        strict=True,
        judgment=load_prompt_template("fragment_kb_judgment_v1.md", strict=True),
        claims_text="\n".join(rows),
    )


def parse_verify_results(text: str, claims: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The session's ``<verify_results>`` rows joined back onto the claim
    batch; rows referencing unknown ids or confirming without a URL are
    dropped (the session's say-so is not evidence)."""

    blocks = find_tag_blocks(text, "verify_results")
    if len(blocks) != 1:
        raise ValueError(f"output must contain exactly one <verify_results> block, found {len(blocks)}")
    data = json.loads(blocks[0].strip())
    if not isinstance(data, list):
        raise ValueError("verify_results must be a JSON array")
    by_id = {claim["claim_id"]: claim for claim in claims}
    rows: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        claim = by_id.get(str(entry.get("claim_id") or ""))
        verdict = str(entry.get("verdict") or "")
        url = str(entry.get("url") or "")
        if claim is None or verdict not in ("confirmed", "refuted", "unverifiable"):
            continue
        if verdict in ("confirmed", "refuted") and not _URL_RE.fullmatch(url):
            continue  # a corroboration OR refutation without a checkable URL is an opinion
        rows.append(
            {
                **claim,
                "verdict": verdict,
                "url": url if verdict in ("confirmed", "refuted") else "",
                "note": str(entry.get("note") or "")[:300],
            }
        )
    return rows


def book_verify_results(store: KnowledgeStore, rows: list[Mapping[str, Any]], *, task_id: str) -> int:
    booked = 0
    for row in rows:
        if record_evidence(
            store,
            node_id=row["node_id"],
            field_path=row["field_path"],
            value_hash=row["value_hash"],
            verdict=row["verdict"],
            evidence_kind="external",
            source_ref=row["url"] or None,
            task_id=task_id,
            algo_version=VERIFY_ALGO_VERSION,
        ):
            booked += 1
    return booked


def run_verify_session(claims: list[Mapping[str, Any]], *, client) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    """One native-search verification call over the batch."""

    from ..routing.config import LLMRole

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": render_verify_prompt(list(claims))}],
        max_tokens=VERIFY_MAX_TOKENS,
        retrieval="native",
        task_group="research",
    )
    return parse_verify_results(result.content, list(claims))
