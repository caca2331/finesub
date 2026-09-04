"""Wire formats and the untrusted-content boundary (plan §6.1/§6.2/§6.4).

Two documents cross the wire:

- **push bundle** (client → server review queue): the current versions of a
  user-selected subject's shareable nodes with their items / memberships /
  links and desensitized claim summaries. Every reference is a bundle-local
  handle (``n1``, ``i1``…) or a canonical id (``c:<id>``) — client
  ``local_id``s never leave the machine (plan §9).
- **snapshot** (server → every client): the server corpus keyed by canonical
  ids, with monotone ``retired`` tombstones, the redirects table, and the
  integrity chain (``chain_hash = sha256(prev_chain_hash + content_digest)``).

Shared text is untrusted in both directions (§6.4): harness protocol tags are
stripped, any other tag-shaped token is defanged, control characters dropped,
lengths capped. The same ``sanitize_text`` runs on push build and pull ingest
— a malicious server is no more trusted than a malicious contributor.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any, Iterable, Mapping

from ..node.model import (
    KINDS,
    LEGACY_KINDS,
    PAYLOAD_CLAIM_GROUPS,
    canonical_json,
    digest,
    payload_group_fields,
    payload_group_hash,
    validate_payload,
)
from ..node.store import KnowledgeStore

BUNDLE_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1

#: Per-field text cap. Generous for prose, tiny next to the reply caps — a
#: pushed field longer than this is data smuggling, not knowledge.
MAX_FIELD_CHARS = 4_000

#: Hard entity-count caps per bundle (server admission). A legitimate subject
#: pack is tens of nodes; thousands is a stuffing attempt.
MAX_BUNDLE_NODES = 500
MAX_BUNDLE_ITEMS = 2_000
MAX_BUNDLE_MEMBERSHIPS = 2_000
MAX_BUNDLE_LINKS = 1_000
MAX_BUNDLE_CLAIMS = 2_000
MAX_CLAIM_SOURCE_REFS = 20

_FIELD_PATH_RE = re.compile(r"items/i\d+|payload\.[A-Za-z_][A-Za-z0-9_]*|payload:[a-z]+")
_VALUE_HASH_RE = re.compile(r"[0-9a-f]{64}")

#: Node kinds admitted on both wire directions (push bundles and pulled
#: snapshots).
#: One definition, in model.py — this module used to keep a second copy
#: that could drift (review 2026-08-29 P2-4).
NODE_KINDS = KINDS + LEGACY_KINDS

#: Harness protocol tags whose whole element is dropped from shared text.
#: Anything else that merely *looks* like a tag is defanged instead — the
#: boundary must hold even for tags invented after this list was written.
RESERVED_TAGS: tuple[str, ...] = (
    "reasoning",
    "knowledge_proposals",
    "mistake_proposals",
    "task_update_feedback",
    "next_advice",
    "keep_entries",
    "requested_entries",
    "search_queries",
    "window_notes",
    "analysis_notes",
    "kb_entries",
    "kb_index",
    "preinjected_entries",
    "carried_entries",
    "context_pack",
    "corrected_csv",
    "refined_csv",
    "raw_csv",
)

_RESERVED_BLOCK_RE = re.compile(
    r"<(?P<tag>" + "|".join(RESERVED_TAGS) + r")\b[^>]*>.*?</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_RESERVED_LONE_RE = re.compile(
    r"</?(?:" + "|".join(RESERVED_TAGS) + r")\b[^>]*>", re.IGNORECASE
)
_TAG_SHAPE_RE = re.compile(r"<(/?[A-Za-z_][\w-]*)((?:\s[^<>]*)?)>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ExchangeError(ValueError):
    pass


def sanitize_text(text: str, *, max_chars: int = MAX_FIELD_CHARS) -> str:
    """The §6.4 boundary, applied field by field in both directions."""

    cleaned = _CONTROL_RE.sub("", str(text or ""))
    cleaned = _RESERVED_BLOCK_RE.sub("", cleaned)
    cleaned = _RESERVED_LONE_RE.sub("", cleaned)
    # Defang any remaining tag-shaped token: fullwidth brackets keep the prose
    # readable while nothing downstream can parse it as a directive.
    cleaned = _TAG_SHAPE_RE.sub(lambda m: f"＜{m.group(1)}{m.group(2)}＞", cleaned)
    return cleaned[:max_chars]


def sanitize_value(value: Any) -> Any:
    """Deep-recursive text boundary: every string anywhere in a JSON value is
    cleaned, however it is nested — a mapping inside a list is exactly where a
    bypass hid (review 2026-08-27 round 6). Mapping KEYS go through the
    boundary at every depth too (rounds 12–13): admission enforces the
    field-name grammar, but a tag smuggled in a nested key would otherwise
    resurface when a renderer stringifies the mapping."""

    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            sanitize_text(str(key), max_chars=100): sanitize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item) for item in value]
    return value


def sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return sanitize_value(dict(payload))


#: Payload keys are code-defined field names, never free text. Enforcing the
#: grammar at admission (round 12) keeps the catch-all ``payload:core`` claim
#: open to future fields while closing the "instructions in a mapping KEY"
#: hole — ``sanitize_text`` only cleans values, and the review prompt embeds
#: payloads via ``json.dumps``, which does not defang ``<`` or key text.
_PAYLOAD_KEY_RE = re.compile(r"[a-z_][a-z0-9_]{0,39}")

#: Payload VALUES are text by construction — the renderers join/insert them
#: verbatim. Known fields are typed exactly (round 14: ``section_order=0`` or
#: ``surface=123`` would pass a scalar check, pull in over a self-consistent
#: snapshot, and crash every later render as persistent poison); unknown
#: future fields must still be text or a list of texts.
_LIST_FIELDS = frozenset({"section_order", "native_names", "aliases"})
_TEXT_FIELDS = frozenset({
    "surface", "intro", "category", "reading", "zh", "desc", "body",
    "field", "value", "occurred_at", "description", "target", "text",
    "sep", "updated_date",
})


def payload_shape_problems(kind: str, payload: Mapping[str, Any], *, where: str) -> list[str]:
    problems: list[str] = []
    try:
        validate_payload(kind, payload)
    except ValueError as exc:
        problems.append(f"{where}: {exc}")
    for key, value in payload.items():
        if not isinstance(key, str) or not _PAYLOAD_KEY_RE.fullmatch(key):
            problems.append(f"{where}: payload key {str(key)[:60]!r} is not a field name")
            continue
        text_list = isinstance(value, list) and all(isinstance(item, str) for item in value)
        if key in _LIST_FIELDS:
            if not text_list:
                problems.append(f"{where}: payload.{key} must be a list of strings")
            continue
        if key in _TEXT_FIELDS:
            if not isinstance(value, str):
                problems.append(f"{where}: payload.{key} must be a string")
            continue
        if key == "valid":
            if not (
                isinstance(value, Mapping)
                and all(
                    isinstance(inner, str) and _PAYLOAD_KEY_RE.fullmatch(inner)
                    and isinstance(item, str)
                    for inner, item in value.items()
                )
            ):
                problems.append(f"{where}: payload.valid must map field names to strings")
            continue
        if not (isinstance(value, str) or text_list):
            problems.append(f"{where}: payload.{key} must be text (or a list of strings)")
    return problems


def new_idempotency_key() -> str:
    return uuid.uuid4().hex


def content_digest(content: Mapping[str, Any]) -> str:
    return digest(content)


def chain_hash(prev: str, digest_value: str) -> str:
    return hashlib.sha256(f"{prev}:{digest_value}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# push bundle


def build_push_bundle(
    store: KnowledgeStore,
    subject_ids: Iterable[str],
    *,
    rev: int | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """The push half of §6.1: current shareable versions under the explicitly
    selected subjects. ``visibility`` gates node by node (§6.4 share_inherit
    decided at write time; a ``local`` relation under a shared subject stays
    home). Claims whose item is not in the bundle are dropped rather than
    leaking a local item id."""

    at = store.current_rev() if rev is None else rev
    handles: dict[str, str] = {}  # local node id -> bundle handle
    nodes: list[dict[str, Any]] = []
    counter = 0

    def include(node_id: str) -> None:
        nonlocal counter
        node = store.node(node_id, at)
        if node is None or node_id in handles:
            return
        if node.visibility != "shareable":
            return
        counter += 1
        handles[node_id] = f"n{counter}"
        nodes.append(
            {
                "handle": handles[node_id],
                "canonical_id": node.canonical_id,
                "kind": node.kind,
                "payload": sanitize_payload(node.payload),
            }
        )

    # Descend only through shareable nodes: the bundle must stay
    # ancestry-closed (a shareable fact under a local subject is NOT pushed —
    # an orphaned top-level fact on the server would be worse than a delayed
    # one; review 2026-08-27 round 6). ``include`` itself gates visibility.
    for subject_id in subject_ids:
        subject = store.node(subject_id, at)
        if subject is None or subject.visibility != "shareable":
            continue
        stack = [subject_id]
        while stack:
            current = stack.pop()
            if current in handles:
                continue
            include(current)
            if current not in handles:
                continue  # not shareable: do not descend past it
            stack.extend(m.child_id for m in store.children(current, at))
    if not nodes:
        raise ExchangeError(
            "nothing shareable under the selected subjects (is the subject itself marked?)"
        )

    def ref(node_id: str) -> str | None:
        if node_id in handles:
            return handles[node_id]
        node = store.node(node_id, at)
        if node is not None and node.canonical_id:
            return f"c:{node.canonical_id}"
        return None  # local-only target outside the bundle: not shareable

    items: list[dict[str, Any]] = []
    item_handles: dict[str, str] = {}
    for node_id, handle in handles.items():
        for item in store.items_of(node_id, at):
            item_handles[item.item_id] = f"i{len(item_handles) + 1}"
            items.append(
                {
                    "handle": item_handles[item.item_id],
                    "node": handle,
                    "field": item.field,
                    "value": sanitize_text(item.value),
                    "canonical_item_id": item.canonical_item_id,
                }
            )
    memberships: list[dict[str, Any]] = []
    for node_id, handle in handles.items():
        for membership in store.children(node_id, at):
            child = ref(membership.child_id)
            if child is None:
                continue
            memberships.append(
                {
                    "parent": handle,
                    "child": child,
                    "section": sanitize_text(membership.section, max_chars=100),
                    "order_key": membership.order_key,
                    "canonical_membership_id": membership.canonical_membership_id,
                }
            )
    links: list[dict[str, Any]] = []
    for node_id, handle in handles.items():
        for link in store.links_from(node_id, at):
            target = ref(link.target_id)
            if target is None:
                continue
            links.append({"source": handle, "rel": link.rel, "target": target})

    claims: list[dict[str, Any]] = []
    rows = store.conn.execute(
        "SELECT node_id, field_path, value_hash, verdict, evidence_kind, source_ref"
        " FROM evidence"
    ).fetchall()
    by_claim: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["node_id"] not in handles:
            continue
        if row["verdict"] not in ("confirmed", "refuted"):
            # ``unverifiable`` is a LOCAL verification bookmark (don't re-try
            # this claim), not self-reportable evidence — indexing the counter
            # dynamically used to KeyError on it (review 2026-08-27 round 10).
            continue
        field_path = row["field_path"]
        if field_path.startswith("items/"):
            item_handle = item_handles.get(field_path.split("/", 1)[1])
            if item_handle is None:
                continue  # never leak a local item id
            field_path = f"items/{item_handle}"
        cell = by_claim.setdefault(
            (handles[row["node_id"]], field_path, row["value_hash"]),
            {
                "node": handles[row["node_id"]],
                "field_path": field_path,
                "value_hash": row["value_hash"],
                "evidence_kinds": [],
                "source_refs": [],
                "confirmed_count": 0,
                "refuted_count": 0,
            },
        )
        cell[f"{row['verdict']}_count"] += 1
        if row["evidence_kind"] not in cell["evidence_kinds"]:
            cell["evidence_kinds"].append(row["evidence_kind"])
        if row["source_ref"] and row["source_ref"] not in cell["source_refs"]:
            cell["source_refs"].append(sanitize_text(row["source_ref"], max_chars=500))
    event_rows = store.conn.execute(
        "SELECT node_id, kind, opportunity, COUNT(*) AS n FROM events"
        " WHERE kind IN ('exposed', 'landed') GROUP BY node_id, kind, opportunity"
    ).fetchall()
    exposure_counts: dict[str, dict[str, int]] = {}
    for row in event_rows:
        if row["node_id"] in handles:
            cell = exposure_counts.setdefault(handles[row["node_id"]], {"exposed": 0, "landed": 0})
            cell[row["kind"]] += row["n"]
    for cell in by_claim.values():
        counts = exposure_counts.get(cell["node"], {})
        cell["exposed_count"] = counts.get("exposed", 0)
        cell["landed_count"] = counts.get("landed", 0)
        claims.append(cell)

    return {
        "schema": BUNDLE_SCHEMA_VERSION,
        "idempotency_key": idempotency_key or new_idempotency_key(),
        "base_rev": at,
        "nodes": nodes,
        "items": items,
        "memberships": memberships,
        "links": links,
        "claim_summaries": claims,
        "handle_map": {handle: node_id for node_id, handle in handles.items()},
        "item_handle_map": {handle: item_id for item_id, handle in item_handles.items()},
    }


def bundle_content_digest(bundle: Mapping[str, Any]) -> str:
    """Semantic digest: what a retried push must match for the idempotency
    key to be honored. Excludes the key itself and ``base_rev`` — an
    unrelated local revision between attempts must not change the digest, or
    the retry after a lost response forks a second queue item (review
    2026-08-27 round 6)."""

    return digest({k: v for k, v in bundle.items() if k not in ("idempotency_key", "base_rev")})


def sanitize_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Server-side normalization before anything durable happens (§6.4: the
    boundary must hold against a contributor who bypassed the CLI, and the
    review queue feeds a maintainer-side LLM — injection has to die *before*
    enqueue, not at apply). Only known keys survive; every text field goes
    through ``sanitize_text``; claim summaries are rebuilt field by field and
    invalid ones dropped."""

    out: dict[str, Any] = {
        "schema": bundle.get("schema"),
        "idempotency_key": sanitize_text(str(bundle.get("idempotency_key") or ""), max_chars=100),
        "base_rev": int(bundle.get("base_rev") or 0),
        "nodes": [],
        "items": [],
        "memberships": [],
        "links": [],
        "claim_summaries": [],
    }
    # Structural strings (handles, refs, kinds, fields, rels) are grammar-
    # checked by ``validate_bundle``; running them through ``sanitize_text``
    # anyway is defense in depth for any path that skips admission (round 11).
    def structural(value: Any) -> str:
        return sanitize_text(str(value or ""), max_chars=100)

    for node in bundle.get("nodes") or []:
        payload = node.get("payload")
        out["nodes"].append(
            {
                "handle": structural(node.get("handle")),
                "canonical_id": structural(node.get("canonical_id")) or None,
                "kind": structural(node.get("kind")),
                "payload": sanitize_payload(payload) if isinstance(payload, Mapping) else {},
            }
        )
    for item in bundle.get("items") or []:
        out["items"].append(
            {
                "handle": structural(item.get("handle")),
                "node": structural(item.get("node")),
                "field": structural(item.get("field")),
                "value": sanitize_text(str(item.get("value") or "")),
                "canonical_item_id": structural(item.get("canonical_item_id")) or None,
            }
        )
    for membership in bundle.get("memberships") or []:
        out["memberships"].append(
            {
                "parent": structural(membership.get("parent")),
                "child": structural(membership.get("child")),
                "section": sanitize_text(str(membership.get("section") or ""), max_chars=100),
                "order_key": int(membership.get("order_key") or 0),
                "canonical_membership_id": structural(membership.get("canonical_membership_id"))
                or None,
            }
        )
    for link in bundle.get("links") or []:
        out["links"].append(
            {
                "source": structural(link.get("source")),
                "rel": structural(link.get("rel")),
                "target": structural(link.get("target")),
            }
        )
    for claim in bundle.get("claim_summaries") or []:
        field_path = str(claim.get("field_path") or "")
        value_hash = str(claim.get("value_hash") or "")
        if not (_FIELD_PATH_RE.fullmatch(field_path) and _VALUE_HASH_RE.fullmatch(value_hash)):
            continue
        out["claim_summaries"].append(
            {
                "node": structural(claim.get("node")),
                "field_path": field_path,
                "value_hash": value_hash,
                "evidence_kinds": [
                    sanitize_text(str(kind), max_chars=50)
                    for kind in (claim.get("evidence_kinds") or [])[:10]
                ],
                "source_refs": [
                    sanitize_text(str(ref), max_chars=500)
                    for ref in (claim.get("source_refs") or [])[:MAX_CLAIM_SOURCE_REFS]
                ],
                **{
                    key: max(0, int(claim.get(key) or 0))
                    for key in ("confirmed_count", "refuted_count", "exposed_count", "landed_count")
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# §6.3 slot thresholds — shared by the maintainer review CLI and the server's
# approval gate, so they cannot drift. Stdlib-only on purpose: the server is a
# thin shell and must not pull the prompt machinery in.

#: Representative field per kind, display only (claim labels).
_LABEL_FIELD_BY_KIND = {
    "term": "zh",
    "fact": "field",
    "event": "occurred_at",
    "relation": "target",
    "subject": "surface",
    "note": "text",
}

#: Which corroboration routes satisfy which slot (§6.3 table). The
#: independent-contributor route exists in the design but its server-side
#: aggregate is not built yet, so it never reads as satisfied here.
SLOT_ROUTES: dict[str, dict[str, bool]] = {
    "term": {"refined": True, "external": True},
    "档案 fact": {"refined": False, "external": True},  # 改本名/人设影响全库：只有出处算
    "关系/经历": {"refined": True, "external": False},
    "prose": {},
}

_HTTP_URL_RE = re.compile(r"https?://\S+")


def _bundle_subject_category(bundle: Mapping[str, Any]) -> dict[str, str | None]:
    """Owning subject's category per node handle, walked through the bundle's
    own membership graph. ``None`` when the ancestry leaves the bundle (an
    anchored ``c:`` parent) — callers treat unknown as the strict slot."""

    nodes = {str(node.get("handle")): node for node in bundle.get("nodes") or []}
    parent_of: dict[str, str] = {}
    for membership in bundle.get("memberships") or []:
        child = str(membership.get("child") or "")
        if child in nodes:
            parent_of.setdefault(child, str(membership.get("parent") or ""))
    out: dict[str, str | None] = {}
    for handle, node in nodes.items():
        walker, seen = handle, {handle}
        category: str | None = None
        while True:
            current = nodes.get(walker)
            if current is None:
                break  # left the bundle (c: anchor): category unknown
            if current.get("kind") == "subject":
                category = str((current.get("payload") or {}).get("category") or "") or None
                break
            walker = parent_of.get(walker, "")
            if not walker or walker in seen:
                break
            seen.add(walker)
        out[handle] = category
    return out


#: Labels whose content shifts the whole library when it changes, so only an
#: external source may corroborate them. Keyed on the LABEL now that `fact`
#: is gone (review 2026-08-29 P2-4): the shell never carried this meaning,
#: the field name did.
IDENTITY_LABELS = ("本名", "别名", "人设", "外观")


def _slot_for(kind: str, group: str, category: str | None, label: str = "") -> str:
    if group != "core":
        return "prose"  # named groups (term body, subject intro) are the manual slot
    if label and label in IDENTITY_LABELS:
        # streamer identity is the strict slot; unknown ancestry is treated as
        # strict too — never downgrade because the subject was out of sight
        return "档案 fact" if category in ("streamer", None) else "term"
    if kind == "term":
        return "term"
    if kind == "subject":
        # identity fields (surface/category/native names): a rename shifts the
        # whole library — external-only, same as 档案 (round 9)
        return "档案 fact"
    if kind in ("event", "relation"):
        return "关系/经历"  # legacy import kinds, alive only between Phase A and B
    return "prose"  # unlabelled note text


def bundle_claims(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every reviewable claim the bundle carries — one per shareable scalar
    and per item, value hashes computed from the actual bundle values. This,
    not ``claim_summaries`` (which only exists for fields that already have
    local evidence), is the reference set: brand-new content must show up as
    unreviewed instead of vanishing from the report (review 2026-08-27
    round 7), and it is also the only set verdict evidence may reference."""

    categories = _bundle_subject_category(bundle)
    claims: list[dict[str, Any]] = []
    nodes = {str(node.get("handle")): node for node in bundle.get("nodes") or []}
    for handle, node in nodes.items():
        kind = str(node.get("kind") or "")
        payload = node.get("payload") or {}
        # One claim per SEMANTIC GROUP (rounds 8–9): the catch-all ``core``
        # group hashes every semantic field not claimed by a named group, so
        # a term with no zh still yields a gated claim and no field escapes —
        # while named prose groups (term body, subject intro) keep their own
        # manual slot instead of riding the core claim's review policy.
        label = str(payload.get(_LABEL_FIELD_BY_KIND.get(kind, "surface")) or "") or kind
        for group in ("core", *PAYLOAD_CLAIM_GROUPS.get(kind, {})):
            if not payload_group_fields(kind, payload, group):
                continue
            claims.append(
                {
                    "node": handle,
                    "field_path": f"payload:{group}",
                    "value_hash": payload_group_hash(kind, payload, group),
                    "slot": _slot_for(kind, group, categories.get(handle), str(payload.get("label") or "")),
                    "label": label if len(label) <= 40 else label[:40] + "…",
                }
            )
    for item in bundle.get("items") or []:
        owner = str(item.get("node") or "")
        value = str(item.get("value") or "")
        if not value or owner not in nodes:
            continue
        claims.append(
            {
                "node": owner,
                "field_path": f"items/{item.get('handle')}",
                "value_hash": digest(value),
                "slot": "term",
                "label": value,
            }
        )
    return claims


def threshold_report(
    bundle: Mapping[str, Any],
    *,
    external_evidence: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """§6.3 pre-pass over ``bundle_claims``: which routes are satisfied.

    **The contributor's own ``claim_summaries`` never satisfy anything**:
    they are self-reported (the client aggregated them from its own local
    evidence), and a gate that trusts them hands its key back to the
    contributor. They surface as ``self_reported_*`` context for the
    reviewer, nothing more. The only satisfying input here is
    ``external_evidence`` — corroborations that went through the maintainer's
    own review flow — and only for slots whose route accepts external
    sources (关系/经历 accepts none the server can verify today, so those
    approvals are always an explicit maintainer override)."""

    # Contributor summaries are keyed by their local per-field paths; a group
    # claim (rounds 8–9) aggregates ONLY the summaries whose field belongs to
    # that group — item summaries never ride a payload claim (a misheard's
    # refined confirmation must not read as "自述有精修印证" for zh/body).
    # Reviewer context either way, never a gate input.
    by_node: dict[str, list[Mapping[str, Any]]] = {}
    for c in bundle.get("claim_summaries") or []:
        by_node.setdefault(str(c.get("node")), []).append(c)
    kinds_by_handle = {
        str(node.get("handle")): str(node.get("kind") or "") for node in bundle.get("nodes") or []
    }

    def _related(claim: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        rows = by_node.get(str(claim["node"]), [])
        path = str(claim["field_path"])
        if path.startswith("payload:"):
            kind = kinds_by_handle.get(str(claim["node"]), "")
            group_fields = set(
                payload_group_fields(
                    kind,
                    next(
                        (
                            n.get("payload") or {}
                            for n in bundle.get("nodes") or []
                            if str(n.get("handle")) == str(claim["node"])
                        ),
                        {},
                    ),
                    path.split(":", 1)[1],
                )
            )
            return [
                c
                for c in rows
                if str(c.get("field_path") or "").startswith("payload.")
                and str(c.get("field_path")).split(".", 1)[1] in group_fields
            ]
        return [
            c
            for c in rows
            if str(c.get("field_path")) == path and str(c.get("value_hash")) == claim["value_hash"]
        ]

    provided: set[tuple[str, str, str]] = set()
    for row in external_evidence or []:
        if _HTTP_URL_RE.fullmatch(str(row.get("url") or "")):
            provided.add(
                (str(row.get("node") or ""), str(row.get("field_path") or ""), str(row.get("value_hash") or ""))
            )
    report = []
    for claim in bundle_claims(bundle):
        key = (claim["node"], claim["field_path"], claim["value_hash"])
        related = _related(claim)
        kinds = [k for c in related for k in (c.get("evidence_kinds") or [])]
        confirmed = sum(int(c.get("confirmed_count") or 0) for c in related)
        sources = list(
            dict.fromkeys(
                f"{c.get('field_path')}: {ref}"
                for c in related
                for ref in (c.get("source_refs") or [])
            )
        )
        routes = SLOT_ROUTES.get(claim["slot"], {})
        external = bool(routes.get("external") and key in provided)
        report.append(
            {
                **claim,
                "external_confirmed": external,
                "satisfied": external,
                "needs_external": not external and claim["slot"] != "prose",
                # reviewer context only — self-reported, unverifiable here:
                "self_reported_refined": bool("refined_srt" in kinds and confirmed > 0),
                "self_reported_sources": sources,
            }
        )
    return report


def unsatisfied_claims(
    report: list[Mapping[str, Any]], *, merge: Mapping[str, str] | None = None
) -> list[Mapping[str, Any]]:
    """Gate input: unsatisfied non-prose claims. Scalar claims of nodes the
    verdict merges into an existing canonical node are exempt — a merge takes
    the node's items/memberships but NOT its payload, so gating content that
    will never land would only manufacture overrides. Item claims still
    count: those do attach."""

    merged_nodes = set((merge or {}).keys())
    return [
        row
        for row in report
        if row.get("needs_external")
        and not (
            row.get("node") in merged_nodes
            and str(row.get("field_path") or "").startswith("payload")
        )
    ]


def strip_local_maps(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """What actually goes on the wire: the handle→local-id maps stay home (they
    exist so the *client* can backfill canonical ids from the push status)."""

    return {k: v for k, v in bundle.items() if k not in ("handle_map", "item_handle_map")}


def validate_bundle(bundle: Mapping[str, Any]) -> list[str]:
    """Server-side admission of an untrusted bundle: shape, reference closure,
    and the same text boundary the client should already have applied."""

    if not isinstance(bundle, Mapping):
        return ["bundle must be a JSON object"]
    problems: list[str] = []
    if bundle.get("schema") != BUNDLE_SCHEMA_VERSION:
        return [f"unsupported bundle schema {bundle.get('schema')!r}"]
    if not str(bundle.get("idempotency_key") or "").strip():
        problems.append("missing idempotency_key")
    # Type-check every collection BEFORE touching elements (round 12): valid
    # JSON with the wrong shape must be a 400 at admission, not a 500 from
    # ``.get()`` on a string later on. A bad collection is reported once and
    # validated as empty from here on.
    collections: dict[str, list[Mapping[str, Any]]] = {}
    for key in ("nodes", "items", "memberships", "links", "claim_summaries"):
        # missing = empty; PRESENT with the wrong type is an error — ``or []``
        # would silently accept falsey wrong types like {} or "" (round 13)
        value = bundle.get(key, [])
        if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
            problems.append(f"{key} must be a list of objects")
            value = []
        collections[key] = value
    nodes = collections["nodes"]
    if not nodes:
        problems.append("empty bundle")
        return problems
    for key, cap in (
        ("nodes", MAX_BUNDLE_NODES),
        ("items", MAX_BUNDLE_ITEMS),
        ("memberships", MAX_BUNDLE_MEMBERSHIPS),
        ("links", MAX_BUNDLE_LINKS),
        ("claim_summaries", MAX_BUNDLE_CLAIMS),
    ):
        if len(collections[key]) > cap:
            problems.append(f"too many {key} (> {cap})")
    handles: set[str] = set()
    for node in nodes:
        handle = str(node.get("handle") or "")
        if not re.fullmatch(r"n\d+", handle) or handle in handles:
            problems.append(f"bad node handle {handle!r}")
            continue
        handles.add(handle)
        kind = node.get("kind")
        if kind not in NODE_KINDS:
            problems.append(f"{handle}: unknown kind {kind!r}")
            continue
        payload = node.get("payload")
        if not isinstance(payload, Mapping):
            problems.append(f"{handle}: payload is not an object")
            continue
        problems.extend(payload_shape_problems(kind, payload, where=handle))

    def check_ref(value: Any, where: str) -> None:
        text = str(value or "")
        if text in handles or text.startswith("c:"):
            return
        problems.append(f"{where}: unresolvable reference {text!r} (local ids never travel)")

    # Structural strings are never prose: ``sanitize_text`` deliberately keeps
    # newlines (values need them), so a free-form handle or rel could smuggle
    # fake lines into the maintainer's review prompt and a duplicate item
    # handle would make the returned assignment ambiguous (round 11). Enforce
    # the exact wire grammar here; sanitization stays as depth, not as gate.
    item_handles: set[str] = set()
    for item in collections["items"]:
        handle = str(item.get("handle") or "")
        if not re.fullmatch(r"i\d+", handle) or handle in item_handles:
            problems.append(f"bad item handle {handle!r}")
        else:
            item_handles.add(handle)
        check_ref(item.get("node"), "item")
        if item.get("field") not in ("aliases", "misheard"):
            problems.append(f"item: unknown field {item.get('field')!r}")
    for membership in collections["memberships"]:
        check_ref(membership.get("parent"), "membership.parent")
        check_ref(membership.get("child"), "membership.child")
    for link in collections["links"]:
        check_ref(link.get("source"), "link.source")
        check_ref(link.get("target"), "link.target")
        if link.get("rel") not in ("see_also", "supersedes"):
            problems.append(f"link: unknown rel {link.get('rel')!r}")
    for claim in collections["claim_summaries"]:
        check_ref(claim.get("node"), "claim.node")
        if not _FIELD_PATH_RE.fullmatch(str(claim.get("field_path") or "")):
            problems.append(f"claim: bad field_path {claim.get('field_path')!r}")
        if not _VALUE_HASH_RE.fullmatch(str(claim.get("value_hash") or "")):
            problems.append("claim: value_hash is not a sha256 hex digest")
        for key in ("confirmed_count", "refuted_count", "exposed_count", "landed_count"):
            value = claim.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(f"claim: {key} is not a non-negative integer")
    for key in ("handle_map", "item_handle_map"):
        if key in bundle:
            problems.append(f"{key} must not travel")

    # Ancestry closure (review 2026-08-27 round 6): every NEW non-subject node
    # must be reachable through the bundle's memberships from a bundle subject
    # or an existing server anchor (a ``c:`` parent) — otherwise an approval
    # would create orphaned top-level facts. Nodes that carry a canonical_id
    # are already anchored server-side.
    kind_of = {str(node.get("handle")): node.get("kind") for node in nodes}
    anchored = {
        str(node.get("handle"))
        for node in nodes
        if node.get("kind") == "subject" or node.get("canonical_id")
    }
    children_of: dict[str, list[str]] = {}
    for membership in collections["memberships"]:
        parent = str(membership.get("parent") or "")
        child = str(membership.get("child") or "")
        if child in kind_of and (parent in kind_of or parent.startswith("c:")):
            children_of.setdefault(parent, []).append(child)
    reachable = set(anchored)
    frontier = list(anchored) + [p for p in children_of if p.startswith("c:")]
    while frontier:
        current = frontier.pop()
        for child in children_of.get(current, []):
            if child not in reachable:
                reachable.add(child)
                frontier.append(child)
    for handle, kind in kind_of.items():
        if handle not in reachable:
            problems.append(
                f"{handle}: {kind} has no membership path from a subject or existing anchor"
            )
    return problems


# ---------------------------------------------------------------------------
# snapshot


def snapshot_content(store: KnowledgeStore, rev: int | None = None) -> dict[str, Any]:
    """The server corpus as canonical-id rows. On the server the store's
    ``local_id`` *is* the canonical id (bundles are re-keyed on approval), so
    this is a straight projection plus monotone tombstones."""

    at = store.current_rev() if rev is None else rev
    nodes: list[dict[str, Any]] = []
    for row in store.conn.execute("SELECT local_id, kind FROM nodes ORDER BY local_id").fetchall():
        node = store.node(row["local_id"], at)
        nodes.append(
            {
                "canonical_id": row["local_id"],
                "kind": row["kind"],
                "payload": sanitize_payload(node.payload) if node else {},
                "retired": node is None,
                # in the content digest by construction (whole-row hash): a
                # maturity flip cannot be replayed away (plan §11.5)
                "maturity": node.maturity if node else "normal",
            }
        )
    items: list[dict[str, Any]] = []
    live_items = {item.item_id: item for item in store.all_items(at)}
    for row in store.conn.execute(
        "SELECT item_id, local_id, field FROM items ORDER BY item_id"
    ).fetchall():
        item = live_items.get(row["item_id"])
        items.append(
            {
                "canonical_item_id": row["item_id"],
                "node": row["local_id"],
                "field": row["field"],
                "value": sanitize_text(item.value) if item else "",
                "retired": item is None,
                "maturity": item.maturity if item else "normal",
            }
        )
    memberships: list[dict[str, Any]] = []
    for row in store.conn.execute(
        "SELECT membership_id FROM memberships ORDER BY membership_id"
    ).fetchall():
        live = store.conn.execute(
            "SELECT * FROM membership_versions WHERE membership_id=?"
            " AND valid_from_rev <= ? AND (valid_to_rev IS NULL OR valid_to_rev > ?)",
            (row["membership_id"], at, at),
        ).fetchone()
        record = live or store.conn.execute(
            "SELECT * FROM membership_versions WHERE membership_id=?"
            " ORDER BY valid_from_rev DESC LIMIT 1",
            (row["membership_id"],),
        ).fetchone()
        memberships.append(
            {
                "canonical_membership_id": row["membership_id"],
                "parent": record["parent_id"],
                "child": record["child_id"],
                "section": record["section"],
                "order_key": record["order_key"],
                "retired": live is None,
            }
        )
    links: list[dict[str, Any]] = []
    for row in store.conn.execute("SELECT link_id FROM links ORDER BY link_id").fetchall():
        live = store.conn.execute(
            "SELECT * FROM link_versions WHERE link_id=?"
            " AND valid_from_rev <= ? AND (valid_to_rev IS NULL OR valid_to_rev > ?)",
            (row["link_id"], at, at),
        ).fetchone()
        record = live or store.conn.execute(
            "SELECT * FROM link_versions WHERE link_id=? ORDER BY valid_from_rev DESC LIMIT 1",
            (row["link_id"],),
        ).fetchone()
        links.append(
            {
                "source": record["source_id"],
                "rel": record["rel"],
                "target": record["target_id"],
                "retired": live is None,
            }
        )
    redirects = {
        row["old_canonical_id"]: row["new_canonical_id"]
        for row in store.conn.execute("SELECT * FROM redirects").fetchall()
    }
    return {
        "schema": SNAPSHOT_SCHEMA_VERSION,
        "nodes": nodes,
        "items": items,
        "memberships": memberships,
        "links": links,
        "redirects": redirects,
    }


def verify_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Integrity of one snapshot message: the whole advertised history must be
    one connected chain — every entry carries its ``content_digest`` and every
    ``chain_hash`` is verified as ``H(prev_chain_hash, content_digest)`` with
    strictly increasing revisions — and the head must commit to exactly the
    content received. "The anchor appears somewhere in the list" is not
    enough (review 2026-08-27): a disconnected entry between the anchor and
    the head would otherwise smuggle a rewritten history past the client.

    The anti-rollback half — "does this verified chain still contain the
    entry I trusted last time" — is the client's check
    (``sync.check_anchor``); with the chain verified here, anchor membership
    implies the head extends the anchored prefix.
    """

    content = snapshot.get("content")
    if not isinstance(content, Mapping) or content.get("schema") != SNAPSHOT_SCHEMA_VERSION:
        raise ExchangeError("unsupported snapshot schema")
    history = snapshot.get("history")
    if not isinstance(history, list) or not history:
        raise ExchangeError("snapshot carries no chain history")
    prev = ""
    last_rev: int | None = None
    for entry in history:
        if not isinstance(entry, Mapping):
            raise ExchangeError("snapshot history entries must be objects")
        rev = entry.get("rev")
        if not isinstance(rev, int) or isinstance(rev, bool) or (
            last_rev is not None and rev <= last_rev
        ):
            raise ExchangeError("snapshot history revisions are not strictly increasing")
        expected_link = chain_hash(prev, str(entry.get("content_digest") or ""))
        if entry.get("chain_hash") != expected_link:
            raise ExchangeError(f"snapshot chain broken at rev {rev}")
        prev = expected_link
        last_rev = rev
    head = history[-1]
    if snapshot.get("server_rev") != head["rev"]:
        raise ExchangeError("snapshot server_rev does not match the chain head")
    expected = content_digest(content)
    if snapshot.get("content_digest") != expected or head.get("content_digest") != expected:
        raise ExchangeError("snapshot content digest mismatch")
    if snapshot.get("chain_hash") != head.get("chain_hash"):
        raise ExchangeError("snapshot chain hash mismatch")


#: Server-issued ids (canonical node/item/membership ids). Our server mints
#: uuid hex; the grammar stays a little wider without ever admitting tags,
#: whitespace or newlines.
_CANONICAL_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,100}")

def snapshot_content_problems(content: Mapping[str, Any]) -> list[str]:
    """Admission for PULLED content (round 13). The verified chain only proves
    the server *said* this — a malicious server is no more trusted than a
    malicious contributor (module docstring), so the same shape/key/grammar
    boundary that guards push admission runs on every snapshot before
    anything is written locally. Retired rows are tombstones and carry an
    empty payload, so only live payloads face the strict shape check."""

    problems: list[str] = []
    rows: dict[str, list[Mapping[str, Any]]] = {}
    for key in ("nodes", "items", "memberships", "links"):
        value = content.get(key, [])
        if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
            problems.append(f"{key} must be a list of objects")
            value = []
        rows[key] = value
    redirects = content.get("redirects", {})
    if not isinstance(redirects, Mapping) or not all(
        isinstance(old, str)
        and isinstance(new, str)
        and _CANONICAL_ID_RE.fullmatch(old)
        and _CANONICAL_ID_RE.fullmatch(new)
        for old, new in redirects.items()
    ):
        problems.append("redirects must map canonical ids to canonical ids")

    def check_id(value: Any, where: str) -> None:
        if not isinstance(value, str) or not _CANONICAL_ID_RE.fullmatch(value):
            problems.append(f"{where}: bad id {str(value)[:60]!r}")

    def check_unique(value: Any, seen: set, where: str) -> None:
        if isinstance(value, str):
            if value in seen:
                problems.append(f"{where}: duplicate id")
            seen.add(value)

    node_ids: set[str] = set()
    item_ids: set[str] = set()
    membership_ids: set[str] = set()
    for node in rows["nodes"]:
        where = f"node {str(node.get('canonical_id'))[:40]!r}"
        check_id(node.get("canonical_id"), where)
        check_unique(node.get("canonical_id"), node_ids, where)
        kind = node.get("kind")
        payload = node.get("payload")
        if kind not in NODE_KINDS:
            problems.append(f"{where}: unknown kind {str(kind)[:40]!r}")
        elif not isinstance(payload, Mapping):
            problems.append(f"{where}: payload is not an object")
        elif not node.get("retired"):
            problems.extend(payload_shape_problems(str(kind), payload, where=where))
        if node.get("maturity", "normal") not in ("normal", "tentative"):
            problems.append(f"{where}: bad maturity")
    for item in rows["items"]:
        where = f"item {str(item.get('canonical_item_id'))[:40]!r}"
        check_id(item.get("canonical_item_id"), where)
        check_unique(item.get("canonical_item_id"), item_ids, where)
        check_id(item.get("node"), where)
        if item.get("field") not in ("aliases", "misheard"):
            problems.append(f"{where}: unknown field {str(item.get('field'))[:40]!r}")
        if not isinstance(item.get("value", ""), str):
            problems.append(f"{where}: value is not text")
        if item.get("maturity", "normal") not in ("normal", "tentative"):
            problems.append(f"{where}: bad maturity")
    live_edges: dict[str, list[str]] = {}
    for membership in rows["memberships"]:
        where = f"membership {str(membership.get('canonical_membership_id'))[:40]!r}"
        check_id(membership.get("canonical_membership_id"), where)
        check_unique(membership.get("canonical_membership_id"), membership_ids, where)
        parent, child = membership.get("parent"), membership.get("child")
        check_id(parent, where)
        check_id(child, where)
        if not isinstance(membership.get("section", ""), str):
            problems.append(f"{where}: section is not text")
        order_key = membership.get("order_key", 0)
        if not isinstance(order_key, int) or isinstance(order_key, bool):
            problems.append(f"{where}: order_key is not an integer")
        if isinstance(parent, str) and isinstance(child, str) and not membership.get("retired"):
            if parent == child:
                problems.append(f"{where}: self-membership")
            else:
                live_edges.setdefault(parent, []).append(child)
    # The server never lets a membership cycle in (its merge check runs the
    # same DFS); a pulled snapshot must restore that invariant instead of
    # trusting the signature (round 14) — the apply loop would happily write
    # the cycle into the local store.
    state: dict[str, int] = {}

    def cyclic(vertex: str) -> bool:
        state[vertex] = 1
        for peer in live_edges.get(vertex, ()):
            mark = state.get(peer)
            if mark == 1 or (mark is None and cyclic(peer)):
                return True
        state[vertex] = 2
        return False

    if any(state.get(v) is None and cyclic(v) for v in list(live_edges)):
        problems.append("memberships contain a cycle")
    for link in rows["links"]:
        check_id(link.get("source"), "link")
        check_id(link.get("target"), "link")
        if link.get("rel") not in ("see_also", "supersedes"):
            problems.append(f"link: unknown rel {str(link.get('rel'))[:40]!r}")
    if isinstance(redirects, Mapping):
        # ``follow()`` on the pull side breaks cycles defensively; admission
        # refuses them outright — a redirect loop is never legitimate data.
        for start in redirects:
            seen: set[Any] = set()
            current: Any = start
            while isinstance(current, str) and current in redirects:
                if current in seen:
                    problems.append(f"redirects contain a cycle through {str(start)[:40]!r}")
                    break
                seen.add(current)
                current = redirects[current]
    return problems


def canonical_envelope_json(value: Any) -> str:
    return canonical_json(value)
