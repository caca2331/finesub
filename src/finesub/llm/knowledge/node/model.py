"""Row models and canonical hashing for the node store (plan §2.1)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

#: Node kinds after the v3 line grammar (kb-line-grammar plan §4): a line is
#: a term body or a note body, optionally carrying a ``label``. ``fact`` /
#: ``event`` / ``relation`` retired — what used to distinguish them is now a
#: label plus the section it lives in.
KINDS: tuple[str, ...] = ("subject", "term", "note")
#: Kinds only the frozen version-1 archive import produces (plan §8 Phase A);
#: Phase B converts them and nothing on the write path may create them.
LEGACY_KINDS: tuple[str, ...] = ("fact", "event", "relation")
ITEM_FIELDS: tuple[str, ...] = ("aliases", "misheard")
#: Categories whose entries are reached by MATCHING a name or alias found in
#: free text: `repo.resolve()` sweeps them, the index and the keyword
#: pre-injection are built from them, and `create_entry` checks new keys
#: against all of them at once (keys are globally unique across the two).
MATCHABLE_CATEGORIES: tuple[str, ...] = ("streamer", "common")
#: Categories addressed ONLY by explicit name (`--style <名字>`) and never
#: matched against text. Keeping `style` out of the matchable set is a
#: decision, not an omission (`presets/style.toml` header): matching would buy
#: it nothing and would make the global key check reject two subtitle groups
#: owning a same-named style. The cost is that name uniqueness inside such a
#: category is nobody else's job — `repo.resolve_in()` is where it happens.
STANDALONE_CATEGORIES: tuple[str, ...] = ("style",)
CATEGORIES: tuple[str, ...] = MATCHABLE_CATEGORIES + STANDALONE_CATEGORIES
LINK_RELS: tuple[str, ...] = ("see_also", "supersedes")
VISIBILITIES: tuple[str, ...] = ("local", "shareable")
MATURITIES: tuple[str, ...] = ("normal", "tentative")  # plan §11.5 distribution lifecycle
REVISION_KINDS: tuple[str, ...] = (
    "harness",
    "user",
    "import",
    "revert",
    "restore",
    "pull",
    "share",  # server-side share-queue approval (was "import" before §11.3 attribution)
)

METADATA_SECTION = "元数据"
PROFILE_SECTION = "档案"
UPDATED_DATE_LABEL = "最近更新日期"

# Deterministic ids for the migration (plan §7.1): UUIDv5 under this namespace.
MIGRATION_NAMESPACE = uuid.UUID("6f2b3e1a-5c2d-4a7e-9f7b-2d1e8c4b9a10")


def canonical_json(value: Any) -> str:
    """Stable JSON for hashing: sorted keys, no whitespace, explicit nulls."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def migration_id(*parts: str) -> str:
    """UUIDv5 over the migration namespace; same inputs -> same id across runs."""

    return str(uuid.uuid5(MIGRATION_NAMESPACE, "\x1f".join(parts)))


# ---- payload claim groups (share review 2026-08-27 rounds 8–9) ---------------
#
# Evidence attaches to ``(node, field_path, value_hash)``. Besides the per-field
# paths (``payload.zh``) and item paths (``items/<id>``), the share protocol
# hashes whole SEMANTIC GROUPS of payload fields as ``payload:<group>`` — one
# claim per review policy, so no field escapes the §6.3 gate (round 8) while a
# single URL never "confirms" fields under a different policy (round 9). The
# tables live here, not in ``share/``, because the report must recompute
# current group hashes without depending on the share package.

#: Payload keys that never change meaning; everything else is semantic.
NON_SEMANTIC_PAYLOAD_KEYS = frozenset({"sep", "section_order", "updated_date"})

#: Named prose groups per kind. Every semantic field NOT claimed by a named
#: group falls into the catch-all ``core`` group — a future payload field is
#: gated by default (fail-safe, round 8).
PAYLOAD_CLAIM_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "term": {"body": ("body",)},
    "subject": {"intro": ("intro",)},
}


def payload_group_fields(kind: str, payload: Mapping[str, Any], group: str) -> dict[str, Any]:
    named = PAYLOAD_CLAIM_GROUPS.get(kind, {})
    if group == "core":
        claimed = {field for fields in named.values() for field in fields}
        return {
            key: value
            for key, value in payload.items()
            if key not in NON_SEMANTIC_PAYLOAD_KEYS
            and key not in claimed
            and value not in ("", [], None)
        }
    return {
        key: payload.get(key)
        for key in named.get(group, ())
        if payload.get(key) not in ("", [], None)
    }


def payload_group_hash(kind: str, payload: Mapping[str, Any], group: str = "core") -> str:
    """Hash of one semantic group, canonical JSON over its non-empty fields."""

    return digest(payload_group_fields(kind, payload, group))


@dataclass(frozen=True)
class NodeVersion:
    local_id: str
    kind: str
    valid_from_rev: int
    valid_to_rev: int | None
    payload: dict[str, Any]
    canonical_id: str | None = None
    visibility: str = "local"
    accepted_rev: int | None = None
    maturity: str = "normal"  # normal | tentative (plan §11.5 distribution lifecycle)

    @property
    def is_current(self) -> bool:
        return self.valid_to_rev is None


@dataclass(frozen=True)
class ItemVersion:
    item_id: str
    local_id: str
    field: str
    valid_from_rev: int
    valid_to_rev: int | None
    value: str
    exact_enabled: bool = True
    fuzzy_enabled: bool = False
    requires_subject_context: bool = False
    min_mora: int = 3
    canonical_item_id: str | None = None
    accepted_rev: int | None = None
    maturity: str = "normal"  # normal | tentative (plan §11.5)


@dataclass(frozen=True)
class MembershipVersion:
    membership_id: str
    valid_from_rev: int
    valid_to_rev: int | None
    parent_id: str
    child_id: str
    section: str
    order_key: int
    canonical_membership_id: str | None = None
    accepted_rev: int | None = None


@dataclass(frozen=True)
class LinkVersion:
    link_id: str
    valid_from_rev: int
    valid_to_rev: int | None
    source_id: str
    rel: str
    target_id: str
    accepted_rev: int | None = None


@dataclass(frozen=True)
class Revision:
    rev: int
    created_at: str
    kind: str
    task_id: str = ""
    proposal_hash: str = ""
    base_rev: int | None = None
    note: str = ""


@dataclass
class MigrationAux:
    """Migration-only sidecar (plan §7.1 step 4): verbatim source line and
    layout so ``render_legacy`` can reproduce the original file."""

    local_id: str
    legacy_raw: str
    source_path: str
    source_line: int
    layout: dict[str, Any] = field(default_factory=dict)


def validate_payload(kind: str, payload: Mapping[str, Any], *, strict: bool = True) -> None:
    """Structural check of the discriminated union (plan §2.1). ``strict``
    additionally enforces content rules for NEW writes (proposal/edit/share
    admission); the store itself passes ``strict=False`` so legacy-imported
    rows keep loading until the second migration cleans them."""

    if kind not in KINDS and kind not in LEGACY_KINDS:
        raise ValueError(f"unknown kind: {kind}")
    if strict and kind in LEGACY_KINDS:
        raise ValueError(f"{kind} is a legacy import kind and cannot be written")
    required = {
        "subject": ("surface", "intro", "category"),
        "term": ("surface", "desc"),
        "note": ("text",),
        "fact": ("field", "value"),
        "event": ("occurred_at", "description"),
        "relation": ("target", "description"),
    }[kind]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{kind} payload missing {missing}")
    if strict and kind == "note" and not str(payload.get("text") or "").strip():
        # Sparseness is expressed by ABSENCE (plan §11.2): an empty labelled
        # line is a rendered slot, never knowledge.
        raise ValueError("note text must be non-empty (an empty slot is not a node)")
    if strict and kind == "term" and not str(payload.get("surface") or "").strip():
        raise ValueError("term surface must be non-empty")
