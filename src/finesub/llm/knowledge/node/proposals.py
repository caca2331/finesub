"""Model-facing proposal vocabulary -> engine envelope -> apply (REST front end).

The knowledge-update model still writes *lines* (that is how entries render),
but addresses entities by the short handles shown in ``<kb_entries>``. This
module translates those line-form proposals into engine ops (plan §6.5),
runs the shared apply engine, and returns the report shape the chunk
ledger consumes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..base import _line_dedup_token, _match_normalize
from .apply import ApplyResult, apply_envelope
from .envelope import Binding, Envelope
from .importer import alias_delta, classify_line, extract_misheard, storable_term_payload
from .model import MATCHABLE_CATEGORIES, METADATA_SECTION
from .presets import preset_for_category
from .render import HandleMap
from .repo import KnowledgeRepo

MODEL_OPS: tuple[str, ...] = (
    "append_lines",
    "update",
    "remove",
    "create_entry",
    "retire_entry",
    "rename_entry",
    "add_item",
    "remove_item",
)
COMMON_ENTRY_TYPES: tuple[str, ...] = ("游戏", "动画", "社区", "其他")

_PROPOSALS_RE = re.compile(
    r"<knowledge_proposals\b[^>]*>(?P<body>.*?)</knowledge_proposals>", re.IGNORECASE | re.DOTALL
)
_CODE_FENCE_RE = re.compile(r"```(?:jsonl|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
#: `<reasoning>` is explicitly outside the output contract, and a model that
#: NAMES the output block while reasoning about it ("候选不判 dismiss，
#: <knowledge_proposals> 输出空块") would otherwise open the match there and
#: swallow `</reasoning>` plus the real opening tag as body lines. Observed in
#: a live repair session (2026-08-29), which then reported them as invalid
#: JSON. Strip the reasoning span before looking for the block.
_REASONING_RE = re.compile(r"<reasoning\b[^>]*>.*?</reasoning>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    return _REASONING_RE.sub("", text or "")


@dataclass
class ProposalRecord:
    category: str
    entry: str
    op: str
    section: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass
class ProposalReport:
    applied: list[ProposalRecord] = field(default_factory=list)
    skipped: list[ProposalRecord] = field(default_factory=list)
    rev: int | None = None
    rolled_back: bool = False
    rollback_reason: str = ""
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    #: indexes (into the parsed proposal list) of proposals with >=1 applied
    #: engine op — the candidate ledger's propose→resolved(applied) evidence
    applied_proposals: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": [r.to_dict() for r in self.applied],
            "skipped": [r.to_dict() for r in self.skipped],
            "rev": self.rev,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "conflicts": self.conflicts,
            "applied_proposals": list(self.applied_proposals),
        }


# ---- parsing ---------------------------------------------------------------------------


def parse_model_proposals(text: str) -> list[dict[str, Any]]:
    match = _PROPOSALS_RE.search(strip_reasoning(text))
    body = match.group("body") if match else (text or "")
    fence = _CODE_FENCE_RE.search(body)
    if fence:
        body = fence.group(1)
    out: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            out.append({"op": "__invalid__", "raw": line})
            continue
        out.append(data if isinstance(data, dict) else {"op": "__invalid__", "raw": line})
    return out


# ---- translation -------------------------------------------------------------------------


@dataclass
class _Context:
    repo: KnowledgeRepo
    rev: int
    bindings: dict[str, Binding]
    #: Categories this caller may create in or address. Model-facing front
    #: ends stay on the matchable set: a category that exists in the store but
    #: has no wired prompt (today: `style`) must not be reachable by a model
    #: guessing at the enum, or an unwired feature silently grows entries.
    allow_categories: tuple[str, ...] = MATCHABLE_CATEGORIES
    ops: list[dict[str, Any]] = field(default_factory=list)
    report: ProposalReport = field(default_factory=ProposalReport)
    touched_subjects: set[str] = field(default_factory=set)
    new_handles: int = 0
    current_proposal: int = -1  # index of the proposal being translated
    created_keys: dict[tuple[str, str], str] = field(default_factory=dict)  # (category, key) -> handle
    # section -> existing dedup tokens, per subject ref (for append dedupe)
    pending_tokens: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    pending_labels: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def require_entry_level(self, category: str, key: str, op: str) -> None:
        """Guard for ops that act on the ENTRY rather than on a line.

        The category gate on `subject_ref` only sees a stated category, and a
        `@k` handle states none — it carries the node's own. That is right for
        line-level work (the handle IS the grant: the caller was shown that
        line), but it must not extend to retiring or renaming the entry the
        line lives in: a knowledge-update task handed one style entry to
        propose conventions into could otherwise retire it outright
        (review 2026-09-02, reproduced with `retire_entry @k1`).
        """

        if category and category not in self.allow_categories:
            raise _Skip(
                f"{op} is not allowed on a {category} entry here",
                category=category, entry=key,
            )

    def subject_ref(self, value: str, category: str | None) -> tuple[str | None, str, str, str]:
        """Return ``(ref_for_ops, category, key, subject_id)``; ref None if unknown.

        An explicit category outside ``allow_categories`` REFUSES the
        proposal. Dropping it instead (what the first cut did) turns "write
        into style" into "write into whatever answers to this name" — and a
        style is usually named after a streamer or group that already has a
        proper-noun entry, so the fallback would land on that entry and edit
        it silently (review 2026-09-02).
        """

        if category is not None and category not in self.allow_categories:
            raise _Skip(f"category {category!r} not allowed here", category=category, entry=value)

        if isinstance(value, str) and value.startswith("@"):
            binding = self.bindings.get(value)
            if binding is None or binding.kind != "node":
                return None, category or "", value, ""
            node = self.repo.store.node(binding.id, self.rev)
            if node is None or node.kind != "subject":
                return None, category or "", value, ""
            return value, node.payload.get("category", ""), node.payload.get("surface", ""), binding.id
        if (category or "", value) in self.created_keys:
            handle = self.created_keys[(category or "", value)]
            return handle, category or "", value, ""
        resolved = self.repo.resolve(value, self.rev, category=category)
        if resolved is None and category in MATCHABLE_CATEGORIES:
            # Within the MATCHABLE namespace keys are globally unique
            # (create_entry's duplicate check and the rename-clash check both
            # resolve across it), so a stated category is a hint, not a
            # partition. Without this fallback a proposal pair could die in the
            # gap: create_entry "already exists" (global) followed by
            # append_lines "does not exist" (scoped) — exactly how the 律にゃー
            # channel-term migration was lost.
            #
            # It must NOT extend past that namespace: a standalone category IS
            # a partition, so falling back would answer "style/目标 does not
            # exist" with the same-named proper-noun entry — silently, and on a
            # write path (review 2026-09-02, `merged_into`).
            resolved = self.repo.resolve(value, self.rev)
        if resolved is None:
            return None, category or "", value, ""
        return resolved.subject_id, resolved.category, resolved.key, resolved.subject_id

    def node_ref(self, value: str) -> tuple[str, Binding] | None:
        binding = self.bindings.get(value) if isinstance(value, str) else None
        if binding is None or binding.kind != "node":
            return None
        return value, binding

    def bind_node(self, local_id: str) -> str | None:
        """Handle for a node the model did not see rendered (item-op sync)."""

        for handle, binding in self.bindings.items():
            if binding.kind == "node" and binding.id == local_id:
                return handle
        node = self.repo.store.node(local_id, self.rev)
        if node is None:
            return None
        handle = f"@ksync{len(self.bindings) + 1}"
        self.bindings[handle] = Binding(
            handle=handle, kind="node", id=local_id, expected_valid_from_rev=node.valid_from_rev
        )
        return handle

    def bind_item(self, item: Any) -> str:
        """Handle for an item the model did not see rendered (alias-column sync)."""

        for handle, binding in self.bindings.items():
            if binding.kind == "item" and binding.id == item.item_id:
                return handle
        handle = f"@isync{len(self.bindings) + 1}"
        self.bindings[handle] = Binding(
            handle=handle, kind="item", id=item.item_id, expected_valid_from_rev=item.valid_from_rev
        )
        return handle

    def record(self, status: str, category: str, entry: str, op: str, section: str, reason: str) -> None:
        record = ProposalRecord(category, entry, op, section, status, reason)
        (self.report.applied if status == "applied" else self.report.skipped).append(record)

    def op(
        self,
        op_dict: dict[str, Any],
        *,
        category: str = "",
        entry: str = "",
        op_name: str = "",
        section: str = "",
        label: str = "",
    ) -> None:
        """Append an engine op carrying its report record in ``_meta``.

        Nothing is written to ``report.applied`` here: the final verdict per op
        is issued from the engine's ``applied_ops``/``rejected_ops`` (plan A4 —
        the old translate-time "applied" records survived engine rejection and
        produced self-contradictory reports). Ops without ``op_name`` are
        bookkeeping (updated_date, scaffold facts) and never reported."""

        op_dict["_meta"] = (
            {"record": ProposalRecord(category, entry, op_name, section, "", label),
             "proposal": self.current_proposal}
            if op_name
            else {"bookkeeping": True}
        )
        self.ops.append(op_dict)


def translate_model_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    repo: KnowledgeRepo,
    knowledge_read_rev: int,
    bindings: Sequence[Binding],
    allow_categories: Sequence[str] = MATCHABLE_CATEGORIES,
) -> tuple[list[dict[str, Any]], ProposalReport, dict[str, str], list[Binding]]:
    """Return ``(engine_ops, report_so_far, draft_handles, bindings)``.

    ``report_so_far`` carries ONLY translate-time skips (invalid JSON, unknown
    op, ``_Skip``) — nothing is marked applied here. Every reportable op
    carries its record in ``_meta``; the caller issues the applied/skipped
    verdicts from the engine result (plan A4). The returned bindings include
    any the translator added itself (alias-column sync), so the envelope must
    be built from them.

    ``allow_categories`` defaults to the MATCHABLE set, not to every category
    that exists: this function is the model-facing door (REST front end and
    the agent's `kb_propose`), and a category whose prompt is not wired yet
    must not be reachable by a model guessing at the enum. The human CLI
    passes the full set."""

    ctx = _Context(
        repo=repo,
        rev=knowledge_read_rev,
        bindings={b.handle: b for b in bindings},
        allow_categories=tuple(allow_categories),
    )
    for index, proposal in enumerate(proposals):
        ctx.current_proposal = index
        op = str(proposal.get("op", ""))
        reason = str(proposal.get("reason", "")).strip()
        try:
            if op == "append_lines":
                _append_lines(ctx, proposal, reason)
            elif op == "update":
                _update_line(ctx, proposal, reason)
            elif op == "remove":
                _remove_node(ctx, proposal, reason)
            elif op == "create_entry":
                _create_entry(ctx, proposal, reason)
            elif op == "retire_entry":
                _retire_entry(ctx, proposal, reason)
            elif op == "rename_entry":
                _rename_entry(ctx, proposal, reason)
            elif op in ("add_item", "remove_item"):
                _item_op(ctx, proposal, reason)
            elif op == "__invalid__":
                ctx.record("skipped", "", "", "invalid", "", f"not JSON: {proposal.get('raw', '')[:80]}")
            else:
                ctx.record("skipped", str(proposal.get("category", "")), str(proposal.get("entry", "")), op, "", f"unknown op {op!r}")
        except _Skip as exc:
            ctx.record("skipped", exc.category, exc.entry, op, exc.section, str(exc))
    ctx.current_proposal = -1
    today = repo.today()
    for subject_ref in sorted(ctx.touched_subjects):
        ctx.op({"op": "update", "id": subject_ref, "set": {"payload.updated_date": today}})
    draft_handles = {handle: key for (category, key), handle in ctx.created_keys.items()}
    return ctx.ops, ctx.report, draft_handles, list(ctx.bindings.values())


class _Skip(Exception):
    def __init__(self, message: str, *, category: str = "", entry: str = "", section: str = "") -> None:
        super().__init__(message)
        self.category = category
        self.entry = entry
        self.section = section


def _append_lines(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    category = str(proposal.get("category", "")) or None
    entry_ref = str(proposal.get("entry", ""))
    section = str(proposal.get("section", "")).strip()
    content = str(proposal.get("content", ""))
    ref, cat, key, subject_id = ctx.subject_ref(entry_ref, category)
    if not section:
        raise _Skip("missing section", category=cat, entry=key)
    if section == METADATA_SECTION:
        raise _Skip("metadata section is harness-owned", category=cat, entry=key, section=section)
    if ref is None:
        raise _Skip("entry does not exist; create_entry first", category=cat, entry=key, section=section)
    preset = preset_for_category(cat)
    spec = preset.section(section)
    if spec is None:
        raise _Skip(f"section {section!r} not allowed for {preset.name}", category=cat, entry=key, section=section)
    existing = ctx.pending_tokens.get((ref, section))
    if existing is None:
        existing = set()
        if subject_id:
            for membership in ctx.repo.store.children(subject_id, ctx.rev):
                if membership.section != section:
                    continue
                aux_child = ctx.repo.store.node(membership.child_id, ctx.rev)
                if aux_child is not None:
                    existing.add(_line_dedup_token(_line_text(aux_child.kind, aux_child.payload)))
        ctx.pending_tokens[(ref, section)] = existing
    # A label is unique within its section (plan §2) — this replaces the old
    # per-field dedup that never actually fired for `字段: 值` lines.
    existing_labels = ctx.pending_labels.get((ref, section))
    if existing_labels is None:
        existing_labels = set()
        if subject_id:
            for membership in ctx.repo.store.children(subject_id, ctx.rev):
                if membership.section != section:
                    continue
                aux_child = ctx.repo.store.node(membership.child_id, ctx.rev)
                label = str(aux_child.payload.get("label") or "") if aux_child else ""
                if label:
                    existing_labels.add(label)
        ctx.pending_labels[(ref, section)] = existing_labels
    rows = [row.rstrip() for row in content.splitlines() if row.strip()]
    appended = 0
    for row in rows:
        token = _line_dedup_token(row)
        if token and token in existing:
            ctx.record("skipped", cat, key, "append_lines", section, "duplicate line (already present)")
            continue
        kind, payload = classify_line(row, spec, preset=preset)
        if kind == "invalid":
            # never silently downgraded to a note: the skip reason feeds the
            # repair round, so the model gets to rewrite the line (plan A1)
            ctx.record("skipped", cat, key, "append_lines", section, str(payload.get("error", "invalid line")))
            continue
        label = str(payload.get("label") or "")
        if label and label in {str(l) for l in existing_labels}:
            ctx.record("skipped", cat, key, "append_lines", section,
                       f"标记 [{label}] 在本节已存在——改用 update")
            continue
        if label:
            existing_labels.add(label)
        aliases: list[str] = []
        if kind == "term":
            payload, aliases = storable_term_payload(payload)
        ctx.new_handles += 1
        handle = f"@new{ctx.new_handles}"
        ctx.op(
            {"op": "create", "handle": handle, "kind": kind, "parent": ref, "section": section, "payload": payload, "reason": reason},
            category=cat, entry=key, op_name="append_lines", section=section, label="1 rows",
        )
        for variant in extract_misheard(row):
            ctx.op(
                {"op": "add_item", "id": handle, "field": "misheard", "value": variant},
                category=cat, entry=key, op_name="append_lines", section=section, label=f"misheard added: {variant}",
            )
        for alias in dict.fromkeys(aliases):
            ctx.op(
                {"op": "add_item", "id": handle, "field": "aliases", "value": alias},
                category=cat, entry=key, op_name="append_lines", section=section, label=f"alias added: {alias}",
            )
        if token:
            existing.add(token)
        appended += 1
    if not rows:
        raise _Skip("empty content", category=cat, entry=key, section=section)
    if appended:
        ctx.touched_subjects.add(ref)


def _line_text(kind: str, payload: Mapping[str, Any]) -> str:
    from .render import format_line  # local import: render depends on store only
    from .model import NodeVersion

    return format_line(NodeVersion("", kind, 0, None, dict(payload)))


def _update_line(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    ref = ctx.node_ref(str(proposal.get("id", "")))
    if ref is None:
        raise _Skip("unknown node handle")
    handle, binding = ref
    node = ctx.repo.store.node(binding.id, ctx.rev)
    if node is None:
        raise _Skip("node not found at read rev")
    if node.kind == "subject":
        raise _Skip("use rename_entry / append_lines for subjects")
    parents = ctx.repo.store.parents(binding.id, ctx.rev)
    if not parents:
        raise _Skip("node has no membership")
    parent = ctx.repo.store.node(parents[0].parent_id, ctx.rev)
    category = parent.payload.get("category", "") if parent else ""
    key = parent.payload.get("surface", "") if parent else ""
    section = parents[0].section
    line = str(proposal.get("line", "")).rstrip()
    if not line.strip():
        raise _Skip("empty line", category=category, entry=key, section=section)
    preset = preset_for_category(category)
    spec = preset.section(section)
    if spec is None:
        raise _Skip(f"section {section!r} not allowed", category=category, entry=key, section=section)
    kind, payload = classify_line(line, spec, preset=preset)
    if kind == "invalid":
        raise _Skip(str(payload.get("error", "invalid line")), category=category, entry=key, section=section)
    if kind != node.kind:
        raise _Skip(f"line would change kind {node.kind} -> {kind}; use remove + append_lines", category=category, entry=key, section=section)
    alias_column: list[str] | None = None
    if kind == "term":
        payload, alias_column = storable_term_payload(payload)
    # A line rewritten without its `[标记]` has no `label` key, and a diff over
    # the NEW payload alone would silently keep the stored one — the label
    # would come back on the next render (review 2026-08-29 P2-1). Removing a
    # label is a legitimate edit, so clear it explicitly.
    changes = {f"payload.{name}": value for name, value in payload.items() if node.payload.get(name) != value}
    if "label" not in payload and node.payload.get("label"):
        changes["payload.label"] = ""
    misheard_ops: list[dict[str, Any]] = []
    known_misheard = {
        item.value
        for item in ctx.repo.store.items_of(binding.id, ctx.rev)
        if item.field == "misheard"
    }
    for variant in extract_misheard(line):
        if variant not in known_misheard:
            misheard_ops.append({"op": "add_item", "id": handle, "field": "misheard", "value": variant})
    alias_ops: list[tuple[dict[str, Any], str]] = []
    if node.kind == "term" and alias_column is not None:
        # Alias items follow the alias column, which now renders FROM items
        # (plan §11.2): diff against the current item values by *normalized*
        # identity (`alias_delta`) so an NFKC-equivalent rewrite moves
        # nothing. Misheard items are an add-only cache, not reclaimed.
        current = {
            _match_normalize(item.value): item
            for item in ctx.repo.store.items_of(binding.id, ctx.rev)
            if item.field == "aliases"
        }
        added, removed = alias_delta([item.value for item in current.values()], alias_column)
        for alias in added:
            if _match_normalize(alias) not in current:
                alias_ops.append((
                    {"op": "add_item", "id": handle, "field": "aliases", "value": alias},
                    f"alias added: {alias}",
                ))
        for alias in removed:
            item = current.get(_match_normalize(alias))
            if item is not None:
                alias_ops.append((
                    {"op": "remove_item", "item": ctx.bind_item(item), "reason": reason},
                    f"alias removed: {item.value}",
                ))
    if not changes and not misheard_ops and not alias_ops:
        raise _Skip("no change", category=category, entry=key, section=section)
    if changes:
        ctx.op(
            {"op": "update", "id": handle, "set": changes, "reason": reason},
            category=category, entry=key, op_name="update", section=section, label="line rewritten",
        )
    for misheard_op in misheard_ops:
        ctx.op(
            misheard_op,
            category=category, entry=key, op_name="update", section=section,
            label=f"misheard added: {misheard_op['value']}",
        )
    for alias_op, label in alias_ops:
        ctx.op(
            alias_op,
            category=category, entry=key, op_name="update", section=section, label=label,
        )
    parent_handle = next((h for h, b in ctx.bindings.items() if b.kind == "node" and b.id == parents[0].parent_id), parents[0].parent_id)
    ctx.touched_subjects.add(parent_handle)


def _remove_node(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    ref = ctx.node_ref(str(proposal.get("id", "")))
    if ref is None:
        raise _Skip("unknown node handle")
    handle, binding = ref
    node = ctx.repo.store.node(binding.id, ctx.rev)
    if node is None or node.kind == "subject":
        raise _Skip("not a removable line node")
    parents = ctx.repo.store.parents(binding.id, ctx.rev)
    if len({m.parent_id for m in parents}) > 1:
        # ``remove`` retires the NODE, everywhere. A node shared by several
        # subjects (share merge attaches memberships to canonical nodes)
        # must not vanish from all of them on one entry's say-so (round 12);
        # detaching a single membership is an edit-surface operation.
        raise _Skip("node is shared by multiple entries: global remove refused")
    parent = ctx.repo.store.node(parents[0].parent_id, ctx.rev) if parents else None
    ctx.op(
        {"op": "retire", "id": handle, "reason": reason},
        category=parent.payload.get("category", "") if parent else "",
        entry=parent.payload.get("surface", "") if parent else "",
        op_name="remove",
        section=parents[0].section if parents else "",
        label="line removed",
    )
    if parents:
        parent_handle = next((h for h, b in ctx.bindings.items() if b.kind == "node" and b.id == parents[0].parent_id), parents[0].parent_id)
        ctx.touched_subjects.add(parent_handle)


def _create_entry(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    category = str(proposal.get("category", ""))
    key = str(proposal.get("entry", "")).strip()
    intro = str(proposal.get("intro", "")).strip()
    entry_type = str(proposal.get("entry_type", "")).strip()
    aliases = [str(a).strip() for a in (proposal.get("aliases") or []) if str(a).strip()]
    if category not in ctx.allow_categories:
        raise _Skip(f"bad category {category!r}", category=category, entry=key)
    if not key or re.search(r'[\\/:*?"<>|\x00-\x1f]', key):
        raise _Skip("invalid entry key", category=category, entry=key)
    if not intro:
        raise _Skip("intro required", category=category, entry=key)
    if not reason:
        raise _Skip("reason required (why the knowledge does not belong to an existing entry)", category=category, entry=key)
    if category == "common" and entry_type not in COMMON_ENTRY_TYPES:
        raise _Skip(f"entry_type must be one of {COMMON_ENTRY_TYPES}", category=category, entry=key)
    if aliases and preset_for_category(category).label_by_role("aliases") is None:
        # An alias is a MATCHING affordance. A preset with no alias field has
        # nowhere to render one, so the alias would be invisible in
        # `rendered/`, invisible in the entry text — and still live in
        # `resolve()`. Two style entries could then share a hidden alias and
        # `style/<别名>` would silently answer with whichever came first
        # (review 2026-09-02). Refuse rather than store what nobody can see.
        raise _Skip(f"category {category!r} has no alias field", category=category, entry=key)
    # Namespace-scoped: matchable keys are unique across streamer+common,
    # while a standalone category answers only for itself (repo.resolve_in).
    if ctx.repo.resolve_in(key, category, ctx.rev) is not None or (category, key) in ctx.created_keys:
        raise _Skip("entry already exists (alias/script variant resolves to it)", category=category, entry=key)
    preset = preset_for_category(category)
    ctx.new_handles += 1
    handle = f"@new{ctx.new_handles}"
    ctx.created_keys[(category, key)] = handle
    payload = {
        "surface": key,
        "intro": intro,
        "category": category,
        "entry_type": entry_type if category == "common" else "",
        "updated_date": ctx.repo.today(),
        "section_order": list(preset.section_names()),
        "native_names": [],
        "reading": "",
    }
    ctx.op(
        {"op": "create", "handle": handle, "kind": "subject", "payload": payload, "reason": reason},
        category=category, entry=key, op_name="create_entry", label="scaffolded",
    )
    for alias in aliases:
        ctx.op(
            {"op": "add_item", "id": handle, "field": "aliases", "value": alias},
            category=category, entry=key, op_name="create_entry", label=f"alias added: {alias}",
        )
    # The scaffold only materializes lines that have content: sparseness is
    # ABSENCE (plan §11.2), and an empty core slot lives in the full preview,
    # never in the store. WHICH label carries identity is asked of the preset
    # (`role`), never matched against the literal 本名 (plan §3).
    identity = preset.label_by_role("identity")
    if identity is not None:
        section_name, label = identity
        ctx.op({
            "op": "create", "kind": "note", "parent": handle,
            "section": section_name, "order_key": 0,
            "payload": {"label": label.name, "text": key},
        })


def _retire_entry(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    category = str(proposal.get("category", "")) or None
    ref, cat, key, _ = ctx.subject_ref(str(proposal.get("entry", "")), category)
    if ref is None:
        raise _Skip("entry does not exist", category=cat, entry=key)
    ctx.require_entry_level(cat, key, "retire_entry")
    merged_ref = None
    if proposal.get("merged_into"):
        # same namespace as the entry being retired: content merges into a
        # sibling, and a bare name resolved without the category would look
        # only in the matchable set (review 2026-09-02)
        merged_ref, _, merged_key, _ = ctx.subject_ref(str(proposal["merged_into"]), cat)
        if merged_ref is None:
            raise _Skip(f"merged_into {merged_key!r} does not exist", category=cat, entry=key)
    if not reason:
        raise _Skip("reason required (where the content was merged)", category=cat, entry=key)
    op: dict[str, Any] = {"op": "retire", "id": ref, "reason": reason}
    if merged_ref:
        op["merged_into"] = merged_ref
    ctx.op(op, category=cat, entry=key, op_name="retire_entry", label="retired")


def _rename_entry(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    category = str(proposal.get("category", "")) or None
    ref, cat, key, _ = ctx.subject_ref(str(proposal.get("entry", "")), category)
    new_key = str(proposal.get("new_key", "")).strip()
    if ref is None:
        raise _Skip("entry does not exist", category=cat, entry=key)
    ctx.require_entry_level(cat, key, "rename_entry")
    if not new_key or new_key == key:
        raise _Skip("new_key missing or unchanged", category=cat, entry=key)
    # the same namespace rule as create_entry: a rename may collide with a
    # style of the same name, and may NOT be refused for colliding with a
    # proper-noun entry in the other namespace
    if ctx.repo.resolve_in(new_key, cat, ctx.rev) is not None:
        raise _Skip(f"new_key already exists ({new_key!r})", category=cat, entry=key)
    ctx.op(
        {"op": "update", "id": ref, "set": {"payload.surface": new_key}, "reason": reason},
        category=cat, entry=key, op_name="rename_entry", label=f"-> {new_key}",
    )
    ctx.touched_subjects.add(ref)


def _item_op(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    op = str(proposal["op"])
    if op == "add_item":
        ref = ctx.node_ref(str(proposal.get("id", "")))
        if ref is None:
            raise _Skip("unknown node handle")
        value = str(proposal.get("value", "")).strip()
        ctx.op(
            {"op": "add_item", "id": ref[0], "field": proposal.get("field"), "value": value, "reason": reason},
            op_name=op, label=str(proposal.get("field", "") or ""),
        )
    else:
        item = str(proposal.get("item", ""))
        binding = ctx.bindings.get(item)
        if binding is None or binding.kind != "item":
            raise _Skip("unknown item handle")
        ctx.op(
            {"op": "remove_item", "item": item, "reason": reason},
            op_name=op, label=str(proposal.get("field", "") or ""),
        )
    # No alias_text sync anymore: the rendered alias column comes FROM the
    # alias items (plan §11.2), so an item-level add/remove is visible the
    # moment it lands.


# ---- apply ------------------------------------------------------------------------------


def apply_model_proposals(
    text: str,
    *,
    repo: KnowledgeRepo,
    allow_categories: Sequence[str] = MATCHABLE_CATEGORIES,
    task_id: str,
    knowledge_read_rev: int,
    handles: HandleMap | None = None,
    assignment_id: str = "",
    context_epoch: int = 0,
    proposal_text_hash: str = "",
) -> ProposalReport:
    """REST front end: parse the model block, translate, apply, refresh the cache.

    ``proposal_text_hash`` (sha256 of the raw model output) is recorded on the
    revision so the chunk ledger can recover a commit that landed before the
    ledger row was written.
    """

    proposals = parse_model_proposals(text)
    ops, report, draft_handles, bindings = translate_model_proposals(
        proposals,
        repo=repo,
        knowledge_read_rev=knowledge_read_rev,
        bindings=[Binding(**b) for b in (handles.bindings() if handles else [])],
        allow_categories=allow_categories,
    )
    engine_ops = [{k: v for k, v in op.items() if k != "_meta"} for op in ops]
    if not engine_ops:
        report.rev = None
        return report
    envelope = Envelope(
        task_id=task_id,
        assignment_id=assignment_id,
        context_epoch=context_epoch,
        knowledge_read_rev=knowledge_read_rev,
        ops=engine_ops,
        handle_bindings=bindings,
        draft_bindings=sorted({op["handle"] for op in engine_ops if op.get("op") == "create" and op.get("handle")}),
    )
    result: ApplyResult = apply_envelope(
        repo.store, envelope, note=f"proposal_text:{proposal_text_hash}" if proposal_text_hash else ""
    )
    report.rev = result.rev
    report.rolled_back = result.rolled_back
    report.rollback_reason = result.rollback_reason
    report.conflicts = [c.__dict__ for c in result.conflicts]
    # The verdict per op comes from the ENGINE result, matched back onto the
    # record each op carries in ``_meta`` (plan A4): an op the engine rejected
    # can no longer show up as applied, and alias/misheard item changes get
    # their own report lines instead of vanishing behind "line rewritten".
    rejected = dict(result.rejected_ops)
    applied_indexes = set(result.applied_ops)
    applied_proposals: set[int] = set()
    for index, op in enumerate(ops):
        meta = op.get("_meta")
        rec = meta.get("record") if isinstance(meta, Mapping) else None
        if rec is None:
            continue
        if result.rolled_back:
            report.skipped.append(replace(rec, status="skipped", reason=f"rolled back: {result.rollback_reason}"))
        elif index in rejected:
            report.skipped.append(replace(rec, status="skipped", reason=rejected[index]))
        elif index in applied_indexes:
            report.applied.append(replace(rec, status="applied"))
            proposal_index = meta.get("proposal", -1)
            if isinstance(proposal_index, int) and proposal_index >= 0:
                applied_proposals.add(proposal_index)
        else:
            report.skipped.append(replace(rec, status="skipped", reason="dropped by CAS conflict"))
    report.applied_proposals = sorted(applied_proposals)
    if not result.rolled_back:
        repo.refresh_rendered()
    return report
