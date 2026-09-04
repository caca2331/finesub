"""Apply engine shared by both front ends (plan §2.5, §6.5):

    parse ops -> build overlay -> validate overlay -> CAS once per entity
              -> drop stale intents -> re-validate -> write versions (or roll back)

The overlay merges every op of one batch per entity, so a batch that
updates ``@k12`` twice yields one new version row and one CAS check. New
nodes get draft handles (``@new1``) usable by later ops in the same batch.
"""

from __future__ import annotations

from collections import deque
import copy
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .envelope import Binding, Envelope, EnvelopeError
from .model import ITEM_FIELDS, KINDS, LINK_RELS, VISIBILITIES, validate_payload
from .presets import preset_for_category
from .store import KnowledgeStore, NotFoundError, StaleWriteError, Transaction
from ..base import _match_normalize


class ApplyError(RuntimeError):
    pass


@dataclass
class Conflict:
    entity: str  # node | item | membership | link
    id: str
    reason: str
    dropped_ops: list[int] = field(default_factory=list)


@dataclass
class ApplyResult:
    rev: int | None
    applied_ops: list[int] = field(default_factory=list)
    rejected_ops: list[tuple[int, str]] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    created: dict[str, str] = field(default_factory=dict)  # draft handle -> local_id
    rolled_back: bool = False
    rollback_reason: str = ""


# ---- overlay ----------------------------------------------------------------------


@dataclass
class ONode:
    local_id: str
    kind: str
    payload: dict[str, Any]
    visibility: str
    expected_from_rev: int | None  # None = new in this batch
    is_new: bool
    ops: list[int] = field(default_factory=list)
    retired: bool = False
    merged_into: str | None = None


@dataclass
class OItem:
    item_id: str
    local_id: str
    field: str
    value: str
    expected_from_rev: int | None
    is_new: bool
    ops: list[int] = field(default_factory=list)
    removed: bool = False
    fuzzy_enabled: bool = False


@dataclass
class OMembership:
    membership_id: str
    parent_id: str
    child_id: str
    section: str
    order_key: int
    expected_from_rev: int | None
    is_new: bool
    ops: list[int] = field(default_factory=list)
    removed: bool = False


@dataclass
class OLink:
    link_id: str
    source_id: str
    rel: str
    target_id: str
    ops: list[int] = field(default_factory=list)


@dataclass
class Overlay:
    read_rev: int
    nodes: dict[str, ONode] = field(default_factory=dict)
    items: dict[str, OItem] = field(default_factory=dict)
    memberships: dict[str, OMembership] = field(default_factory=dict)
    links: dict[str, OLink] = field(default_factory=dict)
    created: dict[str, str] = field(default_factory=dict)
    rejected: list[tuple[int, str]] = field(default_factory=list)


class _Resolver:
    """Handle -> (id, expected_valid_from_rev); draft handles -> new ids."""

    def __init__(self, envelope: Envelope, overlay: Overlay) -> None:
        self.bindings: dict[str, Binding] = {b.handle: b for b in envelope.handle_bindings}
        self.overlay = overlay

    def node(self, ref: str) -> tuple[str, int | None]:
        return self._resolve(ref, "node")

    def item(self, ref: str) -> tuple[str, int | None]:
        return self._resolve(ref, "item")

    def membership(self, ref: str) -> tuple[str, int | None]:
        return self._resolve(ref, "membership")

    def _resolve(self, ref: str, kind: str) -> tuple[str, int | None]:
        if not ref.startswith("@"):
            return ref, None  # raw id (parent harness / tests); no CAS expectation
        if ref in self.overlay.created:
            return self.overlay.created[ref], None
        binding = self.bindings.get(ref)
        if binding is None:
            raise EnvelopeError(f"unbound handle {ref}")
        if binding.kind != kind:
            raise EnvelopeError(f"handle {ref} is a {binding.kind}, expected {kind}")
        return binding.id, binding.expected_valid_from_rev


def _load_node(store: KnowledgeStore, overlay: Overlay, local_id: str, expected: int | None) -> ONode:
    node = overlay.nodes.get(local_id)
    if node is not None:
        return node
    stored = store.node(local_id, overlay.read_rev)
    if stored is None:
        raise NotFoundError(local_id)
    node = ONode(
        local_id=local_id,
        kind=stored.kind,
        payload=dict(stored.payload),
        visibility=stored.visibility,
        expected_from_rev=expected if expected is not None else stored.valid_from_rev,
        is_new=False,
    )
    overlay.nodes[local_id] = node
    return node


def _load_memberships_of(store: KnowledgeStore, overlay: Overlay, child_id: str) -> list[OMembership]:
    out: list[OMembership] = []
    for stored in store.parents(child_id, overlay.read_rev):
        existing = overlay.memberships.get(stored.membership_id)
        if existing is None:
            existing = OMembership(
                membership_id=stored.membership_id,
                parent_id=stored.parent_id,
                child_id=stored.child_id,
                section=stored.section,
                order_key=stored.order_key,
                expected_from_rev=stored.valid_from_rev,
                is_new=False,
            )
            overlay.memberships[stored.membership_id] = existing
        out.append(existing)
    return out


def _new_id() -> str:
    return str(uuid.uuid4())


def build_overlay(store: KnowledgeStore, envelope: Envelope) -> Overlay:
    """Fold every op into per-entity intents. Malformed ops are recorded in
    ``overlay.rejected`` and skipped; the rest of the batch proceeds.

    Each op folds atomically: ``_fold`` may raise after it has already touched
    the overlay (a ``create`` that inserts the node before discovering it has
    no parent), so the overlay is snapshotted per op and restored in place on
    rejection — a rejected op must leave nothing behind to be committed."""

    overlay = Overlay(read_rev=envelope.knowledge_read_rev)
    resolver = _Resolver(envelope, overlay)
    for index, op in enumerate(envelope.ops):
        backup = copy.deepcopy(overlay)
        try:
            _fold(store, overlay, resolver, index, op)
        except (EnvelopeError, NotFoundError, ValueError, KeyError) as exc:
            # in-place restore: the resolver holds a reference to this overlay
            for name in ("nodes", "items", "memberships", "links", "created"):
                table = getattr(overlay, name)
                table.clear()
                table.update(getattr(backup, name))
            overlay.rejected[:] = backup.rejected
            overlay.rejected.append((index, f"{op.get('op')}: {exc}"))
    return overlay


def effective_visibility(
    store: KnowledgeStore,
    overlay: Overlay,
    parent_id: str | None,
    *,
    section: str,
    label: str | None,
) -> str:
    """``share_inherit`` resolved from the SECTION the line lives in plus its
    label (plan §3), not from the node's kind — relations became term bodies,
    so a kind-keyed rule would hand a family member's name to the share
    bundle. Unregistered labels are fail-closed via ``unknown_label_share``.

    Called at creation AND after any update/move: a line relabelled to an
    unregistered label, or moved into a local section, must lose shareable
    (review 2026-08-29 P1-3). The reverse — local becoming shareable — is
    never automatic: downgrade is a safety action, upgrade is authorization.
    """

    if parent_id is None:
        return "local"
    node = overlay.nodes.get(parent_id)
    current = (node.kind, node.visibility, None) if node is not None else None
    if current is None:
        stored = store.node(parent_id)
        if stored is None:
            return "local"
        current = (stored.kind, stored.visibility, stored)
    if current[1] != "shareable":
        return "local"
    # the preset lives on the owning subject: walk up (bounded tree depth)
    walker_id, seen = parent_id, {parent_id}
    walker_kind = current[0]
    while walker_kind != "subject":
        parents = store.parents(walker_id)
        if not parents or parents[0].parent_id in seen:
            return "local"
        walker_id = parents[0].parent_id
        seen.add(walker_id)
        above = overlay.nodes.get(walker_id)
        if above is not None:
            walker_kind = above.kind
        else:
            stored = store.node(walker_id)
            if stored is None:
                return "local"
            walker_kind = stored.kind
    subject = overlay.nodes.get(walker_id)
    payload = subject.payload if subject is not None else (store.node(walker_id).payload)
    try:
        preset = preset_for_category(str(payload.get("category") or ""))
    except ValueError:
        return "local"
    return "shareable" if preset.share_for(section, label) == "inherit" else "local"


def _check_label_uniqueness(store, overlay: Overlay, live_node, problems: list) -> None:  # type: ignore[no-untyped-def]
    """One line per label per section — an INVARIANT, so it is checked here
    rather than at any single write surface.

    Two ways to break it that a one-sided check misses: an ``update`` that
    relabels a line onto a taken name touches only the NODE (review 2026-08-29
    P1-3), and a ``move_membership`` of a local line touches only the
    MEMBERSHIP (review 2026-08-29 P1-1). So collect every ``(parent, section)``
    the batch touches from both sides — all placements of a shared node, not
    just the first — and then check each of those sections whole.

    Siblings are read at the store's CURRENT rev, not at ``overlay.read_rev``
    (review 2026-08-29 P1): two batches that each relabel a DIFFERENT node onto
    the same name never collide on the per-entity CAS, and at the pinned rev
    neither can see the other's line — both committed and the section ended up
    with two ``[丙]``. The re-validation inside ``apply_envelope`` runs under
    ``BEGIN IMMEDIATE``, so "current" there is the newest committed state and
    no further writer can slip in before we finish: the loser rolls back."""

    at = store.current_rev()
    touched: set[tuple[str, str]] = set()
    for membership in overlay.memberships.values():
        if not membership.removed:
            touched.add((membership.parent_id, membership.section))
    for node in overlay.nodes.values():
        if node.retired or not str(node.payload.get("label") or "").strip():
            continue
        touched |= _placements_of(store, overlay, node.local_id, at)

    for parent_id, section in sorted(touched):
        first_by_label: dict[str, str] = {}
        for member_id in sorted(_section_members(store, overlay, parent_id, section, at)):
            member = overlay.nodes.get(member_id)
            if member is not None and member.retired:
                continue
            if member is None:
                stored = store.node(member_id, at)
                if stored is None:
                    continue
                member = ONode(member_id, stored.kind, dict(stored.payload),
                               stored.visibility, stored.valid_from_rev, False)
            label = _match_normalize(str(member.payload.get("label") or ""))
            if not label:
                continue
            if first_by_label.setdefault(label, member_id) != member_id:
                problems.append((
                    "node", member_id,
                    f"标记 [{member.payload.get('label')}] 在小节 {section!r} 内重复"
                    "——同名标记应改用 update",
                ))


def _check_membership_uniqueness(store, overlay: Overlay, problems: list) -> None:  # type: ignore[no-untyped-def]
    """One edge per ``(parent, child)`` — a node has ONE home per subject.

    ``_section_members`` answers with a SET of child ids, so a second edge for
    a child already in that section simply collapsed there and slipped through;
    the entry then rendered the line twice (review 2026-08-29 P2). Sharing one
    node across DIFFERENT subjects stays supported — that is a different parent.
    """

    at = store.current_rev()
    for membership in overlay.memberships.values():
        if membership.removed:
            continue
        pair = (membership.parent_id, membership.child_id)
        others = [
            m.membership_id
            for m in overlay.memberships.values()
            if not m.removed
            and m.membership_id != membership.membership_id
            and (m.parent_id, m.child_id) == pair
        ]
        others += [
            stored.membership_id
            for stored in store.parents(membership.child_id, at)
            if stored.parent_id == membership.parent_id
            and stored.membership_id != membership.membership_id
            and stored.membership_id not in overlay.memberships
        ]
        if others:
            problems.append((
                "membership", membership.membership_id,
                "该节点在同一条目下已有归属边——重复归属会让这行渲染两次，"
                "要换小节请用 move_membership",
            ))


def _placements_of(store, overlay: Overlay, child_id: str, at: int) -> set[tuple[str, str]]:  # type: ignore[no-untyped-def]
    """Every ``(parent, section)`` a node effectively sits in. Read-only: unlike
    ``_load_memberships_of`` this must not pull stored rows into the overlay."""

    out = {
        (m.parent_id, m.section)
        for m in overlay.memberships.values()
        if m.child_id == child_id and not m.removed
    }
    out |= {
        (stored.parent_id, stored.section)
        for stored in store.parents(child_id, at)
        if stored.membership_id not in overlay.memberships
    }
    return out


def _section_members(store, overlay: Overlay, parent_id: str, section: str, at: int) -> set[str]:  # type: ignore[no-untyped-def]
    """Child ids in one section, an overlay membership winning over the stored
    row it supersedes — matched by membership id, so a shared node's *other*
    placement does not vanish just because this one moved."""

    members = {
        stored.child_id
        for stored in store.children(parent_id, at)
        if stored.section == section and stored.membership_id not in overlay.memberships
    }
    members |= {
        m.child_id
        for m in overlay.memberships.values()
        if m.parent_id == parent_id and m.section == section and not m.removed
    }
    return members


def _downgrade_visibility(
    store: KnowledgeStore, overlay: Overlay, local_id: str, node: ONode
) -> None:
    """Re-resolve the share policy after a relabel or a move and drop to
    ``local`` when it no longer says inherit. Only ever downgrades: going the
    other way is an authorization decision and stays with explicit ``mark``
    (review 2026-08-29 P1-3)."""

    if node.visibility != "shareable":
        return
    memberships = [m for m in overlay.memberships.values() if m.child_id == local_id and not m.removed]
    if not memberships:
        memberships = _load_memberships_of(store, overlay, local_id)
    for membership in memberships:
        allowed = effective_visibility(
            store, overlay, membership.parent_id,
            section=membership.section,
            label=str(node.payload.get("label") or "") or None,
        )
        if allowed != "shareable":
            node.visibility = "local"
            return


def _fold(store: KnowledgeStore, overlay: Overlay, resolver: _Resolver, index: int, op: dict[str, Any]) -> None:
    name = op["op"]
    if name == "create":
        kind = op["kind"]
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        payload = dict(op["payload"])
        validate_payload(kind, payload)
        local_id = _new_id()
        handle = op.get("handle")
        if handle:
            if handle in overlay.created or handle in resolver.bindings:
                raise ValueError(f"draft handle {handle} already bound")
            overlay.created[handle] = local_id
        parent_ref = op.get("parent")
        parent_id: str | None = None
        if parent_ref:
            parent_id, parent_expected = resolver.node(parent_ref)
            _load_node(store, overlay, parent_id, parent_expected)
        elif kind != "subject":
            raise ValueError("only subjects may be created without a parent")
        visibility = op.get("visibility") or effective_visibility(
            store, overlay, parent_id,
            section=str(op.get("section") or ""),
            label=str(payload.get("label") or "") or None,
        )
        overlay.nodes[local_id] = ONode(local_id, kind, payload, visibility, None, True, [index])
        if parent_id is not None:
            mid = _new_id()
            overlay.memberships[mid] = OMembership(
                mid, parent_id, local_id, op["section"], int(op.get("order_key", 1_000_000 + index)), None, True, [index]
            )
        return
    if name == "update":
        local_id, expected = resolver.node(op["id"])
        node = _load_node(store, overlay, local_id, expected)
        for key, value in dict(op["set"]).items():
            if key == "visibility":
                if value not in VISIBILITIES:
                    raise ValueError(f"bad visibility {value!r}")
                node.visibility = value
            elif key.startswith("payload."):
                path = key[len("payload.") :].split(".")
                if path[0] in ITEM_FIELDS:
                    raise ValueError(f"{key}: collections are item ops, not set")
                target = node.payload
                for part in path[:-1]:
                    target = target.setdefault(part, {})
                    if not isinstance(target, dict):
                        raise ValueError(f"{key}: not a mapping")
                target[path[-1]] = value
            else:
                raise ValueError(f"unknown set key {key!r}")
        validate_payload(node.kind, node.payload)
        _downgrade_visibility(store, overlay, local_id, node)
        node.ops.append(index)
        return
    if name == "add_item":
        local_id, expected = resolver.node(op["id"])
        _load_node(store, overlay, local_id, expected)
        field_name = op["field"]
        if field_name not in ITEM_FIELDS:
            raise ValueError(f"unknown item field {field_name!r}")
        value = str(op["value"]).strip()
        if not value:
            raise ValueError("empty item value")
        norm = _match_normalize(value)
        for existing in store.items_of(local_id, overlay.read_rev):
            if existing.field == field_name and _match_normalize(existing.value) == norm:
                overlay.rejected.append((index, f"add_item: {value!r} already present"))
                return
        for pending in overlay.items.values():
            if pending.local_id == local_id and pending.field == field_name and _match_normalize(pending.value) == norm and not pending.removed:
                overlay.rejected.append((index, f"add_item: {value!r} duplicated in batch"))
                return
        item_id = _new_id()
        overlay.items[item_id] = OItem(item_id, local_id, field_name, value, None, True, [index], fuzzy_enabled=(field_name == "misheard"))
        return
    if name == "remove_item":
        item_id, expected = resolver.item(op["item"])
        item = overlay.items.get(item_id)
        if item is None:
            stored = next((i for i in store.all_items(overlay.read_rev) if i.item_id == item_id), None)
            if stored is None:
                raise NotFoundError(item_id)
            item = OItem(item_id, stored.local_id, stored.field, stored.value, expected or stored.valid_from_rev, False)
            overlay.items[item_id] = item
        item.removed = True
        item.ops.append(index)
        return
    if name == "add_membership":
        child_id, child_expected = resolver.node(op["id"])
        parent_id, parent_expected = resolver.node(op["parent"])
        _load_node(store, overlay, child_id, child_expected)
        _load_node(store, overlay, parent_id, parent_expected)
        mid = _new_id()
        overlay.memberships[mid] = OMembership(mid, parent_id, child_id, op["section"], int(op.get("order_key", 1_000_000 + index)), None, True, [index])
        return
    if name in ("move_membership", "remove_membership"):
        mid, expected = resolver.membership(op["membership"])
        membership = overlay.memberships.get(mid)
        if membership is None:
            stored = next(
                (m for m in store.conn.execute(
                    "SELECT * FROM membership_versions WHERE membership_id=? AND valid_from_rev<=? AND (valid_to_rev IS NULL OR valid_to_rev>?)",
                    (mid, overlay.read_rev, overlay.read_rev),
                ).fetchall()),
                None,
            )
            if stored is None:
                raise NotFoundError(mid)
            membership = OMembership(mid, stored["parent_id"], stored["child_id"], stored["section"], stored["order_key"], expected or stored["valid_from_rev"], False)
            overlay.memberships[mid] = membership
        if name == "remove_membership":
            membership.removed = True
        else:
            if op.get("parent"):
                parent_id, parent_expected = resolver.node(op["parent"])
                _load_node(store, overlay, parent_id, parent_expected)
                membership.parent_id = parent_id
            if "section" in op:
                membership.section = op["section"]
            if "order_key" in op:
                membership.order_key = int(op["order_key"])
            moved = overlay.nodes.get(membership.child_id)
            if moved is None:
                stored_child = store.node(membership.child_id, overlay.read_rev)
                if stored_child is not None and stored_child.visibility == "shareable":
                    moved = _load_node(store, overlay, membership.child_id, None)
            if moved is not None:
                _downgrade_visibility(store, overlay, membership.child_id, moved)
        membership.ops.append(index)
        return
    if name == "link":
        source_id, source_expected = resolver.node(op["id"])
        target_id, target_expected = resolver.node(op["target"])
        if op["rel"] not in LINK_RELS:
            raise ValueError(f"unknown rel {op['rel']!r}")
        _load_node(store, overlay, source_id, source_expected)
        _load_node(store, overlay, target_id, target_expected)
        lid = _new_id()
        overlay.links[lid] = OLink(lid, source_id, op["rel"], target_id, [index])
        return
    if name == "retire":
        local_id, expected = resolver.node(op["id"])
        node = _load_node(store, overlay, local_id, expected)
        node.retired = True
        node.ops.append(index)
        if op.get("merged_into"):
            target_id, target_expected = resolver.node(op["merged_into"])
            _load_node(store, overlay, target_id, target_expected)
            node.merged_into = target_id
            lid = _new_id()
            overlay.links[lid] = OLink(lid, local_id, "supersedes", target_id, [index])
        for membership in _load_memberships_of(store, overlay, local_id):
            membership.removed = True
            membership.ops.append(index)
        return
    raise ValueError(f"unknown op {name!r}")


# ---- validation ---------------------------------------------------------------------


def _line_length(kind: str, payload: dict) -> int:
    """Length of the line AS RENDERED, label included.

    The label is part of what the injection pays for, and it is free-form: a
    cap that skipped it would be satisfied by `[三十个字的标记] 短正文`
    (review 2026-09-02).
    """

    label = str(payload.get("label") or "")
    prefix = f"[{label}] " if label else ""
    if kind == "term":
        body = "|".join(
            str(payload.get(field, ""))
            for field in ("surface", "zh", "alias_text", "desc")
        )
    else:
        body = str(payload.get("text") or "")
    return len(prefix) + len(body)


def _section_labels(store, overlay: Overlay, live_node, parent_id: str, section: str, at: int) -> set[str]:  # type: ignore[no-untyped-def]
    labels: set[str] = set()
    for member_id in _section_members(store, overlay, parent_id, section, at):
        member = live_node(member_id)
        if member is None:
            continue
        label = _match_normalize(str(member.payload.get("label") or ""))
        if label:
            labels.add(label)
    return labels


def _check_entry_budget(store, overlay: Overlay, live_node, problems: list) -> None:  # type: ignore[no-untyped-def]
    """The preset's `max_entry_tokens`, enforced MONOTONELY.

    The per-section caps bound each line and each section; this bounds the
    whole entry's injected projection. It cannot be a plain refusal: the fix
    for an oversized entry is itself a write, so refusing everything would lock
    the entry at its worst size and leave the model no move (owner 2026-09-02).

    So: while an entry is over budget, a batch may only make it smaller — no
    new lines under it, and no `update` that lengthens a line. Shrinking and
    deleting stay open, which is exactly the repair the notice on the entry's
    projection asks for."""

    from .render import entry_prompt_tokens

    at = store.current_rev()
    parents: set[str] = set()
    for membership in overlay.memberships.values():
        if not membership.removed:
            parents.add(membership.parent_id)
    for node in overlay.nodes.values():
        if node.retired:
            continue
        for parent_id, _section in _placements_of(store, overlay, node.local_id, at):
            parents.add(parent_id)

    for parent_id in sorted(parents):
        parent = live_node(parent_id)
        if parent is None or parent.kind != "subject":
            continue
        budget = preset_for_category(
            parent.payload.get("category") or "common"
        ).max_entry_tokens
        if budget is None or store.node(parent_id, at) is None:
            continue
        used = entry_prompt_tokens(store, parent_id, at)
        if used <= budget:
            continue
        for membership in overlay.memberships.values():
            child = overlay.nodes.get(membership.child_id)
            if membership.parent_id == parent_id and not membership.removed and (
                child is not None and child.is_new
            ):
                problems.append((
                    "membership", membership.membership_id,
                    f"条目已超注入预算（约 {used} / {budget} token），"
                    "本次不能再新增行——先把行改短、并条或删掉最弱的几条",
                ))
        for node in overlay.nodes.values():
            if node.retired or node.is_new:
                continue
            stored = store.node(node.local_id, at)
            if stored is None or (parent_id, ) not in {
                (p, ) for p, _s in _placements_of(store, overlay, node.local_id, at)
            }:
                continue
            if _line_length(node.kind, node.payload) > _line_length(stored.kind, stored.payload):
                problems.append((
                    "node", node.local_id,
                    f"条目已超注入预算（约 {used} / {budget} token），"
                    "这行只能改短不能改长",
                ))


def _check_label_references(store, overlay: Overlay, live_node, problems: list) -> None:  # type: ignore[no-untyped-def]
    """A section whose preset declares `labels_from` may only carry labels that
    exist in that other section.

    Both directions matter and both are the same check, which is why it runs
    over the whole entry rather than over the touched section: adding
    `[没有的约定] a → b` to 正例 is an orphan, and REMOVING the convention that
    a live example points at leaves the same orphan behind (review 2026-09-02).
    An unlabelled line is untouched — it is the deliberate escape for a second
    example (`presets/style.toml`)."""

    at = store.current_rev()
    parents: set[str] = set()
    for membership in overlay.memberships.values():
        parents.add(membership.parent_id)
    for node in overlay.nodes.values():
        for parent_id, _section in _placements_of(store, overlay, node.local_id, at):
            parents.add(parent_id)

    for parent_id in sorted(parents):
        parent = live_node(parent_id)
        if parent is None or parent.kind != "subject":
            continue
        preset = preset_for_category(parent.payload.get("category") or "common")
        sources: dict[str, set[str]] = {}
        for spec in preset.sections:
            if not spec.labels_from:
                continue
            if spec.labels_from not in sources:
                sources[spec.labels_from] = _section_labels(
                    store, overlay, live_node, parent_id, spec.labels_from, at
                )
            known = sources[spec.labels_from]
            for member_id in sorted(_section_members(store, overlay, parent_id, spec.name, at)):
                member = live_node(member_id)
                if member is None:
                    continue
                label = str(member.payload.get("label") or "")
                if label and _match_normalize(label) not in known:
                    problems.append((
                        "node", member.local_id,
                        f"「{spec.name}」的 [{label}] 在「{spec.labels_from}」里没有对应的那一条"
                        "——要么改成它真正演示的那条约定，要么去掉标记（无标记的例子合法，"
                        "只是不参与配对）",
                    ))


def _check_section_caps(store, overlay: Overlay, live_node, problems: list) -> None:  # type: ignore[no-untyped-def]
    """The preset's `max_lines` / `max_body_chars`, checked where every write
    passes (`docs/plans/translation-style-plan.md` §2.3).

    Caps exist to force a choice at write time: with the section full, adding a
    convention means retiring, merging or shortening another one. Enforcing
    them at read time instead would only hide the excess, and enforcing them in
    the prompt alone would make them advisory — a model that ignores the rule
    would silently grow the injection surface.

    Sibling counting mirrors `_check_label_uniqueness`: sections are read at the
    store's CURRENT rev (two batches that each add a line cannot both squeeze
    past a cap they can only see half of), and a section is checked whole
    whenever the batch touches it from either side."""

    at = store.current_rev()
    touched: set[tuple[str, str]] = set()
    for membership in overlay.memberships.values():
        if not membership.removed:
            touched.add((membership.parent_id, membership.section))
    for node in overlay.nodes.values():
        if node.retired:
            continue
        touched |= _placements_of(store, overlay, node.local_id, at)

    for parent_id, section in sorted(touched):
        parent = live_node(parent_id)
        if parent is None or parent.kind != "subject":
            continue
        spec = preset_for_category(parent.payload.get("category") or "common").section(section)
        if spec is None or (spec.max_lines is None and spec.max_body_chars is None):
            continue
        members = []
        for member_id in sorted(_section_members(store, overlay, parent_id, section, at)):
            member = live_node(member_id)
            if member is not None:
                members.append(member)
        if spec.max_lines is not None and len(members) > spec.max_lines:
            problems.append((
                "membership", f"{parent_id}:{section}",
                f"小节「{section}」最多 {spec.max_lines} 行，现在有 {len(members)} 行"
                "——要加新的就得先退掉一条、或把两条并成一条",
            ))
        if spec.max_body_chars is None:
            continue
        for member in members:
            length = _line_length(member.kind, member.payload)
            if length > spec.max_body_chars:
                problems.append((
                    "node", member.local_id,
                    f"小节「{section}」每行最多 {spec.max_body_chars} 字（含 [标记]），"
                    f"这行 {length} 字——写不下说明它该拆成两条",
                ))


def validate_overlay(store: KnowledgeStore, overlay: Overlay) -> list[tuple[str, str, str]]:
    """Return ``(entity, id, problem)`` triples; empty = consistent."""

    problems: list[tuple[str, str, str]] = []

    def live_node(local_id: str) -> ONode | None:
        node = overlay.nodes.get(local_id)
        if node is not None:
            return None if node.retired else node
        stored = store.node(local_id, overlay.read_rev)
        if stored is None:
            return None
        return ONode(local_id, stored.kind, dict(stored.payload), stored.visibility, stored.valid_from_rev, False)

    for node in overlay.nodes.values():
        if node.retired:
            continue
        try:
            validate_payload(node.kind, node.payload)
        except ValueError as exc:
            problems.append(("node", node.local_id, str(exc)))
    _check_label_uniqueness(store, overlay, live_node, problems)
    _check_membership_uniqueness(store, overlay, problems)
    _check_section_caps(store, overlay, live_node, problems)
    _check_label_references(store, overlay, live_node, problems)
    _check_entry_budget(store, overlay, live_node, problems)
    for item in overlay.items.values():
        if item.removed or item.field != "aliases":
            continue
        owner = live_node(item.local_id)
        if owner is None or owner.kind != "subject":
            continue
        preset = preset_for_category(owner.payload.get("category") or "common")
        if preset.label_by_role("aliases") is None:
            # An alias is a matching affordance, and a preset with no alias
            # field has nowhere to render one: it would be invisible in
            # `rendered/` and in the entry text while still answering in
            # `resolve()`. `create_entry` refuses these early; this is the
            # invariant behind that check, so no other path can slip one in
            # (review 2026-09-02, second round).
            problems.append((
                "item", item.item_id,
                f"category {owner.payload.get('category')!r} has no alias field",
            ))
    for membership in overlay.memberships.values():
        if membership.removed:
            continue
        parent = live_node(membership.parent_id)
        child = live_node(membership.child_id)
        if parent is None:
            problems.append(("membership", membership.membership_id, "parent missing or retired"))
            continue
        if child is None:
            problems.append(("membership", membership.membership_id, "child missing or retired"))
            continue
        if parent.kind != "subject":
            problems.append(("membership", membership.membership_id, "parent is not a subject"))
            continue
        preset = preset_for_category(parent.payload.get("category") or "common")
        spec = preset.section(membership.section)
        if spec is None:
            problems.append(("membership", membership.membership_id, f"section {membership.section!r} not allowed by preset {preset.name}"))
            continue
        if child.kind not in spec.body_kinds:
            problems.append(("membership", membership.membership_id, f"kind {child.kind} not allowed in section {membership.section!r}"))
    for link in overlay.links.values():
        if live_node(link.source_id) is None and not (overlay.nodes.get(link.source_id) and link.rel == "supersedes"):
            problems.append(("link", link.link_id, "source missing"))
        if live_node(link.target_id) is None:
            problems.append(("link", link.link_id, "target missing"))
    for item in overlay.items.values():
        if not item.removed and live_node(item.local_id) is None:
            problems.append(("item", item.item_id, "owner missing or retired"))
    return problems


# ---- apply ------------------------------------------------------------------------


class _FifoLock:
    """A fair handoff lock: waiters run strictly in arrival order.

    A plain ``threading.Lock`` promises no fairness, and the write queue's
    whole contract is that the revision order equals the arrival order
    (task-parallelism plan W2). Release hands the lock to the head waiter
    directly, so nobody can barge in between."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: deque[threading.Event] = deque()
        self._held = False

    def __enter__(self) -> "_FifoLock":
        with self._lock:
            if not self._held:
                self._held = True
                return self
            turn = threading.Event()
            self._waiters.append(turn)
        try:
            turn.wait()
        except BaseException:
            # A waiter that leaves without taking its turn must not wedge the
            # queue: the handoff is direct, so an abandoned waiter would hold
            # `_held` forever and block every later apply in this process.
            with self._lock:
                if turn.is_set():
                    self._hand_on_locked()  # already granted: pass it along
                else:
                    try:
                        self._waiters.remove(turn)
                    except ValueError:  # granted between wait() and the lock
                        self._hand_on_locked()
            raise
        return self

    def _hand_on_locked(self) -> None:
        """Release, holding ``self._lock``: next waiter or nobody."""

        if self._waiters:
            self._waiters.popleft().set()  # handoff: stays held
        else:
            self._held = False

    def __exit__(self, *exc_info: Any) -> None:
        with self._lock:
            self._hand_on_locked()


# One ordered single-writer queue per store file (plan W2): the write
# transaction is milliseconds next to the minutes of generation behind it, so
# in-process writers WAIT here instead of falling into the cross-process
# loser path and discarding a finished proposal. Keyed by the physical
# resource (the database path), per the §1.1 rule; process-global on purpose.
_APPLY_QUEUE_GUARD = threading.Lock()
_APPLY_QUEUES: dict[str, _FifoLock] = {}


def _apply_queue(store: KnowledgeStore) -> _FifoLock:
    key = os.path.normcase(str(store.path.resolve()))
    with _APPLY_QUEUE_GUARD:
        return _APPLY_QUEUES.setdefault(key, _FifoLock())


def preview(store: KnowledgeStore, envelope: Envelope) -> tuple[Overlay, list[tuple[str, str, str]]]:
    """Side-effect-free check (``kb_validate_draft``): overlay + problems."""

    shape = envelope.validate_shape()
    if shape:
        raise EnvelopeError("; ".join(shape))
    overlay = build_overlay(store, envelope)
    return overlay, validate_overlay(store, overlay)


def apply_envelope(
    store: KnowledgeStore,
    envelope: Envelope,
    *,
    revision_kind: str = "harness",
    note: str = "",
) -> ApplyResult:
    overlay, problems = preview(store, envelope)
    result = ApplyResult(rev=None, rejected_ops=list(overlay.rejected), created=dict(overlay.created))
    if problems:
        # Uniqueness losses against a concurrently committed row are resolved
        # per-op (plan W2) before anything is declared invalid; every other
        # problem (schema, preset, membership duplicates) still rolls back
        # the whole envelope, exactly as before.
        _resolve_label_conflicts(store, overlay, result)
        problems = validate_overlay(store, overlay)
    if problems:
        result.rolled_back = True
        result.rollback_reason = "overlay invalid before CAS: " + "; ".join(f"{e} {i}: {p}" for e, i, p in problems)
        return result
    if not _overlay_has_writes(overlay):
        return result  # everything resolved away; no empty revision

    try:
        # The per-root single-writer queue (plan W2): in-process writers wait
        # their turn in arrival order instead of racing BEGIN IMMEDIATE's
        # busy timeout; cross-process contention stays with
        # `knowledge_write_lock`'s loser semantics.
        with _apply_queue(store), store.begin(revision_kind, task_id=envelope.task_id, proposal_hash=envelope.proposal_hash(), base_rev=envelope.knowledge_read_rev, note=note) as txn:
            _drop_stale(store, overlay, result)
            # Again, against the authoritative in-txn state: a competing label
            # can land between the preview above and our BEGIN IMMEDIATE.
            _resolve_label_conflicts(store, overlay, result)
            leftover = validate_overlay(store, overlay)
            _drop_dependents(overlay, leftover, result)
            leftover = validate_overlay(store, overlay)
            if leftover:
                raise _Rollback("overlay invalid after conflict removal: " + "; ".join(f"{e} {i}: {p}" for e, i, p in leftover))
            if not _overlay_has_writes(overlay):
                # The in-txn pass is the one that sees the authoritative state,
                # so it can empty an overlay the pre-queue check found non-empty
                # (a competitor committed between `preview` and BEGIN
                # IMMEDIATE -- exactly the W2 race). Without this the revision
                # row inserted by `begin` would commit with zero version rows:
                # a phantom apply in the history, and a rev bump that says
                # "something changed" to everything reading current_rev.
                raise _NoWrites
            _write(txn, overlay, result)
            result.rev = txn.rev
            # Provenance booking (plan §11.4 / O8-O9): a user apply endorses
            # the values it set (evidence_kind=user); a harness apply records
            # that the values came out of this run's material
            # (evidence_kind=transcript -- revision-level granularity for now,
            # hint-level source ids stay in the artifacts). Other kinds
            # (import/pull/share/...) book nothing.
            #
            # INSIDE the transaction, and therefore inside the write queue:
            # these are plain INSERTs on an autocommit connection, so booking
            # them afterwards took the write lock once per row while other
            # queued writers were holding it -- observed as `database is
            # locked` under the very concurrency W2 exists for. It was also
            # the one write that could fail AFTER the revision committed,
            # raising past a successful apply.
            kind_for = {"user": "user", "harness": "transcript"}.get(revision_kind)
            if kind_for:
                from .signals import book_revision_evidence

                book_revision_evidence(
                    store, txn.rev, evidence_kind=kind_for, task_id=envelope.task_id
                )
    except _NoWrites:
        # Not a rollback: the conflicts ARE the outcome, same shape as the
        # pre-queue early return (rev None, rolled_back False).
        result.rev = None
        return result
    except _Rollback as exc:
        result.rolled_back = True
        result.rollback_reason = str(exc)
        result.rev = None
        return result
    except StaleWriteError as exc:  # a race between our CAS check and the write
        result.rolled_back = True
        result.rollback_reason = f"stale write during commit: {exc}"
        result.rev = None
        return result
    except NotFoundError as exc:  # a row vanished between check and write
        result.rolled_back = True
        result.rollback_reason = f"entity vanished during commit: {exc}"
        result.rev = None
        return result
    applied = {op for node in overlay.nodes.values() for op in node.ops}
    applied |= {op for item in overlay.items.values() for op in item.ops}
    applied |= {op for m in overlay.memberships.values() for op in m.ops}
    applied |= {op for link in overlay.links.values() for op in link.ops}
    dropped = {op for conflict in result.conflicts for op in conflict.dropped_ops} | {index for index, _ in result.rejected_ops}
    result.applied_ops = sorted(applied - dropped)
    return result


class _Rollback(Exception):
    pass


class _NoWrites(Exception):
    """Every intent was resolved away inside the transaction; write nothing."""


def _current_from_rev(store: KnowledgeStore, table: str, key: str, ident: str) -> int | None:
    row = store.conn.execute(
        f"SELECT valid_from_rev FROM {table} WHERE {key}=? AND valid_to_rev IS NULL", (ident,)
    ).fetchone()
    return int(row["valid_from_rev"]) if row else None


def _drop_dead_references(store: KnowledgeStore, overlay: Overlay, result: ApplyResult) -> None:
    """Read-only dependencies must still be alive at commit time (plan §2.5).

    ``add_item``'s owner, a link's ends and a membership's parent/child carry
    no write op of their own, so the per-entity CAS below never looks at them —
    without this pass a concurrently retired owner would happily receive new
    items. Liveness is the requirement, not version equality: an owner that
    merely moved forward still accepts additive ops (the merge rule). Dropping
    a dead reference cascades: a node created only to live under it goes too.
    """

    dead = [
        local_id
        for local_id, node in overlay.nodes.items()
        if not node.is_new and not node.retired and store.node(local_id) is None
    ]
    for local_id in dead:
        _drop_node_with_dependents(
            overlay, result, local_id, f"references {local_id}, retired concurrently"
        )


def _drop_node_with_dependents(
    overlay: Overlay, result: ApplyResult, seed_id: str, reason: str
) -> None:
    """Drop one node intent plus its dependency closure — everything this
    envelope hung off it (items, memberships, links, and new line nodes left
    homeless by the cascade). Only overlay intents are touched; committed
    rows are never (task-parallelism plan W2 boundary)."""

    dead = [seed_id]
    while dead:
        local_id = dead.pop()
        node = overlay.nodes.pop(local_id, None)
        if node is None:
            continue
        if node.ops:
            result.conflicts.append(Conflict("node", local_id, "dropped: " + reason, list(node.ops)))
        for item_id, item in list(overlay.items.items()):
            if item.local_id == local_id:
                result.conflicts.append(Conflict("item", item_id, "dropped: " + reason, list(item.ops)))
                del overlay.items[item_id]
        for mid, membership in list(overlay.memberships.items()):
            if membership.parent_id == local_id or membership.child_id == local_id:
                result.conflicts.append(Conflict("membership", mid, "dropped: " + reason, list(membership.ops)))
                del overlay.memberships[mid]
                other = membership.child_id if membership.parent_id == local_id else membership.parent_id
                other_node = overlay.nodes.get(other)
                if (
                    other_node is not None
                    and other_node.is_new
                    and other_node.kind != "subject"
                    and not any(m.child_id == other for m in overlay.memberships.values())
                ):
                    dead.append(other)  # a new line node with no home left
        for lid, link in list(overlay.links.items()):
            if link.source_id == local_id or link.target_id == local_id:
                result.conflicts.append(Conflict("link", lid, "dropped: " + reason, list(link.ops)))
                del overlay.links[lid]


def _our_label_source(
    overlay: Overlay, parent_id: str, section: str, member_id: str, stored_label: str | None
) -> str:
    """How this envelope put ``member_id``'s label into ``(parent, section)``:
    ``"create"`` (new node), ``"relabel"`` (label changed vs the stored row),
    ``"placement"`` (a membership this envelope added or moved here), or ``""``
    (a committed row this envelope is not responsible for)."""

    node = overlay.nodes.get(member_id)
    if node is not None and node.is_new:
        return "create"
    if (
        node is not None
        and node.ops
        and stored_label is not None
        and _match_normalize(str(node.payload.get("label") or "")) != _match_normalize(stored_label)
    ):
        return "relabel"
    for membership in overlay.memberships.values():
        if (
            membership.child_id == member_id
            and membership.parent_id == parent_id
            and membership.section == section
            and not membership.removed
            and (membership.is_new or membership.ops)
        ):
            return "placement"
    return ""


def _label_taken_at(
    store: KnowledgeStore, parent_id: str, section: str, label_norm: str, rev: int
) -> bool:
    """Was this label already visible in ``(parent, section)`` at ``rev``?"""

    for stored in store.children(parent_id, rev):
        if stored.section != section:
            continue
        node = store.node(stored.child_id, rev)
        if node is not None and _match_normalize(str(node.payload.get("label") or "")) == label_norm:
            return True
    return False


def _resolve_label_conflicts(store: KnowledgeStore, overlay: Overlay, result: ApplyResult) -> None:
    """CONCURRENCY losers on the label invariant lose their OP, not the batch
    (task-parallelism plan W2): a relabel onto a just-taken name is reverted
    to the stored label, a node created under one is dropped with its
    dependency closure, a membership moved/added onto one is rejected. The
    committed row that won stays untouched, and everything else in the
    envelope commits.

    Only losses the author could NOT have seen are resolved: a label already
    visible at the envelope's ``knowledge_read_rev`` is an authoring error and
    keeps the whole-envelope rollback (the user edit path relies on being
    told, not silently corrected). Duplicates this pass cannot attribute to
    the envelope at all (two committed rows) are equally left for
    ``validate_overlay`` to roll back on."""

    at = store.current_rev()
    touched: set[tuple[str, str]] = set()
    for membership in overlay.memberships.values():
        if not membership.removed:
            touched.add((membership.parent_id, membership.section))
    for node in overlay.nodes.values():
        if node.retired or not str(node.payload.get("label") or "").strip():
            continue
        touched |= _placements_of(store, overlay, node.local_id, at)

    for parent_id, section in sorted(touched):
        groups: dict[str, list[tuple[str, str, str | None]]] = {}
        for member_id in sorted(_section_members(store, overlay, parent_id, section, at)):
            member = overlay.nodes.get(member_id)
            if member is not None and member.retired:
                continue
            stored = store.node(member_id, at)
            stored_label = str(stored.payload.get("label") or "") if stored is not None else None
            display = (
                str(member.payload.get("label") or "") if member is not None
                else (stored_label or "")
            )
            if not _match_normalize(display):
                continue
            groups.setdefault(_match_normalize(display), []).append(
                (member_id, display, stored_label)
            )
        for label_norm, members in groups.items():
            if len(members) < 2:
                continue
            if _label_taken_at(store, parent_id, section, label_norm, overlay.read_rev):
                continue  # visible when authored: an error to report, not a race
            sources = {
                member_id: _our_label_source(overlay, parent_id, section, member_id, stored_label)
                for member_id, _display, stored_label in members
            }
            losers = [entry for entry in members if sources[entry[0]]]
            if not losers or len(losers) == len(members):
                # committed-vs-committed is not this envelope's to fix, and
                # all-ours (two same-label intents in one batch) is an
                # authoring error: both keep the rollback path.
                continue
            for member_id, display, stored_label in losers:
                reason = f"标记 [{display}] 在小节 {section!r} 内已被占用"
                source = sources[member_id]
                if source == "create":
                    _drop_node_with_dependents(overlay, result, member_id, reason)
                elif source == "relabel":
                    node = overlay.nodes[member_id]
                    node.payload["label"] = stored_label
                    result.conflicts.append(
                        Conflict("node", member_id, "relabel reverted: " + reason, list(node.ops))
                    )
                    stored = store.node(member_id, at)
                    if stored is not None and node.payload == stored.payload and not node.retired:
                        # The revert cancelled the whole intent: leave the node
                        # exactly as stored -- including visibility, which the
                        # fold re-derived because of the now-reverted relabel.
                        node.visibility = stored.visibility
                        node.ops = []
                    elif stored is not None:
                        # Other changes of the same op survive; re-derive the
                        # (downgrade-only) visibility from the stored value so
                        # the reverted label leaves no spurious downgrade.
                        node.visibility = stored.visibility
                        _downgrade_visibility(store, overlay, member_id, node)
                else:  # placement
                    for mid, membership in list(overlay.memberships.items()):
                        if (
                            membership.child_id == member_id
                            and membership.parent_id == parent_id
                            and membership.section == section
                            and not membership.removed
                            and (membership.is_new or membership.ops)
                        ):
                            result.conflicts.append(
                                Conflict("membership", mid, "move/add rejected: " + reason,
                                         list(membership.ops))
                            )
                            del overlay.memberships[mid]


def _overlay_has_writes(overlay: Overlay) -> bool:
    """Would ``_write`` emit any row for this overlay? An envelope whose every
    intent was resolved away must not spend an empty revision."""

    return (
        any(n.is_new or n.retired or n.ops for n in overlay.nodes.values())
        or any(i.is_new or i.removed for i in overlay.items.values())
        or any(m.is_new or m.removed or m.ops for m in overlay.memberships.values())
        or bool(overlay.links)
    )


def _drop_stale(store: KnowledgeStore, overlay: Overlay, result: ApplyResult) -> None:
    """One CAS check per existing entity; stale intents are dropped by type
    (plan §2.5): scalar/prose changes skipped, purely additive ops kept,
    retire/remove rejected."""

    _drop_dead_references(store, overlay, result)
    for node in list(overlay.nodes.values()):
        if node.is_new or not node.ops:
            continue
        current = _current_from_rev(store, "node_versions", "local_id", node.local_id)
        if current == node.expected_from_rev:
            continue
        stored = store.node(node.local_id)  # current version
        reason = f"expected version {node.expected_from_rev}, current {current}"
        if node.retired:
            # reject the retire; keep the node and its memberships as they are now
            result.conflicts.append(Conflict("node", node.local_id, "retire rejected: " + reason, list(node.ops)))
            node.retired = False
            node.merged_into = None
            for membership in overlay.memberships.values():
                if membership.child_id == node.local_id and membership.removed and set(membership.ops) <= set(node.ops):
                    membership.removed = False
            for lid in [l for l, link in overlay.links.items() if link.source_id == node.local_id and link.rel == "supersedes"]:
                del overlay.links[lid]
        if stored is not None and (node.payload != stored.payload or node.visibility != stored.visibility):
            result.conflicts.append(Conflict("node", node.local_id, "update skipped: " + reason, list(node.ops)))
            node.payload = dict(stored.payload)
            node.visibility = stored.visibility
        node.expected_from_rev = current
        node.ops = []  # additive ops on this node live on their own entities
    for item in list(overlay.items.values()):
        if item.is_new:
            continue
        current = _current_from_rev(store, "item_versions", "item_id", item.item_id)
        if current != item.expected_from_rev:
            result.conflicts.append(Conflict("item", item.item_id, f"remove rejected: expected {item.expected_from_rev}, current {current}", list(item.ops)))
            del overlay.items[item.item_id]
    for membership in list(overlay.memberships.values()):
        if membership.is_new or not membership.ops:
            continue
        current = _current_from_rev(store, "membership_versions", "membership_id", membership.membership_id)
        if current != membership.expected_from_rev:
            result.conflicts.append(Conflict("membership", membership.membership_id, f"move/remove rejected: expected {membership.expected_from_rev}, current {current}", list(membership.ops)))
            del overlay.memberships[membership.membership_id]
    # new items whose normalized value now exists (added concurrently): merge = drop as duplicate
    for item in list(overlay.items.values()):
        if not item.is_new:
            continue
        norm = _match_normalize(item.value)
        if any(i.field == item.field and _match_normalize(i.value) == norm for i in store.items_of(item.local_id)):
            result.conflicts.append(Conflict("item", item.item_id, f"add_item merged: {item.value!r} already present", list(item.ops)))
            del overlay.items[item.item_id]


def _drop_dependents(overlay: Overlay, problems: list[tuple[str, str, str]], result: ApplyResult) -> None:
    """Second pass: memberships/links/items that became invalid because their
    node intent was dropped are removed (dependency closure); node-level
    problems are left for the caller to roll back on."""

    for entity, ident, problem in problems:
        if entity == "membership" and ident in overlay.memberships:
            result.conflicts.append(Conflict("membership", ident, "dropped: " + problem, list(overlay.memberships[ident].ops)))
            del overlay.memberships[ident]
        elif entity == "link" and ident in overlay.links:
            result.conflicts.append(Conflict("link", ident, "dropped: " + problem, list(overlay.links[ident].ops)))
            del overlay.links[ident]
        elif entity == "item" and ident in overlay.items:
            result.conflicts.append(Conflict("item", ident, "dropped: " + problem, list(overlay.items[ident].ops)))
            del overlay.items[ident]


def _write(txn: Transaction, overlay: Overlay, result: ApplyResult) -> None:
    for node in overlay.nodes.values():
        if node.is_new:
            txn.create_node(node.local_id, node.kind, node.payload, visibility=node.visibility)
        elif node.retired:
            txn.tombstone_node(node.local_id, expected_from_rev=node.expected_from_rev)
            # retire closes the node's whole footprint in the same revision so
            # `knowledge restore` can bring it back wholesale (plan §2.5):
            # items and pre-existing links are cascaded at write time against
            # the *current* rows — they are implied by the retire, not
            # independent intents, so no per-row CAS. The batch's own
            # `supersedes` link (if any) is created after this loop and stays.
            for item in txn.store.items_of(node.local_id):
                txn.tombstone_item(item.item_id)
            for row in txn.conn.execute(
                "SELECT link_id FROM link_versions WHERE valid_to_rev IS NULL AND (source_id=? OR target_id=?)",
                (node.local_id, node.local_id),
            ).fetchall():
                txn.tombstone_link(row["link_id"])
        elif node.ops:
            txn.update_node(node.local_id, payload=node.payload, visibility=node.visibility, expected_from_rev=node.expected_from_rev)
    for item in overlay.items.values():
        owner = overlay.nodes.get(item.local_id)
        if owner is not None and owner.retired:
            continue  # the retire cascade above already closed (or obviated) it
        if item.is_new:
            txn.create_item(item.item_id, item.local_id, item.field, item.value, fuzzy_enabled=item.fuzzy_enabled)
        elif item.removed:
            txn.tombstone_item(item.item_id, expected_from_rev=item.expected_from_rev)
    for membership in overlay.memberships.values():
        if membership.is_new:
            txn.create_membership(membership.membership_id, membership.parent_id, membership.child_id, membership.section, membership.order_key)
        elif membership.removed:
            txn.tombstone_membership(membership.membership_id, expected_from_rev=membership.expected_from_rev)
        elif membership.ops:
            txn.move_membership(membership.membership_id, parent_id=membership.parent_id, section=membership.section, order_key=membership.order_key, expected_from_rev=membership.expected_from_rev)
    for link in overlay.links.values():
        txn.create_link(link.link_id, link.source_id, link.rel, link.target_id)
