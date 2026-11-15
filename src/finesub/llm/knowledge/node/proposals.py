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
from typing import Any, Callable, Mapping, Sequence

from finesub import jsonl

from ..base import _line_dedup_token, _match_normalize, _split_names
from .apply import ApplyResult, apply_envelope, parked_section
from .envelope import Binding, Envelope
from .importer import alias_row_delta, classify_line, extract_misheard, storable_term_payload
from .model import BODY_KIND_LABELS, METADATA_SECTION, RECATEGORIZED_REASON_FIELD, entry_name_problem
from .presets import MATCHABLE_CATEGORIES, preset_for_category
from .render import HandleMap, subject_aliases
from .repo import KnowledgeRepo

MODEL_OPS: tuple[str, ...] = (
    "append_lines",
    "update",
    "remove",
    "create_entry",
    "retire_entry",
    "rename_entry",
    # The intro line under the H1, replaced whole (owner 2026-10-01). The H1
    # stays with `rename_entry` (dedupe, script folding, namespaces); the
    # entry aliases are the `[别名]` row's `@a` handle (`update` / `remove`,
    # cold read Q5), translated into `add_item` / `remove_item`.
    "set_intro",
    # The two moves (plan §0 requirements 7, 11 and 13). Both existed at the
    # engine layer already -- `move_membership` verbatim, a category change as
    # a plain `update` -- and neither was in the model's vocabulary, so the
    # only way an LLM could relocate anything was to delete it and retype the
    # text, losing the node, its history and its evidence. That is the shape
    # "migration" was forced into, and why every template change wanted
    # bespoke code.
    "move_line",     # a line -- or a whole section -- to another section or entry
    "recategorize",  # a whole entry to another category; reason required
    # One section split off into a sub-entry `母·节`, linked `part_of` (plan
    # kb-entry-size). Only where the category's preset says `splittable`.
    "split_entry",
    "add_item",
    "remove_item",
)
COMMON_ENTRY_TYPES: tuple[str, ...] = ("游戏", "动画", "社区", "其他")

#: The skip reason of a proposal a concurrent writer displaced. The repair
#: round's own section says that in words, so its rows leave this out.
CAS_DROPPED_REASON = "dropped by CAS conflict"

_PROPOSALS_RE = re.compile(
    r"<knowledge_proposals\b[^>]*>(?P<body>.*?)</knowledge_proposals>", re.IGNORECASE | re.DOTALL
)
_CODE_FENCE_RE = re.compile(r"```(?:jsonl|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
#: No prompt asks for `<reasoning>` any more, but a model that writes one and
#: NAMES the output block while reasoning about it ("候选不判 dismiss，
#: <knowledge_proposals> 输出空块") would otherwise open the match there and
#: swallow `</reasoning>` plus the real opening tag as body lines. Observed in
#: a live repair session (2026-08-29), which then reported them as invalid
#: JSON. Strip the reasoning span before looking for the block.
_REASONING_RE = re.compile(r"<reasoning\b[^>]*>.*?</reasoning>", re.IGNORECASE | re.DOTALL)
#: The handle comment every prompt-projected line ends with (`render_subject`).
#: The prompts say it is not content, but a line copied whole and edited would
#: otherwise store it -- in a description, or as part of an alias.
_HANDLE_COMMENT_RE = re.compile(r"\s*<!--\s*@[a-z]+\d+\s*-->")


def _without_handle(line: str) -> str:
    """Wherever it sits: an alias added after the copied comment would carry
    it in the middle of the row. A handle comment is never content."""

    return _HANDLE_COMMENT_RE.sub("", line).rstrip()


def strip_reasoning(text: str) -> str:
    return _REASONING_RE.sub("", text or "")


def _subject_key(repo: KnowledgeRepo, subject_id: str) -> str:
    """The entry name a report should show for an internal subject id."""

    node = repo.store.node(subject_id)
    return str(node.payload.get("surface") or subject_id) if node else subject_id


def _warn_parked(parked: Sequence[Mapping[str, Any]]) -> None:
    """Say out loud that a recategorize moved lines nobody proposed moving.

    Here rather than in the knowledge-update round: every front end that can
    recategorize passes through this function -- the human CLI, `repair`, the
    agent tool -- and a silent restructuring is the one outcome none of them
    should have to read a report to notice. The staging section reports its own
    line count while non-empty, so this is the immediate notice, not the only
    one.
    """

    if not parked:
        return
    from finesub.reporting import current_reporter

    shown = "；".join(
        f"{row['entry']} 的「{row['from_section']}」→ {row['line']}" for row in parked[:5]
    )
    current_reporter().warning(
        "knowledge-recategorize-parked",
        f"重分类把 {len(parked)} 行挪进了暂存区（目标类别的模版放不下）：{shown}",
        impact="这些行仍在库里，但已离开原小节",
        action="下一轮维护把它们归位或删掉",
    )


@dataclass
class ProposalRecord:
    category: str
    entry: str
    op: str
    section: str
    status: str
    reason: str
    #: Index of the proposal this record came from in the parsed block (-1 when
    #: unknown). The repair round quotes that proposal by it: an `update` names
    #: a line by handle, so matching by entry alone could quote another
    #: proposal of the same entry -- one that landed.
    proposal: int = -1

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LineChange:
    """What one proposal does to one line, put to a caller's ``line_gate``: ``op`` (`append_lines` / `update` /
    `remove` / `move_line`), the entry's category and the line's section, and the line's payload before and
    after (None on the side where the line does not exist)."""

    op: str
    category: str
    section: str
    before: Mapping[str, Any] | None
    after: Mapping[str, Any] | None


#: A caller's hold on some lines: "" admits the change, anything else is the reason it is skipped -- reported
#: with the other skipped proposals; whether the proposal repair round shows it to the model is the caller's call.
LineGate = Callable[[LineChange], str]

#: Why a model's change to a line under a label only people write is skipped (`LabelSpec.model_hidden`). Not for
#: the proposal repair round: the model was never told the label exists.
HIDDEN_LABEL_HELD = "这个标记只由用户写，模型不加、不改、不删"


def hidden_label_gate(change: LineChange) -> str:
    """A gate holding every change that adds, rewrites, removes or moves a line under a label its category's
    preset marks `model_hidden` (the style's `[任务]`, owner 2026-11-07)."""

    preset = preset_for_category(change.category)
    sections = (*preset.sections, *([preset.default_section] if preset.default_section else []))
    hidden = {_match_normalize(label.name) for spec in sections for label in spec.labels if label.model_hidden}
    if hidden and any(
        payload is not None and _match_normalize(str(payload.get("label") or "")) in hidden
        for payload in (change.before, change.after)
    ):
        return HIDDEN_LABEL_HELD
    return ""


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
    #: lines a `recategorize` parked in the target's staging section. NOT
    #: skips: the move happened, and these are what the next pass has to put
    #: away. Separate from `applied` because nobody proposed them.
    parked: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": [r.to_dict() for r in self.applied],
            "skipped": [r.to_dict() for r in self.skipped],
            "rev": self.rev,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "conflicts": self.conflicts,
            "applied_proposals": list(self.applied_proposals),
            "parked": list(self.parked),
        }


# ---- parsing ---------------------------------------------------------------------------


def has_proposals_block(text: str) -> bool:
    """Whether the answer has a closed `<knowledge_proposals>` block, as `parse_model_proposals` reads it."""
    return _PROPOSALS_RE.search(strip_reasoning(text or "")) is not None


def parse_model_proposals(text: str) -> list[dict[str, Any]]:
    match = _PROPOSALS_RE.search(strip_reasoning(text))
    body = match.group("body") if match else (text or "")
    fence = _CODE_FENCE_RE.search(body)
    if fence:
        body = fence.group(1)
    out: list[dict[str, Any]] = []
    for line in jsonl.lines(body):
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
    line_gate: LineGate | None = None
    ops: list[dict[str, Any]] = field(default_factory=list)
    report: ProposalReport = field(default_factory=ProposalReport)
    touched_subjects: set[str] = field(default_factory=set)
    new_handles: int = 0
    #: subject id -> sections this batch took away from it (a split, a whole-
    #: section move): its `section_order` drops all of them, not just the
    #: latest -- each op recomputes the order from the STORED payload. Keyed by
    #: id, not by the ref: one entry named by handle and by name is one entry
    trimmed: dict[str, set[str]] = field(default_factory=dict)
    #: subjects that gained or are a sub-entry in this batch (no recategorize)
    split_family: set[str] = field(default_factory=set)
    #: subject id -> the name a `rename_entry` earlier in this batch gave it:
    #: a split later in the batch derives the sub-entry's name from that (the
    #: engine's rename cascade only sees sub-entries already stored)
    renamed: dict[str, str] = field(default_factory=dict)
    current_proposal: int = -1  # index of the proposal being translated
    created_keys: dict[tuple[str, str], str] = field(default_factory=dict)  # (category, key) -> handle
    # section -> existing dedup tokens, per subject ref (for append dedupe)
    pending_tokens: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    pending_labels: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    #: subject id -> the category a `recategorize` in THIS block moved it to
    recategorized: dict[str, str] = field(default_factory=dict)
    #: membership id -> the section a `move_line` in THIS block moved it to
    relocated: dict[str, str] = field(default_factory=dict)
    #: membership id -> ``(category, entry key)`` of the entry a cross-entry
    #: `move_line` (`to_entry`) in THIS block moved it under
    reparented: dict[str, tuple[str, str]] = field(default_factory=dict)

    def placement_of(self, node_id: str, membership: Any) -> tuple[str, str, str]:
        """``(category, entry key, section)`` for a line, AS OF THIS BATCH.

        Every line-level op needs the same three answers, and each was reading
        them out of the store on its own -- so an earlier `recategorize` or
        `move_line` in the same block was invisible to whichever op ran next,
        and the same hole had to be closed once per entry point. It is one
        function now so the next op added gets it for free.

        ⚠ The section decides which SPEC the text is classified against, which
        is why staleness here reads as a grammar error rather than as a
        misplacement: a free sentence carrying pipes is a malformed term row
        under a section that takes terms, and ordinary prose under one that
        does not.
        """

        moved = self.reparented.get(membership.membership_id)
        if moved is not None:
            # Under another entry since an earlier `move_line` in this block;
            # that entry's category and key are the answers now.
            return moved[0], moved[1], self.relocated.get(membership.membership_id, membership.section)
        parent = self.repo.store.node(membership.parent_id, self.rev)
        stored = parent.payload.get("category", "") if parent else ""
        category = self.category_of(membership.parent_id, stored)
        # An explicit `move_line` first (that is the placement the batch chose),
        # then the automatic parking a recategorize would apply on top of it.
        section = self.relocated.get(membership.membership_id, membership.section)
        if category != stored:
            child = self.repo.store.node(membership.child_id, self.rev)
            parked = parked_section(
                target=category, source=stored, section=section,
                kind=child.kind if child else "",
            ) if child else None
            if parked:
                section = parked
        return category, parent.payload.get("surface", "") if parent else "", section

    def category_of(self, subject_id: str, stored: str) -> str:
        """The category this entry has as of this block, not as of the store.

        A `recategorize` does not take effect until the ops are folded, so
        every later proposal resolving the entry from the store was judged
        against the category it is LEAVING. Both directions were wrong and
        they fail differently: judged too strictly the proposal is refused per
        op, judged too loosely it passes translation and dies in
        `validate_overlay` -- which reports per envelope: the whole batch
        rolls back, or in the knowledge update the op is rejected after the
        fact under the folded batch's message (review 2026-09-07, fourth
        round).
        """

        return self.recategorized.get(subject_id) or stored

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
            return (
                value,
                self.category_of(binding.id, node.payload.get("category", "")),
                node.payload.get("surface", ""),
                binding.id,
            )
        created = self.created_entry(value, category)
        if created is not None:
            created_category, handle = created
            return handle, created_category, value, ""
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
        return (
            resolved.subject_id,
            self.category_of(resolved.subject_id, resolved.category),
            resolved.key,
            resolved.subject_id,
        )

    def created_entry(self, value: str, category: str | None) -> tuple[str, str] | None:
        """``(category, handle)`` of an entry this batch creates, by name.

        A name given with no category -- `move_line`'s `to_entry` never carries
        one -- or with a matchable one also finds a matchable entry created
        under another category: those keys share one namespace (the `resolve`
        fallback below), and without this a `create_entry` + `move_line` pair
        in one batch moved nothing ("to_entry does not exist").
        """

        exact = self.created_keys.get((category or "", value))
        if exact is not None:
            return category or "", exact
        if category and category not in MATCHABLE_CATEGORIES:
            return None
        for (created_category, key), handle in self.created_keys.items():
            if key == value and created_category in MATCHABLE_CATEGORIES:
                return created_category, handle
        return None

    def alias_row(self, value: Any) -> str | None:
        """The subject an `@a` handle (its `[别名]` row) stands for."""

        binding = self.bindings.get(value) if isinstance(value, str) else None
        return binding.id if binding is not None and binding.kind == "aliases" else None

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

    def bind_membership(self, membership: Any) -> str:
        """Handle for a membership the model did not see rendered.

        The rendered projection shows lines, not the edges that place them, so
        a `move_line` proposal names the LINE and this resolves the edge under
        it -- same shape as `bind_item` does for the alias column.
        """

        for handle, binding in self.bindings.items():
            if binding.kind == "membership" and binding.id == membership.membership_id:
                return handle
        handle = f"@msync{len(self.bindings) + 1}"
        self.bindings[handle] = Binding(
            handle=handle,
            kind="membership",
            id=membership.membership_id,
            expected_valid_from_rev=membership.valid_from_rev,
        )
        return handle

    def subject_handle(self, local_id: str) -> str:
        """The subject ref `touched_subjects` expects, bound on demand."""

        return self.bind_node(local_id) or local_id

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

    def refused(self, change: LineChange) -> str:
        """Why this caller's ``line_gate`` holds ``change`` back ("" when it does not, or there is none)."""

        return self.line_gate(change) if self.line_gate is not None else ""

    def record(self, status: str, category: str, entry: str, op: str, section: str, reason: str) -> None:
        record = ProposalRecord(category, entry, op, section, status, reason, self.current_proposal)
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
            {"record": ProposalRecord(category, entry, op_name, section, "", label, self.current_proposal),
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
    line_gate: LineGate | None = None,
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
    passes the full set. ``line_gate``: a caller's hold on some line changes
    (`LineChange`), each refused one skipped with the gate's reason."""

    ctx = _Context(
        repo=repo,
        rev=knowledge_read_rev,
        bindings={b.handle: b for b in bindings},
        allow_categories=tuple(allow_categories),
        line_gate=line_gate,
    )
    for index, proposal in enumerate(proposals):
        ctx.current_proposal = index
        op = str(proposal.get("op", ""))
        reason = str(proposal.get("reason", "")).strip()
        try:
            if op in ("update", "remove", "add_item") and ctx.alias_row(proposal.get("id")):
                _alias_row(ctx, proposal, reason)
            elif ctx.alias_row(proposal.get("id")) or ctx.alias_row(proposal.get("entry")):
                raise _Skip("the [别名] row (@a) takes only update / remove / add_item")
            elif op == "append_lines":
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
            elif op == "set_intro":
                _set_intro(ctx, proposal, reason)
            elif op == "move_line":
                _move_line(ctx, proposal, reason)
            elif op == "recategorize":
                _recategorize(ctx, proposal, reason)
            elif op == "split_entry":
                _split_entry(ctx, proposal, reason)
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
    content = "\n".join(_without_handle(line) for line in str(proposal.get("content", "")).splitlines())
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
    existing_labels = _section_labels(ctx, ref, subject_id, section)
    rows = [row.rstrip() for row in content.splitlines() if row.strip()]
    appended = 0
    for row in rows:
        token = _line_dedup_token(row)
        if token and token in existing:
            ctx.record("skipped", cat, key, "append_lines", section, "duplicate line (already present)")
            continue
        kind, payload = classify_line(row, spec, preset=preset)
        # The caller's gate first: a line it holds is held whatever else is wrong with it (an invalid
        # body keeps its label for this, `classify_line`).
        refused = ctx.refused(LineChange("append_lines", cat, section, None, payload))
        if refused:
            ctx.record("skipped", cat, key, "append_lines", section, refused)
            continue
        if kind == "invalid":
            # never silently downgraded to a note: the skip reason feeds the
            # repair round, so the model gets to rewrite the line (plan A1)
            ctx.record("skipped", cat, key, "append_lines", section, str(payload.get("error", "invalid line")))
            continue
        label = str(payload.get("label") or "")
        if label and _match_normalize(label) in existing_labels:
            ctx.record("skipped", cat, key, "append_lines", section,
                       f"标记 [{label}] 在本节已存在——改用 update")
            continue
        if label:
            existing_labels.add(_match_normalize(label))
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
    # ⚠ `parents[0]`, unlike `move_line`'s: an update rewrites the NODE, which
    # every placement shares, so there is nothing to pick between. A line under
    # two sections whose specs disagree can still be refused for "changing
    # kind" -- a spurious refusal, never a wrong write.
    category, key, section = ctx.placement_of(binding.id, parents[0])
    line = _without_handle(str(proposal.get("line", "")).rstrip())
    if not line.strip():
        raise _Skip("empty line", category=category, entry=key, section=section)
    if len(line.splitlines()) != 1:
        # one node is one line: a second line would be read back as a line of its own (a label the gate never
        # saw, a body no check ran on). `splitlines`, as every reader splits an entry, so U+2028, U+0085 and the
        # like count too
        raise _Skip("update rewrites one line; put further lines in append_lines", category=category, entry=key,
                    section=section)
    preset = preset_for_category(category)
    spec = preset.section(section)
    if spec is None:
        raise _Skip(f"section {section!r} not allowed", category=category, entry=key, section=section)
    kind, payload = classify_line(line, spec, preset=preset)
    refused = ctx.refused(LineChange("update", category, section, node.payload, payload))
    if refused:
        raise _Skip(refused, category=category, entry=key, section=section)
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
        # (plan §11.2), tentative ones left out -- so the diff base is the
        # rendered column (`alias_row_delta`). Misheard items are an add-only
        # cache, not reclaimed.
        adds, drops = alias_row_delta(ctx.repo.store.items_of(binding.id, ctx.rev), alias_column)
        for alias in adds:
            alias_ops.append((
                {"op": "add_item", "id": handle, "field": "aliases", "value": alias},
                f"alias added: {alias}",
            ))
        for item in drops:
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


def _alias_row(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    """The `[别名]` row of an entry, addressed by its `@a` handle (cold read
    Q5, owner 2026-10-03/04): `update` rewrites the whole row -- an empty
    body clears it -- `remove` clears it, and `add_item(aliases)` adds one.

    The diff base is the row as the model read it -- the rendered aliases at
    the read revision, which leaves out tentative ones -- so a rewrite or a
    clear never touches what the model could not see, nor what an earlier op
    in this batch adds, and naming a tentative alias is no change. Same split
    and same delta (`alias_row_delta`) as a person editing the row in
    `rendered/` (`edit.py`)."""

    op = str(proposal.get("op"))
    subject_id = str(ctx.alias_row(proposal.get("id")))
    subject = ctx.repo.store.node(subject_id, ctx.rev)
    if subject is None:
        raise _Skip("entry not found at read rev")
    category = str(subject.payload.get("category", ""))
    key = str(subject.payload.get("surface", ""))
    role = preset_for_category(category).label_by_role("aliases")
    section, label = (role[0], role[1].name) if role is not None else ("", "别名")
    where = {"category": category, "entry": key, "section": section}
    if op == "add_item":
        # The row is aliases only: a missing field means aliases.
        if proposal.get("field", "aliases") != "aliases":
            raise _Skip("the [别名] row takes only aliases", **where)
        # One name, as `add_item` is everywhere (owner 2026-10-05): splitting
        # an unbracketed `彼女、お借りします` would register a common word as
        # an alias. A bracketed list form is still read as one name.
        value = _without_handle(str(proposal.get("value", ""))).strip()
        if len(value) >= 2 and value.startswith("「") and value.endswith("」"):
            value = value[1:-1].strip()
        if not re.search(r"\w", value):
            raise _Skip("empty alias", **where)
        new = subject_aliases(ctx.repo.store, subject_id, ctx.rev) + [value]
    else:
        body = ""
        if op == "update":
            line = str(proposal.get("line", "")).strip()
            marker = f"[{label}]"
            if not line.startswith(marker):
                raise _Skip(f"the row must keep its {marker} marker (an empty body clears it)", **where)
            body = _without_handle(line[len(marker):])
        new = _split_names(body)
    adds, drops = alias_row_delta(ctx.repo.store.items_of(subject_id, ctx.rev), new)
    if not adds and not drops:
        raise _Skip("no change", **where)
    handle = ctx.subject_handle(subject_id)
    for alias in adds:
        ctx.op(
            {"op": "add_item", "id": handle, "field": "aliases", "value": alias, "reason": reason},
            op_name=op, label=f"alias added: {alias}", **where,
        )
    for item in drops:
        ctx.op(
            {"op": "remove_item", "item": ctx.bind_item(item), "reason": reason},
            op_name=op, label=f"alias removed: {item.value}", **where,
        )
    ctx.touched_subjects.add(handle)


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
    category = parent.payload.get("category", "") if parent else ""
    entry = parent.payload.get("surface", "") if parent else ""
    section = parents[0].section if parents else ""
    refused = ctx.refused(LineChange("remove", category, section, node.payload, None))
    if refused:
        raise _Skip(refused, category=category, entry=entry, section=section)
    ctx.op(
        {"op": "retire", "id": handle, "reason": reason},
        category=category,
        entry=entry,
        op_name="remove",
        section=section,
        label="line removed",
    )
    if parents:
        parent_handle = next((h for h, b in ctx.bindings.items() if b.kind == "node" and b.id == parents[0].parent_id), parents[0].parent_id)
        ctx.touched_subjects.add(parent_handle)


def _create_entry(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    category = str(proposal.get("category", ""))
    key = str(proposal.get("entry", "")).strip()
    raw_intro = str(proposal.get("intro", ""))
    intro = raw_intro.strip()
    entry_type = str(proposal.get("entry_type", "")).strip()
    aliases = [str(a).strip() for a in (proposal.get("aliases") or []) if str(a).strip()]
    if category not in ctx.allow_categories:
        raise _Skip(f"bad category {category!r}", category=category, entry=key)
    if problem := entry_name_problem(key):
        raise _Skip(f"invalid entry key ({problem})", category=category, entry=key)
    if problem := intro_problem(raw_intro):
        raise _Skip(problem, category=category, entry=key)
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
    if ctx.repo.resolve_in(key, category, ctx.rev) is not None or ctx.created_entry(key, category) is not None:
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
        # Into the batch's ledger too: a later `append_lines` writing its own
        # `[本名]` into this section must be refused here, per op, rather than
        # pass and fail `validate_overlay` as a duplicate label
        _section_labels(ctx, handle, "", section_name).add(_match_normalize(label.name))


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
    ref, cat, key, subject_id = ctx.subject_ref(str(proposal.get("entry", "")), category)
    new_key = str(proposal.get("new_key", "")).strip()
    if ref is None:
        raise _Skip("entry does not exist", category=cat, entry=key)
    ctx.require_entry_level(cat, key, "rename_entry")
    if not new_key or new_key == key:
        raise _Skip("new_key missing or unchanged", category=cat, entry=key)
    if problem := entry_name_problem(new_key):
        raise _Skip(f"invalid new_key ({problem})", category=cat, entry=key)
    # the same namespace rule as create_entry: a rename may collide with a
    # style of the same name, and may NOT be refused for colliding with a
    # proper-noun entry in the other namespace. The entry itself is not a
    # collision: renaming to one of its own aliases, or to a script/width
    # variant of its H1, is exactly what an H1 conflict often asks for (the
    # same rule as the human edit's H1, `edit_subject`). A draft created in
    # this batch has no id yet, so nothing is excluded for it.
    if ctx.repo.resolve_in(new_key, cat, ctx.rev, exclude=subject_id or None) is not None:
        raise _Skip(f"new_key already exists ({new_key!r})", category=cat, entry=key)
    if subject_id:
        ctx.renamed[subject_id] = new_key
    ctx.op(
        {"op": "update", "id": ref, "set": {"payload.surface": new_key}, "reason": reason},
        category=cat, entry=key, op_name="rename_entry", label=f"-> {new_key}",
    )
    ctx.touched_subjects.add(ref)


def intro_problem(intro: str) -> str | None:
    """Why ``intro`` cannot be an entry's intro line, or None when it can.

    The intro is the ONE line between the H1 and the first section. A second
    line there is a parse error (`parse_entry_text`), so an intro holding a
    newline would make every later harvest of that file fail, and the index
    line, which carries the intro, would be cut short. A leading ``#`` would
    read as a heading."""

    if "\n" in intro or "\r" in intro:
        return "intro must be one line"
    if not intro.strip():
        return "intro required"
    if intro.strip().startswith("#"):
        return "intro must not start with #"
    return None


def _set_intro(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    """Replace an entry's intro (owner 2026-10-01). Entry-level like
    `rename_entry`, so a session whose grant does not cover the entry's
    category -- a knowledge update holding a style entry -- is refused."""

    category = str(proposal.get("category", "")) or None
    ref, cat, key, subject_id = ctx.subject_ref(str(proposal.get("entry", "")), category)
    if ref is None:
        raise _Skip("entry does not exist", category=cat, entry=key)
    ctx.require_entry_level(cat, key, "set_intro")
    raw = str(proposal.get("intro", ""))
    if problem := intro_problem(raw):
        raise _Skip(problem, category=cat, entry=key)
    intro = raw.strip()
    subject = ctx.repo.store.node(subject_id, ctx.rev) if subject_id else None
    if subject is not None and str(subject.payload.get("intro") or "").strip() == intro:
        raise _Skip("intro unchanged", category=cat, entry=key)
    ctx.op(
        {"op": "update", "id": ref, "set": {"payload.intro": intro}, "reason": reason},
        category=cat, entry=key, op_name="set_intro", label="intro replaced",
    )
    ctx.touched_subjects.add(ref)


def _section_labels(ctx: _Context, ref: str, subject_id: str, section: str) -> set[str]:
    """The labels a section holds as of this block, normalized the way the
    engine compares them (`_match_normalize`): the store's, plus what earlier
    ops in the block put there. Shared by every op that lands a line in a
    section, because a duplicate label is otherwise caught only by
    `validate_overlay` -- per envelope, so the whole block rolls back (the
    knowledge update rejects the later line instead, under the folded batch's
    message). The identity line `create_entry` and `split_entry` add counts.

    Keyed by the entry's identity, not by how a proposal spelled it: one op
    may name it `@k1` and the next by its key, and two ledgers for one entry
    would each miss the other's label. Only an entry created in this block
    (no id yet) falls back to its ref."""

    account = (subject_id or ref, section)
    labels = ctx.pending_labels.get(account)
    if labels is None:
        labels = set()
        if subject_id:
            for membership in ctx.repo.store.children(subject_id, ctx.rev):
                if membership.section != section:
                    continue
                child = ctx.repo.store.node(membership.child_id, ctx.rev)
                label = str(child.payload.get("label") or "") if child else ""
                if label:
                    labels.add(_match_normalize(label))
        ctx.pending_labels[account] = labels
    return labels


def _move_line(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    """Move one line to another section, of the same entry or -- with
    ``to_entry`` -- of another one.

    Cross-entry moves exist for merges (owner 2026-10-02): before retiring an
    entry into another, the lines worth keeping are moved, not re-typed, so
    the node -- its history, evidence and human-version marks -- travels with
    them. Re-appending the text would mint a new node and cut all three.

    The engine has had `move_membership` since the node model landed, and the
    rendered-markdown harvest already emits it -- it was simply never in the
    model's vocabulary, so the only way an LLM could relocate a line was to
    remove it and re-append the text, losing the node and everything hanging
    off it. That is the shape a "migration" was forced into, and it is why
    every template change wanted bespoke code (plan §0 requirement 11).
    """

    if not str(proposal.get("id", "")).strip() and str(proposal.get("from_section", "")).strip():
        _move_section(ctx, proposal, reason)
        return
    ref = ctx.node_ref(str(proposal.get("id", "")))
    if ref is None:
        raise _Skip("unknown node handle")
    _handle, binding = ref
    node = ctx.repo.store.node(binding.id, ctx.rev)
    if node is None:
        raise _Skip("node not found at read rev")
    if node.kind == "subject":
        raise _Skip("an entry moves with recategorize, not move_line")
    parents = ctx.repo.store.parents(binding.id, ctx.rev)
    if not parents:
        raise _Skip("node has no membership")
    membership = _placement_to_move(ctx, proposal, parents)
    category, key, section = ctx.placement_of(binding.id, membership)
    refused = ctx.refused(LineChange("move_line", category, section, node.payload, node.payload))
    if refused:
        raise _Skip(refused, category=category, entry=key, section=section)
    _move_placement(
        ctx, node, membership,
        to_entry=str(proposal.get("to_entry", "")).strip(),
        target=str(proposal.get("section", "")).strip(),
        reason=reason,
    )


def _move_placement(
    ctx: _Context,
    node: Any,
    membership: Any,
    *,
    to_entry: str,
    target: str,
    reason: str,
    op_name: str = "move_line",
) -> None:
    """Move one placement of a line (the core of `move_line`, the whole-section
    form and `split_entry`)."""

    category, key, current = ctx.placement_of(node.local_id, membership)
    if category and category not in ctx.allow_categories:
        raise _Skip(f"move_line is not allowed on a {category} entry here",
                    category=category, entry=key)
    parent_ref: str | None = None
    to_subject = ""
    parents = ctx.repo.store.parents(node.local_id, ctx.rev)
    if to_entry:
        parent_ref, to_cat, to_key, to_subject = ctx.subject_ref(to_entry, None)
        if parent_ref is None:
            raise _Skip(f"to_entry {to_entry!r} does not exist", category=category, entry=key)
        if to_cat not in ctx.allow_categories:
            raise _Skip(f"move_line into a {to_cat} entry is not allowed here",
                        category=to_cat, entry=to_key)
        if to_subject and to_subject == membership.parent_id:
            parent_ref = None  # the entry it is already under: an ordinary move
        elif to_subject and any(m.parent_id == to_subject for m in parents):
            raise _Skip(
                f"这一行已经挂在「{to_key}」下了，不用再挪；只需处理原条目",
                category=category, entry=key,
            )
        else:
            category, key = to_cat, to_key
            target = target or current
    if not target or (parent_ref is None and target == current):
        raise _Skip("section missing or unchanged", category=category, entry=key)
    preset = preset_for_category(category or "common")
    spec = preset.section(target)
    if spec is None:
        raise _Skip(f"{category} 的模版里没有「{target}」这一节",
                    category=category, entry=key)
    if node.kind not in spec.body_kinds:
        # Refused HERE, per op. Letting it through means `validate_overlay`
        # catches it instead, and that reports at the ENVELOPE level: the whole
        # batch rolls back (the knowledge update rejects the move instead), under
        # a message naming an internal membership id. 归位 out of the staging section is a routine move now
        # that recategorize parks lines there, and the section it goes back
        # into is exactly the one that may not take its shape.
        allowed = "、".join(BODY_KIND_LABELS.get(k, k) for k in spec.body_kinds)
        raise _Skip(
            f"「{target}」节只放{allowed}，这一行是"
            f"{BODY_KIND_LABELS.get(node.kind, node.kind)}"
            "——先把它改写成允许的行体，或换一个收得下它的小节",
            category=category, entry=key,
        )
    # The label must be free in the section it lands in (same reason as the
    # body-kind check above: refuse per op, never roll the block back). A
    # merge is where this bites -- the two entries often share a label.
    label = str(node.payload.get("label") or "")
    if label:
        landing_ref = parent_ref or ctx.subject_handle(membership.parent_id)
        landing_subject = to_subject if parent_ref is not None else membership.parent_id
        taken = _section_labels(ctx, landing_ref, landing_subject, target)
        if _match_normalize(label) in taken:
            raise _Skip(
                f"「{target}」节已有 [{label}] 一行——不要挪这一行，改用 update 改目标那一行",
                category=category, entry=key,
            )
        taken.add(_match_normalize(label))
    ctx.relocated[membership.membership_id] = target
    op: dict[str, Any] = {
        "op": "move_membership",
        "membership": ctx.bind_membership(membership),
        "section": target,
        "reason": reason,
    }
    if parent_ref is not None:
        op["parent"] = parent_ref
        ctx.reparented[membership.membership_id] = (category, key)
        ctx.touched_subjects.add(parent_ref)
    ctx.op(op, category=category, entry=key, op_name=op_name, section=target)
    ctx.touched_subjects.add(ctx.subject_handle(membership.parent_id))


def _section_members(ctx: _Context, subject_id: str, section: str) -> list[Any]:
    """The stored placements under one section of an entry, in order, minus
    any this batch already moved elsewhere."""

    wanted = _match_normalize(section)
    return sorted(
        (
            m for m in ctx.repo.store.children(subject_id, ctx.rev)
            if _match_normalize(m.section) == wanted and m.membership_id not in ctx.reparented
        ),
        key=lambda m: m.order_key,
    )


def _move_members(
    ctx: _Context, members: Sequence[Any], *, to_entry: str, target: str, reason: str, op_name: str
) -> int:
    """Move every placement; one that cannot land is skipped on its own (its
    label is taken in the target, say), the rest still go."""

    moved = 0
    for membership in members:
        node = ctx.repo.store.node(membership.child_id, ctx.rev)
        if node is None:
            continue
        try:
            _move_placement(
                ctx, node, membership, to_entry=to_entry, target=target or membership.section,
                reason=reason, op_name=op_name,
            )
            moved += 1
        except _Skip as exc:
            ctx.record("skipped", exc.category, exc.entry, op_name, exc.section or membership.section,
                       f"{_node_gist(node)}：{exc}")
    return moved


def _node_gist(node: Any) -> str:
    payload = node.payload
    body = str(payload.get("surface") or payload.get("text") or "")
    label = str(payload.get("label") or "")
    gist = f"[{label}] {body}" if label else body
    return gist[:30] + ("…" if len(gist) > 30 else "")


def _move_section(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    """`move_line` with `entry` + `from_section` instead of `id`: every line of
    that section to `to_entry` (into `section`, or the same-named one). For
    "this part is really something else": create the entry, then move the
    section over -- never delete and retype (the nodes would be new)."""

    ref, cat, key, subject_id = ctx.subject_ref(str(proposal.get("entry", "")), None)
    from_section = str(proposal.get("from_section", "")).strip()
    if ref is None or not subject_id:
        raise _Skip("entry does not exist", category=cat, entry=key, section=from_section)
    to_entry = str(proposal.get("to_entry", "")).strip()
    if not to_entry:
        raise _Skip("整节挪动要给 to_entry；同一条目内换节请逐行 move_line",
                    category=cat, entry=key, section=from_section)
    target_ref, _cat, _key, target_subject = ctx.subject_ref(to_entry, None)
    if target_ref is not None and target_subject == subject_id:
        raise _Skip("整节挪动只挪到另一个条目；同一条目内换节请逐行 move_line",
                    category=cat, entry=key, section=from_section)
    members = _section_members(ctx, subject_id, from_section)
    if not members:
        raise _Skip(f"「{from_section}」节不存在或没有行", category=cat, entry=key, section=from_section)
    from .presets import head_section_names

    if members[0].section in head_section_names():
        raise _Skip(f"「{members[0].section}」是条目头，不能整节挪走", category=cat, entry=key, section=from_section)
    moved = _move_members(
        ctx, members, to_entry=to_entry, target=str(proposal.get("section", "")).strip(),
        reason=reason, op_name="move_line",
    )
    if moved == len(members):
        # emptied: drop it from the source's order, as a split does, so the
        # full preview stops offering an empty section to refill
        _trim_section(ctx, ref, subject_id, members[0].section)


def _split_entry(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    """One section of an entry into a sub-entry `母·节` (plan kb-entry-size).

    The engine work is a create (the sub-entry: the parent's category, type
    and visibility -- a sub-entry left `local` would pull a shared section
    out of sharing, line by line, irreversibly), the section's lines moved
    under it unchanged, a `part_of` link, and the parent's `section_order`
    without that section (so its full preview does not offer the section's
    core slots back to be refilled). Every refusal is per op."""

    from .presets import head_section_names
    from .subentries import PART_OF, derived_name, parent_of

    ref, cat, key, subject_id = ctx.subject_ref(str(proposal.get("entry", "")), None)
    section_raw = str(proposal.get("section", "")).strip()
    if ref is None or not subject_id:
        raise _Skip("entry does not exist", category=cat, entry=key, section=section_raw)
    ctx.require_entry_level(cat, key, "split_entry")
    preset = preset_for_category(cat or "common")
    if not preset.splittable:
        raise _Skip(f"{cat} 的条目不能拆分，只能压缩", category=cat, entry=key, section=section_raw)
    store = ctx.repo.store
    if parent_of(store, subject_id, ctx.rev) is not None:
        raise _Skip("子条目不能再拆", category=cat, entry=key, section=section_raw)
    members = _section_members(ctx, subject_id, section_raw)
    if not members:
        raise _Skip(f"「{section_raw}」节不存在或没有行", category=cat, entry=key, section=section_raw)
    section = members[0].section
    if section in head_section_names():
        raise _Skip(f"「{section}」是条目头，不能拆", category=cat, entry=key, section=section)
    child_key = derived_name(ctx.renamed.get(subject_id, key), section)
    if problem := entry_name_problem(child_key):
        raise _Skip(f"子条目名 {child_key!r} 不合法（{problem}）", category=cat, entry=key, section=section)
    if ctx.repo.resolve_in(child_key, cat, ctx.rev) is not None or ctx.created_entry(child_key, cat) is not None:
        raise _Skip(f"已有名为 {child_key!r} 的条目", category=cat, entry=key, section=section)
    raw_intro = str(proposal.get("intro", ""))
    if problem := intro_problem(raw_intro):
        raise _Skip(problem, category=cat, entry=key, section=section)
    if not reason:
        raise _Skip("reason required", category=cat, entry=key, section=section)
    parent = store.node(subject_id, ctx.rev)
    identity = preset.label_by_role("identity")
    head = identity[0] if identity else None
    ctx.new_handles += 1
    handle = f"@new{ctx.new_handles}"
    ctx.created_keys[(cat, child_key)] = handle
    payload = {
        "surface": child_key,
        "intro": raw_intro.strip(),
        "category": cat,
        "entry_type": str(parent.payload.get("entry_type") or "") if cat == "common" else "",
        "updated_date": ctx.repo.today(),
        "section_order": [name for name in (head, section) if name],
        "native_names": [],
        "reading": "",
    }
    ctx.op(
        {"op": "create", "handle": handle, "kind": "subject", "payload": payload,
         "visibility": parent.visibility, "reason": reason},
        category=cat, entry=child_key, op_name="split_entry", section=section, label=f"split from {key}",
    )
    if identity is not None:
        ctx.op({
            "op": "create", "kind": "note", "parent": handle, "section": head, "order_key": 0,
            "payload": {"label": identity[1].name, "text": child_key},
        })
        _section_labels(ctx, handle, "", head).add(_match_normalize(identity[1].name))  # as `_create_entry`
    _move_members(ctx, members, to_entry=child_key, target=section, reason=reason, op_name="split_entry")
    ctx.op({"op": "link", "id": handle, "rel": PART_OF, "target": ref})
    _trim_section(ctx, ref, subject_id, section)
    ctx.split_family.update({subject_id, handle})
    ctx.touched_subjects.add(ref)


def _trim_section(ctx: _Context, ref: str, subject_id: str, section: str) -> None:
    """Drop ``section`` (and every other one this batch took from the same
    entry) from the entry's `section_order`."""

    taken = ctx.trimmed.setdefault(subject_id, set())
    taken.add(section)
    stored = ctx.repo.store.node(subject_id, ctx.rev)
    preset = preset_for_category(str(stored.payload.get("category") or "common"))
    order = [
        name for name in (stored.payload.get("section_order") or preset.section_names()) if name not in taken
    ]
    ctx.op({"op": "update", "id": ref, "set": {"payload.section_order": order}})


def _placement_to_move(ctx: _Context, proposal: Mapping[str, Any], parents: list) -> Any:
    """Which placement of this line the move means.

    A handle names the LINE, and a line can sit under two entries -- both
    renderings then show the SAME handle, because `HandleMap` keys on the node
    id. Taking `parents[0]` moves whichever placement the store happened to
    return first, which may be the one the model was never shown
    (review 2026-09-07, second round).

    So: one placement needs no `entry` and gets none; more than one must be
    disambiguated by `entry`, and anything that does not resolve to exactly
    one of them is refused rather than guessed.
    """

    if len(parents) == 1:
        return parents[0]
    named = str(proposal.get("entry", "")).strip()
    if named:
        _ref, _cat, _key, subject_id = ctx.subject_ref(
            named, str(proposal.get("category", "")) or None
        )
        chosen = [m for m in parents if m.parent_id == subject_id]
        if len(chosen) == 1:
            return chosen[0]
    keys = sorted(
        str((node.payload.get("surface") if node else "") or membership.parent_id)
        for membership in parents
        for node in [ctx.repo.store.node(membership.parent_id, ctx.rev)]
    )
    raise _Skip(
        "这一行同时属于两个条目（" + "、".join(keys) + "）"
        "——用 entry 指明要动哪一个的归属"
    )


def _recategorize(ctx: _Context, proposal: Mapping[str, Any], reason: str) -> None:
    """Move a whole entry to another category, on the record.

    Delete-and-retype was the only way to fix a misfiled entry, and it throws
    away the history, the evidence and the canonical id -- everything that made
    the entry worth more than the text in it (plan §0 requirement 7).

    The reason is required here rather than encouraged: a move with no reason
    is indistinguishable from a mistake, and the next reader has to re-litigate
    it from scratch. It lands in the entry's metadata section and renders in the
    maintenance preview only (§0 requirement 13). Whether the entry FITS the
    target template is `apply._check_recategorized`'s job -- that check has to
    cover a bare `update` of the category too, so it cannot live here.
    """

    category = str(proposal.get("category", "")) or None
    ref, cat, key, subject_id = ctx.subject_ref(str(proposal.get("entry", "")), category)
    target = str(proposal.get("to_category", "")).strip()
    if ref is None:
        raise _Skip("entry does not exist", category=cat, entry=key)
    ctx.require_entry_level(cat, key, "recategorize")
    from .subentries import children_of, parent_of

    if (
        subject_id in ctx.split_family
        or ref in ctx.split_family  # a sub-entry created in this very batch
        or (subject_id and (
            parent_of(ctx.repo.store, subject_id, ctx.rev) or children_of(ctx.repo.store, subject_id, ctx.rev)
        ))
    ):
        raise _Skip("带子条目的条目、以及子条目本身，不能改类别", category=cat, entry=key)
    if not target or target == cat:
        raise _Skip("to_category missing or unchanged", category=cat, entry=key)
    if target not in ctx.allow_categories:
        raise _Skip(f"recategorize into {target} is not allowed here", category=cat, entry=key)
    if not reason:
        raise _Skip("recategorize needs a reason", category=cat, entry=key)
    # Key uniqueness is per NAMESPACE, and a namespace can span categories: the
    # matchable ones share one, so a move between two of them does not change
    # the namespace at all. Resolving the key there finds THIS entry, which is
    # not a collision — the check has to exclude self, or every move between
    # matchable categories is refused as a duplicate of itself.
    clash = ctx.repo.resolve_in(key, target, ctx.rev)
    if clash is not None and clash.subject_id != subject_id:
        raise _Skip(f"{key!r} already exists in {target}", category=cat, entry=key)
    if subject_id:
        ctx.recategorized[subject_id] = target
    ctx.op(
        {
            "op": "update",
            "id": ref,
            "set": {
                "payload.category": target,
                f"payload.{RECATEGORIZED_REASON_FIELD}": reason,
            },
            "reason": reason,
        },
        category=cat, entry=key, op_name="recategorize", label=f"-> {target}",
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


def _is_date_stamp(op: Mapping[str, Any]) -> bool:
    """The `updated_date` op `translate_model_proposals` adds per touched entry."""

    meta = op.get("_meta")
    return (
        isinstance(meta, Mapping) and bool(meta.get("bookkeeping"))
        and op.get("op") == "update" and set(op.get("set") or ()) == {"payload.updated_date"}
    )


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
    line_gate: LineGate | None = None,
    invalid_ops: str = "rollback",
) -> ProposalReport:
    """REST front end: parse the model block, translate, apply, refresh the cache.

    ``proposal_text_hash`` (sha256 of the raw model output) is recorded on the
    revision so the chunk ledger can recover a commit that landed before the
    ledger row was written. ``invalid_ops``: as `apply_envelope`'s.
    """

    return apply_proposals(
        parse_model_proposals(text), repo=repo, allow_categories=allow_categories, task_id=task_id,
        knowledge_read_rev=knowledge_read_rev, handles=handles, assignment_id=assignment_id,
        context_epoch=context_epoch, proposal_text_hash=proposal_text_hash, line_gate=line_gate,
        invalid_ops=invalid_ops,
    )


def apply_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    repo: KnowledgeRepo,
    allow_categories: Sequence[str] = MATCHABLE_CATEGORIES,
    task_id: str,
    knowledge_read_rev: int,
    handles: HandleMap | None = None,
    assignment_id: str = "",
    context_epoch: int = 0,
    proposal_text_hash: str = "",
    line_gate: LineGate | None = None,
    invalid_ops: str = "rollback",
) -> ProposalReport:
    """`apply_model_proposals` on proposals already parsed -- for a caller that
    screens them first (the base-style migration admits only its own op)."""

    ops, report, draft_handles, bindings = translate_model_proposals(
        proposals,
        repo=repo,
        knowledge_read_rev=knowledge_read_rev,
        bindings=[Binding(**b) for b in (handles.bindings() if handles else [])],
        allow_categories=allow_categories,
        line_gate=line_gate,
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
        repo.store, envelope, note=f"proposal_text:{proposal_text_hash}" if proposal_text_hash else "",
        invalid_ops=invalid_ops,
        # the `updated_date` stamps: with every real change rejected, they alone write no revision
        bookkeeping_ops=[index for index, op in enumerate(ops) if _is_date_stamp(op)],
    )
    report.rev = result.rev
    report.rolled_back = result.rolled_back
    report.rollback_reason = result.rollback_reason
    report.conflicts = [c.__dict__ for c in result.conflicts]
    report.parked = [
        {"entry": _subject_key(repo, subject_id), "from_section": section, "line": gist}
        for subject_id, section, gist in result.parked
    ]
    _warn_parked(report.parked)
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
            report.skipped.append(replace(rec, status="skipped", reason=CAS_DROPPED_REASON))
    report.applied_proposals = sorted(applied_proposals)
    if not result.rolled_back:
        repo.refresh_rendered()
    return report
