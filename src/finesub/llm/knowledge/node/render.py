"""The three render projections (plan §2.3).

* ``legacy``: reproduce the imported markdown byte-for-byte from the
  ``migration_aux`` sidecar (shadow phase only; used by parity).
* ``human``: formatted from payloads, no handles; the read-only cache.
* ``prompt``: ``human`` plus per-call short handles (``@k12`` …) so a model
  can reference nodes without ever seeing a ulid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .model import (
    METADATA_SECTION,
    RECATEGORIZED_LABEL,
    RECATEGORIZED_REASON_FIELD,
    ItemVersion,
    MembershipVersion,
    NodeVersion,
    UPDATED_DATE_LABEL,
)
from ...output_tags import neutralize_closing_tags
from ..base import join_names
from .presets import MATCHABLE_CATEGORIES, LabelSpec, preset_for_category
from .store import KnowledgeStore

if TYPE_CHECKING:
    from .human_marks import LineMarks

MODES = ("legacy", "human", "prompt")


@dataclass
class SectionView:
    name: str
    entries: list[tuple[MembershipVersion, NodeVersion]] = field(default_factory=list)


@dataclass
class SubjectTree:
    subject: NodeVersion
    sections: list[SectionView]
    items: list[ItemVersion]
    rev: int


@dataclass
class HandleMap:
    """Per-call handle table (plan §2.3/§2.5): handle -> (id, expected_valid_from_rev)."""

    nodes: dict[str, tuple[str, int]] = field(default_factory=dict)
    items: dict[str, tuple[str, int]] = field(default_factory=dict)
    memberships: dict[str, tuple[str, int]] = field(default_factory=dict)
    #: `@a` -> the subject whose `[别名]` row it marks. The row has no node of
    #: its own (it renders from the subject's alias items), so it gets its own
    #: prefix: sharing `@k` numbering would shift every later line's handle.
    aliases: dict[str, tuple[str, int]] = field(default_factory=dict)
    _reverse: dict[str, str] = field(default_factory=dict)

    def node_handle(self, node: NodeVersion) -> str:
        return self._assign("k", node.local_id, node.valid_from_rev, self.nodes)

    def item_handle(self, item: ItemVersion) -> str:
        return self._assign("i", item.item_id, item.valid_from_rev, self.items)

    def membership_handle(self, membership: MembershipVersion) -> str:
        return self._assign("m", membership.membership_id, membership.valid_from_rev, self.memberships)

    def alias_row_handle(self, subject: NodeVersion) -> str:
        return self._assign("a", subject.local_id, subject.valid_from_rev, self.aliases)

    def _assign(self, prefix: str, ident: str, from_rev: int, table: dict[str, tuple[str, int]]) -> str:
        key = f"{prefix}:{ident}"
        handle = self._reverse.get(key)
        if handle is None:
            # Seeded tables may have sparse numbering: probe past collisions.
            counter = len(table) + 1
            handle = f"@{prefix}{counter}"
            while handle in table:
                counter += 1
                handle = f"@{prefix}{counter}"
            table[handle] = (ident, from_rev)
            self._reverse[key] = handle
        return handle

    def seed(self, bindings: Iterable[Mapping[str, Any]]) -> None:
        """Replay a previous call's handle table.

        Agent tool sessions: the prompt's handles arrive via the task
        manifest, so the session's own reads continue numbering past them and
        ``kb_validate`` resolves both spaces (plan §6.5, 4c)."""

        tables = {
            "node": (self.nodes, "k"), "item": (self.items, "i"),
            "membership": (self.memberships, "m"), "aliases": (self.aliases, "a"),
        }
        for binding in bindings:
            table_prefix = tables.get(str(binding.get("kind")))
            if table_prefix is None:
                # A kind added to `bindings()` but not here would drop its
                # handles from every agent session without a sound.
                raise ValueError(f"unknown handle binding kind {binding.get('kind')!r}")
            table, prefix = table_prefix
            handle = str(binding.get("handle") or "")
            ident = str(binding.get("id") or "")
            if not handle or not ident or handle in table:
                continue
            table[handle] = (ident, int(binding.get("expected_valid_from_rev") or 0))
            self._reverse[f"{prefix}:{ident}"] = handle

    def bindings(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for table, kind in (
            (self.nodes, "node"), (self.items, "item"), (self.memberships, "membership"),
            (self.aliases, "aliases"),
        ):
            for handle, (ident, from_rev) in table.items():
                out.append({"handle": handle, "kind": kind, "id": ident, "expected_valid_from_rev": from_rev})
        return out


def load_subject_tree(store: KnowledgeStore, subject_id: str, rev: int | None = None) -> SubjectTree:
    at = store.current_rev() if rev is None else rev
    subject = store.node(subject_id, at)
    if subject is None or subject.kind != "subject":
        raise KeyError(subject_id)
    sections: dict[str, SectionView] = {}
    order: list[str] = list(subject.payload.get("section_order", []))
    for name in order:
        sections[name] = SectionView(name)
    for membership in store.children(subject_id, at):
        child = store.node(membership.child_id, at)
        if child is None:
            continue
        if child.maturity == "tentative":
            continue  # shadow-only until corroborated (plan §11.5): no projection
        view = sections.get(membership.section)
        if view is None:
            view = SectionView(membership.section)
            sections[membership.section] = view
        view.entries.append((membership, child))
    return SubjectTree(subject=subject, sections=list(sections.values()), items=store.items_of(subject_id, at), rev=at)


# ---- line formatting -----------------------------------------------------------


def format_line(node: NodeVersion, aliases: Iterable[str] | None = None) -> str:
    """One stored line in the v3 grammar: an optional ``[标记]`` prefix plus a
    body. The label is a cross-kind payload key, so prefixing happens once
    here rather than per kind."""

    body = format_body(node, aliases)
    label = str(node.payload.get("label") or "").strip()
    if not label or node.kind == "subject":
        return body
    return f"[{label}] {body}" if body else f"[{label}]"


def format_body(node: NodeVersion, aliases: Iterable[str] | None = None) -> str:
    payload = node.payload
    kind = node.kind
    if kind == "term":
        # Fixed four columns 源|中|别名|desc (kb-followups plan A1): the alias
        # column renders FROM items and stays present even when empty, so an
        # `update` naturally copies it; desc is LAST so it may contain pipes.
        # Payload alias_text/reading are unrendered archive residue; misheard
        # items are matcher-only and never rendered.
        alias_column = join_names(dict.fromkeys(aliases or ()))
        return "|".join(
            (payload.get("surface", ""), payload.get("zh", ""), alias_column, payload.get("desc", ""))
        )
    # Legacy kinds only exist between Phase A and Phase B of the re-import;
    # they keep their verbatim separator so that stretch stays byte-parity.
    if kind == "fact":
        sep = payload.get("sep", ": ")
        return f"{payload.get('field', '')}{sep}{payload.get('value', '')}"
    if kind == "event":
        sep = payload.get("sep", ": ")
        return f"{payload.get('occurred_at', '')}{sep}{payload.get('description', '')}"
    if kind == "relation":
        sep = payload.get("sep", " | ")
        return f"{payload.get('target', '')}{sep}{payload.get('description', '')}"
    if kind == "note":
        return str(payload.get("text", ""))
    if kind == "subject":
        return f"# {payload.get('surface', '')}"
    raise ValueError(kind)


def node_aliases(store: KnowledgeStore, node: NodeVersion, rev: int | None = None) -> list[str]:
    """Alias item values for one node's rendered alias column (terms only —
    other kinds render no alias column and skip the items query)."""

    if node.kind != "term":
        return []
    return [
        item.value
        for item in store.items_of(node.local_id, rev)
        if item.field == "aliases" and item.maturity != "tentative"
    ]


def _legacy_line(store: KnowledgeStore, node: NodeVersion) -> str:
    aux = store.migration_aux(node.local_id)
    if aux is None:
        return format_line(node)
    return aux.legacy_raw


# ---- full-preview scaffolding ----------------------------------------------------

PREVIEWS: tuple[str, ...] = ("partial", "full")

_TERM_SHAPE = "源语言|中文定名|别名|一句话描述——别名在第三列，无则留空；描述在最后，可含竖线"


def line_shape_hint(spec) -> str:  # type: ignore[no-untyped-def]
    """The 「行式」 line, generated from ``body_kinds`` — it is the answer to
    "must every line carry a label?". A column of empty slots otherwise reads
    as "this section only takes labelled lines" (scaffolding's built-in side
    effect), so the text has to say otherwise."""

    kinds = tuple(spec.body_kinds)
    if kinds == ("term",):
        body = f"术语行 {_TERM_SHAPE}"
    elif "term" in kinds:
        body = f"自由句，或术语行 {_TERM_SHAPE}"
    else:
        body = "一句话内容"
    return f"行式：可选 [标记] + {body}。标记不是必须的，无标记行同样合法。"


def entry_prompt_tokens(store: KnowledgeStore, subject_id: str, rev: int | None = None) -> int:
    """What this entry costs a prompt, by the dependency-free upper bound.

    `HeuristicTokenCounter` on purpose: this is called from apply validation
    and from prompt assembly, so it may not reach the network (`countTokens`)
    or spawn the local binary, and over-estimating is the safe direction for a
    budget.
    """

    from ...token_budget import HeuristicTokenCounter

    return HeuristicTokenCounter().count_text(
        render_subject(store, subject_id, rev=rev, mode="prompt")
    )


def render_category_line(categories: Sequence[str]) -> str:
    """The one-sentence roster the structure fragment opens with.

    It read "两个类别：`streamer`（主播）和 `common`（主播之外的一切）" -- a
    hand-written copy of the category list, in the same prompt as the generated
    section table right below it. Adding a category left the model told, in
    consecutive paragraphs, both that it exists and that it does not."""

    named = "、".join(
        f"`{category}`（{preset_for_category(category).blurb or category}）"
        for category in categories
    )
    return f"本库有 {len(tuple(categories))} 个类别：{named}。"


def render_writable_categories(categories: Sequence[str]) -> str:
    """The `category` enum of the ops contract, from the caller's own grant.

    Which categories a session may WRITE is narrower than which ones exist --
    the knowledge-update round gets the matchable ones, the repair sessions get
    all of them -- so this takes the same tuple that is handed to
    `translate_model_proposals`, and the prompt cannot drift from the gate."""

    return "|".join(categories)


def render_structure_spec(categories: Sequence[str] | None = None) -> str:
    """The section/label table the prompts describe, GENERATED from the
    presets (plan §3 single source of truth).

    Hand-writing it is how `fragment_knowledge_structure_v1` came to teach a
    three-segment term line while the parser had required four for a whole
    release — two copies of the same rule, drifting."""

    out: list[str] = []
    for category in categories or MATCHABLE_CATEGORIES:
        preset = preset_for_category(category)
        title = f"{category}（{preset.blurb or preset.name}）"
        rule = "仅允许下列小节" if preset.strict_sections else "档案固定，其余小节自由命名"
        out.append(f"**{title}**（{rule}）")
        specs = list(preset.sections)
        if preset.default_section is not None:
            specs.append(preset.default_section)
        for spec in specs:
            name = spec.name or "（自由命名的分类节）"
            body = "术语行" if spec.body_kinds == ("term",) else (
                "自由句或术语行" if "term" in spec.body_kinds else "自由句"
            )
            out.append(f"- `{name}`：{spec.purpose} 行体是{body}。")
            if spec.exclude:
                out.append(f"  - 不收：{spec.exclude}")
            core = [label for label in spec.labels if label.core]
            if core:
                out.append("  - 标记：" + "；".join(f"`[{l.name}]` {l.note}" for l in core))
        out.append("")
    return "\n".join(out).rstrip()


def section_comment(spec, *, preamble: str = "", for_model: bool = False) -> list[str]:  # type: ignore[no-untyped-def]
    """Full-preview guidance block for one section: purpose, exclusion
    discipline, the line-shape hint and every registered label's note. It is
    an HTML comment so the harvest can drop it without it ever becoming a
    content line.

    ``preamble`` scopes the block when it is not describing one named section --
    see ``default_section_comment``. ``for_model`` leaves out the labels only
    people write (``LabelSpec.model_hidden``)."""

    body: list[str] = []
    if preamble:
        body.append(preamble)
    if spec.purpose:
        body.append(f"用途：{spec.purpose}")
    if spec.exclude:
        body.append(f"不收：{spec.exclude}")
    body.append(line_shape_hint(spec))
    if spec.max_lines is not None or spec.max_body_chars is not None:
        caps = []
        if spec.max_lines is not None:
            caps.append(f"最多 {spec.max_lines} 行")
        if spec.max_body_chars is not None:
            caps.append(f"每行 ≤ {spec.max_body_chars} 字（含 [标记]）")
        body.append(
            "上限：" + "、".join(caps)
            + "。满了还要加，就得先退掉一条、或把两条并成一条——写入时会校验，超了整批拒绝。"
        )
    core = [label for label in spec.labels if label.core]
    extra = [label for label in spec.labels if not label.core and not (for_model and label.model_hidden)]
    if core:
        body.append("标记（缺了完整预览会留一个空槽）：")
        body.extend(_label_guide(label) for label in core)
    if extra:
        # With their notes, not just their names. A registered non-core label
        # is one the model is *allowed* to use but never reminded to -- so the
        # note is the only thing that says what belongs there and in what shape,
        # and a bare name list left it guessing. (`[当前版本]` in the common
        # preset carried a note nobody could see for exactly this reason.)
        body.append("登记的其他标记（可用，不提醒）：")
        body.extend(_label_guide(label) for label in extra)
    if not body:
        return []
    lines = ["<!-- " + body[0]]
    lines.extend("     " + row for row in body[1:])
    lines[-1] += " -->"
    return lines


def _label_guide(label: LabelSpec) -> str:
    """One label's line in a section's guidance: its note, and the bodies it takes when they are fixed (the same
    list the write side checks, `LabelSpec.values`)."""

    values = f"（只能写 {' / '.join(label.values)} 之一）" if label.values else ""
    return f"  [{label.name}] {label.note}{values}".rstrip()


_FREE_SECTION_PREAMBLE = (
    "以下自由节（节名自取）统一按这条规则收录——每节不再重复："
)


def default_section_comment(spec, *, for_model: bool = False) -> list[str]:  # type: ignore[no-untyped-def]
    """The free-section rule, rendered ONCE for the whole subject.

    A preset with ``strict_sections = false`` resolves every unnamed section to
    the same ``default_section`` spec, so rendering the block per section
    printed one identical paragraph under 「角色」, 「地区」, 「组织」... The text
    was never about a section anyway: it states what any free section may hold.
    Saying it once, up front, is both shorter and the truer placement -- a named
    section keeps its own block, because there the text really is section-local.
    """

    return section_comment(spec, preamble=_FREE_SECTION_PREAMBLE, for_model=for_model)


# ---- subject rendering -----------------------------------------------------------


def render_subject(
    store: KnowledgeStore,
    subject_id: str,
    *,
    rev: int | None = None,
    mode: str = "human",
    handles: HandleMap | None = None,
    sections: Iterable[str] | None = None,
    preview: str = "partial",
    marks: "LineMarks | None" = None,
    alias_row_handle: bool = True,
) -> str:
    """Render one subject as markdown in the requested projection.

    ``mode`` picks the *face* (``human`` bullets, ``prompt`` handles, frozen
    ``legacy`` replay); ``preview`` picks *how much* (plan §11):

    * ``full`` — users and the knowledge-update task: every section rendered
      even when empty, guidance comments, and an empty slot per core label.
    * ``partial`` (default) — every other LLM task: no comments, no empty
      sections, no empty slots.

    ``sections`` restricts the output to the named sections (used by the
    REST by-hit pruning and the agent ``kb_read(section=…)``); the H1/intro
    and the metadata section are always included.

    ``marks`` (human mode only) prints the latest human-endorsed value under
    each line the model moved away from it, and a same-shaped comment for a
    retired one in its old section (``human_marks.py``). Only ``entry_text``
    passes it: the edit round-trip diffs against the unmarked rendering, and
    no prompt projection may ever carry a human version.

    ``alias_row_handle`` -- with ``handles``, the `[别名]` slot carries an
    `@a` handle the model may `update` / `remove` (cold read Q5). The share
    conflict repair turns it off: its gate refuses entry aliases.
    """

    if mode not in MODES:
        raise ValueError(mode)
    if marks is not None and mode != "human":
        raise ValueError("human-version marks belong to the human projection only")
    if marks is not None:
        from .human_marks import marker_comment  # human_marks renders lines with this module
    if preview not in PREVIEWS:
        raise ValueError(preview)
    full = preview == "full" and mode != "legacy"
    tree = load_subject_tree(store, subject_id, rev)
    subject = tree.subject
    payload = subject.payload
    aux = store.migration_aux(subject.local_id) if mode == "legacy" else None
    layout = (aux.layout if aux else {}) or {}
    blank_after: dict[str, bool] = layout.get("blank_after_heading", {})
    trailing_blank: dict[str, int] = layout.get("trailing_blank", {})
    preset = preset_for_category(payload.get("category") or "common")
    wanted = set(sections) if sections is not None else None

    out: list[str] = []
    out.append(f"# {payload.get('surface', '')}")
    if mode == "prompt" and handles is not None:
        out[-1] += f"  <!-- {handles.node_handle(subject)} -->"
    if payload.get("intro") and not (mode == "legacy" and layout.get("intro_from_index")):
        out.append(str(payload["intro"]))
    if marks is not None and marks.head is not None:
        # under the intro (H1 when there is none): still before the first
        # section, where the parser only keeps the one intro line
        out.append(marker_comment(marks.head))
    relation = _relation_lines(store, subject.local_id, tree.rev) if mode != "legacy" else []
    for line in relation:
        # Prompt: a plain line. Human: a whole-line comment -- the harvest
        # drops those, and a second plain line between the H1 and the first
        # section is a format error that would fail every edit of the file.
        out.append(f"<!-- {line} -->" if mode == "human" else line)
    is_subentry = any(line.startswith(SUBENTRY_OF) for line in relation)

    ordered = _ordered_sections(tree, preset)
    # One block for every free section, instead of the same paragraph repeated
    # under each of them. Only when a free section is actually being rendered:
    # a `sections=` filter that keeps only named ones has nothing to scope it to.
    free_spec = None if preset.strict_sections else preset.default_section
    if (
        full
        and free_spec is not None
        and any(
            (wanted is None or view.name in wanted)
            and preset.section(view.name) is free_spec
            for view in ordered
        )
    ):
        out.append("")
        out.extend(default_section_comment(free_spec, for_model=mode == "prompt"))
    rendered_gone: set[str] = set()
    for view in ordered:
        if wanted is not None and view.name not in wanted:
            continue
        spec = preset.section(view.name)
        used_labels = {line_label(child.payload) for _, child in view.entries}
        slots = (
            missing_core_slots(spec, used_labels, identity_only=is_subentry)
            if (full and spec is not None)
            else []
        )
        if not view.entries and not full and mode != "legacy":
            continue  # partial preview: an empty section is pure noise
        out.append("")
        out.append(f"## {view.name}")
        if mode == "legacy":
            if blank_after.get(view.name, False):
                out.append("")
        elif view.entries or full:
            # human/prompt: one canonical layout -- blank line after heading only
            # when the section has something under it.
            out.append("")
        if full and spec is not None and spec is not free_spec:
            out.extend(section_comment(spec, for_model=mode == "prompt"))
        for membership, child in view.entries:
            if mode == "legacy":
                child_aux = store.migration_aux(child.local_id)
                for _ in range(int((child_aux.layout if child_aux else {}).get("blank_before", 0))):
                    out.append("")
            line = (
                _legacy_line(store, child)
                if mode == "legacy"
                else format_line(child, aliases=node_aliases(store, child, tree.rev))
            )
            if mode == "human":
                # rendered/ is edited in markdown editors too, some of which
                # edit the RENDERED view — bare consecutive lines merge into
                # one paragraph there. The bullet keeps one entry per line in
                # every viewer; the edit round-trip strips it back off. The
                # prompt projection stays bare (models read text, and the
                # prefix would only cost tokens and a PROMPT_VERSION bump).
                #
                # ⚠ A line that ALREADY starts with `- ` keeps its own instead
                # of getting a second: a real archive whose sections used
                # markdown lists rendered 291 lines as `- - …` across 23 of 27
                # entries (2026-09-01). The cost is that the harvest's
                # `_strip_bullet` then eats the user's own dash, so a leading
                # `- ` is ABSORBED by the bullet convention and does not
                # survive an edit round-trip — accepted deliberately (owner
                # 2026-09-01) over the alternative of escaping it, which would
                # put an escape character in front of a human reader.
                line = line if line.startswith("- ") else f"- {line}"
            if mode == "prompt" and handles is not None:
                # membership handles (@m) are not rendered: no op the model can
                # emit consumes them today, so inline they are only noise tokens
                # (plan §2.3). HandleMap still hands them out for edit/apply.
                line += f"  <!-- {handles.node_handle(child)} -->"
            out.append(line)
            if marks is not None and child.local_id in marks.live:
                out.append(marker_comment(marks.live[child.local_id]))
        if marks is not None:
            out.extend(marker_comment(mark) for mark in marks.gone.get(view.name, ()))
            rendered_gone.add(view.name)
        for label in slots:
            # The alias label has no node of its own: it IS the subject's alias
            # column, rendered from items and diffed back into them (plan §4).
            body = ""
            if label.role == "aliases":
                body = join_names(subject_aliases(store, subject.local_id, tree.rev))
            # Otherwise an empty slot is the whole reminder — no signal, no
            # ledger, nothing stored (owner decision, plan §0.1 item 17).
            line = f"[{label.name}] {body}".rstrip()
            if label.role == "aliases" and mode == "prompt" and handles is not None and alias_row_handle:
                line += f"  <!-- {handles.alias_row_handle(subject)} -->"
            out.append(f"- {line}" if mode == "human" else line)

        if mode == "legacy":
            # The next heading brings one blank with it; anything beyond that
            # was the user's own spacing (see ParsedSection.trailing_blank).
            for _ in range(max(0, int(trailing_blank.get(view.name, 1)) - 1)):
                out.append("")

    if marks is not None and wanted is None:
        # A retired line whose section is no longer rendered at all: its
        # comment names the section and sits at the end of the body, where
        # the harvest drops it like any other whole-line comment.
        leftovers = [
            mark for name, gone in marks.gone.items() if name not in rendered_gone for mark in gone
        ]
        if leftovers:
            out.append("")
            out.extend(marker_comment(mark, name_section=True) for mark in leftovers)
    # Why this entry sits in the category it does, when it was moved. Full
    # preview only: the writable faces (a person, and the knowledge-update
    # session) are the ones that may move an entry again, and the same rule
    # that keeps evidence out of the injection surface keeps this out of it
    # -- a correction round has no use for the move's history and would pay
    # tokens for it (plan §0 requirement 13).
    reason = str(payload.get(RECATEGORIZED_REASON_FIELD, "")).strip()
    # No prompt face carries the update date: nothing in it informs a
    # correction or an update, and the model may not edit it (cold read P057
    # for the read-only face, round 4 D4 for the writable one). The metadata
    # section reaches a prompt only to carry a recategorization reason.
    if mode == "prompt":
        metadata = [f"{RECATEGORIZED_LABEL}: {reason}"] if full and reason else []
    elif mode == "legacy" and layout.get("has_metadata") is False:
        metadata = []
    else:
        metadata = [f"{UPDATED_DATE_LABEL}: {payload.get('updated_date', '')}"]
        if full and reason:
            metadata.append(f"{RECATEGORIZED_LABEL}: {reason}")
    if metadata:
        out.append("")
        out.append(f"## {METADATA_SECTION}")
        if mode != "legacy" or blank_after.get(METADATA_SECTION, False):
            out.append("")
        out.extend(metadata)
    text = "\n".join(out)
    if mode == "prompt":
        # Every entry a model reads comes through here, and an entry may be
        # someone else's text (shared, ingested from material or the web).
        text = neutralize_closing_tags(text)
    if mode == "legacy" and not layout.get("trailing_newline", True):
        return text
    return text + "\n"


def _ordered_sections(tree: SubjectTree, preset) -> list[SectionView]:  # type: ignore[no-untyped-def]
    """Preset order first (empty slots kept), then any free sections in stored order."""

    by_name = {view.name: view for view in tree.sections}
    return [
        by_name.get(name) or SectionView(name)
        for name in ordered_section_names(
            tree.subject.payload, preset, [view.name for view in tree.sections]
        )
    ]


def ordered_section_names(
    payload: Mapping[str, Any], preset, live_section_names: Iterable[str]  # type: ignore[no-untyped-def]
) -> list[str]:
    """Which sections a subject renders, in order: its explicit
    ``section_order`` (else the preset's), then any other section a rendered
    line lives in. One rule for the renderer and for the human-version
    marks, which answer "is the alias slot shown" without rendering."""

    head = list(payload.get("section_order") or preset.section_names())
    seen = set(head)
    return head + [name for name in dict.fromkeys(live_section_names) if name not in seen]


def line_label(payload: Mapping[str, Any]) -> str:
    """The label a line occupies in its section. During the Phase A→B window
    a legacy fact's field IS its label."""

    return str(payload.get("label") or payload.get("field") or "")


def missing_core_slots(spec, used_labels: Iterable[str], *, identity_only: bool = False) -> list:  # type: ignore[no-untyped-def]
    """The core labels of a section no line carries yet: the full preview
    renders an empty slot for each (the alias one shows the entry aliases).

    ``identity_only`` -- a sub-entry (`subentries.py`): only its identity slot
    counts. Its scope is "that section of its parent", said by its name and
    its 属于 line; empty alias / scope slots on every sub-entry would only
    invite proposals to fill them (plan kb-entry-size)."""

    used = set(used_labels)
    return [
        label for label in spec.core_labels()
        if label.name not in used and (not identity_only or label.role == "identity")
    ]


SUBENTRIES_LINE = "子条目："
SUBENTRY_OF = "属于："


def _relation_lines(store: KnowledgeStore, subject_id: str, rev: int) -> list[str]:
    """The `part_of` relation, said on both ends (`subentries.py`)."""

    from .subentries import children_of, parent_of

    lines = []
    parent = parent_of(store, subject_id, rev)
    if parent is not None:
        node = store.node(parent, rev)
        lines.append(SUBENTRY_OF + str(node.payload.get("surface") or "") if node is not None else "")
    children = [store.node(child, rev) for child in children_of(store, subject_id, rev)]
    names = [str(child.payload.get("surface") or "") for child in children if child is not None]
    if names:
        lines.append(SUBENTRIES_LINE + "、".join(names))
    return [line for line in lines if line]


# ---- index rendering ----------------------------------------------------------------


def subject_aliases(store: KnowledgeStore, subject_id: str, rev: int | None = None) -> list[str]:
    """Alias values from items — the single home for aliases (plan §2.1).
    Tentative items are shadow-only (plan §11.5): this list feeds the
    model-facing index/resolve surfaces, so they never appear here."""

    return list(
        dict.fromkeys(
            item.value
            for item in store.items_of(subject_id, rev)
            if item.field == "aliases" and item.maturity != "tentative"
        )
    )


def render_index_line(subject: NodeVersion, aliases: Iterable[str]) -> str:
    payload = subject.payload
    key = payload.get("surface", "")
    entry_type = payload.get("entry_type", "")
    head = f"{key} [{entry_type}]" if entry_type else key
    native = join_names(payload.get("native_names", []))
    return f"- {head} | {native} | {join_names(aliases)} | {payload.get('intro', '')}"


def render_index(store: KnowledgeStore, category: str, *, rev: int | None = None, header: str = "") -> str:
    from .subentries import children_of, parent_of

    lines = [header.rstrip("\n")] if header else []
    listed = [
        subject for subject in store.subjects(rev)
        if subject.payload.get("category") == category
        # tentative: shadow-only (plan §11.5), not in the model-facing index
        and subject.maturity != "tentative"
    ]
    by_id = {subject.local_id: subject for subject in listed}
    for subject in listed:
        if parent_of(store, subject.local_id, rev) in by_id:
            continue  # listed right after its parent
        lines.append(render_index_line(subject, subject_aliases(store, subject.local_id, rev)))
        for child_id in children_of(store, subject.local_id, rev):
            child = by_id.get(child_id)
            if child is not None:
                lines.append(render_index_line(child, subject_aliases(store, child_id, rev)))
    return "\n".join(lines) + "\n"
