"""The three render projections (plan §2.3).

* ``legacy``: reproduce the imported markdown byte-for-byte from the
  ``migration_aux`` sidecar (shadow phase only; used by parity).
* ``human``: formatted from payloads, no handles; the read-only cache.
* ``prompt``: ``human`` plus per-call short handles (``@k12`` …) so a model
  can reference nodes without ever seeing a ulid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .model import (
    MATCHABLE_CATEGORIES,
    METADATA_SECTION,
    ItemVersion,
    MembershipVersion,
    NodeVersion,
    UPDATED_DATE_LABEL,
)
from .presets import preset_for_category
from .store import KnowledgeStore

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
    _reverse: dict[str, str] = field(default_factory=dict)

    def node_handle(self, node: NodeVersion) -> str:
        return self._assign("k", node.local_id, node.valid_from_rev, self.nodes)

    def item_handle(self, item: ItemVersion) -> str:
        return self._assign("i", item.item_id, item.valid_from_rev, self.items)

    def membership_handle(self, membership: MembershipVersion) -> str:
        return self._assign("m", membership.membership_id, membership.valid_from_rev, self.memberships)

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

        tables = {"node": (self.nodes, "k"), "item": (self.items, "i"), "membership": (self.memberships, "m")}
        for binding in bindings:
            table_prefix = tables.get(str(binding.get("kind")))
            if table_prefix is None:
                continue
            table, prefix = table_prefix
            handle = str(binding.get("handle") or "")
            ident = str(binding.get("id") or "")
            if not handle or not ident or handle in table:
                continue
            table[handle] = (ident, int(binding.get("expected_valid_from_rev") or 0))
            self._reverse[f"{prefix}:{ident}"] = handle

    def bindings(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for table, kind in ((self.nodes, "node"), (self.items, "item"), (self.memberships, "membership")):
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
        alias_column = "、".join(dict.fromkeys(aliases or ()))
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


#: What each category is, for the generated structure table. A category with
#: no blurb falls back to its preset name — a missing line is cosmetic, while a
#: hardcoded pair of categories was a table that silently stopped covering the
#: store (2026-09-02).
_CATEGORY_BLURBS = {
    "streamer": "主播",
    "common": "主播之外的一切",
    "style": "一套翻译口味，按 --style 点名注入",
}


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


def over_budget_marker(
    store: KnowledgeStore, subject_id: str, category: str, rev: int | None = None
) -> str:
    """The line the model reads when this entry no longer fits its budget.

    Empty when the category has no budget or the entry is inside it. Attached
    where the entry is INJECTED rather than inside `render_subject`, because
    the measure is taken on that rendering — computing it during the rendering
    would have to render itself.
    """

    budget = preset_for_category(category or "common").max_entry_tokens
    if budget is None:
        return ""
    used = entry_prompt_tokens(store, subject_id, rev)
    if used <= budget:
        return ""
    return (
        f"（\u26a0 本条目已超注入预算：约 {used} / {budget} token。"
        "本次**只能压缩**——把行改短、把两条并成一条、或删掉最弱的几条；新增会被整批拒绝）"
    )


def render_structure_spec(categories: Sequence[str] | None = None) -> str:
    """The section/label table the prompts describe, GENERATED from the
    presets (plan §3 single source of truth).

    Hand-writing it is how `fragment_knowledge_structure_v1` came to teach a
    three-segment term line while the parser had required four for a whole
    release — two copies of the same rule, drifting."""

    out: list[str] = []
    for category in categories or MATCHABLE_CATEGORIES:
        preset = preset_for_category(category)
        title = f"{category}（{_CATEGORY_BLURBS.get(category, preset.name)}）"
        rule = "固定小节，不可新建" if preset.strict_sections else "档案固定，其余小节自由命名"
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


def section_comment(spec, *, preamble: str = "") -> list[str]:  # type: ignore[no-untyped-def]
    """Full-preview guidance block for one section: purpose, exclusion
    discipline, the line-shape hint and every registered label's note. It is
    an HTML comment so the harvest can drop it without it ever becoming a
    content line.

    ``preamble`` scopes the block when it is not describing one named section --
    see ``default_section_comment``."""

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
    extra = [label for label in spec.labels if not label.core]
    if core:
        body.append("标记（缺了完整预览会留一个空槽）：")
        body.extend(f"  [{label.name}] {label.note}".rstrip() for label in core)
    if extra:
        # With their notes, not just their names. A registered non-core label
        # is one the model is *allowed* to use but never reminded to -- so the
        # note is the only thing that says what belongs there and in what shape,
        # and a bare name list left it guessing. (`[当前版本]` in the common
        # preset carried a note nobody could see for exactly this reason.)
        body.append("登记的其他标记（可用，不提醒）：")
        body.extend(f"  [{label.name}] {label.note}".rstrip() for label in extra)
    if not body:
        return []
    lines = ["<!-- " + body[0]]
    lines.extend("     " + row for row in body[1:])
    lines[-1] += " -->"
    return lines


_FREE_SECTION_PREAMBLE = (
    "以下自由节（节名自取）统一按这条规则收录——每节不再重复："
)


def default_section_comment(spec) -> list[str]:  # type: ignore[no-untyped-def]
    """The free-section rule, rendered ONCE for the whole subject.

    A preset with ``strict_sections = false`` resolves every unnamed section to
    the same ``default_section`` spec, so rendering the block per section
    printed one identical paragraph under 「角色」, 「地区」, 「组织」... The text
    was never about a section anyway: it states what any free section may hold.
    Saying it once, up front, is both shorter and the truer placement -- a named
    section keeps its own block, because there the text really is section-local.
    """

    return section_comment(spec, preamble=_FREE_SECTION_PREAMBLE)


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
    """

    if mode not in MODES:
        raise ValueError(mode)
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
        out.extend(default_section_comment(free_spec))
    for view in ordered:
        if wanted is not None and view.name not in wanted:
            continue
        spec = preset.section(view.name)
        used_labels = {
            # during the Phase A→B window a legacy fact's field IS its label
            str(child.payload.get("label") or child.payload.get("field") or "")
            for _, child in view.entries
        }
        slots = (
            [label for label in spec.core_labels() if label.name not in used_labels]
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
            out.extend(section_comment(spec))
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
        for label in slots:
            # The alias label has no node of its own: it IS the subject's alias
            # column, rendered from items and diffed back into them (plan §4).
            body = ""
            if label.role == "aliases":
                body = "、".join(subject_aliases(store, subject.local_id, tree.rev))
            # Otherwise an empty slot is the whole reminder — no signal, no
            # ledger, nothing stored (owner decision, plan §0.1 item 17).
            line = f"[{label.name}] {body}".rstrip()
            out.append(f"- {line}" if mode == "human" else line)

        if mode == "legacy":
            # The next heading brings one blank with it; anything beyond that
            # was the user's own spacing (see ParsedSection.trailing_blank).
            for _ in range(max(0, int(trailing_blank.get(view.name, 1)) - 1)):
                out.append("")

    if not (mode == "legacy" and layout.get("has_metadata") is False):
        out.append("")
        out.append(f"## {METADATA_SECTION}")
        if mode == "legacy":
            if blank_after.get(METADATA_SECTION, False):
                out.append("")
        else:
            out.append("")
        out.append(f"{UPDATED_DATE_LABEL}: {payload.get('updated_date', '')}")
    text = "\n".join(out)
    if mode == "legacy" and not layout.get("trailing_newline", True):
        return text
    return text + "\n"


def _ordered_sections(tree: SubjectTree, preset) -> list[SectionView]:  # type: ignore[no-untyped-def]
    """Preset order first (empty slots kept), then any free sections in stored order."""

    by_name = {view.name: view for view in tree.sections}
    explicit = tree.subject.payload.get("section_order")
    if explicit:
        ordered = [by_name.get(name) or SectionView(name) for name in explicit]
        seen = set(explicit)
        ordered.extend(view for view in tree.sections if view.name not in seen)
        return ordered
    ordered = [by_name.get(name) or SectionView(name) for name in preset.section_names()]
    seen = set(preset.section_names())
    ordered.extend(view for view in tree.sections if view.name not in seen)
    return ordered


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
    native = "、".join(payload.get("native_names", []))
    return f"- {head} | {native} | {'、'.join(aliases)} | {payload.get('intro', '')}"


def render_index(store: KnowledgeStore, category: str, *, rev: int | None = None, header: str = "") -> str:
    lines = [header.rstrip("\n")] if header else []
    for subject in store.subjects(rev):
        if subject.payload.get("category") != category:
            continue
        if subject.maturity == "tentative":
            continue  # shadow-only (plan §11.5): not in the model-facing index
        lines.append(render_index_line(subject, subject_aliases(store, subject.local_id, rev)))
    return "\n".join(lines) + "\n"
