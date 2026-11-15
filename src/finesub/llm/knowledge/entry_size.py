"""The notice a WRITABLE surface hangs on an entry grown past its category's
soft size (plan kb-entry-size, step 2).

Writes are never refused for length (the cap is an injection cap,
`injection_budget`); what keeps entries from growing without end is this
notice, shown where a model can act on it -- the knowledge update's prefetch,
an agent's ``propose``-scoped ``kb_read``, the knowledge repair session --
never on a correction window, which cannot write.

It asks for a cut of at least X% and says nothing else about length: no
target size and no token count (owner 2026-10-02 -- given a target, a model
stops right on it; and its sense of length is poor). X is worked out toward
four fifths of the soft size, so an entry just over 5000 is asked for 20%.
"""

from __future__ import annotations

import math

from ..prompt_compose import load_prompt_template
from .node.presets import preset_for_category
from .node.render import entry_prompt_tokens
from .node.store import KnowledgeStore

#: The cut aims at this share of the soft size: far enough under it that the
#: next update does not hit the notice again straight away.
COMPRESSION_TARGET_SHARE = 0.8


def compression_percent(tokens: int, soft: int) -> int:
    """How much to cut, in whole fives: toward ``soft × 0.8``."""

    target = soft * COMPRESSION_TARGET_SHARE
    if tokens <= target:
        return 0
    raw = (tokens - target) / tokens * 100
    return min(95, int(math.ceil(raw / 5.0) * 5))


#: The split paragraph's own fixed words: how a prompt assembled around
#: injected entries knows one of them offers `split_entry` (and so the op
#: contract must carry its row).
SPLIT_OFFER_MARK = "也可以改为用 `split_entry`"


def offers_split(text: str) -> bool:
    return SPLIT_OFFER_MARK in (text or "")


#: The notice's own fixed words (`fragment_entry_size_notice_v1.md`): how a prompt knows an injected entry asks
#: to be compressed (and so the size rule has something to apply to).
COMPRESS_NOTICE_MARK = "⚠ 本条目偏长"


def asks_to_compress(text: str) -> bool:
    return COMPRESS_NOTICE_MARK in (text or "")


def section_stats(store: KnowledgeStore, subject_id: str, rev: int | None = None) -> str:
    """「角色 ≈2800（术语行 92%）、地区 ≈900（术语行 40%）」: what the split
    criterion is judged on, largest first, head sections left out."""

    from ..token_budget import HeuristicTokenCounter
    from .node.presets import head_section_names
    from .node.render import _ordered_sections, format_line, load_subject_tree, node_aliases

    count = HeuristicTokenCounter().count_text
    tree = load_subject_tree(store, subject_id, rev)
    preset = preset_for_category(tree.subject.payload.get("category") or "common")
    head = head_section_names()
    rows = []
    for view in _ordered_sections(tree, preset):
        if view.name in head or not view.entries:
            continue
        tokens = sum(
            count(format_line(child, aliases=node_aliases(store, child, tree.rev)) + "\n")
            for _membership, child in view.entries
        )
        terms = sum(1 for _membership, child in view.entries if child.kind == "term")
        rows.append((tokens, view.name, round(100 * terms / len(view.entries))))
    rows.sort(reverse=True)
    return "、".join(f"{name} ≈{tokens}（术语行 {share}%）" for tokens, name, share in rows[:6])


def size_notice(
    store: KnowledgeStore, subject_id: str, rev: int | None = None, *, with_contract: bool = False
) -> str:
    """The notice for this entry, or "" when it is within its soft size.

    Where the category may split (`splittable`) and the entry is not itself a
    sub-entry, it adds the split criterion with the per-section numbers it is
    judged on -- compressing comes first, splitting is the alternative for a
    section that qualifies. ``with_contract`` appends the op's own row: for a
    surface (an agent's `kb_read`) whose prompt was assembled before it knew
    which entries it would read.
    """

    subject = store.node(subject_id, rev)
    if subject is None:
        return ""
    preset = preset_for_category(subject.payload.get("category") or "common")
    soft = preset.soft_entry_tokens
    if soft is None:
        return ""
    tokens = entry_prompt_tokens(store, subject_id, rev)
    if tokens <= soft:
        return ""
    over_cap = (
        load_prompt_template("fragment_entry_size_over_cap_v1.md", strict=True).strip()
        if preset.max_entry_tokens is not None and tokens > preset.max_entry_tokens
        else ""
    )
    notice = load_prompt_template(
        "fragment_entry_size_notice_v1.md",
        strict=True,
        percent=compression_percent(tokens, soft),
        over_cap=over_cap,
    ).strip()
    from .node.subentries import parent_of

    if not preset.splittable or parent_of(store, subject_id, rev) is not None:
        return notice
    stats = section_stats(store, subject_id, rev)
    if not stats:
        return notice
    split = load_prompt_template(
        "fragment_entry_size_split_v1.md",
        strict=True,
        entry=str(subject.payload.get("surface") or ""),
        section_stats=stats,
    ).strip()
    if with_contract:
        split += "\n" + load_prompt_template("fragment_knowledge_split_op_row_v1.md").strip()
    return f"{notice}\n{split}"
