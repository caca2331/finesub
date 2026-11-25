"""Knowledge-entry prefetch for the unified knowledge update.

``knowledge_hints`` from the aggregated task feedback name the entries the
models want to touch; this module dedupes them (aliases resolve to their
primary key), ranks them by frequency (research hints weighted), keeps the
top N, and renders their current bodies as one budget-capped prompt block so
the update model edits against the real contents instead of guessing.

Bodies are the *prompt projection* (plan §2.3): every node carries a short
handle (``@k12`` …) the model uses to address it; the returned ``HandleMap``
binds each handle to ``(id, expected_valid_from_rev)`` for the CAS apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..injection_budget import (
    RenderedBlock,
    render_knowledge_entries_block,
    retain_injected_handles,
)
from .feedback import KnowledgeHint, RESEARCH_HINT_WEIGHT
from .node.render import HandleMap
from .node.repo import KnowledgeRepo

# Per docs/knowledge.md: at most 20 prefetched entries per chunk, each capped
# at its category's `max_entry_tokens`, the whole block ≤60k tokens (owner
# 2026-10-02: 40k left a compress-worthy entry over ~20k without a whole share;
# 60k still keeps a full 110k chunk of window material under the 194k input limit; a chunk is budgeted with
# the block at this cap, `update._plan_chunks`).
MAX_PREFETCH_ENTRIES = 20
ENTRY_EXCERPT_BLOCK_MAX_TOKENS = 60_000


@dataclass(frozen=True)
class EntrySelection:
    """One deduped, ranked knowledge entry the update prompt should carry."""

    category: str
    key: str
    score: float
    exists: bool
    applied: bool = False
    hint_names: tuple[str, ...] = ()
    #: Served before the rest of the block (the run's style entries).
    pinned: bool = False

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "key": self.key,
            "score": self.score,
            "exists": self.exists,
            "applied": self.applied,
            "hint_names": list(self.hint_names),
        }


def select_kb_entries(
    hints: Sequence[KnowledgeHint],
    *,
    knowledge_root: str | Path,
    research_origins: Iterable[KnowledgeHint] = (),
    applied_entries: Iterable[tuple[str, str]] = (),
    research_weight: int = RESEARCH_HINT_WEIGHT,
    max_entries: int = MAX_PREFETCH_ENTRIES,
    rev: int | None = None,
) -> list[EntrySelection]:
    """Rank hint entries by frequency and keep the top ``max_entries``.

    ``hints`` are window-level hints (×1 each); ``research_origins`` are the
    research round's hints (×``research_weight`` each). Aliases resolve to the
    primary key so 崩铁 and 崩坏星穹铁道 merge into one candidate; names the
    index does not know keep the hint's own ``(category, entry)`` and are
    flagged ``exists=False``. ``applied_entries`` marks ``(category, key)``
    pairs already written by an earlier chunk of the same task.
    """

    from .node.subentries import child_holding

    repo = KnowledgeRepo.open(knowledge_root)
    at = repo.rev if rev is None else rev

    def _resolve_hint(hint: KnowledgeHint) -> tuple[str, str] | None:
        """``(category, key)`` the hint should score: the entry it names, or
        the sub-entry that by now holds the line it names (``sub``)."""

        found = repo.resolve(hint.entry, at)
        if found is None:
            return None
        if hint.sub:
            holder = child_holding(repo.store, found.subject_id, hint.sub, at)
            node = repo.store.node(holder, at) if holder else None
            if node is not None:
                return (str(node.payload.get("category") or found.category),
                        str(node.payload.get("surface") or ""))
        return (found.category, found.key)

    scores: dict[tuple[str, str], float] = {}
    names: dict[tuple[str, str], dict[str, None]] = {}
    exists_map: dict[tuple[str, str], bool] = {}

    def _accumulate(hint: KnowledgeHint, weight: float) -> None:
        resolved = _resolve_hint(hint)
        if resolved is not None:
            key = resolved
            exists_map[key] = True
        else:
            key = (hint.category, hint.entry)
            exists_map.setdefault(key, False)
        scores[key] = scores.get(key, 0.0) + weight
        names.setdefault(key, {}).setdefault(hint.entry)

    for hint in hints:
        _accumulate(hint, 1.0)
    for hint in research_origins:
        _accumulate(hint, float(research_weight))
    applied = {(category, key) for category, key in applied_entries}
    # Stable ranking: score desc, then first-seen order (dict preserves it).
    order = {key: idx for idx, key in enumerate(scores)}
    ranked = sorted(scores, key=lambda key: (-scores[key], order[key]))
    return [
        EntrySelection(
            category=category,
            key=key,
            score=scores[(category, key)],
            exists=exists_map[(category, key)],
            applied=(category, key) in applied,
            hint_names=tuple(names[(category, key)]),
        )
        for category, key in ranked[: max(0, int(max_entries))]
    ]


def pin_style_entries(
    selections: Sequence[EntrySelection], style_keys: Sequence[str]
) -> list[EntrySelection]:
    """Put the run's style entries at the head of the selection.

    Pinned, not ranked: a style entry is the update task's own target
    (`docs/knowledge.md`, 「style 类别」), so it must
    survive the `max_entries` cut whatever the window hints scored, and it goes
    first so the model reads the conventions before the material. Any style
    entry that arrived through ranking is dropped: it would be a second copy.
    """

    from .style import STYLE_CATEGORY

    pinned = [
        EntrySelection(category=STYLE_CATEGORY, key=key, score=float("inf"), exists=True, pinned=True)
        for key in style_keys
    ]
    return pinned + [s for s in selections if s.category != STYLE_CATEGORY]


def _entry_section_text(
    selection: EntrySelection,
    repo: KnowledgeRepo,
    handles: HandleMap,
    rev: int,
    *,
    notice: str = "",
) -> str:
    markers: list[str] = []
    if selection.applied:
        markers.append("（本任务前序块已更新：仅在有新增证据时再动）")
    if notice:
        markers.append(notice)
    body = ""
    if selection.exists:
        resolved = repo.resolve(selection.key, rev, category=selection.category)
        if resolved is not None:
            body = repo.entry_prompt_text(resolved.subject_id, handles, rev).rstrip("\n")
    if not body:
        markers.append("（库中暂无：如证据充分可用 create_entry 新建条目）")
    # Non-heading delimiter (same style as the window packs): the entry body is
    # verbatim content whose own `#`/`##` headings must stay authoritative.
    header = f"--- {selection.category}/{selection.key} ---"
    if markers:
        header += "\n" + "\n".join(markers)
    return f"{header}\n{body}".rstrip()


def render_kb_entry_excerpt(
    selections: Sequence[EntrySelection],
    knowledge_root: str | Path,
    *,
    count_tokens: Callable[[str], int],
    block_limit: int = ENTRY_EXCERPT_BLOCK_MAX_TOKENS,
    rev: int | None = None,
    ask_to_compress: bool = True,
) -> tuple[RenderedBlock, HandleMap]:
    """Render the selections' bodies (prompt projection at ``rev``) under the
    knowledge-entry budget, together with the handle map the apply needs.

    Served first, whole: the run's style entries (`pin_style_entries`) and
    every entry grown past its category's soft size -- an entry asked to be
    compressed must be visible in full, past its injection cap too, or the
    model cannot compress the lines it does not see (plan kb-entry-size).
    Only those that do get in whole carry the compression notice
    (`entry_size.size_notice`); one pushed back into the shared rest is cut
    like any other and asked for nothing this time. Everything else is capped
    at its category's `max_entry_tokens`. Handles of lines the budget cut are
    unbound: the model never saw them, so it must not be able to name them.

    Reloaded per chunk at the chunk's ``working_rev``: after a chunk's
    proposals apply, the next chunk's excerpt reflects the just-written
    contents.

    ``ask_to_compress`` False (the conflict repair, which may only re-decide
    the dropped rows): oversized entries are still served whole, without the
    notice -- it could only be ignored or overstep (owner 2026-10-03).
    """

    from .entry_size import size_notice

    repo = KnowledgeRepo.open(knowledge_root)
    at = repo.rev if rev is None else rev
    handles = HandleMap()
    label_of = {id(s): f"{s.category}/{s.key}" for s in selections}
    notices: dict[str, str] = {}
    for selection in selections:
        if not selection.exists:
            continue
        resolved = repo.resolve(selection.key, at, category=selection.category)
        notice = size_notice(repo.store, resolved.subject_id, at) if resolved is not None else ""
        if notice:
            notices[label_of[id(selection)]] = notice
    pinned = [label_of[id(s)] for s in selections if s.pinned] + list(notices)

    def cap(label: str) -> int | None:
        return None if label in notices else _category_cap(label)

    def render(noticed: set[str]) -> RenderedBlock:
        bodies = {
            label_of[id(s)]: _entry_section_text(
                s, repo, handles, at, notice=notices[label_of[id(s)]] if label_of[id(s)] in noticed else ""
            )
            for s in selections
        }
        return render_knowledge_entries_block(
            bodies, count_tokens=count_tokens, block_limit=block_limit, entry_limit=cap, pinned=pinned
        )

    if not ask_to_compress:
        block = render(set())
        retain_injected_handles(handles, block.text)
        return block, handles
    # Which compress-worthy entries get in whole decides who carries the
    # notice; the notice's own few tokens can tip one over, so settle it.
    noticed = set(notices) & set(render(set()).included)
    block = render(noticed)
    for _ in range(len(noticed)):
        cut = noticed - set(block.included)
        if not cut:
            break
        noticed -= cut
        block = render(noticed)
    retain_injected_handles(handles, block.text)
    return block, handles


def _category_cap(label: str) -> int | None:
    """``category/key`` -> that category's injection cap."""

    from .node.presets import preset_for_category

    return preset_for_category(label.split("/", 1)[0] or "common").max_entry_tokens
