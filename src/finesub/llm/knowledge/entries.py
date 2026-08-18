"""Knowledge-entry prefetch for the unified knowledge update.

``knowledge_hints`` from the aggregated task feedback name the entries the
models want to touch; this module dedupes them (aliases resolve to their
primary key), ranks them by frequency (research hints weighted), keeps the
top N, and renders their current bodies as one budget-capped prompt block so
the update model edits against the real file contents instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..routing.config import INJECTION_SECTION_MAX_TOKENS
from ..injection_budget import RenderedBlock, render_budgeted_block
from .base import entry_path, resolve_entry_key
from .feedback import KnowledgeHint, RESEARCH_HINT_WEIGHT


# Per §1.6/§1.7 of docs/knowledge.md: at most 20 prefetched
# entries per chunk, ≤4k tokens each, whole block ≤40k tokens.
MAX_PREFETCH_ENTRIES = 20
ENTRY_EXCERPT_BLOCK_MAX_TOKENS = 40_000


@dataclass(frozen=True)
class EntrySelection:
    """One deduped, ranked knowledge entry the update prompt should carry."""

    category: str
    key: str
    score: float
    exists: bool
    applied: bool = False
    hint_names: tuple[str, ...] = ()

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
) -> list[EntrySelection]:
    """Rank hint entries by frequency and keep the top ``max_entries``.

    ``hints`` are window-level hints (×1 each); ``research_origins`` are the
    research round's hints (×``research_weight`` each). Aliases resolve to the
    primary key so 崩铁 and 崩坏星穹铁道 merge into one candidate; names the
    index does not know keep the hint's own ``(category, entry)`` and are
    flagged ``exists=False``. ``applied_entries`` marks ``(category, key)``
    pairs already written by an earlier chunk of the same task.
    """

    scores: dict[tuple[str, str], float] = {}
    names: dict[tuple[str, str], dict[str, None]] = {}
    exists_map: dict[tuple[str, str], bool] = {}

    def _accumulate(hint: KnowledgeHint, weight: float) -> None:
        resolved = resolve_entry_key(knowledge_root, hint.entry)
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


def _entry_section_text(
    selection: EntrySelection, knowledge_root: str | Path
) -> str:
    markers: list[str] = []
    if selection.applied:
        markers.append("（本任务前序块已更新：仅在有新增证据时再动）")
    body = ""
    if selection.exists:
        path = entry_path(knowledge_root, selection.category, selection.key)
        if path.exists():
            # v14: line-numbered render (`N| `) so edit_lines can reference
            # physical file lines; numbers match the file exactly.
            file_lines = path.read_text(encoding="utf-8").rstrip("\n").splitlines()
            body = "\n".join(
                f"{number}| {line}" for number, line in enumerate(file_lines, start=1)
            )
    if not body:
        markers.append("（库中暂无：如证据充分可用相应 op 新建条目）")
    # Non-heading delimiter (same style as the window packs): the entry body is
    # verbatim file content whose own `#`/`##` headings must stay authoritative
    # — a `###` wrapper above a `#` title inverts the heading hierarchy.
    header = f"--- {selection.category}/{selection.key} ---"
    if markers:
        header += "\n" + "\n".join(markers)
    return f"{header}\n{body}".rstrip()


def render_kb_entry_excerpt(
    selections: Sequence[EntrySelection],
    knowledge_root: str | Path,
    *,
    count_tokens: Callable[[str], int],
    entry_limit: int = INJECTION_SECTION_MAX_TOKENS,
    block_limit: int = ENTRY_EXCERPT_BLOCK_MAX_TOKENS,
) -> RenderedBlock:
    """Render the selections' current bodies under the shared budget scheme.

    Reloaded per chunk: after a chunk's proposals apply, the next chunk's
    excerpt reflects the just-written contents.
    """

    sections = [
        (
            f"{selection.category}/{selection.key}",
            _entry_section_text(selection, knowledge_root),
        )
        for selection in selections
    ]
    return render_budgeted_block(
        sections,
        count_tokens=count_tokens,
        section_limit=entry_limit,
        block_limit=block_limit,
    )
