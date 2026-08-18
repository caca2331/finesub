"""Token-budgeted rendering of harness-injected prompt blocks.

Search results, extract results, and knowledge entries all share the same
budget scheme: each unit (one query's results, one extracted URL, one
knowledge entry) is rendered as a *section* capped at a per-section token
limit, and the whole block is capped at a block limit derived from the
round's unit cap (``config.injection_block_token_limit``). Sections are
consumed in priority order; the section that overflows the block budget is
truncated if enough budget remains, otherwise it and everything after it is
dropped (later sections never jump the queue past a dropped one).

The caller learns exactly which sections were fully included, truncated, or
dropped — the search loop uses this to avoid decrementing fact priorities
for queries whose results never (fully) reached the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .token_budget import HeuristicTokenCounter, local_counter_available_for
from .token_truncate import truncate_to_token_window

SECTION_JOINER = "\n\n"
# Budget slice set aside for the trailing harness notice that lists
# truncated/dropped sections (only rendered when something was cut). Small
# blocks reserve block_limit // 3 instead so most of the budget stays usable.
NOTICE_RESERVE_TOKENS = 300
# When the remaining block budget falls below this, a section is dropped
# instead of being truncated to a uselessly small fragment.
MIN_PARTIAL_SECTION_TOKENS = 200
_NOTICE_LABEL_LIMIT = 6
_NOTICE_LABEL_CHARS = 30


@dataclass(frozen=True)
class RenderedBlock:
    """A budget-rendered block plus the fate of every input section."""

    text: str
    tokens: int
    included: tuple[str, ...] = ()
    truncated: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()

    def report(self) -> dict:
        return {
            "tokens": self.tokens,
            "included": list(self.included),
            "truncated": list(self.truncated),
            "dropped": list(self.dropped),
        }


EMPTY_BLOCK = RenderedBlock(text="", tokens=0)


def _notice_text(truncated: Sequence[str], dropped: Sequence[str]) -> str:
    def _labels(labels: Sequence[str]) -> str:
        shown = [label[:_NOTICE_LABEL_CHARS] for label in labels[:_NOTICE_LABEL_LIMIT]]
        text = "、".join(shown)
        if len(labels) > _NOTICE_LABEL_LIMIT:
            text += f" 等 {len(labels)} 项"
        return text

    parts = []
    if truncated:
        parts.append(f"以下条目因超出 token 预算被截断：{_labels(truncated)}")
    if dropped:
        parts.append(f"以下条目被整体丢弃：{_labels(dropped)}")
    if not parts:
        return ""
    return "（注入预算说明：" + "；".join(parts) + "。）"


def render_budgeted_block(
    sections: Sequence[tuple[str, str]],
    *,
    count_tokens: Callable[[str], int],
    heuristic_count: Callable[[str], int] | None = None,
    section_limit: int,
    block_limit: int,
    min_partial_tokens: int = MIN_PARTIAL_SECTION_TOKENS,
) -> RenderedBlock:
    """Assemble ``(label, text)`` sections under per-section and block budgets.

    ``heuristic_count`` explicitly overrides budget estimation. Otherwise a
    runnable local tokcount binary is used directly; when it is unavailable,
    the upper-bound estimator skips API counting for sections with ample
    headroom.
    """

    use_local_exact = (
        heuristic_count is None and local_counter_available_for(count_tokens)
    )
    estimate_tokens = (
        count_tokens
        if use_local_exact
        else heuristic_count or HeuristicTokenCounter().count_text
    )
    truncation_heuristic = None if use_local_exact else estimate_tokens
    sections = [(label, (text or "").strip()) for label, text in sections]
    sections = [(label, text) for label, text in sections if text]
    if not sections or block_limit <= 0:
        dropped = tuple(label for label, _ in sections)
        if not dropped:
            return EMPTY_BLOCK
        notice = _notice_text((), dropped)
        return RenderedBlock(
            text=notice, tokens=estimate_tokens(notice), dropped=dropped
        )

    joiner_tokens = estimate_tokens(SECTION_JOINER)
    budget = block_limit - min(NOTICE_RESERVE_TOKENS, block_limit // 3)

    parts: list[str] = []
    included: list[str] = []
    truncated: list[str] = []
    dropped: list[str] = []
    used = 0

    for pos, (label, text) in enumerate(sections):
        join_cost = joiner_tokens if parts else 0
        remaining = budget - used - join_cost
        effective_limit = min(section_limit, remaining)
        if effective_limit < min_partial_tokens:
            # Not enough budget left for a meaningful fragment: drop this and
            # every later section (priority order is preserved).
            dropped.extend(lbl for lbl, _ in sections[pos:])
            break
        result = truncate_to_token_window(
            text,
            effective_limit,
            count_tokens,
            keep="head",
            heuristic_count=truncation_heuristic,
        )
        if not result.text:
            dropped.extend(lbl for lbl, _ in sections[pos:])
            break
        parts.append(result.text)
        used += result.tokens + join_cost
        if result.text == text:
            included.append(label)
        else:
            truncated.append(label)

    if not parts:
        notice = _notice_text(truncated, dropped)
        return RenderedBlock(
            text=notice,
            tokens=estimate_tokens(notice) if notice else 0,
            truncated=tuple(truncated),
            dropped=tuple(dropped),
        )

    text = SECTION_JOINER.join(parts)
    if truncated or dropped:
        notice = _notice_text(truncated, dropped)
        text = f"{text}\n{notice}"
        used += estimate_tokens(notice) + joiner_tokens
    return RenderedBlock(
        text=text,
        tokens=used,
        included=tuple(included),
        truncated=tuple(truncated),
        dropped=tuple(dropped),
    )


def render_knowledge_entries_block(
    entry_details: Mapping[str, str],
    *,
    count_tokens: Callable[[str], int],
    heuristic_count: Callable[[str], int] | None = None,
    entry_limit: int,
    block_limit: int,
) -> RenderedBlock:
    """Render knowledge entries (``key -> body``) under the shared budgets."""

    sections = [
        (key, (body or "").strip())
        for key, body in entry_details.items()
    ]
    return render_budgeted_block(
        sections,
        count_tokens=count_tokens,
        heuristic_count=heuristic_count,
        section_limit=entry_limit,
        block_limit=block_limit,
    )
