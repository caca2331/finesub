"""Truncate text to a token budget with a minimal number of count calls.

Given a ``count_tokens`` callable (any :class:`~finesub.llm.token_budget.TokenCounter`'s
``count_text``), :func:`truncate_to_token_window` finds the longest slice that
fits ``limit`` and, when possible, lands inside a tight window just below the
limit. It uses interpolation search (with periodic bisection to avoid crawling)
over slice lengths, caching each count, so a typical run needs only a handful of
counter calls even for large inputs.

``keep`` selects which end is preserved:

- ``keep="head"`` (default) keeps a prefix — the tail is truncated.
- ``keep="tail"`` keeps a suffix — the prefix is truncated (useful for keeping
  the most recent content, e.g. the end of a transcript or ledger).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Callable, Optional

from .token_budget import (
    HeuristicTokenCounter,
    TokenCounter,
    default_token_counter,
    local_counter_available_for,
)

KEEP_HEAD = "head"
KEEP_TAIL = "tail"

# Default hit-window tolerance, and the looser tolerance used under `quick`
# (wider window ⇒ the interpolation search terminates in fewer counter calls).
DEFAULT_GOLD_RATIO = 0.98
DEFAULT_ABS_SLACK = 20
QUICK_GOLD_RATIO = 0.95
QUICK_ABS_SLACK = 50
# The lazy fast path inflates the heuristic estimate by this factor before
# comparing to the limit — a small extra guard on top of the heuristic's own
# upper-bound margin, so short-circuiting only happens with a little headroom.
LAZY_SAFETY_FACTOR = 1.02


@dataclass(frozen=True)
class TruncateResult:
    text: str
    tokens: int
    hit_window: bool
    calls: int
    iterations: int
    reason: str


def truncate_to_token_window(
    text: str,
    limit: int,
    count_tokens: Callable[[str], int],
    *,
    keep: str = KEEP_HEAD,
    lazy: bool = True,
    quick: bool = True,
    lazy_safety_factor: float = LAZY_SAFETY_FACTOR,
    heuristic_count: Optional[Callable[[str], int]] = None,
    gold_ratio: Optional[float] = None,
    abs_slack: Optional[int] = None,
    max_iters: Optional[int] = None,
    prefer_natural_boundary: bool = False,
    natural_boundary_scan: int = 300,
) -> TruncateResult:
    """
    将 text 截断到 token 数 <= limit，并尽量命中以下任一窗口：

    1. ceil(gold_ratio * limit) <= tok_cnt <= limit
    2. limit - abs_slack <= tok_cnt <= limit

    默认（`quick=True`）：0.95 * limit <= tok_cnt <= limit 或 limit - 50 <= tok_cnt。
    `quick=False`：收紧为 0.98 * limit / limit - 20（命中更贴上限，但计数次数更多）。
    显式传入 ``gold_ratio``/``abs_slack`` 时以显式值为准，`quick` 不再覆盖。

    ``lazy``（默认开）：本地 tokcount 不可用时，先用启发式 upper-bound counter
    估算整段 token；若 `估算 × lazy_safety_factor`（默认 1.02，额外保险）<=
    limit 则直接原样返回，不做 API 计数。本地 tokcount 可用时跳过估算、直接精确
    计数。显式 ``heuristic_count`` 始终覆盖该分流，便于测试或定制预检。

    ``keep`` 决定保留哪一端：``"head"`` 保留前缀（截断尾部，默认），``"tail"``
    保留后缀（截断前缀）。两种模式的搜索完全对称——保留的字符越多，token 越多。

    如果无法命中，则返回已知 token 数 <= limit 且最接近 limit 的安全切片。
    """
    if keep not in (KEEP_HEAD, KEEP_TAIL):
        raise ValueError("keep must be 'head' or 'tail'")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    # `quick` loosens the tolerance only where the caller left the default.
    if gold_ratio is None:
        gold_ratio = QUICK_GOLD_RATIO if quick else DEFAULT_GOLD_RATIO
    if abs_slack is None:
        abs_slack = QUICK_ABS_SLACK if quick else DEFAULT_ABS_SLACK
    if not 0 < gold_ratio <= 1:
        raise ValueError("gold_ratio must be in (0, 1]")
    if abs_slack < 0:
        raise ValueError("abs_slack must be non-negative")

    n = len(text)

    # Without a local binary, a cheap upper-bound estimate that already fits
    # avoids an API call. A runnable local binary is cheap enough to count
    # exactly; an explicit heuristic override still wins for tests/custom use.
    if lazy and (
        heuristic_count is not None
        or not local_counter_available_for(count_tokens)
    ):
        pre_count = heuristic_count or HeuristicTokenCounter().count_text
        pre_tokens = pre_count(text)
        if pre_tokens * lazy_safety_factor <= limit:
            return TruncateResult(
                text=text,
                tokens=pre_tokens,  # heuristic upper bound, not an exact count
                hit_window=False,
                calls=0,
                iterations=0,
                reason="lazy_within_limit",
            )

    cache: dict[int, int] = {}

    def piece(kept_len: int) -> str:
        # kept_len is the number of characters kept, measured from whichever end
        # ``keep`` preserves. Monotonic in both modes: more kept -> more tokens.
        kept_len = max(0, min(n, kept_len))
        return text[:kept_len] if keep == KEEP_HEAD else text[n - kept_len :]

    def count_prefix(idx: int) -> int:
        idx = max(0, min(n, idx))
        if idx not in cache:
            cache[idx] = count_tokens(piece(idx))
        return cache[idx]

    ratio_lower = ceil(gold_ratio * limit)
    abs_lower = max(0, limit - abs_slack)

    def hit_window(tok_cnt: int) -> bool:
        if tok_cnt > limit:
            return False

        hit_ratio_window = ratio_lower <= tok_cnt <= limit
        hit_abs_window = abs_lower <= tok_cnt <= limit

        return hit_ratio_window or hit_abs_window

    iterations = 0

    empty_tokens = count_prefix(0)

    if empty_tokens > limit:
        return TruncateResult(
            text="",
            tokens=empty_tokens,
            hit_window=False,
            calls=len(cache),
            iterations=iterations,
            reason="empty_exceeds_limit",
        )

    total_tokens = count_prefix(n)

    if total_tokens <= limit:
        return TruncateResult(
            text=text,
            tokens=total_tokens,
            hit_window=hit_window(total_tokens),
            calls=len(cache),
            iterations=iterations,
            reason="already_within_limit",
        )

    low_idx, low_tokens = 0, empty_tokens
    high_idx, high_tokens = n, total_tokens

    best_safe_idx, best_safe_tokens = low_idx, low_tokens

    def record_safe(idx: int, tok_cnt: int) -> None:
        nonlocal best_safe_idx, best_safe_tokens

        if tok_cnt <= limit and (
            tok_cnt > best_safe_tokens
            or (tok_cnt == best_safe_tokens and idx > best_safe_idx)
        ):
            best_safe_idx = idx
            best_safe_tokens = tok_cnt

    if max_iters is None:
        max_iters = max(24, int(log2(max(n, 2))) * 3)

    def finalize(idx: int, tokens: int, reason: str) -> TruncateResult:
        final_idx, final_tokens = maybe_move_to_natural_boundary(
            text=text,
            n=n,
            keep=keep,
            idx=idx,
            tokens=tokens,
            hit_window=hit_window,
            count_prefix=count_prefix,
            scan=natural_boundary_scan,
            enabled=prefer_natural_boundary,
        )
        return TruncateResult(
            text=piece(final_idx),
            tokens=final_tokens,
            hit_window=hit_window(final_tokens),
            calls=len(cache),
            iterations=iterations,
            reason=reason,
        )

    # 第一阶段：插值搜索加速。
    for step in range(max_iters):
        iterations += 1

        if high_idx - low_idx <= 1:
            break

        use_bisect = False
        denom = high_tokens - low_tokens

        if denom <= 0:
            use_bisect = True
        else:
            proportion = (limit - low_tokens) / denom

            if not 0 < proportion < 1:
                use_bisect = True

        # 防止插值法在坏分布中贴边慢爬。
        if step % 4 == 3:
            use_bisect = True

        if use_bisect:
            next_idx = (low_idx + high_idx) // 2
        else:
            next_idx = low_idx + round(proportion * (high_idx - low_idx))

        # 严格推进，防死循环。
        if next_idx <= low_idx:
            next_idx = low_idx + 1
        elif next_idx >= high_idx:
            next_idx = high_idx - 1

        cur_tokens = count_prefix(next_idx)

        if cur_tokens <= limit:
            record_safe(next_idx, cur_tokens)

        if hit_window(cur_tokens):
            return finalize(next_idx, cur_tokens, "hit_window")

        if cur_tokens > limit:
            high_idx = next_idx
            high_tokens = cur_tokens
        else:
            low_idx = next_idx
            low_tokens = cur_tokens

    # 第二阶段：二分兜底，尽量找到最接近 limit 的安全切片。
    while high_idx - low_idx > 1:
        iterations += 1

        mid_idx = (low_idx + high_idx) // 2
        mid_tokens = count_prefix(mid_idx)

        if mid_tokens <= limit:
            record_safe(mid_idx, mid_tokens)

        if hit_window(mid_tokens):
            return finalize(mid_idx, mid_tokens, "hit_window_after_bisect")

        if mid_tokens > limit:
            high_idx = mid_idx
            high_tokens = mid_tokens
        else:
            low_idx = mid_idx
            low_tokens = mid_tokens

    return finalize(best_safe_idx, best_safe_tokens, "best_safe_fallback")


def maybe_move_to_natural_boundary(
    *,
    text: str,
    n: int,
    keep: str,
    idx: int,
    tokens: int,
    hit_window: Callable[[int], bool],
    count_prefix: Callable[[int], int],
    scan: int,
    enabled: bool,
) -> tuple[int, int]:
    """
    可选：把切边挪到更自然的边界（``idx`` 为保留的字符数）。

    - ``keep="head"``：把前缀末端回退到边界之后（缩短保留长度）。
    - ``keep="tail"``：把后缀开头前移到边界之后（同样缩短保留长度）。

    只有回退后仍然命中窗口时才回退，否则保留原切点。
    """
    if not enabled or idx <= 0 or idx >= n or not hit_window(tokens):
        return idx, tokens

    boundary_chars = [
        "\n\n",
        "\n",
        "。", "！", "？",
        ".", "!", "?",
        "；", ";",
        "，", ",",
        " ",
    ]

    candidates: list[int] = []

    if keep == KEEP_HEAD:
        start = max(0, idx - scan)
        window = text[start:idx]
        for boundary in boundary_chars:
            pos = window.rfind(boundary)
            if pos != -1:
                candidates.append(start + pos + len(boundary))
    else:
        # Kept suffix is text[n-idx:]; move its start forward to a boundary so
        # the suffix begins cleanly. Prefer the earliest boundary (drop least).
        front = n - idx
        region = text[front : min(n, front + scan)]
        for boundary in boundary_chars:
            pos = region.find(boundary)
            if pos != -1:
                new_front = front + pos + len(boundary)
                candidates.append(n - new_front)

    candidates = sorted({c for c in candidates if 0 < c < idx}, reverse=True)

    for cand_idx in candidates:
        cand_tokens = count_prefix(cand_idx)
        if hit_window(cand_tokens):
            return cand_idx, cand_tokens

    return idx, tokens


def cap_tokens(
    text: str,
    limit: int,
    count_tokens: Optional[Callable[[str], int]] = None,
    *,
    keep: str = KEEP_HEAD,
    marker: str = "",
    heuristic_count: Optional[Callable[[str], int]] = None,
) -> str:
    """Cap ``text`` at a token limit; the workhorse behind harness block caps.

    ``count_tokens=None`` falls back to :func:`default_token_counter`. A
    runnable local tokcount binary is used directly; otherwise the lazy
    precheck uses :class:`HeuristicTokenCounter` unless explicitly overridden.
    ``marker`` is appended (``keep="head"``) or prepended (``keep="tail"``)
    only when the text was actually truncated.
    """

    if not text:
        return text
    count = count_tokens or default_token_counter().count_text
    result = truncate_to_token_window(
        text, limit, count, keep=keep, heuristic_count=heuristic_count
    )
    if result.text == text or not marker:
        return result.text
    if keep == KEEP_TAIL:
        return f"{marker}{result.text}"
    return f"{result.text}{marker}"


def truncate_text_only(
    text: str,
    limit: int,
    count_tokens: Callable[[str], int],
    **kwargs,
) -> str:
    """
    简化包装：只返回截断后的文本。
    """
    return truncate_to_token_window(
        text=text,
        limit=limit,
        count_tokens=count_tokens,
        **kwargs,
    ).text


def truncate_with_counter(
    text: str,
    limit: int,
    counter: Optional[TokenCounter] = None,
    **kwargs,
) -> TruncateResult:
    """便捷入口：默认使用标准 fallback token counter (本地 -> API -> 启发式)。"""

    counter = counter or default_token_counter()
    return truncate_to_token_window(
        text=text,
        limit=limit,
        count_tokens=counter.count_text,
        **kwargs,
    )
