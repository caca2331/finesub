from __future__ import annotations

import llm.injection_budget as injection_budget
import llm.token_truncate as token_truncate
from llm.config import (
    INJECTION_BLOCK_BASE_TOKENS,
    INJECTION_BLOCK_PER_UNIT_TOKENS,
    injection_block_token_limit,
)
from llm.injection_budget import (
    NOTICE_RESERVE_TOKENS,
    RenderedBlock,
    render_budgeted_block,
    render_knowledge_entries_block,
)


def char_counter(text: str) -> int:
    """Deterministic fake counter: one token per character."""

    return len(text)


def test_injection_block_token_limit_formula() -> None:
    assert injection_block_token_limit(8) == 8 * INJECTION_BLOCK_PER_UNIT_TOKENS + INJECTION_BLOCK_BASE_TOKENS
    assert injection_block_token_limit(8) == 20_000
    assert injection_block_token_limit(16) == 36_000
    assert injection_block_token_limit(0) == INJECTION_BLOCK_BASE_TOKENS
    assert injection_block_token_limit(-3) == INJECTION_BLOCK_BASE_TOKENS


def test_all_sections_included_when_budget_is_ample() -> None:
    block = render_budgeted_block(
        [("q1", "alpha " * 10), ("q2", "beta " * 10)],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=20_000,
    )

    assert block.included == ("q1", "q2")
    assert block.truncated == ()
    assert block.dropped == ()
    assert "alpha" in block.text and "beta" in block.text
    assert "注入预算说明" not in block.text
    assert block.tokens <= 20_000


def test_ample_budget_skips_real_counter_calls() -> None:
    calls = {"n": 0}

    def counting(text: str) -> int:
        calls["n"] += 1
        return len(text)

    block = render_budgeted_block(
        [("q1", "short result"), ("q2", "another short result")],
        count_tokens=counting,
        section_limit=4_000,
        block_limit=20_000,
    )

    assert block.included == ("q1", "q2")
    assert calls["n"] == 0


def test_ample_budget_uses_exact_counter_when_local_is_available(
    monkeypatch,
) -> None:
    calls = {"n": 0}

    def counting(text: str) -> int:
        if text:
            calls["n"] += 1
        return len(text)

    monkeypatch.setattr(
        injection_budget,
        "local_counter_available_for",
        lambda _count_tokens: True,
    )
    monkeypatch.setattr(
        token_truncate,
        "local_counter_available_for",
        lambda _count_tokens: True,
    )
    block = render_budgeted_block(
        [("q1", "short result"), ("q2", "another short result")],
        count_tokens=counting,
        section_limit=4_000,
        block_limit=20_000,
    )

    assert block.included == ("q1", "q2")
    assert block.tokens == len("short result\n\nanother short result")
    assert calls["n"] == 3  # joiner plus one exact count per section


def test_oversized_section_is_truncated_to_section_limit() -> None:
    long_text = "x" * 2_000
    block = render_budgeted_block(
        [("big", long_text), ("small", "ok")],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=500,
        block_limit=20_000,
    )

    assert block.truncated == ("big",)
    assert block.included == ("small",)
    assert "注入预算说明" in block.text
    assert "big" in block.text.split("注入预算说明")[1]


def test_block_budget_drops_tail_sections_in_priority_order() -> None:
    # Working budget = 1100 - 300 = 800: q0 (400) fits, then the remaining
    # ~398 is below min_partial_tokens=400, so q1 AND q2 are dropped — later
    # sections never jump the queue past a dropped one.
    sections = [(f"q{i}", "a" * 400) for i in range(3)]
    block = render_budgeted_block(
        sections,
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=1_100,
        min_partial_tokens=400,
    )

    assert block.included == ("q0",)
    assert block.dropped == ("q1", "q2")
    assert "q1" in block.text  # named in the notice
    assert block.tokens <= 1_100


def test_overflow_section_truncated_when_remaining_is_meaningful() -> None:
    # Working budget = 1300 - 300 = 1000: q0 (400) fits; q1 (800) does not fit
    # in the remaining ~598 and is truncated to it.
    block = render_budgeted_block(
        [("q0", "a" * 400), ("q1", "b" * 800)],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=1_300,
    )

    assert block.included == ("q0",)
    assert block.truncated == ("q1",)
    assert block.dropped == ()
    assert block.tokens <= 1_300


def test_min_partial_tokens_drops_instead_of_tiny_fragment() -> None:
    block = render_budgeted_block(
        [("q0", "a" * 400), ("q1", "b" * 400)],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=1_100,
        min_partial_tokens=400,  # remaining ~398 < 400 -> drop q1
    )

    assert block.included == ("q0",)
    assert block.dropped == ("q1",)


def test_empty_sections_are_skipped_silently() -> None:
    block = render_budgeted_block(
        [("q0", "  "), ("q1", "content")],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=20_000,
    )

    assert block.included == ("q1",)
    assert block.dropped == ()


def test_no_sections_returns_empty_block() -> None:
    block = render_budgeted_block(
        [],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=100,
        block_limit=100,
    )

    assert block == RenderedBlock(text="", tokens=0)


def test_report_lists_section_fates() -> None:
    block = render_budgeted_block(
        [("q0", "a" * 400), ("q1", "b" * 400), ("q2", "c" * 400)],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=1_100,
        min_partial_tokens=400,
    )

    report = block.report()
    assert report["included"] == ["q0"]
    assert report["dropped"] == ["q1", "q2"]
    assert report["tokens"] == block.tokens


def test_small_block_limit_scales_down_notice_reserve() -> None:
    # Small blocks reserve block_limit // 3 (=150) instead of the full 300,
    # leaving a working budget of 300 for the section itself.
    assert NOTICE_RESERVE_TOKENS == 300
    block = render_budgeted_block(
        [("q0", "a" * 600)],
        count_tokens=char_counter,
        heuristic_count=char_counter,
        section_limit=4_000,
        block_limit=450,
    )

    assert block.truncated == ("q0",)
    assert 250 <= block.tokens <= 450


def test_knowledge_entries_block_renders_entry_bodies_without_extra_heading() -> None:
    block = render_knowledge_entries_block(
        {
            "星穹铁道": "# 星穹铁道\n\n开拓者。",
            "佩克拉": "# 佩克拉\n\n兔子。",
        },
        count_tokens=char_counter,
        heuristic_count=char_counter,
        entry_limit=4_000,
        block_limit=20_000,
    )

    assert "# 星穹铁道" in block.text
    assert "# 佩克拉" in block.text
    assert "## 星穹铁道" not in block.text
    assert block.included == ("星穹铁道", "佩克拉")
