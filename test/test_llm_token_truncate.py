from __future__ import annotations

from llm.token_truncate import (
    TruncateResult,
    truncate_text_only,
    truncate_to_token_window,
)


def char_counter(text: str) -> int:
    """Deterministic fake counter: one token per character (plus wrapper)."""

    return len(text) + 1 if text else 0


def test_returns_full_text_when_already_within_limit() -> None:
    # lazy=False so the real counter (not the heuristic pre-check) is exercised.
    result = truncate_to_token_window("hello", 100, char_counter, lazy=False)

    assert result.text == "hello"
    assert result.reason == "already_within_limit"
    assert result.tokens <= 100


def test_lazy_returns_full_text_without_real_counter_calls() -> None:
    calls = {"n": 0}

    def counting(text: str) -> int:
        calls["n"] += 1
        return len(text) + 1

    # Heuristic upper bound says it fits, so no real counter call happens.
    result = truncate_to_token_window(
        "short enough", 10_000, counting, lazy=True
    )

    assert result.text == "short enough"
    assert result.reason == "lazy_within_limit"
    assert result.tokens <= 10_000
    assert calls["n"] == 0


def test_lazy_off_still_counts_and_reports_already_within_limit() -> None:
    result = truncate_to_token_window("hello", 100, char_counter, lazy=False)

    assert result.reason == "already_within_limit"
    assert result.calls >= 1


def test_lazy_falls_through_to_search_when_estimate_exceeds_limit() -> None:
    # A long text whose heuristic estimate exceeds the limit must still truncate.
    text = "a" * 1000
    result = truncate_to_token_window(text, 100, char_counter, lazy=True)

    assert result.reason != "lazy_within_limit"
    assert result.tokens <= 100
    assert len(result.text) < 1000


def test_lazy_custom_heuristic_override() -> None:
    # A custom (deliberately huge) heuristic disables the fast path.
    result = truncate_to_token_window(
        "hello", 100, char_counter, lazy=True, heuristic_count=lambda t: 10_000
    )

    assert result.reason != "lazy_within_limit"


def test_lazy_safety_factor_requires_headroom() -> None:
    # Heuristic estimate is a flat 100; the 1.02 factor makes the effective
    # threshold 102, so a limit of 101 must NOT short-circuit but 103 must.
    hundred = lambda t: 100  # noqa: E731

    no_short = truncate_to_token_window(
        "abc", 101, char_counter, lazy=True, heuristic_count=hundred
    )
    assert no_short.reason != "lazy_within_limit"

    short = truncate_to_token_window(
        "abc", 103, char_counter, lazy=True, heuristic_count=hundred
    )
    assert short.reason == "lazy_within_limit"

    # An explicit factor of 1.0 disables the extra guard (100 <= 101 fits).
    off = truncate_to_token_window(
        "abc", 101, char_counter, lazy=True, heuristic_count=hundred, lazy_safety_factor=1.0
    )
    assert off.reason == "lazy_within_limit"


def test_quick_loosens_window_but_explicit_bounds_win() -> None:
    text = "a" * 1000
    # Explicit abs_slack overrides quick's default loosening.
    tight = truncate_to_token_window(
        text, 100, char_counter, lazy=False, quick=True, gold_ratio=0.98, abs_slack=1
    )
    assert tight.tokens <= 100
    # With abs_slack=1 and gold_ratio 0.98 the window is [98,100]; interpolation
    # on a linear counter still lands on the exact boundary.
    assert tight.tokens >= 99


def test_truncates_to_prefix_within_limit_and_hits_window() -> None:
    text = "a" * 1000
    result = truncate_to_token_window(text, 100, char_counter)

    assert result.tokens <= 100
    assert result.hit_window
    assert result.text == text[: len(result.text)]
    # char_counter => len(prefix)+1 tokens, so 99 chars == 100 tokens.
    assert len(result.text) == 99


def test_never_exceeds_limit_for_any_input() -> None:
    text = "语言 language 测试 " * 500
    for limit in (5, 37, 128, 999):
        result = truncate_to_token_window(text, limit, char_counter)
        assert result.tokens <= limit
        assert result.text == text[: len(result.text)]


def test_interpolation_search_uses_few_counter_calls() -> None:
    calls = {"n": 0}

    def counting(text: str) -> int:
        calls["n"] += 1
        return len(text) + 1 if text else 0

    text = "x" * 100_000
    result = truncate_to_token_window(text, 4096, counting)

    assert result.tokens <= 4096
    assert result.hit_window
    # Should converge in far fewer calls than a linear scan.
    assert calls["n"] < 40


def test_empty_wrapper_exceeds_limit_returns_empty() -> None:
    # A counter whose empty-string count already exceeds the limit.
    result = truncate_to_token_window("abcdef", 0, lambda t: len(t) + 5)

    assert result.text == ""
    assert result.reason == "empty_exceeds_limit"


def test_prefer_natural_boundary_backs_up_to_separator() -> None:
    # Sentence boundary just below the cut should be preferred when it still fits.
    text = "第一句。" + "第二句这里很长" * 50
    result = truncate_to_token_window(
        text,
        6,
        char_counter,
        prefer_natural_boundary=True,
    )

    assert result.tokens <= 6
    # With natural boundary preference it should end on the 。 boundary.
    assert result.text.endswith("。")


def test_keep_tail_preserves_the_suffix() -> None:
    text = "a" * 50 + "b" * 50
    result = truncate_to_token_window(text, 10, char_counter, keep="tail")

    assert result.tokens <= 10
    # char_counter => 9 chars == 10 tokens; suffix of the original text.
    assert len(result.text) == 9
    assert result.text == "b" * 9
    assert text.endswith(result.text)


def test_keep_tail_never_exceeds_limit() -> None:
    text = "语言 language 测试 " * 500
    for limit in (5, 37, 128, 999):
        result = truncate_to_token_window(text, limit, char_counter, keep="tail")
        assert result.tokens <= limit
        assert text.endswith(result.text)


def test_keep_tail_natural_boundary_starts_after_separator() -> None:
    text = ("很长的开头部分" * 50) + "。结尾一句话"
    result = truncate_to_token_window(
        text,
        6,
        char_counter,
        keep="tail",
        prefer_natural_boundary=True,
    )

    assert result.tokens <= 6
    assert text.endswith(result.text)
    # The kept suffix should begin right after the sentence boundary.
    assert result.text.startswith("结尾")


def test_keep_head_and_tail_pick_opposite_ends() -> None:
    text = "HEAD" + "x" * 200 + "TAIL"
    head = truncate_to_token_window(text, 8, char_counter, keep="head")
    tail = truncate_to_token_window(text, 8, char_counter, keep="tail")

    assert head.text.startswith("HEAD")
    assert tail.text.endswith("TAIL")


def test_invalid_keep_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="keep must be"):
        truncate_to_token_window("abc", 2, char_counter, keep="middle")


def test_truncate_text_only_returns_string() -> None:
    out = truncate_text_only("a" * 50, 10, char_counter)

    assert isinstance(out, str)
    assert len(out) == 9


def test_result_is_frozen_dataclass() -> None:
    result = truncate_to_token_window("hi", 100, char_counter)

    assert isinstance(result, TruncateResult)
