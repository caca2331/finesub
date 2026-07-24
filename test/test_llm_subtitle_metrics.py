from __future__ import annotations

from llm.subtitle_metrics import format_weighted_char_count, weighted_char_count


def test_weighted_char_count_uses_shared_subtitle_formula() -> None:
    assert weighted_char_count("中A1， 🙂") == 4.0
    assert weighted_char_count("é") == 0.5
    assert weighted_char_count("e\u0301") == 0.5
    assert weighted_char_count("Я한🙂") == 3.0
    assert weighted_char_count("a\u200db") == 1.0


def test_format_weighted_char_count_omits_redundant_decimal() -> None:
    assert format_weighted_char_count(2.0) == "2"
    assert format_weighted_char_count(1.5) == "1.5"
