"""Shared display-length metrics for subtitle text."""

from __future__ import annotations

import unicodedata


REDUCED_CHAR_WEIGHT = 0.5
DEFAULT_VISIBLE_CHAR_WEIGHT = 1.0


def weighted_char_count(text: str) -> float:
    """Return the shared weighted display length for subtitle text.

    Latin letters, Unicode numbers, punctuation, and spaces count as half a
    character. Other visible characters count as one. Combining marks and
    non-rendering control/format characters do not add display length.
    """

    total = 0.0
    for char in text or "":
        category = unicodedata.category(char)
        if category.startswith("M") or category in {"Cc", "Cf", "Cs", "Cn"}:
            continue
        if (
            char.isspace()
            or category.startswith(("N", "P"))
            or (
                category.startswith("L")
                and "LATIN" in unicodedata.name(char, "")
            )
        ):
            total += REDUCED_CHAR_WEIGHT
        else:
            total += DEFAULT_VISIBLE_CHAR_WEIGHT
    return total


def format_weighted_char_count(value: float | int) -> str:
    """Format a numeric weighted count without a redundant ``.0`` suffix."""

    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"
