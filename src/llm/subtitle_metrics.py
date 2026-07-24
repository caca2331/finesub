"""Compatibility exports for the shared subtitle display-length metrics."""

from __future__ import annotations

from subtitle_metrics import (
    DEFAULT_VISIBLE_CHAR_WEIGHT,
    REDUCED_CHAR_WEIGHT,
    format_weighted_char_count,
    weighted_char_count,
)

__all__ = [
    "DEFAULT_VISIBLE_CHAR_WEIGHT",
    "REDUCED_CHAR_WEIGHT",
    "format_weighted_char_count",
    "weighted_char_count",
]
