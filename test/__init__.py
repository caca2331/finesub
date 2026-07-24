"""Helpers for local comparison/testing utilities."""

from __future__ import annotations

from typing import Any

__all__ = [
    "IntervalComparison",
    "compare_interval_sets",
    "compare_srt_files",
    "format_report",
    "normalize_intervals",
    "parse_srt_intervals",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import compare_vad_srt as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
