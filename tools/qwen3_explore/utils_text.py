"""Punctuation classes, matching `src/utils/text.py: punct_class()` closely enough for analysis.

Kept local so the exploration scripts run in the `qwen-asr` env, which does not have the
production package importable.
"""

from __future__ import annotations

SENTENCE = set("。．｡.!！?？…‥‼⁇⁈⁉")
CLAUSE = set("、，,､；;：:")


def punct_kind(ch: str) -> str | None:
    if ch in SENTENCE:
        return "sentence"
    if ch in CLAUSE:
        return "clause"
    return None
