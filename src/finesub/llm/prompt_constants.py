"""The merge thresholds, defined once.

These numbers used to appear in three places at once -- the ``prompt_variants``
clause fields, the ``fragment_merge_rules_*`` files and the example notes --
which made changing a threshold an exercise in grepping for "20" (plan
``docs/llm_prompts.md``). Every occurrence now resolves to a
name here: Python callers import the constants, prompt fragments and example
material reference them through ``{thr:<name>}`` placeholders.

Adding a threshold means adding a name here, not another literal.
"""

from __future__ import annotations

from typing import Dict, Mapping

# Hard threshold: a merged line above either bound must normally be split back.
# Only a word or unsplittable fixed phrase cut across sources may be excepted,
# with a stated reason.
HARD_MERGE_CHARS = 20
HARD_MERGE_SECONDS = 4

# Absolute threshold: no exception, ever. Single sources are exempt from both --
# an upstream line that is already long is emitted as it stands.
ABSOLUTE_MERGE_CHARS = 36
ABSOLUTE_MERGE_SECONDS = 7

# The filler sandwich: the one legal three-source shape (two sentence fragments
# around a pure interjection). Stricter than the hard threshold and not eligible
# for the absolute-threshold relaxation.
FILLER_MAX_CHARS = 3
SANDWICH_MAX_CHARS = 16
SANDWICH_MAX_SECONDS = 4

# BasicA's single allowance: a word cut across two sources, spliced back.
SPLICE_MAX_SECONDS = 7


THRESHOLDS: Mapping[str, int] = {
    "hard_chars": HARD_MERGE_CHARS,
    "hard_seconds": HARD_MERGE_SECONDS,
    "absolute_chars": ABSOLUTE_MERGE_CHARS,
    "absolute_seconds": ABSOLUTE_MERGE_SECONDS,
    "filler_max_chars": FILLER_MAX_CHARS,
    "sandwich_max_chars": SANDWICH_MAX_CHARS,
    "sandwich_max_seconds": SANDWICH_MAX_SECONDS,
    "splice_max_seconds": SPLICE_MAX_SECONDS,
}


def threshold_params() -> Dict[str, str]:
    """``thr_<name>`` substitution values for prompt templates."""

    return {f"thr_{name}": str(value) for name, value in THRESHOLDS.items()}
