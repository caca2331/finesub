"""The merge thresholds, defined once and scaled by the task's length ratio.

These numbers used to appear in three places at once -- the ``prompt_variants``
clause fields, the ``fragment_merge_rules_*`` files and the example notes --
which made changing a threshold an exercise in grepping for "20" (plan
``docs/llm_prompts.md``). Every occurrence now resolves to a name here: Python
callers take a :class:`Thresholds`, prompt fragments and example material
reference it through ``${thr_<name>}`` / ``{thr:<name>}`` placeholders.

What a merged line may run to follows the task's length ratio (`length_tier`: ``--split-length-scale``,
the style's length tier, ``config.toml``): :meth:`Thresholds.at` scales the hard, absolute and
filler-sandwich bounds, and at 1.0 gives the values below exactly. What a thing *is*
(how short a filler is, a splice, how many sources a row may merge) does not scale.

Adding a threshold means adding a name to :data:`BASE` and deciding whether it
scales (:data:`SCALED_CHARS` / :data:`SCALED_SECONDS`), not another literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Dict, Mapping

#: The thresholds at ratio 1.0 (the standard tier).
BASE: Mapping[str, float] = MappingProxyType({
    # Hard threshold: a merged line above either bound must normally be split back. Only a word or
    # unsplittable fixed phrase cut across sources may be excepted, with a stated reason.
    "hard_chars": 20,
    "hard_seconds": 4,
    # Absolute threshold: no exception, ever. Single sources are exempt from both -- an upstream line
    # that is already long is emitted as it stands.
    "absolute_chars": 36,
    "absolute_seconds": 7,
    # The filler sandwich: the one legal three-source shape (two sentence fragments around a pure
    # interjection). Stricter than the hard threshold and not eligible for the absolute-threshold relaxation.
    "filler_max_chars": 3,
    "sandwich_max_chars": 16,
    "sandwich_max_seconds": 4,
    # BasicA's single allowance: a word cut across two sources, spliced back -- the two adjacent, so the gap
    # before the second one is at most this.
    "splice_max_seconds": 7,
    "splice_max_gap_seconds": 0.3,
    # The single pass capableC judges who is speaking itself and merges raw ASR lines: a subtitle may join
    # up to this many, as many as three passes' pass 2 can reach (a two-line join plus a mergeable seam).
    "fused_max_sources": 4,
})

#: The bounds that follow the ratio: characters round half up to a whole number, seconds to 0.1.
SCALED_CHARS = frozenset({"hard_chars", "absolute_chars", "sandwich_max_chars"})
SCALED_SECONDS = frozenset({"hard_seconds", "absolute_seconds", "sandwich_max_seconds"})


def _scaled(value: float, ratio: float, quantum: str) -> float:
    exact = Decimal(str(value)) * Decimal(str(ratio))
    rounded = exact.quantize(Decimal(quantum), rounding=ROUND_HALF_UP)
    # a whole number renders as today does (`5`, not `5.0`)
    return int(rounded) if rounded == rounded.to_integral_value() else float(rounded)


@dataclass(frozen=True)
class Thresholds:
    """The thresholds at one length ratio. Build with :meth:`at`."""

    scale: float
    values: Mapping[str, float] = field(repr=False)

    @classmethod
    def at(cls, scale: float | None) -> "Thresholds":
        """The thresholds at ``scale``. ``None`` -- a window no path wrote its task's ratio into -- is an
        error, never a silent 1.0."""

        if scale is None:
            raise ValueError("length scale is not set: the task's ratio was never written into this window")
        if not scale > 0:
            raise ValueError(f"length scale must be positive, got {scale!r}")
        values = {}
        for name, value in BASE.items():
            if name in SCALED_CHARS:
                value = _scaled(value, scale, "1")
            elif name in SCALED_SECONDS:
                value = _scaled(value, scale, "0.1")
            values[name] = value
        return cls(scale=float(scale), values=MappingProxyType(values))

    def __getattr__(self, name: str) -> float:
        try:
            return self.__dict__["values"][name]
        except KeyError:
            raise AttributeError(name) from None

    def params(self) -> Dict[str, str]:
        """``thr_<name>`` substitution values for prompt templates."""

        return {f"thr_{name}": str(value) for name, value in self.values.items()}


#: Pass 1's prompt is the same at every ratio, written at the standard tier's numbers: only the script's two
#: checks before pass 2 follow the task's ratio (`grouping.group_lines`).
PASS1 = Thresholds.at(1.0)
