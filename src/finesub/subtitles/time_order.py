"""Is this transcript in the order the thing that reads it will emit?

Nothing between the producers and the subtitle writers ever checked it. The
global DP resegmentation, the stabilization rewrite and the LLM correction pass
all reorder or replace segments, and a producer bug travels all the way to a
downstream complaint about the delivered SRT.

**Which timestamp orders the output is the whole subtlety, and this package has
two writers that answer differently.** ``render_segment_srt`` builds every cue
from ``seg["start"]`` / ``seg["end"]`` and drops empty text; ``render_word_srt``
builds every cue from the word timestamps and never looks at the span. A guard
written against spans passes a list whose *words* run backwards, which is
exactly the shape that ships a subtitle jumping back in time -- so the caller
has to say which quantity its consumer dereferences, and an artifact read by
both is checked for both.

Empty-text items are skipped rather than treated as position 0: the writers drop
them, so counting them would fake a backward jump on whatever follows.

Pure and dependency-free apart from the timestamp formatter, so it can sit at a
JSON chokepoint as easily as at a rendering one.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, NamedTuple, Sequence

from .model import format_srt_timestamp

Quantity = Literal["words", "spans"]


class Backward(NamedTuple):
    """The first place the timeline runs backwards."""

    index: int
    previous_sec: float
    current_sec: float
    quantity: Quantity

    def describe(self) -> str:
        return (
            f"item {self.index} starts at {format_srt_timestamp(self.current_sec)}, "
            f"before {format_srt_timestamp(self.previous_sec)} already emitted "
            f"({self.quantity})"
        )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text_of(item: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _words_of(segment: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    words = segment.get("words")
    if not isinstance(words, list):
        return ()
    return [word for word in words if isinstance(word, Mapping)]


def first_backward(
    segments: Sequence[Mapping[str, Any]],
    *,
    using: Quantity,
) -> Backward | None:
    """The first backward step under `using`, or None when the order holds.

    `using="spans"` mirrors `render_segment_srt`; `using="words"` mirrors
    `render_word_srt`. Naming the quantity is the point: the two can disagree,
    and only the one the consumer reads says anything about what ships.

    A `words` check on a segment that carries none is not an assertion about
    that segment -- it is skipped, the same way the word writer skips it.
    """

    previous: float | None = None
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            continue
        if using == "spans":
            if not _text_of(segment, "text"):
                continue
            start = _number(segment.get("start"))
            if start is None:
                continue
            if previous is not None and start < previous:
                return Backward(index, previous, start, using)
            previous = start
            continue

        for word in _words_of(segment):
            if not _text_of(word, "word", "text"):
                continue
            start = _number(word.get("start"))
            if start is None:
                continue
            if previous is not None and start < previous:
                return Backward(index, previous, start, using)
            previous = start
    return None


def report_backward(
    segments: Sequence[Mapping[str, Any]],
    *,
    using: Quantity,
    where: str,
) -> Backward | None:
    """`first_backward`, and a warning when it finds one.

    Warns rather than raises on purpose: a subtitle that is out of order is
    still a subtitle, and the producer bug it points at is upstream of whoever
    is holding the file. Surfacing it is the whole job -- this class of defect
    used to reach the user as "some lines jump backwards" and nothing else.
    """

    backward = first_backward(segments, using=using)
    if backward is None:
        return None
    from ..reporting import current_reporter

    current_reporter().warning(
        "timeline-out-of-order",
        f"{where}: {backward.describe()}",
        impact="cues built from this run backwards",
        action="the producer reordered or appended out of place; check it rather than the writer",
    )
    return backward
