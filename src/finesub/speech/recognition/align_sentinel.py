"""Did the word timeline collapse? A check on what alignment actually produced.

Alignment does not fail loudly. A run whose word timestamps are degenerate --
every word at (0, 0), or a whole transcript squeezed into half a second --
returns success, writes a well-formed artifact, and renders a subtitle. The text
is right and the timing is garbage, which is the worst shape a defect can take:
nothing downstream can tell it apart from a correct run, and the first anyone
hears of it is that the subtitles do not line up.

So this looks at the produced timestamps rather than at the code that produced
them. Five independent signals, each with the precondition that keeps it from
firing on a legitimately small transcript. Any one of them is enough: they fail
in different ways and a run that trips one is already not worth trusting.

**Report only.** There is a plausible repair -- redistribute the words across
the segment span in proportion to their length -- and it is deliberately not
implemented here. We have never measured how often this fires in production,
and an automatic repair would replace one invisible wrong answer with another.
Surfacing it is the whole job.

Two details that decide whether the thing works at all:

* **Code points, not bytes.** A byte count reads ~3x high on CJK, so a
  bytes-based characters-per-second would fire on every correct Japanese
  alignment.
* **The timeline here is absolute.** `write_aligned_json` is downstream of the
  per-group offset arithmetic, so a word at (0, 0) really is at the origin
  rather than at the start of some chunk. A checker placed upstream of that
  arithmetic would have to subtract the offset first, or a silent zero at
  0:30 would present as (30.0, 30.0) -- a plausible-looking position that
  never trips anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: Words pinned at exactly (0, 0). A handful already means the aligner walked
#: off its own path; this is not a "mostly fine" threshold.
MAX_ZERO_POSITION_FRACTION = 0.10

#: Spans with end <= start. Looser than the zero-position share because a few
#: zero-width words are an ordinary rounding artifact on short tokens.
MAX_ZERO_LENGTH_FRACTION = 0.40

#: ~15-20 is fast Latin speech, ~10 is everyday Japanese. 50 is not a rate any
#: writing system is spoken at, so it means the span, not the speaker.
MAX_CHARS_PER_SECOND = 50.0

#: Transcript span as a fraction of the audio. A real transcript of a long
#: recording covers far more than this even with long silences.
MIN_COVERAGE = 0.05

#: Below this the span is too short to say anything -- unless the audio is much
#: longer, which is the case this catches.
MIN_SPAN_SECONDS = 0.5
SPAN_AUDIO_RATIO = 4.0

#: ...but "much longer" has to mean long in absolute terms too. Without this
#: floor a perfectly ordinary two-second clip carrying one 0.4s word satisfies
#: `audio >= 4 * span` and gets reported, which would make the sentinel fire on
#: half the fixtures in the tree. The signal is "long audio, no timeline"; it
#: needs the audio to actually be long.
MIN_AUDIO_FOR_SPAN_SIGNAL = 10.0

#: Signals that would fire on any short transcript are gated on there being
#: enough text to be talking about.
MIN_CHARACTERS = 10


@dataclass(frozen=True)
class Collapse:
    """Which signal fired, and the numbers it fired on."""

    signal: str
    detail: str

    def describe(self) -> str:
        return f"{self.signal}: {self.detail}"


def _code_points(text: str) -> int:
    return len(text)


def _words(segments: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        out.extend(word for word in words if isinstance(word, Mapping))
    return out


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _word_text(word: Mapping[str, Any]) -> str:
    for key in ("word", "text"):
        value = word.get(key)
        if isinstance(value, str):
            return value
    return ""


def inspect(
    segments: Sequence[Mapping[str, Any]],
    *,
    audio_duration: float | None = None,
) -> Collapse | None:
    """The first signal that fires, or None when the timeline looks sane.

    `audio_duration` is optional: the two signals that need it are skipped when
    it is unknown rather than guessed at, the same way each other signal is
    skipped when its precondition is not met.
    """

    words = _words(segments)
    if not words:
        return None

    starts: list[float] = []
    ends: list[float] = []
    characters = 0
    zero_position = 0
    zero_length = 0
    counted = 0

    for word in words:
        text = _word_text(word)
        start = _number(word.get("start"))
        end = _number(word.get("end"))
        if start is None or end is None:
            continue
        counted += 1
        characters += _code_points(text.strip())
        starts.append(start)
        ends.append(end)
        if start == 0.0 and end == 0.0:
            zero_position += 1
        if end <= start:
            zero_length += 1

    if not counted:
        return None

    zero_position_share = zero_position / counted
    if zero_position_share > MAX_ZERO_POSITION_FRACTION:
        return Collapse(
            "zero-position words",
            f"{zero_position}/{counted} ({zero_position_share:.3f} > "
            f"{MAX_ZERO_POSITION_FRACTION})",
        )

    zero_length_share = zero_length / counted
    if zero_length_share > MAX_ZERO_LENGTH_FRACTION:
        return Collapse(
            "zero-length spans",
            f"{zero_length}/{counted} ({zero_length_share:.3f} > "
            f"{MAX_ZERO_LENGTH_FRACTION})",
        )

    span = max(ends) - min(starts)

    if characters >= MIN_CHARACTERS and span > 0.0:
        rate = characters / span
        if rate > MAX_CHARS_PER_SECOND:
            return Collapse(
                "characters per second",
                f"{characters} code points over {span:.3f}s = {rate:.1f} > "
                f"{MAX_CHARS_PER_SECOND}",
            )

    if audio_duration is not None and audio_duration > 0.0:
        if characters >= MIN_CHARACTERS:
            coverage = span / audio_duration
            if coverage < MIN_COVERAGE:
                return Collapse(
                    "coverage",
                    f"{span:.3f}s of {audio_duration:.3f}s = {coverage:.4f} < "
                    f"{MIN_COVERAGE}",
                )
        if (
            span < MIN_SPAN_SECONDS
            and audio_duration >= MIN_AUDIO_FOR_SPAN_SIGNAL
            and audio_duration >= SPAN_AUDIO_RATIO * span
        ):
            return Collapse(
                "span",
                f"{span:.3f}s < {MIN_SPAN_SECONDS}s while the audio runs "
                f"{audio_duration:.3f}s",
            )

    return None


def report(
    segments: Sequence[Mapping[str, Any]],
    *,
    audio_duration: float | None,
    where: str,
) -> Collapse | None:
    """`inspect`, and a warning when it finds something."""

    collapse = inspect(segments, audio_duration=audio_duration)
    if collapse is None:
        return None
    from ...reporting import current_reporter

    current_reporter().warning(
        "alignment-collapsed",
        f"{where}: {collapse.describe()}",
        impact="the text may be right while the timestamps are not",
        action="check the word timeline before using this artifact; it is not repaired here",
    )
    return collapse
