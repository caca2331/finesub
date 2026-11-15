"""SRT parsing, validation, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import List, Optional, Sequence, Tuple

from . import line_groups
from .metrics import weighted_char_count


TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
#: Layout limit the validator measures a line against (weighted characters).
DEFAULT_MAX_LINE_CHARS = 25


@dataclass(frozen=True)
class SrtSegment:
    index: int
    start: float
    end: float
    text: str
    #: The cue's group key (`line_groups`): not in the file, handed in beside it (`with_keys`).
    key: Optional[str] = None


def with_keys(segments: Sequence[SrtSegment], keys: Sequence[str | None] | None) -> List[SrtSegment]:
    """`segments` with one group key per cue, in order; None leaves them keyless."""
    if keys is None:
        return list(segments)
    if len(keys) != len(segments):
        raise ValueError(f"{len(keys)} group keys for {len(segments)} cues")
    return [replace(segment, key=key) for segment, key in zip(segments, keys)]


def cue_key(segment: SrtSegment) -> str | None:
    return segment.key


@dataclass(frozen=True)
class SrtValidationResult:
    ok: bool
    segments: List[SrtSegment]
    errors: List[str]
    #: `(segment index, weighted length of its longest line)` for every segment
    #: with a line over the budget. Data, not messages: how to report it
    #: depends on the stage (see `warn_on_invalid_srt`).
    over_budget: List[Tuple[int, float]]


def parse_srt_timestamp(value: str) -> float:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    millis = int(value[9:12])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _srt_ms(seconds: float) -> int:
    return max(0, int(round(float(seconds) * 1000.0)))


def _format_srt_ms(total_ms: int) -> str:
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    return _format_srt_ms(_srt_ms(seconds))


def srt_span_ms(start: float, end: float) -> tuple[int, int]:
    """A cue's ends in whole milliseconds, as the file will carry them.

    A span that is positive stays positive: one shorter than a millisecond
    would otherwise round to ``end == start``, a cue the validator rejects.
    A span that was not positive is left as it is -- that is an upstream
    defect for the validator to name, not one to paper over here.
    """

    start_ms, end_ms = _srt_ms(start), _srt_ms(end)
    if end > start and end_ms <= start_ms:
        end_ms = start_ms + 1
    return start_ms, end_ms


def has_text(text: object) -> bool:
    """Whether a cue has anything to show; one without is written as ``""`` or left out (`--empty-cues`)."""

    return bool(str(text or "").strip())


def srt_body_lines(text: str) -> List[str]:
    """A cue's text as the lines the file carries: never a blank one.

    A blank line ends a cue in SRT, so one left inside the text (a line the
    punctuation step emptied, say) would split the cue in two and leave the
    first half without text. No text at all is written as ``""``.
    """

    return [line.rstrip() for line in str(text).strip().splitlines() if line.strip()] or ['""']


def parse_srt(text: str) -> List[SrtSegment]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    lines = [line.rstrip() for line in normalized.splitlines()]
    segments: List[SrtSegment] = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        index = len(segments) + 1
        if (
            lines[i].strip().isdigit()
            and i + 1 < len(lines)
            and TIMESTAMP_RE.fullmatch(lines[i + 1].strip())
        ):
            index = int(lines[i].strip())
            i += 1
        if i >= len(lines):
            raise ValueError(f"SRT segment {index} is missing a timing line.")
        match = TIMESTAMP_RE.fullmatch(lines[i].strip())
        if not match:
            raise ValueError(f"SRT segment {index} has an invalid timing line.")
        i += 1
        body_lines: List[str] = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                break
            next_is_timing = i + 1 < len(lines) and TIMESTAMP_RE.fullmatch(lines[i + 1].strip())
            if (stripped.isdigit() and next_is_timing) or TIMESTAMP_RE.fullmatch(stripped):
                break
            body_lines.append(lines[i])
            i += 1
        if not body_lines:
            raise ValueError(f"SRT segment {index} is missing subtitle text.")
        segments.append(
            SrtSegment(
                index=index,
                start=parse_srt_timestamp(match.group("start")),
                end=parse_srt_timestamp(match.group("end")),
                text="\n".join(body_lines).strip(),
            )
        )
    return segments


def render_srt(segments: Sequence[SrtSegment], *, reindex: bool = True) -> str:
    """The SRT writer: every SRT file the pipeline writes goes through here.

    One cue per segment, none dropped (callers hand group keys in cue order);
    the text and times are made writable by `srt_body_lines` and `srt_span_ms`,
    so what comes out is what `parse_srt` reads back.
    """

    lines: List[str] = []
    for i, segment in enumerate(segments, start=1):
        idx = i if reindex else segment.index
        start_ms, end_ms = srt_span_ms(segment.start, segment.end)
        lines.append(str(idx))
        lines.append(f"{_format_srt_ms(start_ms)} --> {_format_srt_ms(end_ms)}")
        lines.extend(srt_body_lines(segment.text))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def validate_srt_text(
    text: str,
    *,
    max_line_chars: float = DEFAULT_MAX_LINE_CHARS,
    max_lines_per_segment: int = 2,
    keys: Sequence[str | None] | None = None,
) -> SrtValidationResult:
    """`keys`: each cue's group key; an overlap counts only within a group (two speakers of a cloud task
    talking at once are two cues on screen at once)."""
    errors: List[str] = []
    over_budget: List[Tuple[int, float]] = []
    try:
        segments = with_keys(parse_srt(text), keys)
    except ValueError as exc:
        return SrtValidationResult(
            ok=False, segments=[], errors=[str(exc)], over_budget=[]
        )

    group_of = {
        i: g for g, idx in enumerate(line_groups.groups([s.key for s in segments])) for i in idx
    }
    previous_end: dict[int, float] = {}
    for i, segment in enumerate(segments):
        if segment.end <= segment.start:
            errors.append(f"Segment {segment.index} end must be greater than start.")
        if segment.start < previous_end.get(group_of[i], -1.0):
            errors.append(
                f"Segment {segment.index} starts before the previous segment ends."
            )
        previous_end[group_of[i]] = segment.end
        body_lines = segment.text.splitlines()
        if len(body_lines) > max_lines_per_segment:
            errors.append(
                f"Segment {segment.index} has more than {max_lines_per_segment} lines."
            )
        longest = max((weighted_char_count(line) for line in body_lines), default=0.0)
        if longest > max_line_chars:
            over_budget.append((segment.index, longest))
    if not segments:
        errors.append("SRT contains no segments.")
    return SrtValidationResult(
        ok=not errors,
        segments=segments,
        errors=errors,
        over_budget=over_budget,
    )

def over_budget_buckets(
    over_budget: Sequence[Tuple[int, float]],
    *,
    max_line_chars: float = DEFAULT_MAX_LINE_CHARS,
    above: float | None = None,
) -> List[Tuple[str, List[int]]]:
    """Group over-budget segments into 5-character bands, worst band open-ended.

    With the default limit the bands are 26-30, 31-35, 36-40, 41-45 and 46+.
    Lengths are half-integers, so "26-30" holds everything above 25 up to and
    including 30. Empty bands are left out. `above` drops segments at or under
    that length first (a multiple of 5 keeps the bands whole: 35 starts at 36).
    """

    first = int(max_line_chars) + 1
    edges = [max_line_chars + step for step in (5, 10, 15, 20)]
    labels = [f"{first + 5 * n}-{first + 5 * n + 4}" for n in range(len(edges))]
    labels.append(f"{int(edges[-1]) + 1}+")
    bands: List[List[int]] = [[] for _ in labels]
    for index, length in over_budget:
        if above is not None and length <= above:
            continue
        band = next((n for n, edge in enumerate(edges) if length <= edge), len(edges))
        bands[band].append(index)
    return [(label, ids) for label, ids in zip(labels, bands) if ids]


def warn_on_invalid_srt(
    srt_text: str,
    *,
    where: str,
    budget_stage: str | None = None,
    budget_above: float = DEFAULT_MAX_LINE_CHARS,
    keys: Sequence[str | None] | None = None,
) -> None:
    """Report what the SRT validator finds, without blocking the write (`keys` as for `validate_srt_text`).

    Warn-only on purpose. The validator flags an empty file, a cue whose end is
    not after its start, an overlap and a segment over the line budget -- and
    only the second of those is unambiguously a broken cue. An empty result can
    be an honest "nobody spoke", an overlap has already been through
    `resolve_overlaps`, and the line budget is a layout policy the global DP
    splitter owns. Turning any of them into a hard gate would mean a run that
    produced two hours of usable subtitles writes nothing, which is a worse
    failure than the one being reported.

    Escalating `end <= start` to an error is the one change worth discussing
    later; it is left warn-only here so this wiring changes no behaviour.

    The line budget is not a warning at all. A 98-minute run put 168 raw lines
    over it -- nothing anyone could act on, and enough to bury the findings
    that matter. It is reported only when the caller names a stage
    (`budget_stage`), as one summary that lists the segment numbers per band.
    Raw ASR text is before any layout policy, so it asks for `budget_above`
    35: only the lines that are far over.
    """

    from ..reporting import current_reporter

    result = validate_srt_text(srt_text, keys=keys)
    for message in result.errors:
        current_reporter().warning(
            "srt-invalid", f"{where}: {message}"
        )
    if budget_stage is None or not result.over_budget:
        return
    bands = over_budget_buckets(result.over_budget, above=budget_above)
    if not bands:
        return
    current_reporter().summary(
        budget_stage,
        {
            f"单行超 {int(budget_above)} 字的段（编号，按段内最长行分档）": "；".join(
                f"{label} 字：{'、'.join(map(str, ids))}" for label, ids in bands
            )
        },
    )
