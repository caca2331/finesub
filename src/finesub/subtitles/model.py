"""SRT parsing, validation, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence

from .metrics import format_weighted_char_count, weighted_char_count


TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
@dataclass(frozen=True)
class SrtSegment:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SrtValidationResult:
    ok: bool
    segments: List[SrtSegment]
    errors: List[str]
    warnings: List[str]


def parse_srt_timestamp(value: str) -> float:
    hours = int(value[0:2])
    minutes = int(value[3:5])
    seconds = int(value[6:8])
    millis = int(value[9:12])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


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
    lines: List[str] = []
    for i, segment in enumerate(segments, start=1):
        idx = i if reindex else segment.index
        lines.append(str(idx))
        lines.append(
            f"{format_srt_timestamp(segment.start)} --> "
            f"{format_srt_timestamp(segment.end)}"
        )
        lines.extend(segment.text.splitlines() or ['""'])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def validate_srt_text(
    text: str,
    *,
    max_line_chars: int = 25,
    max_lines_per_segment: int = 2,
) -> SrtValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        segments = parse_srt(text)
    except ValueError as exc:
        return SrtValidationResult(ok=False, segments=[], errors=[str(exc)], warnings=[])

    previous_end = -1.0
    for segment in segments:
        if segment.end <= segment.start:
            errors.append(f"Segment {segment.index} end must be greater than start.")
        if segment.start < previous_end:
            errors.append(
                f"Segment {segment.index} starts before the previous segment ends."
            )
        previous_end = segment.end
        body_lines = segment.text.splitlines()
        if len(body_lines) > max_lines_per_segment:
            errors.append(
                f"Segment {segment.index} has more than {max_lines_per_segment} lines."
            )
        for line in body_lines:
            visual_len = weighted_char_count(line)
            if visual_len > max_line_chars:
                warnings.append(
                    f"Segment {segment.index} line has "
                    f"{format_weighted_char_count(visual_len)} weighted characters; "
                    f"limit is {max_line_chars}."
                )
    if not segments:
        errors.append("SRT contains no segments.")
    return SrtValidationResult(
        ok=not errors,
        segments=segments,
        errors=errors,
        warnings=warnings,
    )
