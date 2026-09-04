"""Post-process final translated SRT files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Sequence

from ..reporting import current_reporter
from ..text import t2s_converter
from .model import (
    SrtSegment,
    format_srt_timestamp,
    parse_srt,
    render_srt,
    warn_on_invalid_srt,
)


DEFAULT_POSTPROCESS_PROFILE = 0
DURATION_POSTPROCESS_PROFILE = 1
PUNCTUATION_POSTPROCESS_PROFILE = 2
T2S_POSTPROCESS_PROFILE = 3
OVERLAP_POSTPROCESS_PROFILE = 4
NO_POSTPROCESS_PROFILE = -1
SUPPORTED_POSTPROCESS_PROFILES = (
    NO_POSTPROCESS_PROFILE,
    DEFAULT_POSTPROCESS_PROFILE,
    DURATION_POSTPROCESS_PROFILE,
    PUNCTUATION_POSTPROCESS_PROFILE,
    T2S_POSTPROCESS_PROFILE,
    OVERLAP_POSTPROCESS_PROFILE,
)

# The timeline-only policy, in the order it has to run: resolve overlaps first,
# because the duration step's "never past the next start" cap would otherwise
# silently *shorten* an overlapping cue and report it as an extension. Profile
# ``0`` embeds this, and the ``*-raw.srt`` export applies it directly -- that
# file must keep the ASR text untouched, so it can take the timeline half only.
# Single source of truth so the two paths cannot drift apart.
TIMELINE_POSTPROCESS_PROFILES = (
    OVERLAP_POSTPROCESS_PROFILE,
    DURATION_POSTPROCESS_PROFILE,
)

_PROFILE_STEPS = {
    NO_POSTPROCESS_PROFILE: (),
    DEFAULT_POSTPROCESS_PROFILE: (
        T2S_POSTPROCESS_PROFILE,
        *TIMELINE_POSTPROCESS_PROFILES,
        PUNCTUATION_POSTPROCESS_PROFILE,
    ),
    DURATION_POSTPROCESS_PROFILE: (DURATION_POSTPROCESS_PROFILE,),
    PUNCTUATION_POSTPROCESS_PROFILE: (PUNCTUATION_POSTPROCESS_PROFILE,),
    T2S_POSTPROCESS_PROFILE: (T2S_POSTPROCESS_PROFILE,),
    OVERLAP_POSTPROCESS_PROFILE: (OVERLAP_POSTPROCESS_PROFILE,),
}

# Push every cue end later by this amount first (never past the next start).
END_EXTEND_SECONDS = 0.3
# Then close leftover gaps shorter than this to the next start.
FLASH_GAP_SECONDS = 0.3

# If opencc t2s changes more than this fraction of characters, the text is
# considered traditional Chinese and the simplified conversion is applied.
T2S_DIFF_THRESHOLD = 0.15


@dataclass(frozen=True)
class SrtPostprocessReport:
    profile: int
    input_path: str
    output_path: str
    segment_count: int
    applied_profiles: tuple[int, ...] = ()
    t2s_converted: bool = False
    overlaps_fixed: int = 0
    duration_extended: int = 0
    flash_extended: int = 0
    punctuation_replacements: int = 0
    trimmed_lines: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def postprocess_srt_text(
    text: str,
    *,
    profile: int = DEFAULT_POSTPROCESS_PROFILE,
) -> tuple[str, SrtPostprocessReport]:
    """Return post-processed SRT text plus a report.

    Profile ``-1`` is a semantic no-op re-render. Profile ``1`` adjusts the
    timeline, ``2`` normalizes punctuation, ``3`` converts 繁→简, ``4`` resolves
    overlapping cues, and ``0`` applies ``3``, ``4``, ``1``, ``2`` in that order.
    """

    if profile not in _PROFILE_STEPS:
        expected = ", ".join(str(item) for item in SUPPORTED_POSTPROCESS_PROFILES)
        raise ValueError(
            f"Unsupported SRT postprocess profile: {profile}; expected one of {expected}"
        )

    segments = parse_srt(text)
    updated_segments = list(segments)
    t2s_converted = False
    overlaps_fixed = 0
    duration_extended = 0
    flash_extended = 0
    punctuation_replacements = 0
    trimmed_lines = 0

    for step in _PROFILE_STEPS[profile]:
        if step == T2S_POSTPROCESS_PROFILE:
            updated_segments, t2s_converted = _postprocess_t2s(updated_segments)
        elif step == OVERLAP_POSTPROCESS_PROFILE:
            updated_segments, overlap_report = _postprocess_overlap(updated_segments)
            overlaps_fixed += overlap_report["overlaps_fixed"]
        elif step == DURATION_POSTPROCESS_PROFILE:
            updated_segments, duration_report = _postprocess_duration(
                updated_segments
            )
            duration_extended += duration_report["duration_extended"]
            flash_extended += duration_report["flash_extended"]
        elif step == PUNCTUATION_POSTPROCESS_PROFILE:
            updated_segments, punctuation_report = _postprocess_punctuation(
                updated_segments
            )
            punctuation_replacements += punctuation_report[
                "punctuation_replacements"
            ]
            trimmed_lines += punctuation_report["trimmed_lines"]

    rendered = render_srt(updated_segments)
    return (
        rendered,
        SrtPostprocessReport(
            profile=profile,
            input_path="",
            output_path="",
            segment_count=len(updated_segments),
            applied_profiles=_PROFILE_STEPS[profile],
            t2s_converted=t2s_converted,
            overlaps_fixed=overlaps_fixed,
            duration_extended=duration_extended,
            flash_extended=flash_extended,
            punctuation_replacements=punctuation_replacements,
            trimmed_lines=trimmed_lines,
        ),
    )


def postprocess_srt_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    profile: int = DEFAULT_POSTPROCESS_PROFILE,
    validate: bool = True,
) -> SrtPostprocessReport:
    """Rewrite one SRT in place (or to `output_path`) under `profile`.

    `validate=False` for an intermediate pass -- see `convert_json_to_srt`.
    """

    source = Path(input_path).expanduser().resolve()
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source
    )
    rendered, report = postprocess_srt_text(
        source.read_text(encoding="utf-8"),
        profile=profile,
    )
    if validate:
        warn_on_invalid_srt(rendered, where=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.part{target.suffix}")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, target)
    return SrtPostprocessReport(
        **{
            **report.to_dict(),
            "input_path": str(source),
            "output_path": str(target),
        }
    )


def _postprocess_t2s(
    segments: Sequence[SrtSegment],
) -> tuple[list[SrtSegment], bool]:
    """Detect traditional Chinese output and convert to simplified.

    Joins all segment text, applies opencc t2s, and compares character-level
    difference. If the diff ratio exceeds T2S_DIFF_THRESHOLD the text is
    considered traditional and all segments are replaced with the simplified
    conversion. Otherwise segments are returned unchanged.
    """
    converter = t2s_converter()
    if converter is None or not segments:
        return list(segments), False

    full_text = "\n".join(seg.text for seg in segments)
    converted = converter.convert(full_text)
    if converted == full_text:
        return list(segments), False

    # Character-level diff ratio (simple positional comparison).
    total = max(len(full_text), 1)
    diff_chars = sum(1 for a, b in zip(full_text, converted) if a != b)
    diff_chars += abs(len(full_text) - len(converted))
    if diff_chars / total < T2S_DIFF_THRESHOLD:
        return list(segments), False

    # Apply conversion per-segment (preserving line structure).
    updated = [
        SrtSegment(
            index=seg.index,
            start=seg.start,
            end=seg.end,
            text=converter.convert(seg.text),
        )
        for seg in segments
    ]
    return updated, True


def _postprocess_overlap(
    segments: Sequence[SrtSegment],
) -> tuple[list[SrtSegment], dict[str, int]]:
    """Pull an overlapping cue's end back to the next cue's start, and say so.

    Two cues on screen at once is never intentional output here, so an overlap
    means something upstream produced a broken timeline. The repair is cheap and
    unambiguous -- the later cue's start wins, the earlier one ends there -- but
    it must not be silent, because the timeline step that follows would clamp
    the same ends to the same place while reporting them as extensions.

    Only adjacent pairs are compared, which is exhaustive for the sorted input
    every producer here emits. Unsorted input (a next start before the current
    start) collapses the cue to zero length rather than inverting it; that shows
    up in the warning count too.
    """

    updated: list[SrtSegment] = []
    overlaps_fixed = 0
    first_example = ""
    for idx, segment in enumerate(segments):
        new_end = segment.end
        if idx + 1 < len(segments):
            next_start = segments[idx + 1].start
            if new_end > next_start:
                overlaps_fixed += 1
                if not first_example:
                    first_example = (
                        f"#{segment.index} ends {format_srt_timestamp(segment.end)} "
                        f"but #{segments[idx + 1].index} starts "
                        f"{format_srt_timestamp(next_start)}"
                    )
                new_end = max(segment.start, next_start)
        updated.append(
            SrtSegment(
                index=segment.index,
                start=segment.start,
                end=new_end,
                text=segment.text,
            )
        )

    if overlaps_fixed:
        current_reporter().warning(
            "overlapping-cues-trimmed",
            f"{overlaps_fixed} overlapping subtitle cue(s) trimmed to the next "
            f"start (first: {first_example}). Overlaps are an upstream "
            "timeline defect, not a rendering choice.",
        )
    return updated, {"overlaps_fixed": overlaps_fixed}


def _postprocess_duration(
    segments: Sequence[SrtSegment],
) -> tuple[list[SrtSegment], dict[str, int]]:
    updated = list(segments)
    duration_extended = 0
    flash_extended = 0

    # 1. Extend every cue end by a fixed pad; never overlap the next start.
    duration_fixed: list[SrtSegment] = []
    for idx, segment in enumerate(updated):
        next_start = updated[idx + 1].start if idx + 1 < len(updated) else None
        new_end = segment.end + END_EXTEND_SECONDS
        if next_start is not None:
            new_end = min(new_end, next_start)
        if new_end > segment.end:
            duration_extended += 1
        duration_fixed.append(
            SrtSegment(
                index=segment.index,
                start=segment.start,
                end=max(segment.start, new_end),
                text=segment.text,
            )
        )
    updated = duration_fixed

    # 2. Flash-axis cleanup: close tiny leftover gaps by extending the current
    # end to the next start.
    flash_fixed: list[SrtSegment] = []
    for idx, segment in enumerate(updated):
        new_end = segment.end
        if idx + 1 < len(updated):
            next_start = updated[idx + 1].start
            gap = next_start - segment.end
            if 0.0 < gap < FLASH_GAP_SECONDS:
                new_end = next_start
                flash_extended += 1
        flash_fixed.append(
            SrtSegment(
                index=segment.index,
                start=segment.start,
                end=max(segment.start, new_end),
                text=segment.text,
            )
        )

    return (
        flash_fixed,
        {
            "duration_extended": duration_extended,
            "flash_extended": flash_extended,
        },
    )


def _postprocess_punctuation(
    segments: Sequence[SrtSegment],
) -> tuple[list[SrtSegment], dict[str, int]]:
    punctuation_replacements = 0
    trimmed_lines = 0
    updated: list[SrtSegment] = []
    for segment in segments:
        lines: list[str] = []
        for line in segment.text.splitlines() or [""]:
            replaced = line
            for char in ("，", "。", "　"):
                count = replaced.count(char)
                punctuation_replacements += count
                if count:
                    replaced = replaced.replace(char, " ")
            trimmed = replaced.strip()
            if trimmed != replaced:
                trimmed_lines += 1
            lines.append(trimmed)
        updated.append(
            SrtSegment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                text="\n".join(lines),
            )
        )
    return (
        updated,
        {
            "punctuation_replacements": punctuation_replacements,
            "trimmed_lines": trimmed_lines,
        },
    )
