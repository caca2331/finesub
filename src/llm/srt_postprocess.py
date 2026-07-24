"""Post-process final translated SRT files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .srt_utils import SrtSegment, parse_srt, render_srt
from .subtitle_metrics import weighted_char_count


DEFAULT_POSTPROCESS_PROFILE = 0
DURATION_POSTPROCESS_PROFILE = 1
PUNCTUATION_POSTPROCESS_PROFILE = 2
T2S_POSTPROCESS_PROFILE = 3
NO_POSTPROCESS_PROFILE = -1
SUPPORTED_POSTPROCESS_PROFILES = (
    NO_POSTPROCESS_PROFILE,
    DEFAULT_POSTPROCESS_PROFILE,
    DURATION_POSTPROCESS_PROFILE,
    PUNCTUATION_POSTPROCESS_PROFILE,
    T2S_POSTPROCESS_PROFILE,
)

_PROFILE_STEPS = {
    NO_POSTPROCESS_PROFILE: (),
    DEFAULT_POSTPROCESS_PROFILE: (
        T2S_POSTPROCESS_PROFILE,
        DURATION_POSTPROCESS_PROFILE,
        PUNCTUATION_POSTPROCESS_PROFILE,
    ),
    DURATION_POSTPROCESS_PROFILE: (DURATION_POSTPROCESS_PROFILE,),
    PUNCTUATION_POSTPROCESS_PROFILE: (PUNCTUATION_POSTPROCESS_PROFILE,),
    T2S_POSTPROCESS_PROFILE: (T2S_POSTPROCESS_PROFILE,),
}

DYNAMIC_DURATION_MIN_SECONDS = 0.6
DYNAMIC_DURATION_MAX_SECONDS = 1.2
DYNAMIC_SECONDS_PER_WEIGHTED_CHAR = 0.1
FLASH_GAP_SECONDS = 0.2

# If opencc t2s changes more than this fraction of characters, the text is
# considered traditional Chinese and the simplified conversion is applied.
T2S_DIFF_THRESHOLD = 0.15

_opencc_t2s: Any = None


def _get_t2s_converter() -> Any:
    """Lazy-init opencc t2s converter; None if unavailable."""
    global _opencc_t2s
    if _opencc_t2s is None:
        try:
            from opencc import OpenCC

            _opencc_t2s = OpenCC("t2s")
        except ImportError:
            _opencc_t2s = False  # type: ignore[assignment]
    return _opencc_t2s or None


@dataclass(frozen=True)
class SrtPostprocessReport:
    profile: int
    input_path: str
    output_path: str
    segment_count: int
    applied_profiles: tuple[int, ...] = ()
    t2s_converted: bool = False
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
    timeline, profile ``2`` normalizes punctuation, and profile ``0`` applies
    profiles ``1`` then ``2``.
    """

    if profile not in _PROFILE_STEPS:
        expected = ", ".join(str(item) for item in SUPPORTED_POSTPROCESS_PROFILES)
        raise ValueError(
            f"Unsupported SRT postprocess profile: {profile}; expected one of {expected}"
        )

    segments = parse_srt(text)
    updated_segments = list(segments)
    t2s_converted = False
    duration_extended = 0
    flash_extended = 0
    punctuation_replacements = 0
    trimmed_lines = 0

    for step in _PROFILE_STEPS[profile]:
        if step == T2S_POSTPROCESS_PROFILE:
            updated_segments, t2s_converted = _postprocess_t2s(updated_segments)
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
) -> SrtPostprocessReport:
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
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
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
    converter = _get_t2s_converter()
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


def _postprocess_duration(
    segments: Sequence[SrtSegment],
) -> tuple[list[SrtSegment], dict[str, int]]:
    updated = list(segments)
    duration_extended = 0
    flash_extended = 0

    # 1. Give subtitles at most 1.2s long a content-aware minimum duration.
    # Never shorten an existing subtitle or overlap the next one.
    duration_fixed: list[SrtSegment] = []
    for idx, segment in enumerate(updated):
        next_start = updated[idx + 1].start if idx + 1 < len(updated) else None
        current_duration = max(0.0, segment.end - segment.start)
        new_end = segment.end
        if current_duration <= DYNAMIC_DURATION_MAX_SECONDS:
            target_duration = _dynamic_target_duration(segment.text)
            new_end = max(new_end, segment.start + target_duration)
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

    # 2. Flash-axis cleanup: close tiny gaps by extending the current end to
    # the next start.
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


def _dynamic_target_duration(text: str) -> float:
    """Return the content-aware minimum duration for a short subtitle."""

    return min(
        DYNAMIC_DURATION_MAX_SECONDS,
        max(
            DYNAMIC_DURATION_MIN_SECONDS,
            weighted_char_count(text) * DYNAMIC_SECONDS_PER_WEIGHTED_CHAR,
        ),
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
