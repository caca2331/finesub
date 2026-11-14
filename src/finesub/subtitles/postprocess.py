"""Post-process final translated SRT files."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from typing import Sequence

from .. import settings
from ..reporting import current_reporter
from ..text import t2s_converter
from . import line_groups
from .model import (
    SrtSegment,
    cue_key,
    format_srt_timestamp,
    parse_srt,
    render_srt,
    warn_on_invalid_srt,
    with_keys,
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

@dataclass(frozen=True)
class TimelineParams:
    """The two durations the timeline step works with.

    `end_extend_seconds`: push every cue end later by this much first (never
    past the next start or the audio end). `flash_gap_seconds`: then close
    leftover gaps shorter than this to the next start. Both come from
    `[postprocess]` in config.toml; defaults and bounds live in
    `finesub.settings`.

    Checked against those declarations however it is built: a caller passing
    one explicitly skips the config read, not the rules -- a negative pad would
    otherwise collapse cues to zero length without a word.
    """

    end_extend_seconds: float
    flash_gap_seconds: float

    def __post_init__(self) -> None:
        origin = " (explicit TimelineParams)"
        for name, setting in (
            ("end_extend_seconds", settings.END_EXTEND),
            ("flash_gap_seconds", settings.FLASH_GAP),
        ):
            # Frozen: the checked value (an int becomes a float) goes in this way.
            object.__setattr__(self, name, setting.check(getattr(self, name), origin=origin))

    @classmethod
    def from_config(cls) -> "TimelineParams":
        # One read, so the two values come from the same version of the file.
        config = settings.read()
        return cls(
            end_extend_seconds=settings.END_EXTEND.resolve(config=config),
            flash_gap_seconds=settings.FLASH_GAP.resolve(config=config),
        )


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
    # What the timeline step ran with; None when the profile has no such step.
    end_extend_seconds: float | None = None
    flash_gap_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def postprocess_srt_text(
    text: str,
    *,
    profile: int = DEFAULT_POSTPROCESS_PROFILE,
    audio_end: float | None = None,
    timeline: TimelineParams | None = None,
    keys: Sequence[str | None] | None = None,
) -> tuple[str, SrtPostprocessReport]:
    """Return post-processed SRT text plus a report.

    Profile ``-1`` is a semantic no-op re-render. Profile ``1`` adjusts the
    timeline, ``2`` normalizes punctuation, ``3`` converts 繁→简, ``4`` resolves
    overlapping cues, and ``0`` applies ``3``, ``4``, ``1``, ``2`` in that order.
    `timeline` defaults to `[postprocess]` in config.toml, read only when the
    profile has a timeline step; the report says which values it ran with.
    `keys` are the cues' group keys (`line_groups`), one per cue in order:
    the timeline steps keep cues apart within a group only. No step drops or
    moves a cue, so the same keys hold for the result.
    """

    if profile not in _PROFILE_STEPS:
        expected = ", ".join(str(item) for item in SUPPORTED_POSTPROCESS_PROFILES)
        raise ValueError(
            f"Unsupported SRT postprocess profile: {profile}; expected one of {expected}"
        )
    if DURATION_POSTPROCESS_PROFILE not in _PROFILE_STEPS[profile]:
        # Nothing here uses it, so the report must not claim it ran with one.
        timeline = None
    elif timeline is None:
        timeline = TimelineParams.from_config()

    segments = with_keys(parse_srt(text), keys)
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
                updated_segments, timeline, audio_end=audio_end
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
            end_extend_seconds=timeline.end_extend_seconds if timeline else None,
            flash_gap_seconds=timeline.flash_gap_seconds if timeline else None,
        ),
    )


def postprocess_srt_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    profile: int = DEFAULT_POSTPROCESS_PROFILE,
    validate: bool = True,
    audio_end: float | None = None,
    timeline: TimelineParams | None = None,
    keys: Sequence[str | None] | None = None,
) -> SrtPostprocessReport:
    """Rewrite one SRT in place (or to `output_path`) under `profile`.

    `validate=False` for an intermediate pass -- see `convert_json_to_srt`.
    `keys` as for `postprocess_srt_text`.
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
        audio_end=audio_end,
        timeline=timeline,
        keys=keys,
    )
    if validate:
        warn_on_invalid_srt(rendered, where=str(target), budget_stage="final-srt", keys=keys)
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
    updated = [replace(seg, text=converter.convert(seg.text)) for seg in segments]
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

    Within a group key only: two speakers of a cloud task talking at once are
    meant to be on screen at once.
    """

    overlaps_fixed = 0
    first_example = ""

    def trim(group: list[SrtSegment]) -> list[SrtSegment]:
        nonlocal overlaps_fixed, first_example
        updated: list[SrtSegment] = []
        for idx, segment in enumerate(group):
            new_end = segment.end
            if idx + 1 < len(group):
                next_start = group[idx + 1].start
                if new_end > next_start:
                    overlaps_fixed += 1
                    if not first_example:
                        first_example = (
                            f"#{segment.index} ends {format_srt_timestamp(segment.end)} "
                            f"but #{group[idx + 1].index} starts "
                            f"{format_srt_timestamp(next_start)}"
                        )
                    new_end = max(segment.start, next_start)
            updated.append(replace(segment, end=new_end))
        return updated

    updated = line_groups.in_place(segments, trim, key=cue_key)

    if overlaps_fixed:
        current_reporter().warning(
            "overlapping-cues-trimmed",
            f"{overlaps_fixed} overlapping subtitle cue(s) trimmed to the next "
            f"start (first: {first_example}). Overlaps are an upstream "
            "timeline defect, not a rendering choice.",
        )
    return updated, {"overlaps_fixed": overlaps_fixed}


def _bound_cues_to_audio_end(
    segments: Sequence[SrtSegment], audio_end: float
) -> list[SrtSegment]:
    """Keep tail text inside the media, reserving space backwards when needed (per group key)."""

    def bound(group: list[SrtSegment]) -> list[SrtSegment]:
        bounded = list(group)
        ceiling = audio_end
        # One SRT millisecond per displaced cue; normal cue timing is untouched.
        step = min(0.001, audio_end / max(len(bounded), 1))
        for index in range(len(bounded) - 1, -1, -1):
            cue = bounded[index]
            end = min(cue.end, ceiling)
            start = cue.start
            if start >= ceiling:
                start = max(0.0, end - step)
                ceiling = start
            else:
                ceiling = audio_end
            bounded[index] = replace(cue, start=start, end=end)
        return bounded

    return line_groups.in_place(segments, bound, key=cue_key)


def _end_caps(cues: Sequence[SrtSegment]) -> list[float | None]:
    """How far each cue's end may grow: the next cue's start. With group keys, the earlier of the next start in
    its own group and the earliest start at or after its end, wherever that cue sits in the list (postprocess
    keeps the cues where they are, `line_groups.in_place`, so the list need not be by start) -- growing never
    makes an overlap across groups that was not there."""

    parts = line_groups.groups([cue.key for cue in cues])
    if len(parts) == 1:
        return [cues[i + 1].start if i + 1 < len(cues) else None for i in range(len(cues))]
    caps: list[float | None] = [None] * len(cues)
    for idx in parts:
        for here, after in zip(idx, idx[1:]):
            caps[here] = cues[after].start
    starts = sorted((c.start, j) for j, c in enumerate(cues))
    for i, cue in enumerate(cues):
        at = bisect_left(starts, (cue.end, -1))
        if at < len(starts) and starts[at][1] == i:  # a zero-length cue is not its own bound
            at += 1
        clear = starts[at][0] if at < len(starts) else None
        if clear is not None:
            caps[i] = clear if caps[i] is None else min(caps[i], clear)
    return caps


def _postprocess_duration(
    segments: Sequence[SrtSegment],
    timeline: TimelineParams,
    *,
    audio_end: float | None = None,
) -> tuple[list[SrtSegment], dict[str, int]]:
    """Pad every cue end, then close flash gaps.

    ``audio_end`` bounds the pad. The last cue has no next start, so it used
    to be padded unconditionally -- which is how a stable JSON clamped exactly
    to the audio end came back out 0.3 s past it. ⚠ The bound is a ceiling,
    not a rule against padding the last cue: a last cue with room still gets
    its full pad, which is the behaviour every normal subtitle relies on.
    """

    updated = (
        _bound_cues_to_audio_end(segments, audio_end)
        if audio_end is not None else list(segments)
    )
    duration_extended = 0
    flash_extended = 0

    # 1. Extend every cue end by a fixed pad; never overlap the next start
    # (`_end_caps`), and never run past the audio.
    duration_fixed: list[SrtSegment] = []
    for segment, next_start in zip(updated, _end_caps(updated)):
        new_end = segment.end + timeline.end_extend_seconds
        if next_start is not None:
            new_end = min(new_end, next_start)
        if audio_end is not None:
            new_end = min(new_end, audio_end)
        if new_end > segment.end:
            duration_extended += 1
        duration_fixed.append(replace(segment, end=max(segment.start, new_end)))
    updated = duration_fixed

    # 2. Flash-axis cleanup: close tiny leftover gaps by extending the current
    # end to the next start (the same cap).
    flash_fixed: list[SrtSegment] = []
    for segment, next_start in zip(updated, _end_caps(updated)):
        new_end = segment.end
        if next_start is not None:
            gap = next_start - segment.end
            if 0.0 < gap < timeline.flash_gap_seconds:
                new_end = min(next_start, audio_end) if audio_end is not None else next_start
                flash_extended += 1
        flash_fixed.append(replace(segment, end=max(segment.start, new_end)))

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
        updated.append(replace(segment, text="\n".join(lines)))
    return (
        updated,
        {
            "punctuation_replacements": punctuation_replacements,
            "trimmed_lines": trimmed_lines,
        },
    )
