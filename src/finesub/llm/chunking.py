"""Window planning for context collection and correction translation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field, replace
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from finesub.subtitles.rendering import format_srt_time

from finesub.media.clips import compute_clip_range
from .routing.config import (
    DEFAULT_LIMITS,
    OVERLAP_WINDOW_SECONDS,
    PRECEDING_CONTEXT_MAX_SEGMENTS,
    ModelLimits,
    effective_window_subtitle_cap,
)
from .routing.profiles import (
    DEFAULT_PROFILE,
    TranslationProfile,
    video_tokens_per_second,
    window_output_budget,
)
from .token_budget import (
    CorrectionBudget,
    TokenBudgetError,
    TokenCounter,
    build_correction_budget,
    default_token_counter,
    validate_correction_budget,
)


def media_tokens_per_second(
    profile: TranslationProfile, limits: ModelLimits = DEFAULT_LIMITS
) -> float:
    """Clip media token rate for planning: audio and/or video, 0 for text.

    Window planning budgets the correction call, so the rate follows
    ``correction_media`` (model-routing v2); the query round's clip never enters the
    window geometry.
    """

    rate = 0.0
    if profile.correction_use_audio:
        rate += limits.audio_tokens_per_second
    if profile.correction_use_video:
        rate += video_tokens_per_second(
            high_resolution=limits.video_high_resolution
        )
    return rate


SENTENCE_END_RE = re.compile(r"[。！？!?\.…]+[」』）)\]】》”\"']*$")


@dataclass(frozen=True)
class SubtitleSegment:
    id: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleWindow:
    chunk_id: str
    segments: List[SubtitleSegment]
    overlap_segments: List[SubtitleSegment]
    boundary_reason: str
    budget: CorrectionBudget
    # Audio clip range attached to this window's calls; CSV local times are
    # rendered relative to clip_start (the clip's 0 second).
    clip_start: float = 0.0
    clip_end: float = 0.0
    # Read-only raw ASR lines just before the window (v13): background for
    # conversational continuity, never translated, not part of the clip range;
    # rendered clip-relative so their times are mostly negative.
    preceding_segments: List[SubtitleSegment] = field(default_factory=list)

    @property
    def split_depth(self) -> int:
        """How many times this window has already been halved.

        Read off the id rather than tracked beside it: `split_window_in_half`
        suffixes `-a` / `-b`, so the id IS the lineage and cannot drift out of
        sync with a counter (`0007-a-b` is the second half of the first half).

        Lives on the window because two unrelated layers need it -- the retry
        loop's split budget and the output validator's discard-ratio gate --
        and a second copy of the rule in either of them would be exactly the
        silent drift this repo keeps getting bitten by.
        """

        return self.chunk_id.count("-")

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def source_ids(self) -> List[str]:
        return [segment.id for segment in self.segments]


@dataclass(frozen=True)
class WindowIdMap:
    """Model-facing local ids for one execution window.

    Stable/source ids remain the harness's canonical identity.  A model sees
    target rows as ``1..N`` and read-only preceding-context rows as
    ``1-M..0`` (chronological order, so the nearest reference is always 0).
    """

    source_ids: tuple[str, ...]
    preceding_source_ids: tuple[str, ...] = ()

    @classmethod
    def from_window(cls, window: SubtitleWindow) -> "WindowIdMap":
        return cls(
            source_ids=tuple(window.source_ids),
            preceding_source_ids=tuple(
                segment.id for segment in window.preceding_segments
            ),
        )

    @classmethod
    def from_segments(
        cls, segments: Sequence[SubtitleSegment]
    ) -> "WindowIdMap":
        return cls(source_ids=tuple(segment.id for segment in segments))

    @staticmethod
    def _with_ids(
        segments: Sequence[SubtitleSegment], ids: Sequence[str]
    ) -> List[SubtitleSegment]:
        if len(segments) != len(ids):
            raise ValueError("Segment/id counts must match when localizing a window.")
        return [
            replace(segment, id=local_id)
            for segment, local_id in zip(segments, ids)
        ]

    def localize_segments(
        self, segments: Sequence[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        return self._with_ids(
            segments, tuple(str(index) for index in range(1, len(segments) + 1))
        )

    def localize_preceding_segments(
        self, segments: Sequence[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        count = len(segments)
        return self._with_ids(
            segments,
            tuple(str(index) for index in range(1 - count, 1)),
        )

    def source_id_for_local(self, local_id: str) -> str:
        try:
            index = int(str(local_id).strip())
        except ValueError as exc:
            raise ValueError(f"Invalid window-local source id {local_id!r}.") from exc
        if not 1 <= index <= len(self.source_ids):
            raise ValueError(
                f"Window-local source id {local_id!r} is outside 1..{len(self.source_ids)}."
            )
        return self.source_ids[index - 1]


def load_segments_from_stable_json(path: str | Path) -> List[SubtitleSegment]:
    """Load and validate stable-JSON segments (pure view, no reshaping).

    Segmentation is settled upstream by the global DP in ``segment_split``;
    by the time a stable JSON reaches this loader its segments are final and
    ids are simply positional.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(raw_segments, list):
        raise ValueError("Input JSON must contain a segments list.")
    segments: List[SubtitleSegment] = []
    for idx, seg in enumerate(raw_segments, start=1):
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        text = str(seg.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        try:
            start_s = float(start)
            end_s = float(end)
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        segments.append(
            SubtitleSegment(
                id=str(seg.get("id") or idx),
                start=start_s,
                end=end_s,
                text=text,
            )
        )
    return segments


# --- Window-plan persistence (model-routing v2) ------------------------------
#
# Window checkpoints are keyed by chunk ids, and chunk ids come out of
# planning. Replanning is deterministic only while every input is identical --
# the 3-tier token counter alone can differ between runs. Persisting the plan
# and reusing it under the same stable-source fingerprint keeps completed
# windows' chunk ids stable across any execution-configuration change.
#
# The file carries *boundaries only*. Everything derived from the profile, the
# media or the model envelope -- clip range, token budget, read-only preceding
# context -- is recomputed from current inputs on restore, so a plan cannot
# smuggle a stale envelope's numbers into a rerun. v2 dropped those fields.

WINDOW_PLAN_VERSION = 2


def window_plan_payload(windows: Sequence[SubtitleWindow]) -> Dict[str, Any]:
    """Serialize a planned window list for the task artifact directory."""

    return {
        "version": WINDOW_PLAN_VERSION,
        "windows": [
            {
                "chunk_id": window.chunk_id,
                "segment_ids": [segment.id for segment in window.segments],
                "overlap_ids": [segment.id for segment in window.overlap_segments],
                "boundary_reason": window.boundary_reason,
            }
            for window in windows
        ],
    }


def rebuild_windows_from_plan(
    payload: Any,
    segments: Sequence[SubtitleSegment],
    *,
    counter: TokenCounter | None = None,
    limits: ModelLimits = DEFAULT_LIMITS,
    audio_duration: float | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    context_tokens: int = 0,
    prompt_tokens: int = 0,
) -> List[SubtitleWindow] | None:
    """Rebuild the planned windows against the current segment list.

    Returns ``None`` when the stored plan does not fit the segments any more
    (unknown version, missing ids, shape mismatch) -- the caller then plans
    fresh and overwrites the file. Ids resolve against the *current* segments
    so text edits upstream surface as window input-hash mismatches, not as
    stale text resurrected from the plan file.
    """

    if not isinstance(payload, Mapping) or payload.get("version") != WINDOW_PLAN_VERSION:
        return None
    raw_windows = payload.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        return None
    by_id = {segment.id: segment for segment in segments}
    position_by_id = {segment.id: index for index, segment in enumerate(segments)}

    def _resolve(ids: Any) -> List[SubtitleSegment] | None:
        if not isinstance(ids, list) or any(id_ not in by_id for id_ in ids):
            return None
        return [by_id[id_] for id_ in ids]

    windows: List[SubtitleWindow] = []
    for raw in raw_windows:
        if not isinstance(raw, Mapping):
            return None
        target = _resolve(raw.get("segment_ids"))
        overlap = _resolve(raw.get("overlap_ids"))
        chunk_id = raw.get("chunk_id")
        if (
            target is None
            or not target
            or overlap is None
            or not isinstance(chunk_id, str)
            or not chunk_id
        ):
            return None
        target_positions = [position_by_id[segment.id] for segment in target]
        if target_positions != list(
            range(target_positions[0], target_positions[0] + len(target_positions))
        ):
            return None
        if overlap != target[: len(overlap)]:
            return None
        try:
            clip_start, clip_end = compute_clip_range(
                target,
                global_first_id=segments[0].id,
                global_last_id=segments[-1].id,
                audio_duration=audio_duration,
            )
            budget = estimate_window_budget(
                target,
                audio_seconds=clip_end - clip_start,
                context_tokens=context_tokens,
                prompt_tokens=prompt_tokens,
                counter=counter,
                limits=limits,
                profile=profile,
            )
            start_idx = target_positions[0]
            window = SubtitleWindow(
                chunk_id=chunk_id,
                segments=target,
                overlap_segments=overlap,
                boundary_reason=str(raw.get("boundary_reason") or ""),
                budget=budget,
                clip_start=clip_start,
                clip_end=clip_end,
                preceding_segments=preceding_context_for_boundary(segments, start_idx),
            )
        except (TypeError, ValueError):
            return None
        windows.append(window)
    # The plan must cover the segment list exactly once, in order; anything
    # else means the stable JSON changed shape and the plan is stale.
    planned_ids = [
        segment.id
        for window in windows
        for segment in window.segments[len(window.overlap_segments) :]
    ]
    if planned_ids != [segment.id for segment in segments]:
        return None
    return windows


def render_segments_as_srt(segments: Sequence[SubtitleSegment]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_csv_seconds(seconds: float) -> str:
    rounded = round(max(0.0, seconds), 1)
    return f"{rounded:.1f}"


def _encode_csv_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("|", "｜").replace("\n", r"\n")


ASR_RESULT_CSV_HEADER = "local_id|start|duration|gap|text"


def render_segments_as_csv(
    segments: Sequence[SubtitleSegment],
    *,
    window_start: float | None = None,
    allow_negative_start: bool = False,
) -> str:
    """Render segments as the input CSV; ``allow_negative_start`` keeps
    negative local start times (preceding-context lines sit before the clip's
    0 second) instead of clamping them to 0."""

    if not segments:
        return ""
    base_start = segments[0].start if window_start is None else window_start
    lines: List[str] = []
    for idx, segment in enumerate(segments):
        next_start = segments[idx + 1].start if idx + 1 < len(segments) else segment.end
        local_start = segment.start - base_start
        duration = segment.end - segment.start
        gap = max(0.0, next_start - segment.end)
        if allow_negative_start:
            start_cell = f"{round(local_start, 1):.1f}"
            if start_cell == "-0.0":
                start_cell = "0.0"
        else:
            start_cell = _format_csv_seconds(local_start)
        lines.append(
            "|".join(
                [
                    segment.id,
                    start_cell,
                    _format_csv_seconds(duration),
                    _format_csv_seconds(gap),
                    _encode_csv_text(segment.text),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_window_segments_as_csv(window: SubtitleWindow) -> str:
    """Render one target window with model-local ids ``1..N``."""

    local = WindowIdMap.from_window(window).localize_segments(window.segments)
    return render_segments_as_csv(local, window_start=window.clip_start)


def render_window_preceding_as_csv(window: SubtitleWindow) -> str:
    """Render read-only references as chronological non-positive local ids."""

    local = WindowIdMap.from_window(window).localize_preceding_segments(
        window.preceding_segments
    )
    return render_segments_as_csv(
        local,
        window_start=window.clip_start,
        allow_negative_start=True,
    )


def is_reasonable_boundary(
    segments: Sequence[SubtitleSegment],
    end_idx: int,
    *,
    max_gap_sec: float = 0.8,
) -> bool:
    if end_idx >= len(segments) - 1:
        return True
    current = segments[end_idx]
    nxt = segments[end_idx + 1]
    if SENTENCE_END_RE.search(current.text.strip()):
        return True
    return (nxt.start - current.end) >= max_gap_sec


def overlap_count_for_boundary(
    segments: Sequence[SubtitleSegment],
    cut_idx: int,
    *,
    window_seconds: float = OVERLAP_WINDOW_SECONDS,
) -> int:
    """Segments re-included before ``cut_idx``: all starting within the last
    ``window_seconds`` before the cut. Purely content-driven (v13): a big gap
    before the cut correctly yields 0 — stitching must not span it, and
    continuity is the preceding-context block's job. Callers clamp the result
    so the previous window still makes progress."""
    if cut_idx <= 0:
        return 0
    cut_start = segments[cut_idx].start
    count = 0
    for idx in range(cut_idx - 1, -1, -1):
        if cut_start - segments[idx].start <= window_seconds:
            count += 1
        else:
            break
    return count


def preceding_context_for_boundary(
    segments: Sequence[SubtitleSegment],
    start_idx: int,
    *,
    max_segments: int = PRECEDING_CONTEXT_MAX_SEGMENTS,
) -> List[SubtitleSegment]:
    """Up to ``max_segments`` raw segments before ``start_idx`` (the window's
    first segment, overlap included). Fixed count, deliberately no gap-stop:
    after a hard gap the new window is where cold-start risk peaks, and the
    (mostly negative) rendered timestamps let the model judge staleness."""
    if start_idx <= 0:
        return []
    return list(segments[max(0, start_idx - max_segments) : start_idx])


def estimate_window_budget(
    segments: Sequence[SubtitleSegment],
    *,
    audio_seconds: float | None = None,
    context_tokens: int = 0,
    prompt_tokens: int = 0,
    counter: TokenCounter | None = None,
    limits: ModelLimits = DEFAULT_LIMITS,
    profile: TranslationProfile = DEFAULT_PROFILE,
) -> CorrectionBudget:
    counter = counter or default_token_counter()
    subtitle_text = render_segments_as_csv(segments)
    srt_tokens = counter.count_text(subtitle_text)
    if audio_seconds is None:
        # Fallback: segment span. Window planning passes the clip duration
        # (padding included) so the estimate matches what is actually billed.
        audio_seconds = max(0.0, segments[-1].end - segments[0].start) if segments else 0.0
    audio_tokens = (
        counter.count_audio_seconds(audio_seconds)
        if profile.correction_use_audio
        else 0
    )
    video_tokens = (
        math.ceil(
            audio_seconds
            * video_tokens_per_second(
                high_resolution=limits.video_high_resolution
            )
        )
        if profile.correction_use_video
        else 0
    )
    input_tokens = srt_tokens + audio_tokens + video_tokens + context_tokens + prompt_tokens
    return build_correction_budget(
        input_tokens=input_tokens,
        subtitle_input_tokens=srt_tokens,
        token_counter_source=counter.source,
        profile=profile,
    )


def _segment_masses(
    segments: Sequence[SubtitleSegment],
    counter: TokenCounter,
    media_rate: float,
) -> tuple[List[float], int]:
    """Per-segment planning mass: prorated CSV text tokens plus the media
    (audio/video) tokens of the timeline slice owned by the segment (up to the
    next start); the media rate is 0 on the text route.

    One countTokens call over the whole CSV, prorated by character share —
    the per-window real budget check catches any proration error."""
    csv_text = render_segments_as_csv(segments)
    total_text_tokens = counter.count_text(csv_text)
    lines = csv_text.splitlines()
    char_total = sum(len(line) for line in lines) or 1
    n = len(segments)
    masses: List[float] = []
    for idx, (segment, line) in enumerate(zip(segments, lines)):
        text_mass = total_text_tokens * (len(line) / char_total)
        if idx + 1 < n:
            span = max(0.0, segments[idx + 1].start - segment.start)
        else:
            span = max(0.0, segment.end - segment.start)
        masses.append(text_mass + media_rate * span)
    return masses, total_text_tokens


def _estimate_window_count(
    segments: Sequence[SubtitleSegment],
    *,
    total_text_tokens: int,
    total_mass: float,
    context_tokens: int,
    prompt_tokens: int,
    overlap_window_seconds: float,
    limits: ModelLimits,
    planning_limits: ModelLimits,
    profile: TranslationProfile,
    max_window_subtitle_tokens: int = 0,
) -> int:
    from finesub.media.clips import CLIP_PAD_SECONDS

    n = len(segments)
    media_rate = media_tokens_per_second(profile, limits)
    avg_line_tokens = total_text_tokens / n
    total_span = max(0.0, segments[-1].end - segments[0].start)
    overlap_count_est = (
        n * overlap_window_seconds / total_span if total_span > 0 else float(n)
    )
    overlap_text_tokens = overlap_count_est * avg_line_tokens
    overlap_media_tokens = media_rate * overlap_window_seconds
    pad_media_tokens = media_rate * 2 * CLIP_PAD_SECONDS

    # Output constraint: k x c x csv_tokens must fit the planning budget; the
    # quality guardrail caps the window's <asr_result> input below whatever
    # the output coefficient alone would allow (longer inputs degrade quality
    # even when the output fits). <=0 disables the cap.
    coefficient_cap = planning_limits.output_limit / (
        profile.output_scale * profile.output_coefficient
    )
    max_subtitle_tokens = coefficient_cap
    if max_window_subtitle_tokens > 0:
        max_subtitle_tokens = min(max_subtitle_tokens, float(max_window_subtitle_tokens))
    k_out = math.ceil(
        total_text_tokens / max(1.0, max_subtitle_tokens - overlap_text_tokens)
    )
    input_capacity = (
        limits.prompt_input_limit
        - context_tokens
        - prompt_tokens
        - overlap_text_tokens
        - overlap_media_tokens
        - pad_media_tokens
    )
    k_in = math.ceil(total_mass / max(1.0, input_capacity))
    return max(1, k_out, k_in)


def _place_cuts(
    segments: Sequence[SubtitleSegment],
    prefix: Sequence[float],
    k: int,
) -> tuple[List[int], dict[int, str]]:
    """Place k-1 cuts (index of the next window's first core segment) near the
    even mass targets, snapped to the closest reasonable boundary in radius."""
    n = len(segments)
    total = prefix[n]
    radius = max(1, math.ceil(0.4 * n / k))
    cuts: List[int] = []
    reasons: dict[int, str] = {}
    prev_cut = 0
    for j in range(1, k):
        target = total * j / k
        lo_bound = prev_cut + 1
        if lo_bound > n - 1:
            break
        pos = bisect_left(prefix, target)
        center = min(max(pos, lo_bound), n - 1)
        if lo_bound <= pos - 1 <= n - 1 and abs(prefix[pos - 1] - target) < abs(
            prefix[center] - target
        ):
            center = pos - 1
        chosen: int | None = None
        for offset in range(0, radius + 1):
            candidates = (center,) if offset == 0 else (center + offset, center - offset)
            for cand in candidates:
                if lo_bound <= cand <= n - 1 and is_reasonable_boundary(segments, cand - 1):
                    chosen = cand
                    break
            if chosen is not None:
                break
        if chosen is not None:
            cut = chosen
            reasons[cut] = "even_sentence_or_gap_boundary"
        else:
            cut = center
            reasons[cut] = "forced_even_boundary"
        cuts.append(cut)
        prev_cut = cut
    return cuts, reasons


def _build_windows(
    segments: Sequence[SubtitleSegment],
    cuts: Sequence[int],
    reasons: dict[int, str],
    *,
    context_tokens: int,
    prompt_tokens: int,
    overlap_window_seconds: float,
    counter: TokenCounter,
    limits: ModelLimits,
    planning_limits: ModelLimits,
    audio_duration: float | None,
    profile: TranslationProfile,
    max_window_subtitle_tokens: int = 0,
) -> List[SubtitleWindow]:
    n = len(segments)
    bounds = [0, *cuts, n]
    global_first_id = segments[0].id
    global_last_id = segments[-1].id
    windows: List[SubtitleWindow] = []
    prev_start_idx = 0
    for w in range(len(bounds) - 1):
        core_start, core_end = bounds[w], bounds[w + 1]
        if w == 0:
            start_idx = core_start
            overlap: List[SubtitleSegment] = []
        else:
            count = overlap_count_for_boundary(
                segments,
                core_start,
                window_seconds=overlap_window_seconds,
            )
            count = min(count, max(0, core_start - prev_start_idx - 1))
            start_idx = core_start - count
            overlap = list(segments[start_idx:core_start])
        window_segments = list(segments[start_idx:core_end])
        clip_start, clip_end = compute_clip_range(
            window_segments,
            global_first_id=global_first_id,
            global_last_id=global_last_id,
            audio_duration=audio_duration,
        )
        budget = estimate_window_budget(
            window_segments,
            audio_seconds=clip_end - clip_start,
            context_tokens=context_tokens,
            prompt_tokens=prompt_tokens,
            counter=counter,
            limits=limits,
            profile=profile,
        )
        try:
            validate_correction_budget(budget, limits=planning_limits)
            if max_window_subtitle_tokens > 0 and (
                budget.subtitle_input_tokens > max_window_subtitle_tokens
            ):
                raise TokenBudgetError(
                    "Window <asr_result> exceeds max_window_subtitle_tokens: "
                    f"{budget.subtitle_input_tokens} > {max_window_subtitle_tokens}"
                )
        except TokenBudgetError:
            if len(window_segments) == 1:
                raise ValueError(
                    f"Segment {window_segments[0].id} cannot fit in a correction window."
                ) from None
            raise
        windows.append(
            SubtitleWindow(
                chunk_id=f"{w + 1:04d}",
                segments=window_segments,
                overlap_segments=overlap,
                boundary_reason=reasons.get(core_end, "final_window"),
                budget=budget,
                clip_start=clip_start,
                clip_end=clip_end,
                preceding_segments=preceding_context_for_boundary(segments, start_idx),
            )
        )
        prev_start_idx = start_idx
    return windows


def plan_correction_windows(
    segments: Sequence[SubtitleSegment],
    *,
    context_tokens: int = 0,
    prompt_tokens: int = 0,
    overlap_window_seconds: float = OVERLAP_WINDOW_SECONDS,
    planning_output_limit: int | None = None,
    counter: TokenCounter | None = None,
    limits: ModelLimits = DEFAULT_LIMITS,
    audio_duration: float | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    report_sink: dict | None = None,
    max_window_subtitle_tokens: int | None = None,
) -> List[SubtitleWindow]:
    """Plan near-even correction windows.

    Estimates the window count from prorated token mass, places cuts at even
    mass targets snapped to reasonable boundaries, and validates every window
    against the real countTokens budget. Any over-budget window restarts the
    placement with one more window (evenness preserved), bounded at +16.

    ``planning_output_limit`` defaults to the profile-independent window output
    budget ``0.9 x output_limit - 5000``; the per-profile coefficient turns it
    into the CSV token cap.

    ``max_window_subtitle_tokens`` caps a single window's ``<asr_result>`` CSV
    input (core + overlap rows), as a quality guardrail independent of the
    output budget: beyond it, translation quality drops even when the output
    would fit. ``None`` falls back to ``limits.max_window_subtitle_tokens``;
    ``<= 0`` disables the cap. Every planning call site (research, artifacts,
    correction) must resolve the same value or the window plans silently
    disagree on window ids.

    ``report_sink`` (if given) is filled with ``estimated_windows`` /
    ``planned_windows`` / ``replan_attempts`` / ``last_over_budget_error`` so
    callers can surface budget-driven window shrinking in the task report."""
    if not segments:
        return []
    counter = counter or default_token_counter()
    if planning_output_limit is None:
        planning_output_limit = window_output_budget(limits)
    planning_limits = (
        replace(limits, output_limit=min(limits.output_limit, planning_output_limit))
        if planning_output_limit > 0
        else limits
    )
    max_window_subtitle_tokens = effective_window_subtitle_cap(
        max_window_subtitle_tokens, limits
    )
    n = len(segments)
    masses, total_text_tokens = _segment_masses(
        segments, counter, media_tokens_per_second(profile, limits)
    )
    prefix = [0.0]
    for mass in masses:
        prefix.append(prefix[-1] + mass)
    k0 = _estimate_window_count(
        segments,
        total_text_tokens=total_text_tokens,
        total_mass=prefix[n],
        context_tokens=context_tokens,
        prompt_tokens=prompt_tokens,
        overlap_window_seconds=overlap_window_seconds,
        limits=limits,
        planning_limits=planning_limits,
        profile=profile,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
    )
    k0 = min(k0, n)
    max_k = min(n, k0 + 16)
    last_error: Exception | None = None
    for k in range(k0, max_k + 1):
        cuts, reasons = _place_cuts(segments, prefix, k)
        try:
            windows = _build_windows(
                segments,
                cuts,
                reasons,
                context_tokens=context_tokens,
                prompt_tokens=prompt_tokens,
                overlap_window_seconds=overlap_window_seconds,
                counter=counter,
                limits=limits,
                planning_limits=planning_limits,
                audio_duration=audio_duration,
                profile=profile,
                max_window_subtitle_tokens=max_window_subtitle_tokens,
            )
        except TokenBudgetError as exc:
            # Some window is over budget at this k; re-place all cuts with one
            # more window so the split stays even. Plain ValueError (a single
            # segment that cannot fit at all) propagates.
            last_error = exc
            continue
        if report_sink is not None:
            report_sink.update(
                estimated_windows=k0,
                planned_windows=k,
                replan_attempts=k - k0,
                last_over_budget_error=str(last_error) if last_error else "",
            )
        return windows
    raise ValueError(
        f"Could not plan correction windows within budget after {max_k - k0 + 1} "
        f"attempts (k={k0}..{max_k}): {last_error}"
    )


def split_window_in_half(
    window: SubtitleWindow,
    *,
    overlap_window_seconds: float = OVERLAP_WINDOW_SECONDS,
    counter: TokenCounter | None = None,
    limits: ModelLimits = DEFAULT_LIMITS,
    global_first_id: str = "",
    global_last_id: str = "",
    audio_duration: float | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    context_tokens: int = 0,
    prompt_tokens: int = 0,
) -> tuple[SubtitleWindow, SubtitleWindow] | None:
    """Split a window into two overlapping halves for truncation retries.

    The split point is the reasonable boundary nearest to the middle; the
    halves share the same dynamic overlap as planned windows. Each half gets
    its own audio clip range (edge pads keyed by the global first/last segment
    ids). Returns ``None`` when the window cannot be split.
    """

    segments = window.segments
    n = len(segments)
    if n < 2:
        return None

    # Candidate end index of the first half, searched outward from the middle.
    middle = (n - 1) / 2
    candidates = sorted(range(0, n - 1), key=lambda idx: abs(idx - middle))
    first_end = next(
        (idx for idx in candidates if is_reasonable_boundary(segments, idx)),
        int(middle),
    )
    first_end = max(0, min(first_end, n - 2))

    overlap_count = overlap_count_for_boundary(
        segments,
        first_end + 1,
        window_seconds=overlap_window_seconds,
    )
    effective_overlap = min(overlap_count, first_end + 1)
    second_start = first_end + 1 - effective_overlap
    if second_start <= 0:
        second_start = first_end + 1
        effective_overlap = 0

    first_segments = list(segments[: first_end + 1])
    second_segments = list(segments[second_start:])
    # Preceding context: -a inherits the parent's; -b looks back across the
    # parent's own segments (and, if the split point is close to the parent's
    # start, on into the parent's preceding lines).
    extended = list(window.preceding_segments) + list(segments)
    second_preceding = preceding_context_for_boundary(
        extended, len(window.preceding_segments) + second_start
    )

    def _half(
        chunk_suffix: str,
        half_segments: List[SubtitleSegment],
        overlap: List[SubtitleSegment],
        reason: str,
        preceding: List[SubtitleSegment],
    ) -> SubtitleWindow:
        clip_start, clip_end = compute_clip_range(
            half_segments,
            global_first_id=global_first_id,
            global_last_id=global_last_id,
            audio_duration=audio_duration,
        )
        return SubtitleWindow(
            chunk_id=f"{window.chunk_id}{chunk_suffix}",
            segments=half_segments,
            overlap_segments=overlap,
            boundary_reason=reason,
            budget=estimate_window_budget(
                half_segments,
                audio_seconds=clip_end - clip_start,
                context_tokens=context_tokens,
                prompt_tokens=prompt_tokens,
                counter=counter,
                limits=limits,
                profile=profile,
            ),
            clip_start=clip_start,
            clip_end=clip_end,
            preceding_segments=preceding,
        )

    first = _half(
        "-a",
        first_segments,
        window.overlap_segments,
        "split_retry_first_half",
        list(window.preceding_segments),
    )
    second = _half(
        "-b",
        second_segments,
        list(segments[second_start : first_end + 1]),
        "split_retry_second_half",
        second_preceding,
    )
    return first, second
