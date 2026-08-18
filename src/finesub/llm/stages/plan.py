"""Fast-mode planning: the single whole-input window and the enable decision.

Fast mode treats the entire input as one correction window and fuses research
round 1 with the per-window query round. Auto-enable requires both:

- output: k x c x total_csv_tokens <= 0.8 x output_limit - 10k
- input:  round-1 prompt text + clip media tokens
          <= prompt_input_limit - FAST_ROUND2_INPUT_RESERVE_TOKENS
  (that reserve keeps headroom for round-2 injections: entry details,
  evidence pack / search results, round-1 notes)

Both limits come from the *bound group's* envelope, not the packaged ceiling
(the caller passes ``correction_planning_limits``). The reserve itself is an
absolute token count calibrated against Gemini's 194k, so a group too small to
hold it is rejected by name rather than by a negative budget -- see the gate.

``--fast on`` raises with the measured numbers when any check fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from finesub.media.clips import compute_clip_range, probe_audio_duration
from ..chunking import (
    SubtitleSegment,
    SubtitleWindow,
    estimate_window_budget,
    load_segments_from_stable_json,
    media_tokens_per_second,
)
from ..routing.config import (
    DEFAULT_LIMITS,
    ModelLimits,
    effective_window_subtitle_cap,
    research_search_query_limit,
)
from ..knowledge.base import DEFAULT_KNOWLEDGE_ROOT, load_index_text
from ..routing.profiles import (
    DEFAULT_PROFILE,
    FAST_ROUND2_INPUT_RESERVE_TOKENS,
    TranslationProfile,
    expected_output_tokens,
    window_output_budget,
)
from ..token_budget import default_token_counter, TokenCounter

FAST_WINDOW_CHUNK_ID = "0001"


@dataclass(frozen=True)
class FastDecision:
    mode: str  # requested: "auto" | "on" | "off"
    enabled: bool
    reason: str
    window: SubtitleWindow | None = None
    expected_output_tokens: int = 0
    output_budget: int = 0
    round1_input_tokens: int = 0
    input_budget: int = 0
    subtitle_input_tokens: int = 0
    max_window_subtitle_tokens: int = 0

    def to_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "reason": self.reason,
            "expected_output_tokens": self.expected_output_tokens,
            "output_budget": self.output_budget,
            "round1_input_tokens": self.round1_input_tokens,
            "input_budget": self.input_budget,
            "subtitle_input_tokens": self.subtitle_input_tokens,
            "max_window_subtitle_tokens": self.max_window_subtitle_tokens,
        }


def plan_fast_window(
    segments: Sequence[SubtitleSegment],
    *,
    audio_duration: float | None = None,
    counter: TokenCounter | None = None,
    limits: ModelLimits = DEFAULT_LIMITS,
    profile: TranslationProfile = DEFAULT_PROFILE,
) -> SubtitleWindow:
    """The single fast window: all segments, no overlap, 60s edge pads."""

    segments = list(segments)
    if not segments:
        raise ValueError("Cannot plan a fast window without segments.")
    clip_start, clip_end = compute_clip_range(
        segments,
        global_first_id=segments[0].id,
        global_last_id=segments[-1].id,
        audio_duration=audio_duration,
    )
    return SubtitleWindow(
        chunk_id=FAST_WINDOW_CHUNK_ID,
        segments=segments,
        overlap_segments=[],
        boundary_reason="fast_single_window",
        budget=estimate_window_budget(
            segments,
            audio_seconds=clip_end - clip_start,
            counter=counter,
            limits=limits,
            profile=profile,
        ),
        clip_start=clip_start,
        clip_end=clip_end,
    )


def decide_fast_mode(
    *,
    stable_json: str | Path,
    fast: str = "auto",
    profile: TranslationProfile = DEFAULT_PROFILE,
    audio_path: str | Path | None = None,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    token_counter: TokenCounter | None = None,
    limits: ModelLimits = DEFAULT_LIMITS,
    max_window_subtitle_tokens: int | None = None,
) -> FastDecision:
    """Decide whether the run goes fast (single fused window).

    ``fast="on"`` raises ValueError when the gates do not pass; ``"auto"``
    falls back to the normal multi-window flow with the reason recorded.

    The fast window is the whole input, which makes it the path most exposed to
    the quality guardrail the multi-window planner already enforces, so
    ``max_window_subtitle_tokens`` gates auto-enable the same way the budgets
    do. ``None`` takes the limits default; ``0`` disables the gate.
    """

    mode = (fast or "auto").strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Unknown --fast mode {fast!r}; expected auto/on/off.")
    if mode == "off":
        return FastDecision(mode=mode, enabled=False, reason="fast mode disabled")

    counter = token_counter or default_token_counter()
    if limits.prompt_input_limit <= FAST_ROUND2_INPUT_RESERVE_TOKENS:
        # The round-2 reserve is an absolute token count (the sum of what
        # round 2 injects: entry block + evidence pack + notes + scaffolding),
        # calibrated against Gemini's 194k prompt limit. A bound group with a
        # smaller envelope makes ``prompt_limit - reserve`` negative, and every
        # input then "exceeds" a negative budget -- true, but unreadable. Say
        # the actual thing instead: this group cannot hold a fused window.
        #
        # Scaling the reserve to the envelope is deliberately NOT done here:
        # it is a sum of specific injection caps, not a ratio, so shrinking it
        # without shrinking those caps would let the gate pass and round 2
        # overflow anyway. Doing it properly means re-calibrating the caps
        # themselves (entry block / evidence pack / notes), which is a quality
        # decision, not space accounting -- recorded in
        # docs/llm_followups.md "fast 会话内部预算未按组收缩".
        reason = (
            f"bound group input envelope {limits.prompt_input_limit} <= the "
            f"{FAST_ROUND2_INPUT_RESERVE_TOKENS} round-2 reserve: this group "
            "cannot hold a fused single window; use the normal multi-window "
            "flow (fast mode is calibrated for large-context models)"
        )
        if mode == "on":
            # Same contract as the budget gates below: an explicit --fast on
            # that cannot be honoured is an error, not a silent downgrade.
            raise ValueError(
                "--fast on requested but the input does not fit fast mode: "
                f"{reason}. Use --fast auto or the normal flow."
            )
        return FastDecision(mode=mode, enabled=False, reason=reason)
    segments = load_segments_from_stable_json(stable_json)
    if not segments:
        return FastDecision(mode=mode, enabled=False, reason="no segments")
    audio_duration = (
        probe_audio_duration(audio_path)
        if (audio_path and profile.correction_use_audio)
        else None
    )
    window = plan_fast_window(
        segments,
        audio_duration=audio_duration,
        counter=counter,
        limits=limits,
        profile=profile,
    )

    expected = expected_output_tokens(profile, window.budget.subtitle_input_tokens)
    output_budget = window_output_budget(limits, fast=True)
    round1_input, input_budget = _round1_input_estimate(
        window,
        segments_count=len(segments),
        extra_info=extra_info,
        knowledge_root=knowledge_root,
        knowledge_enabled=knowledge_enabled,
        counter=counter,
        limits=limits,
        profile=profile,
    )
    subtitle_cap = effective_window_subtitle_cap(max_window_subtitle_tokens, limits)
    subtitle_input = window.budget.subtitle_input_tokens
    failures = []
    if expected > output_budget:
        failures.append(
            f"expected output {expected} > fast output budget {output_budget}"
        )
    if round1_input > input_budget:
        failures.append(
            f"round-1 input {round1_input} > input budget {input_budget} "
            f"(prompt limit minus the {FAST_ROUND2_INPUT_RESERVE_TOKENS} round-2 reserve)"
        )
    if subtitle_cap > 0 and subtitle_input > subtitle_cap:
        failures.append(
            f"<asr_result> input {subtitle_input} > max_window_subtitle_tokens "
            f"{subtitle_cap}"
        )
    decision = FastDecision(
        mode=mode,
        enabled=not failures,
        reason="; ".join(failures) or "fits fast budgets",
        window=window,
        expected_output_tokens=expected,
        output_budget=output_budget,
        round1_input_tokens=round1_input,
        input_budget=input_budget,
        subtitle_input_tokens=subtitle_input,
        max_window_subtitle_tokens=subtitle_cap,
    )
    if mode == "on" and not decision.enabled:
        raise ValueError(
            "--fast on requested but the input does not fit fast mode: "
            f"{decision.reason}. Use --fast auto or the normal flow."
        )
    return decision


def _round1_input_estimate(
    window: SubtitleWindow,
    *,
    segments_count: int,
    extra_info: str,
    knowledge_root: str | Path,
    knowledge_enabled: bool,
    counter: TokenCounter,
    limits: ModelLimits,
    profile: TranslationProfile,
) -> tuple[int, int]:
    """Measure the fast round-1 prompt (text via countTokens, media locally).

    On the mm route round 1 is the fused research+query prompt (knowledge
    indices included); on the text route there is no separate round 1, so the
    correction prompt itself is measured against the same reserve. The
    ``--extra-info`` URL pre-extracts are not included (approximation; the
    reserve absorbs them).
    """

    from ..prompts import build_correction_csv_messages, build_fast_round1_messages

    if profile.external_injection:
        # ``--knowledge none`` never reads the base, in the estimate either --
        # a growing knowledge base must not inflate the fast gate for runs
        # that will not see it.
        messages = build_fast_round1_messages(
            window=window,
            extra_info=extra_info,
            streamer_index=(
                load_index_text(knowledge_root, "streamer") if knowledge_enabled else ""
            ),
            common_index=(
                load_index_text(knowledge_root, "common") if knowledge_enabled else ""
            ),
            max_search_queries=research_search_query_limit(segments_count),
            use_search_contract=True,
            profile=profile,
        )
    else:
        messages = build_correction_csv_messages(window=window, profile=profile)
    text_tokens = counter.count_texts(
        str(message.get("content", "")) for message in messages
    )
    media_tokens = math.ceil(
        max(0.0, window.clip_end - window.clip_start)
        * media_tokens_per_second(profile, limits)
    )
    input_budget = limits.prompt_input_limit - FAST_ROUND2_INPUT_RESERVE_TOKENS
    return text_tokens + media_tokens, input_budget
