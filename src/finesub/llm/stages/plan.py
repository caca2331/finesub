"""Fast-mode planning: the single whole-input window and the enable decision.

Fast mode treats the entire input as one correction task (no preparation
task; under ``retrieval=local`` the window asks for its own inputs through
the request channel). Auto-enable requires:

- output: k x c x total_csv_tokens <= 0.8 x output_limit - 10k
- input:  opening prompt text + clip media tokens
          <= prompt_input_limit - FAST_FILL_INPUT_RESERVE_TOKENS
  (that reserve keeps headroom for the fill turn: entry details, search
  results, the model's own request turn)

and, for a task whose pass 3 follows (owner D3: it is one window too), pass 3's
input estimate within its own group's envelope: its system prompt and the
material's notes, plus ``PROOFREAD_SUBTITLE_FACTOR`` x the subtitle tokens
(each finished line shows the ASR text, the corrected text and the
translation; the ASR text only where it differs).

Both limits come from the *bound group's* envelope, not the packaged ceiling
(the caller passes ``correction_planning_limits``). The reserve itself is an
absolute token count calibrated against Gemini's 194k, so a group too small to
hold it is rejected by name rather than by a negative budget -- see the gate.

``--fast on`` raises with the measured numbers when any check fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

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
    MAX_WINDOW_SEARCH_QUERIES,
    ModelLimits,
    effective_window_subtitle_cap,
)
from ..knowledge.base import DEFAULT_KNOWLEDGE_ROOT, index_texts_present, load_index_texts
from .task_graph import window_requests
from ..routing.profiles import (
    DEFAULT_PROFILE,
    FAST_FILL_INPUT_RESERVE_TOKENS,
    TranslationProfile,
    expected_output_tokens,
    window_output_budget,
)
from ..token_budget import default_token_counter, TokenCounter

if TYPE_CHECKING:
    from ..prompts import ContextPack

FAST_WINDOW_CHUNK_ID = "0001"
#: Pass 3's lines against pass 2's input lines, in tokens: three texts for one (see the module docstring).
PROOFREAD_SUBTITLE_FACTOR = 3


@dataclass(frozen=True)
class FastDecision:
    mode: str  # requested: "auto" | "on" | "off"
    enabled: bool
    reason: str
    window: SubtitleWindow | None = None
    expected_output_tokens: int = 0
    output_budget: int = 0
    opening_input_tokens: int = 0
    input_budget: int = 0
    subtitle_input_tokens: int = 0
    max_window_subtitle_tokens: int = 0
    #: Pass 3's whole-input window, when the task's pass 3 follows (0 when it does not).
    proofread_input_tokens: int = 0
    proofread_input_budget: int = 0

    def to_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "reason": self.reason,
            "expected_output_tokens": self.expected_output_tokens,
            "output_budget": self.output_budget,
            "opening_input_tokens": self.opening_input_tokens,
            "input_budget": self.input_budget,
            "subtitle_input_tokens": self.subtitle_input_tokens,
            "max_window_subtitle_tokens": self.max_window_subtitle_tokens,
            **({"proofread_input_tokens": self.proofread_input_tokens,
                "proofread_input_budget": self.proofread_input_budget} if self.proofread_input_budget else {}),
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
    # `--style auto`: the candidate list rides in the opening message, so the
    # gate has to weigh it (the 56k reserve is for the *fill*).
    style_catalog: str = "",
    # Pass 3's group's envelope when the task's pass 3 follows; None skips that gate.
    proofread_limits: ModelLimits | None = None,
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
    if limits.prompt_input_limit <= FAST_FILL_INPUT_RESERVE_TOKENS:
        # The fill reserve is an absolute token count (the sum of what the
        # fill turn injects: entry block + search results + scaffolding),
        # calibrated against Gemini's 194k prompt limit. A bound group with a
        # smaller envelope makes ``prompt_limit - reserve`` negative, and every
        # input then "exceeds" a negative budget -- true, but unreadable. Say
        # the actual thing instead: this group cannot hold a fused window.
        #
        # Scaling the reserve to the envelope is deliberately NOT done here:
        # it is a sum of specific injection caps, not a ratio, so shrinking it
        # without shrinking those caps would let the gate pass and the fill
        # turn overflow anyway. Doing it properly means re-calibrating the caps
        # themselves (entry block / search results), which is a quality
        # decision, not space accounting -- and not one worth making: small-
        # context models have all but gone (owner 2026-09-30).
        reason = (
            f"bound group input envelope {limits.prompt_input_limit} <= the "
            f"{FAST_FILL_INPUT_RESERVE_TOKENS} fill reserve: this group "
            "cannot hold a whole-input window; use the normal multi-window "
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
    opening_input, input_budget = _opening_input_estimate(
        window,
        style_catalog=style_catalog,
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
    if opening_input > input_budget:
        failures.append(
            f"opening input {opening_input} > input budget {input_budget} "
            f"(prompt limit minus the {FAST_FILL_INPUT_RESERVE_TOKENS} fill reserve)"
        )
    if subtitle_cap > 0 and subtitle_input > subtitle_cap:
        failures.append(
            f"<asr_result> input {subtitle_input} > max_window_subtitle_tokens "
            f"{subtitle_cap}"
        )
    proofread_input = proofread_budget = 0
    if proofread_limits is not None:
        from .proofread.prompt import system_prompt

        proofread_budget = proofread_limits.prompt_input_limit
        proofread_input = (
            counter.count_text(system_prompt() + extra_info)
            + PROOFREAD_SUBTITLE_FACTOR * subtitle_input
        )
        if proofread_input > proofread_budget:
            failures.append(
                f"pass 3 input estimate {proofread_input} > its input limit {proofread_budget}"
            )
    decision = FastDecision(
        mode=mode,
        enabled=not failures,
        reason="; ".join(failures) or "fits fast budgets",
        window=window,
        expected_output_tokens=expected,
        output_budget=output_budget,
        opening_input_tokens=opening_input,
        input_budget=input_budget,
        subtitle_input_tokens=subtitle_input,
        max_window_subtitle_tokens=subtitle_cap,
        proofread_input_tokens=proofread_input,
        proofread_input_budget=proofread_budget,
    )
    if mode == "on" and not decision.enabled:
        raise ValueError(
            "--fast on requested but the input does not fit fast mode: "
            f"{decision.reason}. Use --fast auto or the normal flow."
        )
    return decision


def _opening_input_estimate(
    window: SubtitleWindow,
    *,
    style_catalog: str = "",
    extra_info: str,
    knowledge_root: str | Path,
    knowledge_enabled: bool,
    counter: TokenCounter,
    limits: ModelLimits,
    profile: TranslationProfile,
) -> tuple[int, int]:
    """Measure the fast window's opening message (text via countTokens, media locally).

    Wherever the window has something to request (harness search under
    ``retrieval=local``, or a readable knowledge base -- `task_graph.window_flow`)
    or a style to pick (``--style auto`` with a choice) the opening carries the
    request channel, the knowledge indices and the candidate catalog where
    they apply; otherwise it is the correction
    prompt itself, measured against the same reserve. ``extra_info`` rides in the general context (there is no
    preparation task to digest it).

    ⚠ Always the **batched** shape, even for a run whose correction chain
    resolves to ``agentic`` -- that one sends the acquire note instead of the
    request channel and no index at all, so the gate over-counts it. Measured
    2026-09-10: 956 chars with an empty base, ~2.8k at 50 index entries, ~6.7k
    at 200, against a 138k budget. Deliberately left over-counting: it errs
    toward the normal multi-window flow, which is this gate's own fallback, and
    asking the real question here would pull routing (and a driver probe) into
    the planner for ~2% on one narrow cell. Recorded in docs/llm_harness_behavior.md
    (「快速模式」).
    """

    from ..request_prompts import build_window_task_messages

    # ``--knowledge none`` never reads the base, in the estimate either -- a
    # growing knowledge base must not inflate the fast gate for runs that will
    # not see it.
    index_texts = load_index_texts(knowledge_root) if knowledge_enabled else {}
    readable = index_texts_present(index_texts)
    # `task_graph.window_requests`, not `window_flow`: always the batched
    # shape (see above), so no strategy and no driver probe. A catalog opens
    # the channel on its own (auto-style-gaps plan A, the `style_pick` half of
    # `WindowFlow`): a style-only window has a request turn, and measuring it
    # without one would leave the catalog out of the gated message.
    messages = build_window_task_messages(
        request_channel=window_requests(profile, knowledge_readable=readable)
        or bool(style_catalog.strip()),
        knowledge_indexes=index_texts if readable else {},
        max_search_queries=MAX_WINDOW_SEARCH_QUERIES,
        # Under `--style auto` the fused window picks in its own request turn,
        # so the candidate list is part of the message being gated -- not of
        # the fill the reserve covers. (`--extra-style` rides in the same
        # section but is one user-written clause; the reserve absorbs it, like
        # the note URL extracts.)
        style_catalog=style_catalog,
        # The task's length ratio is not known yet (the style is chosen after this gate) and only changes a few
        # digits of the prompt: measured at the standard tier's. The fast window itself gets the task's later.
        window=replace(window, length_scale=1.0),
        context_pack=fast_context_pack(extra_info),
        profile=profile,
        knowledge_enabled=readable,
    )
    text_tokens = counter.count_texts(
        str(message.get("content", "")) for message in messages
    )
    media_tokens = math.ceil(
        max(0.0, window.clip_end - window.clip_start)
        * media_tokens_per_second(profile, limits)
    )
    input_budget = limits.prompt_input_limit - FAST_FILL_INPUT_RESERVE_TOKENS
    return text_tokens + media_tokens, input_budget


def fast_context_pack(extra_info: str) -> ContextPack:
    """What a fast window gets as general context: the user's own notes.

    Fast mode runs no preparation task, so nothing digests ``--extra-info``
    for it; the notes are handed to the window verbatim instead of dropped.
    """

    from ..prompts import ContextPack

    notes = (extra_info or "").strip()
    return ContextPack(general_context={"user_notes": notes} if notes else {})
