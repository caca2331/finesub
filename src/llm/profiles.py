"""Route/level translation profiles and output-budget formulas.

Six strict presets (--route text|mm x --level low|med|high). The mm route's
defining trait is harness-side external injection (two-round research, the
per-window query round, the local search agent); the text route never runs
harness-side retrieval — text-high only enables the model's own search tool.
Levels are named presets, not free toggle combinations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .config import DEFAULT_LIMITS, ModelLimits

# Additive coefficient components (kept for auditability; presets below are
# the only exposed combinations).
TEXT_BASE_COEFF = 2.0
TEXT_THINKING_COEFF = 1.5
TEXT_SEARCH_COEFF = 1.0
MM_BASE_COEFF = 4.5
MM_AUDIO_COEFF = 0.5
MM_VIDEO_COEFF = 1.0

# Normal windows: expected output must fit 0.9 x output_limit - 5k.
WINDOW_OUTPUT_FILL_RATIO = 0.9
WINDOW_OUTPUT_SLACK_TOKENS = 5_000
# Fast mode treats the whole input as one window under a stricter budget.
FAST_OUTPUT_FILL_RATIO = 0.8
FAST_OUTPUT_SLACK_TOKENS = 10_000
# Fast round 1 must leave input headroom for round-2 injections: the entry
# block (<=28k) + evidence pack / search results (<=20k) + round-1 notes (2k)
# + scaffolding/static-prompt delta.
FAST_ROUND2_INPUT_RESERVE_TOKENS = 56_000
DEFAULT_FAST_SEARCH_ROUNDS = 2

# Gemini video tokens: tokens/frame x sample fps (low resolution default).
VIDEO_TOKENS_PER_FRAME_LOW = 71
VIDEO_TOKENS_PER_FRAME_HIGH = 269
VIDEO_SAMPLE_FPS = 0.25

ROUTES = ("text", "mm")
LEVELS = ("low", "med", "high")

_OUTPUT_COEFFICIENTS: dict[tuple[str, str], float] = {
    ("text", "low"): TEXT_BASE_COEFF,                                             # 2.0
    ("text", "med"): TEXT_BASE_COEFF + TEXT_THINKING_COEFF,                       # 3.5
    ("text", "high"): TEXT_BASE_COEFF + TEXT_THINKING_COEFF + TEXT_SEARCH_COEFF,  # 4.5
    ("mm", "low"): MM_BASE_COEFF,                                                 # 4.5
    ("mm", "med"): MM_BASE_COEFF + MM_AUDIO_COEFF,                                # 5.0
    ("mm", "high"): MM_BASE_COEFF + MM_AUDIO_COEFF + MM_VIDEO_COEFF,              # 6.0
}


@dataclass(frozen=True)
class TranslationProfile:
    route: str
    level: str
    # Derived preset traits (strict presets; no free toggle combinations).
    native_search: bool
    external_injection: bool
    use_audio: bool
    use_video: bool
    # Per-call thinking override for the correction call ("low"/"high"; ""
    # keeps the role default, medium everywhere per current policy).
    thinking_override: str
    output_coefficient: float
    # User scale k (--output-scale); larger k means smaller windows.
    output_scale: float = 1.0

    @property
    def profile_id(self) -> str:
        return f"{self.route}-{self.level}"

    def with_output_scale(self, output_scale: float) -> "TranslationProfile":
        return replace(self, output_scale=output_scale)


def resolve_profile(
    route: str = "mm",
    level: str = "med",
    *,
    output_scale: float = 1.0,
) -> TranslationProfile:
    route = (route or "").strip().lower()
    level = (level or "").strip().lower()
    if route not in ROUTES:
        raise ValueError(f"Unknown route {route!r}; expected one of {ROUTES}")
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}; expected one of {LEVELS}")
    if output_scale <= 0:
        raise ValueError("output_scale must be positive")
    return TranslationProfile(
        route=route,
        level=level,
        native_search=(route == "text" and level == "high"),
        external_injection=(route == "mm"),
        use_audio=(route == "mm" and level in ("med", "high")),
        use_video=(route == "mm" and level == "high"),
        thinking_override="low" if (route == "text" and level == "low") else "",
        output_coefficient=_OUTPUT_COEFFICIENTS[(route, level)],
        output_scale=output_scale,
    )


DEFAULT_PROFILE = resolve_profile()


def expected_output_tokens(profile: TranslationProfile, csv_tokens: int) -> int:
    """k x c x csv_tokens; replaces the old ``csv x 5 + 10k`` estimate."""

    return math.ceil(
        profile.output_scale * profile.output_coefficient * max(0, int(csv_tokens))
    )


def window_output_budget(
    limits: ModelLimits = DEFAULT_LIMITS, *, fast: bool = False
) -> int:
    if fast:
        return int(FAST_OUTPUT_FILL_RATIO * limits.output_limit) - FAST_OUTPUT_SLACK_TOKENS
    return int(WINDOW_OUTPUT_FILL_RATIO * limits.output_limit) - WINDOW_OUTPUT_SLACK_TOKENS


def max_window_csv_tokens(
    profile: TranslationProfile,
    *,
    limits: ModelLimits = DEFAULT_LIMITS,
    fast: bool = False,
) -> int:
    """Largest per-window CSV token count whose expected output still fits."""

    budget = window_output_budget(limits, fast=fast)
    return int(budget / (profile.output_scale * profile.output_coefficient))


def video_tokens_per_second(*, high_resolution: bool = False) -> float:
    per_frame = VIDEO_TOKENS_PER_FRAME_HIGH if high_resolution else VIDEO_TOKENS_PER_FRAME_LOW
    return per_frame * VIDEO_SAMPLE_FPS
