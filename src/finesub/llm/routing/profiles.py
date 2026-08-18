"""Orthogonal switch axes for the correction/translation run.

The switch axes live here; ``knowledge`` stays an independent CLI argument
because it is threaded through the ``finesub.llm.knowledge`` modules, which have no
business knowing about profiles. Canonical behavior is documented in
``docs/llm_harness_behavior.md``.

- ``correction_media`` / ``planning_media``  text / audio / video, each a
  ladder (video implies audio). The run-level ``media`` axis was retired
  (model-routing v2 media split): it answered two questions at once -- "what
  media does this run have" (decided by which files are passed) and "which
  task consumes it". The correction window follows ``correction_media``; the
  per-window query round follows ``planning_media``. A switch above what the
  input files provide is a configuration error at the entrypoints, not a
  silent downgrade.
- ``retrieval`` none / local / native. ``local`` is the whole harness-side
  injection machinery (background research, the per-window query round, the
  local search agent); ``native`` is the model's own search tool and never
  coexists with harness search; ``none`` is neither. ``planning_media`` only
  matters under ``local`` (the only vector with a query round); elsewhere it
  is carried but unused.
- ``difficulty`` quality / intermediate / efficiency -- it does exactly two
  things: pick the task-group cell's prompt variant and that cell's thinking
  level. It no longer reads a capability tier off the answering endpoint (that
  column is gone), and it no longer touches the window geometry, which is what
  lets an explicit mid-run switch reuse completed windows. A preset may also
  bind a different model group per difficulty -- the shipped one does for
  correction (capable at ``quality``, the lites at ``intermediate``), so switching down is
  the declared, user-initiated way to keep going when the free capable quota
  runs out. ``efficiency`` additionally pins the other axes (see below).

The six retired presets (``--route text|mm`` x ``--level low|med|high``) map
onto these; docs/llm_harness_behavior.md keeps the table for reading old
artifacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .config import DEFAULT_LIMITS, ModelLimits

MEDIA = ("text", "audio", "video")
RETRIEVAL = ("none", "local", "native")
DIFFICULTY = ("quality", "intermediate", "efficiency")
# The continuity axis (docs/llm_harness_behavior.md): ``serial`` keeps the
# chained inter-window context (advice ledger, entry transfer chain);
# ``parallel`` gives it up and dispatches correction windows concurrently.
# It changes what the prompt may reference, so it is a prompt-affecting axis
# and part of the profile vector -- unlike knowledge, which is an independent
# CLI argument owned by ``finesub.llm.knowledge``; continuity has no second source of
# truth.
CONTINUITY = ("serial", "parallel")

# Rank for the media ladder: "media >= audio" is a real comparison.
_MEDIA_RANK = {name: index for index, name in enumerate(MEDIA)}

# Additive output-budget coefficient, decomposed onto the switches. Reproducing
# the six retired presets is a compatibility check, not a new calibration;
# docs/llm_followups.md owns the remaining calibration work.
BASE_COEFF = 2.0            # correction + translation proper
THINKING_COEFF = 1.5        # unconditional since model-routing v2 (efficiency incl.)
RETRIEVAL_COEFF = 1.0       # native search or local injection
AUDIO_COEFF = 0.5
VIDEO_COEFF = 1.0

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


def output_coefficient(correction_media: str, retrieval: str) -> float:
    # The AUDIO/VIDEO terms follow ``correction_media`` (model-routing v2): what is
    # being budgeted is the correction window's output, and only its clip
    # rides along. The query round has its own fixed output cap.
    #
    # difficulty no longer appears: it only picks the prompt variant and the
    # thinking level, so ``efficiency`` lost its discount and the
    # +1.5 thinking term is unconditional. That moves text/none/efficiency from
    # the measured c=2.0 to 3.5 (window capacity 57%) -- recorded as a stale
    # calibration in ``capabilities.RECALIBRATION_PENDING``.
    coefficient = BASE_COEFF + THINKING_COEFF
    if retrieval != "none":
        coefficient += RETRIEVAL_COEFF
    if _MEDIA_RANK[correction_media] >= _MEDIA_RANK["audio"]:
        coefficient += AUDIO_COEFF
    if correction_media == "video":
        coefficient += VIDEO_COEFF
    return coefficient


@dataclass(frozen=True)
class TranslationProfile:
    correction_media: str
    planning_media: str
    retrieval: str
    difficulty: str
    output_coefficient: float
    # User scale k (--output-scale); larger k means smaller windows.
    output_scale: float = 1.0
    continuity: str = "serial"

    @property
    def profile_id(self) -> str:
        """Canonical switch vector -- the artifact/display form."""

        return (
            f"correction_media={self.correction_media},"
            f"planning_media={self.planning_media},"
            f"retrieval={self.retrieval},"
            f"difficulty={self.difficulty},continuity={self.continuity}"
        )

    @property
    def geometry_id(self) -> str:
        """Exactly the switches that move a window boundary.

        These are the profile attributes the planner actually reads:
        ``estimate_window_budget`` adds the clip's tokens for
        ``correction_media``, and ``max_window_csv_tokens`` divides by
        ``output_scale * output_coefficient`` (``retrieval`` folds into the
        coefficient). ``planning_media``, ``continuity`` and ``difficulty``
        never reach the planner, so they cannot shift a chunk id.

        This is **audit metadata, not a reuse gate**: a persisted plan owns
        boundary identity and refits its pending leaves against the current
        envelope, while research notes are addressed by source-id interval
        rather than by chunk id. It records which geometry placed the
        boundaries -- see ``research.plan_geometry_metadata``.
        """

        return (
            f"correction_media={self.correction_media},"
            f"output_coefficient={self.output_coefficient},"
            f"output_scale={self.output_scale}"
        )

    # No plain ``use_audio``/``use_video``: every call site must say which
    # task's media it means (plan v2 D20). ``uses_media`` is the clip/upload
    # gate -- extraction runs iff *either* switch wants media (plan §8-2).
    @property
    def correction_use_audio(self) -> bool:
        return _MEDIA_RANK[self.correction_media] >= _MEDIA_RANK["audio"]

    @property
    def correction_use_video(self) -> bool:
        return self.correction_media == "video"

    @property
    def planning_use_audio(self) -> bool:
        return _MEDIA_RANK[self.planning_media] >= _MEDIA_RANK["audio"]

    @property
    def planning_use_video(self) -> bool:
        return self.planning_media == "video"

    @property
    def uses_media(self) -> bool:
        return self.correction_media != "text" or self.planning_media != "text"

    @property
    def uses_video(self) -> bool:
        return self.correction_media == "video" or self.planning_media == "video"

    @property
    def native_search(self) -> bool:
        return self.retrieval == "native"

    @property
    def external_injection(self) -> bool:
        """Whether the harness runs its own retrieval (research, query round)."""

        return self.retrieval == "local"

    def with_output_scale(self, output_scale: float) -> "TranslationProfile":
        return replace(self, output_scale=output_scale)


class SwitchConflictError(ValueError):
    """A switch combination the constraint matrix rejects outright."""


def resolve_profile(
    # Defaults are the harness's own default run shape (the retired "mm-med"),
    # not the pipeline's -- ``pipeline.py`` asks for media=video explicitly.
    media: str = "audio",
    retrieval: str = "local",
    difficulty: str = "quality",
    continuity: str = "serial",
    *,
    correction_media: str = "",
    planning_media: str = "",
    output_scale: float = 1.0,
) -> TranslationProfile:
    """Build the switch vector.

    ``media`` is the convenience knob that sets both per-task switches at
    once (the common case is still one dial); ``correction_media`` /
    ``planning_media`` override it individually (plan v2 D20).
    """

    media = (media or "").strip().lower()
    correction_media = (correction_media or "").strip().lower() or media
    planning_media = (planning_media or "").strip().lower() or media
    retrieval = (retrieval or "").strip().lower()
    difficulty = (difficulty or "").strip().lower()
    continuity = (continuity or "").strip().lower()
    if continuity not in CONTINUITY:
        raise ValueError(
            f"Unknown continuity {continuity!r}; expected one of {CONTINUITY}"
        )
    for label, value in (
        ("correction_media", correction_media),
        ("planning_media", planning_media),
    ):
        if value not in MEDIA:
            raise ValueError(
                f"Unknown {label} {value!r}; expected one of {MEDIA}"
            )
    if retrieval not in RETRIEVAL:
        raise ValueError(f"Unknown retrieval {retrieval!r}; expected one of {RETRIEVAL}")
    if difficulty not in DIFFICULTY:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of {DIFFICULTY}"
        )
    if output_scale <= 0:
        raise ValueError("output_scale must be positive")
    # Constraint matrix: efficiency pins the other axes, and a conflicting
    # value is an error rather than a silent downgrade.
    if difficulty == "efficiency" and (
        correction_media != "text" or planning_media != "text" or retrieval != "none"
    ):
        raise SwitchConflictError(
            "difficulty=efficiency pins correction_media=text, planning_media=text "
            f"and retrieval=none; got correction_media={correction_media}, "
            f"planning_media={planning_media}, retrieval={retrieval}. Drop the "
            "conflicting switch or use difficulty=intermediate for a basic-capped prompt "
            "with other axes free."
        )
    return TranslationProfile(
        correction_media=correction_media,
        planning_media=planning_media,
        retrieval=retrieval,
        difficulty=difficulty,
        output_coefficient=output_coefficient(correction_media, retrieval),
        output_scale=output_scale,
        continuity=continuity,
    )


DEFAULT_PROFILE = resolve_profile()


# The six retired presets, for *reading old artifacts only* -- they are not a
# CLI input form any more. Frozen replay fixtures and archived
# research contexts still carry these strings.
LEGACY_PRESET_VECTORS = {
    "text-low": ("text", "none", "efficiency"),
    "text-med": ("text", "none", "quality"),
    "text-high": ("text", "native", "quality"),
    "mm-low": ("text", "local", "quality"),
    "mm-med": ("audio", "local", "quality"),
    "mm-high": ("video", "local", "quality"),
}


def parse_profile_id(profile_id: str) -> TranslationProfile:
    """Rebuild a profile from its canonical vector string.

    Used when reading a fingerprint or an artifact back (session_replay
    fixtures, resume checks). A retired ``route-level`` preset name is still
    accepted here and translated, so artifacts frozen before the switch
    refactor keep loading.
    """

    legacy = LEGACY_PRESET_VECTORS.get((profile_id or "").strip())
    if legacy is not None:
        return resolve_profile(*legacy)
    values = {}
    for part in (profile_id or "").split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key:
            values[key] = value.strip()
    # A pre-D20 vector carries the single ``media`` axis; it meant "both
    # tasks" by construction, so it maps onto both switches.
    has_media = "media" in values or (
        "correction_media" in values and "planning_media" in values
    )
    missing = {"retrieval", "difficulty"} - values.keys()
    if missing or not has_media:
        wanted = sorted(missing) + ([] if has_media else ["media"])
        raise ValueError(f"Invalid profile id {profile_id!r}: missing {wanted}")
    # Vectors frozen before the continuity axis (P7c) carry three components;
    # they were all serial by construction.
    return resolve_profile(
        values.get("media", ""),
        values["retrieval"],
        values["difficulty"],
        values.get("continuity", "serial"),
        correction_media=values.get("correction_media", ""),
        planning_media=values.get("planning_media", ""),
    )


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
