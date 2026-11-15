"""Orthogonal switch axes for the correction/translation run.

The switch axes live here; ``knowledge`` stays an independent CLI argument
because it is threaded through the ``finesub.llm.knowledge`` modules, which have no
business knowing about profiles -- only its value set sits beside the others,
so a preset's ``defaults`` table is checked against one vocabulary. Canonical behavior is documented in
``docs/llm_harness_behavior.md``.

- ``correction_media``  text / audio / video, a ladder (video implies
  audio): what the correction window -- the one task that reads media --
  is given. The CLI knob is ``--media``; the field keeps the task's name
  because the prompt layer reads it as "the correction window's media". A
  switch above what the input files provide is a configuration error at the
  entrypoints, not a silent downgrade.
- ``retrieval`` none / local / native. ``local`` is the whole harness-side
  injection machinery (the preparation task's fill, the window request
  turn, the local search agent); ``native`` is the model's own search tool
  and never coexists with harness search; ``none`` is neither.
- ``difficulty`` quality / intermediate / efficiency -- a positioning, not a
  capability tier read off the answering endpoint (that column is gone): it
  picks the task-group cell's prompt variant and that cell's thinking level,
  and the correction task reads it for how many passes it may run and whether
  pass 3 runs by default (``stages/three_pass``). It does not touch the window
  geometry, which is what lets an explicit mid-run switch reuse completed
  windows, and it pins no other axis. A preset may also bind a different model
  group per difficulty -- the shipped one does for correction (the capable
  group at ``quality`` and ``intermediate``, the lites at ``efficiency``), so
  switching to ``efficiency`` is the declared, user-initiated way to keep going
  when the free capable quota runs out.

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
# Tri-state knowledge switch (docs/knowledge.md; renamed 2026-10-01):
# none = the base is not read; read = read and inject it, nothing collected;
# update = read + correction/research emit task_update_feedback + run the
# unified knowledge update after the task.
KNOWLEDGE_MODES = ("none", "read", "update")
#: The switches a preset may default (`[llm.presets.<id>.defaults]`) and the
#: values each takes -- the same sets as the command-line switches. A preset
#: may also default `parallel_windows` (PRESET_PARALLEL_WINDOWS_KEY), the one
#: integer among them.
PRESET_SWITCH_VALUES = {
    "media": MEDIA,
    "retrieval": RETRIEVAL,
    "difficulty": DIFFICULTY,
    "continuity": CONTINUITY,
    "knowledge": KNOWLEDGE_MODES,
}
#: The integer a preset may default besides the switches (owner 2026-10-01):
#: how many correction windows run at once under continuity=parallel. It is
#: backend-specific -- what agy carries, a free API may not -- so it lives on
#: the preset, not in a global setting.
PRESET_PARALLEL_WINDOWS_KEY = "parallel_windows"

# Rank for the media ladder: "media >= audio" is a real comparison.
_MEDIA_RANK = {name: index for index, name in enumerate(MEDIA)}

# Additive output-budget coefficient. It is not calibrated per switch vector:
# measured output length moves with the model and its reasoning effort far more
# than with any switch (owner 2026-09-30).
BASE_COEFF = 2.0            # correction + translation proper
THINKING_COEFF = 1.5        # unconditional since model-routing v2 (efficiency incl.)
RETRIEVAL_COEFF = 1.0       # native search or local injection

# Normal windows: expected output must fit 0.9 x output_limit - 5k.
WINDOW_OUTPUT_FILL_RATIO = 0.9
WINDOW_OUTPUT_SLACK_TOKENS = 5_000
# Fast mode treats the whole input as one window under a stricter budget.
FAST_OUTPUT_FILL_RATIO = 0.8
FAST_OUTPUT_SLACK_TOKENS = 10_000
# The fast window's opening message must leave input headroom for the fill
# turn of its request channel: the entry block (<=28k) + search results
# (<=20k) + the model's own request turn + scaffolding/static-prompt delta.
FAST_FILL_INPUT_RESERVE_TOKENS = 56_000

# Gemini video tokens: tokens/frame x sample fps (low resolution default).
VIDEO_TOKENS_PER_FRAME_LOW = 71
VIDEO_TOKENS_PER_FRAME_HIGH = 269
VIDEO_SAMPLE_FPS = 0.25


def output_coefficient(retrieval: str) -> float:
    # Media no longer appears (owner 2026-09-30): the clip costs input, not
    # output, and the measured output length moves with the model and its
    # reasoning effort far more than with any switch.
    #
    # difficulty no longer appears either: it only picks the prompt variant and the
    # thinking level, so ``efficiency`` lost its discount and the
    # +1.5 thinking term is unconditional. That moves text/none/efficiency from
    # the measured c=2.0 to 3.5 (window capacity 57%).
    coefficient = BASE_COEFF + THINKING_COEFF
    if retrieval != "none":
        coefficient += RETRIEVAL_COEFF
    return coefficient


@dataclass(frozen=True)
class TranslationProfile:
    correction_media: str
    retrieval: str
    difficulty: str
    output_coefficient: float
    # User scale k (--output-scale); larger k means smaller windows.
    output_scale: float = 1.0
    continuity: str = "serial"
    # The task only corrects (a Chinese source): no switch, the task decides it from the source's language
    # (`stages/three_pass.task_correct_only`). The correction cell's prompt set becomes its correct-only
    # derivation (`prompt_variants.correct_only_variant`).
    correct_only: bool = False

    @property
    def profile_id(self) -> str:
        """Canonical switch vector -- the artifact/display form."""

        return (
            f"correction_media={self.correction_media},"
            f"retrieval={self.retrieval},"
            f"difficulty={self.difficulty},continuity={self.continuity}"
            + (",correct_only=true" if self.correct_only else "")
        )

    @property
    def geometry_id(self) -> str:
        """Exactly the switches that move a window boundary.

        These are the profile attributes the planner actually reads:
        ``estimate_window_budget`` adds the clip's tokens for
        ``correction_media``, and ``max_window_csv_tokens`` divides by
        ``output_scale * output_coefficient`` (``retrieval`` folds into the
        coefficient). ``continuity`` and ``difficulty`` never reach the
        planner, so they cannot shift a chunk id.

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

    # Named for the task that reads the media (plan v2 D20); ``uses_media``
    # is the clip/upload gate.
    @property
    def correction_use_audio(self) -> bool:
        return _MEDIA_RANK[self.correction_media] >= _MEDIA_RANK["audio"]

    @property
    def correction_use_video(self) -> bool:
        return self.correction_media == "video"

    @property
    def uses_media(self) -> bool:
        return self.correction_media != "text"

    @property
    def uses_video(self) -> bool:
        return self.correction_media == "video"

    @property
    def native_search(self) -> bool:
        return self.retrieval == "native"

    @property
    def external_injection(self) -> bool:
        """Whether the harness runs its own retrieval (preparation fill, request turn)."""

        return self.retrieval == "local"

    def with_output_scale(self, output_scale: float) -> "TranslationProfile":
        return replace(self, output_scale=output_scale)


def resolve_profile(
    # Defaults are the harness's own default run shape. Front ends do not lean
    # on them: they resolve every switch first
    # (`correction_request.resolve_switches`) and pass all four. Difficulty
    # defaults to intermediate (owner 2026-11-08): quality's speaker judge and
    # proofread are opt-in.
    media: str = "audio",
    retrieval: str = "local",
    difficulty: str = "intermediate",
    continuity: str = "serial",
    *,
    correction_media: str = "",
    output_scale: float = 1.0,
    correct_only: bool = False,
) -> TranslationProfile:
    """Build the switch vector.

    ``media`` is the CLI knob; ``correction_media`` is the same switch under
    the field's name (an artifact form parses back through it).
    """

    media = (media or "").strip().lower()
    correction_media = (correction_media or "").strip().lower() or media
    retrieval = (retrieval or "").strip().lower()
    difficulty = (difficulty or "").strip().lower()
    continuity = (continuity or "").strip().lower()
    if continuity not in CONTINUITY:
        raise ValueError(
            f"Unknown continuity {continuity!r}; expected one of {CONTINUITY}"
        )
    if correction_media not in MEDIA:
        raise ValueError(f"Unknown media {correction_media!r}; expected one of {MEDIA}")
    if retrieval not in RETRIEVAL:
        raise ValueError(f"Unknown retrieval {retrieval!r}; expected one of {RETRIEVAL}")
    if difficulty not in DIFFICULTY:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of {DIFFICULTY}"
        )
    if output_scale <= 0:
        raise ValueError("output_scale must be positive")
    return TranslationProfile(
        correction_media=correction_media,
        retrieval=retrieval,
        difficulty=difficulty,
        output_coefficient=output_coefficient(retrieval),
        output_scale=output_scale,
        continuity=continuity,
        correct_only=correct_only,
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
    # Two spellings of the one media switch: the artifact form
    # ``correction_media=`` and the bare ``media=``. A vector frozen while the
    # planning-media axis existed (2026-08 to 2026-09-09) still carries
    # ``planning_media=``; it is ignored -- that task is gone.
    has_media = "media" in values or "correction_media" in values
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
        correct_only=values.get("correct_only") == "true",
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
