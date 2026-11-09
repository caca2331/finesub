"""What one LLM correction task is asked to do -- the value every front end
builds and ``correction_translation.run_correction_task`` runs.

The module CLI, ``run_pipeline``'s LLM stage and reference ingest each used to
carry the task's options down their own chain of keyword arguments, repeating
the defaults at every hop, and the CLI orchestrated the task a second time on
its own. Now each front end fills one of these and hands it over; the task's
defaults are declared here once. (``run_pipeline`` keeps its flat signature --
manifest rows are validated against it -- and packs its LLM keys into this
value before ASR runs.)

Deliberately light: ``stages`` builds one without importing the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from finesub import settings
from finesub.subtitles.postprocess import DEFAULT_POSTPROCESS_PROFILE

from .routing.profiles import (
    DEFAULT_PROFILE,
    KNOWLEDGE_MODES,
    PRESET_PARALLEL_WINDOWS_KEY,
    PRESET_SWITCH_VALUES,
    TranslationProfile,
)

#: plan = resolve and render the prompts, call nothing (the CLI's default);
#: research = run the preparation task only; execute = the whole task.
CORRECTION_MODES = ("plan", "research", "execute")


class CorrectionRequestError(ValueError):
    """A request the task refuses before doing any work: a contradiction in
    the options or an input it needs and does not have. The CLI prints it and
    exits 2; other callers see a ValueError, as before."""


#: What a switch nobody set resolves to: the harness's own run shape, plus
#: ``read`` for knowledge.
CODE_DEFAULTS = {
    "media": DEFAULT_PROFILE.correction_media,
    "retrieval": DEFAULT_PROFILE.retrieval,
    "difficulty": DEFAULT_PROFILE.difficulty,
    "continuity": DEFAULT_PROFILE.continuity,
    "knowledge": "read",
}


@dataclass(frozen=True)
class ResolvedSwitches:
    """The five correction switches after the precedence chain, and where each
    came from: ``explicit`` (the caller set it), ``preset:<id>`` or
    ``default`` (``CODE_DEFAULTS``)."""

    media: str
    retrieval: str
    difficulty: str
    continuity: str
    knowledge: str
    sources: Mapping[str, str]
    #: Correction windows at once under continuity=parallel; 1 otherwise when
    #: it came from a preset. Its source is in ``sources`` too.
    parallel_windows: int = 1
    #: The preset the routes resolved to (``[llm] preset`` or ``--llm-preset``).
    active_preset: str = "default"

    def vector(self) -> tuple[str, str, str, str, str]:
        return (self.media, self.retrieval, self.difficulty, self.continuity, self.knowledge)


def resolve_switches(
    explicit: Mapping[str, str | None], *, routes: Any = None, presets: bool = True
) -> ResolvedSwitches:
    """Fill the switches a front end left unset (``None``): the active preset's
    ``defaults``, then the default preset's, then ``CODE_DEFAULTS`` -- the
    same active-then-default order bindings fall back in (owner 2026-09-30).
    Values are not normalised here; ``resolve_profile`` does that.

    ``presets=False`` skips the preset layers without loading the routes: a
    run that stops before the LLM stage still validates the switches it was
    given, but must not fail on a route configuration it never uses.
    """

    layers: list[tuple[str, Mapping[str, Any]]] = []
    active_preset = "default"
    if presets:
        if routes is None:
            from .routing.model_routes import default_model_routes

            routes = default_model_routes()
        active_preset = routes.active_preset_id
        layers = [
            (preset_id, routes.presets[preset_id].defaults)
            for preset_id in dict.fromkeys((routes.active_preset_id, "default"))
        ]
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key in PRESET_SWITCH_VALUES:
        given = explicit.get(key)
        if given is not None:
            values[key], sources[key] = str(given), "explicit"
            continue
        for preset_id, defaults in layers:
            if key in defaults:
                values[key], sources[key] = defaults[key], f"preset:{preset_id}"
                break
        else:
            values[key], sources[key] = CODE_DEFAULTS[key], "default"
    parallel_windows = _resolve_parallel_windows(explicit, layers, values["continuity"], sources)
    return ResolvedSwitches(
        **values,
        sources=MappingProxyType(sources),
        parallel_windows=parallel_windows,
        active_preset=active_preset,
    )


def _resolve_parallel_windows(
    explicit: Mapping[str, Any],
    layers: Sequence[tuple[str, Mapping[str, Any]]],
    continuity: str,
    sources: dict[str, str],
) -> int:
    """Same chain as the switches. A preset's value applies only when the
    resolved continuity is parallel -- written for a parallel run, it is no
    reason to fail a serial one (the command line said serial, or continuity
    came from elsewhere); the source says so."""

    key = PRESET_PARALLEL_WINDOWS_KEY
    given = explicit.get(key)
    if given is not None:
        sources[key] = "explicit"
        return int(given)
    for preset_id, defaults in layers:
        if key in defaults:
            if continuity.strip().lower() != "parallel":
                sources[key] = f"default(preset:{preset_id}={defaults[key]} needs continuity=parallel)"
                return 1
            sources[key] = f"preset:{preset_id}"
            return int(defaults[key])
    sources[key] = "default"
    return 1


@dataclass(frozen=True)
class CorrectionRequest:
    stable_json: Path
    #: The final SRT. Only a plan may leave it out.
    output_path: Path | None = None
    mode: str = "execute"
    audio_path: Path | None = None
    video_path: Path | None = None
    profile: TranslationProfile = DEFAULT_PROFILE
    fast: str = "auto"
    extra_info: str = ""
    extra_style: str = ""
    #: Named style entries (`--style`). `None` falls through to the config,
    #: `""` is a decision to run without one — `resolve_style_selection` owns
    #: that precedence.
    style: str | Sequence[str] | None = None
    #: `none` / `read` / `update`; unset reads `[llm] style_mode`, then `read`.
    style_mode: str | None = None
    #: Whether `difficulty=efficiency` limits the style to the factory default
    #: (`resolve_style_selection`). Reference ingest turns it off: the style is
    #: what it learns, at every difficulty, so its `auto` picks there too
    #: (auto-style-gaps plan C).
    style_efficiency_cap: bool = True
    #: `none` / `read` / `update`; unset = `read` (`CODE_DEFAULTS`).
    knowledge: str | None = None
    #: Whether correction windows and the preparation task emit
    #: `<task_update_feedback>`. None = only under `update` (owner 2026-10-01:
    #: `read` reads and collects nothing). Reference ingest asks for it with
    #: `read`: its own second step updates the base from that feedback.
    collect_feedback: bool | None = None
    #: Where each switch came from (`ResolvedSwitches.sources`), for the
    #: task's `resolved-switches.json`. None = the caller gave the whole vector
    #: (reference ingest does).
    switch_sources: Mapping[str, str] | None = None
    #: None = the runtime's knowledge root (`paths.resolve_knowledge_root`).
    knowledge_root: str | Path | None = None
    refined_srt: Path | None = None
    task_id: str = ""
    task_summary: str = ""
    #: None = beside the output (`<final stem>` + ARTIFACT_DIR_SUFFIX); a plan
    #: without an output writes no artifacts at all.
    task_artifact_dir: Path | None = None
    #: A research context to use as is instead of preparing one.
    context_file: Path | None = None
    parallel_windows: int = 1
    #: None = `[llm] max_retries_per_window`, then the registry default.
    max_retries_per_window: int | None = None
    #: None = `[llm] max_replacements_per_window`, then the registry default.
    max_replacements_per_window: int | None = None
    #: None = stop at the translated SRT; an int = that final postprocess.
    postprocess_profile: int | None = DEFAULT_POSTPROCESS_PROFILE
    #: None = `[chunking] max_window_subtitle_tokens`, then the planning cap.
    max_window_subtitle_tokens: int | None = None
    #: The task's length ratio, which the merge thresholds follow (`--split-length-scale`). None = the style's
    #: length tier, then `[segmentation] length_scale`, then 1.0 (`length_tier.task_length_scale`, after the
    #: style is settled); a resumed task keeps the ratio it recorded.
    split_length_scale: float | None = None
    #: The LLM speaker judge (`stages/speaker_judge`): `off`, or the modality it judges in -- `text` /
    #: `audio` (pipeline: `--diarize judge-text` / `judge-audio`). Needs speaker labels on the stable.
    speaker_judge: str = "off"
    #: The separated vocals the judge listens to; None = beside the stable by the pipeline's rule,
    #: else `audio_path` with a warning.
    vocal_audio: Path | None = None
    #: Pass 3, the proofreading of the finished subtitles (`stages/proofread`): `auto` runs it at the quality
    #: difficulty, `on` at every difficulty, `off` never -- whichever way the task runs (`stages/three_pass`).
    proofread: str = "auto"
    resume: bool = True
    test_profile: bool = False
    #: Called with the rendered prompt artifacts once the task is planned and
    #: before anything runs -- the CLI prints or writes them there. Unset, no
    #: artifacts are rendered.
    on_plan: Callable[[Mapping[str, Any]], None] | None = None


def fast_mode_for_research_only(fast: str, *, research_only: bool) -> str:
    """``--research-only`` asks for the preparation task, which fast mode has
    none of: ``auto`` resolves to ``off`` for that run, ``on`` is a conflict."""

    if not research_only:
        return fast
    if fast == "on":
        raise ValueError(
            "--research-only conflicts with --fast on: fast mode runs no "
            "preparation task; drop one of them"
        )
    return "off"


#: The speaker judge's modes: off, or the modality it judges in (pipeline `--diarize judge-text|judge-audio`).
SPEAKER_JUDGE_MODES = ("off", "text", "audio")
#: Pass 3's modes (`--llm-proofread`).
PROOFREAD_MODES = ("auto", "on", "off")


def resolved_request(request: CorrectionRequest) -> CorrectionRequest:
    """Every refusal the task makes before doing anything, and the fields whose
    unset value depends on the others. Cheap and side-effect free, so a front
    end can ask before its expensive half (``run_pipeline`` asks before ASR);
    the task asks again, which is idempotent."""

    if request.mode not in CORRECTION_MODES:
        raise CorrectionRequestError(f"unknown correction mode {request.mode!r}")
    if request.speaker_judge not in SPEAKER_JUDGE_MODES:
        raise CorrectionRequestError(
            f"unknown speaker judge {request.speaker_judge!r}; use {'|'.join(SPEAKER_JUDGE_MODES)}"
        )
    if request.proofread not in PROOFREAD_MODES:
        raise CorrectionRequestError(
            f"unknown proofread mode {request.proofread!r}; use {'|'.join(PROOFREAD_MODES)}"
        )
    profile = request.profile
    knowledge = CODE_DEFAULTS["knowledge"] if request.knowledge is None else request.knowledge
    if knowledge == "collect":
        raise CorrectionRequestError(
            "--knowledge collect was renamed to read (2026-10-01); it now reads the base without"
            " collecting feedback -- use update to collect and write back"
        )
    if knowledge not in KNOWLEDGE_MODES:
        raise CorrectionRequestError(f"unknown knowledge mode {knowledge!r}; use none|read|update")
    # No coupling to the mode: `knowledge` says what happens *if* the windows
    # run. A plan takes `update` and simply never reaches the write; what
    # update does change there is the plan's product -- its prompts carry the
    # feedback-collection contract.
    if request.refined_srt and knowledge != "update":
        raise CorrectionRequestError("--refined-srt requires --knowledge update")
    research = request.mode == "research"
    execute = request.mode == "execute"
    if research and request.context_file:
        raise CorrectionRequestError("--research-only conflicts with --context-file")
    try:
        fast = fast_mode_for_research_only(request.fast, research_only=research)
    except ValueError as exc:
        raise CorrectionRequestError(str(exc)) from exc
    # A media switch above what the inputs provide is a configuration error,
    # not a silent downgrade (plan v2 D20).
    if execute and profile.uses_media and not request.audio_path:
        raise CorrectionRequestError(
            "--audio is required with --execute when any media switch is not text"
        )
    if request.video_path and not profile.uses_video:
        raise CorrectionRequestError("--video only applies when a media switch is video")
    if execute and profile.uses_video and not request.video_path:
        raise CorrectionRequestError(
            "--video is required with --execute when a media switch is video"
        )
    if execute and not request.output_path:
        raise CorrectionRequestError("--output is required with --execute")
    try:
        # None after this means "no cap written anywhere": the planning
        # limits' own cap applies. `0` disables it.
        window_cap = settings.MAX_WINDOW_SUBTITLE_TOKENS.resolve(request.max_window_subtitle_tokens)
        retries = settings.MAX_RETRIES_PER_WINDOW.resolve(request.max_retries_per_window)
        replacements = settings.MAX_REPLACEMENTS_PER_WINDOW.resolve(
            request.max_replacements_per_window
        )
        # Checked now, resolved once the style is settled (its tier sits between this and the file).
        if request.split_length_scale is not None:
            settings.SPLIT_LENGTH_SCALE.check(
                request.split_length_scale, origin=" (explicit value: a command-line flag, a manifest row or a caller)"
            )
    except ValueError as exc:
        raise CorrectionRequestError(str(exc)) from exc
    return replace(
        request,
        knowledge=knowledge,
        fast=fast,
        max_window_subtitle_tokens=window_cap,
        max_retries_per_window=retries,
        max_replacements_per_window=replacements,
    )
