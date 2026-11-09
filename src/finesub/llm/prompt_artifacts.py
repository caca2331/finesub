"""Dry-run prompt assembly shared by ``correction_translation --prompt-dir``.

``build_prompt_artifacts`` renders the preparation task's opening message and
every correction window's opening message for a stable JSON with placeholder
injections (no API calls); ``write_prompt_artifacts`` writes them as plan.json
+ per-session text files.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from finesub.media.clips import CLIP_AUDIO_SUFFIX, CLIP_VIDEO_SUFFIX
from .chunking import (
    load_segments_from_stable_json,
    plan_correction_windows,
)
from .knowledge.style import render_style_block
from .routing.config import (
    DEFAULT_LIMITS,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    CapabilityTier,
    research_search_query_limit,
)
from .knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    index_texts_present,
    load_index_texts,
)
from .length_tier import TaskLengthScale
from .prompt_variants import correct_only_name, resolve_variant
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile
from .prompts import (
    ContextPack,
    r1_request_cap,
)
from .prepare_prompts import build_prepare_task_messages
from .request_prompts import build_window_task_messages
from .routing.config import MAX_WINDOW_SEARCH_QUERIES
from .research import render_research_transcript
from .stages.task_graph import resolve_execution_strategy, window_flow
from .exchange_log import messages_to_text
from .stages.correction.metadata import _window_audio_label, window_to_metadata
from .token_budget import TokenCounter, default_token_counter, requested_output_limit

PREVIEW_AUDIO_LABEL = "[preview-audio.aac]"
PREVIEW_VIDEO_PATH = Path("preview-video.mp4")


def _cell_variant(profile: TranslationProfile) -> str | None:
    """The prompt set the run sends to the correction cell's first model: the cell's variant unless the model
    group's entry overrides it for that target (`RoleClient.complete` picks per candidate the same way); None
    without a route declaration, leaving the tier default (for a correct-only task, its correct-only set)."""

    from .routing.capabilities import correction_task_group
    from .routing.config import role_config_for

    try:
        config = role_config_for(correction_task_group(profile), profile.difficulty)
    except Exception:  # noqa: BLE001 - no declaration: the tier default, as the run's fallback
        variant = ""
    else:
        first = config.endpoint_chain[0].target_id if config.endpoint_chain else ""
        variant = config.variant_overrides.get(first, config.variant)
    if not profile.correct_only:
        return variant or None
    return correct_only_name(variant or resolve_variant(None).name)


def build_prompt_artifacts(
    *,
    stable_json: str | Path,
    #: The task's length ratio as the dry run resolves it (`length_tier.task_length_scale`): written into the
    #: windows it plans, and stated with its source in the artifacts.
    length_scale: TaskLengthScale,
    audio_path: str | Path | None = None,
    video_path: str | Path | None = None,
    audio_label: str = "",
    extra_info: str = "",
    context_pack: ContextPack | None = None,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    #: Same selection the run would use, so the dry-run artifact shows the
    #: prompt the run would actually send.
    style_names: Sequence[str] | None = (),
    #: `--style auto`: the candidate list both tasks pick from, rendered.
    style_catalog: str = "",
    extra_style: str = "",
    task_update_feedback: bool = False,
    counter: TokenCounter | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    knowledge_indexes: Mapping[str, str] | None = None,
    #: The run's own `--test-profile`: it pins one stub endpoint, so the
    #: strategy it records is the default. Resolving the dry run against the
    #: real chain instead would show a prompt shape the run will not send.
    test_profile: bool = False,
    windows_override: Sequence[Any] | None = None,
    max_window_subtitle_tokens: int | None = None,
    speakers: bool = False,
    #: Three passes: pass 2's prompt, on the ASR lines since pass 1 has not run (`three_pass.preview_windows`).
    grouped: bool = False,
) -> Dict[str, Any]:
    segments = load_segments_from_stable_json(stable_json)
    research_query_limit = research_search_query_limit(len(segments))
    counter = counter or default_token_counter()
    if windows_override is not None:
        windows = list(windows_override)
    else:
        from finesub.media.clips import probe_audio_duration

        from .routing.capabilities import correction_planning_limits

        audio_duration = probe_audio_duration(audio_path) if audio_path else None
        windows = plan_correction_windows(
            segments,
            counter=counter,
            context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
            audio_duration=audio_duration,
            profile=profile,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
            # the run's own envelope, or the preview shows windows the run will not cut
            limits=correction_planning_limits(profile),
        )
    transcript = render_research_transcript(segments, windows)
    windows = [replace(window, length_scale=length_scale.value) for window in windows]
    if grouped:
        from .stages.three_pass import preview_windows

        windows = preview_windows(windows)
    # Knowledge reads follow the tri-state here exactly as they do at run
    # time, or the plan shows indices a `--knowledge none` run never loads.
    index_texts = (
        knowledge_indexes
        if knowledge_indexes is not None
        else (load_index_texts(knowledge_root) if knowledge_enabled else {})
    )
    entries_available = index_texts_present(index_texts)
    # The preparation task: the dry-run must show the prompt the run sends --
    # which means its own strategy, not the window's. The production default
    # (an agent preset plus `--media audio`) is exactly the split where the two
    # differ: the window carries a clip and is batched, the text-only
    # preparation task can be agentic. Mirrors `stages/prepare_task.py`.
    prepare_strategy, _prepare_source = resolve_execution_strategy(
        profile,
        task_group="research",
        test_profile=test_profile,
        knowledge_readable=entries_available,
    )
    prepare_agentic = prepare_strategy == "agentic" and (
        profile.external_injection or entries_available
    )
    # The windows' flow, from the one owner (`task_graph.window_flow`): the
    # preparation task's downstream caps and the window prompts both read it.
    flow = window_flow(profile, knowledge_readable=entries_available, test_profile=test_profile)
    research_messages = [
        build_prepare_task_messages(
            transcript=transcript,
            extra_info=extra_info,
            knowledge_indexes=index_texts,
            max_search_queries=research_query_limit,
            emits_queries=profile.external_injection,
            emits_entries=entries_available,
            native_search=profile.native_search,
            correct_only=profile.correct_only,
            max_requested_entries=r1_request_cap(
                downstream_can_request=flow.window_requests
            ),
            collect_task_feedback=task_update_feedback,
            emits_keep=(flow.window_requests or prepare_agentic) and entries_available,
            agentic=prepare_agentic,
            style_catalog=style_catalog,
            extra_style=extra_style,
            speakers=speakers,
        )
    ]
    # None: a model still picks the style, so no block is certain yet (`_previewed_style_names`).
    style_block = "" if style_names is None else render_style_block(
        knowledge_root, style_names, correct_only=profile.correct_only)
    use_video = bool(video_path) and profile.correction_use_video
    # Plan §8, last bullet: the dry run must show the window prompt the run
    # would send -- the request-channel opening under the batched strategy,
    # the tool-session shape under agentic.
    task_flow_shape: Dict[str, Any] = {
        "knowledge_enabled": entries_available,
    }
    if flow.window_requests:
        task_flow_shape.update(
            {
                "request_channel": flow.request_channel,
                "agentic": flow.agent_fetches,
                "knowledge_indexes": index_texts if entries_available else {},
                "max_search_queries": MAX_WINDOW_SEARCH_QUERIES,
            }
        )
    correction_messages = [
        build_window_task_messages(
            **task_flow_shape,
            window=window,
            context_pack=context_pack,
            audio_file_label=_window_audio_label(
                video_path if use_video else audio_path,
                audio_label or (
                    PREVIEW_VIDEO_PATH.name if use_video else PREVIEW_AUDIO_LABEL
                ),
                window,
                clip_suffix=CLIP_VIDEO_SUFFIX if use_video else CLIP_AUDIO_SUFFIX,
            ),
            style_block=style_block,
            task_update_feedback=task_update_feedback,
            profile=profile,
            variant=_cell_variant(profile),
        )
        for window in windows
    ]
    # The basic tier's default variant (basicA) for the first correction window, so prompt iteration can read
    # it beside a capable cell's prompt; at efficiency it equals the window prompt. Other windows differ only
    # in user content. None under three passes: no basic variant has a grouped prompt set.
    correction_basic_example = (
        build_window_task_messages(
            **task_flow_shape,
            window=windows[0],
            context_pack=context_pack,
            audio_file_label=_window_audio_label(
                video_path if use_video else audio_path,
                audio_label or (
                    PREVIEW_VIDEO_PATH.name if use_video else PREVIEW_AUDIO_LABEL
                ),
                windows[0],
                clip_suffix=CLIP_VIDEO_SUFFIX if use_video else CLIP_AUDIO_SUFFIX,
            ),
            style_block=style_block,
            task_update_feedback=task_update_feedback,
            profile=profile,
            tier=CapabilityTier.BASIC,
            variant=(correct_only_name(resolve_variant(None, CapabilityTier.BASIC).name)
                     if profile.correct_only else None),
        )
        if windows and not grouped
        else None
    )
    return {
        "model_limits": {
            "prompt_input_limit": DEFAULT_LIMITS.prompt_input_limit,
            "output_limit": DEFAULT_LIMITS.output_limit,
            "requested_output_limit": requested_output_limit(DEFAULT_LIMITS),
        },
        "profile": {
            "id": profile.profile_id,
            "output_coefficient": profile.output_coefficient,
            "output_scale": profile.output_scale,
        },
        "research_search_query_limit": research_query_limit,
        "length_scale": length_scale.to_metadata(),
        "correction_windows": [window_to_metadata(window) for window in windows],
        "research_messages": research_messages,
        "correction_messages": correction_messages,
        "correction_basic_example_messages": correction_basic_example,
    }


def write_prompt_artifacts(artifacts: Dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    plan = {
        key: value
        for key, value in artifacts.items()
        if key
        not in {
            "research_messages",
            "correction_messages",
            "correction_basic_example_messages",
            "speaker_judge_messages",
        }
    }
    (output_path / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for messages in artifacts.get("research_messages") or []:
        (output_path / "research-prepare.txt").write_text(
            messages_to_text(messages),
            encoding="utf-8",
        )
    for idx, messages in enumerate(artifacts["correction_messages"], start=1):
        (output_path / f"correction-{idx:04d}.txt").write_text(
            messages_to_text(messages),
            encoding="utf-8",
        )
    for messages in artifacts.get("speaker_judge_messages") or []:
        (output_path / "speaker-judge-0001.txt").write_text(messages_to_text(messages), encoding="utf-8")
    if artifacts.get("correction_basic_example_messages"):
        (output_path / "correction-0001-basic-tier.txt").write_text(
            messages_to_text(artifacts["correction_basic_example_messages"]),
            encoding="utf-8",
        )
