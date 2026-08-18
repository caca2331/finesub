"""Dry-run prompt assembly shared by ``correction_translation --prompt-dir``.

``build_prompt_artifacts`` renders every stage prompt for a stable JSON with
placeholder injections (no API calls); ``write_prompt_artifacts`` writes them
as plan.json + per-session text files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from finesub.media.clips import CLIP_AUDIO_SUFFIX, CLIP_VIDEO_SUFFIX
from .chunking import (
    load_segments_from_stable_json,
    plan_correction_windows,
)
from .knowledge.mistakes import render_featured_mistakes_block
from .routing.config import (
    DEFAULT_LIMITS,
    DEFAULT_RESEARCH_SEARCH_ROUNDS,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    CapabilityTier,
    followup_search_query_limit,
    research_search_query_limit,
)
from .knowledge.base import DEFAULT_KNOWLEDGE_ROOT, load_index_text
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile
from .prompts import (
    ContextPack,
    build_correction_csv_messages,
    build_correction_query_messages,
    build_research_round1_messages,
    build_research_round2_messages,
    build_search_loop_messages,
    r1_request_cap,
)
from .research import render_research_transcript, research_stage_runs
from .exchange_log import messages_to_text
from .stages.correction.metadata import _window_audio_label, window_to_metadata
from .token_budget import TokenCounter, default_token_counter, requested_output_limit

PREVIEW_AUDIO_LABEL = "[preview-audio.aac]"
PREVIEW_VIDEO_PATH = Path("preview-video.mp4")

PLACEHOLDER_ROUND1_NOTES = "（第一轮 <analysis_notes> 分析要点将注入此处）"
PLACEHOLDER_ENTRY_DETAILS = "（第一轮 requested_entries 对应的条目详情将按预算渲染注入此处）"
PLACEHOLDER_EVIDENCE_PACK = "（多轮搜索 loop 生成的 Evidence Pack 将注入此处）"
PLACEHOLDER_SEARCH_RESULTS = "（本地搜索代理对第一轮 query 的搜索结果将注入此处）"
PLACEHOLDER_WINDOW_SEARCH = "（本窗口前置搜索 query 的本地搜索结果将注入此处）"
PLACEHOLDER_LOOP_BACKGROUND = "（用户额外信息与第一轮 <analysis_notes> 将注入此处）"
PLACEHOLDER_LOOP_CONTRACT = "（第一轮 <research_contract> JSON 将注入此处，priority 为当前值）"
PLACEHOLDER_LOOP_EXECUTED = "（已执行过的搜索 query 将注入此处）"
PLACEHOLDER_LOOP_PROGRESS = "（此前各轮 <progress_update> 增量的累积将注入此处）"
PLACEHOLDER_LOOP_ROUND_RESULTS = (
    "--- query: 游戏A 剧情考据 ---\n"
    "provider: preview\n- 示例来源 (https://example.test/search)\n  搜索摘要。\n\n"
    "--- 深度提取 url: https://example.test/page ---\n"
    "provider: preview\n页面正文摘录。"
)


def build_prompt_artifacts(
    *,
    stable_json: str | Path,
    audio_path: str | Path | None = None,
    video_path: str | Path | None = None,
    audio_label: str = "",
    extra_info: str = "",
    context_pack: ContextPack | None = None,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    task_update_feedback: bool = False,
    research_search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    counter: TokenCounter | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    streamer_index: str | None = None,
    common_index: str | None = None,
    windows_override: Sequence[Any] | None = None,
    max_window_subtitle_tokens: int | None = None,
) -> Dict[str, Any]:
    segments = load_segments_from_stable_json(stable_json)
    research_query_limit = research_search_query_limit(len(segments))
    multi_round_research = int(research_search_rounds) > 1
    counter = counter or default_token_counter()
    if windows_override is not None:
        windows = list(windows_override)
    else:
        from finesub.media.clips import probe_audio_duration

        audio_duration = probe_audio_duration(audio_path) if audio_path else None
        windows = plan_correction_windows(
            segments,
            counter=counter,
            context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
            audio_duration=audio_duration,
            profile=profile,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
        )
    transcript = render_research_transcript(segments, windows)
    # Knowledge reads follow the tri-state here exactly as they do at run
    # time, or the plan shows indices a `--knowledge none` run never loads.
    streamer_index_text = (
        streamer_index
        if streamer_index is not None
        else (load_index_text(knowledge_root, "streamer") if knowledge_enabled else "")
    )
    common_index_text = (
        common_index
        if common_index is not None
        else (load_index_text(knowledge_root, "common") if knowledge_enabled else "")
    )
    # Which research rounds this vector runs, decided by the production
    # predicates rather than a second copy of the rule (docs/llm_harness_behavior.md).
    entries_available = bool(streamer_index_text.strip() or common_index_text.strip())
    stage_runs = research_stage_runs(
        profile, knowledge_root=knowledge_root, knowledge_enabled=knowledge_enabled
    )
    run_round1 = stage_runs and (profile.external_injection or entries_available)
    run_round2 = stage_runs and profile.retrieval != "none"

    research_messages = []
    if run_round1:
        research_messages.append(
            build_research_round1_messages(
                transcript=transcript,
                extra_info=extra_info,
                streamer_index=streamer_index_text,
                common_index=common_index_text,
                max_search_queries=research_query_limit,
                use_search_contract=multi_round_research,
                emits_queries=profile.external_injection,
                emits_entries=entries_available,
                emits_notes=run_round2,
                max_requested_entries=r1_request_cap(
                    downstream_can_request=profile.external_injection
                ),
            )
        )
    if run_round2:
        research_messages.append(
            build_research_round2_messages(
                transcript=transcript,
                extra_info=extra_info,
                round1_notes=PLACEHOLDER_ROUND1_NOTES,
                entry_details_text=(
                    PLACEHOLDER_ENTRY_DETAILS if entries_available else ""
                ),
                search_results=(
                    ""
                    if profile.native_search
                    else (
                        PLACEHOLDER_EVIDENCE_PACK
                        if multi_round_research
                        else PLACEHOLDER_SEARCH_RESULTS
                    )
                ),
                use_evidence_pack=multi_round_research and not profile.native_search,
                native_search=profile.native_search,
                emits_keep=profile.external_injection,
            )
        )
    if profile.external_injection:
        search_loop_example = (
            build_search_loop_messages(
                round_index=0,
                max_rounds=int(research_search_rounds),
                is_final_round=False,
                background=PLACEHOLDER_LOOP_BACKGROUND,
                contract_json=PLACEHOLDER_LOOP_CONTRACT,
                executed_queries=[PLACEHOLDER_LOOP_EXECUTED],
                progress_log=PLACEHOLDER_LOOP_PROGRESS,
                search_results=PLACEHOLDER_LOOP_ROUND_RESULTS,
                previous_requested_entries=["条目甲"],
                previous_kept_entries=["条目乙"],
                previous_contract_json=PLACEHOLDER_LOOP_CONTRACT,
                previous_search_queries=["F1|游戏A 剧情考据 >> 确认官方名"],
                previous_extract_urls=[
                    "https://example.test/page >> 提取角色与阵营段落"
                ],
                followup_query_cap=followup_search_query_limit(research_query_limit),
            )
            if multi_round_research
            else None
        )
        # The query round's clip follows ``planning_media`` (model-routing v2).
        planning_video = bool(video_path) and profile.planning_use_video
        correction_query_messages = [
            build_correction_query_messages(
                window=window,
                context_pack=context_pack,
                audio_file_label=_window_audio_label(
                    video_path if planning_video else audio_path,
                    audio_label
                    or (
                        PREVIEW_VIDEO_PATH.name
                        if planning_video
                        else PREVIEW_AUDIO_LABEL
                    ),
                    window,
                    clip_suffix=(
                        CLIP_VIDEO_SUFFIX if planning_video else CLIP_AUDIO_SUFFIX
                    ),
                ),
                profile=profile,
            )
            for window in windows
        ]
    else:
        search_loop_example = None
        correction_query_messages = []

    common_mistakes_block = (
        render_featured_mistakes_block(knowledge_root) if knowledge_enabled else ""
    )
    use_video = bool(video_path) and profile.correction_use_video
    correction_messages = [
        build_correction_csv_messages(
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
            search_results=(
                PLACEHOLDER_WINDOW_SEARCH if profile.external_injection else ""
            ),
            common_mistakes_block=common_mistakes_block,
            task_update_feedback=task_update_feedback,
            profile=profile,
        )
        for window in windows
    ]
    # BASIC-tier variant of the first correction window so dry-run prompt
    # iteration can inspect the 1:1 merge fragments (sent only when a call
    # falls back to a basic-capability endpoint); other windows differ only
    # in user content, not in the tiered system prompt.
    correction_basic_example = (
        build_correction_csv_messages(
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
            search_results=(
                PLACEHOLDER_WINDOW_SEARCH if profile.external_injection else ""
            ),
            common_mistakes_block=common_mistakes_block,
            task_update_feedback=task_update_feedback,
            profile=profile,
            tier=CapabilityTier.BASIC,
        )
        if windows
        else None
    )
    return {
        "model_limits": {
            "context_limit": DEFAULT_LIMITS.context_limit,
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
        "research_search_rounds": int(research_search_rounds),
        "correction_windows": [window_to_metadata(window) for window in windows],
        "research_messages": research_messages,
        "search_loop_example_messages": search_loop_example,
        "correction_query_messages": correction_query_messages,
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
            "search_loop_example_messages",
            "correction_query_messages",
            "correction_messages",
            "correction_basic_example_messages",
            "fast_round1_messages",
        }
    }
    (output_path / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for idx, messages in enumerate(artifacts.get("research_messages") or [], start=1):
        (output_path / f"research-round{idx}.txt").write_text(
            messages_to_text(messages),
            encoding="utf-8",
        )
    if artifacts.get("search_loop_example_messages"):
        (output_path / "research-search-loop-example.txt").write_text(
            messages_to_text(artifacts["search_loop_example_messages"]),
            encoding="utf-8",
        )
    if artifacts.get("fast_round1_messages"):
        (output_path / "fast-round1.txt").write_text(
            messages_to_text(artifacts["fast_round1_messages"]),
            encoding="utf-8",
        )
    for idx, messages in enumerate(artifacts.get("correction_query_messages", []), start=1):
        (output_path / f"correction-{idx:04d}-query.txt").write_text(
            messages_to_text(messages),
            encoding="utf-8",
        )
    for idx, messages in enumerate(artifacts["correction_messages"], start=1):
        (output_path / f"correction-{idx:04d}.txt").write_text(
            messages_to_text(messages),
            encoding="utf-8",
        )
    if artifacts.get("correction_basic_example_messages"):
        (output_path / "correction-0001-basic-tier.txt").write_text(
            messages_to_text(artifacts["correction_basic_example_messages"]),
            encoding="utf-8",
        )
