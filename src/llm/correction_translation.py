"""CLI and orchestration for LLM subtitle correction translation.

The window-execution loop lives in ``llm.stages.correction_loop``; research
acquisition in ``llm.research``; the unified knowledge update in
``llm.knowledge.update``. This module keeps the CLI, dry-run prompt artifacts
and the top-level orchestration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping

from asr_playground.run_metadata import (
    metadata_path_for_output,
    stage_record,
    summarize_llm_rounds,
    update_run_metadata,
)

from asr_playground.media.clips import probe_audio_duration
from .knowledge.mistakes import render_featured_mistakes_block
from .api_keys import read_config
from .config import (
    DEFAULT_RESEARCH_SEARCH_ROUNDS,
    research_search_query_limit,
)
from .knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
)
from .knowledge.update import (
    ensure_research_context_path,
    run_knowledge_update,
)
from .profiles import (
    DEFAULT_FAST_SEARCH_ROUNDS,
    DEFAULT_PROFILE,
    LEVELS,
    ROUTES,
    TranslationProfile,
    resolve_profile,
)
from .prompts import (
    ContextPack,
    build_fast_round1_messages,
)
from .prompt_artifacts import build_prompt_artifacts, write_prompt_artifacts
from .research import (
    load_research_context,
    planning_metadata,
    load_preinjected_entries,
    run_research_stage,
)
from asr_playground.subtitles.postprocess import (
    DEFAULT_POSTPROCESS_PROFILE,
    SUPPORTED_POSTPROCESS_PROFILES,
    postprocess_srt_file,
)
from .stages.correction_loop import execute_correction_windows
from .stages.fast_session import FastSessionResult, acquire_fast_context
from .stages.plan import FastDecision, decide_fast_mode
from .task_report import write_task_report
from .token_budget import default_token_counter, TokenCounter


# Tri-state knowledge switch (docs/knowledge.md):
# none = no collection, collect = correction/research emit task_update_feedback,
# update = collect + run the unified knowledge update after the task.
KNOWLEDGE_MODES = ("none", "collect", "update")


def knowledge_collects(knowledge: str) -> bool:
    return knowledge in ("collect", "update")


def run_post_correction_knowledge_update(
    *,
    task_id: str,
    task_summary: str,
    result_srt_path: str | Path,
    output_path: str | Path,
    stable_json: str | Path,
    artifact_dir: str | Path,
    refined_srt: str | Path | None = None,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    test_profile: bool = False,
    counter: TokenCounter | None = None,
) -> Dict[str, Any]:
    """Run the unified knowledge update right after a correction task.

    ``result_srt_path`` is whatever SRT the run produced (the postprocessed
    final SRT, or the translated SRT when postprocess was deferred) — the
    final_csv overlay uses it directly. Sibling paths derive from
    ``output_path`` (the task's final SRT anchor); ``stable_json`` is passed
    explicitly because it may live under a different stem (reference ingest).
    """

    out = Path(output_path).expanduser().resolve()
    base = out.with_suffix("")
    artifact_path = Path(artifact_dir).expanduser().resolve()
    report = run_knowledge_update(
        final_srt=result_srt_path,
        stable_json=stable_json,
        annotated_csv=base.with_name(f"{base.name}-annotated.csv"),
        research_context=ensure_research_context_path(
            artifact_dir=artifact_path,
            stem=base.name,
            run_dir=out.parent,
        ),
        artifact_dir=artifact_path,
        refined_srt=refined_srt,
        task_id=task_id,
        task_summary=task_summary,
        knowledge_root=knowledge_root,
        test_profile=test_profile,
        token_counter=counter,
    )
    write_task_report(
        artifact_path,
        task_id=task_id,
        outputs={
            "final_srt": str(Path(result_srt_path).expanduser().resolve()),
            "knowledge_update_ledger": report["ledger_path"],
        },
    )
    return report


def _fast_execute_kwargs(
    decision: FastDecision,
    fast_ctx: FastSessionResult | None,
    fast_file_ref: Any | None,
    profile: TranslationProfile,
) -> Dict[str, Any]:
    """execute_correction_windows kwargs for an enabled fast decision."""

    kwargs: Dict[str, Any] = {"windows_override": [decision.window]}
    if profile.external_injection and fast_ctx is not None:
        kwargs.update(
            seed_query_results=fast_ctx.seed_query_results(),
            entry_details=fast_ctx.entry_details_text,
            evidence_pack_mode=fast_ctx.evidence_pack_mode,
            extra_fingerprint=fast_ctx.fingerprint(),
        )
        if fast_file_ref is not None:
            kwargs["file_ref_seed"] = {decision.window.chunk_id: fast_file_ref}
    return kwargs


def resolve_chunking_subtitle_cap() -> int | None:
    """``[chunking] max_window_subtitle_tokens`` from config.toml.

    Returns ``None`` when the key is absent (callers then fall back to
    ``limits.max_window_subtitle_tokens``, default 10k); ``0`` disables the
    cap; a malformed or negative value is a hard error so a typo never
    silently changes windowing.
    """

    data = read_config()
    section = data.get("chunking") if isinstance(data, Mapping) else None
    if not isinstance(section, Mapping):
        return None
    raw = section.get("max_window_subtitle_tokens")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "[chunking] max_window_subtitle_tokens must be an integer"
        ) from None
    if value < 0:
        raise ValueError("[chunking] max_window_subtitle_tokens must be >= 0")
    return value


def _run_full_correction_impl(
    *,
    stable_json: str | Path,
    output_path: str | Path,
    audio_path: str | Path | None,
    video_path: str | Path | None = None,
    extra_info: str = "",
    profile: TranslationProfile = DEFAULT_PROFILE,
    fast: str = "auto",
    fast_search_rounds: int = DEFAULT_FAST_SEARCH_ROUNDS,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    max_retries_per_window: int = 5,
    enable_web_search: bool = True,
    research_search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    postprocess_profile: int | None = DEFAULT_POSTPROCESS_PROFILE,
    extra_style: str = "",
    task_id: str = "",
    task_summary: str = "",
    task_artifact_dir: str | Path | None = None,
    knowledge: str = "none",
    refined_srt: str | Path | None = None,
    test_profile: bool = False,
    resume: bool = True,
    max_window_subtitle_tokens: int | None = None,
) -> Path:
    """Research + correction windows + optional unified knowledge update.

    ``knowledge`` is the tri-state switch (none/collect/update); ``update``
    implies collection. ``refined_srt`` (with ``knowledge="update"``) switches
    the update to the refined_aligned evidence mode.

    ``max_window_subtitle_tokens`` caps one window's ``<asr_result>`` CSV input
    (quality guardrail, default from config.toml ``[chunking]`` then
    ``limits.max_window_subtitle_tokens``); ``0`` disables it.

    Programmatic equivalent of the CLI's --execute path, used by
    asr_playground.workflows.reference_ingest. An existing
    *-research-context.json under the task
    artifact directory (or a legacy sibling next to the output SRT) is reused
    so reruns skip the research rounds.
    """
    if knowledge not in KNOWLEDGE_MODES:
        raise ValueError(f"unknown knowledge mode {knowledge!r}; use none|collect|update")
    collect_feedback = knowledge_collects(knowledge)
    if max_window_subtitle_tokens is None:
        max_window_subtitle_tokens = resolve_chunking_subtitle_cap()
    out = Path(output_path).expanduser().resolve()
    task_id = task_id or Path(stable_json).stem
    task_summary = task_summary or f"LLM subtitle correction task {task_id}"
    artifact_dir = Path(task_artifact_dir).expanduser().resolve() if task_artifact_dir else out.with_suffix(".llm-artifacts")
    translated_path = out.with_name(f"{out.stem}-translated.srt")

    def _maybe_knowledge_update(result_srt: Path) -> None:
        if knowledge != "update":
            return
        run_post_correction_knowledge_update(
            task_id=task_id,
            task_summary=task_summary,
            result_srt_path=result_srt,
            output_path=out,
            stable_json=stable_json,
            artifact_dir=artifact_dir,
            refined_srt=refined_srt,
            knowledge_root=knowledge_root,
            test_profile=test_profile,
        )

    if postprocess_profile is None and translated_path.exists():
        _maybe_knowledge_update(translated_path)
        return translated_path
    if postprocess_profile is not None and out.exists():
        _maybe_knowledge_update(out)
        return out
    if postprocess_profile is not None and translated_path.exists():
        report = postprocess_srt_file(translated_path, output_path=out, profile=postprocess_profile)
        append_task_artifact(
            artifact_dir,
            kind="final_srt",
            task_id=task_id,
            payload={
                "path": str(out),
                "translated_path": str(translated_path),
                "postprocess": report.to_dict(),
            },
        )
        write_task_report(
            artifact_dir,
            task_id=task_id,
            outputs={
                "translated_srt": str(translated_path),
                "final_srt": str(out),
                "task_artifact_dir": str(artifact_dir),
            },
        )
        _maybe_knowledge_update(out)
        return out
    token_counter = default_token_counter()
    fast_decision = decide_fast_mode(
        stable_json=stable_json,
        fast=fast,
        profile=profile,
        audio_path=audio_path,
        extra_info=extra_info,
        knowledge_root=knowledge_root,
        token_counter=token_counter,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
    )
    append_task_artifact(
        artifact_dir,
        kind="fast_decision",
        task_id=task_id,
        payload=fast_decision.to_metadata(),
    )
    context_pack: ContextPack | None = None
    fast_kwargs: Dict[str, Any] = {}
    context_path = ensure_research_context_path(
        artifact_dir=artifact_dir,
        stem=out.stem,
        run_dir=out.parent,
    )
    if fast_decision.enabled:
        fast_ctx = None
        fast_file_ref = None
        if profile.external_injection:
            fast_ctx, fast_file_ref, _ = acquire_fast_context(
                context_path=context_path,
                window=fast_decision.window,
                segment_count=len(fast_decision.window.segments),
                audio_path=audio_path,
                video_path=video_path,
                stable_json_stem=Path(stable_json).stem,
                extra_info=extra_info,
                knowledge_root=knowledge_root,
                enable_web_search=enable_web_search,
                search_rounds=fast_search_rounds,
                test_profile=test_profile,
                task_artifact_dir=artifact_dir,
                task_id=task_id,
                token_counter=token_counter,
                profile=profile,
                collect_task_feedback=collect_feedback,
                resume=resume,
            )
        fast_kwargs = _fast_execute_kwargs(fast_decision, fast_ctx, fast_file_ref, profile)
    elif profile.external_injection:
        context_pack = None
        if context_path.exists():
            # Reuse only a context planned under the same prompt version /
            # reserve / profile — otherwise its window ids no longer match the
            # correction plan and window_contexts would silently misalign.
            saved = json.loads(context_path.read_text(encoding="utf-8"))
            saved_planning = saved.get("planning") or {}
            current_planning = planning_metadata(
                profile,
                stable_json=stable_json,
                extra_info=extra_info,
                knowledge_root=knowledge_root,
                enable_web_search=enable_web_search,
                search_rounds=research_search_rounds,
                collect_task_feedback=collect_feedback,
                audio_duration=(
                    probe_audio_duration(audio_path) if audio_path else None
                ),
                max_window_subtitle_tokens=max_window_subtitle_tokens,
            )
            if saved_planning == current_planning:
                context_pack = load_research_context(context_path)
            else:
                print(
                    f"Warning: {context_path} was planned under different "
                    f"parameters ({saved_planning or 'no planning metadata'}); "
                    "re-running research.",
                    file=sys.stderr,
                )
        if context_pack is None:
            context_pack = run_research_stage(
                stable_json=stable_json,
                context_path=context_path,
                audio_path=audio_path,
                extra_info=extra_info,
                knowledge_root=knowledge_root,
                enable_web_search=enable_web_search,
                search_rounds=research_search_rounds,
                test_profile=test_profile,
                task_artifact_dir=artifact_dir,
                task_id=task_id,
                token_counter=token_counter,
                profile=profile,
                collect_task_feedback=collect_feedback,
                resume=resume,
                max_window_subtitle_tokens=max_window_subtitle_tokens,
            )
        # v17: research round 2's <keep_entries> seeds the first window's
        # transfer chain (persisted in the context JSON, so reuse keeps it).
        _seed_transfer_from_context(context_path, fast_kwargs)

    # Text route runs no research/round-1: note-keyword matches seed the FIRST
    # window's transfer chain (v17); each window's <keep_entries> then decides
    # what carries forward instead of a permanent global injection.
    if not profile.external_injection and extra_info.strip():
        seed_entries, seed_matches = load_preinjected_entries(
            knowledge_root, extra_info
        )
        if seed_matches:
            append_task_artifact(
                artifact_dir,
                kind="knowledge_preinjection",
                task_id=task_id,
                payload={
                    "source": "text_route_correction",
                    "matches": [match.to_dict() for match in seed_matches],
                    "seed_keys": list(seed_entries),
                },
            )
        if seed_entries:
            fast_kwargs["initial_transfer_keys"] = list(seed_entries)

    result = execute_correction_windows(
        stable_json=stable_json,
        output_path=out,
        context_pack=context_pack,
        audio_label=str(audio_path) if audio_path else "",
        audio_path=audio_path,
        video_path=video_path,
        test_profile=test_profile,
        max_retries_per_window=max_retries_per_window,
        enable_web_search=enable_web_search,
        postprocess_profile=postprocess_profile,
        extra_style=extra_style,
        common_mistakes_block=render_featured_mistakes_block(knowledge_root),
        task_artifact_dir=artifact_dir,
        task_id=task_id,
        task_update_feedback=collect_feedback,
        token_counter=token_counter,
        resume=resume,
        profile=profile,
        knowledge_root=knowledge_root,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        **fast_kwargs,
    )
    _maybe_knowledge_update(result)
    return result


def run_full_correction(*args: Any, **kwargs: Any) -> Path:
    """Timed public wrapper around the correction harness."""

    output_path = kwargs.get("output_path")
    if output_path is None:
        return _run_full_correction_impl(*args, **kwargs)
    out = Path(output_path).expanduser().resolve()
    artifact_dir = (
        Path(kwargs["task_artifact_dir"]).expanduser().resolve()
        if kwargs.get("task_artifact_dir")
        else out.with_suffix(".llm-artifacts")
    )
    existed = out.exists() or out.with_name(f"{out.stem}-translated.srt").exists()
    started = time.perf_counter()
    status = (
        "reused"
        if existed and str(kwargs.get("knowledge") or "none") != "update"
        else "executed"
    )
    try:
        result = _run_full_correction_impl(*args, **kwargs)
        return result
    except BaseException:
        status = "failed"
        raise
    finally:
        metadata_path = metadata_path_for_output(out)
        update_run_metadata(
            metadata_path,
            {
                "task_id": str(kwargs.get("task_id") or Path(kwargs["stable_json"]).stem),
                "timing": {
                    "stages": {
                        "llm_harness": stage_record(
                            status=status,
                            elapsed_sec=(
                                None
                                if status == "reused"
                                else time.perf_counter() - started
                            ),
                        )
                    }
                },
                "llm_rounds": summarize_llm_rounds(artifact_dir),
            },
        )
        if status != "failed":
            write_task_report(
                artifact_dir,
                task_id=str(
                    kwargs.get("task_id") or Path(kwargs["stable_json"]).stem
                ),
                outputs={
                    "translated_srt": str(
                        out.with_name(f"{out.stem}-translated.srt")
                    ),
                    **({"final_srt": str(out)} if out.exists() else {}),
                },
                run_metadata_path=metadata_path,
            )


def _seed_transfer_from_context(context_path: Path, fast_kwargs: dict) -> None:
    """Read research round 2's persisted <keep_entries> into the transfer seed."""

    try:
        saved_payload = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    transfer_seed = [
        key
        for key in (saved_payload.get("keep_entries") or [])
        if isinstance(key, str) and key
    ]
    if transfer_seed:
        fast_kwargs["initial_transfer_keys"] = transfer_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run LLM-based ASR correction and Chinese subtitle translation."
    )
    parser.add_argument(
        "input",
        help="Path to *-stable.json from VAD-ASR; correction input is rendered from it.",
    )
    parser.add_argument("-o", "--output", help="Path to corrected translated SRT.")
    parser.add_argument(
        "--route",
        choices=list(ROUTES),
        default=DEFAULT_PROFILE.route,
        help=(
            "Translation route: 'mm' uses harness-side research/search injection "
            "(audio at med/high); 'text' is text-only with no harness retrieval."
        ),
    )
    parser.add_argument(
        "--level",
        choices=list(LEVELS),
        default=DEFAULT_PROFILE.level,
        help=(
            "Route level. text: low=cheap fast, med=deeper thinking, high=+model "
            "native web search (internet_capable role). mm: low=text-only with "
            "injection, med=+audio (current default), high=+audio+video."
        ),
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=1.0,
        help=(
            "Scale k on the expected-output estimate k x c x csv_tokens; larger "
            "values plan smaller windows."
        ),
    )
    parser.add_argument(
        "--fast",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Fast mode: one fused window (research round 1 merged into the "
            "correction session). auto enables it when the whole input fits "
            "0.8 x output_limit - 10k output and the round-1 input leaves a 20k "
            "reserve; on errors out when it does not fit; off forces the normal flow."
        ),
    )
    parser.add_argument(
        "--fast-search-rounds",
        type=int,
        default=DEFAULT_FAST_SEARCH_ROUNDS,
        help=(
            "Total search rounds in fast mode (round 0 included, default 2); "
            "--research-search-rounds does not apply to fast runs."
        ),
    )
    parser.add_argument(
        "--audio",
        help=(
            "Path to original audio, not *-vocal.ogg. Required with --execute "
            "on audio profiles (mm med/high): each window uploads its own "
            "mono-16k AAC clip (segment span plus padding) instead of the whole file."
        ),
    )
    parser.add_argument(
        "--video",
        help=(
            "Path to the source video (mm-high only; required with --execute). "
            "The correction round gets a low-res video+audio mp4 clip per "
            "window instead of the .aac; the query round stays audio-only."
        ),
    )
    parser.add_argument(
        "--extra-info",
        default="",
        help="Extra user-provided info for research: source URL, content notes, requirements.",
    )
    parser.add_argument(
        "--extra-info-file",
        help="Path to a file with extra user-provided info for research.",
    )
    parser.add_argument(
        "--context-file",
        help="Existing research-context.json; skips the research rounds.",
    )
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Run only the two research rounds, write research-context.json, then exit.",
    )
    parser.add_argument(
        "--extra-style",
        default="",
        help="Extra translation style prompt injected into the correction system prompt.",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help=(
            "Disable the local web search agent (Tavily/DuckDuckGo): research round 2 "
            "gets no search results and correction query rounds are skipped."
        ),
    )
    parser.add_argument(
        "--research-search-rounds",
        type=int,
        default=DEFAULT_RESEARCH_SEARCH_ROUNDS,
        help=(
            "Total background-research search rounds (round 0 included). Values >1 "
            "enable the multi-round search loop (Research Contract / Evidence Pack); "
            "1 restores the legacy single-round search."
        ),
    )
    parser.add_argument(
        "--postprocess-profile",
        type=int,
        choices=SUPPORTED_POSTPROCESS_PROFILES,
        default=DEFAULT_POSTPROCESS_PROFILE,
        help=(
            "Final SRT postprocess profile: -1 semantic no-op re-render; "
            "0 t2s, overlap, duration, punctuation; 1 duration only; "
            "2 punctuation only; 3 t2s only; 4 overlap repair only."
        ),
    )
    parser.add_argument(
        "--task-summary",
        default="",
        help="Task summary included in knowledge update prompts.",
    )
    parser.add_argument(
        "--task-id",
        default="",
        help="Stable task id used in retained artifacts and knowledge commits.",
    )
    parser.add_argument(
        "--task-artifact-dir",
        help="Retain selected LLM outputs and task metadata as JSONL in this directory.",
    )
    parser.add_argument(
        "--knowledge",
        choices=list(KNOWLEDGE_MODES),
        default="none",
        help=(
            "Knowledge switch: 'collect' makes correction/research emit "
            "task_update_feedback; 'update' additionally runs the unified "
            "knowledge update after the task (requires --execute)."
        ),
    )
    parser.add_argument(
        "--refined-srt",
        help=(
            "User-refined SRT for the knowledge update (refined_aligned mode); "
            "only with --knowledge update."
        ),
    )
    parser.add_argument(
        "--knowledge-root",
        default=(
            str(DEFAULT_KNOWLEDGE_ROOT)
            if DEFAULT_KNOWLEDGE_ROOT is not None
            else None
        ),
        help="Root directory of the local Markdown knowledge base (embedded git repo).",
    )
    parser.add_argument(
        "--prompt-dir",
        help="Write plan.json plus Chinese research/correction prompts to this directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured LLM APIs. Default only writes/prints the plan.",
    )
    parser.add_argument(
        "--test-profile",
        action="store_true",
        help="Use gemini-3.5-flash-lite for every role.",
    )
    parser.add_argument(
        "--max-retries-per-window",
        type=int,
        default=5,
        help="Maximum correction retry attempts for each window.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "Disable LLM session and correction-window checkpoint reads/writes "
            "(default: resume validated sessions and completed windows in the "
            "task artifact directory)."
        ),
    )
    return parser.parse_args()


def _load_extra_info(args: argparse.Namespace) -> str:
    parts = [args.extra_info.strip()] if args.extra_info.strip() else []
    if args.extra_info_file:
        parts.append(
            Path(args.extra_info_file).expanduser().read_text(encoding="utf-8").strip()
        )
    return "\n".join(part for part in parts if part)


def _default_research_context_path(args: argparse.Namespace) -> Path:
    artifact_dir = _default_task_artifact_dir(args)
    if args.output:
        stem = Path(args.output).expanduser().resolve().stem
        run_dir = Path(args.output).expanduser().resolve().parent
    else:
        source = Path(args.input).expanduser().resolve()
        stem = source.stem
        run_dir = source.parent
    return ensure_research_context_path(
        artifact_dir=artifact_dir,
        stem=stem,
        run_dir=run_dir,
    )

def _default_task_artifact_dir(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve().with_suffix(".llm-artifacts")
    source = Path(args.input).expanduser().resolve()
    return source.with_name(f"{source.stem}.llm-artifacts")


def main() -> int:
    args = parse_args()
    task_id = args.task_id or Path(args.input).stem
    try:
        profile = resolve_profile(args.route, args.level, output_scale=args.output_scale)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.knowledge != "none" and not args.execute:
        print("--knowledge collect/update requires --execute", file=sys.stderr)
        return 2
    if args.refined_srt and args.knowledge != "update":
        print("--refined-srt requires --knowledge update", file=sys.stderr)
        return 2
    if args.research_only and args.context_file:
        print("--research-only conflicts with --context-file", file=sys.stderr)
        return 2
    if not profile.external_injection and (args.research_only or args.context_file):
        print(
            "--research-only/--context-file require the mm route (the text "
            "route has no research stage)",
            file=sys.stderr,
        )
        return 2
    if args.execute and profile.use_audio and not args.audio:
        print("--audio is required with --execute on audio profiles", file=sys.stderr)
        return 2
    if args.video and not profile.use_video:
        print("--video only applies to --route mm --level high", file=sys.stderr)
        return 2
    if args.execute and profile.use_video and not args.video:
        print("--video is required with --execute on mm-high", file=sys.stderr)
        return 2
    audio_label = args.audio or ""
    extra_info = _load_extra_info(args)
    try:
        max_window_subtitle_tokens = resolve_chunking_subtitle_cap()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    task_artifact_dir = (
        Path(args.task_artifact_dir).expanduser().resolve()
        if args.task_artifact_dir
        else (_default_task_artifact_dir(args) if (args.execute or args.research_only) else None)
    )
    # Shared across planning/research/execution: the sha cache makes repeated
    # countTokens calls over identical window texts free.
    token_counter = default_token_counter()
    try:
        fast_decision = decide_fast_mode(
            stable_json=args.input,
            fast=args.fast,
            profile=profile,
            audio_path=args.audio,
            extra_info=extra_info,
            knowledge_root=args.knowledge_root,
            token_counter=token_counter,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    artifacts = build_prompt_artifacts(
        stable_json=args.input,
        audio_path=args.audio,
        video_path=args.video,
        audio_label=audio_label,
        extra_info=extra_info,
        knowledge_root=args.knowledge_root,
        task_update_feedback=knowledge_collects(args.knowledge),
        research_search_rounds=(
            1 if args.no_web_search else args.research_search_rounds
        ),
        counter=token_counter,
        profile=profile,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
    )
    artifacts["fast_decision"] = fast_decision.to_metadata()
    if fast_decision.enabled and profile.external_injection:
        artifacts["fast_round1_messages"] = build_fast_round1_messages(
            window=fast_decision.window,
            extra_info=extra_info,
            streamer_index="（运行时注入 streamer index）",
            common_index="（运行时注入 common index）",
            max_search_queries=research_search_query_limit(
                len(fast_decision.window.segments)
            ),
            use_search_contract=(not args.no_web_search and args.fast_search_rounds > 1),
            profile=profile,
        )
    if args.prompt_dir:
        write_prompt_artifacts(artifacts, args.prompt_dir)
        print(f"Wrote prompt artifacts: {Path(args.prompt_dir).expanduser().resolve()}")
    else:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in artifacts.items()
                    if key
                    not in {
                        "research_messages",
                        "search_loop_example_messages",
                        "correction_query_messages",
                        "correction_messages",
                        "fast_round1_messages",
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if not args.execute and not args.research_only:
        return 0

    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="fast_decision",
            task_id=task_id,
            payload=fast_decision.to_metadata(),
        )
    context_pack: ContextPack | None = None
    fast_kwargs: Dict[str, Any] = {}
    if fast_decision.enabled and profile.external_injection:
        context_path = _default_research_context_path(args)
        fast_ctx, fast_file_ref, reused = acquire_fast_context(
            context_path=context_path,
            context_file=args.context_file,
            window=fast_decision.window,
            segment_count=len(fast_decision.window.segments),
            audio_path=args.audio,
            video_path=args.video,
            stable_json_stem=Path(args.input).stem,
            extra_info=extra_info,
            knowledge_root=args.knowledge_root,
            enable_web_search=not args.no_web_search,
            search_rounds=args.fast_search_rounds,
            test_profile=args.test_profile,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            token_counter=token_counter,
            profile=profile,
            collect_task_feedback=knowledge_collects(args.knowledge),
            resume=args.resume,
        )
        if not reused:
            print(f"Wrote fast research context: {context_path}")
        if args.research_only:
            if task_artifact_dir:
                write_task_report(
                    task_artifact_dir,
                    task_id=task_id,
                    outputs={"research_context": str(context_path)},
                )
                print(f"Task report: {Path(task_artifact_dir) / 'task-report.md'}")
            return 0
        fast_kwargs = _fast_execute_kwargs(fast_decision, fast_ctx, fast_file_ref, profile)
    elif fast_decision.enabled:
        fast_kwargs = _fast_execute_kwargs(fast_decision, None, None, profile)
    elif not profile.external_injection:
        pass  # text route: no research stage, no injected context
    elif args.context_file:
        context_pack = load_research_context(args.context_file)
        _seed_transfer_from_context(Path(args.context_file), fast_kwargs)
    else:
        context_path = _default_research_context_path(args)
        context_pack = run_research_stage(
            stable_json=args.input,
            context_path=context_path,
            audio_path=args.audio,
            extra_info=extra_info,
            knowledge_root=args.knowledge_root,
            enable_web_search=not args.no_web_search,
            search_rounds=args.research_search_rounds,
            test_profile=args.test_profile,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            token_counter=token_counter,
            profile=profile,
            collect_task_feedback=knowledge_collects(args.knowledge),
            resume=args.resume,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
        )
        print(f"Wrote research context: {context_path}")
        _seed_transfer_from_context(context_path, fast_kwargs)
        if args.research_only:
            if task_artifact_dir:
                write_task_report(
                    task_artifact_dir,
                    task_id=task_id,
                    outputs={"research_context": str(context_path)},
                )
                print(f"Task report: {Path(task_artifact_dir) / 'task-report.md'}")
            return 0

    if not args.output:
        print("--output is required with --execute", file=sys.stderr)
        return 2

    # Text route: note-keyword matches seed the first window's transfer chain
    # (mirrors run_full_correction; there is no research round to carry them).
    if not profile.external_injection and extra_info.strip():
        seed_entries, seed_matches = load_preinjected_entries(
            args.knowledge_root, extra_info
        )
        if task_artifact_dir and seed_matches:
            append_task_artifact(
                task_artifact_dir,
                kind="knowledge_preinjection",
                task_id=task_id,
                payload={
                    "source": "text_route_correction",
                    "matches": [match.to_dict() for match in seed_matches],
                    "seed_keys": list(seed_entries),
                },
            )
        if seed_entries:
            fast_kwargs["initial_transfer_keys"] = list(seed_entries)

    out = execute_correction_windows(
        stable_json=args.input,
        output_path=args.output,
        context_pack=context_pack,
        audio_label=audio_label,
        audio_path=args.audio,
        video_path=args.video,
        test_profile=args.test_profile,
        max_retries_per_window=args.max_retries_per_window,
        enable_web_search=not args.no_web_search,
        postprocess_profile=args.postprocess_profile,
        extra_style=args.extra_style,
        common_mistakes_block=render_featured_mistakes_block(args.knowledge_root),
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
        task_update_feedback=knowledge_collects(args.knowledge),
        token_counter=token_counter,
        resume=args.resume,
        profile=profile,
        knowledge_root=args.knowledge_root,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        **fast_kwargs,
    )
    print(f"Wrote {out}")
    if task_artifact_dir:
        report_path = Path(task_artifact_dir) / "task-report.md"
        if report_path.exists():
            print(f"Task report: {report_path}")
    if args.knowledge == "update":
        update_report = run_post_correction_knowledge_update(
            task_id=task_id,
            task_summary=args.task_summary or f"LLM subtitle correction task {task_id}",
            result_srt_path=out,
            output_path=args.output,
            stable_json=args.input,
            artifact_dir=task_artifact_dir or _default_task_artifact_dir(args),
            refined_srt=args.refined_srt,
            knowledge_root=args.knowledge_root,
            test_profile=args.test_profile,
        )
        print(
            f"Knowledge update ({update_report['mode']}): "
            f"{len(update_report['chunks'])} chunk(s); ledger: {update_report['ledger_path']}"
        )
        report_path = Path(task_artifact_dir or _default_task_artifact_dir(args)) / "task-report.md"
        if report_path.exists():
            print(f"Task report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
