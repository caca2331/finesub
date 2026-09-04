"""CLI and orchestration for LLM subtitle correction translation.

The window-execution loop lives in ``finesub.llm.stages.correction``; research
acquisition in ``finesub.llm.research``; the unified knowledge update in
``finesub.llm.knowledge.update``. This module keeps the CLI, dry-run prompt artifacts
and the top-level orchestration.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any, ContextManager, Dict, Mapping, Sequence

from finesub.run_metadata import (
    metadata_path_for_output,
    stage_record,
    summarize_llm_rounds,
    update_run_metadata,
)

from finesub.media.clips import probe_audio_duration
from finesub.reporting import current_reporter, reporting_to, terminal_reporter
from finesub_bootstrap.artifacts import ARTIFACT_DIR_SUFFIX
from .agent.agent_session_host import agent_session_scope, set_run_evidence_destination
from .client import write_agent_session_usage
from .knowledge.style import render_style_block, resolve_style_selection
from .routing.api_keys import read_config
from .routing.config import (
    DEFAULT_RESEARCH_SEARCH_ROUNDS,
    research_search_query_limit,
)
from .knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    pinned_generation_rev,
)
from .knowledge.update import (
    ensure_research_context_path,
    run_knowledge_update,
)
from .routing.profiles import (
    CONTINUITY,
    DEFAULT_FAST_SEARCH_ROUNDS,
    DEFAULT_PROFILE,
    DIFFICULTY,
    MEDIA,
    RETRIEVAL,
    TranslationProfile,
    resolve_profile,
)
from .routing.capabilities import (
    CapabilityUnavailableError,
    correction_planning_limits,
    validate_profile_capabilities,
)
from .prompts import (
    ContextPack,
    build_fast_round1_messages,
)
from .prompt_artifacts import build_prompt_artifacts, write_prompt_artifacts
from .research import (
    backup_unrecoverable_json,
    load_research_context,
    planning_metadata,
    research_reuse_key,
    research_stage_runs,
    run_research_stage,
)
from finesub.subtitles.postprocess import (
    DEFAULT_POSTPROCESS_PROFILE,
    SUPPORTED_POSTPROCESS_PROFILES,
    postprocess_srt_file,
)
from .stages.correction import execute_correction_windows
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


def _load_reusable_research_context(
    context_path: Path,
    *,
    expected_planning: Mapping[str, Any],
) -> ContextPack | None:
    """Load a committed research context when it is still the right one (L3).

    The comparison is :func:`research_reuse_key`, not the whole planning dict:
    model, preset, policy, difficulty, continuity and knowledge content all
    changed how this context was produced without making it wrong. Window
    geometry does not gate it either -- the notes carry source-id intervals and
    remap onto any later plan. What still gates it is the source, the retrieval
    mode and the knobs a user turns *at this artifact* -- notes and search
    rounds -- where silently reusing the old result would ignore the request.

    Corrupt or incompatible files are not a recoverable checkpoint; they are
    copied aside before the caller overwrites them. A pack whose notes carry no
    source ranges counts as incompatible: it can only be injected half-empty.
    """

    try:
        saved = json.loads(context_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("research context root is not an object")
        saved_planning = saved.get("planning")
        if not isinstance(saved_planning, dict):
            raise ValueError("missing planning metadata")
        saved_key = research_reuse_key(saved_planning)
        expected_key = research_reuse_key(expected_planning)
        if saved_key != expected_key:
            differing = sorted(
                key
                for key in set(saved_key) | set(expected_key)
                if saved_key.get(key) != expected_key.get(key)
            )
            current_reporter().warning(
                "research-context-stale",
                f"{context_path} was planned under different parameters "
                f"({', '.join(differing)})",
                impact="重跑 research",
            )
            return None
        return load_research_context(context_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        backup = backup_unrecoverable_json(context_path)
        current_reporter().warning(
            "research-context-unrecoverable",
            f"{context_path} is not a recoverable research context ({exc})",
            impact=f"已备份到 {backup}，重跑 research",
        )
        return None


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
    difficulty: str = "quality",
    style_names: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run the unified knowledge update right after a correction task.

    ``difficulty`` is the run's own: it selects the knowledge cell, so a preset
    binding a cheaper group at intermediate has to reach this call rather than
    being resolved as the top tier.

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
        difficulty=difficulty,
        style_names=style_names,
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
    max_replacements_per_window: int = 1,
    parallel_windows: int = 1,
    research_search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    postprocess_profile: int | None = DEFAULT_POSTPROCESS_PROFILE,
    extra_style: str = "",
    #: Named style entries (`--style`). `None` falls through to the config,
    #: `""` is a decision to run without one — `resolve_style_names` owns that
    #: precedence so both front ends get the same answer.
    style: str | Sequence[str] | None = None,
    #: What this run may do with it: `none` / `read` / `update`. Unset reads
    #: `[llm] style_mode`, then defaults to `read` — injecting is what a style
    #: is for; writing back is asked for.
    style_mode: str | None = None,
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
    finesub.workflows.reference_ingest. An existing
    *-research-context.json under the task
    artifact directory (or a legacy sibling next to the output SRT) is reused
    so reruns skip the research rounds.
    """
    if knowledge not in KNOWLEDGE_MODES:
        raise ValueError(f"unknown knowledge mode {knowledge!r}; use none|collect|update")
    # Resolved once, up here: the early-return paths (an existing translated
    # output that still owes a knowledge update) reach the post-task update
    # without passing the correction call.
    style_selection = resolve_style_selection(
        style, style_mode, knowledge_root=knowledge_root, difficulty=profile.difficulty
    )
    style_names = style_selection.names
    if style_selection.writable:
        # `update` rides on the post-task knowledge update, and that task may
        # only touch a style with refined subtitles in front of it. Asking for
        # the write without either is a run that quietly behaves like `read`.
        missing = []
        if knowledge != "update":
            missing.append("--knowledge update")
        if not refined_srt:
            missing.append("--refined-srt")
        if missing:
            current_reporter().warning(
                "style-update-inert",
                f"--style-mode update 需要 {' 与 '.join(missing)}，本次缺少它们",
                impact="这套风格照常注入提示词，但本次学到的东西不会写回去",
                action="补上缺的参数，或改成 --style-mode read 让意图和行为一致",
            )
    if profile.difficulty == "efficiency" and knowledge != "none":
        # efficiency disables knowledge outright (owner decision 2026-08-12; it
        # used to cap at collect). The cheapest shape reads no indices, injects
        # no entries and runs no post-task update -- and its thinking knob is
        # low everywhere, which is not a tier that should be feeding the
        # knowledge base either.
        raise ValueError(
            "difficulty=efficiency disables knowledge (no injection, no "
            "collection, no post-task update); use --knowledge none, or "
            "--difficulty intermediate for a cheap run that still reads the base."
        )
    collect_feedback = knowledge_collects(knowledge)
    if max_window_subtitle_tokens is None:
        max_window_subtitle_tokens = resolve_chunking_subtitle_cap()
    out = Path(output_path).expanduser().resolve()
    task_id = task_id or Path(stable_json).stem
    task_summary = task_summary or f"LLM subtitle correction task {task_id}"
    artifact_dir = (
        Path(task_artifact_dir).expanduser().resolve()
        if task_artifact_dir
        else out.with_suffix(ARTIFACT_DIR_SUFFIX)
    )
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
            difficulty=profile.difficulty,
            style_names=style_selection.writable,
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
        knowledge_enabled=collect_feedback,
        token_counter=token_counter,
        # Group planning envelope (plan v2 D13): the fused window is budgeted
        # against the correction cell's group minimum too.
        limits=correction_planning_limits(profile),
        max_window_subtitle_tokens=max_window_subtitle_tokens,
    )
    # Planning determines whether retrieval=local uses the ordinary per-window
    # query chain or the fused fast round-1 chain. Validate only the shape this
    # run will execute; checking both can reject a healthy --fast off run for a
    # model pool it never calls.
    validate_profile_capabilities(
        profile,
        fast_enabled=fast_decision.enabled,
        test_profile=test_profile,
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
    if fast_decision.enabled and profile.external_injection:
        fast_ctx, fast_file_ref, _ = acquire_fast_context(
            context_path=context_path,
            stable_json=stable_json,
            window=fast_decision.window,
            segment_count=len(fast_decision.window.segments),
            audio_path=audio_path,
            video_path=video_path,
            stable_json_stem=Path(stable_json).stem,
            extra_info=extra_info,
            knowledge_root=knowledge_root,
            knowledge_enabled=collect_feedback,
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
    elif research_stage_runs(
        profile, knowledge_root=knowledge_root, knowledge_enabled=collect_feedback
    ):
        # Runs for fast vectors too when they are not retrieval=local: their
        # fused round has no query/entry step, so without this stage a fast
        # none/native run would never read the knowledge base at all (r1 is
        # the session's only entry pick on those vectors; see
        # docs/llm_harness_behavior.md). Under
        # ``none`` this is r1 only; under ``native`` r2 also runs and its
        # general context reaches the fused window (window-specific contexts
        # are keyed by the multi-window plan and simply miss).
        if fast_decision.enabled:
            fast_kwargs = _fast_execute_kwargs(fast_decision, None, None, profile)
        context_pack = None
        if context_path.exists():
            context_pack = _load_reusable_research_context(
                context_path,
                expected_planning=planning_metadata(
                    profile,
                    stable_json=stable_json,
                    extra_info=extra_info,
                    knowledge_root=knowledge_root,
                    knowledge_enabled=collect_feedback,
                    search_rounds=research_search_rounds,
                    collect_task_feedback=collect_feedback,
                    audio_duration=(
                        probe_audio_duration(audio_path) if audio_path else None
                    ),
                    max_window_subtitle_tokens=max_window_subtitle_tokens,
                    limits=correction_planning_limits(profile),
                    test_profile=test_profile,
                ),
            )
        if context_pack is None:
            context_pack = run_research_stage(
                stable_json=stable_json,
                context_path=context_path,
                audio_path=audio_path,
                extra_info=extra_info,
                knowledge_root=knowledge_root,
                knowledge_enabled=collect_feedback,
                search_rounds=research_search_rounds,
                test_profile=test_profile,
                task_artifact_dir=artifact_dir,
                task_id=task_id,
                token_counter=token_counter,
                profile=profile,
                collect_task_feedback=collect_feedback,
                resume=resume,
                max_window_subtitle_tokens=max_window_subtitle_tokens,
                parallel_windows=parallel_windows,
            )
        # v17: the research stage's <keep_entries> seeds the first window's
        # transfer chain (persisted in the context JSON, so reuse keeps it).
        # This is also what replaced the text route's note-keyword seeding --
        # R1 now picks entries for every vector with a readable index, by
        # judgment over the whole transcript rather than by keyword match.
        _seed_transfer_from_context(context_path, fast_kwargs)
    elif fast_decision.enabled:
        # Fast on a vector with nothing to research (no retrieval, no index).
        fast_kwargs = _fast_execute_kwargs(fast_decision, None, None, profile)

    result = execute_correction_windows(
        stable_json=stable_json,
        output_path=out,
        context_pack=context_pack,
        audio_label=str(audio_path) if audio_path else "",
        audio_path=audio_path,
        video_path=video_path,
        test_profile=test_profile,
        max_retries_per_window=max_retries_per_window,
        max_replacements_per_window=max_replacements_per_window,
        parallel_window_limit=parallel_windows,
        postprocess_profile=postprocess_profile,
        extra_style=extra_style,
        style_block=render_style_block(knowledge_root, style_names),
        knowledge_enabled=collect_feedback,
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


def _generation_pin(kwargs: Mapping[str, Any]) -> ContextManager[Any]:
    """Pin every knowledge read of this run to one revision (plan §2.5).

    research, the windows and the query rounds all read through the pin; the
    post-run knowledge update passes its own explicit ``working_rev`` per
    chunk, which overrides it. ``--knowledge none`` never touches the base, so
    it never opens (or creates) the store either.
    """

    if str(kwargs.get("knowledge") or "none") == "none":
        return nullcontext()
    root = kwargs.get("knowledge_root", DEFAULT_KNOWLEDGE_ROOT)
    # Absorb pending rendered/ user edits BEFORE pinning (plan §11.3 / O5):
    # the run then reads the freshly harvested revision. Guarded (worktree
    # gate + write lock) and per-file isolated inside; never blocks the run.
    try:
        from .knowledge.node.edit import harvest_rendered_edits_at_run_start

        harvest_rendered_edits_at_run_start(root)
    except Exception as exc:
        current_reporter().warning("knowledge-rendered-harvest-failed", str(exc))
    return pinned_generation_rev(root)


def _task_slots(kwargs: Mapping[str, Any]) -> ContextManager[Any]:
    """This run's agent slot account (task-parallelism plan W4).

    A task whose routing can reach an agent target reserves its mandatory
    lane for the whole run; a pure-API or test-profile run books nothing.
    """

    from .run_context import default_task_slots

    return default_task_slots(test_profile=bool(kwargs.get("test_profile")))


def run_full_correction(*args: Any, **kwargs: Any) -> Path:
    """Timed public wrapper around the correction harness."""

    output_path = kwargs.get("output_path")
    if output_path is None:
        with _task_slots(kwargs), agent_session_scope(), _generation_pin(kwargs):
            return _run_full_correction_impl(*args, **kwargs)
    out = Path(output_path).expanduser().resolve()
    artifact_dir = (
        Path(kwargs["task_artifact_dir"]).expanduser().resolve()
        if kwargs.get("task_artifact_dir")
        else out.with_suffix(ARTIFACT_DIR_SUFFIX)
    )
    existed = out.exists() or out.with_name(f"{out.stem}-translated.srt").exists()
    started = time.perf_counter()
    status = (
        "reused"
        if existed and str(kwargs.get("knowledge") or "none") != "update"
        else "executed"
    )
    # One scope for the whole run: research, the windows and the knowledge
    # update each build their own `RoleClient`, and a run-long agent session
    # is shared by all of them (docs/llm_local_agent.md §12.1.3). It closes
    # before the report below, which is what lets the report see what those
    # sessions spent.
    sessions: Any = None
    try:
        with _task_slots(kwargs), agent_session_scope() as sessions, _generation_pin(kwargs):
            # Said once, while the scope is open: it files what the sessions
            # kept when it closes, which is after this block and before the
            # booking below.
            set_run_evidence_destination(artifact_dir)
            result = _run_full_correction_impl(*args, **kwargs)
        return result
    except BaseException:
        status = "failed"
        raise
    finally:
        if sessions is not None:
            write_agent_session_usage(artifact_dir, sessions)
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
    raw_keep_entries = saved_payload.get("keep_entries")
    if raw_keep_entries is None and isinstance(saved_payload.get("fast"), dict):
        raw_keep_entries = saved_payload["fast"].get("keep_entries")
    transfer_seed = [
        key
        for key in (raw_keep_entries or [])
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
        "--media",
        choices=list(MEDIA),
        default=DEFAULT_PROFILE.correction_media,
        help=(
            "Convenience knob: sets both --correction-media and "
            "--planning-media at once (text / +audio / +video, a ladder -- "
            "video implies audio)."
        ),
    )
    parser.add_argument(
        "--correction-media",
        choices=list(MEDIA),
        default="",
        help=(
            "What the correction window sees; overrides --media for that "
            "task only (e.g. text to put a text-only strong model on the "
            "correction window while the query round keeps the clip)."
        ),
    )
    parser.add_argument(
        "--planning-media",
        choices=list(MEDIA),
        default="",
        help=(
            "What the per-window query round sees; overrides --media for "
            "that task only. video sends the window's video clip to the "
            "query round (it no longer force-cuts an audio-only .aac)."
        ),
    )
    parser.add_argument(
        "--retrieval",
        choices=list(RETRIEVAL),
        default=DEFAULT_PROFILE.retrieval,
        help=(
            "none = no retrieval at all; local = the harness injection machinery "
            "(background research, per-window query round, local search agent); "
            "native = the model's own search tool, no harness search."
        ),
    )
    parser.add_argument(
        "--difficulty",
        choices=list(DIFFICULTY),
        default=DEFAULT_PROFILE.difficulty,
        help=(
            "Which prompt/thinking cell to use. quality = the task group's "
            "top cell; intermediate = the declared step down (correction "
            "binds the lite group there); efficiency = cheapest shape that "
            "still corrects and translates (pins both media switches to text, "
            "retrieval=none and knowledge=none)."
        ),
    )
    parser.add_argument(
        "--continuity",
        choices=list(CONTINUITY),
        default=DEFAULT_PROFILE.continuity,
        help=(
            "serial (default) keeps the chained inter-window context (advice "
            "ledger, entry transfer chain); parallel gives it up and "
            "dispatches correction windows concurrently for wall-clock "
            "(docs/llm_harness_behavior.md)."
        ),
    )
    parser.add_argument(
        "--parallel-windows",
        type=int,
        default=1,
        help=(
            "Concurrency cap for --continuity parallel (default 1 -- a "
            "preference, not a calibrated constant; see docs/manual/agent.md)."
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
            "when a media switch uses audio: each window uploads its own "
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
        "--style",
        default=None,
        help="翻译风格条目名（可逗号分隔；同名时写 style/<名字>）。不给则读配置 [llm] style",
    )
    parser.add_argument(
        "--style-mode",
        dest="style_mode",
        default=None,
        choices=("none", "read", "update"),
        help="拿这套风格做什么：none 不用 / read 只注入 / update 还把本次学到的写回去"
             "（需要 --refined-srt + --knowledge update）。不给则读配置 [llm] style_mode，默认 read",
    )
    parser.add_argument(
        "--extra-style",
        default="",
        help="Extra translation style prompt injected into the correction system prompt.",
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
        default=None,
        help=(
            "Knowledge switch. Default 'collect': read and inject the base "
            "(indices, entries, style) and emit "
            "task_update_feedback, without writing anything back (the default "
            "resolves to 'none' at --difficulty efficiency, which disables "
            "knowledge by construction). 'none': the knowledge base is not "
            "read, injected, collected from, or updated. 'update': everything "
            "collect does, plus the unified knowledge update after the task -- "
            "which a dry run never reaches, so it is accepted there and simply "
            "writes nothing."
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
        help=(
            "Tier 1 of the per-window retry budget: repair retries within "
            "one session chain -- each carries the previous output and the "
            "validation errors, and an agent backend resumes the same "
            "conversation. No longer the total-call cap: total calls per "
            "window = (this+1) x (--max-replacements-per-window+1)."
        ),
    )
    parser.add_argument(
        "--max-replacements-per-window",
        type=int,
        default=1,
        help=(
            "Tier 2 of the per-window retry budget: how many times a window "
            "whose repair chain is spent is handed to a fresh session -- "
            "repair context dropped, an agent starts a new conversation, a "
            "stateless endpoint throws blind."
        ),
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
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        action="append",
        metavar="[TASKGROUP=]VALUE",
        help=(
            "Runtime LLM routing override (repeatable; value = model group or "
            "route target, rebinding the task group's cell with NO fallback; "
            "bare value = default). Process-local — config.toml is never "
            "touched; routing identity changes, so resume caches invalidate "
            "by design."
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
        return Path(args.output).expanduser().resolve().with_suffix(ARTIFACT_DIR_SUFFIX)
    source = Path(args.input).expanduser().resolve()
    return source.with_name(f"{source.stem}{ARTIFACT_DIR_SUFFIX}")


def main() -> int:
    """The module CLI: binds the terminal renderer, then runs.

    The binding lives here rather than in the ``__main__`` guard so that any
    other caller of ``main()`` gets it too. Without it every
    ``current_reporter().warning(...)`` under this entry point is handed to the
    silent thread-local default and dropped -- measured 2026-09-03 on an
    unparseable hand edit in ``rendered/``: the run exited 0 and said nothing.
    The renderer writes to stderr, so redirecting stdout to capture a dry run's
    prompt JSON still shows the warnings that describe producing it.
    """

    with reporting_to(terminal_reporter()):
        return _main()


def _main() -> int:
    """The run itself: its own run boundary, like `run_full_correction`.

    This entry point does not go through `run_full_correction` -- it drives
    research, the windows and the knowledge update itself -- so the run's
    agent session scope *and* the knowledge generation pin have to be opened
    here too (docs/llm_local_agent.md §12.1.3, and the knowledge pin per
    docs/plans/knowledge-node-plan.md §2.5). The knowledge
    switches resolve before the pin so `--knowledge none` never opens the
    store; every knowledge read of the run then happens inside the pin.
    The report is written inside the run, so it is refreshed afterwards with
    what those sessions turned out to have spent.
    """

    args = parse_args()
    # Installed unconditionally (empty → clears) BEFORE any routing consumer:
    # planning envelopes, preflight, research and the correction drivers all
    # construct their clients off the memoized route loader.
    from .routing.model_routes import install_runtime_preferred, parse_llm_model_args

    install_runtime_preferred(parse_llm_model_args(args.llm_model))
    try:
        profile = resolve_profile(
            args.media,
            args.retrieval,
            args.difficulty,
            args.continuity,
            correction_media=args.correction_media,
            planning_media=args.planning_media,
            output_scale=args.output_scale,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # An unset switch is not a user statement: efficiency disables knowledge by
    # construction, so it resolves the default to none instead of refusing to
    # start. An *explicit* --knowledge on an efficiency run is still the hard
    # error it was, because that one really is a contradiction.
    if args.knowledge is None:
        args.knowledge = "none" if profile.difficulty == "efficiency" else "collect"
    elif profile.difficulty == "efficiency" and args.knowledge != "none":
        print(
            "--difficulty efficiency disables --knowledge entirely (no "
            "injection, no collection, no post-task update); drop the switch, "
            "or use --difficulty intermediate for a cheap run that still reads "
            "the base.",
            file=sys.stderr,
        )
        return 2
    # No --execute coupling: `--knowledge` says what happens *if* the LLM stage
    # runs, which is a different axis from whether this invocation calls the
    # API at all. A dry run takes `update` and simply never reaches the write --
    # it returns after the prompt artifacts, and the post-task update lives
    # inside `run_correction_task`, past that return. What `collect`/`update` do
    # change here is the dry run's actual product -- the dumped correction
    # prompts carry the feedback-collection contract -- which is the reason to
    # let the switch through rather than refuse the combination.
    if args.refined_srt and args.knowledge != "update":
        print("--refined-srt requires --knowledge update", file=sys.stderr)
        return 2
    booking: dict[str, Any] = {}
    sessions: Any = None
    try:
        with _task_slots(
            {"test_profile": args.test_profile}
        ), agent_session_scope() as sessions, _generation_pin(
            {"knowledge": args.knowledge, "knowledge_root": args.knowledge_root}
        ):
            return _main_impl(args, booking, profile)
    finally:
        _book_agent_session_usage(sessions, booking)


def _book_agent_session_usage(sessions: Any, booking: Mapping[str, Any]) -> None:
    """Rewrite the report with what this run's agent sessions actually spent.

    Unconditionally, once the book is settled: a run that spent nothing
    *clears* the previous one's book, and the report that was written during
    the run had already folded it in. Skipping the rewrite there left the
    Markdown quoting tokens whose JSON no longer exists.
    """

    artifact_dir = booking.get("artifact_dir")
    outputs = booking.get("outputs")
    if sessions is None or not artifact_dir:
        return
    write_agent_session_usage(artifact_dir, sessions)
    if outputs is None:
        # No report of ours to refresh (a dry run, or an early exit): the
        # outputs are only known where one was written.
        return
    write_task_report(
        artifact_dir,
        task_id=str(booking.get("task_id") or ""),
        outputs=dict(outputs),
    )


def _main_impl(args: argparse.Namespace, booking: dict[str, Any], profile: Any) -> int:
    task_id = args.task_id or Path(args.input).stem
    booking["task_id"] = task_id
    if args.research_only and args.context_file:
        print("--research-only conflicts with --context-file", file=sys.stderr)
        return 2
    if (args.research_only or args.context_file) and not research_stage_runs(
        profile,
        knowledge_root=args.knowledge_root,
        knowledge_enabled=knowledge_collects(args.knowledge),
    ):
        print(
            "--research-only/--context-file need a vector that runs a research "
            "stage: retrieval=local|native, or retrieval=none with a readable "
            "knowledge base (--knowledge collect|update)",
            file=sys.stderr,
        )
        return 2
    # A media switch above what the input files provide is a configuration
    # error, not a silent downgrade (plan v2 D20).
    if args.execute and profile.uses_media and not args.audio:
        print(
            "--audio is required with --execute when any media switch is not "
            "text",
            file=sys.stderr,
        )
        return 2
    if args.video and not profile.uses_video:
        print("--video only applies when a media switch is video", file=sys.stderr)
        return 2
    if args.execute and profile.uses_video and not args.video:
        print(
            "--video is required with --execute when a media switch is video",
            file=sys.stderr,
        )
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
    booking["artifact_dir"] = task_artifact_dir
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
            knowledge_enabled=knowledge_collects(args.knowledge),
            token_counter=token_counter,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.execute:
        # Dry runs deliberately work without configured provider keys. Full
        # execution validates the shape selected by the fast planner before
        # the first model call.
        try:
            validate_profile_capabilities(
                profile,
                fast_enabled=fast_decision.enabled,
                test_profile=args.test_profile,
            )
        except CapabilityUnavailableError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    # Resolved BEFORE the dry-run branch: the artifact must be the prompt the
    # run would really send. With style on by default, a resolution that
    # happened only on the execution path made the dry run miss a whole block
    # (review 2026-09-02).
    cli_style = resolve_style_selection(
        args.style, args.style_mode, knowledge_root=args.knowledge_root,
        difficulty=profile.difficulty,
    )
    artifacts = build_prompt_artifacts(
        style_names=cli_style.names,
        stable_json=args.input,
        audio_path=args.audio,
        video_path=args.video,
        audio_label=audio_label,
        extra_info=extra_info,
        knowledge_root=args.knowledge_root,
        knowledge_enabled=knowledge_collects(args.knowledge),
        task_update_feedback=knowledge_collects(args.knowledge),
        research_search_rounds=args.research_search_rounds,
        counter=token_counter,
        profile=profile,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
    )
    artifacts["fast_decision"] = fast_decision.to_metadata()
    if fast_decision.enabled and profile.external_injection:
        # The dry-run placeholder must match what execution will inject:
        # with knowledge off the round carries no index at all, so the plan
        # must not claim one will be injected at run time.
        knowledge_on = knowledge_collects(args.knowledge)
        artifacts["fast_round1_messages"] = build_fast_round1_messages(
            window=fast_decision.window,
            extra_info=extra_info,
            streamer_index="（运行时注入 streamer index）" if knowledge_on else "",
            common_index="（运行时注入 common index）" if knowledge_on else "",
            max_search_queries=research_search_query_limit(
                len(fast_decision.window.segments)
            ),
            use_search_contract=args.fast_search_rounds > 1,
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
            stable_json=args.input,
            window=fast_decision.window,
            segment_count=len(fast_decision.window.segments),
            audio_path=args.audio,
            video_path=args.video,
            stable_json_stem=Path(args.input).stem,
            extra_info=extra_info,
            knowledge_root=args.knowledge_root,
            knowledge_enabled=knowledge_collects(args.knowledge),
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
                booking["outputs"] = {"research_context": str(context_path)}
                write_task_report(
                    task_artifact_dir,
                    task_id=task_id,
                    outputs={"research_context": str(context_path)},
                )
                print(f"Task report: {Path(task_artifact_dir) / 'task-report.md'}")
            return 0
        fast_kwargs = _fast_execute_kwargs(fast_decision, fast_ctx, fast_file_ref, profile)
    elif not research_stage_runs(
        profile,
        knowledge_root=args.knowledge_root,
        knowledge_enabled=knowledge_collects(args.knowledge),
    ):
        # Nothing to research: no retrieval and no index to pick entries off.
        if fast_decision.enabled:
            fast_kwargs = _fast_execute_kwargs(fast_decision, None, None, profile)
    elif args.context_file:
        if fast_decision.enabled:
            fast_kwargs = _fast_execute_kwargs(fast_decision, None, None, profile)
        try:
            context_pack = load_research_context(args.context_file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            # An explicitly named file is a user instruction, not a checkpoint:
            # say why it cannot be used instead of quietly researching again.
            print(
                f"Error: {args.context_file} is not a usable research context "
                f"({exc}).",
                file=sys.stderr,
            )
            return 2
        _seed_transfer_from_context(Path(args.context_file), fast_kwargs)
    else:
        # Also reached by fast runs on non-local vectors: their fused round has
        # no query/entry step, so this stage is the session's only entry pick
        # (r1; plus r2's own search under retrieval=native); see
        # docs/llm_harness_behavior.md.
        if fast_decision.enabled:
            fast_kwargs = _fast_execute_kwargs(fast_decision, None, None, profile)
        context_path = _default_research_context_path(args)
        context_pack = (
            _load_reusable_research_context(
                context_path,
                expected_planning=planning_metadata(
                    profile,
                    stable_json=args.input,
                    extra_info=extra_info,
                    knowledge_root=args.knowledge_root,
                    knowledge_enabled=knowledge_collects(args.knowledge),
                    search_rounds=args.research_search_rounds,
                    collect_task_feedback=knowledge_collects(args.knowledge),
                    audio_duration=(
                        probe_audio_duration(args.audio) if args.audio else None
                    ),
                    max_window_subtitle_tokens=max_window_subtitle_tokens,
                    limits=correction_planning_limits(profile),
                    test_profile=args.test_profile,
                ),
            )
            if context_path.exists()
            else None
        )
        if context_pack is None:
            context_pack = run_research_stage(
                stable_json=args.input,
                context_path=context_path,
                audio_path=args.audio,
                extra_info=extra_info,
                knowledge_root=args.knowledge_root,
                knowledge_enabled=knowledge_collects(args.knowledge),
                search_rounds=args.research_search_rounds,
                test_profile=args.test_profile,
                task_artifact_dir=task_artifact_dir,
                task_id=task_id,
                token_counter=token_counter,
                profile=profile,
                collect_task_feedback=knowledge_collects(args.knowledge),
                resume=args.resume,
                max_window_subtitle_tokens=max_window_subtitle_tokens,
                parallel_windows=args.parallel_windows,
            )
            print(f"Wrote research context: {context_path}")
        else:
            print(f"Reused research context: {context_path}")
        _seed_transfer_from_context(context_path, fast_kwargs)
        if args.research_only:
            if task_artifact_dir:
                booking["outputs"] = {"research_context": str(context_path)}
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

    out = execute_correction_windows(
        stable_json=args.input,
        output_path=args.output,
        context_pack=context_pack,
        audio_label=audio_label,
        audio_path=args.audio,
        video_path=args.video,
        test_profile=args.test_profile,
        max_retries_per_window=args.max_retries_per_window,
        max_replacements_per_window=args.max_replacements_per_window,
        parallel_window_limit=args.parallel_windows,
        postprocess_profile=args.postprocess_profile,
        extra_style=args.extra_style,
        style_block=render_style_block(args.knowledge_root, cli_style.names),
        knowledge_enabled=knowledge_collects(args.knowledge),
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
    booking["outputs"] = {
        "translated_srt": str(Path(args.output).with_name(f"{Path(args.output).stem}-translated.srt")),
        **({"final_srt": str(out)} if Path(out).exists() else {}),
    }
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
            difficulty=profile.difficulty,
            style_names=cli_style.writable,
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
