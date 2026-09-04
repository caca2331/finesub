"""Plan a correction task's windows, execute them, render what came back.

``execute_correction_windows`` is the public entry point: it resolves the
window plan (or restores the one an earlier pass recorded), builds the
:class:`~finesub.llm.stages.correction.context.CorrectionRun` every window shares,
hands it to the serial or the parallel driver, and turns the accumulated rows
into the run's SRT/CSV outputs. The per-window work lives next door.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import threading
from .progress import WindowProgress
from finesub.reporting import current_reporter

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from finesub.media.clips import (
    CLIP_VIDEO_SUFFIX,
    default_clip_dir,
    extract_window_clip,
    extract_window_video_clip,
    probe_audio_duration,
)
from finesub.subtitles.postprocess import (
    DEFAULT_POSTPROCESS_PROFILE,
    TIMELINE_POSTPROCESS_PROFILES,
    postprocess_srt_file,
)
from finesub_bootstrap.artifacts import ARTIFACT_DIR_SUFFIX

from ...routing.capabilities import (
    correction_planning_envelope_description,
    correction_planning_limits,
    correction_task_group,
    planning_task_group,
)
from ...chunking import (
    SubtitleWindow,
    load_segments_from_stable_json,
    plan_correction_windows,
    rebuild_windows_from_plan,
    render_segments_as_srt,
    window_plan_payload,
)
from ...agent.agent_session_host import (
    set_run_evidence_destination,
    within_agent_session_scope,
)
from .attempts import correction_role_for_profile
from ...client import RoleClient, sum_token_distributions
from ...media_upload import (
    UploadedFileRef,
    window_media_ref,
)
from ...clip_prefetch import WindowClipPrefetcher
from ...routing.config import (
    LLMRole,
    MAX_WINDOW_SEARCH_QUERIES,
    ModelLimits,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    effective_window_subtitle_cap,
)
from ...content_filter import load_content_filter_blacklist
from ...exchange_log import ExchangeLogger
from ...knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    knowledge_version as current_knowledge_version,
    load_index_text,
)
from ...output_protocol import (
    render_corrected_segments_as_srt,
    render_translated_segments_as_csv,
    render_translated_segments_as_srt,
)
from ...routing.profiles import DEFAULT_PROFILE, TranslationProfile, max_window_csv_tokens
from ...prompts import PROMPT_VERSION, ContextPack
from ...research import plan_geometry_metadata, stable_json_source_hash
from ...session_checkpoint import SessionCheckpointStore
from ...task_report import write_task_report
from ...token_budget import TokenCounter, default_token_counter
from ...token_truncate import cap_tokens
from ...web_search import WebSearchClient
from .commit import (
    WINDOW_CACHE_FILENAME,
    WINDOW_PLAN_FILENAME,
    _media_identity,
    _task_fingerprint,
    _window_input_hash,
    _write_text_atomic,
)
from .context import (
    CarriedContext,
    CorrectionRun,
    ResumeLedger,
    WindowGeometry,
    WindowMedia,
)
from .metadata import _response_reference_metadata
from .parallel import run_parallel_windows
from .query_round import QueryRoundProduct
from .serial import run_serial_windows


def _shadow_scan_windows(
    knowledge_root: Any, task_id: str, version: str, windows: Sequence[Any]
) -> None:
    """Exact-match shadow pass (plan §4.2 step 4a): scan each window's raw
    text against the pinned corpus and book flat ``matched`` events. Pure
    telemetry — nothing about injection reads it — so it is never worth
    failing a run over, and reruns deduplicate on ``(task, window, rev)``.
    """

    try:
        from ...knowledge.base import knowledge_root_path
        from ...knowledge.node.matching import shadow_scan
        from ...knowledge.node.repo import KnowledgeRepo

        rev = int(version.rsplit(":", 1)[1]) if ":" in version else None
        store = KnowledgeRepo.open(knowledge_root_path(knowledge_root)).store
        inserted = shadow_scan(
            store,
            (
                (window.chunk_id, " ".join(segment.text for segment in window.segments))
                for window in windows
            ),
            task_id=task_id,
            rev=rev,
        )
        current_reporter().debug(
            f"knowledge shadow scan: {inserted} new matched event(s) across "
            f"{len(windows)} window(s) at {version or 'current rev'}"
        )
    except Exception as exc:  # telemetry must never sink the run
        current_reporter().debug(f"knowledge shadow scan skipped: {exc}")


def _log_landed_windows(
    knowledge_root: Any,
    task_id: str,
    version: str,
    windows: Sequence[Any],
    rendered_segments: Sequence[Any],
) -> None:
    """Post-run ``landed`` pass (plan §5.1): per window, canonical names that
    appear in the corrected text but not in the raw text. Rendered segments
    are assigned back to their window through ``source_ids`` — the same
    ownership `commit_window` merged them under. Fail-soft like the shadow
    scan; reruns deduplicate on the event key."""

    try:
        from ...knowledge.base import knowledge_root_path
        from ...knowledge.node.repo import KnowledgeRepo
        from ...knowledge.node.signals import log_landed_windows

        window_of: dict[str, str] = {}
        for window in windows:
            for source_id in window.source_ids:
                window_of[source_id] = window.chunk_id
        corrected_parts: dict[str, list[str]] = {}
        for segment in rendered_segments:
            owner = next(
                (window_of[sid] for sid in segment.source_ids if sid in window_of), None
            )
            if owner is not None and segment.corrected_text:
                corrected_parts.setdefault(owner, []).append(segment.corrected_text)
        rev = int(version.rsplit(":", 1)[1]) if ":" in version else None
        store = KnowledgeRepo.open(knowledge_root_path(knowledge_root)).store
        inserted = log_landed_windows(
            store,
            (
                (
                    window.chunk_id,
                    " ".join(segment.text for segment in window.segments),
                    " ".join(corrected_parts.get(window.chunk_id, [])),
                )
                for window in windows
                if corrected_parts.get(window.chunk_id)
            ),
            task_id=task_id,
            rev=rev,
        )
        current_reporter().debug(
            f"knowledge landed scan: {inserted} new event(s) at {version or 'current rev'}"
        )
    except Exception as exc:  # telemetry must never sink the run
        current_reporter().debug(f"knowledge landed scan skipped: {exc}")


def _report_correction_summary(run: CorrectionRun) -> None:
    """Close the stage with the numbers, not a sentence.

    Most of it is derived from the token rows the run already accumulated --
    one per call, carrying the attempt index and the tier that served it. Two
    tallies (repair rounds, content-filter recoveries) are counted on the run
    as they happen, because deriving them would cost more than the line is
    worth. These are a closing one-liner, not an audit trail: the per-window
    truth lives in `correction-windows.jsonl` and the exchange logs, and this
    summary does not try to restate it exactly.
    """

    calls = [row for row in run.token_rows if row.get("call") == "correction_window"]
    metrics: dict[str, object] = {}
    if run.progress is not None:
        done, total = run.progress.counts
        metrics["窗口"] = f"{done}/{total}" if done != total else done
        metrics["拆窗"] = run.progress.splits
    metrics["调用"] = len(calls)
    metrics["重试"] = sum(1 for row in calls if int(row.get("attempt") or 0) > 0)
    metrics["修复轮"] = run.repair_rounds
    metrics["内容过滤"] = run.content_filter_recoveries
    by_tier: dict[str, int] = {}
    for row in calls:
        tier = str(row.get("capability_tier") or "")
        if tier:
            by_tier[tier] = by_tier.get(tier, 0) + 1
    for tier, count in sorted(by_tier.items()):
        metrics[f"{tier} 调用"] = count
    current_reporter().summary("translated-srt", metrics)


@dataclass(frozen=True)
class _RefitEnvelope:
    """What a reused window has to fit today, and how to say it did not.

    A reused boundary plan is identity, not a promise that each pending leaf
    still fits the current profile. Both the predicate and the diagnostic live
    here because they name the same four limits, and a diagnostic that lists
    limits the predicate did not use is worse than none.
    """

    limits: ModelLimits
    output_csv_limit: int
    quality_csv_limit: int
    source: str

    @classmethod
    def of(
        cls,
        profile: TranslationProfile,
        limits: ModelLimits,
        max_window_subtitle_tokens: int,
    ) -> "_RefitEnvelope":
        return cls(
            limits=limits,
            output_csv_limit=max_window_csv_tokens(profile, limits=limits),
            # Resolved through the same limits the planner used
            # (`plan_correction_windows` -> `effective_window_subtitle_cap`).
            # Reading the cap through DEFAULT_LIMITS instead would make the
            # refit disagree with the planner the day a catalog entry carries
            # its own cap -- and the disagreement's direction is "split every
            # window on every resume, permanently".
            quality_csv_limit=effective_window_subtitle_cap(
                max_window_subtitle_tokens, limits
            ),
            source=correction_planning_envelope_description(profile),
        )

    def failures(self, win: SubtitleWindow) -> List[str]:
        budget = win.budget
        failures: List[str] = []
        if budget.input_tokens > self.limits.prompt_input_limit:
            failures.append("input_envelope")
        if budget.estimated_output_tokens > self.limits.output_limit:
            failures.append("raw_output_envelope")
        if budget.subtitle_input_tokens > self.output_csv_limit:
            failures.append("output_envelope")
        if 0 < self.quality_csv_limit < budget.subtitle_input_tokens:
            failures.append("quality_cap")
        return failures

    def error(
        self, win: SubtitleWindow, failures: List[str], reason: str
    ) -> ValueError:
        return ValueError(
            "Reused window cannot fit current execution envelope and is "
            f"{reason}: chunk={win.chunk_id}, source_ids={win.source_ids}, "
            f"estimated_input={win.budget.input_tokens}, "
            f"csv_tokens={win.budget.subtitle_input_tokens}, "
            f"input_limit={self.limits.prompt_input_limit}, "
            f"output_csv_limit={self.output_csv_limit}, "
            f"quality_csv_limit={self.quality_csv_limit}, "
            f"culprit={','.join(failures)}, envelope_source={self.source}"
        )


# The run opens the scope (`correction_translation.run_full_correction`), so
# that research and the knowledge update share the windows' sessions. This is
# for the callers that enter here directly -- a replay, a test -- and is
# re-entrant, so inside a run it is the run's scope that stays in charge.
@within_agent_session_scope
def execute_correction_windows(
    *,
    stable_json: str | Path,
    output_path: str | Path,
    context_pack: ContextPack | None = None,
    audio_label: str = "",
    audio_path: str | Path | None = None,
    video_path: str | Path | None = None,
    clip_dir: str | Path | None = None,
    test_profile: bool = False,
    max_retries_per_window: int = 5,
    max_replacements_per_window: int = 1,
    search_client: WebSearchClient | None = None,
    max_search_queries_per_window: int = MAX_WINDOW_SEARCH_QUERIES,
    postprocess_profile: int | None = DEFAULT_POSTPROCESS_PROFILE,
    extra_style: str = "",
    style_block: str = "",
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    task_update_feedback: bool = False,
    token_counter: TokenCounter | None = None,
    resume: bool = True,
    profile: TranslationProfile = DEFAULT_PROFILE,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    windows_override: List[SubtitleWindow] | None = None,
    parallel_window_limit: int = 4,
    seed_query_results: Mapping[str, "QueryRoundProduct"] | None = None,
    entry_details: str = "",
    evidence_pack_mode: bool = False,
    file_ref_seed: Mapping[str, UploadedFileRef] | None = None,
    extra_fingerprint: str = "",
    initial_transfer_keys: Sequence[str] = (),
    max_window_subtitle_tokens: int | None = None,
) -> Path:
    """Execute the correction windows (planned here unless overridden).

    Fast mode passes ``windows_override`` (the single fused window),
    ``seed_query_results`` (round-1 products keyed by base chunk id, replacing
    the per-window query round), ``entry_details`` / ``evidence_pack_mode``
    (round-2 injections), ``file_ref_seed`` (the round-1 clip upload, reused
    instead of re-uploading) and ``extra_fingerprint`` (folds the seeded
    injections into the resume fingerprint). ``knowledge_root`` feeds the
    query round's knowledge-index exposure and entry requests.

    ``video_path``: whichever switch says ``video`` reads a low-res
    video+audio ``.mp4`` clip of the window; a switch saying ``audio`` reads
    the ``.aac``. When both switches name the same kind the two rounds share
    one clip and one upload (model-routing v2).
    """

    segments = load_segments_from_stable_json(stable_json)
    context_pack = (context_pack or ContextPack()).with_source_order(
        [segment.id for segment in segments]
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_srt_path = out.with_name(f"{out.stem}-raw.srt")
    _write_text_atomic(raw_srt_path, render_segments_as_srt(segments))
    if postprocess_profile is None or postprocess_profile in (0, 1):
        # Keep raw ASR text untouched while applying the same timeline-only
        # policy as the final profile; `validate=False` -- two passes, one file.
        for timeline_profile in TIMELINE_POSTPROCESS_PROFILES:
            postprocess_srt_file(raw_srt_path, profile=timeline_profile, validate=False)

    token_counter = token_counter or default_token_counter()
    audio_duration = probe_audio_duration(audio_path) if audio_path else None
    # Group planning envelope (plan v2 D13): min limits over the correction
    # cell's bound group; DEFAULT_LIMITS for whole-Gemini groups.
    planning_limits = correction_planning_limits(profile)
    plan_report: Dict[str, Any] = {}
    if windows_override is not None:
        windows = list(windows_override)
    else:
        windows = plan_correction_windows(
            segments,
            context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
            counter=token_counter,
            audio_duration=audio_duration,
            profile=profile,
            report_sink=plan_report,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
            limits=planning_limits,
        )
    global_first_id = segments[0].id if segments else ""
    global_last_id = segments[-1].id if segments else ""
    # Where this task's own files go. A run names it; a stage entered directly
    # does not, and derives the same directory the pipeline would have -- the
    # alternative is scratch under the *working directory*, which in a packaged
    # install is the runtime source snapshot an update replaces.
    artifact_home = (
        Path(task_artifact_dir)
        if task_artifact_dir
        else Path(output_path).expanduser().with_suffix(ARTIFACT_DIR_SUFFIX)
    )
    # Window clips are scratch; anchoring them here also means the task's
    # cleanup takes them with it.
    clip_base_dir = Path(clip_dir) if clip_dir else default_clip_dir(artifact_home)
    # A conversational session's kept exchanges are filed when the run's scope
    # closes; whichever scope that is -- the run's, or the one the decorator
    # opened for a direct call -- this is where the destination is known. The
    # derived directory counts: a direct call is exactly the case with no other
    # record of what was asked and answered, so falling back to "nowhere" threw
    # away the only copy.
    set_run_evidence_destination(artifact_home)
    client = RoleClient(
        test_profile=test_profile,
    )
    if profile.continuity == "parallel" and getattr(
        client, "routes_to_conversational", lambda *a, **k: False
    )(
        correction_role_for_profile(profile),
        task_group=correction_task_group(profile),
        difficulty=profile.difficulty,
    ):
        # Task-parallelism plan W6: a conversational chain is served by a
        # person's own agent -- one queue, however many agents actually join.
        # Fan-out would queue N assignments behind (usually) one agent: the
        # wall clock of serial with the advice ledger stripped from every
        # prompt. Forced, not warned-and-honoured; on-demand concurrency is
        # deferred (plan §4.1).
        current_reporter().warning(
            "conversational-forced-serial",
            "conversational 后端不支持窗口并行，continuity 强制为 serial",
            impact="窗口顺序执行，advice 台账保留",
        )
        profile = dataclasses.replace(profile, continuity="serial")
    ensure_eligible = getattr(client, "ensure_eligible_target", None)
    if callable(ensure_eligible):
        if audio_path and profile.correction_use_audio:
            ensure_eligible(
                LLMRole.AUDIO_MULTIMODAL,
                needs_audio=True,
                needs_video=bool(video_path) and profile.correction_use_video,
                native_search=profile.native_search,
                task_group=correction_task_group(profile),
                difficulty=profile.difficulty,
            )
        if profile.external_injection and audio_path and profile.planning_use_audio:
            ensure_eligible(
                LLMRole.LIGHTWEIGHT_MULTIMODAL,
                needs_audio=True,
                needs_video=bool(video_path) and profile.planning_use_video,
                task_group=planning_task_group(profile),
                difficulty=profile.difficulty,
            )
    # One clip + upload per executed window (exact chunk id: -a/-b halves get
    # their own clips); reused across the query round, the correction round
    # and same-window validation retries. Extraction + upload run on a
    # background thread: window 0 is scheduled after planning; at the start of
    # window i we schedule i+1 so the main loop rarely waits on ffmpeg.
    #
    # Clip ownership (model-routing v2): a clip kind is cut iff *either* switch asks
    # for it -- the query round is a first-class owner now, so
    # ``correction_media=text`` no longer turns the machinery off while
    # ``planning_media`` still wants a clip. When both switches name the same
    # kind, the two rounds share one clip and one upload.
    clip_prefetcher: WindowClipPrefetcher | None = None
    video_prefetcher: WindowClipPrefetcher | None = None
    # One-level video->audio safety-net (model-routing v2): the ladder's on-demand
    # .aac cutter, created only if a video-incapable target ever answers.
    make_ladder: Callable[[], WindowClipPrefetcher] | None = None
    planning_active = profile.external_injection  # only ``local`` runs r1
    wants_audio_clip = profile.correction_media == "audio" or (
        planning_active and profile.planning_media == "audio"
    )
    wants_video_clip = bool(video_path) and (
        profile.correction_use_video
        or (planning_active and profile.planning_use_video)
    )
    correction_use_video = bool(video_path) and profile.correction_use_video
    if audio_path and (wants_audio_clip or wants_video_clip):

        def _tracked_upload(path: Path, cancel: threading.Event) -> UploadedFileRef:
            ref = window_media_ref(
                path,
                execution_settings=getattr(client, "execution_settings", None),
                routes=getattr(getattr(client, "router", None), "routes", None),
                cancel=cancel,
            )
            if task_artifact_dir:
                append_task_artifact(
                    task_artifact_dir,
                    kind="api_call",
                    task_id=task_id,
                    payload={
                        "category": "gemini_file_upload",
                        "filename": path.name,
                        "file_id": ref.file_id,
                    },
                )
            return ref

        prefetch_workers = 2 if profile.continuity == "parallel" else 1
        if wants_audio_clip:
            clip_prefetcher = WindowClipPrefetcher(
                audio_path,
                clip_base_dir,
                extract_fn=extract_window_clip,
                upload_fn=_tracked_upload,
                max_workers=prefetch_workers,
            )
        if wants_video_clip:
            video_prefetcher = WindowClipPrefetcher(
                video_path,
                clip_base_dir,
                extract_fn=extract_window_video_clip,
                upload_fn=_tracked_upload,
                clip_suffix=CLIP_VIDEO_SUFFIX,
                max_workers=prefetch_workers,
            )
        if correction_use_video:

            def _build_ladder() -> WindowClipPrefetcher:
                return WindowClipPrefetcher(
                    audio_path,
                    clip_base_dir,
                    extract_fn=extract_window_clip,
                    upload_fn=_tracked_upload,
                    max_workers=1,
                )

            make_ladder = _build_ladder

    media = WindowMedia(
        profile=profile,
        audio_path=Path(audio_path) if audio_path else None,
        video_path=Path(video_path) if video_path else None,
        audio_label=audio_label,
        file_ref_seed=dict(file_ref_seed or {}),
        audio_clips=clip_prefetcher,
        video_clips=video_prefetcher,
        _make_ladder=make_ladder,
    )

    # The per-window query round and injected search only exist on the mm
    # route; the text route never runs harness-side retrieval.
    external_injection = profile.external_injection
    if external_injection and search_client is None and not seed_query_results:
        search_client = WebSearchClient(
            execution_settings=getattr(client, "execution_settings", None)
        )
    # Query round output is fetched once per planned window (keyed by the base
    # chunk id) and reused across validation retries and -a/-b split halves.
    # Fast mode seeds it with round-1 products so no query round ever runs.
    query_round_cache: Dict[str, QueryRoundProduct] = dict(seed_query_results or {})
    # ``knowledge_enabled`` is the caller's tri-state (``--knowledge none``
    # means the base is not read at all); an empty or unreadable base disables
    # the machinery just the same, rather than showing entry rules against an
    # "(empty)" index. Both have to hold, and the indices are not even loaded
    # when the switch is off.
    streamer_index_text = (
        load_index_text(knowledge_root, "streamer") if knowledge_enabled else ""
    )
    common_index_text = (
        load_index_text(knowledge_root, "common") if knowledge_enabled else ""
    )
    knowledge_enabled = knowledge_enabled and bool(
        streamer_index_text.strip() or common_index_text.strip()
    )
    content_filter_blacklist = load_content_filter_blacklist(task_artifact_dir)
    # v17 entry pass-through: canonical keys kept by the previous step (the
    # research round 2 seeds window one; each window's correction round emits
    # <keep_entries> for the next). The set is injected into both the query
    # round (context) and the correction round, and its keys+bodies enter the
    # per-window resume input hash.
    # Without a query round no window can request an entry back, so the
    # transfer chain is the only copy: it carries the whole window budget
    # rather than the increment-sized reserve, and is never pruned (§1.4).
    transfer_cap = (
        KB_TRANSFER_MAX_ENTRIES if external_injection else KB_WINDOW_TOTAL_ENTRIES
    )
    carried = CarriedContext.starting_from(
        initial_transfer_keys, knowledge_root=knowledge_root, cap=transfer_cap
    )

    exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)

    # Mid-loop resume: a committed window survives every configuration change
    # the whitelist above lists, and only those.
    task_fingerprint = _task_fingerprint(
        prompt_version=PROMPT_VERSION,
        test_profile=test_profile,
        source_fingerprint=stable_json_source_hash(stable_json),
        media_identity={
            "audio": _media_identity(audio_path),
            "video": _media_identity(video_path),
        },
        extra=extra_fingerprint,
    )
    # Recorded per window, never a fingerprint input: the knowledge base
    # auto-commits, so gating on it would let any other task's update discard
    # this task's progress (docs/llm_local_agent.md SS8 binds a *task's* retries
    # to one version, not a whole run's windows).
    knowledge_version = (
        current_knowledge_version(knowledge_root) if knowledge_enabled else ""
    )
    plan_geometry = plan_geometry_metadata(
        profile,
        audio_duration=audio_duration,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        limits=planning_limits,
    )
    window_cache_path = (
        Path(task_artifact_dir) / WINDOW_CACHE_FILENAME if task_artifact_dir else None
    )
    geometry = WindowGeometry(
        profile=profile,
        counter=token_counter,
        limits=planning_limits,
        global_first_id=global_first_id,
        global_last_id=global_last_id,
        audio_duration=audio_duration,
    )
    ledger = ResumeLedger(
        enabled=bool(resume and window_cache_path is not None),
        path=window_cache_path,
        task_fingerprint=task_fingerprint,
    )
    plan_reused = False
    # The saved plan owns only boundary identity.  Its media ranges and token
    # budgets are derived again under the current profile/envelope below.
    if windows_override is None and ledger.enabled and task_artifact_dir:
        plan_path = Path(task_artifact_dir) / WINDOW_PLAN_FILENAME
        restored: List[SubtitleWindow] | None = None
        if plan_path.exists():
            try:
                stored = json.loads(plan_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = None
            if (
                isinstance(stored, dict)
                and stored.get("source_fingerprint") == task_fingerprint
            ):
                restored = rebuild_windows_from_plan(
                    stored.get("plan"),
                    segments,
                    counter=token_counter,
                    limits=planning_limits,
                    audio_duration=audio_duration,
                    profile=profile,
                    context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
                )
        if restored is not None:
            windows = restored
            plan_reused = True
            ledger.load()
        else:
            _write_text_atomic(
                plan_path,
                json.dumps(
                    {
                        "source_fingerprint": task_fingerprint,
                        "geometry": plan_geometry,
                        "plan": window_plan_payload(windows),
                    },
                    ensure_ascii=False,
                ),
            )
            # A lost, corrupt or re-planned plan means the old chunk ids no
            # longer address anything. The ledger stays empty even if some ids
            # happen to collide.
    elif ledger.enabled:
        # Fast mode supplies its single full-source window explicitly and does
        # not need a separate plan file.
        ledger.load()
    if (
        not plan_reused
        and task_artifact_dir
        and plan_report.get("replan_attempts")
    ):
        append_task_artifact(
            task_artifact_dir,
            kind="window_plan_report",
            task_id=task_id,
            payload={"phase": "correction", **plan_report},
        )
    # Reuse the parsed append-only ledger across windows. Constructing a store
    # per query would repeatedly rescan the same JSONL file on long tasks.
    session_checkpoint_store = SessionCheckpointStore(
        task_artifact_dir, enabled=resume
    )

    windows = [
        leaf
        for window in windows
        for leaf in ledger.expand_cached_splits(window, geometry)
    ]

    # G2: a reused boundary plan is identity, not a promise that each pending
    # leaf fits today's profile/envelope.  Keep valid cached leaves untouched;
    # recursively refit only work that would otherwise be dispatched.
    refit_rows: List[Dict[str, Any]] = []
    envelope = _RefitEnvelope.of(profile, planning_limits, max_window_subtitle_tokens)

    def _refit_pending(win: SubtitleWindow) -> List[SubtitleWindow]:
        if ledger.enabled and ledger.leaf_is_replayable(win):
            return [win]
        failures = envelope.failures(win)
        if not failures:
            return [win]
        # The refit creates splits, so it spends the same budget the retry loop
        # does -- and recursing without asking is how a reused plan used to
        # halve past the cap once per resume.
        if not geometry.may_split(win):
            raise envelope.error(
                win, failures, f"already {geometry.MAX_SPLITS} split(s) deep"
            )
        halves = geometry.split(win)
        if halves is None:
            raise envelope.error(win, failures, "unsplittable")
        marker = {
            "chunk_id": win.chunk_id,
            "source_ids": list(win.source_ids),
            "input_hash_core": _window_input_hash(win),
            "split_into": [half.chunk_id for half in halves],
            "task_fingerprint": task_fingerprint,
            "continuity": profile.continuity,
            "reason": "resume_refit",
        }
        ledger.commit(marker)
        refit_rows.append(
            {
                "chunk_id": win.chunk_id,
                "source_ids": list(win.source_ids),
                "split_into": marker["split_into"],
                "failures": failures,
                "estimated_input": win.budget.input_tokens,
                "csv_tokens": win.budget.subtitle_input_tokens,
            }
        )
        return [leaf for half in halves for leaf in _refit_pending(half)]

    if plan_reused:
        windows = [leaf for window in windows for leaf in _refit_pending(window)]
        if refit_rows:
            current_reporter().warning(
                "correction-windows-refit",
                f"refit {len(refit_rows)} reused correction window(s) to the "
                "current execution envelope",
            )
            if task_artifact_dir:
                append_task_artifact(
                    task_artifact_dir,
                    kind="window_refit_report",
                    task_id=task_id,
                    payload={
                        "input_limit": envelope.limits.prompt_input_limit,
                        "output_csv_limit": envelope.output_csv_limit,
                        "quality_csv_limit": envelope.quality_csv_limit,
                        "envelope_source": envelope.source,
                        "splits": refit_rows,
                    },
                )
    # Primed here rather than at construction, so the resume cache is known: a
    # window that will be answered from it needs no clip, and cutting one costs
    # an ffmpeg run plus a Gemini Files upload for nothing.
    if (
        windows
        and not (file_ref_seed and windows[0].chunk_id in file_ref_seed)
        and not ledger.holds(windows[0].chunk_id)
    ):
        media.schedule_correction(windows[0])

    if knowledge_enabled:
        _shadow_scan_windows(knowledge_root, task_id, knowledge_version, windows)

    run = CorrectionRun(
        profile=profile,
        context_pack=context_pack,
        knowledge_root=knowledge_root,
        knowledge_enabled=knowledge_enabled,
        extra_style=extra_style,
        style_block=style_block,
        task_update_feedback=task_update_feedback,
        test_profile=test_profile,
        resume=resume,
        max_retries_per_window=max_retries_per_window,
        max_replacements_per_window=max_replacements_per_window,
        max_search_queries_per_window=max_search_queries_per_window,
        parallel_window_limit=parallel_window_limit,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
        entry_details=entry_details,
        evidence_pack_mode=evidence_pack_mode,
        external_injection=external_injection,
        transfer_cap=transfer_cap,
        client=client,
        search_client=search_client,
        token_counter=token_counter,
        exchange_logger=exchange_logger,
        session_checkpoint_store=session_checkpoint_store,
        planning_limits=planning_limits,
        content_filter_blacklist=content_filter_blacklist,
        streamer_index_text=streamer_index_text,
        common_index_text=common_index_text,
        media=media,
        geometry=geometry,
        ledger=ledger,
        carried=carried,
        task_fingerprint=task_fingerprint,
        knowledge_version=knowledge_version,
        query_round_cache=query_round_cache,
    )

    run.progress = WindowProgress(len(windows))
    current_reporter().debug(
        "correction plan",
        {
            "windows": len(windows),
            "reused_plan": plan_reused,
            "driver": profile.continuity,
        },
    )
    try:
        if profile.continuity == "parallel":
            # Parallel dispatch has its own two-phase executor and never
            # mutates ``windows``; the serial driver is bypassed wholesale.
            run_parallel_windows(run, windows)
        else:
            run_serial_windows(run, windows)
    finally:
        media.shutdown()
        # In `finally` on purpose -- a run that died three windows in still
        # wants to say how far it got -- and guarded, because this block runs
        # while an exception may be in flight and commentary must never
        # replace the failure it was commenting on.
        try:
            _report_correction_summary(run)
        except Exception:  # noqa: BLE001 - reporting is never worth a run
            pass

    if knowledge_enabled:
        _log_landed_windows(
            knowledge_root, task_id, knowledge_version, windows, run.rendered_segments
        )

    merged = render_translated_segments_as_srt(run.rendered_segments)
    corrected = render_corrected_segments_as_srt(run.rendered_segments)
    translated_srt_path = out.with_name(f"{out.stem}-translated.srt")
    corrected_srt_path = out.with_name(f"{out.stem}-corrected.srt")
    _write_text_atomic(translated_srt_path, merged)
    _write_text_atomic(corrected_srt_path, corrected)
    # Full annotated CSV retains the model's type/conf/note (and inserts), which
    # the text-only SRTs drop; downstream analysis reads it.
    annotated_csv_path = out.with_name(f"{out.stem}-annotated.csv")
    _write_text_atomic(
        annotated_csv_path,
        "# type|position|duration|gap|corrected|translation|conf|char_count|note\n"
        + render_translated_segments_as_csv(run.rendered_segments),
    )
    postprocess_report = None
    result_path = translated_srt_path
    if postprocess_profile is not None:
        postprocess_report = postprocess_srt_file(
            translated_srt_path,
            output_path=out,
            profile=postprocess_profile,
        )
        result_path = out
    final_text = result_path.read_text(encoding="utf-8")
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="final_srt",
            task_id=task_id,
            payload={
                "path": str(result_path),
                "summary": _response_reference_metadata(final_text),
                "excerpt": cap_tokens(final_text, 12_000, token_counter.count_text),
                "raw_path": str(raw_srt_path),
                "translated_path": str(translated_srt_path),
                "translated_summary": _response_reference_metadata(merged),
                "translated_excerpt": cap_tokens(merged, 12_000, token_counter.count_text),
                "corrected_path": str(corrected_srt_path),
                "corrected_summary": _response_reference_metadata(corrected),
                "corrected_excerpt": cap_tokens(corrected, 12_000, token_counter.count_text),
                "postprocess": (
                    postprocess_report.to_dict() if postprocess_report is not None else None
                ),
            },
        )
        append_task_artifact(
            task_artifact_dir,
            kind="token_distribution_report",
            task_id=task_id,
            payload={
                "phase": "correction",
                "rows": run.token_rows,
                "totals": sum_token_distributions(row["tokens"] for row in run.token_rows),
            },
        )
        write_task_report(
            task_artifact_dir,
            task_id=task_id,
            outputs={
                "raw_srt": str(raw_srt_path),
                "translated_srt": str(translated_srt_path),
                "corrected_srt": str(corrected_srt_path),
                "final_srt": str(result_path),
            },
        )
    return result_path
