"""Fast-mode fused session: round 1 (research + query round) plus local search.

Round 1 gets the whole clip, the full CSV and the knowledge indices, and emits
<analysis_notes> (medium summary, wider cap than research round 1),
<requested_entries> and the search contract/queries. The optional search loop
(default 2 total rounds) then runs; the resulting evidence pack (or raw
results), the requested entry files and the notes are seeded straight into the
single correction window executed by ``execute_correction_windows``.

The session products persist into the ``*-research-context.json`` slot with a
``"mode": "fast"`` marker so ``--context-file`` reuse and coarse resume work
exactly like normal research.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from ..client import (
    GeminiPromptBlockedError,
    LiteLLMRoleClient,
    UploadedFileRef,
    sum_token_distributions,
    upload_gemini_file,
)
from ..config import (
    FAST_ANALYSIS_NOTES_MAX_TOKENS,
    INJECTION_SECTION_MAX_TOKENS,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    LLMRole,
    MAX_WINDOW_SEARCH_QUERIES,
    followup_search_query_limit,
    injection_block_token_limit,
    research_search_query_limit,
)
from ..content_filter import (
    load_content_filter_blacklist,
    run_injection_ladder,
    split_rendered_search_block,
)
from ..chunking import SubtitleWindow
from ..exchange_log import ExchangeLogger
from ..injection_budget import render_knowledge_entries_block
from ..knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    load_index_text,
)
from ..output_tags import (
    extract_single_tag_block,
    parse_guided_line_items,
    parse_line_items,
)
from ..profiles import (
    DEFAULT_FAST_SEARCH_ROUNDS,
    DEFAULT_PROFILE,
    TranslationProfile,
)
from ..prompts import PROMPT_VERSION, build_fast_round1_messages
from ..research import (
    ResearchRound1Result,
    _call_and_parse,
    _extract_optional_block,
    _render_note_url_extracts,
    check_research_input_limit,
    extract_round_task_feedback,
    research_knowledge_inputs_hash,
    render_preinjected_entries,
    resolve_round1_entries,
)
from ..search_loop import run_search_loop
from ..session_checkpoint import SessionCheckpointStore
from ..token_budget import default_token_counter, TokenCounter
from ..web_search import (
    QuerySearchResult,
    SearchRequest,
    WebSearchClient,
    extract_results_metadata,
    render_search_results,
    search_results_metadata,
)
from .correction_loop import QueryRoundProduct, _window_audio_label
from .plan import FAST_WINDOW_CHUNK_ID


@dataclass(frozen=True)
class FastSessionResult:
    analysis_notes: str = ""
    entry_details_text: str = ""
    search_results_text: str = ""
    evidence_pack_mode: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)

    def seed_query_results(self) -> Dict[str, "QueryRoundProduct"]:
        """Query-round seed for the single fast window (results + notes)."""

        return {
            FAST_WINDOW_CHUNK_ID: QueryRoundProduct(
                search_results=self.search_results_text,
                window_notes=self.analysis_notes,
            )
        }

    def fingerprint(self) -> str:
        """Hash of the seeded injections; feeds the window resume fingerprint."""

        blob = json.dumps(
            {
                "analysis_notes": self.analysis_notes,
                "entry_details_text": self.entry_details_text,
                "search_results_text": self.search_results_text,
                "evidence_pack_mode": self.evidence_pack_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return "fast:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _fast_context_planning_metadata(session_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Inputs that must match before a persisted fast context can be reused."""

    window = session_kwargs["window"]
    profile = session_kwargs.get("profile", DEFAULT_PROFILE)
    extra_info = str(session_kwargs.get("extra_info") or "")
    knowledge_root = session_kwargs.get("knowledge_root", DEFAULT_KNOWLEDGE_ROOT)
    enable_web_search = bool(session_kwargs.get("enable_web_search", True))
    search_rounds = (
        int(session_kwargs.get("search_rounds", DEFAULT_FAST_SEARCH_ROUNDS))
        if enable_web_search
        else 0
    )
    source_payload = {
        "clip_start": round(float(window.clip_start), 3),
        "clip_end": round(float(window.clip_end), 3),
        "segments": [
            [segment.id, segment.start, segment.end, segment.text]
            for segment in window.segments
        ],
    }

    def _path_signature(value: Any) -> Dict[str, Any] | None:
        if not value:
            return None
        path = Path(value).expanduser().resolve()
        try:
            stat = path.stat()
        except OSError:
            return {"path": str(path), "missing": True}
        return {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        "prompt_version": PROMPT_VERSION,
        "profile_id": profile.profile_id,
        "output_scale": profile.output_scale,
        "source_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                source_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()[:16],
        "extra_info_hash": "sha256:"
        + hashlib.sha256(extra_info.encode("utf-8")).hexdigest()[:16],
        "knowledge_inputs_hash": research_knowledge_inputs_hash(
            knowledge_root, extra_info
        ),
        "enable_web_search": enable_web_search,
        "search_rounds": search_rounds,
        "collect_task_feedback": bool(
            session_kwargs.get("collect_task_feedback", False)
        ),
        "audio": _path_signature(session_kwargs.get("audio_path")),
        "video": _path_signature(session_kwargs.get("video_path")),
    }


def parse_fast_round1_output(
    text: str,
    *,
    expect_contract: bool = False,
    collect_task_feedback: bool = False,
    count_tokens: Callable[[str], int] | None = None,
) -> ResearchRound1Result:
    """Like research round 1, but with the wider fast notes cap.

    Fast round 1 is the research final round's equivalent feedback collection
    point; the block is best-effort and never fails the parse.
    """

    entries_body = extract_single_tag_block(text, "requested_entries")
    keep_body = extract_single_tag_block(text, "keep_entries")
    queries_body = extract_single_tag_block(text, "search_queries")
    contract = ""
    if expect_contract:
        contract = extract_single_tag_block(text, "research_contract").strip()
    return ResearchRound1Result(
        requested_entries=tuple(parse_line_items(entries_body)),
        keep_entries=tuple(parse_line_items(keep_body)),
        search_queries=tuple(parse_line_items(queries_body)),
        analysis_notes=_extract_optional_block(
            text,
            "analysis_notes",
            max_tokens=FAST_ANALYSIS_NOTES_MAX_TOKENS,
            count_tokens=count_tokens,
        ),
        research_contract=contract,
        task_update_feedback=(
            extract_round_task_feedback(text, count_tokens=count_tokens)
            if collect_task_feedback
            else ""
        ),
    )


def _result_from_payload(payload: Dict[str, Any]) -> FastSessionResult:
    fast = payload.get("fast") or {}
    return FastSessionResult(
        analysis_notes=str(fast.get("analysis_notes") or ""),
        entry_details_text=str(fast.get("entry_details_text") or ""),
        search_results_text=str(fast.get("search_results_text") or ""),
        evidence_pack_mode=bool(fast.get("evidence_pack_mode")),
        payload=payload,
    )


def load_fast_context(path: str | Path) -> FastSessionResult:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if data.get("mode") != "fast":
        raise ValueError(
            f"{path} is not a fast-mode context (mode={data.get('mode')!r}). "
            "Delete it or run with --fast off to reuse it as normal research."
        )
    return _result_from_payload(data)


def render_entry_details_text(
    entry_details: Dict[str, str],
    *,
    count_tokens: Callable[[str], int],
) -> str:
    """Budget-rendered entry block; consumer is the fast correction window."""

    if not entry_details:
        return ""
    return render_knowledge_entries_block(
        entry_details,
        count_tokens=count_tokens,
        entry_limit=INJECTION_SECTION_MAX_TOKENS,
        block_limit=injection_block_token_limit(KB_WINDOW_TOTAL_ENTRIES),
    ).text


def _dump_fast_round1_input(
    task_artifact_dir: str | Path,
    *,
    window: Any,
    audio_file_label: str,
    extra_info: str,
    note_url_extracts: str,
    streamer_index: str,
    common_index: str,
    preinjected_entries: str,
    max_search_queries: int,
    use_search_contract: bool,
    collect_task_feedback: bool,
) -> None:
    """Persist fast round-1 input state for replay fixture extraction."""

    payload = {
        "window": {
            "chunk_id": getattr(window, "chunk_id", ""),
            "segment_count": len(getattr(window, "segments", [])),
            "clip_start": getattr(window.segments[0], "start", 0.0)
            if getattr(window, "segments", None)
            else 0.0,
            "clip_end": getattr(window.segments[-1], "end", 0.0)
            if getattr(window, "segments", None)
            else 0.0,
        },
        "audio_file_label": audio_file_label,
        "extra_info": extra_info,
        "note_url_extracts": note_url_extracts,
        "streamer_index": streamer_index,
        "common_index": common_index,
        "preinjected_entries": preinjected_entries,
        "max_search_queries": max_search_queries,
        "use_search_contract": use_search_contract,
        "collect_task_feedback": collect_task_feedback,
    }
    path = Path(task_artifact_dir) / "fast-round-input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fast_session(
    *,
    window: SubtitleWindow,
    segment_count: int,
    audio_path: str | Path | None = None,
    video_path: str | Path | None = None,
    clip_dir: str | Path | None = None,
    stable_json_stem: str = "",
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    client: LiteLLMRoleClient | None = None,
    search_client: WebSearchClient | None = None,
    enable_web_search: bool = True,
    search_rounds: int = DEFAULT_FAST_SEARCH_ROUNDS,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    max_parse_retries: int = 1,
    token_counter: TokenCounter | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    collect_task_feedback: bool = False,
    resume: bool = True,
) -> tuple[FastSessionResult, UploadedFileRef | None]:
    """Run fast round 1 + searches; returns the session result and the
    uploaded clip ref (reused by the correction window, never re-uploaded).

    On mm-high with ``video_path`` the clip is the low-res video+audio mp4
    (round 1 is fused with the correction flow, so it gets the same media as
    the correction round)."""

    from ..audio_clips import (
        CLIP_AUDIO_SUFFIX,
        CLIP_VIDEO_SUFFIX,
        extract_window_clip,
        extract_window_video_clip,
    )

    client = client or LiteLLMRoleClient(test_profile=test_profile)
    token_counter = token_counter or default_token_counter()
    exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)
    token_rows: List[Dict[str, Any]] = []
    multi_round = enable_web_search and int(search_rounds) > 1
    checkpoint_identity = _fast_context_planning_metadata(
        {
            "window": window,
            "profile": profile,
            "extra_info": extra_info,
            "knowledge_root": knowledge_root,
            "enable_web_search": enable_web_search,
            "search_rounds": search_rounds,
            "collect_task_feedback": collect_task_feedback,
            "audio_path": audio_path,
            "video_path": video_path,
        }
    )

    file_ref: UploadedFileRef | None = None
    audio_label = ""
    use_video = bool(video_path) and profile.use_video
    if audio_path and profile.use_audio:
        clip_base_dir = (
            Path(clip_dir)
            if clip_dir
            else Path("tmp") / "llm-audio-clips" / (stable_json_stem or "fast")
        )
        clip_base_dir.mkdir(parents=True, exist_ok=True)
        if use_video:
            clip_path = extract_window_video_clip(
                video_path,
                window.clip_start,
                window.clip_end,
                clip_base_dir / f"{window.chunk_id}{CLIP_VIDEO_SUFFIX}",
            )
        else:
            clip_path = extract_window_clip(
                audio_path,
                window.clip_start,
                window.clip_end,
                clip_base_dir / f"{window.chunk_id}{CLIP_AUDIO_SUFFIX}",
            )
        file_ref = upload_gemini_file(clip_path)
        audio_label = _window_audio_label(
            video_path if use_video else audio_path,
            "",
            window,
            clip_suffix=CLIP_VIDEO_SUFFIX if use_video else CLIP_AUDIO_SUFFIX,
        )
        if task_artifact_dir:
            append_task_artifact(
                task_artifact_dir,
                kind="api_call",
                task_id=task_id,
                payload={
                    "category": "gemini_file_upload",
                    "filename": Path(clip_path).name,
                    "file_id": file_ref.file_id,
                },
            )

    note_url_extracts = ""
    if extra_info.strip() and enable_web_search:
        search_client = search_client or WebSearchClient()
        note_url_extracts, note_urls, note_results = _render_note_url_extracts(
            extra_info, search_client, count_tokens=token_counter.count_text
        )
        if task_artifact_dir and note_urls:
            append_task_artifact(
                task_artifact_dir,
                kind="api_call",
                task_id=task_id,
                payload={
                    "category": "web_extract",
                    "source": "extra_info_urls",
                    "urls": note_urls,
                    "executed": extract_results_metadata(note_results),
                    "rendered_tokens": token_counter.count_text(note_url_extracts),
                },
            )

    preinjected_entries_text = ""
    preinjection_report: Dict[str, Any] = {"matches": []}
    if extra_info.strip():
        preinjected_entries_text, preinjection_report = render_preinjected_entries(
            knowledge_root, extra_info, count_tokens=token_counter.count_text
        )
        if task_artifact_dir and preinjection_report["matches"]:
            append_task_artifact(
                task_artifact_dir,
                kind="knowledge_preinjection",
                task_id=task_id,
                payload={"source": "fast_round1", **preinjection_report},
            )

    max_queries = research_search_query_limit(segment_count)
    content_filter_blacklist = load_content_filter_blacklist(task_artifact_dir)

    def _round1_call(extracts_text: str) -> ResearchRound1Result:
        messages = build_fast_round1_messages(
            window=window,
            audio_file_label=audio_label,
            extra_info=extra_info,
            note_url_extracts=extracts_text,
            streamer_index=load_index_text(knowledge_root, "streamer"),
            common_index=load_index_text(knowledge_root, "common"),
            preinjected_entries=preinjected_entries_text,
            max_search_queries=max_queries,
            use_search_contract=multi_round,
            collect_task_feedback=collect_task_feedback,
            profile=profile,
        )
        # Persist the full round-1 input state for session_replay fixture
        # extraction (docs/session_replay.md 补中间态).
        if task_artifact_dir:
            _dump_fast_round1_input(
                task_artifact_dir,
                window=window,
                audio_file_label=audio_label,
                extra_info=extra_info,
                note_url_extracts=extracts_text,
                streamer_index=load_index_text(knowledge_root, "streamer"),
                common_index=load_index_text(knowledge_root, "common"),
                preinjected_entries=preinjected_entries_text,
                max_search_queries=max_queries,
                use_search_contract=multi_round,
                collect_task_feedback=collect_task_feedback,
            )
        check_research_input_limit(
            messages, round_name="fast round 1", counter=token_counter
        )
        return _call_and_parse(
            client,
            messages,
            parser=lambda text: parse_fast_round1_output(
                text,
                expect_contract=multi_round,
                collect_task_feedback=collect_task_feedback,
                count_tokens=token_counter.count_text,
            ),
            round_name="fast round 1",
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind="fast_round1_response",
            max_parse_retries=max_parse_retries,
            token_rows=token_rows,
            exchange_logger=exchange_logger,
            token_counter=token_counter,
            input_components_kwargs={"note_url_extracts": extracts_text},
            role=LLMRole.GENERAL_CAPABLE,
            file_ref=file_ref,
            session_prefix="fast-round1",
            checkpoint_store=checkpoint_store,
            checkpoint_session="fast-round1",
            checkpoint_key=window.chunk_id,
            checkpoint_extra_identity=checkpoint_identity,
        )

    note_extract_block = split_rendered_search_block(note_url_extracts)
    round1_outcome = run_injection_ladder(
        block=note_extract_block,
        call=_round1_call,
        stage="fast_round1",
        blocked_exception=GeminiPromptBlockedError,
        blacklist=content_filter_blacklist,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
        plain_retry=not note_extract_block.units,
    )
    if round1_outcome.level >= 0:
        print(
            "Warning: fast round 1 prompt was blocked by the content filter; "
            f"recovered at ladder level {round1_outcome.level} "
            f"(dropped {len(round1_outcome.dropped_units)} injection unit(s)).",
            file=sys.stderr,
        )
    round1_result: ResearchRound1Result = round1_outcome.result

    visible_preinjected_keys = [
        *preinjection_report.get("included", []),
        *preinjection_report.get("truncated", []),
    ]
    (
        entry_details,
        missing_entries,
        ignored_keep_entries,
        dropped_entry_keys,
    ) = resolve_round1_entries(
        knowledge_root,
        requested_names=round1_result.requested_entries,
        keep_names=round1_result.keep_entries,
        visible_keep_keys=visible_preinjected_keys,
        max_requested_entries=KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
        max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
        max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
    )
    entry_details_text = render_entry_details_text(
        entry_details, count_tokens=token_counter.count_text
    )

    search_results: List[QuerySearchResult] = []
    search_loop_metadata: Dict[str, Any] = {}
    search_results_text = ""
    search_render_report: Dict[str, Any] = {}
    if multi_round and (round1_result.search_queries or round1_result.research_contract):
        search_client = search_client or WebSearchClient()
        background_parts = [
            part for part in (extra_info.strip(), round1_result.analysis_notes) if part
        ]
        loop_result = run_search_loop(
            contract_body=round1_result.research_contract,
            round0_queries=round1_result.search_queries,
            client=client,
            search_client=search_client,
            background="\n\n".join(background_parts),
            max_rounds=int(search_rounds),
            round0_query_cap=max_queries,
            followup_query_cap=followup_search_query_limit(max_queries),
            max_parse_retries=max_parse_retries,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind="search_loop_round",
            exchange_prefix="fast-search-loop",
            token_rows=token_rows,
            exchange_logger=exchange_logger,
            token_counter=token_counter,
            knowledge_root=knowledge_root,
            persistent_entries_text=entry_details_text,
            persistent_entry_keys=list(entry_details),
            persistent_requested_entry_names=round1_result.requested_entries,
            persistent_kept_entry_names=round1_result.keep_entries,
            content_filter_blacklist=content_filter_blacklist,
            resume=resume,
        )
        search_results_text = loop_result.evidence_pack
        search_loop_metadata = loop_result.to_metadata()
    elif enable_web_search and round1_result.search_queries:
        search_client = search_client or WebSearchClient()
        search_requests = [
            SearchRequest(query=query, guided_query=guided)
            for query, guided in parse_guided_line_items(
                "\n".join(round1_result.search_queries)
            )
        ]
        search_results = search_client.search_many(
            search_requests, max_queries=max_queries
        )
        # The consumer is the single fast correction window, so the block
        # budget aligns with the correction-round injection budget (unit cap
        # 8), not with the potentially larger fast round-0 query cap.
        single_round_block = render_search_results(
            search_results,
            max_total_tokens=injection_block_token_limit(MAX_WINDOW_SEARCH_QUERIES),
            count_tokens=token_counter.count_text,
        )
        search_results_text = single_round_block.text
        search_render_report = single_round_block.report()
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="research_search_results",
            task_id=task_id,
            payload={
                "queries": list(round1_result.search_queries),
                "multi_round": multi_round,
                "executed": [] if multi_round else search_results_metadata(search_results),
                "search_loop": (
                    {
                        "degraded": search_loop_metadata.get("degraded"),
                        "search_rounds_executed": search_loop_metadata.get(
                            "search_rounds_executed"
                        ),
                        "executed_queries": search_loop_metadata.get(
                            "executed_queries", []
                        ),
                    }
                    if multi_round
                    else {}
                ),
                "rendered_tokens": token_counter.count_text(search_results_text),
                "render_report": search_render_report,
            },
        )

    if collect_task_feedback and round1_result.task_update_feedback and task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="research_task_feedback",
            task_id=task_id,
            payload={
                "source": "fast_round1",
                "feedback": round1_result.task_update_feedback,
            },
        )

    token_report = {
        "phase": "research",
        "rows": token_rows,
        "totals": sum_token_distributions(row["tokens"] for row in token_rows),
    }
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="token_distribution_report",
            task_id=task_id,
            payload=token_report,
        )

    payload = {
        "mode": "fast",
        "prompt_version": PROMPT_VERSION,
        # Fast mode has no research round 2, but the post-task knowledge
        # update reads its <general_context> from this pack — carry the fast
        # research substance (analysis notes + search/evidence conclusions)
        # so that call can reuse what this run already established instead of
        # re-guessing proper nouns without evidence.
        "context_pack": {
            "general_context": {
                key: value
                for key, value in (
                    ("analysis_notes", round1_result.analysis_notes),
                    ("search_results", search_results_text),
                )
                if value
            },
            "window_contexts": {},
        },
        "fast": {
            "analysis_notes": round1_result.analysis_notes,
            "requested_entries": list(round1_result.requested_entries),
            "keep_entries": list(round1_result.keep_entries),
            "search_queries": list(round1_result.search_queries),
            "research_contract": round1_result.research_contract,
            "task_update_feedback": round1_result.task_update_feedback,
            "entry_details_text": entry_details_text,
            "search_results_text": search_results_text,
            "evidence_pack_mode": bool(multi_round),
        },
        "search_results": search_results_metadata(search_results),
        "search_loop": search_loop_metadata,
        "injected_entries": sorted(entry_details),
        "missing_entries": missing_entries,
        "ignored_keep_entries": ignored_keep_entries,
        "dropped_entry_keys": dropped_entry_keys,
        "token_report": token_report,
    }
    return _result_from_payload(payload), file_ref


def acquire_fast_context(
    *,
    context_path: str | Path,
    context_file: str | Path | None = None,
    **session_kwargs: Any,
) -> tuple[FastSessionResult, UploadedFileRef | None, bool]:
    """Reuse a persisted fast context or run the session and persist it.

    Returns ``(result, file_ref, reused)``; ``file_ref`` is only set on a
    fresh run (reuse lets the window executor cut/upload the clip itself).
    """

    if context_file:
        return load_fast_context(context_file), None, True
    expected_planning = _fast_context_planning_metadata(session_kwargs)
    context_path = Path(context_path)
    if context_path.exists():
        # An explicit --context-file still overrides this guard deliberately.
        data = json.loads(context_path.read_text(encoding="utf-8"))
        if data.get("planning") == expected_planning:
            return load_fast_context(context_path), None, True
        print(
            f"Warning: {context_path} was saved under different fast-session "
            "inputs; re-running the fast session.",
            file=sys.stderr,
        )
    result, file_ref = run_fast_session(**session_kwargs)
    result.payload["planning"] = expected_planning
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(
        json.dumps(result.payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result, file_ref, False
