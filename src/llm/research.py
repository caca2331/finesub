"""Two-round background research session feeding correction windows.

Round 1 has no web access: it picks knowledge entries and emits search queries.
The harness runs those queries through the local search agent (Tavily pool then
DuckDuckGo) and round 2 gets the rendered results injected; no further querying
is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .client import (
    GeminiPromptBlockedError,
    LiteLLMRoleClient,
    extract_finish_reason,
    extract_token_distribution,
    is_likely_output_limited,
    is_prompt_blocked,
    sum_token_distributions,
    validation_retry_sampling_kwargs,
)
from .content_filter import (
    load_content_filter_blacklist,
    run_injection_ladder,
    split_rendered_search_block,
)
from .exchange_metadata import infer_session_name, llm_exchange_metadata, research_input_components
from .config import (
    ANALYSIS_NOTES_MAX_TOKENS,
    DEFAULT_LIMITS,
    DEFAULT_RESEARCH_SEARCH_QUERIES,
    DEFAULT_RESEARCH_SEARCH_ROUNDS,
    INJECTION_SECTION_MAX_TOKENS,
    KB_PREINJECT_MAX_ENTRIES,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    LLMRole,
    ModelLimits,
    SESSION_OUTPUT_MAX_TOKENS,
    TASK_FEEDBACK_MAX_TOKENS,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    followup_search_query_limit,
    injection_block_token_limit,
    research_search_query_limit,
)
from .audio_clips import probe_audio_duration
from .chunking import (
    SubtitleSegment,
    SubtitleWindow,
    load_segments_from_stable_json,
    plan_correction_windows,
)
from .exchange_log import ExchangeLogger
from .injection_budget import render_knowledge_entries_block
from .knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    load_entry_texts,
    load_index_text,
    load_preinjected_entries,
)
from .output_tags import (
    extract_single_tag_block,
    parse_guided_line_items,
    parse_json_tag_block,
    parse_line_items,
)
from .profiles import DEFAULT_PROFILE, TranslationProfile
from .prompts import (
    PROMPT_VERSION,
    ContextPack,
    build_research_round1_messages,
    build_research_round2_messages,
)
from .search_loop import run_search_loop
from .session_checkpoint import SessionCheckpointStore, session_input_hash
from .token_budget import default_token_counter, TokenCounter
from .token_truncate import cap_tokens
from .web_search import (
    EXTRA_INFO_URL_EXTRACT_LIMIT,
    ExtractRequest,
    QueryExtractResult,
    QuerySearchResult,
    SearchRequest,
    WebSearchClient,
    extract_results_metadata,
    extract_urls_from_text,
    render_extract_results,
    render_search_results,
    search_results_metadata,
)


@dataclass(frozen=True)
class ResearchRound1Result:
    requested_entries: tuple[str, ...] = ()
    keep_entries: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    analysis_notes: str = ""
    # Raw <research_contract> JSON body (multi-round search only; "" otherwise).
    research_contract: str = ""
    # Raw <task_update_feedback> body (fast round 1 with collection only).
    task_update_feedback: str = ""


def render_research_transcript(
    segments: Sequence[SubtitleSegment],
    windows: Sequence[SubtitleWindow],
) -> str:
    """Render ``id|text`` lines with ``--- window N ---`` markers at window starts.

    Overlapping segments are rendered once, under the window that introduced them.
    """

    printed: set[str] = set()
    lines: List[str] = []
    for window in windows:
        new_segments = [seg for seg in window.segments if seg.id not in printed]
        if not new_segments:
            continue
        lines.append(f"--- window {window.chunk_id} ---")
        for segment in new_segments:
            text = (segment.text or "").replace("\r\n", "\n").replace("\r", "\n")
            lines.append(f"{segment.id}|{text.replace(chr(10), ' ')}")
            printed.add(segment.id)
    return "\n".join(lines) + ("\n" if lines else "")


def _extract_optional_block(
    text: str,
    tag: str,
    *,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Best-effort optional tag extraction: missing/duplicated yields ''."""

    try:
        body = extract_single_tag_block(text, tag, required=False)
    except ValueError:
        return ""
    body = body.strip()
    if max_tokens is None:
        return body
    return cap_tokens(body, max_tokens, count_tokens)


def parse_round1_output(
    text: str,
    *,
    expect_contract: bool = False,
    count_tokens: Callable[[str], int] | None = None,
) -> ResearchRound1Result:
    entries_body = extract_single_tag_block(text, "requested_entries")
    keep_body = extract_single_tag_block(text, "keep_entries")
    queries_body = extract_single_tag_block(text, "search_queries")
    contract = ""
    if expect_contract:
        # Required in multi-round mode: a missing contract triggers the normal
        # parse retry; the loop degrades gracefully later if it stays absent.
        contract = extract_single_tag_block(text, "research_contract").strip()
    return ResearchRound1Result(
        requested_entries=tuple(parse_line_items(entries_body)),
        keep_entries=tuple(parse_line_items(keep_body)),
        search_queries=tuple(parse_line_items(queries_body)),
        analysis_notes=_extract_optional_block(
            text,
            "analysis_notes",
            max_tokens=ANALYSIS_NOTES_MAX_TOKENS,
            count_tokens=count_tokens,
        ),
        research_contract=contract,
    )


def resolve_round1_entries(
    knowledge_root: str | Path,
    *,
    requested_names: Sequence[str],
    keep_names: Sequence[str],
    visible_keep_keys: Sequence[str],
    max_requested_entries: int = KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    max_keep_entries: int = KB_TRANSFER_MAX_ENTRIES,
    max_total_entries: int = KB_WINDOW_TOTAL_ENTRIES,
) -> tuple[dict[str, str], list[str], list[str], list[str]]:
    """Resolve R1 request/keep channels into one keep-first entry set.

    ``requested_names`` may name any indexed entry. ``keep_names`` may only
    resolve to an entry that was actually visible in R1's preinjection block
    (fully included or truncated, never wholly dropped). Canonical keys are
    deduped within each channel, capped independently, then merged keep-first
    under the shared total cap.

    Returns ``(selected, missing_requests, ignored_keeps, dropped_keys)``.
    """

    request_cap = max(0, int(max_requested_entries))
    keep_cap = max(0, int(max_keep_entries))
    total_cap = max(0, int(max_total_entries))
    visible = set(visible_keep_keys)
    requested: dict[str, str] = {}
    missing_requests: list[str] = []
    for name in requested_names:
        found, missing = load_entry_texts(knowledge_root, [name])
        missing_requests.extend(missing)
        for key, body in found.items():
            requested.setdefault(key, body)

    kept: dict[str, str] = {}
    ignored_keeps: list[str] = []
    for name in keep_names:
        found, missing = load_entry_texts(knowledge_root, [name])
        if missing:
            ignored_keeps.append(name)
            continue
        key, body = next(iter(found.items()))
        if key not in visible:
            ignored_keeps.append(name)
            continue
        kept.setdefault(key, body)

    capped_kept = dict(list(kept.items())[:keep_cap])
    capped_requested = dict(list(requested.items())[:request_cap])

    merged = dict(capped_kept)
    for key, body in capped_requested.items():
        merged.setdefault(key, body)
    selected = dict(list(merged.items())[:total_cap])
    selected_keys = set(selected)
    dropped_keys = [
        key
        for key in dict.fromkeys([*kept, *requested])
        if key not in selected_keys
    ]
    return selected, missing_requests, ignored_keeps, dropped_keys


def parse_round2_output(text: str) -> ContextPack:
    return ContextPack.from_dict(parse_json_tag_block(text, "context_pack"))


def extract_round_task_feedback(
    text: str,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Best-effort ``<task_update_feedback>`` body (capped); '' on any issue.

    Feedback is advisory: a missing or malformed block never retries the
    round — the context pack (or fast round-1 products) stays the hard output.
    """

    return _extract_optional_block(
        text,
        "task_update_feedback",
        max_tokens=TASK_FEEDBACK_MAX_TOKENS,
        count_tokens=count_tokens,
    )


def check_research_input_limit(
    messages: List[Dict[str, Any]],
    *,
    round_name: str,
    limits: ModelLimits = DEFAULT_LIMITS,
    counter: TokenCounter | None = None,
) -> int:
    counter = counter or default_token_counter()
    tokens = counter.count_texts(str(message.get("content", "")) for message in messages)
    if tokens > limits.prompt_input_limit:
        raise ValueError(
            f"Research {round_name} input (~{tokens} tokens) exceeds the prompt input "
            f"limit {limits.prompt_input_limit}. Split the audio into shorter clips first."
        )
    return tokens


def _render_note_url_extracts(
    extra_info: str,
    search_client: WebSearchClient,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, list[str], list[QueryExtractResult]]:
    """Extract up to eight deduped URLs from ``extra_info`` and deep-fetch them."""

    urls = extract_urls_from_text(extra_info)
    if not urls:
        return "", [], []
    results = search_client.extract_many([ExtractRequest(url=url) for url in urls])
    rendered = render_extract_results(
        results,
        max_total_tokens=injection_block_token_limit(EXTRA_INFO_URL_EXTRACT_LIMIT),
        count_tokens=count_tokens,
    ).text
    if rendered:
        rendered = f"<search_results>\n{rendered}\n</search_results>"
    return rendered, urls, results


def render_preinjected_entries(
    knowledge_root: str | Path,
    text: str,
    *,
    count_tokens: Callable[[str], int],
    max_entries: int = KB_PREINJECT_MAX_ENTRIES,
) -> tuple[str, Dict[str, Any]]:
    """Budget-rendered knowledge entries matched from note keywords, + report.

    Used by research round 1, fast round 1, and (on the text route, which runs
    no research) the correction windows' ``entry_details`` injection.
    """

    entries, matches = load_preinjected_entries(
        knowledge_root, text, max_entries=max_entries
    )
    report: Dict[str, Any] = {"matches": [match.to_dict() for match in matches]}
    if not entries:
        return "", report
    block = render_knowledge_entries_block(
        entries,
        count_tokens=count_tokens,
        entry_limit=INJECTION_SECTION_MAX_TOKENS,
        block_limit=injection_block_token_limit(max_entries),
    )
    report.update(block.report())
    return block.text, report


def _dump_research_round_input(
    task_artifact_dir: str | Path,
    round_name: str,
    payload: Mapping[str, Any],
) -> None:
    """Persist the exact semantic builder inputs for session replay."""

    path = Path(task_artifact_dir) / f"research-{round_name}-input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_research(
    *,
    transcript: str,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    client: LiteLLMRoleClient | None = None,
    search_client: WebSearchClient | None = None,
    enable_web_search: bool = True,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    max_parse_retries: int = 1,
    max_search_queries: int = DEFAULT_RESEARCH_SEARCH_QUERIES,
    search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    token_counter: TokenCounter | None = None,
    collect_task_feedback: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run both research rounds and return the research context payload.

    With ``search_rounds > 1`` (and web search enabled), round 1 emits a
    Research Contract plus round-0 queries, the multi-round search loop runs
    up to ``search_rounds`` total search rounds, and round 2 receives the
    resulting Evidence Pack instead of raw search results. ``search_rounds=1``
    keeps the legacy single-round behavior.

    ``collect_task_feedback`` asks round 2 for a trailing
    ``<task_update_feedback>`` block (schema v2) and persists it as a
    ``research_task_feedback`` artifact for the unified knowledge update.

    The payload contains the parsed ``context_pack`` plus round outputs and is
    what gets persisted as ``research-context.json``.
    """

    client = client or LiteLLMRoleClient(test_profile=test_profile)
    token_counter = token_counter or default_token_counter()
    streamer_index = load_index_text(knowledge_root, "streamer")
    common_index = load_index_text(knowledge_root, "common")
    token_rows: List[Dict[str, Any]] = []
    exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)
    multi_round = enable_web_search and int(search_rounds) > 1

    note_url_extracts = ""
    note_extract_urls: list[str] = []
    if extra_info.strip() and enable_web_search:
        search_client = search_client or WebSearchClient()
        note_url_extracts, note_extract_urls, note_extract_results = _render_note_url_extracts(
            extra_info, search_client, count_tokens=token_counter.count_text
        )
        if task_artifact_dir and note_extract_urls:
            append_task_artifact(
                task_artifact_dir,
                kind="api_call",
                task_id=task_id,
                payload={
                    "category": "web_extract",
                    "source": "extra_info_urls",
                    "urls": note_extract_urls,
                    "executed": extract_results_metadata(note_extract_results),
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
                payload={"source": "research_round1", **preinjection_report},
            )

    content_filter_blacklist = load_content_filter_blacklist(task_artifact_dir)

    def _round1_call(extracts_text: str):
        messages = build_research_round1_messages(
            transcript=transcript,
            extra_info=extra_info,
            note_url_extracts=extracts_text,
            streamer_index=streamer_index,
            common_index=common_index,
            preinjected_entries=preinjected_entries_text,
            max_search_queries=max_search_queries,
            use_search_contract=multi_round,
        )
        check_research_input_limit(messages, round_name="round 1", counter=token_counter)
        if task_artifact_dir:
            _dump_research_round_input(
                task_artifact_dir,
                "round1",
                {
                    "transcript": transcript,
                    "extra_info": extra_info,
                    "note_url_extracts": extracts_text,
                    "streamer_index": streamer_index,
                    "common_index": common_index,
                    "preinjected_entries": preinjected_entries_text,
                    "max_search_queries": max_search_queries,
                    "use_search_contract": multi_round,
                },
            )
        return _call_and_parse(
            client,
            messages,
            parser=lambda text: parse_round1_output(
                text, expect_contract=multi_round, count_tokens=token_counter.count_text
            ),
            round_name="round 1",
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind="research_round1_response",
            max_parse_retries=max_parse_retries,
            token_rows=token_rows,
            exchange_logger=exchange_logger,
            token_counter=token_counter,
            input_components_kwargs={
                "transcript": transcript,
                "note_url_extracts": extracts_text,
            },
            checkpoint_store=checkpoint_store,
            checkpoint_session="research-r1",
            checkpoint_key="main",
        )

    note_extract_block = split_rendered_search_block(note_url_extracts)
    round1_outcome = run_injection_ladder(
        block=note_extract_block,
        call=_round1_call,
        stage="research_round1",
        blocked_exception=GeminiPromptBlockedError,
        blacklist=content_filter_blacklist,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
        # No units → one same-prompt retry is still worth trying (filter
        # can be flaky); with units the ladder skips it (deterministic).
        plain_retry=not note_extract_block.units,
    )
    if round1_outcome.level >= 0:
        print(
            "Warning: research round 1 prompt was blocked by the content "
            f"filter; recovered at ladder level {round1_outcome.level} "
            f"(dropped {len(round1_outcome.dropped_units)} injection unit(s)).",
            file=sys.stderr,
        )
    round1_result = round1_outcome.result

    # R1 has two distinct channels: request new indexed entries, or keep an
    # entry that was actually visible in <preinjected_entries>. Each channel
    # has an independent 8-entry cap; together they share a 12-entry/token
    # budget, with keep winning after canonical key resolution. Loop rounds
    # see the selected entries read-only; round 2 receives the same set.
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
    entry_details_text = ""
    entry_render_report: Dict[str, Any] = {}
    if entry_details:
        entry_block = render_knowledge_entries_block(
            entry_details,
            count_tokens=token_counter.count_text,
            entry_limit=INJECTION_SECTION_MAX_TOKENS,
            block_limit=injection_block_token_limit(KB_WINDOW_TOTAL_ENTRIES),
        )
        entry_details_text = entry_block.text
        entry_render_report = entry_block.report()

    search_results: List[QuerySearchResult] = []
    search_loop_metadata: Dict[str, Any] = {}
    search_results_text = ""
    search_render_report: Dict[str, Any] = {}
    source_results_text = ""
    if multi_round and (round1_result.search_queries or round1_result.research_contract):
        search_client = search_client or WebSearchClient()
        background_parts = [part for part in (extra_info.strip(), round1_result.analysis_notes) if part]
        loop_result = run_search_loop(
            contract_body=round1_result.research_contract,
            round0_queries=round1_result.search_queries,
            client=client,
            search_client=search_client,
            background="\n\n".join(background_parts),
            max_rounds=int(search_rounds),
            round0_query_cap=max_search_queries,
            followup_query_cap=followup_search_query_limit(max_search_queries),
            max_parse_retries=max_parse_retries,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind="search_loop_round",
            exchange_prefix="research-search-loop",
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
        source_results_text = loop_result.source_results_text
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
            search_requests, max_queries=max_search_queries
        )
        single_round_block = render_search_results(
            search_results,
            max_total_tokens=injection_block_token_limit(max_search_queries),
            count_tokens=token_counter.count_text,
        )
        search_results_text = single_round_block.text
        source_results_text = single_round_block.text
        search_render_report = single_round_block.report()
    if task_artifact_dir:
        # Per-query executed metadata for multi-round runs lives in the
        # per-round ``search_loop_round`` artifacts; this summary must not
        # duplicate it (the task report counts providers per artifact kind).
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

    def _round2_call(search_text: str):
        # Contaminated Evidence Pack is never re-injected; a rebuilt/reduced
        # injection is framed as raw search results.
        use_pack = (
            multi_round
            and bool(search_text.strip())
            and search_text.strip() == search_results_text.strip()
        )
        round2_messages = build_research_round2_messages(
            transcript=transcript,
            extra_info=extra_info,
            round1_notes=round1_result.analysis_notes,
            entry_details_text=entry_details_text,
            search_results=search_text,
            use_evidence_pack=use_pack,
            collect_task_feedback=collect_task_feedback,
        )
        check_research_input_limit(
            round2_messages, round_name="round 2", counter=token_counter
        )
        if task_artifact_dir:
            _dump_research_round_input(
                task_artifact_dir,
                "round2",
                {
                    "transcript": transcript,
                    "extra_info": extra_info,
                    "round1_notes": round1_result.analysis_notes,
                    "entry_details_text": entry_details_text,
                    "search_results": search_text,
                    "use_evidence_pack": use_pack,
                    "collect_task_feedback": collect_task_feedback,
                },
            )
        return _call_and_parse(
            client,
            round2_messages,
            parser=lambda text: (
                parse_round2_output(text),
                extract_round_task_feedback(
                    text, count_tokens=token_counter.count_text
                )
                if collect_task_feedback
                else "",
                parse_line_items(
                    extract_single_tag_block(text, "keep_entries", required=False)
                ),
            ),
            round_name="round 2",
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind="research_round2_response",
            max_parse_retries=max_parse_retries,
            token_rows=token_rows,
            exchange_logger=exchange_logger,
            token_counter=token_counter,
            input_components_kwargs={
                "transcript": transcript,
                "search_results": search_text,
            },
            checkpoint_store=checkpoint_store,
            checkpoint_session="research-r2",
            checkpoint_key="main",
        )

    # Evidence Pack is opaque (no per-URL surgery). On block, rebuild from the
    # persisted source units — never re-inject the contaminated pack text.
    if multi_round and search_results_text.strip():
        try:
            context_pack, round2_task_feedback, round2_keep_raw = _round2_call(
                search_results_text
            )
        except GeminiPromptBlockedError:
            print(
                "Warning: research round 2 prompt was blocked by the content "
                "filter with the Evidence Pack; rebuilding from source search "
                "units (contaminated pack discarded).",
                file=sys.stderr,
            )
            source_block = split_rendered_search_block(source_results_text)
            round2_outcome = run_injection_ladder(
                block=source_block,
                call=_round2_call,
                stage="research_round2",
                blocked_exception=GeminiPromptBlockedError,
                blacklist=content_filter_blacklist,
                task_artifact_dir=task_artifact_dir,
                task_id=task_id,
            )
            if round2_outcome.level >= 0:
                print(
                    "Warning: research round 2 recovered at ladder level "
                    f"{round2_outcome.level} (dropped "
                    f"{len(round2_outcome.dropped_units)} source unit(s)).",
                    file=sys.stderr,
                )
            context_pack, round2_task_feedback, round2_keep_raw = round2_outcome.result
    else:
        search_block = split_rendered_search_block(search_results_text)
        round2_outcome = run_injection_ladder(
            block=search_block,
            call=_round2_call,
            stage="research_round2",
            blocked_exception=GeminiPromptBlockedError,
            blacklist=content_filter_blacklist,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            plain_retry=not search_block.units,
        )
        if round2_outcome.level >= 0:
            print(
                "Warning: research round 2 prompt was blocked by the content "
                f"filter; recovered at ladder level {round2_outcome.level} "
                f"(dropped {len(round2_outcome.dropped_units)} injection unit(s)).",
                file=sys.stderr,
            )
        context_pack, round2_task_feedback, round2_keep_raw = round2_outcome.result

    if collect_task_feedback and round2_task_feedback and task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="research_task_feedback",
            task_id=task_id,
            payload={"source": "research_round2", "feedback": round2_task_feedback},
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

    # v17 pass-through: round 2 may keep up to KB_TRANSFER_MAX_ENTRIES of the
    # entries it was shown; they seed the first correction window's transfer
    # chain. Canonicalized against the actually-injected set.
    keep_found, _keep_missing = load_entry_texts(knowledge_root, round2_keep_raw)
    keep_entries = [key for key in keep_found if key in entry_details][
        :KB_TRANSFER_MAX_ENTRIES
    ]

    return {
        "context_pack": context_pack.to_dict(),
        "keep_entries": keep_entries,
        "task_update_feedback": round2_task_feedback,
        "round1": {
            "requested_entries": list(round1_result.requested_entries),
            "keep_entries": list(round1_result.keep_entries),
            "search_queries": list(round1_result.search_queries),
            "analysis_notes": round1_result.analysis_notes,
            "research_contract": round1_result.research_contract,
        },
        "search_results": search_results_metadata(search_results),
        "search_loop": search_loop_metadata,
        "injected_entries": sorted(entry_details),
        "missing_entries": missing_entries,
        "ignored_keep_entries": ignored_keep_entries,
        "dropped_entry_keys": dropped_entry_keys,
        "entry_render_report": entry_render_report,
        "token_report": token_report,
    }


def _call_and_parse(
    client: LiteLLMRoleClient,
    messages: List[Dict[str, Any]],
    *,
    parser,
    round_name: str,
    task_artifact_dir: str | Path | None,
    task_id: str,
    artifact_kind: str,
    max_parse_retries: int,
    token_rows: List[Dict[str, Any]] | None = None,
    exchange_logger: ExchangeLogger | None = None,
    token_counter: TokenCounter | None = None,
    input_components_kwargs: Dict[str, Any] | None = None,
    role: LLMRole = LLMRole.GENERAL_CAPABLE,
    file_ref: Any | None = None,
    session_prefix: str = "",
    checkpoint_store: SessionCheckpointStore | None = None,
    checkpoint_session: str = "",
    checkpoint_key: str = "main",
    checkpoint_extra_identity: Mapping[str, Any] | None = None,
):
    checkpoint_hash = ""
    if checkpoint_store is not None and checkpoint_session:
        checkpoint_hash = session_input_hash(
            messages,
            prompt_version=PROMPT_VERSION,
            call_config={
                "role": role.value,
                "max_tokens": SESSION_OUTPUT_MAX_TOKENS,
                "file_backed": file_ref is not None,
            },
            extra_identity=checkpoint_extra_identity,
        )
        cached = checkpoint_store.get(
            checkpoint_session, checkpoint_key, checkpoint_hash
        )
        if cached is not None:
            try:
                parsed = parser(cached.content)
            except (ValueError, json.JSONDecodeError):
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="session_checkpoint_invalid",
                        task_id=task_id,
                        payload={
                            "session": checkpoint_session,
                            "key": checkpoint_key,
                            "input_hash": checkpoint_hash,
                        },
                    )
            else:
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="session_checkpoint_replay",
                        task_id=task_id,
                        payload={
                            "session": checkpoint_session,
                            "key": checkpoint_key,
                            "input_hash": checkpoint_hash,
                        },
                    )
                return parsed

    last_error: Exception | None = None
    session_base = artifact_kind.removesuffix("_response")
    for attempt in range(max_parse_retries + 1):
        result = client.complete(
            role,
            messages,
            max_tokens=SESSION_OUTPUT_MAX_TOKENS,
            file_ref=file_ref,
            **validation_retry_sampling_kwargs(attempt),
        )
        output_limited = is_likely_output_limited(
            result.raw_response, max_tokens=SESSION_OUTPUT_MAX_TOKENS
        )
        finish_reason = extract_finish_reason(result.raw_response)
        prompt_blocked = is_prompt_blocked(result.content, result.raw_response)
        session_payload = {"attempt": attempt}
        if session_prefix:
            session_payload["session"] = f"{session_prefix}-attempt{attempt}"
        session = infer_session_name(artifact_kind, session_payload)
        input_components = research_input_components(
            counter=token_counter,
            messages=messages,
            **(input_components_kwargs or {}),
        )
        if token_rows is not None:
            token_rows.append(
                {
                    "call": session_base,
                    "attempt": attempt,
                    "model": result.model,
                    "tokens": extract_token_distribution(result.raw_response),
                }
            )
        try:
            parsed = parser(result.content or "")
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            parse_error = str(exc)
        else:
            parse_error = ""
        if prompt_blocked:
            parse_error = (
                f"prompt blocked by content filter (finish_reason={finish_reason})"
            )
        if task_artifact_dir:
            append_task_artifact(
                task_artifact_dir,
                kind=artifact_kind,
                task_id=task_id,
                payload={
                    "session": session,
                    "attempt": attempt,
                    "model": result.model,
                    "fallback_used": result.fallback_used,
                    "usage": extract_token_distribution(result.raw_response),
                    "input_components": input_components,
                    "output_limited": output_limited,
                    "finish_reason": finish_reason,
                    "parse_error": parse_error,
                    "response_content": result.content,
                },
            )
        if exchange_logger:
            exchange_logger.log(
                session,
                messages=messages,
                response_text=result.content,
                metadata=llm_exchange_metadata(
                    result,
                    session=session,
                    input_components=input_components,
                    attempt=attempt,
                    output_limited=output_limited,
                    finish_reason=finish_reason,
                    **({"parse_error": parse_error} if parse_error else {}),
                ),
            )
        if prompt_blocked:
            # Deterministic for the exact prompt — retrying unchanged wastes
            # quota. Callers may rebuild without optional injected blocks.
            raise GeminiPromptBlockedError(
                f"Research {round_name} prompt was blocked by the content "
                f"filter (finish_reason={finish_reason})."
            )
        if not parse_error:
            if checkpoint_store is not None and checkpoint_hash:
                checkpoint_store.commit(
                    session=checkpoint_session,
                    key=checkpoint_key,
                    input_hash=checkpoint_hash,
                    content=result.content or "",
                    metadata={
                        "model": result.model,
                        "fallback_used": result.fallback_used,
                        "role": role.value,
                    },
                )
            return parsed
        if file_ref is not None and not (result.content or "").strip():
            # A media-backed call that returns literally nothing is almost
            # always transient file readiness (upload probe should prevent it,
            # but keep a backstop) — give the backend time before retrying.
            time.sleep(min(30.0, 10.0 * (attempt + 1)))
    raise RuntimeError(
        f"Research {round_name} output could not be parsed after "
        f"{max_parse_retries + 1} attempts: {last_error}"
    )


def load_research_context(path: str | Path) -> ContextPack:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if "context_pack" in data:
        return ContextPack.from_dict(data["context_pack"])
    return ContextPack.from_dict(data)


# ---------------------------------------------------------------------------
# Research acquisition stage (plans windows, runs research, persists context)


def _short_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()[:16]


def research_knowledge_inputs_hash(
    knowledge_root: str | Path,
    extra_info: str,
) -> str:
    """Hash the knowledge text that can enter research before web search."""

    preinjected, _matches = load_preinjected_entries(knowledge_root, extra_info)
    payload = {
        "streamer_index": load_index_text(knowledge_root, "streamer"),
        "common_index": load_index_text(knowledge_root, "common"),
        "preinjected_entries": preinjected,
    }
    return _short_sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def planning_metadata(
    profile: TranslationProfile,
    *,
    stable_json: str | Path,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    enable_web_search: bool = True,
    search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    collect_task_feedback: bool = False,
    audio_duration: float | None = None,
) -> dict:
    """All inputs that make a persisted research context safe to reuse."""

    source_path = Path(stable_json).expanduser()
    effective_rounds = int(search_rounds) if enable_web_search else 0
    return {
        "prompt_version": PROMPT_VERSION,
        "context_reserve_tokens": WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
        "profile_id": profile.profile_id,
        "output_scale": profile.output_scale,
        "stable_json_hash": _short_sha256(source_path.read_bytes()),
        "extra_info_hash": _short_sha256((extra_info or "").encode("utf-8")),
        "knowledge_inputs_hash": research_knowledge_inputs_hash(
            knowledge_root, extra_info
        ),
        "enable_web_search": bool(enable_web_search),
        "search_rounds": effective_rounds,
        "collect_task_feedback": bool(collect_task_feedback),
        "audio_duration": None
        if audio_duration is None
        else round(float(audio_duration), 3),
    }


def run_research_stage(
    *,
    stable_json: str | Path,
    context_path: str | Path,
    audio_path: str | Path | None = None,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    enable_web_search: bool = True,
    search_rounds: int,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    token_counter: TokenCounter | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    collect_task_feedback: bool = False,
    resume: bool = True,
) -> ContextPack:
    """Plan the correction windows, run both research rounds, persist the
    research context JSON next to the output, and return the ContextPack.
    Both the CLI and run_full_correction go through here."""

    token_counter = token_counter or default_token_counter()
    segments = load_segments_from_stable_json(stable_json)
    plan_report: dict = {}
    # `profile` must match the correction side's planning call exactly, or the
    # two plans disagree on window ids and window_contexts silently misalign.
    audio_duration = probe_audio_duration(audio_path) if audio_path else None
    windows = plan_correction_windows(
        segments,
        context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
        counter=token_counter,
        audio_duration=audio_duration,
        profile=profile,
        report_sink=plan_report,
    )
    if task_artifact_dir and plan_report.get("replan_attempts"):
        append_task_artifact(
            task_artifact_dir,
            kind="window_plan_report",
            task_id=task_id,
            payload={"phase": "research", **plan_report},
        )
    research_payload = run_research(
        transcript=render_research_transcript(segments, windows),
        extra_info=extra_info,
        knowledge_root=knowledge_root,
        enable_web_search=enable_web_search,
        test_profile=test_profile,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
        max_search_queries=research_search_query_limit(len(segments)),
        search_rounds=search_rounds,
        token_counter=token_counter,
        collect_task_feedback=collect_task_feedback,
        resume=resume,
    )
    research_payload["planning"] = planning_metadata(
        profile,
        stable_json=stable_json,
        extra_info=extra_info,
        knowledge_root=knowledge_root,
        enable_web_search=enable_web_search,
        search_rounds=search_rounds,
        collect_task_feedback=collect_task_feedback,
        audio_duration=audio_duration,
    )
    context_file = Path(context_path)
    context_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.write_text(
        json.dumps(research_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ContextPack.from_dict(research_payload["context_pack"])
