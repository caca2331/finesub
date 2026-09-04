"""Two-round background research session feeding correction windows.

Round 1 has no web access: it picks knowledge entries and emits search queries.
The harness runs those queries through the local search agent (Exa, Gemma4,
Tavily) and round 2 gets the rendered results injected; no further querying
is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .client import (
    GeminiPromptBlockedError,
    RoleClient,
    extract_finish_reason,
    extract_token_distribution,
    is_likely_output_limited,
    is_prompt_blocked,
    sum_token_distributions,
    validation_retry_sampling_kwargs,
)
from .content_filter import (
    report_ladder_outcome,
    load_content_filter_blacklist,
    run_injection_ladder,
    split_rendered_search_block,
)
from .exchange_metadata import infer_session_name, llm_exchange_metadata, research_input_components
from .routing.config import (
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
    effective_window_subtitle_cap,
    TASK_FEEDBACK_MAX_TOKENS,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    followup_search_query_limit,
    injection_block_token_limit,
    research_search_query_limit,
)
from finesub.media.clips import probe_audio_duration
from finesub.reporting import current_reporter
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
    load_index_entries,
    load_index_text,
    load_preinjected_entries,
    match_terms,
    render_term_matches,
)
from .output_tags import (
    extract_single_tag_block,
    parse_guided_line_items,
    parse_json_tag_block,
    parse_line_items,
)
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile
from .prompts import (
    PROMPT_VERSION,
    ContextPack,
    build_research_round1_messages,
    build_research_round2_messages,
    r1_request_cap,
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
    expect_queries: bool = True,
    expect_entries: bool = True,
    count_tokens: Callable[[str], int] | None = None,
) -> ResearchRound1Result:
    """Parse R1's reply, requiring only the blocks this vector asked for.

    ``expect_queries`` / ``expect_entries`` mirror the prompt's own halves
    (docs/llm_harness_behavior.md) -- a block the prompt never described must not be demanded
    back, or every ``retrieval=none`` run would burn its parse retries.
    """

    # Not routed through ``research_round1_contract``: that contract also
    # demands ``<reasoning>``, which is a *soft* requirement in production (a
    # missing block never fails a round). The contract's job is the replay
    # validators; the parser only enforces the blocks it has to read.
    entries_body = extract_single_tag_block(
        text, "requested_entries", required=expect_entries
    )
    keep_body = extract_single_tag_block(text, "keep_entries", required=expect_entries)
    queries_body = extract_single_tag_block(
        text, "search_queries", required=expect_queries
    )
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
    # Sub-entry hits, addressed by parent and section. A separate pass because
    # it answers a different question: `load_preinjected_entries` asks "was
    # this ENTRY named", which on transcript text almost never happens -- the
    # shadow ledger measured 4 subject hits against 347 term ones, 87% of them
    # under a subject nobody mentioned. Entry names are game titles and
    # streamer names; what people say are the characters inside them.
    term_matches = match_terms(
        knowledge_root,
        text,
        exclude_subjects=[match.key for match in matches],
    )
    if term_matches:
        report["term_matches"] = [match.to_dict() for match in term_matches]
    term_block = render_term_matches(term_matches)
    if not entries:
        return term_block, report
    block = render_knowledge_entries_block(
        entries,
        count_tokens=count_tokens,
        entry_limit=INJECTION_SECTION_MAX_TOKENS,
        block_limit=injection_block_token_limit(max_entries),
    )
    report.update(block.report())
    text_out = "\n".join(part for part in (block.text, term_block) if part)
    return text_out, report


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
    client: RoleClient | None = None,
    search_client: WebSearchClient | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    knowledge_enabled: bool = True,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    max_parse_retries: int = 5,
    max_search_queries: int = DEFAULT_RESEARCH_SEARCH_QUERIES,
    search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    token_counter: TokenCounter | None = None,
    collect_task_feedback: bool = False,
    resume: bool = True,
    chunk_label: str = "",
    exchange_logger: "ExchangeLogger | None" = None,
    checkpoint_store: "SessionCheckpointStore | None" = None,
) -> Dict[str, Any]:
    """Run the research rounds this switch vector calls for (docs/llm_harness_behavior.md).

    ``chunk_label`` names this call's slice of a chunked ultra-long-material
    run (plan W7/P8): it keys the r1/r2 checkpoints, the round-input dumps and
    the search-loop's exchange prefix so chunks never overwrite each other's
    resume state. Empty = the whole-transcript run, whose identities are
    unchanged. Concurrent chunks must share ONE ``exchange_logger`` and ONE
    ``checkpoint_store`` (both internally locked): separate instances race the
    exchange numbering and interleave the checkpoint JSONL's appends.

    The skeleton is always *r1 asks -> harness fetches -> r2 digests*, but each
    half belongs to a different axis, so the vector decides which rounds run at
    all:

    - ``retrieval=local``  -- r1 emits queries + entry requests, the search loop
      runs, r2 digests the evidence pack and the entry texts.
    - ``retrieval=native`` -- r1 only picks entries (skipped without a readable
      knowledge base); r2 runs with the model's own search tool.
    - ``retrieval=none``   -- r1 only picks entries, and **r2 is skipped**: with
      no external input its context pack would be pure self-reasoning presented
      downstream as evidence, and what it could offer the correction window
      already has (full window text, preceding context, advice ledger).

    With ``search_rounds > 1`` under ``local``, round 1 emits a Research
    Contract plus round-0 queries, the multi-round search loop runs up to
    ``search_rounds`` total search rounds, and round 2 receives the resulting
    Evidence Pack instead of raw search results. ``search_rounds=1`` keeps the
    single-round behavior.

    ``collect_task_feedback`` asks round 2 for a trailing
    ``<task_update_feedback>`` block (schema v2) and persists it as a
    ``research_task_feedback`` artifact for the unified knowledge update.

    The payload contains the parsed ``context_pack`` plus round outputs and is
    what gets persisted as ``research-context.json``.
    """

    client = client or RoleClient(test_profile=test_profile)
    token_counter = token_counter or default_token_counter(
        execution_settings=getattr(client, "execution_settings", None)
    )
    checkpoint_key = chunk_label or "main"
    dump_suffix = f"-{chunk_label}" if chunk_label else ""
    streamer_index = load_index_text(knowledge_root, "streamer") if knowledge_enabled else ""
    common_index = load_index_text(knowledge_root, "common") if knowledge_enabled else ""
    token_rows: List[Dict[str, Any]] = []
    if exchange_logger is None:
        exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    if checkpoint_store is None:
        checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)

    # --- Round structure for this vector ---
    local_search = profile.external_injection
    # Entries are only worth asking for when there is an index to read them off.
    entries_available = bool(streamer_index.strip() or common_index.strip())
    run_round2 = profile.retrieval != "none"
    # r1's only job under none/native is picking entries, so no index means no r1.
    run_round1 = local_search or entries_available
    multi_round = local_search and int(search_rounds) > 1

    note_url_extracts = ""
    note_extract_urls: list[str] = []
    if extra_info.strip() and local_search:
        search_client = search_client or WebSearchClient(
            execution_settings=getattr(client, "execution_settings", None)
        )
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
    if extra_info.strip() and knowledge_enabled:
        preinjected_entries_text, preinjection_report = render_preinjected_entries(
            knowledge_root, extra_info, count_tokens=token_counter.count_text
        )
        # Either kind of hit counts: a window can be reached by a sub-entry
        # (term) match with no entry-level match at all -- measured, that is
        # 87% of term hits -- and gating the record on `matches` alone means
        # the prompt carried knowledge the report says nothing about.
        if task_artifact_dir and (
            preinjection_report["matches"] or preinjection_report.get("term_matches")
        ):
            append_task_artifact(
                task_artifact_dir,
                kind="knowledge_preinjection",
                task_id=task_id,
                payload={"source": "research_round1", **preinjection_report},
            )

    content_filter_blacklist = load_content_filter_blacklist(task_artifact_dir)
    # No per-window query round downstream means R1 is the session's only pick.
    round1_request_cap = r1_request_cap(downstream_can_request=local_search)

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
            emits_queries=local_search,
            emits_entries=entries_available,
            emits_notes=run_round2,
            max_requested_entries=round1_request_cap,
        )
        check_research_input_limit(messages, round_name="round 1", counter=token_counter)
        if task_artifact_dir:
            _dump_research_round_input(
                task_artifact_dir,
                f"round1{dump_suffix}",
                {
                    "transcript": transcript,
                    "extra_info": extra_info,
                    "note_url_extracts": extracts_text,
                    "streamer_index": streamer_index,
                    "common_index": common_index,
                    "preinjected_entries": preinjected_entries_text,
                    "max_search_queries": max_search_queries,
                    "use_search_contract": multi_round,
                    "emits_queries": local_search,
                    "emits_entries": entries_available,
                    "emits_notes": run_round2,
                    "max_requested_entries": round1_request_cap,
                },
            )
        return _call_and_parse(
            client,
            messages,
            parser=lambda text: parse_round1_output(
                text,
                expect_contract=multi_round,
                expect_queries=local_search,
                expect_entries=entries_available,
                count_tokens=token_counter.count_text,
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
            checkpoint_key=checkpoint_key,
            kb_read=entries_available,
        )

    if run_round1:
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
        report_ladder_outcome(round1_outcome, what="research round 1")
        round1_result = round1_outcome.result
    else:
        # Neither half applies: no local search agent to serve queries and no
        # index to pick entries off. The round would have nothing to ask for.
        round1_result = ResearchRound1Result()

    # R1 has two distinct channels: request new indexed entries, or keep an
    # entry that was actually visible in <preinjected_entries>. Each channel
    # has an independent cap; together they share a 12-entry/token budget, with
    # keep winning after canonical key resolution. Loop rounds see the selected
    # entries read-only; round 2 receives the same set.
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
        max_requested_entries=round1_request_cap,
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
        search_client = search_client or WebSearchClient(
            execution_settings=getattr(client, "execution_settings", None)
        )
        background_parts = [part for part in (extra_info.strip(), round1_result.analysis_notes) if part]
        loop_result = run_search_loop(
            contract_body=round1_result.research_contract,
            round0_queries=round1_result.search_queries,
            client=client,
            search_client=search_client,
            background="\n\n".join(background_parts),
            max_rounds=int(search_rounds),
            difficulty=profile.difficulty,
            round0_query_cap=max_search_queries,
            followup_query_cap=followup_search_query_limit(max_search_queries),
            max_parse_retries=max_parse_retries,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind="search_loop_round",
            exchange_prefix=f"research-search-loop{dump_suffix}",
            token_rows=token_rows,
            exchange_logger=exchange_logger,
            token_counter=token_counter,
            # None is the loop's own "no entry channel" signal; handing it a
            # root would let it load the indices the run is meant to ignore.
            knowledge_root=knowledge_root if knowledge_enabled else None,
            persistent_entries_text=entry_details_text,
            persistent_entry_keys=list(entry_details),
            persistent_requested_entry_names=round1_result.requested_entries,
            persistent_kept_entry_names=round1_result.keep_entries,
            content_filter_blacklist=content_filter_blacklist,
            resume=resume,
            checkpoint_store=checkpoint_store,
        )
        search_results_text = loop_result.evidence_pack
        source_results_text = loop_result.source_results_text
        search_loop_metadata = loop_result.to_metadata()
    elif local_search and round1_result.search_queries:
        search_client = search_client or WebSearchClient(
            execution_settings=getattr(client, "execution_settings", None)
        )
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
    if task_artifact_dir and local_search:
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

    # Pruning is only safe when something downstream can ask an entry back.
    # Without a per-window query round, a dropped entry is gone for the whole
    # run, so the harness transfers the full set instead of asking (docs/llm_harness_behavior.md).
    round2_emits_keep = local_search

    # The parsed tuple loses the call result; the sink keeps the last raw
    # response so the feedback artifact can carry the harness retrieval
    # ledger's URLs (report 2026-08-28 §2.3).
    round2_raw_responses: list[Any] = []

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
            native_search=profile.native_search,
            emits_keep=round2_emits_keep,
        )
        check_research_input_limit(
            round2_messages, round_name="round 2", counter=token_counter
        )
        if task_artifact_dir:
            _dump_research_round_input(
                task_artifact_dir,
                f"round2{dump_suffix}",
                {
                    "transcript": transcript,
                    "extra_info": extra_info,
                    "round1_notes": round1_result.analysis_notes,
                    "entry_details_text": entry_details_text,
                    "search_results": search_text,
                    "use_evidence_pack": use_pack,
                    "collect_task_feedback": collect_task_feedback,
                    "native_search": profile.native_search,
                    "emits_keep": round2_emits_keep,
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
            checkpoint_key=checkpoint_key,
            retrieval=profile.retrieval,
            kb_read=entries_available,
            raw_response_sink=round2_raw_responses,
        )

    # Evidence Pack is opaque (no per-URL surgery). On block, rebuild from the
    # persisted source units — never re-inject the contaminated pack text.
    if not run_round2:
        # retrieval=none: nothing was fetched, so R2 would be pure
        # self-reasoning served downstream as evidence. Entries still flow.
        context_pack = ContextPack()
        round2_task_feedback = ""
        round2_keep_raw: list[str] = []
    elif multi_round and search_results_text.strip():
        try:
            context_pack, round2_task_feedback, round2_keep_raw = _round2_call(
                search_results_text
            )
        except GeminiPromptBlockedError:
            current_reporter().warning(
                "content-filter-pack-discarded",
                "research round 2 prompt was blocked by the content filter "
                "with the Evidence Pack; rebuilding from source search units",
                impact="被污染的证据包整个丢弃",
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
            report_ladder_outcome(round2_outcome, what="research round 2")
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
        report_ladder_outcome(round2_outcome, what="research round 2")
        context_pack, round2_task_feedback, round2_keep_raw = round2_outcome.result

    if collect_task_feedback and round2_task_feedback and task_artifact_dir:
        from .knowledge.feedback import retrieval_urls_from_response

        append_task_artifact(
            task_artifact_dir,
            kind="research_task_feedback",
            task_id=task_id,
            payload={
                "source": "research_round2",
                "feedback": round2_task_feedback,
                "retrieval_urls": list(
                    retrieval_urls_from_response(
                        round2_raw_responses[-1] if round2_raw_responses else None
                    )
                ),
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

    # v17 pass-through: the entries seeding the first correction window's
    # transfer chain. With a query round downstream, R2 prunes to at most
    # KB_TRANSFER_MAX_ENTRIES of what it was shown (canonicalized against the
    # actually-injected set); without one, pruning would be permanent, so the
    # harness transfers everything under the shared window total (docs/llm_harness_behavior.md).
    if round2_emits_keep:
        keep_found, _keep_missing = load_entry_texts(knowledge_root, round2_keep_raw)
        keep_entries = [key for key in keep_found if key in entry_details][
            :KB_TRANSFER_MAX_ENTRIES
        ]
    else:
        keep_entries = sorted(entry_details)[:KB_WINDOW_TOTAL_ENTRIES]

    return {
        "context_pack": context_pack.to_dict(),
        "keep_entries": keep_entries,
        "rounds": {
            "round1": run_round1,
            "round2": run_round2,
            "local_search": local_search,
            "native_search": profile.native_search,
            "entries_available": entries_available,
        },
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
        # Whether R1's ceiling actually bound, recorded next to the ceiling that
        # was in force. The §9 recall question ("does a session-level pick miss
        # what per-window rounds would have caught") is only answerable on runs
        # where it did not bind -- otherwise the measurement reads the cap, not
        # R1's judgment -- and the cap has changed once already.
        "entry_selection": {
            "request_cap": round1_request_cap,
            "requested_count": len(round1_result.requested_entries),
            "hit_cap": len(round1_result.requested_entries) >= round1_request_cap,
            "dropped_by_cap": len(dropped_entry_keys),
            "indexed_entry_count": (
                count_indexed_entries(knowledge_root) if knowledge_enabled else 0
            ),
        },
        "entry_render_report": entry_render_report,
        "token_report": token_report,
    }


# Backends whose event stream carries no result URLs at all, so an unmarked
# `sources` list there really is uncorroborated. Everything else keeps a
# per-call URL ledger: Codex and Claude Code from their own tool events, Gemini
# REST `google_search` from the grounding metadata the client now parses. Only
# agy still reports queries without URLs, and even there the gate below is the
# ledger being empty -- a backend in this set that did report URLs is not
# marked.
_SOURCE_UNVERIFIABLE_BACKENDS = frozenset({"local_agent"})


def _mark_unverified_sources(parsed: Any, execution_attempts: Any) -> Any:
    """Flag a native context pack's `sources` when nothing can corroborate them.

    The native prompt asks the round to cite what it verified, and the model
    complies -- but on agy the reply comes back with site-level "sources" and
    the stream carries no URL at all, so nothing downstream can check them. In
    the 2026-08-15 A/B the one checkable claim carried that way was a wrong
    official title that the local arm got right with three specific URLs.

    Scoped to the backends that genuinely report nothing. Marking on "no
    `search_events` with URLs" alone caught the **shipped default's** native
    path too, which is served by the paid Gemini target's `google_search`
    (`test_llm_switch_matrix` asserts that route exists): Gemini does return
    grounding URIs, and since 2026-08-15 the client parses them into the same
    ledger -- so that path now exits at the URL check above and never reaches
    the mark, which is the correct outcome and no longer an accident.
    """

    if not isinstance(parsed, ContextPack):
        return parsed
    general = dict(parsed.general_context or {})
    sources = general.get("sources")
    if not isinstance(sources, list) or not sources:
        return parsed
    saw_unverifiable_backend = False
    for attempt in execution_attempts or ():
        if not isinstance(attempt, Mapping):
            continue
        if str(attempt.get("backend") or "") in _SOURCE_UNVERIFIABLE_BACKENDS:
            saw_unverifiable_backend = True
        for event in attempt.get("search_events") or ():
            if isinstance(event, Mapping) and event.get("urls"):
                return parsed
    if not saw_unverifiable_backend:
        return parsed
    general["sources"] = [
        {**source, "verified": False} if isinstance(source, Mapping) else source
        for source in sources
    ]
    general["sources_note"] = (
        "模型自述来源；本次应答的 agent 后端不回报检索结果 URL，harness 无法核对，"
        "不可当作可追溯出处使用。"
    )
    return dataclass_replace(parsed, general_context=general)


def _call_and_parse(
    client: RoleClient,
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
    retrieval: str = "none",
    # v2 cell routing (plan §6): research rounds default to their own task
    # group; the fast fusion round passes correction-mm/-text (or research
    # when text-only) explicitly.
    task_group: str = "research",
    difficulty: str = "",
    # §4.3 authorization matrix: the research/fast rounds that read the
    # knowledge base grant the read-only kb tools; judges and other sessions
    # leave it off and the tools never attach. ``kb_signal_window`` names the
    # window this call serves in the exposure ledger (fast is a per-window
    # consumer; research rounds are task-wide and leave it empty).
    kb_read: bool = False,
    kb_signal_window: str = "",
    # The parsed return value drops the LLMCallResult; a caller that needs the
    # raw response afterwards (retrieval-ledger URL backfill) passes a list to
    # collect the answering attempt's raw_response.
    raw_response_sink: list[Any] | None = None,
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
                # A native-search reply is not interchangeable with a plain one
                # for the same prompt: it carries grounded facts this checkpoint
                # would otherwise replay as if the tool had run.
                "native_search": retrieval == "native",
            },
            extra_identity=checkpoint_extra_identity,
            execution_identity_override=getattr(
                client, "execution_identity", None
            ),
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
        try:
            result = client.complete(
                role,
                messages,
                max_tokens=SESSION_OUTPUT_MAX_TOKENS,
                file_ref=file_ref,
                retrieval=retrieval,
                task_group=task_group,
                difficulty=difficulty,
                agent_task_extras=(
                    {
                        "kb_tools": "read",
                        "kb_signal_task": task_id,
                        **({"kb_signal_window": kb_signal_window} if kb_signal_window else {}),
                    }
                    if kb_read
                    else None
                ),
                **validation_retry_sampling_kwargs(attempt),
            )
        except Exception as exc:  # pragma: no cover - provider behavior
            if task_artifact_dir:
                append_task_artifact(
                    task_artifact_dir,
                    kind=f"{session_base}_call_error",
                    task_id=task_id,
                    payload={
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "api_attempts": list(
                            getattr(exc, "_harness_api_attempts", []) or []
                        ),
                        "execution_attempts": list(
                            getattr(exc, "_harness_execution_attempts", []) or []
                        ),
                        "route_decision": dict(
                            getattr(exc, "_harness_route_decision", {}) or {}
                        ),
                    },
                )
            raise
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
            if retrieval == "native":
                parsed = _mark_unverified_sources(
                    parsed, result.execution_attempts
                )
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
                    "api_attempts": list(result.api_attempts),
                    "execution_attempts": list(result.execution_attempts),
                    "route_decision": dict(result.route_decision),
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
            if raw_response_sink is not None:
                raw_response_sink.append(result.raw_response)
            # Gate D answer C (docs/llm_local_agent.md §7): a call that leaned
            # on implicit provider history is not replayable from its hash, so
            # it must not seed the L1 store. A later resume re-sends this one
            # call; nothing upstream is invalidated.
            if (
                checkpoint_store is not None
                and checkpoint_hash
                and getattr(result, "resumable", True)
            ):
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
    """Load a persisted context pack; refuse one whose notes lack ranges.

    Unbound notes cannot be injected, so serving such a pack would hand the
    correction stage a silently halved research product -- the general context
    only -- for work the user already paid quota for. Re-running the stage is
    the same trade every other reuse gate here makes: cost a stage, stay whole.
    """

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    pack = ContextPack.from_dict(data.get("context_pack", data))
    if pack.has_unbound_window_contexts:
        raise ValueError(
            "research window contexts carry no source-id ranges "
            f"({len(pack.unbound_window_contexts)} notes); the artifact predates "
            "interval addressing and cannot be injected"
        )
    return pack


# ---------------------------------------------------------------------------
# Research acquisition stage (plans windows, runs research, persists context)


def _short_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()[:16]


def stable_json_source_hash(stable_json: str | Path) -> str:
    """Identity of the persisted source, parsed rather than read as bytes.

    Reformatting the JSON or touching a field no stage reads is not a source
    change.  Execution choices (model, preset, policy, retrieval knobs) are
    deliberately absent: they change how unfinished work runs without making an
    already committed result unusable (docs/llm_local_agent.md §11).
    """

    segments = load_segments_from_stable_json(stable_json)
    payload = [
        [segment.id, segment.start, segment.end, segment.text]
        for segment in segments
    ]
    return _short_sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def plan_geometry_metadata(
    profile: TranslationProfile,
    *,
    audio_duration: float | None = None,
    max_window_subtitle_tokens: int | None = None,
    limits: ModelLimits | None = None,
) -> dict:
    """Audit metadata describing what placed the original window boundaries.

    Geometry is not a reuse gate: correction plans preserve boundary identity
    and refit pending leaves, while research notes are source-interval addressed.

    ``max_window_subtitle_tokens`` is resolved through the same ``limits`` the
    planner uses rather than recorded raw: ``None`` (unset, take the limits
    default) and ``0`` (cap disabled) plan different windows.
    """

    effective_limits = limits or DEFAULT_LIMITS
    return {
        "prompt_version": PROMPT_VERSION,
        "context_reserve_tokens": WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
        "geometry_profile_id": profile.geometry_id,
        "max_window_subtitle_tokens": effective_window_subtitle_cap(
            max_window_subtitle_tokens, effective_limits
        ),
        # The bound group's planning envelope (D13).
        "input_limit": effective_limits.prompt_input_limit,
        "output_limit": effective_limits.output_limit,
        "audio_duration": None
        if audio_duration is None
        else round(float(audio_duration), 3),
    }


def backup_unrecoverable_json(path: str | Path) -> Path:
    """Copy an incompatible JSON artifact aside before a caller replaces it."""

    source = Path(path)
    candidate = source.with_name(f"{source.name}.invalid")
    suffix = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.name}.invalid.{suffix}")
        suffix += 1
    shutil.copy2(source, candidate)
    return candidate


def count_indexed_entries(knowledge_root: str | Path) -> int:
    """How many entries the indices offer.

    Recorded alongside R1's request cap so a later reader can tell a genuine
    selection from "the whole knowledge base fit under the ceiling".
    """

    return sum(
        len(load_index_entries(knowledge_root, category))
        for category in ("streamer", "common")
    )


def knowledge_index_available(
    knowledge_root: str | Path, *, knowledge_enabled: bool = True
) -> bool:
    """Whether there is an index to pick entries off."""

    if not knowledge_enabled:
        return False
    return bool(
        load_index_text(knowledge_root, "streamer").strip()
        or load_index_text(knowledge_root, "common").strip()
    )


def research_stage_runs(
    profile: TranslationProfile,
    *,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
) -> bool:
    """Whether the research stage has any round to run for this vector.

    ``local``/``native`` always run r2 (evidence pack / native search). Under
    ``none`` only r1 can run, and only to pick entries -- with no index to read,
    the whole stage is skipped and the correction windows start cold.
    """

    if profile.retrieval != "none":
        return True
    return knowledge_index_available(
        knowledge_root, knowledge_enabled=knowledge_enabled
    )


def research_knowledge_inputs_hash(
    knowledge_root: str | Path,
    extra_info: str,
) -> str:
    """Hash the knowledge text that can enter research before web search.

    ⚠ It has to cover the sub-entry hits too. Those reach the prompt without
    any entry-level match (measured: 87% of term hits sit under a subject that
    was never named), so a hash over entries alone reports "the knowledge input
    did not change" while the injected text did -- which is the one thing this
    hash exists to prevent.
    """

    preinjected, matches = load_preinjected_entries(knowledge_root, extra_info)
    terms = match_terms(
        knowledge_root, extra_info, exclude_subjects=[match.key for match in matches]
    )
    payload = {
        "streamer_index": load_index_text(knowledge_root, "streamer"),
        "common_index": load_index_text(knowledge_root, "common"),
        "preinjected_entries": preinjected,
        "preinjected_terms": [match.to_dict() for match in terms],
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
    knowledge_enabled: bool = True,
    search_rounds: int = DEFAULT_RESEARCH_SEARCH_ROUNDS,
    collect_task_feedback: bool = False,
    audio_duration: float | None = None,
    max_window_subtitle_tokens: int | None = None,
    limits: ModelLimits | None = None,
    test_profile: bool = False,
    execution_identity_override: Mapping[str, Any] | None = None,
) -> dict:
    """Provenance for a persisted research context.

    The whole dict is recorded; :func:`research_reuse_key` selects the narrow
    semantic subset that gates reuse.

    ``knowledge_enabled`` is recorded explicitly rather than left to
    ``knowledge_inputs_hash``: with knowledge off the harness never reads the
    indices, so the hash of an unread base would match a run that did read it.
    Keeping the switch beside the hash keeps "not read" and "read an empty
    base" distinguishable in task reports.
    """

    from .routing.execution_policy import execution_identity

    source_path = Path(stable_json).expanduser()
    effective_rounds = int(search_rounds) if profile.external_injection else 0
    return {
        **plan_geometry_metadata(
            profile,
            audio_duration=audio_duration,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
            limits=limits,
        ),
        # Two of the three are gates; the difference matters.
        #
        # ``profile_id`` is the full vector and is audit-only (exempt): it
        # carries continuity and difficulty, which never reach the research
        # rounds. Gating on them would make "switch to intermediate and
        # continue" -- the documented way out of an exhausted free-tier capable
        # quota -- re-run both rounds, burning the very quota the switch exists
        # to stop burning.
        #
        # ``planning_profile_id`` is audit-only too: it still carries
        # ``correction_media`` (geometry) and ``planning_media`` (the *query*
        # round's clip), neither of which the research rounds read. The narrower
        # ``research_semantics_id`` is the gate.
        "profile_id": profile.profile_id,
        "planning_profile_id": dataclass_replace(
            profile, continuity="serial", difficulty="quality"
        ).profile_id,
        "research_semantics_id": profile.retrieval,
        "output_scale": profile.output_scale,
        "stable_json_hash": stable_json_source_hash(source_path),
        "extra_info_hash": _short_sha256((extra_info or "").encode("utf-8")),
        "knowledge_inputs_hash": research_knowledge_inputs_hash(
            knowledge_root, extra_info
        )
        if knowledge_enabled
        else "",
        "knowledge_enabled": bool(knowledge_enabled),
        "search_rounds": effective_rounds,
        "collect_task_feedback": bool(collect_task_feedback),
        # A gate at both L2 and L3: a smoke run and a real run must not mix
        # their artifacts in either direction. (Absent here until 2026-08-12,
        # which let a --test-profile context be reused by a real run.)
        "test_profile": bool(test_profile),
        "execution_identity": dict(
            execution_identity_override or execution_identity()
        ),
    }


# What a persisted research context is compared on (L3). Window geometry is
# deliberately absent: interval-addressed notes remap onto any later correction
# plan. What remains is the source, the output contract, and the three knobs a
# user turns *at this artifact* -- reusing it past those would ignore a request.
#
#   research_semantics_id  = retrieval. The only thing separating a context that
#                            never searched from a run asking for `native`:
#                            ``external_injection`` is true for `local` alone,
#                            so both leave ``search_rounds`` at 0.
RESEARCH_REUSE_GATES = (
    "stable_json_hash",
    "prompt_version",
    "extra_info_hash",
    "search_rounds",
    "test_profile",
    "research_semantics_id",
)

# Recorded for audit and for task reports, never a reason to throw a committed
# stage away. Split out from the gates so that adding a field to
# ``planning_metadata`` without classifying it is a test failure rather than a
# silent reuse of something it should have invalidated.
RESEARCH_REUSE_AUDIT_ONLY = (
    "execution_identity",
    "knowledge_inputs_hash",
    "knowledge_enabled",
    "collect_task_feedback",
    "profile_id",
    "planning_profile_id",
    "output_scale",
    # plan_geometry_metadata's payload
    "context_reserve_tokens",
    "geometry_profile_id",
    "max_window_subtitle_tokens",
    "input_limit",
    "output_limit",
    "audio_duration",
)


def research_reuse_key(planning: Mapping[str, Any]) -> dict:
    """The comparable part of a research context's planning metadata."""

    return {key: planning.get(key) for key in RESEARCH_REUSE_GATES}


# --- chunked research for ultra-long material (task-parallelism plan W7 / P8) ---

#: Transcript size past which the research phase splits into chunks, each
#: running the full r1 + search loop + r2. Owner decision 2026-08-30: default
#: ON at 100k tokens; ``[chunking] research_transcript_max_tokens`` overrides,
#: 0 disables (single-shot regardless of size). Below the gate the current
#: single-call path runs unchanged.
DEFAULT_RESEARCH_TRANSCRIPT_MAX_TOKENS = 100_000

#: The merged general block's own ceiling. Deliberately NOT
#: ``injection_block_token_limit`` -- that budget is sized for knowledge entry
#: dicts; this is model-authored context that every correction window carries.
RESEARCH_MERGED_GENERAL_MAX_TOKENS = 20_000

#: Fixed text, part of the deterministic assembly (no extra LLM merge round):
#: it must tell the correction model what to do when two chunks disagree.
MERGED_RESEARCH_GENERAL_NOTE = (
    "本次调查按素材分段独立进行（c01、c02……对应素材的先后区间），以下各键是各段的全局结论。"
    "若不同段对同一专名/称呼给出不同写法，优先采用证据更充分（有来源、出现次数多）的一段的结论，"
    "并在该行的 note 中标注存在分歧。"
)


def resolve_research_transcript_cap() -> int:
    """``[chunking] research_transcript_max_tokens`` (0 disables chunking)."""

    from .routing.api_keys import read_config

    data = read_config()
    section = data.get("chunking") if isinstance(data, Mapping) else None
    raw = section.get("research_transcript_max_tokens") if isinstance(section, Mapping) else None
    if raw is None:
        return DEFAULT_RESEARCH_TRANSCRIPT_MAX_TOKENS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "[chunking] research_transcript_max_tokens must be an integer"
        ) from None
    if value < 0:
        raise ValueError("[chunking] research_transcript_max_tokens must be >= 0")
    return value


def plan_research_chunks(
    windows: Sequence[SubtitleWindow],
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[list[SubtitleWindow]]:
    """Split the research windows into transcript chunks of whole windows.

    Greedy and deterministic: windows accumulate in order until the next one
    would push the chunk's rendered transcript past ``max_tokens``; a window
    is never split (its note ranges must stay meaningful), so a single window
    larger than the cap becomes its own chunk. ``max_tokens <= 0`` disables
    chunking."""

    if max_tokens <= 0 or not windows:
        return [list(windows)]
    chunks: list[list[SubtitleWindow]] = []
    current: list[SubtitleWindow] = []
    current_tokens = 0
    printed: set[str] = set()
    for window in windows:
        lines = [f"--- window {window.chunk_id} ---"] + [
            f"{segment.id}|{segment.text or ''}"
            for segment in window.segments
            if segment.id not in printed
        ]
        printed.update(segment.id for segment in window.segments)
        window_tokens = count_tokens("\n".join(lines))
        if current and current_tokens + window_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(window)
        current_tokens += window_tokens
    if current:
        chunks.append(current)
    return chunks


def merge_research_general_contexts(
    per_chunk: Mapping[str, Mapping[str, Any]],
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int = RESEARCH_MERGED_GENERAL_MAX_TOKENS,
) -> dict[str, Any]:
    """Fold each chunk's general context into one block, deterministically.

    Fixed instruction first, then the chunks by stable id. Own token ceiling:
    when it binds, whole trailing chunk blocks are dropped (never truncated
    mid-JSON) with a warning naming them -- silent truncation would read as
    "covered everything"."""

    merged: dict[str, Any] = {}
    kept = {key: dict(value) for key, value in per_chunk.items() if value}
    if not kept:
        return merged
    merged["合并说明"] = MERGED_RESEARCH_GENERAL_NOTE
    merged.update(sorted(kept.items()))
    dropped: list[str] = []
    while (
        len([key for key in merged if key != "合并说明"]) > 1
        and count_tokens(json.dumps(merged, ensure_ascii=False)) > max_tokens
    ):
        last = sorted(key for key in merged if key != "合并说明")[-1]
        merged.pop(last)
        dropped.append(last)
    if dropped:
        current_reporter().warning(
            "research-merged-context-capped",
            f"合并后的全局调查块超过 {max_tokens} tokens，"
            f"丢弃了 {len(dropped)} 个分段的全局部分（{', '.join(sorted(dropped))}）",
            impact="这些分段的逐窗笔记仍然保留",
        )
    return merged


def _run_chunked_research(
    chunk_plan: Sequence[Sequence[SubtitleWindow]],
    *,
    segments: Sequence[Any],
    context_file: Path,
    extra_info: str,
    knowledge_root: str | Path,
    knowledge_enabled: bool,
    profile: TranslationProfile,
    test_profile: bool,
    task_artifact_dir: str | Path | None,
    task_id: str,
    search_rounds: int,
    token_counter: TokenCounter,
    collect_task_feedback: bool,
    resume: bool,
    client: RoleClient,
    transcript_cap: int,
    parallel_windows: int = 1,
) -> Dict[str, Any]:
    """Run research per chunk and assemble one payload (plan W7 / P8).

    Every chunk runs the complete r1 + search loop + r2 over its own slice of
    the transcript and lands in its own ``<context>-<chunk>.json`` (the resume
    unit: a rerun replays finished chunks off their input hash). Only after
    every chunk is done is the final pack assembled -- correction never starts
    against a partial context. Assembly is a pure function: fixed chunk ids,
    fixed order, fixed instruction text, so two runs over the same inputs
    produce comparable injections (the pack is not fingerprinted since
    2026-08-12, so this is comparability, not cache identity).

    Chunks are independent, so under ``continuity=parallel`` they fan out --
    each chunk rides ONE worker (= one lane: its r1, judge rounds and r2 share
    that lane's conversation and prefix cache), chunks spread across workers,
    and the fan-out is bounded by ``parallel_windows`` and the task's
    ``claim_cap`` like any other intra-task parallel face (plan W4). The
    per-artifact-dir writers are shared single instances -- the exchange
    logger, the checkpoint store, the search client -- because their instance
    locks are what serializes exchange numbering and JSONL appends; exchange
    file numbers become completion-ordered (the same A/B noise the parallel
    correction windows already carry).
    """

    from .run_context import current_task_account

    def _research_routes_to_conversational(role_client: RoleClient) -> bool:
        """Does the research cell land on a person's own agent?

        Asked with the rounds' own routing arguments (r1/r2 take the module
        defaults: GENERAL_CAPABLE in the ``research`` task group), because the
        correction stage's answer says nothing about this phase's cell.
        """

        probe = getattr(role_client, "routes_to_conversational", None)
        if not callable(probe):
            return False
        try:
            return bool(probe(LLMRole.GENERAL_CAPABLE, task_group="research"))
        except Exception:  # noqa: BLE001 -- an unroutable cell fails loudly later
            return False

    fan_out = 1
    if profile.continuity == "parallel":
        if _research_routes_to_conversational(client):
            # Plan W6, the research face of it: a person's own agent serves ONE
            # queue, so N chunks fan out into N assignments behind (usually) one
            # agent -- serial's wall clock, minus nothing gained. The correction
            # stage forces serial for the same reason, but it does so later and
            # on its own role, which left this phase unguarded.
            current_reporter().warning(
                "conversational-forced-serial",
                "conversational 后端不支持研究块并行，本次分块调查串行执行",
                impact="块顺序执行，产物与合并口径不变",
            )
        else:
            fan_out = min(len(chunk_plan), max(1, int(parallel_windows)))
            account = current_task_account()
            if account is not None:
                fan_out = max(1, min(fan_out, account.claim_cap()))
    current_reporter().debug(
        "research-chunked",
        {"chunks": len(chunk_plan), "cap_tokens": transcript_cap, "fan_out": fan_out},
    )
    shared_exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    shared_checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)
    shared_search_client = (
        WebSearchClient(execution_settings=getattr(client, "execution_settings", None))
        if profile.external_injection
        else None
    )

    def _run_chunk(index: int) -> dict[str, Any]:
        """One chunk end to end; returns the chunk record with its parts."""

        chunk_windows = chunk_plan[index]
        chunk_id = f"c{index + 1:02d}"
        chunk_transcript = render_research_transcript(segments, chunk_windows)
        chunk_segment_count = sum(
            len(window.segments[len(window.overlap_segments):])
            for window in chunk_windows
        )
        # The same gates as the stage-level L3 reuse (RESEARCH_REUSE_GATES),
        # with the chunk's own transcript standing in for stable_json_hash --
        # finer-grained, and geometry changes reshape it anyway. Difficulty is
        # deliberately absent, exactly as it is absent from the gates: "switch
        # to intermediate and continue" must replay finished chunks, not burn
        # quota re-running them (reviewer 2026-08-30 P1-2: the first cut both
        # missed prompt_version/test_profile and over-gated difficulty).
        chunk_input_hash = _short_sha256(
            json.dumps(
                {
                    "transcript": chunk_transcript,
                    "extra_info": extra_info,
                    "prompt_version": PROMPT_VERSION,
                    "test_profile": bool(test_profile),
                    "research_semantics_id": profile.retrieval,
                    "search_rounds": int(search_rounds) if profile.external_injection else 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        chunk_file = context_file.with_name(
            f"{context_file.stem}-{chunk_id}{context_file.suffix}"
        )
        payload: Dict[str, Any] | None = None
        if resume and chunk_file.exists():
            try:
                saved = json.loads(chunk_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = None
            if isinstance(saved, dict) and saved.get("chunk_input_hash") == chunk_input_hash:
                payload = saved
        replayed = payload is not None
        if payload is None:
            payload = run_research(
                transcript=chunk_transcript,
                extra_info=extra_info,
                knowledge_root=knowledge_root,
                knowledge_enabled=knowledge_enabled,
                search_client=shared_search_client,
                profile=profile,
                test_profile=test_profile,
                task_artifact_dir=task_artifact_dir,
                task_id=task_id,
                max_search_queries=research_search_query_limit(chunk_segment_count),
                search_rounds=search_rounds,
                token_counter=token_counter,
                collect_task_feedback=collect_task_feedback,
                resume=resume,
                client=client,
                chunk_label=chunk_id,
                exchange_logger=shared_exchange_logger,
                checkpoint_store=shared_checkpoint_store,
            )
            payload["chunk_input_hash"] = chunk_input_hash
            payload["chunk_id"] = chunk_id
            payload["chunk_window_ids"] = [window.chunk_id for window in chunk_windows]
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            chunk_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # Bind against this chunk's own windows: their source-id intervals are
        # disjoint across chunks, which is what makes the note merge a plain
        # concatenation.
        bind_report: dict[str, Any] = {}
        pack = ContextPack.from_dict(payload["context_pack"]).bind_window_ranges(
            list(chunk_windows), report_sink=bind_report
        )
        if bind_report.get("unplaceable_window_ids"):
            current_reporter().warning(
                "research-notes-unplaceable",
                f"research chunk {chunk_id} named "
                f"{len(bind_report['unplaceable_window_ids'])} window(s) outside "
                f"its slice ({', '.join(bind_report['unplaceable_window_ids'])})",
                impact=f"这些笔记被丢弃，该块剩余 {bind_report.get('bound', 0)} 条",
            )
        return {
            "pack": pack,
            "payload": payload,
            "record": {
                "chunk_id": chunk_id,
                "window_ids": [window.chunk_id for window in chunk_windows],
                "segment_count": chunk_segment_count,
                "input_hash": chunk_input_hash,
                "file": chunk_file.name,
                "replayed": replayed,
                "bound_notes": bind_report.get("bound", 0),
            },
        }

    if fan_out <= 1:
        outcomes = [_run_chunk(index) for index in range(len(chunk_plan))]
    else:
        # Same worker channel as the correction pools: reporter, pins,
        # registry, lane and the mandatory-lane claim all ride the
        # initializer; lanes go back to the run when the phase ends.
        import concurrent.futures as cf

        from .run_context import bind_llm_worker, llm_worker_context

        worker_ctx = llm_worker_context()
        try:
            with cf.ThreadPoolExecutor(
                max_workers=fan_out,
                thread_name_prefix="llm-research-chunk",
                initializer=bind_llm_worker,
                initargs=(worker_ctx,),
            ) as pool:
                # Completion order is free; assembly below is index-ordered.
                outcomes = list(pool.map(_run_chunk, range(len(chunk_plan))))
        finally:
            worker_ctx.release_lanes()

    chunk_records: list[dict[str, Any]] = []
    merged_notes: list[Any] = []
    per_chunk_general: dict[str, Mapping[str, Any]] = {}
    keep_union: list[str] = []
    injected_union: list[str] = []
    token_totals: list[Mapping[str, Any]] = []
    rounds_meta: Mapping[str, Any] = {}
    for outcome in outcomes:
        pack, payload = outcome["pack"], outcome["payload"]
        merged_notes.extend(pack.window_contexts)
        if pack.general_context:
            per_chunk_general[outcome["record"]["chunk_id"]] = dict(pack.general_context)
        keep_union.extend(payload.get("keep_entries") or [])
        injected_union.extend(payload.get("injected_entries") or [])
        totals = (payload.get("token_report") or {}).get("totals")
        if isinstance(totals, Mapping):
            token_totals.append(totals)
        rounds_meta = payload.get("rounds") or rounds_meta
        chunk_records.append(outcome["record"])
    merged_pack = ContextPack(
        general_context=merge_research_general_contexts(
            per_chunk_general, count_tokens=token_counter.count_text
        ),
        window_contexts=tuple(merged_notes),
    )
    keep_cap = (
        KB_TRANSFER_MAX_ENTRIES if profile.external_injection else KB_WINDOW_TOTAL_ENTRIES
    )
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="research_chunk_plan",
            task_id=task_id,
            payload={
                "cap_tokens": transcript_cap,
                "fan_out": fan_out,
                "chunks": chunk_records,
            },
        )
    return {
        "context_pack": merged_pack.to_dict(),
        "keep_entries": list(dict.fromkeys(keep_union))[:keep_cap],
        "rounds": dict(rounds_meta),
        "research_chunks": chunk_records,
        "chunking": {"research_transcript_max_tokens": transcript_cap},
        "injected_entries": sorted(set(injected_union)),
        "token_report": {
            "phase": "research",
            "rows": [],
            "totals": sum_token_distributions(token_totals),
        },
    }


def run_research_stage(
    *,
    stable_json: str | Path,
    context_path: str | Path,
    audio_path: str | Path | None = None,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    search_rounds: int,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    token_counter: TokenCounter | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    collect_task_feedback: bool = False,
    resume: bool = True,
    max_window_subtitle_tokens: int | None = None,
    parallel_windows: int = 1,
) -> ContextPack:
    """Plan the correction windows, run both research rounds, persist the
    research context JSON next to the output, and return the ContextPack.
    Both the CLI and run_full_correction go through here.

    ``parallel_windows`` is the task's intra-task willingness knob
    (``--llm-parallel-windows``); under ``continuity=parallel`` it also bounds
    how many research CHUNKS run at once on an ultra-long material (plan W7)."""

    token_counter = token_counter or default_token_counter()
    stage_client = RoleClient(test_profile=test_profile)
    segments = load_segments_from_stable_json(stable_json)
    plan_report: dict = {}
    # Research uses the current correction planner for sensible transcript
    # chunks, but correctness no longer requires later correction geometry to
    # match: the resulting notes are bound to source-id intervals below.
    from .routing.capabilities import correction_planning_limits

    planning_limits = correction_planning_limits(profile)
    audio_duration = probe_audio_duration(audio_path) if audio_path else None
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
    if task_artifact_dir and plan_report.get("replan_attempts"):
        append_task_artifact(
            task_artifact_dir,
            kind="window_plan_report",
            task_id=task_id,
            payload={"phase": "research", **plan_report},
        )
    context_file = Path(context_path)
    # Ultra-long material splits into transcript chunks of whole windows, each
    # running the full r1 + search-loop + r2 (plan W7 / P8) -- chunking only
    # r2 would leave an equally oversized r1 blind on retrieval topics.
    # ``retrieval=native`` never enters this path.
    transcript_cap = (
        resolve_research_transcript_cap() if profile.retrieval != "native" else 0
    )
    chunk_plan = plan_research_chunks(
        windows, count_tokens=token_counter.count_text, max_tokens=transcript_cap
    )
    if len(chunk_plan) <= 1:
        research_payload = run_research(
            transcript=render_research_transcript(segments, windows),
            extra_info=extra_info,
            knowledge_root=knowledge_root,
            knowledge_enabled=knowledge_enabled,
            profile=profile,
            test_profile=test_profile,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            max_search_queries=research_search_query_limit(len(segments)),
            search_rounds=search_rounds,
            token_counter=token_counter,
            collect_task_feedback=collect_task_feedback,
            resume=resume,
            client=stage_client,
        )
    else:
        research_payload = _run_chunked_research(
            chunk_plan,
            segments=segments,
            context_file=context_file,
            extra_info=extra_info,
            knowledge_root=knowledge_root,
            knowledge_enabled=knowledge_enabled,
            profile=profile,
            test_profile=test_profile,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            search_rounds=search_rounds,
            token_counter=token_counter,
            collect_task_feedback=collect_task_feedback,
            resume=resume,
            client=stage_client,
            transcript_cap=transcript_cap,
            parallel_windows=parallel_windows,
        )
    # The model names the research windows, while the harness owns their exact
    # source-id coverage. Persist both so later correction geometry changes do
    # not make a surviving chunk id point at the wrong note.
    bind_report: dict[str, Any] = {}
    bound_context = ContextPack.from_dict(
        research_payload["context_pack"]
    ).bind_window_ranges(windows, report_sink=bind_report)
    research_payload["context_pack"] = bound_context.to_dict()
    if bind_report.get("unplaceable_window_ids"):
        # Not fatal -- general context still covers every window -- but it
        # must not be silent: a pack that lost all of its per-window notes is
        # indistinguishable from one that never had any.
        research_payload["window_context_bind_report"] = bind_report
        current_reporter().warning(
            "research-notes-unplaceable",
            "research round 2 named "
            f"{len(bind_report['unplaceable_window_ids'])} window(s) that are "
            f"not in the plan ({', '.join(bind_report['unplaceable_window_ids'])})",
            impact=f"这些笔记被丢弃，剩余 {bind_report['bound']} 条",
            file=sys.stderr,
        )
    research_payload["planning"] = planning_metadata(
        profile,
        stable_json=stable_json,
        extra_info=extra_info,
        knowledge_root=knowledge_root,
        knowledge_enabled=knowledge_enabled,
        search_rounds=search_rounds,
        collect_task_feedback=collect_task_feedback,
        audio_duration=audio_duration,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        limits=planning_limits,
        test_profile=test_profile,
        execution_identity_override=getattr(
            stage_client, "execution_identity", None
        ),
    )
    context_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.write_text(
        json.dumps(research_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return bound_context
