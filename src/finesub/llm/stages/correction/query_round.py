"""Round 1 of a correction window: light analysis, then local search.

The round is best-effort throughout -- every failure path returns an empty
:class:`QueryRoundProduct` and the correction round proceeds without the
input. It takes explicit arguments rather than the run context: fast mode
seeds its product from round 1 and never calls this at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from ...client import (
    GeminiPromptBlockedError,
    LLMCallResult,
    RoleClient,
    extract_token_distribution,
    is_prompt_blocked,
    validation_retry_sampling_kwargs,
)
from ...media_upload import UploadedFileRef
from ...routing.capabilities import planning_task_group
from ...chunking import SubtitleWindow
from ...routing.config import (
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    LLMRole,
    MAX_WINDOW_SEARCH_QUERIES,
    QUERY_ROUND_MAX_TOKENS,
    WINDOW_NOTES_MAX_TOKENS,
    injection_block_token_limit,
)
from ...content_filter import (
    ContentFilterExhaustedError,
    run_injection_ladder,
    split_rendered_search_block,
)
from ...exchange_log import ExchangeBlock, ExchangeLogger
from ...exchange_metadata import (
    correction_input_components,
    llm_exchange_metadata,
    result_uses_high_resolution_video,
)
from ...knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    load_entry_texts,
)
from ...output_tags import (
    extract_single_tag_block,
    missing_top_level_tags,
    parse_guided_line_items,
    parse_line_items,
)
from ...routing.profiles import DEFAULT_PROFILE, TranslationProfile
from ...prompts import PROMPT_VERSION, ContextPack, build_correction_query_messages
from ...session_checkpoint import SessionCheckpointStore, session_input_hash
from ...session_contract import query_round_contract
from ...token_truncate import cap_tokens
from ...web_search import (
    SearchRequest,
    WebSearchClient,
    render_search_results,
    search_results_metadata,
)
from .metadata import _output_limit_check, _response_finish_reason


# Query round is best-effort; a reply missing one of the query contract's
# present blocks (a sibling swallowed inside another — structural corruption)
# gets this many plain retries before the round proceeds with whatever parsed.
# The gate uses the contract's may-be-empty present blocks (single source of
# truth in finesub.llm.session_contract) — window_notes/keep_entries/search_queries.
# The contract's nonempty <reasoning> is excluded from the retry gate: it is
# always emitted first and carries no downstream data.
QUERY_ROUND_FORMAT_RETRIES = 1
# Resolved per run: with no readable index the query prompt drops the entry
# rules, so demanding <keep_entries> back would retry a round for omitting
# a block it was never asked for.


# Output cap for the per-window query round: queries are tiny, but medium
# thinking on the lite model needs headroom above the visible output.


def _extract_window_notes(
    text: str,
    *,
    max_tokens: int = WINDOW_NOTES_MAX_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Best-effort <window_notes> extraction: any format issue yields ''."""

    try:
        body = extract_single_tag_block(text, "window_notes", required=False)
    except ValueError:
        return ""
    return cap_tokens(body.strip(), max_tokens, count_tokens)


@dataclass(frozen=True)
class QueryRoundProduct:
    """Everything a window's query round produces for its correction round.

    ``requested_entry_keys`` (v17) are the canonical primary keys of the
    round's resolved entry requests; the window loop merges them with the
    transfer set and renders the union once before the correction round.
    ``keep_entry_keys`` are carried entries the query round confirms as
    still relevant; merged into the transfer set for subsequent windows.
    """

    search_results: str = ""
    window_notes: str = ""
    requested_entry_keys: tuple[str, ...] = ()
    keep_entry_keys: tuple[str, ...] = ()


def run_window_query_round(
    *,
    client: RoleClient,
    window: SubtitleWindow,
    context_pack: ContextPack | None,
    audio_label: str,
    previous_advice: str,
    file_ref: UploadedFileRef | None,
    search_client: WebSearchClient,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    streamer_index: str = "",
    common_index: str = "",
    knowledge_enabled: bool = True,
    max_queries: int = MAX_WINDOW_SEARCH_QUERIES,
    carried_entries_text: str = "",
    carried_key_count: int = 0,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    token_rows: List[Dict[str, Any]] | None = None,
    exchange_logger: ExchangeLogger | ExchangeBlock | None = None,
    token_counter: Any | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    resume: bool = True,
    checkpoint_store: SessionCheckpointStore | None = None,
    checkpoint_extra_identity: Mapping[str, Any] | None = None,
) -> QueryRoundProduct:
    """Round 1 of a correction window: light analysis plus local search.

    Best-effort: any failure (call error, format error, search failure)
    degrades to empty strings and the correction round proceeds without that
    input. mm-low runs the same round text-only on the lightweight role. The
    model sees both knowledge indices and may emit ``<requested_entries>``;
    the resolved entry bodies are budget-rendered into ``entry_details`` for
    the correction round (independent budget from the search block).
    """

    remaining_new = max(
        0,
        min(
            KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
            KB_WINDOW_TOTAL_ENTRIES - max(0, int(carried_key_count)),
        ),
    )
    messages = build_correction_query_messages(
        window=window,
        context_pack=context_pack,
        audio_file_label=audio_label,
        previous_advice=previous_advice,
        streamer_index=streamer_index,
        common_index=common_index,
        carried_entries=carried_entries_text,
        carried_entry_count=carried_key_count,
        max_search_queries=max_queries,
        profile=profile,
    )
    # 纠错 r1 always uses the multimodal lightweight role (same 3.5-lite
    # chain as LIGHTWEIGHT); search-loop judge stays on LIGHTWEIGHT.
    query_role = LLMRole.LIGHTWEIGHT_MULTIMODAL
    checkpoint_store = checkpoint_store or SessionCheckpointStore(
        task_artifact_dir, enabled=resume
    )
    checkpoint_hash = session_input_hash(
        messages,
        prompt_version=PROMPT_VERSION,
        call_config={
            "role": query_role.value,
            "output_reserve": QUERY_ROUND_MAX_TOKENS,
            "file_backed": bool(profile.planning_use_audio and file_ref is not None),
        },
        extra_identity=checkpoint_extra_identity,
        execution_identity_override=getattr(client, "execution_identity", None),
    )
    cached = checkpoint_store.get("query", window.chunk_id, checkpoint_hash)
    cached_valid = False
    if cached is not None and not query_round_contract(knowledge_enabled=knowledge_enabled).validate(cached.content):
        try:
            parse_guided_line_items(
                extract_single_tag_block(cached.content, "search_queries")
            )
        except ValueError:
            pass
        else:
            cached_valid = True
    if cached is not None and not cached_valid and task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="session_checkpoint_invalid",
            task_id=task_id,
            payload={
                "session": "query",
                "key": window.chunk_id,
                "input_hash": checkpoint_hash,
            },
        )

    # Bumped by the format-retry loop below; read by _query_complete (closure)
    # to raise sampling temperature on re-tries so a re-roll can differ.
    query_attempt = 0

    def _query_complete(_injection: str = ""):
        result = client.complete(
            query_role,
            messages,
            output_reserve=QUERY_ROUND_MAX_TOKENS,
            file_ref=file_ref if profile.planning_use_audio else None,
            task_group=planning_task_group(profile),
            difficulty=profile.difficulty,
            # §4.3 matrix: the query round reads the kb like the window it serves
            agent_task_extras=(
                {
                    "kb_tools": "read",
                    "kb_signal_task": task_id,
                    "kb_signal_window": window.chunk_id,
                }
                if knowledge_enabled
                else None
            ),
            **(validation_retry_sampling_kwargs(query_attempt) if query_attempt else {}),
        )
        if is_prompt_blocked(result.content, result.raw_response):
            raise GeminiPromptBlockedError(
                f"窗口 {window.chunk_id} 查询轮 prompt was blocked by the content filter"
            )
        return result

    tag_errors: List[str] = []
    checkpoint_replayed = False
    if cached_valid:
        assert cached is not None
        result = LLMCallResult(
            content=cached.content,
            role=query_role,
            model=str(cached.metadata.get("model") or "checkpoint"),
            fallback_used=bool(cached.metadata.get("fallback_used", False)),
            raw_response={},
        )
        checkpoint_replayed = True
    else:
        while True:
            try:
                # No droppable web-retrieval units in the query round — plain retry only.
                query_outcome = run_injection_ladder(
                    block=split_rendered_search_block(""),
                    call=_query_complete,
                    stage=f"correction_query_{window.chunk_id}",
                    blocked_exception=GeminiPromptBlockedError,
                    task_artifact_dir=task_artifact_dir,
                    task_id=task_id,
                    plain_retry=True,
                )
                result = query_outcome.result
            except ContentFilterExhaustedError as exc:
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="correction_query_call_error",
                        task_id=task_id,
                        payload={
                            "chunk_id": window.chunk_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                return QueryRoundProduct()
            except Exception as exc:  # pragma: no cover - provider behavior
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="correction_query_call_error",
                        task_id=task_id,
                        payload={
                            "chunk_id": window.chunk_id,
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
                return QueryRoundProduct()
            # A missing first-level block means a sibling was swallowed inside
            # another block (structural corruption) — retry. Nesting itself is
            # fine (e.g. <reasoning> may name-drop other tags).
            tag_errors = missing_top_level_tags(
                result.content,
                list(query_round_contract(knowledge_enabled=knowledge_enabled).present),
            )
            if not tag_errors or query_attempt >= QUERY_ROUND_FORMAT_RETRIES:
                break
            if task_artifact_dir:
                append_task_artifact(
                    task_artifact_dir,
                    kind="correction_query_format_retry",
                    task_id=task_id,
                    payload={
                        "chunk_id": window.chunk_id,
                        "attempt": query_attempt,
                        "errors": tag_errors,
                        "api_attempts": list(result.api_attempts),
                        "execution_attempts": list(result.execution_attempts),
                        "route_decision": dict(result.route_decision),
                    },
                )
            query_attempt += 1
    finish_reason = _response_finish_reason(result.raw_response)
    output_limit_check = _output_limit_check(
        result.raw_response, result.requested_output_tokens or QUERY_ROUND_MAX_TOKENS
    )
    output_limited = bool(output_limit_check["limited"])
    if token_rows is not None and not checkpoint_replayed:
        token_rows.append(
            {
                "call": "correction_query",
                "chunk_id": window.chunk_id,
                "model": result.model,
                "finish_reason": finish_reason,
                "output_limit_check": output_limit_check,
                "tokens": extract_token_distribution(result.raw_response),
            }
        )
    window_notes = _extract_window_notes(
        result.content, count_tokens=token_counter.count_text
    )
    # Residual missing-top-level errors after retries feed the logged parse_error.
    parse_error = "; ".join(tag_errors)
    query_pairs: List[tuple[str, str]] = []
    try:
        query_pairs = parse_guided_line_items(
            extract_single_tag_block(result.content, "search_queries")
        )
    except ValueError as exc:
        parse_error = "; ".join(filter(None, [parse_error, str(exc)]))
    # Optional knowledge-entry requests: resolved to canonical keys here; the
    # window loop merges them with the transfer set and renders the union once
    # (v17), so no rendering happens in this round anymore.
    requested_entries: List[str] = []
    try:
        requested_entries = parse_line_items(
            extract_single_tag_block(result.content, "requested_entries", required=False)
        )[:remaining_new]
    except ValueError:
        requested_entries = []
    # keep_entries: carried entries the model confirms as still relevant.
    keep_entries_raw: List[str] = []
    try:
        keep_entries_raw = parse_line_items(
            extract_single_tag_block(result.content, "keep_entries", required=False)
        )
    except ValueError:
        keep_entries_raw = []
    resolved_entry_keys: List[str] = []
    missing_entries: List[str] = []
    if requested_entries:
        found_entries, missing_entries = load_entry_texts(
            knowledge_root, requested_entries
        )
        resolved_entry_keys = list(found_entries)
    pack = context_pack or ContextPack()
    context_report: Dict[str, Any] = {}
    context_window = pack.window_context_for(
        window, counter=token_counter, report_sink=context_report
    )
    if context_report and task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="window_context_truncated",
            task_id=task_id,
            payload={"phase": "query", **context_report},
        )
    input_components = correction_input_components(
        window=window,
        counter=token_counter,
        context_general=pack.general_prompt_text(),
        context_window=context_window,
        messages=messages,
        max_output_tokens=QUERY_ROUND_MAX_TOKENS,
        file_ref=file_ref,
        video_high_resolution=result_uses_high_resolution_video(result),
    )
    session = f"correction-{window.chunk_id}-query-attempt{query_attempt}"
    queries = [query for query, _ in query_pairs]
    if exchange_logger and not checkpoint_replayed:
        exchange_logger.log(
            session,
            messages=messages,
            response_text=result.content,
            metadata=llm_exchange_metadata(
                result,
                session=session,
                input_components=input_components,
                chunk_id=window.chunk_id,
                queries="; ".join(queries) or "（无）",
                window_notes_chars=len(window_notes),
                finish_reason=finish_reason,
                output_limited=output_limited,
                output_limit_basis=output_limit_check["basis"],
                output_limit_observed_tokens=output_limit_check[
                    "observed_output_tokens"
                ],
                output_limit_threshold_tokens=output_limit_check["threshold_tokens"],
                output_limit_max_tokens=output_limit_check["max_output_tokens"],
                output_limit_margin_tokens=output_limit_check["margin_tokens"],
                **({"parse_error": parse_error} if parse_error else {}),
            ),
        )
    if task_artifact_dir and not checkpoint_replayed:
        append_task_artifact(
            task_artifact_dir,
            kind="correction_query_response",
            task_id=task_id,
            payload={
                "session": session,
                "chunk_id": window.chunk_id,
                "attempt": query_attempt,
                "model": result.model,
                "fallback_used": result.fallback_used,
                "usage": extract_token_distribution(result.raw_response),
                "api_attempts": list(result.api_attempts),
                "execution_attempts": list(result.execution_attempts),
                "route_decision": dict(result.route_decision),
                "input_components": input_components,
                "finish_reason": finish_reason,
                "output_limited": output_limited,
                "output_limit_check": output_limit_check,
                "parse_error": parse_error,
                "queries": queries,
                "window_notes": window_notes,
                "requested_entries": requested_entries,
                "resolved_entry_keys": resolved_entry_keys,
                "missing_entries": missing_entries,
                "response_content": result.content,
            },
        )
    checkpoint_valid = not query_round_contract(knowledge_enabled=knowledge_enabled).validate(result.content)
    if checkpoint_replayed:
        if task_artifact_dir:
            append_task_artifact(
                task_artifact_dir,
                kind="session_checkpoint_replay",
                task_id=task_id,
                payload={
                    "session": "query",
                    "key": window.chunk_id,
                    "input_hash": checkpoint_hash,
                },
            )
    elif (
        checkpoint_valid
        and not parse_error
        and not output_limited
        # Gate D answer C: an implicit-history call must not seed L1
        # (docs/llm_local_agent.md §7); the resume re-sends it instead.
        and getattr(result, "resumable", True)
    ):
        checkpoint_store.commit(
            session="query",
            key=window.chunk_id,
            input_hash=checkpoint_hash,
            content=result.content,
            metadata={
                "model": result.model,
                "fallback_used": result.fallback_used,
            },
        )
    if not queries:
        return QueryRoundProduct(
            window_notes=window_notes,
            requested_entry_keys=tuple(resolved_entry_keys),
            keep_entry_keys=tuple(keep_entries_raw),
        )
    search_requests = [
        SearchRequest(query=query, guided_query=guided) for query, guided in query_pairs
    ]
    results = search_client.search_many(search_requests, max_queries=max_queries)
    rendered = render_search_results(
        results,
        max_total_tokens=injection_block_token_limit(max_queries),
        count_tokens=token_counter.count_text,
    )
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="correction_search_results",
            task_id=task_id,
            payload={
                "chunk_id": window.chunk_id,
                "queries": queries,
                "executed": search_results_metadata(results),
                "rendered_tokens": rendered.tokens,
                "render_report": rendered.report(),
            },
        )
    return QueryRoundProduct(
        search_results=rendered.text,
        window_notes=window_notes,
        requested_entry_keys=tuple(resolved_entry_keys),
        keep_entry_keys=tuple(keep_entries_raw),
    )
