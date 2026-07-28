"""Correction-window execution loop (query round, retries, resume, rendering).

Moved out of correction_translation.py as part of the stage split; the public
entrypoint is :func:`execute_correction_windows`, re-exported from
``llm.correction_translation`` for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from to_srt import format_srt_time

from ..audio_clips import (
    CLIP_AUDIO_SUFFIX,
    CLIP_VIDEO_SUFFIX,
    extract_window_clip,
    extract_window_video_clip,
    probe_audio_duration,
)
from ..clip_prefetch import WindowClipPrefetcher
from ..chunking import (
    SubtitleWindow,
    WindowIdMap,
    load_segments_from_stable_json,
    plan_correction_windows,
    render_segments_as_srt,
    render_window_preceding_as_csv,
    render_window_segments_as_csv,
    split_window_in_half,
)
from ..client import (
    GeminiPromptBlockedError,
    LLMCallResult,
    LiteLLMRoleClient,
    UploadedFileRef,
    extract_token_distribution,
    is_prompt_blocked,
    sum_token_distributions,
    upload_gemini_file,
    validation_retry_sampling_kwargs,
)
from ..config import (
    ADVICE_LEDGER_MAX_TOKENS,
    CapabilityTier,
    DEFAULT_LIMITS,
    INJECTION_SECTION_MAX_TOKENS,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    LLMRole,
    MAX_WINDOW_SEARCH_QUERIES,
    NEXT_ADVICE_MAX_TOKENS,
    QUERY_ROUND_MAX_TOKENS,
    WINDOW_NOTES_MAX_TOKENS,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    injection_block_token_limit,
    thinking_budget_for_level,
)
from ..content_filter import (
    ContentFilterExhaustedError,
    evidence_pack_block,
    load_content_filter_blacklist,
    run_injection_ladder,
    split_rendered_search_block,
)
from ..csv_utils import (
    PACING_PASS_RATIO,
    PACING_PASS_RATIO_TEST_PROFILE,
    CsvValidationResult,
    TranslatedCsvSegment,
    merge_translated_csv_windows,
    render_corrected_segments_as_srt,
    render_translated_segments_as_csv,
    render_translated_segments_as_srt,
    score_translated_segments,
    validate_correction_window_output,
)
from ..exchange_log import ExchangeLogger
from ..exchange_metadata import correction_input_components, llm_exchange_metadata
from ..injection_budget import render_knowledge_entries_block
from ..session_contract import SESSION_CONTRACTS
from ..knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    load_entry_texts,
    load_index_text,
)
from ..knowledge.feedback import remap_feedback_source_ids
from ..output_tags import (
    extract_single_tag_block,
    parse_guided_line_items,
    parse_line_items,
)
from ..prompt_variants import resolve_variant
from ..profiles import DEFAULT_PROFILE, TranslationProfile
from ..session_checkpoint import SessionCheckpointStore, session_input_hash


def correction_role_for_profile(profile: TranslationProfile) -> LLMRole:
    """LLM role for the correction window ("纠错 r2" / fast correction step).

    Uses ``audio_multimodal``'s 3.6-first chain for every non-native-search
    profile (including text routes) so subtitle merge work prefers 3.6 Flash.
    text-high keeps ``internet_capable``.
    """

    if profile.native_search:
        return LLMRole.INTERNET_CAPABLE
    return LLMRole.AUDIO_MULTIMODAL


def _thinking_override_kwargs(profile: TranslationProfile) -> Dict[str, Any]:
    """Per-call thinking override (text-low -> low); budget derives from level."""

    if not profile.thinking_override:
        return {}
    return {
        "thinking_level": profile.thinking_override,
        "thinking_budget": thinking_budget_for_level(profile.thinking_override),
    }
from ..prompts import (
    PROMPT_VERSION,
    ContextPack,
    build_correction_csv_messages,
    build_correction_query_messages,
    render_advice_ledger,
)
from ..srt_postprocess import DEFAULT_POSTPROCESS_PROFILE, postprocess_srt_file
from ..task_report import write_task_report
from ..token_budget import default_token_counter, TokenCounter
from ..token_truncate import cap_tokens, truncate_text_only
from ..web_search import (
    SearchRequest,
    WebSearchClient,
    render_search_results,
    search_results_metadata,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _segments_metadata(segments: Iterable[Any]) -> Dict[str, Any]:
    segment_list = list(segments)
    if not segment_list:
        return {
            "count": 0,
            "source_ids": [],
            "start": None,
            "end": None,
            "duration_seconds": 0.0,
        }
    start = float(segment_list[0].start)
    end = float(segment_list[-1].end)
    return {
        "count": len(segment_list),
        "source_ids": [segment.id for segment in segment_list],
        "start": start,
        "end": end,
        "duration_seconds": round(max(0.0, end - start), 3),
    }


def window_to_metadata(window: SubtitleWindow) -> Dict[str, Any]:
    overlap = _segments_metadata(window.overlap_segments)
    return {
        "chunk_id": window.chunk_id,
        "start": window.start,
        "end": window.end,
        "duration_seconds": round(max(0.0, window.end - window.start), 3),
        "segment_count": len(window.segments),
        "source_ids": window.source_ids,
        "source_id_range": [window.source_ids[0], window.source_ids[-1]],
        "overlap": overlap,
        "overlap_source_ids": overlap["source_ids"],
        "preceding_source_ids": [segment.id for segment in window.preceding_segments],
        "clip_start": round(window.clip_start, 3),
        "clip_end": round(window.clip_end, 3),
        "clip_duration_seconds": round(max(0.0, window.clip_end - window.clip_start), 3),
        "boundary_reason": window.boundary_reason,
        "budget": {
            "input_tokens": window.budget.input_tokens,
            "subtitle_input_tokens": window.budget.subtitle_input_tokens,
            "estimated_output_tokens": window.budget.estimated_output_tokens,
            "total_with_margin": window.budget.total_with_margin,
            "token_counter": window.budget.token_counter_source,
        },
    }


def _iter_message_text_parts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_message_text_parts(item)
        return
    if isinstance(value, Mapping):
        if value.get("type") == "text":
            yield str(value.get("text", ""))
            return
        if "text" in value and len(value) == 1:
            yield str(value.get("text", ""))
            return
        for key, item in value.items():
            if key in {"file", "fileData", "file_data"}:
                continue
            yield from _iter_message_text_parts(item)
        return
    yield str(value)


def _message_fingerprints(messages: List[Dict[str, Any]]) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for message in messages:
        text = "\n".join(_iter_message_text_parts(message.get("content", "")))
        fingerprints.append(
            {
                "role": str(message.get("role", "")),
                "text_chars": len(text),
                "text_sha256": _sha256_text(text),
            }
        )
    return fingerprints


def _request_reference_metadata(
    *,
    messages: List[Dict[str, Any]],
    file_ref: UploadedFileRef | None,
    max_tokens: int,
) -> Dict[str, Any]:
    # Real token numbers come from the provider usage metadata recorded in the
    # response payload and the token_distribution_report artifact; the request
    # side only keeps fingerprints for reproducibility checks.
    message_text_chars = sum(
        len("\n".join(_iter_message_text_parts(message.get("content", ""))))
        for message in messages
    )
    metadata: Dict[str, Any] = {
        "requested_output_tokens": max_tokens,
        "message_text_chars": message_text_chars,
        "message_fingerprints": _message_fingerprints(messages),
    }
    if file_ref:
        metadata["audio_file"] = {
            "attached": True,
            "file_id": file_ref.file_id,
            "filename": file_ref.filename,
            "mime_type": file_ref.mime_type,
            "note": (
                "The attached audio is this window's clip (padding included), "
                "so billing follows the clip duration; see the "
                "token_distribution_report artifact for the real modality split."
            ),
        }
    else:
        metadata["audio_file"] = {"attached": False}
    return metadata


def _response_reference_metadata(content: str) -> Dict[str, Any]:
    return {
        "content_chars": len(content or ""),
        "content_sha256": _sha256_text(content or ""),
    }


def _provider_reference_metadata(raw_response: Any) -> Dict[str, Any]:
    if not isinstance(raw_response, Mapping):
        return {}
    metadata: Dict[str, Any] = {}
    for key in ("usageMetadata", "usage_metadata", "usage"):
        value = raw_response.get(key)
        if isinstance(value, Mapping):
            metadata[key] = dict(value)
    for key in ("modelVersion", "responseId", "id", "created", "promptFeedback"):
        value = raw_response.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _response_finish_reason(raw_response: Any) -> str:
    if not isinstance(raw_response, dict):
        return ""
    candidates = raw_response.get("candidates") or raw_response.get("choices") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    return str(
        candidates[0].get("finishReason")
        or candidates[0].get("finish_reason")
        or candidates[0].get("finish_reason".upper())
        or ""
    )


OUTPUT_LIMIT_TOKEN_MARGIN = 100

# Query round is best-effort; a reply missing one of the query contract's
# present blocks (a sibling swallowed inside another — structural corruption)
# gets this many plain retries before the round proceeds with whatever parsed.
# The gate uses the contract's may-be-empty present blocks (single source of
# truth in llm.session_contract) — window_notes/keep_entries/search_queries.
# The contract's nonempty <reasoning> is excluded from the retry gate: it is
# always emitted first and carries no downstream data.
QUERY_ROUND_FORMAT_RETRIES = 1
_QUERY_REQUIRED_TOP_LEVEL = SESSION_CONTRACTS["query"].present


def _output_limit_check(
    raw_response: Any,
    max_tokens: int = DEFAULT_LIMITS.output_limit,
    margin: int = OUTPUT_LIMIT_TOKEN_MARGIN,
) -> Dict[str, Any]:
    """Describe a token-only output-limit decision for logs and artifacts."""

    distribution = extract_token_distribution(raw_response)
    visible_tokens = int(distribution.get("output_tokens") or 0)
    thinking_tokens = int(distribution.get("thinking_tokens") or 0)
    observed_tokens = visible_tokens + thinking_tokens
    configured_limit = max(0, int(max_tokens))
    configured_margin = max(0, int(margin))
    threshold_tokens = max(0, configured_limit - configured_margin)
    return {
        "basis": "output_tokens_plus_thinking_tokens",
        "visible_output_tokens": visible_tokens,
        "thinking_tokens": thinking_tokens,
        "observed_output_tokens": observed_tokens,
        "max_output_tokens": configured_limit,
        "margin_tokens": configured_margin,
        "threshold_tokens": threshold_tokens,
        "limited": observed_tokens > 0 and observed_tokens >= threshold_tokens,
    }


def _is_output_limited(
    raw_response: Any,
    max_tokens: int = DEFAULT_LIMITS.output_limit,
    margin: int = OUTPUT_LIMIT_TOKEN_MARGIN,
) -> bool:
    """Return whether provider-reported output usage reached the token cap."""

    return bool(_output_limit_check(raw_response, max_tokens, margin)["limited"])


TASK_UPDATE_FEEDBACK_RE = re.compile(
    r"<task_update_feedback\b[^>]*>(?P<body>.*?)</task_update_feedback>",
    re.IGNORECASE | re.DOTALL,
)
NEXT_ADVICE_RE = re.compile(
    r"<next_advice\b[^>]*>(?P<body>.*?)</next_advice>",
    re.IGNORECASE | re.DOTALL,
)


def _extract_task_update_feedback(
    text: str,
    *,
    max_tokens: int = 4_000,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    match = TASK_UPDATE_FEEDBACK_RE.search(text or "")
    if not match:
        return ""
    body = match.group("body").strip()
    return cap_tokens(body, max_tokens, count_tokens, marker="\n...[truncated]")


# Per-window advice cap: advice is now cumulative across windows, so each
# window only contributes its incremental notes (prompt states the same limit).


def _extract_next_advice(
    text: str,
    *,
    max_tokens: int = NEXT_ADVICE_MAX_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    match = NEXT_ADVICE_RE.search(text or "")
    if not match:
        return ""
    body = match.group("body").strip()
    return cap_tokens(body, max_tokens, count_tokens)


def _window_audio_label(
    audio_path: str | Path | None,
    audio_label: str,
    window: SubtitleWindow,
    *,
    clip_suffix: str = CLIP_AUDIO_SUFFIX,
) -> str:
    if not audio_path:
        return audio_label
    base = Path(audio_path).name
    return (
        f"{window.chunk_id}{clip_suffix}（{base} 的 "
        f"[{format_srt_time(window.clip_start)} - {format_srt_time(window.clip_end)}] "
        f"剪辑，含前后 padding）"
    )


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
    client: LiteLLMRoleClient,
    window: SubtitleWindow,
    context_pack: ContextPack | None,
    audio_label: str,
    previous_advice: str,
    file_ref: UploadedFileRef | None,
    search_client: WebSearchClient,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    streamer_index: str = "",
    common_index: str = "",
    max_queries: int = MAX_WINDOW_SEARCH_QUERIES,
    carried_entries_text: str = "",
    carried_key_count: int = 0,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    token_rows: List[Dict[str, Any]] | None = None,
    exchange_logger: ExchangeLogger | None = None,
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
            "max_tokens": QUERY_ROUND_MAX_TOKENS,
            "file_backed": bool(profile.use_audio and file_ref is not None),
        },
        extra_identity=checkpoint_extra_identity,
    )
    cached = checkpoint_store.get("query", window.chunk_id, checkpoint_hash)
    cached_valid = False
    if cached is not None and not SESSION_CONTRACTS["query"].validate(cached.content):
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
            max_tokens=QUERY_ROUND_MAX_TOKENS,
            file_ref=file_ref if profile.use_audio else None,
            **(validation_retry_sampling_kwargs(query_attempt) if query_attempt else {}),
        )
        if is_prompt_blocked(result.content, result.raw_response):
            raise GeminiPromptBlockedError(
                f"窗口 {window.chunk_id} 查询轮 prompt was blocked by the content filter"
            )
        return result

    from ..output_tags import missing_top_level_tags

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
                        },
                    )
                return QueryRoundProduct()
            # A missing first-level block means a sibling was swallowed inside
            # another block (structural corruption) — retry. Nesting itself is
            # fine (e.g. <reasoning> may name-drop other tags).
            tag_errors = missing_top_level_tags(
                result.content, list(_QUERY_REQUIRED_TOP_LEVEL)
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
                    },
                )
            query_attempt += 1
    finish_reason = _response_finish_reason(result.raw_response)
    output_limit_check = _output_limit_check(
        result.raw_response, QUERY_ROUND_MAX_TOKENS
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
    input_components = correction_input_components(
        window=window,
        counter=token_counter,
        context_general=pack.general_prompt_text(),
        context_window=pack.window_context_for(window.chunk_id),
        messages=messages,
        max_output_tokens=QUERY_ROUND_MAX_TOKENS,
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
    checkpoint_valid = not SESSION_CONTRACTS["query"].validate(result.content)
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
    elif checkpoint_valid and not parse_error and not output_limited:
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


# Mid-loop resume: each successfully committed window's raw response is appended
# here so a rerun replays completed windows instead of re-calling the LLM.
WINDOW_CACHE_FILENAME = "correction-windows.jsonl"


def _task_fingerprint(
    *,
    extra_style: str,
    common_mistakes_block: str,
    context_pack: ContextPack | None,
    test_profile: bool,
    task_update_feedback: bool,
    profile: TranslationProfile = DEFAULT_PROFILE,
    entry_details: str = "",
    extra: str = "",
    media_identity: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint the shared prompt inputs that affect every window.

    A rerun with a different prompt version, context, style, mistakes ledger,
    route/level profile, output scale, source media identity or fast-mode
    injection seed (``extra``) gets a different fingerprint, so stale cached
    windows are ignored.
    """

    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "extra_style": extra_style or "",
            "common_mistakes_block": common_mistakes_block or "",
            "context_pack": context_pack.to_dict() if context_pack else {},
            "test_profile": bool(test_profile),
            "task_update_feedback": bool(task_update_feedback),
            "profile_id": profile.profile_id,
            "output_scale": profile.output_scale,
            "entry_details": entry_details or "",
            "extra": extra or "",
            "media_identity": dict(media_identity or {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _media_identity(path: str | Path | None) -> Dict[str, Any]:
    """Return a stable-enough local source identity for resume invalidation."""

    if path is None:
        return {}
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _window_input_hash(window: SubtitleWindow, entry_details_sig: str = "") -> str:
    """Hash a window's exact model input (ids, timings, text, clip origin,
    read-only preceding context, plus all injected entry keys+bodies)."""

    text = render_window_segments_as_csv(window)
    preceding = render_window_preceding_as_csv(window)
    return (
        "sha256:"
        + hashlib.sha256(
            f"{preceding}\x1f{text}\x1f{entry_details_sig}".encode("utf-8")
        ).hexdigest()[:16]
    )


def _load_window_cache(path: Path, task_fingerprint: str) -> Dict[str, Dict[str, Any]]:
    """Load cached window records matching the task fingerprint (last wins)."""

    if not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("task_fingerprint") != task_fingerprint:
            continue
        chunk_id = record.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            cache[chunk_id] = record
    return cache


def _append_window_cache(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.part{path.suffix}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


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
    enable_web_search: bool = True,
    search_client: WebSearchClient | None = None,
    max_search_queries_per_window: int = MAX_WINDOW_SEARCH_QUERIES,
    postprocess_profile: int | None = DEFAULT_POSTPROCESS_PROFILE,
    extra_style: str = "",
    common_mistakes_block: str = "",
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    task_update_feedback: bool = False,
    token_counter: TokenCounter | None = None,
    resume: bool = True,
    profile: TranslationProfile = DEFAULT_PROFILE,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    windows_override: List[SubtitleWindow] | None = None,
    seed_query_results: Mapping[str, "QueryRoundProduct"] | None = None,
    entry_details: str = "",
    evidence_pack_mode: bool = False,
    file_ref_seed: Mapping[str, UploadedFileRef] | None = None,
    extra_fingerprint: str = "",
    initial_transfer_keys: Sequence[str] = (),
) -> Path:
    """Execute the correction windows (planned here unless overridden).

    Fast mode passes ``windows_override`` (the single fused window),
    ``seed_query_results`` (round-1 products keyed by base chunk id, replacing
    the per-window query round), ``entry_details`` / ``evidence_pack_mode``
    (round-2 injections), ``file_ref_seed`` (the round-1 clip upload, reused
    instead of re-uploading) and ``extra_fingerprint`` (folds the seeded
    injections into the resume fingerprint). ``knowledge_root`` feeds the
    query round's knowledge-index exposure and entry requests.

    ``video_path`` (mm-high only): the correction round takes a low-res
    video+audio ``.mp4`` clip instead of the ``.aac``; the ``.aac`` still
    feeds the per-window query round (audio-only by design), cut on demand.
    """

    segments = load_segments_from_stable_json(stable_json)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_srt_path = out.with_name(f"{out.stem}-raw.srt")
    _write_text_atomic(raw_srt_path, render_segments_as_srt(segments))

    token_counter = token_counter or default_token_counter()
    audio_duration = probe_audio_duration(audio_path) if audio_path else None
    if windows_override is not None:
        windows = list(windows_override)
    else:
        plan_report: Dict[str, Any] = {}
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
                payload={"phase": "correction", **plan_report},
            )
    global_first_id = segments[0].id if segments else ""
    global_last_id = segments[-1].id if segments else ""
    clip_base_dir = (
        Path(clip_dir)
        if clip_dir
        else Path("tmp") / "llm-audio-clips" / Path(stable_json).stem
    )
    # One clip + upload per executed window (exact chunk id: -a/-b halves get
    # their own clips); reused across the query round, the correction round
    # and same-window validation retries. Extraction + upload run on a
    # background thread: window 0 is scheduled after planning; at the start of
    # window i we schedule i+1 so the main loop rarely waits on ffmpeg.
    clip_prefetcher: WindowClipPrefetcher | None = None
    video_prefetcher: WindowClipPrefetcher | None = None
    use_video = bool(video_path) and profile.use_video
    if audio_path and profile.use_audio:

        def _tracked_upload(path: Path) -> UploadedFileRef:
            ref = upload_gemini_file(path)
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

        clip_prefetcher = WindowClipPrefetcher(
            audio_path,
            clip_base_dir,
            extract_fn=extract_window_clip,
            upload_fn=_tracked_upload,
        )
        if use_video:
            video_prefetcher = WindowClipPrefetcher(
                video_path,
                clip_base_dir,
                extract_fn=extract_window_video_clip,
                upload_fn=_tracked_upload,
                clip_suffix=CLIP_VIDEO_SUFFIX,
            )
    # The prefetch pipeline follows the correction round's media: the mp4 on
    # mm-high, the .aac otherwise. On mm-high the .aac only feeds the query
    # round (one per base window) and is cut synchronously when it runs.
    correction_prefetcher = video_prefetcher or clip_prefetcher
    if (
        correction_prefetcher is not None
        and windows
        and not (file_ref_seed and windows[0].chunk_id in file_ref_seed)
    ):
        correction_prefetcher.schedule(windows[0])

    def ensure_window_media_ref(win: SubtitleWindow) -> UploadedFileRef | None:
        if file_ref_seed and win.chunk_id in file_ref_seed:
            return file_ref_seed[win.chunk_id]
        if correction_prefetcher is None:
            return None
        return correction_prefetcher.get_ref(win)

    client = LiteLLMRoleClient(test_profile=test_profile)
    # The per-window query round and injected search only exist on the mm
    # route; the text route never runs harness-side retrieval.
    external_injection = profile.external_injection
    if (
        external_injection
        and enable_web_search
        and search_client is None
        and not seed_query_results
    ):
        search_client = WebSearchClient()
    # Query round output is fetched once per planned window (keyed by the base
    # chunk id) and reused across validation retries and -a/-b split halves.
    # Fast mode seeds it with round-1 products so no query round ever runs.
    query_round_cache: Dict[str, QueryRoundProduct] = dict(seed_query_results or {})
    streamer_index_text = load_index_text(knowledge_root, "streamer")
    common_index_text = load_index_text(knowledge_root, "common")
    content_filter_blacklist = load_content_filter_blacklist(task_artifact_dir)
    # v17 entry pass-through: canonical keys kept by the previous step (the
    # research round 2 seeds window one; each window's correction round emits
    # <keep_entries> for the next). The set is injected into both the query
    # round (context) and the correction round, and its keys+bodies enter the
    # per-window resume input hash.
    transfer_keys: List[str] = list(dict.fromkeys(initial_transfer_keys))[
        :KB_TRANSFER_MAX_ENTRIES
    ]

    def _load_transfer_entries() -> Dict[str, str]:
        if not transfer_keys:
            return {}
        found, _missing = load_entry_texts(knowledge_root, transfer_keys)
        return found

    def _entry_details_signature(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def _render_cached_entry_details(keys: Sequence[str]) -> str:
        """Rebuild a cached window's entry block from current KB contents."""

        if not keys:
            return ""
        found, _missing = load_entry_texts(knowledge_root, keys)
        ordered = {
            key: found[key]
            for key in keys
            if key in found
        }
        if not ordered:
            return ""
        return render_knowledge_entries_block(
            ordered,
            count_tokens=token_counter.count_text,
            entry_limit=INJECTION_SECTION_MAX_TOKENS,
            block_limit=injection_block_token_limit(KB_WINDOW_TOTAL_ENTRIES),
        ).text
    token_rows: List[Dict[str, Any]] = []
    exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    rendered_segments: List[TranslatedCsvSegment] = []
    # Cumulative (chunk_id, advice) ledger: every window sees all earlier
    # windows' advice, rendered with window labels.
    advice_ledger: List[tuple[str, str]] = []

    # Mid-loop resume: cache each committed window so a rerun replays completed
    # windows instead of re-calling the LLM. Requires a task artifact dir.
    task_fingerprint = _task_fingerprint(
        extra_style=extra_style,
        common_mistakes_block=common_mistakes_block,
        context_pack=context_pack,
        test_profile=test_profile,
        task_update_feedback=task_update_feedback,
        profile=profile,
        entry_details=entry_details,
        extra=extra_fingerprint,
        media_identity={
            "audio": _media_identity(audio_path),
            "video": _media_identity(video_path),
        },
    )
    window_cache_path = (
        Path(task_artifact_dir) / WINDOW_CACHE_FILENAME if task_artifact_dir else None
    )
    resume_enabled = bool(resume and window_cache_path is not None)
    resume_cache = (
        _load_window_cache(window_cache_path, task_fingerprint) if resume_enabled else {}
    )
    # Reuse the parsed append-only ledger across windows. Constructing a store
    # per query would repeatedly rescan the same JSONL file on long tasks.
    session_checkpoint_store = SessionCheckpointStore(
        task_artifact_dir, enabled=resume
    )

    def _commit_window(
        current: SubtitleWindow,
        validation: CsvValidationResult,
        next_advice: str,
    ) -> None:
        """Fold one window's result into the accumulated output + advice ledger.

        Continuity for later windows is input-only since v13 (the read-only
        preceding-context block planned into each window), so committing no
        longer feeds any prompt state besides the advice ledger.
        """

        nonlocal rendered_segments
        rendered_segments = merge_translated_csv_windows(
            rendered_segments, current.source_ids, validation.segments
        )
        if next_advice.strip():
            advice_ledger.append((current.chunk_id, next_advice))

    i = 0
    try:
        while i < len(windows):
            window = windows[i]
            if correction_prefetcher is not None:
                correction_prefetcher.prefetch_next(windows, i)
            current = window
            transfer_found = _load_transfer_entries()
            if resume_enabled and current.chunk_id in resume_cache:
                record = resume_cache[current.chunk_id]
                cached_keys = [
                    str(key)
                    for key in (record.get("injected_entries") or [])
                    if str(key).strip()
                ]
                cached_entry_details = (
                    entry_details
                    if entry_details
                    else _render_cached_entry_details(cached_keys)
                )
                cached_entry_sig = _entry_details_signature(cached_entry_details)
                if record.get("input_hash") == _window_input_hash(
                    current, cached_entry_sig
                ):
                    cached_content = str(record.get("content") or "")
                    cached_tier = CapabilityTier(
                        record.get("capability_tier", CapabilityTier.CAPABLE.value)
                    )
                    cached_variant = resolve_variant(None, cached_tier)
                    cached_validation = validate_correction_window_output(
                        cached_content,
                        current,
                        variant=cached_variant,
                        allow_insert=False,
                    )
                    if cached_validation.ok:
                        _commit_window(
                            current,
                            cached_validation,
                            _extract_next_advice(
                                cached_content, count_tokens=token_counter.count_text
                            ),
                        )
                        if task_artifact_dir:
                            append_task_artifact(
                                task_artifact_dir,
                                kind="correction_window_cached",
                                task_id=task_id,
                                payload={
                                    "chunk_id": current.chunk_id,
                                    "source_ids": list(current.source_ids),
                                    "row_count": len(cached_validation.segments),
                                },
                            )
                        # Replay continues the transfer chain from the cached
                        # keep list so later windows hash consistently.
                        transfer_keys = [
                            key
                            for key in (record.get("keep_entries") or [])
                            if isinstance(key, str) and key
                        ][:KB_TRANSFER_MAX_ENTRIES]
                        i += 1
                        continue
            for attempt in range(max_retries_per_window + 1):
                window_file_ref = ensure_window_media_ref(current)
                window_audio_label = _window_audio_label(
                    video_path if use_video else audio_path,
                    audio_label,
                    current,
                    clip_suffix=CLIP_VIDEO_SUFFIX if use_video else CLIP_AUDIO_SUFFIX,
                )
                # Cumulative ledger cap: keep the most recent windows' advice
                # (front-truncated) within the token budget.
                previous_advice = truncate_text_only(
                    render_advice_ledger(advice_ledger),
                    ADVICE_LEDGER_MAX_TOKENS,
                    token_counter.count_text,
                    keep="tail",
                    heuristic_count=token_counter.count_text,
                    prefer_natural_boundary=True,
                )
                query_product = QueryRoundProduct()
                if external_injection and enable_web_search:
                    base_chunk_id = current.chunk_id.split("-", 1)[0]
                    if base_chunk_id not in query_round_cache and search_client is None:
                        query_round_cache[base_chunk_id] = QueryRoundProduct()
                    if base_chunk_id not in query_round_cache:
                        # The query round stays audio-only on mm-high: cut the
                        # .aac on demand instead of sending the video clip to
                        # the lite model.
                        query_file_ref = (
                            clip_prefetcher.get_ref(current)
                            if use_video and clip_prefetcher is not None
                            else window_file_ref
                        )
                        carried_text = ""
                        if transfer_found:
                            carried_block = render_knowledge_entries_block(
                                transfer_found,
                                count_tokens=token_counter.count_text,
                                entry_limit=INJECTION_SECTION_MAX_TOKENS,
                                block_limit=injection_block_token_limit(
                                    KB_TRANSFER_MAX_ENTRIES
                                ),
                            )
                            carried_text = carried_block.text
                        query_round_cache[base_chunk_id] = run_window_query_round(
                            client=client,
                            window=current,
                            context_pack=context_pack,
                            audio_label=_window_audio_label(
                                audio_path, audio_label, current
                            ),
                            previous_advice=previous_advice,
                            file_ref=query_file_ref,
                            search_client=search_client,
                            knowledge_root=knowledge_root,
                            streamer_index=streamer_index_text,
                            common_index=common_index_text,
                            carried_entries_text=carried_text,
                            carried_key_count=len(transfer_found),
                            max_queries=max_search_queries_per_window,
                            task_artifact_dir=task_artifact_dir,
                            task_id=task_id,
                            token_rows=token_rows,
                            exchange_logger=exchange_logger,
                            token_counter=token_counter,
                            profile=profile,
                            resume=resume,
                            checkpoint_store=session_checkpoint_store,
                            checkpoint_extra_identity={
                                "task_fingerprint": task_fingerprint,
                            },
                        )
                    query_product = query_round_cache[base_chunk_id]
                # v17: one unified render — transfers first (they win the
                # budget), then this window's new requests, capped to the
                # window total. The fast-mode global entry_details (single
                # window) still takes precedence unchanged.
                injected_keys: List[str] = []
                window_entry_details = entry_details
                if not window_entry_details:
                    union: Dict[str, str] = dict(transfer_found)
                    fresh_keys = [
                        key
                        for key in query_product.requested_entry_keys
                        if key not in union
                    ]
                    if fresh_keys:
                        fresh_found, _fresh_missing = load_entry_texts(
                            knowledge_root, fresh_keys
                        )
                        for key, body in fresh_found.items():
                            if key not in union:
                                union[key] = body
                    union = dict(list(union.items())[:KB_WINDOW_TOTAL_ENTRIES])
                    injected_keys = list(union)
                    if union:
                        window_entry_block = render_knowledge_entries_block(
                            union,
                            count_tokens=token_counter.count_text,
                            entry_limit=INJECTION_SECTION_MAX_TOKENS,
                            block_limit=injection_block_token_limit(
                                KB_WINDOW_TOTAL_ENTRIES
                            ),
                        )
                        window_entry_details = window_entry_block.text
                window_entry_sig = _entry_details_signature(window_entry_details)
                # Content-filter ladder is independent of validation retries:
                # drop injected retrieval units until the prompt clears, then
                # validate. A contaminated search injection is rewritten into
                # the query-round cache so -a/-b halves and later attempts see
                # the cleaned text.
                if evidence_pack_mode and query_product.search_results.strip():
                    search_block = evidence_pack_block(query_product.search_results)
                    window_evidence_mode = True
                else:
                    search_block = split_rendered_search_block(
                        query_product.search_results
                    )
                    window_evidence_mode = False

                def _correction_call(search_text: str):
                    use_pack = (
                        window_evidence_mode
                        and bool(search_text.strip())
                        and search_text.strip() == query_product.search_results.strip()
                    )

                    def compose_for_tier(tier: CapabilityTier):
                        return build_correction_csv_messages(
                            window=current,
                            context_pack=context_pack,
                            audio_file_label=window_audio_label,
                            previous_advice=previous_advice,
                            query_round_notes=query_product.window_notes,
                            search_results=search_text,
                            entry_details=window_entry_details,
                            extra_style=extra_style,
                            common_mistakes_block=common_mistakes_block,
                            task_update_feedback=task_update_feedback,
                            evidence_pack_mode=use_pack,
                            profile=profile,
                            tier=tier,
                        )

                    call_result = client.complete(
                        correction_role_for_profile(profile),
                        compose_for_tier,
                        max_tokens=DEFAULT_LIMITS.output_limit,
                        file_ref=window_file_ref,
                        **validation_retry_sampling_kwargs(attempt),
                        **_thinking_override_kwargs(profile),
                    )
                    if is_prompt_blocked(call_result.content, call_result.raw_response):
                        raise GeminiPromptBlockedError(
                            f"窗口 {current.chunk_id} prompt was blocked by "
                            "the content filter"
                        )
                    # Re-assemble the messages the answering endpoint's tier
                    # actually received, for artifacts and exchange logs.
                    call_messages = compose_for_tier(call_result.capability_tier)
                    return call_result, call_messages

                try:
                    ladder_outcome = run_injection_ladder(
                        block=search_block,
                        call=_correction_call,
                        stage=f"窗口 {current.chunk_id}",
                        blocked_exception=GeminiPromptBlockedError,
                        blacklist=content_filter_blacklist,
                        task_artifact_dir=task_artifact_dir,
                        task_id=task_id,
                        plain_retry=not search_block.units,
                    )
                except Exception as exc:  # pragma: no cover - provider behavior
                    if isinstance(exc, ContentFilterExhaustedError):
                        raise
                    if task_artifact_dir:
                        # Failure-path request snapshot for the error artifact
                        # (never sent): complete() raised, so no LLMCallResult
                        # carries the answering tier — assemble at the default
                        # CAPABLE tier.
                        err_messages = build_correction_csv_messages(
                            window=current,
                            context_pack=context_pack,
                            audio_file_label=window_audio_label,
                            previous_advice=previous_advice,
                            query_round_notes=query_product.window_notes,
                            search_results=query_product.search_results,
                            entry_details=window_entry_details,
                            extra_style=extra_style,
                            common_mistakes_block=common_mistakes_block,
                            task_update_feedback=task_update_feedback,
                            evidence_pack_mode=evidence_pack_mode,
                            profile=profile,
                        )
                        append_task_artifact(
                            task_artifact_dir,
                            kind="correction_window_call_error",
                            task_id=task_id,
                            payload={
                                "chunk_id": current.chunk_id,
                                "attempt": attempt,
                                "window": window_to_metadata(current),
                                "request": _request_reference_metadata(
                                    messages=err_messages,
                                    file_ref=window_file_ref,
                                    max_tokens=DEFAULT_LIMITS.output_limit,
                                ),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "api_attempts": list(
                                    getattr(exc, "_harness_api_attempts", []) or []
                                ),
                            },
                        )
                    raise
                result, messages = ladder_outcome.result
                cleaned_search = search_block.render(
                    [
                        unit
                        for unit in search_block.units
                        if unit.content_hash not in content_filter_blacklist
                    ]
                )
                if cleaned_search != query_product.search_results:
                    query_product = QueryRoundProduct(
                        search_results=cleaned_search,
                        window_notes=query_product.window_notes,
                        requested_entry_keys=query_product.requested_entry_keys,
                    )
                    base_chunk_id = current.chunk_id.split("-", 1)[0]
                    query_round_cache[base_chunk_id] = query_product
                request_reference = _request_reference_metadata(
                    messages=messages,
                    file_ref=window_file_ref,
                    max_tokens=DEFAULT_LIMITS.output_limit,
                )
                response_variant = resolve_variant(None, result.capability_tier)
                validation = validate_correction_window_output(
                    result.content,
                    current,
                    variant=response_variant,
                    allow_insert=False,
                )
                validation_ok = validation.ok
                finish_reason = _response_finish_reason(result.raw_response)
                output_limit_check = _output_limit_check(
                    result.raw_response,
                    DEFAULT_LIMITS.output_limit,
                )
                output_limited = bool(output_limit_check["limited"])
                token_rows.append(
                    {
                        "call": "correction_window",
                        "chunk_id": current.chunk_id,
                        "attempt": attempt,
                        "model": result.model,
                        "capability_tier": result.capability_tier.value,
                        "finish_reason": finish_reason,
                        "output_limit_check": output_limit_check,
                        "tokens": extract_token_distribution(result.raw_response),
                    }
                )
                pack = context_pack or ContextPack()
                window_input_components = correction_input_components(
                    window=current,
                    counter=token_counter,
                    search_results=query_product.search_results,
                    context_general=pack.general_prompt_text(),
                    context_window=pack.window_context_for(current.chunk_id),
                    messages=messages,
                    max_output_tokens=DEFAULT_LIMITS.output_limit,
                    use_video=use_video,
                )
                window_session = f"correction-{current.chunk_id}-attempt{attempt}"
                if exchange_logger:
                    exchange_logger.log(
                        window_session,
                        messages=messages,
                        response_text=result.content,
                        metadata=llm_exchange_metadata(
                            result,
                            session=window_session,
                            input_components=window_input_components,
                            chunk_id=current.chunk_id,
                            attempt=attempt,
                            capability_tier=result.capability_tier.value,
                            finish_reason=finish_reason,
                            validation_ok=validation.ok,
                            output_limited=output_limited,
                            output_limit_basis=output_limit_check["basis"],
                            output_limit_observed_tokens=output_limit_check[
                                "observed_output_tokens"
                            ],
                            output_limit_threshold_tokens=output_limit_check[
                                "threshold_tokens"
                            ],
                            output_limit_max_tokens=output_limit_check[
                                "max_output_tokens"
                            ],
                            output_limit_margin_tokens=output_limit_check[
                                "margin_tokens"
                            ],
                        ),
                    )
                update_feedback = (
                    remap_feedback_source_ids(
                        _extract_task_update_feedback(
                            result.content, count_tokens=token_counter.count_text
                        ),
                        WindowIdMap.from_window(current),
                    )
                    if task_update_feedback
                    else ""
                )
                next_advice = _extract_next_advice(
                    result.content, count_tokens=token_counter.count_text
                )
                # v17: the correction round's <keep_entries> is the transfer
                # chain's only continuation point — canonicalized against the
                # entries actually injected this window, capped, and handed to
                # the next window (absent/empty block drops the chain).
                # The query round's keep_entry_keys also contributes (union).
                keep_raw: List[str] = []
                try:
                    keep_raw = parse_line_items(
                        extract_single_tag_block(
                            result.content, "keep_entries", required=False
                        )
                    )
                except ValueError:
                    keep_raw = []
                if query_product.keep_entry_keys:
                    keep_raw = list(
                        dict.fromkeys(keep_raw + list(query_product.keep_entry_keys))
                    )
                next_transfer: List[str] = []
                if keep_raw and injected_keys:
                    keep_found, _keep_missing = load_entry_texts(
                        knowledge_root, keep_raw
                    )
                    injected_set = set(injected_keys)
                    next_transfer = [
                        key for key in keep_found if key in injected_set
                    ][:KB_TRANSFER_MAX_ENTRIES]
                if task_artifact_dir:
                    response_payload = {
                        "session": window_session,
                        "chunk_id": current.chunk_id,
                        "attempt": attempt,
                        "model": result.model,
                        "fallback_used": result.fallback_used,
                        "finish_reason": finish_reason,
                        "output_limited": output_limited,
                        "output_limit_check": output_limit_check,
                        "validation_ok": validation_ok,
                        "validation_errors": validation.errors,
                        "validation_warnings": validation.warnings,
                        "voided_rows": validation.voided_rows,
                        # v15 phase 1: pacing score is observational only —
                        # recorded for threshold calibration, never rejects.
                        # Test profile relaxes the pass ratio to 1.0 (surface
                        # problems for prompt iteration instead of failing).
                        "pacing_score": (
                            score_translated_segments(
                                validation.segments,
                                pass_ratio=(
                                    PACING_PASS_RATIO_TEST_PROFILE
                                    if test_profile
                                    else PACING_PASS_RATIO
                                ),
                            )
                            if validation_ok
                            else None
                        ),
                        "next_advice": next_advice,
                        "injected_entries": injected_keys,
                        "keep_entries": next_transfer,
                        "usage": extract_token_distribution(result.raw_response),
                        "input_components": window_input_components,
                        "window": window_to_metadata(current),
                        "request": request_reference,
                        "provider": _provider_reference_metadata(result.raw_response),
                        "response": _response_reference_metadata(result.content),
                        "response_content": result.content,
                    }
                    if task_update_feedback:
                        response_payload["task_update_feedback"] = update_feedback
                    append_task_artifact(
                        task_artifact_dir,
                        kind="correction_window_response",
                        task_id=task_id,
                        payload=response_payload,
                    )
                    if update_feedback:
                        append_task_artifact(
                            task_artifact_dir,
                            kind="correction_window_task_feedback",
                            task_id=task_id,
                            payload={
                                "chunk_id": current.chunk_id,
                                "attempt": attempt,
                                "window": window_to_metadata(current),
                                "feedback": update_feedback,
                            },
                        )
                if validation_ok and not output_limited:
                    _commit_window(current, validation, next_advice)
                    if resume_enabled:
                        _append_window_cache(
                            window_cache_path,
                            {
                                "chunk_id": current.chunk_id,
                                "source_ids": list(current.source_ids),
                                "clip_start": round(current.clip_start, 3),
                                "input_hash": _window_input_hash(
                                    current, window_entry_sig
                                ),
                                "task_fingerprint": task_fingerprint,
                                "content": result.content,
                                "capability_tier": result.capability_tier.value,
                                "injected_entries": injected_keys,
                                "keep_entries": next_transfer,
                            },
                        )
                    transfer_keys = next_transfer
                    break
                if attempt >= max_retries_per_window:
                    errors = "; ".join(validation.errors) or "output appears truncated"
                    raise RuntimeError(f"Window {window.chunk_id} failed validation: {errors}")
                failed_window = current
                second_half: SubtitleWindow | None = None
                if output_limited:
                    halves = split_window_in_half(
                        current,
                        counter=token_counter,
                        global_first_id=global_first_id,
                        global_last_id=global_last_id,
                        audio_duration=audio_duration,
                        profile=profile,
                    )
                    if halves is None:
                        retry_reason = "output_limited_unsplittable_same_window"
                    else:
                        current, second_half = halves
                        retry_reason = "output_limited_split_in_half"
                else:
                    retry_reason = "validation_same_window"
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="correction_window_retry",
                        task_id=task_id,
                        payload={
                            "chunk_id": window.chunk_id,
                            "attempt": attempt,
                            "reason": retry_reason,
                            "finish_reason": finish_reason,
                            "output_limit_check": output_limit_check,
                            "failed_window": window_to_metadata(failed_window),
                            "retry_chunk_id": current.chunk_id,
                            "retry_window": window_to_metadata(current),
                            "tail_chunk_ids": (
                                [second_half.chunk_id] if second_half is not None else []
                            ),
                            "tail_windows": (
                                [window_to_metadata(second_half)]
                                if second_half is not None
                                else []
                            ),
                        },
                    )
                if second_half is not None:
                    windows.insert(i + 1, second_half)
            else:  # pragma: no cover
                raise RuntimeError(f"Window {window.chunk_id} failed unexpectedly.")
            i += 1

    finally:
        if clip_prefetcher is not None:
            clip_prefetcher.shutdown()
        if video_prefetcher is not None:
            video_prefetcher.shutdown()

    merged = render_translated_segments_as_srt(rendered_segments)
    corrected = render_corrected_segments_as_srt(rendered_segments)
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
        + render_translated_segments_as_csv(rendered_segments),
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
                "rows": token_rows,
                "totals": sum_token_distributions(row["tokens"] for row in token_rows),
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
