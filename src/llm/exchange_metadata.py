"""Exchange header metadata: input-component estimates and session usage."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from .chunking import SubtitleWindow, render_window_segments_as_csv
# Canonical top-level (non-nested) sibling extraction lives in output_tags.
from .output_tags import find_top_level_tag_blocks as extract_top_level_tagged_blocks
from .profiles import video_tokens_per_second
from .token_budget import TokenCounter

_EMPTY_MARKERS = frozenset({"", "（无）", "（空）"})

SESSION_RESPONSE_KINDS = frozenset(
    {
        "research_round1_response",
        "fast_round1_response",
        "research_round2_response",
        "search_loop_round",
        "correction_query_response",
        "correction_window_response",
        "knowledge_update_response",
    }
)


def extract_tagged_block(text: str, tag: str) -> str:
    """Return the body of the first top-level ``<tag>...</tag>`` block."""
    blocks = extract_top_level_tagged_blocks(text, tag)
    return blocks[0] if blocks else ""


def infer_session_name(kind: str, payload: Mapping[str, Any]) -> str:
    if payload.get("session"):
        return str(payload["session"])
    if kind == "research_round1_response":
        return f"research-round1-attempt{payload.get('attempt', 0)}"
    if kind == "fast_round1_response":
        return f"fast-round1-attempt{payload.get('attempt', 0)}"
    if kind == "research_round2_response":
        return f"research-round2-attempt{payload.get('attempt', 0)}"
    if kind == "search_loop_round":
        return (
            f"research-search-loop-round{payload.get('round', 0)}"
            f"-attempt{payload.get('attempt', 0)}"
        )
    if kind == "correction_query_response":
        return (
            f"correction-{payload.get('chunk_id', '?')}-query"
            f"-attempt{payload.get('attempt', 0)}"
        )
    if kind == "correction_window_response":
        return f"correction-{payload.get('chunk_id', '?')}-attempt{payload.get('attempt', 0)}"
    if kind == "knowledge_update_response":
        return f"knowledge-update-chunk{payload.get('chunk', 0):02d}"
    return kind


def normalize_session_usage(usage: Mapping[str, Any]) -> Dict[str, int]:
    """Normalize provider usage into report/session totals."""

    thinking = int(usage.get("thinking_tokens") or 0)
    visible = int(usage.get("output_tokens") or 0)
    total_input = int(usage.get("total_input_tokens") or usage.get("prompt_tokens") or 0)
    total_output = int(usage.get("total_output_tokens") or 0)
    if not total_output:
        total_output = visible + thinking
    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "thinking_tokens": thinking,
        "output_tokens": visible,
    }


def _user_message_text(messages: List[Mapping[str, Any]] | None) -> str:
    if not messages:
        return ""
    for message in reversed(messages):
        if str(message.get("role", "")) == "user":
            content = message.get("content", "")
            return str(content) if content is not None else ""
    return ""


def _is_empty_injection(text: str) -> bool:
    return not (text or "").strip() or text.strip() in _EMPTY_MARKERS


def count_text_tokens(counter: TokenCounter | None, text: str) -> int:
    if _is_empty_injection(text):
        return 0
    if counter is None:
        return max(0, len(text) // 2)
    return counter.count_text(text)


def research_input_components(
    *,
    counter: TokenCounter | None = None,
    transcript: str = "",
    note_url_extracts: str = "",
    entry_details: Mapping[str, str] | None = None,
    search_results: str = "",
    messages: List[Mapping[str, Any]] | None = None,
) -> Dict[str, int]:
    user_text = _user_message_text(messages)
    if not transcript and user_text:
        transcript = extract_tagged_block(user_text, "transcript")
    if not note_url_extracts and user_text:
        note_url_extracts = extract_tagged_block(user_text, "note_url_extracts")

    knowledge_text = ""
    if entry_details:
        knowledge_text = "\n\n".join(
            value.strip() for value in entry_details.values() if value
        )
    elif user_text:
        knowledge_text = extract_tagged_block(user_text, "knowledge_entries")

    search_text = search_results
    if _is_empty_injection(search_text) and user_text:
        search_text = extract_tagged_block(user_text, "search_results")

    return {
        "transcript_input_tokens": count_text_tokens(counter, transcript),
        "extra_info_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "extra_info")
        ),
        "note_url_extract_tokens": count_text_tokens(counter, note_url_extracts),
        "streamer_index_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "streamer_index")
        ),
        "common_index_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "common_index")
        ),
        "preinjected_entry_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "preinjected_entries")
        ),
        "round1_notes_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "round1_notes")
        ),
        "knowledge_injection_tokens": count_text_tokens(counter, knowledge_text),
        "search_injection_tokens": count_text_tokens(counter, search_text),
    }


def correction_input_components(
    *,
    window: SubtitleWindow,
    counter: TokenCounter | None = None,
    search_results: str = "",
    context_general: str = "",
    context_window: str = "",
    messages: List[Mapping[str, Any]] | None = None,
    max_output_tokens: int | None = None,
    use_video: bool = False,
) -> Dict[str, Any]:
    user_text = _user_message_text(messages)
    if _is_empty_injection(search_results) and user_text:
        search_results = extract_tagged_block(user_text, "search_results")
    if _is_empty_injection(context_window) and user_text:
        context_window = extract_tagged_block(user_text, "window_context")

    knowledge_text = "\n\n".join(
        part
        for part in (context_general, context_window)
        if not _is_empty_injection(part)
    )
    csv_text = render_window_segments_as_csv(window)
    clip_seconds = max(0.0, window.clip_end - window.clip_start)
    # Planning-estimate of the media (audio + optional low-res video) token
    # cost; real billing comes from provider usage metadata. v17: renamed from
    # audio_input_tokens and extended with the video estimate on video calls.
    media_tokens = (
        counter.count_audio_seconds(clip_seconds)
        if counter is not None
        else int(clip_seconds * 32)
    )
    if use_video:
        media_tokens += int(clip_seconds * video_tokens_per_second())

    components: Dict[str, Any] = {
        "csv_input_tokens": count_text_tokens(counter, csv_text),
        "media_input_tokens": media_tokens,
        "knowledge_injection_tokens": count_text_tokens(counter, knowledge_text),
        "entry_details_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "entry_details")
        ),
        "advice_ledger_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "previous_advice")
        ),
        "pre_round_notes_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "pre_round_notes")
        ),
        "preceding_context_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "preceding_context")
        ),
        # Present only on the query round's prompt (the correction round does
        # not carry the indices).
        "streamer_index_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "streamer_index")
        ),
        "common_index_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "common_index")
        ),
        "search_injection_tokens": count_text_tokens(counter, search_results),
        "expected_output_tokens": window.budget.estimated_output_tokens,
    }
    if max_output_tokens is not None:
        components["max_output_tokens"] = max_output_tokens
    return components


def search_loop_input_components(
    *,
    counter: TokenCounter | None = None,
    search_results: str = "",
    messages: List[Mapping[str, Any]] | None = None,
) -> Dict[str, int]:
    user_text = _user_message_text(messages)
    search_text = search_results
    if _is_empty_injection(search_text) and user_text:
        search_text = extract_tagged_block(user_text, "search_results")
    return {
        "background_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "background")
        ),
        "contract_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "research_contract")
        ),
        "executed_queries_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "executed_queries")
        ),
        "progress_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "research_progress")
        ),
        "streamer_index_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "streamer_index")
        ),
        "common_index_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "common_index")
        ),
        "knowledge_injection_tokens": count_text_tokens(
            counter, extract_tagged_block(user_text, "knowledge_entries")
        ),
        "search_injection_tokens": count_text_tokens(counter, search_text),
    }


def flatten_input_components(components: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in components.items():
        if key in {"expected_output_tokens", "max_output_tokens"}:
            if value is not None:
                flat[key] = value
            continue
        if isinstance(value, (int, float)) and int(value) > 0:
            flat[key] = int(value)
    return flat


def summarize_validation_locations(errors: Any) -> str:
    if not errors:
        return ""
    if isinstance(errors, str):
        error_items = [errors]
    else:
        error_items = [str(item) for item in errors if str(item)]
    locations: List[str] = []
    for error in error_items:
        found = re.findall(r"\bRow\s+\d+\b|\bSource id\s+[^ .]+|\bsource id\s+[^ .]+", error)
        locations.extend(found)
    seen: set[str] = set()
    unique = []
    for item in locations:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return "; ".join(unique[:12])


def llm_exchange_metadata(
    result,
    *,
    input_components: Mapping[str, Any] | None = None,
    session: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    from .client import extract_token_distribution
    from .prompt_compose import PROMPT_VERSION

    usage = extract_token_distribution(result.raw_response)
    api_attempts = list(getattr(result, "api_attempts", None) or [])
    metadata: Dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "thinking_level": result.thinking_level or "（默认）",
        "input_tokens": (
            f"{usage['uncached_input_tokens']} / {usage['cached_input_tokens']} / "
            f"{usage['total_input_tokens']} (uncached / cached / total)"
        ),
        "output_tokens_breakdown": (
            f"{usage['output_tokens']} / {usage['thinking_tokens']} / "
            f"{usage['total_output_tokens']} (visible / thinking / total)"
        ),
    }
    if session:
        metadata["session"] = session
    if input_components:
        metadata.update(flatten_input_components(input_components))
    metadata.update(extra)
    if metadata.get("validation_errors") and not metadata.get("validation_locations"):
        locations = summarize_validation_locations(metadata.get("validation_errors"))
        if locations:
            metadata["validation_locations"] = locations
    if api_attempts:
        metadata["api_attempts"] = api_attempts
    return metadata
