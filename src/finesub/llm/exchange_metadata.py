"""Exchange header metadata: input-component estimates and session usage."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from .chunking import SubtitleWindow, render_window_segments_as_csv
# Canonical top-level (non-nested) sibling extraction lives in output_tags.
from .output_tags import find_top_level_tag_blocks as extract_top_level_tagged_blocks
from .routing.profiles import video_tokens_per_second
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


def is_session_response_record(kind: str, payload: Mapping[str, Any]) -> bool:
    """Whether one artifact record represents an LLM response session.

    ``search_loop_round`` is used for both the search execution ledger and
    the subsequent judge response. Only the latter is an exchange and carries
    response/call metadata; counting the ledger creates a fake zero-token
    session in the task report.
    """

    if kind not in SESSION_RESPONSE_KINDS:
        return False
    if kind != "search_loop_round":
        return True
    return any(
        key in payload
        for key in (
            "response_content",
            "usage",
            "call_error",
            "api_attempts",
            "execution_attempts",
        )
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


# What a run's agent sessions spent, written next to the task artifacts once
# their CLIs have left (`client.write_agent_session_usage`). Session-level by
# decision: the per-call records of such a run carry no tokens at all
# (docs/llm_local_agent.md §12.1.3).
AGENT_SESSION_USAGE_FILENAME = "agent-session-usage.json"


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
    file_ref: Any | None = None,
    video_high_resolution: bool = False,
) -> Dict[str, Any]:
    """Estimated input components for one correction/query call.

    ``file_ref`` is the media actually attached to the call (anything with an
    ``is_video`` flag), not what the media switch asked for. The two part
    company whenever a clip could not be produced, and an estimate that bills
    a text-only request for audio it never carried is worse than no estimate.
    """

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
    media_tokens = 0
    if file_ref is not None:
        media_tokens = (
            counter.count_audio_seconds(clip_seconds)
            if counter is not None
            else int(clip_seconds * 32)
        )
        # A video clip carries its audio track, so the video estimate is on
        # top of the audio one rather than instead of it.
        if getattr(file_ref, "is_video", False):
            media_tokens += int(
                clip_seconds
                * video_tokens_per_second(
                    high_resolution=video_high_resolution
                )
            )

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


def result_uses_high_resolution_video(result: Any) -> bool:
    """Whether the selected target is forced onto the high video tier."""

    target_id = str(getattr(result, "target_id", "") or "")
    if not target_id:
        return False
    try:
        from .routing.model_routes import default_model_routes

        return bool(
            default_model_routes()
            .target_fact(target_id)
            .video_high_resolution_only
        )
    except (KeyError, OSError, ValueError):
        return False


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


def _fold_output_limit_fields(metadata: Dict[str, Any]) -> None:
    """Four near-identical lines -> one readable one.

    The margin is dropped: it is just ``max - threshold``, a constant of the
    check rather than an observation about this call.
    """

    observed = metadata.pop("output_limit_observed_tokens", None)
    threshold = metadata.pop("output_limit_threshold_tokens", None)
    maximum = metadata.pop("output_limit_max_tokens", None)
    basis = metadata.pop("output_limit_basis", None)
    metadata.pop("output_limit_margin_tokens", None)
    if observed is None and threshold is None and maximum is None:
        return
    metadata["output_limit"] = (
        f"observed {observed} / threshold {threshold} / max {maximum}"
        + (f" (basis: {basis})" if basis else "")
    )


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
    execution_attempts = list(getattr(result, "execution_attempts", None) or [])
    route_decision = dict(getattr(result, "route_decision", None) or {})
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
    _fold_output_limit_fields(metadata)
    if metadata.get("validation_errors") and not metadata.get("validation_locations"):
        locations = summarize_validation_locations(metadata.get("validation_errors"))
        if locations:
            metadata["validation_locations"] = locations
    if api_attempts:
        metadata["api_attempts"] = api_attempts
    if execution_attempts:
        metadata["execution_attempts"] = execution_attempts
    if route_decision:
        metadata["route_decision"] = route_decision
    return metadata
