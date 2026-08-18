"""What one correction call is recorded as, and what is read back out of it.

Window/request/response metadata for the task artifacts, plus the token-only
output-limit decision the retry ladder acts on. Nothing here calls a model or
touches run state: every function turns one value into a description of it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from finesub.media.clips import CLIP_AUDIO_SUFFIX
from finesub.subtitles.rendering import format_srt_time

from ...chunking import SubtitleWindow
from ...client import UploadedFileRef, extract_token_distribution
from ...routing.config import DEFAULT_LIMITS
from ...routing.profiles import TranslationProfile
from ...token_budget import CorrectionBudget


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


def _output_budget_row(
    budget: CorrectionBudget,
    tokens: Mapping[str, Any],
    profile: TranslationProfile,
) -> Dict[str, Any]:
    """Predicted vs observed output size for this window.

    The window plan sizes itself on ``k x c x csv_tokens``; this records what
    that prediction was next to what actually came back, so a truncation or a
    split-retry can be read against the budget that allowed it. Accumulating
    ``observed_coefficient`` across runs is also what makes the c calibration
    (docs/llm_followups.md) a matter of reading artifacts rather than running
    experiments.
    """

    csv_tokens = max(0, int(budget.subtitle_input_tokens))
    observed = int(tokens.get("total_output_tokens") or 0)
    return {
        "csv_tokens": csv_tokens,
        "predicted_output_tokens": int(budget.estimated_output_tokens),
        "observed_output_tokens": observed,
        "coefficient": profile.output_coefficient,
        "output_scale": profile.output_scale,
        "observed_coefficient": (
            round(observed / csv_tokens, 3) if csv_tokens else None
        ),
    }


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
