"""Validators the task runtime can resolve by id, in any process.

The harness validates a candidate with logic that lives in
``stages/correction`` (the window CSV contract), but under the tool protocol
(docs/llm_agent_tool_protocol.md §1) ``submit`` is served by the MCP server
-- a process the agent CLI spawned, not the harness. A closure cannot cross
that boundary; an id plus JSON-serializable parameters can. So a caller
names a validator here and puts its parameters in the task's ``metadata``,
and both the harness (step A, in-process) and the server (step B) resolve
the same function from the same manifest.

Adding a validator: register a builder under a stable id. The builder
receives the task manifest and returns the ``Validator`` the runtime calls.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .agent_task_runtime import ValidationResult, Validator

ValidatorBuilder = Callable[[Mapping[str, Any]], Validator]


def serialize_window(window: Any) -> dict[str, Any]:
    """A ``SubtitleWindow`` as JSON, enough to validate a candidate against it."""

    def segments(items: Any) -> list[dict[str, Any]]:
        return [
            {"id": str(seg.id), "start": float(seg.start), "end": float(seg.end), "text": str(seg.text)}
            for seg in items
        ]

    budget = window.budget
    return {
        "chunk_id": str(window.chunk_id),
        "segments": segments(window.segments),
        "overlap_segments": segments(window.overlap_segments),
        "preceding_segments": segments(window.preceding_segments),
        "boundary_reason": str(window.boundary_reason),
        "clip_start": float(window.clip_start),
        "clip_end": float(window.clip_end),
        "budget": {
            "input_tokens": int(budget.input_tokens),
            "subtitle_input_tokens": int(budget.subtitle_input_tokens),
            "estimated_output_tokens": int(budget.estimated_output_tokens),
            "token_counter_source": str(budget.token_counter_source),
        },
    }


def deserialize_window(payload: Mapping[str, Any]) -> Any:
    from ..chunking import CorrectionBudget, SubtitleSegment, SubtitleWindow

    def segments(items: Any) -> list[SubtitleSegment]:
        return [
            SubtitleSegment(
                id=str(item["id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
            )
            for item in items or []
        ]

    budget = dict(payload.get("budget") or {})
    return SubtitleWindow(
        chunk_id=str(payload["chunk_id"]),
        segments=segments(payload.get("segments")),
        overlap_segments=segments(payload.get("overlap_segments")),
        boundary_reason=str(payload.get("boundary_reason") or ""),
        budget=CorrectionBudget(
            input_tokens=int(budget.get("input_tokens", 0)),
            subtitle_input_tokens=int(budget.get("subtitle_input_tokens", 0)),
            estimated_output_tokens=int(budget.get("estimated_output_tokens", 0)),
            token_counter_source=str(budget.get("token_counter_source") or ""),
        ),
        clip_start=float(payload.get("clip_start", 0.0)),
        clip_end=float(payload.get("clip_end", 0.0)),
        preceding_segments=segments(payload.get("preceding_segments")),
    )


def _accept(_manifest: Mapping[str, Any]) -> Validator:
    return lambda candidate, _m: ValidationResult.accepted(candidate)


def _correction_window(manifest: Mapping[str, Any]) -> Validator:
    """The correction window contract, rebuilt from the manifest's metadata.

    ``metadata.validator`` carries the serialized window; ``metadata.variant``
    and ``metadata.capability_tier`` name the prompt variant the answering
    candidate received, which the CSV rules depend on.
    """

    from ..output_protocol import validate_correction_window_output
    from ..prompt_variants import resolve_variant
    from ..routing.config import CapabilityTier

    metadata = dict(manifest.get("metadata") or {})
    params = dict(metadata.get("validator") or {})
    window = deserialize_window(params["window"])
    tier = CapabilityTier(str(metadata.get("capability_tier") or CapabilityTier.CAPABLE.value))
    variant = resolve_variant(str(metadata.get("variant") or "") or None, tier)

    def validate(candidate: Any, _manifest: Mapping[str, Any]) -> ValidationResult:
        text = str(candidate)
        result = validate_correction_window_output(text, window, variant=variant)
        errors = [str(item) for item in result.errors if str(item).strip()]
        if errors:
            return ValidationResult.repairable(*errors)
        return ValidationResult.accepted(text)

    return validate


VALIDATOR_BUILDERS: dict[str, ValidatorBuilder] = {
    "accept": _accept,
    "correction-window": _correction_window,
}


def build_validator(validator_id: str, manifest: Mapping[str, Any]) -> Validator:
    builder = VALIDATOR_BUILDERS.get(validator_id)
    if builder is None:
        raise KeyError(f"Unknown validator {validator_id!r}")
    return builder(manifest)


def runtime_validators(validator_id: str) -> dict[str, Validator]:
    """A runtime ``validators`` table resolving ``validator_id`` lazily.

    The runtime hands the validator the manifest on every submit, so the
    builder can run then -- the table only has to promise the id exists.
    """

    if validator_id not in VALIDATOR_BUILDERS:
        raise KeyError(f"Unknown validator {validator_id!r}")

    def validate(candidate: Any, manifest: Mapping[str, Any]) -> ValidationResult:
        return build_validator(validator_id, manifest)(candidate, manifest)

    return {validator_id: validate}
