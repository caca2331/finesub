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

What a validator sees is the manifest **plus** ``observed``: what the runtime
saw this task do, which no manifest can know in advance -- today
``observed.kb_reads``, the answering context's successful ``kb_read`` replies
(``AgentTaskRuntime.validator_view``). Both judges read the same ledger: the
MCP server's submit through the runtime, the harness through the copy folded
into the call's execution attempt (``agent_session_host.fold_kb_reads``). A
verdict that depends on it is flagged ``observed_dependent`` so the runtime
does not cache it against the candidate: the same answer may pass once the
read has happened.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .agent_task_runtime import PatchOutcome, ValidationResult, Validator

ValidatorBuilder = Callable[[Mapping[str, Any]], Validator]
#: ``(candidate, draft, manifest)``: what a patch candidate makes of the draft it lays over
#: (`AgentTaskRuntime._lay_patch_locked`).
Merge = Callable[[str, str, Mapping[str, Any]], PatchOutcome]


def serialize_window(window: Any) -> dict[str, Any]:
    """A ``SubtitleWindow`` as JSON, enough to validate a candidate against it."""

    def segments(items: Any) -> list[dict[str, Any]]:
        return [
            {"id": str(seg.id), "start": float(seg.start), "end": float(seg.end), "text": str(seg.text),
             "seam": None if seg.seam is None else seg.seam.value,
             **({"group_key": seg.group_key} if seg.group_key is not None else {})}
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
        **({"grouped": True} if window.grouped else {}),
        "length_scale": window.length_scale,
        "budget": {
            "input_tokens": int(budget.input_tokens),
            "subtitle_input_tokens": int(budget.subtitle_input_tokens),
            "estimated_output_tokens": int(budget.estimated_output_tokens),
            "token_counter_source": str(budget.token_counter_source),
        },
    }


def deserialize_window(payload: Mapping[str, Any]) -> Any:
    from ..chunking import CorrectionBudget, Seam, SubtitleSegment, SubtitleWindow

    def segments(items: Any) -> list[SubtitleSegment]:
        return [
            SubtitleSegment(
                id=str(item["id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                seam=None if item.get("seam") is None else Seam(item["seam"]),
                group_key=item.get("group_key"),
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
        grouped=bool(payload.get("grouped", False)),
        length_scale=None if payload.get("length_scale") is None else float(payload["length_scale"]),
    )


def _accept(_manifest: Mapping[str, Any]) -> Validator:
    return lambda candidate, _m: ValidationResult.accepted(candidate)


def observed_kb_reads(view: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The task's successful ``kb_read`` rows from a validator view."""

    observed = view.get("observed") if isinstance(view, Mapping) else None
    rows = (observed or {}).get("kb_reads") if isinstance(observed, Mapping) else None
    return [row for row in rows or () if isinstance(row, Mapping)]


def style_pick_params(catalog: Any) -> dict[str, Any]:
    """What the window validator needs to judge a `--style auto` pick: the
    candidate names and the revision they were read at, as JSON."""

    return {"candidates": list(catalog.keys), "rev": catalog.rev}


def judge_style_pick(
    text: str, params: Mapping[str, Any], kb_reads: Any
) -> tuple[Any, list[str], bool]:
    """``(choice, errors, read_missing)`` for an agentic window's delivery.

    auto-style-gaps plan A1 step 3: the delivery names one candidate in its
    `<style_selection>` block (the forced-pick contract, `StyleCatalog.choose`)
    and -- unless that pick is the legal "none" -- the task must have read
    ``style/<pick>`` in full and successfully: a reply cut by the size cap is
    never booked, and a ``sections`` read is not the whole style. Only the
    harness-side ledger counts (``kb_reads``), never the driver's events,
    which keep the tool name and drop the arguments. ``read_missing`` marks
    the one verdict that depends on that ledger rather than on the text.
    """

    from ..knowledge.style import STYLE_CATEGORY, StyleCatalog, StyleContractError
    from ..output_tags import find_top_level_tag_blocks
    from ..prepare_prompts import STYLE_SELECTION_TAG

    catalog = StyleCatalog.from_keys(
        [str(key) for key in params.get("candidates") or ()],
        rev=params.get("rev") if isinstance(params.get("rev"), int) else None,
    )
    try:
        choice = catalog.choose(find_top_level_tag_blocks(text or "", STYLE_SELECTION_TAG))
    except StyleContractError as exc:
        return None, [str(exc)], False
    if not choice.key:
        return choice, [], False
    rows = [
        row
        for row in kb_reads or ()
        if isinstance(row, Mapping)
        and str(row.get("category") or "") == STYLE_CATEGORY
        and str(row.get("entry") or "") == choice.key
    ]
    if any(not row.get("sections") for row in rows):
        return choice, [], False
    partial = "（只读了部分章节）" if rows else ""
    return (
        None,
        [
            f"<style_selection> 选了 {choice.key!r}，但本任务没有用 kb_read 完整读过 "
            f"style/{choice.key}{partial}：先不带 sections 读全文、按它翻译，再交付"
        ],
        True,
    )


def _correction_window(manifest: Mapping[str, Any]) -> Validator:
    """The correction window contract, rebuilt from the manifest's metadata.

    ``metadata.validator`` carries the serialized window; ``metadata.variant``
    and ``metadata.capability_tier`` name the prompt variant the answering
    candidate received, which the CSV rules depend on. ``style_pick`` in the
    same params is a fast `--style auto` window that picks in its delivery
    (`judge_style_pick`); the harness runs the same judge on the same ledger
    (`stages/correction/attempts.py`), so the two never disagree. A reply that
    is leaked reasoning is judged by the answer it ends with, as the harness
    judges it (`strip_leaked_thought`); it is accepted as submitted, so the
    harness strips it again where the exchange keeps the reply and the record
    counts what was stripped.
    """

    from ..correction_patch import patch_instruction, repair_base, rows_to_fix
    from ..output_protocol import CORRECTION_ANSWER_TAGS, strip_leaked_thought, validate_correction_window_output

    window, variant = _window_and_variant(manifest)
    style_pick = dict(dict(manifest.get("metadata") or {}).get("validator") or {}).get("style_pick")

    def validate(candidate: Any, view: Mapping[str, Any]) -> ValidationResult:
        submitted = str(candidate)
        text = strip_leaked_thought(submitted, CORRECTION_ANSWER_TAGS) or submitted
        result = validate_correction_window_output(text, window, variant=variant)
        errors = [str(item) for item in result.errors if str(item).strip()]
        read_missing = False
        if isinstance(style_pick, Mapping):
            _choice, style_errors, read_missing = judge_style_pick(
                text, style_pick, observed_kb_reads(view)
            )
            errors.extend(style_errors)
        if errors:
            # Merges across a turn never send an answer back on their own; one going back anyway fixes them too.
            errors.extend(result.turn_notes)
            # A mostly right answer is repaired with a patch, as the harness's own repair round asks for one; a
            # style error belongs to no row, so that answer goes back whole.
            base = None if len(errors) != len(result.errors) + len(result.turn_notes) else repair_base(
                text, result, window, variant)
            metadata = {
                **({"observed_dependent": True} if read_missing else {}),
                **({"repair_instruction": patch_instruction(base, errors), "rows_to_fix": rows_to_fix(base)}
                   if base is not None else {}),
            }
            return ValidationResult.repairable(*errors, metadata=metadata)
        return ValidationResult.accepted(submitted)

    return validate


def _window_and_variant(manifest: Mapping[str, Any]) -> tuple[Any, Any]:
    """The window and the prompt variant a correction window's manifest names."""

    from ..prompt_variants import resolve_variant
    from ..routing.config import CapabilityTier

    metadata = dict(manifest.get("metadata") or {})
    window = deserialize_window(dict(metadata.get("validator") or {})["window"])
    tier = CapabilityTier(str(metadata.get("capability_tier") or CapabilityTier.CAPABLE.value))
    return window, resolve_variant(str(metadata.get("variant") or "") or None, tier)


def _lay_patch(candidate: str, draft: str, *, answer_tags: tuple[str, ...], base_of: Callable[[str], Any],
               apply: Callable[[Any, str], Any]) -> PatchOutcome:
    """A `<patch>` candidate laid over the session's last rejected whole answer (``draft``): ``base_of`` makes the
    base of a draft read by ``answer_tags`` (None when it takes no patch), ``apply`` lays the patch on it
    (`correction_patch.apply_patch` and its kin). The draft is judged again here: it may be one the runtime kept
    without judging it (a submit that still owed blocks)."""

    from ..correction_patch import PATCH_TAG
    from ..exchange_metadata import extract_top_level_tagged_blocks
    from ..output_protocol import strip_leaked_thought

    patch = strip_leaked_thought(candidate, (PATCH_TAG,)) or candidate
    if not extract_top_level_tagged_blocks(patch, PATCH_TAG):
        return PatchOutcome()
    text = strip_leaked_thought(draft, answer_tags) or draft
    base = base_of(text) if text else None
    if base is None:
        return PatchOutcome(errors=(
            "A <patch> goes over a mostly right earlier answer in this session, and there is none; "
            "submit the whole answer.",
        ), spends_repair=False)
    merge = apply(base, patch)
    if merge.mode != "patch":
        return PatchOutcome(errors=merge.errors or ("Send only the <patch> block.",))
    return PatchOutcome(text=merge.text)


def _merge_correction_patch(candidate: str, draft: str, manifest: Mapping[str, Any]) -> PatchOutcome:
    """A `<patch>` laid over the session's last rejected window answer (`correction_patch`)."""

    from ..correction_patch import apply_patch, repair_base
    from ..output_protocol import CORRECTION_ANSWER_TAGS, validate_correction_window_output

    window, variant = _window_and_variant(manifest)
    return _lay_patch(
        candidate, draft, answer_tags=CORRECTION_ANSWER_TAGS,
        base_of=lambda text: repair_base(text, validate_correction_window_output(text, window, variant=variant),
                                         window, variant),
        apply=apply_patch,
    )


def _proofread_window(manifest: Mapping[str, Any]) -> Validator:
    """Three passes' pass 3: the `<proofread>` block of a window of ``metadata.validator.count`` lines, read by
    the harness's own parser (`stages/proofread/answers.py`), so the two never disagree."""

    from ..stages.proofread.answers import AnswerError, answer_of, parse_changes

    count = int(dict(dict(manifest.get("metadata") or {}).get("validator") or {})["count"])

    def validate(candidate: Any, view: Mapping[str, Any]) -> ValidationResult:
        try:
            parse_changes(answer_of(str(candidate)), count)
        except AnswerError as error:
            return ValidationResult.repairable(str(error))
        return ValidationResult.accepted(str(candidate))

    return validate


def _judge_params(manifest: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    """The judged row ids and the tracks the answer may name."""

    params = dict(dict(manifest.get("metadata") or {}).get("validator") or {})
    return [str(key) for key in params["ids"]], [int(track) for track in params["tracks"]]


def _speaker_judge(manifest: Mapping[str, Any]) -> Validator:
    """The speaker judge's (and pass 1's) `<judge>` block for ``metadata.validator.ids``, read by the harness's
    own reader (`stages/speaker_judge/answers.survey`), so the two never disagree. A reply whose block has
    something to keep is repaired with a patch of the rest (`stages/speaker_judge/patch.repair_base`), as the
    harness's own repair round asks; any other refusal carries only its errors, which ask for the whole answer
    again. Leaked reasoning is judged by the answer it ends with and accepted as submitted."""

    from ..output_protocol import strip_leaked_thought
    from ..stages.speaker_judge.answers import survey
    from ..stages.speaker_judge.patch import patch_instruction, repair_base, rows_to_fix

    ids, tracks = _judge_params(manifest)

    def validate(candidate: Any, view: Mapping[str, Any]) -> ValidationResult:
        submitted = str(candidate)
        text = strip_leaked_thought(submitted, ("judge",)) or submitted
        errors = survey(text, ids, tracks).errors
        if not errors:
            return ValidationResult.accepted(submitted)
        base = repair_base(text, ids, tracks)
        metadata = {"repair_instruction": patch_instruction(base, errors), "rows_to_fix": rows_to_fix(base)} \
            if base is not None else {}
        return ValidationResult.repairable(*errors, metadata=metadata)

    return validate


def _merge_judge_patch(candidate: str, draft: str, manifest: Mapping[str, Any]) -> PatchOutcome:
    """A `<patch>` laid over the session's last rejected `<judge>` answer (`stages/speaker_judge/patch.py`)."""

    from ..stages.speaker_judge.patch import apply_patch, repair_base

    ids, tracks = _judge_params(manifest)
    return _lay_patch(candidate, draft, answer_tags=("judge",), base_of=lambda text: repair_base(text, ids, tracks),
                      apply=apply_patch)


VALIDATOR_BUILDERS: dict[str, ValidatorBuilder] = {
    "accept": _accept,
    "correction-window": _correction_window,
    "proofread-window": _proofread_window,
    "speaker-judge": _speaker_judge,
}

#: Validators whose repairs may come as a patch, and what lays the patch over the draft.
VALIDATOR_MERGES: dict[str, Merge] = {
    "correction-window": _merge_correction_patch,
    "speaker-judge": _merge_judge_patch,
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

    merge = VALIDATOR_MERGES.get(validator_id)
    if merge is not None:
        # Read by the runtime as an optional attribute: a validator table entry stays a plain callable.
        validate.merge = merge  # type: ignore[attr-defined]
    return {validator_id: validate}
