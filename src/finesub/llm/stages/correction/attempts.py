"""One window's validation-retry/split loop -- the part both drivers share.

Serial and parallel dispatch differ in what they hand this loop, not in how a
window is attempted: ``chained`` is the whole difference (only a serial run
reads the advice ledger back and evolves the entry transfer chain).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace

import re
import threading
from typing import Any, Callable, Dict, List, Mapping

from finesub.reporting import current_reporter

from ...agent.agent_validators import judge_style_pick, serialize_window, style_pick_params
from ...routing.capabilities import correction_task_group
from ...chunking import SubtitleWindow, WindowIdMap
from ...client import (
    GeminiPromptBlockedError,
    extract_token_distribution,
    is_prompt_blocked,
    validation_retry_sampling_kwargs,
)
from ...routing.config import LLMRole, NEXT_ADVICE_MAX_TOKENS
from ...content_filter import (
    ContentFilterExhaustedError,
    run_injection_ladder,
    split_rendered_search_block,
)
from ...degeneration import Thinking, pick_after_retry, thinking_of, zero_thinking_record
from ...exchange_log import ExchangeBlock
from ...exchange_metadata import (
    correction_input_components,
    llm_exchange_metadata,
    result_uses_high_resolution_video,
)
from ...knowledge.base import append_task_artifact, load_entry_texts
from ...knowledge.feedback import remap_feedback_source_ids
from ...knowledge.style import record_style_injected, render_style_block
from ...length_tier import warn_window_pick
from ...correction_patch import PATCH_TAG, PatchMerge, RepairBase, apply_patch, patch_instruction, repair_base
from ...output_protocol import (
    PACING_PASS_RATIO,
    PACING_PASS_RATIO_TEST_PROFILE,
    CsvValidationResult,
    CORRECTION_ANSWER_TAGS,
    looks_truncated_translated,
    score_translated_segments,
    strip_leaked_thought,
    validate_correction_window_output,
)
from ...output_tags import extract_single_tag_block, parse_line_items
from ...routing.profiles import TranslationProfile
from ...prompt_variants import correct_only_name, grouped_variant, resolve_variant
from ...prompts import ContextPack
from ...token_truncate import cap_tokens
from .commit import WindowEntries, _window_input_hash, window_prompt_version
from .context import CorrectionRun, RequestTurnProduct, _WindowRunOutcome
from .metadata import (
    _output_budget_row,
    _output_limit_check,
    _provider_reference_metadata,
    _request_reference_metadata,
    _response_finish_reason,
    _response_reference_metadata,
    window_to_metadata,
)
from ...request_prompts import build_window_task_messages


@dataclass(frozen=True)
class _Answer:
    """One reply to a window and what its commit takes from it."""

    result: Any
    validation: Any
    variant: Any
    style_choice: Any
    next_advice: str
    next_transfer: List[str]
    thinking: Thinking


@dataclass(frozen=True)
class _Repair:
    """What the next attempt sends back: the answer to fix and why it failed.

    With a ``base`` the round asks for a patch (`correction_patch`); ``patch_errors`` are what was wrong
    with the last patch itself, which leaves the base as it was."""

    output: str = ""
    errors: tuple[str, ...] = ()
    base: RepairBase | None = None
    patch_errors: tuple[str, ...] = ()

    @property
    def sent(self) -> bool:
        return bool(self.output and self.errors)

    @property
    def all_errors(self) -> List[str]:
        return [*self.patch_errors, *self.errors]

    def instruction(self) -> str:
        return patch_instruction(self.base, self.all_errors) if self.base is not None and self.sent else ""


@dataclass(frozen=True)
class _Reply:
    """A reply as validated: the result whose content is the whole answer, and how a patch went."""

    result: Any
    validation: Any
    merge: PatchMerge | None
    thought_stripped: int

    @property
    def patched(self) -> bool:
        """Whether the answer is a patch laid over the last one -- a few rows, so not thinking over it is
        normal and does not earn the zero-thinking retry (owner 2026-10-24)."""

        return self.merge is not None and self.merge.mode == "patch"


def _read_reply(result: Any, repair: _Repair, window: SubtitleWindow, variant: Any) -> _Reply:
    """Lay a patch reply over its base, or take the reply as a whole answer; then validate.

    A leaked-thought reply is stripped by the block it must end with: ``<patch>`` on a patch round."""

    reply = result.content
    merge = None
    if repair.sent and repair.base is not None:
        patch_text = strip_leaked_thought(reply, (PATCH_TAG,))
        merge = apply_patch(repair.base, reply if patch_text is None else patch_text)
    if merge is not None and merge.mode == "patch":
        content, stripped = merge.text, len(reply) - len(reply if patch_text is None else patch_text)
    else:
        answer = strip_leaked_thought(reply, CORRECTION_ANSWER_TAGS)
        content = reply if answer is None else answer
        stripped = len(reply) - len(content)
    result = dataclass_replace(result, content=content)
    if merge is not None and merge.mode == "invalid":
        validation = CsvValidationResult(ok=False, segments=[], errors=list(merge.errors), warnings=[])
    else:
        validation = validate_correction_window_output(content, window, variant=variant)
    return _Reply(result, validation, merge, stripped)


def _patch_allowed(run: CorrectionRun) -> bool:
    """Whether a repair may ask for a patch: not when a model of the correction cell has its own variant, since
    the next attempt may go to a candidate that writes rows in another format than the base."""

    from ...routing.config import role_config_for

    try:
        config = role_config_for(correction_task_group(run.profile), run.profile.difficulty)
    except Exception:  # noqa: BLE001 - no declaration: no override either
        return True
    return not config.variant_overrides


def _next_repair(run: CorrectionRun, previous: _Repair, merge: PatchMerge | None, answer: str, validation: Any,
                 window: SubtitleWindow, variant: Any) -> _Repair:
    """The repair the next attempt carries after ``answer`` failed ``validation`` (style errors included) on the
    same window."""

    if merge is not None and merge.mode == "invalid":
        return dataclass_replace(previous, patch_errors=tuple(merge.errors))
    if not validation.errors:
        return _Repair()
    base = repair_base(answer, validation, window, variant) if _patch_allowed(run) else None
    # A merge across a turn never sends a window back on its own; one going back anyway fixes it in the same
    # answer (owner 2026-10-16).
    return _Repair(answer, (*validation.errors, *validation.turn_notes), base)


def _repair_record(sent: bool, repair: _Repair, merge: PatchMerge | None) -> Dict[str, Any]:
    """How a repair round was asked and answered, for `correction_window_response`."""

    if not sent:
        return {}
    if repair.base is None:
        return {"repair_mode": "full"}
    assert merge is not None
    return {"repair_mode": "patch", "patch_reply": merge.mode, "patch_units": merge.patch_units,
            "replaced_units": merge.replaced_units}


def _pass_fields(run: CorrectionRun, window: SubtitleWindow, validation: Any) -> Dict[str, Any]:
    """Three passes' fields of a window's record: `pass1: missing` when it got the single pass prompt for want
    of pass 1, its rows that merge more original lines than the soft cap (the validator counted grouped
    lines), and the signs of a degenerate answer (`degeneration.pass2_signals`, recorded only)."""

    if run.grouping is None:
        return {}
    from ...degeneration import pass2_signals
    from ...grouping import over_cap

    if not window.grouped:
        return {"pass1": "missing"}
    rows = over_cap(validation.segments, run.grouping)
    return {**({"over_cap_rows": rows} if rows else {}),
            "degeneration": pass2_signals(window, validation.segments).to_metadata()}


def correction_role_for_profile(profile: TranslationProfile) -> LLMRole:
    """LLM role for the correction window ("纠错 r2" / fast correction step).

    Always ``audio_multimodal``: the role names the job, not the capability.
    A native-search profile still uses this role and asks for the capability
    via ``complete(..., retrieval="native")``, which swaps in the overlay chain
    (a per-call capability filter over the bound group since v2 D4).
    """

    return LLMRole.AUDIO_MULTIMODAL


# (The old per-call efficiency->low thinking override is gone: thinking is a
# preset-level knob resolved into the cell config -- the packaged default's
# correction efficiency knob carries the same "low".)


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


def _log_window_exposures(
    run: CorrectionRun,
    current: SubtitleWindow,
    injected_keys: List[str],
    injected_extent: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    """Book ``exposed`` events for what this window's prompt carried (plan
    §5.1/§5.2): the entries that went in, and of a cut one only its lines
    that did (``injected_extent``, the budget's ``kept``). Same discipline as
    the shadow scan: pure telemetry, idempotent, never worth failing a window
    over."""

    if not (run.knowledge_enabled and injected_keys):
        return
    try:
        from ...knowledge.base import knowledge_root_path, knowledge_signals_enabled, version_rev
        from ...knowledge.node.repo import KnowledgeRepo
        from ...knowledge.node.signals import log_exposed_entries

        if not knowledge_signals_enabled(run.knowledge_root):
            return
        rev = version_rev(run.knowledge_version)
        log_exposed_entries(
            KnowledgeRepo.open(knowledge_root_path(run.knowledge_root)),
            injected_keys,
            task_id=run.task_id,
            window_id=current.chunk_id,
            window_text=" ".join(segment.text for segment in current.segments),
            rev=rev,
            kept=injected_extent,
        )
    except Exception as exc:  # telemetry must never sink the window
        current_reporter().debug(f"knowledge exposure log skipped: {exc}")


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


def window_task_kwargs(
    run: CorrectionRun,
    window: SubtitleWindow,
    *,
    audio_label: str,
    previous_advice: str,
    entry_details: str,
    search_results: str,
    window_notes: str = "",
    variant: str | None = None,
    phonetic_hints: str = "",
    entries_slimmed: bool = False,
) -> Dict[str, Any]:
    """``build_correction_csv_messages`` kwargs for one window of this run."""

    return dict(
        window=window,
        context_pack=run.context_pack,
        audio_file_label=audio_label,
        previous_advice=previous_advice,
        window_notes=window_notes,
        search_results=search_results,
        entry_details=entry_details,
        phonetic_hints=phonetic_hints,
        entries_slimmed=entries_slimmed,
        extra_style=run.extra_style,
        style_block=run.style_block,
        task_update_feedback=run.task_update_feedback,
        profile=run.profile,
        variant=variant,
        knowledge_enabled=run.knowledge_enabled,
    )


def _default_variant(run: CorrectionRun) -> str:
    """The registry default prompt set ("" for a translation task), as the run's task mode has it."""
    return correct_only_name(resolve_variant(None).name) if run.profile.correct_only else ""


def _compose_window_messages(
    run: CorrectionRun,
    window: SubtitleWindow,
    request_product: RequestTurnProduct,
    *,
    audio_label: str,
    previous_advice: str,
    search_text: str,
    entry_details: str,
    variant: str,
    style_catalog: Any = None,
    phonetic_hints: str = "",
    entries_slimmed: bool = False,
) -> List[Dict[str, Any]]:
    """The messages one correction attempt sends, in the run's flow shape.

    Without a request channel the two-message correction prompt; with it
    (``run.request_channel``) the same prompt plus
    the request channel, and -- after a request turn -- the replayed assistant
    turn and the fill turn (``request_prompts.build_window_task_messages``);
    under ``run.agent_fetches`` the tool-session shape. The variant name comes
    from the answering candidate (plan v2 D2/D3); "" is the registry default.
    ``style_catalog`` is the attempt's own snapshot of ``run.style_catalog``:
    an agentic window's pick can pin the style mid-run (`_pin_style_pick`),
    and the messages an attempt is logged with must be the ones it was sent.
    """

    return build_window_task_messages(
        request_channel=run.request_channel,
        agentic=run.agent_fetches,
        request_reply=request_product.request_reply,
        knowledge_indexes=run.knowledge_index_texts if run.knowledge_enabled else None,
        carried_entries=request_product.carried_entries_text,
        requested_entries=request_product.requested_entries,
        requested_search=request_product.requested_search,
        max_search_queries=run.max_search_queries_per_window,
        style_catalog=style_catalog.render() if style_catalog else "",
        extra_style_hint=run.extra_style,
        style_fill_block=request_product.style_fill_block,
        **window_task_kwargs(
            run,
            window,
            audio_label=audio_label,
            previous_advice=previous_advice,
            entry_details=entry_details,
            phonetic_hints=phonetic_hints,
            entries_slimmed=entries_slimmed,
            search_results=search_text,
            # The request turn's <window_notes> reach the delivery inside the
            # replayed assistant message; the opening's notes slot stays empty.
            variant=variant or None,
        ),
    )


def _complete_window(
    run: CorrectionRun,
    window: SubtitleWindow,
    compose_for_variant: Callable[[str], List[Dict[str, Any]]],
    *,
    attempt: int,
    file_ref: Any,
    repair: _Repair,
    replacement_round: bool,
    validator_spec: Mapping[str, Any],
    predelivered: Any = None,
    style_grant: Any = None,
):
    """One correction call -- or the request turn's own delivery.

    ``predelivered`` is the ``LLMCallResult`` the task flow's request turn
    came back with when the model delivered at once (plan §2.2: no empty
    second call). It stands in for attempt 0; every later attempt calls.
    ``style_grant`` is the catalog an agentic window picks from: on a run
    that does not read the base it buys the narrow ``kb_tools: "style"``
    grant (only `kb_read` of `style/<名字>`), pinned to the revision the
    catalog was read at.
    """

    if predelivered is not None:
        return predelivered
    agent_task_extras: Dict[str, Any] | None = None
    if run.knowledge_enabled:
        # §4.3 authorization matrix: correction windows get the read-only kb
        # tools; the pin supplies the revision. The signal identity joins
        # agent-side exposures with this window's matched/landed events.
        agent_task_extras = {
            "kb_tools": "read",
            "kb_signal_task": run.task_id,
            "kb_signal_window": window.chunk_id,
        }
    elif style_grant is not None and style_grant.rev is not None:
        # Only the style to read (auto-style-gaps plan A, 「读权限」): the
        # narrow grant names its own root and revision -- a run that does not
        # read the base has no generation pin to borrow them from.
        agent_task_extras = {
            "kb_tools": "style",
            "knowledge_root": str(run.knowledge_root),
            "knowledge_identity": f"rev:{style_grant.rev}",
            "kb_signal_task": run.task_id,
            "kb_signal_window": window.chunk_id,
        }
    return run.client.complete(
        correction_role_for_profile(run.profile),
        compose_for_variant,
        max_tokens=run.planning_limits.output_limit,
        file_ref=file_ref,
        fallback_audio_ref=(
            (lambda: run.media.ladder_audio_ref(window))
            if file_ref is not None and file_ref.is_video
            else None
        ),
        retrieval=run.profile.retrieval,
        task_group=correction_task_group(run.profile),
        difficulty=run.profile.difficulty,
        previous_output=repair.output,
        validation_errors=repair.all_errors,
        repair_instruction=repair.instruction(),
        # One chain per window: a repair resumes the conversation that
        # wrote the output, and the next window starts a fresh one.
        repair_session_key=f"correction-{window.chunk_id}",
        fresh_session=replacement_round,
        validator_spec=validator_spec,
        correct_only=run.profile.correct_only,
        max_repair_attempts=max(0, run.max_retries_per_window),
        agent_task_extras=agent_task_extras,
        **validation_retry_sampling_kwargs(attempt, repair=repair.sent),
    )


def _accept(run: CorrectionRun, current: SubtitleWindow, answer: _Answer, *, injected_keys: List[str],
            entries: WindowEntries, thinking: Mapping[str, Any] | None = None) -> _WindowRunOutcome:
    """Commit ``answer`` as the window's: its ledger record (``thinking``: a retry for missing thinking, as
    `degeneration.zero_thinking_record` says it went) and the tallies the closing summary reads."""

    validation, content = answer.validation, answer.result.content
    _warn_accepted_turns(current, validation)
    if answer.style_choice is not None:
        _pin_style_pick(run, current, answer.style_choice)
    _log_window_exposures(run, current, injected_keys, entries.extent)
    pass_fields = _pass_fields(run, current, validation)
    if "unmerged" in pass_fields.get("degeneration", {}).get("signals", ()):
        run.note_unmerged(current.chunk_id)
    if thinking is not None:
        run.note_rethought(current.chunk_id)
    run.ledger.commit(
        {
            "chunk_id": current.chunk_id,
            "source_ids": list(current.source_ids),
            "clip_start": round(current.clip_start, 3),
            "input_hash": _window_input_hash(
                current, entries.signature
            ),
            # Entry-free core hash + continuity: what the
            # serial->parallel directional reuse compares
            # (plan A.5 (5)).
            "input_hash_core": _window_input_hash(current),
            "continuity": run.profile.continuity,
            "task_fingerprint": run.task_fingerprint,
            # Knowledge identity per window
            # (docs/llm_local_agent.md SS8): entry text is out of
            # the fingerprint, so this is what makes a mixed-KB
            # task auditable -- and it is the handle the future
            # task runtime needs to bind one task's retries to
            # one knowledge version.
            "knowledge_version": run.knowledge_version,
            "content": content,
            "capability_tier": answer.result.capability_tier.value,
            # The variant this window really used (plan v2
            # §9): difficulty is out of the fingerprint, so a
            # switch-and-resume yields mixed variants -- this
            # is the per-window record that makes the mix
            # auditable.
            "variant": (grouped_variant(answer.variant) if current.grouped else answer.variant).name,
            # The prompt version it was answered under: out of the fingerprint, so a resume after a bump
            # replays it and counts it in the run's warning (`CorrectionRun.note_replay`).
            "prompt_version": window_prompt_version(current),
            "difficulty": run.profile.difficulty,
            "injected_entries": list(injected_keys),
            "keep_entries": answer.next_transfer,
            # Merges across a turn the window was accepted with, by source id.
            "turn_violations": [list(ids) for ids in validation.turn_violation_rows],
            **({"thinking": dict(thinking)} if thinking is not None else {}),
            **pass_fields,
        },
    )
    return _WindowRunOutcome(
        window=current,
        validation=validation,
        next_advice=answer.next_advice,
        next_transfer=answer.next_transfer,
    )


def _call_kb_reads(result: Any) -> List[Mapping[str, Any]]:
    """The answering task's booked ``kb_read`` rows (`fold_kb_reads` put them
    on its execution attempt). Any other backend read nothing through the
    harness, which is what an empty list says."""

    attempts = list(getattr(result, "execution_attempts", None) or [])
    last = attempts[-1] if attempts else {}
    return [row for row in (last.get("kb_reads") or ()) if isinstance(row, Mapping)]


def _pin_style_pick(run: CorrectionRun, window: SubtitleWindow, choice: Any) -> None:
    """The first valid agentic pick fixes the style for the whole task.

    auto-style-gaps plan A1 steps 4-5: recorded like the batched request
    turn's (a `style_selection` artifact with ``chunk_id``, which
    `_writable_style_names` and a resume read back); and any later half of a
    split window no longer sees the catalog -- it gets the picked style's full
    rules up front, so the two halves cannot pick differently. The pick's
    length tier does not move the task's ratio (`length_tier.warn_window_pick`).
    """

    with run._tally_lock:
        if run.style_catalog is None:
            # Another half already pinned it (parallel lanes).
            return
        run.style_catalog = None
        run.style_block = render_style_block(run.knowledge_root, choice.names, rev=choice.rev,
                                             correct_only=run.profile.correct_only)
        run.picked_style_block = run.style_block
    record_style_injected(run.knowledge_root, choice.names, task_id=run.task_id, rev=choice.rev)
    warn_window_pick(run.knowledge_root, choice.names, rev=choice.rev, task_scale=window.length_scale)
    if run.task_artifact_dir:
        append_task_artifact(
            run.task_artifact_dir,
            kind="style_selection",
            task_id=run.task_id,
            payload={"chunk_id": window.chunk_id, **choice.to_dict()},
        )


def _warn_accepted_turns(window: SubtitleWindow, validation: Any) -> None:
    """A merge across a speaker turn is accepted with a warning, never sent back for repair on its own (owner
    2026-10-16, docs/llm_harness_behavior.md 「Prompt 信息」)."""

    if validation.turn_violation_rows:
        # three passes' pass 2 merges only where pass 1 marked 可并
        seam, impact = (("没标可并的接缝", "这几行并了上一步没判为同一个人接着说的行") if window.grouped
                        else ("换人", "这几行把两个人的话并在了一行"))
        current_reporter().warning(
            "correction-turn-merge",
            f"窗口 {window.chunk_id} 有跨{seam}的合并，照收：源段 "
            + "；".join(",".join(ids) for ids in validation.turn_violation_rows),
            impact=impact,
        )


def run_window_attempts(
    run: CorrectionRun,
    current: SubtitleWindow,
    *,
    previous_advice: str,
    entries: WindowEntries,
    chained: bool,
    add_tail: Callable[[SubtitleWindow], None],
    exchange_block: ExchangeBlock | None,
    abort: threading.Event | None = None,
    on_chain_exhausted: Callable[[], None] | None = None,
) -> _WindowRunOutcome:
    """One window's validation-retry/split loop, shared by both modes.

    ``chained`` is the serial-continuity switch: only a serial run reads
    the advice ledger back or evolves the entry transfer chain, so only
    there are <next_advice>/<keep_entries> extracted and recorded. The
    entry set itself always arrives resolved by the caller (per-window
    union in serial, session-fixed in parallel). Mid-loop splits keep the
    first half inside this loop's remaining retry budget and hand the
    tail to ``add_tail``; a split forced on the *final* attempt instead
    returns ``restart_halves`` so both halves rejoin the caller's queue
    as fresh units.

    The retry budget is two-tiered (owner decision 2026-08-19,
    docs/llm_harness_behavior.md 「重试与拼接」): ``max_retries_per_window`` repairs
    *within one session chain* -- each retry carries the previous output
    and the exact validation errors, and an agent backend resumes the
    conversation that wrote them -- and once that is spent the window is
    handed to a *fresh* session (``max_replacements_per_window`` times):
    repair context dropped, so an agent starts a new conversation and a
    stateless endpoint throws blind, and the call re-routes as every call
    does. A degenerate session -- looping, or locked into one wrong
    reading -- would otherwise eat the whole budget while repair's value
    premise ("the model can fix it at a glance") no longer holds. Total
    calls are the product of the two knobs' (n+1) forms.

    An answer that passes but reported no thinking although it was asked to
    (`degeneration.thinking_of`) is held, and the window is asked once more
    in a fresh session -- outside that budget, so a pass on the last attempt
    still gets it; under `resume` the fresh session is where the chain goes
    on. The window keeps one of the two (`degeneration.pick_after_retry`),
    the held one whenever the retry cannot answer (owner 2026-10-17).
    """

    base_chunk_id = current.chunk_id.split("-", 1)[0]
    request_product = run.request_turn_cache.get(base_chunk_id) or RequestTurnProduct()
    injected_keys = list(entries.keys)
    # Repair context for the next attempt, carried only across a retry that
    # sends the *same* window back. A split hands the next attempt a
    # different window, and then last round's output describes rows this
    # one will not be asked about -- worse than saying nothing.
    repair = _Repair()
    # `max(0, …)`: a negative knob would make the product <= 0, and a loop that
    # never runs falls through to the "failed unexpectedly" raise below --
    # an error message about the wrong thing. One call is the floor.
    chain_calls = max(0, run.max_retries_per_window) + 1
    total_calls = chain_calls * (max(0, run.max_replacements_per_window) + 1)
    # A backend that runs tier 1 itself (an agent on the task runtime) spends
    # a whole chain in one call; the attempts it would have been are skipped.
    skip_until = 0
    # A passing answer that did not think, waiting on the one fresh retry it gets.
    held: _Answer | None = None

    def keep_held(error: str) -> _WindowRunOutcome:
        assert held is not None
        return _accept(run, current, held, injected_keys=injected_keys, entries=entries,
                       thinking=zero_thinking_record(held.thinking, None, "original", error=error))

    for attempt in range(total_calls + 1):
        if attempt < skip_until:
            continue
        if attempt == total_calls and held is None:
            break  # the slot past the budget is the held answer's retry only
        if abort is not None and abort.is_set():
            if held is not None:
                return keep_held("aborted")
            # The parallel circuit breaker tripped while this window was in
            # flight. Its batch is already doomed (drain-then-raise), so
            # spending the rest of a 2-tier budget on it is exactly the quota
            # burn the breaker exists to stop -- see PARALLEL_BREAKER_FAILURES.
            raise RuntimeError(
                f"Window {current.chunk_id} abandoned mid-retry: the "
                "parallel circuit breaker tripped."
            )
        sent_repair = repair.sent
        # First call of a tier-2 chain: any conversation the previous chain
        # left cached (notably `resume` mode's run-long one) must be retired,
        # or the "fresh" session would resume the degenerate conversation the
        # replacement exists to escape.
        replacement_round = held is not None or (attempt > 0 and attempt % chain_calls == 0)
        if sent_repair:
            run.count_repair_round()
        if attempt:
            current_reporter().debug(
                "correction window attempt",
                {
                    "chunk": current.chunk_id,
                    "attempt": attempt,
                    "repair": sent_repair,
                    "replacement": replacement_round,
                    "errors": "; ".join(repair.all_errors[:3]),
                },
            )
        window_file_ref = run.media.correction_ref(current)
        window_audio_label = run.media.correction_label(current)
        # Content-filter ladder is independent of validation retries:
        # drop injected retrieval units until the prompt clears, then
        # validate. A contaminated search injection is rewritten into
        # the request-turn cache so -a/-b halves and later attempts see
        # the cleaned text.
        search_block = split_rendered_search_block(request_product.search_results)

        # The same check the loop below applies, named for a backend that can
        # run the repair loop itself (the task runtime resolves the id in
        # whichever process serves `submit`), so the harness and the runtime
        # never disagree about what a valid window is.
        #
        # An agentic `--style auto` window picks in its delivery (A1), so the
        # pick -- and "it really read the picked style" -- is part of that same
        # check. The catalog is snapshotted per attempt: a pin by an earlier
        # half changes the run, not an attempt already composed.
        attempt_catalog = run.style_catalog
        style_pick = (
            style_pick_params(attempt_catalog)
            if run.agent_fetches and attempt_catalog is not None
            else None
        )
        window_validator_spec = {
            "id": "correction-window",
            "params": {
                "window": serialize_window(current),
                **({"style_pick": style_pick} if style_pick is not None else {}),
            },
        }

        def _correction_call(search_text: str):
            def compose_for_variant(variant: str):
                return _compose_window_messages(
                    run, current, request_product, audio_label=window_audio_label,
                    previous_advice=previous_advice, search_text=search_text,
                    entry_details=entries.details, phonetic_hints=entries.hints,
                    entries_slimmed=entries.slimmed, variant=variant,
                    style_catalog=attempt_catalog,
                )

            call_result = _complete_window(
                run, current, compose_for_variant, attempt=attempt, file_ref=window_file_ref,
                repair=repair,
                replacement_round=replacement_round, validator_spec=window_validator_spec,
                style_grant=attempt_catalog if style_pick is not None else None,
                # The request turn's own delivery stands in for attempt 0 of
                # the window it answered -- never of a split half, whose rows
                # it could not have covered.
                predelivered=(
                    request_product.delivered
                    if attempt == 0 and not sent_repair and current.split_depth == 0
                    else None
                ),
            )
            if is_prompt_blocked(call_result.content, call_result.raw_response):
                raise GeminiPromptBlockedError(
                    f"窗口 {current.chunk_id} prompt was blocked by "
                    "the content filter"
                )
            # Re-assemble the messages the answering candidate actually
            # received, for artifacts and exchange logs.
            call_messages = compose_for_variant(call_result.variant)
            return call_result, call_messages

        try:
            ladder_outcome = run_injection_ladder(
                block=search_block,
                call=_correction_call,
                stage=f"窗口 {current.chunk_id}",
                blocked_exception=GeminiPromptBlockedError,
                blacklist=run.content_filter_blacklist,
                task_artifact_dir=run.task_artifact_dir,
                task_id=run.task_id,
                plain_retry=not search_block.units,
            )
        except Exception as exc:  # pragma: no cover - provider behavior
            if isinstance(exc, ContentFilterExhaustedError) and held is None:
                raise
            if run.task_artifact_dir:
                # Failure-path request snapshot for the error artifact
                # (never sent): complete() raised, so no LLMCallResult
                # carries the answering tier — assemble at the default
                # CAPABLE tier (its correct-only set for a correct-only task).
                err_messages = _compose_window_messages(
                    run, current, request_product, audio_label=window_audio_label,
                    previous_advice=previous_advice, search_text=request_product.search_results,
                    entry_details=entries.details, phonetic_hints=entries.hints,
                    entries_slimmed=entries.slimmed, variant=_default_variant(run),
                    style_catalog=attempt_catalog,
                )
                append_task_artifact(
                    run.task_artifact_dir,
                    kind="correction_window_call_error",
                    task_id=run.task_id,
                    payload={
                        "chunk_id": current.chunk_id,
                        "attempt": attempt,
                        "window": window_to_metadata(current),
                        "request": _request_reference_metadata(
                            messages=err_messages,
                            file_ref=window_file_ref,
                            max_tokens=run.planning_limits.output_limit,
                        ),
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
            if held is not None:
                return keep_held(f"{type(exc).__name__}: {exc}")
            raise
        if ladder_outcome.recovered:
            run.count_content_filter_recovery()
        result, messages = ladder_outcome.result
        for _ in range(int(getattr(result, "agent_repair_rounds", 0) or 0)):
            # Repairs the task runtime ran inside this one call; the tally
            # counts rounds, not `complete` calls.
            run.count_repair_round()
        cleaned_search = search_block.render(
            [
                unit
                for unit in search_block.units
                if unit.content_hash not in run.content_filter_blacklist
            ]
        )
        if cleaned_search != request_product.search_results:
            request_product = dataclass_replace(request_product, search_results=cleaned_search)
            run.request_turn_cache[base_chunk_id] = request_product
        request_reference = _request_reference_metadata(
            messages=messages,
            file_ref=window_file_ref,
            max_tokens=run.planning_limits.output_limit,
        )
        # The variant the answering candidate really received (cell
        # default or per-entry override; plan v2 D2/D3). A result without
        # a variant name (stub clients) falls back to its tier's default.
        response_variant = resolve_variant(
            result.variant or None, result.capability_tier
        )
        # Reasoning shipped as the reply: the answer at its end is validated
        # like any reply; the exchange keeps the reply as it came.
        reply = result.content
        read = _read_reply(result, repair, current, response_variant)
        result, validation, thought_stripped = read.result, read.validation, read.thought_stripped
        style_choice = None
        if style_pick is not None:
            # The harness's half of the window validator's style check: the
            # same judge over the same ledger the runtime's submit read
            # (`agent_validators.judge_style_pick`), so the two agree on what a
            # valid window is -- including after the runtime gave up.
            style_choice, style_errors, _read_missing = judge_style_pick(
                result.content, style_pick, _call_kb_reads(result)
            )
            if style_errors:
                validation = dataclass_replace(
                    validation, ok=False, errors=[*validation.errors, *style_errors]
                )
        finish_reason = _response_finish_reason(result.raw_response)
        output_limit_check = _output_limit_check(
            result.raw_response,
            run.planning_limits.output_limit,
        )
        output_limited = bool(output_limit_check["limited"])
        token_distribution = extract_token_distribution(result.raw_response)
        run.token_rows.append(
            {
                "call": "correction_window",
                "chunk_id": current.chunk_id,
                "attempt": attempt,
                "model": result.model,
                "capability_tier": result.capability_tier.value,
                "finish_reason": finish_reason,
                "output_limit_check": output_limit_check,
                "output_budget": _output_budget_row(
                    current.budget, token_distribution, run.profile
                ),
                "tokens": token_distribution,
            }
        )
        pack = run.context_pack or ContextPack()
        context_report: Dict[str, Any] = {}
        context_window = pack.window_context_for(
            current, counter=run.token_counter, report_sink=context_report
        )
        if context_report and run.task_artifact_dir:
            append_task_artifact(
                run.task_artifact_dir,
                kind="window_context_truncated",
                task_id=run.task_id,
                payload={"phase": "correction", **context_report},
            )
        window_input_components = correction_input_components(
            window=current,
            counter=run.token_counter,
            search_results=request_product.search_results,
            context_general=pack.general_text_for(current),
            context_window=context_window,
            messages=messages,
            max_output_tokens=run.planning_limits.output_limit,
            file_ref=window_file_ref,
            video_high_resolution=result_uses_high_resolution_video(result),
        )
        window_session = f"correction-{current.chunk_id}-attempt{attempt}"
        if exchange_block is not None:
            exchange_block.log(
                window_session,
                messages=messages,
                response_text=reply,
                metadata=llm_exchange_metadata(
                    result,
                    session=window_session,
                    input_components=window_input_components,
                    chunk_id=current.chunk_id,
                    attempt=attempt,
                    capability_tier=result.capability_tier.value,
                    finish_reason=finish_reason,
                    validation_ok=validation.ok,
                    # The reasons, not just the verdict: without them an
                    # exchange says a window failed and nothing about why,
                    # and the answer lived only in correction-windows.jsonl.
                    validation_errors=list(validation.errors),
                    validation_warnings=list(validation.warnings),
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
                    **({"thought_stripped": thought_stripped} if thought_stripped else {}),
                    # Prose, and only on a repair round. Two reasons: an
                    # exchange that repeats a window is otherwise
                    # indistinguishable from a blind retry, and the request
                    # rendered below is the *base* prompt -- the repair
                    # turns are added inside the client, per backend, so
                    # they are not in this message list. A bare `True`
                    # would leave a reader looking for them here.
                    # `correction_window_response.repair_round` keeps the
                    # machine-readable flag.
                    **(
                        {
                            "repair_round": (
                                f"上一轮输出 + {len(repair.all_errors)} 条校验错误"
                                "随本次请求发出；原文见上一 attempt 的 exchange"
                                "（下方 message 列表是基础 prompt，不含这两轮）"
                            )
                        }
                        if sent_repair
                        else {}
                    ),
                ),
            )
        update_feedback = (
            remap_feedback_source_ids(
                _extract_task_update_feedback(
                    result.content, count_tokens=run.token_counter.count_text
                ),
                WindowIdMap.from_window(current),
            )
            if run.task_update_feedback
            else ""
        )
        next_advice = (
            _extract_next_advice(
                result.content, count_tokens=run.token_counter.count_text
            )
            if chained
            else ""
        )
        next_transfer: List[str] = []
        if chained:
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
            if request_product.keep_entry_keys:
                keep_raw = list(
                    dict.fromkeys(
                        keep_raw + list(request_product.keep_entry_keys)
                    )
                )
            if not run.window_requests:
                # No request capability anywhere downstream: pruning here
                # would lose the entry for the rest of the run, so the
                # harness passes the current set through untouched. The
                # prompt does not ask for <keep_entries> in this shape.
                next_transfer = list(run.carried.keys)
            elif run.knowledge_enabled and keep_raw and (injected_keys or run.agent_fetches):
                keep_found, _keep_missing = load_entry_texts(
                    run.knowledge_root, keep_raw
                )
                # What this window could have read. Batched: what the harness
                # injected. Agentic: the base itself -- the executor reads it
                # with its own `kb_read`, so filtering against the injected set
                # would stop the transfer chain at whatever the preparation
                # task seeded (the same mistake the preparation task's own keep
                # filter made; plan §10.12).
                #
                # ⚠ `run.knowledge_enabled` is the base gate and has to be
                # explicit here: it used to be implied by `injected_keys` being
                # empty with the base off, and adding `or run.agent_fetches`
                # removed that implication -- an unsolicited `<keep_entries>`
                # then read the store on a run told not to touch it (plan §8).
                # It already folds in "the index is readable" (`run.py`).
                injected_set = set(injected_keys)
                next_transfer = [
                    key
                    for key in keep_found
                    if run.agent_fetches or key in injected_set
                ][:run.transfer_cap]
        if run.task_artifact_dir:
            response_payload = {
                "session": window_session,
                "chunk_id": current.chunk_id,
                "attempt": attempt,
                "model": result.model,
                "fallback_used": result.fallback_used,
                "finish_reason": finish_reason,
                "output_limited": output_limited,
                "output_limit_check": output_limit_check,
                "validation_ok": validation.ok,
                "validation_errors": validation.errors,
                "validation_warnings": validation.warnings,
                "voided_rows": validation.voided_rows,
                "repair_round": sent_repair,
                **_repair_record(sent_repair, repair, read.merge),
                # v15 phase 1: pacing score is observational only —
                # recorded for threshold calibration, never rejects.
                # Test profile relaxes the pass ratio to 1.0 (surface
                # problems for prompt iteration instead of failing).
                "pacing_score": (
                    score_translated_segments(
                        validation.segments,
                        pass_ratio=(
                            PACING_PASS_RATIO_TEST_PROFILE
                            if run.test_profile
                            else PACING_PASS_RATIO
                        ),
                    )
                    if validation.ok
                    else None
                ),
                **({"next_advice": next_advice} if chained else {}),
                "injected_entries": list(injected_keys),
                **({"kb_slim": dict(entries.slim)} if entries.slim is not None else {}),
                **({"keep_entries": next_transfer} if chained else {}),
                "usage": token_distribution,
                "api_attempts": list(result.api_attempts),
                "execution_attempts": list(result.execution_attempts),
                "route_decision": dict(result.route_decision),
                "input_components": window_input_components,
                "window": window_to_metadata(current),
                "request": request_reference,
                "provider": _provider_reference_metadata(result.raw_response),
                "response": _response_reference_metadata(result.content),
                "response_content": result.content,
                **({"thought_stripped": thought_stripped} if thought_stripped else {}),
            }
            if run.task_update_feedback:
                response_payload["task_update_feedback"] = update_feedback
            append_task_artifact(
                run.task_artifact_dir,
                kind="correction_window_response",
                task_id=run.task_id,
                payload=response_payload,
            )
            if update_feedback:
                from ...knowledge.feedback import retrieval_urls_from_response

                append_task_artifact(
                    run.task_artifact_dir,
                    kind="correction_window_task_feedback",
                    task_id=run.task_id,
                    payload={
                        "chunk_id": current.chunk_id,
                        "attempt": attempt,
                        "window": window_to_metadata(current),
                        "feedback": update_feedback,
                        # Harness-side call evidence (docs/knowledge.md 任务反馈采集):
                        # what this call's retrieval ledger actually returned,
                        # attached here because the model's self-reported
                        # sources are exactly what went missing.
                        "retrieval_urls": list(
                            retrieval_urls_from_response(result.raw_response)
                        ),
                    },
                )
        answer = _Answer(result, validation, response_variant, style_choice, next_advice, next_transfer,
                         thinking_of(result))
        passed = validation.ok and not output_limited
        if held is not None:
            refused = 0 if passed else max(1, len(validation.errors))
            chosen = pick_after_retry(0, answer.thinking, refused)
            return _accept(run, current, answer if chosen == "retry" else held, injected_keys=injected_keys,
                           entries=entries, thinking=zero_thinking_record(held.thinking, answer.thinking, chosen))
        if passed and answer.thinking.skipped and not read.patched:
            held = answer
            repair = _Repair()
            current_reporter().debug("correction window answered without thinking",
                                     {"chunk": current.chunk_id, "attempt": attempt})
            continue
        if passed:
            return _accept(run, current, answer, injected_keys=injected_keys, entries=entries)
        current_reporter().debug(
            "correction window validation failed",
            {
                "chunk": current.chunk_id,
                "attempt": attempt,
                "output_limited": output_limited,
                "errors": "; ".join(validation.errors[:3]),
            },
        )
        if getattr(result, "repair_exhausted", False):
            # The backend already ran this chain's repairs and gave up, so the
            # chain is spent: jump to its last slot so the tier-2 logic below
            # (replacement, breaker count, final-attempt handling) sees the
            # boundary it expects, and skip the slots in between.
            chain_end = (attempt // chain_calls + 1) * chain_calls - 1
            skip_until = chain_end + 1
            attempt = chain_end
        final_attempt = attempt >= total_calls - 1
        # Usage counts are the primary signal and stay that way:
        # 46206b1 demoted `finish_reason` because flash reports
        # `length` on complete answers, and window 0001 of a real run
        # threw away a validation-passing result over it. But usage can
        # be absent or under-report (a provider shape change, a paid
        # fallback, plain `observed_tokens == 0`), and then a genuinely
        # truncated reply looked like an ordinary validation failure:
        # the same oversized window went out five more times and the
        # run died on `RuntimeError: Window NNNN failed validation`.
        # That commit promised the content heuristic would move "to
        # validation's responsibility" -- this is that handoff. Only
        # after the retries are spent, so a window that merely needs
        # another attempt still gets one.
        truncated_output = (
            final_attempt
            and not output_limited
            and looks_truncated_translated(result.content)
        )
        if final_attempt and not (output_limited or truncated_output):
            errors = "; ".join(validation.errors) or "output appears truncated"
            raise RuntimeError(
                f"Window {current.chunk_id} failed validation: {errors}"
            )
        failed_window = current
        second_half: SubtitleWindow | None = None
        if output_limited or truncated_output:
            if not run.geometry.may_split(current):
                # The split budget is spent and the answer still does not fit.
                # Splitting again spends another pair of calls on the same
                # failure, so the task stops here with what it knows.
                errors = "; ".join(validation.errors) or "output appears truncated"
                raise RuntimeError(
                    f"Window {current.chunk_id} still fails after "
                    f"{run.geometry.MAX_SPLITS} splits: {errors}"
                )
            halves = run.geometry.split(current)
            if halves is None:
                if final_attempt:
                    errors = (
                        "; ".join(validation.errors)
                        or "output appears truncated"
                    )
                    raise RuntimeError(
                        f"Window {current.chunk_id} failed validation "
                        f"(truncated, unsplittable): {errors}"
                    )
                retry_reason = "output_limited_unsplittable_same_window"
            else:
                current, second_half = halves
                retry_reason = "output_limited_split_in_half"
        else:
            retry_reason = "validation_same_window"
        # Tier-2 boundary: this chain's repair budget is spent, so the next
        # call replaces the session -- no repair context (an agent starts a
        # fresh conversation, a stateless endpoint throws blind). Decided
        # here, before the retry artifact below, so its `repair_context`
        # field describes what the next attempt really carries. `not
        # final_attempt` because on the last attempt of the budget there is
        # no next call, and an artifact promising one misreads as a
        # replacement that never happened.
        chain_exhausted = (attempt + 1) % chain_calls == 0
        replacement_next = chain_exhausted and not final_attempt
        # The next attempt gets the answer it has to fix and the exact reasons
        # it was rejected. Only on the same window: a truncated or split window
        # is a different problem, and its output is not what the next attempt
        # is asked to produce.
        same_window = retry_reason == "validation_same_window" and not replacement_next
        repair = (_next_repair(run, repair, read.merge, result.content, validation, failed_window, response_variant)
                  if same_window else _Repair())
        if (
            chain_exhausted
            and retry_reason == "validation_same_window"
            and on_chain_exhausted is not None
        ):
            # What the parallel circuit breaker counts. Before the budget was
            # a product, "a failed window" *was* one exhausted chain, and the
            # breaker's threshold was priced in those units; counting whole
            # windows now would silently double what it lets a doomed batch
            # spend. Only same-window validation failure counts -- a split or
            # a truncation is a size problem, not the systemic one this caps.
            on_chain_exhausted()
        if run.task_artifact_dir:
            append_task_artifact(
                run.task_artifact_dir,
                kind="correction_window_retry",
                task_id=run.task_id,
                payload={
                    "chunk_id": failed_window.chunk_id,
                    "attempt": attempt,
                    "reason": retry_reason,
                    # Whether the *next* attempt carries this attempt's
                    # output and errors, i.e. whether it is a repair round
                    # or another blind throw.
                    "repair_context": repair.sent,
                    **({"repair_mode": "patch" if repair.base else "full"} if repair.sent else {}),
                    # Whether the next attempt replaces the session (tier 2
                    # of the two-tier budget): fresh conversation, blind.
                    "replacement": replacement_next,
                    "finish_reason": finish_reason,
                    "output_limit_check": output_limit_check,
                    "failed_window": window_to_metadata(failed_window),
                    "retry_chunk_id": current.chunk_id,
                    "retry_window": window_to_metadata(current),
                    "tail_chunk_ids": (
                        [second_half.chunk_id]
                        if second_half is not None
                        else []
                    ),
                    "tail_windows": (
                        [window_to_metadata(second_half)]
                        if second_half is not None
                        else []
                    ),
                },
            )
        if second_half is not None:
            run.ledger.commit(
                {
                    "chunk_id": failed_window.chunk_id,
                    "split_into": [
                        current.chunk_id,
                        second_half.chunk_id,
                    ],
                    "continuity": run.profile.continuity,
                    "task_fingerprint": run.task_fingerprint,
                },
            )
            if final_attempt:
                # Retry budget spent, but the reply itself says the window
                # is too big (provider-confirmed or the content heuristic
                # above): both halves rejoin the caller's queue as fresh
                # units instead of dying on "failed validation".
                return _WindowRunOutcome(
                    window=failed_window,
                    restart_halves=(current, second_half),
                )
            add_tail(second_half)
    raise RuntimeError(  # pragma: no cover - the loop returns or raises
        f"Window {current.chunk_id} failed unexpectedly."
    )
