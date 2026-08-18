"""One window's validation-retry/split loop -- the part both drivers share.

Serial and parallel dispatch differ in what they hand this loop, not in how a
window is attempted: ``chained`` is the whole difference (only a serial run
reads the advice ledger back and evolves the entry transfer chain).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

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
    evidence_pack_block,
    run_injection_ladder,
    split_rendered_search_block,
)
from ...exchange_log import ExchangeBlock
from ...exchange_metadata import (
    correction_input_components,
    llm_exchange_metadata,
    result_uses_high_resolution_video,
)
from ...knowledge.base import append_task_artifact, load_entry_texts
from ...knowledge.feedback import remap_feedback_source_ids
from ...output_protocol import (
    PACING_PASS_RATIO,
    PACING_PASS_RATIO_TEST_PROFILE,
    looks_truncated_translated,
    score_translated_segments,
    validate_correction_window_output,
)
from ...output_tags import extract_single_tag_block, parse_line_items
from ...routing.profiles import TranslationProfile
from ...prompt_variants import resolve_variant
from ...prompts import ContextPack, build_correction_csv_messages
from ...token_truncate import cap_tokens
from .commit import _window_input_hash
from .context import CorrectionRun, _WindowRunOutcome
from .metadata import (
    _output_budget_row,
    _output_limit_check,
    _provider_reference_metadata,
    _request_reference_metadata,
    _response_finish_reason,
    _response_reference_metadata,
    window_to_metadata,
)
from .query_round import QueryRoundProduct


def correction_role_for_profile(profile: TranslationProfile) -> LLMRole:
    """LLM role for the correction window ("纠错 r2" / fast correction step).

    Always ``audio_multimodal``: the role names the job, not the capability.
    A native-search profile still uses this role and asks for the capability
    via ``complete(..., native_search=True)``, which swaps in the overlay chain
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


def run_window_attempts(
    run: CorrectionRun,
    current: SubtitleWindow,
    *,
    previous_advice: str,
    window_entry_details: str,
    window_entry_sig: str,
    injected_keys: List[str],
    chained: bool,
    add_tail: Callable[[SubtitleWindow], None],
    exchange_block: ExchangeBlock | None,
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
    """

    base_chunk_id = current.chunk_id.split("-", 1)[0]
    query_product = run.query_round_cache.get(base_chunk_id) or QueryRoundProduct()
    # Repair context for the next attempt, carried only across a retry that
    # sends the *same* window back. A split hands the next attempt a
    # different window, and then last round's output describes rows this
    # one will not be asked about -- worse than saying nothing.
    repair_output = ""
    repair_errors: List[str] = []
    for attempt in range(run.max_retries_per_window + 1):
        sent_repair = bool(repair_output and repair_errors)
        window_file_ref = run.media.correction_ref(current)
        window_audio_label = run.media.correction_label(current)
        # Content-filter ladder is independent of validation retries:
        # drop injected retrieval units until the prompt clears, then
        # validate. A contaminated search injection is rewritten into
        # the query-round cache so -a/-b halves and later attempts see
        # the cleaned text.
        if run.evidence_pack_mode and query_product.search_results.strip():
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

            def compose_for_variant(variant: str):
                # The variant name arrives from the answering candidate's
                # cell/entry (plan v2 D2/D3); "" falls back to the
                # registry default.
                return build_correction_csv_messages(
                    window=current,
                    context_pack=run.context_pack,
                    audio_file_label=window_audio_label,
                    previous_advice=previous_advice,
                    query_round_notes=query_product.window_notes,
                    search_results=search_text,
                    entry_details=window_entry_details,
                    extra_style=run.extra_style,
                    common_mistakes_block=run.common_mistakes_block,
                    task_update_feedback=run.task_update_feedback,
                    evidence_pack_mode=use_pack,
                    profile=run.profile,
                    variant=variant or None,
                    knowledge_enabled=run.knowledge_enabled,
                )

            call_result = run.client.complete(
                correction_role_for_profile(run.profile),
                compose_for_variant,
                max_tokens=run.planning_limits.output_limit,
                file_ref=window_file_ref,
                fallback_audio_ref=(
                    (lambda: run.media.ladder_audio_ref(current))
                    if window_file_ref is not None and window_file_ref.is_video
                    else None
                ),
                native_search=run.profile.native_search,
                task_group=correction_task_group(run.profile),
                difficulty=run.profile.difficulty,
                previous_output=repair_output,
                validation_errors=repair_errors,
                # One chain per window: a repair resumes the conversation that
                # wrote the output, and the next window starts a fresh one.
                repair_session_key=f"correction-{current.chunk_id}",
                **validation_retry_sampling_kwargs(attempt),
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
            if isinstance(exc, ContentFilterExhaustedError):
                raise
            if run.task_artifact_dir:
                # Failure-path request snapshot for the error artifact
                # (never sent): complete() raised, so no LLMCallResult
                # carries the answering tier — assemble at the default
                # CAPABLE tier.
                err_messages = build_correction_csv_messages(
                    window=current,
                    context_pack=run.context_pack,
                    audio_file_label=window_audio_label,
                    previous_advice=previous_advice,
                    query_round_notes=query_product.window_notes,
                    search_results=query_product.search_results,
                    entry_details=window_entry_details,
                    extra_style=run.extra_style,
                    common_mistakes_block=run.common_mistakes_block,
                    task_update_feedback=run.task_update_feedback,
                    evidence_pack_mode=run.evidence_pack_mode,
                    profile=run.profile,
                    knowledge_enabled=run.knowledge_enabled,
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
            raise
        result, messages = ladder_outcome.result
        cleaned_search = search_block.render(
            [
                unit
                for unit in search_block.units
                if unit.content_hash not in run.content_filter_blacklist
            ]
        )
        if cleaned_search != query_product.search_results:
            query_product = QueryRoundProduct(
                search_results=cleaned_search,
                window_notes=query_product.window_notes,
                requested_entry_keys=query_product.requested_entry_keys,
            )
            run.query_round_cache[base_chunk_id] = query_product
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
        validation = validate_correction_window_output(
            result.content,
            current,
            variant=response_variant,
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
            search_results=query_product.search_results,
            context_general=pack.general_prompt_text(),
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
                                f"上一轮输出 + {len(repair_errors)} 条校验错误"
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
            if query_product.keep_entry_keys:
                keep_raw = list(
                    dict.fromkeys(
                        keep_raw + list(query_product.keep_entry_keys)
                    )
                )
            if not run.external_injection:
                # No request capability anywhere downstream: pruning here
                # would lose the entry for the rest of the run, so the
                # harness passes the current set through untouched. The
                # prompt does not ask for <keep_entries> in this shape.
                next_transfer = list(run.carried.keys)
            elif keep_raw and injected_keys:
                keep_found, _keep_missing = load_entry_texts(
                    run.knowledge_root, keep_raw
                )
                injected_set = set(injected_keys)
                next_transfer = [
                    key for key in keep_found if key in injected_set
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
                append_task_artifact(
                    run.task_artifact_dir,
                    kind="correction_window_task_feedback",
                    task_id=run.task_id,
                    payload={
                        "chunk_id": current.chunk_id,
                        "attempt": attempt,
                        "window": window_to_metadata(current),
                        "feedback": update_feedback,
                    },
                )
        if validation.ok and not output_limited:
            run.ledger.commit(
                {
                    "chunk_id": current.chunk_id,
                    "source_ids": list(current.source_ids),
                    "clip_start": round(current.clip_start, 3),
                    "input_hash": _window_input_hash(
                        current, window_entry_sig
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
                    "content": result.content,
                    "capability_tier": result.capability_tier.value,
                    # The variant this window really used (plan v2
                    # §9): difficulty is out of the fingerprint, so a
                    # switch-and-resume yields mixed variants -- this
                    # is the per-window record that makes the mix
                    # auditable.
                    "variant": response_variant.name,
                    "difficulty": run.profile.difficulty,
                    "injected_entries": list(injected_keys),
                    "keep_entries": next_transfer,
                },
            )
            return _WindowRunOutcome(
                window=current,
                validation=validation,
                next_advice=next_advice,
                next_transfer=next_transfer,
            )
        final_attempt = attempt >= run.max_retries_per_window
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
        if retry_reason == "validation_same_window" and validation.errors:
            # The next attempt gets the answer it has to fix and the exact
            # reasons it was rejected. Only here: a truncated or split
            # window is a different problem, and its output is not the
            # thing the next attempt is being asked to produce.
            repair_output = result.content
            repair_errors = list(validation.errors)
        else:
            repair_output = ""
            repair_errors = []
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
                    "repair_context": bool(repair_output and repair_errors),
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
