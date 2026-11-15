"""The correction task's request turn (plan §3 / §5.1).

A batched window with something to ask for (harness search under
``retrieval=local``, entries wherever the base is readable --
`task_graph.window_flow`) is ONE task on the correction cell: its first
call carries the full correction contract plus the request channel
(:mod:`finesub.llm.request_prompts`). The reply is either the deliverable
already -- then this module hands the ``LLMCallResult`` to the attempt
loop, which validates it as attempt 0 without calling again -- or one batch
of requests, which the harness fills (entries off the base, one round of
local search where granted) and the attempt loop continues in the same
conversation.

Best-effort: a call error degrades to an empty product and the window is
corrected without that input -- except where the turn carries the `--style
auto` pick (``run.style_catalog``, auto-style-gaps plan A). There the first
reply must pick (a direct delivery is malformed), a reply that still has not
picked validly when the format retries run out fails the task, and so does a
call error: running on without a style would be exactly the silent fallback
the forced pick rules out.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ...chunking import SubtitleWindow
from ...client import (
    GeminiPromptBlockedError,
    LLMCallResult,
    extract_token_distribution,
    is_prompt_blocked,
    validation_retry_sampling_kwargs,
)
from ...content_filter import run_injection_ladder, split_rendered_search_block
from ...exchange_log import ExchangeBlock, ExchangeLogger
from ...exchange_metadata import (
    correction_input_components,
    llm_exchange_metadata,
    result_uses_high_resolution_video,
)
from ...knowledge.base import append_task_artifact, load_entry_texts
from ...knowledge.style import (
    StyleChoice,
    StyleContractError,
    record_style_injected,
    render_style_block,
)
from ...length_tier import warn_window_pick
from ...prepare_prompts import STYLE_SELECTION_TAG
from ...output_tags import (
    extract_single_tag_block,
    find_top_level_tag_blocks,
    parse_guided_line_items,
    parse_line_items,
)
from ...token_truncate import cap_tokens
from ...prompt_compose import PROMPT_VERSION
from ...prompts import ContextPack
from ...request_prompts import build_window_task_messages
from ...routing.capabilities import correction_task_group
from ...routing.config import (
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    WINDOW_NOTES_MAX_TOKENS,
    injection_block_token_limit,
)
from ...session_checkpoint import session_input_hash
from ...web_search import SearchRequest, render_search_results, search_results_metadata
from .attempts import correction_role_for_profile, window_task_kwargs
from .context import CorrectionRun, RequestTurnProduct
from .metadata import _output_limit_check, _response_finish_reason
#: Checkpoint session of the request turn.
REQUEST_SESSION = "request"
REQUEST_TURN_FORMAT_RETRIES = 2
#: The block that marks a reply as a request batch, per grant: the search
#: block where harness search runs (as it always has), else either entry block.
SEARCH_REQUEST_TAGS = ("search_queries",)
ENTRY_REQUEST_TAGS = ("requested_entries", "keep_entries")


def request_tags(search: bool) -> tuple[str, ...]:
    return SEARCH_REQUEST_TAGS if search else ENTRY_REQUEST_TAGS


def shape_parse_error(search: bool) -> str:
    """What a reply that is neither shape failed on, recorded as that call's
    ``parse_error`` so the artifact says why it was redone."""

    tags = " / ".join(f"<{tag}>" for tag in request_tags(search))
    return f"neither <translated> nor {tags} at top level"


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


def delivered_directly(text: str) -> bool:
    """A reply that carries a top-level ``<translated>`` is the deliverable."""

    return bool(find_top_level_tag_blocks(text or "", "translated"))


def request_turn_messages(
    run: CorrectionRun,
    window: SubtitleWindow,
    *,
    audio_label: str,
    previous_advice: str,
    carried_entries_text: str,
    variant: str = "",
    phonetic_hints: str = "",
    entries_slimmed: bool = False,
) -> List[Dict[str, Any]]:
    return build_window_task_messages(
        request_channel=True,
        knowledge_indexes=run.knowledge_index_texts if run.knowledge_enabled else None,
        carried_entries=carried_entries_text,
        max_search_queries=run.max_search_queries_per_window,
        style_catalog=run.style_catalog.render() if run.style_catalog else "",
        extra_style_hint=run.extra_style,
        **window_task_kwargs(
            run,
            window,
            audio_label=audio_label,
            previous_advice=previous_advice,
            entry_details="",
            search_results="",
            variant=variant or None,
            phonetic_hints=phonetic_hints,
            entries_slimmed=entries_slimmed,
        ),
    )


def reply_shape_errors(run: CorrectionRun, text: str) -> List[str]:
    """Why a request-turn reply is malformed; empty when it is well-formed.

    Without a pick to make: a delivery, or a request carrying a block of its
    grant. With one (``run.style_catalog``): never a delivery, and always one
    valid `<style_selection>` -- AND, not OR: a reply that requests but does
    not pick is as malformed as one that picks a name outside the catalog.
    Where the turn exists only for the pick (``not run.window_requests``,
    A2), a request block is malformed too: nothing would serve it.
    """

    search = run.external_injection
    catalog = run.style_catalog
    if catalog is None:
        ok = delivered_directly(text) or any(
            find_top_level_tag_blocks(text, tag) for tag in request_tags(search)
        )
        return [] if ok else [shape_parse_error(search)]
    if delivered_directly(text):
        return ["本任务必须先在请求回复里选定翻译风格（<style_selection>），不能直接交付终稿"]
    errors: List[str] = []
    try:
        catalog.choose(find_top_level_tag_blocks(text, STYLE_SELECTION_TAG))
    except StyleContractError as exc:
        errors.append(str(exc))
    if not run.window_requests:
        stray = [
            tag
            for tag in (*SEARCH_REQUEST_TAGS, *ENTRY_REQUEST_TAGS)
            if find_top_level_tag_blocks(text, tag)
        ]
        if stray:
            errors.append(
                "本窗口只选风格，没有可请求的输入："
                + " / ".join(f"<{tag}>" for tag in stray)
                + " 不能出现"
            )
    return errors


def _parse_requests(
    text: str, *, remaining_new: int, search: bool, queries_required: bool = True
) -> Dict[str, Any]:
    pairs: List[tuple[str, str]] = []
    parse_error = ""
    # Without harness search a stray <search_queries> is not a request: there
    # is nothing to run it, so it is not read either. With a style pick the
    # reply may be the pick alone, so the block is optional there.
    if search:
        try:
            pairs = parse_guided_line_items(
                extract_single_tag_block(text, "search_queries", required=queries_required)
            )
        except ValueError as exc:
            parse_error = str(exc)
    try:
        requested = parse_line_items(
            extract_single_tag_block(text, "requested_entries", required=False)
        )[:remaining_new]
    except ValueError:
        requested = []
    try:
        keep = parse_line_items(extract_single_tag_block(text, "keep_entries", required=False))
    except ValueError:
        keep = []
    return {
        "pairs": pairs,
        "requested": requested,
        "keep": keep,
        "parse_error": parse_error,
    }


def run_window_request_turn(
    run: CorrectionRun,
    window: SubtitleWindow,
    *,
    previous_advice: str,
    carried_entries_text: str,
    carried_key_count: int,
    exchange_logger: ExchangeLogger | ExchangeBlock | None,
    phonetic_hints: str = "",
    entries_slimmed: bool = False,
) -> RequestTurnProduct:
    """First call of the window task; returns what the delivery turn needs.

    ``phonetic_hints`` / ``entries_slimmed``: the serial window's slimmed
    carried entries (`kb_phonetic_hints`); the delivery shows the same."""

    file_ref = run.media.correction_ref(window)
    audio_label = run.media.correction_label(window)
    role = correction_role_for_profile(run.profile)
    remaining_new = max(
        0, min(KB_WINDOW_NEW_REQUEST_MAX_ENTRIES, KB_WINDOW_TOTAL_ENTRIES - max(0, carried_key_count))
    )

    def compose(variant: str) -> List[Dict[str, Any]]:
        return request_turn_messages(
            run,
            window,
            audio_label=audio_label,
            previous_advice=previous_advice,
            carried_entries_text=carried_entries_text,
            variant=variant,
            phonetic_hints=phonetic_hints,
            entries_slimmed=entries_slimmed,
        )

    messages = compose("")
    checkpoint_hash = session_input_hash(
        messages,
        prompt_version=PROMPT_VERSION,
        call_config={
            "role": role.value,
            "output_limit": run.planning_limits.output_limit,
            "file_backed": file_ref is not None,
            "task_flow": "batched",
        },
        extra_identity={"task_fingerprint": run.task_fingerprint},
        execution_identity_override=getattr(run.client, "execution_identity", None),
    )
    search = run.external_injection
    cached = run.session_checkpoint_store.get(REQUEST_SESSION, window.chunk_id, checkpoint_hash)
    # A checkpoint that does not meet today's contract (an old reply that never
    # picked) is called again, never replayed.
    replayed = cached is not None and not reply_shape_errors(run, cached.content)
    attempt = 0
    shape_errors: List[str] = []
    if replayed:
        result = LLMCallResult(
            content=cached.content,
            role=role,
            model=str(cached.metadata.get("model") or "checkpoint"),
            fallback_used=bool(cached.metadata.get("fallback_used", False)),
            raw_response={},
        )
    else:

        def _complete(_injection: str = "") -> LLMCallResult:
            result = run.client.complete(
                role,
                compose,
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
                correct_only=run.profile.correct_only,
                # The delivery attempt continues this conversation.
                repair_session_key=f"correction-{window.chunk_id}",
                agent_task_extras=(
                    {"kb_tools": "read", "kb_signal_task": run.task_id, "kb_signal_window": window.chunk_id}
                    if run.knowledge_enabled
                    else None
                ),
                # The fill re-sends this whole turn.
                followup_expected=True,
                **validation_retry_sampling_kwargs(attempt),
            )
            if is_prompt_blocked(result.content, result.raw_response):
                raise GeminiPromptBlockedError(
                    f"窗口 {window.chunk_id} request turn prompt was blocked by the content filter"
                )
            return result

        while True:
            try:
                result = run_injection_ladder(
                    block=split_rendered_search_block(""),
                    call=_complete,
                    stage=f"correction_request_{window.chunk_id}",
                    blocked_exception=GeminiPromptBlockedError,
                    task_artifact_dir=run.task_artifact_dir,
                    task_id=run.task_id,
                    plain_retry=True,
                ).result
            except Exception as exc:  # pragma: no cover - provider behavior
                if run.task_artifact_dir:
                    append_task_artifact(
                        run.task_artifact_dir,
                        kind="correction_query_call_error",
                        task_id=run.task_id,
                        payload={
                            "chunk_id": window.chunk_id,
                            "task_flow": "batched",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "api_attempts": list(getattr(exc, "_harness_api_attempts", []) or []),
                            "execution_attempts": list(
                                getattr(exc, "_harness_execution_attempts", []) or []
                            ),
                            "route_decision": dict(getattr(exc, "_harness_route_decision", {}) or {}),
                        },
                    )
                if run.style_catalog is not None:
                    # The pick cannot be skipped: no request turn, no style.
                    raise
                return RequestTurnProduct(carried_entries_text=carried_entries_text)
            shape_errors = reply_shape_errors(run, result.content)
            if not shape_errors or attempt >= REQUEST_TURN_FORMAT_RETRIES:
                break
            if run.task_artifact_dir:
                append_task_artifact(
                    run.task_artifact_dir,
                    kind="correction_query_format_retry",
                    task_id=run.task_id,
                    payload={
                        "chunk_id": window.chunk_id,
                        "attempt": attempt,
                        "task_flow": "batched",
                        "errors": list(shape_errors),
                    },
                )
            # A malformed reply is still a call that was made and paid for.
            # Only the last one used to be recorded, so a retried window's
            # request turn under-reported both its request count and its
            # tokens. `_record_turn` will not seed the checkpoint from it (the
            # shape check gates that), and the retry event above stays: this
            # records the *call*, that one records the decision to redo it.
            _record_turn(
                run,
                window,
                result=result,
                messages=compose(result.variant),
                file_ref=file_ref,
                attempt=attempt,
                replayed=False,
                delivered=False,
                queries=[],
                window_notes="",
                requested=[],
                resolved_keys=[],
                missing=[],
                parse_error="; ".join(shape_errors),
                checkpoint_hash=checkpoint_hash,
                exchange_logger=exchange_logger,
            )
            attempt += 1
        messages = compose(result.variant)
        if shape_errors and run.style_catalog is not None:
            # The forced pick's end state (auto-style-gaps plan 「已裁定」 8):
            # the retries are spent and the reply still has not picked
            # validly. Parsing on would accept a direct delivery, or a request
            # with no style, as if no pick had been asked for.
            raise RuntimeError(
                f"Window {window.chunk_id} request turn: no valid style pick after "
                f"{attempt + 1} attempt(s): {'; '.join(shape_errors)}"
            )

    delivered = delivered_directly(result.content)
    parsed = (
        {"pairs": [], "requested": [], "keep": [], "parse_error": ""}
        if delivered
        else _parse_requests(
            result.content,
            remaining_new=remaining_new,
            search=search,
            queries_required=run.style_catalog is None,
        )
    )
    # `--style auto` on the fused window: the pick is a request-turn result
    # (auto-style plan §4), already validated by the shape check above; the
    # picked rules ride in the fill, before anything is translated.
    style_choice: StyleChoice | None = None
    style_fill_block = ""
    if run.style_catalog is not None:
        style_choice = run.style_catalog.choose(
            find_top_level_tag_blocks(result.content, STYLE_SELECTION_TAG)
        )
        style_fill_block = render_style_block(
            run.knowledge_root, style_choice.names, rev=style_choice.rev, correct_only=run.profile.correct_only
        )
        with run._tally_lock:
            if run.picked_style_block is None:
                run.picked_style_block = style_fill_block
        record_style_injected(
            run.knowledge_root, style_choice.names, task_id=run.task_id, rev=style_choice.rev
        )
        # the window's prompt already states the task's thresholds: the pick's tier only warns
        warn_window_pick(run.knowledge_root, style_choice.names, rev=style_choice.rev,
                         task_scale=window.length_scale)
        if run.task_artifact_dir:
            append_task_artifact(
                run.task_artifact_dir,
                kind="style_selection",
                task_id=run.task_id,
                payload={"chunk_id": window.chunk_id, **style_choice.to_dict()},
            )
    resolved_keys: List[str] = []
    missing: List[str] = []
    if parsed["requested"]:
        found, missing = load_entry_texts(run.knowledge_root, parsed["requested"])
        resolved_keys = list(found)
    queries = [query for query, _ in parsed["pairs"]]
    window_notes = "" if delivered else _extract_window_notes(
        result.content, count_tokens=run.token_counter.count_text
    )
    _record_turn(
        run,
        window,
        result=result,
        messages=messages,
        file_ref=file_ref,
        attempt=attempt,
        replayed=replayed,
        delivered=delivered,
        queries=queries,
        window_notes=window_notes,
        requested=parsed["requested"],
        resolved_keys=resolved_keys,
        missing=missing,
        parse_error=parsed["parse_error"],
        checkpoint_hash=checkpoint_hash,
        exchange_logger=exchange_logger,
    )
    if delivered:
        # No ``request_reply``: the delivery is attempt 0 itself, and a repair
        # of it must not see a fill turn claiming requests were served.
        return RequestTurnProduct(
            delivered=result,
            carried_entries_text=carried_entries_text,
            style_choice=style_choice,
        )
    search_text = ""
    if queries and run.search_client is not None:
        results = run.search_client.search_many(
            [SearchRequest(query=q, guided_query=g) for q, g in parsed["pairs"]],
            max_queries=run.max_search_queries_per_window,
        )
        rendered = render_search_results(
            results,
            max_total_tokens=injection_block_token_limit(run.max_search_queries_per_window),
            count_tokens=run.token_counter.count_text,
        )
        search_text = rendered.text
        if run.task_artifact_dir:
            append_task_artifact(
                run.task_artifact_dir,
                kind="correction_search_results",
                task_id=run.task_id,
                payload={
                    "chunk_id": window.chunk_id,
                    "task_flow": "batched",
                    "queries": queries,
                    "executed": search_results_metadata(results),
                    "rendered_tokens": rendered.tokens,
                    "render_report": rendered.report(),
                },
            )
    return RequestTurnProduct(
        search_results=search_text,
        window_notes=window_notes,
        requested_entry_keys=tuple(resolved_keys),
        keep_entry_keys=tuple(parsed["keep"]),
        request_reply=result.content,
        carried_entries_text=carried_entries_text,
        requested_search=bool(queries),
        requested_entries=bool(parsed["requested"] or parsed["keep"]),
        style_choice=style_choice,
        style_fill_block=style_fill_block,
    )


def _record_turn(
    run: CorrectionRun,
    window: SubtitleWindow,
    *,
    result: LLMCallResult,
    messages: List[Dict[str, Any]],
    file_ref: Any,
    attempt: int,
    replayed: bool,
    delivered: bool,
    queries: List[str],
    window_notes: str,
    requested: List[str],
    resolved_keys: List[str],
    missing: List[str],
    parse_error: str,
    checkpoint_hash: str,
    exchange_logger: ExchangeLogger | ExchangeBlock | None,
) -> None:
    """Exchange log, ``correction_query_response`` artifact, token row, checkpoint.

    A *direct delivery* is not a second call: the same ``LLMCallResult`` goes
    on to be attempt 0 of the window, which logs its own exchange and its own
    token row over the same ``raw_response``. Only the artifact is written
    here -- it carries ``delivered`` and the queries, which is the audit face
    for "how often is the request channel actually used" -- and the readers of
    that record know not to count it as a session (``exchange_metadata``).
    """

    finish_reason = _response_finish_reason(result.raw_response)
    output_limit_check = _output_limit_check(
        result.raw_response, result.requested_output_tokens or run.planning_limits.output_limit
    )
    output_limited = bool(output_limit_check["limited"])
    if not replayed and not delivered:
        run.token_rows.append(
            {
                "call": "correction_query",
                "chunk_id": window.chunk_id,
                "model": result.model,
                "finish_reason": finish_reason,
                "output_limit_check": output_limit_check,
                "tokens": extract_token_distribution(result.raw_response),
            }
        )
    pack = run.context_pack or ContextPack()
    input_components = correction_input_components(
        window=window,
        counter=run.token_counter,
        context_general=pack.general_text_for(window),
        context_window=pack.window_context_for(window, counter=run.token_counter),
        messages=messages,
        max_output_tokens=run.planning_limits.output_limit,
        file_ref=file_ref,
        video_high_resolution=result_uses_high_resolution_video(result),
    )
    session = f"correction-{window.chunk_id}-request-attempt{attempt}"
    if exchange_logger and not replayed and not delivered:
        exchange_logger.log(
            session,
            messages=messages,
            response_text=result.content,
            metadata=llm_exchange_metadata(
                result,
                session=session,
                input_components=input_components,
                chunk_id=window.chunk_id,
                queries="; ".join(queries) or ("（直接交付）" if delivered else "（无）"),
                window_notes_chars=len(window_notes),
                finish_reason=finish_reason,
                output_limited=output_limited,
                **({"parse_error": parse_error} if parse_error else {}),
            ),
        )
    if run.task_artifact_dir and not replayed:
        append_task_artifact(
            run.task_artifact_dir,
            kind="correction_query_response",
            task_id=run.task_id,
            payload={
                "session": session,
                "chunk_id": window.chunk_id,
                "attempt": attempt,
                "task_flow": "batched",
                "delivered": delivered,
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
                "requested_entries": requested,
                "resolved_entry_keys": resolved_keys,
                "missing_entries": missing,
                "response_content": result.content,
            },
        )
    if replayed:
        if run.task_artifact_dir:
            append_task_artifact(
                run.task_artifact_dir,
                kind="session_checkpoint_replay",
                task_id=run.task_id,
                payload={"session": REQUEST_SESSION, "key": window.chunk_id, "input_hash": checkpoint_hash},
            )
    elif (
        not reply_shape_errors(run, result.content)
        and not parse_error
        and not output_limited
        and getattr(result, "resumable", True)
    ):
        run.session_checkpoint_store.commit(
            session=REQUEST_SESSION,
            key=window.chunk_id,
            input_hash=checkpoint_hash,
            content=result.content,
            metadata={"model": result.model, "fallback_used": result.fallback_used},
        )
