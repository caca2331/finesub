"""The preparation stage's entry point and product.

``run_research_stage`` plans the correction windows, splits over-long material
into chunks, decides whether a persisted ``research-context.json`` is still
reusable and otherwise runs the preparation task (``stages.prepare_task``:
one logical conversation -- a request batch the harness fills once, then the
``<context_pack>`` delivery) and persists its payload. The request-block
parsers, the entry caps and the context-pack binding live here because they
predate the task and did not change with it.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace as dataclass_replace
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .client import (
    GeminiPromptBlockedError,
    RoleClient,
    extract_finish_reason,
    extract_token_distribution,
    is_likely_output_limited,
    is_prompt_blocked,
    sum_token_distributions,
    validation_retry_sampling_kwargs,
)
from .exchange_metadata import infer_session_name, llm_exchange_metadata, research_input_components
from .routing.config import (
    ANALYSIS_NOTES_MAX_TOKENS,
    DEFAULT_LIMITS,
    KB_PREINJECT_MAX_ENTRIES,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    LLMRole,
    ModelLimits,
    SESSION_OUTPUT_MAX_TOKENS,
    effective_window_subtitle_cap,
    TASK_FEEDBACK_MAX_TOKENS,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
    injection_block_token_limit,
    planning_limits_for,
    research_search_query_limit,
)
from finesub import settings
from finesub_bootstrap.fsops import write_atomic
from finesub.media.clips import probe_audio_duration
from finesub.reporting import current_reporter
from .chunking import (
    SubtitleSegment,
    SubtitleWindow,
    load_segments_from_stable_json,
    plan_correction_windows,
)
from .exchange_log import ExchangeLogger
from .injection_budget import render_knowledge_entries_block
from .knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    load_entry_texts,
    load_index_entries,
    index_texts_present,
    load_index_texts,
    load_preinjected_entries,
    match_terms,
    render_term_matches,
)
from .knowledge.node.presets import MATCHABLE_CATEGORIES
from .output_tags import (
    extract_single_tag_block,
    parse_json_tag_block,
    parse_line_items,
)
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile
from .prompt_compose import PROMPT_VERSION
from . import general_context
from .general_context import (
    NAME_FIELD_KEYS,
    NAME_TABLE_KEY,
    GeneralChunk,
    carry_text,
    fit_name_table,
    merge_name_tables,
)
from .prompts import ContextPack
from .session_checkpoint import SessionCheckpointStore, session_input_hash
from .token_budget import HeuristicTokenCounter, default_token_counter, TokenCounter
from .token_truncate import cap_tokens
from .web_search import (
    EXTRA_INFO_URL_EXTRACT_LIMIT,
    ExtractRequest,
    QueryExtractResult,
    WebSearchClient,
    extract_urls_from_text,
    render_extract_results,
)


@dataclass(frozen=True)
class ResearchRound1Result:
    requested_entries: tuple[str, ...] = ()
    keep_entries: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()
    analysis_notes: str = ""
    # Raw <task_update_feedback> body (with collection only).
    task_update_feedback: str = ""


def render_research_transcript(
    segments: Sequence[SubtitleSegment],
    windows: Sequence[SubtitleWindow],
) -> str:
    """Render ``id|text`` lines with ``--- window N ---`` markers at window starts.

    Overlapping segments are rendered once, under the window that introduced them.
    """

    printed: set[str] = set()
    lines: List[str] = []
    for window in windows:
        new_segments = [seg for seg in window.segments if seg.id not in printed]
        if not new_segments:
            continue
        lines.append(f"--- window {window.chunk_id} ---")
        for segment in new_segments:
            text = (segment.text or "").replace("\r\n", "\n").replace("\r", "\n")
            lines.append(f"{segment.id}|{text.replace(chr(10), ' ')}")
            printed.add(segment.id)
    return "\n".join(lines) + ("\n" if lines else "")


def _extract_optional_block(
    text: str,
    tag: str,
    *,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Best-effort optional tag extraction: missing/duplicated yields ''."""

    try:
        body = extract_single_tag_block(text, tag, required=False)
    except ValueError:
        return ""
    body = body.strip()
    if max_tokens is None:
        return body
    return cap_tokens(body, max_tokens, count_tokens)


def parse_round1_output(
    text: str,
    *,
    expect_queries: bool = True,
    expect_entries: bool = True,
    count_tokens: Callable[[str], int] | None = None,
) -> ResearchRound1Result:
    """Parse a request turn, requiring only the blocks this vector asked for.

    ``expect_queries`` / ``expect_entries`` mirror the prompt's own halves
    (docs/llm_harness_behavior.md) -- a block the prompt never described must not be demanded
    back, or every ``retrieval=none`` run would burn its parse retries.
    """

    # Not routed through ``prepare_task_contract``: the contract's job is the
    # replay validators; the parser only enforces the blocks it has to read.
    entries_body = extract_single_tag_block(
        text, "requested_entries", required=expect_entries
    )
    keep_body = extract_single_tag_block(text, "keep_entries", required=expect_entries)
    queries_body = extract_single_tag_block(
        text, "search_queries", required=expect_queries
    )
    return ResearchRound1Result(
        requested_entries=tuple(parse_line_items(entries_body)),
        keep_entries=tuple(parse_line_items(keep_body)),
        search_queries=tuple(parse_line_items(queries_body)),
        analysis_notes=_extract_optional_block(
            text,
            "analysis_notes",
            max_tokens=ANALYSIS_NOTES_MAX_TOKENS,
            count_tokens=count_tokens,
        ),
    )


def resolve_round1_entries(
    knowledge_root: str | Path,
    *,
    requested_names: Sequence[str],
    keep_names: Sequence[str],
    visible_keep_keys: Sequence[str],
    max_requested_entries: int = KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    max_keep_entries: int = KB_TRANSFER_MAX_ENTRIES,
    max_total_entries: int = KB_WINDOW_TOTAL_ENTRIES,
) -> tuple[dict[str, str], list[str], list[str], list[str]]:
    """Resolve R1 request/keep channels into one keep-first entry set.

    ``requested_names`` may name any indexed entry. ``keep_names`` may only
    resolve to an entry that was actually visible in R1's preinjection block
    (fully included or truncated, never wholly dropped). Canonical keys are
    deduped within each channel, capped independently, then merged keep-first
    under the shared total cap.

    Returns ``(selected, missing_requests, ignored_keeps, dropped_keys)``.
    """

    request_cap = max(0, int(max_requested_entries))
    keep_cap = max(0, int(max_keep_entries))
    total_cap = max(0, int(max_total_entries))
    visible = set(visible_keep_keys)
    requested: dict[str, str] = {}
    missing_requests: list[str] = []
    for name in requested_names:
        found, missing = load_entry_texts(knowledge_root, [name])
        missing_requests.extend(missing)
        for key, body in found.items():
            requested.setdefault(key, body)

    kept: dict[str, str] = {}
    ignored_keeps: list[str] = []
    for name in keep_names:
        found, missing = load_entry_texts(knowledge_root, [name])
        if missing:
            ignored_keeps.append(name)
            continue
        key, body = next(iter(found.items()))
        if key not in visible:
            ignored_keeps.append(name)
            continue
        kept.setdefault(key, body)

    capped_kept = dict(list(kept.items())[:keep_cap])
    capped_requested = dict(list(requested.items())[:request_cap])

    merged = dict(capped_kept)
    for key, body in capped_requested.items():
        merged.setdefault(key, body)
    selected = dict(list(merged.items())[:total_cap])
    selected_keys = set(selected)
    dropped_keys = [
        key
        for key in dict.fromkeys([*kept, *requested])
        if key not in selected_keys
    ]
    return selected, missing_requests, ignored_keeps, dropped_keys


def parse_round2_output(text: str) -> ContextPack:
    """The model's ``<context_pack>``, turned into the harness's unbound shape.

    The model lists the windows a note covers (``window_ids``), so one note can
    serve several windows; the harness keeps one note per window, so a shared
    note becomes a copy per listed window, and a window named by several notes
    gets them joined in output order. Only this layer reads ``window_ids``: the
    shapes the harness persists keep one ``window_id`` per note."""

    data = parse_json_tag_block(text, "context_pack")
    raw_notes = data.get("window_contexts") if isinstance(data, Mapping) else None
    if not isinstance(raw_notes, list):
        return ContextPack.from_dict(data)
    per_window: dict[str, list[str]] = {}
    for item in raw_notes:
        if not isinstance(item, Mapping):
            continue
        if "window_id" in item:
            raise ValueError(
                "window_contexts items list their windows in `window_ids` (an array), not `window_id`"
            )
        ids = item.get("window_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("every window_contexts item needs a non-empty `window_ids` array")
        context = str(item.get("context", "")).strip()
        for window_id in dict.fromkeys(str(value).strip() for value in ids):
            if window_id and context and context not in per_window.setdefault(window_id, []):
                per_window[window_id].append(context)
    return ContextPack.from_dict({
        **data,
        "window_contexts": [
            {"window_id": window_id, "context": "\n\n".join(texts)} for window_id, texts in per_window.items()
        ],
    })


def extract_round_task_feedback(
    text: str,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Best-effort ``<task_update_feedback>`` body (capped); '' on any issue.

    Feedback is advisory: a missing or malformed block never retries the
    turn — the context pack stays the hard output.
    """

    return _extract_optional_block(
        text,
        "task_update_feedback",
        max_tokens=TASK_FEEDBACK_MAX_TOKENS,
        count_tokens=count_tokens,
    )


def check_research_input_limit(
    messages: List[Dict[str, Any]],
    *,
    round_name: str,
    limits: ModelLimits = DEFAULT_LIMITS,
    counter: TokenCounter | None = None,
) -> int:
    counter = counter or default_token_counter()
    tokens = counter.count_texts(str(message.get("content", "")) for message in messages)
    if tokens > limits.prompt_input_limit:
        raise ValueError(
            f"Research {round_name} input (~{tokens} tokens) exceeds the prompt input "
            f"limit {limits.prompt_input_limit}. Split the audio into shorter clips first."
        )
    return tokens


def _render_note_url_extracts(
    extra_info: str,
    search_client: WebSearchClient,
    *,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, list[str], list[QueryExtractResult]]:
    """Extract up to eight deduped URLs from ``extra_info`` and deep-fetch them."""

    urls = extract_urls_from_text(extra_info)
    if not urls:
        return "", [], []
    results = search_client.extract_many([ExtractRequest(url=url) for url in urls])
    rendered = render_extract_results(
        results,
        max_total_tokens=injection_block_token_limit(EXTRA_INFO_URL_EXTRACT_LIMIT),
        count_tokens=count_tokens,
    ).text
    if rendered:
        rendered = f"<search_results>\n{rendered}\n</search_results>"
    return rendered, urls, results


def render_preinjected_entries(
    knowledge_root: str | Path,
    text: str,
    *,
    count_tokens: Callable[[str], int],
    max_entries: int = KB_PREINJECT_MAX_ENTRIES,
) -> tuple[str, Dict[str, Any]]:
    """Budget-rendered knowledge entries matched from note keywords, + report.

    Used by the preparation task's opening message.
    """

    entries, matches = load_preinjected_entries(
        knowledge_root, text, max_entries=max_entries
    )
    report: Dict[str, Any] = {"matches": [match.to_dict() for match in matches]}
    # Sub-entry hits, addressed by parent and section. A separate pass because
    # it answers a different question: `load_preinjected_entries` asks "was
    # this ENTRY named", which on transcript text almost never happens -- the
    # shadow ledger measured 4 subject hits against 347 term ones, 87% of them
    # under a subject nobody mentioned. Entry names are game titles and
    # streamer names; what people say are the characters inside them.
    term_matches = match_terms(
        knowledge_root,
        text,
        exclude_subjects=[match.key for match in matches],
    )
    if term_matches:
        report["term_matches"] = [match.to_dict() for match in term_matches]
    term_block = render_term_matches(term_matches)
    if not entries:
        return term_block, report
    block = render_knowledge_entries_block(
        entries,
        count_tokens=count_tokens,
        block_limit=injection_block_token_limit(max_entries),
    )
    report.update(block.report())
    text_out = "\n".join(part for part in (block.text, term_block) if part)
    return text_out, report


def _dump_research_input(
    task_artifact_dir: str | Path,
    round_name: str,
    payload: Mapping[str, Any],
) -> None:
    """Persist the exact semantic builder inputs for session replay."""

    path = Path(task_artifact_dir) / f"research-{round_name}-input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Backends whose event stream carries no result URLs at all, so an unmarked
# `sources` list there really is uncorroborated. Everything else keeps a
# per-call URL ledger: Codex and Claude Code from their own tool events, Gemini
# REST `google_search` from the grounding metadata the client now parses, an
# Anthropic endpoint from its search blocks. agy reports queries without URLs,
# and an OpenAI-compatible endpoint's `search_request` search reports nothing
# the harness can read; even there the gate below is the ledger being empty --
# a backend in this set that did report URLs is not marked.
_SOURCE_UNVERIFIABLE_BACKENDS = frozenset({"local_agent", "openai_compat"})


def _mark_unverified_sources(parsed: Any, execution_attempts: Any) -> Any:
    """Flag a native context pack's `sources` when nothing can corroborate them.

    The native prompt asks the round to cite what it verified, and the model
    complies -- but on agy the reply comes back with site-level "sources" and
    the stream carries no URL at all, so nothing downstream can check them. In
    the 2026-08-15 A/B the one checkable claim carried that way was a wrong
    official title that the local arm got right with three specific URLs.

    Scoped to the backends that genuinely report nothing. Marking on "no
    `search_events` with URLs" alone caught the **shipped default's** native
    path too, which is served by the paid Gemini target's `google_search`
    (`test_llm_switch_matrix` asserts that route exists): Gemini does return
    grounding URIs, and since 2026-08-15 the client parses them into the same
    ledger -- so that path now exits at the URL check above and never reaches
    the mark, which is the correct outcome and no longer an accident.
    """

    if not isinstance(parsed, ContextPack):
        # The preparation task's parser wraps the pack in its own reply type
        # (delivery or request batch); mark inside and hand the wrapper back,
        # or a native run would ship the model's self-reported `verified=true`
        # untouched. Anything else -- a request turn, another session's parse
        # result -- has no pack and passes through.
        inner = getattr(parsed, "context_pack", None)
        if not isinstance(inner, ContextPack):
            return parsed
        marked = _mark_unverified_sources(inner, execution_attempts)
        return parsed if marked is inner else dataclass_replace(parsed, context_pack=marked)
    general = dict(parsed.general_context or {})
    sources = general.get("sources")
    if not isinstance(sources, list) or not sources:
        return parsed
    saw_unverifiable_backend = False
    for attempt in execution_attempts or ():
        if not isinstance(attempt, Mapping):
            continue
        if str(attempt.get("backend") or "") in _SOURCE_UNVERIFIABLE_BACKENDS:
            saw_unverifiable_backend = True
        for event in attempt.get("search_events") or ():
            if isinstance(event, Mapping) and event.get("urls"):
                return parsed
    if not saw_unverifiable_backend:
        return parsed
    general["sources"] = [
        {**source, "verified": False} if isinstance(source, Mapping) else source
        for source in sources
    ]
    general["sources_note"] = (
        "模型自述来源；本次应答的 agent 后端不回报检索结果 URL，harness 无法核对，"
        "不可当作可追溯出处使用。"
    )
    return dataclass_replace(parsed, general_context=general)


def _call_and_parse(
    client: RoleClient,
    messages: List[Dict[str, Any]],
    *,
    parser,
    round_name: str,
    task_artifact_dir: str | Path | None,
    task_id: str,
    artifact_kind: str,
    max_parse_retries: int,
    token_rows: List[Dict[str, Any]] | None = None,
    exchange_logger: ExchangeLogger | None = None,
    token_counter: TokenCounter | None = None,
    input_components_kwargs: Dict[str, Any] | None = None,
    role: LLMRole = LLMRole.GENERAL_CAPABLE,
    file_ref: Any | None = None,
    session_prefix: str = "",
    checkpoint_store: SessionCheckpointStore | None = None,
    checkpoint_session: str = "",
    checkpoint_key: str = "main",
    checkpoint_extra_identity: Mapping[str, Any] | None = None,
    retrieval: str = "none",
    # The run's `--output-scale`. It multiplies the *reserve* only: how much of
    # a shared context this round is expected to spend on its answer. The
    # request itself is unrationed (the candidate's own ceiling), so scaling
    # here cannot truncate anything -- it only decides which candidates are
    # judged to fit (owner 2026-09-04).
    output_scale: float = 1.0,
    # v2 cell routing (plan §6): the preparation task runs on its own task
    # group.
    task_group: str = "research",
    difficulty: str = "",
    # §4.3 authorization matrix: a preparation turn that reads the knowledge
    # base grants the read-only kb tools (``kb_tools="read"``); one that only
    # has `--style auto` candidates to look at gets the narrow ``"style"``
    # grant, whose root and revision ride in ``kb_binding`` (no generation
    # pin to borrow them from); other sessions leave it off and the tools
    # never attach. ``kb_signal_window`` names the window this call serves in
    # the exposure ledger (the task-wide preparation leaves it empty).
    kb_tools: str = "",
    kb_binding: Mapping[str, str] | None = None,
    # The conversation this round may continue (a preparation task's t2 continues t1).
    session_key: str = "",
    # This round's reply may be continued (t1 with requests): `RoleClient.complete`.
    followup_expected: bool = False,
    kb_signal_window: str = "",
    # The parsed return value drops the LLMCallResult; a caller that needs the
    # raw response afterwards (retrieval-ledger URL backfill) passes a list to
    # collect the answering attempt's raw_response.
    raw_response_sink: list[Any] | None = None,
):
    checkpoint_hash = ""
    if checkpoint_store is not None and checkpoint_session:
        checkpoint_hash = session_input_hash(
            messages,
            prompt_version=PROMPT_VERSION,
            call_config={
                "role": role.value,
                "output_reserve": SESSION_OUTPUT_MAX_TOKENS,
                "file_backed": file_ref is not None,
                # A native-search reply is not interchangeable with a plain one
                # for the same prompt: it carries grounded facts this checkpoint
                # would otherwise replay as if the tool had run.
                "native_search": retrieval == "native",
            },
            extra_identity=checkpoint_extra_identity,
            execution_identity_override=getattr(
                client, "execution_identity", None
            ),
        )
        cached = checkpoint_store.get(
            checkpoint_session, checkpoint_key, checkpoint_hash
        )
        if cached is not None:
            try:
                parsed = parser(cached.content)
            except (ValueError, json.JSONDecodeError):
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="session_checkpoint_invalid",
                        task_id=task_id,
                        payload={
                            "session": checkpoint_session,
                            "key": checkpoint_key,
                            "input_hash": checkpoint_hash,
                        },
                    )
            else:
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="session_checkpoint_replay",
                        task_id=task_id,
                        payload={
                            "session": checkpoint_session,
                            "key": checkpoint_key,
                            "input_hash": checkpoint_hash,
                        },
                    )
                return parsed

    last_error: Exception | None = None
    session_base = artifact_kind.removesuffix("_response")
    for attempt in range(max_parse_retries + 1):
        try:
            result = client.complete(
                role,
                messages,
                # A reserve, not a cap: the request fills whichever candidate's
                # own ceiling answers, and this only says how much of a shared
                # context to keep free for it (owner 2026-09-04).
                output_reserve=math.ceil(output_scale * SESSION_OUTPUT_MAX_TOKENS),
                file_ref=file_ref,
                retrieval=retrieval,
                task_group=task_group,
                difficulty=difficulty,
                agent_task_extras=(
                    {
                        "kb_tools": kb_tools,
                        **dict(kb_binding or {}),
                        "kb_signal_task": task_id,
                        **({"kb_signal_window": kb_signal_window} if kb_signal_window else {}),
                    }
                    if kb_tools
                    else None
                ),
                repair_session_key=session_key,
                followup_expected=followup_expected,
                **validation_retry_sampling_kwargs(attempt),
            )
        except Exception as exc:  # pragma: no cover - provider behavior
            if task_artifact_dir:
                append_task_artifact(
                    task_artifact_dir,
                    kind=f"{session_base}_call_error",
                    task_id=task_id,
                    payload={
                        "attempt": attempt,
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
        # Against what was actually requested -- the answering model's own
        # ceiling -- not against the planning reserve, which is no longer a cap
        # and would flag every healthy long answer as truncated.
        output_limited = is_likely_output_limited(
            result.raw_response,
            max_tokens=result.requested_output_tokens or SESSION_OUTPUT_MAX_TOKENS,
        )
        finish_reason = extract_finish_reason(result.raw_response)
        prompt_blocked = is_prompt_blocked(result.content, result.raw_response)
        session_payload = {"attempt": attempt}
        if session_prefix:
            session_payload["session"] = f"{session_prefix}-attempt{attempt}"
        session = infer_session_name(artifact_kind, session_payload)
        input_components = research_input_components(
            counter=token_counter,
            messages=messages,
            **(input_components_kwargs or {}),
        )
        if token_rows is not None:
            token_rows.append(
                {
                    "call": session_base,
                    "attempt": attempt,
                    "model": result.model,
                    "tokens": extract_token_distribution(result.raw_response),
                }
            )
        try:
            parsed = parser(result.content or "")
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            parse_error = str(exc)
        else:
            parse_error = ""
            if retrieval == "native":
                parsed = _mark_unverified_sources(
                    parsed, result.execution_attempts
                )
        if prompt_blocked:
            parse_error = (
                f"prompt blocked by content filter (finish_reason={finish_reason})"
            )
        if task_artifact_dir:
            append_task_artifact(
                task_artifact_dir,
                kind=artifact_kind,
                task_id=task_id,
                payload={
                    "session": session,
                    "attempt": attempt,
                    "model": result.model,
                    "fallback_used": result.fallback_used,
                    "usage": extract_token_distribution(result.raw_response),
                    "input_components": input_components,
                    "output_limited": output_limited,
                    "finish_reason": finish_reason,
                    "parse_error": parse_error,
                    "api_attempts": list(result.api_attempts),
                    "execution_attempts": list(result.execution_attempts),
                    "route_decision": dict(result.route_decision),
                    "response_content": result.content,
                },
            )
        if exchange_logger:
            exchange_logger.log(
                session,
                messages=messages,
                response_text=result.content,
                metadata=llm_exchange_metadata(
                    result,
                    session=session,
                    input_components=input_components,
                    attempt=attempt,
                    output_limited=output_limited,
                    finish_reason=finish_reason,
                    **({"parse_error": parse_error} if parse_error else {}),
                ),
            )
        if prompt_blocked:
            # Deterministic for the exact prompt — retrying unchanged wastes
            # quota. Callers may rebuild without optional injected blocks.
            raise GeminiPromptBlockedError(
                f"Research {round_name} prompt was blocked by the content "
                f"filter (finish_reason={finish_reason})."
            )
        if not parse_error:
            if raw_response_sink is not None:
                raw_response_sink.append(result.raw_response)
            # Gate D answer C (docs/llm_local_agent.md §7): a call that leaned
            # on implicit provider history is not replayable from its hash, so
            # it must not seed the L1 store. A later resume re-sends this one
            # call; nothing upstream is invalidated.
            if (
                checkpoint_store is not None
                and checkpoint_hash
                and getattr(result, "resumable", True)
            ):
                checkpoint_store.commit(
                    session=checkpoint_session,
                    key=checkpoint_key,
                    input_hash=checkpoint_hash,
                    content=result.content or "",
                    metadata={
                        "model": result.model,
                        "fallback_used": result.fallback_used,
                        "role": role.value,
                    },
                )
            return parsed
        if file_ref is not None and not (result.content or "").strip():
            # A media-backed call that returns literally nothing is almost
            # always transient file readiness (upload probe should prevent it,
            # but keep a backstop) — give the backend time before retrying.
            time.sleep(min(30.0, 10.0 * (attempt + 1)))
    raise RuntimeError(
        f"Research {round_name} output could not be parsed after "
        f"{max_parse_retries + 1} attempts: {last_error}"
    )


def load_research_context(path: str | Path) -> ContextPack:
    """Load a persisted context pack; refuse one whose notes lack ranges.

    Unbound notes cannot be injected, so serving such a pack would hand the
    correction stage a silently halved research product -- the general context
    only -- for work the user already paid quota for. Re-running the stage is
    the same trade every other reuse gate here makes: cost a stage, stay whole.
    """

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    pack = ContextPack.from_dict(data.get("context_pack", data))
    if pack.has_unbound_window_contexts:
        raise ValueError(
            "research window contexts carry no source-id ranges "
            f"({len(pack.unbound_window_contexts)} notes); the artifact predates "
            "interval addressing and cannot be injected"
        )
    return pack


# ---------------------------------------------------------------------------
# Research acquisition stage (plans windows, runs research, persists context)


def _short_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()[:16]


def stable_json_source_hash(stable_json: str | Path) -> str:
    """Identity of the persisted source, parsed rather than read as bytes.

    Reformatting the JSON or touching a field no stage reads is not a source
    change.  Execution choices (model, preset, policy, retrieval knobs) are
    deliberately absent: they change how unfinished work runs without making an
    already committed result unusable (docs/llm_local_agent.md §11).
    """

    segments = load_segments_from_stable_json(stable_json)
    payload = [
        [segment.id, segment.start, segment.end, segment.text]
        for segment in segments
    ]
    return _short_sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def plan_geometry_metadata(
    profile: TranslationProfile,
    *,
    audio_duration: float | None = None,
    max_window_subtitle_tokens: int | None = None,
    limits: ModelLimits | None = None,
) -> dict:
    """Audit metadata describing what placed the original window boundaries.

    Geometry is not a reuse gate: correction plans preserve boundary identity
    and refit pending leaves, while research notes are source-interval addressed.

    ``max_window_subtitle_tokens`` is resolved through the same ``limits`` the
    planner uses rather than recorded raw: ``None`` (unset, take the limits
    default) and ``0`` (cap disabled) plan different windows.
    """

    effective_limits = limits or DEFAULT_LIMITS
    return {
        "prompt_version": PROMPT_VERSION,
        "context_reserve_tokens": WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
        "geometry_profile_id": profile.geometry_id,
        "max_window_subtitle_tokens": effective_window_subtitle_cap(
            max_window_subtitle_tokens, effective_limits
        ),
        # The bound group's planning envelope (D13).
        "input_limit": effective_limits.prompt_input_limit,
        "output_limit": effective_limits.output_limit,
        "audio_duration": None
        if audio_duration is None
        else round(float(audio_duration), 3),
    }


def backup_unrecoverable_json(path: str | Path) -> Path:
    """Copy an incompatible JSON artifact aside before a caller replaces it."""

    source = Path(path)
    candidate = source.with_name(f"{source.name}.invalid")
    suffix = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.name}.invalid.{suffix}")
        suffix += 1
    shutil.copy2(source, candidate)
    return candidate


def count_indexed_entries(knowledge_root: str | Path) -> int:
    """How many entries the indices offer.

    Recorded alongside R1's request cap so a later reader can tell a genuine
    selection from "the whole knowledge base fit under the ceiling".
    """

    return sum(
        len(load_index_entries(knowledge_root, category))
        for category in MATCHABLE_CATEGORIES
    )


def knowledge_index_available(
    knowledge_root: str | Path, *, knowledge_enabled: bool = True
) -> bool:
    """Whether there is an index to pick entries off."""

    if not knowledge_enabled:
        return False
    return index_texts_present(load_index_texts(knowledge_root))


def research_knowledge_inputs_hash(
    knowledge_root: str | Path,
    extra_info: str,
) -> str:
    """Hash the knowledge text that can enter research before web search.

    ⚠ It has to cover the sub-entry hits too. Those reach the prompt without
    any entry-level match (measured: 87% of term hits sit under a subject that
    was never named), so a hash over entries alone reports "the knowledge input
    did not change" while the injected text did -- which is the one thing this
    hash exists to prevent.
    """

    preinjected, matches = load_preinjected_entries(knowledge_root, extra_info)
    terms = match_terms(
        knowledge_root, extra_info, exclude_subjects=[match.key for match in matches]
    )
    payload = {
        "knowledge_indexes": load_index_texts(knowledge_root),
        "preinjected_entries": preinjected,
        "preinjected_terms": [match.to_dict() for match in terms],
    }
    return _short_sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def planning_metadata(
    profile: TranslationProfile,
    *,
    stable_json: str | Path,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    collect_task_feedback: bool = False,
    audio_duration: float | None = None,
    max_window_subtitle_tokens: int | None = None,
    limits: ModelLimits | None = None,
    test_profile: bool = False,
    execution_identity_override: Mapping[str, Any] | None = None,
) -> dict:
    """Provenance for a persisted research context.

    The whole dict is recorded; :func:`research_reuse_key` selects the narrow
    semantic subset that gates reuse.

    ``knowledge_enabled`` is recorded explicitly rather than left to
    ``knowledge_inputs_hash``: with knowledge off the harness never reads the
    indices, so the hash of an unread base would match a run that did read it.
    Keeping the switch beside the hash keeps "not read" and "read an empty
    base" distinguishable in task reports.
    """

    from .routing.execution_policy import execution_identity

    source_path = Path(stable_json).expanduser()
    return {
        **plan_geometry_metadata(
            profile,
            audio_duration=audio_duration,
            max_window_subtitle_tokens=max_window_subtitle_tokens,
            limits=limits,
        ),
        # Two of the three are gates; the difference matters.
        #
        # ``profile_id`` is the full vector and is audit-only (exempt): it
        # carries continuity and difficulty, which never reach the research
        # rounds. Gating on them would make "switch to intermediate and
        # continue" -- the documented way out of an exhausted free-tier capable
        # quota -- re-run both rounds, burning the very quota the switch exists
        # to stop burning.
        #
        # ``planning_profile_id`` is audit-only too: it still carries
        # ``correction_media`` (geometry), which the preparation task does
        # not read. The narrower ``research_semantics_id`` is the gate.
        "profile_id": profile.profile_id,
        "planning_profile_id": dataclass_replace(
            profile, continuity="serial", difficulty="quality"
        ).profile_id,
        "research_semantics_id": research_semantics_id(profile),
        "output_scale": profile.output_scale,
        "stable_json_hash": stable_json_source_hash(source_path),
        "extra_info_hash": _short_sha256((extra_info or "").encode("utf-8")),
        "knowledge_inputs_hash": research_knowledge_inputs_hash(
            knowledge_root, extra_info
        )
        if knowledge_enabled
        else "",
        "knowledge_enabled": bool(knowledge_enabled),
        "collect_task_feedback": bool(collect_task_feedback),
        # A gate at both L2 and L3: a smoke run and a real run must not mix
        # their artifacts in either direction. (Absent here until 2026-08-12,
        # which let a --test-profile context be reused by a real run.)
        "test_profile": bool(test_profile),
        "execution_identity": dict(
            execution_identity_override or execution_identity()
        ),
    }


def research_semantics_id(profile: TranslationProfile) -> str:
    """What the preparation task's product depends on besides its input: the retrieval it could use, and for a
    correct-only task the mode (its context pack names terms for the subtitle as written, and its style pick is
    among the styles written for correction). A translation task's id is the retrieval alone, as before."""

    return profile.retrieval + (",correct_only" if profile.correct_only else "")


# What a persisted research context is compared on (L3). Window geometry is
# deliberately absent: interval-addressed notes remap onto any later correction
# plan. What remains is the source and the two knobs a user turns *at this
# artifact* -- reusing it past those would ignore a request.
#
#   research_semantics_id  = retrieval: what the preparation task was allowed
#                            to fetch (nothing / the local agent / its own
#                            search), which its product depends on; and the
#                            task's mode (`research_semantics_id`).
#
# The prompt version is audit-only: it is the correction's `PROMPT_VERSION`,
# not the preparation task's, and the stored context is parsed again on load
# (`load_research_context`): one that no longer reads is copied aside and redone.
RESEARCH_REUSE_GATES = (
    "stable_json_hash",
    "extra_info_hash",
    "test_profile",
    "research_semantics_id",
)

# Recorded for audit and for task reports, never a reason to throw a committed
# stage away. Split out from the gates so that adding a field to
# ``planning_metadata`` without classifying it is a test failure rather than a
# silent reuse of something it should have invalidated.
RESEARCH_REUSE_AUDIT_ONLY = (
    "prompt_version",
    "execution_identity",
    "knowledge_inputs_hash",
    "knowledge_enabled",
    "collect_task_feedback",
    "profile_id",
    "planning_profile_id",
    "output_scale",
    # plan_geometry_metadata's payload
    "context_reserve_tokens",
    "geometry_profile_id",
    "max_window_subtitle_tokens",
    "input_limit",
    "output_limit",
    "audio_duration",
)


def research_reuse_key(planning: Mapping[str, Any]) -> dict:
    """The comparable part of a research context's planning metadata."""

    return {key: planning.get(key) for key in RESEARCH_REUSE_GATES}


# --- chunked research for ultra-long material (task-parallelism plan W7 / P8) ---

def research_transcript_cap(profile: TranslationProfile) -> int:
    """Transcript size past which the preparation task runs in chunks, each a whole task.

    `[chunking] research_transcript_max_tokens` when set, else the preparation cell's planning limits
    (`ModelLimits.research_transcript_max_tokens`, 50k since owner 2026-11-08); 0 disables. ``retrieval=native``
    never chunks. Below the cap the single-task path runs unchanged."""

    if profile.retrieval == "native":
        return 0
    explicit = settings.RESEARCH_TRANSCRIPT_MAX_TOKENS.resolve()
    if explicit is not None:
        return int(explicit)
    return planning_limits_for("research", profile.difficulty).research_transcript_max_tokens


def plan_research_chunks(
    windows: Sequence[SubtitleWindow],
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[list[SubtitleWindow]]:
    """Split the research windows into transcript chunks of whole windows.

    Greedy and deterministic: windows accumulate in order until the next one
    would push the chunk's rendered transcript past ``max_tokens``; a window
    is never split (its note ranges must stay meaningful), so a single window
    larger than the cap becomes its own chunk. ``max_tokens <= 0`` disables
    chunking."""

    if max_tokens <= 0 or not windows:
        return [list(windows)]
    chunks: list[list[SubtitleWindow]] = []
    current: list[SubtitleWindow] = []
    current_tokens = 0
    printed: set[str] = set()
    for window in windows:
        lines = [f"--- window {window.chunk_id} ---"] + [
            f"{segment.id}|{segment.text or ''}"
            for segment in window.segments
            if segment.id not in printed
        ]
        printed.update(segment.id for segment in window.segments)
        window_tokens = count_tokens("\n".join(lines))
        if current and current_tokens + window_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(window)
        current_tokens += window_tokens
    if current:
        chunks.append(current)
    return chunks


def warn_dropped_general_chunks(pack: ContextPack) -> None:
    """One warning per chunk whose windows cannot read every other chunk's general context, or whose
    own block alone is past the ceiling (every window of a chunk reads the same blocks)."""
    for chunk in pack.general_chunks:
        selection = pack.chunk_selection((chunk.chunk_id,))
        if selection.fallback:
            return
        if selection.dropped:
            current_reporter().warning(
                "research-general-chunks-dropped",
                f"准备任务分块 {chunk.chunk_id} 的窗口放不下 {', '.join(selection.dropped)} 的通用背景"
                f"（上限 {general_context.GENERAL_MAX_TOKENS} tokens，由近到远取块）",
                impact="这些窗口读不到这几块的全局结论；它们自己那块与逐窗笔记照常",
            )
        if selection.own_over_cap:
            current_reporter().warning(
                "research-general-chunk-over-cap",
                f"准备任务分块 {chunk.chunk_id} 自己的通用背景超过 {general_context.GENERAL_MAX_TOKENS} tokens",
                impact="照样整块给它的窗口，别的块都放不下",
            )


def _windows_keep_cap(
    profile: TranslationProfile,
    knowledge_root: str | Path,
    *,
    knowledge_enabled: bool,
) -> int:
    """How many kept keys the assembled pack hands the windows: the windows'
    answer, not the preparation task's -- a pruned reserve only where a later
    window can ask for a dropped entry back (`task_graph.window_flow`)."""

    from .stages.task_graph import window_requests

    readable = knowledge_index_available(knowledge_root, knowledge_enabled=knowledge_enabled)
    return (
        KB_TRANSFER_MAX_ENTRIES
        if window_requests(profile, knowledge_readable=readable)
        else KB_WINDOW_TOTAL_ENTRIES
    )


#: Counts the names handed between chunks and kept for the whole file: the same on every machine (the local
#: Gemini counter is not everywhere), since the carried text is part of the next chunk's input hash.
_NAME_COUNTER = HeuristicTokenCounter()


def _chunk_blocks(outcomes: Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    """``(chunk id, general block)`` of each chunk that delivered one, in chunk order."""
    return [(outcome["record"]["chunk_id"], outcome["pack"].general_context)
            for outcome in outcomes if outcome["pack"].general_context]


def _merge_chunk_packs(outcomes: Sequence[Mapping[str, Any]], chunk_plan: Sequence[Sequence[SubtitleWindow]],
                       segments: Sequence[SubtitleSegment]) -> ContextPack:
    """The chunks' packs as one. The names go into one table for the whole file, the first chunk to name
    something winning (`general_context.merge_name_tables`); every chunk's block keeps the rest, with the first
    and last source id its research windows' bodies cover, so each correction window reads the table, its own
    block and the nearest others that fit; the window notes are concatenated (their intervals are disjoint
    across chunks); the user's requests are every chunk's, deduplicated, so one a chunk relayed and another
    missed still reaches every window."""
    notes: list[Any] = []
    general: dict[str, Mapping[str, Any]] = {}
    chunks: list[GeneralChunk] = []
    for outcome, windows in zip(outcomes, chunk_plan):
        pack = outcome["pack"]
        notes.extend(pack.window_contexts)
        if pack.general_context:
            chunk_id = outcome["record"]["chunk_id"]
            general[chunk_id] = {key: value for key, value in pack.general_context.items()
                                 if key not in NAME_FIELD_KEYS}
            bodies = [segment.id for window in windows for segment in window.segments[len(window.overlap_segments):]]
            chunks.append(GeneralChunk(chunk_id, bodies[0], bodies[-1]))
    table, _relisted = merge_name_tables(_chunk_blocks(outcomes))
    if table:
        table, over = fit_name_table(table, count=_NAME_COUNTER.count_text)
        if over:
            current_reporter().warning(
                "research-name-table-over-cap",
                f"分块准备任务的全片名字表去掉说明之后仍超过 {general_context.NAME_TABLE_MAX_TOKENS} tokens",
                impact="照样整张给每个纠错窗口（窗口的人名只在这张表里），挤掉的是别的块的通用背景",
            )
        general = {NAME_TABLE_KEY: table, **general}
    merged = ContextPack(
        general_context=general, window_contexts=tuple(notes), general_chunks=tuple(chunks),
        user_requests=tuple(dict.fromkeys(r for outcome in outcomes for r in outcome["pack"].user_requests)),
    )
    warn_dropped_general_chunks(merged.with_source_order([segment.id for segment in segments]))
    return merged


_IDENTITY_LABELS = {"zh": "中文写法", "type": "类型", "role": "身份"}


def _identity_text(fields: Mapping[str, Any]) -> str:
    return "、".join(f"{_IDENTITY_LABELS.get(name, name)} {value or '（空）'}" for name, value in fields.items())


def _warn_name_disputes(records: Sequence[Mapping[str, Any]]) -> None:
    """One warning listing where a later chunk disagreed with the names the earlier ones settled: what it said
    in ``name_disputes``, and what it re-listed another way (dropped, the first spelling kept)."""
    lines = [
        *(f"{record['chunk_id']}：{item.get('name', '')} → {item.get('suggest', '')}（{item.get('reason', '')}）"
          for record in records for item in record.get("name_disputes") or ()),
        *(f"{record['chunk_id']}：{item['key']} 写成 {_identity_text(item['dropped'])}，"
          f"保留前面的 {_identity_text(item['kept'])}"
          for record in records for item in record.get("relisted") or ()),
    ]
    if lines:
        current_reporter().warning(
            "research-name-disputes",
            "分块准备任务里，靠后的段对前面定的写法有不同意见：\n" + "\n".join(lines),
            impact="全片按先定的写法；觉得后面的对，把写法写进 --extra-info 再跑，准备任务会自动重做；"
                   "改的是知识库，要先删掉 research-context*.json（含各块的文件）",
        )


def _run_chunked_research(
    chunk_plan: Sequence[Sequence[SubtitleWindow]],
    *,
    segments: Sequence[Any],
    context_file: Path,
    extra_info: str,
    knowledge_root: str | Path,
    knowledge_enabled: bool,
    profile: TranslationProfile,
    test_profile: bool,
    task_artifact_dir: str | Path | None,
    task_id: str,
    token_counter: TokenCounter,
    collect_task_feedback: bool,
    resume: bool,
    client: RoleClient,
    transcript_cap: int,
    research_runner: Callable[..., Dict[str, Any]],
    research_runner_kwargs: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run research per chunk and assemble one payload (plan W7 / P8).

    Every chunk runs the complete preparation task over its own slice of
    the transcript and lands in its own ``<context>-<chunk>.json`` (the resume
    unit: a rerun replays finished chunks off their input hash). Only after
    every chunk is done is the final pack assembled -- correction never starts
    against a partial context. Assembly is a pure function: fixed chunk ids,
    fixed order, fixed instruction text, so two runs over the same inputs
    produce comparable injections (the pack is not fingerprinted since
    2026-08-12, so this is comparability, not cache identity).

    Chunks run one after another: from the second on, a chunk is handed the names and summaries the chunks
    before it settled (`general_context.carry_text`), so a long file is named once, not once per chunk; its
    disagreements are collected for the user. The per-artifact-dir writers are shared single instances -- the
    exchange logger, the checkpoint store, the search client.
    """

    # `--style auto` on an over-long material (auto-style-gaps plan B, owner
    # 2026-10-01: votes, overturning auto-style §5's "no majority vote"): every
    # chunk's preparation task sees the catalog and picks under the forced-pick
    # contract; each pick is that chunk's vote, tallied below.
    chunk_runner_kwargs = dict(research_runner_kwargs or {})
    style_catalog = chunk_runner_kwargs.get("style_catalog")
    auto_style = bool(style_catalog)

    current_reporter().debug(
        "research-chunked",
        {"chunks": len(chunk_plan), "cap_tokens": transcript_cap},
    )
    shared_exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    shared_checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)
    shared_search_client = (
        WebSearchClient(execution_settings=getattr(client, "execution_settings", None))
        if profile.external_injection
        else None
    )

    def _run_chunk(index: int, earlier: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """One chunk end to end, handed what the ``earlier`` chunks settled; returns the chunk record with its
        parts."""

        chunk_windows = chunk_plan[index]
        chunk_id = f"c{index + 1:02d}"
        chunk_transcript = render_research_transcript(segments, chunk_windows)
        chunk_segment_count = sum(
            len(window.segments[len(window.overlap_segments):])
            for window in chunk_windows
        )
        # The same gates as the stage-level L3 reuse (RESEARCH_REUSE_GATES),
        # with the chunk's own transcript standing in for stable_json_hash --
        # finer-grained, and geometry changes reshape it anyway. Difficulty and
        # the prompt version are deliberately absent, exactly as they are absent
        # from the gates: "switch to intermediate and continue" must replay
        # finished chunks, not burn quota re-running them (reviewer 2026-08-30
        # P1-2: the first cut missed test_profile and over-gated difficulty).
        carried, carry_over = carry_text(_chunk_blocks(earlier), count=_NAME_COUNTER.count_text)
        if carry_over:
            current_reporter().warning(
                "research-carry-over-cap",
                f"带给准备任务分块 {chunk_id} 的前面各段名字表去掉说明与摘要之后仍超过 "
                f"{general_context.NAME_TABLE_MAX_TOKENS} tokens",
                impact="照样整张带给它（看不到的条目会被它按自己的写法重写）",
            )
        # What the earlier chunks delivered is part of this chunk's input: one re-run before it re-runs this
        # one. The first chunk carries nothing, so its hash is what it always was.
        chunk_input_hash = _short_sha256(
            json.dumps(
                {
                    "transcript": chunk_transcript,
                    "extra_info": extra_info,
                    "test_profile": bool(test_profile),
                    "research_semantics_id": research_semantics_id(profile),
                    **({"earlier_chunks": carried} if carried else {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
        chunk_file = context_file.with_name(
            f"{context_file.stem}-{chunk_id}{context_file.suffix}"
        )
        payload: Dict[str, Any] | None = None
        if resume and chunk_file.exists():
            try:
                saved = json.loads(chunk_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = None
            if isinstance(saved, dict) and saved.get("chunk_input_hash") == chunk_input_hash:
                try:
                    ContextPack.from_dict(saved["context_pack"])
                    payload = saved
                except (KeyError, TypeError, ValueError):
                    # The prompt version is no longer in the hash, so a chunk in a shape this code cannot
                    # read is redone here rather than failing the stage on every resume.
                    payload = None
            if payload is not None and auto_style and not isinstance(
                payload.get("style_selection"), Mapping
            ):
                # A chunk finished before it carried a vote (or under a run
                # without `--style auto`): replaying it would drop its vote, so
                # this one chunk runs again -- the shape change that loses data
                # (README_DEV 复用的依据是任务身份).
                payload = None
        replayed = payload is not None
        if payload is None:
            payload = research_runner(
                **chunk_runner_kwargs,
                transcript=chunk_transcript,
                extra_info=extra_info,
                knowledge_root=knowledge_root,
                knowledge_enabled=knowledge_enabled,
                search_client=shared_search_client,
                profile=profile,
                test_profile=test_profile,
                task_artifact_dir=task_artifact_dir,
                task_id=task_id,
                max_search_queries=research_search_query_limit(chunk_segment_count),
                token_counter=token_counter,
                collect_task_feedback=collect_task_feedback,
                resume=resume,
                client=client,
                chunk_label=chunk_id,
                exchange_logger=shared_exchange_logger,
                checkpoint_store=shared_checkpoint_store,
                earlier_chunks=carried,
            )
            payload["chunk_input_hash"] = chunk_input_hash
            payload["chunk_id"] = chunk_id
            payload["chunk_window_ids"] = [window.chunk_id for window in chunk_windows]
            write_atomic(chunk_file, json.dumps(payload, ensure_ascii=False, indent=2))
        # Bind against this chunk's own windows: their source-id intervals are
        # disjoint across chunks, which is what makes the note merge a plain
        # concatenation.
        bind_report: dict[str, Any] = {}
        pack = ContextPack.from_dict(payload["context_pack"]).bind_window_ranges(
            list(chunk_windows), report_sink=bind_report
        )
        if bind_report.get("unplaceable_window_ids"):
            current_reporter().warning(
                "research-notes-unplaceable",
                f"research chunk {chunk_id} named "
                f"{len(bind_report['unplaceable_window_ids'])} window(s) outside "
                f"its slice ({', '.join(bind_report['unplaceable_window_ids'])})",
                impact=f"这些笔记被丢弃，该块剩余 {bind_report.get('bound', 0)} 条",
            )
        # Recomputed on a replay too: the carry and this chunk's delivery are both on disk.
        _table, relisted = merge_name_tables([*_chunk_blocks(earlier), (chunk_id, pack.general_context)])
        return {
            "pack": pack,
            "payload": payload,
            "record": {
                "chunk_id": chunk_id,
                **({"name_disputes": [dict(item) for item in pack.name_disputes]} if pack.name_disputes else {}),
                **({"relisted": [item for item in relisted if item["chunk"] == chunk_id]}
                   if any(item["chunk"] == chunk_id for item in relisted) else {}),
                "window_ids": [window.chunk_id for window in chunk_windows],
                "segment_count": chunk_segment_count,
                "input_hash": chunk_input_hash,
                "file": chunk_file.name,
                "replayed": replayed,
                "bound_notes": bind_report.get("bound", 0),
            },
        }

    outcomes: list[dict[str, Any]] = []
    for index in range(len(chunk_plan)):
        outcomes.append(_run_chunk(index, outcomes))
    _warn_name_disputes([outcome["record"] for outcome in outcomes])

    chunk_records: list[dict[str, Any]] = []
    keep_union: list[str] = []
    injected_union: list[str] = []
    token_totals: list[Mapping[str, Any]] = []
    rounds_meta: Mapping[str, Any] = {}
    for outcome in outcomes:
        payload = outcome["payload"]
        keep_union.extend(payload.get("keep_entries") or [])
        injected_union.extend(payload.get("injected_entries") or [])
        totals = (payload.get("token_report") or {}).get("totals")
        if isinstance(totals, Mapping):
            token_totals.append(totals)
        rounds_meta = payload.get("rounds") or rounds_meta
        chunk_records.append(outcome["record"])
    merged_pack = _merge_chunk_packs(outcomes, chunk_plan, segments)
    keep_cap = _windows_keep_cap(profile, knowledge_root, knowledge_enabled=knowledge_enabled)
    style_choice = None
    if auto_style:
        from .knowledge.style import StyleChoice, style_vote_facts, tally_style_votes

        # Each chunk's pick already passed that chunk's own delivery check
        # (the forced-pick contract), so there is no invalid name to handle.
        style_choice = tally_style_votes(
            [
                (outcome["record"]["chunk_id"], StyleChoice.from_dict(outcome["payload"]["style_selection"]))
                for outcome in outcomes
            ],
            style_catalog,
            style_vote_facts(knowledge_root, style_catalog),
        )
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="research_chunk_plan",
            task_id=task_id,
            payload={
                "cap_tokens": transcript_cap,
                "chunks": chunk_records,
            },
        )
    return {
        "context_pack": merged_pack.to_dict(),
        "keep_entries": list(dict.fromkeys(keep_union))[:keep_cap],
        "rounds": dict(rounds_meta),
        # `--style auto` on chunked material: the tallied votes, persisted in
        # the same place a single preparation task puts its pick.
        **({"style_selection": style_choice.to_dict()} if style_choice is not None else {}),
        "research_chunks": chunk_records,
        "chunking": {"research_transcript_max_tokens": transcript_cap},
        "injected_entries": sorted(set(injected_union)),
        "token_report": {
            "phase": "research",
            "rows": [],
            "totals": sum_token_distributions(token_totals),
        },
    }


def run_research_stage(
    *,
    stable_json: str | Path,
    context_path: str | Path,
    audio_path: str | Path | None = None,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    knowledge_enabled: bool = True,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    token_counter: TokenCounter | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    collect_task_feedback: bool = False,
    resume: bool = True,
    max_window_subtitle_tokens: int | None = None,
    # Extra inputs of the preparation task (`--style auto`'s catalog and the
    # user's own style requirements).
    runner_kwargs: Mapping[str, Any] | None = None,
) -> ContextPack:
    """Plan the correction windows, run the preparation task, persist the
    research context JSON next to the output, and return the ContextPack.
    ``correction_translation.run_correction_task`` is the caller."""

    token_counter = token_counter or default_token_counter()
    stage_client = RoleClient(test_profile=test_profile)
    segments = load_segments_from_stable_json(stable_json)
    plan_report: dict = {}
    # Research uses the current correction planner for sensible transcript
    # chunks, but correctness no longer requires later correction geometry to
    # match: the resulting notes are bound to source-id intervals below.
    from .routing.capabilities import correction_planning_limits

    planning_limits = correction_planning_limits(profile)
    audio_duration = probe_audio_duration(audio_path) if audio_path else None
    windows = plan_correction_windows(
        segments,
        context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
        counter=token_counter,
        audio_duration=audio_duration,
        profile=profile,
        report_sink=plan_report,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        limits=planning_limits,
    )
    if task_artifact_dir and plan_report.get("replan_attempts"):
        append_task_artifact(
            task_artifact_dir,
            kind="window_plan_report",
            task_id=task_id,
            payload={"phase": "research", **plan_report},
        )
    context_file = Path(context_path)
    # Ultra-long material splits into transcript chunks of whole windows, each
    # running the whole preparation task (plan W7 / P8).
    transcript_cap = research_transcript_cap(profile)
    chunk_plan = plan_research_chunks(
        windows, count_tokens=token_counter.count_text, max_tokens=transcript_cap
    )
    # The preparation task (docs/llm_harness_research.md, 「准备任务」) runs
    # as one logical conversation; local import: stages imports this module.
    from .stages.prepare_task import run_prepare_task

    research_runner = run_prepare_task
    research_runner_kwargs: Dict[str, Any] = dict(runner_kwargs or {})
    if len(chunk_plan) <= 1:
        research_payload = research_runner(
            **research_runner_kwargs,
            transcript=render_research_transcript(segments, windows),
            extra_info=extra_info,
            knowledge_root=knowledge_root,
            knowledge_enabled=knowledge_enabled,
            profile=profile,
            test_profile=test_profile,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            max_search_queries=research_search_query_limit(len(segments)),
            token_counter=token_counter,
            collect_task_feedback=collect_task_feedback,
            resume=resume,
            client=stage_client,
        )
    else:
        research_payload = _run_chunked_research(
            chunk_plan,
            segments=segments,
            context_file=context_file,
            extra_info=extra_info,
            knowledge_root=knowledge_root,
            knowledge_enabled=knowledge_enabled,
            profile=profile,
            test_profile=test_profile,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            token_counter=token_counter,
            collect_task_feedback=collect_task_feedback,
            resume=resume,
            client=stage_client,
            transcript_cap=transcript_cap,
            research_runner=research_runner,
            research_runner_kwargs=research_runner_kwargs,
        )
    # The model names the research windows, while the harness owns their exact
    # source-id coverage. Persist both so later correction geometry changes do
    # not make a surviving chunk id point at the wrong note.
    bind_report: dict[str, Any] = {}
    bound_context = ContextPack.from_dict(
        research_payload["context_pack"]
    ).bind_window_ranges(windows, report_sink=bind_report)
    research_payload["context_pack"] = bound_context.to_dict()
    if bind_report.get("unplaceable_window_ids"):
        # Not fatal -- general context still covers every window -- but it
        # must not be silent: a pack that lost all of its per-window notes is
        # indistinguishable from one that never had any.
        research_payload["window_context_bind_report"] = bind_report
        current_reporter().warning(
            "research-notes-unplaceable",
            "the preparation task named "
            f"{len(bind_report['unplaceable_window_ids'])} window(s) that are "
            f"not in the plan ({', '.join(bind_report['unplaceable_window_ids'])})",
            impact=f"这些笔记被丢弃，剩余 {bind_report['bound']} 条",
        )
    research_payload["planning"] = planning_metadata(
        profile,
        stable_json=stable_json,
        extra_info=extra_info,
        knowledge_root=knowledge_root,
        knowledge_enabled=knowledge_enabled,
        collect_task_feedback=collect_task_feedback,
        audio_duration=audio_duration,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        limits=planning_limits,
        test_profile=test_profile,
        execution_identity_override=getattr(
            stage_client, "execution_identity", None
        ),
    )
    # Atomic: a torn file is backed up and the whole research run again, paid again.
    write_atomic(context_file, json.dumps(research_payload, ensure_ascii=False, indent=2))
    return bound_context
