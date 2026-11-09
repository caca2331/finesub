"""The global preparation task as one logical conversation (plan §2.1, stage B).

Batched execution strategy: the first message carries the whole task; the
reply is either the deliverable (``<context_pack>``) or one batch of tool
requests. The harness executes the batch -- knowledge entries, one round of
local search -- and appends the results as the next user turn of the *same*
conversation (the model's own request turn is replayed as its assistant
message, so a REST adapter with no server-side session still sends one
legal history). The next reply must deliver: the request channel is closed,
and a reply that asks again is a parse failure retried under the usual
budget, never a second fill (plan §2.2: bounded, no dependent guessing).

``run_research_stage`` persists the payload as ``research-context.json``;
the correction windows and the knowledge update consume it from there.

The request parser, the entry caps and the delivery parser live in
``research`` and predate the task: the blocks did not change with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from ..client import GeminiPromptBlockedError, RoleClient, sum_token_distributions
from ..content_filter import (
    load_content_filter_blacklist,
    report_ladder_outcome,
    run_injection_ladder,
    split_rendered_search_block,
)
from ..exchange_log import ExchangeLogger
from ..injection_budget import render_knowledge_entries_block
from ..knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    append_task_artifact,
    index_texts_present,
    load_entry_texts,
    load_index_texts,
)
from ..output_tags import (
    extract_single_tag_block,
    find_top_level_tag_blocks,
    parse_guided_line_items,
    parse_line_items,
)
from ..knowledge.feedback import retrieval_urls_from_response
from ..knowledge.style import StyleCatalog, StyleChoice
from ..prepare_prompts import (
    STYLE_SELECTION_TAG,
    build_prepare_results_message,
    build_prepare_task_messages,
)
from ..prompts import ContextPack, r1_request_cap
from ..research import (
    ResearchRound1Result,
    _call_and_parse,
    _dump_research_input,
    _render_note_url_extracts,
    check_research_input_limit,
    count_indexed_entries,
    extract_round_task_feedback,
    parse_round1_output,
    parse_round2_output,
    render_preinjected_entries,
    resolve_round1_entries,
)
from ..routing.config import (
    DEFAULT_RESEARCH_SEARCH_QUERIES,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    injection_block_token_limit,
)
from ..routing.profiles import DEFAULT_PROFILE, TranslationProfile
from ..session_checkpoint import SessionCheckpointStore
from .task_graph import resolve_execution_strategy, window_requests
from ..token_budget import TokenCounter, default_token_counter
from ..web_search import (
    SearchRequest,
    WebSearchClient,
    extract_results_metadata,
    render_search_results,
    search_results_metadata,
)

#: Artifact kinds of the two turns. ``_call_and_parse`` derives the exchange
#: session name from the prefix it is given, so these only key the records.
REQUEST_TURN_KIND = "prepare_task_request"
DELIVERY_TURN_KIND = "prepare_task_response"


@dataclass(frozen=True)
class PrepareReply:
    """One parsed reply of the task: a delivery, or a request batch."""

    raw: str
    context_pack: ContextPack | None = None
    requests: ResearchRound1Result | None = None
    task_update_feedback: str = ""
    keep_entries_raw: tuple[str, ...] = ()
    #: `--style auto`: the delivery's pick, already held to the forced-pick
    #: contract by the parser (None when the task was not asked to pick).
    style_choice: StyleChoice | None = None

    @property
    def delivered(self) -> bool:
        return self.context_pack is not None


def parse_prepare_reply(
    text: str,
    *,
    allow_requests: bool,
    expect_queries: bool,
    expect_entries: bool,
    collect_task_feedback: bool = False,
    count_tokens: Callable[[str], int] | None = None,
    style_catalog: StyleCatalog | None = None,
) -> PrepareReply:
    """A delivery wins whenever ``<context_pack>`` is present (top-level);
    otherwise the reply must be a well-formed request batch, and only while
    the channel is open. Anything else is a parse error for the retry loop.

    ``style_catalog`` (`--style auto`): the delivery must also carry a valid
    pick (auto-style-gaps plan, 「必须选，没有回退」). Judged here, inside the
    parse, so a missed or invalid pick takes the same parse retries as any
    other malformed delivery -- and a checkpoint replaying an old delivery
    that never picked is invalid, not reused."""

    if find_top_level_tag_blocks(text, "context_pack"):
        style_choice = (
            style_catalog.choose(find_top_level_tag_blocks(text, STYLE_SELECTION_TAG))
            if style_catalog is not None
            else None
        )
        return PrepareReply(
            raw=text,
            context_pack=parse_round2_output(text),
            task_update_feedback=(
                extract_round_task_feedback(text, count_tokens=count_tokens)
                if collect_task_feedback
                else ""
            ),
            keep_entries_raw=tuple(
                parse_line_items(
                    extract_single_tag_block(text, "keep_entries", required=False)
                )
            ),
            style_choice=style_choice,
        )
    if not allow_requests:
        raise ValueError("preparation delivery is missing its <context_pack> block")
    requests = parse_round1_output(
        text,
        expect_queries=expect_queries,
        expect_entries=expect_entries,
        count_tokens=count_tokens,
    )
    if not (requests.search_queries or requests.requested_entries or requests.keep_entries):
        raise ValueError(
            "preparation reply neither delivered <context_pack> nor requested anything"
        )
    return PrepareReply(raw=text, requests=requests)


def _fill_requests(
    requests: ResearchRound1Result,
    *,
    knowledge_root: str | Path,
    entries_available: bool,
    local_search: bool,
    visible_keep_keys: List[str],
    request_cap: int,
    max_search_queries: int,
    search_client: WebSearchClient | None,
    execution_settings: Any,
    token_counter: TokenCounter,
) -> Dict[str, Any]:
    """Execute one request batch: entries off the base, one search round.

    ``entries_available`` is the base gate: with no readable index the prompt
    described no entry blocks at all, so a request for one is unsolicited --
    and serving it would read entry bodies out of a base this run was told not
    to touch (plan §8: "KB 关闭不读普通索引／正文").
    """

    entry_details: Dict[str, str] = {}
    missing: List[str] = []
    ignored_keep: List[str] = []
    dropped: List[str] = []
    if entries_available:
        entry_details, missing, ignored_keep, dropped = resolve_round1_entries(
            knowledge_root,
            requested_names=requests.requested_entries,
            keep_names=requests.keep_entries,
            visible_keep_keys=visible_keep_keys,
            max_requested_entries=request_cap,
            max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
            max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
        )
    entry_text, entry_report = "", {}
    if entry_details:
        block = render_knowledge_entries_block(
            entry_details,
            count_tokens=token_counter.count_text,
            block_limit=injection_block_token_limit(KB_WINDOW_TOTAL_ENTRIES),
        )
        entry_text, entry_report = block.text, block.report()
    search_results: list = []
    search_text, search_report = "", {}
    if local_search and requests.search_queries:
        search_client = search_client or WebSearchClient(execution_settings=execution_settings)
        search_results = search_client.search_many(
            [
                SearchRequest(query=query, guided_query=guided)
                for query, guided in parse_guided_line_items(
                    "\n".join(requests.search_queries)
                )
            ],
            max_queries=max_search_queries,
        )
        block = render_search_results(
            search_results,
            max_total_tokens=injection_block_token_limit(max_search_queries),
            count_tokens=token_counter.count_text,
        )
        search_text, search_report = block.text, block.report()
    return {
        "entry_details": entry_details,
        "entry_text": entry_text,
        "entry_report": entry_report,
        "missing_entries": missing,
        "ignored_keep_entries": ignored_keep,
        "dropped_entry_keys": dropped,
        "search_results": search_results,
        "search_text": search_text,
        "search_report": search_report,
    }


def _opening_inputs(
    extra_info: str,
    *,
    local_search: bool,
    knowledge_enabled: bool,
    knowledge_root: str | Path,
    search_client: WebSearchClient | None,
    execution_settings: Any,
    token_counter: TokenCounter,
    task_artifact_dir: str | Path | None,
    task_id: str,
) -> tuple[str, str, Dict[str, Any], WebSearchClient | None]:
    """What the opening message shows before the model has said anything.

    Both halves are keyed off ``--extra-info``: the pages its URLs point at
    (fetched by the harness, only where it may retrieve) and the knowledge
    entries its keywords match. Returns them plus the search client, which is
    built lazily here and reused by the fill.
    """

    note_url_extracts = ""
    if extra_info.strip() and local_search:
        search_client = search_client or WebSearchClient(execution_settings=execution_settings)
        note_url_extracts, note_urls, note_results = _render_note_url_extracts(
            extra_info, search_client, count_tokens=token_counter.count_text
        )
        if task_artifact_dir and note_urls:
            append_task_artifact(
                task_artifact_dir,
                kind="api_call",
                task_id=task_id,
                payload={
                    "category": "web_extract",
                    "source": "extra_info_urls",
                    "urls": note_urls,
                    "executed": extract_results_metadata(note_results),
                    "rendered_tokens": token_counter.count_text(note_url_extracts),
                },
            )
    preinjected_text, preinjection_report = "", {"matches": []}
    if extra_info.strip() and knowledge_enabled:
        preinjected_text, preinjection_report = render_preinjected_entries(
            knowledge_root, extra_info, count_tokens=token_counter.count_text
        )
        if task_artifact_dir and (
            preinjection_report["matches"] or preinjection_report.get("term_matches")
        ):
            append_task_artifact(
                task_artifact_dir,
                kind="knowledge_preinjection",
                task_id=task_id,
                payload={"source": "prepare_task", **preinjection_report},
            )
    return note_url_extracts, preinjected_text, preinjection_report, search_client


def _record_delivery(
    reply: PrepareReply,
    *,
    token_rows: List[Dict[str, Any]],
    delivery_raw: List[Any],
    collect_task_feedback: bool,
    task_artifact_dir: str | Path | None,
    task_id: str,
) -> Dict[str, Any]:
    """What the task leaves behind once it has delivered; returns the token report."""

    if collect_task_feedback and reply.task_update_feedback and task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="research_task_feedback",
            task_id=task_id,
            payload={
                "source": "prepare_task",
                "feedback": reply.task_update_feedback,
                # Harness-side call evidence: what the answering model actually
                # retrieved. The knowledge update reads it as the sources behind
                # this feedback, so hardcoding [] silently dropped every agent
                # search URL (the window path never stopped filling it).
                "retrieval_urls": list(
                    retrieval_urls_from_response(delivery_raw[-1] if delivery_raw else None)
                ),
            },
        )
    token_report = {
        "phase": "research",
        "rows": token_rows,
        "totals": sum_token_distributions(row["tokens"] for row in token_rows),
    }
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind="token_distribution_report",
            task_id=task_id,
            payload=token_report,
        )
    return token_report


def _prepare_kb_grant(
    *,
    entries_available: bool,
    agentic: bool,
    style_catalog: StyleCatalog | None,
    knowledge_root: str | Path,
) -> tuple[str, Dict[str, str] | None]:
    """What the executor may read (§4.3 matrix): the base where it is readable;
    otherwise an agentic task with `--style auto` candidates gets the narrow
    style grant its prompt already offers (`kb_read` of `style/<名>`) -- the
    same grant a fast window gets (kb-followups batch 2, item 6), naming its
    own root and revision because there is no generation pin to borrow."""

    if entries_available:
        return "read", None
    if agentic and style_catalog is not None and style_catalog.rev is not None:
        return "style", {
            "knowledge_root": str(knowledge_root),
            "knowledge_identity": f"rev:{style_catalog.rev}",
        }
    return "", None


def run_prepare_task(
    *,
    transcript: str,
    extra_info: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    client: RoleClient | None = None,
    search_client: WebSearchClient | None = None,
    profile: TranslationProfile = DEFAULT_PROFILE,
    knowledge_enabled: bool = True,
    test_profile: bool = False,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    max_parse_retries: int = 5,
    max_search_queries: int = DEFAULT_RESEARCH_SEARCH_QUERIES,
    token_counter: TokenCounter | None = None,
    collect_task_feedback: bool = False,
    resume: bool = True,
    chunk_label: str = "",
    exchange_logger: ExchangeLogger | None = None,
    checkpoint_store: SessionCheckpointStore | None = None,
    # `--style auto` (stage E): the candidates the delivery picks from, and
    # the user's own requirements the pick must respect. None = no selection.
    style_catalog: StyleCatalog | None = None,
    extra_style: str = "",
    # `--speaker-judge on`: the delivery also lists who speaks (the judge's cast candidates).
    speakers: bool = False,
    # A long file's later chunk: what the chunks before settled (`general_context.carry_text`).
    earlier_chunks: str = "",
) -> Dict[str, Any]:
    """Run the preparation task; return the research-context payload."""

    client = client or RoleClient(test_profile=test_profile)
    token_counter = token_counter or default_token_counter()
    if style_catalog is not None and not style_catalog.has_choice:
        # Nothing to choose (no styles, or only `default_style`): the task is
        # not asked to pick, and the orchestrator records the no-choice result
        # itself (auto-style-gaps plan, C 「先读记录」 step 2).
        style_catalog = None
    checkpoint_key = chunk_label or "main"
    dump_suffix = f"-{chunk_label}" if chunk_label else ""
    if exchange_logger is None:
        exchange_logger = ExchangeLogger.for_task_artifact_dir(task_artifact_dir)
    if checkpoint_store is None:
        checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)
    knowledge_indexes = load_index_texts(knowledge_root) if knowledge_enabled else {}
    entries_available = index_texts_present(knowledge_indexes)
    local_search = profile.external_injection
    # Stage C: an agentic executor (a tool session on the harness server)
    # fetches for itself -- the request channel closes and one reply delivers.
    strategy, _source = resolve_execution_strategy(
        profile, task_group="research", test_profile=test_profile, knowledge_readable=entries_available
    )
    agentic = strategy == "agentic" and (local_search or entries_available)
    can_request = (local_search or entries_available) and not agentic
    catalog_text = style_catalog.render() if style_catalog else ""
    kb_tools, kb_binding = _prepare_kb_grant(
        entries_available=entries_available,
        agentic=agentic,
        style_catalog=style_catalog,
        knowledge_root=knowledge_root,
    )
    # Whether a *window* can ask for a dropped entry back -- the windows'
    # answer (`task_graph.window_flow`), not this task's own grant above.
    downstream_can_request = window_requests(profile, knowledge_readable=entries_available)
    # Who names what survives into the windows: a pruning keep only where a
    # window can ask back; otherwise the harness transfers the whole selection
    # (docs/llm_harness_behavior.md). Agentic injected nothing, on any vector,
    # so its own keep is the only way anything reaches the windows.
    # ⚠ `and entries_available`: with the base off (or empty) there is nothing
    # to keep, and honouring an unsolicited `<keep_entries>` would read entry
    # bodies out of a base this run was told not to touch (plan §8: "KB 关闭不读
    # 普通索引／正文"). The prompt never asks for the block there either.
    emits_keep = (downstream_can_request or agentic) and entries_available
    request_cap = r1_request_cap(downstream_can_request=downstream_can_request)
    token_rows: List[Dict[str, Any]] = []
    # The answering attempt's raw response, kept for the retrieval-ledger URL
    # backfill below (`_call_and_parse` returns only the parsed value).
    delivery_raw: List[Any] = []
    execution_settings = getattr(client, "execution_settings", None)

    note_url_extracts, preinjected_text, preinjection_report, search_client = _opening_inputs(
        extra_info,
        local_search=local_search,
        knowledge_enabled=knowledge_enabled,
        knowledge_root=knowledge_root,
        search_client=search_client,
        execution_settings=execution_settings,
        token_counter=token_counter,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
    )
    blacklist = load_content_filter_blacklist(task_artifact_dir)

    def _parser(allow_requests: bool):
        return lambda text: parse_prepare_reply(
            text,
            allow_requests=allow_requests,
            expect_queries=local_search,
            expect_entries=entries_available,
            collect_task_feedback=collect_task_feedback,
            count_tokens=token_counter.count_text,
            style_catalog=style_catalog,
        )

    def _call(messages, *, turn: str, allow_requests: bool, kind: str):
        check_research_input_limit(messages, round_name=f"prepare {turn}", counter=token_counter)
        return _call_and_parse(
            client,
            messages,
            output_scale=profile.output_scale,
            parser=_parser(allow_requests),
            round_name=f"prepare {turn}",
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            artifact_kind=kind,
            max_parse_retries=max_parse_retries,
            token_rows=token_rows,
            exchange_logger=exchange_logger,
            token_counter=token_counter,
            input_components_kwargs={"transcript": transcript},
            session_prefix=f"prepare-{turn}{dump_suffix}",
            checkpoint_store=checkpoint_store,
            checkpoint_session=f"prepare-{turn}",
            checkpoint_key=checkpoint_key,
            raw_response_sink=delivery_raw,
            retrieval=profile.retrieval,
            # The same cell the strategy was derived from (`capabilities.chain_serves_tool_session` reads
            # `role_config_for("research", profile.difficulty)`), so an agentic prompt reaches the agent that
            # was asked to fetch for itself; the cell also picks the prompt variant and thinking level.
            difficulty=profile.difficulty,
            kb_tools=kb_tools,
            kb_binding=kb_binding,
            session_key=f"prepare-{task_id}{dump_suffix}",  # t2 continues t1
            followup_expected=allow_requests,
        )

    def _first_turn(extracts_text: str) -> tuple[list, PrepareReply]:
        messages = build_prepare_task_messages(
            transcript=transcript,
            extra_info=extra_info,
            note_url_extracts=extracts_text,
            knowledge_indexes=knowledge_indexes,
            preinjected_entries=preinjected_text,
            max_search_queries=max_search_queries,
            emits_queries=local_search,
            emits_entries=entries_available,
            native_search=profile.native_search,
            correct_only=profile.correct_only,
            max_requested_entries=request_cap,
            collect_task_feedback=collect_task_feedback,
            emits_keep=emits_keep,
            agentic=agentic,
            style_catalog=catalog_text,
            extra_style=extra_style,
            speakers=speakers,
            earlier_chunks=earlier_chunks,
        )
        if task_artifact_dir:
            _dump_research_input(
                task_artifact_dir,
                f"prepare{dump_suffix}",
                {
                    "transcript": transcript,
                    "extra_info": extra_info,
                    "note_url_extracts": extracts_text,
                    "knowledge_indexes": knowledge_indexes,
                    "preinjected_entries": preinjected_text,
                    "max_search_queries": max_search_queries,
                    "emits_queries": local_search,
                    "emits_entries": entries_available,
                    "native_search": profile.native_search,
                    "max_requested_entries": request_cap,
                    "collect_task_feedback": collect_task_feedback,
                    "emits_keep": emits_keep,
                    "agentic": agentic,
                    "style_catalog": catalog_text,
                    "extra_style": extra_style,
                    "earlier_chunks": earlier_chunks,
                },
            )
        reply = _call(messages, turn="t1", allow_requests=can_request, kind=REQUEST_TURN_KIND)
        return messages, reply

    first_block = split_rendered_search_block(note_url_extracts)
    first_outcome = run_injection_ladder(
        block=first_block,
        call=_first_turn,
        stage="prepare_task",
        blocked_exception=GeminiPromptBlockedError,
        blacklist=blacklist,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
        plain_retry=not first_block.units,
    )
    report_ladder_outcome(first_outcome, what="preparation task")
    messages, reply = first_outcome.result

    requests = reply.requests or ResearchRound1Result()
    fill: Dict[str, Any] = {
        "entry_details": {},
        "entry_text": "",
        "entry_report": {},
        "missing_entries": [],
        "ignored_keep_entries": [],
        "dropped_entry_keys": [],
        "search_results": [],
        "search_text": "",
        "search_report": {},
    }
    if not reply.delivered:
        fill = _fill_requests(
            requests,
            knowledge_root=knowledge_root,
            entries_available=entries_available,
            local_search=local_search,
            visible_keep_keys=[
                *preinjection_report.get("included", []),
                *preinjection_report.get("truncated", []),
            ],
            request_cap=request_cap,
            max_search_queries=max_search_queries,
            search_client=search_client,
            execution_settings=execution_settings,
            token_counter=token_counter,
        )
        if task_artifact_dir and local_search:
            append_task_artifact(
                task_artifact_dir,
                kind="research_search_results",
                task_id=task_id,
                payload={
                    "queries": list(requests.search_queries),
                    "multi_round": False,
                    "batched": True,
                    "executed": search_results_metadata(fill["search_results"]),
                    "rendered_tokens": token_counter.count_text(fill["search_text"]),
                    "render_report": fill["search_report"],
                },
            )

        def _second_turn(search_text: str) -> PrepareReply:
            history = [
                *messages,
                {"role": "assistant", "content": reply.raw},
                build_prepare_results_message(
                    entry_details_text=fill["entry_text"],
                    search_results=search_text,
                    requested_entries=bool(requests.requested_entries or requests.keep_entries),
                    requested_search=bool(local_search and requests.search_queries),
                    collect_task_feedback=collect_task_feedback,
                    emits_keep=emits_keep,
                    style_selection=bool(catalog_text),
                    knowledge=entries_available,
                ),
            ]
            return _call(history, turn="t2", allow_requests=False, kind=DELIVERY_TURN_KIND)

        search_block = split_rendered_search_block(fill["search_text"])
        second_outcome = run_injection_ladder(
            block=search_block,
            call=_second_turn,
            stage="prepare_task_delivery",
            blocked_exception=GeminiPromptBlockedError,
            blacklist=blacklist,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id,
            plain_retry=not search_block.units,
        )
        report_ladder_outcome(second_outcome, what="preparation delivery")
        reply = second_outcome.result

    token_report = _record_delivery(
        reply,
        token_rows=token_rows,
        delivery_raw=delivery_raw,
        collect_task_feedback=collect_task_feedback,
        task_artifact_dir=task_artifact_dir,
        task_id=task_id,
    )
    return _payload(
        reply,
        requests=requests,
        fill=fill,
        visible_keys=[
            *preinjection_report.get("included", []),
            *preinjection_report.get("truncated", []),
        ],
        agentic=agentic,
        style_choice=reply.style_choice,
        tool_round=not first_outcome.result[1].delivered,
        local_search=local_search,
        native_search=profile.native_search,
        entries_available=entries_available,
        emits_keep=emits_keep,
        request_cap=request_cap,
        knowledge_root=knowledge_root,
        knowledge_enabled=knowledge_enabled,
        token_report=token_report,
    )


def _payload(
    reply: PrepareReply,
    *,
    requests: ResearchRound1Result,
    fill: Mapping[str, Any],
    #: Preinjected entries the opening message showed (fully or truncated): a
    #: delivery may keep one of them even when no fill ever ran.
    visible_keys: Sequence[str],
    agentic: bool,
    style_choice: StyleChoice | None,
    tool_round: bool,
    local_search: bool,
    native_search: bool,
    entries_available: bool,
    emits_keep: bool,
    request_cap: int,
    knowledge_root: str | Path,
    knowledge_enabled: bool,
    token_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """The payload persisted as ``research-context.json``.

    ``rounds.round1`` / ``rounds.round2`` say "did the ask half / the digest
    half happen"; ``rounds.task_flow`` and ``rounds.tool_round`` say what
    actually ran.
    """

    entry_details = fill["entry_details"]
    if emits_keep:
        keep_found, _missing = load_entry_texts(knowledge_root, list(reply.keep_entries_raw))
        # A task may only keep what it could actually read. Batched: whatever
        # the harness put in front of it -- the fill's entries plus the
        # preinjections the opening showed. Agentic: the executor reads the base
        # through its own ``kb_read``, so the whole base was reachable and the
        # store lookup above (which already drops names that do not resolve) is
        # the only filter there is. Filtering agentic keeps against a fill that
        # never ran dropped every one of them.
        visible = set(entry_details) | set(visible_keys)
        keep_entries = [
            key for key in keep_found if agentic or key in visible
        ][:KB_TRANSFER_MAX_ENTRIES]
    else:
        keep_entries = sorted(entry_details)[:KB_WINDOW_TOTAL_ENTRIES]
    context_pack = reply.context_pack or ContextPack()
    return {
        "context_pack": context_pack.to_dict(),
        "keep_entries": keep_entries,
        "rounds": {
            "task_flow": "agentic" if agentic else "batched",
            "round1": tool_round,
            "round2": True,
            "tool_round": tool_round,
            "local_search": local_search,
            "native_search": native_search,
            "entries_available": entries_available,
        },
        "task_update_feedback": reply.task_update_feedback,
        "round1": {
            "requested_entries": list(requests.requested_entries),
            "keep_entries": list(requests.keep_entries),
            "search_queries": list(requests.search_queries),
            "analysis_notes": requests.analysis_notes,
        },
        "search_results": search_results_metadata(fill["search_results"]),
        "injected_entries": sorted(entry_details),
        "missing_entries": fill["missing_entries"],
        "ignored_keep_entries": fill["ignored_keep_entries"],
        "dropped_entry_keys": fill["dropped_entry_keys"],
        "entry_selection": {
            "request_cap": request_cap,
            "requested_count": len(requests.requested_entries),
            "hit_cap": len(requests.requested_entries) >= request_cap,
            "dropped_by_cap": len(fill["dropped_entry_keys"]),
            "indexed_entry_count": count_indexed_entries(knowledge_root) if knowledge_enabled else 0,
        },
        "entry_render_report": fill["entry_report"],
        "token_report": token_report,
        # `--style auto`: the task-level choice, persisted with the context so
        # a resume and every window read the same pick (auto-style plan §5).
        **({"style_selection": style_choice.to_dict()} if style_choice is not None else {}),
    }
