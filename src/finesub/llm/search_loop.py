"""Multi-round local search loop (Research Contract / Progress / Evidence Pack).

The main conversation (background research round 1) emits a Research Contract
plus round-0 search queries. The harness executes each search round locally and
then calls a lightweight text model that filters/dedups results, extracts
evidence into an incremental Research Progress ledger, and decides whether the
contract is satisfied. When it is — or when the round cap is reached — the
model emits an Evidence Pack that replaces raw search results in the main
conversation's next step.

Fact priorities (1-5) are advisory: the harness decrements the priority of
every fact targeted by an executed follow-up query (floor 0), but a fact at 0
may still be queried — the only hard anti-rathole boundary is ``max_rounds``.
"""

from __future__ import annotations

from finesub.reporting import current_reporter

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re as _re
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .client import (
    GeminiPromptBlockedError,
    LLMCallResult,
    RoleClient,
    extract_token_distribution,
    is_prompt_blocked,
    validation_retry_sampling_kwargs,
)
from .routing.config import (
    INJECTION_SECTION_MAX_TOKENS,
    LLMRole,
    SEARCH_LOOP_MAX_TOKENS,
    EVIDENCE_PACK_MAX_TOKENS,
    PROGRESS_UPDATE_MAX_TOKENS,
    injection_block_token_limit,
)
from .content_filter import (
    run_injection_ladder,
    split_rendered_search_block,
)
from .injection_budget import render_budgeted_block, render_knowledge_entries_block
from .knowledge.base import (
    append_task_artifact,
    load_entry_texts,
    load_index_text,
)
from .exchange_log import ExchangeLogger
from .exchange_metadata import (
    infer_session_name,
    llm_exchange_metadata,
    search_loop_input_components,
)
from .output_tags import (
    GUIDED_QUERY_SEPARATOR,
    extract_single_tag_block,
    parse_guided_line_items,
    parse_json_object,
    parse_line_items,
)
from .prompts import PROMPT_VERSION, build_search_loop_messages, build_search_loop_v2_messages
from .session_checkpoint import SessionCheckpointStore, session_input_hash
from .token_budget import default_token_counter
from .token_truncate import cap_tokens
from .web_search import (
    ExtractRequest,
    QueryExtractResult,
    QuerySearchResult,
    SearchRequest,
    WebSearchClient,
    extract_result_sections,
    extract_urls_from_text,
    guided_query_key,
    normalize_url_key,
    request_label,
    search_result_sections,
    search_results_metadata,
    extract_results_metadata,
)




EVIDENCE_PACK_HEADER = (
    "（本 Evidence Pack 由本地多轮搜索代理围绕 Research Contract 检索整理；"
    "引用前仍需交叉验证。）"
)
DEGRADED_PACK_NOTICE = (
    "（多轮搜索代理未能生成 Evidence Pack，以下为 Research Progress 台账与"
    "全部原始搜索结果。）"
)

# Per text blob, when harvesting the fetchable-URL corpus. Unrelated to
# EXTRA_INFO_URL_EXTRACT_LIMIT, which caps how many of the *user's* URLs are
# deep-extracted up front; here we only build a lookup table, so the limit
# exists to bound work on a pathological page, not to ration anything.
SEEN_URL_HARVEST_LIMIT = 200

# Shown back to the judge in the request snapshot for a URL it asked for that
# never appeared in anything it was given.
EXTRACT_URL_REJECTED_NOTE = "未执行：该 URL 未在此前的输入中出现过，只能提取已出现的 URL"




@dataclass
class SearchLoopResult:
    evidence_pack: str = ""
    degraded: bool = False
    search_rounds_executed: int = 0
    progress_log: str = ""
    contract: Dict[str, Any] = field(default_factory=dict)
    executed_queries: List[str] = field(default_factory=list)
    executed_extract_urls: List[str] = field(default_factory=list)
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    # Budget-rendered raw search+extract text (no Evidence Pack wrapper).
    # Research round 2 rebuilds from these units when the pack itself trips
    # the content filter — the contaminated pack must never be re-injected.
    source_results_text: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        """Compact payload for research-context.json / task artifacts."""

        return {
            "degraded": self.degraded,
            "search_rounds_executed": self.search_rounds_executed,
            "executed_queries": list(self.executed_queries),
            "executed_extract_urls": list(self.executed_extract_urls),
            "contract": dict(self.contract),
            "progress_log": self.progress_log,
            "rounds": list(self.rounds),
            "evidence_pack_chars": len(self.evidence_pack),
        }


def parse_contract_json(body: str) -> Dict[str, Any]:
    """Tolerant Research Contract parse; any failure yields an empty contract."""

    if not (body or "").strip():
        return {}
    try:
        return parse_json_object(body)
    except (ValueError, json.JSONDecodeError):
        return {}


def _contract_fact_index(contract: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    facts = contract.get("facts")
    if not isinstance(facts, list):
        return index
    for fact in facts:
        if isinstance(fact, dict) and str(fact.get("id", "")).strip():
            index[str(fact["id"]).strip().upper()] = fact
    return index


def _split_guided(text: str) -> tuple[str, str]:
    """Split an optional ` >> guided query` suffix off a query/URL line."""

    if GUIDED_QUERY_SEPARATOR in text:
        head, guided = text.split(GUIDED_QUERY_SEPARATOR, 1)
        return head.strip(), guided.strip()
    return text.strip(), ""


def _split_fact_tag(query: str, fact_ids: Sequence[str]) -> tuple[str, str]:
    """Split an optional ``F1|query`` prefix; unknown prefixes stay in the query."""

    if "|" in query:
        head, rest = query.split("|", 1)
        head_norm = head.strip().upper()
        if head_norm in fact_ids and rest.strip():
            return head_norm, rest.strip()
    return "", query.strip()


def _decrement_priority(fact: Dict[str, Any]) -> None:
    try:
        current = int(fact.get("priority", 0))
    except (TypeError, ValueError):
        current = 0
    fact["priority"] = max(0, current - 1)


_PROGRESS_FACT_RE = _re.compile(
    r"^(F\d+)\s*:\s*(confirmed|partial|not_found|dead_end)", _re.MULTILINE
)


def _premature_pack_warnings(
    progress_text: str,
    fact_index: Mapping[str, Dict[str, Any]],
    *,
    min_priority: int = 2,
) -> List[Dict[str, Any]]:
    """Facts still unresolved (latest status partial/not_found) at priority >= gate.

    ``progress_text`` must be the ACCUMULATED ledger (all rounds joined), not a
    single round's delta: ``<progress_update>`` only carries the current round's
    increment, so a fact marked ``not_found`` in an early round and never
    restated would otherwise be missed. Later status lines for a fact override
    earlier ones (last-wins), so a fact resolved in a later round is not
    flagged. ``min_priority`` mirrors the continue-notice prompt (priority >= 2
    partial/not_found is "worth another round"); ``fact.priority`` here is the
    per-round-decremented value the model actually saw.

    Used to emit a warning artifact when the judge produces an Evidence Pack on
    a non-final round while such facts remain unresolved.
    """
    latest_status: Dict[str, str] = {}
    for match in _PROGRESS_FACT_RE.finditer(progress_text):
        latest_status[match.group(1).upper()] = match.group(2)
    warnings: List[Dict[str, Any]] = []
    for fact_id, status in latest_status.items():
        if status not in ("partial", "not_found"):
            continue
        fact = fact_index.get(fact_id)
        if fact is None:
            continue
        try:
            priority = int(fact.get("priority", 0))
        except (TypeError, ValueError):
            priority = 0
        if priority >= min_priority:
            warnings.append(
                {"fact_id": fact_id, "status": status, "priority": priority}
            )
    return warnings


_PACK_CONCLUSION_FACT_RE = _re.compile(
    r"^(?:\[unresolved\]\s*)?(F\d+)\s", _re.MULTILINE
)


def _check_fact_coverage(
    pack_text: str, fact_index: Mapping[str, Dict[str, Any]]
) -> List[str]:
    """Return contract fact IDs missing from the pack's ## 结论 section.

    Soft check: the result is logged as a warning artifact, never blocks.
    """
    # Isolate the ## 结论 section (up to the next ## heading or end).
    conclusion = ""
    in_conclusion = False
    for line in pack_text.splitlines():
        if line.strip().startswith("## 结论"):
            in_conclusion = True
            continue
        if in_conclusion and line.strip().startswith("## "):
            break
        if in_conclusion:
            conclusion += line + "\n"
    mentioned = {
        m.group(1).upper() for m in _PACK_CONCLUSION_FACT_RE.finditer(conclusion)
    }
    return sorted(set(fact_index) - mentioned)


def _render_contract(contract: Mapping[str, Any], raw_body: str) -> str:
    if contract:
        return json.dumps(dict(contract), ensure_ascii=False, indent=2)
    return (raw_body or "").strip()


def _extract_optional_block(
    text: str,
    tag: str,
    *,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    try:
        body = extract_single_tag_block(text, tag, required=False)
    except ValueError:
        return ""
    body = body.strip()
    if max_tokens is None:
        return body
    return cap_tokens(body, max_tokens, count_tokens)


def _dump_search_loop_round_input(
    task_artifact_dir: str | Path,
    *,
    round_index: int,
    max_rounds: int,
    is_final_round: bool,
    background: str,
    contract_json: str,
    executed_queries: List[str],
    progress_log: str,
    search_results: str,
    streamer_index: str,
    common_index: str,
    knowledge_entries: str,
    previous_requested_entries: List[str],
    previous_kept_entries: List[str],
    previous_contract_json: str,
    previous_search_queries: List[str],
    previous_extract_urls: List[str],
    followup_query_cap: int,
    previous_evidence_pack: str = "",
) -> None:
    """Persist one search-loop round's full input state for replay fixtures."""

    payload = {
        "round_index": round_index,
        "max_rounds": max_rounds,
        "is_final_round": is_final_round,
        "background": background,
        "contract_json": contract_json,
        "executed_queries": executed_queries,
        "progress_log": progress_log,
        "previous_evidence_pack": previous_evidence_pack,
        "search_results": search_results,
        "streamer_index": streamer_index,
        "common_index": common_index,
        "knowledge_entries": knowledge_entries,
        "previous_requested_entries": previous_requested_entries,
        "previous_kept_entries": previous_kept_entries,
        "previous_contract_json": previous_contract_json,
        "previous_search_queries": previous_search_queries,
        "previous_extract_urls": previous_extract_urls,
        "followup_query_cap": followup_query_cap,
    }
    path = Path(task_artifact_dir) / f"search-loop-round-{round_index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_search_loop(
    *,
    contract_body: str,
    round0_queries: Sequence[str],
    client: RoleClient,
    search_client: WebSearchClient,
    background: str = "",
    max_rounds: int = 3,
    round0_query_cap: int = 8,
    followup_query_cap: int = 4,
    max_parse_retries: int = 5,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    artifact_kind: str = "search_loop_round",
    exchange_prefix: str = "search-loop",
    token_rows: List[Dict[str, Any]] | None = None,
    exchange_logger: ExchangeLogger | None = None,
    token_counter: Any | None = None,
    knowledge_root: str | Path | None = None,
    persistent_entries_text: str = "",
    persistent_entry_keys: Sequence[str] = (),
    persistent_requested_entry_names: Sequence[str] = (),
    persistent_kept_entry_names: Sequence[str] = (),
    content_filter_blacklist: set[str] | None = None,
    loop_version: str = "v1",
    # The run's difficulty. It selects the search_judge cell, so it must come
    # from the caller rather than defaulting -- a preset binding a different
    # group at intermediate/efficiency was silently ignored otherwise.
    difficulty: str = "quality",
    resume: bool = True,
) -> SearchLoopResult:
    """Run the multi-round search loop and return the evidence pack.

    ``max_rounds`` is the total number of search rounds including round 0.
    One lightweight loop call follows each search round; the final call must
    emit the evidence pack. Never raises for model/format trouble: the result
    degrades to a progress-ledger + raw-results pack instead.

    With ``knowledge_root`` set, every non-final loop round sees both local
    knowledge indices and may emit ``<requested_entries>`` (one key/alias per
    line, cap = ``followup_query_cap``, cross-round deduped); the harness
    injects the budget-rendered bodies into the next round.

    ``persistent_entries_text``/``persistent_entry_keys``: entries selected by
    main research R1. They are visible in every loop round (read-only base,
    deduped against loop-side requests). ``persistent_requested_entry_names``
    and ``persistent_kept_entry_names`` preserve R1's two output channels for
    the round-0 judge prompt.
    """

    max_rounds = max(1, int(max_rounds))
    if token_counter is None:
        token_counter = default_token_counter(
            execution_settings=getattr(client, "execution_settings", None)
        )
    loop_count_tokens = token_counter.count_text
    contract = parse_contract_json(contract_body)
    fact_index = _contract_fact_index(contract)
    result = SearchLoopResult(contract=contract)
    # Dedup is "have I already spent budget on exactly this request", and a
    # request is (what to fetch, what to focus on): the guided query changes
    # what the provider highlights or extracts, so the same page with a new
    # focus is a new request, not a repeat. Keying on the URL/query alone --
    # as this did until 2026-08-13 -- silently swallowed the model's second,
    # differently-aimed look and told it nothing.
    executed_norm: set[tuple[str, str]] = set()
    executed_extract_norm: set[tuple[str, str]] = set()
    # URLs the model has actually been shown, keyed by `normalize_url_key`.
    # `<extract_urls>` may only select from this; see `_remember_seen_urls`.
    seen_urls: Dict[str, str] = {}
    all_results: List[QuerySearchResult] = []
    all_extract_results: List[QueryExtractResult] = []
    progress_sections: List[str] = []
    previous_pack = ""  # v2: latest evidence pack for cross-round continuity
    streamer_index = (
        load_index_text(knowledge_root, "streamer") if knowledge_root is not None else ""
    )
    common_index = (
        load_index_text(knowledge_root, "common") if knowledge_root is not None else ""
    )
    entry_channel_enabled = knowledge_root is not None and bool(
        streamer_index.strip() or common_index.strip()
    )
    injected_entry_keys: set[str] = set(persistent_entry_keys)
    knowledge_entries_text = ""
    previous_requested_entries = list(persistent_requested_entry_names)
    previous_kept_entries = list(persistent_kept_entry_names)
    blacklist = (
        content_filter_blacklist
        if content_filter_blacklist is not None
        else set()
    )
    checkpoint_store = SessionCheckpointStore(task_artifact_dir, enabled=resume)

    def _remember_seen_urls(*texts: str) -> None:
        """Record every URL the model has been shown, first spelling wins.

        The corpus is deliberately everything visible -- background, result
        links, the body of pages we extracted, and the knowledge-base indexes
        and entry bodies injected into every non-final round -- so the rule
        the prompt states ("only a URL from above") is the rule the harness
        enforces. Anything less makes the rejection notice a lie: the model
        did see that link, in an entry we handed it.

        Owner decision 2026-08-13: page bodies are included even though their
        links are attacker-plantable, i.e. this gate stops a *fabricated*
        destination, not a chosen one. Knowledge entries are the harness's own
        asset and so are strictly safer than what is already in.

        Model-authored text is never a source: a progress ledger or evidence
        pack must not be able to mint its own destination.
        """

        for text in texts:
            for url in extract_urls_from_text(text, limit=SEEN_URL_HARVEST_LIMIT):
                key = normalize_url_key(url)
                if key:
                    seen_urls.setdefault(key, url)

    _remember_seen_urls(
        background, streamer_index, common_index, persistent_entries_text
    )

    def _checkpoint_content_accepted(content: str, *, is_final: bool) -> bool:
        """Re-run the current operational parser before replaying a judge."""

        pack = _extract_optional_block(
            content,
            "evidence_pack",
            max_tokens=EVIDENCE_PACK_MAX_TOKENS,
            count_tokens=loop_count_tokens,
        )
        if pack:
            return True
        if loop_version != "v1":
            return False
        try:
            search_items = parse_guided_line_items(
                extract_single_tag_block(content, "search_queries", required=False)
            )
            extract_items = parse_guided_line_items(
                extract_single_tag_block(content, "extract_urls", required=False)
            )
            requested_entries = (
                parse_line_items(
                    extract_single_tag_block(
                        content, "requested_entries", required=False
                    )
                )
                if entry_channel_enabled and not is_final
                else []
            )
        except ValueError:
            return False
        return bool(search_items or extract_items or requested_entries)

    def _combined_entries_text(round_text: str) -> str:
        parts = [
            part
            for part in (persistent_entries_text.strip(), round_text.strip())
            if part
        ]
        return "\n\n".join(parts)

    def _log_round_artifact(payload: Dict[str, Any]) -> None:
        if task_artifact_dir:
            append_task_artifact(
                task_artifact_dir,
                kind=artifact_kind,
                task_id=task_id,
                payload=payload,
            )

    def _run_search_round(
        search_pairs: Sequence[tuple[str, str, str]],
        extract_pairs: Sequence[tuple[str, str]],
        cap: int,
        round_index: int,
    ) -> tuple[str, list[str], list[str]]:
        """Execute one search+extract round under a combined query-unit cap.

        ``search_pairs`` are ``(fact_id, query, guided)``; ``extract_pairs`` are
        ``(url, guided)``. The cap is counted in half-units: each query costs 2,
        each extract URL costs 1 (i.e. every 2 URLs == 1 query unit). Both are
        cross-round deduped on ``(request, guided)``. Returns rendered results
        plus the exact selected query/extract request lines for the next judge
        prompt's request snapshot.

        An extract URL is resolved through ``seen_urls``: the model's spelling
        only selects a URL it was shown, and the request is issued against the
        stored original. Nothing the model writes reaches the network verbatim,
        so the lookup can be forgiving about scheme, entities and IDN without
        that tolerance becoming a way to reach somewhere new.
        """

        logical_started_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        budget = max(0, int(cap)) * 2
        selected_search: List[SearchRequest] = []
        selected_meta: List[str] = []
        selected_request_lines: List[str] = []
        touched_facts: set[str] = set()
        query_facts: Dict[str, set[str]] = {}
        for fact_id, query, guided in search_pairs:
            normalized = (query.lower(), guided_query_key(guided))
            if not query or normalized in executed_norm:
                continue
            if budget < 2:
                break
            executed_norm.add(normalized)
            budget -= 2
            selected_search.append(SearchRequest(query=query, guided_query=guided))
            selected_meta.append(query)
            request_line = f"{fact_id}|{query}" if fact_id else query
            if guided:
                request_line += f"{GUIDED_QUERY_SEPARATOR}{guided}"
            selected_request_lines.append(request_line)
            if fact_id:
                touched_facts.add(fact_id)
                # Keyed by the identity a rendered section reports, so the
                # priority decrement below lands on the request that was
                # actually rendered -- never on its neighbour.
                query_facts.setdefault(request_label(query, guided), set()).add(
                    fact_id
                )
        selected_extract: List[ExtractRequest] = []
        selected_extract_lines: List[str] = []
        rejected_extract_urls: List[str] = []
        for url, guided in extract_pairs:
            if not url:
                continue
            if budget < 1:
                break
            resolved = seen_urls.get(normalize_url_key(url))
            if resolved is None:
                # Told, not ignored: an unexplained drop makes the judge re-ask
                # for the same URL every round until the budget is gone. A
                # rejected attempt still consumes its half-unit, so an output
                # full of invented destinations cannot make the next snapshot
                # exceed the advertised request cap.
                budget -= 1
                rejected_extract_urls.append(url)
                continue
            request_key = (normalize_url_key(resolved), guided_query_key(guided))
            if request_key in executed_extract_norm:
                continue
            executed_extract_norm.add(request_key)
            budget -= 1
            selected_extract.append(
                ExtractRequest(url=resolved, guided_query=guided)
            )
            extract_line = resolved
            if guided:
                extract_line += f"{GUIDED_QUERY_SEPARATOR}{guided}"
            selected_extract_lines.append(extract_line)
        results = search_client.search_many(selected_search) if selected_search else []
        extract_results = (
            search_client.extract_many(selected_extract) if selected_extract else []
        )
        # Everything this round put in front of the model becomes selectable
        # next round -- result links and the bodies of the pages we fetched.
        for search_result in results:
            _remember_seen_urls(
                *[item.url for item in search_result.items],
                *[item.snippet for item in search_result.items],
                search_result.answer,
            )
        for extract_result in extract_results:
            _remember_seen_urls(extract_result.url, extract_result.content)
        # The snapshot block already means "what the harness actually ran after
        # its caps and dedup", so a rejection belongs in it rather than in a
        # channel the judge has to be taught to read.
        selected_extract_lines.extend(
            f"{url}{GUIDED_QUERY_SEPARATOR}{EXTRACT_URL_REJECTED_NOTE}"
            for url in rejected_extract_urls
        )
        all_results.extend(results)
        all_extract_results.extend(extract_results)
        result.executed_queries.extend(selected_meta)
        result.executed_extract_urls.extend(req.url for req in selected_extract)
        result.search_rounds_executed = round_index + 1
        # Render BEFORE decrementing priorities: a query whose results were
        # truncated or dropped by the block budget never fully reached the
        # model, so its facts keep their priority and it may be re-issued.
        rendered = render_budgeted_block(
            search_result_sections(results, count_tokens=loop_count_tokens)
            + extract_result_sections(extract_results, count_tokens=loop_count_tokens),
            count_tokens=loop_count_tokens,
            section_limit=INJECTION_SECTION_MAX_TOKENS,
            block_limit=injection_block_token_limit(cap),
        )
        fully_rendered = set(rendered.included)
        decremented_facts: set[str] = set()
        # Round 0 queries are untagged (emitted by the main conversation), so
        # only follow-up rounds populate query_facts / decrement priorities.
        for label, fact_ids in query_facts.items():
            if label not in fully_rendered:
                continue
            for fact_id in fact_ids:
                if fact_id not in decremented_facts:
                    _decrement_priority(fact_index[fact_id])
                    decremented_facts.add(fact_id)
        round_payload = {
            "session": f"{exchange_prefix}-round{round_index}",
            "round": round_index,
            "logical_started_at": logical_started_at,
            "queries": selected_meta,
            "request_query_lines": selected_request_lines,
            "extract_urls": [req.url for req in selected_extract],
            "request_extract_lines": selected_extract_lines,
            "rejected_extract_urls": rejected_extract_urls,
            "touched_facts": sorted(touched_facts),
            "decremented_facts": sorted(decremented_facts),
            "executed": search_results_metadata(results),
            "extracted": extract_results_metadata(extract_results),
            "render_report": rendered.report(),
        }
        result.rounds.append(round_payload)
        _log_round_artifact(round_payload)
        return rendered.text, selected_request_lines, selected_extract_lines

    def _degraded_pack() -> SearchLoopResult:
        parts = [EVIDENCE_PACK_HEADER, DEGRADED_PACK_NOTICE]
        progress = "\n\n".join(progress_sections)
        if progress:
            parts.append(progress)
        rendered = render_budgeted_block(
            search_result_sections(all_results, count_tokens=loop_count_tokens)
            + extract_result_sections(all_extract_results, count_tokens=loop_count_tokens),
            count_tokens=loop_count_tokens,
            section_limit=INJECTION_SECTION_MAX_TOKENS,
            block_limit=injection_block_token_limit(round0_query_cap),
        )
        if rendered.text:
            parts.append(rendered.text)
        result.evidence_pack = "\n\n".join(parts)
        result.degraded = True
        result.progress_log = progress
        result.source_results_text = rendered.text
        return result

    def _persist_source_results() -> None:
        rendered = render_budgeted_block(
            search_result_sections(all_results, count_tokens=loop_count_tokens)
            + extract_result_sections(all_extract_results, count_tokens=loop_count_tokens),
            count_tokens=loop_count_tokens,
            section_limit=INJECTION_SECTION_MAX_TOKENS,
            block_limit=injection_block_token_limit(round0_query_cap),
        )
        result.source_results_text = rendered.text

    round0_search_pairs = [
        ("", *_split_guided(str(raw or ""))) for raw in round0_queries
    ]
    previous_contract_json = _render_contract(contract, contract_body)
    (
        round_results_text,
        previous_search_queries,
        previous_extract_urls,
    ) = _run_search_round(
        round0_search_pairs, [], round0_query_cap, 0
    )

    for search_round in range(max_rounds):
        is_final = search_round >= max_rounds - 1
        current_reporter().debug(
            "search round",
            {
                "round": search_round + 1,
                "of": max_rounds,
                "queries": len(result.executed_queries),
                "extracts": len(result.executed_extract_urls),
            },
        )
        progress_log = "\n\n".join(progress_sections)
        executed_display = list(result.executed_queries) + [
            f"[extract] {url}" for url in result.executed_extract_urls
        ]
        search_block = split_rendered_search_block(round_results_text)

        def _loop_complete(search_text: str):
            if loop_version == "v2":
                messages = build_search_loop_v2_messages(
                    round_index=search_round,
                    max_rounds=max_rounds,
                    is_final_round=is_final,
                    background=background,
                    contract_json=_render_contract(contract, contract_body),
                    executed_queries=executed_display,
                    previous_evidence_pack=previous_pack,
                    search_results=search_text,
                    streamer_index="" if is_final else streamer_index,
                    common_index="" if is_final else common_index,
                    knowledge_entries=_combined_entries_text(knowledge_entries_text),
                    previous_requested_entries=previous_requested_entries,
                    previous_kept_entries=previous_kept_entries,
                    previous_contract_json=previous_contract_json,
                    previous_search_queries=previous_search_queries,
                    previous_extract_urls=previous_extract_urls,
                    followup_query_cap=followup_query_cap,
                )
            else:
                messages = build_search_loop_messages(
                    round_index=search_round,
                    max_rounds=max_rounds,
                    is_final_round=is_final,
                    background=background,
                    contract_json=_render_contract(contract, contract_body),
                    executed_queries=executed_display,
                    progress_log=progress_log,
                    search_results=search_text,
                    streamer_index="" if is_final else streamer_index,
                    common_index="" if is_final else common_index,
                    knowledge_entries=_combined_entries_text(knowledge_entries_text),
                    previous_requested_entries=previous_requested_entries,
                    previous_kept_entries=previous_kept_entries,
                    previous_contract_json=previous_contract_json,
                    previous_search_queries=previous_search_queries,
                    previous_extract_urls=previous_extract_urls,
                    followup_query_cap=followup_query_cap,
                )
            # Persist the full round input state for session_replay fixture
            # extraction (docs/session_replay.md 补中间态).
            if task_artifact_dir:
                _dump_search_loop_round_input(
                    task_artifact_dir,
                    round_index=search_round,
                    max_rounds=max_rounds,
                    is_final_round=is_final,
                    background=background,
                    contract_json=_render_contract(contract, contract_body),
                    executed_queries=executed_display,
                    progress_log=progress_log,
                    search_results=search_text,
                    streamer_index="" if is_final else streamer_index,
                    common_index="" if is_final else common_index,
                    knowledge_entries=_combined_entries_text(knowledge_entries_text),
                    previous_requested_entries=list(previous_requested_entries),
                    previous_kept_entries=list(previous_kept_entries),
                    previous_contract_json=previous_contract_json,
                    previous_search_queries=list(previous_search_queries),
                    previous_extract_urls=list(previous_extract_urls),
                    followup_query_cap=followup_query_cap,
                    previous_evidence_pack=previous_pack,
                )
            checkpoint_hash = session_input_hash(
                messages,
                prompt_version=PROMPT_VERSION,
                call_config={
                    "role": LLMRole.LIGHTWEIGHT.value,
                    "max_tokens": SEARCH_LOOP_MAX_TOKENS,
                    "loop_version": loop_version,
                    # In the checkpoint key: difficulty can bind a different
                    # model group (and thinking knob) for search_judge, so a
                    # cached judge round from another difficulty must not be
                    # replayed into this one.
                    "difficulty": difficulty,
                },
                execution_identity_override=getattr(
                    client, "execution_identity", None
                ),
            )
            checkpoint_key = f"{exchange_prefix}:round{search_round}"
            cached = checkpoint_store.get(
                "search-judge", checkpoint_key, checkpoint_hash
            )
            if cached is not None and _checkpoint_content_accepted(
                cached.content, is_final=is_final
            ):
                call = LLMCallResult(
                    content=cached.content,
                    role=LLMRole.LIGHTWEIGHT,
                    model=str(cached.metadata.get("model") or "checkpoint"),
                    fallback_used=bool(cached.metadata.get("fallback_used", False)),
                    raw_response={},
                )
                return call, messages, search_text, checkpoint_hash, checkpoint_key, True
            if cached is not None and task_artifact_dir:
                append_task_artifact(
                    task_artifact_dir,
                    kind="session_checkpoint_invalid",
                    task_id=task_id,
                    payload={
                        "session": "search-judge",
                        "key": checkpoint_key,
                        "input_hash": checkpoint_hash,
                    },
                )
            try:
                call = client.complete(
                    LLMRole.LIGHTWEIGHT,
                    messages,
                    max_tokens=SEARCH_LOOP_MAX_TOKENS,
                    task_group="search_judge",
                    difficulty=difficulty,
                    **validation_retry_sampling_kwargs(0),
                )
            except Exception as exc:  # pragma: no cover - provider behavior
                _log_round_artifact(
                    {
                        "round": search_round,
                        "call_error": f"{type(exc).__name__}: {exc}",
                        "api_attempts": list(
                            getattr(exc, "_harness_api_attempts", []) or []
                        ),
                        "execution_attempts": list(
                            getattr(exc, "_harness_execution_attempts", []) or []
                        ),
                        "route_decision": dict(
                            getattr(exc, "_harness_route_decision", {}) or {}
                        ),
                    }
                )
                raise
            if is_prompt_blocked(call.content, call.raw_response):
                raise GeminiPromptBlockedError(
                    f"search loop round {search_round} prompt was blocked by "
                    "the content filter"
                )
            return call, messages, search_text, checkpoint_hash, checkpoint_key, False

        try:
            ladder_outcome = run_injection_ladder(
                block=search_block,
                call=lambda text: _loop_complete(text),
                stage=f"search_loop_round_{search_round}",
                blocked_exception=GeminiPromptBlockedError,
                blacklist=blacklist,
                task_artifact_dir=task_artifact_dir,
                task_id=task_id,
                plain_retry=not search_block.units,
            )
        except Exception:
            # Provider errors and content-filter exhaustion both degrade —
            # the search loop never fails the surrounding task.
            return _degraded_pack()
        (
            call,
            messages,
            active_search_text,
            checkpoint_hash,
            checkpoint_key,
            checkpoint_replayed,
        ) = ladder_outcome.result
        round_results_text = active_search_text

        pack = ""
        followup_search_pairs: List[tuple[str, str, str]] = []
        followup_extract_pairs: List[tuple[str, str]] = []
        requested_entry_names: List[str] = []
        progress_update = ""
        parse_error = ""
        for attempt in range(max_parse_retries + 1):
            if attempt > 0:
                checkpoint_replayed = False
                try:
                    call = client.complete(
                        LLMRole.LIGHTWEIGHT,
                        messages,
                        max_tokens=SEARCH_LOOP_MAX_TOKENS,
                        task_group="search_judge",
                        difficulty=difficulty,
                        **validation_retry_sampling_kwargs(attempt),
                    )
                except Exception as exc:  # pragma: no cover - provider behavior
                    _log_round_artifact(
                        {
                            "round": search_round,
                            "call_error": f"{type(exc).__name__}: {exc}",
                            "attempt": attempt,
                            "api_attempts": list(
                                getattr(exc, "_harness_api_attempts", []) or []
                            ),
                            "execution_attempts": list(
                                getattr(exc, "_harness_execution_attempts", []) or []
                            ),
                            "route_decision": dict(
                                getattr(exc, "_harness_route_decision", {}) or {}
                            ),
                        }
                    )
                    return _degraded_pack()
                if is_prompt_blocked(call.content, call.raw_response):
                    # Same prompt already cleared the ladder; a new block on a
                    # parse retry is treated as unrecoverable for this round.
                    return _degraded_pack()
            if token_rows is not None and not checkpoint_replayed:
                token_rows.append(
                    {
                        "call": "search_loop",
                        "round": search_round,
                        "attempt": attempt,
                        "model": call.model,
                        "tokens": extract_token_distribution(call.raw_response),
                    }
                )
            progress_update = _extract_optional_block(
                call.content,
                "progress_update",
                max_tokens=PROGRESS_UPDATE_MAX_TOKENS,
                count_tokens=loop_count_tokens,
            ) if loop_version == "v1" else ""
            pack = _extract_optional_block(
                call.content,
                "evidence_pack",
                max_tokens=EVIDENCE_PACK_MAX_TOKENS,
                count_tokens=loop_count_tokens,
            )
            followup_search_pairs = []
            followup_extract_pairs = []
            requested_entry_names = []
            if loop_version == "v2":
                # v2: pack is required; queries are optional (termination signal).
                try:
                    raw_search = parse_guided_line_items(
                        extract_single_tag_block(
                            call.content, "search_queries", required=False
                        )
                    )
                    raw_extract = parse_guided_line_items(
                        extract_single_tag_block(
                            call.content, "extract_urls", required=False
                        )
                    )
                    fact_ids = tuple(fact_index)
                    followup_search_pairs = [
                        (*_split_fact_tag(text, fact_ids), guided)
                        for text, guided in raw_search
                    ]
                    followup_extract_pairs = list(raw_extract)
                    if entry_channel_enabled and not is_final:
                        requested_entry_names = parse_line_items(
                            extract_single_tag_block(
                                call.content, "requested_entries", required=False
                            )
                        )
                except ValueError:
                    pass  # queries are optional in v2
                parse_error = "" if pack else "missing <evidence_pack> (required in v2)"
            elif not pack:
                try:
                    raw_search = parse_guided_line_items(
                        extract_single_tag_block(
                            call.content, "search_queries", required=False
                        )
                    )
                    raw_extract = parse_guided_line_items(
                        extract_single_tag_block(
                            call.content, "extract_urls", required=False
                        )
                    )
                    fact_ids = tuple(fact_index)
                    followup_search_pairs = [
                        (*_split_fact_tag(text, fact_ids), guided)
                        for text, guided in raw_search
                    ]
                    followup_extract_pairs = list(raw_extract)
                    if entry_channel_enabled and not is_final:
                        requested_entry_names = parse_line_items(
                            extract_single_tag_block(
                                call.content, "requested_entries", required=False
                            )
                        )
                    parse_error = (
                        ""
                        if (
                            followup_search_pairs
                            or followup_extract_pairs
                            or requested_entry_names
                        )
                        else "output has no <search_queries> or <extract_urls> block"
                    )
                except ValueError as exc:
                    parse_error = str(exc)
            else:
                parse_error = ""
            session = infer_session_name(
                "search_loop_round",
                {
                    "round": search_round,
                    "attempt": attempt,
                    "session": f"{exchange_prefix}-round{search_round}-attempt{attempt}",
                },
            )
            input_components = search_loop_input_components(
                counter=token_counter,
                search_results=round_results_text,
                messages=messages,
            )
            if exchange_logger and not checkpoint_replayed:
                exchange_logger.log(
                    session,
                    messages=messages,
                    response_text=call.content,
                    metadata=llm_exchange_metadata(
                        call,
                        session=session,
                        input_components=input_components,
                        round=search_round,
                        attempt=attempt,
                        is_final_round=is_final,
                        has_evidence_pack=bool(pack),
                        query_count=len(followup_search_pairs),
                        extract_count=len(followup_extract_pairs),
                        **({"parse_error": parse_error} if parse_error else {}),
                    ),
                )
            if not checkpoint_replayed:
                _log_round_artifact(
                    {
                        "session": session,
                        "round": search_round,
                        "attempt": attempt,
                        "model": call.model,
                        "is_final_round": is_final,
                        "has_evidence_pack": bool(pack),
                        "queries": [query for _, query, _ in followup_search_pairs],
                        "extract_urls": [url for url, _ in followup_extract_pairs],
                        "progress_update_chars": len(progress_update),
                        "parse_error": parse_error,
                        "usage": extract_token_distribution(call.raw_response),
                        "api_attempts": list(call.api_attempts),
                        "execution_attempts": list(call.execution_attempts),
                        "route_decision": dict(call.route_decision),
                        "input_components": input_components,
                        "response_content": call.content,
                    }
                )
            if loop_version == "v2":
                if pack:
                    break
            elif (
                pack
                or followup_search_pairs
                or followup_extract_pairs
                or requested_entry_names
            ):
                break
        accepted = bool(
            pack
            or (
                loop_version == "v1"
                and (
                    followup_search_pairs
                    or followup_extract_pairs
                    or requested_entry_names
                )
            )
        )
        if accepted:
            if checkpoint_replayed:
                if task_artifact_dir:
                    append_task_artifact(
                        task_artifact_dir,
                        kind="session_checkpoint_replay",
                        task_id=task_id,
                        payload={
                            "session": "search-judge",
                            "key": checkpoint_key,
                            "input_hash": checkpoint_hash,
                        },
                    )
            elif getattr(call, "resumable", True):
                # Gate D answer C: an implicit-history call must not seed L1
                # (docs/llm_local_agent.md §7); the resume re-sends it instead.
                checkpoint_store.commit(
                    session="search-judge",
                    key=checkpoint_key,
                    input_hash=checkpoint_hash,
                    content=call.content,
                    metadata={
                        "model": call.model,
                        "fallback_used": call.fallback_used,
                    },
                )
        if progress_update:
            progress_sections.append(f"## 搜索轮 {search_round}\n{progress_update}")
        if pack:
            result.evidence_pack = f"{EVIDENCE_PACK_HEADER}\n\n{pack}"
            result.progress_log = "\n\n".join(progress_sections)
            missing_facts = _check_fact_coverage(pack, fact_index)
            if missing_facts:
                _log_round_artifact(
                    {
                        "round": search_round,
                        "warning": "evidence_pack_missing_facts",
                        "missing_fact_ids": missing_facts,
                    }
                )
            if loop_version == "v2":
                # v2: store pack for next round's <previous_evidence_pack>.
                previous_pack = pack
                has_followup = bool(
                    followup_search_pairs
                    or followup_extract_pairs
                    or requested_entry_names
                )
                if is_final or not has_followup:
                    _persist_source_results()
                    return result
                # Otherwise fall through to execute queries.
            else:
                # v1: pack = immediate termination.
                if not is_final:
                    premature_warnings = _premature_pack_warnings(
                        result.progress_log, fact_index
                    )
                    if premature_warnings:
                        _log_round_artifact(
                            {
                                "round": search_round,
                                "warning": "premature_evidence_pack",
                                "unresolved_high_priority_facts": premature_warnings,
                                "remaining_rounds": max_rounds - search_round - 1,
                            }
                        )
                _persist_source_results()
                return result
        if is_final or not (
            followup_search_pairs or followup_extract_pairs or requested_entry_names
        ):
            # Final round without a pack, or a non-final round that produced
            # neither pack nor any request after retries: degrade instead of
            # failing the task.
            return _degraded_pack()
        # Knowledge-entry requests: resolved locally, cross-round deduped by
        # primary key, count-capped like follow-up queries, budget-rendered
        # into the NEXT round's <knowledge_entries> block (independent budget
        # from the search-results block).
        knowledge_entries_text = ""
        if requested_entry_names and knowledge_root is not None:
            found_entries, missing_entries = load_entry_texts(
                knowledge_root, requested_entry_names
            )
            fresh_entries = {
                key: body
                for key, body in found_entries.items()
                if key not in injected_entry_keys
            }
            fresh_entries = dict(list(fresh_entries.items())[:followup_query_cap])
            entry_report: Dict[str, Any] = {}
            if fresh_entries:
                entry_block = render_knowledge_entries_block(
                    fresh_entries,
                    count_tokens=loop_count_tokens,
                    entry_limit=INJECTION_SECTION_MAX_TOKENS,
                    block_limit=injection_block_token_limit(followup_query_cap),
                )
                knowledge_entries_text = entry_block.text
                entry_report = entry_block.report()
                injected_entry_keys.update(fresh_entries)
                # Entry bodies routinely cite sources; the next round may
                # extract those, same as any other link it was shown.
                _remember_seen_urls(knowledge_entries_text)
            _log_round_artifact(
                {
                    "round": search_round,
                    "requested_entries": requested_entry_names,
                    "injected_entries": sorted(fresh_entries),
                    "missing_entries": missing_entries,
                    "entry_render_report": entry_report,
                }
            )
        previous_requested_entries = list(requested_entry_names)
        previous_kept_entries = []
        previous_contract_json = _render_contract(contract, contract_body)
        (
            round_results_text,
            previous_search_queries,
            previous_extract_urls,
        ) = _run_search_round(
            followup_search_pairs, followup_extract_pairs, followup_query_cap, search_round + 1
        )

    return _degraded_pack()  # pragma: no cover - loop always returns earlier
