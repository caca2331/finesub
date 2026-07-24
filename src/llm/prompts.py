"""Chinese prompt templates for subtitle correction and translation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Mapping, Sequence

from .chunking import (
    SubtitleSegment,
    SubtitleWindow,
    render_segments_as_csv,
)
from .config import (
    CapabilityTier,
    DEFAULT_RESEARCH_SEARCH_QUERIES,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    MAX_WINDOW_SEARCH_QUERIES,
)
from .profiles import DEFAULT_PROFILE, TranslationProfile
from .prompt_compose import (
    PROMPT_TEMPLATE_DIR,
    PROMPT_VERSION,
    compose_correction_query_system,
    compose_correction_system,
    compose_correction_user,
    compose_fast_round1_system,
    compose_fast_round1_user,
    ensure_csv_block_headers,
    load_prompt_template as _load_prompt_template,
    reasoning_clause,
)


def _query_style_rules() -> str:
    """Query-writing style rules shared by both search emission fragments."""

    return _load_prompt_template("fragment_query_style_v1.md").strip()


def _search_queries_rules(max_queries: int) -> str:
    return _load_prompt_template(
        "fragment_search_queries_output_v1.md",
        max_queries=max_queries,
        query_style=_query_style_rules(),
    ).strip()


def _search_contract_rules(max_queries: int) -> str:
    """Multi-round variant of the search emission rules (contract + round 0)."""

    return _load_prompt_template(
        "fragment_search_contract_output_v1.md",
        max_queries=max_queries,
        query_style=_query_style_rules(),
    ).strip()


def _search_results_usage() -> str:
    return _load_prompt_template("fragment_search_results_usage_v1.md").strip()


def _evidence_pack_usage() -> str:
    """Multi-round variant of the search results usage rules."""

    return _load_prompt_template("fragment_evidence_pack_usage_v1.md").strip()


def _feedback_schema() -> str:
    return _load_prompt_template("fragment_task_feedback_schema_v3.md").strip()


def _research_task_feedback_block() -> str:
    """Feedback-collection addendum for the research final round / fast round 1."""

    return "\n" + _load_prompt_template(
        "research_task_feedback_v1.md", feedback_schema=_feedback_schema()
    ).strip() + "\n"


# Appended to the round's closing reminder when feedback collection is on.
TASK_FEEDBACK_REMINDER = "，随后按 system 要求输出一个 `<task_update_feedback>` 块"


@dataclass(frozen=True)
class ContextPack:
    """Research output injected into correction windows.

    ``general_context`` covers every window; ``window_contexts`` maps window ids
    (correction chunk ids) to window-specific context text.
    """

    general_context: Mapping[str, Any] = field(default_factory=dict)
    window_contexts: Mapping[str, str] = field(default_factory=dict)

    def general_prompt_text(self) -> str:
        if not self.general_context:
            return "{}"
        return json.dumps(dict(self.general_context), ensure_ascii=False, indent=2)

    def window_context_for(self, chunk_id: str) -> str:
        if chunk_id in self.window_contexts:
            return self.window_contexts[chunk_id]
        # Split-retry windows ("0001-a", "0001-a-b", ...) inherit the parent context.
        base = chunk_id.split("-", 1)[0]
        return self.window_contexts.get(base, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "general_context": dict(self.general_context),
            "window_contexts": dict(self.window_contexts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextPack":
        general = data.get("general_context") or {}
        if not isinstance(general, Mapping):
            general = {"global_summary": str(general)}
        raw_windows = data.get("window_contexts") or {}
        window_contexts: dict[str, str] = {}
        if isinstance(raw_windows, Mapping):
            window_contexts = {str(k): str(v) for k, v in raw_windows.items()}
        elif isinstance(raw_windows, Sequence):
            for item in raw_windows:
                if isinstance(item, Mapping) and item.get("window_id"):
                    window_contexts[str(item["window_id"])] = str(item.get("context", ""))
        return cls(general_context=dict(general), window_contexts=window_contexts)


def render_advice_ledger(entries: Sequence[tuple[str, str]]) -> str:
    """Render the accumulated per-window ``<next_advice>`` ledger.

    Each entry is ``(chunk_id, advice)``; empty advice entries are skipped.
    The full ledger is injected into both the query round and the correction
    round, so each window sees every earlier window's advice.
    """

    parts = [
        f"[window {chunk_id}]\n{advice.strip()}"
        for chunk_id, advice in entries
        if (advice or "").strip()
    ]
    return "\n\n".join(parts)


def build_research_round1_messages(
    *,
    transcript: str,
    extra_info: str = "",
    note_url_extracts: str = "",
    streamer_index: str = "",
    common_index: str = "",
    preinjected_entries: str = "",
    max_search_queries: int = DEFAULT_RESEARCH_SEARCH_QUERIES,
    use_search_contract: bool = False,
) -> List[Dict[str, str]]:
    """Round 1 research prompt.

    ``use_search_contract`` swaps the single-round query emission fragment for
    the multi-round variant (Research Contract + round-0 queries); everything
    else stays identical, which is the pluggable-prompt seam.
    ``preinjected_entries`` carries the budget-rendered knowledge entries the
    harness matched against the user note's keys/aliases.
    """

    search_rules = (
        _search_contract_rules(max_search_queries)
        if use_search_contract
        else _search_queries_rules(max_search_queries)
    )
    system = _load_prompt_template(
        "research_round1_v1.md",
        search_queries_rules=search_rules,
        reasoning_clause=reasoning_clause("medium"),
        max_requested_entries=KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
        max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
        max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
    )
    user = _load_prompt_template(
        "research_round1_user_v1.md",
        extra_info=extra_info or "（无）",
        note_url_extracts=note_url_extracts.strip() or "（无）",
        streamer_index=streamer_index or "（空）",
        common_index=common_index or "（空）",
        preinjected_entries=preinjected_entries.strip() or "（无）",
        transcript=transcript,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_research_round2_messages(
    *,
    transcript: str,
    extra_info: str = "",
    round1_notes: str = "",
    entry_details_text: str = "",
    search_results: str = "",
    use_evidence_pack: bool = False,
    collect_task_feedback: bool = False,
) -> List[Dict[str, str]]:
    """Round 2 research prompt.

    ``use_evidence_pack`` swaps the raw-search-results usage fragment for the
    evidence-pack variant; ``search_results`` then carries the rendered
    evidence pack text in the same prompt slot. ``entry_details_text`` is the
    budget-rendered knowledge-entry block from round 1's requests.
    ``collect_task_feedback`` additionally asks for a trailing
    ``<task_update_feedback>`` block (schema v3).
    """

    system = _load_prompt_template(
        "research_round2_v1.md",
        reasoning_clause=reasoning_clause("medium"),
        search_results_usage=(
            _evidence_pack_usage() if use_evidence_pack else _search_results_usage()
        ),
        task_update_feedback_block=(
            _research_task_feedback_block() if collect_task_feedback else ""
        ),
    )
    if isinstance(search_results, list):
        search_results = "\n".join(search_results)
    user = _load_prompt_template(
        "research_round2_user_v1.md",
        extra_info=extra_info or "（无）",
        round1_notes=round1_notes.strip() or "（无）",
        entry_details=entry_details_text.strip() or "（无）",
        search_results=search_results.strip() or "（无）",
        transcript=transcript,
        task_feedback_reminder=(
            TASK_FEEDBACK_REMINDER if collect_task_feedback else ""
        ),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_search_loop_messages(
    *,
    round_index: int,
    max_rounds: int,
    is_final_round: bool,
    background: str = "",
    contract_json: str = "",
    executed_queries: Sequence[str] = (),
    progress_log: str = "",
    search_results: str = "",
    streamer_index: str = "",
    common_index: str = "",
    knowledge_entries: str = "",
    previous_requested_entries: Sequence[str] = (),
    previous_kept_entries: Sequence[str] = (),
    previous_contract_json: str = "",
    previous_search_queries: Sequence[str] = (),
    previous_extract_urls: Sequence[str] = (),
    followup_query_cap: int = 4,
) -> List[Dict[str, str]]:
    """Prompt for one lightweight search-loop call (after each search round).

    ``streamer_index``/``common_index`` expose the local knowledge indices on
    non-final rounds (the harness passes "" on the final round, where entry
    requests are forbidden); ``knowledge_entries`` carries the budget-rendered
    bodies requested in the previous round. The ``previous_*`` fields preserve
    the exact entry selection and executed search request that produced the
    immediately following raw results block.
    """

    system = _load_prompt_template(
        "search_loop_v1.md",
        followup_cap=followup_query_cap,
        reasoning_clause=reasoning_clause("medium"),
    )
    # The round notice text lives in template fragments (prompt text never
    # hardcoded in Python); Python only selects which one and fills the
    # remaining-round count.
    if is_final_round:
        round_notice = _load_prompt_template(
            "fragment_search_loop_final_notice_v1.md"
        ).strip()
    else:
        round_notice = _load_prompt_template(
            "fragment_search_loop_continue_notice_v1.md",
            remaining_rounds=max(0, max_rounds - round_index - 1),
        ).strip()
    user = _load_prompt_template(
        "search_loop_user_v1.md",
        round_index=round_index,
        max_rounds=max_rounds,
        round_notice=round_notice,
        background=background.strip() or "（无）",
        contract_json=contract_json.strip() or "（无）",
        executed_queries="\n".join(executed_queries) or "（无）",
        progress_log=progress_log.strip() or "（无）",
        streamer_index=streamer_index.strip() or "（空）",
        common_index=common_index.strip() or "（空）",
        previous_requested_entries=(
            "\n".join(previous_requested_entries) or "（无）"
        ),
        previous_kept_entries="\n".join(previous_kept_entries) or "（无）",
        knowledge_entries=knowledge_entries.strip() or "（无）",
        previous_contract_json=previous_contract_json.strip() or "（无）",
        previous_search_queries="\n".join(previous_search_queries) or "（无）",
        previous_extract_urls="\n".join(previous_extract_urls) or "（无）",
        search_results=search_results.strip() or "（无）",
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_search_loop_v2_messages(
    *,
    round_index: int,
    max_rounds: int,
    is_final_round: bool,
    background: str = "",
    contract_json: str = "",
    executed_queries: Sequence[str] = (),
    previous_evidence_pack: str = "",
    search_results: str = "",
    streamer_index: str = "",
    common_index: str = "",
    knowledge_entries: str = "",
    previous_requested_entries: Sequence[str] = (),
    previous_kept_entries: Sequence[str] = (),
    previous_contract_json: str = "",
    previous_search_queries: Sequence[str] = (),
    previous_extract_urls: Sequence[str] = (),
    followup_query_cap: int = 4,
) -> List[Dict[str, str]]:
    """V2 search-loop prompt: every round emits a full Evidence Pack.

    Replaces the v1 binary "continue OR pack" with "always pack, optionally
    queries". ``previous_evidence_pack`` carries the prior round's pack for
    incremental update (empty on round 0). No progress_update block.
    """

    system = _load_prompt_template(
        "search_loop_v2.md",
        followup_cap=followup_query_cap,
        reasoning_clause=reasoning_clause("medium"),
    )
    if is_final_round:
        round_notice = _load_prompt_template(
            "fragment_search_loop_final_notice_v1.md"
        ).strip()
    else:
        round_notice = _load_prompt_template(
            "fragment_search_loop_continue_notice_v1.md",
            remaining_rounds=max(0, max_rounds - round_index - 1),
        ).strip()
    user = _load_prompt_template(
        "search_loop_user_v2.md",
        round_index=round_index,
        max_rounds=max_rounds,
        round_notice=round_notice,
        background=background.strip() or "（无）",
        contract_json=contract_json.strip() or "（无）",
        executed_queries="\n".join(executed_queries) or "（无）",
        previous_evidence_pack=previous_evidence_pack.strip() or "（无）",
        streamer_index=streamer_index.strip() or "（空）",
        common_index=common_index.strip() or "（空）",
        previous_requested_entries=(
            "\n".join(previous_requested_entries) or "（无）"
        ),
        previous_kept_entries="\n".join(previous_kept_entries) or "（无）",
        knowledge_entries=knowledge_entries.strip() or "（无）",
        previous_contract_json=previous_contract_json.strip() or "（无）",
        previous_search_queries="\n".join(previous_search_queries) or "（无）",
        previous_extract_urls="\n".join(previous_extract_urls) or "（无）",
        search_results=search_results.strip() or "（无）",
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_correction_query_messages(
    *,
    window: SubtitleWindow,
    context_pack: ContextPack | None = None,
    audio_file_label: str = "",
    previous_advice: str = "",
    streamer_index: str = "",
    common_index: str = "",
    carried_entries: str = "",
    carried_entry_count: int = 0,
    max_search_queries: int = MAX_WINDOW_SEARCH_QUERIES,
    profile: TranslationProfile = DEFAULT_PROFILE,
) -> List[Dict[str, Any]]:
    """Query-round messages.

    ``carried_entries`` is the budget-rendered full text of entries kept by
    the previous window's correction round (v17 pass-through); its count
    shrinks the new-request allowance shown to the model
    (min(KB_WINDOW_NEW_REQUEST_MAX_ENTRIES, KB_WINDOW_TOTAL_ENTRIES - carried)).
    """

    pack = context_pack or ContextPack()
    remaining_entries = max(
        0,
        min(
            KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
            KB_WINDOW_TOTAL_ENTRIES - max(0, int(carried_entry_count)),
        ),
    )
    system = compose_correction_query_system(
        profile,
        search_queries_rules=_search_queries_rules(max_search_queries),
        max_entries=remaining_entries,
        total_entries=KB_WINDOW_TOTAL_ENTRIES,
    )
    user = ensure_csv_block_headers(_load_prompt_template(
        "correction_query_user_v1.md",
        general_context_json=pack.general_prompt_text(),
        window_context=pack.window_context_for(window.chunk_id) or "（无）",
        previous_advice=previous_advice.strip() or "（无）",
        streamer_index=streamer_index or "（空）",
        common_index=common_index or "（空）",
        carried_entries=carried_entries.strip() or "（无）",
        remaining_entries=remaining_entries,
        current_asr_csv=render_segments_as_csv(
            window.segments, window_start=window.clip_start
        ).strip(),
    ))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_correction_csv_messages(
    *,
    window: SubtitleWindow,
    context_pack: ContextPack | None = None,
    audio_file_label: str = "",
    previous_advice: str = "",
    query_round_notes: str = "",
    search_results: str = "",
    entry_details: str = "",
    extra_style: str = "",
    common_mistakes_block: str = "",
    task_update_feedback: bool = False,
    evidence_pack_mode: bool = False,
    profile: TranslationProfile = DEFAULT_PROFILE,
    tier: CapabilityTier = CapabilityTier.CAPABLE,
    variant: str | None = None,
) -> List[Dict[str, Any]]:
    """Correction-round messages.

    ``query_round_notes`` fills the generic ``<pre_round_notes>`` slot (the
    query round's window notes in normal mode, fast round 1's analysis notes
    in fast mode); ``entry_details`` and ``evidence_pack_mode`` are only
    non-default in fast mode. ``tier`` follows the capability tier of the
    endpoint that answers (see ``compose_correction_system``); ``variant``
    overrides the tier-derived prompt set by name (session_replay control).
    """

    pack = context_pack or ContextPack()
    current_csv = render_segments_as_csv(
        window.segments, window_start=window.clip_start
    ).strip()
    # Read-only raw lines before the window (v13), rendered on the same time
    # base as current_asr_csv so they come out mostly negative.
    preceding_csv = render_segments_as_csv(
        window.preceding_segments,
        window_start=window.clip_start,
        allow_negative_start=True,
    ).strip()
    system = compose_correction_system(
        profile,
        tier=tier,
        variant=variant,
        evidence_pack_mode=evidence_pack_mode,
        extra_style=extra_style,
        common_mistakes_block=common_mistakes_block,
    )
    if task_update_feedback:
        system = (
            f"{system.rstrip()}\n\n"
            + _load_prompt_template(
                "correction_task_update_feedback_v2.md",
                feedback_schema=_feedback_schema(),
            )
        )
    user = compose_correction_user(
        profile,
        general_context_json=pack.general_prompt_text(),
        window_context=pack.window_context_for(window.chunk_id) or "（无）",
        entry_details=entry_details.strip() or "（无）",
        previous_advice=previous_advice.strip() or "（无）",
        pre_round_notes=query_round_notes.strip() or "（无）",
        search_results=search_results.strip() or "（无）",
        preceding_context_csv=preceding_csv,
        current_asr_csv=current_csv,
        current_asr_row_count=len(window.segments),
        tier=tier,
        variant=variant,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_fast_round1_messages(
    *,
    window: SubtitleWindow,
    audio_file_label: str = "",
    extra_info: str = "",
    note_url_extracts: str = "",
    streamer_index: str = "",
    common_index: str = "",
    preinjected_entries: str = "",
    max_search_queries: int = DEFAULT_RESEARCH_SEARCH_QUERIES,
    use_search_contract: bool = False,
    collect_task_feedback: bool = False,
    profile: TranslationProfile = DEFAULT_PROFILE,
) -> List[Dict[str, Any]]:
    """Fast-mode round 1: fused research round 1 + per-window query round.

    ``window`` is the single fast window covering the whole input; the CSV is
    rendered clip-relative exactly like a correction window. Fast round 1 is
    the research final round's equivalent feedback collection point, so
    ``collect_task_feedback`` adds the same trailing block request.
    """

    search_rules = (
        _search_contract_rules(max_search_queries)
        if use_search_contract
        else _search_queries_rules(max_search_queries)
    )
    # Request and keep have independent caps; keep wins their shared cap.
    system = compose_fast_round1_system(
        profile,
        search_queries_rules=search_rules,
        task_update_feedback_block=(
            _research_task_feedback_block() if collect_task_feedback else ""
        ),
        max_requested_entries=KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
        max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
        max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
    )
    user = compose_fast_round1_user(
        extra_info=extra_info or "（无）",
        note_url_extracts=note_url_extracts.strip() or "（无）",
        streamer_index=streamer_index or "（空）",
        common_index=common_index or "（空）",
        preinjected_entries=preinjected_entries,
        current_asr_csv=render_segments_as_csv(
            window.segments, window_start=window.clip_start
        ).strip(),
        task_feedback_reminder=(
            TASK_FEEDBACK_REMINDER if collect_task_feedback else ""
        ),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_knowledge_update_messages(
    *,
    refined: bool,
    task_summary: str,
    window_packs: str,
    general_context: str = "",
    research_feedback: str = "",
    aggregated_feedback: str = "",
    kb_entries: str = "",
    streamer_index: str = "",
    common_index: str = "",
    chunk_index: int = 1,
    multi_chunk: bool = False,
    window_range: str = "",
    prompt_version: str = PROMPT_VERSION,
) -> List[Dict[str, str]]:
    """Unified knowledge-update prompt (docs/knowledge.md).

    ``refined`` selects the ``refined_aligned`` variant (mistake ledger +
    harness notes enabled); the ``artifacts_only`` variant never mentions the
    mistake block — the harness ignores one if the model emits it anyway.
    ``window_packs`` is the rendered per-window material blocks of ONE chunk;
    ``multi_chunk`` marks a 100k-budget chunked run (the notice states no
    total, since over-limit chunks may split further at run time).
    Existing common-mistake / good-example ledgers are not injected here;
    cross-task ledger maintenance is deferred to a dedicated module.
    """

    structure = _load_prompt_template("fragment_knowledge_structure_v1.md").strip()
    output_rules = _load_prompt_template(
        "fragment_knowledge_output_v1.md",
        reasoning_clause=reasoning_clause("medium"),
    ).strip()
    inputs_block = _load_prompt_template(
        "fragment_knowledge_update_inputs_v1.md",
        refined_csv_bullet=(
            (
                "\n   - `<refined_csv>`：落在该窗口时间范围内的人工精修字幕行，"
                "`start|end|text`（已按开始时间重排；可能缺失，表示该窗口没有精修行）。"
            )
            if refined
            else ""
        ),
    ).strip()
    template_name = (
        "knowledge_update_refined_v1.md"
        if refined
        else "knowledge_update_artifacts_only_v1.md"
    )
    system = _load_prompt_template(
        template_name,
        knowledge_inputs=inputs_block,
        knowledge_structure=structure,
        knowledge_output=output_rules,
    )
    if multi_chunk:
        chunk_notice = (
            f"\n材料分块说明：本任务材料按 token 预算分为多块，本次调用是第 "
            f"{chunk_index} 块，只含窗口 {window_range}；其余块会在独立调用中处理并"
            "依次写入知识库。只根据本块材料提出更新，不要为块外内容预留占位。\n"
        )
    else:
        chunk_notice = ""
    if refined:
        final_reminder = (
            "先以一个 `<reasoning>` 块开头；随后依次输出 `<knowledge_proposals>`、"
            "`<mistake_proposals>`（均可为空块）；除上述块外不要输出任何其他文字。"
        )
    else:
        final_reminder = (
            "先以一个 `<reasoning>` 块开头；随后输出有且仅有一个"
            " `<knowledge_proposals>` 块（可为空块）；除上述块外不要输出任何其他文字。"
        )
    user = _load_prompt_template(
        "knowledge_update_user_v1.md",
        task_summary=task_summary.strip() or "（无）",
        task_prompt_version=prompt_version,
        chunk_notice=chunk_notice,
        general_context=general_context.strip() or "（无）",
        research_feedback=research_feedback.strip() or "（无）",
        aggregated_feedback=aggregated_feedback.strip() or "（无）",
        kb_entries=kb_entries.strip() or "（无）",
        streamer_index=streamer_index.strip() or "（空）",
        common_index=common_index.strip() or "（空）",
        window_packs=window_packs.strip() or "（无）",
        final_reminder=final_reminder,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
