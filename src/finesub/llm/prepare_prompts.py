"""Prompts of the global preparation task (plan §2.1: one task, one logical conversation).

The old research session was two heterogeneous prompts -- R1 "ask" and R2
"digest", each with its own system prompt and its own parser. The
preparation task is ONE contract: the first message states the whole task
and the deliverable; the model either delivers ``<context_pack>`` at once or
first emits one batch of tool requests (the same ``<search_queries>`` /
``<requested_entries>`` / ``<keep_entries>`` blocks R1 used, so their parsers
and caps carry over). The harness executes the batch and appends the results
as the next user turn (:func:`build_prepare_results_message`); the same
conversation then delivers.

Capabilities shape *what may be requested*, never whether the task exists
(plan §4): with no search and no readable index there is simply no request
channel, and the prompt says the pack must not dress inference up as
verified fact. Under the agentic strategy (stage C) the request channel is
replaced by the executor's own tools; ``--style auto`` (stage E,
``docs/knowledge.md``, 「style 类别」) adds a candidate catalog and a
``<style_selection>`` block to the delivery.

Every sentence the model reads lives in ``prompt_templates/``; this module
only chooses and fills fragments.
"""

from __future__ import annotations

from string import Template
from typing import Dict, List, Mapping

from .prompts import (
    _load_prompt_template,
    _numbered,
    _r1_entry_blocks,
    _research_task_feedback_block,
    _search_queries_rules,
    render_knowledge_indexes,
)
from .prompt_compose import (
    prompt_phrases, renumber_rules, search_results_usage, search_rules_shared, task_mode_texts,
)
from .knowledge.style import FACTORY_STYLE_NAME, catalog_offers_default, factory_style_text
from .routing.config import (
    DEFAULT_RESEARCH_SEARCH_QUERIES,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
)

STYLE_SELECTION_TAG = "style_selection"


def _tool_list(*, search: bool, knowledge: bool) -> str:
    return "、".join(
        part
        for part, on in (
            ("联网检索 `web_search` / `web_fetch`", search),
            ("知识库 `kb_search` / `kb_read`", knowledge),
        )
        if on
    )


def _acquire_section(
    *,
    emits_queries: bool,
    emits_entries: bool,
    native_search: bool,
    max_search_queries: int,
    max_requested_entries: int,
    agentic: bool = False,
    task_update_feedback: bool = False,
) -> str:
    """The "how to get what you lack" section, per grant and strategy."""

    parts: List[str] = []
    if agentic and (emits_queries or emits_entries):
        return _load_prompt_template(
            "fragment_prepare_agentic_acquire_v1.md",
            tools=_tool_list(search=emits_queries, knowledge=emits_entries),
        )
    if native_search:
        parts.append(
            renumber_rules(
                _load_prompt_template(
                    "fragment_native_search_research_v1.md",
                    shared_rules=search_rules_shared(
                        prepare=True, task_update_feedback=task_update_feedback, knowledge=emits_entries
                    ),
                ).strip()
            )
        )
    request_blocks: List[str] = []
    if emits_entries:
        request_blocks.append(_r1_entry_blocks(max_requested_entries, keep_source_block="`<preinjected_entries>`"))
    if emits_queries:
        request_blocks.append(
            _search_queries_rules(max_search_queries, knowledge_enabled=emits_entries)
        )
    if request_blocks:
        what = "、".join(
            part
            for part, on in (("知识库词条", emits_entries), ("联网搜索", emits_queries))
            if on
        )
        parts.append(
            _load_prompt_template(
                "fragment_prepare_request_channel_v1.md",
                what=what,
                request_blocks="\n".join(request_blocks),
            ).rstrip()
        )
    if not parts:
        parts.append(_load_prompt_template("fragment_prepare_no_evidence_v1.md").strip())
    return "\n\n".join(parts) + "\n"


#: Where the `<style_selection>` block goes, per shape of the asking task
#: (auto-style-gaps plan A2: "按授权拼"). ``request``: a batched window whose
#: request turn can also ask for search / entries; ``style_only``: one whose
#: request turn exists only to pick; ``agentic``: a tool-session window, which
#: picks in its delivery.
STYLE_OUTPUT_CLAUSES: Dict[str, str] = {
    "prepare": "fragment_style_output_prepare_v1.md",
    "request": "fragment_style_output_request_v1.md",
    "style_only": "fragment_style_output_style_only_v1.md",
    "agentic": "fragment_style_output_agentic_v1.md",
}
#: ``optional``: an agentic preparation task may read a candidate in full;
#: ``required``: an agentic window must read the one it picks before it
#: translates, and the harness checks that it did (A1 step 3).
STYLE_READ_CLAUSES: Dict[str, str] = {
    "optional": "fragment_style_read_optional_v1.md",
    "required": "fragment_style_read_required_v1.md",
}


def render_style_selection_section(
    catalog: str,
    *,
    extra_style: str = "",
    shape: str = "prepare",
    read: str = "",
    correct_only: bool = False,
) -> str:
    """The ``--style auto`` section: candidates as material, one forced pick, a JSON block. A ``correct_only``
    task (a Chinese source) has no factory default and no `--extra-style` to mention.

    The pick is part of the delivery contract (auto-style-gaps plan 「已裁定」 8):
    with `default_style` among the candidates "nothing fits" is answered by
    choosing it; without one, by an explicit empty pick.
    """

    if not catalog.strip():
        return ""
    mode = task_mode_texts(correct_only)
    words = prompt_phrases("style_pick")
    if correct_only:
        extra_style = ""
    offers_default = catalog_offers_default(catalog)
    # An agentic window picks and translates in one delivery, so a pick of "none" has to find the factory
    # default in front of it already -- conditionally, or it would be in force before the pick is made.
    # Everywhere else the harness brings the settled style in after the pick.
    factory_clause = ""
    if read == "required" and not offers_default and not correct_only:
        factory_clause = "\n" + _load_prompt_template(
            "fragment_style_no_match_factory_v1.md", style_name=FACTORY_STYLE_NAME, factory=factory_style_text(),
            # it sits in the pick section, not under the system's extra-style line that outranks styles
            extra_clause=words["factory_extra_first"] if extra_style.strip() else "",
        ).strip() + "\n"
    return "\n" + _load_prompt_template(
        "fragment_prepare_style_selection_v1.md",
        **mode,
        output_clause=_load_prompt_template(STYLE_OUTPUT_CLAUSES[shape]).strip(),
        # Who brings the picked rules in front of the translation: the
        # harness everywhere except the agentic window, which reads them.
        after_pick_clause=_load_prompt_template(
            "fragment_style_after_pick_read_v1.md"
            if read == "required"
            else "fragment_style_after_pick_injected_v1.md",
            **mode,
        ).strip(),
        no_match_clause=_load_prompt_template(
            "fragment_style_no_match_default_v1.md"
            if offers_default
            else "fragment_style_no_match_none_v1.md",
            **mode,
        ).strip(),
        extra_style_clause=(
            Template(words["extra_style"]).substitute(extra_style=extra_style.strip()) if extra_style.strip() else ""
        ),
        read_clause=(
            _load_prompt_template(STYLE_READ_CLAUSES[read], **mode).strip() + "\n" if read else ""
        ),
        catalog=catalog.strip(),
    ).rstrip() + "\n" + factory_clause


def build_prepare_task_messages(
    *,
    transcript: str,
    extra_info: str = "",
    note_url_extracts: str = "",
    knowledge_indexes: Mapping[str, str] | None = None,
    preinjected_entries: str = "",
    max_search_queries: int = DEFAULT_RESEARCH_SEARCH_QUERIES,
    emits_queries: bool = True,
    emits_entries: bool = True,
    native_search: bool = False,
    max_requested_entries: int = KB_WINDOW_TOTAL_ENTRIES,
    collect_task_feedback: bool = False,
    emits_keep: bool = True,
    agentic: bool = False,
    style_catalog: str = "",
    extra_style: str = "",
    speakers: bool = False,
    correct_only: bool = False,
    earlier_chunks: str = "",
) -> List[Dict[str, str]]:
    """The task's first message: the whole contract, plus the request channel.

    ``emits_queries`` is a local search agent behind the harness
    (``retrieval=local``); ``native_search`` is the model searching inside the
    call (``retrieval=native``); ``emits_entries`` is a readable knowledge
    base. ``emits_keep`` asks the delivery for ``<keep_entries>`` -- only
    meaningful where a later round can request entries again. ``agentic``
    swaps the request channel for the executor's own tools (stage C): the
    same grant, fetched by the model inside its tool session, delivered in
    one reply. ``style_catalog`` (non-empty under ``--style auto``) adds the
    candidate list and asks the delivery for a ``<style_selection>`` block.
    ``speakers`` (only with the speaker judge on) also asks, lightly, for
    the people who speak in the material: the judge's candidates. Off, the
    prompt is what it always was. ``earlier_chunks`` (a long file's later
    chunk: the names and summaries the chunks before settled,
    `general_context.carry_text`) adds that block, the duty to reuse it and the
    `name_disputes` field.
    """

    can_request = (emits_queries or emits_entries) and not agentic
    phrases = prompt_phrases("prepare")
    # Agentic: the index is a required block of the manifest (the submit
    # gate checks that read), so it is not inlined as well -- the same as
    # the agentic window (owner 2026-10-03, cold read Q2). Where that block
    # is depends on the driver (opening, file, tool), so the prompt names
    # the block, not a way to fetch it (cold read P449).
    input_sources = [
        phrases[key]
        for key, on in (
            ("input_kb_agentic" if agentic else "input_kb", emits_entries),
            ("input_url_extracts", bool(note_url_extracts.strip())),
        )
        if on
    ]
    mode = task_mode_texts(correct_only)
    # Each background line belongs to one axis, so a disabled half removes its own line together with its
    # input section -- the injection-reality principle (docs/llm_prompts.md) applied to a task.
    background_lines = prompt_phrases("prepare_background")
    background = [
        Template(background_lines[key]).substitute(mode)
        for key, on in (
            ("asr", True),
            ("window_marks", True),
            ("line_format", True),
            ("downstream", True),
            ("anchors", True),
            ("knowledge", emits_entries),
            ("local_search", emits_queries and not agentic),
        )
        if on
    ]
    uncertainty_sources = "、".join(
        part
        for part, on in (
            ("知识库", emits_entries),
            ("检索结果", emits_queries or native_search),
        )
        if on
    )
    style_section = render_style_selection_section(
        style_catalog, extra_style=extra_style, read="optional" if agentic else "", correct_only=correct_only
    )
    # No search and no knowledge base: nothing can back a name or fill `sources`.
    no_evidence = not (emits_queries or emits_entries or native_search)
    sources_clause = "；".join(
        phrases[key] for key, on in (("sources_web", not no_evidence), ("sources_kb", emits_entries)) if on
    )
    system = _load_prompt_template(
        "prepare_task_v1.md",
        **{key: value for key, value in mode.items() if key.startswith("prep_") and key != "prep_evidence_names"},
        prep_evidence_names="" if no_evidence else mode["prep_evidence_names"],
        input_sources=("、" + "、".join(input_sources)) if input_sources else "",
        background_items=_numbered(background),
        acquire_section=_acquire_section(
            emits_queries=emits_queries,
            emits_entries=emits_entries,
            native_search=native_search,
            max_search_queries=max_search_queries,
            max_requested_entries=max_requested_entries,
            agentic=agentic,
            task_update_feedback=collect_task_feedback,
        ),
        style_section=style_section,
        uncertainty_sources=uncertainty_sources + ("都" if "、" in uncertainty_sources else ""),
        sources_clause=(sources_clause + "。") if sources_clause else "",
        name_conflict_clause=_load_prompt_template(
            "fragment_prepare_name_conflict_v1.md"
            if emits_entries
            else "fragment_prepare_name_conflict_no_kb_v1.md"
        ).strip(),
        sources_example="[]" if no_evidence else phrases["sources_example"],
        speakers_duty=(
            "\n" + _load_prompt_template(
                "fragment_prepare_speakers_duty_v1.md", strict=True,
                # Only a task that can look things up needs telling not to for this duty.
                no_search=_load_prompt_template("fragment_prepare_speakers_no_search_v1.md").strip()
                if (emits_queries or emits_entries or native_search) else "",
            ).rstrip("\n") if speakers else ""
        ),
        speakers_field=(
            "\n" + _load_prompt_template("fragment_prepare_speakers_field_v1.md").rstrip("\n") if speakers else ""
        ),
        earlier_chunks_duty=(
            "\n" + _load_prompt_template("fragment_prepare_earlier_chunks_duty_v1.md") if earlier_chunks else ""
        ),
        name_disputes_field=(
            _load_prompt_template("fragment_prepare_name_disputes_field_v1.md") if earlier_chunks else ""
        ),
        delivery_lead=(phrases["delivery_lead"] if can_request else ""),
        keep_entries_block=(
            _load_prompt_template(
                "fragment_research_keep_entries_v1.md",
                max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
                # Where the keys it may keep came from, as wide as the
                # delivery check (`stages/prepare_task.py`): batched, the fill
                # turn's block plus the preinjected one; agentic, the
                # preinjected entries plus whatever it read with `kb_read` --
                # there is no fill turn, so `<knowledge_entries>` would name a
                # block that does not exist.
                key_source=phrases["keep_source_agentic" if agentic else "keep_source"],
                ignored_clause=phrases["keep_ignored_agentic" if agentic else "keep_ignored"],
            ).rstrip()
            + "\n"
            if emits_keep and emits_entries
            else ""
        ),
        task_update_feedback_block=(
            _research_task_feedback_block(local_source_ids=False)
            if collect_task_feedback
            else ""
        ),
    )
    duties = [
        phrases["duty_deliver"],
        *((phrases["duty_request_entries"],) if emits_entries and not agentic else ()),
        *((phrases["duty_request_search"],) if emits_queries and not agentic else ()),
        *((phrases["duty_tools"],) if agentic and (emits_queries or emits_entries) else ()),
        *((mode["prep_style_duty"],) if style_section else ()),
    ]
    closing = phrases["closing_may_request" if can_request else "closing_deliver"]
    closing += _trailing_blocks_reminder(
        keep=emits_keep and emits_entries,
        style_selection=bool(style_section),
        task_update_feedback=collect_task_feedback,
        lead=phrases["delivery_lead"] if can_request else "",
    )
    knowledge_inputs = (
        _load_prompt_template(
            "fragment_knowledge_inputs_v1.md",
            index_block=(
                ""
                if agentic
                else "知识库索引：\n" + render_knowledge_indexes(knowledge_indexes) + "\n\n"
            ),
            preinjected_entries=preinjected_entries.strip() or "（无）",
        )
        if emits_entries
        else "\n"
    )
    user = _load_prompt_template(
        "prepare_task_user_v1.md",
        task_summary="、".join(duties),
        extra_info=extra_info or "（无）",
        note_url_extracts_block=(
            _load_prompt_template(
                "fragment_prepare_url_extracts_v1.md", note_url_extracts=note_url_extracts.strip()
            )
            if note_url_extracts.strip()
            else ""
        ),
        knowledge_inputs=knowledge_inputs,
        earlier_chunks_block=(
            _load_prompt_template("fragment_prepare_earlier_chunks_v1.md", strict=True, earlier_chunks=earlier_chunks)
            if earlier_chunks
            else ""
        ),
        transcript=transcript,
        closing_reminder=closing,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _trailing_blocks_reminder(
    *, keep: bool, style_selection: bool, task_update_feedback: bool, lead: str = ""
) -> str:
    """The blocks after `<context_pack>`, named once in delivery order (cold read P478)."""

    tags = [
        tag
        for tag, on in (
            ("keep_entries", keep),
            (STYLE_SELECTION_TAG, style_selection),
            ("task_update_feedback", task_update_feedback),
        )
        if on
    ]
    if not tags:
        return ""
    phrases = prompt_phrases("prepare")
    return Template(phrases["trailing_blocks"]).substitute(
        lead=lead, order=phrases["in_order"] if len(tags) > 1 else "", tags="、".join(f"`<{tag}>`" for tag in tags)
    )


def build_prepare_results_message(
    *,
    entry_details_text: str = "",
    search_results: str = "",
    requested_entries: bool = False,
    requested_search: bool = False,
    collect_task_feedback: bool = False,
    emits_keep: bool = True,
    style_selection: bool = False,
    knowledge: bool,
) -> Dict[str, str]:
    """The fill turn: what the batch produced, and the order to deliver.
    ``knowledge``: the task has a readable knowledge base (the search rules then say how to weigh a
    search result against an entry).

    A channel the model asked for that came back empty is said so explicitly
    -- an execution failure or a miss is a result, not silence (plan §2.1).
    """

    empty = prompt_phrases("fill")
    knowledge_input = (
        "\n" + _load_prompt_template(
            "fragment_prepare_entries_result_v1.md", strict=True, entries=entry_details_text.strip() or empty["no_entries"]
        )
        if requested_entries
        else ""
    )
    search_input = (
        "\n" + _load_prompt_template(
            "fragment_prepare_search_result_v1.md", strict=True, results=search_results.strip() or empty["no_search_results"]
        )
        if requested_search
        else ""
    )
    usage = (
        search_results_usage(prepare=True, task_update_feedback=collect_task_feedback, knowledge=knowledge)
        if requested_search and search_results.strip()
        else ""
    )
    content = _load_prompt_template(
        "prepare_task_results_v1.md",
        knowledge_entries_input=knowledge_input,
        search_results_input=search_input,
        usage_rules=usage,
        trailing_blocks_reminder=_trailing_blocks_reminder(
            keep=emits_keep and requested_entries,
            style_selection=style_selection,
            task_update_feedback=collect_task_feedback,
        ),
    )
    return {"role": "user", "content": content}
