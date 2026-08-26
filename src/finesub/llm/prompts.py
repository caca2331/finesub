"""Chinese prompt templates for subtitle correction and translation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Mapping, Sequence

from .chunking import (
    SubtitleSegment,
    SubtitleWindow,
    render_window_preceding_as_csv,
    render_window_segments_as_csv,
)
from .routing.config import (
    CapabilityTier,
    DEFAULT_RESEARCH_SEARCH_QUERIES,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    MAX_WINDOW_SEARCH_QUERIES,
    WINDOW_CONTEXT_MAX_TOKENS,
)
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile
from .token_budget import TokenCounter, default_token_counter
from .token_truncate import truncate_to_token_window
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


def _search_queries_rules(max_queries: int, *, knowledge_enabled: bool = True) -> str:
    # The knowledge-owned tail of the "don't waste queries" rule leaves with
    # the knowledge switch: it argues against entries a knowledge-off round
    # will never be shown (docs/llm_prompts.md, injection reality).
    return _load_prompt_template(
        "fragment_search_queries_output_v1.md",
        max_queries=max_queries,
        query_style=_query_style_rules(),
        knowledge_query_note=(
            "已注入的知识库条目足以回答的主题也不要花 query（尤其主播本人相关），"
            "只为其增量动态、时效信息或条目标注「待定」的点发 query。"
            "index 中有但尚未注入的条目：请求词条与发 query **并行**进行，"
            "不要等词条注入再决定；若 index 简介已表明词条大概率覆盖该主题，"
            "可降低对应 query 的优先级或数量。"
            if knowledge_enabled
            else ""
        ),
    ).strip()


def _search_contract_rules(max_queries: int, *, knowledge_enabled: bool = True) -> str:
    """Multi-round variant of the search emission rules (contract + round 0)."""

    return _load_prompt_template(
        "fragment_search_contract_output_v1.md",
        max_queries=max_queries,
        query_style=_query_style_rules(),
        knowledge_query_note=(
            "已注入的知识库条目足以回答的主题也不要花 query（尤其主播本人相关），"
            "只为其增量动态、时效信息或条目标注「待定」的点立 fact/发 query。"
            "index 中有但尚未注入的条目：请求词条与发 query **并行**，"
            "不要等词条注入再决定；若 index 简介已表明词条大概率覆盖该主题，"
            "可调低对应 fact 的 priority。"
            if knowledge_enabled
            else ""
        ),
    ).strip()


def _search_results_usage() -> str:
    return _load_prompt_template("fragment_search_results_usage_v1.md").strip()


def _evidence_pack_usage() -> str:
    """Multi-round variant of the search results usage rules."""

    return _load_prompt_template("fragment_evidence_pack_usage_v1.md").strip()


def _feedback_schema(*, local_source_ids: bool) -> str:
    source_ids_rule = (
        "`source_ids`：支撑该线索的本窗口局部字幕序号（字符串数组，只能用 "
        "`<asr_result>` 中从 1 开始的正整数），可省略；harness 会映射回稳定源字幕行号。"
        if local_source_ids
        else "`source_ids`：支撑该线索的稳定源字幕序号（字符串数组，可引用 transcript "
        "中的源序号），可省略。"
    )
    return _load_prompt_template(
        "fragment_task_feedback_schema_v3.md",
        source_ids_rule=source_ids_rule,
    ).strip()


def _research_task_feedback_block(*, local_source_ids: bool) -> str:
    """Feedback-collection addendum for the research final round / fast round 1."""

    return "\n" + _load_prompt_template(
        "research_task_feedback_v1.md",
        feedback_schema=_feedback_schema(local_source_ids=local_source_ids),
    ).strip() + "\n"


# Appended to the round's closing reminder when feedback collection is on.
TASK_FEEDBACK_REMINDER = "，随后按 system 要求输出一个 `<task_update_feedback>` 块"


@dataclass(frozen=True)
class WindowContextNote:
    window_id: str
    first_source_id: str
    last_source_id: str
    context: str

    def to_dict(self) -> dict[str, str]:
        return {
            "window_id": self.window_id,
            "first_source_id": self.first_source_id,
            "last_source_id": self.last_source_id,
            "context": self.context,
        }


@dataclass(frozen=True)
class ContextPack:
    """Research output injected into correction windows.

    ``general_context`` covers every window.  Window notes carry their original
    source-id interval so a saved research result can be reused after correction
    windows are split or otherwise re-shaped.

    ``unbound_window_contexts`` is the *transient* shape the model produces: r2
    names windows by chunk id and knows nothing about source ids, so
    :func:`parse_round2_output` lands here and the research stage calls
    :meth:`bind_window_ranges` before writing the artifact.  Unbound notes are
    never injected -- a persisted context that still has them is a stale
    artifact, and its loader rejects it rather than serving half a pack.
    """

    general_context: Mapping[str, Any] = field(default_factory=dict)
    window_contexts: tuple[WindowContextNote, ...] = ()
    unbound_window_contexts: Mapping[str, str] = field(default_factory=dict)
    source_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.window_contexts, Mapping):
            raise TypeError(
                "ContextPack.window_contexts takes WindowContextNote records; "
                "an id->text mapping is the model's shape, so build it with "
                "ContextPack.from_dict and bind ranges before injecting."
            )

    def general_prompt_text(self) -> str:
        if not self.general_context:
            return "{}"
        return json.dumps(dict(self.general_context), ensure_ascii=False, indent=2)

    @property
    def has_unbound_window_contexts(self) -> bool:
        return bool(self.unbound_window_contexts)

    def with_source_order(self, source_ids: Sequence[str]) -> "ContextPack":
        return ContextPack(
            general_context=self.general_context,
            window_contexts=self.window_contexts,
            unbound_window_contexts=self.unbound_window_contexts,
            source_order=tuple(str(source_id) for source_id in source_ids),
        )

    def bind_window_ranges(
        self,
        windows: Sequence[SubtitleWindow],
        *,
        report_sink: dict[str, Any] | None = None,
    ) -> "ContextPack":
        """Attach research-window source ranges to model output keyed by id.

        A note whose window id is not in the plan cannot be placed and is
        dropped. ``report_sink`` receives the ids that were dropped: if r2
        gets the ids systematically wrong, every note disappears and the
        artifact still looks well-formed -- bound notes, none of them, which
        the loader's unbound check cannot see. The caller is expected to
        surface a non-empty report.
        """

        by_id = {window.chunk_id: window for window in windows}
        notes = list(self.window_contexts)
        unplaceable: list[str] = []
        for window_id, context in self.unbound_window_contexts.items():
            window = by_id.get(window_id)
            body = (
                window.segments[len(window.overlap_segments) :]
                if window is not None
                else []
            )
            if not body:
                unplaceable.append(str(window_id))
                continue
            notes.append(
                WindowContextNote(
                    window_id=window_id,
                    first_source_id=body[0].id,
                    last_source_id=body[-1].id,
                    context=context,
                )
            )
        if report_sink is not None:
            report_sink["bound"] = len(notes)
            report_sink["unplaceable_window_ids"] = unplaceable
            report_sink["plan_window_ids"] = [window.chunk_id for window in windows]
        source_order = [
            segment.id
            for window in windows
            for segment in window.segments[len(window.overlap_segments) :]
        ]
        return ContextPack(
            general_context=self.general_context,
            window_contexts=tuple(notes),
            source_order=tuple(dict.fromkeys(source_order)),
        )

    def window_context_for(
        self,
        window: SubtitleWindow,
        *,
        counter: TokenCounter | None = None,
        max_tokens: int = WINDOW_CONTEXT_MAX_TOKENS,
        report_sink: dict[str, Any] | None = None,
    ) -> str:
        body = list(window.segments[len(window.overlap_segments) :])
        return self.window_context_for_source_ids(
            [segment.id for segment in body],
            chunk_id=window.chunk_id,
            counter=counter,
            max_tokens=max_tokens,
            report_sink=report_sink,
        )

    def window_context_for_source_ids(
        self,
        source_ids: Sequence[str],
        *,
        chunk_id: str = "",
        counter: TokenCounter | None = None,
        max_tokens: int = WINDOW_CONTEXT_MAX_TOKENS,
        report_sink: dict[str, Any] | None = None,
    ) -> str:
        body_ids = list(source_ids)
        if not body_ids or not self.window_contexts:
            return ""
        if not self.source_order:
            # Without the full source order a note that *contains* this window
            # has endpoints we cannot place, so the containing-note clause --
            # the one every split window depends on -- would silently return
            # nothing. Every consumer binds the current segment list first.
            raise ValueError(
                "ContextPack.window_context_for needs the source order; call "
                "with_source_order(current segment ids) after loading."
            )
        positions = {
            source_id: index for index, source_id in enumerate(self.source_order)
        }
        if any(source_id not in positions for source_id in body_ids):
            return ""
        first, last = positions[body_ids[0]], positions[body_ids[-1]]
        ranged: list[tuple[int, int, WindowContextNote]] = []
        for note in self.window_contexts:
            if (
                note.first_source_id not in positions
                or note.last_source_id not in positions
            ):
                continue
            note_first = positions[note.first_source_id]
            note_last = positions[note.last_source_id]
            if note_first <= note_last:
                ranged.append((note_first, note_last, note))

        contained = sorted(
            (item for item in ranged if first <= item[0] and item[1] <= last),
            key=lambda item: (item[0], item[1]),
        )
        if contained:
            selected = contained
        else:
            containers = [
                item for item in ranged if item[0] <= first and last <= item[1]
            ]
            selected = (
                [min(containers, key=lambda item: (item[1] - item[0], item[0]))]
                if containers
                else []
            )
        text = "\n\n".join(
            item[2].context.strip()
            for item in selected
            if item[2].context.strip()
        )
        if not text:
            return ""
        token_counter = counter or default_token_counter()
        result = truncate_to_token_window(text, max_tokens, token_counter.count_text)
        if report_sink is not None and result.text != text:
            report_sink.update(
                {
                    "chunk_id": chunk_id,
                    "selected_window_ids": [item[2].window_id for item in selected],
                    "original_tokens": token_counter.count_text(text),
                    "kept_tokens": result.tokens,
                    "limit": max_tokens,
                }
            )
        return result.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "general_context": dict(self.general_context),
            "window_contexts": [
                *[note.to_dict() for note in self.window_contexts],
                *[
                    {"window_id": window_id, "context": context}
                    for window_id, context in self.unbound_window_contexts.items()
                ],
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContextPack":
        general = data.get("general_context") or {}
        if not isinstance(general, Mapping):
            general = {"global_summary": str(general)}
        # The model emits ``[{window_id, context}]`` (research_round2_v2.md); the
        # harness adds the source range before persisting. Anything without a
        # range therefore lands unbound and is never injected.
        raw_windows = data.get("window_contexts") or []
        if isinstance(raw_windows, Mapping):
            # The pre-interval on-disk shape. Refused rather than read as "no
            # notes", so the caller re-runs research instead of injecting a pack
            # that quietly lost every window-specific note it once had.
            raise ValueError(
                "window_contexts is an id->text mapping; that artifact predates "
                "source-interval addressing and cannot be placed on any window"
            )
        window_contexts: list[WindowContextNote] = []
        unbound: dict[str, str] = {}
        if isinstance(raw_windows, Sequence) and not isinstance(raw_windows, (str, bytes)):
            for item in raw_windows:
                if isinstance(item, Mapping) and item.get("window_id"):
                    window_id = str(item["window_id"])
                    first = str(item.get("first_source_id") or "")
                    last = str(item.get("last_source_id") or "")
                    if first and last:
                        window_contexts.append(
                            WindowContextNote(
                                window_id,
                                first,
                                last,
                                str(item.get("context", "")),
                            )
                        )
                    else:
                        unbound[window_id] = str(item.get("context", ""))
        return cls(
            general_context=dict(general),
            window_contexts=tuple(window_contexts),
            unbound_window_contexts=unbound,
        )


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


def _numbered(items: Sequence[str]) -> str:
    """Render an ordered list whose numbering follows the items actually kept.

    R1's background/duty lists are assembled per axis (docs/llm_harness_behavior.md), so the
    numbers cannot live in the template -- a disabled half would leave a hole.
    """

    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


# --- R1 axis fragments -------------------------------------------------------
# Each entry is owned by exactly one axis, so a disabled half removes its own
# background line, its own duty, its own output block and its own input section
# together -- the injection-reality principle (docs/llm_prompts.md) applied to a round.

_R1_BACKGROUND_COMMON = (
    "ASR 文本来自 Whisper 识别，可能存在误听：专有名词可能被识别成音似的假名、汉字、"
    "英文或另一种语言。判断「这段内容在说什么」时要考虑这种可能。",
    "文本中的 `--- window N ---` 标记是后续纠错处理的窗口边界，仅用于定位，"
    "不需要在本轮输出中引用。",
    "每行格式是 `源序号|文本`。",
)
_R1_BACKGROUND_KNOWLEDGE = (
    "本地知识库收集的是公开网络中很少存在、难以进入 LLM 语料的知识；"
    "索引里的条目可能正是理解本段内容的关键。",
)
_R1_BACKGROUND_SEARCH = (
    "你自己没有联网搜索能力。你提出的搜索 query 会由本地搜索代理执行，"
    "结果供第二轮调查使用。",
)

_R1_DUTY_ENTRIES = (
    "对照两份索引，找出与本段内容相关、尚未预注入且需要完整详情的条目"
    "（主播本人、提到的其他主播、游戏、梗、事件等），列入 `<requested_entries>`。"
    "词条 key = index 行首主 key = 条目 Markdown 文件的一级标题（`# 源语言本名`）；"
    "每行写主 key 或别名，按重要性从高到低排列。只请求强相关条目，"
    "不要用边缘请求挤占共享额度。",
    "检查 `<preinjected_entries>`，把后续仍需使用的已注入条目列入 `<keep_entries>`；"
    "只能引用本轮实际可见的预注入词条，不要把未注入词条写进 keep。",
)
_R1_DUTY_SEARCH = (
    "提出联网搜索 query，覆盖所有你「感兴趣」、值得查证的内容：游戏剧情/系统/角色名、"
    "近期事件、社区语境、直播来源信息等。优先覆盖理解内容所必需、"
    "且你自身知识可能不足或过时的主题。",
)

def r1_request_cap(*, downstream_can_request: bool) -> int:
    """How many entries R1 may request.

    ``KB_WINDOW_NEW_REQUEST_MAX_ENTRIES`` was sized for a *per-window increment*
    -- it leaves room for the next window to ask for more. When no later round
    can request (no per-window query round; docs/llm_harness_behavior.md), R1 is the session's only
    pick and that reserve is pure loss, so it gets the whole shared budget.
    """

    return (
        KB_WINDOW_NEW_REQUEST_MAX_ENTRIES
        if downstream_can_request
        else KB_WINDOW_TOTAL_ENTRIES
    )


def _r1_notes_block() -> str:
    return _load_prompt_template("fragment_research_analysis_notes_v1.md").strip()


def _r1_entry_blocks(max_requested_entries: int) -> str:
    return _load_prompt_template(
        "fragment_research_entry_blocks_v1.md",
        max_requested_entries=max_requested_entries,
        max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
        max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
    ).strip()


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
    emits_queries: bool = True,
    emits_entries: bool = True,
    emits_notes: bool = True,
    max_requested_entries: int = KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
) -> List[Dict[str, str]]:
    """Round 1 research prompt, assembled from the axis halves it actually runs.

    ``emits_queries`` is ``retrieval=local`` (the local search agent executes
    what this round asks for), ``emits_entries`` is a readable knowledge base,
    and ``emits_notes`` is "round 2 will run and read them". A caller that
    passes all three ``False`` should not be running R1 at all.

    ``use_search_contract`` swaps the single-round query emission fragment for
    the multi-round variant (Research Contract + round-0 queries); everything
    else stays identical, which is the pluggable-prompt seam.
    ``preinjected_entries`` carries the budget-rendered knowledge entries the
    harness matched against the user note's keys/aliases.
    """

    input_sources = []
    if emits_entries:
        input_sources.append(
            "本地知识库的两份索引（主播 index 和 common index）"
            "，以及 harness 根据用户备注关键词预注入的知识库条目全文（可能为空）"
        )
    background = [
        *_R1_BACKGROUND_COMMON,
        *(_R1_BACKGROUND_KNOWLEDGE if emits_entries else ()),
        *(_R1_BACKGROUND_SEARCH if emits_queries else ()),
    ]
    analysis_purpose = "、".join(
        part
        for part, on in (("条目挑选", emits_entries), ("搜索 query", emits_queries))
        if on
    )
    duties = [
        "中轻量分析：先快速把握整段内容——主题/游戏/主播的初步判断、剧情或事件线索、"
        "关键疑点和可能的 ASR 误听风险。"
        + (
            f"分析是为了提高后续{analysis_purpose}的质量，不要逐句展开。"
            if analysis_purpose
            else "不要逐句展开。"
        )
        + (
            "把对第二轮调查有用的要点写入 `<analysis_notes>` 块。"
            if emits_notes
            else "分析只写在 `<reasoning>` 里，本轮没有下游调查轮会读它。"
        ),
        *(_R1_DUTY_ENTRIES if emits_entries else ()),
        *(_R1_DUTY_SEARCH if emits_queries else ()),
    ]
    output_blocks = [
        *((_r1_notes_block(),) if emits_notes else ()),
        *((_r1_entry_blocks(max_requested_entries),) if emits_entries else ()),
        *(
            (
                _search_contract_rules(
                    max_search_queries, knowledge_enabled=emits_entries
                )
                if use_search_contract
                else _search_queries_rules(
                    max_search_queries, knowledge_enabled=emits_entries
                ),
            )
            if emits_queries
            else ()
        ),
    ]
    system = _load_prompt_template(
        "research_round1_v2.md",
        input_sources=("、" + "".join(input_sources)) if input_sources else "",
        background_items=_numbered(background),
        duty_items=_numbered(duties),
        reasoning_clause=reasoning_clause(),
        output_blocks="\n".join(output_blocks),
    )

    block_names = [
        *(("`<analysis_notes>`",) if emits_notes else ()),
        *(("`<requested_entries>`", "`<keep_entries>`") if emits_entries else ()),
        *(("搜索相关标签块",) if emits_queries else ()),
    ]
    task_summary = "、".join(
        part
        for part, on in (
            ("先做中轻量分析", True),
            ("request 尚未注入的词条、keep 仍需使用的预注入词条", emits_entries),
            ("提出值得联网查证的搜索 query", emits_queries),
        )
        if on
    )
    knowledge_inputs = (
        _load_prompt_template(
            "fragment_research_knowledge_inputs_v1.md",
            streamer_index=streamer_index or "（空）",
            common_index=common_index or "（空）",
            preinjected_entries=preinjected_entries.strip() or "（无）",
        )
        if emits_entries
        else "\n"
    )
    user = _load_prompt_template(
        "research_round1_user_v2.md",
        task_summary=task_summary,
        output_block_list="、".join(block_names),
        # Only worth saying where something downstream reads the notes and can
        # act on the hedge -- with no round 2 there is no such reader.
        unverified_clause=(
            "本轮没有联网结果，未经证实的判断必须标注「待定」；" if emits_notes else ""
        ),
        extra_info=extra_info or "（无）",
        note_url_extracts=note_url_extracts.strip() or "（无）",
        knowledge_inputs=knowledge_inputs,
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
    native_search: bool = False,
    emits_keep: bool = True,
) -> List[Dict[str, str]]:
    """Round 2 research prompt.

    ``use_evidence_pack`` swaps the raw-search-results usage fragment for the
    evidence-pack variant; ``search_results`` then carries the rendered
    evidence pack text in the same prompt slot. ``entry_details_text`` is the
    budget-rendered knowledge-entry block from round 1's requests.
    ``collect_task_feedback`` additionally asks for a trailing
    ``<task_update_feedback>`` block (schema v3).

    ``native_search`` is ``retrieval=native``: nothing was fetched for this
    round, the model searches inside the call, and the ``<search_results>``
    input section disappears along with its usage rules. ``emits_keep`` is off
    whenever no later round can request entries -- pruning there would lose
    entries permanently, so the harness transfers the full set instead of
    asking (docs/llm_harness_behavior.md).
    """

    has_entries = bool(entry_details_text.strip())
    system = _load_prompt_template(
        "research_round2_v2.md",
        input_sources="".join(
            part
            for part, on in (
                ("、第一轮调查索取的知识库条目详情", has_entries),
                ("、本地搜索代理对第一轮提出的 query 返回的搜索结果", not native_search),
            )
            if on
        ),
        reasoning_clause=reasoning_clause(),
        retrieval_usage=(
            _load_prompt_template("fragment_native_search_research_v1.md").strip()
            if native_search
            else (_evidence_pack_usage() if use_evidence_pack else _search_results_usage())
        ),
        uncertainty_sources=(
            "、".join(
                part
                for part, on in (("知识库", has_entries), ("检索结果", True))
                if on
            )
        ),
        keep_entries_block=(
            _load_prompt_template(
                "fragment_research_keep_entries_v1.md",
                max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
            ).rstrip()
            + "\n"
            if emits_keep
            else ""
        ),
        task_update_feedback_block=(
            _research_task_feedback_block(local_source_ids=False)
            if collect_task_feedback
            else ""
        ),
    )
    if isinstance(search_results, list):
        search_results = "\n".join(search_results)
    user = _load_prompt_template(
        "research_round2_user_v2.md",
        extra_info=extra_info or "（无）",
        round1_cross_check=(
            "、".join(
                part
                for part, on in (
                    ("知识库", has_entries),
                    ("你自己检索到的资料", native_search),
                    ("搜索结果", not native_search),
                )
                if on
            )
        ),
        round1_notes=round1_notes.strip() or "（无）",
        knowledge_entries_input=(
            "\n第一轮索取的知识库条目详情：\n"
            f"<knowledge_entries>\n{entry_details_text.strip()}\n</knowledge_entries>\n"
            if has_entries
            else ""
        ),
        search_results_input=(
            ""
            if native_search
            else (
                "\n本地搜索代理返回的搜索结果（按第一轮提出的 query 分组，可能为空）：\n"
                f"<search_results>\n{search_results.strip() or '（无）'}\n</search_results>\n"
            )
        ),
        transcript=transcript,
        keep_entries_reminder=(
            "，再输出 `<keep_entries>` 块（可为空）" if emits_keep else ""
        ),
        search_closing=(
            "检索完成后不要再发起新的搜索"
            if native_search
            else "不能再发起搜索"
        ),
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
        reasoning_clause=reasoning_clause(),
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
        reasoning_clause=reasoning_clause(),
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

    Index injection and the ``<requested_entries>`` rules share one predicate
    (docs/llm_prompts.md): with no index to read, asking the model to pick entries off it
    is asking for nothing. Passing both indices empty therefore also drops the
    request rules from the system prompt.
    """

    pack = context_pack or ContextPack()
    knowledge_enabled = bool(streamer_index.strip() or common_index.strip())
    remaining_entries = max(
        0,
        min(
            KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
            KB_WINDOW_TOTAL_ENTRIES - max(0, int(carried_entry_count)),
        ),
    )
    system = compose_correction_query_system(
        profile,
        search_queries_rules=_search_queries_rules(
            max_search_queries, knowledge_enabled=knowledge_enabled
        ),
        max_entries=remaining_entries,
        total_entries=KB_WINDOW_TOTAL_ENTRIES,
        knowledge_enabled=knowledge_enabled,
    )
    user = ensure_csv_block_headers(_load_prompt_template(
        "correction_query_user_v1.md",
        general_context_json=pack.general_prompt_text(),
        window_context=pack.window_context_for(window) or "（无）",
        previous_advice=previous_advice.strip() or "（无）",
        # The knowledge-owned *input sections* follow the same predicate as the
        # request rules: with the base off, the round must not be shown empty
        # index/carried blocks it was given no rules for (injection reality).
        knowledge_index_block=(
            _load_prompt_template(
                "fragment_query_index_input_v1.md",
                streamer_index=streamer_index or "（空）",
                common_index=common_index or "（空）",
            ).strip()
            + "\n\n"
            if knowledge_enabled
            else ""
        ),
        knowledge_carried_block=(
            _load_prompt_template(
                "fragment_query_carried_input_v1.md",
                carried_entries=carried_entries.strip() or "（无）",
                remaining_entries=remaining_entries,
            ).strip()
            + "\n\n"
            if knowledge_enabled
            else ""
        ),
        # The block enumerations name entry blocks only where the rules for
        # them ship (same predicate as $knowledge_request_block); otherwise the
        # round is told to emit a block it was given no rules for and whose
        # absence its contract no longer checks.
        entry_block_list=(
            "、可选的 `<requested_entries>` 块、一个 `<keep_entries>` 块"
            if knowledge_enabled
            else ""
        ),
        entry_block_reminder=(
            "、可选的 `<requested_entries>`（勿重复请求已透传词条）、"
            "`<keep_entries>`（每行一个已透传词条 key；没有需保留的条目时输出空块）"
            if knowledge_enabled
            else ""
        ),
        current_asr_csv=render_window_segments_as_csv(window).strip(),
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
    knowledge_enabled: bool = True,
) -> List[Dict[str, Any]]:
    """Correction-round messages.

    ``query_round_notes`` fills the generic ``<pre_round_notes>`` slot (the
    query round's window notes in normal mode, fast round 1's analysis notes
    in fast mode); ``entry_details`` and ``evidence_pack_mode`` are only
    non-default in fast mode. Production passes ``variant`` from the resolved
    task-group cell. ``tier`` only supplies the compatibility default when a
    caller (notably old session-replay fixtures) omits that explicit variant.
    """

    pack = context_pack or ContextPack()
    current_csv = render_window_segments_as_csv(window).strip()
    # Read-only raw lines before the window (v13), rendered on the same time
    # base as current_asr_csv so they come out mostly negative.
    preceding_csv = render_window_preceding_as_csv(window).strip()
    system = compose_correction_system(
        profile,
        tier=tier,
        variant=variant,
        evidence_pack_mode=evidence_pack_mode,
        extra_style=extra_style,
        common_mistakes_block=common_mistakes_block,
        knowledge_enabled=knowledge_enabled,
    )
    if task_update_feedback:
        system = (
            f"{system.rstrip()}\n\n"
            + _load_prompt_template(
                "correction_task_update_feedback_v2.md",
                feedback_schema=_feedback_schema(local_source_ids=True),
                # The block anchors after whatever really closes the reply:
                # parallel has no <next_advice>/<keep_entries> (plan A.7).
                feedback_position_clause=(
                    " `<next_advice>` 与 `<keep_entries>` 块之后"
                    if profile.continuity == "serial"
                    else " `<translated>` 块之后"
                ),
            )
        )
    user = compose_correction_user(
        profile,
        knowledge_enabled=knowledge_enabled,
        general_context_json=pack.general_prompt_text(),
        window_context=pack.window_context_for(window) or "（无）",
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

    Index injection and the entry blocks share one predicate (docs/llm_prompts.md):
    empty indices and no preinjected entries -- ``--knowledge none`` or an
    empty base -- drop every knowledge-owned piece of both prompts, so the
    round is never told to pick entries off an index it does not have.
    """

    knowledge_enabled = bool(
        streamer_index.strip()
        or common_index.strip()
        or preinjected_entries.strip()
    )
    search_rules = (
        _search_contract_rules(max_search_queries, knowledge_enabled=knowledge_enabled)
        if use_search_contract
        else _search_queries_rules(max_search_queries, knowledge_enabled=knowledge_enabled)
    )
    # Request and keep have independent caps; keep wins their shared cap.
    system = compose_fast_round1_system(
        profile,
        search_queries_rules=search_rules,
        task_update_feedback_block=(
            _research_task_feedback_block(local_source_ids=True)
            if collect_task_feedback
            else ""
        ),
        max_requested_entries=KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
        max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
        max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
        knowledge_enabled=knowledge_enabled,
    )
    user = compose_fast_round1_user(
        extra_info=extra_info or "（无）",
        note_url_extracts=note_url_extracts.strip() or "（无）",
        streamer_index=streamer_index or "（空）",
        common_index=common_index or "（空）",
        preinjected_entries=preinjected_entries,
        current_asr_csv=render_window_segments_as_csv(window).strip(),
        task_feedback_reminder=(
            TASK_FEEDBACK_REMINDER if collect_task_feedback else ""
        ),
        knowledge_enabled=knowledge_enabled,
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
        reasoning_clause=reasoning_clause(),
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


def agent_tool_worker_bootstrap(*, assignment_id: str, worker_id: str) -> str:
    """The prompt for a headless agent that takes and submits its task over
    the harness MCP server (docs/llm_agent_tool_protocol.md §1)."""

    return _load_prompt_template(
        "agent_tool_worker_v1.md",
        assignment_id=assignment_id,
        worker_id=worker_id,
    )


def agent_tool_worker_session_bootstrap(*, assignment_id: str, worker_id: str) -> str:
    """The prompt for a pseudo-conversational worker: one CLI session that
    takes the run's tasks one after another over the harness MCP server
    (docs/llm_local_agent.md §12.1.3)."""

    return _load_prompt_template(
        "agent_tool_worker_session_v1.md",
        assignment_id=assignment_id,
        worker_id=worker_id,
    )


def conversational_correction_effort() -> str:
    """The one effort note that belongs to the conversational path only.

    Not in the shared output contract, and not in the worker bootstrap:

    * the contract is what every backend is held to, and measured 2026-08-25
      the same sentence raised a REST model's thinking by 55% while buying
      nothing there -- the behaviour it prevents (counting characters by hand,
      writing a script to verify them) is one only an agent with its own tools
      can afford in the first place;
    * the bootstrap is task-agnostic. It says how to take a task and submit
      one, for whatever task the assignment queues. A note about one column of
      one session type does not belong in it.

    So it is appended to the protocol document of a correction task on that
    one road, where the first live test showed a real agent spending most of
    an hour on exactly this.
    """

    return _load_prompt_template("fragment_conversational_correction_effort_v1.md").rstrip()


def agent_worker_bootstrap(
    *,
    assignment_root: str,
    assignment_id: str,
    worker_id: str,
    task_command: str,
    watch_minutes: int,
    durable_status: str,
) -> str:
    """The control-protocol contract handed to a conversational Agent worker.

    This is prompt text a model reads and acts on, so it lives in
    ``prompt_templates/`` with the rest -- unlike the driver framing that is
    welded to one CLI's argv and belongs beside it.
    """

    return _load_prompt_template(
        "agent_worker_bootstrap_v1.md",
        assignment_root=assignment_root,
        assignment_id=assignment_id,
        worker_id=worker_id,
        task_command=task_command,
        watch_minutes=watch_minutes,
        durable_status=durable_status,
    )
