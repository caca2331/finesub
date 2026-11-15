"""Chinese prompt templates for subtitle correction and translation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Dict, List, Mapping, Sequence

from finesub.reporting import current_reporter
from finesub.subtitles import line_groups

from .general_context import (
    NAME_DISPUTES_KEY,
    NAME_TABLE_KEY,
    NOTE_KEY,
    USER_REQUEST_MARK,
    USER_REQUESTS_KEY,
    GeneralChunk,
    GeneralSelection,
    render_chunked_general,
    render_single_general,
    select_general_chunks,
    whole_chunk_ids,
)
from .chunking import (
    Seam,
    SubtitleWindow,
    has_seams,
    render_window_preceding_as_csv,
    render_window_segments_as_csv,
)
from .routing.config import (
    CapabilityTier,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_NEW_REQUEST_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    WINDOW_CONTEXT_MAX_TOKENS,
)
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile
from .token_budget import TokenCounter, default_token_counter
from .token_truncate import truncate_to_token_window
from .prompt_compose import (
    compose_correction_system,
    PromptContext,
    compose_correction_user,
    render_entry_details_label,
    load_prompt_template as _load_prompt_template,
    mode_texts,
    render_knowledge_indexes,
    render_knowledge_ops_contract,
)


def _any_index(indexes: Mapping[str, str] | None) -> bool:
    """Is there any index to read at all?

    "No index" is a real state with prompt consequences: with nothing to pick
    entries off, the request rules come out of the system prompt entirely
    (`docs/llm_prompts.md`). It used to be spelled `streamer_index.strip() or
    common_index.strip()` at each site, which is the two-category assumption in
    miniature -- a third matchable category would simply not have counted.
    """

    return any(str(text).strip() for text in (indexes or {}).values())


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
            _load_prompt_template("fragment_search_queries_kb_note_v1.md").strip() if knowledge_enabled else ""
        ),
    ).strip()




def _feedback_schema(*, local_source_ids: bool) -> str:
    # one window's local line numbers (mapped back by the harness), or the stable source ids of the transcript
    source_ids_rule = _load_prompt_template(
        "fragment_task_feedback_source_ids_local_v1.md"
        if local_source_ids
        else "fragment_task_feedback_source_ids_stable_v1.md"
    ).strip()
    return _load_prompt_template(
        "fragment_task_feedback_schema_v3.md",
        source_ids_rule=source_ids_rule,
    ).strip()


def _research_task_feedback_block(*, local_source_ids: bool) -> str:
    """Feedback-collection addendum for the preparation task's delivery."""

    return "\n" + _load_prompt_template(
        "research_task_feedback_v1.md",
        feedback_schema=_feedback_schema(local_source_ids=local_source_ids),
    ).strip() + "\n"


def task_feedback_reminder() -> str:
    """Appended to a fill turn's closing order when feedback collection is on."""

    return _load_prompt_template("fragment_task_feedback_reminder_v1.md").strip()


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

    ``general_context`` covers every window: one block, or for a file prepared in
    chunks one block per chunk keyed by chunk id, with ``general_chunks`` naming
    the source-id range each covers (``general_context.py``: a window reads its own
    chunk and the nearest others that fit).  Window notes carry their original
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
    general_chunks: tuple[GeneralChunk, ...] = ()
    #: what the preparation task relayed from the user's notes for every window: every reader gets it whole
    user_requests: tuple[str, ...] = ()
    #: a later chunk's objections to the names the earlier ones settled (`general_context.merge_name_tables`):
    #: kept for the user's report, read by no window
    name_disputes: tuple[Mapping[str, Any], ...] = ()
    #: chunk selections by own chunks (and the whole view), computed once per pack
    _selections: dict = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.window_contexts, Mapping):
            raise TypeError(
                "ContextPack.window_contexts takes WindowContextNote records; "
                "an id->text mapping is the model's shape, so build it with "
                "ContextPack.from_dict and bind ranges before injecting."
            )

    def general_text_for(self, window: SubtitleWindow) -> str:
        return self.general_text_for_source_ids(_body_ids(window))

    def general_text_for_source_ids(self, source_ids: Sequence[str]) -> str:
        """The general context as the correction window holding ``source_ids`` (its body) reads it.
        `speakers` (asked only for the speaker judge) stays out, also inside a chunked run's per-chunk
        blocks: the correction's prompt is the same whether the judge runs or not."""
        if not self.general_chunks:
            return self._single_text()
        selection = self.general_selection(source_ids)
        return render_chunked_general(self.general_context, selection.shown, selection.own, self.user_requests)

    def general_text_whole(self) -> str:
        """Every chunk in order up to the ceiling, trailing chunks dropped: for the knowledge update,
        which reads many windows at once and has no own chunk."""
        if not self.general_chunks:
            return self._single_text()
        return render_chunked_general(self.general_context, self._whole().shown, (), self.user_requests)

    def general_blocks(self, source_ids: Sequence[str] | None = None) -> list[Mapping[str, Any]]:
        """The general context as blocks: the whole-file name table (when the chunks have one), then every
        chunk's in order (``None``), else the chunks the window holding ``source_ids`` reads, own first, then
        nearest first."""
        general = self.general_context
        if not general:
            return []
        if not self.general_chunks:
            return [general]
        ids = ([chunk.chunk_id for chunk in self.general_chunks] if source_ids is None
               else self.general_selection(source_ids).order)
        # the whole-file name table leads: every window's names are in it, none in the chunk blocks
        return [general[key] for key in (NAME_TABLE_KEY, *ids) if isinstance(general.get(key), Mapping)]

    def general_selection(self, source_ids: Sequence[str]) -> GeneralSelection:
        """The chunks the window holding ``source_ids`` reads: the chunks its body overlaps are its own.
        A window that cannot be placed (its rows or a chunk's ends missing from the source order) reads
        the whole view, with a warning, rather than no own chunk."""
        own = self._own_chunks(source_ids)
        return self._fallback() if own is None else self.chunk_selection(own)

    def chunk_selection(self, own: Sequence[str]) -> GeneralSelection:
        """The chunks a window whose own chunks are ``own`` reads (``general_context.select_general_chunks``)."""
        key = tuple(own)
        selection = self._selections.get(key)
        if selection is None:
            spans = self._spans()
            if spans is None:
                return self._fallback()
            counter = default_token_counter()
            selection = select_general_chunks(spans, own, measure=lambda shown, own_ids: counter.count_text(
                render_chunked_general(self.general_context, shown, own_ids)))
            self._selections[key] = selection
        return selection

    def _single_text(self) -> str:
        return render_single_general(self.general_context, self.user_requests)

    def _positions(self) -> dict[str, int]:
        if not self.source_order:
            raise ValueError("ContextPack's chunked general context needs the source order; call "
                             "with_source_order(current segment ids) after loading.")
        return {source_id: index for index, source_id in enumerate(self.source_order)}

    def _spans(self) -> list[tuple[str, int, int]] | None:
        positions = self._positions()
        spans = [(chunk.chunk_id, positions.get(chunk.first_source_id), positions.get(chunk.last_source_id))
                 for chunk in self.general_chunks]
        if any(low is None or high is None or low > high for _chunk_id, low, high in spans):
            return None
        return spans  # type: ignore[return-value]

    def _own_chunks(self, source_ids: Sequence[str]) -> tuple[str, ...] | None:
        positions, spans = self._positions(), self._spans()
        # Rows the source order leaves out (empty text, no duration: the speaker judge's rows keep them) do not
        # unplace the window; it is placed by the rows that are there.
        body = [position for position in (positions.get(str(source_id)) for source_id in source_ids)
                if position is not None]
        if spans is None or not body:
            return None
        first, last = min(body), max(body)
        own = tuple(chunk_id for chunk_id, low, high in spans if low <= last and first <= high)
        return own or None

    def _whole(self) -> GeneralSelection:
        selection = self._selections.get("whole")
        if selection is None:
            counter = default_token_counter()
            shown = whole_chunk_ids([chunk.chunk_id for chunk in self.general_chunks], measure=lambda ids: (
                counter.count_text(render_chunked_general(self.general_context, ids))))
            selection = self._selections["whole"] = GeneralSelection(order=shown, shown=shown, fallback=True)
        return selection

    def _fallback(self) -> GeneralSelection:
        if "fallback-warned" not in self._selections:
            self._selections["fallback-warned"] = True
            current_reporter().warning(
                "research-general-unplaced",
                "准备任务分块的区间或某个纠错窗口的字幕在当前字幕里定位不到",
                impact="定位不到的窗口读按先后排的整片通用背景（超上限从尾部丢块），不一定含它所在的那块",
            )
        return self._whole()

    @property
    def has_unbound_window_contexts(self) -> bool:
        return bool(self.unbound_window_contexts)

    def with_source_order(self, source_ids: Sequence[str]) -> "ContextPack":
        return replace(self, source_order=tuple(str(source_id) for source_id in source_ids))

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
        return replace(self, window_contexts=tuple(notes), unbound_window_contexts={},
                       source_order=tuple(dict.fromkeys(source_order)))

    def window_context_for(
        self,
        window: SubtitleWindow,
        *,
        counter: TokenCounter | None = None,
        max_tokens: int = WINDOW_CONTEXT_MAX_TOKENS,
        report_sink: dict[str, Any] | None = None,
    ) -> str:
        return self.window_context_for_source_ids(
            _body_ids(window),
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
        # A note shared across windows is stored once per window, so a window spanning two of them would read
        # it twice: identical paragraphs are kept once.
        text = "\n\n".join(dict.fromkeys(
            paragraph.strip()
            for item in selected
            for paragraph in item[2].context.split("\n\n")
            if paragraph.strip()
        ))
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
            **({"general_chunks": [chunk.to_dict() for chunk in self.general_chunks]}
               if self.general_chunks else {}),
            **({USER_REQUESTS_KEY: list(self.user_requests)} if self.user_requests else {}),
            **({NAME_DISPUTES_KEY: [dict(item) for item in self.name_disputes]} if self.name_disputes else {}),
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
        if NOTE_KEY in general:
            # The chunked shape before per-window selection: one merged block whose dropped chunks are
            # gone. Refused rather than read as one block, so the preparation task runs again.
            raise ValueError(
                "general_context is the old merged shape of a chunked preparation task "
                f"(it has `{NOTE_KEY}`); re-run the preparation task"
            )
        general_chunks = tuple(
            GeneralChunk(str(item["chunk_id"]), str(item["first_source_id"]), str(item["last_source_id"]))
            for item in data.get("general_chunks") or ()
        )
        # The model writes the requests inside `general_context` (prepare_task_v1.md); the pack keeps them
        # in their own field, so a chunked run's union and every reader find them in one place.
        lists = [value if isinstance(value, list) else []
                 for value in (data.get(USER_REQUESTS_KEY), general.get(USER_REQUESTS_KEY))]
        # A later chunk writes its objections to the earlier names inside `general_context` too; they are the
        # user's to read, so no reader of the general context sees them.
        disputes = [dict(item) for value in (data.get(NAME_DISPUTES_KEY), general.get(NAME_DISPUTES_KEY))
                    if isinstance(value, list) for item in value if isinstance(item, Mapping)]
        general = {key: value for key, value in general.items() if key not in (USER_REQUESTS_KEY, NAME_DISPUTES_KEY)}
        user_requests = tuple(dict.fromkeys(
            text for text in (str(item).strip() for items in lists for item in items) if text))
        # The model emits ``[{window_id, context}]`` (prepare_task_v1.md); the
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
            general_chunks=general_chunks,
            user_requests=user_requests,
            name_disputes=tuple(disputes),
        )


def _body_ids(window: SubtitleWindow) -> list[str]:
    """A window's own rows' source ids: the overlap rows it re-reads are the window before's."""
    return [segment.id for segment in window.segments[len(window.overlap_segments):]]


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

    The preparation task's background list is assembled per axis
    (docs/llm_harness_behavior.md), so the numbers cannot live in the template
    -- a disabled half would leave a hole.
    """

    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def r1_request_cap(*, downstream_can_request: bool) -> int:
    """How many entries R1 may request.

    ``KB_WINDOW_NEW_REQUEST_MAX_ENTRIES`` was sized for a *per-window increment*
    -- it leaves room for the next window to ask for more. When no window can
    request (no request channel; docs/llm_harness_behavior.md), the preparation
    task is the session's only pick and that reserve is pure loss, so it gets
    the whole shared budget.
    """

    return (
        KB_WINDOW_NEW_REQUEST_MAX_ENTRIES
        if downstream_can_request
        else KB_WINDOW_TOTAL_ENTRIES
    )


def _r1_entry_blocks(max_requested_entries: int, *, keep_source_block: str) -> str:
    """The request blocks; ``keep_source_block`` is the block the reader sees
    the entries it may keep in -- `<preinjected_entries>` in a preparation
    task, `<entry_details>` in a correction window."""

    return _load_prompt_template(
        "fragment_research_entry_blocks_v1.md",
        keep_source_block=keep_source_block,
        max_requested_entries=max_requested_entries,
        max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
        max_total_entries=KB_WINDOW_TOTAL_ENTRIES,
    ).strip()


def build_correction_csv_messages(
    *,
    window: SubtitleWindow,
    context_pack: ContextPack | None = None,
    audio_file_label: str = "",
    previous_advice: str = "",
    window_notes: str = "",
    search_results: str = "",
    entry_details: str = "",
    phonetic_hints: str = "",
    entries_slimmed: bool = False,
    entries_requestable: bool = False,
    extra_style: str = "",
    style_block: str = "",
    task_update_feedback: bool = False,
    self_search: bool = False,
    profile: TranslationProfile = DEFAULT_PROFILE,
    tier: CapabilityTier = CapabilityTier.CAPABLE,
    variant: str | None = None,
    knowledge_enabled: bool = True,
    style_pick_reminder: str = "",
    agentic: bool = False,
) -> List[Dict[str, Any]]:
    """Correction-round messages.

    ``window_notes`` fills the generic ``<pre_round_notes>`` slot (the
    request turn's window notes); ``entry_details`` carries the window's
    knowledge entries and ``phonetic_hints`` the rows of terms its ASR may
    have misheard (`kb_phonetic_hints`; empty unless ``FINESUB_KB_SLIM`` is
    on). ``entries_slimmed`` says the entries were slimmed, which the block's
    label then says; ``entries_requestable`` (a request channel) adds that a
    slimmed entry can be requested whole. Production passes ``variant`` from
    the resolved task-group cell. ``tier`` only supplies the compatibility default when a
    caller (notably old session-replay fixtures) omits that explicit variant.
    ``agentic``: a tool-session window that reads entries and the picked style
    with its own `kb_read` (:func:`prompt_compose.compose_correction_system`).
    """

    pack = context_pack or ContextPack()
    current_csv = render_window_segments_as_csv(window).strip()
    turns = has_seams(window.segments)
    grouped = window.grouped
    # a local window can hold the row a lane line covers (同时) without the lane line itself
    keyed = turns and any(s.group_key is not None or s.seam in (Seam.TOGETHER, Seam.BACK) for s in window.segments)
    cloud = turns and line_groups.speakers([s.group_key for s in window.segments])
    # Read-only raw lines before the window (v13), rendered on the same time
    # base as current_asr_csv so they come out mostly negative.
    preceding_csv = render_window_preceding_as_csv(window).strip()
    system = compose_correction_system(
        profile,
        tier=tier,
        variant=variant,
        extra_style=extra_style,
        style_block=style_block,
        knowledge_enabled=knowledge_enabled,
        turns=turns,
        grouped=grouped,
        keyed=keyed,
        cloud=cloud,
        task_update_feedback=task_update_feedback,
        self_search=self_search,
        agentic=agentic,
        length_scale=window.length_scale,
    )
    if task_update_feedback:
        system = (
            f"{system.rstrip()}\n\n"
            + _load_prompt_template(
                "correction_task_update_feedback_v2.md",
                **mode_texts(variant, tier),
                feedback_schema=_feedback_schema(local_source_ids=True),
                # The block anchors after whatever really closes the reply:
                # parallel has no <next_advice>/<keep_entries> (plan A.7).
                feedback_position_clause=(
                    " `<next_advice>` 与 `<keep_entries>` 块之后"
                    if PromptContext(profile=profile, knowledge_enabled=knowledge_enabled).keeps_entries
                    else " `<next_advice>` 块之后"
                    if profile.continuity == "serial"
                    else " `<translated>` 块之后"
                ),
            )
        )
    # Plan 5r B2 (experiment): empty unless FINESUB_OVERLAP_CANDIDATES is set.
    from .overlap_candidates import candidate_rows

    overlap_rows = candidate_rows(window)
    overlap_candidates_block = (
        _load_prompt_template("fragment_overlap_candidates_v1.md", candidate_rows=overlap_rows).strip() + "\n\n"
        if overlap_rows
        else ""
    )
    if phonetic_hints.strip():
        overlap_candidates_block += (
            _load_prompt_template(
                "fragment_kb_phonetic_hints_v1.md", hint_rows=phonetic_hints.strip(), **mode_texts(variant, tier)
            ).strip()
            + "\n\n"
        )
    window_context = pack.window_context_for(window)
    user = compose_correction_user(
        profile,
        knowledge_enabled=knowledge_enabled,
        general_context_json=pack.general_text_for(window),
        window_context=window_context or "（无）",
        # the preparation task marks a window's own relayed requests so (`prepare_task_v1.md` duty 5)
        user_requests=bool(pack.user_requests) or USER_REQUEST_MARK in window_context,
        extra_style=extra_style,
        entry_details=entry_details.strip() or "（无）",
        entry_details_label=render_entry_details_label(
            slimmed=entries_slimmed, requestable=entries_requestable, agentic=agentic
        ),
        previous_advice=previous_advice.strip() or "（无）",
        pre_round_notes=window_notes.strip(),
        search_results=search_results,
        preceding_context_csv=preceding_csv,
        current_asr_csv=current_csv,
        current_asr_row_count=len(window.segments),
        overlap_candidates_block=overlap_candidates_block,
        tier=tier,
        variant=variant,
        style_pick_reminder=style_pick_reminder,
        turns=turns,
        grouped=grouped,
        keyed=keyed,
        task_update_feedback=task_update_feedback,
        agentic=agentic,
        length_scale=window.length_scale,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _knowledge_update_system(
    refined: bool,
    *,
    inputs_block: str = "",
    style_marks_section: str | None = None,
    split: bool = False,
    compress: bool = False,
    repair: bool = False,
) -> str:
    """The knowledge-update system message: structure spec + output contract.

    The proposal-repair round (``repair``) speaks the same proposal grammar and
    the same output contract, so it shares both; its head is its own: it has
    no window material, only the dropped proposals and the current entries,
    and the original proposal's ``reason`` is its evidence (cold read round 4,
    D2 -- the update's evidence rules sent the model hunting for material the
    round does not carry).
    """

    from .knowledge.node.presets import CATEGORIES, MATCHABLE_CATEGORIES
    from .knowledge.node.render import render_category_line, render_structure_spec

    # The refined-alignment marks (`knowledge/style_marks.py`) are asked for
    # only when a style is pinned -- there is nothing to mark otherwise -- and
    # never in the proposal-repair round, which re-decides proposals only.
    style_marks_contract = (
        "\n"
        + _load_prompt_template(
            "fragment_knowledge_style_marks_v1.md",
            strict=True,
            convention_section=style_marks_section,
        ).strip()
        if refined and style_marks_section
        else ""
    )
    if repair:
        template, head = "knowledge_proposal_repair_system_v1.md", {}
    elif refined:
        template = "knowledge_update_refined_v1.md"
        head = {"knowledge_inputs": inputs_block, "style_marks_contract": style_marks_contract}
    else:
        template, head = "knowledge_update_artifacts_only_v1.md", {"knowledge_inputs": inputs_block}
    return _load_prompt_template(
        template,
        strict=True,
        **head,
        knowledge_structure=_load_prompt_template(
            "fragment_knowledge_structure_v1.md",
            strict=True,
            # EVERY category, not just the matchable ones (plan §0 requirement 3).
            # ⚠ Two different category sets meet in this file and they are not
            # interchangeable: the INDEX面 is matchable-only (a standalone
            # category has no index -- `repo.index_text` raises for one), while
            # the TEMPLATE面 is all of them. Calling this with no argument
            # defaulted to the matchable set, which is how `style`'s template
            # went unseen by the maintenance session it was written for -- the
            # refined one. Without refined subtitles (and in the conflict
            # repair, which speaks this contract) there is no style evidence
            # and no style to write: the templates are the writable ones only
            # (owner 2026-10-03, cold read Q6); the roster line stays whole.
            sections_spec=render_structure_spec(CATEGORIES if refined else MATCHABLE_CATEGORIES),
            categories_line=render_category_line(CATEGORIES),
            # The conflict repair is shown the conflicted entries, not the index.
            create_index_check=(
                _load_prompt_template("fragment_knowledge_create_index_check_v1.md").strip() + " "
                if not repair
                else ""
            ),
            # Only where `split_entry` is in the op table below: the conflict
            # repair round is never offered it, and a rule naming it there is
            # a dead clause the gate rejects (cold read round 2, P181).
            # The size rule only where an injected entry carries the notice it points at: never in the
            # conflict repair (no notice there), nor for entries all within their soft size. An agent
            # session can still meet the notice on a long entry it reads whole through `kb_read`
            # without this rule: the notice itself says what to cut, the rule only adds the priority
            # (describe shorter before deleting lines). Left so (acceptance review, 2026-10-19): rare,
            # mild, and the prompt is built before the route is known.
            size_rule=_load_prompt_template(
                "fragment_knowledge_size_rule_v1.md",
                strict=True,
                split_clause=(
                    _load_prompt_template("fragment_knowledge_split_clause_v1.md").strip()
                    if split
                    else ""
                ),
            ).strip() if compress else "",
        ).strip(),
        # ...and a THIRD set, narrower than both: what this session may write.
        # `apply_model_proposals` gates the round on `MATCHABLE_CATEGORIES`, so
        # the contract has to name those and only those -- a template the model
        # can read is not a category it may write into.
        knowledge_output=render_knowledge_ops_contract(MATCHABLE_CATEGORIES, split=split),
    )



def build_knowledge_proposal_repair_messages(
    *,
    task_summary: str,
    read_rev: int,
    current_rev: int,
    conflict_rows: str,
    rejected_rows: str,
    kb_entries: str,
) -> List[Dict[str, str]]:
    """Feed the proposals that did not land back to the model: those a concurrent writer displaced
    (``conflict_rows``, plan B') and those the apply layer rejected (``rejected_rows``). Each group gets its
    own section, with what to do with it, only when it has rows.

    Only the knowledge proposals are repairable this way, so the round always
    speaks the ``artifacts_only`` contract (one `<knowledge_proposals>` block)
    even when the task itself is ``refined_aligned``: since the translation
    ledgers retired, every proposal is a knowledge proposal, so there is no
    other block to repair.

    No fresh material is injected on purpose -- the model is asked to re-decide
    the dropped items against the CURRENT entries, not to think of new ones it
    can no longer see the evidence for.
    """

    sections = [
        _load_prompt_template(name, strict=True, rows=rows.strip(), read_rev=read_rev, current_rev=current_rev)
        .strip()
        for name, rows in (
            ("fragment_kb_repair_conflict_v1.md", conflict_rows),
            ("fragment_kb_repair_rejected_v1.md", rejected_rows),
        )
        if rows.strip()
    ]
    user = _load_prompt_template(
        "knowledge_proposal_repair_v1.md",
        strict=True,
        task_summary=task_summary.strip() or "（无）",
        dropped_sections="\n\n".join(sections) or "（无）",
        current_rev=current_rev,
        kb_entries=kb_entries.strip() or "（无）",
    )
    from .knowledge.entry_size import asks_to_compress, offers_split

    return [
        {"role": "system", "content": _knowledge_update_system(
            False, split=offers_split(kb_entries), compress=asks_to_compress(kb_entries), repair=True,
        )},
        {"role": "user", "content": user},
    ]


def build_knowledge_update_messages(
    *,
    refined: bool,
    task_summary: str,
    window_packs: str,
    general_context: str = "",
    research_feedback: str = "",
    aggregated_feedback: str = "",
    kb_entries: str = "",
    knowledge_indexes: Mapping[str, str] | None = None,
    chunk_index: int = 1,
    multi_chunk: bool = False,
    window_range: str = "",
    style_marks_section: str | None = None,
    heard: bool = False,
) -> List[Dict[str, str]]:
    """Unified knowledge-update prompt (docs/knowledge.md).

    ``refined`` selects the ``refined_aligned`` variant (the refined subtitles
    as evidence, plus the style entry it may propose into); the
    ``artifacts_only`` variant has neither.
    ``window_packs`` is the rendered per-window material blocks of ONE chunk;
    ``multi_chunk`` marks a 100k-budget chunked run (the notice states no
    total, since over-limit chunks may split further at run time).
    The style entry, when the run named one, arrives through ``kb_entries``
    like every other entry — same handles, same apply path.
    ``style_marks_section`` (refined variant with a pinned style) names that
    style's convention section and asks for the ``<style_marks>`` block.
    ``heard``: some raw line of the chunk is one the speaker judge heard
    (`knowledge.materials.HEARD_MARK`), and the inputs say what the mark means.
    """

    inputs_block = _load_prompt_template(
        "fragment_knowledge_update_inputs_v1.md",
        strict=True,
        refined_csv_bullet=(
            "\n" + _load_prompt_template("fragment_knowledge_update_refined_csv_v1.md").rstrip("\n") if refined else ""
        ),
        heard_bullet=_load_prompt_template("fragment_knowledge_update_heard_v1.md").rstrip("\n") if heard else "",
    ).strip()
    marks = bool(refined and style_marks_section)
    from .knowledge.entry_size import asks_to_compress, offers_split

    system = _knowledge_update_system(
        refined, inputs_block=inputs_block,
        style_marks_section=style_marks_section if marks else None,
        split=offers_split(kb_entries),
        compress=asks_to_compress(kb_entries),
    )
    chunk_notice = (
        "\n" + _load_prompt_template(
            "fragment_knowledge_update_chunk_notice_v1.md", strict=True, chunk_index=chunk_index,
            window_range=window_range,
        )
        if multi_chunk
        else ""
    )
    # One proposal block in both variants since the style entry took over from
    # the mistake ledger (`docs/knowledge.md`, 「两个翻译台账（已退役，2026-09-02）」): a
    # style convention is an entry update like any other, so it travels in
    # `<knowledge_proposals>`. The one extra block, `<style_marks>`, is not an
    # update but a judgement on the pinned style's conventions, and is asked
    # for only when there is one (`knowledge/style_marks.py`).
    user = _load_prompt_template(
        "knowledge_update_user_v1.md",
        strict=True,
        task_summary=task_summary.strip() or "（无）",
        chunk_notice=chunk_notice,
        general_context=general_context.strip() or "（无）",
        research_feedback=research_feedback.strip() or "（无）",
        aggregated_feedback=aggregated_feedback.strip() or "（无）",
        kb_entries=kb_entries.strip() or "（无）",
        knowledge_indexes=render_knowledge_indexes(knowledge_indexes),
        window_packs=window_packs.strip() or "（无）",
        style_marks_reminder=(
            _load_prompt_template("fragment_knowledge_style_marks_reminder_v1.md").rstrip("\n")
            if marks
            else ""
        ),
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _agent_tool_retrieval_exception(native_search: bool) -> str:
    """The carve-out rule 4 needs when the call's project entitles retrieval.

    Without it the two inputs contradict each other: the session protocol's
    native-search fragment tells the model to look things up, while the
    bootstrap forbids every tool but the harness server's. Measured on the
    2026-08-30 batch, the model obeyed the protocol, the project hook refused
    the call, and the refusal produced no result step -- so the search was
    neither performed nor recorded.
    """

    if not native_search:
        return ""
    # `rstrip()` only: the fragment is spliced mid-sentence and its leading
    # space is load-bearing.
    return _load_prompt_template("fragment_agent_tool_retrieval_v1.md").rstrip()


def _agent_tool_patch_repair(takes_patch: bool) -> str:
    """How a repair round of a task whose repairs may come as a patch reads
    (`agent_validators.VALIDATOR_MERGES`: correction windows and the speaker
    judge); any other task's repair is the whole answer again, so it is not told."""

    return _load_prompt_template("fragment_agent_tool_patch_repair_v1.md").rstrip("\n") if takes_patch else ""


def agent_tool_worker_bootstrap(
    *, assignment_id: str, worker_id: str, native_search: bool = False
) -> str:
    """The prompt for a headless agent that takes and submits its task over
    the harness MCP server (docs/llm_agent_tool_protocol.md §1)."""

    return _load_prompt_template(
        "agent_tool_worker_v1.md",
        assignment_id=assignment_id,
        worker_id=worker_id,
        retrieval_exception=_agent_tool_retrieval_exception(native_search),
        # the bootstrap is written before the task is fetched, so it says how a patch repair reads
        patch_repair=_agent_tool_patch_repair(True),
    )


def agent_tool_opening_blocks(blocks: Sequence[tuple[str, str]]) -> str:
    """Blocks as an opening carries them, inline or in the files it names."""

    return "".join(
        f'\n<block kind="{kind}">\n{text.rstrip()}\n</block>\n' for kind, text in blocks
    )


def agent_tool_worker_opening(
    *,
    assignment_id: str,
    worker_id: str,
    manifest: Mapping[str, Any],
    shared_blocks: Sequence[tuple[str, str]],
    task_blocks: Sequence[tuple[str, str]],
    block_files: str = "",
    native_search: bool = False,
    takes_patch: bool,
) -> str:
    """The opening of a single-task tool session whose harness already
    claimed the task: the manifest and the blocks the driver can carry, so
    the session needs no `next_task` (docs/llm_agent_tool_protocol.md §1).

    What every window shares -- the rules, the protocol, the run's knowledge
    index -- comes first and the task's own manifest and payload last, so a
    provider's prefix cache can reach past the shared part. ``block_files``
    names the files holding the `read: file` blocks (agy), last.
    ``takes_patch``: the task's repairs may come as a patch, so the opening
    says how one reads.
    """

    return _load_prompt_template(
        "agent_tool_worker_opening_v1.md",
        strict=True,
        assignment_id=assignment_id,
        worker_id=worker_id,
        retrieval_exception=_agent_tool_retrieval_exception(native_search),
        patch_repair=_agent_tool_patch_repair(takes_patch),
        manifest=json.dumps(manifest, ensure_ascii=False, indent=2),
        shared_blocks=agent_tool_opening_blocks(shared_blocks),
        task_blocks=agent_tool_opening_blocks(task_blocks),
        block_files=(
            "\n" + _load_prompt_template("fragment_opening_block_files_v1.md", files=block_files).strip()
            if block_files
            else ""
        ),
    )


def agent_tool_worker_session_bootstrap(
    *, assignment_id: str, worker_id: str, native_search: bool = False
) -> str:
    """The prompt for a pseudo-conversational worker: one CLI session that
    takes the run's tasks one after another over the harness MCP server
    (docs/llm_local_agent.md §12.1.3)."""

    return _load_prompt_template(
        "agent_tool_worker_session_v1.md",
        assignment_id=assignment_id,
        worker_id=worker_id,
        retrieval_exception=_agent_tool_retrieval_exception(native_search),
        # the queue may hold any task, so the session is told how a patch repair reads
        patch_repair=_agent_tool_patch_repair(True),
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
