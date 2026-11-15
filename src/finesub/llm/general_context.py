"""The preparation task's general context as each reader gets it.

A single preparation task leaves one block. A long file prepared in chunks (`research._run_chunked_research`)
leaves one block per chunk, keyed by chunk id, and the source-id range each chunk's windows cover
(`GeneralChunk`). A correction window then reads its own chunk and as many others, nearest first, as fit
`GENERAL_MAX_TOKENS` (`select_general_chunks`); the knowledge update, which reads many windows at once and has
no own chunk, reads them in order up to the same ceiling (`whole_chunk_ids`).

Chunks run one after another. Each later chunk is handed the names the earlier ones settled (`carry_text`), and
the merged pack keeps one name table for the whole file (`merge_name_tables`): the first chunk to name something
wins, so every window reads the same names; a later chunk's disagreement is reported to the user, not applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from string import Template
from typing import Any, Callable, Mapping, Sequence

from .prompt_tables import prompt_phrases

#: The chunked general context's ceiling as one reader gets it. Deliberately NOT
#: ``injection_block_token_limit`` -- that budget is sized for knowledge entry dicts; this is
#: model-authored context that every correction window carries.
GENERAL_MAX_TOKENS = 20_000

NOTE_KEY = "合并说明"
#: What the preparation task relays from the user's notes for every window (`prepare_task_v1.md`); the
#: pack keeps it apart from the blocks (`ContextPack.user_requests`), so it is never weighed against the ceiling.
USER_REQUESTS_KEY = "user_requests"
OWN_CHUNK_KEY = "本窗所在分段"
#: How the preparation task marks a window note relayed from the user (`prepare_task_v1.md` duty 5); a window
#: whose note carries it gets the correction block on following user requests (`fragment_user_requests_v1.md`).
USER_REQUEST_MARK = "出自用户"

#: The merged pack's one table of names for the whole file, beside the chunk blocks in ``general_context``.
NAME_TABLE_KEY = "全片名字表"
#: What a later chunk thinks the table got wrong (`prepare_task_v1.md`): reported to the user, read by no window.
NAME_DISPUTES_KEY = "name_disputes"
#: The fields the name table holds, each keyed by its original-text field(s). `asr_risks` are local -- one
#: mishearing can stand for different words in two parts -- so the fix is part of their key and differing fixes
#: all stay.
NAME_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "entities": ("name",),
    "terminology": ("source",),
    "asr_risks": ("heard_as", "should_be"),
    "speakers": ("name",),
}
#: A later re-listing that differs in these is a disagreement; one that differs only in its note is not.
NAME_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "entities": ("zh", "type"),
    "terminology": ("zh",),
    "speakers": ("role",),
}
#: The explanatory fields, cut first when the names outgrow their ceiling.
NAME_DETAIL_FIELDS = ("note", "rule", "reason")
#: The ceiling of the names (and earlier summaries) handed to a later chunk, and of the whole-file table.
NAME_TABLE_MAX_TOKENS = 8_000
#: The carry's earlier summaries, keyed by chunk id, beside the table in `<earlier_chunks>`.
EARLIER_SUMMARIES_KEY = "前情摘要"


@dataclass(frozen=True)
class GeneralChunk:
    """One chunk's block: its id (the key in ``general_context``) and the first and last source id its
    research windows' bodies cover."""

    chunk_id: str
    first_source_id: str
    last_source_id: str

    def to_dict(self) -> dict[str, str]:
        return {"chunk_id": self.chunk_id, "first_source_id": self.first_source_id,
                "last_source_id": self.last_source_id}


@dataclass(frozen=True)
class GeneralSelection:
    """The blocks one reader gets. ``order`` is own chunk(s) first, then the others nearest first (the
    order the judge's and pass 3's summaries are joined in); ``shown`` the same blocks in chunk order (the
    correction's JSON). ``fallback``: the window could not be placed, so it reads the whole view."""

    order: tuple[str, ...]
    shown: tuple[str, ...]
    own: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    own_over_cap: bool = False
    fallback: bool = False


def general_for_correction(general: Mapping[str, Any]) -> dict[str, Any]:
    """The general context without the speaker judge's `speakers` (top level, and inside a chunked run's
    per-chunk blocks): what the correction reads, and what its size is counted on."""
    return {key: ({k: v for k, v in value.items() if k != "speakers"} if isinstance(value, Mapping) else value)
            for key, value in general.items() if key != "speakers"}


def render_single_general(general: Mapping[str, Any], user_requests: Sequence[str] = ()) -> str:
    """One preparation task's block as every reader gets it, the user's requests first."""
    payload = {**({USER_REQUESTS_KEY: list(user_requests)} if user_requests else {}), **general}
    return json.dumps(general_for_correction(payload), ensure_ascii=False, indent=2) if payload else "{}"


def render_chunked_general(general: Mapping[str, Any], shown: Sequence[str], own: Sequence[str] = (),
                           user_requests: Sequence[str] = ()) -> str:
    """The JSON a reader gets: the user's requests (every window gets all of them, outside the ceiling),
    the fixed note, the whole-file name table, the shown blocks in chunk order, then (for one window) the
    sentence naming its own chunk. Every window of a chunk gets the same text, so the prefix cache holds
    across them."""
    phrases = prompt_phrases("chunked_general")
    table = general.get(NAME_TABLE_KEY)
    payload: dict[str, Any] = {USER_REQUESTS_KEY: list(user_requests)} if user_requests else {}
    payload[NOTE_KEY] = phrases["note"] + (phrases["name_table"] if table else "")
    if table:
        payload[NAME_TABLE_KEY] = table
    payload.update((chunk_id, general[chunk_id]) for chunk_id in shown if chunk_id in general)
    if own:
        payload[OWN_CHUNK_KEY] = Template(phrases["own_chunk"]).substitute(chunks="、".join(own))
    return json.dumps(general_for_correction(payload), ensure_ascii=False, indent=2)


def merge_name_tables(blocks: Sequence[tuple[str, Mapping[str, Any]]]) -> tuple[dict[str, list], list[dict]]:
    """One name table out of the chunks' blocks, given as ``(chunk id, block)`` in chunk order, and the
    re-listings it dropped that disagree with what it kept.

    The first chunk to list a key wins and keeps its place. A later chunk was told to reuse the table and not
    list it again (`fragment_prepare_earlier_chunks_duty_v1.md`): a re-listing that differs only in its note is
    dropped quietly; one whose identity fields (`NAME_IDENTITY_FIELDS`) differ is dropped and returned, as
    ``{chunk, field, key, kept, dropped}``, for the user to see."""
    table: dict[str, list[dict]] = {field: [] for field in NAME_FIELD_KEYS}
    seen: dict[str, dict[tuple[str, ...], dict]] = {field: {} for field in NAME_FIELD_KEYS}
    drift: list[dict] = []
    for chunk_id, block in blocks:
        for field, key_fields in NAME_FIELD_KEYS.items():
            for item in block.get(field) or ():
                if not isinstance(item, Mapping):
                    continue
                key = tuple(str(item.get(name, "")).strip() for name in key_fields)
                if not key[0]:
                    continue
                kept = seen[field].get(key)
                if kept is None:
                    seen[field][key] = dict(item)
                    table[field].append(seen[field][key])
                    continue
                identity = NAME_IDENTITY_FIELDS.get(field, ())
                if any(str(kept.get(name, "")).strip() != str(item.get(name, "")).strip() for name in identity):
                    drift.append({"chunk": chunk_id, "field": field, "key": key[0],
                                  "kept": {name: kept.get(name) for name in identity},
                                  "dropped": {name: item.get(name) for name in identity}})
    return {field: items for field, items in table.items() if items}, drift


def without_details(table: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict]]:
    """The table with its explanatory fields cut: what goes first when the names outgrow their ceiling."""
    return {field: [{key: value for key, value in item.items() if key not in NAME_DETAIL_FIELDS} for item in items]
            for field, items in table.items()}


def fit_name_table(table: Mapping[str, Sequence[Mapping[str, Any]]], *, count: Callable[[str], int],
                   max_tokens: int = NAME_TABLE_MAX_TOKENS) -> tuple[dict[str, list], bool]:
    """The whole-file table within the ceiling: the explanatory fields cut if it is over, never an entry --
    names leave the chunk blocks, so a dropped entry would be a window's own name gone. Returns the table and
    whether it is still over."""
    fitted = {field: [dict(item) for item in items] for field, items in table.items()}
    if count(json.dumps(fitted, ensure_ascii=False, indent=2)) <= max_tokens:
        return fitted, False
    fitted = without_details(fitted)
    return fitted, count(json.dumps(fitted, ensure_ascii=False, indent=2)) > max_tokens


def carry_text(blocks: Sequence[tuple[str, Mapping[str, Any]]], *, count: Callable[[str], int],
               max_tokens: int = NAME_TABLE_MAX_TOKENS) -> tuple[str, bool]:
    """What the chunks so far hand the next one (`<earlier_chunks>`): their name table and each one's
    summary. Over the ceiling, the explanatory fields go first, then the earliest summaries; entries never --
    a later chunk that cannot see one writes the name its own way. Returns the text and whether it is still
    over. ``count`` must be deterministic (the text is part of the next chunk's input hash)."""
    if not blocks:
        return "", False
    table, _drift = merge_name_tables(blocks)
    summaries = {chunk_id: str(block.get("global_summary") or "").strip() for chunk_id, block in blocks}
    summaries = {chunk_id: text for chunk_id, text in summaries.items() if text}

    def render() -> str:
        payload: dict[str, Any] = {NAME_TABLE_KEY: table}
        if summaries:
            payload[EARLIER_SUMMARIES_KEY] = summaries
        return json.dumps(payload, ensure_ascii=False, indent=2)

    text = render()
    if count(text) > max_tokens:
        table = without_details(table)
        text = render()
    while count(text) > max_tokens and summaries:
        summaries.pop(next(iter(summaries)))
        text = render()
    return text, count(text) > max_tokens


def select_general_chunks(
    spans: Sequence[tuple[str, int, int]],
    own: Sequence[str],
    *,
    measure: Callable[[Sequence[str], Sequence[str]], int],
    max_tokens: int | None = None,
) -> GeneralSelection:
    """Which chunks a reader whose own chunks are ``own`` gets.

    ``spans`` are ``(chunk id, first position, last position)`` in source order, in chunk order;
    ``measure(shown, own)`` counts the rendered text. Own chunks always go in, even past the ceiling (there
    is no smaller unit to drop). The others go by their distance in rows from the own chunks' span, ties
    to the earlier chunk, and adding stops at the first that does not fit -- no skipping it for a smaller
    one further off. Measured from the own chunks, not from the window, so every window of a chunk gets the
    same blocks.

    The judge and pass 3 render lists and a summary, not this JSON; they reuse this selection as an
    approximation and are not counted on their own."""
    max_tokens = GENERAL_MAX_TOKENS if max_tokens is None else max_tokens
    index = {chunk_id: position for position, (chunk_id, _low, _high) in enumerate(spans)}
    own = tuple(sorted(dict.fromkeys(own), key=index.__getitem__))
    low = min(spans[index[chunk_id]][1] for chunk_id in own)
    high = max(spans[index[chunk_id]][2] for chunk_id in own)

    def distance(span: tuple[str, int, int]) -> int:
        return span[1] - high if span[1] > high else low - span[2]

    others = sorted((span for span in spans if span[0] not in own),
                    key=lambda span: (distance(span), index[span[0]]))
    in_order = lambda ids: tuple(sorted(ids, key=index.__getitem__))  # noqa: E731
    order = list(own)
    dropped: tuple[str, ...] = ()
    for position, span in enumerate(others):
        if measure(in_order([*order, span[0]]), own) > max_tokens:
            dropped = in_order(other[0] for other in others[position:])
            break
        order.append(span[0])
    return GeneralSelection(order=tuple(order), shown=in_order(order), own=own, dropped=dropped,
                            own_over_cap=measure(own, own) > max_tokens)


def whole_chunk_ids(chunk_ids: Sequence[str], *, measure: Callable[[Sequence[str]], int],
                    max_tokens: int | None = None) -> tuple[str, ...]:
    """The whole view: every chunk in order, whole trailing chunks dropped past the ceiling (never cut
    mid-JSON), at least one kept."""
    max_tokens = GENERAL_MAX_TOKENS if max_tokens is None else max_tokens
    kept = list(chunk_ids)
    while len(kept) > 1 and measure(kept) > max_tokens:
        kept.pop()
    return tuple(kept)
