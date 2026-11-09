"""What pass 3 is shown: the material's notes, the task's style, and one window of finished subtitles.

A line reads `序号|时长|字/秒|识别原文|纠错文|译文|把握|备注`: the seconds it stays on screen (the final SRT's,
after the timeline's padding: what a viewer reads by), the translation's weighted characters over them, the ASR
text only where it differs from the corrected text (`=` otherwise, compared without punctuation, symbols and
spaces), the corrected text, the translation, the doubt the correction marked (`low` / `median`) and its note.
The window's lines are the correction window's own (`windows_of`), with up to `NEIGHBOUR_ROWS` read-only lines
on each side. What production adds to the trial: the knowledge base's names and terms the correction used in
that window, and the task's own style block.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ....subtitles.metrics import weighted_char_count
from ...text_cells import encode_text_cell
from ...prompt_compose import load_prompt_template
from ...general_context import USER_REQUEST_MARK
from ...prompt_tables import prompt_phrases
from ..speaker_judge.run import general_blocks, listed, summary_of

SYSTEM_TEMPLATE = "proofread_system_v1.md"
USER_TEMPLATE = "proofread_user_v1.md"
KB_TERMS_TEMPLATE = "proofread_kb_terms_v1.md"
BEFORE_TEMPLATE = "proofread_before_v1.md"
AFTER_TEMPLATE = "proofread_after_v1.md"
EXTRA_STYLE_TEMPLATE = "fragment_extra_style_v1.md"
USER_REQUESTS_TEMPLATE = "proofread_user_requests_v1.md"
WINDOW_CONTEXT_TEMPLATE = "proofread_window_context_v1.md"

#: Read-only lines shown before and after a window, for reading a sentence across its edge.
NEIGHBOUR_ROWS = 2
#: The doubt marks worth showing; a confident line leaves the cell empty.
DOUBTS = ("low", "median")
NOTHING = "（无）"


@dataclass(frozen=True)
class Line:
    """One finished subtitle as pass 3 reads it."""

    seconds: float
    asr: str
    corrected: str
    translation: str
    conf: str | None = None
    note: str = ""


@dataclass(frozen=True)
class Window:
    """One proofreading call: the correction window's id, the indices of its lines in the finished subtitles,
    the read-only lines around them, and its lines' source ids (which chunks of a chunked preparation task's
    general context it reads)."""

    chunk_id: str
    rows: tuple[int, ...]
    before: tuple[int, ...] = ()
    after: tuple[int, ...] = ()
    source_ids: tuple[str, ...] = ()


def windows_of(source_ids: Sequence[Sequence[str]], windows: Sequence[Any]) -> list[Window]:
    """The finished subtitles (each by its source ids) cut by the correction windows (each with `chunk_id` and
    `source_ids`, in order): a line belongs to the first window holding its first source id -- the one whose
    version stitching kept (core ownership) -- and a line naming none (an inserted line) to the line before it. Windows left
    without lines are left out."""
    owner_of: dict[str, str] = {}
    for window in windows:
        for source_id in window.source_ids:
            owner_of.setdefault(str(source_id), window.chunk_id)
    order = [window.chunk_id for window in windows]
    owners: list[str] = []
    for ids in source_ids:
        owner = next((owner_of[str(key)] for key in ids if str(key) in owner_of), None)
        owners.append(owner or (owners[-1] if owners else order[0]))
    out = []
    for chunk_id in order:
        rows = tuple(index for index, owner in enumerate(owners) if owner == chunk_id)
        if not rows:
            continue
        low, high = rows[0], rows[-1] + 1
        out.append(Window(chunk_id, rows, tuple(range(max(0, low - NEIGHBOUR_ROWS), low)),
                          tuple(range(high, min(len(owners), high + NEIGHBOUR_ROWS))),
                          tuple(str(key) for index in rows for key in source_ids[index])))
    return out


def bare(text: str) -> str:
    """The characters a viewer reads: punctuation, symbols and spaces left out."""
    return "".join(c for c in text if not unicodedata.category(c).startswith(("P", "Z", "S")))


def line_cells(number: int, line: Line) -> str:
    speed = f"{weighted_char_count(line.translation) / line.seconds:.1f}" if line.seconds > 0 else ""
    asr = "=" if bare(line.asr) == bare(line.corrected) else line.asr
    doubt = line.conf if line.conf in DOUBTS else ""
    return "|".join([str(number), f"{line.seconds:.1f}", speed, encode_text_cell(asr), encode_text_cell(line.corrected),
                     encode_text_cell(line.translation), doubt, encode_text_cell(line.note)])


def _listed(items: Sequence[Any], render: Any) -> str:
    return "\n".join(f"- {render(item)}" for item in items) or NOTHING


def _plain(context_pack: Any, field: str, source_ids: Sequence[str]) -> list[str]:
    """A general-context field that lists sentences (`streamer_notes`, `uncertainties`)."""
    return [str(item).strip() for block in general_blocks(context_pack, source_ids) for item in block.get(field) or ()
            if str(item).strip()]


def style_text(style_block: str, extra_style: str) -> str:
    """The task's style as the correction got it: the user's own words for this run (`--extra-style`, which
    outranks the styles below it, owner 2026-11-03), then the selected styles' block -- the factory default
    when the task has no user style."""
    parts = [style_block.strip()]
    if extra_style.strip():
        parts.insert(0, load_prompt_template(EXTRA_STYLE_TEMPLATE, strict=True, extra=extra_style.strip()).strip())
    return "\n\n".join(part for part in parts if part)


def proofread_prompt(window: Window, lines: Sequence[Line], *, title: str, context_pack: Any, style: str,
                     kb_terms: str = "") -> tuple[str, str]:
    """`(system, user)` for one window. `style` is `style_text`'s; `kb_terms` the names and terms of the
    knowledge entries the correction used in this window (empty: the block is left out)."""
    get = lambda item, key: str(item.get(key) or "").strip()  # noqa: E731
    ids = window.source_ids
    speakers = _listed(listed(context_pack, "speakers", ids),
                       lambda s: f"{get(s, 'name')}（{get(s, 'role')}）：{get(s, 'note')}")
    terms = _listed([*listed(context_pack, "entities", ids), *listed(context_pack, "terminology", ids)],
                    lambda item: f"{get(item, 'name')} → {get(item, 'zh')}（{get(item, 'type')}）：{get(item, 'note')}"
                    if "name" in item else f"{get(item, 'source')} → {get(item, 'zh')}：{get(item, 'rule')}")
    risks = _listed(listed(context_pack, "asr_risks", ids),
                    lambda r: f"{get(r, 'heard_as')} → {get(r, 'should_be')}：{get(r, 'reason')}")
    window_context = window_context_of(context_pack, ids)
    neighbours = lambda indices: "\n".join(  # noqa: E731
        f"{encode_text_cell(lines[index].corrected)}|{encode_text_cell(lines[index].translation)}" for index in indices)
    user = load_prompt_template(
        USER_TEMPLATE, strict=True, title=title, summary=summary_of(context_pack, ids) or NOTHING, speakers=speakers,
        terms=terms, risks=risks, notes=_listed(_plain(context_pack, "streamer_notes", ids), str),
        uncertain=_listed(_plain(context_pack, "uncertainties", ids), str),
        window_context=load_prompt_template(WINDOW_CONTEXT_TEMPLATE, strict=True, context=window_context).strip()
        + "\n\n" if window_context else "",
        user_requests=_user_requests(context_pack, style, window_context),
        style=style.strip(),
        kb_terms=load_prompt_template(KB_TERMS_TEMPLATE, strict=True, entries=kb_terms.strip())
        if kb_terms.strip() else "",
        before=load_prompt_template(BEFORE_TEMPLATE, strict=True, rows=neighbours(window.before))
        if window.before else "",
        after=load_prompt_template(AFTER_TEMPLATE, strict=True, rows=neighbours(window.after))
        if window.after else "",
        rows="\n".join(line_cells(number, lines[index]) for number, index in enumerate(window.rows, start=1)),
        count=len(window.rows),
    )
    return system_prompt(), user


def window_context_of(context_pack: Any, source_ids: Sequence[str]) -> str:
    """The preparation task's background for the window holding `source_ids`, picked as the correction picks
    it (`ContextPack.window_context_for_source_ids`), or empty. The range runs over every source of the window's
    lines, where a pass 2 grouped window's body names only each group's first line: when the last group crosses a
    note's edge (research windows not those of this run) pass 3 may pick nothing the correction did. Rare and
    light, so left as it is."""
    if context_pack is None or not getattr(context_pack, "window_contexts", None):
        return ""
    return context_pack.window_context_for_source_ids(source_ids).strip()


def _user_requests(context_pack: Any, style: str, window_context: str) -> str:
    """The block of what the preparation task relayed from the user: the requests for every window, and the
    window's own background where it marks some as the user's (empty without either), ranked below
    `--extra-style` where the style carries it (`style_text` puts it first)."""
    requests = getattr(context_pack, "user_requests", ())
    in_window = USER_REQUEST_MARK in window_context
    if not requests and not in_window:
        return ""
    words = prompt_phrases("proofread_requests")
    if not in_window:
        head = [words["general_head"], _listed(requests, str)]
    else:
        head = [words["head"], *([words["general_part"], _listed(requests, str)] if requests else []),
                words["window_part"]]
    extra_lead = load_prompt_template(EXTRA_STYLE_TEMPLATE, strict=True, extra="").strip()
    return load_prompt_template(
        USER_REQUESTS_TEMPLATE, strict=True, requests_head="\n".join(head) + "\n",
        style_rank=prompt_phrases("user_requests")["proofread_style_rank"] if style.strip().startswith(extra_lead) else "",
    ).strip() + "\n\n"


def system_prompt() -> str:
    """Pass 3's system message. Its style rule always applies: a task without a user style has the factory
    default (owner 2026-10-31)."""
    return load_prompt_template(SYSTEM_TEMPLATE, strict=True)


def entry_terms(entries: Mapping[str, str]) -> str:
    """The names and terms of knowledge entries (key -> the body the correction was shown): each entry's term
    lines without their descriptions, under its headings (`knowledge.base.term_lines_only`)."""
    from ...knowledge.base import term_lines_only

    return "\n".join(filter(None, (term_lines_only(body).strip() for body in entries.values())))
