"""Repair a refused judge reply with a patch instead of a whole new answer: the correction's patch round
(`correction_patch`), for the judge's JSON lines.

When the reply's `<judge>` block has something to keep, the repair round asks only for the rest -- the rows
missing, not JSON, not an object, given twice, or whose `track` is not among the answers: a
``<patch>`` of lines in the `<judge>` format. Every id a patch line names takes the base's row with that id
out; the harness lays the patch over the rows that answered and reads the whole again, so nothing
downstream sees a patch. A `<tracks>` block in the reply adds or replaces names line by line. A reply a patch
cannot help (`repair_base` is None) is the caller's to ask again whole (`run.py`), with `whole_instruction`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ...correction_patch import PATCH_TAG, PatchMerge
from ...exchange_metadata import extract_top_level_tagged_blocks
from ...output_protocol import LEAKED_THOUGHT_ERROR, LEAKED_THOUGHT_PREFIX
from ...prompt_compose import load_prompt_template
from .answers import BLOCK, collapsed, line_id, survey

#: The repair turn asking for a patch, and the one asking for the whole answer again.
PATCH_TEMPLATE = "speaker_judge_repair_patch_v1.md"
WHOLE_TEMPLATE = "speaker_judge_repair_whole_v1.md"


#: A patch's faulty lines, each kind reported once (`answers.collapsed`).
NO_ID = ("<patch> line names no row id", "<patch> lines name no row id")
OUTSIDE = ("<patch> line names an id outside this window", "<patch> lines name ids outside this window")


@dataclass(frozen=True)
class RepairBase:
    """A refused reply a patch can be laid over: ``text`` as it was read, its rows that answered (``rows``:
    `(id, line)` in window order), the ids to answer again (``pending``, in window order) and the lines those
    stand on now (``faulty``; a missing id has none)."""

    text: str
    ids: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]
    pending: tuple[str, ...]
    faulty: tuple[tuple[str, str], ...]


def repair_base(text: str, ids: Sequence[str], tracks: Sequence[int]) -> RepairBase | None:
    """The base a patch round repairs, or None when this reply goes back whole: it has no `<judge>` block,
    nothing it names is left to fix (only stray ids or rows whose id cannot be told), or no row in it answers.
    A row whose id cannot be told leaves that id missing, so the patch asks for it. ``tracks`` the ones the
    answer may name (`answers.survey`)."""
    read = survey(text, ids, tracks)
    order = {key: index for index, key in enumerate(ids)}
    pending = sorted(set(read.pending), key=order.__getitem__)
    if not read.found or not pending or not read.rows:
        return None
    return RepairBase(
        text=text, ids=tuple(ids),
        rows=tuple((key, read.lines[key]) for key in sorted(read.rows, key=order.__getitem__)),
        pending=tuple(pending),
        faulty=tuple((key, read.lines[key]) for key in pending if key in read.lines),
    )


def _error_list(errors: Sequence[str]) -> str:
    return "\n".join(f"- {text}" for text in (str(item).strip() for item in errors) if text)


def patch_instruction(base: RepairBase, errors: Sequence[str]) -> str:
    """The patch round's user turn: the errors, the ids to answer again, and the patch rules."""
    return load_prompt_template(
        PATCH_TEMPLATE, strict=True,
        validation_errors=_error_list(errors),
        pending_ids="、".join(base.pending),
    ).strip()


def whole_instruction(errors: Sequence[str]) -> str:
    """The user turn asking for the whole answer again with ``errors`` fixed, in the judge's own format (the
    client's generic round speaks of a header and columns the judge does not have)."""
    return load_prompt_template(WHOLE_TEMPLATE, strict=True, validation_errors=_error_list(errors)).strip()


def rows_to_fix(base: RepairBase) -> str:
    """The base's lines for the ids still to fix, as they stand: what a tool session that only sent a patch is
    shown of the whole it made (a missing id has no line)."""
    return "\n".join(line for _key, line in base.faulty)


def apply_patch(base: RepairBase, reply: str) -> PatchMerge:
    """Lay ``reply`` over ``base``: ``patch`` (merged into ``text``), ``full`` (the reply wrote a whole
    `<judge>` and is ``text`` as it came) or ``invalid`` (``errors`` say why; the next round repairs the same
    base). A patch line's content is not checked here: the whole it makes is read as any reply is."""
    patches = extract_top_level_tagged_blocks(reply or "", PATCH_TAG)
    wholes = extract_top_level_tagged_blocks(reply or "", "judge")
    if patches and wholes:
        return PatchMerge("invalid", errors=(
            "The reply holds both a <patch> and a <judge> block; send only the <patch>.",))
    if wholes:
        return PatchMerge("full", text=reply)
    if len(patches) != 1:
        leaked = (LEAKED_THOUGHT_ERROR,) if (reply or "").lstrip().startswith(LEAKED_THOUGHT_PREFIX) else ()
        return PatchMerge("invalid", errors=(
            *leaked, "The reply must hold exactly one top-level <patch>...</patch> block."))
    wanted = set(base.ids)
    lines: list[tuple[str, str]] = []
    no_id: list[str] = []
    outside: list[str] = []
    for number, line in enumerate((line.strip() for line in patches[0].splitlines() if line.strip()), start=1):
        key = line_id(line)
        if key is None:
            no_id.append(f"line {number}")
        elif key not in wanted:
            outside.append(repr(key))
        else:
            lines.append((key, line))
    errors = [*([collapsed(NO_ID, no_id)] if no_id else []), *([collapsed(OUTSIDE, outside)] if outside else [])]
    if not lines and not errors:
        errors.append("<patch> has no rows.")
    if errors:
        return PatchMerge("invalid", errors=tuple(errors))
    touched = {key for key, _line in lines}
    kept = [(key, line) for key, line in base.rows if key not in touched]
    order = {key: index for index, key in enumerate(base.ids)}
    merged = sorted([*kept, *lines], key=lambda item: order[item[0]])  # stable: a repeated patch id stays repeated
    match = re.search(BLOCK.format("judge"), base.text, re.S)
    assert match is not None  # a base has its block
    body = "\n" + "\n".join(line for _key, line in merged) + "\n"
    text = base.text[:match.start(1)] + body + base.text[match.end(1):]
    return PatchMerge("patch", text=_merge_tracks(text, reply), patch_units=len(lines),
                      replaced_units=len(base.rows) - len(kept))


def _merge_tracks(text: str, reply: str) -> str:
    """The reply's `<tracks>` lines replace the base's lines for the same track or label, or are added."""
    new = re.search(BLOCK.format("tracks"), reply or "", re.S)
    if new is None:
        return text
    given = [line.strip() for line in new.group(1).splitlines() if "=" in line]
    if not given:
        return text
    old = re.search(BLOCK.format("tracks"), text, re.S)
    if old is None:
        return text.rstrip() + "\n<tracks>\n" + "\n".join(given) + "\n</tracks>\n"
    names = {line.split("=", 1)[0].strip(): line for line in given}
    lines = [line.strip() for line in old.group(1).splitlines() if line.strip()]
    kept = [names.pop(line.split("=", 1)[0].strip(), line) if "=" in line else line for line in lines]
    body = "\n" + "\n".join([*kept, *names.values()]) + "\n"
    return text[:old.start(1)] + body + text[old.end(1):]
