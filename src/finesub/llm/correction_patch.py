"""Repair a rejected correction answer with a patch instead of a rewrite.

A repair round used to ask for the whole window again. When the rejected answer is mostly right, the
model now sends only a ``<patch>`` block: rows in the `<translated>` format for the sources that need
fixing. The harness lays the patch over the rows of the rejected answer that passed validation and hands
the resulting whole answer to the usual validator, so nothing downstream ever sees a patch.

One rule decides what a patch replaces: every source id a patch row names takes the base unit holding it
out, whole. A unit is a row that passed validation together with any ``#`` line right above it (no prompt
asks for those since capableC dropped its reasoning comments; one a model still writes goes with its row).
Units are keyed by the source ids the model wrote, never by line numbers.

Ids here are the model's window-local ones (``1..N``), as written in the reply; the caller's validation
result is the remapped one (`validate_correction_window_output`), whose turn rows this module maps back.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ..text import looks_escaped
from .chunking import SubtitleWindow
from .exchange_metadata import extract_top_level_tagged_blocks
from .output_protocol import (
    KIND_DISCARD,
    KIND_SUB,
    LEAKED_THOUGHT_ERROR,
    LEAKED_THOUGHT_PREFIX,
    CsvValidationResult,
    correction_output_header,
    row_kind,
)
from .prompt_compose import load_prompt_template
from .prompt_variants import CorrectionVariant, grouped_variant

PATCH_TAG = "patch"
PATCH_INSTRUCTION_FRAGMENT = "fragment_repair_patch_v1.md"
# Below this share of the window covered by rows that passed, the answer is
# a systematic format error and a patch would list most of the window anyway -- the round asks for a rewrite.
PATCH_MIN_COVERAGE = 0.5

_TAG_NAME_RE = re.compile(r"<([A-Za-z_][\w-]*)\s*>")


@dataclass(frozen=True)
class _Unit:
    lines: tuple[str, ...]
    ids: tuple[str, ...]


@dataclass(frozen=True)
class RepairBase:
    """A rejected answer a patch can be laid over."""

    text: str
    block: str
    units: tuple[_Unit, ...]
    pending: tuple[str, ...]
    header: str
    source_count: int


@dataclass(frozen=True)
class PatchMerge:
    """What laying a reply over a base gave.

    ``mode`` is ``patch`` (a patch was merged into ``text``), ``full`` (the reply rewrote the whole
    `<translated>` and is ``text`` as it came) or ``invalid`` (the patch could not be used: ``text`` is
    empty, ``errors`` say why, and the next round repairs the same base)."""

    mode: str
    text: str = ""
    errors: tuple[str, ...] = ()
    patch_units: int = 0
    replaced_units: int = 0


def _row_ids(row: str) -> tuple[str, ...]:
    """The source ids in a row's position column (the second one for both `sub` and `discard`)."""

    parts = row.split("|")
    if len(parts) < 2:
        return ()
    return tuple(part.strip() for part in parts[1].split(",") if part.strip())


def repair_base(
    text: str,
    result: CsvValidationResult,
    window: SubtitleWindow,
    variant: CorrectionVariant,
) -> RepairBase | None:
    """The base a patch round repairs, or ``None`` when this answer goes back for a rewrite.

    A patch is offered when the answer has one `<translated>` block,
    every error belongs to a row, to the missing sources or to a shift (`serial_shift`; the shifted rows go
    back as if they had failed), rows that passed cover at least
    :data:`PATCH_MIN_COVERAGE` of the window, and some source needs fixing. The caller checks what this
    cannot see: that the call's cell has no ``variant_overrides``. A grouped window's base is in the variant's
    grouped set, as its validation was (`output_protocol.validate_correction_output_text`).
    """

    if window.grouped:
        variant = grouped_variant(variant)
    blocks = extract_top_level_tagged_blocks(text or "", "translated")
    if len(blocks) != 1 or not blocks[0] or text.count(blocks[0]) != 1:
        return None
    local = tuple(str(index) for index in range(1, len(window.source_ids) + 1))
    to_local = {source_id: str(index) for index, source_id in enumerate(window.source_ids, start=1)}
    shifted = {to_local[source_id] for source_id in result.shifted_sources if source_id in to_local}
    units: list[_Unit] = []
    comments: list[str] = []
    failed: set[str] = set()
    for verdict in result.rows:
        if verdict.kind == "comment":
            comments.append(verdict.raw)
            continue
        if verdict.kind == "plan":
            continue
        if verdict.kind in (KIND_SUB, KIND_DISCARD):
            if verdict.ok:
                units.append(_Unit((*comments, verdict.raw), _row_ids(verdict.raw)))
            else:
                failed.update(_row_ids(verdict.raw))
        comments = []
    covered = {source_id for unit in units for source_id in unit.ids}
    uncovered = [source_id for source_id in local if source_id not in covered]
    attributed = sum(len(verdict.errors) for verdict in result.rows) + (1 if uncovered else 0) + (1 if shifted else 0)
    if attributed != len(result.errors):
        return None
    # Shifted rows passed every row check but go back like failed ones.
    failed.update(source_id for unit in units if shifted.intersection(unit.ids) for source_id in unit.ids)
    units = [unit for unit in units if not shifted.intersection(unit.ids)]
    covered = {source_id for unit in units for source_id in unit.ids}
    if len(covered) < PATCH_MIN_COVERAGE * len(local):
        return None
    turned = {to_local[source_id] for ids in result.turn_violation_rows for source_id in ids if source_id in to_local}
    pending = (set(uncovered) | failed | turned) & set(local)
    if not pending:
        return None
    return RepairBase(
        text=text,
        block=blocks[0],
        units=tuple(units),
        pending=tuple(sorted(pending, key=int)),
        header=correction_output_header(variant),
        source_count=len(local),
    )


def patch_instruction(base: RepairBase, validation_errors: Sequence[str]) -> str:
    """The repair round's user turn for a patch: the errors, the sources to fix and the patch rules."""

    errors = [text for text in (str(item).strip() for item in validation_errors) if text]
    return load_prompt_template(
        PATCH_INSTRUCTION_FRAGMENT,
        strict=True,
        validation_errors="\n".join(f"- {text}" for text in errors),
        pending_sources="、".join(base.pending),
        output_csv_header=base.header,
    ).strip()


def rows_to_fix(base: RepairBase) -> str:
    """The base's rows that hold a source still to fix, as they stand: what a tool session that only sent a
    patch is shown of the whole it made (a missing source has no row)."""

    pending = set(base.pending)
    return "\n".join(
        line for line in base.block.splitlines()
        if not line.lstrip().startswith("#") and pending.intersection(_row_ids(line))
    )


def apply_patch(base: RepairBase, reply: str) -> PatchMerge:
    """Lay ``reply`` over ``base``; see :class:`PatchMerge` for the three outcomes."""

    patches = extract_top_level_tagged_blocks(reply or "", PATCH_TAG)
    rewrites = extract_top_level_tagged_blocks(reply or "", "translated")
    if patches and rewrites:
        return PatchMerge("invalid", errors=(
            "The reply holds both a <patch> and a <translated> block; send only the <patch>.",
        ))
    if rewrites:
        return PatchMerge("full", text=reply)
    if len(patches) != 1:
        # As on a whole answer (`validate_correction_window_output`): a leaked reasoning reply says why first.
        leaked = (LEAKED_THOUGHT_ERROR,) if (reply or "").lstrip().startswith(LEAKED_THOUGHT_PREFIX) else ()
        return PatchMerge("invalid", errors=(
            *leaked, "The reply must hold exactly one top-level <patch>...</patch> block.",
        ))
    if looks_escaped(patches[0]):
        # Laid over a base with real non-ASCII text, the whole would no longer look escaped and the validator's
        # refusal (`output_protocol._refused_reply`) would let the literal escapes through into the subtitles.
        return PatchMerge("invalid", errors=(
            "The <patch> is pure ASCII but carries literal escapes decoding to non-ASCII text: a transport fault, "
            "not a content one. Send it again with the characters as they are.",
        ))

    local = {str(index) for index in range(1, base.source_count + 1)}
    rows = [row.strip() for row in patches[0].splitlines() if row.strip()]
    if rows and rows[0].lower().startswith("type|position"):
        rows = rows[1:]
    units: list[_Unit] = []
    comments: list[str] = []
    errors: list[str] = []
    for number, row in enumerate(rows, start=1):
        kind = row_kind(row)
        if kind == "comment":
            comments.append(row)
            continue
        if kind == "plan":
            continue
        if kind == "void":
            comments = []
            continue
        ids = _row_ids(row)
        outside = [source_id for source_id in ids if source_id not in local]
        if not ids:
            errors.append(f"<patch> row {number} has no source ids.")
        elif outside:
            errors.append(f"<patch> row {number} names source ids outside this window: {', '.join(outside)}.")
        else:
            units.append(_Unit((*comments, row), ids))
        comments = []
    if not units and not errors:
        errors.append("<patch> has no rows.")
    if errors:
        return PatchMerge("invalid", errors=tuple(errors))

    touched = {source_id for unit in units for source_id in unit.ids}
    kept = [unit for unit in base.units if not touched.intersection(unit.ids)]
    merged = sorted([*kept, *units], key=lambda unit: int(unit.ids[0]))
    body = "\n".join([base.header, *(line for unit in merged for line in unit.lines)])
    text = base.text.replace(base.block, body, 1)
    return PatchMerge(
        "patch",
        text=_carry_other_blocks(text, reply),
        patch_units=len(units),
        replaced_units=len(base.units) - len(kept),
    )


def _carry_other_blocks(text: str, reply: str) -> str:
    """Blocks the reply wrote besides the patch replace the base's of the same name, or are appended."""

    for name in dict.fromkeys(match.group(1) for match in _TAG_NAME_RE.finditer(reply)):
        if name.lower() in {PATCH_TAG, "translated"}:
            continue
        bodies = extract_top_level_tagged_blocks(reply, name)
        if len(bodies) != 1:
            continue
        old = extract_top_level_tagged_blocks(text, name)
        if len(old) == 1 and old[0] and text.count(old[0]) == 1:
            text = text.replace(old[0], bodies[0], 1)
        else:
            text = f"{text.rstrip()}\n<{name}>\n{bodies[0]}\n</{name}>\n"
    return text
