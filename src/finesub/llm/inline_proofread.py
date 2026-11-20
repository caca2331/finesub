"""The correction window's own proofread turn (乙 a′ of the intermediate one-pass plan; `--llm-proofread-inline`).

Once a window's answer passes validation, the same conversation is asked once more, before the answer is
committed: proofread it as a finished subtitle by pass 3's criteria (`stages.proofread.prompt.criteria`) and send
only a ``<patch>`` that changes text. The harness lays the patch over the answer the way a patch repair round
does (`correction_patch.apply_patch`), checks that only the two text columns changed, and validates the whole
again. A reply that cannot be laid over (no single ``<patch>``, a row that changes more than text, a whole that
fails validation) gets one repair round (`repair_instruction`: the errors, and the whole patch again; owner Q16);
when that reply fails too, the answer stays as it was (the draft), so the turn can never cost a window. Only the
patch is taken from a reply: the draft's other blocks (`<next_advice>`, `<keep_entries>`, `<task_update>`) stand.

Where it runs (`next_step` is the one rule for what follows a reply):
- API and capsule: `stages.correction.attempts` sends the window's base prompt, the draft as the assistant turn
  and `instruction` as the user turn (the repair round's shape, `prompt_compose.compose_repair_turns`); the repair
  round adds the failed reply and `repair_instruction` after them. A capsule that keeps the window's conversation
  continues it with the instruction alone (`client._run_local_agent`).
- A tool session: the task runtime's `submit` answers the passing answer with `instruction` instead of accepting
  it, and takes the next submit as the proofread reply (`agent_task_runtime`, `agent_validators`); whatever ends
  the session before that reply is accepted leaves the draft.

The instruction points at rows a script flags (reading speed past pass 3's line, a prepared name missing from the
translation); the flags are hints, not the scope (step 0 of the plan: about four in ten of pass 3's changes are on
rows no script flags).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from ..subtitles.metrics import weighted_char_count
from .chunking import SubtitleWindow
from .correction_patch import PATCH_TAG, RepairBase, _row_ids, _Unit, apply_patch
from .exchange_metadata import extract_top_level_tagged_blocks
from .output_protocol import KIND_SUB, CsvValidationResult, correction_output_header, row_kind
from .prompt_compose import load_prompt_template
from .prompt_tables import prompt_phrases
from .prompt_variants import CorrectionVariant, for_window, grouped_variant

INSTRUCTION_TEMPLATE = "fragment_proofread_inline_v1.md"
REPAIR_TEMPLATE = "fragment_proofread_inline_repair_v1.md"
#: Pass 3's reading speed line (its criteria, item 2): characters per second past this on a line of at least
#: `CPS_MIN_CHARS` weighted characters.
CPS_LIMIT = 9.0
CPS_MIN_CHARS = 10.0
_FORM_SPLIT = re.compile(r"[／/、]")

#: What became of the turn: the patch went in, the model had nothing to change, the draft stands, or the turn was
#: not asked (a cell whose candidates may answer in another prompt set: a patch over the draft could not be read).
APPLIED, NO_CHANGE, FALLBACK, SKIPPED = "applied", "no_change", "fallback", "skipped"
_DRAFT_FAILS = "the draft is not a single passing <translated> answer"
#: The patch made the translations slide off their rows where the draft's had not (`translation_shift`, which only
#: warns): the draft stands, with no repair round -- a turn that may not fail the window should not pass one on
#: either (owner 2026-11-11).
_SHIFTED = "the patch made the translations slide off their rows"


@dataclass(frozen=True)
class Change:
    position: str
    before: tuple[str, str]
    after: tuple[str, str]
    note: str


@dataclass(frozen=True)
class Outcome:
    """``text`` is what the window commits: the patched answer when ``status`` is `APPLIED`, else the draft."""

    status: str
    text: str
    changes: tuple[Change, ...] = ()
    errors: tuple[str, ...] = ()
    validation: Any = None
    flagged: tuple[str, ...] = field(default_factory=tuple)

    @property
    def repairable(self) -> bool:
        """The reply could not be laid over a passing draft: worth the one repair round."""
        return self.status == FALLBACK and not (self.errors and self.errors[0] in (_DRAFT_FAILS, _SHIFTED))

    def to_record(self, window: SubtitleWindow) -> dict[str, Any]:
        """The turn as the window's records keep it (`correction_window_proofread`); a change's ``source_ids`` are
        its position as source ids."""
        ids = list(window.source_ids)

        def sources(position: str) -> list[str]:
            return [ids[int(part) - 1] for part in position.split(",") if part.strip().isdigit()
                    and 0 < int(part) <= len(ids)]

        return {"status": self.status, "errors": list(self.errors), "flagged": list(self.flagged),
                "changes": [{"position": change.position, "source_ids": sources(change.position),
                             "before": list(change.before), "after": list(change.after), "note": change.note}
                            for change in self.changes]}


def _forms(text: str) -> list[str]:
    return [part.strip() for part in _FORM_SPLIT.split(text or "") if len(part.strip()) >= 2]


def names_table(general_context: Mapping[str, Any]) -> list[tuple[list[str], list[str]]]:
    """The prepared names (`entities`, `terminology`) as (source forms, Chinese forms); a context prepared in chunks
    (one block per chunk id, `general_context`) gives every chunk's."""

    blocks = [general_context] if {"entities", "terminology"} & set(general_context) else [
        block for block in general_context.values() if isinstance(block, Mapping)]
    table = []
    for block in blocks:
        for entry in list(block.get("entities") or []) + list(block.get("terminology") or []):
            if not isinstance(entry, Mapping):
                continue
            source = _forms(str(entry.get("name") or entry.get("source") or ""))
            chinese = _forms(str(entry.get("zh") or ""))
            if source and chinese and (source, chinese) not in table:
                table.append((source, chinese))
    return table


def flagged_rows(result: CsvValidationResult, window: SubtitleWindow,
                 names: Sequence[tuple[list[str], list[str]]]) -> list[str]:
    """One line per flagged row of a passing answer, in the model's own position ids. Reading speed is taken on
    the row's span before splicing and tail extension, so it flags a few rows more than pass 3 would see."""

    local = {source_id: str(index) for index, source_id in enumerate(window.source_ids, start=1)}
    lines = []
    for segment in result.segments:
        if segment.kind != KIND_SUB:
            continue
        position = ",".join(local.get(source_id, source_id) for source_id in segment.source_ids)
        chars = weighted_char_count(segment.translation or "")
        seconds = max(segment.end - segment.start, 1e-3)
        if chars >= CPS_MIN_CHARS and chars / seconds > CPS_LIMIT:
            lines.append(f"- {position}（读速）：跨度 {seconds:.1f} 秒、{chars:g} 个字，{chars / seconds:.1f} 字/秒")
        for source, chinese in names:
            hit = next((form for form in source if form in (segment.corrected_text or "")), None)
            if hit and not any(form in (segment.translation or "") for form in chinese):
                lines.append(f"- {position}（专名）：纠错文里有「{hit}」，资料的译法「{'／'.join(chinese)}」不在译文里")
    return lines


def _header(variant: CorrectionVariant, window: SubtitleWindow) -> str:
    return correction_output_header(grouped_variant(variant) if window.grouped else for_window(variant, window))


def _copied_keys(header: str) -> list[str]:
    """The columns a patch row copies from the draft row besides `type` and `position`: `conf`, and `spk` where the
    window names who speaks (`capableC-tracked`)."""
    return [key for key in header.split("|") if key not in ("type", "position", "note", *_text_keys(header))]


def _copied(header: str) -> str:
    return "、".join(f"`{key}`" for key in _copied_keys(header))


def instruction(variant: CorrectionVariant, window: SubtitleWindow, flagged: Sequence[str], *,
                tool_session: bool = False) -> str:
    """The proofread turn's user message; ``tool_session``: the reply to a `submit`, whose task came as the
    `protocol` and `payload` documents rather than a system and a user message."""

    from .stages.proofread.prompt import criteria

    header = _header(variant, window)
    where = prompt_phrases("proofread_inline")["tool_session" if tool_session else "api"]
    return load_prompt_template(
        INSTRUCTION_TEMPLATE,
        strict=True,
        instructions=where["instructions"],
        first_message=where["first_message"],
        criteria=criteria(),
        flagged="\n".join(flagged) if flagged else "（这一窗没有点名的行）",
        output_csv_header=header,
        copied=_copied(header),
        speaker_kept=prompt_phrases("proofread_inline")["speaker_kept"] if "spk" in _copied_keys(header) else "",
    ).strip()


def repair_instruction(variant: CorrectionVariant, window: SubtitleWindow, errors: Sequence[str]) -> str:
    """The repair round's user message, after the reply `apply` could not lay over the draft."""

    header = _header(variant, window)
    return load_prompt_template(
        REPAIR_TEMPLATE,
        strict=True,
        errors="\n".join(f"- {error}" for error in errors),
        output_csv_header=header,
        copied=_copied(header),
    ).strip()


def draft_base(text: str, result: CsvValidationResult, window: SubtitleWindow,
               variant: CorrectionVariant) -> RepairBase | None:
    """A passing answer as a base a patch can be laid over (every row kept, nothing pending), or None when the
    answer does not hold exactly one `<translated>` block."""

    blocks = extract_top_level_tagged_blocks(text or "", "translated")
    if len(blocks) != 1 or not blocks[0] or text.count(blocks[0]) != 1:
        return None
    units, comments = [], []
    for verdict in result.rows:
        if verdict.kind == "comment":
            comments.append(verdict.raw)
            continue
        if verdict.kind in ("sub", "discard"):
            units.append(_Unit((*comments, verdict.raw), _row_ids(verdict.raw)))
        comments = []
    return RepairBase(text=text, block=blocks[0], units=tuple(units), pending=(),
                      header=_header(variant, window), source_count=len(window.source_ids))


def _patch_rows(reply: str) -> list[str] | None:
    patches = extract_top_level_tagged_blocks(reply or "", PATCH_TAG)
    if len(patches) != 1:
        return None
    rows = [row.strip() for row in patches[0].splitlines() if row.strip()]
    return [row for row in rows if not row.lower().startswith("type|position") and row_kind(row) not in ("comment", "plan")]


def text_only_errors(base: RepairBase, rows: Sequence[str]) -> list[str]:
    """Why the patch rows change more than text: a row that is not `sub`, whose position is not exactly the
    position of a `sub` row of the draft, whose text is emptied (pass 3's criteria rule that out too; the
    window's validator lets an empty cell through), or whose `spk` or `conf` is not the draft row's (the window's
    validator would let another legal speaker through, and the turn is not the place to re-judge who speaks)."""

    positions, drafted = {}, {}
    for unit in base.units:
        raw = unit.lines[-1]
        positions[unit.ids] = row_kind(raw)
        drafted[unit.ids] = _cells(raw, base.header)
    errors = []
    for number, row in enumerate(rows, start=1):
        ids = _row_ids(row)
        if row_kind(row) != KIND_SUB:
            errors.append(f"<patch> row {number} is not a `sub` row: the proofread turn changes text only.")
        elif positions.get(ids) != KIND_SUB:
            errors.append(f"<patch> row {number} has position {','.join(ids)}, which is not the position of a row "
                          "in the answer: copy the row's position as it was.")
        elif any(not _cells(row, base.header).get(key, "").strip() for key in _text_keys(base.header)):
            errors.append(f"<patch> row {number} empties a text cell: a row keeps its text.")
        else:
            cells = _cells(row, base.header)
            for key in _copied_keys(base.header):
                old, new = drafted[ids].get(key, "").strip(), cells.get(key, "").strip()
                if new != old:
                    errors.append(f"<patch> row {number} changes `{key}` from `{old}` to `{new}`: the proofread turn "
                                  f"changes text only; copy `{key}` as it was.")
    return errors


def _text_keys(header: str) -> list[str]:
    return [key for key in header.split("|") if key in ("corrected_text", "translation", "polished")]


def _cells(row: str, header: str) -> dict[str, str]:
    names = header.split("|")
    parts = row.split("|", len(names) - 1)
    return dict(zip(names, parts))


def apply(draft: str, reply: str, window: SubtitleWindow, variant: CorrectionVariant,
          validate) -> Outcome:  # noqa: ANN001 - validate(text) -> CsvValidationResult
    """Lay the proofread reply over the passing ``draft``; ``validate`` is the window's validator
    (`output_protocol.validate_correction_window_output` bound to the window and variant)."""

    draft_result = validate(draft)
    base = draft_base(draft, draft_result, window, variant)
    if base is None or not draft_result.ok:
        return Outcome(FALLBACK, draft, errors=(_DRAFT_FAILS,))
    rows = _patch_rows(reply)
    if rows is None and (reply or "").strip() == draft.strip():
        # the draft sent back whole: nothing to change (a tool session's agent may answer so)
        return Outcome(NO_CHANGE, draft)
    if rows is None:
        return Outcome(FALLBACK, draft, errors=("the reply must hold exactly one top-level <patch> block",))
    if not rows:
        return Outcome(NO_CHANGE, draft)
    errors = text_only_errors(base, rows)
    if errors:
        return Outcome(FALLBACK, draft, errors=tuple(errors))
    # the patch alone: a block the reply wrote besides it would replace the draft's (`correction_patch`)
    merge = apply_patch(base, f"<{PATCH_TAG}>\n{extract_top_level_tagged_blocks(reply, PATCH_TAG)[0]}\n</{PATCH_TAG}>")
    if merge.mode != "patch":
        return Outcome(FALLBACK, draft, errors=merge.errors or (f"patch merge gave {merge.mode}",))
    result = validate(merge.text)
    if not result.ok:
        return Outcome(FALLBACK, draft, errors=tuple(result.errors), validation=result)
    shift = result.translation_shift
    if shift is not None and draft_result.translation_shift is None:
        return Outcome(FALLBACK, draft,
                       errors=(_SHIFTED, f"source rows {shift.start}–{shift.end}, offset {shift.offset:+d}"))
    before = {unit.ids: _cells(unit.lines[-1], base.header) for unit in base.units}
    changes = []
    for row in rows:
        old, new = before[_row_ids(row)], _cells(row, base.header)
        changes.append(Change(",".join(_row_ids(row)), tuple(old.get(key, "") for key in _text_keys(base.header)),
                              tuple(new.get(key, "") for key in _text_keys(base.header)), new.get("note", "")))
    return Outcome(APPLIED, merge.text, tuple(changes), validation=result)


def next_step(draft: str, reply: str, window: SubtitleWindow, variant: CorrectionVariant, validate, *,
              round_: int, flagged: Sequence[str] = ()) -> tuple[Outcome, str]:  # noqa: ANN001
    """What follows proofread reply number ``round_`` (0: the answer to `instruction`): its outcome, and the repair
    instruction when one more round is due (the first reply, when it could not be laid over; owner Q16), else
    "". The one rule both the attempts loop and the tool-session runtime follow."""

    outcome = replace(apply(draft, reply, window, variant, validate), flagged=tuple(flagged))
    if outcome.repairable and round_ == 0:
        return outcome, repair_instruction(variant, window, outcome.errors)
    return outcome, ""
