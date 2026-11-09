"""The correction round's output protocol: CSV-like block parsing, validation
and SRT reconstruction.

This is the one contract `session_contract` does not cover -- a correction
window is judged by what this parser accepts (schema, coverage, adjacency),
not by which tags came back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Callable, List, Mapping, Optional, Sequence

from .chunking import Seam, SubtitleSegment, SubtitleWindow, WindowIdMap
from .exchange_metadata import extract_top_level_tagged_blocks
from .output_tags import last_complete_answer
from .prompt_constants import Thresholds
from .prompt_variants import CorrectionVariant, grouped_variant
from .serial_shift import find_shift
from .text_cells import decode_text_cell, encode_text_cell
from finesub.subtitles import line_groups
from finesub.text import looks_escaped
from finesub.subtitles.model import SrtSegment, render_srt
from finesub.subtitles.metrics import (
    format_weighted_char_count,
    weighted_char_count,
)


# Row kinds in the output CSV `type` column. "" and "sub" both mean the default
# behavior (merge/correct/translate anchored on source ids); "insert" is a
# model-proposed subtitle for audio the source ASR missed, positioned by its own
# clip-relative (start, duration) instead of source ids. "plan" is a retired
# v33 row kind that remains ignored for compatibility with old replay output.
KIND_SUB = "sub"
KIND_INSERT = "insert"
KIND_PLAN = "plan"
KIND_DISCARD = "discard"
CONFIDENCE_LEVELS = frozenset({"high", "median", "low"})

# A reply may discard sources, but a window that keeps almost none of itself is
# a failure wearing a valid shape. The 2026-08-22 canary: an agent that never
# saw the window text answered with one `sub` row plus `discard` for everything
# else; every structural check passed and the finished subtitle kept one line.
#
# The rule (owner 2026-09-30): reject when the sources *kept* (not discarded)
# number at most `min(KEPT_FLOOR, total * 0.5 - KEPT_FLOOR)`. On a window of
# 40+ sources that is "10 or fewer kept"; it shrinks toward zero below that,
# and at 20 or fewer it never fires on a reply that kept anything -- what is
# left there is the all-discard check ("Translated CSV contains no valid rows").
# It replaced a plain 50% discard ratio, which had no small-sample guard (a
# one-minute clip that is mostly song correctly discards most of six lines)
# and could not be applied to a split leaf at all (the worst correct half in
# the archive discards 43.8%, `bench-baselines.md` 二十五). An absolute floor
# has neither problem, so it applies to every window, leaves included.
#
# ⚠ Still a **declared gap**: a reply that never saw the body but keeps more
# than the floor passes. Closing it needs a *different* signal (did the model
# receive the window text at all), not a different number -- filed in
# `llm_followups.md`.
KEPT_FLOOR = 10

# End-of-line marker (v12) letting the model retract a row it already wrote
# (e.g. it realized the merge span ran away).
# A marked row is treated as if the physical line does not exist: no structure
# validation, its source ids stay unclaimed so follow-up rows may re-emit them.
VOID_ROW_MARKER = "<void>"


def _uncovered_sources_error(
    expected_ids: Sequence[str], covered: set[str]
) -> str | None:
    """Every window source must appear in a row or be explicitly discarded.

    Silent omission is no longer allowed (v52)."""

    uncovered = [sid for sid in expected_ids if sid not in covered]
    if not uncovered:
        return None
    preview = ", ".join(uncovered[:12])
    return (
        f"Translated missing source id(s): {preview}"
        + ("…" if len(uncovered) > 12 else "")
        + ". Every source must be covered by a sub row or "
        "explicitly discarded with 'discard|<id>'."
    )


def _too_few_kept_error(discarded_ids: set[str], expected_ids: Sequence[str]) -> str | None:
    """Coverage says every source was *accounted for*; this says the window
    still produced subtitles. See KEPT_FLOOR for the rule."""

    total = len(expected_ids)
    kept = total - len(discarded_ids)
    floor = min(KEPT_FLOOR, total * 0.5 - KEPT_FLOOR)
    if kept > floor:
        return None
    return (
        f"Translated keeps {kept} of {total} sources and discards the rest, "
        f"at or under the floor of {floor:g} kept. Discard is for individual "
        "sources that carry no speech; a window where almost every source is "
        "dropped means the window text was not read."
    )


def _row_is_voided(row: str) -> bool:
    """Whether a translated row is self-retracted with the void marker.

    The contract puts ``<void>`` at the row end, but models sometimes drop it
    into a column (typically ``conf``) with trailing cells after it. Treat the
    marker as a void wherever it appears as a whole pipe-delimited field, in
    addition to the canonical trailing form."""

    lowered = row.lower().rstrip()
    if lowered.endswith(VOID_ROW_MARKER):
        return True
    return any(field.strip() == VOID_ROW_MARKER for field in lowered.split("|"))
# Soft merged-source cap per translated row. The prompt still instructs the
# model to keep judgment merges to two consecutive sources (the gap-adaptive
# upstream already pre-merges same-sentence fragments), but validation no longer
# rejects over-cap rows (relaxed 2026-07-20) — it only records a warning. Merged
# sources must still be adjacent (that stays a hard error, for timeline sanity).
TRANSLATED_MAX_MERGED_SOURCES = 2


@dataclass(frozen=True)
class TranslatedCsvSegment:
    source_ids: tuple[str, ...]
    start: float
    end: float
    corrected_text: str
    translation: str
    kind: str = KIND_SUB
    # Model self-reported confidence tier; None when absent/invalid (advisory,
    # never fails a row). Numeric v38 values are mapped for compatibility.
    conf: Optional[str] = None
    note: str = ""

    @property
    def text(self) -> str:
        return self.translation


@dataclass(frozen=True)
class RowVerdict:
    """What the validator made of one non-empty `<translated>` line after the header.

    ``kind`` is ``sub``, ``discard``, ``comment`` (a ``#`` line), ``void``
    or ``plan``; ``errors`` are the messages this line added to the result's
    ``errors``, so a line passed when it has none. ``raw`` is the stripped line
    as the model wrote it, in its window-local ids."""

    number: int
    raw: str
    kind: str
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def row_kind(row: str) -> str:
    """The verdict kind of a row, in the order the validator tells them apart."""

    if _row_is_voided(row):
        return "void"
    prefix = row.split("|", 1)[0].strip().lower()
    if prefix == KIND_PLAN:
        return "plan"
    if prefix == KIND_DISCARD:
        return KIND_DISCARD
    if row.startswith("#"):
        return "comment"
    return KIND_SUB


@dataclass(frozen=True)
class CsvValidationResult:
    ok: bool
    segments: List[TranslatedCsvSegment]
    errors: List[str]
    warnings: List[str]
    # Rows the model retracted with VOID_ROW_MARKER (observability for how
    # often the self-abort channel actually gets used).
    voided_rows: int = 0
    # Source ids explicitly discarded by the model via ``discard|<ids>`` rows
    # (hallucination/invalid content). Together with segment source_ids these
    # must cover the full window; uncovered ids are a validation error.
    discarded_ids: tuple[str, ...] = ()
    # Source ids (model-local before remapping, source ids after) of each row
    # merging a line into the one before it across a speaker turn
    # (`SubtitleSegment.seam`), or, in three passes' pass 2, across any seam
    # not marked 可并. Not an error and not part of ``ok``: such a
    # merge is accepted with a warning, never sent back (owner 2026-10-16).
    turn_violation_rows: tuple[tuple[str, ...], ...] = ()
    # The same rows as notes for the model, in its local ids: when the window
    # goes back for a structural error anyway, they ride along on the repair.
    turn_notes: tuple[str, ...] = ()
    # Sources of the rows that slid off their own sources (`serial_shift`), model-local before remapping,
    # source ids after: the stretch a repair has to redo. The error saying so is in ``errors`` and belongs to
    # no row, so this is how a patch repair knows which rows to send back.
    shifted_sources: tuple[str, ...] = ()
    # One verdict per `<translated>` line after the header, in order: what a
    # patch repair (`correction_patch`) keeps of a rejected answer and which
    # sources it sends back. Empty when the block itself is missing.
    rows: tuple[RowVerdict, ...] = ()


def _parse_conf(cell: str) -> Optional[str]:
    """Parse v39 confidence tiers, mapping v38 integers for compatibility."""

    value = (cell or "").strip().lower()
    if not value:
        return None
    if value in CONFIDENCE_LEVELS:
        return value
    try:
        legacy = int(value)
    except ValueError:
        return None
    if not 1 <= legacy <= 9:
        return None
    if legacy >= 7:
        return "high"
    if legacy >= 4:
        return "median"
    return "low"


@dataclass(frozen=True)
class _RowFields:
    kind: str
    position: str
    corrected: str
    translation: str
    conf: str
    note: str
    start: str = ""
    #: Product artifact only (`<stem>-annotated.csv`): computed when it is
    #: written, never present in a model row.
    duration: str = ""
    gap: str = ""
    #: Product artifact only, and empty when the run had no diarization.
    spk: str = ""


def _is_numeric_cell(cell: str) -> bool:
    try:
        float((cell or "").strip())
    except ValueError:
        return False
    return bool((cell or "").strip())


_CHAR_COUNT_RE = re.compile(r"\d+(?:\.\d+)?(?:\+\d+(?:\.\d+)?)*")


def _is_char_count_cell(cell: str) -> bool:
    """Return whether a char-count cell is ``N`` or a per-line ``N+N`` sum."""

    return bool(_CHAR_COUNT_RE.fullmatch((cell or "").strip()))


def _window_errors(
    errors: List[str],
    segments: List["TranslatedCsvSegment"],
    expected_ids: Sequence[str],
    covered: set[str],
    discarded_ids: set[str],
) -> List[str]:
    """The errors that are about the window as a whole rather than any row.

    `errors` is read, never appended to: the empty-window message is
    suppressed when the block already failed to parse, because "no valid rows"
    adds nothing to a reader who has just been told the CSV was unreadable.
    """

    empty = (
        "Translated CSV contains no valid rows."
        if not segments and not any(e.startswith("Translated CSV") for e in errors)
        else ""
    )
    return [
        message
        for message in (
            empty,
            _uncovered_sources_error(expected_ids, covered),
            _too_few_kept_error(discarded_ids, expected_ids),
        )
        if message
    ]


#: The output row contract, in order; a start-bearing set (basicA) has ``start``
#: right after ``position``. Every cell is split by position and a row must have
#: exactly these cells, so a raw `|` is never allowed in a text cell (the prompt
#: asks for the full-width `｜`).
_ROW_COLUMNS = ("kind", "position", "corrected", "translation", "conf", "note")


def output_row_spec(*, start: bool = False) -> tuple[str, ...]:
    """The output row's columns in order (internal names)."""

    return (*_ROW_COLUMNS[:2], "start", *_ROW_COLUMNS[2:]) if start else _ROW_COLUMNS


#: The second column's name in a correct-only task's header (`CorrectionVariant.correct_only`): a light polish
#: of the corrected text, never called a translation in front of the model.
POLISHED_COLUMN = "polished"


def output_csv_header(*, start: bool = False, polished: bool = False) -> str:
    """The header rows with these columns carry; ``polished`` names the second text column `polished`."""

    names = {"kind": "type", "corrected": "corrected_text", **({"translation": POLISHED_COLUMN} if polished else {})}
    return "|".join(names.get(name, name) for name in output_row_spec(start=start))


def correction_output_header(variant: CorrectionVariant) -> str:
    """The header of ``variant``'s `<translated>` block."""

    return output_csv_header(start=variant.output_has_start, polished=variant.correct_only)


#: The *product* artifact's columns -- `<stem>-annotated.csv`, not what the model
#: emits. Besides the model's cells it carries `duration`, `gap` and `char_count`
#: (computed when the file is written, `render_translated_segments_as_csv`) and
#: `spk`, holding whichever speaker tracks the line's source segments fell in
#: (empty when the run had no diarization).
#:
#: `spk` sits **before** `note` so `note` stays last: the split stops before the
#: last cell, which keeps a stray `|` in a note from shifting the cells before
#: it, and anything after `note` would be swallowed. The model never emits this column --
#: speakers are decided long after the correction round, by a different layer.
_ANNOTATED_ROW_COLUMNS = (
    "kind",
    "position",
    "duration",
    "gap",
    "corrected",
    "translation",
    "conf",
    "char_count",
    "spk",
    "note",
)

#: Written at the top of every `<stem>-annotated.csv`, and the single definition
#: of that file's layout. Readers split by position, so this and
#: `_ANNOTATED_ROW_COLUMNS` have to move together.
ANNOTATED_CSV_HEADER = "# " + "|".join(
    "type" if name == "kind" else name for name in _ANNOTATED_ROW_COLUMNS
)


def split_annotated_row(row: str) -> Optional[_RowFields]:
    """Map one `<stem>-annotated.csv` row onto its columns, strictly by position.

    Separate from `_split_row_fields` because the product file carries columns
    the model's output does not. Sharing one splitter would mean either asking
    the model for columns it cannot know, or letting a row shift a cell to the
    left in silence -- the failure the positional split exists to prevent.
    No reader uses the `char_count` cell, so it is not mapped.
    """

    parts = row.split("|", len(_ANNOTATED_ROW_COLUMNS) - 1)
    if len(parts) < len(_ANNOTATED_ROW_COLUMNS):
        return None
    cells = dict(zip(_ANNOTATED_ROW_COLUMNS, parts))
    return _RowFields(
        kind=cells["kind"],
        position=cells["position"],
        corrected=cells["corrected"],
        translation=cells["translation"],
        conf=cells["conf"],
        note=cells["note"],
        duration=cells["duration"],
        gap=cells["gap"],
        spk=cells["spk"],
    )


def _split_row_fields(row: str, *, has_start_column: bool = False) -> Optional[_RowFields]:
    """Map one output row onto the column spec strictly by position.

    ``type|position|corrected|translation|conf|note`` (with ``start`` after
    ``position`` for the basic tier). Apart from ``start``, no cell after
    ``position`` has to be numeric, so a row carrying one cell too many or too
    few would pass with every cell one place off: the row must have exactly the
    spec's cells, and a raw `|` in ``note`` is not tolerated (owner 2026-10-23).

    This used to infer the layout from cell *contents*, so a row carrying one
    extra column was silently shifted a cell to the left and the untranslated
    source text ended up in ``translation`` with no error at all. Layout
    guessing is gone with the pre-v39 tolerances it existed for.

    Returns ``None`` when the row has any other count of cells than the spec's.
    """

    spec = output_row_spec(start=has_start_column)
    parts = row.split("|")
    if len(parts) != len(spec):
        return None
    cells = dict(zip(spec, parts))
    return _RowFields(
        kind=cells["kind"],
        position=cells["position"],
        corrected=cells["corrected"],
        translation=cells["translation"],
        conf=cells["conf"],
        note=cells["note"],
        start=cells.get("start", ""),
    )


def _carries_start_and_guide_columns(parts: Sequence[str]) -> bool:
    """Whether a row's cells unambiguously read as ``start`` plus the guide columns (ten cells)."""

    return (
        len(parts) >= 10
        and parts[0].strip().lower() in {"", KIND_SUB, KIND_INSERT}
        and _is_numeric_cell(parts[2])
        and _is_numeric_cell(parts[3])
        and _is_gap_cell(parts[4])
        and parts[7].strip().lower() in CONFIDENCE_LEVELS
        and _is_char_count_cell(parts[8])
    )


def _carries_duration_after_start(parts: Sequence[str], cells: int) -> bool:
    """Whether a row of a ``start``-bearing layout has one cell too many and it is a number right after
    ``start``: the input's ``duration`` copied over (3.5 Flash Lite repeats this on every row and every retry
    when the error only counts cells)."""

    return (
        len(parts) == cells + 1
        and parts[0].strip().lower() in {"", KIND_SUB, KIND_INSERT}
        and _is_numeric_cell(parts[2])
        and _is_numeric_cell(parts[3])
        and parts[cells - 1].strip().lower() in CONFIDENCE_LEVELS
    )


def _is_gap_cell(cell: str) -> bool:
    """A copied ``gap``: a number, or empty on the window's first row, which has no row before it."""

    return not cell.strip() or _is_numeric_cell(cell)


def _carries_guide_columns(parts: Sequence[str]) -> bool:
    """Whether a row's cells unambiguously read as the guide columns without ``start`` (nine cells)."""

    return (
        len(parts) >= 9
        and parts[0].strip().lower() in {"", KIND_SUB}
        and _is_numeric_cell(parts[2])
        and _is_gap_cell(parts[3])
        and parts[6].strip().lower() in CONFIDENCE_LEVELS
        and _is_char_count_cell(parts[7])
    )


def _long_merge_errors(
    row_number: int, start: float, end: float, translation: str, limits: Thresholds
) -> List[str]:
    """A merged row past the absolute threshold: its span from ``start`` to ``end`` -- the cue it becomes, at
    the 0.1 s the window shows -- or its translation's weighted characters, the same bounds the prompt states."""

    span = round(end - start, 1)
    chars = weighted_char_count(translation)
    if span <= limits.absolute_seconds and chars <= limits.absolute_chars:
        return []
    return [
        f"Row {row_number} merges sources into {span:.1f}s and {format_weighted_char_count(chars)} weighted chars, "
        f"past the absolute limit for a merged row ({limits.absolute_seconds}s / {limits.absolute_chars} chars); "
        "split it into rows within the limit."
    ]


def looks_truncated_translated(text: str) -> bool:
    text = text or ""
    if "<translated" in text.lower() and not re.search(
        r"</translated\s*>", text, flags=re.IGNORECASE
    ):
        return True
    return False


def _refused_reply(
    text: str, require_start_column: bool, forbid_start_column: bool
) -> Optional[CsvValidationResult]:
    """Whether this call can proceed at all, before anything is parsed.

    Two preconditions that answer in two different ways, deliberately. The
    column flags contradicting each other is a *caller* bug and raises; a reply
    whose non-ASCII arrived as literal escapes is a *transport* fault and comes
    back as a refusal, because it is the model's turn that failed, not ours.

    The refusal has to sit here, ahead of every other check, because all of
    them read this text: coverage, adjacency, the length limits and scoring
    would each have something to say about characters the model never wrote.
    The one measured case went the whole way -- into the SRT and into the
    resume cache -- with the stage reporting success; the only check that saw
    it then was a comparison with the model's own `char_count` column, which
    no output contract carries any more.

    `agent_mcp_server` repairs this shape upstream, so a refusal here means the
    repair missed a *new* shape. That is exactly when a loud failure beats a
    quiet one, and it is why both exist (plan: decision one).
    """

    if require_start_column and forbid_start_column:
        raise ValueError(
            "require_start_column and forbid_start_column cannot both be true."
        )
    if not looks_escaped(text):
        return None
    return CsvValidationResult(
        ok=False,
        segments=[],
        errors=[
            "The reply is pure ASCII but carries literal escapes decoding to "
            f"non-ASCII text ({len(text)} chars, {text.count(chr(92))} "
            "backslashes). That is a transport fault rather than a content "
            "one: the CLI escaped its own tool arguments."
        ],
        warnings=[],
    )


def _discard_row(
    row_number: int,
    row: str,
    source_by_id: Mapping[str, SubtitleSegment],
    source_index: Mapping[str, int],
    seen_source_ids: set[str],
    discarded_ids: set[str],
    previous_last_position: int,
) -> tuple[tuple[str, ...], List[int], List[str]]:
    """Check one ``discard|<source_ids>[|reason]`` row: its ids, their window positions, and its errors."""

    parts = [p.strip() for p in row.split("|")]
    if len(parts) < 2 or not parts[1]:
        return (), [], [f"Row {row_number}: discard row needs at least 'discard|<source_ids>'."]
    disc_ids = tuple(sid.strip() for sid in parts[1].split(",") if sid.strip())
    if not disc_ids:
        return (), [], [f"Row {row_number}: discard row has no source ids."]
    row_errors: List[str] = []
    positions: List[int] = []
    local_seen: set[str] = set()
    for disc_id in disc_ids:
        if disc_id in local_seen:
            row_errors.append(f"Row {row_number}: discard repeats source id {disc_id}.")
            continue
        local_seen.add(disc_id)
        if disc_id not in source_by_id:
            row_errors.append(f"Row {row_number}: discard references unknown source id {disc_id}.")
        elif disc_id in seen_source_ids:
            row_errors.append(f"Row {row_number}: source id {disc_id} already covered by a sub row; cannot discard.")
        elif disc_id in discarded_ids:
            row_errors.append(f"Row {row_number}: source id {disc_id} discarded more than once.")
        else:
            positions.append(source_index[disc_id])
    if positions and positions != sorted(positions):
        row_errors.append(f"Row {row_number}: discard source ids must be in source order.")
    if len(positions) > 1 and any(b - a != 1 for a, b in zip(positions, positions[1:])):
        row_errors.append(f"Row {row_number}: discard source ids must be consecutive.")
    if positions and positions[0] <= previous_last_position:
        row_errors.append(f"Row {row_number} appears before an earlier output row.")
    return disc_ids, positions, row_errors


def validate_translated_csv_text(
    text: str,
    source_segments: Sequence[SubtitleSegment],
    *,
    clip_start: float = 0.0,
    require_headers: bool = False,
    require_start_column: bool = False,
    forbid_start_column: bool = False,
    polished: bool = False,
    max_merged_sources: int = TRANSLATED_MAX_MERGED_SOURCES,
    long_merge_limits: Thresholds | None = None,
) -> CsvValidationResult:
    """Validate and parse the `<translated>` block into typed segments.

    Uses **top-level** `<translated>` extraction (sibling blocks only): mid-prose
    tag name-drops inside `<reasoning>` do not count.

    When `require_headers` is true (v55+ production and replay contract), the
    first non-empty line of the block must be the variant's CSV header. The default remains false so historical saved
    replies stay parseable for offline audits.

    When ``forbid_start_column`` is true (every set without ``start``), a row whose
    cells unambiguously read as ``start`` plus the guide columns is named as such
    (in the row loop, where the error belongs to its row). It is mutually exclusive
    with ``require_start_column``.

    Every `sub` row must have exactly the header's cells (see `_split_row_fields`);
    a row that unambiguously carries the guide columns ``duration|gap|char_count``
    (asked for until csv-v104; the input still carries ``duration`` and ``gap``)
    is named as such.

    ``polished``: a correct-only task's two columns (`CorrectionVariant.correct_only`); the header names the
    second one `polished`, and it may be empty (a line that is only filler, kept out of the subtitle). The
    length checks then measure the corrected text.

    ``max_merged_sources``: how many sources a row merges before it is warned about
    (a soft cap, never an error; the variant's, `CorrectionVariant.max_merged_sources`).
    ``long_merge_limits``: given, a row merging sources whose span is over their
    ``absolute_seconds`` (at the 0.1 s the window shows) or whose translation is over their
    ``absolute_chars`` weighted characters fails and goes back for repair (owner Q13,
    2026-10-30: the single pass capableC, at the window's thresholds); a single source past
    them passes. None rejects no merge for its length.

    `clip_start` is the absolute time of the window clip's 0 second; `insert`
    rows carry clip-relative `(start, duration)` that is shifted by it into the
    original timeline. `type`/`conf`/`note` are advisory: an absent/invalid
    `conf` degrades to `None` (no error) and an unknown `type` degrades to
    `sub`; legacy `plan` lines (`plan|…`, removed from prompts in v34) are
    still skipped silently if present and never become segments. Only structural
    problems (bad ids/order, empty text) fail a row
    and trigger a retry upstream.

    `insert` rows are always a structural error (v63 retired the row kind);
    `KIND_INSERT` itself lives on because knowledge materials still read the
    type back out of older `annotated.csv`.

    Rows ending with :data:VOID_ROW_MARKER are self-retracted by the model
    (v12): they are dropped before any structural checks, counted in
    `voided_rows`, and their source ids remain free for later rows.
    """

    errors: List[str] = []
    warnings: List[str] = []

    refusal = _refused_reply(text, require_start_column, forbid_start_column)
    if refusal is not None:
        return refusal

    source_by_id: dict[str, SubtitleSegment] = {}
    source_index: dict[str, int] = {}
    for idx, segment in enumerate(source_segments):
        if segment.id in source_by_id:
            errors.append(f"Source id {segment.id} appears more than once in the window.")
            continue
        source_by_id[segment.id] = segment
        source_index[segment.id] = idx
    if errors:
        return CsvValidationResult(ok=False, segments=[], errors=errors, warnings=warnings)

    expected_ids = [segment.id for segment in source_segments]
    expected_header = output_csv_header(start=require_start_column, polished=polished)
    text_cells = f"corrected_text, {POLISHED_COLUMN if polished else 'translation'} or note"
    translated_blocks = extract_top_level_tagged_blocks(text or "", "translated")
    if len(translated_blocks) != 1:
        errors.append(
            "Output must contain exactly one top-level "
            "<translated>...</translated> block."
        )
        return CsvValidationResult(ok=False, segments=[], errors=errors, warnings=warnings)

    translated_segments: List[TranslatedCsvSegment] = []
    seen_source_ids: set[str] = set()
    discarded_ids: set[str] = set()
    previous_last_position = -1
    voided_rows = 0

    payload = translated_blocks[0]
    rows = [row.strip() for row in payload.splitlines() if row.strip()]
    if rows and rows[0] == expected_header:
        rows = rows[1:]
    elif require_headers:
        errors.append(
            "<translated> first non-empty line must be the exact CSV header: "
            f"{expected_header}"
        )
    if not rows and not expected_ids:
        # Nothing to cover, nothing to report. Only this case short-circuits:
        # a window that *has* sources must fall through so the v52 coverage
        # check and the "no valid rows" check below can speak.
        #
        # Emptiness used to be read as an intentional "wipe this window", which
        # made a truncated reply indistinguishable from a deliberate one and
        # let it delete rows earlier windows had already produced. Dropping
        # sources is what `discard|<id>` is for, and the coverage error already
        # says so; a window that yields no rows at all is a failure either way
        # (see test_discarding_every_source_is_rejected_too).
        return CsvValidationResult(
            ok=not errors, segments=[], errors=errors, warnings=warnings
        )

    # Where each row's errors start in `errors`: a row's own errors run up to
    # the next row's start (every row-level error is appended while its row is
    # current, whichever `continue` it leaves by).
    row_starts: List[int] = []
    for row_number, row in enumerate(rows, start=1):
        row_starts.append(len(errors))
        if _row_is_voided(row):
            voided_rows += 1
            warnings.append(f"Row {row_number} retracted by the model (<void>).")
            continue
        # Planning lines: free text after the first pipe; not 7-column rows.
        kind_prefix = row.split("|", 1)[0].strip().lower()
        if kind_prefix == KIND_PLAN:
            continue
        # Explicit discard rows: ``discard|<source_ids>[|reason]``. The model
        # uses these to declare hallucination/invalid sources instead of
        # silently omitting them (v52 contract).
        if kind_prefix == KIND_DISCARD:
            disc_ids, positions, row_errors = _discard_row(
                row_number, row, source_by_id, source_index, seen_source_ids, discarded_ids, previous_last_position
            )
            errors.extend(row_errors)
            if not row_errors:
                discarded_ids.update(disc_ids)
                previous_last_position = positions[-1]
            continue
        # ``#`` lines: once capableC's reasoning comments above the row they
        # explained; no prompt asks for them now, and one a model still
        # writes carries no source ids: skipped, not failed.
        if row.startswith("#"):
            continue
        fields = _split_row_fields(row, has_start_column=require_start_column)
        if fields is None:
            parts = row.split("|")
            if forbid_start_column and _carries_start_and_guide_columns(parts):
                what = "unexpected start, duration, gap and char_count columns"
            elif require_start_column and _carries_start_and_guide_columns(parts):
                what = "unexpected duration, gap and char_count columns"
            elif not require_start_column and _carries_guide_columns(parts):
                what = "unexpected duration, gap and char_count columns"
            elif require_start_column and _carries_duration_after_start(parts, expected_header.count("|") + 1):
                what = (f"an extra cell after start (`{parts[3].strip()}`, the input's duration copied over; "
                        "write start only)")
            else:
                what = ""
            errors.append(
                f"Row {row_number} has {what}; expected {expected_header}."
                if what
                else f"Row {row_number} has {len(parts)} cells; every row has exactly "
                f"{expected_header.count('|') + 1} ({expected_header}). A `|` inside {text_cells} "
                "must be the full-width `｜`."
            )
            continue

        if require_start_column and not _is_numeric_cell(fields.start):
            errors.append(
                f"Row {row_number} must carry numeric start seconds as column 3."
            )
            continue

        kind = fields.kind.strip().lower() or KIND_SUB
        if kind == KIND_PLAN:
            continue
        if kind not in (KIND_SUB, KIND_INSERT):
            warnings.append(
                f"Row {row_number} has unknown type '{fields.kind.strip()}'; treated as sub."
            )
            kind = KIND_SUB
        conf = _parse_conf(fields.conf)
        if fields.conf.strip() and conf is None:
            # Stays advisory (see TranslatedCsvSegment.conf): a junk conf never
            # fails a row on its own. A drifted row is caught upstream by the
            # cell count.
            warnings.append(
                f"Row {row_number} has invalid conf '{fields.conf.strip()}'; dropped."
            )
        note = decode_text_cell(fields.note)

        if kind == KIND_INSERT:
            # v63 abolished inserts: every production caller had already pinned
            # allow_insert=False, so the emit/dedup/merge machinery behind this
            # branch was unreachable. `KIND_INSERT` itself stays -- knowledge
            # materials still read the type back out of older annotated.csv.
            errors.append(
                f"Row {row_number} uses type=insert, which is no longer part "
                "of the output contract."
            )
            continue

        row_errors: List[str] = []
        source_ids = tuple(
            part.strip() for part in fields.position.split(",") if part.strip()
        )
        if not source_ids:
            row_errors.append(f"Row {row_number} has no source ids.")
        # EXPERIMENT (relaxed 2026-07-20): the merged-source cap no longer
        # rejects. Over-cap merges are recorded as warnings only, so the models'
        # natural merge behavior can be observed without validation-forced
        # retries. Adjacency/order checks below stay hard (timeline integrity).
        if len(source_ids) > max_merged_sources:
            warnings.append(
                f"Row {row_number} merges {len(source_ids)} source ids "
                f"(soft cap {max_merged_sources}; not rejected)."
            )

        local_seen: set[str] = set()
        positions: List[int] = []
        for source_id in source_ids:
            if source_id in local_seen:
                row_errors.append(f"Row {row_number} repeats source id {source_id}.")
                continue
            local_seen.add(source_id)
            if source_id not in source_by_id:
                row_errors.append(
                    f"Row {row_number} references unknown source id {source_id}."
                )
                continue
            if source_id in seen_source_ids:
                row_errors.append(
                    f"Source id {source_id} appears in more than one output row."
                )
                continue
            if source_id in discarded_ids:
                row_errors.append(
                    f"Source id {source_id} was already discarded; it cannot also "
                    "appear in a sub row."
                )
                continue
            positions.append(source_index[source_id])

        if positions and positions != sorted(positions):
            row_errors.append(f"Row {row_number} source ids must be in source order.")
        if len(positions) > 1 and any(
            b - a != 1 for a, b in zip(positions, positions[1:])
        ):
            row_errors.append(
                f"Row {row_number} merges non-consecutive source ids; merged "
                "sources must be adjacent."
            )
        if positions and positions[0] <= previous_last_position:
            row_errors.append(f"Row {row_number} appears before an earlier output row.")

        corrected_text = decode_text_cell(fields.corrected)
        if not corrected_text.strip():
            row_errors.append(f"Row {row_number} has empty corrected text.")
        translation = decode_text_cell(fields.translation)
        if not polished and not translation.strip():
            row_errors.append(f"Row {row_number} has empty translation text.")
        # what the row's length is measured by: an empty polished cell leaves the corrected text
        shown_text = translation if translation.strip() else corrected_text

        if row_errors:
            errors.extend(row_errors)
            continue

        assert positions
        first = source_segments[positions[0]]
        # the last source's end, or later where a source of another speaker ends later
        end = line_groups.run_end(source_segments[positions[0]:positions[-1] + 1])
        if long_merge_limits is not None and len(positions) > 1:
            long_merge = _long_merge_errors(row_number, first.start, end, shown_text, long_merge_limits)
            if long_merge:
                errors.extend(long_merge)
                continue

        translated_segments.append(
            TranslatedCsvSegment(
                source_ids=source_ids,
                start=first.start,
                end=end,
                corrected_text=corrected_text,
                translation=translation,
                kind=KIND_SUB,
                conf=conf,
                note=note,
            )
        )
        seen_source_ids.update(source_ids)
        previous_last_position = positions[-1]

    ends = [*row_starts[1:], len(errors)]
    verdicts = tuple(
        RowVerdict(number, row, row_kind(row), tuple(errors[start:end]))
        for number, (row, start, end) in enumerate(zip(rows, row_starts, ends), start=1)
    )
    errors.extend(
        _window_errors(
            errors,
            translated_segments,
            expected_ids,
            seen_source_ids | discarded_ids,
            discarded_ids,
        )
    )

    return CsvValidationResult(
        ok=not errors,
        segments=translated_segments,
        errors=errors,
        warnings=warnings,
        voided_rows=voided_rows,
        discarded_ids=tuple(sorted(discarded_ids, key=lambda s: source_index.get(s, 0))),
        rows=verdicts,
    )


#: How a reply opens when the model shipped its reasoning as the answer: agy's
#: `result.response` starts with this literal (docs/llm_local_agent_agy.md §7,
#: about 4 in 26 calls on 2026-09-02).
LEAKED_THOUGHT_PREFIX = "thought\n"
LEAKED_THOUGHT_ERROR = (
    "The previous reply was your reasoning, not the answer. Return only the "
    "answer, in the required blocks."
)


def strip_leaked_thought(text: str, required: Sequence[str]) -> str | None:
    """The answer a leaked-reasoning reply ends with: from its last complete answer on, the reasoning before it
    dropped. ``None`` for a reply that does not open with :data:`LEAKED_THOUGHT_PREFIX`, or one with no complete
    answer (`output_tags.last_complete_answer`); that reply is validated as it came and fails as before.

    Each task names its own ``required`` blocks. A leak does not cost the answer after it (owner 2026-10-17:
    on the leaks read, that answer validated and was as good as a retry), so it is validated like any reply.
    """

    if not (text or "").lstrip().startswith(LEAKED_THOUGHT_PREFIX):
        return None
    start = last_complete_answer(text, required)
    return None if start is None else text[start:]


#: The blocks a correction reply must hold, whatever the variant.
CORRECTION_ANSWER_TAGS: tuple[str, ...] = ("translated",)


def validate_correction_output_text(
    text: str,
    source_segments: Sequence[SubtitleSegment],
    *,
    variant: CorrectionVariant,
    thresholds: Thresholds,
    clip_start: float = 0.0,
    grouped: bool = False,
) -> CsvValidationResult:
    """Validate one correction reply against the exact served prompt variant.
    ``grouped``: three passes' pass 2 (`SubtitleWindow.grouped`), where only a 可并 seam may be merged over,
    and the set served is the variant's grouped one (`prompt_variants.grouped_variant`), whatever name the
    caller holds. ``thresholds``: the window's (`SubtitleWindow.length_scale`); a set that does not follow the
    ratio validates at the standard tier's (`CorrectionVariant.own_thresholds`), the ones its prompt stated.

    The variant owns which columns the CSV rows carry. Keeping that projection here prevents
    production, resume, and replay call sites from silently validating a reply
    against a different variant than the answering endpoint received.
    """

    if grouped:
        variant = grouped_variant(variant)
    thresholds = variant.own_thresholds(thresholds)
    result = validate_translated_csv_text(
        text,
        source_segments,
        clip_start=clip_start,
        require_headers=True,
        require_start_column=variant.output_has_start,
        forbid_start_column=not variant.output_has_start,
        polished=variant.correct_only,
        max_merged_sources=variant.max_merged_sources or TRANSLATED_MAX_MERGED_SOURCES,
        long_merge_limits=thresholds if variant.rejects_long_merges else None,
    )
    # A leaked reasoning reply fails on structure anyway; saying *why* first
    # points the repair round at the cause instead of at missing blocks. A
    # reply that passes keeps passing: reasoning outside the blocks is not
    # part of the product (owner 2026-10-01).
    if not result.ok and text.lstrip().startswith(LEAKED_THOUGHT_PREFIX):
        return replace(result, errors=[LEAKED_THOUGHT_ERROR, *result.errors])
    from .prompt_compose import load_prompt_template

    if result.ok:
        # Only once the structure holds: a placeholder answer is well-formed by construction, and a malformed
        # one already has its repair errors.
        result = _with_placeholder(result, source_segments)
    result = _with_shift(result, source_segments)
    rows = _turn_violation_rows(result.segments, source_segments, grouped=grouped)
    turn = {segment.id: _separates(segment, grouped) for segment in source_segments}
    seam = {segment.id: segment.seam for segment in source_segments}
    note = "fragment_turn_merge_note_grouped_v1.md" if grouped else "fragment_turn_merge_note_v1.md"

    def source(ids: tuple[str, ...]) -> str:
        return next((i for i in ids[1:] if turn.get(i)), ids[-1])

    return replace(
        result,
        turn_violation_rows=rows,
        turn_notes=tuple(
            load_prompt_template(
                note, ids=",".join(ids), source=source(ids),
                # which value the turn column shows there: 换人, 同时 or 换回 (one pass)
                seam=(seam.get(source(ids)) or Seam.CHANGE).value,
            ).strip()
            for ids in rows
        ),
    )


#: Cells a model writes when it probes the answer channel instead of answering. Nonoka X (its engine patch 0009)
#: saw an agy window whose real answers the tool guard kept refusing submit `text|text` on all 52 rows, and the
#: row checks took it: every subtitle of that window read "text".
PLACEHOLDER_CELLS = frozenset({"text", "test", "placeholder", "todo", "tbd"})
#: From this many subtitle rows on, one line repeated on every row is not a subtitle.
UNIFORM_ROWS_MIN = 5


def _with_placeholder(result: CsvValidationResult, source_segments: Sequence[SubtitleSegment]) -> CsvValidationResult:
    """Send back a well-formed answer that is not one: every subtitle row the same, or most of them a placeholder
    word. Judged on the whole window, never on one row -- a single short line may well be either -- and a window
    whose sources all say the same thing (a chant repeated) may well be answered by one line repeated. The error
    belongs to no row, so the repair round asks for a rewrite, not a patch (`correction_patch.repair_base`)."""

    cells = [(row.corrected_text.strip(), row.translation.strip()) for row in result.segments if row.kind == KIND_SUB]
    if not cells:
        return result
    from .prompt_compose import load_prompt_template

    sources_differ = len({segment.text.strip() for segment in source_segments}) > 1
    if len(cells) >= UNIFORM_ROWS_MIN and len(set(cells)) == 1 and sources_differ:
        error = load_prompt_template(
            "fragment_uniform_rows_error_v1.md", strict=True, rows=len(cells), sample=" | ".join(cells[0]),
        ).strip()
        return replace(result, ok=False, errors=[*result.errors, error])
    placeholders = sum(
        1 for corrected, translation in cells
        if corrected.casefold() in PLACEHOLDER_CELLS or translation.casefold() in PLACEHOLDER_CELLS
    )
    if placeholders * 2 > len(cells):
        error = load_prompt_template(
            "fragment_placeholder_rows_error_v1.md", strict=True, rows=len(cells), count=placeholders,
        ).strip()
        return replace(result, ok=False, errors=[*result.errors, error])
    return result


def _with_shift(result: CsvValidationResult, source_segments: Sequence[SubtitleSegment]) -> CsvValidationResult:
    """Send back an answer whose rows slid off their sources (`serial_shift`), naming where it starts and what
    the start source itself says."""

    text = {segment.id: segment.text for segment in source_segments}
    shift = find_shift(
        [(segment.source_ids, segment.corrected_text) for segment in result.segments if segment.kind == KIND_SUB],
        text,
    )
    if shift is None:
        return result
    from .prompt_compose import load_prompt_template

    error = load_prompt_template(
        "fragment_serial_shift_error_v1.md", strict=True, start=shift.start, rows=shift.rows, match=shift.match,
        sample=shift.sample, end=shift.sources[-1], own=text[shift.start],
    ).strip()
    return replace(result, ok=False, errors=[*result.errors, error], shifted_sources=shift.sources)


def _separates(segment: SubtitleSegment, grouped: bool) -> bool:
    """Whether no merge may join this line to the one before it."""

    return segment.seam is not Seam.MAY if grouped else segment.seam is not None and segment.seam.separates


def _turn_violation_rows(
    rows: Sequence[TranslatedCsvSegment], source_segments: Sequence[SubtitleSegment], *, grouped: bool = False
) -> tuple[tuple[str, ...], ...]:
    """Source ids of the rows merging lines a turn separates.

    For every two sources a row joins in turn, a turn on any line after the first, up to and including the
    second, separates them: with adjacent sources (all the validator allows today) that is the second's own
    mark, and a merge skipping lines -- which subtitles overlapping in time might one day need -- can only
    come out stricter, never let a change of speaker through."""

    position = {segment.id: index for index, segment in enumerate(source_segments)}
    turn = [_separates(segment, grouped) for segment in source_segments]
    out = []
    for row in rows:
        for earlier, later in zip(row.source_ids, row.source_ids[1:]):
            if earlier not in position or later not in position:
                continue
            low, high = sorted((position[earlier], position[later]))
            if any(turn[low + 1:high + 1]):
                out.append(row.source_ids)
                break
    return tuple(out)


def remap_validation_source_ids(
    result: CsvValidationResult,
    id_map: WindowIdMap,
) -> CsvValidationResult:
    """Restore model-local ids to canonical source ids after validation."""

    return replace(
        result,
        segments=[
            replace(
                segment,
                source_ids=tuple(
                    id_map.source_id_for_local(source_id)
                    for source_id in segment.source_ids
                ),
            )
            for segment in result.segments
        ],
        discarded_ids=tuple(
            id_map.source_id_for_local(source_id)
            for source_id in result.discarded_ids
        ),
        turn_violation_rows=tuple(
            tuple(id_map.source_id_for_local(source_id) for source_id in ids)
            for ids in result.turn_violation_rows
        ),
        shifted_sources=tuple(id_map.source_id_for_local(source_id) for source_id in result.shifted_sources),
    )


def validate_correction_window_output(
    text: str,
    window: SubtitleWindow,
    *,
    variant: CorrectionVariant,
) -> CsvValidationResult:
    """Validate local model positions, then restore harness source ids. The thresholds are the window's, taken
    first: a window no path wrote its task's length ratio into fails here, whatever the reply holds."""

    thresholds = Thresholds.at(window.length_scale)
    id_map = WindowIdMap.from_window(window)
    result = validate_correction_output_text(
        text,
        id_map.localize_segments(window.segments),
        variant=variant,
        thresholds=thresholds,
        clip_start=window.clip_start,
        grouped=window.grouped,
    )
    return remap_validation_source_ids(result, id_map)

# ---------------------------------------------------------------------------
# v15 pacing scorer (docs/llm_quality_iteration_v15_plan.md §1). Phase 1 is
# observation-only: scores land in the correction_window_response artifact to
# calibrate thresholds on real runs; rejection (with a rework cap of 2 and
# best-of-attempts fallback) is phase 2 and is NOT wired up yet.

# Step penalties (machine "sub" rows only — future long-annotation features
# would use a different row type, so artistic long subtitles are out of scope):
# span >10s/+1, >15s/+2, >25s/+4; lines 3/+1, 4/+2, 5+/+4; per-line chars over
# 20 weighted chars 0.1/char summed across lines, capped at 2. Row
# severity: >=4 critical, >=2 severe, >=1 minor. Window passes when
# total_penalty / (row_count + 5) <= 0.3 (≈ critical<=2% / severe<=3% /
# minor<=10%; the +5 keeps tiny windows from failing on one bad row).
PACING_PASS_RATIO = 0.3
# Test profile relaxes the pass ratio to 1.0: prompt iteration wants the
# problems surfaced in artifacts (and later NOT retried away), not hidden by
# rework luck; production keeps the strict ratio.
PACING_PASS_RATIO_TEST_PROFILE = 1.0


def score_translated_segments(
    segments: Sequence[TranslatedCsvSegment],
    *,
    pass_ratio: float = PACING_PASS_RATIO,
) -> dict:
    """Pacing/merge quality score for one window's parsed rows."""

    rows = []
    for segment in segments:
        span = max(0.0, segment.end - segment.start)
        text = (segment.translation or "").strip()
        lines = [line for line in text.split("\n")] if text else []
        span_penalty = 4 if span > 25 else 2 if span > 15 else 1 if span > 10 else 0
        n_lines = len(lines)
        line_penalty = 4 if n_lines >= 5 else 2 if n_lines == 4 else 1 if n_lines == 3 else 0
        char_excess = sum(
            max(0.0, weighted_char_count(line) - 20.0) for line in lines
        )
        char_penalty = min(2.0, 0.1 * char_excess)
        penalty = span_penalty + line_penalty + char_penalty
        rows.append(
            {
                "source_ids": list(segment.source_ids),
                "span_seconds": round(span, 2),
                "line_count": n_lines,
                "char_penalty": round(char_penalty, 2),
                "penalty": round(penalty, 2),
            }
        )
    penalties = [row["penalty"] for row in rows]
    total = sum(penalties)
    normalized = total / (len(rows) + 5)
    return {
        "rows": rows,
        "total_penalty": round(total, 2),
        "normalized_penalty": round(normalized, 4),
        "critical_rows": sum(1 for p in penalties if p >= 4),
        "severe_rows": sum(1 for p in penalties if 2 <= p < 4),
        "minor_rows": sum(1 for p in penalties if 1 <= p < 2),
        "pass_ratio": pass_ratio,
        "passed": normalized <= pass_ratio,
    }


def _row_position(segment: TranslatedCsvSegment) -> str:
    return ",".join(segment.source_ids)


def render_translated_segments_as_csv(
    segments: Sequence[TranslatedCsvSegment],
) -> str:
    """Render segments as the product file's rows, `spk` aside (`render_annotated_csv`).

    ``type|position|duration|gap|corrected|translation|conf|char_count|note``:
    duration, gap and char_count are computed here; the model writes none of
    them. Gap is the pause before the row, from the previous rendered
    subtitle's end (0.0 on an overlap, empty on the first row) -- the direction
    of the correction input's gap.
    """

    ordered = list(segments)
    lines = []
    for index, segment in enumerate(ordered):
        gap = "" if index == 0 else f"{max(0.0, segment.start - ordered[index - 1].end):.1f}"
        lines.append(
            "|".join(
                (
                    segment.kind,
                    _row_position(segment),
                    f"{max(0.0, segment.end - segment.start):.1f}",
                    gap,
                    encode_text_cell(segment.corrected_text),
                    encode_text_cell(segment.translation),
                    "" if segment.conf is None else str(segment.conf),
                    format_weighted_char_count(
                        weighted_char_count(segment.translation)
                    ),
                    encode_text_cell(segment.note),
                )
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def render_annotated_csv(
    segments: Sequence[TranslatedCsvSegment],
    speaker_cell_for: Callable[[Sequence[str]], str | None] | None = None,
) -> str:
    """The product artifact: the model's row plus a `spk` cell, with a header.

    `speaker_cell_for` turns the source ids a rendered line covers into its
    cell. Joining several sources' speakers is the caller's job, done once on
    the speakers themselves -- a finished cell is a rendering, and unioning
    renderings as strings lists one track twice (`s1?` and `s1`). Pass nothing
    (a run without diarization) and every `spk` cell is empty -- the column
    still exists, so one layout parses every file.
    """

    lines = [ANNOTATED_CSV_HEADER]
    rendered = render_translated_segments_as_csv(segments).splitlines()
    for segment, row in zip(segments, rendered):
        cell = (speaker_cell_for(segment.source_ids) or "") if speaker_cell_for else ""
        # `note` is the last cell and may hold a raw `|`, so insert before it
        # by splitting from the right exactly once.
        head, separator, note = row.rpartition("|")
        lines.append(f"{head}{separator}{cell}|{note}" if separator else row)
    return "\n".join(lines) + "\n"


def respeak_annotated_csv(text: str, speaker_cell_for: Callable[[Sequence[str]], str | None]) -> str:
    """The same `-annotated.csv` with every `spk` cell rendered again from its row's `position`.

    For a finished task whose speakers changed after the correction (the speaker judge ran
    later): the rows are the correction's, only the column the correction never wrote moves.
    A row that does not split into the layout, or whose cell `speaker_cell_for` cannot render
    (None), is left as it is."""

    out = []
    for line in text.splitlines():
        fields = None if line.startswith("#") or not line.strip() else split_annotated_row(line)
        if fields is None:
            out.append(line)
            continue
        parts = line.split("|", len(_ANNOTATED_ROW_COLUMNS) - 1)
        ids = [part for part in fields.position.split(",") if part]
        cell = speaker_cell_for(ids) if ids else ""
        if cell is not None:
            parts[_ANNOTATED_ROW_COLUMNS.index("spk")] = cell
        out.append("|".join(parts))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def merge_translated_csv_windows(
    existing_segments: Sequence[TranslatedCsvSegment],
    current_source_ids: Sequence[str],
    new_segments: Sequence[TranslatedCsvSegment],
    *,
    overlap_ids: Sequence[str],
) -> List[TranslatedCsvSegment]:
    """Merge a window's output into prior results: each id keeps the row of the window whose core holds it.

    A window is its overlap prefix (``overlap_ids``: the window before it also answered them) plus its core.
    The overlap keeps the earlier window's rows and the new window's rows wholly inside it are dropped, so the
    seam sits at the cut, which the planner puts at a sentence end or a pause
    (`chunking.is_reasonable_boundary`). A new row merged across the cut takes its overlap ids with it: the
    ids the new window takes over grow over every row, of either window, touching them, until no row inside
    the window is cut. An earlier row reaching back past the window's start is kept whole even then, and the
    ids it shares with a new row show twice -- a duplicate `window_merge_gaps` reports, where cutting a row
    would have lost text. Ids the new window took over and discarded stay dropped.
    """
    window = set(current_source_ids)
    taken = window - set(overlap_ids)
    existing, new = list(existing_segments), list(new_segments)
    while True:
        grown = set(taken)
        for segment in new:
            if grown & set(segment.source_ids):
                grown |= set(segment.source_ids)
        for segment in existing:
            ids = set(segment.source_ids)
            if grown & ids and ids <= window:
                grown |= ids
        if grown == taken:
            break
        taken = grown
    kept_old = [segment for segment in existing
                if not taken & set(segment.source_ids) or not set(segment.source_ids) <= window]
    kept_new = [segment for segment in new if taken & set(segment.source_ids)]
    return sorted(kept_old + kept_new, key=lambda segment: (segment.start, segment.end, segment.source_ids))


def window_merge_gaps(
    existing_segments: Sequence[TranslatedCsvSegment],
    current_source_ids: Sequence[str],
    new_segments: Sequence[TranslatedCsvSegment],
    merged_segments: Sequence[TranslatedCsvSegment],
    *,
    overlap_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The ids one `merge_translated_csv_windows` call lost or doubled: ``(missing, duplicated)``.

    Every id that had output must come out covered exactly once: the earlier windows' ids, and the new
    window's outside the overlap (an overlap id only the new window answered was discarded by the window
    whose core it is). Every id inside the window the new window had no row for is let go: the ones it took
    over go by design, and the check does not replay the closure to tell an overlap id it left with the earlier
    window, so losing one of those would slip past it. Duplicates an earlier merge already reported are not
    reported again.
    """
    answered = {source_id for segment in new_segments for source_id in segment.source_ids}
    discarded = set(current_source_ids) - answered
    overlap = set(overlap_ids)
    expected: dict[str, None] = {}
    for segment in existing_segments:
        expected.update(dict.fromkeys(i for i in segment.source_ids if i not in discarded))
    for segment in new_segments:
        expected.update(dict.fromkeys(i for i in segment.source_ids if i not in overlap))

    def doubled(segments: Sequence[TranslatedCsvSegment]) -> dict[str, None]:
        seen: set[str] = set()
        twice: dict[str, None] = {}
        for segment in segments:
            for source_id in segment.source_ids:
                if source_id in seen:
                    twice[source_id] = None
                seen.add(source_id)
        return twice

    covered = {source_id for segment in merged_segments for source_id in segment.source_ids}
    missing = tuple(source_id for source_id in expected if source_id not in covered)
    already = doubled(existing_segments)
    duplicated = tuple(source_id for source_id in doubled(merged_segments) if source_id not in already)
    return missing, duplicated


def render_translated_segments_as_srt(
    segments: Sequence[TranslatedCsvSegment],
) -> str:
    if not segments:
        return ""
    srt_segments = [
        SrtSegment(
            index=idx,
            start=segment.start,
            end=segment.end,
            text=segment.translation,
        )
        for idx, segment in enumerate(segments, start=1)
    ]
    return render_srt(srt_segments, reindex=True)


def render_corrected_segments_as_srt(
    segments: Sequence[TranslatedCsvSegment],
) -> str:
    if not segments:
        return ""
    srt_segments = [
        SrtSegment(
            index=idx,
            start=segment.start,
            end=segment.end,
            text=segment.corrected_text,
        )
        for idx, segment in enumerate(segments, start=1)
    ]
    return render_srt(srt_segments, reindex=True)
