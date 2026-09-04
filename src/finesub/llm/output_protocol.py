"""The correction round's output protocol: CSV-like block parsing, validation
and SRT reconstruction.

This is the one contract `session_contract` does not cover -- a correction
window is judged by what this parser accepts (schema, coverage, adjacency),
not by which tags came back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import List, Optional, Sequence

from .chunking import SubtitleSegment, SubtitleWindow, WindowIdMap
from .exchange_metadata import extract_top_level_tagged_blocks
from .prompt_variants import CorrectionVariant
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

# A reply may discard sources, but a window that discards most of itself is a
# failure wearing a valid shape. The 2026-08-22 canary: an agent that never saw
# the window text answered with one `sub` row plus `discard` for everything
# else; every structural check passed and the finished subtitle kept one line.
#
# This is the same rule the all-discard case has always had ("Translated CSV
# contains no valid rows" -- see the short-circuit comment below), just moved
# off the 100% boundary, so it adds no new class of rejection.
#
# The threshold is calibrated against replies that were *correct*, because the
# only cost of getting it wrong is rejecting one of those. `tools/
# discard_ratio_scan.py` over the local archive (63 whole windows / 49 runs,
# 2026-09-03) reads p50 0.007, p95 0.096, **max 0.219** -- and that max is the
# singing/English-PV material where discarding most of a song is correct. 0.5
# sits 2.3x above anything real, so it is a wrongness detector, not a quality
# knob. `bench-baselines.md` 二十五 has the full record.
#
# **Whole windows only.** The same scan replays each window through the
# production `split_window_in_half` (cut on the reasonable boundary nearest the
# middle, second half re-including the overlap tail -- so halves are neither
# equal nor disjoint), and reads the worst half at **43.8%** (H6dTZf9QFTY
# 0007-a: 128 sources, 56 discarded -- a stretch of song inside a window that
# averages far less). Gating a leaf like that would fail validation, exhaust
# the retries and stop the task on an answer that was right.
#
# 0.5 buys nothing there anyway: it is 2.3x the worst whole window but only
# **1.14x** the worst half, which is a coin flip rather than a detector. And
# applying the same 2.3x calibration to the half maximum lands above 100%, i.e.
# back on the all-discard check that already runs. So on a leaf the discard
# ratio simply has no discriminating power, and the protection there is that
# pre-existing check.
#
# ⚠ That leaves a **declared gap, not a proven absence**: a leaf is its own API
# call, so a reply that never saw the body can in principle arrive there too,
# and this gate would not catch it. Closing it needs a *different* signal (did
# the model receive the window text at all), not a different number -- filed in
# `llm_followups.md`.
MAX_DISCARD_RATIO = 0.5

# End-of-line marker (v12) letting the model retract a row it already wrote
# (e.g. it computed the duration cell and realized the merge span ran away).
# A marked row is treated as if the physical line does not exist: no structure
# validation, its source ids stay unclaimed so follow-up rows may re-emit them.
VOID_ROW_MARKER = "<void>"
OUTPUT_CSV_HEADER = (
    "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
)
OUTPUT_CSV_HEADER_WITH_START = (
    "type|position|start|duration|gap|corrected_text|translation|conf|char_count|note"
)


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
        + ". Every source must be covered by a sub/insert row or "
        "explicitly discarded with 'discard|<id>'."
    )


def _majority_discard_error(
    discarded_ids: set[str], expected_ids: Sequence[str], *, enabled: bool
) -> str | None:
    """Coverage says every source was *accounted for*; this says the window
    still produced subtitles. See MAX_DISCARD_RATIO for why 0.5, and why
    `enabled` is false on a split leaf."""

    if not enabled or not expected_ids:
        return None
    if len(discarded_ids) <= MAX_DISCARD_RATIO * len(expected_ids):
        return None
    percent = round(100 * len(discarded_ids) / len(expected_ids))
    return (
        f"Translated discards {len(discarded_ids)} of {len(expected_ids)} "
        f"sources ({percent}%), over the "
        f"{round(100 * MAX_DISCARD_RATIO)}% limit. Discard is for "
        "individual sources that carry no speech; a window where most "
        "sources are dropped means the window text was not read."
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
    # Locally normalized weighted translation chars. The model-reported value
    # is checked for drift, then replaced with the shared metric's result.
    char_count: str = ""
    note: str = ""

    @property
    def text(self) -> str:
        return self.translation


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
    # Inter-line reasoning comments emitted by the capableC variant
    # (``# <text>`` lines). Observability only; never affects ok.
    reasoning_rows: int = 0


def _encode_text_cell(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("|", "｜").replace("\n", r"\n")


def _decode_text_cell(text: str) -> str:
    decoded = (text or "").strip().replace(r"\n", "\n")
    return re.sub(r"\n{2,}", "\n", decoded)


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
    duration: str
    corrected: str
    translation: str
    conf: str
    note: str
    gap: str = ""
    has_gap: bool = False
    char_count: str = ""
    has_char_count: bool = False
    start: str = ""
    has_start: bool = False


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


def _reported_char_count(cell: str) -> float:
    return sum(float(part) for part in (cell or "").strip().split("+") if part)


def _normalized_char_count(
    row_number: int,
    translation: str,
    reported: str,
    warnings: List[str],
    *,
    row_label: str = "",
) -> str:
    actual = weighted_char_count(translation)
    # Only warn when the reported count is off by more than a slack of 2 plus
    # 20% of the actual value; the char_count is always normalized to the
    # computed value regardless. Small rounding-scale discrepancies are noise.
    if reported:
        tolerance = 2.0 + 0.20 * actual
        if abs(_reported_char_count(reported) - actual) > tolerance:
            label = row_label or f"Row {row_number}"
            warnings.append(
                f"{label} char_count {reported!r} does not match the "
                f"computed value {format_weighted_char_count(actual)}; normalized."
            )
    return format_weighted_char_count(actual)


#: The output row contract, in order. `note` is last because it is the only
#: cell allowed to contain a raw `|` (the prompt asks for the full-width `｜`
#: inside text), which is what lets the split below stay unambiguous.
_ROW_COLUMNS = (
    "kind",
    "position",
    "duration",
    "gap",
    "corrected",
    "translation",
    "conf",
    "char_count",
    "note",
)
_ROW_COLUMNS_WITH_START = (
    "kind",
    "position",
    "start",
    "duration",
    "gap",
    "corrected",
    "translation",
    "conf",
    "char_count",
    "note",
)


def _split_row_fields(row: str, *, has_start_column: bool = False) -> Optional[_RowFields]:
    """Map one output row onto the column spec strictly by position.

    ``type|position|duration|gap|corrected|translation|conf|char_count|note``
    (with ``start`` after ``position`` for the basic tier). Splitting with
    ``maxsplit`` keeps a legitimate piped ``note`` whole while an extra column
    lands in the wrong cell -- ``conf`` above all -- which the row validator
    then rejects as a structural error.

    This used to infer the layout from cell *contents*, so a row carrying one
    extra column was silently shifted a cell to the left and the untranslated
    source text ended up in ``translation`` with no error at all. Layout
    guessing is gone with the pre-v39 tolerances it existed for; those
    artifacts (`annotated.csv`) are regenerated by re-running the correction.

    Returns ``None`` when the row has too few cells to be this contract.
    """

    spec = _ROW_COLUMNS_WITH_START if has_start_column else _ROW_COLUMNS
    parts = row.split("|", len(spec) - 1)
    if len(parts) < len(spec):
        return None
    cells = dict(zip(spec, parts))
    start = cells.get("start", "")
    return _RowFields(
        kind=cells["kind"],
        position=cells["position"],
        duration=cells["duration"],
        corrected=cells["corrected"],
        translation=cells["translation"],
        conf=cells["conf"],
        note=cells["note"],
        gap=cells["gap"],
        has_gap=True,
        char_count=cells["char_count"],
        has_char_count=True,
        start=start,
        has_start=bool(start.strip()),
    )


def _unexpected_start_column_errors_in_rows(
    rows: Sequence[str], *, row_prefix: str = "Row"
) -> List[str]:
    """Find unambiguous 10-column rows under a 9-column CSV contract."""

    errors: List[str] = []
    for row_number, raw in enumerate(rows, start=1):
        row = raw.strip()
        if not row or row.startswith("#") or _row_is_voided(row):
            continue
        parts = row.split("|")
        if (
            len(parts) >= 10
            and parts[0].strip().lower() in {"", KIND_SUB, KIND_INSERT}
            and _is_numeric_cell(parts[2])
            and _is_numeric_cell(parts[3])
            and _is_numeric_cell(parts[4])
            and parts[7].strip().lower() in CONFIDENCE_LEVELS
            and _is_char_count_cell(parts[8])
        ):
            errors.append(
                f"{row_prefix} {row_number} has an unexpected start column before "
                "duration; expected type|position|duration|gap|corrected_text|"
                "translation|conf|char_count|note."
            )
    return errors


def looks_truncated_translated(text: str) -> bool:
    text = text or ""
    if "<translated" in text.lower() and not re.search(
        r"</translated\s*>", text, flags=re.IGNORECASE
    ):
        return True
    return False


def validate_translated_csv_text(
    text: str,
    source_segments: Sequence[SubtitleSegment],
    *,
    clip_start: float = 0.0,
    require_singles: bool = True,
    require_headers: bool = False,
    require_start_column: bool = False,
    forbid_start_column: bool = False,
    check_discard_ratio: bool = True,
) -> CsvValidationResult:
    """Validate and parse the `<translated>` block into typed segments.

    Uses **top-level** `<translated>` / `<singles>` extraction (sibling blocks
    only): mid-prose tag name-drops inside `<reasoning>` do not count.

    When `require_singles` is true (default; v34+ contract), also require a
    top-level `<singles>` block with exactly one single-source row per window
    source id in order. When `require_headers` is true (v55+ production and
    replay contract), the first non-empty line of every required CSV block must
    be :data:`OUTPUT_CSV_HEADER`. The default remains false so historical saved
    replies stay parseable for offline audits.

    When ``forbid_start_column`` is true, rows that unambiguously carry the
    ten-column experimental layout are rejected instead of being shifted into
    the nine-column fields. It is mutually exclusive with
    ``require_start_column``.

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

    if require_start_column and forbid_start_column:
        raise ValueError(
            "require_start_column and forbid_start_column cannot both be true."
        )

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
    expected_header = (
        OUTPUT_CSV_HEADER_WITH_START if require_start_column else OUTPUT_CSV_HEADER
    )
    if require_singles:
        errors.extend(
            _validate_singles_block(
                text or "",
                expected_ids=expected_ids,
                warnings=warnings,
                require_header=require_headers,
                require_start_column=require_start_column,
                forbid_start_column=forbid_start_column,
            )
        )

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
    reasoning_rows = 0

    payload = translated_blocks[0]
    rows = [row.strip() for row in payload.splitlines() if row.strip()]
    if rows and rows[0] == expected_header:
        rows = rows[1:]
    elif require_headers:
        errors.append(
            "<translated> first non-empty line must be the exact CSV header: "
            f"{expected_header}"
        )
    if forbid_start_column:
        errors.extend(_unexpected_start_column_errors_in_rows(rows))
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

    for row_number, row in enumerate(rows, start=1):
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
            parts = [p.strip() for p in row.split("|")]
            if len(parts) < 2 or not parts[1]:
                errors.append(
                    f"Row {row_number}: discard row needs at least "
                    "'discard|<source_ids>'."
                )
                continue
            disc_ids = tuple(
                sid.strip() for sid in parts[1].split(",") if sid.strip()
            )
            if not disc_ids:
                errors.append(f"Row {row_number}: discard row has no source ids.")
                continue
            row_errors: List[str] = []
            positions: List[int] = []
            local_seen: set[str] = set()
            for disc_id in disc_ids:
                if disc_id in local_seen:
                    row_errors.append(
                        f"Row {row_number}: discard repeats source id {disc_id}."
                    )
                    continue
                local_seen.add(disc_id)
                if disc_id not in source_by_id:
                    row_errors.append(
                        f"Row {row_number}: discard references unknown "
                        f"source id {disc_id}."
                    )
                elif disc_id in seen_source_ids:
                    row_errors.append(
                        f"Row {row_number}: source id {disc_id} already "
                        "covered by a sub/insert row; cannot discard."
                    )
                elif disc_id in discarded_ids:
                    row_errors.append(
                        f"Row {row_number}: source id {disc_id} discarded "
                        "more than once."
                    )
                else:
                    positions.append(source_index[disc_id])
            if positions and positions != sorted(positions):
                row_errors.append(
                    f"Row {row_number}: discard source ids must be in source order."
                )
            if len(positions) > 1 and any(
                b - a != 1 for a, b in zip(positions, positions[1:])
            ):
                row_errors.append(
                    f"Row {row_number}: discard source ids must be consecutive."
                )
            if positions and positions[0] <= previous_last_position:
                row_errors.append(
                    f"Row {row_number} appears before an earlier output row."
                )
            if row_errors:
                errors.extend(row_errors)
                continue
            discarded_ids.update(disc_ids)
            previous_last_position = positions[-1]
            continue
        # Inter-line reasoning comments (capableC variant): lines starting
        # with ``#`` are pre-positioned annotations explaining the merge/
        # discard decision of the immediately following sub/discard row.
        # No source ids — positional anchoring is sufficient. Skipped for
        # SRT; counted for observability.
        if row.startswith("#"):
            reasoning_rows += 1
            continue
        fields = _split_row_fields(row, has_start_column=require_start_column)
        if fields is None:
            errors.append(
                f"Row {row_number} has too few fields "
                "(need type|position|duration|corrected|translation)."
            )
            continue

        # The duration column is there to make the model notice each row's
        # time span before writing text; its value is discarded, but its
        # presence is part of the contract (missing it means the habit —
        # and usually the span discipline — slipped).
        if require_start_column and not (
            fields.has_start and _is_numeric_cell(fields.start)
        ):
            errors.append(
                f"Row {row_number} must carry numeric start seconds as column 3."
            )
            continue
        duration_cell = fields.duration.strip()
        try:
            float(duration_cell)
        except ValueError:
            errors.append(
                f"Row {row_number} must carry the merged duration in "
                "seconds as column 3."
            )
            continue
        if not _is_numeric_cell(fields.gap):
            errors.append(
                f"Row {row_number} must carry the trailing gap in "
                "seconds as column 4."
            )
            continue
        if not _is_char_count_cell(fields.char_count):
            errors.append(
                f"Row {row_number} must carry weighted translation chars "
                "as column 8 (for example 12.5 or 9+8.5)."
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
            # positional shape checks -- the extra cell pushes a non-numeric
            # value into char_count, which is an error.
            warnings.append(
                f"Row {row_number} has invalid conf '{fields.conf.strip()}'; dropped."
            )
        note = _decode_text_cell(fields.note)

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
        if len(source_ids) > TRANSLATED_MAX_MERGED_SOURCES:
            warnings.append(
                f"Row {row_number} merges {len(source_ids)} source ids "
                f"(soft cap {TRANSLATED_MAX_MERGED_SOURCES}; not rejected)."
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

        corrected_text = _decode_text_cell(fields.corrected)
        if not corrected_text.strip():
            row_errors.append(f"Row {row_number} has empty corrected text.")
        translation = _decode_text_cell(fields.translation)
        if not translation.strip():
            row_errors.append(f"Row {row_number} has empty translation text.")

        if row_errors:
            errors.extend(row_errors)
            continue

        assert positions
        first = source_segments[positions[0]]
        last = source_segments[positions[-1]]
        translated_segments.append(
            TranslatedCsvSegment(
                source_ids=source_ids,
                start=first.start,
                end=last.end,
                corrected_text=corrected_text,
                translation=translation,
                kind=KIND_SUB,
                conf=conf,
                char_count=_normalized_char_count(
                    row_number,
                    translation,
                    fields.char_count.strip(),
                    warnings,
                ),
                note=note,
            )
        )
        seen_source_ids.update(source_ids)
        previous_last_position = positions[-1]

    if not translated_segments and not any(
        e.startswith("Translated CSV") for e in errors
    ):
        errors.append("Translated CSV contains no valid rows.")

    for message in (
        _uncovered_sources_error(expected_ids, seen_source_ids | discarded_ids),
        _majority_discard_error(
            discarded_ids, expected_ids, enabled=check_discard_ratio
        ),
    ):
        if message:
            errors.append(message)

    return CsvValidationResult(
        ok=not errors,
        segments=translated_segments,
        errors=errors,
        warnings=warnings,
        voided_rows=voided_rows,
        discarded_ids=tuple(sorted(discarded_ids, key=lambda s: source_index.get(s, 0))),
        reasoning_rows=reasoning_rows,
    )


def _validate_singles_block(
    text: str,
    *,
    expected_ids: Sequence[str],
    warnings: List[str],
    require_header: bool = False,
    require_start_column: bool = False,
    forbid_start_column: bool = False,
) -> List[str]:
    """Require top-level `<singles>` with one single-source row per window id."""

    errors: List[str] = []
    blocks = extract_top_level_tagged_blocks(text, "singles")
    if len(blocks) != 1:
        errors.append(
            "Output must contain exactly one top-level <singles>...</singles> block "
            "(one row per source id)."
        )
        return errors

    rows = [row.strip() for row in blocks[0].splitlines() if row.strip()]
    expected_header = (
        OUTPUT_CSV_HEADER_WITH_START if require_start_column else OUTPUT_CSV_HEADER
    )


    if rows and rows[0] == expected_header:
        rows = rows[1:]
    elif require_header:
        errors.append(
            "<singles> first non-empty line must be the exact CSV header: "
            f"{expected_header}"
        )
    if forbid_start_column:
        errors.extend(
            _unexpected_start_column_errors_in_rows(rows, row_prefix="<singles> row")
        )
    if not rows:
        errors.append("<singles> block is empty.")
        return errors

    seen: list[str] = []
    seen_set: set[str] = set()
    expected_set = set(expected_ids)

    for row_number, row in enumerate(rows, start=1):
        if _row_is_voided(row):
            errors.append(
                f"<singles> row {row_number}: <void> is not allowed in singles."
            )
            continue
        kind_prefix = row.split("|", 1)[0].strip().lower()
        if kind_prefix == KIND_PLAN:
            errors.append(f"<singles> row {row_number}: plan| lines are not allowed.")
            continue
        fields = _split_row_fields(row, has_start_column=require_start_column)
        if fields is None:
            errors.append(
                f"<singles> row {row_number} has too few fields "
                "(need type|position|duration|corrected|translation)."
            )
            continue
        kind = fields.kind.strip().lower() or KIND_SUB
        if kind == KIND_INSERT:
            errors.append(
                f"<singles> row {row_number}: insert is not allowed in singles."
            )
            continue
        if kind not in (KIND_SUB, ""):
            errors.append(
                f"<singles> row {row_number}: type must be sub "
                f"(got '{fields.kind.strip()}')."
            )
            continue
        if require_start_column and not (
            fields.has_start and _is_numeric_cell(fields.start)
        ):
            errors.append(
                f"<singles> row {row_number} must carry numeric start seconds "
                "as column 3."
            )
            continue
        try:
            float(fields.duration.strip())
        except ValueError:
            errors.append(
                f"<singles> row {row_number} must carry duration seconds as column 3."
            )
            continue
        if not _is_char_count_cell(fields.char_count):
            errors.append(
                f"<singles> row {row_number} must carry weighted translation "
                "chars as column 8."
            )
            continue
        position = fields.position.strip()
        if "," in position:
            errors.append(
                f"<singles> row {row_number}: position must be a single source id "
                f"(no merge); got '{position}'."
            )
            continue
        source_id = position
        if not source_id:
            errors.append(f"<singles> row {row_number} has no source id.")
            continue
        if source_id not in expected_set:
            errors.append(
                f"<singles> row {row_number} references unknown source id {source_id}."
            )
            continue
        if source_id in seen_set:
            errors.append(f"<singles> repeats source id {source_id}.")
            continue
        if not _decode_text_cell(fields.corrected).strip():
            errors.append(f"<singles> row {row_number} has empty corrected text.")
            continue
        translation = _decode_text_cell(fields.translation)
        if not translation.strip():
            errors.append(f"<singles> row {row_number} has empty translation text.")
            continue
        if fields.has_char_count:
            _normalized_char_count(
                row_number,
                translation,
                fields.char_count.strip(),
                warnings,
                row_label=f"<singles> row {row_number}",
            )
        seen.append(source_id)
        seen_set.add(source_id)

    if seen != list(expected_ids):
        missing = [sid for sid in expected_ids if sid not in seen_set]
        extra = [sid for sid in seen if sid not in expected_set]
        if missing:
            preview = ", ".join(missing[:12])
            errors.append(
                f"<singles> missing source id(s): {preview}"
                + ("…" if len(missing) > 12 else "")
            )
        if extra:
            preview = ", ".join(extra[:12])
            errors.append(
                f"<singles> unexpected source id(s): {preview}"
                + ("…" if len(extra) > 12 else "")
            )
        if not missing and not extra and seen != list(expected_ids):
            errors.append("<singles> source ids must appear in window order.")

    return errors


def validate_correction_output_text(
    text: str,
    source_segments: Sequence[SubtitleSegment],
    *,
    variant: CorrectionVariant,
    clip_start: float = 0.0,
    check_discard_ratio: bool = True,
) -> CsvValidationResult:
    """Validate one correction reply against the exact served prompt variant.

    The variant owns whether a full ``<singles>`` block exists and whether CSV
    rows carry the experimental ``start`` column. Keeping that projection here prevents
    production, resume, and replay call sites from silently validating a reply
    against a different variant than the answering endpoint received.
    """

    return validate_translated_csv_text(
        text,
        source_segments,
        clip_start=clip_start,
        require_singles=variant.require_full_singles,
        require_headers=True,
        require_start_column=variant.output_has_start,
        forbid_start_column=not variant.output_has_start,
        check_discard_ratio=check_discard_ratio,
    )


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
    )


def validate_correction_window_output(
    text: str,
    window: SubtitleWindow,
    *,
    variant: CorrectionVariant,
) -> CsvValidationResult:
    """Validate local model positions, then restore harness source ids."""

    id_map = WindowIdMap.from_window(window)
    result = validate_correction_output_text(
        text,
        id_map.localize_segments(window.segments),
        variant=variant,
        clip_start=window.clip_start,
        # Only whole windows: see MAX_DISCARD_RATIO.
        check_discard_ratio=window.split_depth == 0,
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
    """Render segments as the 9-column output CSV.

    ``type|position|duration|gap|corrected|translation|conf|char_count|note`` —
    the same schema the model emits (v39; char_count precedes note). Insert
    positions are absolute ``start,duration`` in seconds. Gap is trailing
    silence to the next rendered subtitle (0.0 on the last row).
    """

    ordered = list(segments)
    lines = []
    for index, segment in enumerate(ordered):
        if index + 1 < len(ordered):
            gap = max(0.0, ordered[index + 1].start - segment.end)
        else:
            gap = 0.0
        lines.append(
            "|".join(
                (
                    segment.kind,
                    _row_position(segment),
                    f"{max(0.0, segment.end - segment.start):.1f}",
                    f"{gap:.1f}",
                    _encode_text_cell(segment.corrected_text),
                    _encode_text_cell(segment.translation),
                    "" if segment.conf is None else str(segment.conf),
                    format_weighted_char_count(
                        weighted_char_count(segment.translation)
                    ),
                    _encode_text_cell(segment.note),
                )
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def merge_translated_csv_windows(
    existing_segments: Sequence[TranslatedCsvSegment],
    current_source_ids: Sequence[str],
    new_segments: Sequence[TranslatedCsvSegment],
) -> List[TranslatedCsvSegment]:
    """Merge a window's output into prior results, newest-wins on overlap ids.

    Normal (``sub``) rows from earlier windows that merged ids straddling the
    overlap boundary (i.e. containing at least one id outside the current
    window) are kept whole, and any new row colliding with them is dropped; ids
    the dropped new row also covered are backfilled from the earlier window so
    no id silently loses its output. Ids the current window intentionally
    dropped stay dropped, except when such an id shares an earlier merged row
    with a backfilled id — the row is restored whole, which may resurrect the
    dropped id alongside it.
    """
    existing_subs = list(existing_segments)
    new_subs = list(new_segments)

    current_sources = set(current_source_ids)
    kept_old: List[TranslatedCsvSegment] = []
    displaced_old: List[TranslatedCsvSegment] = []
    claimed: set[str] = set()
    for segment in existing_subs:
        ids = set(segment.source_ids)
        overlap = ids & current_sources
        if not overlap:
            kept_old.append(segment)
        elif ids - current_sources:
            # Straddling row: part of it lies outside the current window, so
            # dropping it would lose output for those ids. Keep it whole and
            # claim its in-window ids so conflicting new rows are rejected.
            kept_old.append(segment)
            claimed |= overlap
        else:
            displaced_old.append(segment)
    kept_new: List[TranslatedCsvSegment] = []
    conflict_lost: set[str] = set()
    for segment in new_subs:
        ids = set(segment.source_ids)
        if ids & claimed:
            conflict_lost |= ids - claimed
        else:
            kept_new.append(segment)
    covered: set[str] = set()
    for segment in kept_old:
        covered |= set(segment.source_ids)
    for segment in kept_new:
        covered |= set(segment.source_ids)
    merged = kept_old + kept_new
    need = conflict_lost - covered
    for segment in displaced_old:
        ids = set(segment.source_ids)
        if ids & need and not ids & covered:
            merged.append(segment)
            covered |= ids
            need -= ids
    return sorted(merged, key=lambda segment: (segment.start, segment.end, segment.source_ids))


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
