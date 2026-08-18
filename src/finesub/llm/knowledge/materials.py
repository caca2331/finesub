"""Structured evidence materials for the unified knowledge update.

Builds the per-window CSV packs described in docs/knowledge.md
§1.4: for every correction window (by post-stitch ownership, so nothing is
duplicated across the physical window overlaps) the pack groups

- ``context_slice``   — the research context for that window,
- ``feedback_slice``  — the window's ``task_update_feedback``,
- ``raw_csv``         — the window's source rows from ``*-stable.json``,
- ``final_csv``       — the corresponding final rows (annotated.csv overlaid
                        with the postprocessed final SRT's timing/translation),
- ``refined_csv``     — (refined mode only) the user-refined SRT rows whose
                        time range falls into the window.

Packs are then chunked so the combined CSV text stays under the 100k-token
budget; chunk boundaries always sit on window boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..chunking import (
    SubtitleSegment,
    _format_csv_seconds,
    load_segments_from_stable_json,
    render_segments_as_csv,
)
from ..output_protocol import (
    KIND_INSERT,
    KIND_SUB,
    _decode_text_cell,
    _encode_text_cell,
    _parse_conf,
    _split_row_fields,
)
from .feedback import (
    FeedbackAggregate,
    aggregate_task_update_feedback,
    iter_task_artifacts,
)
from ..prompts import ContextPack
from ..research import load_research_context
from finesub.subtitles.model import SrtSegment, parse_srt


# §1.7: combined raw/final/refined CSV text per LLM call.
KNOWLEDGE_CSV_TOKEN_BUDGET = 100_000

# Evidence modes (§1.3).
MODE_ARTIFACTS_ONLY = "artifacts_only"
MODE_REFINED_ALIGNED = "refined_aligned"

# Pseudo window used when no executed-window metadata is available (degraded
# artifacts_only runs): all rows form one pack.
FALLBACK_WINDOW_ID = "all"

FINAL_CSV_HEADER = "# type|position|start|end|gap|corrected|translation|conf|char_count|note"
REFINED_CSV_HEADER = "# start|end|text"


# ---------------------------------------------------------------------------
# final_csv: annotated.csv + final SRT overlay


@dataclass(frozen=True)
class FinalRow:
    """One final subtitle row: annotated.csv columns + overlaid timing.

    ``start``/``end``/``translation`` come from the postprocessed final SRT
    (same index, 1:1 — annotated.csv and the SRT render from the same stitched
    segment list and postprocess never adds/removes/reorders entries);
    ``gap`` is recomputed on the postprocessed timeline. ``corrected`` is NOT
    overlaid: corrected.srt skips postprocess, so the model's original
    correction is preserved. ``sub`` rows anchor on their ``source_ids`` (one
    final row may merge several raw rows); ``insert`` rows have none.
    """

    kind: str
    source_ids: tuple[str, ...]
    corrected: str
    translation: str
    conf: str | None
    char_count: str
    note: str
    start: float
    end: float
    gap: float

    @property
    def is_insert(self) -> bool:
        return self.kind == KIND_INSERT

    @property
    def position(self) -> str:
        if self.is_insert:
            return f"{self.start:.1f},{max(0.0, self.end - self.start):.1f}"
        return ",".join(self.source_ids)


def _parse_annotated_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate((text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split_row_fields(line)
        if fields is None:
            raise ValueError(f"annotated.csv line {line_no} has too few fields: {line!r}")
        try:
            float(fields.duration.strip())
        except ValueError:
            # Columns are positional now, so an older layout lands its text in
            # the numeric cells instead of shifting quietly — fail loudly and
            # ask for a re-run. annotated.csv is a correction-run artifact, so
            # regenerating it is always available.
            raise ValueError(
                f"annotated.csv line {line_no} has no numeric duration "
                "column; the file predates the current output contract — "
                "re-run the correction to regenerate it."
            )
        kind = fields.kind.strip().lower() or KIND_SUB
        source_ids: tuple[str, ...] = ()
        if kind != KIND_INSERT:
            kind = KIND_SUB
            source_ids = tuple(
                part.strip() for part in fields.position.split(",") if part.strip()
            )
        rows.append(
            {
                "kind": kind,
                "source_ids": source_ids,
                "corrected": _decode_text_cell(fields.corrected),
                "translation": _decode_text_cell(fields.translation),
                "conf": _parse_conf(fields.conf),
                "char_count": fields.char_count.strip(),
                "note": _decode_text_cell(fields.note),
            }
        )
    return rows


def build_final_rows(annotated_text: str, final_srt_text: str) -> list[FinalRow]:
    """Overlay the final SRT onto annotated.csv rows by index (1:1)."""

    annotated = _parse_annotated_rows(annotated_text)
    srt_segments = parse_srt(final_srt_text)
    if len(annotated) != len(srt_segments):
        raise ValueError(
            "annotated.csv and the final SRT disagree on row count "
            f"({len(annotated)} vs {len(srt_segments)}); they must come from "
            "the same correction run."
        )
    rows: list[FinalRow] = []
    for idx, (annotated_row, srt_segment) in enumerate(zip(annotated, srt_segments)):
        next_start = (
            srt_segments[idx + 1].start if idx + 1 < len(srt_segments) else None
        )
        gap = 0.0 if next_start is None else max(0.0, next_start - srt_segment.end)
        rows.append(
            FinalRow(
                kind=annotated_row["kind"],
                source_ids=annotated_row["source_ids"],
                corrected=annotated_row["corrected"],
                translation=srt_segment.text,
                conf=annotated_row["conf"],
                char_count=annotated_row["char_count"],
                note=annotated_row["note"],
                start=srt_segment.start,
                end=srt_segment.end,
                gap=gap,
            )
        )
    return rows


def render_final_csv(rows: Sequence[FinalRow]) -> str:
    lines = [FINAL_CSV_HEADER]
    for row in rows:
        lines.append(
            "|".join(
                (
                    row.kind,
                    row.position,
                    _format_csv_seconds(row.start),
                    _format_csv_seconds(row.end),
                    _format_csv_seconds(row.gap),
                    _encode_text_cell(row.corrected),
                    _encode_text_cell(row.translation),
                    "" if row.conf is None else str(row.conf),
                    row.char_count,
                    _encode_text_cell(row.note),
                )
            )
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Executed windows (post-stitch ownership)


@dataclass(frozen=True)
class ExecutedWindow:
    chunk_id: str
    source_ids: tuple[str, ...]


def load_executed_windows(
    artifact_paths: Iterable[str | Path],
) -> list[ExecutedWindow]:
    """Committed correction windows in execution order, from task artifacts.

    A window counts as committed when its response validated and was not
    output-limited, or when it replayed from the resume cache
    (``correction_window_cached``). Re-appearances (reruns) keep the latest
    record but windows always complete in order, so order is stable.
    """

    windows: dict[str, tuple[str, ...]] = {}
    for record in iter_task_artifacts(artifact_paths):
        kind = record.get("kind")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if kind == "correction_window_cached":
            chunk_id = str(payload.get("chunk_id") or "")
            source_ids = tuple(str(item) for item in payload.get("source_ids") or ())
        elif kind == "correction_window_response":
            if not payload.get("validation_ok") or payload.get("output_limited"):
                continue
            chunk_id = str(payload.get("chunk_id") or "")
            window_meta = payload.get("window")
            if not isinstance(window_meta, Mapping):
                continue
            source_ids = tuple(
                str(item) for item in window_meta.get("source_ids") or ()
            )
        else:
            continue
        if not chunk_id or not source_ids:
            continue
        # Move-to-end so a rerun's replay order wins.
        windows.pop(chunk_id, None)
        windows[chunk_id] = source_ids
    return [
        ExecutedWindow(chunk_id=chunk_id, source_ids=source_ids)
        for chunk_id, source_ids in windows.items()
    ]


# ---------------------------------------------------------------------------
# Window grouping (ownership partition)


@dataclass
class WindowGroup:
    """One window's owned evidence rows (non-overlapping across groups)."""

    chunk_id: str
    final_rows: list[FinalRow] = field(default_factory=list)
    raw_segments: list[SubtitleSegment] = field(default_factory=list)

    @property
    def start(self) -> float:
        starts = [seg.start for seg in self.raw_segments]
        starts += [row.start for row in self.final_rows]
        return min(starts) if starts else 0.0

    @property
    def end(self) -> float:
        ends = [seg.end for seg in self.raw_segments]
        ends += [row.end for row in self.final_rows]
        return max(ends) if ends else 0.0


def group_rows_by_window(
    final_rows: Sequence[FinalRow],
    windows: Sequence[ExecutedWindow],
    stable_segments: Sequence[SubtitleSegment],
) -> list[WindowGroup]:
    """Partition final rows and raw segments into per-window groups.

    Ownership follows the stitch semantics without replaying it: an id belongs
    to the LAST executed window containing it (newest wins on the physical
    overlap), a ``sub`` row to the owner of its first id, and its remaining
    ids follow the row so a straddling (backfilled) row stays in one pack.
    Raw ids no final row covers (rows the model dropped) still surface in
    their owner's ``raw_csv``. ``insert`` rows are assigned by time. With no
    window metadata everything lands in a single fallback group.
    """

    if not windows:
        windows = [
            ExecutedWindow(
                chunk_id=FALLBACK_WINDOW_ID,
                source_ids=tuple(segment.id for segment in stable_segments),
            )
        ]
    owner_index: dict[str, int] = {}
    for win_idx, window in enumerate(windows):
        for source_id in window.source_ids:
            owner_index[source_id] = win_idx

    groups = [WindowGroup(chunk_id=window.chunk_id) for window in windows]
    segment_by_id = {segment.id: segment for segment in stable_segments}
    segment_order = {segment.id: idx for idx, segment in enumerate(stable_segments)}
    claimed_ids: set[str] = set()
    inserts: list[FinalRow] = []

    for row in final_rows:
        if row.is_insert:
            inserts.append(row)
            continue
        indices = [owner_index[sid] for sid in row.source_ids if sid in owner_index]
        target = indices[0] if indices else len(groups) - 1
        groups[target].final_rows.append(row)
        claimed_ids.update(row.source_ids)

    for group_idx, group in enumerate(groups):
        raw_ids = {sid for row in group.final_rows for sid in row.source_ids}
        raw_ids |= {
            sid
            for sid, owner in owner_index.items()
            if owner == group_idx and sid not in claimed_ids
        }
        group.raw_segments = sorted(
            (segment_by_id[sid] for sid in raw_ids if sid in segment_by_id),
            key=lambda segment: segment_order[segment.id],
        )

    non_empty = [group for group in groups if group.raw_segments or group.final_rows]
    for row in inserts:
        target = _group_for_time(non_empty, row.start, row.end)
        if target is not None:
            target.final_rows.append(row)
    for group in non_empty:
        group.final_rows.sort(key=lambda row: (row.start, row.end))
    return non_empty


def _overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _group_for_time(
    groups: Sequence[WindowGroup], start: float, end: float
) -> WindowGroup | None:
    """The group whose time range overlaps most; nearest when none overlaps."""

    if not groups:
        return None
    best = max(
        groups,
        key=lambda group: _overlap_seconds(start, end, group.start, group.end),
    )
    if _overlap_seconds(start, end, best.start, best.end) > 0:
        return best
    return min(
        groups,
        key=lambda group: min(
            abs(start - group.end), abs(group.start - end)
        ),
    )


def render_raw_csv(segments: Sequence[SubtitleSegment]) -> str:
    """Raw stable rows in the correction-input column layout, global seconds."""

    return render_segments_as_csv(segments, window_start=0.0)


# ---------------------------------------------------------------------------
# refined_csv (refined_aligned mode only)


def load_refined_segments(text: str) -> list[SrtSegment]:
    """Parse a user-refined SRT and re-sort by start time.

    Refined files may carry broken indices and overlapping annotation
    subtitles; sorting by time is the only normalization (per design J the
    remaining noise is disclosed to the model in the prompt, not cleaned).
    """

    return sorted(parse_srt(text), key=lambda segment: (segment.start, segment.end))


def split_refined_by_window(
    refined_segments: Sequence[SrtSegment],
    groups: Sequence[WindowGroup],
) -> dict[str, list[SrtSegment]]:
    """Assign each refined row to exactly one window group by time overlap."""

    assignment: dict[str, list[SrtSegment]] = {group.chunk_id: [] for group in groups}
    for segment in refined_segments:
        target = _group_for_time(groups, segment.start, segment.end)
        if target is not None:
            assignment[target.chunk_id].append(segment)
    return assignment


def render_refined_csv(segments: Sequence[SrtSegment]) -> str:
    lines = [REFINED_CSV_HEADER]
    for segment in segments:
        lines.append(
            "|".join(
                (
                    _format_csv_seconds(segment.start),
                    _format_csv_seconds(segment.end),
                    _encode_text_cell(segment.text),
                )
            )
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Window packs and chunk planning


@dataclass(frozen=True)
class WindowMaterials:
    """All rendered blocks of one window pack."""

    chunk_id: str
    start: float
    end: float
    context_slice: str
    feedback_slice: str
    raw_csv: str
    final_csv: str
    refined_csv: str = ""

    @property
    def csv_text(self) -> str:
        return f"{self.raw_csv}{self.final_csv}{self.refined_csv}"

    def pack_text(self) -> str:
        parts = [
            f"--- window {self.chunk_id} "
            f"[{_format_csv_seconds(self.start)}s – {_format_csv_seconds(self.end)}s] ---",
            f"<context_slice>\n{self.context_slice.strip() or '（无）'}\n</context_slice>",
            f"<feedback_slice>\n{self.feedback_slice.strip() or '（无）'}\n</feedback_slice>",
            f"<raw_csv>\n{self.raw_csv.strip()}\n</raw_csv>",
            f"<final_csv>\n{self.final_csv.strip()}\n</final_csv>",
        ]
        if self.refined_csv:
            parts.append(f"<refined_csv>\n{self.refined_csv.strip()}\n</refined_csv>")
        return "\n".join(parts)


@dataclass(frozen=True)
class KnowledgeChunk:
    """One knowledge-update LLM call's worth of window packs."""

    index: int
    windows: tuple[WindowMaterials, ...]
    csv_tokens: int

    @property
    def window_ids(self) -> tuple[str, ...]:
        return tuple(window.chunk_id for window in self.windows)

    def packs_text(self) -> str:
        return "\n\n".join(window.pack_text() for window in self.windows)

    def input_hash_payload(self) -> dict[str, Any]:
        """Chunk identity for the apply ledger: material text only.

        Deliberately excludes the KB-entry excerpt (it changes as earlier
        chunks apply) so a rerun recognizes an already-applied chunk.
        """

        return {
            "window_ids": list(self.window_ids),
            "packs_sha_source": self.packs_text(),
        }


def plan_knowledge_chunks(
    window_materials: Sequence[WindowMaterials],
    *,
    count_tokens: Callable[[str], int],
    csv_token_budget: int = KNOWLEDGE_CSV_TOKEN_BUDGET,
) -> list[KnowledgeChunk]:
    """Sequentially group window packs so CSV text stays under the budget.

    Cuts only on window boundaries and never overlaps chunks. A single window
    over the budget still becomes its own chunk (windows are never split);
    the assembled-prompt hard check downstream is the final guard.
    """

    chunks: list[KnowledgeChunk] = []
    current: list[WindowMaterials] = []
    current_tokens = 0
    for materials in window_materials:
        tokens = count_tokens(materials.csv_text)
        if current and current_tokens + tokens > csv_token_budget:
            chunks.append(
                KnowledgeChunk(
                    index=len(chunks), windows=tuple(current), csv_tokens=current_tokens
                )
            )
            current = []
            current_tokens = 0
        current.append(materials)
        current_tokens += tokens
    if current:
        chunks.append(
            KnowledgeChunk(
                index=len(chunks), windows=tuple(current), csv_tokens=current_tokens
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Top-level assembly


@dataclass(frozen=True)
class KnowledgeMaterials:
    """Everything the unified knowledge update needs, ready for prompting."""

    mode: str
    chunks: tuple[KnowledgeChunk, ...]
    feedback: FeedbackAggregate
    general_context: str
    warnings: tuple[str, ...] = ()

    @property
    def window_count(self) -> int:
        return sum(len(chunk.windows) for chunk in self.chunks)


def _load_context_pack(path: str | Path | None) -> ContextPack:
    if not path:
        return ContextPack()
    context_path = Path(path).expanduser()
    if not context_path.exists():
        return ContextPack()
    # Same rule as the correction side (``research.load_research_context``): a
    # note without a source range cannot be placed on any window, so the pack is
    # rejected rather than quietly contributing general context alone.
    return load_research_context(context_path)


def build_knowledge_materials(
    *,
    stable_json: str | Path,
    annotated_csv: str | Path,
    final_srt: str | Path,
    research_context: str | Path | None = None,
    artifact_dirs: Sequence[str | Path] = (),
    refined_srt: str | Path | None = None,
    count_tokens: Callable[[str], int],
    csv_token_budget: int = KNOWLEDGE_CSV_TOKEN_BUDGET,
) -> KnowledgeMaterials:
    """Load task outputs and assemble the chunked window packs."""

    warnings: list[str] = []
    stable_segments = load_segments_from_stable_json(stable_json)
    final_rows = build_final_rows(
        Path(annotated_csv).expanduser().read_text(encoding="utf-8"),
        Path(final_srt).expanduser().read_text(encoding="utf-8"),
    )
    windows = load_executed_windows(artifact_dirs)
    if not windows:
        warnings.append(
            "no executed-window metadata in task artifacts; using one fallback window"
        )
    feedback = aggregate_task_update_feedback(artifact_dirs)
    if feedback.is_empty:
        warnings.append(
            "no task_update_feedback artifacts found; knowledge hints are empty "
            "(was the task run with feedback collection enabled?)"
        )
    warnings.extend(feedback.warnings)
    context_pack = _load_context_pack(research_context).with_source_order(
        [segment.id for segment in stable_segments]
    )

    groups = group_rows_by_window(final_rows, windows, stable_segments)
    refined_assignment: dict[str, list[SrtSegment]] = {}
    mode = MODE_ARTIFACTS_ONLY
    if refined_srt is not None:
        mode = MODE_REFINED_ALIGNED
        refined_segments = load_refined_segments(
            Path(refined_srt).expanduser().read_text(encoding="utf-8")
        )
        refined_assignment = split_refined_by_window(refined_segments, groups)

    window_materials = [
        WindowMaterials(
            chunk_id=group.chunk_id,
            start=group.start,
            end=group.end,
            context_slice=context_pack.window_context_for_source_ids(
                [segment.id for segment in group.raw_segments],
                chunk_id=group.chunk_id,
            ),
            feedback_slice=feedback.feedback_slice_text(group.chunk_id),
            raw_csv=render_raw_csv(group.raw_segments),
            final_csv=render_final_csv(group.final_rows),
            refined_csv=(
                render_refined_csv(refined_assignment[group.chunk_id])
                if refined_assignment.get(group.chunk_id)
                else ""
            ),
        )
        for group in groups
    ]
    chunks = plan_knowledge_chunks(
        window_materials,
        count_tokens=count_tokens,
        csv_token_budget=csv_token_budget,
    )
    return KnowledgeMaterials(
        mode=mode,
        chunks=tuple(chunks),
        feedback=feedback,
        general_context=context_pack.general_prompt_text(),
        warnings=tuple(warnings),
    )
