"""Task-update feedback (schema v3): parsing and cross-stage aggregation.

Correction windows and the research final round (fast round 1 included) emit a
``<task_update_feedback>`` JSON block when feedback collection is enabled. The
harness persists each block as a task artifact (``correction_window_task_feedback``
/ ``research_task_feedback``); this module reads those artifacts back and
aggregates them into the structured inputs of the unified knowledge update:
per-window feedback slices, the research-wide slice, and the merged
``knowledge_hints`` that drive knowledge-entry prefetching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..chunking import WindowIdMap
from ..output_tags import parse_json_object
from .base import KNOWLEDGE_CATEGORIES, TASK_ARTIFACT_FILENAME


FEEDBACK_SCHEMA_VERSION = 3

# ``direction`` mirrors the proposal ops plus ``new_entry`` for entries the
# knowledge base does not have yet.
KNOWLEDGE_HINT_DIRECTIONS = ("new_entry", "replace_section", "append_lines")

WINDOW_FEEDBACK_ARTIFACT_KIND = "correction_window_task_feedback"
RESEARCH_FEEDBACK_ARTIFACT_KIND = "research_task_feedback"

# Research hints weigh double in entry-frequency scoring: the research round
# sees the whole transcript, so its hints are less likely to be window-local
# noise.
RESEARCH_HINT_WEIGHT = 2


@dataclass(frozen=True)
class KnowledgeHint:
    """One model-declared update intention for a knowledge-base entry."""

    category: str
    entry: str
    sub: str = ""
    direction: str = ""
    focus: str = ""
    reason: str = ""
    source_ids: tuple[str, ...] = ()
    confidence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "category": self.category,
            "entry": self.entry,
            "direction": self.direction,
            "focus": self.focus,
            "reason": self.reason,
        }
        if self.sub:
            data["sub"] = self.sub
        if self.source_ids:
            data["source_ids"] = list(self.source_ids)
        if self.confidence is not None:
            data["confidence"] = self.confidence
        return data


@dataclass(frozen=True)
class TaskFeedback:
    """One parsed ``<task_update_feedback>`` block."""

    origin: str  # "window" | "research"
    chunk_id: str = ""  # empty for research feedback
    hints: tuple[KnowledgeHint, ...] = ()
    asr_corrections: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.hints or self.asr_corrections or self.uncertainties)

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_hints": [hint.to_dict() for hint in self.hints],
            "asr_corrections": list(self.asr_corrections),
            "uncertainties": list(self.uncertainties),
        }

    def render_text(self) -> str:
        """Compact JSON used as the prompt's per-window feedback slice."""

        if self.is_empty:
            return ""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _string_items(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _parse_confidence(value: Any) -> int | None:
    try:
        confidence = int(value)
    except (TypeError, ValueError):
        return None
    return confidence if 1 <= confidence <= 9 else None


def _parse_hint(data: Any, warnings: list[str]) -> KnowledgeHint | None:
    if not isinstance(data, Mapping):
        warnings.append("knowledge_hints item is not an object; dropped")
        return None
    category = str(data.get("category", "")).strip()
    entry = str(data.get("entry", "")).strip()
    direction = str(data.get("direction", "")).strip()
    if category not in KNOWLEDGE_CATEGORIES:
        # Models sometimes read the schema's "streamer|common" enum notation
        # as a template and emit "streamer|game_lore"; salvage the leading
        # segment instead of dropping the hint.
        head = category.split("|", 1)[0].strip().lower()
        if head in KNOWLEDGE_CATEGORIES:
            warnings.append(
                f"hint {entry!r} category {category!r} coerced to {head!r}"
            )
            category = head
        else:
            warnings.append(f"hint {entry!r} has unknown category {category!r}; dropped")
            return None
    if not entry:
        warnings.append("hint with empty entry; dropped")
        return None
    if direction and direction not in KNOWLEDGE_HINT_DIRECTIONS:
        warnings.append(
            f"hint {entry!r} has unknown direction {direction!r}; kept without direction"
        )
        direction = ""
    return KnowledgeHint(
        category=category,
        entry=entry,
        sub=str(data.get("sub", "")).strip(),
        direction=direction,
        focus=str(data.get("focus", "")).strip(),
        reason=str(data.get("reason", "")).strip(),
        source_ids=_string_items(data.get("source_ids")),
        confidence=_parse_confidence(data.get("confidence")),
    )


def parse_task_update_feedback(
    body: str,
    *,
    origin: str,
    chunk_id: str = "",
) -> TaskFeedback:
    """Parse one feedback block body (lenient; never raises).

    Invalid JSON degrades to an empty feedback with a warning; invalid hints
    are dropped individually. Feedback is advisory input to the knowledge
    update, so a malformed block must never fail the main task.
    """

    body = (body or "").strip()
    if not body:
        return TaskFeedback(origin=origin, chunk_id=chunk_id)
    try:
        data = parse_json_object(body)
    except (ValueError, json.JSONDecodeError) as exc:
        return TaskFeedback(
            origin=origin,
            chunk_id=chunk_id,
            warnings=(f"feedback JSON parse failed: {exc}",),
        )
    warnings: list[str] = []
    hints: list[KnowledgeHint] = []
    raw_hints = data.get("knowledge_hints")
    if isinstance(raw_hints, Sequence) and not isinstance(raw_hints, str):
        for item in raw_hints:
            hint = _parse_hint(item, warnings)
            if hint is not None:
                hints.append(hint)
    elif raw_hints not in (None, []):
        warnings.append("knowledge_hints is not a list; ignored")
    return TaskFeedback(
        origin=origin,
        chunk_id=chunk_id,
        hints=tuple(hints),
        asr_corrections=_string_items(data.get("asr_corrections")),
        uncertainties=_string_items(data.get("uncertainties")),
        warnings=tuple(warnings),
    )


def remap_feedback_source_ids(body: str, id_map: WindowIdMap) -> str:
    """Restore local ``knowledge_hints[].source_ids`` before persistence.

    Feedback is best-effort: malformed JSON is left untouched for the existing
    lenient parser to diagnose later, while invalid/non-target local ids are
    dropped from otherwise valid hints.
    """

    body = (body or "").strip()
    if not body:
        return ""
    try:
        data = parse_json_object(body)
    except (ValueError, json.JSONDecodeError):
        return body
    raw_hints = data.get("knowledge_hints")
    if isinstance(raw_hints, Sequence) and not isinstance(raw_hints, str):
        for hint in raw_hints:
            if not isinstance(hint, dict) or "source_ids" not in hint:
                continue
            local_ids = _string_items(hint.get("source_ids"))
            restored: list[str] = []
            for local_id in local_ids:
                try:
                    restored.append(id_map.source_id_for_local(local_id))
                except ValueError:
                    continue
            if restored:
                hint["source_ids"] = restored
            else:
                hint.pop("source_ids", None)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class FeedbackAggregate:
    """All collected feedback of one task, keyed for per-window injection."""

    window_feedback: Mapping[str, TaskFeedback] = field(default_factory=dict)
    research_feedback: TaskFeedback | None = None
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.window_feedback and (
            self.research_feedback is None or self.research_feedback.is_empty
        )

    def feedback_slice_text(self, chunk_id: str) -> str:
        feedback = self.window_feedback.get(chunk_id)
        return feedback.render_text() if feedback else ""

    def research_slice_text(self) -> str:
        if self.research_feedback is None:
            return ""
        return self.research_feedback.render_text()

    def all_hints(self) -> list[KnowledgeHint]:
        hints = [
            hint
            for feedback in self.window_feedback.values()
            for hint in feedback.hints
        ]
        if self.research_feedback is not None:
            hints.extend(self.research_feedback.hints)
        return hints

    def merged_uncertainties(self) -> list[str]:
        seen: dict[str, None] = {}
        for feedback in (*self.window_feedback.values(), self.research_feedback):
            if feedback is None:
                continue
            for item in feedback.uncertainties:
                seen.setdefault(item)
        return list(seen)

    def merged_asr_corrections(self) -> list[str]:
        seen: dict[str, None] = {}
        for feedback in (*self.window_feedback.values(), self.research_feedback):
            if feedback is None:
                continue
            for item in feedback.asr_corrections:
                seen.setdefault(item)
        return list(seen)

    def hint_scores(
        self, *, research_weight: int = RESEARCH_HINT_WEIGHT
    ) -> dict[tuple[str, str], float]:
        """Frequency score per ``(category, entry)``: window ×1, research ×N."""

        scores: dict[tuple[str, str], float] = {}
        for feedback in self.window_feedback.values():
            for hint in feedback.hints:
                key = (hint.category, hint.entry)
                scores[key] = scores.get(key, 0.0) + 1.0
        if self.research_feedback is not None:
            for hint in self.research_feedback.hints:
                key = (hint.category, hint.entry)
                scores[key] = scores.get(key, 0.0) + float(research_weight)
        return scores


def iter_task_artifacts(
    artifact_paths: Iterable[str | Path],
) -> Iterable[dict[str, Any]]:
    """Yield task artifact records from JSONL files or artifact directories."""

    for raw_path in artifact_paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            path = path / TASK_ARTIFACT_FILENAME
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def aggregate_task_update_feedback(
    artifact_paths: Iterable[str | Path],
) -> FeedbackAggregate:
    """Read feedback artifacts and aggregate them (last record wins per key).

    Reads only the structured feedback artifact kinds — never the full window
    responses. Retries and reruns append multiple records for the same window;
    the latest one reflects the committed output.
    """

    window_feedback: dict[str, TaskFeedback] = {}
    research_feedback: TaskFeedback | None = None
    warnings: list[str] = []
    for record in iter_task_artifacts(artifact_paths):
        kind = record.get("kind")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        body = str(payload.get("feedback") or "")
        if kind == WINDOW_FEEDBACK_ARTIFACT_KIND:
            chunk_id = str(payload.get("chunk_id") or "")
            feedback = parse_task_update_feedback(
                body, origin="window", chunk_id=chunk_id
            )
            window_feedback[chunk_id] = feedback
            warnings.extend(
                f"window {chunk_id}: {warning}" for warning in feedback.warnings
            )
        elif kind == RESEARCH_FEEDBACK_ARTIFACT_KIND:
            research_feedback = parse_task_update_feedback(body, origin="research")
            warnings.extend(
                f"research: {warning}" for warning in research_feedback.warnings
            )
    return FeedbackAggregate(
        window_feedback=window_feedback,
        research_feedback=research_feedback,
        warnings=tuple(warnings),
    )
