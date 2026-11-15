"""The lines the correction and everything after it read: the judge's rows with what pass 1 heard applied.

Rows live in two layers (`docs/llm_harness_behavior.md`「说话人裁判」):

* the judge's layer -- `subtitles.stable_rows.read_rows`: `-stable.json` with `-diar-rows.json` (acoustic row
  splitting's halves when the task chose them). The judge reads it, and its sidecar's `rows_hash` and every row
  id its verdicts name are this layer's, so applying pass 1's cuts there would void the sidecar just written;
* the product layer (this module): that layer with what a judge that heard the audio changed. A row it split is
  two lines, the first keeping the row's id and the second `<id>b` (owner 2026-10-16); a row it filled carries
  the text heard whole and its grown time; a line it heard missing follows its row as `<id>i`, where it has room
  (a deferred one is not a line yet: subtitles do not overlap). Every line carries its id explicitly
  (`judgments.segment_ids` would number by position, and `stable_rows.halves` drops the id).

The correction's lines (pass 2's input, `-annotated.csv`, the knowledge harvest's ASR side), the `spk` column and
the per-speaker export read the product layer; the judge and the sidecar's checks read the judge's. The same
stable and sidecar always give the same lines, so a resumed task rebuilds them line for line. A split's first half
and a whole row keep the row's verdict; a second half and an inserted line take the tracks the judge gave them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...subtitles.stable_rows import halves, read_rows
from .judgments import Judgments, decided, read_judgments, segment_ids
from .labels import UNLABELLED, SegmentSpeaker, speaker_of

#: What a product line is of its judge-layer row.
WHOLE, FIRST, SECOND, INSERTED = "whole", "first", "second", "inserted"
#: The id suffixes of a second half and an inserted line.
SECOND_SUFFIX, INSERTED_SUFFIX = "b", "i"


@dataclass(frozen=True)
class Line:
    """One product line: the line itself (explicit `id`), the judge-layer row it comes from and what it is of it,
    and whether its text holds what the judge heard and the ASR did not write (a filled row, an inserted line)."""

    row: Mapping[str, Any]
    source: str
    part: str
    heard: bool = False

    @property
    def id(self) -> str:
        return str(self.row["id"])


@dataclass(frozen=True)
class Product:
    """The product layer of one stable: its lines in reading order, the speaker of each line a verdict decides
    (a moved row, a second half, an inserted line) by id -- every other line's speaker is its own acoustic one --
    the judge-layer rows it was made from, and the speaker of each line the judge heard missing that has no room
    yet (`deferred`), by the row it follows: no line, but someone spoke there."""

    lines: tuple[Line, ...]
    judged: Mapping[str, SegmentSpeaker] = field(default_factory=dict)
    source_rows: tuple[Mapping[str, Any], ...] = ()
    deferred: Mapping[str, SegmentSpeaker] = field(default_factory=dict)

    def rows(self) -> list[Mapping[str, Any]]:
        return [line.row for line in self.lines]

    def speakers(self) -> dict[str, SegmentSpeaker] | None:
        """Each line's speaker, verdicts applied, by id; None when no line has an acoustic track (nothing to tell
        the correction). A malformed acoustic label counts as unlabelled (and then no verdict was applied)."""
        acoustic = acoustic_speakers(self.lines)
        if not any(speaker.track is not None for speaker in acoustic.values()):
            return None
        return {key: self.judged.get(key, speaker) for key, speaker in acoustic.items()}


def acoustic_speakers(lines: Sequence[Line]) -> dict[str, SegmentSpeaker]:
    """Each line's own acoustic speaker by id (an inserted line has none; a malformed label counts as none)."""
    out = {}
    for line in lines:
        try:
            out[line.id] = speaker_of(line.row)
        except (TypeError, ValueError, KeyError):  # SpeakerLabelError is a ValueError
            out[line.id] = UNLABELLED
    return out


def product(segments: Sequence[Mapping[str, Any]], judgments: Judgments | None) -> Product:
    """The product layer of the judge-layer rows `segments` under `judgments` (None: the rows as they are, with
    explicit ids)."""
    judgments = judgments or Judgments(moved={})
    inserted = {insert.after: insert for insert in judgments.inserts if insert.shown is not None}
    deferred = {insert.after: decided(insert.track, UNLABELLED) for insert in judgments.inserts if insert.shown is None}
    lines: list[Line] = []
    judged: dict[str, SegmentSpeaker] = {}
    for key, segment in zip(segment_ids(segments), segments):
        own = _own_speaker(segment)
        if key in judgments.moved:
            judged[key] = decided(judgments.moved[key], own)
        cut, fill = judgments.halves.get(key), judgments.fills.get(key)
        if cut is not None:
            first, second = halves(segment, cut[0])
            lines += [Line({**first, "id": key}, key, FIRST), Line({**second, "id": key + SECOND_SUFFIX}, key, SECOND)]
            judged[key + SECOND_SUFFIX] = decided(cut[1], own)
        elif fill is not None:
            # its words no longer spell its text: no reader may rebuild the text from them
            row = {name: value for name, value in segment.items() if name != "words"}
            lines.append(Line({**row, "id": key, "text": fill.text, "start": fill.start, "end": fill.end}, key, WHOLE,
                              heard=True))
        else:
            lines.append(Line({**segment, "id": key}, key, WHOLE))
        insert = inserted.get(key)
        if insert is not None:
            start, end = insert.shown
            lines.append(Line({"id": key + INSERTED_SUFFIX, "start": start, "end": end, "text": insert.text},
                              key, INSERTED, heard=True))
            judged[key + INSERTED_SUFFIX] = decided(insert.track, UNLABELLED)
    return Product(tuple(lines), judged, tuple(segments), deferred)


def _own_speaker(segment: Mapping[str, Any]) -> SegmentSpeaker:
    try:
        return speaker_of(segment)
    except (TypeError, ValueError, KeyError):
        return UNLABELLED


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)  # product output


def read_product(stable_json: str | Path, *, warn: Callable[[str], None] = _warn) -> Product:
    """The product layer of the stable at `stable_json`: its judge-layer rows (`read_rows`) under the verdicts of
    its sidecar that still hold. Raises ValueError for a stable with no segments list. The verdicts are not read
    when no row has an acoustic track or a label is malformed, since they are checked against the rows' whole
    speaker objects; then the lines are the rows as they are."""
    segments = read_rows(stable_json)
    if not isinstance(segments, list):
        raise ValueError("Input JSON must contain a segments list.")
    rows = [segment for segment in segments if isinstance(segment, Mapping)]
    if len(rows) != len(segments):
        raise ValueError("Input JSON segments must be objects.")
    speakers = []
    for segment in rows:
        try:
            speakers.append(speaker_of(segment))
        except (TypeError, ValueError, KeyError):
            return product(rows, None)
    if not any(speaker.track is not None for speaker in speakers):
        return product(rows, None)
    return product(rows, read_judgments(stable_json, rows, warn=warn))
