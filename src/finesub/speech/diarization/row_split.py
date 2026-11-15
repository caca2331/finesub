"""Acoustic row splitting: cut a subtitle line in two where the diarizer clearly hears the speaker change.

ASR often writes two people speaking one after the other into one line. Where the acoustic evidence is
strong, the line is cut once at the change, with no LLM and no grammar dictionary; the halves then go
to the speaker judge and the correction like any other line (owner 2026-10-14). The rule, in order:

1. **Detect** on Nemotron's own streaming slots (`ClusterEvidence.slot_grid`), not on the vbx tracks:
   global clustering re-labels mid-sentence where the slots stay put. A word is *clean* for a slot
   covering at least `CLEAN` of it while no other covers more than `DIRTY`; the line must run clean A
   words, then clean B words and nothing else clean, with at least `MIN_SECONDS` of clean speech and
   `MIN_CHARS` characters on each side.
2. **Place** the cut, inside the range the acoustics allow (after the last clean A word, up to the first
   clean B word), on the boundary the segmentation DP scores lowest (`segmentation.row_boundaries`:
   punctuation, VAD gaps, word pauses, ASR seams), never on a banned one; ties go to the boundary nearest
   the acoustic cut, then left.
3. **Glue** the word left of the cut to the right half when it has a break before it and none after it
   -- the DP's gap-word rule. The diarizer's change point lags, so the second speaker's first word often
   lands on the first speaker's side.
4. **Gate**: no cut on a bare boundary (DP score above `GATE_B`), and the two halves must lead
   different vbx tracks, judged at the acoustic cut (the tracks lag like the slots do; the question is
   "two people in this line", not "where").

⚠ **Read on Japanese only** (two rulers, 11.3 hours, hina streams and a drama): 37 of 121 two-person
lines improved (34 cut exactly), 1 two-person and 3 one-person cuts severe, 0.79 grammar-breaking cuts
per hour. Other languages are not cut (`LANGUAGES`) until a reading exists. Evidence:
`docs/diarization-findings.md`; the experiment this ports is `tools/diarization_explore/row_split*.py`.

Only `--diarize` runs reach this, and only with nemotron3's evidence; without it nothing is cut. The cuts are
written to `-diar-rows.json` with the labels (`stage.write_rows`) and read through `subtitles.stable_rows`.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .segments import TrackSegment, segments_from_activity
from ..postprocessing import segmentation

CLEAN = 0.7
DIRTY = 0.2
MIN_SECONDS = 0.15
MIN_CHARS = 4
#: The DP boundary score a cut may have at most. A bare CJK word boundary (no punctuation, no space,
#: no pause) scores `BARE`; the score takes only a few discrete values, so this is not a tuned point.
GATE_B = 2.5
BARE = 3.0
#: Languages the rule has a reading on.
LANGUAGES = frozenset({"ja"})
#: A word shorter than this is widened to it before measuring coverage.
_MIN_WORD_SECONDS = 0.02


@dataclass(frozen=True)
class Cut:
    """Line `row` (0-based) is cut before its word `word`."""

    row: int
    word: int


def _characters(text: str) -> int:
    return len(re.sub(r"[\W_ー]", "", text))


class _Index:
    """Track segments by start time, for the ones overlapping a span."""

    def __init__(self, segments: Sequence[TrackSegment]):
        self.order = sorted(segments, key=lambda s: s.start)
        self.starts = [s.start for s in self.order]
        self.longest = max((s.end - s.start for s in self.order), default=0.0)

    def near(self, low: float, high: float) -> list[TrackSegment]:
        i = bisect.bisect_left(self.starts, low - self.longest)
        j = bisect.bisect_right(self.starts, high)
        return [s for s in self.order[i:j] if s.end > low and s.start < high]


def _clean_labels(words: Sequence[Mapping[str, Any]], near: Sequence[TrackSegment]) -> list[int | None]:
    out: list[int | None] = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if end - start < _MIN_WORD_SECONDS:
            middle = (start + end) / 2
            start, end = middle - _MIN_WORD_SECONDS / 2, middle + _MIN_WORD_SECONDS / 2
        cover: Counter = Counter()
        for segment in near:
            if segment.end > start and segment.start < end:
                cover[segment.track] += min(segment.end, end) - max(segment.start, start)
        if not cover:
            out.append(None)
            continue
        track, share = max(cover.items(), key=lambda item: item[1])
        share /= end - start
        others = all(value / (end - start) <= DIRTY for key, value in cover.items() if key != track)
        out.append(track if share >= CLEAN and others else None)
    return out


def _labels(row: Mapping[str, Any], index: _Index) -> list[int | None]:
    near = index.near(float(row["start"]) - 0.5, float(row["end"]) + 0.5)
    return _clean_labels(row.get("words") or [], near)


def _propose(row: Mapping[str, Any], slots: _Index) -> tuple[int, tuple[int, int]] | None:
    """(acoustic cut, (first, last) word index the cut may take), or None when the slots do not show one
    clean change."""
    words = row.get("words") or []
    if len(words) < 2:
        return None
    near = slots.near(float(row["start"]) - 0.5, float(row["end"]) + 0.5)
    labels = _clean_labels(words, near)
    clean = [(i, k) for i, k in enumerate(labels) if k is not None]
    if not clean:
        return None
    first = clean[0][1]
    change = next((n for n, (_, k) in enumerate(clean) if k != first), None)
    if change is None:
        return None
    second = clean[change][1]
    left, right = clean[:change], clean[change:]
    if any(k != second for _, k in right):
        return None
    seconds = [float(w["end"]) - float(w["start"]) for w in words]
    if sum(seconds[i] for i, _ in left) < MIN_SECONDS or sum(seconds[i] for i, _ in right) < MIN_SECONDS:
        return None
    last_a, first_b = left[-1][0], right[0][0]
    # The second slot's onset: where its segment touching the first clean B word starts.
    onset = float(words[first_b]["start"])
    for segment in near:
        if segment.track == second and segment.start <= float(words[first_b]["end"]) \
                and segment.end >= float(words[first_b]["start"]):
            onset = segment.start
            break
    cut = min(range(last_a + 1, first_b + 1), key=lambda b: abs(float(words[b]["start"]) - onset))
    text = [str(w.get("word", "")) for w in words]
    if _characters("".join(text[:cut])) < MIN_CHARS or _characters("".join(text[cut:])) < MIN_CHARS:
        return None
    return cut, (last_a + 1, first_b)


def _place(bounds: Sequence[segmentation.Boundary], cut: int, span: tuple[int, int]) -> int | None:
    """The lowest-scoring boundary in range that is not banned, ties to the acoustic cut, then left; None
    when every boundary in range is banned (the acoustic cut is in range, so it is banned too)."""
    low, high = span
    candidates = [(bounds[c - 1].b, abs(c - cut), c) for c in range(max(1, low), high + 1) if not bounds[c - 1].banned]
    return min(candidates)[2] if candidates else None


def _glue(bounds: Sequence[segmentation.Boundary], cut: int) -> int:
    if cut >= 2:
        before, after = bounds[cut - 2], bounds[cut - 1]
        if not before.banned and before.b < BARE <= after.b:
            return cut - 1
    return cut


def _majority(labels: Sequence[int | None]) -> int | None:
    counts = Counter(k for k in labels if k is not None)
    return counts.most_common(1)[0][0] if counts else None


def find_cuts(
    segments: Sequence[Mapping[str, Any]],
    *,
    slots: Sequence[TrackSegment],
    tracks: Sequence[TrackSegment],
    boundaries: Sequence[Sequence[segmentation.Boundary]],
) -> list[Cut]:
    """Every line the rule cuts, in order. `boundaries` is `segmentation.row_boundaries` of `segments`."""
    slot_index, track_index = _Index(slots), _Index(tracks)
    out: list[Cut] = []
    for row, segment in enumerate(segments):
        proposed = _propose(segment, slot_index)
        if proposed is None:
            continue
        acoustic, span = proposed
        bounds = boundaries[row]
        cut = _place(bounds, acoustic, span)
        if not cut:
            continue
        cut = _glue(bounds, cut)
        if bounds[cut - 1].b > GATE_B:
            continue
        words = segment["words"]
        if float(words[cut - 1]["end"]) <= float(segment["start"]) or float(words[cut]["start"]) >= float(segment["end"]):
            continue  # a half with no length: the glue can move the last clean A word right
        labels = _labels(segment, track_index)
        left, right = _majority(labels[:acoustic]), _majority(labels[acoustic:])
        if left is None or right is None or left == right:
            continue
        out.append(Cut(row, cut))
    return out


def cuts_for(payload: Mapping[str, Any], tracks: Sequence[TrackSegment], evidence: Any) -> list[Cut]:
    """The cuts for a whole stable payload: nothing without nemotron3's slots or the VAD intervals the DP
    scored against, and only on lines in a language the rule was read on.

    The language is each line's own `lang` (what recognition detected), falling back to the run's
    requested one: `metadata.asr_align.language` is `"None"` under the default auto-detection."""
    if evidence is None or getattr(evidence, "slot_grid", None) is None:
        return []
    segments = payload.get("segments") or []
    requested = ((payload.get("metadata") or {}).get("asr_align") or {}).get("language")
    if not any((segment.get("lang") or requested) in LANGUAGES for segment in segments):
        return []
    params = segmentation.split_params_of(payload)
    intervals = (payload.get("vad_timeline") or {}).get("intervals") or []
    boundaries = segmentation.row_boundaries(segments, intervals, params=params)
    if boundaries is None:
        return []
    grid = evidence.slot_grid
    slots = segments_from_activity(grid.tolist() if hasattr(grid, "tolist") else grid,
                                   frame_seconds=float(evidence.native_step))
    return [cut for cut in find_cuts(segments, slots=slots, tracks=tracks, boundaries=boundaries)
            if (segments[cut.row].get("lang") or requested) in LANGUAGES]
