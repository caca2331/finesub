"""Word-start correction: disfluency-block handling plus VAD-anchored clamps.

Consumes the ``[*]`` disfluency blocks and leading ``disfluency_candidate``
events emitted by the fw-refine backend (``detect_disfluencies``), the VAD
energy track, the VAD speech intervals and the scorer's ``pause_hints``, and
rewrites first/next-word start times. Timestamp-only by construction: no rule
here ever removes or rewrites lexical text, and the anchor clamps only ever
move starts later.

Rules for a ``[*]`` block (consecutive ``[*]`` words collapse into one block;
the block always belongs to the next lexical word):

1. blocks shorter than ``MERGE_SHORT_SEC`` merge into the following word
   (too few energy frames to judge);
2. every handled block is recorded on the following word as
   ``disfluency_span`` = [block start, block end] plus ``disfluency_action``,
   word-level so both survive the DP re-segmentation;
3. blocks whose energy shape passes the quiet gate are "deleted": the
   following word starts at the end of the last quiet run inside the block
   (handles partial blocks whose onset lies mid-block). The gate is position-
   independent: a position restriction never added word-onset protection on
   gold — the energy gate alone measured 0/25 there across all positions —
   and only cost recall, so it was dropped (2026-08-05, user-approved; the
   one >1s hazard candidate was listened to and confirmed a previous-word
   residual). The only backstop is a cap: moves are truncated at
   ``DELETE_MOVE_CAP_SEC`` — expected to never fire, kept in case;
4. every other block merges into the following word (the plain-decode
   behavior).

Leading candidates (attention evidence before a segment's first lexical word;
never materialized as ``[*]``, see fw_refine.align_span_words) go through the
same gate: the refined start the backend adopted unconditionally is kept only
when the gate agrees, otherwise the word reverts to its original start
(bounded by the previous segment so overlap clamping cannot eat its tail).

Calibration and false-positive audits: docs/wt-refine-validation.md (gold
quiet_frac separation, zero word-onset deletions, VAD-lead distribution).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...text import DISFLUENCY_WORD, words_to_text
from ..preprocessing.energy import VadEnergyTrack
from .segments import coerce_optional_float

DISFLUENCY_SPAN_KEY = "disfluency_span"
DISFLUENCY_ACTION_KEY = "disfluency_action"
DISFLUENCY_QUIET_FRAC_KEY = "disfluency_quiet_frac"

# Blocks shorter than this merge back unconditionally: ~12 energy frames is
# the floor where the quiet-run measurement is reliable.
MERGE_SHORT_SEC = 0.12
# Quiet gate: fraction of block frames below the local reference minus
# QUIET_REL_DB. Gold (BV1cqLR6hEp3, n=61): filled_pause median 0.70,
# word_onset median 0.00; at 0.4 recall 25/32 with 0/25 word-onset deletions.
QUIET_GATE = 0.4
QUIET_REL_DB = 12.0
QUIET_REF_CONTEXT_SEC = 2.0
# Hard cap on how far a deletion may move a start. Largest observed block is
# 2.4s and both audited >1s cases were legitimate deletions, so this is not
# expected to ever fire — pure in-case insurance against a pathological block.
DELETE_MOVE_CAP_SEC = 3.0

# Interval-start clamp (layer 3): true onsets sit at a median +0.10s after the
# VAD interval start (gold, post-rework VAD), so a first word reporting
# earlier than S+CLAMP_LEAD_SEC is clamped up to it.
CLAMP_LEAD_SEC = 0.1
CLAMP_MAX_PULL_SEC = 0.5
CLAMP_MIN_END_AFTER_SEC = 0.15
CLAMP_MIN_PREV_GAP_SEC = 0.3
CLAMP_MIN_WORD_SEC = 0.05


def _track_frames(
    track: Optional[VadEnergyTrack],
) -> Tuple[Sequence[float], float]:
    """No track (standalone asr-align) degrades to an empty frame list: the
    quiet gate can never pass, so every block merges back — the plain-decode
    behavior plus span labels."""

    if track is None:
        return [], float("nan")
    energy = track.energy_db.detach().cpu().tolist()
    return energy, float(track.hop_sec)


def _quiet_run_end(
    energy: Sequence[float],
    hop_sec: float,
    start: float,
    end: float,
) -> Tuple[Optional[float], Optional[float]]:
    """(end of last quiet run inside [start, end], quiet_frac) or (None, qf).

    Quiet means below the median energy of [start-2s, end+2s] minus
    QUIET_REL_DB. Matches the gold calibration frame indexing exactly.
    """

    n = len(energy)
    if n <= 0 or not math.isfinite(hop_sec) or hop_sec <= 0:
        return None, None
    i0 = max(0, int(start / hop_sec))
    i1 = min(n, max(i0, int(end / hop_sec)))
    if i1 <= i0:
        return None, None
    r0 = max(0, int((start - QUIET_REF_CONTEXT_SEC) / hop_sec))
    r1 = min(n, int((end + QUIET_REF_CONTEXT_SEC) / hop_sec))
    context = sorted(energy[r0:r1])
    if not context:
        return None, None
    mid = len(context) // 2
    ref = (
        context[mid]
        if len(context) % 2
        else (context[mid - 1] + context[mid]) / 2.0
    )
    threshold = ref - QUIET_REL_DB
    quiet = [value < threshold for value in energy[i0:i1]]
    quiet_frac = sum(quiet) / len(quiet)
    last_quiet = None
    for offset, is_quiet in enumerate(quiet):
        if is_quiet:
            last_quiet = offset
    if last_quiet is None:
        return None, quiet_frac
    return start + (last_quiet + 1) * hop_sec, quiet_frac


def _finite(value: object) -> Optional[float]:
    number = coerce_optional_float(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def _apply_block(
    block_start: float,
    block_end: float,
    target: Dict[str, object],
    *,
    energy: Sequence[float],
    hop_sec: float,
    stats: Dict[str, int],
) -> None:
    """Resolve one disfluency block onto its following word (rules 1/3/4)."""

    action = "merge"
    new_start = block_start
    quiet_frac: Optional[float] = None
    if block_end - block_start < MERGE_SHORT_SEC:
        action = "merge_short"
    else:
        onset, quiet_frac = _quiet_run_end(
            energy, hop_sec, block_start, block_end
        )
        if quiet_frac is not None and quiet_frac >= QUIET_GATE and onset is not None:
            action = "delete"
            new_start = min(
                max(onset, block_start),
                block_end,
                block_start + DELETE_MOVE_CAP_SEC,
            )
    target[DISFLUENCY_SPAN_KEY] = [round(block_start, 3), round(block_end, 3)]
    target[DISFLUENCY_ACTION_KEY] = action
    if quiet_frac is not None:
        # The gate's own measurement, persisted because the action alone cannot
        # separate the two populations behind a *merged* block: on gold, a
        # filled pause sits at ~0.70 quiet and a following word's own onset at
        # ~0.00, and everything the gate merges is by definition below 0.4.
        # Anyone asking "is an absorbed disfluency evidence of a prosodic
        # boundary?" needs this number; it cannot be recovered afterwards
        # without the energy track.
        target[DISFLUENCY_QUIET_FRAC_KEY] = round(quiet_frac, 3)
    target["start"] = new_start
    stats[action] = stats.get(action, 0) + 1


def _resolve_disfluency_blocks(
    segment: Dict[str, object],
    *,
    previous_end: float,
    energy: Sequence[float],
    hop_sec: float,
    stats: Dict[str, int],
) -> Dict[str, object]:
    words = list(segment.get("words") or [])
    out_words: List[Dict[str, object]] = []
    index = 0
    changed = False
    while index < len(words):
        word = words[index]
        if str(word.get("word") or "") != DISFLUENCY_WORD:
            out_words.append(dict(word))
            index += 1
            continue
        changed = True
        block_start = _finite(word.get("start"))
        block_end = _finite(word.get("end"))
        index += 1
        # Consecutive blocks collapse into one span owned by the next word.
        while index < len(words) and str(words[index].get("word") or "") == DISFLUENCY_WORD:
            tail_end = _finite(words[index].get("end"))
            if tail_end is not None:
                block_end = tail_end if block_end is None else max(block_end, tail_end)
            index += 1
        if index >= len(words):
            # Trailing block with no following lexical word: nothing to own
            # it; drop the marker and leave every timestamp untouched.
            stats["orphan_dropped"] = stats.get("orphan_dropped", 0) + 1
            continue
        target = dict(words[index])
        index += 1
        target_end = _finite(target.get("end"))
        if block_start is None or block_end is None or target_end is None:
            out_words.append(target)
            continue
        if not out_words:
            # Extending a segment's first word earlier must not cross into the
            # previous segment: overlap clamping would eat its tail word.
            block_start = max(block_start, previous_end)
            block_start = min(block_start, block_end)
        _apply_block(
            block_start,
            block_end,
            target,
            energy=energy,
            hop_sec=hop_sec,
            stats=stats,
        )
        out_words.append(target)
    if not changed:
        return segment
    item = dict(segment)
    item["words"] = out_words
    if out_words:
        item["start"] = float(out_words[0]["start"])
        item["end"] = float(out_words[-1]["end"])
        item["text"] = words_to_text(out_words)
    return item


def _gate_leading_candidate(
    segment: Dict[str, object],
    *,
    previous_end: float,
    energy: Sequence[float],
    hop_sec: float,
    stats: Dict[str, int],
) -> Dict[str, object]:
    """Re-decide the backend's unconditional leading-start adoption."""

    events = segment.get("alignment_events")
    words = segment.get("words") or []
    if not isinstance(events, list) or not words:
        return segment
    candidate = next(
        (
            event
            for event in events
            if isinstance(event, dict)
            and str(event.get("type")) == "disfluency_candidate"
            and event.get("is_leading_word")
        ),
        None,
    )
    if candidate is None:
        return segment
    original = _finite(candidate.get("original_start"))
    refined = _finite(candidate.get("refined_start"))
    first = words[0]
    first_start = _finite(first.get("start"))
    if (
        original is None
        or refined is None
        or first_start is None
        or refined <= original
        # The adoption must still be visible on the word; segment splits or
        # earlier corrections may have detached it.
        or abs(first_start - refined) > 0.02
        or first.get(DISFLUENCY_ACTION_KEY) is not None
    ):
        return segment
    block_start = min(max(original, previous_end), refined)
    target = dict(first)
    _apply_block(
        block_start,
        refined,
        target,
        energy=energy,
        hop_sec=hop_sec,
        stats={},
    )
    action = f"leading_{target.get(DISFLUENCY_ACTION_KEY)}"
    target[DISFLUENCY_ACTION_KEY] = action
    stats[action] = stats.get(action, 0) + 1
    item = dict(segment)
    item["words"] = [target] + [dict(word) for word in words[1:]]
    item["start"] = float(target["start"])
    return item


def apply_disfluency_rules(
    segments: List[Dict[str, object]],
    *,
    energy_track: Optional[VadEnergyTrack],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Resolve all ``[*]`` blocks and leading candidates (rules 1-4)."""

    energy, hop_sec = _track_frames(energy_track)
    stats: Dict[str, int] = {}
    out: List[Dict[str, object]] = []
    previous_end = 0.0
    for segment in segments:
        item = _resolve_disfluency_blocks(
            segment,
            previous_end=previous_end,
            energy=energy,
            hop_sec=hop_sec,
            stats=stats,
        )
        item = _gate_leading_candidate(
            item,
            previous_end=previous_end,
            energy=energy,
            hop_sec=hop_sec,
            stats=stats,
        )
        end = _finite(item.get("end"))
        if end is not None:
            previous_end = max(previous_end, end)
        out.append(item)
    return out, stats


def _iter_words(
    segments: List[Dict[str, object]],
) -> List[Tuple[Dict[str, object], Dict[str, object], Optional[Dict[str, object]]]]:
    """(word, owning segment, previous word) chronologically."""

    flat: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            flat.append((word, segment))
    flat.sort(key=lambda item: (_finite(item[0].get("start")) or 0.0))
    out = []
    for index, (word, segment) in enumerate(flat):
        previous = flat[index - 1][0] if index else None
        out.append((word, segment, previous))
    return out


def _clamp_word(
    word: Dict[str, object],
    segment: Dict[str, object],
    anchor: float,
    lead: float,
    stats: Dict[str, int],
    key: str,
) -> None:
    start = _finite(word.get("start"))
    end = _finite(word.get("end"))
    if start is None or end is None:
        return
    new_start = min(anchor + lead, end - CLAMP_MIN_WORD_SEC)
    if new_start <= start:
        return
    word["start"] = new_start
    if _finite(segment.get("start")) == start:
        segment["start"] = new_start
    stats[key] = stats.get(key, 0) + 1


def clamp_word_starts(
    segments: List[Dict[str, object]],
    *,
    vad_intervals: Sequence[Dict[str, object]],
    pause_hints: Sequence[float],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Anchor clamps: interval starts (layer 3) and pause hints (layer 2).

    Both share the guard set and only ever move starts later; composing with
    the disfluency rules is therefore a max over proposals.
    """

    out = [
        {**segment, "words": [dict(word) for word in segment.get("words") or []]}
        for segment in segments
    ]
    stats: Dict[str, int] = {}
    ordered = _iter_words(out)

    def first_word_reaching(anchor: float):
        for word, segment, previous in ordered:
            end = _finite(word.get("end"))
            if end is not None and end > anchor:
                return word, segment, previous
        return None, None, None

    def guards_pass(word, previous, anchor: float, max_start: float) -> bool:
        start = _finite(word.get("start"))
        end = _finite(word.get("end"))
        if start is None or end is None:
            return False
        if not (anchor - CLAMP_MAX_PULL_SEC <= start < max_start):
            return False
        if end <= anchor + CLAMP_MIN_END_AFTER_SEC:
            return False
        previous_end = _finite(previous.get("end")) if previous else None
        if previous_end is not None and anchor - previous_end < CLAMP_MIN_PREV_GAP_SEC:
            return False
        return True

    for interval in vad_intervals:
        anchor = _finite(interval.get("start"))
        if anchor is None:
            continue
        word, segment, previous = first_word_reaching(anchor)
        if word is None:
            continue
        if guards_pass(word, previous, anchor, anchor + CLAMP_LEAD_SEC):
            _clamp_word(word, segment, anchor, CLAMP_LEAD_SEC, stats, "clamp_interval")

    interval_starts = [
        start
        for start in (_finite(item.get("start")) for item in vad_intervals)
        if start is not None
    ]
    for hint in pause_hints:
        anchor = coerce_optional_float(hint)
        if anchor is None:
            continue
        # Interval heads belong to layer 3 with its +0.1s lead; skip hints
        # that duplicate one.
        if any(abs(anchor - start) <= CLAMP_LEAD_SEC for start in interval_starts):
            continue
        word, segment, previous = first_word_reaching(anchor)
        if word is None:
            continue
        # The hint already sits ~40ms before the raw onset, so it clamps with
        # no extra lead.
        if guards_pass(word, previous, anchor, anchor):
            _clamp_word(word, segment, anchor, 0.0, stats, "clamp_hint")

    for segment in out:
        words = segment.get("words") or []
        if words:
            start = _finite(words[0].get("start"))
            if start is not None and (_finite(segment.get("start")) or 0.0) < start:
                segment["start"] = start
    return out, stats
