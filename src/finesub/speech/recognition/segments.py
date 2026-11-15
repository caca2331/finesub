"""Recognition-output segment timeline normalization."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ...subtitles import line_groups
from ...text import (
    coerce_optional_float,
    min_word_confidence,
    normalized_compact,
    words_to_text,
)


ZERO_LENGTH_SEGMENT_EXTEND_SEC = 0.01

# How much real audio after a VAD interval the decode keeps before the
# synthetic gap (`transcribe.inserted_gap_parts`); the splitter's zones read it
# too. Lives here, not in `transcribe`, so the splitter -- and stabilize, which
# calls it -- stay torch-free.
GAP_KEEP_REAL_MAX_SEC = 0.7
# Synthetic silence inserted right before the next interval. Here rather than
# in `transcribe` so `run_pipeline`'s signature can carry it without torch.
DEFAULT_GAP_SEC = 0.3

# A ghost segment is frame-quantized to a few 20ms frames; with the 2+
# character minimum below, anything inside this span is already past the
# 20 chars/s rate that profile-2 treats as impossibly fast speech.
GHOST_SEGMENT_MAX_SPAN_SEC = 0.1
# Neighborhood searched for the duplicated text that identifies a ghost.
GHOST_SEGMENT_CONTEXT_SEC = 3.0
# Single normalized characters duplicate real neighbors far too easily.
GHOST_SEGMENT_MIN_CHARS = 2
# The decode itself must testify the segment is a squeezed chunk-tail
# artifact. Without this gate, real rapid repeats (twice-shouted calls, sung
# refrains) whose timing got quantized would match the duplicate check.
GHOST_SEGMENT_EVENT_TYPES = frozenset(
    {"zero_duration_chunk_tail", "alignment_stack"}
)


def _absorb_words_into(
    target: Dict[str, object],
    orphans: List[Dict[str, object]],
    *,
    as_prefix: bool,
) -> Dict[str, object]:
    """Merge orphaned words' text into ``target`` while keeping its time span."""

    merged = dict(target)
    orphan_text = words_to_text(orphans)
    target_text = str(merged.get("word") or "")
    if as_prefix:
        space = " " if bool(merged.get("space_before", False)) else ""
        merged["word"] = orphan_text + space + target_text
        merged["space_before"] = bool(orphans[0].get("space_before", False))
    else:
        space = " " if bool(orphans[0].get("space_before", False)) else ""
        merged["word"] = target_text + space + orphan_text
    confidence = min_word_confidence([merged] + orphans)
    if confidence is not None:
        merged["confidence"] = confidence
    return merged


def _segment_is_ghost(segment: Dict[str, object]) -> bool:
    if not segment.get("words"):
        return False
    start = coerce_optional_float(segment.get("start"))
    end = coerce_optional_float(segment.get("end"))
    if start is None or end is None:
        return False
    return end - start <= GHOST_SEGMENT_MAX_SPAN_SEC


def _segment_has_ghost_evidence(segment: Dict[str, object]) -> bool:
    events = segment.get("alignment_events")
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, dict)
        and str(event.get("type")) in GHOST_SEGMENT_EVENT_TYPES
        for event in events
    )


def drop_ghost_duplicate_segments(
    segments: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Drop whole-segment decode ghosts that echo a neighboring segment.

    A ghost must satisfy all three of:

    - frame-quantized span (<= ``GHOST_SEGMENT_MAX_SPAN_SEC`` with >= 2
      normalized chars, an impossible speaking rate);
    - decode evidence: the segment carries a ``zero_duration_chunk_tail`` /
      ``alignment_stack`` event, i.e. the decoder itself reported the squeeze
      (real rapid repeats with merely quantized timing carry no such event);
    - a duplicate source: its normalized text is contained in a non-ghost
      segment within ``GHOST_SEGMENT_CONTEXT_SEC``, so it is the decoder
      re-emitting adjacent content and removing it cannot lose real speech.

    Non-duplicate or event-less short segments are kept: those still go
    through the normal abnormality ladder.

    Returns the surviving segments plus one RECORD per dropped ghost
    (`index`/`start`/`end`/`text`) -- a span, not a sentence, so an audit
    can go back to the audio and ask what was really there.
    """

    dropped: List[Dict[str, object]] = []
    keys = [normalized_compact(str(segment.get("text") or "")) for segment in segments]
    out: List[Dict[str, object]] = []
    for index, segment in enumerate(segments):
        if not _segment_is_ghost(segment) or not _segment_has_ghost_evidence(
            segment
        ):
            out.append(segment)
            continue
        key = keys[index]
        if len(key) < GHOST_SEGMENT_MIN_CHARS:
            out.append(segment)
            continue
        start = coerce_optional_float(segment.get("start")) or 0.0
        end = coerce_optional_float(segment.get("end")) or start
        is_duplicate = False
        for other_index, other in enumerate(segments):
            if other_index == index or not keys[other_index]:
                continue
            if _segment_is_ghost(other):
                # Only real segments count as the echoed source: ghosts must
                # never confirm each other.
                continue
            other_start = coerce_optional_float(other.get("start")) or 0.0
            other_end = coerce_optional_float(other.get("end")) or other_start
            if (
                other_start - GHOST_SEGMENT_CONTEXT_SEC > end
                or other_end + GHOST_SEGMENT_CONTEXT_SEC < start
            ):
                continue
            if key in keys[other_index]:
                is_duplicate = True
                break
        if is_duplicate:
            dropped.append(
                {
                    "index": index,
                    "start": round(start, 3),
                    # ⚠ `end` is why this is a record and not a string. Without
                    # it a dropped ghost's span cannot be reconstructed, so
                    # nothing downstream (an audit, a second opinion from the
                    # referee) can ask what was actually in that audio -- and
                    # two ghosts sharing a start read as one entry written
                    # twice. Both cost a measurement in 2026-08-31's
                    # bench-baselines §20.
                    "end": round(end, 3),
                    "text": str(segment.get("text") or "")[:40],
                }
            )
            continue
        out.append(segment)
    return out, dropped


def drop_empty_segments(
    segments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Remove segments that contain neither words nor non-whitespace text."""

    cleaned: List[Dict[str, object]] = []
    for segment in segments:
        words = segment.get("words") or []
        text = segment.get("text")
        has_text = bool(str(text).strip()) if text is not None else False
        if words or has_text:
            cleaned.append(segment)
    return cleaned


def clamp_segment_overlaps(
    segments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Pull an overlapping segment end back to the next segment's start.

    Raw word coordinates are globally monotonic, so overlaps only come from
    speculative energy-based last-word extension. Words left beyond the new
    end merge into the nearest word on the owning side.
    """

    out = [dict(segment) for segment in segments]
    for index in range(len(out) - 1):
        previous, current = out[index], out[index + 1]
        previous_start = coerce_optional_float(previous.get("start"))
        previous_end = coerce_optional_float(previous.get("end"))
        current_start = coerce_optional_float(current.get("start"))
        if (
            previous_start is None
            or previous_end is None
            or current_start is None
            or current_start >= previous_end
        ):
            continue
        new_end = max(current_start, previous_start)
        previous["end"] = new_end
        surviving: List[Dict[str, object]] = []
        orphans: List[Dict[str, object]] = []
        for word in previous.get("words") or []:
            word_start = coerce_optional_float(word.get("start"))
            word_end = coerce_optional_float(word.get("end"))
            if word_start is not None and word_start >= new_end:
                orphans.append(word)
                continue
            if word_end is not None and word_end > new_end:
                word = dict(word)
                word["end"] = new_end
            surviving.append(word)
        if orphans:
            if surviving:
                surviving[-1] = _absorb_words_into(
                    surviving[-1], orphans, as_prefix=False
                )
            else:
                current_words = list(current.get("words") or [])
                if current_words:
                    current_words[0] = _absorb_words_into(
                        current_words[0], orphans, as_prefix=True
                    )
                    current["words"] = current_words
                    current["text"] = words_to_text(current_words)
                    previous["text"] = ""
                else:
                    for word in orphans:
                        word = dict(word)
                        word["start"] = new_end
                        word["end"] = new_end
                        surviving.append(word)
        previous["words"] = surviving
    return drop_empty_segments(out)


def enforce_word_containment(
    segments: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Make every segment span its own words, and record what had to change.

    "A segment's bounds are the span of its words" is the house invariant --
    `fw_refine_backend` rebuilds it on every model exit, `split_segments`
    derives piece bounds from it, `clamp_segment_overlaps` truncates words to
    keep it. It is re-derived in six places and asserted in none, and two of
    those places each maintain only half of it (`clamp_word_starts` only
    raises a segment start, `extend_zero_length_segments` only lifts an end),
    so a segment can reach the artifact not containing its own words. That
    costs text: every downstream consumer drops `end <= start` silently
    (`subtitles/rendering.py`, `llm/chunking.py`), and a word outside the
    bounds is a subtitle whose timing does not match what it says.

    ⚠ **Widening stops at the neighbours.** Growing a segment to cover a word
    that reaches into the next one would trade a containment defect for an
    overlap -- and the stable JSON and the LLM entry both keep overlaps (only
    the SRT path repairs them). So the expansion is capped at the neighbour's
    bound and the word that still sticks out is pulled in instead, which is
    the same trade `clamp_segment_overlaps` already makes upstream.
    ⚠ The cap never shrinks a segment below its own recorded span: when the
    neighbours already overlap -- 7 segments across the artifact bed do, up to
    12.6 s -- that is a content defect with its own owner (`resolve_overlaps`
    at render time, deliberately non-destructive: it never drops a cue).
    Resolving it here would silently move text between segments and delete the
    emptied one, which is a different change than this contract.

    This is the one place that answers for the artifact rather than for its
    own pass. Repairs are RECORDED, not warned about (owner, 2026-09-06):
    across 61 artifacts / 23648 segments the defect never changed what a
    viewer sees, so a per-run warning would cost more attention than it is
    worth -- but the record has to survive into the stable JSON so an audit
    can find these without re-running anything.

    Returns the repaired segments plus one record per repair.
    """

    out: List[Dict[str, object]] = []
    repairs: List[Dict[str, object]] = []
    for index, segment in enumerate(segments):
        words = segment.get("words") or []
        starts = [
            value
            for value in (coerce_optional_float(word.get("start")) for word in words)
            if value is not None
        ]
        ends = [
            value
            for value in (coerce_optional_float(word.get("end")) for word in words)
            if value is not None
        ]
        start = coerce_optional_float(segment.get("start"))
        end = coerce_optional_float(segment.get("end"))
        if not starts or not ends or start is None or end is None:
            out.append(segment)
            continue
        # min/max over ALL words, not first/last: the DP explicitly allows
        # word starts to be non-monotonic inside a piece, so a middle word can
        # be the one sticking out.
        new_start, new_end = start, end
        if min(starts) < start:
            # The REPAIRED previous end, not the original one. Both neighbours
            # may want the same gap -- `[1,2]` with a word to 2.8 next to
            # `[3,4]` with a word from 2.2 -- and reading the input twice lets
            # them both take it. Going left to right, the left neighbour is
            # already final and the right one has not been reached yet, so
            # this is the only bound that is settled.
            previous_end = (
                coerce_optional_float(out[-1].get("end")) if out else None
            )
            floor = float("-inf") if previous_end is None else min(start, previous_end)
            new_start = max(min(starts), floor)
        if max(ends) > end:
            following_start = (
                coerce_optional_float(segments[index + 1].get("start"))
                if index + 1 < len(segments)
                else None
            )
            ceiling = (
                float("inf") if following_start is None else max(end, following_start)
            )
            new_end = min(max(ends), ceiling)
        fitted = [_fit_word(word, new_start, new_end) for word in words]
        if new_start == start and new_end == end and fitted == words:
            out.append(segment)
            continue
        repaired = dict(segment)
        repaired["start"] = new_start
        repaired["end"] = new_end
        repaired["words"] = fitted
        repairs.append(
            {
                "index": index,
                "kind": "contain",
                "start": start,
                "end": end,
                "new_start": new_start,
                "new_end": new_end,
                "text": str(segment.get("text") or "")[:40],
            }
        )
        out.append(repaired)
    return out, repairs


def _fit_word(
    word: Dict[str, object], start: float, end: float
) -> Dict[str, object]:
    """A copy of ``word`` pulled inside ``[start, end]``, or ``word`` itself."""

    word_start = coerce_optional_float(word.get("start"))
    word_end = coerce_optional_float(word.get("end"))
    if word_start is None or word_end is None:
        return word
    fitted_start = min(max(word_start, start), end)
    fitted_end = min(max(word_end, fitted_start), end)
    if fitted_start == word_start and fitted_end == word_end:
        return word
    moved = dict(word)
    moved["start"] = fitted_start
    moved["end"] = fitted_end
    return moved


def _finite_span(item: Dict[str, object]) -> Tuple[float, float] | None:
    start = coerce_optional_float(item.get("start"))
    end = coerce_optional_float(item.get("end"))
    if (
        start is None
        or end is None
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
    ):
        return None
    return start, end


def uncovered_speech_pieces(
    intervals: Sequence[Dict[str, object]],
    segments: Sequence[Dict[str, object]],
) -> List[Tuple[float, float]]:
    """Parts of each VAD interval that no segment claims.

    Cut PER INTERVAL, deliberately: the silence between two intervals is not
    speech anybody expected to hear, so it is never part of a piece. Pieces
    are returned whole and unfiltered -- callers decide what size matters to
    them, which is the difference between "worth a second listen" (the
    referee's ``GAP_MIN_SEC``) and "the viewer lost this" (any size at all).
    """

    covered = sorted(
        span for span in (_finite_span(item) for item in segments) if span
    )
    pieces: List[Tuple[float, float]] = []
    for interval in intervals:
        span = _finite_span(interval)
        if span is None:
            continue
        current = [span]
        for cover_start, cover_end in covered:
            remaining: List[Tuple[float, float]] = []
            for piece_start, piece_end in current:
                if cover_end <= piece_start or cover_start >= piece_end:
                    remaining.append((piece_start, piece_end))
                    continue
                if cover_start > piece_start:
                    remaining.append((piece_start, cover_start))
                if cover_end < piece_end:
                    remaining.append((cover_end, piece_end))
            current = remaining
        pieces.extend(current)
    return pieces


def speech_gap_summary(
    intervals: Sequence[Dict[str, object]],
    segments: Sequence[Dict[str, object]],
    *,
    max_spans: int = 50,
) -> Dict[str, object]:
    """How much VAD speech the finished segment set leaves unclaimed.

    ⚠ A different question from the decode stage's per-group coverage: this
    one is measured AFTER stabilization drops. It measures temporal coverage,
    not whether missing audio contains real speech or text is correct. A group can pass its
    coverage check and still end up here, because stabilize drops segments
    afterwards.

    ⚠ And a different question from the referee's `collect_gaps`, which keeps
    only pieces long enough to be worth re-listening to. Fragmented loss --
    several 1-2 s pieces adding up -- is invisible to that criterion and
    visible here, which is the whole point of recording it.

    Observational, with no threshold attached, for the reason
    `vad_asr_stage.audio_coverage` gives: nothing here is a decision, and the
    pipeline exposes no knob to act on it with.

    ``spans`` holds the ``max_spans`` LONGEST pieces, listed in TIME order --
    selected by size because those are the ones worth opening the audio for,
    ordered by time because that is how a person reads them back. ⚠ So
    ``spans[0]`` is the earliest of the longest, not the longest; the longest
    one's duration is `longest_piece_sec`. The cap is there so a pathological
    artifact cannot turn its own audit trail into the biggest thing in the
    file.
    """

    pieces = uncovered_speech_pieces(intervals, segments)
    speech = sum(
        end - start
        for start, end in (
            span for span in (_finite_span(item) for item in intervals) if span
        )
    )
    uncovered = sum(end - start for start, end in pieces)
    longest = sorted(pieces, key=lambda piece: piece[1] - piece[0], reverse=True)
    return {
        "speech_sec": round(speech, 3),
        "uncovered_sec": round(uncovered, 3),
        "uncovered_ratio": round(uncovered / speech, 4) if speech > 0 else None,
        "pieces": len(pieces),
        "longest_piece_sec": (
            round(longest[0][1] - longest[0][0], 3) if longest else 0.0
        ),
        "spans": [
            [round(start, 3), round(end, 3)]
            for start, end in sorted(longest[:max_spans])
        ],
    }


def clamp_to_audio_end(
    segments: List[Dict[str, object]],
    audio_end: float,
    *,
    min_sec: float = ZERO_LENGTH_SEGMENT_EXTEND_SEC,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Pull segments (and their words) back inside ``[.., audio_end]``.

    A written segment that ends after the audio does is a defect nothing
    downstream can see: it renders as a cue hanging past the end of the media,
    and a player just shows it over black. Measured on a delivered artifact:
    the last cue ended 1.866 s past the source.

    Work backwards so consecutive out-of-bounds segments each keep a positive
    span. Moving a start back reserves space by shortening the preceding end;
    if that consumes its whole span, the reservation propagates backwards.
    This avoids both new overlaps and zero-span cues that renderers discard.
    Only the affected tail changes; this is a format repair, not inferred timing.
    """

    out = list(segments)
    repairs: List[Dict[str, object]] = []
    ceiling = audio_end
    step = min(min_sec, audio_end / max(len(segments), 1))
    for index in range(len(segments) - 1, -1, -1):
        segment = segments[index]
        start = coerce_optional_float(segment.get("start"))
        end = coerce_optional_float(segment.get("end"))
        if start is None or end is None:
            continue
        new_end = min(end, ceiling)
        new_start = start
        if start >= ceiling:
            new_start = max(0.0, new_end - step)
            ceiling = new_start
        else:
            # No start moved: there is no new overlap to propagate upstream.
            ceiling = audio_end
        if new_start == start and new_end == end:
            continue
        fitted = [_fit_word(word, new_start, new_end) for word in segment.get("words") or []]
        repaired = dict(segment)
        repaired["start"] = new_start
        repaired["end"] = new_end
        if segment.get("words") is not None:
            repaired["words"] = fitted
        repairs.append(
            {
                "index": index,
                "kind": "overrun",
                "start": start,
                "end": end,
                "new_start": new_start,
                "new_end": new_end,
                "audio_end": audio_end,
                "text": str(segment.get("text") or "")[:40],
            }
        )
        out[index] = repaired
    return out, list(reversed(repairs))


def normalize_segment_timeline(
    segments: List[Dict[str, object]],
    *,
    audio_end: float | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """The timeline contract every written artifact has to satisfy, per group
    key (`subtitles/line_groups.py`): lines of different speakers in a cloud
    task may overlap, so containment, the end clamp and the extension only
    look at the same speaker's neighbours. A repair's `index` is the
    segment's position in ``segments``. Without keys (a local task without the overlap lane) one
    group, exactly the pass below.
    """

    parts = line_groups.groups([line_groups.key_of(s) for s in segments])
    if len(parts) == 1:
        return _normalize_group(segments, audio_end=audio_end)
    out: List[Dict[str, object]] = []
    repairs: List[Dict[str, object]] = []
    for idx in parts:
        fixed, found = _normalize_group([segments[i] for i in idx], audio_end=audio_end)
        out.extend(fixed)
        repairs.extend(dict(r, index=idx[int(r["index"])]) for r in found)
    return line_groups.in_order(out), sorted(repairs, key=lambda r: int(r["index"]))


def _normalize_group(
    segments: List[Dict[str, object]],
    *,
    audio_end: float | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """The contract on one group's segments.

    Two existing passes: `enforce_word_containment` (a segment spans its own
    words, without growing into a neighbour) then `extend_zero_length_segments`
    (a zero-length segment is exactly what loses its text downstream).

    Called on BOTH artifact write paths. The recognition stage is where these
    defects are actually produced, but a stable JSON can also be rebuilt from
    an aligned artifact that predates those fixes -- or written by hand -- and
    "the stable JSON is well-formed" is the contract we owe downstream:
    `subtitles/rendering.py` and `llm/chunking.py` both drop `end <= start`
    silently, so a segment that loses the invariant loses its text with no
    trace at all.

    ⚠ Pre-existing overlaps are deliberately NOT repaired here (the reason is
    in `enforce_word_containment`); they are recorded, so an audit still finds
    them without re-running anything.

    ``audio_end`` adds the dimension the contract was missing: a segment may
    not end after the audio does. It is optional because the rebuild path can
    face an artifact that never recorded a duration -- there the dimension is
    simply not checked. ⚠ "Not checked" is NOT recorded here: this list is
    repairs, and a marker in it would be counted as one. The two callers each
    record whether they had a duration, which is where an audit looks anyway.
    """

    contained, repairs = enforce_word_containment(segments)
    if audio_end is not None:
        # Before the extension pass, whose own bound is `audio_end` from here
        # on: clamping after it would just let it push the tail back out.
        contained, clamped = clamp_to_audio_end(contained, audio_end)
        repairs.extend(clamped)
    # The extension pass fixes in place without reporting, and overlaps are
    # left to their own owner -- so both are recorded here, before the first
    # one disappears. A read-only scan: an audit needs "this artifact had
    # one", not the arithmetic of the repair.
    for index, segment in enumerate(contained):
        start = coerce_optional_float(segment.get("start"))
        end = coerce_optional_float(segment.get("end"))
        if start is None or end is None:
            continue
        previous_end = (
            coerce_optional_float(contained[index - 1].get("end")) if index else None
        )
        for kind, hit in (
            ("overlap", previous_end is not None and previous_end > start),
            ("zero-span", end <= start),
        ):
            if hit:
                repairs.append(
                    {
                        "index": index,
                        "kind": kind,
                        "start": start,
                        "end": end,
                        "text": str(segment.get("text") or "")[:40],
                    }
                )
    # The extension cascades: it pushes the following segment's start (and its
    # words) forward, which on an already out-of-order timeline can leave that
    # segment shorter than the words it just moved. Containment runs once more
    # over the result. Measured: never fires on the artifact bed (120 files),
    # and 89 of 4000 randomized adversarial timelines needed it -- all of them
    # with inverted words or out-of-order segments.
    extended, cascade = enforce_word_containment(
        extend_zero_length_segments(contained, audio_end=audio_end)
    )
    return extended, repairs + cascade


def extend_zero_length_segments(
    segments: List[Dict[str, object]],
    *,
    min_sec: float = ZERO_LENGTH_SEGMENT_EXTEND_SEC,
    audio_end: float | None = None,
) -> List[Dict[str, object]]:
    """Give zero-length segments a minimal duration.

    The extension may squeeze the following segment. A squeezed segment that
    becomes zero-length is handled later in the same forward sweep.

    ⚠ ``audio_end`` bounds the extension. Without it this pass and
    `clamp_to_audio_end` cancel each other on the last segment: the clamp
    pulls it to the audio end, the extension pushes it `min_sec` past. A
    segment already sitting exactly at ``audio_end`` therefore stays
    zero-length -- honest, and recorded as `zero-span` by the caller.
    """

    out = [dict(segment) for segment in segments]
    for index, segment in enumerate(out):
        start = coerce_optional_float(segment.get("start"))
        end = coerce_optional_float(segment.get("end"))
        if start is None or end is None or end > start:
            continue
        new_end = start + min_sec
        if audio_end is not None:
            new_end = min(new_end, audio_end)
        if new_end <= start:
            continue
        segment["end"] = new_end
        words = [dict(word) for word in segment.get("words") or []]
        if words:
            last = words[-1]
            last_start = coerce_optional_float(last.get("start"))
            last_end = coerce_optional_float(last.get("end"))
            if (
                last_start is not None
                and last_end is not None
                and last_end <= last_start
            ):
                last["end"] = new_end
            segment["words"] = words
        if index + 1 >= len(out):
            continue
        following = out[index + 1]
        following_start = coerce_optional_float(following.get("start"))
        if following_start is None or following_start >= new_end:
            continue
        following["start"] = new_end
        following_words: List[Dict[str, object]] = []
        for word in following.get("words") or []:
            word_start = coerce_optional_float(word.get("start"))
            if word_start is not None and word_start < new_end:
                word = dict(word)
                word["start"] = new_end
                word_end = coerce_optional_float(word.get("end"))
                if word_end is not None and word_end < new_end:
                    word["end"] = new_end
            following_words.append(word)
        following["words"] = following_words
    return out
