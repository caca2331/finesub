"""Recognition-output segment timeline normalization."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ...text import (
    coerce_optional_float,
    min_word_confidence,
    normalized_compact,
    words_to_text,
)


ZERO_LENGTH_SEGMENT_EXTEND_SEC = 0.01

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
) -> Tuple[List[Dict[str, object]], List[str]]:
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

    Returns the surviving segments plus a description per dropped ghost.
    """

    dropped: List[str] = []
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
                f"start={start:.3f} text='{str(segment.get('text') or '')[:40]}'"
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


def extend_zero_length_segments(
    segments: List[Dict[str, object]],
    *,
    min_sec: float = ZERO_LENGTH_SEGMENT_EXTEND_SEC,
) -> List[Dict[str, object]]:
    """Give zero-length segments a minimal duration.

    The extension may squeeze the following segment. A squeezed segment that
    becomes zero-length is handled later in the same forward sweep.
    """

    out = [dict(segment) for segment in segments]
    for index, segment in enumerate(out):
        start = coerce_optional_float(segment.get("start"))
        end = coerce_optional_float(segment.get("end"))
        if start is None or end is None or end > start:
            continue
        new_end = start + min_sec
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
