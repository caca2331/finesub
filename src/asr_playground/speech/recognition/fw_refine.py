"""One-pass faster-whisper word alignment compatible with WT refine.

The module deliberately contains no OpenAI Whisper or whisper-timestamped
imports.  It turns the compact trace emitted by the patched CTranslate2
greedy decoder into the word schema consumed by the existing ASR pipeline.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


AUDIO_TIME_PER_TOKEN = 0.02
MEDIAN_FILTER_WIDTH = 9
_REPLACEMENT = "�"
_PUNCTUATION = (
    "".join(character for character in string.punctuation if character not in "-'")
    + "。，！？：”、…"
)
_NO_SPACE_LANGUAGES = {
    "chinese",
    "japanese",
    "lao",
    "myanmar",
    "thai",
    "yue",
    "zh",
    "ja",
    "lo",
    "my",
    "th",
}


@dataclass(frozen=True)
class TimestampSpan:
    """A normal timestamp-delimited decoded segment and its trace rows."""

    token_start: int
    token_end: int
    tokens: tuple[int, ...]
    unfinished: bool = False


@dataclass(frozen=True)
class AlignedSpan:
    """WT-shaped words and confidence for one decoded segment."""

    words: tuple[dict[str, object], ...]
    confidence: float
    events: tuple[dict[str, object], ...] = ()


def should_use_space(language: str | None) -> bool:
    """Return WT's word-splitting mode for a language code or common name."""

    return (language or "en").strip().lower() not in _NO_SPACE_LANGUAGES


def split_timestamp_spans(
    tokens: Sequence[int],
    *,
    timestamp_begin: int,
) -> list[TimestampSpan]:
    """Split FW's timestamp-delimited segments while retaining trace indices."""

    values = tuple(int(token) for token in tokens)
    boundaries = [
        index
        for index in range(1, len(values))
        if values[index - 1] >= timestamp_begin and values[index] >= timestamp_begin
    ]
    if len(values) >= 2 and values[-2] < timestamp_begin <= values[-1]:
        boundaries.append(len(values))
    if (
        not boundaries
        and len(values) >= 3
        and values[0] >= timestamp_begin
        and values[-1] >= timestamp_begin
    ):
        boundaries.append(len(values))

    spans: list[TimestampSpan] = []
    start = 0
    for end in boundaries:
        raw = values[start:end]
        if len(raw) >= 3 and raw[0] >= timestamp_begin and raw[-1] >= timestamp_begin:
            spans.append(TimestampSpan(start, end - 1, raw))
        start = end
    return spans


def repair_early_eot_span(
    tokens: Sequence[int],
    *,
    timestamp_begin: int,
    eot: int,
    attention_steps: int,
    logprob_steps: int,
) -> TimestampSpan | None:
    """Append WT's temporary EOT boundary when decoding stops before an end timestamp.

    The patched generator retains the terminal EOT query and selected logprob
    even though faster-whisper strips EOT from ``sequences_ids``.  That makes
    this repair a genuine one-pass alignment rather than a teacher-force retry.
    """

    values = tuple(int(token) for token in tokens)
    if (
        len(values) < 2
        or values[0] < timestamp_begin
        or values[-1] >= timestamp_begin
        or attention_steps <= len(values)
        or logprob_steps <= len(values)
    ):
        return None
    repaired = (*values, int(eot))
    return TimestampSpan(0, len(values), repaired)


def repair_nonincreasing_end_span(
    span: TimestampSpan,
    *,
    timestamp_begin: int,
    endpoint_logprobs: Sequence[float],
) -> TimestampSpan:
    """Apply WT's constrained end-timestamp re-estimate to a temporary span."""

    start_token = span.tokens[0]
    end_token = span.tokens[-1]
    if start_token < timestamp_begin or end_token < timestamp_begin:
        return span
    if end_token > start_token:
        return span
    first_candidate = start_token + 1
    if first_candidate >= len(endpoint_logprobs):
        raise ValueError("tail logits do not contain a timestamp after the segment start")
    suffix = np.asarray(endpoint_logprobs[first_candidate:], dtype=np.float32)
    replacement = first_candidate + int(np.argmax(suffix))
    return TimestampSpan(
        span.token_start,
        span.token_end,
        (*span.tokens[:-1], replacement),
    )


def trace_alignment_path(
    attention: np.ndarray,
    *,
    span: TimestampSpan,
    frame_start: int,
    frame_end: int,
    real_audio_frames: int,
) -> list[tuple[int, int]]:
    """Run WT attention postprocessing and symmetric1 DTW on one trace span."""

    if attention.ndim != 3:
        raise ValueError("attention must have shape steps x heads x frames")
    if not 0 <= frame_start < frame_end <= attention.shape[-1]:
        raise ValueError("invalid alignment frame range")
    rows = attention[span.token_start : span.token_end + 1]
    if rows.shape[0] != len(span.tokens):
        raise ValueError("trace rows do not cover the complete timestamp span")

    weights = rows.transpose(1, 0, 2)[..., frame_start:frame_end].astype(
        np.float32,
        copy=True,
    )
    weights = median_filter(
        weights,
        size=(1, 1, MEDIAN_FILTER_WIDTH),
        mode="reflect",
    )
    weights -= weights.max(axis=-1, keepdims=True)
    np.exp(weights, out=weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    weights = weights.mean(axis=0)
    weights /= np.maximum(
        np.linalg.norm(weights, axis=0, keepdims=True),
        np.finfo(np.float32).tiny,
    )

    local_real_end = int(real_audio_frames) - frame_start
    if 0 < local_real_end < weights.shape[1]:
        weights[:-1, local_real_end:] = 0
    weights[0, 0] = weights.max()

    import dtw

    alignment = dtw.dtw(-weights.astype(np.float64), step_pattern=dtw.stepPattern.symmetric1)
    return [
        (int(token), int(frame) + frame_start)
        for token, frame in zip(alignment.index1s, alignment.index2s)
    ]


def _decode_with_timestamps(tokenizer: Any, tokens: Sequence[int]) -> str:
    decoder = getattr(tokenizer, "decode_with_timestamps", None)
    if decoder is not None:
        return str(decoder(list(tokens)))
    return str(tokenizer.decode(list(tokens)))


def split_tokens_on_unicode(
    tokens: Sequence[int],
    tokenizer: Any,
) -> tuple[list[str], list[list[str]], list[list[int]]]:
    """Split tokens at valid Unicode boundaries, gluing trailing punctuation.

    Every token must land in exactly one group: the caller pairs the groups with
    the decoder trace, so a dropped token desynchronises the whole span.
    """

    def decode(subset: Sequence[int]) -> str:
        return _decode_with_timestamps(
            tokenizer,
            [
                token
                for token in subset
                if token < tokenizer.eot or token >= tokenizer.timestamp_begin
            ],
        )

    words: list[str] = []
    token_texts: list[list[str]] = []
    token_ids: list[list[int]] = []
    current: list[int] = []
    decoded_full = decode([int(token) for token in tokens])
    unicode_offset = 0

    for raw_token in tokens:
        current.append(int(raw_token))
        decoded = decode(current)
        if _REPLACEMENT in decoded:
            # Usually an unfinished multi-byte character that the next token
            # completes. When the same offset is unrepresentable in the full
            # decode as well no token ever will, so emit it now rather than
            # carry it to the end of the loop and silently drop it.
            position = unicode_offset + decoded.index(_REPLACEMENT)
            if position >= len(decoded_full) or decoded_full[position] != _REPLACEMENT:
                continue
        empty = [""] * (len(current) - 1)
        punctuation = bool(decoded.strip()) and decoded.strip() in _PUNCTUATION
        previous_special = bool(token_ids) and token_ids[-1][-1] >= tokenizer.timestamp_begin
        if punctuation and not previous_special:
            if not words:
                words.append("")
                token_texts.append([])
                token_ids.append([])
            words[-1] += decoded
            token_texts[-1].extend([*empty, decoded])
            token_ids[-1].extend(current)
        else:
            words.append(decoded)
            token_texts.append([*empty, decoded])
            token_ids.append(list(current))
        unicode_offset += len(decoded)
        current.clear()
    if current:
        # Defensive: keep the group/token invariant total even if a decode never
        # resolves, so the caller fails on real trace mismatches only.
        decoded = decode(current)
        words.append(decoded)
        token_texts.append([*[""] * (len(current) - 1), decoded])
        token_ids.append(list(current))
    return words, token_texts, token_ids


def split_tokens_on_spaces(
    tokens: Sequence[int],
    tokenizer: Any,
) -> tuple[list[str], list[list[str]], list[list[int]]]:
    """Merge Unicode-safe pieces using WT's space-delimited language rules."""

    subwords, subword_texts, subword_ids = split_tokens_on_unicode(tokens, tokenizer)
    words: list[str] = []
    token_texts: list[list[str]] = []
    token_ids: list[list[int]] = []
    for index, (subword, texts, ids) in enumerate(
        zip(subwords, subword_texts, subword_ids)
    ):
        special = ids[0] >= tokenizer.timestamp_begin
        previous_special = index > 0 and subword_ids[index - 1][0] >= tokenizer.timestamp_begin
        next_special = (
            index + 1 < len(subword_ids)
            and subword_ids[index + 1][0] >= tokenizer.timestamp_begin
        )
        previous_space = index > 0 and not subwords[index - 1].strip()
        is_space = not subword.strip()
        with_space = subword.startswith(" ") and not is_space
        punctuation = not is_space and subword.strip() in _PUNCTUATION
        starts_word = special or (
            not previous_space
            and (
                previous_special
                or (with_space and not punctuation)
                or (is_space and not next_special)
            )
        )
        if starts_word or not words:
            words.append(subword.strip())
            token_texts.append(list(texts))
            token_ids.append(list(ids))
        else:
            words[-1] += subword.strip()
            token_texts[-1].extend(texts)
            token_ids[-1].extend(ids)
    return words, token_texts, token_ids


def _confidence(logprobs: Sequence[float]) -> float:
    return round(math.exp(sum(logprobs) / len(logprobs)), 3) if logprobs else 0.0


def _alignment_stack_events(
    span: TimestampSpan,
    jump_frames: np.ndarray,
) -> list[dict[str, object]]:
    """Describe text-token runs consuming at most one 20 ms frame each."""

    content_stop = len(span.tokens) if span.unfinished else len(span.tokens) - 1
    events: list[dict[str, object]] = []
    run_start: int | None = None

    def finish(run_end: int) -> None:
        nonlocal run_start
        if run_start is None:
            return
        token_count = run_end - run_start
        if token_count >= 3:
            frame_begin = int(jump_frames[run_start])
            frame_end = int(jump_frames[run_end])
            active_frames = max(0, frame_end - frame_begin)
            events.append(
                {
                    "type": "alignment_stack",
                    "token_start": run_start,
                    "token_end": run_end,
                    "token_count": token_count,
                    "start": round(frame_begin * AUDIO_TIME_PER_TOKEN, 3),
                    "end": round(frame_end * AUDIO_TIME_PER_TOKEN, 3),
                    "active_frames": active_frames,
                    "tokens_per_active_frame": round(
                        token_count / max(1, active_frames), 3
                    ),
                }
            )
        run_start = None

    for token_index in range(1, max(1, content_stop)):
        frame_advance = int(jump_frames[token_index + 1] - jump_frames[token_index])
        if frame_advance * AUDIO_TIME_PER_TOKEN >= 5.0:
            events.append(
                {
                    "type": "long_token_span",
                    "token_index": token_index,
                    "start": round(
                        int(jump_frames[token_index]) * AUDIO_TIME_PER_TOKEN, 3
                    ),
                    "end": round(
                        int(jump_frames[token_index + 1]) * AUDIO_TIME_PER_TOKEN, 3
                    ),
                    "duration": round(frame_advance * AUDIO_TIME_PER_TOKEN, 3),
                }
            )
        if frame_advance <= 1:
            if run_start is None:
                run_start = token_index
            continue
        finish(token_index)
    finish(content_stop)
    return events


def _decoder_repetition_events(
    span: TimestampSpan,
    jump_frames: np.ndarray,
) -> list[dict[str, object]]:
    """Return the strongest exact consecutive token motif in the span."""

    content_stop = len(span.tokens) if span.unfinished else len(span.tokens) - 1
    content = span.tokens[1:content_stop]
    best: tuple[int, int, int] | None = None
    for start in range(len(content)):
        remaining = len(content) - start
        for motif_size in range(1, min(16, remaining // 4) + 1):
            motif = content[start : start + motif_size]
            repeat_count = 1
            while (
                start + (repeat_count + 1) * motif_size <= len(content)
                and content[
                    start + repeat_count * motif_size : start + (repeat_count + 1) * motif_size
                ]
                == motif
            ):
                repeat_count += 1
            token_count = motif_size * repeat_count
            if repeat_count < 4 or token_count < 8:
                continue
            candidate = (token_count, start, motif_size)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return []
    token_count, content_start, motif_size = best
    token_start = content_start + 1
    token_end = token_start + token_count
    return [
        {
            "type": "decoder_repetition",
            "token_start": token_start,
            "token_end": token_end,
            "token_count": token_count,
            "motif_token_count": motif_size,
            "repeat_count": token_count // motif_size,
            "start": round(
                int(jump_frames[token_start]) * AUDIO_TIME_PER_TOKEN, 3
            ),
            "end": round(int(jump_frames[token_end]) * AUDIO_TIME_PER_TOKEN, 3),
        }
    ]


def _boundary_uncertainty_events(
    span: TimestampSpan,
    weights: np.ndarray,
    frame_start: int,
) -> list[dict[str, object]]:
    """Expose uncalibrated first/last text-query concentration metrics."""

    content_stop = len(span.tokens) if span.unfinished else len(span.tokens) - 1
    if content_stop <= 1:
        return []
    rows = (("start", 1), ("end", content_stop - 1))
    events: list[dict[str, object]] = []
    for boundary, token_index in rows:
        if token_index >= weights.shape[0] or weights.shape[1] == 0:
            continue
        values = np.asarray(weights[token_index], dtype=np.float64)
        values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
        total = float(values.sum())
        if total <= 0:
            continue
        probabilities = values / total
        nonzero = probabilities[probabilities > 0]
        entropy = -float(np.sum(nonzero * np.log(nonzero)))
        if len(values) > 1:
            entropy /= math.log(len(values))
        order = np.partition(values, -2) if len(values) > 1 else values
        highest = float(order[-1])
        second = float(order[-2]) if len(order) > 1 else 0.0
        peak_index = int(np.argmax(values))
        events.append(
            {
                "type": "boundary_uncertainty",
                "boundary": boundary,
                "token_index": token_index,
                "normalized_entropy": round(entropy, 6),
                "secondary_peak_ratio": round(second / highest, 6)
                if highest > 0
                else 0.0,
                "peak_time": round(
                    (int(frame_start) + peak_index) * AUDIO_TIME_PER_TOKEN, 3
                ),
                "touches_weight_window": peak_index <= 1
                or peak_index >= len(values) - 2,
            }
        )
    return events


def _disfluency_starts(
    *,
    span: TimestampSpan,
    path: Sequence[tuple[int, int]],
    weights: np.ndarray,
    frame_start: int,
    tokenizer: Any,
) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    """Return WT's multi-peak word starts and candidate gap intervals.

    Only the independent start coordinate is adjusted.  Lexical end
    coordinates continue to come from the unmodified DTW jumps.
    """

    token_indices = np.asarray([item[0] for item in path], dtype=np.int64)
    frame_indices = np.asarray([item[1] for item in path], dtype=np.int64)
    jumps = np.pad(np.diff(token_indices), (1, 0), constant_values=1).astype(bool)
    jump_frames = frame_indices[jumps]
    jump_frames = np.pad(jump_frames, (0, 1), constant_values=frame_indices[-1])
    starts = jump_frames.copy()

    disfluencies: dict[int, dict[str, object]] = {}
    local_frame_start = int(frame_start)
    for token_index, (begin, end) in enumerate(zip(jump_frames[:-1], jump_frames[1:])):
        local_begin = max(0, int(begin) - local_frame_start)
        local_end = min(weights.shape[1], int(end) - local_frame_start)
        if local_end <= local_begin or token_index >= weights.shape[0]:
            continue
        peaks, properties = find_peaks(
            weights[token_index, local_begin:local_end],
            width=3,
            prominence=0.02,
        )
        left_ips = properties.get("left_ips")
        if left_ips is None or len(left_ips) <= 1:
            continue
        new_begin = int(round(float(left_ips[-1]))) + local_begin + local_frame_start
        token_text = _decode_with_timestamps(tokenizer, [span.tokens[token_index]])
        target_token = token_index + 1 if token_text in _PUNCTUATION else token_index
        starts[token_index] = new_begin
        prominences = properties.get("prominences")
        disfluencies[target_token] = {
            "type": "disfluency_candidate",
            "token_index": target_token,
            "original_start": round(int(begin) * AUDIO_TIME_PER_TOKEN, 3),
            "refined_start": round(new_begin * AUDIO_TIME_PER_TOKEN, 3),
            "peak_count": int(len(peaks)),
            "max_prominence": round(float(np.max(prominences)), 6)
            if prominences is not None and len(prominences)
            else None,
        }

    return starts, disfluencies


def align_span_words(
    *,
    span: TimestampSpan,
    path: Sequence[tuple[int, int]],
    tokenizer: Any,
    language: str | None,
    chosen_logprobs: Sequence[float],
    alignment_weights: Sequence[float] | np.ndarray = (),
    alignment_frame_start: int = 0,
    detect_disfluencies: bool = False,
    collect_refine_signals: bool = False,
    collect_attention_signals: bool = False,
) -> AlignedSpan:
    """Convert one WT DTW path into production words and confidences."""

    if len(chosen_logprobs) != len(span.tokens):
        raise ValueError("chosen logprobs do not match the decoded timestamp span")
    if not path:
        return AlignedSpan((), 0.0)

    splitter = split_tokens_on_spaces if should_use_space(language) else split_tokens_on_unicode
    words, token_texts, word_token_ids = splitter(span.tokens, tokenizer)
    token_indices = np.asarray([item[0] for item in path], dtype=np.int64)
    frame_indices = np.asarray([item[1] for item in path], dtype=np.int64)
    jumps = np.pad(np.diff(token_indices), (1, 0), constant_values=1).astype(bool)
    jump_frames = frame_indices[jumps]
    jump_frames = np.pad(jump_frames, (0, 1), constant_values=frame_indices[-1])
    boundaries = np.pad(np.cumsum([len(group) for group in token_texts]), (1, 0))
    punctuation_counts = np.asarray(
        [0 if len(group) == 1 or group[-1] not in _PUNCTUATION else 1 for group in token_texts],
        dtype=np.int64,
    )
    start_frames = jump_frames
    events: list[dict[str, object]] = []
    if collect_refine_signals:
        events.extend(_alignment_stack_events(span, jump_frames))
        events.extend(_decoder_repetition_events(span, jump_frames))
        if span.unfinished:
            events.append({"type": "unfinished", "token_count": len(span.tokens) - 1})

    disfluencies: dict[int, dict[str, object]] = {}
    if detect_disfluencies or collect_attention_signals:
        weights = np.asarray(alignment_weights, dtype=np.float32)
        if weights.size == 0 or weights.size % len(span.tokens):
            raise ValueError("refine signal collection requires rectangular alignment weights")
        weights = weights.reshape(len(span.tokens), -1)
        candidate_starts, disfluencies = _disfluency_starts(
            span=span,
            path=path,
            weights=weights,
            frame_start=alignment_frame_start,
            tokenizer=tokenizer,
        )
        if detect_disfluencies:
            start_frames = candidate_starts
        if collect_attention_signals:
            events.extend(_boundary_uncertainty_events(span, weights, alignment_frame_start))
        # ``detect_disfluencies`` changes word starts, so its candidates must
        # remain auditable even when the much noisier boundary metrics are not
        # requested.  This also lets the ASR controller pass the evidence to a
        # downstream consumer without enabling two per-span uncertainty rows.
        content_word_stop = (
            len(word_token_ids) if span.unfinished else len(word_token_ids) - 1
        )
        valid_word_starts = {
            int(boundaries[word_index])
            for word_index in range(1, content_word_stop)
        }
        if valid_word_starts:
            first_word_start = min(valid_word_starts)
            last_word_start = max(valid_word_starts)
            for token_start, item in disfluencies.items():
                if token_start not in valid_word_starts:
                    continue
                event = dict(item)
                event["is_leading_word"] = token_start == first_word_start
                event["is_trailing_word"] = token_start == last_word_start
                events.append(event)

    starts = start_frames[boundaries[:-1]] * AUDIO_TIME_PER_TOKEN
    ends = jump_frames[boundaries[1:] - punctuation_counts] * AUDIO_TIME_PER_TOKEN

    if detect_disfluencies and disfluencies:
        additions: list[tuple[int, float, float]] = []
        token_start = 0
        for word_index, group in enumerate(word_token_ids[:-1]):
            # A candidate before the first lexical word describes empty leading
            # gap, not a display word.  Keeping it as ``[*]`` can move the
            # segment boundary backwards and make downstream overlap clamping
            # shorten the preceding segment's final word.  Retain the refined
            # lexical start but do not materialize that boundary-affecting gap.
            if token_start in disfluencies and word_index > 1:
                candidate = disfluencies[token_start]
                begin = float(candidate["original_start"])
                end = float(candidate["refined_start"])
                additions.append(
                    (
                        word_index,
                        begin,
                        end,
                    )
                )
            token_start += len(group)
        for word_index, begin, end in reversed(additions):
            words.insert(word_index, "[*]")
            token_texts.insert(word_index, [])
            word_token_ids.insert(word_index, [])
            starts = np.insert(starts, word_index, begin)
            ends = np.insert(ends, word_index, end)

    output: list[dict[str, object]] = []
    content_slice = slice(1, None) if span.unfinished else slice(1, -1)
    text_logprobs = tuple(float(value) for value in chosen_logprobs[content_slice])
    cursor = 0
    segment_confidence_values: list[float] = []
    for word, group, start, end in zip(
        words[content_slice],
        token_texts[content_slice],
        starts[content_slice],
        ends[content_slice],
    ):
        group_logprobs = text_logprobs[cursor : cursor + len(group)]
        cursor += len(group)
        confidence_group = list(group_logprobs)
        while len(confidence_group) > 1 and group[len(confidence_group) - 1] in _PUNCTUATION:
            confidence_group.pop()
        segment_confidence_values.extend(confidence_group)
        if word and not word.startswith("<|"):
            output.append(
                {
                    "word": word,
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                    "confidence": _confidence(confidence_group),
                }
            )
    if cursor != len(text_logprobs):
        raise ValueError("word token groups do not consume the decoded text tokens")
    if collect_refine_signals:
        lexical_words = [word for word in output if word["word"] != "[*]"]
        if lexical_words:
            tail = lexical_words[-1]
            if float(tail["end"]) <= float(tail["start"]):
                events.append(
                    {
                        "type": "zero_duration_chunk_tail",
                        "word": str(tail["word"]),
                        "start": float(tail["start"]),
                        "end": float(tail["end"]),
                    }
                )
    return AlignedSpan(
        tuple(output),
        _confidence(segment_confidence_values),
        tuple(events),
    )
