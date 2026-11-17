"""Timestamp splitting for MLX 0.4.3, retaining CT2-style unfinished tails.

The pinned upstream loop drops the tail after a consecutive timestamp pair.
FineSub aligns that tail using its actual final query, and consumes the full
window instead of decoding it again. Early EOT uses its recorded query as a
temporary alignment boundary; it is not an invented decoded timestamp.
"""
from __future__ import annotations

import numpy as np


def split_window(tokens, tokenizer, new_segment, result, time_offset,
                 segment_duration, seek, segment_size, input_stride, time_precision):
    values = np.asarray(tokens)
    timestamp = values >= tokenizer.timestamp_begin
    single_ending = timestamp[-2:].tolist() == [False, True]
    cuts = (np.flatnonzero(timestamp[:-1] & timestamp[1:]) + 1).tolist()
    segments = []
    if not cuts:
        duration = segment_duration
        stamps = values[timestamp]
        unfinished = bool(len(values) >= 2 and timestamp[0] and not timestamp[-1])
        if len(stamps) and not unfinished and stamps[-1] != tokenizer.timestamp_begin:
            duration = (int(stamps[-1]) - tokenizer.timestamp_begin) * time_precision
        segments.append(new_segment(start=time_offset, end=time_offset + duration,
                                    tokens=values, result=result))
        return segments, seek + segment_size, single_ending or unfinished
    tail_start = cuts[-1]
    tail = values[tail_start:]
    unfinished = bool(len(tail) >= 2 and tail[0] >= tokenizer.timestamp_begin
                      and tail[-1] < tokenizer.timestamp_begin)
    if single_ending or unfinished:
        cuts.append(len(values))
    start = 0
    for stop in cuts:
        part = values[start:stop]
        begin = (int(part[0]) - tokenizer.timestamp_begin) * time_precision
        end = (segment_duration if unfinished and stop == len(values)
               else (int(part[-1]) - tokenizer.timestamp_begin) * time_precision)
        segments.append(new_segment(start=time_offset + begin, end=time_offset + end,
                                    tokens=part, result=result))
        start = stop
    if single_ending or unfinished:
        return segments, seek + segment_size, True
    last_stamp = int(values[start - 1]) - tokenizer.timestamp_begin
    return segments, seek + last_stamp * input_stride, False


def adapt_seek_source(source: str) -> str:
    """Change one bounded block; fail on incompatible repackaged source.

    This executes only the installed, version-checked MIT-licensed 0.4.3
    transcribe function in a private globals namespace. All other seek,
    suppression and fallback behavior stays upstream. No module is mutated.
    """
    begin = "                timestamp_tokens = tokens >= tokenizer.timestamp_begin\n"
    end = "                if word_timestamps:\n                    add_word_timestamps(\n"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError("mlx-whisper 0.4.3 seek contract changed; reinstall pinned dependencies")
    left, remaining = source.split(begin, 1)
    _, right = remaining.split(end, 1)
    replacement = (
        "                current_segments, seek, single_timestamp_ending = _finesub_split_window(\n"
        "                    tokens, tokenizer, new_segment, result, time_offset,\n"
        "                    segment_duration, seek, segment_size, input_stride, time_precision)\n\n"
    )
    return left + replacement + end + right
