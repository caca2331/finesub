from __future__ import annotations

import numpy as np

from tools.wt_refine_port.one_pass_probe import (
    split_timestamp_spans,
    trace_alignment_path,
)


def test_split_timestamp_spans_retains_trace_indices() -> None:
    spans = split_timestamp_spans(
        [1000, 10, 1100, 1150, 11, 1200],
        timestamp_begin=1000,
    )
    assert [span.raw_tokens for span in spans] == [
        (1000, 10, 1100),
        (1150, 11, 1200),
    ]
    assert [(span.token_start, span.token_end) for span in spans] == [(0, 2), (3, 5)]


def test_trace_alignment_path_keeps_global_frame_offset() -> None:
    attention = np.zeros((3, 2, 14), dtype=np.float32)
    attention[0, :, 10:14] = [4, 3, 1, 0]
    attention[1, :, 10:14] = [1, 4, 3, 0]
    attention[2, :, 10:14] = [0, 1, 3, 4]
    path = trace_alignment_path(
        attention,
        token_start=0,
        token_end=2,
        frame_start=10,
        frame_end=14,
        real_audio_frames=14,
        median_filter_width=1,
    )
    assert path[0][1] == 10
    assert path[-1][1] == 13
