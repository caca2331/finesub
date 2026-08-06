from __future__ import annotations

from tools.wt_refine_port.full_window_probe import parse_decoded_span


def test_parse_decoded_span_uses_decoded_timestamps() -> None:
    span = parse_decoded_span(
        index=2,
        text=" hello",
        raw_tokens=[1000, 42, 43, 1100],
        timestamp_begin=1000,
        eot=900,
        refine_frames=50,
    )
    assert span is not None
    assert span.text_tokens == (42, 43)
    assert (span.frame_start, span.frame_end) == (0, 150)
    assert (span.start_token, span.end_token) == (1000, 1100)


def test_parse_decoded_span_rejects_missing_end_timestamp() -> None:
    assert parse_decoded_span(
        index=0,
        text="broken",
        raw_tokens=[1000, 42, 43],
        timestamp_begin=1000,
        eot=900,
        refine_frames=50,
    ) is None
