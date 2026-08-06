from __future__ import annotations

import pytest

from tools.wt_refine_port.state_machine import (
    AttentionRows,
    FlushTracker,
    partition_confidence,
    plan_alignment_repair,
    reached_decoding_limit,
)


TIMESTAMP_BEGIN = 1000
SOT = 900
EOT = 999


def test_consecutive_timestamps_flush_once() -> None:
    tracker = FlushTracker(timestamp_begin=TIMESTAMP_BEGIN, sot=SOT)
    assert not tracker.observe([42], [SOT, TIMESTAMP_BEGIN]).flush
    decision = tracker.observe([1100], [SOT, TIMESTAMP_BEGIN, 42, 1050])
    assert decision.flush
    assert not decision.discard
    assert tracker.saw_consecutive_timestamps

    boundary = tracker.observe([SOT, 1, 2], [1050])
    assert not boundary.flush
    assert boundary.discard
    assert not tracker.saw_consecutive_timestamps


def test_prompt_flushes_unclosed_segment() -> None:
    tracker = FlushTracker(timestamp_begin=TIMESTAMP_BEGIN, sot=SOT)
    decision = tracker.observe([SOT, 1, 2], [SOT, TIMESTAMP_BEGIN, 42])
    assert decision.flush
    assert not decision.discard


def test_modern_whisper_last_chunk_timestamp_forces_prompt_flush() -> None:
    tracker = FlushTracker(timestamp_begin=TIMESTAMP_BEGIN, sot=SOT)
    decision = tracker.observe(None, [SOT], last_chunk_token=1100)
    assert decision.flush


@pytest.mark.parametrize(
    ("without_sot", "prompt", "expected"),
    [(9, 3, True), (8, 3, False), (2, 15, True)],
)
def test_reached_decoding_limit(
    without_sot: int,
    prompt: int,
    expected: bool,
) -> None:
    assert reached_decoding_limit(
        chunk_tokens_without_sot=without_sot,
        initial_prompt_tokens=prompt,
        max_sample_len=11,
        text_context=17,
    ) is expected


def test_normal_end_timestamp_drops_its_prediction_query() -> None:
    plan = plan_alignment_repair(
        [SOT, TIMESTAMP_BEGIN, 42, 1100],
        timestamp_begin=TIMESTAMP_BEGIN,
        eot=EOT,
        unfinished_decoding=False,
        last_logprobs=[0.0] * 1200,
    )
    assert plan.alignment_tokens == (TIMESTAMP_BEGIN, 42, 1100)
    assert plan.recorded_segment_tokens == (SOT, TIMESTAMP_BEGIN, 42, 1100)
    assert plan.attention_rows is AttentionRows.DROP_LAST
    assert plan.last_logprobs_index == -2


def test_early_eot_appends_eot_and_keeps_all_queries() -> None:
    plan = plan_alignment_repair(
        [SOT, TIMESTAMP_BEGIN, 42],
        timestamp_begin=TIMESTAMP_BEGIN,
        eot=EOT,
        unfinished_decoding=False,
        last_logprobs=[0.0] * 1200,
    )
    assert plan.alignment_tokens == (TIMESTAMP_BEGIN, 42, EOT)
    assert plan.recorded_segment_tokens[-1] == EOT
    assert plan.appended_token == EOT
    assert plan.attention_rows is AttentionRows.ALL


def test_unfinished_decode_appends_fallback_and_marks_reliability() -> None:
    plan = plan_alignment_repair(
        [SOT, TIMESTAMP_BEGIN, 42],
        timestamp_begin=TIMESTAMP_BEGIN,
        eot=EOT,
        unfinished_decoding=True,
        fallback_token=43,
        fallback_reliable=False,
        last_logprobs=[0.0] * 1200,
    )
    assert plan.alignment_tokens == (TIMESTAMP_BEGIN, 42, 43)
    assert plan.recorded_segment_tokens[-1] == 43
    assert plan.unfinished_decoding
    assert not plan.last_token_reliable


def test_nonincreasing_end_timestamp_is_reestimated_only_for_alignment() -> None:
    logits = [0.0] * 1200
    logits[1042] = 5.0
    plan = plan_alignment_repair(
        [SOT, 1040, 42, 1040],
        timestamp_begin=TIMESTAMP_BEGIN,
        eot=EOT,
        unfinished_decoding=False,
        last_logprobs=logits,
    )
    assert plan.alignment_tokens[-1] == 1042
    assert plan.recorded_segment_tokens[-1] == 1040
    assert plan.reestimated_end_token == 1042


def test_confidence_skips_timestamp_boundaries() -> None:
    result = partition_confidence(
        chosen_tokens=[1000, 42, 1100, 1100, 43, 1200],
        chosen_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        segment_tokens=[[1000, 42, 1100], [1100, 43, 1200]],
    )
    assert result.average_logprob == pytest.approx(-0.35)
    assert result.segment_logprobs == ((-0.2,), (-0.5,))


def test_unfinished_confidence_keeps_final_fallback() -> None:
    result = partition_confidence(
        chosen_tokens=[1000, 42, 43],
        chosen_logprobs=[-0.1, -0.2, -0.3],
        segment_tokens=[[1000, 42, 43]],
        unfinished_last_segment=True,
        average_denominator=4,
    )
    assert result.average_logprob == pytest.approx(-0.15)
    assert result.segment_logprobs == ((-0.2, -0.3),)
