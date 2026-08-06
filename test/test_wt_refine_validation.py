from __future__ import annotations

import pytest

from tools.wt_refine_validation import run


def test_validation_group_selector_fails_closed_on_vad_drift() -> None:
    groups = [
        [{"start": 1.0, "end": 2.0}],
        [{"start": 3.0, "end": 4.0}],
    ]

    assert run.select_group(
        groups,
        {"mode": "production_group_containing", "interval_start": 3.0},
    ) == groups[1]
    with pytest.raises(RuntimeError, match="corpus drifted"):
        run.select_group(
            groups,
            {"mode": "production_group_containing", "interval_start": 3.1},
        )


def test_validation_route_requires_one_localized_hard_site() -> None:
    events = [
        {"type": "alignment_stack", "token_count": 4, "at": 2},
        {"type": "zero_duration_chunk_tail", "at": 2},
    ]
    route = run.route_events(
        events,
        interval_for_event=lambda event: event["at"],
    )

    assert route["route"] == "asr_immediate_isolation"
    assert route["interval_index"] == 2

    events.append({"type": "alignment_stack", "token_count": 3, "at": 3})
    route = run.route_events(
        events,
        interval_for_event=lambda event: event["at"],
    )
    assert route["route"] == "defer_finesub_regroup"


def test_disfluency_alone_is_observation_while_zero_tail_is_deferred() -> None:
    route = run.route_events(
        [{"type": "disfluency_candidate", "at": 1}],
        interval_for_event=lambda event: event["at"],
    )
    assert route["route"] == "keep_with_signals"

    route = run.route_events(
        [{"type": "zero_duration_chunk_tail", "at": 1}],
        interval_for_event=lambda event: event["at"],
    )
    assert route["route"] == "defer_finesub_decision"


def test_validation_similarity_normalizes_spacing_case_and_punctuation() -> None:
    assert run.edit_similarity("Hello, WORLD!", "hello world") == 1.0
