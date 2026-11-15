from __future__ import annotations

from finesub.llm.client import extract_token_distribution
from finesub.llm.task_report import render_task_report


def test_render_task_report_aggregates_api_calls_and_tokens() -> None:
    records = [
        {
            "kind": "prepare_task_request",
            "payload": {
                "usage": {
                    "uncached_input_tokens": 100,
                    "cached_input_tokens": 10,
                    "total_input_tokens": 110,
                    "thinking_tokens": 20,
                    "output_tokens": 30,
                    "total_output_tokens": 50,
                }
            },
        },
        {
            "kind": "prepare_task_response",
            "payload": {
                "attempt": 1,
                "usage": {
                    "total_input_tokens": 50,
                    "thinking_tokens": 5,
                    "output_tokens": 10,
                    "total_output_tokens": 15,
                },
            },
        },
        {
            "kind": "correction_window_response",
            "payload": {
                "usage": {
                    "uncached_input_tokens": 1000,
                    "cached_input_tokens": 0,
                    "total_input_tokens": 1000,
                    "thinking_tokens": 200,
                    "output_tokens": 100,
                    "total_output_tokens": 300,
                }
            },
        },
        {
            "kind": "correction_query_response",
            "payload": {
                "chunk_id": "0001",
                "attempt": 0,
                "usage": {
                    "total_input_tokens": 40,
                    "output_tokens": 6,
                    "total_output_tokens": 6,
                },
            },
        },
        {
            "kind": "api_call",
            "payload": {"category": "gemini_file_upload", "filename": "0001.aac"},
        },
        {
            "kind": "api_call",
            "payload": {
                "category": "web_extract",
                "source": "extra_info_urls",
                "urls": ["https://example.com/a"],
                "executed": [{"provider": "exa", "url": "https://example.com/a"}],
            },
        },
        {
            "kind": "correction_window_call_error",
            "payload": {
                "chunk_id": "0002",
                "error_type": "APIError",
                "error": "403 PERMISSION_DENIED File v67dvd0wgpq4",
            },
        },
    ]

    text = render_task_report(records, task_id="yui")

    assert "llm_prepare_request: 1" in text
    assert "llm_prepare_delivery: 1" in text
    assert "llm_correction_query: 1" in text
    assert "llm_correction: 1" in text
    assert "gemini_file_upload: 1" in text
    assert "web_extract: 1" in text
    # The phase lines read the call records themselves: prepare 100 + window 1000 uncached.
    assert "uncached_input_tokens=1100" in text
    assert "Session Token Totals" in text
    assert "| prepare-request-attempt0 | 110 | 50 |" in text
    assert "| prepare-delivery-attempt1 | 50 | 15 |" in text
    assert "| correction-0001-request-attempt0 | 40 | 6 |" in text
    assert "| **task total** | 1200 | 371 |" in text
    assert "background-prefetched clip upload" in text
    assert "window `0002` Gemini File access denied" in text


def _routed(target_id: str, tier: str, model: str) -> dict:
    return {
        "effective_chain": [
            {"target_id": target_id, "provider_tier": tier, "model": model},
            {"target_id": "other", "provider_tier": "GEMINI_PAID", "model": model},
        ],
        "candidates": [
            {"target_id": target_id, "outcome": "success"},
            {"target_id": "other", "decision": "skipped"},
        ],
    }


def test_render_task_report_groups_tokens_the_way_they_are_billed() -> None:
    """One session can fall back across tiers and one tier serves many sessions.

    The session table answers "which round spent this"; neither table derives
    from the other, and the tier is what a bill is keyed on -- the same model on
    the free and the paid tier is two different charges.
    """

    records = [
        {
            "kind": "prepare_task_response",
            "payload": {
                "model": "gemini/gemini-3.6-flash",
                "route_decision": _routed(
                    "gemini-free-3_6-flash", "GEMINI_FREE", "gemini/gemini-3.6-flash"
                ),
                "usage": {
                    "uncached_input_tokens": 100,
                    "cached_input_tokens": 10,
                    "total_input_tokens": 110,
                    "thinking_tokens": 20,
                    "output_tokens": 30,
                },
            },
        },
        {
            "kind": "correction_window_response",
            "payload": {
                "model": "gemini/gemini-3.8-flash",
                "route_decision": _routed(
                    "gemini-paid-3_8-flash", "GEMINI_PAID", "gemini/gemini-3.8-flash"
                ),
                "usage": {
                    "uncached_input_tokens": 1000,
                    "cached_input_tokens": 400,
                    "total_input_tokens": 1400,
                    "thinking_tokens": 200,
                    "output_tokens": 100,
                },
            },
        },
        {
            "kind": "correction_window_response",
            "payload": {
                "model": "gemini/gemini-3.8-flash",
                "route_decision": _routed(
                    "gemini-paid-3_8-flash", "GEMINI_PAID", "gemini/gemini-3.8-flash"
                ),
                "usage": {
                    "uncached_input_tokens": 500,
                    "cached_input_tokens": 100,
                    "total_input_tokens": 600,
                    "thinking_tokens": 50,
                    "output_tokens": 25,
                },
            },
        },
    ]

    text = render_task_report(records, task_id="yui")

    assert "Provider Token Totals" in text
    assert (
        "| GEMINI_FREE | gemini/gemini-3.6-flash | 1 | 0 | 110 | 10 | 0 | 30 | 20 |" in text
    )
    # The two paid calls are one row, summed.
    assert (
        "| GEMINI_PAID | gemini/gemini-3.8-flash | 2 | 0 | 2000 | 500 | 0 | 125 | 250 |"
        in text
    )
    assert "| **task total** | | 3 | 0 | 2110 | 510 | 0 | 155 | 270 |" in text


def _free_36_call() -> dict:
    return {
        "kind": "prepare_task_response",
        "payload": {
            "model": "gemini/gemini-3.6-flash",
            "route_decision": _routed(
                "gemini-free-3_6-flash", "GEMINI_FREE", "gemini/gemini-3.6-flash"
            ),
            "usage": {"uncached_input_tokens": 100, "total_input_tokens": 100, "output_tokens": 30},
        },
    }


def test_a_call_that_names_no_provider_and_spent_nothing_is_not_a_bill_line() -> None:
    """A failed call with no winner and no usage used to render `unknown / unknown`.

    That row carried a call and zeros, and read as a provider the run could not
    identify -- when what happened is that no provider reported anything at all.
    The provider table is a bill; a line that bills nobody for nothing is noise.
    """

    failed = {
        "kind": "correction_window_response",
        "payload": {
            # The shape a response record really carries: every key, zero-filled.
            # An absent `usage` never reaches here, so testing only that would
            # pass while the row still rendered.
            "usage": extract_token_distribution({}),
            "route_decision": {
                "effective_chain": [
                    {
                        "target_id": "local-codebuddy-glm-5_3-flash",
                        "provider_tier": "LOCAL_CODEBUDDY",
                        "model": "glm-5.3-flash",
                    }
                ],
                "candidates": [
                    {"target_id": "local-codebuddy-glm-5_3-flash", "outcome": "failed"}
                ],
            },
        },
    }

    text = render_task_report([_free_36_call(), failed], task_id="t")

    assert "| unknown | unknown |" not in text
    assert "| **task total** | | 1 |" in text


def test_a_call_with_a_known_provider_counts_even_when_it_reported_no_usage() -> None:
    """The other half of the rule: an identified provider is still a bill line.

    A local agent that answered but whose CLI never emitted a usage event is a
    real call on a real tier. Dropping it would under-count calls to hide a
    reporting gap, which is the opposite of what this table is for.
    """

    answered_without_usage = {
        "kind": "correction_window_response",
        "payload": {
            "model": "glm-5.3-flash",
            "route_decision": _routed(
                "local-codebuddy-glm-5_3-flash", "LOCAL_CODEBUDDY", "glm-5.3-flash"
            ),
        },
    }

    text = render_task_report([_free_36_call(), answered_without_usage], task_id="t")

    assert "| LOCAL_CODEBUDDY | glm-5.3-flash | 1 | 0 | 0 | 0 | 0 | 0 |" in text
    assert "| **task total** | | 2 |" in text


def test_every_failed_call_is_named_under_fallbacks_and_warnings() -> None:
    """Not only the IP-risk and File-403 classes, which have sections of their own.

    A call that failed any other way -- an agent that left without a submit --
    used to leave this section reading "No fallback was recorded" and nothing
    else: true about fallbacks, silent about the failure a reader came to find.
    """

    call_error = {
        "kind": "correction_window_call_error",
        "payload": {
            "chunk_id": "0001",
            "attempt": 0,
            "error_type": "AgentRuntimeCallError",
            "error": (
                "agent session ended without an accepted submit\n"
                "(task status 'queued'; no submit seen)"
            ),
        },
    }

    text = render_task_report([_free_36_call(), call_error], task_id="t")
    section = text.split("## Fallbacks And Warnings", 1)[1]

    assert "No fallback was recorded" in section, "still true: nothing fell back"
    assert "Calls failed during this task" in section
    assert "`correction_window_call_error` failed: AgentRuntimeCallError" in section
    # Quoted on one line, so a multi-line error cannot break the bullet list.
    assert "without an accepted submit (task status 'queued'; no submit seen)" in section
    assert "chunk_id=0001" in section


def test_a_clean_run_lists_no_failed_calls() -> None:
    text = render_task_report([_free_36_call()], task_id="t")

    assert "Calls failed during this task" not in text


def test_provider_totals_read_a_bare_input_total_as_a_total(monkeypatch) -> None:
    """Not every usage payload splits cached from uncached.

    Reporting the uncached column would render such a call as zero input, which
    is worse than coarse in a table whose whole job is accounting -- so the
    primary column is the full prompt side and cached is the breakdown.

    The fallback to `model` also covers an artifact written before the winning
    candidate was traced.
    """

    text = render_task_report(
        [
            {
                "kind": "prepare_task_response",
                "payload": {
                    "model": "gemini/gemini-3.6-flash",
                    "usage": {"total_input_tokens": 70, "output_tokens": 2},
                },
            }
        ],
        task_id="yui",
    )

    assert "| unknown | gemini/gemini-3.6-flash | 1 | 0 | 70 | 0 | 0 | 2 | 0 |" in text


def test_render_task_report_describes_composed_postprocess_profiles() -> None:
    text = render_task_report(
        [
            {
                "kind": "final_srt",
                "payload": {
                    "path": "out/final.srt",
                    "postprocess": {
                        "profile": 0,
                        "applied_profiles": [1, 2],
                        "segment_count": 3,
                        "duration_extended": 2,
                        "flash_extended": 1,
                        "punctuation_replacements": 4,
                        "trimmed_lines": 1,
                    },
                },
            }
        ],
        task_id="postprocess",
    )

    assert "profile 0: steps 1→2, 3 segments" in text
    assert "duration 2, flash 1, punctuation 4, trimmed 1" in text


def test_render_task_report_includes_core_timing_workers_and_rounds() -> None:
    text = render_task_report(
        [],
        task_id="timed",
        run_metadata={
            "timing": {
                "stages": {
                    "download": {"status": "executed", "elapsed_sec": 1.25},
                    "asr": {"status": "executed", "elapsed_sec": 8.5},
                    "llm_harness": {"status": "executed", "elapsed_sec": 12.0},
                },
                "total_sec": 22.5,
            },
            "workers": {
                "vocal_separation": {"profile_limit": 2, "effective": 1},
                "asr": {"profile_limit": 2, "requested": 2, "effective": 1},
            },
            "llm_rounds": [
                {
                    "round": "research-r1",
                    "elapsed_sec": 3.0,
                    "api_sec": 2.5,
                    "api_attempts": 2,
                    "retries": 1,
                    "status": "completed",
                }
            ],
        },
    )

    assert "Download: 1.250s (executed)" in text
    assert "Pipeline total: 22.500s" in text
    assert "ASR WT: effective=1, requested=2, profile limit=2" in text
    assert "| research-r1 | 3.000s | 2.500s | 2 | 1 | completed |" in text
