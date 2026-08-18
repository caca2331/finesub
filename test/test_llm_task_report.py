from __future__ import annotations

from finesub.llm.task_report import render_task_report


def test_render_task_report_aggregates_api_calls_and_tokens() -> None:
    records = [
        {
            "kind": "research_round1_response",
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
            "kind": "fast_round1_response",
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
            "kind": "token_distribution_report",
            "payload": {
                "phase": "research",
                "totals": {
                    "call_count": 1,
                    "uncached_input_tokens": 100,
                    "cached_input_tokens": 10,
                    "total_input_tokens": 110,
                    "thinking_tokens": 20,
                    "output_tokens": 30,
                    "total_output_tokens": 50,
                },
            },
        },
        {
            "kind": "token_distribution_report",
            "payload": {
                "phase": "correction",
                "totals": {
                    "call_count": 1,
                    "uncached_input_tokens": 1000,
                    "cached_input_tokens": 0,
                    "total_input_tokens": 1000,
                    "thinking_tokens": 200,
                    "output_tokens": 100,
                    "total_output_tokens": 300,
                },
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

    assert "llm_research_round1: 1" in text
    assert "llm_fast_round1: 1" in text
    assert "llm_correction_query: 1" in text
    assert "llm_correction: 1" in text
    assert "gemini_file_upload: 1" in text
    assert "web_extract: 1" in text
    assert "uncached_input_tokens=1100" in text
    assert "Session Token Totals" in text
    assert "| research-round1-attempt0 | 110 | 50 |" in text
    assert "| fast-round1-attempt1 | 50 | 15 |" in text
    assert "| correction-0001-query-attempt0 | 40 | 6 |" in text
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
            "kind": "research_round1_response",
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
                "model": "gemini/gemini-3.7-flash",
                "route_decision": _routed(
                    "gemini-paid-3_7-flash", "GEMINI_PAID", "gemini/gemini-3.7-flash"
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
                "model": "gemini/gemini-3.7-flash",
                "route_decision": _routed(
                    "gemini-paid-3_7-flash", "GEMINI_PAID", "gemini/gemini-3.7-flash"
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
        "| GEMINI_FREE | gemini/gemini-3.6-flash | 1 | 110 | 10 | 30 | 20 |" in text
    )
    # The two paid calls are one row, summed.
    assert (
        "| GEMINI_PAID | gemini/gemini-3.7-flash | 2 | 2000 | 500 | 125 | 250 |"
        in text
    )
    assert "| **task total** | | 3 | 2110 | 510 | 155 | 270 |" in text


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
                "kind": "research_round1_response",
                "payload": {
                    "model": "gemini/gemini-3.6-flash",
                    "usage": {"total_input_tokens": 70, "output_tokens": 2},
                },
            }
        ],
        task_id="yui",
    )

    assert "| unknown | gemini/gemini-3.6-flash | 1 | 70 | 0 | 2 | 0 |" in text


def test_render_task_report_does_not_count_search_ledger_as_llm_session() -> None:
    text = render_task_report(
        [
            {
                "kind": "search_loop_round",
                "payload": {
                    "round": 1,
                    "executed": [{"provider": "exa", "query": "example"}],
                },
            },
            {
                "kind": "search_loop_round",
                "payload": {
                    "round": 1,
                    "attempt": 0,
                    "response_content": "<evidence_pack>ok</evidence_pack>",
                    "usage": {
                        "total_input_tokens": 120,
                        "thinking_tokens": 10,
                        "output_tokens": 20,
                        "total_output_tokens": 30,
                    },
                },
            },
        ],
        task_id="search-ledger",
    )

    assert "web_search: 1" in text
    assert "llm_search_loop: 1" in text
    assert text.count("| research-search-loop-round1-attempt0 |") == 1
    assert "| **task total** | 120 | 30 | 10 | 20 |" in text


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
