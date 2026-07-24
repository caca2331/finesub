from __future__ import annotations

from llm.task_report import render_task_report


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
