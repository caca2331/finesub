from __future__ import annotations

import json

from asr_playground.run_metadata import summarize_llm_rounds, update_run_metadata


def test_update_run_metadata_merges_nested_sections(tmp_path) -> None:
    path = tmp_path / "input-metadata.json"
    update_run_metadata(
        path,
        {"timing": {"stages": {"asr": {"status": "executed", "elapsed_sec": 2.0}}}},
    )
    update_run_metadata(
        path,
        {"timing": {"total_sec": 3.0}, "workers": {"asr": {"effective": 1}}},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["timing"]["stages"]["asr"]["elapsed_sec"] == 2.0
    assert data["timing"]["total_sec"] == 3.0
    assert data["workers"]["asr"]["effective"] == 1


def test_update_run_metadata_replaces_complete_stage_record(tmp_path) -> None:
    path = tmp_path / "input-metadata.json"
    update_run_metadata(
        path,
        {"timing": {"stages": {"asr": {"status": "executed", "elapsed_sec": 2.0}}}},
    )
    update_run_metadata(
        path,
        {"timing": {"stages": {"asr": {"status": "reused"}}}},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["timing"]["stages"]["asr"] == {"status": "reused"}


def test_summarize_llm_rounds_groups_failed_and_successful_attempts(tmp_path) -> None:
    artifact_dir = tmp_path / "input.llm-artifacts"
    artifact_dir.mkdir()
    records = [
        {
            "kind": "correction_window_call_error",
            "created_at": "2026-01-01T00:00:02+00:00",
            "payload": {
                "chunk_id": "0001",
                "api_attempts": [
                    {
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "returned_at": "2026-01-01T00:00:01+00:00",
                        "elapsed_sec": 1.0,
                        "return_code": "429",
                    }
                ],
            },
        },
        {
            "kind": "correction_window_response",
            "created_at": "2026-01-01T00:00:05+00:00",
            "payload": {
                "chunk_id": "0001",
                "validation_ok": True,
                "output_limited": False,
                "api_attempts": [
                    {
                        "started_at": "2026-01-01T00:00:03+00:00",
                        "returned_at": "2026-01-01T00:00:04+00:00",
                        "elapsed_sec": 1.0,
                        "return_code": "200",
                    }
                ],
            },
        },
    ]
    path = artifact_dir / "task-artifacts.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert summarize_llm_rounds(artifact_dir) == [
        {
            "round": "correction-0001-answer",
            "elapsed_sec": 5.0,
            "api_sec": 2.0,
            "api_attempts": 2,
            "retries": 1,
            "status": "completed",
        }
    ]
