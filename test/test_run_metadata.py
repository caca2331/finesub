from __future__ import annotations

import json

from finesub.run_metadata import (
    record_scratch_file,
    scratch_files,
    summarize_llm_rounds,
    update_run_metadata,
)
from finesub_bootstrap.artifacts import cleanup_intermediate, recorded_scratch_files


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


def test_a_scratch_file_is_recorded_once_and_read_back(tmp_path) -> None:
    path = tmp_path / "clip-metadata.json"
    decoded = tmp_path / "some-source-decoded.flac"
    decoded.write_bytes(b"audio")

    record_scratch_file(path, decoded)
    record_scratch_file(path, decoded)
    update_run_metadata(path, {"timing": {"total_sec": 1.0}})

    assert scratch_files(path) == [str(decoded.resolve())]


def test_cleanup_reaches_a_decoded_copy_it_could_never_have_derived(
    tmp_path,
) -> None:
    # The failing case this exists for: `-o` renames the run, so the decoded
    # copy carries the *source* media's stem and no amount of deriving from the
    # delivered subtitle would find it.
    delivered = tmp_path / "my-subtitles.srt"
    delivered.write_text("1\n", encoding="utf-8")
    decoded = tmp_path / "BV1xx-audio-source-decoded.flac"
    decoded.write_bytes(b"audio")
    metadata = tmp_path / "my-subtitles-metadata.json"
    record_scratch_file(metadata, decoded)

    assert recorded_scratch_files(delivered) == (decoded.resolve(),)

    cleanup_intermediate(delivered, preserve=[delivered])

    assert not decoded.exists()
    assert not metadata.exists()
    assert delivered.exists()


def test_a_recorded_path_outside_the_run_directory_is_not_deleted(
    tmp_path,
) -> None:
    # The record is ours, but it is JSON that outlives the run and what reads
    # it deletes what it names. A sidecar edited by hand -- or half-written --
    # must not reach anything the run did not write.
    delivered = tmp_path / "run" / "clip.srt"
    delivered.parent.mkdir()
    delivered.write_text("1\n", encoding="utf-8")
    elsewhere = tmp_path / "precious.flac"
    elsewhere.write_bytes(b"not ours")
    record_scratch_file(tmp_path / "run" / "clip-metadata.json", elsewhere)

    assert recorded_scratch_files(delivered) == ()

    cleanup_intermediate(delivered, preserve=[delivered])

    assert elsewhere.exists()


def test_cleanup_without_a_sidecar_still_removes_what_it_can_name(
    tmp_path,
) -> None:
    delivered = tmp_path / "clip.srt"
    delivered.write_text("1\n", encoding="utf-8")
    same_stem = tmp_path / "clip-decoded.flac"
    same_stem.write_bytes(b"audio")

    cleanup_intermediate(delivered, preserve=[delivered])

    assert not same_stem.exists()


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


def test_summarize_llm_rounds_accepts_local_execution_attempts(tmp_path) -> None:
    artifact_dir = tmp_path / "input.llm-artifacts"
    artifact_dir.mkdir()
    record = {
        "kind": "research_round1_response",
        "created_at": "2026-01-01T00:00:03+00:00",
        "payload": {
            "validation_ok": True,
            "execution_attempts": [
                {
                    "backend": "local_agent",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "returned_at": "2026-01-01T00:00:02+00:00",
                    "duration_ms": 2000,
                    "return_code": 0,
                }
            ],
        },
    }
    (artifact_dir / "task-artifacts.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    row = summarize_llm_rounds(artifact_dir)[0]
    assert row["api_attempts"] == 1
    assert row["api_sec"] == 2.0
    assert row["status"] == "completed"
