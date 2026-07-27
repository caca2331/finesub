from __future__ import annotations

from desktop.backend.worker.protocol import (
    WorkerEvent,
    decode_event,
    encode_event,
    parse_worker_line,
)


def test_event_round_trip_is_one_json_line() -> None:
    event = WorkerEvent.progress(
        "task-1",
        stage="aligned",
        message="语音识别",
    )

    encoded = encode_event(event)

    assert encoded.endswith("\n")
    assert "\n" not in encoded[:-1]
    assert decode_event(encoded) == event


def test_non_protocol_worker_output_becomes_log_event() -> None:
    event = parse_worker_line("Loading model...\n", task_id="task-1")

    assert event.type == "log"
    assert event.task_id == "task-1"
    assert event.payload["message"] == "Loading model..."


def test_protocol_event_uses_frontend_camel_case_keys() -> None:
    encoded = encode_event(WorkerEvent.started("task-1"))

    assert '"taskId":"task-1"' in encoded
    assert "task_id" not in encoded
