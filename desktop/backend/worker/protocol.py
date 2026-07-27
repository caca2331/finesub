from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "started",
    "stage",
    "log",
    "completed",
    "failed",
    "cancelled",
]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EventType
    task_id: str
    timestamp: str = Field(default_factory=_utc_timestamp)
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def started(cls, task_id: str) -> "WorkerEvent":
        return cls(type="started", task_id=task_id)

    @classmethod
    def progress(
        cls,
        task_id: str,
        *,
        stage: str,
        message: str,
    ) -> "WorkerEvent":
        return cls(
            type="stage",
            task_id=task_id,
            payload={"stage": stage, "message": message},
        )

    @classmethod
    def log(cls, task_id: str, message: str) -> "WorkerEvent":
        return cls(type="log", task_id=task_id, payload={"message": message})

    @classmethod
    def completed(
        cls,
        task_id: str,
        outputs: dict[str, str],
    ) -> "WorkerEvent":
        return cls(
            type="completed",
            task_id=task_id,
            payload={"outputs": outputs},
        )

    @classmethod
    def failed(cls, task_id: str, message: str) -> "WorkerEvent":
        return cls(type="failed", task_id=task_id, payload={"message": message})

    @classmethod
    def cancelled(cls, task_id: str) -> "WorkerEvent":
        return cls(type="cancelled", task_id=task_id)


def encode_event(event: WorkerEvent) -> str:
    body = {
        "type": event.type,
        "taskId": event.task_id,
        "timestamp": event.timestamp,
        "payload": event.payload,
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def decode_event(line: str) -> WorkerEvent:
    body = json.loads(line)
    if not isinstance(body, dict):
        raise ValueError("worker event must be a JSON object")
    return WorkerEvent.model_validate(
        {
            "type": body["type"],
            "task_id": body["taskId"],
            "timestamp": body["timestamp"],
            "payload": body.get("payload", {}),
        }
    )


def parse_worker_line(line: str, *, task_id: str) -> WorkerEvent:
    stripped = line.rstrip("\r\n")
    try:
        event = decode_event(stripped)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return WorkerEvent.log(task_id, stripped)
    if event.task_id != task_id:
        return WorkerEvent.log(task_id, stripped)
    return event


class EventLogWriter(TextIO):
    """Convert arbitrary pipeline stdout writes into protocol log events."""

    def __init__(self, task_id: str, emit) -> None:
        self.task_id = task_id
        self.emit = emit
        self._buffer = ""

    def write(self, value: str) -> int:
        self._buffer += value
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.rstrip("\r"):
                self.emit(WorkerEvent.log(self.task_id, line.rstrip("\r")))
        return len(value)

    def flush(self) -> None:
        if self._buffer:
            self.emit(WorkerEvent.log(self.task_id, self._buffer))
            self._buffer = ""

    @property
    def encoding(self) -> str:
        return "utf-8"
