from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "started",
    "stage",
    "log",
    "debug",
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
        reused: bool = False,
    ) -> "WorkerEvent":
        """Report the stage the run has entered.

        `reused` says the stage produced nothing this run because its output
        was already on disk -- the progress list shows those differently, so
        that a rerun does not claim to have redone work it skipped.
        """

        return cls(
            type="stage",
            task_id=task_id,
            payload={"stage": stage, "message": message, "reused": reused},
        )

    @classmethod
    def log(cls, task_id: str, message: str) -> "WorkerEvent":
        return cls(type="log", task_id=task_id, payload={"message": message})

    @classmethod
    def debug(cls, task_id: str, message: str) -> "WorkerEvent":
        """Diagnosis for the file, not for the window.

        Same destination on disk as `log`, but it never enters the bounded
        event deque the drawer renders: per-group recovery detail would push
        everything else out of a UI that only holds a few hundred rows, while
        being exactly what someone reads afterwards when a run went wrong.
        """

        return cls(type="debug", task_id=task_id, payload={"message": message})

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
    """Convert arbitrary pipeline stdout writes into protocol log events.

    Carriage returns are rendered, not kept. A progress bar rewrites one line
    hundreds of times with `\\r` and ends it with a single `\\n`; splitting on
    `\\n` alone meant every one of those redraws accumulated in the pending
    line, so one block of separation produced an unbounded buffer and finally a
    multi-megabyte log line. Only the text after the last `\\r` can still be on
    screen, so only that is what the line *is*.

    The pending line is capped as well, for the producer that emits neither
    separator: past the cap the line is emitted truncated and the rest is
    dropped until the next line break, which bounds memory without hiding that
    something was cut.
    """

    #: Generous for a log line, small enough that a runaway writer cannot grow
    #: the process.
    MAX_LINE_CHARS = 8192

    def __init__(self, task_id: str, emit) -> None:
        self.task_id = task_id
        self.emit = emit
        self._buffer = ""
        self._dropping = False

    def write(self, value: str) -> int:
        remaining = value
        while True:
            # Split on "\n" only. `str.splitlines` would also break on "\r",
            # "\x85" and " ", turning a line that merely contains one into
            # several -- and it would hand back "abc\r\n" as one piece whose
            # carriage return is not the progress kind.
            head, newline, tail = remaining.partition("\n")
            self._consume(head, ends_line=bool(newline))
            if not newline:
                break
            remaining = tail
        return len(value)

    def _consume(self, body: str, *, ends_line: bool) -> None:
        # A trailing "\r" is the CRLF half of this very line break, not a
        # redraw: dropping it here keeps CRLF output from losing its content.
        if ends_line and body.endswith("\r"):
            body = body[:-1]
        # Whatever preceded the last carriage return has been overwritten on
        # screen; it never was a line of its own.
        if "\r" in body:
            body = body.rsplit("\r", 1)[1]
            self._buffer = ""
            self._dropping = False
        if self._dropping:
            if ends_line:
                self._dropping = False
            return
        self._buffer += body
        if ends_line:
            self._emit_buffer()
            return
        if len(self._buffer) > self.MAX_LINE_CHARS:
            self._buffer = self._buffer[: self.MAX_LINE_CHARS] + " …(截断)"
            self._emit_buffer()
            self._dropping = True

    def _emit_buffer(self) -> None:
        line = self._buffer
        self._buffer = ""
        if line:
            self.emit(WorkerEvent.log(self.task_id, line))

    def flush(self) -> None:
        self._emit_buffer()

    @property
    def encoding(self) -> str:
        return "utf-8"
