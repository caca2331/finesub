"""Durable checkpoints for validated, replayable LLM session responses.

The production harness rebuilds deterministic/local state on restart and may
redo searches, extracts, media clipping, or uploads.  Immediately before an
LLM call it hashes the exact assembled messages plus any non-message identity
(for example source-media metadata), then reuses a previously validated raw
response when that hash matches.

This deliberately stores responses rather than parsed Python objects: current
production parsers remain the single source of truth, and ``PROMPT_VERSION``
invalidation means no checkpoint-schema migration layer is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SESSION_CHECKPOINT_FILENAME = "session-checkpoints.jsonl"
SESSION_CHECKPOINT_SCHEMA_VERSION = 1


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def session_input_hash(
    messages: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str,
    call_config: Mapping[str, Any] | None = None,
    extra_identity: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint the exact session input and non-message call identity."""

    payload = {
        "prompt_version": prompt_version,
        "messages": list(messages),
        "call_config": dict(call_config or {}),
        "extra_identity": dict(extra_identity or {}),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _content_hash(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionCheckpointRecord:
    session: str
    key: str
    input_hash: str
    content: str
    metadata: Mapping[str, Any]


class SessionCheckpointStore:
    """Append-only validated-response ledger scoped to one task artifact dir."""

    def __init__(
        self,
        task_artifact_dir: str | Path | None,
        *,
        enabled: bool = True,
    ) -> None:
        self.path = (
            Path(task_artifact_dir) / SESSION_CHECKPOINT_FILENAME
            if task_artifact_dir is not None
            else None
        )
        self.enabled = bool(enabled and self.path is not None)
        self._records: dict[tuple[str, str, str], SessionCheckpointRecord] = {}
        if self.enabled:
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") != SESSION_CHECKPOINT_SCHEMA_VERSION:
                continue
            if payload.get("status") != "committed":
                continue
            session = payload.get("session")
            key = payload.get("key")
            input_hash = payload.get("input_hash")
            content = payload.get("content")
            if not all(isinstance(value, str) and value for value in (session, key, input_hash)):
                continue
            if not isinstance(content, str):
                continue
            if payload.get("content_hash") != _content_hash(content):
                continue
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            record = SessionCheckpointRecord(
                session=session,
                key=key,
                input_hash=input_hash,
                content=content,
                metadata=metadata,
            )
            self._records[(session, key, input_hash)] = record

    def get(
        self, session: str, key: str, input_hash: str
    ) -> SessionCheckpointRecord | None:
        if not self.enabled:
            return None
        return self._records.get((session, key, input_hash))

    def commit(
        self,
        *,
        session: str,
        key: str,
        input_hash: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionCheckpointRecord | None:
        """Append one parser-validated response and make it immediately reusable."""

        if not self.enabled:
            return None
        assert self.path is not None
        safe_metadata = json.loads(_stable_json(dict(metadata or {})))
        payload = {
            "schema_version": SESSION_CHECKPOINT_SCHEMA_VERSION,
            "status": "committed",
            "session": session,
            "key": key,
            "input_hash": input_hash,
            "content_hash": _content_hash(content),
            "content": content,
            "metadata": safe_metadata,
            "committed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        record = SessionCheckpointRecord(
            session=session,
            key=key,
            input_hash=input_hash,
            content=content,
            metadata=safe_metadata,
        )
        self._records[(session, key, input_hash)] = record
        return record
