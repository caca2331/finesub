"""Durable checkpoints for validated, replayable LLM session responses.

The production harness rebuilds deterministic/local state on restart and may
redo searches, extracts, media clipping, or uploads.  Immediately before an
LLM call it hashes the exact assembled messages plus any non-message identity
(for example source-media metadata), then reuses a previously validated raw
response when that hash matches.

This deliberately stores responses rather than parsed Python objects: current
production parsers remain the single source of truth. Exact input identity is
only the recovery boundary for this uncommitted call; it never invalidates an
already committed outer research/window/stage result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
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
    execution_identity_override: Mapping[str, Any] | None = None,
) -> str:
    """Fingerprint the exact session input and non-message call identity."""

    from .routing.execution_policy import execution_identity

    payload = {
        "prompt_version": prompt_version,
        "messages": list(messages),
        "call_config": dict(call_config or {}),
        "extra_identity": dict(extra_identity or {}),
        "execution_identity": dict(
            execution_identity_override or execution_identity()
        ),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def agent_conversation_identity(
    *,
    session_scope: str,
    logical_context_digest: str,
    conversation_epoch: int = 0,
    protocol_digest: str = "",
    context_digest: str = "",
    knowledge_digest: str = "",
    conversation_handle: str = "",
    parent_turn_identity: str = "",
    turn_generation: int | None = None,
    harness_ack_digest: str = "",
) -> dict[str, Any]:
    """Identity contribution for task-scoped or long-lived Agent sessions.

    Task scope is the full-replay baseline: the complete logical context digest
    is sufficient and no provider history is trusted. Assignment scope has
    hidden provider history, so it fails closed unless the driver supplies a
    stable handle, epoch, parent-turn lineage and the last harness ack.
    """

    if session_scope == "task":
        if not logical_context_digest:
            raise ValueError("task scope requires logical_context_digest")
        return {
            "session_scope": "task",
            "logical_context_digest": logical_context_digest,
        }
    if session_scope != "assignment":
        raise ValueError("session_scope must be 'task' or 'assignment'")
    if conversation_epoch < 1:
        raise ValueError("assignment scope requires a positive conversation_epoch")
    required = {
        "logical_context_digest": logical_context_digest,
        "protocol_digest": protocol_digest,
        "context_digest": context_digest,
        "knowledge_digest": knowledge_digest,
        "conversation_handle": conversation_handle,
        "harness_ack_digest": harness_ack_digest,
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        raise ValueError(
            "assignment scope conversation identity is missing: " + ", ".join(missing)
        )
    if not parent_turn_identity and turn_generation is None:
        raise ValueError(
            "assignment scope requires parent_turn_identity or turn_generation"
        )
    if turn_generation is not None and turn_generation < 0:
        raise ValueError("turn_generation must be non-negative")
    return {
        "session_scope": "assignment",
        "logical_context_digest": logical_context_digest,
        "conversation_epoch": conversation_epoch,
        "protocol_digest": protocol_digest,
        "context_digest": context_digest,
        "knowledge_digest": knowledge_digest,
        "conversation_handle": conversation_handle,
        "parent_turn_identity": parent_turn_identity,
        "turn_generation": turn_generation,
        "harness_ack_digest": harness_ack_digest,
    }


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
        # Append + in-memory-map update must be atomic per store: the JSONL
        # semantics (append-only, last wins) are concurrency-safe, the bare
        # write and dict update were not (docs/llm_harness_behavior.md).
        self._lock = threading.Lock()
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
        with self._lock:
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
        record = SessionCheckpointRecord(
            session=session,
            key=key,
            input_hash=input_hash,
            content=content,
            metadata=safe_metadata,
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._records[(session, key, input_hash)] = record
        return record
