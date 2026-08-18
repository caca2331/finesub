from __future__ import annotations

import json

import pytest

from finesub.llm.session_checkpoint import (
    SESSION_CHECKPOINT_FILENAME,
    SessionCheckpointStore,
    agent_conversation_identity,
    session_input_hash,
)


def test_session_input_hash_covers_messages_config_and_extra_identity() -> None:
    messages = [{"role": "user", "content": "hello"}]
    base = session_input_hash(messages, prompt_version="v1")

    assert base == session_input_hash(messages, prompt_version="v1")
    assert base != session_input_hash(messages, prompt_version="v2")
    assert base != session_input_hash(
        messages, prompt_version="v1", call_config={"max_tokens": 10}
    )
    assert base != session_input_hash(
        messages, prompt_version="v1", extra_identity={"media": "changed"}
    )
    assert base != session_input_hash(
        messages,
        prompt_version="v1",
        execution_identity_override={"policy_id": "injected-agent-only"},
    )


def test_store_round_trips_latest_valid_record_and_ignores_corrupt_tail(tmp_path) -> None:
    store = SessionCheckpointStore(tmp_path)
    first = store.commit(
        session="query",
        key="0001",
        input_hash="sha256:input",
        content="first",
        metadata={"model": "m1"},
    )
    assert first is not None
    store.commit(
        session="query",
        key="0001",
        input_hash="sha256:input",
        content="second",
        metadata={"model": "m2"},
    )
    path = tmp_path / SESSION_CHECKPOINT_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":1,"status":"committed"')

    reloaded = SessionCheckpointStore(tmp_path)
    record = reloaded.get("query", "0001", "sha256:input")

    assert record is not None
    assert record.content == "second"
    assert record.metadata["model"] == "m2"


def test_store_rejects_tampered_content_and_can_be_disabled(tmp_path) -> None:
    store = SessionCheckpointStore(tmp_path)
    store.commit(
        session="research-r1",
        key="main",
        input_hash="sha256:input",
        content="valid",
    )
    path = tmp_path / SESSION_CHECKPOINT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    payload["content"] = "tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert SessionCheckpointStore(tmp_path).get(
        "research-r1", "main", "sha256:input"
    ) is None
    disabled = SessionCheckpointStore(tmp_path, enabled=False)
    assert disabled.get("research-r1", "main", "sha256:input") is None
    assert disabled.commit(
        session="research-r1",
        key="main",
        input_hash="sha256:input",
        content="ignored",
    ) is None


def test_agent_conversation_identity_keeps_task_baseline_minimal() -> None:
    assert agent_conversation_identity(
        session_scope="task", logical_context_digest="sha256:full"
    ) == {
        "session_scope": "task",
        "logical_context_digest": "sha256:full",
    }


def test_assignment_conversation_identity_requires_and_hashes_lineage() -> None:
    base = agent_conversation_identity(
        session_scope="assignment",
        logical_context_digest="sha256:logical",
        conversation_epoch=1,
        protocol_digest="sha256:protocol",
        context_digest="sha256:context",
        knowledge_digest="sha256:knowledge",
        conversation_handle="conversation-1",
        parent_turn_identity="turn-1",
        harness_ack_digest="sha256:ack",
    )
    changed = {**base, "conversation_epoch": 2}
    messages = [{"role": "user", "content": "task delta"}]
    assert session_input_hash(
        messages,
        prompt_version="v1",
        extra_identity={"agent_conversation": base},
        execution_identity_override={"policy": "test"},
    ) != session_input_hash(
        messages,
        prompt_version="v1",
        extra_identity={"agent_conversation": changed},
        execution_identity_override={"policy": "test"},
    )
    with pytest.raises(ValueError, match="parent_turn_identity"):
        agent_conversation_identity(
            session_scope="assignment",
            logical_context_digest="sha256:logical",
            conversation_epoch=1,
            protocol_digest="sha256:protocol",
            context_digest="sha256:context",
            knowledge_digest="sha256:knowledge",
            conversation_handle="conversation-1",
            harness_ack_digest="sha256:ack",
        )
