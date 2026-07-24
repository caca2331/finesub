from __future__ import annotations

import json

from llm.session_checkpoint import (
    SESSION_CHECKPOINT_FILENAME,
    SessionCheckpointStore,
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
