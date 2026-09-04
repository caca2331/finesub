"""Read-only ``kb_*`` tools on the harness MCP server (plan §4.3, step 4a).

Exposure needs FINESUB_MCP_KNOWLEDGE_ROOT; admission needs the task manifest's
``metadata.knowledge_identity`` ("rev:N"), which is also the revision every
reply is pinned to.
"""

from __future__ import annotations

import json

import pytest

from finesub.llm.agent.agent_mcp_server import KB_TOOL_NAMES, HarnessToolServer
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _seed(root) -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "entry_type": "游戏",
             "native_names": ["原神"], "section_order": ["档案", "角色"]},
        )
        txn.create_node(
            "T", "term", {"surface": "スカラマシュ", "zh": "散兵", "alias_text": "", "reading": "", "desc": "角色"}
        )
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I1", "S", "aliases", "Genshin")
        txn.create_item("I2", "T", "misheard", "残表", fuzzy_enabled=True)
    return repo


def _server(
    monkeypatch, root, identity: str | None = "rev:1", entitlement: str | None = "read"
) -> HarnessToolServer:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(root))
    server = HarnessToolServer(
        object(),
        assignment_id="a",
        worker_id="w",
        session_id="s",
        instance_id="i",
        tools=list(KB_TOOL_NAMES),
    )
    server._task = {"task_id": "t"}
    metadata: dict = {}
    if identity:
        metadata["knowledge_identity"] = identity
    if entitlement:
        metadata["kb_tools"] = entitlement
    server._manifest = {"metadata": metadata}
    return server


def _payload(server: HarnessToolServer, rpc_id: int, name: str, **arguments):  # type: ignore[no-untyped-def]
    reply = server._call(name, arguments, rpc_id=rpc_id)
    assert reply["isError"] is False, reply["content"][0]["text"]
    return json.loads(reply["content"][0]["text"])


def _error(server: HarnessToolServer, rpc_id: int, name: str, **arguments) -> str:  # type: ignore[no-untyped-def]
    reply = server._call(name, arguments, rpc_id=rpc_id)
    assert reply["isError"] is True
    return reply["content"][0]["text"]


def test_kb_tools_exposure_follows_the_spawner(tmp_path, monkeypatch) -> None:
    """Explicit tools list exposes them; without a list, only an env root does.
    Admission is per task either way (`_kb_context`)."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path)
    assert {t["name"] for t in server.tool_definitions} >= set(KB_TOOL_NAMES)
    monkeypatch.delenv("FINESUB_MCP_KNOWLEDGE_ROOT")
    ungranted = HarnessToolServer(
        object(), assignment_id="a", worker_id="w", session_id="s", instance_id="i",
        tools=["next_task"],  # spawner did not authorize the kb set
    )
    assert not ({t["name"] for t in ungranted.tool_definitions} & set(KB_TOOL_NAMES))


def test_kb_admission_requires_an_entitlement(tmp_path, monkeypatch) -> None:
    """A pin/identity alone grants nothing (§4.3 matrix): a task without
    kb_tools — a search judge — is refused."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, entitlement=None)
    assert "no kb entitlement" in _error(server, 1, "kb_index")


def test_kb_index_read_search_and_node_round_trip(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path)

    index = _payload(server, 1, "kb_index")
    assert index["knowledge_read_rev"] == 1 and "原神" in index["text"] and index["result_digest"]

    search = _payload(server, 2, "kb_search", query="この残表って何")
    row = next(r for r in search["matches"] if r["kind"] == "misheard")
    assert row["entry"] == "原神" and row["value"] == "残表" and row["handle"].startswith("@k")

    node = _payload(server, 3, "kb_read_node", handle=row["handle"])
    assert node["kind"] == "term" and "スカラマシュ" in node["text"]

    read = _payload(server, 4, "kb_read", entry="Genshin")  # alias resolves
    assert read["entry"] == "原神" and "# 原神" in read["text"] and "@k" in read["text"]
    narrowed = _payload(server, 5, "kb_read", entry="原神", sections=["档案"])
    assert "## 角色" not in narrowed["text"] and "## 元数据" in narrowed["text"]


def test_kb_reads_are_pinned_to_the_manifest_identity(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    with repo.store.begin("harness") as txn:
        txn.create_item("I3", "S", "aliases", "GI")
    pinned = _server(monkeypatch, tmp_path, identity="rev:1")
    assert "GI" not in _payload(pinned, 1, "kb_index")["text"]
    live = _server(monkeypatch, tmp_path, identity="rev:2")
    assert "GI" in _payload(live, 1, "kb_index")["text"]


def test_kb_admission_requires_the_task_identity(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, identity=None)
    assert "no knowledge identity" in _error(server, 1, "kb_index")


class _PullRuntime:
    """Just enough runtime for the kb_index booking path."""

    def __init__(self) -> None:
        self.pulled: list[dict] = []

    def record_pull(self, **kwargs):  # type: ignore[no-untyped-def]
        self.pulled.append(kwargs)
        return {"owed_blocks": []}


def test_kb_index_books_the_required_block_only_on_digest_match(tmp_path, monkeypatch) -> None:
    """4b fail-closed gate: booked when the reply's digest equals the
    manifest's declared digest — 'called it once' does not pass."""

    import hashlib

    from finesub.llm.knowledge.base import kb_index_block_text

    _seed(tmp_path)
    runtime = _PullRuntime()
    monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(tmp_path))
    server = HarnessToolServer(
        runtime, assignment_id="a", worker_id="w", session_id="s", instance_id="i",
        tools=list(KB_TOOL_NAMES),
    )
    digest = hashlib.sha256(kb_index_block_text(tmp_path, 1).encode("utf-8")).hexdigest()
    server._task = {"task_id": "t", "lease_generation": 1}
    server._manifest = {
        "metadata": {"knowledge_identity": "rev:1", "kb_tools": "read"},
        "required_blocks": [{"kind": "kb_index", "ref": "kb_index", "digest": digest}],
    }
    payload = _payload(server, 1, "kb_index")
    assert payload["owed_blocks"] == []
    assert len(runtime.pulled) == 1 and runtime.pulled[0]["blocks"][0]["kind"] == "kb_index"

    # a wrong declared digest is never booked
    runtime.pulled.clear()
    server._replies.clear()
    server._manifest["required_blocks"][0]["digest"] = "0" * 64
    payload = _payload(server, 2, "kb_index")
    assert "owed_blocks" not in payload and runtime.pulled == []

    # an oversized reply errors BEFORE booking: the gate must not open on an
    # index the model never saw
    import finesub.llm.agent.agent_mcp_server as mcp

    runtime.pulled.clear()
    server._replies.clear()
    server._manifest["required_blocks"][0]["digest"] = digest
    monkeypatch.setattr(mcp, "KB_REPLY_MAX_CHARS", 40)
    assert "narrow the request" in _error(server, 3, "kb_index")
    assert runtime.pulled == []


def test_read_context_refuses_the_kb_index_ref(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path)
    server._task = {"task_id": "t", "lease_generation": 1}
    server._manifest = {
        "metadata": {"knowledge_identity": "rev:1", "kb_tools": "read"},
        "required_blocks": [{"kind": "kb_index", "ref": "kb_index", "digest": "d"}],
    }
    assert "kb_index tool" in _error(server, 1, "read_context", ref="kb_index")


def test_agent_knowledge_binding_needs_an_explicit_grant(tmp_path) -> None:
    """client-side 4b wiring: the caller's kb_tools grant is the authorization;
    the pin only supplies the default root/rev."""

    import hashlib

    from finesub.llm.client import _agent_knowledge_binding
    from finesub.llm.knowledge.base import kb_index_block_text, pinned_generation_rev

    _seed(tmp_path)
    with pinned_generation_rev(tmp_path) as rev:
        # a pin alone grants nothing (a search judge stays out of the gate)
        assert _agent_knowledge_binding({}) is None
        binding = _agent_knowledge_binding({"kb_tools": "read"})
        assert binding is not None and binding["identity"] == f"rev:{rev}"
        assert binding["entitlement"] == "read"
        expected = hashlib.sha256(kb_index_block_text(tmp_path, rev).encode("utf-8")).hexdigest()
        assert binding["block"] == {"kind": "kb_index", "ref": "kb_index", "digest": expected}
        override = _agent_knowledge_binding({"kb_tools": "read", "knowledge_identity": "rev:1"})
        assert override is not None and override["identity"] == "rev:1"
    assert _agent_knowledge_binding({"kb_tools": "read"}) is None  # no pin, no root named


def test_agent_knowledge_binding_standalone_root_needs_no_pin(tmp_path) -> None:
    """reference_ingest / standalone updates name root + identity explicitly
    and get the full binding without any generation pin (review 2026-08-26)."""

    from finesub.llm.client import _agent_knowledge_binding

    _seed(tmp_path)
    binding = _agent_knowledge_binding(
        {"kb_tools": "propose", "knowledge_root": str(tmp_path), "knowledge_identity": "rev:1"}
    )
    assert binding is not None
    assert binding["entitlement"] == "propose" and binding["identity"] == "rev:1"
    assert binding["block"]["kind"] == "kb_index"
    # an explicit root without its revision is refused, not guessed
    assert (
        _agent_knowledge_binding({"kb_tools": "propose", "knowledge_root": str(tmp_path)}) is None
    )


def test_kb_validate_prechecks_proposals_with_seeded_handles(tmp_path, monkeypatch) -> None:
    """4c: the prompt's handle table rides the manifest; kb_validate runs the
    shared translate+preview without writing anything."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, entitlement="propose")
    server._manifest["metadata"]["kb_handle_bindings"] = [
        {"handle": "@k9", "kind": "node", "id": "T", "expected_valid_from_rev": 1}
    ]
    ok = _payload(
        server, 1, "kb_validate",
        proposals='{"op":"update","id":"@k9","line":"スカラマシュ|散兵|||新描述","reason":"r"}',
    )
    assert ok["ops_translated"] >= 1 and ok["ops_appliable"] == ok["ops_translated"]
    assert ok["problems"] == [] and ok["skipped"] == [] and ok["rejected"] == []
    repo = KnowledgeRepo.open(tmp_path)
    assert repo.rev == 1  # nothing was written

    bad = _payload(server, 2, "kb_validate", proposals='{"op":"update","id":"@k404","line":"x|y|||z"}')
    assert bad["ops_translated"] == 0 and bad["skipped"]


def test_kb_validate_surfaces_fold_rejections(tmp_path, monkeypatch) -> None:
    """An op the apply would skip (NFKC-duplicate alias) must not pre-check as
    a clean pass (review 2026-08-26)."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, entitlement="propose")
    server._manifest["metadata"]["kb_handle_bindings"] = [
        {"handle": "@k1", "kind": "node", "id": "S", "expected_valid_from_rev": 1}
    ]
    result = _payload(
        server, 1, "kb_validate",
        proposals='{"op":"add_item","id":"@k1","field":"aliases","value":"Ｇｅｎｓｈｉｎ","reason":"r"}',
    )
    assert result["ops_translated"] == 1 and result["ops_appliable"] == 0
    assert result["rejected"] and "already present" in result["rejected"][0]["reason"]


def test_kb_validate_is_knowledge_update_only(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, entitlement="read")
    assert "not admitted" in _error(server, 1, "kb_validate", proposals='{"op":"noop"}')


def test_kb_search_never_silently_truncates(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    with repo.store.begin("harness") as txn:
        for index in range(60):
            # same value, sixty owners-of-record: sixty candidate rows
            txn.create_item(f"A{index}", "S", "misheard", "多重误听样本")
    server = _server(monkeypatch, tmp_path, identity="rev:2")
    hits = _payload(server, 1, "kb_search", query="这段多重误听样本要全部返回")
    assert len(hits["matches"]) == 60  # none hidden behind a silent cap


def test_kb_extras_ride_metadata_never_driver_kwargs(tmp_path) -> None:
    """driver.run has an explicit signature: the kb extras must be stripped
    from run_kwargs and appear only in the manifest metadata."""

    from finesub.llm.client import _agent_task_inputs
    from finesub.llm.routing.config import CapabilityTier

    _seed(tmp_path)
    inputs = _agent_task_inputs(
        [{"role": "system", "content": "p"}, {"role": "user", "content": "u"}],
        validator_spec=None,
        variant="",
        capability_tier=CapabilityTier.CAPABLE,
        max_repair_attempts=0,
        call_kwargs={
            "task": "knowledge",
            "kb_tools": "propose",
            "knowledge_root": str(tmp_path),
            "knowledge_identity": "rev:1",
            "kb_signal_task": "run-7",
            "kb_signal_window": "w3",
            "kb_handle_bindings": [
                {"handle": "@k1", "kind": "node", "id": "T", "expected_valid_from_rev": 1}
            ],
        },
        retrieval="local",
    )
    # The switch is an argument of its own for the same reason the kb extras
    # are stripped: it decides the task's mode, and the driver's `run` would
    # reject it.
    assert inputs["retrieval_mode"] == "local"
    assert "retrieval" not in inputs["run_kwargs"]
    for key in (
        "knowledge_identity",
        "knowledge_root",
        "kb_tools",
        "kb_handle_bindings",
        "kb_signal_task",
        "kb_signal_window",
    ):
        assert key not in inputs["run_kwargs"]
    assert inputs["metadata"]["knowledge_identity"] == "rev:1"
    assert inputs["metadata"]["kb_tools"] == "propose"
    assert inputs["metadata"]["kb_root"]
    assert inputs["metadata"]["kb_signal_task"] == "run-7"
    assert inputs["metadata"]["kb_signal_window"] == "w3"
    assert inputs["metadata"]["kb_handle_bindings"][0]["handle"] == "@k1"
    assert inputs["knowledge"] is not None and inputs["knowledge"]["entitlement"] == "propose"


def test_kb_handles_reseed_per_task(tmp_path, monkeypatch) -> None:
    """A pseudo-conversational session serves many tasks; each task's prompt
    handle table must replace the previous one, not silently mix."""

    repo = _seed(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "T2", "term", {"surface": "ボブ", "zh": "鲍勃", "alias_text": "", "reading": "", "desc": "x"}
        )
        txn.create_membership("M2", "S", "T2", "角色", 1)
    server = _server(monkeypatch, tmp_path, identity="rev:2")
    server._manifest["metadata"]["kb_handle_bindings"] = [
        {"handle": "@k1", "kind": "node", "id": "T", "expected_valid_from_rev": 1}
    ]
    node = _payload(server, 1, "kb_read_node", handle="@k1")
    assert "スカラマシュ" in node["text"]
    # next task: same handle, different node
    server._task = {"task_id": "t2", "lease_generation": 1}
    server._manifest = {
        "metadata": {
            "knowledge_identity": "rev:2",
            "kb_tools": "read",
            "kb_handle_bindings": [
                {"handle": "@k1", "kind": "node", "id": "T2", "expected_valid_from_rev": 2}
            ],
        }
    }
    node2 = _payload(server, 2, "kb_read_node", handle="@k1")
    assert "ボブ" in node2["text"]


def test_kb_reply_cap_asks_for_a_narrower_request(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    import finesub.llm.agent.agent_mcp_server as mcp

    monkeypatch.setattr(mcp, "KB_REPLY_MAX_CHARS", 40)
    server = _server(monkeypatch, tmp_path)
    message = _error(server, 1, "kb_read", entry="原神")
    assert "narrow the request" in message


def test_kb_reads_book_exposed_events(tmp_path, monkeypatch) -> None:
    """Agent-side ``exposed`` (plan §4.2 item 5): what a kb tool actually
    returned is booked, idempotently, under the caller's signal identity —
    the runtime's constant task_id ("call") must never collapse distinct
    windows into one row (review 2026-08-27). An errored (capped) reply books
    nothing."""

    repo = _seed(tmp_path)
    server = _server(monkeypatch, tmp_path)
    server._manifest["metadata"].update(
        {"kb_signal_task": "run-1", "kb_signal_window": "w1"}
    )
    _payload(server, 1, "kb_read", entry="原神")
    rows = repo.store.conn.execute(
        "SELECT node_id, opportunity, task_id, window_id FROM events WHERE kind='exposed'"
        " ORDER BY node_id"
    ).fetchall()
    assert [(r["node_id"], r["opportunity"], r["task_id"], r["window_id"]) for r in rows] == [
        ("S", "context", "run-1", "w1"),
        ("T", "context", "run-1", "w1"),
    ]
    _payload(server, 2, "kb_read", entry="原神")  # replay deduplicates
    search = _payload(server, 3, "kb_search", query="この残表って何")
    handle = next(r["handle"] for r in search["matches"] if r["kind"] == "misheard")
    _payload(server, 4, "kb_read_node", handle=handle)
    count = repo.store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind='exposed'"
    ).fetchone()["n"]
    assert count == 2  # the single-node read resolves to T, already booked

    # a second window is a second fact, not a dedupe hit
    server._manifest["metadata"]["kb_signal_window"] = "w2"
    _payload(server, 5, "kb_read", entry="原神")
    assert repo.store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind='exposed'"
    ).fetchone()["n"] == 4

    # without signal identity the assignment id keeps calls distinct
    bare = _server(monkeypatch, tmp_path)
    _payload(bare, 6, "kb_read", entry="原神")
    fallback = repo.store.conn.execute(
        "SELECT DISTINCT task_id FROM events WHERE kind='exposed' AND task_id LIKE 'a:%'"
    ).fetchall()
    assert [r["task_id"] for r in fallback] == ["a:t"]

    # an errored (capped) reply books nothing
    import finesub.llm.agent.agent_mcp_server as mcp

    monkeypatch.setattr(mcp, "KB_REPLY_MAX_CHARS", 40)
    capped = _server(monkeypatch, tmp_path)
    capped._manifest["metadata"]["kb_signal_task"] = "capped-run"
    _error(capped, 7, "kb_read", entry="原神")
    assert repo.store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE task_id='capped-run'"
    ).fetchone()["n"] == 0


def test_kb_read_preview_tier_follows_the_task_entitlement(tmp_path, monkeypatch) -> None:
    """Review 2026-08-29 P2: `kb_read` rendered the FULL preview for every
    task. The scaffolding (empty sections, collection discipline, core empty
    slots) is there to tell a WRITER what belongs where, so it follows the
    `kb_tools` entitlement like every other model-facing surface — otherwise a
    read-only correction agent pays budget for slots it cannot fill and gets
    invited to report gaps nobody asked about."""

    _seed(tmp_path)

    reader = _payload(_server(monkeypatch, tmp_path, entitlement="read"),
                      1, "kb_read", entry="原神")["text"]
    writer = _payload(_server(monkeypatch, tmp_path, entitlement="propose"),
                      1, "kb_read", entry="原神")["text"]

    # prompt mode has no markdown bullet: the core slot renders as a bare line
    assert "\n[别名]" in writer and "用途：" in writer      # core slots + discipline
    assert "\n[别名]" not in reader and "用途：" not in reader
    assert "スカラマシュ" in reader and "スカラマシュ" in writer   # content, both tiers
