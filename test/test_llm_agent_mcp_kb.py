"""Read-only ``kb_*`` tools on the harness MCP server (plan §4.3, step 4a).

Exposure needs FINESUB_MCP_KNOWLEDGE_ROOT; admission needs the task manifest's
``metadata.knowledge_identity`` ("rev:N"), which is also the revision every
reply is pinned to.
"""

from __future__ import annotations

import json

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


class _ReadLedger:
    """Just enough runtime for `kb_read`: it books every successful reply
    (the `--style auto` read-check stands on that ledger)."""

    def __init__(self) -> None:
        self.reads: list[dict] = []

    def record_kb_read(self, **kwargs):  # type: ignore[no-untyped-def]
        self.reads.append(kwargs)


def _server(
    monkeypatch, root, identity: str | None = "rev:1", entitlement: str | None = "read",
    tools=KB_TOOL_NAMES,
) -> HarnessToolServer:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(root))
    server = HarnessToolServer(
        _ReadLedger(),
        assignment_id="a",
        worker_id="w",
        session_id="s",
        instance_id="i",
        tools=list(tools),
    )
    server._task = {"task_id": "t", "lease_generation": 1}
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
    assert "## 角色" not in narrowed["text"] and narrowed["text"].startswith("# 原神")


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

    # the knowledge cap does not apply to the index: it is a required block,
    # and refusing it would shut the submit gate for good. Its size is
    # the channel's business.
    import finesub.llm.agent.agent_mcp_server as mcp

    runtime.pulled.clear()
    server._replies.clear()
    server._manifest["required_blocks"][0]["digest"] = digest
    monkeypatch.setattr(mcp, "KB_REPLY_MAX_CHARS", 40)
    assert _payload(server, 3, "kb_index")["owed_blocks"] == []
    assert len(runtime.pulled) == 1


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


def test_the_alias_row_handle_rides_the_manifest_and_the_writer_read(tmp_path, monkeypatch) -> None:
    """Cold read Q5: the `[别名]` row's `@a` handle reaches agent sessions both
    ways -- replayed from the prompt's handle table and rendered by a
    propose-grant `kb_read` -- and `kb_validate` resolves it."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, entitlement="propose")
    server._manifest["metadata"]["kb_handle_bindings"] = [
        {"handle": "@a1", "kind": "aliases", "id": "S", "expected_valid_from_rev": 1}
    ]
    ok = _payload(server, 1, "kb_validate", proposals='{"op":"update","id":"@a1","line":"[别名] Genshin、GI","reason":"r"}')
    assert ok["ops_translated"] >= 1 and ok["ops_appliable"] == ok["ops_translated"]
    assert ok["skipped"] == [] and ok["problems"] == []
    writer = _payload(_server(monkeypatch, tmp_path, entitlement="propose"), 2, "kb_read", entry="原神")["text"]
    assert "[别名] Genshin  <!-- @a1 -->" in writer


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
    """Cold read P062: a reading task is not offered `kb_validate`; a
    pseudo-conversational session that exposes it anyway is refused per call,
    with the tools the grant does admit (not the style grant's hint)."""

    from finesub.llm.agent.agent_mcp_server import KB_ENTITLEMENT_TOOLS

    assert "kb_validate" not in KB_ENTITLEMENT_TOOLS["read"]
    assert "kb_validate" in KB_ENTITLEMENT_TOOLS["propose"]
    assert KB_ENTITLEMENT_TOOLS["style"] == ("kb_read",)
    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, entitlement="read")
    error = _error(server, 1, "kb_validate", proposals='{"op":"noop"}')
    assert "does not admit kb_validate" in error and "it admits kb_" in error
    assert "style" not in error


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
    # The model was refused the entry, so the error names what
    # `sections=[...]` takes, with each section's size.
    assert "sections=[...]" in message and "档案 (" in message and "角色 (" in message
    assert len(message) < 11_000
    assert "narrow the request" in _error(server, 2, "kb_search", query="原")


def test_a_kb_read_within_the_cap_builds_no_sections_hint(tmp_path, monkeypatch) -> None:
    """The hint renders every section (and hands out their handles): only a
    refused reply pays for it."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path)

    def refuse(*_args):
        raise AssertionError("hint built for a reply that fits")

    monkeypatch.setattr(server, "_kb_sections_hint", refuse)
    assert "# 原神" in _payload(server, 1, "kb_read", entry="原神")["text"]


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


# --- the narrow `style` grant and the kb_read ledger (auto-style-gaps plan A) ----


def _seed_style(root) -> None:  # type: ignore[no-untyped-def]
    from finesub.llm.knowledge.maintain import main as maintain_main

    _seed(root)
    KnowledgeRepo.forget(root)
    maintain_main(["--root", str(root), "new", "style", "甲组", "--intro", "甲组的口味"])
    maintain_main(["--root", str(root), "new", "style", "原神", "--intro", "与游戏条目同名"])
    KnowledgeRepo.forget(root)


def test_kb_read_books_each_successful_reply_into_the_task_ledger(tmp_path, monkeypatch) -> None:
    """The `--style auto` read-check stands on this ledger: the resolved entry
    and whether `sections` narrowed it, booked after the size cap passed -- a
    reply the model never received is never "read"."""

    _seed_style(tmp_path)
    server = _server(monkeypatch, tmp_path)
    rev = KnowledgeRepo.open(tmp_path).rev
    server._manifest["metadata"]["knowledge_identity"] = f"rev:{rev}"
    _payload(server, 1, "kb_read", entry="style/甲组")
    _payload(server, 2, "kb_read", entry="原神", sections=["档案"])
    assert [(r["entry"], r["category"], r["sections"]) for r in server.runtime.reads] == [
        ("甲组", "style", False),
        ("原神", "common", True),
    ]
    assert server.runtime.reads[0]["task_id"] == "t" and server.runtime.reads[0]["lease_generation"] == 1

    import finesub.llm.agent.agent_mcp_server as mcp

    monkeypatch.setattr(mcp, "KB_REPLY_MAX_CHARS", 40)
    _error(server, 3, "kb_read", entry="style/甲组")
    assert len(server.runtime.reads) == 2


def test_the_style_grant_reads_style_entries_and_nothing_else(tmp_path, monkeypatch) -> None:
    """Spawned for a style-only window, the server offers `kb_read` alone; and
    because a pseudo-conversational session exposes every tool at launch, the
    same limit is enforced per call against the task's entitlement."""

    from finesub.llm.agent.agent_mcp_server import KB_ENTITLEMENT_TOOLS, TOOL_NAMES

    _seed_style(tmp_path)
    rev = KnowledgeRepo.open(tmp_path).rev
    narrow = _server(
        monkeypatch, tmp_path, identity=f"rev:{rev}", entitlement="style",
        tools=[*TOOL_NAMES, *KB_ENTITLEMENT_TOOLS["style"]],
    )
    offered = {t["name"] for t in narrow.tool_definitions} & set(KB_TOOL_NAMES)
    assert offered == {"kb_read"}
    assert "Unknown tool" in _error(narrow, 1, "kb_index")

    wide = _server(monkeypatch, tmp_path, identity=f"rev:{rev}", entitlement="style")
    assert "does not admit kb_index" in _error(wide, 2, "kb_index")
    assert "does not admit kb_search" in _error(wide, 3, "kb_search", query="原神")
    assert "style/<名字>" in _error(wide, 4, "kb_read", entry="原神")  # bare name
    assert "only read style entries" in _error(wide, 5, "kb_read", entry="common/原神")
    read = _payload(wide, 6, "kb_read", entry="style/原神")  # same name, style namespace
    assert read["entry"] == "原神" and "与游戏条目同名" in read["text"]


def test_the_style_binding_owes_no_index_block(tmp_path) -> None:
    from finesub.llm.client import _agent_knowledge_binding

    _seed(tmp_path)
    binding = _agent_knowledge_binding(
        {"kb_tools": "style", "knowledge_root": str(tmp_path), "knowledge_identity": "rev:1"}
    )
    assert binding is not None and binding["entitlement"] == "style"
    assert binding["tools"] == ("kb_read",) and binding["block"] is None
    read = _agent_knowledge_binding(
        {"kb_tools": "read", "knowledge_root": str(tmp_path), "knowledge_identity": "rev:1"}
    )
    assert read["tools"] == tuple(t for t in KB_TOOL_NAMES if t != "kb_validate")
    assert read["block"]["kind"] == "kb_index"


def test_only_a_propose_grant_hears_of_the_alias_row_and_kb_validate(tmp_path, monkeypatch) -> None:
    """Cold read round 2, P178: a reading session's `kb_read` description must
    not point at `kb_validate` or `<knowledge_proposals>`, which it lacks."""

    def kb_read_description(tools):  # type: ignore[no-untyped-def]
        server = HarnessToolServer(
            object(), assignment_id="a", worker_id="w", session_id="s", instance_id="i", tools=tools,
        )
        return next(t["description"] for t in server.tool_definitions if t["name"] == "kb_read")

    reading = kb_read_description(["next_task", *(n for n in KB_TOOL_NAMES if n != "kb_validate")])
    proposing = kb_read_description(["next_task", *KB_TOOL_NAMES])
    assert "kb_validate" not in reading and "@a" not in reading
    assert "kb_validate" in proposing and "@a" in proposing


# --- batch: independent lookups in one round (Codex) ---------------------------


def _batch_server(monkeypatch, root):  # type: ignore[no-untyped-def]
    from finesub.llm.agent.channels import ChannelLimits

    server = _server(monkeypatch, root)
    server.channel = ChannelLimits(
        reply_ok_bytes=700_000, reply_error_bytes=700_000, overflow="tool", measure="json",
        batch_tool=True,
    )
    return server


def test_batch_runs_each_lookup_as_if_called_alone(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    server = _batch_server(monkeypatch, tmp_path)
    assert {"batch", "read_spill"} <= {tool["name"] for tool in server.tool_definitions}
    seen: list[str] = []
    dispatch = server._dispatch

    def spy(name, arguments, *, request_id):  # type: ignore[no-untyped-def]
        seen.append(request_id)
        return dispatch(name, arguments, request_id=request_id)

    monkeypatch.setattr(server, "_dispatch", spy)
    results = _payload(server, 1, "batch", calls=[
        {"tool": "kb_read", "arguments": {"entry": "原神"}},
        {"tool": "kb_search", "arguments": {"query": "この残表って何"}},
        {"tool": "submit", "arguments": {"payload": "x"}},
        {"tool": "kb_read", "arguments": {"entry": "不存在"}},
    ])["results"]
    assert [item["tool"] for item in results] == ["kb_read", "kb_search", "submit", "kb_read"]
    assert results[0]["result"]["entry"] and results[1]["result"]["matches"]
    assert "cannot run in a batch" in results[2]["error"]
    assert "unknown entry" in results[3]["error"]
    # One request id per call, none the batch's own: the replay cache and the
    # runtime's idempotency key on it.
    batch_id, *sub_ids = seen
    assert sub_ids == [f"{batch_id}-b1", f"{batch_id}-b2", f"{batch_id}-b4"]
    assert len(server.runtime.reads) == 1
    # A replayed batch is answered from the cache: nothing runs again.
    seen.clear()
    server._call("batch", {"calls": [
        {"tool": "kb_read", "arguments": {"entry": "原神"}},
        {"tool": "kb_search", "arguments": {"query": "この残表って何"}},
        {"tool": "submit", "arguments": {"payload": "x"}},
        {"tool": "kb_read", "arguments": {"entry": "不存在"}},
    ]}, rpc_id=1)
    assert seen == [] and len(server.runtime.reads) == 1


def test_batch_refuses_an_empty_call_list_and_nesting(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    server = _batch_server(monkeypatch, tmp_path)
    assert "non-empty list" in _error(server, 1, "batch", calls=[])
    nested = _payload(server, 2, "batch", calls=[{"tool": "batch", "arguments": {"calls": []}}])
    assert "cannot run in a batch" in nested["results"][0]["error"]


def test_every_codex_tool_passes_codex_auto_approval(tmp_path, monkeypatch) -> None:
    """Codex runs tools under `default_tools_approval_mode="auto"`, which lets
    a call through only when it is read-only, or neither destructive nor
    open-world (`requires_mcp_tool_approval`, Codex 0.160). The annotations
    stay honest, so the members of `batch` have to be read-only for `batch`'s
    own read-only annotation to be true."""

    from finesub.llm.agent.agent_mcp_server import BATCH_MEMBERS, TOOL_NAMES, WEB_TOOL_NAMES

    _seed(tmp_path)
    server = _batch_server(monkeypatch, tmp_path)
    server._web_tools = True
    tools = {tool["name"]: tool for tool in server.tool_definitions}
    assert set(tools) >= {*TOOL_NAMES, *WEB_TOOL_NAMES, *KB_TOOL_NAMES, "batch", "read_spill"}
    for name, tool in tools.items():
        hints = tool["annotations"]
        approved = hints["readOnlyHint"] or (not hints["destructiveHint"] and not hints["openWorldHint"])
        assert approved, name
    for name in BATCH_MEMBERS:
        assert tools[name]["annotations"]["readOnlyHint"] is True, name


def test_result_size_meta_rides_every_tool_only_when_declared(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.channels import ChannelLimits

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path)
    assert all("_meta" not in tool for tool in server.tool_definitions)
    server.channel = ChannelLimits(reply_ok_bytes=400_000, reply_error_bytes=10_000, result_size_meta=500_000)
    assert all(
        tool["_meta"] == {"anthropic/maxResultSizeChars": 500_000} for tool in server.tool_definitions
    )
