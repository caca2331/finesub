"""KB verification task (§11.4 / O9) and repair task (§11.3 / O3):
deterministic halves plus the session round-trips with a canned client."""

from __future__ import annotations

import json
from types import SimpleNamespace

from finesub.llm.knowledge.node.model import digest, payload_group_hash
from finesub.llm.knowledge.node.repair import (
    render_repair_prompt,
    repair_targets,
    run_repair_session,
)
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.signals import record_evidence
from finesub.llm.knowledge.node.store import KnowledgeStore
from finesub.llm.knowledge.verify import (
    book_verify_results,
    parse_verify_results,
    render_verify_prompt,
    run_verify_session,
    unverified_claims,
)

import pytest


def _seed(store: KnowledgeStore) -> None:
    with store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": ["角色"]},
        )
        txn.create_node("T1", "term", {"surface": "ラウマ", "zh": "菈乌玛", "desc": "角色"})
        txn.create_node("T2", "term", {"surface": "フリンズ", "zh": "菲林斯", "desc": "角色"})
        txn.create_node("T3", "term", {"surface": "アイノ", "zh": "", "desc": "暂定"},
                        maturity="tentative")
        txn.create_node("R1", "term",
                        {"surface": "律にゃー", "zh": "律喵",
                         "desc": "Ver.3.0 起对听众的统称。"})
        for order, node_id in enumerate(("T1", "T2", "T3", "R1")):
            txn.create_membership(f"M{order}", "S", node_id, "角色", order)


def test_unverified_claims_scope(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        record_evidence(
            store, node_id="T1", field_path="payload:core",
            value_hash=payload_group_hash("term", store.node("T1").payload),
            verdict="confirmed", evidence_kind="external",
            source_ref="https://x", task_id="t",
        )
        claims = unverified_claims(store)
    nodes = {c["node_id"] for c in claims}
    assert "T2" in nodes           # no evidence yet
    assert "T1" not in nodes       # already confirmed for the current hash
    assert "T3" not in nodes       # tentative earns its way via shadow, not quota
    assert "R1" in nodes           # a common 分类节 term: external route open
    row = next(c for c in claims if c["node_id"] == "T2")
    assert row["subject"] == "原神" and "フリンズ|菲林斯||角色" in row["line"]
    assert "フリンズ" in render_verify_prompt(claims)


def test_parse_and_book_verify_results(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        claims = unverified_claims(store)
        by_node = {c["node_id"]: c for c in claims}
        text = (
            "查了官方站。\n<verify_results>\n"
            + json.dumps(
                [
                    {"claim_id": by_node["T2"]["claim_id"], "verdict": "confirmed",
                     "url": "https://official.example/x", "note": "官方译名"},
                    {"claim_id": by_node["T1"]["claim_id"], "verdict": "confirmed",
                     "note": "我确信"},  # confirmed WITHOUT url: an opinion, dropped
                    {"claim_id": "c999", "verdict": "confirmed", "url": "https://x"},  # unknown id
                ]
                + [{"claim_id": c["claim_id"], "verdict": "unverifiable", "note": "查不到"}
                   for c in claims if c["node_id"] not in ("T1", "T2")],
                ensure_ascii=False,
            )
            + "\n</verify_results>"
        )
        rows = parse_verify_results(text, claims)
        verdicts = {r["node_id"]: r["verdict"] for r in rows}
        assert verdicts["T2"] == "confirmed"
        assert "T1" not in verdicts  # url-less confirmation dropped
        booked = book_verify_results(store, rows, task_id="kb-verify")
        assert booked == len(rows)
        kinds = {
            (r["node_id"], r["verdict"], r["evidence_kind"])
            for r in store.conn.execute("SELECT node_id, verdict, evidence_kind FROM evidence")
        }
        assert ("T2", "confirmed", "external") in kinds
        # a second sweep no longer offers the settled claims
        assert all(c["node_id"] not in verdicts for c in unverified_claims(store))


def test_verify_session_roundtrip(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        claims = unverified_claims(store, limit=1)

        class _FakeLLM:
            def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
                assert kwargs.get("retrieval") == "native"
                return SimpleNamespace(
                    content="<verify_results>"
                    + json.dumps([{"claim_id": claims[0]["claim_id"],
                                   "verdict": "unverifiable", "note": "n"}])
                    + "</verify_results>"
                )

        rows = run_verify_session(claims, client=_FakeLLM())
    assert rows and rows[0]["verdict"] == "unverifiable"


def test_repair_targets_group_by_subject_and_session_reclassifies(tmp_path) -> None:
    """The repair session gets the surviving judgement calls grouped by
    subject — after v3 that is the episodic-desc / duplicate-term family
    (relation-review and grab-bag went away with the kinds they named)."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    targets = repair_targets(repo)
    assert set(targets) == {"S"}
    kinds = {c["kind"] for c in targets["S"]}
    assert "episodic-desc" in kinds

    prompt = render_repair_prompt(repo, "S", candidates=targets["S"])
    assert "律にゃー" in prompt and "<knowledge_proposals>" in prompt
    # material mode carries the material and the judgment fragment
    material_prompt = render_repair_prompt(repo, "S", material="新角色イルーガ登场了")
    assert "イルーガ" in material_prompt and "人际关系" in material_prompt

    class _FakeLLM:
        def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
            text = messages[0]["content"]
            handle = next(
                part.split()[0]
                for part in text.split("<!-- ")
                if part.startswith("@k") and "律にゃー" in text[: text.index(part)]
            )
            ops = [
                {"op": "remove", "id": handle.rstrip("->"), "reason": "候选1：频道名词不是人际关系"},
                {"op": "append_lines", "category": "common", "entry": "原神", "section": "频道用语",
                 "content": "律にゃー|律喵||对听众的统称，崩坏三律者捏他", "reason": "候选1：迁为频道用语"},
            ]
            return SimpleNamespace(
                content="<knowledge_proposals>\n"
                + "\n".join(json.dumps(op, ensure_ascii=False) for op in ops)
                + "\n</knowledge_proposals>"
            )

    result = run_repair_session(repo, "S", client=_FakeLLM(), candidates=targets["S"], apply=True)
    report = result["apply_report"]
    assert not report["rolled_back"] and len(report["applied"]) == 2
    assert repo.store.node("R1") is None  # relation retired
    terms = {n.payload.get("surface"): n for n in repo.store.nodes_of_kind("term")}
    assert terms["律にゃー"].payload["zh"] == "律喵"


def test_verify_accepts_url_backed_refuted(tmp_path) -> None:
    """Round 10: a source that CONTRADICTS the claim books refuted (with a
    URL), distinct from unverifiable; url-less refutations are opinions."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        claims = unverified_claims(store, limit=2)
        text = (
            "<verify_results>"
            + json.dumps([
                {"claim_id": claims[0]["claim_id"], "verdict": "refuted",
                 "url": "https://official.example/actual", "note": "official name differs"},
                {"claim_id": claims[1]["claim_id"], "verdict": "refuted", "note": "no url"},
            ])
            + "</verify_results>"
        )
        rows = parse_verify_results(text, claims)
        assert [r["verdict"] for r in rows] == ["refuted"]
        book_verify_results(store, rows, task_id="kb-verify")
        # a refuted hash is settled: it leaves the sweep until the value changes
        assert claims[0]["node_id"] not in {c["node_id"] for c in unverified_claims(store)}


def test_cli_llm_model_flag_installs_runtime_preference() -> None:
    """``--llm-model`` installs the process-local overlay on the memoized
    route loader (config.toml untouched): the value rebinds every cell to
    EXACTLY the named target/group — no fallback chain — and a bare catalog
    fact id fails loudly instead of silently routing elsewhere."""

    from finesub.llm.knowledge.maintain import _role_client
    from finesub.llm.routing.model_routes import (
        ModelRouteConfigError,
        install_runtime_preferred,
    )

    try:
        with pytest.raises(ModelRouteConfigError, match="no known target or model group"):
            _role_client(["local-claude-opus-5"])  # a FACT id is not a route target
        client = _role_client(["local-claude-completion-opus-5"])
        routes = client.router.routes
        preset = routes.presets[routes.active_preset_id]
        for group_id in set(preset.bindings.values()):
            assert routes.model_groups[group_id].target_ids == (
                "local-claude-completion-opus-5",
            )
    finally:
        install_runtime_preferred(None)


def test_a_restrictive_parent_keeps_a_shared_node_home(tmp_path) -> None:
    """Review 2026-08-29 P1-2: one node under two subjects took whichever
    membership was walked first, so a permissive section could speak for a
    restrictive one. Every placement has to agree."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("SC", "subject", {"surface": "原神", "intro": "x",
                                              "category": "common", "section_order": ["角色"]})
            txn.create_node("SS", "subject", {"surface": "某主播", "intro": "x",
                                              "category": "streamer", "section_order": ["人际关系"]})
            txn.create_node("T", "term", {"surface": "某人", "zh": "某人", "desc": "d"})
            txn.create_membership("M1", "SC", "T", "角色", 0)        # verify = external
            txn.create_membership("M2", "SS", "T", "人际关系", 0)     # verify = none
        assert "T" not in {c["node_id"] for c in unverified_claims(store)}
        # with only the permissive parent it goes out normally
        m2 = next(m for m in store.parents("T") if m.parent_id == "SS")
        with store.begin("user") as txn:
            txn.tombstone_membership("M2", expected_from_rev=m2.valid_from_rev)
        assert "T" in {c["node_id"] for c in unverified_claims(store)}
