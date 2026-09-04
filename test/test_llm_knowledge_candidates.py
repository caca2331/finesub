"""Candidate decision ledger + verdict protocol (kb-followups plan A6):
schema v4 migration, key/digest supersede semantics, repair_targets
suppression, and the propose/dismiss/needs_human booking rules."""

from __future__ import annotations

import json
from types import SimpleNamespace

from finesub.llm.knowledge.node.candidates import (
    candidate_identity,
    filter_undecided,
    pending_human,
    record_candidate_decision,
    resolve_candidate,
    standing_decisions,
)
from finesub.llm.knowledge.node.repair import repair_targets, run_repair_session
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.store import KnowledgeStore


def _seed(store: KnowledgeStore) -> None:
    with store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": ["角色"]},
        )
        # an episodic desc: still a judgement call after the v3 migration
        # (relation-review / grab-bag went away with the kinds they named)
        txn.create_node(
            "R1", "term",
            {"surface": "律にゃー", "zh": "律喵", "desc": "Ver.3.0 新登场的听众统称。"},
        )
        txn.create_membership("M0", "S", "R1", "角色", 0)


def test_schema_v3_store_migrates_to_v5_with_data_intact(tmp_path) -> None:
    path = tmp_path / "kb.sqlite"
    with KnowledgeStore(path) as store:
        _seed(store)
        assert store.node("R1") is not None
        # simulate a v3-era store: no ledger table, stored version 3
        store.conn.execute("DROP TABLE candidate_decisions")
        store.conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    with KnowledgeStore(path) as store:
        row = store.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert row["value"] == "5"
        assert store.conn.execute("SELECT COUNT(*) c FROM candidate_decisions").fetchone()["c"] == 0
        columns = {r["name"] for r in store.conn.execute("PRAGMA table_info(candidate_decisions)")}
        assert {"candidate", "missing"} <= columns
        # versioned rows untouched by the migration
        assert store.node("R1").payload["surface"] == "律にゃー"


def test_schema_v4_store_gains_the_snapshot_columns(tmp_path) -> None:
    """A store already stamped 4 (the real KB migrated before v5 existed) gets
    the two human-facing columns from the 4→5 ladder step."""

    path = tmp_path / "kb.sqlite"
    with KnowledgeStore(path) as store:
        _seed(store)
        store.conn.execute("ALTER TABLE candidate_decisions DROP COLUMN candidate")
        store.conn.execute("ALTER TABLE candidate_decisions DROP COLUMN missing")
        store.conn.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
    with KnowledgeStore(path) as store:
        assert store.meta("schema_version") == "5"
        columns = {r["name"] for r in store.conn.execute("PRAGMA table_info(candidate_decisions)")}
        assert {"candidate", "missing"} <= columns


def test_ledger_key_content_split_and_supersede(tmp_path) -> None:
    candidate = {"kind": "episodic-desc", "subject_id": "S", "node": "T1", "desc": "旧内容"}
    key, digest1 = candidate_identity(candidate)
    key2, digest2 = candidate_identity({**candidate, "desc": "新内容"})
    assert key == key2 and digest1 != digest2  # identity stable, content moves

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        record_candidate_decision(
            store, candidate_key=key, content_digest=digest1,
            status="resolved", resolution="dismissed", reason="不成立", task_id="t1",
        )
        assert filter_undecided(store, [candidate]) == []  # settled content suppressed
        changed = {**candidate, "desc": "新内容"}
        assert filter_undecided(store, [changed]) == [changed]  # content moved on: visible
        # booking against the new digest supersedes the stale row
        record_candidate_decision(
            store, candidate_key=key, content_digest=digest2,
            status="pending_human", reason="查不到佐证", task_id="t2",
        )
        standing = standing_decisions(store)
        assert standing[key]["content_digest"] == digest2
        assert standing[key]["status"] == "pending_human"
        statuses = [r["status"] for r in store.conn.execute(
            "SELECT status FROM candidate_decisions ORDER BY decision_id")]
        assert statuses == ["superseded", "pending_human"]
        # pending rows are what the report lists; human resolve closes them
        assert [r["candidate_key"] for r in pending_human(store)] == [key]
        assert resolve_candidate(store, key, reason="人工确认")
        assert pending_human(store) == []
        assert standing_decisions(store)[key]["resolution"] == "human"


def _fake_llm(payload_text: str):  # type: ignore[no-untyped-def]
    class _FakeLLM:
        def complete(self, role, messages, **kwargs):
            return SimpleNamespace(content=payload_text)

    return _FakeLLM()


def test_verdicts_book_dismiss_needs_human_and_gated_propose(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    targets = repair_targets(repo)
    assert set(targets) == {"S"}  # explicit subject_id routing
    candidates = targets["S"]
    assert [c["kind"] for c in candidates] == ["episodic-desc"]

    # needs_human → pending_human row; the candidate stops resurfacing
    output = (
        "<knowledge_proposals>\n</knowledge_proposals>\n"
        "<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "needs_human", "reason": "身份查不到佐证"},
                     ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )
    result = run_repair_session(
        repo, "S", client=_fake_llm(output), candidates=candidates, apply=True,
    )
    assert result["candidate_ledger"] == [{"candidate": "@c1", "status": "pending_human"}]
    assert len(pending_human(repo.store)) == 1
    assert repair_targets(repo) == {}  # suppressed until a human decides


def test_propose_resolves_only_when_the_scan_stops_finding_the_candidate(tmp_path) -> None:
    """Review 2026-08-28 P1-2: the completion assertion is the POST-APPLY scan
    — resolved(applied) must mean the underlying condition is gone, never
    "some op landed"."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    candidates = repair_targets(repo)["S"]

    # propose whose op is REFUSED at translate (metadata is harness-owned):
    # nothing changes, the scan still finds the candidate → stays open
    bad = (
        "<knowledge_proposals>\n"
        + json.dumps({"op": "append_lines", "entry": "原神", "category": "common",
                      "section": "元数据", "content": "律にゃー|律喵||听众统称",
                      "reason": "@c1 迁为频道用语"}, ensure_ascii=False)
        + "\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "propose", "reason": "迁移"}, ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )
    result = run_repair_session(repo, "S", client=_fake_llm(bad), candidates=candidates, apply=True)
    assert result["candidate_ledger"][0]["status"] == "open"
    assert repair_targets(repo) != {}  # still visible

    # PARTIAL fix: the append half lands but the relation itself survives —
    # under the old "any op landed + handle cited" rule this closed the
    # candidate and the ledger then suppressed it forever
    partial = (
        "<knowledge_proposals>\n"
        + json.dumps({"op": "append_lines", "entry": "原神", "category": "common",
                      "section": "频道用语", "content": "律にゃー|律喵||听众统称",
                      "reason": "@c1 迁为频道用语（删除半边缺失）"}, ensure_ascii=False)
        + "\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "propose", "reason": "迁移"}, ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )
    result = run_repair_session(repo, "S", client=_fake_llm(partial), candidates=candidates, apply=True)
    assert result["candidate_ledger"][0]["status"] == "open"
    assert repair_targets(repo) != {}  # relation survives: candidate resurfaces

    # COMPLETE fix: the relation is removed too → scan no longer finds the
    # candidate → resolved(applied)
    good = (
        "<knowledge_proposals>\n"
        + json.dumps({"op": "remove", "id": "__HANDLE__", "reason": "@c1 频道名词不是人际关系"})
        + "\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "propose", "reason": "迁移"}, ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )

    class _HandleLLM:
        def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
            text = messages[0]["content"]
            handle = next(
                part.split()[0].rstrip("->")
                for part in text.split("<!-- ")
                if part.startswith("@k") and "律にゃー" in text[: text.index(part)]
            )
            return SimpleNamespace(content=good.replace("__HANDLE__", handle))

    result = run_repair_session(repo, "S", client=_HandleLLM(), candidates=candidates, apply=True)
    ledger = result["candidate_ledger"]
    assert ledger == [{"candidate": "@c1", "status": "resolved", "resolution": "applied"}]
    assert repair_targets(repo) == {}  # relation retired AND ledger settled


def test_pending_row_carries_snapshot_and_reconciles_freshness(tmp_path) -> None:
    """Review 2026-08-28 P2-4: a pending row must be legible without the hash
    (candidate snapshot + missing evidence), and the listing reconciles
    against the current scan instead of showing stale decisions as live."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    candidates = repair_targets(repo)["S"]
    output = (
        "<knowledge_proposals>\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "needs_human",
                      "reason": "对象身份查不到", "missing": "该名号的官方出处"},
                     ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )
    run_repair_session(repo, "S", client=_fake_llm(output), candidates=candidates, apply=True)

    from finesub.llm.knowledge.node.candidates import pending_human_reconciled

    rows = pending_human_reconciled(repo.store)
    assert len(rows) == 1
    assert rows[0]["missing"] == "该名号的官方出处"
    assert "律にゃー" in rows[0]["candidate"]  # human-legible snapshot, not a hash
    assert rows[0]["freshness"] == "current"

    # the underlying relation gets removed by an unrelated edit: the pending
    # row is now stale and the reconciled listing says so
    with repo.store.begin("user") as txn:
        txn.tombstone_membership("M0")
        txn.tombstone_node("R1")
    rows = pending_human_reconciled(repo.store)
    assert rows[0]["freshness"] == "gone"


def test_snapshot_stays_valid_json_when_bounded(tmp_path) -> None:
    """Review 2026-08-28 P2-1: bounding happens by shrinking fields, never by
    slicing the serialized text — the column must always hold valid JSON."""

    from finesub.llm.knowledge.node.candidates import _candidate_snapshot

    huge = {"kind": "episodic-desc", "subject_id": "S", "node": "T1",
            "desc": "长" * 3000, "hint": "提示" * 500}
    text = _candidate_snapshot(huge)
    assert len(text) <= 2000
    parsed = json.loads(text)  # must not raise
    assert parsed["kind"] == "episodic-desc" and parsed["node"] == "T1"


def test_needs_human_missing_falls_back_to_reason_or_stays_open(tmp_path) -> None:
    """Review 2026-08-28 P2-2: a pending row must always say what evidence is
    missing — absent ``missing`` falls back to the reason; a verdict carrying
    neither books nothing and the candidate stays open."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    candidates = repair_targets(repo)["S"]

    empty = (
        "<knowledge_proposals>\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "needs_human", "reason": ""})
        + "\n</candidate_verdicts>"
    )
    result = run_repair_session(repo, "S", client=_fake_llm(empty), candidates=candidates, apply=True)
    assert result["candidate_ledger"][0]["status"] == "open"
    assert pending_human(repo.store) == []

    # JSON null must not sneak past as the string "None" (post-merge review):
    # nullable fields are common model output
    nulls = (
        "<knowledge_proposals>\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "needs_human",
                      "reason": None, "missing": None})
        + "\n</candidate_verdicts>"
    )
    result = run_repair_session(repo, "S", client=_fake_llm(nulls), candidates=candidates, apply=True)
    assert result["candidate_ledger"][0]["status"] == "open"
    assert pending_human(repo.store) == []

    fallback = (
        "<knowledge_proposals>\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "needs_human",
                      "reason": "身份查不到佐证"}, ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )
    run_repair_session(repo, "S", client=_fake_llm(fallback), candidates=candidates, apply=True)
    rows = pending_human(repo.store)
    assert rows and rows[0]["missing"] == "身份查不到佐证"  # reason fallback


def test_dry_run_books_nothing(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    candidates = repair_targets(repo)["S"]
    output = (
        "<knowledge_proposals>\n</knowledge_proposals>\n<candidate_verdicts>\n"
        + json.dumps({"candidate": "@c1", "verdict": "dismiss", "reason": "不成立"}, ensure_ascii=False)
        + "\n</candidate_verdicts>"
    )
    result = run_repair_session(repo, "S", client=_fake_llm(output), candidates=candidates, apply=False)
    assert result["candidate_verdicts"] and "candidate_ledger" not in result
    assert standing_decisions(repo.store) == {}  # dry-run wrote nothing


def test_human_resolve_settles_a_candidate_whose_content_moved(tmp_path) -> None:
    """The usual way a person settles a pending row is to go FIX the line —
    which changes the content digest. Re-stamping the standing row booked the
    decision against content that no longer exists, so `filter_undecided` kept
    surfacing the candidate forever and `--resolve` could never close it."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        from finesub.llm.knowledge.node.scan import scan_candidates

        before = scan_candidates(store).candidates
        assert len(before) == 1
        key, digest_before = candidate_identity(before[0])
        record_candidate_decision(
            store, candidate_key=key, content_digest=digest_before,
            status="pending_human", reason="需要人工", candidate=before[0],
            missing="出处",
        )
        assert filter_undecided(store, before) == []

        # the person edits the line; the same question now has new content
        node = store.node("R1")
        with store.begin("user") as txn:
            txn.update_node("R1", payload={**node.payload, "desc": "近期新登场的听众统称。"},
                            expected_from_rev=node.valid_from_rev)
        after = scan_candidates(store).candidates
        assert candidate_identity(after[0]) != (key, digest_before)
        assert filter_undecided(store, after) == after      # standing row is stale

        assert resolve_candidate(store, key, reason="已处理", candidates=after)
        assert filter_undecided(store, after) == []
        assert pending_human(store) == []
        standing = standing_decisions(store)[key]
        assert standing["resolution"] == "human"
        assert standing["content_digest"] == candidate_identity(after[0])[1]


def test_human_resolve_still_closes_a_candidate_that_is_gone(tmp_path) -> None:
    """The other half: when the fix REMOVED the candidate there is nothing
    live to book against, and the standing row must still close."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        from finesub.llm.knowledge.node.scan import scan_candidates

        candidates = scan_candidates(store).candidates
        key, content = candidate_identity(candidates[0])
        record_candidate_decision(
            store, candidate_key=key, content_digest=content,
            status="pending_human", reason="需要人工", candidate=candidates[0],
            missing="出处",
        )
        assert resolve_candidate(store, key, reason="行已删", candidates=[])
        assert pending_human(store) == []
        assert standing_decisions(store)[key]["resolution"] == "human"
