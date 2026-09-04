from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.knowledge import base as knowledge_base
from finesub.llm.knowledge.base import (
    append_task_artifact,
    knowledge_version,
    load_entry_texts,
    load_index_entries,
    load_index_text,
    load_preinjected_entries,
    match_index_keywords,
    parse_knowledge_proposals_jsonl,
    read_task_artifacts,
    resolve_entry_key,
)
from finesub.llm.knowledge.node.proposals import apply_model_proposals, parse_model_proposals
from finesub.llm.knowledge.node.render import HandleMap
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _seed(root: Path) -> None:
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text(
        "- 星野灯 | Hoshino Akari | 阿灯、小灯 | 个人势 VTuber\n", encoding="utf-8"
    )
    (root / "streamer" / "星野灯.md").write_text(
        "# 星野灯\n个人势 VTuber\n\n## 档案\n本名: 星野灯（ほしの あかり）/ Hoshino Akari\n别名: 阿灯、小灯\n\n"
        "## 直播内容\n\n杂谈\n\n## 说话风格\n自称: 灯\n\n## 喜好 / 特点\n\n## 重要经历\n\n## 人际关系\n\n"
        "## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (root / "common" / "index.md").write_text(
        "- 崩坏星穹铁道 [游戏] | Honkai: Star Rail | 崩铁、星铁 | HoYoverse RPG\n", encoding="utf-8"
    )
    (root / "common" / "崩坏星穹铁道.md").write_text(
        "# 崩坏星穹铁道\nHoYoverse RPG\n\n## 档案\n本名: 崩坏星穹铁道 / Honkai: Star Rail\n别名: 崩铁、星铁\n\n"
        "## 角色\n\n三月七|三月七|March 7th||开朗少女。常被 ASR 误听为「三月期」。\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )


def _apply(root: Path, proposals: list[dict], *, with_handles: bool = True) -> dict:
    repo = KnowledgeRepo.open(root)
    handles = HandleMap()
    if with_handles:
        for subject in repo.subjects():
            repo.entry_prompt_text(subject.local_id, handles)
    text = "<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals) + "\n</knowledge_proposals>"
    return apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles).to_dict()


def test_default_knowledge_root_is_repo_root_knowledge_dir() -> None:
    assert knowledge_base.DEFAULT_KNOWLEDGE_ROOT is None or knowledge_base.DEFAULT_KNOWLEDGE_ROOT.name == "knowledge"


def test_parse_knowledge_proposals_jsonl_accepts_wrapped_and_rejects_bad_json() -> None:
    rows = parse_knowledge_proposals_jsonl('<knowledge_proposals>\n{"op":"remove","id":"@k2"}\n</knowledge_proposals>')
    assert rows == [{"op": "remove", "id": "@k2"}]
    with pytest.raises(ValueError):
        parse_knowledge_proposals_jsonl("<knowledge_proposals>\n{bad\n</knowledge_proposals>")
    assert parse_model_proposals("```jsonl\n{\"op\":\"remove\",\"id\":\"@k2\"}\n```") == [{"op": "remove", "id": "@k2"}]


def test_legacy_tree_is_imported_once_and_read_through_the_store(tmp_path) -> None:
    _seed(tmp_path)
    assert knowledge_version(tmp_path) == "rev:1"
    assert (tmp_path / "knowledge.sqlite").exists()
    assert "- 星野灯 | Hoshino Akari | 阿灯、小灯 | 个人势 VTuber" in load_index_text(tmp_path, "streamer")
    assert [e.key for e in load_index_entries(tmp_path, "common")] == ["崩坏星穹铁道"]
    assert resolve_entry_key(tmp_path, "崩铁") == ("common", "崩坏星穹铁道")
    found, missing = load_entry_texts(tmp_path, ["阿灯", "星野灯", "不存在"])
    assert list(found) == ["星野灯"] and missing == ["不存在"]
    assert found["星野灯"].startswith("# 星野灯\n个人势 VTuber\n")
    # derived cache exists, markdown source is no longer consulted
    assert (tmp_path / "rendered" / "streamer" / "星野灯.md").exists()
    (tmp_path / "streamer" / "星野灯.md").write_text("# 星野灯\n改了也没用\n\n## 元数据\n最近更新日期: 2026-08-02\n", encoding="utf-8")
    assert "改了也没用" not in load_entry_texts(tmp_path, ["星野灯"])[0]["星野灯"]


def test_match_index_keywords_matches_keys_and_aliases_with_frequency_rank(tmp_path) -> None:
    _seed(tmp_path)
    matches = match_index_keywords(tmp_path, "今天阿灯和崩铁，崩铁又上新了，星铁也行")
    assert [(m.category, m.key, m.hits) for m in matches] == [("common", "崩坏星穹铁道", 3), ("streamer", "星野灯", 1)]
    assert matches[0].matched_terms == ("崩铁", "星铁")


def test_match_index_keywords_skips_short_terms_and_caps_entries(tmp_path) -> None:
    _seed(tmp_path)
    assert match_index_keywords(tmp_path, "灯") == []
    assert len(match_index_keywords(tmp_path, "阿灯 崩铁", max_entries=1)) == 1


def test_load_preinjected_entries_returns_bodies_in_rank_order(tmp_path) -> None:
    _seed(tmp_path)
    found, matches = load_preinjected_entries(tmp_path, "崩铁 崩铁 阿灯")
    assert list(found) == ["崩坏星穹铁道", "星野灯"]
    assert [m.key for m in matches] == ["崩坏星穹铁道", "星野灯"]


def test_task_artifact_retention_round_trip(tmp_path) -> None:
    path = append_task_artifact(tmp_path, kind="x", payload={"a": 1}, task_id="t")
    assert path.name == knowledge_base.TASK_ARTIFACT_FILENAME
    text = read_task_artifacts([tmp_path], count_tokens=len)
    assert '"kind": "x"' in text


# ---- model proposals through the store -------------------------------------------------


def test_append_lines_creates_nodes_dedups_and_indexes_misheard(tmp_path) -> None:
    _seed(tmp_path)
    report = _apply(tmp_path, [
        {"op": "append_lines", "category": "common", "entry": "崩铁", "section": "角色",
         "content": "三月七|三月七||重复行\n丹恒|丹恒|Dan Heng|冷静青年。误听: 单横", "reason": "seed"},
    ])
    assert report["rev"] == 2 and not report["rolled_back"]
    assert report["applied"][0]["reason"].startswith("1 rows")
    text = load_entry_texts(tmp_path, ["崩铁"])[0]["崩坏星穹铁道"]
    # 误听 notation is transport syntax: the variant lands as an item, the
    # stored desc drops it, the alias column renders from items (plan §11.2)
    assert "丹恒|丹恒|Dan Heng|冷静青年。" in text
    assert "误听: 单横" not in text
    assert sum(1 for line in text.splitlines() if line.startswith("三月七|")) == 1
    assert "最近更新日期: " + KnowledgeRepo.open(tmp_path).today() in text
    repo = KnowledgeRepo.open(tmp_path)
    values = {i.value for i in repo.store.all_items() if i.field == "misheard"}
    assert {"三月期", "单横"} <= values
    # harness applies book revision-level transcript provenance (plan §11.4)
    kinds = {r["evidence_kind"] for r in repo.store.conn.execute("SELECT evidence_kind FROM evidence")}
    assert "transcript" in kinds


def test_update_and_remove_by_handle(tmp_path) -> None:
    _seed(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    handles = HandleMap()
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    rendered = repo.entry_prompt_text(subject.local_id, handles)
    line_handle = next(h for h, (ident, _) in handles.nodes.items() if repo.store.node(ident).kind == "term")
    assert f"<!-- {line_handle} " in rendered
    text = "<knowledge_proposals>\n" + json.dumps({"op": "update", "id": line_handle, "line": "三月七|三月七|March 7th、さんがつなのか|开朗少女。", "reason": "r"}, ensure_ascii=False) + "\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles).to_dict()
    assert report["applied"][0]["op"] == "update"
    # the alias column (third, 顿号-separated) diffs into alias items and the
    # rendered line stays fixed four-column (plan A1)
    assert "三月七|三月七|March 7th、さんがつなのか|开朗少女。" in load_entry_texts(tmp_path, ["崩铁"])[0]["崩坏星穹铁道"]
    # kind change is refused; remove works
    handles2 = HandleMap()
    repo.entry_prompt_text(subject.local_id, handles2)
    line_handle2 = next(h for h, (ident, _) in handles2.nodes.items() if repo.store.node(ident).kind == "term")
    text = "<knowledge_proposals>\n" + json.dumps({"op": "update", "id": line_handle2, "line": "不是五段行", "reason": "r"}, ensure_ascii=False) + "\n" + json.dumps({"op": "remove", "id": line_handle2, "reason": "r"}) + "\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles2).to_dict()
    assert [r["op"] for r in report["skipped"]] == ["update"] and "行体" in report["skipped"][0]["reason"]
    assert "三月七|" not in load_entry_texts(tmp_path, ["崩铁"])[0]["崩坏星穹铁道"]


def test_create_entry_scaffolds_and_index_line_follows(tmp_path) -> None:
    _seed(tmp_path)
    report = _apply(tmp_path, [
        {"op": "create_entry", "category": "common", "entry": "原神", "entry_type": "游戏", "intro": "开放世界", "aliases": ["げんしん"], "reason": "没有母词条"},
        {"op": "append_lines", "category": "common", "entry": "原神", "section": "角色", "content": "派蒙|派蒙|Paimon|向导", "reason": "seed"},
        {"op": "create_entry", "category": "common", "entry": "绝区零", "entry_type": "桌游", "intro": "x", "reason": "r"},
        {"op": "create_entry", "category": "common", "entry": "崩铁", "entry_type": "游戏", "intro": "x", "reason": "r"},
    ])
    # per-op records (plan A4): alias registrations report their own lines
    assert [(r["op"], r["reason"]) for r in report["applied"]] == [
        ("create_entry", "scaffolded"),
        ("create_entry", "alias added: げんしん"),
        ("append_lines", "1 rows"),
        ("append_lines", "alias added: Paimon"),
    ]
    assert "entry_type" in report["skipped"][0]["reason"]
    assert "already exists" in report["skipped"][1]["reason"]
    text = load_entry_texts(tmp_path, ["げんしん"])[0]["原神"]
    # no empty-value scaffold facts (其他: is gone), term lines are 3/4-segment
    # load_entry_texts is model-facing: PROMPT projection, bare lines (round 12)
    # model-facing PARTIAL preview: bare lines, no guidance comments, no slots
    assert text.startswith("# 原神\n开放世界\n\n## 档案\n\n[本名] 原神\n\n## 角色\n\n派蒙|派蒙|Paimon|向导\n")
    assert "- 原神 [游戏] |  | げんしん | 开放世界" in load_index_text(tmp_path, "common")
    streamer = _apply(tmp_path, [{"op": "create_entry", "category": "streamer", "entry": "新人", "intro": "x", "reason": "r"}])
    assert streamer["applied"]
    new_text = load_entry_texts(tmp_path, ["新人"])[0]["新人"]
    # this is the model INJECTION face (partial preview): empty sections and
    # empty slots are noise there, and no blank line was ever stored
    assert "## 特点" not in new_text and "自称" not in new_text
    # the human/update face (full preview) is where the checklist lives
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    full = repo.entry_text(repo.resolve("新人").subject_id)
    assert "## 特点" in full and "## 频道用语" in full
    assert "- [自称]" in full and "自称: " not in full


def test_retire_rename_and_append_to_strict_sections(tmp_path) -> None:
    _seed(tmp_path)
    report = _apply(tmp_path, [
        {"op": "append_lines", "category": "streamer", "entry": "星野灯", "section": "自由节", "content": "x", "reason": "r"},
        {"op": "append_lines", "category": "streamer", "entry": "星野灯", "section": "元数据", "content": "x", "reason": "r"},
        {"op": "retire_entry", "category": "streamer", "entry": "星野灯", "merged_into": "不存在", "reason": "r"},
        {"op": "rename_entry", "category": "common", "entry": "崩铁", "new_key": "崩坏：星穹铁道", "reason": "r"},
    ])
    assert [r["op"] for r in report["skipped"]] == ["append_lines", "append_lines", "retire_entry"]
    assert report["applied"][0]["op"] == "rename_entry"
    assert resolve_entry_key(tmp_path, "崩铁") == ("common", "崩坏：星穹铁道")
    report = _apply(tmp_path, [{"op": "retire_entry", "category": "streamer", "entry": "星野灯", "reason": "并入"}])
    assert report["applied"][0]["op"] == "retire_entry"
    assert resolve_entry_key(tmp_path, "星野灯") is None
    assert knowledge_version(tmp_path) == "rev:3"


def test_invalid_rows_are_reported_not_fatal(tmp_path) -> None:
    _seed(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    text = "<knowledge_proposals>\n{not json\n{\"op\":\"fly\"}\n{\"op\":\"update\",\"id\":\"@k99\",\"line\":\"x\"}\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev).to_dict()
    assert [r["op"] for r in report["skipped"]] == ["invalid", "fly", "update"]
    assert report["rev"] is None and repo.rev == 1


def test_generation_pin_freezes_default_reads(tmp_path) -> None:
    """plan §2.5: one run reads one revision; explicit rev overrides the pin."""

    from finesub.llm.knowledge.base import load_index_text, pinned_generation_rev

    root = tmp_path / "pinned-kb"
    root.mkdir()
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("user") as txn:
        txn.create_node(
            "A",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": [],
             "native_names": [], "entry_type": "游戏"},
        )
    with pinned_generation_rev(root) as rev:
        assert rev == repo.rev
        before = load_index_text(root, "common")
        assert "原神" in before
        with repo.store.begin("user") as txn:
            txn.create_node(
                "B",
                "subject",
                {"surface": "新条目", "intro": "x", "category": "common", "section_order": [],
                 "native_names": [], "entry_type": "其他"},
            )
        assert load_index_text(root, "common") == before  # pinned: the new rev is invisible
        assert "新条目" in load_index_text(root, "common", rev=repo.rev)  # explicit rev wins
    assert "新条目" in load_index_text(root, "common")  # pin released


def test_generation_pins_of_concurrent_runs_are_isolated(tmp_path) -> None:
    """Two runs pinning one root see their own rev, concurrently (plan W1).

    The pin is context-scoped: run B pins while run A still holds its pin --
    no serialization -- and each run's default reads resolve to its own rev,
    including after the other run has exited (no stale restore)."""

    import threading

    from finesub.llm.knowledge.base import load_index_text, pinned_generation_rev

    root = tmp_path / "pin-race-kb"
    root.mkdir()
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("user") as txn:
        txn.create_node(
            "A",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": [],
             "native_names": [], "entry_type": "游戏"},
        )
    a_pinned = threading.Event()
    b_done = threading.Event()
    seen: dict[str, tuple[int, str]] = {}

    def run_a() -> None:
        with pinned_generation_rev(root) as rev:
            a_pinned.set()
            assert b_done.wait(10)  # B pins, reads and exits while A holds
            seen["a"] = (rev, load_index_text(root, "common"))

    def run_b() -> None:
        assert a_pinned.wait(10)
        repo_b = KnowledgeRepo.open(root)  # SQLite: one connection per thread
        with repo_b.store.begin("user") as txn:  # a new revision lands mid-run-A
            txn.create_node(
                "B",
                "subject",
                {"surface": "新条目", "intro": "x", "category": "common", "section_order": [],
                 "native_names": [], "entry_type": "其他"},
            )
        with pinned_generation_rev(root) as rev:  # does NOT wait for A
            seen["b"] = (rev, load_index_text(root, "common"))
        b_done.set()

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(10)
    thread_b.join(10)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert seen["a"][0] == 1 and "新条目" not in seen["a"][1]
    assert seen["b"][0] == 2 and "新条目" in seen["b"][1]


def test_phase_repin_moves_the_snapshot_and_the_exit_still_restores(tmp_path) -> None:
    """Plan W3: a phase boundary re-reads the current rev — the run's default
    reads move to it — while leaving the run's pin scope intact on exit."""

    from finesub.llm.knowledge.base import (
        load_index_text,
        pinned_generation_rev,
        repin_generation_rev,
    )

    root = tmp_path / "repin-kb"
    root.mkdir()
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("user") as txn:
        txn.create_node(
            "A",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": [],
             "native_names": [], "entry_type": "游戏"},
        )
    assert repin_generation_rev(root) is None  # no pin active: no-op
    with pinned_generation_rev(root) as rev:
        assert rev == 1
        with repo.store.begin("user") as txn:  # another task commits mid-run
            txn.create_node(
                "B",
                "subject",
                {"surface": "新条目", "intro": "x", "category": "common", "section_order": [],
                 "native_names": [], "entry_type": "其他"},
            )
        assert "新条目" not in load_index_text(root, "common")  # still phase 1
        assert repin_generation_rev(root) == 2  # the barrier
        assert "新条目" in load_index_text(root, "common")  # phase 2 snapshot
    assert repin_generation_rev(root) is None  # pin gone after the run


def test_in_process_writers_share_the_write_lock(tmp_path) -> None:
    """Plan W2: two tasks of one process both get True — the loser path is
    for other PROCESSES; in-process serialization is the apply queue's job."""

    import threading

    from finesub.llm.knowledge.base import knowledge_write_lock

    root = tmp_path / "kb"
    root.mkdir()
    first_in = threading.Event()
    release = threading.Event()
    seen: list[bool] = []

    def holder() -> None:
        with knowledge_write_lock(root) as acquired:
            seen.append(acquired)
            first_in.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    assert first_in.wait(10)
    with knowledge_write_lock(root) as acquired:  # while the other thread holds it
        seen.append(acquired)
    release.set()
    thread.join(10)
    assert seen == [True, True]
    with knowledge_write_lock(root) as acquired:  # fully released afterwards
        assert acquired


def test_another_process_holding_the_lock_still_wins(tmp_path, monkeypatch) -> None:
    """The cross-process loser semantics stay: a foreign holder means False."""

    from finesub.llm.knowledge import base as base_module
    from finesub.llm.knowledge.base import knowledge_lock_path, knowledge_write_lock
    from finesub_bootstrap.locks import holding_lock

    root = tmp_path / "kb"
    root.mkdir()
    monkeypatch.setattr(base_module, "KNOWLEDGE_LOCK_TIMEOUT_SECONDS", 0.2)
    # A second file handle on the same byte lock stands in for another process.
    with holding_lock(knowledge_lock_path(root), timeout=0):
        with knowledge_write_lock(root) as acquired:
            assert acquired is False


def test_cli_generation_pin_helper_skips_knowledge_none(tmp_path) -> None:
    """--knowledge none must not open (or create) the store just to pin it."""

    from finesub.llm.correction_translation import _generation_pin

    root = tmp_path / "cli-pin-kb"
    root.mkdir()
    with _generation_pin({"knowledge": "none", "knowledge_root": root}):
        pass
    assert not (root / "knowledge.sqlite").exists()
    with _generation_pin({"knowledge": "collect", "knowledge_root": root}):
        pass
    assert (root / "knowledge.sqlite").exists()


def test_shared_node_remove_is_refused(tmp_path) -> None:
    """Round 12: ``remove`` retires the node globally — on a node shared by
    two subjects (share merge creates these) that would delete it from the
    OTHER entry too, so the proposal is skipped instead."""

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S1", "subject", {"surface": "主播甲", "intro": "x", "category": "streamer",
                                          "section_order": ["角色"]})
        txn.create_node("S2", "subject", {"surface": "主播乙", "intro": "y", "category": "streamer",
                                          "section_order": ["角色"]})
        txn.create_node("T", "term", {"surface": "共有梗", "zh": "共有", "desc": "两人共用"})
        txn.create_membership("M1", "S1", "T", "角色", 0)
        txn.create_membership("M2", "S2", "T", "角色", 0)
    handles = HandleMap()
    repo.entry_prompt_text("S1", handles)
    line_handle = next(h for h, (ident, _) in handles.nodes.items() if ident == "T")
    text = ("<knowledge_proposals>\n"
            + json.dumps({"op": "remove", "id": line_handle, "reason": "r"})
            + "\n</knowledge_proposals>")
    report = apply_model_proposals(text, repo=repo, task_id="t",
                                   knowledge_read_rev=repo.rev, handles=handles).to_dict()
    assert report["skipped"] and "shared" in report["skipped"][0]["reason"]
    assert repo.store.node("T") is not None  # still alive under both subjects
