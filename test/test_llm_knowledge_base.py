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
    read_task_artifacts,
    resolve_entry_key,
)
from finesub.llm.knowledge.node.proposals import apply_model_proposals, parse_model_proposals
from finesub.llm.knowledge.node.render import HandleMap
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.update import _validate_knowledge_update_jsonl


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


def test_the_retry_gate_reads_proposals_as_the_commit_does() -> None:
    _validate_knowledge_update_jsonl('<knowledge_proposals>\n{"op":"remove","id":"@k2"}\n</knowledge_proposals>')
    with pytest.raises(ValueError):
        _validate_knowledge_update_jsonl("<knowledge_proposals>\n{bad\n</knowledge_proposals>")
    # a draft line in the reasoning is not the answer (the commit strips <reasoning> too)
    _validate_knowledge_update_jsonl(
        "<reasoning>先写 <knowledge_proposals> 草稿：{op: remove}</reasoning>\n"
        '<knowledge_proposals>\n{"op":"remove","id":"@k2"}\n</knowledge_proposals>'
    )
    # a line the commit would skip as invalid sends the reply back, whatever it starts with
    with pytest.raises(ValueError, match="块里有 1 行不是 JSON 对象"):
        _validate_knowledge_update_jsonl('<knowledge_proposals>\n- {"op":"remove","id":"@k2"}\n</knowledge_proposals>')
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
    # the read-only injection carries no metadata; the full projection does
    assert "最近更新日期" not in text
    repo = KnowledgeRepo.open(tmp_path)
    assert "最近更新日期: " + repo.today() in repo.entry_text(repo.resolve("崩铁").subject_id)
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
    # `角色` takes free sentences now, so the refusal comes from the kind guard
    # rather than from the section: rewriting a term into prose in place is
    # still remove + append_lines.
    assert [r["op"] for r in report["skipped"]] == ["update"]
    assert "change kind" in report["skipped"][0]["reason"], report["skipped"]
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


def test_the_identity_line_create_entry_adds_is_in_the_batch_s_label_ledger(tmp_path) -> None:
    """A new entry, then its own `[本名]` in 档案 (Haiku, 2026-11). The identity line `create_entry` adds was not in
    the batch's label ledger, so the second `[本名]` passed translation and failed the whole batch in
    `validate_overlay`. Now it is refused per op with a reason the repair round takes, and the proposal's other line
    lands."""

    from finesub.llm.knowledge.update import _dropped_rows_text, _is_rejection, rejected_entries

    _seed(tmp_path)
    proposals = [
        {"op": "create_entry", "category": "common", "entry": "原神", "entry_type": "游戏", "intro": "开放世界",
         "reason": "没有母词条"},
        {"op": "append_lines", "category": "common", "entry": "原神", "section": "档案",
         "content": "[本名] 原神 / Genshin Impact\n[收录范围] 只收角色名", "reason": "seed"},
    ]
    report = _apply(tmp_path, proposals)

    assert not report["rolled_back"] and report["rev"] is not None
    assert [(r["op"], r["section"], r["reason"]) for r in report["skipped"]] == [
        ("append_lines", "档案", "标记 [本名] 在本节已存在——改用 update")
    ]
    text = load_entry_texts(tmp_path, ["原神"])[0]["原神"]
    assert "[收录范围] 只收角色名" in text and "Genshin Impact" not in text
    assert rejected_entries(report) == [("common", "原神")]
    block = "<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals) \
        + "\n</knowledge_proposals>"
    rows = _dropped_rows_text(report, [("common", "原神")], keep=_is_rejection, proposal_text=block)
    assert "标记 [本名] 在本节已存在" in rows and "Genshin Impact" in rows


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
    with _generation_pin("none", root):
        pass
    assert not (root / "knowledge.sqlite").exists()
    with _generation_pin("read", root):
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


# ---------------------------------------------------------------------------
# 2026-09-07: the two moves (plan §0 requirements 7, 11, 13)


def _subject(root: Path, surface: str):  # type: ignore[no-untyped-def]
    repo = KnowledgeRepo.open(root)
    return repo, next(s for s in repo.subjects() if s.payload["surface"] == surface)


def test_move_line_relocates_a_line_instead_of_retyping_it(tmp_path: Path) -> None:
    """The engine op existed all along; the model just could not reach it.

    What matters is that the NODE survives: deleting and re-appending the text
    would produce a different node, and everything hanging off the old one --
    history, evidence, canonical id -- would be silently orphaned.
    """

    _seed(tmp_path)
    repo, subject = _subject(tmp_path, "星野灯")
    handles = HandleMap()
    repo.entry_prompt_text(subject.local_id, handles)
    membership = next(
        m for m in repo.store.children(subject.local_id) if m.section == "直播内容"
    )
    before = membership.child_id
    line_handle = next(
        handle for handle, (ident, _) in handles.nodes.items() if ident == before
    )

    report = _apply(tmp_path, [
        {"op": "move_line", "id": line_handle, "section": "特点", "reason": "杂谈是特点不是内容"}
    ])
    assert [record["op"] for record in report["applied"]] == ["move_line"], report

    repo = KnowledgeRepo.open(tmp_path)
    placed = {m.section: m.child_id for m in repo.store.children(subject.local_id)}
    assert placed.get("特点") == before, "the node itself must survive the move"
    assert "直播内容" not in placed or placed["直播内容"] != before


def test_move_line_refuses_a_section_the_template_does_not_have(tmp_path: Path) -> None:
    _seed(tmp_path)
    repo, subject = _subject(tmp_path, "星野灯")
    handles = HandleMap()
    repo.entry_prompt_text(subject.local_id, handles)
    membership = next(
        m for m in repo.store.children(subject.local_id) if m.section == "直播内容"
    )
    line_handle = next(
        handle for handle, (ident, _) in handles.nodes.items()
        if ident == membership.child_id
    )
    report = _apply(tmp_path, [
        {"op": "move_line", "id": line_handle, "section": "不存在的节", "reason": "r"}
    ])
    assert report["applied"] == []
    assert "没有" in report["skipped"][0]["reason"]


def test_recategorize_needs_a_reason_and_records_it_for_maintenance_only(
    tmp_path: Path,
) -> None:
    """§0 requirement 13: the reason is stored, and shown only where the entry
    can be moved again -- never on the injection surface a correction round
    pays tokens for."""

    _seed(tmp_path)

    # No reason: refused outright rather than applied with an empty one.
    silent = _apply(tmp_path, [
        {"op": "recategorize", "entry": "星野灯", "category": "streamer",
         "to_category": "common"}
    ])
    assert silent["applied"] == []
    assert "reason" in silent["skipped"][0]["reason"]

    moved = _apply(tmp_path, [
        {"op": "recategorize", "entry": "星野灯", "category": "streamer",
         "to_category": "common", "reason": "这条其实不是主播"}
    ])
    assert [record["op"] for record in moved["applied"]] == ["recategorize"], moved

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "星野灯")
    assert subject.payload["category"] == "common"

    from finesub.llm.knowledge.node.render import render_subject

    full = render_subject(repo.store, subject.local_id, mode="human", preview="full")
    partial = render_subject(repo.store, subject.local_id, mode="human", preview="partial")
    assert "重分类理由" in full and "不是主播" in full
    assert "重分类理由" not in partial


def test_recategorize_parks_a_section_the_target_template_does_not_have(
    tmp_path: Path,
) -> None:
    """The move is where the target preset first applies, so it is where the
    mismatch has to surface — afterwards the entry just looks like it was
    always wrong.

    Surfacing it is not the same as refusing it, though. The seeded `common`
    entry keeps a `角色` section and `streamer` fixes its sections with no such
    name; the line goes to `streamer`'s staging section and the move lands.
    Refusing instead rolled back the whole ENVELOPE and told nobody who could
    act (owner 2026-09-07) — see the blast-radius test below.
    """

    _seed(tmp_path)
    report = _apply(tmp_path, [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "试着挪一下"}
    ])
    assert [r["op"] for r in report["applied"]] == ["recategorize"], report
    assert [(p["entry"], p["from_section"]) for p in report["parked"]] == [
        ("崩坏星穹铁道", "角色")
    ], report["parked"]

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    placed = {m.section for m in repo.store.children(subject.local_id)}
    assert "角色" not in placed and "待归类" in placed


def test_recategorize_still_refuses_when_the_target_cannot_park(
    tmp_path: Path, monkeypatch
) -> None:
    """Parking is the remedy; the shape check is still the invariant.

    A preset with no staging section has nowhere to put a homeless line, and
    then the only honest answer is to refuse the move rather than leave the
    line sitting in a section that forbids it.
    """

    from finesub.llm.knowledge.node.presets import Preset

    _seed(tmp_path)
    monkeypatch.setattr(Preset, "staging_section", lambda self: None)
    report = _apply(tmp_path, [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "试着挪一下"}
    ])
    assert report["applied"] == []
    assert "角色" in report["rollback_reason"]

# 2026-09-07, second review round: what the two move ops still let through.


def _raw_apply(root: Path, ops: list[dict], bindings: list):  # type: ignore[no-untyped-def]
    """Engine-level apply, for shapes the model vocabulary cannot express."""

    from finesub.llm.knowledge.node.apply import apply_envelope
    from finesub.llm.knowledge.node.envelope import Envelope

    repo = KnowledgeRepo.open(root)
    return apply_envelope(
        repo.store,
        Envelope(
            task_id="t", assignment_id="", context_epoch=0,
            knowledge_read_rev=repo.rev, ops=ops, handle_bindings=bindings,
        ),
    )


def _line_in(root: Path, surface: str, section: str) -> str:
    repo = KnowledgeRepo.open(root)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == surface)
    return next(
        m.child_id for m in repo.store.children(subject.local_id) if m.section == section
    )


def _handle_for(root: Path, surface: str, node_id: str) -> tuple[HandleMap, str]:
    repo = KnowledgeRepo.open(root)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == surface)
    handles = HandleMap()
    repo.entry_prompt_text(subject.local_id, handles)
    return handles, next(
        handle for handle, (ident, _) in handles.nodes.items() if ident == node_id
    )


def test_recategorize_parks_a_line_the_target_section_cannot_hold(
    tmp_path: Path, reported, monkeypatch
) -> None:
    """The section NAME fitting is not the same as the LINE fitting.

    A non-strict target accepts every section name, so a name-only check has
    nothing to report while the section can still forbid the line's shape.
    ⚠ The shipped presets stopped producing this case when `common`'s free
    sections were widened to take free sentences too (owner 2026-09-07), so it
    is driven through the preset seam — the branch is still exactly what the
    next category with term-only sections will hit.
    """

    import shutil

    from finesub.llm.knowledge.node import presets as presets_module

    folder = tmp_path / "presets"
    folder.mkdir()
    shipped = presets_module._preset_folder()
    for name in presets_module.CATEGORIES:
        shutil.copy(str(shipped / f"{name}.toml"), folder / f"{name}.toml")
    # `_seed` imports the legacy markdown tree, which loads the v1 presets from
    # a subfolder — the redirect has to bring those along.
    (folder / "legacy").mkdir()
    for name in presets_module.CATEGORIES:
        source = shipped / "legacy" / f"{name}.toml"
        try:
            shutil.copy(str(source), folder / "legacy" / f"{name}.toml")
        except (FileNotFoundError, OSError):
            pass
    narrowed = (folder / "common.toml").read_text(encoding="utf-8")
    narrowed = narrowed.replace('body_kinds = ["term", "note"]', 'body_kinds = ["term"]', 1)
    (folder / "common.toml").write_text(narrowed, encoding="utf-8")
    monkeypatch.setattr(presets_module, "_preset_folder", lambda: folder)
    presets_module.load_preset.cache_clear()
    try:
        _seed(tmp_path)
        report = _apply(tmp_path, [
            {"op": "recategorize", "entry": "星野灯", "category": "streamer",
             "to_category": "common", "reason": "这条其实不是主播"}
        ])
        assert [r["op"] for r in report["applied"]] == ["recategorize"], report
        assert "直播内容" in [p["from_section"] for p in report["parked"]], report["parked"]

        repo = KnowledgeRepo.open(tmp_path)
        subject = next(s for s in repo.subjects() if s.payload["surface"] == "星野灯")
        staged = [
            repo.store.node(m.child_id).payload.get("text")
            for m in repo.store.children(subject.local_id) if m.section == "待归类"
        ]
        assert "杂谈" in staged, staged
        # ...and it is announced. A silent restructuring is the one outcome no
        # front end should have to read a report to notice.
        assert "knowledge-recategorize-parked" in reported.codes()
    finally:
        presets_module.load_preset.cache_clear()


def test_one_unmovable_line_no_longer_drops_the_whole_batch(tmp_path: Path) -> None:
    """Why parking, and not just a nicer refusal message.

    The shape check reports at the ENVELOPE level, so a refused move used to
    take every unrelated proposal in the same chunk with it — and the reason
    reached nobody who could act on it: the knowledge-update round only warns
    a human, and `conflicted_entries` excludes a rolled-back report from the
    B' repair round on purpose (owner 2026-09-07).
    """

    _seed(tmp_path)
    report = _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "角色", "content": "丹恒|丹恒|Dan Heng|列车护卫。", "reason": "无关的一条"},
        {"op": "recategorize", "entry": "星野灯", "category": "streamer",
         "to_category": "common", "reason": "会被拒的那条"},
    ])
    assert not report["rolled_back"], report["rollback_reason"]
    assert {r["op"] for r in report["applied"]} == {"append_lines", "recategorize"}, report

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    surfaces = {
        repo.store.node(m.child_id).payload.get("surface")
        for m in repo.store.children(subject.local_id) if m.section == "角色"
    }
    assert "丹恒" in surfaces, surfaces


def test_recategorize_demotes_lines_the_target_would_not_share(tmp_path: Path) -> None:
    """The target preset's share policy governs the lines already there.

    `[人设]` is registered under `streamer` and inherits 档案's `inherit`;
    `common` does not register it at all, and its `unknown_label_share` is
    fail-closed `local`. A move that leaves the line `shareable` hands the next
    push a line the new category never agreed to share.
    """

    _seed(tmp_path)
    prep = _apply(tmp_path, [
        {"op": "append_lines", "entry": "星野灯", "category": "streamer",
         "section": "档案", "content": "[人设] 元气后辈", "reason": "r"},
    ])
    assert len(prep["applied"]) == 1, prep

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "星野灯")
    persona = next(
        m.child_id for m in repo.store.children(subject.local_id)
        if m.section == "档案"
        and (repo.store.node(m.child_id).payload.get("label") or "") == "人设"
    )
    with repo.store.begin("user", note="mark") as txn:
        txn.update_node(subject.local_id, visibility="shareable")
        txn.update_node(persona, visibility="shareable")

    moved = _apply(tmp_path, [
        {"op": "recategorize", "entry": "星野灯", "category": "streamer",
         "to_category": "common", "reason": "这条其实不是主播"}
    ])
    assert [r["op"] for r in moved["applied"]] == ["recategorize"], moved
    repo = KnowledgeRepo.open(tmp_path)
    assert repo.store.node(persona).visibility == "local", (
        "a label the target category does not register must not stay shareable"
    )


def test_move_line_refuses_a_line_that_sits_under_two_entries(tmp_path: Path) -> None:
    """One handle names the LINE, and a line can be placed under two entries.

    Taking the first membership moves whichever placement the store happens to
    return first — possibly the one the model was never shown.
    """

    _seed(tmp_path)
    from finesub.llm.knowledge.node.envelope import Binding

    repo = KnowledgeRepo.open(tmp_path)
    streamer = next(s for s in repo.subjects() if s.payload["surface"] == "星野灯")
    shared = _line_in(tmp_path, "崩坏星穹铁道", "角色")
    result = _raw_apply(
        tmp_path,
        [{"op": "add_membership", "id": "@line", "parent": "@streamer",
          "section": "人际关系"}],
        [Binding(handle="@line", kind="node", id=shared,
                 expected_valid_from_rev=repo.store.node(shared).valid_from_rev),
         Binding(handle="@streamer", kind="node", id=streamer.local_id,
                 expected_valid_from_rev=streamer.valid_from_rev)],
    )
    assert result.rev is not None, result.rollback_reason

    handles, line_handle = _handle_for(tmp_path, "崩坏星穹铁道", shared)
    text = ("<knowledge_proposals>\n" + json.dumps(
        {"op": "move_line", "id": line_handle, "section": "待归类", "reason": "r"},
        ensure_ascii=False,
    ) + "\n</knowledge_proposals>")
    repo = KnowledgeRepo.open(tmp_path)
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert report["applied"] == [], report
    assert "两个条目" in report["skipped"][0]["reason"], report

    # ...and it goes through once the proposal says which entry it means.
    text = ("<knowledge_proposals>\n" + json.dumps(
        {"op": "move_line", "id": line_handle, "section": "待归类",
         "entry": "崩坏星穹铁道", "reason": "r"},
        ensure_ascii=False,
    ) + "\n</knowledge_proposals>")
    repo = KnowledgeRepo.open(tmp_path)
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert [r["op"] for r in report["applied"]] == ["move_line"], report
    repo = KnowledgeRepo.open(tmp_path)
    placed = {
        m.parent_id: m.section for m in repo.store.parents(shared)
    }
    assert placed[streamer.local_id] == "人际关系", "the other entry must not move"


def test_a_recategorized_entry_still_round_trips_through_the_editor(tmp_path: Path) -> None:
    """The full preview is an EDITABLE face: whatever it renders must parse.

    The move reason renders there (§0 requirement 13) — and the editor handed
    that same text back rejected it as an unknown metadata line, so a moved
    entry could not be edited at all.
    """

    _seed(tmp_path)
    moved = _apply(tmp_path, [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "common", "reason": "x"}
    ])
    assert moved["applied"] == []  # unchanged category is refused; set it directly

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    payload = dict(subject.payload)
    payload["recategorized_reason"] = "本来放错了类别"
    with repo.store.begin("user", note="reason") as txn:
        txn.update_node(subject.local_id, payload=payload)

    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.render import render_subject

    repo = KnowledgeRepo.open(tmp_path)
    subject = repo.store.node(subject.local_id)
    text = render_subject(repo.store, subject.local_id, mode="human", preview="full")
    assert "重分类理由" in text
    report = edit_subject(repo, subject, text)  # must not raise
    assert not report.changed, report


def test_move_line_refuses_a_section_that_will_not_take_the_line(tmp_path: Path) -> None:
    """归位 out of the staging section is the move parking makes routine, and
    the section it goes back into is exactly the one that may not take its
    shape.

    Per OP, not per envelope: letting it through means `validate_overlay`
    catches it instead, and that rolls the whole chunk back under a message
    naming an internal membership id (owner 2026-09-07).
    """

    _seed(tmp_path)
    term = _line_in(tmp_path, "崩坏星穹铁道", "角色")
    handles, handle = _handle_for(tmp_path, "崩坏星穹铁道", term)

    # `档案` takes free sentences only, so a TERM is what it cannot hold. The
    # batch also carries an unrelated line, which has to survive.
    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "角色", "content": "彦卿|彦卿|Yanqing|云骑军将领。", "reason": "无关的一条"},
        {"op": "move_line", "id": handle, "section": "档案", "reason": "归位"},
    ]) + "\n</knowledge_proposals>")
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert not report["rolled_back"], report["rollback_reason"]
    assert {r["op"] for r in report["applied"]} == {"append_lines"}, report
    assert "只放自由句" in report["skipped"][0]["reason"], report["skipped"]

    # ...and a free section does take it, now that those hold both shapes.
    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + json.dumps(
        {"op": "move_line", "id": handle, "section": "地区", "reason": "归位"},
        ensure_ascii=False) + "\n</knowledge_proposals>")
    ok = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert [r["op"] for r in ok["applied"]] == ["move_line"], ok


def test_a_parked_line_loses_shareable_for_where_it_LANDED(tmp_path: Path) -> None:
    """The share policy resolves from the section a line sits in, so it has to
    be answered about the section the line ENDS in.

    `直播内容` inherits sharing; the staging section shares nothing. Demoting
    before the move asked the question about the section the line just left,
    and the answer was "keep sharing" (review 2026-09-07, third round).
    """

    _seed(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    line = _line_in(tmp_path, "崩坏星穹铁道", "角色")
    with repo.store.begin("user", note="mark") as txn:
        txn.update_node(subject.local_id, visibility="shareable")
        txn.update_node(line, visibility="shareable")

    # `角色` inherits sharing (`common`'s free sections are `"*" = "inherit"`);
    # `streamer` has no such section, so the line lands in the staging one,
    # which shares nothing.
    report = _apply(tmp_path, [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"}
    ])
    assert [r["op"] for r in report["applied"]] == ["recategorize"], report

    repo = KnowledgeRepo.open(tmp_path)
    placed = {m.child_id: m.section for m in repo.store.children(subject.local_id)}
    assert placed[line] == "待归类"
    assert repo.store.node(line).visibility == "local", (
        "the staging section shares nothing; a line parked into it must not "
        "still be a share candidate"
    )


def test_parking_reads_the_placement_this_batch_ends_with(tmp_path: Path) -> None:
    """An explicit move earlier in the batch is an instruction, not noise.

    Moving a line somewhere the target template CAN hold and then
    recategorizing is exactly what the contract tells a session to do. Judging
    against the section the store still records parked it anyway — an
    automatic move undoing an explicit one.
    """

    _seed(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    term = _line_in(tmp_path, "崩坏星穹铁道", "角色")
    handles, handle = _handle_for(tmp_path, "崩坏星穹铁道", term)

    text = ("<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in [
        # `人际关系` is a streamer section that takes term bodies
        {"op": "move_line", "id": handle, "section": "人际关系", "reason": "先安顿好"},
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"},
    ]) + "\n</knowledge_proposals>")
    repo = KnowledgeRepo.open(tmp_path)
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert [r["op"] for r in report["applied"]] == ["move_line", "recategorize"], report
    assert report["parked"] == [], report["parked"]

    repo = KnowledgeRepo.open(tmp_path)
    placed = {m.child_id: m.section for m in repo.store.children(subject.local_id)}
    assert placed[term] == "人际关系", placed


def test_a_line_added_in_the_same_batch_is_parked_too(tmp_path: Path) -> None:
    """The batch's own new lines are judged by the shape check, so they have to
    reach the parking pass as well — otherwise appending one before the move
    is the one way left to make the move roll back."""

    _seed(tmp_path)
    report = _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "地区", "content": "贝洛伯格|贝洛伯格|Belobog|雅利洛的城市。", "reason": "r"},
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"},
    ])
    assert not report["rolled_back"], report["rollback_reason"]
    assert {p["from_section"] for p in report["parked"]} == {"角色", "地区"}, (
        report["parked"]
    )


def test_a_later_proposal_sees_the_category_this_batch_moved_the_entry_INTO(
    tmp_path: Path,
) -> None:
    """`recategorize` does not take effect until the ops are folded, so every
    proposal after it resolved the entry from the STORE and was judged against
    the category it is leaving (review 2026-09-07, fourth round).

    Both directions were wrong, and they fail differently: judged too strictly
    it is refused per op, judged too loosely it passes translation and dies in
    validation, which rolls the whole batch back.
    """

    _seed(tmp_path)
    # A common entry with a free sentence parked in its staging section.
    _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "待归类", "content": "开播时常提到这部作品。", "reason": "r"},
    ])
    parked = _line_in(tmp_path, "崩坏星穹铁道", "待归类")
    handles, handle = _handle_for(tmp_path, "崩坏星穹铁道", parked)

    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"},
        # `直播内容` exists only in streamer, and takes free sentences
        {"op": "move_line", "id": handle, "section": "直播内容", "reason": "归位"},
    ]) + "\n</knowledge_proposals>")
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert [r["op"] for r in report["applied"]] == ["recategorize", "move_line"], report

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    placed = {m.child_id: m.section for m in repo.store.children(subject.local_id)}
    assert placed[parked] == "直播内容", placed


def test_an_append_after_a_recategorize_is_judged_by_the_new_template(
    tmp_path: Path,
) -> None:
    """The other direction: judged by the category being LEFT, an append that
    the target forbids passes translation and only fails in `validate_overlay`
    — which reports per envelope, so the whole batch rolls back."""

    _seed(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"},
        # a freely named section is legal under `common` and does not exist
        # under `streamer`, whose six are fixed
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "地区", "content": "贝洛伯格|贝洛伯格|Belobog|雅利洛的城市。", "reason": "r"},
    ]) + "\n</knowledge_proposals>")
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=HandleMap()
    ).to_dict()
    assert not report["rolled_back"], report["rollback_reason"]
    assert [r["op"] for r in report["applied"]] == ["recategorize"], report
    assert report["skipped"], report


def test_a_nickname_left_in_the_alias_column_injects_two_renderings(tmp_path: Path) -> None:
    """Why "有稳定独立译法的叫法提升为独立术语行" comes with "并从别名列摘掉".

    The match index covers term surfaces AND alias items, and buckets hits by
    node, so a nickname that is both its own term and an alias of the formal
    name matches twice — and each hit injects its own line, carrying its own
    中文定名. The model then gets two answers to one occurrence.

    Also pins the other half of the rule: a term hit injects `format_line` of
    the matched node and nothing else, so a condition has to live in that
    line's own fourth column (owner 2026-09-07).
    """

    from finesub.llm.knowledge.base import match_terms

    _seed(tmp_path)
    _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common", "section": "角色",
         "content": "ゆきの|雪乃|ユキノ、ゆっきー|人物正式名。", "reason": "昵称还挂在别名列"},
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common", "section": "角色",
         "content": "ゆっきー|小雪|ユッキー|雪乃的昵称；原文用此称呼时译作「小雪」。",
         "reason": "拆出来的第二条"},
    ])

    lines = [m.line for m in match_terms(tmp_path, "今日はゆっきーと一緒に配信します")]
    assert len(lines) == 2, lines
    assert any(line.startswith("ゆっきー|小雪|") for line in lines)
    assert any(line.startswith("ゆきの|雪乃|") for line in lines), (
        "one occurrence, two lines, two different 中文定名 — this is the state "
        "the discipline exists to prevent"
    )

    # The remedy is an ordinary `update`: the alias column syncs both ways.
    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    formal = next(
        m.child_id for m in repo.store.children(subject.local_id)
        if (repo.store.node(m.child_id).payload or {}).get("surface") == "ゆきの"
    )
    handles, handle = _handle_for(tmp_path, "崩坏星穹铁道", formal)
    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + json.dumps(
        {"op": "update", "id": handle, "line": "ゆきの|雪乃|ユキノ|人物正式名。",
         "reason": "昵称已独立成行"}, ensure_ascii=False) + "\n</knowledge_proposals>")
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert "alias removed" in report["applied"][0]["reason"], report

    lines = [m.line for m in match_terms(tmp_path, "今日はゆっきーと一緒に配信します")]
    assert len(lines) == 1 and lines[0].startswith("ゆっきー|小雪|"), lines


def test_update_reads_the_category_and_section_this_batch_gave_the_line(
    tmp_path: Path,
) -> None:
    """The third entry point with the same hole.

    `recategorize` and `move_line` both land later, at fold time, so an
    `update` after them resolved the line's owning category AND its section
    from the store — and the section is what picks the spec the new text is
    classified against. Judged under `common`'s staging section, a free
    sentence carrying two pipes was read as a malformed term row; under
    `streamer`'s `直播内容`, which takes free sentences only, a pipe is just a
    character (review 2026-09-07, fifth round).
    """

    _seed(tmp_path)
    _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "待归类", "content": "开播时常提到这部作品。", "reason": "r"},
    ])
    line = _line_in(tmp_path, "崩坏星穹铁道", "待归类")
    handles, handle = _handle_for(tmp_path, "崩坏星穹铁道", line)

    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"},
        {"op": "move_line", "id": handle, "section": "直播内容", "reason": "归位"},
        {"op": "update", "id": handle,
         "line": "常玩的几款：原神 | 星铁 | 绝区零。", "reason": "补全"},
    ]) + "\n</knowledge_proposals>")
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert not report["rolled_back"], report["rollback_reason"]
    assert [r["op"] for r in report["applied"]] == [
        "recategorize", "move_line", "update"
    ], report

    repo = KnowledgeRepo.open(tmp_path)
    assert repo.store.node(line).payload["text"] == "常玩的几款：原神 | 星铁 | 绝区零。"


def test_prose_with_pipes_is_read_as_a_term_in_a_mixed_section(tmp_path: Path) -> None:
    """A known limit, pinned so it is not rediscovered as a bug.

    The v3 grammar infers a line's kind from its shape, and it scoped the pipe
    rule to term-accepting sections precisely because subtitle lines join
    speakers with " | ". A section that takes BOTH shapes cannot have it both
    ways — so widening `common`'s free sections to hold free sentences
    (owner 2026-09-07) moved that ambiguity into the sections prose now lives
    in. Three segments is refused, four or more is stored as a term and enters
    the match index.

    The mitigation is in the contract (write 、 or / instead of |); the real
    fix, if this ever bites, is an explicit marker for prose in mixed sections,
    which is a grammar change.
    """

    _seed(tmp_path)
    refused = _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "角色", "content": "常玩的几款：原神 | 星铁 | 绝区零。", "reason": "r"},
    ])
    assert refused["applied"] == [] and "四列" in refused["skipped"][0]["reason"]

    _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "角色", "content": "同台的四人：A | B | C | D。", "reason": "r"},
    ])
    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    stored = {
        repo.store.node(m.child_id).kind: repo.store.node(m.child_id).payload
        for m in repo.store.children(subject.local_id)
        if m.section == "角色" and "同台" in str(repo.store.node(m.child_id).payload)
    }
    assert "term" in stored, stored
    assert stored["term"]["surface"].startswith("同台的四人"), stored

    # ...and the same sentence written without pipes is prose, as intended.
    _apply(tmp_path, [
        {"op": "append_lines", "entry": "崩坏星穹铁道", "category": "common",
         "section": "角色", "content": "同台的另外四人：A、B、C、D。", "reason": "r"},
    ])
    repo = KnowledgeRepo.open(tmp_path)
    kinds = {
        repo.store.node(m.child_id).kind
        for m in repo.store.children(subject.local_id)
        if m.section == "角色" and "另外四人" in str(repo.store.node(m.child_id).payload)
    }
    assert kinds == {"note"}, kinds


def test_update_after_a_recategorize_follows_the_line_into_the_staging_section(
    tmp_path: Path,
) -> None:
    """The automatic move has to be as visible to the batch as the explicit one.

    Parking happens at fold time inside the engine, so the translation layer
    went on judging a later `update` against the section the line was being
    moved OUT of — and the sequence the contract teaches (recategorize, then
    fix the line) was the one shape that failed, while the identical edit in a
    second batch worked (review 2026-09-07, sixth round).
    """

    _seed(tmp_path)
    term = _line_in(tmp_path, "崩坏星穹铁道", "角色")
    handles, handle = _handle_for(tmp_path, "崩坏星穹铁道", term)

    repo = KnowledgeRepo.open(tmp_path)
    text = ("<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in [
        {"op": "recategorize", "entry": "崩坏星穹铁道", "category": "common",
         "to_category": "streamer", "reason": "r"},
        {"op": "update", "id": handle,
         "line": "三月七|三月七|March 7th|开朗少女，列车组成员。", "reason": "补描述"},
    ]) + "\n</knowledge_proposals>")
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    ).to_dict()
    assert not report["rolled_back"], report["rollback_reason"]
    assert [r["op"] for r in report["applied"]] == ["recategorize", "update"], report
    assert [p["from_section"] for p in report["parked"]] == ["角色"], report["parked"]

    repo = KnowledgeRepo.open(tmp_path)
    subject = next(s for s in repo.subjects() if s.payload["surface"] == "崩坏星穹铁道")
    placed = {m.child_id: m.section for m in repo.store.children(subject.local_id)}
    assert placed[term] == "待归类"
    assert repo.store.node(term).payload["desc"] == "开朗少女，列车组成员。"
