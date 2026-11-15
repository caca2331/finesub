"""Splitting a section off into a sub-entry, and moving a whole section
(plan kb-entry-size, step 3)."""

from __future__ import annotations

import json

from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.subentries import children_of, parent_of


def _repo(tmp_path, *, visibility: str = "local") -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    KnowledgeRepo.forget(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("G", "subject", {"surface": "原神", "intro": "开放世界游戏", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案", "角色", "地区"]},
                        visibility=visibility)
        txn.create_node("GI", "note", {"label": "本名", "text": "原神"}, visibility=visibility)
        txn.create_membership("GM", "G", "GI", "档案", 0)
        for i, (surface, zh) in enumerate((("アリス", "爱丽丝"), ("ボブ", "鲍勃"))):
            txn.create_node(f"C{i}", "term", {"surface": surface, "zh": zh, "desc": "角色"}, visibility=visibility)
            txn.create_membership(f"CM{i}", "G", f"C{i}", "角色", i)
        txn.create_node("R0", "term", {"surface": "モンド", "zh": "蒙德", "desc": "地区"}, visibility=visibility)
        txn.create_membership("RM0", "G", "R0", "地区", 0)
        txn.create_node("Y", "subject", {"surface": "游戏Y", "intro": "另一个游戏", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案", "角色"]})
        txn.create_node("V", "subject", {"surface": "某主播", "intro": "一位主播", "category": "streamer",
                                         "section_order": ["档案", "频道用语"]})
        txn.create_node("VT", "term", {"surface": "おはよ", "zh": "早", "desc": "问候"})
        txn.create_membership("VM", "V", "VT", "频道用语", 0)
    return repo


def _apply(repo: KnowledgeRepo, *proposals: dict, allow=None):  # type: ignore[no-untyped-def]
    kwargs = {"allow_categories": allow} if allow else {}
    return apply_model_proposals(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=HandleMap(), **kwargs,
    )


def _split(**extra) -> dict:  # type: ignore[no-untyped-def]
    return {"op": "split_entry", "entry": "原神", "section": "角色", "intro": "《原神》的登场角色",
            "reason": "角色节偏长", **extra}


def _child(repo: KnowledgeRepo) -> str:
    resolved = repo.resolve("原神·角色")
    assert resolved is not None
    return resolved.subject_id


def test_a_section_becomes_a_sub_entry_and_its_lines_keep_their_nodes(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, _split())
    assert report.rev is not None and not report.skipped, [r.reason for r in report.skipped]
    child = _child(repo)
    store = repo.store
    assert sorted(m.child_id for m in store.children(child) if m.section == "角色") == ["C0", "C1"]
    assert parent_of(store, child) == "G" and children_of(store, "G") == [child]
    node = store.node(child)
    assert node.payload["section_order"] == ["档案", "角色"] and node.payload["entry_type"] == "游戏"
    assert node.payload["intro"] == "《原神》的登场角色"
    assert "角色" not in store.node("G").payload["section_order"]
    identity = [store.node(m.child_id) for m in store.children(child) if m.section == "档案"]
    assert [n.payload["text"] for n in identity] == ["原神·角色"]


def test_the_identity_line_of_the_sub_entry_is_in_the_batch_s_label_ledger(tmp_path) -> None:
    """As for `create_entry`: a `[本名]` written into the new sub-entry's 档案 in the same batch is refused per op,
    not left to fail the whole batch as a duplicate label."""

    from finesub.llm.knowledge.update import rejected_entries

    repo = _repo(tmp_path)
    report = _apply(repo, _split(), {"op": "append_lines", "entry": "原神·角色", "category": "common",
                                     "section": "档案", "content": "[本名] 原神·角色", "reason": "r"})
    assert not report.rolled_back and report.rev is not None
    assert [(r.op, r.reason) for r in report.skipped] == [("append_lines", "标记 [本名] 在本节已存在——改用 update")]
    assert rejected_entries(report.to_dict()) == [("common", "原神·角色")]


def test_a_later_row_of_the_batch_appends_to_the_new_sub_entry_by_name(tmp_path) -> None:
    """The split-op note: in the same batch, a row after the split names the
    sub-entry by its derived name and the parent's category (cold read P081)."""

    from finesub.llm.prompt_compose import render_knowledge_ops_contract

    assert "同批先拆后追加时，`entry` 写派生的子条目名" in render_knowledge_ops_contract(("common",), split=True)
    repo = _repo(tmp_path)
    report = _apply(repo, _split(), {"op": "append_lines", "entry": "原神·角色", "category": "common",
                                     "section": "角色", "content": "セス|赛斯||新登场角色", "reason": "r"})
    assert report.rev is not None and not report.skipped, [r.reason for r in report.skipped]
    child = _child(repo)
    surfaces = {repo.store.node(m.child_id).payload.get("surface")
                for m in repo.store.children(child) if m.section == "角色"}
    assert surfaces == {"アリス", "ボブ", "セス"}


def test_a_shared_entry_splits_into_a_shared_sub_entry(tmp_path) -> None:
    repo = _repo(tmp_path, visibility="shareable")
    _apply(repo, _split())
    child = _child(repo)
    assert repo.store.node(child).visibility == "shareable"
    assert repo.store.node("C0").visibility == "shareable"


def test_both_ends_say_so_and_the_human_projection_round_trips(tmp_path) -> None:
    from finesub.llm.knowledge.node.edit import edit_subject

    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    parent_prompt = render_subject(repo.store, "G", mode="prompt")
    assert "子条目：原神·角色" in parent_prompt.split("\n## ", 1)[0]
    assert "属于：原神" in render_subject(repo.store, child, mode="prompt")
    human = repo.entry_text("G")
    assert "<!-- 子条目：原神·角色 -->" in human
    assert not edit_subject(repo, repo.store.node("G"), human, refresh=False).changed
    assert not edit_subject(repo, repo.store.node(child), repo.entry_text(child), refresh=False).changed


def test_a_sub_entry_asks_only_for_its_identity_slot(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    full = render_subject(repo.store, _child(repo), mode="prompt", preview="full")
    slots = [line for line in full.splitlines() if line.startswith("[")]  # comments are indented
    assert not any(line.startswith(("[别名]", "[收录范围]")) for line in slots), slots
    assert "## 地区" not in full


def test_sub_entries_follow_their_parent_in_the_index(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    rows = [line for line in repo.index_text("common").splitlines() if line.startswith("- ")]
    names = [row[2:].split(" ", 1)[0] for row in rows]
    assert names.index("原神·角色") == names.index("原神") + 1


def test_refusals_are_per_op(tmp_path) -> None:
    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("Z", "subject", {"surface": "原神·地区", "intro": "撞名", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案"]})
    report = _apply(
        repo,
        {"op": "split_entry", "entry": "某主播", "section": "频道用语", "intro": "x", "reason": "r"},
        _split(section="档案"),
        _split(section="不存在"),
        _split(section="地区"),  # 原神·地区 is taken
        _split(intro="第一行\n第二行"),
        _split(),  # still lands
        allow=("common", "streamer", "style"),
    )
    reasons = [r.reason for r in report.skipped]
    assert any("不能拆分" in r for r in reasons) and any("条目头" in r for r in reasons)
    assert any("不存在或没有行" in r for r in reasons) and any("已有名为" in r for r in reasons)
    assert any("intro must be one line" in r for r in reasons)
    assert report.rev is not None and repo.resolve("原神·角色") is not None


def test_a_sub_entry_is_not_split_again(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    report = _apply(repo, {"op": "split_entry", "entry": "原神·角色", "section": "角色", "intro": "x", "reason": "r"})
    assert any("子条目不能再拆" in r.reason for r in report.skipped)


def test_a_whole_section_moves_to_another_entry(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, {"op": "move_line", "entry": "原神", "from_section": "角色", "to_entry": "游戏Y",
                           "reason": "其实是游戏Y的角色"})
    assert report.rev is not None and not report.skipped
    assert sorted(m.child_id for m in repo.store.children("Y") if m.section == "角色") == ["C0", "C1"]


def test_create_then_move_a_section_in_one_batch_and_same_entry_is_refused(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(
        repo,
        {"op": "create_entry", "category": "common", "entry": "原神2", "entry_type": "游戏",
         "intro": "续作", "reason": "地区属于续作"},
        {"op": "move_line", "entry": "原神", "from_section": "地区", "to_entry": "原神2", "reason": "r"},
        {"op": "move_line", "entry": "原神", "from_section": "角色", "to_entry": "原神", "reason": "r"},
    )
    assert report.rev is not None
    assert [m.parent_id for m in repo.store.parents("R0")] == [repo.resolve("原神2").subject_id]
    assert any("另一个条目" in r.reason for r in report.skipped)


def test_renaming_the_parent_renames_derived_sub_entries(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    report = _apply(repo, {"op": "rename_entry", "entry": "原神", "new_key": "原神正式名", "reason": "r"})
    assert report.rev is not None and not report.skipped
    assert repo.store.node(child).payload["surface"] == "原神正式名·角色"
    identity = [repo.store.node(m.child_id) for m in repo.store.children(child) if m.section == "档案"]
    assert identity[0].payload["text"] == "原神正式名·角色"


def test_parents_and_sub_entries_keep_their_category(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    report = _apply(
        repo,
        {"op": "recategorize", "entry": "原神", "to_category": "streamer", "reason": "r"},
        {"op": "recategorize", "entry": "原神·角色", "to_category": "streamer", "reason": "r"},
    )
    assert sum("不能改类别" in r.reason for r in report.skipped) == 2


def test_restoring_the_parent_takes_back_only_a_live_sub_entry(tmp_path) -> None:
    from finesub.llm.knowledge.node.history import restore_node

    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    _apply(repo, {"op": "retire_entry", "entry": "原神", "merged_into": "游戏Y", "reason": "r"})
    assert parent_of(repo.store, child) is None
    restore_node(repo.store, "G")
    assert parent_of(repo.store, child) == "G"
    # retired child: the link stays closed
    _apply(repo, {"op": "retire_entry", "entry": "原神", "merged_into": "游戏Y", "reason": "r"})
    with repo.store.begin("user") as txn:
        txn.tombstone_node(child)
    restore_node(repo.store, "G")
    assert children_of(repo.store, "G") == []


def test_a_hint_naming_a_moved_line_scores_the_sub_entry(tmp_path) -> None:
    from finesub.llm.knowledge.entries import select_kb_entries
    from finesub.llm.knowledge.feedback import KnowledgeHint

    repo = _repo(tmp_path)
    _apply(repo, _split())
    picks = select_kb_entries(
        [KnowledgeHint(category="common", entry="原神", sub="アリス"),
         KnowledgeHint(category="common", entry="原神", sub="モンド")],
        knowledge_root=tmp_path,
    )
    assert {(p.category, p.key) for p in picks} == {("common", "原神·角色"), ("common", "原神")}


def test_the_bundle_carries_the_part_of_link(tmp_path) -> None:
    from finesub.llm.knowledge.share.exchange import build_push_bundle, strip_local_maps, validate_bundle

    repo = _repo(tmp_path, visibility="shareable")
    _apply(repo, _split())
    wire = strip_local_maps(build_push_bundle(repo.store, ["G", _child(repo)], idempotency_key="k"))
    assert validate_bundle(wire) == []
    assert [link["rel"] for link in wire["links"]] == ["part_of"]


def test_the_cli_splits_too(tmp_path) -> None:
    from finesub.llm.knowledge.maintain import main

    _repo(tmp_path)
    assert main(["--root", str(tmp_path), "split", "原神", "角色", "--intro", "《原神》的登场角色"]) == 0
    KnowledgeRepo.forget(tmp_path)
    assert KnowledgeRepo.open(tmp_path).resolve("原神·角色") is not None


def test_the_split_offer_appears_only_for_a_splittable_entry_asked_to_compress(tmp_path) -> None:
    from finesub.llm.knowledge.entry_size import offers_split, size_notice
    from finesub.llm.prompt_compose import render_knowledge_ops_contract

    assert "split_entry" not in render_knowledge_ops_contract(("common",))
    assert '"op":"split_entry"' in render_knowledge_ops_contract(("common",), split=True)
    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:
        for i in range(160):
            txn.create_node(f"X{i}", "term", {"surface": f"名字{i}", "zh": f"译名{i}", "desc": "很长的描述" * 12})
            txn.create_membership(f"XM{i}", "G", f"X{i}", "角色", 10 + i)
            txn.create_node(f"W{i}", "term", {"surface": f"词{i}", "zh": f"译{i}", "desc": "很长的描述" * 12})
            txn.create_membership(f"WM{i}", "V", f"W{i}", "频道用语", 10 + i)
    common = size_notice(repo.store, "G")
    assert offers_split(common) and "角色 ≈" in common and '"op":"split_entry"' not in common
    assert '"op":"split_entry"' in size_notice(repo.store, "G", with_contract=True)
    streamer = size_notice(repo.store, "V")
    assert "至少删减" in streamer and "split_entry" not in streamer


def test_a_sub_entry_renamed_away_from_its_derived_name_is_left_alone(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    _apply(repo, {"op": "rename_entry", "entry": "原神·角色", "new_key": "原神角色表", "reason": "r"})
    _apply(repo, {"op": "rename_entry", "entry": "原神", "new_key": "原神正式名", "reason": "r"})
    assert repo.store.node(child).payload["surface"] == "原神角色表"


def test_two_splits_of_one_entry_in_one_batch_drop_both_sections(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, _split(), _split(section="地区", intro="《原神》的地区"))
    assert report.rev is not None and not report.skipped
    order = repo.store.node("G").payload["section_order"]
    assert "角色" not in order and "地区" not in order


def test_a_section_moved_away_whole_leaves_the_source_order(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, {"op": "move_line", "entry": "原神", "from_section": "地区", "to_entry": "游戏Y", "reason": "r"})
    assert "地区" not in repo.store.node("G").payload["section_order"]


def test_split_then_recategorize_in_one_batch_is_refused(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, _split(), {"op": "recategorize", "entry": "原神", "to_category": "streamer", "reason": "r"})
    assert any("不能改类别" in r.reason for r in report.skipped)
    assert repo.store.node("G").payload["category"] == "common"


def test_the_rename_cascade_carries_the_native_name_too(tmp_path) -> None:
    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    with repo.store.begin("user") as txn:
        node = repo.store.node(child)
        txn.update_node(child, payload={**node.payload, "native_names": ["原神·角色"]})
    _apply(repo, {"op": "rename_entry", "entry": "原神", "new_key": "原神正式名", "reason": "r"})
    assert repo.store.node(child).payload["native_names"] == ["原神正式名·角色"]


def test_one_entry_named_by_handle_and_by_name_still_drops_both_sections(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles = HandleMap()
    render_subject(repo.store, "G", mode="prompt", handles=handles)
    h1 = next(handle for handle, bound in handles.nodes.items() if bound[0] == "G")
    report = apply_model_proposals(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in (
            _split(entry=h1), _split(section="地区", intro="《原神》的地区"),
        )),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    assert report.rev is not None and not report.skipped
    order = repo.store.node("G").payload["section_order"]
    assert "角色" not in order and "地区" not in order


def test_a_sub_entry_created_in_the_batch_is_not_recategorized_in_it(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, _split(), {"op": "recategorize", "entry": "原神·角色", "to_category": "streamer", "reason": "r"})
    assert any("不能改类别" in r.reason for r in report.skipped)


def test_a_human_renaming_the_parent_h1_renames_the_sub_entry(tmp_path) -> None:
    """The reason the cascade lives in the engine: a rendered/ H1 edit never
    goes through `rename_entry`."""

    from finesub.llm.knowledge.node.edit import edit_subject

    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    text = repo.entry_text("G").replace("# 原神\n", "# 原神改\n", 1)
    assert edit_subject(repo, repo.store.node("G"), text, refresh=False).changed
    assert repo.store.node(child).payload["surface"] == "原神改·角色"


def test_a_cascade_collision_keeps_the_old_name_and_says_so(tmp_path) -> None:
    from finesub.reporting import reporting_to

    repo = _repo(tmp_path)
    _apply(repo, _split())
    child = _child(repo)
    with repo.store.begin("import") as txn:
        txn.create_node("Q", "subject", {"surface": "原神改·角色", "intro": "撞名", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案"]})
    warnings: list[str] = []

    class _Collect:
        def warning(self, code, message, **_kwargs):  # type: ignore[no-untyped-def]
            warnings.append(code)

        def __getattr__(self, _name):  # type: ignore[no-untyped-def]
            return lambda *a, **k: None

    with reporting_to(_Collect()):
        _apply(repo, {"op": "rename_entry", "entry": "原神", "new_key": "原神改", "reason": "r"})
    assert repo.store.node(child).payload["surface"] == "原神·角色"
    assert "knowledge-subentry-rename-kept" in warnings


def test_exposure_matches_exactly_the_lines_that_went_in(tmp_path) -> None:
    from finesub.llm.injection_budget import render_knowledge_entries_block
    from finesub.llm.knowledge.node.signals import injected_node_ids

    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:
        for i in range(60):
            txn.create_node(f"X{i}", "term", {"surface": f"名字{i:02d}", "zh": f"译名{i}", "desc": "描述" * 10})
            txn.create_membership(f"XM{i}", "G", f"X{i}", "角色", 10 + i)
    text = repo.entry_injection_text("G")
    block = render_knowledge_entries_block({"原神": text}, count_tokens=len, heuristic_count=len,
                                           block_limit=1500, entry_limit=None)
    assert block.truncated == ("原神",)
    shown = set(injected_node_ids(repo.store, "G", repo.rev, block.kept["原神"]))
    in_text = {nid for nid in ("C0", "C1", "R0", *(f"X{i}" for i in range(60)))
               if repo.store.node(nid).payload["surface"] in block.text}
    assert shown - {"G", "GI"} == in_text


def test_a_split_carries_the_human_version_mark_with_its_origin(tmp_path) -> None:
    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.human_marks import EDITED_LABEL

    repo = _repo(tmp_path)
    human = repo.entry_text("G").replace("アリス|爱丽丝||角色", "アリス|爱丽丝||女主角", 1)
    assert edit_subject(repo, repo.store.node("G"), human, refresh=False).changed
    handles = HandleMap()
    render_subject(repo.store, "G", mode="prompt", handles=handles)
    h = next(handle for handle, bound in handles.nodes.items() if bound[0] == "C0")
    apply_model_proposals(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in (
            {"op": "update", "id": h, "line": "アリス|爱丽丝||模型改写的描述", "reason": "r"}, _split(),
        )),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    marks = [line for line in repo.entry_text(_child(repo)).splitlines() if line.startswith(f"<!-- {EDITED_LABEL}")]
    assert len(marks) == 1 and "迁自「原神」" in marks[0] and "女主角" in marks[0]


def test_the_identity_section_cannot_be_moved_away_whole(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, {"op": "move_line", "entry": "原神", "from_section": "档案", "to_entry": "游戏Y", "reason": "r"})
    assert any("条目头" in r.reason for r in report.skipped)


def test_rename_then_split_in_one_batch_names_the_sub_entry_after_the_new_name(tmp_path) -> None:
    repo = _repo(tmp_path)
    report = _apply(repo, {"op": "rename_entry", "entry": "原神", "new_key": "原神改", "reason": "r"}, _split())
    assert report.rev is not None and not report.skipped
    assert repo.resolve("原神改·角色") is not None
