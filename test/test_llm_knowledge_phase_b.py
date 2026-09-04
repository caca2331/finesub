"""Phase B (kb-line-grammar plan §8) and the entry-structure changes it makes:
v1 archive shape -> v3 grammar, items-only surface forms, misheard as a
matcher-only index, sparseness as absence, relation bodies becoming terms.
The judgement half it leaves behind lives in `scan.py`."""

from __future__ import annotations

from finesub.llm.knowledge.node.importer import strip_misheard_prose
from finesub.llm.knowledge.node.matching import ExactIndex
from finesub.llm.knowledge.node.presets import load_preset
from finesub.llm.knowledge.node.render import render_subject
from finesub.llm.knowledge.node import phase_b
from finesub.llm.knowledge.node.scan import scan_candidates
from finesub.llm.knowledge.node.store import KnowledgeStore

import pytest


def _seed_legacy(store: KnowledgeStore) -> None:
    """A store shaped like the lossless import left it: verbatim residue on."""

    with store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "原神", "intro": "游戏", "category": "common",
             "section_order": ["角色"], "aliases": ["げんしん", "空月之歌"], "preset": "common"},
        )
        txn.create_item("SA1", "S", "aliases", "げんしん")
        txn.create_item("SA2", "S", "aliases", "空月之歌")
        txn.create_node(
            "T1", "term",
            {"surface": "散兵", "zh": "散兵", "alias_text": "国崩、流浪者", "reading": "さんぺい",
             "desc": "愚人众前第六席。日语口语常被 ASR 误听为「残表」。"},
        )
        txn.create_item("I1", "T1", "aliases", "国崩")
        txn.create_item("IP", "T1", "aliases", "—")  # legacy english-column placeholder
        txn.create_node(
            "T2", "term",
            {"surface": "ラウマ", "zh": "菈乌玛", "alias_text": "", "reading": "らうま",
             "desc": "登场角色，Ver.5.0 新皮肤相关。", "sep": ": "},
        )
        txn.create_node(
            "T3", "term",
            {"surface": "空月之歌", "zh": "空月之歌", "alias_text": "", "reading": "", "desc": "篇章"},
        )
        txn.create_node("F1", "fact", {"field": "口癖", "sep": ": ", "value": ""})
        txn.create_node("F2", "fact", {"field": "其他", "sep": ": ", "value": "所属 个人势；出道 2022-12"})
        txn.create_node("F3", "fact", {"field": "生日", "sep": "：", "value": "1月1日"})
        txn.create_node("R1", "relation", {"target": "パパベルト", "sep": " | ", "description": "父亲"})
        for order, node_id in enumerate(("T1", "T2", "T3", "F1", "F2", "F3", "R1")):
            txn.create_membership(f"M{order}", "S", node_id, "角色", order)


def test_phase_b_converts_the_v1_archive_shape(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed_legacy(store)
        plan = phase_b.build_plan(store)
        actions = {(row.action, row.old_line) for row in plan.rows}
        assert ("drop", "口癖: ") in actions            # empty value: sparse = absent
        assert any(row.action == "split" for row in plan.rows)  # 其他 grab-bag
        phase_b.execute_plan(store, plan)

        t1 = store.node("T1")
        # reading さんぺい is informative (fold != surface): became an alias item
        assert "reading" not in t1.payload and "alias_text" not in t1.payload
        values = {(i.field, i.value) for i in store.items_of("T1")}
        assert ("aliases", "さんぺい") in values and ("aliases", "流浪者") in values
        assert ("misheard", "残表") in values
        assert ("aliases", "—") not in values  # placeholder retired, not a name
        assert "误听" not in t1.payload["desc"]
        # redundant reading らうま (fold == surface) just vanishes
        t2 = store.node("T2")
        assert "reading" not in t2.payload
        assert not any(i.value == "らうま" for i in store.items_of("T2"))
        # empty fact retired, its membership closed
        assert store.node("F1") is None
        # the relation became a term body; the fact became a labelled note
        assert store.node("R1").kind == "term"
        assert store.node("R1").payload["surface"] == "パパベルト"
        assert store.node("F3").kind == "note"
        assert store.node("F3").payload == {"text": "1月1日", "label": "生日"}
        # sep carried the v1 separator verbatim; v3 has no separator at all
        assert "sep" not in store.node("R1").payload
        assert "sep" not in store.node("F3").payload


def test_render_after_phase_b(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed_legacy(store)
        phase_b.execute_plan(store, phase_b.build_plan(store))
        text = render_subject(store, "S", mode="human")
    assert "散兵|散兵|国崩、流浪者、さんぺい|愚人众前第六席。" in text
    assert "口癖" not in text and "残表" not in text  # misheard is matcher-only


def test_relation_targets_join_the_match_corpus(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed_legacy(store)
        with store.begin("import") as txn:
            txn.create_item("RM1", "R1", "misheard", "パパベルド")
        index = ExactIndex.build(store)
        by_kind = {(m.key.kind, m.key.node_id) for m in index.scan("パパベルト来了 パパベルド")}
    assert ("surface", "R1") in by_kind  # the relation target itself
    assert ("misheard", "R1") in by_kind  # and its misheard item


def test_streamer_preset_has_channel_terms_section() -> None:
    preset = load_preset("streamer")
    spec = preset.section("频道用语")
    assert spec is not None and spec.body_kinds == ("term",)
    assert [label.name for label in spec.core_labels()] == ["自称"]


def test_strict_validation_rejects_empty_lines_and_legacy_kinds(tmp_path) -> None:
    from finesub.llm.knowledge.node.model import validate_payload

    # an empty slot is a rendering, never a node
    with pytest.raises(ValueError, match="non-empty"):
        validate_payload("note", {"text": " ", "label": "口癖"})
    # legacy kinds still LOAD (Phase A wrote them) but can never be written
    validate_payload("fact", {"field": "口癖", "value": " "}, strict=False)
    with pytest.raises(ValueError, match="legacy import kind"):
        validate_payload("fact", {"field": "口癖", "value": "x"})


def test_strip_misheard_prose_forms() -> None:
    assert strip_misheard_prose("身份句。常被 ASR 误听为「残表」。") == "身份句。"
    assert strip_misheard_prose("主角（误听:「アリズ」）") == "主角"
    assert strip_misheard_prose("冷静青年。误听: 单横") == "冷静青年。"
    assert strip_misheard_prose("没有标记的句子。") == "没有标记的句子。"
    # a 误听 mention without extractable variants stays put
    assert strip_misheard_prose("这个词容易被误听。") == "这个词容易被误听。"


def test_phase_b_dedupes_aliases_within_the_batch(tmp_path) -> None:
    """Round 10: the reading repeated in the alias column (or a width/kana
    variant of it) must produce one alias item, not two."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node(
                "S", "subject",
                {"surface": "原神", "intro": "", "category": "common", "section_order": ["角色"]},
            )
            txn.create_node(
                "T", "term",
                {"surface": "月兆", "zh": "月兆", "alias_text": "げっちょう、ゲッチョウ",
                 "reading": "げっちょう", "desc": "系统名"},
            )
            txn.create_membership("M", "S", "T", "角色", 0)
        phase_b.execute_plan(store, phase_b.build_plan(store))
        values = [i.value for i in store.items_of("T") if i.field == "aliases"]
    assert values == ["げっちょう"]  # reading == column value == kana variant: ONE item


def test_scan_flags_duplicate_terms(tmp_path) -> None:
    """Two live terms with one folded surface under a subject (the real store
    has ロスカリファ twice with contradictory zh) become a repair candidate."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed_legacy(store)
        with store.begin("import") as txn:
            txn.create_node("T4", "term", {"surface": "ラウマ", "zh": "拉乌玛", "desc": "另一条"})
            txn.create_node("T5", "term", {"surface": "らうま", "zh": "菈乌玛", "desc": ""})
            txn.create_membership("M9", "S", "T4", "角色", 9)
            txn.create_membership("M10", "S", "T5", "角色", 10)
        dup = [c for c in scan_candidates(store).candidates if c["kind"] == "duplicate-term"]
    assert len(dup) == 1 and set(dup[0]["nodes"]) == {"T2", "T4", "T5"}

def test_duplicate_term_candidate_routes_by_explicit_subject_id(tmp_path) -> None:
    """Round 12: the dup lives under SA, but the shared node's parents[0] is
    SB — routing must use the candidate's explicit subject_id, not a guess."""

    from finesub.llm.knowledge.node.repair import repair_targets
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("SA", "subject", {"surface": "游戏甲", "intro": "x", "category": "common",
                                          "section_order": ["角色"]})
        txn.create_node("SB", "subject", {"surface": "游戏乙", "intro": "y", "category": "common",
                                          "section_order": ["角色"]})
        txn.create_node("T1", "term", {"surface": "ラウマ", "zh": "菈乌玛", "desc": ""})
        txn.create_node("T2", "term", {"surface": "らうま", "zh": "拉乌玛", "desc": ""})
        txn.create_membership("MB", "SB", "T1", "角色", 0)  # parents[0] of T1 is SB
        txn.create_membership("MA1", "SA", "T1", "角色", 0)
        txn.create_membership("MA2", "SA", "T2", "角色", 1)
    targets = repair_targets(repo)
    dup_owner = [sid for sid, cands in targets.items()
                 if any(c["kind"] == "duplicate-term" for c in cands)]
    assert dup_owner == ["SA"]


def test_phase_b_resets_section_order_to_the_current_preset(tmp_path) -> None:
    """The imported section_order pins the v1 skeleton, so sections the v3
    preset added (频道用语 / 特点 / 待归类) would never render an empty slot.
    Phase B resets the order to the preset's and drops payload residue."""

    from finesub.llm.knowledge.node import phase_b
    from finesub.llm.knowledge.node.render import render_subject

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node(
                "S", "subject",
                {"surface": "某主播", "intro": "x", "category": "streamer",
                 "reading": "残留",
                 "section_order": ["档案", "直播内容", "说话风格", "喜好 / 特点",
                                    "重要经历", "人际关系"]},
            )
        phase_b.execute_plan(store, phase_b.build_plan(store))
        payload = store.node("S").payload
        assert payload["section_order"] == [
            "档案", "直播内容", "频道用语", "特点", "人际关系", "待归类",
        ]
        assert "reading" not in payload
        text = render_subject(store, "S", mode="human", preview="full")
        assert "## 频道用语" in text and "## 待归类" in text
        # the full preview shows the core slots, and none of them is stored
        assert "- [本名]" in text and not store.children("S")


def test_append_lines_category_hint_falls_back_to_global_resolve(tmp_path) -> None:
    """Round: the 律にゃー loss — create_entry checks duplicates globally but
    append_lines resolved category-scoped, so a wrong category hint stranded
    the content. The hint now falls back to the entry wherever it lives, and
    section admission follows the REAL preset."""

    import json

    from finesub.llm.knowledge.node.proposals import apply_model_proposals
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "浅瀬みやこ", "intro": "x",
                                         "category": "streamer",
                                         "section_order": ["频道用语"]})
    text = "<knowledge_proposals>\n" + json.dumps(
        {"op": "append_lines", "category": "common", "entry": "浅瀬みやこ",
         "section": "频道用语", "content": "律にゃー|律喵||对听众的统称", "reason": "r"},
        ensure_ascii=False) + "\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t",
                                   knowledge_read_rev=repo.rev).to_dict()
    assert not report["skipped"] and report["applied"]
    terms = [n for n in repo.store.nodes_of_kind("term")]
    assert terms and terms[0].payload["surface"] == "律にゃー"


def test_a_grab_bag_row_that_repeats_a_chunk_still_splits(tmp_path) -> None:
    """Review 2026-08-29 P2-2: the create id was scoped to the source row but
    not to the position inside it, so a row repeating a chunk derived the same
    UUID5 twice and tripped nodes.local_id's UNIQUE constraint mid-migration."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("S", "subject", {"surface": "某主播", "intro": "x",
                                             "category": "streamer",
                                             "section_order": ["档案"], "preset": "streamer"})
            txn.create_node("F", "fact", {"field": "其他", "sep": ": ",
                                          "value": "重复；重复；重复"})
            txn.create_membership("M", "S", "F", "档案", 0)
        plan = phase_b.build_plan(store)
        phase_b.execute_plan(store, plan)
        texts = [store.node(m.child_id).payload.get("text")
                 for m in store.children("S")]
        assert texts == ["重复", "重复", "重复"]
