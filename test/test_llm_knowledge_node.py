from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.knowledge.node import migrate as migrate_cli
from finesub.llm.knowledge.node.importer import (
    ImportFormatError,
    classify_line,
    classify_legacy_line,
    extract_misheard,
    import_knowledge_root,
    parse_entry_text,
)
from finesub.llm.knowledge.node.model import migration_id
from finesub.llm.knowledge.node.parity import check_parity
from finesub.llm.knowledge.node.presets import load_preset, preset_for_category
from finesub.llm.knowledge.node.render import HandleMap, render_index, render_subject
from finesub.llm.knowledge.node.store import KnowledgeStore, StaleWriteError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "knowledge"

STREAMER = """# 星野灯
个人势 VTuber

## 档案
本名: 星野灯（ほしの あかり）/ 星野灯 / Hoshino Akari
别名: あかり、小灯
人设:

## 直播内容

杂谈
歌回

## 说话风格
自称: 灯
口癖: てぇてぇ（常被 ASR 误听为「ててえ」「てええ」）
语体:

## 喜好 / 特点

## 重要经历

2024-01-01: 出道

## 人际关系

## 元数据
最近更新日期: 2026-08-01
"""

COMMON = """# 测试游戏
一个游戏

## 档案
本名: 测试游戏（てすと）/ 测试游戏 / Test Game
别名: TG
其他:

## 角色

アリス|爱丽丝|Alice|ありす|主角。常被 ASR 误听为「アリズ」。
ボブ|鲍勃|Bob||配角
ボブ|鲍伯|Bob||另一处

## 元数据

最近更新日期: 2026-08-02
"""


def _write_tree(root: Path) -> None:
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text(
        "# 主播索引\n- 星野灯 | Hoshino Akari | あかり、小灯 | 个人势 VTuber\n", encoding="utf-8"
    )
    (root / "common" / "index.md").write_text(
        "# Common\n- 测试游戏 [游戏] | Test Game | TG | 一个游戏\n", encoding="utf-8"
    )
    (root / "streamer" / "星野灯.md").write_text(STREAMER, encoding="utf-8")
    (root / "common" / "测试游戏.md").write_text(COMMON, encoding="utf-8")


# ---- store -------------------------------------------------------------------


def test_store_versions_pinned_reads_and_cas(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("user") as txn:
            txn.create_node("n1", "note", {"text": "v1"})
        rev1 = store.current_rev()
        with store.begin("user") as txn:
            txn.update_node("n1", payload={"text": "v2"}, expected_from_rev=rev1)
        rev2 = store.current_rev()
        assert rev2 == rev1 + 1
        assert store.node("n1", rev1).payload["text"] == "v1"
        assert store.node("n1", rev2).payload["text"] == "v2"
        assert store.node("n1").valid_from_rev == rev2
        # stale writer: still believes rev1 is current
        with pytest.raises(StaleWriteError):
            with store.begin("user") as txn:
                txn.update_node("n1", payload={"text": "v3"}, expected_from_rev=rev1)
        # the failed transaction did not burn a revision or leave rows behind
        assert store.current_rev() == rev2
        assert store.node("n1").payload["text"] == "v2"
        with store.begin("user") as txn:
            txn.tombstone_node("n1", expected_from_rev=rev2)
        assert store.node("n1") is None
        assert store.node("n1", rev2).payload["text"] == "v2"


def test_store_items_and_memberships_are_versioned(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("s", "subject", {"surface": "S", "intro": "", "category": "common"})
            txn.create_node("t", "term", {"surface": "x", "desc": "d"})
            txn.create_membership("m", "s", "t", "角色", 0)
            txn.create_item("i", "t", "misheard", "y", fuzzy_enabled=True)
        r1 = store.current_rev()
        with store.begin("harness") as txn:
            txn.move_membership("m", section="人名", expected_from_rev=r1)
            txn.update_item("i", exact_enabled=False, expected_from_rev=r1)
        assert store.children("s", r1)[0].section == "角色"
        assert store.children("s")[0].section == "人名"
        assert store.items_of("t", r1)[0].exact_enabled is True
        assert store.items_of("t")[0].exact_enabled is False
        assert [s.local_id for s in store.subjects()] == ["s"]


# ---- parsing / classification ------------------------------------------------


def test_parse_entry_layout_flags_and_empty_sections(tmp_path) -> None:
    parsed = parse_entry_text(tmp_path / "x.md", STREAMER)
    flags = {s.name: s.blank_after_heading for s in parsed.sections}
    assert flags["档案"] is False
    assert flags["直播内容"] is True
    assert flags["人际关系"] is False  # empty section: the blank is the separator
    assert parsed.updated_date == "2026-08-01"
    assert parsed.metadata_blank_after is False
    assert [s.name for s in parsed.sections][-1] == "人际关系"


def test_parse_rejects_unknown_metadata_lines(tmp_path) -> None:
    bad = STREAMER.replace("最近更新日期: 2026-08-01", "最近更新日期: 2026-08-01\n来源: 手写")
    with pytest.raises(ImportFormatError):
        parse_entry_text(tmp_path / "x.md", bad)


def test_classify_line_forms() -> None:
    """v3: an optional ``[标记]`` prefix, then the body's shape decides."""

    note_only = load_preset("streamer").section("档案")
    terms = load_preset("streamer").section("频道用语")

    # no label: the body is the whole line, and a colon means nothing
    assert classify_line("游戏实况，主打《原神》《崩坏：星穹铁道》等", note_only) == (
        "note", {"text": "游戏实况，主打《原神》《崩坏：星穹铁道》等"},
    )
    # a label rides the payload as a cross-kind key
    kind, payload = classify_line("[本名] 星野灯 / 星野灯", note_only)
    assert kind == "note" and payload == {"text": "星野灯 / 星野灯", "label": "本名"}
    # unregistered labels are legal (brackets disambiguate; registration only
    # carries core/role/share/verify)
    assert classify_line("[音域] 三个八度", note_only)[1]["label"] == "音域"

    # four columns, desc last and absorbing further pipes
    assert classify_line("a|b|x、y|c", terms) == (
        "term", {"surface": "a", "zh": "b", "desc": "c", "alias_text": "x、y"},
    )
    kind, payload = classify_line("a|b|c|d|e|f", terms)
    assert kind == "term" and payload["desc"] == "d|e|f" and payload["alias_text"] == "c"
    assert classify_line("a|b||c", terms)[1]["alias_text"] == ""
    # a labelled term body keeps both
    kind, payload = classify_line("[自称] ワタヌキ|我|わたぬき|读评论时译作四月一日", terms)
    assert kind == "term" and payload["label"] == "自称" and payload["surface"] == "ワタヌキ"
    # exactly three segments: a format ERROR, never a silent note
    kind, payload = classify_line("a|b|c", terms)
    assert kind == "invalid" and "四列" in payload["error"]
    # a body whose kind the section does not admit is invalid, not downgraded
    assert classify_line("a|b", terms)[0] == "invalid"
    # ...but the pipe rules are term diagnostics: where no term can occur, a
    # pipe is a character. Subtitle lines join speakers with ` | `, so a style
    # entry's 正例 must be able to hold one.
    assert classify_line("a|b|x|c", note_only) == ("note", {"text": "a|b|x|c"})
    assert classify_line("a|b|c", note_only) == ("note", {"text": "a|b|c"})
    assert classify_line("[正例] 甲|乙 → 甲、乙", note_only)[1]["text"] == "甲|乙 → 甲、乙"
    # an empty label name is a format error
    assert classify_line("[] x", note_only)[0] == "invalid"


def test_classify_legacy_line_keeps_the_frozen_archive_grammar() -> None:
    assert classify_legacy_line("本名: A", "fact")[0] == "fact"
    assert classify_legacy_line("杂谈", "fact") == ("note", {"text": "杂谈"})
    assert classify_legacy_line("2024-01-01: 出道", "event")[1]["occurred_at"] == "2024-01-01"
    assert classify_legacy_line("パパ | 父亲", "relation")[1] == {
        "target": "パパ", "sep": " | ", "description": "父亲",
    }
    assert classify_legacy_line("a|b|c", "term")[0] == "note"
    assert classify_legacy_line("a|b|c|d|e", "term")[1]["reading"] == "d"


def test_extract_misheard_forms() -> None:
    assert extract_misheard("口语常被 ASR 误听为「残表」。") == ["残表"]
    assert extract_misheard("常见 ASR 误听为“ユプカ竜”、“ユブカリュウ”。") == ["ユプカ竜", "ユブカリュウ"]
    assert extract_misheard("误听为音近的「しおり（栞）」。") == ["しおり"]
    assert extract_misheard("误听: 面言、面談。其他") == ["面言", "面談"]
    assert extract_misheard("没有标记") == []


def test_misheard_extraction_stops_at_the_clause_not_the_sentence() -> None:
    # the counter-example in the NEXT clause is not a misheard variant of this
    # term — collecting it would drive corrections in the wrong direction
    text = "纳塔五星冰元素角色。口语常被 ASR 误听为「ストラリア」等，纠错时极易被误更正为「夏洛特」。"
    assert extract_misheard(text) == ["ストラリア"]
    # but a run of variants really does spill past a 、
    assert extract_misheard("常见 ASR 误听为“ユプカ竜”、“ユブカリュウ”。") == ["ユプカ竜", "ユブカリュウ"]


def test_strip_misheard_prose_never_empties_a_description() -> None:
    from finesub.llm.knowledge.node.importer import strip_misheard_prose

    # definition and 误听 note share one sentence, joined by commas: the old
    # sentence-granularity rule wiped the whole description
    comma_joined = "枫丹地区的常见鸟类，特征为伞状的头部羽毛，常被 ASR 误听为“チュレート傘柄”"
    assert strip_misheard_prose(comma_joined) == "枫丹地区的常见鸟类，特征为伞状的头部羽毛"
    assert (
        strip_misheard_prose("纳塔“悬木人”部族共生的龙类，常见 ASR 误听为“ユプカ竜”、“ユブカリュウ”。")
        == "纳塔“悬木人”部族共生的龙类"
    )
    # a 误听 clause that IS the whole description keeps the prose rather than
    # leaving an empty desc behind
    only_misheard = "口语中提到“秩序のホラガイダン”等时常被 ASR 误听为「実情のホラガイダン」。"
    assert strip_misheard_prose(only_misheard).strip()
    # sentences that do not carry the marker are untouched
    assert (
        strip_misheard_prose("雪国妖精。常被 ASR 误听为「リンレア」。具有极强的学者探究精神。")
        == "雪国妖精。具有极强的学者探究精神。"
    )


# ---- presets -------------------------------------------------------------------


def test_presets_load() -> None:
    streamer = load_preset("streamer")
    assert streamer.strict_sections and streamer.section("新节") is None
    assert streamer.section("频道 用语") is not None  # whitespace-insensitive
    assert streamer.section("档案").body_kinds == ("note",)
    assert [label.name for label in streamer.section("档案").core_labels()] == [
        "本名", "别名", "人设", "外观",
    ]
    common = preset_for_category("common")
    assert not common.strict_sections and common.section("任意分类").body_kinds == ("term",)
    with pytest.raises(ValueError):
        preset_for_category("translation")
    # the frozen v1 preset is a separate file and keeps the old shape
    legacy = load_preset("streamer", legacy=True)
    assert legacy.version == 1 and legacy.section("说话风格").line_form == "fact"


def test_preset_policy_resolution_is_fail_closed_for_unknown_labels() -> None:
    streamer = load_preset("streamer")
    assert streamer.share_for("档案", "本名") == "inherit"
    assert streamer.share_for("档案", "中之人") == "local"      # unregistered
    assert streamer.verify_for("档案", "中之人") == "none"
    assert streamer.verify_for("人际关系") == "none"            # real people stay home
    assert streamer.verify_for("频道用语") == "external"
    common = load_preset("common")
    # registered but not core: the registration exists to carry `share`
    assert common.share_for("档案", "当前版本") == "local"
    assert common.share_for("档案", "本名") == "inherit"


def test_preset_loader_rejects_instead_of_coercing() -> None:
    from finesub.llm.knowledge.node.presets import PresetError, parse_preset

    base = {
        "name": "x", "version": 2,
        "sections": [{"name": "档案", "body_kinds": ["note"],
                      "labels": [{"name": "本名", "core": True, "note": "n"}]}],
    }

    def mutate(**over):
        import copy
        data = copy.deepcopy(base)
        data.update(over)
        return data

    # `bool("false")` is True — a string here used to silently promote a label
    bad_core = mutate(sections=[{"name": "档案", "body_kinds": ["note"],
                                 "labels": [{"name": "本名", "core": "false", "note": "n"}]}])
    with pytest.raises(PresetError):
        parse_preset(bad_core)
    with pytest.raises(PresetError):  # unknown key must not be swallowed
        parse_preset(mutate(strict_sction=True))
    with pytest.raises(PresetError):  # enum outside its set
        parse_preset(mutate(unknown_label_share="maybe"))
    with pytest.raises(PresetError):  # core label without a note = no guidance
        parse_preset(mutate(sections=[{"name": "档案", "body_kinds": ["note"],
                                       "labels": [{"name": "本名", "core": True}]}]))
    with pytest.raises(PresetError):  # duplicate role
        parse_preset(mutate(sections=[{"name": "档案", "body_kinds": ["note"], "labels": [
            {"name": "本名", "role": "identity"}, {"name": "别称", "role": "identity"}]}]))

    # `labels_from` names another section, so a typo would load fine and then
    # reject every labelled line in that section as an orphan — far from its
    # cause. Resolved at load, like `share_inherit`.
    two = [{"name": "约定", "body_kinds": ["note"]}, {"name": "正例", "body_kinds": ["note"]}]
    assert parse_preset(mutate(sections=[
        two[0], {**two[1], "labels_from": "约定"},
    ])).section("正例").labels_from == "约定"
    with pytest.raises(PresetError):  # target does not exist
        parse_preset(mutate(sections=[two[0], {**two[1], "labels_from": "没有这一节"}]))
    with pytest.raises(PresetError):  # a section cannot source from itself
        parse_preset(mutate(sections=[two[0], {**two[1], "labels_from": "正例"}]))
    with pytest.raises(PresetError):  # ...nor in a cycle
        parse_preset(mutate(sections=[
            {**two[0], "labels_from": "正例"}, {**two[1], "labels_from": "约定"},
        ]))
    # a chain is legal: every label still resolves upstream
    assert parse_preset(mutate(sections=[
        {"name": "a", "body_kinds": ["note"]},
        {"name": "b", "body_kinds": ["note"], "labels_from": "a"},
        {"name": "c", "body_kinds": ["note"], "labels_from": "b"},
    ])).section("c").labels_from == "b"
    with pytest.raises(PresetError):  # duplicate label after NFKC normalization
        parse_preset(mutate(sections=[{"name": "档案", "body_kinds": ["note"], "labels": [
            {"name": "本名"}, {"name": "本 名"}]}]))
    assert parse_preset(base).name == "x"


# ---- importer + parity ------------------------------------------------------------


def test_import_is_lossless_and_deterministic(tmp_path) -> None:
    src = tmp_path / "kb"
    _write_tree(src)
    with KnowledgeStore(tmp_path / "a.sqlite") as store:
        report = import_knowledge_root(src, store)
        assert report.subjects == 2
        parity = check_parity(store, src, rev=report.rev)
        assert parity.legacy_ok, [f.legacy_diff for f in parity.files]
        assert parity.human_ok
        assert not parity.index_mismatches
        # derived indexes, verbatim text untouched
        subjects = {s.payload["surface"]: s for s in store.subjects()}
        game = subjects["测试游戏"]
        assert game.payload["entry_type"] == "游戏"
        assert game.payload["native_names"] == ["Test Game"]
        assert game.payload["reading"] == "てすと"
        misheard = [i.value for i in store.all_items() if i.field == "misheard"]
        assert {"アリズ", "ててえ", "てええ"} <= set(misheard)
        # one conflict candidate: ボブ with two different zh names
        assert [(c.surface, c.kind) for c in report.merge_candidates] == [("ボブ", "conflict")]
        ids_first = sorted(row["local_id"] for row in report.id_map)
    with KnowledgeStore(tmp_path / "b.sqlite") as store:
        report2 = import_knowledge_root(src, store)
        assert sorted(row["local_id"] for row in report2.id_map) == ids_first
        assert report2.source_lock == report.source_lock


def test_import_preserves_sections_outside_the_strict_preset(tmp_path) -> None:
    """An unknown section is archived, not refused (owner 2026-09-01).

    The importer used to raise, which meant one custom section anywhere made a
    whole knowledge base unmigratable -- measured on a real third-party root:
    30 refusals, `翻译约定` alone in 15 of 17 entries. Format problems are not
    to be aligned by special cases; they degrade. So the lines come in
    VERBATIM as notes under their own name, the archive still round-trips, and
    the report says what had to be degraded. Reshaping them is Phase B's job.
    """

    from finesub.llm.knowledge.node.render import render_subject

    src = tmp_path / "kb"
    _write_tree(src)
    path = src / "streamer" / "星野灯.md"
    original = path.read_text(encoding="utf-8").replace("## 人际关系", "## 自由节")
    path.write_text(original, encoding="utf-8")
    with KnowledgeStore(tmp_path / "a.sqlite") as store:
        report = import_knowledge_root(src, store)
        assert store.current_rev() > 0
        assert [
            (row["section"], row["source_path"]) for row in report.degraded_sections
        ] == [("自由节", "streamer/星野灯.md")]
        subject = next(
            row["local_id"] for row in report.id_map
            if row["kind"] == "subject" and row["source_path"] == "streamer/星野灯.md"
        )
        # verbatim: the section keeps its name and its lines come back byte for
        # byte, which is what lets the parity gate stay a byte-for-byte gate.
        assert render_subject(store, subject, mode="legacy") == original


def test_import_keeps_the_users_own_html_comments(tmp_path) -> None:
    """A whole-line comment in the ARCHIVE is the user's content.

    `strip_comments` drops them because the full preview writes its own
    scaffolding into comments -- true of a file we wrote, false of the legacy
    tree. A third party's tooling had written traceability comments into their
    entries; dropping them lost the mapping AND failed the parity gate with an
    error that named the file but not the cause.
    """

    from finesub.llm.knowledge.node.render import render_subject

    src = tmp_path / "kb"
    _write_tree(src)
    path = src / "streamer" / "星野灯.md"
    marked = path.read_text(encoding="utf-8").replace(
        "## 人际关系", "<!-- tool:v1 id=abc -->\n## 人际关系", 1
    )
    path.write_text(marked, encoding="utf-8")
    with KnowledgeStore(tmp_path / "b.sqlite") as store:
        report = import_knowledge_root(src, store)
        subject = next(
            row["local_id"] for row in report.id_map
            if row["kind"] == "subject" and row["source_path"] == "streamer/星野灯.md"
        )
        assert "<!-- tool:v1 id=abc -->" in render_subject(store, subject, mode="legacy")


def test_import_records_a_long_blank_run_between_sections(tmp_path) -> None:
    """One blank is the separator every section gets; a longer run is the
    user's own spacing. Unrecorded, it collapsed on projection and that alone
    failed the parity gate on a real root."""

    from finesub.llm.knowledge.node.render import render_subject

    src = tmp_path / "kb"
    _write_tree(src)
    path = src / "streamer" / "星野灯.md"
    spaced = path.read_text(encoding="utf-8").replace(
        "\n## 元数据", "\n\n\n\n## 元数据", 1
    )
    path.write_text(spaced, encoding="utf-8")
    with KnowledgeStore(tmp_path / "c.sqlite") as store:
        report = import_knowledge_root(src, store)
        subject = next(
            row["local_id"] for row in report.id_map
            if row["kind"] == "subject" and row["source_path"] == "streamer/星野灯.md"
        )
        assert render_subject(store, subject, mode="legacy") == spaced


def test_examples_tree_is_written_in_the_current_grammar() -> None:
    """The tracked sample tree is v3, so the FROZEN archive importer must not
    be pointed at it — its round-trip lives in the CLI suite, which drives the
    real ingest path (edit_subject)."""

    from finesub.llm.knowledge.node.importer import split_label

    for path in sorted(EXAMPLES.rglob("*.md")):
        if path.name in ("index.md", "README.md"):
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw[2:] if raw.startswith("- ") else raw
            if not line or line.startswith("#") or line.startswith("最近更新日期"):
                continue
            label, body = split_label(line)
            # no line may still be carrying the v1 `字段: 值` shape
            assert label is not None or "|" in body or ": " not in body[:12], (path.name, raw)


def test_render_prompt_handles_and_index(tmp_path) -> None:
    src = tmp_path / "kb"
    _write_tree(src)
    with KnowledgeStore(tmp_path / "a.sqlite") as store:
        import_knowledge_root(src, store)
        subject = next(s for s in store.subjects() if s.payload["category"] == "streamer")
        handles = HandleMap()
        text = render_subject(store, subject.local_id, mode="prompt", handles=handles)
        assert "<!-- @k1 -->" in text
        # membership handles are never rendered (no model op consumes them)
        assert "@m" not in text
        bindings = handles.bindings()
        assert any(b["handle"] == "@k1" and b["id"] == subject.local_id for b in bindings)
        assert all(b["expected_valid_from_rev"] == 1 for b in bindings)
        human = render_subject(store, subject.local_id, mode="human")
        assert "<!--" not in human
        only_profile = render_subject(store, subject.local_id, mode="human", sections=["档案"])
        assert "## 档案" in only_profile and "## 直播内容" not in only_profile and "## 元数据" in only_profile
        index = render_index(store, "common")
        assert index == "- 测试游戏 [游戏] | Test Game | TG | 一个游戏\n"


def test_free_section_guidance_is_rendered_once_not_per_section(tmp_path) -> None:
    """`common` resolves every free section to the same `default_section` spec.

    Rendering the block per section printed one identical paragraph under
    「角色」, 「地区」, 「组织」... -- and the text was never about a section
    anyway: it states what any free section may hold. It is now hoisted above
    the sections and scoped by a preamble; a *named* section keeps its own
    block, because there the text really is section-local.
    """

    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "某作品", "intro": "x", "category": "common",
             "entry_type": "游戏",
             "section_order": ["档案", "角色", "地区", "组织"]},
        )
    text = render_subject(repo.store, "S", mode="human", preview="full")

    free_purpose = "作品内的固定名词"
    assert text.count(free_purpose) == 1, text
    assert "以下自由节" in text
    # hoisted: it precedes every section heading, free or named
    assert text.index("以下自由节") < text.index("## 档案")
    # the named section still carries its own, section-local block
    assert text.count("作品身份与本条目的收录边界") == 1
    assert "## 角色" in text and "## 地区" in text and "## 组织" in text


def test_free_section_guidance_is_absent_when_no_free_section_renders(tmp_path) -> None:
    """A `sections=` filter down to named sections has nothing to scope it to."""

    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "某作品", "intro": "x", "category": "common",
             "entry_type": "游戏", "section_order": ["档案", "角色"]},
        )
    text = render_subject(
        repo.store, "S", mode="human", preview="full", sections=["档案"]
    )

    assert "以下自由节" not in text
    assert "作品身份与本条目的收录边界" in text


def test_strict_preset_keeps_every_section_block(tmp_path) -> None:
    """streamer has no `default_section`: each of its six specs differs, so
    each section's block really is about that section and stays where it is."""

    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "某主播", "intro": "x", "category": "streamer"},
        )
    text = render_subject(repo.store, "S", mode="human", preview="full")

    assert "以下自由节" not in text
    assert "频道自造词" in text and "会改变译文选择的稳定特征" in text


def test_streamer_registers_the_greeting_and_signoff_tags(tmp_path) -> None:
    """`[问候语]` / `[结束语]` are registered, not core.

    Registration is what makes them inherit 频道用语's sharing policy -- an
    unregistered label is fail-closed to `local` and would be dropped from a
    shared bundle. `core` stays false because not every streamer has a fixed
    greeting, and a core slot would put two standing reminders in every entry.
    """

    preset = preset_for_category("streamer")
    section = preset.section("频道用语")
    registered = {label.name: label for label in section.labels}

    assert {"问候语", "结束语"} <= set(registered)
    assert not registered["问候语"].core and not registered["结束语"].core
    for name in ("问候语", "结束语"):
        assert preset.share_for("频道用语", name) == "inherit"
    # unregistered labels stay fail-closed, which is the reason to register
    assert preset.share_for("频道用语", "中之人") == "local"


def test_a_registered_non_core_label_still_shows_its_note(tmp_path) -> None:
    """Otherwise `core = false` means "the model cannot learn what this is for".

    Non-core labels used to render as a bare name list, and the structure
    fragment lists core labels only -- so a knowledge-update task saw
    `[问候语]` with nothing saying what belongs there or in what shape. The
    note is the whole answer, and it costs one line to show it.
    """

    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "某主播", "intro": "x", "category": "streamer"},
        )
    text = render_subject(repo.store, "S", mode="human", preview="full")

    assert "[问候语] 每场开头都说的那句定式" in text
    assert "[结束语] 每场收尾都说的那句定式" in text
    # ...and they are still not core: no empty slot is rendered for them.
    assert "- [问候语]" not in text and "- [结束语]" not in text
    assert "- [自称]" in text  # the section's one core label still gets a slot


def test_index_aliases_render_from_items(tmp_path) -> None:
    """items are the single home for aliases: add_item alone must reach the index."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node(
                "s",
                "subject",
                {"surface": "原神", "intro": "游戏", "category": "common", "entry_type": "游戏",
                 "native_names": ["原神"], "section_order": []},
            )
            txn.create_item("a1", "s", "aliases", "Genshin")
        assert render_index(store, "common") == "- 原神 [游戏] | 原神 | Genshin | 游戏\n"
        with store.begin("harness") as txn:
            txn.create_item("a2", "s", "aliases", "GI")
        assert "| Genshin、GI |" in render_index(store, "common")


def test_add_item_alias_keeps_the_rendered_column_in_step(tmp_path) -> None:
    """add_item on a term rewrites alias_text too — matchable must equal visible."""

    from finesub.llm.knowledge.node.proposals import apply_model_proposals
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "游戏", "intro": "", "category": "common", "section_order": ["角色"]})
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "alias_text": "Alice", "reading": "", "desc": "主角"})
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I", "T", "aliases", "Alice")
    handles = HandleMap()
    repo.entry_prompt_text("S", handles)
    term_handle = next(h for h, (ident, _) in handles.nodes.items() if ident == "T")
    text = (
        "<knowledge_proposals>\n"
        + json.dumps({"op": "add_item", "id": term_handle, "field": "aliases", "value": "Ally", "reason": "r"}, ensure_ascii=False)
        + "\n</knowledge_proposals>"
    )
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles)
    assert not report.rolled_back and report.rev is not None
    # the rendered alias column comes FROM items (plan §11.2): the payload's
    # residual alias_text is untouched, the new item shows up in the render
    assert [i.value for i in repo.store.items_of("T") if i.field == "aliases"] == ["Alice", "Ally"]
    from finesub.llm.knowledge.node.render import render_subject

    assert "アリス|爱丽丝|Alice、Ally|主角" in render_subject(repo.store, "S", mode="human")


def test_update_line_resyncs_alias_items(tmp_path) -> None:
    """Rewriting a term line diffs the alias column into items (plan §2.1)."""

    from finesub.llm.knowledge.node.proposals import apply_model_proposals
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "游戏", "intro": "", "category": "common", "section_order": ["角色"]})
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "alias_text": "Alice", "reading": "", "desc": "主角"})
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I", "T", "aliases", "Alice")
    handles = HandleMap()
    repo.entry_prompt_text("S", handles)
    term_handle = next(h for h, (ident, _) in handles.nodes.items() if ident == "T")
    text = (
        "<knowledge_proposals>\n"
        + json.dumps({"op": "update", "id": term_handle, "line": "アリス|爱丽丝|Ally|主角", "reason": "r"}, ensure_ascii=False)
        + "\n</knowledge_proposals>"
    )
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles)
    assert not report.rolled_back and report.rev is not None
    assert [i.value for i in repo.store.items_of("T") if i.field == "aliases"] == ["Ally"]


def test_migration_ids_are_namespaced_uuid5() -> None:
    assert migration_id("a", "b") == migration_id("a", "b")
    assert migration_id("a", "b") != migration_id("a", "c")


def test_migrate_cli_writes_reports(tmp_path) -> None:
    src = tmp_path / "kb"
    _write_tree(src)
    store = tmp_path / "out" / "kb.sqlite"
    report_dir = tmp_path / "out" / "report"
    assert migrate_cli.main(["--source", str(src), "--store", str(store), "--report", str(report_dir)]) == 0
    assert migrate_cli.main(["--source", str(src), "--store", str(store), "--report", str(report_dir)]) == 2
    assert migrate_cli.main(["--source", str(src), "--store", str(store), "--report", str(report_dir), "--force"]) == 0
    summary = json.loads((report_dir / "import-summary.json").read_text(encoding="utf-8"))
    assert summary["merge_candidates"]["conflict"] == 1
    assert (report_dir / "migration-id-map.jsonl").read_text(encoding="utf-8").count("\n") == summary["nodes"] + summary["subjects"]
    assert "legacy byte-equal: OK" in (report_dir / "parity.md").read_text(encoding="utf-8")


def test_a_line_that_is_already_a_bullet_does_not_get_a_second(tmp_path) -> None:
    """One entry, one bullet -- and the user's own dash is absorbed.

    A real archive whose sections were written as markdown lists rendered 291
    lines as `- - …` across 23 of 27 entries (2026-09-01). Not adding a second
    bullet costs the leading dash on the way back in, because `_strip_bullet`
    removes exactly one: accepted deliberately over escaping it, which would
    put an escape character in front of a human reader. The archive
    projection carries no bullets and is unaffected.
    """

    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.render import render_subject
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "灯", "intro": "", "category": "streamer",
                                         "section_order": ["档案"]})
        txn.create_node("N", "note", {"text": "- 已经是个列表项"})
        txn.create_membership("M", "S", "N", "档案", 0)

    human = render_subject(repo.store, "S", mode="human")
    assert "- 已经是个列表项" in human
    assert "- - " not in human
    # legacy (the archive face) never had a bullet to begin with
    assert "- 已经是个列表项" in render_subject(repo.store, "S", mode="legacy")

    # the documented cost: harvesting that same text back eats the dash
    edit_subject(repo, repo.store.node("S"), human)
    assert repo.store.node("N").payload["text"] == "已经是个列表项"


def test_two_files_with_one_h1_are_ranked_and_folded(tmp_path) -> None:
    """Same title, two files: one entry, not two silently identical subjects.

    A real third-party root had exactly this (`魔法少女の魔女裁判.md` and
    `魔法少女ノ魔女裁判.md`, same H1). Imported as-is it produced two subjects
    with the same surface, `warnings` and `merge_candidates` both empty, and
    name resolution reaching only one of them.

    Ranking is newest-then-longest (owner 2026-09-01). ⚠ The mtime tie is the
    NORMAL case -- a clone or an unpacked archive stamps every file the same --
    so this fixture writes both at once and lets length decide, which is what
    will actually happen in the field.
    """

    import os

    from finesub.llm.knowledge.node import phase_b
    from finesub.llm.knowledge.node.importer import STAGING_SECTION
    from finesub.llm.knowledge.node.render import render_subject

    src = tmp_path / "kb"
    _write_tree(src)
    short = src / "common" / "测试游戏-旧.md"
    short.write_text(COMMON, encoding="utf-8")
    long_path = src / "common" / "测试游戏.md"
    long_path.write_text(
        COMMON.replace(
            "ボブ|鲍伯|Bob||另一处",
            "ボブ|鲍伯|Bob||另一处\nキャロル|卡萝|Carol||只有长的那份有",
        ),
        encoding="utf-8",
    )
    stamp = 1_700_000_000
    for path in (short, long_path):
        os.utime(path, (stamp, stamp))  # the tie the field actually produces

    with KnowledgeStore(tmp_path / "a.sqlite") as store:
        report = import_knowledge_root(src, store)
        (row,) = report.duplicate_subjects
        assert row["winner"] == "common/测试游戏.md"  # same mtime -> longer wins
        assert row["losers"] == ["common/测试游戏-旧.md"]
        assert any("标题都是" in w for w in report.warnings)

        # both are still imported in full: the archive round-trips file by file
        ids = {
            r["source_path"]: r["local_id"] for r in report.id_map if r["kind"] == "subject"
        }
        for rel, subject_id in ids.items():
            if rel.startswith("common/测试游戏"):
                assert render_subject(store, subject_id, mode="legacy") == (
                    src / rel
                ).read_text(encoding="utf-8")

        # phase B is where they become one entry
        plan = phase_b.build_plan(store)
        (merge,) = [r for r in plan.rows if r.action == "merge"]
        assert merge.new_section == STAGING_SECTION
        phase_b.execute_plan(store, plan)

        loser = ids["common/测试游戏-旧.md"]
        winner = ids["common/测试游戏.md"]
        assert store.node(loser) is None  # retired
        parked = [
            m for m in store.children(winner) if m.section == STAGING_SECTION
        ]
        assert parked, "the loser's lines must land in the winner's staging section"


def test_a_duplicate_whose_winner_vanished_is_left_alone(tmp_path) -> None:
    """Folding into nothing would delete content, so it does not happen."""

    import os

    from finesub.llm.knowledge.node import phase_b

    src = tmp_path / "kb"
    _write_tree(src)
    (src / "common" / "测试游戏-旧.md").write_text(COMMON, encoding="utf-8")
    stamp = 1_700_000_000
    for name in ("测试游戏.md", "测试游戏-旧.md"):
        os.utime(src / "common" / name, (stamp, stamp))

    with KnowledgeStore(tmp_path / "b.sqlite") as store:
        report = import_knowledge_root(src, store)
        ids = {
            r["source_path"]: r["local_id"] for r in report.id_map if r["kind"] == "subject"
        }
        # both files are byte-identical here, so mtime and length both tie and
        # the path decides -- read the winner off the report rather than
        # assuming which way that fell.
        (dup,) = report.duplicate_subjects
        with store.begin("user") as txn:
            txn.tombstone_node(ids[dup["winner"]])
        plan = phase_b.build_plan(store)
        assert not [r for r in plan.rows if r.action == "merge"]
        assert any("胜出条目已不存在" in r.note for r in plan.rows)
        assert store.node(ids[dup["losers"][0]]) is not None
