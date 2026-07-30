from __future__ import annotations

import json
import subprocess

import pytest

from llm.knowledge.base import (
    LineEdit,
    append_lines_text,
    append_task_artifact,
    apply_knowledge_proposals,
    apply_line_edits,
    ensure_knowledge_git,
    ensure_latest_update_date_text,
    load_entry_texts,
    load_index_entries,
    parse_knowledge_proposals,
    read_task_artifacts,
    replace_section_text,
)


def test_ensure_knowledge_git_refuses_dirty_existing_repo(tmp_path) -> None:
    report = apply_knowledge_proposals(_proposal(), knowledge_root=tmp_path)
    assert report.committed
    entry = tmp_path / "streamer" / "星野灯.md"
    entry.write_text(entry.read_text(encoding="utf-8") + "\n本地未提交。\n", encoding="utf-8")

    assert not ensure_knowledge_git(tmp_path)


def test_auto_apply_snapshots_preexisting_user_adjustments(tmp_path) -> None:
    first = apply_knowledge_proposals(
        _proposal(),
        knowledge_root=tmp_path,
        task_id="first",
    )
    assert first.committed
    entry = tmp_path / "streamer" / "星野灯.md"
    entry.write_text(
        entry.read_text(encoding="utf-8") + "\n用户补充。\n",
        encoding="utf-8",
    )

    second = apply_knowledge_proposals(
        _proposal(section="备注", content="自动更新。"),
        knowledge_root=tmp_path,
        task_id="second",
    )

    assert second.committed
    subjects = subprocess.run(
        ["git", "log", "-2", "--format=%s"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert subjects[0].startswith("[second]")
    assert subjects[1] == "[user-adjustment] snapshot before auto-apply"


def _proposal(**overrides) -> str:
    data = {
        "category": "streamer",
        "entry": "星野灯",
        "aliases": ["ほしのあかり", "阿灯"],
        "intro": "虚拟主播，杂谈和恐怖游戏为主",
        "op": "replace_section",
        "section": "档案",
        "content": "关西腔，喜欢恐怖游戏。",
        "reason": "测试",
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_default_knowledge_root_is_repo_root_knowledge_dir() -> None:
    # Regression: the llm.knowledge package split once left this pointing at
    # src/knowledge, silently emptying every default-path lookup.
    from llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT

    assert DEFAULT_KNOWLEDGE_ROOT.name == "knowledge"
    assert (DEFAULT_KNOWLEDGE_ROOT.parent / "pyproject.toml").exists()


def test_parse_knowledge_proposals_accepts_wrapped_jsonl() -> None:
    text = f"<knowledge_proposals>\n{_proposal()}\n</knowledge_proposals>"

    proposals = parse_knowledge_proposals(text)

    assert len(proposals) == 1
    assert proposals[0].category == "streamer"
    assert proposals[0].entry == "星野灯"
    assert proposals[0].aliases == ("ほしのあかり", "阿灯")


def test_apply_creates_entry_file_and_index_line(tmp_path) -> None:
    report = apply_knowledge_proposals(
        _proposal(),
        knowledge_root=tmp_path,
        task_id="task-1",
        commit=False,
    )

    assert [record.status for record in report.applied] == ["applied"]
    entry_text = (tmp_path / "streamer" / "星野灯.md").read_text(encoding="utf-8")
    assert entry_text.startswith("# 星野灯")
    # Auto-created entries scaffold from the streamer preset: fixed sections
    # kept, metadata last with the harness-owned date line.
    assert "## 元数据" in entry_text
    assert "最近更新日期: " in entry_text
    assert "## 说话风格" in entry_text
    assert "## 档案" in entry_text
    assert "关西腔，喜欢恐怖游戏。" in entry_text
    assert entry_text.rstrip().splitlines()[1] == "虚拟主播，杂谈和恐怖游戏为主"
    entries = load_index_entries(tmp_path, "streamer")
    assert entries[0].key == "星野灯"
    assert "阿灯" in entries[0].aliases
    assert entries[0].intro == "虚拟主播，杂谈和恐怖游戏为主"


def test_apply_replace_section_overwrites_previous_body(tmp_path) -> None:
    apply_knowledge_proposals(_proposal(), knowledge_root=tmp_path, commit=False)
    apply_knowledge_proposals(
        _proposal(content="改用标准语，最近玩galgame。"),
        knowledge_root=tmp_path,
        commit=False,
    )

    entry_text = (tmp_path / "streamer" / "星野灯.md").read_text(encoding="utf-8")
    assert "改用标准语" in entry_text
    assert "关西腔，喜欢恐怖游戏。" not in entry_text
    assert entry_text.count("## 档案") == 1


def test_apply_append_lines_appends_and_dedups_by_first_field(tmp_path) -> None:
    apply_knowledge_proposals(
        _proposal(
            op="append_lines",
            section="重要经历",
            content="2026-05: 通关了游戏A。",
        ),
        knowledge_root=tmp_path,
        commit=False,
    )
    report = apply_knowledge_proposals(
        _proposal(
            op="append_lines",
            section="重要经历",
            content="2026-05: 通关了游戏A。\n2026-07: 开坑游戏B。",
        ),
        knowledge_root=tmp_path,
        commit=False,
    )

    entry_text = (tmp_path / "streamer" / "星野灯.md").read_text(encoding="utf-8")
    # First-field dedup: the 2026-05 row is not written twice.
    assert entry_text.count("通关了游戏A。") == 1
    assert entry_text.index("通关了游戏A。") < entry_text.index("开坑游戏B。")
    assert "(1 rows)" in report.applied[0].reason


def test_append_lines_creates_missing_section_before_metadata(tmp_path) -> None:
    apply_knowledge_proposals(_proposal(), knowledge_root=tmp_path, commit=False)
    apply_knowledge_proposals(
        _proposal(
            category="streamer",
            op="append_lines",
            section="自定义观察",
            content="喜欢在深夜直播。",
        ),
        knowledge_root=tmp_path,
        commit=False,
    )

    entry_text = (tmp_path / "streamer" / "星野灯.md").read_text(encoding="utf-8")
    assert "## 自定义观察" in entry_text
    # 元数据 stays the last section.
    assert entry_text.index("## 自定义观察") < entry_text.index("## 元数据")
    assert entry_text.rstrip().rsplit("## ", 1)[1].startswith("元数据")


def test_apply_edit_lines_changes_lines_against_snapshot(tmp_path) -> None:
    apply_knowledge_proposals(
        _proposal(
            op="append_lines",
            section="喜好 / 特点",
            content="喜欢恐怖游戏。\n怕辣。",
        ),
        knowledge_root=tmp_path,
        commit=False,
    )
    text = (tmp_path / "streamer" / "星野灯.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    wrong = lines.index("怕辣。") + 1
    right = lines.index("喜欢恐怖游戏。") + 1

    proposal = json.dumps(
        {
            "category": "streamer",
            "entry": "星野灯",
            "op": "edit_lines",
            "edits": [
                {"action": "change", "line": wrong, "content": "其实很能吃辣。"},
                {"action": "insert_after", "line": right, "content": "尤其是 8 番出口类。"},
            ],
            "reason": "测试",
        },
        ensure_ascii=False,
    )
    report = apply_knowledge_proposals(
        proposal, knowledge_root=tmp_path, commit=False
    )

    assert report.applied and "2 edits" in report.applied[0].reason
    updated = (tmp_path / "streamer" / "星野灯.md").read_text(encoding="utf-8")
    assert "其实很能吃辣。" in updated and "怕辣。" not in updated
    # Snapshot numbering: the insert after the lower line is unaffected by the
    # change at the higher line.
    assert updated.index("喜欢恐怖游戏。") < updated.index("尤其是 8 番出口类。")


def test_edit_lines_guard_rejects_uninjected_entries(tmp_path) -> None:
    apply_knowledge_proposals(_proposal(), knowledge_root=tmp_path, commit=False)
    proposal = json.dumps(
        {
            "category": "streamer",
            "entry": "星野灯",
            "op": "edit_lines",
            "edits": [{"action": "remove", "line": 5}],
            "reason": "测试",
        },
        ensure_ascii=False,
    )

    report = apply_knowledge_proposals(
        proposal,
        knowledge_root=tmp_path,
        commit=False,
        line_editable=[],  # nothing was injected untruncated this round
    )

    assert not report.applied
    assert "not injected or truncated" in report.skipped[0].reason


def test_apply_line_edits_guards_h1_and_metadata() -> None:
    text = "# 条目\n简介。\n\n## 档案\n本名: 条目\n\n## 元数据\n最近更新日期: 2026-07-01\n"
    _, err = apply_line_edits(text, [LineEdit("change", 1, "# 新名字")])
    assert "line 1 is the H1" in err or "out of range" in err
    meta_line = text.splitlines().index("最近更新日期: 2026-07-01") + 1
    _, err = apply_line_edits(text, [LineEdit("remove", meta_line)])
    assert "元数据" in err


def test_apply_create_entry_scaffolds_from_preset(tmp_path) -> None:
    proposal = json.dumps(
        {
            "category": "common",
            "entry": "原神",
            "op": "create_entry",
            "entry_type": "游戏",
            "intro": "米哈游开放世界 RPG",
            "aliases": ["Genshin"],
            "reason": "库中无既有母词条可并入（测试）",
        },
        ensure_ascii=False,
    )

    report = apply_knowledge_proposals(proposal, knowledge_root=tmp_path, commit=False)

    assert report.applied and report.applied[0].op == "create_entry"
    text = (tmp_path / "common" / "原神.md").read_text(encoding="utf-8")
    assert text.startswith("# 原神\n米哈游开放世界 RPG")
    # v15: common preset keeps only 档案 + 元数据 — category sections are
    # model-named (append_lines auto-creates them).
    assert "## 档案" in text and "## 元数据" in text
    entries = load_index_entries(tmp_path, "common")
    assert entries[0].key == "原神"
    assert entries[0].entry_type == "游戏"
    assert "Genshin" in entries[0].aliases


def test_apply_redirects_script_variant_keys_to_existing_entry(tmp_path) -> None:
    # 绯月ゆい vs 緋月ゆい: new-entry keys that match an existing key/alias
    # after 繁→简归一 (opencc t2s) must land in the existing file.
    apply_knowledge_proposals(
        _proposal(entry="緋月ゆい", aliases=["绯月唯"]),
        knowledge_root=tmp_path,
        commit=False,
    )
    report = apply_knowledge_proposals(
        _proposal(
            entry="绯月ゆい",
            op="append_lines",
            section="重要经历",
            content="2026-07: 二创观看反应。",
        ),
        knowledge_root=tmp_path,
        commit=False,
    )

    assert report.applied and report.applied[0].entry == "緋月ゆい"
    assert "redirected" in report.applied[0].reason
    assert not (tmp_path / "streamer" / "绯月ゆい.md").exists()
    entry_text = (tmp_path / "streamer" / "緋月ゆい.md").read_text(encoding="utf-8")
    assert "2026-07: 二创观看反应。" in entry_text
    assert len(load_index_entries(tmp_path, "streamer")) == 1


def test_apply_skips_invalid_proposals(tmp_path) -> None:
    text = "\n".join(
        [
            _proposal(),
            _proposal(category="translation_style"),
            # The common-mistake ledger only accepts <mistake_proposals>;
            # "translation" stays invalid for <knowledge_proposals>.
            _proposal(category="translation"),
            _proposal(entry="非法/名字"),
            _proposal(op="rewrite_all"),
            _proposal(content=""),
        ]
    )

    report = apply_knowledge_proposals(text, knowledge_root=tmp_path, commit=False)

    assert len(report.applied) == 1
    assert len(report.skipped) == 5
    reasons = " ".join(record.reason for record in report.skipped)
    assert "category" in reasons
    assert "filename" in reasons
    assert "op" in reasons
    assert "content" in reasons


@pytest.mark.slow
def test_apply_commits_to_embedded_git_repo(tmp_path) -> None:
    report = apply_knowledge_proposals(
        _proposal(),
        knowledge_root=tmp_path,
        task_id="task-9",
    )

    assert report.committed
    assert (tmp_path / ".git").exists()
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert "task-9" in log.stdout


def test_common_entry_with_type_and_alias_resolution(tmp_path) -> None:
    apply_knowledge_proposals(
        _proposal(
            category="common",
            entry="米糕之泪",
            entry_type="游戏",
            aliases=["MikoTears", "米糕"],
            intro="独立恐怖游戏",
            section="概要",
            content="2026 年发售的独立恐怖游戏。",
        ),
        knowledge_root=tmp_path,
        commit=False,
    )

    entries = load_index_entries(tmp_path, "common")
    assert entries[0].entry_type == "游戏"
    found, missing = load_entry_texts(tmp_path, ["米糕", "不存在的条目"])
    assert "米糕之泪" in found
    assert "2026 年发售" in found["米糕之泪"]
    assert missing == ["不存在的条目"]


def test_replace_and_append_text_helpers_create_missing_sections() -> None:
    text = "# 条目\n\n## 元数据\n\n最近更新日期: 2026-07-01\n"
    replaced = replace_section_text(text, "档案", "本名: 条目")
    assert "## 档案" in replaced
    appended, added = append_lines_text(replaced, "重要经历", "2026-07-02: 事件。")
    assert added == 1
    # New sections land before 元数据, which stays last.
    assert appended.index("## 档案") < appended.index("## 重要经历")
    assert appended.index("## 重要经历") < appended.index("## 元数据")
    # Section names match whitespace-insensitively (术语/系统… vs 术语 / 系统…).
    spaced, _ = append_lines_text(appended, "术语 / 系统 / 其他专有名词", "A|甲|||测试")
    merged, added = append_lines_text(spaced, "术语/系统/其他专有名词", "B|乙|||测试")
    assert added == 1
    assert merged.count("术语 / 系统 / 其他专有名词") == 1


def test_latest_update_date_metadata_is_inserted_and_replaced() -> None:
    text = ensure_latest_update_date_text("# 条目\n\n## 档案\n\n旧内容。\n", "2026-07-04")
    assert "## 元数据" in text
    assert "最近更新日期: 2026-07-04" in text
    updated = ensure_latest_update_date_text(text, "2026-07-05")
    assert "最近更新日期: 2026-07-05" in updated
    assert "2026-07-04" not in updated
    # Legacy bulleted label lines are replaced, not duplicated.
    legacy = "# 条目\n\n## 元数据\n\n- 最新更新日期：2026-01-01\n"
    migrated = ensure_latest_update_date_text(legacy, "2026-07-05")
    assert "最新更新日期" not in migrated
    assert migrated.count("最近更新日期") == 1


def test_task_artifact_retention_round_trip(tmp_path) -> None:
    append_task_artifact(
        tmp_path,
        kind="search_result",
        task_id="task-1",
        payload={"query": "角色A", "summary": "角色A 是术语。"},
    )

    retained = read_task_artifacts([tmp_path])

    assert "search_result" in retained
    assert "角色A 是术语" in retained


def _write_kb(tmp_path):
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text(
        "# 主播索引\n\n"
        "- 兔田佩克拉 | 佩克拉、Pekora | hololive 三期生\n"
        "- 星野灯 | ほしのあかり、灯 | 杂谈主播\n",
        encoding="utf-8",
    )
    (root / "common" / "index.md").write_text(
        "# common 索引\n\n"
        "- 崩坏星穹铁道 [游戏] | 崩铁、星铁 | 回合制 RPG\n",
        encoding="utf-8",
    )
    (root / "streamer" / "兔田佩克拉.md").write_text("# 兔田佩克拉\n\n## 档案\n\n- 兔子\n", encoding="utf-8")
    (root / "common" / "崩坏星穹铁道.md").write_text("# 崩坏星穹铁道\n\n## 简介\n\n- 游戏\n", encoding="utf-8")
    return root


def test_match_index_keywords_matches_keys_and_aliases_with_frequency_rank(tmp_path) -> None:
    from llm.knowledge.base import match_index_keywords

    root = _write_kb(tmp_path)
    note = "今天佩克拉直播玩崩铁，还提到星铁的新版本，佩克拉说很好玩。pekora 加油"

    matches = match_index_keywords(root, note)

    assert [(m.category, m.key) for m in matches] == [
        ("streamer", "兔田佩克拉"),
        ("common", "崩坏星穹铁道"),
    ]
    # 佩克拉 ×2 + casefold("Pekora") ×1 = 3；崩铁 + 星铁 = 2
    assert matches[0].hits == 3
    assert matches[1].hits == 2
    assert "崩铁" in matches[1].matched_terms and "星铁" in matches[1].matched_terms


def test_match_index_keywords_skips_short_terms_and_caps_entries(tmp_path) -> None:
    from llm.knowledge.base import match_index_keywords

    root = _write_kb(tmp_path)
    # "灯" is 1 char -> never matched even though it appears.
    matches = match_index_keywords(root, "灯灯灯灯")
    assert matches == []

    ranked = match_index_keywords(root, "佩克拉 崩铁", max_entries=1)
    assert len(ranked) == 1


def test_load_preinjected_entries_returns_bodies_in_rank_order(tmp_path) -> None:
    from llm.knowledge.base import load_preinjected_entries

    root = _write_kb(tmp_path)
    entries, matches = load_preinjected_entries(root, "崩铁 崩铁 佩克拉")

    assert list(entries) == ["崩坏星穹铁道", "兔田佩克拉"]
    assert entries["崩坏星穹铁道"].startswith("# 崩坏星穹铁道")
    assert [m.key for m in matches] == ["崩坏星穹铁道", "兔田佩克拉"]


def test_delete_entry_requires_merge_first(tmp_path) -> None:
    apply_knowledge_proposals(
        _proposal(
            category="common", entry="原神", op="replace_section",
            section="角色", content="角色行。", entry_type="游戏",
        ),
        knowledge_root=tmp_path, commit=False,
    )
    apply_knowledge_proposals(
        _proposal(
            category="common", entry="布伦妮", op="replace_section",
            section="档案", content="本名: 布伦妮", entry_type="游戏",
        ),
        knowledge_root=tmp_path, commit=False,
    )
    deletion = json.dumps(
        {"category": "common", "entry": "布伦妮", "op": "delete_entry", "reason": "测试"},
        ensure_ascii=False,
    )
    refused = apply_knowledge_proposals(deletion, knowledge_root=tmp_path, commit=False)
    assert refused.skipped and "merge it first" in refused.skipped[0].reason
    assert (tmp_path / "common" / "布伦妮.md").exists()

    # Same batch merges the fragment into 原神, then the delete is allowed.
    batch = "\n".join([
        json.dumps(
            {"category": "common", "entry": "原神", "op": "append_lines",
             "section": "角色", "content": "ブレンニ|布伦妮|||风元素法器角色", "reason": "测试"},
            ensure_ascii=False,
        ),
        deletion,
    ])
    report = apply_knowledge_proposals(batch, knowledge_root=tmp_path, commit=False)
    assert [r.op for r in report.applied] == ["append_lines", "delete_entry"]
    assert not (tmp_path / "common" / "布伦妮.md").exists()
    assert all(e.key != "布伦妮" for e in load_index_entries(tmp_path, "common"))


def test_rename_entry_moves_file_and_index(tmp_path) -> None:
    apply_knowledge_proposals(_proposal(), knowledge_root=tmp_path, commit=False)
    report = apply_knowledge_proposals(
        json.dumps(
            {"category": "streamer", "entry": "星野灯", "op": "rename_entry",
             "new_key": "星野あかり", "reason": "测试"},
            ensure_ascii=False,
        ),
        knowledge_root=tmp_path, commit=False,
    )
    assert report.applied and "renamed" in report.applied[0].reason
    assert not (tmp_path / "streamer" / "星野灯.md").exists()
    text = (tmp_path / "streamer" / "星野あかり.md").read_text(encoding="utf-8")
    assert text.startswith("# 星野あかり")
    keys = [e.key for e in load_index_entries(tmp_path, "streamer")]
    assert keys == ["星野あかり"]


def test_entry_type_enum_is_enforced(tmp_path) -> None:
    report = apply_knowledge_proposals(
        json.dumps(
            {"category": "common", "entry": "寿命论", "op": "create_entry",
             "entry_type": "梗 / 社区用语", "intro": "测试", "reason": "测试"},
            ensure_ascii=False,
        ),
        knowledge_root=tmp_path, commit=False,
    )
    assert not report.applied
    assert "entry_type" in report.skipped[0].reason
