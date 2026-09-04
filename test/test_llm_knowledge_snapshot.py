from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.snapshot import KnowledgeSnapshot, KnowledgeSnapshotError


def _seed(root: Path) -> None:
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text("- 主播A |  | エーちゃん | 测试主播\n", encoding="utf-8")
    (root / "streamer" / "主播A.md").write_text(
        "# 主播A\n测试主播\n\n## 档案\n本名: 主播A\n别名: エーちゃん\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (root / "common" / "index.md").write_text("- 游戏X [游戏] |  |  | 一个游戏\n", encoding="utf-8")
    (root / "common" / "游戏X.md").write_text(
        "# 游戏X\n一个游戏\n\n## 档案\n本名: 游戏X\n\n## 角色\n\nA|甲|||主播A 常玩\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )


def _append(root: Path, line: str) -> None:
    repo = KnowledgeRepo.open(root)
    text = "<knowledge_proposals>\n" + json.dumps(
        {"op": "append_lines", "category": "common", "entry": "游戏X", "section": "角色", "content": line, "reason": "r"},
        ensure_ascii=False,
    ) + "\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev)
    assert report.rev is not None


def test_snapshot_reads_its_pinned_revision_after_later_writes(tmp_path) -> None:
    _seed(tmp_path)
    snapshot = KnowledgeSnapshot.capture(tmp_path)
    assert snapshot.rev == 1 and snapshot.identity.startswith("rev:1:index:")
    _append(tmp_path, "B|乙||后来加的")
    assert KnowledgeRepo.open(tmp_path).rev == 2
    content = snapshot.read("游戏X")["content"]
    assert "A|甲||主播A 常玩" in content and "B|乙" not in content
    later = KnowledgeSnapshot.capture(tmp_path)
    assert later.rev == 2 and "B|乙||后来加的" in later.read("common/游戏X")["content"]
    assert KnowledgeSnapshot.at(tmp_path, 1).read("游戏X")["content"] == content
    with pytest.raises(KnowledgeSnapshotError):
        KnowledgeSnapshot.at(tmp_path, 9)


def test_capture_fails_closed_on_missing_root(tmp_path) -> None:
    with pytest.raises(KnowledgeSnapshotError, match="does not exist"):
        KnowledgeSnapshot.capture(tmp_path / "nope")


def test_list_search_read_many_and_references(tmp_path) -> None:
    _seed(tmp_path)
    snapshot = KnowledgeSnapshot.capture(tmp_path)
    described = snapshot.describe()
    assert described["entry_counts"] == {"streamer": 1, "common": 1} and described["read_only"] is True
    listed = snapshot.list(prefix="主播")
    assert [row["key"] for row in listed["entries"]] == ["主播A"] and listed["next_cursor"] is None
    assert [row["key"] for row in snapshot.search("エーちゃん")["entries"]] == ["主播A"]
    many = snapshot.read_many(["エーちゃん", "主播A", "不存在"])
    assert [row["key"] for row in many["entries"]] == ["主播A"] and many["missing"] == ["不存在"]
    assert many["entries"][0]["content_digest"]
    refs = snapshot.references("主播A")
    assert [row["key"] for row in refs["referenced_by"]] == ["游戏X"]
    with pytest.raises(ValueError):
        snapshot.list(limit=0)
    with pytest.raises(KeyError):
        snapshot.read("translation/x")
