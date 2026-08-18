from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from finesub.llm.knowledge.snapshot import KnowledgeSnapshot, KnowledgeSnapshotError


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _knowledge_repo(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir()
    (root / "streamer" / "index.md").write_text(
        "- Foo | フー | foo酱 | first entry\n", encoding="utf-8"
    )
    (root / "streamer" / "Foo.md").write_text(
        "# Foo\n\nKnows Bar.\n", encoding="utf-8"
    )
    (root / "common" / "index.md").write_text(
        "- Bar [人物] | バー | B | second entry\n", encoding="utf-8"
    )
    (root / "common" / "Bar.md").write_text("# Bar\n\nOriginal.\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root


def test_snapshot_reads_fixed_commit_after_live_tree_changes(tmp_path) -> None:
    root = _knowledge_repo(tmp_path)
    snapshot = KnowledgeSnapshot.capture(root)

    assert snapshot.describe() == {
        "snapshot_identity": snapshot.identity,
        "categories": ["streamer", "common"],
        "entry_counts": {"streamer": 1, "common": 1},
        "read_only": True,
    }
    assert snapshot.read("foo酱")["content"].endswith("Knows Bar.\n")
    assert snapshot.read("common/B")["key"] == "Bar"
    assert snapshot.read_many(["Foo", "foo酱", "missing"])["missing"] == [
        "missing"
    ]
    assert snapshot.search("second")["entries"][0]["key"] == "Bar"
    assert snapshot.list(prefix="F")["entries"][0]["key"] == "Foo"
    assert snapshot.references("Bar")["references"][0]["key"] == "Foo"

    (root / "common" / "Bar.md").write_text("# Bar\n\nChanged live.\n", encoding="utf-8")
    assert "Original." in snapshot.read("Bar")["content"]
    assert snapshot.read("Bar")["snapshot_identity"] == snapshot.identity


def test_capture_fails_closed_on_dirty_or_non_git_knowledge(tmp_path) -> None:
    root = _knowledge_repo(tmp_path)
    (root / "untracked.md").write_text("not committed", encoding="utf-8")
    with pytest.raises(KnowledgeSnapshotError, match="uncommitted"):
        KnowledgeSnapshot.capture(root)
    with pytest.raises(KnowledgeSnapshotError, match="not an embedded git"):
        KnowledgeSnapshot.capture(tmp_path / "not-git")


def test_old_snapshot_remains_readable_after_a_new_commit(tmp_path) -> None:
    root = _knowledge_repo(tmp_path)
    old = KnowledgeSnapshot.capture(root)
    (root / "common" / "Bar.md").write_text("# Bar\n\nNew commit.\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "update")
    new = KnowledgeSnapshot.capture(root)

    assert old.identity != new.identity
    assert "Original." in old.read("Bar")["content"]
    assert "New commit." in new.read("Bar")["content"]
