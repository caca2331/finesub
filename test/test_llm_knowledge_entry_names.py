"""Entry names and the files they render to (security assessment 2026-09-25,
F1; relaxed 2026-10-05). A name may carry path characters -- ``Re:ゼロ``,
``Fate/stay night`` -- because the file name is derived from it: path
characters become full-width look-alikes, and what is still no file name falls
back to the entry id. Whatever the name, the file lands inside ``rendered/``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.knowledge.node.edit import EditError, edit_subject
from finesub.llm.knowledge.node.model import entry_name_problem
from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.share.exchange import (
    SNAPSHOT_SCHEMA_VERSION,
    build_push_bundle,
    chain_hash,
    content_digest,
    strip_local_maps,
    validate_bundle,
)
from finesub.llm.knowledge.share.sync import apply_snapshot

TRAVERSALS = ("../../../AGENTS", "..\\..\\..\\AGENTS", "C:\\Users\\x\\AGENTS", "/etc/AGENTS")


@pytest.mark.parametrize("name", ["", " 前后空格 ", "x" * 101, "a\x07b", "a|b", "a\u2028b", "a\x85b", "style/xx", "common/游戏"])
def test_names_the_rule_refuses(name: str) -> None:
    assert entry_name_problem(name) is not None


@pytest.mark.parametrize(
    "name",
    ["原神·角色", "崩坏：星穹铁道", "Re:ゼロ", "Fate/stay night", "CON", ".", "x" * 100, "a.b", *TRAVERSALS],
)
def test_ordinary_and_path_like_names_pass(name: str) -> None:
    assert entry_name_problem(name) is None


@pytest.mark.parametrize(
    ("name", "file"),
    [
        ("Re:ゼロから始める異世界生活", "Re：ゼロから始める異世界生活.md"),
        ("Fate/stay night", "Fate／stay night.md"),
        ("../../../AGENTS", "..／..／..／AGENTS.md"),
        ("C:\\Users\\x\\AGENTS", "C：＼Users＼x＼AGENTS.md"),
    ],
)
def test_path_characters_render_as_look_alikes_inside_rendered(tmp_path, name, file) -> None:
    victim = tmp_path / "repo" / "AGENTS.md"
    victim.parent.mkdir(parents=True)
    victim.write_text("ORIGINAL", encoding="utf-8")
    root, _subject_id = _repo_with_legacy_subject(tmp_path, name)
    report = KnowledgeRepo.open(root).refresh_rendered()
    assert report.written == [f"streamer/{file}"]
    assert f"# {name}" in (root / "rendered" / "streamer" / file).read_text(encoding="utf-8")
    assert victim.read_text(encoding="utf-8") == "ORIGINAL"


@pytest.mark.parametrize("name", ["CON", "nul", "com1.x", "COM¹", ".", ".."])
def test_what_is_still_no_file_name_renders_under_its_id(tmp_path, name) -> None:
    root, subject_id = _repo_with_legacy_subject(tmp_path, name)
    report = KnowledgeRepo.open(root).refresh_rendered()
    assert report.written == [f"streamer/{subject_id}.md"]


def test_a_file_name_already_taken_falls_back_to_the_id(tmp_path) -> None:
    """The category's `index.md`, and two names one file system cannot tell
    apart (only a legacy row can get past the name dedupe)."""

    root = tmp_path / "repo" / "knowledge"
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node("I", "subject", {"surface": "index", "intro": "x", "category": "streamer"})
        txn.create_node("A", "subject", {"surface": "Re:ゼロ", "intro": "x", "category": "streamer"})
        txn.create_node("B", "subject", {"surface": "Re：ゼロ", "intro": "x", "category": "streamer"})
    report = KnowledgeRepo.open(root).refresh_rendered()
    files = sorted(report.written)
    assert "streamer/I.md" in files
    assert not (root / "rendered" / "streamer" / "index.md").read_text(encoding="utf-8").startswith("# index")
    assert ("streamer/Re：ゼロ.md" in files) and (("streamer/A.md" in files) != ("streamer/B.md" in files))


def _repo_with_legacy_subject(tmp_path: Path, surface: str) -> tuple[Path, str]:
    """A subject stored before the rule: the store itself loads with
    strict=False, so this is how an old import or pull looks on disk."""

    root = tmp_path / "repo" / "knowledge"
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "LEGACY", "subject",
            {"surface": surface, "intro": "INJECTED", "category": "streamer"},
            visibility="shareable",
        )
    return root, "LEGACY"


def test_a_snapshot_with_a_path_for_a_name_stays_inside_rendered(tmp_path) -> None:
    root = tmp_path / "repo" / "knowledge"
    victim = tmp_path / "repo" / "AGENTS.md"
    victim.parent.mkdir(parents=True)
    victim.write_text("ORIGINAL", encoding="utf-8")
    repo = KnowledgeRepo.open(root)
    corpus = "c" * 32
    content = {
        "schema": SNAPSHOT_SCHEMA_VERSION, "items": [], "memberships": [], "links": [], "redirects": {},
        "nodes": [{"canonical_id": "n1", "kind": "subject", "retired": False, "maturity": "normal",
                   "payload": {"surface": "../../../AGENTS", "intro": "x", "category": "streamer"}}],
    }
    digest_value = content_digest(content)
    head = chain_hash(corpus, digest_value)
    snapshot = {
        "corpus_id": corpus, "server_rev": 1, "content": content, "content_digest": digest_value,
        "chain_hash": head, "history": [{"rev": 1, "content_digest": digest_value, "chain_hash": head}],
    }
    apply_snapshot(repo.store, snapshot, remote=corpus)
    report = KnowledgeRepo.open(root).refresh_rendered()
    assert "streamer/..／..／..／AGENTS.md" in report.written
    assert victim.read_text(encoding="utf-8") == "ORIGINAL"


def test_a_pushed_bundle_with_a_bad_name_is_refused(tmp_path) -> None:
    root, subject_id = _repo_with_legacy_subject(tmp_path, "a|b")
    repo = KnowledgeRepo.open(root)
    wire = strip_local_maps(build_push_bundle(repo.store, [subject_id], idempotency_key="k1"))
    assert any("entry name" in problem for problem in validate_bundle(wire))


def test_a_model_rename_to_a_path_lands_inside_rendered(tmp_path) -> None:
    root = tmp_path / "knowledge"
    repo = KnowledgeRepo.open(root)
    create = {"op": "create_entry", "category": "common", "entry": "崩铁", "intro": "游戏",
              "entry_type": "游戏", "reason": "新条目"}
    rename = {"op": "rename_entry", "category": "common", "entry": "崩铁",
              "new_key": "..\\..\\..\\victim\\AGENTS", "reason": "r"}
    text = "<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in (create, rename)) + "\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=HandleMap()).to_dict()
    assert [row["op"] for row in report["applied"]] == ["create_entry", "rename_entry"]
    assert (root / "rendered" / "common" / "..＼..＼..＼victim＼AGENTS.md").is_file()
    assert not (tmp_path / "victim").exists()


def test_an_edited_h1_the_rule_refuses_is_rejected(tmp_path) -> None:
    root = tmp_path / "knowledge"
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "星野灯", "intro": "VTuber", "category": "streamer"})
    repo = KnowledgeRepo.open(root)
    subject = repo.store.node("S")
    text = repo.entry_text("S").replace("# 星野灯", "# 星野|灯", 1)
    with pytest.raises(EditError, match="entry name"):
        edit_subject(repo, subject, text)


def test_a_legacy_name_the_rule_refuses_renders_under_its_id(tmp_path) -> None:
    root, subject_id = _repo_with_legacy_subject(tmp_path, "a|b")
    report = KnowledgeRepo.open(root).refresh_rendered()
    assert report.written == [f"streamer/{subject_id}.md"]
    assert "# a|b" in (root / "rendered" / "streamer" / f"{subject_id}.md").read_text(encoding="utf-8")


def test_a_long_cjk_name_renders_under_its_id(tmp_path) -> None:
    """100 characters is a legal name, but 100 CJK characters are 300 UTF-8
    bytes -- past the 255-byte file-name limit of ext4 and APFS."""

    name = "字" * 100
    assert entry_name_problem(name) is None
    root, subject_id = _repo_with_legacy_subject(tmp_path, name)
    report = KnowledgeRepo.open(root).refresh_rendered()
    assert report.written == [f"streamer/{subject_id}.md"]


def test_a_manifest_row_outside_rendered_is_not_harvested(tmp_path) -> None:
    root = tmp_path / "repo" / "knowledge"
    repo = KnowledgeRepo.open(root)
    repo.refresh_rendered()
    outside = tmp_path / "repo" / "AGENTS.md"
    outside.write_text("edited by someone", encoding="utf-8")
    manifest_path = root / "rendered" / ".manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["streamer/../../../AGENTS.md"] = {"sha256": "0" * 64, "subject": "S"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert KnowledgeRepo.open(root).rendered_dirty_files() == []


def test_a_name_with_a_slash_is_looked_up_whole(tmp_path) -> None:
    """`<类别>/<名字>` only when the prefix is a category: `Fate/stay night`
    is a name, in the human lookup, the snapshot read and the agent read."""

    from finesub.llm.knowledge.snapshot import KnowledgeSnapshot

    root = tmp_path / "knowledge"
    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node("F", "subject", {"surface": "Fate/stay night", "intro": "游戏", "category": "common", "entry_type": "游戏"})
    repo = KnowledgeRepo.open(root)
    assert repo.resolve_qualified("Fate/stay night").subject_id == "F"
    assert repo.resolve_qualified("common/Fate/stay night").subject_id == "F"
    KnowledgeRepo.forget(root)
    assert KnowledgeSnapshot.capture(root).read("Fate/stay night")["key"] == "Fate/stay night"
