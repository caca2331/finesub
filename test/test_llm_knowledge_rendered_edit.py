"""rendered/ as an editable projection (plan §11.3): diff-faced refresh with
per-file manifest, dirty-file protection, and the run-start harvest."""

from __future__ import annotations

import json

import pytest

from finesub.llm.knowledge.node.edit import (
    EditError,
    edit_subject,
    harvest_rendered_edits,
    harvest_rendered_edits_at_run_start,
)
from finesub.llm.knowledge.node.repo import MANIFEST_FILENAME, RENDERED_DIRNAME, KnowledgeRepo


def _repo(tmp_path):  # type: ignore[no-untyped-def]
    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "游戏X", "intro": "一个游戏", "category": "common", "section_order": ["角色"]},
        )
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "desc": "主角"})
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_node(
            "S2", "subject",
            {"surface": "游戏Y", "intro": "另一个", "category": "common", "section_order": []},
        )
    return repo


def test_refresh_is_diff_faced_and_manifest_backed(tmp_path) -> None:
    repo = _repo(tmp_path)
    first = repo.refresh_rendered()
    assert "common/游戏X.md" in first.written and "common/游戏Y.md" in first.written
    manifest = json.loads((tmp_path / RENDERED_DIRNAME / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["files"]["common/游戏X.md"]["subject"] == "S"
    # nothing changed: nothing rewritten
    second = repo.refresh_rendered()
    assert second.written == [] and second.dirty == []
    # one subject changes: only its file is rewritten
    with repo.store.begin("user") as txn:
        txn.update_node("S2", payload={"surface": "游戏Y", "intro": "改了", "category": "common", "section_order": []})
    third = repo.refresh_rendered()
    assert third.written == ["common/游戏Y.md"]


def test_user_edited_file_is_never_clobbered(tmp_path) -> None:
    repo = _repo(tmp_path)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    edited = path.read_text(encoding="utf-8").replace("アリス|爱丽丝||主角", "アリス|爱丽丝||女主角")
    path.write_text(edited, encoding="utf-8")
    # a store change elsewhere + refresh must leave the edited file alone
    with repo.store.begin("user") as txn:
        txn.update_node("S2", payload={"surface": "游戏Y", "intro": "改了", "category": "common", "section_order": []})
    report = repo.refresh_rendered()
    assert "common/游戏X.md" in report.dirty and path.read_text(encoding="utf-8") == edited
    assert repo.rendered_dirty_files() == [("common/游戏X.md", path, "S")]


def test_harvest_applies_clean_edit_as_user_revision(tmp_path) -> None:
    repo = _repo(tmp_path)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("アリス|爱丽丝||主角", "アリス|爱丽丝||女主角"),
        encoding="utf-8",
    )
    results = harvest_rendered_edits(repo)
    assert [r["status"] for r in results] == ["applied"]
    assert repo.store.node("T").payload["desc"] == "女主角"
    revision = repo.store.conn.execute(
        "SELECT kind, note FROM revisions ORDER BY rev DESC LIMIT 1"
    ).fetchone()
    assert revision["kind"] == "user" and revision["note"].startswith("rendered-edit:")
    # the user apply endorsed the value it set (plan §11.4 / O8)
    row = repo.store.conn.execute(
        "SELECT evidence_kind, verdict FROM evidence WHERE field_path='payload.desc'"
    ).fetchone()
    assert row is not None and row["evidence_kind"] == "user" and row["verdict"] == "confirmed"
    # the file is back to canonical render and no longer dirty
    assert repo.rendered_dirty_files() == []
    assert "アリス|爱丽丝||女主角" in path.read_text(encoding="utf-8")


def test_harvest_isolates_a_broken_file(tmp_path) -> None:
    repo = _repo(tmp_path)
    repo.refresh_rendered()
    good = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    bad = tmp_path / RENDERED_DIRNAME / "common" / "游戏Y.md"
    good.write_text(
        good.read_text(encoding="utf-8").replace("アリス|爱丽丝||主角", "アリス|爱丽丝||女主角"),
        encoding="utf-8",
    )
    bad.write_text("完全不是条目格式\n", encoding="utf-8")
    results = {r["file"]: r["status"] for r in harvest_rendered_edits(repo)}
    assert results["common/游戏X.md"] == "applied"
    assert results["common/游戏Y.md"] == "failed"
    # the broken file survives untouched and stays dirty for the human
    assert bad.read_text(encoding="utf-8") == "完全不是条目格式\n"
    assert [rel for rel, _, _ in repo.rendered_dirty_files()] == ["common/游戏Y.md"]


def test_a_broken_file_says_so_out_loud(tmp_path, reported) -> None:
    """The skip has to reach a person.

    Isolation alone is not the contract: a hand edit that silently does
    nothing is indistinguishable from one that was absorbed. The CLIs that
    trigger this bind a terminal reporter for exactly this warning -- until
    2026-09-03 they did not, and the run exited 0 saying nothing.
    """

    repo = _repo(tmp_path)
    repo.refresh_rendered()
    (tmp_path / RENDERED_DIRNAME / "common" / "游戏Y.md").write_text(
        "完全不是条目格式\n", encoding="utf-8"
    )

    harvest_rendered_edits(repo)

    assert "knowledge-rendered-edit-failed" in reported.codes()
    assert "common/游戏Y.md" in reported.joined()


def test_the_correction_cli_binds_a_reporter_before_it_runs(monkeypatch) -> None:
    """`main` is a binding wrapper, and the harvest runs under it.

    Asserted at the entry point rather than on the `__main__` guard: the
    knowledge CLI had its binding in that guard and it never ran, because
    `python -m finesub.llm.knowledge` enters through the package's
    `__main__.py` and calls `main()` directly.
    """

    from finesub.reporting import current_reporter, reporter_delivers
    from finesub.llm import correction_translation

    seen: list[bool] = []
    monkeypatch.setattr(
        correction_translation,
        "_main",
        lambda: seen.append(reporter_delivers(current_reporter())) or 0,
    )
    assert correction_translation.main() == 0
    assert seen == [True]


def test_run_start_harvest_respects_worktree_gate(tmp_path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("アリス|爱丽丝||主角", "アリス|爱丽丝||女主角"),
        encoding="utf-8",
    )
    # simulate running inside a linked worktree without the opt-in: skip
    monkeypatch.setattr("finesub.paths.is_linked_worktree", lambda: True)
    monkeypatch.delenv("FINESUB_KNOWLEDGE_WRITE", raising=False)
    assert harvest_rendered_edits_at_run_start(tmp_path) == []
    assert repo.store.node("T").payload["desc"] == "主角"
    # with the opt-in the same call harvests
    monkeypatch.setenv("FINESUB_KNOWLEDGE_WRITE", "1")
    results = harvest_rendered_edits_at_run_start(tmp_path)
    assert [r["status"] for r in results] == ["applied"]
    assert repo.store.node("T").payload["desc"] == "女主角"


def test_harvest_parks_a_line_it_cannot_classify(tmp_path, reported) -> None:
    """One bad line must not cost the user every other edit in the file.

    Nobody is watching a run-start harvest, so the deterministic answer is the
    importer's (owner 2026-09-01): keep the bytes, park them in the staging
    section, let the existing `repair` candidate flow do the judgement call.
    Rejecting the file instead threw the whole edit away and told a terminal
    that had already scrolled past.
    """

    repo = _repo(tmp_path)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "アリス|爱丽丝||主角",
            "アリス|爱丽丝||女主角\n- ボブ|鲍勃|一个配角",   # three columns: no alias slot
        ),
        encoding="utf-8",
    )

    results = harvest_rendered_edits(repo)

    assert [r["status"] for r in results] == ["applied"]
    assert results[0]["staged"] == ["ボブ|鲍勃|一个配角"]
    # the good edit in the same file still landed
    assert repo.store.node("T").payload["desc"] == "女主角"
    assert "knowledge-rendered-edit-staged" in reported.codes()
    # parked verbatim, and visible as a repair candidate by construction
    text = path.read_text(encoding="utf-8")
    assert "ボブ|鲍勃|一个配角" in text
    assert repo.rendered_dirty_files() == []

    from finesub.llm.knowledge.node.scan import scan_candidates

    staged = [
        c for c in scan_candidates(repo.store, repo.rev).candidates
        if c.get("kind") == "staging-line"
    ]
    assert [c["line"] for c in staged] == ["ボブ|鲍勃|一个配角"]


def test_the_interactive_editor_still_refuses_the_same_line(tmp_path) -> None:
    """`knowledge edit` keeps rejecting: the human is right there.

    Parking is for the unattended path. In the editor the error names the line
    you just typed, and silently moving it to 待归类 would be the worse answer.
    """

    repo = _repo(tmp_path)
    repo.refresh_rendered()
    subject = repo.store.node("S")
    text = (tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md").read_text(encoding="utf-8")
    broken = text.replace("アリス|爱丽丝||主角", "ボブ|鲍勃|一个配角")

    with pytest.raises(EditError):
        edit_subject(repo, subject, broken)

    # …and the same text parks instead when the caller asks for it
    report = edit_subject(repo, subject, broken, on_invalid="stage")
    assert report.staged == ["ボブ|鲍勃|一个配角"]


def test_every_preset_can_park(tmp_path) -> None:
    """A preset without a staging section silently falls back to rejecting.

    That is the failure this change is trying to remove, and it would come
    back the moment someone adds a preset without one -- `common` had no
    staging section until 2026-09-03.
    """

    from finesub.llm.knowledge.node.presets import preset_for_category

    for category in ("common", "streamer", "style"):
        assert preset_for_category(category).staging_section() is not None


def test_a_parked_line_is_not_parked_again(tmp_path, reported) -> None:
    """The staging section is a destination, not another thing to scan.

    Found by probe 2026-09-03: the pre-pass walked every section including
    the staging one, so a parked three-column row classified as `invalid`
    again on the next harvest of the same file. It was re-parked each time —
    the same warning about nothing new, a retire+create that handed the line a
    NEW node id (dropping any `repair` verdict recorded against the old one),
    and a revision with no semantic content. Three harvests, three node ids.
    """

    repo = _repo(tmp_path)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "アリス|爱丽丝||主角", "アリス|爱丽丝||主角\n- ボブ|鲍勃|一个配角"
        ),
        encoding="utf-8",
    )
    first = harvest_rendered_edits(repo)
    assert first[0]["staged"] == ["ボブ|鲍勃|一个配角"]
    parked = [
        m.child_id for m in repo.store.children("S", repo.rev) if m.section == "待归类"
    ]
    assert len(parked) == 1

    # an unrelated edit to the same file, later
    path.write_text(
        path.read_text(encoding="utf-8").replace("アリス|爱丽丝||主角", "アリス|爱丽丝||女主角"),
        encoding="utf-8",
    )
    reported.warnings.clear()
    second = harvest_rendered_edits(repo)

    assert second[0]["staged"] == []
    assert "knowledge-rendered-edit-staged" not in reported.codes()
    # same node, still there exactly once — a verdict against it would survive
    assert [
        m.child_id for m in repo.store.children("S", repo.rev) if m.section == "待归类"
    ] == parked
    assert repo.store.node("T").payload["desc"] == "女主角"


def test_a_line_written_into_staging_by_hand_becomes_a_note(tmp_path) -> None:
    """Whatever you put in the parking lot is a note, by construction.

    Without this the harvest would raise on it — the section declares `term`
    among its body kinds, so a three-column row is a format error there too.
    """

    repo = _repo(tmp_path)
    repo.refresh_rendered()
    path = tmp_path / RENDERED_DIRNAME / "common" / "游戏X.md"
    # The entry carries an explicit section_order, so the empty staging section
    # is not rendered into the file — the user writes the heading themselves.
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("## 元数据", "## 待归类\n\n- ボブ|鲍勃|一个配角\n\n## 元数据"),
        encoding="utf-8",
    )

    results = harvest_rendered_edits(repo)

    assert [r["status"] for r in results] == ["applied"]
    # it went in as itself, not through the staging pass
    assert results[0]["staged"] == []
    parked = [m for m in repo.store.children("S", repo.rev) if m.section == "待归类"]
    assert len(parked) == 1
    assert repo.store.node(parked[0].child_id).kind == "note"
