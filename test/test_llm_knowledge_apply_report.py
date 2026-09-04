"""Apply-report consistency (kb-followups plan A4): verdicts come from the
engine result — a rejected op can no longer be reported applied — and alias
item changes driven by a term-line ``update`` get their own report lines."""

from __future__ import annotations

import json

from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _seed(repo: KnowledgeRepo) -> None:
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": ["角色"]},
        )
        txn.create_node("T1", "term", {"surface": "ラウマ", "zh": "菈乌玛", "desc": "角色"})
        txn.create_membership("M0", "S", "T1", "角色", 0)
        txn.create_item("I1", "T1", "aliases", "Rauma")


def _proposals(*rows: dict) -> str:
    return (
        "<knowledge_proposals>\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + "\n</knowledge_proposals>"
    )


def _term_handle(repo: KnowledgeRepo, handles: HandleMap) -> str:
    text = repo.entry_prompt_text("S", handles)
    line = next(l for l in text.splitlines() if "ラウマ" in l)
    return line.rsplit("<!-- ", 1)[1].split()[0]


def test_rejected_op_never_reported_applied(tmp_path) -> None:
    """An op the engine rejects (duplicate alias item) lands in skipped with
    the engine's reason — the old translate-time 'applied' record is gone."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo)
    handles = HandleMap()
    handle = _term_handle(repo, handles)
    report = apply_model_proposals(
        _proposals({"op": "add_item", "id": handle, "field": "aliases", "value": "Rauma", "reason": "dup"}),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    assert not report.rolled_back
    assert [r.op for r in report.applied] == []
    assert any(r.op == "add_item" and r.status == "skipped" for r in report.skipped)


def test_three_segment_term_line_is_refused_not_noted(tmp_path) -> None:
    """Plan A1: an exactly-three-segment term line (almost always a term
    missing its alias column) skips with an actionable reason on both write
    paths — never silently downgraded to a note."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo)
    handles = HandleMap()
    handle = _term_handle(repo, handles)
    report = apply_model_proposals(
        _proposals(
            {"op": "update", "id": handle, "line": "ラウマ|菈乌玛|新描述。", "reason": "r"},
            {"op": "append_lines", "category": "common", "entry": "原神", "section": "角色",
             "content": "ノルム|诺姆|补给站老板。", "reason": "r"},
        ),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    assert report.applied == []
    assert len(report.skipped) == 2
    assert all("四列" in r.reason for r in report.skipped)
    # nothing landed as a note either
    assert list(repo.store.nodes_of_kind("note")) == []


def test_update_reports_alias_add_and_remove_lines(tmp_path) -> None:
    """A term-line rewrite that changes the alias column reports every alias
    add/remove as its own line — never again a silent deletion behind one
    'line rewritten' row."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo)
    handles = HandleMap()
    handle = _term_handle(repo, handles)
    report = apply_model_proposals(
        _proposals({"op": "update", "id": handle,
                    "line": "ラウマ|菈乌玛|Lauma|新描述。", "reason": "候选1"}),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    assert not report.rolled_back
    reasons = [r.reason for r in report.applied]
    assert "line rewritten" in reasons
    assert "alias added: Lauma" in reasons
    assert "alias removed: Rauma" in reasons
    values = {
        (item.field, item.value)
        for item in repo.store.items_of("T1")
    }
    assert ("aliases", "Lauma") in values and ("aliases", "Rauma") not in values
