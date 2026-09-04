"""Prompt-layer guards for the knowledge sessions (kb-followups plan A2/A3):
strict template loading — placeholder coverage checked on the TEMPLATE's
declared identifiers, never by scanning rendered text (injected material may
legally contain ``$WORD``) — and parser↔template enum consistency."""

from __future__ import annotations

import pytest

from finesub.llm.knowledge.node.repair import render_repair_prompt
from finesub.llm.knowledge.node.presets import load_preset
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.share.review import REVIEW_VERDICTS, render_review_prompt
from finesub.llm.knowledge.verify import render_verify_prompt
from finesub.llm.prompt_compose import (
    PROMPT_TEMPLATE_DIR,
    PromptAssemblyError,
    load_prompt_template,
)
from finesub.llm.prompts import build_knowledge_update_messages


def _seed_minimal(repo: KnowledgeRepo) -> None:
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S", "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": ["角色"]},
        )
        txn.create_node("T1", "term", {"surface": "ラウマ", "zh": "菈乌玛", "desc": "角色"})
        txn.create_membership("M0", "S", "T1", "角色", 0)


def test_strict_load_refuses_missing_placeholder() -> None:
    """The exact defect this guards against: ``fragment_knowledge_output_v1``
    loaded without ``reasoning_clause`` shipped a literal ``$reasoning_clause``
    (and no <reasoning> demand) to six repair rounds."""

    with pytest.raises(PromptAssemblyError, match="reasoning_clause"):
        load_prompt_template("fragment_knowledge_output_v1.md", strict=True)


def test_repair_prompt_fully_rendered_and_tolerates_dollar_content(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    _seed_minimal(repo)
    prompt = render_repair_prompt(repo, "S", material="素材里合法出现 $PATH 与 ${HOME}")
    # every declared placeholder is filled...
    assert "$reasoning_clause" not in prompt and "$entry_text" not in prompt
    # ...the opening-<reasoning> demand actually made it into the ops contract
    assert "<reasoning>" in prompt
    # entries are presented under the block name the ops contract references
    assert "<kb_entries>" in prompt
    # strictness lives on the template layer: untrusted $WORD passes through
    assert "$PATH" in prompt and "${HOME}" in prompt


def test_verify_review_and_update_prompts_render() -> None:
    verify = render_verify_prompt(
        [{"claim_id": "c1", "subject": "原神", "kind": "term", "line": "ラウマ|菈乌玛|角色"}]
    )
    assert "c1" in verify and "$judgment" not in verify
    review = render_review_prompt({"bundle": {}})
    assert "<review_verdict>" in review and "$bundle_text" not in review
    messages = build_knowledge_update_messages(
        refined=False, task_summary="s", window_packs="w"
    )
    joined = "\n".join(m["content"] for m in messages)
    assert "$reasoning_clause" not in joined and "<knowledge_proposals>" in joined


def test_ops_contract_worked_example_passes_the_real_machinery(tmp_path) -> None:
    """A5 guard: the fewshot example in ``fragment_knowledge_output_v1`` runs
    through the REAL parse→translate→apply path against a repo seeded to match
    the excerpt — the example cannot drift from the code. Counter-examples are
    asserted refused."""

    from finesub.llm.knowledge.node.importer import classify_line
    from finesub.llm.knowledge.node.proposals import MODEL_OPS, apply_model_proposals, parse_model_proposals
    from finesub.llm.knowledge.node.render import HandleMap as NodeHandleMap

    text = (PROMPT_TEMPLATE_DIR / "fragment_knowledge_output_v1.md").read_text(encoding="utf-8")
    # the EXAMPLE block is the last <knowledge_proposals> in the file (the
    # first mention is inline prose in the format rules)
    start = text.rindex("<knowledge_proposals>")
    end = text.index("</knowledge_proposals>", start) + len("</knowledge_proposals>")
    block = text[start:end]
    proposals = parse_model_proposals(block)
    assert len(proposals) == 4 and all(p.get("op") in MODEL_OPS for p in proposals)
    for p in proposals:
        for line in filter(None, [p.get("line"), *str(p.get("content", "")).splitlines()]):
            spec = load_preset("common").section("角色")  # a free 分类节: term bodies
            kind, _payload = classify_line(str(line), spec)
            assert kind == "term", line

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "开放世界游戏。",
                                         "category": "common", "section_order": ["角色"]})
        txn.create_node("T1", "term", {"surface": "ヴィリーナ", "zh": "维琳娜", "desc": "Ver.3.0 新登场代理人。"})
        txn.create_node("T2", "term", {"surface": "ノルム", "zh": "诺姆", "desc": "补给站老板。"})
        txn.create_membership("M1", "S", "T1", "角色", 0)
        txn.create_membership("M2", "S", "T2", "角色", 1)
        txn.create_item("I1", "T1", "aliases", "Vilina")
    handles = NodeHandleMap()
    repo.entry_prompt_text("S", handles)
    by_id = {ident: h for h, (ident, _) in handles.nodes.items()}
    live = block.replace("@k8", by_id["T1"]).replace("@k9", by_id["T2"]).replace("@k1", by_id["S"])
    report = apply_model_proposals(live, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles)
    assert not report.rolled_back and report.skipped == [] and report.applied

    # counter-examples from the 反例 section really are refused
    assert classify_line("ヴィリーナ|维琳娜|新描述。", load_preset("common").section("角色"))[0] == "invalid"
    bad = ('<knowledge_proposals>\n{"op":"create_entry","category":"streamer|游戏",'
           '"entry":"X","intro":"x","reason":"r"}\n</knowledge_proposals>')
    refused = apply_model_proposals(bad, repo=repo, task_id="t", knowledge_read_rev=repo.rev)
    assert refused.applied == [] and "category" in refused.skipped[0].reason


def test_review_verdict_enum_matches_template_schema() -> None:
    """The parser enum is the truth source; the schema block the model copies
    must list every member — a prose-only member is dead in practice (the
    model copies the schema, which is how approve_tentative went dark)."""

    text = (PROMPT_TEMPLATE_DIR / "share_review_v1.md").read_text(encoding="utf-8")
    schema = text[text.rindex("<review_verdict>"):]
    for verdict in REVIEW_VERDICTS:
        assert f'"{verdict}"' in schema, f"schema block missing {verdict!r}"


def test_output_blocks_survive_being_named_inside_reasoning() -> None:
    """Live repair session (2026-08-29): the model wrote
    "候选不判 dismiss，<knowledge_proposals> 输出空块" INSIDE <reasoning>, and the
    non-greedy match opened there — swallowing </reasoning> and the real
    opening tag as body lines, which then reported as invalid JSON."""

    from finesub.llm.knowledge.node.proposals import parse_model_proposals
    from finesub.llm.knowledge.node.repair import parse_candidate_verdicts

    text = (
        "<reasoning>\n候选不判 dismiss，<knowledge_proposals> 输出空块；"
        "<candidate_verdicts> 里写 needs_human。\n</reasoning>\n"
        '<knowledge_proposals>\n{"op":"remove","id":"@k2","reason":"r"}\n</knowledge_proposals>\n'
        '<candidate_verdicts>\n{"candidate":"@c1","verdict":"needs_human",'
        '"reason":"查不到官方译名","missing":"可核实来源"}\n</candidate_verdicts>\n'
    )
    assert parse_model_proposals(text) == [{"op": "remove", "id": "@k2", "reason": "r"}]
    verdicts = parse_candidate_verdicts(text, {"@c1"})
    assert [v["verdict"] for v in verdicts] == ["needs_human"]
