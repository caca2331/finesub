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
    """The defect this guards against: ``fragment_knowledge_output_v1`` once
    loaded without one of its values shipped the literal placeholder to six
    repair rounds."""

    with pytest.raises(PromptAssemblyError, match="term_line_rules"):
        load_prompt_template("fragment_knowledge_output_v1.md", strict=True)


def test_repair_prompt_fully_rendered_and_tolerates_dollar_content(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    _seed_minimal(repo)
    prompt = render_repair_prompt(repo, "S", material="素材里合法出现 $PATH 与 ${HOME}")
    # every declared placeholder is filled...
    assert "$term_line_rules" not in prompt and "$entry_text" not in prompt
    # ...the ops contract actually made it in
    assert "<knowledge_proposals>" in prompt
    # entries are presented under the block name the ops contract references
    assert "<kb_entries>" in prompt
    # strictness lives on the template layer: untrusted $WORD passes through
    assert "$PATH" in prompt and "${HOME}" in prompt


def test_the_ops_contract_says_what_a_rejection_costs_by_how_the_caller_applies(tmp_path) -> None:
    """The knowledge update rejects only the ops a batch-wide problem traces to; a maintenance session rolls the
    whole batch back (`apply_envelope(invalid_ops=...)`), and its prompt must not promise otherwise."""
    from finesub.llm.prompt_compose import render_knowledge_ops_contract

    repo = KnowledgeRepo.open(tmp_path)
    _seed_minimal(repo)
    repair = render_repair_prompt(repo, "S", material="素材")
    assert "全部不提交" in repair and "不影响其余行" not in repair
    update = render_knowledge_ops_contract(("common",))
    assert "不影响其余行" in update and "全部不提交" not in update


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
    assert "$writable_categories" not in joined and "<knowledge_proposals>" in joined


def test_verify_and_review_keep_what_they_checked_in_their_json() -> None:
    """With no `<reasoning>` block, the directions checked for a claim that found nothing go into the field
    the session already prints: the verify row's `note`, the review's `reason` (last, so a cut drops it
    first)."""

    verify = render_verify_prompt(
        [{"claim_id": "c1", "subject": "原神", "kind": "term", "line": "ラウマ|菈乌玛|角色"}]
    )
    assert "note 写查过哪些方向" in verify
    review = render_review_prompt({"bundle": {}})
    assert "末尾写查过哪些方向" in review


@pytest.mark.parametrize("name", ["proposals", "candidate", "conflict", "skipped", "verify", "review"])
def test_parsers_read_past_a_reasoning_block_the_model_wrote_on_its_own(name) -> None:
    """No prompt asks for `<reasoning>`, but a model may still open with one that names the answer block;
    each parser reads the real block after it."""

    from finesub.llm.knowledge.node.base_style import parse_skipped
    from finesub.llm.knowledge.node.proposals import parse_model_proposals
    from finesub.llm.knowledge.node.repair import parse_candidate_verdicts
    from finesub.llm.knowledge.share.conflict_repair import parse_conflict_verdicts
    from finesub.llm.knowledge.share.review import parse_review_verdict
    from finesub.llm.knowledge.verify import parse_verify_results

    tag, body, parse, check = {
        "proposals": ("knowledge_proposals", '{"op":"remove","id":"@k2","reason":"r"}',
                      parse_model_proposals, lambda rows: [row["id"] for row in rows] == ["@k2"]),
        "candidate": ("candidate_verdicts", '{"candidate":"@c1","verdict":"dismiss","reason":"r"}',
                      lambda text: parse_candidate_verdicts(text, {"@c1"}), lambda rows: len(rows) == 1),
        "conflict": ("conflict_verdicts", '{"conflict":"@x1","verdict":"keep_local","reason":"r"}',
                     lambda text: parse_conflict_verdicts(text, {"@x1"}), lambda rows: len(rows) == 1),
        "skipped": ("skipped", '{"label":"[约定]","section":"约定","why":"放不下","line":""}',
                    parse_skipped, lambda rows: rows[0]["why"] == "放不下"),
        "verify": ("verify_results", '[{"claim_id": "c1", "verdict": "unverifiable", "note": "查过官网"}]',
                   lambda text: parse_verify_results(text, [{"claim_id": "c1"}]),
                   lambda rows: rows[0]["note"] == "查过官网"),
        "review": ("review_verdict", '{"verdict": "reject", "merge": {}, "reason": "r", "external_evidence": []}',
                   lambda text: parse_review_verdict(text, {}), lambda data: data["verdict"] == "reject"),
    }[name]
    text = f"<reasoning>\n先想好再输出 <{tag}> 块\n</reasoning>\n<{tag}>\n{body}\n</{tag}>\n"
    assert check(parse(text)), name


def test_judgment_and_writing_rules_go_by_session(tmp_path) -> None:
    """Cold read Q7: every maintenance session gets the judgment standard;
    only the ones that write entries get the writing rules."""

    repo = KnowledgeRepo.open(tmp_path)
    _seed_minimal(repo)
    verify = render_verify_prompt(
        [{"claim_id": "c1", "subject": "原神", "kind": "term", "line": "ラウマ|菈乌玛|角色"}]
    )
    review = render_review_prompt({"bundle": {}})
    repair = render_repair_prompt(repo, "S", material="素材")
    for prompt in (verify, review, repair):
        assert "知识库判断标准" in prompt and "证据纪律" in prompt
    for prompt in (verify, review):
        assert "写条目的规则" not in prompt and "描述纪律" not in prompt
    assert "写条目的规则" in repair and "术语行格式（固定四列）" in repair


def test_both_op_sections_keep_the_update_contract() -> None:
    """The share conflict repair's op section repeats `update` from the full
    grammar; the two copies must not drift on what an update is."""

    for name in ("fragment_knowledge_output_v1.md", "fragment_share_conflict_ops_v1.md"):
        text = (PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8")
        assert "用新的整行替换该句柄对应的行" in text and "行的类型不能变" in text, name


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
    assert len(proposals) == 3 and all(p.get("op") in MODEL_OPS for p in proposals)
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


def test_the_structure_examples_of_term_lines_are_four_columns() -> None:
    """The 「正确」 example of absorbing fragments into a parent entry writes
    term lines; a three-column one is refused by the engine (cold read N1)."""

    import re

    from finesub.llm.knowledge.node.importer import classify_line

    text = (PROMPT_TEMPLATE_DIR / "fragment_knowledge_structure_v1.md").read_text(encoding="utf-8")
    correct = next(line for line in text.splitlines() if "正确：" in line)
    lines = [body for body in re.findall(r"`([^`]+)`", correct) if "|" in body]
    assert len(lines) == 2
    spec = load_preset("common").section("角色")
    for line in lines:
        assert classify_line(line, spec)[0] == "term", line


def test_an_empty_slot_is_filled_by_appending_its_labelled_line(tmp_path) -> None:
    """An empty slot has no handle, so the output contract (every writable
    session has it, cold read round 4 P470) says to fill it with `append_lines`
    carrying its [标记]; once the marker has a line, a second one is refused --
    that is an `update` (cold read P015)."""

    import json

    from finesub.llm.knowledge.node.proposals import apply_model_proposals
    from finesub.llm.knowledge.node.render import HandleMap as NodeHandleMap

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "开放世界游戏。", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案", "角色"]})

    def append(content: str):  # type: ignore[no-untyped-def]
        row = {"op": "append_lines", "entry": "原神", "category": "common", "section": "档案",
               "content": content, "reason": "r"}
        return apply_model_proposals(json.dumps(row, ensure_ascii=False), repo=repo, task_id="t",
                                     knowledge_read_rev=repo.rev, handles=NodeHandleMap())

    output = (PROMPT_TEMPLATE_DIR / "fragment_knowledge_output_v1.md").read_text(encoding="utf-8")
    assert "空槽没有句柄，要填就用 `append_lines`" in output
    assert "[收录范围]" in repo.entry_prompt_text("S", NodeHandleMap())  # the slot shows, without a handle
    first = append("[收录范围] 只收 3.0 之后的新内容")
    assert first.applied and not first.rolled_back and not first.skipped
    second = append("[收录范围] 只收主线")
    assert second.rolled_back or not second.applied
    scopes = [repo.store.node(m.child_id).payload.get("text") for m in repo.store.children("S")
              if m.section == "档案" and repo.store.node(m.child_id).payload.get("label") == "收录范围"]
    assert scopes == ["只收 3.0 之后的新内容"]


def test_no_prompt_face_carries_the_update_date(tmp_path) -> None:
    """No prompt projection carries the 元数据 section's update date: nothing in
    it informs a correction or an update, and the model may not edit it (cold
    read P057 for the read-only face, round 4 D4 for the writable one). The
    writable face keeps the section only for a recategorization reason; the
    human projection keeps it whole. The prohibitions that guarded it are gone."""

    from finesub.llm.knowledge.node.render import render_subject

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "开放世界游戏。", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案", "角色"]})
        txn.create_node("M", "subject", {"surface": "米哈游", "intro": "厂商。", "category": "common",
                                         "entry_type": "社区", "recategorized_reason": "是厂商不是主播"})
    for preview in ("partial", "full"):
        assert "## 元数据" not in render_subject(repo.store, "S", mode="prompt", preview=preview)
    moved = render_subject(repo.store, "M", mode="prompt", preview="full")
    assert "## 元数据" in moved and "是厂商不是主播" in moved and "最近更新日期" not in moved
    assert "## 元数据" in render_subject(repo.store, "S", mode="human")
    for name in ("fragment_knowledge_structure_v1.md", "fragment_knowledge_output_v1.md"):
        assert "元数据的修改" not in (PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8")
        assert "`元数据` 小节" not in (PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def test_verify_and_review_results_are_read_at_top_level() -> None:
    """Both prompts ask for a <reasoning> block before the result (cold read
    P067); the result block named inside it is prose, not a second block."""

    from finesub.llm.knowledge.share.review import parse_review_verdict
    from finesub.llm.knowledge.verify import parse_verify_results

    review = ("<reasoning>查了官网，结论写进 <review_verdict>。</reasoning>\n"
              '<review_verdict>{"verdict": "approve", "reason": "官网确认"}</review_verdict>')
    assert parse_review_verdict(review, {})["verdict"] == "approve"
    verify = ("<reasoning>结论写进 <verify_results>。</reasoning>\n"
              '<verify_results>[{"claim_id": "c1", "verdict": "unverifiable", "note": "无公开来源"}]</verify_results>')
    assert [row["verdict"] for row in parse_verify_results(verify, [{"claim_id": "c1"}])] == ["unverifiable"]


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


def test_the_split_rule_only_where_split_entry_is_offered() -> None:
    """Cold read round 2, P181: the structure rule that says "use split_entry"
    rides with the op; the proposal repair round never has it."""

    from finesub.llm.knowledge.entry_size import COMPRESS_NOTICE_MARK, SPLIT_OFFER_MARK
    from finesub.llm.prompts import build_knowledge_proposal_repair_messages

    def system(messages):  # type: ignore[no-untyped-def]
        return next(m["content"] for m in messages if m["role"] == "system")

    plain = system(build_knowledge_update_messages(refined=False, task_summary="s", window_packs="w"))
    offered = system(build_knowledge_update_messages(
        refined=False, task_summary="s", window_packs="w", kb_entries=f"# X\n{COMPRESS_NOTICE_MARK}\n{SPLIT_OFFER_MARK}",
    ))
    repair = system(build_knowledge_proposal_repair_messages(
        task_summary="s", read_rev=1, current_rev=2, conflict_rows="r", rejected_rows="", kb_entries="# X",
    ))
    assert "split_entry" not in plain and "split_entry" not in repair
    assert '"op":"split_entry"' in offered and "改用 `split_entry` 把那一节拆成子条目" in offered
    assert "尺寸纪律" in offered and "尺寸纪律" not in plain and "尺寸纪律" not in repair, \
        "the size rule only where an entry carries the compression notice it points at"


def test_the_review_tells_a_misheard_claim_from_an_alias() -> None:
    """Owner D16 (2026-10-19): a misheard form is one no web page can confirm, so the review may approve without
    it; the pending list has to say which item claims are misheard forms for the model to apply that."""

    bundle = {"nodes": [{"handle": "n1", "kind": "term", "payload": {"surface": "X", "zh": "x"}}],
              "items": [{"handle": "i1", "node": "n1", "field": "misheard", "value": "エックス"},
                        {"handle": "i2", "node": "n1", "field": "aliases", "value": "X2"}],
              "memberships": [], "links": [], "claim_summaries": []}
    review = render_review_prompt({"bundle": bundle})
    assert "items/i1（误听） [term]" in review and "items/i2（别名） [term]" in review
    assert "`[关系/经历]` 与标着「误听」的 claim" in review
