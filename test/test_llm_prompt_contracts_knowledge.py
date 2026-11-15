"""Prompt contracts of the knowledge sessions fixed in the 2026-11-06 cold read (round 4)."""

from __future__ import annotations

from finesub.llm.prompt_compose import PROMPT_TEMPLATE_DIR
from finesub.llm.prompts import build_knowledge_proposal_repair_messages


def test_the_update_rules_do_not_restate_the_entry_markers() -> None:
    """The two entry markers say themselves what they mean (`knowledge/entries.py`) and appear only in
    chunked runs; the evidence rule that repeated them is gone (P448)."""

    for name in ("knowledge_update_artifacts_only_v1.md", "knowledge_update_refined_v1.md"):
        text = (PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8")
        assert "本任务前序块已更新" not in text and "库中暂无" not in text, name


def test_the_proposal_repair_system_carries_no_update_material_rules() -> None:
    """The repair round has the dropped proposals and the current entries only: the update's window-pack
    inputs, the hint confidence threshold and the sub-entry rule are dead clauses there (P440, P447)."""

    system = build_knowledge_proposal_repair_messages(
        task_summary="t", read_rev=1, current_rev=2,
        conflict_rows="- common/甲：`append_lines` ", rejected_rows="", kb_entries="",
    )[0]["content"]
    assert "输入只有下面两样" in system and "原提案的 `reason` 就是本轮的证据" in system
    for absent in ("window N", "raw_csv", "final_csv", "char_count", "confidence", "sub` 字段", "无精修模式"):
        assert absent not in system, absent
