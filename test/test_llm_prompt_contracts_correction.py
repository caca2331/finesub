"""Contract tests for the correction prompts' cold-read fixes (round 4, 2026-11-06).

Each test pins what a fixed passage says now, so a later edit cannot quietly bring the defect back.
"""

from __future__ import annotations

import pytest

from finesub.llm.knowledge.style import StyleCatalog, StyleContractError, render_style_block
from finesub.llm.prompt_compose import compose_correction_system, compose_correction_user
from finesub.llm.request_prompts import build_correction_fill_message, render_window_request_channel
from finesub.llm.routing.profiles import resolve_profile


def _user(variant: str, *, knowledge_enabled: bool = True, agentic: bool = False) -> str:
    return compose_correction_user(
        resolve_profile("audio", "local", "quality"),
        variant=variant,
        knowledge_enabled=knowledge_enabled,
        agentic=agentic,
        general_context_json="",
        window_context="",
        entry_details="",
        previous_advice="",
        pre_round_notes="",
        search_results="",
        preceding_context_csv="",
        current_asr_csv="1|0.0|1.0|0.0|x",
        current_asr_row_count=1,
        length_scale=1.0,
    )


@pytest.mark.parametrize("variant", ["basicA", "capableC"])
def test_the_closing_reminder_drops_only_lines_it_is_sure_are_hallucinated(variant: str) -> None:
    """P436: 「疑似幻觉」 is the note mark for a line kept in doubt; the closing reminder's discard is for the
    highly suspect ones only, as the reminders above it say."""

    closing = _user(variant).split("最后提醒", 1)[1]
    assert "（高度疑似幻觉以 `discard|<源序号>` 显式丢弃）" in closing
    assert "（疑似幻觉以" not in closing


def test_the_factory_default_lead_names_the_empty_pick_the_rules_ask_for() -> None:
    """P435: the pick rules say an empty `selected` is "none fits"; the factory default's lead says the same,
    and the rejection of a name outside the catalog points to it."""

    from finesub.llm.prepare_prompts import render_style_selection_section

    section = render_style_selection_section("- 甲组｜简介｜作者：未记｜适用范围：未记", shape="agentic", read="required")
    assert "`selected` 填空字符串时，这一窗按下面这套「出厂默认」风格翻译" in section
    assert "选「无」" not in section
    with pytest.raises(StyleContractError, match="或填空字符串表示都对不上"):
        StyleCatalog.from_keys(["甲组", "乙组"]).decide(selected="无", reason="")
    # with `default_style` an empty pick is not an answer, so the hint is not given
    with pytest.raises(StyleContractError) as raised:
        StyleCatalog.from_keys(["甲组", "default_style"]).decide(selected="无", reason="")
    assert "空字符串" not in str(raised.value)


def test_an_agentic_window_keeps_the_entries_it_read() -> None:
    """P438 / D7: an agentic window reads entries with `kb_read` and the harness takes any key of the base, so
    it is not told to keep only what its `<entry_details>` shows; a batched window still is."""

    profile = resolve_profile("audio", "local", "quality")
    agentic = compose_correction_system(profile, variant="capableC", agentic=True, length_scale=1.0)
    batched = compose_correction_system(profile, variant="capableC", length_scale=1.0)
    assert "只能引用本任务里读过的词条（`<entry_details>` 里给出的，或你用 `kb_read` 读过的）" in agentic
    assert "只能引用本窗口「知识库条目详情」（`<entry_details>`）中实际出现的词条" in batched
    assert "引用 `<entry_details>` 之外的 key" not in agentic + batched


def test_the_knowledge_index_lets_a_window_keep_what_the_fill_brought() -> None:
    """P445: the index header and the keep rules agree that an entry the fill brought in `<entry_details>` may
    be kept, so a later window need not ask for it again."""

    channel = render_window_request_channel(knowledge_indexes={"streamer": "- 主播A | | 测试主播"})
    assert "`<keep_entries>` 只能写本窗口 `<entry_details>` 里实际出现过的词条，首条 user 消息与回填里的都算" in channel
    assert "只能写首条 user 消息" not in channel


def test_the_factory_default_is_injected_without_its_maintainer_notes() -> None:
    """P450: a session tells the factory default by `<style name="出厂默认">`; its title, intro and 档案 are
    for the maintainer and are not injected."""

    block = render_style_block("C:/no-such-knowledge-root", [])
    assert '<style name="出厂默认">' in block and "## 约定" in block
    assert "# 出厂默认" not in block and "## 档案" not in block and "[作者]" not in block


def test_a_window_without_a_knowledge_base_shows_no_entry_block() -> None:
    """P452: no knowledge base, no `<entry_details>` block (an empty block named an input that never comes);
    an agentic window's block does not speak of a request turn it does not have."""

    assert "<entry_details>" not in _user("capableC", knowledge_enabled=False)
    assert "<entry_details>" in _user("capableC")
    agentic = _user("capableC", agentic=True)
    assert "<entry_details>" in agentic and "前置轮索取" not in agentic


def test_the_fill_names_the_blocks_a_delivery_opens_with() -> None:
    """P437: basicA delivers `<decisions>` before `<translated>`; the fill turn says so too."""

    basic = build_correction_fill_message(delivery="先 `<decisions>`、再 `<translated>` 完整终稿")["content"]
    assert "（先 `<decisions>`、再 `<translated>` 完整终稿" in basic
    assert "（`<translated>` 完整终稿" in build_correction_fill_message()["content"]
