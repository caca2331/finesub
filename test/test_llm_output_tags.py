from __future__ import annotations

import pytest

from finesub.llm.output_tags import (
    extract_single_tag_block,
    find_tag_blocks,
    find_top_level_tag_blocks,
    looks_truncated_tag_block,
    missing_top_level_tags,
    parse_guided_line_items,
    parse_json_tag_block,
    parse_line_items,
)
from finesub.llm.session_contract import SESSION_CONTRACTS, SessionContract


def test_parse_guided_line_items_splits_guided_suffix_and_dedupes() -> None:
    """Dedup is per (text, guided).

    The guided query decides what the provider highlights or extracts, so the
    same target with a different focus is a different request. Only a repeat of
    both is dropped; whitespace and case in the focus are not a difference.
    """

    body = (
        "- 绝区零 诺姆 >> 诺姆的人际关系\n"
        "1. 游戏B 剧情\n"
        "绝区零 诺姆 >> 换个角度：诺姆的登场作品\n"
        "绝区零 诺姆 >>  诺姆的人际关系 \n"
        "https://a.test/page >> 重点提取阵营\n"
        "   \n"
    )

    items = parse_guided_line_items(body)

    assert items == [
        ("绝区零 诺姆", "诺姆的人际关系"),
        ("游戏B 剧情", ""),
        ("绝区零 诺姆", "换个角度：诺姆的登场作品"),
        ("https://a.test/page", "重点提取阵营"),
    ]


def test_find_top_level_tag_blocks_ignores_reasoning_name_drop() -> None:
    # A <reasoning> block that name-drops other tags must not satisfy or steal
    # a later sibling — only the genuine top-level block counts.
    text = (
        "<reasoning>then I'll emit <window_notes> and <search_queries></reasoning>"
        "<window_notes>真正的要点</window_notes>"
        "<search_queries></search_queries>"
    )
    assert find_top_level_tag_blocks(text, "window_notes") == ["真正的要点"]
    assert find_top_level_tag_blocks(text, "search_queries") == [""]


def test_find_top_level_tag_blocks_keeps_nested_void_inside_translated() -> None:
    # <void> lives one level deep inside <translated>; it is part of the parent
    # body (the correction CSV parser handles it) and is never a top-level block.
    text = "<translated>\nsub|1|x\n<void>2</void>\nsub|3|y\n</translated>"
    body = find_top_level_tag_blocks(text, "translated")[0]
    assert "<void>2</void>" in body
    assert find_top_level_tag_blocks(text, "void") == []


def test_missing_top_level_tags_flags_swallowed_sibling_only() -> None:
    # window_notes swallowed inside search_queries -> missing at top level.
    bad = "<reasoning>r</reasoning><search_queries><window_notes>x</window_notes></search_queries>"
    assert missing_top_level_tags(bad, ["window_notes"]) == [
        "<window_notes> missing at top level (nested or absent)"
    ]
    good = "<window_notes>x</window_notes><search_queries></search_queries>"
    assert missing_top_level_tags(good, ["window_notes", "search_queries"]) == []


def test_session_contract_validate_nonempty_vs_present() -> None:
    contract = SessionContract(
        nonempty=("reasoning", "analysis_notes"),
        present=("keep_entries", "search_queries"),
    )
    ok = (
        "<reasoning>r</reasoning><analysis_notes>笔记</analysis_notes>"
        "<keep_entries></keep_entries><search_queries></search_queries>"
    )
    assert contract.validate(ok) == []
    # analysis_notes present but empty -> error; keep_entries absent -> error.
    bad = "<reasoning>r</reasoning><analysis_notes>  </analysis_notes><search_queries></search_queries>"
    errors = contract.validate(bad)
    assert "empty <analysis_notes> block" in errors
    assert "missing <keep_entries> block" in errors


def test_query_contract_allows_empty_list_blocks() -> None:
    contract = SESSION_CONTRACTS["query"]
    reply = (
        "<reasoning>分析</reasoning><window_notes></window_notes>"
        "<keep_entries></keep_entries><search_queries></search_queries>"
    )
    assert contract.validate(reply) == []


def test_extract_single_tag_block_tolerates_prose_and_case() -> None:
    text = "前言\n<Search_Queries>\nq1\n</search_queries>\n后记"

    assert extract_single_tag_block(text, "search_queries") == "q1"


def test_extract_single_tag_block_rejects_missing_and_duplicates() -> None:
    with pytest.raises(ValueError, match="missing"):
        extract_single_tag_block("没有块", "search_queries")
    assert extract_single_tag_block("没有块", "search_queries", required=False) == ""
    duplicated = "<a>1</a><a>2</a>"
    with pytest.raises(ValueError, match="exactly one"):
        extract_single_tag_block(duplicated, "a")
    assert find_tag_blocks(duplicated, "a") == ["1", "2"]


def test_looks_truncated_tag_block_detects_missing_closer() -> None:
    assert looks_truncated_tag_block("<translated>\n1|a|b", "translated")
    assert not looks_truncated_tag_block("<translated></translated>", "translated")
    assert not looks_truncated_tag_block("没有块", "translated")


def test_parse_line_items_strips_bullets_numbering_and_dupes() -> None:
    body = "\n- q1\n2. q2\nq2\n・q3\n  \n\"q4\"\n"

    assert parse_line_items(body) == ["q1", "q2", "q3", "q4"]


def test_parse_json_tag_block_prefers_tag_and_falls_back() -> None:
    tagged = '<context_pack>\n{"a": 1}\n</context_pack>'
    assert parse_json_tag_block(tagged, "context_pack") == {"a": 1}

    bare = '说明文字 {"a": 2} 尾巴'
    assert parse_json_tag_block(bare, "context_pack") == {"a": 2}

    fenced = '<context_pack>\n```json\n{"a": 3}\n```\n</context_pack>'
    assert parse_json_tag_block(fenced, "context_pack") == {"a": 3}

    with pytest.raises(ValueError):
        parse_json_tag_block("没有 JSON", "context_pack")


# --- One definition of "a block", tolerant of unclosed scaffolding ----------


def test_unclosed_pseudo_tag_does_not_hide_later_top_level_blocks() -> None:
    """A stray `<br>` used to make every later top-level block invisible.

    The scanner pushed any opening tag onto the stack and only popped it on a
    matching close, so one unclosed sibling left the stack permanently dirty
    and a perfectly well-formed payload was discarded -- burning the window's
    whole validation-retry budget.
    """
    text = (
        "<reasoning>plan</reasoning>\n"
        "<br>\n"
        "<translated>\n"
        "sub|1|1.0|0.0|a|甲|high|1|\n"
        "</translated>"
    )
    blocks = find_top_level_tag_blocks(text, "translated")
    assert len(blocks) == 1
    assert "sub|1|1.0|0.0|a|甲|high|1|" in blocks[0]


def test_unclosed_tag_of_the_wanted_name_still_yields_nothing() -> None:
    """Tolerance is for scaffolding, not for a truncated payload."""
    text = "<translated>\nsub|1|1.0|0.0|a|甲|high|1|\n"
    assert find_top_level_tag_blocks(text, "translated") == []


def test_extract_single_tag_block_uses_top_level_semantics() -> None:
    """The payload reader and the contract gate must agree on what a block is."""
    nested_only = "<reasoning>I will write <answer>no</answer> next</reasoning>"
    with pytest.raises(ValueError):
        extract_single_tag_block(nested_only, "answer")

    sibling = "<reasoning>thinking</reasoning>\n<answer>yes</answer>"
    assert extract_single_tag_block(sibling, "answer") == "yes"


def test_extract_single_tag_block_still_rejects_duplicate_top_level_blocks() -> None:
    text = "<answer>one</answer>\n<answer>two</answer>"
    with pytest.raises(ValueError):
        extract_single_tag_block(text, "answer")


def test_research_round1_contract_follows_the_axis_halves() -> None:
    from finesub.llm.session_contract import research_round1_contract

    local = research_round1_contract(
        emits_queries=True, emits_entries=True, emits_notes=True
    )
    assert local.nonempty == ("reasoning", "analysis_notes")
    assert local.present == ("requested_entries", "keep_entries", "search_queries")

    # retrieval=none with a knowledge base: entries only, and no analysis_notes
    # because round 2 -- their only reader -- is skipped.
    entries_only = research_round1_contract(
        emits_queries=False, emits_entries=True, emits_notes=False
    )
    assert entries_only.nonempty == ("reasoning",)
    assert entries_only.present == ("requested_entries", "keep_entries")

    # retrieval=local without a knowledge base: queries only.
    queries_only = research_round1_contract(
        emits_queries=True, emits_entries=False, emits_notes=True
    )
    assert queries_only.present == ("search_queries",)


def test_research_round2_contract_never_emits_queries() -> None:
    from finesub.llm.session_contract import research_round2_contract

    with_keep = research_round2_contract(emits_keep=True)
    assert with_keep.nonempty == ("reasoning", "context_pack")
    assert with_keep.present == ("keep_entries",)
    # native: no query round downstream, so pruning is not asked for.
    assert research_round2_contract(emits_keep=False).present == ()


def test_shipped_contracts_are_derived_from_the_same_functions() -> None:
    from finesub.llm.session_contract import (
        SESSION_CONTRACTS,
        research_round1_contract,
        research_round2_contract,
    )

    assert SESSION_CONTRACTS["research_round1"] == research_round1_contract(
        emits_queries=True, emits_entries=True, emits_notes=True
    )
    assert SESSION_CONTRACTS["research_round2"] == research_round2_contract(
        emits_keep=True
    )
