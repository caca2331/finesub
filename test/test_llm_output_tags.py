from __future__ import annotations

import pytest

from llm.output_tags import (
    extract_single_tag_block,
    find_tag_blocks,
    find_top_level_tag_blocks,
    looks_truncated_tag_block,
    missing_top_level_tags,
    parse_guided_line_items,
    parse_json_tag_block,
    parse_line_items,
)
from llm.session_contract import SESSION_CONTRACTS, SessionContract


def test_parse_guided_line_items_splits_guided_suffix_and_dedupes() -> None:
    body = (
        "- 绝区零 诺姆 >> 诺姆的人际关系\n"
        "1. 游戏B 剧情\n"
        "绝区零 诺姆 >> 重复行按文本去重\n"
        "https://a.test/page >> 重点提取阵营\n"
        "   \n"
    )

    items = parse_guided_line_items(body)

    assert items == [
        ("绝区零 诺姆", "诺姆的人际关系"),
        ("游戏B 剧情", ""),
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
