"""The output-contract fragments must agree with the code that parses them.

``docs/llm_prompts.md``: these fragments are not
generated -- they describe the output shape in prose -- but every literal in
them that the parser also knows about (column headers, row kinds, the void
marker, block names, confidence levels) is checked against the constant rather
than reviewed by eye. A renamed token that only lives in prose is exactly how a
prompt starts describing a contract the harness no longer enforces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub.llm.output_protocol import (
    CONFIDENCE_LEVELS,
    KIND_DISCARD,
    KIND_SUB,
    OUTPUT_CSV_HEADER,
    OUTPUT_CSV_HEADER_WITH_START,
    VOID_ROW_MARKER,
)
from finesub.llm.output_tags import find_top_level_tag_blocks
from finesub.llm.prompt_compose import PROMPT_TEMPLATE_DIR
from finesub.llm.session_contract import SESSION_CONTRACTS

# Fragments whose whole job is to describe the output shape.
CONTRACT_FRAGMENTS = (
    "fragment_output_contract_v1.md",
    "fragment_output_contract_nosingles_v1.md",
    "fragment_output_contract_nosingles_reasoning_v1.md",
)


def _read(name: str) -> str:
    return (PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", CONTRACT_FRAGMENTS)
def test_contract_fragment_names_the_real_row_kinds(name: str) -> None:
    text = _read(name)
    assert KIND_SUB in text, f"{name} never names the sub row kind"
    assert KIND_DISCARD in text, f"{name} never names the discard row kind"
    for level in CONFIDENCE_LEVELS:
        assert level in text, f"{name} omits confidence level {level!r}"


@pytest.mark.parametrize("name", CONTRACT_FRAGMENTS)
def test_contract_fragment_uses_no_stale_column_header(name: str) -> None:
    """A header literal must be the constant, or a placeholder for it."""

    text = _read(name)
    for line in text.splitlines():
        if "type|position" not in line:
            continue
        for header in (OUTPUT_CSV_HEADER_WITH_START, OUTPUT_CSV_HEADER):
            if header in line:
                break
        else:
            raise AssertionError(
                f"{name}: a header-shaped literal that matches neither "
                f"OUTPUT_CSV_HEADER nor the start-bearing one: {line!r}"
            )


def test_void_marker_is_only_described_where_it_is_accepted() -> None:
    """``<void>`` retracts a translated row; singles rejects it."""

    singles_contract = _read("fragment_output_contract_v1.md")
    assert VOID_ROW_MARKER in singles_contract
    # output_protocol rejects <void> inside <singles>, so the fragment that describes
    # both blocks has to say so.
    assert "singles" in singles_contract


def _assembled_prompts() -> dict:
    """The prompts as the model receives them, not the raw templates.

    The templates are mostly ``$slot`` placeholders, so the block names only
    appear after assembly -- which is the state that has to match the contract.
    """

    from finesub.llm.prompt_compose import compose_correction_query_system
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.prompts import (
        build_fast_round1_messages,
        build_research_round1_messages,
        build_research_round2_messages,
        build_search_loop_messages,
    )
    from finesub.llm.chunking import SubtitleSegment, plan_correction_windows

    class _Counter:
        source = "test-fake"

        def count_text(self, text):
            return max(1, len(text or "") // 2)

        def count_texts(self, texts):
            return sum(self.count_text(t) for t in texts)

        def count_audio_seconds(self, seconds):
            return max(0, int(seconds * 32))

    window = plan_correction_windows(
        [SubtitleSegment("1", 0.0, 1.0, "テスト")], counter=_Counter()
    )[0]
    return {
        "query": compose_correction_query_system(
            resolve_profile("audio", "local", "quality"),
            search_queries_rules="（示例搜索规则）",
        ),
        "fast_round1": build_fast_round1_messages(
            window=window, streamer_index="- k | a | d", common_index=""
        )[0]["content"],
        "research_round1": build_research_round1_messages(transcript="1|hi")[0][
            "content"
        ],
        "research_round2": build_research_round2_messages(transcript="1|hi")[0][
            "content"
        ],
        "search_loop": build_search_loop_messages(
            round_index=0, max_rounds=2, is_final_round=False, contract_json="{}"
        )[0]["content"],
    }


def test_session_output_blocks_exist_in_their_prompt() -> None:
    """Every block a contract demands must be described by its own prompt."""

    for session, text in _assembled_prompts().items():
        contract = SESSION_CONTRACTS[session]
        for tag in (*contract.nonempty, *contract.present):
            if tag == "reasoning":
                continue  # injected through $reasoning_clause, worded not tagged
            assert f"<{tag}>" in text, (
                f"the {session} prompt never mentions <{tag}>, which its "
                "contract requires the model to emit"
            )


def test_extractor_and_prompt_agree_on_top_level_blocks() -> None:
    """A block the prompt demonstrates must parse as a top-level sibling.

    The runtime validator uses nesting-aware extraction, so an example block
    accidentally shown *inside* another block would satisfy review and fail
    validation.
    """

    text = _assembled_prompts()["research_round1"]
    for tag in ("requested_entries", "keep_entries"):
        assert find_top_level_tag_blocks(text, tag), tag
