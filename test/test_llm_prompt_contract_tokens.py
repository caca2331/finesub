"""The output-contract fragments must agree with the code that parses them.

``docs/llm_prompts.md``: these fragments are not
generated -- they describe the output shape in prose -- but every literal in
them that the parser also knows about (column headers, row kinds, the void
marker, block names, confidence levels) is checked against the constant rather
than reviewed by eye. A renamed token that only lives in prose is exactly how a
prompt starts describing a contract the harness no longer enforces.
"""

from __future__ import annotations

import dataclasses

import pytest

from finesub.llm.output_protocol import (
    CONFIDENCE_LEVELS,
    KIND_DISCARD,
    KIND_SUB,
    VOID_ROW_MARKER,
    correction_output_header,
)
from finesub.llm.output_tags import find_top_level_tag_blocks
from finesub.llm.prompt_compose import PROMPT_TEMPLATE_DIR
from finesub.llm.prompt_variants import VARIANTS
from finesub.llm.session_contract import SESSION_CONTRACTS

# Fragments whose whole job is to describe the output shape.
CONTRACT_FRAGMENTS = (
    "fragment_output_contract_nosingles_v1.md",
    "fragment_output_contract_grouped_v1.md",
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
    """A header literal must be a variant's own header, or a placeholder for it."""

    text = _read(name)
    for line in text.splitlines():
        if "type|position" not in line:
            continue
        for header in {correction_output_header(v) for v in VARIANTS.values()}:
            if header in line:
                break
        else:
            raise AssertionError(
                f"{name}: a header-shaped literal that is no variant's "
                f"output header: {line!r}"
            )


def test_void_marker_is_described_in_every_contract() -> None:
    """``<void>`` retracts a translated row, the one block every set writes."""

    for name in CONTRACT_FRAGMENTS:
        assert VOID_ROW_MARKER in _read(name), name


def _assembled_prompts() -> dict:
    """The prompts as the model receives them, not the raw templates.

    The templates are mostly ``$slot`` placeholders, so the block names only
    appear after assembly -- which is the state that has to match the contract.
    """

    from finesub.llm.prepare_prompts import build_prepare_task_messages
    from finesub.llm.request_prompts import build_window_task_messages
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.chunking import SubtitleSegment, plan_correction_windows

    class _Counter:
        source = "test-fake"

        def count_text(self, text):
            return max(1, len(text or "") // 2)

        def count_texts(self, texts):
            return sum(self.count_text(t) for t in texts)

        def count_audio_seconds(self, seconds):
            return max(0, int(seconds * 32))

    window = dataclasses.replace(plan_correction_windows(
        [SubtitleSegment("1", 0.0, 1.0, "テスト")], counter=_Counter()
    )[0], length_scale=1.0)
    return {
        "prepare_task_request": build_prepare_task_messages(
            transcript="1|hi", knowledge_indexes={"streamer": "- k | a | d", "common": ""}
        )[0]["content"],
        "prepare_task_delivery": build_prepare_task_messages(transcript="1|hi")[0]["content"],
        "correction_request": build_window_task_messages(
            request_channel=True,
            knowledge_indexes={"streamer": "- k | a | d", "common": ""},
            window=window,
            profile=resolve_profile("audio", "local", "quality"),
        )[0]["content"],
    }


def test_session_output_blocks_exist_in_their_prompt() -> None:
    """Every block a contract demands must be described by its own prompt."""

    for session, text in _assembled_prompts().items():
        contract = SESSION_CONTRACTS[session]
        for tag in (*contract.nonempty, *contract.present):
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

    text = _assembled_prompts()["prepare_task_request"]
    # The prompt names `<context_pack>` in prose long before the example
    # blocks, which the nesting-aware extractor would read as an unclosed
    # opener; measure from the request-block examples onward instead.
    examples = text[text.index("请求回复的输出块") :]
    for tag in ("requested_entries", "keep_entries"):
        assert find_top_level_tag_blocks(examples, tag), tag
