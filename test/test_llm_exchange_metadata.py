from __future__ import annotations

from finesub.llm.chunking import SubtitleSegment, SubtitleWindow
from finesub.llm.exchange_metadata import (
    SESSION_RESPONSE_KINDS,
    correction_input_components,
    extract_tagged_block,
    infer_session_name,
    is_session_response_record,
    research_input_components,
)
from finesub.llm.token_budget import CorrectionBudget


class FakeCounter:
    source = "fake"

    def count_text(self, text: str) -> int:
        return len(text)

    def count_texts(self, texts):
        return sum(self.count_text(t) for t in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return int(seconds * 32)


def _window() -> SubtitleWindow:
    segments = [SubtitleSegment("1", 0.0, 1.0, "hello")]
    budget = CorrectionBudget(
        input_tokens=100,
        subtitle_input_tokens=10,
        estimated_output_tokens=150,
        total_with_margin=260,
        token_counter_source="fake",
    )
    return SubtitleWindow(
        chunk_id="0001",
        segments=segments,
        overlap_segments=[],
        boundary_reason="test",
        budget=budget,
        clip_start=0.0,
        clip_end=10.0,
    )


def test_extract_tagged_block_reads_transcript() -> None:
    text = "<transcript>\n1|hello\n</transcript>"
    assert extract_tagged_block(text, "transcript") == "1|hello"


def test_extract_tagged_block_uses_top_level_not_nested_mention() -> None:
    # Nested name drop inside reasoning must not win over the sibling block.
    text = (
        "<reasoning>\n在 `<translated>` 省略幻觉；并按 `<singles>` 写对照\n</reasoning>\n"
        "<singles>\nsub|1|0.5|0.0|a|甲|8|1|译1字；宜独立\n</singles>\n"
        "<translated>\nsub|1|0.5|0.0|a|甲|8|1|译1字\n</translated>"
    )
    assert extract_tagged_block(text, "translated") == "sub|1|0.5|0.0|a|甲|8|1|译1字"
    assert extract_tagged_block(text, "singles") == "sub|1|0.5|0.0|a|甲|8|1|译1字；宜独立"


def test_extract_tagged_block_skips_nested_real_block() -> None:
    text = (
        "<outer>\n<translated>\nnested\n</translated>\n</outer>\n"
        "<translated>\ntop\n</translated>"
    )
    assert extract_tagged_block(text, "translated") == "top"


def test_extract_top_level_ignores_inline_opens_of_other_tags() -> None:
    """Regression: reasoning mentioning <singles> must not poison translated extract."""
    from finesub.llm.exchange_metadata import extract_top_level_tagged_blocks

    text = (
        "<reasoning>\n严格按照 `<singles>` 和 `<translated>` 输出\n</reasoning>\n"
        "<singles>\nsub|1|0.5|0.0|a|甲|8|1|译1字；宜独立\n</singles>\n"
        "<translated>\nsub|1|0.5|0.0|a|甲|8|1|译1字\n</translated>\n"
        "<next_advice>\nok\n</next_advice>"
    )
    assert extract_top_level_tagged_blocks(text, "translated") == ["sub|1|0.5|0.0|a|甲|8|1|译1字"]
    assert extract_top_level_tagged_blocks(text, "singles") == [
        "sub|1|0.5|0.0|a|甲|8|1|译1字；宜独立"
    ]


def test_research_input_components_count_transcript_and_injections() -> None:
    counter = FakeCounter()
    components = research_input_components(
        counter=counter,
        transcript="1|hello world",
        entry_details={"主播A": "档案内容"},
        search_results="query: foo",
    )
    assert components["transcript_input_tokens"] == len("1|hello world")
    assert components["knowledge_injection_tokens"] == len("档案内容")
    assert components["search_injection_tokens"] == len("query: foo")


class _FakeFileRef:
    def __init__(self, *, is_video: bool = False) -> None:
        self.is_video = is_video


def test_correction_input_components_include_csv_audio_and_output() -> None:
    window = _window()
    counter = FakeCounter()
    components = correction_input_components(
        window=window,
        counter=counter,
        search_results="search block",
        context_general='{"summary":"x"}',
        context_window="window ctx",
        max_output_tokens=8192,
        file_ref=_FakeFileRef(),
    )
    assert components["csv_input_tokens"] > 0
    assert components["media_input_tokens"] == 320
    assert components["search_injection_tokens"] == len("search block")
    assert components["knowledge_injection_tokens"] == len('{"summary":"x"}') + len(
        "window ctx"
    ) + 2
    assert components["expected_output_tokens"] == 150
    assert components["max_output_tokens"] == 8192


def test_correction_input_components_count_the_media_that_was_attached() -> None:
    """The estimate follows the attached file, not the media switch.

    A switch asking for media whose clip could not be produced used to bill
    the window for audio the request never carried; taking the ref itself
    leaves the two with no way to disagree.
    """

    text_only = correction_input_components(window=_window(), counter=FakeCounter())
    assert text_only["media_input_tokens"] == 0

    audio = correction_input_components(
        window=_window(), counter=FakeCounter(), file_ref=_FakeFileRef()
    )
    video = correction_input_components(
        window=_window(), counter=FakeCounter(), file_ref=_FakeFileRef(is_video=True)
    )
    # A video clip carries its audio track: video costs strictly more.
    assert audio["media_input_tokens"] == 320
    assert video["media_input_tokens"] > audio["media_input_tokens"]


def test_research_components_cover_indices_extra_info_and_preinjection() -> None:
    user = (
        "<extra_info>\n来源说明\n</extra_info>\n"
        "<note_url_extracts>\n（无）\n</note_url_extracts>\n"
        "<streamer_index>\n- 主播A | 别名 | 简介\n</streamer_index>\n"
        "<common_index>\n（空）\n</common_index>\n"
        "<preinjected_entries>\n## 主播A\n\n档案内容\n</preinjected_entries>\n"
        "<transcript>\n1|你好\n</transcript>"
    )
    components = research_input_components(
        counter=FakeCounter(), messages=[{"role": "user", "content": user}]
    )

    assert components["extra_info_tokens"] == len("来源说明")
    assert components["streamer_index_tokens"] == len("- 主播A | 别名 | 简介")
    assert components["common_index_tokens"] == 0  # empty marker
    assert components["preinjected_entry_tokens"] == len("## 主播A\n\n档案内容")
    assert components["round1_notes_tokens"] == 0


def test_search_loop_components_cover_contract_progress_and_entries() -> None:
    from finesub.llm.exchange_metadata import search_loop_input_components

    user = (
        "<background>\n背景\n</background>\n"
        "<research_contract>\n{\"goal\":\"g\"}\n</research_contract>\n"
        "<executed_queries>\nq1\n</executed_queries>\n"
        "<research_progress>\n进展\n</research_progress>\n"
        "<streamer_index>\n- 主播A | 别名 | 简介\n</streamer_index>\n"
        "<common_index>\n（空）\n</common_index>\n"
        "<knowledge_entries>\n## 条目\n\n内容\n</knowledge_entries>\n"
        "<search_results>\n--- query: q1 ---\n结果\n</search_results>"
    )
    components = search_loop_input_components(
        counter=FakeCounter(), messages=[{"role": "user", "content": user}]
    )

    assert components["background_tokens"] == len("背景")
    assert components["contract_tokens"] == len('{"goal":"g"}')
    assert components["executed_queries_tokens"] == len("q1")
    assert components["progress_tokens"] == len("进展")
    assert components["streamer_index_tokens"] > 0
    assert components["common_index_tokens"] == 0
    assert components["knowledge_injection_tokens"] == len("## 条目\n\n内容")
    assert components["search_injection_tokens"] == len("--- query: q1 ---\n结果")


def test_correction_components_cover_entries_advice_and_notes() -> None:
    user = (
        "<entry_details>\n## 条目\n\n内容\n</entry_details>\n"
        "<previous_advice>\n[window 0001]\n建议\n</previous_advice>\n"
        "<pre_round_notes>\n要点\n</pre_round_notes>\n"
        "<search_results>\n（无）\n</search_results>"
    )
    components = correction_input_components(
        window=_window(),
        counter=FakeCounter(),
        messages=[{"role": "user", "content": user}],
    )

    assert components["entry_details_tokens"] == len("## 条目\n\n内容")
    assert components["advice_ledger_tokens"] == len("[window 0001]\n建议")
    assert components["pre_round_notes_tokens"] == len("要点")
    assert components["preceding_context_tokens"] == 0
    assert components["streamer_index_tokens"] == 0


def test_knowledge_update_session_name_uses_chunk_number() -> None:
    assert "knowledge_update_response" in SESSION_RESPONSE_KINDS
    assert (
        infer_session_name("knowledge_update_response", {"chunk": 3})
        == "knowledge-update-chunk03"
    )


def test_search_execution_ledger_is_not_a_session_response() -> None:
    assert not is_session_response_record(
        "search_loop_round", {"round": 0, "executed": [{"provider": "exa"}]}
    )
    assert is_session_response_record(
        "search_loop_round", {"round": 0, "response_content": "ok", "usage": {}}
    )


def test_output_limit_fields_fold_into_one_line() -> None:
    """Four near-identical lines cost more to read than they inform.

    The margin is dropped outright: it is ``max - threshold``, a constant of
    the check rather than anything observed about this call.
    """

    from finesub.llm.exchange_metadata import _fold_output_limit_fields

    metadata = {
        "finish_reason": "STOP",
        "output_limited": False,
        "output_limit_basis": "output_tokens_plus_thinking_tokens",
        "output_limit_observed_tokens": 60850,
        "output_limit_threshold_tokens": 65436,
        "output_limit_max_tokens": 65536,
        "output_limit_margin_tokens": 100,
    }
    _fold_output_limit_fields(metadata)

    assert metadata["output_limit"] == (
        "observed 60850 / threshold 65436 / max 65536 "
        "(basis: output_tokens_plus_thinking_tokens)"
    )
    assert not [key for key in metadata if key.startswith("output_limit_")]
    # The verdict keeps its own line: it is what a reader greps for.
    assert metadata["output_limited"] is False


def test_folding_is_a_no_op_without_an_output_limit_check() -> None:
    from finesub.llm.exchange_metadata import _fold_output_limit_fields

    metadata = {"finish_reason": "STOP"}
    _fold_output_limit_fields(metadata)
    assert metadata == {"finish_reason": "STOP"}
