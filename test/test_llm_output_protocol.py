from __future__ import annotations

from types import SimpleNamespace

import pytest

from finesub.llm.chunking import (
    SubtitleSegment,
    SubtitleWindow,
    WindowIdMap,
    render_segments_as_csv,
)
from finesub.llm.prompt_constants import Thresholds
from finesub.llm.routing.config import CapabilityTier
from finesub.llm.text_cells import decode_text_cell, encode_text_cell
from finesub.llm.output_protocol import (
    LEAKED_THOUGHT_ERROR,
    CORRECTION_ANSWER_TAGS,
    CsvValidationResult,
    strip_leaked_thought,
    TranslatedCsvSegment,
    looks_truncated_translated,
    merge_translated_csv_windows,
    window_merge_gaps,
    render_corrected_segments_as_srt,
    render_translated_segments_as_csv,
    render_translated_segments_as_srt,
    validate_correction_output_text,
    validate_correction_window_output,
    validate_translated_csv_text,
)
from finesub.llm.prompt_variants import resolve_variant
from finesub.subtitles.model import parse_srt

STANDARD = Thresholds.at(1.0)


def test_render_segments_as_csv_uses_local_tenths_and_escaped_text() -> None:
    segments = [
        SubtitleSegment("3", 10.11, 10.61, "你\n好|呀"),
        SubtitleSegment("4", 11.01, 12.01, "好"),
    ]

    csv_text = render_segments_as_csv(segments, window_start=9.01)

    assert csv_text.splitlines() == [
        r"3|1.1|0.5||你\n好｜呀",  # the first line has no line before it
        "4|2.0|1.0|0.4|好",  # the pause before it: 11.01 - 10.61
    ]
    assert segments[0].start == 10.11
    assert segments[0].end == 10.61


def test_window_id_map_uses_positive_targets_and_nonpositive_references() -> None:
    targets = [
        SubtitleSegment("188", 10.0, 11.0, "a"),
        SubtitleSegment("189", 11.1, 12.0, "b"),
    ]
    references = [
        SubtitleSegment("186", 8.0, 8.5, "x"),
        SubtitleSegment("187", 8.6, 9.0, "y"),
    ]
    id_map = WindowIdMap(
        source_ids=("188", "189"),
        preceding_source_ids=("186", "187"),
    )

    assert [segment.id for segment in id_map.localize_segments(targets)] == ["1", "2"]
    assert [
        segment.id for segment in id_map.localize_preceding_segments(references)
    ] == ["-1", "0"]
    assert id_map.source_id_for_local("2") == "189"


def test_validate_can_require_v55_headers_without_breaking_legacy_audits() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    without_headers = "<translated>\nsub|1|a|甲|high|\n</translated>"
    assert validate_translated_csv_text(without_headers, source).ok
    strict_missing = validate_translated_csv_text(
        without_headers, source, require_headers=True
    )
    assert not strict_missing.ok
    assert sum("exact CSV header" in error for error in strict_missing.errors) == 1

    with_headers = (
        f"<translated>\n{SIX}\n"
        "sub|1|a|甲|high|\n</translated>"
    )
    strict_ok = validate_translated_csv_text(
        with_headers, source, require_headers=True
    )
    assert strict_ok.ok, strict_ok.errors


def test_variant_aware_validator_projects_the_served_output_contract() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    translated_only = (
        f"<translated>\n{SIX}\n"
        "sub|1|a|甲|high|\n</translated>"
    )

    capable = validate_correction_output_text(
        translated_only,
        source,
        variant=resolve_variant(None, CapabilityTier.CAPABLE),
        thresholds=STANDARD,
    )
    # The basic default (basicA) writes seven columns, `start` but no guide columns, in one <translated> block.
    basic = validate_correction_output_text(
        f"<translated>\n{BASIC_SEVEN}\nsub|1|0.0|a|甲|high|\n</translated>",
        source,
        variant=resolve_variant(None, CapabilityTier.BASIC),
        thresholds=STANDARD,
    )
    assert capable.ok, capable.errors
    assert basic.ok, basic.errors


# --- six columns: no guide columns (capableC and `capableC-grouped`, owner 2026-10-23) ---------------------------

SIX = "type|position|corrected_text|translation|conf|note"
BASIC_SEVEN = "type|position|start|corrected_text|translation|conf|note"
GROUPED = resolve_variant("capableC-grouped")


def six(*rows: str) -> str:
    return f"<translated>\n{SIX}\n" + "\n".join(rows) + "\n</translated>"


def test_the_grouped_set_writes_six_columns_and_parses_them_by_position() -> None:
    from finesub.llm.output_protocol import correction_output_header, output_csv_header

    assert correction_output_header(GROUPED) == SIX
    assert correction_output_header(resolve_variant("capableC")) == SIX == output_csv_header()
    assert correction_output_header(resolve_variant("basicA")) == BASIC_SEVEN == output_csv_header(start=True)
    source = [SubtitleSegment("1", 0.0, 1.0, "テスト"), SubtitleSegment("2", 1.2, 2.0, "です")]
    result = validate_correction_output_text(six("sub|1,2|テストです|是测试|high|同一句"), source, variant=GROUPED, thresholds=STANDARD)
    assert result.ok, result.errors
    [row] = result.segments
    assert (row.corrected_text, row.translation, row.conf, row.note) == ("テストです", "是测试", "high", "同一句")
    assert validate_correction_output_text(six("sub|1|テスト|测试|median|", "discard|2|复读幻觉"), source,
                                           variant=GROUPED, thresholds=STANDARD).ok
    # the nine-column header the guide columns had is the wrong header here
    nine = "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
    assert not validate_correction_output_text(
        f"<translated>\n{nine}\nsub|1,2|テストです|是测试|high|\n</translated>", source,
        variant=GROUPED, thresholds=STANDARD).ok


@pytest.mark.parametrize("row", [
    "sub|1|1.0|テスト|测试|high|",  # a duration written in: every cell one place off, the note empty
    "sub|1|テスト|测试|high|ok|",  # a trailing `|` -- no exemption: it is the shape above
    "sub|1|テスト|测试|high|a|b",  # a raw `|` in the note
    "sub|1|テスト|测试|high",  # a cell short
])
def test_six_columns_reject_any_other_cell_count(row: str) -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "テスト")]
    result = validate_correction_output_text(six(row), source, variant=GROUPED, thresholds=STANDARD)
    assert not result.ok
    assert any(f"every row has exactly 6 ({SIX})" in error and "｜" in error for error in result.errors)
    [verdict] = result.rows
    assert verdict.errors, "the error belongs to the row, so a patch round can send just that row back"
    assert validate_correction_output_text(six("sub|1|テスト|测试|high|a｜b"), source, variant=GROUPED, thresholds=STANDARD).ok


def test_six_columns_name_the_guide_and_start_columns_when_a_row_carries_them() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "テスト")]
    nine = validate_correction_output_text(six("sub|1|1.0|0.0|テスト|测试|high|2|"), source, variant=GROUPED, thresholds=STANDARD)
    assert any("unexpected duration, gap and char_count columns" in error for error in nine.errors)
    ten = validate_correction_output_text(six("sub|1|0.0|1.0|0.0|テスト|测试|high|2|"), source, variant=GROUPED, thresholds=STANDARD)
    assert any("unexpected start, duration, gap and char_count columns" in error for error in ten.errors), (
        "the start column check runs without the guide columns too")
    first = validate_correction_output_text(six("sub|1|1.0||テスト|测试|high|2|"), source, variant=GROUPED, thresholds=STANDARD)
    assert any("unexpected duration, gap and char_count columns" in error for error in first.errors), (
        "a copied first row has no pause before it, so its gap cell is empty")


def test_basic_a_takes_seven_columns_and_names_the_guide_columns_when_a_row_carries_them() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "テスト")]
    basic = resolve_variant("basicA")
    assert validate_correction_output_text(
        f"<translated>\n{BASIC_SEVEN}\nsub|1|0.0|テスト|测试|high|\n</translated>", source, variant=basic, thresholds=STANDARD).ok
    ten = validate_correction_output_text(
        f"<translated>\n{BASIC_SEVEN}\nsub|1|0.0|1.0|0.0|テスト|测试|high|2|\n</translated>", source, variant=basic, thresholds=STANDARD)
    assert any("unexpected duration, gap and char_count columns" in error for error in ten.errors), ten.errors
    piped = validate_correction_output_text(
        f"<translated>\n{BASIC_SEVEN}\nsub|1|0.0|テスト|测试|high|a|b\n</translated>", source, variant=basic, thresholds=STANDARD)
    assert any("every row has exactly 7" in error for error in piped.errors), piped.errors


def test_validate_basic_start_column_contract() -> None:
    source = [SubtitleSegment("1", 2.5, 3.5, "a")]
    content = (
        f"<translated>\n{BASIC_SEVEN}\n"
        "sub|1|2.5|a|甲|high|\n</translated>"
    )
    result = validate_translated_csv_text(
        content,
        source,
        require_headers=True,
        require_start_column=True,
    )
    assert result.ok, result.errors

    missing_start = content.replace("|2.5|", "|")
    rejected = validate_translated_csv_text(
        missing_start,
        source,
        require_headers=True,
        require_start_column=True,
    )
    assert not rejected.ok


def test_window_validator_restores_local_positions_to_source_ids() -> None:
    sources = [
        SubtitleSegment("188", 12.3, 13.0, "a"),
        SubtitleSegment("189", 13.1, 14.0, "b"),
    ]
    window = SubtitleWindow(
        chunk_id="0002",
        segments=sources,
        overlap_segments=[],
        boundary_reason="test",
        budget=None,  # type: ignore[arg-type]
        length_scale=1.0,
    )
    text = f"<translated>\n{SIX}\nsub|1,2|ab|甲乙|high|\n</translated>"

    result = validate_correction_window_output(
        text,
        window,
        variant=resolve_variant("capableC"),
    )

    assert result.ok, result.errors
    assert result.segments[0].source_ids == ("188", "189")


def test_window_validator_rejects_reference_ids() -> None:
    source = [SubtitleSegment("188", 12.3, 13.0, "a")]
    window = SubtitleWindow(
        chunk_id="0002",
        segments=source,
        overlap_segments=[],
        boundary_reason="test",
        budget=None,  # type: ignore[arg-type]
        length_scale=1.0,
    )
    for invalid_id in ("0", "-1"):
        text = f"<translated>\n{SIX}\nsub|{invalid_id}|a|甲|high|\n</translated>"
        result = validate_correction_window_output(
            text,
            window,
            variant=resolve_variant("capableC"),
        )
        assert not result.ok
        assert any("unknown source id" in error for error in result.errors)


def test_translated_csv_merges_sources_and_restores_srt_newlines() -> None:
    source = [
        SubtitleSegment("3", 1.001, 1.501, "你"),
        SubtitleSegment("4", 2.001, 3.001, "好"),
        SubtitleSegment("5", 4.001, 9.101, "好好好好好好好好好好"),
        SubtitleSegment("6", 70.123, 71.023, "你好"),
    ]
    output = (
        "<translated>\n"
        "sub|3,4|good morning|你好|high|\n"
        "discard|5|重复幻觉\n"
        r"sub|6|source line|第一行\n第二行|high|"
        "\n</translated>"
    )

    result = validate_translated_csv_text(output, source)
    srt = render_translated_segments_as_srt(result.segments)
    corrected_srt = render_corrected_segments_as_srt(result.segments)
    srt_segments = parse_srt(srt)
    corrected_segments = parse_srt(corrected_srt)

    assert result.ok
    assert result.segments[0].source_ids == ("3", "4")
    assert result.segments[0].corrected_text == "good morning"
    assert result.segments[0].translation == "你好"
    assert result.segments[0].start == 1.001
    assert result.segments[0].end == 3.001
    assert [segment.text for segment in srt_segments] == ["你好", "第一行\n第二行"]
    assert [segment.text for segment in corrected_segments] == [
        "good morning",
        "source line",
    ]
    assert srt_segments[1].start == 70.123
    assert srt_segments[1].end == 71.023


def test_translated_csv_collapses_consecutive_newlines_on_parse() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.5, 2.5, "二"),
    ]
    output = (
        "<translated>\n"
        r"sub|1|line a\n\nline b|译一\n\n\n译二|high|"
        "\n"
        r"sub|2|单行|单译|high|"
        "\n</translated>"
    )

    result = validate_translated_csv_text(output, source)

    assert result.ok
    assert result.segments[0].corrected_text == "line a\nline b"
    assert result.segments[0].translation == "译一\n译二"
    assert result.segments[1].corrected_text == "单行"
    assert result.segments[1].translation == "单译"


def test_translated_csv_rejects_bad_blocks_rows_and_source_ids() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.5, 2.5, "二"),
    ]

    cases = [
        "1|一",
        "<translated>\nsub|9|nine|九|high|\n</translated>",
        "<translated>\nsub|1|one|一|high|\nsub|1|repeat|重复|high|\n</translated>",
        "<translated>\n1\n</translated>",
        "<translated>\n1|one\n</translated>",
        "<translated>\nsub|1||一|high|\n</translated>",
        "<translated>\n1|one|\n</translated>",
    ]

    for content in cases:
        assert not validate_translated_csv_text(content, source).ok


def test_empty_translated_block_no_longer_drops_the_window() -> None:
    """Inverted deliberately: emptiness is a truncated reply, not a wipe.

    This test used to assert the opposite. Treating an empty block as an
    intentional "drop the whole window" made a cut-off reply indistinguishable
    from a deliberate one, and the commit path then deleted rows earlier
    windows had already produced for the overlap ids.
    """
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text("<translated></translated>", source)

    assert not result.ok
    assert result.segments == []
    assert render_translated_segments_as_srt(result.segments) == ""


def test_merge_keeps_the_earlier_windows_answer_on_the_overlap() -> None:
    # The new window also answered the overlap line 2, and even discarded it: the overlap is the earlier
    # window's core, so its row stays and the new window's say on it does not count.
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
        SubtitleSegment("3", 2.0, 3.0, "三"),
    ]
    first = validate_translated_csv_text(
        "<translated>\n"
        "sub|1|one|一|high|\n"
        "sub|2|two|二|high|\n</translated>",
        source[:2]).segments
    second = validate_translated_csv_text(
        "<translated>\n"
        "sub|3|three|三|high|\n"
        "discard|2|窗口重叠，由本窗重写\n</translated>",
        source[1:]).segments

    merged = merge_translated_csv_windows(first, ["2", "3"], second, overlap_ids=["2"])

    assert [segment.source_ids for segment in merged] == [("1",), ("2",), ("3",)]
    assert merged[1].translation == "二"


def _translated(source_ids: tuple[str, ...], start: float, end: float) -> object:
    from finesub.llm.output_protocol import TranslatedCsvSegment

    return TranslatedCsvSegment(
        source_ids=source_ids,
        start=start,
        end=end,
        corrected_text="src " + ",".join(source_ids),
        translation="译 " + ",".join(source_ids),
    )


def test_merge_drops_core_ids_the_new_window_discarded() -> None:
    old = [_translated(("1",), 0.0, 1.0), _translated(("2",), 1.0, 2.0)]
    new = [_translated(("2",), 1.0, 2.0), _translated(("3",), 2.0, 3.0)]  # 4 discarded

    merged = merge_translated_csv_windows(old, ["2", "3", "4"], new, overlap_ids=["2"])

    assert [segment.source_ids for segment in merged] == [("1",), ("2",), ("3",)]
    assert [segment.translation for segment in merged][:2] == ["译 1", "译 2"]
    assert window_merge_gaps(old, ["2", "3", "4"], new, merged, overlap_ids=["2"]) == ((), ())


def test_a_row_merged_across_the_cut_takes_its_overlap_ids_along() -> None:
    old = [_translated(("1",), 0.0, 1.0), _translated(("2",), 1.0, 2.0), _translated(("3",), 2.0, 3.0)]
    new = [_translated(("2",), 1.0, 2.0), _translated(("3", "4"), 2.0, 4.0), _translated(("5",), 4.0, 5.0)]

    merged = merge_translated_csv_windows(old, ["2", "3", "4", "5"], new, overlap_ids=["2", "3"])

    assert [segment.source_ids for segment in merged] == [("1",), ("2",), ("3", "4"), ("5",)]
    assert merged[1].translation == "译 2"  # still the earlier window's
    assert window_merge_gaps(old, ["2", "3", "4", "5"], new, merged, overlap_ids=["2", "3"]) == ((), ())


def test_taking_over_grows_until_no_row_inside_the_window_is_cut() -> None:
    # The new {3,4} crosses the cut and takes 3; the earlier {2,3} would be cut, so 2 goes over too, and
    # with it the new {2}.
    old = [_translated(("1",), 0.0, 1.0), _translated(("2", "3"), 1.0, 3.0)]
    new = [_translated(("2",), 1.0, 2.0), _translated(("3", "4"), 2.0, 4.0)]

    merged = merge_translated_csv_windows(old, ["2", "3", "4"], new, overlap_ids=["2", "3"])

    assert [segment.source_ids for segment in merged] == [("1",), ("2",), ("3", "4")]
    assert merged[1].translation == "译 2"
    assert window_merge_gaps(old, ["2", "3", "4"], new, merged, overlap_ids=["2", "3"]) == ((), ())


def test_an_earlier_row_reaching_past_the_window_start_stays_whole_and_the_id_shows_twice() -> None:
    old = [_translated(("1", "2", "3"), 0.0, 3.0)]
    new = [_translated(("2",), 1.0, 2.0), _translated(("3", "4"), 2.0, 4.0)]

    merged = merge_translated_csv_windows(old, ["2", "3", "4"], new, overlap_ids=["2", "3"])

    assert [segment.source_ids for segment in merged] == [("1", "2", "3"), ("3", "4")]
    assert window_merge_gaps(old, ["2", "3", "4"], new, merged, overlap_ids=["2", "3"]) == ((), ("3",))


def test_window_merge_gaps_names_lost_ids_and_lets_discards_go() -> None:
    old = [_translated(("1",), 0.0, 1.0), _translated(("2",), 1.0, 2.0), _translated(("3",), 2.0, 3.0)]
    new = [_translated(("3",), 2.0, 3.0), _translated(("4",), 3.0, 4.0)]  # the window discarded 2
    # a merge that lost 4 (the new window's) and 1 (outside the window)
    lossy = [_translated(("3",), 2.0, 3.0)]

    assert window_merge_gaps(old, ["2", "3", "4"], new, lossy, overlap_ids=["2", "3"]) == (("1", "4"), ())
    merged = merge_translated_csv_windows(old, ["2", "3", "4"], new, overlap_ids=["2", "3"])
    assert [segment.source_ids for segment in merged] == [("1",), ("2",), ("3",), ("4",)]
    assert window_merge_gaps(old, ["2", "3", "4"], new, merged, overlap_ids=["2", "3"]) == ((), ())


def test_an_overlap_id_the_earlier_window_discarded_is_not_reported_missing() -> None:
    # The earlier window discarded 2; the new window answered it in its overlap, which does not count.
    old = [_translated(("1",), 0.0, 1.0)]
    new = [_translated(("2",), 1.0, 2.0), _translated(("3",), 2.0, 3.0)]

    merged = merge_translated_csv_windows(old, ["2", "3"], new, overlap_ids=["2"])

    assert [segment.source_ids for segment in merged] == [("1",), ("3",)]
    assert window_merge_gaps(old, ["2", "3"], new, merged, overlap_ids=["2"]) == ((), ())


def test_commit_window_warns_on_a_merge_gap_and_keeps_going(monkeypatch) -> None:
    from finesub import reporting
    from finesub.llm.stages.correction import context

    class Recorder(reporting.NullReporter):
        def __init__(self) -> None:
            self.warnings: list[tuple[str, str]] = []

        def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
            self.warnings.append((code, message))

    run = context.CorrectionRun.__new__(context.CorrectionRun)  # only what commit_window touches
    run.rendered_segments, run.grouping, run.progress, run.carried = [_translated(("1",), 0.0, 1.0)], None, None, None
    window = SimpleNamespace(chunk_id="0002", source_ids=["2", "3"], overlap_segments=[], grouped=False)
    monkeypatch.setattr(context, "merge_translated_csv_windows", lambda existing, ids, rows, **_: [rows[0]])
    rows = [_translated(("2",), 1.0, 2.0), _translated(("3",), 2.0, 3.0)]

    with reporting.reporting_to(Recorder()) as recorder:
        run.commit_window(window, CsvValidationResult(True, rows, [], []), "")

    assert [segment.source_ids for segment in run.rendered_segments] == [("2",)]
    [(code, message)] = recorder.warnings
    assert code == "correction-merge-coverage" and "0002" in message and "1,3" in message


def test_translated_csv_skips_plan_lines() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "plan|单源；字少gap正常，不合并。\n"
        "sub|1|one|一|8|译1字\n"
        "plan|两源口播碎片gap小→合并；ASR「新書」应为「新衣装」。\n"
        "sub|2|two|二|7|译1字；短句\n"
        "PLAN|大小写也应跳过\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source)

    assert result.ok
    assert len(result.segments) == 2
    assert [seg.source_ids for seg in result.segments] == [("1",), ("2",)]
    assert not any("unknown type" in w for w in result.warnings)


def test_translated_csv_parses_conf_and_note_columns() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "sub|1|one|一|high|术语note\n"
        "|2|two|二|median|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source)

    assert result.ok
    first, second = result.segments
    assert first.kind == "sub" and first.conf == "high"
    assert first.note == "术语note"
    assert second.kind == "sub" and second.conf == "median"
    assert second.note == ""


def test_translated_csv_conf_out_of_range_degrades_without_failing() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text(
        "<translated>\nsub|1|one|一|42|\n</translated>", source)

    assert result.ok
    assert result.segments[0].conf is None
    assert any("invalid conf" in warning for warning in result.warnings)


def test_translated_csv_maps_legacy_numeric_confidence_to_tiers() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]
    result = validate_translated_csv_text(
        "<translated>\nsub|1|one|一|8|\n</translated>",
        source,
    )
    assert result.ok
    assert result.segments[0].conf == "high"


def test_insert_rows_are_rejected_now_that_v63_retired_them() -> None:
    """Replaces the old clip-relative-timing test: the emit path is gone.

    Both production call sites had already pinned `allow_insert=False`, so the
    parse/dedup/merge machinery behind this row kind was unreachable. The type
    name survives only because knowledge materials read it back out of older
    `annotated.csv`.
    """
    source = [
        SubtitleSegment("20", 105.0, 106.0, "I think"),
        SubtitleSegment("22", 109.5, 110.7, "that's right"),
    ]
    output = (
        "<translated>\n"
        "sub|20|I think|我觉得|high|\n"
        "insert|7.0,0.8|まって|等一下|median|漏识别\n"
        "sub|22|that's right|没错|high|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, clip_start=100.0)

    assert not result.ok
    assert any("no longer part of the output contract" in e for e in result.errors)
    assert all(segment.kind == "sub" for segment in result.segments)


def test_translated_csv_rejects_bad_insert_timing() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    for bad in ("insert|abc|x|译|5|", "insert|1.0|x|译|5|", "insert|1.0,0|x|译|5|"):
        result = validate_translated_csv_text(f"<translated>\n{bad}\n</translated>", source)
        assert not result.ok


def test_render_translated_segments_as_csv_computes_the_product_columns() -> None:
    segments = [
        TranslatedCsvSegment(("3", "4"), 1.0, 3.0, "good morning", "你好", conf="high", note="n|1"),
        TranslatedCsvSegment(("5",), 7.0, 7.8, "wait", "等一下", conf="median"),
    ]

    text = render_translated_segments_as_csv(segments)

    # duration, gap and the weighted count of the translation are computed here
    assert text.splitlines() == [
        "sub|3,4|2.0||good morning|你好|high|2|n｜1",
        "sub|5|0.8|4.0|wait|等一下|median|3|",
    ]


def test_translated_csv_over_cap_merges_warn_not_reject() -> None:
    # Relaxed 2026-07-20: the merged-source count is no longer a hard reject.
    # The prompt still tells the model to keep to two consecutive sources, but
    # validation only records a warning when a row exceeds the soft cap, so the
    # models' natural merge behavior can be observed without forced retries.
    source = [
        SubtitleSegment("1", 0.0, 1.0, "うちらが知ってるのは"),
        SubtitleSegment("2", 1.0, 1.6, "ちょっと"),
        SubtitleSegment("3", 2.0, 3.0, "少年期だけなんだよね"),
        SubtitleSegment("4", 3.0, 4.0, "それは"),
    ]
    row = "sub|{pos}|x|译文|high|\n"
    two = validate_translated_csv_text(
        "<translated>\n" + row.format(pos="1,2")
        + row.format(pos="3") + row.format(pos="4") + "</translated>",
        source,
    )
    assert two.ok
    assert not any("soft cap" in w for w in two.warnings)

    three = validate_translated_csv_text(
        "<translated>\n" + row.format(pos="1,2,3") + row.format(pos="4")
        + "</translated>",
        source,
    )
    assert three.ok
    assert any("soft cap" in w for w in three.warnings)

    four = validate_translated_csv_text(
        "<translated>\n" + row.format(pos="1,2,3,4") + "</translated>",
        source,
    )
    assert four.ok
    assert any("soft cap" in w for w in four.warnings)


def test_translated_csv_voids_row_with_marker_in_a_column() -> None:
    # Models sometimes drop <void> into the conf column with trailing cells
    # after it instead of at the row end. The row is still a retraction, so its
    # source ids stay free and the rewrite that follows must not collide.
    source = [
        SubtitleSegment("65", 0.0, 2.5, "パジャマで部屋に集まり直して"),
        SubtitleSegment("66", 2.5, 4.7, "ケーキを4人で"),
    ]
    text = (
        "<translated>\n"
        "sub|65,66|合并稿|合并译文|<void>|\n"
        "sub|65|パジャマで部屋に集まり直して|重新集合|high|\n"
        "sub|66|ケーキを4人で|四人分蛋糕|high|\n"
        "</translated>"
    )
    result = validate_translated_csv_text(text, source)
    assert result.ok, result.errors
    assert result.voided_rows == 1
    # The voided merge freed 65/66, so the two single rows are the only output.
    assert [seg.source_ids for seg in result.segments] == [("65",), ("66",)]


def test_translated_csv_rejects_non_consecutive_merges() -> None:
    # Dropping the middle row and merging around it would lie at the
    # source-id layer; merged sources must be adjacent.
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
        SubtitleSegment("3", 2.0, 3.0, "三"),
    ]
    skip = validate_translated_csv_text(
        "<translated>\nsub|1,3|一三|一三|high|\n</translated>",
        source,
    )
    assert not skip.ok
    assert any("adjacent" in error for error in skip.errors)


def test_void_marker_drops_row_and_frees_its_source_ids() -> None:
    # v12: a row ending with <void> is retracted by the model — dropped before
    # any structural checks, and its ids may be re-emitted by later rows.
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
        SubtitleSegment("3", 2.0, 3.0, "三"),
    ]
    output = (
        "<translated>\n"
        "sub|1,2,3|runaway merge|失控合并|7|<void>\n"
        "sub|1,2|one two|一二|8|\n"
        "sub|3|three|三|8|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source)

    assert result.ok
    assert result.voided_rows == 1
    assert [seg.source_ids for seg in result.segments] == [("1", "2"), ("3",)]
    assert any("<void>" in warning for warning in result.warnings)


def test_void_marker_skips_structural_validation_of_the_voided_row() -> None:
    # A retracted row is treated as nonexistent even if it is malformed
    # (too few fields, unknown ids) or upper-cased.
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]
    output = (
        "<translated>\n"
        "sub|999|garbage<void>\n"
        "sub|1|one|一|8|中途放弃<VOID>\n"
        "sub|1|one|一|8|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source)

    assert result.ok
    assert result.voided_rows == 2
    assert [seg.source_ids for seg in result.segments] == [("1",)]


def test_all_rows_voided_counts_as_no_valid_rows() -> None:
    # Voiding everything without re-emitting is an incomplete output -> retry.
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text(
        "<translated>\nsub|1|one|一|8|<void>\n</translated>", source)

    assert not result.ok
    assert result.voided_rows == 1
    assert any("no valid rows" in error for error in result.errors)


def test_unclosed_translated_block_looks_truncated() -> None:
    assert looks_truncated_translated("<translated>\nsub|1|one|一|high|\n")
    assert not looks_truncated_translated("<translated></translated>")


def test_pacing_scorer_step_penalties_and_pass_ratio() -> None:
    from finesub.llm.output_protocol import score_translated_segments

    good = TranslatedCsvSegment(("1", "2"), 0.0, 3.0, "src", "正常长度的一行字幕")
    bad = TranslatedCsvSegment(
        tuple(str(i) for i in range(3, 30)), 10.0, 74.8, "src",
        '\n'.join(["超长的一行字幕文本内容再多塞一点字数超过二十个汉字"] * 5),
    )
    report = score_translated_segments([good, bad])

    assert report["rows"][0]["penalty"] == 0.0
    # bad: span>25 (+4), 5 lines (+4), char excess capped (+2) = 10
    assert report["rows"][1]["penalty"] == 10.0
    assert report["critical_rows"] == 1
    # 10 / (2 rows + 5) > 0.3 -> fails
    assert report["normalized_penalty"] > 0.3 and not report["passed"]
    ok = score_translated_segments([good] * 20)
    assert ok["passed"] and ok["total_penalty"] == 0.0


def test_pacing_scorer_uses_shared_half_weight_for_latin() -> None:
    from finesub.llm.output_protocol import score_translated_segments

    # 40 Latin letters = 20 weighted -> no excess; 42 = 21 -> 0.1 penalty.
    at_limit = TranslatedCsvSegment(("1",), 0.0, 3.0, "src", "a" * 40)
    over_limit = TranslatedCsvSegment(("1",), 0.0, 3.0, "src", "a" * 42)
    assert score_translated_segments([at_limit])["rows"][0]["penalty"] == 0.0
    assert score_translated_segments([over_limit])["rows"][0]["penalty"] == 0.1


def test_validate_requires_full_coverage_and_top_level_translated() -> None:
    source = [
        SubtitleSegment("1", 0.0, 0.8, "a"),
        SubtitleSegment("2", 0.9, 1.5, "b"),
    ]
    # Reasoning name-drops must not break extraction / validation.
    good = (
        "<reasoning>\n先想清楚再写 `<translated>`\n</reasoning>\n"
        "<translated>\n"
        "sub|1,2|a b|甲乙|8|译2字\n"
        "</translated>"
    )
    ok = validate_translated_csv_text(good, source)
    assert ok.ok, ok.errors
    assert ok.segments[0].source_ids == ("1", "2")

    # v54: silent omission no longer allowed — missing source 2 is an error
    truncated = "<translated>\nsub|1|a|甲|8|译1字\n</translated>"
    result = validate_translated_csv_text(truncated, source)
    assert not result.ok
    assert any("missing source id" in e for e in result.errors)

    # Explicit discard makes it pass
    with_discard = "<translated>\nsub|1|a|甲|8|译1字\ndiscard|2|幻觉\n</translated>"
    result_discard = validate_translated_csv_text(with_discard, source)
    assert result_discard.ok
    assert result_discard.discarded_ids == ("2",)

def test_discard_rows_participate_in_order_and_duplicate_validation() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "a"),
        SubtitleSegment("2", 1.0, 2.0, "b"),
    ]
    out_of_order = (
        "<translated>\n"
        "discard|2|幻觉\n"
        "sub|1|a|甲|high|\n"
        "</translated>"
    )
    duplicate = (
        "<translated>\n"
        "sub|1|a|甲|high|\n"
        "discard|2|幻觉\n"
        "discard|2|重复决定\n"
        "</translated>"
    )

    reversed_result = validate_translated_csv_text(
        out_of_order, source
    )
    duplicate_result = validate_translated_csv_text(
        duplicate, source
    )

    assert not reversed_result.ok
    assert any(
        "before an earlier output row" in error for error in reversed_result.errors
    )
    assert not duplicate_result.ok
    assert any("discarded more than once" in error for error in duplicate_result.errors)


def test_validate_rejects_prose_swallowed_first_match_translated() -> None:
    """Old regex would start at the mid-reasoning `<translated>` mention."""
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    text = (
        "<reasoning>\n务必输出 `<translated>` 终稿\n</reasoning>\n"
        "<translated>\nsub|1|a|甲|8|译1字\n</translated>"
    )
    result = validate_translated_csv_text(text, source)
    assert result.ok, result.errors


# ---------------------------------------------------------------------------
# `#` lines (once capableC's reasoning comments; no prompt asks for them now): skipped for SRT, counted
# ---------------------------------------------------------------------------


def test_reasoning_rows_are_skipped_and_do_not_cover() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    # A reasoning comment above the merged sub row; both sources covered.
    output = (
        "<translated>\n"
        "# gap=0.0 同一句切开，合并后 2 字在界内\n"
        "sub|1,2|one two|一二|median|同一句合并\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source)
    assert result.ok, result.errors
    # The reasoning comment never becomes a segment.
    assert len(result.segments) == 1
    assert result.segments[0].source_ids == ("1", "2")


def test_reasoning_row_does_not_satisfy_coverage() -> None:
    """A source mentioned only in a reasoning comment is still uncovered."""
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "sub|1|one|一|high|\n"
        "# 说明为何丢弃源2\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source)
    assert not result.ok
    assert any("missing source id" in e for e in result.errors)


def test_reasoning_comment_before_discard() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "sub|1|one|一|high|\n"
        "# 复读幻觉，三特征叠加\n"
        "discard|2|复读幻觉\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source)
    assert result.ok, result.errors
    assert result.discarded_ids == ("2",)


# --- Empty <translated> is a degenerate reply, not an intentional wipe -------
#
# `discard|<id>` is the explicit channel for dropping sources, and v52 coverage
# already requires every id to be covered or discarded. An empty block is a
# redundant implicit channel that a truncated reply lands in by accident, so it
# must fail validation and let the retry correct it.


def _header_only_output() -> str:
    return (
        f"<translated>\n{SIX}\n</translated>"
    )


def test_header_only_translated_is_rejected_when_window_has_sources() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "one")]
    result = validate_translated_csv_text(
        _header_only_output(), source, require_headers=True
    )
    assert not result.ok
    assert result.segments == []
    assert any("discard" in e for e in result.errors), result.errors


def test_rowless_translated_is_rejected_without_header_requirement() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "one")]
    output = "<translated>\n</translated>"
    result = validate_translated_csv_text(output, source)
    assert not result.ok
    assert result.segments == []


def test_discarding_every_source_is_rejected_too() -> None:
    """A window that yields no rows at all fails, whichever way it got there.

    Pinned deliberately: `discard` covers individual ids, but discarding the
    whole window still trips "no valid rows". Before the empty-block fix the
    two paths disagreed -- all-discard failed while an empty block silently
    wiped the window -- which is exactly backwards. This test exists so the
    two stay consistent.
    """
    source = [
        SubtitleSegment("1", 0.0, 1.0, "one"),
        SubtitleSegment("2", 1.0, 2.0, "two"),
    ]
    output = (
        "<translated>\n"
        "discard|1|复读幻觉\n"
        "discard|2|复读幻觉\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source)
    assert not result.ok
    assert result.segments == []
    # Coverage is satisfied -- the rejection is "no rows", not "missing ids".
    assert set(result.discarded_ids) == {"1", "2"}
    assert any("no valid rows" in e for e in result.errors), result.errors


def _mostly_discarded_output(kept: int, discarded: int) -> tuple[str, list[SubtitleSegment]]:
    source = [
        SubtitleSegment(str(i), float(i), float(i) + 1.0, f"line {i}")
        for i in range(1, kept + discarded + 1)
    ]
    rows = [
        f"sub|{i}|line {i}|第{i}行|high|" for i in range(1, kept + 1)
    ]
    rows += [
        f"discard|{i}|复读幻觉"
        for i in range(kept + 1, kept + discarded + 1)
    ]
    return "<translated>\n" + "\n".join(rows) + "\n</translated>", source


def test_a_window_that_keeps_almost_nothing_is_rejected() -> None:
    """The 2026-08-22 canary: one `sub` row plus `discard` for everything else.

    Every structural check passed -- ids covered, order kept, one valid row --
    and the finished subtitle kept a single line. An agent that cannot see the
    window text produces exactly this shape, so the rejection has to happen
    here, at the same seam `agent-task lint` uses, and not by eye downstream.
    """
    output, source = _mostly_discarded_output(kept=1, discarded=59)
    result = validate_translated_csv_text(output, source)
    assert not result.ok
    assert any("at or under the floor" in e for e in result.errors), result.errors
    # Not the coverage error: every id *was* accounted for.
    assert not any("missing source id" in e for e in result.errors), result.errors


@pytest.mark.parametrize(
    ("kept", "discarded", "ok"),
    [
        (10, 30, False),  # 40 sources: the floor is min(10, 20 - 10) = 10
        (11, 30, True),
        (10, 290, False),  # big windows: still 10, however many were dropped
        (11, 289, True),
        (5, 25, False),  # 30 sources: the floor is min(10, 15 - 10) = 5
        (6, 24, True),
        (2, 4, True),  # a short, mostly-song clip: 6 sources, the floor is negative
        (5, 15, True),  # 20 sources: floor 0, only all-discard is left
    ],
)
def test_the_kept_floor_is_min_10_or_half_the_window_less_10(kept, discarded, ok) -> None:
    """Owner rule 2026-09-30: reject when kept <= min(10, total * 0.5 - 10)."""

    output, source = _mostly_discarded_output(kept=kept, discarded=discarded)
    result = validate_translated_csv_text(output, source)
    assert result.ok is ok, result.errors


def _mostly_discarded_window_output(
    kept: int, discarded: int
) -> tuple[str, list[SubtitleSegment]]:
    """`_mostly_discarded_output` in the served window contract: header
    required (capableC)."""

    body, sources = _mostly_discarded_output(kept, discarded)
    return body.replace("<translated>\n", f"<translated>\n{SIX}\n", 1), sources


def _window(chunk_id: str, sources: list[SubtitleSegment]) -> SubtitleWindow:
    return SubtitleWindow(
        chunk_id=chunk_id,
        segments=sources,
        overlap_segments=[],
        boundary_reason="test",
        budget=None,  # type: ignore[arg-type]  -- unused by validation
        length_scale=1.0,
    )


def test_a_split_leaf_is_held_to_the_same_floor() -> None:
    """A leaf is its own call, so a reply that never saw the body can land
    there too. The old 50% ratio could not be applied to leaves (the worst
    correct half in the archive discards 43.8%); an absolute floor can."""

    output, sources = _mostly_discarded_window_output(kept=1, discarded=59)
    leaf = _window("0007-a", sources)
    result = validate_correction_window_output(
        output, leaf, variant=resolve_variant("capableC")
    )
    assert not result.ok
    assert any("at or under the floor" in e for e in result.errors), result.errors
    # The 43.8% half that passed before still passes.
    output, sources = _mostly_discarded_window_output(kept=72, discarded=56)
    result = validate_correction_window_output(
        output, _window("0007-a", sources), variant=resolve_variant("capableC")
    )
    assert result.ok, result.errors


def test_split_depth_has_one_owner() -> None:
    """The retry loop's split budget reads the lineage off the window id.

    A second copy of "the id is the lineage" anywhere is the silent drift this
    repo keeps getting bitten by, so every reader goes through the window.
    """

    from finesub.llm.stages.correction.context import WindowGeometry

    for chunk_id, depth in (("0007", 0), ("0007-a", 1), ("0007-a-b", 2)):
        window = _window(chunk_id, [SubtitleSegment("1", 0.0, 1.0, "x")])
        assert window.split_depth == depth
        assert WindowGeometry.split_depth(window) == depth


def test_empty_translated_stays_ok_when_the_window_has_no_sources() -> None:
    """Nothing to cover means nothing to report -- the guard must not overfire."""
    result = validate_translated_csv_text(
        "<translated>\n</translated>", []
    )
    assert result.ok, result.errors
    assert result.segments == []


def test_merge_still_clears_core_ids_and_validation_is_what_guards_it() -> None:
    """Pins where the guard lives, so nobody "fixes" it in the wrong layer.

    `merge_translated_csv_windows` cannot tell a truncated window from one that
    legitimately discarded its ids -- the window owns its core and real
    `discard` rows depend on it. So the merge keeps clearing the core's ids,
    and the protection against a *truncated* window reaching it lives in
    validation: the empty block is rejected before commit. Changing the merge
    to preserve prior rows would silently resurrect discarded subtitles.
    """
    # 1. Validation is the layer that stops a truncated window.
    source = [SubtitleSegment("8", 8.0, 9.0, "eight")]
    rejected = validate_translated_csv_text(
        "<translated></translated>", source
    )
    assert not rejected.ok, "validation is the layer that must stop this"

    # 2. The merge itself deliberately still clears the core's ids (9); the overlap (8) is the earlier window's.
    earlier = [
        TranslatedCsvSegment(
            source_ids=("8",),
            start=8.0,
            end=9.0,
            corrected_text="eight",
            translation="八",
        ),
        TranslatedCsvSegment(
            source_ids=("9",),
            start=9.0,
            end=10.0,
            corrected_text="nine",
            translation="九",
        ),
    ]
    assert merge_translated_csv_windows(earlier, ["8", "9", "10"], [], overlap_ids=["8"]) == earlier[:1]


# --- Strict arity: a wrong column count must never be reinterpreted ----------
#
# The parser used to guess the layout from cell *content*, so a row with one
# extra column was silently shifted one cell left -- putting the untranslated
# source text into the translation column. Columns are now mapped by position
# and a row must have exactly the header's cells.


def _one_source() -> list[SubtitleSegment]:
    return [SubtitleSegment("1", 0.0, 1.0, "hello there")]


def test_extra_column_is_a_structural_error_not_a_silent_shift() -> None:
    """The SUMMARY sample: one cell too many used to shift every field left."""
    output = (
        "<translated>\n"
        f"{SIX}\n"
        "sub|1|0.2|hello there|你好|high|\n"
        "</translated>"
    )
    result = validate_translated_csv_text(
        output, _one_source(), require_headers=True
    )
    assert not result.ok, "a 7-field row against a 6-column spec must fail"
    # The specific corruption this guards: source text becoming the translation.
    assert all(s.translation != "hello there" for s in result.segments)


def test_half_width_pipe_in_translation_is_rejected_not_dropped() -> None:
    """Used to warn and silently discard the text after the pipe."""
    output = (
        "<translated>\n"
        "sub|1|hello|你好|世界|high|\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, _one_source())
    assert not result.ok
    assert all("世界" not in (s.translation or "") for s in result.segments)


def test_legacy_three_column_layout_is_no_longer_accepted() -> None:
    """Dropped with the heuristic parser; annotated.csv is regenerable.

    The row below is deliberately the old `position|corrected|translation`
    shape -- do not "modernise" it, that is the whole point of the test.
    """
    output = "<translated>\n" + "1|one|" + "一" + "\n</translated>"
    result = validate_translated_csv_text(output, _one_source())
    assert not result.ok
    assert any("every row has exactly 6" in e for e in result.errors), result.errors


def _as_a_broken_cli_would_write(text: str) -> str:
    return "".join(
        character if ord(character) < 128 else f"\\u{ord(character):04x}"
        for character in text
    )


def _reply_over(rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(
        f"sub|{position}|{source}|{translation}|high|"
        for position, source, translation in rows
    )
    return f"<translated>\n{SIX}\n{body}\n</translated>"


_CJK_ROWS = [
    ("1", "こんにちは皆さん今日もよろしく", "大家好今天也请多关照"),
    ("2", "今日は配信の話をしましょうか", "今天来聊聊直播的事吧"),
    ("3", "それでは始めていきましょう", "那么我们就开始吧"),
]
_CJK_SOURCE = [
    SubtitleSegment(position, index * 2.0, index * 2.0 + 1.0, source)
    for index, (position, source, _) in enumerate(_CJK_ROWS)
]


def test_an_intact_reply_is_still_accepted() -> None:
    """The control. Without it the refusal below would prove nothing about
    the predicate and everything about the fixture."""

    result = validate_translated_csv_text(_reply_over(_CJK_ROWS), _CJK_SOURCE)

    assert result.ok, result.errors


def test_an_escaped_reply_is_refused_as_a_transport_fault() -> None:
    r"""A reply whose non-ASCII arrived as literal escapes is refused, not scored.

    Measured once end to end: the window was validated, normalized, delivered
    into the SRT and written to the resume cache, with the stage reporting
    success the whole way. Only a comparison with the model's own `char_count`
    column noticed -- 15 rows, every one off by a factor of three -- and it
    normalized the evidence away (no contract carries that column now), which
    is why the refusal has to come before any of that.
    """

    escaped = _as_a_broken_cli_would_write(_reply_over(_CJK_ROWS))
    assert escaped.isascii()

    result = validate_translated_csv_text(escaped, _CJK_SOURCE)

    assert not result.ok
    assert result.segments == []
    # One error, not a pile of secondary parse complaints: every check below
    # reads this text, and all of them would have something to say about it.
    assert len(result.errors) == 1
    message = result.errors[0].lower()
    # Named as what it is. "row 1 is malformed" would send the next reader to
    # look at the model's content, which is not where the fault is.
    assert "transport" in message
    assert "backslash" in message


def test_a_stray_literal_escape_now_costs_the_window() -> None:
    r"""The accepted trade, written down as behaviour (owner, 2026-09-04).

    A reply that is ASCII end to end *and* carries a literal escape is refused,
    even though a model may have meant those six characters. The count floor
    that used to spare it also dropped the real fault on small windows, and
    this pipeline's correction target is fixed Chinese (`PROMPT_VERSION`), so a
    legitimate reply is essentially never pure ASCII -- the case given up here
    is close to unreachable, and it fails loudly when it happens.
    """

    rows = [("1", "hello everyone", r"hi there \u2019 as written")]
    source = [SubtitleSegment("1", 0.0, 1.0, "hello everyone")]
    reply = _reply_over(rows)
    assert reply.isascii(), "the point of this case is a legitimately ASCII reply"

    result = validate_translated_csv_text(reply, source)

    assert not result.ok
    assert "transport" in result.errors[0].lower()


class TestLeakedThoughtReply:
    """agy sometimes ships its reasoning as the answer: the reply opens with
    `thought\\n` and English prose (docs/llm_local_agent_agy.md §7). The callers
    validate the answer it ends with (`strip_leaked_thought`); the contract on
    the reply as it came is what a reply with no complete answer meets, and
    these pin what it does with each shape such a reply can take."""

    SOURCE = [SubtitleSegment("1", 0.0, 1.0, "a")]
    ANSWER = "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|a|甲|high|\n</translated>"

    def _validate(self, text: str):
        return validate_correction_output_text(
            text, self.SOURCE, variant=resolve_variant(None, CapabilityTier.CAPABLE), thresholds=STANDARD
        )

    def test_reasoning_alone_is_rejected_and_the_repair_says_why(self) -> None:
        result = self._validate("thought\nThe user wants row 1 corrected. I should keep it.")

        assert not result.ok
        assert result.errors[0] == LEAKED_THOUGHT_ERROR

    def test_reasoning_with_an_incomplete_draft_is_rejected(self) -> None:
        draft = f"thought\nDraft:\n<translated>\n{SIX}\n</translated>\nNow check coverage."

        result = self._validate(draft)

        assert not result.ok
        assert result.errors[0] == LEAKED_THOUGHT_ERROR

    def test_reasoning_that_names_the_tag_swallows_the_answer_and_is_rejected(self) -> None:
        """Pinned as today's behaviour: the mention opens a block that closes
        at the real answer's tag, so its first line is prose, not the header.
        A false rejection, but on the safe side -- the repair round follows."""

        result = self._validate("thought\nI will write <translated> with the header.\n" + self.ANSWER)

        assert not result.ok
        assert result.errors[0] == LEAKED_THOUGHT_ERROR

    def test_reasoning_outside_a_complete_answer_is_accepted(self) -> None:
        """Owner 2026-10-01: prose outside the blocks is not part of the
        product, so a complete answer after leaked reasoning passes."""

        result = self._validate("thought\nRow 1 is fine as is.\n" + self.ANSWER)

        assert result.ok, result.errors

    def test_stripping_the_reasoning_leaves_the_answer_to_validate(self) -> None:
        """The reply the validator alone rejects above, once the callers strip it."""

        reply = "thought\nI will write <translated> with the header.\n" + self.ANSWER
        answer = strip_leaked_thought(reply, CORRECTION_ANSWER_TAGS)

        assert answer == self.ANSWER
        assert self._validate(answer).ok
        assert strip_leaked_thought(self.ANSWER, ("translated",)) is None, "only a reply that opens with `thought`"
        assert strip_leaked_thought("thought\nDraft:\n<translated>\nsub|1|", ("translated",)) is None

    def test_an_ordinary_failure_gets_no_reasoning_hint(self) -> None:
        result = self._validate("I could not finish.")

        assert not result.ok
        assert LEAKED_THOUGHT_ERROR not in result.errors


def test_a_polish_set_heads_its_second_column_polished_and_may_leave_it_empty() -> None:
    """A Chinese source's set: corrected_text is the faithful transcript, `polished` its light polish, empty on a
    line that is only filler; the length checks then read the corrected text."""

    source = [SubtitleSegment("1", 0.0, 1.0, "呃"), SubtitleSegment("2", 1.2, 2.0, "我待了家里")]
    text = (
        "<translated>\ntype|position|corrected_text|polished|conf|note\n"
        "sub|1|呃||high|\nsub|2|我待了家里|我待在家里|high|\n</translated>"
    )

    result = validate_correction_output_text(
        text, source, variant=resolve_variant("capableC-polish"), thresholds=Thresholds.at(1.0)
    )

    assert result.ok, result.errors
    assert [(s.corrected_text, s.translation) for s in result.segments] == [
        ("呃", ""), ("我待了家里", "我待在家里")
    ]


def test_a_translating_set_still_rejects_an_empty_translation() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "えっと")]
    text = "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|えっと||high|\n</translated>"

    result = validate_correction_output_text(
        text, source, variant=resolve_variant("capableC"), thresholds=Thresholds.at(1.0)
    )

    assert "Row 1 has empty translation text." in result.errors


def test_a_polish_set_rejects_the_translation_header() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "然后")]
    text = "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|然后|然后|high|\n</translated>"

    result = validate_correction_output_text(
        text, source, variant=resolve_variant("capableC-polish"), thresholds=Thresholds.at(1.0)
    )

    assert not result.ok


def test_a_polish_set_reports_an_empty_row_once() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "然后")]
    text = "<translated>\ntype|position|corrected_text|polished|conf|note\nsub|1|||high|\n</translated>"

    result = validate_correction_output_text(
        text, source, variant=resolve_variant("capableC-polish"), thresholds=Thresholds.at(1.0)
    )

    assert [e for e in result.errors if e.startswith("Row 1")] == ["Row 1 has empty corrected text."]


@pytest.mark.parametrize("text, back", [
    ("plain", "plain"),
    ("a|b", "a｜b"),
    ("  two\r\nlines\rhere  ", "two\nlines\nhere"),
    ("C:\\path\\to", "C:\\path\\to"),
    ("gap\n\n\nbelow", "gap\nbelow"),
])
def test_a_text_cell_reads_back_as_written(text, back):
    cell = encode_text_cell(text)
    assert "|" not in cell and "\n" not in cell
    assert decode_text_cell(cell) == back


# --- placeholder answers (ported from Nonoka X's engine patch 0009) -----------------------------------------------

def _lines(count: int) -> list[SubtitleSegment]:
    return [SubtitleSegment(str(index), index * 2.0, index * 2.0 + 1.5, f"今日は{index}回目の話") for index in range(1, count + 1)]


def _check(rows: list[str], count: int):
    return validate_correction_output_text(six(*rows), _lines(count), variant=resolve_variant("capableC"), thresholds=STANDARD)


def test_a_window_of_placeholder_rows_goes_back_for_a_rewrite() -> None:
    result = _check([f"sub|{index}|text|text|high|" for index in range(1, 9)], 8)

    assert not result.ok
    assert len(result.errors) == 1 and "text" in result.errors[0]
    # No row carries it: the repair round asks for the whole window again, not a patch.
    assert not any(verdict.errors for verdict in result.rows)


def test_most_rows_a_placeholder_word_is_refused_whatever_the_case() -> None:
    rows = [f"sub|{index}|TODO|待定{index}|high|" for index in range(1, 4)] + ["sub|4|今日は4回目|第四次|high|"]
    assert not _check(rows, 4).ok


def test_one_line_repeated_on_every_row_is_refused_from_five_rows_on() -> None:
    assert not _check([f"sub|{index}|うん|嗯|high|" for index in range(1, 6)], 5).ok
    assert _check([f"sub|{index}|うん|嗯|high|" for index in range(1, 5)], 4).ok


def test_one_line_repeated_passes_where_every_source_says_it() -> None:
    chant = [SubtitleSegment(str(index), index * 2.0, index * 2.0 + 1.5, "せーの！") for index in range(1, 7)]
    rows = [f"sub|{index}|せーの！|预备！|high|" for index in range(1, 7)]
    assert validate_correction_output_text(six(*rows), chant, variant=resolve_variant("capableC"), thresholds=STANDARD).ok


def test_a_lone_placeholder_word_among_real_rows_passes() -> None:
    rows = ["sub|1|テスト|test|high|", "sub|2|今日は2回目|第二次|high|", "sub|3|今日は3回目|第三次|high|"]
    assert _check(rows, 3).ok
