from __future__ import annotations

from llm.chunking import SubtitleSegment, render_segments_as_csv
from llm.config import CapabilityTier
from llm.csv_utils import (
    OUTPUT_CSV_HEADER,
    OUTPUT_CSV_HEADER_WITH_START,
    TranslatedCsvSegment,
    looks_truncated_translated,
    merge_translated_csv_windows,
    render_corrected_segments_as_srt,
    render_translated_segments_as_csv,
    render_translated_segments_as_srt,
    validate_correction_output_text,
    validate_translated_csv_text,
    validate_translated_jsonl_text,
)
from llm.prompt_variants import resolve_variant
from llm.srt_utils import parse_srt


def test_render_segments_as_csv_uses_local_tenths_and_escaped_text() -> None:
    segments = [
        SubtitleSegment("3", 10.11, 10.61, "你\n好|呀"),
        SubtitleSegment("4", 11.01, 12.01, "好"),
    ]

    csv_text = render_segments_as_csv(segments, window_start=9.01)

    assert csv_text.splitlines() == [
        r"3|1.1|0.5|0.4|你\n好｜呀",
        "4|2.0|1.0|0.0|好",
    ]
    assert segments[0].start == 10.11
    assert segments[0].end == 10.61


def test_validate_can_require_v55_headers_without_breaking_legacy_audits() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    without_headers = (
        "<singles>\nsub|1|1.0|0.0|a|甲|high|1|宜独立\n</singles>\n"
        "<translated>\nsub|1|1.0|0.0|a|甲|high|1|\n</translated>"
    )
    assert validate_translated_csv_text(without_headers, source).ok
    strict_missing = validate_translated_csv_text(
        without_headers, source, require_headers=True
    )
    assert not strict_missing.ok
    assert sum("exact CSV header" in error for error in strict_missing.errors) == 2

    with_headers = (
        f"<singles>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1|1.0|0.0|a|甲|high|1|宜独立\n</singles>\n"
        f"<translated>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1|1.0|0.0|a|甲|high|1|\n</translated>"
    )
    strict_ok = validate_translated_csv_text(
        with_headers, source, require_headers=True
    )
    assert strict_ok.ok, strict_ok.errors


def test_variant_aware_validator_projects_the_served_output_contract() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    no_singles = (
        f"<translated>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1|1.0|0.0|a|甲|high|1|\n</translated>"
    )

    capable = validate_correction_output_text(
        no_singles,
        source,
        variant=resolve_variant(None, CapabilityTier.CAPABLE),
    )
    # Production basic default is basicB (no full-window singles; start column).
    basic = validate_correction_output_text(
        no_singles.replace(OUTPUT_CSV_HEADER, OUTPUT_CSV_HEADER_WITH_START).replace(
            "sub|1|1.0|0.0|", "sub|1|0.0|1.0|0.0|"
        ),
        source,
        variant=resolve_variant(None, CapabilityTier.BASIC),
    )
    basic_a = validate_correction_output_text(
        no_singles.replace(OUTPUT_CSV_HEADER, OUTPUT_CSV_HEADER_WITH_START),
        source,
        variant=resolve_variant("basicA"),
    )

    assert capable.ok, capable.errors
    assert not any("<singles>" in error for error in capable.errors)
    assert basic.ok, basic.errors
    assert not basic_a.ok
    assert any("<singles>" in error for error in basic_a.errors)


def test_variant_aware_validator_rejects_unexpected_start_column_only() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    variant = resolve_variant(None, CapabilityTier.CAPABLE)
    malformed = (
        f"<translated>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1|0.0|1.0|0.0|a|甲|high|1|note\n</translated>"
    )
    valid_with_pipe_note = (
        f"<translated>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1|1.0|0.0|a|甲|high|1|note|with pipe\n</translated>"
    )

    rejected = validate_correction_output_text(malformed, source, variant=variant)
    accepted = validate_correction_output_text(
        valid_with_pipe_note, source, variant=variant
    )

    assert not rejected.ok
    assert any("unexpected start column" in error for error in rejected.errors)
    assert accepted.ok, accepted.errors


def test_validate_basic_start_column_contract() -> None:
    source = [SubtitleSegment("1", 2.5, 3.5, "a")]
    content = (
        f"<singles>\n{OUTPUT_CSV_HEADER_WITH_START}\n"
        "sub|1|2.5|1.0|0.0|a|甲|high|1|宜独立\n</singles>\n"
        f"<translated>\n{OUTPUT_CSV_HEADER_WITH_START}\n"
        "sub|1|2.5|1.0|0.0|a|甲|high|1|\n</translated>"
    )
    result = validate_translated_csv_text(
        content,
        source,
        require_headers=True,
        require_start_column=True,
    )
    assert result.ok, result.errors

    missing_start = content.replace("|2.5|1.0|", "|1.0|")
    rejected = validate_translated_csv_text(
        missing_start,
        source,
        require_headers=True,
        require_start_column=True,
    )
    assert not rejected.ok


def test_validate_basicc_jsonl_contract_and_reasoning() -> None:
    source = [
        SubtitleSegment("1", 12.3, 13.0, "a"),
        SubtitleSegment("2", 13.1, 14.0, "b"),
    ]
    text = (
        "<translated>\n"
        '{"type":"sub","position":"1","start":12.3,"duration":0.7,'
        '"gap":0.1,"corrected_text":"a|x","translation":"甲",'
        '"conf":"high","char_count":1,"note":""}\n'
        '{"type":"reasoning","reasoning":"与前后脱节"}\n'
        '{"type":"discard","position":"2","note":"幻觉"}\n'
        "</translated>"
    )
    result = validate_translated_jsonl_text(text, source)
    assert result.ok, result.errors
    assert result.segments[0].corrected_text == "a|x"
    assert result.discarded_ids == ("2",)
    assert result.reasoning_rows == 1


def test_validate_basicc_jsonl_rejects_inline_reasoning_field() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    text = (
        "<translated>\n"
        '{"type":"sub","position":"1","start":0.0,"duration":1.0,'
        '"gap":0.0,"corrected_text":"a","translation":"甲",'
        '"conf":"high","char_count":1,"note":"","reasoning":"inline"}\n'
        "</translated>"
    )
    result = validate_translated_jsonl_text(text, source)
    assert not result.ok
    assert any("unexpected keys: reasoning" in error for error in result.errors)


def test_validate_basicc_jsonl_reports_syntax_and_schema_errors() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "a")]
    text = (
        "<translated>\n"
        '{"type":"sub","position":"1","start":0.0}\n'
        "not-json\n"
        "</translated>"
    )
    result = validate_translated_jsonl_text(text, source)
    assert not result.ok
    assert any("missing required keys" in error for error in result.errors)
    assert any("invalid JSON" in error for error in result.errors)


def test_translated_csv_merges_sources_and_restores_srt_newlines() -> None:
    source = [
        SubtitleSegment("3", 1.001, 1.501, "你"),
        SubtitleSegment("4", 2.001, 3.001, "好"),
        SubtitleSegment("5", 4.001, 9.101, "好好好好好好好好好好"),
        SubtitleSegment("6", 70.123, 71.023, "你好"),
    ]
    output = (
        "<translated>\n"
        "3,4|good morning|你好\n"
        "discard|5|重复幻觉\n"
        r"6|source line|第一行\n第二行"
        "\n</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)
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
        r"1|line a\n\nline b|译一\n\n\n译二"
        "\n"
        r"2|单行|单译"
        "\n</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)

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
        "<translated>\n9|nine|九\n</translated>",
        "<translated>\n1|one|一\n1|repeat|重复\n</translated>",
        "<translated>\n1\n</translated>",
        "<translated>\n1|one\n</translated>",
        "<translated>\n1||一\n</translated>",
        "<translated>\n1|one|\n</translated>",
    ]

    for content in cases:
        assert not validate_translated_csv_text(content, source, require_singles=False).ok


def test_empty_translated_block_can_drop_entire_window() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text("<translated></translated>", source, require_singles=False)

    assert result.ok
    assert result.segments == []
    assert render_translated_segments_as_srt(result.segments) == ""


def test_translated_window_merge_removes_current_window_source_ids() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
        SubtitleSegment("3", 2.0, 3.0, "三"),
    ]
    first = validate_translated_csv_text(
        "<translated>\n1|one|一\n2|two|二\n</translated>",
        source[:2], require_singles=False).segments
    second = validate_translated_csv_text(
        "<translated>\n3|three|三\n</translated>",
        source[1:], require_singles=False).segments

    merged = merge_translated_csv_windows(first, ["2", "3"], second)

    assert [segment.source_ids for segment in merged] == [("1",), ("3",)]


def _translated(source_ids: tuple[str, ...], start: float, end: float) -> object:
    from llm.csv_utils import TranslatedCsvSegment

    return TranslatedCsvSegment(
        source_ids=source_ids,
        start=start,
        end=end,
        corrected_text="src " + ",".join(source_ids),
        translation="译 " + ",".join(source_ids),
    )


def test_merge_keeps_straddling_old_row_and_drops_conflicting_new_row() -> None:
    # Previous window merged [79,80,81]; 81 falls into the new window's overlap.
    old = [_translated(("79", "80", "81"), 79.0, 82.0)]
    new = [
        _translated(("81", "82"), 81.0, 83.0),
        _translated(("83",), 83.0, 84.0),
    ]

    merged = merge_translated_csv_windows(old, ["81", "82", "83"], new)

    assert [segment.source_ids for segment in merged] == [("79", "80", "81"), ("83",)]


def test_merge_backfills_ids_lost_to_a_conflict_from_old_rows() -> None:
    # New row [81,82] is dropped because 81 is claimed; 82 must be backfilled
    # from the displaced old row that covered it.
    old = [
        _translated(("79", "80", "81"), 79.0, 82.0),
        _translated(("82",), 82.0, 83.0),
    ]
    new = [
        _translated(("81", "82"), 81.0, 83.0),
        _translated(("83",), 83.0, 84.0),
    ]

    merged = merge_translated_csv_windows(old, ["81", "82", "83"], new)

    assert [segment.source_ids for segment in merged] == [
        ("79", "80", "81"),
        ("82",),
        ("83",),
    ]


def test_merge_does_not_resurrect_ids_the_new_window_dropped() -> None:
    # The new window intentionally omitted 82 (no new row covers it); the old
    # row for 82 stays displaced because nothing was lost to a conflict.
    old = [
        _translated(("81",), 81.0, 82.0),
        _translated(("82",), 82.0, 83.0),
    ]
    new = [_translated(("81",), 81.0, 82.0)]

    merged = merge_translated_csv_windows(old, ["81", "82"], new)

    assert [segment.source_ids for segment in merged] == [("81",)]
    assert merged[0].translation == "译 81"


def test_merge_plain_overlap_still_prefers_newest_window() -> None:
    old = [
        _translated(("1",), 0.0, 1.0),
        _translated(("2",), 1.0, 2.0),
    ]
    new = [_translated(("2", "3"), 1.0, 3.0)]

    merged = merge_translated_csv_windows(old, ["2", "3"], new)

    assert [segment.source_ids for segment in merged] == [("1",), ("2", "3")]


def test_translated_csv_skips_plan_lines() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "plan|单源；字少gap正常，不合并。\n"
        "sub|1|1.0|one|一|8|译1字\n"
        "plan|两源口播碎片gap小→合并；ASR「新書」应为「新衣装」。\n"
        "sub|2|1.0|two|二|7|译1字；短句\n"
        "PLAN|大小写也应跳过\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)

    assert result.ok
    assert len(result.segments) == 2
    assert [seg.source_ids for seg in result.segments] == [("1",), ("2",)]
    assert not any("unknown type" in w for w in result.warnings)


def test_translated_csv_parses_conf_char_count_and_note_columns() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "sub|1|1.0|0.2|one|一|high|1|术语note\n"
        "|2|1.0|0.0|two|二|median|1|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)

    assert result.ok
    first, second = result.segments
    assert first.kind == "sub" and first.conf == "high"
    assert first.char_count == "1" and first.note == "术语note"
    assert second.kind == "sub" and second.conf == "median"
    assert second.char_count == "1" and second.note == ""


def test_translated_csv_recomputes_and_normalizes_char_count() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]
    output = (
        "<translated>\n"
        "sub|1|1.0|0.0|one|A中1|high|99|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)

    assert result.ok
    assert result.segments[0].char_count == "2"
    assert any("char_count '99'" in warning for warning in result.warnings)


def test_singles_char_count_drift_is_reported() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]
    output = (
        "<singles>\n"
        "sub|1|1.0|0.0|one|一|high|99|宜独立\n"
        "</singles>\n"
        "<translated>\n"
        "sub|1|1.0|0.0|one|一|high|1|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source)

    assert result.ok
    assert any("<singles> row 1 char_count '99'" in w for w in result.warnings)


def test_translated_csv_conf_out_of_range_degrades_without_failing() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|one|一|42|\n</translated>", source, require_singles=False)

    assert result.ok
    assert result.segments[0].conf is None
    assert any("invalid conf" in warning for warning in result.warnings)


def test_translated_csv_maps_legacy_numeric_confidence_to_tiers() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]
    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|one|一|8|\n</translated>",
        source,
        require_singles=False,
    )
    assert result.ok
    assert result.segments[0].conf == "high"


def test_translated_csv_note_may_contain_pipes_as_last_column() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|0.0|one|一|median|1|a|b|c\n</translated>", source, require_singles=False)

    assert result.ok
    assert result.segments[0].note == "a|b|c"


def test_translated_csv_insert_row_uses_clip_relative_timing() -> None:
    source = [
        SubtitleSegment("20", 105.0, 106.0, "I think"),
        SubtitleSegment("22", 109.5, 110.7, "that's right"),
    ]
    output = (
        "<translated>\n"
        "sub|20|1.0|2.0|I think|我觉得|high|3|\n"
        "insert|7.0,0.8|0.8|1.7|まって|等一下|median|3|漏识别\n"
        "sub|22|1.2|0.0|that's right|没错|high|2|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, clip_start=100.0, require_singles=False)
    srt = parse_srt(render_translated_segments_as_srt(result.segments))

    assert result.ok
    insert = [seg for seg in result.segments if seg.is_insert][0]
    assert insert.source_ids == ()
    assert insert.start == 107.0 and round(insert.end, 1) == 107.8
    assert insert.note == "漏识别"
    # Insert is time-sorted between its neighbors in the rendered SRT.
    assert [seg.text for seg in srt] == ["我觉得", "等一下", "没错"]


def test_translated_csv_rejects_bad_insert_timing() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    for bad in ("insert|abc|x|译|5|", "insert|1.0|x|译|5|", "insert|1.0,0|x|译|5|"):
        result = validate_translated_csv_text(f"<translated>\n{bad}\n</translated>", source, require_singles=False)
        assert not result.ok


def test_render_translated_segments_as_csv_round_trips_nine_columns() -> None:
    segments = [
        TranslatedCsvSegment(("3", "4"), 1.0, 3.0, "good morning", "你好", conf="high", char_count="99", note="n|1"),
        TranslatedCsvSegment((), 7.0, 7.8, "まって", "等一下", kind="insert", conf="median", char_count="99"),
    ]

    text = render_translated_segments_as_csv(segments)

    assert text.splitlines() == [
        "sub|3,4|2.0|4.0|good morning|你好|high|2|n｜1",
        "insert|7.0,0.8|0.8|0.0|まって|等一下|median|3|",
    ]


def test_translated_csv_accepts_gap_column_after_duration() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "one")]
    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|0.3|one|一|high|1|\n</translated>",
        source,
        require_singles=False,
    )
    assert result.ok
    assert result.segments[0].translation == "一"


def test_translated_csv_rejects_non_numeric_gap_without_shifting_columns() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "one")]
    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|later|one|一|high|1|\n</translated>",
        source,
        require_singles=False,
    )
    assert not result.ok
    assert any("gap" in error for error in result.errors)


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
    row = "sub|{pos}|2.0|0.0|x|译文|high|2|\n"
    two = validate_translated_csv_text(
        "<translated>\n" + row.format(pos="1,2")
        + row.format(pos="3") + row.format(pos="4") + "</translated>",
        source,
        require_singles=False,
    )
    assert two.ok
    assert not any("soft cap" in w for w in two.warnings)

    three = validate_translated_csv_text(
        "<translated>\n" + row.format(pos="1,2,3") + row.format(pos="4")
        + "</translated>",
        source,
        require_singles=False,
    )
    assert three.ok
    assert any("soft cap" in w for w in three.warnings)

    four = validate_translated_csv_text(
        "<translated>\n" + row.format(pos="1,2,3,4") + "</translated>",
        source,
        require_singles=False,
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
        "sub|65,66|7.2|0.0|合并稿|合并译文|<void>|0|\n"
        "sub|65|2.5|1.2|パジャマで部屋に集まり直して|重新集合|high|5|\n"
        "sub|66|2.2|0.0|ケーキを4人で|四人分蛋糕|high|6|\n"
        "</translated>"
    )
    result = validate_translated_csv_text(text, source, require_singles=False)
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
        "<translated>\nsub|1,3|3.0|0.0|一三|一三|high|2|\n</translated>",
        source,
        require_singles=False,
    )
    assert not skip.ok
    assert any("adjacent" in error for error in skip.errors)


def test_translated_csv_rejects_rows_missing_the_duration_column() -> None:
    # v11: the duration column is part of the contract — a 6-column row means
    # the model skipped the span self-check and the row must fail (retry).
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text(
        "<translated>\nsub|1|one|一|8|note\n</translated>", source, require_singles=False)

    assert not result.ok
    assert any("duration" in error for error in result.errors)


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
        "sub|1,2,3|64.8|runaway merge|失控合并|7|<void>\n"
        "sub|1,2|2.0|one two|一二|8|\n"
        "sub|3|1.0|three|三|8|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)

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
        "sub|1|1.0|one|一|8|中途放弃<VOID>\n"
        "sub|1|1.0|one|一|8|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, require_singles=False)

    assert result.ok
    assert result.voided_rows == 2
    assert [seg.source_ids for seg in result.segments] == [("1",)]


def test_all_rows_voided_counts_as_no_valid_rows() -> None:
    # Voiding everything without re-emitting is an incomplete output -> retry.
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|one|一|8|<void>\n</translated>", source, require_singles=False)

    assert not result.ok
    assert result.voided_rows == 1
    assert any("no valid rows" in error for error in result.errors)


def test_merge_dedups_overlapping_inserts_newest_wins() -> None:
    def _insert(start: float, translation: str) -> TranslatedCsvSegment:
        return TranslatedCsvSegment(
            (), start, start + 0.5, translation, translation, kind="insert"
        )

    old = [_translated(("1",), 0.0, 1.0), _insert(5.0, "旧")]
    # New window re-inserts near 5.0 (within tolerance) plus a far insert.
    new = [_translated(("1",), 0.0, 1.0), _insert(5.2, "新"), _insert(30.0, "远")]

    merged = merge_translated_csv_windows(old, ["1"], new)

    inserts = [seg for seg in merged if seg.is_insert]
    assert [seg.translation for seg in inserts] == ["新", "远"]
    # The normal row is untouched by inserts.
    assert [seg.source_ids for seg in merged if not seg.is_insert] == [("1",)]


def test_unclosed_translated_block_looks_truncated() -> None:
    assert looks_truncated_translated("<translated>\n1|one|一\n")
    assert not looks_truncated_translated("<translated></translated>")


def test_pacing_scorer_step_penalties_and_pass_ratio() -> None:
    from llm.csv_utils import score_translated_segments

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
    from llm.csv_utils import score_translated_segments

    # 40 Latin letters = 20 weighted -> no excess; 42 = 21 -> 0.1 penalty.
    at_limit = TranslatedCsvSegment(("1",), 0.0, 3.0, "src", "a" * 40)
    over_limit = TranslatedCsvSegment(("1",), 0.0, 3.0, "src", "a" * 42)
    assert score_translated_segments([at_limit])["rows"][0]["penalty"] == 0.0
    assert score_translated_segments([over_limit])["rows"][0]["penalty"] == 0.1


def test_validate_requires_singles_one_to_one_and_top_level_translated() -> None:
    source = [
        SubtitleSegment("1", 0.0, 0.8, "a"),
        SubtitleSegment("2", 0.9, 1.5, "b"),
    ]
    # Reasoning name-drops must not break extraction / validation.
    good = (
        "<reasoning>\n写完 `<singles>` 再写 `<translated>`\n</reasoning>\n"
        "<singles>\n"
        "sub|1|0.8|a|甲|8|译1字；宜保持独立\n"
        "sub|2|0.6|b|乙|8|译1字；宜与前一句合并\n"
        "</singles>\n"
        "<translated>\n"
        "sub|1,2|1.5|a b|甲乙|8|译2字\n"
        "</translated>"
    )
    ok = validate_translated_csv_text(good, source)
    assert ok.ok, ok.errors
    assert ok.segments[0].source_ids == ("1", "2")

    missing = (
        "<singles>\nsub|1|0.8|a|甲|8|译1字；宜保持独立\n</singles>\n"
        "<translated>\nsub|1|0.8|a|甲|8|译1字\n</translated>"
    )
    bad_cover = validate_translated_csv_text(missing, source)
    assert not bad_cover.ok
    assert any("missing source id" in e for e in bad_cover.errors)

    merged_single = (
        "<singles>\n"
        "sub|1,2|1.5|a b|甲乙|8|译2字；宜合并\n"
        "</singles>\n"
        "<translated>\nsub|1,2|1.5|a b|甲乙|8|译2字\n</translated>"
    )
    bad_merge = validate_translated_csv_text(merged_single, source)
    assert not bad_merge.ok
    assert any("single source id" in e for e in bad_merge.errors)

    truncated = (
        "<singles>\n"
        "sub|1|0.8|a|甲|8|译1字；宜保持独立\n"
        "sub|2|0.6|b|乙|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\nsub|1|0.8|a|甲|8|译1字\n</translated>"
    )
    # v54: silent omission no longer allowed — missing source 2 is an error
    result = validate_translated_csv_text(truncated, source)
    assert not result.ok
    assert any("missing source id" in e for e in result.errors)

    # Explicit discard makes it pass
    with_discard = (
        "<singles>\n"
        "sub|1|0.8|a|甲|8|译1字；宜保持独立\n"
        "sub|2|0.6|b|乙|8|译1字；宜丢弃\n"
        "</singles>\n"
        "<translated>\nsub|1|0.8|a|甲|8|译1字\ndiscard|2|幻觉\n</translated>"
    )
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
        "sub|1|1.0|0.0|a|甲|high|1|\n"
        "</translated>"
    )
    duplicate = (
        "<translated>\n"
        "sub|1|1.0|0.0|a|甲|high|1|\n"
        "discard|2|幻觉\n"
        "discard|2|重复决定\n"
        "</translated>"
    )

    reversed_result = validate_translated_csv_text(
        out_of_order, source, require_singles=False
    )
    duplicate_result = validate_translated_csv_text(
        duplicate, source, require_singles=False
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
        "<singles>\nsub|1|1.0|a|甲|8|译1字；宜保持独立\n</singles>\n"
        "<translated>\nsub|1|1.0|a|甲|8|译1字\n</translated>"
    )
    result = validate_translated_csv_text(text, source)
    assert result.ok, result.errors


# ---------------------------------------------------------------------------
# capableC: inter-line reasoning rows (skipped for SRT, counted, anchor-checked)
# ---------------------------------------------------------------------------


def test_reasoning_rows_are_skipped_counted_and_do_not_cover() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    # A reasoning comment above the merged sub row; both sources covered.
    output = (
        "<translated>\n"
        "# gap=0.0 同一句切开，合并后 2 字在界内\n"
        "sub|1,2|2.0|0.0|one two|一二|median|2|同一句合并\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source, require_singles=False)
    assert result.ok, result.errors
    assert result.reasoning_rows == 1
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
        "sub|1|1.0|0.0|one|一|high|1|\n"
        "# 说明为何丢弃源2\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source, require_singles=False)
    assert not result.ok
    assert any("missing source id" in e for e in result.errors)


def test_reasoning_comment_before_discard() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
    ]
    output = (
        "<translated>\n"
        "sub|1|1.0|0.0|one|一|high|1|\n"
        "# 复读幻觉，三特征叠加\n"
        "discard|2|复读幻觉\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, source, require_singles=False)
    assert result.ok, result.errors
    assert result.discarded_ids == ("2",)
    assert result.reasoning_rows == 1
