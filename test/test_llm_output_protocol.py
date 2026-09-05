from __future__ import annotations

from finesub.llm.chunking import (
    SubtitleSegment,
    SubtitleWindow,
    WindowIdMap,
    render_segments_as_csv,
)
from finesub.llm.routing.config import CapabilityTier
from finesub.llm.output_protocol import (
    OUTPUT_CSV_HEADER,
    OUTPUT_CSV_HEADER_WITH_START,
    TranslatedCsvSegment,
    looks_truncated_translated,
    merge_translated_csv_windows,
    render_corrected_segments_as_srt,
    render_translated_segments_as_csv,
    render_translated_segments_as_srt,
    validate_correction_output_text,
    validate_correction_window_output,
    validate_translated_csv_text,
)
from finesub.llm.prompt_variants import resolve_variant
from finesub.subtitles.metrics import format_weighted_char_count, weighted_char_count
from finesub.subtitles.model import parse_srt


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
    )
    text = (
        f"<translated>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1,2|1.7|0.0|ab|甲乙|high|2|\n</translated>"
    )

    result = validate_correction_window_output(
        text,
        window,
        variant=resolve_variant("capableB"),
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
    )
    for invalid_id in ("0", "-1"):
        text = (
            f"<translated>\n{OUTPUT_CSV_HEADER}\n"
            f"sub|{invalid_id}|0.7|0.0|a|甲|high|1|\n</translated>"
        )
        result = validate_correction_window_output(
            text,
            window,
            variant=resolve_variant("capableB"),
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
        "sub|3,4|2.0|0.5|good morning|你好|high|2|\n"
        "discard|5|重复幻觉\n"
        r"sub|6|0.9|0.0|source line|第一行\n第二行|high|6|"
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
        r"sub|1|1.0|0.5|line a\n\nline b|译一\n\n\n译二|high|4|"
        "\n"
        r"sub|2|1.0|0.0|单行|单译|high|2|"
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
        "<translated>\nsub|9|1.0|0.0|nine|九|high|1|\n</translated>",
        "<translated>\nsub|1|1.0|0.0|one|一|high|1|\nsub|1|1.0|0.0|repeat|重复|high|2|\n</translated>",
        "<translated>\n1\n</translated>",
        "<translated>\n1|one\n</translated>",
        "<translated>\nsub|1|1.0|0.0||一|high|1|\n</translated>",
        "<translated>\n1|one|\n</translated>",
    ]

    for content in cases:
        assert not validate_translated_csv_text(content, source, require_singles=False).ok


def test_empty_translated_block_no_longer_drops_the_window() -> None:
    """Inverted deliberately: emptiness is a truncated reply, not a wipe.

    This test used to assert the opposite. Treating an empty block as an
    intentional "drop the whole window" made a cut-off reply indistinguishable
    from a deliberate one, and the commit path then deleted rows earlier
    windows had already produced for the overlap ids.
    """
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    result = validate_translated_csv_text("<translated></translated>", source, require_singles=False)

    assert not result.ok
    assert result.segments == []
    assert render_translated_segments_as_srt(result.segments) == ""


def test_translated_window_merge_removes_current_window_source_ids() -> None:
    source = [
        SubtitleSegment("1", 0.0, 1.0, "一"),
        SubtitleSegment("2", 1.0, 2.0, "二"),
        SubtitleSegment("3", 2.0, 3.0, "三"),
    ]
    first = validate_translated_csv_text(
        "<translated>\n"
        "sub|1|1.0|0.0|one|一|high|1|\n"
        "sub|2|1.0|0.0|two|二|high|1|\n</translated>",
        source[:2], require_singles=False).segments
    second = validate_translated_csv_text(
        "<translated>\n"
        "sub|3|1.0|0.0|three|三|high|1|\n"
        "discard|2|窗口重叠，由本窗重写\n</translated>",
        source[1:], require_singles=False).segments

    merged = merge_translated_csv_windows(first, ["2", "3"], second)

    assert [segment.source_ids for segment in merged] == [("1",), ("3",)]


def _translated(source_ids: tuple[str, ...], start: float, end: float) -> object:
    from finesub.llm.output_protocol import TranslatedCsvSegment

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
        "sub|1|1.0|0.0|one|一|8|1|译1字\n"
        "plan|两源口播碎片gap小→合并；ASR「新書」应为「新衣装」。\n"
        "sub|2|1.0|0.0|two|二|7|1|译1字；短句\n"
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
        "<translated>\nsub|1|1.0|0.0|one|一|42|1|\n</translated>", source, require_singles=False)

    assert result.ok
    assert result.segments[0].conf is None
    assert any("invalid conf" in warning for warning in result.warnings)


def test_translated_csv_maps_legacy_numeric_confidence_to_tiers() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]
    result = validate_translated_csv_text(
        "<translated>\nsub|1|1.0|0.0|one|一|8|1|\n</translated>",
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
        "sub|20|1.0|2.0|I think|我觉得|high|3|\n"
        "insert|7.0,0.8|0.8|1.7|まって|等一下|median|3|漏识别\n"
        "sub|22|1.2|0.0|that's right|没错|high|2|\n"
        "</translated>"
    )

    result = validate_translated_csv_text(output, source, clip_start=100.0, require_singles=False)

    assert not result.ok
    assert any("no longer part of the output contract" in e for e in result.errors)
    assert all(segment.kind == "sub" for segment in result.segments)


def test_translated_csv_rejects_bad_insert_timing() -> None:
    source = [SubtitleSegment("1", 0.0, 1.0, "一")]

    for bad in ("insert|abc|x|译|5|", "insert|1.0|x|译|5|", "insert|1.0,0|x|译|5|"):
        result = validate_translated_csv_text(f"<translated>\n{bad}\n</translated>", source, require_singles=False)
        assert not result.ok


def test_render_translated_segments_as_csv_round_trips_nine_columns() -> None:
    segments = [
        TranslatedCsvSegment(("3", "4"), 1.0, 3.0, "good morning", "你好", conf="high", char_count="99", note="n|1"),
        TranslatedCsvSegment(("5",), 7.0, 7.8, "wait", "等一下", conf="median", char_count="99"),
    ]

    text = render_translated_segments_as_csv(segments)

    assert text.splitlines() == [
        "sub|3,4|2.0|4.0|good morning|你好|high|2|n｜1",
        "sub|5|0.8|0.0|wait|等一下|median|3|",
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
        "sub|1,2,3|64.8|0.0|runaway merge|失控合并|7|4|<void>\n"
        "sub|1,2|2.0|0.0|one two|一二|8|2|\n"
        "sub|3|1.0|0.0|three|三|8|1|\n"
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
        "sub|1|1.0|0.0|one|一|8|1|中途放弃<VOID>\n"
        "sub|1|1.0|0.0|one|一|8|1|\n"
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
        "<translated>\nsub|1|1.0|0.0|one|一|8|1|<void>\n</translated>", source, require_singles=False)

    assert not result.ok
    assert result.voided_rows == 1
    assert any("no valid rows" in error for error in result.errors)


def test_unclosed_translated_block_looks_truncated() -> None:
    assert looks_truncated_translated("<translated>\nsub|1|1.0|0.0|one|一|high|1|\n")
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


def test_validate_requires_singles_one_to_one_and_top_level_translated() -> None:
    source = [
        SubtitleSegment("1", 0.0, 0.8, "a"),
        SubtitleSegment("2", 0.9, 1.5, "b"),
    ]
    # Reasoning name-drops must not break extraction / validation.
    good = (
        "<reasoning>\n写完 `<singles>` 再写 `<translated>`\n</reasoning>\n"
        "<singles>\n"
        "sub|1|0.8|0.0|a|甲|8|1|译1字；宜保持独立\n"
        "sub|2|0.6|0.0|b|乙|8|1|译1字；宜与前一句合并\n"
        "</singles>\n"
        "<translated>\n"
        "sub|1,2|1.5|0.0|a b|甲乙|8|2|译2字\n"
        "</translated>"
    )
    ok = validate_translated_csv_text(good, source)
    assert ok.ok, ok.errors
    assert ok.segments[0].source_ids == ("1", "2")

    missing = (
        "<singles>\nsub|1|0.8|0.0|a|甲|8|1|译1字；宜保持独立\n</singles>\n"
        "<translated>\nsub|1|0.8|0.0|a|甲|8|1|译1字\n</translated>"
    )
    bad_cover = validate_translated_csv_text(missing, source)
    assert not bad_cover.ok
    assert any("missing source id" in e for e in bad_cover.errors)

    merged_single = (
        "<singles>\n"
        "sub|1,2|1.5|0.0|a b|甲乙|8|2|译2字；宜合并\n"
        "</singles>\n"
        "<translated>\nsub|1,2|1.5|0.0|a b|甲乙|8|2|译2字\n</translated>"
    )
    bad_merge = validate_translated_csv_text(merged_single, source)
    assert not bad_merge.ok
    assert any("single source id" in e for e in bad_merge.errors)

    truncated = (
        "<singles>\n"
        "sub|1|0.8|0.0|a|甲|8|1|译1字；宜保持独立\n"
        "sub|2|0.6|0.0|b|乙|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\nsub|1|0.8|0.0|a|甲|8|1|译1字\n</translated>"
    )
    # v54: silent omission no longer allowed — missing source 2 is an error
    result = validate_translated_csv_text(truncated, source)
    assert not result.ok
    assert any("missing source id" in e for e in result.errors)

    # Explicit discard makes it pass
    with_discard = (
        "<singles>\n"
        "sub|1|0.8|0.0|a|甲|8|1|译1字；宜保持独立\n"
        "sub|2|0.6|0.0|b|乙|8|1|译1字；宜丢弃\n"
        "</singles>\n"
        "<translated>\nsub|1|0.8|0.0|a|甲|8|1|译1字\ndiscard|2|幻觉\n</translated>"
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
        "<singles>\nsub|1|1.0|0.0|a|甲|8|1|译1字；宜保持独立\n</singles>\n"
        "<translated>\nsub|1|1.0|0.0|a|甲|8|1|译1字\n</translated>"
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


# --- Empty <translated> is a degenerate reply, not an intentional wipe -------
#
# `discard|<id>` is the explicit channel for dropping sources, and v52 coverage
# already requires every id to be covered or discarded. An empty block is a
# redundant implicit channel that a truncated reply lands in by accident, so it
# must fail validation and let the retry correct it.


def _header_only_output() -> str:
    return (
        f"<singles>\n{OUTPUT_CSV_HEADER}\n"
        "sub|1|1.0|0.0|one|一|high|1|\n</singles>\n"
        f"<translated>\n{OUTPUT_CSV_HEADER}\n</translated>"
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
    result = validate_translated_csv_text(output, source, require_singles=False)
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
    result = validate_translated_csv_text(output, source, require_singles=False)
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
        f"sub|{i}|1.0|0.0|line {i}|第{i}行|high|3|" for i in range(1, kept + 1)
    ]
    rows += [
        f"discard|{i}|复读幻觉"
        for i in range(kept + 1, kept + discarded + 1)
    ]
    return "<translated>\n" + "\n".join(rows) + "\n</translated>", source


def test_a_window_that_discards_most_of_itself_is_rejected() -> None:
    """The 2026-08-22 canary: one `sub` row plus `discard` for everything else.

    Every structural check passed -- ids covered, order kept, one valid row --
    and the finished subtitle kept a single line. An agent that cannot see the
    window text produces exactly this shape, so the rejection has to happen
    here, at the same seam `agent-task lint` uses, and not by eye downstream.
    """
    output, source = _mostly_discarded_output(kept=1, discarded=19)
    result = validate_translated_csv_text(output, source, require_singles=False)
    assert not result.ok
    assert any("over the 50% limit" in e for e in result.errors), result.errors
    # Not the coverage error: every id *was* accounted for.
    assert not any("missing source id" in e for e in result.errors), result.errors


def test_a_heavily_but_not_mostly_discarded_window_still_passes() -> None:
    """The threshold is a wrongness detector, not a quality knob.

    Real production discards up to 21.9% of a window (the singing/English-PV
    material, where dropping most of a song is correct); p95 is 9.6%. A window
    at twice that maximum must still pass, or the gate starts eating the very
    material it was measured against.
    """
    output, source = _mostly_discarded_output(kept=11, discarded=9)
    result = validate_translated_csv_text(output, source, require_singles=False)
    assert result.ok, result.errors
    assert len(result.discarded_ids) == 9


def _mostly_discarded_window_output(
    kept: int, discarded: int
) -> tuple[str, list[SubtitleSegment]]:
    """`_mostly_discarded_output` in the served window contract: header
    required, no start column (capableB)."""

    body, sources = _mostly_discarded_output(kept, discarded)
    rows = [
        line
        for line in body.splitlines()
        if line and not line.startswith("<")
    ]
    text = "\n".join(["<translated>", OUTPUT_CSV_HEADER, *rows, "</translated>"])
    return text, sources


def _window(chunk_id: str, sources: list[SubtitleSegment]) -> SubtitleWindow:
    return SubtitleWindow(
        chunk_id=chunk_id,
        segments=sources,
        overlap_segments=[],
        boundary_reason="test",
        budget=None,  # type: ignore[arg-type]  -- unused by validation
    )


def test_a_split_leaf_may_discard_most_of_itself() -> None:
    """The ratio was measured on whole windows, so it only applies to those.

    `tools/discard_ratio_scan.py` over the archive (63 whole windows / 49
    runs), replaying each through the production `split_window_in_half`: the
    worst whole window discards 21.9%, but the worst half reaches **43.8%** --
    a stretch of song inside a window that averages far less. That leaf is
    *correct* output. Gating it would fail validation, burn the retries and
    stop the task on an answer that was right, and 0.5 is only 1.14x above it
    besides. Full record: `bench-baselines.md` 二十五.

    This case is synthetic (5 kept / 15 discarded) because the archive holds
    only four real leaves and all four discard nothing -- rarity, not safety.
    What it pins is the *rule*, not the distribution.
    """

    output, sources = _mostly_discarded_window_output(kept=5, discarded=15)
    leaf = _window("0007-a", sources)
    result = validate_correction_window_output(
        output, leaf, variant=resolve_variant("capableB")
    )
    assert result.ok, result.errors
    assert len(result.discarded_ids) == 15


def test_a_whole_window_is_still_held_to_the_limit() -> None:
    """Same reply, same shape, unsplit id: the canary case stays caught.

    It has to be this way round -- `attempts.py` only ever splits after an
    output-limited or truncated reply, so an agent answering without reading
    the window text is judged here, on attempt one, before any split exists.
    """

    output, sources = _mostly_discarded_window_output(kept=5, discarded=15)
    whole = _window("0007", sources)
    result = validate_correction_window_output(
        output, whole, variant=resolve_variant("capableB")
    )
    assert not result.ok
    assert any("over the 50% limit" in e for e in result.errors), result.errors


def test_split_depth_has_one_owner() -> None:
    """The retry loop's split budget and this gate read the same rule.

    A second copy of "the id is the lineage" in either place is the silent
    drift this repo keeps getting bitten by, so both go through the window.
    """

    from finesub.llm.stages.correction.context import WindowGeometry

    for chunk_id, depth in (("0007", 0), ("0007-a", 1), ("0007-a-b", 2)):
        window = _window(chunk_id, [SubtitleSegment("1", 0.0, 1.0, "x")])
        assert window.split_depth == depth
        assert WindowGeometry.split_depth(window) == depth


def test_the_discard_limit_is_exclusive_at_exactly_half() -> None:
    """Half discarded still yields half a window of subtitles: not a failure."""
    output, source = _mostly_discarded_output(kept=10, discarded=10)
    result = validate_translated_csv_text(output, source, require_singles=False)
    assert result.ok, result.errors


def test_empty_translated_stays_ok_when_the_window_has_no_sources() -> None:
    """Nothing to cover means nothing to report -- the guard must not overfire."""
    result = validate_translated_csv_text(
        "<translated>\n</translated>", [], require_singles=False
    )
    assert result.ok, result.errors
    assert result.segments == []


def test_merge_still_clears_window_ids_and_validation_is_what_guards_it() -> None:
    """Pins where the guard lives, so nobody "fixes" it in the wrong layer.

    `merge_translated_csv_windows` cannot tell a truncated window from one that
    legitimately discarded its ids -- newest-wins is the documented semantic and
    real `discard` rows depend on it. So the merge keeps clearing the window's
    ids, and the protection against a *truncated* window reaching it lives in
    validation: the empty block is rejected before commit. Changing the merge
    to preserve prior rows would silently resurrect discarded subtitles.
    """
    # 1. Validation is the layer that stops a truncated window.
    source = [SubtitleSegment("8", 8.0, 9.0, "eight")]
    rejected = validate_translated_csv_text(
        "<translated></translated>", source, require_singles=False
    )
    assert not rejected.ok, "validation is the layer that must stop this"

    # 2. The merge itself deliberately still clears the window's ids.
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
    assert merge_translated_csv_windows(earlier, ["8", "9", "10"], []) == []


# --- Strict arity: a wrong column count must never be reinterpreted ----------
#
# The parser used to guess the layout from cell *content*, so a row with one
# extra column was silently shifted one cell left -- putting the untranslated
# source text into the translation column. Columns are now mapped by position
# and their shapes validated, which is what separates a drifted row from the
# legitimate case of a half-width pipe inside the free-text `note`.


def _one_source() -> list[SubtitleSegment]:
    return [SubtitleSegment("1", 0.0, 1.0, "hello there")]


def test_extra_column_is_a_structural_error_not_a_silent_shift() -> None:
    """The SUMMARY sample: an empty conf cell used to shift every field left."""
    output = (
        "<translated>\n"
        f"{OUTPUT_CSV_HEADER}\n"
        "sub|1|1.0|0.8|0.2|hello there|你好||4|\n"
        "</translated>"
    )
    result = validate_translated_csv_text(
        output, _one_source(), require_singles=False, require_headers=True
    )
    assert not result.ok, "a 10-field row against a 9-column spec must fail"
    # The specific corruption this guards: source text becoming the translation.
    assert all(s.translation != "hello there" for s in result.segments)


def test_half_width_pipe_in_translation_is_rejected_not_dropped() -> None:
    """Used to warn and silently discard the text after the pipe."""
    output = (
        "<translated>\n"
        "sub|1|0.8|0.2|hello|你好|世界|high|4|\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, _one_source(), require_singles=False)
    assert not result.ok
    assert all("世界" not in (s.translation or "") for s in result.segments)


def test_pipe_inside_note_is_still_accepted() -> None:
    """The tail column keeps its documented leniency -- only it."""
    output = (
        "<translated>\n"
        "sub|1|1.0|0.0|hello there|你好|high|2|note with | pipe\n"
        "</translated>"
    )
    result = validate_translated_csv_text(output, _one_source(), require_singles=False)
    assert result.ok, result.errors
    assert result.segments[0].translation == "你好"
    assert result.segments[0].note == "note with | pipe"


def test_legacy_three_column_layout_is_no_longer_accepted() -> None:
    """Dropped with the heuristic parser; annotated.csv is regenerable.

    The row below is deliberately the old `position|corrected|translation`
    shape -- do not "modernise" it, that is the whole point of the test.
    """
    output = "<translated>\n" + "1|one|" + "一" + "\n</translated>"
    result = validate_translated_csv_text(output, _one_source(), require_singles=False)
    assert not result.ok
    assert any("too few fields" in e for e in result.errors), result.errors


def _as_a_broken_cli_would_write(text: str) -> str:
    return "".join(
        character if ord(character) < 128 else f"\\u{ord(character):04x}"
        for character in text
    )


def _reply_over(rows: list[tuple[str, str, str]]) -> str:
    body = "\n".join(
        f"sub|{position}|1.0|0.0|{source}|{translation}|high|3|"
        for position, source, translation in rows
    )
    return (
        f"<singles>\n{OUTPUT_CSV_HEADER}\n{body}\n</singles>\n"
        f"<translated>\n{OUTPUT_CSV_HEADER}\n{body}\n</translated>"
    )


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
    success the whole way. The `char_count` check did notice -- 15 rows, every
    one off by a factor of three -- and normalized the evidence away, which is
    why the refusal has to come before any of that.
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


def _reply_with_counts(rows: list[tuple[str, str, str, str]]) -> str:
    body = "\n".join(
        f"sub|{position}|1.0|0.0|{source}|{translation}|high|{count}|"
        for position, source, translation, count in rows
    )
    return (
        f"<singles>\n{OUTPUT_CSV_HEADER}\n{body}\n</singles>\n"
        f"<translated>\n{OUTPUT_CSV_HEADER}\n{body}\n</translated>"
    )


#: Ten CJK characters, so the computed weighted count is 10.
_TEN = "大家好今天也请多关照"


def _counted(reported: list[str]):
    rows = [
        (str(index + 1), f"source {index + 1}", _TEN, count)
        for index, count in enumerate(reported)
    ]
    source = [
        SubtitleSegment(str(index + 1), index * 2.0, index * 2.0 + 1.0, f"source {index + 1}")
        for index in range(len(reported))
    ]
    return validate_translated_csv_text(_reply_with_counts(rows), source)


def _systematic(result) -> list[str]:
    return [w for w in result.warnings if "in one direction" in w]


def test_a_whole_window_under_reporting_is_called_out() -> None:
    """The incident's shape: every row disagreed, all the same way, at a ratio
    near three -- because the text being measured was not the text the model
    wrote. Neither the share nor the direction has any precedent in the
    116-exchange baseline (plan §12, 离线基线测量)."""

    result = _counted(["3", "3", "3", "3"])

    assert result.ok, result.errors
    assert len(_systematic(result)) == 1
    assert "4 of 4 rows" in _systematic(result)[0]


def test_one_bad_row_is_not_systematic() -> None:
    """Ordinary. Models are not good at counting their own characters, and the
    per-row warning already says so."""

    result = _counted(["3", "10", "10", "10"])

    assert _systematic(result) == []


def test_over_reporting_is_not_called_out_however_consistent() -> None:
    """The direction condition, and the reason it is not redundant with the
    share: *every* baseline ratio (38 of 38) was below 1 -- models over-report
    their own length. ⚠ Those 38 come from two Gemini models; the other five in
    the corpus had no disagreeing rows at all, so they say nothing about
    direction. A window full of over-reporting is the normal weakness, not
    evidence that the text changed underneath us."""

    result = _counted(["20", "20", "20", "20"])

    # Every row still warns individually...
    assert sum("does not match" in w for w in result.warnings) == 8
    # ...but the window-level line stays quiet.
    assert _systematic(result) == []


def test_the_message_counts_the_directions_rather_than_assuming_them() -> None:
    """The gate is a *median*, so a minority of rows may lean the other way.

    Saying "N of N rows report less" would then be false about some of them --
    and a diagnostic that misdescribes its own evidence is worse than none,
    because the next reader checks the wrong column.
    """

    from finesub.llm.output_protocol import _systematic_char_count_warning

    # One row over-reports, two under-report: median is 3, the gate opens.
    mixed = [(20.0, 10.0), (5.0, 15.0), (5.0, 15.0)]

    (message,) = _systematic_char_count_warning(mixed, 3)

    assert "3 of 3 rows disagree" in message
    assert "2 of them reporting less" in message


def test_the_count_is_still_normalized_either_way() -> None:
    """The warning is added, nothing is taken away: the char_count column is
    still replaced by the computed value, as it was before."""

    result = _counted(["3", "3", "3", "3"])

    assert {segment.char_count for segment in result.segments} == {
        format_weighted_char_count(weighted_char_count(_TEN))
    }
