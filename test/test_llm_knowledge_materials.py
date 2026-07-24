from __future__ import annotations

import json

import pytest

from llm.chunking import SubtitleSegment
from llm.knowledge.base import append_task_artifact, apply_knowledge_proposals
from llm.knowledge.entries import (
    render_kb_entry_excerpt,
    select_kb_entries,
)
from llm.knowledge.feedback import (
    KnowledgeHint,
    aggregate_task_update_feedback,
    parse_task_update_feedback,
)
from llm.knowledge.materials import (
    ExecutedWindow,
    FinalRow,
    MODE_ARTIFACTS_ONLY,
    MODE_REFINED_ALIGNED,
    WindowMaterials,
    build_final_rows,
    build_knowledge_materials,
    group_rows_by_window,
    load_executed_windows,
    load_refined_segments,
    plan_knowledge_chunks,
    render_final_csv,
    render_refined_csv,
    split_refined_by_window,
)
from llm.srt_utils import SrtSegment, render_srt


def _count(text: str) -> int:
    return len(text)


# ---------------------------------------------------------------------------
# feedback v2 parsing / aggregation


def _feedback_json(**overrides) -> str:
    data = {
        "knowledge_hints": [
            {
                "category": "streamer",
                "entry": "星野灯",
                "direction": "append_lines",
                "focus": "新增3D周年配信",
                "reason": "窗口音频与搜索结果",
                "source_ids": ["12", "13"],
                "confidence": 7,
            }
        ],
        "asr_corrections": ["Akari 误听为 光"],
        "uncertainties": ["联动对象存疑"],
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


def test_parse_task_update_feedback_reads_v2_schema() -> None:
    feedback = parse_task_update_feedback(
        _feedback_json(), origin="window", chunk_id="0001"
    )

    assert feedback.chunk_id == "0001"
    assert feedback.hints[0].entry == "星野灯"
    assert feedback.hints[0].direction == "append_lines"
    assert feedback.hints[0].source_ids == ("12", "13")
    assert feedback.hints[0].confidence == 7
    assert feedback.asr_corrections == ("Akari 误听为 光",)
    assert not feedback.warnings


def test_parse_task_update_feedback_drops_invalid_hints_without_failing() -> None:
    body = json.dumps(
        {
            "knowledge_hints": [
                {"category": "translation", "entry": "x"},  # bad category
                {"category": "common", "entry": ""},  # empty entry
                {"category": "common", "entry": "原神", "direction": "weird"},
                "not-an-object",
            ],
            "uncertainties": [],
        },
        ensure_ascii=False,
    )

    feedback = parse_task_update_feedback(body, origin="window", chunk_id="0001")

    assert [hint.entry for hint in feedback.hints] == ["原神"]
    assert feedback.hints[0].direction == ""  # unknown direction dropped, hint kept
    assert len(feedback.warnings) == 4


def test_parse_task_update_feedback_coerces_concatenated_categories() -> None:
    # Models sometimes read the schema's "streamer|common" enum notation as a
    # template and emit "streamer|game_lore"; the leading segment is salvaged.
    body = json.dumps(
        {
            "knowledge_hints": [
                {"category": "streamer|game_lore", "entry": "星野灯"},
                {"category": "Common|梗", "entry": "原神"},
            ],
        },
        ensure_ascii=False,
    )

    feedback = parse_task_update_feedback(body, origin="window", chunk_id="0001")

    assert [(hint.category, hint.entry) for hint in feedback.hints] == [
        ("streamer", "星野灯"),
        ("common", "原神"),
    ]
    assert len(feedback.warnings) == 2


def test_parse_task_update_feedback_bad_json_degrades_to_empty() -> None:
    feedback = parse_task_update_feedback("not json {", origin="research")

    assert feedback.is_empty
    assert feedback.warnings


def test_aggregate_reads_artifacts_last_wins_and_scores(tmp_path) -> None:
    append_task_artifact(
        tmp_path,
        kind="correction_window_task_feedback",
        task_id="t",
        payload={"chunk_id": "0001", "feedback": _feedback_json()},
    )
    # Retry of the same window: last record wins.
    append_task_artifact(
        tmp_path,
        kind="correction_window_task_feedback",
        task_id="t",
        payload={
            "chunk_id": "0001",
            "feedback": _feedback_json(uncertainties=["retry版"]),
        },
    )
    append_task_artifact(
        tmp_path,
        kind="correction_window_task_feedback",
        task_id="t",
        payload={
            "chunk_id": "0002",
            "feedback": json.dumps(
                {
                    "knowledge_hints": [
                        {"category": "common", "entry": "原神", "direction": "new_entry"}
                    ]
                }
            ),
        },
    )
    append_task_artifact(
        tmp_path,
        kind="research_task_feedback",
        task_id="t",
        payload={"feedback": _feedback_json()},
    )

    aggregate = aggregate_task_update_feedback([tmp_path])

    assert aggregate.window_feedback["0001"].uncertainties == ("retry版",)
    assert aggregate.research_feedback is not None
    assert aggregate.merged_uncertainties() == ["retry版", "联动对象存疑"]
    scores = aggregate.hint_scores()
    # window ×1 + research ×2
    assert scores[("streamer", "星野灯")] == 3.0
    assert scores[("common", "原神")] == 1.0
    assert "星野灯" in aggregate.feedback_slice_text("0001")
    assert aggregate.research_slice_text()


# ---------------------------------------------------------------------------
# entry selection / excerpt rendering


def _seed_knowledge(tmp_path) -> None:
    apply_knowledge_proposals(
        json.dumps(
            {
                "category": "streamer",
                "entry": "星野灯",
                "aliases": ["阿灯"],
                "intro": "虚拟主播",
                "op": "replace_section",
                "section": "档案",
                "content": "关西腔，喜欢恐怖游戏。",
                "reason": "seed",
            },
            ensure_ascii=False,
        ),
        knowledge_root=tmp_path,
        commit=False,
    )


def test_select_kb_entries_merges_aliases_and_ranks(tmp_path) -> None:
    _seed_knowledge(tmp_path)
    window_hints = [
        KnowledgeHint(category="streamer", entry="阿灯"),
        KnowledgeHint(category="streamer", entry="星野灯"),
        KnowledgeHint(category="common", entry="新游戏X", direction="new_entry"),
    ]
    research_hints = [KnowledgeHint(category="common", entry="新游戏X")]

    selections = select_kb_entries(
        window_hints,
        knowledge_root=tmp_path,
        research_origins=research_hints,
        applied_entries=[("streamer", "星野灯")],
    )

    assert [(s.category, s.key) for s in selections] == [
        ("common", "新游戏X"),  # 1 + 2 = 3
        ("streamer", "星野灯"),  # 1 + 1 = 2 (alias merged)
    ]
    assert selections[0].exists is False
    assert selections[1].exists is True
    assert selections[1].applied is True
    assert set(selections[1].hint_names) == {"阿灯", "星野灯"}


def test_select_kb_entries_caps_count(tmp_path) -> None:
    hints = [
        KnowledgeHint(category="common", entry=f"条目{i}") for i in range(30)
    ]
    selections = select_kb_entries(hints, knowledge_root=tmp_path, max_entries=20)
    assert len(selections) == 20


def test_render_kb_entry_excerpt_annotates_states(tmp_path) -> None:
    _seed_knowledge(tmp_path)
    selections = select_kb_entries(
        [
            KnowledgeHint(category="streamer", entry="星野灯"),
            KnowledgeHint(category="common", entry="新游戏X"),
        ],
        knowledge_root=tmp_path,
        applied_entries=[("streamer", "星野灯")],
    )

    block = render_kb_entry_excerpt(selections, tmp_path, count_tokens=_count)

    # Non-heading delimiter: the entry body's own #/## headings must stay the
    # only markdown headings inside the block.
    assert "--- streamer/星野灯 ---" in block.text
    assert "关西腔" in block.text
    assert "本任务前序块已更新" in block.text
    assert "--- common/新游戏X ---" in block.text
    assert "库中暂无" in block.text


# ---------------------------------------------------------------------------
# final_csv overlay


def _annotated_text() -> str:
    return (
        "# type|position|duration|gap|corrected|translation|conf|char_count|note\n"
        "sub|1,2|3.5|16.5|Hello there|你好啊|high|3|\n"
        "insert|30.0,2.0|2.0|7.5|Missed line|漏掉的一句|median|5|插轴\n"
        "sub|4|1.0|0.0|Bye|再见|high|2|\n"
    )


def _final_srt_text() -> str:
    return render_srt(
        [
            SrtSegment(index=1, start=10.0, end=13.5, text="你好啊！"),
            SrtSegment(index=2, start=30.0, end=32.5, text="漏掉的一句"),
            SrtSegment(index=3, start=40.0, end=41.0, text="再见"),
        ]
    )


def test_build_final_rows_overlays_srt_timing_and_translation() -> None:
    rows = build_final_rows(_annotated_text(), _final_srt_text())

    assert rows[0].source_ids == ("1", "2")
    assert rows[0].translation == "你好啊！"  # postprocessed text wins
    assert rows[0].corrected == "Hello there"  # corrected NOT overlaid
    assert rows[0].start == 10.0 and rows[0].end == 13.5
    assert rows[0].gap == pytest.approx(16.5)  # 30.0 - 13.5, recomputed
    assert rows[1].is_insert and rows[1].source_ids == ()
    assert rows[2].gap == 0.0  # last row


def test_build_final_rows_rejects_count_mismatch() -> None:
    srt = render_srt([SrtSegment(index=1, start=0.0, end=1.0, text="只有一行")])
    with pytest.raises(ValueError, match="row count"):
        build_final_rows(_annotated_text(), srt)


def test_render_final_csv_has_ten_columns() -> None:
    rows = build_final_rows(_annotated_text(), _final_srt_text())
    text = render_final_csv(rows)
    lines = text.strip().splitlines()
    assert lines[0].startswith("# type|position|start|end|gap|")
    assert lines[1] == "sub|1,2|10.0|13.5|16.5|Hello there|你好啊！|high|3|"
    assert lines[2].startswith("insert|30.0,2.5|30.0|32.5|")


# ---------------------------------------------------------------------------
# window grouping (stitch ownership)


def _segments(count: int, *, step: float = 10.0) -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            id=str(i + 1), start=i * step, end=i * step + 5.0, text=f"line {i + 1}"
        )
        for i in range(count)
    ]


def _sub_row(ids: tuple[str, ...], start: float, end: float) -> FinalRow:
    return FinalRow(
        kind="sub",
        source_ids=ids,
        corrected="c",
        translation="t",
        conf=None,
        char_count="1",
        note="",
        start=start,
        end=end,
        gap=0.0,
    )


def test_group_rows_by_window_owns_overlap_by_last_window() -> None:
    segments = _segments(6)
    windows = [
        ExecutedWindow(chunk_id="0001", source_ids=("1", "2", "3", "4")),
        ExecutedWindow(chunk_id="0002", source_ids=("3", "4", "5", "6")),
    ]
    rows = [
        _sub_row(("1", "2"), 0.0, 15.0),
        _sub_row(("3",), 20.0, 25.0),
        _sub_row(("4", "5"), 30.0, 45.0),
        _sub_row(("6",), 50.0, 55.0),
    ]

    groups = group_rows_by_window(rows, windows, segments)

    assert [g.chunk_id for g in groups] == ["0001", "0002"]
    # Overlap ids 3/4 belong to window 0002 (newest wins).
    assert [row.source_ids for row in groups[0].final_rows] == [("1", "2")]
    assert [row.source_ids for row in groups[1].final_rows] == [
        ("3",),
        ("4", "5"),
        ("6",),
    ]
    # Raw rows follow their final rows: no id rendered twice.
    ids_0 = [seg.id for seg in groups[0].raw_segments]
    ids_1 = [seg.id for seg in groups[1].raw_segments]
    assert ids_0 == ["1", "2"]
    assert ids_1 == ["3", "4", "5", "6"]
    assert not set(ids_0) & set(ids_1)


def test_group_rows_keeps_straddling_row_whole_and_surfaces_dropped_ids() -> None:
    segments = _segments(6)
    windows = [
        ExecutedWindow(chunk_id="0001", source_ids=("1", "2", "3", "4")),
        ExecutedWindow(chunk_id="0002", source_ids=("3", "4", "5", "6")),
    ]
    rows = [
        # Backfilled straddling row: first id owned by 0001, second by 0002.
        _sub_row(("2", "3"), 10.0, 25.0),
        _sub_row(("5",), 40.0, 45.0),
        # id 1, 4 and 6 dropped by the model -> raw only.
    ]

    groups = group_rows_by_window(rows, windows, segments)

    group_by_id = {g.chunk_id: g for g in groups}
    assert [row.source_ids for row in group_by_id["0001"].final_rows] == [("2", "3")]
    # The straddling row pulls raw id 3 into 0001's pack; dropped id 1 stays.
    assert [seg.id for seg in group_by_id["0001"].raw_segments] == ["1", "2", "3"]
    # Dropped ids 4/6 surface in their owner window's raw_csv.
    assert [seg.id for seg in group_by_id["0002"].raw_segments] == ["4", "5", "6"]


def test_group_rows_assigns_inserts_by_time_and_falls_back_without_windows() -> None:
    segments = _segments(4)
    insert = FinalRow(
        kind="insert",
        source_ids=(),
        corrected="x",
        translation="x",
        conf=None,
        char_count="1",
        note="",
        start=12.0,
        end=13.0,
        gap=0.0,
    )
    rows = [_sub_row(("1", "2"), 0.0, 15.0), insert, _sub_row(("3", "4"), 20.0, 35.0)]

    groups = group_rows_by_window(rows, [], segments)

    assert len(groups) == 1 and groups[0].chunk_id == "all"
    assert len(groups[0].final_rows) == 3

    windows = [
        ExecutedWindow(chunk_id="0001", source_ids=("1", "2")),
        ExecutedWindow(chunk_id="0002", source_ids=("3", "4")),
    ]
    groups = group_rows_by_window(rows, windows, segments)
    group_by_id = {g.chunk_id: g for g in groups}
    assert any(row.is_insert for row in group_by_id["0001"].final_rows)
    assert not any(row.is_insert for row in group_by_id["0002"].final_rows)


def test_load_executed_windows_orders_and_dedupes(tmp_path) -> None:
    append_task_artifact(
        tmp_path,
        kind="correction_window_response",
        task_id="t",
        payload={
            "chunk_id": "0001",
            "validation_ok": True,
            "output_limited": False,
            "window": {"source_ids": ["1", "2"]},
        },
    )
    # Failed attempt: ignored.
    append_task_artifact(
        tmp_path,
        kind="correction_window_response",
        task_id="t",
        payload={
            "chunk_id": "0002",
            "validation_ok": False,
            "output_limited": False,
            "window": {"source_ids": ["2", "3"]},
        },
    )
    append_task_artifact(
        tmp_path,
        kind="correction_window_cached",
        task_id="t",
        payload={"chunk_id": "0001", "source_ids": ["1", "2"]},
    )
    append_task_artifact(
        tmp_path,
        kind="correction_window_response",
        task_id="t",
        payload={
            "chunk_id": "0002",
            "validation_ok": True,
            "output_limited": False,
            "window": {"source_ids": ["2", "3"]},
        },
    )

    windows = load_executed_windows([tmp_path])

    assert [(w.chunk_id, w.source_ids) for w in windows] == [
        ("0001", ("1", "2")),
        ("0002", ("2", "3")),
    ]


# ---------------------------------------------------------------------------
# refined_csv


def test_load_refined_segments_resorts_by_time() -> None:
    text = render_srt(
        [
            SrtSegment(index=5, start=20.0, end=22.0, text="后面"),
            SrtSegment(index=1, start=1.0, end=3.0, text="前面"),
        ],
        reindex=False,
    )

    segments = load_refined_segments(text)

    assert [seg.text for seg in segments] == ["前面", "后面"]


def test_split_refined_by_window_assigns_by_overlap_then_nearest() -> None:
    segments = _segments(4)
    windows = [
        ExecutedWindow(chunk_id="0001", source_ids=("1", "2")),
        ExecutedWindow(chunk_id="0002", source_ids=("3", "4")),
    ]
    rows = [_sub_row(("1", "2"), 0.0, 15.0), _sub_row(("3", "4"), 20.0, 35.0)]
    groups = group_rows_by_window(rows, windows, segments)
    refined = [
        SrtSegment(index=1, start=2.0, end=4.0, text="第一窗"),
        # Straddles both, overlaps window 2 more.
        SrtSegment(index=2, start=14.0, end=30.0, text="偏后"),
        # Outside every range: nearest is window 2.
        SrtSegment(index=3, start=100.0, end=101.0, text="远处注释"),
    ]

    assignment = split_refined_by_window(refined, groups)

    assert [seg.text for seg in assignment["0001"]] == ["第一窗"]
    assert [seg.text for seg in assignment["0002"]] == ["偏后", "远处注释"]


def test_render_refined_csv_uses_start_end_columns() -> None:
    text = render_refined_csv([SrtSegment(index=1, start=1.25, end=3.0, text="a|b")])
    assert text.splitlines()[0] == "# start|end|text"
    assert text.splitlines()[1] == "1.2|3.0|a｜b"


# ---------------------------------------------------------------------------
# chunk planning


def _materials(chunk_id: str, csv_chars: int) -> WindowMaterials:
    return WindowMaterials(
        chunk_id=chunk_id,
        start=0.0,
        end=1.0,
        context_slice="",
        feedback_slice="",
        raw_csv="r" * csv_chars,
        final_csv="",
        refined_csv="",
    )


def test_plan_knowledge_chunks_cuts_on_window_boundaries() -> None:
    windows = [_materials("0001", 60), _materials("0002", 60), _materials("0003", 10)]

    chunks = plan_knowledge_chunks(windows, count_tokens=_count, csv_token_budget=100)

    assert [chunk.window_ids for chunk in chunks] == [("0001",), ("0002", "0003")]
    assert chunks[0].csv_tokens == 60
    assert chunks[1].csv_tokens == 70


def test_plan_knowledge_chunks_oversized_window_gets_own_chunk() -> None:
    windows = [_materials("0001", 500), _materials("0002", 10)]

    chunks = plan_knowledge_chunks(windows, count_tokens=_count, csv_token_budget=100)

    assert [chunk.window_ids for chunk in chunks] == [("0001",), ("0002",)]


# ---------------------------------------------------------------------------
# end-to-end material assembly


def _write_task_fixture(tmp_path, *, with_refined: bool):
    stable = tmp_path / "x-stable.json"
    stable.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 10.0, "end": 13.0, "text": "hello"},
                    {"id": "2", "start": 14.0, "end": 16.0, "text": "there"},
                    {"id": "3", "start": 40.0, "end": 41.0, "text": "bye"},
                ]
            }
        ),
        encoding="utf-8",
    )
    annotated = tmp_path / "x-annotated.csv"
    annotated.write_text(
        "# type|position|duration|corrected|translation|conf|note\n"
        "sub|1,2|3.5|Hello there|你好啊|8|\n"
        "insert|30.0,2.0|2.0|Missed line|漏掉的一句|5|插轴\n"
        "sub|3|1.0|Bye|再见||\n",
        encoding="utf-8",
    )
    final_srt = tmp_path / "x.srt"
    final_srt.write_text(_final_srt_text(), encoding="utf-8")
    context = tmp_path / "x-research-context.json"
    context.write_text(
        json.dumps(
            {
                "context_pack": {
                    "general_context": {"global_summary": "整体摘要"},
                    "window_contexts": {"0001": "第一窗背景", "0002": "第二窗背景"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "x.llm-artifacts"
    for chunk_id, ids in (("0001", ["1", "2"]), ("0002", ["2", "3"])):
        append_task_artifact(
            artifact_dir,
            kind="correction_window_response",
            task_id="t",
            payload={
                "chunk_id": chunk_id,
                "validation_ok": True,
                "output_limited": False,
                "window": {"source_ids": ids},
            },
        )
    append_task_artifact(
        artifact_dir,
        kind="correction_window_task_feedback",
        task_id="t",
        payload={"chunk_id": "0001", "feedback": _feedback_json()},
    )
    append_task_artifact(
        artifact_dir,
        kind="research_task_feedback",
        task_id="t",
        payload={"feedback": _feedback_json()},
    )
    refined = None
    if with_refined:
        refined = tmp_path / "refined.srt"
        refined.write_text(
            render_srt(
                [
                    SrtSegment(index=2, start=40.0, end=41.0, text="拜拜"),
                    SrtSegment(index=1, start=10.0, end=13.0, text="你好呀"),
                ],
                reindex=False,
            ),
            encoding="utf-8",
        )
    return stable, annotated, final_srt, context, artifact_dir, refined


def test_build_knowledge_materials_artifacts_only(tmp_path) -> None:
    stable, annotated, final_srt, context, artifact_dir, _ = _write_task_fixture(
        tmp_path, with_refined=False
    )

    materials = build_knowledge_materials(
        stable_json=stable,
        annotated_csv=annotated,
        final_srt=final_srt,
        research_context=context,
        artifact_dirs=[artifact_dir],
        count_tokens=_count,
    )

    assert materials.mode == MODE_ARTIFACTS_ONLY
    assert materials.window_count == 2
    assert not materials.warnings
    pack = materials.chunks[0].packs_text()
    assert "--- window 0001" in pack
    assert "<context_slice>\n第一窗背景" in pack
    assert "星野灯" in pack  # feedback slice
    assert "<raw_csv>" in pack and "<final_csv>" in pack
    assert "<refined_csv>" not in pack
    assert "整体摘要" in materials.general_context


def test_build_knowledge_materials_refined_mode(tmp_path) -> None:
    stable, annotated, final_srt, context, artifact_dir, refined = _write_task_fixture(
        tmp_path, with_refined=True
    )

    materials = build_knowledge_materials(
        stable_json=stable,
        annotated_csv=annotated,
        final_srt=final_srt,
        research_context=context,
        artifact_dirs=[artifact_dir],
        refined_srt=refined,
        count_tokens=_count,
    )

    assert materials.mode == MODE_REFINED_ALIGNED
    pack = materials.chunks[0].packs_text()
    assert "<refined_csv>" in pack
    assert "你好呀" in pack and "拜拜" in pack


def test_build_knowledge_materials_degrades_without_artifacts(tmp_path) -> None:
    stable, annotated, final_srt, context, _, _ = _write_task_fixture(
        tmp_path, with_refined=False
    )

    materials = build_knowledge_materials(
        stable_json=stable,
        annotated_csv=annotated,
        final_srt=final_srt,
        research_context=context,
        artifact_dirs=[tmp_path / "missing-artifacts"],
        count_tokens=_count,
    )

    assert materials.window_count == 1
    assert materials.chunks[0].windows[0].chunk_id == "all"
    assert any("fallback window" in warning for warning in materials.warnings)
    assert any("task_update_feedback" in warning for warning in materials.warnings)
