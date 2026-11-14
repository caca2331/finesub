from __future__ import annotations

import dataclasses
import re
from dataclasses import replace as dataclass_replace

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.prompt_compose import PROMPT_TEMPLATE_DIR
from finesub.llm.prompts import (
    ContextPack,
    build_correction_csv_messages,
    build_knowledge_update_messages,
)
from finesub.llm.prompt_artifacts import write_prompt_artifacts
from finesub.llm.length_tier import TaskLengthScale

#: A dry run's ratio with nothing to resolve it from (`length_tier.task_length_scale`).
STANDARD_SCALE = TaskLengthScale(1.0, "default")


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def _direct_input_block(content: str, tag: str) -> str:
    match = re.search(
        rf"(?ms)^[ \t]*<{re.escape(tag)}>[ \t]*\r?\n"
        rf"(.*?)^[ \t]*</{re.escape(tag)}>[ \t]*$",
        content,
    )
    return match.group(1).strip() if match else ""



def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "えっと、あの子ちゃんが来た。"),
        SubtitleSegment("2", 1.2, 2.0, "Haha やばい。"),
    ]


def test_correction_prompt_forbids_omitting_long_singles_or_translated() -> None:
    window = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    messages = build_correction_csv_messages(window=window)
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "不得因输出较长而省略" in combined
    assert "<singles>" not in combined
    assert "translated 必须给出完整终稿" in combined
    assert "严格保持标签、header、6 列字段与顺序" in combined


def test_context_pack_matches_source_coverage_after_split() -> None:
    base = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    split = type(base)(
        chunk_id="0001-a",
        segments=[base.segments[0]],
        overlap_segments=[],
        boundary_reason="test",
        budget=base.budget,
    )
    pack = ContextPack.from_dict(
        {
            "general_context": {},
            "window_contexts": [{
                "window_id": "0001",
                "first_source_id": "1",
                "last_source_id": "2",
                "context": "窗口背景",
            }],
        }
    ).with_source_order(["1", "2"])

    assert pack.window_context_for(split, counter=FakeTokenCounter()) == "窗口背景"


def test_context_pack_from_dict_accepts_window_context_list() -> None:
    pack = ContextPack.from_dict(
        {
            "general_context": {"global_summary": "摘要"},
            "window_contexts": [
                {"window_id": "0001", "context": "背景一"},
                {"window_id": "0002", "context": "背景二"},
            ],
        }
    )

    assert pack.unbound_window_contexts == {"0001": "背景一", "0002": "背景二"}
    assert pack.window_contexts == ()


def test_bind_window_ranges_reports_ids_it_could_not_place() -> None:
    """A pack that loses every note must not look like a pack that had none.

    Binding drops notes whose window id is not in the plan. With no report,
    a round 2 that gets the ids systematically wrong writes a well-formed
    artifact holding zero notes -- and the loader's unbound-note check, which
    exists to refuse half a pack, sees nothing wrong with it.
    """

    segments = [
        SubtitleSegment(str(index), float(index), float(index) + 0.5, str(index))
        for index in range(1, 7)
    ]
    windows = [dataclasses.replace(w, length_scale=1.0) for w in plan_correction_windows(segments, counter=FakeTokenCounter())]
    planned_id = windows[0].chunk_id

    report: dict[str, object] = {}
    bound = ContextPack.from_dict(
        {
            "general_context": {"global_summary": "摘要"},
            "window_contexts": [
                {"window_id": planned_id, "context": "能放下的笔记"},
                {"window_id": "chunk-1", "context": "模型自己编的 id"},
            ],
        }
    ).bind_window_ranges(windows, report_sink=report)

    assert [note.context for note in bound.window_contexts] == ["能放下的笔记"]
    assert report["unplaceable_window_ids"] == ["chunk-1"]
    assert report["bound"] == 1

    # Every id wrong -> zero notes, and the report is the only evidence.
    all_wrong: dict[str, object] = {}
    empty = ContextPack.from_dict(
        {"window_contexts": [{"window_id": "chunk-1", "context": "笔记"}]}
    ).bind_window_ranges(windows, report_sink=all_wrong)
    assert empty.window_contexts == ()
    assert not empty.has_unbound_window_contexts
    assert all_wrong["unplaceable_window_ids"] == ["chunk-1"]


def test_context_pack_concatenates_contained_notes_and_drops_crossing_notes() -> None:
    segments = [
        SubtitleSegment(str(index), float(index), float(index) + 0.5, str(index))
        for index in range(1, 7)
    ]
    base = dataclasses.replace(plan_correction_windows(segments, counter=FakeTokenCounter())[0], length_scale=1.0)
    window = dataclass_replace(
        base,
        chunk_id="new",
        segments=segments[1:5],
        overlap_segments=[segments[1]],
    )
    pack = ContextPack.from_dict(
        {
            "window_contexts": [
                {"window_id": "left", "first_source_id": "2", "last_source_id": "3", "context": "越界"},
                {"window_id": "a", "first_source_id": "3", "last_source_id": "3", "context": "三"},
                {"window_id": "b", "first_source_id": "4", "last_source_id": "5", "context": "四五"},
            ]
        }
    ).with_source_order([segment.id for segment in segments])

    assert pack.window_context_for(window, counter=FakeTokenCounter()) == "三\n\n四五"


def test_a_correction_window_reads_its_own_chunks_general_context() -> None:
    window = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    ids = [segment.id for segment in _segments()]
    pack = ContextPack.from_dict({
        "general_context": {"c01": {"global_summary": "前段"}, "c02": {"global_summary": "后段"}},
        "general_chunks": [{"chunk_id": "c01", "first_source_id": ids[0], "last_source_id": ids[-1]},
                           {"chunk_id": "c02", "first_source_id": "x1", "last_source_id": "x2"}],
    }).with_source_order([*ids, "x1", "x2"])
    user = build_correction_csv_messages(window=window, context_pack=pack, audio_file_label="clip.wav",
                                         variant="basicA")[1]["content"]
    general = pack.general_text_for(window)
    assert general in user and "本窗口位于 c01 段" in general and "后段" in general
    assert pack.general_text_for_source_ids(["x1"]) != general


def test_only_the_knowledge_update_reads_the_whole_view_and_the_accounting_reads_the_window() -> None:
    """The correction's token accounting must count what its prompt sends (`general_text_for`)."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "finesub"
    whole = sorted(str(path.relative_to(src)).replace("\\", "/") for path in src.rglob("*.py")
                   if "general_text_whole()" in path.read_text(encoding="utf-8"))
    assert whole == ["llm/knowledge/materials.py"]
    accounting = [line.strip() for path in src.rglob("*.py")
                  for line in path.read_text(encoding="utf-8").splitlines() if re.search(r"context_general=", line)]
    assert accounting and all("general_text_for(" in line for line in accounting), accounting


@pytest.mark.parametrize(("media", "retrieval", "knowledge", "correct_only"), [
    ("text", "local", True, False), ("text", "none", False, False), ("audio", "local", True, True)])
def test_a_correction_user_prompt_says_how_to_follow_relayed_requests_only_when_there_are_some(
        media, retrieval, knowledge, correct_only) -> None:
    from finesub.llm.prompt_compose import compose_correction_user
    from finesub.llm.routing.profiles import resolve_profile

    profile = dataclasses.replace(resolve_profile(media, retrieval, "quality"), correct_only=correct_only)
    for grouped, variant in ((False, "basicA"), (True, "capableC")):
        variant += "-polish" if correct_only else ""

        def user(**switches):
            return compose_correction_user(
                profile, general_context_json="{}", window_context="", entry_details="", previous_advice="",
                pre_round_notes="", search_results="", preceding_context_csv="", current_asr_csv="",
                current_asr_row_count=1, knowledge_enabled=knowledge, grouped=grouped, variant=variant,
                length_scale=1.0, **switches)

        assert "`user_requests`" not in user() and "`user_requests`" not in user(extra_style="口语化")
        relayed = user(user_requests=True)
        assert "`user_requests`" in relayed and "可以变通" in relayed and "内容照实际的写" in relayed
        assert f"用在 `{'polished' if correct_only else 'translation'}`" in relayed
        assert "特殊翻译风格要求" not in relayed
        # `--extra-style` asks for a translation; a correct-only system has no such block to rank against
        assert ("特殊翻译风格要求" in user(user_requests=True, extra_style="口语化")) is not correct_only


def test_the_requests_block_follows_what_the_windows_background_relays() -> None:
    window = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    ids = [segment.id for segment in window.segments]

    def user(pack):
        return build_correction_csv_messages(window=window, context_pack=pack)[1]["content"]

    assert "`user_requests`" not in user(ContextPack(general_context={"global_summary": "s"}))
    assert "`user_requests`" in user(ContextPack(general_context={"global_summary": "s"}, user_requests=("歌名用《》",)))
    noted = ContextPack.from_dict({"general_context": {"global_summary": "s"}, "window_contexts": [
        {"window_id": "0001", "first_source_id": ids[0], "last_source_id": ids[-1],
         "context": "出自用户：这段副歌照给的原文写"}]}).with_source_order(ids)
    assert "`user_requests`" in user(noted)


def test_the_knowledge_update_names_the_columns_its_csv_headers_name() -> None:
    from finesub.llm.knowledge.materials import FINAL_CSV_HEADER, RAW_CSV_HEADER, REFINED_CSV_HEADER, render_raw_csv

    system = build_knowledge_update_messages(refined=True, task_summary="t", window_packs="p")[0]["content"]
    for tag, header in (("raw_csv", RAW_CSV_HEADER), ("final_csv", FINAL_CSV_HEADER), ("refined_csv", REFINED_CSV_HEADER)):
        named = re.search(rf"`<{tag}>`：[^`]*`([^`]+)`", system).group(1)
        assert named == header.removeprefix("# "), tag
    assert render_raw_csv([]) == RAW_CSV_HEADER + "\n", "a window without raw rows keeps the block's shape"
    assert render_raw_csv([SubtitleSegment("3", 1.0, 2.0, "a")]).splitlines()[0] == RAW_CSV_HEADER


def test_basic_a_prompt_locks_one_row_per_source_and_style_rules() -> None:
    window = dataclasses.replace(plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0], length_scale=1.0)
    messages = build_correction_csv_messages(
        window=window,
        context_pack=ContextPack.from_dict(
            {
                "general_context": {"global_summary": "主播正在玩游戏。", "must": ["角色名：小明"]},
                "window_contexts": [{
                    "window_id": window.chunk_id,
                    "first_source_id": "1",
                    "last_source_id": "2",
                    "context": "本窗口在打 BOSS。",
                }],
            }
        ).with_source_order(["1", "2"]),
        audio_file_label="clip.wav",
        previous_advice="BOSS 名固定译为「王」。",
        variant="basicA",
    )
    system = messages[0]["content"]

    assert "<singles>" not in system
    assert "<translated>...</translated>" in system or "<translated>" in system
    assert "不要 Markdown 代码块" in system
    assert "不要无标签散文" in system
    assert "一次性完整输出" in system
    assert "<asr_result>" in system
    assert "Haha -> 哈哈" in system
    assert "音似的假名" in system
    assert "原始音频" in system
    assert "不擅自添加原文没有的人称" in system
    # the style-like tendencies live in the factory default style now (owner 2026-10-31), not the general rules
    assert "やばい" not in system and "微调语序" not in system and "保留原文语气" not in system
    assert "本档一行对一个源" in system
    assert "仅词中切断的接回" in system
    assert "<translated>" in system
    assert "\nplan|" not in system  # no plan example rows (v34)
    assert "五选一" not in system
    oneshot = system.split("终稿（一行对一个源，仅词中接回可两源）", 1)[1]
    oneshot_translated = oneshot.split("<translated>", 1)[1].split(
        "</translated>", 1
    )[0]
    assert "莉奈娅" in system or "リンネ" in system
    assert "ルビルビルビルビ" in system
    assert "sub|1|" in system and "sub|43|" in system
    assert all(
        len(line.split("|")) == 7
        for line in oneshot_translated.splitlines()
        if line.startswith("sub|")
    )
    # BasicA 主 oneshot 一行对一个源；唯一的多源行是契约里词中接回的那条行例。
    assert "sub|10,11|" not in oneshot_translated
    assert "`sub|57,58|" in system
    assert "三源及以上禁止" in system or "禁止三源及以上" in system
    assert "（…同样的“啊”连续共 12 条…）" not in system
    assert "1,2|1.8|good morning|你好" not in system
    assert "分行重试" not in system
    assert "|high|" in system and "|median|" in system and "|low|" in system
    # No guide columns (duration|gap|char_count) in the output.
    assert "char_count" not in system
    for line in system.splitlines():
        if line.startswith(("sub|", "insert|")):
            assert len(line.split("|")) == 7, line
    assert "不要为了保持逐行语义对齐而强行合并成长字幕" in system
    assert r"\n" not in system
    assert "harness 代为检索，结果放在" in system
    assert "<search_results>" in system
    assert "google_search" not in system
    assert "<next_advice>" in system
    assert "task_update_feedback" not in system
    assert "特殊翻译风格要求" not in system
    assert "主播正在玩游戏。" not in system

    user = messages[1]["content"]
    assert user.index("通用背景知识和术语") < user.index("本窗口待处理 ASR")
    assert "<search_results>" not in user  # search results arrive in the fill turn
    assert "<pre_round_notes>" not in user  # empty in the task flow: not shown (cold read P044)
    assert "此前所有窗口的累积建议" in user
    assert "type|position|start|corrected_text|translation|conf|note" in user
    assert "char_count" not in user
    assert "共有 2 条字幕" in user
    assert "<singles>" not in user
    assert "不要输出其他文字" in user
    assert "translation 为中文语序小幅前后错位" in user or "允许 translation 为中文语序小幅前后错位" in user
    assert "主播正在玩游戏。" in user
    assert "角色名：小明" in user
    assert "本窗口在打 BOSS。" in user
    assert "BOSS 名固定译为「王」。" in user
    assert "<next_advice>" in user
    assert "本窗口易忘要求" in user
    assert '"current_asr_csv"' not in user
    assert '"budget"' not in user
    assert '"chunk_id"' not in user
    assert "overlap_source_ids" not in user
    # First window: the direct preceding block exists but is empty.
    assert _direct_input_block(user, "preceding_context") == ""
    asr = _direct_input_block(user, "asr_result")
    assert "\\n2|" not in asr
    # CSV local times are relative to the window's audio clip start.
    window = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    assert asr.splitlines()[0] == "local_id|start|duration|gap|text"
    first_line = asr.splitlines()[1]
    expected_local_start = window.segments[0].start - window.clip_start
    assert first_line.split("|")[1] == f"{expected_local_start:.1f}"


def test_correction_prompt_renders_preceding_context_with_negative_times() -> None:
    from finesub.llm.chunking import SubtitleWindow

    base = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    preceding = [
        SubtitleSegment("p1", base.clip_start - 12.4, base.clip_start - 11.1, "前文一"),
        SubtitleSegment("p2", base.clip_start - 10.6, base.clip_start - 9.6, "前文二"),
    ]
    window = SubtitleWindow(
        chunk_id=base.chunk_id,
        segments=base.segments,
        overlap_segments=base.overlap_segments,
        boundary_reason=base.boundary_reason,
        budget=base.budget,
        clip_start=base.clip_start,
        clip_end=base.clip_end,
        preceding_segments=preceding,
        length_scale=1.0,
    )

    messages = build_correction_csv_messages(window=window)
    user = messages[1]["content"]

    block = _direct_input_block(user, "preceding_context")
    assert block.splitlines()[0] == "-1|-12.4|1.3||前文一"
    assert "0|-10.6|1.0|0.5|前文二" in block
    # Preceding ids never leak into the translatable input CSV.
    assert "p1" not in _direct_input_block(user, "asr_result")


def test_correction_prompt_injects_window_notes_when_provided() -> None:
    window = dataclasses.replace(plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0], length_scale=1.0)
    messages = build_correction_csv_messages(
        window=window,
        window_notes="本窗口疑似在打BOSS，BOSS名待定。",
        # session replay freezes both into the opening message
        search_results="冻结的搜索结果",
    )
    user = messages[1]["content"]

    assert "本窗口疑似在打BOSS，BOSS名待定。" in user
    assert user.index("<pre_round_notes>") < user.index("<search_results>")


def test_correction_opening_has_no_search_block_without_results() -> None:
    window = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    user = build_correction_csv_messages(window=window)[1]["content"]

    assert "<search_results>" not in user


def test_correction_prompt_injects_extra_style_when_provided() -> None:
    window = dataclasses.replace(plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0], length_scale=1.0)
    messages = build_correction_csv_messages(
        window=window,
        extra_style="所有语气词保留日语原文。",
    )
    system = messages[0]["content"]

    assert "特殊翻译风格要求" in system
    assert "所有语气词保留日语原文。" in system


def test_correction_prompt_can_request_task_update_feedback() -> None:
    window = dataclasses.replace(plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0], length_scale=1.0)
    messages = build_correction_csv_messages(
        window=window,
        task_update_feedback=True,
    )
    system = messages[0]["content"]

    assert "任务反馈采集" in system
    assert "<task_update_feedback>" in system
    assert "knowledge_hints" in system
    assert "new_entry|replace_section|append_lines" in system
    assert "本窗口局部字幕序号" in system
    assert "harness 会映射回稳定源字幕行号" in system
    assert "不要提出翻译风格、prompt 或 harness 修改建议" in system


def test_the_feedback_block_is_asked_for_only_when_collected() -> None:
    """Cold read P025/P050/P008: with feedback collection on, the closing
    reminder (serial and parallel), the fill turn's delivery list and the
    search rules name `<task_update_feedback>`; off, none of them does, and
    a parallel window has no trailing reminder at all."""

    from finesub.llm.request_prompts import build_correction_fill_message
    from finesub.llm.routing.profiles import resolve_profile

    window = dataclasses.replace(plan_correction_windows(_segments(), counter=FakeTokenCounter())[0], length_scale=1.0)
    serial = resolve_profile("audio", "local", "quality", "serial")
    parallel = resolve_profile("audio", "local", "quality", "parallel")
    on_system, on_user = (m["content"] for m in build_correction_csv_messages(
        window=window, profile=serial, task_update_feedback=True))
    assert "，最后是 `<task_update_feedback>`" in on_user
    assert "knowledge_hints 里另记一条" in on_system
    off_system, off_user = (m["content"] for m in build_correction_csv_messages(window=window, profile=serial))
    assert "task_update_feedback" not in off_user and "knowledge_hints" not in off_system

    parallel_on = build_correction_csv_messages(window=window, profile=parallel, task_update_feedback=True)[1]["content"]
    parallel_off = build_correction_csv_messages(window=window, profile=parallel)[1]["content"]
    assert "最后附上 `<task_update_feedback>`" in parallel_on
    assert "最后附上" not in parallel_off and "<next_advice>" not in parallel_off

    assert "<task_update_feedback>" in build_correction_fill_message(
        search_results="r", requested_search=True, task_update_feedback=True)["content"]
    assert "task_update_feedback" not in build_correction_fill_message(
        search_results="r", requested_search=True)["content"]


def test_the_text_only_grouped_prompt_has_no_audio_gate_for_parentheticals() -> None:
    """Cold read P019 in three passes' pass 2: with no audio the hallucination
    section does not make audio the test for a parenthetical."""

    from finesub.llm.prompt_compose import compose_correction_system
    from finesub.llm.routing.profiles import resolve_profile

    for grouped in (False, True):
        system = compose_correction_system(resolve_profile("text", "none", "quality", "serial"), grouped=grouped, length_scale=1.0)
        assert "括注只属于音频" not in system


def test_prompt_templates_are_loaded_from_src_package() -> None:
    assert PROMPT_TEMPLATE_DIR.name == "prompt_templates"
    assert PROMPT_TEMPLATE_DIR.parent.name == "llm"
    expected = {
        "correction_main_v1.md",
        "correction_user_nosingles_v1.md",
        "fragment_corr_role_audio_v1.md",
        "fragment_corr_role_text_v1.md",
        "fragment_corr_role_video_v1.md",
        "fragment_output_contract_nosingles_v1.md",
        "fragment_hallucination_v1.md",
        "fragment_merge_rules_basic_v1.md",
        "fragment_translated_common_noaux_v1.md",
        "fragment_native_search_v1.md",
        "fragment_effort_low_v1.md",
        "fragment_effort_deep_v1.md",
        "fragment_search_queries_output_v1.md",
        "fragment_query_style_v1.md",
        "fragment_search_results_usage_v1.md",
        "knowledge_update_artifacts_only_v1.md",
        "knowledge_update_refined_v1.md",
        "knowledge_update_user_v1.md",
        "fragment_knowledge_structure_v1.md",
        "fragment_knowledge_output_v1.md",
        "fragment_knowledge_update_inputs_v1.md",
        "fragment_task_feedback_schema_v3.md",
        "correction_task_update_feedback_v2.md",
        "research_task_feedback_v1.md",
    }

    assert expected.issubset({path.name for path in PROMPT_TEMPLATE_DIR.glob("*.md")})


def test_knowledge_update_artifacts_only_prompt_never_mentions_mistakes() -> None:
    messages = build_knowledge_update_messages(
        refined=False,
        task_summary="测试任务",
        window_packs="--- window 0001 [0.0s – 10.0s] ---\n<raw_csv>\n1|0.0|1.0|0.0|a\n</raw_csv>",
        general_context='{"global_summary":"摘要"}',
        research_feedback='{"knowledge_hints":[]}',
        kb_entries="### streamer/星野灯\n档案内容",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "无精修模式" in system
    assert "streamer" in system and "common" in system
    assert "append_lines" in system and "update" in system
    assert "retire_entry" in system and "create_entry" in system
    assert "edit_lines" not in system and "replace_section" not in system
    assert "<knowledge_proposals>" in system
    assert "宁缺毋滥" in system
    # Design G: the artifacts_only prompt must not define/mention the mistake
    # block at all (the harness ignores a stray one and never applies it).
    assert "mistake_proposals" not in system
    assert "add_mistake" not in system and "add_mistake" not in user
    # harness_notes has been removed from knowledge update; prompt iteration
    # is now fully owned by session_replay.
    assert "<harness_notes>" not in system
    assert "--- window 0001" in user
    assert "星野灯" in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user


def test_knowledge_update_refined_prompt_covers_style_and_noise() -> None:
    """The refined variant proposes STYLE conventions (plan §3 step 3): the
    mistake ledger and its second output block are gone, so a style
    convention is an ordinary entry update travelling in the one block."""

    messages = build_knowledge_update_messages(
        refined=True,
        task_summary="测试任务",
        window_packs="--- window 0001 ---\n<refined_csv>\n0.0|1.0|你好\n</refined_csv>",
        chunk_index=2,
        multi_chunk=True,
        window_range="0003–0004",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "精修对照模式" in system
    # one output block, and the style rules live in it
    assert "<mistake_proposals>" not in system and "add_mistake" not in system
    assert "翻译风格条目（`style` 类别，只用句柄改，不填 `category`）" in system
    assert "只动本次注入的那个 style 条目里的行" in system  # the retire scope (§2.4)
    assert "不要为此新建条目" in system  # a style is created by a human, not here
    assert "有且仅有一个 `<knowledge_proposals>` 块" in user
    # harness_notes removed; prompt iteration owned by session_replay.
    assert "<harness_notes>" not in system
    # Refined-noise disclosure (design J): annotations, split/merge, offsets.
    assert "非音频内容" in system
    assert "明确对应" in system
    assert "已按开始时间重排" in system
    # Chunk notice (multi-chunk, no total stated).
    assert "第 2 块" in user
    assert "0003–0004" in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user
    assert "<common_mistakes>" not in system


def test_correction_prompt_injects_the_style_block() -> None:
    """The slot that used to carry the curated mistake list now carries the
    style entry the run named (`docs/knowledge.md`, 「style 类别」)."""

    window = dataclasses.replace(plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0], length_scale=1.0)
    with_block = build_correction_csv_messages(
        window=window,
        style_block='本任务的翻译风格约定：\n<style name="某字幕组">\n[自称转换] 用「我」\n</style>',
    )
    without_block = build_correction_csv_messages(window=window)

    assert "[自称转换] 用「我」" in with_block[0]["content"]
    assert "翻译风格约定" not in without_block[0]["content"]
    assert "$style_block" not in without_block[0]["content"]


def test_write_prompt_artifacts_writes_plan_and_prompts(tmp_path) -> None:
    artifacts = {
        "model_limits": {"output_limit": 1},
        "research_messages": [
            [
                {"role": "system", "content": "准备系统"},
                {"role": "user", "content": "准备用户"},
            ],
        ],
        "correction_messages": [
            [
                {"role": "system", "content": "纠错系统"},
                {"role": "user", "content": "纠错用户"},
            ],
        ],
        "correction_basic_example_messages": [
            {"role": "system", "content": "basic纠错系统"},
            {"role": "user", "content": "basic纠错用户"},
        ],
    }

    write_prompt_artifacts(artifacts, tmp_path)

    assert (tmp_path / "plan.json").read_text(encoding="utf-8")
    assert "准备系统" not in (tmp_path / "plan.json").read_text(encoding="utf-8")
    assert "准备用户" in (tmp_path / "research-prepare.txt").read_text(encoding="utf-8")
    assert (tmp_path / "correction-0001.txt").read_text(encoding="utf-8")
    assert "basic纠错系统" in (
        tmp_path / "correction-0001-basic-tier.txt"
    ).read_text(encoding="utf-8")


def test_prompt_artifacts_carry_the_style_block(tmp_path) -> None:
    """Dry run is the product's default mode, so its artifact has to be the
    prompt the run would really send. With style on by default, a resolution
    that happened only on the execution path made the two differ.

    Rendered, not inspected: a source-level guard stays green while the
    rendering is broken (review 2026-09-02)."""

    import inspect
    import json

    from finesub.llm import correction_translation
    from finesub.llm.knowledge.maintain import main as maintain_main
    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.prompt_artifacts import build_prompt_artifacts

    root = tmp_path / "knowledge"
    assert maintain_main(["--root", str(root), "new", "style", "default_style",
                          "--intro", "默认口味"]) == 0
    repo = KnowledgeRepo.open(root)
    edit_subject(repo, repo.subjects("style")[0], (
        "# default_style\n\n默认口味\n\n## 档案\n\n- [本名] default_style\n\n"
        "## 约定\n\n- [自称转换] 说话者用自己名字自指时改用「我」\n"
    ))

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    artifacts = build_prompt_artifacts(
        length_scale=STANDARD_SCALE,
        stable_json=stable_json,
        knowledge_root=root,
        counter=FakeTokenCounter(),
        style_names=("default_style",),
    )
    system = artifacts["correction_messages"][0][0]["content"]
    assert '<style name="default_style">' in system
    assert "[自称转换] 说话者用自己名字自指时改用「我」" in system

    # ...and the task resolves the selection BEFORE it renders the plan, so the
    # plan and the run get the same names
    source = inspect.getsource(correction_translation._run_task)
    assert source.index("style_selection = _task_style_selection(") < source.index(
        "build_prompt_artifacts("
    )


def test_the_dry_run_shows_the_preparation_prompt_this_run_would_send(
    tmp_path, monkeypatch
) -> None:
    """`build_prompt_artifacts` promises the prompt the run sends. The window
    half branched on the strategy; the preparation half did not -- and the
    production default (an agent preset plus `--media audio`) is exactly the
    split where they differ: a clip makes the window batched while the
    text-only preparation task can be agentic. A user reading the dry run then
    saw "ask for a batch and wait for the fill" for a task that really fetches
    for itself."""

    import json

    from finesub.llm import prompt_artifacts
    from finesub.llm.routing.profiles import resolve_profile

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    strategies = {"research": "agentic", "correction": "batched"}
    monkeypatch.setattr(
        prompt_artifacts,
        "resolve_execution_strategy",
        lambda profile, **kw: (strategies[kw.get("task_group", "correction")], "stub"),
    )
    artifacts = prompt_artifacts.build_prompt_artifacts(
        length_scale=STANDARD_SCALE,
        stable_json=stable_json,
        counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_indexes={},
        knowledge_enabled=False,
    )

    prepare_system = artifacts["research_messages"][0][0]["content"]
    assert "由你自行取得" in prepare_system
    assert "获取缺失输入（本任务没有回填轮" in prepare_system or "本任务没有回填轮" in prepare_system
    # ...while the window, which carries the clip, still shows the batched shape.
    window_system = artifacts["correction_messages"][0][0]["content"]
    assert "获取缺失输入" in window_system and "由你自行取得" not in window_system



def test_dry_run_window_prompt_follows_the_correction_cell_variant(tmp_path) -> None:
    """The run sends the cell's variant (`RoleClient.complete`); the dry run used to render the tier default, so
    an efficiency task showed capableC instead of its cell's basicA."""

    import json

    from finesub.llm import prompt_artifacts
    from finesub.llm.output_protocol import correction_output_header
    from finesub.llm.prompt_variants import resolve_variant
    from finesub.llm.routing.profiles import resolve_profile

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    shown = {}
    for difficulty in ("quality", "efficiency"):
        artifacts = prompt_artifacts.build_prompt_artifacts(
            length_scale=STANDARD_SCALE,
            stable_json=stable_json,
            counter=FakeTokenCounter(),
            profile=resolve_profile("text", "none", difficulty),
            knowledge_indexes={},
            knowledge_enabled=False,
        )
        shown[difficulty] = artifacts["correction_messages"][0][0]["content"]
    capable, basic = (correction_output_header(resolve_variant(name)) for name in ("capableC", "basicA"))
    assert capable in shown["quality"] and basic not in shown["quality"]
    assert basic in shown["efficiency"] and capable not in shown["efficiency"]


def test_dry_run_window_prompt_follows_the_first_target_s_override(tmp_path, monkeypatch) -> None:
    """A model-group entry may override the cell's variant per target; the run picks per candidate, so the dry
    run shows what the cell's first model is sent."""

    import json

    from finesub.llm import prompt_artifacts
    from finesub.llm.output_protocol import correction_output_header
    from finesub.llm.prompt_variants import resolve_variant
    from finesub.llm.routing import config as routing_config
    from finesub.llm.routing.profiles import resolve_profile

    real = routing_config.role_config_for

    def overridden(*args, **kwargs):
        config = real(*args, **kwargs)
        return dataclass_replace(config, variant_overrides={config.endpoint_chain[0].target_id: "basicA"})

    monkeypatch.setattr(routing_config, "role_config_for", overridden)
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    artifacts = prompt_artifacts.build_prompt_artifacts(
        length_scale=STANDARD_SCALE,
        stable_json=stable_json,
        counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
        knowledge_indexes={},
        knowledge_enabled=False,
    )
    shown = artifacts["correction_messages"][0][0]["content"]
    assert correction_output_header(resolve_variant("basicA")) in shown

@pytest.mark.parametrize("strategy", ["batched", "agentic"])
def test_dry_run_shows_the_none_window_the_run_sends(tmp_path, monkeypatch, strategy) -> None:
    """window-kb-request-channel plan, acceptance 7a: the dry run (the product
    default) and the run read the same `task_graph.window_flow`, so a `none`
    window with a readable base shows its entries-only channel -- or, agentic,
    the knowledge-only tool note -- and the preparation task its pruning keep
    at the request-capable cap."""

    import json

    from finesub.llm import prompt_artifacts
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages import task_graph

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        task_graph, "resolve_execution_strategy", lambda profile, **kw: (strategy, "stub")
    )
    monkeypatch.setattr(
        prompt_artifacts, "resolve_execution_strategy", lambda profile, **kw: ("batched", "stub")
    )
    artifacts = prompt_artifacts.build_prompt_artifacts(
        length_scale=STANDARD_SCALE,
        stable_json=stable_json,
        counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
        knowledge_indexes={"streamer": "- 主播A | | 测试主播", "common": ""},
    )
    window_system = artifacts["correction_messages"][0][0]["content"]
    assert "搜索 query 输出规则" not in window_system
    assert "`web_search`" not in window_system
    if strategy == "batched":
        assert "获取缺失输入（同一任务内" in window_system and "<requested_entries>" in window_system
    else:
        assert "由你自行取得" in window_system and "`kb_search`" in window_system
    assert "<keep_entries>" in window_system
    prepare_system = artifacts["research_messages"][0][0]["content"]
    assert "<keep_entries>" in prepare_system and "单独上限 8 条" in prepare_system
