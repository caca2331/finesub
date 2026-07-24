from __future__ import annotations

import re

from llm.chunking import SubtitleSegment, plan_correction_windows
from llm.prompts import (
    ContextPack,
    PROMPT_TEMPLATE_DIR,
    PROMPT_VERSION,
    build_correction_csv_messages,
    build_correction_query_messages,
    build_knowledge_update_messages,
    build_research_round1_messages,
    build_research_round2_messages,
    build_search_loop_messages,
)
from llm.prompt_artifacts import write_prompt_artifacts
from llm.subtitle_metrics import weighted_char_count


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


def test_research_round1_prompt_covers_indices_and_search() -> None:
    messages = build_research_round1_messages(
        transcript="--- window 0001 ---\n1|こんにちは\n",
        extra_info="来源：https://example.test/video",
        streamer_index="- 主播A | エーちゃん | 测试主播",
        common_index="- 游戏B [游戏] | Game B | 测试游戏",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "第一轮" in system
    assert "requested_entries" in system
    assert "<keep_entries>" in system
    assert "<analysis_notes>" in system
    assert "<search_queries>" in system
    assert "最多 8 条" in system
    assert "按重要性从高到低排列" in system
    assert "单独上限 8 条" in system
    assert "共享 12 条总上限" in system
    assert "keep 优先于 requested" in system
    assert "没有联网搜索能力" in system
    assert "不逐句翻译" in system
    assert "音似" in system
    assert "不能因为预计输出较长" in system
    assert "<research_contract>" not in system
    assert "<streamer_index>" in user
    assert "主播A" in user
    assert "游戏B" in user
    assert "https://example.test/video" in user
    assert "--- window 0001 ---" in user


def test_research_round1_prompt_swaps_in_contract_fragment_for_multi_round() -> None:
    messages = build_research_round1_messages(
        transcript="1|こんにちは\n",
        max_search_queries=10,
        use_search_contract=True,
    )
    system = messages[0]["content"]

    assert "<research_contract>" in system
    assert "<analysis_notes>" in system
    assert '"priority"' in system
    assert "out_of_scope" in system
    assert "第 0 轮" in system
    assert "最多 $max_queries 条" not in system
    assert "最多 10 条" in system


def test_research_round2_prompt_injects_entries_and_search_results() -> None:
    messages = build_research_round2_messages(
        transcript="--- window 0001 ---\n1|こんにちは\n",
        extra_info="",
        entry_details_text="# 主播A\n\n## 档案\n\n关西腔。",
        search_results="--- query: 游戏B ---\nprovider: tavily\n- wiki (https://example.test)\n  恐怖游戏",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "第二轮" in system
    assert "general_context" in system
    assert "window_contexts" in system
    assert "<context_pack>" in system
    assert "不能再发起新的搜索" in system
    assert "不逐句翻译" in system
    assert "不逐句判断" in system
    assert "google_search" not in system
    assert "# 主播A" in user
    assert "关西腔" in user
    assert "<search_results>" in user
    assert "恐怖游戏" in user
    assert "<round1_notes>" in user


def test_research_round2_prompt_swaps_in_evidence_pack_fragment() -> None:
    messages = build_research_round2_messages(
        transcript="1|こんにちは\n",
        round1_notes="主播在玩游戏B（待定）。",
        search_results="## 结论\nF1 confirmed：BOSS 官方名「王」",
        use_evidence_pack=True,
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "Evidence Pack" in system
    assert "Research Contract" in system
    assert "不能再发起新的搜索" in system
    assert "主播在玩游戏B（待定）。" in user
    assert "BOSS 官方名「王」" in user


def test_search_loop_prompt_covers_contract_progress_and_decision() -> None:
    messages = build_search_loop_messages(
        round_index=1,
        max_rounds=3,
        is_final_round=False,
        background="主播在玩游戏B（待定）。",
        contract_json='{"goal": "查证游戏B"}',
        executed_queries=["游戏B 剧情"],
        progress_log="## 搜索轮 0\nF1: partial 候选名",
        search_results="--- query: 游戏B BOSS ---\n- wiki",
        previous_requested_entries=["主播A"],
        previous_kept_entries=["游戏B"],
        previous_contract_json='{"goal": "发起时查证游戏B"}',
        previous_search_queries=["F1|游戏B BOSS >> 查官方名"],
        previous_extract_urls=["https://example.test/wiki >> 查角色段落"],
        followup_query_cap=4,
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "<progress_update>" in system
    assert "<evidence_pack>" in system
    assert "<search_queries>" in system
    assert "<extract_urls>" in system
    assert "one-shot" in system
    assert "银鸦" in system
    assert "有相关性的资料宁多勿漏" in system
    assert "priority" in system
    assert "仍然可以继续查" in system
    assert "不得超过 4 条" in system
    assert "第 1 轮" in user
    assert "查证游戏B" in user
    assert "游戏B 剧情" in user
    assert "F1: partial 候选名" in user
    assert "<current_research_contract>" in user
    assert "<previous_requested_entries>\n主播A" in user
    assert "<previous_kept_entries>\n游戏B" in user
    assert "发起时查证游戏B" in user
    assert "F1|游戏B BOSS >> 查官方名" in user
    assert "https://example.test/wiki >> 查角色段落" in user
    assert user.index("<previous_search_request>") < user.index("<search_results>")
    assert "本轮必须输出" not in user

    final_messages = build_search_loop_messages(
        round_index=2,
        max_rounds=3,
        is_final_round=True,
        followup_query_cap=4,
    )
    assert "本轮必须输出 `<evidence_pack>` 收尾" in final_messages[1]["content"]


def test_correction_prompt_forbids_omitting_long_singles_or_translated() -> None:
    window = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
    messages = build_correction_csv_messages(window=window)
    combined = "\n".join(str(message["content"]) for message in messages)

    assert "不能因为预计输出较长而省略" in combined
    assert "<singles>" not in combined
    assert "translated 必须给出完整终稿" in combined
    assert "严格保持标签、header、九列字段与顺序" in combined


def test_correction_query_prompt_requests_search_queries_only() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_query_messages(
        window=window,
        context_pack=ContextPack(
            general_context={"global_summary": "主播正在玩游戏。"},
            window_contexts={window.chunk_id: "本窗口在打 BOSS。"},
        ),
        audio_file_label="clip.wav",
        previous_advice="BOSS 名固定译为「王」。",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "搜索请求代理" in system
    assert "<search_queries>" in system
    assert "<window_notes>" in system
    assert "待定" in system
    assert "最多 8 条" in system
    assert "不纠错、不翻译" in system
    assert "此前所有窗口的累积建议" in user
    assert "主播正在玩游戏。" in user
    assert "本窗口在打 BOSS。" in user
    assert "BOSS 名固定译为「王」。" in user
    assert "<asr_result>" in user
    asr = _direct_input_block(user, "asr_result")
    assert asr.splitlines()[0] == "source_id|start|duration|gap|text"
    assert asr.splitlines()[1].startswith("1|")
    assert '"current_asr_csv"' not in user
    assert "\\n2|" not in asr


def test_context_pack_window_lookup_falls_back_to_parent() -> None:
    pack = ContextPack(
        general_context={"global_summary": "主播正在玩游戏。"},
        window_contexts={"0001": "窗口一背景", "0002": "窗口二背景"},
    )

    assert pack.window_context_for("0001") == "窗口一背景"
    assert pack.window_context_for("0001-a") == "窗口一背景"
    assert pack.window_context_for("0001-a-b") == "窗口一背景"
    assert pack.window_context_for("0003") == ""


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

    assert pack.window_contexts == {"0001": "背景一", "0002": "背景二"}


def test_capable_a_prompt_locks_legacy_singles_and_style_rules() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        context_pack=ContextPack(
            general_context={"global_summary": "主播正在玩游戏。", "must": ["角色名：小明"]},
            window_contexts={window.chunk_id: "本窗口在打 BOSS。"},
        ),
        audio_file_label="clip.wav",
        previous_advice="BOSS 名固定译为「王」。",
        variant="capableA",
    )
    system = messages[0]["content"]

    assert "<singles>...</singles>" in system or "<singles>" in system
    assert "<translated>...</translated>" in system or "<translated>" in system
    assert "不要 Markdown 代码块" in system
    assert "不要无标签散文" in system
    assert "一次性完整输出" in system
    assert "<asr_result>" in system
    assert "Haha -> 哈哈" in system
    assert "やばい" in system
    assert "酱" in system
    assert "音似的假名" in system
    assert "原始音频" in system
    assert "不擅自添加原文没有的人称" in system
    assert "保留原文语气、情感和句式" in system
    assert "不可分语义单元" in system
    assert "软门槛是默认边界" in system
    assert "微调语序" in system
    assert "短片段合并策略" in system
    assert "最多合并两个连续" in system
    assert "harness 不会提供合并候选" in system
    assert "<singles>" in system
    assert "<translated>" in system
    assert "不要写 `plan|" in system or "不要输出 `plan|" in system
    assert "\nplan|" not in system  # no plan example rows (v34)
    assert "两阶段 oneshot" in system or "一一对应" in system
    assert "宜与前一句合并" in system
    assert "视情况可向后合并" in system
    assert "宜独立" in system
    assert "宜丢弃" in system
    assert "五选一取舍结论" in system
    assert "五选一邻接结论" not in system
    assert "邻接预判" in system or "因为…" in system
    # Oneshot (not the short insert example) must use the forward-looking verdict.
    oneshot = system.split("两阶段 oneshot", 1)[1]
    oneshot_singles = oneshot.split("<singles>", 1)[1].split("</singles>", 1)[0]
    oneshot_translated = oneshot.split("<translated>", 1)[1].split(
        "</translated>", 1
    )[0]
    assert "视情况可向后合并" in oneshot_singles
    assert "（待与后" not in oneshot_singles
    assert "等与后句" not in oneshot_singles
    assert "与后句合流" not in oneshot_singles
    assert "莉奈娅" in system or "リンネ" in system
    assert "ルビルビルビルビ" in system
    assert "sub|1|" in system and "sub|42|" in system
    assert len([line for line in oneshot_singles.splitlines() if line.startswith("sub|")]) == 42
    assert all(
        len(line.split("|")) == 9
        for line in oneshot_singles.splitlines()
        if line.startswith("sub|")
    )
    verdicts = (
        "宜与前一句合并",
        "宜与前两句合并",
        "视情况可向后合并",
        "宜独立",
        "宜丢弃",
    )
    assert all(
        line.endswith(verdicts)
        for line in oneshot_singles.splitlines()
        if line.startswith("sub|")
    )
    assert all(
        len(line.split("|")) == 9
        for line in oneshot_translated.splitlines()
        if line.startswith("sub|")
    )
    # 新上游预合并后三源例外不再有 oneshot 实例（契约仍规定 ≤3 连续源）；
    # 主 oneshot 的唯一多源行是 10,11 的填充词接续。
    assert "sub|10,11|" in system
    assert "sub|29,30,31|" not in system
    assert "三源及以上禁止" in system or "禁止三源及以上" in system
    assert "（…同样的“啊”连续共 12 条…）" not in system
    assert "1,2|1.8|good morning|你好" not in system
    assert "分行重试" not in system
    assert "|high|" in system and "|median|" in system and "|low|" in system
    assert "char_count" in system
    assert "拉丁字母、数字和标点每个计 0.5" in system
    for line in system.splitlines():
        if not line.startswith(("sub|", "insert|")):
            continue
        fields = line.split("|")
        assert len(fields) == 9, line
        assert float(fields[7]) == weighted_char_count(fields[5]), line
    assert "缩窄范围" in system
    assert "不要为了保持逐行语义对齐而强行合并成长字幕" in system
    assert r"\n" not in system
    assert "注入的搜索结果" in system
    assert "<search_results>" in system
    assert "google_search" not in system
    assert "<next_advice>" in system
    assert "task_update_feedback" not in system
    assert "特殊翻译风格要求" not in system
    assert "主播正在玩游戏。" not in system

    user = messages[1]["content"]
    assert user.index("通用背景知识和术语") < user.index("本窗口待处理 ASR")
    assert "<search_results>" in user
    assert "<pre_round_notes>" in user
    assert "此前所有窗口的累积建议" in user
    assert "type|position|duration|gap|corrected_text|translation|conf|char_count|note" in user
    assert "共有 2 条字幕" in user
    assert "恰好有 **2 条字幕行**" in user
    assert "<singles>" in user
    assert "不要输出 `plan|" in user or "plan|" in user  # forbid plan in recap
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
    window = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
    assert asr.splitlines()[0] == "source_id|start|duration|gap|text"
    first_line = asr.splitlines()[1]
    expected_local_start = window.segments[0].start - window.clip_start
    assert first_line.split("|")[1] == f"{expected_local_start:.1f}"


def test_correction_prompt_renders_preceding_context_with_negative_times() -> None:
    from llm.chunking import SubtitleWindow

    base = plan_correction_windows(_segments(), counter=FakeTokenCounter())[0]
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
    )

    messages = build_correction_csv_messages(window=window)
    user = messages[1]["content"]

    block = _direct_input_block(user, "preceding_context")
    assert block.splitlines()[0] == "p1|-12.4|1.3|0.5|前文一"
    assert "p2|-10.6|1.0|0.0|前文二" in block
    # Preceding ids never leak into the translatable input CSV.
    assert "p1" not in _direct_input_block(user, "asr_result")


def test_correction_prompt_injects_query_round_notes_when_provided() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        query_round_notes="本窗口疑似在打BOSS，BOSS名待定。",
    )
    user = messages[1]["content"]

    assert "本窗口疑似在打BOSS，BOSS名待定。" in user
    assert user.index("<pre_round_notes>") < user.index("<search_results>")


def test_correction_prompt_injects_extra_style_when_provided() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        extra_style="所有语气词保留日语原文。",
    )
    system = messages[0]["content"]

    assert "特殊翻译风格要求" in system
    assert "所有语气词保留日语原文。" in system


def test_correction_prompt_can_request_task_update_feedback() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    messages = build_correction_csv_messages(
        window=window,
        task_update_feedback=True,
    )
    system = messages[0]["content"]

    assert "任务反馈采集" in system
    assert "<task_update_feedback>" in system
    assert "knowledge_hints" in system
    assert "new_entry|replace_section|append_lines" in system
    assert "不要提出翻译风格、prompt 或 harness 修改建议" in system


def test_prompt_templates_are_loaded_from_src_package() -> None:
    assert PROMPT_TEMPLATE_DIR.name == "prompt_templates"
    assert PROMPT_TEMPLATE_DIR.parent.name == "llm"
    expected = {
        "research_round1_v1.md",
        "research_round1_user_v1.md",
        "research_round2_v1.md",
        "research_round2_user_v1.md",
        "correction_main_v1.md",
        "correction_user_v2.md",
        "correction_query_v2.md",
        "correction_query_user_v1.md",
        "fast_round1_v1.md",
        "fast_round1_user_v1.md",
        "fragment_corr_role_audio_v1.md",
        "fragment_corr_role_text_v1.md",
        "fragment_corr_role_video_v1.md",
        "fragment_output_contract_v1.md",
        "fragment_hallucination_v1.md",
        "fragment_examples_merge_v1.md",
        "fragment_examples_merge_basic_v1.md",
        "fragment_merge_rules_basic_v1.md",
        "fragment_translated_common_v1.md",
        "fragment_native_search_v1.md",
        "fragment_effort_low_v1.md",
        "fragment_effort_deep_v1.md",
        "fragment_search_queries_output_v1.md",
        "fragment_query_style_v1.md",
        "fragment_search_results_usage_v1.md",
        "fragment_search_contract_output_v1.md",
        "fragment_evidence_pack_usage_v1.md",
        "search_loop_v1.md",
        "search_loop_user_v1.md",
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
    assert "append_lines" in system and "edit_lines" in system
    assert "replace_section" in system and "create_entry" in system
    assert "<knowledge_proposals>" in system
    assert "宁缺毋滥" in system
    # Design G: the artifacts_only prompt must not define/mention the mistake
    # block at all (the harness ignores a stray one and never applies it).
    assert "mistake_proposals" not in system
    assert "add_mistake" not in system
    # harness_notes has been removed from knowledge update; prompt iteration
    # is now fully owned by session_replay.
    assert "<harness_notes>" not in system
    assert "--- window 0001" in user
    assert "星野灯" in user
    assert PROMPT_VERSION in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user


def test_knowledge_update_refined_prompt_covers_mistakes_and_noise() -> None:
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
    assert "<mistake_proposals>" in system
    assert "add_mistake" in system
    # 精选 is curated manually now: the schema must not define set_featured
    # (the prompt only mentions it to forbid it).
    assert '"op":"set_featured"' not in system
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


def test_correction_prompt_injects_common_mistakes_block() -> None:
    window = plan_correction_windows(
        _segments(),
        counter=FakeTokenCounter(),
    )[0]
    with_block = build_correction_csv_messages(
        window=window,
        common_mistakes_block="常见翻译错误对照：\n1. 原文「run」曾被误译为「跑步」",
    )
    without_block = build_correction_csv_messages(window=window)

    assert "常见翻译错误对照" in with_block[0]["content"]
    assert "常见翻译错误对照" not in without_block[0]["content"]
    assert "$common_mistakes_block" not in without_block[0]["content"]


def test_write_prompt_artifacts_writes_plan_and_prompts(tmp_path) -> None:
    artifacts = {
        "model_limits": {"output_limit": 1},
        "research_messages": [
            [
                {"role": "system", "content": "研究一系统"},
                {"role": "user", "content": "研究一用户"},
            ],
            [
                {"role": "system", "content": "研究二系统"},
                {"role": "user", "content": "研究二用户"},
            ],
        ],
        "correction_query_messages": [
            [
                {"role": "system", "content": "查询系统"},
                {"role": "user", "content": "查询用户"},
            ],
        ],
        "search_loop_example_messages": [
            {"role": "system", "content": "loop系统"},
            {"role": "user", "content": "loop用户"},
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
    assert "loop系统" not in (tmp_path / "plan.json").read_text(encoding="utf-8")
    assert (tmp_path / "research-round1.txt").read_text(encoding="utf-8")
    assert (tmp_path / "research-round2.txt").read_text(encoding="utf-8")
    assert "loop用户" in (
        tmp_path / "research-search-loop-example.txt"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "correction-0001-query.txt").read_text(encoding="utf-8")
    assert (tmp_path / "correction-0001.txt").read_text(encoding="utf-8")
    assert "basic纠错系统" in (
        tmp_path / "correction-0001-basic-tier.txt"
    ).read_text(encoding="utf-8")
