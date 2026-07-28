from __future__ import annotations

import re

from llm.chunking import SubtitleSegment, plan_correction_windows
from llm.config import CapabilityTier
from llm.profiles import resolve_profile
from llm.prompt_compose import (
    PROMPT_VERSION,
    compose_correction_query_system,
    compose_correction_system,
    compose_correction_user,
    compose_fast_round1_system,
    load_prompt_template,
)
from llm.prompts import build_fast_round1_messages


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def _window():
    segments = [
        SubtitleSegment("1", 0.0, 1.0, "えっと、あの子ちゃんが来た。"),
        SubtitleSegment("2", 1.2, 2.0, "Haha やばい。"),
    ]
    return plan_correction_windows(segments, counter=FakeTokenCounter())[0]


def test_prompt_version_bumped_to_v65() -> None:
    assert PROMPT_VERSION == "zh-subtitle-correction-csv-v65"


def test_variant_default_matches_tier_and_unknown_raises() -> None:
    import pytest

    from llm.prompt_variants import DEFAULT_VARIANT_FOR_TIER, VARIANTS, resolve_variant

    # The tier still picks a default variant; passing None must equal it.
    assert resolve_variant(None, CapabilityTier.CAPABLE).name == "capableC"
    assert resolve_variant(None, CapabilityTier.BASIC).name == "basicB"
    assert resolve_variant("capableB").name == "capableB"
    assert set(VARIANTS) == {"basicA", "capableB", "capableC", "basicB"}
    assert DEFAULT_VARIANT_FOR_TIER[CapabilityTier.CAPABLE] == "capableC"
    assert DEFAULT_VARIANT_FOR_TIER[CapabilityTier.BASIC] == "basicB"
    with pytest.raises(ValueError):
        resolve_variant("nope")


def test_variant_selection_matches_tier_default_byte_for_byte() -> None:
    # Selecting the variant explicitly must equal the tier-derived default.
    profile = resolve_profile("mm", "high")
    for tier, name in ((CapabilityTier.CAPABLE, "capableC"), (CapabilityTier.BASIC, "basicB")):
        assert compose_correction_system(
            profile, tier=tier
        ) == compose_correction_system(profile, variant=name)


def test_variant_b_drops_the_singles_block() -> None:
    profile = resolve_profile("mm", "high")
    c_sys = compose_correction_system(profile, variant="capableB")
    basic_a_sys = compose_correction_system(profile, variant="basicA")
    c_user = compose_correction_user(
        profile,
        variant="capableB",
        general_context_json="",
        window_context="",
        entry_details="",
        previous_advice="",
        pre_round_notes="",
        search_results="",
        preceding_context_csv="",
        current_asr_csv="1|0|1|0|x|x|high|1|n",
        current_asr_row_count=1,
    )
    # C emits <translated> only and never mentions singles at all (a model that
    # never had a singles stage should not be told about one).
    assert "singles" not in c_sys
    assert "singles" not in c_user
    assert "<translated>" in c_sys
    assert "<singles>" in basic_a_sys


def test_capablec_composes_with_reasoning_rows_and_no_singles() -> None:
    import re

    profile = resolve_profile("mm", "high")
    sys_msg = compose_correction_system(profile, variant="capableC")
    user_msg = compose_correction_user(
        profile,
        variant="capableC",
        general_context_json="",
        window_context="",
        entry_details="",
        previous_advice="",
        pre_round_notes="",
        search_results="",
        preceding_context_csv="",
        current_asr_csv="1|0|1|0|x|x|high|1|n",
        current_asr_row_count=1,
    )
    # Composes cleanly (no unresolved $placeholder leaked from a fragment).
    assert not re.search(r"\$[a-zA-Z_]+", sys_msg), "unresolved placeholder in system"
    assert not re.search(r"\$[a-zA-Z_]+", user_msg), "unresolved placeholder in user"
    # capableC = capableB (no singles) + inter-line reasoning comments.
    assert "singles" not in sys_msg
    assert "# " in sys_msg  # reasoning comments use # prefix
    assert "# " in user_msg


def test_capable_b_and_c_use_hard_and_absolute_threshold_names() -> None:
    profile = resolve_profile("mm", "high")
    for name in ("capableB", "capableC", "basicB"):
        system = compose_correction_system(profile, variant=name)
        assert "20 字/4 秒硬门槛" in system
        assert "36 字/7 秒" in system and "绝对门槛" in system
        assert "软门槛" not in system


def test_only_full_oneshot_examples_carry_csv_headers() -> None:
    profile = resolve_profile("mm", "high")
    input_header = "local_id|start|duration|gap|text"
    output_header = (
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
    )
    basic_output_header = (
        "type|position|start|duration|gap|corrected_text|translation|conf|char_count|note"
    )
    for name in ("basicA", "capableB", "capableC", "basicB"):
        system = compose_correction_system(profile, variant=name)
        asr_blocks = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", system)
        # capableB/basicB use fragment_examples_merge_nosingles_v1.md which has
        # an extra <asr_result> block for the 缩窄合并反例 (needed for start injection).
        expected_input_count = 2 if name in ("capableB", "basicB") else 1
        assert sum(
            block.splitlines()[0] == input_header for block in asr_blocks
        ) == expected_input_count
        for block in asr_blocks:
            local_ids = [
                line.split("|", 1)[0]
                for line in block.splitlines()
                if re.match(r"^\d+\|", line)
            ]
            assert local_ids == [str(index) for index in range(1, len(local_ids) + 1)]

        preceding_blocks = re.findall(
            r"(?ms)^<preceding_context>\n(.*?)^</preceding_context>$", system
        )
        assert [
            line.split("|", 1)[0]
            for line in preceding_blocks[0].splitlines()
            if re.match(r"^-?\d+\|", line)
        ] == ["-1", "0"]

        translated_blocks = re.findall(
            r"(?ms)^<translated>\n(.*?)^</translated>$", system
        )
        expected_output_header = (
            basic_output_header if name in ("basicA", "basicB") else output_header
        )
        expected_count = 1
        assert sum(
            block.splitlines()[0] == expected_output_header
            for block in translated_blocks
        ) == expected_count

        singles_blocks = re.findall(r"(?ms)^<singles>\n(.*?)^</singles>$", system)
        if name == "basicA":
            assert sum(
                block.splitlines()[0] == expected_output_header
                for block in singles_blocks
            ) == expected_count
        else:
            assert not singles_blocks


def test_basic_b_freezes_start_csv_while_capable_b_and_c_revert() -> None:
    profile = resolve_profile("mm", "high")
    start_header = (
        "type|position|start|duration|gap|corrected_text|translation|conf|char_count|note"
    )
    legacy_header = (
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
    )
    basic_b = compose_correction_system(profile, variant="basicB")
    assert start_header in basic_b
    assert "固定 10 列" in basic_b
    for name in ("capableB", "capableC"):
        capable = compose_correction_system(profile, variant=name)
        assert legacy_header in capable
        assert start_header not in capable


def test_capable_c_has_full_43_row_oneshot() -> None:
    profile = resolve_profile("mm", "high")
    system = compose_correction_system(profile, variant="capableC")
    oneshot = system.split("完整示例", 1)[1]
    asr = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", oneshot)[0]
    translated = re.findall(
        r"(?ms)^<translated>\n(.*?)^</translated>$", oneshot
    )[0]
    assert len([line for line in asr.splitlines() if re.match(r"^\d+\|", line)]) == 43
    rows = [
        line for line in translated.splitlines()
        if line.startswith(("sub|", "discard|"))
    ]
    assert any(line.startswith("# ") for line in translated.splitlines())
    assert len(rows) == 41


def test_basic_a_has_full_43_row_oneshot_with_conservative_final_output() -> None:
    system = compose_correction_system(resolve_profile("mm", "high"), variant="basicA")
    oneshot = system.split("两阶段完整 oneshot", 1)[1]
    asr = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", oneshot)[0]
    singles = re.findall(r"(?ms)^<singles>\n(.*?)^</singles>$", oneshot)[0]
    translated = re.findall(r"(?ms)^<translated>\n(.*?)^</translated>$", oneshot)[0]

    assert len([line for line in asr.splitlines() if re.match(r"^\d+\|", line)]) == 43
    assert len([line for line in singles.splitlines() if line.startswith("sub|")]) == 43
    assert len([line for line in translated.splitlines() if line.startswith("sub|")]) == 41
    assert "discard|18|" in translated
    assert "sub|8|" in translated and "sub|9|" in translated
    assert "sub|10|" in translated and "sub|11|" in translated
    assert "sub|8,9|" not in translated
    assert "sub|10,11|" not in translated
    # The only permitted BasicA merge is demonstrated separately as a short,
    # header-free mid-word rejoin example.
    assert "sub|1,2|" in system


def test_correction_system_tier_selects_merge_fragments() -> None:
    char_rule = load_prompt_template("fragment_weighted_char_count_v1.md").strip()
    for route, level in (("mm", "med"), ("text", "low"), ("text", "high")):
        profile = resolve_profile(route, level)
        capable = compose_correction_system(profile, tier=CapabilityTier.CAPABLE)
        basic = compose_correction_system(profile, tier=CapabilityTier.BASIC)
        basic_a = compose_correction_system(profile, variant="basicA")

        # The tier-independent discipline fragment lands in both variants.
        assert "translated 产出纪律" in capable
        assert "translated 产出纪律" in basic
        # Production basic default (basicB) inherits capableB merge rules;
        # basicA remains the conservative 1:1 control.
        assert "至少 2/3 的源片段通常无需合并" in capable
        assert "至少 2/3 的源片段通常无需合并" in basic
        assert "至少 2/3 的源片段通常无需合并" not in basic_a
        assert "保守 1:1 策略" in basic_a
        assert "保守 1:1 策略" not in capable
        assert "保守 1:1 策略" not in basic
        # Merge examples follow the production defaults: capableC / basicB
        # both drop the full-window <singles> pass.
        assert "输入完整 43 条" in capable or "输入 43 条" in capable
        assert "<singles>" not in capable
        assert "<singles>" not in basic
        assert "singles 恰好 43 行" in basic_a
        assert "口播碎片成一句" not in basic_a
        # The weighted char-count algorithm is injected exactly once (via the
        # output contract) — the old merge-rules duplicate is gone.
        assert capable.count(char_rule) == 1
        assert basic.count(char_rule) == 1
        # Default tier is capable.
        assert compose_correction_system(profile) == capable

def test_text_low_system_has_no_audio_insert_or_search() -> None:
    system = compose_correction_system(resolve_profile("text", "low"))

    assert "本次任务没有音频" in system
    assert "窗口内开始时间" in system
    assert "不要输出其他 type 值" in system
    assert "低成本快速翻译" in system
    # Audio-only material must be absent.
    assert "插轴" not in system
    assert "本窗口剪辑音频" not in system
    assert "insert" not in system
    assert "原始音频" not in system
    assert "<search_results>" not in system
    assert "联网搜索（内置工具）" not in system
    # Shared core stays.
    assert "短片段合并策略" in system
    assert "最多合并两个连续" in system
    assert "harness 不会提供合并候选" in system
    assert "<next_advice>" in system
    assert "ご視聴ありがとうございました" in system
    assert "<asr_result>" in system
    assert "$" not in system  # every slot resolved


def test_text_med_and_high_swap_effort_and_native_search() -> None:
    med = compose_correction_system(resolve_profile("text", "med"))
    high = compose_correction_system(resolve_profile("text", "high"))

    assert "允许较深入思考" in med
    assert "低成本快速翻译" not in med
    assert "联网搜索（内置工具）" not in med
    assert "联网搜索（内置工具）" in high
    assert "允许较深入思考" in high
    # text-med gets injected search results usage rules (search agent still runs);
    # text-high uses native search (model's built-in google_search tool) instead.
    assert "注入的搜索结果" in med
    assert "注入的搜索结果" not in high
    native_block = load_prompt_template("fragment_native_search_v1.md").strip()
    assert native_block in high
    assert native_block not in med


def test_mm_low_is_text_modal_with_injected_search() -> None:
    system = compose_correction_system(resolve_profile("mm", "low"))

    assert "本次任务没有音频" in system
    assert "注入的搜索结果" in system
    assert "<search_results>" in system
    assert "插轴" not in system
    assert "思考与速度" not in system


def test_mm_med_is_audio_modal_with_injected_search() -> None:
    system = compose_correction_system(resolve_profile("mm", "med"))

    assert "原始音频" in system
    assert "剪辑内开始时间" in system
    # Insert/插轴 deprecated for all variants (v63+).
    assert "插轴（插入源字幕遗漏的字幕）" not in system
    assert "插轴示例" not in system
    assert "type=insert" not in system
    assert "注入的搜索结果" in system
    assert "视频画面" not in system
    assert "思考与速度" not in system
    assert "$" not in system


def test_mm_high_adds_video_addendum() -> None:
    system = compose_correction_system(resolve_profile("mm", "high"))

    assert "视频画面" in system
    assert "画面上出现、但主播没有说出的文字" in system
    assert "原始音频" in system


def test_evidence_pack_mode_swaps_usage_fragment() -> None:
    normal = compose_correction_system(resolve_profile("mm", "med"))
    evidence = compose_correction_system(
        resolve_profile("mm", "med"), evidence_pack_mode=True
    )

    assert normal != evidence
    assert "Evidence Pack" in evidence


def test_correction_user_reminders_follow_modality() -> None:
    kwargs = dict(
        general_context_json="{}",
        window_context="（无）",
        entry_details="（无）",
        previous_advice="（无）",
        pre_round_notes="（无）",
        search_results="（无）",
        preceding_context_csv="",
        current_asr_csv="1|0.0|1.0|0.0|测试",
        current_asr_row_count=17,
    )
    audio_user = compose_correction_user(resolve_profile("mm", "med"), **kwargs)
    text_user = compose_correction_user(resolve_profile("text", "low"), **kwargs)

    assert "type=insert" not in audio_user
    assert "剪辑音频的 0 秒" in audio_user
    assert "type=insert" not in text_user
    assert "不要输出任何时间戳" in text_user
    assert "不得残留日语假名或助词" in audio_user
    assert "<entry_details>" in audio_user
    assert "<pre_round_notes>" in text_user
    assert "共有 17 条字幕" in audio_user
    assert "共有 **17 条输入字幕**" in text_user


def test_query_round_system_text_variant_drops_audio() -> None:
    audio = compose_correction_query_system(
        resolve_profile("mm", "med"), search_queries_rules="RULES"
    )
    text = compose_correction_query_system(
        resolve_profile("mm", "low"), search_queries_rules="RULES"
    )

    assert "原始音频剪辑" in audio
    assert "结合音频听清" in audio
    assert "原始音频剪辑" not in text
    assert "本次任务没有音频" in text
    assert "发音相似度" in text
    assert "RULES" in text


def test_fast_round1_system_variants_and_messages() -> None:
    audio = compose_fast_round1_system(
        resolve_profile("mm", "med"), search_queries_rules="RULES"
    )
    video = compose_fast_round1_system(
        resolve_profile("mm", "high"), search_queries_rules="RULES"
    )
    text = compose_fast_round1_system(
        resolve_profile("mm", "low"), search_queries_rules="RULES"
    )

    assert "快速模式" in audio
    assert "整段原始音频的剪辑" in audio
    assert "视频画面" in video
    assert "整段原始音频" not in text
    assert "<analysis_notes>" in audio
    assert "<requested_entries>" in audio
    assert "单独上限为 8 条" in audio
    assert "共享 12 条总上限" in audio
    assert "keep 优先于 requested" in audio
    assert "2000 token" in audio

    messages = build_fast_round1_messages(
        window=_window(),
        audio_file_label="clip 0001",
        extra_info="来源：https://example.test",
        streamer_index="- 主播A | 别名 | 简介",
        common_index="- 游戏B [游戏] | Game B | 简介",
        max_search_queries=8,
        use_search_contract=True,
        profile=resolve_profile("mm", "med"),
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "<research_contract>" in system
    assert "主播A" in user
    assert "游戏B" in user
    assert "<asr_result>" in user
    assert "https://example.test" in user
    assert '"current_asr_csv"' not in user
    assert "\\n" not in user.split("<asr_result>", 1)[1].split("</asr_result>", 1)[0]


def test_user_prompts_end_with_task_recap() -> None:
    """Every user template restates the task goal after the bulk input (plan A
    recap): the last paragraph must carry the 最后提醒 marker."""

    from llm.chunking import SubtitleWindow
    from llm.prompts import (
        build_correction_csv_messages,
        build_correction_query_messages,
        build_research_round1_messages,
        build_research_round2_messages,
        build_search_loop_messages,
    )

    window = _window()
    user_texts = {
        "correction": build_correction_csv_messages(window=window)[1]["content"],
        "query": build_correction_query_messages(window=window)[1]["content"],
        "research1": build_research_round1_messages(transcript="1|你好\n")[1]["content"],
        "research2": build_research_round2_messages(transcript="1|你好\n")[1]["content"],
        "loop": build_search_loop_messages(
            round_index=0, max_rounds=2, is_final_round=False
        )[1]["content"],
        "fast1": build_fast_round1_messages(window=window)[1]["content"],
    }
    for name, text in user_texts.items():
        tail = text.strip()[-800:]
        assert "最后提醒" in tail, name


def test_query_round_prompts_expose_indices_and_entry_requests() -> None:
    from llm.prompts import build_correction_query_messages

    messages = build_correction_query_messages(
        window=_window(),
        streamer_index="- 主播A | エーちゃん | 测试",
        common_index="- 游戏B [游戏] | B游 | 测试",
        profile=resolve_profile("mm", "med"),
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "<requested_entries>" in system
    assert "上限 8 条" in system
    assert "主播A | エーちゃん" in user
    assert "游戏B [游戏]" in user
