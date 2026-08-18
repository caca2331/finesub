from __future__ import annotations

import re

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.routing.config import CapabilityTier
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompt_compose import (
    PROMPT_VERSION,
    compose_correction_query_system,
    compose_correction_system,
    compose_correction_user,
    compose_fast_round1_system,
    compose_repair_turns,
    load_prompt_template,
)
from finesub.llm.prompts import build_fast_round1_messages


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


def test_prompt_version_bumped_to_v77() -> None:
    assert PROMPT_VERSION == "zh-subtitle-correction-csv-v77"


def test_variant_default_matches_tier_and_unknown_raises() -> None:
    import pytest

    from finesub.llm.prompt_variants import DEFAULT_VARIANT_FOR_TIER, VARIANTS, resolve_variant

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
    profile = resolve_profile("video", "local", "quality")
    for tier, name in ((CapabilityTier.CAPABLE, "capableC"), (CapabilityTier.BASIC, "basicB")):
        assert compose_correction_system(
            profile, tier=tier
        ) == compose_correction_system(profile, variant=name)


def test_variant_b_drops_the_singles_block() -> None:
    profile = resolve_profile("video", "local", "quality")
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

    profile = resolve_profile("video", "local", "quality")
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
    profile = resolve_profile("video", "local", "quality")
    for name in ("capableB", "capableC", "basicB"):
        system = compose_correction_system(profile, variant=name)
        assert "20 字/4 秒硬门槛" in system
        assert "36 字/7 秒" in system and "绝对门槛" in system
        assert "软门槛" not in system


def test_only_full_oneshot_examples_carry_csv_headers() -> None:
    profile = resolve_profile("video", "local", "quality")
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
        # Every variant now draws on the same material set, so the header count
        # is a property of the material (main oneshot + 缩窄合并反例), not of the
        # variant: they used to differ only because each variant had its own
        # hand-written fragment.
        assert sum(
            block.splitlines()[0] == input_header for block in asr_blocks
        ) == 2
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
    profile = resolve_profile("video", "local", "quality")
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
    profile = resolve_profile("video", "local", "quality")
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
    system = compose_correction_system(resolve_profile("video", "local", "quality"), variant="basicA")
    # The main oneshot is the first example block; its <singles>/<translated>
    # are the BasicA overlay of the shared scene.
    asr = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", system)[0]
    singles = re.findall(r"(?ms)^<singles>\n(.*?)^</singles>$", system)[0]
    translated = re.findall(r"(?ms)^<translated>\n(.*?)^</translated>$", system)[0]

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
    for switches in (
        ("audio", "local", "quality"),
        ("text", "none", "efficiency"),
        ("text", "native", "quality"),
    ):
        profile = resolve_profile(*switches)
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
    system = compose_correction_system(resolve_profile("text", "none", "efficiency"))

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
    med = compose_correction_system(resolve_profile("text", "none", "quality"))
    high = compose_correction_system(resolve_profile("text", "native", "quality"))

    assert "允许较深入思考" in med
    assert "低成本快速翻译" not in med
    assert "联网搜索（内置工具）" not in med
    assert "联网搜索（内置工具）" in high
    assert "允许较深入思考" in high
    # Injection reality: neither retrieval=none nor retrieval=native ever
    # receives a <search_results> block, so neither documents one. (v65 and
    # earlier shipped that note to text-med, describing an input that could
    # not arrive -- see docs/llm_prompts.md.)
    assert "注入的搜索结果" not in med
    assert "注入的搜索结果" not in high
    native_block = load_prompt_template("fragment_native_search_v1.md").strip()
    assert native_block in high
    assert native_block not in med


def test_mm_low_is_text_modal_with_injected_search() -> None:
    system = compose_correction_system(resolve_profile("text", "local", "quality"))

    assert "本次任务没有音频" in system
    assert "注入的搜索结果" in system
    assert "<search_results>" in system
    assert "插轴" not in system
    assert "思考与速度" not in system


def test_mm_med_is_audio_modal_with_injected_search() -> None:
    system = compose_correction_system(resolve_profile("audio", "local", "quality"))

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
    system = compose_correction_system(resolve_profile("video", "local", "quality"))

    assert "视频画面" in system
    assert "画面上出现、但主播没有说出的文字" in system
    assert "原始音频" in system


def test_evidence_pack_mode_swaps_usage_fragment() -> None:
    normal = compose_correction_system(resolve_profile("audio", "local", "quality"))
    evidence = compose_correction_system(
        resolve_profile("audio", "local", "quality"), evidence_pack_mode=True
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
    audio_user = compose_correction_user(resolve_profile("audio", "local", "quality"), **kwargs)
    text_user = compose_correction_user(resolve_profile("text", "none", "efficiency"), **kwargs)

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
        resolve_profile("audio", "local", "quality"), search_queries_rules="RULES"
    )
    text = compose_correction_query_system(
        resolve_profile("text", "local", "quality"), search_queries_rules="RULES"
    )

    assert "原始音频剪辑" in audio
    assert "结合音频听清" in audio
    assert "原始音频剪辑" not in text
    assert "本次任务没有音频" in text
    assert "发音相似度" in text
    assert "RULES" in text


def test_fast_round1_system_variants_and_messages() -> None:
    audio = compose_fast_round1_system(
        resolve_profile("audio", "local", "quality"), search_queries_rules="RULES"
    )
    video = compose_fast_round1_system(
        resolve_profile("video", "local", "quality"), search_queries_rules="RULES"
    )
    text = compose_fast_round1_system(
        resolve_profile("text", "local", "quality"), search_queries_rules="RULES"
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
        profile=resolve_profile("audio", "local", "quality"),
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

    from finesub.llm.chunking import SubtitleWindow
    from finesub.llm.prompts import (
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
    from finesub.llm.prompts import build_correction_query_messages

    messages = build_correction_query_messages(
        window=_window(),
        streamer_index="- 主播A | エーちゃん | 测试",
        common_index="- 游戏B [游戏] | B游 | 测试",
        profile=resolve_profile("audio", "local", "quality"),
    )
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "<requested_entries>" in system
    assert "上限 8 条" in system
    assert "主播A | エーちゃん" in user
    assert "游戏B [游戏]" in user


def test_reasoning_clause_is_neutral_across_the_switches() -> None:
    """The depth tiering is gone (owner decision 2026-08-12).

    The clause exists for the case where the model's own thinking does not
    happen or degrades, which does not scale with the switches -- and the old
    (difficulty, retrieval) tiering was never measured. The BASIC-tier bounded
    wording stays conditional: it guards a measured failure mode.
    """

    from finesub.llm.prompt_compose import reasoning_clause

    neutral = reasoning_clause()
    assert neutral == reasoning_clause(bounded=False)
    for media, retrieval, difficulty in (
        ("text", "none", "efficiency"),
        ("text", "none", "quality"),
        ("audio", "local", "quality"),
        ("video", "native", "intermediate"),
    ):
        profile = resolve_profile(media, retrieval, difficulty)
        system = compose_correction_system(profile, variant="capableC")
        assert neutral in system, (media, retrieval, difficulty)
    # No depth adjective survives anywhere in the neutral wording.
    assert "数百 token" not in neutral and "千余 token" not in neutral

    bounded = reasoning_clause(bounded=True)
    assert bounded != neutral and "8 行" in bounded


def test_retrieval_note_ships_only_where_search_results_can_arrive() -> None:
    for switches in (("text", "none", "quality"), ("text", "none", "efficiency")):
        assert "注入的搜索结果" not in compose_correction_system(
            resolve_profile(*switches)
        )
    for media in ("text", "audio", "video"):
        assert "注入的搜索结果" in compose_correction_system(
            resolve_profile(media, "local", "quality")
        )


def test_verify_basis_names_only_the_inputs_that_arrive() -> None:
    """The cross-check phrase is composed per axis, not written per preset."""

    from finesub.llm.prompt_compose import _verify_basis

    # local injection: audio (or context) + both injected kinds.
    assert _verify_basis(resolve_profile("audio", "local", "quality")) == (
        "音频、背景资料和搜索结果"
    )
    assert _verify_basis(resolve_profile("text", "local", "quality")) == (
        "上下文、背景资料和搜索结果"
    )
    # native: the model's own retrieval, not a harness injection.
    assert _verify_basis(resolve_profile("text", "native", "quality")) == (
        "上下文和你自己检索到的资料"
    )
    # No retrieval at all -> the phrase must not promise any.
    assert _verify_basis(resolve_profile("text", "none", "quality")) == "上下文"
    assert _verify_basis(resolve_profile("audio", "none", "quality")) == "音频"


def test_unresolved_placeholders_are_a_hard_error() -> None:
    """safe_substitute leaves unknown keys in place; assembly must not ship them."""

    import pytest

    from finesub.llm.prompt_compose import PromptAssemblyError, assert_fully_substituted

    assert assert_fully_substituted("no tokens here", what="x") == "no tokens here"
    with pytest.raises(PromptAssemblyError) as excinfo:
        assert_fully_substituted("请结合$verify_basis 交叉验证", what="x")
    assert "$verify_basis" in str(excinfo.value)
    with pytest.raises(PromptAssemblyError):
        assert_fully_substituted("${judgment_basis} 判断", what="x")


def test_every_shipped_combination_renders_without_leftovers() -> None:
    from finesub.llm.routing.config import CapabilityTier

    for media in ("text", "audio", "video"):
        for retrieval in ("none", "local", "native"):
            for tier in (CapabilityTier.CAPABLE, CapabilityTier.BASIC):
                # raises PromptAssemblyError if any $token survived
                compose_correction_system(
                    resolve_profile(media, retrieval, "quality"), tier=tier
                )
    compose_correction_system(resolve_profile("text", "none", "efficiency"))


def test_index_injection_and_entry_requests_share_one_predicate() -> None:
    """No index to read -> no rules telling the model to read it."""

    from finesub.llm.prompts import build_correction_query_messages
    from finesub.llm.chunking import SubtitleSegment, SubtitleWindow
    from finesub.llm.token_budget import CorrectionBudget

    window = SubtitleWindow(
        chunk_id="0001",
        segments=[SubtitleSegment(id="1", start=0.0, end=1.0, text="一。")],
        overlap_segments=[],
        boundary_reason="test",
        budget=CorrectionBudget(
            input_tokens=10,
            subtitle_input_tokens=5,
            estimated_output_tokens=50,
            total_with_margin=60,
            token_counter_source="test",
        ),
        clip_start=0.0,
        clip_end=10.0,
    )

    with_kb = build_correction_query_messages(
        window=window, streamer_index="# 甲\n", common_index="# 乙\n"
    )[0]["content"]
    assert "<requested_entries>" in with_kb

    # knowledge off (or an empty knowledge root): the round used to keep the
    # request rules while facing a "（空）" index.
    without_kb = build_correction_query_messages(window=window)[0]["content"]
    assert "<requested_entries>" not in without_kb
    assert "知识库词条请求规则" not in without_kb

    # The knowledge-owned *input sections* follow the same predicate: no empty
    # index/carried blocks shown to a round that has no rules for them.
    user_without_kb = build_correction_query_messages(window=window)[1]["content"]
    for marker in ("<streamer_index>", "<common_index>", "<carried_entries>", "剩余额度"):
        assert marker not in user_without_kb, marker
    user_with_kb = build_correction_query_messages(
        window=window, streamer_index="# 甲\n", common_index="# 乙\n"
    )[1]["content"]
    for marker in ("<streamer_index>", "<common_index>", "<carried_entries>", "剩余额度"):
        assert marker in user_with_kb, marker


def test_slot_registry_picks_the_first_matching_rule() -> None:
    from finesub.llm.prompt_compose import CORRECTION_SLOTS, PromptContext

    def fragment(slot, switches, **kwargs):
        ctx = PromptContext(profile=resolve_profile(*switches), **kwargs)
        return CORRECTION_SLOTS.resolve(slot, ctx).fragment

    assert fragment("retrieval", ("audio", "local", "quality")) == (
        "fragment_retrieval_injected_v1.md"
    )
    assert fragment("retrieval", ("text", "native", "quality")) == (
        "fragment_native_search_v1.md"
    )
    assert fragment("retrieval", ("text", "none", "quality")) == ""

    assert fragment("effort", ("audio", "local", "quality")) == ""
    assert fragment("effort", ("text", "none", "efficiency")) == (
        "fragment_effort_low_v1.md"
    )
    assert fragment("effort", ("text", "none", "quality")) == (
        "fragment_effort_deep_v1.md"
    )

    assert fragment("keep_entries", ("audio", "local", "quality")) != ""
    assert fragment("keep_entries", ("text", "none", "quality")) == ""
    assert fragment(
        "keep_entries", ("audio", "local", "quality"), knowledge_enabled=False
    ) == ""


def test_slot_requires_are_checked_against_what_is_really_injected() -> None:
    """A fragment may not ship into a context that lacks the block it describes."""

    import pytest

    from finesub.llm.prompt_compose import (
        CORRECTION_SLOTS,
        PromptAssemblyError,
        PromptContext,
        SlotRegistry,
        SlotRule,
    )

    text_only = PromptContext(profile=resolve_profile("text", "none", "quality"))
    assert "search_results" not in text_only.injected_blocks
    assert "audio" not in text_only.injected_blocks

    # The real registry never selects such a rule...
    assert CORRECTION_SLOTS.resolve("retrieval", text_only).fragment == ""

    # ...and if a future edit made it, assembly fails instead of shipping a
    # prompt that describes an input which never arrives.
    broken = SlotRegistry(
        {
            "retrieval": (
                SlotRule(
                    when=lambda c: True,
                    fragment="fragment_retrieval_injected_v1.md",
                    requires=("search_results",),
                ),
            )
        }
    )
    with pytest.raises(PromptAssemblyError, match="search_results"):
        broken.resolve("retrieval", text_only)


def test_injected_blocks_track_the_switch_vector() -> None:
    from finesub.llm.prompt_compose import PromptContext

    local = PromptContext(profile=resolve_profile("video", "local", "quality"))
    assert {"audio", "video", "search_results", "context_pack", "entry_details"} <= (
        local.injected_blocks
    )

    native = PromptContext(profile=resolve_profile("text", "native", "quality"))
    assert "context_pack" in native.injected_blocks  # round 2 still runs
    assert "search_results" not in native.injected_blocks

    none = PromptContext(
        profile=resolve_profile("text", "none", "quality"), knowledge_enabled=False
    )
    assert "context_pack" not in none.injected_blocks
    assert "entry_details" not in none.injected_blocks


def test_media_wording_names_only_what_is_attached() -> None:
    video = compose_correction_system(resolve_profile("video", "local", "quality"))
    audio = compose_correction_system(resolve_profile("audio", "local", "quality"))
    text = compose_correction_system(resolve_profile("text", "local", "quality"))

    # video is its own parameter layer, not audio plus an addendum (docs/llm_prompts.md):
    # the phrases enumerating what to judge from name the frames too. The merge
    # and noisy-span wording lives in the variants that use those parameters
    # (capableC's examples fragment references neither).
    # The role fragment's input list is the thing that must name only what is
    # attached (the text prompt still *mentions* audio, to say there is none).
    def inventory(text_):
        return [line for line in text_.splitlines() if "你会同时参考" in line or "你只能依据" in line][0]

    assert "音频与视频画面" in inventory(video)
    assert "音频" in inventory(audio) and "画面" not in inventory(audio)
    assert "音频" not in inventory(text).split("；", 1)[1]

    video_b = compose_correction_system(
        resolve_profile("video", "local", "quality"), variant="capableB"
    )
    audio_b = compose_correction_system(
        resolve_profile("audio", "local", "quality"), variant="capableB"
    )
    assert "音画和语义" in video_b and "必要时看画面" in video_b
    assert "音画和语义" not in audio_b and "画面" not in audio_b

    # The role fragment's input list is composed per axis, so a profile with no
    # retrieval is not told it will receive background material.
    assert "背景资料" in audio
    assert "背景资料" not in compose_correction_system(
        resolve_profile("audio", "none", "quality")
    )
    assert "知识库词条" not in compose_correction_system(
        resolve_profile("audio", "local", "quality"), knowledge_enabled=False
    )


def test_background_conflict_rules_need_something_to_conflict_with() -> None:
    marker = "严防背景污染"

    assert marker in compose_correction_system(resolve_profile("audio", "local", "quality"))
    assert marker in compose_correction_system(
        resolve_profile("audio", "none", "quality"), knowledge_enabled=True
    )
    # No audio: the rules are about trusting your ears.
    assert marker not in compose_correction_system(
        resolve_profile("text", "local", "quality")
    )
    # Audio but nothing injected to be polluted by.
    assert marker not in compose_correction_system(
        resolve_profile("audio", "none", "quality"), knowledge_enabled=False
    )

def test_parallel_continuity_drops_the_ledger_from_both_prompts() -> None:
    """continuity=parallel has no advice ledger to inject and no reader for
    <next_advice>/<keep_entries>; neither may be mentioned (plan A.7)."""

    from finesub.llm.prompt_compose import PromptContext

    serial = resolve_profile("audio", "local", "quality", "serial")
    parallel = resolve_profile("audio", "local", "quality", "parallel")

    serial_sys = compose_correction_system(serial)
    parallel_sys = compose_correction_system(parallel)
    assert "<next_advice>" in serial_sys and "累积" in serial_sys
    assert "<next_advice>" not in parallel_sys
    assert "<keep_entries>" not in parallel_sys
    assert "累积建议台账" not in parallel_sys

    user_kwargs = dict(
        general_context_json="{}",
        window_context="（无）",
        entry_details="（无）",
        previous_advice="",
        pre_round_notes="（无）",
        search_results="（无）",
        preceding_context_csv="",
        current_asr_csv="1|0|1|0|x",
        current_asr_row_count=1,
    )
    serial_user = compose_correction_user(serial, **user_kwargs)
    parallel_user = compose_correction_user(parallel, **user_kwargs)
    assert "<previous_advice>" in serial_user and "<next_advice>" in serial_user
    assert "<previous_advice>" not in parallel_user
    assert "<next_advice>" not in parallel_user
    assert "<keep_entries>" not in parallel_user

    assert "next_advice_ledger" in PromptContext(profile=serial).injected_blocks
    assert "next_advice_ledger" not in PromptContext(profile=parallel).injected_blocks


def test_repair_turns_list_every_reason_and_ask_for_a_whole_answer() -> None:
    turns = compose_repair_turns(
        "sub|1|wrong",
        ["Row 1 references unknown source id 3.", "  ", "Translated missing id 2."],
    )

    assert [turn["role"] for turn in turns] == ["assistant", "user"]
    assert turns[0]["content"] == "sub|1|wrong"
    body = turns[1]["content"]
    assert "- Row 1 references unknown source id 3." in body
    assert "- Translated missing id 2." in body
    assert "- \n" not in body  # blank entries are dropped, not rendered empty
    # The failure this exists for produced correct content and lost it to a
    # partial re-send; the ask has to be for the whole thing.
    assert "完整" in body


def test_repair_turns_need_something_to_repair_and_a_reason() -> None:
    assert compose_repair_turns("", ["boom"]) == []
    assert compose_repair_turns("   ", ["boom"]) == []
    assert compose_repair_turns("out", []) == []
    assert compose_repair_turns("out", ["", "   "]) == []

