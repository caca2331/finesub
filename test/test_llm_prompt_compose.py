from __future__ import annotations

import dataclasses
import re

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.output_protocol import output_row_spec
from finesub.llm.prompt_constants import BASE

SPLICE_MAX_GAP_SECONDS = BASE["splice_max_gap_seconds"]
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.routing.config import CapabilityTier
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompt_compose import (
    PROMPT_TEMPLATE_DIR,
    PROMPT_VERSION,
    compose_correction_system,
    compose_correction_user,
    compose_repair_turns,
    load_prompt_template,
    mode_texts,
)


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
    return dataclasses.replace(plan_correction_windows(segments, counter=FakeTokenCounter())[0], length_scale=1.0)


def test_prompt_version_is_current() -> None:
    assert PROMPT_VERSION == "zh-subtitle-correction-csv-v114"


def test_variant_default_matches_tier_and_unknown_raises() -> None:
    import pytest

    from finesub.llm.prompt_variants import DEFAULT_VARIANT_FOR_TIER, VARIANTS, resolve_variant

    # The tier still picks a default variant; passing None must equal it.
    assert resolve_variant(None, CapabilityTier.CAPABLE).name == "capableC"
    assert resolve_variant(None, CapabilityTier.BASIC).name == "basicA"
    assert resolve_variant("basicA").name == "basicA"
    assert set(VARIANTS) == {"basicA", "capableC", "basicA-polish", "capableC-polish"}
    assert DEFAULT_VARIANT_FOR_TIER[CapabilityTier.CAPABLE] == "capableC"
    assert DEFAULT_VARIANT_FOR_TIER[CapabilityTier.BASIC] == "basicA"
    with pytest.raises(ValueError):
        resolve_variant("nope")


def test_variant_selection_matches_tier_default_byte_for_byte() -> None:
    # Selecting the variant explicitly must equal the tier-derived default.
    profile = resolve_profile("video", "local", "quality")
    for tier, name in ((CapabilityTier.CAPABLE, "capableC"), (CapabilityTier.BASIC, "basicA")):
        assert compose_correction_system(
            profile, tier=tier, length_scale=1.0
        ) == compose_correction_system(profile, variant=name, length_scale=1.0)


def test_no_variant_mentions_a_singles_stage() -> None:
    """Every set answers in one <translated> block; a model that has no singles stage is never told
    about one (basicA dropped its own, owner 2026-10-31)."""

    profile = resolve_profile("video", "local", "quality")
    for name in ("basicA", "capableC"):
        system = compose_correction_system(profile, variant=name, length_scale=1.0)
        user = compose_correction_user(
            profile,
            variant=name,
            general_context_json="",
            window_context="",
            entry_details="",
            previous_advice="",
            pre_round_notes="",
            search_results="",
            preceding_context_csv="",
            current_asr_csv="1|0|1|0|x|x|high|1|n",
            current_asr_row_count=1,
            length_scale=1.0,
        )
        assert "singles" not in system, name
        assert "singles" not in user, name
        assert "<translated>" in system


def test_basic_a_keeps_no_judgment_merge_wording() -> None:
    """basicA merges only a word cut in two: the user template's judgment-merge reminder, the length
    thresholds and the examples that teach judgment merges stay out (owner 2026-10-31)."""

    profile = resolve_profile("audio", "local", "efficiency")
    system = compose_correction_system(profile, variant="basicA", length_scale=1.0)
    user = compose_correction_user(
        profile,
        variant="basicA",
        general_context_json="",
        window_context="",
        entry_details="",
        previous_advice="",
        pre_round_notes="",
        search_results="",
        preceding_context_csv="",
        current_asr_csv="1|0|1|0|x",
        current_asr_row_count=1,
        length_scale=1.0,
    )
    assert "可以合并语义连贯的短片段" not in user
    assert "酌情缩窄" not in user
    assert "一行对一个源" in user
    # Every restatement of the splice test carries all four conditions, the gap as a number (cold read v102).
    splice = f"两个连续源、边界无标点无空格、拼回同一个词、后一行的 gap ≤{SPLICE_MAX_GAP_SECONDS} 秒"
    assert user.count(splice) == 2
    assert f"后一行的 gap ≤{SPLICE_MAX_GAP_SECONDS} 秒" in system
    assert "前一行 gap" not in system + user
    assert "接近 0" not in system
    assert "硬门槛" not in system
    assert "filler 三明治" not in system
    assert "缩窄" not in system

def test_inline_row_examples_carry_the_variant_header_columns() -> None:
    """Every backticked `sub|…` / `insert|…` row a contract quotes has as many cells as that variant's
    header: a column change regenerates the goldens wholesale, so a stale inline row would slip through."""

    profile = resolve_profile("audio", "local", "efficiency")
    quoted = {}
    for name in ("basicA", "capableC"):
        spec = resolve_variant(name)
        width = len(output_row_spec(start=spec.output_has_start))
        quoted[name] = re.findall(r"`((?:sub|insert)\|[^`]*)`", compose_correction_system(profile, variant=name, length_scale=1.0))
        for row in quoted[name]:
            assert len(row.split("|")) == width, (name, row)
    assert quoted["basicA"]  # the splice row the contract quotes


def test_basic_a_renders_only_the_examples_that_fit_one_row_per_source() -> None:
    """basicA keeps example 50 (one source, translation) and drops the merge examples 10, 30 and 40,
    which three passes' pass 2 still shows."""

    profile = resolve_profile("audio", "local", "efficiency")
    basic_a = compose_correction_system(profile, variant="basicA", length_scale=1.0)
    grouped = compose_correction_system(profile, variant="capableC", grouped=True, length_scale=1.0)
    assert "女皇陛下" in basic_a
    for lead in ("缩窄合并反例", "filler 三明治三源特例", "跨说话人反例"):
        assert lead in grouped, lead
        assert lead not in basic_a, lead


def test_capablec_composes_with_the_shape_section_and_no_singles() -> None:
    import re

    profile = resolve_profile("video", "local", "quality")
    sys_msg = compose_correction_system(profile, variant="capableC", length_scale=1.0)
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
        length_scale=1.0,
    )
    # Composes cleanly (no unresolved $placeholder leaked from a fragment).
    assert not re.search(r"\$[a-zA-Z_]+", sys_msg), "unresolved placeholder in system"
    assert not re.search(r"\$[a-zA-Z_]+", user_msg), "unresolved placeholder in user"
    # capableC: no singles, no `#` comments; it judges who is speaking before it corrects.
    assert "singles" not in sys_msg
    assert "`#`" not in sys_msg and "`#`" not in user_msg
    assert sys_msg.index("认人与定形状") < sys_msg.index("短片段合并策略（仅作用于")
    assert "先在内部思考里认人、判接缝" in user_msg


def test_capable_c_names_the_usual_length_and_the_absolute_threshold() -> None:
    profile = resolve_profile("video", "local", "quality")
    # capableC words the usual length as pass 2 does: about 20 / 4 s, the absolute threshold the one bound.
    capable = compose_correction_system(profile, variant="capableC", length_scale=1.0)
    assert "一般在 20 字、4 秒左右以内" in capable and "36 字/7 秒绝对门槛" in capable
    assert "硬门槛" not in capable and "软门槛" not in capable


def test_worked_examples_carry_the_csv_headers_their_material_gives() -> None:
    profile = resolve_profile("video", "local", "quality")
    input_header = "local_id|start|duration|gap|text"
    output_header = (
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
    )
    basic_output_header = "type|position|start|corrected_text|translation|conf|note"
    six_column_header = "type|position|corrected_text|translation|conf|note"
    assert output_header not in compose_correction_system(profile, variant="capableC", length_scale=1.0)
    for name in ("basicA", "capableC"):
        system = compose_correction_system(profile, variant=name, length_scale=1.0)
        asr_blocks = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", system)
        # Every example's input block opens with the header, as the real `<asr_result>` does.
        assert asr_blocks and all(block.splitlines()[0] == input_header for block in asr_blocks)
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
            basic_output_header if name == "basicA" else six_column_header
        )
        # Every worked example's `<translated>` opens with the header, as the contract's item 2 says; only the
        # bad-output example (错误输出) shows a block without one.
        assert [block.splitlines()[0] == expected_output_header for block in translated_blocks] == (
            [True] * (len(translated_blocks) - 1) + [False]
        )
        assert len(translated_blocks) == (3 if name == "basicA" else 4)

        assert not re.findall(r"(?ms)^<singles>\n(.*?)^</singles>$", system)


def test_basic_a_keeps_start_csv_while_capable_c_has_six_columns() -> None:
    profile = resolve_profile("video", "local", "quality")
    start_header = "type|position|start|corrected_text|translation|conf|note"
    legacy_header = (
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note"
    )
    basic_a = compose_correction_system(profile, variant="basicA", length_scale=1.0)
    assert start_header in basic_a
    assert "固定 7 列" in basic_a
    assert legacy_header not in basic_a
    capable = compose_correction_system(profile, variant="capableC", length_scale=1.0)
    assert legacy_header not in capable
    assert "type|position|corrected_text|translation|conf|note" in capable
    assert start_header not in capable


def test_capable_c_has_full_43_row_oneshot() -> None:
    profile = resolve_profile("video", "local", "quality")
    system = compose_correction_system(profile, variant="capableC", length_scale=1.0)
    oneshot = system.split("完整 oneshot", 1)[1]
    asr = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", oneshot)[0]
    translated = re.findall(
        r"(?ms)^<translated>\n(.*?)^</translated>$", oneshot
    )[0]
    assert len([line for line in asr.splitlines() if re.match(r"^\d+\|", line)]) == 43
    rows = [
        line for line in translated.splitlines()
        if line.startswith(("sub|", "discard|"))
    ]
    assert not any(line.startswith("#") for line in translated.splitlines())
    assert len(rows) == 41


def test_basic_a_has_full_43_row_oneshot_one_row_per_source() -> None:
    system = compose_correction_system(resolve_profile("video", "local", "quality"), variant="basicA", length_scale=1.0)
    # The main oneshot is the first example block; its <translated> is the BasicA overlay of the shared scene.
    asr = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>$", system)[0]
    translated = re.findall(r"(?ms)^<translated>\n(.*?)^</translated>$", system)[0]

    assert len([line for line in asr.splitlines() if re.match(r"^\d+\|", line)]) == 43
    assert len([line for line in translated.splitlines() if line.startswith("sub|")]) == 41
    assert "discard|18|" in translated
    assert "sub|8|" in translated and "sub|9|" in translated
    assert "sub|10|" in translated and "sub|11|" in translated
    assert "sub|8,9|" not in translated
    assert "sub|10,11|" not in translated

def test_correction_system_tier_selects_merge_fragments() -> None:
    char_rule = load_prompt_template("fragment_weighted_char_count_v1.md", **mode_texts(None)).strip()
    for switches in (
        ("audio", "local", "quality"),
        ("text", "none", "efficiency"),
        ("text", "native", "quality"),
    ):
        profile = resolve_profile(*switches)
        capable = compose_correction_system(profile, tier=CapabilityTier.CAPABLE, length_scale=1.0)
        basic_a = compose_correction_system(profile, tier=CapabilityTier.BASIC, length_scale=1.0)

        # The tier-independent discipline fragment lands in both variants.
        assert "translated 产出纪律" in capable
        assert "translated 产出纪律" in basic_a
        # The basic default (basicA) is one row per source; capableC judges the seams itself and merges by
        # pass 2's rules.
        assert "认人与定形状" in capable and "判成 `may_join` 的接缝也不一定要并" in capable
        assert "本档一行对一个源" in basic_a
        assert "本档一行对一个源" not in capable
        assert "输入完整 43 条" in capable or "输入 43 条" in capable
        assert "<singles>" not in basic_a
        assert "口播碎片成一句" not in basic_a
        # The weighted char-count algorithm is injected exactly once (via the
        # output contract) — the old merge-rules duplicate is gone. basicA merges
        # only a word cut in two, bounded by span, and writes no char count: it
        # never needs the rule.
        assert capable.count(char_rule) == 1
        assert basic_a.count(char_rule) == 0
        # Default tier is capable.
        assert compose_correction_system(profile, length_scale=1.0) == capable

def test_text_low_system_has_no_audio_insert_or_search() -> None:
    system = compose_correction_system(resolve_profile("text", "none", "efficiency"), length_scale=1.0)

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
    assert "一条合并行最多 4 个连续源" in system
    assert "<next_advice>" in system
    assert "感谢观看、订阅点赞一类" in system  # capableC drops by pass 1's criteria, not the single pass list
    assert "<asr_result>" in system
    assert "$" not in system  # every slot resolved


def test_text_med_and_high_swap_effort_and_native_search() -> None:
    med = compose_correction_system(resolve_profile("text", "none", "quality"), length_scale=1.0)
    high = compose_correction_system(resolve_profile("text", "native", "quality"), length_scale=1.0)

    assert "允许较深入思考" in med
    assert "低成本快速翻译" not in med
    assert "联网搜索（内置工具）" not in med
    assert "联网搜索（内置工具）" in high
    assert "允许较深入思考" in high
    # Injection reality: neither retrieval=none nor retrieval=native ever
    # receives a <search_results> block, so neither documents one. (v65 and
    # earlier shipped that note to text-med, describing an input that could
    # not arrive -- see docs/llm_prompts.md.)
    assert "harness 代为检索，结果放在" not in med
    assert "harness 代为检索，结果放在" not in high
    native_block = load_prompt_template("fragment_native_search_v1.md", **mode_texts(None)).strip()
    assert native_block in high
    assert native_block not in med


def test_mm_low_is_text_modal_with_injected_search() -> None:
    system = compose_correction_system(resolve_profile("text", "local", "quality"), length_scale=1.0)

    assert "本次任务没有音频" in system
    assert "harness 代为检索，结果放在" in system
    assert "<search_results>" in system
    assert "插轴" not in system
    assert "思考与速度" not in system


def test_mm_med_is_audio_modal_with_injected_search() -> None:
    system = compose_correction_system(resolve_profile("audio", "local", "quality"), length_scale=1.0)

    assert "原始音频" in system
    assert "剪辑内开始时间" in system
    # Insert/插轴 deprecated for all variants (v63+).
    assert "插轴（插入源字幕遗漏的字幕）" not in system
    assert "插轴示例" not in system
    assert "type=insert" not in system
    assert "harness 代为检索，结果放在" in system
    assert "视频画面" not in system
    assert "思考与速度" not in system
    assert "$" not in system


def test_mm_high_adds_video_addendum() -> None:
    system = compose_correction_system(resolve_profile("video", "local", "quality"), length_scale=1.0)

    assert "视频画面" in system
    assert "画面上出现、但主播没有说出的文字" in system
    assert "原始音频" in system


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
    audio_user = compose_correction_user(resolve_profile("audio", "local", "quality"), **kwargs, length_scale=1.0)
    text_user = compose_correction_user(resolve_profile("text", "none", "efficiency"), **kwargs, length_scale=1.0)

    assert "type=insert" not in audio_user
    assert "剪辑音频的 0 秒" in audio_user
    assert "type=insert" not in text_user
    assert "不要输出 SRT 时间戳" in text_user
    assert "不得残留日语假名或助词" in audio_user
    assert "<entry_details>" in audio_user
    assert "<pre_round_notes>" in text_user
    assert "共有 17 条字幕" in audio_user
    assert "共有 **17 条输入字幕**" in text_user


def test_user_prompts_end_with_task_recap() -> None:
    """Every user template restates the task goal after the bulk input (plan A
    recap): the last paragraph must carry the 最后提醒 marker."""

    from finesub.llm.prepare_prompts import build_prepare_task_messages
    from finesub.llm.prompts import build_correction_csv_messages

    window = _window()
    user_texts = {
        "correction": build_correction_csv_messages(window=window)[1]["content"],
        "prepare": build_prepare_task_messages(transcript="1|你好\n")[1]["content"],
    }
    for name, text in user_texts.items():
        tail = text.strip()[-800:]
        assert "最后提醒" in tail, name


def test_no_prompt_asks_for_a_reasoning_block() -> None:
    """No session asks for a visible `<reasoning>` block -- the correction window on any switch or variant, the
    preparation, the knowledge sessions -- since Claude refuses a prompt that asks for one
    (`reasoning_extraction`); basicA alone opens its delivery with a `<decisions>` list instead."""

    for media, retrieval, difficulty in (
        ("text", "none", "efficiency"),
        ("text", "none", "quality"),
        ("audio", "local", "quality"),
        ("video", "native", "intermediate"),
    ):
        profile = resolve_profile(media, retrieval, difficulty)
        for variant in ("capableC", "basicA"):
            system = compose_correction_system(profile, variant=variant, length_scale=1.0)
            assert "reasoning" not in system, (media, retrieval, difficulty, variant)
            assert ("<decisions>" in system) == (variant == "basicA"), (media, retrieval, difficulty, variant)
    for path in PROMPT_TEMPLATE_DIR.rglob("*"):
        if path.is_file():
            assert "reasoning" not in path.read_text(encoding="utf-8"), path.name


def test_retrieval_note_ships_only_where_search_results_can_arrive() -> None:
    for switches in (("text", "none", "quality"), ("text", "none", "efficiency")):
        assert "harness 代为检索，结果放在" not in compose_correction_system(
            resolve_profile(*switches), length_scale=1.0
        )
    for media in ("text", "audio", "video"):
        assert "harness 代为检索，结果放在" in compose_correction_system(
            resolve_profile(media, "local", "quality"), length_scale=1.0
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
                    resolve_profile(media, retrieval, "quality"), tier=tier, length_scale=1.0
                )
    compose_correction_system(resolve_profile("text", "none", "efficiency"), length_scale=1.0)


def test_index_injection_and_entry_requests_share_one_predicate() -> None:
    """No index to read -> no rules telling the model to read it."""

    from finesub.llm.request_prompts import build_window_task_messages
    from finesub.llm.chunking import SubtitleSegment, SubtitleWindow
    from finesub.llm.token_budget import CorrectionBudget

    def opening(**kwargs):
        return build_window_task_messages(
            request_channel=True,
            window=window,
            profile=resolve_profile("text", "local", "quality"),
            **kwargs,
        )

    window = SubtitleWindow(
        chunk_id="0001",
        segments=[SubtitleSegment(id="1", start=0.0, end=1.0, text="一。")],
        overlap_segments=[],
        boundary_reason="test",
        budget=CorrectionBudget(
            input_tokens=10,
            subtitle_input_tokens=5,
            estimated_output_tokens=50,
            token_counter_source="test",
        ),
        clip_start=0.0,
        clip_end=10.0,
        length_scale=1.0,
    )

    with_kb = opening(knowledge_indexes={"streamer": "# 甲\n", "common": "# 乙\n"})[0]["content"]
    assert "<requested_entries>" in with_kb
    assert "<streamer_index>" in with_kb and "<common_index>" in with_kb

    # knowledge off (or an empty knowledge root): no request rules, and no
    # empty index shown to a turn that has no rules for it.
    without_kb = opening(knowledge_indexes={})[0]["content"]
    assert "<requested_entries>" not in without_kb
    for marker in ("<streamer_index>", "<common_index>"):
        assert marker not in without_kb, marker


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
    # A readable base gives none/native an entries-only request channel.
    assert fragment("keep_entries", ("text", "none", "quality")) != ""
    assert fragment("keep_entries", ("text", "none", "quality"), knowledge_enabled=False) == ""
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
    video = compose_correction_system(resolve_profile("video", "local", "quality"), length_scale=1.0)
    audio = compose_correction_system(resolve_profile("audio", "local", "quality"), length_scale=1.0)
    text = compose_correction_system(resolve_profile("text", "local", "quality"), length_scale=1.0)

    # video is its own parameter layer, not audio plus an addendum (docs/llm_prompts.md):
    # the phrases enumerating what to judge from name the frames too. The merge
    # and noisy-span wording lives in the variants that use those parameters
    # (capableC's examples fragment references neither).
    # The role fragment's input list is the thing that must name only what is
    # attached (the text prompt still *mentions* audio, to say there is none).
    def inventory(text_):
        return [line for line in text_.splitlines() if "你会同时参考" in line or "只能依据" in line][0]

    assert "音频与视频画面" in inventory(video)
    assert "音频" in inventory(audio) and "画面" not in inventory(audio)
    assert "音频" not in inventory(text).split("；", 1)[1]

    video_a = compose_correction_system(
        resolve_profile("video", "local", "quality"), variant="basicA", length_scale=1.0
    )
    audio_a = compose_correction_system(
        resolve_profile("audio", "local", "quality"), variant="basicA", length_scale=1.0
    )
    assert "必要时看画面" in video_a
    assert "画面" not in audio_a

    # The role fragment's input list is composed per axis. The preparation's
    # background is in every window's user message, retrieval or not (P724).
    assert "背景资料" in inventory(audio)
    assert "背景资料" in inventory(
        compose_correction_system(resolve_profile("text", "none", "quality"), length_scale=1.0)
    )
    assert "知识库词条" not in compose_correction_system(
        resolve_profile("audio", "local", "quality"), knowledge_enabled=False, length_scale=1.0
    )


def test_background_conflict_rules_need_something_to_conflict_with() -> None:
    marker = "严防背景污染"

    assert marker in compose_correction_system(resolve_profile("audio", "local", "quality"), length_scale=1.0)
    assert marker in compose_correction_system(
        resolve_profile("audio", "none", "quality"), knowledge_enabled=True, length_scale=1.0
    )
    # No audio: the rules are about trusting your ears.
    assert marker not in compose_correction_system(
        resolve_profile("text", "local", "quality"), length_scale=1.0
    )
    # Audio but nothing injected to be polluted by.
    assert marker not in compose_correction_system(
        resolve_profile("audio", "none", "quality"), knowledge_enabled=False, length_scale=1.0
    )

def test_parallel_continuity_drops_the_ledger_from_both_prompts() -> None:
    """continuity=parallel has no advice ledger to inject and no reader for
    <next_advice>/<keep_entries>; neither may be mentioned (plan A.7)."""

    from finesub.llm.prompt_compose import PromptContext

    serial = resolve_profile("audio", "local", "quality", "serial")
    parallel = resolve_profile("audio", "local", "quality", "parallel")

    serial_sys = compose_correction_system(serial, length_scale=1.0)
    parallel_sys = compose_correction_system(parallel, length_scale=1.0)
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
    serial_user = compose_correction_user(serial, **user_kwargs, length_scale=1.0)
    parallel_user = compose_correction_user(parallel, **user_kwargs, length_scale=1.0)
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


def test_contract_cross_references_point_at_the_block_order_rule() -> None:
    # The final-block rule points at the block-order rule by number; a rule inserted above it would leave the
    # pointer aimed at the wrong rule without any golden noticing.
    from pathlib import Path

    import finesub.llm

    templates = Path(finesub.llm.__file__).parent / "prompt_templates"
    found = 0
    for path in sorted(templates.glob("fragment_output_contract_*_v1.md")):
        text = path.read_text(encoding="utf-8")
        for number in re.findall(r"本契约第 (\d+) 条", text):
            rule = next(line for line in text.splitlines() if line.startswith(f"{number}. "))
            assert "规定顺序为" in rule, (path.name, number)
            found += 1
    assert found == 2
