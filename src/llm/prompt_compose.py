"""Fragment selection and assembly for correction / fast-mode prompts.

Prompt text lives in ``prompt_templates/*.md``; this module only decides which
fragments a given :class:`TranslationProfile` gets and stitches them into the
correction skeletons. Fragment inventory and the slot selection table are
documented in ``docs/llm_prompts.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Dict

from .chunking import ASR_RESULT_CSV_HEADER
from .config import CapabilityTier
from .csv_utils import OUTPUT_CSV_HEADER, OUTPUT_CSV_HEADER_WITH_START
from .prompt_variants import resolve_variant
from .profiles import DEFAULT_PROFILE, TranslationProfile

PROMPT_VERSION = "zh-subtitle-correction-csv-v63"

# v17: every session must OPEN with a visible <reasoning> block (soft
# requirement: a missing block never fails validation or retries; parsers
# ignore its content). The expected depth scales with the call's thinking
# tier; low/medium carry a rough token-scale hint, high is unconstrained.
REASONING_DEPTH_CLAUSES: Dict[str, str] = {
    "low": "在其中快速列出分组、取舍、疑点要点即可，不做长链条推理（一般数百 token 内）；",
    "medium": "在其中梳理本次输入的话题、疑点、候选与执行计划（一般千余 token 内）；",
    "high": "在其中充分推理：可疑点、候选写法、取舍权衡与验证思路，想清楚再落笔；",
}


def reasoning_clause(depth: str = "medium", *, bounded: bool = False) -> str:
    """The mandatory opening-<reasoning> clause, worded for ``depth``.

    ``bounded`` swaps in the BASIC-tier wording: weak models that write an
    open-ended plan first tend to substitute the plan for the work (round-46:
    a thorough reasoning block followed by placeholder singles), so the block
    is hard-capped and forbidden from rehearsing output lines."""

    if bounded:
        return (
            "回复必须以有且仅有一个 `<reasoning>...</reasoning>` 块开头：只写不超过 "
            "8 行要点（专名决定、高风险区间、分组难点），禁止逐行预演输出、复述流程"
            "或写任何字幕行草稿——推理的产出只能是接下来完整写出的字幕行本身；"
            "写完要点后立即开始输出规定的标签块。"
        )
    depth_clause = REASONING_DEPTH_CLAUSES.get(
        (depth or "").strip().lower(), REASONING_DEPTH_CLAUSES["medium"]
    )
    return (
        "回复必须以有且仅有一个 `<reasoning>...</reasoning>` 块开头"
        f"：{depth_clause}随后再输出规定的标签块。"
    )


def correction_reasoning_depth(profile: TranslationProfile) -> str:
    """Reasoning depth tier for correction-family calls.

    The text route mirrors its binary effort fragments (low vs deep) so
    text-med and text-high stay structurally identical apart from the
    native-search block; the mm route reserves the high tier for mm-high.
    """

    if profile.route == "text":
        return "low" if profile.level == "low" else "high"
    return "high" if profile.level == "high" else "medium"


PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"


def load_prompt_template(name: str, **values: object) -> str:
    path = PROMPT_TEMPLATE_DIR / name
    template = Template(path.read_text(encoding="utf-8"))
    defaults = {"prompt_version": PROMPT_VERSION}
    defaults.update({key: str(value) for key, value in values.items()})
    return template.safe_substitute(defaults)


# Phrase-level modal parameters (structural differences are whole fragment
# files instead; see the design doc's "变体优先做整文件" rule).
_AUDIO_MODAL_PARAMS: Dict[str, str] = {
    "csv_time_col_name": "剪辑内开始时间",
    "csv_time_note": "`剪辑内开始时间` 以你收到的剪辑音频的 0 秒为基准，可直接用来在剪辑中定位对应语音。",
    "judgment_basis": "音频、ASR 文本、语义连贯性、语气停顿和中文字幕长度",
    "judgment_basis_short": "音频和语义",
    "merge_connect_basis": "音频和语义",
    "speaker_basis": "音视频/语境",
    "insert_type_clause": "；不要输出其他 type 值",
    "insert_position_clause": "",
    "insert_duration_clause": "",
    "insert_note_clause": "",
    "discard_insert_clause": (
        "\n5. 丢弃取舍：打算以 discard|<源序号> 丢弃之前，先重听该区间音频——"
        "能辨识出实词或短语的，应修正后保留，而不是整体丢弃；只有音频本身"
        "也没有语义内容（纯哭声、喘息、感叹）时才写 discard 行丢弃。"
    ),
    "verify_basis": "音频、背景资料和搜索结果",
    "noisy_span_handling": (
        "重听该区间逐句处理：能听出实词、短语的，按听到的内容重写并按停顿拆条；"
        "确认只是喊叫、喘息等无语义感叹的，以 discard|<源序号> 显式丢弃"
    ),
    "preceding_audibility_note": (
        "最贴近窗口的前文行可能落在剪辑开头几秒内、其语音可听，但它们同样是只读背景，"
        "不要为其输出字幕。"
    ),
    "paren_rule": (
        "括注采用正证据门槛：只有音频中能独立确认存在实际的非语音声响（有声响、没有"
        "可转写话语）时，才可写极简中性括注（如「（提示音）」），不夹带解释；若音频中"
        "没有对应声响或无法确认声响性质，不得把文本改写成括注。可听见的系统语音仍按话语处理。"
    ),
}
_TEXT_MODAL_PARAMS: Dict[str, str] = {
    "csv_time_col_name": "窗口内开始时间",
    "csv_time_note": "`窗口内开始时间` 以本窗口第一条字幕为 0 秒基准，仅用于把握说话节奏与间隔。",
    "judgment_basis": "ASR 文本、时间信息（开始/时长/gap）、语义连贯性和中文字幕长度",
    "judgment_basis_short": "语义与发音相似度",
    "merge_connect_basis": "语义与时间间隔",
    "speaker_basis": "语境/文本线索",
    "insert_type_clause": "；不要输出其他 type 值",
    "insert_position_clause": "",
    "insert_duration_clause": "",
    "insert_note_clause": "",
    "discard_insert_clause": "",
    "verify_basis": "上下文、背景资料和搜索结果",
    "noisy_span_handling": (
        "在无法核听的前提下逐条取舍：有语义的照常纠错翻译，确认无语义的重复/感叹以"
        " discard|<源序号> 显式丢弃；证据不足时保留原文并标低可信度，绝不能凭上下文编造或「还原」台词"
    ),
    "preceding_audibility_note": "",
    "paren_rule": (
        "本次没有音频，禁止把 ASR 文本改写成非语音声响括注；确信为幻觉就写 discard|<源序号> 丢弃，"
        "不确信就保留纠错并在 note 标记「疑似幻觉」。系统语音等可转写文本仍按话语处理。"
    ),
}


def _modal_params(profile: TranslationProfile) -> Dict[str, str]:
    return _AUDIO_MODAL_PARAMS if profile.use_audio else _TEXT_MODAL_PARAMS


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def ensure_csv_block_headers(
    text: str,
    *,
    output_header: str = OUTPUT_CSV_HEADER,
    include_output_blocks: bool = True,
) -> str:
    """Insert canonical headers into line-anchored live CSV input blocks."""

    rendered = text or ""
    blocks = [("asr_result", ASR_RESULT_CSV_HEADER)]
    if include_output_blocks:
        blocks.extend((("singles", output_header), ("translated", output_header)))
    for tag, header in blocks:
        pattern = re.compile(
            rf"(?m)(^<{tag}>[ \t]*\r?\n)(?!{re.escape(header)}[ \t]*\r?$)"
        )
        rendered = pattern.sub(rf"\1{header}\n", rendered)
    return rendered


def _looks_numeric(value: str) -> bool:
    try:
        float(value.strip())
    except ValueError:
        return False
    return True


def _add_start_to_example_outputs(text: str) -> str:
    """Mirror example ASR starts into BasicA singles/translated rows."""

    starts: dict[str, str] = {}
    active = ""
    rendered: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        opening = re.fullmatch(r"<(asr_result|singles|translated)>", stripped)
        closing = re.fullmatch(r"</(asr_result|singles|translated)>", stripped)
        if opening:
            active = opening.group(1)
            rendered.append(line)
            continue
        if closing:
            active = ""
            rendered.append(line)
            continue
        if active == "asr_result" and stripped and not stripped.startswith("source_id|"):
            parts = stripped.split("|", 4)
            if len(parts) >= 4 and _looks_numeric(parts[1]):
                starts[parts[0].strip()] = parts[1].strip()
        if active in {"singles", "translated"}:
            if stripped == OUTPUT_CSV_HEADER:
                rendered.append(line.replace(OUTPUT_CSV_HEADER, OUTPUT_CSV_HEADER_WITH_START))
                continue
            parts = line.split("|")
            if len(parts) >= 9 and parts[0].strip().lower() in {"sub", "insert"}:
                first = parts[1].strip().split(",", 1)[0]
                start = starts.get(first)
                if start is None and parts[0].strip().lower() == "insert":
                    start = first
                if start is not None:
                    parts.insert(2, start)
                    line = "|".join(parts)
        rendered.append(line)
    return "\n".join(rendered)


def _search_results_usage(*, evidence_pack_mode: bool) -> str:
    name = (
        "fragment_evidence_pack_usage_v1.md"
        if evidence_pack_mode
        else "fragment_search_results_usage_v1.md"
    )
    return load_prompt_template(name).strip()


def compose_correction_system(
    profile: TranslationProfile = DEFAULT_PROFILE,
    *,
    tier: CapabilityTier = CapabilityTier.CAPABLE,
    variant: str | None = None,
    evidence_pack_mode: bool = False,
    extra_style: str = "",
    common_mistakes_block: str = "",
) -> str:
    """Assemble the correction system prompt for a profile.

    ``evidence_pack_mode`` swaps the injected-search usage fragment for the
    evidence-pack variant (fast mode with a completed search loop).

    The prompt *set* comes from a :class:`CorrectionVariant`. ``tier`` classifies
    the answering endpoint and picks the default variant; pass ``variant`` to
    override by name (A/B/C…). The variant bundles the merge fragments, the
    reasoning-bounded flag, and the contract clauses that used to be tier
    ternaries; the tier-independent translated discipline lives in
    ``fragment_translated_common_v1.md``, prepended to every variant.
    """

    v = resolve_variant(variant, tier)
    params = dict(_modal_params(profile))
    jsonl_output = v.output_format == "jsonl"
    if jsonl_output:
        for key in ("discard_insert_clause", "noisy_span_handling", "paren_rule"):
            params[key] = (
                params[key]
                .replace("discard|<源序号>", "type=discard 的 JSON object")
                .replace("discard 行", "discard object")
            )
    if profile.use_audio:
        video_addendum = (
            load_prompt_template("fragment_corr_role_video_v1.md").strip()
            if profile.use_video
            else ""
        )
        role_block = load_prompt_template(
            "fragment_corr_role_audio_v1.md", video_role_addendum=video_addendum
        ).strip()
        if jsonl_output:
            role_block = role_block.replace("中文字幕类 CSV", "中文字幕 JSONL")
        goals_correction = load_prompt_template("fragment_goals_correction_audio_v1.md")
    else:
        role_block = load_prompt_template("fragment_corr_role_text_v1.md").strip()
        if jsonl_output:
            role_block = role_block.replace("中文字幕类 CSV", "中文字幕 JSONL")
        goals_correction = load_prompt_template("fragment_goals_correction_text_v1.md")

    if profile.external_injection or (
        profile.route == "text" and profile.level == "med"
    ):
        retrieval_block = load_prompt_template(
            "fragment_retrieval_injected_v1.md",
            search_results_usage=_search_results_usage(
                evidence_pack_mode=evidence_pack_mode
            ),
        ).strip()
    elif profile.native_search:
        retrieval_block = load_prompt_template("fragment_native_search_v1.md").strip()
    else:
        retrieval_block = ""

    if profile.route == "text":
        effort_name = (
            "fragment_effort_low_v1.md"
            if profile.level == "low"
            else "fragment_effort_deep_v1.md"
        )
        effort_block = load_prompt_template(effort_name).strip()
    else:
        effort_block = ""

    extra_style_block = (
        f"\n特殊翻译风格要求：\n{extra_style.strip()}\n" if extra_style.strip() else ""
    )
    mistakes_block = (
        common_mistakes_block.rstrip() + "\n" if common_mistakes_block.strip() else ""
    )
    weighted_char_count_rule = load_prompt_template(
        "fragment_weighted_char_count_v1.md"
    ).strip()
    examples_block = load_prompt_template(
        v.examples_fragment,
        judgment_basis_short=params["judgment_basis_short"],
        merge_connect_basis=params["merge_connect_basis"],
        noisy_span_handling=params["noisy_span_handling"],
    ).strip()
    if v.output_has_start and not jsonl_output:
        examples_block = _add_start_to_example_outputs(examples_block)
    # Variant-independent translated discipline + variant-selected merge
    # strategy. The weighted char-count algorithm is injected only via the
    # output contract; the common fragment keeps just the column discipline.
    merge_block = (
        load_prompt_template(
            "fragment_translated_common_jsonl_v1.md"
            if jsonl_output
            else "fragment_translated_common_v1.md"
        ).strip()
        + "\n\n"
        + load_prompt_template(
            v.merge_rules_fragment, speaker_basis=params["speaker_basis"]
        ).strip()
    )

    assembled = load_prompt_template(
        "correction_main_v1.md",
        role_block=role_block,
        goals_correction_block=goals_correction.strip(),
        goals_translation_block=load_prompt_template(
            "fragment_goals_translation_jsonl_v1.md"
            if jsonl_output
            else "fragment_goals_translation_v1.md",
            paren_rule=params["paren_rule"],
            granule_record_clause=v.granule_record_clause,
        ).strip(),
        extra_style_block=extra_style_block,
        common_mistakes_block=mistakes_block,
        retrieval_block=retrieval_block,
        csv_input_block=load_prompt_template(
            "fragment_csv_input_v1.md",
            csv_time_col_name=params["csv_time_col_name"],
            csv_time_note=params["csv_time_note"],
        ).strip(),
        output_contract_block=load_prompt_template(
            v.output_contract_fragment,
            reasoning_clause=reasoning_clause(
                correction_reasoning_depth(profile), bounded=v.reasoning_bounded
            ),
            insert_type_clause=params["insert_type_clause"],
            insert_position_clause=params["insert_position_clause"],
            insert_duration_clause=params["insert_duration_clause"],
            insert_note_clause=params["insert_note_clause"],
            weighted_char_count_rule=weighted_char_count_rule,
            output_csv_header=(
                OUTPUT_CSV_HEADER_WITH_START if v.output_has_start else OUTPUT_CSV_HEADER
            ),
            output_column_count=(10 if v.output_has_start else 9),
            output_start_clause=(
                "\n   - `start`：直接抄该行首源在 `<asr_result>` 中的 start；"
                "合并行抄首源 start；保留 1 位小数。"
                if v.output_has_start
                else ""
            ),
            translated_position_clause=v.translated_position_clause,
            translated_merge_rule=v.translated_merge_rule,
            pacing_merge_clause=v.pacing_merge_clause,
            singles_note_style=v.singles_note_style,
            note_gap_clause=v.note_gap_clause,
        ).strip(),
        advice_block=load_prompt_template("fragment_advice_v1.md").strip(),
        keep_block=load_prompt_template("fragment_keep_entries_v1.md").strip(),
        alignment_block=load_prompt_template("fragment_alignment_v1.md").strip(),
        window_block=load_prompt_template(
            "fragment_window_overlap_jsonl_v1.md"
            if jsonl_output
            else "fragment_window_overlap_v1.md",
            preceding_audibility_note=params["preceding_audibility_note"],
        ).strip(),
        merge_block=merge_block,
        hallucination_block=load_prompt_template(
            "fragment_hallucination_jsonl_v1.md"
            if jsonl_output
            else "fragment_hallucination_v1.md",
            discard_insert_clause=params["discard_insert_clause"],
        ).strip(),
        examples_block=examples_block,
        effort_block=effort_block,
    )
    return _collapse_blank_lines(assembled)


def compose_correction_user(
    profile: TranslationProfile = DEFAULT_PROFILE,
    *,
    general_context_json: str,
    window_context: str,
    entry_details: str,
    previous_advice: str,
    pre_round_notes: str,
    search_results: str,
    preceding_context_csv: str,
    current_asr_csv: str,
    current_asr_row_count: int,
    tier: CapabilityTier = CapabilityTier.CAPABLE,
    variant: str | None = None,
) -> str:
    v = resolve_variant(variant, tier)
    params = dict(_modal_params(profile))
    jsonl_output = v.output_format == "jsonl"
    if jsonl_output:
        for key in ("discard_insert_clause", "noisy_span_handling", "paren_rule"):
            params[key] = (
                params[key]
                .replace("discard|<源序号>", "type=discard 的 JSON object")
                .replace("discard 行", "discard object")
            )
    reminders_name = (
        "fragment_user_reminders_audio_v1.md"
        if profile.use_audio
        else "fragment_user_reminders_text_v1.md"
    )
    output_header = (
        OUTPUT_CSV_HEADER_WITH_START if v.output_has_start else OUTPUT_CSV_HEADER
    )
    rendered = load_prompt_template(
        v.user_template,
        general_context_json=general_context_json,
        window_context=window_context,
        entry_details=entry_details,
        previous_advice=previous_advice,
        verify_basis=params["verify_basis"],
        pre_round_notes=pre_round_notes,
        search_results=search_results,
        reminder_tail=load_prompt_template(reminders_name).strip(),
        preceding_context_csv=preceding_context_csv,
        current_asr_csv=current_asr_csv,
        current_asr_row_count=current_asr_row_count,
        merge_reminder=v.merge_reminder,
        mid_reminder_merge_rule=v.mid_reminder_merge_rule,
        singles_note_reminder=v.singles_note_reminder,
    )
    if v.output_has_start and not jsonl_output:
        rendered = (
            rendered.replace(OUTPUT_CSV_HEADER, output_header)
            .replace("固定 9 列", "固定 10 列")
            .replace("完整 9 列 header，再写 9 列", "完整 10 列 header，再写 10 列")
            .replace("完整 9 列 header", "完整 10 列 header")
            .replace("九列字段与顺序", "十列字段与顺序")
        )
    return ensure_csv_block_headers(
        rendered,
        output_header=output_header,
        include_output_blocks=not jsonl_output,
    )


def compose_correction_query_system(
    profile: TranslationProfile = DEFAULT_PROFILE,
    *,
    search_queries_rules: str,
    max_entries: int = 8,
    total_entries: int = 12,
) -> str:
    params = _modal_params(profile)
    if profile.use_audio:
        modal_slots = {
            "query_input_desc": (
                "本窗口对应的原始音频剪辑（前后含少量 padding）、当前窗口的 ASR 类 CSV、"
                "背景调查资料和此前窗口的累积建议"
            ),
            "query_suspect_desc": "结合音频定位可疑的 ASR 误听点并推断正确候选",
            "query_point_1": (
                "结合音频听清可疑专名的实际发音；对背景资料对不上的生僻假名串，"
                "考虑常用词连读/吞音变形的还原候选；query 中写出你推断的正确候选"
                "（可并列 2-3 个候选写法），不要照抄明显错误的 ASR 文本。"
            ),
            "query_time_note": "开始时间以剪辑音频的 0 秒为基准",
        }
    else:
        modal_slots = {
            "query_input_desc": (
                "当前窗口的 ASR 类 CSV、背景调查资料和此前窗口的累积建议"
                "（本次任务没有音频）"
            ),
            "query_suspect_desc": "结合上下文与发音相似度推断可疑的 ASR 误听点及正确候选",
            "query_point_1": (
                "结合上下文与发音相似度推断可疑专名的正确候选（含常用词连读/吞音变形的"
                "还原候选——此类候选未经音频验证，只能作为待定候选写进 query，不得据此断言）；"
                "query 中写出你推断的候选（可并列 2-3 个），不要照抄明显错误的 ASR 文本。"
            ),
            "query_time_note": "开始时间以本窗口第一条字幕为 0 秒基准",
        }
    return load_prompt_template(
        "correction_query_v2.md",
        search_queries_rules=search_queries_rules,
        reasoning_clause=reasoning_clause(correction_reasoning_depth(profile)),
        max_entries=max_entries,
        total_entries=total_entries,
        csv_time_col_name=params["csv_time_col_name"],
        **modal_slots,
    )


def compose_fast_round1_system(
    profile: TranslationProfile = DEFAULT_PROFILE,
    *,
    search_queries_rules: str,
    task_update_feedback_block: str = "",
    max_requested_entries: int = 8,
    max_keep_entries: int = 8,
    max_total_entries: int = 12,
) -> str:
    params = _modal_params(profile)
    if profile.use_video:
        media_desc = "整段原始音频及同区间低采样率视频画面的剪辑（首尾含少量 padding，同一剪辑文件）、"
        suspect_desc = "结合音频与画面定位可疑的 ASR 误听点并推断正确候选，判断哪些疑点需要联网查证。"
    elif profile.use_audio:
        media_desc = "整段原始音频的剪辑（首尾含少量 padding）、"
        suspect_desc = "结合音频定位可疑的 ASR 误听点并推断正确候选，判断哪些疑点需要联网查证。"
    else:
        media_desc = ""
        suspect_desc = "结合上下文与发音相似度推断可疑的 ASR 误听点及正确候选，判断哪些疑点需要联网查证。"
    return load_prompt_template(
        "fast_round1_v1.md",
        search_queries_rules=search_queries_rules,
        task_update_feedback_block=task_update_feedback_block,
        reasoning_clause=reasoning_clause(correction_reasoning_depth(profile)),
        max_requested_entries=max_requested_entries,
        max_keep_entries=max_keep_entries,
        max_total_entries=max_total_entries,
        fast_media_desc=media_desc,
        fast_suspect_desc=suspect_desc,
        csv_time_col_name=params["csv_time_col_name"],
        csv_time_note=params["csv_time_note"],
    )


def compose_fast_round1_user(
    *,
    extra_info: str,
    note_url_extracts: str,
    streamer_index: str,
    common_index: str,
    current_asr_csv: str,
    preinjected_entries: str = "",
    task_feedback_reminder: str = "",
) -> str:
    return ensure_csv_block_headers(load_prompt_template(
        "fast_round1_user_v1.md",
        extra_info=extra_info,
        note_url_extracts=note_url_extracts,
        streamer_index=streamer_index,
        common_index=common_index,
        preinjected_entries=preinjected_entries.strip() or "（无）",
        current_asr_csv=current_asr_csv,
        task_feedback_reminder=task_feedback_reminder,
    ))
