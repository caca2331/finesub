"""Named correction-prompt variants (Design A).

A *variant* bundles every choice that used to be a ``tier is BASIC`` branch in
``prompt_compose``: which merge fragments to load, whether the reasoning block
is bounded, whether the full ``<singles>`` block is required, and the handful of
short clauses injected into the shared output-contract / user-reminder
templates. Selecting a variant is decoupled from the endpoint's capability tier:
the tier still classifies the answering model, but it only picks a *default*
variant (``DEFAULT_VARIANT_FOR_TIER``). Callers — notably session_replay — may
override the variant by name to serve any registered prompt set
(``capableA``/``basicA``/``capableB``…).

Adding a variant is a registry entry (plus any new fragment file it names); no
branching code changes. Each variant is fully specified — no sparse inheritance
— so coupled clauses always move together and cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

from .config import CapabilityTier


@dataclass(frozen=True)
class CorrectionVariant:
    """One fully-specified correction prompt set."""

    name: str
    # Whole-fragment choices.
    examples_fragment: str
    merge_rules_fragment: str
    # Behavioural toggles.
    reasoning_bounded: bool
    # Whether validation requires the full-window <singles> block. A no-singles
    # variant sets this False and swaps in the singles-free templates below.
    require_full_singles: bool
    # Short clauses injected into the output-contract fragment.
    translated_position_clause: str
    translated_merge_rule: str
    pacing_merge_clause: str
    singles_note_style: str
    note_gap_clause: str
    # Short clauses injected into the user template.
    merge_reminder: str
    mid_reminder_merge_rule: str
    singles_note_reminder: str
    # Whole templates. Default to the singles-based pair; a structural variant
    # (e.g. C) swaps in singles-free versions. Kept as whole-file choices rather
    # than param holes because dropping singles is a pervasive, structural edit
    # (block structure, coverage clause, reasoning order) that would riddle a
    # shared template with conditionals.
    output_contract_fragment: str = "fragment_output_contract_v1.md"
    user_template: str = "correction_user_v2.md"
    # Small singles-referencing clauses in otherwise-shared fragments, gated so a
    # no-singles variant leaves no dangling reference to a block it never emits.
    insert_singles_clause: str = (
        "；**`<singles>` 禁止 `insert`**（漏识别没有源序号，无法进一一对应表）"
    )
    granule_record_clause: str = "singles 一律如实记录，取舍只发生在 translation。"
    # BasicA mirrors the input timeline by keeping ``start`` between position
    # and duration.
    output_has_start: bool = False
    # ``jsonl`` means one JSON object per physical line inside <translated>.
    # Input ASR remains the compact headered CSV shared by every variant.
    output_format: str = "csv"


# --- Shared clause bodies (verbatim from the pre-refactor tier ternaries) ---

_CAPABLE_A = CorrectionVariant(
    name="capableA",
    examples_fragment="fragment_examples_merge_v1.md",
    merge_rules_fragment="fragment_merge_rules_v1.md",
    reasoning_bounded=False,
    require_full_singles=True,
    translated_position_clause=(
        "translated 通常使用单源；同一句被切开时最多合并两个连续"
        "源序号（如 `3,4`）。三源仅限一种情况：filler 三明治——两段正句"
        "碎片夹一个 ≤3 字纯语气词/口吃碎片（如 `70,71,72` 中间为「えっと」）；"
        "其余三源及以上禁止"
    ),
    translated_merge_rule=(
        "合并后 `char_count` >20 或跨度 >4 秒的，**默认不合并、拆开输出**——"
        "软门槛是默认边界不是“尽量”；只有拆开会把一个不可分语义单元切碎时才可"
        "越过，越过须在 note 写明理由，且字数仍不得超过 36。多源合并绝超 7 秒"
        "必须拒绝，且**绝不能仅因相邻、连贯或话题相关就并**。"
        "filler 三明治三源行更严：合计 ≤4 秒且合并后 ≤16 字，不适用 36 字放宽。"
    ),
    pacing_merge_clause=(
        "1. 拟合并时先估算合并后的字数和跨度。字数 >20 或跨度 >4 秒的"
        "**默认不并、拆开输出**（软门槛是默认边界）；只有拆开会切碎不可分语义"
        "单元时才越过，字数不得超过 36。多源合并绝不得超过 7 秒；源数遵守 "
        "position 规则。单个源自身已超 7 秒时如实输出，不因该上限丢弃。\n"
        "2. 多源 position 用英文逗号连接，例如：\n"
        "   `sub|3,4|1.9|0.2|good morning|你好|high|2|ASR错分，同属问候短句`\n"
        "   本地时间轴取首源开始到末源结束；gap 取末源到下一源的间隔。"
    ),
    singles_note_style="singles 必须写简短取舍理由",
    note_gap_clause=(
        "涉及合并判断并提及 gap 时，必须写清方向和数值：判断后句写"
        "“本行 gap=Xs”；判断前句写“前一行 gap=Xs”。禁止用“正常/适中/较小”"
        "等相对词代替数值。"
    ),
    merge_reminder=(
        "多数源保持独立；合并严守 20 字/4 秒软门槛（默认边界，非“尽量”）"
        "与 7 秒硬上限，只有不可分语义单元可越过，字数不得超过 36。"
    ),
    mid_reminder_merge_rule=(
        "合并后超 20 字或超 4 秒**默认拒绝、拆开**；只有不可分语义单元可"
        "越过，字数不得超过 36；多源绝超 7 秒禁止（单源豁免）。"
    ),
    singles_note_reminder="note 只写简短取舍理由与五选一结论。",
)

_BASIC_A = CorrectionVariant(
    name="basicA",
    examples_fragment="fragment_examples_merge_basic_v1.md",
    merge_rules_fragment="fragment_merge_rules_basic_v1.md",
    reasoning_bounded=True,
    require_full_singles=True,
    translated_position_clause=(
        "translated 与 singles 一一对应、通常保持单源；仅词中切断的接回"
        "可用两个连续源序号（如 `5,6`），禁止三源及以上"
    ),
    translated_merge_rule=(
        "除词中切断的接回外不得合并——相邻、相关、未完句都不构成接回；"
        "接回行跨度不得超过 7 秒。"
    ),
    pacing_merge_clause=(
        "1. 唯一允许的多源行是词中切断的接回：position 用英文逗号连接"
        "两个连续源序号，例如：\n"
        "   `sub|5,6|3.3|0.9|ごめんごめん、怖がらないで|抱歉抱歉，别害怕|high|7.5|词中接回`\n"
        "   本地时间轴取首源开始到末源结束；gap 取末源到下一源的间隔。\n"
        "2. 单个源自身超长时如实输出，不做丢弃。"
    ),
    singles_note_style="singles 的理由可极简或省略",
    note_gap_clause=(
        "若写合并判断并引用 gap，须写清方向和数值（判断后句“本行 gap=Xs”；"
        "判断前句写“前一行 gap=Xs”），不要用“正常/较小”等相对词。"
    ),
    merge_reminder=(
        "translated 与 singles 一一对应；多源行仅允许词中切断的接回"
        "（边界无标点无空格、拼回同一个词），其余一律保持单源，"
        "不做任何判断型合并。"
    ),
    mid_reminder_merge_rule=(
        "仅词中切断的接回可合并（两个连续源、跨度 ≤7 秒），其余一律单源。"
    ),
    singles_note_reminder="note 以五选一结论收束即可，理由可极简，不强制附 gap 数值。",
    output_has_start=True,
)


# capableB = capableA's merge behaviour, but without the full-window <singles>
# pass: it emits <translated> directly (token savings; the 5-failure structural
# retries in the v51 3.5 round were all singles-coverage errors). The no-singles
# templates carry the structural difference; everything else is inherited.
_CAPABLE_B = replace(
    _CAPABLE_A,
    name="capableB",
    require_full_singles=False,
    examples_fragment="fragment_examples_merge_nosingles_v1.md",
    merge_rules_fragment="fragment_merge_rules_nosingles_v1.md",
    output_contract_fragment="fragment_output_contract_nosingles_v1.md",
    user_template="correction_user_nosingles_v1.md",
    insert_singles_clause="",
    granule_record_clause="取舍按上述三步在 translated 中直接处理。",
    translated_merge_rule=(
        "合并后 `char_count` >20 或跨度 >4 秒即越过**硬门槛**，原则上必须"
        "拒绝合并、拆开输出；唯一可特批的是同一个词或不可拆固定短语被源切断，"
        "且须在该变体规定的局部理由位置写明。即使特批，`char_count` >36 "
        "或跨度 >7 秒也触发**绝对门槛**，任何情况都必须拒绝。绝不能仅因相邻、"
        "连贯、同一句或话题相关就并。filler 三明治三源行更严：合计 ≤4 秒且"
        "合并后 ≤16 字，不适用任何特批。"
    ),
    pacing_merge_clause=(
        "1. 拟合并时先估算合并后的字数和跨度。字数 >20 或跨度 >4 秒即越过"
        "**硬门槛**，原则上必须拒绝；仅同一个词或不可拆固定短语被源切断可特批，"
        "并写明理由。字数 >36 或跨度 >7 秒是**绝对门槛**，任何多源合并都不得"
        "越过；单个源自身超限时如实输出，不因该门槛丢弃。源数遵守 position 规则。\n"
        "2. 多源 position 用英文逗号连接两个连续源序号。本地时间轴取首源开始"
        "到末源结束；gap 取末源到下一源的间隔。"
    ),
    merge_reminder=(
        "多数源保持独立；合并严守 20 字/4 秒硬门槛。只有同一个词或不可拆固定"
        "短语被源切断才可特批；36 字/7 秒为绝对门槛，任何情况不得越过。"
    ),
    mid_reminder_merge_rule=(
        "合并后超 20 字或超 4 秒即越硬门槛，原则上拒绝；仅词或不可拆固定短语"
        "被切断可特批。超 36 字或超 7 秒即越绝对门槛，任何情况禁止（单源豁免）。"
    ),
)

# capableC = capableB + inter-line reasoning comments at decision points. The
# model writes a ``# ...`` line immediately before the sub/discard row it
# governs, giving "think before you write" without the full singles pass.
# Gate: reasoning REQUIRED when merge ≥2 sources / discard / conf=low / exceeds
# the hard threshold; FORBIDDEN for pure single-source in-bounds high-conf rows.
_CAPABLE_C = replace(
    _CAPABLE_B,
    name="capableC",
    examples_fragment="fragment_examples_merge_nosingles_reasoning_v1.md",
    output_contract_fragment="fragment_output_contract_nosingles_reasoning_v1.md",
    user_template="correction_user_nosingles_reasoning_v1.md",
)

# BasicB freezes the v59 cross-tier experiment: capableB's no-singles merge
# behaviour with a start-bearing, headered CSV output for Flash Lite.
_BASIC_B = replace(
    _CAPABLE_B,
    name="basicB",
    output_has_start=True,
)

# BasicC keeps the capableC decision-reasoning behaviour but experiments with
# JSONL output. Reasoning is a separate ``type=reasoning`` object immediately
# before its governed row, so every physical line inside <translated> remains
# valid JSON without mixing local reasoning into subtitle records.
_BASIC_C = replace(
    _CAPABLE_C,
    name="basicC",
    output_has_start=True,
    output_format="jsonl",
    examples_fragment="fragment_examples_merge_nosingles_jsonl_reasoning_v1.md",
    merge_rules_fragment="fragment_merge_rules_nosingles_jsonl_reasoning_v1.md",
    output_contract_fragment="fragment_output_contract_nosingles_jsonl_reasoning_v1.md",
    user_template="correction_user_nosingles_jsonl_reasoning_v1.md",
)

VARIANTS: Dict[str, CorrectionVariant] = {
    _CAPABLE_A.name: _CAPABLE_A,
    _BASIC_A.name: _BASIC_A,
    _CAPABLE_B.name: _CAPABLE_B,
    _CAPABLE_C.name: _CAPABLE_C,
    _BASIC_B.name: _BASIC_B,
    _BASIC_C.name: _BASIC_C,
}

# The endpoint's capability tier only picks a *default* variant. Variant names
# are ``<tier><letter>`` — the tier is baked into the name (capableA/capableB
# target the capable tier / 3.6 or 3.5 Flash; basicB targets the basic tier /
# 3.5 Flash Lite as the production default; basicA remains the 1:1 control), so a
# variant is never further split by tier.
DEFAULT_VARIANT_FOR_TIER: Dict[CapabilityTier, str] = {
    CapabilityTier.CAPABLE: "capableC",
    CapabilityTier.BASIC: "basicB",
}


def resolve_variant(
    variant: str | None, tier: CapabilityTier = CapabilityTier.CAPABLE
) -> CorrectionVariant:
    """Resolve a variant by name, falling back to the tier's default.

    A ``None`` name means "use whatever the answering tier maps to" (production
    behaviour). An explicit name overrides the tier — an unknown name is a hard
    error so a typo never silently serves the wrong prompt set.
    """

    if variant is None:
        variant = DEFAULT_VARIANT_FOR_TIER[tier]
    try:
        return VARIANTS[variant]
    except KeyError:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown correction variant {variant!r}; known: {known}")
