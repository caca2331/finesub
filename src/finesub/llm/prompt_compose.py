"""Fragment selection and assembly for correction / fast-mode prompts.

Prompt text lives in ``prompt_templates/*.md``; this module only decides which
fragments a given :class:`TranslationProfile` gets and stitches them into the
correction skeletons. Fragment inventory and the slot selection table are
documented in ``docs/llm_prompts.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from string import Template
from typing import Callable, Dict, Mapping, Sequence, Tuple

from .chunking import ASR_RESULT_CSV_HEADER, Seam, asr_result_header
from .routing.config import CapabilityTier, KB_TRANSFER_MAX_ENTRIES
from .output_protocol import correction_output_header
from .example_builder import EXAMPLES_DIR, EXAMPLES_ZH_DIR, build_examples
from .prompt_constants import Thresholds
from .prompt_tables import PROMPT_TEMPLATE_DIR, prompt_phrases, task_mode_texts
from .prompt_variants import CorrectionVariant, grouped_variant, resolve_variant
from .routing.profiles import DEFAULT_PROFILE, TranslationProfile

PROMPT_VERSION = "zh-subtitle-correction-csv-v114"
#: Three passes' pass 2 (the correction window over lines pass 1 grouped) versions its prompt on its own, so a
#: change to it does not void the windows single-pass tasks already delivered, nor the other way round.
GROUPED_PROMPT_VERSION = "zh-subtitle-correction-3pass-v20"

# No session asks for a visible reasoning block: in real zero-thinking calls of capable models the thinking never
# moved into it (docs/llm_followups.md), and a reasoning block asked for in the open gets Claude's
# `reasoning_extraction` refusal -- the correction window dropped it in v107–v109, the other sessions on 2026-11-08.
# basicA alone opens its delivery with `<decisions>` (v109, its own output contract and user template): a list of
# conclusions, not a derivation. Flash Lite thinks zero tokens at low, and without such a block its whole-window
# repair re-sent the same wrong answer up to the attempt cap where the old block's restated rule fixed it in one
# round (docs/llm_design_notes.md, the efficiency decisions). A `<reasoning>` a model still writes on its own is
# tolerated: parsers read only the blocks they need (`output_tags`, `proposals.strip_reasoning`).


def render_knowledge_indexes(indexes: Mapping[str, str] | None) -> str:
    """The ``<{category}_index>`` blocks every knowledge-bearing prompt carries.

    One rendered block instead of the ``streamer_index`` / ``common_index``
    pair the templates used to name. Four templates spelled that pair out by
    hand, so adding a category meant editing prompt TEXT -- which is why a
    category could never be added from data alone
    (2026-09-07; the wiring now: `docs/knowledge.md`, 「加一个类别就是两步」).

    The tag names stay ``<{category}_index>``: `exchange_metadata` counts each
    block's tokens by extracting it from the rendered user text, and that keeps
    working per category.

    Empty means ``（空）``, decided here rather than at each call site. Five
    sites wrote ``or "（空）"`` and one (fast round 1) passed the empty string
    through -- a divergence nobody chose.
    """

    from .knowledge.node.presets import MATCHABLE_CATEGORIES

    supplied = indexes or {}
    blocks = []
    for category in MATCHABLE_CATEGORIES:
        body = str(supplied.get(category) or "").strip() or "（空）"
        blocks.append(f"<{category}_index>\n{body}\n</{category}_index>")
    return "\n".join(blocks)


def render_knowledge_ops_contract(
    allow_categories: Sequence[str], *, split: bool = False, whole_batch: bool = False
) -> str:
    """The proposal grammar, with its ``category`` enum bound to the grant.

    Every caller of ``fragment_knowledge_output_v1`` also hands an
    ``allow_categories`` to ``translate_model_proposals``; rendering the enum
    from that same tuple is what stops the prompt and the gate from
    disagreeing. They did: the template named `streamer|common` outright, so a
    new category reached the store, the index and the structure table and was
    still declared illegal three lines later -- and the two repair sessions,
    which have always applied with every category, were reading an enum that
    named two of them (review 2026-09-07, second round).

    ``whole_batch``: the caller applies with ``invalid_ops="rollback"``, so a
    batch-wide problem sinks the whole batch, not just its ops.
    """

    from .knowledge.node.render import render_writable_categories

    names = tuple(allow_categories)
    # `split_entry` is offered only when an entry in front of the model is
    # asked to compress and may split (plan kb-entry-size): the op is one
    # line of JSON and compressing is many, so a standing offer would win
    # over the compression the notice asks for first.
    return load_prompt_template(
        "fragment_knowledge_output_v1.md",
        strict=True,
        split_op_row=load_prompt_template("fragment_knowledge_split_op_row_v1.md") if split else "",
        split_op_note=load_prompt_template("fragment_knowledge_split_op_note_v1.md") if split else "",
        term_line_rules=render_term_line_rules(),
        rejected_rows=prompt_phrases("knowledge_rejects")["whole_batch" if whole_batch else "one_by_one"],
        writable_categories=render_writable_categories(names),
        # The "do not concatenate two enums" counter-example has to name a real
        # category to land, and a hand-written one is the same stale copy in
        # miniature.
        concat_counterexample=f"{names[0] if names else 'common'}|游戏",
    ).strip()


def render_term_line_rules() -> str:
    """The four-column term-line rules, shared by the full proposal grammar
    and the share conflict repair's own short op section (cold read Q7: that
    session may only rewrite existing lines, so it does not get the grammar)."""

    return load_prompt_template("fragment_knowledge_term_lines_v1.md").strip()


def mode_texts(variant: CorrectionVariant | str | None, tier: CapabilityTier = CapabilityTier.CAPABLE) -> Dict[str, str]:
    """`task_mode_texts` for a correction prompt set (`CorrectionVariant.correct_only`)."""

    if not isinstance(variant, CorrectionVariant):
        variant = resolve_variant(variant, tier)
    return task_mode_texts(variant.correct_only)


@lru_cache(maxsize=None)
def _template(name: str) -> Template:
    """A template file, read once per process like the TOML tables in `prompt_tables`: a run renders the same few
    dozen files once per window and session, and a template edited mid-run would mix two prompt versions."""

    return Template((PROMPT_TEMPLATE_DIR / name).read_text(encoding="utf-8"))


def load_prompt_template(name: str, *, strict: bool = False, **values: object) -> str:
    """Render one template file with ``safe_substitute``.

    ``strict`` checks placeholder coverage at the TEMPLATE layer -- every
    identifier the template declares must be supplied -- and never by scanning
    the rendered text: injected material (knowledge entries, user-supplied
    repair text) may legally contain ``$WORD``. Knowledge-session assembly
    always passes strict; a missed parameter raises instead of shipping a
    literal ``$term_line_rules`` to the model."""

    template = _template(name)
    defaults = {"prompt_version": PROMPT_VERSION}
    defaults.update({key: str(value) for key, value in values.items()})
    if strict:
        missing = sorted(set(template.get_identifiers()) - set(defaults))
        if missing:
            raise PromptAssemblyError(
                f"prompt template {name} rendered without values for: "
                + ", ".join(missing)
            )
    return template.safe_substitute(defaults)


# Phrase-level parameters, one layer per switch axis (docs/llm_prompts.md). Structural
# differences stay whole fragment files ("变体优先做整文件"). Every key belongs
# to exactly one layer; phrases that used to weld several axes together --
# `verify_basis` said "音频、背景资料和搜索结果" regardless of whether audio or
# search results were actually coming -- are now composed from per-layer pieces
# by `_modal_params`, so the wording can only name inputs that really arrive.
_MEDIA_PARAMS: Dict[str, str] = {
    "csv_time_col_name": "剪辑内开始时间",
    "csv_time_note": "`剪辑内开始时间` 以你收到的剪辑音频的 0 秒为基准，可直接用来在剪辑中定位对应语音。",
    "correction_basis": "重听",
    "speaker_basis": "音视频/语境",
    "insert_type_clause": "；不要输出其他 type 值",
    "insert_position_clause": "",
    "insert_duration_clause": "",
    "insert_note_clause": "",
    "audio_fill_clause": "${goals_audio_fill}",
    "grain_empty_audio_clause": "${goals_grain_empty_audio}",
    "discard_insert_clause": (
        "\n5. 丢弃取舍：打算以 discard|<源序号> 丢弃之前，先重听该区间音频——"
        "能辨识出实词或短语的，应修正后保留，而不是整体丢弃；只有音频本身"
        "也没有语义内容（纯哭声、喘息、感叹）时才写 discard 行丢弃。"
    ),
    "noisy_span_handling": (
        "重听该区间逐句处理：能听出实词、短语的，按听到的内容重写并按停顿拆条；"
        "确认只是喊叫、喘息等无语义感叹的，以 discard|<源序号> 显式丢弃"
    ),
    "preceding_audibility_note": (
        "最贴近窗口的前文行可能落在剪辑开头几秒内、其语音可听，但它们同样是只读背景，"
        "不要为其输出字幕。"
    ),
    "voice_clause": "\n声音（音色、口吻）可以佐证，但仍以台词内容为主。",
    "drop_listen_clause": "\n判成 `drop` 之前先重听那一行：能听出实词或短语的，修正后保留。",
    "paren_rule": (
        "括注采用正证据门槛：只有音频中能独立确认存在实际的非语音声响（有声响、没有"
        "可转写话语）时，才可写极简中性括注（如「（提示音）」），不夹带解释；若音频中"
        "没有对应声响或无法确认声响性质，不得把文本改写成括注。可听见的系统语音仍按话语处理。"
    ),
}
_TEXT_MEDIA_PARAMS: Dict[str, str] = {
    "csv_time_col_name": "窗口内开始时间",
    "csv_time_note": "`窗口内开始时间` 以本窗口起点为 0 秒基准（第一条不一定从 0 开始），仅用于把握说话节奏与间隔。",
    "correction_basis": "音近与上下文推断",
    "speaker_basis": "语境/文本线索",
    "insert_type_clause": "；不要输出其他 type 值",
    "insert_position_clause": "",
    "insert_duration_clause": "",
    "insert_note_clause": "",
    "audio_fill_clause": "",
    "grain_empty_audio_clause": "",
    "discard_insert_clause": "",
    "noisy_span_handling": (
        "在无法核听的前提下逐条取舍：有语义的照常${work_compact}，确认无语义的重复/感叹以"
        " discard|<源序号> 显式丢弃；证据不足时保留原文并标低可信度，绝不能凭上下文编造或「还原」台词"
    ),
    "preceding_audibility_note": "",
    "voice_clause": "",
    "drop_listen_clause": "",
    "paren_rule": (
        "本次没有音频，禁止把 ASR 文本改写成非语音声响括注；确信为幻觉就写 discard|<源序号> 丢弃，"
        "不确信就保留纠错并在 note 标记「疑似幻觉」。系统语音等可转写文本仍按话语处理。"
    ),
}


# media=video is its own layer on top of audio, not a role addendum bolted onto
# an otherwise audio-worded prompt (docs/llm_prompts.md): once frames are attached, the
# phrases that enumerate what to judge from have to name them.
_VIDEO_PARAM_OVERRIDES: Dict[str, str] = {
    "noisy_span_handling": (
        "重听该区间音频、必要时看画面逐句处理：能听出实词、短语的，按听到的内容重写"
        "并按停顿拆条；确认只是喊叫、喘息等无语义感叹的，以 discard|<源序号> 显式丢弃"
    ),
}


# The retrieval layer contributes only the material it actually injects.
_VERIFY_BASIS_MEDIA = {"text": "上下文", "audio": "音频", "video": "音画"}
_VERIFY_BASIS_RETRIEVAL = {
    "local": ["背景资料", "搜索结果"],
    "native": ["你自己检索到的资料"],
    "none": [],
}
# What the correction window is actually handed, named per axis. "背景资料" is
# the background block every window's user message carries: the preparation's
# context pack (it runs under retrieval=none too), or the user's notes in fast mode.
_INPUT_INVENTORY_MEDIA = {
    "text": "ASR 源字幕片段",
    "audio": "音频、ASR 源字幕片段",
    "video": "音频与视频画面、ASR 源字幕片段",
}


def _join_cn(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "、".join(parts[:-1]) + "和" + parts[-1]


def _verify_basis(profile: TranslationProfile) -> str:
    """"Cross-check against ..." -- named from what this call really receives.

    Composed rather than written out per combination: a profile with no
    retrieval must not be told to cross-check against search results it will
    never see (injection reality, plan §5.2). These helpers phrase the
    *correction window's* prompt, so the media layer reads
    ``correction_media`` (model-routing v2).
    """

    return _join_cn(
        [
            _VERIFY_BASIS_MEDIA[profile.correction_media],
            *_VERIFY_BASIS_RETRIEVAL[profile.retrieval],
        ]
    )


def _input_inventory(profile: TranslationProfile, *, knowledge_enabled: bool) -> str:
    """The "you will be given ..." list in the role fragments."""

    parts = [_INPUT_INVENTORY_MEDIA[profile.correction_media], "背景资料"]
    if knowledge_enabled:
        parts.append("知识库词条")
    if profile.continuity == "serial":
        parts.append("此前窗口的累积建议台账")
    parts.append("本窗只读前文 ASR")
    return _join_cn(parts)


def _background_sources(*, knowledge_enabled: bool) -> str:
    """The injected material the "trust your ears" rules argue against."""

    return _join_cn(["背景资料", *(["知识库词条"] if knowledge_enabled else [])])


def _modal_params(
    profile: TranslationProfile, *, knowledge_enabled: bool = True
) -> Dict[str, str]:
    """Merge the per-axis layers into the substitution map."""

    params = dict(
        _MEDIA_PARAMS if profile.correction_use_audio else _TEXT_MEDIA_PARAMS
    )
    if profile.correction_use_video:
        params.update(_VIDEO_PARAM_OVERRIDES)
    params["verify_basis"] = _verify_basis(profile)
    params["input_inventory"] = _input_inventory(
        profile, knowledge_enabled=knowledge_enabled
    )
    return params


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


class PromptAssemblyError(RuntimeError):
    """A rendered prompt still carries an unresolved placeholder."""


# ``$``-substitutions go through ``Template.safe_substitute``, which leaves an
# unknown key in place instead of raising. That is what lets a renamed
# parameter ship a literal "$verify_basis" to the model, so the finished text
# is checked here (docs/llm_prompts.md).
_UNRESOLVED_TOKEN = re.compile(r"\$(?:\{\w+\}|[A-Za-z_]\w*)")


def assert_fully_substituted(text: str, *, what: str) -> str:
    leftovers = sorted(set(_UNRESOLVED_TOKEN.findall(text or "")))
    if leftovers:
        raise PromptAssemblyError(
            f"{what} still contains unresolved placeholders: " + ", ".join(leftovers)
        )
    return text


# --- Slot registry -----------------------------------------------------------
# Which fragment fills a switch-driven slot is a table, not a chain of ifs
# (docs/llm_prompts.md). Each rule carries the blocks the fragment assumes will be
# injected alongside it (``requires``); rendering asserts they really are, which
# is the injection-reality principle made structural rather than reviewed.


@dataclass(frozen=True)
class PromptContext:
    """Everything a slot predicate is allowed to look at."""

    profile: TranslationProfile
    knowledge_enabled: bool = True
    #: an agentic window: it searches with its own tools, no block is filled in
    self_search: bool = False

    @property
    def can_request(self) -> bool:
        """A dropped entry can be asked for again in a later window
        (`task_graph.window_requests`): harness search, or a readable base --
        ``knowledge_enabled`` here already means the index is readable."""

        return self.profile.external_injection or self.knowledge_enabled

    @property
    def keeps_entries(self) -> bool:
        """The window answers with ``<keep_entries>``: pruning is asked for
        only where it is honoured -- something injected to prune, a later
        round that can ask it back, and a chain to carry it."""

        return self.knowledge_enabled and self.can_request and self.profile.continuity == "serial"

    @property
    def injected_blocks(self) -> frozenset[str]:
        """The input blocks this call will really carry."""

        blocks = {"asr_result", "preceding_context"}
        if self.profile.continuity == "serial":
            # Parallel dispatch retires the chained advice ledger; nothing is
            # accumulated across windows to inject (plan A.3/A.7).
            blocks.add("next_advice_ledger")
        if self.profile.correction_use_audio:
            blocks.add("audio")
        if self.profile.correction_use_video:
            blocks.add("video")
        if self.profile.external_injection:
            blocks.add("search_results")
        if self.can_request:
            blocks.add("window_notes")
        if self.profile.retrieval != "none":
            blocks.add("context_pack")
        if self.knowledge_enabled:
            blocks.add("entry_details")
        return frozenset(blocks)


@dataclass(frozen=True)
class SlotRule:
    """One candidate fragment for a slot, with its guard and its assumptions."""

    when: Callable[[PromptContext], bool]
    fragment: str = ""
    requires: Tuple[str, ...] = ()


class SlotRegistry:
    """Slot -> ordered rules; the first matching rule wins."""

    def __init__(self, rules: Mapping[str, Sequence[SlotRule]]) -> None:
        self._rules = {slot: tuple(items) for slot, items in rules.items()}

    def resolve(self, slot: str, context: PromptContext) -> SlotRule:
        for rule in self._rules[slot]:
            if rule.when(context):
                missing = [
                    block
                    for block in rule.requires
                    if block not in context.injected_blocks
                ]
                if missing:
                    raise PromptAssemblyError(
                        f"slot {slot!r} selected {rule.fragment or '(empty)'}, which "
                        f"assumes {', '.join(missing)} will be injected, but this "
                        f"context injects only {sorted(context.injected_blocks)}"
                    )
                return rule
        raise PromptAssemblyError(f"no rule matched slot {slot!r}")

    def slots(self) -> Tuple[str, ...]:
        return tuple(self._rules)


_ALWAYS: Callable[[PromptContext], bool] = lambda _context: True

CORRECTION_SLOTS = SlotRegistry(
    {
        # The retrieval-usage note ships only where its block can arrive. The
        # text-med special case (b2b7feb) documented an input that never showed
        # up; `requires` is what now makes that shape impossible to reintroduce.
        "retrieval": (
            SlotRule(
                when=lambda c: c.profile.external_injection and c.self_search,
                fragment="fragment_retrieval_self_v1.md",
            ),
            SlotRule(
                when=lambda c: c.profile.external_injection,
                fragment="fragment_retrieval_injected_v1.md",
                requires=("search_results",),
            ),
            SlotRule(
                when=lambda c: c.profile.native_search,
                fragment="fragment_native_search_v1.md",
            ),
            SlotRule(when=_ALWAYS),
        ),
        # Effort wording is for profiles that work without injected material;
        # efficiency gets the cheap variant of it.
        "effort": (
            SlotRule(when=lambda c: c.profile.external_injection),
            SlotRule(
                when=lambda c: c.profile.difficulty == "efficiency",
                fragment="fragment_effort_low_v1.md",
            ),
            SlotRule(when=_ALWAYS, fragment="fragment_effort_deep_v1.md"),
        ),
        "keep_entries": (
            SlotRule(
                when=lambda c: c.keeps_entries,
                fragment="fragment_keep_entries_v1.md",
                requires=("entry_details",),
            ),
            SlotRule(when=_ALWAYS),
        ),
        # The advice ledger is the serial chain's continuity; a parallel run
        # accumulates nothing across windows, so neither the ledger input nor
        # the <next_advice> output may be described (plan A.7).
        "advice": (
            SlotRule(
                when=lambda c: c.profile.continuity == "serial",
                fragment="fragment_advice_v1.md",
                requires=("next_advice_ledger",),
            ),
            SlotRule(when=_ALWAYS),
        ),
        "role": (
            SlotRule(
                when=lambda c: c.profile.correction_media == "text",
                fragment="fragment_corr_role_text_v1.md",
            ),
            SlotRule(
                when=_ALWAYS,
                fragment="fragment_corr_role_audio_v1.md",
                requires=("audio",),
            ),
        ),
        "goals_correction": (
            SlotRule(
                when=lambda c: c.profile.correction_media == "text",
                fragment="fragment_goals_correction_text_v1.md",
            ),
            SlotRule(
                when=_ALWAYS,
                fragment="fragment_goals_correction_audio_v1.md",
                requires=("audio",),
            ),
        ),
        "user_reminders": (
            SlotRule(
                when=lambda c: c.profile.correction_media == "text",
                fragment="fragment_user_reminders_text_v1.md",
            ),
            SlotRule(
                when=_ALWAYS,
                fragment="fragment_user_reminders_audio_v1.md",
                requires=("audio",),
            ),
        ),
        "video_role_addendum": (
            SlotRule(
                when=lambda c: c.profile.correction_use_video,
                fragment="fragment_corr_role_video_v1.md",
                requires=("video",),
            ),
            SlotRule(when=_ALWAYS),
        ),
        # Audio's "trust your ears over the injected material" rules only make
        # sense where injected material exists to conflict with.
        "background_conflict": (
            SlotRule(
                when=lambda c: c.profile.correction_use_audio
                and (c.profile.retrieval != "none" or c.knowledge_enabled),
                fragment="fragment_background_conflict_v1.md",
                requires=("audio",),
            ),
            SlotRule(when=_ALWAYS),
        ),
    }
)


def render_slot(
    slot: str, context: PromptContext, **values: object
) -> str:
    """Resolve a slot for this context and render its fragment ("" if none)."""

    rule = CORRECTION_SLOTS.resolve(slot, context)
    if not rule.fragment:
        return ""
    return load_prompt_template(rule.fragment, **values).strip()


REPAIR_ROUND_FRAGMENT = "fragment_repair_round_v1.md"


def _dropped_columns_clause(output_has_start: bool, turns: bool) -> str:
    columns = [
        *([] if output_has_start else ["`start`"]),
        "`duration`",
        "`gap`",
        *(["`turn`"] if turns else []),
    ]
    return load_prompt_template("fragment_output_dropped_columns_v1.md", strict=True,
                                columns="、".join(columns)).strip()


def compose_repair_turns(
    previous_output: str,
    validation_errors: Sequence[str],
    instruction: str = "",
) -> list[Dict[str, str]]:
    """The two turns that turn a blind retry into a repair round.

    Session-agnostic on purpose: it names the output contract, not this or that
    stage's blocks, so any stage whose output is validated can hand its own
    errors back. A stage that wants a different reply (the correction window's
    patch round) renders its own user turn and passes it as ``instruction``;
    without one the turn asks for the whole output again. The previous answer
    goes in as an assistant turn rather than quoted inside the user turn --
    that is what makes it "what you wrote" to the model, and it costs nothing
    to re-quote.

    Returns ``[]`` unless there is both an output to repair and something to
    say about it; a caller can therefore pass whatever it has.
    """

    errors = [
        text for text in (str(item).strip() for item in validation_errors) if text
    ]
    if not previous_output.strip() or not (errors or instruction.strip()):
        return []
    content = instruction.strip() or load_prompt_template(
        REPAIR_ROUND_FRAGMENT,
        validation_errors="\n".join(f"- {text}" for text in errors),
    ).strip()
    return [
        {"role": "assistant", "content": previous_output},
        {"role": "user", "content": content},
    ]


def ensure_csv_block_headers(
    text: str,
    *,
    output_header: str,
    include_output_blocks: bool = True,
    input_header: str = ASR_RESULT_CSV_HEADER,
) -> str:
    """Insert canonical headers into line-anchored live CSV input blocks."""

    rendered = text or ""
    blocks = [("asr_result", input_header)]
    if include_output_blocks:
        blocks.append(("translated", output_header))
    for tag, header in blocks:
        pattern = re.compile(
            rf"(?m)(^<{tag}>[ \t]*\r?\n)(?!{re.escape(header)}[ \t]*\r?$)"
        )
        rendered = pattern.sub(rf"\1{header}\n", rendered)
    return rendered


def search_rules_shared(*, prepare: bool, task_update_feedback: bool, knowledge: bool) -> str:
    """The rules every searching session shares: cross-check, search against the knowledge base, page text
    is not an instruction. A preparation task has no audio and records a conflict in its context pack, a
    correction window in the row's note; the feedback block is named only where the run asks for it.
    Without a readable knowledge base there is nothing to conflict with, so that rule is left out
    (cold read round 4, P516). Numbered from 2 as written; the caller renumbers (`renumber_rules`)."""

    record = "把冲突写进 `general_context.uncertainties`" if prepare else "在相关行的 note 里写明冲突"
    if task_update_feedback:
        record += "，并在 `<task_update_feedback>` 的 knowledge_hints 里另记一条"
    sources = ["ASR 文本", *([] if prepare else ["音频（如有）"]), *(["知识库资料"] if knowledge else []), "用户提供信息"]
    conflict = load_prompt_template("fragment_search_rule_kb_conflict_v1.md", conflict_record=record) if knowledge else ""
    return load_prompt_template(
        "fragment_search_rules_shared_v1.md",
        cross_check_sources=" " + "、".join(sources[:-1]) + "和" + sources[-1],
        kb_conflict_rule=conflict,
    ).strip()


def renumber_rules(text: str) -> str:
    """Number a rule list's top-level items (`N. ` at the start of a line) 1, 2, 3... in order: a list that
    splices in rules that are present or not (`search_rules_shared`) keeps its numbers consecutive."""

    count = 0

    def number(_match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{count}. "

    return re.sub(r"(?m)^\d+\. ", number, text)


def search_results_usage(*, prepare: bool, task_update_feedback: bool, knowledge: bool, self_search: bool = False) -> str:
    """The search-result rules around the shared ones (`search_rules_shared`). ``self_search``: an agentic
    window searches with its own tools, so the rules about a harness-filled `<search_results>` block give
    way to its own source and coverage rules."""

    return renumber_rules(load_prompt_template(
        "fragment_self_search_usage_v1.md" if self_search else "fragment_search_results_usage_v1.md",
        shared_rules=search_rules_shared(prepare=prepare, task_update_feedback=task_update_feedback, knowledge=knowledge),
    ).strip())


#: The `turn` values beyond 换人 / 可并 a keyed task's prompt explains, in the order it names them.
_KEYED_FRAGMENTS = {Seam.TOGETHER: "fragment_turn_together_v1.md", Seam.BACK: "fragment_turn_back_v1.md"}


def _turn_seams(keyed: bool) -> list[str]:
    return [Seam.CHANGE.value, *(seam.value for seam in _KEYED_FRAGMENTS if keyed)]


def _turn_words(keyed: bool, cloud: bool = False) -> Dict[str, str]:
    """What the `turn` sentences say about the task's labels. ``turn_values``: the format line's turn cell
    (换人, or 换人、同时、换回); ``turn_marks``: what no merge may cross (「换人」, or 「换人」「同时」或「换回」);
    ``turn_more``: the sentences on 同时 and 换回; ``turn_source``: who gave the labels (the acoustic layer, or
    with ``cloud`` a cloud task's transcription; a local window keyed only by the overlap lane is acoustic)."""

    seams = _turn_seams(keyed)
    quoted = [f"「{value}」" for value in seams]
    return {
        "turn_values": "、".join(seams),
        "turn_marks": quoted[0] if len(quoted) == 1 else "".join(quoted[:-1]) + "或" + quoted[-1],
        "turn_more": "".join(load_prompt_template(name).strip() for name in _KEYED_FRAGMENTS.values()) if keyed else "",
        "turn_source": "转写模型" if cloud else "声学层",
    }


def _turn_fragment(name: str, on: bool, keyed: bool, cloud: bool = False) -> str:
    """A `turn` sentence where the task has speaker labels and the set has the sentence at all."""

    return load_prompt_template(name, **_turn_words(keyed, cloud)).strip() if on and name else ""


def _with_mode(params: Mapping[str, str], mode: Mapping[str, str]) -> Dict[str, str]:
    """The modal phrases with the task mode's words filled in (``noisy_span_handling`` names the work)."""

    return {key: Template(value).safe_substitute(mode) if "${" in value else value for key, value in params.items()}


def _hole_fragment(name: str, thresholds: Thresholds, mode: Mapping[str, str]) -> str:
    """A paragraph-level hole of the variant's, rendered with the merge thresholds and the task mode's phrases
    ("" where it has none)."""

    return load_prompt_template(name, strict=True, **thresholds.params(), **mode).strip() if name else ""


def _merge_holes(v: CorrectionVariant, thresholds: Thresholds) -> Dict[str, str]:
    """The holes in pass 2's merge rules, contract and user template that say where a merge may go
    (`CorrectionVariant.merge_intro_fragment` and the rest)."""

    mode = mode_texts(v)
    return {
        "merge_intro": _hole_fragment(v.merge_intro_fragment, thresholds, mode),
        "merge_scope_clause": _hole_fragment(v.merge_scope_fragment, thresholds, mode),
        "merge_seams_rule": _hole_fragment(v.merge_seams_fragment, thresholds, mode),
        "may_merge_lead": v.may_merge_lead,
        "merge_check_seams": _hole_fragment(v.merge_check_seams_fragment, thresholds, mode),
        "sandwich_limit_clause": Template(v.sandwich_limit_clause).substitute(thresholds.params()),
    }


#: How the single pass tells one person from another: from the lines alone, with no speaker numbers.
SHAPE_SAME_PERSON_FRAGMENT = "fragment_shape_same_person_single_v1.md"
SHAPE_TURN_CLUE_FRAGMENT = "fragment_turn_clue_single_v1.md"


def _shape_block(v: CorrectionVariant, params: Mapping[str, str], turns: bool, keyed: bool, cloud: bool,
                 thresholds: Thresholds) -> str:
    """capableC's section on who is speaking and which seams are one person going on, before it corrects:
    pass 1's criteria (`shape_criteria`) worded for one call, with the `turn` column's changes of speaker as a
    clue where the task has speaker labels. Empty for the sets without one (`CorrectionVariant.shape_fragment`)."""

    if not v.shape_fragment:
        return ""
    from .shape_criteria import shape_criteria

    criteria = shape_criteria("single", same_person=load_prompt_template(SHAPE_SAME_PERSON_FRAGMENT).strip(),
                              correct_only=v.correct_only, thresholds=thresholds)
    return load_prompt_template(
        v.shape_fragment,
        strict=True,
        **mode_texts(v),
        promo=criteria["promo"],
        who=criteria["who"],
        join_criteria=criteria["join"],
        may_criteria=criteria["may_join"],
        drop_criteria=criteria["drop"],
        turn_clue="\n" + _turn_fragment(SHAPE_TURN_CLUE_FRAGMENT, True, keyed, cloud) if turns else "",
        voice_clause=params["voice_clause"],
        drop_listen=params["drop_listen_clause"],
    ).strip()


def _style_conflict_rule(style_block: str, extra_style: str, agentic: bool) -> str:
    """Item 8 of the grouped translation goals, for the styles this window really has.

    A settled selection injects either the user's styles or the factory default (`render_style_block`); an
    empty ``style_block`` is a pick still to come, which may end in either. Only the branches that can apply
    are said, and the `[字幕长度]` line only where a style can carry one."""

    from .knowledge.style import FACTORY_STYLE_NAME

    factory = load_prompt_template("fragment_style_conflict_factory_v1.md").strip()
    pending = not style_block.strip()
    factory_only = not pending and f'<style name="{FACTORY_STYLE_NAME}">' in style_block \
        and style_block.count("<style name=") == 1
    extra = bool(extra_style.strip())
    if factory_only and not extra:
        return factory
    words = prompt_phrases("style_conflict")
    # who "the user's styles" are, by where the window has them (`phrases_v1.toml`)
    shape = ("agentic" if agentic else "request") if pending else ("" if factory_only else "injected")
    return load_prompt_template(
        "fragment_style_conflict_v1.md",
        strict=True,
        user_styles="与".join(words[key] for key, on in ((shape, bool(shape)), ("extra", extra)) if on),
        factory_clause="" if shape == "injected" else factory,
        length_line_clause=words["length_line"] if pending or "[字幕长度]" in style_block else "",
    ).strip()


def render_entry_details_label(*, slimmed: bool = False, requestable: bool = False, agentic: bool = False) -> str:
    """What the user message says its `<entry_details>` holds. A batched window's come from its request
    turn or the user's note; an agentic window has no request turn (it reads with its own `kb_read`), so its
    come from the note or the windows before it. ``slimmed`` (`kb_phonetic_hints`) says how they were
    slimmed, and ``requestable`` (a request channel) that one can be asked for whole."""

    if slimmed:
        return load_prompt_template(
            "fragment_entry_details_label_slimmed_v1.md",
            request_clause=(
                load_prompt_template("fragment_entry_details_request_whole_v1.md").strip() if requestable else ""
            ),
        ).strip()
    return load_prompt_template(
        "fragment_entry_details_label_agentic_v1.md" if agentic else "fragment_entry_details_label_v1.md"
    ).strip()


def compose_correction_system(
    profile: TranslationProfile = DEFAULT_PROFILE,
    *,
    tier: CapabilityTier = CapabilityTier.CAPABLE,
    variant: str | None = None,
    extra_style: str = "",
    style_block: str = "",
    knowledge_enabled: bool = True,
    turns: bool = False,
    grouped: bool = False,
    keyed: bool = False,
    cloud: bool = False,
    task_update_feedback: bool = False,
    self_search: bool = False,
    agentic: bool = False,
    length_scale: float | None = None,
) -> str:
    """Assemble the correction system prompt for a profile.

    ``length_scale``: the window's length ratio (`SubtitleWindow.length_scale`), which the merge thresholds the
    prompt states, its worked examples and its shape criteria follow (`prompt_constants.Thresholds`). Taken
    here, before anything else: a caller that never set it fails whatever the window holds. A set that does
    not follow the ratio (basicA, `CorrectionVariant.follows_length_scale`) then renders at the standard tier's.

    ``turns``: the task has speaker labels, so the window input carries the `turn` column and the merge
    rules forbid merging across it (`chunking.SubtitleSegment.seam`). Off, the prompt is what it was before
    the column existed, word for word.

    ``grouped``: three passes' pass 2 (`SubtitleWindow.grouped`). The lines come grouped by pass 1, the
    `turn` column also reads 可并, and the prompt set is the variant's grouped one
    (`prompt_variants.grouped_variant`), with its own merge rules and examples marked to match; it is
    versioned by ``GROUPED_PROMPT_VERSION``.

    ``keyed``: the window's lines carry group keys, so `turn` may also read 同时 and 换回 (`chunking.turn_seams`);
    the window explains both and the merge rules name them. In a cloud task every window has keys, so the system
    prompt stays one per task (a window that shows neither would otherwise break the prefix cache); in a local
    task only the windows holding an overlap lane line do. Without keys it is as before.

    ``cloud``: the speaker labels come from a cloud task's transcription (a key other than the overlap lane), so
    the single pass's clue names the transcription model as their source, not the acoustic layer.

    ``agentic``: a tool-session window that reads entries with its own `kb_read` (`request_prompts`), so it
    may keep any entry it read, not only those in its `<entry_details>`; and a style it picks is one it read,
    not a `<style>` block.

    The ``<keep_entries>`` block only appears when the window's pruning
    decision is actually honoured: with no knowledge base there is nothing to
    prune, and with no per-window query round the harness force-transfers the
    whole set instead, because a dropped entry could never be requested back
    (docs/llm_harness_behavior.md).

    The prompt *set* comes from a :class:`CorrectionVariant`. Production passes ``variant``: the name the
    task-group cell for the profile's difficulty carries (quality and intermediate: capableC, for capable
    models; efficiency: basicA, for lower ones -- a difficulty is positioning, it does not cap the endpoint).
    Without one, ``tier`` (the answering endpoint's class) picks the default; any name overrides it (capableC,
    basicA). The variant bundles the merge fragments and the contract clauses that used to be tier
    ternaries; the variant-independent translated discipline
    (``fragment_translated_common_noaux_v1.md``) is prepended to its merge rules.
    """

    thresholds = Thresholds.at(length_scale)
    v = resolve_variant(variant, tier)
    if grouped:
        v = grouped_variant(v)
        turns = True
    thresholds = v.own_thresholds(thresholds)
    mode = mode_texts(v)
    params = _with_mode(_modal_params(profile, knowledge_enabled=knowledge_enabled), mode)
    context = PromptContext(
        profile=profile,
        knowledge_enabled=knowledge_enabled,
        self_search=self_search,
    )
    role_block = render_slot(
        "role",
        context,
        input_inventory=params["input_inventory"],
        video_role_addendum=render_slot("video_role_addendum", context),
        **mode,
    )
    goals_correction = render_slot(
        "goals_correction",
        context,
        verify_basis=params["verify_basis"],
        background_conflict_block=render_slot(
            "background_conflict",
            context,
            background_sources=_background_sources(knowledge_enabled=knowledge_enabled),
            knowledge_convention_clause=(
                load_prompt_template("fragment_background_knowledge_convention_v1.md", **mode).strip()
                if knowledge_enabled
                else ""
            ),
            **mode,
        ),
        **mode,
    )
    retrieval_block = render_slot(
        "retrieval",
        context,
        **mode,
        search_results_usage=search_results_usage(
            prepare=False, task_update_feedback=task_update_feedback, knowledge=knowledge_enabled,
            self_search=self_search,
        ),
    )
    effort_block = render_slot("effort", context, **mode)
    output_header = correction_output_header(v)

    if v.correct_only:
        # `--extra-style` asks for a translation; a correct-only task's style block holds only the styles written
        # for it, never the factory default (`render_style_block(correct_only=True)`).
        extra_style = ""
    extra_style_block = (
        "\n" + load_prompt_template("fragment_extra_style_v1.md", strict=True, extra=extra_style.strip())
        if extra_style.strip() else ""
    )
    style_block_text = (
        style_block.rstrip() + "\n" if style_block.strip() else ""
    )
    weighted_char_count_rule = load_prompt_template(
        "fragment_weighted_char_count_v1.md", **mode
    ).strip()
    # Every worked example is generated from curated material
    # (docs/llm_prompts.md); the start-bearing variants get their start
    # column from the row data, which is what retired the old text-splicing
    # post-pass.
    examples_block = build_examples(
        variant=v.examples_name,
        with_start=v.output_has_start,
        thresholds=thresholds,
        polished=v.correct_only,
        profile=profile,
        turns=turns,
        grouped=grouped,
        directory=EXAMPLES_ZH_DIR if v.correct_only else EXAMPLES_DIR,
        params={
            **params,
            "output_column_count": str(len(output_header.split("|"))),
            "over_cap_sources": v.over_cap_sources_words,
        },
    )
    if mode["examples_note"] and examples_block.strip():
        examples_block = mode["examples_note"] + "\n\n" + examples_block
    # Variant-independent translated discipline + variant-selected merge
    # strategy. The weighted char-count algorithm is injected only via the
    # output contract; the common fragment keeps just the column discipline.
    merge_block = (
        load_prompt_template("fragment_translated_common_noaux_v1.md").strip()
        + "\n\n"
        + load_prompt_template(
            v.merge_rules_fragment,
            speaker_basis=params["speaker_basis"],
            approval_reason_slot=v.approval_reason_slot,
            turn_rule=_turn_fragment(v.turn_rule_fragment, turns, keyed),
            turn_check=_turn_fragment(v.turn_check_fragment, turns, keyed),
            **_merge_holes(v, thresholds),
            **mode,
            # The merge thresholds live in prompt_constants only; fragments
            # reference them as ${thr_*} so a changed constant propagates.
            **thresholds.params(),
        ).strip()
    )

    assembled = load_prompt_template(
        "correction_main_v1.md",
        role_block=role_block,
        goals_correction_block=goals_correction.strip(),
        goals_translation_block=load_prompt_template(
            v.goals_translation_fragment,
            strict=True,
            paren_rule=params["paren_rule"],
            audio_fill_clause=params["audio_fill_clause"],
            grain_empty_audio_clause=params["grain_empty_audio_clause"],
            **mode,
            # a correct-only task has no translation style to weigh against the merge rules (`goals_style_conflict_lead`)
            style_conflict_rule="" if v.correct_only else _style_conflict_rule(style_block, extra_style, agentic),
        ).strip(),
        extra_style_block=extra_style_block,
        style_block=style_block_text,
        retrieval_block=retrieval_block,
        csv_input_block=load_prompt_template(
            "fragment_csv_input_v1.md",
            csv_time_col_name=params["csv_time_col_name"],
            csv_time_note=params["csv_time_note"],
            asr_result_header=asr_result_header(turns),
            turn_cell=_turn_fragment(v.turn_cell_fragment, turns, keyed),
            turn_column_item=(
                "\n" + _turn_fragment(v.turn_column_fragment, True, keyed) if turns else ""
            ),
        ).strip(),
        output_contract_block=load_prompt_template(
            v.output_contract_fragment,
            **thresholds.params(),
            insert_type_clause=params["insert_type_clause"],
            insert_position_clause=params["insert_position_clause"],
            insert_duration_clause=params["insert_duration_clause"],
            insert_note_clause=params["insert_note_clause"],
            weighted_char_count_rule=weighted_char_count_rule,
            output_csv_header=output_header,
            output_column_count=len(output_header.split("|")),
            # The input columns the output leaves out, said where the header is (weak models carry the
            # input's column layout over, v55–v58).
            dropped_columns_clause=_dropped_columns_clause(v.output_has_start, turns),
            output_start_clause=(
                "\n" + load_prompt_template("fragment_output_start_v1.md").rstrip("\n") if v.output_has_start else ""
            ),
            translated_position_clause=v.translated_position_clause,
            translated_merge_rule=v.translated_merge_rule,
            pacing_merge_clause=v.pacing_merge_clause,
            note_gap_clause=v.note_gap_clause,
            **_merge_holes(v, thresholds),
            **mode,
        ).strip(),
        advice_block=render_slot("advice", context, **mode),
        keep_block=render_slot(
            "keep_entries",
            context,
            max_keep_entries=KB_TRANSFER_MAX_ENTRIES,
            keep_source=load_prompt_template(
                "fragment_keep_source_agentic_v1.md" if agentic else "fragment_keep_source_injected_v1.md"
            ).strip(),
        ),
        alignment_block=load_prompt_template(v.alignment_fragment).strip() if v.alignment_fragment else "",
        window_block=load_prompt_template(
            "fragment_window_overlap_v1.md",
            preceding_audibility_note=params["preceding_audibility_note"],
            **mode,
        ).strip(),
        shape_block=_shape_block(v, params, turns, keyed, cloud, thresholds),
        merge_block=merge_block,
        hallucination_block=load_prompt_template(
            v.hallucination_fragment,
            discard_insert_clause=params["discard_insert_clause"],
            **mode,
        ).strip()
        if v.hallucination_fragment
        else "",
        examples_block=examples_block,
        effort_block=effort_block,
    )
    return assert_fully_substituted(
        _collapse_blank_lines(assembled), what="correction system prompt"
    )


def compose_correction_user(
    profile: TranslationProfile = DEFAULT_PROFILE,
    *,
    general_context_json: str,
    window_context: str,
    entry_details: str,
    previous_advice: str,
    pre_round_notes: str,
    entry_details_label: str = "",
    search_results: str,
    preceding_context_csv: str,
    current_asr_csv: str,
    current_asr_row_count: int,
    tier: CapabilityTier = CapabilityTier.CAPABLE,
    variant: str | None = None,
    knowledge_enabled: bool = True,
    overlap_candidates_block: str = "",
    style_pick_reminder: str = "",
    turns: bool = False,
    grouped: bool = False,
    keyed: bool = False,
    task_update_feedback: bool = False,
    agentic: bool = False,
    length_scale: float | None = None,
    user_requests: bool = False,
    extra_style: str = "",
) -> str:
    """``turns``, ``grouped``, ``keyed``, ``agentic`` and ``length_scale`` as in :func:`compose_correction_system`. ``style_pick_reminder`` leads the first reminder item and the closing
    reminder when the window carries a style pick: the reminders otherwise
    describe a direct delivery, which a request-turn pick forbids.
    ``task_update_feedback``: the system asks for the feedback block, so the
    closing reminder names it too. ``user_requests``: the window's background carries requests the
    preparation task relayed from the user, which a block then says how to follow; it ranks them below
    ``--extra-style`` only where that block is in the system (a translation)."""

    thresholds = Thresholds.at(length_scale)
    v = resolve_variant(variant, tier)
    if grouped:
        v = grouped_variant(v)
        turns = True
    thresholds = v.own_thresholds(thresholds)
    mode = mode_texts(v)
    params = _with_mode(_modal_params(profile), mode)
    serial = profile.continuity == "serial"
    reminders_block = render_slot(
        "user_reminders",
        PromptContext(profile=profile),
        # The <next_advice> demand is the ledger's output half; it leaves with
        # the axis (plan A.7).
        next_advice_reminder_item=(
            load_prompt_template("fragment_next_advice_reminder_v1.md").strip()
            if serial
            else ""
        ),
        **mode,
    )
    output_header = correction_output_header(v)
    # Same predicates as the system-side slots: the ledger section and the
    # <next_advice>/<keep_entries> demands exist only where something reads
    # them back (docs/llm_harness_behavior.md).
    if serial:
        trailing_blocks_reminder = load_prompt_template(
            "fragment_trailing_blocks_serial_v1.md",
            keep_entries_reminder=(
                "、`<keep_entries>`"
                if PromptContext(profile=profile, knowledge_enabled=knowledge_enabled).keeps_entries
                else ""
            ),
            feedback_reminder="，最后是 `<task_update_feedback>`" if task_update_feedback else "",
        ).strip()
    elif task_update_feedback:
        trailing_blocks_reminder = load_prompt_template("fragment_trailing_blocks_parallel_v1.md").strip()
    else:
        trailing_blocks_reminder = ""
    rendered = load_prompt_template(
        v.user_template,
        general_context_json=general_context_json,
        user_requests_block=(
            load_prompt_template(
                "fragment_user_requests_v1.md",
                strict=True,
                style_rank_clause=(
                    prompt_phrases("user_requests")["style_rank"]
                    if extra_style.strip() and not v.correct_only else ""
                ),
                **mode,
            ).strip() + "\n\n"
            if user_requests
            else ""
        ),
        window_context=window_context,
        # The block only where there is a knowledge base to have entries from.
        entry_details_block=(
            load_prompt_template(
                "fragment_entry_details_input_v1.md",
                entry_details_label=entry_details_label or render_entry_details_label(agentic=agentic),
                entry_details=entry_details,
            ).strip() + "\n\n"
            if knowledge_enabled
            else ""
        ),
        advice_input_block=(
            load_prompt_template(
                "fragment_advice_input_v1.md",
                previous_advice=previous_advice,
            ).strip() + "\n\n"
            if serial
            else ""
        ),
        # Only session replay still fills the slot (the task flow's request
        # turn keeps its notes in its own reply): an empty slot is not shown.
        pre_round_notes_block=(
            load_prompt_template(
                "fragment_pre_round_notes_input_v1.md",
                verify_basis=params["verify_basis"],
                pre_round_notes=pre_round_notes.strip(),
            ).strip() + "\n\n"
            if pre_round_notes.strip()
            else ""
        ),
        # Only session replay fills the slot: production's search results
        # arrive in the request turn's fill (`request_prompts`).
        search_results_block=(
            load_prompt_template(
                "fragment_search_results_input_v1.md", search_results=search_results.strip()
            ).strip() + "\n\n"
            if search_results.strip()
            else ""
        ),
        reminder_tail=reminders_block,
        preceding_context_csv=preceding_context_csv,
        current_asr_csv=current_asr_csv,
        current_asr_row_count=current_asr_row_count,
        overlap_candidates_block=overlap_candidates_block,
        merge_reminder=v.merge_reminder + _turn_fragment(v.turn_reminder_fragment, turns, keyed),
        mid_reminder_merge_rule=v.mid_reminder_merge_rule,
        merge_habit_reminder=v.merge_habit_reminder,
        trailing_blocks_reminder=trailing_blocks_reminder,
        style_pick_reminder=style_pick_reminder,
        output_csv_header=output_header,
        output_column_count=len(output_header.split("|")),
        **thresholds.params(),
        **mode,
    )
    return ensure_csv_block_headers(
        rendered,
        output_header=output_header,
        include_output_blocks=True,
        input_header=asr_result_header(turns),
    )
