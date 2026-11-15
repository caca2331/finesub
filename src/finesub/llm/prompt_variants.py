"""Named correction-prompt variants (Design A).

A *variant* bundles every choice that used to be a ``tier is BASIC`` branch in
``prompt_compose``: which merge fragments to load, which output contract and user template it takes, and
the handful of short clauses injected into them (their text: `prompt_templates/phrases_v1.toml`,
``[variant.<name>]``). Selecting a variant is decoupled from the endpoint's capability tier:
the tier still classifies the answering model, but it only picks a *default*
variant (``DEFAULT_VARIANT_FOR_TIER``). Callers — notably session_replay — may
override the variant by name to serve any registered prompt set
(``capableC``/``basicA``).

Adding a variant is a registry entry (plus its clauses and any new fragment file it names); no
branching code changes. Each variant is fully specified — no sparse inheritance
— so coupled clauses always move together and cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from string import Template
from typing import Dict

from .routing.config import CapabilityTier
from .prompt_constants import BASE, Thresholds
from .prompt_tables import prompt_phrases, task_mode_texts


@dataclass(frozen=True)
class CorrectionVariant:
    """One fully-specified correction prompt set."""

    name: str
    # Whole-fragment choices. Worked examples are *not* here: they come from
    # curated material through ``example_builder``, selected by ``applies``.
    merge_rules_fragment: str
    # Short clauses injected into the output-contract fragment.
    translated_position_clause: str
    translated_merge_rule: str
    pacing_merge_clause: str
    note_gap_clause: str
    # Short clauses injected into the user template. What they remind of differs by set: in basicA that only
    # a word cut in two is spliced back, in the capable sets which seams may merge at all (capableC: the ones it judged
    # `join` / `may_join`; its grouped set: the lines marked 可并).
    merge_reminder: str
    mid_reminder_merge_rule: str
    # Whole templates: the output contract and the user template. basicA has its own pair, the capable sets
    # share another (their merge holes and six-column rows).
    output_contract_fragment: str
    user_template: str
    # The user template's third reminder, on merging (basicA only, empty for the capable sets): only a word
    # cut in two is spliced back.
    merge_habit_reminder: str
    # Where the reason for running past the usual length is written (the capable sets' merge rules), said in
    # the prompt as such: the model never learns which variant it runs under.
    approval_reason_slot: str = ""
    # What the request turn's fill (`request_prompts.build_correction_fill_message`) says a delivery is: the
    # blocks the user template's closing reminder puts before the trailing ones.
    fill_delivery: str = "`<translated>` 完整终稿"
    # basicA mirrors the input timeline by keeping ``start`` right after
    # position.
    output_has_start: bool = False
    # A Chinese source's set (`correct_only_variant`): its phrases are the `[correct_only]` mode texts; the
    # second column, headed `polished`, is a light polish of the faithful corrected text, and is left empty on a
    # line that is only filler (a stutter is compressed instead) -- that line stays in the corrected products,
    # out of the subtitle (`stages.correction.run._write_products`).
    correct_only: bool = False
    # Sections of the system prompt a set may word its own way. The single pass sets share one wording; the
    # sets that read pass 2's merge rules (capableC and its grouped set) take its translation goals and
    # alignment, and the grouped set its own `turn` column (三态). An empty hallucination fragment leaves the
    # section out (capableC drops lines by pass 1's criteria instead).
    goals_translation_fragment: str = "fragment_goals_translation_v1.md"
    alignment_fragment: str = "fragment_alignment_v1.md"
    hallucination_fragment: str = "fragment_hallucination_v1.md"
    turn_cell_fragment: str = "fragment_turn_cell_v1.md"
    turn_column_fragment: str = "fragment_turn_column_v1.md"
    # The sentences about a `换人` mark, when the task has speaker labels: in the merge rules, their merged-row
    # check and the user template's closing reminder. The grouped set's merge rules and reminders say where
    # merges go themselves (its `turn` column also reads 可并) and take none of them.
    turn_rule_fragment: str = "fragment_turn_rule_v1.md"
    turn_check_fragment: str = "fragment_turn_check_v1.md"
    turn_reminder_fragment: str = "fragment_turn_reminder_v1.md"
    # capableC: the section where the model judges, before correcting, who is speaking and which seams are one
    # person going on (pass 1's criteria, `shape_criteria`). Empty for the other sets.
    shape_fragment: str = ""
    # The holes in pass 2's merge rules (`fragment_merge_rules_grouped_v1.md`), its output contract and user
    # template, that say where a merge may go: the grouped set's lines come marked 可并 by pass 1, capableC's
    # seams are its own calls. The paragraph-level ones are template files (the merge rules' opening, the
    # scope and source cap of their item 1, the seams rule of item 2, the seams check of item 4); the few
    # words in between are fields. Empty for the sets whose templates have no such holes.
    merge_intro_fragment: str = ""
    merge_scope_fragment: str = ""
    merge_seams_fragment: str = ""
    merge_check_seams_fragment: str = ""
    may_merge_lead: str = ""
    # A `${thr_*}` template: the sandwich's bounds follow the window's length ratio.
    sandwich_limit_clause: str = ""
    # The worked bad-output example's word for a merge past the source cap.
    over_cap_sources_words: str = "四源以上"
    # How many sources a row may merge before the validator warns (a soft cap, never rejected); None is
    # `output_protocol.TRANSLATED_MAX_MERGED_SOURCES`.
    max_merged_sources: int | None = None
    # Whether a merged row past the absolute threshold (the window's `absolute_seconds` / `absolute_chars`, the
    # weighted count computed from its translation) fails validation and goes to the repair round (owner Q13,
    # 2026-10-30: capableC only; three passes' script holds pass 1's joins back instead).
    rejects_long_merges: bool = False
    # A set derived from another (three passes' pass 2) shows that one's worked examples.
    base: str = ""
    # Whether the set's prompt and validation follow the window's length ratio. The capable sets merge by the
    # thresholds, which scale; basicA states no number that scales, so its prompt (worked examples' durations
    # included) and validation stay at the standard tier's whatever the task's ratio (efficiency is unaffected
    # by the length tier, docs/llm_harness_behavior.md「合并门槛与任务的长度比例」).
    follows_length_scale: bool = True

    def own_thresholds(self, thresholds: Thresholds) -> Thresholds:
        """The thresholds this set renders and validates at: the window's ``thresholds`` (taken first by the
        caller, so a window without a ratio fails for every set), or the standard tier's for a set that does
        not follow the ratio."""

        return thresholds if self.follows_length_scale else _STANDARD

    @property
    def examples_name(self) -> str:
        """The variant whose worked examples this set renders."""
        return self.base or self.name


_STANDARD = Thresholds.at(1.0)


def _clauses(name: str, *, correct_only: bool = False) -> Dict[str, str]:
    """The clause fields of the set named ``name`` (`prompt_templates/phrases_v1.toml`, ``[variant.<name>]``).

    They state the thresholds that do not follow the length ratio straight from `BASE`; one stating a threshold
    that does (`prompt_constants.SCALED_*`) keeps its ``${thr_*}`` placeholder, filled with the window's when the
    prompt is rendered. ``correct_only`` fills the same clauses with the ``[correct_only]`` mode texts (the worked
    rows and cue words they quote)."""

    base = {key: str(value) for key, value in BASE.items()}
    modes, common = task_mode_texts(correct_only), prompt_phrases("variant_common")
    shadowed = (base.keys() & modes.keys()) | ((base.keys() | modes.keys()) & common.keys())
    if shadowed:
        raise KeyError(f"BASE, the mode texts and [variant_common] share {sorted(shadowed)}: one would hide another")
    values = base | modes
    values |= {key: Template(text).substitute(values) for key, text in common.items()}
    clauses = {}
    for field_name, text in prompt_phrases("variant")[name].items():
        template = Template(text)
        unknown = [key for key in template.get_identifiers() if key not in values and not key.startswith("thr_")]
        if unknown:
            raise KeyError(f"phrases_v1.toml [variant.{name}] {field_name}: no value for {unknown}")
        clauses[field_name] = template.safe_substitute(values)
    return clauses


# --- Fully specified variants ---


# basicA: one row per source for the efficiency tier's low-capability models (owner 2026-10-31): it only
# splices a word cut across two rows, and may drop hallucinated and pure-filler lines.
# It replaced basicB's judgment merges, which merged wrongly far more often than basicA left a line split
# (owner 2026-11-02). Its reminders restate the splice test the merge rules fragment spells out
# (`[variant_common]`'s `splice_test`).
_BASIC_A = CorrectionVariant(
    name="basicA",
    merge_rules_fragment="fragment_merge_rules_basic_v1.md",
    output_contract_fragment="fragment_output_contract_nosingles_v1.md",
    user_template="correction_user_nosingles_v1.md",
    output_has_start=True,
    follows_length_scale=False,
    **_clauses("basicA"),
)


# capableC: one call does what three passes' first two do in two (owner 2026-10-30, docs/llm_prompts.md
# 「单 pass 的拼法」). The model first judges who is speaking and which seams are one person going on, by
# pass 1's criteria (`shape_fragment`, `shape_criteria`), then corrects, translates and merges by pass 2's
# merge rules, whose holes say the seams are its own calls: a `join` seam merges, a `may_join` seam merges by
# pause, length and translation style, no other seam does. Pass 1's drops become `discard` rows, so the
# single pass hallucination section is left out. A row may merge up to `fused_max_sources` raw lines (the
# filler sandwich is one shape of it), and a merged row past the absolute threshold goes back for repair
# (`rejects_long_merges`). Six columns, no `#` comments, as pass 2 (owner 2026-10-23).
#
# The C first meant "with comments". The name stays while the capable sets drop the comments and the guide
# columns (owner 2026-10-23). Window records and agent sessions name the set they were served; an old
# nine-column answer recorded under `capableC` or `capableC-grouped` no longer validates against the
# six-column set (by design: no backward compatibility, `docs/llm_prompts.md`).
_CAPABLE_C = CorrectionVariant(
    name="capableC",
    merge_rules_fragment="fragment_merge_rules_grouped_v1.md",
    translated_merge_rule="",
    pacing_merge_clause="",
    output_contract_fragment="fragment_output_contract_grouped_v1.md",
    user_template="correction_user_grouped_v1.md",
    merge_habit_reminder="",
    goals_translation_fragment="fragment_goals_translation_grouped_v1.md",
    alignment_fragment="fragment_alignment_grouped_v1.md",
    hallucination_fragment="",
    turn_rule_fragment="fragment_turn_rule_single_v1.md",
    shape_fragment="fragment_shape_single_v1.md",
    merge_intro_fragment="fragment_merge_intro_single_v1.md",
    merge_scope_fragment="fragment_merge_scope_single_v1.md",
    merge_seams_fragment="fragment_merge_seams_single_v1.md",
    merge_check_seams_fragment="fragment_merge_check_seams_single_v1.md",
    sandwich_limit_clause="",
    max_merged_sources=BASE["fused_max_sources"],
    rejects_long_merges=True,
    **_clauses("capableC"),
)

# Three passes' pass 2 reads the lines pass 1 grouped and merges only where it marked 可并 (`grouping.py`).
# Its merge rules, contract and user template are capableC's, their holes filled with the grouped wording;
# its `turn` column also reads 可并, and pass 1 has dropped the lines it would discard, so its hallucination
# section is its own. Pass 1 did the speaker calls, so it has no shape section, and its sources are grouped
# lines: two at most, three for a filler sandwich. Six columns, the reason for running past the usual length
# in the row's note (docs/llm_harness_behavior.md「Prompt 信息」). It shows capableC's worked examples, in
# their grouped overlay where they have one. The window, not the cell, says its lines are grouped
# (`SubtitleWindow.grouped`): keyed by the cell's variant, named on its own so a window record says which set
# it got.
_CAPABLE_C_GROUPED = CorrectionVariant(
    name="capableC-grouped",
    merge_rules_fragment="fragment_merge_rules_grouped_v1.md",
    translated_merge_rule="",
    pacing_merge_clause="",
    output_contract_fragment="fragment_output_contract_grouped_v1.md",
    user_template="correction_user_grouped_v1.md",
    merge_habit_reminder="",
    goals_translation_fragment="fragment_goals_translation_grouped_v1.md",
    alignment_fragment="fragment_alignment_grouped_v1.md",
    hallucination_fragment="fragment_hallucination_grouped_v1.md",
    turn_cell_fragment="fragment_turn_cell_grouped_v1.md",
    turn_column_fragment="fragment_turn_column_grouped_v1.md",
    turn_rule_fragment="",
    turn_check_fragment="",
    turn_reminder_fragment="",
    merge_intro_fragment="fragment_merge_intro_grouped_v1.md",
    merge_scope_fragment="fragment_merge_scope_grouped_v1.md",
    merge_seams_fragment="fragment_merge_seams_grouped_v1.md",
    merge_check_seams_fragment="fragment_merge_check_seams_grouped_v1.md",
    base=_CAPABLE_C.name,
    **_clauses("capableC-grouped"),
)
_GROUPED: Dict[str, CorrectionVariant] = {_CAPABLE_C.name: _CAPABLE_C_GROUPED}

_GROUPED_BY_NAME: Dict[str, CorrectionVariant] = {variant.name: variant for variant in _GROUPED.values()}

VARIANTS: Dict[str, CorrectionVariant] = {
    _BASIC_A.name: _BASIC_A,
    _CAPABLE_C.name: _CAPABLE_C,
}

# Capability tier only picks the compatibility default when no explicit
# variant is supplied. Production resolves the variant from its task-group
# cell before entering the endpoint loop. Variant names
# are ``<tier><letter>`` — the tier is baked into the name (capableC
# targets the capable tier / 3.8 or 3.6 Flash; basicA targets the basic tier /
# 3.5 Flash Lite), so a
# variant is never further split by tier.
DEFAULT_VARIANT_FOR_TIER: Dict[CapabilityTier, str] = {
    CapabilityTier.CAPABLE: "capableC",
    CapabilityTier.BASIC: "basicA",
}


#: The suffix of a correct-only set's name (`correct_only_variant`).
CORRECT_ONLY_SUFFIX = "-polish"


def correct_only_variant(base: CorrectionVariant) -> CorrectionVariant:
    """``base`` for a correct-only task (a Chinese source, docs/llm_prompts.md「只纠错」): the corrected text is a
    faithful transcript and the second column a light polish of it (`CorrectionVariant.correct_only`); the phrases
    that differ by mode come from `prompt_templates/mode_texts_v1.toml` (`prompt_compose.mode_texts`), and so do
    the worked rows and cue words ``base``'s clauses quote (`_clauses`). It shows ``base``'s worked examples.
    Registered under its own name so that every place resolving a variant by name -- window records, caches,
    agent validators -- gets this set back. The clauses are read again by ``base.name``, so a ``base`` whose
    clauses were changed with ``replace()`` gets the registered ones back."""

    return replace(
        base,
        name=base.name + CORRECT_ONLY_SUFFIX,
        base=base.examples_name,
        correct_only=True,
        alignment_fragment="",  # the polish keeps to its row: no reordering across rows
        **_clauses(base.name, correct_only=True),
    )


def correct_only_name(name: str) -> str:
    """The correct-only set's name for the set named ``name`` (unchanged when it already is one)."""

    return name if not name or name.endswith(CORRECT_ONLY_SUFFIX) else name + CORRECT_ONLY_SUFFIX


def resolve_variant(
    variant: str | None, tier: CapabilityTier = CapabilityTier.CAPABLE
) -> CorrectionVariant:
    """Resolve a variant by name, falling back to the tier's default.

    Production passes an explicit name now: the variant comes from the
    task-group cell (the old ``effective_tier`` difficulty
    ceiling is encoded in the cell data instead: correction's efficiency cells
    carry basicA). ``None`` falls back to the tier's default; an unknown name
    is a hard error so a typo never silently serves the wrong prompt set.
    """

    if variant is None:
        variant = DEFAULT_VARIANT_FOR_TIER[tier]
    if variant in _GROUPED_BY_NAME:  # a window record of three passes' pass 2
        return _GROUPED_BY_NAME[variant]
    try:
        return VARIANTS[variant]
    except KeyError:
        known = ", ".join(sorted(VARIANTS))
        raise ValueError(f"Unknown correction variant {variant!r}; known: {known}")


# The correct-only sets, beside the ones they derive from (`correct_only_variant`).
_CORRECT_ONLY_GROUPED = {
    correct_only_name(name): correct_only_variant(grouped) for name, grouped in _GROUPED.items()
}
_GROUPED.update(_CORRECT_ONLY_GROUPED)
_GROUPED_BY_NAME.update({variant.name: variant for variant in _CORRECT_ONLY_GROUPED.values()})
VARIANTS.update({correct_only_name(name): correct_only_variant(variant) for name, variant in list(VARIANTS.items())})


def has_grouped_variant(name: str) -> bool:
    """Whether the variant named ``name`` has a prompt set for three passes' pass 2."""

    return name in _GROUPED


def grouped_variant(variant: CorrectionVariant) -> CorrectionVariant:
    """The prompt set three passes' pass 2 serves in place of ``variant`` (a grouped set is its own)."""

    if variant.name in _GROUPED_BY_NAME:
        return variant
    try:
        return _GROUPED[variant.name]
    except KeyError:
        known = ", ".join(sorted(_GROUPED))
        raise ValueError(f"Three passes' pass 2 has no prompt set for {variant.name!r}; known: {known}") from None
