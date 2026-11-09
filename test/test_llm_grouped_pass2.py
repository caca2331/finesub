"""Three passes' pass 2: the correction window over the lines pass 1 grouped (`SubtitleWindow.grouped`) --
its own prompt set and version, examples marked 可并 where they merge, and merges checked against 可并
(docs/llm_harness_behavior.md 「三个 pass」)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from finesub.llm.agent.agent_validators import deserialize_window, serialize_window
from finesub.llm.prompt_constants import Thresholds
from finesub.llm.chunking import Seam, SubtitleSegment, render_window_segments_as_csv, split_window_in_half
from finesub.llm.example_builder import ExampleMaterialError, build_examples, parse_example, render_example
from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.prompt_compose import (
    GROUPED_PROMPT_VERSION,
    PROMPT_VERSION,
    compose_correction_system,
    compose_correction_user,
)
from finesub.llm.prompt_variants import grouped_variant, resolve_variant
from finesub.llm.prompts import build_correction_csv_messages
from finesub.llm.routing.profiles import resolve_profile

from .test_llm_speaker_turns import FakeTokenCounter, grouped_reply, reply, window_of

PROFILE = resolve_profile("audio", "local", "quality")
CAPABLE_C = resolve_variant("capableC")


def user(**kwargs):
    return compose_correction_user(
        PROFILE, variant="capableC", general_context_json="{}", window_context="（无）", entry_details="（无）",
        previous_advice="（无）", pre_round_notes="", search_results="（无）", preceding_context_csv="",
        current_asr_csv="1|0.0|1.0|||テスト", current_asr_row_count=1, **kwargs, length_scale=1.0)


def test_the_grouped_prompt_merges_only_where_marked_and_keeps_its_own_version():
    system = compose_correction_system(PROFILE, variant="capableC", grouped=True, length_scale=1.0)
    assert GROUPED_PROMPT_VERSION != PROMPT_VERSION
    assert "**只并「可并」的行**" in system and "换人、可并或空|" in system
    # Sol's disambiguations and the relaxed length wording
    assert "harness 不会提供合并候选" not in system and "最终并不并由你决定" in system
    # six columns and no `#` comments (owner 2026-10-23): the reason for a long merge goes in the row's note
    assert "`type|position|corrected_text|translation|conf|note`" in system and "char_count" not in system
    assert "`#` 注释" not in system and "前置 reasoning" not in system
    assert "并把理由用一句短结论写在该行 note 里" in system
    # the effort note (absent with injected search results) is the single pass one
    unsearched = compose_correction_system(resolve_profile("audio", "none", "quality"), variant="capableC",
                                           grouped=True, length_scale=1.0)
    assert "推理留在内部思考里，回复里不写推理散文" in unsearched and "`#` 注释" not in unsearched
    assert "硬门槛" not in system and "多数源" not in system
    # pass 1's keep list reaches pass 2's drop rules
    assert "应答与附和（うん、うんうん、はい）" in system
    assert "写得非常明确的，按风格来；说得不那么肯定的，两边折中" in system  # owner's D9, as pass 3
    assert "出厂默认风格与合并规则在合并的倾向上冲突时，按合并规则来" in system  # O2, owner 2026-11-03
    # the length tier: the style's tier sets both thresholds, its text does not relax them
    assert "在合并的倾向上冲突时" in system and "不再另外放宽这两道门槛" in system
    text = user(grouped=True)
    assert "只有 `turn` 列标「可并」的行可以与上一行合并。" in text and "硬门槛" not in text
    assert "local_id|start|duration|gap|turn|text" in text
    assert "`type|position|corrected_text|translation|conf|note`" in text and "固定 6 列" in text
    assert "char_count" not in text and "`#`" not in text


def test_the_single_pass_prompt_does_not_change_with_the_grouped_one():
    assert compose_correction_system(PROFILE, variant="capableC", turns=True, length_scale=1.0) != compose_correction_system(
        PROFILE, variant="capableC", grouped=True, length_scale=1.0)
    assert "可并" not in compose_correction_system(PROFILE, variant="capableC", turns=True, length_scale=1.0)


def test_only_capable_c_has_a_grouped_prompt_set():
    grouped = grouped_variant(CAPABLE_C)
    assert grouped.user_template == "correction_user_grouped_v1.md"
    assert grouped.name == "capableC-grouped" and grouped.examples_name == "capableC"
    assert resolve_variant("capableC-grouped") is grouped and grouped_variant(grouped) is grouped
    with pytest.raises(ValueError, match="basicA"):
        compose_correction_system(PROFILE, variant="basicA", grouped=True, length_scale=1.0)


def test_grouped_examples_mark_the_rows_they_merge():
    single = build_examples(variant="capableC", with_start=False, profile=PROFILE, turns=True, thresholds=Thresholds.at(1.0))
    grouped = build_examples(variant="capableC", with_start=False, profile=PROFILE, grouped=True, thresholds=Thresholds.at(1.0))
    rows = [line for line in grouped.splitlines() if line.startswith(("sub|", "type|"))]
    assert rows and all(line.removesuffix("<void>").count("|") == 5 for line in rows)
    assert not any(line.startswith("# ") for line in grouped.splitlines())
    assert "|可并|" not in single
    marked = [line.split("|")[0] for line in grouped.split("<asr_result>")[1].splitlines() if "|可并|" in line]
    assert marked == ["9", "11", "34"]  # s09, s11 merged; s34 kept apart for length (no 可并 past 7 s: grouping never marks one)
    assert "硬门槛" not in grouped


EXAMPLE = """---
id: t
kind: mini
may: [{may}]
---
<input>
a | 0.0 | 1.0 | あ
b | 1.1 | 1.0 | い
</input>

<output>
sub | a,b | あい | 啊 | high |
</output>
"""


def test_an_example_merging_a_row_not_marked_may_fails_in_three_passes_only():
    unmarked = parse_example(EXAMPLE.format(may=""), source="t.md")
    assert render_example(unmarked, variant="capableC", with_start=False, thresholds=Thresholds.at(1.0))
    with pytest.raises(ExampleMaterialError, match="not marked may"):
        render_example(unmarked, variant="capableC", with_start=False, grouped=True, thresholds=Thresholds.at(1.0))
    marked = parse_example(EXAMPLE.format(may="b"), source="t.md")
    assert "2|1.1|1.0|0.1|可并|い" in render_example(
        marked, variant="capableC", with_start=False, grouped=True, thresholds=Thresholds.at(1.0))
    with pytest.raises(ExampleMaterialError, match="no input row"):
        parse_example(EXAMPLE.format(may="c"), source="t.md")


def grouped_window(*seams):
    segments = [SubtitleSegment(str(n), n * 2.0, n * 2.0 + 1.0, f"{n}。", seam=seam)
                for n, seam in enumerate(seams, start=1)]
    return replace(window_of(segments), grouped=True)


def test_a_grouped_window_reads_mergeable_seams_but_never_on_its_first_line():
    window = grouped_window(Seam.MAY, Seam.MAY, Seam.CHANGE)
    cells = [line.split("|")[4] for line in render_window_segments_as_csv(window).splitlines()]
    assert cells == ["", "可并", "换人"]
    messages = build_correction_csv_messages(window=window, profile=PROFILE, variant="capableC")
    assert "**只并「可并」的行**" in messages[0]["content"] and "|可并|2。" in messages[1]["content"]


def test_a_merge_over_a_seam_not_marked_mergeable_is_noted_in_a_grouped_window():
    window = grouped_window(Seam.NONE, Seam.MAY, Seam.NONE)
    result = validate_correction_window_output(
        grouped_reply("sub|1,2|一。二。|一二|high|", "sub|3|三。|三|high|"), window, variant=CAPABLE_C)
    assert result.ok and result.turn_violation_rows == ()
    result = validate_correction_window_output(
        grouped_reply("sub|1|一。|一|high|", "sub|2,3|二。三。|二三|high|"), window, variant=CAPABLE_C)
    assert result.ok and result.turn_violation_rows == (("2", "3"),)
    assert result.turn_notes == ("顺带：position 为 2,3 的合并行里，源 3 的 turn 列没标「可并」，拆回各自的行",)
    # the same merge in a single pass window is fine: no change of speaker there
    single = replace(window, grouped=False)
    assert validate_correction_window_output(
        reply("sub|1|1.0|1.0|一。|一|high|1|", "sub|2,3|3.0|0.0|二。三。|二三|high|2|"), single,
        variant=CAPABLE_C).turn_violation_rows == ()


def test_a_grouped_window_stays_grouped_when_halved_and_through_the_agent_validator():
    window = grouped_window(*[Seam.NONE] * 6)
    assert deserialize_window(serialize_window(window)).grouped
    assert "grouped" not in serialize_window(replace(window, grouped=False))

    class TinyCounter(FakeTokenCounter):
        def count_text(self, text: str) -> int:
            return 1

    halves = split_window_in_half(window, counter=TinyCounter())
    assert halves is not None and all(half.grouped for half in halves)
