"""The single pass capableC: three passes' first two in one call (docs/llm_prompts.md「单 pass 的拼法」).

It judges who is speaking and which seams are one person going on by pass 1's criteria, then merges by pass 2's
rules; a row may merge up to four raw lines, and a merged row past the absolute threshold goes back for repair.
"""

from __future__ import annotations

import pytest

from finesub.llm.chunking import SubtitleSegment
from finesub.llm.example_builder import build_examples
from finesub.llm.output_protocol import TRANSLATED_MAX_MERGED_SOURCES, validate_correction_output_text
from finesub.llm.prompt_compose import (
    SHAPE_SAME_PERSON_FRAGMENT,
    compose_correction_system,
    compose_correction_user,
    load_prompt_template,
)
from finesub.llm.prompt_constants import BASE, Thresholds
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.shape_criteria import shape_criteria
from finesub.llm.stages.speaker_judge import prompt as judge
from finesub.llm.stages.speaker_judge.windows import JudgeWindow

STANDARD = Thresholds.at(1.0)

SIX = "type|position|corrected_text|translation|conf|note"
PROFILES = [resolve_profile(*switches) for switches in (("text", "none", "quality"), ("audio", "local", "quality"))]


def _user(profile, *, turns: bool) -> str:
    return compose_correction_user(
        profile, variant="capableC", general_context_json="{}", window_context="", entry_details="",
        previous_advice="", pre_round_notes="", search_results="", preceding_context_csv="",
        current_asr_csv="1|0.0|1.0|0.0||テスト" if turns else "1|0.0|1.0|0.0|テスト", current_asr_row_count=1,
        turns=turns, length_scale=1.0)


@pytest.mark.parametrize("turns", [False, True])
@pytest.mark.parametrize("profile", PROFILES, ids=["text", "audio"])
def test_the_single_pass_reads_pass_1_s_criteria_and_no_two_call_words(profile, turns: bool) -> None:
    system = compose_correction_system(profile, variant="capableC", turns=turns, length_scale=1.0)
    user = _user(profile, turns=turns)
    same_person = load_prompt_template(SHAPE_SAME_PERSON_FRAGMENT).strip()
    for text in shape_criteria("single", same_person=same_person, correct_only=False, thresholds=STANDARD).values():
        assert text in system
    # One call: nothing says another step grouped, marked or will fix the lines.
    for word in ("可并", "上一步", "下一步", "这一步并", "超长的接法会被自动挡下", "隔着 drop"):
        assert word not in system and word not in user, word
    # Its own hallucination section is pass 1's drop criteria, not the single pass list.
    assert "ご視聴ありがとうございました" not in system
    # The criteria keep lines from the speaker calls only: the correction's discards of empty fillers stand, and
    # a set phrase is a hallucination only past the criteria's three points, whatever "高度疑似幻觉" elsewhere says.
    assert "其余的行一律不 drop" not in system and "认人这一步不因下面这些 drop" in system
    assert "照「翻译与内容取舍」第 4 条丢弃" in system and "别处说的「高度疑似幻觉」不放宽这一条" in system
    assert system.index("认人与定形状") < system.index("短片段合并策略（仅作用于")
    assert ("「换人」是声学层给的线索" in system) is turns
    assert ("`turn` 列标「换人」的行不得与上一行合并" in user) is turns
    assert ("先重听那一行" in system) is profile.correction_use_audio


def test_pass_1_reads_the_same_criteria_worded_for_a_step_of_its_own() -> None:
    from .test_speaker_judge_stage import segments_of, speaker

    rows = judge.rows_of(segments_of([(speaker(i % 2), 2, f"台词{i}") for i in range(1, 5)]))
    window = JudgeWindow("0001", "0001", 0, 4)
    system, _user_text, _ids = judge.judge_prompt(rows, window, title="clip", summary="", cast=(), marks=False,
                                                  profile=judge.profiles(rows, window), shape=True)
    same_person = load_prompt_template(judge.SHAPE_SAME_PERSON_TEMPLATES["acoustic"]).strip()
    criteria = shape_criteria("pass1", same_person=same_person, correct_only=False)
    for slot in ("join", "may_join", "drop"):  # who is speaking the judge's own prompt words its way
        assert criteria[slot] in system, slot
    assert "下一步会纠错" in system and "超长的接法会被自动挡下" in system
    assert "其余的行一律不 drop" in system and "认人这一步" not in system


BASIC_HEADER = "type|position|start|corrected_text|translation|conf|note"


def test_a_merged_row_past_the_absolute_threshold_goes_back_for_repair() -> None:
    long = [SubtitleSegment("1", 0.0, 4.0, "あ"), SubtitleSegment("2", 4.1, 7.2, "い")]  # 7.2 s merged
    result = validate_correction_output_text(f"<translated>\n{SIX}\nsub|1,2|あい|啊咿|high|\n</translated>", long,
                                             variant=resolve_variant("capableC"), thresholds=STANDARD)
    assert not result.ok
    [bad] = [verdict for verdict in result.rows if not verdict.ok]
    assert any("absolute limit" in error and "7.2s" in error for error in bad.errors)

    short = [SubtitleSegment("1", 0.0, 1.0, "あ"), SubtitleSegment("2", 1.1, 2.0, "い")]
    wordy = "啊" * 37  # 37 weighted chars, computed here from the translation
    result = validate_correction_output_text(f"<translated>\n{SIX}\nsub|1,2|あい|{wordy}|high|\n</translated>",
                                             short, variant=resolve_variant("capableC"), thresholds=STANDARD)
    assert not result.ok and any("37 weighted chars" in error for error in result.errors)
    ok = validate_correction_output_text(f"<translated>\n{SIX}\nsub|1,2|あい|{'啊' * 36}|high|\n</translated>", short,
                                         variant=resolve_variant("capableC"), thresholds=STANDARD)
    assert ok.ok, ok.errors


@pytest.mark.parametrize(("end", "passes"), [(7.04, True), (7.06, False)])
def test_the_span_is_judged_at_the_tenth_of_a_second_the_window_shows(end: float, passes: bool) -> None:
    # 7.04 s shows as 7.0 (at the threshold, passes); 7.06 s shows as 7.1 (past it)
    sources = [SubtitleSegment("1", 0.0, 3.0, "あ"), SubtitleSegment("2", 3.1, end, "い")]
    result = validate_correction_output_text(f"<translated>\n{SIX}\nsub|1,2|あい|啊咿|high|\n</translated>", sources,
                                             variant=resolve_variant("capableC"), thresholds=STANDARD)
    assert result.ok is passes, result.errors


def test_a_single_source_past_the_threshold_passes_and_basic_a_does_not_check() -> None:
    long_line = [SubtitleSegment("1", 0.0, 9.0, "あ" * 40)]
    result = validate_correction_output_text(f"<translated>\n{SIX}\nsub|1|{'あ' * 40}|{'啊' * 40}|high|\n</translated>",
                                             long_line, variant=resolve_variant("capableC"), thresholds=STANDARD)
    assert result.ok, result.errors

    long = [SubtitleSegment("1", 0.0, 4.0, "あ"), SubtitleSegment("2", 4.1, 7.2, "い")]
    row = "sub|1,2|0.0|あい|啊咿|high|"
    basic_a = validate_correction_output_text(f"<translated>\n{BASIC_HEADER}\n{row}\n</translated>", long,
                                              variant=resolve_variant("basicA"), thresholds=STANDARD)
    assert basic_a.ok, basic_a.errors
    assert not resolve_variant("basicA").rejects_long_merges
    assert not resolve_variant("capableC-grouped").rejects_long_merges


def test_the_source_cap_warning_is_the_variant_s() -> None:
    def warned(variant: str, ids: str, header: str, cells: str) -> bool:
        sources = [SubtitleSegment(str(n), n * 1.0, n * 1.0 + 0.8, "あ") for n in range(1, ids.count(",") + 2)]
        result = validate_correction_output_text(f"<translated>\n{header}\nsub|{ids}|{cells}\n</translated>", sources,
                                                 variant=resolve_variant(variant), thresholds=STANDARD)
        assert result.ok, result.errors
        return any("soft cap" in warning for warning in result.warnings)

    assert not warned("capableC", "1,2,3,4", SIX, "ああああ|啊|high|")
    assert warned("capableC", "1,2,3,4,5", SIX, "あああああ|啊|high|")
    assert warned("basicA", "1,2,3", BASIC_HEADER, "1.0|あああ|啊|high|")
    assert resolve_variant("capableC").max_merged_sources == BASE["fused_max_sources"] == 4
    # Three passes keep their cap of two lines (`grouping`), whatever the single pass allows.
    assert TRANSLATED_MAX_MERGED_SOURCES == 2
    assert resolve_variant("capableC-grouped").max_merged_sources is None


def test_only_three_passes_show_the_filler_sandwich_example() -> None:
    profile = resolve_profile("text", "none", "quality")
    title = "filler 三明治三源特例"
    assert title not in build_examples(variant="capableC", with_start=False, profile=profile, thresholds=STANDARD)
    assert title in build_examples(variant="capableC", with_start=False, profile=profile, grouped=True, thresholds=STANDARD)
    # basicA merges only a word cut in two (owner 2026-10-31): no sandwich either.
    assert title not in build_examples(variant="basicA", with_start=True, profile=profile, thresholds=STANDARD)


def test_every_example_capable_c_shows_renders_in_three_passes_or_says_it_does_not() -> None:
    from finesub.llm.example_builder import load_examples, render_example

    profile = resolve_profile("audio", "local", "quality")
    for example in load_examples():
        if example.applies and "capableC" not in example.applies:
            continue
        if example.grouped == "no":
            continue
        assert render_example(example, variant="capableC", with_start=False, profile=profile,
                              grouped=True, thresholds=STANDARD).strip(), example.id
