"""The length tier's thresholds: one ratio per task, the numbers the model reads and the ones the script holds it
to computed from it (`prompt_constants.Thresholds`, `SubtitleWindow.length_scale`;
`docs/llm_harness_behavior.md`「合并门槛与任务的长度比例」)."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from finesub.llm.agent.agent_validators import deserialize_window, serialize_window
from finesub.llm.chunking import CorrectionBudget, Seam, SubtitleSegment, SubtitleWindow
from finesub.llm.client import LLMCallResult
from finesub.llm.grouping import group_lines
from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.prompt_compose import compose_correction_system, compose_correction_user
from finesub.llm.prompt_constants import BASE, PASS1, Thresholds
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.prompts import build_correction_csv_messages
from finesub.llm.routing.config import LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.shape_criteria import shape_criteria
from finesub.speech.diarization.judgments import ACT_JOIN, ACT_MAY

from .conftest import setattr_correction

QUALITY = resolve_profile("text", "none", "quality")
SIX = "type|position|corrected_text|translation|conf|note"


# --- the thresholds ------------------------------------------------------------------------------------------


def test_the_standard_ratio_gives_todays_numbers_as_they_were_rendered() -> None:
    standard = Thresholds.at(1.0)
    assert dict(standard.values) == dict(BASE) and standard.values == PASS1.values
    assert standard.params() == {
        "thr_hard_chars": "20", "thr_hard_seconds": "4", "thr_absolute_chars": "36", "thr_absolute_seconds": "7",
        "thr_filler_max_chars": "3", "thr_sandwich_max_chars": "16", "thr_sandwich_max_seconds": "4",
        "thr_splice_max_seconds": "7", "thr_splice_max_gap_seconds": "0.3", "thr_fused_max_sources": "4",
    }


@pytest.mark.parametrize("scale, hard, absolute, sandwich", [
    (0.8, (16, "3.2"), (29, "5.6"), (13, "3.2")),  # 28.8 -> 29, 12.8 -> 13
    (1.25, (25, "5"), (45, "8.8"), (20, "5")),  # 8.75 -> 8.8 (half up), 5.0 renders as 5
])
def test_the_three_tiers_scale_the_merged_lines_bounds_only(scale, hard, absolute, sandwich) -> None:
    tier = Thresholds.at(scale)
    params = tier.params()
    assert (tier.hard_chars, params["thr_hard_seconds"]) == hard
    assert (tier.absolute_chars, params["thr_absolute_seconds"]) == absolute
    assert (tier.sandwich_max_chars, params["thr_sandwich_max_seconds"]) == sandwich
    # what a thing is does not scale: a filler, a splice, the source cap
    for name in ("filler_max_chars", "splice_max_seconds", "splice_max_gap_seconds", "fused_max_sources"):
        assert tier.values[name] == BASE[name], name


def test_a_window_no_path_wrote_a_ratio_into_is_an_error_not_the_standard_tier() -> None:
    with pytest.raises(ValueError, match="length scale is not set"):
        Thresholds.at(None)
    with pytest.raises(ValueError, match="length scale is not set"):
        compose_correction_system(QUALITY, variant="capableC")
    window = _window(None, (0.0, 1.0))
    # taken at the entry: a reply with no merged row fails the same way
    with pytest.raises(ValueError, match="length scale is not set"):
        validate_correction_window_output(f"<translated>\n{SIX}\nsub|1|a|甲|high|\n</translated>", window,
                                          variant=resolve_variant("capableC"))


# --- the prompt ----------------------------------------------------------------------------------------------


def test_the_prompt_states_the_windows_thresholds() -> None:
    short = compose_correction_system(QUALITY, variant="capableC", length_scale=0.8)
    standard = compose_correction_system(QUALITY, variant="capableC", length_scale=1.0)
    assert "一般在 16 字、3.2 秒左右以内" in short and "加权字数 >29 或跨度 >5.6 秒" in short
    assert "一般在 20 字、4 秒左右以内" in standard and "20 字" not in short
    # the single pass's own shape criteria follow the window too
    assert "大约 3.2 秒以内" in short and "大约 4 秒以内" in standard
    grouped = compose_correction_system(QUALITY, variant="capableC", grouped=True, length_scale=0.8)
    assert "跨度 ≤3.2 秒且合并后 ≤13 字" in grouped and "越过 29 字/5.6 秒绝对门槛" in grouped
    user = compose_correction_user(
        QUALITY, variant="capableC", general_context_json="{}", window_context="", entry_details="",
        previous_advice="", pre_round_notes="", search_results="", preceding_context_csv="",
        current_asr_csv="1|0.0|1.0|0.0||テスト", current_asr_row_count=1, grouped=True, length_scale=1.25)
    assert "45" in user and "8.8" in user and "36" not in user
    # the window's own ratio, through the messages production builds
    system, _user = build_correction_csv_messages(window=_window(0.8, (0.0, 1.0)), profile=QUALITY,
                                                  variant="capableC")
    assert system["content"] == short


def test_basica_is_unaffected_by_the_ratio() -> None:
    """efficiency's basicA states no number that scales: its system and user prompts are byte-identical at every
    tier (worked examples' durations included), and it validates at the standard tier's, as its prompt shows.
    A window without a ratio still fails for it."""

    efficiency = resolve_profile("text", "none", "efficiency")
    variant = "basicA"
    systems = {scale: compose_correction_system(efficiency, variant=variant, length_scale=scale)
               for scale in (0.8, 1.0, 1.25)}
    assert systems[0.8] == systems[1.0] == systems[1.25]
    users = {scale: compose_correction_user(
        efficiency, variant=variant, general_context_json="{}", window_context="", entry_details="",
        previous_advice="", pre_round_notes="", search_results="", preceding_context_csv="",
        current_asr_csv="1|0.0|1.0|0.0||テスト", current_asr_row_count=1, length_scale=scale)
        for scale in (0.8, 1.0, 1.25)}
    assert users[0.8] == users[1.0] == users[1.25]
    with pytest.raises(ValueError, match="length scale is not set"):
        compose_correction_system(efficiency, variant=variant)
    with pytest.raises(ValueError, match="length scale is not set"):
        validate_correction_window_output("", _window(None, (0.0, 1.0)), variant=resolve_variant(variant))
    # the capable set's prompt does follow it
    assert (compose_correction_system(QUALITY, variant="capableC", length_scale=0.8)
            != compose_correction_system(QUALITY, variant="capableC", length_scale=1.0))


def test_basica_validates_at_the_standard_tiers_thresholds() -> None:
    from finesub.llm.prompt_variants import grouped_variant

    basic = resolve_variant("basicA")
    assert not basic.follows_length_scale
    assert basic.own_thresholds(Thresholds.at(0.8)).values == Thresholds.at(1.0).values
    capable = resolve_variant("capableC")
    for variant in (capable, grouped_variant(capable)):
        assert variant.follows_length_scale and variant.own_thresholds(Thresholds.at(0.8)).scale == 0.8


def test_pass_1s_criteria_are_the_standard_tiers_at_every_ratio() -> None:
    same = "同一个人"
    assert "大约 4 秒以内" in shape_criteria("pass1", same_person=same, correct_only=False)["join"]
    with pytest.raises(ValueError):
        shape_criteria("pass1", same_person=same, correct_only=False, thresholds=Thresholds.at(0.8))
    with pytest.raises(ValueError):
        shape_criteria("single", same_person=same, correct_only=False)


# --- the validator -------------------------------------------------------------------------------------------


def _window(scale, *spans, grouped: bool = False) -> SubtitleWindow:
    return SubtitleWindow(
        chunk_id="0001", segments=[SubtitleSegment(str(i), a, b, f"{i}。", seam=Seam.MAY if grouped and i > 1 else
                                                   (Seam.NONE if grouped else None))
                                   for i, (a, b) in enumerate(spans, start=1)],
        overlap_segments=[], boundary_reason="end", budget=CorrectionBudget(0, 0, 0, ""), grouped=grouped,
        length_scale=scale)


#: two sources whose merge spans 6.0 s: within the standard absolute threshold (7 s), past the short one (5.6 s)
SPANS = ((0.0, 3.0), (3.1, 6.0))
MERGED = f"<translated>\n{SIX}\nsub|1,2|一。二。|一二|high|同一句\n</translated>"


@pytest.mark.parametrize("scale, ok", [(0.8, False), (1.0, True), (1.25, True)])
def test_the_single_pass_rejects_a_merge_past_its_windows_absolute_threshold(scale, ok) -> None:
    result = validate_correction_window_output(MERGED, _window(scale, *SPANS), variant=resolve_variant("capableC"))
    assert result.ok is ok, result.errors
    if not ok:
        assert "(5.6s / 29 chars)" in result.errors[0]


def test_pass_2_still_rejects_no_merge_for_its_length() -> None:
    # three passes' pass 2 (the variant's grouped set) leans on pass 1's checks, at every ratio
    result = validate_correction_window_output(MERGED, _window(0.8, *SPANS, grouped=True),
                                               variant=resolve_variant("capableC"))
    assert result.ok, result.errors


def test_the_agent_manifest_carries_the_ratio() -> None:
    window = _window(0.8, *SPANS)
    payload = json.loads(json.dumps(serialize_window(window)))
    assert deserialize_window(payload) == window
    assert deserialize_window({**payload, "length_scale": None}).length_scale is None


# --- pass 1's joins ------------------------------------------------------------------------------------------


def _lines(*spans) -> list[SubtitleSegment]:
    return [SubtitleSegment(str(i), a, b, f"line {i}") for i, (a, b) in enumerate(spans, start=1)]


def _group(scale, segments, acts):
    return group_lines(segments, acts, different=lambda a, b: False, thresholds=Thresholds.at(scale))


def test_the_scripts_checks_on_pass_1s_joins_follow_the_tier() -> None:
    # a join run to 3.6 s: within the standard 4 s, past the short tier's 3.2 s, which marks it mergeable instead
    joined = _lines((0.0, 2.0), (2.1, 3.6))
    assert _group(1.0, joined, {"2": ACT_JOIN}).groups == (("1", "2"),)
    short = _group(0.8, joined, {"2": ACT_JOIN})
    assert short.groups == (("1",), ("2",)) and short.refused == (("2", "seconds"),)
    assert short.seams["2"] is Seam.MAY
    # a join run to 4.5 s: past the standard tier, within the long one's 5 s
    longer = _lines((0.0, 2.5), (2.6, 4.5))
    assert _group(1.25, longer, {"2": ACT_JOIN}).groups == (("1", "2"),)
    assert _group(1.0, longer, {"2": ACT_JOIN}).refused == (("2", "seconds"),)
    # two lines spanning 6.0 s read mergeable within the standard absolute threshold, not past the short one's
    pair = _lines(*SPANS)
    assert _group(1.0, pair, {"2": ACT_MAY}).seams["2"] is Seam.MAY
    assert _group(0.8, pair, {"2": ACT_MAY}).seams["2"] is Seam.NONE


def test_pass_2s_windows_carry_the_tasks_ratio_the_ones_without_pass_1_too() -> None:
    from finesub.llm.grouping import pass2_windows

    segments = _lines((0.0, 2.0), (2.1, 3.6), (4.0, 5.0), (5.2, 6.0))
    windows = [replace(_window(0.8, (0.0, 1.0)), chunk_id="0001", segments=segments[:2]),
               replace(_window(0.8, (0.0, 1.0)), chunk_id="0002", segments=segments[2:])]
    out, _grouping = pass2_windows(windows, segments, {"2": ACT_JOIN, "4": ACT_MAY}, {"3"},
                                   different=lambda a, b: False, thresholds=Thresholds.at(0.8))
    assert [(w.chunk_id, w.grouped, w.length_scale) for w in out] == [("0001", True, 0.8), ("0002", False, 0.8)]
    # the window falling back to the single pass prompt states the short tier's numbers
    system, _user = build_correction_csv_messages(window=out[1], profile=QUALITY, variant="capableC")
    assert "一般在 16 字、3.2 秒左右以内" in system["content"]


# --- where the task's ratio is written -----------------------------------------------------------------------


def test_a_task_writes_its_ratio_into_every_window_and_groups_at_it(tmp_path, monkeypatch) -> None:
    """The ratio resolved once the style is settled (here `--split-length-scale`) is recorded with the choice of
    passes, in `correction-mode.json` and the `correction_mode` artifact -- the one `tools/session_replay` reads
    a replayed window's ratio from -- and written into every window, which pass 1's lines are grouped at."""

    import finesub.llm.correction_translation as orchestration
    from finesub.llm.stages import three_pass
    from tools.session_replay.fixture import _recorded_length_scale

    from .test_speaker_judge_stage import FakeClient, fast_whole_file, judge_files, task_request

    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    fast_whole_file(monkeypatch, orchestration)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    grouped_at = []
    real = three_pass.pass2_windows

    def pass2_windows(*args, thresholds, **kwargs):
        grouped_at.append(thresholds.scale)
        return real(*args, thresholds=thresholds, **kwargs)

    monkeypatch.setattr(three_pass, "pass2_windows", pass2_windows)
    windows = []

    def execute(**kwargs):
        windows.extend(kwargs["plan"].windows)
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", execute)
    orchestration.run_correction_task(task_request(tmp_path, judge_files(tmp_path), mode="execute",
                                                   split_length_scale=0.8))
    assert windows and all(window.grouped and window.length_scale == 0.8 for window in windows)
    assert grouped_at and set(grouped_at) == {0.8}
    art = tmp_path / "out" / "clip.llm-artifacts"
    recorded = three_pass.recorded_choice(art)
    assert (recorded.length_scale, recorded.length_source) == (0.8, "explicit")
    assert _recorded_length_scale(art) == 0.8


def test_the_dry_run_writes_its_ratio_into_the_windows_it_renders(tmp_path, monkeypatch) -> None:
    import finesub.llm.prompt_artifacts as prompt_artifacts
    import finesub.llm.prompts as prompts

    from finesub.llm.length_tier import TaskLengthScale

    rendered = []
    real = prompts.compose_correction_system

    def compose(*args, **kwargs):
        rendered.append(kwargs["length_scale"])
        return real(*args, **kwargs)

    monkeypatch.setattr(prompts, "compose_correction_system", compose)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "A。"},
                                               {"id": "2", "start": 1.2, "end": 2.0, "text": "B。"}]}),
                      encoding="utf-8")
    from .test_llm_correction_translation import FakeTokenCounter

    artifacts = prompt_artifacts.build_prompt_artifacts(
        stable_json=stable, knowledge_enabled=False, counter=FakeTokenCounter(), profile=QUALITY,
        length_scale=TaskLengthScale(0.8, "explicit"))
    assert rendered and set(rendered) == {0.8}
    assert "3.2 秒" in json.dumps(artifacts, ensure_ascii=False)
    assert artifacts["length_scale"] == {"value": 0.8, "source": "explicit"}


# --- the run: every validation at the window's ratio ---------------------------------------------------------


#: the run's lines: the two of `SPANS` and two more, so a reply that only gets the merge wrong is repaired by
#: a patch (`correction_patch.PATCH_MIN_COVERAGE`)
RUN_SPANS = (*SPANS, (7.5, 8.5), (9.0, 10.0))
RUN_MERGED = f"<translated>\n{SIX}\nsub|1,2|一。二。|一二|high|同一句\nsub|3|三。|三|high|\nsub|4|四。|四|high|\n</translated>"


def _stable(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": str(i), "start": a, "end": b, "text": f"{i}。"}
                                               for i, (a, b) in enumerate(RUN_SPANS, start=1)]}), encoding="utf-8")
    return stable


def _reply(content: str) -> LLMCallResult:
    return LLMCallResult(content=content, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                         raw_response={"candidates": [{"finishReason": "STOP"}],
                                       "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1,
                                                         "totalTokenCount": 2}})


PATCH = f"<patch>\n{SIX}\nsub|1|一。|一|high|\nsub|2|二。|二|high|\n</patch>"


@pytest.mark.parametrize("continuity", ["serial", "parallel"])
def test_every_validation_of_a_run_holds_the_windows_ratio(tmp_path, monkeypatch, continuity) -> None:
    """First validation, the patch repair round, and a resumed run's replay of the ledger, serial and parallel:
    each validates the window at the ratio the task wrote into it. A window at another ratio is another input,
    asked again."""

    from finesub.llm.stages.correction import execute_correction_windows
    from finesub.llm.stages.correction import attempts

    from .test_llm_correction_translation import FakeTokenCounter

    calls: list[str] = []
    replies = iter([])

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            # a repair round's instruction rides beside the messages, as the real client appends it
            calls.append(str(kwargs.get("repair_instruction") or ""))
            try:
                return _reply(next(replies))
            except StopIteration:
                raise AssertionError(f"unexpected call {len(calls)}") from None

    validated: list[float | None] = []

    def validate(text, window, *, variant):
        validated.append(window.length_scale)
        return validate_correction_window_output(text, window, variant=variant)

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    setattr_correction(monkeypatch, "validate_correction_window_output", validate)
    assert attempts.validate_correction_window_output is validate
    stable = _stable(tmp_path)
    profile = resolve_profile("text", "none", "quality", continuity)

    def run(scale, *answers):
        nonlocal replies
        replies = iter(answers)
        calls.clear()
        validated.clear()
        return execute_correction_windows(
            stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(), profile=profile,
            task_artifact_dir=tmp_path / "artifacts", task_id="t", knowledge_enabled=False, length_scale=scale,
            postprocess_profile=None, max_retries_per_window=1)

    # the merge past the short tier's absolute threshold goes back for repair; the patch makes it two rows
    run(0.8, RUN_MERGED, PATCH)
    assert len(calls) == 2 and "5.6s" in calls[1] and "<patch>" in calls[1]
    assert validated and set(validated) == {0.8}
    # resumed at the same ratio: the ledger replays the window, validated at it again
    run(0.8)
    assert calls == [] and validated and set(validated) == {0.8}
    # at the standard ratio the window is another input: asked again, and the same merge passes
    run(1.0, RUN_MERGED)
    assert len(calls) == 1 and set(validated) == {1.0}


def test_a_caller_handing_a_plan_states_no_second_ratio(tmp_path) -> None:
    """The plan's windows carry the ratio the run wrote into them: a ``length_scale`` beside it would be dropped
    silently, so it is an error."""

    from finesub.llm.stages.correction import execute_correction_windows
    from finesub.llm.stages.correction.run import CorrectionPlan

    plan = CorrectionPlan.__new__(CorrectionPlan)
    with pytest.raises(ValueError, match="hands no plan"):
        execute_correction_windows(stable_json=_stable(tmp_path), output_path=tmp_path / "out.srt", plan=plan,
                                   length_scale=0.8, task_artifact_dir=tmp_path / "a", task_id="t",
                                   knowledge_enabled=False)
