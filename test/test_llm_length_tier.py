"""Where a task's length ratio comes from (`length_tier`; `docs/llm_harness_behavior.md`「合并门槛与任务的长度比例」): an explicit `--split-length-scale`, the injected style's `[字幕长度]` tier, `config.toml`, 1.0 --
resolved once the style is settled, recorded with the choice of passes, and kept by a resumed task."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from finesub.llm import correction_translation as orchestration
from finesub.llm.correction_request import CorrectionRequest, CorrectionRequestError, resolved_request
from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.edit import edit_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.style import StyleSelection, load_style_catalog, profile_fields, resolve_style_selection
from finesub.llm.length_tier import (
    STYLE_UNDECIDED,
    TIER_RATIOS,
    TaskLengthScale,
    resolve_task_length_scale,
    styles_tier,
    task_length_scale,
    warn_window_pick,
)
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages import three_pass

QUALITY = resolve_profile("text", "none", "quality")


def _entry_text(key: str, tier: str | None) -> str:
    lines = [f"# {key}", "", f"{key}的口味", "", "## 档案", "", f"- [本名] {key}", f"- [作者] {key}",
             f"- [适用范围] {key}的投稿"]
    if tier is not None:
        lines.append(f"- [字幕长度] {tier}")
    return "\n".join(lines + ["", "## 约定", "", "- [直角引号] 对话统一用「」", ""])


def _store(tmp_path: Path, tiers: dict[str, str | None]) -> Path:
    root = tmp_path / "kb"
    for name in tiers:
        maintain_main(["--root", str(root), "new", "style", name, "--intro", f"{name}的口味"])
    KnowledgeRepo.forget(root)
    _set_tiers(root, tiers)
    return root


def _set_tiers(root: Path, tiers: dict[str, str | None]) -> None:
    repo = KnowledgeRepo.open(root)
    for subject in repo.subjects("style"):
        key = subject.payload.get("surface")
        if key in tiers:
            edit_subject(repo, subject, _entry_text(key, tiers[key]))
    KnowledgeRepo.forget(root)


def _force_tier(root: Path, name: str, written: str) -> None:
    """A `[字幕长度]` line no write path takes (the proposal check refuses a body outside the three), put in the
    store directly: what an older store or a hand-edited database could still hold."""

    repo = KnowledgeRepo.open(root)
    subject = next(s for s in repo.subjects("style") if s.payload.get("surface") == name)
    with repo.store.begin("import") as txn:
        txn.create_node("forced-tier", "note", {"text": written, "label": "字幕长度"})
        txn.create_membership("forced-tier-m", subject.local_id, "forced-tier", "档案", 99)
    KnowledgeRepo.forget(root)


def _config(tmp_path: Path, scale: float) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f"[segmentation]\nlength_scale = {scale}\n", encoding="utf-8")
    return path


# --- the resolver --------------------------------------------------------------------------------------------


def test_the_ratio_comes_from_the_flag_then_the_tier_then_the_file_then_one(tmp_path) -> None:
    config = _config(tmp_path, 1.1)
    assert resolve_task_length_scale(0.9, tier="偏长", config_path=config) == TaskLengthScale(0.9, "explicit")
    assert resolve_task_length_scale(None, tier="偏长", style="甲组", config_path=config) == TaskLengthScale(
        1.25, "style", tier="偏长", style="甲组")
    assert resolve_task_length_scale(None, config_path=config) == TaskLengthScale(1.1, "config")
    assert resolve_task_length_scale(None, config_path=tmp_path / "none.toml") == TaskLengthScale(1.0, "default")
    assert dict(TIER_RATIOS) == {"偏短": 0.8, "标准": 1.0, "偏长": 1.25}
    # the setting's range holds whichever layer wins
    with pytest.raises(ValueError, match="length_scale"):
        resolve_task_length_scale(3.0)


def test_an_explicit_ratio_out_of_range_is_refused_before_any_work() -> None:
    with pytest.raises(CorrectionRequestError, match="length_scale"):
        resolved_request(CorrectionRequest(stable_json=Path("s.json"), mode="plan", split_length_scale=2.0))


def test_the_tier_is_read_from_the_profile_section_only(tmp_path) -> None:
    from finesub.llm.knowledge.style import load_style_entries

    root = _store(tmp_path, {"甲组": "偏长", "乙组": None})
    _set_tiers(root, {"乙组": None})
    (_key, injected), = load_style_entries(root, ("style/甲组",))
    assert profile_fields(injected).length == "偏长"
    # a 字幕长度 line under 约定 is a rule the model reads, not the tier
    assert profile_fields(injected.replace("## 档案", "## 约定")).length == ""
    (_key, plain), = load_style_entries(root, ("style/乙组",))
    assert profile_fields(plain).length == ""


def test_a_tier_the_module_does_not_know_is_warned_about_and_not_used(tmp_path, reported) -> None:
    root = _store(tmp_path, {"甲组": None})
    _force_tier(root, "甲组", "很长")
    assert styles_tier(root, ("style/甲组",), rev=None) == ("", "")
    assert reported.codes() == ["style-length-tier-unknown"] and "很长" in reported.joined()
    scale = task_length_scale(None, None, knowledge_root=root, names=("style/甲组",), rev=None)
    assert (scale.value, scale.source) == (1.0, "default")


def test_stacked_styles_take_the_first_named_that_has_a_tier(tmp_path, reported) -> None:
    root = _store(tmp_path, {"甲组": None, "乙组": "偏长", "丙组": "偏短"})
    assert styles_tier(root, ("style/甲组", "style/乙组", "style/丙组"), rev=None) == ("偏长", "乙组")
    assert reported.codes() == ["style-length-tier-stacked"]
    # the same tier further on is no conflict
    _set_tiers(root, {"丙组": "偏长"})
    assert styles_tier(root, ("style/乙组", "style/丙组"), rev=None) == ("偏长", "乙组")
    assert reported.codes() == ["style-length-tier-stacked"]


def test_an_explicit_ratio_does_not_read_the_styles_tiers(tmp_path, reported) -> None:
    """The flag decides, so stacked styles with different tiers are not warned about as if a tier did."""

    root = _store(tmp_path, {"乙组": "偏长", "丙组": "偏短"})
    scale = task_length_scale(0.9, None, knowledge_root=root, names=("style/乙组", "style/丙组"), rev=None)
    assert scale == TaskLengthScale(0.9, "explicit") and reported.codes() == []


def test_a_resumed_task_keeps_its_recorded_ratio_over_the_flag_and_the_file(tmp_path, reported, monkeypatch) -> None:
    from finesub import config as app_config

    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(_config(tmp_path, 1.25)))
    app_config.clear_config_cache()
    kept = task_length_scale(None, 0.8, knowledge_root=None, names=(), rev=None)
    assert kept == TaskLengthScale(0.8, "record") and reported.codes() == []
    # the same ratio given again says nothing; another one is warned about once and not used
    assert task_length_scale(0.8, 0.8, knowledge_root=None, names=(), rev=None).value == 0.8
    assert reported.codes() == []
    assert task_length_scale(1.0, 0.8, knowledge_root=None, names=(), rev=None).value == 0.8
    assert reported.codes() == ["length-scale-recorded"]


def test_a_style_still_to_be_picked_does_not_count(tmp_path) -> None:
    scale = task_length_scale(None, None, knowledge_root=tmp_path, names=None, rev=None)
    assert scale.to_metadata() == {"value": 1.0, "source": "default", "note": STYLE_UNDECIDED}
    assert task_length_scale(0.9, None, knowledge_root=tmp_path, names=None, rev=None).note == ""


# --- which revision, which selection -------------------------------------------------------------------------


def _with_scale(root: Path, selection: StyleSelection | None) -> three_pass.PassChoice:
    request = CorrectionRequest(stable_json=Path("s.json"))
    return orchestration._with_length_scale(request, three_pass.PassChoice("single", False), selection,
                                            knowledge_root=root)


def _auto(tmp_path: Path, root: Path, saved: dict) -> StyleSelection:
    context = tmp_path / "ctx.json"
    context.write_text(json.dumps({"context_pack": {"general_context": {}, "window_contexts": []},
                                   "keep_entries": [], "style_selection": saved}, ensure_ascii=False),
                       encoding="utf-8")
    selection, window_picks = orchestration._apply_auto_style(
        resolve_style_selection("auto", "read", knowledge_root=root, config_path=tmp_path / "none.toml"),
        load_style_catalog(root), knowledge_root=root, context_path=context, artifact_dir=None, task_id="t",
        fast=False)
    assert not window_picks
    return selection


def test_the_tier_is_the_injected_revisions(tmp_path) -> None:
    """`auto` resumes on the revision it picked at: the tier written since does not count. A named style reads
    the current revision, as it is injected. A record whose revision cannot render the style falls back to the
    current one, and so does its tier."""

    root = _store(tmp_path, {"甲组": "偏短", "default_style": None})
    picked_at = load_style_catalog(root).rev
    _set_tiers(root, {"甲组": "偏长"})

    saved = {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "r", "rev": picked_at}
    picked = _with_scale(root, _auto(tmp_path, root, saved))
    assert (picked.length_scale, picked.length_source) == (0.8, "style")

    named = _with_scale(root, StyleSelection(mode="read", names=("甲组",)))
    assert named.length_scale == 1.25

    another_store = {**saved, "rev": 0}
    fallen_back = _auto(tmp_path, root, another_store)
    assert fallen_back.rev is None
    assert _with_scale(root, fallen_back).length_scale == 1.25


def test_only_default_style_counts_its_tier_and_a_window_pick_does_not(tmp_path) -> None:
    """With nothing to choose `auto` settles on `default_style`, whose tier counts. A fast window that picks among
    candidates itself leaves the task's ratio to the other layers."""

    root = _store(tmp_path, {"default_style": "偏长"})
    settled, window_picks = orchestration._apply_auto_style(
        resolve_style_selection("auto", "read", knowledge_root=root, config_path=tmp_path / "none.toml"),
        load_style_catalog(root), knowledge_root=root, context_path=tmp_path / "none.json", artifact_dir=None,
        task_id="t", fast=True)
    assert not window_picks and settled.names == ("style/default_style",)
    assert _with_scale(root, settled).length_scale == 1.25
    window_pick = _with_scale(root, None)
    assert (window_pick.length_scale, window_pick.length_source) == (1.0, "default")


def test_a_window_pick_with_another_tier_is_warned_about(tmp_path, reported) -> None:
    root = _store(tmp_path, {"甲组": "偏长", "乙组": "标准", "丙组": None})
    warn_window_pick(root, ("style/乙组",), rev=None, task_scale=1.0)
    warn_window_pick(root, ("style/丙组",), rev=None, task_scale=0.8)
    assert reported.codes() == []
    warn_window_pick(root, ("style/甲组",), rev=None, task_scale=1.0)
    assert reported.codes() == ["style-length-tier-window-pick"]
    warn_window_pick(root, ("style/乙组",), rev=None, task_scale=0.8)
    assert reported.codes() == ["style-length-tier-window-pick"] * 2


# --- the task --------------------------------------------------------------------------------------------------


def _run_execute(tmp_path, monkeypatch, **overrides):
    """One executed task with the windows stubbed: the windows it would correct."""

    from .test_speaker_judge_stage import FakeClient, fast_whole_file, judge_files, task_request

    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    fast_whole_file(monkeypatch, orchestration)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    windows = []

    def execute(**kwargs):
        windows.extend(kwargs["plan"].windows)
        return kwargs["output_path"]  # nothing written: the next run corrects again

    monkeypatch.setattr(orchestration, "execute_correction_windows", execute)
    stable = tmp_path / "clip-stable.json"
    if not stable.exists():
        judge_files(tmp_path)
    orchestration.run_correction_task(replace(task_request(tmp_path, stable, mode="execute"), **overrides))
    return windows


def test_a_resumed_task_runs_at_its_recorded_ratio(tmp_path, monkeypatch, reported) -> None:
    from tools.session_replay.fixture import _recorded_length_scale

    art = tmp_path / "out" / "clip.llm-artifacts"
    first = _run_execute(tmp_path, monkeypatch, split_length_scale=0.8)
    assert {window.length_scale for window in first} == {0.8}
    again = _run_execute(tmp_path, monkeypatch, split_length_scale=1.25)
    assert {window.length_scale for window in again} == {0.8}
    assert "length-scale-recorded" in reported.codes()
    assert three_pass.recorded_choice(art).length_scale == 0.8 and _recorded_length_scale(art) == 0.8
    # --no-resume chooses again
    fresh = _run_execute(tmp_path, monkeypatch, split_length_scale=1.25, resume=False)
    assert {window.length_scale for window in fresh} == {1.25}


def test_a_one_pass_task_records_its_ratio_too(tmp_path, monkeypatch) -> None:
    from tools.session_replay.fixture import _recorded_length_scale

    windows = _run_execute(tmp_path, monkeypatch, speaker_judge="off", split_length_scale=1.25)
    art = tmp_path / "out" / "clip.llm-artifacts"
    assert {window.length_scale for window in windows} == {1.25} and not any(w.grouped for w in windows)
    choice = three_pass.recorded_choice(art)
    assert (choice.mode, choice.length_scale) == ("single", 1.25) and _recorded_length_scale(art) == 1.25


def test_a_research_only_run_records_no_ratio(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(orchestration, "run_research_stage", lambda **kwargs: None)
    from .test_speaker_judge_stage import judge_files

    stable = judge_files(tmp_path)
    art = tmp_path / "art"
    orchestration.run_correction_task(CorrectionRequest(
        stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="research", knowledge="none",
        profile=QUALITY, test_profile=True, task_artifact_dir=art, split_length_scale=0.8, fast="off"))
    assert three_pass.recorded_choice(art) is None


# --- the dry run ---------------------------------------------------------------------------------------------


def _plan(tmp_path: Path, root: Path, **kwargs) -> dict:
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
                      encoding="utf-8")
    seen: dict = {}
    orchestration.run_correction_task(CorrectionRequest(
        stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan", knowledge_root=root,
        profile=QUALITY, test_profile=True, on_plan=seen.update, **kwargs))
    return seen


def test_a_dry_run_states_its_ratio_and_where_it_came_from(tmp_path) -> None:
    root = _store(tmp_path, {"甲组": "偏短", "default_style": "偏长"})
    # a model still picks among the styles: the tier is left out, and the plan says so
    pending = _plan(tmp_path, root)
    assert pending["length_scale"] == {"value": 1.0, "source": "default", "note": STYLE_UNDECIDED}
    assert pending["correction_windows"] and "一般在 20 字、4 秒左右以内" in pending["correction_messages"][0][0][
        "content"]
    named = _plan(tmp_path, root, style="甲组")
    assert named["length_scale"] == {"value": 0.8, "source": "style", "tier": "偏短", "style": "甲组"}
    assert "一般在 16 字、3.2 秒左右以内" in named["correction_messages"][0][0]["content"]
    flag = _plan(tmp_path, root, style="甲组", split_length_scale=1.25)
    assert flag["length_scale"] == {"value": 1.25, "source": "explicit"}
    # the task's choice beside it shows the same ratio, not the one still unset before the style
    assert (named["correction_mode"]["length_scale"], named["correction_mode"]["length_scale_source"]) == (0.8, "style")
    assert (flag["correction_mode"]["length_scale"], flag["correction_mode"]["length_scale_source"]) == (
        1.25, "explicit")


def test_a_dry_run_with_only_default_style_counts_its_tier(tmp_path) -> None:
    root = _store(tmp_path, {"default_style": "偏长"})
    shown = _plan(tmp_path, root)
    assert shown["length_scale"] == {"value": 1.25, "source": "style", "tier": "偏长", "style": "default_style"}


def test_efficiency_runs_at_the_task_ratio_but_its_prompt_does_not_follow_it(tmp_path) -> None:
    root = _store(tmp_path, {"default_style": "偏长"})
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
                      encoding="utf-8")
    shown: dict = {}
    for scale in (0.8, 1.25):
        seen: dict = {}
        orchestration.run_correction_task(CorrectionRequest(
            stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan", knowledge_root=root,
            profile=resolve_profile("text", "none", "efficiency"), test_profile=True, on_plan=seen.update,
            split_length_scale=scale))
        shown[scale] = seen["correction_messages"][0]
    assert shown[0.8] == shown[1.25]


def test_the_pipeline_hands_its_flag_to_the_llm_as_given(tmp_path, monkeypatch) -> None:
    """The pipeline's `--split-length-scale` reaches the correction task unresolved: the style's tier sits between
    it and `config.toml`, which only the task can know once its style is settled."""

    from finesub import stages

    class Built(Exception):
        pass

    seen = {}

    def correction_request(*args, **options):
        seen.update(options)
        raise Built

    monkeypatch.setattr(stages, "_correction_request", correction_request)
    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    for given in (0.9, None):
        with pytest.raises(Built):
            stages.run_pipeline(source, output_path=tmp_path / "out" / "talk.srt", stage="translated-srt",
                                diarize="off", split_length_scale=given, knowledge="none")
        assert seen["split_length_scale"] == given


# --- the style side: the 档案 line, the proposal check, learned tiers ----------------------------------------


def test_the_style_preset_registers_the_tier_label_with_the_three_tiers() -> None:
    from finesub.llm.knowledge.node.presets import load_preset
    from finesub.llm.knowledge.node.render import section_comment

    spec = load_preset("style").section("档案")
    label = spec.label("字幕长度")
    assert label is not None and not label.core
    assert label.values == tuple(TIER_RATIOS)
    # a model writing a style reads the note and the three bodies in the entry's 档案 guidance
    guide = next(line for line in section_comment(spec) if "[字幕长度]" in line)
    assert "偏长（少拆）或偏短" in guide and "不写成约定" in guide and "只能写 偏短 / 标准 / 偏长 之一" in guide


def _apply(root: Path, *rows: dict) -> object:
    from finesub.llm.knowledge.node.proposals import apply_model_proposals

    repo = KnowledgeRepo.open(root)
    block = "<knowledge_proposals>\n" + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) \
        + "\n</knowledge_proposals>"
    return apply_model_proposals(block, repo=repo, allow_categories=("style",), task_id="t",
                                 knowledge_read_rev=repo.rev, handles=_handles(repo))


def _handles(repo: KnowledgeRepo):  # type: ignore[no-untyped-def]
    from finesub.llm.knowledge.node.render import HandleMap, render_subject

    handles = HandleMap()
    for subject in repo.subjects():
        render_subject(repo.store, subject.local_id, mode="prompt", handles=handles)
    return handles


def _tier_of(root: Path, name: str) -> str:
    from finesub.llm.knowledge.style import load_style_entries

    KnowledgeRepo.forget(root)
    (_key, text), = load_style_entries(root, (f"style/{name}",))
    return profile_fields(text).length


def test_a_proposal_writing_a_tier_outside_the_three_is_refused(tmp_path) -> None:
    root = _store(tmp_path, {"甲组": None})
    append = {"op": "append_lines", "category": "style", "entry": "甲组", "section": "档案", "reason": "r"}
    report = _apply(root, {**append, "content": "[字幕长度] 很长"})
    assert report.applied == [] and "只能写 偏短 / 标准 / 偏长 之一" in report.skipped[0].reason
    assert _tier_of(root, "甲组") == ""
    assert _apply(root, {**append, "content": "[字幕长度] 偏长"}).applied
    assert _tier_of(root, "甲组") == "偏长"
    # rewriting the line goes through the same check
    repo = KnowledgeRepo.open(root)
    handles = _handles(repo)
    subject = repo.subjects("style")[0].local_id
    text = repo.entry_prompt_text(subject, handles)
    handle = next(line for line in text.splitlines() if line.startswith("[字幕长度]")).rsplit("<!-- ", 1)[1].split()[0]
    report = _apply(root, {"op": "update", "id": handle, "line": "[字幕长度] 1.5", "reason": "r"})
    assert report.applied == [] and "只能写" in report.skipped[0].reason
    assert _tier_of(root, "甲组") == "偏长"


def test_a_human_edit_writing_a_tier_outside_the_three_is_refused_or_parked(tmp_path, reported) -> None:
    """The same check holds for a human's edit: `knowledge edit` refuses it (the human is right there), the
    `rendered/` write-back parks the line in 待归类 with a warning -- either way the line is kept or still in the
    editor, never dropped, and a parked one is no tier."""

    from finesub.llm.knowledge.node.edit import EditError, harvest_rendered_edits
    from finesub.llm.knowledge.node.repo import RENDERED_DIRNAME

    root = _store(tmp_path, {"甲组": "偏长"})
    repo = KnowledgeRepo.open(root)
    subject = repo.subjects("style")[0]
    edited = _entry_text("甲组", "很长")
    with pytest.raises(EditError, match="只能写 偏短 / 标准 / 偏长 之一"):
        edit_subject(repo, subject, edited)
    assert _tier_of(root, "甲组") == "偏长"

    repo = KnowledgeRepo.open(root)
    repo.refresh_rendered()
    path = next((root / RENDERED_DIRNAME / "style").glob("*.md"))
    path.write_text(path.read_text(encoding="utf-8").replace("[字幕长度] 偏长", "[字幕长度] 很长"), encoding="utf-8")
    results = harvest_rendered_edits(repo)
    assert [result["status"] for result in results] == ["applied"]
    assert results[0]["staged"] == ["[字幕长度] 很长"]
    assert "knowledge-rendered-edit-staged" in reported.codes()
    assert "[字幕长度] 很长" in path.read_text(encoding="utf-8")
    parked = [membership for membership in repo.store.children(subject.local_id, repo.rev)
              if membership.section == "待归类"]
    assert len(parked) == 1
    assert _tier_of(root, "甲组") == ""


def test_a_new_style_and_the_factory_overlay_write_no_tier(tmp_path) -> None:
    from finesub.llm.knowledge.node.base_style import admit_proposals
    from finesub.llm.knowledge.style import factory_style_lines

    root = tmp_path / "kb"
    maintain_main(["--root", str(root), "new", "style", "甲组", "--intro", "甲组的口味"])
    assert _tier_of(root, "甲组") == ""
    assert not any("字幕长度" in line for lines in factory_style_lines().values() for line in lines)
    # adopt-base-style admits only the factory default's own labels in its seed sections
    repo = KnowledgeRepo.open(root)
    subject = repo.subjects("style")[0].local_id
    admitted, refused = admit_proposals(
        [{"op": "append_lines", "entry": "甲组", "section": "档案", "content": "[字幕长度] 偏长"},
         {"op": "append_lines", "entry": "甲组", "section": "约定", "content": "[字幕长度] 偏长"}],
        repo=repo, subject_id=subject, handles=_handles(repo),
    )
    assert admitted == [] and len(refused) == 2


def _refined_update_prompt(tmp_path: Path, *, tier: str | None) -> str:
    from finesub.llm.knowledge.update import run_knowledge_update

    from .test_llm_knowledge_update import FakeTokenCounter, _write_task_outputs

    final_srt, _paths, refined_srt = _write_task_outputs(tmp_path, with_refined=True)
    root = _store(tmp_path, {"甲组": tier})
    prompts = tmp_path / "prompts"
    run_knowledge_update(
        final_srt=final_srt, refined_srt=refined_srt, task_id="t", task_summary="t", knowledge_root=root,
        token_counter=FakeTokenCounter(), execute=False, prompt_dir=prompts, style_names=["甲组"],
    )
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(prompts.glob("*.txt")))


def test_a_refined_update_learns_no_tier_on_its_own(tmp_path) -> None:
    """owner 2026-11-05: the update measures nothing to write a tier from, with or without one in the style; the
    system prompt says the tier changes only when the user's task summary asks."""

    for tier in (None, "偏长"):
        workdir = tmp_path / (tier or "none")
        workdir.mkdir()
        prompt = _refined_update_prompt(workdir, tier=tier)
        assert "<subtitle_length>" not in prompt, tier
        assert "本任务不主动加、改、删这一行，只照用户在 `<task_summary>` 里对这套\n   风格字幕长短的明确要求办" in prompt, tier


# --- the knowledge update's hold on the [字幕长度] line (owner 2026-11-05) -------------------------------------


def _tier_handle(root: Path) -> str:
    repo = KnowledgeRepo.open(root)
    text = repo.entry_prompt_text(repo.subjects("style")[0].local_id, _handles(repo))
    return next(line for line in text.splitlines() if line.startswith("[字幕长度]")).rsplit("<!-- ", 1)[1].split()[0]


def _gated(root: Path, user_summary: bool | None, *rows: dict, ascii_json: bool = False) -> object:
    from finesub.llm.knowledge.node.proposals import apply_model_proposals
    from finesub.llm.length_tier import tier_line_gate

    repo = KnowledgeRepo.open(root)
    handles = _handles(repo)
    block = "<knowledge_proposals>\n" + "\n".join(json.dumps(row, ensure_ascii=ascii_json) for row in rows) \
        + "\n</knowledge_proposals>"
    return apply_model_proposals(block, repo=repo, allow_categories=("style",), task_id="t",
                                 knowledge_read_rev=repo.rev, handles=handles,
                                 line_gate=None if user_summary is None else tier_line_gate(user_summary))


_APPEND = {"op": "append_lines", "category": "style", "entry": "甲组", "reason": "r"}


def test_without_the_users_summary_the_update_leaves_the_tier_line_alone(tmp_path) -> None:
    from finesub.llm.length_tier import TIER_HELD

    root = _store(tmp_path, {"甲组": "偏长"})
    handle = _tier_handle(root)
    for change in ({"op": "update", "id": handle, "line": "[字幕长度] 偏短", "reason": "r"},
                   {"op": "remove", "id": handle, "reason": "r"},
                   {"op": "move_line", "id": handle, "section": "待归类", "reason": "r"}):
        report = _gated(root, False, change, {**_APPEND, "section": "约定", "content": "[自称] 自称一律用「我」"})
        assert [record.reason for record in report.skipped] == [TIER_HELD], change["op"]
        assert report.skipped[0].entry == "甲组" and report.skipped[0].section == "档案"
        assert _tier_of(root, "甲组") == "偏长"
        # the rest of the batch lands; the next round starts clean
        _set_tiers(root, {"甲组": "偏长"})
    bare = _store(tmp_path / "bare", {"甲组": None})
    report = _gated(bare, False, {**_APPEND, "section": "档案", "content": "[字幕长度] 偏长"})
    assert report.applied == [] and report.skipped[0].reason == TIER_HELD
    assert _tier_of(bare, "甲组") == ""


def test_without_the_users_summary_a_tier_the_check_refuses_is_held_too(tmp_path) -> None:
    """A tier body outside the three, or emptied, is held (not fed back for a rewrite) when no tier change is
    allowed; with the user's summary the same body is refused by the value check and goes back to the model."""
    from finesub.llm.knowledge.update import _is_rejection
    from finesub.llm.length_tier import TIER_HELD

    root = _store(tmp_path, {"甲组": "偏长"})
    for change in ({**_APPEND, "section": "档案", "content": "[字幕长度] 很长"},
                   {"op": "update", "id": _tier_handle(root), "line": "[字幕长度]", "reason": "r"}):
        report = _gated(root, False, change)
        assert [record.reason for record in report.skipped] == [TIER_HELD], change
    bare = _store(tmp_path / "bare", {"甲组": None})
    report = _gated(bare, True, {**_APPEND, "section": "档案", "content": "[字幕长度] 很长"})
    assert report.applied == [] and "只能写 偏短 / 标准 / 偏长 之一" in report.skipped[0].reason
    assert _is_rejection({"reason": report.skipped[0].reason})


def test_an_update_cannot_slip_a_second_line_past_the_gate(tmp_path) -> None:
    """An update rewrites one line: one holding a newline is refused before anything is read off it, so it can
    neither add a tier nor carry a body no check ran on."""

    root = _store(tmp_path, {"甲组": None})
    repo = KnowledgeRepo.open(root)
    text = repo.entry_prompt_text(repo.subjects("style")[0].local_id, _handles(repo))
    scope = next(line for line in text.splitlines() if line.startswith("[适用范围]")).rsplit("<!-- ", 1)[1].split()[0]
    # every separator `str.splitlines` (and so the style reader) breaks on, escaped in the JSON a model writes
    # (a raw one would break the proposal line itself)
    for separator in ("\n", "\r", "\u2028", "\u2029", "\u0085", "\x0b", "\x1c"):
        for user_summary in (False, True, None):
            report = _gated(root, user_summary, {"op": "update", "id": scope,
                                                 "line": f"[适用范围] demo{separator}[字幕长度] 偏长", "reason": "r"},
                            ascii_json=True)
            assert report.applied == [] and "one line" in report.skipped[0].reason, (repr(separator), user_summary)
            assert _tier_of(root, "甲组") == ""


def test_with_the_users_summary_the_tier_line_changes_in_its_section_only(tmp_path) -> None:
    from finesub.llm.length_tier import TIER_PROFILE_ONLY

    root = _store(tmp_path, {"甲组": None})
    report = _gated(root, True, {**_APPEND, "section": "约定", "content": "[字幕长度] 偏长"})
    assert report.skipped[0].reason == TIER_PROFILE_ONLY
    assert _gated(root, True, {**_APPEND, "section": "档案", "content": "[字幕长度] 偏长"}).applied
    assert _tier_of(root, "甲组") == "偏长"
    assert _gated(root, True, {"op": "update", "id": _tier_handle(root), "line": "[字幕长度] 偏短",
                               "reason": "r"}).applied
    assert _tier_of(root, "甲组") == "偏短"
    report = _gated(root, True, {"op": "move_line", "id": _tier_handle(root), "section": "待归类", "reason": "r"})
    assert report.applied == [] and report.skipped[0].reason == TIER_PROFILE_ONLY
    assert _gated(root, True, {"op": "remove", "id": _tier_handle(root), "reason": "r"}).applied
    assert _tier_of(root, "甲组") == ""


def test_the_flows_that_maintain_a_style_still_change_its_tier(tmp_path) -> None:
    """`ingest`, `repair` and a human edit apply without the gate: the user's words change the tier there."""

    root = _store(tmp_path, {"甲组": "偏长"})
    assert _gated(root, None, {"op": "update", "id": _tier_handle(root), "line": "[字幕长度] 偏短", "reason": "r"}).applied
    assert _tier_of(root, "甲组") == "偏短"


class _TierClient:
    """Proposes, on the handles of the prompt it is given, to set the style's tier to 偏短 and add a convention."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
        import re

        from finesub.llm.client import LLMCallResult, LLMRole

        self.calls.append(messages)
        text = "\n".join(str(message.get("content", "")) for message in messages)
        subject = re.search(r"^# 甲组.*?<!-- (@k\d+)", text, re.M).group(1)
        tier = re.search(r"\[字幕长度\] \S+\s+<!-- (@k\d+)", text)
        change = ({"op": "update", "id": tier.group(1), "line": "[字幕长度] 偏短", "reason": "r"} if tier else
                  {"op": "append_lines", "entry": subject, "section": "档案", "content": "[字幕长度] 偏短", "reason": "r"})
        convention = {"op": "append_lines", "entry": subject, "section": "约定", "content": "[自称] 自称一律用「我」",
                      "reason": "r"}
        content = "<knowledge_proposals>\n" + "\n".join(json.dumps(row, ensure_ascii=False)
                                                        for row in (change, convention)) + "\n</knowledge_proposals>"
        return LLMCallResult(content=content, role=LLMRole.GENERAL_CAPABLE, model="fake", fallback_used=False,
                             raw_response={})


def test_a_refined_update_holds_the_tier_line_unless_the_user_wrote_the_summary(tmp_path) -> None:
    """Wired into the update: without the user's summary a change to the tier is skipped with the reason beside
    the other skips and is not asked again; with it the change lands, on a style with a tier or without."""

    from finesub.llm.knowledge.update import run_knowledge_update
    from finesub.llm.length_tier import TIER_HELD

    from .test_llm_knowledge_update import FakeTokenCounter, _write_task_outputs

    for tier in ("偏长", None):
        for user_summary in (False, True):
            workdir = tmp_path / f"{tier or 'none'}-{user_summary}"
            workdir.mkdir()
            final_srt, _paths, refined_srt = _write_task_outputs(workdir, with_refined=True)
            root = _store(workdir, {"甲组": tier})
            client = _TierClient()
            report = run_knowledge_update(
                final_srt=final_srt, refined_srt=refined_srt, task_id="t", task_summary="这套风格要短句",
                knowledge_root=root, token_counter=FakeTokenCounter(), client=client, style_names=["甲组"],
                user_summary=user_summary,
            )
            knowledge_report = report["chunks"][0]["knowledge_report"]
            held = [record for record in knowledge_report["skipped"] if record["reason"] == TIER_HELD]
            assert len(held) == (0 if user_summary else 1), (tier, user_summary)
            assert _tier_of(root, "甲组") == ("偏短" if user_summary else tier or ""), (tier, user_summary)
            assert len(client.calls) == 1, "a held change is not a rejection to re-decide"


def test_only_a_misplaced_tier_goes_back_to_the_model() -> None:
    """Without the user's summary the tier may not change at all; with it a held change is a tier moved or put
    in the wrong section, which the proposal repair round can put right."""
    from finesub.llm.knowledge.update import _is_rejection
    from finesub.llm.length_tier import TIER_HELD, TIER_PROFILE_ONLY

    assert not _is_rejection({"reason": TIER_HELD})
    assert _is_rejection({"reason": TIER_PROFILE_ONLY})


def test_a_task_hands_the_update_whether_the_user_wrote_its_summary(tmp_path, monkeypatch) -> None:
    out = tmp_path / "clip.srt"
    (tmp_path / "clip-translated.srt").write_text("existing", encoding="utf-8")
    updates: list[dict] = []
    monkeypatch.setattr(orchestration, "run_post_correction_knowledge_update", lambda **kw: updates.append(kw))
    for summary in ("", "这套风格要短句"):
        orchestration.run_correction_task(CorrectionRequest(
            stable_json=tmp_path / "clip-stable.json", output_path=out, audio_path=tmp_path / "clip.wav",
            postprocess_profile=None, knowledge="update", task_summary=summary,
        ))
    assert [update["user_summary"] for update in updates] == [False, True]
    assert updates[0]["task_summary"].startswith("LLM subtitle correction task")
