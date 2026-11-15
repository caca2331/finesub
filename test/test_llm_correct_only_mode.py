"""A Chinese source only corrects (docs/llm_harness_behavior.md「只纠错」): how a task decides it, records it and
resumes it, and what it switches -- the prompt set, the style, pass 3, the knowledge base."""

from __future__ import annotations

import json

import pytest

from finesub.llm.client import RoleClient
from finesub.llm.correction_request import CorrectionRequest
from finesub.llm.prompt_compose import compose_correction_system
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.routing.config import (
    GEMINI_FREE_TIER,
    CapabilityTier,
    LLMRole,
    ModelEndpoint,
    RoleModelConfig,
)
from finesub.llm.routing.profiles import parse_profile_id, resolve_profile
from finesub.llm.stages import three_pass
from finesub.llm.stages.correction.commit import WINDOW_PLAN_FILENAME

QUALITY = resolve_profile("audio", "local", "quality")


def _stable(tmp_path, langs, *, told=None):
    segments = [{"id": str(i), "start": i * 2.0, "end": i * 2.0 + 1.0, "text": text, **({"lang": lang} if lang else {})}
                for i, (lang, text) in enumerate(langs, start=1)]
    metadata = {"asr_align": {"language": told}} if told else {}
    path = tmp_path / "clip-stable.json"
    path.write_text(json.dumps({"metadata": metadata, "segments": segments}, ensure_ascii=False), encoding="utf-8")
    return path


def _request(stable, **overrides):
    return CorrectionRequest(stable_json=stable, output_path=stable.with_name("out.srt"),
                             profile=overrides.pop("profile", QUALITY), **overrides)


# --- deciding and recording -------------------------------------------------------------------------------------


@pytest.mark.parametrize("langs, told, expected", [
    ([("zh", "大家好"), ("zh", "开始吧")], None, (True, "zh")),
    # the language weighted by characters; a short Japanese line does not flip a Chinese episode
    ([("zh", "今天我们来玩这个游戏"), ("ja", "はい")], None, (True, "zh")),
    ([("ja", "こんにちは"), ("ja", "はい")], None, (False, "ja")),
    ([("yue", "點解")], None, (False, "yue")),  # Cantonese still translates
    ([(None, "大家好")], "zh", (True, "zh")),  # what ASR was told wins
])
def test_a_new_task_reads_its_mode_from_the_source_language(tmp_path, langs, told, expected):
    warnings: list[str] = []
    stable = _stable(tmp_path, langs, told=told)
    assert three_pass.task_correct_only(_request(stable), tmp_path / "art", warn=warnings.append) == expected
    assert warnings == []


def test_a_source_whose_language_cannot_be_told_translates_and_says_how_to_fix_it(tmp_path):
    warnings: list[str] = []
    stable = _stable(tmp_path, [(None, "大家好")])
    assert three_pass.task_correct_only(_request(stable), tmp_path / "art", warn=warnings.append) == (False, None)
    assert len(warnings) == 1 and "--language zh" in warnings[0]


def test_the_mode_is_recorded_and_a_resumed_task_keeps_it(tmp_path, monkeypatch):
    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    stable, art = _stable(tmp_path, [("zh", "大家好")]), tmp_path / "art"
    profile = resolve_profile("audio", "local", "quality", correct_only=True)
    choice = three_pass.task_choice(_request(stable, profile=profile), art, [], source_language="zh")
    assert (choice.correct_only, choice.source_language) == (True, "zh")
    three_pass.record_choice(art, choice)
    payload = json.loads((art / three_pass.MODE_FILENAME).read_text(encoding="utf-8"))
    assert payload["version"] == three_pass.MODE_VERSION == 1, "an optional field: the record's version stays"
    assert (payload["correct_only"], payload["source_language"]) == (True, "zh")
    assert three_pass.recorded_choice(art) == choice
    # the source now reads as Japanese (say, re-transcribed): the task keeps what it recorded
    _stable(tmp_path, [("ja", "こんにちは")])
    assert three_pass.task_correct_only(_request(stable), art, warn=lambda _: None) == (True, "zh")
    # --no-resume decides again
    assert three_pass.task_correct_only(_request(stable, resume=False), art, warn=lambda _: None) == (False, "ja")


def test_a_record_from_before_the_field_and_a_task_started_before_records_translate(tmp_path):
    stable = _stable(tmp_path, [("zh", "大家好")])
    old = tmp_path / "old"
    old.mkdir()
    (old / three_pass.MODE_FILENAME).write_text(
        json.dumps({"version": 1, "mode": "single", "proofread": False}), encoding="utf-8")
    assert three_pass.recorded_choice(old).correct_only is False
    assert three_pass.task_correct_only(_request(stable), old, warn=lambda _: None) == (False, None)
    started = tmp_path / "started"
    started.mkdir()
    (started / WINDOW_PLAN_FILENAME).write_text("{}", encoding="utf-8")
    assert three_pass.task_correct_only(_request(stable), started, warn=lambda _: None) == (False, None)
    assert three_pass.task_choice(_request(stable), started, []).correct_only is False


def test_the_profile_id_names_the_mode_only_when_it_is_on():
    assert "correct_only" not in QUALITY.profile_id
    profile = resolve_profile("audio", "local", "quality", correct_only=True)
    assert profile.profile_id.endswith(",correct_only=true")
    assert parse_profile_id(profile.profile_id) == profile
    assert parse_profile_id(QUALITY.profile_id) == QUALITY


# --- what it switches -------------------------------------------------------------------------------------------


def test_pass_3_runs_only_when_asked():
    def choose(proofread, correct_only):
        return three_pass.choose(difficulty="quality", grouped_prompts=True, speaker_judge="text",
                                 proofread=proofread, correct_only=correct_only).proofread

    assert choose("auto", False) and not choose("auto", True)
    assert choose("on", True), "an explicit on still runs (and warns)"


def test_the_grouped_check_reads_the_correct_only_sets(monkeypatch):
    from finesub.llm.routing import config as routing_config

    names = []

    def fake_has_grouped(name):
        names.append(name)
        return True

    monkeypatch.setattr("finesub.llm.prompt_variants.has_grouped_variant", fake_has_grouped)
    monkeypatch.setattr(routing_config, "role_config_for", lambda *_: RoleModelConfig(
        role=LLMRole.AUDIO_MULTIMODAL, endpoint_chain=(), test_endpoint=None, variant="capableC"))
    assert three_pass.grouped_prompts(resolve_profile("audio", "local", "quality", correct_only=True))
    assert names == ["capableC-polish"]


def test_the_client_hands_each_candidate_its_correct_only_set(monkeypatch):
    strong = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.6-flash", target_id="strong")
    lite = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash-lite", target_id="lite")
    configs = {LLMRole.AUDIO_MULTIMODAL: RoleModelConfig(
        role=LLMRole.AUDIO_MULTIMODAL, endpoint_chain=(strong, lite), test_endpoint=strong,
        variant="capableC", variant_overrides={"lite": "basicA"})}
    calls: list[str] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        if model == "gemini/gemini-3.6-flash":
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(role_configs=configs, max_retries=0, rate_limiter=ModelRateLimiter(enabled=False))
    result = client.complete(LLMRole.AUDIO_MULTIMODAL,
                             lambda variant: calls.append(variant) or [{"role": "user", "content": variant}],
                             correct_only=True)
    assert calls == ["capableC-polish", "basicA-polish"]
    assert result.variant == "basicA-polish" and result.capability_tier is CapabilityTier.BASIC


def test_a_correct_only_prompt_takes_its_style_block_but_no_extra_style():
    """The block a correct-only task renders holds only styles written for it (`render_style_block`); the
    composer passes it on and drops `--extra-style`, a translation request."""
    style = "\n本任务的风格约定：保持口语\n"
    translate = compose_correction_system(QUALITY, variant="capableC", style_block=style, extra_style="多用短句",
                                          length_scale=1.0)
    correct = compose_correction_system(QUALITY, variant="capableC-polish", style_block=style,
                                        extra_style="多用短句", length_scale=1.0)
    assert "保持口语" in translate and "多用短句" in translate
    assert "保持口语" in correct and "多用短句" not in correct


# --- styles written for a correct-only task (`[任务] 中文纠错`, owner 2026-11-05) -------------------------------


def _store(tmp_path, **styles):
    """A knowledge store with these styles: name -> 档案 lines."""
    from finesub.llm.knowledge.maintain import main as maintain_main
    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    root = tmp_path / "knowledge"
    for name, profile in styles.items():
        assert maintain_main(["--root", str(root), "new", "style", name, "--intro", f"{name}简介"]) == 0
        repo = KnowledgeRepo.open(root)
        subject = next(s for s in repo.subjects("style") if (s.payload.get("surface") or s.payload.get("key")) == name)
        edit_subject(repo, subject, f"# {name}\n\n{name}简介\n\n## 档案\n\n{profile}\n\n## 约定\n\n- [加注] {name}的谐音梗加括注\n")
    return root


def test_the_tag_is_read_from_the_profile_section():
    from finesub.llm.knowledge.style import for_correct_only

    assert for_correct_only("# x\n\n## 档案\n\n[任务] 中文纠错\n")
    assert for_correct_only("# x\n\n## 档案\n\n[任务] 翻译、中文纠错\n")
    assert not for_correct_only("# x\n\n## 档案\n\n[适用范围] 中文纠错\n")
    assert not for_correct_only("# x\n\n## 约定\n\n[任务] 中文纠错\n")


def test_a_correct_only_task_injects_only_tagged_styles(tmp_path, monkeypatch):
    from finesub.llm import correction_translation as orchestration

    root = _store(tmp_path, 甲组="[任务] 中文纠错", 乙组="[字幕长度] 偏长")
    warnings: list[str] = []
    monkeypatch.setattr(orchestration, "_correct_only_warning", warnings.append)
    stable = _stable(tmp_path, [("zh", "大家好")])
    profile = resolve_profile("audio", "local", "quality", correct_only=True)

    def selection(**kwargs):
        warnings.clear()
        return orchestration._task_style_selection(_request(stable, profile=profile, knowledge_root=root, **kwargs))

    named = selection(style="甲组,乙组", style_mode="update")
    assert (named.names, named.tier_names, named.mode) == (("甲组",), ("乙组",), "read")
    assert any("乙组" in w and "[字幕长度]" in w for w in warnings) and any("update" in w for w in warnings)
    # named by the config (`style` there), an untagged style is not read at all
    from finesub.llm.knowledge import style as style_module
    monkeypatch.setattr(style_module, "resolve_style_names", lambda style, **_: ("乙组",))
    monkeypatch.setattr(style_module, "style_was_named", lambda style, **_: True)
    configured = selection()
    assert (configured.auto, configured.names, configured.tier_names) == (False, (), ())
    # auto picks among the tagged ones alone
    assert orchestration._auto_style_catalog(root, correct_only=True).keys == ("甲组",)
    assert set(orchestration._auto_style_catalog(root).keys) == {"甲组", "乙组"}


def test_a_pick_recorded_while_translating_is_not_injected_into_correct_only(tmp_path, monkeypatch):
    """A task that ran as a translation and is judged again with `--no-resume` keeps its `auto` record (a fast
    window's pick); a style it names that is not written for correct-only is dropped, and said so."""
    from finesub.llm import correction_translation as orchestration
    from finesub.llm.knowledge.style import StyleChoice, StyleSelection

    root = _store(tmp_path, 甲组="[任务] 中文纠错", 乙组="[字幕长度] 偏长")
    warnings: list[str] = []
    monkeypatch.setattr(orchestration, "_correct_only_warning", warnings.append)
    auto = StyleSelection(mode="read", names=(), auto=True)

    kept = orchestration._drop_untagged_record(root, auto.resolved(StyleChoice(outcome="selected", key="甲组").names))
    assert kept.names == StyleChoice(outcome="selected", key="甲组").names and warnings == []
    dropped = orchestration._drop_untagged_record(root, auto.resolved(StyleChoice(outcome="selected", key="乙组").names))
    assert dropped.names == () and len(warnings) == 1 and "乙组" in warnings[0]


def test_a_preparation_made_for_a_translation_is_not_reused_by_correct_only(tmp_path):
    """The context pack names terms and picks a style for the task's mode, so the mode gates its reuse; a
    translation task's key is the retrieval alone, as before."""
    from finesub.llm.research import research_semantics_id

    assert research_semantics_id(QUALITY) == "local"
    assert research_semantics_id(resolve_profile("audio", "local", "quality", correct_only=True)) == "local,correct_only"


def test_a_correct_only_block_has_no_factory_default_and_its_own_header(tmp_path):
    from finesub.llm.knowledge.style import render_style_block

    root = _store(tmp_path, 甲组="[任务] 中文纠错")
    assert render_style_block(root, (), correct_only=True) == ""
    assert "出厂默认" in render_style_block(root, ())
    block = render_style_block(root, ("甲组",), correct_only=True)
    assert "谐音梗加括注" in block and "只用在 polished" in block and "翻译" not in block


def test_only_people_write_the_task_line(tmp_path):
    """Owner 2026-11-07: no model preview says what `[任务]` is, and the knowledge update holds every change to it,
    without feeding it back; people still see its note and edit it."""
    from finesub.llm.knowledge.node.proposals import HIDDEN_LABEL_HELD, apply_model_proposals
    from finesub.llm.knowledge.node.render import HandleMap, render_subject
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.knowledge.update import _is_rejection, _style_line_gate

    root = _store(tmp_path, 甲组="[任务] 中文纠错", 乙组="[作者] 乙")
    repo = KnowledgeRepo.open(root)
    subjects = {s.payload.get("surface"): s for s in repo.subjects("style")}
    note = "这套风格写给哪种任务"
    assert note in render_subject(repo.store, subjects["甲组"].local_id, mode="human", preview="full")
    handles = HandleMap()
    for subject in subjects.values():
        model_view = render_subject(repo.store, subject.local_id, mode="prompt", handles=handles, preview="full")
        assert note not in model_view and "[字幕长度]" in model_view
    task_handle = next(line for line in render_subject(repo.store, subjects["甲组"].local_id, mode="prompt",
                                                       handles=handles).splitlines()
                       if line.startswith("[任务]")).rsplit("<!-- ", 1)[1].split()[0]
    profile = {"op": "append_lines", "category": "style", "section": "档案", "reason": "r"}
    for user_summary in (False, True):
        for change in ({**profile, "entry": "乙组", "content": "[任务] 翻译、中文纠错"},
                       {"op": "update", "id": task_handle, "line": "[任务] 翻译", "reason": "r"},
                       {"op": "remove", "id": task_handle, "reason": "r"},
                       {"op": "move_line", "id": task_handle, "section": "约定", "reason": "r"}):
            block = "<knowledge_proposals>\n" + json.dumps(change, ensure_ascii=False) + "\n</knowledge_proposals>"
            report = apply_model_proposals(block, repo=repo, allow_categories=("style",), task_id="t",
                                           knowledge_read_rev=repo.rev, handles=handles,
                                           line_gate=_style_line_gate(user_summary))
            assert report.applied == [] and [r.reason for r in report.skipped] == [HIDDEN_LABEL_HELD], change
            assert not _is_rejection({"reason": HIDDEN_LABEL_HELD})


@pytest.mark.parametrize("knowledge, source, warned", [
    ("read", "default", False),  # the default turns off silently
    ("read", "explicit", True),
    ("update", "preset:p", True),
    ("none", "explicit", False),
])
def test_a_correct_only_task_keeps_out_of_the_knowledge_base(tmp_path, monkeypatch, knowledge, source, warned):
    """Owner 2026-11-06: a Chinese task neither reads nor writes the base, whatever was asked; the user who asked
    is told once; nor is the model asked for feedback to update it with, even where the caller asked for it. A
    translating task keeps its knowledge mode."""
    from finesub.llm import correction_translation as orchestration

    warnings: list[str] = []
    monkeypatch.setattr(orchestration, "_correct_only_warning", warnings.append)
    refined = tmp_path / "refined.srt" if knowledge == "update" else None
    for langs, correct_only in (([("zh", "大家好")], True), ([("ja", "こんにちは")], False)):
        warnings.clear()
        request, _ = orchestration._with_task_mode(
            _request(_stable(tmp_path, langs), knowledge=knowledge, refined_srt=refined,
                     switch_sources={"knowledge": source}, collect_feedback=True), tmp_path / "art")
        assert request.knowledge == ("none" if correct_only else knowledge)
        assert request.collect_feedback is not correct_only
        assert (request.refined_srt is None) is (correct_only or refined is None)
        assert any("知识库" in w for w in warnings) is (warned and correct_only)


def test_a_plan_of_a_chinese_source_previews_the_correct_only_task(tmp_path):
    """The dry run: the window prompt is the correct-only one (no translation column, no style), the preparation
    task is told so, and `correction_mode` says it. No style is written for it here: none is injected, not even
    the factory default."""

    from finesub.llm import correction_translation as orchestration

    stable = _stable(tmp_path, [("zh", "大家好。"), ("zh", "开始吧。")])
    seen: dict = {}
    orchestration.run_correction_task(CorrectionRequest(
        stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan",
        knowledge_root=tmp_path / "knowledge", profile=resolve_profile("text", "local", "quality"),
        test_profile=True, on_plan=seen.update, extra_style="多用短句",
    ))
    window = seen["correction_messages"][0][0]["content"]
    assert "`type|position|corrected_text|polished|conf|note`" in window
    assert "<style" not in window and "风格约定" not in window and "多用短句" not in window
    assert "中文 ASR 字幕纠错的全局准备代理" in seen["research_messages"][0][0]["content"]
    assert (seen["correction_mode"]["correct_only"], seen["correction_mode"]["source_language"]) == (True, "zh")
    assert seen["profile"]["id"].endswith("correct_only=true")


def test_a_dry_run_without_a_route_declaration_previews_the_correct_only_set(monkeypatch):
    from finesub.llm.prompt_artifacts import _cell_variant
    from finesub.llm.routing import config as routing_config

    def no_declaration(*_):
        raise KeyError("no declaration")

    monkeypatch.setattr(routing_config, "role_config_for", no_declaration)
    assert _cell_variant(QUALITY) is None, "translating: the tier default, as before"
    assert _cell_variant(resolve_profile("audio", "local", "quality", correct_only=True)) == "capableC-polish"


def test_a_correct_only_task_drops_only_hallucinations_before_correcting():
    """Pass 1 (and the single pass's shape section) of a Chinese source drops no line that is only filler: the
    correction writes it into `corrected_text` and leaves `polished` empty. A translating task still drops it."""
    from finesub.llm.prompt_constants import Thresholds
    from finesub.llm.shape_criteria import shape_criteria

    for words, thresholds in (("pass1", None), ("single", Thresholds.at(1.0))):
        drop = shape_criteria(words, same_person="", correct_only=True, thresholds=thresholds)["drop"]
        assert "垫词" not in drop.splitlines()[0] and drop.startswith("- 确信的 ASR 幻觉")
        assert shape_criteria(words, same_person="", correct_only=False, thresholds=thresholds)["drop"].startswith(
            "- 整行只有不带信息的垫词或口吃")
    single = compose_correction_system(QUALITY, variant="capableC-polish", length_scale=1.0)
    assert "整行只有垫词或口吃的行不 drop：按「纠错与内容取舍」第 4 条写" in single
    grouped = compose_correction_system(QUALITY, variant="capableC-polish", grouped=True, length_scale=1.0)
    assert "上一步已经丢掉了确信的套话幻觉，" in grouped
