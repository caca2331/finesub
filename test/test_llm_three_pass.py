"""Three passes in a task (`finesub.llm.stages.three_pass`): which way a task runs and keeps running, what pass 1
hands pass 2, and what the run records -- the window ledger, the fast decision, the dry run, the task report and
the resume of an unfinished pass 3."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from finesub.llm.chunking import load_segments_from_stable_json
from finesub.llm.prompt_constants import Thresholds
from finesub.llm.correction_request import CorrectionRequest, CorrectionRequestError, resolved_request
from finesub.llm.grouping import Grouping
from finesub.llm.length_tier import TaskLengthScale
from finesub.llm.output_protocol import TranslatedCsvSegment
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages import three_pass
from finesub.llm.stages.correction.commit import WINDOW_PLAN_FILENAME, _window_input_hash
from finesub.llm.task_report import render_task_report
from finesub.speech.diarization import judgments as sidecar
from finesub.subtitles.stable_rows import read_rows

from .test_llm_grouping import window
from .test_llm_speaker_turns import FakeTokenCounter, labelled_stable, speaker

QUALITY = resolve_profile("text", "none", "quality")

#: A dry run's ratio with nothing to resolve it from (`length_tier.task_length_scale`).
STANDARD_SCALE = TaskLengthScale(1.0, "default")


def request(tmp_path, stable=None, **overrides):
    return CorrectionRequest(stable_json=stable or tmp_path / "clip-stable.json", output_path=tmp_path / "out.srt",
                             profile=overrides.pop("profile", QUALITY), **overrides)


# --- the choice ----------------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty, grouped, judge, proofread, expected", [
    # the judge on: the three-pass split at both capable tiers, with or without acoustic labels (without them the
    # judge skips and every window gets the single pass prompt); pass 3 under auto at quality only
    ("quality", True, "text", "auto", ("three-pass", True)),
    ("intermediate", True, "text", "auto", ("three-pass", False)),
    ("intermediate", True, "audio", "on", ("three-pass", True)),
    # the judge off: one pass whatever the tier, the prompt set or the labels; pass 3 still by tier
    ("quality", True, "off", "auto", ("single", True)),
    ("quality", True, "off", "off", ("single", False)),
    ("quality", False, "off", "auto", ("single", True)),
    ("intermediate", True, "off", "auto", ("single", False)),
    ("intermediate", True, "off", "on", ("single", True)),
    ("efficiency", True, "off", "auto", ("single", False)),
    ("efficiency", False, "off", "on", ("single", True)),
])
def test_the_judge_decides_the_split_and_the_tier_pass_3(difficulty, grouped, judge, proofread, expected):
    choice = three_pass.choose(difficulty=difficulty, grouped_prompts=grouped, speaker_judge=judge,
                               proofread=proofread)
    assert (choice.mode, choice.proofread) == expected and choice.reason and choice.speaker_judge == judge


@pytest.mark.parametrize("judge, mode", [("off", "single"), ("text", "three-pass")])
def test_acoustic_labels_do_not_change_the_choice(tmp_path, monkeypatch, judge, mode):
    """The judge off: one pass with acoustic labels too, the fused capableC reading their 换人 in the turn column."""
    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    intermediate = resolve_profile("text", "none", "intermediate")
    for rows in (read_rows(labelled_stable(tmp_path, [speaker(1), speaker(2)])), [{"id": "1", "text": "x"}]):
        choice = three_pass.task_choice(request(tmp_path, profile=intermediate, speaker_judge=judge), None, rows)
        assert (choice.mode, choice.proofread) == (mode, False)


@pytest.mark.parametrize("difficulty, grouped", [("efficiency", False), ("efficiency", True), ("quality", False),
                                                 ("intermediate", False)])
def test_the_judge_where_the_split_cannot_run_is_refused_not_run_with_one_pass(difficulty, grouped):
    with pytest.raises(CorrectionRequestError, match="three-pass split"):
        three_pass.choose(difficulty=difficulty, grouped_prompts=grouped, speaker_judge="text", proofread="auto")


EFFICIENCY = resolve_profile("text", "none", "efficiency")


def test_the_correction_module_refuses_a_new_judge_at_efficiency_and_lets_its_own_resume_through(tmp_path,
                                                                                                 monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2)])
    rows, art = read_rows(stable), tmp_path / "art"
    judged = request(tmp_path, stable, profile=EFFICIENCY, speaker_judge="text")
    with pytest.raises(CorrectionRequestError, match="efficiency"):
        three_pass.task_choice(judged, art, rows)
    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: False)
    with pytest.raises(CorrectionRequestError, match="grouped version"):
        three_pass.task_choice(request(tmp_path, stable, speaker_judge="audio"), art, rows)
    # a task recorded with the judge (three passes at quality), resumed at efficiency after the quota ran out
    recorded = three_pass.PassChoice("three-pass", True, "quality, speaker judge text", "text")
    three_pass.record_choice(art, recorded)
    assert three_pass.task_choice(judged, art, rows) == recorded, "the judge it recorded: no error"
    with pytest.raises(CorrectionRequestError):
        three_pass.task_choice(replace(judged, speaker_judge="audio"), art, rows)
    with pytest.raises(CorrectionRequestError):
        three_pass.task_choice(replace(judged, resume=False), art, rows)
    assert three_pass.task_choice(replace(judged, speaker_judge="off"), art, rows) == recorded


def test_a_choice_torn_mid_write_leaves_the_recorded_one(tmp_path, monkeypatch):
    from finesub_bootstrap import fsops

    art = tmp_path / "art"
    recorded = three_pass.PassChoice("three-pass", True, "quality, speaker judge text", "text")
    three_pass.record_choice(art, recorded)

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fsops, "replace_path", crash)
    with pytest.raises(KeyboardInterrupt):
        three_pass.record_choice(art, three_pass.PassChoice("single", False, "no judge", "off"))
    assert three_pass.recorded_choice(art) == recorded, "a resume still runs three passes"


def test_every_prompt_set_the_cell_may_answer_with_needs_a_grouped_version(monkeypatch):
    import finesub.llm.routing.config as config

    def cell(variant, overrides=None):
        monkeypatch.setattr(config, "role_config_for",
                            lambda *args, **kwargs: SimpleNamespace(variant=variant, variant_overrides=overrides or {}))
        return three_pass.grouped_prompts(QUALITY)

    assert cell("capableC") is True
    assert cell("basicA") is False, "another variant: one pass without the judge (refused with it)"
    assert cell("capableC", {"some-target": "basicA"}) is False
    monkeypatch.setattr(config, "role_config_for", lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("x")))
    assert three_pass.grouped_prompts(QUALITY) is False


def test_pass_1_is_the_judge_and_without_it_there_is_none_said_once():
    warnings = []
    assert three_pass.pass1_shape("text", warn=warnings.append) == sidecar.SHAPE_TRACKED
    assert three_pass.pass1_shape("audio", warn=warnings.append) == sidecar.SHAPE_TRACKED
    assert warnings == [], "the judge on: its pass 1, nothing to say"
    assert three_pass.pass1_shape("off", warn=warnings.append) is None
    assert len(warnings) == 1 and "单 pass" in warnings[0]


def test_a_resumed_task_follows_the_choice_it_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2)])
    rows, art = read_rows(stable), tmp_path / "art"
    first = three_pass.task_choice(request(tmp_path, stable, speaker_judge="text"), art, rows)
    assert first.mode == "three-pass" and first.proofread
    assert first.speaker_judge == "text", "the judge goes into the record"
    three_pass.record_choice(art, first)
    assert three_pass.recorded_choice(art) == first
    # later runs ask for something else: the task keeps its own choice
    for later in (request(tmp_path, stable, proofread="off"),
                  request(tmp_path, stable, speaker_judge="text",
                          profile=resolve_profile("text", "none", "efficiency"))):
        assert three_pass.task_choice(later, art, rows) == first
    off = three_pass.PassChoice("three-pass", False, "r")
    three_pass.record_choice(art, off)
    assert three_pass.task_choice(request(tmp_path, stable), art, rows) == off, "a task's --llm-proofread off holds"
    # three passes on acoustic labels alone, recorded before the judge off meant one pass: still three passes
    # (its windows then have no pass 1, `pass1_shape`)
    unjudged = three_pass.PassChoice("three-pass", False, "intermediate, with acoustic labels", "off")
    three_pass.record_choice(art, unjudged)
    assert three_pass.task_choice(request(tmp_path, stable), art, rows) == unjudged
    single = three_pass.PassChoice("single", False, "difficulty intermediate runs one pass", "off")
    three_pass.record_choice(art, single)
    assert three_pass.task_choice(request(tmp_path, stable), art, rows) == single, "one pass recorded: one pass"
    # --no-resume chooses again
    assert three_pass.task_choice(request(tmp_path, stable, resume=False), art, rows).mode == "single"
    again = request(tmp_path, stable, resume=False, speaker_judge="text")
    assert three_pass.task_choice(replace(again, proofread="off"), art, rows).proofread is False
    assert three_pass.task_choice(again, art, rows) == first


def test_a_task_whose_windows_started_before_the_choice_was_recorded_runs_one_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2)])
    art = tmp_path / "art"
    art.mkdir()
    (art / WINDOW_PLAN_FILENAME).write_text("{}", encoding="utf-8")
    choice = three_pass.task_choice(request(tmp_path, stable), art, read_rows(stable))
    assert (choice.mode, choice.proofread) == ("single", False)
    assert three_pass.task_choice(request(tmp_path, stable, proofread="on"), art, read_rows(stable)).proofread


def test_a_record_that_does_not_read_counts_as_none(tmp_path):
    (tmp_path / three_pass.MODE_FILENAME).write_text(json.dumps({"version": 1, "mode": "both", "proofread": True}),
                                                     encoding="utf-8")
    assert three_pass.recorded_choice(tmp_path) is None
    assert three_pass.recorded_choice(None) is None


def test_an_unknown_proofread_mode_is_refused():
    with pytest.raises(CorrectionRequestError, match="proofread"):
        resolved_request(CorrectionRequest(stable_json="x", output_path="y.srt", proofread="sometimes"))


# --- pass 1 -> pass 2 ----------------------------------------------------------------------------


def acts(monkeypatch, answers, missing=()):
    monkeypatch.setattr(sidecar, "read_acts", lambda *args, **kwargs: sidecar.ShapeActs(
        answers, {row: "w" for row, act in answers.items() if act == "drop"}, frozenset(missing), frozenset()))


def test_without_a_pass_1_every_window_keeps_the_single_pass_prompt(tmp_path):
    stable = labelled_stable(tmp_path, [speaker(1)] * 4)
    segments = load_segments_from_stable_json(stable)
    windows = [window("0001", segments[:2]), window("0002", segments[2:])]
    out, grouping, record = three_pass.pass2_plan(windows, stable, None, warn=pytest.fail, thresholds=Thresholds.at(1.0))
    assert [w.grouped for w in out] == [False, False] and record["without_pass1"] == ["0001", "0002"]
    assert grouping.groups == tuple((s.id,) for s in segments)


def test_pass_1_groups_the_lines_and_a_join_across_this_runs_window_edge_is_broken_once(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1)] * 5)
    segments = load_segments_from_stable_json(stable)
    windows = [window("0001", segments[:3]), window("0002", segments[3:])]
    acts(monkeypatch, {"2": "join", "3": "drop", "4": "join", "5": "may_join"})
    warned = []
    out, grouping, record = three_pass.pass2_plan(windows, stable, sidecar.SHAPE_TRACKED, warn=warned.append, thresholds=Thresholds.at(1.0))
    assert [w.grouped for w in out] == [True, True]
    assert grouping.groups == (("1", "2"), ("4",), ("5",)) and grouping.dropped == ("3",)
    assert len(warned) == 1 and "4" in warned[0], "the judge cut its windows on an earlier run"
    assert record == {"shape": "tracked", "windows": 2, "without_pass1": [], "skipped": [], "dropped": 1, "joined": 1,
                      "refused": {"window": 1}}


def test_a_window_whose_core_lines_were_all_dropped_is_left_out(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1)] * 4)
    segments = load_segments_from_stable_json(stable)
    acts(monkeypatch, {"3": "drop", "4": "drop"})
    out, _grouping, record = three_pass.pass2_plan(
        [window("0001", segments[:2]), window("0002", segments[2:])], stable, sidecar.SHAPE_TRACKED,
        warn=pytest.fail, thresholds=Thresholds.at(1.0))
    assert [w.chunk_id for w in out] == ["0001"] and record["skipped"] == ["0002"]


def test_a_window_with_a_line_pass_1_left_unanswered_falls_back_as_a_whole(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1)] * 4)
    segments = load_segments_from_stable_json(stable)
    acts(monkeypatch, {"2": "join", "4": "join"}, missing={"3"})
    out, grouping, record = three_pass.pass2_plan(
        [window("0001", segments[:2]), window("0002", segments[2:])], stable, sidecar.SHAPE_TRACKED,
        warn=pytest.fail, thresholds=Thresholds.at(1.0))
    assert [w.grouped for w in out] == [True, False] and record["without_pass1"] == ["0002"]
    assert grouping.groups == (("1", "2"), ("3",), ("4",)), "the fallback window's acts are not used"


def test_a_resumed_task_whose_cell_lost_its_grouped_prompts_replays_its_grouped_windows_and_corrects_the_rest_on_one_pass(
        tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1)] * 4)
    segments = load_segments_from_stable_json(stable)
    acts(monkeypatch, {"2": "join", "4": "join"})
    warned = []
    out, grouping, record = three_pass.pass2_plan(
        [window("0001", segments[:2]), window("0002", segments[2:])], stable, sidecar.SHAPE_TRACKED,
        warn=warned.append, replayable=lambda w: w.chunk_id == "0001", thresholds=Thresholds.at(1.0))
    assert [w.grouped for w in out] == [True, False] and record["without_pass1"] == ["0002"]
    assert grouping.groups == (("1", "2"), ("3",), ("4",)) and len(warned) == 1 and "0002" in warned[0]


def test_grouping_that_fails_leaves_every_window_on_the_single_pass_prompt_with_a_warning(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1)] * 2)
    segments = load_segments_from_stable_json(stable)
    monkeypatch.setattr(sidecar, "read_acts", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")))
    warned = []
    out, _grouping, record = three_pass.pass2_plan([window("0001", segments)], stable, sidecar.SHAPE_TRACKED,
                                                   warn=warned.append, thresholds=Thresholds.at(1.0))
    assert [w.grouped for w in out] == [False] and record["without_pass1"] == ["0001"] and "bad" in warned[0]


def test_the_dry_run_previews_pass_2_on_the_asr_lines_under_the_grouped_prompt(tmp_path):
    from finesub.llm.prompt_artifacts import build_prompt_artifacts

    stable = labelled_stable(tmp_path, [speaker(1), speaker(2)])
    shown = build_prompt_artifacts(stable_json=stable, knowledge_enabled=False, counter=FakeTokenCounter(),
                                   profile=QUALITY, grouped=True, length_scale=STANDARD_SCALE)
    user = shown["correction_messages"][0][-1]["content"]
    assert "可并" not in "".join(line.split("|")[4] for line in user.splitlines() if line[:2] in ("1|", "2|"))
    assert shown["correction_basic_example_messages"] is None, "no basic variant has a grouped prompt set"
    plain = build_prompt_artifacts(stable_json=stable, knowledge_enabled=False, counter=FakeTokenCounter(),
                                   profile=QUALITY, length_scale=STANDARD_SCALE)
    assert plain["correction_messages"][0][0]["content"] != shown["correction_messages"][0][0]["content"]


@pytest.mark.parametrize("labels", [True, False])
def test_the_dry_run_previews_the_prompt_the_windows_will_get_judge_on_with_or_without_labels(tmp_path, monkeypatch,
                                                                                             labels):
    """The judge on records three passes either way; without labels it skips, every window runs one pass, and the
    dry run shows that prompt rather than the grouped one."""
    import finesub.llm.correction_translation as orchestration
    import finesub.llm.prompt_artifacts as artifacts
    from .test_speaker_judge_stage import judge_files, segments_of, task_request

    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    if labels:
        stable = judge_files(tmp_path)
    else:
        stable = tmp_path / "clip-stable.json"
        stable.write_text(json.dumps({"segments": segments_of([(None, 2, f"line {i}") for i in range(1, 9)])}),
                          encoding="utf-8")
    built, shown = {}, {}
    real = artifacts.build_prompt_artifacts
    monkeypatch.setattr(orchestration, "build_prompt_artifacts",
                        lambda **kwargs: built.update(grouped=kwargs["grouped"]) or real(**kwargs))
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="plan", on_plan=shown.update))
    assert shown["correction_mode"]["mode"] == "three-pass"
    assert built["grouped"] is labels
    assert ("可并" in shown["correction_mode"]["pass2_input"]) is labels


def test_a_three_pass_task_resumed_at_efficiency_is_previewed_on_one_pass(tmp_path):
    """Its cell (basicA) has no grouped version: the windows the ledger does not replay run one pass, so the
    preview -- which the module CLI builds on every run -- shows that prompt rather than failing to build a
    grouped one."""
    import finesub.llm.correction_translation as orchestration
    from .test_speaker_judge_stage import judge_files, task_request

    stable = judge_files(tmp_path)
    art = tmp_path / "art"
    three_pass.record_choice(art, three_pass.PassChoice("three-pass", True, "quality, speaker judge text", "text"))
    shown = {}
    orchestration.run_correction_task(replace(
        task_request(tmp_path, stable, mode="plan", on_plan=shown.update, task_artifact_dir=art), profile=EFFICIENCY))
    assert shown["correction_mode"]["mode"] == "three-pass"
    assert "no grouped version" in shown["correction_mode"]["pass2_input"]


# --- what the run records ------------------------------------------------------------------------


def test_a_windows_input_hash_carries_its_ratio_and_a_grouped_ones_a_fixed_marker():
    from finesub.llm.chunking import SubtitleSegment
    import hashlib

    from finesub.llm.chunking import render_window_preceding_as_csv, render_window_segments_as_csv

    plain = window("0001", [SubtitleSegment("1", 0.0, 1.0, "あ"), SubtitleSegment("2", 1.0, 2.0, "い")])
    old = "sha256:" + hashlib.sha256(
        f"{render_window_preceding_as_csv(plain)}\x1f{render_window_segments_as_csv(plain)}\x1f\x1fscale=1.0"
        .encode()).hexdigest()[:16]
    assert _window_input_hash(plain) == old
    # a grouped window and the fallback window of the same chunk never answer for each other, and a bump of
    # either prompt version moves neither hash
    grouped = replace(plain, grouped=True)
    marked = "sha256:" + hashlib.sha256(
        f"{render_window_preceding_as_csv(grouped)}\x1f{render_window_segments_as_csv(grouped)}\x1f\x1fgrouped\x1fscale=1.0"
        .encode()).hexdigest()[:16]
    assert _window_input_hash(grouped) == marked != _window_input_hash(plain)
    # a window at another length ratio is another input (its prompt and its validation differ), never replayed
    assert _window_input_hash(replace(plain, length_scale=0.8)) != _window_input_hash(plain)
    with pytest.raises(ValueError, match="length scale is not set"):
        _window_input_hash(replace(plain, length_scale=None))


def test_a_three_pass_window_record_names_a_missing_pass_1_and_every_record_its_prompt_version():
    from finesub.llm.prompt_compose import GROUPED_PROMPT_VERSION, PROMPT_VERSION
    from finesub.llm.stages.correction.attempts import _pass_fields
    from finesub.llm.stages.correction.commit import window_prompt_version

    from finesub.llm.chunking import SubtitleSegment

    grouping = Grouping((("1", "2"), ("3",)), {})
    rows = [TranslatedCsvSegment(("1", "3"), 0.0, 3.0, "x", "x")]
    grouped = replace(window("0001", [SubtitleSegment("1", 0.0, 2.0, "あ"), SubtitleSegment("3", 2.0, 3.0, "い")]),
                      grouped=True)
    assert _pass_fields(SimpleNamespace(grouping=None), grouped, None) == {}
    assert _pass_fields(SimpleNamespace(grouping=grouping), SimpleNamespace(grouped=False), None) == {
        "pass1": "missing"}
    assert _pass_fields(SimpleNamespace(grouping=grouping), grouped, SimpleNamespace(segments=rows)) == {
        "over_cap_rows": [["1", "2", "3"]],  # counted in original lines
        "degeneration": {"may_seams": 0, "may_merged": 0, "signals": []}}
    # the version a window record carries, whatever the task's mode: a grouped window its own
    assert window_prompt_version(grouped) == GROUPED_PROMPT_VERSION
    assert window_prompt_version(replace(grouped, grouped=False)) == PROMPT_VERSION


def test_the_replay_warning_counts_each_prompt_version_apart(tmp_path, reported):
    """A window is compared with its own constant (a grouped one with `GROUPED_PROMPT_VERSION`), counted as
    it is committed, and the run's one warning names each older version with its count."""
    import threading

    from finesub.llm.chunking import SubtitleSegment
    from finesub.llm.prompt_compose import GROUPED_PROMPT_VERSION, PROMPT_VERSION
    from finesub.llm.stages.correction.context import CorrectionRun, ResumeLedger
    from finesub.llm.stages.correction.run import _warn_older_prompt_windows

    def one(chunk_id, grouped=False):
        return replace(window(chunk_id, [SubtitleSegment(chunk_id, 0.0, 1.0, "あ")]), grouped=grouped)

    records = {"0001": "csv-old", "0002": "3pass-old", "0003": PROMPT_VERSION, "0004": GROUPED_PROMPT_VERSION,
               "0005": "csv-old"}
    run = SimpleNamespace(
        ledger=ResumeLedger(True, tmp_path / "correction-windows.jsonl", "f",
                            {key: {"prompt_version": version} for key, version in records.items()}),
        _tally_lock=threading.Lock(), older_prompt_windows={})
    for replayed in (one("0001"), one("0002", True), one("0003"), one("0004", True), one("0005")):
        CorrectionRun.note_replay(run, replayed)
    assert run.older_prompt_windows == {"csv-old": 2, "3pass-old": 1}
    _warn_older_prompt_windows(run)
    assert reported.codes() == ["correction-older-prompt-windows"]
    assert "3pass-old 1 窗，csv-old 2 窗" in reported.warnings[0].message
    assert "输入没变" in reported.warnings[0].impact


def test_a_fast_task_whose_pass_3_would_not_fit_one_window_does_not_go_fast(tmp_path):
    from finesub.llm.stages.plan import decide_fast_mode
    from test.test_llm_fast_plan import _stable_json

    options = dict(stable_json=_stable_json(tmp_path), fast="auto", profile=resolve_profile("text", "none", "quality"),
                   knowledge_root=tmp_path / "kb", knowledge_enabled=False, token_counter=FakeTokenCounter())
    fits = decide_fast_mode(**options, proofread_limits=SimpleNamespace(prompt_input_limit=1_000_000))
    assert fits.enabled and fits.to_metadata()["proofread_input_budget"] == 1_000_000
    tight = decide_fast_mode(**options, proofread_limits=SimpleNamespace(prompt_input_limit=10))
    assert not tight.enabled and "pass 3 input estimate" in tight.reason
    assert "proofread_input_budget" not in decide_fast_mode(**options).to_metadata()
    with pytest.raises(ValueError, match="pass 3"):
        decide_fast_mode(**{**options, "fast": "on"}, proofread_limits=SimpleNamespace(prompt_input_limit=10))


def test_the_task_report_tells_how_the_task_ran_what_pass_1_dropped_and_what_pass_3_did():
    records = [
        {"kind": "speaker_judge_report", "payload": {"windows": 2, "shape": "tracked",
                                                      "acts": {"join": 3, "may_join": 2, "drop": 1},
                                                      "dropped": [{"row": "7", "why": "垫词"}], "without_pass1": []}},
        {"kind": "correction_mode", "payload": {"mode": "three-pass", "proofread": True, "shape": "tracked",
                                                "windows": 2, "without_pass1": ["0002"], "skipped": [], "joined": 3,
                                                "dropped": 1, "refused": {"seconds": 1}}},
        {"kind": "proofread", "payload": {"status": "failed", "windows": 2, "reused": 0, "failed": ["0002"],
                                          "changes": [{"row": 1}], "stopped": "", "usage": {}}},
    ]
    text = render_task_report(records)
    assert "pass 1 (tracked): 3 join, 2 may_join, 1 drop" in text and "row 7 dropped: 垫词" in text
    assert "## Passes" in text and "1 on the single pass prompt for want of pass 1 (0002)" in text
    assert "joins kept apart: seconds 1" in text and "pass 3 on" in text
    assert "pass 3: failed, 2 window(s), 0 reused, 1 line(s) changed; not answered (pass 2's lines kept): 0002" in text
    single = render_task_report([{"kind": "correction_mode", "payload": {"mode": "single", "proofread": False,
                                                                          "reason": "no speaker labels"}}])
    assert "one pass (no speaker labels); pass 3 off" in single
    assert "## Passes" not in render_task_report([])


# --- the task ------------------------------------------------------------------------------------


def test_an_unfinished_pass_3_keeps_a_resumed_run_from_taking_the_finished_output(tmp_path):
    import finesub.llm.correction_translation as orchestration
    from finesub.llm.stages.proofread.run import mark_failed

    out, art = tmp_path / "clip.srt", tmp_path / "art"
    out.write_text("pass 2's", encoding="utf-8")
    finished = lambda **overrides: orchestration._finished_output(  # noqa: E731
        request(tmp_path, postprocess_profile=0, **overrides), out, artifact_dir=art, task_id="t")
    assert finished() == out
    mark_failed(art)
    assert finished() == out, "no recorded choice with pass 3: an old record owes nothing"
    three_pass.record_choice(art, three_pass.PassChoice("three-pass", True))
    assert finished() is None, "the windows replay from the ledger and pass 3 asks what it has no answer for"
    assert finished(resume=False) == out
    # a rerun of that pass 3 stopped half way leaves the status empty: still owed
    record = art / "proofread.json"
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), "status": ""}), encoding="utf-8")
    assert finished() is None
    three_pass.record_choice(art, three_pass.PassChoice("single", False))
    assert finished() == out, "chosen again without pass 3 (--no-resume): the stale record owes nothing"
    three_pass.record_choice(art, three_pass.PassChoice("three-pass", True))
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), "status": "done"}), encoding="utf-8")
    assert finished() == out


def test_a_three_pass_task_records_its_choice_and_hands_the_windows_pass_2s(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from .test_speaker_judge_stage import FakeClient, fast_whole_file, judge_files, task_request

    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    stable = judge_files(tmp_path)
    fast_whole_file(monkeypatch, orchestration)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    seen = {}

    def fake_execute(**kwargs):
        seen.update(grouping=kwargs["grouping"], proofread=kwargs["proofread"], title=kwargs["proofread_title"],
                    grouped=[w.grouped for w in kwargs["plan"].windows])
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    art = tmp_path / "out" / "clip.llm-artifacts"
    assert three_pass.recorded_choice(art).mode == "three-pass"
    assert seen["grouping"] is not None and seen["proofread"] is True and seen["title"] == "clip"
    assert seen["grouped"] == [True], "pass 1 answered: the window reads the grouped lines"
    records = [json.loads(line) for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    mode = next(record["payload"] for record in records if record["kind"] == "correction_mode")
    assert mode["mode"] == "three-pass" and mode["shape"] == "tracked"
    judge = next(record["payload"] for record in records if record["kind"] == "speaker_judge_report")
    assert judge["shape"] == "tracked", "the judge ran as pass 1"
    # A later run of the finished task owes the judge nothing more: the sidecar holds pass 1's units.
    assert not orchestration._judge_owed(task_request(tmp_path, stable), art)


def _run_seen(tmp_path, monkeypatch, request):
    """Run the task with the windows faked; what they were handed, and the task's artifacts."""
    import finesub.llm.correction_translation as orchestration
    from .test_speaker_judge_stage import FakeClient, fast_whole_file

    fast_whole_file(monkeypatch, orchestration)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    seen = {}

    def fake_execute(**kwargs):
        seen.update(grouping=kwargs["grouping"], proofread=kwargs["proofread"])
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(request)
    art = tmp_path / "out" / "clip.llm-artifacts"
    records = [json.loads(line) for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    return seen, art, records


def test_intermediate_with_the_judge_runs_the_tracked_split_without_pass_3(tmp_path, monkeypatch):
    from .test_speaker_judge_stage import judge_files, task_request

    stable = judge_files(tmp_path)
    seen, art, records = _run_seen(tmp_path, monkeypatch, replace(
        task_request(tmp_path, stable, mode="execute"), profile=resolve_profile("text", "none", "intermediate")))
    choice = three_pass.recorded_choice(art)
    assert (choice.mode, choice.proofread, choice.speaker_judge) == ("three-pass", False, "text")
    assert seen["grouping"] is not None and seen["proofread"] is False
    mode = next(record["payload"] for record in records if record["kind"] == "correction_mode")
    assert mode["shape"] == "tracked"


def test_a_one_pass_task_runs_as_before(tmp_path, monkeypatch):
    from .test_speaker_judge_stage import judge_files, task_request

    stable = judge_files(tmp_path)
    seen, art, records = _run_seen(tmp_path, monkeypatch, replace(
        task_request(tmp_path, stable, mode="execute"), profile=EFFICIENCY, speaker_judge="off"))
    assert seen == {"grouping": None, "proofread": False}
    assert three_pass.recorded_choice(art).mode == "single"
    assert not any(record["kind"] == "speaker_judge_report" for record in records)


def test_a_three_pass_task_recorded_without_the_judge_resumes_on_one_pass_with_one_warning(tmp_path, monkeypatch,
                                                                                           reported):
    """Recorded as three passes on acoustic labels alone (before the judge off meant one pass): there is no pass 1
    to run, so every window it still corrects gets the single pass prompt, said once; the judge does not run."""
    from .test_speaker_judge_stage import judge_files, task_request

    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    stable = judge_files(tmp_path)
    art = tmp_path / "out" / "clip.llm-artifacts"
    recorded = three_pass.PassChoice("three-pass", True, "quality, with acoustic labels", "off", 1.0, "default")
    three_pass.record_choice(art, recorded)
    seen, art, records = _run_seen(tmp_path, monkeypatch, replace(
        task_request(tmp_path, stable, mode="execute"), speaker_judge="off"))
    assert three_pass.recorded_choice(art) == recorded, "the task keeps its own choice"
    assert seen["grouping"] is not None and seen["proofread"] is True
    mode = next(record["payload"] for record in records if record["kind"] == "correction_mode")
    assert mode["mode"] == "three-pass" and mode["shape"] is None and mode["without_pass1"] == ["0001"]
    assert not any(record["kind"] == "speaker_judge_report" for record in records)
    assert reported.codes() == ["three-pass"] and "单 pass" in reported.joined()


def test_the_judge_at_efficiency_is_refused_before_any_work(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from .test_speaker_judge_stage import judge_files, task_request

    stable = judge_files(tmp_path)
    monkeypatch.setattr(orchestration, "run_research_stage", lambda **kwargs: pytest.fail("prepared"))
    with pytest.raises(CorrectionRequestError, match="three-pass split"):
        orchestration.run_correction_task(replace(task_request(tmp_path, stable, mode="execute"), profile=EFFICIENCY))
    assert not (tmp_path / "out" / "clip.llm-artifacts" / three_pass.MODE_FILENAME).exists()


def test_a_finished_task_is_not_judged_by_a_judge_it_did_not_record_where_the_split_cannot_run(tmp_path,
                                                                                               monkeypatch):
    """The finished-task shortcuts (reuse, or judge the finished task) come before `task_choice`: the refusal
    comes before them, and a judge the task recorded still goes through."""
    import finesub.llm.correction_translation as orchestration
    from .test_speaker_judge_stage import judge_files, task_request

    stable = judge_files(tmp_path)
    out = tmp_path / "out" / "clip.srt"
    out.parent.mkdir(parents=True)
    out.write_text("1\n00:00:00,000 --> 00:00:01,000\nline\n", encoding="utf-8")
    judged = []
    monkeypatch.setattr(orchestration, "_finished_output", lambda *args, **kwargs: out)
    monkeypatch.setattr(orchestration, "_judge_finished_task", lambda *args, **kwargs: judged.append(True))
    for owed in (False, True):
        monkeypatch.setattr(orchestration, "_judge_owed", lambda *args, owed=owed: owed)
        with pytest.raises(CorrectionRequestError, match="three-pass split"):
            orchestration.run_correction_task(
                replace(task_request(tmp_path, stable, mode="execute"), profile=EFFICIENCY))
    assert not judged
    three_pass.record_choice(tmp_path / "out" / "clip.llm-artifacts",
                             three_pass.PassChoice("three-pass", True, "quality, speaker judge text", "text"))
    orchestration.run_correction_task(replace(task_request(tmp_path, stable, mode="execute"), profile=EFFICIENCY))
    assert judged, "the judge it recorded"


def test_a_finished_three_pass_task_is_judged_with_pass_1s_shape(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from .test_speaker_judge_stage import FakeClient, run, judge_files, task_request

    stable = judge_files(tmp_path)
    run(stable, FakeClient(), tmp_path)  # a one-pass judge's units: no acts
    art = tmp_path / "art"
    assert not orchestration._judge_owed(task_request(tmp_path, stable), art), "no record: one pass, covered"
    three_pass.record_choice(art, three_pass.PassChoice("three-pass", True))
    assert orchestration._judge_owed(task_request(tmp_path, stable), art), "pass 1's units need their acts"
