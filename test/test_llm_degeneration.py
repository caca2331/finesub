"""Signs of a degenerate answer (`finesub.llm.degeneration`): pass 2's mergeable lines left apart."""

from __future__ import annotations

import json
from dataclasses import replace

from finesub.llm.chunking import Seam, SubtitleSegment
from finesub.llm.degeneration import UNMERGED_MIN_SEAMS, pass2_signals, tight
from finesub.llm.output_protocol import TranslatedCsvSegment
from finesub.llm.routing.profiles import resolve_profile

from .test_llm_grouping import window

PROFILE = resolve_profile("text", "none", "quality")


def lines(count, seam=Seam.MAY, step=1.0, duration=0.5):
    """Lines 1..count, ``step`` s apart and ``duration`` s long: the defaults leave 0.5 s between two lines (a
    loose seam), ``step=0.7`` 0.2 s (a tight one)."""
    return [SubtitleSegment(str(n), n * step, n * step + duration, f"あ{n}", seam=Seam.NONE if n == 1 else seam)
            for n in range(1, count + 1)]


def rows(groups):
    return [TranslatedCsvSegment(tuple(group), 0.0, 1.0, "x", "x") for group in groups]


def apart(segments, merged=0):
    """Rows leaving every line apart but the first ``merged`` seams."""
    ids = [line.id for line in segments]
    return rows([ids[: merged + 1], *([line] for line in ids[merged + 1:])])


def test_counts_the_mergeable_lines_merged_with_the_line_before():
    segments = lines(5)
    signals = pass2_signals(window("0001", segments), rows([["1", "2"], ["3"], ["4", "5"]]))
    assert (signals.may_seams, signals.may_merged) == (4, 2)
    # a line not read 可并 does not count, merged or not
    plain = [replace(line, seam=Seam.NONE) if line.id == "2" else line for line in segments]
    signals = pass2_signals(window("0001", plain), rows([["1", "2"], ["3"], ["4", "5"]]))
    assert (signals.may_seams, signals.may_merged) == (3, 1)


def test_the_overlap_carried_in_is_not_the_window_s_own():
    segments = lines(4)
    signals = pass2_signals(window("0002", segments[2:], overlap=segments[:2]), rows([["3", "4"]]))
    assert (signals.may_seams, signals.may_merged) == (2, 1), "line 3 follows the overlap's line 2"


def test_a_tight_seam_is_a_short_pause_and_two_short_lines():
    def seam(before, line):
        return tight(SubtitleSegment("2", *line, "b"), SubtitleSegment("1", *before, "a"))

    assert seam((0.0, 1.0), (1.3, 2.0)), "0.3 s of pause is tight"
    assert not seam((0.0, 1.0), (1.31, 2.0))
    assert seam((0.0, 2.0), (2.1, 5.0)), "the two lines together exactly 5 s"
    assert not seam((0.0, 2.0), (2.1, 5.01))
    assert seam((0.0, 2.0), (1.5, 3.0)), "an overlap reads as no pause"
    segments = lines(5, step=0.7)
    signals = pass2_signals(window("0001", segments), rows([["1", "2"], ["3"], ["4", "5"]]))
    assert (signals.tight_seams, signals.tight_merged) == (4, 2)
    assert pass2_signals(window("0001", lines(5)), rows([["1", "2"]])).tight_seams == 0


def test_unmerged_with_enough_tight_seams_needs_almost_none_of_them_merged():
    count = UNMERGED_MIN_SEAMS + 1  # the first line has no seam: UNMERGED_MIN_SEAMS mergeable ones, all tight
    segments = lines(count, step=0.7)
    assert pass2_signals(window("0001", segments), apart(segments)).signals == ["unmerged"]
    assert pass2_signals(window("0001", segments), apart(segments, 2)).unmerged, "2 of 20 is 10%"
    assert not pass2_signals(window("0001", segments), apart(segments, 3)).unmerged
    fewer = lines(count - 1, step=0.7)
    assert not pass2_signals(window("0001", fewer), apart(fewer)).unmerged, "19 mergeable seams are too few"


def test_the_tight_seams_merged_decide_where_the_window_merged_little_overall():
    # 30 seams, the first 10 tight: 3 merged overall (10%) all among the tight ones (30%) -- a model that merges
    # little but still merges the seams a normal answer merges most is not degenerate
    tight_part, loose_part = lines(11, step=0.7), lines(20)
    loose_part = [replace(line, id=str(11 + n), start=line.start + 20, end=line.end + 20, seam=Seam.MAY)
                  for n, line in enumerate(loose_part, start=1)]
    segments = tight_part + loose_part
    signals = pass2_signals(window("0001", segments), apart(segments, 3))
    assert (signals.may_seams, signals.tight_seams, signals.tight_merged) == (30, 10, 3)
    assert not signals.unmerged
    assert pass2_signals(window("0001", segments), apart(segments, 1)).unmerged, "1 of 10 tight ones"


def test_with_few_tight_seams_only_an_answer_that_merged_none_is_unmerged():
    count = UNMERGED_MIN_SEAMS + 1
    loose = lines(count)  # 0.5 s pauses: no tight seam
    assert pass2_signals(window("0001", loose), apart(loose)).signals == ["unmerged"]
    assert not pass2_signals(window("0001", loose), apart(loose, 1)).unmerged, "leaving loose seams may be right"
    # seven tight seams are still too few to read
    seven = lines(8, step=0.7) + [replace(line, id=str(8 + n), start=line.start + 20, end=line.end + 20, seam=Seam.MAY)
                                  for n, line in enumerate(lines(13), start=1)]
    signals = pass2_signals(window("0001", seven), apart(seven, 1))
    assert (signals.may_seams, signals.tight_seams) == (20, 7) and not signals.unmerged


def test_eight_tight_seams_are_enough_to_read_and_seven_are_not():
    def window_with(tight_count):
        # 20 seams: loose ones first, then `tight_count` tight ones; the answer merges the first, a loose one
        loose = lines(20 - tight_count)
        close = [replace(line, id=str(len(loose) + n), start=line.start + 40, end=line.end + 40, seam=Seam.MAY)
                 for n, line in enumerate(lines(tight_count + 1, step=0.7), start=1)]
        segments = loose + close
        signals = pass2_signals(window("0001", segments), apart(segments, 1))
        assert (signals.may_seams, signals.may_merged, signals.tight_seams, signals.tight_merged) == (20, 1, tight_count, 0)
        return signals.unmerged

    assert window_with(8), "none of eight tight seams merged"
    assert not window_with(7), "too few tight seams to read, and one loose seam merged"


def test_windows_left_unmerged_or_asked_again_are_notes_in_the_stage_summary_not_warnings():
    from types import SimpleNamespace

    from finesub.reporting import reporting_to
    from finesub.llm.stages.correction.run import _report_correction_summary

    said = []
    reporter = SimpleNamespace(summary=lambda stage, metrics: said.append(("summary", dict(metrics))),
                               warning=lambda *args, **kwargs: said.append(("warning", args)))
    run = SimpleNamespace(token_rows=[], progress=None, repair_rounds=0, content_filter_recoveries=0,
                          unmerged_windows=["0003", "0001"], older_prompt_windows={},
                          redone_windows=[("0004", ["unmerged"]), ("0002", ["zero_thinking", "unmerged"])],
                          inline_proofread=False, proofread_windows={}, speaker_book=None)
    with reporting_to(reporter):
        _report_correction_summary(run)
    assert said == [("summary", {"调用": 0, "重试": 0, "修复轮": 0, "内容过滤": 0,
                                 "重答": "2 窗（0002 没思考、可并几乎没并，0004 可并几乎没并）",
                                 "可并几乎没并": "2 窗（0001, 0003）"})]


# --- a call that did not think ------------------------------------------------------------------------------

from finesub.llm.client import LLMCallResult, agent_usage_payload  # noqa: E402
from finesub.llm.degeneration import Draw, Pass2Signals, Thinking, pick_after_retry, reports_thinking, thinking_of  # noqa: E402,E501
from finesub.llm.routing.config import LLMRole  # noqa: E402


def api_result(thoughts=None, level="medium", **fields):
    usage = {"promptTokenCount": 1000, "candidatesTokenCount": 200,
             **({"thoughtsTokenCount": thoughts} if thoughts is not None else {})}
    return LLMCallResult(content="x", role=LLMRole.AUDIO_MULTIMODAL, model="m", fallback_used=False,
                         raw_response={"usageMetadata": usage}, thinking_level=level, **fields)


def agent_result(thinking=0, *, reported=True, effort="medium", resumable=True, turn=None):
    usage = {"input_tokens": 1000, "output_tokens": 200, "reasoning_output_tokens": thinking}
    attempt = {"driver": "agy", "thinking_reported": reported, "isolation": {"reasoning_effort": effort},
               "usage": {**usage, **({"turn_reasoning_output_tokens": turn} if turn is not None else {})}}
    return LLMCallResult(content="x", role=LLMRole.AUDIO_MULTIMODAL, model="m", fallback_used=False,
                         raw_response={"usage": agent_usage_payload(usage)}, backend="local_agent",
                         thinking_level="high", execution_attempts=[attempt], resumable=resumable)


def test_only_a_call_asked_to_think_past_low_skips_it_by_answering_without():
    assert Thinking(0, "medium").skipped and Thinking(0, "high").skipped and Thinking(0, "", 8192).skipped
    for level in ("low", "LOW", "minimal", "none", "off", ""):
        assert not Thinking(0, level).skipped, level
    assert not Thinking(None, "high").skipped and not Thinking(5, "high").skipped
    assert Thinking(0, "low").fine and Thinking(5, "high").fine and Thinking(None, "low").fine
    assert not Thinking(None, "high").fine and not Thinking(0, "high").fine


def test_who_reports_thinking():
    assert reports_thinking("gemini_rest", {})
    assert reports_thinking("local_agent", {}, {"thinking_reported": True})
    # a Claude model thinks adaptively whichever CLI runs it, as on the API below
    assert not reports_thinking("local_agent", {}, {"thinking_reported": True, "reported_model": "claude-haiku-5-5"})
    # the factory Claude Code targets ask by alias, so the driver decides, not the name
    assert not reports_thinking("local_agent", {}, {"thinking_reported": True, "driver": "claude-code", "reported_model": "haiku"})
    assert not reports_thinking("local_agent", {"completion_tokens_details": {"reasoning_tokens": 0}}, {})
    assert reports_thinking("openai_compatible", {"completion_tokens_details": {"reasoning_tokens": 0}})
    assert not reports_thinking("anthropic", {"completion_tokens_details": {}})
    # adaptive thinking: a counted zero is the model's call, not a degenerate answer
    assert not reports_thinking("anthropic", {"completion_tokens_details": {"reasoning_tokens": 0}})


def test_how_much_a_call_thought():
    assert thinking_of(api_result()) == Thinking(0, "medium"), "Gemini leaves a zero count out"
    assert thinking_of(api_result(300)).tokens == 300
    empty = LLMCallResult(content="x", role=LLMRole.AUDIO_MULTIMODAL, model="m", fallback_used=False,
                          raw_response={}, thinking_level="high")
    assert thinking_of(empty).tokens is None, "no usage is a gap, not a call that skipped thinking"
    assert thinking_of(agent_result()) == Thinking(0, "medium"), "the effort the CLI was handed, not the knob"
    assert thinking_of(agent_result(reported=False)).tokens is None
    assert thinking_of(agent_result(4000, resumable=False)).fine, "a running total only overstates the turn"
    assert thinking_of(agent_result(0, resumable=False)).skipped, "and its 0 is this turn's 0 too"
    assert thinking_of(agent_result(4000, resumable=False, turn=0)) == Thinking(0, "medium")


def test_which_answer_a_window_keeps_after_asking_again():
    thought, skipped, unknown = Thinking(800, "medium"), Thinking(0, "medium"), Thinking(None, "medium")
    unmerged, merged = Pass2Signals(30, 0), Pass2Signals(30, 12)
    no_thought = Draw(skipped)
    assert no_thought.held_for() == ["zero_thinking"] and Draw(skipped, patched=True).held_for() == []
    assert Draw(thought, unmerged).held_for() == ["unmerged"]
    assert Draw(skipped, unmerged).held_for() == ["zero_thinking", "unmerged"]
    assert Draw(skipped, unmerged, patched=True).held_for() == ["unmerged"], "a patch leaves the merging as it was"
    # only the zero-thinking sign: as before the shared retry
    assert pick_after_retry(no_thought, Draw(thought), True) == "retry"
    assert pick_after_retry(no_thought, Draw(thought), False) == "original", "a retry that thought but failed"
    assert pick_after_retry(no_thought, Draw(skipped), True) == "original", "neither thought: the first, on a tie"
    assert pick_after_retry(no_thought, Draw(unknown), True) == "original", "a retry that cannot be judged"
    assert pick_after_retry(no_thought, Draw(Thinking(None, "low")), True) == "retry", "not asked to think"
    # pass 2's sign too
    assert pick_after_retry(Draw(thought, unmerged), Draw(thought, merged), True) == "retry"
    assert pick_after_retry(Draw(thought, unmerged), Draw(thought, unmerged), True) == "original"
    assert pick_after_retry(Draw(thought, unmerged), Draw(thought, merged), False) == "original"
    assert pick_after_retry(Draw(skipped, unmerged), Draw(thought, unmerged), True) == "retry", "2 signs to 1"
    assert pick_after_retry(Draw(thought, unmerged), Draw(skipped, merged), True) == "original", "1 to 1, any kind"
    assert pick_after_retry(Draw(thought, unmerged), Draw(unknown, merged), True) == "retry", \
        "a missing count is a gap in the books unless the first was held for not thinking"
    # held for both: a retry that cannot be judged counts as not thinking, and merged it is still one sign to two
    assert pick_after_retry(Draw(skipped, unmerged), Draw(unknown, merged), True) == "retry"
    assert pick_after_retry(Draw(skipped, unmerged), Draw(unknown, unmerged), True) == "original"


def correction_run(tmp_path, monkeypatch, answers, *, feedback: bool = False):
    """One two-line window through the correction stage; ``answers`` gives each call's (translation, thoughts),
    or an exception to raise. ``feedback``: the task collects feedback, and each answer reports its translation
    as an uncertainty."""

    from finesub.llm.stages.correction import execute_correction_windows

    from .conftest import setattr_correction
    from .support.fakes import FakeTokenCounter
    from .support.speaker_turns import labelled_stable, reply

    tmp_path.mkdir(parents=True, exist_ok=True)
    stable = labelled_stable(tmp_path, [None, None], texts=["そうだね", "でもさ"])
    calls = []

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            calls.append(kwargs)
            answer = answers[len(calls) - 1]
            if isinstance(answer, Exception):
                raise answer
            translation, thoughts = answer
            result = api_result(thoughts)
            said = (f'\n<task_update_feedback>{{"uncertainties": ["{translation}"]}}</task_update_feedback>'
                    if feedback else "")
            return LLMCallResult(
                content=reply(f"sub|1|そうだね|{translation}|high|", "sub|2|でもさ|不过|high|") + said,
                role=role, model="fake", fallback_used=False,
                raw_response={**result.raw_response, "candidates": [{"finishReason": "STOP"}]}, thinking_level="medium")

    setattr_correction(monkeypatch, "RoleClient", Client)
    output = execute_correction_windows(
        stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
        max_retries_per_window=0, max_replacements_per_window=0, profile=PROFILE,
        task_artifact_dir=tmp_path / "art", length_scale=1.0, task_update_feedback=feedback)
    ledger = [json.loads(line) for path in tmp_path.rglob("correction-windows.jsonl")
              for line in path.read_text(encoding="utf-8").splitlines()]
    return output.read_text(encoding="utf-8"), calls, [row for row in ledger if "content" in row]


def test_a_correction_window_that_did_not_think_is_asked_once_more_in_a_fresh_session(tmp_path, monkeypatch):
    srt, calls, ledger = correction_run(tmp_path, monkeypatch, [("是啊", None), ("对呀", 900)])
    assert len(calls) == 2, "past a budget of one call: the retry is not one of its attempts"
    assert not calls[0].get("fresh_session") and calls[1]["fresh_session"] is True
    assert "对呀" in srt and "是啊" not in srt
    assert ledger[-1]["redo"]["chosen"] == "retry"
    assert ledger[-1]["redo"]["original"]["thinking"] == {"tokens": 0, "level": "medium"}


def test_the_first_answer_stays_when_the_retry_does_not_think_either_or_fails(tmp_path, monkeypatch):
    srt, calls, ledger = correction_run(tmp_path / "a", monkeypatch, [("是啊", None), ("对呀", None)])
    assert len(calls) == 2 and "是啊" in srt and ledger[-1]["redo"]["chosen"] == "original"
    srt, calls, ledger = correction_run(tmp_path / "b", monkeypatch, [("是啊", None), RuntimeError("down")])
    assert "是啊" in srt and ledger[-1]["redo"]["retry_error"] == "RuntimeError: down"
    srt, calls, ledger = correction_run(tmp_path / "c", monkeypatch, [("是啊", 700)])
    assert len(calls) == 1 and "redo" not in ledger[-1]


def test_the_knowledge_update_reads_the_feedback_of_the_answer_the_window_kept(tmp_path, monkeypatch):
    """The feedback is written when an answer is committed: a window that keeps its first answer after the retry
    hands on that answer's feedback, not the retry's, which came last."""

    from finesub.llm.knowledge.feedback import aggregate_task_update_feedback

    def uncertainties(root):
        return aggregate_task_update_feedback([root / "art"]).merged_uncertainties()

    srt, calls, ledger = correction_run(tmp_path / "a", monkeypatch, [("是啊", None), ("对呀", None)], feedback=True)
    assert len(calls) == 2 and ledger[-1]["redo"]["chosen"] == "original"
    assert uncertainties(tmp_path / "a") == ["是啊"]
    srt, calls, ledger = correction_run(tmp_path / "b", monkeypatch, [("是啊", None), ("对呀", 900)], feedback=True)
    assert ledger[-1]["redo"]["chosen"] == "retry" and uncertainties(tmp_path / "b") == ["对呀"]


def test_a_judge_answer_that_did_not_think_is_asked_once_more_and_a_leak_is_read_by_its_answer(tmp_path):
    from .support.speaker_judge import FakeClient, judge_files, run

    from finesub.speech.diarization import judgments as jd

    stable = judge_files(tmp_path)

    class Client(FakeClient):
        def __init__(self, thoughts, leak=False):
            super().__init__()
            self.thoughts, self.leak, self.kwargs = list(thoughts), leak, []

        def complete(self, role, messages, **kwargs):
            self.kwargs.append(kwargs)
            result = super().complete(role, messages, **kwargs)
            text = ("thought\nI will write `<judge></judge>` and <tracks>.\n" if self.leak else "") + result.content
            return LLMCallResult(content=text, role=role, model="fake", fallback_used=False,
                                 raw_response=api_result(self.thoughts.pop(0)).raw_response, thinking_level="medium")

    client = Client([None, 600, 500])
    report = run(stable, client, tmp_path, cut=((1, 4), (5, 8)))
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert report.failed == 0 and len(client.kwargs) == 3
    assert client.kwargs[1].get("fresh_session") is True and "fresh_session" not in client.kwargs[2]
    assert windows[0]["redo"]["chosen"] == "retry" and "redo" not in windows[1]
    (tmp_path / "leak").mkdir()
    stable = judge_files(tmp_path / "leak")
    leaking = Client([400, 400], leak=True)
    run(stable, leaking, tmp_path / "leak", cut=((1, 4), (5, 8)))
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert all(window["status"] == "success" and window["thought_stripped"] > 0 for window in windows)


def test_a_proofread_answer_that_did_not_think_is_asked_once_more_and_a_leak_is_read_by_its_answer(tmp_path):
    from .test_llm_proofread import FakeClient, proofread, rows, windows

    class Client(FakeClient):
        def __init__(self, replies):
            super().__init__()
            self.replies = list(replies)

        def complete(self, role, messages, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            text, thoughts = self.replies.pop(0)
            return LLMCallResult(content=text, role=role, model="fake", fallback_used=False,
                                 raw_response=api_result(thoughts).raw_response, thinking_level="medium")

    first, second = "<proofread>\n1|1|行一|译一|x\n</proofread>", "<proofread>\n1|1|行一|第一|y\n</proofread>"
    client = Client([(first, None), (second, 300)])
    result = proofread(tmp_path, client, rows(2), windows([1, 2]), retries=0)
    assert result.segments[0].translation == "第一" and len(client.calls) == 2
    assert client.calls[1]["fresh_session"] is True and len(client.calls[1]["messages"]) == 2
    client = Client([("thought\nThe <proofread> block.\n" + first, 300)])
    result = proofread(tmp_path / "leak", client, rows(2), windows([1, 2]))
    assert result.segments[0].translation == "译一" and len(client.calls) == 1


def test_a_stop_during_the_retry_keeps_the_answer_already_held(tmp_path):
    from .test_llm_proofread import FakeClient as ProofreadClient, proofread, rows, windows
    from .support.speaker_judge import FakeClient, judge_files, run

    from finesub.speech.diarization import judgments as jd

    quota = RuntimeError("429 RESOURCE_EXHAUSTED quota")

    class Judge(FakeClient):
        def __init__(self):
            super().__init__()
            self.models = ["first", "second"]

        def complete(self, role, messages, **kwargs):
            if kwargs.get("fresh_session"):
                raise quota
            result = super().complete(role, messages, **kwargs)
            return LLMCallResult(content=result.content, role=role, model="first", fallback_used=False,
                                 raw_response=api_result().raw_response, thinking_level="medium")

    stable = judge_files(tmp_path)
    report = run(stable, Judge(), tmp_path, cut=((1, 8),))
    window = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]
    assert report.stopped and window["status"] == "success" and window["model"] == "first"
    assert window["redo"]["chosen"] == "original" and window["redo"]["retry_error"].startswith("stopped")

    class Proofread(ProofreadClient):
        def complete(self, role, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("fresh_session"):
                raise quota
            return LLMCallResult(content="<proofread>\n1|1|行一|译一|x\n</proofread>", role=role, model="fake",
                                 fallback_used=False, raw_response=api_result().raw_response, thinking_level="medium")

    (tmp_path / "proof").mkdir()
    result = proofread(tmp_path / "proof", Proofread(), rows(2), windows([1, 2]))
    entry = json.loads((tmp_path / "proof" / "proofread.json").read_text(encoding="utf-8"))["windows"]["0001"]
    assert result.segments[0].translation == "译一" and entry["redo"]["retry_error"].startswith("stopped")


def test_the_kept_original_is_credited_to_the_model_that_wrote_it(tmp_path):
    from .support.speaker_judge import FakeClient, judge_files, run

    from finesub.speech.diarization import judgments as jd

    class Judge(FakeClient):
        def complete(self, role, messages, **kwargs):
            result = super().complete(role, messages, **kwargs)
            return LLMCallResult(content=result.content, role=role, fallback_used=False,
                                 model="second" if kwargs.get("fresh_session") else "first",
                                 raw_response=api_result().raw_response, thinking_level="medium")

    stable = judge_files(tmp_path)
    run(stable, Judge(), tmp_path, cut=((1, 8),))
    window = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]
    assert window["redo"]["chosen"] == "original" and window["model"] == "first"


# --- pass 2 left its mergeable lines apart ------------------------------------------------------------------

def grouped_run(tmp_path, monkeypatch, answers, *, count=UNMERGED_MIN_SEAMS + 1, retries=0, resume=False):
    """One three pass window of ``count`` lines of one speaker, every seam after the first read 可并 (loose: a
    second's pause), through the correction stage. ``answers`` gives each call's reply -- "apart" (no line
    merged), "merged" (the first two merged), "bad" (fails validation), "short" (apart, the last line missing),
    "patch" (a repair round's patch adding that line) or an exception -- with its thinking."""

    from .conftest import setattr_correction
    from .support.fakes import FakeTokenCounter
    from .support.speaker_turns import HEADER, grouped_reply, labelled_stable, speaker
    from .test_llm_grouping import window as grouping_window

    from finesub.llm.chunking import load_segments_from_stable_json
    from finesub.llm.grouping import pass2_windows
    from finesub.llm.prompt_constants import Thresholds
    from finesub.llm.stages.correction import execute_correction_windows

    tmp_path.mkdir(parents=True, exist_ok=True)
    stable = labelled_stable(tmp_path, [speaker(1)] * count, texts=[f"行{n}" for n in range(1, count + 1)])
    segments = load_segments_from_stable_json(stable)
    verdicts = {str(n): "may_join" for n in range(2, count + 1)}
    windows, grouping = pass2_windows([grouping_window("0001", segments)], segments, verdicts, frozenset(),
                                      different=lambda a, b: False, thresholds=Thresholds.at(1.0))
    calls = []

    def rows(merged):
        out = ["sub|1,2|行1行2|译1|high|"] if merged else ["sub|1|行1|译1|high|", "sub|2|行2|译2|high|"]
        return out + [f"sub|{n}|行{n}|译{n}|high|" for n in range(3, count + 1)]

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            calls.append(kwargs)
            answer = answers[len(calls) - 1]
            if isinstance(answer, Exception):
                raise answer
            kind, thoughts = answer
            content = ("no csv here" if kind == "bad"
                       else f"<patch>\n{HEADER}\n{rows(False)[-1]}\n</patch>" if kind == "patch"
                       else grouped_reply(*rows(False)[:-1]) if kind == "short"
                       else grouped_reply(*rows(kind == "merged")))
            return LLMCallResult(content=content, role=role, model="fake", fallback_used=False,
                                 raw_response={**api_result(thoughts).raw_response,
                                               "candidates": [{"finishReason": "STOP"}]}, thinking_level="medium")

    setattr_correction(monkeypatch, "RoleClient", Client)
    output = execute_correction_windows(
        stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
        max_retries_per_window=retries, max_replacements_per_window=0, profile=PROFILE,
        task_artifact_dir=tmp_path / "art", windows_override=windows, grouping=grouping, length_scale=1.0)
    ledger = [json.loads(line) for path in tmp_path.rglob("correction-windows.jsonl")
              for line in path.read_text(encoding="utf-8").splitlines()]
    return output.read_text(encoding="utf-8"), calls, [row for row in ledger if "content" in row]


def test_a_pass_2_answer_that_merged_none_is_asked_once_more_and_the_merged_retry_kept(tmp_path, monkeypatch):
    srt, calls, ledger = grouped_run(tmp_path, monkeypatch, [("apart", 900), ("merged", 900)])
    assert len(calls) == 2 and calls[1]["fresh_session"] is True
    record = ledger[-1]
    assert record["redo"]["signals"] == ["unmerged"] and record["redo"]["chosen"] == "retry"
    assert record["redo"]["original"]["degeneration"]["may_merged"] == 0
    assert record["degeneration"]["signals"] == [], "the committed answer is the retry"
    assert "\n译2\n" not in srt, "lines 1 and 2 are one subtitle"


def test_the_first_pass_2_answer_stays_when_the_retry_is_no_better_or_fails(tmp_path, monkeypatch):
    srt, calls, ledger = grouped_run(tmp_path / "a", monkeypatch, [("apart", 900), ("apart", 800)])
    assert len(calls) == 2 and ledger[-1]["redo"]["chosen"] == "original" and "\n译2\n" in srt
    assert ledger[-1]["degeneration"]["signals"] == ["unmerged"], "kept, and still a note in the summary"
    # the retry gets no repair round: a reply that fails validation keeps the first, even with repairs left
    srt, calls, ledger = grouped_run(tmp_path / "b", monkeypatch, [("apart", 900), ("bad", 900)], retries=2)
    assert len(calls) == 2 and ledger[-1]["redo"]["chosen"] == "original"
    assert calls[0]["max_repair_attempts"] == 2 and calls[1]["max_repair_attempts"] == 0, \
        "nor does a task runtime inside the retry's one call"
    srt, calls, ledger = grouped_run(tmp_path / "c", monkeypatch, [("apart", 900), RuntimeError("down")])
    assert ledger[-1]["redo"]["retry_error"] == "RuntimeError: down" and ledger[-1]["redo"]["chosen"] == "original"


def test_a_patched_answer_is_read_for_unmerged_but_not_for_not_thinking(tmp_path, monkeypatch):
    srt, calls, ledger = grouped_run(tmp_path, monkeypatch, [("short", 900), ("patch", None), ("merged", 900)],
                                     retries=1)
    assert len(calls) == 3 and calls[2]["fresh_session"] is True
    redo = ledger[-1]["redo"]
    assert redo["signals"] == ["unmerged"], "the patch reported no thinking: no sign on a few rows"
    assert redo["original"]["patched"] is True and redo["chosen"] == "retry"


def test_both_signs_share_the_one_retry(tmp_path, monkeypatch):
    srt, calls, ledger = grouped_run(tmp_path, monkeypatch, [("apart", None), ("apart", None)])
    assert len(calls) == 2, "not thinking and leaving the lines apart: still one retry"
    assert ledger[-1]["redo"]["signals"] == ["zero_thinking", "unmerged"]
    assert ledger[-1]["redo"]["chosen"] == "original"


def test_a_pass_2_answer_passing_on_its_last_attempt_still_gets_the_retry(tmp_path, monkeypatch):
    srt, calls, ledger = grouped_run(tmp_path, monkeypatch, [("bad", 900), ("apart", 900), ("merged", 900)],
                                     retries=1)
    assert len(calls) == 3 and ledger[-1]["redo"]["chosen"] == "retry" and "\n译2\n" not in srt


def test_a_window_with_too_few_mergeable_lines_is_not_asked_again(tmp_path, monkeypatch):
    srt, calls, ledger = grouped_run(tmp_path, monkeypatch, [("apart", 900)], count=UNMERGED_MIN_SEAMS)
    assert len(calls) == 1 and "redo" not in ledger[-1]


def test_a_resumed_task_replays_the_kept_answer_without_asking_again(tmp_path, monkeypatch):
    grouped_run(tmp_path, monkeypatch, [("apart", 900), ("merged", 900)])
    srt, calls, ledger = grouped_run(tmp_path, monkeypatch, [])
    assert calls == [] and "\n译2\n" not in srt


def test_only_a_grouped_window_reads_pass_2_s_signs():
    from types import SimpleNamespace

    from finesub.llm.stages.correction.attempts import _pass2

    segments = lines(UNMERGED_MIN_SEAMS + 1)
    grouped = replace(window("0001", segments), grouped=True)
    answer = SimpleNamespace(segments=apart(segments))
    assert _pass2(SimpleNamespace(grouping=object()), grouped, answer).unmerged
    assert _pass2(SimpleNamespace(grouping=None), grouped, answer) is None, "a single pass"
    assert _pass2(SimpleNamespace(grouping=object()), replace(grouped, grouped=False), answer) is None, \
        "three passes' window on the single pass prompt for want of pass 1"
