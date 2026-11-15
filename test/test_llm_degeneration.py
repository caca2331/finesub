"""Signs of a degenerate answer (`finesub.llm.degeneration`): pass 2's mergeable lines left apart."""

from __future__ import annotations

import json
from dataclasses import replace

from finesub.llm.chunking import Seam, SubtitleSegment
from finesub.llm.degeneration import UNMERGED_MIN_SEAMS, pass2_signals
from finesub.llm.output_protocol import TranslatedCsvSegment
from finesub.llm.routing.profiles import resolve_profile

from .test_llm_grouping import window

PROFILE = resolve_profile("text", "none", "quality")


def lines(count, seam=Seam.MAY):
    return [SubtitleSegment(str(n), float(n), n + 0.5, f"あ{n}", seam=Seam.NONE if n == 1 else seam)
            for n in range(1, count + 1)]


def rows(groups):
    return [TranslatedCsvSegment(tuple(group), 0.0, 1.0, "x", "x") for group in groups]


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


def test_unmerged_needs_enough_mergeable_lines_and_almost_none_merged():
    count = UNMERGED_MIN_SEAMS + 1  # the first line has no seam: UNMERGED_MIN_SEAMS mergeable ones
    segments = lines(count)
    apart = rows([[line.id] for line in segments])
    assert pass2_signals(window("0001", segments), apart).signals == ["unmerged"]
    one = rows([["1", "2"], *([line.id] for line in segments[2:])])
    assert pass2_signals(window("0001", segments), one).signals == ["unmerged"], "1 of 30 is under 5%"
    two = rows([["1", "2"], ["3", "4"], *([line.id] for line in segments[4:])])
    assert pass2_signals(window("0001", segments), two).signals == []
    fewer = lines(count - 1)
    assert pass2_signals(window("0001", fewer), rows([[line.id] for line in fewer])).signals == []


def test_windows_left_unmerged_or_asked_again_are_notes_in_the_stage_summary_not_warnings():
    from types import SimpleNamespace

    from finesub.reporting import reporting_to
    from finesub.llm.stages.correction.run import _report_correction_summary

    said = []
    reporter = SimpleNamespace(summary=lambda stage, metrics: said.append(("summary", dict(metrics))),
                               warning=lambda *args, **kwargs: said.append(("warning", args)))
    run = SimpleNamespace(token_rows=[], progress=None, repair_rounds=0, content_filter_recoveries=0,
                          unmerged_windows=["0003", "0001"], rethought_windows=["0002"], older_prompt_windows={})
    with reporting_to(reporter):
        _report_correction_summary(run)
    assert said == [("summary", {"调用": 0, "重试": 0, "修复轮": 0, "内容过滤": 0, "可并几乎没并": "2 窗（0001, 0003）",
                                 "没思考、重答": "1 窗（0002）"})]


# --- a call that did not think ------------------------------------------------------------------------------

from finesub.llm.client import LLMCallResult, agent_usage_payload  # noqa: E402
from finesub.llm.degeneration import Thinking, pick_after_retry, reports_thinking, thinking_of  # noqa: E402
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
    assert not reports_thinking("local_agent", {"completion_tokens_details": {"reasoning_tokens": 0}}, {})
    assert reports_thinking("openai_compatible", {"completion_tokens_details": {"reasoning_tokens": 0}})
    assert not reports_thinking("anthropic", {"completion_tokens_details": {}})


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
    assert pick_after_retry(0, thought, 0) == "retry"
    assert pick_after_retry(0, thought, 2) == "original", "a retry that thought but failed"
    assert pick_after_retry(0, skipped, 0) == "original", "neither thought: the first, on a tie"
    assert pick_after_retry(3, skipped, 1) == "retry", "neither thought: fewer errors"
    assert pick_after_retry(0, unknown, 0) == "original"
    assert pick_after_retry(0, Thinking(None, "low"), 0) == "retry", "not asked to think: no count needed"


def correction_run(tmp_path, monkeypatch, answers):
    """One two-line window through the correction stage; ``answers`` gives each call's (translation, thoughts),
    or an exception to raise."""

    from finesub.llm.stages.correction import execute_correction_windows

    from .conftest import setattr_correction
    from .test_llm_speaker_turns import FakeTokenCounter, labelled_stable, reply

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
            return LLMCallResult(
                content=reply(f"sub|1|そうだね|{translation}|high|", "sub|2|でもさ|不过|high|"),
                role=role, model="fake", fallback_used=False,
                raw_response={**result.raw_response, "candidates": [{"finishReason": "STOP"}]}, thinking_level="medium")

    setattr_correction(monkeypatch, "RoleClient", Client)
    output = execute_correction_windows(
        stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
        max_retries_per_window=0, max_replacements_per_window=0, profile=PROFILE,
        task_artifact_dir=tmp_path / "art", length_scale=1.0)
    ledger = [json.loads(line) for path in tmp_path.rglob("correction-windows.jsonl")
              for line in path.read_text(encoding="utf-8").splitlines()]
    return output.read_text(encoding="utf-8"), calls, [row for row in ledger if "content" in row]


def test_a_correction_window_that_did_not_think_is_asked_once_more_in_a_fresh_session(tmp_path, monkeypatch):
    srt, calls, ledger = correction_run(tmp_path, monkeypatch, [("是啊", None), ("对呀", 900)])
    assert len(calls) == 2, "past a budget of one call: the retry is not one of its attempts"
    assert not calls[0].get("fresh_session") and calls[1]["fresh_session"] is True
    assert "对呀" in srt and "是啊" not in srt
    assert ledger[-1]["thinking"]["chosen"] == "retry"
    assert ledger[-1]["thinking"]["original"] == {"tokens": 0, "level": "medium"}


def test_the_first_answer_stays_when_the_retry_does_not_think_either_or_fails(tmp_path, monkeypatch):
    srt, calls, ledger = correction_run(tmp_path / "a", monkeypatch, [("是啊", None), ("对呀", None)])
    assert len(calls) == 2 and "是啊" in srt and ledger[-1]["thinking"]["chosen"] == "original"
    srt, calls, ledger = correction_run(tmp_path / "b", monkeypatch, [("是啊", None), RuntimeError("down")])
    assert "是啊" in srt and ledger[-1]["thinking"]["retry_error"] == "RuntimeError: down"
    srt, calls, ledger = correction_run(tmp_path / "c", monkeypatch, [("是啊", 700)])
    assert len(calls) == 1 and "thinking" not in ledger[-1]


def test_a_judge_answer_that_did_not_think_is_asked_once_more_and_a_leak_is_read_by_its_answer(tmp_path):
    from .test_speaker_judge_stage import FakeClient, judge_files, run

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
    assert windows[0]["thinking"]["chosen"] == "retry" and "thinking" not in windows[1]
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
    from .test_speaker_judge_stage import FakeClient, judge_files, run

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
    assert window["thinking"]["chosen"] == "original" and window["thinking"]["retry_error"].startswith("stopped")

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
    assert result.segments[0].translation == "译一" and entry["thinking"]["retry_error"].startswith("stopped")


def test_the_kept_original_is_credited_to_the_model_that_wrote_it(tmp_path):
    from .test_speaker_judge_stage import FakeClient, judge_files, run

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
    assert window["thinking"]["chosen"] == "original" and window["model"] == "first"
