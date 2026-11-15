"""Three passes' pass 3: proofreading the finished subtitles (`stages/proofread`): what each window is shown, how
its reply is read (the agent path's validator reads it the same way), and how the changes go back into the rows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from finesub.llm.agent.agent_validators import build_validator
from finesub.llm.client import LLMCallResult
from finesub.llm.output_protocol import TranslatedCsvSegment
from finesub.llm.prompts import ContextPack, WindowContextNote
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.proofread import answers, prompt
from finesub.llm.stages.proofread import run as proofread_run

PROFILE = resolve_profile("text", "none", "quality")


def test_a_reply_lists_the_changed_lines():
    text = "<proofread>\n2|1|そうだね|是啊|漏了语气\n\n3|5|はい|好|删掉「嗯」\n</proofread>"
    assert answers.parse_changes(text, 3) == [answers.Change(2, 1, "そうだね", "是啊", "漏了语气"),
                                              answers.Change(3, 5, "はい", "好", "删掉「嗯」")]
    assert answers.parse_changes("<proofread>\n</proofread>", 3) == []
    assert answers.parse_changes("<proofread>\n1|1|一行\\n两行|一行\\n两行|x\n</proofread>", 1)[0].translation == \
        "一行\n两行", "a line break comes back as the prompt wrote it"


@pytest.mark.parametrize("body, reason", [
    ("2|1|a|b", "不是五个字段"),
    ("2|2|A|B 原文里的竖线|改后的译文|统一译名", "不是五个字段"),
    ("2|1|a|b|理由|还有一处", "不是五个字段"),
    ("4|1|a|b|c", "序号不在 1–3 里"),
    ("１|1|a|b|c", "序号不在 1–3 里"),
    ("2|1|a|b|c\n2|3|a|b|c", "序号 2 写了两次"),
    ("2|7|a|b|c", "类别不是 1–6"),
    ("2|1|a||c", "纠错文或译文是空的"),
    ("2|1|a|\\n|c", "纠错文或译文是空的"),
    ("1|1|草稿|草稿|最终答案：<proofread>\n2|1|a|b|c", "又写了 <proofread>"),
])
def test_a_reply_that_does_not_read_is_refused_with_its_reason(body, reason):
    with pytest.raises(answers.AnswerError, match=reason):
        answers.parse_changes(f"<proofread>\n{body}\n</proofread>", 3)
    with pytest.raises(answers.AnswerError, match="没有闭合的"):
        answers.parse_changes("<proofread>\n1|1|a|b|c", 3)


def test_the_agent_validator_refuses_what_the_harness_refuses():
    validate = build_validator("proofread-window", {"metadata": {"validator": {"count": 2}}})
    assert validate("<proofread>\n2|1|a|b|c\n</proofread>", {}).status == "accepted"
    refused = validate("<proofread>\n3|1|a|b|c\n</proofread>", {})
    assert refused.status != "accepted" and "序号不在 1–2 里" in " ".join(refused.errors)
    # leaked reasoning: judged by its answer, handed back as submitted for the harness to strip and record
    leaked = "thought\nThe <proofread> block.\n<proofread>\n2|1|a|b|c\n</proofread>"
    assert validate(leaked, {}).artifact == leaked


def windows(*spans):
    return [SimpleNamespace(chunk_id=f"{n:04d}", source_ids=[str(i) for i in span]) for n, span in enumerate(spans, 1)]


def test_lines_belong_to_the_first_window_holding_their_first_source_and_see_two_neighbours_a_side():
    cut = prompt.windows_of([("1",), ("2", "3"), ("4",), (), ("5",), ("6",)],
                            windows([1, 2, 3, 4], [4, 5, 6], [7]))
    assert [(w.chunk_id, w.rows, w.before, w.after) for w in cut] == [
        ("0001", (0, 1, 2, 3), (), (4, 5)),  # line 4 (the overlap) stays with the earlier window; the insert follows it
        ("0002", (4, 5), (2, 3), ()),
    ], "a window left without lines is left out"


def test_a_window_carries_its_lines_source_ids():
    cut = prompt.windows_of([("1",), ("2", "3"), ("4",)], windows([1, 2], [3, 4]))
    assert [w.source_ids for w in cut] == [("1", "2", "3"), ("4",)]


def test_a_window_reads_the_chunks_of_its_own_general_context_own_first():
    pack = ContextPack.from_dict({
        "general_context": {"c01": {"global_summary": "前半", "entities": [{"name": "甲", "zh": "甲"}]},
                            "c02": {"global_summary": "后半", "entities": [{"name": "乙", "zh": "乙"}]}},
        "general_chunks": [{"chunk_id": "c01", "first_source_id": "1", "last_source_id": "2"},
                           {"chunk_id": "c02", "first_source_id": "3", "last_source_id": "4"}],
    }).with_source_order(["1", "2", "3", "4"])
    lines = [line(corrected=f"行{n}") for n in range(1, 5)]
    _system, user = prompt.proofread_prompt(prompt.Window("0002", (2, 3), source_ids=("3", "4")), lines, title="clip",
                                            context_pack=pack, style=prompt.style_text("", ""))
    assert user.index("后半") < user.index("前半") and user.index("- 乙") < user.index("- 甲")


def line(seconds=2.0, asr="そうだね", corrected="そうだね。", translation="是啊。", conf=None, note=""):
    return prompt.Line(seconds, asr, corrected, translation, conf, note)


def test_a_line_shows_its_reading_speed_and_its_asr_text_only_where_it_differs():
    assert prompt.line_cells(1, line()) == "1|2.0|1.2|=|そうだね。|是啊。||"  # 2.5 weighted chars in 2 s
    assert prompt.line_cells(2, line(asr="そうだ", conf="low", note="a|b", seconds=0.0)) == \
        "2|0.0||そうだ|そうだね。|是啊。|low|a｜b"
    assert prompt.line_cells(3, line(conf="high")).endswith("|是啊。||")
    assert prompt.line_cells(4, line(translation="一行\n两行")).split("|")[5] == "一行\\n两行"


def test_the_prompt_adds_the_used_terms_the_neighbours_and_the_style():
    lines = [line(corrected=f"行{n}", translation=f"译{n}") for n in range(1, 6)]
    window = prompt.Window("0002", (2, 3), before=(0, 1), after=(4,))
    context = ContextPack(general_context={"global_summary": "雑談", "entities": [{"name": "佐藤", "zh": "佐藤"}]})
    system, user = prompt.proofread_prompt(window, lines, title="clip", context_pack=context,
                                           style=prompt.style_text("", ""), kb_terms="# 佐藤\nさとう|佐藤|")
    assert "`<proofread>`" in system and "$" not in system + user
    # a task always has a style (the factory default without a user one), so rule 6 always applies
    assert "6. 与翻译风格明显不符的地方" in system and "（1–6）" in system
    assert "知识库里的专名与术语" in user and "さとう|佐藤|" in user
    assert "这一段之前的几行" in user and "行1|译1\n行2|译2" in user and "这一段之后的几行" in user and "行5|译5" in user
    assert "1|2.0|0.8|そうだね|行3|译3||\n2|2.0|0.8|そうだね|行4|译4||" in user and "共 2 行" in user
    assert "- 佐藤 → 佐藤（）：" in user
    styled, bare = prompt.proofread_prompt(prompt.Window("0001", (0,)), lines, title="clip", context_pack=None,
                                           style=prompt.style_text("风格：口语", "人名不译"))
    assert "写得非常明确的，按风格来；说得不那么肯定的，两边折中" in styled  # owner's D9
    assert "出厂默认风格与长度、读速冲突时，按长度、读速来" in styled  # O2, owner 2026-11-03
    assert "资料里定好的专名与术语" in styled, "owner's D7"
    assert "知识库里的专名与术语" not in bare and "这一段之前" not in bare and "这一段之后" not in bare
    # the run's own words outrank the styles below them (O3, owner 2026-11-03)
    assert "特殊翻译风格要求（本次运行另写的；与下面的风格约定冲突时，按这里的来）：\n人名不译\n\n风格：口语" in bare
    assert "人物：\n（无）" in bare
    assert "专名与术语：\n（无）\n\n" in bare


def test_the_users_relayed_requests_are_shown_where_the_task_has_some():
    lines = [line(corrected="行1", translation="译1")]
    window = prompt.Window("0001", (0,))
    asked = ContextPack(general_context={"global_summary": "s"}, user_requests=("歌名用《》括起来",))
    _system, user = prompt.proofread_prompt(window, lines, title="clip", context_pack=asked, style=prompt.style_text("", ""))
    assert "用户的要求" in user and "- 歌名用《》括起来" in user and "不要改回去" in user
    assert user.index("用户的要求") < user.index("翻译风格：") and "特殊翻译风格要求" not in user
    _system, extra = prompt.proofread_prompt(window, lines, title="clip", context_pack=asked,
                                             style=prompt.style_text("", "口语化"))
    assert "与下面翻译风格里的「特殊翻译风格要求」冲突时，按后者" in extra
    for pack in (None, ContextPack(general_context={"global_summary": "s"})):
        _system, plain = prompt.proofread_prompt(window, lines, title="clip", context_pack=pack,
                                                 style=prompt.style_text("", "口语化"))
        assert "用户的要求" not in plain


def test_relayed_requests_join_a_windows_input_digest_only_when_there_are_some():
    lines = [line(corrected="行1", translation="译1")]
    window = prompt.Window("0001", (0,))
    digest = lambda pack: proofread_run._input_digest(window, lines, proofread_run.ProofreadInputs("t", pack, "s"))  # noqa: E731
    general = {"global_summary": "s"}
    assert digest(ContextPack(general_context=general)) != digest(ContextPack(general_context=general,
                                                                                user_requests=("要求",)))


def test_a_window_reads_its_own_background_and_the_requests_marked_in_it():
    """Pass 3 sees the background the correction saw for the window (picked by its source ids), and a request
    written there as the user's gets the requests block even when the task has none for every window."""
    lines = [line(corrected=f"行{n}", translation=f"译{n}") for n in range(1, 5)]
    note = WindowContextNote("0002", "3", "4", "这段在唱《子守唄》。出自用户：歌词逐句放进字幕，原文照录、中文另译。")
    pack = ContextPack(general_context={"global_summary": "s"},
                       window_contexts=(note,)).with_source_order(["1", "2", "3", "4"])
    ask = lambda window: prompt.proofread_prompt(window, lines, title="clip", context_pack=pack,  # noqa: E731
                                                 style=prompt.style_text("", ""))[1]
    own = ask(prompt.Window("0002", (2, 3), source_ids=("3", "4")))
    assert "这一段的背景" in own and "这段在唱《子守唄》" in own
    assert own.index("这一段的背景") < own.index("资料里没能确定的事") < own.index("只关这一段的：上面")
    assert "全片适用" not in own and "不按第 2 条改短" in own
    other = ask(prompt.Window("0001", (0, 1), source_ids=("1", "2")))
    assert "这一段的背景" not in other and "用户的要求" not in other and "出自用户" not in other
    both = ContextPack(general_context={"global_summary": "s"}, window_contexts=(note,),
                       user_requests=("歌名用《》括起来",)).with_source_order(["1", "2", "3", "4"])
    _system, user = prompt.proofread_prompt(prompt.Window("0002", (2, 3), source_ids=("3", "4")), lines, title="clip",
                                            context_pack=both, style=prompt.style_text("", ""))
    assert "转述的）：\n全片适用的：\n- 歌名用《》括起来\n只关这一段的：上面「这一段的背景」" in user
    digest = lambda pack, ids: proofread_run._input_digest(  # noqa: E731
        prompt.Window("w", (0,), source_ids=ids), lines, proofread_run.ProofreadInputs("t", pack, "s"))
    bare = ContextPack(general_context={"global_summary": "s"}).with_source_order(["1", "2", "3", "4"])
    assert digest(pack, ("3", "4")) != digest(bare, ("3", "4")), "the window's background is part of its input"
    assert digest(pack, ("1", "2")) == digest(bare, ("1", "2")), "a window without one digests as before"


def test_the_knowledge_terms_are_the_names_without_their_descriptions():
    assert prompt.entry_terms({"k": "# 佐藤\n简介一行\n## 人物\nさとう|佐藤|サトウ|主播本人"}) == \
        "## 人物\nさとう|佐藤|サトウ"


class FakeClient:
    """Answers each window by a function of its user prompt; `fail` raises instead."""

    def __init__(self, answer=None, fail=None):
        self.answer = answer or (lambda user, messages: "<proofread>\n</proofread>")
        self.fail = fail
        self.calls = []

    def complete(self, role, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.fail is not None:
            raise self.fail
        return LLMCallResult(content=self.answer(messages[1]["content"], messages), role=role, model="fake",
                             fallback_used=False, raw_response={})


def rows(count):
    return [TranslatedCsvSegment((str(n),), float(n), n + 0.8, f"行{n}", f"译{n}", conf="high",
                                 note=f"注{n}") for n in range(1, count + 1)]


def proofread(tmp_path, client, segments, cut, **options):
    lines = proofread_run.lines_of(segments, [1.0] * len(segments), {str(n): f"行{n}" for n in range(1, 9)})
    return proofread_run.run_proofread(
        segments, lines, prompt.windows_of([s.source_ids for s in segments], cut),
        inputs=proofread_run.ProofreadInputs("clip", None, prompt.style_text("", "")), client=client,
        profile=PROFILE, artifact_dir=tmp_path, task_id="t", warn=lambda message: None, **options)


def test_changes_rewrite_the_text_and_keep_the_rest_of_the_row(tmp_path):
    def answer(user, messages):
        if "|行1|" in user:
            return "<proofread>\n1|1|行一|第一行，改长了|漏译\n2|5|行2|译2|没改\n</proofread>"
        return "<proofread>\n1|3|行三|译3|专名\n</proofread>"

    segments = rows(4)
    result = proofread(tmp_path, FakeClient(answer), segments, windows([1, 2], [3, 4]))
    assert result.status == "done" and result.failed == []
    assert result.segments[0] == TranslatedCsvSegment(("1",), 1.0, 1.8, "行一", "第一行，改长了", conf="high",
                                                      note="注1")
    assert result.segments[1] == segments[1] and result.segments[3] == segments[3]
    assert result.changes == [
        {"row": 1, "window": "0001", "source_ids": ["1"], "category": 1, "why": "漏译",
         "corrected": ["行1", "行一"], "translation": ["译1", "第一行，改长了"]},
        {"row": 3, "window": "0002", "source_ids": ["3"], "category": 3, "why": "专名", "corrected": ["行3", "行三"]}]
    assert proofread_run.proofread_status(tmp_path) == "done"
    # what tools/session_replay's `proofread` session rebuilds a window's prompt from
    dumped = json.loads((tmp_path / proofread_run.INPUT_NAME).read_text(encoding="utf-8"))
    assert [w["chunk_id"] for w in dumped["windows"]] == ["0001", "0002"] and dumped["lines"][0]["corrected"] == "行1"
    assert dumped["profile_id"] == PROFILE.profile_id and dumped["title"] == "clip"


def test_a_reply_that_does_not_read_is_asked_again_with_the_reason(tmp_path):
    def answer(user, messages):
        return "<proofread>\n9|1|a|b|c\n</proofread>" if len(messages) == 2 else "<proofread>\n</proofread>"

    client = FakeClient(answer)
    result = proofread(tmp_path, client, rows(2), windows([1, 2]))
    assert result.status == "done" and len(client.calls) == 2
    retry = client.calls[1]["messages"]
    assert retry[2] == {"role": "assistant", "content": "<proofread>\n9|1|a|b|c\n</proofread>"}
    assert "上面的回答不合格：序号不在 1–2 里" in retry[3]["content"]
    assert client.calls[0]["validator_spec"] == {"id": "proofread-window", "params": {"count": 2}}


def test_an_agent_sessions_repair_rounds_come_out_of_the_windows_budget(tmp_path):
    from dataclasses import replace as with_fields

    class AgentClient(FakeClient):
        def complete(self, role, messages, **kwargs):
            # the session already ran every repair round it was allowed, and still did not read
            return with_fields(super().complete(role, messages, **kwargs),
                               agent_repair_rounds=kwargs["max_repair_attempts"], repair_exhausted=True)

    client = AgentClient(lambda user, messages: "no block")
    result = proofread(tmp_path, client, rows(2), windows([1, 2]), retries=2)
    assert result.failed == ["0001"] and len(client.calls) == 1, "no second session on a spent budget"


def test_a_window_that_never_answers_keeps_pass_2_and_a_rerun_asks_only_it(tmp_path):
    def answer(user, messages):
        return "<proofread>\n1|1|行一|译一|x\n</proofread>" if "|行1|" in user else "no block"

    client = FakeClient(answer)
    first = proofread(tmp_path, client, rows(4), windows([1, 2], [3, 4]), retries=2)
    assert first.status == "failed" and first.failed == ["0002"]
    assert len(client.calls) == 1 + 3, "a window that does not read is asked again up to `retries` times"
    assert [call["max_repair_attempts"] for call in client.calls[1:]] == [2, 1, 0], "one budget, re-asks and repairs"
    assert first.segments[0].translation == "译一" and first.segments[2:] == rows(4)[2:]
    assert proofread_run.proofread_status(tmp_path) == "failed"
    again = FakeClient(lambda user, messages: "<proofread>\n</proofread>")
    second = proofread(tmp_path, again, rows(4), windows([1, 2], [3, 4]))
    assert second.status == "done" and second.reused == 1 and len(again.calls) == 1
    assert second.segments[0].translation == "译一", "the kept answer is applied again"
    changed = rows(4)
    changed[2] = TranslatedCsvSegment(("3",), 3.0, 3.8, "行3", "别的译文", conf="high")
    third = FakeClient(lambda user, messages: "<proofread>\n</proofread>")
    proofread(tmp_path, third, changed, windows([1, 2], [3, 4]))
    assert len(third.calls) == 2, "a neighbour is part of what a window is shown"
    fourth = FakeClient(lambda user, messages: "<proofread>\n</proofread>")
    proofread(tmp_path, fourth, rows(6), windows([1, 2, 3], [4, 5, 6]))
    changed = rows(6)
    changed[5] = TranslatedCsvSegment(("6",), 6.0, 6.8, "行6", "别的译文", conf="high")
    fifth = FakeClient(lambda user, messages: "<proofread>\n</proofread>")
    proofread(tmp_path, fifth, changed, windows([1, 2, 3], [4, 5, 6]))
    assert len(fifth.calls) == 1, "a window whose lines changed is asked again; the other is kept"


def test_an_answer_is_kept_when_only_the_prompts_wording_changed(tmp_path, monkeypatch):
    """The answers are kept under what a window was given, not the rendered prompt: a template or version
    change alone re-asks nothing, and the kept answers from another version are counted for the warning."""
    proofread(tmp_path, FakeClient(), rows(6), windows([1, 2, 3], [4, 5, 6]))
    rendered = proofread_run.proofread_prompt
    monkeypatch.setattr(proofread_run, "proofread_prompt",
                        lambda *args, **kwargs: tuple(part + "\n改过的措辞" for part in rendered(*args, **kwargs)))
    monkeypatch.setattr(proofread_run, "PROMPT_VERSION", "proofread-next")
    again = FakeClient()
    second = proofread(tmp_path, again, rows(6), windows([1, 2, 3], [4, 5, 6]))
    assert second.reused == 2 and again.calls == []
    first_version = json.loads(proofread_run.sidecar_path(tmp_path).read_text(encoding="utf-8"))["windows"]["0001"]
    assert second.older_prompt == {first_version["prompt_version"]: 2}
    changed = rows(6)
    changed[5] = TranslatedCsvSegment(("6",), 6.0, 6.8, "行6", "别的译文", conf="high")
    third = FakeClient()
    result = proofread(tmp_path, third, changed, windows([1, 2, 3], [4, 5, 6]))
    assert len(third.calls) == 1 and "别的译文" in third.calls[0]["messages"][1]["content"], "only its window"
    assert result.older_prompt == {first_version["prompt_version"]: 1}


def test_kept_answers_from_another_prompt_version_are_reported_once(tmp_path, reported):
    from finesub.llm.stages.correction.run import _warn_older_prompt_answers

    _warn_older_prompt_answers({}, tmp_path)
    assert reported.warnings == []
    _warn_older_prompt_answers({"proofread-v6": 2}, tmp_path)
    assert reported.codes() == ["proofread-older-prompt-windows"]
    assert "proofread-v6 2 窗" in reported.warnings[0].message
    assert str(proofread_run.sidecar_path(tmp_path)) in reported.warnings[0].action


def test_without_resume_every_window_is_asked_again(tmp_path):
    proofread(tmp_path, FakeClient(), rows(4), windows([1, 2], [3, 4]))
    again = FakeClient()
    assert proofread(tmp_path, again, rows(4), windows([1, 2], [3, 4]), resume=False).reused == 0
    assert len(again.calls) == 2, "--no-resume does not take the answers an earlier run kept"


def test_a_quota_error_stops_the_pass(tmp_path):
    client = FakeClient(fail=RuntimeError("429 RESOURCE_EXHAUSTED quota"))
    result = proofread(tmp_path, client, rows(4), windows([1, 2], [3, 4]))
    assert result.status == "failed" and result.stopped and result.segments == rows(4)
    assert len(client.calls) == 1


def test_the_lines_need_one_cue_each():
    with pytest.raises(ValueError, match="2 rows but 1 cues"):
        proofread_run.lines_of(rows(2), [1.0], {})
    assert proofread_run.lines_of(rows(1), [1.5], {"1": "あ"}) == [prompt.Line(1.5, "あ", "行1", "译1", "high", "注1")]


def test_the_status_of_a_task_that_never_proofread_is_none(tmp_path):
    assert proofread_run.proofread_status(tmp_path) is None
    proofread_run.sidecar_path(tmp_path).write_text(json.dumps({"version": 1, "status": "pending"}), encoding="utf-8")
    assert proofread_run.proofread_status(tmp_path) is None


# --- in the correction run: pass 2's products, then pass 3's -------------------------------------------


def correct_and_proofread(tmp_path, monkeypatch, proofread_reply, *, proofread=True, artifacts="art"):
    from finesub.llm.routing.config import LLMRole
    from finesub.llm.stages.correction import execute_correction_windows

    from .conftest import setattr_correction
    from .test_llm_speaker_turns import FakeTokenCounter, labelled_stable, reply

    stable = labelled_stable(tmp_path, [None, None], texts=["そうだね", "でもさ"])
    seen: list[list[dict]] = []

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            if messages[0]["content"].startswith("你在校对"):
                seen.append(messages)
                content = proofread_reply(messages)
            else:
                content = reply("sub|1|そうだね|是啊|high|", "sub|2|でもさ|不过|high|")
            return LLMCallResult(content=content, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", Client)
    result = execute_correction_windows(
        stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
        max_retries_per_window=1, max_replacements_per_window=0, profile=PROFILE,
        task_artifact_dir=tmp_path / artifacts if artifacts else None, proofread=proofread, proofread_title="clip", length_scale=1.0)
    folder = tmp_path / (artifacts or "out.llm-artifacts")
    records = [json.loads(line) for line in (folder / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    return result, seen, records


def test_pass_2_keeps_its_products_and_the_final_ones_carry_pass_3s_changes(tmp_path, monkeypatch):
    result, seen, artifacts = correct_and_proofread(
        tmp_path, monkeypatch, lambda messages: "<proofread>\n2|1|でもさ|可是啊|语气\n</proofread>")
    assert result == tmp_path / "out.srt"
    assert "不过" in (tmp_path / "out-pass2.srt").read_text(encoding="utf-8")
    assert "不过" in (tmp_path / "out-pass2-translated.srt").read_text(encoding="utf-8")
    assert (tmp_path / "out-pass2-corrected.srt").exists()
    assert "|不过|" in (tmp_path / "out-pass2-annotated.csv").read_text(encoding="utf-8")
    assert "可是啊" in result.read_text(encoding="utf-8") and "不过" not in result.read_text(encoding="utf-8")
    assert "|可是啊|" in (tmp_path / "out-annotated.csv").read_text(encoding="utf-8")
    assert "可是啊" in (tmp_path / "out-translated.srt").read_text(encoding="utf-8")
    rows = [line for line in seen[0][1]["content"].splitlines() if line.startswith(("1|", "2|"))]
    assert rows[0].startswith("1|") and rows[0].endswith("|そうだね|是啊|high|") is False
    assert rows[1].split("|")[3:6] == ["=", "でもさ", "不过"]
    record = next(item["payload"] for item in artifacts if item["kind"] == "proofread")
    assert record["status"] == "done" and record["changes"][0]["translation"] == ["不过", "可是啊"]
    final = next(item["payload"] for item in artifacts if item["kind"] == "final_srt")
    assert final["pass2_path"].endswith("out-pass2.srt")


def test_a_pass_3_that_cannot_answer_leaves_pass_2_as_the_final_products(tmp_path, monkeypatch):
    result, seen, artifacts = correct_and_proofread(tmp_path, monkeypatch, lambda messages: "no block")
    assert len(seen) == 2 and "不过" in result.read_text(encoding="utf-8")
    assert (tmp_path / "out-annotated.csv").read_text(encoding="utf-8") == \
        (tmp_path / "out-pass2-annotated.csv").read_text(encoding="utf-8")
    assert next(item["payload"] for item in artifacts if item["kind"] == "proofread")["status"] == "failed"


def test_without_pass_3_there_are_no_pass_2_products(tmp_path, monkeypatch):
    _, seen, artifacts = correct_and_proofread(tmp_path, monkeypatch, lambda messages: "", proofread=False)
    assert not seen and not list(tmp_path.glob("out-pass2*"))
    assert not any(item["kind"] == "proofread" for item in artifacts)


def test_a_pass_3_that_cannot_run_does_not_leave_an_earlier_done_standing(tmp_path, monkeypatch):
    correct_and_proofread(tmp_path, monkeypatch, lambda messages: "<proofread>\n</proofread>")
    assert proofread_run.proofread_status(tmp_path / "art") == "done"

    def broken(*args, **kwargs):
        raise ValueError("2 rows but 1 cues")

    monkeypatch.setattr(proofread_run, "lines_of", broken)
    result, _, artifacts = correct_and_proofread(tmp_path, monkeypatch, lambda messages: "<proofread>\n</proofread>")
    assert proofread_run.proofread_status(tmp_path / "art") == "failed"
    assert [item["payload"]["status"] for item in artifacts if item["kind"] == "proofread"][-1] == "failed"
    assert (tmp_path / "out-annotated.csv").read_text(encoding="utf-8") == \
        (tmp_path / "out-pass2-annotated.csv").read_text(encoding="utf-8")


def test_without_a_task_directory_pass_3_writes_beside_the_window_clips_not_the_subtitles(tmp_path, monkeypatch):
    correct_and_proofread(tmp_path, monkeypatch, lambda messages: "<proofread>\n</proofread>", artifacts=None)
    assert (tmp_path / "out.llm-artifacts" / "proofread.json").exists()
    assert not (tmp_path / "proofread.json").exists() and not (tmp_path / "task-artifacts.jsonl").exists()


def test_each_window_is_told_the_terms_its_own_correction_used_at_that_knowledge_version(monkeypatch):
    from finesub.llm.stages.correction import run as correction_run

    loaded = []

    def load(root, keys, rev=None):
        loaded.append((tuple(keys), rev))
        return {key: f"## 人物\n{key}|{key}定名|别名|描述" for key in keys}, []

    monkeypatch.setattr(correction_run, "load_entry_texts", load)
    ledger = SimpleNamespace(
        records={"0003-a": {"injected_entries": ["甲"], "knowledge_version": "rev:5"},  # replayed from an earlier run
                 "0004": {"injected_entries": ["旧"], "knowledge_version": "rev:1"}},
        committed={"0003-b": {"injected_entries": ["甲", "乙"], "knowledge_version": "rev:7"},
                   "0004-a": {"injected_entries": ["丙"], "knowledge_version": ""}})
    run = SimpleNamespace(knowledge_enabled=True, knowledge_root="kb", ledger=ledger)
    terms = correction_run._window_terms(run, ["0003", "0004-a"])
    assert terms == {"0003": "## 人物\n甲|甲定名|别名\n## 人物\n乙|乙定名|别名", "0004-a": "## 人物\n丙|丙定名|别名"}
    assert loaded == [(("甲",), 5), (("乙",), 7), (("丙",), None)], "halves together, each at its own version"
    assert correction_run._window_terms(SimpleNamespace(knowledge_enabled=False), ["0003"]) == {}

    def unreadable(root, keys, rev=None):
        raise OSError("locked")

    monkeypatch.setattr(correction_run, "load_entry_texts", unreadable)
    assert correction_run._window_terms(run, ["0003"]) == {}


def test_each_call_leaves_what_the_token_books_read(tmp_path):
    """An answered call carries its route and attempts; a failed one writes them in `proofread_call_error`."""

    from finesub import jsonl
    from finesub.llm.knowledge.base import TASK_ARTIFACT_FILENAME

    class Traced(FakeClient):
        def complete(self, role, messages, **kwargs):
            result = super().complete(role, messages, **kwargs)
            return LLMCallResult(content=result.content, role=role, model="fake", fallback_used=False,
                                 raw_response={}, execution_attempts=[{"backend": "local_agent"}],
                                 route_decision={"candidates": []})

    proofread(tmp_path / "ok", Traced(), rows(2), windows([1, 2]))
    error = RuntimeError("exit 1")
    error._harness_execution_attempts = [{"backend": "local_agent", "usage": {"input_tokens": 9}}]
    proofread(tmp_path / "bad", FakeClient(fail=error), rows(2), windows([1, 2]))

    answered = [r["payload"] for r in jsonl.iter_records(tmp_path / "ok" / TASK_ARTIFACT_FILENAME)
                if r["kind"] == "proofread_response"]
    assert answered[0]["execution_attempts"] == [{"backend": "local_agent"}]
    assert answered[0]["route_decision"] == {"candidates": []}
    failed = [r["payload"] for r in jsonl.iter_records(tmp_path / "bad" / TASK_ARTIFACT_FILENAME)
              if r["kind"] == "proofread_call_error"]
    assert failed and failed[0]["execution_attempts"] == error._harness_execution_attempts
