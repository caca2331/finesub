"""The speaker judge's repair round: a reply whose `<judge>` block mostly answers is fixed with a `<patch>`,
on each transport -- an API conversation, a capsule agent, an agent tool session's `submit` -- and a reply
with nothing usable is asked again whole, as before (`stages/speaker_judge/patch.py`)."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from finesub.llm.agent import agent_validators
from finesub.llm.agent.agent_mcp_server import HarnessToolServer
from finesub.llm.agent.agent_task_runtime import AgentTaskRuntime
from finesub.llm.agent.agent_validators import build_validator, runtime_validators
from finesub.llm.agent.channels import ChannelLimits
from finesub.llm.agent.local_agent import DriverProbe
from finesub.llm.client import LLMCallResult, RoleClient
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import load_model_routes
from finesub.llm.stages.speaker_judge import answers, patch as judge_patch, run as judge_run
from finesub.speech.diarization import judgments as jd

from .test_speaker_judge_stage import judge_files, run

IDS = ["q1", "q2", "q3", "q4"]


def row(i: int, track: str = "s1", **fields) -> str:
    return json.dumps({"id": f"q{i}", "track": track, "why": "w", **fields}, ensure_ascii=False)


def whole(*rows: str, tracks: str = "s1 = 佐藤") -> str:
    return "<judge>\n" + "\n".join(rows) + "\n</judge>\n<tracks>\n" + tracks + "\n</tracks>"


def patch(*rows: str) -> str:
    return "<patch>\n" + "\n".join(rows) + "\n</patch>"


BROKEN = '{"id":"q2","track":"s1" "why":"w"}'
ALL = whole(*(row(i) for i in range(1, 5)))


# --- reading a reply line by line, and the patch -------------------------------------------------------


def test_the_survey_lists_every_reason_and_the_rows_that_answer():
    read = answers.survey(whole(row(1), BROKEN, row(3, "s9")), IDS, [0, 1])
    assert list(read.rows) == ["q1"]
    assert set(read.pending) == {"q2", "q3", "q4"} and read.missing == ("q4",)
    assert read.errors == ("1 row is not JSON: q2", "1 row's track is not one of the answers: q3 ('s9')",
                           "reply misses 1 of 4 rows: q4")
    repeated = answers.survey(whole(row(1), row(2), row(2), row(3), row(4)), IDS, [0, 1])
    assert set(repeated.rows) == {"q1", "q3", "q4"} and repeated.faulty == ("q2",), "a repeated row answers neither time"
    with pytest.raises(answers.AnswerError) as refused:
        answers.parse_judge(whole(row(1), BROKEN, row(3, "s9")), IDS, [0, 1])
    assert len(refused.value.errors) == 3, "the strict parser refuses with every reason, not the first"


def test_reading_the_rows_alone_checks_neither_their_tracks_nor_the_rows_left_out():
    """What the field parsers read (`answers._rows`), once `parse_judge` has read the reply."""
    read = answers.survey("<judge>\n" + row(2, "anything", join=True) + "\n</judge>", IDS, None)
    assert not read.errors and list(read.rows) == ["q2"]


def test_a_broken_row_and_a_good_one_for_the_same_id_are_both_void_and_the_broken_one_is_shown():
    read = answers.survey(whole(row(1), BROKEN, row(2), row(3), row(4)), IDS, [0, 1])
    assert "q2" not in read.rows and read.pending == ("q2",)
    assert "1 id is written more than once: q2" in read.errors, "the second row for q2 is a repeat"
    base = judge_patch.repair_base(whole(row(1), BROKEN, row(2), row(3), row(4)), IDS, [0, 1])
    assert base.pending == ("q2",) and judge_patch.rows_to_fix(base) == BROKEN
    later = whole(row(1), row(2), row(3), row(4), BROKEN)
    read = answers.survey(later, IDS, [0, 1])
    assert "q2" not in read.rows and read.pending == ("q2",), "the good row first: still both void, q2 asked once"
    assert read.errors == ("1 row is not JSON: q2", "1 id is written more than once: q2")
    assert judge_patch.rows_to_fix(judge_patch.repair_base(later, IDS, [0, 1])) == BROKEN, "the row the error names"


@pytest.mark.parametrize("reply", ["（想了很久）", "<judge>\n" + row(1), "<judge>\nnope\n</judge>",
                                   whole(row(1), row(2), row(3), row(4), row(9))])
def test_a_reply_a_patch_cannot_help_takes_no_patch(reply):
    assert judge_patch.repair_base(reply, IDS, [0, 1]) is None


def test_a_broken_row_whose_id_cannot_be_told_leaves_that_id_missing_for_the_patch():
    tracked = whole(row(1), BROKEN, '{"track": "s1" "why": "w"}', row(3))
    assert judge_patch.repair_base(tracked, IDS, [0, 1]).pending == ("q2", "q4")


def test_a_patch_replaces_the_rows_it_names_and_keeps_the_rest_in_window_order():
    base = judge_patch.repair_base(whole(row(1), BROKEN, row(3)), IDS, [0, 1])
    assert base.pending == ("q2", "q4") and judge_patch.rows_to_fix(base) == BROKEN
    merged = judge_patch.apply_patch(base, patch(row(4, "@甲"), row(2)) + "\n<tracks>\n@甲 = 不明\n</tracks>")
    assert merged.mode == "patch" and (merged.patch_units, merged.replaced_units) == (2, 0)
    assert answers.parse_judge(merged.text, IDS, [0, 1])["q4"].label == "甲"
    body = re.search("<judge>(.*)</judge>", merged.text, re.S).group(1)
    assert [answers.line_id(line) for line in body.splitlines() if line] == IDS
    assert answers.parse_tracks(merged.text) == {"s1": "佐藤", "@甲": "不明"}, "names are added line by line"


@pytest.mark.parametrize("reply, error", [
    ("no block", "exactly one"),
    (patch(row(2)) + "\n" + ALL, "both"),
    (patch(row(9)), "outside"),
    (patch("garbage"), "names no row id"),
    (patch(), "no rows"),
])
def test_a_patch_that_cannot_be_laid_says_why(reply, error):
    base = judge_patch.repair_base(whole(row(1), row(2), row(3)), IDS, [0, 1])
    merged = judge_patch.apply_patch(base, reply)
    assert merged.mode == "invalid" and error in " ".join(merged.errors)


def test_a_fault_many_rows_share_is_one_error_line_naming_them_and_each_row_is_still_its_own():
    ids = [f"q{i}" for i in range(1, 16)]
    broken = [f'{{"id":"q{i}","track":"s1" "why":"w"}}' for i in range(1, 14)]
    read = answers.survey(whole(*broken, "nope", "[1]", row(14, "s9"), row(15, "s8")), ids, [0, 1])
    assert read.errors == (
        "14 rows are not JSON: q1, q2, q3, q4, q5, q6, q7, q8, q9, q10 and 4 more",
        "1 row is not a JSON object: line 15",
        "2 rows' tracks are not among the answers: q14 ('s9'), q15 ('s8')",
    )
    assert read.faulty == tuple(ids) and read.lines["q13"] == broken[-1], "the patch still knows every row"
    base = judge_patch.repair_base(whole(row(1), *broken[1:]), ids, [0, 1])
    assert base.pending == tuple(ids[1:]) and len(base.faulty) == 12
    laid = judge_patch.apply_patch(base, patch("x", "y", row(30), row(31)))
    assert laid.errors == ("2 <patch> lines name no row id: line 1, line 2",
                           "2 <patch> lines name ids outside this window: 'q30', 'q31'")


def test_a_whole_answer_on_a_patch_round_is_taken_whole():
    base = judge_patch.repair_base(whole(row(1), row(2), row(3)), IDS, [0, 1])
    assert judge_patch.apply_patch(base, ALL).mode == "full"


def test_the_repair_turn_names_the_errors_the_rows_and_the_tracks_rule():
    base = judge_patch.repair_base(whole(row(1), row(2), row(3)), IDS, [0, 1])
    text = judge_patch.patch_instruction(base, ["reply misses 1 of 4 rows: q4"])
    assert "- reply misses 1 of 4 rows: q4" in text and "要重写的行：q4。" in text and "<tracks>" in text
    assert "$" not in text


def test_the_whole_answer_turn_asks_for_the_judges_own_format_not_a_csv():
    text = judge_patch.whole_instruction(["1 row is not JSON: q2"])
    assert "- 1 row is not JSON: q2" in text and "**完整**" in text
    assert "<judge>" in text and "<tracks>" in text and "不多不少" in text
    assert "$" not in text and "header" not in text and "列数" not in text and "<patch>" not in text
    assert "不只是点名的那几行" in text, "a list cut short still means every row of that kind"


# --- the validator a tool session's `submit` runs --------------------------------------------------------


def manifest(tracks=(0, 1)):
    return {"metadata": {"validator": {"ids": IDS, "tracks": list(tracks)}}}


def test_the_judge_validator_accepts_what_the_harness_reads_and_asks_a_patch_of_the_rest():
    validate = build_validator("speaker-judge", manifest())
    assert validate(ALL, {}).status == "accepted"
    leaked = validate("thought\nI write <judge>.\n" + ALL, {})
    assert leaked.status == "accepted" and leaked.artifact.startswith("thought"), "accepted as submitted"
    refused = validate(whole(row(1), BROKEN, row(3), row(4)), {})
    assert refused.status == "repairable" and "row is not JSON" in refused.errors[0]
    assert "要重写的行：q2。" in refused.metadata["repair_instruction"] and refused.metadata["rows_to_fix"] == BROKEN
    blind = validate("（想了很久）", {})
    assert blind.status == "repairable" and blind.metadata == {}, "nothing usable: the whole answer again"


def test_the_judge_merge_lays_a_patch_over_the_draft():
    merge = agent_validators.VALIDATOR_MERGES["speaker-judge"]
    laid = merge(patch(row(4)), whole(row(1), row(2), row(3)), manifest())
    assert laid.text is not None and not answers.survey(laid.text, IDS, [0, 1]).errors
    assert merge(ALL, "", manifest()).text is None and not merge(ALL, "", manifest()).errors, "not a patch"
    none = merge(patch(row(4)), "", manifest())
    assert none.errors and none.spends_repair is False, "no draft: submit the whole answer, at no cost"


# --- the harness's loop, whatever the transport ----------------------------------------------------------


class Script:
    """Answers from a script and keeps every call's keyword arguments."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, role, messages, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, LLMCallResult):
            return reply
        return LLMCallResult(content=reply, role=role, model="fake", fallback_used=False, raw_response={})


def entry_of(stable):
    return json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]


def test_missing_rows_are_asked_for_as_a_patch_and_the_merged_answer_is_kept(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    first = whole(row(1), row(2), row(3))
    client = Script([first, patch(row(4))])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert (report.judged, report.failed) == (1, 0) and len(client.calls) == 2
    assert client.calls[0]["previous_output"] == "" and client.calls[0]["repair_instruction"] == ""
    assert client.calls[1]["previous_output"] == first
    assert list(client.calls[1]["validation_errors"]) == ["reply misses 1 of 4 rows: q4"]
    assert "<patch>" in client.calls[1]["repair_instruction"] and "要重写的行：q4。" in client.calls[1]["repair_instruction"]
    assert {call["repair_session_key"] for call in client.calls} == {"speaker_judge-0001"}, "one chain per unit"
    assert client.calls[0]["validator_spec"] == {"id": "speaker-judge", "params": {"ids": IDS, "tracks": [0, 1]}}
    assert entry_of(stable)["repairs"] == 1
    assert jd.read_judgments(stable).moved == {"2": 1, "4": 1}, "the patched row counts like any other"


def test_a_row_that_is_not_json_is_patched(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    client = Script([whole(row(1), BROKEN, row(3), row(4)), patch(row(2))])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.judged == 1 and "row is not JSON" in client.calls[1]["validation_errors"][0]
    assert "要重写的行：q2。" in client.calls[1]["repair_instruction"]
    assert jd.read_judgments(stable).moved == {"2": 1, "4": 1}


def test_a_reply_with_no_judge_block_is_asked_again_whole(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    client = Script(["（想了很久）", ALL])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.judged == 1 and client.calls[1]["previous_output"] == "" and client.calls[1]["repair_instruction"] == ""
    assert "repairs" not in entry_of(stable)


def test_a_closed_block_with_nothing_to_keep_is_rewritten_whole_with_its_errors(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    first = "<judge>\nnope\n</judge>"
    client = Script([first, ALL])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.judged == 1 and client.calls[1]["previous_output"] == first
    assert list(client.calls[1]["validation_errors"]) == ["1 row is not JSON: line 1", "reply misses 4 of 4 rows: q1, q2, q3, q4"]
    instruction = client.calls[1]["repair_instruction"]
    assert "**完整**" in instruction and "<tracks>" in instruction and "- 1 row is not JSON: line 1" in instruction
    assert entry_of(stable)["repairs"] == 1


def test_a_whole_answer_round_answered_with_a_patch_is_asked_again_blind(tmp_path):
    """The whole-answer round has no base to lay a patch on: a `<patch>` there is a reply with no `<judge>`."""
    stable = judge_files(tmp_path, rows=4)
    client = Script(["<judge>\nnope\n</judge>", patch(*(row(i) for i in range(1, 5))), ALL])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.judged == 1 and len(client.calls) == 3
    assert "**完整**" in client.calls[1]["repair_instruction"]
    assert client.calls[2]["previous_output"] == "" and client.calls[2]["repair_instruction"] == "", "blind"
    assert client.calls[2]["max_repair_attempts"] == judge_run.MAX_ATTEMPTS - 3, "charged as any call"
    assert entry_of(stable)["repairs"] == 1, "the blind call is no repair"


def test_a_repair_still_failing_spends_the_budget_and_the_unit_fails_as_before(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    first = whole(row(1), row(2))
    client = Script([first, patch(row(3)), "no patch", patch(row(3))])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.failed == 1 and len(client.calls) == judge_run.MAX_ATTEMPTS
    second, third, fourth = client.calls[1:]
    assert second["previous_output"] == first and "要重写的行：q3、q4。" in second["repair_instruction"]
    assert "<patch>" not in third["previous_output"] and row(3) in third["previous_output"], "the merged whole is next"
    assert "要重写的行：q4。" in third["repair_instruction"]
    assert fourth["previous_output"] == third["previous_output"], "a patch that could not be laid: the same base"
    assert "exactly one" in fourth["validation_errors"][0]
    entry = entry_of(stable)
    assert entry["status"] == "failed" and entry["repairs"] == 3
    assert jd.read_judgments(stable).moved == {}


def test_a_tool_sessions_own_repairs_count_on_the_budget(tmp_path):
    """The session ran three repair rounds inside its one call and gave up: the unit's four calls are spent."""
    stable = judge_files(tmp_path, rows=4)
    exhausted = LLMCallResult(content=whole(row(1), row(2), row(3)), role=LLMRole.AUDIO_MULTIMODAL, model="m",
                              fallback_used=False, raw_response={}, agent_repair_rounds=3, repair_exhausted=True)
    client = Script([exhausted])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.failed == 1 and len(client.calls) == 1
    assert client.calls[0]["max_repair_attempts"] == judge_run.MAX_ATTEMPTS - 1


def test_a_patched_answer_is_not_asked_again_for_not_thinking(tmp_path):
    from .test_llm_degeneration import api_result

    stable = judge_files(tmp_path, rows=4)

    def reply(text, thoughts):
        return LLMCallResult(content=text, role=LLMRole.AUDIO_MULTIMODAL, model="m", fallback_used=False,
                             raw_response=api_result(thoughts).raw_response, thinking_level="medium")

    client = Script([reply(whole(row(1), row(2), row(3)), 500), reply(patch(row(4)), None)])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.judged == 1 and len(client.calls) == 2 and "thinking" not in entry_of(stable)


# --- the transports, through the real client --------------------------------------------------------------


def api_client(monkeypatch, replies):
    """A stateless API endpoint answering from a script; keeps the messages each call sent."""
    sent = []

    def chat_complete(messages, *, model, **kwargs):
        sent.append(messages)
        return {"choices": [{"message": {"content": replies.pop(0)}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", chat_complete)
    base = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False),
                        role_configs={LLMRole.AUDIO_MULTIMODAL: base.role_configs[LLMRole.GENERAL_CAPABLE]})
    return client, sent


def test_over_an_api_the_repair_is_two_more_turns_of_the_same_conversation(tmp_path, monkeypatch):
    stable = judge_files(tmp_path, rows=4)
    first = whole(row(1), BROKEN, row(3), row(4))
    client, sent = api_client(monkeypatch, [first, patch(row(2))])
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.judged == 1 and len(sent) == 2
    assert [m["role"] for m in sent[1]] == ["system", "user", "assistant", "user"]
    assert sent[1][:2] == sent[0] and sent[1][2]["content"] == first
    assert "<patch>" in sent[1][3]["content"] and "row is not JSON" in sent[1][3]["content"]


@pytest.mark.parametrize("first", ["", "<judge>\n" + row(1) + "\n" + row(2)], ids=["empty", "cut-off"])
def test_over_an_api_an_empty_or_cut_off_reply_is_the_same_prompt_again(tmp_path, monkeypatch, first):
    stable = judge_files(tmp_path, rows=4)
    client, sent = api_client(monkeypatch, [first, ALL])
    assert run(stable, client, tmp_path, cut=((1, 4),)).judged == 1 and sent[1] == sent[0]


def test_over_an_api_a_closed_block_with_nothing_to_keep_is_asked_whole_with_its_errors(tmp_path, monkeypatch):
    stable = judge_files(tmp_path, rows=4)
    first = "<judge>\nnope\n</judge>"
    client, sent = api_client(monkeypatch, [first, ALL])
    assert run(stable, client, tmp_path, cut=((1, 4),)).judged == 1
    assert [m["role"] for m in sent[1]] == ["system", "user", "assistant", "user"] and sent[1][2]["content"] == first
    turn = sent[1][3]["content"]
    assert "row is not JSON" in turn and "**完整**" in turn and "<patch>" not in turn
    assert "header" not in turn and "列数" not in turn and "<judge>" in turn, "the judge's own whole-answer turn"
    assert turn.count("row is not JSON") == 1, "the errors are listed once"


class CapsuleAgent:
    """A capsule agent that keeps its conversation (codex-like): one scripted output per spawn."""

    conversation_ttl_seconds = 0.0

    def __init__(self, outputs):
        self.config = SimpleNamespace(model="")
        self.outputs, self.calls = list(outputs), []

    def probe(self, *, refresh: bool = False):
        return DriverProbe(available=True, structured_events=True, no_persisted_session=True, no_user_config=True,
                           no_user_rules=True, can_restrict_tools=True, has_web_search=True, supports_session_reuse=True,
                           sandbox_kind="process_read_only")

    def meets_requirements(self, probe=None, *, native_search: bool = False):
        return True

    def run(self, messages, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=self.outputs.pop(0), reported_model="gpt-6-luna",
            execution_attempt={"backend": "local_agent", "capsule_id": f"cap-{len(self.calls)}"},
            episode_id=f"cap-{len(self.calls)}", conversation_handle=kwargs.get("conversation_handle") or "conv-1",
            turn_identity=f"sha256:turn-{len(self.calls)}", normalized_events=({"event": "turn.completed"},),
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})


def agent_client(driver, tmp_path, target):
    routes = load_model_routes(user_config={"model_groups": {"research-default": {"targets": [target]}}})
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    return RoleClient(router=ModelRouter(routes, policy_id=settings.policy_id), execution_settings=settings,
                      role_configs={LLMRole.AUDIO_MULTIMODAL: role_config_for("research", "quality", routes=routes)},
                      local_agent_driver=driver, rate_limiter=ModelRateLimiter(enabled=False),
                      agent_assignment_root=tmp_path / "assignments")


def no_api(monkeypatch):
    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")))


def test_a_capsule_agent_repairs_in_the_conversation_that_wrote_the_reply(tmp_path, monkeypatch):
    no_api(monkeypatch)
    stable = judge_files(tmp_path, rows=4)
    first = whole(row(1), row(2), row(3))
    driver = CapsuleAgent([first, patch(row(4))])
    report = run(stable, agent_client(driver, tmp_path, "local-codex-gpt-6-luna"), tmp_path, cut=((1, 4),))
    assert report.judged == 1 and len(driver.calls) == 2
    repair = driver.calls[1]
    assert repair["conversation_handle"] == "conv-1", "the repair resumes the conversation that wrote the reply"
    assert repair["previous_output"] == first and list(repair["validation_errors"]) == ["reply misses 1 of 4 rows: q4"]
    assert "<patch>" in repair["repair_instruction"] and "要重写的行：q4。" in repair["repair_instruction"]
    assert jd.read_judgments(stable).moved == {"2": 1, "4": 1}


class ToolAgent:
    """Plays an agent in a tool session: drives the harness server in-process, as the CLI would."""

    conversation_ttl_seconds = 0.0

    def __init__(self, script):
        self.config = SimpleNamespace(model="", mcp_opening="", channel=ChannelLimits())
        self.script, self.calls, self.seen = script, [], []

    def probe(self, *, refresh: bool = False):
        return DriverProbe(available=True, structured_events=True, no_persisted_session=True, no_user_config=True,
                           no_user_rules=True, can_restrict_tools=True, has_web_search=True, supports_session_reuse=True,
                           sandbox_kind="named_tool_allowlist", supports_mcp_config=True)

    def meets_requirements(self, probe=None, *, native_search: bool = False):
        return True

    def run(self, messages, **kwargs):
        self.calls.append(kwargs)
        env = kwargs["mcp_server"]["env"]
        runtime = AgentTaskRuntime(env["FINESUB_MCP_ROOT"], validators=runtime_validators("speaker-judge"))
        server = HarnessToolServer(runtime, assignment_id=env["FINESUB_MCP_ASSIGNMENT"],
                                   worker_id=env["FINESUB_MCP_WORKER"], session_id=env["FINESUB_MCP_SESSION"],
                                   instance_id="fake")

        def call(name, **arguments):
            reply = server.handle({"jsonrpc": "2.0", "id": len(self.seen) + 1, "method": "tools/call",
                                   "params": {"name": name, "arguments": arguments}})
            result = reply["result"]
            out = json.loads(result["content"][0]["text"]) if not result["isError"] else result
            self.seen.append((name, out))
            return out

        task = call("next_task")
        for block in task["required_blocks"]:
            call("read_context", ref=block["ref"])
        for payload in self.script:
            if call("submit", payload=payload)["status"] != "repairable":
                break
        return SimpleNamespace(content="submitted", reported_model="claude",
                               execution_attempt={"backend": "local_agent", "capsule_id": "cap-1"}, episode_id="cap-1",
                               conversation_handle="", turn_identity="", normalized_events=({"event": "result"},),
                               usage={"input_tokens": 7, "output_tokens": 1, "total_tokens": 8})


def submits(agent):
    return [out for name, out in agent.seen if name == "submit"]


def test_a_tool_session_gets_the_patch_instruction_at_submit_and_its_patch_is_merged(tmp_path, monkeypatch):
    no_api(monkeypatch)
    stable = judge_files(tmp_path, rows=4)
    agent = ToolAgent([whole(row(1), row(2), row(3)), patch(row(4))])
    report = run(stable, agent_client(agent, tmp_path, "local-claude-sonnet"), tmp_path, cut=((1, 4),))
    rejected, accepted = submits(agent)
    assert rejected["status"] == "repairable" and rejected["validation_errors"] == ["reply misses 1 of 4 rows: q4"]
    assert "要重写的行：q4。" in rejected["repair_instruction"] and accepted["status"] == "accepted"
    assert report.judged == 1 and len(agent.calls) == 1, "repaired inside the one call"
    assert jd.read_judgments(stable).moved == {"2": 1, "4": 1}


def test_a_tool_session_patches_a_row_that_is_not_json(tmp_path, monkeypatch):
    no_api(monkeypatch)
    stable = judge_files(tmp_path, rows=4)
    agent = ToolAgent([whole(row(1), BROKEN, row(3), row(4)), patch(row(2))])
    report = run(stable, agent_client(agent, tmp_path, "local-claude-sonnet"), tmp_path, cut=((1, 4),))
    assert report.judged == 1 and jd.read_judgments(stable).moved == {"2": 1, "4": 1}


def test_a_tool_session_with_nothing_usable_is_asked_for_the_whole_answer(tmp_path, monkeypatch):
    no_api(monkeypatch)
    stable = judge_files(tmp_path, rows=4)
    agent = ToolAgent(["（想了很久）", ALL])
    report = run(stable, agent_client(agent, tmp_path, "local-claude-sonnet"), tmp_path, cut=((1, 4),))
    rejected = submits(agent)[0]
    assert rejected["status"] == "repairable" and "repair_instruction" not in rejected and report.judged == 1


def test_a_tool_session_still_failing_spends_the_budget_and_the_unit_fails(tmp_path, monkeypatch):
    """The session is handed the unit's calls left after this one as its repairs; spent there, the unit fails."""
    no_api(monkeypatch)
    stable = judge_files(tmp_path, rows=4)
    agent = ToolAgent([whole(row(1), row(2), row(3)), patch(row(3, "s0")), patch(row(3, "keep")),
                       patch(row(3, "abstain")), patch(row(3, "@甲"))])
    report = run(stable, agent_client(agent, tmp_path, "local-claude-sonnet"), tmp_path, cut=((1, 4),))
    assert report.failed == 1 and len(agent.calls) == 1
    assert [(out["status"], out["repair_rounds_remaining"]) for out in submits(agent)] == [
        ("repairable", 3), ("repairable", 2), ("repairable", 1), ("repair_exhausted", 0)], (
        "the session's repairs are the unit's calls left after this one: as many answers as the API path")
    assert entry_of(stable)["status"] == "failed"


def test_a_patch_that_leaked_its_reasoning_is_read_by_the_patch_it_ends_with(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    leaked = "thought\nI will send <patch> with q4.\n" + patch(row(4))
    report = run(stable, Script([whole(row(1), row(2), row(3)), leaked]), tmp_path, cut=((1, 4),))
    assert report.judged == 1
    assert entry_of(stable)["thought_stripped"] == len(leaked) - len(patch(row(4))), "what was cut from the reply"
