"""The inline proofread turn wired into the runs (`--llm-proofread-inline`, 乙 of the intermediate one-pass plan):
the rule both places follow (`inline_proofread.next_step`), the tool-session runtime's owed proofread, and the
per-window loop on the API and capsule shape (`stages.correction.attempts._proofread`)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from finesub.llm import inline_proofread
from finesub.llm.agent.agent_task_runtime import PROOFREAD_PARAM, AgentTaskRuntime, AgentTaskSpec
from finesub.llm.agent.agent_validators import runtime_validators, serialize_window
from finesub.llm.chunking import CorrectionBudget, SubtitleSegment, SubtitleWindow
from finesub.llm.client import LLMCallResult, _proofread_left
from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.routing.config import LLMRole
from finesub.llm.routing.profiles import resolve_profile

from .conftest import setattr_correction

CAPABLE_C = resolve_variant("capableC")
HEADER = "type|position|corrected_text|translation|conf|note"


def _sub(position: str, translation: str | None = None) -> str:
    return f"sub|{position}|{position}。|{translation or position}|high|"


def _whole(*rows: str, advice: str = "") -> str:
    return f"<translated>\n{HEADER}\n" + "\n".join(rows) + "\n</translated>" + (
        f"\n<next_advice>\n{advice}\n</next_advice>" if advice else "")


def _patch(*rows: str) -> str:
    return f"<patch>\n{HEADER}\n" + "\n".join(rows) + "\n</patch>"


DRAFT = _whole(*(_sub(str(i)) for i in range(1, 5)))


def _window() -> SubtitleWindow:
    return SubtitleWindow(
        chunk_id="0001", segments=[SubtitleSegment(str(i), i * 1.5, i * 1.5 + 1.0, f"{i}。") for i in range(1, 5)],
        overlap_segments=[], boundary_reason="end", budget=CorrectionBudget(0, 0, 0, ""), length_scale=1.0)


def _validate(window: SubtitleWindow):
    return lambda text: validate_correction_window_output(text, window, variant=CAPABLE_C)


# --- the rule ------------------------------------------------------------------------------------------------


def test_a_reply_that_cannot_be_laid_over_gets_one_repair_then_the_draft_stands() -> None:
    """Owner Q16: one repair round for a reply that cannot be laid over -- a whole that fails validation
    included -- and the draft when the repair fails too."""

    window, bad = _window(), _patch(_sub("3", "三|三"))
    first, repair = inline_proofread.next_step(DRAFT, bad, window, CAPABLE_C, _validate(window), round_=0)
    assert first.status == inline_proofread.FALLBACK and repair and all(error in repair for error in first.errors)
    second, again = inline_proofread.next_step(DRAFT, bad, window, CAPABLE_C, _validate(window), round_=1)
    assert second.status == inline_proofread.FALLBACK and second.text == DRAFT and again == ""


def test_only_the_patch_is_taken_from_a_reply() -> None:
    """A block the reply writes besides the patch does not replace the draft's: the window's advice and transfer
    chain come from the answer it was asked about."""

    window, draft = _window(), _whole(*(_sub(str(i)) for i in range(1, 5)), advice="草稿的建议")
    outcome = inline_proofread.apply(draft, _patch(_sub("3", "改")) + "\n<next_advice>\n校对的建议\n</next_advice>",
                                     window, CAPABLE_C, _validate(window))
    assert outcome.status == inline_proofread.APPLIED
    assert "草稿的建议" in outcome.text and "校对的建议" not in outcome.text and "sub|3|3。|改|high|" in outcome.text


def test_the_draft_sent_back_whole_is_nothing_to_change() -> None:
    window = _window()
    outcome = inline_proofread.apply(DRAFT, DRAFT, window, CAPABLE_C, _validate(window))
    assert outcome.status == inline_proofread.NO_CHANGE and outcome.text == DRAFT


def test_the_names_of_a_context_prepared_in_chunks_are_all_read() -> None:
    chunked = {"c1": {"entities": [{"name": "タルタリヤ", "zh": "达达利亚"}]},
               "c2": {"terminology": [{"source": "元素爆発", "zh": "元素爆发"}]}}
    assert inline_proofread.names_table(chunked) == [(["タルタリヤ"], ["达达利亚"]), (["元素爆発"], ["元素爆发"])]


def test_the_instruction_names_where_the_task_came_from() -> None:
    """The API and capsule turn follows a system and a user message; a tool session's follows a `submit` of a
    task read as documents. The API wording is the one the fused test measured."""

    api = inline_proofread.instruction(CAPABLE_C, _window(), [])
    tool = inline_proofread.instruction(CAPABLE_C, _window(), [], tool_session=True)
    assert "是 system 指令与第一条 user 消息里给你的背景资料" in api and "代替 system 指令里交完整" in api
    assert "是 `protocol` 块与 `payload` 块里给你的背景资料" in tool and "system 指令" not in tool
    assert "$" not in api + tool


# --- a tool session: the runtime owes a proofread ------------------------------------------------------------


def _runtime(tmp_path, *, proofread: bool = True) -> tuple[AgentTaskRuntime, dict]:
    validator = {"window": serialize_window(_window()),
                 **({PROOFREAD_PARAM: {"names": []}} if proofread else {})}
    spec = AgentTaskSpec(
        task_id="window", session_type="correction", input_hash="sha256:window", goal="correct the window",
        validator_id="correction-window", protocol_key="correction", context_key="run",
        metadata={"validator": validator, "variant": "capableC", "capability_tier": "capable",
                  "max_repair_attempts": 1})
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment", assignment_id="a", worker_goal="g", tasks=[spec], bootstrap_text="b",
        protocol_documents={"correction": "p"}, context_documents={"run": "c"}, knowledge_ref="k",
        knowledge_snapshot_identity="k", execution_identity={"driver": "test"},
        validators=runtime_validators("correction-window"), session_scope="task")
    claimed = runtime.next_task(assignment_id="a", worker_id="w", request_id="claim", expected_control_generation=1)
    return runtime, dict(assignment_id="a", task_id="window", worker_id="w",
                         lease_generation=claimed["task"]["lease_generation"])


def _submit(runtime, lease, request_id: str, candidate: str) -> dict:
    return runtime.submit(**lease, request_id=request_id, input_hash="sha256:window", candidate=candidate)


def _record(runtime) -> dict:
    return runtime.task_record(assignment_id="a", task_id="window")


def _accepted_text(runtime) -> str:
    return json.loads(runtime.read_artifact(_record(runtime)["accepted_artifact_ref"]))["artifact"]


def test_a_passing_answer_owes_the_proofread_turn_before_it_is_accepted(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    owed = _submit(runtime, lease, "s1", DRAFT)
    assert owed["status"] == "repairable" and owed["proofread"] is True and owed["validation_errors"] == []
    assert "交卷之前再校对一遍" in owed["repair_instruction"] and "`payload` 块" in owed["repair_instruction"]
    # the passing answer is what a session ending now leaves; the turn spends no repair budget
    assert _record(runtime)["last_candidate"] == DRAFT and owed["repair_attempts"] == 0
    assert _record(runtime)["proofread"] == {"state": "owed", "round": 0}

    accepted = _submit(runtime, lease, "s2", _patch(_sub("3", "改")))
    assert accepted.get("accepted_task_id") == "window"
    assert "sub|3|3。|改|high|" in _accepted_text(runtime)
    assert _record(runtime)["proofread"]["outcome"]["status"] == inline_proofread.APPLIED


def test_the_draft_sent_back_unchanged_is_accepted_not_replayed_into_a_loop(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", DRAFT)
    assert _submit(runtime, lease, "s2", DRAFT).get("accepted_task_id") == "window"
    assert _accepted_text(runtime) == DRAFT
    assert _record(runtime)["proofread"]["outcome"]["status"] == inline_proofread.NO_CHANGE


def test_a_proofread_answer_that_fails_twice_accepts_the_draft(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", DRAFT)
    repair = _submit(runtime, lease, "s2", _patch(_sub("3", "三|三")))
    assert repair["status"] == "repairable" and repair["proofread"] is True and repair["validation_errors"]
    assert "没能叠到" in repair["repair_instruction"]
    assert _submit(runtime, lease, "s3", _patch(_sub("3", "三|三"))).get("accepted_task_id") == "window"
    assert _accepted_text(runtime) == DRAFT
    assert _record(runtime)["proofread"]["outcome"]["status"] == inline_proofread.FALLBACK


def test_without_the_param_a_passing_answer_is_accepted_at_once(tmp_path) -> None:
    runtime, lease = _runtime(tmp_path, proofread=False)
    assert _submit(runtime, lease, "s1", DRAFT).get("accepted_task_id") == "window"
    assert _record(runtime)["proofread"] == {}


def test_a_session_that_ends_with_the_turn_owed_leaves_its_passing_answer(tmp_path) -> None:
    """The proofread answer failed and the CLI left: the harness gets the passing answer as accepted, not the
    patch nor a spent chain that would send the window to a new session."""

    runtime, lease = _runtime(tmp_path)
    _submit(runtime, lease, "s1", DRAFT)
    _submit(runtime, lease, "s2", _patch(_sub("3", "三|三")))
    record = _record(runtime)
    left = _proofread_left(record["proofread"], record["last_candidate"])
    assert record["last_candidate"] == DRAFT and left["status"] == inline_proofread.FALLBACK
    assert _proofread_left({}, record["last_candidate"]) is None


def test_a_driver_error_during_the_owed_turn_leaves_the_passing_answer(tmp_path, monkeypatch) -> None:
    """The CLI times out after its first answer passed and before its proofread answer was taken: the window keeps
    that answer (a fallback), as it would after a clean exit, instead of failing and being asked again whole."""

    from finesub.llm.agent import agent_validators
    from finesub.llm.agent.local_agent import LocalAgentTimeoutError

    from .test_llm_agent_tool_session import _call, _ToolUsingFakeDriver
    from .test_llm_agent_tool_session_extras import MESSAGES, _client, _no_api

    accept = lambda candidate, _view: agent_validators.ValidationResult.accepted(candidate)  # noqa: E731
    # The fake driver's runtime asks for this id; every passing answer owes the turn, which nothing answers.
    monkeypatch.setitem(agent_validators.VALIDATOR_BUILDERS, "test-reject-bad", lambda _manifest: accept)
    monkeypatch.setitem(agent_validators.VALIDATOR_PROOFREADS, "test-reject-bad",
                        (lambda draft, manifest: "proofread it", None))
    _no_api(monkeypatch)

    def agent(server):
        task = _call(server, 1, "next_task")
        for index, block in enumerate(task["required_blocks"]):
            _call(server, 10 + index, "read_context", ref=block["ref"])
        _call(server, 20, "submit", payload="good")

    class Driver(_ToolUsingFakeDriver):
        def run(self, messages, **kwargs):
            super().run(messages, **kwargs)
            raise LocalAgentTimeoutError("no output for 600 s")

    result = _client(Driver(agent), tmp_path).complete(
        LLMRole.GENERAL_CAPABLE, MESSAGES, validator_spec={"id": "test-reject-bad", "params": {}})
    assert result.content == "good"
    assert result.inline_proofread["status"] == inline_proofread.FALLBACK


def test_the_server_reports_no_repair_budget_on_a_proofread_turn(tmp_path) -> None:
    """A draft that passed on the session's last repair round still gets its proofread turn: the server must not
    tell the agent its budget is spent."""

    from finesub.llm.agent.agent_mcp_server import HarnessToolServer

    runtime, lease = _runtime(tmp_path)
    server = HarnessToolServer(runtime, assignment_id="a", worker_id="w", session_id="s", instance_id="i")
    server._task = {"task_id": "window", "lease_generation": lease["lease_generation"]}
    server._manifest = json.loads(runtime.read_artifact(_record(runtime)["manifest_ref"]))
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "submit", "arguments": {"payload": DRAFT}}})
    out = json.loads(reply["result"]["content"][0]["text"])
    assert out["status"] == "repairable" and "repair_instruction" in out
    assert "repair_rounds_remaining" not in out and "message" not in out


# --- the per-window loop: API and capsule ------------------------------------------------------------------


SPANS = ((0.0, 3.0), (3.1, 6.0), (7.5, 8.5), (9.0, 10.0))
ANSWER = _whole(*(_sub(str(i)) for i in range(1, 5)))


def _stable(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": str(i), "start": a, "end": b, "text": f"{i}。"}
                                               for i, (a, b) in enumerate(SPANS, start=1)]}), encoding="utf-8")
    return stable


def _reply(content: str, **fields) -> LLMCallResult:
    return LLMCallResult(content=content, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                         raw_response={"candidates": [{"finishReason": "STOP"}],
                                       "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1,
                                                         "totalTokenCount": 2}}, **fields)


def _run(tmp_path, monkeypatch, *replies, inline: bool = True, retries: int = 1):
    from finesub.llm.stages.correction import execute_correction_windows

    from .support.fakes import FakeTokenCounter

    calls: list[dict] = []
    queue = iter(replies)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append({"messages": messages("capableC") if callable(messages) else messages, **kwargs})
            try:
                reply = next(queue)
            except StopIteration:
                raise AssertionError(f"unexpected call {len(calls)}") from None
            if isinstance(reply, Exception):
                raise reply
            return reply

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    profile = resolve_profile("text", "none", "intermediate", "serial")
    execute_correction_windows(
        stable_json=_stable(tmp_path), output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
        profile=profile, task_artifact_dir=tmp_path / "artifacts", task_id="t", knowledge_enabled=False,
        postprocess_profile=None, max_retries_per_window=retries, inline_proofread=inline, length_scale=1.0)
    ledger = [json.loads(line) for line in (tmp_path / "artifacts").rglob("correction-windows.jsonl").__next__()
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    return calls, [record for record in ledger if "content" in record]


def _artifacts(tmp_path, kind: str) -> list[dict]:
    rows = []
    for path in (tmp_path / "artifacts").rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("kind") == kind:
                rows.append(json.loads(line))
    return rows


def test_a_passing_window_is_proofread_in_its_conversation_before_it_is_committed(tmp_path, monkeypatch) -> None:
    calls, ledger = _run(tmp_path, monkeypatch, _reply(ANSWER), _reply(_patch(_sub("3", "改"))))
    first, turn = calls
    # the base prompt, then the answer and the instruction; the clip (none here) would stay on the base
    assert turn["messages"] == first["messages"] and not turn["previous_output"]
    assert [t["role"] for t in turn["followup_turns"]] == ["assistant", "user"]
    assert turn["followup_turns"][0]["content"] == ANSWER and "（这一窗没有点名的行）" in turn["followup_turns"][1]["content"]
    assert turn["repair_session_key"] == first["repair_session_key"]
    assert ledger[0]["inline_proofread"] == inline_proofread.APPLIED and "sub|3|3。|改|high|" in ledger[0]["content"]
    [artifact] = _artifacts(tmp_path, "correction_window_proofread")
    assert artifact["payload"]["changes"][0]["source_ids"] == ["3"]
    # the call is counted and booked in the task report like the window's own
    from finesub.llm import task_report, token_books

    [response] = _artifacts(tmp_path, "correction_proofread_response")
    assert response["payload"]["call"].endswith("-proofread0")
    assert len(token_books.book([response])) == 1 and "correction_proofread_response" in task_report._CALL_COUNTS
    # a tool session would be asked through the validator
    assert first["validator_spec"]["params"][PROOFREAD_PARAM] == {"names": []}
    # the window's own call says a next turn re-sends its opening (an Anthropic call writes it to the cache)
    assert first["followup_expected"] is True and "followup_expected" not in turn


def test_the_turn_follows_the_last_passing_whole_after_a_repair(tmp_path, monkeypatch) -> None:
    """A draft that passed in a repair round: the turn goes on from the base prompt and the merged whole, never
    from the patch or the earlier rounds; the repair rounds' budget is the correction's, not the turn's."""

    calls, ledger = _run(tmp_path, monkeypatch, _reply(_whole(_sub("1"), _sub("2"), _sub("3"))),
                         _reply(_patch(_sub("4"))), _reply(_patch()), retries=1)
    assert calls[1]["repair_instruction"] and "<patch>" in calls[1]["repair_instruction"]
    turn = calls[-1]
    assert turn["followup_turns"][0]["content"].count("sub|") == 4 and "<patch>" not in turn["followup_turns"][0]["content"]
    assert turn["messages"] == calls[0]["messages"] and ledger[0]["inline_proofread"] == inline_proofread.NO_CHANGE


def test_a_failed_proofread_reply_gets_one_repair_round_beside_the_conversation(tmp_path, monkeypatch) -> None:
    bad = _patch(_sub("3", "三|三"))
    calls, ledger = _run(tmp_path, monkeypatch, _reply(ANSWER), _reply(bad), _reply(bad))
    repair = calls[2]
    assert repair["followup_turns"] == calls[1]["followup_turns"] and repair["previous_output"] == bad
    assert repair["validation_errors"] and "没能叠到" in repair["repair_instruction"]
    assert ledger[0]["inline_proofread"] == inline_proofread.FALLBACK and ledger[0]["content"] == ANSWER


def test_a_call_that_fails_leaves_the_answer(tmp_path, monkeypatch) -> None:
    calls, ledger = _run(tmp_path, monkeypatch, _reply(ANSWER), RuntimeError("endpoint down"))
    assert ledger[0]["inline_proofread"] == inline_proofread.FALLBACK and ledger[0]["content"] == ANSWER


def test_the_zero_thinking_retry_comes_before_the_turn(tmp_path, monkeypatch) -> None:
    silent = replace(_reply(ANSWER), thinking_level="high")
    calls, ledger = _run(tmp_path, monkeypatch, silent, _reply(ANSWER), _reply(_patch()))
    assert calls[1]["fresh_session"] and "followup_turns" not in calls[1]
    assert calls[2]["followup_turns"] and ledger[0]["redo"]["chosen"] == "retry"
    assert ledger[0]["inline_proofread"] == inline_proofread.NO_CHANGE


def test_a_tool_session_reports_its_own_turn_and_none_is_run_after_it(tmp_path, monkeypatch) -> None:
    reported = {"status": inline_proofread.APPLIED, "errors": [], "flagged": [], "changes": []}
    calls, ledger = _run(tmp_path, monkeypatch, _reply(ANSWER, inline_proofread=reported))
    assert len(calls) == 1 and ledger[0]["inline_proofread"] == inline_proofread.APPLIED


def test_a_cell_whose_candidates_answer_in_another_set_skips_the_turn(tmp_path, monkeypatch) -> None:
    """API and capsule run no turn; a tool session is not asked either: the validator params carry no proofread,
    so the runtime asks none. The window records `skipped`."""
    from finesub.llm.stages import three_pass

    monkeypatch.setattr(three_pass, "inline_proofread_cell", lambda profile: False)  # the cell changed on resume
    calls, ledger = _run(tmp_path, monkeypatch, _reply(ANSWER))
    assert len(calls) == 1 and ledger[0]["inline_proofread"] == inline_proofread.SKIPPED
    assert PROOFREAD_PARAM not in calls[0]["validator_spec"]["params"]


def test_the_switch_off_asks_nothing_and_records_nothing(tmp_path, monkeypatch) -> None:
    calls, ledger = _run(tmp_path, monkeypatch, _reply(ANSWER), inline=False)
    assert len(calls) == 1 and "inline_proofread" not in ledger[0]
    assert PROOFREAD_PARAM not in calls[0]["validator_spec"]["params"]
    assert calls[0]["followup_expected"] is False


@pytest.mark.parametrize("transport", ["conversational", "tool-session"])
def test_a_session_the_turn_is_not_run_in_says_so(transport) -> None:
    from finesub.llm.client import _session_proofread, _without_proofread

    spec = {"id": "correction-window", "params": {"window": {}, PROOFREAD_PARAM: {"names": []}}}
    assert PROOFREAD_PARAM not in _without_proofread(spec)["params"]
    assert _session_proofread(object(), transport, spec)["status"] == inline_proofread.SKIPPED
    assert _session_proofread(object(), "capsule", spec) is None


# --- the client: the turn goes on from the base prompt ------------------------------------------------------


def test_a_capsule_that_keeps_the_conversation_is_handed_the_instruction_alone(monkeypatch) -> None:
    """The window's conversation read the base prompt and wrote the answer: the turn continues it with the
    instruction; a capsule that keeps no conversation gets the whole of it, the shape the fused test measured."""

    from .test_llm_agent_continuation import REPLY, SYSTEM, TASK, _call, _client
    from .test_llm_model_router import _FakeAgentDriver, _agent_client_with_mode

    instruction = {"role": "user", "content": "proofread it"}
    driver, client = _client(monkeypatch)
    _call(client, [SYSTEM, TASK])
    _call(client, [SYSTEM, TASK], followup_turns=[REPLY, instruction])
    messages, kwargs = driver.calls[1]
    assert messages == [instruction] and kwargs["continuation"] is True

    fresh = _FakeAgentDriver(session_reuse=False)
    _call(_agent_client_with_mode(fresh, "per-window"), [SYSTEM, TASK], followup_turns=[REPLY, instruction])
    sent, kwargs = fresh.calls[0]
    assert sent == [SYSTEM, TASK, REPLY, instruction] and not kwargs.get("previous_output")
