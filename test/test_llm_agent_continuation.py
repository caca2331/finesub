"""A capsule turn that only adds messages continues the chain's conversation
with just those (docs/llm_local_agent.md §12.1).

The client keeps what each per-window conversation has read; a call whose
messages -- media aside -- start with all of it sends the rest and resumes.
Anything else is today's call: full messages, and a first attempt starts a
fresh conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.client import attach_file_to_messages
from finesub.llm.media_upload import UploadedFileRef
from finesub.llm.routing.config import LLMRole
from test.test_llm_model_router import _FakeAgentDriver, _agent_client_with_mode, _no_api

KEY = "correction-0001"
SYSTEM = {"role": "system", "content": "the contract"}
TASK = {"role": "user", "content": "the window"}
REPLY = {"role": "assistant", "content": "agent-ok"}
FILL = {"role": "user", "content": "the fill"}


def _client(monkeypatch, mode: str = "per-window", **driver_kwargs):
    _no_api(monkeypatch)
    driver = _FakeAgentDriver(session_reuse=True, **driver_kwargs)
    return driver, _agent_client_with_mode(driver, mode)


def _call(client, messages, **kwargs):
    return client.complete(LLMRole.GENERAL_CAPABLE, messages, repair_session_key=KEY, **kwargs)


def _ref(path: str = "C:/clips/w1.mp3") -> UploadedFileRef:
    return UploadedFileRef(
        file_id="w1", filename="w1.mp3", mime_type="audio/mpeg", local_path=path
    )


def test_a_turn_that_adds_messages_sends_only_those(monkeypatch) -> None:
    driver, client = _client(monkeypatch)

    _call(client, [SYSTEM, TASK])
    second = _call(client, [SYSTEM, TASK, REPLY, FILL])

    messages, kwargs = driver.calls[1]
    assert messages == [FILL]
    assert kwargs["conversation_handle"] == "conv-1"
    assert kwargs["continuation"] is True
    # Nothing the messages do not state: replaying them in full is the same task.
    assert second.resumable is True
    continuation = second.execution_attempts[-1]["continuation"]
    assert continuation == {"continued": True, "delta_messages": 1, "reason": ""}


def test_media_moved_to_the_fill_turn_still_continues(monkeypatch) -> None:
    """`attach_file_to_messages` hangs the media on the last user message --
    the task on the request turn, the fill on the delivery -- and turns that
    message's string into ``[text, file]``."""

    driver, client = _client(monkeypatch)

    _call(client, attach_file_to_messages([SYSTEM, TASK], _ref()))
    _call(client, attach_file_to_messages([SYSTEM, TASK, REPLY, FILL], _ref()))

    messages, kwargs = driver.calls[1]
    # The conversation holds the media already; the delta carries none.
    assert messages == [FILL]
    assert kwargs["continuation"] is True


@pytest.mark.parametrize(
    "first_ref, second_ref",
    [(None, _ref()), (_ref(), _ref("C:/clips/w1-downgraded.mp3"))],
    ids=["media-not-in-the-conversation", "another-media"],
)
def test_media_the_conversation_does_not_hold_starts_afresh(
    monkeypatch, first_ref, second_ref
) -> None:
    driver, client = _client(monkeypatch)
    first = [SYSTEM, TASK]
    _call(client, attach_file_to_messages(first, first_ref) if first_ref else first)
    second = _call(client, attach_file_to_messages([SYSTEM, TASK, REPLY, FILL], second_ref))

    messages, kwargs = driver.calls[1]
    assert len(messages) == 4 and kwargs["conversation_handle"] == ""
    assert "continuation" not in kwargs
    assert second.execution_attempts[-1]["continuation"]["reason"] == "media"


@pytest.mark.parametrize(
    "messages",
    [
        # A replayed reply that differs by a word.
        [SYSTEM, TASK, {"role": "assistant", "content": "agent-ok!"}, FILL],
        # A parse retry re-sending the same messages.
        [SYSTEM, TASK],
    ],
    ids=["prefix", "parse-retry"],
)
def test_messages_that_do_not_extend_the_record_start_afresh(
    monkeypatch, messages
) -> None:
    driver, client = _client(monkeypatch)

    _call(client, [SYSTEM, TASK])
    second = _call(client, messages)

    sent, kwargs = driver.calls[1]
    assert sent == messages and kwargs["conversation_handle"] == ""
    assert second.execution_attempts[-1]["continuation"]["reason"] == "prefix"


def test_a_ladder_retry_after_a_continued_turn_starts_afresh(monkeypatch) -> None:
    """The content filter ladder re-sends a changed fill turn. The continued
    turn's reply -- blocked or not -- is in the record, so the retry no longer
    extends it and never lands in the conversation that was blocked."""

    driver, client = _client(monkeypatch)

    _call(client, [SYSTEM, TASK])
    _call(client, [SYSTEM, TASK, REPLY, FILL])
    _call(client, [SYSTEM, TASK, REPLY, {"role": "user", "content": "the fill, redacted"}])

    sent, kwargs = driver.calls[2]
    assert len(sent) == 4 and kwargs["conversation_handle"] == ""


def test_the_record_follows_the_handle_the_cli_returns(monkeypatch) -> None:
    driver, client = _client(monkeypatch)
    later_fill = {"role": "user", "content": "more"}

    _call(client, [SYSTEM, TASK])
    _call(client, [SYSTEM, TASK, REPLY, FILL])
    _call(client, [SYSTEM, TASK, REPLY, FILL, REPLY, later_fill])

    sent, kwargs = driver.calls[2]
    # The fake hands out a new handle every call: the third turn resumes the
    # one the second returned.
    assert sent == [later_fill] and kwargs["conversation_handle"] == "conv-2"


def test_a_call_that_returns_no_handle_leaves_nothing_to_continue(monkeypatch) -> None:
    driver, client = _client(monkeypatch)
    real_run = driver.run

    def run(messages, **kwargs):
        result = real_run(messages, **kwargs)
        result.conversation_handle = ""
        return result

    driver.run = run
    _call(client, [SYSTEM, TASK])
    _call(client, [SYSTEM, TASK, REPLY, FILL])

    sent, kwargs = driver.calls[1]
    assert len(sent) == 4 and kwargs["conversation_handle"] == ""


def test_a_failed_continuation_rebuilds_in_full(monkeypatch) -> None:
    from finesub.llm.agent.local_agent import LocalAgentTransientError

    driver, client = _client(monkeypatch, resume_failure=LocalAgentTransientError("cold"))

    _call(client, [SYSTEM, TASK])
    result = _call(client, [SYSTEM, TASK, REPLY, FILL])

    resumed, rebuilt = driver.calls[1:]
    assert resumed[0] == [FILL] and resumed[1]["continuation"] is True
    assert rebuilt[0] == [SYSTEM, TASK, REPLY, FILL]
    assert rebuilt[1]["conversation_handle"] == "" and "continuation" not in rebuilt[1]
    assert result.execution_attempts[-1]["continuation"]["reason"] == "rebuilt"


def test_a_repair_is_unchanged_and_ends_the_record(monkeypatch) -> None:
    driver, client = _client(monkeypatch)
    messages = [SYSTEM, TASK, REPLY, FILL]

    _call(client, [SYSTEM, TASK])
    _call(client, messages)
    repair = _call(client, messages, previous_output="agent-ok", validation_errors=["row 1"])
    _call(client, [*messages, REPLY, {"role": "user", "content": "more"}])

    repaired, repair_kwargs = driver.calls[2]
    assert repaired == messages
    assert repair_kwargs["conversation_handle"] == "conv-2"
    assert repair_kwargs["previous_output"] == "agent-ok"
    assert "continuation" not in repair_kwargs
    assert repair.resumable is False
    # The repair context is in the conversation now: nothing extends it.
    after, after_kwargs = driver.calls[3]
    assert len(after) == 6 and after_kwargs["conversation_handle"] == ""


def test_retiring_a_key_drops_its_record(monkeypatch) -> None:
    driver, client = _client(monkeypatch)

    _call(client, [SYSTEM, TASK])
    client._retire_agent_conversations(KEY)

    assert client._agent_conversation_records == {}
    _call(client, [SYSTEM, TASK, REPLY, FILL])
    assert driver.calls[1][1]["conversation_handle"] == ""


def test_resume_mode_keeps_inheriting_in_full(monkeypatch) -> None:
    driver, client = _client(monkeypatch, mode="resume")

    _call(client, [SYSTEM, TASK])
    second = _call(client, [SYSTEM, TASK, REPLY, FILL])

    sent, kwargs = driver.calls[1]
    assert len(sent) == 4 and kwargs["conversation_handle"] == "conv-1"
    assert "continuation" not in kwargs
    assert second.resumable is False
    assert client._agent_conversation_records == {}


def test_a_call_with_no_session_says_so(monkeypatch) -> None:
    driver, client = _client(monkeypatch, mode="api")

    result = _call(client, [SYSTEM, TASK])

    assert result.execution_attempts[-1]["continuation"]["reason"] == "no_session"


def test_the_request_turn_carries_the_windows_key(tmp_path, monkeypatch) -> None:
    from test.test_llm_auto_style_window import DIRECT_DELIVERY, _Client, _request, _run

    _run(tmp_path, monkeypatch, [_request(), DIRECT_DELIVERY])

    (_opening, opening_kwargs), (_delivery, delivery_kwargs) = _Client.calls
    assert opening_kwargs["repair_session_key"] == "correction-0001"
    assert delivery_kwargs["repair_session_key"] == "correction-0001"


@pytest.mark.parametrize("chunk_label", ["", "c2"])
def test_the_preparation_rounds_carry_a_key_per_chunk(tmp_path, monkeypatch, chunk_label) -> None:
    """Chunks of an over-long source run in parallel on one client under one
    task id: each needs its own key, or one chunk's first round drops the
    conversation the other is about to continue."""

    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages.prepare_task import run_prepare_task
    from test.test_llm_prepare_task import (
        DELIVERY,
        REQUEST,
        TRANSCRIPT,
        FakeSearchClient,
        RecordingClient,
        _knowledge_root,
    )

    client = RecordingClient([REQUEST, DELIVERY])
    run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
        chunk_label=chunk_label,
    )

    expected = f"prepare-t-{chunk_label}" if chunk_label else "prepare-t"
    assert [kwargs["repair_session_key"] for _role, _m, kwargs in client.calls] == [expected] * 2


# --- drivers ------------------------------------------------------------------


def test_a_continued_capsule_marks_its_messages_as_a_delta(tmp_path) -> None:
    from finesub.llm.agent.local_agent import AgentCapsuleManager

    manager = AgentCapsuleManager(tmp_path / "runtime", keep_all=True)
    location = manager.resolve_location()
    capsule, payload = manager.create(
        location,
        [FILL],
        task="correction",
        native_search=False,
        profile_id="",
        max_result_bytes=1024,
        timeout_seconds=10,
        continuation_of="conv-1",
    )

    assert capsule.continuation_of == "conv-1"
    assert json.loads(payload)["messages"] == [FILL]
    ledger = json.loads(capsule.manifest_path.read_text(encoding="utf-8"))["inputs"][0]
    assert ledger["delta"] is True and ledger["base_conversation"] == "conv-1"


def test_claude_tells_a_continued_turn_that_only_new_messages_follow(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import (
        AGENT_TASK_PROMPT_CONTINUATION,
        AGENT_TASK_PROMPT_STDIN_ONLY,
    )
    from test.test_llm_local_agent import _claude_driver

    driver = _claude_driver(tmp_path, monkeypatch, max_result_bytes=8192)

    def prompt(**kwargs) -> str:
        result = driver.run(
            [{"role": "user", "content": "argv"}],
            task="correction",
            session_scope="assignment",
            conversation_key="correction-0001",
            conversation_handle="11111111-2222-3333-4444-555555555555",
            **kwargs,
        )
        return json.loads(result.content)[-1]

    assert prompt(continuation=True) == AGENT_TASK_PROMPT_CONTINUATION
    assert prompt() == AGENT_TASK_PROMPT_STDIN_ONLY
    assert "only the messages that are new" in AGENT_TASK_PROMPT_CONTINUATION


def test_codex_frames_a_continued_turn_with_the_continuation_prompt(tmp_path) -> None:
    from types import SimpleNamespace

    from finesub.llm.agent.local_agent import (
        AGENT_TASK_PROMPT_CONTINUATION,
        CodexDriverConfig,
        CodexLocalAgentDriver,
        _without_user_rules_notice,
    )

    driver = CodexLocalAgentDriver(CodexDriverConfig(runtime_root=tmp_path / "runtime"))
    driver._uses_harness_home = lambda probe: False  # type: ignore[assignment]
    driver.probe = lambda refresh=False: None  # type: ignore[assignment]
    capsule = SimpleNamespace(continuation_of="thread-1")
    framed, meta = driver._prepare_capsule_input(capsule, b"{}", conversation_handle="thread-1")

    assert framed.decode("utf-8").startswith(AGENT_TASK_PROMPT_CONTINUATION)
    assert meta == {"resume_prompt_via_stdin": True}
    assert _without_user_rules_notice(AGENT_TASK_PROMPT_CONTINUATION) != AGENT_TASK_PROMPT_CONTINUATION


def test_codebuddy_tells_a_continued_turn_that_only_new_messages_follow(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import (
        AGENT_TASK_PROMPT_CONTINUATION,
        AGENT_TASK_PROMPT_STDIN_ONLY,
    )
    from test.test_llm_agent_codebuddy import _driver

    driver = _driver(tmp_path, monkeypatch)

    def prompt(**kwargs) -> str:
        result = driver.run(
            [{"role": "user", "content": "argv"}],
            task="correction",
            session_scope="assignment",
            conversation_key="correction-0001",
            conversation_handle="11111111-2222-3333-4444-555555555555",
            **kwargs,
        )
        return json.loads(result.content)[-1]

    assert prompt(continuation=True) == AGENT_TASK_PROMPT_CONTINUATION
    assert prompt() == AGENT_TASK_PROMPT_STDIN_ONLY


def test_agy_names_the_task_files_as_holding_only_new_messages(tmp_path) -> None:
    from types import SimpleNamespace

    from finesub.llm.agent.local_agent import agent_continuation_sentence
    from test.test_llm_local_agent import _agy_driver, _agy_task_parts

    driver = _agy_driver(tmp_path)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")
    driver._grant_permissions = lambda *a, **k: None

    def prompt(continuation_of: str) -> str:
        capsule = SimpleNamespace(
            root=tmp_path / "domain" / "episode",
            messages_path=tmp_path / "domain" / "episode" / "task.json",
            continuation_of=continuation_of,
        )
        _agy_task_parts(capsule.root)
        argv = driver._argv(
            capsule,
            native_search=False,
            probe=driver.probe(),
            session_scope="assignment",
            conversation_handle=continuation_of or "conv",
        )
        return argv[argv.index("--print") + 1]

    sentence = agent_continuation_sentence("the task files named below")
    continued, fresh = prompt("conv-1"), prompt("")
    assert sentence in continued and sentence not in fresh
    # Replaced, not appended: "the exact chat messages are in the task files"
    # would contradict "it holds only the new ones".
    assert "The exact chat messages" in fresh and "The exact chat messages" not in continued


def test_codex_books_a_resumed_turn_without_the_turns_before_it(tmp_path) -> None:
    """`codex exec resume` reports the whole session in `turn.completed`."""

    from finesub.llm.agent.local_agent import CodexDriverConfig, CodexLocalAgentDriver

    driver = CodexLocalAgentDriver(CodexDriverConfig(runtime_root=tmp_path / "runtime"))
    first = {"input_tokens": 17583, "cached_input_tokens": 0, "output_tokens": 1864, "source": "x"}
    assert driver._own_session_usage("", "t-1", first) == first
    cumulative = {"input_tokens": 57001, "cached_input_tokens": 17408, "output_tokens": 8987, "source": "x"}
    own = driver._own_session_usage("t-1", "t-1", cumulative)
    assert own == {
        "input_tokens": 39418,
        "cached_input_tokens": 17408,
        "output_tokens": 7123,
        "source": "x",
        "resumed_usage": "subtracted",
    }
    # The next resume subtracts the count this turn reported, not the first one.
    later = driver._own_session_usage("t-1", "t-1", {**cumulative, "input_tokens": 60000})
    assert later["input_tokens"] == 2999 and later["resumed_usage"] == "subtracted"
    unseen = driver._own_session_usage("t-9", "t-9", dict(first))
    assert unseen["resumed_usage"] == "cumulative"
    shrunk = driver._own_session_usage("t-1", "t-1", {"input_tokens": 5})
    assert shrunk["resumed_usage"] == "mismatch_kept" and shrunk["input_tokens"] == 5


def test_agy_books_a_resumed_turn_without_the_turns_before_it(tmp_path) -> None:
    """agy's `result` counts the whole conversation; a repair round on the same
    conversation booked the first turn twice (2026-11-08: 68,682 + 126,849 for
    126,849 spent). Its steps count only their turn and add to the total."""

    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver, _agy_usage_shape

    driver = AgyLocalAgentDriver(AgyDriverConfig(runtime_root=tmp_path / "runtime"))

    def result(input_tokens, output, thinking, turn_thinking):
        usage = _agy_usage_shape({"input_tokens": input_tokens, "output_tokens": output,
                                  "thinking_tokens": thinking}, source="agy_result_event", cumulative=True)
        return {**usage, "turn_reasoning_output_tokens": turn_thinking}

    assert driver._own_session_usage("", "c", result(13383, 1521, 280, 280))["input_tokens"] == 13383
    own = driver._own_session_usage("c", "c", result(33231, 1867, 479, 199))
    assert (own["input_tokens"], own["output_tokens"], own["reasoning_output_tokens"]) == (19848, 346, 199)
    assert own["turn_reasoning_output_tokens"] == 199, "already this turn's: never subtracted"
    assert own["conversation_cumulative"] is False and own["resumed_usage"] == "subtracted"
    # A turn ended on accept reports its steps: kept as is, added to the running total.
    steps = _agy_usage_shape({"input_tokens": 20000, "output_tokens": 300, "thinking_tokens": 100},
                             source="agy_step_events", cumulative=False)
    assert driver._own_session_usage("c", "c", steps) == steps
    third = driver._own_session_usage("c", "c", result(73231, 2367, 679, 100))
    assert (third["input_tokens"], third["output_tokens"]) == (20000, 200)


def test_a_resume_answered_by_another_conversation_is_not_subtracted(tmp_path) -> None:
    """The call then fails on the changed handle; its spend is that other conversation's own."""

    from finesub.llm.agent.local_agent import CodexDriverConfig, CodexLocalAgentDriver

    driver = CodexLocalAgentDriver(CodexDriverConfig(runtime_root=tmp_path / "runtime"))
    driver._own_session_usage("", "t-1", {"input_tokens": 5000, "output_tokens": 100})
    other = driver._own_session_usage("t-1", "t-2", {"input_tokens": 3000, "output_tokens": 40})
    assert other == {"input_tokens": 3000, "output_tokens": 40}
    # It is booked as t-2's start.
    assert driver._own_session_usage("t-2", "t-2", {"input_tokens": 3500, "output_tokens": 45})["input_tokens"] == 500


def test_codebuddy_books_a_resumed_turn_without_the_turns_before_it(tmp_path) -> None:
    """hy3 on 2.164 (2026-11-08): `--resume`'s `result` counts every turn so far."""

    from finesub.llm.agent.local_agent import CodeBuddyDriverConfig, CodeBuddyLocalAgentDriver

    driver = CodeBuddyLocalAgentDriver(CodeBuddyDriverConfig(runtime_root=tmp_path / "runtime"))
    def turn(input_tokens, read, output):
        return {"input_tokens": input_tokens, "cache_read_input_tokens": read, "cache_creation_input_tokens": 0,
                "output_tokens": output, "source": "codebuddy_result_event", "total_cost_usd": 0.0}

    driver._own_session_usage("", "s", turn(11034, 512, 620))
    own = driver._own_session_usage("s", "s", turn(22853, 11456, 623))
    assert (own["input_tokens"], own["cache_read_input_tokens"], own["output_tokens"]) == (11819, 10944, 3)
    # The per-message sum after exit-on-accept is the turn's own.
    messages = {**turn(12000, 11000, 5), "source": "codebuddy_message_events"}
    assert driver._own_session_usage("s", "s", messages) == messages


def test_codex_resume_is_booked_through_the_driver(tmp_path, monkeypatch) -> None:
    from test.test_llm_local_agent import _driver

    driver = _driver(tmp_path, monkeypatch)
    # Unframed: the resume prompt's own words ("tools") would trip the fake's markers.
    driver._prepare_capsule_input = lambda capsule, data, **kwargs: (data, {})
    common = dict(task="correction", session_scope="assignment", conversation_key="k")
    first = driver.run([{"role": "user", "content": "hi"}], **common)
    resumed = driver.run(
        [{"role": "user", "content": "more"}],
        conversation_handle=first.conversation_handle,
        **common,
    )

    assert "resumed_usage" not in first.usage
    # The fake reports the same count every turn: the resumed turn's own is zero.
    assert resumed.usage["resumed_usage"] == "subtracted"
    assert resumed.usage["input_tokens"] == 0
