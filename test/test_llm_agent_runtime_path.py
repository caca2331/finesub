"""The durable worker contract the tool session and the transport tiers rest on.

The capsule worker glue that once sat behind ``local_agent_task_runtime`` is
gone (owner decision 2026-08-22: the transport derives from the session tier,
and the capsule transport is the narrow path). What remains here is the
worker's own contract and the harness's handling of an exhausted chain.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from finesub.llm.agent.agent_task_runtime import AgentTaskRuntime, AgentTaskSpec
from finesub.llm.agent.agent_transports import HeadlessTaskWorker
from finesub.llm.client import LLMCallResult, RoleClient
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import load_model_routes


class _ScriptedAgentDriver:
    """A session-reusing fake that answers from a script, one output per spawn."""

    conversation_ttl_seconds = 0.0

    def __init__(self, outputs, *, session_reuse: bool = True) -> None:
        self.config = SimpleNamespace(model="")
        self.outputs = list(outputs)
        self.session_reuse = session_reuse
        self.calls: list[tuple[list, dict]] = []

    def probe(self, *, refresh: bool = False):
        from finesub.llm.agent.local_agent import DriverProbe

        return DriverProbe(
            available=True,
            structured_events=True,
            no_persisted_session=True,
            no_user_config=True,
            no_user_rules=True,
            can_restrict_tools=True,
            has_web_search=True,
            supports_session_reuse=self.session_reuse,
            sandbox_kind="process_read_only",
        )

    def meets_requirements(self, probe=None, *, native_search: bool = False):
        return True

    def run(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        content = self.outputs.pop(0)
        return SimpleNamespace(
            content=content,
            reported_model="gpt-5.6-luna",
            execution_attempt={"backend": "local_agent", "capsule_id": f"cap-{len(self.calls)}"},
            episode_id=f"cap-{len(self.calls)}",
            conversation_handle=(
                (kwargs.get("conversation_handle") or f"conv-{len(self.calls)}")
                if self.session_reuse
                else ""
            ),
            turn_identity=f"sha256:turn-{len(self.calls)}",
            normalized_events=({"event": "turn.completed"},),
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def _client(driver, tmp_path, *, session_mode: str = ""):
    routes = load_model_routes(
        user_config={
            "model_groups": {
                "research-default": {
                    "targets": ["local-codex-completion-gpt-5_6-luna"]
                }
            }
        }
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    config = role_config_for("research", "quality", routes=routes)
    if session_mode:
        config = replace(config, agent_session_mode=session_mode)
    return RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={LLMRole.GENERAL_CAPABLE: config},
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
        agent_assignment_root=tmp_path / "assignments",
    )


def _no_api(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )


REJECT_BAD = {"id": "test-reject-bad", "params": {}}


@pytest.fixture(autouse=True)
def _register_reject_bad(monkeypatch):
    """A registry validator for the tests: the id is what crosses processes."""

    from finesub.llm.agent import agent_validators
    from finesub.llm.agent.agent_task_runtime import ValidationResult

    def build(_manifest):
        def validate(candidate, _m):
            if candidate == "bad":
                return ValidationResult.repairable("say good, not bad")
            return ValidationResult.accepted(candidate)

        return validate

    monkeypatch.setitem(agent_validators.VALIDATOR_BUILDERS, "test-reject-bad", build)


def test_a_worker_given_messages_sends_them_verbatim_every_turn(tmp_path) -> None:
    """The harness's prompt assembly, media parts and all, is what the agent sees."""

    from finesub.llm.agent.agent_task_runtime import ValidationResult

    def validator(candidate, _manifest):
        if candidate != "good":
            return ValidationResult.repairable("must say good")
        return ValidationResult.accepted(candidate)

    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="answer",
        tasks=[
            AgentTaskSpec(
                task_id="call",
                session_type="correction",
                input_hash="sha256:1",
                goal="answer",
                validator_id="strict",
            )
        ],
        validators={"strict": validator},
    )
    driver = _ScriptedAgentDriver(["bad", "good"], session_reuse=False)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [{"text": "fix"}, {"file_path": "clip.ogg"}]},
    ]
    worker = HeadlessTaskWorker(
        runtime,
        driver,
        assignment_id="assignment-1",
        worker_id="worker-1",
        max_repair_attempts=2,
        turn_messages=messages,
        run_kwargs={"reasoning_effort": "high", "profile_id": "route=x"},
    )

    outcome = worker.run_one()

    assert outcome["status"] == "assignment_complete"
    assert [sent for sent, _ in driver.calls] == [messages, messages]
    assert all(kwargs["reasoning_effort"] == "high" for _m, kwargs in driver.calls)
    assert all(kwargs["profile_id"] == "route=x" for _m, kwargs in driver.calls)
    assert [item.content for item in worker.turn_results] == ["bad", "good"]


_GOOD_WINDOW = (
    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
    "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
    "sub|2|1.0|0.0|二。|二|8|1|译1字；宜保持独立\n"
    "</singles>\n"
    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
    "sub|1|1.0|0.0|一。|一|high|1|\n"
    "sub|2|1.0|0.0|二。|二|high|1|\n"
    "</translated>"
)
_BAD_WINDOW = "<translated>\nsub|9|1.0|0.0|nine|九|high|1|\n</translated>"


def test_an_exhausted_chain_skips_straight_to_the_replacement(tmp_path, monkeypatch) -> None:
    """A backend that ran tier 1 itself has spent the chain; the harness moves on.

    With two repairs per chain and one replacement, the old loop would make
    up to six calls. A `repair_exhausted` result closes its chain in one, so
    the next call is the replacement -- fresh session, no repair context.
    """

    from finesub.llm.stages.correction import execute_correction_windows
    from finesub.subtitles.model import parse_srt
    from finesub.llm.routing.profiles import resolve_profile
    from .conftest import setattr_correction
    from .test_llm_correction_translation import FakeTokenCounter

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 1.5, "end": 2.5, "text": "二。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    responses = [(_BAD_WINDOW, True), (_GOOD_WINDOW, False)]
    seen: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            seen.append(kwargs)
            content, exhausted = responses.pop(0)
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
                backend="local_agent",
                agent_repair_rounds=2 if exhausted else 0,
                repair_exhausted=exhausted,
            )

    setattr_correction(monkeypatch, "RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        max_replacements_per_window=1,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
    )

    assert len(seen) == 2
    assert seen[0]["validator_spec"]["id"] == "correction-window"
    assert seen[0]["validator_spec"]["params"]["window"]["chunk_id"] == "0001"
    assert seen[0]["max_repair_attempts"] == 2
    assert seen[0]["fresh_session"] is False
    assert seen[1]["fresh_session"] is True
    assert seen[1]["previous_output"] == ""
    segments = parse_srt(output.read_text(encoding="utf-8"))
    assert [segment.text for segment in segments] == ["一", "二"]
    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    retries = [row for row in artifacts if row["kind"] == "correction_window_retry"]
    assert len(retries) == 1
    assert retries[0]["payload"]["replacement"] is True
    assert retries[0]["payload"]["attempt"] == 2
