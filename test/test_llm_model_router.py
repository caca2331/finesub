from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.client import RoleClient, UploadedFileRef, extract_token_distribution
from finesub.llm.routing.config import (
    GEMINI_FREE_TIER,
    LLMRole,
    ModelEndpoint,
    RoleModelConfig,
    default_role_configs,
    role_config_for,
)
from finesub.llm.routing.model_router import FailureKind, ModelRouter, classify_failure
from finesub.llm.routing.model_routes import default_model_routes, load_model_routes
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.agent.local_agent import LocalAgentUnavailableError
from finesub.llm.rate_limit import ModelRateLimiter


# (The v1 per-step fallback stop test died with the chains: D8 flattened the
# groups because every non-final step's fallback set was identical anyway --
# a uniform fallback policy is the declared behaviour now, covered below.)


def test_uniform_fallback_advances_through_the_group(monkeypatch) -> None:
    routes = load_model_routes()
    calls: list[str] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        calls.append(model)
        if len(calls) < 4:
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        router=ModelRouter(routes),
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    assert calls == [
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.5-flash",
        "gemini/gemini-3.7-flash",
        "gemini/gemini-3.7-flash",
    ]
    assert result.target_id == "gemini-paid-3_7-flash"
    assert (
        result.route_decision["routing_identity_digest"]
        == routes.routing_identity_digest
    )
    assert result.route_decision["advisory_digest"] == routes.advisory_digest
    assert [
        row.get("outcome") for row in result.route_decision["candidates"]
    ] == [
        "failed",
        "failed",
        "failed",
        "success",
    ]


def test_target_aware_prompt_estimate_skips_small_context_candidate(
    monkeypatch,
) -> None:
    gemma = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemma-4-31b-it")
    flash = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.6-flash")
    config = RoleModelConfig(
        role=LLMRole.LIGHTWEIGHT,
        endpoint_chain=(gemma, flash),
        test_endpoint=flash,
    )
    factory_calls = []
    calls: list[str] = []

    def factory(variant):
        factory_calls.append(variant)
        return [{"role": "user", "content": variant or "single-template"}]

    monkeypatch.setattr("finesub.llm.client.estimate_call_input_tokens", lambda *a, **k: 20_000)

    def fake_chat_complete(messages, *, model, **kwargs):
        calls.append(model)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        role_configs={LLMRole.LIGHTWEIGHT: config},
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.LIGHTWEIGHT,
        factory,
        max_tokens=32_768,
    )

    assert calls == ["gemini/gemini-3.6-flash"]
    # No per-candidate variant overrides here, so the messages are assembled
    # once and reused across candidates.
    assert factory_calls == [""]
    assert result.route_decision["candidates"][0]["reason"] == "input_limit"
    assert result.route_decision["candidates"][1]["outcome"] == "success"


def _video_incapable_first_research_target(monkeypatch) -> None:
    """Make the free 3.6 Flash fact video-incapable for the ladder tests."""

    from dataclasses import replace as dc_replace

    import finesub.llm.routing.capabilities as capabilities

    real = capabilities.runtime_fact_for

    def fake(endpoint):
        entry = real(endpoint)
        if entry is not None and entry.fact_id == "gemini-free-3_6-flash":
            return dc_replace(entry, supports_video=False)
        return entry

    monkeypatch.setattr(capabilities, "runtime_fact_for", fake)


def test_video_call_downgrades_one_rung_on_video_incapable_target(
    monkeypatch, capsys
) -> None:
    """The video->audio ladder is a runtime safety net -- the
    hear-but-not-watch candidate answers with the audio clip, warns once, and
    the decision trace records the downgrade."""

    import json as json_module

    _video_incapable_first_research_target(monkeypatch)
    sent: list[tuple[str, str]] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        sent.append((model, json_module.dumps(messages, ensure_ascii=False)))
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        max_retries=0, rate_limiter=ModelRateLimiter(enabled=False)
    )
    video_ref = UploadedFileRef("files/w.mp4", "w.mp4", "video/mp4")
    audio_ref = UploadedFileRef("files/w.aac", "w.aac", "audio/aac")
    audio_cuts: list[int] = []

    def fallback() -> UploadedFileRef:
        audio_cuts.append(1)
        return audio_ref

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        file_ref=video_ref,
        fallback_audio_ref=fallback,
    )

    # First candidate (3.6 Flash) answers with the audio clip swapped in.
    assert result.target_id == "gemini-free-3_6-flash"
    assert len(sent) == 1 and "files/w.aac" in sent[0][1]
    assert "files/w.mp4" not in sent[0][1]
    assert audio_cuts == [1]
    first = result.route_decision["candidates"][0]
    assert first["decision"] == "accepted"
    assert first["media_downgrade"] == "video->audio"
    assert "video->audio" in capsys.readouterr().err

    # Warned once per target: a second call stays silent.
    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi again"}],
        file_ref=video_ref,
        fallback_audio_ref=fallback,
    )
    assert "video->audio" not in capsys.readouterr().err


def test_video_call_without_ladder_still_hard_filters(monkeypatch) -> None:
    _video_incapable_first_research_target(monkeypatch)
    sent: list[tuple[str, str]] = []

    import json as json_module

    def fake_chat_complete(messages, *, model, **kwargs):
        sent.append((model, json_module.dumps(messages, ensure_ascii=False)))
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        max_retries=0, rate_limiter=ModelRateLimiter(enabled=False)
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        file_ref=UploadedFileRef("files/w.mp4", "w.mp4", "video/mp4"),
    )

    # No fallback provider -> the incapable candidate is skipped as before
    # and the video ref rides through to the next (video-capable) target.
    assert result.target_id == "gemini-free-3_5-flash"
    assert (
        result.route_decision["candidates"][0]["reason"] == "capability_mismatch"
    )
    assert len(sent) == 1 and "files/w.mp4" in sent[0][1]


def test_packaged_plan_exposes_group_identity_and_fact_snapshot() -> None:
    config = default_role_configs()[LLMRole.GENERAL_CAPABLE]
    plan = ModelRouter().plan(config)
    trace = plan.decision_trace()

    assert trace["policy_id"] == "agent-text-preferred"
    assert trace["task_group_id"] == "research"
    assert trace["difficulty"] == "quality"
    assert trace["model_group_id"] == "research-default"
    assert [item["group_id"] for item in trace["effective_chain"]] == [
        "research-default"
    ] * 4
    assert trace["effective_chain"][0]["fact"]["fact_id"] == "gemini-free-3_6-flash"


def test_all_prefiltered_candidates_are_explained(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.routing.model_router.api_keys.provider_tier_enabled", lambda _tier: False
    )
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    with pytest.raises(RuntimeError, match="No eligible target") as raised:
        client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    trace = raised.value._harness_route_decision
    assert len(trace["candidates"]) == 4
    assert all(row["decision"] == "skipped" for row in trace["candidates"])
    assert all(row["reason"] == "provider_disabled" for row in trace["candidates"])
    # `reason x count`, not `reason=count`: "input_limit=2" in a real run was
    # read as "the input limit is 2" rather than "2 candidates hit it".
    assert "provider_disabledx4" in str(raised.value)


def test_native_search_errors_when_the_bound_group_has_no_native_target(
    monkeypatch,
) -> None:
    """Plan v2 D4: native search is a per-call filter over the bound group,
    with no fallback chain. A group whose members carry no search tool (the
    lightweight lites) errors instead of silently downgrading -- even though
    the paid lite's *fact* says it could ground, its target has no tool
    wired."""

    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")),
    )
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    with pytest.raises(RuntimeError, match="No eligible target") as raised:
        client.complete(
            LLMRole.LIGHTWEIGHT,
            [{"role": "user", "content": "hi"}],
            native_search=True,
        )
    trace = raised.value._harness_route_decision
    assert trace["native_search"] is True
    assert all(
        row["reason"] == "capability_mismatch" for row in trace["candidates"]
    )


def test_default_preset_serves_native_search_from_paid_grounding(
    monkeypatch,
) -> None:
    """The paid 3.7 target declares the search tool, so
    ``--retrieval native`` works with the shipped preset. The free members are
    filtered out per call (they cannot ground), and the request goes out with
    the tool enabled."""

    captured: dict[str, object] = {}

    def fake_chat_complete(messages, *, model, native_search_tool, **kwargs):
        captured.update(model=model, native_search_tool=native_search_tool)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        native_search=True,
    )

    assert captured == {
        "model": "gemini/gemini-3.7-flash",
        "native_search_tool": "google_search",
    }
    assert result.target_id == "gemini-paid-3_7-flash"
    trace = result.route_decision
    assert [row["reason"] for row in trace["candidates"][:3]] == [
        "capability_mismatch",
        "capability_mismatch",
        "capability_mismatch",
    ]


def test_native_search_serves_from_a_native_capable_binding(monkeypatch) -> None:
    """Binding a group with native-capable members makes retrieval=native
    work, and difficulty picks the variant -- the v1 "native forces the basic
    prompt" side effect is gone."""

    captured = {}

    def fake_chat_complete(messages, *, model, native_search_tool, **kwargs):
        captured.update(model=model, native_search_tool=native_search_tool)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)

    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    text = source.read_text(encoding="utf-8").replace(
        '"research/quality" = "research-default"',
        '"research/quality" = "gemini-native-search"',
        1,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "routes.toml"
        path.write_text(text, encoding="utf-8")
        routes = load_model_routes(path)
    from finesub.llm.routing.config import role_config_for

    config = role_config_for("research", "quality", routes=routes)
    client = RoleClient(
        router=ModelRouter(routes),
        rate_limiter=ModelRateLimiter(enabled=False),
        role_configs={LLMRole.GENERAL_CAPABLE: config},
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        native_search=True,
    )

    assert captured == {
        "model": "gemini/gemini-2.5-flash",
        "native_search_tool": "google_search",
    }
    assert result.route_decision["native_search"] is True


def test_known_non_429_4xx_stays_permanent_despite_quota_words() -> None:
    assert classify_failure(RuntimeError("HTTP 400 daily quota quotaId PerDay")) is (
        FailureKind.PERMANENT
    )
    assert classify_failure(RuntimeError("HTTP 401 quotaId PerMinute")) is (
        FailureKind.PERMANENT
    )
    assert classify_failure(RuntimeError("HTTP 408 request timeout")) is (
        FailureKind.PERMANENT
    )


def test_custom_endpoint_cannot_borrow_another_fact_id() -> None:
    endpoint = ModelEndpoint(
        GEMINI_FREE_TIER,
        "gemini/not-the-declared-model",
        fact_id="gemini-free-3_6-flash",
    )
    config = RoleModelConfig(
        role=LLMRole.GENERAL_CAPABLE,
        endpoint_chain=(endpoint,),
        test_endpoint=endpoint,
    )

    with pytest.raises(ValueError, match="resolves to"):
        ModelRouter().plan(config)


def test_a_plan_is_the_bound_group_gated_by_the_policy() -> None:
    """The whole of routing: membership picks the models, the policy subtracts.

    A policy can never *add* a target the bound group does not name -- which is
    what the retired group prepends did, leaving "which model answers" written
    down in two places that could disagree.
    """

    from finesub.llm.routing.config import role_config_for

    routes = default_model_routes()
    default_cell = default_role_configs()[LLMRole.GENERAL_CAPABLE]

    mixed = ModelRouter(policy_id="agent-text-preferred").plan(default_cell)
    assert mixed.task_group_id == "research"
    assert mixed.model_group_id == "research-default"
    # The default preset names no agent, so the mixed policy yields no agent.
    assert [item.target_id for item in mixed.candidates] == list(
        routes.model_groups["research-default"].target_ids
    )
    assert ModelRouter(policy_id="agent-only").plan(default_cell).candidates == ()

    # The agy preset is where agents are members, so it is where they appear.
    agy_cell = role_config_for("research", "quality", preset_id="agy")
    agy_mixed = ModelRouter(policy_id="agent-text-preferred").plan(agy_cell)
    assert [item.target_id for item in agy_mixed.candidates] == list(
        routes.model_groups["agy-basic"].target_ids
    )
    assert [
        item.target_id
        for item in ModelRouter(policy_id="agent-only").plan(agy_cell).candidates
    ] == [
        "local-agy-opus-4_6",
        "local-agy-media-gemini-3_7-flash",
        "local-agy-native-gemini-3_7-flash",
    ]
    assert all(
        item.endpoint.backend == "gemini_rest"
        for item in ModelRouter(policy_id="api-only").plan(agy_cell).candidates
    )


class _FakeAgentDriver:
    def __init__(
        self,
        *,
        model: str = "",
        failure: BaseException | None = None,
        usable: bool = True,
        session_reuse: bool = False,
        resume_failure: BaseException | None = None,
    ) -> None:
        self.config = SimpleNamespace(model=model)
        self.failure = failure
        self.usable = usable
        # Off by default: `assignment` scope fails before the spawn on a driver
        # that cannot resume, so every other test here exercises full replay.
        self.session_reuse = session_reuse
        self.resume_failure = resume_failure
        self.calls = []

    def probe(self, *, refresh: bool = False):
        """The pre-filter asks the client's own driver whether it can serve.

        A stub that cannot answer this would be filtered out before dispatch,
        so the fake declares the same capability set the real gate requires.
        """

        from finesub.llm.agent.local_agent import DriverProbe

        return DriverProbe(
            available=self.usable,
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
        current = probe or self.probe()
        return current.available and (not native_search or current.has_web_search)

    def run(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.failure is not None:
            raise self.failure
        if self.resume_failure is not None and kwargs.get("conversation_handle"):
            # The real driver hangs the attempt record off the exception
            # (`fail_before_spawn`); a fake that did not would let the caller
            # silently drop that evidence and still look correct.
            setattr(
                self.resume_failure,
                "_harness_execution_attempts",
                [{"backend": "local_agent", "capsule_id": "cap-cold"}],
            )
            raise self.resume_failure
        events = (
            (
                {
                    "event": "item.completed",
                    "item_type": "web_search",
                    "query": "q",
                },
            )
            if kwargs.get("native_search")
            else ({"event": "turn.completed"},)
        )
        # The real transport extracts the search rows itself and hands them
        # over on the attempt. A fake that left that out would invite the
        # client to grow a second copy of the same filter unnoticed.
        return SimpleNamespace(
            content="agent-ok",
            reported_model=self.config.model or "gpt-5.6-luna",
            execution_attempt={
                "backend": "local_agent",
                "capsule_id": "cap-1",
                "search_events": [
                    dict(event)
                    for event in events
                    if event.get("item_type") == "web_search"
                ],
            },
            episode_id="cap-1",
            conversation_handle=(
                f"conv-{len(self.calls)}" if self.session_reuse else ""
            ),
            normalized_events=events,
            usage={
                "input_tokens": 11,
                "cached_input_tokens": 3,
                "output_tokens": 5,
                "reasoning_output_tokens": 2,
                "total_tokens": 16,
            },
        )


def _routes_with_research_group(*targets: str):
    """Routes whose research cell is the given chain.

    The packaged presets other than `agy` name no agent, and no policy adds
    one any more, so reaching a Codex or Claude target is a *membership*
    question: a user group that lists it. That is what this builds, and it is
    exactly the migration path for anyone who used to rely on the retired
    policy prepends.
    """

    return load_model_routes(
        user_config={"model_groups": {"research-default": {"targets": list(targets)}}}
    )


def _agy_config(task_group: str, role: LLMRole) -> RoleModelConfig:
    return {role: role_config_for(task_group, "quality", preset_id="agy", role=role)}


def test_an_agent_receives_repair_context_as_arguments_not_as_extra_turns(
    monkeypatch,
) -> None:
    """An agent takes the previous output and the errors as capsule inputs.

    Putting them in the messages as well would hand it the same thing twice --
    and the driver is the layer that knows whether the answer even needs
    re-sending (a resumed session already has it).
    """

    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )
    routes = _routes_with_research_group("local-codex-completion-gpt-5_6-luna")
    driver = _FakeAgentDriver()
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
    )

    messages, kwargs = driver.calls[0]
    assert messages == [{"role": "user", "content": "hi"}]
    assert kwargs["previous_output"] == "sub|1|wrong"
    assert kwargs["validation_errors"] == ["Row 1 references unknown source id 3."]


def _agent_client(driver):
    routes = _routes_with_research_group("local-codex-completion-gpt-5_6-luna")
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    return RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )


def _no_api(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )


def test_an_agent_repair_resumes_the_conversation_that_wrote_the_output(
    monkeypatch,
) -> None:
    """A repair is a follow-up turn, not a stranger reading its own answer.

    Production does not go through the durable task runtime, so a repair used
    to spawn a *fresh* agent and hand it its own previous output as plain text.
    agy declines that outright and so retried blind. Carrying the handle for
    the length of one window's attempt chain makes the repair a second turn of
    the same conversation, which is what the agent backends are built for.
    """

    _no_api(monkeypatch)
    driver = _FakeAgentDriver(session_reuse=True)
    client = _agent_client(driver)

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        repair_session_key="correction-0001",
    )
    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
        repair_session_key="correction-0001",
    )

    first, repair = (kwargs for _messages, kwargs in driver.calls)
    assert first["session_scope"] == "assignment"
    assert first["conversation_handle"] == ""
    # The repair lands in the conversation the first attempt opened.
    assert repair["conversation_handle"] == "conv-1"
    assert repair["validation_errors"] == ["Row 1 references unknown source id 3."]


def test_a_driver_without_session_reuse_keeps_replaying_in_full(monkeypatch) -> None:
    """`assignment` scope on such a driver fails before the spawn, not softly.

    So reuse is opt-in per driver: everything that cannot prove it resumes
    keeps the behaviour it had, repair context and all.
    """

    _no_api(monkeypatch)
    driver = _FakeAgentDriver()
    client = _agent_client(driver)

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
        repair_session_key="correction-0001",
    )

    _messages, kwargs = driver.calls[0]
    assert kwargs.get("session_scope", "task") == "task"
    assert kwargs["previous_output"] == "sub|1|wrong"


def test_a_cold_conversation_falls_back_to_a_replay_instead_of_failing(
    monkeypatch,
) -> None:
    """A handle can go stale -- TTL, a compact, a killed CLI.

    Rebuilding costs one full replay, which is exactly the call this used to
    be, so the chain degrades to the old behaviour rather than losing the
    window to an error.
    """

    from finesub.llm.agent.local_agent import LocalAgentTransientError

    _no_api(monkeypatch)
    driver = _FakeAgentDriver(
        session_reuse=True,
        resume_failure=LocalAgentTransientError("no such session"),
    )
    client = _agent_client(driver)

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        repair_session_key="correction-0001",
    )
    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
        repair_session_key="correction-0001",
    )

    assert result.content == "agent-ok"
    resumed, rebuilt = (kwargs for _m, kwargs in driver.calls[1:])
    assert resumed["conversation_handle"] == "conv-1"
    # The rebuild re-sends the previous output, because nothing carries it now.
    assert rebuilt["conversation_handle"] == ""
    assert rebuilt["previous_output"] == "sub|1|wrong"
    # Both spawns are on the record: an audit that sees only the survivor reads
    # one call where two agent processes ran.
    assert len(result.execution_attempts) == 2


def test_a_spent_subscription_is_not_retried_as_a_cold_conversation(
    monkeypatch,
) -> None:
    """Rebuilding the conversation cannot refill an allowance.

    Retrying would spend a second spawn on a call that cannot succeed, and
    hide one of the two consecutive failures the quota ledger counts before it
    freezes the pool -- which is the mechanism that exists to stop us walking
    into a subscription we already know is spent.
    """

    from finesub.llm.agent.local_agent import LocalAgentQuotaError

    _no_api(monkeypatch)
    driver = _FakeAgentDriver(
        session_reuse=True, resume_failure=LocalAgentQuotaError("allowance spent")
    )
    client = _agent_client(driver)

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        repair_session_key="correction-0001",
    )
    with pytest.raises(Exception):
        client.complete(
            LLMRole.GENERAL_CAPABLE,
            [{"role": "user", "content": "hi"}],
            previous_output="sub|1|wrong",
            validation_errors=["Row 1 references unknown source id 3."],
            repair_session_key="correction-0001",
        )

    # The opening call, then exactly one failed resume -- not a second spawn.
    assert len(driver.calls) == 2


def test_a_repair_that_re_routes_does_not_inherit_another_drivers_session(
    monkeypatch,
) -> None:
    """A session id belongs to the CLI that issued it.

    Attempts route independently, so a repair can land on another vendor or on
    the other provider tier of the same model. Keying the cache on the chain
    alone handed that call a handle it does not own -- at best one wasted
    spawn, at worst a paid-tier call silently resuming a free-tier session.
    """

    _no_api(monkeypatch)
    driver = _FakeAgentDriver(session_reuse=True)
    client = _agent_client(driver)

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        repair_session_key="correction-0001",
    )
    from finesub.llm.routing.execution_policy import normalized_tier

    stored = list(client._agent_repair_conversations)

    assert stored == [
        (normalized_tier("LOCAL_CODEX"), "gpt-5.6-luna", "correction-0001")
    ]


def test_a_new_window_never_inherits_the_previous_windows_conversation(
    monkeypatch,
) -> None:
    """Chains are per window, deliberately.

    Reuse *between* independent units is the thing the agy A/B found to be a
    net loss; this only reuses within one window's attempt chain, so a fresh
    window must start a fresh conversation even though the client is shared.
    """

    _no_api(monkeypatch)
    driver = _FakeAgentDriver(session_reuse=True)
    client = _agent_client(driver)

    for chunk in ("0001", "0002"):
        client.complete(
            LLMRole.GENERAL_CAPABLE,
            [{"role": "user", "content": "hi"}],
            repair_session_key=f"correction-{chunk}",
        )

    assert [kwargs["conversation_handle"] for _m, kwargs in driver.calls] == ["", ""]


def test_an_agent_over_its_input_limit_loses_the_repair_context_not_the_call(
    monkeypatch,
) -> None:
    """The estimate has to count what the agent really receives.

    An agent gets the repair context as capsule inputs rather than as messages,
    so measuring the bare message list would under-count exactly the calls that
    grew -- and the candidate would be admitted over its own input limit while
    the drop that protects the stateless path never fires.
    """

    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )

    def estimate(messages, **kwargs):
        has_repair = any(msg.get("role") == "assistant" for msg in messages)
        return 2_000_000 if has_repair else 10

    monkeypatch.setattr("finesub.llm.client.estimate_call_input_tokens", estimate)
    routes = _routes_with_research_group("local-codex-completion-gpt-5_6-luna")
    driver = _FakeAgentDriver()
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
    )

    assert result.content == "agent-ok"
    _messages, kwargs = driver.calls[0]
    assert kwargs["previous_output"] == ""
    assert list(kwargs["validation_errors"]) == []
    accepted = [
        row
        for row in result.route_decision["candidates"]
        if row.get("decision") == "accepted"
    ]
    assert accepted[0]["repair_context"] == "dropped_input_limit"


def test_an_agent_member_dispatches_through_its_driver(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )
    routes = _routes_with_research_group("local-codex-completion-gpt-5_6-luna")
    driver = _FakeAgentDriver()
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    assert result.content == "agent-ok"
    assert result.backend == "local_agent"
    assert result.execution_attempts[0]["target_id"] == result.target_id
    assert result.api_attempts == []
    distribution = extract_token_distribution(result.raw_response)
    assert distribution["total_input_tokens"] == 11
    assert distribution["cached_input_tokens"] == 3
    assert distribution["uncached_input_tokens"] == 8
    assert distribution["thinking_tokens"] == 2
    assert distribution["output_tokens"] == 3
    assert driver.calls[0][1]["native_search"] is False
    assert driver.calls[0][1]["reasoning_effort"] == "xhigh"
    assert result.thinking_level == "xhigh"


def test_a_model_that_takes_no_thinking_parameter_is_never_forced_one(
    monkeypatch,
) -> None:
    """`thinking = false` means the model rejects the parameter, not "default".

    The global `local_agent_reasoning_effort` override used to win over that
    fact. agy refuses `--effort` for its Claude models before the call even
    starts, and a hard refusal classifies as transient -- two of them and the
    quota probe freezes the whole allowance for five hours over a flag.
    """

    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )
    routes = _routes_with_research_group("local-agy-opus-4_6")
    assert routes.target_fact("local-agy-opus-4_6").thinking_levels is None

    driver = _FakeAgentDriver()
    settings = ExecutionSettings(
        policy_id="agent-only", local_agent_reasoning_effort="xhigh"
    )
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    assert result.target_id == "local-agy-opus-4_6"
    assert driver.calls[0][1]["reasoning_effort"] == ""


def test_local_agent_fallback_applies_the_selected_models_thinking_map(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )
    routes = _routes_with_research_group(
        "local-codex-completion-gpt-5_6-luna",
        "local-codex-completion-gpt-5_6-sol",
    )
    drivers = {}

    def factory(provider_tier: str, model: str):
        driver = _FakeAgentDriver(
            model=model,
            failure=(
                LocalAgentUnavailableError("luna unavailable")
                if model == "gpt-5.6-luna"
                else None
            ),
        )
        drivers[model] = driver
        return driver

    settings = ExecutionSettings(policy_id="agent-only")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver_factory=factory,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    assert result.target_id == "local-codex-completion-gpt-5_6-sol"
    assert drivers["gpt-5.6-luna"].calls[0][1]["reasoning_effort"] == "xhigh"
    assert drivers["gpt-5.6-sol"].calls[0][1]["reasoning_effort"] == "high"
    assert result.thinking_level == "high"


def test_agent_unavailable_falls_back_to_the_next_member(monkeypatch) -> None:
    calls = []

    def fake_chat_complete(messages, *, model, **kwargs):
        calls.append(model)
        return {"choices": [{"message": {"content": "api-ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    routes = _routes_with_research_group(
        "local-codex-completion-gpt-5_6-luna", "gemini-free-3_6-flash"
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=_FakeAgentDriver(
            failure=LocalAgentUnavailableError("not installed")
        ),
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    assert result.content == "api-ok"
    assert result.backend == "gemini_rest"
    assert calls == ["gemini/gemini-3.6-flash"]
    assert result.route_decision["candidates"][0]["failure_kind"] == "unavailable"


def test_agent_native_search_uses_driver_tool_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )
    routes = _routes_with_research_group("local-codex-native-gpt-5_6-luna")
    driver = _FakeAgentDriver()
    settings = ExecutionSettings(policy_id="agent-only")
    client = RoleClient(
        router=ModelRouter(routes, policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs={
            LLMRole.GENERAL_CAPABLE: role_config_for(
                "research", "quality", routes=routes
            )
        },
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "research"}],
        native_search=True,
    )

    assert result.target_id == "local-codex-native-gpt-5_6-luna"
    assert driver.calls[0][1]["native_search"] is True
    assert result.execution_attempts[0]["search_events"][0]["query"] == "q"


def test_injected_agent_only_client_never_uses_count_tokens_api(monkeypatch) -> None:
    monkeypatch.setattr(
        "finesub.llm.token_budget.LocalGeminiTokenCounter.count_text",
        lambda self, text: (_ for _ in ()).throw(RuntimeError("local unavailable")),
    )
    monkeypatch.setattr(
        "finesub.llm.token_budget.GeminiCountTokensCounter.count_text",
        lambda self, text: (_ for _ in ()).throw(
            AssertionError("Gemini countTokens must not be constructed in agent-only")
        ),
    )
    settings = ExecutionSettings(policy_id="agent-only")
    client = RoleClient(
        router=ModelRouter(policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs=_agy_config("research", LLMRole.GENERAL_CAPABLE),
        local_agent_driver=_FakeAgentDriver(),
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "count locally"}],
    )

    assert result.backend == "local_agent"


def test_a_media_call_skips_the_text_only_agent_for_the_one_agy_fronts(
    monkeypatch,
) -> None:
    """Opus sits ahead of the media target in `agy-capable` deliberately.

    It is text-only, so this is how "prefer Opus for text windows, hand
    multimodal ones to the Gemini agy fronts" is expressed -- one group, the
    capability filter doing the choosing, no second binding.
    """

    monkeypatch.setattr(
        "finesub.llm.llm_runtime.chat_complete",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("API must not run")),
    )
    driver = _FakeAgentDriver()
    settings = ExecutionSettings(policy_id="agent-only")
    client = RoleClient(
        router=ModelRouter(policy_id=settings.policy_id),
        execution_settings=settings,
        role_configs=_agy_config("correction-mm", LLMRole.AUDIO_MULTIMODAL),
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "audio"}],
        file_ref=UploadedFileRef("f", "clip.aac", "audio/aac"),
    )

    assert result.backend == "local_agent"
    assert result.target_id == "local-agy-media-gemini-3_7-flash"
    assert len(driver.calls) == 1
    assert result.route_decision["candidates"][0]["group_id"] == "agy-capable"


def test_agent_preferred_media_uploads_only_after_agy_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "clip.aac"
    source.write_bytes(b"audio")
    local_ref = UploadedFileRef(
        "", source.name, "audio/aac", local_path=str(source)
    )
    driver = _FakeAgentDriver(
        failure=LocalAgentUnavailableError("agy unavailable")
    )
    settings = ExecutionSettings(policy_id="agent-text-preferred")
    uploaded = []

    def fake_upload(path, *, api_key=None):
        uploaded.append((Path(path), api_key))
        return UploadedFileRef(
            "files/remote-audio", source.name, "audio/aac", local_path=str(source)
        )

    sent = []

    def fake_chat_complete(messages, **_kwargs):
        sent.append(messages)
        return {"choices": [{"message": {"content": "api-ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.client.upload_gemini_file", fake_upload)
    monkeypatch.setattr("finesub.llm.client._first_gemini_api_key", lambda tier: f"key-{tier}")
    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        router=ModelRouter(policy_id=settings.policy_id),
        execution_settings=settings,
        local_agent_driver=driver,
        rate_limiter=ModelRateLimiter(enabled=False),
        max_retries=0,
    )

    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "audio"}],
        file_ref=local_ref,
        task_group="correction-mm",
    )

    assert result.backend == "gemini_rest"
    assert result.content == "api-ok"
    assert uploaded == [(source, "key-GEMINI_FREE")]
    assert sent[0][-1]["content"][-1]["file"]["file_id"] == "files/remote-audio"


def test_local_drivers_are_cached_per_tier_and_model() -> None:
    """A factory has to see the tier too.

    Otherwise an injected or future dynamic factory cannot honour the rule the
    default registry does -- and two tiers sharing one model id would collapse
    onto whichever driver was built first.
    """

    created = []

    def factory(provider_tier: str, model: str):
        created.append((provider_tier, model))
        return _FakeAgentDriver(model=model)

    settings = ExecutionSettings(policy_id="agent-only")
    client = RoleClient(
        router=ModelRouter(policy_id=settings.policy_id),
        execution_settings=settings,
        local_agent_driver_factory=factory,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    first = client._local_driver_for_model("model-a", "LOCAL_CODEX")
    assert client._local_driver_for_model("model-a", "LOCAL_CODEX") is first
    assert client._local_driver_for_model("model-b", "LOCAL_CODEX") is not first
    # Same model id, other vendor -> its own driver, not the cached one.
    assert client._local_driver_for_model("model-a", "LOCAL_CLAUDE") is not first
    # Tier spelling is normalised before it reaches the cache or the factory.
    assert client._local_driver_for_model("model-a", " local_codex ") is first
    assert created == [
        ("LOCAL_CODEX", "model-a"),
        ("LOCAL_CODEX", "model-b"),
        ("LOCAL_CLAUDE", "model-a"),
    ]
