from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from finesub import config as app_config
from finesub.llm.routing.execution_policy import execution_identity, load_execution_settings
from finesub.llm.client import RoleClient
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import default_model_routes
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.session_checkpoint import session_input_hash


@pytest.fixture(autouse=True)
def _clear_config_cache():
    app_config.clear_config_cache()
    default_model_routes.cache_clear()
    yield
    app_config.clear_config_cache()
    default_model_routes.cache_clear()


def _config(tmp_path: Path, monkeypatch, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()
    return path


def test_execution_policy_defaults_to_the_gate_that_forbids_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """Safe as a default only because a policy cannot add an agent.

    The packaged `default` preset names none, so out of the box this behaves
    exactly like `api-only`; it stops being a no-op the moment somebody binds
    a group that lists an agent.
    """

    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "absent.toml"))

    settings = load_execution_settings()

    assert settings.policy_id == "agent-text-preferred"
    assert settings.local_agent_allow_unisolated_user_config is False
    assert settings.local_agent_reasoning_effort == ""
    assert execution_identity(settings)["policy_id"] == "agent-text-preferred"
    assert default_model_routes().binds_local_agent() is False


def test_execution_identity_names_driver_toolset_and_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "absent.toml"))

    drivers = execution_identity()["local_agent_drivers"]

    assert drivers["LOCAL_CODEX"]["driver_id"] == "codex"
    assert drivers["LOCAL_CODEX"]["protocol_version"] == "codex-jsonl-v1"
    assert drivers["LOCAL_CODEX"]["sandbox"] == "process_read_only"
    assert drivers["LOCAL_CODEX"]["toolset"]["native"][-1] == "web_search"
    assert len(drivers["LOCAL_CODEX"]["configuration_digest"]) == 64
    assert drivers["LOCAL_CLAUDE"]["driver_id"] == "claude-code"
    assert drivers["LOCAL_CLAUDE"]["toolset"]["completion"] == []
    assert drivers["LOCAL_CLAUDE"]["sandbox"] == "named_tool_allowlist"
    assert drivers["LOCAL_AGY"]["driver_id"] == "agy"
    assert drivers["LOCAL_AGY"]["toolset"]["completion"] == [
        "project_bounded_view_file"
    ]
    assert drivers["LOCAL_AGY"]["sandbox"] == "project_pretool_hook"


def test_agent_policy_builds_bounded_codex_config(tmp_path: Path, monkeypatch) -> None:
    _config(
        tmp_path,
        monkeypatch,
        """[llm]
execution_policy = "agent-text-preferred"
local_agent_timeout_seconds = 120
local_agent_reasoning_effort = "xhigh"
local_agent_service_tier = "fast"
local_agent_allow_unisolated_user_config = true
""",
    )

    settings = load_execution_settings()
    driver = settings.codex_driver_config(model="gpt-5.6-luna")

    assert settings.policy_id == "agent-text-preferred"
    assert driver.timeout_seconds == 120
    assert driver.allow_unisolated_user_config is True
    assert 'service_tier="fast"' in driver.config_overrides
    assert 'model_reasoning_effort="xhigh"' in driver.config_overrides


def test_a_long_agent_wait_is_the_owners_business(tmp_path: Path, monkeypatch) -> None:
    """The one-hour ceiling is gone (2026-08-24).

    It made sense while this was the whole wall clock of a call, where a large
    value could strand a run. It now budgets only the time a task spends with
    nobody on it, so "I will be back in three hours" is a legitimate thing to
    say -- and capping it was what forced the first live test to sit at 3600.
    """

    _config(
        tmp_path,
        monkeypatch,
        """[llm]
execution_policy = "agent-text-preferred"
local_agent_timeout_seconds = 21600
""",
    )

    assert load_execution_settings().local_agent_timeout_seconds == 21600


def test_agent_media_planning_uses_high_resolution_envelope(
    tmp_path: Path, monkeypatch
) -> None:
    """`agent-only` needs a preset whose groups actually name agents.

    The policy is a gate, not a source of models: pointing it at a preset made
    of Gemini targets leaves nothing to call, which is why this pairs it with
    `agy` rather than the default.
    """

    _config(
        tmp_path,
        monkeypatch,
        '[llm]\nexecution_policy = "agent-only"\npreset = "agy"\n',
    )
    from finesub.llm.routing.config import planning_limits_for

    limits = planning_limits_for("correction-mm", "quality")

    assert limits.video_high_resolution is True


@pytest.mark.parametrize(
    "line,match",
    [
        ('execution_policy = "surprise"', "execution_policy"),
        ("local_agent_timeout_seconds = 1", "at least 10"),
        ('local_agent_service_tier = "default"', "service_tier"),
        ('local_agent_reasoning_effort = "max"', "reasoning_effort"),
        ('local_agent_allow_unisolated_user_config = "yes"', "true/false"),
    ],
)
def test_invalid_execution_settings_fail_closed(
    tmp_path: Path, monkeypatch, line: str, match: str
) -> None:
    _config(tmp_path, monkeypatch, f"[llm]\n{line}\n")

    with pytest.raises(ValueError, match=match):
        load_execution_settings()


def test_policy_change_invalidates_session_resume_key(tmp_path: Path, monkeypatch) -> None:
    path = _config(
        tmp_path,
        monkeypatch,
        '[llm]\nexecution_policy = "api-only"\n',
    )
    messages = [{"role": "user", "content": "same"}]
    api_hash = session_input_hash(messages, prompt_version="v1")
    path.write_text(
        '[llm]\nexecution_policy = "agent-text-preferred"\n', encoding="utf-8"
    )
    app_config.clear_config_cache()

    agent_hash = session_input_hash(messages, prompt_version="v1")

    assert api_hash != agent_hash


def test_client_identity_uses_its_injected_router_catalog() -> None:
    routes = replace(
        default_model_routes(), routing_identity_digest="custom-route-digest"
    )
    router = ModelRouter(routes=routes, policy_id="api-only")
    client = RoleClient(
        router=router,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    assert (
        client.execution_identity["routing_identity_digest"] == "custom-route-digest"
    )
    # The advisory digest never enters the resume-key identity.
    assert "advisory_digest" not in client.execution_identity
