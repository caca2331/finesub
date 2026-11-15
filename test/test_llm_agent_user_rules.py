"""The opening sentence that sets aside the user's injected rule files."""

from __future__ import annotations

from pathlib import Path

from finesub.llm.agent import local_agent
from finesub.llm.agent.local_agent import (
    AGENT_TASK_PROMPT_AGY_MEDIA,
    AGENT_TASK_PROMPT_READABLE_CAPSULE,
    AGENT_TASK_PROMPT_STDIN_ONLY,
    AGENT_TASK_PROMPT_TOOL_SESSION,
    AGENT_TASK_PROMPT_TOOL_SESSION_OPENING,
    AGENT_USER_RULES_NOTICE,
    DshDriverConfig,
    local_agent_execution_profiles,
)

TEMPLATE = (
    Path(local_agent.__file__).resolve().parents[1]
    / "prompt_templates"
    / "fragment_agent_user_rules_v1.md"
)


def test_notice_is_the_template_text() -> None:
    assert AGENT_USER_RULES_NOTICE == TEMPLATE.read_text(encoding="utf-8").strip()
    assert "AGENTS.md" in AGENT_USER_RULES_NOTICE


def test_every_opening_text_leads_with_the_notice() -> None:
    for opening in (
        AGENT_TASK_PROMPT_READABLE_CAPSULE,
        AGENT_TASK_PROMPT_STDIN_ONLY,
        AGENT_TASK_PROMPT_TOOL_SESSION,
        AGENT_TASK_PROMPT_TOOL_SESSION_OPENING,
        AGENT_TASK_PROMPT_AGY_MEDIA,
    ):
        assert opening.startswith(AGENT_USER_RULES_NOTICE + " ")


def test_dsh_denies_the_rule_injection_plugin_in_identity() -> None:
    assert DshDriverConfig().disabled_instruction_plugins == ("agent-instructions",)
    dsh = local_agent_execution_profiles()["LOCAL_DSH"]
    assert dsh["configuration"]["disabled_instruction_plugins"] == ["agent-instructions"]


def test_codex_profile_says_global_instructions_are_injected() -> None:
    codex = local_agent_execution_profiles()["LOCAL_CODEX"]["configuration"]
    assert codex["user_configuration"] == "ignored_except_connection_when_supported"
    assert codex["global_instructions"] == "injected_by_cli"
