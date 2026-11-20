"""The user's global rule files: kept out where each CLI allows, and the
notice that sets them aside sent exactly where they still come in.

One answer per call (`_global_rules_injected`) drives both the notice and the
audit's `global_instructions`; these tests hold the two together per driver
and per fallback. Codex's homes are tested in `test_llm_agent_codex_connection`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent import local_agent
from finesub.llm.agent.local_agent import (
    AGENT_TASK_PROMPT_AGY_CONTINUATION,
    AGENT_TASK_PROMPT_AGY_MEDIA,
    AGENT_TASK_PROMPT_CONTINUATION,
    AGENT_TASK_PROMPT_READABLE_CAPSULE,
    AGENT_TASK_PROMPT_STDIN_ONLY,
    AGENT_TASK_PROMPT_TOOL_SESSION,
    AGENT_TASK_PROMPT_TOOL_SESSION_OPENING,
    AGENT_USER_RULES_NOTICE,
    AGY_AGENT_DOCUMENT,
    AGY_NATIVE_AGENT_DOCUMENT,
    AGY_TOOL_AGENT_DOCUMENT,
    AGY_TOOL_NATIVE_AGENT_DOCUMENT,
    CODEBUDDY_RULES_SWITCH_VERSION,
    DriverProbe,
    DshDriverConfig,
    DshLocalAgentDriver,
    LocalAgentPolicyViolationError,
    local_agent_execution_profiles,
    rules_audit,
    with_rules_notice,
)

TEMPLATE = (
    Path(local_agent.__file__).resolve().parents[1]
    / "prompt_templates"
    / "fragment_agent_user_rules_v1.md"
)


def test_notice_is_the_template_text() -> None:
    assert AGENT_USER_RULES_NOTICE == TEMPLATE.read_text(encoding="utf-8").strip()
    assert "AGENTS.md" in AGENT_USER_RULES_NOTICE


def test_no_opening_text_carries_the_notice_itself() -> None:
    """Each driver adds it per call; a constant that carried it would send it
    to calls whose rule files are kept out."""

    for opening in (
        AGENT_TASK_PROMPT_READABLE_CAPSULE,
        AGENT_TASK_PROMPT_STDIN_ONLY,
        AGENT_TASK_PROMPT_TOOL_SESSION,
        AGENT_TASK_PROMPT_TOOL_SESSION_OPENING,
        AGENT_TASK_PROMPT_AGY_MEDIA,
        AGENT_TASK_PROMPT_CONTINUATION,
        AGENT_TASK_PROMPT_AGY_CONTINUATION,
    ):
        assert AGENT_USER_RULES_NOTICE not in opening


def test_the_notice_and_the_audit_read_one_answer() -> None:
    assert with_rules_notice("task", True) == f"{AGENT_USER_RULES_NOTICE} task"
    assert with_rules_notice("task", True, separator="\n\n") == f"{AGENT_USER_RULES_NOTICE}\n\ntask"
    assert with_rules_notice("task", False) == "task"
    assert (rules_audit(True), rules_audit(False)) == ("injected", "excluded")


# -- Claude Code ---------------------------------------------------------------


@pytest.mark.parametrize("setting_sources", [True, False])
def test_claude_notice_follows_setting_sources(tmp_path: Path, monkeypatch, setting_sources) -> None:
    from .support.local_agent import _claude_driver

    driver = _claude_driver(tmp_path, monkeypatch)
    driver._resolved_command = ("claude",)
    probe = DriverProbe(available=True, no_user_config=True, no_user_rules=setting_sources)
    injected = not setting_sources

    for continuation_of in ("", "thread-1"):
        capsule = SimpleNamespace(root=tmp_path, continuation_of=continuation_of)
        prompt = driver._argv(capsule, native_search=False, probe=probe)[-1]
        assert prompt.startswith(AGENT_USER_RULES_NOTICE) is injected

    assert driver._isolation_metadata(probe, "")["global_instructions"] == rules_audit(injected)


# -- CodeBuddy -------------------------------------------------------------------


@pytest.mark.parametrize(
    "version,injected",
    [(CODEBUDDY_RULES_SWITCH_VERSION, False), ("2.190.0", False), ("2.164.0", True), ("garbled", True)],
)
def test_codebuddy_notice_follows_the_version_that_keeps_codebuddy_md_out(
    tmp_path: Path, monkeypatch, version, injected
) -> None:
    from .test_llm_agent_codebuddy import _driver

    driver = _driver(tmp_path, monkeypatch)
    driver._resolved_command = ("codebuddy",)
    probe = DriverProbe(available=True, version=version)
    capsule = SimpleNamespace(root=tmp_path, continuation_of="")

    prompt = driver._argv(capsule, native_search=False, probe=probe)[-1]

    assert prompt.startswith(AGENT_USER_RULES_NOTICE) is injected
    assert driver._isolation_metadata(probe, "")["global_instructions"] == rules_audit(injected)


@pytest.mark.parametrize("tool_session", [False, True])
def test_codebuddy_drops_its_memory_reminder_on_every_spawn(
    tmp_path: Path, monkeypatch, tool_session
) -> None:
    from .test_llm_agent_codebuddy import _driver

    server = {"command": "python", "args": [], "env": {}} if tool_session else None
    env = _driver(tmp_path, monkeypatch)._spawn_environment(mcp_server=server)

    assert env["CODEBUDDY_DISABLE_SYSTEM_REMINDER_MD"] == "1"


# -- agy ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [AGY_AGENT_DOCUMENT, AGY_NATIVE_AGENT_DOCUMENT, AGY_TOOL_AGENT_DOCUMENT, AGY_TOOL_NATIVE_AGENT_DOCUMENT],
)
def test_every_agy_agent_document_excludes_the_default_components(document: str) -> None:
    """The key that keeps `~/.gemini/GEMINI.md` out; `inheritCustomizations`
    is the one that would switch the project hook off, and stays unset."""

    frontmatter = document.split("---")[1]
    assert "excludeDefaultComponents: true" in frontmatter.splitlines()
    assert "inheritCustomizations" not in frontmatter


def test_the_hook_digest_covers_the_agent_documents(tmp_path: Path) -> None:
    from .support.local_agent import _agy_driver

    driver = _agy_driver(tmp_path)
    paths = driver._write_project_resources(tmp_path / "project")
    before = driver._hook_digest(paths)
    paths["agent"].write_text(AGY_AGENT_DOCUMENT.replace("excludeDefaultComponents: true\n", ""), encoding="utf-8")

    assert driver._hook_digest(paths) != before


def _agy_capsule_prompt(driver, tmp_path: Path) -> str:
    from .support.local_agent import _agy_task_parts

    capsule = SimpleNamespace(
        root=tmp_path / "domain" / "episode",
        messages_path=tmp_path / "domain" / "episode" / "task.json",
        continuation_of="",
    )
    _agy_task_parts(capsule.root)
    argv = driver._argv(capsule, native_search=False, probe=driver.probe())
    return argv[argv.index("--print") + 1]


def test_agy_capsule_notice_only_once_its_agent_document_was_dropped(tmp_path: Path) -> None:
    """A capsule project runs on when agy drops its agent document
    (owner 2026-10-06), and the default agent brings GEMINI.md back."""

    from .support.local_agent import _agy_driver

    driver = _agy_driver(tmp_path)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")
    driver._grant_permissions = lambda *a, **k: None
    probe = driver.probe()

    assert not _agy_capsule_prompt(driver, tmp_path).startswith(AGENT_USER_RULES_NOTICE)
    assert driver._isolation_metadata(probe, "")["global_instructions"] == "excluded"

    driver._dropped_agent_projects.add(str((tmp_path / "domain").resolve()))

    assert _agy_capsule_prompt(driver, tmp_path).startswith(AGENT_USER_RULES_NOTICE)
    assert driver._isolation_metadata(probe, "")["global_instructions"] == "injected"


def test_agy_starts_each_call_with_no_rules_answer_left_from_the_last(tmp_path: Path) -> None:
    """A call that fails before its argv must not be audited with the last
    call's answer (`_begin_call`)."""

    from .support.local_agent import _agy_driver

    driver = _agy_driver(tmp_path)
    driver._rules_local.injected = True

    driver._begin_call()

    assert driver._isolation_metadata(driver.probe(), "")["global_instructions"] == "excluded"

def test_agy_project_check_records_a_dropped_capsule_document_and_its_recovery(tmp_path: Path) -> None:
    from .support.local_agent import _agy_driver

    driver = _agy_driver(tmp_path)
    project = tmp_path / "project"
    paths = driver._write_project_resources(project)
    driver._hooks_command = lambda *a, **k: SimpleNamespace(stdout="")
    driver._verify_hook_output = lambda *a: None
    verdict: dict[str, Exception | None] = {"error": LocalAgentPolicyViolationError("agent ignored")}

    def resolved(_log: str, _name: str) -> None:
        if verdict["error"] is not None:
            raise verdict["error"]

    driver._verify_agent_resolved = resolved

    driver._check_project(project, "project-id", paths)
    assert str(project.resolve()) in driver._dropped_agent_projects

    verdict["error"] = None
    driver._check_project(project, "project-id", paths)
    assert str(project.resolve()) not in driver._dropped_agent_projects

    # A tool project fails closed instead: it never runs on the default agent.
    verdict["error"] = LocalAgentPolicyViolationError("agent ignored")
    with pytest.raises(LocalAgentPolicyViolationError):
        driver._check_project(project, "project-id", paths, tool=True)


# -- dsh -----------------------------------------------------------------------------


def test_dsh_denies_the_rule_injection_plugin_in_identity() -> None:
    assert DshDriverConfig().disabled_instruction_plugins == ("agent-instructions",)
    dsh = local_agent_execution_profiles()["LOCAL_DSH"]
    assert dsh["configuration"]["disabled_instruction_plugins"] == ["agent-instructions"]


@pytest.mark.parametrize("plugins,expected", [(("agent-instructions",), "excluded"), ((), "injected")])
def test_dsh_audit_follows_the_rule_plugin(plugins, expected) -> None:
    driver = DshLocalAgentDriver(DshDriverConfig(disabled_instruction_plugins=plugins))
    driver._route = lambda: ("other", "")

    assert driver._isolation_metadata(DriverProbe(available=True), "")["global_instructions"] == expected

    driver._resolved_command = ("dsh",)
    driver._bootstrap_text = lambda _capsule: "BOOTSTRAP"
    driver._patch_entries = lambda **_kwargs: []
    driver._write_patch = lambda _capsule, _entries: Path("patch.json")
    argv = driver._argv(
        SimpleNamespace(root=Path("episode")), native_search=False,
        probe=DriverProbe(available=True), mcp_server={"name": "finesub"},
    )
    assert argv[-1].startswith(AGENT_USER_RULES_NOTICE) is (expected == "injected")


# -- Codex identity ------------------------------------------------------------------


def test_codex_profile_names_what_isolation_keeps_out_not_this_machine() -> None:
    codex = local_agent_execution_profiles()["LOCAL_CODEX"]
    configuration = codex["configuration"]
    assert configuration["user_configuration"] == "ignored_except_connection_when_supported"
    assert configuration["global_instructions"] == "excluded_when_isolated"
    assert configuration["disabled_features"] == sorted(local_agent.CODEX_DISABLED_FEATURES)
    assert codex["toolset"] == {"completion": [], "native": ["web_search"]}
