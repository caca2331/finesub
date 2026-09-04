"""dsh plugin drift: what a deny list cannot see, and how it gets told.

`DshDriverConfig.disabled_tool_plugins` switches plugins off by id, so it only
bounds a bundle somebody has already read -- anything installed into
`$DSH_HOME`, or added by a dsh upgrade, is enabled and unexamined. dsh will
describe its own composition, and these cover the part that reads it.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent import local_agent
from finesub.llm.agent.local_agent import (
    DriverProbe,
    DshDriverConfig,
    DshLocalAgentDriver,
)


@pytest.fixture(autouse=True)
def _fresh_state():
    local_agent._DSH_COMPOSITION_CACHE.clear()
    local_agent._READINESS_REPORTED.clear()
    yield
    local_agent._DSH_COMPOSITION_CACHE.clear()
    local_agent._READINESS_REPORTED.clear()


def _driver() -> DshLocalAgentDriver:
    driver = DshLocalAgentDriver(
        DshDriverConfig(model="deepseek-official/deepseek-v4-flash")
    )
    driver._resolved_command = ("node.exe", "bin.js")
    return driver


DUMP_SAMPLE = """\
# == @deepseek-ai/dsh-base
- id: timer
  name: '@deepseek-ai/cordis-plugin-timer'
- id: session-persistence-jsonl
  name: '@deepseek-ai/dsh-session-persistence-jsonl'
  config:
    root: !!js dshHomePath('sessions')
- id: tool-fs
  name: '@deepseek-ai/dsh-tool-fs'
"""


def test_the_dump_is_read_without_a_yaml_parser() -> None:
    """Its values carry `!!js` tags, which a YAML loader would reject outright."""

    assert local_agent._DSH_ENTRY_ID_RE.findall(DUMP_SAMPLE) == [
        "timer",
        "session-persistence-jsonl",
        "tool-fs",
    ]


def _stub_dumps(monkeypatch, composed, stock=None) -> list[str]:
    calls: list[str] = []

    def fake(command, profile, flag):
        calls.append(flag)
        return composed if flag == "--dump-config" else stock

    monkeypatch.setattr(local_agent, "_dsh_dump_plugin_ids", fake)
    return calls


def test_a_plugin_the_driver_has_never_seen_is_reported(monkeypatch, reported) -> None:
    known = DshDriverConfig().expected_plugin_ids
    _stub_dumps(monkeypatch, composed=(*known, "tool-http"), stock=known)

    _driver().check_environment()

    assert reported.codes() == ["agent-cli-plugin-drift"]
    assert "tool-http" in reported.joined()
    assert "$DSH_HOME" in reported.joined()


def test_drift_already_in_the_stock_bundle_is_not_blamed_on_the_user(
    monkeypatch, reported
) -> None:
    """A dsh upgrade adding a plugin reads differently from the user adding one."""

    with_extra = (*DshDriverConfig().expected_plugin_ids, "tool-http")
    _stub_dumps(monkeypatch, composed=with_extra, stock=with_extra)

    _driver().check_environment()

    assert reported.codes() == ["agent-cli-plugin-drift"]
    assert "$DSH_HOME" not in reported.joined()
    assert "dsh upgrade" in reported.joined()


def test_the_known_bundle_says_nothing_and_asks_dsh_once(monkeypatch, reported) -> None:
    """Composition is read once per process, however often readiness is asked."""

    known = DshDriverConfig().expected_plugin_ids
    calls = _stub_dumps(monkeypatch, composed=known, stock=known)

    driver = _driver()
    driver.check_environment()
    driver.check_environment()

    assert reported.codes() == []
    assert calls == ["--dump-config", "--dump-default-config"]


def test_a_dsh_that_will_not_describe_itself_is_unusable(monkeypatch, reported) -> None:
    """With the policy on it cannot serve one call, and the composition is
    cached for the process -- so it leaves the chain instead of spending a
    `backend_unavailable` per window to rediscover that."""

    _stub_dumps(monkeypatch, composed=None)
    driver = _driver()
    monkeypatch.setattr(
        driver,
        "probe",
        lambda: DriverProbe(available=True, version=DshDriverConfig().min_version),
    )
    monkeypatch.setattr(driver, "meets_requirements", lambda *a, **k: True)

    ready, detail = local_agent.driver_readiness(driver)

    assert ready is False
    assert "unusable" in detail and "deny_unknown_plugins=False" in detail
    assert reported.codes() == ["agent-cli-unusable"]


def test_without_the_policy_a_missing_inventory_only_warns(
    monkeypatch, reported
) -> None:
    """Turning the policy off says "run without the inventory", so it runs."""

    _stub_dumps(monkeypatch, composed=None)
    driver = DshLocalAgentDriver(
        DshDriverConfig(
            model="deepseek-official/deepseek-v4-flash", deny_unknown_plugins=False
        )
    )
    driver._resolved_command = ("node.exe", "bin.js")

    assert driver.check_environment() == ""
    assert reported.codes() == ["agent-cli-plugin-inventory-unavailable"]


def test_an_unparseable_dump_is_a_failure_not_an_empty_bundle(monkeypatch) -> None:
    """Zero ids means the output stopped looking the way the parser expects.

    Reading it as an empty bundle would make every plugin "known" and switch
    the whole policy off without a word.
    """

    monkeypatch.setattr(
        local_agent.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="# nothing here\n"),
    )

    assert (
        local_agent._dsh_dump_plugin_ids(("dsh",), "headless", "--dump-config") is None
    )


def test_drift_of_unknown_origin_is_not_blamed_on_the_user(
    monkeypatch, reported
) -> None:
    """The stock dump can fail on its own; "could not tell" is not "you did it"."""

    _stub_dumps(
        monkeypatch,
        composed=(*DshDriverConfig().expected_plugin_ids, "tool-http"),
        stock=None,
    )

    _driver().check_environment()

    assert reported.codes() == ["agent-cli-plugin-drift"]
    assert "cannot tell" in reported.joined()


def test_the_check_never_gates_the_driver(monkeypatch, reported) -> None:
    """A raising advisory must not remove a working CLI from the chain."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("dump exploded")

    monkeypatch.setattr(local_agent, "_dsh_plugin_drift", boom)
    driver = _driver()
    # A version the pin is happy with, so the only warning is the one under test.
    monkeypatch.setattr(
        driver,
        "probe",
        lambda: local_agent.DriverProbe(
            available=True, version=DshDriverConfig().min_version
        ),
    )
    monkeypatch.setattr(driver, "meets_requirements", lambda *a, **k: True)

    assert local_agent.driver_readiness(driver) == (True, "")
    assert reported.codes() == ["agent-cli-environment-uncheckable"]


def test_every_disabled_plugin_is_one_the_bundle_snapshot_knows() -> None:
    """Denying an id absent from the snapshot means one of the two went stale."""

    config = DshDriverConfig()
    assert set(config.disabled_tool_plugins) - set(config.expected_plugin_ids) == set()


def test_the_write_capable_editor_is_removed_rather_than_merely_unapproved() -> None:
    """Declared entitlement is `tool-fs_read`; read-only mode was the only guard."""

    assert "tool-str-replace-editor" in DshDriverConfig().disabled_tool_plugins


def test_the_plugins_measured_to_add_nothing_this_driver_needs_are_off() -> None:
    """Each was removed on evidence: the tool it contributed, and a call that
    still answered without it (v4f, 2026-09-02).

    Kept deliberately: `commands`/`command-*` contribute no model-facing tool
    (and compaction has a job on a long window), and `plan-mode` was left
    alone because the agent loop's use of it is unmeasured.
    """

    disabled = set(DshDriverConfig().disabled_tool_plugins)
    assert {
        "tool-fs-search",  # glob, grep
        "tool-subagent-control",  # interrupt_agent
        "tool-subagent-list-agents",  # list_agents
        "tool-subagent-report",  # send_message
    } <= disabled
    assert not disabled & {"commands", "plan-mode", "tools", "fs-sandbox", "tool-fs"}


# --- blocking, not just noticing ---------------------------------------------


def _patch_disables(tmp_path: Path, **config_overrides) -> list[str]:
    """The ids one call's patch overlay switches off."""

    messages = tmp_path / "input" / "messages.json"
    messages.parent.mkdir(parents=True, exist_ok=True)
    messages.write_text(
        json.dumps([{"role": "user", "content": "TASK"}]), encoding="utf-8"
    )
    capsule = SimpleNamespace(
        root=tmp_path, episode_id="ep", messages_path=messages
    )
    driver = DshLocalAgentDriver(
        DshDriverConfig(
            model="deepseek-official/deepseek-v4-flash", **config_overrides
        )
    )
    driver._resolved_command = ("node.exe", "bin.js")
    argv = driver._argv(
        capsule,
        native_search=False,
        probe=DriverProbe(available=True),
        mcp_server={"command": "python", "args": ["-m", "x"], "env": {}},
    )
    entries = json.loads(
        Path(argv[argv.index("--patch") + 1]).read_text(encoding="utf-8")
    )
    return [entry["id"] for entry in entries if entry.get("disabled")]


def _drift(monkeypatch, unknown: tuple[str, ...]) -> None:
    monkeypatch.setattr(
        local_agent,
        "_dsh_plugin_drift",
        lambda *_a, **_k: local_agent.DshPluginDrift(
            unknown=unknown, from_user_layer=unknown
        ),
    )


def test_an_unvetted_plugin_is_switched_off_for_the_call(
    monkeypatch, tmp_path
) -> None:
    """Measured against a real plugin dropped into `$DSH_HOME`: composed, and
    offered to the model as `ask_user_question` until this denies it."""

    _drift(monkeypatch, ("tool-ask-user",))

    assert "tool-ask-user" in _patch_disables(tmp_path)


def test_the_policy_can_be_turned_off(monkeypatch, tmp_path) -> None:
    """The escape hatch for a bundle whose new plugin the runtime needs."""

    _drift(monkeypatch, ("tool-ask-user",))

    disabled = _patch_disables(tmp_path, deny_unknown_plugins=False)
    assert "tool-ask-user" not in disabled
    assert "tool-bash" in disabled  # the explicit list still applies


def test_an_unvetted_id_is_not_denied_twice(monkeypatch, tmp_path) -> None:
    _drift(monkeypatch, ("tool-bash",))

    disabled = _patch_disables(tmp_path)
    assert disabled.count("tool-bash") == 1


def test_a_call_is_refused_when_the_inventory_cannot_be_read(
    monkeypatch, tmp_path
) -> None:
    """Fail CLOSED: no inventory, no way to honour "only vetted plugins run".

    Going ahead would quietly ship the one outcome the policy exists to
    prevent. Unavailable rather than a policy violation, so the chain moves to
    another target instead of failing the task.
    """

    monkeypatch.setattr(local_agent, "_dsh_plugin_drift", lambda *_a, **_k: None)

    with pytest.raises(local_agent.LocalAgentUnavailableError) as excinfo:
        _patch_disables(tmp_path)
    assert "deny_unknown_plugins=False" in str(excinfo.value)


def test_without_the_policy_a_silent_dsh_still_gets_the_explicit_list(
    monkeypatch, tmp_path
) -> None:
    """Turning the policy off is what "run without the inventory" means."""

    monkeypatch.setattr(local_agent, "_dsh_plugin_drift", lambda *_a, **_k: None)

    disabled = _patch_disables(tmp_path, deny_unknown_plugins=False)
    assert set(DshDriverConfig().disabled_tool_plugins) <= set(disabled)


def test_the_snapshot_can_be_retaken_as_a_paste_ready_literal(monkeypatch) -> None:
    """`deny_unknown_plugins` defaults on, so an upgrade needs a way back.

    Includes a hyphenated id on purpose: the default wrapping splits
    `workflow-worker-thread` across lines and emits a literal that names a
    plugin nobody has.
    """

    composed = ("zeta", "alpha", "workflow-worker-thread", "tool-fs")
    _stub_dumps(monkeypatch, composed=composed, stock=composed)
    monkeypatch.setattr(
        local_agent, "_resolve_shell_free_command", lambda _c: ("node.exe", "bin.js")
    )

    rendered = local_agent.format_dsh_expected_plugin_ids()

    assert ast.literal_eval(rendered[rendered.index("(") :]) == tuple(sorted(composed))
    assert all(len(line) <= 79 for line in rendered.splitlines())


def test_retaking_the_snapshot_says_so_when_dsh_will_not_answer(monkeypatch) -> None:
    _stub_dumps(monkeypatch, composed=None)
    monkeypatch.setattr(
        local_agent, "_resolve_shell_free_command", lambda _c: ("node.exe", "bin.js")
    )

    with pytest.raises(local_agent.LocalAgentUnavailableError):
        local_agent.format_dsh_expected_plugin_ids()


def test_dsh_home_reaches_the_cli_like_codex_home_does() -> None:
    """dsh declares `user_configuration: inherited` -- stripping the pointer to
    that configuration made every call read the default `~/.dsh` instead, and
    the drift warning name a directory that was not the user's."""

    assert {"CODEX_HOME", "DSH_HOME"} <= local_agent.ENV_ALLOWLIST
    assert not {"DSH_HOME"} & local_agent.SENSITIVE_ENV_NAMES


def test_identity_records_the_policy_not_this_machines_resolved_set() -> None:
    """The resolved set is a fact about this machine, like a probe result."""

    dsh = local_agent.local_agent_execution_profiles()["LOCAL_DSH"]
    configuration = dsh["configuration"]

    assert configuration["deny_unknown_plugins"] is True
    assert configuration["disabled_tool_plugins"] == sorted(
        DshDriverConfig().disabled_tool_plugins
    )
    assert configuration["expected_plugins"]["count"] == len(
        DshDriverConfig().expected_plugin_ids
    )
    # The resolved set -- what this machine happened to disable today -- is a
    # machine fact and stays out, like probe results.
    assert "unknown_plugins" not in configuration


def test_retaking_the_snapshot_changes_execution_identity(monkeypatch) -> None:
    """A re-taken snapshot can turn a denied plugin into an allowed one, which
    is a toolset change: an uncommitted checkpoint must not resume under it."""

    before = local_agent.local_agent_execution_profiles()["LOCAL_DSH"]
    monkeypatch.setattr(
        local_agent,
        "_DSH_IDENTITY",
        dataclasses.replace(
            local_agent._DSH_IDENTITY,
            expected_plugin_ids=(
                *DshDriverConfig().expected_plugin_ids,
                "tool-newly-vetted",
            ),
        ),
    )
    after = local_agent.local_agent_execution_profiles()["LOCAL_DSH"]

    assert (
        before["configuration"]["expected_plugins"]
        != after["configuration"]["expected_plugins"]
    )
