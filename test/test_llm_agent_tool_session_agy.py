"""agy on the tool protocol: slot projects, per-call server identity,
project-record grants, and the entitled `call_mcp_tool` events."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent.local_agent import (
    AGY_MCP_CALL_TOOL,
    AGY_TOOL_AGENT_NAME,
    AGY_TOOL_GUARD_SCRIPT,
    AgyDriverConfig,
    AgyLocalAgentDriver,
    LocalAgentTransientError,
    LocalAgentUnavailableError,
    _normalize_agy_events,
)


def _guard(tool_call: dict) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", AGY_TOOL_GUARD_SCRIPT],
        input=json.dumps({"toolCall": tool_call}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_the_tool_guard_allows_only_finesub_mcp_calls() -> None:
    allowed = _guard({"name": AGY_MCP_CALL_TOOL, "args": {"ServerName": "finesub", "ToolName": "submit"}})
    assert allowed["decision"] == "allow"
    for decoy in (
        {"name": AGY_MCP_CALL_TOOL, "args": {"ServerName": "other", "ToolName": "submit"}},
        {"name": "view_file", "args": {"AbsolutePath": "C:/x"}},
        {"name": "run_command", "args": {"CommandLine": "dir"}},
        {"name": "write_to_file", "args": {}},
    ):
        assert _guard(decoy)["decision"] == "deny", decoy


def test_tool_argv_binds_the_slot_project_and_grants_the_tools(tmp_path, monkeypatch) -> None:
    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",), max_parallel=2))
    driver._resolved_command = ("agy",)
    records = tmp_path / "records"
    records.mkdir()
    registered: list[Path] = []

    def fake_ensure(project_root: Path, *, native: bool = False, tool: bool = False):
        assert tool and not native
        registered.append(project_root)
        project_id = "11111111-2222-3333-4444-555555555555"
        record = records / f"{project_id}.json"
        if not record.exists():
            record.write_text(json.dumps({"id": project_id, "permissionGrants": {
                "permissionGrants": {"allow": ["read_url(example.com)"]}}}), encoding="utf-8")
        return project_id, "digest"

    monkeypatch.setattr(driver, "_ensure_project", fake_ensure)
    monkeypatch.setattr(AgyLocalAgentDriver, "agy_project_records_dir", staticmethod(lambda: records))
    domain = tmp_path / "domain"
    episode = domain / "episode-1"
    (episode / "input").mkdir(parents=True)
    messages_path = episode / "input" / "messages.json"
    messages_path.write_text(json.dumps([{"role": "user", "content": "call next_task"}]), encoding="utf-8")
    capsule = SimpleNamespace(root=episode, episode_id="episode-1", messages_path=messages_path)
    server = {
        "command": "python",
        "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
        "env": {"FINESUB_MCP_SESSION": "s1", "FINESUB_MCP_ROOT": "R"},
        "tools": ["next_task", "submit"],
        "view_roots": [str(domain / "assignments" / "session-x")],
    }

    driver._tool_slot_local.slot = 1
    argv = driver._argv(capsule, native_search=False, probe=None, mcp_server=server)

    assert registered == [domain / ".finesub-tool-1"]
    roots = json.loads((domain / ".finesub-tool-1" / ".agents" / "view_roots.json").read_text(encoding="utf-8"))
    assert roots == [str((domain / "assignments" / "session-x").resolve())]
    config = json.loads((domain / ".finesub-tool-1" / ".agents" / "mcp_config.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["finesub"]["env"]["FINESUB_MCP_SESSION"] == "s1"
    record = json.loads((records / "11111111-2222-3333-4444-555555555555.json").read_text(encoding="utf-8"))
    assert record["permissionGrants"]["permissionGrants"]["allow"] == [
        "read_url(example.com)", "mcp(finesub/next_task)", "mcp(finesub/submit)",
    ]
    assert argv[argv.index("--print") + 1] == "call next_task"
    assert argv[argv.index("--agent") + 1] == AGY_TOOL_AGENT_NAME
    assert argv[argv.index("--project") + 1] == "11111111-2222-3333-4444-555555555555"

    # A second call rewrites the identity in place and adds no duplicate grants.
    server["env"]["FINESUB_MCP_SESSION"] = "s2"
    driver._argv(capsule, native_search=False, probe=None, mcp_server=server)
    config = json.loads((domain / ".finesub-tool-1" / ".agents" / "mcp_config.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["finesub"]["env"]["FINESUB_MCP_SESSION"] == "s2"
    record = json.loads((records / "11111111-2222-3333-4444-555555555555.json").read_text(encoding="utf-8"))
    assert len(record["permissionGrants"]["permissionGrants"]["allow"]) == 3


def test_a_missing_project_record_fails_closed(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.local_agent import LocalAgentUnavailableError

    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))
    monkeypatch.setattr(AgyLocalAgentDriver, "agy_project_records_dir", staticmethod(lambda: tmp_path))
    with pytest.raises(LocalAgentUnavailableError):
        driver._grant_permissions("no-such-project", mcp_tools=["submit"])


def test_a_native_tool_call_grants_the_fetch_permission(tmp_path, monkeypatch) -> None:
    """agy gates `read_url_content` behind a permission, not just the hook.

    Regression for 2026-08-30: headless mode cannot prompt, so an ungranted
    fetch is auto-denied and the CLI ends the turn with no assistant message
    -- the search that preceded it is lost with the call.
    """

    records = tmp_path / "records"
    records.mkdir()
    project_id = "11111111-2222-3333-4444-555555555555"
    (records / f"{project_id}.json").write_text(
        json.dumps({"id": project_id, "permissionGrants": {}}), encoding="utf-8"
    )
    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))
    monkeypatch.setattr(
        AgyLocalAgentDriver, "agy_project_records_dir", staticmethod(lambda: records)
    )

    driver._grant_permissions(project_id, mcp_tools=["submit"])
    allow = json.loads((records / f"{project_id}.json").read_text(encoding="utf-8"))[
        "permissionGrants"
    ]["permissionGrants"]["allow"]
    assert "read_url(*)" not in allow

    from finesub.llm.agent.local_agent import AGY_NATIVE_PERMISSION_RULES

    driver._grant_permissions(
        project_id, mcp_tools=["submit"], rules=AGY_NATIVE_PERMISSION_RULES
    )
    allow = json.loads((records / f"{project_id}.json").read_text(encoding="utf-8"))[
        "permissionGrants"
    ]["permissionGrants"]["allow"]
    assert "read_url(*)" in allow
    # Idempotent: a second call must not duplicate the rule.
    driver._grant_permissions(
        project_id, mcp_tools=["submit"], rules=AGY_NATIVE_PERMISSION_RULES
    )
    allow = json.loads((records / f"{project_id}.json").read_text(encoding="utf-8"))[
        "permissionGrants"
    ]["permissionGrants"]["allow"]
    assert allow.count("read_url(*)") == 1


def test_slots_are_bounded_by_max_parallel(tmp_path) -> None:
    driver = AgyLocalAgentDriver(
        AgyDriverConfig(command=("agy",), max_parallel=2, runtime_root=tmp_path)
    )
    first, second = driver._acquire_tool_slot(), driver._acquire_tool_slot()
    assert {first, second} == {0, 1}
    driver._release_tool_slot(first)
    assert driver._acquire_tool_slot() == first
    driver._release_tool_slot(first)
    driver._release_tool_slot(second)


def test_two_drivers_on_one_domain_root_never_share_a_slot(tmp_path) -> None:
    """Plan §1.1: the slot names a directory (`.finesub-tool-<slot>`), so the
    pool is keyed by the domain root, not the driver instance. Two clients'
    drivers counting from 0 each used to hand two tasks the same project --
    one task's CLI wired to the other task's MCP server."""

    config = AgyDriverConfig(command=("agy",), max_parallel=2, runtime_root=tmp_path)
    one = AgyLocalAgentDriver(config)
    other = AgyLocalAgentDriver(config)
    first = one._acquire_tool_slot()
    second = other._acquire_tool_slot()  # while the first driver holds its slot
    assert first != second
    one._release_tool_slot(first)
    other._release_tool_slot(second)

    # A different domain root is a different pool: slot 0 is free there.
    elsewhere = AgyLocalAgentDriver(
        AgyDriverConfig(command=("agy",), max_parallel=2, runtime_root=tmp_path / "other")
    )
    slot = elsewhere._acquire_tool_slot()
    assert slot == 0
    elsewhere._release_tool_slot(slot)


def test_all_models_of_one_vendor_share_the_in_flight_budget(tmp_path) -> None:
    """Plan §1.1 + reviewer 2026-08-30 P1-1: `_in_flight` guards the
    machine+subscription, so every driver instance of one VENDOR draws on
    ONE pool -- whatever model, timeout or effort its config names. Keying
    on a config digest handed each model its own max_parallel."""

    from finesub.llm.agent.local_agent import AgentDriverConfig, LocalAgentDriver

    one = AgyLocalAgentDriver(
        AgyDriverConfig(command=("agy",), model="flash-3.7", runtime_root=tmp_path)
    )
    other = AgyLocalAgentDriver(
        AgyDriverConfig(command=("agy",), model="opus-4.6", runtime_root=tmp_path)
    )
    assert one._in_flight is other._in_flight
    # A different vendor is a different subscription: a different budget.
    generic = LocalAgentDriver(
        AgentDriverConfig(command=("x",), runtime_root=tmp_path)
    )
    assert generic._in_flight is not one._in_flight


def test_agy_events_entitle_finesub_mcp_calls_only(tmp_path) -> None:
    def step(tool, server, state="DONE"):
        return {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "state": state,
                "tool_name": AGY_MCP_CALL_TOOL,
                "tool_info": {"parameters": {"ServerName": server, "ToolName": tool}},
            },
        }

    rows = [
        {"event": "init", "init": {"tools": ["view_file", AGY_MCP_CALL_TOOL], "permission_mode": "x"}},
        step("submit", "finesub", "ACTIVE"),
        step("submit", "finesub"),
        step("submit", "other"),
        {"event": "result", "result": {"status": "SUCCESS", "response": "submitted"}},
    ]
    raw = tmp_path / "events.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    normalized, _usage, violations, content = _normalize_agy_events(
        raw, native_search=False, max_bytes=1_000_000, extra_entitled=frozenset({"submit"})
    )
    assert content == "submitted"
    assert [row["tool"] for row in normalized if row.get("event") == "tool_use"] == ["submit", "submit"]
    assert violations == [f"completion target invoked {AGY_MCP_CALL_TOOL}:other/submit"]


def _guard_with_roots(tmp_path, tool_call: dict, roots: list) -> dict:
    """Run the tool guard from a project layout with a view_roots.json."""

    scripts = tmp_path / ".agents" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "guard_view_file.py").write_text(AGY_TOOL_GUARD_SCRIPT, encoding="utf-8")
    (tmp_path / ".agents" / "view_roots.json").write_text(json.dumps(roots), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(scripts / "guard_view_file.py")],
        input=json.dumps({"toolCall": tool_call}),
        capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def test_the_tool_guard_allows_view_file_only_under_the_listed_roots(tmp_path) -> None:
    root = tmp_path / "assignment"
    (root / "contexts").mkdir(parents=True)
    inside = root / "contexts" / "payload.md"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path / "elsewhere.md"
    outside.write_text("x", encoding="utf-8")

    view = lambda path: {"name": "view_file", "args": {"AbsolutePath": str(path)}}  # noqa: E731
    assert _guard_with_roots(tmp_path, view(inside), [str(root)])["decision"] == "allow"
    assert _guard_with_roots(tmp_path, view(outside), [str(root)])["decision"] == "deny"
    assert _guard_with_roots(tmp_path, view(root / "missing.md"), [str(root)])["decision"] == "deny"
    # No roots listed (a single-task call without block files): nothing is readable.
    assert _guard_with_roots(tmp_path, view(inside), [])["decision"] == "deny"
    # The MCP channel stays open regardless; other native tools stay shut.
    assert _guard_with_roots(
        tmp_path, {"name": AGY_MCP_CALL_TOOL, "args": {"ServerName": "finesub", "ToolName": "submit"}}, []
    )["decision"] == "allow"
    assert _guard_with_roots(tmp_path, {"name": "run_command", "args": {}}, [str(root)])["decision"] == "deny"


HEADLESS_DENIAL = (
    'no output produced — a tool required the "read_file" permission that '
    "headless mode cannot prompt for\n"
)


def _capsule(tmp_path: Path, stderr: str) -> SimpleNamespace:
    path = tmp_path / "stderr.log"
    path.write_text(stderr, encoding="utf-8")
    return SimpleNamespace(episode_id="cap-1", stderr_path=path)


class TestEmptyTurnClassification:
    """A permission agy cannot ask about must not be reported as a flake.

    Its shape is exit 0, an empty answer and one stderr line. Called
    transient, the router drops this target and walks the rest of the chain,
    so what the user finally reads is whatever the *last* link says -- a
    sentence about some other provider's API key, for a failure that was a
    denied file read. The same shape already cost us one investigation for
    `read_url_content` (docs/llm_local_agent_agy.md §6.2), where the missing
    grant was fixed and the classification was left alone.
    """

    def _driver(self):
        return AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))

    def test_a_denied_permission_is_attributable_not_transient(
        self, tmp_path
    ) -> None:
        error = self._driver()._empty_answer_error(_capsule(tmp_path, HEADLESS_DENIAL))

        assert isinstance(error, LocalAgentUnavailableError)
        # The tool it was denied, so the reader knows which grant is missing.
        assert "read_file" in str(error)
        assert "cap-1" in str(error)

    def test_a_reworded_line_still_classifies_without_the_tool_name(
        self, tmp_path
    ) -> None:
        """The two substrings decide; the quoted name is a bonus. A CLI that
        rephrases its message should cost the detail, never the verdict."""

        error = self._driver()._empty_answer_error(
            _capsule(tmp_path, "headless mode has no permission prompt available\n")
        )

        assert isinstance(error, LocalAgentUnavailableError)
        assert "read_file" not in str(error)

    def test_any_other_empty_turn_stays_transient(self, tmp_path) -> None:
        error = self._driver()._empty_answer_error(
            _capsule(tmp_path, "provider stream closed early\n")
        )

        assert isinstance(error, LocalAgentTransientError)
        # The pointer the generic branch never used to carry: without it the
        # message names no file anybody can open.
        assert "cap-1" in str(error)
        assert "events/stderr.log" in str(error)
