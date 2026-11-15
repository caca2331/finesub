"""agy on the tool protocol: slot projects, per-call server identity,
project-record grants, and the entitled (or refused) `call_mcp_tool` events."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent.local_agent import (
    AGENT_USER_RULES_NOTICE,
    AGY_MCP_CALL_TOOL,
    AGY_TOOL_AGENT_NAME,
    AGY_GUARD_SCRIPT,
    AGY_NATIVE_GUARD_SCRIPT,
    AGY_TOOL_GUARD_SCRIPT,
    AGY_TOOL_NATIVE_GUARD_SCRIPT,
    AgyDriverConfig,
    AgyLocalAgentDriver,
    LocalAgentRefusedCallError,
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


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (AGY_GUARD_SCRIPT, "deny"),
        (AGY_NATIVE_GUARD_SCRIPT, "deny"),
        (AGY_TOOL_GUARD_SCRIPT, "allow"),
        (AGY_TOOL_NATIVE_GUARD_SCRIPT, "allow"),
    ],
    ids=["media", "media-native", "tool", "tool-native"],
)
def test_every_guard_reads_its_utf8_payload_whatever_the_code_page(tmp_path, script: str, expected: str) -> None:
    # agy writes UTF-8; a text stdin decoding as GBK (Chinese Windows) paired the last byte of "确" with the
    # backslash of the escaped quote, the JSON broke and the dead hook read as a refusal (Nonoka X 0008).
    guard = tmp_path / "scripts" / "guard_view_file.py"  # the media guards find their project from __file__
    guard.parent.mkdir()
    guard.write_text(script, encoding="utf-8")
    call = {"name": AGY_MCP_CALL_TOOL, "args": {"ServerName": "finesub", "ToolName": "submit",
                                                 "Arguments": {"text": 'sub|1|这里不明确"|x'}}}
    completed = subprocess.run(
        [sys.executable, str(guard)],
        input=json.dumps({"toolCall": call}, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "gbk", "PYTHONUTF8": "0"},
        check=True,
    )
    assert json.loads(completed.stdout)["decision"] == expected


def test_tool_argv_binds_the_slot_project_and_grants_the_tools(tmp_path, monkeypatch) -> None:
    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",), max_parallel=2))
    driver._resolved_command = ("agy",)
    records = tmp_path / "records"
    records.mkdir()
    registered: list[Path] = []

    def fake_ensure(project_root: Path, *, native: bool = False, tool: bool = False, protocol: str = ""):
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
        "read_url(example.com)", "mcp(finesub/next_task)", "mcp(finesub/submit)", "read_file(*)",
    ]
    # The user-rules notice leads: this path builds its prompt from the
    # messages, not from the shared opening constants.
    assert argv[argv.index("--print") + 1] == f"{AGENT_USER_RULES_NOTICE}\n\ncall next_task"
    assert argv[argv.index("--agent") + 1] == AGY_TOOL_AGENT_NAME
    assert argv[argv.index("--project") + 1] == "11111111-2222-3333-4444-555555555555"

    # A second call rewrites the identity in place and adds no duplicate grants.
    server["env"]["FINESUB_MCP_SESSION"] = "s2"
    driver._argv(capsule, native_search=False, probe=None, mcp_server=server)
    config = json.loads((domain / ".finesub-tool-1" / ".agents" / "mcp_config.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["finesub"]["env"]["FINESUB_MCP_SESSION"] == "s2"
    record = json.loads((records / "11111111-2222-3333-4444-555555555555.json").read_text(encoding="utf-8"))
    assert len(record["permissionGrants"]["permissionGrants"]["allow"]) == 4


def test_tool_argv_sends_the_effort_like_the_plain_argv(tmp_path, monkeypatch) -> None:
    """The twin of `test_agy_sends_every_model_an_effort_it_has` for the
    tool protocol: agy's Claude needs `--effort`, and `xhigh` goes out as `high`."""

    from dataclasses import replace as dataclass_replace

    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",), max_parallel=1))
    driver._resolved_command = ("agy",)
    records = tmp_path / "records"
    records.mkdir()
    project_id = "11111111-2222-3333-4444-555555555555"
    (records / f"{project_id}.json").write_text(json.dumps({"id": project_id}), encoding="utf-8")
    monkeypatch.setattr(driver, "_ensure_project", lambda root, *, native=False, tool=False, protocol="": (project_id, "d"))
    monkeypatch.setattr(AgyLocalAgentDriver, "agy_project_records_dir", staticmethod(lambda: records))
    episode = tmp_path / "domain" / "episode-1"
    (episode / "input").mkdir(parents=True)
    messages_path = episode / "input" / "messages.json"
    messages_path.write_text(json.dumps([{"role": "user", "content": "go"}]), encoding="utf-8")
    capsule = SimpleNamespace(root=episode, episode_id="episode-1", messages_path=messages_path)
    server = {"command": "python", "args": [], "env": {}, "tools": ["submit"], "view_roots": []}
    driver._tool_slot_local.slot = 0

    def sent(reasoning_effort: str, forced: str = "") -> str | None:
        driver.config = dataclass_replace(driver.config, model="claude-sonnet-5-5", effort=forced)
        argv = driver._tool_argv(capsule, mcp_server=server, native_search=False,
                                 reasoning_effort=reasoning_effort)
        return argv[argv.index("--effort") + 1] if "--effort" in argv else None

    assert sent("medium") == "medium"
    assert sent("medium", forced="xhigh") == "high"


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


def test_a_slot_another_process_holds_is_skipped(tmp_path) -> None:
    """Every finesub process on one checkout resolves the same domain root;
    four runs at once each took slot 0 and rewrote each other's
    `view_roots.json` (2026-10-23). A slot is held by a lock file too."""

    import subprocess
    import sys

    from finesub.llm.agent.local_agent import AGY_TOOL_SLOT_LOCK_NAME

    driver = AgyLocalAgentDriver(
        AgyDriverConfig(command=("agy",), max_parallel=1, runtime_root=tmp_path)
    )
    domain = driver._tool_slot_pool().domain_root
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from pathlib import Path; from finesub_bootstrap.locks import holding_lock\n"
         "with holding_lock(Path(sys.argv[1]), timeout=5):\n"
         "    print('held', flush=True); sys.stdin.read()",
         str(domain / AGY_TOOL_SLOT_LOCK_NAME.format(slot=0))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        # Slot 0 is the other process's; `max_parallel=1` caps how many this
        # process holds, not which number it gets.
        slot = driver._acquire_tool_slot()
        assert slot == 1
        driver._release_tool_slot(slot)
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)
    slot = driver._acquire_tool_slot()
    assert slot == 0
    driver._release_tool_slot(slot)


def test_a_slot_taken_before_the_location_moved_is_not_used_and_goes_back(tmp_path, monkeypatch) -> None:
    """The domain root is resolved again inside the activity lease; a slot
    whose lock guards the old root must not open the new root's project, and
    the run gives it back to the pool it came from."""

    from finesub.llm.agent.local_agent import LocalAgentDriver, LocalAgentTransientError, _ToolSlotPool

    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",), runtime_root=tmp_path))
    driver._resolved_command = ["agy"]
    old, new = _ToolSlotPool(tmp_path / "old-domain"), _ToolSlotPool(tmp_path / "new-domain")
    pools = iter([old])
    monkeypatch.setattr(driver, "_tool_slot_pool", lambda: next(pools, new))
    episode = tmp_path / "new-domain" / "episode-1"
    episode.mkdir(parents=True)
    capsule = SimpleNamespace(root=episode, episode_id="episode-1", messages_path=episode / "m.json")

    def run_after_the_move(self, messages, **kwargs):
        return self._tool_argv(capsule, mcp_server=kwargs["mcp_server"], native_search=False, reasoning_effort="")

    monkeypatch.setattr(LocalAgentDriver, "run", run_after_the_move)
    with pytest.raises(LocalAgentTransientError, match="location moved"):
        driver.run([], mcp_server={"tools": []})
    assert old.in_use == {} and new.in_use == {}
    assert driver._tool_slot_local.slot is None and driver._tool_slot_local.pool is None


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
        AgyDriverConfig(command=("agy",), model="claude-sonnet-5-5", runtime_root=tmp_path)
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


def _agy_step(index, state, tool="", server="finesub", step_type="tool"):
    update = {"step_type": step_type, "state": state, "step_index": index}
    if tool:
        update["tool_name"] = AGY_MCP_CALL_TOOL
        update["tool_info"] = {"parameters": {"ServerName": server, "ToolName": tool}}
    return {"event": "step_update", "step_update": update}


def _agy_stream(tmp_path, *steps, response=""):
    rows = [
        {"event": "init", "conversation_id": "c50bb92f",
         "init": {"tools": ["view_file", AGY_MCP_CALL_TOOL], "permission_mode": "request-review"}},
        _agy_step(0, "DONE", step_type="user_input"),
        *steps,
        {"event": "result", "result": {"status": "SUCCESS", "response": response, "conversation_id": "c50bb92f"}},
    ]
    raw = tmp_path / "events.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return raw


#: What the research session's server offered (2026-10-31 run).
RESEARCH_TOOLS = frozenset({"next_task", "read_context", "pull_status", "submit", "web_search", "web_fetch"})


def test_a_nested_call_agy_refused_is_a_malformed_call_not_a_violation(tmp_path) -> None:
    """The 2026-10-31 run (agy 1.4.11, Gemini 3.8 Flash, research tool
    session): two good `web_search` calls, then `call_mcp_tool` nested as the
    ToolName. agy refused it (ACTIVE, then ERROR; it never reached the
    server), wrote "a tool required the "mcp" permission" to stderr and ended
    the turn empty. Read as a policy violation, the whole task died unretried."""

    raw = _agy_stream(
        tmp_path,
        _agy_step(4, "ACTIVE", "web_search"), _agy_step(4, "DONE", "web_search"),
        _agy_step(9, "ACTIVE", "web_search"), _agy_step(9, "DONE", "web_search"),
        _agy_step(15, "ACTIVE", AGY_MCP_CALL_TOOL), _agy_step(15, "ERROR", AGY_MCP_CALL_TOOL),
    )
    normalized, _usage, violations, content = _normalize_agy_events(
        raw, native_search=False, max_bytes=1_000_000, extra_entitled=RESEARCH_TOOLS
    )
    assert violations == []
    assert content == ""
    refused = [row for row in normalized if row.get("refused_call")]
    assert [(row["step_index"], row["state"]) for row in refused] == [(15, "ACTIVE"), (15, "ERROR")]
    assert {(row["server"], row["tool"]) for row in refused} == {("finesub", AGY_MCP_CALL_TOOL)}
    # The execution attempt names it, so an accepted session still shows it.
    attempt: dict = {"reported_model": "gemini-3.8-flash"}
    AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))._note_stream_facts(
        attempt, normalized, False
    )
    assert attempt["refused_tool_calls"] == [f"finesub/{AGY_MCP_CALL_TOOL}"]


@pytest.mark.parametrize(
    "steps, entitled, label",
    [
        # A tool the server serves but this call was not given: agy refuses it
        # too (no grant), yet reaching for it is the breach the check exists for.
        ((_agy_step(5, "ACTIVE", "web_search"), _agy_step(5, "ERROR", "web_search")),
         frozenset({"next_task", "submit"}), "finesub/web_search"),
        # Another server, refused or not.
        ((_agy_step(5, "ACTIVE", "fetch", server="other"), _agy_step(5, "ERROR", "fetch", server="other")),
         RESEARCH_TOOLS, "other/fetch"),
        # A name the server lacks, but agy ran it: not the refused shape.
        ((_agy_step(5, "ACTIVE", AGY_MCP_CALL_TOOL), _agy_step(5, "DONE", AGY_MCP_CALL_TOOL)),
         RESEARCH_TOOLS, f"finesub/{AGY_MCP_CALL_TOOL}"),
        # ... nor a step agy never ended ERROR ...
        ((_agy_step(5, "ACTIVE", AGY_MCP_CALL_TOOL),),
         RESEARCH_TOOLS, f"finesub/{AGY_MCP_CALL_TOOL}"),
        # ... nor one that went DONE as well, in either order.
        ((_agy_step(5, "ACTIVE", AGY_MCP_CALL_TOOL), _agy_step(5, "ERROR", AGY_MCP_CALL_TOOL),
          _agy_step(5, "DONE", AGY_MCP_CALL_TOOL)),
         RESEARCH_TOOLS, f"finesub/{AGY_MCP_CALL_TOOL}"),
        ((_agy_step(5, "ACTIVE", AGY_MCP_CALL_TOOL), _agy_step(5, "DONE", AGY_MCP_CALL_TOOL),
          _agy_step(5, "ERROR", AGY_MCP_CALL_TOOL)),
         RESEARCH_TOOLS, f"finesub/{AGY_MCP_CALL_TOOL}"),
        # No server in this session: any MCP call is unentitled.
        ((_agy_step(5, "ACTIVE", AGY_MCP_CALL_TOOL), _agy_step(5, "ERROR", AGY_MCP_CALL_TOOL)),
         frozenset(), f"finesub/{AGY_MCP_CALL_TOOL}"),
    ],
)
def test_other_unentitled_mcp_calls_stay_violations(tmp_path, steps, entitled, label) -> None:
    raw = _agy_stream(tmp_path, *steps, response="done")
    normalized, _usage, violations, _content = _normalize_agy_events(
        raw, native_search=False, max_bytes=1_000_000, extra_entitled=entitled
    )
    assert violations == [f"completion target invoked {AGY_MCP_CALL_TOOL}:{label}"]
    assert not any(row.get("refused_call") for row in normalized)


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

    def test_a_tool_session_ended_by_a_refused_malformed_call_is_asked_again(
        self, tmp_path
    ) -> None:
        """The same stderr line follows agy refusing a call to a tool name the
        server does not have. That is no missing grant: the tool session takes
        it as a premature stop rather than dropping the target."""

        capsule = _capsule(tmp_path, HEADLESS_DENIAL.replace("read_file", "mcp"))
        refused = {"refused_tool_calls": [f"finesub/{AGY_MCP_CALL_TOOL}"]}
        with pytest.raises(LocalAgentRefusedCallError) as excinfo:
            self._driver()._settle_answer(capsule, "", dict(refused), tool_session=True)
        assert f"finesub/{AGY_MCP_CALL_TOOL}" in str(excinfo.value)
        # Outside a tool session the shape is what it was.
        with pytest.raises(LocalAgentUnavailableError):
            self._driver()._settle_answer(capsule, "", dict(refused), tool_session=False)
        with pytest.raises(LocalAgentUnavailableError):
            self._driver()._settle_answer(capsule, "", {}, tool_session=True)


class TestLeakedThoughtTag:
    """agy sometimes ships its reasoning as the answer (`thought\\n` + prose,
    docs/llm_local_agent_agy.md §7). The driver only tags it: the output
    contract rejects it and the repair round resumes this conversation, where
    a transient error would drop the session and walk the chain instead."""

    def _driver(self):
        return AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))

    def test_a_leaked_thought_is_tagged_not_raised(self, tmp_path) -> None:
        attempt: dict = {}
        self._driver()._settle_answer(
            _capsule(tmp_path, ""),
            "thought\nThe user wants the CSV corrected; first I check row 3...",
            attempt,
            tool_session=False,
        )

        assert attempt["answer_shape"] == "thought_leak"

    def test_a_normal_answer_and_a_tool_session_carry_no_tag(self, tmp_path) -> None:
        normal: dict = {}
        self._driver()._settle_answer(
            _capsule(tmp_path, ""), "<translated>\n...", normal, tool_session=False
        )
        # A tool session's final text is not the product; the answer went
        # through `submit`, so nothing is read from it.
        tool: dict = {}
        self._driver()._settle_answer(
            _capsule(tmp_path, ""), "thought\nsubmitted", tool, tool_session=True
        )

        assert "answer_shape" not in normal
        assert "answer_shape" not in tool

    def test_an_empty_answer_still_raises(self, tmp_path) -> None:
        with pytest.raises(LocalAgentTransientError):
            self._driver()._settle_answer(
                _capsule(tmp_path, "stream closed\n"), "", {}, tool_session=False
            )


def test_the_tool_agent_documents_use_the_one_spelling_agy_runs() -> None:
    """agy 1.2.16, measured 2026-10-06: an `mcpServers` key makes agy drop the
    whole document and run its default agent; `call_mcp_tool` under `tools`
    is accepted but fails the executor. `inheritMcp: true` brings the server."""

    from finesub.llm.agent import local_agent as la

    for document in (la.AGY_TOOL_AGENT_DOCUMENT, la.AGY_TOOL_NATIVE_AGENT_DOCUMENT):
        frontmatter = document.split("---")[1]
        assert "mcpServers" not in frontmatter
        assert la.AGY_MCP_CALL_TOOL not in frontmatter
        assert "inheritMcp: true" in frontmatter
        assert "  - view_file" in frontmatter
    assert la.AGY_SEARCH_TOOL in la.AGY_TOOL_NATIVE_AGENT_DOCUMENT.split("---")[1]


@pytest.mark.parametrize("ignored", [False, True])
def test_project_check_fails_closed_when_agy_ignores_the_agent(tmp_path: Path, monkeypatch, ignored) -> None:
    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._project_paths(tmp_path, tool=True)
    seen: dict[str, object] = {}

    def fake_hooks(root, *, project_id="", create=False, log_path=None, agent_name=""):
        seen.update(project_id=project_id, agent_name=agent_name, log_path=log_path)
        warning = f'W session.go:94] Agent "{agent_name}" not found, falling back to default\n' if ignored else ""
        log_path.write_text("I server.go:1663] Language server version: 1.2.16\n" + warning, encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(driver, "_hooks_command", fake_hooks)
    monkeypatch.setattr(driver, "_verify_hook_output", lambda stdout, source: None)
    paths["agents_root"].mkdir(parents=True)
    if ignored:
        with pytest.raises(la.LocalAgentPolicyViolationError, match="default agent"):
            driver._check_project(tmp_path, "pid", paths, tool=True)
    else:
        driver._check_project(tmp_path, "pid", paths, tool=True)
    assert seen["project_id"] == "pid" and seen["agent_name"] == la.AGY_TOOL_AGENT_NAME
    assert not Path(str(seen["log_path"])).exists()


@pytest.mark.parametrize("log", [None, "", "I something else\n"])
def test_project_check_cannot_pass_on_a_log_agy_did_not_write(tmp_path: Path, monkeypatch, log) -> None:
    """No version line, no verdict: a missing or foreign log must not read as
    'the agent resolved'."""

    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._project_paths(tmp_path, tool=True)
    paths["agents_root"].mkdir(parents=True)

    def fake_hooks(root, *, project_id="", create=False, log_path=None, agent_name=""):
        if log is not None:
            log_path.write_text(log, encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(driver, "_hooks_command", fake_hooks)
    monkeypatch.setattr(driver, "_verify_hook_output", lambda stdout, source: None)
    with pytest.raises(la.LocalAgentUnavailableError, match="no readable log"):
        driver._check_project(tmp_path, "pid", paths, tool=True)


def test_a_log_that_cannot_be_removed_does_not_hide_the_verdict(tmp_path: Path, monkeypatch) -> None:
    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._project_paths(tmp_path, tool=True)
    paths["agents_root"].mkdir(parents=True)

    def fake_hooks(root, *, project_id="", create=False, log_path=None, agent_name=""):
        log_path.write_text(
            "I Language server version: 1.2.16\n" f'W Agent "{agent_name}" not found, falling back to default\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, "", "")

    def locked_unlink(self, missing_ok=False):
        raise PermissionError("held by the language server")

    monkeypatch.setattr(driver, "_hooks_command", fake_hooks)
    monkeypatch.setattr(driver, "_verify_hook_output", lambda stdout, source: None)
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    with pytest.raises(la.LocalAgentPolicyViolationError, match="default agent"):
        driver._check_project(tmp_path, "pid", paths, tool=True)


@pytest.mark.parametrize(("native", "tool"), [(False, False), (True, False), (False, True), (True, True)])
def test_every_project_kind_checks_the_agent_its_document_names(tmp_path: Path, native, tool) -> None:
    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._write_project_resources(tmp_path, native=native, tool=tool)
    frontmatter = paths["agent"].read_text(encoding="utf-8").split("---")[1]
    names = [line.split(":", 1)[1].strip() for line in frontmatter.splitlines() if line.startswith("name:")]
    assert names == [paths["agent"].parent.name]


def test_the_fast_path_does_not_reregister_over_a_policy_violation(tmp_path: Path, monkeypatch) -> None:
    """An ignored agent document is not a stale registration: it fails, and
    the recorded project stays as it was."""

    import json as _json

    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._write_project_resources(tmp_path, tool=True)
    state = {
        "protocol_version": la.AGY_PROJECT_PROTOCOL_VERSION,
        "project_id": "0" * 8 + "-0000-0000-0000-" + "0" * 12,
        "project_root": str(tmp_path.resolve()),
        "hook_digest": driver._hook_digest(paths),
    }
    paths["state"].write_text(_json.dumps(state), encoding="utf-8")
    before = paths["state"].read_text(encoding="utf-8")

    def ignored(root, project_id, paths, *, tool=False):
        raise la.LocalAgentPolicyViolationError("ignored the FineSub agent document")

    def no_register(*args, **kwargs):
        raise AssertionError("must not re-register")

    monkeypatch.setattr(driver, "_check_project", ignored)
    monkeypatch.setattr(driver, "_hooks_command", no_register)
    with pytest.raises(la.LocalAgentPolicyViolationError):
        driver._ensure_project(tmp_path, tool=True)
    assert paths["state"].read_text(encoding="utf-8") == before


def test_the_fast_path_does_not_reregister_when_the_log_cannot_be_judged(tmp_path: Path, monkeypatch) -> None:
    """Registering again cannot make agy write the log it did not write: one
    failure, no fresh project per call."""

    import json as _json

    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._write_project_resources(tmp_path, tool=True)
    state = {
        "protocol_version": la.AGY_PROJECT_PROTOCOL_VERSION,
        "project_id": "0" * 8 + "-0000-0000-0000-" + "0" * 12,
        "project_root": str(tmp_path.resolve()),
        "hook_digest": driver._hook_digest(paths),
    }
    paths["state"].write_text(_json.dumps(state), encoding="utf-8")
    before = paths["state"].read_text(encoding="utf-8")
    created: list[bool] = []

    def fake_hooks(root, *, project_id="", create=False, log_path=None, agent_name=""):
        created.append(create)
        log_path.write_text("I unrelated line\n", encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(driver, "_hooks_command", fake_hooks)
    monkeypatch.setattr(driver, "_verify_hook_output", lambda stdout, source: None)
    with pytest.raises(la.AgyAgentUnverifiableError):
        driver._ensure_project(tmp_path, tool=True)
    assert created == [False]
    assert paths["state"].read_text(encoding="utf-8") == before


@pytest.mark.parametrize("log", ["", 'I Language server version: 1.2.16\nW Agent "finesub-media" not found\n'])
def test_a_capsule_project_warns_and_runs_on_when_its_agent_is_ignored(tmp_path: Path, monkeypatch, reported, log) -> None:
    """Owner 2026-10-06: agy is the only route for audio / video windows, and
    the hook bounds them either way -- a broken document parse warns once."""

    from finesub.llm.agent import local_agent as la

    la._READINESS_REPORTED.clear()
    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._project_paths(tmp_path)
    paths["agents_root"].mkdir(parents=True)

    def fake_hooks(root, *, project_id="", create=False, log_path=None, agent_name=""):
        log_path.write_text(log, encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(driver, "_hooks_command", fake_hooks)
    monkeypatch.setattr(driver, "_verify_hook_output", lambda stdout, source: None)
    driver._check_project(tmp_path, "pid", paths)
    expected = "agent-cli-agent-document-unverifiable" if not log else "agent-cli-agent-document-ignored"
    assert reported.codes() == [expected]
    la._READINESS_REPORTED.clear()


def test_the_hooks_check_names_the_agent_on_the_command_line(tmp_path: Path, monkeypatch) -> None:
    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    driver._resolved_command = ("agy",)
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(la.subprocess, "run", fake_run)
    driver._hooks_command(tmp_path, project_id="pid", log_path=tmp_path / "x.log", agent_name="finesub-tool")
    argv = seen[0]
    assert argv[argv.index("--agent") + 1] == "finesub-tool"
    assert argv[argv.index("--project") + 1] == "pid"
    assert argv[argv.index("--log-file") + 1] == str(tmp_path / "x.log")


def test_a_tool_project_ignored_at_registration_stays_failed_without_reregistering(tmp_path: Path, monkeypatch) -> None:
    """First registration writes the state, then the final check fails closed;
    the next call takes the fast path and fails the same way, registering
    nothing new."""

    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    pid = "11111111-2222-3333-4444-555555555555"
    created: list[bool] = []

    def fake_hooks(root, *, project_id="", create=False, log_path=None, agent_name=""):
        created.append(create)
        text = "I Language server version: 1.2.16\n"
        if create:
            text += f'project: created project "x" (id={pid}) at fake.json\n'
        else:
            text += f'W Agent "{agent_name}" not found, falling back to default\n'
        log_path.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(driver, "_hooks_command", fake_hooks)
    monkeypatch.setattr(driver, "_verify_hook_output", lambda stdout, source: None)
    with pytest.raises(la.LocalAgentPolicyViolationError):
        driver._ensure_project(tmp_path, tool=True)
    assert created == [True, False]
    with pytest.raises(la.LocalAgentPolicyViolationError):
        driver._ensure_project(tmp_path, tool=True)
    assert created == [True, False, False]


def _agent_text(paths) -> str:
    return paths["agent"].read_text(encoding="utf-8")


def test_the_protocol_rides_the_agent_document_outside_the_digest(tmp_path: Path) -> None:
    """agy re-reads the agent document every call, so this call's protocol
    goes there; the project digest covers the fixed part only."""

    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    bare = driver._hook_digest(driver._write_project_resources(tmp_path, tool=True))
    paths = driver._write_project_resources(tmp_path, tool=True, protocol="RULES ONE")
    one = driver._hook_digest(paths)
    assert "RULES ONE" in _agent_text(paths)
    assert _agent_text(paths).startswith(la.AGY_TOOL_AGENT_DOCUMENT.rstrip("\n"))
    paths = driver._write_project_resources(tmp_path, tool=True, protocol="RULES TWO")
    assert "RULES TWO" in _agent_text(paths) and "RULES ONE" not in _agent_text(paths)
    assert driver._hook_digest(paths) == one == bare
    # The identity names the fixed document only.
    assert la.AGY_AGENT_PROTOCOL_MARKER not in la.AGY_TOOL_AGENT_DOCUMENT


def test_a_frontmatter_change_still_moves_the_digest(tmp_path: Path, monkeypatch) -> None:
    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    before = driver._hook_digest(driver._write_project_resources(tmp_path, tool=True, protocol="P"))
    monkeypatch.setattr(
        la,
        "AGY_TOOL_AGENT_DOCUMENT",
        la.AGY_TOOL_AGENT_DOCUMENT.replace("commandExecutionPolicy: sandbox", "commandExecutionPolicy: off"),
    )
    after = driver._hook_digest(driver._write_project_resources(tmp_path, tool=True, protocol="P"))
    assert after != before


def test_the_check_sees_the_document_this_call_runs_with(tmp_path: Path, monkeypatch) -> None:
    """Written before the digest and the zero-token check, so a document agy
    would drop fails this call, not the next one."""

    import json as _json

    from finesub.llm.agent import local_agent as la

    driver = la.AgyLocalAgentDriver(la.AgyDriverConfig(command=("agy",)))
    paths = driver._write_project_resources(tmp_path, tool=True)
    paths["state"].write_text(_json.dumps({
        "protocol_version": la.AGY_PROJECT_PROTOCOL_VERSION,
        "project_id": "0" * 8 + "-0000-0000-0000-" + "0" * 12,
        "project_root": str(tmp_path.resolve()),
        "hook_digest": driver._hook_digest(paths),
    }), encoding="utf-8")
    checked: list[str] = []

    def check(root, project_id, paths, *, tool=False):
        checked.append(paths["agent"].read_text(encoding="utf-8"))

    def no_register(*args, **kwargs):
        raise AssertionError("a new protocol must not register a new project")

    monkeypatch.setattr(driver, "_check_project", check)
    monkeypatch.setattr(driver, "_hooks_command", no_register)
    for protocol in ("RULES ONE", "RULES TWO", "RULES TWO"):
        driver._ensure_project(tmp_path, tool=True, protocol=protocol)
    assert ["RULES ONE" in text for text in checked] == [True, False, False]
    assert ["RULES TWO" in text for text in checked] == [False, True, True]


def test_tool_argv_hands_the_protocol_to_the_project(tmp_path, monkeypatch) -> None:
    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",), max_parallel=1))
    driver._resolved_command = ("agy",)
    records = tmp_path / "records"
    records.mkdir()
    project_id = "11111111-2222-3333-4444-555555555555"
    (records / f"{project_id}.json").write_text(json.dumps({"id": project_id}), encoding="utf-8")
    handed: list[str] = []

    def ensure(root, *, native=False, tool=False, protocol=""):
        handed.append(protocol)
        return project_id, "d"

    monkeypatch.setattr(driver, "_ensure_project", ensure)
    monkeypatch.setattr(AgyLocalAgentDriver, "agy_project_records_dir", staticmethod(lambda: records))
    episode = tmp_path / "domain" / "episode-1"
    (episode / "input").mkdir(parents=True)
    messages_path = episode / "input" / "messages.json"
    messages_path.write_text(json.dumps([{"role": "user", "content": "go"}]), encoding="utf-8")
    capsule = SimpleNamespace(root=episode, episode_id="episode-1", messages_path=messages_path)
    server = {"command": "python", "args": [], "env": {}, "tools": ["submit"], "view_roots": []}
    driver._tool_slot_local.slot = 0
    driver._tool_argv(capsule, mcp_server=server, native_search=False, reasoning_effort="")
    driver._tool_argv(capsule, mcp_server={**server, "opening": True, "agent_protocol": "RULES"},
                      native_search=False, reasoning_effort="")
    assert handed == ["", "RULES"]
