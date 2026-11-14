"""Codex tool sessions forked from one parent session per call shape (1c)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from finesub.llm.agent.local_agent import (
    CODEX_PARENT_CWD_NAME,
    CODEX_PARENTS_NAME,
    CodexDriverConfig,
    CodexLocalAgentDriver,
)

from finesub.llm.agent import codex_connection

from .test_llm_agent_codex_connection import PROXY_CONFIG, _write_config

FAKE_CODEX = r'''
import json
import os
from pathlib import Path
import sys
import uuid

args = sys.argv[1:]
here = Path(__file__).parent
if "--version" in args:
    print("codex-cli 0.160.0")
    raise SystemExit(0)
if "--help" in args:
    print("SESSION_ID --json --ephemeral --sandbox read-only --search --ignore-user-config --ignore-rules --config mcp")
    raise SystemExit(0)
if args[:2] == ["features", "list"]:
    print("apps  stable  true\nplugins  stable  true\nremote_plugin  stable  true")
    raise SystemExit(0)
payload = sys.stdin.buffer.read().decode("utf-8")
if ("fork" in args or "resume" in args) and args[-1] != "-":
    payload = ""  # the real CLI reads stdin only when the prompt is `-`


def turn(thread, text, usage):
    print(json.dumps({"type": "thread.started", "thread_id": thread}))
    print(json.dumps({"type": "turn.started"}))
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}))
    print(json.dumps({"type": "turn.completed", "usage": usage}))


if "fork" in args:
    if (here / "fail-fork").exists():
        sys.stderr.write("Error: thread/fork failed\n")
        raise SystemExit(1)
    report = {"argv": args, "home": os.environ.get("CODEX_HOME"), "protocol_in_stdin": "## Protocol" in payload,
              "opening_in_stdin": "OPENING-TEXT" in payload, "prompt_in_stdin": "task-runtime worker" in payload,
              "stdin": payload[:600]}
    turn("child-" + uuid.uuid4().hex[:6], json.dumps(report),
         {"input_tokens": 20000, "cached_input_tokens": 15000, "output_tokens": 50})
elif args[-1] == "-" and "## Protocol" in payload and "There is no task yet" in payload:
    with (here / "parent-attempts.log").open("a", encoding="utf-8") as handle:
        handle.write("x\n")
    if (here / "fail-parent").exists():
        raise SystemExit(3)
    log = here / "parents.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": args, "cwd": os.getcwd()}) + "\n")
    count = len(log.read_text(encoding="utf-8").splitlines())
    rollouts = Path(os.environ["CODEX_HOME"]) / "sessions" / "2026" / "10" / "17"
    rollouts.mkdir(parents=True, exist_ok=True)
    (rollouts / f"rollout-2026-10-17T00-00-00-parent-{count}.jsonl").write_text("{}", encoding="utf-8")
    reply = (here / "parent-reply").read_text(encoding="utf-8") if (here / "parent-reply").exists() else "READY"
    for kind in ("error", "mcp_tool_call"):
        if (here / f"parent-{kind}").exists():
            print(json.dumps({"type": "item.completed", "item": {"type": kind, "message": "x"}}))
    turn(f"parent-{count}", reply, {"input_tokens": 13000, "cached_input_tokens": 0, "output_tokens": 5})
else:
    report = {"argv": args, "home": os.environ.get("CODEX_HOME"), "protocol_in_stdin": "## Protocol" in payload}
    turn("solo", json.dumps(report), {"input_tokens": 30000, "cached_input_tokens": 0, "output_tokens": 60})
'''

PROTOCOL = "RULES: answer good"


def _driver(tmp_path: Path, **config) -> CodexLocalAgentDriver:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    return CodexLocalAgentDriver(CodexDriverConfig(
        command=(sys.executable, str(script)),
        runtime_root=tmp_path / "runtime",
        timeout_seconds=15,
        max_result_bytes=1_000_000,
        **config,
    ))


def _server(**extra) -> dict:
    return {
        "command": sys.executable,
        "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
        "env": {"FINESUB_MCP_ROOT": "C:/call-root", "FINESUB_MCP_ASSIGNMENT": "call-1",
                "FINESUB_MCP_WORKER": "worker-1", "FINESUB_MCP_SESSION": "s1",
                "FINESUB_MCP_LOG": "C:/call-root/frames.jsonl", "FINESUB_MCP_TOOLS": "submit"},
        "tools": ["submit"],
        "opening": True,
        "agent_protocol": PROTOCOL,
        **extra,
    }


def _call(driver: CodexLocalAgentDriver):
    result = driver.run([{"role": "user", "content": "OPENING-TEXT"}], task="general", mcp_server=_server())
    return result, json.loads(result.content)


def _parent_attempts(tmp_path: Path) -> int:
    log = tmp_path / "parent-attempts.log"
    return len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0


def _table(driver: CodexLocalAgentDriver) -> dict:
    domain = driver.capsules.resolve_location().parent.resolve()
    return json.loads((domain / CODEX_PARENTS_NAME).read_text(encoding="utf-8"))


def _parents(tmp_path: Path) -> list[dict]:
    log = tmp_path / "parents.log"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


@pytest.fixture
def proxy(tmp_path: Path, monkeypatch) -> None:
    # Every call here makes several connects (parent, child, retries); the
    # pre-flight is `test_llm_agent_codex_connection`'s subject, not this one's.
    monkeypatch.setattr(codex_connection, "check_ready", lambda *args, **kwargs: None)
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=8317))


def test_the_first_call_of_a_key_only_records_it(tmp_path: Path, proxy) -> None:
    """A protocol used once would pay a whole parent request for nothing."""

    driver = _driver(tmp_path, parent_fork=True)

    result, report = _call(driver)

    assert "fork" not in report["argv"] and report["protocol_in_stdin"] is True
    assert _parents(tmp_path) == []
    assert "fork_parent" not in result.execution_attempt["isolation"]
    domain = driver.capsules.resolve_location().parent.resolve()
    table = json.loads((domain / CODEX_PARENTS_NAME).read_text(encoding="utf-8"))
    assert [sorted(entry) for entry in table.values()] == [["first_seen"]]


def test_the_second_call_builds_the_parent_and_later_ones_reuse_it(tmp_path: Path, proxy) -> None:
    driver = _driver(tmp_path, parent_fork=True)
    _call(driver)

    first, one = _call(driver)
    second, two = _call(driver)

    parents = _parents(tmp_path)
    assert len(parents) == 1
    parent_argv = parents[0]["argv"]
    assert "--ephemeral" not in parent_argv and "fork" not in parent_argv
    assert Path(parents[0]["cwd"]).name == CODEX_PARENT_CWD_NAME
    # The parent's server lists the same tools under an identity bound to no task.
    override = parent_argv[parent_argv.index("--config", parent_argv.index("--sandbox")) + 1]
    assert 'FINESUB_MCP_ASSIGNMENT = "fork-parent"' in override and "FINESUB_MCP_LOG" not in override
    for report in (one, two):
        argv = report["argv"]
        assert "--ephemeral" in argv
        # `fork <parent> -`: the prompt and the opening both ride stdin.
        assert argv[-3:] == ["fork", "parent-1", "-"]
        assert report["opening_in_stdin"] is True and report["prompt_in_stdin"] is True
        assert report["protocol_in_stdin"] is False
        assert "user-level files" not in report["stdin"]  # the harness home injects none
    domain = driver.capsules.resolve_location().parent.resolve()
    table = json.loads((domain / CODEX_PARENTS_NAME).read_text(encoding="utf-8"))
    assert [entry["session_id"] for entry in table.values()] == ["parent-1"]
    assert first.execution_attempt["isolation"]["fork_parent"] == {
        "session_id": "parent-1", "created_by_this_call": True}
    assert second.execution_attempt["isolation"]["fork_parent"]["created_by_this_call"] is False


def test_the_dev_switch_keeps_a_fork_childs_rollout(tmp_path: Path, proxy, monkeypatch) -> None:
    """For acceptance runs: the child still forks, without `--ephemeral`."""

    from finesub.llm.agent.local_agent import CODEX_FORK_ROLLOUT_ENV

    driver = _driver(tmp_path, parent_fork=True)
    _call(driver)
    _call(driver)
    monkeypatch.setenv(CODEX_FORK_ROLLOUT_ENV, "1")
    _result, report = _call(driver)
    assert report["argv"][-3:] == ["fork", "parent-1", "-"] and "--ephemeral" not in report["argv"]


def test_only_the_building_call_books_the_parents_turn(tmp_path: Path, proxy) -> None:
    driver = _driver(tmp_path, parent_fork=True)
    _call(driver)

    first, _ = _call(driver)
    second, _ = _call(driver)

    assert first.usage["input_tokens"] == 20000 and first.usage["fork_parent_usage"] == "included"
    assert second.usage["input_tokens"] == 7000
    assert second.usage["cached_input_tokens"] == 15000
    assert second.usage["output_tokens"] == 45
    assert second.usage["fork_parent_usage"] == "subtracted"


def test_a_parent_whose_rollout_is_gone_is_rebuilt_before_forking(tmp_path: Path, proxy) -> None:
    driver = _driver(tmp_path, parent_fork=True)
    _call(driver)
    _, first = _call(driver)
    for rollout in Path(first["home"], "sessions").rglob("rollout-*.jsonl"):
        rollout.unlink()

    second, report = _call(driver)

    assert len(_parents(tmp_path)) == 2
    assert report["argv"][-3:] == ["fork", "parent-2", "-"]
    assert second.execution_attempt["isolation"]["fork_parent"]["created_by_this_call"] is True


def test_a_failing_fork_is_an_ordinary_failure(tmp_path: Path, proxy) -> None:
    from finesub.llm.agent.local_agent import LocalAgentTransientError

    driver = _driver(tmp_path, parent_fork=True)
    _call(driver)
    _call(driver)
    (tmp_path / "fail-fork").write_text("", encoding="utf-8")

    with pytest.raises(LocalAgentTransientError, match="exited with status 1"):
        _call(driver)
    assert len(_parents(tmp_path)) == 1  # the parent is not blamed


def test_a_parent_that_fails_to_build_falls_back_and_says_so(
    tmp_path: Path, proxy, reported
) -> None:
    driver = _driver(tmp_path, parent_fork=True)
    (tmp_path / "fail-parent").write_text("", encoding="utf-8")
    _call(driver)

    _, report = _call(driver)

    assert "fork" not in report["argv"] and report["protocol_in_stdin"] is True
    assert [w.code for w in reported.warnings] == ["agent-codex-fork-parent"]
    domain = driver.capsules.resolve_location().parent.resolve()
    assert list((domain / ".codex-parent-logs").glob("*.stderr.log"))


def test_a_failed_parent_is_not_rebuilt_until_the_retry_window_passes(
    tmp_path: Path, proxy, reported
) -> None:
    """A failing build would otherwise cost every later window a parent turn, under the lock."""

    driver = _driver(tmp_path, parent_fork=True)
    (tmp_path / "fail-parent").write_text("", encoding="utf-8")
    _call(driver)
    _call(driver)

    _, report = _call(driver)

    assert _parent_attempts(tmp_path) == 1
    assert "fork" not in report["argv"] and report["protocol_in_stdin"] is True
    (entry,) = _table(driver).values()
    assert entry["failed_at"] and "first_seen" in entry and "session_id" not in entry

    (tmp_path / "fail-parent").unlink()
    entry["failed_at"] = "2026-01-01T00:00:00+00:00"
    domain = driver.capsules.resolve_location().parent.resolve()
    (domain / CODEX_PARENTS_NAME).write_text(json.dumps({k: entry for k in _table(driver)}), encoding="utf-8")
    _, report = _call(driver)

    assert _parent_attempts(tmp_path) == 2 and report["argv"][-3:] == ["fork", "parent-1", "-"]


def test_a_parent_cwd_that_is_not_empty_builds_no_parent(tmp_path: Path, proxy, reported) -> None:
    """Codex reads rule files from its cwd, and every child inherits the parent's turn."""

    driver = _driver(tmp_path, parent_fork=True)
    _call(driver)
    domain = driver.capsules.resolve_location().parent.resolve()
    (domain / CODEX_PARENT_CWD_NAME).mkdir(parents=True, exist_ok=True)
    (domain / CODEX_PARENT_CWD_NAME / "AGENTS.md").write_text("rules", encoding="utf-8")

    _, report = _call(driver)

    assert _parent_attempts(tmp_path) == 0
    assert "fork" not in report["argv"] and report["protocol_in_stdin"] is True
    assert [w.code for w in reported.warnings] == ["agent-codex-fork-parent"]
    assert "AGENTS.md" in next(iter(_table(driver).values()))["error"]


def test_a_parent_that_does_not_just_reply_ready_is_not_used(tmp_path: Path, proxy, reported) -> None:
    driver = _driver(tmp_path, parent_fork=True)
    (tmp_path / "parent-reply").write_text("Sure! Here is a summary of the protocol.", encoding="utf-8")
    _call(driver)

    _, report = _call(driver)

    assert _parent_attempts(tmp_path) == 1
    assert "fork" not in report["argv"]
    assert "READY" in next(iter(_table(driver).values()))["error"]


@pytest.mark.parametrize("kind, used", [("error", True), ("mcp_tool_call", False)])
def test_a_parent_turn_with_a_tool_call_is_not_used_but_a_diagnostic_is_fine(
    tmp_path: Path, proxy, reported, kind: str, used: bool
) -> None:
    """Codex reports recoverable diagnostics as `error` items in a turn that still succeeds."""

    driver = _driver(tmp_path, parent_fork=True)
    (tmp_path / f"parent-{kind}").write_text("", encoding="utf-8")
    _call(driver)

    _, report = _call(driver)

    assert ("fork" in report["argv"]) is used


@pytest.mark.parametrize("configured, expected", [(None, "fork"), (True, "fork"), (False, "inline")])
def test_opening_mode_follows_the_config(
    tmp_path: Path, proxy, configured: bool | None, expected: str
) -> None:
    config = {} if configured is None else {"parent_fork": configured}

    assert _driver(tmp_path, **config).opening_mode() == expected


@pytest.mark.parametrize("config", [None, 'model_provider = "openai"\n'], ids=["no-config", "openai"])
def test_a_login_connection_forks_from_a_parent_in_the_users_home(
    tmp_path: Path, monkeypatch, config
) -> None:
    """A ChatGPT login cannot move to the harness home, so its parent stays in
    the user's: one `exec` session per call shape, which Codex Desktop does not
    list. The user's AGENTS.md reaches the child too, so its notice stays."""

    user_home = _write_config(monkeypatch, tmp_path, config)
    driver = _driver(tmp_path, parent_fork=True)

    assert driver.opening_mode() == "fork"
    _, first = _call(driver)  # the key's first call only records it
    _, built = _call(driver)
    _, reused = _call(driver)

    assert "fork" not in first["argv"] and first["protocol_in_stdin"] is True
    assert len(_parents(tmp_path)) == 1
    assert list((user_home / "sessions").rglob("rollout-*-parent-1.jsonl"))
    for report in (built, reused):
        assert report["argv"][-3:] == ["fork", "parent-1", "-"]
        assert report["home"] == str(user_home)
    assert "user-level files" in reused["stdin"]
