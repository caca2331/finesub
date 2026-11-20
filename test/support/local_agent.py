"""Fake agent CLIs (Codex, Claude Code, agy) and the drivers that run them.

The drivers' own tests are `test_llm_local_agent.py`; the session host, continuation, user-rules
and Codex connection tests start the same fakes.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from finesub.llm.agent.local_agent import (
    AgyDriverConfig,
    AgyLocalAgentDriver,
    CODEX_DISABLED_FEATURES,
    CodexDriverConfig,
    CodexLocalAgentDriver,
)


FAKE_CODEX = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FEATURES = __FEATURES__
args = sys.argv[1:]
if "--version" in args:
    print("codex-cli 0.169.1")
    raise SystemExit(0)
if "--help" in args:
    print("SESSION_ID --json --output-last-message --ephemeral --sandbox read-only --search --ignore-user-config --ignore-rules")
    raise SystemExit(0)
if args[:2] == ["features", "list"]:
    print("\n".join(f"{name}  stable  true" for name in FEATURES) + "\napply_patch_freeform  removed  false")
    raise SystemExit(0)
if args[-2:] == ["login", "status"]:
    if (Path(os.environ["CODEX_HOME"]) / "auth.json").is_file():
        print("Logged in using ChatGPT")
        raise SystemExit(0)
    print("Not logged in")
    raise SystemExit(1)
if args[:3] == ["debug", "models", "--bundled"]:
    print(json.dumps({"models": [{"slug": "gpt-fake", "context_window": 1000, "tool_mode": "code",
                                  "multi_agent_version": "v2", "apply_patch_tool_type": "freeform",
                                  "experimental_supported_tools": ["request_user_input_async"]}]}))
    raise SystemExit(0)
payload = sys.stdin.buffer.read()
print(json.dumps({"type":"thread.started","thread_id":"11111111-2222-3333-4444-555555555555"}))
mode = "ok"
for marker in ("sleep", "child", "tool", "search", "search-started", "fail", "config-fail", "catalog-fail", "auth-fail", "large", "env", "envdump", "argv", "unknown", "todo", "item-error", "turn-failed", "stdout-flood", "stderr-flood", "stale-cache-turn-failed"):
    if marker.encode() in payload:
        mode = marker
if mode == "sleep":
    time.sleep(30)
if mode == "child":
    code = "import time; from pathlib import Path; time.sleep(1.5); Path('child-survived.txt').write_text('alive')"
    subprocess.Popen([sys.executable, "-c", code])
    time.sleep(30)
if mode == "tool":
    print(json.dumps({"type":"item.completed","item":{"type":"command_execution","status":"completed"}}))
if mode == "search":
    print(json.dumps({"type":"item.completed","item":{"type":"web_search","query":"FineSub test"}}))
if mode == "search-started":
    print(json.dumps({"type":"item.started","item":{"type":"web_search","query":"FineSub test","status":"in_progress"}}))
if mode == "unknown":
    print(json.dumps({"type":"item.completed","item":{"type":"future_tool","status":"completed"}}))
if mode == "todo":
    print(json.dumps({"type":"item.completed","item":{"type":"todo_list","items":[],"status":"completed"}}))
if mode == "item-error":
    print(json.dumps({"type":"item.completed","item":{"type":"error","message":"optional setting omitted"}}))
if mode == "stdout-flood":
    sys.stdout.write("x" * 100000)
    sys.stdout.flush()
    time.sleep(30)
if mode == "stderr-flood":
    sys.stderr.write("x" * 100000)
    sys.stderr.flush()
    time.sleep(30)
if mode == "fail":
    raise SystemExit(7)
if mode == "config-fail":
    sys.stderr.write("Error loading config.toml: invalid test setting\n")
    raise SystemExit(7)
if mode == "catalog-fail":
    sys.stderr.write("failed to decode models response: future schema\n")
    raise SystemExit(7)
if mode == "auth-fail":
    sys.stderr.write("ERROR: unexpected status 401 Unauthorized: refresh token was revoked\n")
    raise SystemExit(1)
if mode == "stale-cache-turn-failed":
    sys.stderr.write("ERROR codex_models_manager::manager: failed to load models cache: missing field\n")
    print(json.dumps({"type":"turn.started"}))
    print(json.dumps({"type":"error","message":"You've hit your usage limit."}))
    print(json.dumps({"type":"turn.failed","error":{"message":"You've hit your usage limit."}}))
    raise SystemExit(1)
if mode == "large":
    result = "x" * 100
elif mode == "env":
    result = str("GEMINI_FREE" in os.environ)
elif mode == "envdump":
    result = json.dumps({"env": dict(os.environ), "argv": args})
elif mode == "argv":
    result = json.dumps(args)
else:
    result = "<translated>ok</translated>"
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":result,"status":"completed"}}))
if mode == "turn-failed":
    print(json.dumps({"type":"turn.failed","error":{"message":"failed after partial"}}))
else:
    print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}))
'''.replace("__FEATURES__", repr(sorted(CODEX_DISABLED_FEATURES)))


FAKE_CLAUDE = r'''
import json
import os
import sys

args = sys.argv[1:]
if "--version" in args:
    print("2.1.227 (Claude Code)")
    raise SystemExit(0)
if "--help" in args:
    print("--output-format stream-json --no-session-persistence --tools --allowed-tools "
          "--safe-mode --setting-sources --effort --model --resume --session-id")
    raise SystemExit(0)
payload = sys.stdin.buffer.read()
mode = "ok"
for marker in ("argv", "search", "tool", "auth-fail", "api-fail", "leak", "env"):
    if marker.encode() in payload:
        mode = marker

session = {"type": "system", "subtype": "init", "tools": [], "model": "fake",
           "session_id": "11111111-2222-3333-4444-555555555555"}
if mode == "leak":
    session["tools"] = ["Bash", "Read"]
print(json.dumps(session))

if mode == "tool":
    print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "whoami"}}]}}))
if mode == "search" or (mode == "argv" and "--allowed-tools" in args):
    print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "server_tool_use", "name": "WebSearch",
         "input": {"query": "FineSub test"}}]}}))

if mode == "argv":
    result = json.dumps(args)
elif mode == "env":
    result = str("GEMINI_FREE" in os.environ)
else:
    result = "<translated>ok</translated>"

if mode == "auth-fail":
    print(json.dumps({"type": "result", "subtype": "success", "is_error": True,
                      "terminal_reason": "api_error", "usage": {},
                      "result": "Not logged in · Please run /login"}))
elif mode == "api-fail":
    print(json.dumps({"type": "result", "subtype": "success", "is_error": True,
                      "terminal_reason": "api_error", "usage": {},
                      "result": "Overloaded"}))
else:
    print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": result}]}}))
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "terminal_reason": "end_turn", "total_cost_usd": 0.01,
                      "permission_denials": [],
                      "usage": {"input_tokens": 10, "output_tokens": 3,
                                "cache_read_input_tokens": 2,
                                "cache_creation_input_tokens": 0}}))
'''


FAKE_AGY = r'''
import json
from pathlib import Path
import sys

args = sys.argv[1:]
if "--version" in args:
    print("1.1.13")
    raise SystemExit(0)
if "--help" in args:
    print("--print --output-format stream-json --new-project --project --agent "
          "--sandbox --effort --model --print-timeout --log-file --conversation")
    raise SystemExit(0)
if "/hooks" in args:
    root = Path.cwd()
    if "--log-file" in args:
        log = Path(args[args.index("--log-file") + 1])
        log.write_text(
            "I server.go:1663] Language server version: 1.1.13\n"
            'project: created project "runtime" '
            '(id=11111111-2222-3333-4444-555555555555) at fake.json\n',
            encoding="utf-8",
        )
    hook = {
        "name": "finesub-view-boundary",
        "enabled": True,
        "source": str((root / ".agents" / "hooks.json").resolve()),
        "actions": [{
            "event": "PreToolUse",
            "matcher": "*",
            "type": "command",
            "command": "python scripts/guard_read_file.py",
            "timeout_seconds": 10,
        }],
    }
    command = {"name": "hooks", "data": {"hooks": [hook]}}
    print(json.dumps({"event": "command_result", "command": command}))
    print(json.dumps({"event": "result", "result": {
        "conversation_id": "", "status": "SUCCESS", "response": "",
        "duration_seconds": 0, "num_turns": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "thinking_tokens": 0, "cache_read_tokens": 0,
                  "total_tokens": 0},
        "command": command,
    }}))
    raise SystemExit(0)

sys.stdin.buffer.read()
messages = json.loads((Path.cwd() / "input" / "messages.json").read_text(encoding="utf-8"))
print(json.dumps({"event": "init", "conversation_id": "agy-conv", "init": {
    "cwd": str(Path.cwd()), "tools": ["run_command", "read_file"],
    "permission_mode": "request-review",
}}))
print(json.dumps({"event": "step_update", "step_update": {
    "conversation_id": "agy-conv", "step_index": 0,
    "state": "DONE", "step_type": "agent_response",
}}))
asked = json.dumps(messages)
if "ROGUE" in asked:  # a tool this call is not entitled to
    print(json.dumps({"event": "step_update", "step_update": {
        "conversation_id": "agy-conv", "step_index": 1, "state": "DONE",
        "step_type": "tool", "tool_name": "search_web",
        "tool_info": {"name": "search_web", "parameters": {"query": "q"}},
    }}))
cut_off = "CUTOFF" in asked
print(json.dumps({"event": "result", "result": {
    "conversation_id": "agy-conv", "status": "ERROR" if cut_off else "SUCCESS",
    **({"error": "Your previous response was cut off because it exceeded the "
                 "output token limit\nRetries remaining: 3"} if cut_off else {}),
    "response": (
        ""
        if cut_off
        # SUCCESS after an empty-reply retry: only the last reply
        else "gy|b\n</translated>\n"
        if "RETRIED" in asked
        else "thought\nI should check row 1 first."
        if "LEAK" in asked
        else "<translated>agy</translated>"
    ),
    "duration_seconds": 1, "num_turns": 1,
    "usage": {"input_tokens": 12, "output_tokens": 3,
              "thinking_tokens": 2, "cache_read_tokens": 4,
              "total_tokens": 17},
}}))
if "EXIT3" in asked:
    raise SystemExit(3)
'''


def _claude_driver(tmp_path: Path, monkeypatch, **overrides):
    from finesub.llm.agent.local_agent import ClaudeCodeDriverConfig, ClaudeCodeLocalAgentDriver

    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    monkeypatch.setenv("GEMINI_FREE", "secret-gemini")
    return ClaudeCodeLocalAgentDriver(
        ClaudeCodeDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            model=overrides.get("model", "claude-opus-5"),
            timeout_seconds=overrides.get("timeout_seconds", 5),
            max_result_bytes=overrides.get("max_result_bytes", 1024),
            max_event_bytes=overrides.get("max_event_bytes", 8192),
            max_stderr_bytes=overrides.get("max_stderr_bytes", 8192),
        )
    )


def _driver(tmp_path: Path, monkeypatch, **overrides) -> CodexLocalAgentDriver:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    monkeypatch.setenv("GEMINI_FREE", "secret-gemini")
    config = CodexDriverConfig(
        command=(sys.executable, str(script)),
        runtime_root=tmp_path / "runtime",
        timeout_seconds=overrides.get("timeout_seconds", 5),
        max_result_bytes=overrides.get("max_result_bytes", 1024),
        max_event_bytes=overrides.get("max_event_bytes", 8192),
        max_stderr_bytes=overrides.get("max_stderr_bytes", 8192),
        config_overrides=overrides.get("config_overrides", ()),
        max_parallel=overrides.get("max_parallel", 1),
        stall_timeout_seconds=overrides.get("stall_timeout_seconds", 0.0),
        keep_episodes=overrides.get("keep_episodes", False),
        model=overrides.get("model", ""),
        **({"harness_home": overrides["harness_home"]} if "harness_home" in overrides else {}),
    )
    return CodexLocalAgentDriver(config)


def _agy_task_parts(root: Path, count: int = 1) -> Path:
    """What `_prepare_capsule_input` leaves for an agy `_argv` to name."""

    task = root / "input" / "task"
    task.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        (task / f"part-{index:02d}.md").write_text("@@ message 1 · user\n", encoding="utf-8")
    return task


def _agy_driver(tmp_path: Path) -> AgyLocalAgentDriver:
    script = tmp_path / "fake_agy.py"
    script.write_text(FAKE_AGY, encoding="utf-8")
    driver = AgyLocalAgentDriver(
        AgyDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            model="gemini-3.7-flash",
            timeout_seconds=5,
            project_setup_timeout_seconds=5,
            max_result_bytes=1024,
            max_event_bytes=8192,
            max_stderr_bytes=8192,
        )
    )
    # The fake CLI writes no project record; grants have their own tests.
    driver._grant_permissions = lambda *a, **k: None  # type: ignore[assignment]
    return driver


def _make_directory_link(link: Path, target: Path) -> None:
    """A directory link, by whichever mechanism this machine allows.

    On Windows `os.symlink` needs a privilege an ordinary account lacks, but a
    *junction* needs none -- and a junction is the shape users actually create
    (`finesub relocate` documents redirecting a big directory off the system
    drive), so falling back to it keeps this covered rather than skipped.
    """

    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name != "nt":
        pytest.skip("this platform/user cannot create directory links")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not os.path.isjunction(link):
        pytest.skip("this machine cannot create directory links")
