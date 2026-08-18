from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from finesub.llm.agent.local_agent import (
    RETAINED_FAILED_EPISODES,
    AgyDriverConfig,
    AgyLocalAgentDriver,
    AgentCapsuleManager,
    CodexDriverConfig,
    CodexLocalAgentDriver,
    LocalAgentError,
    LocalAgentPolicyViolationError,
    LocalAgentTimeoutError,
    LocalAgentUnavailableError,
    _normalize_events,
)
import finesub.llm.agent.local_agent as local_agent_module


FAKE_CODEX = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

args = sys.argv[1:]
if "--version" in args:
    print("codex-cli fake-1")
    raise SystemExit(0)
if "--help" in args:
    print("SESSION_ID --json --output-last-message --ephemeral --sandbox read-only --search --ignore-user-config --ignore-rules")
    raise SystemExit(0)
payload = sys.stdin.buffer.read()
print(json.dumps({"type":"thread.started","thread_id":"11111111-2222-3333-4444-555555555555"}))
mode = "ok"
for marker in ("sleep", "child", "tool", "search", "search-started", "fail", "config-fail", "catalog-fail", "large", "env", "argv", "unknown", "todo", "item-error", "turn-failed", "stdout-flood", "stderr-flood"):
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
if mode == "large":
    result = "x" * 100
elif mode == "env":
    result = str("GEMINI_FREE" in os.environ)
elif mode == "argv":
    result = json.dumps(args)
else:
    result = "<translated>ok</translated>"
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":result,"status":"completed"}}))
if mode == "turn-failed":
    print(json.dumps({"type":"turn.failed","error":{"message":"failed after partial"}}))
else:
    print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}))
'''

RACE_CODEX = r'''
from pathlib import Path
import subprocess
import sys
import time

args = sys.argv[1:]
if "--version" in args:
    print("codex-cli race-fake")
    raise SystemExit(0)
if "--help" in args:
    print("--json --ephemeral --sandbox read-only --search --ignore-user-config --ignore-rules")
    raise SystemExit(0)
code = "import time; from pathlib import Path; time.sleep(1.5); Path('spawn-race-survived.txt').write_text('alive')"
subprocess.Popen([sys.executable, "-c", code])
sys.stdin.buffer.read()
time.sleep(30)
'''


FAKE_CLAUDE = r'''
import json
import os
import sys

args = sys.argv[1:]
if "--version" in args:
    print("2.1.227 (Claude Code)")
    raise SystemExit(0)
if "--help" in args:
    print("--output-format stream-json --no-session-persistence --disallowed-tools "
          "--allowed-tools --safe-mode --setting-sources --effort --model --resume --session-id")
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
            "command": "python scripts/guard_view_file.py",
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
    "cwd": str(Path.cwd()), "tools": ["run_command", "view_file"],
    "permission_mode": "request-review",
}}))
print(json.dumps({"event": "step_update", "step_update": {
    "conversation_id": "agy-conv", "step_index": 0,
    "state": "DONE", "step_type": "agent_response",
}}))
print(json.dumps({"event": "result", "result": {
    "conversation_id": "agy-conv", "status": "SUCCESS",
    "response": "<translated>agy</translated>",
    "duration_seconds": 1, "num_turns": 1,
    "usage": {"input_tokens": 12, "output_tokens": 3,
              "thinking_tokens": 2, "cache_read_tokens": 4,
              "total_tokens": 17},
}}))
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
            effort=overrides.get("effort", ""),
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
    )
    return CodexLocalAgentDriver(config)


def _agy_driver(tmp_path: Path) -> AgyLocalAgentDriver:
    script = tmp_path / "fake_agy.py"
    script.write_text(FAKE_AGY, encoding="utf-8")
    return AgyLocalAgentDriver(
        AgyDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            model="gemini-3.7-flash",
            effort="high",
            timeout_seconds=5,
            project_setup_timeout_seconds=5,
            max_result_bytes=1024,
            max_event_bytes=8192,
            max_stderr_bytes=8192,
        )
    )


def test_agy_sends_effort_only_to_the_models_that_accept_it() -> None:
    """agy rejects `--effort` for every model that bakes the level into its id.

    Verified against the real CLI on 2026-08-15:

        Error: invalid model selection (--model "claude-opus-4-6-thinking"
        --effort "high"): --effort is not supported for model "..."

    That is a hard pre-flight failure, which classifies as transient -- two of
    them and the quota probe freezes the whole Antigravity allowance for five
    hours over a flag. Re-derive the split with `agy models`.
    """

    from finesub.llm.agent.local_agent import _agy_model_takes_effort

    assert _agy_model_takes_effort("gemini-3.7-flash") is True
    assert _agy_model_takes_effort("gemini-3.5-flash") is True
    assert _agy_model_takes_effort("claude-opus-4-6-thinking") is False
    assert _agy_model_takes_effort("claude-sonnet-4-6") is False
    assert _agy_model_takes_effort("gpt-oss-120b-medium") is False


def test_agy_argv_drops_effort_for_a_model_that_refuses_it(tmp_path: Path) -> None:
    """Even when `[llm].local_agent_reasoning_effort` forces one globally.

    That override reaches the driver config directly, bypassing the catalog's
    `thinking = false`, so this is the last line of defence.
    """

    from dataclasses import replace as dataclass_replace

    driver = _agy_driver(tmp_path)
    driver.config = dataclass_replace(
        driver.config, model="claude-opus-4-6-thinking", effort="high"
    )
    capsule = SimpleNamespace(
        root=tmp_path / "capsule" / "episode",
        messages_path=tmp_path / "capsule" / "episode" / "task.json",
    )
    capsule.root.mkdir(parents=True, exist_ok=True)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")

    argv = driver._argv(
        capsule,
        native_search=False,
        probe=SimpleNamespace(),
        reasoning_effort="high",
    )

    assert "--effort" not in argv
    assert argv[argv.index("--model") + 1] == "claude-opus-4-6-thinking"

    driver.config = dataclass_replace(driver.config, model="gemini-3.7-flash")
    gemini_argv = driver._argv(
        capsule,
        native_search=False,
        probe=SimpleNamespace(),
        reasoning_effort="high",
    )
    assert gemini_argv[gemini_argv.index("--effort") + 1] == "high"


def test_agy_repairs_only_inside_the_session_that_wrote_the_answer(
    tmp_path: Path,
) -> None:
    """Owner decision, 2026-08-15.

    A fresh agy session would have to be handed the whole window *and* the
    previous output, and that second copy is not the same object to the model:
    in its own session the answer is what it just wrote. So no session reuse
    means the retry stays blind, as it was before repair rounds existed.
    """

    driver = _agy_driver(tmp_path)

    assert not driver.accepts_repair_context(
        session_scope="task", conversation_handle=""
    )
    assert not driver.accepts_repair_context(
        session_scope="task", conversation_handle="handle-1"
    )
    assert not driver.accepts_repair_context(
        session_scope="assignment", conversation_handle=""
    )
    assert driver.accepts_repair_context(
        session_scope="assignment", conversation_handle="handle-1"
    )

    # The stateless drivers re-send the whole task every call anyway, so the
    # repair context is the only new cost and it is small next to what it saves.
    assert CodexLocalAgentDriver().accepts_repair_context(
        session_scope="task", conversation_handle=""
    )


def test_agy_is_told_where_the_validation_errors_are(tmp_path: Path) -> None:
    """agy is the one driver whose input is a path, not stdin.

    Its prompt names `messages.json` and nothing else, so a repair file it is
    not told about is a file it never opens -- the capsule would carry the
    errors and the model would never see them.
    """

    driver = _agy_driver(tmp_path)
    capsule = SimpleNamespace(
        root=tmp_path / "domain" / "episode",
        messages_path=tmp_path / "domain" / "episode" / "task.json",
    )
    (capsule.root / "input").mkdir(parents=True, exist_ok=True)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")

    plain = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
    assert "validation-errors.txt" not in plain[plain.index("--print") + 1]

    errors_path = capsule.root / "input" / "validation-errors.txt"
    errors_path.write_text("Row 1 references unknown source id 3.", encoding="utf-8")
    repair = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
    prompt = repair[repair.index("--print") + 1]

    assert str(errors_path.resolve()) in prompt
    # The answer itself is already in the resumed conversation; naming the file
    # would only pay for a second copy of it.
    assert "previous-output.txt" not in prompt


def test_agy_native_and_media_calls_use_separate_projects(tmp_path: Path) -> None:
    """The entitlement *is* the project's `.agents/` tree.

    Sharing one project would mean rewriting the guard for every call, with
    `max_parallel` concurrent calls racing over the file that decides what the
    model may touch. Two projects means each entitlement is written once.
    """

    from finesub.llm.agent.local_agent import AGY_AGENT_NAME, AGY_NATIVE_AGENT_NAME

    driver = _agy_driver(tmp_path)
    capsule = SimpleNamespace(
        root=tmp_path / "domain" / "episode",
        messages_path=tmp_path / "domain" / "episode" / "task.json",
    )
    capsule.root.mkdir(parents=True, exist_ok=True)
    driver._resolved_command = ("agy",)
    seen: list[tuple[Path, bool]] = []

    def _fake_ensure(root, native=False):
        seen.append((Path(root), native))
        return ("native-id" if native else "media-id"), "digest"

    driver._ensure_project = _fake_ensure

    media = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
    native = driver._argv(capsule, native_search=True, probe=SimpleNamespace())

    assert media[media.index("--agent") + 1] == AGY_AGENT_NAME
    assert native[native.index("--agent") + 1] == AGY_NATIVE_AGENT_NAME
    assert media[media.index("--project") + 1] == "media-id"
    assert native[native.index("--project") + 1] == "native-id"

    (media_root, media_native), (native_root, native_native) = seen
    assert media_native is False and native_native is True
    assert media_root == capsule.root.parent
    # Nested inside the domain, not beside it: the guard walks up to the domain
    # so `view_file` can still reach the capsules.
    assert native_root.parent == capsule.root.parent
    assert native_root != media_root


def test_the_native_guard_entitles_search_and_still_bounds_view_file(
    tmp_path: Path,
) -> None:
    """Run the shipped guard the way agy does: payload on stdin, JSON out."""

    import subprocess
    import sys as _sys

    from finesub.llm.agent.local_agent import AgyLocalAgentDriver

    domain = tmp_path / "domain"
    driver = _agy_driver(tmp_path)
    native_root = driver.native_project_root(domain)
    native_root.mkdir(parents=True, exist_ok=True)
    paths = driver._write_project_resources(native_root, native=True)
    guard = paths["guard"]

    inside = domain / "episode" / "task.json"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("{}", encoding="utf-8")
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("secret", encoding="utf-8")

    def _decide(tool: str, **args) -> str:
        payload = json.dumps({"toolCall": {"name": tool, "args": args}})
        done = subprocess.run(
            [_sys.executable, str(guard)],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(done.stdout)["decision"]

    assert _decide("search_web", query="anything") == "allow"
    assert _decide("read_url_content", Url="https://example.com") == "allow"
    # Reads are still bounded to the *domain*, one level above this project.
    assert _decide("view_file", AbsolutePath=str(inside)) == "allow"
    assert _decide("view_file", AbsolutePath=str(outside)) == "deny"
    # Everything else is still denied by default.
    assert _decide("run_command", Command="whoami") == "deny"
    assert _decide("write_to_file", AbsolutePath=str(inside)) == "deny"


def test_agy_native_search_rows_carry_the_query_and_no_urls(tmp_path: Path) -> None:
    """agy reports the call, not its sources.

    Measured against a real `search_web` run: the stream carries the query and
    no URL anywhere, so the row is emitted with an empty `urls` rather than a
    fabricated one. Native provenance here is thinner than Codex/Claude.
    """

    from finesub.llm.agent.local_agent import _normalize_agy_events

    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            [
                json.dumps({"event": "init", "conversation_id": "c1",
                            "init": {"tools": ["view_file", "search_web"],
                                     "permission_mode": "request-review"}}),
                json.dumps({"event": "step_update", "step_update": {
                    "conversation_id": "c1", "step_type": "tool", "state": "ACTIVE",
                    "tool_name": "search_web",
                    "tool_info": {"name": "search_web",
                                  "parameters": {"query": "who won"}}}}),
                json.dumps({"event": "step_update", "step_update": {
                    "conversation_id": "c1", "step_type": "tool", "state": "DONE",
                    "tool_name": "search_web",
                    "tool_info": {"name": "search_web",
                                  "parameters": {"query": "who won"}}}}),
                json.dumps({"event": "result", "result": {
                    "conversation_id": "c1", "status": "SUCCESS",
                    "response": "Spain", "usage": {}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    normalized, _usage, violations, content = _normalize_agy_events(
        raw, native_search=True, max_bytes=1_000_000
    )

    assert violations == []
    assert content == "Spain"
    searches = [
        row
        for row in normalized
        if row.get("event") == "item.completed"
        and row.get("item_type") == "web_search"
    ]
    # One row for the completed call, not for the ACTIVE half.
    assert len(searches) == 1
    assert searches[0]["query"] == "who won"
    assert searches[0]["tool"] == "search_web"
    assert searches[0]["urls"] == []


def test_agy_driver_verifies_project_and_prepares_audio_inside_episode(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "clip.aac"
    source.write_bytes(b"fake-audio")
    driver = _agy_driver(tmp_path)

    def fake_containerize(input_path, out_path, **_kwargs):
        Path(out_path).write_bytes(Path(input_path).read_bytes() + b"-mp4")
        return Path(out_path)

    monkeypatch.setattr(
        "finesub.media.ffmpeg.containerize_audio_for_agy",
        fake_containerize,
    )

    def retain(*_args) -> None:
        raise OSError("retain for inspection")

    monkeypatch.setattr(driver.capsules, "remove", retain)
    result = driver.run(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "file",
                        "file": {
                            "file_id": "files/clip",
                            "filename": "clip.aac",
                            "format": "audio/aac",
                            "local_path": str(source.resolve()),
                        },
                    },
                ],
            }
        ],
        task="correction",
        profile_id="agent-only",
        reasoning_effort="high",
    )

    assert result.content == "<translated>agy</translated>"
    assert result.usage["cached_input_tokens"] == 4
    # agy reports the conversation's running totals, not this turn's cost.
    # Summing these across a resumed assignment counts earlier turns again.
    assert result.usage["conversation_cumulative"] is True
    assert result.execution_attempt["isolation"]["read_isolation"] == (
        "realpath-bounded FineSub project"
    )
    preparation = result.execution_attempt["input_preparation"]["media"][0]
    assert preparation["mode"] == "audio_single_frame_mp4"
    assert preparation["visual_frames_for_audio"] == 1
    episode = Path(result.execution_attempt["evidence_locator"]["absolute_at_write"])
    rewritten = json.loads((episode / "input" / "messages.json").read_text("utf-8"))
    local_path = Path(rewritten[0]["content"][1]["file"]["local_path"])
    assert local_path.parent == episode / "media"
    assert local_path.read_bytes() == b"fake-audio-mp4"
    state = json.loads(
        (tmp_path / "runtime" / ".agents" / "finesub-project.json").read_text(
            "utf-8"
        )
    )
    assert state["project_id"] == "11111111-2222-3333-4444-555555555555"
    assert len(state["hook_digest"]) == 64


def test_agy_media_preparation_failure_is_retained_but_redacted(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "clip.aac"
    source.write_bytes(b"fake-audio")
    driver = _agy_driver(tmp_path)

    def fail_containerize(*_args, **_kwargs):
        raise RuntimeError("private ffmpeg diagnostic")

    monkeypatch.setattr(
        "finesub.media.ffmpeg.containerize_audio_for_agy",
        fail_containerize,
    )
    with pytest.raises(LocalAgentUnavailableError) as caught:
        driver.run(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "format": "audio/aac",
                                "local_path": str(source.resolve()),
                            },
                        }
                    ],
                }
            ],
            task="correction",
        )

    assert "private ffmpeg diagnostic" not in str(caught.value)
    attempt = caught.value._harness_execution_attempts[0]
    locator = attempt["evidence_locator"]
    assert locator["episode_id"] == attempt["capsule_id"]
    evidence = Path(locator["absolute_at_write"])
    assert "private ffmpeg diagnostic" in (
        evidence / "events" / "stderr.log"
    ).read_text("utf-8")


def test_codex_driver_deletes_successful_episode(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)
    result = driver.run(
        [{"role": "user", "content": "hello"}],
        task="general",
        profile_id="api-replacement",
    )

    assert result.content == "<translated>ok</translated>"
    assert result.episode_id
    assert not (tmp_path / "runtime" / result.episode_id).exists()
    assert "evidence_locator" not in result.execution_attempt
    assert result.execution_attempt["isolation"]["read_isolation"] is False
    assert result.execution_attempt["isolation"]["sandbox_kind"] == (
        "process_read_only"
    )


def test_codex_assignment_scope_persists_and_resumes_exact_handle(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)
    first = driver.run(
        [{"role": "user", "content": "argv"}],
        task="general",
        session_scope="assignment",
        conversation_key="assignment-1:worker-1",
    )
    first_argv = json.loads(first.content)
    assert first.conversation_handle == "11111111-2222-3333-4444-555555555555"
    assert first.turn_identity.startswith("sha256:")
    assert "--ephemeral" not in first_argv

    second = driver.run(
        [{"role": "user", "content": "argv"}],
        task="general",
        session_scope="assignment",
        conversation_key="assignment-1:worker-1",
        conversation_handle=first.conversation_handle,
    )
    second_argv = json.loads(second.content)
    resume = second_argv.index("resume")
    handle = second_argv.index(first.conversation_handle)
    assert resume < handle
    assert second_argv[-1] != first.conversation_handle
    assert second_argv[second_argv.index("--sandbox") + 1] == "read-only"
    assert second.conversation_handle == first.conversation_handle


def test_cleanup_failure_does_not_change_a_successful_result(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    def fail_cleanup(*_args) -> None:
        raise OSError("held by test")

    monkeypatch.setattr(driver.capsules, "remove", fail_cleanup)
    result = driver.run([{"role": "user", "content": "hello"}], task="general")

    assert result.content == "<translated>ok</translated>"
    locator = result.execution_attempt["evidence_locator"]
    assert locator["episode_id"] == result.episode_id
    assert Path(locator["absolute_at_write"]).is_dir()
    assert result.execution_attempt["warnings"] == [
        {
            "event": "episode_cleanup_failed",
            "message": "held by test",
            "evidence_locator": locator,
        }
    ]


def test_codex_driver_applies_per_call_reasoning_effort(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run(
        [{"role": "user", "content": "argv"}],
        task="general",
        reasoning_effort="xhigh",
    )

    argv = json.loads(result.content)
    index = argv.index("--config")
    assert argv[index + 1] == 'model_reasoning_effort="xhigh"'


def test_codex_driver_explicit_reasoning_override_wins_per_call_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(
        tmp_path,
        monkeypatch,
        config_overrides=('model_reasoning_effort="low"',),
    )

    result = driver.run(
        [{"role": "user", "content": "argv"}],
        task="general",
        reasoning_effort="xhigh",
    )

    argv = json.loads(result.content)
    assert argv.count('model_reasoning_effort="low"') == 1
    assert 'model_reasoning_effort="xhigh"' not in argv


def test_completion_rejects_tool_events(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(
        LocalAgentPolicyViolationError, match="command_execution"
    ) as caught:
        driver.run([{"role": "user", "content": "tool"}], task="general")

    attempt = caught.value._harness_execution_attempts[0]
    locator = attempt["evidence_locator"]
    assert locator["episode_id"] == attempt["capsule_id"]
    assert Path(locator["absolute_at_write"]).is_dir()


def test_native_mode_accepts_web_search_event(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)
    result = driver.run(
        [{"role": "user", "content": "search"}],
        task="research",
        native_search=True,
    )

    assert any(event.get("item_type") == "web_search" for event in result.normalized_events)


def test_completion_rejects_web_search_event(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentPolicyViolationError, match="web_search"):
        driver.run([{"role": "user", "content": "search"}], task="general")


def test_native_mode_notes_when_search_is_not_used(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run(
        [{"role": "user", "content": "hello"}],
        task="research",
        native_search=True,
    )

    assert result.execution_attempt["search_events"] == []
    assert result.execution_attempt["notes"] == [
        {
            "event": "native_search_not_used",
            "message": "The native-search target completed without searching.",
        }
    ]


def test_native_mode_does_not_count_started_search_as_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run(
        [{"role": "user", "content": "search-started"}],
        task="research",
        native_search=True,
    )

    assert result.execution_attempt["search_events"] == []
    assert result.execution_attempt["notes"][0]["event"] == "native_search_not_used"


def test_feature_probe_failure_has_execution_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    driver = CodexLocalAgentDriver(
        CodexDriverConfig(
            command=(str(tmp_path / "missing-codex.exe"),),
            runtime_root=tmp_path / "runtime",
        )
    )

    with pytest.raises(LocalAgentUnavailableError) as caught:
        driver.run([{"role": "user", "content": "hello"}], task="general")

    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["backend"] == "local_agent"
    assert attempt["reported_model"] == "configured-default"
    assert attempt["returned_at"]


def test_driver_admits_only_max_parallel_calls_at_once(
    tmp_path: Path, monkeypatch
) -> None:
    import concurrent.futures

    driver = _driver(tmp_path, monkeypatch, max_parallel=2)
    driver.probe()
    live = 0
    peak = 0
    guard = threading.Lock()
    real_popen = local_agent_module.subprocess.Popen

    def counting_popen(*args, **kwargs):
        nonlocal live, peak
        with guard:
            live += 1
            peak = max(peak, live)
        try:
            return real_popen(*args, **kwargs)
        finally:
            time.sleep(0.05)
            with guard:
                live -= 1

    monkeypatch.setattr(local_agent_module.subprocess, "Popen", counting_popen)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        results = list(
            executor.map(
                lambda index: driver.run(
                    [{"role": "user", "content": f"call {index}"}], task="general"
                ),
                range(6),
            )
        )

    assert len(results) == 6
    assert peak <= 2


def test_probe_is_serialized_and_cached_across_threads(
    tmp_path: Path, monkeypatch
) -> None:
    import concurrent.futures

    driver = _driver(tmp_path, monkeypatch)
    real_run = local_agent_module.subprocess.run
    calls = 0

    def counted_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(local_agent_module.subprocess, "run", counted_run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        probes = list(executor.map(lambda _index: driver.probe(), range(8)))

    assert calls == 4  # version + exec help + exec resume help + global help
    assert all(probe is probes[0] for probe in probes)


@pytest.mark.skipif(os.name != "nt", reason="Windows npm shim resolution")
def test_resolver_supports_current_npm_native_binary_layout(
    tmp_path: Path, monkeypatch
) -> None:
    shim = tmp_path / "npm" / "codex.cmd"
    shim.parent.mkdir()
    shim.write_text("@echo off\n", encoding="utf-8")
    native = (
        shim.parent
        / "node_modules"
        / "@openai"
        / "codex"
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    native.parent.mkdir(parents=True)
    native.write_bytes(b"MZ")

    def fake_which(name: str) -> str | None:
        if name == "codex.cmd":
            return str(shim)
        if name == "codex.exe":
            return str(tmp_path / "WindowsApps" / "codex.exe")
        return None

    monkeypatch.setattr(local_agent_module.shutil, "which", fake_which)

    assert local_agent_module._resolve_shell_free_command(("codex",)) == (
        str(native.resolve()),
    )


def test_config_parse_failure_is_permanent_and_points_to_capsule(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(
        LocalAgentPolicyViolationError, match=r"rejected config\.toml \(capsule "
    ) as caught:
        driver.run([{"role": "user", "content": "config-fail"}], task="general")

    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["capsule_id"] in str(caught.value)


def test_model_catalog_schema_failure_is_unavailable_and_points_to_capsule(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(
        LocalAgentUnavailableError, match="cannot decode the current model catalog"
    ) as caught:
        driver.run([{"role": "user", "content": "catalog-fail"}], task="general")

    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["capsule_id"] in str(caught.value)


def test_spawn_failure_has_capsule_execution_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import DriverProbe

    driver = CodexLocalAgentDriver(
        CodexDriverConfig(runtime_root=tmp_path / "runtime", model="gpt-test")
    )
    driver._resolved_command = (str(tmp_path / "vanished-codex.exe"),)
    driver._probe = DriverProbe(
        available=True,
        version="fake",
        structured_events=True,
        no_persisted_session=True,
        no_user_config=True,
        no_user_rules=True,
        can_restrict_tools=True,
        has_web_search=True,
        sandbox_kind="process_read_only",
    )

    with pytest.raises(LocalAgentUnavailableError) as caught:
        driver.run([{"role": "user", "content": "hello"}], task="general")

    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["capsule_id"]
    assert attempt["driver_version"] == "fake"
    assert attempt["reported_model"] == "gpt-test"


def test_capsule_write_failure_keeps_a_locatable_episode(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)
    monkeypatch.setattr(
        local_agent_module,
        "write_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(LocalAgentUnavailableError, match="disk full") as caught:
        driver.run([{"role": "user", "content": "hello"}], task="general")

    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["capsule_id"]
    assert Path(attempt["evidence_locator"]["absolute_at_write"]).is_dir()


def test_timeout_terminates_agent(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch, timeout_seconds=1)

    with pytest.raises(LocalAgentTimeoutError) as caught:
        driver.run([{"role": "user", "content": "sleep"}], task="general")
    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["capsule_id"]
    assert attempt["duration_ms"] >= 900
    assert attempt["return_code"] is not None


def test_stall_watchdog_gives_up_before_the_total_deadline(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(
        tmp_path, monkeypatch, timeout_seconds=30, stall_timeout_seconds=0.5
    )

    with pytest.raises(LocalAgentTimeoutError) as caught:
        driver.run([{"role": "user", "content": "sleep"}], task="general")

    assert "produced no output for" in str(caught.value)
    attempt = caught.value._harness_execution_attempts[0]
    assert attempt["duration_ms"] < 10_000
    assert attempt["stall_timeout_seconds"] == 0.5
    assert attempt["max_event_gap_seconds"] >= 0.5


def test_a_healthy_call_records_its_worst_silence(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "hello"}], task="general")

    # The knob stays off until a real silence distribution justifies a number,
    # but every call still reports the gap that would have to clear it.
    assert result.execution_attempt["stall_timeout_seconds"] == 0.0
    assert result.execution_attempt["max_event_gap_seconds"] >= 0.0


def test_timeout_kills_descendant_process(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch, timeout_seconds=1)

    with pytest.raises(LocalAgentTimeoutError) as caught:
        driver.run([{"role": "user", "content": "child"}], task="general")
    capsule_id = caught.value._harness_execution_attempts[0]["capsule_id"]
    marker = tmp_path / "runtime" / capsule_id / "child-survived.txt"
    time.sleep(2)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows suspended-spawn regression")
def test_windows_job_assignment_has_no_spawn_race(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "race_codex.py"
    script.write_text(RACE_CODEX, encoding="utf-8")
    original_init = local_agent_module._ProcessTree.__init__

    def delayed_assignment(self, process):
        time.sleep(0.25)
        original_init(self, process)

    monkeypatch.setattr(
        local_agent_module._ProcessTree, "__init__", delayed_assignment
    )
    driver = CodexLocalAgentDriver(
        CodexDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            timeout_seconds=1,
        )
    )

    with pytest.raises(LocalAgentTimeoutError) as caught:
        driver.run([{"role": "user", "content": "hello"}], task="race")
    capsule_id = caught.value._harness_execution_attempts[0]["capsule_id"]
    marker = tmp_path / "runtime" / capsule_id / "spawn-race-survived.txt"
    time.sleep(2)
    assert not marker.exists()


def test_result_size_is_bounded(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=20)

    with pytest.raises(LocalAgentPolicyViolationError, match="result size"):
        driver.run([{"role": "user", "content": "large"}], task="general")


@pytest.mark.parametrize("mode", ["stdout-flood", "stderr-flood"])
def test_runtime_output_is_capped(tmp_path: Path, monkeypatch, mode: str) -> None:
    driver = _driver(
        tmp_path,
        monkeypatch,
        max_event_bytes=1024,
        max_stderr_bytes=1024,
    )

    with pytest.raises(LocalAgentPolicyViolationError, match="output limit") as caught:
        driver.run([{"role": "user", "content": mode}], task="general")
    assert caught.value._harness_execution_attempts[0]["capsule_id"]


def test_unknown_item_type_fails_closed(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentPolicyViolationError, match="unknown Codex item"):
        driver.run([{"role": "user", "content": "unknown"}], task="general")


def test_official_todo_list_event_is_accepted(tmp_path: Path, monkeypatch) -> None:
    result = _driver(tmp_path, monkeypatch).run(
        [{"role": "user", "content": "todo"}], task="general"
    )

    assert any(row.get("item_type") == "todo_list" for row in result.normalized_events)


def test_recoverable_item_error_is_retained_when_turn_completes(
    tmp_path: Path, monkeypatch
) -> None:
    result = _driver(tmp_path, monkeypatch).run(
        [{"role": "user", "content": "item-error"}], task="general"
    )

    diagnostic = next(
        row for row in result.normalized_events if row.get("item_type") == "error"
    )
    assert diagnostic["error"] == "optional setting omitted"


def test_failed_turn_cannot_commit_partial_agent_message(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(LocalAgentPolicyViolationError, match="terminal failure"):
        _driver(tmp_path, monkeypatch).run(
            [{"role": "user", "content": "turn-failed"}], task="general"
        )


def test_event_stream_requires_strict_utf8(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(
        b'{"type":"item.completed","item":{"type":"agent_message","text":"bad\xff"}}\n'
    )

    with pytest.raises(LocalAgentPolicyViolationError, match="not UTF-8"):
        _normalize_events(raw, native_search=False, max_bytes=1024)


def test_old_cli_requires_explicit_user_config_opt_in(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "legacy_fake_codex.py"
    script.write_text(
        FAKE_CODEX.replace(" --ignore-user-config --ignore-rules", ""),
        encoding="utf-8",
    )
    driver = CodexLocalAgentDriver(
        CodexDriverConfig(
            command=(sys.executable, str(script)), runtime_root=tmp_path / "runtime"
        )
    )

    with pytest.raises(LocalAgentUnavailableError, match="required execution features"):
        driver.run([{"role": "user", "content": "hello"}], task="general")


def test_requirement_lists_distinguish_completion_from_native_search(
    tmp_path: Path,
) -> None:
    from finesub.llm.agent.local_agent import AgentDriverConfig, DriverProbe, LocalAgentDriver

    driver = LocalAgentDriver(AgentDriverConfig(runtime_root=tmp_path / "runtime"))
    completion_only = DriverProbe(
        available=True,
        structured_events=True,
        no_persisted_session=True,
        no_user_config=True,
        no_user_rules=True,
        can_restrict_tools=True,
        has_web_search=False,
        sandbox_kind="process_read_only",
    )

    assert driver.meets_requirements(completion_only)
    assert not driver.meets_requirements(completion_only, native_search=True)


def test_config_override_allowlist_rejects_tool_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX, encoding="utf-8")
    driver = CodexLocalAgentDriver(
        CodexDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            config_overrides=("mcp_servers={}",),
        )
    )

    with pytest.raises(LocalAgentPolicyViolationError, match="not allowlisted"):
        driver.run([{"role": "user", "content": "hello"}], task="general")


def test_runtime_root_inside_repository_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()

    with pytest.raises(LocalAgentPolicyViolationError):
        AgentCapsuleManager(tmp_path / "out").resolve_location()


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


def _episode(parent: Path, name: str, *, age_seconds: float) -> Path:
    root = parent / name
    (root / "events").mkdir(parents=True)
    (root / "events" / "stderr.log").write_text("boom", encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(root, (stamp, stamp))
    return root


def test_a_domain_keeps_only_the_newest_retained_episodes(tmp_path: Path) -> None:
    """Failed capsules are never deleted by their own call, so they accumulate.

    Evidence loses value oldest-first -- the failure someone is investigating
    is the one that just happened -- so the cap drops the oldest.
    """

    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()
    location.parent.mkdir(parents=True, exist_ok=True)
    old_day = 86_400.0
    for index in range(RETAINED_FAILED_EPISODES + 5):
        _episode(location.parent, f"research-{index:03d}", age_seconds=old_day + index)

    removed = manager.prune(location, timeout_seconds=900)

    survivors = sorted(p.name for p in location.parent.iterdir() if p.is_dir())
    assert len(survivors) == RETAINED_FAILED_EPISODES
    assert len(removed) == 5
    # Oldest-first: the five highest indices carry the oldest mtimes.
    assert set(removed) == {f"research-{i:03d}" for i in range(20, 25)}


def test_pruning_never_touches_a_capsule_young_enough_to_be_running(
    tmp_path: Path,
) -> None:
    """The newest-N guard alone breaks down if one domain exceeds N in flight.

    A call is killed at `timeout_seconds`, so anything younger than twice that
    may still have a live owner and is left alone however many there are.
    """

    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()
    location.parent.mkdir(parents=True, exist_ok=True)
    for index in range(RETAINED_FAILED_EPISODES + 5):
        _episode(location.parent, f"research-{index:03d}", age_seconds=60.0)

    assert manager.prune(location, timeout_seconds=900) == []
    assert len(list(location.parent.iterdir())) == RETAINED_FAILED_EPISODES + 5


def test_pruning_leaves_conversation_and_agy_project_state_alone(
    tmp_path: Path,
) -> None:
    """Both live beside the episodes and neither is evidence.

    `.conversations` is assignment-scope working state; `.agents` is agy's
    controlled project, whose hook is the read boundary -- deleting it would
    break a live run rather than free anything.
    """

    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()
    location.parent.mkdir(parents=True, exist_ok=True)
    for index in range(RETAINED_FAILED_EPISODES + 3):
        _episode(location.parent, f"research-{index:03d}", age_seconds=86_400.0)
    for reserved in (".conversations", ".agents"):
        kept = location.parent / reserved / "deep"
        kept.mkdir(parents=True)
        stamp = time.time() - 86_400.0
        os.utime(location.parent / reserved, (stamp, stamp))

    manager.prune(location, timeout_seconds=900)

    assert (location.parent / ".conversations" / "deep").is_dir()
    assert (location.parent / ".agents" / "deep").is_dir()


def test_pruning_never_follows_a_link_out_of_the_domain(tmp_path: Path) -> None:
    """`create` makes episodes with mkdtemp, so a link is never one of ours.

    Following one would delete whatever it points at -- and redirecting a big
    directory off the system drive with a junction is a documented setup, so
    this is not a hypothetical shape for a machine to be in.

    **This passes on 3.12 even with the guard removed**, because that is the
    release where `shutil.rmtree` stopped recursing into junctions. The guard
    earns its keep on 3.10/3.11, which the CLI wheel still supports and which
    this suite does not run -- so do not read a green test here as evidence
    that skipping links is redundant. Same reasoning as `fsops.remove_tree`.
    """

    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()
    location.parent.mkdir(parents=True, exist_ok=True)
    for index in range(RETAINED_FAILED_EPISODES + 3):
        _episode(location.parent, f"research-{index:03d}", age_seconds=86_400.0)

    outside = tmp_path / "somebody-elses-data"
    outside.mkdir()
    (outside / "keep.txt").write_text("precious", encoding="utf-8")
    link = location.parent / "research-999"
    # Backdate the *target*: `stat()` follows the link, so this is what the
    # pruner sees, and it makes the link the oldest candidate of all -- the
    # link guard has to be what saves it, not the newest-N or age rules.
    stamp = time.time() - 86_400.0 * 30
    os.utime(outside, (stamp, stamp))
    _make_directory_link(link, outside)

    manager.prune(location, timeout_seconds=900)

    assert (outside / "keep.txt").read_text(encoding="utf-8") == "precious"
    assert outside.is_dir()


def test_a_new_episode_prunes_before_it_grows_the_domain(tmp_path: Path) -> None:
    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()
    location.parent.mkdir(parents=True, exist_ok=True)
    for index in range(RETAINED_FAILED_EPISODES + 4):
        _episode(location.parent, f"research-{index:03d}", age_seconds=86_400.0)

    capsule, _payload = manager.create(
        location,
        [{"role": "user", "content": "hi"}],
        task="research",
        native_search=False,
        profile_id="test",
        max_result_bytes=1024,
        timeout_seconds=900,
    )

    # Pruned to the cap, then the new one -- never N+1 even mid-call.
    directories = [p for p in location.parent.iterdir() if p.is_dir()]
    assert len(directories) == RETAINED_FAILED_EPISODES + 1
    assert capsule.root.is_dir()


def test_provider_secrets_are_not_inherited(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch)
    result = driver.run([{"role": "user", "content": "env"}], task="general")

    assert result.content == "False"


# --- Claude Code driver ----------------------------------------------------
#
# Same transport, same capsule guarantees, different CLI dialect. These pin
# the four things that actually differ: the argv (isolation + tool denial),
# the event dialect, how a clean-exit runtime failure is classified, and that
# the shared secret-stripping still applies.


def test_claude_driver_returns_content_and_usage(tmp_path: Path, monkeypatch) -> None:
    driver = _claude_driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "hi"}], task="correction")

    assert result.content == "<translated>ok</translated>"
    assert result.usage["input_tokens"] == 10
    assert result.usage["output_tokens"] == 3
    assert result.usage["source"] == "claude_code_result_event"
    assert result.execution_attempt["driver"] == "claude-code"
    isolation = result.execution_attempt["isolation"]
    assert isolation["sandbox_kind"] == "named_tool_denylist"
    assert isolation["write_restriction"] == "named write/execute tool denylist"
    assert not (tmp_path / "runtime" / result.episode_id).exists()


def test_claude_argv_isolates_and_denies_write_tools(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import CLAUDE_ALL_TOOLS

    driver = _claude_driver(
        tmp_path, monkeypatch, effort="high", max_result_bytes=8192
    )
    result = driver.run([{"role": "user", "content": "argv"}], task="correction")
    args = json.loads(result.content)

    assert "--print" in args
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--no-session-persistence" in args
    assert "--disable-slash-commands" in args
    assert "--strict-mcp-config" in args
    # `--safe-mode` is this CLI's "ignore user config and rules": CLAUDE.md,
    # hooks, plugins, MCP and custom commands all off, auth left working.
    assert "--safe-mode" in args
    assert args[args.index("--setting-sources") + 1] == ""
    assert args[args.index("--model") + 1] == "claude-opus-5"
    assert args[args.index("--effort") + 1] == "high"
    # A completion call is entitled to nothing at all, so every known tool --
    # retrieval included -- is denied up front rather than merely watched.
    # `--allowed-tools` is not used: it grants permission, it does not remove
    # a tool, so it cannot narrow the session.
    # One comma-joined argument, not a variadic list: with no model or effort
    # configured the prompt would otherwise be parsed as another tool name.
    denied = set(args[args.index("--disallowed-tools") + 1].split(","))
    assert denied == set(CLAUDE_ALL_TOOLS)
    assert "--allowed-tools" not in args
    # Nothing points the agent at files it cannot open.
    assert "--add-dir" not in args


def test_claude_native_call_permits_only_the_search_tools(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import CLAUDE_ALL_TOOLS, CLAUDE_SEARCH_TOOLS

    driver = _claude_driver(tmp_path, monkeypatch, max_result_bytes=8192)
    result = driver.run(
        [{"role": "user", "content": "argv"}], task="research", native_search=True
    )
    args = json.loads(result.content)

    denied = set(args[args.index("--disallowed-tools") + 1].split(","))
    assert not (CLAUDE_SEARCH_TOOLS & denied)
    assert denied == set(CLAUDE_ALL_TOOLS - CLAUDE_SEARCH_TOOLS)
    assert set(args[args.index("--allowed-tools") + 1].split(",")) == set(
        CLAUDE_SEARCH_TOOLS
    )


def test_claude_completion_rejects_a_tool_call(tmp_path: Path, monkeypatch) -> None:
    driver = _claude_driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentPolicyViolationError, match="forbidden tool Bash"):
        driver.run([{"role": "user", "content": "tool"}], task="correction")


def test_claude_session_offering_denied_tools_warns_but_answers(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An offered-but-unused tool is news, not a failure.

    `--disallowed-tools` denies by name, so a tool a newer CLI adds is offered
    until someone extends CLAUDE_ALL_TOOLS. Failing the call over that turned a
    routine CLI upgrade into a *permanent* route failure -- which skips the
    whole API fallback chain -- for a tool nothing had touched. What refuses is
    the event-stream check on tools actually invoked, which is the precise
    version of the same question; see
    ``test_claude_completion_rejects_a_search_event``.
    """

    driver = _claude_driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "leak"}], task="correction")

    assert result.content
    messages = [
        str(row.get("message") or "")
        for row in result.execution_attempt.get("warnings") or []
    ]
    assert any("not entitled" in message for message in messages)
    assert "Warning:" in capsys.readouterr().err


def test_claude_native_mode_records_the_search_event(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _claude_driver(tmp_path, monkeypatch)

    result = driver.run(
        [{"role": "user", "content": "search"}], task="research", native_search=True
    )

    events = result.execution_attempt["search_events"]
    assert events and events[0]["item_type"] == "web_search"
    assert events[0]["query"] == "FineSub test"


def test_claude_completion_rejects_a_search_event(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _claude_driver(tmp_path, monkeypatch)

    with pytest.raises(
        LocalAgentPolicyViolationError, match="forbidden tool WebSearch"
    ):
        driver.run([{"role": "user", "content": "search"}], task="correction")


def test_claude_auth_failure_is_unavailable_not_permanent(
    tmp_path: Path, monkeypatch
) -> None:
    """An expired login must let the route fall back to the API chain.

    Claude Code reports it inside a clean exit, so without explicit
    classification it would surface as a policy violation -- permanent, and
    the run would fail instead of using Gemini.
    """

    driver = _claude_driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="not authenticated"):
        driver.run([{"role": "user", "content": "auth-fail"}], task="correction")


def test_claude_provider_error_is_transient(tmp_path: Path, monkeypatch) -> None:
    from finesub.llm.agent.local_agent import LocalAgentTransientError

    driver = _claude_driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentTransientError, match="provider error"):
        driver.run([{"role": "user", "content": "api-fail"}], task="correction")


def test_claude_child_environment_has_no_secrets(tmp_path: Path, monkeypatch) -> None:
    driver = _claude_driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "env"}], task="correction")

    assert result.content == "False"


def test_provider_tier_picks_the_driver() -> None:
    """The tier is the only thing that selects a CLI, so both the readiness
    pre-filter and dispatch land on the same one for a given candidate."""

    from finesub.llm.routing.execution_policy import ExecutionSettings, driver_for_provider_tier
    from finesub.llm.agent.local_agent import ClaudeCodeLocalAgentDriver, CodexLocalAgentDriver

    settings = ExecutionSettings()
    codex = driver_for_provider_tier(
        settings, provider_tier="LOCAL_CODEX", model="gpt-5.6-luna"
    )
    claude = driver_for_provider_tier(
        settings, provider_tier="LOCAL_CLAUDE", model="claude-opus-5"
    )

    assert isinstance(codex, CodexLocalAgentDriver)
    assert isinstance(claude, ClaudeCodeLocalAgentDriver)
    assert claude.config.model == "claude-opus-5"
    assert claude.driver_id == "claude-code"


def test_claude_prompt_survives_a_bare_argv(tmp_path: Path, monkeypatch) -> None:
    """No model, no effort, no native search -- the prompt must still arrive.

    `--disallowed-tools` is variadic, so a space-separated list would absorb
    the positional prompt that follows it and the agent would be handed the
    task with no instructions at all, silently.
    """

    from finesub.llm.agent.local_agent import CLAUDE_ALL_TOOLS

    driver = _claude_driver(
        tmp_path, monkeypatch, model="", effort="", max_result_bytes=8192
    )
    result = driver.run([{"role": "user", "content": "argv"}], task="correction")
    args = json.loads(result.content)

    assert "--model" not in args and "--effort" not in args
    assert args[-1].startswith("You are a FineSub text execution backend")
    assert set(args[args.index("--disallowed-tools") + 1].split(",")) == set(
        CLAUDE_ALL_TOOLS
    )


def test_claude_prompt_does_not_promise_files_it_cannot_read(
    tmp_path: Path, monkeypatch
) -> None:
    """Codex reads the capsule; this CLI has every read tool denied."""

    from finesub.llm.agent.local_agent import (
        AGENT_TASK_PROMPT_READABLE_CAPSULE,
        AGENT_TASK_PROMPT_STDIN_ONLY,
    )

    driver = _claude_driver(tmp_path, monkeypatch, max_result_bytes=8192)
    args = json.loads(
        driver.run([{"role": "user", "content": "argv"}], task="correction").content
    )

    assert args[-1] == AGENT_TASK_PROMPT_STDIN_ONLY
    assert "input/messages.json" not in args[-1]
    assert "input/messages.json" in AGENT_TASK_PROMPT_READABLE_CAPSULE


def test_claude_unknown_future_tool_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch
) -> None:
    """A tool nobody enumerated is surfaced, so the list can be extended.

    This is not hypothetical: a name-based deny list of the obvious
    write/execute tools still left this CLI offering Artifact, CronCreate,
    RemoteTrigger and the Task family. What it must not do is pass unnoticed --
    if the model actually invokes it, the tool_use check refuses the call.
    """

    from finesub.llm.agent.local_agent import _normalize_claude_events

    stream = tmp_path / "events.jsonl"
    stream.write_text(
        json.dumps({"type": "system", "subtype": "init", "tools": ["FutureTool"]})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            }
        )
        + "\n"
        + json.dumps(
            {"type": "result", "subtype": "success", "is_error": False, "usage": {}}
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _usage, violations, content = _normalize_claude_events(
        stream, native_search=False, max_bytes=1 << 20
    )

    assert content == "hi"
    assert not violations
    assert any("FutureTool" in (row.get("unentitled_tools_offered") or ()) for row in rows)

    from finesub.llm.agent.local_agent import ClaudeCodeLocalAgentDriver

    warnings = ClaudeCodeLocalAgentDriver()._stream_warnings(rows)
    assert warnings and "FutureTool" in warnings[0]


def test_claude_counts_every_search_in_one_message(tmp_path: Path, monkeypatch) -> None:
    """A message may carry several tool calls; folding them into one row would
    under-report what a native round actually ran."""

    from finesub.llm.agent.local_agent import _normalize_claude_events

    stream = tmp_path / "events.jsonl"
    stream.write_text(
        json.dumps({"type": "system", "subtype": "init", "tools": ["WebSearch", "WebFetch"]})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "server_tool_use", "name": "WebSearch", "input": {"query": "a"}},
                        {"type": "server_tool_use", "name": "WebSearch", "input": {"query": "b"}},
                        {"type": "text", "text": "done"},
                    ],
                },
            }
        )
        + "\n"
        + json.dumps(
            {"type": "result", "subtype": "success", "is_error": False, "usage": {}}
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _usage, violations, content = _normalize_claude_events(
        stream, native_search=True, max_bytes=1 << 20
    )

    assert violations == []
    assert content == "done"
    searches = [
        row for row in rows if row.get("item_type") == "web_search"
    ]
    assert [row["query"] for row in searches] == ["a", "b"]


def test_harvest_urls_reads_the_structure_not_its_json() -> None:
    """Scanning `json.dumps(...)` glued the next line onto every URL.

    In serialized form a newline is the two characters `\\n`, which the URL
    pattern treats as ordinary path characters -- so a live native round
    recorded `https://docs.anthropic.com/\\n-` as provenance.
    """

    from finesub.llm.agent.local_agent import _harvest_urls

    block = {
        "type": "tool_result",
        "content": "Links:\nhttps://docs.anthropic.com/\n- next line\n",
        "nested": [{"u": "see [docs](https://example.test/a) and https://example.test/b."}],
    }

    assert _harvest_urls(block) == [
        "https://docs.anthropic.com/",
        "https://example.test/a",
        "https://example.test/b",
    ]
    # Balanced parens belong to the address; an unmatched one does not.
    assert _harvest_urls("https://en.wikipedia.org/wiki/Foo_(bar)") == [
        "https://en.wikipedia.org/wiki/Foo_(bar)"
    ]


def test_claude_search_rows_take_urls_from_the_result_message(tmp_path: Path) -> None:
    """The call carries only the query; the URLs arrive on the next message.

    Without joining them on tool_use_id the native path records searches with
    no provenance at all, which is thinner than the Codex path for no reason.
    """

    from finesub.llm.agent.local_agent import _normalize_claude_events

    stream = tmp_path / "events.jsonl"
    stream.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"type": "system", "subtype": "init", "tools": ["WebSearch", "WebFetch"]},
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "WebSearch",
                                "input": {"query": "finesub"},
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Links: https://example.test/hit\nmore",
                            }
                        ],
                    },
                },
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                    },
                },
                {"type": "result", "subtype": "success", "is_error": False, "usage": {}},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _usage, violations, content = _normalize_claude_events(
        stream, native_search=True, max_bytes=1 << 20
    )

    assert violations == [] and content == "done"
    search = [row for row in rows if row.get("item_type") == "web_search"]
    assert len(search) == 1
    assert search[0]["query"] == "finesub"
    assert search[0]["urls"] == ["https://example.test/hit"]


def test_unknown_provider_tier_fails_closed() -> None:
    """A tier with no registered driver is a refusal, not a Codex call.

    Defaulting to Codex meant a typo in a catalog row would silently run the
    task on another vendor's CLI, and the answer would look perfectly normal.
    """

    from finesub.llm.routing.execution_policy import ExecutionSettings, driver_for_provider_tier

    with pytest.raises(ValueError, match="No local-agent driver"):
        driver_for_provider_tier(
            ExecutionSettings(), provider_tier="LOCAL_TYPO", model="x"
        )


def test_driver_cache_is_keyed_by_tier_and_model() -> None:
    """Two tiers sharing a model id must not share a driver.

    The cache was keyed on the model alone, so whichever tier asked first won
    and the second silently reused the wrong vendor's CLI -- reachable as soon
    as a third agent tier uses a model id that looks like someone else's.
    """

    from finesub.llm.client import RoleClient
    from finesub.llm.routing.model_router import ModelRouter
    from finesub.llm.rate_limit import ModelRateLimiter

    client = RoleClient(
        router=ModelRouter(), rate_limiter=ModelRateLimiter(enabled=False)
    )

    codex = client._local_driver_for_model("shared-model-id", "LOCAL_CODEX")
    claude = client._local_driver_for_model("shared-model-id", "LOCAL_CLAUDE")

    assert codex.driver_id == "codex"
    assert claude.driver_id == "claude-code"
    # ...and each is still cached, so the probe is not repeated per call.
    assert client._local_driver_for_model("shared-model-id", "LOCAL_CODEX") is codex


def test_every_session_is_recorded_where_cleanup_cannot_reach_it(
    tmp_path: Path, monkeypatch
) -> None:
    """A durable cross-reference for the vendor's own history.

    Every other record of a session dies with the thing holding it: a
    successful call's capsule is deleted immediately, and exchange files go
    with the run's artifacts. The CLI's own transcripts outlive all of it, so
    without this list there is no way to look at `~/.claude/projects/...` and
    tell which sessions FineSub created.
    """

    from finesub.llm.agent.agent_paths import SESSION_LEDGER_NAME

    driver = _claude_driver(tmp_path, monkeypatch)
    location = driver.capsules.resolve_location()
    ledger = location.activity_root / SESSION_LEDGER_NAME

    driver.run([{"role": "user", "content": "hi"}], task="correction-mm")

    rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["driver"] == "claude-code"
    assert rows[0]["task"] == "correction-mm"
    assert rows[0]["session_id"]
    assert rows[0]["episode_id"]

    # It sits beside config.toml in the activity root, which neither
    # `agent-clean` nor an ordinary uninstall removes -- unlike the capsule
    # parent, which both do.
    assert ledger.parent == location.activity_root
    assert ledger.parent != location.parent


def test_a_failed_call_is_attributable_too(tmp_path: Path, monkeypatch) -> None:
    """The failure paths all used to raise before the id was ever read.

    That is exactly backwards: a call that failed is the one whose vendor-side
    transcript someone will want to open.
    """

    from finesub.llm.agent.agent_paths import SESSION_LEDGER_NAME

    driver = _claude_driver(tmp_path, monkeypatch)
    location = driver.capsules.resolve_location()

    with pytest.raises(LocalAgentError):
        driver.run([{"role": "user", "content": "search"}], task="correction")

    ledger = location.activity_root / SESSION_LEDGER_NAME
    rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows and rows[-1]["session_id"]
