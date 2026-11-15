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
    AGENT_USER_RULES_NOTICE,
    RETAINED_FAILED_EPISODES,
    RETAINED_UNACCEPTED_EPISODES,
    UNACCEPTED_EPISODE_MARKER,
    AgyDriverConfig,
    AgyLocalAgentDriver,
    AgentCapsuleManager,
    CodexDriverConfig,
    CodexLocalAgentDriver,
    LocalAgentError,
    LocalAgentPolicyViolationError,
    LocalAgentTimeoutError,
    LocalAgentTransientError,
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
if args[:2] == ["features", "list"]:
    print("apps  stable  true\nplugins  stable  true\nremote_plugin  under development  false\napply_patch_freeform  removed  false")
    raise SystemExit(0)
payload = sys.stdin.buffer.read()
print(json.dumps({"type":"thread.started","thread_id":"11111111-2222-3333-4444-555555555555"}))
mode = "ok"
for marker in ("sleep", "child", "tool", "search", "search-started", "fail", "config-fail", "catalog-fail", "large", "env", "envdump", "argv", "unknown", "todo", "item-error", "turn-failed", "stdout-flood", "stderr-flood", "stale-cache-turn-failed"):
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
if args[:2] == ["features", "list"]:
    # The driver's feature probe: answered here, or the race below runs in it
    # instead -- outside any job, in the test's own cwd -- and the call that
    # is supposed to race starts 15 s late.
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
        keep_episodes=overrides.get("keep_episodes", False),
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
            effort="high",
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


def test_every_agy_spawn_switches_its_update_check_off(monkeypatch) -> None:
    """agy's update check runs `agy --version` in a detached console that
    `no_window()` cannot hide; only the environment switch keeps it away.
    The model call, the probe and the `/hooks` check all carry it."""

    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))
    for env in (driver._spawn_environment(), driver._spawn_environment(native_search=True)):
        assert env["AGY_CLI_DISABLE_AUTO_UPDATE"] == "true"

    seen: list[dict[str, str]] = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, "1.2.13", "")

    monkeypatch.setattr(local_agent_module, "_resolve_shell_free_command", lambda command: ("agy",))
    monkeypatch.setattr(local_agent_module.subprocess, "run", fake_run)
    driver._probe_driver()
    driver._hooks_command(Path("."), create=True)
    assert len(seen) == 3  # version, help, /hooks
    assert all(env["AGY_CLI_DISABLE_AUTO_UPDATE"] == "true" for env in seen)


def test_agy_sends_every_model_an_effort_it_has(tmp_path: Path) -> None:
    """agy's Claude models refuse to start without `--effort` (verified
    2026-10-03, agy 1.2.16):

        invalid model selection (--model "claude-sonnet-5-5" --effort ""):
        --model claude-sonnet-5-5 requires --effort (available: low, medium, high)

    and agy has no `xhigh`, which `[llm].local_agent_reasoning_effort` may
    force on every driver: it goes out as `high`.
    """

    from dataclasses import replace as dataclass_replace

    driver = _agy_driver(tmp_path)
    capsule = SimpleNamespace(
        root=tmp_path / "capsule" / "episode",
        messages_path=tmp_path / "capsule" / "episode" / "task.json",
        continuation_of="",
    )
    _agy_task_parts(capsule.root)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")

    def sent(model: str, reasoning_effort: str, forced: str = "") -> str | None:
        driver.config = dataclass_replace(driver.config, model=model, effort=forced)
        argv = driver._argv(
            capsule,
            native_search=False,
            probe=SimpleNamespace(),
            reasoning_effort=reasoning_effort,
        )
        return argv[argv.index("--effort") + 1] if "--effort" in argv else None

    assert sent("claude-sonnet-5-5", "medium") == "medium"
    assert sent("claude-opus-5-5", "high") == "high"
    assert sent("gemini-3.7-flash", "low") == "low"
    assert sent("claude-sonnet-5-5", "medium", forced="xhigh") == "high"
    assert sent("gemini-3.7-flash", "") is None


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
    """agy is the one driver whose input is a path, not stdin: the errors
    ride the task parts, last, and the prompt says so -- a repair it is not
    told about is one it never reads."""

    driver = _agy_driver(tmp_path)
    capsule = SimpleNamespace(
        root=tmp_path / "domain" / "episode",
        messages_path=tmp_path / "domain" / "episode" / "task.json",
        continuation_of="",
    )
    _agy_task_parts(capsule.root)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")

    plain = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
    assert "failed the harness output contract" not in plain[plain.index("--print") + 1]

    errors_path = capsule.root / "input" / "validation-errors.txt"
    errors_path.write_text("Row 1 references unknown source id 3.", encoding="utf-8")
    repair = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
    prompt = repair[repair.index("--print") + 1]

    assert "the errors are at the end of the task files" in prompt
    # The answer itself is already in the resumed conversation; naming the file
    # would only pay for a second copy of it.
    assert "previous-output.txt" not in prompt
    text = local_agent_module.agy_task_text(
        [{"role": "user", "content": "fix it"}], errors_path.read_text(encoding="utf-8")
    )
    assert text.endswith("fix it\n@@ harness validation errors\nRow 1 references unknown source id 3.\n")


def test_a_repair_instruction_replaces_the_stock_wording(tmp_path: Path) -> None:
    """The correction window's patch round asks for part of the answer: the
    capsule carries its wording instead of "a complete result", and agy -- whose
    session then holds only a patch -- is pointed at the previous answer too."""

    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()

    def create(**repair):
        return manager.create(
            location, [{"role": "user", "content": "hi"}], task="correction",
            native_search=False, profile_id="test", max_result_bytes=1024,
            timeout_seconds=900, previous_output="<translated>…</translated>",
            validation_errors=["Row 1 is empty."], **repair,
        )

    stock_capsule, stock = create()
    assert "not a diff" in json.loads(stock)["repair_instruction"]
    assert not (stock_capsule.root / "input" / local_agent_module.REPAIR_INSTRUCTION_FILE).exists()

    capsule, payload = create(repair_instruction="只输出 <patch> 块")
    assert json.loads(payload)["repair_instruction"] == "只输出 <patch> 块"
    instruction = capsule.root / "input" / local_agent_module.REPAIR_INSTRUCTION_FILE
    assert instruction.read_text(encoding="utf-8") == "只输出 <patch> 块"

    driver = _agy_driver(tmp_path)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")

    def prepared(capsule):
        driver._prepare_capsule_input(capsule, b"")
        task = capsule.root.joinpath(*local_agent_module.AGY_TASK_PARTS_DIR)
        text = "".join(path.read_text(encoding="utf-8") for path in sorted(task.glob("part-*.md")))
        argv = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
        return text, argv[argv.index("--print") + 1]

    # The patch round: the parts carry the answer the patch is laid over and
    # the instruction (which holds the errors), read in the first step.
    text, prompt = prepared(capsule)
    assert text.index("@@ your previous answer\n<translated>…</translated>") < text.index(
        "@@ harness repair instruction\n只输出 <patch> 块")
    assert "@@ harness validation errors" not in text
    assert "at the end of the task files" in prompt and "not a diff" not in prompt
    text, prompt = prepared(stock_capsule)
    assert text.endswith("@@ harness validation errors\nRow 1 is empty.\n")
    assert "@@ your previous answer" not in text and "not a diff" in prompt


def test_agy_task_text_keeps_every_message_raw() -> None:
    text = local_agent_module.agy_task_text([
        {"role": "system", "content": "RULES \"quoted\"\n## a heading"},
        {"role": "user", "content": [
            {"type": "text", "text": "台詞"},
            {"type": "file", "file": {"format": "video/mp4", "local_path": "C:/m/attachment-01.mp4"}},
            {"type": "other", "value": 1},
        ]},
    ])
    assert text == (
        '@@ message 1 · system\nRULES "quoted"\n## a heading\n'
        "@@ message 2 · user\n台詞\n@@ attachment · video/mp4 · C:/m/attachment-01.mp4\n"
        '{"type": "other", "value": 1}\n'
    )


def test_agy_names_every_task_part_or_a_range_within_its_argv_bound(tmp_path: Path) -> None:
    driver = _agy_driver(tmp_path)
    root = tmp_path / ("d" * 120) / "episode"
    task = _agy_task_parts(root, count=200)
    capsule = SimpleNamespace(root=root, messages_path=root / "x.json", continuation_of="")
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")
    prompt = driver._argv(capsule, native_search=False, probe=SimpleNamespace())[2]
    assert len(prompt.encode("utf-16-le")) // 2 <= driver.config.channel.opening_max_chars
    assert prompt.endswith(f"{task.resolve()}{os.sep}part-01.md to part-200.md (200 files, numbered in order)")
    few = tmp_path / "few" / "episode"
    _agy_task_parts(few, count=3)
    listed = driver._argv(SimpleNamespace(root=few, messages_path=few / "x.json", continuation_of=""),
                          native_search=False, probe=SimpleNamespace())[2]
    assert listed.endswith("\n".join(str((few / "input" / "task" / f"part-0{i}.md").resolve()) for i in (1, 2, 3)))


FIRST_STEP = "In your very first step, open every task file together with the attachments they name, at "


def _agy_media_call(
    tmp_path: Path, monkeypatch, *, conversation_handle: str = "", media: int = 2
):
    """One agy capsule call carrying ``media`` prepared videos: its argv prompt,
    its execution attempt and the ``local_path`` values of the rewritten task file."""

    clips = []
    for name in ("a", "b")[:media]:
        clip = tmp_path / f"{name}.mp4"
        clip.write_bytes(name.encode())
        clips.append(clip)
    driver = _agy_driver(tmp_path)

    def retain(*_args) -> None:
        raise OSError("retain for inspection")

    monkeypatch.setattr(driver.capsules, "remove", retain)
    monkeypatch.setattr("finesub.media.ffmpeg.agy_media_cuts", lambda *_args, **_kwargs: [])  # short clips
    prompts = []
    real_argv = driver._argv

    def recording_argv(*args, **kwargs):
        argv = real_argv(*args, **kwargs)
        prompts.append(argv[argv.index("--print") + 1])
        return argv

    monkeypatch.setattr(driver, "_argv", recording_argv)
    result = driver.run(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    *(
                        {
                            "type": "file",
                            "file": {
                                "file_id": clip.name,
                                "filename": clip.name,
                                "format": "video/mp4",
                                "local_path": str(clip.resolve()),
                                "agy_prepared": True,
                            },
                        }
                        for clip in clips
                    ),
                ],
            }
        ],
        task="correction",
        # The fake CLI reads the task from its cwd, the capsule of a task-scoped
        # call; the handle alone is what the first-step decision looks at.
        conversation_handle=conversation_handle,
    )
    episode = Path(result.execution_attempt["evidence_locator"]["absolute_at_write"])
    rewritten = json.loads((episode / "input" / "messages.json").read_text("utf-8"))
    local_paths = [
        block["file"]["local_path"] for block in rewritten[0]["content"] if block["type"] == "file"
    ]
    return prompts[-1], result.execution_attempt, local_paths


def test_a_fresh_agy_media_call_opens_the_task_file_and_media_in_one_step(
    tmp_path: Path, monkeypatch
) -> None:
    """agy drops every cache entry written before a session's first media, so
    reading the task file and then the audio pays for the task file twice."""

    prompt, attempt, local_paths = _agy_media_call(tmp_path, monkeypatch)

    assert len(local_paths) == 2
    # The very strings the task file names, in attachment order: the prompt's
    # "view_file only on the local_path named in the message" still holds.
    assert f"{FIRST_STEP}`{local_paths[0]}`, `{local_paths[1]}`, with parallel view_file calls" \
        " in that same step, each whole (no line range)." in prompt
    preparation = attempt["input_preparation"]
    assert preparation["first_step_media"] == [row["path"] for row in preparation["media"]]


def test_an_agy_capsule_call_opens_its_task_parts_in_the_first_step(tmp_path: Path, monkeypatch) -> None:
    prompt, attempt, _local_paths = _agy_media_call(tmp_path, monkeypatch, media=0)

    episode = Path(attempt["evidence_locator"]["absolute_at_write"])
    part = episode / "input" / "task" / "part-01.md"
    assert prompt == (
        local_agent_module.AGENT_TASK_PROMPT_AGY_MEDIA
        + " In your very first step, open every task file, with parallel view_file calls in that"
        " same step, each whole (no line range).\nThe task files, in order:\n"
        + str(part.resolve())
    )
    assert part.read_text(encoding="utf-8") == "@@ message 1 · user\ntranscribe\n"
    assert attempt["input_preparation"]["task_parts"] == ["part-01.md"]
    assert "first_step_media" not in attempt["input_preparation"]


def test_a_resumed_agy_call_does_not_reopen_its_media(tmp_path: Path, monkeypatch) -> None:
    """Its media is already in the conversation (a continuation carries none)."""

    prompt, attempt, _local_paths = _agy_media_call(
        tmp_path, monkeypatch, conversation_handle="agy-conv"
    )

    assert FIRST_STEP not in prompt
    assert "first_step_media" not in attempt["input_preparation"]


def test_the_first_step_sentence_sits_with_the_repair_and_native_clauses(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _agy_driver(tmp_path)
    root = tmp_path / "domain" / "episode"
    (root / "input").mkdir(parents=True)
    media = str((root / "media" / "attachment-01.mp4").resolve())
    messages_path = root / "input" / "messages.json"
    messages_path.write_text(json.dumps([{"role": "user", "content": [
        {"type": "file", "file": {"local_path": media, "agy_prepared": True}},
    ]}]), encoding="utf-8")
    (root / "input" / "validation-errors.txt").write_text("Row 1 is empty.", encoding="utf-8")
    capsule = SimpleNamespace(root=root, messages_path=messages_path, continuation_of="")
    _agy_task_parts(root)
    driver._resolved_command = ("agy",)
    driver._ensure_project = lambda _root, native=False: ("project-id", "digest")
    driver._grant_permissions = lambda *_args, **_kwargs: None

    def prompt_of(**kwargs) -> str:
        argv = driver._argv(capsule, probe=SimpleNamespace(), **kwargs)
        return argv[argv.index("--print") + 1]

    # String composition only: a fresh repair never has an errors file (agy
    # declines repair context outside its own conversation).
    fresh = prompt_of(native_search=True)
    assert local_agent_module.AGENT_TASK_PROMPT_AGY_NATIVE_CLAUSE in fresh
    assert f"they name, at `{media}`, with parallel" in fresh
    assert "the errors are at the end of the task files" in fresh

    resumed = prompt_of(
        native_search=False, session_scope="assignment", conversation_handle="agy-conv"
    )
    # No media to reopen: the first step opens the task parts, then the repair clause.
    assert resumed.startswith(
        local_agent_module.AGENT_TASK_PROMPT_AGY_MEDIA
        + " In your very first step, open every task file, with parallel view_file calls in that"
        " same step, each whole (no line range)."
        + " Your previous answer to this same task failed the harness output contract"
    )
    # A tool session has no media to open.
    assert local_agent_module.agy_first_step_media(capsule, mcp_server={}) == []


def test_agy_native_and_media_calls_use_separate_projects(tmp_path: Path) -> None:
    """The entitlement *is* the project's `.agents/` tree.

    Sharing one project would mean rewriting the guard for every call, with
    `max_parallel` concurrent calls racing over the file that decides what the
    model may touch. Two projects means each entitlement is written once.
    """

    from finesub.llm.agent.local_agent import (
        AGY_AGENT_NAME,
        AGY_NATIVE_AGENT_NAME,
        AGY_NATIVE_PERMISSION_RULES,
        AGY_READ_PERMISSION_RULES,
    )

    driver = _agy_driver(tmp_path)
    capsule = SimpleNamespace(
        root=tmp_path / "domain" / "episode",
        messages_path=tmp_path / "domain" / "episode" / "task.json",
        continuation_of="",
    )
    _agy_task_parts(capsule.root)
    driver._resolved_command = ("agy",)
    seen: list[tuple[Path, bool]] = []

    def _fake_ensure(root, native=False):
        seen.append((Path(root), native))
        return ("native-id" if native else "media-id"), "digest"

    driver._ensure_project = _fake_ensure
    granted: list[tuple[str, tuple[str, ...]]] = []
    driver._grant_permissions = lambda pid, *, mcp_tools=(), rules=(): granted.append(
        (pid, tuple(rules))
    )

    media = driver._argv(capsule, native_search=False, probe=SimpleNamespace())
    native = driver._argv(capsule, native_search=True, probe=SimpleNamespace())

    # Both get the read permission (agy 1.2.13 checks `view_file` too); only
    # the entitled project gets the fetch permission: agy auto-denies an
    # ungranted call in headless mode and ends the turn with it (2026-08-30).
    assert granted == [
        ("media-id", AGY_READ_PERMISSION_RULES),
        ("native-id", AGY_READ_PERMISSION_RULES + AGY_NATIVE_PERMISSION_RULES),
    ]

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

    def fake_split(input_path, out_dir, stem, **_kwargs):
        piece = Path(out_dir) / f"{stem}-p1.mp4"
        piece.write_bytes(Path(input_path).read_bytes() + b"-mp4")
        return [(piece, 0.0, 10.0)]

    monkeypatch.setattr("finesub.media.ffmpeg.split_audio_for_agy", fake_split)

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
    # The `--effort` the argv carried, and whether this model's usage can say it did not think (`degeneration`).
    assert result.execution_attempt["isolation"]["reasoning_effort"] == "high"
    assert result.execution_attempt["thinking_reported"] is True
    preparation = result.execution_attempt["input_preparation"]["media"][0]
    assert preparation["mode"] == "audio_piece_mp4"
    assert preparation["piece_seconds"] == [0.0, 10.0]
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


def test_agy_tags_a_leaked_thought_on_the_returned_attempt(tmp_path: Path) -> None:
    """The wiring, end to end: a capsule answer that opens with agy's leaked
    reasoning comes back as content (the output contract decides) with
    `answer_shape` on the attempt; an ordinary answer carries no tag."""

    driver = _agy_driver(tmp_path)

    leaked = driver.run([{"role": "user", "content": "LEAK"}], task="correction")
    plain = driver.run([{"role": "user", "content": "fine"}], task="correction")

    assert leaked.content.startswith("thought\n")
    assert leaked.execution_attempt["answer_shape"] == "thought_leak"
    assert "answer_shape" not in plain.execution_attempt


AGY_CUT = "Your previous response was cut off because it exceeded the output token limit\nRetries remaining: 3"


@pytest.fixture(autouse=True)
def agy_data(tmp_path: Path, monkeypatch) -> Path:
    """Every test here reads agy's transcripts from tmp, never `~/.gemini`."""

    from finesub.llm.agent import agy_transcript

    data = tmp_path / "agy-data"
    monkeypatch.setattr(agy_transcript, "agy_data_dir", lambda: data)
    return data


def _agy_cut_off_transcript(
    data: Path, *, error: str, replies: tuple[str, str] = ("<translated>\n1|a", "gy|b\n</translated>\n")
) -> None:
    """agy's transcript of `agy-conv` as a cut-off turn leaves it: the reply,
    agy's inserted error step, the model's continuation."""

    path = data / "brain" / "agy-conv" / ".system_generated" / "logs" / "transcript_full.jsonl"
    path.parent.mkdir(parents=True)
    steps = [
        {"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>CUTOFF"},
        {"step_index": 1, "type": "PLANNER_RESPONSE", "content": replies[0]},
        {"step_index": 2, "type": "ERROR_MESSAGE", "error": error, "content": f"Error: {error}"},
        {"step_index": 3, "type": "PLANNER_RESPONSE", "content": replies[1]},
    ]
    path.write_text("".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8")


def test_agy_keeps_the_answer_it_continued_past_the_output_limit(tmp_path: Path, agy_data: Path) -> None:
    """agy continues a reply cut off at the output limit inside the same call
    but reports the turn as `ERROR`; the answer is read off its transcript and
    handed on as content, the error no longer a provider failure."""

    _agy_cut_off_transcript(agy_data, error=AGY_CUT)

    result = _agy_driver(tmp_path).run([{"role": "user", "content": "CUTOFF"}], task="correction")

    assert result.content == "<translated>\n1|agy|b\n</translated>\n"
    assert result.execution_attempt["transcript_recovery"] == {
        "status": "ERROR",
        "segments": 2,
        "dropped": False,
        "agy_error": AGY_CUT,
    }


@pytest.mark.parametrize("transcript", ["missing", "another error"])
def test_agy_cut_off_turn_without_its_answer_stays_a_provider_failure(
    tmp_path: Path, agy_data: Path, transcript: str
) -> None:
    """No transcript, or one whose last in-turn error is not the one the
    result reports (the turn answered, then failed for good): today's
    transient failure, so the route moves on."""

    if transcript != "missing":
        _agy_cut_off_transcript(agy_data, error="The stream was interrupted.")

    with pytest.raises(LocalAgentTransientError, match="cut off"):
        _agy_driver(tmp_path).run([{"role": "user", "content": "CUTOFF"}], task="correction")


def test_agy_answer_after_an_empty_reply_retry_is_the_whole_answer(tmp_path: Path, agy_data: Path) -> None:
    """`SUCCESS`, but `response` is only the reply after the retry."""

    _agy_cut_off_transcript(agy_data, error="model output error: empty", replies=("<translated>\n1|a", "gy|b\n</translated>\n"))

    result = _agy_driver(tmp_path).run([{"role": "user", "content": "RETRIED"}], task="correction")

    assert result.content == "<translated>\n1|agy|b\n</translated>\n"
    assert result.execution_attempt["transcript_recovery"]["status"] == "SUCCESS"


def test_agy_success_whose_response_was_the_whole_answer_records_nothing(tmp_path: Path, agy_data: Path) -> None:
    _agy_cut_off_transcript(agy_data, error="model output error: empty", replies=("", "gy|b\n</translated>"))

    result = _agy_driver(tmp_path).run([{"role": "user", "content": "RETRIED"}], task="correction")

    assert result.content == "gy|b\n</translated>\n"
    assert "transcript_recovery" not in result.execution_attempt


def test_agy_continued_leaked_reasoning_is_still_tagged(tmp_path: Path, agy_data: Path) -> None:
    """Both records: the answer came off the transcript, and it is reasoning."""

    _agy_cut_off_transcript(agy_data, error=AGY_CUT, replies=("thought\nfirst ", "thought\nsecond"))

    result = _agy_driver(tmp_path).run([{"role": "user", "content": "CUTOFF"}], task="correction")

    assert result.execution_attempt["answer_shape"] == "thought_leak"
    assert result.execution_attempt["transcript_recovery"]["segments"] == 2


@pytest.mark.parametrize(
    ("asked", "tool_session", "error"),
    [
        ("CUTOFF EXIT3", False, LocalAgentTransientError),  # a non-zero exit stays that failure
        ("CUTOFF ROGUE", False, LocalAgentPolicyViolationError),  # a recovered answer clears no violation
        ("CUTOFF", True, LocalAgentTransientError),  # a tool session's answer is its `submit`
    ],
)
def test_agy_recovery_stays_out_of_failures_it_does_not_own(
    tmp_path: Path, agy_data: Path, asked: str, tool_session: bool, error: type
) -> None:
    _agy_cut_off_transcript(agy_data, error=AGY_CUT)
    server = {"command": "python", "args": [], "env": {}, "tools": ["submit"]} if tool_session else None

    with pytest.raises(error) as raised:
        _agy_driver(tmp_path).run([{"role": "user", "content": asked}], task="correction", mcp_server=server)

    if "EXIT3" in asked:
        assert "exited with status 3" in str(raised.value)
    # Recovered, then stopped by the violation; the other two never recover.
    (attempt,) = raised.value._harness_execution_attempts
    assert ("transcript_recovery" in attempt) == ("ROGUE" in asked)


def test_agy_driver_sends_long_audio_as_pieces_cut_between_rows(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.aac"
    source.write_bytes(b"fake-audio")
    driver = _agy_driver(tmp_path)
    seen = {}

    def fake_split(input_path, out_dir, stem, *, speech=(), **_kwargs):
        seen["speech"] = speech
        pieces = []
        for k, (start, end) in enumerate([(0.0, 290.5), (290.5, 600.0)], 1):
            piece = Path(out_dir) / f"{stem}-p{k}.mp4"
            piece.write_bytes(b"piece")
            pieces.append((piece, start, end))
        return pieces

    monkeypatch.setattr("finesub.media.ffmpeg.split_audio_for_agy", fake_split)
    monkeypatch.setattr(driver.capsules, "remove", lambda *_args: (_ for _ in ()).throw(OSError("retain")))
    result = driver.run(
        [{"role": "user", "content": [
            {"type": "text", "text": "transcribe"},
            {"type": "file", "file": {"file_id": "files/clip", "filename": "clip.aac", "format": "audio/aac",
                                      "local_path": str(source.resolve()),
                                      "speech_spans": [[1.0, 280.0], [291.0, 599.0]]}},
            {"type": "text", "text": "after"},
        ]}],
        task="correction",
        profile_id="agent-only",
        reasoning_effort="high",
    )

    assert seen["speech"] == [(1.0, 280.0), (291.0, 599.0)]
    media = result.execution_attempt["input_preparation"]["media"]
    assert [row["piece_seconds"] for row in media] == [[0.0, 290.5], [290.5, 600.0]]
    episode = Path(result.execution_attempt["evidence_locator"]["absolute_at_write"])
    content = json.loads((episode / "input" / "messages.json").read_text("utf-8"))[0]["content"]
    assert [block["type"] for block in content] == ["text", "file", "file", "text", "text"]
    assert [Path(block["file"]["local_path"]).name for block in content[1:3]] == [
        "attachment-01-p1.mp4", "attachment-01-p2.mp4"]
    assert "speech_spans" not in content[1]["file"]
    assert "2 个附件" in content[3]["text"]
    assert "附件 2 `attachment-01-p2.mp4` 是整段的 290.5–600.0 秒" in content[3]["text"]
    assert content[4]["text"] == "after"
    assert result.execution_attempt["input_preparation"]["first_step_media"] == [
        "media/attachment-01-p1.mp4", "media/attachment-01-p2.mp4"]


def test_agy_sends_a_video_without_a_picture_as_audio_pieces_and_a_prepared_video_whole(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    driver = _agy_driver(tmp_path)
    probed = []

    def fake_split(input_path, out_dir, stem, *, speech=(), **_kwargs):
        piece = Path(out_dir) / f"{stem}-p1.mp4"
        piece.write_bytes(b"piece")
        return [(piece, 0.0, 200.0)]

    def no_picture(path):
        probed.append(path)
        return False

    monkeypatch.setattr("finesub.media.ffmpeg.split_audio_for_agy", fake_split)
    monkeypatch.setattr("finesub.media.ffmpeg.media_has_video_stream", no_picture)
    monkeypatch.setattr("finesub.media.ffmpeg.agy_media_cuts", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(driver.capsules, "remove", lambda *_args: (_ for _ in ()).throw(OSError("retain")))

    def send(**extra):
        result = driver.run(
            [{"role": "user", "content": [
                {"type": "file", "file": {"format": "video/mp4", "local_path": str(source.resolve()),
                                          "speech_spans": [[1.0, 199.0]], **extra}}]}],
            task="correction", profile_id="agent-only", reasoning_effort="high")
        episode = Path(result.execution_attempt["evidence_locator"]["absolute_at_write"])
        content = json.loads((episode / "input" / "messages.json").read_text("utf-8"))[0]["content"]
        return result.execution_attempt["input_preparation"]["media"], content

    media, content = send()
    assert [row["mode"] for row in media] == ["audio_container_repaired_piece_mp4"] and len(probed) == 1
    assert [block["type"] for block in content] == ["file"], "one piece: no note on the pieces"
    media, content = send(agy_prepared=True)
    assert [row["mode"] for row in media] == ["video_pre_sampled_0.25fps_copy"]
    assert len(probed) == 1, "a prepared video is not probed"
    assert [block["type"] for block in content] == ["file"] and "speech_spans" not in content[0]["file"]


def test_agy_sends_a_video_over_five_minutes_as_pieces_cut_between_rows(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    driver = _agy_driver(tmp_path)
    seen = {}

    def cuts(path, *, speech=(), **_kwargs):
        seen["speech"] = speech
        return [298.0]

    def fake_split(input_path, out_dir, stem, at, **_kwargs):
        seen["at"] = at
        pieces = []
        for k, (start, end) in enumerate([(0.0, 298.0), (298.0, 540.0)], 1):
            piece = Path(out_dir) / f"{stem}-p{k}.mp4"
            piece.write_bytes(b"piece")
            pieces.append((piece, start, end))
        return pieces

    monkeypatch.setattr("finesub.media.ffmpeg.agy_media_cuts", cuts)
    monkeypatch.setattr("finesub.media.ffmpeg.split_video_for_agy", fake_split)
    monkeypatch.setattr(driver.capsules, "remove", lambda *_args: (_ for _ in ()).throw(OSError("retain")))
    result = driver.run(
        [{"role": "user", "content": [
            {"type": "file", "file": {"format": "video/mp4", "local_path": str(source.resolve()), "agy_prepared": True,
                                      "speech_spans": [[1.0, 297.0], [299.0, 539.0]]}}]}],
        task="correction", profile_id="agent-only", reasoning_effort="high")

    assert seen == {"speech": [(1.0, 297.0), (299.0, 539.0)], "at": [298.0]}
    media = result.execution_attempt["input_preparation"]["media"]
    assert [(row["mode"], row["piece_seconds"]) for row in media] == [
        ("video_piece_0.25fps", [0.0, 298.0]), ("video_piece_0.25fps", [298.0, 540.0])]
    episode = Path(result.execution_attempt["evidence_locator"]["absolute_at_write"])
    content = json.loads((episode / "input" / "messages.json").read_text("utf-8"))[0]["content"]
    assert [block["type"] for block in content] == ["file", "file", "text"]
    assert "speech_spans" not in content[0]["file"]
    assert "这段视频较长" in content[2]["text"] and "附件 2 `attachment-01-p2.mp4` 是整段的 298.0–540.0 秒" in content[2]["text"]


def test_agy_media_preparation_failure_is_retained_but_redacted(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "clip.aac"
    source.write_bytes(b"fake-audio")
    driver = _agy_driver(tmp_path)

    def fail_split(*_args, **_kwargs):
        raise RuntimeError("private ffmpeg diagnostic")

    monkeypatch.setattr("finesub.media.ffmpeg.split_audio_for_agy", fail_split)
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


def test_keep_episodes_keeps_a_clean_run_and_codex_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    """`[llm] local_agent_keep_episodes`: nothing is deleted, and Codex keeps
    its own rollout (no --ephemeral) -- its only per-request ledger."""

    driver = _driver(tmp_path, monkeypatch, keep_episodes=True, max_result_bytes=8192)
    result = driver.run([{"role": "user", "content": "argv"}], task="general")

    assert result.execution_attempt["episode_kept_by_setting"] is True
    assert (tmp_path / "runtime").exists()
    assert Path(result.execution_attempt["evidence_locator"]["absolute_at_write"]).is_dir()
    assert "--ephemeral" not in json.loads(result.content)


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
    # `exec resume` reads stdin only when the prompt is `-`.
    assert second_argv[-1] == "-"
    assert second_argv[second_argv.index("--sandbox") + 1] == "read-only"
    assert second.conversation_handle == first.conversation_handle


def test_codex_resume_sends_the_prompt_and_the_task_through_stdin(
    tmp_path: Path, monkeypatch
) -> None:
    """Measured 2026-09-30: with the prompt text as the argument, every resumed
    turn answered "no <stdin> JSON" -- `exec resume` ignores stdin then."""

    from finesub.llm.agent import local_agent as la

    driver = _driver(tmp_path, monkeypatch)
    payload = b'{"messages": []}'

    fresh, fresh_meta = driver._prepare_capsule_input(None, payload)
    resumed, resumed_meta = driver._prepare_capsule_input(
        None, payload, conversation_handle="11111111-2222-3333-4444-555555555555"
    )

    assert (fresh, dict(fresh_meta)) == (payload, {})
    assert resumed.startswith(la.AGENT_TASK_PROMPT_STDIN_ONLY.encode("utf-8"))
    assert resumed.endswith(b"<stdin>\n" + payload + b"\n</stdin>\n")
    assert resumed_meta == {"resume_prompt_via_stdin": True}


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
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=8192)

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
        max_result_bytes=8192,
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
    note = result.execution_attempt["notes"]
    assert [row["event"] for row in note] == ["native_search_not_used"]
    # The message states the observation and both readings of it. A denied
    # search leaves exactly this evidence too (docs/llm_local_agent_agy.md
    # §6.1), so the note must not assert that the model chose not to search.
    assert "No completed search was observed" in note[0]["message"]
    assert "never completed" in note[0]["message"]


def test_a_driver_without_tool_events_says_so_instead_of_claiming_no_search(
    tmp_path: Path, monkeypatch
) -> None:
    """"Completed without searching" is a reading of the events.

    A CLI that reports none (dsh headless prints its answer and nothing else)
    produces an empty list either way, so the same sentence would be an
    assertion about tool use nobody observed. The audit record has to say
    which of the two it is looking at.
    """

    driver = _driver(tmp_path, monkeypatch)
    monkeypatch.setattr(type(driver), "observes_tool_events", False)

    result = driver.run(
        [{"role": "user", "content": "hello"}],
        task="research",
        native_search=True,
    )

    assert result.execution_attempt["search_events"] == []
    note = result.execution_attempt["notes"][0]
    assert note["event"] == "native_search_unobserved"
    assert "cannot be read off this call" in note["message"]

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
    # The budget is a PROCESS-global, one per vendor, and its limit is fixed by
    # whichever driver builds it first -- a later config asking for something
    # else is reported, not honoured (`_shared_in_flight_pool`). So assert the
    # driver actually got the limit this test is about before measuring: an
    # earlier test that built a codex driver with the default `max_parallel=4`
    # would otherwise make the count below land on 2, 3 or 4 by timing, and
    # this test would fail as a flake instead of naming its cause.
    assert driver._in_flight.limit == 2, (
        "the shared codex slot pool was built by an earlier test; "
        "conftest's `_fresh_agent_slot_pools` is what keeps that from happening"
    )
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


def test_a_stale_cache_warning_does_not_outrank_a_turn_that_ran(
    tmp_path: Path, monkeypatch
) -> None:
    """Codex logs the cache warning, refetches, and runs the turn anyway.

    Seen 2026-09-16: the turn then failed on the usage limit, and the call was
    reported as "cannot decode the current model catalog; update the CLI".
    A CLI that reached a turn has a usable catalog; what failed is the turn.
    """

    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentError) as caught:
        driver.run([{"role": "user", "content": "stale-cache-turn-failed"}], task="general")

    assert not isinstance(caught.value, LocalAgentUnavailableError)
    assert "model catalog" not in str(caught.value)


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


def test_ctrl_c_while_the_cli_runs_takes_its_tree_down(tmp_path: Path, monkeypatch) -> None:
    driver = _driver(tmp_path, monkeypatch, timeout_seconds=30)
    first_sleep: list[float] = []

    def interrupted_sleep(seconds: float) -> None:
        # Let the fake CLI spawn its child, then Ctrl-C the supervisor's poll loop.
        first_sleep.append(first_sleep[0] if first_sleep else time.monotonic())
        if time.monotonic() - first_sleep[0] > 0.5:
            raise KeyboardInterrupt
        time.sleep(seconds)

    monkeypatch.setattr(
        local_agent_module, "time", SimpleNamespace(monotonic=time.monotonic, time=time.time, sleep=interrupted_sleep)
    )
    with pytest.raises(KeyboardInterrupt):
        driver.run([{"role": "user", "content": "child"}], task="general")
    monkeypatch.undo()
    time.sleep(2)
    assert not list((tmp_path / "runtime").rglob("child-survived.txt"))


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


def _unaccepted_episode(parent: Path, name: str, *, age_seconds: float) -> Path:
    root = _episode(parent, name, age_seconds=age_seconds)
    (root / UNACCEPTED_EPISODE_MARKER).write_text("{}", encoding="utf-8")
    # Writing the marker bumps the directory's mtime; put the age back.
    stamp = time.time() - age_seconds
    os.utime(root, (stamp, stamp))
    return root


def test_unaccepted_sessions_roll_in_a_pool_of_their_own(tmp_path: Path) -> None:
    """Two caps, counted apart, so a burst of one kind cannot evict the other.

    Counted as one pool of `RETAINED_FAILED_EPISODES`, the 37 old episodes
    below would lose their 17 oldest across both kinds -- and a run that left
    many sessions unaccepted would push every transport failure out of reach.
    """

    manager = AgentCapsuleManager(tmp_path / "runtime")
    location = manager.resolve_location()
    location.parent.mkdir(parents=True, exist_ok=True)
    old_day = 86_400.0
    for index in range(RETAINED_FAILED_EPISODES + 3):
        _episode(location.parent, f"failed-{index:03d}", age_seconds=old_day + index)
    for index in range(RETAINED_UNACCEPTED_EPISODES + 4):
        _unaccepted_episode(
            location.parent, f"unaccepted-{index:03d}", age_seconds=old_day + index
        )

    removed = manager.prune(location, timeout_seconds=900)

    names = {p.name for p in location.parent.iterdir() if p.is_dir()}
    assert len({n for n in names if n.startswith("failed-")}) == RETAINED_FAILED_EPISODES
    assert (
        len({n for n in names if n.startswith("unaccepted-")})
        == RETAINED_UNACCEPTED_EPISODES
    )
    # Oldest-first within each pool: the highest indices carry the oldest mtimes.
    assert set(removed) == (
        {f"failed-{i:03d}" for i in range(RETAINED_FAILED_EPISODES, RETAINED_FAILED_EPISODES + 3)}
        | {
            f"unaccepted-{i:03d}"
            for i in range(RETAINED_UNACCEPTED_EPISODES, RETAINED_UNACCEPTED_EPISODES + 4)
        }
    )


def test_a_predicate_that_raises_counts_as_not_accepted() -> None:
    """Keeping evidence that turns out to be unneeded costs disk; dropping it costs the diagnosis."""

    def unreadable() -> bool:
        raise RuntimeError("runtime state unreadable")

    assert local_agent_module._completion_holds(unreadable) is False
    assert local_agent_module._completion_holds(lambda: True) is True


FAKE_CLAUDE_CLEAN_EXIT = r'''
import json
import pathlib
import sys

args = sys.argv[1:]
if "--version" in args:
    print("2.1.231 (Claude Code)")
    raise SystemExit(0)
if "--help" in args:
    print("--output-format stream-json --no-session-persistence --tools --allowed-tools "
          "--safe-mode --setting-sources --effort --model --resume --session-id "
          "--mcp-config --strict-mcp-config")
    raise SystemExit(0)
sys.stdin.buffer.read()
print(json.dumps({"type": "system", "subtype": "init", "tools": [], "model": "fake",
                  "session_id": "11111111-2222-3333-4444-555555555555"}), flush=True)
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "I have the answer, but I never called submit."}]}}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "terminal_reason": "end_turn", "permission_denials": [],
                  "usage": {"input_tokens": 10, "output_tokens": 3}}), flush=True)
# The very last thing a real run does. A test that wants "accepted" to turn true
# only once this session has finished uses this file as its predicate.
pathlib.Path(__file__).with_suffix(".done").write_text("1", encoding="utf-8")
'''


def _clean_exit_driver(tmp_path: Path) -> tuple[object, Path]:
    from finesub.llm.agent import local_agent as la

    script = tmp_path / "fake_claude_clean_exit.py"
    script.write_text(FAKE_CLAUDE_CLEAN_EXIT, encoding="utf-8")
    driver = la.ClaudeCodeLocalAgentDriver(
        la.ClaudeCodeDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            timeout_seconds=30,
            max_result_bytes=1024,
            max_event_bytes=8192,
            max_stderr_bytes=8192,
        )
    )
    return driver, script


def test_a_session_that_leaves_unaccepted_keeps_its_capsule(tmp_path: Path) -> None:
    """A clean exit is a transport success, but a harness session's text is not its artifact.

    Removing the capsule of such a run used to take the only record of what the
    agent did -- the harness writes no exchange for a call it then fails.
    """

    driver, _script = _clean_exit_driver(tmp_path)

    result = driver.run(
        [{"role": "user", "content": "call next_task"}],
        task="research",
        completion=lambda: False,
    )

    attempt = result.execution_attempt
    assert attempt["retained_unaccepted"] is True
    capsule_root = Path(attempt["evidence_locator"]["absolute_at_write"])
    assert (capsule_root / UNACCEPTED_EPISODE_MARKER).is_file()
    assert (capsule_root / "events" / "agent-events.jsonl").exists()


def test_a_session_accepted_after_the_last_poll_is_not_kept_as_unaccepted(
    tmp_path: Path,
) -> None:
    """The supervisor polls about once a second, and only while the CLI runs.

    A CLI that submits and exits inside that second is accepted without ever
    being observed so, and keyed on the poll alone that successful call would
    be kept as unaccepted. The fake CLI here exits within milliseconds of its
    last write, which a real one rarely does (CodeBuddy left 4.6-11.7s after
    accept in ten measured calls) -- this pins the case, not its frequency.
    Which path observed the accept is a race; that it is never filed as
    unaccepted is not.
    """

    driver, script = _clean_exit_driver(tmp_path)

    result = driver.run(
        [{"role": "user", "content": "call next_task"}],
        task="research",
        completion=script.with_suffix(".done").exists,
    )

    attempt = result.execution_attempt
    assert "retained_unaccepted" not in attempt
    # Either path removes the capsule of an accepted, clean session.
    assert "evidence_locator" not in attempt


FAKE_CLAUDE_EXITS_AFTER_ACCEPT = FAKE_CLAUDE_CLEAN_EXIT.replace(
    "sys.stdin.buffer.read()\n",
    "sys.stdin.buffer.read()\n"
    "import time\n"
    "pathlib.Path(__file__).with_suffix('.done').write_text('1', encoding='utf-8')\n"
    "time.sleep(2.5)\n",
)


def test_a_session_seen_accepted_while_running_removes_its_capsule(tmp_path: Path) -> None:
    """The supervisor saw the accept, then the CLI left cleanly within its grace.

    That used to return before the capsule was settled, so every such success
    stayed on disk, unmarked, in the transport-failure pool -- CodeBuddy took
    this path ten times out of ten, enough for one long run to roll the real
    failures out of that pool.
    """

    from finesub.llm.agent import local_agent as la

    script = tmp_path / "fake_claude_exits_after_accept.py"
    script.write_text(FAKE_CLAUDE_EXITS_AFTER_ACCEPT, encoding="utf-8")
    driver = la.ClaudeCodeLocalAgentDriver(
        la.ClaudeCodeDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            timeout_seconds=30,
            max_result_bytes=1024,
            max_event_bytes=8192,
            max_stderr_bytes=8192,
        )
    )

    result = driver.run(
        [{"role": "user", "content": "call next_task"}],
        task="research",
        completion=script.with_suffix(".done").exists,
    )

    attempt = result.execution_attempt
    assert attempt["completed_by_runtime_at"], "the poll must have seen the accept"
    assert not attempt.get("reclaimed_after_completion")
    assert "evidence_locator" not in attempt
    episodes = [p for p in (tmp_path / "runtime").rglob("*") if p.is_dir() and p.name == "events"]
    assert episodes == []


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
    assert isolation["sandbox_kind"] == "named_tool_allowlist"
    assert isolation["write_restriction"] == "exact named built-in tool allowlist"
    assert not (tmp_path / "runtime" / result.episode_id).exists()


def test_claude_argv_isolates_and_exposes_no_builtin_tools(
    tmp_path: Path, monkeypatch
) -> None:
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
    # A completion call is entitled to no built-ins. `--tools` is the exact
    # availability boundary; `--allowed-tools` only grants permission.
    # One comma-joined argument, not a variadic list: with no model or effort
    # configured the prompt would otherwise be parsed as another tool name.
    assert args[args.index("--tools") + 1] == ""
    assert "--disallowed-tools" not in args
    assert "--allowed-tools" not in args
    # Nothing points the agent at files it cannot open.
    assert "--add-dir" not in args


def test_claude_native_call_permits_only_the_search_tools(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import CLAUDE_SEARCH_TOOLS

    driver = _claude_driver(tmp_path, monkeypatch, max_result_bytes=8192)
    result = driver.run(
        [{"role": "user", "content": "argv"}], task="research", native_search=True
    )
    args = json.loads(result.content)

    assert set(args[args.index("--tools") + 1].split(",")) == set(
        CLAUDE_SEARCH_TOOLS
    )
    assert "--disallowed-tools" not in args
    assert set(args[args.index("--allowed-tools") + 1].split(",")) == set(
        CLAUDE_SEARCH_TOOLS
    )


def test_claude_completion_rejects_a_tool_call(tmp_path: Path, monkeypatch) -> None:
    driver = _claude_driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentPolicyViolationError, match="forbidden tool Bash"):
        driver.run([{"role": "user", "content": "tool"}], task="correction")


def test_claude_session_offering_denied_tools_warns_but_answers(
    tmp_path: Path, monkeypatch, reported
) -> None:
    """An offered-but-unused tool is news, not a failure.

    `--tools` is the exact built-in boundary, so an unexpected offered tool is
    evidence that the CLI contract changed. Failing the call over an unused
    tool would still skip the whole API fallback chain; the event-stream check
    refuses an actual unentitled invocation. See
    ``test_claude_completion_rejects_a_search_event``.
    """

    driver = _claude_driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "leak"}], task="correction")

    assert result.content
    messages = [
        str(row.get("message") or "")
        for row in result.execution_attempt.get("warnings") or []
    ]
    assert any("outside this call's exact --tools set" in message for message in messages)
    assert reported.codes() == ["agent-driver-stream"]


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


FAKE_CLAUDE_SYSTEM_PROTOCOL = r'''
import json
import sys

args = sys.argv[1:]
if "--version" in args:
    print("2.1.302 (Claude Code)")
    raise SystemExit(0)
if "--help" in args:
    print("--output-format stream-json --no-session-persistence --tools --allowed-tools "
          "--safe-mode --setting-sources --resume --session-id --mcp-config "
          "--strict-mcp-config" + HELP_EXTRA)
    raise SystemExit(0)
sys.stdin.buffer.read()
system = ""
if "--append-system-prompt-file" in args:
    path = args[args.index("--append-system-prompt-file") + 1]
    system = open(path, encoding="utf-8").read()
print(json.dumps({"type": "system", "subtype": "init", "tools": [], "model": "fake",
                  "session_id": "11111111-2222-3333-4444-555555555555"}))
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
    {"type": "text", "text": json.dumps({"args": args, "system": system})}]}}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "terminal_reason": "end_turn", "permission_denials": [],
                  "usage": {"input_tokens": 10, "output_tokens": 3}}))
'''


def _claude_system_driver(
    tmp_path: Path, *, flags: bool = True, mcp_opening: str = "system_prompt"
):
    from finesub.llm.agent.local_agent import ClaudeCodeDriverConfig, ClaudeCodeLocalAgentDriver

    extra = " --append-system-prompt --exclude-dynamic-system-prompt-sections" if flags else ""
    script = tmp_path / "fake_claude_system.py"
    script.write_text(
        FAKE_CLAUDE_SYSTEM_PROTOCOL.replace("HELP_EXTRA", repr(extra)), encoding="utf-8"
    )
    return ClaudeCodeLocalAgentDriver(
        ClaudeCodeDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            timeout_seconds=10,
            max_result_bytes=200_000,
            max_event_bytes=400_000,
            mcp_opening=mcp_opening,
            keep_episodes=True,
        )
    )


def _claude_tool_server(protocol: str = "") -> dict:
    server = {"command": "srv", "args": [], "env": {}, "tools": ["submit"]}
    if protocol:
        server["agent_protocol"] = protocol
    return server


def test_claude_system_prompt_opening_appends_the_same_protocol_every_window(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub.llm.agent.local_agent import claude_system_protocol

    driver = _claude_system_driver(tmp_path)
    assert driver.opening_mode() == "system_prompt"

    def window(text: str):
        result = driver.run(
            [{"role": "user", "content": text}],
            task="correction",
            mcp_server=_claude_tool_server("RULES: answer good"),
        )
        return result, json.loads(result.content)

    first, seen = window("window one")
    second, seen_again = window("window two")
    args = seen["args"]
    assert "--exclude-dynamic-system-prompt-sections" in args
    assert seen["system"] == claude_system_protocol("RULES: answer good")
    assert "## Protocol" in seen["system"] and "RULES: answer good" in seen["system"]
    # Byte for byte the same across windows: that is what the next window's
    # first request reads from cache.
    assert seen_again["system"] == seen["system"]
    prepared = first.execution_attempt["input_preparation"]["system_protocol"]
    assert prepared == second.execution_attempt["input_preparation"]["system_protocol"]
    assert prepared["path"] == "input/system-protocol.md"
    assert "system_protocol" not in first.execution_attempt["isolation"]
    (manifest_path,) = (tmp_path / "runtime").glob(f"**/{first.episode_id}/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = {row["path"]: row for row in manifest["inputs"]}
    assert ledger["input/system-protocol.md"]["sha256"] == prepared["sha256"]
    assert ledger["input/system-protocol.md"]["kind"] == "harness_system_protocol"


@pytest.mark.parametrize("mcp_opening", ["inline", "system_prompt"])
def test_claude_carries_no_system_protocol_without_one_to_carry(
    tmp_path: Path, mcp_opening
) -> None:
    driver = _claude_system_driver(tmp_path, mcp_opening=mcp_opening)
    assert driver.opening_mode() == mcp_opening
    result = driver.run(
        [{"role": "user", "content": "window"}], task="correction", mcp_server=_claude_tool_server()
    )
    seen = json.loads(result.content)
    assert "--append-system-prompt-file" not in seen["args"]
    assert "--exclude-dynamic-system-prompt-sections" not in seen["args"]
    assert "input_preparation" not in result.execution_attempt


def test_claude_falls_back_to_inline_when_the_cli_lacks_the_flags(tmp_path: Path) -> None:
    driver = _claude_system_driver(tmp_path, flags=False)
    assert driver.opening_mode() == "inline"
    result = driver.run(
        [{"role": "user", "content": "window"}], task="correction", mcp_server=_claude_tool_server()
    )
    assert result.execution_attempt["isolation"]["system_protocol"] == "unsupported_by_cli"


def test_provider_tier_picks_the_driver() -> None:
    """The tier is the only thing that selects a CLI, so both the readiness
    pre-filter and dispatch land on the same one for a given candidate."""

    from finesub.llm.routing.execution_policy import ExecutionSettings, driver_for_provider_tier
    from finesub.llm.agent.local_agent import ClaudeCodeLocalAgentDriver, CodexLocalAgentDriver

    settings = ExecutionSettings()
    codex = driver_for_provider_tier(
        settings, provider_tier="LOCAL_CODEX", model="gpt-6-luna"
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

    `--tools` is variadic, so omitting its explicit empty argument would absorb
    the positional prompt that follows it and the agent would be handed the
    task with no instructions at all, silently.
    """

    driver = _claude_driver(
        tmp_path, monkeypatch, model="", effort="", max_result_bytes=8192
    )
    result = driver.run([{"role": "user", "content": "argv"}], task="correction")
    args = json.loads(result.content)

    assert "--model" not in args and "--effort" not in args
    assert args[-1].startswith(AGENT_USER_RULES_NOTICE + " You are a FineSub text execution backend")
    assert args[args.index("--tools") + 1] == ""


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


# A CLI that submits (as far as the harness is concerned, the runtime says
# `accepted`) and then keeps talking: a complete init + message, then a cut
# line, then silence until it is reclaimed.
FAKE_CLAUDE_LINGERING = r'''
import json
import pathlib
import sys
import time

args = sys.argv[1:]
if "--version" in args:
    print("2.1.231 (Claude Code)")
    raise SystemExit(0)
if "--help" in args:
    print("--output-format stream-json --no-session-persistence --tools --allowed-tools "
          "--safe-mode --setting-sources --effort --model --resume --session-id "
          "--mcp-config --strict-mcp-config")
    raise SystemExit(0)
sys.stdin.buffer.read()
print(json.dumps({"type": "system", "subtype": "init", "tools": [], "model": "fake",
                  "session_id": "11111111-2222-3333-4444-555555555555"}), flush=True)
print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
    {"type": "text", "text": "submitted"}]}}), flush=True)
sys.stdout.write('{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "te')
sys.stdout.flush()
# Everything this run means to emit is now in the pipe. The test waits for
# this file before reporting completion, so the reclaim never races our own
# interpreter start-up.
pathlib.Path(__file__).with_suffix(".ready").write_text("1", encoding="utf-8")
time.sleep(60)
'''


def test_an_accepted_task_reclaims_a_lingering_cli_and_keeps_its_raw_stream(
    tmp_path: Path, monkeypatch
) -> None:
    """docs/llm_agent_tool_protocol.md §4: `accepted` is the completion;
    the CLI gets a grace period, then its tree is reclaimed, and what it was
    still writing stays in the capsule as evidence."""

    from finesub.llm.agent import local_agent as la

    script = tmp_path / "fake_claude_lingering.py"
    script.write_text(FAKE_CLAUDE_LINGERING, encoding="utf-8")
    monkeypatch.setenv("GEMINI_FREE", "secret-gemini")
    monkeypatch.setattr(la, "ACCEPTED_EXIT_GRACE_SECONDS", 0.2)
    driver = la.ClaudeCodeLocalAgentDriver(
        la.ClaudeCodeDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            timeout_seconds=30,
            max_result_bytes=1024,
            max_event_bytes=8192,
            max_stderr_bytes=8192,
        )
    )
    started = time.monotonic()

    # Completion is what starts the grace clock, so reporting it before the
    # fake CLI has written anything makes the evidence assertions a race
    # against interpreter start-up -- lost often enough under a loaded
    # parallel run to fail with an empty `raw.jsonl`.
    emitted = script.with_suffix(".ready")
    result = driver.run(
        [{"role": "user", "content": "call next_task"}],
        task="correction",
        completion=emitted.exists,
    )

    attempt = result.execution_attempt
    assert time.monotonic() - started < 20, "reclaimed long before the 60s sleep"
    assert attempt["reclaimed_after_completion"] is True
    assert attempt["completed_by_runtime_at"]
    # Killed, not exited on its own (the job-object kill on Windows reports
    # whatever code it was given); recorded, never raised.
    assert attempt["return_code"] is not None
    assert attempt["duration_ms"] < 20_000
    # Reclaimed after accept: the raw original is kept beside the normalized
    # file (the reader drops the cut line rather than failing, so the stream
    # is not flagged partial here).
    assert attempt["raw_events_retained"] is True
    capsule_root = Path(attempt["evidence_locator"]["absolute_at_write"])
    raw = (capsule_root / "events" / "raw.jsonl").read_text(encoding="utf-8")
    assert '"subtype": "init"' in raw and raw.rstrip().endswith('"te')
    assert (capsule_root / "events" / "agent-events.jsonl").exists()


def test_the_tool_guard_denies_retrieval_and_its_native_twin_allows_it(
    tmp_path: Path,
) -> None:
    """The tool protocol's two projects, run as agy runs them.

    Regression for 2026-08-30: research rounds take the tool-session
    transport, whose only project denied `search_web` by falling through to
    the deny default. agy issued the call, the hook refused it, and the
    refusal produced no result step -- so the model went on without the
    search and the harness recorded none.
    """

    import subprocess
    import sys as _sys


    driver = _agy_driver(tmp_path)
    domain = tmp_path / "domain"
    assignment = domain / "assignment"
    assignment.mkdir(parents=True, exist_ok=True)
    inside = assignment / "payload.md"
    inside.write_text("block", encoding="utf-8")
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("secret", encoding="utf-8")

    def _guard_for(*, native: bool) -> Path:
        root = domain / (".finesub-tool-native-1" if native else ".finesub-tool-1")
        root.mkdir(parents=True, exist_ok=True)
        paths = driver._write_project_resources(root, tool=True, native=native)
        (paths["agents_root"] / "view_roots.json").write_text(
            json.dumps([str(assignment.resolve())]), encoding="utf-8"
        )
        return paths["guard"]

    def _decide(guard: Path, tool: str, **args) -> str:
        payload = json.dumps({"toolCall": {"name": tool, "args": args}})
        done = subprocess.run(
            [_sys.executable, str(guard)],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(done.stdout)["decision"]

    plain = _guard_for(native=False)
    entitled = _guard_for(native=True)

    # The defect: retrieval was denied on the transport research runs on.
    assert _decide(plain, "search_web", query="anything") == "deny"
    assert _decide(plain, "read_url_content", Url="https://example.com") == "deny"
    assert _decide(entitled, "search_web", query="anything") == "allow"
    assert _decide(entitled, "read_url_content", Url="https://example.com") == "allow"

    # Everything the plain guard already allowed or denied is unchanged, and
    # the entitled twin is not a wider door than the two extra tools.
    for guard in (plain, entitled):
        assert _decide(guard, "call_mcp_tool", ServerName="finesub") == "allow"
        assert _decide(guard, "call_mcp_tool", ServerName="elsewhere") == "deny"
        assert _decide(guard, "view_file", AbsolutePath=str(inside)) == "allow"
        assert _decide(guard, "view_file", AbsolutePath=str(outside)) == "deny"
        assert _decide(guard, "run_command", Command="whoami") == "deny"
        assert _decide(guard, "write_to_file", AbsolutePath=str(inside)) == "deny"


def test_a_tool_call_picks_its_project_by_retrieval_entitlement(
    tmp_path: Path,
) -> None:
    """Two projects per slot, not one project rewritten between calls."""

    from finesub.llm.agent.local_agent import (
        AGY_TOOL_AGENT_NAME,
        AGY_TOOL_NATIVE_AGENT_NAME,
    )

    driver = _agy_driver(tmp_path)
    driver._resolved_command = ("python", "agy")
    driver._tool_slot_local.slot = 1
    domain = tmp_path / "domain"
    capsule_root = domain / "episode"
    capsule_root.mkdir(parents=True, exist_ok=True)
    messages = capsule_root / "messages.json"
    messages.write_text(
        json.dumps([{"role": "user", "content": "do the task"}]), encoding="utf-8"
    )

    seen: list[tuple[str, bool]] = []

    def _fake_ensure(project_root, *, native=False, tool=False, protocol=""):
        seen.append((project_root.name, native))
        return "11111111-1111-1111-1111-111111111111", "digest"

    driver._ensure_project = _fake_ensure  # type: ignore[assignment]
    driver._grant_permissions = lambda *a, **k: None  # type: ignore[assignment]

    capsule = SimpleNamespace(root=capsule_root, messages_path=messages)
    server = {"command": "srv", "args": (), "env": {}, "tools": (), "view_roots": ()}

    plain = driver._tool_argv(
        capsule, mcp_server=server, native_search=False, reasoning_effort=""
    )
    entitled = driver._tool_argv(
        capsule, mcp_server=server, native_search=True, reasoning_effort=""
    )

    assert seen == [(".finesub-tool-1", False), (".finesub-tool-native-1", True)]
    assert plain[plain.index("--agent") + 1] == AGY_TOOL_AGENT_NAME
    assert entitled[entitled.index("--agent") + 1] == AGY_TOOL_NATIVE_AGENT_NAME


def test_the_worker_bootstrap_stops_forbidding_the_tools_it_entitles() -> None:
    """The hook and the prompt have to agree about retrieval.

    They did not: the session protocol's native-search fragment told the model
    to look things up while rule 4 forbade every tool but the server's.
    """

    from finesub.llm.prompts import (
        agent_tool_worker_bootstrap,
        agent_tool_worker_session_bootstrap,
    )

    for build in (agent_tool_worker_bootstrap, agent_tool_worker_session_bootstrap):
        plain = build(assignment_id="a", worker_id="w")
        entitled = build(assignment_id="a", worker_id="w", native_search=True)
        assert "$retrieval_exception" not in plain
        assert "$retrieval_exception" not in entitled
        assert "web search" not in plain
        assert "web search" in entitled
        # The carve-out is spliced into rule 4's own sentence, not appended as
        # a second paragraph that a summariser could drop (agy compresses the
        # bootstrap on step one, docs/llm_local_agent_agy.md §3).
        rule = next(line for line in entitled.splitlines() if line.startswith("4."))
        assert "web search" in rule


def test_the_opening_explains_patch_repairs_only_to_a_task_that_takes_patches() -> None:
    """A preparation task's repair is the whole JSON again: its opening does not
    spend a paragraph on `patch_merged` and `rows_to_fix` (cold read round 4,
    P443)."""

    from finesub.llm.prompts import agent_tool_worker_opening

    def opening(takes_patch: bool) -> str:
        return agent_tool_worker_opening(
            assignment_id="a", worker_id="w", manifest={}, shared_blocks=(), task_blocks=(),
            takes_patch=takes_patch,
        )

    assert "patch_merged" in opening(True) and "repair_instruction" in opening(True)
    plain = opening(False)
    assert "patch_merged" not in plain and "rows_to_fix" not in plain and "repair_instruction" not in plain
    assert "submit` again in this same session. If the response says the repair budget is spent" in plain


def test_a_pieces_note_does_not_reopen_media_a_resumed_session_already_heard() -> None:
    """A repair round on a resumed agy session rewrites the task files, notes
    and all, while the media is already in context: the note must not send it
    back to open every piece again (cold read round 4, P444)."""

    from finesub.llm.agent.local_agent import agy_pieces_note

    pieces = [("a-01.mp4", 0.0, 290.0), ("a-02.mp4", 290.0, 560.0)]
    for note in ("agy_audio_pieces_v1.md", "agy_video_pieces_v1.md"):
        text = agy_pieces_note(note, pieces)
        assert "还没打开过的，用 view_file 打开" in text
        assert "每个都要用 view_file 打开" not in text


# --- exit-on-accept ---------------------------------------------------------

# A CLI parked on the harness's held `accepted` reply: it has written every
# request it finished -- one of them as two `assistant` events sharing a
# message id, the way Claude Code splits content blocks -- and now waits for a
# tool result that never comes. `cut-tail` adds half a line ending inside a
# multi-byte character; `bad-usage` makes one request's usage unreadable.
FAKE_CLAUDE_HELD_ON_ACCEPT = r'''
import json
import pathlib
import sys
import time

args = sys.argv[1:]
if "--version" in args:
    print("2.1.231 (Claude Code)")
    raise SystemExit(0)
if "--help" in args:
    print("--output-format stream-json --no-session-persistence --tools --allowed-tools "
          "--safe-mode --setting-sources --effort --model --resume --session-id "
          "--mcp-config --strict-mcp-config")
    raise SystemExit(0)
payload = sys.stdin.buffer.read()


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def assistant(message_id, blocks, usage):
    emit({"type": "assistant", "message": {"id": message_id, "role": "assistant",
          "model": "fake", "content": blocks, "usage": usage}})


emit({"type": "system", "subtype": "init", "tools": [], "model": "fake",
      "session_id": "11111111-2222-3333-4444-555555555555"})
assistant("msg-1", [{"type": "tool_use", "name": "mcp__finesub__next_task", "input": {}}],
          {"input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 0,
           "cache_creation_input_tokens": 900})
assistant("msg-2", [{"type": "text", "text": "submitting"}],
          {"input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 900,
           "cache_creation_input_tokens": 300})
assistant("msg-2", [{"type": "tool_use", "name": "mcp__finesub__submit", "input": {"payload": "x"}}],
          "garbage" if b"bad-usage" in payload else
          {"input_tokens": 20, "output_tokens": 400, "cache_read_input_tokens": 900,
           "cache_creation_input_tokens": 300})
if b"cut-tail" in payload:
    sys.stdout.buffer.write('{"type": "assistant", "message": {"content": "é'.encode("utf-8")[:-1])
    sys.stdout.flush()
pathlib.Path(__file__).with_suffix(".ready").write_text("1", encoding="utf-8")
time.sleep(60)
'''


def _held_driver(tmp_path: Path, *, keep_episodes: bool = False):
    from finesub.llm.agent import local_agent as la

    script = tmp_path / "fake_claude_held.py"
    script.write_text(FAKE_CLAUDE_HELD_ON_ACCEPT, encoding="utf-8")
    driver = la.ClaudeCodeLocalAgentDriver(
        la.ClaudeCodeDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            timeout_seconds=30,
            max_result_bytes=1024,
            max_event_bytes=8192,
            max_stderr_bytes=8192,
            keep_episodes=keep_episodes,
        )
    )
    return driver, script.with_suffix(".ready")


@pytest.mark.parametrize("keep_episodes", [False, True])
def test_exit_on_accept_ends_the_cli_at_once_and_settles_like_a_clean_exit(
    tmp_path: Path, keep_episodes: bool
) -> None:
    """The kill is ours and expected: no grace, no retained raw stream, no
    failure-pool capsule -- and the usage comes from the per-request events,
    one request per message id, since the `result` was never written."""

    driver, ready = _held_driver(tmp_path, keep_episodes=keep_episodes)
    started = time.monotonic()

    result = driver.run(
        [{"role": "user", "content": "call next_task cut-tail"}],
        task="correction",
        completion=ready.exists,
        exit_on_completion=True,
    )

    attempt = result.execution_attempt
    # Well inside the 15s grace and the fake's 60s sleep.
    assert time.monotonic() - started < 10
    assert attempt["ended_on_accept"] is True
    assert "reclaimed_after_completion" not in attempt
    assert "raw_events_retained" not in attempt
    assert "retained_unaccepted" not in attempt
    assert result.usage == {
        "input_tokens": 120,
        "output_tokens": 405,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 1200,
        "source": "claude_code_message_events",
    }
    assert attempt["usage"] == result.usage
    assert not attempt.get("warnings")
    if keep_episodes:
        assert attempt["episode_kept_by_setting"] is True
        capsule_root = Path(attempt["evidence_locator"]["absolute_at_write"])
        # Settled as a success: the raw original was replaced by the
        # normalized rows, exactly as on a clean exit.
        assert (capsule_root / "events" / "agent-events.jsonl").exists()
        assert not (capsule_root / "events" / "raw.jsonl").exists()
    else:
        assert "evidence_locator" not in attempt
        assert [p for p in (tmp_path / "runtime").rglob("events") if p.is_dir()] == []


def test_exit_on_accept_with_unreadable_usage_still_succeeds(tmp_path: Path) -> None:
    """Best effort: an unreadable per-request usage makes the usage unknown
    and says so once; the call is not failed, retried or held up over it."""

    driver, ready = _held_driver(tmp_path)

    result = driver.run(
        [{"role": "user", "content": "call next_task bad-usage"}],
        task="correction",
        completion=ready.exists,
        exit_on_completion=True,
    )

    attempt = result.execution_attempt
    assert attempt["ended_on_accept"] is True
    assert attempt["usage"] == {"source": "unavailable"}
    assert [w["event"] for w in attempt["warnings"]] == ["usage_unknown_after_accept"]
    assert "evidence_locator" not in attempt


def test_without_exit_on_accept_the_cli_keeps_its_grace(tmp_path: Path, monkeypatch) -> None:
    """The setting off (or the pseudo host, which never sets it): the same
    held CLI is left its grace and then reclaimed, raw stream kept."""

    from finesub.llm.agent import local_agent as la

    monkeypatch.setattr(la, "ACCEPTED_EXIT_GRACE_SECONDS", 1.5)
    driver, ready = _held_driver(tmp_path)
    started = time.monotonic()

    result = driver.run(
        [{"role": "user", "content": "call next_task"}],
        task="correction",
        completion=ready.exists,
    )

    attempt = result.execution_attempt
    assert time.monotonic() - started >= 1.5
    assert attempt["reclaimed_after_completion"] is True
    assert "ended_on_accept" not in attempt
    assert attempt["raw_events_retained"] is True


def test_codebuddy_message_usage_is_summed_per_request(tmp_path: Path) -> None:
    """Measured shape (CodeBuddy 2.137.1, hy3, 2026-09-30): a thinking-only
    message with null usage, then the request's text and tool_use under one
    id, the second carrying the numbers; input = cache write + cache read."""

    from finesub.llm.agent.local_agent import _normalize_codebuddy_events

    def assistant(message_id, kind, usage):
        return {"type": "assistant", "message": {"id": message_id, "model": "hy3",
                "content": [{"type": kind, "text": "t"}], "usage": usage}}

    zero = {"input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": None, "cache_read_input_tokens": None}
    rows = [
        {"type": "system", "subtype": "init", "tools": [], "session_id": "s"},
        assistant("think-1", "thinking", zero),
        assistant("req-1", "text", {**zero, "cache_creation_input_tokens": 0}),
        assistant("req-1", "text", {"input_tokens": 9564, "output_tokens": 40,
                                    "cache_creation_input_tokens": 9052,
                                    "cache_read_input_tokens": 512}),
        assistant("req-2", "text", {"input_tokens": 10228, "output_tokens": 674,
                                    "cache_creation_input_tokens": 756,
                                    "cache_read_input_tokens": 9472}),
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    _rows, usage, _violations, _content = _normalize_codebuddy_events(
        raw, native_search=False, max_bytes=1_000_000
    )

    assert usage == {
        "input_tokens": 19792,
        "output_tokens": 714,
        "cache_read_input_tokens": 9984,
        "cache_creation_input_tokens": 9808,
        "source": "codebuddy_message_events",
    }

    # With its `result`, the CLI's own total wins.
    rows.append({"type": "result", "subtype": "success", "is_error": False,
                 "usage": {"input_tokens": 1, "output_tokens": 1}})
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _rows, usage, _violations, _content = _normalize_codebuddy_events(
        raw, native_search=False, max_bytes=1_000_000
    )
    assert usage["source"] == "codebuddy_result_event" and usage["input_tokens"] == 1


def test_agy_step_usage_is_summed_when_the_result_never_came(tmp_path: Path) -> None:
    """Each finished request's `step_update` (state DONE) carries its usage; on
    a measured seven-request tool session those add up to `result.usage`
    exactly. A step's ACTIVE rows carry none, a tool step has none."""

    from finesub.llm.agent.local_agent import _normalize_agy_events

    def step(index, state, step_type, usage=None):
        update = {"conversation_id": "c", "step_index": index, "state": state,
                  "step_type": step_type}
        if usage is not None:
            update["usage"] = usage
        return {"event": "step_update", "step_update": update}

    first = {"input_tokens": 8326, "output_tokens": 187, "thinking_tokens": 125,
             "cache_read_tokens": 8140, "total_tokens": 8513}
    second = {"input_tokens": 3657, "output_tokens": 5215, "thinking_tokens": 4595,
              "cache_read_tokens": 16258, "total_tokens": 8872}
    rows = [
        {"event": "init", "conversation_id": "c",
         "init": {"tools": ["view_file"], "permission_mode": "x"}},
        step(0, "DONE", "user_input"),
        step(1, "ACTIVE", "agent_response"),
        step(1, "DONE", "agent_response", first),
        step(2, "ACTIVE", "tool"),
        step(2, "DONE", "tool"),
        step(3, "DONE", "agent_response", second),
        step(4, "ACTIVE", "tool"),
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(
        "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        # A line the kill cut inside a multi-byte character.
        + '{"event": "step_update", "step_update": {"text_delta": "é'.encode("utf-8")[:-1]
    )

    _rows, usage, violations, _content = _normalize_agy_events(
        raw, native_search=False, max_bytes=1_000_000
    )

    assert usage == {
        "input_tokens": 11983,
        "output_tokens": 5402,
        "reasoning_output_tokens": 4720,
        "cached_input_tokens": 24398,
        "total_tokens": 17385,
        "conversation_cumulative": False,
        "source": "agy_step_events",
    }
    # The cut line is dropped, not reported; only the missing result is.
    assert violations == ["agy event stream requires exactly one result event; got 0"]


def test_agy_turn_thinking_is_its_own_steps_beside_the_conversations_total(tmp_path: Path) -> None:
    """On a resumed conversation the result's usage is the conversation's; the turn's thinking is what its
    own finished steps reported (`degeneration.thinking_of` judges a turn by it)."""

    from finesub.llm.agent.local_agent import AgyLocalAgentDriver, _normalize_agy_events

    rows = [
        {"event": "init", "conversation_id": "c", "init": {"tools": ["view_file"], "permission_mode": "x"}},
        {"event": "step_update", "step_update": {"conversation_id": "c", "step_index": 7, "state": "DONE",
                                                 "step_type": "agent_response",
                                                 "usage": {"input_tokens": 900, "output_tokens": 300}}},
        {"event": "result", "result": {"status": "SUCCESS", "conversation_id": "c", "response": "ok",
                                       "usage": {"input_tokens": 5000, "output_tokens": 900,
                                                 "thinking_tokens": 4000}}},
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _rows, usage, _violations, _content = _normalize_agy_events(raw, native_search=False, max_bytes=1_000_000)
    assert usage["reasoning_output_tokens"] == 4000 and usage["conversation_cumulative"] is True
    assert usage["turn_reasoning_output_tokens"] == 0
    driver = AgyLocalAgentDriver(AgyDriverConfig(command=("agy",)))
    assert driver.reports_thinking("gemini-3.8-flash") and not driver.reports_thinking("claude-sonnet-4-6")


def test_only_a_cut_last_line_is_forgiven(tmp_path: Path) -> None:
    """Bad bytes or JSON on a complete line still count against the stream."""

    from finesub.llm.agent.local_agent import _read_event_text

    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b'{"a": 1}\n{"b": "\xff"}\n')
    with pytest.raises(LocalAgentPolicyViolationError, match="not UTF-8"):
        _read_event_text(raw, vendor="X", max_bytes=1024)
    raw.write_bytes(b'{"a": 1}\n{"b": 2')
    assert _read_event_text(raw, vendor="X", max_bytes=1024) == '{"a": 1}\n'
    # A complete last object without its newline is still an event.
    raw.write_bytes(b'{"a": 1}\n{"b": 2}')
    assert _read_event_text(raw, vendor="X", max_bytes=1024) == '{"a": 1}\n{"b": 2}'


def test_codex_reply_limit_sits_under_its_direct_tool_output_cut() -> None:
    """Direct calls are cut at `tool_output_token_limit` x 4 bytes of the
    result; the server measures the result as compact JSON (never smaller),
    so the limit sits under that with room left."""

    config = CodexDriverConfig()
    limit = config.channel.reply_ok_bytes
    assert config.channel.measure == "json" and config.channel.batch_tool
    assert 0 < limit <= config.tool_output_token_limit * 4 * 0.9
