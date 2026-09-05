"""The WorkBuddy driver: a Claude Code fork, and the five places it is not one.

Everything shared with Claude Code (capsule, process tree, deadline, secret
stripping, the `stream-json` parse) is covered by `test_llm_local_agent.py` and
is not repeated here. What is pinned below is exactly what measurement said
differs on this CLI (codebuddy 2.137.1, 2026-09-04), because each of those is a
place where copying the parent driver would have been silently wrong:

1. it has no "ignore user config/rules" mode, so two capability bits are False
   and the completion requirements are narrowed instead of faked;
2. `system.init` announces the whole built-in registry whatever `--tools` says,
   so the announced-set audit is off and the per-`tool_use` audit is what
   guards a call;
3. MCP tools are deferred unless the environment says otherwise, so the
   entitlement is only real with `CODEBUDDY_DEFER_TOOL_LOADING=0`;
4. `--tools` is the *only* boundary, so the harness's MCP tools go in it and
   `--allowedTools` (variadic, and it would swallow the prompt) is never sent;
5. a model the account cannot reach is reported in `errors`, not `result`, and
   is permanent rather than an auth failure or a transient one.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from finesub.llm.agent.local_agent import (
    LocalAgentError,
    LocalAgentPolicyViolationError,
    LocalAgentUnavailableError,
    WorkBuddyDriverConfig,
    WorkBuddyLocalAgentDriver,
)


# Mirrors the real CLI closely enough that the driver cannot tell: a bare
# version string with no vendor name, a help text without `--safe-mode`, an
# `init` that announces every built-in whatever `--tools` said, and a `result`
# whose failure channel is `errors`/`errors_info` rather than `result`.
FAKE_CODEBUDDY = r'''
import json
import os
import sys

ANNOUNCED = ["Agent", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
             "WebFetch", "WebSearch", "Skill", "ToolSearch"]

args = sys.argv[1:]
if "--version" in args:
    print("2.137.1")
    raise SystemExit(0)
if "--help" in args:
    print("--print --output-format stream-json --verbose --tools --allowedTools "
          "--disallowedTools --mcp-config --strict-mcp-config --setting-sources "
          "--no-session-persistence --resume --session-id --effort --model "
          "--fallback-model --permission-mode --max-turns")
    raise SystemExit(0)
payload = sys.stdin.buffer.read()
mode = "ok"
for marker in ("argv", "search", "tool", "auth-fail", "model-fail", "quota-fail",
               "rate-fail", "switched", "switched-fail", "env"):
    if marker.encode() in payload:
        mode = marker

# Announced unconditionally: the fork lists its registry, not this call's set.
print(json.dumps({"type": "system", "subtype": "init", "tools": ANNOUNCED,
                  "model": "fake", "mcp_servers": [],
                  "session_id": "11111111-2222-3333-4444-555555555555"}))
print(json.dumps({"type": "system", "subtype": "status", "status": None,
                  "session_id": "11111111-2222-3333-4444-555555555555"}))

if mode == "tool":
    print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "whoami"}}]}}))
if mode == "search":
    print(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_1", "name": "WebSearch",
         "input": {"query": "FineSub test"}}]}}))
    print(json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": [
            {"type": "text", "text": "see https://example.org/a"}]}]}}))

if mode == "argv":
    result = json.dumps(args)
elif mode == "env":
    result = json.dumps({
        "secret": "GEMINI_FREE" in os.environ,
        "defer": os.environ.get("CODEBUDDY_DEFER_TOOL_LOADING", ""),
        "mcp_cap": os.environ.get("MAX_MCP_OUTPUT_TOKENS", ""),
    })
else:
    result = "<translated>ok</translated>"

if mode == "switched-fail":
    # The paid model took the session over and then the session failed anyway.
    print(json.dumps({"type": "assistant", "message": {
        "role": "assistant", "model": "hy3-x",
        "content": [{"type": "text", "text": "partial"}]}}))
    detail = "500 upstream error"
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "usage": {}, "errors": [detail],
                      "errors_info": [{"status": 500, "code": 11141,
                                       "details": detail}]}))
elif mode in ("auth-fail", "model-fail", "quota-fail", "rate-fail"):
    if mode == "auth-fail":
        detail, info = "Not logged in - please sign in", {"status": 401, "code": 0}
    elif mode == "model-fail":
        detail = ("400 model [glm-9] service info not found\nCurrently "
                  "supported models for your account:\n  - glm-5.3")
        info = {"status": 400, "code": 11102, "category": "auth"}
    elif mode == "rate-fail":
        # Same status, same label, different business code: 6005-6008 is the
        # vendor's own `quota_request_limit` range.
        detail = "429 too many requests, please retry"
        info = {"status": 429, "code": 6006, "category": "quota"}
    else:
        detail = ("429 your usage has exceeded the rate limit, resetting at "
                  "2026-09-05 02:41:26 UTC+8; you can switch to another model")
        info = {"status": 429, "code": 6004, "category": "quota"}
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "usage": {},
                      "errors": [detail],
                      "errors_info": [{**info, "details": detail}]}))
elif mode != "switched-fail":
    message = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "not part of the answer"},
        {"type": "text", "text": result}]}
    if mode == "switched":
        # What the CLI's own interceptor leaves behind: the session opened on
        # the free line (system.init above) and a later message reports the
        # paid one.
        message["model"] = "hy3-x"
    print(json.dumps({"type": "assistant", "message": message}))
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "total_cost_usd": 0.0, "permission_denials": [],
                      "usage": {"input_tokens": 11, "output_tokens": 5,
                                "cache_read_input_tokens": 2,
                                "cache_creation_input_tokens": 0}}))
'''


def _driver(tmp_path: Path, monkeypatch, **overrides) -> WorkBuddyLocalAgentDriver:
    script = tmp_path / "fake_codebuddy.py"
    source = FAKE_CODEBUDDY
    if overrides.get("no_fallback_flag"):
        # An older CLI is one whose `--help` does not list the flag. It cannot
        # be an environment marker: the probe runs with an allowlisted
        # environment, so nothing unknown to `ENV_ALLOWLIST` reaches the CLI.
        source = source.replace("--fallback-model ", "")
    script.write_text(source, encoding="utf-8")
    monkeypatch.setenv("GEMINI_FREE", "secret-gemini")
    return WorkBuddyLocalAgentDriver(
        WorkBuddyDriverConfig(
            command=(sys.executable, str(script)),
            runtime_root=tmp_path / "runtime",
            model=overrides.get("model", "glm-5.3-flash"),
            effort=overrides.get("effort", ""),
            output_ceiling_hint=overrides.get("output_ceiling_hint", 0),
            fallback_model=overrides.get("fallback_model", ""),
            timeout_seconds=overrides.get("timeout_seconds", 15),
            max_result_bytes=overrides.get("max_result_bytes", 8192),
            max_event_bytes=overrides.get("max_event_bytes", 16384),
            max_stderr_bytes=overrides.get("max_stderr_bytes", 8192),
        )
    )


def _argv(driver: WorkBuddyLocalAgentDriver, **kwargs) -> list[str]:
    result = driver.run(
        [{"role": "user", "content": "argv"}], task="correction", **kwargs
    )
    return json.loads(result.content)


# --- capability probe -------------------------------------------------------


def test_probe_reports_the_two_isolation_bits_this_cli_lacks(
    tmp_path: Path, monkeypatch
) -> None:
    """No `--safe-mode`, and `--setting-sources ""` does not cover rule files.

    Reporting either of these True would be the one thing the driver contract
    forbids -- a second driver quietly claiming a guarantee it does not have.
    """

    probe = _driver(tmp_path, monkeypatch).probe()

    assert probe.available and probe.version == "2.137.1"
    assert probe.no_user_config is False
    assert probe.no_user_rules is False
    assert probe.structured_events and probe.no_persisted_session
    assert probe.can_restrict_tools and probe.supports_mcp_config
    assert probe.supports_session_reuse and probe.has_web_search


def test_narrowed_requirements_keep_the_driver_usable(
    tmp_path: Path, monkeypatch
) -> None:
    """The two missing bits are dropped from the requirement list, not faked."""

    driver = _driver(tmp_path, monkeypatch)
    probe = driver.probe()

    assert "no_user_config" not in driver.completion_requirements
    assert "no_user_rules" not in driver.completion_requirements
    assert driver.meets_requirements(probe) is True
    assert driver.meets_requirements(probe, native_search=True) is True


def test_isolation_metadata_admits_the_inheritance(
    tmp_path: Path, monkeypatch
) -> None:
    result = _driver(tmp_path, monkeypatch).run(
        [{"role": "user", "content": "hi"}], task="correction"
    )
    isolation = result.execution_attempt["isolation"]

    assert isolation["user_config"] == "inherited"
    assert isolation["user_rules"] == "inherited"
    # And says what does keep this machine's rule files out of a call.
    assert isolation["rule_isolation"] == "fresh_capsule_cwd"


# --- argv -------------------------------------------------------------------


def test_completion_argv_entitles_nothing_and_sends_no_permission_flags(
    tmp_path: Path, monkeypatch
) -> None:
    args = _argv(_driver(tmp_path, monkeypatch, effort="high"))

    assert "--print" in args
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in args
    assert "--strict-mcp-config" in args
    assert "--no-session-persistence" in args
    assert args[args.index("--setting-sources") + 1] == ""
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "glm-5.3-flash"
    assert args[args.index("--effort") + 1] == "high"
    # `-y` is not needed and is not sent: MCP tools, reads and `WebSearch` all
    # run under the default permission mode (measured 2026-09-04), and this
    # flag is `--dangerously-skip-permissions`, wider than anything the other
    # four drivers use.
    assert "-y" not in args and "--dangerously-skip-permissions" not in args
    assert "--permission-mode" not in args
    # Variadic, so a list here would swallow the positional prompt -- and it
    # grants nothing, because `--tools` is already the whole boundary.
    assert "--allowedTools" not in args and "--allowed-tools" not in args
    # Flags this fork does not have; sending one would abort the call.
    assert "--safe-mode" not in args
    assert "--disable-slash-commands" not in args
    assert "--add-dir" not in args
    # No output ceiling reaches this CLI, which is what keeps the API-side
    # single-pool clamp from ever rationing an agent's answer.
    assert not any("max-output" in arg or "max_tokens" in arg for arg in args)


def test_reusing_a_session_swaps_persistence_for_a_resume_handle(
    tmp_path: Path, monkeypatch
) -> None:
    """The fork keeps its parent's `--resume` / `--session-id` pair.

    Only the argv is pinned: the probe reports the capability off the help
    text, exactly as the Claude Code driver does, and a live cross-invocation
    resume has not been run on this CLI.
    """

    args = _argv(
        _driver(tmp_path, monkeypatch),
        session_scope="assignment",
        conversation_key="k1",
        conversation_handle="11111111-2222-3333-4444-555555555555",
    )

    assert "--no-session-persistence" not in args
    assert args[args.index("--resume") + 1] == (
        "11111111-2222-3333-4444-555555555555"
    )


def test_native_argv_entitles_only_the_tool_that_can_actually_run(
    tmp_path: Path, monkeypatch
) -> None:
    """`WebFetch` is offered by the CLI and refused by its permission layer."""

    args = _argv(_driver(tmp_path, monkeypatch), native_search=True)

    assert args[args.index("--tools") + 1] == "WebSearch"


def test_tool_session_argv_puts_the_mcp_tools_in_the_tool_set(
    tmp_path: Path, monkeypatch
) -> None:
    """An MCP tool missing from `--tools` is never offered on this CLI."""

    args = _argv(
        _driver(tmp_path, monkeypatch),
        mcp_server={
            "command": "python",
            "args": ["-m", "server"],
            "env": {"FINESUB_MCP_SESSION": "s1"},
            "tools": ["next_task", "submit"],
        },
    )

    declared = json.loads(args[args.index("--mcp-config") + 1])
    assert list(declared["mcpServers"]) == ["finesub"]
    assert declared["mcpServers"]["finesub"]["env"] == {"FINESUB_MCP_SESSION": "s1"}
    assert args[args.index("--tools") + 1] == (
        "mcp__finesub__next_task,mcp__finesub__submit"
    )
    assert "--allowedTools" not in args


# --- environment ------------------------------------------------------------


def test_tool_deferral_is_off_and_the_mcp_cap_is_raised_only_when_needed(
    tmp_path: Path, monkeypatch
) -> None:
    """Both knobs are prerequisites, not tuning.

    With deferral on, the model is offered `ToolSearch` instead of the tools
    the call entitled it to. With the stock MCP cap, a production-sized
    `next_task` reply is replaced by a path to a file this worker may not open.
    """

    driver = _driver(tmp_path, monkeypatch)

    plain = json.loads(
        driver.run([{"role": "user", "content": "env"}], task="correction").content
    )
    assert plain["secret"] is False, "the API key must not reach the CLI"
    assert plain["defer"] == "0"
    assert plain["mcp_cap"] == ""

    with_tools = json.loads(
        driver.run(
            [{"role": "user", "content": "env"}],
            task="correction",
            mcp_server={
                "command": "python",
                "args": [],
                "env": {},
                "tools": ["next_task"],
            },
        ).content
    )
    assert with_tools["defer"] == "0"
    assert with_tools["mcp_cap"] == "200000"


# --- event stream -----------------------------------------------------------


def test_the_announced_registry_is_recorded_but_never_a_leak_warning(
    tmp_path: Path, monkeypatch
) -> None:
    """`system.init` lists every built-in on this fork, `--tools` regardless.

    Auditing that announcement would raise a warning on every single call,
    which is a guard that has stopped meaning anything.
    """

    driver = _driver(tmp_path, monkeypatch)
    result = driver.run([{"role": "user", "content": "hi"}], task="correction")

    announcements = [
        row for row in result.normalized_events if row.get("event") == "system"
    ]
    # The registry is still recorded -- a contract change stays visible in the
    # capsule -- it just is not read as a leak.
    assert any("Bash" in (row.get("tools") or ()) for row in announcements)
    assert not any("unentitled_tools_offered" in row for row in announcements)
    assert result.content == "<translated>ok</translated>"


def test_a_real_tool_call_outside_the_entitlement_still_fails_the_call(
    tmp_path: Path, monkeypatch
) -> None:
    """The guard that matters is the invocation, not the announcement."""

    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentPolicyViolationError, match="forbidden tool Bash"):
        driver.run([{"role": "user", "content": "tool"}], task="correction")


def test_a_native_round_records_the_search_and_its_urls(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run(
        [{"role": "user", "content": "search"}],
        task="research",
        native_search=True,
    )

    searches = [
        row
        for row in result.normalized_events
        if row.get("event") == "item.completed"
        and row.get("item_type") == "web_search"
    ]
    assert [row["tool"] for row in searches] == ["WebSearch"]
    assert searches[0]["query"] == "FineSub test"
    # The URLs ride the *next* message, so the join has to happen for this
    # fork exactly as it does for its parent.
    assert "https://example.org/a" in searches[0]["urls"]


def test_usage_names_this_fork_rather_than_its_parent(
    tmp_path: Path, monkeypatch
) -> None:
    result = _driver(tmp_path, monkeypatch).run(
        [{"role": "user", "content": "hi"}], task="correction"
    )

    assert result.usage["source"] == "workbuddy_result_event"
    assert result.usage["input_tokens"] == 11
    # A `thinking` block is not part of the answer.
    assert result.content == "<translated>ok</translated>"


# --- failure classification -------------------------------------------------


def test_a_model_the_account_cannot_reach_is_permanent_not_auth(
    tmp_path: Path, monkeypatch
) -> None:
    """It looks like an auth failure and is a configuration one.

    The vendor answers with HTTP 400/401 and `category: "auth"`, but the fix is
    a catalog row, not a login. Classifying it as unavailable would send the
    operator to sign in again; classifying it as transient would spend the
    pool's failure streak on a call that can never succeed.
    """

    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentPolicyViolationError) as excinfo:
        driver.run([{"role": "user", "content": "model-fail"}], task="correction")
    # The reason is read out of `errors`, where this CLI puts it -- `result` is
    # absent on a failing terminal record.
    assert "service info not found" in str(excinfo.value)


def test_a_spent_daily_allowance_is_quota_not_transient(
    tmp_path: Path, monkeypatch
) -> None:
    """The vendor says so outright, so the router should not have to probe.

    Measured 2026-09-04: `hy4-preview`'s free daily allowance ran out while
    `hy3` answered normally in the same minute, and the failure carried
    `status: 429`, `code: 6004`, `category: "quota"` plus a reset time. Left as
    transient, that costs a probe and then a call on every other model of the
    same tier before anything is recorded.

    `6004` is the deciding field, not the 429 -- see
    `test_a_rate_limit_is_not_a_spent_allowance` for the other 429.
    """

    from finesub.llm.agent.local_agent import LocalAgentQuotaError

    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentQuotaError) as excinfo:
        driver.run([{"role": "user", "content": "quota-fail"}], task="correction")
    assert "out of allowance" in str(excinfo.value)


def test_the_typed_fields_are_read_not_the_prose(
    tmp_path: Path, monkeypatch
) -> None:
    """§11.1 forbids guessing exhaustion from wording, not reading a field.

    The row keeps the vendor's own `code` / `category` / `status`, and the
    classifier asks those -- so a vendor that rephrases its message changes
    nothing, and a vendor that stops emitting the fields degrades to
    `transient` (the old behaviour) rather than to a wrong guess.
    """

    from finesub.llm.agent.local_agent import _workbuddy_quota_exhausted

    assert _workbuddy_quota_exhausted({"error_categories": ["quota"]}) is True
    assert _workbuddy_quota_exhausted({"error_statuses": [429]}) is True
    assert _workbuddy_quota_exhausted({"error_categories": ["auth"]}) is False
    # Prose alone is never enough.
    assert _workbuddy_quota_exhausted({"error": "usage limit exceeded"}) is False


def test_a_rate_limit_is_not_a_spent_allowance(tmp_path: Path, monkeypatch) -> None:
    """The 429 that means "slow down" must not freeze the line for two hours.

    ⚠ `category` carries no information the status does not: the function
    building `errors_info` for the stream
    (`ResultMessageUtils.extractStructuredErrorInfo`, bundle 2.137.1) sets it
    from the status alone -- `429 -> "quota"` -- and never consults the
    classifier that knows the codes. Reading either as evidence, as this did
    until 2026-09-04, turns every burst of requests into a two-hour freeze of
    a model line that is answering fine.

    The **code** is the field that separates them, and the band's own names
    are the whole answer (`ServerErrorCode`)::

        6000 CraftRateLimit    6001 TPS  6002 TPM  6003 TPH  6004 TPD
                               6005 RPS  6006 RPM  6007 RPH  6008 RPD

    -- so the line is **per day vs per shorter window**, not tokens vs
    requests, which is how this read on its first pass. The CLI draws it in
    the same place: `isCraftDailyQuotaBusinessCode` is exactly `{6004, 6008}`
    and `isRequestLevelRetryableError` refuses to retry on it, while
    `isTransientRateLimitBusinessCode` takes the rest of the band plus `14003`.
    An unrecognised code keeps the 429 rule, which is what the CLI's own
    fallback does with a 429 it cannot place.
    """

    from finesub.llm.agent.local_agent import (
        LocalAgentQuotaError,
        LocalAgentTransientError,
        _workbuddy_quota_exhausted,
    )

    # Per second / minute / hour, on both halves of the band.
    for code in (6000, 6001, 6002, 6003, 6005, 6006, 6007, 10105, 14003, 15001):
        assert (
            _workbuddy_quota_exhausted(
                {"error_codes": [code], "error_statuses": [429],
                 "error_categories": ["quota"]}
            )
            is False
        ), code
    # Per day (both halves), the `UsageLimit*` set, and an unplaceable 429.
    for code in (6004, 6008, 14001, 14012, 14018, 99999):
        assert (
            _workbuddy_quota_exhausted(
                {"error_codes": [code], "error_statuses": [429],
                 "error_categories": ["quota"]}
            )
            is True
        ), code
    # A daily code answers on its own, whatever status carried it.
    assert _workbuddy_quota_exhausted({"error_codes": [6008]}) is True

    driver = _driver(tmp_path, monkeypatch)
    with pytest.raises(LocalAgentTransientError) as excinfo:
        driver.run([{"role": "user", "content": "rate-fail"}], task="correction")
    assert not isinstance(excinfo.value, LocalAgentQuotaError)


def test_the_paid_twin_is_offered_only_where_the_catalog_says_so(
    tmp_path: Path, monkeypatch
) -> None:
    """`--fallback-model` is the one flag here that can spend money.

    Owner decision 2026-09-04: `hy3` and `hy4-preview` should switch to their
    paid twins when the free allowance runs out and carry on, rather than stop
    the run. The twin is a **catalog** value, so a row that names none sends no
    flag at all -- the vendor's default is to fail, and that stays the default
    for every row nobody decided about.
    """

    from finesub.llm.routing.execution_policy import _workbuddy_fallback_model
    from finesub.llm.routing.model_catalog import get_model_catalog_entry_for_tier

    # The pairs, straight from the vendor's product config (2026-09-04):
    # the free lines bill x0.00 and these bill x0.05 / x0.29.
    assert _workbuddy_fallback_model("hy3") == "hy3-x"
    assert _workbuddy_fallback_model("hy4-preview") == "hy4-preview-x"
    # ⚠ `hy4-preview-x`, not `hy4-x`: the vendor's id keeps the `-preview`.
    # A wrong id would make the interceptor skip silently.
    assert (
        get_model_catalog_entry_for_tier("hy4-preview", "LOCAL_WORKBUDDY")
        is not None
    )
    # Everything else bills credits already and has no free twin to fall back
    # *from*, so nothing is sent for them.
    for model in ("glm-5.3-flash", "deepseek-v4-flash", "deepseek-v4-pro"):
        assert _workbuddy_fallback_model(model) == "", model
    assert _workbuddy_fallback_model("not-in-the-catalog") == ""

    args = _argv(_driver(tmp_path, monkeypatch, fallback_model="hy3-x"))
    assert "--fallback-model" in args
    assert args[args.index("--fallback-model") + 1] == "hy3-x"

    bare = _argv(_driver(tmp_path, monkeypatch))
    assert "--fallback-model" not in bare

    # A row pointing at itself is not a fallback; the CLI logs and skips it,
    # so there is no reason to send it.
    same = _argv(
        _driver(tmp_path, monkeypatch, model="hy3-x", fallback_model="hy3-x")
    )
    assert "--fallback-model" not in same


def test_a_cli_too_old_for_the_flag_loses_the_net_and_says_so(
    tmp_path: Path, monkeypatch, reported
) -> None:
    """Drop the fallback, not the call -- and do not do it quietly.

    An unrecognised option is a hard exit for this CLI, so sending a flag it
    has never heard of would turn "no paid safety net" into "no target". The
    flag is therefore gated on the vendor's own `--help`, the same way every
    other optional flag here is.

    The warning exists because the silent version is worse than the failure it
    prevents: the operator configured a paid twin precisely so that a spent
    allowance would not stop the run, and on this CLI it still will.
    """

    args = _argv(
        _driver(
            tmp_path, monkeypatch, fallback_model="hy3-x", no_fallback_flag=True
        )
    )
    assert "--fallback-model" not in args

    notices = [
        w for w in reported.warnings if w.code == "agent-fallback-unsupported"
    ]
    assert len(notices) == 1
    assert "hy3-x" in notices[0].message


def test_a_session_the_paid_twin_finished_says_so_once(
    tmp_path: Path, monkeypatch, reported
) -> None:
    """It carries on -- and both the operator and the artifact hear about it.

    The switch happens inside the CLI, so by the time the stream reaches us it
    is done and nothing here could veto it. That is the argument for saying it
    out loud: the run succeeded, but it cost credits the binding did not
    obviously ask for. It goes through `_stream_warnings` so the execution
    attempt keeps a copy -- a bill deserves a record, not just a console line.

    ⚠ The answer is filed under **who answered**, not who was asked. Recording
    a paid reply under the free model's name would make the task report say the
    free line produced something it never produced.
    """

    driver = _driver(tmp_path, monkeypatch, fallback_model="hy3-x", model="hy3")
    result = driver.run(
        [{"role": "user", "content": "switched"}], task="correction"
    )
    assert "ok" in result.content

    notices = [
        w
        for w in reported.warnings
        if w.code == "agent-driver-stream" and "hy3-x" in w.message
    ]
    assert len(notices) == 1
    assert "完成本次会话" in notices[0].message
    assert "fallback_model" in notices[0].message

    # Both names survive into the attempt, and the result carries the one that
    # actually answered.
    assert result.reported_model == "hy3-x"
    assert result.execution_attempt["configured_model"] == "hy3"
    recorded = [
        row
        for row in result.execution_attempt.get("warnings", [])
        if "hy3-x" in str(row.get("message", ""))
    ]
    assert len(recorded) == 1

    # The same stream without the switch is silent, and nothing is rewritten:
    # the notice keys off the answering model, not off the flag being present.
    reported.warnings.clear()
    quiet = _driver(tmp_path, monkeypatch, fallback_model="hy3-x", model="hy3")
    quiet_result = quiet.run(
        [{"role": "user", "content": "hello"}], task="correction"
    )
    assert quiet_result.reported_model == "hy3"
    assert "configured_model" not in quiet_result.execution_attempt
    assert not [w for w in reported.warnings if "hy3-x" in w.message]


def test_a_switch_that_did_not_save_the_call_does_not_claim_it_did(
    tmp_path: Path, monkeypatch, reported
) -> None:
    """The notice runs before the stream is classified, so it cannot promise.

    A paid model can take the session over, call a tool, and still fail. The
    first version announced "并完成本次会话" the moment it saw the paid model
    answer, so the operator read that the run had finished and then watched the
    task error out. The terminal record is already in the same rows, so the
    wording is taken from it rather than assumed.
    """

    driver = _driver(tmp_path, monkeypatch, fallback_model="hy3-x", model="hy3")
    with pytest.raises(LocalAgentError):
        driver.run(
            [{"role": "user", "content": "switched-fail"}], task="correction"
        )

    notices = [w for w in reported.warnings if "hy3-x" in w.message]
    assert len(notices) == 1
    assert "仍然失败" in notices[0].message
    assert "完成本次会话" not in notices[0].message
    # Still says the call was billed: the switch happened either way.
    assert "计费" in notices[0].message


def test_the_manual_rows_for_turning_off_billing_actually_load(
    tmp_path: Path,
) -> None:
    """The one instruction that has to work, checked against the parser.

    Telling an operator how to stop automatic billing is worth nothing if the
    snippet does not load. The first version printed `fact_id|fallback_model`,
    which fails twice over: `provider_tier` / `api_model_id` /
    `max_input_tokens` are required, and an override of a packaged id is a
    **whole-row replacement** -- the columns left out do not keep the shipped
    values, they fall back to dataclass defaults.

    So the manual carries the full rows, and this reads them straight out of
    it: the block cannot rot into something that no longer parses, and it
    cannot silently drift from the packaged row it is copied from.
    """

    from finesub.llm.routing.model_catalog import (
        load_model_catalog,
        merge_catalogs,
    )

    manual = Path("docs/manual/agent.md").read_text(encoding="utf-8")
    blocks = [
        block
        for block in manual.split("```")
        if "local-workbuddy-hy3|" in block and "fact_id|" in block
    ]
    assert len(blocks) == 1, "one snippet, or this guard is reading the wrong one"
    rows = [
        line.strip()
        for line in blocks[0].splitlines()
        if "|" in line and not line.strip().startswith("text")
    ]

    override_path = tmp_path / "model_catalog.psv"
    override_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    override = load_model_catalog(override_path, self_reported=True)
    assert {entry.fact_id for entry in override} == {
        "local-workbuddy-hy3",
        "local-workbuddy-hy4",
    }
    assert all(entry.fallback_model == "" for entry in override)

    # And it is otherwise the shipped row: every other fact matches, so a
    # packaged change that the snippet does not follow turns this red instead
    # of quietly handing operators a stale copy.
    packaged = {entry.fact_id: entry for entry in load_model_catalog()}
    for entry in override:
        shipped = packaged[entry.fact_id]
        assert shipped.fallback_model, "the row this turns off must have one"
        for field in (
            "max_input_tokens",
            "max_output_tokens",
            "context_window",
            "thinking_levels",
            "quota_pool",
            "quality_score",
            "api_model_id",
        ):
            assert getattr(entry, field) == getattr(shipped, field), field

    # Layered on the packaged catalog, the switch is off and nothing else moved.
    merged = {
        entry.fact_id: entry
        for entry in merge_catalogs(load_model_catalog(), override)
    }
    assert merged["local-workbuddy-hy3"].fallback_model == ""
    assert merged["local-workbuddy-hy3"].quota_pool == "WORKBUDDY_HY3"
    assert merged["local-workbuddy-glm-5_3-flash"].hint_output_ceiling is True


def test_every_row_meters_its_own_allowance(tmp_path: Path, monkeypatch) -> None:
    """One login, several separately metered model lines.

    Measured 2026-09-04: `hy4-preview` was exhausted while `hy3` answered, and
    the server's own advice was "switch to another model and carry on". A
    shared pool would freeze all six for two hours the first time one free line
    ran out -- the expensive direction under docs/llm_local_agent.md §11.1.
    """

    from finesub.llm.routing.model_routes import default_model_routes

    rows = [
        entry
        for entry in default_model_routes().facts.values()
        if entry.provider_tier == "LOCAL_WORKBUDDY"
    ]
    pools = [entry.effective_quota_pool for entry in rows]

    assert len(set(pools)) == len(rows)
    assert not any(pool == "LOCAL_WORKBUDDY" for pool in pools)


def test_a_signed_out_cli_is_unavailable_rather_than_a_policy_violation(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="not authenticated"):
        driver.run([{"role": "user", "content": "auth-fail"}], task="correction")


# --- wiring -----------------------------------------------------------------


def test_the_tier_reaches_this_driver_and_nothing_else() -> None:
    from finesub.llm.routing.execution_policy import (
        ExecutionSettings,
        driver_for_provider_tier,
    )

    driver = driver_for_provider_tier(
        ExecutionSettings(), provider_tier="LOCAL_WORKBUDDY", model="glm-5.3-flash"
    )

    assert isinstance(driver, WorkBuddyLocalAgentDriver)
    assert driver.config.model == "glm-5.3-flash"
    assert driver.driver_id == "codebuddy"


def test_execution_identity_records_what_this_driver_cannot_promise() -> None:
    from finesub.llm.agent.local_agent import local_agent_execution_profiles

    profile = local_agent_execution_profiles()["LOCAL_WORKBUDDY"]

    assert profile["driver_id"] == "codebuddy"
    # Its own protocol version: a fork that drifts invalidates its own
    # checkpoints, not Claude Code's.
    assert profile["protocol_version"] == "codebuddy-stream-json-v1"
    assert profile["configuration"]["user_configuration"] == "inherited"
    assert profile["configuration"]["tool_deferral"] == "disabled"
    assert profile["toolset"] == {"completion": [], "native": ["WebSearch"]}
    assert profile["configuration_digest"]


def test_every_packaged_row_names_a_model_and_a_target() -> None:
    """The account's menu is not the CLI's, so the shipped rows are the
    verified ones -- and each has to be reachable."""

    from finesub.llm.routing.model_routes import default_model_routes

    routes = default_model_routes()
    facts = {
        fact_id
        for fact_id, entry in routes.facts.items()
        if entry.provider_tier == "LOCAL_WORKBUDDY"
    }
    assert facts == {
        "local-workbuddy-hy3",
        "local-workbuddy-hy4",
        "local-workbuddy-glm-5_3-flash",
        "local-workbuddy-deepseek-v4-flash",
        "local-workbuddy-deepseek-v4-pro",
    }
    claimed = {
        routes.targets[target_id].fact_id
        for target_id in routes.targets
        if target_id.startswith("local-workbuddy-")
    }
    assert claimed == facts


# --- executable resolution --------------------------------------------------


def test_the_bash_shim_is_read_rather_than_executed(tmp_path, monkeypatch) -> None:
    """PATH holds a bash script with no extension, which must never be spawned.

    The shim names the entry script, so it is parsed for that -- and the
    interpreter is resolved separately, because the shim pins a Node version
    directory that an app update renames (observed 2026-09-04).
    """

    from finesub.llm.agent import local_agent as la

    if la.os.name != "nt":
        pytest.skip("the shim branch is Windows-only")

    entry = tmp_path / "cli" / "bin" / "codebuddy"
    entry.parent.mkdir(parents=True)
    entry.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    shim = tmp_path / "bin" / "codebuddy"
    shim.parent.mkdir(parents=True)
    shim.write_text(
        f'#!/bin/bash\nexec "{tmp_path}/stale/node.exe" "{entry}" "$@"\n',
        encoding="utf-8",
    )
    node = tmp_path / "node" / "22.0.0" / "node.exe"
    node.parent.mkdir(parents=True)
    node.write_text("", encoding="utf-8")
    (tmp_path / "node" / "current").write_text("22.0.0", encoding="utf-8")

    monkeypatch.setattr(la, "_workbuddy_node_versions", lambda: tmp_path / "node")
    monkeypatch.setattr(
        la.shutil, "which", lambda name: str(shim) if name == "codebuddy" else None
    )

    assert la._resolve_shell_free_command(("codebuddy",)) == (
        str(node.resolve()),
        str(entry.resolve()),
    )


# --- the output-ceiling hint (allowlisted) ----------------------------------


def test_only_the_allowlisted_model_is_told_its_ceiling(
    tmp_path: Path, monkeypatch
) -> None:
    """One model measured to benefit, so one model gets the clause.

    Measured 2026-09-04 on one real correction window at `high` effort:
    `glm-5.3-flash` finished only when told its true 32000 (and still timed out
    when told 64000), while `deepseek-v4-flash` timed out at every number
    including its true 50000 and a halved 25000 -- its thinking blocks stayed
    134k-139k characters throughout. An allowlist rather than a default is the
    honest shape for a mechanism that worked on one of three models.
    """

    from finesub.llm.routing.execution_policy import (
        ExecutionSettings,
        driver_for_provider_tier,
    )
    from finesub.llm.routing.model_routes import default_model_routes

    # The switch is a catalog column, so the allowlist is whatever the rows say.
    routes = default_model_routes()
    switched_on = {
        entry.api_model_id
        for entry in routes.facts.values()
        if entry.provider_tier == "LOCAL_WORKBUDDY" and entry.hint_output_ceiling
    }
    assert switched_on == {"glm-5.3-flash"}
    hints = {
        model: driver_for_provider_tier(
            ExecutionSettings(), provider_tier="LOCAL_WORKBUDDY", model=model
        ).config.output_ceiling_hint
        for model in ("glm-5.3-flash", "deepseek-v4-flash", "hy3", "hy4-preview")
    }
    assert hints == {
        # The catalog's number, not a literal: a wrong one is worse than none.
        "glm-5.3-flash": 32_000,
        "deepseek-v4-flash": 0,
        "hy3": 0,
        "hy4-preview": 0,
    }


def test_the_clause_reaches_the_tool_session_bootstrap_only(
    tmp_path: Path, monkeypatch
) -> None:
    """It asks the worker to break a turn with a tool call, so it needs tools."""

    driver = _driver(tmp_path, monkeypatch, output_ceiling_hint=32_000)

    with_tools = _argv(
        driver,
        mcp_server={
            "command": "python",
            "args": [],
            "env": {},
            "tools": ["next_task", "pull_status"],
        },
    )
    assert any("32000 output tokens" in arg for arg in with_tools)
    assert any("pull_status once" in arg for arg in with_tools)

    # The capsule path has no tools to call, so the clause would be an
    # instruction the worker cannot follow.
    assert not any("32000 output tokens" in arg for arg in _argv(driver))


def test_a_model_off_the_allowlist_is_told_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    args = _argv(
        _driver(tmp_path, monkeypatch),
        mcp_server={"command": "python", "args": [], "env": {}, "tools": ["submit"]},
    )

    assert not any("output tokens" in arg for arg in args)


def test_an_agent_call_is_sent_no_output_ceiling_at_all() -> None:
    """What makes the single-pool clamp API-only, stated where it can break.

    The clamp added on 2026-09-04 caps what the API asks for. It cannot touch
    an agent call because nothing sends one an output ceiling: the agent branch
    builds its own kwargs without `max_tokens`, and no WorkBuddy argv carries
    an output flag. Both halves are asserted here -- a future edit that added
    either would silently start rationing agent output.
    """

    import ast
    from pathlib import Path as _Path

    import finesub.llm.client as client_module

    tree = ast.parse(_Path(client_module.__file__).read_text(encoding="utf-8"))
    agent_kwargs = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "agent_call_kwargs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
    ]
    assert agent_kwargs, "the agent branch still builds its own call kwargs"
    for call in agent_kwargs:
        names = {keyword.arg for keyword in call.keywords}
        assert "max_tokens" not in names, (
            "the agent branch would now ration output; the clamp is meant to be "
            "API-only"
        )


def test_the_switch_rides_routing_identity_not_the_driver_digest() -> None:
    """It is catalog data, so it invalidates checkpoints through the catalog.

    Flipping the column changes the prompt a worker reads, which has to move
    resume identity -- and it does, because facts are hashed into
    `routing_identity_digest` unless they are listed as advisory. Duplicating
    it in the driver's own digest would make one edit move two hashes.
    """

    from finesub.llm.agent.local_agent import local_agent_execution_profiles
    from finesub.llm.routing.model_routes import ADVISORY_FACT_FIELDS

    configuration = local_agent_execution_profiles()["LOCAL_WORKBUDDY"]["configuration"]
    assert "output_ceiling_hint_models" not in configuration
    assert "hint_output_ceiling" not in ADVISORY_FACT_FIELDS
