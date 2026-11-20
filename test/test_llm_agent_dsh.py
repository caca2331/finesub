"""The dsh driver: what it configures per call, and what it refuses.

dsh is the one agent backend with no input channel but its command line, so
these tests are mostly about the two consequences: everything a call needs
rides one patch overlay, and a call that cannot fit on a command line is
refused instead of truncated.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.agent.agent_mcp_server import TOOL_NAMES
from finesub.llm.agent.local_agent import (
    DSH_INLINE_PROBE_MAX_CHARS,
    DriverProbe,
    DshDriverConfig,
    DshLocalAgentDriver,
    LocalAgentPolicyViolationError,
    LocalAgentTransientError,
    MCP_SERVER_NAME,
    probing_driver,
)


@pytest.fixture(autouse=True)
def _vetted_bundle(monkeypatch):
    """Stand in for a dsh whose composed profile is exactly the vetted snapshot.

    These tests drive `_argv` with a made-up command, so the real inventory
    dump cannot run -- and the vetted-plugin policy fails closed rather than
    letting a call go out unprotected. Stubbing the dump keeps them on the
    production path (policy on, nothing unknown) instead of switching it off.
    Plugin-policy behaviour itself lives in `test_llm_agent_dsh_plugins.py`.
    """

    from finesub.llm.agent import local_agent

    ids = DshDriverConfig().expected_plugin_ids
    monkeypatch.setattr(
        local_agent, "_dsh_dump_plugin_ids", lambda *_a, **_k: ids
    )
    local_agent._DSH_COMPOSITION_CACHE.clear()
    yield
    local_agent._DSH_COMPOSITION_CACHE.clear()


def _capsule(tmp_path: Path, prompt: str = "BOOTSTRAP TEXT") -> SimpleNamespace:
    messages = tmp_path / "input" / "messages.json"
    messages.parent.mkdir(parents=True, exist_ok=True)
    messages.write_text(
        json.dumps([{"role": "user", "content": prompt}]), encoding="utf-8"
    )
    return SimpleNamespace(root=tmp_path, episode_id="ep", messages_path=messages)


def _server() -> dict:
    return {
        "command": "python",
        "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
        "env": {"FINESUB_MCP_SESSION": "s1", "FINESUB_MCP_ROOT": "r"},
        "tools": list(TOOL_NAMES),
    }


def _driver(**overrides) -> DshLocalAgentDriver:
    driver = DshLocalAgentDriver(
        DshDriverConfig(model="deepseek-official/deepseek-flash", **overrides)
    )
    driver._resolved_command = ("node.exe", "bin.js")
    return driver


def _entries(driver: DshLocalAgentDriver, tmp_path: Path, **kwargs) -> list[dict]:
    argv = driver._argv(
        _capsule(tmp_path),
        native_search=kwargs.pop("native_search", False),
        probe=DriverProbe(available=True),
        mcp_server=kwargs.pop("mcp_server", _server()),
        **kwargs,
    )
    return json.loads(Path(argv[argv.index("--patch") + 1]).read_text(encoding="utf-8"))


def _by_id(entries: list[dict], entry_id: str) -> dict:
    return next(entry for entry in entries if entry.get("id") == entry_id)


def _effort(entries: list[dict]) -> str | None:
    """The thinking level the overlay selects, if any."""

    return _by_id(entries, "agent-default-model")["config"].get("reasoningEffort")


def test_a_call_without_the_harness_server_is_refused_not_squeezed_into_argv(
    tmp_path,
) -> None:
    """The capsule transport is impossible here, so it must not be attempted.

    dsh reads its task from its command line and nowhere else; a production
    window would run past what Windows accepts and fail as a spawn error with
    nothing to say about the cause. Refusing names the cause.
    """

    with pytest.raises(LocalAgentPolicyViolationError, match="tool protocol only"):
        _driver()._argv(
            _capsule(tmp_path),
            native_search=False,
            probe=DriverProbe(available=True),
            mcp_server=None,
        )


def test_a_quota_probe_rides_argv_without_the_harness_server(tmp_path) -> None:
    """The one call dsh takes without MCP: a probe asks for a word, not a window.

    Before this, the refusal above also caught every probe, so `agent-ping`
    reported each dsh row `failed` without calling dsh and the router's
    exhaustion probe never reached it. The overlay is the production one minus
    the MCP entry -- same model route, same denied plugins.
    """

    driver = probing_driver(_driver())
    driver._resolved_command = ("node.exe", "bin.js")
    argv = driver._argv(
        _capsule(tmp_path, "Reply with the single word: OK"),
        native_search=False,
        probe=DriverProbe(available=True),
        mcp_server=None,
    )

    assert argv[-1] == "Reply with the single word: OK"
    entries = json.loads(
        Path(argv[argv.index("--patch") + 1]).read_text(encoding="utf-8")
    )
    assert not any("insert" in entry for entry in entries)
    assert _by_id(entries, "agent-default-model")["config"] == {
        "provider": "deepseek-official",
        "model": "deepseek-flash",
    }
    denied = {entry["id"] for entry in entries if entry.get("disabled")}
    assert set(DshDriverConfig().disabled_tool_plugins) | {"tool-web"} <= denied
    # The user's `$DSH_HOME/AGENTS.md` stays out of every call.
    assert "agent-instructions" in denied


def test_the_probe_opening_is_only_for_probes_and_only_for_short_asks(
    tmp_path,
) -> None:
    """A production driver still refuses; a probe carrying a window still refuses."""

    assert DshDriverConfig().inline_probe is False
    assert _driver().config.inline_probe is False
    probe_driver = probing_driver(_driver())
    probe_driver._resolved_command = ("node.exe", "bin.js")
    with pytest.raises(LocalAgentPolicyViolationError, match="inline limit"):
        probe_driver._argv(
            _capsule(tmp_path, "x" * (DSH_INLINE_PROBE_MAX_CHARS + 1)),
            native_search=False,
            probe=DriverProbe(available=True),
            mcp_server=None,
        )


def test_resuming_a_dsh_session_is_refused_rather_than_faked(tmp_path) -> None:
    """`headless` starts a fresh session every time; there is nothing to resume."""

    with pytest.raises(LocalAgentPolicyViolationError, match="no\n?\\s*conversation"):
        _driver()._argv(
            _capsule(tmp_path),
            native_search=False,
            probe=DriverProbe(available=True),
            mcp_server=_server(),
            conversation_handle="whatever",
        )


def test_the_prompt_rides_argv_and_the_rest_rides_one_overlay(tmp_path) -> None:
    """Only the bootstrap is on the command line; the task is fetched over MCP."""

    driver = _driver()
    argv = driver._argv(
        _capsule(tmp_path, "BOOTSTRAP TEXT"),
        native_search=False,
        probe=DriverProbe(available=True),
        mcp_server=_server(),
    )

    assert argv[:2] == ["node.exe", "bin.js"]
    assert argv[argv.index("--profile") + 1] == "headless"
    assert argv[-1] == "BOOTSTRAP TEXT"
    patch = Path(argv[argv.index("--patch") + 1])
    # Inside the capsule, so it is evidence and it is thrown away with it.
    assert patch.parent == tmp_path / "input"


def test_the_overlay_adds_the_harness_server_rather_than_overriding_one(
    tmp_path,
) -> None:
    """`insert:` is the only patch form that can add a plugin.

    A bare entry is an id-targeted override, and the headless profile has no
    mcp-client to target -- the patch engine warns "entry not found" and skips
    it, which would leave the model toolless and the harness none the wiser.
    """

    entries = _entries(_driver(), tmp_path)
    inserted = next(entry for entry in entries if "insert" in entry)["insert"]

    assert len(inserted) == 1
    config = inserted[0]["config"]
    assert inserted[0]["name"] == "@deepseek-ai/dsh-mcp-client"
    assert config["serverName"] == MCP_SERVER_NAME
    assert config["transport"] == "stdio"
    assert config["env"]["FINESUB_MCP_SESSION"] == "s1"
    # Without this a server that failed to spawn yields a confident answer
    # made of nothing.
    assert config["failOnStartupError"] is True


def test_the_overlay_names_the_route_the_catalog_chose(tmp_path) -> None:
    """dsh picks a model from plugin config, not a flag.

    So the catalog's `api_model_id` carries both halves. Left unsaid, the call
    would run on whatever the profile defaults to and the catalog row would
    mean nothing.
    """

    entries = _entries(_driver(), tmp_path)
    assert _by_id(entries, "agent-default-model")["config"] == {
        "provider": "deepseek-official",
        "model": "deepseek-flash",
    }


def test_a_catalog_row_that_does_not_name_a_provider_is_refused(tmp_path) -> None:
    driver = DshLocalAgentDriver(DshDriverConfig(model="deepseek-v4-flash"))
    driver._resolved_command = ("node.exe", "bin.js")
    with pytest.raises(LocalAgentPolicyViolationError, match="<provider>/<model>"):
        _entries(driver, tmp_path)


def test_the_tool_result_cap_is_removed_by_omission_not_by_a_big_number(
    tmp_path,
) -> None:
    """`maxInlineBytes` present at all is a cap, and its spill is a file path.

    The harness wants the text, so the way to switch the policy off is to
    leave the field out, which the overlay always does.
    """

    entries = _entries(_driver(), tmp_path)
    assert _by_id(entries, "spill-policy")["config"] == {}


def test_the_web_tool_is_left_on_only_for_a_call_entitled_to_search(
    tmp_path,
) -> None:
    """dsh has no flag that narrows a tool set; a patch disables the plugin."""

    disabled = {
        entry["id"] for entry in _entries(_driver(), tmp_path) if entry.get("disabled")
    }
    assert "tool-web" in disabled
    assert "tool-bash" in disabled and "tool-subagent" in disabled
    # dsh 0.1.5 moved fetching into its own backend plugin; it follows the tool.
    assert "web-fetch-http" in disabled

    entitled = {
        entry["id"]
        for entry in _entries(_driver(), tmp_path, native_search=True)
        if entry.get("disabled")
    }
    assert "tool-web" not in entitled
    # Denying only the backend would leave `web_fetch` offered and failing:
    # `dsh-tool-web` registers it by product switch, not backend presence.
    assert "web-fetch-http" not in entitled
    assert "tool-bash" in entitled


def test_the_thinking_level_rides_the_model_selection_not_the_transport(
    tmp_path,
) -> None:
    """The level is the selection's explicit choice, beside provider and model.

    A patch replaces an entry's whole config, so setting it on `llm-deepseek`
    (the transport's default) wiped whatever the user configured there --
    endpoint, key name. The selection entry is the driver's own anyway.
    """

    entries = _entries(_driver(), tmp_path, reasoning_effort="high")
    assert _by_id(entries, "agent-default-model")["config"] == {
        "provider": "deepseek-official",
        "model": "deepseek-flash",
        "reasoningEffort": "high",
    }
    assert not any(entry.get("id") == "llm-deepseek" for entry in entries)

    plain = _entries(_driver(), tmp_path)
    assert "reasoningEffort" not in _by_id(plain, "agent-default-model")["config"]


def test_the_overlay_is_json_so_a_windows_path_cannot_change_meaning(
    tmp_path,
) -> None:
    """Hand-built YAML plus backslashes is a parse that succeeds and lies."""

    server = _server()
    server["command"] = r"C:\Users\x\ci-venv\Scripts\python.exe"
    entries = _entries(_driver(), tmp_path, mcp_server=server)
    inserted = next(entry for entry in entries if "insert" in entry)["insert"]
    assert inserted[0]["config"]["command"] == r"C:\Users\x\ci-venv\Scripts\python.exe"


def test_stdout_is_the_answer_and_there_is_no_usage_to_book(tmp_path) -> None:
    """Empty usage is what this CLI offers, not a parse waiting to be fixed."""

    raw = tmp_path / "raw.txt"
    raw.write_text("  the final message  \n", encoding="utf-8")
    normalized, usage, violations, content = _driver()._normalize(
        raw, native_search=False, max_bytes=0
    )
    assert (normalized, usage, violations) == ([], {}, [])
    assert content == "the final message"


def test_an_oversized_answer_is_a_policy_failure_not_a_silent_truncation(
    tmp_path,
) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(LocalAgentPolicyViolationError, match="result cap"):
        _driver()._normalize(raw, native_search=False, max_bytes=10)


def test_an_unreadable_stream_is_transient_not_a_protocol_violation(
    tmp_path,
) -> None:
    with pytest.raises(LocalAgentTransientError):
        _driver()._normalize(
            tmp_path / "absent.txt", native_search=False, max_bytes=0
        )


def test_the_driver_promises_only_what_headless_can_deliver() -> None:
    """The narrowed requirements are the honest ones.

    Requiring `structured_events` would make dsh permanently unavailable
    rather than usable-with-no-usage-numbers, which is the actual trade.
    """

    driver = _driver()
    assert "structured_events" not in driver.required_capabilities(native_search=False)
    assert "no_persisted_session" not in driver.required_capabilities(
        native_search=False
    )
    assert driver.required_capabilities(native_search=True) == (
        "can_restrict_tools",
        "has_web_search",
    )
    # No conversation is worth keeping, so nothing tries to resume one.
    assert driver.config.conversation_ttl_seconds == 0.0


def test_a_dsh_row_a_person_wrote_routes_without_a_packaged_target(tmp_path) -> None:
    """dsh is a harness, so its reachable models are the owner's, not ours.

    Every other agent CLI *is* its account -- the vendor decides the model
    list, so the packaged catalog can carry it. dsh points at whatever the
    owner declared in `$DSH_HOME/settings.yaml`, which nobody knows when this
    package is built. A local-agent row used to be dropped on the floor here:
    the fact loaded, no target was built, and nothing said so.
    """

    from finesub.llm.routing.model_catalog import load_effective_catalog
    from finesub.llm.routing.model_routes import load_model_routes

    catalog = tmp_path / "model_catalog.toml"
    catalog.write_text(
        "# a person's own file, comments and all\n"
        "\n"
        "[models.mine-plain]\n"
        'provider_tier = "LOCAL_DSH"\n'
        'api_model_id = "my-gateway/some-model"\n'
        "max_input_tokens = 194000\n"
        "supports_native_search = false\n",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={},
        catalog=load_effective_catalog(catalog),
    )

    target = routes.targets["mine-plain"]
    assert target.backend == "local_agent"
    assert target.enabled_by == "local_agent"
    # The tool is a driver property (it comes off the tier); the row only
    # says whether the model may use it.
    assert target.native_search_tool == ""


def test_a_row_that_claims_search_gets_the_target_that_entitles_it(tmp_path) -> None:
    from finesub.llm.routing.model_catalog import load_effective_catalog
    from finesub.llm.routing.model_routes import load_model_routes

    catalog = tmp_path / "model_catalog.toml"
    catalog.write_text(
        """
[models.mine-search]
provider_tier = "LOCAL_DSH"
api_model_id = "my-gateway/some-model"
max_input_tokens = 194000
supports_native_search = true
""",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={},
        catalog=load_effective_catalog(catalog),
    )
    assert routes.targets["mine-search"].native_search_tool == "web_search"


def test_every_agent_tier_wires_a_model_a_person_adds(tmp_path) -> None:
    """A new Codex release is one table in the override catalog (owner
    2026-11-09), the same as a dsh gateway."""

    from finesub.llm.routing.model_catalog import load_effective_catalog
    from finesub.llm.routing.model_routes import load_model_routes

    catalog = tmp_path / "model_catalog.toml"
    catalog.write_text(
        """
[models.experiment]
provider_tier = "LOCAL_CODEX"
api_model_id = "gpt-experimental"
max_input_tokens = 194000
""",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={},
        catalog=load_effective_catalog(catalog),
    )
    assert routes.targets["experiment"].fact_id == "experiment"
    assert routes.targets["experiment"].native_search_tool == "web_search"


def test_a_declared_route_gets_the_rows_level_as_the_row_maps_it(
    tmp_path,
) -> None:
    """A user's gateway takes the catalog row's level like any other model.

    `thinking = true` sends high / medium / low as they are, a three-word list
    renames them, `false` reaches the driver as no level at all. `llm-pi-ai`
    carries off/minimal/low/medium/high/xhigh/max, so a word the DeepSeek
    adapter lacks is not this driver's to refuse there; only the model's own
    declaration decides, and `run` handles a refusal. The user's provider
    entries are never part of the overlay.
    """

    packaged = _entries(_driver(), tmp_path, reasoning_effort="high")
    assert _effort(packaged) == "high"

    mine = DshLocalAgentDriver(DshDriverConfig(model="my-gateway/some-model"))
    mine._resolved_command = ("node.exe", "bin.js")
    for level in ("high", "medium", "minimal"):
        entries = _entries(mine, tmp_path, reasoning_effort=level)
        assert _effort(entries) == level
        assert not any(entry.get("id") in {"llm-pi-ai", "llm-deepseek"} for entry in entries)
    assert _effort(_entries(mine, tmp_path, reasoning_effort="")) is None


def test_the_packaged_rows_map_thinking_to_words_deepseek_accepts() -> None:
    """Identity would send `medium`, which this adapter's union rejects.

    `llm-pi-ai` does carry low/medium/high, so a user gateway can keep the
    identity default; `@deepseek-ai/dsh-llm-deepseek` offers off/low/high/max
    (re-read off the installed 0.1.1-rc.2 bundle, 2026-09-04) and simply has
    no middle word.
    """

    from finesub.llm.routing.model_catalog import default_model_catalog

    rows = {
        entry.fact_id: entry
        for entry in default_model_catalog()
        if entry.provider_tier == "LOCAL_DSH"
    }
    assert rows, "the packaged catalog still ships the dsh routes"
    for fact_id, entry in rows.items():
        # high, medium, low -- in that order, and every word one dsh accepts.
        # The middle cell is `low`, not `high` (owner, 2026-09-04). It is the
        # one that matters: `[presets.default.thinking]` leaves
        # `correction-*/quality` unset, so it falls to
        # `DEFAULT_THINKING_LEVEL = "medium"` and the correction window sends
        # whatever sits here. The same model family on CodeBuddy was measured
        # thinking without converging at `high` on a real window -- 28 minutes
        # to the hard timeout, clean in one call at `low`
        # (docs/llm_local_agent.md §12.1.5). The top cell stays `high` because
        # `max` is a step beyond what the abstract top asks for, and research /
        # knowledge quality (abstract `high`) still reaches it.
        assert entry.thinking_levels == ("high", "low", "low"), fact_id
        assert "medium" not in entry.thinking_levels, fact_id


def test_the_global_effort_setting_is_translated_into_dshs_vocabulary(
    tmp_path,
) -> None:
    """`medium` and `xhigh` are legal levels that dsh's adapter has no word for.

    `[llm].local_agent_reasoning_effort` is validated against the other
    CLIs' shared vocabulary and reaches the call as its level, so a person may
    legitimately configure either one. Sent through, the DeepSeek adapter
    answers `UNSUPPORTED_REASONING_EFFORT` before any network I/O.
    """

    for level, expected in (("medium", "high"), ("xhigh", "max")):
        entries = _entries(_driver(), tmp_path, reasoning_effort=level)
        assert _effort(entries) == expected, level


def test_an_effort_word_dsh_cannot_take_fails_as_a_policy_violation(
    tmp_path,
) -> None:
    """Permanent and attributable beats a transient exit the freeze counts."""

    with pytest.raises(LocalAgentPolicyViolationError, match="reasoning effort"):
        _entries(_driver(), tmp_path, reasoning_effort="turbo")


def test_the_isolation_record_names_the_word_the_call_actually_ran_on(
    tmp_path,
) -> None:
    """An audit line reading `medium` would name a level no dsh call ever had."""

    del tmp_path
    metadata = _driver()._isolation_metadata(DriverProbe(available=True), "medium")
    assert metadata["reasoning_effort"] == "high"


def test_dsh_reads_its_tool_events_out_of_the_session_transcript(tmp_path) -> None:
    """Headless prints only the answer -- but it also writes a transcript.

    Until 2026-08-30 this driver declared `observes_tool_events = False`,
    because stdout carries nothing but the final message. The persistence
    plugin was writing every tool call the whole time; `_patch_entries` now
    redirects that log into the capsule uncompressed, so the flag would be a
    lie and the search events are real.
    """

    from finesub.llm.agent.local_agent import (
        DSH_SESSION_DIRNAME,
        LocalAgentDriver,
        _dsh_session_rows,
    )

    assert LocalAgentDriver.observes_tool_events is True
    assert DshLocalAgentDriver.observes_tool_events is True

    root = tmp_path / "events" / DSH_SESSION_DIRNAME
    log = root / "--some-cwd--" / "session-abc" / "session.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "tool/call", "data": {
                    "callId": "c1", "name": "mcp__finesub__next_task",
                    "arguments": "{}"}}),
                json.dumps({"type": "tool/call", "data": {
                    "callId": "c2", "name": "web_search",
                    "arguments": json.dumps({"queries": ["who won", "when"]})}}),
                json.dumps({"type": "tool/result", "data": {
                    "message": {"source": {"kind": "tool", "callId": "c2"},
                                "content": [{"type": "tool-result", "content": [
                                    {"type": "text",
                                     "text": "see https://example.com/a"}]}]}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _dsh_session_rows(root)
    searches = [r for r in rows if r.get("item_type") == "web_search"]
    assert len(searches) == 1
    # Both queries survive: the tool takes a list and dropping the tail would
    # under-report what the call actually asked.
    assert searches[0]["query"] == "who won | when"
    assert searches[0]["urls"] == ["https://example.com/a"]
    # The MCP call is still recorded, just not as a search.
    assert [r["tool"] for r in rows] == ["mcp__finesub__next_task", "web_search"]
    # A capsule whose plugin never wrote a log is thinner evidence, not an
    # error: the answer already succeeded.
    assert _dsh_session_rows(tmp_path / "nope") == []


def test_dsh_0_1_5_renamed_its_transcript_and_the_audit_still_reads_it(
    tmp_path,
) -> None:
    """dsh 0.1.5 writes `session.v3.jsonl`; reading only `session.jsonl` saw nothing.

    Found 2026-09-12 on a synthetic native window: the model searched twice and
    fetched twice, and the attempt said `native_search_not_used` -- every tool
    event of every dsh call had dropped out of the audit, with the suite green.
    The event format itself did not change. `web_fetch` is a retrieval row too,
    the same shape Claude Code's `WebFetch` and agy's fetch tool produce, so the
    fetched URL reaches provenance instead of vanishing.
    """

    from finesub.llm.agent.local_agent import DSH_SESSION_DIRNAME, _dsh_session_rows

    root = tmp_path / "events" / DSH_SESSION_DIRNAME
    log = root / "--some-cwd--" / "session-v3" / "session.v3.jsonl"
    log.parent.mkdir(parents=True)
    # Not a transcript, and must not be read as one.
    (log.parent / "session.meta.jsonl").write_text(
        json.dumps({"type": "tool/call", "data": {"callId": "x", "name": "web_search"}})
        + "\n",
        encoding="utf-8",
    )
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "tool/call", "data": {
                    "callId": "s1", "name": "web_search",
                    "arguments": json.dumps({"queries": ["原神 小吉祥草王"]})}}),
                json.dumps({"type": "tool/result", "data": {
                    "message": {"source": {"kind": "tool", "callId": "s1"},
                                "content": [{"type": "tool-result", "content": [
                                    {"type": "text",
                                     "text": "Sources:\n- https://wiki.example/nahida"}]}]}}}),
                json.dumps({"type": "tool/call", "data": {
                    "callId": "f1", "name": "web_fetch",
                    "arguments": json.dumps({"url": "http://dict.example/kusanali"})}}),
                # Redirected to https, and the page body links elsewhere -- as
                # the real dictionary page's footer did (2026-09-12).
                json.dumps({"type": "tool/result", "data": {
                    "message": {"source": {"kind": "tool", "callId": "f1"},
                                "content": [{"type": "tool-result", "content": [
                                    {"type": "text",
                                     "text": "Fetched https://dict.example/kusanali (HTTP 200)\n\n"
                                             "External web content follows.\n\n"
                                             "# Kusanali\n\nMade by https://github.com/someone"}]}]}}}),
                json.dumps({"type": "tool/call", "data": {
                    "callId": "m1", "name": "mcp__finesub__submit",
                    "arguments": "{}"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _dsh_session_rows(root)
    assert [row["tool"] for row in rows] == [
        "web_search",
        "web_fetch",
        "mcp__finesub__submit",
    ]
    retrieval = [row for row in rows if row.get("item_type") == "web_search"]
    assert [row["tool"] for row in retrieval] == ["web_search", "web_fetch"]
    assert retrieval[0]["urls"] == ["https://wiki.example/nahida"]
    # The page asked for and the page read after the redirect -- and not the
    # footer link in its body, which a search result would have counted.
    assert retrieval[1]["urls"] == [
        "http://dict.example/kusanali",
        "https://dict.example/kusanali",
    ]


def test_dsh_gets_the_deepseek_key_from_the_environment_on_every_call(monkeypatch) -> None:
    """The allowlist strips the driver's own provider key.

    dsh 0.2.2's headless profile switches off its credential store, so the
    model route reads the key from the launch environment (or dsh's `.env`
    files): without it every call failed `MISSING_CREDENTIAL` (2026-11-15),
    and before 2026-08-30 the search tool failed the same way. It is the
    account's own key, not the harness's, so it rides only this driver.
    """

    from finesub.llm.agent.local_agent import DSH_SEARCH_KEY_ENV

    driver = DshLocalAgentDriver(DshDriverConfig(command=("dsh",)))
    monkeypatch.setenv(DSH_SEARCH_KEY_ENV, "sk-not-a-real-key")

    for native in (False, True):
        env = driver._spawn_environment(native_search=native)
        assert env[DSH_SEARCH_KEY_ENV] == "sk-not-a-real-key"

    # Nothing invented when the owner has not set one: dsh's `.env` files
    # are then where it looks.
    monkeypatch.delenv(DSH_SEARCH_KEY_ENV, raising=False)
    assert DSH_SEARCH_KEY_ENV not in driver._spawn_environment(native_search=True)


def test_dsh_carries_its_own_execution_identity() -> None:
    """Every wired local-agent tier has to be in the resume identity.

    The table goes into `execution_identity` verbatim, so a tier missing from
    it contributes nothing: its protocol, tool set or sandbox could change
    and an uncommitted checkpoint taken under the old contract would still
    look reusable.
    """

    from finesub.llm.agent.local_agent import local_agent_execution_profiles
    from finesub.llm.routing.model_routes import BACKEND_PROVIDER_TIERS

    profiles = local_agent_execution_profiles()
    assert set(BACKEND_PROVIDER_TIERS["local_agent"]) <= set(profiles)
    dsh = profiles["LOCAL_DSH"]
    assert dsh["driver_id"] == "dsh"
    assert dsh["configuration"]["user_configuration"] == "inherited"
    # Read off the config rather than copied, so the two cannot drift.
    assert dsh["configuration"]["disabled_tool_plugins"] == sorted(
        DshDriverConfig().disabled_tool_plugins
    )
    assert dsh["configuration_digest"]


def test_changing_what_dsh_switches_off_moves_the_execution_identity(
    monkeypatch,
) -> None:
    """The point of the digest: a weaker sandbox invalidates old work."""

    from finesub.llm.agent import local_agent as module

    before = module.local_agent_execution_profiles()["LOCAL_DSH"]
    monkeypatch.setattr(
        module,
        "_DSH_IDENTITY",
        DshDriverConfig(disabled_tool_plugins=("tool-bash",)),
    )
    after = module.local_agent_execution_profiles()["LOCAL_DSH"]
    assert before["configuration_digest"] != after["configuration_digest"]


def test_repointing_an_effort_alias_moves_the_execution_identity(
    monkeypatch,
) -> None:
    """The alias table is a mapping layer, and nothing else covers it.

    A row's own thinking column rides `routing_identity_digest`. What the
    driver does to the *global* override is code: point `medium` at `max` and
    every call thinks harder while the identity -- and so an uncommitted
    checkpoint -- says nothing changed.
    """

    from finesub.llm.agent import local_agent as module

    before = module.local_agent_execution_profiles()["LOCAL_DSH"]
    monkeypatch.setattr(
        module, "DSH_EFFORT_ALIASES", {"medium": "max", "xhigh": "max"}
    )
    after = module.local_agent_execution_profiles()["LOCAL_DSH"]
    assert before["configuration_digest"] != after["configuration_digest"]


def test_a_custom_route_records_the_level_it_sent_or_the_owners(
    tmp_path,
) -> None:
    """A level sent is recorded as sent; with none sent (`thinking = false`)
    the level that call ran on sits in the owner's own profile patch, and the
    audit says so rather than leaving the field empty."""

    del tmp_path
    driver = DshLocalAgentDriver(DshDriverConfig(model="my-gateway/some-model"))
    probe = DriverProbe(available=True)
    assert driver._isolation_metadata(probe, "low")["reasoning_effort"] == "low"
    metadata = driver._isolation_metadata(probe, "")
    assert metadata["reasoning_effort"] == "owner_managed"
    assert "reasoning_effort_refused" not in metadata


# --- a level the declared model does not take --------------------------------


def _starts(monkeypatch, *, refuse: bool, recorded_meanwhile: bool = False,
            rerun_fails: bool = False) -> list[str]:
    """Stand in for one CLI start each, recording the level each one sent.

    With ``refuse`` dsh turns down every level before any request, the way it
    turns down one a declared model does not advertise; each failure carries
    its start's attempt record, as the real one does. ``recorded_meanwhile``
    has a parallel call record the refusal before this one sees it.
    """

    from finesub.llm.agent import local_agent as module
    from finesub.llm.agent.local_agent import AgentExecutionResult

    sent: list[str] = []

    def fake_run(self, messages, **kwargs):
        del messages
        level = str(kwargs.get("reasoning_effort") or "")
        sent.append(self._effort_sent(level))
        attempt = {
            "capsule_id": f"start-{len(sent)}",
            "isolation": self._isolation_metadata(DriverProbe(available=True), level),
        }
        if sent[-1] and refuse:
            if recorded_meanwhile:
                module._DSH_UNSUPPORTED_EFFORTS.add((self.config.model, sent[-1]))
            error = module.DshUnsupportedEffortError("refused before any request")
            error._harness_execution_attempts = [attempt]
            raise error
        if rerun_fails:
            error = LocalAgentTransientError("down")
            error._harness_execution_attempts = [attempt]
            raise error
        return AgentExecutionResult(
            content="OK", reported_model="", episode_id=attempt["capsule_id"],
            execution_attempt=attempt,
        )

    monkeypatch.setattr(module.LocalAgentDriver, "run", fake_run)
    monkeypatch.setattr(module, "_DSH_UNSUPPORTED_EFFORTS", set())
    return sent


def _gateway() -> DshLocalAgentDriver:
    return DshLocalAgentDriver(DshDriverConfig(model="my-gateway/some-model"))


def test_a_level_the_declared_model_does_not_take_reruns_without_it(
    monkeypatch, reported
) -> None:
    """A model declared under `llm-pi-ai` without `reasoningEfforts` takes no
    level at all (measured 2026-11-15), so the default mapping would fail
    every call. The call reruns once without the level -- the user's own
    provider default -- and later calls on the model skip the level, so
    only the first pays the second start-up and the warning comes once.
    The refused start spent a start-up too, so it stays on the record."""

    sent = _starts(monkeypatch, refuse=True)
    driver = _gateway()

    first = driver.run([], task="correction", reasoning_effort="medium")
    assert sent == ["medium", ""]
    assert [a["capsule_id"] for a in first.prior_attempts] == ["start-1"]
    isolation = first.execution_attempt["isolation"]
    assert isolation["reasoning_effort"] == "owner_managed"
    assert isolation["reasoning_effort_refused"] == "medium"

    later = driver.run([], task="correction", reasoning_effort="medium")
    assert sent == ["medium", "", ""]
    assert later.prior_attempts == ()
    assert [w.code for w in reported.warnings] == ["agent-dsh-effort-unsupported"]

    # Another level is its own question: it is tried, not assumed refused.
    driver.run([], task="correction", reasoning_effort="high")
    assert sent[3:] == ["high", ""]


def test_a_refusal_another_call_recorded_first_still_reruns(monkeypatch, reported) -> None:
    """Two calls start with the same level and both are refused; whichever
    records the refusal second must still rerun rather than read the shared
    record as "nothing was sent" and fail for good."""

    sent = _starts(monkeypatch, refuse=True, recorded_meanwhile=True)
    result = _gateway().run([], task="correction", reasoning_effort="medium")
    assert sent == ["medium", ""]
    assert result.content == "OK"
    # The call that recorded it first is the one that warned.
    assert reported.warnings == []


def test_a_rerun_that_fails_carries_both_starts(monkeypatch) -> None:
    sent = _starts(monkeypatch, refuse=True, rerun_fails=True)
    with pytest.raises(LocalAgentTransientError) as raised:
        _gateway().run([], task="correction", reasoning_effort="medium")
    assert sent == ["medium", ""]
    assert [a["capsule_id"] for a in raised.value._harness_execution_attempts] == [
        "start-1", "start-2",
    ]


def test_the_packaged_route_does_not_retry_a_refused_level(monkeypatch) -> None:
    """The packaged adapter's words are translated before the call, so a
    refusal there is a real fault to report, not a declaration to work
    around."""

    from finesub.llm.agent.local_agent import DshUnsupportedEffortError

    sent = _starts(monkeypatch, refuse=True)
    with pytest.raises(DshUnsupportedEffortError):
        _driver().run([], task="correction", reasoning_effort="medium")
    assert sent == ["high"]


def test_a_refused_level_on_stderr_is_permanent_and_any_other_exit_transient(
    tmp_path,
) -> None:
    """dsh's own line (0.2.2-rc.1). Transient would let two refusals freeze
    the tier's allowance over a setting that fails the same way every time."""

    from finesub.llm.agent.local_agent import DshUnsupportedEffortError

    stderr = tmp_path / "stderr.log"
    capsule = SimpleNamespace(stderr_path=stderr, episode_id="ep")
    stderr.write_text(
        "dsh: warning: 3 entries did not activate\n"
        'dsh: UNSUPPORTED_REASONING_EFFORT: provider "probe-ds" model '
        '"deepseek-flash" does not support reasoning effort "medium"\n',
        encoding="utf-8",
    )
    refused = _driver()._nonzero_exit(capsule, 1)
    assert isinstance(refused, DshUnsupportedEffortError)
    assert isinstance(refused, LocalAgentPolicyViolationError)
    assert 'reasoning effort "medium"' in str(refused)

    stderr.write_text("dsh: MISSING_CREDENTIAL\n", encoding="utf-8")
    assert isinstance(_driver()._nonzero_exit(capsule, 1), LocalAgentTransientError)
