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
    DriverProbe,
    DshDriverConfig,
    DshLocalAgentDriver,
    LocalAgentPolicyViolationError,
    LocalAgentTransientError,
    MCP_SERVER_NAME,
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
        DshDriverConfig(model="deepseek-official/deepseek-v4-flash", **overrides)
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
        "model": "deepseek-v4-flash",
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
    leave the field out -- which is what a page size of 0 asks for here.
    """

    entries = _entries(_driver(), tmp_path)
    assert _by_id(entries, "spill-policy")["config"] == {}

    entries = _entries(_driver(mcp_page_chars=2800), tmp_path)
    assert _by_id(entries, "spill-policy")["config"] == {"maxInlineBytes": 2800}


def test_the_web_tool_is_left_on_only_for_a_call_entitled_to_search(
    tmp_path,
) -> None:
    """dsh has no flag that narrows a tool set; a patch disables the plugin."""

    disabled = {
        entry["id"] for entry in _entries(_driver(), tmp_path) if entry.get("disabled")
    }
    assert "tool-web" in disabled
    assert "tool-bash" in disabled and "tool-subagent" in disabled

    entitled = {
        entry["id"]
        for entry in _entries(_driver(), tmp_path, native_search=True)
        if entry.get("disabled")
    }
    assert "tool-web" not in entitled
    assert "tool-bash" in entitled


def test_the_thinking_level_rides_the_overlay_because_there_is_no_flag(
    tmp_path,
) -> None:
    entries = _entries(_driver(), tmp_path, reasoning_effort="high")
    assert _by_id(entries, "llm-deepseek")["config"] == {"reasoningEffort": "high"}

    plain = _entries(_driver(), tmp_path)
    assert not any(entry.get("id") == "llm-deepseek" for entry in plain)


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

    from finesub.llm.routing.model_catalog import (
        default_model_catalog,
        load_model_catalog,
        merge_catalogs,
    )
    from finesub.llm.routing.model_routes import load_model_routes

    catalog = tmp_path / "model_catalog.psv"
    catalog.write_text(
        "# a person's own file, comments and all\n"
        "\n"
        "fact_id|provider_tier|api_model_id|max_input_tokens|supports_native_search\n"
        "mine-plain|LOCAL_DSH|my-gateway/some-model|194000|false\n",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={},
        catalog=merge_catalogs(
            default_model_catalog(), load_model_catalog(catalog, self_reported=True)
        ),
    )

    target = routes.targets["mine-plain"]
    assert target.backend == "local_agent"
    assert target.enabled_by == "local_agent"
    # The profile is a driver property, not a model one, so it comes off the
    # tier; the row only chooses between the tier's two.
    assert target.execution_profile == "dsh-default"
    assert routes.execution_profiles[target.execution_profile].native_search_tool == ""


def test_a_row_that_claims_search_gets_the_target_that_entitles_it(tmp_path) -> None:
    from finesub.llm.routing.model_catalog import (
        default_model_catalog,
        load_model_catalog,
        merge_catalogs,
    )
    from finesub.llm.routing.model_routes import load_model_routes

    catalog = tmp_path / "model_catalog.psv"
    catalog.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|supports_native_search\n"
        "mine-search|LOCAL_DSH|my-gateway/some-model|194000|true\n",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={},
        catalog=merge_catalogs(
            default_model_catalog(), load_model_catalog(catalog, self_reported=True)
        ),
    )
    target = routes.targets["mine-search"]
    assert target.execution_profile == "dsh-web-search"
    assert (
        routes.execution_profiles[target.execution_profile].native_search_tool
        == "web_search"
    )


def test_the_other_agent_tiers_are_still_not_auto_wired(tmp_path) -> None:
    """Only dsh opted in.

    A codex or agy row with no packaged target is a fact recorded for an
    experiment, and wiring it up silently would be the opposite of what the
    catalog promises there.
    """

    from finesub.llm.routing.model_catalog import (
        default_model_catalog,
        load_model_catalog,
        merge_catalogs,
    )
    from finesub.llm.routing.model_routes import load_model_routes

    catalog = tmp_path / "model_catalog.psv"
    catalog.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens\n"
        "experiment|LOCAL_CODEX|gpt-experimental|194000\n",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={},
        catalog=merge_catalogs(
            default_model_catalog(), load_model_catalog(catalog, self_reported=True)
        ),
    )
    assert "experiment" in routes.facts
    assert "experiment" not in routes.targets


def test_the_thinking_level_only_reaches_the_route_this_package_ships(
    tmp_path,
) -> None:
    """A user's own gateway keeps its own knob.

    `llm-pi-ai` carries the level on the provider entry, inside its `providers`
    dict, and a patch override assigns whole keys -- writing there would
    replace that dict and take the endpoint and credential of the provider
    being used with it. So nothing is written, and the catalog row says
    `thinking = false` rather than pretending the harness is steering it.
    """

    packaged = _entries(_driver(), tmp_path, reasoning_effort="high")
    assert _by_id(packaged, "llm-deepseek")["config"] == {"reasoningEffort": "high"}

    mine = DshLocalAgentDriver(DshDriverConfig(model="my-gateway/some-model"))
    mine._resolved_command = ("node.exe", "bin.js")
    entries = _entries(mine, tmp_path, reasoning_effort="high")
    assert not any(entry.get("id") == "llm-pi-ai" for entry in entries)
    assert not any(entry.get("id") == "llm-deepseek" for entry in entries)


def test_the_packaged_rows_map_thinking_to_words_deepseek_accepts() -> None:
    """Identity would send `medium`, which this adapter's union rejects.

    `llm-pi-ai` does carry low/medium/high, so a user gateway can keep the
    identity default; `@deepseek-ai/dsh-llm-deepseek` offers off/low/high/max
    and simply has no middle word.
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
        # high and medium land on the same word on purpose (owner, 2026-08-25):
        # the adapter has no middle level, and `max` is a step beyond what the
        # abstract top asks for, so the two upper cells collapse rather than
        # one of them being inflated.
        assert entry.thinking_levels == ("high", "high", "low"), fact_id
        assert "medium" not in entry.thinking_levels, fact_id


def test_a_configured_effort_overrides_the_cells_mapped_level(tmp_path) -> None:
    """`local_agent_reasoning_effort` is a deliberate global override.

    Same precedence as Claude Code and agy. The other order looks harmless
    until you notice a packaged row always maps a level, so the setting would
    never once win.
    """

    driver = _driver(effort="max")
    entries = _entries(driver, tmp_path, reasoning_effort="low")
    assert _by_id(entries, "llm-deepseek")["config"] == {"reasoningEffort": "max"}


def test_the_global_effort_setting_is_translated_into_dshs_vocabulary(
    tmp_path,
) -> None:
    """`medium` and `xhigh` are legal settings that dsh has no word for.

    `[llm].local_agent_reasoning_effort` is validated against the other three
    CLIs' shared vocabulary, so a person may legitimately configure either
    one. Sent through, the DeepSeek adapter answers `UNSUPPORTED_REASONING_
    EFFORT` before any network I/O -- a non-zero exit, which classifies
    transient, and two of those freeze the tier's allowance over a spelling.
    """

    for configured, expected in (("medium", "high"), ("xhigh", "max")):
        entries = _entries(_driver(effort=configured), tmp_path)
        assert _by_id(entries, "llm-deepseek")["config"] == {
            "reasoningEffort": expected
        }, configured


def test_an_effort_word_dsh_cannot_take_fails_as_a_policy_violation(
    tmp_path,
) -> None:
    """Permanent and attributable beats a transient exit the freeze counts."""

    with pytest.raises(LocalAgentPolicyViolationError, match="reasoning effort"):
        _entries(_driver(effort="turbo"), tmp_path)


def test_the_isolation_record_names_the_word_the_call_actually_ran_on(
    tmp_path,
) -> None:
    """An audit line reading `medium` would name a level no dsh call ever had."""

    del tmp_path
    driver = _driver(effort="medium")
    metadata = driver._isolation_metadata(DriverProbe(available=True), "low")
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


def test_dsh_hands_its_search_backend_the_key_only_when_entitled(monkeypatch) -> None:
    """The allowlist strips the driver's own provider key.

    Measured 2026-08-30: `--retrieval native` gave the model a `web_search`
    tool that failed every call with "no API key for DEEPSEEK_API_KEY",
    because the key lives in the user's environment and the sanitizer drops
    everything unlisted. It is the account's own key, not the harness's, so
    it rides only this driver and only an entitled call.
    """

    from finesub.llm.agent.local_agent import DSH_SEARCH_KEY_ENV

    driver = DshLocalAgentDriver(DshDriverConfig(command=("dsh",)))
    monkeypatch.setenv(DSH_SEARCH_KEY_ENV, "sk-not-a-real-key")

    assert DSH_SEARCH_KEY_ENV not in driver._spawn_environment()
    assert driver._spawn_environment(native_search=True)[DSH_SEARCH_KEY_ENV] == (
        "sk-not-a-real-key"
    )

    # Nothing invented when the owner has not set one.
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


def test_a_custom_route_records_the_level_as_the_owners_not_as_ours(
    tmp_path,
) -> None:
    """The driver sends no level there, so it must not claim one.

    `llm-pi-ai` keeps its thinking knob inside `providers.<id>`, and a patch
    override assigns whole keys, so writing there would take the endpoint and
    credential reference down with it. Nothing is sent -- and an audit line
    reading `high` would name a level this call never carried, while the one
    it did run on sits in the owner's own `settings.yaml`.
    """

    driver = DshLocalAgentDriver(
        DshDriverConfig(model="my-gateway/some-model", effort="medium")
    )
    driver._resolved_command = ("node.exe", "bin.js")

    entries = _entries(driver, tmp_path)
    assert not any(entry.get("id") == "llm-deepseek" for entry in entries)
    metadata = driver._isolation_metadata(DriverProbe(available=True), "low")
    assert metadata["reasoning_effort"] == "owner_managed"


def test_a_custom_routes_own_thinking_words_are_not_this_drivers_to_refuse(
    tmp_path,
) -> None:
    """`llm-pi-ai` carries off/minimal/low/medium/high/xhigh/max.

    A row for a user gateway may legitimately map onto `minimal`, which the
    DeepSeek adapter has no word for. Refusing it would kill a call over a
    value this driver was never going to pass on.
    """

    driver = DshLocalAgentDriver(DshDriverConfig(model="my-gateway/some-model"))
    driver._resolved_command = ("node.exe", "bin.js")

    entries = _entries(driver, tmp_path, reasoning_effort="minimal")
    assert not any(entry.get("id") == "llm-deepseek" for entry in entries)
