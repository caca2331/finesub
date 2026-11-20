"""The Codex user's connection settings carried past `--ignore-user-config`."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import socket
import tomllib

import pytest

from finesub.llm.agent import codex_connection
from finesub.llm.agent.codex_connection import (
    PROVIDER_TOKEN_ENV,
    CodexConnection,
    CodexConnectionError,
    check_ready,
    connection_from_config,
    local_endpoint,
    read_connection,
    toml_literal,
)
from finesub.llm.agent.local_agent import (
    CODEX_DISABLED_FEATURES,
    CodexDriverConfig,
    CodexLocalAgentDriver,
    LocalAgentProviderError,
    LocalAgentUnavailableError,
)

from .support.local_agent import _driver


PROXY_CONFIG = """
model_provider = "cliproxyapi"
model = "gpt-6.2-sol"
notify = ["notifier.exe", "turn-ended"]

[mcp_servers.node_repl]
command = "node_repl.exe"

[model_providers.cliproxyapi]
name = "OpenAI"
base_url = "http://127.0.0.1:{port}/v1"
experimental_bearer_token = "sk-proxy-secret"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = true
"""


def _write_config(monkeypatch, tmp_path: Path, text: str | None) -> Path:
    """A user Codex home holding `text` as its config; None leaves it without one."""

    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    if text is not None:
        (home / "config.toml").write_text(text, encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


@pytest.fixture
def listening_port():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


@pytest.fixture
def closed_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _overrides(connection: CodexConnection) -> dict[str, object]:
    """Parse the `-c` values back, the way Codex reads them."""

    parsed: dict[str, object] = {}
    for override in connection.overrides:
        key, _, literal = override.partition("=")
        parsed[key] = tomllib.loads(f"v = {literal}")["v"]
    return parsed


# -- reading the config --------------------------------------------------------


def test_missing_config_carries_nothing(tmp_path: Path) -> None:
    assert read_connection(tmp_path / "nowhere") == CodexConnection()


def test_unparsable_config_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("model_provider = [", encoding="utf-8")
    with pytest.raises(CodexConnectionError, match="not valid TOML"):
        read_connection(tmp_path)


def test_only_connection_keys_and_the_named_provider_are_carried() -> None:
    document = tomllib.loads(PROXY_CONFIG.format(port=8317))
    document["model_providers"]["other"] = {"name": "Other", "base_url": "http://x"}

    connection = connection_from_config(document)
    carried = _overrides(connection)

    assert set(carried) == {"model_provider", "model_providers.cliproxyapi"}
    assert carried["model_provider"] == "cliproxyapi"
    assert connection.keys == ("model_provider",)
    assert connection.provider == "cliproxyapi"
    assert connection.base_url == "http://127.0.0.1:8317/v1"


def test_builtin_provider_carries_no_table() -> None:
    connection = connection_from_config(
        {"model_provider": "openai", "openai_base_url": "http://localhost:9000/v1"}
    )

    assert _overrides(connection) == {
        "model_provider": "openai",
        "openai_base_url": "http://localhost:9000/v1",
    }
    assert connection.base_url == "http://localhost:9000/v1"


def test_unset_keys_are_not_passed() -> None:
    assert connection_from_config({"model": "gpt-6"}).overrides == ()


def test_legacy_profile_selector_is_not_followed() -> None:
    """Codex 0.157.1 no longer reads `profile = "..."`; neither does FineSub."""

    document = {
        "model_provider": "openai",
        "profile": "work",
        "profiles": {"work": {"model_provider": "corp"}},
    }

    assert _overrides(connection_from_config(document)) == {"model_provider": "openai"}
    assert connection_from_config({"profile": "missing"}).overrides == ()


def test_a_provider_without_a_table_is_left_to_codex() -> None:
    """A name Codex defines itself (or adds later): carried as is, Codex judges it."""

    connection = connection_from_config({"model_provider": "ghost"})
    assert _overrides(connection) == {"model_provider": "ghost"}
    assert connection.base_url == ""


# -- secrets -----------------------------------------------------------------


def test_bearer_token_moves_to_the_environment() -> None:
    connection = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=8317)))
    table = _overrides(connection)["model_providers.cliproxyapi"]

    assert "experimental_bearer_token" not in table
    assert table["env_key"] == PROVIDER_TOKEN_ENV
    assert connection.env == {PROVIDER_TOKEN_ENV: "sk-proxy-secret"}
    assert all("sk-proxy-secret" not in override for override in connection.overrides)


def test_env_key_and_header_variables_pass_through() -> None:
    connection = connection_from_config({
        "model_provider": "corp",
        "model_providers": {"corp": {
            "base_url": "https://corp.example/v1",
            "env_key": "CORP_KEY",
            "env_http_headers": {"X-Org": "CORP_ORG"},
        }},
    })

    assert connection.passthrough_env == ("CORP_KEY", "CORP_ORG")
    assert connection.env == {}


@pytest.mark.parametrize("literal", ["http_headers", "query_params"])
def test_literal_headers_and_query_params_are_refused(literal: str) -> None:
    with pytest.raises(CodexConnectionError, match="env_http_headers"):
        connection_from_config({
            "model_provider": "corp",
            "model_providers": {"corp": {"base_url": "https://x", literal: {"k": "v"}}},
        })


def test_token_and_env_key_together_are_refused() -> None:
    with pytest.raises(CodexConnectionError, match="keep one"):
        connection_from_config({
            "model_provider": "corp",
            "model_providers": {"corp": {"experimental_bearer_token": "t", "env_key": "K"}},
        })


# -- TOML literals -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        'C:\\Users\\a "quoted" path\n\ttab\x01',
        True,
        False,
        42,
        1.5,
        ["a", 'b"c'],
        {"plain": "x", "needs quoting": 1, "nested": {"k": ["v"]}},
    ],
)
def test_literals_round_trip(value: object) -> None:
    assert tomllib.loads(f"v = {toml_literal(value)}")["v"] == value


@pytest.mark.parametrize("value", [[1, 2], float("inf"), None, object()])
def test_unsupported_values_are_refused(value: object) -> None:
    with pytest.raises(CodexConnectionError):
        toml_literal(value)


# -- pre-flight ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8317/v1", ("127.0.0.1", 8317)),
        ("http://127.5.6.7:9/v1", ("127.5.6.7", 9)),
        ("http://localhost/v1", ("localhost", 80)),
        ("https://localhost/v1", ("localhost", 443)),
        ("http://[::1]:8000/v1", ("::1", 8000)),
        ("https://api.example.com/v1", None),
        ("http://192.168.1.2:8317/v1", None),
        ("", None),
    ],
)
def test_local_endpoint(url: str, expected) -> None:
    assert local_endpoint(url) == expected


def test_ready_when_the_local_provider_answers(listening_port: int) -> None:
    connection = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=listening_port)))
    check_ready(connection, {PROVIDER_TOKEN_ENV: "sk-proxy-secret"})


def test_local_provider_not_answering_is_refused(closed_port: int) -> None:
    connection = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=closed_port)))
    with pytest.raises(CodexConnectionError, match="not answering"):
        check_ready(connection, {PROVIDER_TOKEN_ENV: "sk-proxy-secret"})


def test_connect_goes_through_create_connection(monkeypatch) -> None:
    seen = []

    class _Connected:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_connect(address, timeout):
        seen.append((address, timeout))
        return _Connected()

    monkeypatch.setattr(codex_connection.socket, "create_connection", fake_connect)
    check_ready(CodexConnection(base_url="http://localhost/v1"), {})
    check_ready(CodexConnection(base_url="https://api.example.com/v1"), {})

    assert seen == [(("localhost", 80), codex_connection.LOCAL_CONNECT_TIMEOUT_SECONDS)]


def test_missing_provider_variable_is_refused() -> None:
    connection = CodexConnection(passthrough_env=("CORP_KEY",))
    with pytest.raises(CodexConnectionError, match="CORP_KEY"):
        check_ready(connection, {"CORP_KEY": ""})


# -- the driver ----------------------------------------------------------------


def _run_envdump(driver: CodexLocalAgentDriver) -> dict:
    result = driver.run([{"role": "user", "content": "envdump"}], task="general")
    return json.loads(result.content)


def test_driver_carries_the_proxy_and_keeps_the_token_off_argv(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    dump = _run_envdump(driver)
    argv = dump["argv"]

    assert "--ignore-user-config" in argv
    values = [argv[i + 1] for i, item in enumerate(argv) if item == "--config"]
    assert 'model_provider="cliproxyapi"' in values
    assert any(value.startswith("model_providers.cliproxyapi=") for value in values)
    assert all("sk-proxy-secret" not in item for item in argv)
    assert dump["env"][PROVIDER_TOKEN_ENV] == "sk-proxy-secret"
    # Behaviour stays isolated: nothing else from the file is carried.
    assert not any(value.startswith(("model=", "notify", "mcp_servers.node_repl")) for value in values)


def test_driver_refuses_before_spawning_when_the_proxy_is_down(
    tmp_path: Path, monkeypatch, closed_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=closed_port))
    driver = _driver(tmp_path, monkeypatch)

    # Its own error class (a subclass of unavailable), not a bare unavailable.
    with pytest.raises(LocalAgentProviderError, match="not answering") as caught:
        driver.run([{"role": "user", "content": "ok"}], task="general")
    attempts = getattr(caught.value, "_harness_execution_attempts")
    assert attempts[0]["return_code"] is None


def test_driver_refuses_a_provider_that_names_a_finesub_credential(
    tmp_path: Path, monkeypatch
) -> None:
    _write_config(monkeypatch, tmp_path, """
model_provider = "corp"
[model_providers.corp]
base_url = "https://corp.example/v1"
env_key = "OPENAI_API_KEY"
""")
    monkeypatch.setenv("OPENAI_API_KEY", "finesub-own-key")
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="OPENAI_API_KEY.*a variable of its own"):
        driver.run([{"role": "user", "content": "ok"}], task="general")


def test_driver_passes_the_providers_own_variable(tmp_path: Path, monkeypatch) -> None:
    _write_config(monkeypatch, tmp_path, """
model_provider = "corp"
[model_providers.corp]
base_url = "https://corp.example/v1"
env_key = "CORP_KEY"
""")
    monkeypatch.setenv("CORP_KEY", "corp-secret")
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    assert _run_envdump(driver)["env"]["CORP_KEY"] == "corp-secret"


def test_driver_reports_a_broken_config_as_unavailable(tmp_path: Path, monkeypatch) -> None:
    _write_config(monkeypatch, tmp_path, "model_provider = [")
    driver = _driver(tmp_path, monkeypatch)

    with pytest.raises(LocalAgentUnavailableError, match="not valid TOML"):
        driver.run([{"role": "user", "content": "ok"}], task="general")


def test_driver_carries_regardless_of_the_unisolated_opt_in(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    template = _driver(tmp_path, monkeypatch)
    opted_in = CodexLocalAgentDriver(CodexDriverConfig(
        command=template.config.command,
        runtime_root=tmp_path / "runtime-opt-in",
        timeout_seconds=5,
        max_result_bytes=1_000_000,
        allow_unisolated_user_config=True,
    ))

    argv = _run_envdump(opted_in)["argv"]

    assert "--ignore-user-config" in argv
    assert 'model_provider="cliproxyapi"' in argv


def test_isolation_record_names_the_provider_never_the_token(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch)

    result = driver.run([{"role": "user", "content": "ok"}], task="general")
    isolation = result.execution_attempt["isolation"]

    assert isolation["user_config"] == "ignored_except_connection"
    assert isolation["user_connection"] == {
        "carried_keys": ["model_provider"],
        "model_provider": "cliproxyapi",
        "base_url": f"http://127.0.0.1:{listening_port}/v1",
    }
    assert isolation["execpolicy_rules"] == "ignored"
    # The proxy's credentials ride the environment, so the call runs from the
    # harness home and no global AGENTS.md reaches it.
    assert isolation["global_instructions"] == "excluded"
    assert "sk-proxy-secret" not in json.dumps(isolation)
    assert "user_rules" not in isolation


def test_proxy_and_config_dir_variables_reach_the_child(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("OPENAI_API_KEY", "finesub-own-key")
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    env = {key.upper(): value for key, value in _run_envdump(driver)["env"].items()}

    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "claude")
    assert "OPENAI_API_KEY" not in env
    assert os.environ["OPENAI_API_KEY"] == "finesub-own-key"


def test_resume_carries_the_connection_too(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import DriverProbe

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch)
    driver._resolved_command = ("codex",)
    probe = DriverProbe(available=True, no_user_config=True, no_user_rules=True)
    driver._call_local.connection = read_connection()
    driver._call_local.error = None
    capsule = type("Capsule", (), {"root": tmp_path})()

    argv = driver._argv(
        capsule, native_search=False, probe=probe,
        session_scope="assignment", conversation_handle="thread-1",
    )

    assert argv[argv.index("exec") + 1] == "resume"
    assert 'model_provider="cliproxyapi"' in argv
    assert argv.index('model_provider="cliproxyapi"') < argv.index("exec")


def test_concurrent_calls_keep_their_own_connection(tmp_path: Path, monkeypatch) -> None:
    """One driver, two threads, two configs: neither argv borrows the other's."""

    import threading

    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, max_parallel=2)
    connections = {
        "a": CodexConnection(overrides=('model_provider="alpha"',)),
        "b": CodexConnection(overrides=('model_provider="beta"',)),
    }
    gate = threading.Barrier(2)

    def read_for_thread(home=None):
        gate.wait(timeout=10)  # both calls hold their config at once
        return connections[threading.current_thread().name]

    monkeypatch.setattr(codex_connection, "read_connection", read_for_thread)
    seen: dict[str, list[str]] = {}

    def call(name: str) -> None:
        seen[name] = _run_envdump(driver)["argv"]

    threads = [threading.Thread(target=call, args=(n,), name=n) for n in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert 'model_provider="alpha"' in seen["a"] and 'model_provider="beta"' not in seen["a"]
    assert 'model_provider="beta"' in seen["b"] and 'model_provider="alpha"' not in seen["b"]


# -- the harness home ----------------------------------------------------------


def test_credentials_from_env_only_when_the_provider_reads_a_variable() -> None:
    proxy = connection_from_config(tomllib.loads(PROXY_CONFIG.format(port=8317)))
    corp = connection_from_config({
        "model_provider": "corp",
        "model_providers": {"corp": {"base_url": "https://corp.example/v1", "env_key": "CORP_KEY"}},
    })
    login = connection_from_config({"model_provider": "openai"})

    assert proxy.credentials_from_env and corp.credentials_from_env
    assert not login.credentials_from_env
    assert not CodexConnection().credentials_from_env


def test_feature_names_leave_out_removed_ones() -> None:
    from finesub.llm.agent.local_agent import _codex_feature_names

    listing = (
        "plugins                                  stable             true\n"
        "remote_plugin                            under development  false\n"
        "apply_patch_freeform                     removed            false\n"
        "garbage\n"
    )

    assert _codex_feature_names(listing) == {"plugins", "remote_plugin"}


def _harness_home(driver: CodexLocalAgentDriver) -> Path:
    from finesub.llm.agent.local_agent import CODEX_HARNESS_HOME_DIRNAME

    return driver.capsules.resolve_location().parent.resolve() / CODEX_HARNESS_HOME_DIRNAME


def test_harness_home_runs_from_the_domain_and_keeps_the_connection(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import AGENT_USER_RULES_NOTICE

    user_home = _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    result = driver.run([{"role": "user", "content": "envdump"}], task="general")
    dump = json.loads(result.content)
    argv, isolation = dump["argv"], result.execution_attempt["isolation"]

    assert dump["env"]["CODEX_HOME"] == str(_harness_home(driver))
    assert dump["env"]["CODEX_HOME"] != str(user_home)
    # The connection still comes from the user's own home.
    assert 'model_provider="cliproxyapi"' in argv
    assert dump["env"][PROVIDER_TOKEN_ENV] == "sk-proxy-secret"
    exec_at = argv.index("exec")
    assert {argv[i + 1] for i, item in enumerate(argv[:exec_at]) if item == "--disable"} == set(
        CODEX_DISABLED_FEATURES)
    assert "check_for_update_on_startup=false" in argv[:exec_at]
    # Project-level AGENTS.md is off whatever the cwd holds.
    assert argv[argv.index("project_doc_max_bytes=0") - 1] == "--config"
    assert not argv[-1].startswith(AGENT_USER_RULES_NOTICE)
    assert isolation["global_instructions"] == "excluded"
    assert isolation["codex_home"] == "harness"
    assert isolation["disabled_features"] == list(CODEX_DISABLED_FEATURES)
    assert isolation["missing_features"] == []


def test_harness_home_off_runs_from_the_users_home(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import AGENT_USER_RULES_NOTICE

    user_home = _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=False)

    result = driver.run([{"role": "user", "content": "envdump"}], task="general")
    dump = json.loads(result.content)

    assert dump["env"]["CODEX_HOME"] == str(user_home)
    # The same tools stay off in the user's home: a worker uses none of them.
    assert dump["argv"].count("--disable") == len(CODEX_DISABLED_FEATURES)
    assert result.execution_attempt["isolation"]["disabled_features"] == list(CODEX_DISABLED_FEATURES)
    assert "check_for_update_on_startup=false" not in dump["argv"]
    assert "project_doc_max_bytes=0" in dump["argv"]  # in the user's home too
    assert dump["argv"][-1].startswith(AGENT_USER_RULES_NOTICE)
    assert result.execution_attempt["isolation"]["global_instructions"] == "injected"
    assert result.execution_attempt["isolation"]["codex_home"] == "user"
    assert not _harness_home(driver).exists()


def test_harness_home_is_the_default(tmp_path: Path, monkeypatch, listening_port: int) -> None:
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = CodexLocalAgentDriver(CodexDriverConfig(
        command=_driver(tmp_path, monkeypatch).config.command,
        runtime_root=tmp_path / "runtime",
        timeout_seconds=5,
        max_result_bytes=1_000_000,
    ))

    assert _run_envdump(driver)["env"]["CODEX_HOME"] == str(_harness_home(driver))


@pytest.mark.parametrize("config", [None, 'model_provider = "openai"\n'], ids=["no-config", "openai"])
def test_a_login_connection_keeps_the_users_home(tmp_path: Path, monkeypatch, config) -> None:
    """Copying a ChatGPT login would rotate its refresh token and log the user out.

    The same for an API key kept by `codex login`: both live in the user's home."""

    user_home = _write_config(monkeypatch, tmp_path, config)
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    result = driver.run([{"role": "user", "content": "envdump"}], task="general")

    argv = json.loads(result.content)["argv"]

    assert json.loads(result.content)["env"]["CODEX_HOME"] == str(user_home)
    assert result.execution_attempt["isolation"]["global_instructions"] == "injected"
    # A ChatGPT login brings the account's connected apps (`codex_apps`) along; a worker gets none.
    assert argv[argv.index("--disable") + 1] == "apps" and argv.index("--disable") < argv.index("exec")


def test_a_missing_feature_is_left_off_and_warned_once(
    tmp_path: Path, monkeypatch, listening_port: int, reported
) -> None:
    from finesub.llm.agent import local_agent as la

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    monkeypatch.setattr(la, "_WARNED_MISSING_CODEX_FEATURES", set())
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)
    # As if `features list` dropped one.
    driver._features = frozenset(CODEX_DISABLED_FEATURES) - {"remote_plugin"}

    first = driver.run([{"role": "user", "content": "envdump"}], task="general")
    driver.run([{"role": "user", "content": "envdump"}], task="general")
    argv = json.loads(first.content)["argv"]

    assert argv.count("--disable") == len(CODEX_DISABLED_FEATURES) - 1 and "remote_plugin" not in argv
    assert first.execution_attempt["isolation"]["missing_features"] == ["remote_plugin"]
    notices = [w for w in reported.warnings if w.code == "agent-codex-feature-missing"]
    assert len(notices) == 1 and "remote_plugin" in notices[0].message


@pytest.mark.parametrize("planted", ["AGENTS.md", "AGENTS.override.md", "config.toml", "skills/mine", "rules", "hooks.json", ".env"])
def test_a_file_codex_would_read_fails_before_spawn(
    tmp_path: Path, monkeypatch, listening_port: int, planted: str
) -> None:
    from finesub.llm.agent.local_agent import LocalAgentPolicyViolationError

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, harness_home=True)
    home = _harness_home(driver)
    (home / "skills" / ".system").mkdir(parents=True)
    (home / "sessions").mkdir()
    (home / "state_5.sqlite-wal").write_text("", encoding="utf-8")
    target = home / planted
    if planted.startswith("skills/"):
        target.mkdir()
    else:
        target.write_text("rules", encoding="utf-8")

    with pytest.raises(LocalAgentPolicyViolationError, match=planted.split("/")[-1]) as caught:
        driver.run([{"role": "user", "content": "ok"}], task="general")

    assert getattr(caught.value, "_harness_execution_attempts")[0]["return_code"] is None
    assert target.exists()


def test_an_unfamiliar_entry_is_reported_once_and_not_refused(
    tmp_path: Path, monkeypatch, listening_port: int, reported
) -> None:
    """Codex makes some of its state lazily; refusing it would stop the domain."""

    from finesub.llm.agent import local_agent as la

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    monkeypatch.setattr(la, "_REPORTED_CODEX_HOME_ENTRIES", set())
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)
    home = _harness_home(driver)
    home.mkdir(parents=True)
    (home / "session_index.jsonl").write_text("{}", encoding="utf-8")
    (home / ".sqlite-maintenance.lock").write_text("", encoding="utf-8")
    for own in ("log", "tmp", ".sandbox"):  # Codex 0.171.0 makes these
        (home / own).mkdir()

    _run_envdump(driver)
    _run_envdump(driver)

    notices = [w for w in reported.warnings if w.code == "agent-codex-home-unknown"]
    assert len(notices) == 1 and "session_index.jsonl" in notices[0].message

def test_a_repair_chain_resumes_in_the_harness_home(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import AGENT_TASK_PROMPT_STDIN_ONLY, AGENT_USER_RULES_NOTICE

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)
    key = "assignment-1:worker-1"

    first = driver.run([{"role": "user", "content": "envdump"}], task="general",
                       session_scope="assignment", conversation_key=key)
    second = driver.run([{"role": "user", "content": "envdump"}], task="general",
                        session_scope="assignment", conversation_key=key,
                        conversation_handle=first.conversation_handle)
    one, two = json.loads(first.content), json.loads(second.content)

    assert one["env"]["CODEX_HOME"] == two["env"]["CODEX_HOME"] == str(_harness_home(driver))
    argv = two["argv"]
    assert argv[argv.index("exec") + 1] == "resume"
    assert "check_for_update_on_startup=false" in argv[:argv.index("exec")]
    assert not one["argv"][-1].startswith(AGENT_USER_RULES_NOTICE)
    driver._call_local.connection = read_connection()  # as `run` sets it
    framed, _ = driver._prepare_capsule_input(None, b"{}", conversation_handle="thread-1")
    assert not framed.startswith(AGENT_USER_RULES_NOTICE.encode("utf-8"))
    assert framed.startswith(AGENT_TASK_PROMPT_STDIN_ONLY.encode("utf-8"))


# -- FineSub's own login home ----------------------------------------------------


def _login_home(logged_in: bool = True) -> Path:
    """The conftest's stand-in for `paths.resolve_codex_login_home`, made."""

    from finesub import paths

    home = paths.resolve_codex_login_home()
    home.mkdir(parents=True, exist_ok=True)
    if logged_in:
        (home / "auth.json").write_text("{}", encoding="utf-8")
        (home / "auth.lock").write_text("", encoding="utf-8")
    return home


@pytest.mark.parametrize("config", [None, 'model_provider = "openai"\n'], ids=["no-config", "openai"])
def test_a_login_in_finesubs_home_runs_from_it(tmp_path: Path, monkeypatch, config, reported) -> None:
    from finesub.llm.agent.local_agent import AGENT_USER_RULES_NOTICE

    user_home = _write_config(monkeypatch, tmp_path, config)
    home = _login_home()
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    result = driver.run([{"role": "user", "content": "envdump"}], task="general")
    dump, isolation = json.loads(result.content), result.execution_attempt["isolation"]
    argv = dump["argv"]

    assert dump["env"]["CODEX_HOME"] == str(home.resolve()) != str(user_home)
    assert isolation["codex_home"] == "login"
    assert isolation["global_instructions"] == "excluded"
    assert not argv[-1].startswith(AGENT_USER_RULES_NOTICE)
    # The same flags as the harness home: no plugin clone, no update check.
    exec_at = argv.index("exec")
    assert "check_for_update_on_startup=false" in argv[:exec_at]
    assert {"plugins", "remote_plugin"} <= {argv[i + 1] for i, item in enumerate(argv[:exec_at]) if item == "--disable"}
    assert not [w for w in reported.warnings if w.code == "agent-codex-login-missing"]


def test_without_finesubs_login_the_call_uses_the_users_home_and_says_how_to_fix_it(
    tmp_path: Path, monkeypatch, reported
) -> None:
    from finesub.llm.agent import codex_login, local_agent as la

    monkeypatch.setattr(la, "_WARNED_CODEX_LOGIN_MISSING", set())
    user_home = _write_config(monkeypatch, tmp_path, None)
    _login_home(logged_in=False)
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    first = driver.run([{"role": "user", "content": "envdump"}], task="general")
    driver.run([{"role": "user", "content": "envdump"}], task="general")
    dump = json.loads(first.content)

    assert dump["env"]["CODEX_HOME"] == str(user_home)
    assert dump["argv"][-1].startswith(la.AGENT_USER_RULES_NOTICE)
    assert first.execution_attempt["isolation"]["codex_home"] == "user"
    assert first.execution_attempt["isolation"]["global_instructions"] == "injected"
    notices = [w for w in reported.warnings if w.code == "agent-codex-login-missing"]
    assert len(notices) == 1 and codex_login.login_command() in notices[0].message


def test_a_login_status_nobody_could_read_falls_back_to_the_users_home_with_its_own_warning(
    tmp_path: Path, monkeypatch, reported
) -> None:
    """Logged in or not is unknown: the user's home, as without a login
    (owner 2026-11-14), but the warning does not tell a logged-in user to log in."""

    from finesub.llm.agent import codex_login, local_agent as la

    monkeypatch.setattr(la, "_WARNED_CODEX_LOGIN_MISSING", set())
    user_home = _write_config(monkeypatch, tmp_path, None)
    _login_home()
    monkeypatch.setattr(
        codex_login, "status",
        lambda *a: codex_login.LoginStatus(False, "timed out after 30 s", answered=False),
    )
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    result = driver.run([{"role": "user", "content": "envdump"}], task="general")
    dump = json.loads(result.content)

    assert dump["env"]["CODEX_HOME"] == str(user_home)
    assert dump["argv"][-1].startswith(la.AGENT_USER_RULES_NOTICE)
    assert result.execution_attempt["isolation"]["global_instructions"] == "injected"
    assert result.execution_attempt["isolation"]["codex_login"] == "unknown"
    (notice,) = [w for w in reported.warnings if w.code.startswith("agent-codex-login")]
    assert notice.code == "agent-codex-login-unknown"
    assert "timed out after 30 s" in notice.message and codex_login.login_command() not in notice.message

def test_a_connection_with_env_credentials_never_hears_about_the_login(
    tmp_path: Path, monkeypatch, listening_port: int, reported
) -> None:
    from finesub.llm.agent import local_agent as la

    monkeypatch.setattr(la, "_WARNED_CODEX_LOGIN_MISSING", set())
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    _run_envdump(driver)

    assert not [w for w in reported.warnings if w.code == "agent-codex-login-missing"]


def test_the_login_state_is_read_once_per_driver(tmp_path: Path, monkeypatch) -> None:
    from finesub.llm.agent import codex_login

    _write_config(monkeypatch, tmp_path, None)
    _login_home()
    asked: list[Path] = []
    real = codex_login.status

    def counted(command, home, connection, base_env):
        asked.append(home)
        return real(command, home, connection, base_env)

    monkeypatch.setattr(codex_login, "status", counted)
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    _run_envdump(driver)
    _run_envdump(driver)

    assert len(asked) == 1


def test_a_rule_file_in_the_login_home_fails_before_spawn(tmp_path: Path, monkeypatch) -> None:
    from finesub.llm.agent.local_agent import LocalAgentPolicyViolationError

    _write_config(monkeypatch, tmp_path, None)
    home = _login_home()
    (home / "AGENTS.md").write_text("rules", encoding="utf-8")
    driver = _driver(tmp_path, monkeypatch, harness_home=True)

    with pytest.raises(LocalAgentPolicyViolationError, match="AGENTS.md"):
        driver.run([{"role": "user", "content": "ok"}], task="general")


def test_the_login_home_may_sit_in_the_checkout_only_as_its_own_root_entry(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub import paths
    from finesub.llm.agent.local_agent import LocalAgentPolicyViolationError, _codex_login_home

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(paths, "own_checkout_root", lambda: checkout)

    monkeypatch.setattr(paths, "resolve_codex_login_home", lambda: checkout / paths.CODEX_LOGIN_HOME_DIRNAME)
    assert _codex_login_home() == (checkout / paths.CODEX_LOGIN_HOME_DIRNAME).resolve()

    monkeypatch.setattr(paths, "resolve_codex_login_home", lambda: checkout / "nested" / "home")
    with pytest.raises(LocalAgentPolicyViolationError, match="outside a repository"):
        _codex_login_home()


def test_without_a_data_root_the_warning_names_no_login_command(
    tmp_path: Path, monkeypatch, reported
) -> None:
    from finesub import paths
    from finesub.llm.agent import codex_login, local_agent as la

    monkeypatch.setattr(la, "_WARNED_CODEX_LOGIN_MISSING", set())
    _write_config(monkeypatch, tmp_path, None)
    monkeypatch.setattr(paths, "resolve_codex_login_home", lambda: None)
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, harness_home=True)

    _run_envdump(driver)

    (notice,) = [w for w in reported.warnings if w.code == "agent-codex-login-missing"]
    assert codex_login.login_command() not in notice.message
    assert "FINESUB_CHECKOUT_DATA=0" in notice.message


@pytest.mark.parametrize(
    "stderr, rejected",
    [
        ("unexpected status 401 Unauthorized", True),
        ("HTTP 401", True),
        ("request id 84017 failed: 500", False),
        ("took 1.401 s, then 503", False),
    ],
)
def test_a_login_rejection_is_read_from_a_401_status_not_any_digits(
    tmp_path: Path, stderr: str, rejected: bool
) -> None:
    from types import SimpleNamespace

    from finesub.llm.agent.local_agent import _codex_login_rejected

    path = tmp_path / "stderr.log"
    path.write_text(stderr, encoding="utf-8")

    assert _codex_login_rejected(SimpleNamespace(stderr_path=path)) is rejected

def test_an_expired_login_names_the_login_command_and_is_not_a_quota_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Unavailable, so a run of them never reaches the quota probe's freeze."""

    from finesub.llm.agent import codex_login
    from finesub.llm.agent.local_agent import LocalAgentUnavailableError

    _write_config(monkeypatch, tmp_path, None)
    _login_home()
    driver = _driver(tmp_path, monkeypatch, harness_home=True)

    with pytest.raises(LocalAgentUnavailableError, match=re.escape(codex_login.login_command())):
        driver.run([{"role": "user", "content": "auth-fail"}], task="general")


def test_a_linked_login_home_is_refused_before_it_is_followed(tmp_path: Path, monkeypatch) -> None:
    """A junction at the checkout's login home pointing at a sibling would
    pass a check made on the resolved path: the calls would use another
    directory's login, kept where git does not ignore it."""

    from finesub import paths
    from finesub.llm.agent.local_agent import LocalAgentPolicyViolationError, _codex_login_home

    from .support.local_agent import _make_directory_link

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "elsewhere").mkdir()
    _make_directory_link(checkout / paths.CODEX_LOGIN_HOME_DIRNAME, checkout / "elsewhere")
    monkeypatch.setattr(paths, "own_checkout_root", lambda: checkout)
    monkeypatch.setattr(paths, "resolve_codex_login_home", lambda: checkout / paths.CODEX_LOGIN_HOME_DIRNAME)

    with pytest.raises(LocalAgentPolicyViolationError, match="symlink/reparse"):
        _codex_login_home()

    # Nor does a sibling directory pass as the login home by its parent alone.
    monkeypatch.setattr(paths, "resolve_codex_login_home", lambda: checkout / "elsewhere")
    with pytest.raises(LocalAgentPolicyViolationError, match="outside a repository"):
        _codex_login_home()


# -- what the request carries ------------------------------------------------------


def test_every_call_drops_the_sections_a_worker_has_no_use_for(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import CODEX_INSTRUCTION_OVERRIDES, CODEX_NON_NATIVE_OVERRIDES

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000)

    plain = _run_envdump(driver)["argv"]
    native = json.loads(driver.run(
        [{"role": "user", "content": "envdump"}], task="general", native_search=True
    ).content)["argv"]

    def configs(argv: list[str]) -> list[str]:
        head = argv[:argv.index("exec")]
        return [head[i + 1] for i, item in enumerate(head) if item == "--config"]

    for argv in (plain, native):
        assert set(CODEX_INSTRUCTION_OVERRIDES) <= set(configs(argv))
    assert set(CODEX_NON_NATIVE_OVERRIDES) <= set(configs(plain))
    # The native round keeps its search tool and its only date.
    assert not set(CODEX_NON_NATIVE_OVERRIDES) & set(configs(native))
    assert "--search" in native


def test_code_mode_stays_on_and_its_tools_go_through_the_catalog() -> None:
    from finesub.llm.agent.local_agent import CODEX_CATALOG_CLEARED, CODEX_DISABLED_FEATURES

    assert not [name for name in CODEX_DISABLED_FEATURES if name.startswith("code_mode")]
    assert set(CODEX_CATALOG_CLEARED) == {
        "tool_mode", "multi_agent_version", "apply_patch_tool_type", "experimental_supported_tools"}


def test_the_catalog_is_content_addressed_in_the_domain(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    from finesub.llm.agent.local_agent import (
        CODEX_CATALOG_CLEARED,
        CODEX_HARNESS_HOME_DIRNAME,
        CODEX_MODEL_CATALOG_DIRNAME,
    )

    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, model="gpt-fake")

    def catalog(argv: list[str]) -> Path:
        value = next(item for item in argv if item.startswith("model_catalog_json="))
        assert argv.index(value) < argv.index("exec")
        return Path(value.partition("=")[2].strip('"'))

    first, second = catalog(_run_envdump(driver)["argv"]), catalog(_run_envdump(driver)["argv"])
    domain = driver.capsules.resolve_location().parent.resolve()

    assert first == second
    assert first.parent == domain / CODEX_MODEL_CATALOG_DIRNAME
    assert CODEX_HARNESS_HOME_DIRNAME not in first.parts
    (entry,) = json.loads(first.read_text(encoding="utf-8"))["models"]
    assert entry["slug"] == "gpt-fake" and entry["context_window"] == 1000
    assert {key: entry[key] for key in CODEX_CATALOG_CLEARED} == CODEX_CATALOG_CLEARED


def test_a_model_the_catalog_lacks_gets_none_and_a_warning(
    tmp_path: Path, monkeypatch, listening_port: int, reported
) -> None:
    from finesub.llm.agent import local_agent as la

    monkeypatch.setattr(la, "_WARNED_CODEX_CATALOG", set())
    _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, model="gpt-unlisted")

    argv = _run_envdump(driver)["argv"]
    _run_envdump(driver)

    assert not [item for item in argv if item.startswith("model_catalog_json=")]
    notices = [w for w in reported.warnings if w.code == "agent-codex-catalog-missing"]
    assert len(notices) == 1 and "gpt-unlisted" in notices[0].message


def test_the_users_cached_catalog_comes_before_the_bundled_one(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    user_home = _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    (user_home / "models_cache.json").write_text(json.dumps({"client_version": "0.169.1", "models": [
        {"slug": "gpt-fake", "context_window": 2000, "tool_mode": "code"}]}), encoding="utf-8")
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, model="gpt-fake")

    argv = _run_envdump(driver)["argv"]
    value = next(item for item in argv if item.startswith("model_catalog_json="))
    (entry,) = json.loads(Path(value.partition("=")[2].strip('"')).read_text(encoding="utf-8"))["models"]

    assert entry["context_window"] == 2000 and entry["tool_mode"] is None


def test_a_cached_catalog_from_another_codex_version_is_passed_over(
    tmp_path: Path, monkeypatch, listening_port: int
) -> None:
    """The user's home is also written by other Codex clients; an entry this
    binary may not parse is skipped for the bundled catalog."""

    user_home = _write_config(monkeypatch, tmp_path, PROXY_CONFIG.format(port=listening_port))
    (user_home / "models_cache.json").write_text(json.dumps({"client_version": "0.170.0", "models": [
        {"slug": "gpt-fake", "context_window": 2000, "tool_mode": "code"}]}), encoding="utf-8")
    driver = _driver(tmp_path, monkeypatch, max_result_bytes=1_000_000, model="gpt-fake")

    argv = _run_envdump(driver)["argv"]
    value = next(item for item in argv if item.startswith("model_catalog_json="))
    (entry,) = json.loads(Path(value.partition("=")[2].strip('"')).read_text(encoding="utf-8"))["models"]

    assert entry["context_window"] == 1000  # the bundled one
