"""`finesub agent-codex-login`: FineSub's own Codex login, apart from ~/.codex."""

from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from finesub import paths
from finesub.llm.agent import codex_login, local_agent


@pytest.fixture
def codex(monkeypatch, tmp_path: Path):
    """Every `codex` the command starts, recorded instead of run."""

    calls: list[dict] = []
    answers = {"status": 1}

    def run(argv, **kwargs):
        calls.append({"argv": list(argv), "env": dict(kwargs.get("env") or {}), "kwargs": kwargs})
        if argv[-2:] == ["login", "status"]:
            code = answers["status"]
            said = "Logged in using ChatGPT" if code == 0 else "Not logged in"
            return SimpleNamespace(returncode=code, stdout=said + "\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(codex_login.subprocess, "run", run)
    monkeypatch.setattr(local_agent, "_resolve_shell_free_command", lambda command: ("codex",))
    user_home = tmp_path / "user-codex"
    user_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(user_home))
    return SimpleNamespace(calls=calls, answers=answers, user_home=user_home)


def test_the_login_command_follows_how_finesub_runs(monkeypatch, tmp_path: Path) -> None:
    """A checkout has no `finesub` on PATH, and an installed `finesub` run in
    one logs into its own user-data: a checkout is told to run the module."""

    monkeypatch.setattr(paths, "checkout_data_enabled", lambda: True)
    monkeypatch.setattr(paths, "own_checkout_root", lambda: tmp_path)
    assert codex_login.login_command() == "python -m finesub.llm.agent.codex_login"

    monkeypatch.setattr(paths, "own_checkout_root", lambda: None)
    assert codex_login.login_command() == "finesub agent-codex-login"

    monkeypatch.setattr(paths, "own_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "checkout_data_enabled", lambda: False)
    assert codex_login.login_command() == "finesub agent-codex-login"

def test_login_runs_codex_login_in_finesubs_home_on_the_terminal(codex, capsys) -> None:
    (codex.user_home / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n', encoding="utf-8"
    )
    home = paths.resolve_codex_login_home()

    code = codex_login.main(["--device-auth"])

    login = codex.calls[0]
    assert code == 0
    assert login["argv"] == [
        "codex", "--config", 'cli_auth_credentials_store="keyring"', "login", "--device-auth"]
    assert login["env"]["CODEX_HOME"] == str(home)
    # Interactive: the child gets the terminal, nothing is captured.
    assert "capture_output" not in login["kwargs"] and "stdout" not in login["kwargs"]
    assert home.is_dir()
    out = capsys.readouterr().out
    assert str(home) in out
    assert "separate from your own ~/.codex" in out
    assert "personal rules" in out
    assert f"{codex_login.login_command()} --logout" in out


def test_status_says_where_and_whether(codex, capsys) -> None:
    home = paths.resolve_codex_login_home()
    home.mkdir(parents=True)

    assert codex_login.main(["--status"]) == 1
    assert "Not logged in" in capsys.readouterr().out

    codex.answers["status"] = 0
    assert codex_login.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert str(home) in out and "Logged in using ChatGPT" in out
    assert all(call["env"]["CODEX_HOME"] == str(home) for call in codex.calls)


def test_logout_runs_in_finesubs_home_only(codex) -> None:
    home = paths.resolve_codex_login_home()
    home.mkdir(parents=True)

    assert codex_login.main(["--logout"]) == 0
    assert codex.calls[-1]["argv"] == ["codex", "logout"]
    assert codex.calls[-1]["env"]["CODEX_HOME"] == str(home)


def test_a_provider_keyed_from_the_environment_needs_no_login(codex, capsys) -> None:
    (codex.user_home / "config.toml").write_text(
        'model_provider = "corp"\n[model_providers.corp]\nbase_url = "https://corp.example/v1"\n'
        'env_key = "CORP_KEY"\n',
        encoding="utf-8",
    )

    assert codex_login.main([]) == 0
    assert "no login needed" in capsys.readouterr().out
    assert codex.calls == []


def test_no_data_root_means_no_login_home(codex, monkeypatch, capsys) -> None:
    monkeypatch.setattr(paths, "resolve_codex_login_home", lambda: None)

    assert codex_login.main([]) == 2
    assert "your own Codex home" in capsys.readouterr().out
    assert codex.calls == []


def test_a_rule_file_in_the_home_stops_the_login(codex, capsys) -> None:
    home = paths.resolve_codex_login_home()
    home.mkdir(parents=True)
    (home / "AGENTS.md").write_text("rules", encoding="utf-8")

    assert codex_login.main([]) == 2
    assert "AGENTS.md" in capsys.readouterr().out
    assert codex.calls == []


def test_status_of_a_home_not_made_yet_asks_nobody(tmp_path: Path) -> None:
    state = codex_login.status(("codex",), tmp_path / "nowhere", local_agent.codex_connection.CodexConnection(), {})

    assert state == codex_login.LoginStatus(False, "no login home yet")


def test_status_reads_codex_and_skips_its_warnings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(codex_login.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout="WARNING: proceeding, even though ...\nLogged in using ChatGPT\n", stderr=""))

    state = codex_login.status(("codex",), tmp_path, local_agent.codex_connection.CodexConnection(), {})

    assert state == codex_login.LoginStatus(True, "Logged in using ChatGPT")


def test_a_status_that_does_not_answer_is_unknown_not_logged_out(monkeypatch, tmp_path: Path) -> None:
    from finesub.llm.agent import codex_connection

    def hang(*_a, **_k):
        raise subprocess.TimeoutExpired("codex", 30)

    monkeypatch.setattr(codex_login.subprocess, "run", hang)
    (tmp_path / "user-codex").mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "user-codex"))  # not the real ~/.codex
    home = tmp_path / "home"
    home.mkdir()
    state = codex_login.status(("codex",), home, codex_connection.read_connection(), {})

    assert not state.logged_in and not state.answered


@pytest.mark.parametrize("answered", [True, False])
def test_status_advises_a_login_only_when_codex_said_there_is_none(
    codex, monkeypatch, capsys, answered
) -> None:
    home = paths.resolve_codex_login_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        codex_login, "status",
        lambda *a: codex_login.LoginStatus(False, "Not logged in" if answered else "timed out", answered=answered),
    )

    assert codex_login.main(["--status"]) == 1
    out = capsys.readouterr().out
    assert ("to log in" in out) is answered
    assert ("Could not tell" in out) is not answered


def test_agent_ping_does_not_advise_a_login_when_the_status_was_unknown() -> None:
    from finesub.llm.agent import agent_ping

    def row(isolation):
        outcome = SimpleNamespace(execution_attempt={"isolation": isolation})
        return agent_ping._with_codex_home({}, outcome)["codex_home"]

    unknown = row({"codex_home": "user", "codex_login": "unknown"})
    assert unknown == "user-unknown"
    assert "give FineSub a login" not in agent_ping._codex_home_note(unknown)
    assert "give FineSub a login" in agent_ping._codex_home_note(row({"codex_home": "user"}))
