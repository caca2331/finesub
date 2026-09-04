from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from finesub_bootstrap.capabilities import (
    capabilities_from_arguments,
    preferred_capabilities_from_arguments,
)
from finesub_bootstrap.environment import shared_environment_overrides
from finesub_bootstrap.system_tools import SystemTool
from finesub_cli import main as cli


def test_home_prefers_the_explicit_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_HOME", str(tmp_path / "elsewhere"))

    assert cli.resolve_home() == (tmp_path / "elsewhere").resolve()


def test_home_defaults_to_local_app_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FINESUB_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert cli.resolve_home() == (
        (tmp_path / "LocalAppData").resolve() / "FineSub"
    )


def test_no_arguments_prints_usage_and_fails(capsys) -> None:
    assert cli.main([]) == 2
    assert "finesub <input>" in capsys.readouterr().out


def test_help_prints_usage_and_succeeds(capsys) -> None:
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    for subcommand in (
        "setup",
        "doctor",
        "keys",
        "uninstall",
        "agent-clean",
    ):
        assert subcommand in output
    # `batch` is gone (2026-08-30): the bare form takes any number of sources
    # and `--manifest`, so advertising a subcommand would restate a distinction
    # the pipeline no longer makes.
    assert "batch" not in output


def test_commands_go_to_the_shared_shell(monkeypatch) -> None:
    # Dispatch itself is shared with the desktop package (test_shell.py); this
    # front end only has to hand it the arguments untouched.
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli,
        "_shell",
        lambda: SimpleNamespace(
            dispatch=lambda arguments: calls.append(list(arguments)) or 0,
            # The front end also reads the shared data location, for the
            # update notice. Captured stderr is not a TTY, so the check itself
            # stays inert here -- but the attribute has to exist.
            paths=SimpleNamespace(user_data=Path("."), data_root=Path(".")),
        ),
    )

    assert cli.main(["input.wav", "--language", "en", "--word"]) == 0
    assert calls == [["input.wav", "--language", "en", "--word"]]


def _fake_shell(monkeypatch, tmp_path: Path, calls: list[list[str]]) -> None:
    monkeypatch.setattr(
        cli,
        "_shell",
        lambda: SimpleNamespace(
            dispatch=lambda arguments: calls.append(list(arguments)) or 0,
            paths=SimpleNamespace(user_data=tmp_path, data_root=tmp_path),
        ),
    )


def test_the_update_notice_lands_on_stderr_after_the_command(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """stdout carries pipeline output, and a notice printed first is unread.

    The command's own exit status has to survive it, too -- a version notice
    that changed what a script sees would be a far worse bug than a missing one.
    """

    import json

    from finesub_bootstrap import update_check

    (tmp_path / update_check.STATE_FILENAME).write_text(
        json.dumps({"latest": "9.9.9", "checked_at": 0.0}), encoding="utf-8"
    )
    calls: list[list[str]] = []
    _fake_shell(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(cli, "installed_version", lambda: "0.4.2")
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(update_check, "fetch_latest", lambda: "")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)

    assert cli.main(["input.wav"]) == 0

    captured = capsys.readouterr()
    assert "9.9.9" in captured.err and update_check.UPGRADE_COMMAND in captured.err
    assert "9.9.9" not in captured.out
    assert calls == [["input.wav"]]


def test_no_update_notice_without_a_tty_or_on_a_quiet_command(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import json

    from finesub_bootstrap import update_check

    (tmp_path / update_check.STATE_FILENAME).write_text(
        json.dumps({"latest": "9.9.9", "checked_at": 0.0}), encoding="utf-8"
    )
    calls: list[list[str]] = []
    _fake_shell(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(cli, "installed_version", lambda: "0.4.2")
    monkeypatch.setattr(update_check, "fetch_latest", lambda: "")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)

    # Captured stderr is not a TTY: somebody is redirecting, not reading.
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: False, raising=False)
    assert cli.main(["input.wav"]) == 0
    assert "9.9.9" not in capsys.readouterr().err

    # `setup` is the run right after installing; nagging there reads as a bug.
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    assert cli.main(["setup"]) == 0
    assert "9.9.9" not in capsys.readouterr().err


def test_uninstall_removes_rebuildable_state_and_keeps_the_rest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    # Sorted by whether it can be recreated: the runtime, models and downloads
    # go by default; finished subtitles and personal data need a flag.
    home = tmp_path / "FineSub"
    monkeypatch.setenv("FINESUB_HOME", str(home))
    _vendored(tmp_path, monkeypatch)
    paths = cli._shell().paths
    for directory in (
        paths.runtime,
        paths.models,
        paths.cache,
        paths.tasks,
        paths.agent_capsules,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "content.bin").write_bytes(b"data")
    paths.user_data.mkdir(parents=True, exist_ok=True)
    (paths.user_data / ".env").write_text("GEMINI_FREE=key", "utf-8")

    assert cli.main(["uninstall"]) == 0

    assert not paths.runtime.exists()
    assert not paths.models.exists()
    assert not paths.cache.exists()
    assert not paths.agent_capsules.exists()
    assert (paths.tasks / "content.bin").is_file()
    assert (paths.user_data / ".env").is_file()
    output = capsys.readouterr().out
    assert "--purge-user-data" in output
    assert "--purge-tasks" in output


def test_uninstall_purges_the_rest_only_on_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "FineSub"
    monkeypatch.setenv("FINESUB_HOME", str(home))
    _vendored(tmp_path, monkeypatch)
    paths = cli._shell().paths
    paths.user_data.mkdir(parents=True, exist_ok=True)
    (paths.user_data / ".env").write_text("GEMINI_FREE=key", "utf-8")
    paths.tasks.mkdir(parents=True, exist_ok=True)
    (paths.tasks / "clip").mkdir()

    assert cli.main(["uninstall", "--purge-user-data", "--purge-tasks"]) == 0

    assert not paths.user_data.exists()
    assert not paths.tasks.exists()
    assert not home.exists()


def test_uninstall_keeps_a_shared_store_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    # Once the data has been pointed somewhere else, another installation is
    # probably reading it: leaving a few GB costs disk, deleting someone else's
    # copy costs them a download.
    from finesub_bootstrap.paths import ensure_store

    home = tmp_path / "FineSub"
    monkeypatch.setenv("FINESUB_HOME", str(home))
    _vendored(tmp_path, monkeypatch)
    shared = cli._shell().paths.with_big_data(tmp_path / "shared")
    ensure_store(shared)
    shared.models.mkdir(parents=True, exist_ok=True)
    (shared.models / "weights.bin").write_bytes(b"data")
    shared.agent_capsules.mkdir(parents=True)
    (shared.agent_capsules / "failed-call").mkdir()

    assert cli.main(["uninstall"]) == 0

    assert (shared.models / "weights.bin").is_file()
    output = capsys.readouterr().out
    assert "--purge-big-data" in output
    assert str(shared.agent_capsules) in output
    assert "finesub agent-clean" in output


def test_uninstall_rejects_unknown_options(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_HOME", str(tmp_path / "FineSub"))
    _vendored(tmp_path, monkeypatch)

    assert cli.main(["uninstall", "--force"]) == 2


def test_shared_environment_defers_to_explicit_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    user_data = tmp_path / "user-data"
    user_data.mkdir()
    (user_data / ".env").write_text("GEMINI_FREE=key", "utf-8")
    (user_data / "config.toml").write_text("[pools]", "utf-8")
    paths = SimpleNamespace(
        user_data=user_data,
        cache=tmp_path / "cache",
        agent_capsules=tmp_path / "agent-capsules",
    )

    monkeypatch.delenv("FINESUB_ENV_FILE", raising=False)
    monkeypatch.delenv("FINESUB_CONFIG_FILE", raising=False)
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.delenv("FINESUB_STATE_DIR", raising=False)
    overrides = shared_environment_overrides(paths)
    assert overrides["FINESUB_ENV_FILE"] == str(user_data / ".env")
    assert overrides["FINESUB_CONFIG_FILE"] == str(user_data / "config.toml")
    assert overrides["FINESUB_KNOWLEDGE_ROOT"] == str(user_data / "knowledge")
    assert overrides["FINESUB_STATE_DIR"] == str(tmp_path / "cache" / "state")

    monkeypatch.setenv("FINESUB_ENV_FILE", "explicit.env")
    monkeypatch.setenv("FINESUB_KNOWLEDGE_ROOT", "explicit-knowledge")
    monkeypatch.setenv("FINESUB_STATE_DIR", "explicit-state")
    overrides = shared_environment_overrides(paths)
    assert "FINESUB_ENV_FILE" not in overrides
    assert "FINESUB_KNOWLEDGE_ROOT" not in overrides
    assert "FINESUB_STATE_DIR" not in overrides


def test_capability_rules_are_shared_with_the_desktop() -> None:
    # The desktop reads a TaskRequest, the CLI reads a command line. If the two
    # disagreed, a task could start on one and be refused on the other.
    #
    # A knowledge update no longer needs anything on demand: the knowledge base
    # became a SQLite store, so the embedded git repo -- and the `git`
    # capability every `--knowledge update` run used to require -- is gone
    # (`required_capabilities` says so in as many words). This test kept
    # asserting the pre-SQLite contract and had been red ever since; nothing
    # noticed because `cli/tests` only runs at release time.
    for arguments in (
        ["a.wav"],
        ["a.wav", "--knowledge", "update"],
        ["a.wav", "--knowledge", "update", "--stage", "final-srt"],
        ["a.wav", "--knowledge=update", "--stage=translated-srt"],
        ["a.wav", "--knowledge=update", "--llm-correct-translate"],
        ["a.wav", "--knowledge", "collect", "--stage", "final-srt"],
    ):
        assert capabilities_from_arguments(arguments) == (), arguments
    # A URL still needs the downloader, whatever else the run asks for.
    assert capabilities_from_arguments(["https://example.test/v"]) == ("yt-dlp",)
    assert capabilities_from_arguments(
        ["https://example.test/v", "--knowledge=update", "--stage=final-srt"]
    ) == ("yt-dlp",)


def test_an_explicit_stage_beats_the_convenience_flag() -> None:
    """`pipeline.main`: `args.stage or (... if correct_translate else ...)`.

    Read in argument order, `--stage raw-srt --llm-correct-translate` looked
    like an LLM run to everything here while the pipeline stopped at the
    transcript -- fetching tokcount for nothing, demanding git when knowledge
    was set to update, and filing the task under a stage it never reached.
    """

    for arguments in (
        ["a.wav", "--stage", "raw-srt", "--llm-correct-translate"],
        ["a.wav", "--llm-correct-translate", "--stage", "raw-srt"],
        ["a.wav", "--llm-correct-translate", "--stage=raw-srt"],
    ):
        assert preferred_capabilities_from_arguments(arguments) == ()
        assert capabilities_from_arguments([*arguments, "--knowledge=update"]) == ()

    # Without one, the flag still selects it, which is what the flag is for.
    assert preferred_capabilities_from_arguments(
        ["a.wav", "--llm-correct-translate"]
    ) == ("tokcount",)


def test_the_token_counter_is_preferred_and_never_required() -> None:
    """It has to stay out of the list the front ends gate a run on.

    The pipeline counts tokens through the free countTokens endpoint without
    it, so a failed download must cost a network round trip per count -- not
    the run.
    """

    assert "tokcount" not in capabilities_from_arguments(
        ["a.wav", "--stage", "final-srt"]
    )
    assert preferred_capabilities_from_arguments(["a.wav"]) == ()
    assert preferred_capabilities_from_arguments(
        ["a.wav", "--stage", "final-srt"]
    ) == ("tokcount",)
    assert preferred_capabilities_from_arguments(
        ["a.wav", "--llm-correct-translate"]
    ) == ("tokcount",)


def _vendored(tmp_path: Path, monkeypatch) -> Path:
    """A stand-in for the _vendor tree the wheel build assembles.

    It no longer stages a manifest or a lock: since the desktop split those
    ship *inside* `finesub_bootstrap`, so the shell reads them from the package
    it already imports and the real ones are what these tests exercise. What
    `_VENDOR` still answers is where the vendored sources sit -- the runtime's
    `app_source`, and the `src` this process puts on PYTHONPATH.
    """

    vendor = tmp_path / "_vendor"
    (vendor / "src").mkdir(parents=True)
    monkeypatch.setattr(cli, "_VENDOR", vendor)
    return vendor


def test_the_cli_offers_every_manifest_resource_except_uv(
    tmp_path: Path, monkeypatch
) -> None:
    # uv arrives as a wheel dependency; everything else the desktop manages is
    # available to the CLI too, so the two agree on versions and hashes.
    vendor = _vendored(tmp_path, monkeypatch)
    monkeypatch.setenv("FINESUB_HOME", str(tmp_path / "home"))

    shell = cli._shell()

    assert set(shell.resources.resources) == {
        "ffmpeg",
        "git",
        "yt-dlp",
        "tokcount",
    }
    assert shell.can_provision
    assert shell.runtime.app_source == vendor.resolve()
    assert shell.ask_big_data_dir is cli.ask_big_data_dir


class _Stream:
    """A stdin/stdout stand-in whose interactivity the test decides."""

    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive
        self.written: list[str] = []

    def isatty(self) -> bool:
        return self._interactive

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        return


def _streams(monkeypatch, *, interactive: bool) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _Stream(interactive=interactive))
    monkeypatch.setattr(cli.sys, "stdout", _Stream(interactive=interactive))


def test_a_non_interactive_install_is_never_left_waiting(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`irm ... | iex`, CI and a redirected console all answer "the default".

    A prompt nobody can see would hang an install with no one at it.
    """

    _streams(monkeypatch, interactive=False)

    def refuse(*_args):
        raise AssertionError("a prompt was shown with no terminal to show it on")

    monkeypatch.setattr("builtins.input", refuse)

    assert cli.ask_big_data_dir(tmp_path / "default") is None
    assert str(tmp_path / "default") in capsys.readouterr().err


def test_an_empty_answer_means_the_default(tmp_path: Path, monkeypatch) -> None:
    _streams(monkeypatch, interactive=True)
    monkeypatch.setattr("builtins.input", lambda *_: "   ")

    assert cli.ask_big_data_dir(tmp_path / "default") is None


def test_a_typed_path_is_taken(tmp_path: Path, monkeypatch) -> None:
    _streams(monkeypatch, interactive=True)
    monkeypatch.setattr("builtins.input", lambda *_: f"  {tmp_path / 'chosen'}  ")

    assert cli.ask_big_data_dir(tmp_path / "default") == tmp_path / "chosen"


def test_a_closed_or_interrupted_prompt_falls_back_to_the_default(
    tmp_path: Path, monkeypatch
) -> None:
    _streams(monkeypatch, interactive=True)

    for error in (EOFError, KeyboardInterrupt):

        def raising(*_args, _error=error):
            raise _error

        monkeypatch.setattr("builtins.input", raising)
        assert cli.ask_big_data_dir(tmp_path / "default") is None
