from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from finesub_bootstrap.capabilities import capabilities_from_arguments
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
    for subcommand in ("setup", "doctor", "keys", "uninstall", "batch"):
        assert subcommand in output


def test_commands_go_to_the_shared_shell(monkeypatch) -> None:
    # Dispatch itself is shared with the desktop package (test_shell.py); this
    # front end only has to hand it the arguments untouched.
    calls: list[list[str]] = []
    monkeypatch.setattr(
        cli,
        "_shell",
        lambda: SimpleNamespace(
            dispatch=lambda arguments: calls.append(list(arguments)) or 0
        ),
    )

    assert cli.main(["input.wav", "--language", "en", "--word"]) == 0
    assert calls == [["input.wav", "--language", "en", "--word"]]


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
    for directory in (paths.runtime, paths.models, paths.cache, paths.tasks):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "content.bin").write_bytes(b"data")
    paths.user_data.mkdir(parents=True, exist_ok=True)
    (paths.user_data / ".env").write_text("GEMINI_FREE=key", "utf-8")

    assert cli.main(["uninstall"]) == 0

    assert not paths.runtime.exists()
    assert not paths.models.exists()
    assert not paths.cache.exists()
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

    assert cli.main(["uninstall"]) == 0

    assert (shared.models / "weights.bin").is_file()
    assert "--purge-big-data" in capsys.readouterr().out


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
    paths = SimpleNamespace(user_data=user_data, cache=tmp_path / "cache")

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
    assert capabilities_from_arguments(["a.wav"]) == ()
    # The knowledge update runs inside the correction stage, so asking for it on
    # a plain transcription needs no git -- nothing is going to run.
    assert capabilities_from_arguments(["a.wav", "--knowledge", "update"]) == ()
    assert capabilities_from_arguments(
        ["a.wav", "--knowledge", "update", "--stage", "final-srt"]
    ) == ("git",)
    assert capabilities_from_arguments(
        ["a.wav", "--knowledge=update", "--stage=translated-srt"]
    ) == ("git",)
    assert capabilities_from_arguments(
        ["a.wav", "--knowledge=update", "--llm-correct-translate"]
    ) == ("git",)
    assert capabilities_from_arguments(
        ["a.wav", "--knowledge", "collect", "--stage", "final-srt"]
    ) == ()
    assert capabilities_from_arguments(["https://example.test/v"]) == ("yt-dlp",)
    assert capabilities_from_arguments(
        ["https://example.test/v", "--knowledge=update", "--stage=final-srt"]
    ) == ("git", "yt-dlp")


def _vendored(tmp_path: Path, monkeypatch) -> Path:
    """A stand-in for the _vendor tree the wheel build assembles."""

    vendor = tmp_path / "_vendor"
    vendor.mkdir()
    manifest = (
        Path(__file__).resolve().parents[2]
        / "desktop"
        / "resources"
        / "runtime-manifest.json"
    )
    (vendor / "runtime-manifest.json").write_text(
        manifest.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (vendor / "pylock.win-py312.toml").write_text("", encoding="utf-8")
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

    assert set(shell.resources.resources) == {"ffmpeg", "git", "yt-dlp"}
    assert shell.can_provision
    assert shell.runtime.app_source == vendor.resolve()
