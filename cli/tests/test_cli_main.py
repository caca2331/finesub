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
    for subcommand in ("setup", "doctor", "uninstall", "batch"):
        assert subcommand in output


def test_pipeline_arguments_are_forwarded_verbatim(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        cli,
        "_run_in_runtime",
        lambda module, arguments: calls.append((module, arguments)) or 0,
    )

    assert cli.main(["input.wav", "--language", "en", "--word"]) == 0
    assert calls == [
        ("asr_playground.pipeline", ["input.wav", "--language", "en", "--word"])
    ]


def test_batch_dispatches_to_the_batch_runner(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        cli,
        "_run_in_runtime",
        lambda module, arguments: calls.append((module, arguments)) or 0,
    )

    assert cli.main(["batch", "--manifest", "tasks.jsonl"]) == 0
    assert calls == [("asr_playground.batch", ["--manifest", "tasks.jsonl"])]


def test_uninstall_removes_rebuildable_state_and_keeps_personal_data(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    home = tmp_path / "FineSub"
    monkeypatch.setenv("FINESUB_HOME", str(home))
    for name in ("runtime", "models", "cache", "user-data"):
        (home / name).mkdir(parents=True)
        (home / name / "content.bin").write_bytes(b"data")

    assert cli.main(["uninstall"]) == 0

    assert not (home / "runtime").exists()
    assert not (home / "models").exists()
    assert not (home / "cache").exists()
    assert (home / "user-data" / "content.bin").is_file()
    assert "--purge-user-data" in capsys.readouterr().out


def test_uninstall_purges_personal_data_only_on_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "FineSub"
    monkeypatch.setenv("FINESUB_HOME", str(home))
    (home / "user-data").mkdir(parents=True)
    (home / "user-data" / ".env").write_text("GEMINI_FREE=key", "utf-8")

    assert cli.main(["uninstall", "--purge-user-data"]) == 0

    assert not home.exists()


def test_uninstall_rejects_unknown_options() -> None:
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
    _vendored(tmp_path, monkeypatch)

    _, resources, _ = cli._services(tmp_path / "home")

    assert set(resources.resources) == {"ffmpeg", "git", "yt-dlp"}


def test_a_system_tool_is_reported_as_ready_without_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    _vendored(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli,
        "_system_tool",
        lambda resource_id: found if resource_id == "ffmpeg" else None,
    )
    _, resources, _ = cli._services(tmp_path / "home")
    installed: list[str] = []
    monkeypatch.setattr(
        resources, "install", lambda *a, **k: installed.append(a[0])
    )

    cli._ensure_resource(resources, "ffmpeg", "reason")

    assert installed == []
    assert cli._tool_state(resources, "ffmpeg") == "ready"
    assert "system" in cli._tool_report(resources, "ffmpeg", "")


def test_on_demand_tools_install_only_when_the_command_needs_them(
    tmp_path: Path, monkeypatch
) -> None:
    _vendored(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_system_tool", lambda _resource_id: None)
    _, resources, _ = cli._services(tmp_path / "home")
    installed: list[str] = []
    monkeypatch.setattr(
        resources, "install", lambda *a, **k: installed.append(a[0])
    )

    cli._ensure_capabilities(resources, ["a.wav"])
    assert installed == []

    cli._ensure_capabilities(resources, ["https://example.test/v"])
    assert installed == ["yt-dlp"]
