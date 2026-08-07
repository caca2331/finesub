from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub_bootstrap import shell as shell_module
from finesub_bootstrap.environment import RuntimeEnvironment
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap.shell import Shell, package_shell
from finesub_bootstrap.system_tools import SystemTool


def _shell(tmp_path: Path, *, can_provision: bool = True) -> Shell:
    paths = AppPaths.for_root(tmp_path / "root")
    return Shell(
        paths=paths,
        resources=ResourceManager(paths, []),
        runtime=RuntimeEnvironment(
            paths=paths,
            app_source=tmp_path / "source",
            runtime_lock=tmp_path / "source" / "pylock.win-py312.toml",
            uv_executable=lambda: tmp_path / "uv.exe",
        ),
        can_provision=can_provision,
    )


def test_pipeline_arguments_are_forwarded_verbatim(tmp_path: Path) -> None:
    # The shell must not have an opinion about pipeline flags.
    shell = _shell(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append((module, list(arguments))) or 0
    )

    assert shell.dispatch(["input.wav", "--language", "en", "--word"]) == 0
    assert calls == [
        ("asr_playground.pipeline", ["input.wav", "--language", "en", "--word"])
    ]


def test_batch_dispatches_to_the_batch_runner(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append((module, list(arguments))) or 0
    )

    assert shell.dispatch(["batch", "--manifest", "tasks.jsonl"]) == 0
    assert calls == [("asr_playground.batch", ["--manifest", "tasks.jsonl"])]


def test_keys_is_a_shell_command_not_a_pipeline_argument(
    tmp_path: Path, capsys
) -> None:
    # Unregistered commands fall through to the pipeline; a typo'd "keys"
    # would otherwise be sent there as an input file.
    shell = _shell(tmp_path)
    shell.run_in_runtime = lambda *arguments: pytest.fail(
        "keys must not reach the pipeline"
    )

    assert shell.dispatch(["keys"]) == 0
    assert "尚未配置任何 API key" in capsys.readouterr().out


def test_keys_masks_by_default_and_reveals_on_request(
    tmp_path: Path, capsys
) -> None:
    shell = _shell(tmp_path)
    env_path = shell.paths.user_data / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        'GEMINI_FREE={"main":"AIzaVeryLongSecret123"}\n', encoding="utf-8"
    )

    assert shell.keys([]) == 0
    masked_output = capsys.readouterr().out
    assert "AIzaVeryLongSecret123" not in masked_output
    assert "main=AIza…t123" in masked_output

    assert shell.keys(["--reveal"]) == 0
    revealed = capsys.readouterr().out
    assert 'GEMINI_FREE={"main":"AIzaVeryLongSecret123"}' in revealed

    assert shell.keys(["--bogus"]) == 2


def test_a_system_tool_is_reported_as_ready_without_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    monkeypatch.setattr(
        shell_module,
        "system_tool",
        lambda resource_id: found if resource_id == "ffmpeg" else None,
    )
    shell = _shell(tmp_path)
    installed: list[str] = []
    monkeypatch.setattr(
        shell.resources, "install", lambda *a, **k: installed.append(a[0])
    )

    shell._ensure_resource("ffmpeg", "reason")

    assert installed == []
    assert shell._tool_state("ffmpeg") == "ready"
    assert "system" in shell._tool_report("ffmpeg", "")


def test_on_demand_tools_install_only_when_the_command_needs_them(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(shell_module, "system_tool", lambda _resource_id: None)
    shell = _shell(tmp_path)
    paths = shell.paths
    shell.resources = ResourceManager(
        paths,
        shell_module.resource_specs(
            json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "resources"
                    / "runtime-manifest.json"
                ).read_text(encoding="utf-8")
            ),
            exclude=("uv",),
        ),
    )
    installed: list[str] = []
    monkeypatch.setattr(
        shell.resources, "install", lambda *a, **k: installed.append(a[0])
    )

    shell._ensure_capabilities(["a.wav"])
    assert installed == []

    shell._ensure_capabilities(["https://example.test/v"])
    assert installed == ["yt-dlp"]


def test_a_shell_that_cannot_provision_sends_the_user_to_the_app(
    tmp_path: Path, monkeypatch
) -> None:
    # The packaged command line runs *on* the managed interpreter, so it cannot
    # be what installs or replaces it -- and a dead-end "run setup" would be
    # worse than saying where setup actually lives.
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    monkeypatch.setattr(shell_module, "system_tool", lambda _resource_id: found)
    monkeypatch.setattr(shell_module.os, "name", "nt")
    shell = _shell(tmp_path, can_provision=False)
    monkeypatch.setattr(
        shell.resources,
        "install",
        lambda *a, **k: pytest.fail("a packaged shell must not provision"),
    )
    monkeypatch.setattr(
        shell.runtime,
        "install",
        lambda *a, **k: pytest.fail("a packaged shell must not provision"),
    )

    with pytest.raises(SystemExit, match="FineSub Desktop"):
        shell.ensure_ready()


def test_relocate_moves_the_big_three_and_leaves_the_runtime(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import ensure_store, recorded_big_data

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.models.mkdir(parents=True, exist_ok=True)
    (shell.paths.models / "weights.bin").write_bytes(b"data")
    shell.paths.runtime.mkdir(parents=True, exist_ok=True)
    (shell.paths.runtime / "python").mkdir()

    assert shell.relocate([str(tmp_path / "elsewhere")]) == 0

    moved = (tmp_path / "elsewhere").resolve()
    assert (moved / "models" / "weights.bin").is_file()
    assert not (tmp_path / "root" / "models").exists()
    # The runtime is version-bound and hardlinks out of the download cache, so
    # it stays where the application is.
    assert (tmp_path / "root" / "runtime" / "python").is_dir()
    assert recorded_big_data(shell.paths.data_root) == moved
    assert (moved / ".finesub-store.json").is_file()
    assert (moved / "register-location.cmd").is_file()


def test_relocate_reset_brings_the_data_back_to_the_installation(
    tmp_path: Path,
) -> None:
    # The install root is the one destination that legitimately contains the
    # runtime, so the "do not swallow the runtime" guard has to exempt it --
    # otherwise --reset can never succeed.
    from finesub_bootstrap.paths import ensure_store, recorded_big_data

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.models.mkdir(parents=True, exist_ok=True)
    (shell.paths.models / "weights.bin").write_bytes(b"data")
    shell.paths.runtime.mkdir(parents=True, exist_ok=True)
    assert shell.relocate([str(tmp_path / "elsewhere")]) == 0

    assert shell.relocate(["--reset"]) == 0

    home = (tmp_path / "root").resolve()
    assert (home / "models" / "weights.bin").is_file()
    assert not (tmp_path / "elsewhere" / "models").exists()
    assert recorded_big_data(shell.paths.data_root) == home


def test_relocate_refuses_a_destination_that_is_not_ours(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    occupied = tmp_path / "someone-elses"
    occupied.mkdir()
    (occupied / "important.txt").write_text("do not touch", encoding="utf-8")

    assert shell.relocate([str(occupied)]) == 2
    assert (occupied / "important.txt").is_file()


def test_relocate_waits_for_a_running_task(tmp_path: Path) -> None:
    from finesub_bootstrap.locks import holding_lock
    from finesub_bootstrap.paths import ensure_store

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.tasks.mkdir(parents=True, exist_ok=True)

    with holding_lock(shell.paths.tasks / ".active.lock"):
        assert shell.relocate([str(tmp_path / "elsewhere")]) == 1

    assert not (tmp_path / "elsewhere").exists()


def test_relocate_adopts_an_existing_store_without_copying(tmp_path: Path) -> None:
    from finesub_bootstrap.paths import AppPaths, ensure_store, recorded_big_data

    shell = _shell(tmp_path)
    existing = AppPaths.for_root(
        tmp_path / "shared", data_root=shell.paths.data_root
    )
    ensure_store(existing)
    existing.models.mkdir(parents=True, exist_ok=True)
    (existing.models / "weights.bin").write_bytes(b"already here")

    assert shell.relocate([str(existing.big_data)]) == 0

    assert (existing.models / "weights.bin").read_bytes() == b"already here"
    assert recorded_big_data(shell.paths.data_root) == existing.big_data


def _packaged_install(root: Path, version: str = "2.3.4") -> Path:
    source = root / "app" / "versions" / version
    (source / "src" / "asr_playground").mkdir(parents=True)
    (source / "src" / "asr_playground" / "pipeline.py").write_text(
        "PIPELINE = True\n", "utf-8"
    )
    (source / "desktop" / "resources").mkdir(parents=True)
    (source / "desktop" / "resources" / "runtime-manifest.json").write_text(
        json.dumps({"resources": []}), "utf-8"
    )
    (source / "desktop" / "runtime").mkdir(parents=True)
    (source / "desktop" / "runtime" / "pylock.win-py312.toml").write_text(
        'lock-version = "1.0"\n', "utf-8"
    )
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "app" / "current.json").write_text(
        json.dumps({"current": version}), "utf-8"
    )
    return source


def test_a_packaged_shell_drives_the_install_it_sits_in(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "FineSub-portable"
    source = _packaged_install(root)
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    shell = package_shell(root)

    assert shell.paths.root == root.resolve()
    assert shell.runtime.app_source == source.resolve()
    assert shell.paths.runtime == (root / "runtime").resolve()
    # Personal data is shared with every other front end; the big, rebuildable
    # half stays with this installation.
    assert shell.paths.user_data == (
        local_app_data / "FineSub" / "user-data"
    ).resolve()
    assert shell.paths.models == (root / "models").resolve()
    assert not shell.can_provision


def test_a_packaged_shell_adopts_a_registered_store(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub_bootstrap.paths import AppPaths, ensure_store

    root = tmp_path / "FineSub-portable"
    _packaged_install(root)
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    elsewhere = AppPaths.for_root(
        tmp_path / "shared", data_root=(local_app_data / "FineSub")
    )
    ensure_store(elsewhere)

    shell = package_shell(root)

    assert shell.paths.models == elsewhere.models
    assert shell.paths.runtime == (root / "runtime").resolve()
