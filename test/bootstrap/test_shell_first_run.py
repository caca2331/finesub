"""Where the big files land, decided once and before anything downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub_bootstrap.environment import RuntimeEnvironment
from finesub_bootstrap.paths import AppPaths, recorded_big_data
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap.shell import Shell


def _shell(tmp_path: Path, *, ask=None, can_provision: bool = True) -> Shell:
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
        ask_big_data_dir=ask,
    )


def test_a_fresh_machine_is_asked_once(tmp_path: Path) -> None:
    chosen = tmp_path / "D" / "FineSub"
    asked: list[Path] = []

    def ask(default_root: Path) -> Path:
        asked.append(default_root)
        return chosen

    shell = _shell(tmp_path, ask=ask)
    shell.settle_big_data_location()

    assert asked == [tmp_path / "root"]
    assert shell.paths.big_data == chosen.resolve()
    assert recorded_big_data(shell.paths.data_root) == chosen.resolve()
    # Recorded before any download: the marker is what makes the location real.
    assert (chosen / ".finesub-store.json").is_file()


def test_the_store_is_registered_before_anything_is_downloaded(
    tmp_path: Path,
) -> None:
    chosen = tmp_path / "D" / "FineSub"
    seen: list[Path | None] = []

    def ask(default_root: Path) -> Path:
        # Nothing may be recorded while the question is still open.
        seen.append(recorded_big_data(AppPaths.for_root(tmp_path / "root").data_root))
        return chosen

    shell = _shell(tmp_path, ask=ask)
    shell.settle_big_data_location()

    assert seen == [None]
    assert recorded_big_data(shell.paths.data_root) == chosen.resolve()


def test_an_empty_answer_keeps_the_default_and_settles_it(tmp_path: Path) -> None:
    """Accepting the default is an answer; recording it is what ends the asking."""

    shell = _shell(tmp_path, ask=lambda root: None)
    shell.settle_big_data_location()

    assert shell.paths.big_data == (tmp_path / "root").resolve()
    assert recorded_big_data(shell.paths.data_root) == (tmp_path / "root").resolve()
    assert shell.is_new_install() is False


def test_a_front_end_without_a_prompt_is_never_asked(tmp_path: Path) -> None:
    """The desktop package shares this class and answers this elsewhere."""

    shell = _shell(tmp_path, ask=None, can_provision=False)
    shell.settle_big_data_location()

    assert shell.paths.big_data == (tmp_path / "root").resolve()


def test_an_existing_record_ends_the_question(tmp_path: Path) -> None:
    first = _shell(tmp_path, ask=lambda root: tmp_path / "D" / "FineSub")
    first.settle_big_data_location()

    asked: list[Path] = []
    second = _shell(tmp_path, ask=lambda root: asked.append(root) or None)
    second.settle_big_data_location()

    assert asked == []


def test_a_provisioned_machine_with_a_lost_record_is_not_asked(
    tmp_path: Path,
) -> None:
    """`runtime` present means this was set up before; a missing record is damage.

    Asking then would offer to place data that is already placed, and the
    answer would silently orphan whatever is on disk.
    """

    (tmp_path / "root" / "runtime").mkdir(parents=True)
    asked: list[Path] = []
    shell = _shell(tmp_path, ask=lambda root: asked.append(root) or None)

    assert shell.is_new_install() is False
    shell.settle_big_data_location()
    assert asked == []


def test_existing_big_data_is_not_asked_about(tmp_path: Path) -> None:
    (tmp_path / "root" / "models").mkdir(parents=True)
    asked: list[Path] = []
    shell = _shell(tmp_path, ask=lambda root: asked.append(root) or None)

    assert shell.is_new_install() is False
    shell.settle_big_data_location()
    assert asked == []


def test_the_environment_answers_without_a_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = tmp_path / "E" / "FineSub"
    monkeypatch.setenv("FINESUB_BIG_DATA_DIR", str(chosen))
    asked: list[Path] = []
    shell = _shell(tmp_path, ask=lambda root: asked.append(root) or None)

    shell.settle_big_data_location()

    assert asked == [], "an answer already given must not be asked for"
    assert shell.paths.big_data == chosen.resolve()


def test_an_explicit_directory_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINESUB_BIG_DATA_DIR", str(tmp_path / "E"))
    shell = _shell(tmp_path)

    shell.settle_big_data_location(tmp_path / "F")

    assert shell.paths.big_data == (tmp_path / "F").resolve()


def test_a_bad_destination_is_refused_rather_than_used(tmp_path: Path) -> None:
    shell = _shell(tmp_path)

    with pytest.raises(ValueError):
        # Inside the installation is refused for the same reason `relocate`
        # refuses it.
        shell.settle_big_data_location(tmp_path / "root" / "inside")


def test_a_relative_path_is_refused(tmp_path: Path) -> None:
    """`resolve()` anchors a relative path to the current directory.

    The absolute-path check used to sit after it, so it could never fire and
    `finesub relocate data` moved several GB into wherever the user stood.
    """

    shell = _shell(tmp_path)

    with pytest.raises(ValueError, match="绝对路径"):
        shell.settle_big_data_location(Path("somewhere"))

    with pytest.raises(ValueError, match="绝对路径"):
        shell._checked_destination(Path("data"), force=False)


def test_data_dir_on_an_installed_machine_points_at_relocate(
    tmp_path: Path,
) -> None:
    """Silently moving several GB from a setup command is not an option."""

    first = _shell(tmp_path, ask=lambda root: None)
    first.settle_big_data_location()

    second = _shell(tmp_path)
    with pytest.raises(ValueError, match="relocate"):
        second.settle_big_data_location(tmp_path / "G")


def test_dirs_only_settles_the_location_and_downloads_nothing(
    tmp_path: Path, capsys
) -> None:
    provisioned: list[str] = []
    shell = _shell(tmp_path, ask=lambda root: tmp_path / "D" / "FineSub")
    shell.ensure_ready = lambda: provisioned.append("ensure_ready")

    assert shell.setup(["--dirs-only"]) == 0

    assert provisioned == []
    assert recorded_big_data(shell.paths.data_root) == (
        tmp_path / "D" / "FineSub"
    ).resolve()


def test_setup_takes_the_directory_as_an_argument(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    shell.ensure_ready = lambda: None

    assert shell.setup(["--dirs-only", "--data-dir", str(tmp_path / "H")]) == 0
    assert shell.paths.big_data == (tmp_path / "H").resolve()


def test_setup_rejects_unknown_options(tmp_path: Path) -> None:
    shell = _shell(tmp_path)

    assert shell.setup(["--nope"]) == 2
    assert shell.setup(["--data-dir"]) == 2
