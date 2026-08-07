from pathlib import Path

import pytest

from finesub_bootstrap.paths import (
    AppPaths,
    default_data_root,
    ensure_store,
    load_app_paths,
    recorded_big_data,
    resolve_big_data_root,
)


def test_a_fresh_install_is_self_contained_except_for_personal_data(
    tmp_path: Path,
) -> None:
    # Big, rebuildable data sits with the installation so deleting the folder
    # takes it along; the small, irreplaceable half lives in one shared place
    # so a user's knowledge base does not depend on which front end opened it.
    paths = AppPaths.for_root(tmp_path)

    assert paths.app_versions == tmp_path / "app" / "versions"
    assert paths.app_current == tmp_path / "app" / "current.json"
    assert paths.runtime == tmp_path / "runtime"
    assert paths.models == tmp_path / "models"
    assert paths.cache == tmp_path / "cache"
    assert paths.tasks == tmp_path / "tasks"
    assert paths.user_data == default_data_root() / "user-data"
    assert paths.logs == paths.user_data / "logs"


def test_app_paths_resolve_relative_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    paths = AppPaths.for_root(Path("install"))

    assert paths.root == (tmp_path / "install").resolve()
    assert paths.logs == paths.user_data / "logs"


def test_the_big_data_root_moves_the_big_three_and_nothing_else(
    tmp_path: Path,
) -> None:
    # runtime stays with the installation: it is version-bound, and it has to
    # share a volume with the download cache or their hardlinks break.
    paths = AppPaths.for_root(tmp_path / "install", big_data=tmp_path / "elsewhere")

    assert paths.models == (tmp_path / "elsewhere").resolve() / "models"
    assert paths.cache == (tmp_path / "elsewhere").resolve() / "cache"
    assert paths.tasks == (tmp_path / "elsewhere").resolve() / "tasks"
    assert paths.runtime == (tmp_path / "install").resolve() / "runtime"


def test_a_recorded_store_is_adopted_by_another_installation(
    tmp_path: Path,
) -> None:
    first = AppPaths.for_root(tmp_path / "first", data_root=tmp_path / "data")
    ensure_store(first)

    second = load_app_paths(tmp_path / "second", data_root=tmp_path / "data")

    assert second.big_data == first.big_data
    assert second.models == first.models
    # Only the shared three follow; the runtime stays private.
    assert second.runtime == (tmp_path / "second").resolve() / "runtime"


def test_a_store_that_was_deleted_falls_back_and_says_so(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    paths = AppPaths.for_root(tmp_path / "gone", data_root=data_root)
    ensure_store(paths)
    import shutil

    shutil.rmtree(paths.big_data)
    messages: list[str] = []

    resolved = resolve_big_data_root(
        tmp_path / "install", data_root, log=messages.append
    )

    assert resolved == (tmp_path / "install").resolve()
    assert any("register-location" in message for message in messages)
    # Resolving must not rewrite the record: the user may have moved that
    # folder and still has to re-register it.
    assert recorded_big_data(data_root) == paths.big_data


def test_an_offline_volume_falls_back_without_touching_the_record(
    tmp_path: Path, monkeypatch
) -> None:
    # A portable install on a removable disk that is simply unplugged must not
    # cause a several-GB re-download and a permanently rewritten record.
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "locations.json").write_text(
        '{"schemaVersion": 1, "bigData": "Z:\\\\FineSub"}', encoding="utf-8"
    )
    if Path("Z:/").exists():  # pragma: no cover - depends on the machine
        pytest.skip("this machine has a Z: drive")
    messages: list[str] = []

    resolved = resolve_big_data_root(
        tmp_path / "install", data_root, log=messages.append
    )

    assert resolved == (tmp_path / "install").resolve()
    assert any("磁盘当前不可用" in message for message in messages)
    assert recorded_big_data(data_root) == Path("Z:\\FineSub")


def test_ensure_store_leaves_a_self_describing_folder(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "install", data_root=tmp_path / "data")

    ensure_store(paths)

    assert (paths.big_data / ".finesub-store.json").is_file()
    # The script is what a user double-clicks after moving the folder by hand,
    # so it has to travel inside the folder it describes.
    script = paths.big_data / "register-location.cmd"
    assert script.is_file()
    # cmd.exe fails silently on both: LF-only endings execute fragments of the
    # next line, and non-ASCII is read in the console code page.
    body = script.read_bytes()
    assert b"\n" not in body.replace(b"\r\n", b"")
    assert body.decode("ascii")
    assert recorded_big_data(paths.data_root) == paths.big_data


def test_ensure_store_defers_to_a_store_registered_while_we_decided(
    tmp_path: Path,
) -> None:
    # Two first-time installs racing: adopting the winner's store beats leaving
    # several GB of duplicate downloads behind.
    data_root = tmp_path / "data"
    winner = AppPaths.for_root(tmp_path / "winner", data_root=data_root)
    ensure_store(winner)
    loser = AppPaths.for_root(tmp_path / "loser", data_root=data_root)

    ensure_store(loser)

    assert recorded_big_data(data_root) == winner.big_data
