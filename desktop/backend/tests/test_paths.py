from pathlib import Path

from finesub_bootstrap.paths import AppPaths


def test_app_paths_keep_mutable_data_outside_version(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path)

    assert paths.app_versions == tmp_path / "app" / "versions"
    assert paths.app_current == tmp_path / "app" / "current.json"
    assert paths.runtime == tmp_path / "runtime"
    assert paths.models == tmp_path / "models"
    assert paths.user_data == tmp_path / "user-data"
    assert paths.cache == tmp_path / "cache"


def test_app_paths_resolve_relative_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    paths = AppPaths.for_root(Path("install"))

    assert paths.root == (tmp_path / "install").resolve()
    assert paths.logs == paths.user_data / "logs"


def test_app_paths_relocate_only_personal_data(tmp_path: Path) -> None:
    # Installed copies keep settings/logs outside the disposable install dir;
    # rebuildable state (runtime, models, cache) must stay under the root.
    personal = tmp_path / "local-app-data" / "FineSub" / "user-data"

    paths = AppPaths.for_root(tmp_path / "install", user_data=personal)

    assert paths.user_data == personal.resolve()
    assert paths.logs == personal.resolve() / "logs"
    assert paths.runtime == (tmp_path / "install").resolve() / "runtime"
    assert paths.models == (tmp_path / "install").resolve() / "models"
    assert paths.cache == (tmp_path / "install").resolve() / "cache"
