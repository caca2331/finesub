"""settings.json (front-end state) and the config.toml half of the panel."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from finesub import config as app_config
from desktop.backend.common.models import SharedSettings
from desktop.backend.settings.preferences import PreferencesStore
from desktop.backend.settings.store import SettingsStore


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    app_config.clear_config_cache()
    yield
    app_config.clear_config_cache()


# ------------------------------------------------------------ preferences

def test_absent_file_reads_as_all_defaults(tmp_path) -> None:
    store = PreferencesStore(tmp_path)

    loaded = store.load()

    assert loaded.ui == {}
    assert loaded.task_defaults.model_dump(exclude_none=True) == {}


def test_only_what_was_chosen_is_stored(tmp_path) -> None:
    store = PreferencesStore(tmp_path)

    store.save(task_defaults={"gpu_budget_gb": 8}, ui={"theme": "dark"})

    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    # Sparse: everything not chosen stays absent, so improving a code default
    # still reaches this user.
    assert written["task_defaults"] == {"gpu_budget_gb": 8}
    assert written["ui"] == {"theme": "dark"}
    assert written["schema"] == 1


def test_saving_one_section_leaves_the_other_alone(tmp_path) -> None:
    store = PreferencesStore(tmp_path)
    store.save(ui={"theme": "dark"}, task_defaults={"gpu_budget_gb": 8})

    store.save(task_defaults={"language": "ja"})

    loaded = store.load()
    assert loaded.ui == {"theme": "dark"}
    assert loaded.task_defaults.gpu_budget_gb == 8
    assert loaded.task_defaults.language == "ja"


def test_null_resets_a_setting_instead_of_writing_a_default(tmp_path) -> None:
    store = PreferencesStore(tmp_path)
    store.save(task_defaults={"gpu_budget_gb": 8}, ui={"theme": "dark"})

    store.save(task_defaults={"gpu_budget_gb": None}, ui={"theme": None})

    written = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert written["task_defaults"] == {}
    assert written["ui"] == {}


def test_leftover_state_from_another_version_is_dropped_silently(tmp_path) -> None:
    # Program-written state: an unknown or invalid key is not something the
    # user can fix, so it must not cost them the rest of the file.
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "ui": {"theme": "dark"},
                "task_defaults": {
                    "gpu_budget_gb": 8,
                    "retired_option": "whatever",
                    "device": "quantum",
                },
            }
        ),
        encoding="utf-8",
    )
    store = PreferencesStore(tmp_path)

    loaded = store.load()

    assert loaded.task_defaults.gpu_budget_gb == 8
    assert loaded.task_defaults.device is None
    assert loaded.ui == {"theme": "dark"}


def test_a_damaged_file_does_not_stop_the_app(tmp_path) -> None:
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")

    assert PreferencesStore(tmp_path).load().ui == {}


def test_the_subtitle_knob_is_not_a_front_end_preference(tmp_path) -> None:
    # It is shared with the CLI, so config.toml owns it. One key, one home.
    store = PreferencesStore(tmp_path)

    store.save(task_defaults={"split_length_scale": 0.8})

    assert store.load().task_defaults.model_dump(exclude_none=True) == {}


def test_overlapping_saves_do_not_drop_each_other(tmp_path) -> None:
    # Bridge calls arrive on their own threads and the front end does not await
    # its saves, so two can overlap. Unlocked, both read the old file and the
    # later writer wins outright -- the earlier field simply vanishes.
    store = PreferencesStore(tmp_path)
    start = threading.Barrier(2)
    errors: list[BaseException] = []

    # Widen the read-modify-write window so the race is deterministic rather
    # than a matter of scheduling luck: with the lock the second thread waits
    # its turn, without it both read the same empty file.
    real_load = store.load

    def slow_load():
        loaded = real_load()
        time.sleep(0.15)
        return loaded

    store.load = slow_load  # type: ignore[method-assign]

    def save(patch):
        try:
            start.wait(timeout=5)
            store.save(**patch)
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [
        threading.Thread(target=save, args=({"ui": {"language": "en"}},)),
        threading.Thread(target=save, args=({"task_defaults": {"gpu_budget_gb": 8}},)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert errors == []
    store.load = real_load  # type: ignore[method-assign]
    loaded = store.load()
    assert loaded.ui == {"language": "en"}
    assert loaded.task_defaults.gpu_budget_gb == 8


# --------------------------------------------------------- shared settings

def test_shared_setting_round_trips_through_config_toml(tmp_path) -> None:
    store = SettingsStore(tmp_path)

    saved = store.save_shared_settings(SharedSettings(split_length_scale=0.8))

    assert saved.split_length_scale == 0.8
    assert store.shared_settings().split_length_scale == 0.8
    assert "length_scale = 0.8" in Path(store.config_path).read_text(encoding="utf-8")


def test_clearing_a_shared_setting_removes_the_line(tmp_path) -> None:
    store = SettingsStore(tmp_path)
    store.save_shared_settings(SharedSettings(split_length_scale=0.8))

    store.save_shared_settings(SharedSettings(split_length_scale=None))

    assert store.shared_settings().split_length_scale is None
    assert "length_scale" not in Path(store.config_path).read_text(encoding="utf-8")


def test_a_hand_written_comment_survives_a_panel_write(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "# keys I care about\n[providers]\ntavily = false\n", encoding="utf-8"
    )
    store = SettingsStore(tmp_path)

    store.save_shared_settings(SharedSettings(split_length_scale=1.2))

    text = path.read_text(encoding="utf-8")
    assert text.startswith("# keys I care about\n[providers]\ntavily = false\n")
    assert "length_scale = 1.2" in text


def test_an_out_of_range_value_is_refused_before_it_reaches_the_file(
    tmp_path,
) -> None:
    store = SettingsStore(tmp_path)

    with pytest.raises(ValueError, match="length scale"):
        store.save_shared_settings(SharedSettings(split_length_scale=4.0))

    assert not Path(store.config_path).exists()


def test_a_hand_edit_made_while_the_app_runs_is_not_reverted(tmp_path) -> None:
    store = SettingsStore(tmp_path)
    store.save_shared_settings(SharedSettings(split_length_scale=0.8))
    store.shared_settings()  # the panel has read (and cached) the old state

    path = Path(store.config_path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\n[providers]\ntavily = false\n",
        encoding="utf-8",
    )
    store.save_shared_settings(SharedSettings(split_length_scale=0.9))

    text = path.read_text(encoding="utf-8")
    assert "tavily = false" in text
    assert "length_scale = 0.9" in text
