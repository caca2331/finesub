from __future__ import annotations

import pytest

from finesub import config as app_config


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    app_config.clear_config_cache()
    yield
    app_config.clear_config_cache()


def _write(tmp_path, monkeypatch, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    return path


def test_missing_file_reads_as_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "absent.toml"))

    assert app_config.read_config() == {}
    assert app_config.config_float("segmentation", "length_scale") is None


def test_absent_table_or_key_means_follow_the_code_default(
    tmp_path, monkeypatch
) -> None:
    _write(tmp_path, monkeypatch, "[providers]\ntavily = false\n")

    assert app_config.config_float("segmentation", "length_scale") is None

    _write(tmp_path, monkeypatch, "[segmentation]\n")
    app_config.clear_config_cache()

    assert app_config.config_float("segmentation", "length_scale") is None


def test_numeric_settings_read_as_float(tmp_path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "[segmentation]\nlength_scale = 0.85\nwhole = 1\n")

    assert app_config.config_float("segmentation", "length_scale") == 0.85
    assert app_config.config_float("segmentation", "whole") == 1.0


@pytest.mark.parametrize("literal", ['"0.85"', "true"])
def test_non_numeric_setting_is_an_error_naming_the_file(
    tmp_path, monkeypatch, literal
) -> None:
    # `length_scale = true` is a typo, not a 1.0 -- bool is an int subclass, so
    # this needs its own guard.
    path = _write(tmp_path, monkeypatch, f"[segmentation]\nlength_scale = {literal}\n")

    with pytest.raises(ValueError, match=r"segmentation\.length_scale must be a number"):
        app_config.config_float("segmentation", "length_scale")
    with pytest.raises(ValueError, match=str(path.name)):
        app_config.config_float("segmentation", "length_scale")


def test_section_that_is_not_a_table_is_an_error(tmp_path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, 'segmentation = "nope"\n')

    with pytest.raises(ValueError, match=r"\[segmentation\] must be a TOML table"):
        app_config.config_float("segmentation", "length_scale")


def test_malformed_toml_names_the_file(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "[segmentation\n")

    with pytest.raises(ValueError, match="Invalid FineSub config TOML"):
        app_config.read_config()
    assert path.exists()
