from pathlib import Path

from desktop.backend.settings.store import SettingsStore


def test_missing_api_key_keeps_raw_srt_available(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)

    capabilities = store.get_capabilities()

    assert capabilities.raw_srt is True
    assert capabilities.translation is False
    assert store.validate_stage("raw-srt") is None


def test_translation_request_without_gemini_key_returns_structured_error(
    tmp_path: Path,
) -> None:
    store = SettingsStore(tmp_path)

    error = store.validate_stage("final-srt")

    assert error is not None
    assert error.code == "api_key_required"
    assert error.action == "open_settings"


def test_worker_environment_contains_saved_keys_without_exposing_them(
    tmp_path: Path,
) -> None:
    store = SettingsStore(tmp_path)
    store.save_api_keys(gemini="secret-g", exa="", tavily="secret-t")

    public = store.public_settings().model_dump(mode="json")
    worker_env = store.build_worker_env()

    assert public["api_keys"] == {
        "gemini": "configured",
        "exa": "missing",
        "tavily": "configured",
    }
    assert "secret-g" not in str(public)
    assert worker_env == {
        "GEMINI_API_KEY": "secret-g",
        "TAVILY_API_KEY": "secret-t",
    }


def test_api_keys_round_trip_through_private_env_file(tmp_path: Path) -> None:
    SettingsStore(tmp_path).save_api_keys(
        gemini="line-safe-key",
        exa="exa-key",
        tavily="",
    )

    reloaded = SettingsStore(tmp_path)

    assert reloaded.build_worker_env() == {
        "GEMINI_API_KEY": "line-safe-key",
        "EXA_API_KEY": "exa-key",
    }
    assert (tmp_path / ".env").is_file()


def test_delete_api_key_removes_only_selected_provider(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.save_api_keys(gemini="g", exa="e", tavily="t")

    store.delete_api_key("gemini")

    assert store.build_worker_env() == {
        "EXA_API_KEY": "e",
        "TAVILY_API_KEY": "t",
    }
