from pathlib import Path

import pytest

from desktop.backend.settings.store import SettingsStore
from llm.api_keys import (
    EXA_POOL,
    GEMINI_FREE_POOL,
    TAVILY_POOL,
    resolve_pool,
)


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
        "GEMINI_FREE": "secret-g",
        "TAVILY_KEYS": "secret-t",
    }


def test_api_keys_round_trip_through_private_env_file(tmp_path: Path) -> None:
    SettingsStore(tmp_path).save_api_keys(
        gemini="line-safe-key",
        exa="exa-key",
        tavily="",
    )

    reloaded = SettingsStore(tmp_path)

    assert reloaded.build_worker_env() == {
        "GEMINI_FREE": "line-safe-key",
        "EXA_KEYS": "exa-key",
    }
    assert (tmp_path / ".env").is_file()


def test_legacy_desktop_keys_are_migrated_once(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "GEMINI_API_KEY=legacy-gemini",
                "EXA_API_KEY=legacy-exa",
                "TAVILY_API_KEY=legacy-tavily",
                "",
            )
        ),
        encoding="utf-8",
    )

    worker_env = SettingsStore(tmp_path).build_worker_env()
    migrated = env_path.read_text(encoding="utf-8")

    assert worker_env == {
        "GEMINI_FREE": "legacy-gemini",
        "EXA_KEYS": "legacy-exa",
        "TAVILY_KEYS": "legacy-tavily",
    }
    assert "GEMINI_API_KEY" not in migrated
    assert "EXA_API_KEY" not in migrated
    assert "TAVILY_API_KEY" not in migrated


def test_saved_keys_are_visible_to_cli_provider_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("GEMINI_FREE", "EXA_KEYS", "TAVILY_KEYS"):
        monkeypatch.delenv(name, raising=False)
    store = SettingsStore(tmp_path)
    store.save_api_keys(
        gemini="gemini-key",
        exa="exa-key",
        tavily="tavily-key",
    )
    worker_env = store.build_worker_env()

    assert [
        entry.key
        for entry in resolve_pool(GEMINI_FREE_POOL, worker_env, config={})
    ] == ["gemini-key"]
    assert [
        entry.key
        for entry in resolve_pool(EXA_POOL, worker_env, config={})
    ] == ["exa-key"]
    assert [
        entry.key
        for entry in resolve_pool(TAVILY_POOL, worker_env, config={})
    ] == ["tavily-key"]


def test_delete_api_key_removes_only_selected_provider(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path)
    store.save_api_keys(gemini="g", exa="e", tavily="t")

    store.delete_api_key("gemini")

    assert store.build_worker_env() == {
        "EXA_KEYS": "e",
        "TAVILY_KEYS": "t",
    }
