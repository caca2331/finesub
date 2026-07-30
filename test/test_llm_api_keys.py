from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm import api_keys


ENV_MAP = {
    "GEMINI_FREE": "{main:key-main,spare:key-spare,third:key-third}",
    "GEMINI_PAID": "{paid-a:key-paid-a,paid-b:key-paid-b,paid-c:key-paid-c}",
    "EXA_KEYS": "{exa-a:key-exa-a,exa-b:key-exa-b,exa-c:key-exa-c,exa-d:key-exa-d}",
    "TAVILY_KEYS": "{tv-a:key-tv-a,tv-b:key-tv-b,tv-c:key-tv-c,tv-d:key-tv-d}",
}


@pytest.fixture(autouse=True)
def _clear_pool_warnings(monkeypatch) -> None:
    api_keys._WARNED_OVERSIZED_POOLS.clear()
    api_keys.clear_config_cache()
    for name in ("GEMINI_FREE", "GEMINI_PAID", "EXA_KEYS", "TAVILY_KEYS"):
        monkeypatch.delenv(name, raising=False)


def _names(entries: list[api_keys.ApiKeyEntry]) -> list[str]:
    return [entry.name for entry in entries]


def test_empty_or_missing_pool_uses_provider_default_size() -> None:
    config = {"pools": {"gemini_free": [], "exa": []}}

    assert _names(api_keys.resolve_pool("gemini_free", ENV_MAP, config=config)) == [
        "main",
        "spare",
    ]
    assert _names(api_keys.resolve_pool("exa", ENV_MAP, config=config)) == [
        "exa-a",
        "exa-b",
        "exa-c",
    ]
    assert _names(api_keys.resolve_pool("tavily", ENV_MAP, config=config)) == [
        "tv-a",
        "tv-b",
        "tv-c",
    ]


def test_paid_defaults_to_all_and_explicit_pool_reorders_without_warning(capsys) -> None:
    assert _names(api_keys.resolve_pool("gemini_paid", ENV_MAP, config={})) == [
        "paid-a",
        "paid-b",
        "paid-c",
    ]

    selected = api_keys.resolve_pool(
        "gemini_paid",
        ENV_MAP,
        config={"pools": {"gemini_paid": ["paid-c", "paid-a", "paid-b"]}},
    )

    assert _names(selected) == ["paid-c", "paid-a", "paid-b"]
    assert capsys.readouterr().err == ""


def test_explicit_oversized_pool_warns_once_without_truncating_or_leaking_keys(
    capsys,
) -> None:
    config = {"pools": {"gemini_free": ["third", "main", "spare"]}}

    first = api_keys.resolve_pool("gemini_free", ENV_MAP, config=config)
    second = api_keys.resolve_pool("gemini_free", ENV_MAP, config=config)

    assert _names(first) == ["third", "main", "spare"]
    assert second == first
    stderr = capsys.readouterr().err
    assert stderr.count("Warning:") == 1
    assert "recommended maximum is 2" in stderr
    assert "key-main" not in stderr
    assert "key-spare" not in stderr
    assert "key-third" not in stderr


def test_explicit_pool_deduplicates_names_and_rejects_unknown_names() -> None:
    selected = api_keys.resolve_pool(
        "gemini_free",
        ENV_MAP,
        config={"pools": {"gemini_free": ["spare", "spare", "main"]}},
    )
    assert _names(selected) == ["spare", "main"]

    with pytest.raises(ValueError, match="unknown key name.*missing"):
        api_keys.resolve_pool(
            "gemini_free",
            ENV_MAP,
            config={"pools": {"gemini_free": ["main", "missing"]}},
        )


def test_disabled_provider_has_no_pool_and_paid_can_be_first_gemini() -> None:
    config = {
        "providers": {"gemini_free": False, "gemini_paid": True},
        "pools": {"gemini_paid": ["paid-b", "paid-a"]},
    }

    assert api_keys.resolve_pool("gemini_free", ENV_MAP, config=config) == []
    entry, tier = api_keys.first_enabled_gemini_entry(ENV_MAP, config=config)
    assert tier == "GEMINI_PAID"
    assert entry.name == "paid-b"


def test_config_validation_rejects_unknown_provider_and_non_boolean_flag() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        api_keys.provider_enabled("gemini_free", config={"providers": {"gemni": False}})
    with pytest.raises(ValueError, match="must be true or false"):
        api_keys.provider_enabled(
            "gemini_free", config={"providers": {"gemini_free": "no"}}
        )


def test_read_config_toml_from_configured_path(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        """
[providers]
tavily = false

[pools]
gemini_free = ["spare", "main"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config_path))

    loaded = api_keys.read_config()

    assert loaded["providers"]["tavily"] is False
    assert loaded["pools"]["gemini_free"] == ["spare", "main"]


def test_read_config_is_cached_until_the_file_changes(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom.toml"
    config_path.write_text("[providers]\ntavily = false\n", encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config_path))

    reads = {"count": 0}
    real_resolve = api_keys.resolve_config_file

    def counting_resolve(path=None):
        reads["count"] += 1
        return real_resolve(path)

    monkeypatch.setattr(api_keys, "resolve_config_file", counting_resolve)

    assert api_keys.read_config()["providers"]["tavily"] is False
    assert api_keys.read_config()["providers"]["tavily"] is False
    assert reads["count"] == 1

    # An edit must still be picked up: the cache revalidates on mtime/size.
    config_path.write_text("[providers]\ntavily = true\n", encoding="utf-8")
    os.utime(config_path, (0, 0))

    assert api_keys.read_config()["providers"]["tavily"] is True
    assert reads["count"] == 2


def test_read_config_cache_does_not_leak_across_configured_paths(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text("[providers]\nexa = false\n", encoding="utf-8")
    second.write_text("[providers]\nexa = true\n", encoding="utf-8")

    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(first))
    assert api_keys.read_config()["providers"]["exa"] is False
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(second))
    assert api_keys.read_config()["providers"]["exa"] is True


def test_tracked_example_config_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]

    loaded = api_keys.read_config(root / "config.example.toml")

    assert loaded["providers"]["gemini_free"] is True
    assert loaded["pools"]["gemini_free"] == ["main", "spare"]
