"""Named API-key stores, provider switches, and pool selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping

from finesub.config import clear_config_cache, read_config_with_path
from finesub.reporting import current_reporter


GEMINI_FREE_POOL = "gemini_free"
GEMINI_PAID_POOL = "gemini_paid"
EXA_POOL = "exa"
TAVILY_POOL = "tavily"

GEMMA4_GROUNDED_PROVIDER = "gemma4_grounded"


@dataclass(frozen=True)
class PoolSpec:
    env_name: str
    recommended_max: int | None


@dataclass(frozen=True)
class ApiKeyEntry:
    name: str
    key: str
    named: bool = True

    @property
    def key_id(self) -> str:
        if self.named:
            return self.name
        digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

    @property
    def label(self) -> str:
        if self.named:
            return self.name
        return self.key[-6:] if len(self.key) >= 6 else self.key


POOL_SPECS: Mapping[str, PoolSpec] = {
    GEMINI_FREE_POOL: PoolSpec("GEMINI_FREE", 2),
    GEMINI_PAID_POOL: PoolSpec("GEMINI_PAID", None),
    EXA_POOL: PoolSpec("EXA_KEYS", 3),
    TAVILY_POOL: PoolSpec("TAVILY_KEYS", 3),
}

PROVIDER_NAMES = frozenset(
    {
        *POOL_SPECS,
        GEMMA4_GROUNDED_PROVIDER,
    }
)

TIER_TO_POOL: Mapping[str, str] = {
    "GEMINI_FREE": GEMINI_FREE_POOL,
    "GEMINI_PAID": GEMINI_PAID_POOL,
}

_WARNED_OVERSIZED_POOLS: set[tuple[str, tuple[str, ...]]] = set()


class ProviderUnavailableError(RuntimeError):
    """A disabled provider or one without selected credentials."""


def parse_key_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    if not value:
        return []
    return [
        item.strip().strip('"').strip("'")
        for item in value.split(",")
        if item.strip()
    ]


def parse_key_map(value: str) -> list[tuple[str, str]]:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    if not value:
        return []
    pairs: list[tuple[str, str]] = []
    for item in value.split(","):
        if not item.strip() or ":" not in item:
            continue
        name, key = item.split(":", 1)
        name = name.strip().strip('"').strip("'")
        key = key.strip().strip('"').strip("'")
        if name and key:
            pairs.append((name, key))
    return pairs


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    """The parsed FineSub config, validated for the tables this module owns.

    Locating, parsing and memoizing live in ``finesub.config`` (the
    recognition stage reads the same file for ``[segmentation]`` and cannot
    import this package). Validation runs per call rather than per load: it is
    a loop over a handful of provider and pool names, and the alternative --
    caching *validatedness* next to a cache this module does not own -- would
    let a config error stay hidden until something else happened to reload it.

    The returned mapping is the cached object -- callers must not mutate it.
    """

    data, config_path = read_config_with_path(path)
    if data:
        _validate_config(data, config_path)
    return data


def _validate_config(config: Mapping[str, Any], path: Path | None = None) -> None:
    location = f" in {path}" if path else ""
    providers = config.get("providers", {})
    pools = config.get("pools", {})
    if not isinstance(providers, Mapping):
        raise ValueError(f"[providers] must be a TOML table{location}")
    if not isinstance(pools, Mapping):
        raise ValueError(f"[pools] must be a TOML table{location}")

    unknown_providers = sorted(set(providers) - PROVIDER_NAMES)
    unknown_pools = sorted(set(pools) - set(POOL_SPECS))
    if unknown_providers:
        raise ValueError(
            f"Unknown provider name(s){location}: {', '.join(unknown_providers)}"
        )
    if unknown_pools:
        raise ValueError(f"Unknown pool name(s){location}: {', '.join(unknown_pools)}")

    for name, enabled in providers.items():
        if not isinstance(enabled, bool):
            raise ValueError(f"providers.{name} must be true or false{location}")
    for name, selector in pools.items():
        if not isinstance(selector, list) or any(
            not isinstance(item, str) for item in selector
        ):
            raise ValueError(f"pools.{name} must be an array of key names{location}")


def _loaded_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Resolve a config argument, validating only what the caller supplied.

    ``read_config`` already validated (and memoized) what it returns, so the
    internal handoffs below must not pay for it again.
    """

    if config is None:
        return read_config()
    _validate_config(config)
    return config


def provider_enabled(
    provider: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> bool:
    if provider not in PROVIDER_NAMES:
        raise ValueError(f"Unknown provider name: {provider}")
    return _provider_enabled(provider, _loaded_config(config))


def _provider_enabled(provider: str, loaded: Mapping[str, Any]) -> bool:
    providers = loaded.get("providers", {})
    return bool(providers.get(provider, True))


def pool_name_for_tier(provider_tier: str) -> str | None:
    return TIER_TO_POOL.get(provider_tier)


def provider_tier_enabled(
    provider_tier: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> bool:
    pool_name = pool_name_for_tier(provider_tier)
    return True if pool_name is None else provider_enabled(pool_name, config=config)


def _raw_key_entries(pool_name: str, env_map: Mapping[str, str]) -> list[ApiKeyEntry]:
    spec = POOL_SPECS[pool_name]
    raw = os.getenv(spec.env_name)
    if raw is None:
        raw = str(env_map.get(spec.env_name, "") or "")
    pairs = parse_key_map(raw)
    if pairs:
        entries: list[ApiKeyEntry] = []
        seen_names: set[str] = set()
        for name, key in pairs:
            if name in seen_names:
                continue
            seen_names.add(name)
            entries.append(ApiKeyEntry(name=name, key=key))
        return entries
    return [
        ApiKeyEntry(name="", key=key, named=False)
        for key in parse_key_list(raw)
    ]


def resolve_pool(
    pool_name: str,
    env_map: Mapping[str, str],
    *,
    config: Mapping[str, Any] | None = None,
) -> list[ApiKeyEntry]:
    if pool_name not in POOL_SPECS:
        raise ValueError(f"Unknown API key pool: {pool_name}")
    loaded = _loaded_config(config)
    if not _provider_enabled(pool_name, loaded):
        return []

    entries = _raw_key_entries(pool_name, env_map)
    selector = loaded.get("pools", {}).get(pool_name)
    spec = POOL_SPECS[pool_name]
    if not selector:
        if spec.recommended_max is None:
            return entries
        return entries[: spec.recommended_max]

    by_name = {entry.name: entry for entry in entries if entry.named}
    selected: list[ApiKeyEntry] = []
    seen: set[str] = set()
    missing: list[str] = []
    for name in selector:
        if name in seen:
            continue
        seen.add(name)
        entry = by_name.get(name)
        if entry is None:
            missing.append(name)
        else:
            selected.append(entry)
    if missing:
        raise ValueError(
            f"Pool {pool_name!r} references unknown key name(s): {', '.join(missing)}"
        )

    if spec.recommended_max is not None and len(selected) > spec.recommended_max:
        warning_key = (pool_name, tuple(entry.name for entry in selected))
        if warning_key not in _WARNED_OVERSIZED_POOLS:
            _WARNED_OVERSIZED_POOLS.add(warning_key)
            current_reporter().warning(
                "key-pool-oversized",
                f"API key pool {pool_name!r} selects {len(selected)} keys; the "
                f"recommended maximum is {spec.recommended_max}",
                impact="过大的池可能触发供应商风控",
            )
    return selected


def resolve_tier_pool(
    provider_tier: str,
    env_map: Mapping[str, str],
    *,
    config: Mapping[str, Any] | None = None,
) -> list[ApiKeyEntry]:
    pool_name = pool_name_for_tier(provider_tier)
    if pool_name is None:
        return []
    return resolve_pool(pool_name, env_map, config=config)


def first_enabled_gemini_entry(
    env_map: Mapping[str, str],
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[ApiKeyEntry, str]:
    loaded = _loaded_config(config)
    for tier in ("GEMINI_FREE", "GEMINI_PAID"):
        entries = resolve_tier_pool(tier, env_map, config=loaded)
        if entries:
            return entries[0], tier
    raise RuntimeError(
        "No enabled Gemini provider has an API key. Configure GEMINI_FREE or "
        "GEMINI_PAID in .env and enable it in config.toml."
    )
