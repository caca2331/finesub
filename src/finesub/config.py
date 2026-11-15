"""``config.toml``: the settings shared by every front end.

The file is optional, hand-editable and resolved by
:func:`finesub.paths.resolve_config_file` (source checkout first, then
the packaged user-data root -- the same order as ``.env``). It holds what a
user may legitimately want to change *and* what more than one front end reads,
so the CLI and the pipeline worker see one answer.

This module owns only the generic half -- locating, parsing and memoizing the
document. Scalar settings are declared and checked in ``finesub.settings``;
structured tables are validated by the domain that owns them
(``finesub.api_keys`` for ``[providers]`` / ``[pools]``,
``model_routes`` for the presets under ``[llm]``). Unknown tables are ignored on
purpose, so a newer front end's settings do not break an older reader.

Purely stdlib, and it must stay that way: ``finesub.speech`` reads it
and may not grow an ``llm`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any, Dict, Tuple

from .paths import resolve_config_file


@dataclass(frozen=True)
class _CachedConfig:
    path: Path | None
    signature: Tuple[int, int] | None
    data: dict


# Parsed configs, keyed by everything ``resolve_config_file`` looks at. Every
# LLM call asks whether its provider is enabled and then which keys the pool
# selects, so an uncached read means locating the checkout root and re-parsing
# the TOML twice per call.
_CONFIG_CACHE: Dict[Tuple[str, str, str, str], _CachedConfig] = {}


def _config_cache_key(path: str | Path | None) -> Tuple[str, str, str, str]:
    return (
        str(path) if path is not None else "",
        os.environ.get("FINESUB_CONFIG_FILE", ""),
        os.environ.get("FINESUB_ROOT", ""),
        os.getcwd(),
    )


def _stat_signature(path: Path | None) -> Tuple[int, int] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def clear_config_cache() -> None:
    """Drop every memoized config (tests, and anything creating config.toml)."""

    _CONFIG_CACHE.clear()


def read_config_with_path(
    path: str | Path | None = None,
) -> Tuple[dict[str, Any], Path | None]:
    """Parsed config plus the file it came from (``None`` when there is none).

    A cached entry is re-read once the file's mtime/size move, so editing
    ``config.toml`` mid-run still takes effect -- including edits made by hand
    while a run is in flight. A *missing* config caches the empty
    result until :func:`clear_config_cache`; creating the file mid-run is not
    worth a stat of every candidate root on every call.

    The returned mapping is the cached object -- callers must not mutate it.
    """

    cache_key = _config_cache_key(path)
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None and _stat_signature(cached.path) == cached.signature:
        return cached.data, cached.path

    config_path = resolve_config_file(path)
    data: dict[str, Any] = {}
    if config_path is not None:
        try:
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(
                f"Invalid FineSub config TOML at {config_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"FineSub config must be a TOML table: {config_path}")
    _CONFIG_CACHE[cache_key] = _CachedConfig(
        path=config_path,
        signature=_stat_signature(config_path),
        data=data,
    )
    return data, config_path


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    """The parsed config alone; see :func:`read_config_with_path`."""

    return read_config_with_path(path)[0]
