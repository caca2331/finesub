"""``config.toml``: the settings shared by every front end.

The file is optional, hand-editable and resolved by
:func:`finesub.paths.resolve_config_file` (source checkout first, then
the packaged user-data root -- the same order as ``.env``). It holds what a
user may legitimately want to change *and* what more than one front end reads,
so the CLI and the pipeline worker see one answer.

This module owns only the generic half -- locating, parsing and memoizing the
document. Each domain validates its own table: ``finesub.llm.routing.api_keys`` for
``[providers]`` / ``[pools]``, the recognition stage for ``[segmentation]``.
Unknown tables are ignored on purpose, so a newer front end's settings do not
break an older reader.

Purely stdlib, and it must stay that way: ``finesub.speech`` reads it
and may not grow an ``llm`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tomllib
from typing import Any, Dict, Mapping, Tuple

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


# --------------------------------------------------------- setting bounds
#
# The accepted range of a shared setting is part of the file's contract, so it
# lives with the reader rather than with the algorithm that consumes the value:
# a settings writer (the desktop had one) has to reject a bad value at write
# time -- the only moment it can tell the user *why* -- and it cannot import
# the speech stack to find out, since that pulls in torch. What the value then
# *does* stays in
# ``speech.postprocessing.segmentation``.
SPLIT_LENGTH_SCALE_MIN = 0.6
SPLIT_LENGTH_SCALE_MAX = 1.6


def validate_split_length_scale(value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not (
        SPLIT_LENGTH_SCALE_MIN <= number <= SPLIT_LENGTH_SCALE_MAX
    ):
        raise ValueError(
            "segment split length scale must be within "
            f"[{SPLIT_LENGTH_SCALE_MIN}, {SPLIT_LENGTH_SCALE_MAX}], got {value!r}"
        )
    return number


def config_float(
    section: str,
    key: str,
    *,
    path: str | Path | None = None,
) -> float | None:
    """One numeric setting, or ``None`` when it is not in the file.

    Absent means "follow the code default", which is what keeps the file sparse
    enough to stay hand-editable: a setting only appears once someone actually
    chose it, and defaults stay changeable for everyone who never did.
    """

    data, config_path = read_config_with_path(path)
    table = data.get(section)
    if table is None:
        return None
    location = f" in {config_path}" if config_path else ""
    if not isinstance(table, Mapping):
        raise ValueError(f"[{section}] must be a TOML table{location}")
    if key not in table:
        return None
    value = table[key]
    # bool is an int subclass, and `length_scale = true` is a typo, not a 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be a number{location}")
    return float(value)


def config_str(
    section: str,
    key: str,
    *,
    path: str | Path | None = None,
) -> str | None:
    """One string setting, or ``None`` when it is not in the file.

    Same contract as :func:`config_float` and :func:`config_bool`: absent means
    "follow the code default". A present-but-blank value is *not* absent -- it
    is somebody writing "" on purpose, and the domain that owns the key decides
    what that means.
    """

    data, config_path = read_config_with_path(path)
    table = data.get(section)
    if table is None:
        return None
    location = f" in {config_path}" if config_path else ""
    if not isinstance(table, Mapping):
        raise ValueError(f"[{section}] must be a TOML table{location}")
    if key not in table:
        return None
    value = table[key]
    if not isinstance(value, str):
        raise ValueError(f"{section}.{key} must be a string{location}")
    return value


def config_bool(
    section: str,
    key: str,
    *,
    path: str | Path | None = None,
) -> bool | None:
    """One boolean setting, or ``None`` when it is not in the file.

    Same contract as :func:`config_float`: absent means "follow the code
    default", so the file stays sparse and a default stays changeable for
    everyone who never wrote it down.

    Only a real TOML boolean counts. `1` / `"true"` are rejected rather than
    coerced -- a config file that half-works is worse than one that says what
    is wrong, and this is the layer that decides whether a switch is on.
    """

    data, config_path = read_config_with_path(path)
    table = data.get(section)
    if table is None:
        return None
    location = f" in {config_path}" if config_path else ""
    if not isinstance(table, Mapping):
        raise ValueError(f"[{section}] must be a TOML table{location}")
    if key not in table:
        return None
    value = table[key]
    if not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be true or false{location}")
    return value
