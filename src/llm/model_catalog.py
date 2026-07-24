"""Pipe-delimited model/provider-tier facts for LLM harness reporting.

``tpm`` and ``tpd`` are input-token limits only (output/thinking excluded).
``rpd``/``tpd`` are informational; runtime daily caps use ``.state`` exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List


CATALOG_FILENAME = "model_catalog.psv"
CATALOG_COLUMNS = (
    "provider_tier",
    "model",
    "litellm_model",
    "max_input_tokens",
    "max_output_tokens",
    "supports_video_audio",
    "supports_native_search",
    "supports_reasoning",
    "rpm",
    "tpm",
    "rpd",
    "tpd",
    "is_free",
    "capability",
)


@dataclass(frozen=True)
class ModelCatalogEntry:
    provider_tier: str
    model: str
    litellm_model: str
    max_input_tokens: int
    max_output_tokens: int
    supports_video_audio: bool
    supports_native_search: bool
    supports_reasoning: bool
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    is_free: bool
    capability: int


def _catalog_path() -> Path:
    return Path(__file__).resolve().with_name(CATALOG_FILENAME)


def _parse_bool(value: str, *, field: str, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{CATALOG_FILENAME}:{line_number}: {field} must be true/false")


def _parse_int(value: str, *, field: str, line_number: int) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: {field} must be an integer"
        ) from exc


def _entry_from_row(row: Dict[str, str], *, line_number: int) -> ModelCatalogEntry:
    for key in CATALOG_COLUMNS:
        if not row.get(key, "").strip():
            raise ValueError(f"{CATALOG_FILENAME}:{line_number}: missing {key}")
    return ModelCatalogEntry(
        provider_tier=row["provider_tier"].strip(),
        model=row["model"].strip(),
        litellm_model=row["litellm_model"].strip(),
        max_input_tokens=_parse_int(
            row["max_input_tokens"], field="max_input_tokens", line_number=line_number
        ),
        max_output_tokens=_parse_int(
            row["max_output_tokens"], field="max_output_tokens", line_number=line_number
        ),
        supports_video_audio=_parse_bool(
            row["supports_video_audio"],
            field="supports_video_audio",
            line_number=line_number,
        ),
        supports_native_search=_parse_bool(
            row["supports_native_search"],
            field="supports_native_search",
            line_number=line_number,
        ),
        supports_reasoning=_parse_bool(
            row["supports_reasoning"], field="supports_reasoning", line_number=line_number
        ),
        rpm=_parse_int(row["rpm"], field="rpm", line_number=line_number),
        tpm=_parse_int(row["tpm"], field="tpm", line_number=line_number),
        rpd=_parse_int(row["rpd"], field="rpd", line_number=line_number),
        tpd=_parse_int(row["tpd"], field="tpd", line_number=line_number),
        is_free=_parse_bool(row["is_free"], field="is_free", line_number=line_number),
        capability=_parse_int(row["capability"], field="capability", line_number=line_number),
    )


def load_model_catalog(path: str | Path | None = None) -> List[ModelCatalogEntry]:
    catalog_path = Path(path).expanduser() if path is not None else _catalog_path()
    lines = catalog_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{catalog_path} is empty")
    columns = tuple(part.strip() for part in lines[0].split("|"))
    if columns != CATALOG_COLUMNS:
        raise ValueError(
            f"{catalog_path}: header must be {'|'.join(CATALOG_COLUMNS)}"
        )
    entries: List[ModelCatalogEntry] = []
    for line_number, raw in enumerate(lines[1:], start=2):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != len(CATALOG_COLUMNS):
            raise ValueError(
                f"{catalog_path}:{line_number}: expected {len(CATALOG_COLUMNS)} fields"
            )
        entries.append(
            _entry_from_row(dict(zip(CATALOG_COLUMNS, parts)), line_number=line_number)
        )
    return entries


@lru_cache(maxsize=1)
def default_model_catalog() -> tuple[ModelCatalogEntry, ...]:
    return tuple(load_model_catalog())


def catalog_by_litellm_model(
    entries: Iterable[ModelCatalogEntry] | None = None,
) -> Dict[str, ModelCatalogEntry]:
    result: Dict[str, ModelCatalogEntry] = {}
    for entry in entries or default_model_catalog():
        if entry.litellm_model not in result:
            result[entry.litellm_model] = entry
    return result


def get_model_catalog_entry(litellm_model: str) -> ModelCatalogEntry | None:
    return catalog_by_litellm_model().get(litellm_model)


def get_model_catalog_entry_for_tier(
    litellm_model: str,
    provider_tier: str,
) -> ModelCatalogEntry | None:
    for entry in default_model_catalog():
        if entry.litellm_model == litellm_model and entry.provider_tier == provider_tier:
            return entry
    return None


def provider_tier_for_model(litellm_model: str, fallback: str = "") -> str:
    entry = get_model_catalog_entry(litellm_model)
    return entry.provider_tier if entry is not None else fallback


def supports_reasoning(litellm_model: str, default: bool = True) -> bool:
    entry = get_model_catalog_entry(litellm_model)
    return default if entry is None else entry.supports_reasoning
