"""Pipe-delimited model facts: one row per callable (provider, model).

The catalog is **the** place a model's facts live -- capabilities, limits, the
endpoint that serves it and the thinking dialect it speaks -- while
``model_routes.toml`` composes them into model groups, task groups and
presets. Facts vs composition, one file each (owner decision 2026-08-12).

Two files, same format, merged by ``fact_id`` with the later one winning
(:func:`load_merged_catalog`): the **packaged default** that ships with the
code, and an optional **override** in the data root. Installed front ends put
that override in ``user-data``; a source checkout has no separate user-data, so
it is simply the checkout root -- exactly the resolution ``.env`` and
``config.toml`` already use, and it stays in the paths layer rather than here.

``tpm`` and ``tpd`` are input-token limits only (output/thinking excluded).
``rpd``/``tpd`` are informational; runtime daily caps use ``.state`` exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CATALOG_FILENAME = "model_catalog.psv"
# Columns a row must carry: everything else has a defensible default, and a
# short header is what makes a hand-written override file practical.
# ``max_input_tokens`` is required on purpose -- windows are planned against
# it, so an optimistic default explodes every window far from its cause.
REQUIRED_COLUMNS = (
    "fact_id",
    "provider_tier",
    "api_model_id",
    "max_input_tokens",
)
CATALOG_COLUMNS = (
    "fact_id",
    "provider_tier",
    "provider_kind",
    "base_url",
    "key_env",
    "display_name",
    "api_model_id",
    "max_input_tokens",
    "max_output_tokens",
    "supports_audio",
    "supports_video",
    "video_high_resolution_only",
    "supports_native_search",
    "thinking",
    "token_scale",
    "rpm",
    "tpm",
    "rpd",
    "tpd",
    "is_free",
    "quality_score",
    "quota_pool",
)
# Names an older override file may still carry. They are rejected with the new
# name rather than silently ignored: a dropped `api_model_id` column would only
# surface as "this model is not in the catalog", far from the header that caused
# it. No compatibility shim -- an override file is a handful of lines a person
# wrote, and the message says exactly what to type.
RETIRED_COLUMNS = {
    "litellm_model": "api_model_id",
    "model": "display_name",
}

# Blank cell defaults (owner decision 2026-08-12: "允许某些项留空，留空时取
# 默认值"). The conservative generic RPM default is 100; provider-specific
# packaged rows continue to declare their measured/published quotas explicitly.
GEMINI_KIND = "gemini"
LOCAL_AGENT_KIND = "local_agent"
# A worker somebody is already running. It gets a catalog row for the same
# reason every other target does -- the harness has to know what it can be
# asked to do (audio? video? search?) and whether it clears the cell's quality
# floor -- but nothing here ever launches it.
CONVERSATIONAL_KIND = "conversational_agent"
OPENAI_COMPAT_KIND = "openai_compat"
ANTHROPIC_KIND = "anthropic"
PROVIDER_KINDS = (
    GEMINI_KIND,
    LOCAL_AGENT_KIND,
    CONVERSATIONAL_KIND,
    OPENAI_COMPAT_KIND,
    ANTHROPIC_KIND,
)
# A row that names no dialect speaks the one most third-party endpoints do --
# unless its provider tier is one the harness ships a transport for, in which
# case overriding a packaged row (say, to re-tune a limit) must not force the
# user to restate the dialect.
DEFAULT_PROVIDER_KIND = OPENAI_COMPAT_KIND
PACKAGED_PROVIDER_KINDS = {
    "GEMINI_FREE": GEMINI_KIND,
    "GEMINI_PAID": GEMINI_KIND,
    "LOCAL_CODEX": LOCAL_AGENT_KIND,
    "LOCAL_CLAUDE": LOCAL_AGENT_KIND,
    "LOCAL_AGY": LOCAL_AGENT_KIND,
    "LOCAL_DSH": LOCAL_AGENT_KIND,
    "LOCAL_CONVERSATIONAL": CONVERSATIONAL_KIND,
}
DEFAULT_MAX_OUTPUT_TOKENS = 65_536
DEFAULT_RPM = 100
DEFAULT_TPM = 4_000_000
DEFAULT_QUALITY_SCORE = 50

# ``quality_score`` is advisory only (plan D5): it feeds the task-group floor
# warning and the artifacts, and never enters any routing decision.
# ``correction_prompt_tier`` is gone (plan v2 D2): variant ownership lives on
# the task-group cell (and optional model-group entry overrides), not on the
# model row.
QUALITY_SCORE_RANGE = range(0, 101)

# ``thinking`` column (owner design 2026-08-11, identity default 2026-08-12):
#
# - ``true`` (and an omitted user-model field) -- the **identity mapping**: the
#   harness's abstract high/medium/low goes out verbatim. All three dialects
#   happen to spell the levels the same way (Gemini ``thinkingLevel``,
#   OpenAI-compatible ``reasoning_effort``, Anthropic ``output_config.effort``),
#   so identity is the right default rather than a lucky coincidence per row.
# - ``false`` -- the model takes no thinking parameter at all (self-hosted
#   endpoints that 400 on unknown fields declare this).
# - three comma-separated provider values -- an explicit override mapping the
#   abstract high, medium, low requests, in that order (e.g. a provider whose
#   words are ``max,default,minimal``).
THINKING_DISABLED = "false"
THINKING_IDENTITY = "true"
IDENTITY_THINKING_LEVELS = ("high", "medium", "low")
_ABSTRACT_THINKING_INDEX = {"high": 0, "medium": 1, "low": 2}


def parse_thinking_spec(value: str, *, owner: str) -> tuple[str, str, str] | None:
    normalized = (value or "").strip()
    if normalized.lower() == THINKING_DISABLED:
        return None
    if normalized.lower() == THINKING_IDENTITY:
        return IDENTITY_THINKING_LEVELS
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            f"{owner}: thinking must be true (identity), false (no thinking "
            "parameter) or three comma-separated values mapping high,medium,low"
        )
    return (parts[0], parts[1], parts[2])


def thinking_value_for(entry: "ModelCatalogEntry", level: str | None) -> str:
    """The provider value this model uses for an abstract thinking level.

    ``""`` means "send no thinking parameter" -- either the model declared
    ``none`` or the caller asked for the model default.
    """

    if entry.thinking_levels is None:
        return ""
    index = _ABSTRACT_THINKING_INDEX.get((level or "").strip().lower())
    if index is None:
        return ""
    return entry.thinking_levels[index]


@dataclass(frozen=True)
class ModelCatalogEntry:
    fact_id: str
    provider_tier: str
    # What a human calls this row (artifacts, warnings); defaults to the API id.
    display_name: str
    # What the endpoint is called by name in the request.
    api_model_id: str
    max_input_tokens: int
    max_output_tokens: int
    supports_audio: bool
    supports_video: bool
    # True when the transport cannot request Gemini's low media-resolution
    # tier. Planning must use the most expensive member in the routed group.
    video_high_resolution_only: bool
    supports_native_search: bool
    # None = the model takes no thinking parameter; otherwise provider values
    # for the harness's (high, medium, low) requests.
    thinking_levels: tuple[str, str, str] | None
    rpm: int
    tpm: int
    rpd: int
    tpd: int
    is_free: bool
    quality_score: int
    # Which API dialect serves this row, and where. ``gemini`` / ``local_agent``
    # are the packaged kinds and need no URL; the two text dialects do.
    provider_kind: str = DEFAULT_PROVIDER_KIND
    base_url: str = ""
    # Env name holding this provider's key. Empty means "the packaged pool for
    # this tier" (Gemini) or the ``FINESUB_KEY_<TIER>`` convention.
    key_env: str = ""
    # Local-estimate correction factor (plan v2 D14). The 3-tier counter uses
    # Gemini's vocabulary; another model's tokenizer differs, so estimates
    # carry a systematic bias. Multiplied into every local estimate, never into
    # reported usage -- reports and calibration read the API's own numbers.
    token_scale: float = 1.0
    # Which allowance this row draws on when it runs out. Empty means "the
    # provider tier", which is right wherever one subscription backs every
    # model on a tier -- one Codex plan, one Claude plan. It is *not* right for
    # Antigravity, where the Gemini and Opus models are metered separately, and
    # booking their exhaustion together would take a working model out of
    # service for hours because its neighbour ran dry.
    quota_pool: str = ""

    # Whether this row came from the override file rather than the packaged
    # default. Artifacts label those numbers as user-typed, not measured.
    self_reported: bool = False

    @property
    def effective_quota_pool(self) -> str:
        return self.quota_pool or self.provider_tier


def _catalog_path() -> Path:
    return Path(__file__).resolve().with_name(CATALOG_FILENAME)


def _parse_bool(value: str, *, field: str, line_number: int, default: bool) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{CATALOG_FILENAME}:{line_number}: {field} must be true/false")


def _parse_int(value: str, *, field: str, line_number: int, default: int | None = None) -> int:
    text = value.strip()
    if not text and default is not None:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: {field} must be an integer"
        ) from exc


def _parse_float(value: str, *, field: str, line_number: int, default: float) -> float:
    text = value.strip()
    if not text:
        return default
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: {field} must be a number"
        ) from exc
    if not number > 0:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: {field} must be positive"
        )
    return number


def _entry_from_row(
    row: Dict[str, str], *, line_number: int, self_reported: bool
) -> ModelCatalogEntry:
    for key in REQUIRED_COLUMNS:
        if not row.get(key, "").strip():
            raise ValueError(f"{CATALOG_FILENAME}:{line_number}: missing {key}")
    quality_score = _parse_int(
        row.get("quality_score", ""),
        field="quality_score",
        line_number=line_number,
        default=DEFAULT_QUALITY_SCORE,
    )
    if quality_score not in QUALITY_SCORE_RANGE:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: quality_score must be in "
            f"[{QUALITY_SCORE_RANGE.start}, {QUALITY_SCORE_RANGE[-1]}]"
        )
    provider_tier = row["provider_tier"].strip()
    provider_kind = row.get("provider_kind", "").strip().lower() or (
        PACKAGED_PROVIDER_KINDS.get(provider_tier, DEFAULT_PROVIDER_KIND)
    )
    if provider_kind not in PROVIDER_KINDS:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: provider_kind must be one of "
            f"{sorted(PROVIDER_KINDS)}"
        )
    base_url = row.get("base_url", "").strip().rstrip("/")
    if provider_kind in (OPENAI_COMPAT_KIND, ANTHROPIC_KIND) and not base_url:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: base_url is required for "
            f"provider_kind={provider_kind}"
        )
    if provider_kind in (GEMINI_KIND, LOCAL_AGENT_KIND, CONVERSATIONAL_KIND) and base_url:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: provider_kind={provider_kind} "
            "takes no base_url (the packaged transports own their endpoint)"
        )
    supports_video = _parse_bool(
        row.get("supports_video", ""),
        field="supports_video",
        line_number=line_number,
        default=False,
    )
    video_high_resolution_only = _parse_bool(
        row.get("video_high_resolution_only", ""),
        field="video_high_resolution_only",
        line_number=line_number,
        default=False,
    )
    if video_high_resolution_only and not supports_video:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: "
            "video_high_resolution_only=true requires supports_video=true"
        )
    # Custom HTTP media remains unsupported: the two text transports flatten
    # messages to plain strings, so a row claiming media on one of them would
    # pass capability filtering and then have its clip
    # silently dropped -- the exact silent degradation the design forbids.
    # Declare it here rather than trusting the writer to know.
    if provider_kind in (OPENAI_COMPAT_KIND, ANTHROPIC_KIND):
        for field_name in ("supports_audio", "supports_video"):
            if _parse_bool(
                row.get(field_name, ""),
                field=field_name,
                line_number=line_number,
                default=False,
            ):
                raise ValueError(
                    f"{CATALOG_FILENAME}:{line_number}: {field_name}=true is not "
                    f"available on provider_kind={provider_kind} (custom HTTP "
                    "media is unsupported; use --correction-media text to "
                    "put a text model on media-bearing material)"
                )
    max_input = _parse_int(
        row["max_input_tokens"], field="max_input_tokens", line_number=line_number
    )
    if max_input <= 0:
        raise ValueError(
            f"{CATALOG_FILENAME}:{line_number}: max_input_tokens must be positive"
        )
    api_model_id = row["api_model_id"].strip()
    return ModelCatalogEntry(
        fact_id=row["fact_id"].strip(),
        provider_tier=provider_tier,
        display_name=row.get("display_name", "").strip() or api_model_id,
        api_model_id=api_model_id,
        max_input_tokens=max_input,
        max_output_tokens=_parse_int(
            row.get("max_output_tokens", ""),
            field="max_output_tokens",
            line_number=line_number,
            default=DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        supports_audio=_parse_bool(
            row.get("supports_audio", ""),
            field="supports_audio",
            line_number=line_number,
            default=False,
        ),
        supports_video=supports_video,
        video_high_resolution_only=video_high_resolution_only,
        supports_native_search=_parse_bool(
            row.get("supports_native_search", ""),
            field="supports_native_search",
            line_number=line_number,
            default=False,
        ),
        thinking_levels=parse_thinking_spec(
            row.get("thinking", "") or THINKING_IDENTITY,
            owner=f"{CATALOG_FILENAME}:{line_number}",
        ),
        rpm=_parse_int(
            row.get("rpm", ""), field="rpm", line_number=line_number, default=DEFAULT_RPM
        ),
        tpm=_parse_int(
            row.get("tpm", ""), field="tpm", line_number=line_number, default=DEFAULT_TPM
        ),
        rpd=_parse_int(
            row.get("rpd", ""), field="rpd", line_number=line_number, default=-1
        ),
        tpd=_parse_int(
            row.get("tpd", ""), field="tpd", line_number=line_number, default=-1
        ),
        is_free=_parse_bool(
            row.get("is_free", ""),
            field="is_free",
            line_number=line_number,
            default=False,
        ),
        quality_score=quality_score,
        quota_pool=row.get("quota_pool", "").strip(),
        provider_kind=provider_kind,
        base_url=base_url,
        key_env=row.get("key_env", "").strip(),
        token_scale=_parse_float(
            row.get("token_scale", ""),
            field="token_scale",
            line_number=line_number,
            default=1.0,
        ),
        self_reported=self_reported,
    )


def load_model_catalog(
    path: str | Path | None = None, *, self_reported: bool = False
) -> List[ModelCatalogEntry]:
    """Parse one catalog file.

    The header declares which columns the file carries: an override file may
    list just the four required ones plus whatever it wants to state, and
    blank cells inside a declared column fall back to the same defaults.
    Unknown column names are rejected rather than ignored -- a typo in a
    header is otherwise a silently missing fact.
    """

    catalog_path = Path(path).expanduser() if path is not None else _catalog_path()
    lines = catalog_path.read_text(encoding="utf-8").splitlines()
    # Comments and blanks are skipped below the header, so they are skipped
    # above it too: an override file is meant to be hand-written, and the
    # first thing a person writes is a line saying what the file is for.
    header_index = next(
        (
            index
            for index, raw in enumerate(lines)
            if raw.strip() and not raw.strip().startswith("#")
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"{catalog_path} is empty")
    columns = tuple(part.strip() for part in lines[header_index].split("|"))
    retired = [name for name in columns if name in RETIRED_COLUMNS]
    if retired:
        pairs = ", ".join(f"{name} -> {RETIRED_COLUMNS[name]}" for name in retired)
        raise ValueError(
            f"{catalog_path}: the header uses column names this version retired "
            f"({pairs}). Rename them in the header row; the cells underneath are "
            "unchanged."
        )
    unknown = [name for name in columns if name not in CATALOG_COLUMNS]
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if unknown or missing:
        raise ValueError(
            f"{catalog_path}: header must use known columns "
            f"({'|'.join(CATALOG_COLUMNS)}) and include {list(REQUIRED_COLUMNS)}; "
            f"unknown={unknown}, missing={missing}"
        )
    entries: List[ModelCatalogEntry] = []
    for line_number, raw in enumerate(
        lines[header_index + 1 :], start=header_index + 2
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != len(columns):
            raise ValueError(
                f"{catalog_path}:{line_number}: expected {len(columns)} fields"
            )
        entries.append(
            _entry_from_row(
                dict(zip(columns, parts)),
                line_number=line_number,
                self_reported=self_reported,
            )
        )
    _reject_duplicates(entries, catalog_path)
    return entries


def _reject_duplicates(
    entries: Sequence[ModelCatalogEntry], where: Any
) -> None:
    fact_ids: set[str] = set()
    endpoint_keys: set[tuple[str, str]] = set()
    for entry in entries:
        if entry.fact_id in fact_ids:
            raise ValueError(f"{where}: duplicate fact_id {entry.fact_id!r}")
        # (provider, model) is the rate limiter's accounting key and the
        # fallback fact lookup, so two rows on one endpoint would share a
        # bucket and resolve to whichever came first.
        endpoint_key = (entry.provider_tier, entry.api_model_id)
        if endpoint_key in endpoint_keys:
            raise ValueError(
                f"{where}: duplicate provider/model fact "
                f"{entry.provider_tier}|{entry.api_model_id}"
            )
        fact_ids.add(entry.fact_id)
        endpoint_keys.add(endpoint_key)


def merge_catalogs(
    packaged: Sequence[ModelCatalogEntry],
    override: Sequence[ModelCatalogEntry],
) -> List[ModelCatalogEntry]:
    """Packaged defaults with the override file layered on top, by ``fact_id``.

    Overriding a packaged id replaces that row in place (order preserved, so a
    re-tuned limit does not move the model); a new id appends.
    """

    merged = {entry.fact_id: entry for entry in packaged}
    for entry in override:
        merged[entry.fact_id] = entry
    result = list(merged.values())
    _reject_duplicates(result, "merged catalog")
    return result


@lru_cache(maxsize=1)
def default_model_catalog() -> tuple[ModelCatalogEntry, ...]:
    """The effective catalog: packaged defaults + the data-root override."""

    from finesub.paths import resolve_model_catalog_override

    packaged = load_model_catalog()
    override_path = resolve_model_catalog_override()
    if override_path is None:
        return tuple(packaged)
    override = load_model_catalog(override_path, self_reported=True)
    return tuple(merge_catalogs(packaged, override))


def catalog_by_api_model_id(
    entries: Iterable[ModelCatalogEntry] | None = None,
) -> Dict[str, ModelCatalogEntry]:
    result: Dict[str, ModelCatalogEntry] = {}
    for entry in entries or default_model_catalog():
        if entry.api_model_id not in result:
            result[entry.api_model_id] = entry
    return result


def catalog_by_fact_id(
    entries: Iterable[ModelCatalogEntry] | None = None,
) -> Dict[str, ModelCatalogEntry]:
    return {
        entry.fact_id: entry for entry in entries or default_model_catalog()
    }


def get_model_catalog_entry_by_fact(
    fact_id: str,
) -> ModelCatalogEntry | None:
    return catalog_by_fact_id().get(fact_id)


def get_model_catalog_entry(api_model_id: str) -> ModelCatalogEntry | None:
    return catalog_by_api_model_id().get(api_model_id)


def get_model_catalog_entry_for_tier(
    api_model_id: str,
    provider_tier: str,
) -> ModelCatalogEntry | None:
    for entry in default_model_catalog():
        if entry.api_model_id == api_model_id and entry.provider_tier == provider_tier:
            return entry
    return None


def provider_tier_for_model(api_model_id: str, fallback: str = "") -> str:
    entry = get_model_catalog_entry(api_model_id)
    return entry.provider_tier if entry is not None else fallback


def supports_reasoning(api_model_id: str, default: bool = True) -> bool:
    entry = get_model_catalog_entry(api_model_id)
    return default if entry is None else entry.thinking_levels is not None


def quality_floor_warnings(
    entries: Iterable[ModelCatalogEntry],
    *,
    floor_score: int,
    reference_model: str,
    owner: str,
) -> List[str]:
    """Advisory warnings for members scoring below a task-group floor.

    ``quality_score`` never enters routing decisions (plan v2 D5); this only
    phrases the warning that preset binding emits once per cell (§5.4). The
    message is quantified and names the member, so the user knows what to move
    rather than being told "the gap is too large".
    """

    return [
        f"{owner}: 成员 {entry.fact_id} 的 quality_score={entry.quality_score} "
        f"低于下限 {floor_score}（参考模型 {reference_model}）"
        for entry in entries
        if entry.quality_score < floor_score
    ]
