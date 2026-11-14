"""Model facts: one TOML table per callable (provider, model).

The catalog is **the** place a model's facts live -- capabilities, limits, the
endpoint that serves it and the thinking dialect it speaks -- while
``model_routes.toml`` composes them into model groups, task groups and
presets. Facts vs composition, one file each (owner decision 2026-08-12).

Two files, same format (``[models.<fact_id>]`` tables): the **packaged default**
that ships with the code, and an optional **override** in the data root.
Installed front ends put that override in ``user-data``; a source checkout has
no separate user-data, so it is simply the checkout root -- exactly the
resolution ``.env`` and ``config.toml`` already use, and it stays in the paths
layer rather than here.

The override is a **per-key patch** (:func:`load_effective_catalog`): a table
naming a packaged model states only the keys it changes, a new model states
at least :data:`REQUIRED_KEYS`. ``""`` on any other key means "the loader's
default" -- which is how a patch takes a packaged value *away* (an unjudged
capability, no joint context window, no fallback model); deleting the key
inherits the packaged value instead.

``tpm`` and ``tpd`` are input-token limits only (output/thinking excluded).
``rpd``/``tpd`` are informational; runtime daily caps use ``.state`` exhaustion.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CATALOG_FILENAME = "model_catalog.toml"
#: The format before 2026-11-09. A file of that name in the data root is
#: refused with the conversion command rather than ignored: ignoring it would
#: silently drop every model the user had added.
LEGACY_CATALOG_FILENAME = "model_catalog.psv"
MIGRATE_COMMAND = "python -m finesub.llm.routing.migrate_catalog"
# Keys a model must state: everything else has a defensible default.
# ``max_input_tokens`` is required on purpose -- windows are planned against
# it, so an optimistic default explodes every window far from its cause.
REQUIRED_KEYS = (
    "provider_tier",
    "api_model_id",
    "max_input_tokens",
)
_STRING_KEYS = frozenset(
    {
        "provider_tier",
        "provider_kind",
        "base_url",
        "key_env",
        "display_name",
        "api_model_id",
        "fallback_model",
        "quota_pool",
        "note",
    }
)
_INT_KEYS = frozenset(
    {"max_input_tokens", "max_output_tokens", "context_window", "rpm", "tpm", "rpd", "tpd"}
)
_BOOL_KEYS = frozenset(
    {
        "supports_audio",
        "supports_video",
        "video_high_resolution_only",
        "supports_native_search",
        "is_free",
        "hint_output_ceiling",
    }
)
#: Per-model quality caps (owner 2026-11-08): the prepare task's chunk, a correction window's subtitles, a
#: knowledge-update chunk's CSV, in tokens. Unstated is the built-in default (`routing.config.QUALITY_CAPS`).
QUALITY_CAP_KEYS = ("prepare_chunk_tokens", "window_subtitle_tokens", "knowledge_csv_tokens")
#: Every key a model table may carry. ``thinking``, ``capability`` and
#: ``avoid`` take more than one shape and are parsed on their own.
CATALOG_KEYS = (
    _STRING_KEYS
    | _INT_KEYS
    | _BOOL_KEYS
    | {"token_scale", "thinking", "capability", "avoid", "search_request", *QUALITY_CAP_KEYS}
)
#: Keys that only tell a person something: patching just these does not make
#: a packaged model's numbers "user-typed" (``self_reported``).
ANNOTATION_KEYS = frozenset({"note", "avoid"})
# The score column's old name. Rejected with the new one rather than ignored.
RETIRED_KEYS = {"quality_score": "capability"}
#: `quota_pool` sentinel: this row draws on an allowance of its own.
#: Deliberately lower case, so it cannot be mistaken for a pool identifier --
#: every real one is written in caps. Same idea as `-1` for an unlimited rate.
PER_MODEL_QUOTA_POOL = "per-model"
#: Tiers where one CLI login covers several separately metered allowances, so
#: "the provider tier" is not an answer -- an empty `quota_pool` there is a
#: question nobody answered rather than a default worth guessing. Guessing
#: either way is wrong for some row on these tiers, and the expensive wrong
#: guess (merging) takes a working line out for hours with no call left to
#: thaw it, so they are made to say.
TIERS_METERED_PER_LINE = frozenset({"LOCAL_AGY", "LOCAL_CODEBUDDY"})

# Unstated-key defaults (owner decision 2026-08-12: "允许某些项留空，留空时取
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
    "LOCAL_CODEBUDDY": LOCAL_AGENT_KIND,
    "LOCAL_CONVERSATIONAL": CONVERSATIONAL_KIND,
}
DEFAULT_MAX_OUTPUT_TOKENS = 65_536
DEFAULT_RPM = 100
DEFAULT_TPM = 4_000_000
UNLIMITED = -1
#: ``rpm`` / ``tpm`` when a model states none, by dialect. A local or attached
#: agent is metered by its vendor's subscription, not by a per-minute budget
#: the harness could know, so it is unlimited; the 100 / 4M guard is for a
#: third-party HTTP endpoint (and Gemini rows, which state their published
#: limits anyway).
DEFAULT_RATES_BY_KIND = {
    LOCAL_AGENT_KIND: (UNLIMITED, UNLIMITED),
    CONVERSATIONAL_KIND: (UNLIMITED, UNLIMITED),
}
#: The vendor search tool each tier's transport can entitle, spelled the
#: vendor's way: Codex, Claude Code, dsh and CodeBuddy call it `web_search`,
#: agy calls it `search_web`. A target carries its tier's tool when its model
#: ``supports_native_search``; the tool goes out only on a call that asks for
#: native retrieval. Tiers absent here (custom HTTP endpoints, the attached
#: conversational agent) have no tool the harness could hand over.
NATIVE_SEARCH_TOOLS = {
    "GEMINI_FREE": "google_search",
    "GEMINI_PAID": "google_search",
    "LOCAL_CODEX": "web_search",
    "LOCAL_CLAUDE": "web_search",
    "LOCAL_AGY": "search_web",
    "LOCAL_DSH": "web_search",
    "LOCAL_CODEBUDDY": "web_search",
}
#: The two user-declared HTTP dialects, by kind rather than tier (a custom
#: tier is whatever name the user gave it). Anthropic's is its server-side
#: `web_search` tool; an OpenAI-compatible endpoint has no common switch, so
#: its "tool" is the model's own `search_request` merged into the body.
CUSTOM_SEARCH_TOOLS = {
    ANTHROPIC_KIND: "web_search",
    OPENAI_COMPAT_KIND: "search_request",
}
#: Top-level body keys the OpenAI-compatible transport writes itself; a
#: `search_request` naming one would overwrite the call, so it is refused at
#: load. `provider_transports.openai_compat_generate` writes exactly these (a
#: test reads its request body and holds the two equal).
OPENAI_COMPAT_REQUEST_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "reasoning_effort",
        "prompt_cache_key",
    }
)
# ``capability`` is advisory only (plan D5): it feeds the task-group floor
# warning and the artifacts, and never enters any routing decision.
# ``correction_prompt_tier`` is gone (plan v2 D2): variant ownership lives on
# the task-group cell (and optional model-group entry overrides), not on the
# model row.


class Capability(IntEnum):
    """A model's capability: four fixed tiers (owner 2026-11-08).

    They replaced a 0-100 score whose fine grain nobody had measured. The
    values are the scores each tier stands for, so a score still reads as a
    tier (:func:`parse_capability`) and the shipped floors keep their meaning:
    the old 50 / 70 floors are now "needs low" / "needs medium".
    """

    VERY_LOW = 30
    LOW = 50
    MEDIUM = 70
    HIGH = 90

    @property
    def label(self) -> str:
        return CAPABILITY_LABELS[self]


CAPABILITY_NAMES = {
    "very-low": Capability.VERY_LOW,
    "low": Capability.LOW,
    "medium": Capability.MEDIUM,
    "high": Capability.HIGH,
}
CAPABILITY_LABELS = {
    Capability.VERY_LOW: "极低",
    Capability.LOW: "低",
    Capability.MEDIUM: "中",
    Capability.HIGH: "高",
}
CAPABILITY_SCORE_RANGE = range(0, 101)
#: What an unstated ``capability`` counts as (owner 2026-09-04, when it was a
#: score). A blank cell used to sit below every shipped floor -- so the row
#: that said nothing was treated as the worst thing anyone had measured, and a
#: person adding a model they had not judged got a warning phrased as if they
#: had judged it badly. The honest reading of "unstated" is "no claim", and the
#: cheap direction is to let it through: capability gates nothing, it only
#: phrases a warning. A blank row therefore passes every floor and says so once
#: as a **note** (`preset_binding_notes`), not a warning.
UNSTATED_CAPABILITY = Capability.HIGH


def parse_capability(value: object, *, owner: str) -> Capability:
    """A tier name, or a 0-100 integer score read as the highest tier it reaches.

    A score below the low tier's 50 is very low; 82 is medium. Scores are the
    compatibility spelling (owner 2026-11-08); the shipped files write names.
    A score written as a string (``"77"``) is refused: TOML has integers.
    """

    score: object = value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in CAPABILITY_NAMES:
            return CAPABILITY_NAMES[text]
        score = None
    if isinstance(score, bool) or not isinstance(score, int) or score not in CAPABILITY_SCORE_RANGE:
        raise ValueError(
            f"{owner} must be one of {'|'.join(CAPABILITY_NAMES)} or a score in "
            f"[{CAPABILITY_SCORE_RANGE.start}, {CAPABILITY_SCORE_RANGE[-1]}]; got {value!r}"
        )
    reached = [tier for tier in Capability if tier >= Capability.LOW and score >= tier]
    return reached[-1] if reached else Capability.VERY_LOW

# ``thinking`` (owner design 2026-08-11, identity default 2026-08-12). The
# value's type says which of three things it is:
#
# - ``true`` (and unstated) -- the **identity mapping**: the harness's abstract
#   high/medium/low goes out verbatim. All three dialects happen to spell the
#   levels the same way (Gemini ``thinkingLevel``, OpenAI-compatible
#   ``reasoning_effort``, Anthropic ``output_config.effort``), so identity is
#   the right default rather than a lucky coincidence per row.
# - ``false`` -- the model takes no thinking parameter at all (self-hosted
#   endpoints that 400 on unknown fields declare this).
# - a list of three -- an explicit mapping of the abstract high, medium, low
#   requests, in that order. Strings are the provider's words (e.g.
#   ``["max", "default", "minimal"]``); integers are token budgets, for a model
#   whose thinking is a budget (Gemini 2.5), because the harness's derived ones
#   (a share of the output limit) can exceed the model's own ceiling and the
#   API rejects the call outright. Budgets are stored as digit strings, which
#   is how :func:`dispatched_thinking` tells the two apart.
IDENTITY_THINKING_LEVELS = ("high", "medium", "low")
_ABSTRACT_THINKING_INDEX = {"high": 0, "medium": 1, "low": 2}


def parse_thinking(value: object, *, owner: str) -> tuple[str, str, str] | None:
    if value is True:
        return IDENTITY_THINKING_LEVELS
    if value is False:
        return None
    if isinstance(value, list) and len(value) == 3:
        if all(isinstance(part, int) and not isinstance(part, bool) and part > 0 for part in value):
            return (str(value[0]), str(value[1]), str(value[2]))
        if all(
            isinstance(part, str) and part.strip() and not part.strip().isdigit()
            for part in value
        ):
            return (value[0].strip(), value[1].strip(), value[2].strip())
    raise ValueError(
        f"{owner}: thinking must be true (identity), false (no thinking "
        "parameter), three level names mapping high, medium, low "
        '(["xhigh", "high", "medium"]), or three positive integer token budgets '
        f"([24576, 16384, 8192]); got {value!r}"
    )


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


def dispatched_thinking(
    entry: "ModelCatalogEntry", level: str | None, derived_budget: int
) -> tuple[str, int]:
    """``(thinking_level, thinking_budget)`` actually sent for this model.

    An integer mapping is the model's own budget for that level and replaces
    the derived one (and no level word goes out); otherwise the mapped word
    goes out next to the derived budget, as before. A model that declares
    ``false`` gets neither.
    """

    if entry.thinking_levels is None:
        return "", 0
    value = thinking_value_for(entry, level)
    if value.isdigit():
        return "", int(value)
    return value, derived_budget


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
    #: The pool input and output share, when the provider has one. Blank in the
    #: file means "this row declares no joint constraint" and is stored as
    #: ``max_input_tokens + max_output_tokens``, which makes that constraint
    #: non-binding. A single-pool provider (Codex, Anthropic) needs to state it:
    #: for them an input of ``max_input_tokens`` leaves no room for the answer.
    #:
    #: ⚠ **Nothing enforces that.** The loader cannot know the pool's shape:
    #: ``provider_kind`` says which dialect a row speaks, and no dialect implies
    #: a pool -- ``openai_compat`` covers both kinds of provider, and every
    #: local agent (agy fronts Claude, codex fronts GPT) declares ``local_agent``.
    #: So a row that omits the cell plans against ``max_input + max_output``,
    #: and only its author knows whether that is true.
    #:
    #: The agy Claude rows (``local-agy-opus-5_5``, ``local-agy-sonnet-5_5``)
    #: are deliberately left blank (owner 2026-09-03, for the Opus 4.6 row they
    #: replaced). Claude is a single pool, but nobody has checked whether agy
    #: passes its full context through, so the rows keep the envelope that row
    #: always had (194000) instead of taking the vendor's 1M on faith -- filling
    #: ``max_input`` *and* the pool with 1M would raise it to 934464 on an
    #: unverified assumption. Those rows' ``max_input`` is not a vendor number
    #: either (``docs/llm_design_notes.md``, 「模型路由 v2 的持久决策」).
    #:
    #: Gemini publishes its two limits separately, so blank is the honest
    #: reading and the **free** rows use it. The **paid** rows and agy are still
    #: filled (``ctx = max_input = 1048576``, envelope 983040) -- an owner
    #: decision (2026-09-03) to plan them as one pool rather than trust that the
    #: two published numbers can be spent at the same time.
    #:
    #: Its **only** job is to cap the input envelope (owner 2026-09-03); it
    #: enters no other decision.
    context_window: int
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
    #: ``None`` = the row states no capability. Distinct from any tier, because
    #: "nobody has judged this model" and "somebody judged it low" call for
    #: different messages -- see ``UNSTATED_CAPABILITY``.
    capability: Capability | None
    #: Tell the worker, in the agent bootstrap, that its per-turn output is
    #: capped at this row's ``max_output_tokens`` and that it should break a
    #: long stretch with a cheap tool call rather than run into the cap.
    #:
    #: A switch, not a number: what gets stated is always this row's own
    #: ceiling, because a *wrong* one is measurably worse than silence
    #: (2026-09-04: `glm-5.3-flash` told its true 32000 finished a window it
    #: had twice timed out on, and told 64000 timed out again). Off by default
    #: -- it changes the prompt, and it helped exactly one of the three models
    #: it was measured on; `deepseek-v4-flash` was unmoved at every number.
    hint_output_ceiling: bool = False
    #: A second model to hand the session to when this one stops answering,
    #: passed to the CLI as its own fallback flag; blank means "no fallback,
    #: fail instead".
    #:
    #: ⚠ **This spends money.** It exists for the free/paid pairs a
    #: subscription offers -- CodeBuddy's `hy3` -> `hy3-x` -- where the
    #: free line is a daily allowance and the paid one bills credits
    #: (x0.05 against the free line's x0.00, vendor product config
    #: 2026-09-04; `hy4-preview` used to be a second pair, dropped
    #: 2026-09-11 when that line started billing credits). Owner
    #: decision 2026-09-04: for that row, a spent allowance should
    #: switch and carry on rather than stop the run, with one notice
    #: per session.
    #:
    #: Left blank for every other row, including `glm-5.3-flash` and
    #: DeepSeek -- those bill credits already and have no free twin to
    #: fall back *from*.
    fallback_model: str = ""
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
    # Which allowance this row draws on when it runs out -- an allowance is a
    # property of the line the vendor meters, and a row only *points* at one.
    #
    # Three ways to write it:
    #   ""            the provider tier, right wherever one subscription backs
    #                 every model on a tier (one Codex plan, one Claude plan);
    #   PER_MODEL     this row has an allowance of its own. For a vendor whose
    #                 free lines are metered one by one, so a name never has to
    #                 be invented and a line that stops being free is a
    #                 one-value edit rather than a rename;
    #   anything else a pool shared by every row that names it.
    #
    # The tier default is *not* right for Antigravity, where the Gemini and
    # Claude models are metered separately, nor for CodeBuddy, where each free
    # line has its own daily quota and everything payable shares one credit
    # balance. Booking those together would take a working model out of
    # service for hours because its neighbour ran dry -- which is why
    # `TIERS_METERED_PER_LINE` refuses to guess for them.
    quota_pool: str = ""
    #: For a person reading the catalog; nothing reads it.
    note: str = ""
    #: ``(task group, reason)`` pairs this model is not recommended for. Binding
    #: it into any cell of such a group warns with the reason; it never routes.
    avoid: tuple[tuple[str, str], ...] = ()
    #: OpenAI-compatible only: what turns the provider's own search on, merged
    #: into the top level of the request body on a call that searches. Kept as
    #: canonical JSON so the entry stays hashable and the routing identity
    #: sees a change to it; `search_request_body` reads it back.
    search_request: str = ""

    # Whether the user's override stated any of this model's facts (a new
    # model, or a patch beyond `ANNOTATION_KEYS`). Artifacts label those
    # numbers as user-typed, not measured.
    self_reported: bool = False
    #: ``QUALITY_CAP_KEYS``, ``None`` where the model states none. They shape planning, never a call.
    prepare_chunk_tokens: int | None = None
    window_subtitle_tokens: int | None = None
    knowledge_csv_tokens: int | None = None

    @property
    def effective_quota_pool(self) -> str:
        if self.quota_pool == PER_MODEL_QUOTA_POOL:
            # Derived rather than written down, so a new free line needs no
            # invented name (and two rows cannot silently share one by typo).
            # Keyed on (tier, model) rather than fact_id because that is what
            # an answering model name can be looked up by -- see
            # `get_model_catalog_entry_for_tier`.
            return f"{self.provider_tier}:{self.api_model_id}"
        return self.quota_pool or self.provider_tier

    @property
    def search_request_body(self) -> Dict[str, Any]:
        return json.loads(self.search_request) if self.search_request else {}

    @property
    def search_tool(self) -> str:
        """The search tool a call on this model hands over, or "" for none.

        Its tier's (`NATIVE_SEARCH_TOOLS`), or its dialect's for a user-declared
        endpoint (`CUSTOM_SEARCH_TOOLS`), when the model can search at all.
        """

        if not self.supports_native_search:
            return ""
        return NATIVE_SEARCH_TOOLS.get(self.provider_tier) or CUSTOM_SEARCH_TOOLS.get(
            self.provider_kind, ""
        )

    @property
    def effective_capability(self) -> Capability:
        """The tier comparisons use; an unstated one passes every floor."""

        return UNSTATED_CAPABILITY if self.capability is None else self.capability


def _catalog_path() -> Path:
    return Path(__file__).resolve().with_name(CATALOG_FILENAME)


def read_catalog_tables(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """One catalog file as raw ``{fact_id: {key: value}}``, keys checked, values not.

    Values are parsed only after the override has been laid over the packaged
    tables (:func:`load_effective_catalog`), so derived defaults and the
    cross-key checks see the merged model.
    """

    catalog_path = Path(path).expanduser()
    if catalog_path.suffix.lower() == ".psv":
        raise ValueError(legacy_catalog_message(catalog_path))
    try:
        data = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{catalog_path}: {exc}") from exc
    extra = sorted(set(data) - {"models"})
    if extra:
        raise ValueError(
            f"{catalog_path}: only [models.<name>] tables are allowed; found {extra}"
        )
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"{catalog_path}: models must be a table")
    tables: Dict[str, Dict[str, Any]] = {}
    for fact_id, raw in models.items():
        where = f"{catalog_path}: models.{fact_id}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a table")
        for key in raw:
            if key in RETIRED_KEYS:
                raise ValueError(f"{where}: {key} is now called {RETIRED_KEYS[key]}")
            if key not in CATALOG_KEYS:
                raise ValueError(
                    f"{where}: unknown key {key!r}; known keys: {', '.join(sorted(CATALOG_KEYS))}"
                )
        tables[fact_id] = dict(raw)
    return tables


def legacy_catalog_message(path: Path) -> str:
    return (
        f"{path} uses the catalog format retired on 2026-11-09 (pipe-delimited "
        f"rows); this version reads {CATALOG_FILENAME}. Convert it with "
        f"`{MIGRATE_COMMAND} {path}` (it keeps the old file as a .bak)."
    )


def overlay_catalog_tables(
    packaged: Mapping[str, Mapping[str, Any]],
    override: Mapping[str, Mapping[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], frozenset[str]]:
    """The override laid over the packaged tables key by key.

    Returns the merged tables (packaged order, new models appended) and the
    models whose facts the override stated: every new model, and every patch
    that touches more than `ANNOTATION_KEYS` -- a note added to a packaged
    model leaves its numbers measured.
    """

    merged = {fact_id: dict(raw) for fact_id, raw in packaged.items()}
    stated: set[str] = set()
    for fact_id, raw in override.items():
        if fact_id in merged:
            merged[fact_id].update(raw)
            if set(raw) - ANNOTATION_KEYS:
                stated.add(fact_id)
        else:
            merged[fact_id] = dict(raw)
            stated.add(fact_id)
    return merged, frozenset(stated)


def _typed(raw: Mapping[str, Any], key: str, kinds: tuple[type, ...], where: str) -> Any:
    """``raw[key]`` when it has one of ``kinds``; None when unstated or ``""``.

    ``""`` is "the loader's default" on every key; a value of any other wrong
    type is an error, so ``"true"`` or ``"100"`` never pass for a bool or a
    number, and ``0`` / ``false`` are values, not "unstated".
    """

    value = raw.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool) and bool not in kinds:
        value_ok = False
    else:
        value_ok = isinstance(value, kinds)
    if not value_ok:
        names = " or ".join("integer" if k is int else k.__name__ for k in kinds)
        raise ValueError(f"{where}: {key} must be {names} (or \"\" for the default); got {value!r}")
    return value


def _entry_from_table(
    fact_id: str, raw: Mapping[str, Any], *, where: str, self_reported: bool
) -> ModelCatalogEntry:
    for key in REQUIRED_KEYS:
        if raw.get(key) in (None, ""):
            raise ValueError(f"{where}: missing {key}")

    def string(key: str) -> str:
        value = _typed(raw, key, (str,), where)
        return "" if value is None else value.strip()

    def integer(key: str, default: int) -> int:
        value = _typed(raw, key, (int,), where)
        return default if value is None else value

    def flag(key: str, default: bool = False) -> bool:
        value = _typed(raw, key, (bool,), where)
        return default if value is None else value

    provider_tier = string("provider_tier")
    provider_kind = string("provider_kind").lower() or PACKAGED_PROVIDER_KINDS.get(
        provider_tier, DEFAULT_PROVIDER_KIND
    )
    if provider_kind not in PROVIDER_KINDS:
        raise ValueError(f"{where}: provider_kind must be one of {sorted(PROVIDER_KINDS)}")
    base_url = string("base_url").rstrip("/")
    if provider_kind in (OPENAI_COMPAT_KIND, ANTHROPIC_KIND) and not base_url:
        raise ValueError(f"{where}: base_url is required for provider_kind={provider_kind}")
    if provider_kind in (GEMINI_KIND, LOCAL_AGENT_KIND, CONVERSATIONAL_KIND) and base_url:
        raise ValueError(
            f"{where}: provider_kind={provider_kind} takes no base_url "
            "(the packaged transports own their endpoint)"
        )
    supports_audio = flag("supports_audio")
    supports_video = flag("supports_video")
    video_high_resolution_only = flag("video_high_resolution_only")
    if video_high_resolution_only and not supports_video:
        raise ValueError(
            f"{where}: video_high_resolution_only = true requires supports_video = true"
        )
    # Custom HTTP media remains unsupported: the two text transports flatten
    # messages to plain strings, so a row claiming media on one of them would
    # pass capability filtering and then have its clip
    # silently dropped -- the exact silent degradation the design forbids.
    # Declare it here rather than trusting the writer to know.
    if provider_kind in (OPENAI_COMPAT_KIND, ANTHROPIC_KIND):
        for key, claimed in (("supports_audio", supports_audio), ("supports_video", supports_video)):
            if claimed:
                raise ValueError(
                    f"{where}: {key} = true is not available on "
                    f"provider_kind={provider_kind} (custom HTTP media is "
                    "unsupported; use --media text to put a text model on "
                    "media-bearing material)"
                )
    max_input = integer("max_input_tokens", 0)
    if max_input <= 0:
        raise ValueError(f"{where}: max_input_tokens must be positive")
    max_output = integer("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    if max_output <= 0:
        raise ValueError(f"{where}: max_output_tokens must be positive")
    # Unstated means "this model declares no joint constraint", which the sum
    # expresses exactly: the constraint can then never bind. A stated number is
    # checked instead of clamped: a pool smaller than either half it must hold
    # is a declaration error, and clamping it would silently plan against a
    # window the provider does not have -- so `0` fails here rather than
    # reading as unstated.
    context_window = integer("context_window", max_input + max_output)
    if context_window < max(max_input, max_output):
        raise ValueError(
            f"{where}: context_window ({context_window}) is smaller than "
            f"max_input_tokens ({max_input}) or max_output_tokens ({max_output}); "
            "leave it out when the model declares no joint constraint"
        )
    default_rpm, default_tpm = DEFAULT_RATES_BY_KIND.get(
        provider_kind, (DEFAULT_RPM, DEFAULT_TPM)
    )
    raw_capability = _typed(raw, "capability", (str, int), where)
    thinking = raw.get("thinking")
    token_scale = _typed(raw, "token_scale", (int, float), where)
    if token_scale is not None and not token_scale > 0:
        raise ValueError(f"{where}: token_scale must be positive")
    quality_caps = {key: _typed(raw, key, (int,), where) for key in QUALITY_CAP_KEYS}
    for key, value in quality_caps.items():
        # 0 means "no cap" in `config.toml`; here it would win every group minimum and quietly lift the cap
        if value is not None and value < 1:
            raise ValueError(f"{where}: {key} must be at least 1 (leave it out for the default)")
    api_model_id = string("api_model_id")
    search_request = _parse_search_request(raw.get("search_request"), provider_kind, where)
    # Every model a search tool exists for is taken to be able to use it
    # (owner 2026-11-09); one that cannot says so. An OpenAI-compatible
    # endpoint has a tool only once its `search_request` says how to ask.
    searchable = (
        provider_tier in NATIVE_SEARCH_TOOLS
        or provider_kind == ANTHROPIC_KIND
        or bool(search_request)
    )
    supports_native_search = flag("supports_native_search", searchable)
    if supports_native_search and provider_kind == OPENAI_COMPAT_KIND and not search_request:
        raise ValueError(
            f"{where}: supports_native_search = true on provider_kind=openai_compat "
            "needs search_request, the request fields that turn this provider's "
            "search on (docs/manual/model-routing.md)"
        )
    return ModelCatalogEntry(
        fact_id=fact_id,
        provider_tier=provider_tier,
        display_name=string("display_name") or api_model_id,
        api_model_id=api_model_id,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        context_window=context_window,
        supports_audio=supports_audio,
        supports_video=supports_video,
        video_high_resolution_only=video_high_resolution_only,
        supports_native_search=supports_native_search,
        thinking_levels=(
            IDENTITY_THINKING_LEVELS
            if thinking is None or thinking == ""
            else parse_thinking(thinking, owner=where)
        ),
        rpm=integer("rpm", default_rpm),
        tpm=integer("tpm", default_tpm),
        rpd=integer("rpd", UNLIMITED),
        tpd=integer("tpd", UNLIMITED),
        is_free=flag("is_free"),
        capability=(
            None
            if raw_capability is None
            else parse_capability(raw_capability, owner=f"{where}: capability")
        ),
        hint_output_ceiling=flag("hint_output_ceiling"),
        fallback_model=string("fallback_model"),
        quota_pool=string("quota_pool"),
        provider_kind=provider_kind,
        base_url=base_url,
        key_env=string("key_env"),
        token_scale=1.0 if token_scale is None else float(token_scale),
        note=string("note"),
        avoid=_parse_avoid(raw.get("avoid"), where),
        search_request=search_request,
        self_reported=self_reported,
        **quality_caps,
    )


def _parse_search_request(value: object, provider_kind: str, where: str) -> str:
    """A table of request-body fields, as canonical JSON; "" when unstated."""

    if value is None or value == "":
        return ""
    if provider_kind != OPENAI_COMPAT_KIND:
        raise ValueError(
            f"{where}: search_request is only for provider_kind=openai_compat "
            f"(this model is {provider_kind})"
        )
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"{where}: search_request must be a table of request fields, "
            f"e.g. {{ enable_search = true }}; got {value!r}"
        )
    clashing = sorted(set(value) & OPENAI_COMPAT_REQUEST_KEYS)
    if clashing:
        raise ValueError(
            f"{where}: search_request may not set {clashing}: the transport "
            "writes those itself"
        )
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        # A TOML date or time has no JSON form.
        raise ValueError(f"{where}: search_request must hold only JSON values: {exc}") from exc


def _parse_avoid(value: object, where: str) -> tuple[tuple[str, str], ...]:
    """``{task group: reason}``; group names are checked against the routes."""

    if value is None or value == "":
        return ()
    if not isinstance(value, dict) or not all(
        isinstance(group, str) and isinstance(reason, str) and reason.strip()
        for group, reason in value.items()
    ):
        raise ValueError(
            f'{where}: avoid must be a table of task group = "reason"; got {value!r}'
        )
    return tuple(sorted((group, reason.strip()) for group, reason in value.items()))


def _entries_from_tables(
    tables: Mapping[str, Mapping[str, Any]],
    where: Any,
    *,
    self_reported: Iterable[str] = (),
) -> List[ModelCatalogEntry]:
    stated = frozenset(self_reported)
    entries = [
        _entry_from_table(
            fact_id,
            raw,
            where=f"{where}: models.{fact_id}",
            self_reported=fact_id in stated,
        )
        for fact_id, raw in tables.items()
    ]
    _validate_catalog_table(entries, where)
    return entries


def load_model_catalog(path: str | Path | None = None) -> List[ModelCatalogEntry]:
    """Parse one complete catalog file (the packaged one by default)."""

    catalog_path = Path(path).expanduser() if path is not None else _catalog_path()
    return _entries_from_tables(read_catalog_tables(catalog_path), catalog_path)


def load_effective_catalog(override: str | Path | None) -> List[ModelCatalogEntry]:
    """The packaged catalog with an override file patched over it, key by key."""

    packaged = read_catalog_tables(_catalog_path())
    if override is None:
        return _entries_from_tables(packaged, _catalog_path())
    override_path = Path(override).expanduser()
    merged, stated = overlay_catalog_tables(packaged, read_catalog_tables(override_path))
    return _entries_from_tables(
        merged, f"{override_path} over the packaged catalog", self_reported=stated
    )


def _validate_catalog_table(
    entries: Sequence[ModelCatalogEntry], where: Any
) -> None:
    """Every check a whole table has to pass, in one call.

    One entry point on purpose: the effective catalog is reached by two paths
    -- the packaged file alone when there is no override, and the merged table
    when there is -- and a check wired into only one of them is a check the
    other silently skips.
    """

    _reject_duplicates(entries, where)
    _reject_bad_quota_pools(entries, where)


def _reject_bad_quota_pools(
    entries: Sequence[ModelCatalogEntry], where: Any
) -> None:
    """Two things a table of allowances must not say.

    Neither is detectable later: a wrong pool produces no error, just a freeze
    booked against the wrong line -- or a working line taken out for hours.
    """

    by_pool: dict[str, set[bool]] = {}
    for entry in entries:
        if (
            entry.provider_tier in TIERS_METERED_PER_LINE
            and not entry.quota_pool
        ):
            raise ValueError(
                f"{where}: {entry.fact_id!r} leaves quota_pool empty, but "
                f"{entry.provider_tier} meters several allowances behind one "
                f"login, so the tier is not an answer. Write "
                f"{PER_MODEL_QUOTA_POOL!r} if this line has a quota of its "
                f"own, or the name of the pool it shares (e.g. "
                f"CODEBUDDY_CREDITS)."
            )
        by_pool.setdefault(entry.effective_quota_pool, set()).add(entry.is_free)
    # A free line and a paid one cannot draw on the same allowance, so this
    # says the table is wrong -- but it *warns*, and the asymmetry is the
    # point. `is_free` drives nothing else and never has, so its values across
    # the table are unverified annotations; promoting one to a load-blocking
    # gate would stop a person's working configuration over a column nobody
    # had reason to keep accurate. It found a real one the first time it ran
    # (a free row sharing the tier pool with paid ones), which is exactly the
    # service it should render: say so, let the owner decide.
    for pool, free_flags in by_pool.items():
        if len(free_flags) > 1:
            sharing = sorted(
                entry.fact_id
                for entry in entries
                if entry.effective_quota_pool == pool
            )
            _warn_catalog(
                f"{where}: quota_pool {pool!r} is shared by rows that "
                f"disagree about is_free ({', '.join(sharing)}). A free line "
                f"and a paid one cannot draw on the same allowance, so one of "
                f"them is booked against a quota it does not spend."
            )


def _warn_catalog(message: str) -> None:
    from finesub.reporting import current_reporter

    current_reporter().warning("model-catalog", message)


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


@lru_cache(maxsize=1)
def default_model_catalog() -> tuple[ModelCatalogEntry, ...]:
    """The effective catalog: packaged defaults + the data-root override."""

    from finesub.paths import resolve_model_catalog_override

    return tuple(load_effective_catalog(resolve_model_catalog_override()))


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


def supports_reasoning(api_model_id: str, default: bool = True) -> bool:
    entry = get_model_catalog_entry(api_model_id)
    return default if entry is None else entry.thinking_levels is not None


def capability_floor_warnings(
    entries: Iterable[ModelCatalogEntry],
    *,
    min_capability: Capability,
    reference_model: str,
    owner: str,
) -> List[str]:
    """Advisory warnings for members below a task group's minimum capability.

    ``capability`` never enters routing decisions (plan v2 D5); this only phrases
    the warning that preset binding emits once per cell (§5.4). The message
    names the member and both tiers, so the user knows what to move rather
    than being told "the gap is too large".
    """

    return [
        f"{owner}: 成员 {entry.fact_id} 的能力档「{entry.effective_capability.label}」"
        f"低于下限「{min_capability.label}」（参考模型 {reference_model}）"
        for entry in entries
        if entry.effective_capability < min_capability
    ]


def unstated_capability_notes(
    entries: Iterable[ModelCatalogEntry], *, owner: str
) -> List[str]:
    """A note per member whose row states no ``capability``.

    Deliberately not a warning: nothing is wrong, and the run is going to
    proceed on the most permissive reading of the blank cell
    (``UNSTATED_CAPABILITY``). Saying it once per bound group is what stops
    that permissiveness from being invisible -- a floor that never fires
    because nobody filled the column in looks exactly like a floor that passed.
    """

    return [
        f"{owner}: 成员 {entry.fact_id} 未声明能力档，"
        f"按「{UNSTATED_CAPABILITY.label}」计（不触发任何能力下限）"
        for entry in entries
        if entry.capability is None
    ]
