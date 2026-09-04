"""Local web search + extract agent: Exa primary, then Gemma4, then Tavily.

Gemini 3 free-tier google_search grounding is unavailable, so correction and
research calls do not enable native search directly. The harness runs retrieval
locally and injects rendered results into prompts. Two kinds of retrieval are
supported:

* **search** — keyword/neural web search (Exa → Gemma4 → Tavily).
* **extract** — deep single-URL page-content extraction (Exa → Gemma4 → Tavily).

Neither has a keyless local fallback: once every configured pool fails, the
request returns an error result rather than a degraded one. A scraped-HTML
fallback used to sit at the end of the search chain and was removed in 0.5.0 --
it answered a provider outage with silently empty results, which reads exactly
like "the web has nothing on this" and is the one answer retrieval must never
invent.

Each API provider (Exa keys in ``EXA_KEYS``, Gemma4 keys in ``GEMINI_FREE``,
Tavily keys in ``TAVILY_KEYS``) is driven through a persistent bounded key pool;
keys that hit auth/quota errors are locked for a cooldown. A provider with no
configured keys is skipped silently. Both searches and extracts accept an
optional one-sentence *guided query* that biases what the provider
highlights/extracts from a page without changing the search keywords (Exa
``highlights.query`` / Gemma4 prompt goals / Tavily extract ``query``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import html as html_module
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Sequence
from urllib.parse import unquote, urlsplit

from finesub import state as state_store
from finesub.reporting import current_reporter, redact_credentials
from finesub.paths import resolve_state_file
from .http import llm_http_client
from .routing import api_keys
from .output_tags import GUIDED_QUERY_SEPARATOR

if TYPE_CHECKING:
    from .injection_budget import RenderedBlock


TAVILY_KEYS_ENV = "TAVILY_KEYS"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
DEFAULT_TAVILY_POOL_SIZE = 3
DEFAULT_TAVILY_LOCK_SECONDS = 24 * 60 * 60

EXA_KEYS_ENV = "EXA_KEYS"
EXA_SEARCH_URL = "https://api.exa.ai/search"
EXA_CONTENTS_URL = "https://api.exa.ai/contents"
DEFAULT_EXA_POOL_SIZE = 3
DEFAULT_EXA_LOCK_SECONDS = 24 * 60 * 60
# Exa deep-search content freshness / chunk knobs (per the Exa contents API).
EXA_MAX_AGE_HOURS = 168
TAVILY_EXTRACT_CHUNKS_PER_SOURCE = 5
DEFAULT_SEARCH_MAX_RESULTS = 10
EXTRA_INFO_URL_EXTRACT_LIMIT = 8

# Rendering caps (tokens). Per-snippet/answer/content caps are soft formatting
# limits inside a section; the per-section and whole-block budgets are the
# harness contract (see injection_budget / config.injection_block_token_limit).
SEARCH_SNIPPET_MAX_TOKENS = 600
EXTRACT_CONTENT_MAX_TOKENS = 1_800
_ERROR_TEXT_MAX_TOKENS = 200

EXA_PROVIDER = "exa"
TAVILY_PROVIDER = "tavily"
GEMMA4_PROVIDER = "gemma4"
GEMMA4_MODEL = "gemini/gemma-4-31b-it"
GEMMA4_REST_MODEL = "gemma-4-31b-it"
GEMMA4_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMMA4_REST_MODEL}:generateContent"
)
DEFAULT_GEMMA4_POOL_SIZE = 2
DEFAULT_GEMMA4_LOCK_SECONDS = 24 * 60 * 60
DEFAULT_GEMMA4_TIMEOUT_SECONDS = 1200.0
GEMMA4_MAX_OUTPUT_TOKENS = 32_768
GEMMA4_THINKING_PREFIX = "<|think|>"
GEMMA4_SEARCH_BATCH_QUERY_LIMIT = 8

# Provider order lives in one place so fallback-position experiments are a
# constant edit instead of a search/extract control-flow rewrite.
SEARCH_PROVIDER_ORDER = (
    EXA_PROVIDER,
    GEMMA4_PROVIDER,
    TAVILY_PROVIDER,
)
EXTRACT_PROVIDER_ORDER = (
    EXA_PROVIDER,
    GEMMA4_PROVIDER,
    TAVILY_PROVIDER,
)

_URL_PATTERN = re.compile(r"""https?://[^\s<>"'\)\]\}，。；]+""", re.IGNORECASE)


def extract_urls_from_text(text: str, *, limit: int = EXTRA_INFO_URL_EXTRACT_LIMIT) -> List[str]:
    """Deduped HTTP(S) URLs from free text (e.g. ``--extra-info``), order preserved."""

    seen: set[str] = set()
    urls: List[str] = []
    for match in _URL_PATTERN.finditer(text or ""):
        url = match.group(0).rstrip(".,;:)")
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max(0, int(limit)):
            break
    return urls


# RFC 3986 unreserved: percent-escaping these carries no meaning, so decoding
# them is the one percent-decode that cannot change how a server reads the URL.
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_PERCENT_TRIPLET = re.compile(r"%[0-9A-Fa-f]{2}")


def _normalize_percent(text: str) -> str:
    """Decode unreserved escapes, uppercase the rest (RFC 3986 §6.2.2.2).

    Deliberately *not* a full unquote: turning ``%2F`` into ``/`` would change
    the path structure, i.e. two different resources would compare equal.
    """

    def _replace(match: re.Match[str]) -> str:
        decoded = unquote(match.group(0))
        return decoded if decoded in _UNRESERVED else match.group(0).upper()

    return _PERCENT_TRIPLET.sub(_replace, text)


def guided_query_key(guided: str) -> str:
    """Dedup key for a guided query: strict on purpose.

    A loose key would fold "look at this page again, but for X" into the
    earlier fetch for Y and drop it -- exactly the request worth keeping.
    Re-asking with a reworded focus only costs the model its own budget.
    """

    return " ".join((guided or "").split()).lower()


def request_label(text: str, guided: str = "") -> str:
    """The name one retrieval request goes by in budgets and reports.

    Request identity is ``(query-or-url, guided query)`` everywhere -- the
    guided query changes what the provider highlights and extracts, so the
    same page read for a different purpose is a different request. The label
    carries both so a rendered section can name the request it answers
    instead of being matched back to it by list position.
    """

    guided = (guided or "").strip()
    return f"{text}{GUIDED_QUERY_SEPARATOR}{guided}" if guided else text


def normalize_url_key(url: str) -> str:
    """A comparison key for "is this the URL I already saw?".

    Only used to *look one up*; the request is then made against the stored
    original, so this can afford to be forgiving. Folding http/https, decoding
    HTML entities and normalizing IDN/percent forms all mean "the model wrote
    the same link a bit differently", never "fetch something new".

    Returns ``""`` for anything that is not an absolute http(s) URL.
    """

    text = html_module.unescape((url or "").strip()).strip("<>\"'")
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return ""
    # http and https fold together: the same page over either scheme is the
    # same page, and a model copying a link often normalizes the scheme itself.
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return ""
    host = parts.hostname.rstrip(".").lower()
    try:
        # IDN both ways: 例え.jp and xn--r8jz45g.jp are one host.
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        pass
    try:
        parsed_port = parts.port
    except ValueError:
        return ""
    port = "" if parsed_port in (None, 80, 443) else f":{parsed_port}"
    if ":" in host:
        host = f"[{host}]"
    path = _normalize_percent(parts.path) or "/"
    # The fragment never reaches the server, so it cannot distinguish two
    # fetches. The query does reach it, so it is kept: parameter order and
    # every reserved escape (``%2F``, ``%3D``, ``%26`` ...) survive, and only
    # unreserved escapes are folded -- ``?a=%7Ex`` and ``?a=~x`` address the
    # same resource. Percent-encoding a signature is therefore preserved too,
    # and either way the fetch goes to the stored original, not to this key.
    query = f"?{_normalize_percent(parts.query)}" if parts.query else ""
    return f"{host}{port}{path}{query}"

# Tavily statuses that mean "this key is unusable" (bad key, plan/quota limits).
_TAVILY_KEY_ERROR_STATUSES = {401, 403, 429, 432, 433}
# Exa statuses that mean "this key is unusable" (auth, payment, rate/quota).
_EXA_KEY_ERROR_STATUSES = {401, 402, 403, 429}
# Gemini auth / quota statuses that make a free-tier key unusable for retrieval.
_GEMMA4_KEY_ERROR_STATUSES = {401, 403, 429}


def _default_state_path() -> Path:
    return resolve_state_file()


def _key_id_for_secret(secret: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


@dataclass(frozen=True)
class SearchApiKey:
    key_id: str
    key: str


@dataclass(frozen=True)
class SearchFallbackEvent:
    provider: str
    reason: str
    detail: str = ""
    key_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "reason": self.reason,
            "detail": self.detail,
            "key_id": self.key_id,
        }


@dataclass(frozen=True)
class SearchResultItem:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class QuerySearchResult:
    query: str
    provider: str = ""
    items: tuple[SearchResultItem, ...] = ()
    answer: str = ""
    error: str = ""
    fallbacks: tuple[SearchFallbackEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Stamped back from the request by ``search_many`` so a result can name
    # the request it answers without the caller relying on list positions.
    guided_query: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and (bool(self.items) or bool(self.answer))

    @property
    def request_label(self) -> str:
        return request_label(self.query, self.guided_query)


@dataclass(frozen=True)
class SearchRequest:
    """A search query plus an optional one-sentence extraction-focus hint.

    ``guided_query`` biases what the provider highlights on a result page (Exa
    ``highlights.query``); it must not alter the search keywords themselves and
    is ignored by providers that have no highlight concept (Tavily search).
    """

    query: str
    guided_query: str = ""


@dataclass(frozen=True)
class ExtractRequest:
    """A URL to deep-extract plus an optional one-sentence focus hint."""

    url: str
    guided_query: str = ""


@dataclass(frozen=True)
class QueryExtractResult:
    url: str
    provider: str = ""
    title: str = ""
    content: str = ""
    error: str = ""
    fallbacks: tuple[SearchFallbackEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # See ``QuerySearchResult.guided_query``.
    guided_query: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.content)

    @property
    def request_label(self) -> str:
        return request_label(self.url, self.guided_query)


def _request_text(request: "SearchRequest | ExtractRequest") -> str:
    """What this request was for, for a log line: the query, or the URL."""

    return getattr(request, "query", "") or getattr(request, "url", "")


def load_search_api_keys(env_name: str) -> List[str]:
    """Read ``name:key`` map entries from the environment / project .env."""

    from . import llm_runtime

    pool_by_env = {
        EXA_KEYS_ENV: api_keys.EXA_POOL,
        TAVILY_KEYS_ENV: api_keys.TAVILY_POOL,
        "GEMINI_FREE": api_keys.GEMINI_FREE_POOL,
    }
    pool_name = pool_by_env.get(env_name)
    if pool_name is None:
        return llm_runtime._get_key_list(env_name, llm_runtime._read_dotenv())
    return [
        entry.key
        for entry in api_keys.resolve_pool(pool_name, llm_runtime._read_dotenv())
    ]


def load_search_api_key_entries(env_name: str) -> List[SearchApiKey]:
    """Read named API keys without exposing key material in persistent state."""

    from . import llm_runtime

    pool_by_env = {
        EXA_KEYS_ENV: api_keys.EXA_POOL,
        TAVILY_KEYS_ENV: api_keys.TAVILY_POOL,
        "GEMINI_FREE": api_keys.GEMINI_FREE_POOL,
    }
    pool_name = pool_by_env.get(env_name)
    if pool_name is None:
        return [
            SearchApiKey(key_id=_key_id_for_secret(key), key=key)
            for key in llm_runtime._get_key_list(env_name, llm_runtime._read_dotenv())
        ]
    return [
        SearchApiKey(key_id=entry.key_id, key=entry.key)
        for entry in api_keys.resolve_pool(pool_name, llm_runtime._read_dotenv())
    ]


class ApiKeyPool:
    """Persistent key pool that prefers the most recently successful key."""

    def __init__(
        self,
        provider: str,
        entries: Sequence[SearchApiKey],
        *,
        state_path: str | Path | None = None,
        lock_seconds: float = DEFAULT_TAVILY_LOCK_SECONDS,
        now_func: Callable[[], float] = time.time,
    ) -> None:
        self.provider = provider
        self.entries = list(entries)
        self.state_path = Path(state_path) if state_path else _default_state_path()
        self.lock_seconds = float(lock_seconds)
        self.now_func = now_func

    def available_entries(self) -> list[SearchApiKey]:
        now = self.now_func()
        state = self._provider_state()
        indexed = [
            (idx, entry)
            for idx, entry in enumerate(self.entries)
            if not self._is_locked(entry.key_id, state, now)
        ]
        indexed.sort(
            key=lambda item: (
                state.get(item[1].key_id, {}).get("last_used_at") is not None,
                float(state.get(item[1].key_id, {}).get("last_used_at") or -1.0),
                -item[0],
            ),
            reverse=True,
        )
        return [entry for _, entry in indexed]

    def locked_key_ids(self) -> list[str]:
        now = self.now_func()
        state = self._provider_state()
        return [
            entry.key_id
            for entry in self.entries
            if self._is_locked(entry.key_id, state, now)
        ]

    def mark_used(self, entry: SearchApiKey) -> None:
        with self._section() as section:
            row = section.setdefault("keys", {}).setdefault(entry.key_id, {})
            row["last_used_at"] = self.now_func()
            row.pop("locked_until", None)
            row.pop("locked_at", None)
            row.pop("lock_reason", None)

    def lock(self, entry: SearchApiKey, reason: str) -> None:
        now = self.now_func()
        with self._section() as section:
            row = section.setdefault("keys", {}).setdefault(entry.key_id, {})
            row["locked_at"] = now
            row["locked_until"] = now + self.lock_seconds
            row["lock_reason"] = reason

    def _section(self):
        """Locked read-modify-write of this provider's slice of .state."""

        return state_store.state_section(self.provider, self.state_path)

    def _provider_state(self) -> dict[str, dict[str, Any]]:
        return state_store.read_section(self.provider, self.state_path).get("keys", {})

    def _is_locked(
        self,
        key_id: str,
        state: Mapping[str, Mapping[str, Any]],
        now: float,
    ) -> bool:
        row = state.get(key_id, {})
        locked_until = row.get("locked_until")
        return isinstance(locked_until, (int, float)) and float(locked_until) > now



def _select_pool(
    entries: Sequence[SearchApiKey],
    *,
    pool_selector: Sequence[str],
    pool_size: int,
) -> list[SearchApiKey]:
    if pool_selector:
        selected = [
            entry
            for entry in entries
            if entry.key_id in pool_selector or entry.key in pool_selector
        ]
        if len(selected) > max(0, int(pool_size)):
            current_reporter().warning(
                "key-pool-oversized",
                f"API key pool selects {len(selected)} keys; the recommended "
                f"maximum is {max(0, int(pool_size))}",
                impact="过大的池可能触发供应商风控",
            )
        return selected
    return list(entries[: max(0, int(pool_size))])


def _with_fallbacks(
    result: QuerySearchResult,
    fallbacks: Sequence[SearchFallbackEvent],
) -> QuerySearchResult:
    if not fallbacks:
        return result
    return QuerySearchResult(
        query=result.query,
        provider=result.provider,
        items=result.items,
        answer=result.answer,
        error=result.error,
        fallbacks=tuple(fallbacks),
        metadata=result.metadata,
    )


def _extract_with_fallbacks(
    result: QueryExtractResult,
    fallbacks: Sequence[SearchFallbackEvent],
) -> QueryExtractResult:
    if not fallbacks:
        return result
    return QueryExtractResult(
        url=result.url,
        provider=result.provider,
        title=result.title,
        content=result.content,
        error=result.error,
        fallbacks=tuple(fallbacks),
        metadata=result.metadata,
    )


def _as_search_request(item: "str | SearchRequest") -> SearchRequest:
    if isinstance(item, SearchRequest):
        return SearchRequest(query=(item.query or "").strip(), guided_query=item.guided_query)
    return SearchRequest(query=(str(item) or "").strip())


def _chunks(items: Sequence[int], size: int) -> list[list[int]]:
    size = max(1, int(size))
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


class WebSearchClient:
    """Fallback-chain retrieval: Exa, then Gemma4, then Tavily.

    Extract (deep page content) uses Exa/Gemma4/Tavily but has no local fallback
    yet, so it degrades to an error result when all providers fail.
    """

    def __init__(
        self,
        *,
        exa_keys: Optional[Sequence[str]] = None,
        exa_pool: Optional[Sequence[str]] = None,
        exa_pool_size: int = DEFAULT_EXA_POOL_SIZE,
        exa_lock_seconds: float = DEFAULT_EXA_LOCK_SECONDS,
        gemma_keys: Optional[Sequence[str]] = None,
        gemma_pool_size: int = DEFAULT_GEMMA4_POOL_SIZE,
        gemma_lock_seconds: float = DEFAULT_GEMMA4_LOCK_SECONDS,
        gemma_timeout_seconds: float = DEFAULT_GEMMA4_TIMEOUT_SECONDS,
        gemma_rate_limiter: Any | None = None,
        tavily_keys: Optional[Sequence[str]] = None,
        tavily_pool: Optional[Sequence[str]] = None,
        tavily_pool_size: int = DEFAULT_TAVILY_POOL_SIZE,
        tavily_lock_seconds: float = DEFAULT_TAVILY_LOCK_SECONDS,
        state_path: str | Path | None = None,
        now_func: Callable[[], float] = time.time,
        max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
        timeout_seconds: float = 30.0,
        query_interval_seconds: float = 1.5,
        client_factory: Callable[..., Any] = llm_http_client,
        sleep_func: Callable[[float], None] = time.sleep,
        max_retries: int = 7,
        provider_flags: Mapping[str, bool] | None = None,
        execution_settings: Any | None = None,
    ) -> None:
        configured_flags = {
            EXA_PROVIDER: api_keys.provider_enabled(api_keys.EXA_POOL),
            GEMMA4_PROVIDER: api_keys.provider_enabled(
                api_keys.GEMMA4_GROUNDED_PROVIDER
            ),
            TAVILY_PROVIDER: api_keys.provider_enabled(api_keys.TAVILY_POOL),
        }
        if provider_flags is not None:
            configured_flags.update(
                {str(name): bool(enabled) for name, enabled in provider_flags.items()}
            )
        if execution_settings is None:
            from .routing.execution_policy import load_execution_settings

            execution_settings = load_execution_settings()
        # agent-only forbids Gemini model APIs. Exa/Tavily remain valid
        # retrieval services, but Gemma4 grounded is generateContent and must
        # not sneak around the execution backend policy as a search fallback.
        if execution_settings.policy_id == "agent-only":
            configured_flags[GEMMA4_PROVIDER] = False
        self.provider_flags = configured_flags

        if exa_keys is None:
            self.exa_entries = (
                load_search_api_key_entries(EXA_KEYS_ENV)
                if self.provider_flags[EXA_PROVIDER]
                else []
            )
        else:
            self.exa_entries = _select_pool(
                self._entries_for(exa_keys, EXA_KEYS_ENV),
                pool_selector=list(exa_pool or ()),
                pool_size=exa_pool_size,
            )
        self.exa_pool = ApiKeyPool(
            "exa",
            self.exa_entries,
            state_path=state_path,
            lock_seconds=exa_lock_seconds,
            now_func=now_func,
        )
        from .routing.config import GEMINI_FREE_TIER

        if gemma_keys is None:
            self.gemma_entries = (
                load_search_api_key_entries(GEMINI_FREE_TIER)
                if self.provider_flags[GEMMA4_PROVIDER]
                else []
            )
        else:
            self.gemma_entries = _select_pool(
                self._entries_for(gemma_keys, GEMINI_FREE_TIER),
                pool_selector=(),
                pool_size=gemma_pool_size,
            )
        self.gemma_pool = ApiKeyPool(
            GEMMA4_PROVIDER,
            self.gemma_entries,
            state_path=state_path,
            lock_seconds=gemma_lock_seconds,
            now_func=now_func,
        )
        if gemma_rate_limiter is None:
            from .rate_limit import ModelRateLimiter

            gemma_rate_limiter = ModelRateLimiter(state_path=state_path)
        self.gemma_rate_limiter = gemma_rate_limiter
        self.gemma_timeout_seconds = float(gemma_timeout_seconds)
        if tavily_keys is None:
            self.tavily_entries = (
                load_search_api_key_entries(TAVILY_KEYS_ENV)
                if self.provider_flags[TAVILY_PROVIDER]
                else []
            )
        else:
            self.tavily_entries = _select_pool(
                self._entries_for(tavily_keys, TAVILY_KEYS_ENV),
                pool_selector=list(tavily_pool or ()),
                pool_size=tavily_pool_size,
            )
        self.tavily_pool = ApiKeyPool(
            "tavily",
            self.tavily_entries,
            state_path=state_path,
            lock_seconds=tavily_lock_seconds,
            now_func=now_func,
        )
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.query_interval_seconds = query_interval_seconds
        self.client_factory = client_factory
        self.sleep_func = sleep_func
        self.max_retries = int(max_retries)
        self._last_query_at: float | None = None
        # Serializes _pace across threads (parallel query rounds share one
        # client): the read-sleep-write on _last_query_at is a check-then-act.
        self._pace_lock = threading.Lock()

    @staticmethod
    def _entries_for(
        keys: Optional[Sequence[str]], env_name: str
    ) -> list[SearchApiKey]:
        if keys is not None:
            return [SearchApiKey(key_id=_key_id_for_secret(key), key=key) for key in keys]
        return load_search_api_key_entries(env_name)

    #: How much of a query or URL a log line carries. Long enough to recognise
    #: which request this was, short enough that the line stays one line.
    _LABEL_CHARS = 120

    @classmethod
    def _one_line(cls, text: str) -> str:
        """Collapse, redact, trim -- in that order.

        A provider error quotes the request URL, and a user's own endpoint or
        proxy may be written `https://user:token@host`. This file exists to be
        sent to a developer, so it follows the same rule `doctor` output does.
        """

        collapsed = " ".join(redact_credentials(str(text)).split())
        if len(collapsed) <= cls._LABEL_CHARS:
            return collapsed
        return collapsed[: cls._LABEL_CHARS - 1] + "…"

    def _report_request(
        self,
        provider: str,
        label: str,
        status: str,
        elapsed: float | None,
        attempt: int,
    ) -> None:
        """One line per provider request: who, what for, how it went.

        Status plus a one-line description, deliberately -- result bodies and
        extracted page text belong in the artifacts, not in the file a user
        sends when something went wrong. Both halves are trimmed: a provider
        that answers an error with a page of HTML would otherwise put all of it
        on this line.
        """

        fields: dict[str, object] = {"provider": provider, "status": self._one_line(status)}
        if label:
            fields["for"] = self._one_line(label)
        if elapsed is not None:
            fields["sec"] = f"{elapsed:.2f}"
        if attempt:
            fields["retry"] = attempt
        current_reporter().debug("web search request", fields)

    def _try_pool(
        self,
        pool: ApiKeyPool,
        entries: Sequence[SearchApiKey],
        provider: str,
        call: Callable[[str], Any],
        *,
        errors: List[str],
        fallbacks: List[SearchFallbackEvent],
        label: str = "",
    ) -> Any | None:
        """Run ``call(api_key)`` against a key pool; None means fall through.

        A provider with no configured keys is skipped silently (no fallback
        event). Auth/quota errors lock the key and continue to the next; a
        transient provider error records the event and stops the pool.

        Every provider request in the harness passes through here -- search and
        extract, one-by-one and batched -- so this is where the run log learns
        that retrieval happened at all. ``label`` is what the request was for
        (the query, or the URL being read): short by nature, unlike a prompt,
        and the one detail that makes the line worth reading.
        """

        if not entries:
            return None
        if not pool.available_entries():
            fallbacks.append(
                SearchFallbackEvent(
                    provider=provider,
                    reason="all_pool_keys_locked",
                    detail=",".join(pool.locked_key_ids()),
                )
            )
            return None
        for entry in pool.available_entries():
            for attempt in range(self.max_retries + 1):  # 1 original + max_retries
                try:
                    started = time.monotonic()
                    result = call(entry.key)
                    pool.mark_used(entry)
                    self._report_request(
                        provider, label, "ok", time.monotonic() - started, attempt
                    )
                    return result
                except _KeyUnusableError as exc:
                    self._report_request(
                        provider, label, f"key-unusable: {exc}", None, attempt
                    )
                    if attempt >= self.max_retries:
                        errors.append(f"{provider}: {exc}")
                        pool.lock(entry, str(exc))
                        fallbacks.append(
                            SearchFallbackEvent(
                                provider=provider,
                                reason="key_locked",
                                detail=str(exc),
                                key_id=entry.key_id,
                            )
                        )
                        break  # proceed to next key
                    self.sleep_func(0.5 * (2**attempt))
                except Exception as exc:
                    self._report_request(
                        provider, label, f"{type(exc).__name__}: {exc}", None, attempt
                    )
                    if attempt >= self.max_retries:
                        errors.append(f"{provider}: {exc}")
                        fallbacks.append(
                            SearchFallbackEvent(
                                provider=provider,
                                reason="provider_error",
                                detail=str(exc),
                                key_id=entry.key_id,
                            )
                        )
                        return None  # stop pool search (equivalent to break of outer loop)
                    self.sleep_func(0.5 * (2**attempt))
        return None

    def search(self, query: str, guided_query: str = "") -> QuerySearchResult:
        query = (query or "").strip()
        if not query:
            return QuerySearchResult(query=query, error="empty query")
        results = self.search_many([SearchRequest(query=query, guided_query=guided_query)])
        return results[0] if results else QuerySearchResult(query=query, error="empty query")

    def extract(self, url: str, guided_query: str = "") -> QueryExtractResult:
        """Deep-extract a single URL's page content (Exa → Gemma4 → Tavily)."""

        url = (url or "").strip()
        if not url:
            return QueryExtractResult(url=url, error="empty url")
        results = self.extract_many([ExtractRequest(url=url, guided_query=guided_query)])
        return results[0] if results else QueryExtractResult(url=url, error="empty url")

    def search_many(
        self,
        queries: Sequence["str | SearchRequest"],
        *,
        max_queries: int | None = None,
    ) -> List[QuerySearchResult]:
        """Search deduplicated queries in order, keeping at most ``max_queries``.

        Items may be plain strings or :class:`SearchRequest` (query + optional
        guided query). Dedup is by ``(query, guided query)``: the guided query
        changes what the provider highlights, so the same keywords aimed at a
        different question are a second request, not a repeat. Results come
        back one per surviving request, in order, each stamped with its guided
        query.
        """

        seen: set[tuple[str, str]] = set()
        selected: List[SearchRequest] = []
        for item in queries:
            request = _as_search_request(item)
            normalized = (request.query.lower(), guided_query_key(request.guided_query))
            if not request.query or normalized in seen:
                continue
            seen.add(normalized)
            selected.append(request)
            if max_queries is not None and len(selected) >= max_queries:
                break
        results: list[QuerySearchResult | None] = [None] * len(selected)
        errors: list[list[str]] = [[] for _ in selected]
        fallbacks: list[list[SearchFallbackEvent]] = [[] for _ in selected]

        pending = list(range(len(selected)))
        for provider in SEARCH_PROVIDER_ORDER:
            if not pending:
                break
            if not self.provider_flags.get(provider, True):
                continue
            pending = self._search_pending_with_provider(
                provider,
                pending,
                selected,
                results,
                errors,
                fallbacks,
            )
        for idx in pending:
            results[idx] = QuerySearchResult(
                query=selected[idx].query,
                error="; ".join(errors[idx]) or "no provider available",
                fallbacks=tuple(fallbacks[idx]),
            )
        # Providers build results without knowing the focus hint; stamp it
        # here, where request and result are still index-aligned, so nothing
        # downstream has to reconstruct that pairing by position.
        return [
            replace(result, guided_query=selected[idx].guided_query)
            for idx, result in enumerate(results)
            if result is not None
        ]

    def extract_many(
        self,
        requests: Sequence[ExtractRequest],
        *,
        max_urls: int | None = None,
    ) -> List[QueryExtractResult]:
        """Extract deduplicated URLs in order, keeping at most ``max_urls``.

        Dedup is by ``(url, guided query)``: re-reading one page for a
        different purpose returns different content, so it is a second
        request. See :meth:`search_many`.
        """

        seen: set[tuple[str, str]] = set()
        selected: List[ExtractRequest] = []
        for request in requests:
            url = (request.url or "").strip()
            key = (url, guided_query_key(request.guided_query))
            if not url or key in seen:
                continue
            seen.add(key)
            selected.append(ExtractRequest(url=url, guided_query=request.guided_query))
            if max_urls is not None and len(selected) >= max_urls:
                break
        results: list[QueryExtractResult | None] = [None] * len(selected)
        errors: list[list[str]] = [[] for _ in selected]
        fallbacks: list[list[SearchFallbackEvent]] = [[] for _ in selected]

        pending = list(range(len(selected)))
        for provider in EXTRACT_PROVIDER_ORDER:
            if not pending:
                break
            if not self.provider_flags.get(provider, True):
                continue
            pending = self._extract_pending_with_provider(
                provider,
                pending,
                selected,
                results,
                errors,
                fallbacks,
            )
        for idx in pending:
            results[idx] = QueryExtractResult(
                url=selected[idx].url,
                error="; ".join(errors[idx]) or "no extract provider available",
                fallbacks=tuple(fallbacks[idx]),
            )
        # See ``search_many``: stamp the focus hint while indexes still align.
        return [
            replace(result, guided_query=selected[idx].guided_query)
            for idx, result in enumerate(results)
            if result is not None
        ]

    def _search_pending_with_provider(
        self,
        provider: str,
        pending: Sequence[int],
        selected: Sequence[SearchRequest],
        results: list[QuerySearchResult | None],
        errors: list[list[str]],
        fallbacks: list[list[SearchFallbackEvent]],
    ) -> list[int]:
        if provider == EXA_PROVIDER:
            return self._search_pending_one_by_one(
                provider,
                pending,
                selected,
                results,
                errors,
                fallbacks,
                lambda request, key: self._exa_search(
                    request.query, request.guided_query, api_key=key
                ),
                self.exa_pool,
                self.exa_entries,
            )
        if provider == GEMMA4_PROVIDER:
            return self._search_pending_gemma4(
                pending, selected, results, errors, fallbacks
            )
        if provider == TAVILY_PROVIDER:
            return self._search_pending_one_by_one(
                provider,
                pending,
                selected,
                results,
                errors,
                fallbacks,
                lambda request, key: self._tavily_search(request.query, api_key=key),
                self.tavily_pool,
                self.tavily_entries,
            )
        raise ValueError(f"Unknown search provider '{provider}'")

    def _search_pending_one_by_one(
        self,
        provider: str,
        pending: Sequence[int],
        selected: Sequence[SearchRequest],
        results: list[QuerySearchResult | None],
        errors: list[list[str]],
        fallbacks: list[list[SearchFallbackEvent]],
        call: Callable[[SearchRequest, str], QuerySearchResult],
        pool: ApiKeyPool,
        entries: Sequence[SearchApiKey],
    ) -> list[int]:
        next_pending: list[int] = []
        for idx in pending:
            self._pace()
            result = self._try_pool(
                pool,
                entries,
                provider,
                lambda key, req=selected[idx]: call(req, key),
                errors=errors[idx],
                fallbacks=fallbacks[idx],
                label=_request_text(selected[idx]),
            )
            if result is not None:
                results[idx] = _with_fallbacks(result, fallbacks[idx])
            else:
                next_pending.append(idx)
        return next_pending

    def _search_pending_gemma4(
        self,
        pending: Sequence[int],
        selected: Sequence[SearchRequest],
        results: list[QuerySearchResult | None],
        errors: list[list[str]],
        fallbacks: list[list[SearchFallbackEvent]],
    ) -> list[int]:
        next_pending: list[int] = []
        for batch in _chunks(list(pending), GEMMA4_SEARCH_BATCH_QUERY_LIMIT):
            self._pace()
            batch_errors: list[str] = []
            batch_fallbacks: list[SearchFallbackEvent] = []
            batch_requests = [selected[idx] for idx in batch]
            batch_results = self._try_pool(
                self.gemma_pool,
                self.gemma_entries,
                GEMMA4_PROVIDER,
                lambda key, reqs=batch_requests: self._gemma4_search_many(
                    reqs, api_key=key
                ),
                errors=batch_errors,
                fallbacks=batch_fallbacks,
                label=" | ".join(_request_text(req) for req in batch_requests),
            )
            if batch_results is not None:
                for idx, result in zip(batch, batch_results):
                    results[idx] = _with_fallbacks(
                        result, [*fallbacks[idx], *batch_fallbacks]
                    )
            else:
                for idx in batch:
                    errors[idx].extend(batch_errors)
                    fallbacks[idx].extend(batch_fallbacks)
                    next_pending.append(idx)
        return next_pending

    def _extract_pending_with_provider(
        self,
        provider: str,
        pending: Sequence[int],
        selected: Sequence[ExtractRequest],
        results: list[QueryExtractResult | None],
        errors: list[list[str]],
        fallbacks: list[list[SearchFallbackEvent]],
    ) -> list[int]:
        if provider == EXA_PROVIDER:
            return self._extract_pending_one_by_one(
                provider,
                pending,
                selected,
                results,
                errors,
                fallbacks,
                lambda request, key: self._exa_extract(
                    request.url, request.guided_query, api_key=key
                ),
                self.exa_pool,
                self.exa_entries,
            )
        if provider == GEMMA4_PROVIDER:
            return self._extract_pending_gemma4(
                pending, selected, results, errors, fallbacks
            )
        if provider == TAVILY_PROVIDER:
            return self._extract_pending_one_by_one(
                provider,
                pending,
                selected,
                results,
                errors,
                fallbacks,
                lambda request, key: self._tavily_extract(
                    request.url, request.guided_query, api_key=key
                ),
                self.tavily_pool,
                self.tavily_entries,
            )
        raise ValueError(f"Unknown extract provider '{provider}'")

    def _extract_pending_one_by_one(
        self,
        provider: str,
        pending: Sequence[int],
        selected: Sequence[ExtractRequest],
        results: list[QueryExtractResult | None],
        errors: list[list[str]],
        fallbacks: list[list[SearchFallbackEvent]],
        call: Callable[[ExtractRequest, str], QueryExtractResult],
        pool: ApiKeyPool,
        entries: Sequence[SearchApiKey],
    ) -> list[int]:
        next_pending: list[int] = []
        for idx in pending:
            self._pace()
            result = self._try_pool(
                pool,
                entries,
                provider,
                lambda key, req=selected[idx]: call(req, key),
                errors=errors[idx],
                fallbacks=fallbacks[idx],
                label=_request_text(selected[idx]),
            )
            if result is not None:
                results[idx] = _extract_with_fallbacks(result, fallbacks[idx])
            else:
                next_pending.append(idx)
        return next_pending

    def _extract_pending_gemma4(
        self,
        pending: Sequence[int],
        selected: Sequence[ExtractRequest],
        results: list[QueryExtractResult | None],
        errors: list[list[str]],
        fallbacks: list[list[SearchFallbackEvent]],
    ) -> list[int]:
        if not pending:
            return []
        self._pace()
        batch_errors: list[str] = []
        batch_fallbacks: list[SearchFallbackEvent] = []
        batch_requests = [selected[idx] for idx in pending]
        batch_results = self._try_pool(
            self.gemma_pool,
            self.gemma_entries,
            GEMMA4_PROVIDER,
            lambda key, reqs=batch_requests: self._gemma4_extract_many(
                reqs, api_key=key
            ),
            errors=batch_errors,
            fallbacks=batch_fallbacks,
            label=" | ".join(_request_text(req) for req in batch_requests),
        )
        if batch_results is not None:
            for idx, result in zip(pending, batch_results):
                results[idx] = _extract_with_fallbacks(
                    result, [*fallbacks[idx], *batch_fallbacks]
                )
            return []
        for idx in pending:
            errors[idx].extend(batch_errors)
            fallbacks[idx].extend(batch_fallbacks)
        return list(pending)

    def _pace(self) -> None:
        # Sleeping under the lock is the point: concurrent callers queue up
        # and depart one interval apart instead of all reading the same
        # _last_query_at and firing together.
        with self._pace_lock:
            now = time.monotonic()
            if self._last_query_at is not None:
                wait = self.query_interval_seconds - (now - self._last_query_at)
                if wait > 0:
                    self.sleep_func(wait)
            self._last_query_at = time.monotonic()

    def _post_json(
        self,
        url: str,
        *,
        payload: Mapping[str, Any],
        api_key: str,
        headers: Mapping[str, str],
        key_error_statuses: set[int],
    ) -> dict[str, Any]:
        with self.client_factory(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=dict(payload), headers=dict(headers))
        if response.status_code in key_error_statuses:
            body = response.text[:200].replace(api_key, "[redacted]")
            raise _KeyUnusableError(f"HTTP {response.status_code}: {body}")
        if response.status_code >= 400:
            body = response.text[:200].replace(api_key, "[redacted]")
            raise RuntimeError(f"HTTP {response.status_code}: {body}")
        data = response.json()
        return data if isinstance(data, Mapping) else {}

    def _exa_search(
        self, query: str, guided_query: str, *, api_key: str
    ) -> QuerySearchResult:
        highlight_query = (guided_query or query).strip()
        payload = {
            "query": query,
            "numResults": self.max_results,
            "type": "deep",
            "contents": {
                "highlights": {"query": highlight_query},
                "summary": True,
                "maxAgeHours": EXA_MAX_AGE_HOURS,
            },
        }
        data = self._post_json(
            EXA_SEARCH_URL,
            payload=payload,
            api_key=api_key,
            headers={"x-api-key": api_key},
            key_error_statuses=_EXA_KEY_ERROR_STATUSES,
        )
        items = tuple(
            SearchResultItem(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=_exa_snippet(item),
            )
            for item in data.get("results", [])
            if isinstance(item, Mapping)
        )
        return QuerySearchResult(query=query, provider="exa", items=items)

    def _tavily_search(self, query: str, *, api_key: str) -> QuerySearchResult:
        payload = {
            "query": query,
            "auto_parameters": True,
            "include_answer": "advanced",
            "max_results": self.max_results,
        }
        data = self._post_json(
            TAVILY_SEARCH_URL,
            payload=payload,
            api_key=api_key,
            headers={"Authorization": f"Bearer {api_key}"},
            key_error_statuses=_TAVILY_KEY_ERROR_STATUSES,
        )
        items = tuple(
            SearchResultItem(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
            )
            for item in data.get("results", [])
            if isinstance(item, Mapping)
        )
        return QuerySearchResult(
            query=query,
            provider="tavily",
            items=items,
            answer=str(data.get("answer") or ""),
        )

    def _exa_extract(
        self, url: str, guided_query: str, *, api_key: str
    ) -> QueryExtractResult:
        payload: dict[str, Any] = {
            "ids": [url],
            "summary": True,
            "maxAgeHours": EXA_MAX_AGE_HOURS,
        }
        guided = (guided_query or "").strip()
        if guided:
            payload["highlights"] = {"query": guided}
        data = self._post_json(
            EXA_CONTENTS_URL,
            payload=payload,
            api_key=api_key,
            headers={"x-api-key": api_key},
            key_error_statuses=_EXA_KEY_ERROR_STATUSES,
        )
        item = next(
            (row for row in data.get("results", []) if isinstance(row, Mapping)), {}
        )
        return QueryExtractResult(
            url=url,
            provider="exa",
            title=str(item.get("title", "")),
            content=_exa_snippet(item),
        )

    def _tavily_extract(
        self, url: str, guided_query: str, *, api_key: str
    ) -> QueryExtractResult:
        payload: dict[str, Any] = {
            "urls": [url],
            "chunks_per_source": TAVILY_EXTRACT_CHUNKS_PER_SOURCE,
        }
        guided = (guided_query or "").strip()
        if guided:
            payload["query"] = guided
        data = self._post_json(
            TAVILY_EXTRACT_URL,
            payload=payload,
            api_key=api_key,
            headers={"Authorization": f"Bearer {api_key}"},
            key_error_statuses=_TAVILY_KEY_ERROR_STATUSES,
        )
        item = next(
            (row for row in data.get("results", []) if isinstance(row, Mapping)), None
        )
        if item is None:
            failed = data.get("failed_results") or []
            reason = ""
            if failed and isinstance(failed[0], Mapping):
                reason = str(failed[0].get("error", ""))
            raise RuntimeError(
                "tavily extract returned no content" + (f": {reason}" if reason else "")
            )
        content = str(item.get("raw_content") or item.get("content") or "")
        return QueryExtractResult(url=url, provider="tavily", content=content)

    def _gemma4_generate_grounded(self, prompt: str, *, api_key: str) -> dict[str, Any]:
        from .routing.config import GEMINI_FREE_TIER, ModelEndpoint
        from .token_budget import default_token_counter

        endpoint = ModelEndpoint(GEMINI_FREE_TIER, GEMMA4_MODEL)
        estimated_input = default_token_counter().count_text(prompt)
        rate_ticket = self.gemma_rate_limiter.acquire(endpoint, estimated_input)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "maxOutputTokens": GEMMA4_MAX_OUTPUT_TOKENS,
                "temperature": 0.2,
            },
        }
        with self.client_factory(timeout=self.gemma_timeout_seconds) as client:
            response = client.post(
                GEMMA4_GENERATE_URL,
                json=payload,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
            )
        if response.status_code in _GEMMA4_KEY_ERROR_STATUSES:
            body = response.text[:200].replace(api_key, "[redacted]")
            raise _KeyUnusableError(f"HTTP {response.status_code}: {body}")
        if response.status_code >= 400:
            body = response.text[:200].replace(api_key, "[redacted]")
            raise RuntimeError(f"HTTP {response.status_code}: {body}")
        data = response.json()
        if not isinstance(data, Mapping):
            raise RuntimeError("Gemma4 returned non-object response")
        actual_input = _gemma4_prompt_tokens(data)
        self.gemma_rate_limiter.settle(
            endpoint,
            actual_input_tokens=actual_input or estimated_input,
            estimated_input_tokens=estimated_input,
            ticket=rate_ticket,
        )
        return dict(data)

    def _gemma4_search_many(
        self, requests: Sequence[SearchRequest], *, api_key: str
    ) -> list[QuerySearchResult]:
        if not requests:
            return []
        response = self._gemma4_generate_grounded(
            _gemma4_search_prompt(requests), api_key=api_key
        )
        text = _gemma4_response_text(response)
        grounding = gemini_grounding_metadata(response)
        chunks = gemini_grounding_chunks(grounding)
        supports = _gemma4_grounding_supports(grounding)
        retried_without_visible_thinking = False
        if not chunks:
            response = self._gemma4_generate_grounded(
                _gemma4_search_prompt(requests, include_thinking=False),
                api_key=api_key,
            )
            text = _gemma4_response_text(response)
            grounding = gemini_grounding_metadata(response)
            chunks = gemini_grounding_chunks(grounding)
            supports = _gemma4_grounding_supports(grounding)
            retried_without_visible_thinking = True
        if not chunks:
            raise RuntimeError("Gemma4 returned no usable grounding chunks")
        rows = _gemma4_rows_by_request(
            text,
            tag="gemma4_search_results",
            row_key="results",
            expected_ids=[f"q{idx + 1}" for idx in range(len(requests))],
        )
        metadata_base = _gemma4_metadata(
            grounding,
            chunks,
            supports,
            retried_without_visible_thinking=retried_without_visible_thinking,
        )
        results: list[QuerySearchResult] = []
        for idx, request in enumerate(requests):
            row = rows.get(f"q{idx + 1}")
            if not row:
                # The model answered some of the batch and not this one. Saying
                # so is what lets the caller fall through to Tavily for it:
                # a result with no `error` counts as success, drops out of
                # `pending`, and never reaches another provider. Worse, an
                # absent row used to take the "no sources listed" path and
                # inherit *the whole batch's* grounding chunks, so one query's
                # sources were served as another's -- with a summary that was
                # really the raw model text.
                results.append(
                    QuerySearchResult(
                        query=request.query,
                        provider=GEMMA4_PROVIDER,
                        items=(),
                        answer="",
                        error="gemma4 returned no row for this query",
                        metadata=dict(metadata_base),
                    )
                )
                continue
            summary = str(row.get("summary") or row.get("answer") or "").strip()
            source_indices = _gemma4_source_indices(row, chunks)
            items = _gemma4_items_for_indices(
                chunks,
                supports,
                source_indices,
                fallback_snippet=summary or text,
            )
            results.append(
                QuerySearchResult(
                    query=request.query,
                    provider=GEMMA4_PROVIDER,
                    items=tuple(items),
                    answer=summary,
                    metadata={
                        **metadata_base,
                        "model_row": _compact_model_row(row),
                    },
                )
            )
        return results

    def _gemma4_extract_many(
        self, requests: Sequence[ExtractRequest], *, api_key: str
    ) -> list[QueryExtractResult]:
        if not requests:
            return []
        response = self._gemma4_generate_grounded(
            _gemma4_extract_prompt(requests), api_key=api_key
        )
        text = _gemma4_response_text(response)
        grounding = gemini_grounding_metadata(response)
        chunks = gemini_grounding_chunks(grounding)
        supports = _gemma4_grounding_supports(grounding)
        retried_without_visible_thinking = False
        if not chunks:
            response = self._gemma4_generate_grounded(
                _gemma4_extract_prompt(requests, include_thinking=False),
                api_key=api_key,
            )
            text = _gemma4_response_text(response)
            grounding = gemini_grounding_metadata(response)
            chunks = gemini_grounding_chunks(grounding)
            supports = _gemma4_grounding_supports(grounding)
            retried_without_visible_thinking = True
        if not chunks:
            raise RuntimeError("Gemma4 returned no usable grounding chunks")
        rows = _gemma4_rows_by_request(
            text,
            tag="gemma4_extract_results",
            row_key="results",
            expected_ids=[f"u{idx + 1}" for idx in range(len(requests))],
        )
        metadata_base = _gemma4_metadata(
            grounding,
            chunks,
            supports,
            retried_without_visible_thinking=retried_without_visible_thinking,
        )
        results: list[QueryExtractResult] = []
        for idx, request in enumerate(requests):
            row = rows.get(f"u{idx + 1}")
            if not row:
                # Same reasoning as the search path: without an error this
                # counts as a successful extract, so the URL never falls
                # through to another provider -- and `content or text` handed
                # back the entire raw model response, which for a URL the model
                # never discussed is another page's content under this one's
                # heading. That goes into the context pack and, via
                # knowledge_hints, into the knowledge base.
                results.append(
                    QueryExtractResult(
                        url=request.url,
                        provider=GEMMA4_PROVIDER,
                        error="gemma4 returned no row for this url",
                        metadata={
                            **metadata_base,
                            "decoded_url": unquote(request.url),
                        },
                    )
                )
                continue
            content = str(
                row.get("content")
                or row.get("summary")
                or row.get("answer")
                or ""
            ).strip()
            source_indices = _gemma4_source_indices(row, chunks)
            support_text = _gemma4_support_text_for_indices(supports, source_indices)
            if support_text:
                content = "\n".join(part for part in (content, support_text) if part)
            title = str(row.get("title") or "").strip()
            if not title and source_indices:
                title = chunks[source_indices[0]].get("title", "")
            results.append(
                QueryExtractResult(
                    url=request.url,
                    provider=GEMMA4_PROVIDER,
                    title=title,
                    content=content or text,
                    metadata={
                        **metadata_base,
                        "decoded_url": unquote(request.url),
                        "model_row": _compact_model_row(row),
                    },
                )
            )
        return results

def _exa_snippet(item: Mapping[str, Any]) -> str:
    """Clean an Exa search/contents row to text (summary + highlights).

    Only the textual fields are kept; image/favicon URLs are intentionally
    dropped so they never reach the model.
    """

    parts: List[str] = []
    summary = str(item.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    highlights = item.get("highlights")
    if isinstance(highlights, (list, tuple)):
        for highlight in highlights:
            text = str(highlight or "").strip()
            if text:
                parts.append(text)
    if not parts:
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _json_dumps_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _gemma4_search_prompt(
    requests: Sequence[SearchRequest], *, include_thinking: bool = True
) -> str:
    rows = [
        {
            "id": f"q{idx + 1}",
            "query": request.query,
            "search_goal": request.guided_query or "",
        }
        for idx, request in enumerate(requests)
    ]
    thinking = (
        f"{GEMMA4_THINKING_PREFIX}\nUse medium-depth thinking.\n"
        if include_thinking
        else ""
    )
    return (
        thinking
        + "Search the web for each query in this JSON array. If search_goal is non-empty, "
        "focus the summary on that goal. First answer each item in a numbered list with "
        "source names. Then include compact JSON inside <gemma4_search_results> tags "
        "with this shape: "
        '{"results":[{"id":"q1","query":"...","summary":"...","sources":[{"title":"...","url":"..."}]}]}。\n'
        f"Queries JSON:\n{_json_dumps_compact(rows)}"
    )


def _gemma4_extract_prompt(
    requests: Sequence[ExtractRequest], *, include_thinking: bool = True
) -> str:
    rows = [
        {
            "id": f"u{idx + 1}",
            "url": unquote(request.url),
            "original_url": request.url,
            "extract_goal": request.guided_query or "",
        }
        for idx, request in enumerate(requests)
    ]
    thinking = (
        f"{GEMMA4_THINKING_PREFIX}\nUse medium-depth thinking.\n"
        if include_thinking
        else ""
    )
    return (
        thinking
        + "Search the web for each URL in this JSON array. The url field has percent-escapes "
        "decoded into standard characters; copy and search that decoded URL to avoid transcription "
        "errors. If extract_goal is non-empty, focus extraction on it. Return concise extracted "
        "content as JSON inside <gemma4_extract_results> tags with this shape: "
        '{"results":[{"id":"u1","url":"...","title":"...","content":"...","sources":[{"title":"...","url":"..."}]}]}。\n'
        f"URLs JSON:\n{_json_dumps_compact(rows)}"
    )


def _gemma4_prompt_tokens(response: Mapping[str, Any]) -> int:
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    if not isinstance(usage, Mapping):
        return 0
    value = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    return int(value) if isinstance(value, (int, float)) else 0


def _gemma4_response_text(response: Mapping[str, Any]) -> str:
    candidates = response.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, Mapping):
        return ""
    content = first.get("content")
    if not isinstance(content, Mapping):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, Sequence):
        return ""
    texts: list[str] = []
    for part in parts:
        if isinstance(part, Mapping) and part.get("text") is not None:
            texts.append(str(part.get("text") or ""))
    return "\n".join(texts).strip()


# The three readers below are about the *Gemini response shape*, not about the
# Gemma4 search provider that first needed them: `client.py` reads the same
# `groundingMetadata` off a `google_search` reply to build its URL ledger. One
# owner, so a shape change is found once.
def gemini_grounding_metadata(response: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = response.get("candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        return {}
    first = candidates[0]
    if not isinstance(first, Mapping):
        return {}
    metadata = first.get("groundingMetadata") or first.get("grounding_metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def gemini_grounding_chunks(metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_chunks = metadata.get("groundingChunks") or metadata.get("grounding_chunks") or []
    chunks: list[dict[str, str]] = []
    if not isinstance(raw_chunks, Sequence):
        return chunks
    for chunk in raw_chunks:
        if not isinstance(chunk, Mapping):
            continue
        web = chunk.get("web")
        if not isinstance(web, Mapping):
            continue
        url = str(web.get("uri") or web.get("url") or "").strip()
        title = str(web.get("title") or "").strip()
        if url or title:
            chunks.append({"url": url, "title": title})
    return chunks


def _gemma4_grounding_supports(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_supports = metadata.get("groundingSupports") or metadata.get("grounding_supports") or []
    supports: list[dict[str, Any]] = []
    if not isinstance(raw_supports, Sequence):
        return supports
    for support in raw_supports:
        if not isinstance(support, Mapping):
            continue
        segment = support.get("segment")
        text = ""
        if isinstance(segment, Mapping):
            text = str(segment.get("text") or "").strip()
        raw_indices = (
            support.get("groundingChunkIndices")
            or support.get("grounding_chunk_indices")
            or []
        )
        indices = [int(idx) for idx in raw_indices if isinstance(idx, (int, float))]
        if text or indices:
            supports.append({"text": text, "groundingChunkIndices": indices})
    return supports


def gemini_grounding_queries(metadata: Mapping[str, Any]) -> list[str]:
    """The queries the model actually ran, in the order it reports them."""

    raw_queries = (
        metadata.get("webSearchQueries") or metadata.get("web_search_queries") or []
    )
    if not isinstance(raw_queries, Sequence) or isinstance(raw_queries, (str, bytes)):
        return []
    return [str(item) for item in raw_queries]


def _gemma4_metadata(
    metadata: Mapping[str, Any],
    chunks: Sequence[Mapping[str, str]],
    supports: Sequence[Mapping[str, Any]],
    *,
    retried_without_visible_thinking: bool = False,
) -> dict[str, Any]:
    return {
        "web_search_queries": gemini_grounding_queries(metadata),
        "grounding_chunks": [dict(chunk) for chunk in chunks],
        "grounding_supports": [
            {
                "text": str(support.get("text") or ""),
                "groundingChunkIndices": list(support.get("groundingChunkIndices") or []),
            }
            for support in supports
        ],
        "retried_without_visible_thinking": retried_without_visible_thinking,
    }


def _extract_tagged_json(text: str, tag: str) -> Any:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}\s*>",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return json.loads(match.group(1).strip())
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text or ""):
        try:
            parsed, _ = decoder.raw_decode((text or "")[match.start() :])
            return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _gemma4_rows_by_request(
    text: str,
    *,
    tag: str,
    row_key: str,
    expected_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    try:
        parsed = _extract_tagged_json(text, tag)
    except json.JSONDecodeError:
        parsed = {}
    rows: Any = []
    if isinstance(parsed, Mapping):
        rows = parsed.get(row_key) or parsed.get("items") or []
    elif isinstance(parsed, list):
        rows = parsed
    by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, Sequence):
        # Position is only trustworthy when the model returned a row for every
        # request. A *partial* unlabelled answer -- rows for q1 and q3, say --
        # would otherwise be read as q1 and q2, serving one query's findings as
        # another's. Anything shorter is matched by explicit id only, and the
        # requests with no row are reported as failures so they can fall
        # through to the next provider.
        positional = len(rows) == len(expected_ids)
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            declared = str(row.get("id") or "").strip()
            row_id = declared or (expected_ids[idx] if positional else "")
            if row_id:
                by_id[row_id] = row
    return by_id


def _normalize_url_for_match(url: str) -> str:
    return unquote(url or "").rstrip("/").lower()


def _gemma4_source_indices(
    row: Mapping[str, Any],
    chunks: Sequence[Mapping[str, str]],
) -> list[int]:
    sources = row.get("sources") if isinstance(row, Mapping) else None
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return list(range(len(chunks)))
    wanted_urls: list[str] = []
    wanted_titles: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        url = str(source.get("url") or source.get("uri") or "").strip()
        title = str(source.get("title") or "").strip()
        if url:
            wanted_urls.append(_normalize_url_for_match(url))
        if title:
            wanted_titles.append(title.lower())
    indices: list[int] = []
    for idx, chunk in enumerate(chunks):
        chunk_url = _normalize_url_for_match(str(chunk.get("url") or ""))
        chunk_title = str(chunk.get("title") or "").lower()
        if (
            chunk_url
            and any(url == chunk_url or url in chunk_url or chunk_url in url for url in wanted_urls)
        ) or (chunk_title and any(title in chunk_title or chunk_title in title for title in wanted_titles)):
            indices.append(idx)
    return indices or list(range(len(chunks)))


def _gemma4_support_text_for_indices(
    supports: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> str:
    wanted = set(indices)
    parts: list[str] = []
    seen: set[str] = set()
    for support in supports:
        support_indices = set(support.get("groundingChunkIndices") or [])
        text = str(support.get("text") or "").strip()
        if text and wanted.intersection(support_indices) and text not in seen:
            seen.add(text)
            parts.append(text)
    return "\n".join(parts)


def _gemma4_items_for_indices(
    chunks: Sequence[Mapping[str, str]],
    supports: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    fallback_snippet: str,
) -> list[SearchResultItem]:
    items: list[SearchResultItem] = []
    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        snippet = _gemma4_support_text_for_indices(supports, [idx]) or fallback_snippet
        items.append(
            SearchResultItem(
                title=str(chunk.get("title") or ""),
                url=str(chunk.get("url") or ""),
                snippet=snippet,
            )
        )
    return items


def _compact_model_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    compact: dict[str, Any] = {}
    for key in ("id", "query", "url", "title", "summary", "answer"):
        value = row.get(key)
        if value is not None:
            compact[key] = str(value)[:1000]
    sources = row.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        compact["sources"] = [
            {
                "title": str(source.get("title") or "")[:300],
                "url": str(source.get("url") or source.get("uri") or "")[:1000],
            }
            for source in sources
            if isinstance(source, Mapping)
        ][:10]
    return compact


class _KeyUnusableError(RuntimeError):
    """Provider key rejected (auth/quota); disable it and fall back."""


def _cap_soft(
    text: str, limit: int, count_tokens: Callable[[str], int]
) -> str:
    from .token_truncate import cap_tokens

    return cap_tokens((text or "").strip(), limit, count_tokens, marker="…")


def search_result_sections(
    results: Sequence[QuerySearchResult],
    *,
    count_tokens: Callable[[str], int],
    max_snippet_tokens: int = SEARCH_SNIPPET_MAX_TOKENS,
) -> List[tuple[str, str]]:
    """Per-query ``(label, text)`` sections.

    The label is the request identity (``query`` or ``query >> guided``), so
    a caller can tell which request a rendered section answers without
    matching by position. The section body still shows the bare query.
    """

    sections: List[tuple[str, str]] = []
    for result in results:
        lines = [f"--- query: {result.query} ---"]
        if result.error:
            lines.append(
                f"（搜索失败：{_cap_soft(result.error, _ERROR_TEXT_MAX_TOKENS, count_tokens)}）"
            )
        else:
            lines.append(f"provider: {result.provider}")
            if result.answer:
                lines.append(
                    f"answer: {_cap_soft(result.answer, max_snippet_tokens, count_tokens)}"
                )
            for item in result.items:
                head = " ".join(part for part in (item.title, f"({item.url})" if item.url else "") if part)
                if head:
                    lines.append(f"- {head}")
                snippet = _cap_soft(item.snippet, max_snippet_tokens, count_tokens)
                if snippet:
                    lines.append(f"  {snippet}")
        sections.append((result.request_label, "\n".join(lines)))
    return sections


def extract_result_sections(
    results: Sequence[QueryExtractResult],
    *,
    count_tokens: Callable[[str], int],
    max_content_tokens: int = EXTRACT_CONTENT_MAX_TOKENS,
) -> List[tuple[str, str]]:
    """Per-URL ``(label, text)`` deep-extract sections.

    Label is the request identity (``url`` or ``url >> guided``); see
    :func:`search_result_sections`.
    """

    sections: List[tuple[str, str]] = []
    for result in results:
        lines = [f"--- 深度提取 url: {result.url} ---"]
        if result.error:
            lines.append(
                f"（提取失败：{_cap_soft(result.error, _ERROR_TEXT_MAX_TOKENS, count_tokens)}）"
            )
        else:
            lines.append(f"provider: {result.provider}")
            if result.title:
                lines.append(f"title: {result.title}")
            content = _cap_soft(result.content, max_content_tokens, count_tokens)
            if content:
                lines.append(content)
        sections.append((result.request_label, "\n".join(lines)))
    return sections


def render_search_results(
    results: Sequence[QuerySearchResult],
    *,
    max_total_tokens: int,
    count_tokens: Callable[[str], int] | None = None,
    max_section_tokens: int | None = None,
    max_snippet_tokens: int = SEARCH_SNIPPET_MAX_TOKENS,
) -> "RenderedBlock":
    """Token-budgeted rendering grouped by query, for prompt injection.

    Returns a :class:`~finesub.llm.injection_budget.RenderedBlock`; use ``.text`` for
    the injection and ``.report()`` for artifacts. ``included``/``truncated``/
    ``dropped`` carry query strings.
    """

    from .routing.config import INJECTION_SECTION_MAX_TOKENS
    from .injection_budget import EMPTY_BLOCK, render_budgeted_block

    if not results:
        return EMPTY_BLOCK
    if count_tokens is None:
        from .token_budget import default_token_counter

        count_tokens = default_token_counter().count_text
    return render_budgeted_block(
        search_result_sections(
            results, count_tokens=count_tokens, max_snippet_tokens=max_snippet_tokens
        ),
        count_tokens=count_tokens,
        section_limit=(
            max_section_tokens if max_section_tokens is not None else INJECTION_SECTION_MAX_TOKENS
        ),
        block_limit=max_total_tokens,
    )


def render_extract_results(
    results: Sequence[QueryExtractResult],
    *,
    max_total_tokens: int,
    count_tokens: Callable[[str], int] | None = None,
    max_section_tokens: int | None = None,
    max_content_tokens: int = EXTRACT_CONTENT_MAX_TOKENS,
) -> "RenderedBlock":
    """Token-budgeted rendering of deep-extract results grouped by URL."""

    from .routing.config import INJECTION_SECTION_MAX_TOKENS
    from .injection_budget import EMPTY_BLOCK, render_budgeted_block

    if not results:
        return EMPTY_BLOCK
    if count_tokens is None:
        from .token_budget import default_token_counter

        count_tokens = default_token_counter().count_text
    return render_budgeted_block(
        extract_result_sections(
            results, count_tokens=count_tokens, max_content_tokens=max_content_tokens
        ),
        count_tokens=count_tokens,
        section_limit=(
            max_section_tokens if max_section_tokens is not None else INJECTION_SECTION_MAX_TOKENS
        ),
        block_limit=max_total_tokens,
    )


def search_results_metadata(results: Sequence[QuerySearchResult]) -> List[dict[str, Any]]:
    """Compact per-query metadata for task artifacts (no full snippets)."""

    rows: List[dict[str, Any]] = []
    for result in results:
        row = {
            "query": result.query,
            "provider": result.provider,
            "item_count": len(result.items),
            "has_answer": bool(result.answer),
            "error": result.error,
            "urls": [item.url for item in result.items if item.url][:10],
            "fallbacks": [event.to_dict() for event in result.fallbacks],
        }
        if result.metadata:
            row["provider_metadata"] = _compact_provider_metadata(result.metadata)
        rows.append(row)
    return rows


def extract_results_metadata(
    results: Sequence[QueryExtractResult],
) -> List[dict[str, Any]]:
    """Compact per-URL extract metadata for task artifacts (no full content)."""

    rows: List[dict[str, Any]] = []
    for result in results:
        row = {
            "url": result.url,
            "provider": result.provider,
            "content_chars": len(result.content),
            "error": result.error,
            "fallbacks": [event.to_dict() for event in result.fallbacks],
        }
        if result.metadata:
            row["provider_metadata"] = _compact_provider_metadata(result.metadata)
        rows.append(row)
    return rows


def _compact_provider_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    queries = metadata.get("web_search_queries")
    if isinstance(queries, Sequence) and not isinstance(queries, (str, bytes)):
        compact["web_search_queries"] = [str(item) for item in queries][:10]
    chunks = metadata.get("grounding_chunks")
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
        compact["grounding_chunks"] = [
            {
                "title": str(chunk.get("title") or "")[:300],
                "url": str(chunk.get("url") or "")[:1000],
            }
            for chunk in chunks
            if isinstance(chunk, Mapping)
        ][:10]
    supports = metadata.get("grounding_supports")
    if isinstance(supports, Sequence) and not isinstance(supports, (str, bytes)):
        compact["grounding_supports"] = [
            {
                "text": str(support.get("text") or "")[:500],
                "groundingChunkIndices": list(
                    support.get("groundingChunkIndices") or []
                )[:10],
            }
            for support in supports
            if isinstance(support, Mapping)
        ][:20]
    model_row = metadata.get("model_row")
    if isinstance(model_row, Mapping):
        compact["model_row"] = dict(model_row)
    decoded_url = metadata.get("decoded_url")
    if decoded_url:
        compact["decoded_url"] = str(decoded_url)
    if metadata.get("retried_without_visible_thinking"):
        compact["retried_without_visible_thinking"] = True
    return compact
