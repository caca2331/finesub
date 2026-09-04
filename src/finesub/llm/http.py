"""The one place an LLM-layer HTTP client is built.

Everything this package talks to over the network -- Gemini REST, the
OpenAI-compat / Anthropic transports, Exa / Gemma4 / Tavily, and the free
`countTokens` endpoint -- goes out through :func:`llm_http_client`, so "does
this request use the proxy?" has one answer that can be read off the code.

**Scope is API calls, deliberately.** The download family
(`finesub_bootstrap.http_client`) has its own route discovery -- Windows system
proxy, environment, a liveness probe on local ports -- and is not touched by
this setting. That separation is why the proxy is resolved into an explicit
`proxy=` argument rather than written into `os.environ`: the environment is
read by `apply_network_environment`, so an ambient proxy would silently
reroute downloads too.

⚠ **Turning a proxy on changes the failure modes**, not just the route. Gemini
scores datacenter and shared-exit IPs, and answers a suspicious one with a
block that `client.LLMIPRiskError` recognises. A proxy that works for search
can still make correction calls fail this way.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

#: `[llm] proxy = "direct"` means "ignore the environment too". Without it,
#: unsetting the key would still leave `HTTPS_PROXY` in charge, and there would
#: be no way to say "no proxy for LLM calls" on a machine that needs one for
#: everything else.
DIRECT = "direct"


@lru_cache(maxsize=1)
def configured_proxy() -> str | None:
    """`[llm] proxy` from config.toml: a URL, :data:`DIRECT`, or None.

    Memoized because it is read once per HTTP call. `config.read_config` is
    itself cached and stat-invalidated, but this avoids re-parsing the section
    on every request.
    """

    from ..config import config_str

    value = (config_str("llm", "proxy") or "").strip()
    return value or None


def reset_proxy_cache() -> None:
    """Forget the resolved value (config reloads, and tests)."""

    configured_proxy.cache_clear()


def client_kwargs() -> dict[str, Any]:
    """`proxy` / `trust_env` for one client, from the configured value.

    Three states, and the middle one is the reason this is not a bool:

    * key absent -- `trust_env=True`, i.e. `HTTPS_PROXY` still works. This is
      what the layer did before the setting existed, so adding it breaks nobody.
    * `proxy = "direct"` -- no proxy, and the environment is ignored.
    * `proxy = "<url>"` -- that proxy, and the environment is ignored. The file
      is more specific than the ambient environment, so it wins outright rather
      than being merged with it.
    """

    configured = configured_proxy()
    if configured is None:
        return {"trust_env": True}
    if configured.lower() == DIRECT:
        return {"proxy": None, "trust_env": False}
    return {"proxy": configured, "trust_env": False}


def llm_http_client(timeout: Any = None, **kwargs: Any) -> httpx.Client:
    """An `httpx.Client` for an LLM-layer API call.

    Signature-compatible with `httpx.Client` for the call sites that pass a
    bare `timeout=`, and with the `client_factory` seam the search and token
    clients already inject in tests.
    """

    return httpx.Client(timeout=timeout, **{**client_kwargs(), **kwargs})
