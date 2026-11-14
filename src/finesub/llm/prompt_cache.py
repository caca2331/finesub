"""Provider prompt caching on the direct API transports.

The calls of one run share a long prefix (the system message, then the run's
context pack), and every provider we bill against discounts a prefix it has
seen recently -- DeepSeek and OpenAI on their own, Anthropic only where a
request marks a breakpoint. Three things here help those discounts land:

- :func:`prefix_key`: one routing key per shared prefix. OpenAI routes by
  ``prompt_cache_key`` and OpenRouter keeps a session on one upstream by it,
  so requests that share a system message also share a machine. It hashes the
  system text alone: windows of one run, and runs with the same prompt, are
  the same prefix and must not be scattered across keys.
- :func:`cache_key_rejected` / :func:`note_cache_key_rejected`: not every
  OpenAI-compatible endpoint takes the field -- some self-hosted servers 400 on
  unknown fields -- so the transport sends it, and an endpoint that refuses it
  is remembered for the rest of the process and asked again without it. A
  rejected 400 is not billed, so learning costs one failed request.
- :func:`warm_prefix`: concurrent requests with a cold prefix all pay for it
  in full, because a cache entry is readable only once a request has written
  it (Anthropic: once its response starts; DeepSeek: seconds after). The first
  request of a prefix therefore goes alone and the rest wait for it. Serial
  runs never wait; a parallel run waits one call's time per prefix.

Why each choice, with the provider readings it rests on, is in
docs/provider-adapters.md「缓存」.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import threading
import time
from typing import Any, Dict, Iterator, List, Mapping, Set, Tuple

#: How long a prefix counts as warm after a request on it succeeded. The
#: shortest lifetime among the providers we target (Anthropic's default
#: five-minute entry); past it a burst of concurrent requests may all be cold
#: again, so the first one leads once more.
WARM_SECONDS = 300.0

_lock = threading.Lock()
_rejecting_endpoints: Set[str] = set()
_warm_until: Dict[Tuple[str, ...], float] = {}
_in_flight: Dict[Tuple[str, ...], threading.Event] = {}


def system_text(messages: List[Mapping[str, Any]]) -> str:
    def text_of(content: Any) -> str:
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
        return str(content or "")

    return "\n\n".join(text_of(msg.get("content")) for msg in messages if msg.get("role") == "system")


def prefix_key(messages: List[Mapping[str, Any]]) -> str:
    """Empty without a system message: there is no shared prefix to name."""

    text = system_text(messages)
    if not text:
        return ""
    return "finesub-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def cache_key_rejected(base_url: str) -> bool:
    with _lock:
        return base_url in _rejecting_endpoints


def note_cache_key_rejected(base_url: str) -> None:
    with _lock:
        _rejecting_endpoints.add(base_url)


def is_cache_key_rejection(exc: BaseException) -> bool:
    """A 400 -- or the 422 a strict FastAPI-style server answers an unknown
    field with -- that names the field. Anything else is the request's own
    fault and must surface as it is, not be retried without the key."""

    return getattr(exc, "status_code", None) in (400, 422) and "prompt_cache_key" in str(exc)


@contextmanager
def warm_prefix(scope: Tuple[str, ...]) -> Iterator[None]:
    """Hold back a request whose prefix another request is writing right now.

    ``scope`` names the cache: endpoint, key, model and :func:`prefix_key`. The
    first request on a cold scope leads; requests arriving while it is in
    flight wait until it ends, success or not, and then go together. A failed
    leader leaves the scope cold, so the next request leads again.
    """

    if not scope[-1]:
        yield
        return
    with _lock:
        warm = _warm_until.get(scope, 0.0) > time.monotonic()
        leader_event = None if warm else _in_flight.get(scope)
        leads = not warm and leader_event is None
        if leads:
            _in_flight[scope] = threading.Event()
    if leader_event is not None:
        leader_event.wait()
    if not leads:
        yield
        _mark_warm(scope)
        return
    try:
        yield
        _mark_warm(scope)
    finally:
        with _lock:
            _in_flight.pop(scope).set()


def _mark_warm(scope: Tuple[str, ...]) -> None:
    with _lock:
        _warm_until[scope] = time.monotonic() + WARM_SECONDS
