"""Subscription exhaustion for local agent CLIs.

An agent call is metered by a subscription, not by the API rate limiter, and
none of the three CLIs exposes a way to ask how much of it is left: `codex`,
`claude` and `agy` all have no usage/quota subcommand, and Claude Code's
`/usage` is an in-session slash command the driver disables. So exhaustion can
only be inferred from a failed call.

Inferring it matters because the failure is a property of the *subscription*,
not of one model. Without this the router walked straight from a dead Codex
model to the other Codex model, and from a dead Claude model through two more,
burning a full CLI launch on each -- every call, for the rest of the run, with
nothing remembered across processes.

The evidence is deliberately cheap and deliberately late: one minimal ping,
only after a pool has failed twice in a row (or once, if the text already says
so plainly). One failure is noise; two on the same subscription is worth a
question. Waiting for the second means the vendor's exact wording never has to
be guessed -- and guessing wrong is the expensive direction, since a freeze
takes a working vendor out for hours.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import threading
from typing import Any, Callable, Mapping

from finesub import state as state_store


_STATE_NAMESPACE = "llm_agent_quota"
_FROZEN_UNTIL_KEY = "frozen_until"

# Deliberately shorter than any vendor window (Claude Code's limit is a
# five-hour rolling one; Codex adds a weekly one on top), because the two ways
# of being wrong are not symmetric. Thawing too early costs one minimal ping,
# which fails and re-freezes -- call it free. Thawing too late takes a working
# subscription out of every chain for the remainder of the window. So freeze
# for less than the window and let the probe find the real reset, rather than
# trying to match the longest window and eating the expensive error.
QUOTA_FREEZE_SECONDS = 2 * 60 * 60
CONSECUTIVE_FAILURES_BEFORE_PING = 2


def normalized_pool(pool: str) -> str:
    return (pool or "").strip().upper()


# Nothing here reads the vendor's wording, in either direction.
#
# Matching phrases like "usage limit" would let a spent plan be spotted one
# call earlier than the probe finds it -- a small gain against an unbounded
# surface, since there is no way to enumerate what a vendor may put in an error
# field, and a false match takes a working subscription out for hours.
#
# Reading a reset time out of that same text has the same shape: a wrong parse
# sets a wrong deadline, while a flat freeze self-corrects the moment any call
# succeeds and costs at most one small probe per window.


class AgentQuotaLedger:
    """Per-allowance exhaustion, shared across processes through ``.state``.

    The key is the catalog's ``quota_pool``, which defaults to the provider
    tier. It is a separate concept only because Antigravity meters its Gemini
    and Opus models apart: booking them together would take a working model
    out of service for hours because its neighbour ran dry.
    """

    def __init__(self, state_path: Any | None = None) -> None:
        self.state_path = state_path
        self._lock = threading.Lock()
        self._frozen_until: dict[str, str] = {}
        self._cleared: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}
        self._load()

    # --- durable state -------------------------------------------------

    def _load(self) -> None:
        section = state_store.read_section(_STATE_NAMESPACE, self.state_path)
        stored = section.get(_FROZEN_UNTIL_KEY, {})
        if isinstance(stored, Mapping):
            self._frozen_until = {str(k): str(v) for k, v in stored.items()}

    def _persist(self) -> None:
        # Merge rather than assign, for the reason the rate limiter does: the
        # in-memory copy was read once, and another FineSub process may have
        # recorded a freeze since. What this process cleared stays cleared.
        with state_store.state_section(_STATE_NAMESPACE, self.state_path) as section:
            stored = section.get(_FROZEN_UNTIL_KEY)
            if isinstance(stored, dict):
                for key, value in stored.items():
                    name = str(key)
                    if name not in self._frozen_until and name not in self._cleared:
                        self._frozen_until[name] = str(value)
            section[_FROZEN_UNTIL_KEY] = dict(self._frozen_until)

    # --- queries -------------------------------------------------------

    def frozen_until(self, pool: str, *, now: datetime | None = None) -> datetime | None:
        """When this subscription is expected back, or None if it is usable."""

        key = normalized_pool(pool)
        if not key:
            return None
        with self._lock:
            stamp = self._frozen_until.get(key)
            if not stamp:
                return None
            try:
                deadline = datetime.fromisoformat(stamp)
            except ValueError:
                self._frozen_until.pop(key, None)
                return None
            moment = now or datetime.now(timezone.utc)
            if deadline <= moment:
                self._frozen_until.pop(key, None)
                self._cleared.add(key)
                self._persist()
                return None
            return deadline

    def is_frozen(self, pool: str) -> bool:
        return self.frozen_until(pool) is not None

    # --- updates -------------------------------------------------------

    def note_success(self, pool: str) -> None:
        """A working call is the cheapest proof there is."""

        key = normalized_pool(pool)
        if not key:
            return
        with self._lock:
            self._consecutive_failures.pop(key, None)
            if self._frozen_until.pop(key, None) is not None:
                self._cleared.add(key)
                self._persist()

    def note_failure(self, pool: str) -> bool:
        """Record a failed call; say whether it now deserves a probe.

        One failure is noise. Two in a row on the same subscription is worth
        one small question, and asking it that way needs no knowledge of how
        any vendor words the answer.
        """

        key = normalized_pool(pool)
        if not key:
            return False
        with self._lock:
            count = self._consecutive_failures.get(key, 0) + 1
            self._consecutive_failures[key] = count
        return count >= CONSECUTIVE_FAILURES_BEFORE_PING

    def freeze(self, pool: str, *, seconds: float, reason: str = "") -> datetime:
        key = normalized_pool(pool)
        deadline = datetime.now(timezone.utc) + timedelta(
            seconds=max(60.0, float(seconds))
        )
        with self._lock:
            self._frozen_until[key] = deadline.isoformat(timespec="seconds")
            self._cleared.discard(key)
            self._persist()
        return deadline


_LEDGER: AgentQuotaLedger | None = None
_LEDGER_LOCK = threading.Lock()


def default_ledger() -> AgentQuotaLedger:
    global _LEDGER
    with _LEDGER_LOCK:
        if _LEDGER is None:
            _LEDGER = AgentQuotaLedger()
        return _LEDGER


# The probe itself: the smallest thing that still proves the subscription
# answers. Anything larger would make the diagnosis cost more than the call it
# is diagnosing.
QUOTA_PING_MESSAGES = [
    {"role": "user", "content": "Reply with the single word: OK"}
]


def evaluate_agent_failure(
    *,
    pool: str,
    exc: BaseException,
    ping: Callable[[], None] | None,
    ledger: AgentQuotaLedger | None = None,
    warn: Callable[[str], None] | None = None,
) -> BaseException:
    """Decide whether a failed agent call means the subscription is spent.

    Returns the exception the router should classify -- the original one, or a
    quota error once a probe has confirmed the subscription itself is refusing.

    Only transient failures are considered. Everything else already has a more
    specific answer, and in particular an expired login is reported as
    `unavailable` upstream: auth must not be mistaken for exhaustion, since the
    fix is completely different and a freeze would hide it for hours.
    """

    from .agent_paths import vendor_error_text
    from .local_agent import (
        LocalAgentQuotaError,
        LocalAgentTransientError,
        LocalAgentUnavailableError,
    )

    book = ledger or default_ledger()
    if isinstance(exc, LocalAgentQuotaError):
        # The vendor said so outright, so there is nothing to probe -- but it
        # still has to be *recorded*, or the very next candidate on the same
        # subscription gets launched to hear the same answer.
        deadline = book.freeze(
            pool, seconds=float(QUOTA_FREEZE_SECONDS), reason=str(exc)
        )
        if warn is not None:
            warn(
                f"{pool} local agent is out of quota; skipping it until "
                f"{deadline.isoformat(timespec='minutes')}. {exc}"
            )
        return exc
    if not isinstance(exc, LocalAgentTransientError):
        return exc
    detail = f"{type(exc).__name__}: {exc}"
    if not book.note_failure(pool) or ping is None:
        return exc

    try:
        ping()
    except LocalAgentUnavailableError:
        # The probe says the CLI or the login is the problem, not the balance.
        return exc
    except Exception as probe_exc:
        deadline = book.freeze(
            pool, seconds=float(QUOTA_FREEZE_SECONDS), reason=detail
        )
        # The probe reuses this target's own model and driver config, so it
        # cannot separate "no allowance left" from "this target is misconfigured
        # and always fails" -- a model name the CLI does not serve fails exactly
        # like a spent plan, twice, and then the probe agrees with itself. Say
        # both, and quote the CLI so a person can tell them apart in one look.
        said = vendor_error_text(probe_exc) or vendor_error_text(exc)
        message = (
            f"{pool} local agent stopped answering; skipping it until "
            f"{deadline.isoformat(timespec='minutes')}. Usually a spent "
            "subscription -- check its usage and that you are still logged in. "
            "A target that cannot work at all fails identically (a model name "
            "this CLI does not serve, say), so check that too."
            + (f" The CLI said: {said}" if said else f" ({detail})")
        )
        if warn is not None:
            warn(message)
        return LocalAgentQuotaError(message)
    else:
        # It answers. Whatever went wrong, it was not the subscription.
        book.note_success(pool)
        return exc
