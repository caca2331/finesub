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

from finesub import key_freeze
from finesub import state as state_store


_STATE_NAMESPACE = "llm_agent_quota"
#: Where freezes used to live in this section; dropped on the next write (now in `key_freeze`).
_FROZEN_UNTIL_KEY = "frozen_until"
#: The family of this module's rows in `key_freeze`: `agent/<POOL>`.
FREEZE_FAMILY = "agent"
# `{pool: {"count": int, "at": iso}}`. The freeze was durable from the start;
# the evidence that leads to one was not, so a caller that runs one file per
# process never reached the second failure and never froze anything -- it kept
# launching the CLI for every file, which is the exact cost this module exists
# to stop. An added key: an older reader ignores it.
_FAILURE_STREAKS_KEY = "failure_streaks"

# Deliberately shorter than any vendor window (Claude Code's limit is a
# five-hour rolling one; Codex adds a weekly one on top), because the two ways
# of being wrong are not symmetric. Thawing too early costs one minimal ping,
# which fails and re-freezes -- call it free. Thawing too late takes a working
# subscription out of every chain for the remainder of the window. So freeze
# for less than the window and let the probe find the real reset, rather than
# trying to match the longest window and eating the expensive error.
QUOTA_FREEZE_SECONDS = 2 * 60 * 60
CONSECUTIVE_FAILURES_BEFORE_PING = 2

# How long a failure keeps counting toward the next one. Longer than a freeze
# on purpose: after a thaw the first failure should reach the probe straight
# away (one minimal ping, then a re-freeze if the plan is still spent), which
# is the cheap direction this module already prefers. A TTL shorter than the
# freeze would restart the count at every thaw and spend two full calls
# rediscovering what the freeze already knew. It exists at all so evidence
# from some unrelated incident days ago cannot combine with today's first
# failure.
FAILURE_STREAK_TTL_SECONDS = 2 * QUOTA_FREEZE_SECONDS


def normalized_pool(pool: str) -> str:
    return (pool or "").strip().upper()


def freeze_row(pool: str) -> str:
    return f"{FREEZE_FAMILY}/{normalized_pool(pool)}"


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

    The freeze is a `key_freeze` row (`agent/<POOL>`), read on every question
    and never cached, so `finesub quota thaw` in another process holds at once
    and is never written back. The failure streak stays in this module's own
    section.

    The key is the catalog's ``quota_pool``, which defaults to the provider
    tier. It is a separate concept only because Antigravity meters its Gemini
    and Claude models apart: booking them together would take a working model
    out of service for hours because its neighbour ran dry.
    """

    def __init__(self, state_path: Any | None = None) -> None:
        self.state_path = state_path
        self._lock = threading.Lock()
        # Not the streak itself -- that lives on disk, read under the lock at
        # the moment it is used, because any cached copy is stale by
        # definition. This only answers "has *this* process recorded a failure
        # I would have to clear", so an ordinary success costs no file write.
        self._streak_seen: dict[str, int] = {}

    # --- durable state -------------------------------------------------

    def _mutate(self, apply: Callable[[dict[str, Any]], None]) -> None:
        """One locked read-modify-write of this module's state section.

        Everything that changes durable state goes through here, and nothing
        that runs inside `apply` may open a second section: the lock is a real
        OS file lock, so nesting would deadlock rather than recurse.
        """

        with state_store.state_section(_STATE_NAMESPACE, self.state_path) as section:
            section.pop(_FROZEN_UNTIL_KEY, None)
            apply(section)

    @staticmethod
    def _live_streak(
        section: Mapping[str, Any], key: str, *, now: datetime
    ) -> int:
        """The stored streak for `key`, or 0 once it is too old to count."""

        streaks = section.get(_FAILURE_STREAKS_KEY)
        entry = streaks.get(key) if isinstance(streaks, Mapping) else None
        if not isinstance(entry, Mapping):
            return 0
        try:
            seen = datetime.fromisoformat(str(entry.get("at") or ""))
        except ValueError:
            return 0
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=timezone.utc)
        if (now - seen).total_seconds() >= FAILURE_STREAK_TTL_SECONDS:
            return 0
        try:
            return max(0, int(entry.get("count") or 0))
        except (TypeError, ValueError):
            return 0

    # --- queries -------------------------------------------------------

    def frozen_until(self, pool: str, *, now: datetime | None = None) -> datetime | None:
        """When this subscription is expected back, or None if it is usable."""

        if not normalized_pool(pool):
            return None
        held = key_freeze.frozen(freeze_row(pool), now=now, state_path=self.state_path)
        return held.until if held is not None else None

    def is_frozen(self, pool: str) -> bool:
        return self.frozen_until(pool) is not None

    # --- updates -------------------------------------------------------

    def note_success(self, pool: str) -> None:
        """A working call is the cheapest proof there is."""

        key = normalized_pool(pool)
        if not key:
            return
        with self._lock:
            mine = self._streak_seen.pop(key, None)
            thawed = self.frozen_until(key) is not None and bool(
                key_freeze.thaw([freeze_row(key)], state_path=self.state_path)
            )
            # This runs after every successful agent call, so it must not write
            # the file when there is nothing to undo (the freeze check above
            # only reads). A streak another process wrote is therefore left
            # alone unless we are lifting a freeze anyway -- the stale count
            # then costs at most one probe, which succeeds and clears it, while
            # a write per success would serialise every call on the state lock.
            if not thawed and mine is None:
                return

            def apply(section: dict[str, Any]) -> None:
                streaks = section.get(_FAILURE_STREAKS_KEY)
                streaks = dict(streaks) if isinstance(streaks, Mapping) else {}
                if streaks.pop(key, None) is not None:
                    section[_FAILURE_STREAKS_KEY] = streaks

            self._mutate(apply)

    def note_failure(self, pool: str) -> bool:
        """Record a failed call; say whether it now deserves a probe.

        One failure is noise. Two in a row on the same subscription is worth
        one small question, and asking it that way needs no knowledge of how
        any vendor words the answer.
        """

        key = normalized_pool(pool)
        if not key:
            return False
        now = datetime.now(timezone.utc)
        count = 0

        def apply(section: dict[str, Any]) -> None:
            nonlocal count
            # Read and write inside the one file lock, so two processes each
            # failing once accumulate to two rather than overwriting each
            # other with one. Over-counting is the cheap direction anyway: it
            # buys one minimal ping, and the freeze still needs that ping to
            # fail on its own.
            count = self._live_streak(section, key, now=now) + 1
            streaks = section.get(_FAILURE_STREAKS_KEY)
            streaks = dict(streaks) if isinstance(streaks, Mapping) else {}
            streaks[key] = {
                "count": count,
                "at": now.isoformat(timespec="milliseconds"),
            }
            section[_FAILURE_STREAKS_KEY] = streaks

        with self._lock:
            self._mutate(apply)
            self._streak_seen[key] = count
        return count >= CONSECUTIVE_FAILURES_BEFORE_PING

    def freeze(self, pool: str, *, seconds: float, reason: str = "") -> datetime:
        # The streak is deliberately left standing. It outlives the freeze by
        # design, so the first failure after a thaw reaches the probe at once
        # instead of spending two more full calls to re-learn this.
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(seconds=max(60.0, float(seconds)))
        key_freeze.freeze(
            freeze_row(pool),
            deadline,
            " ".join(reason.split())[:160] or "the subscription refused a minimal ping",
            now=now,
            state_path=self.state_path,
        )
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
    attribution_certain: bool = True,
) -> BaseException:
    """Decide whether a failed agent call means the subscription is spent.

    Returns the exception the router should classify -- the original one, or a
    quota error once a probe has confirmed the subscription itself is refusing.

    Only transient failures are considered. Everything else already has a more
    specific answer, and in particular an expired login is reported as
    `unavailable` upstream: auth must not be mistaken for exhaustion, since the
    fix is completely different and a freeze would hide it for hours.

    `attribution_certain=False` says the caller knows *something* was spent but
    not *whose*: a row with a paid twin whose stream named no model. The
    vendor's own verdict is then unusable as a shortcut -- it is a statement
    about an allowance nobody identified -- so it takes the probe path like
    any other failure. Skipping that is how an hourly rate limit on the free
    line, plus a spent credit balance behind it, freezes the *free* daily
    quota that was never the problem.
    """

    from .local_agent import LocalAgentQuotaError, LocalAgentTransientError

    book = ledger or default_ledger()
    if isinstance(exc, LocalAgentQuotaError) and not attribution_certain:
        detail = f"{type(exc).__name__}: {exc}"
        if ping is None:
            return exc
        return _probe_then_freeze(book, pool, ping, detail, exc, warn) or exc
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
    return _probe_then_freeze(book, pool, ping, detail, exc, warn) or exc


def evaluate_fallback_switch(
    *,
    pool: str,
    ping: Callable[[], None] | None,
    ledger: AgentQuotaLedger | None = None,
    warn: Callable[[str], None] | None = None,
) -> BaseException | None:
    """A paid twin answered, so the line it replaced is in trouble.

    This is the only evidence that exists in the case that matters most: when
    the vendor's own fallback rescues the call, the harness sees a success and
    the spent line leaves no trace at all. Route the next task to the same row
    and the CLI switches again, so credits drain while the ledger keeps
    believing the free line is healthy.

    The switch is worth exactly two failures, not a freeze. The interceptor
    retries the primary once and only then hands the session over, so one
    switch *is* two consecutive failures on that line -- which is the existing
    threshold, and reaching it the ordinary way means the probe still decides.
    That matters because the interceptor also switches on sustained rate
    limiting: "the twin answered" proves the line failed twice, never that it
    is out of quota, and freezing on that would be the expensive direction.

    Returns the quota error the freeze produced, or None. Callers on the
    success path have nothing to raise and can ignore it; it is returned so
    the decision is inspectable rather than buried.
    """

    book = ledger or default_ledger()
    if book.is_frozen(pool):
        # Already recorded, and unlike every other freeze this one does not
        # stop the calls: a row with a twin goes out of routing only when
        # *every* allowance behind it is frozen, so it keeps running and keeps
        # switching. Without this the switch would buy a fresh probe -- a real
        # CLI launch -- on *every* call for the whole freeze, which is the cost
        # this module exists to remove.
        return None
    reached = False
    for _ in range(CONSECUTIVE_FAILURES_BEFORE_PING):
        reached = book.note_failure(pool)
    if not reached or ping is None:
        return None
    return _probe_then_freeze(
        book,
        pool,
        ping,
        "the CLI handed this session to its paid twin",
        None,
        warn,
        # Recorded, not enforced. This pool is reached only from a row that
        # declares a twin, and such a row leaves routing only when *every*
        # allowance behind it is frozen -- so the run carries on over the
        # fallback and the freeze is here to be *seen* (`agent-ping`, and one
        # warning instead of one per call).
        consequence="its free daily quota looks spent, so this row runs on its "
        "paid twin from here",
    )


def _probe_then_freeze(
    book: AgentQuotaLedger,
    pool: str,
    ping: Callable[[], None],
    detail: str,
    exc: BaseException | None,
    warn: Callable[[str], None] | None,
    consequence: str = "",
) -> BaseException | None:
    """Ask the allowance itself, and freeze only if it refuses.

    Shared by both ways of reaching the threshold -- a run of failures, and a
    fallback switch -- so the freeze always rests on the same evidence: one
    minimal call to the line in question, made through a driver that cannot
    answer from anywhere else.
    """

    from .agent_paths import vendor_error_text
    from .local_agent import LocalAgentQuotaError, LocalAgentUnavailableError

    try:
        ping()
    except LocalAgentUnavailableError:
        # The probe says the CLI or the login is the problem, not the balance.
        return None
    except Exception as probe_exc:
        deadline = book.freeze(
            pool, seconds=float(QUOTA_FREEZE_SECONDS), reason=detail
        )
        # The probe reuses this target's own model and driver config, so it
        # cannot separate "no allowance left" from "this target is misconfigured
        # and always fails" -- a model name the CLI does not serve fails exactly
        # like a spent plan, twice, and then the probe agrees with itself. Say
        # both, and quote the CLI so a person can tell them apart in one look.
        said = vendor_error_text(probe_exc) or (
            vendor_error_text(exc) if exc is not None else ""
        )
        said_consequence = (
            consequence
            or f"skipping it until {deadline.isoformat(timespec='minutes')}"
        )
        message = (
            f"{pool} local agent stopped answering; {said_consequence}"
            f" (until {deadline.isoformat(timespec='minutes')}). Usually a spent "
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
        return None
