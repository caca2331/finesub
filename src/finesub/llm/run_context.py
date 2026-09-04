"""Per-run context that worker threads must inherit explicitly (W1).

Three pieces of state are scoped to one run (docs/task-parallelism-plan.md
W1): the knowledge generation pin, the agent session registry and the lane
ordinals. The first two live in ContextVars owned by their own modules
(``knowledge.base``, ``agent.agent_session_host``); this module owns the lane
pool and the one channel that carries all of it onto a pool's workers.

A ContextVar does not reach a ``ThreadPoolExecutor`` worker by itself: a new
thread starts with an *empty* Context, not a copy of its parent's. So every
pool a run opens rebinds the run's state through its ``initializer`` -- the
same channel the reporter already rides (``bind_reporter``). One snapshot
object per pool, never a shared ``contextvars.Context``: the same Context
cannot be entered by two threads at once, so the copy-context form breaks
under exactly the concurrency the pools exist for.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
from typing import Any, Mapping, Sequence

from finesub.reporting import Reporter, bind_reporter, current_reporter


class LaneOrdinalPool:
    """One run's agent-lane ordinals (1..N), leased smallest-free.

    The ordinal is the lane's *identity* -- ``resume`` conversations and
    pseudo-conversational hosts key on it -- so it belongs to the run, never
    to a thread (the OS recycles thread ids) or to one pool (a run's phases
    open pools in sequence, and the second phase's fresh threads must land on
    the first phase's hosts). Ordinals are numbers, not slots: leasing is
    free; ``driver.max_parallel`` is accounted elsewhere (plan W4).

    The owner tag exists for one narrow case: a serial thread's lease is lent
    to a pool phase (``llm_worker_context`` releases it) and taken back on
    that thread's next use -- ``reacquire`` must tell "free again" and "still
    mine" apart from "a worker holds it right now".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[int, int] = {}

    def lease(self, owner: int) -> int:
        with self._lock:
            ordinal = 1
            while ordinal in self._owners:
                ordinal += 1
            self._owners[ordinal] = owner
            return ordinal

    def release(self, ordinal: int) -> None:
        with self._lock:
            self._owners.pop(ordinal, None)

    def reacquire(self, ordinal: int, owner: int) -> bool:
        """Take this ordinal back if it is free or already this owner's."""

        with self._lock:
            holder = self._owners.get(ordinal)
            if holder is None:
                self._owners[ordinal] = owner
                return True
            return holder == owner

    def leased(self) -> set[int]:
        with self._lock:
            return set(self._owners)


# (pool, ordinal): the pool reference scopes the binding to one run, so a
# reused thread (a batch worker serving its next task) re-leases from the new
# run's pool instead of carrying a stale number across runs.
_LANE: ContextVar[tuple[LaneOrdinalPool, int] | None] = ContextVar(
    "finesub_llm_lane", default=None
)


def current_lane_ordinal(pool: LaneOrdinalPool) -> int | None:
    """This thread's bound ordinal, if it belongs to ``pool``."""

    bound = _LANE.get()
    if bound is not None and bound[0] is pool:
        return bound[1]
    return None


def bind_lane_ordinal(pool: LaneOrdinalPool, ordinal: int) -> None:
    _LANE.set((pool, ordinal))


def lane_ordinal_for_thread(pool: LaneOrdinalPool) -> int:
    """This thread's lane in the run that owns ``pool``, leasing on first use.

    A binding from another run's pool (a reused batch worker) is ignored; a
    binding whose ordinal was lent to a pool phase is taken back if it is
    free again, else the thread gets a fresh one.
    """

    me = threading.get_ident()
    bound = _LANE.get()
    if bound is not None and bound[0] is pool and pool.reacquire(bound[1], me):
        return bound[1]
    ordinal = pool.lease(me)
    _LANE.set((pool, ordinal))
    return ordinal


# --- agent slot accounting (plan W4) ----------------------------------------


@dataclass
class MandatoryLaneClaim:
    """This thread's calls are the task's mandatory lane on ``budget``.

    The driver's slot budget reads it at call time: a covered call redeems
    the task's reservation (reserved -> held) and swings it back on exit
    (held -> reserved), so the mandatory lane never blocks behind other
    tasks' optional fan-out (invariant I1) and the backstop survives past the
    first window (the reviewer round-3 fix).

    ONE claim per (task, budget), shared by every carrier of the mandatory
    lane -- the run thread, whichever pool worker leases lane ordinal 1, a
    pseudo host's supervisor. A single object is what keeps the books whole:
    ``redeemed`` (guarded by the budget's lock) says whether the reservation
    is currently converted into a held slot, so the task's teardown can tell
    "release the reservation" apart from "a live call still holds it -- its
    exit must release to free instead of swinging back" (a pseudo CLI that
    outlives the close grace used to crash the run's teardown here and then
    leak one reserved slot for the life of the process)."""

    budget: Any
    active: bool = True
    redeemed: int = 0


class TaskSlotClaims:
    """A task's mandatory-lane claims, one per agent budget of its chain.

    The routing chain may name more than one vendor (a fallback mixing
    codex and agy pools), and which pool a call actually lands on is decided
    per attempt -- so the reservation must exist on EVERY pool the chain can
    reach, and a budget resolves its own claim out of this set at call time
    (reviewer 2026-08-30 P1-1: reserving only the catalog's first pool left
    calls routed to a second vendor uncovered -- no starvation guarantee, and
    the reserved slot idled on the wrong pool)."""

    def __init__(self, budgets: Sequence[Any]) -> None:
        self._by_budget = {id(budget): MandatoryLaneClaim(budget) for budget in budgets}

    def claim_for(self, budget: Any) -> MandatoryLaneClaim | None:
        return self._by_budget.get(id(budget))

    def all(self) -> tuple[MandatoryLaneClaim, ...]:
        return tuple(self._by_budget.values())


_SLOT_CLAIM: ContextVar["TaskSlotClaims | None"] = ContextVar(
    "finesub_slot_claim", default=None
)
_TASK_ACCOUNT: ContextVar["TaskSlotAccount | None"] = ContextVar(
    "finesub_task_slot_account", default=None
)

_ACCOUNT_REGISTRY_LOCK = threading.Lock()
_ACTIVE_AGENT_ACCOUNTS: set[int] = set()


def current_slot_claim() -> "TaskSlotClaims | None":
    return _SLOT_CLAIM.get()


def bind_slot_claim(claims: "TaskSlotClaims | None") -> None:
    _SLOT_CLAIM.set(claims)


def current_task_account() -> "TaskSlotAccount | None":
    return _TASK_ACCOUNT.get()


def active_agent_task_count() -> int:
    """``A`` of the allocator: tasks with agent demand alive right now.

    Only these compete for lanes -- API-only tasks never touch
    ``driver.max_parallel`` (plan W4's two-counters rule; admission uses the
    task total, which lives with the batch runner)."""

    with _ACCOUNT_REGISTRY_LOCK:
        return len(_ACTIVE_AGENT_ACCOUNTS)


class TaskSlotAccount:
    """One task's standing in its chain's agent slot budgets for a whole run.

    Opened when the task starts (reserving the mandatory-lane backstop on
    every budget the chain can reach), closed when it ends. ``claim_cap()``
    is the allocator's fan-out bound: ``1 + floor(free / A)`` on the
    TIGHTEST budget -- the mandatory lane plus this task's fair share of what
    is genuinely free. When agent tasks crowd a pool it decays to 1 and every
    task runs its windows serially on its guaranteed lane, which is the
    intended degradation (same slots, better tokens)."""

    def __init__(self, budgets: Sequence[Any]) -> None:
        self.budgets = tuple(budgets)
        # The task's mandatory-lane claims, one per budget -- every carrier
        # binds this same set (see TaskSlotClaims's docstring).
        self.claims = TaskSlotClaims(self.budgets)
        self._reserved: list[Any] = []

    def open(self, timeout: float | None = None) -> bool:
        """Reserve the mandatory lane on every budget; register in ``A``."""

        for budget in self.budgets:
            if not budget.reserve(timeout):
                for reserved in self._reserved:
                    reserved.retire_reservation(self.claims.claim_for(reserved))
                self._reserved.clear()
                return False
            self._reserved.append(budget)
        with _ACCOUNT_REGISTRY_LOCK:
            _ACTIVE_AGENT_ACCOUNTS.add(id(self))
        return True

    def close(self) -> None:
        with _ACCOUNT_REGISTRY_LOCK:
            _ACTIVE_AGENT_ACCOUNTS.discard(id(self))
        reserved, self._reserved = self._reserved, []
        for budget in reserved:
            budget.retire_reservation(self.claims.claim_for(budget))

    def claim_cap(self) -> int:
        tasks = max(1, active_agent_task_count())
        return 1 + min(
            (max(0, budget.free()) for budget in self.budgets),
            default=0,
        ) // tasks


def default_task_slots(*, test_profile: bool = False):
    """This run's slot account against every agent budget its chain can reach.

    The one place that answers "which budgets does a task book against"
    (`_task_slots` in the correction entry point and the standalone knowledge
    update both come here). Never fatal: the account is scheduling fairness,
    not correctness, and a routing config that cannot even be loaded fails
    later with its own error. A test-profile run books nothing.
    """

    budgets: tuple = ()
    if not test_profile:
        try:
            from .routing.execution_policy import default_agent_slot_budgets

            budgets = default_agent_slot_budgets()
        except Exception as exc:  # noqa: BLE001 -- see docstring
            current_reporter().debug(f"agent slot budget unavailable: {exc}")
    return llm_task_slots(budgets)


@contextmanager
def llm_task_slots(budgets: Sequence[Any] | None):
    """One task's slot account, bound for the run (plan W4).

    ``budgets`` are the in-flight budgets of every agent vendor the routing
    chain can reach (``default_agent_slot_budgets``), or None/empty for a
    pure-API task -- then nothing is reserved and nothing changes. Binds the
    account AND the mandatory-lane claims on the calling thread: the run's
    serial phases (research, knowledge update) run here and are exactly the
    mandatory lane; pool phases re-bind both onto whichever worker leases
    lane ordinal 1 (`bind_llm_worker`)."""

    if not budgets:
        yield None
        return
    existing = _TASK_ACCOUNT.get()
    if existing is not None:
        # Already inside a task's account (a knowledge update running as the
        # tail of a correction run): one task books one set of reservations,
        # so nesting must reuse, never double-reserve.
        yield existing
        return
    account = TaskSlotAccount(budgets)
    if not account.open():
        # Nothing was booked, so nothing may claim: binding the claims here
        # would let this task redeem ANOTHER task's reservation (the budget
        # counts reservations, it cannot tell whose is whose).
        current_reporter().debug("agent slot reservation timed out; running unbooked")
        yield None
        return
    account_token = _TASK_ACCOUNT.set(account)
    claim_token = _SLOT_CLAIM.set(account.claims)
    try:
        yield account
    finally:
        _SLOT_CLAIM.reset(claim_token)
        _TASK_ACCOUNT.reset(account_token)
        account.close()  # deactivates the claims and settles the reservations


@dataclass
class LlmWorkerContext:
    """What one pool's workers inherit from the run thread.

    Snapshot once per pool on the thread that opens it, pass as the pool's
    ``initializer`` argument, and call :meth:`release_lanes` after the pool
    has joined -- the ordinals then return *to the run*, so the next phase's
    pool leases the same 1..N and lands on the same hosts (plan W1).
    """

    reporter: Reporter
    pins: Mapping[str, int]
    registry: Any
    account: "TaskSlotAccount | None" = None
    _leased: list[int] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def note_lease(self, ordinal: int) -> None:
        with self._lock:
            self._leased.append(ordinal)

    def release_lanes(self) -> None:
        if self.registry is None:
            return
        pool: LaneOrdinalPool = self.registry.lanes
        with self._lock:
            ordinals, self._leased = self._leased, []
        # High-to-low, so what stays leased is always the low prefix and the
        # hosts it names stay reachable (plan W1's shrink rule).
        for ordinal in sorted(ordinals, reverse=True):
            pool.release(ordinal)


def llm_worker_context() -> LlmWorkerContext:
    """Snapshot the calling thread's run state for a pool's workers.

    Also lends the calling thread's own lane to the phase: the run thread is
    about to park on the pool's futures, and its ordinal (the serial phases'
    lane, usually 1) is exactly the one whose host the workers should reuse.
    The thread's binding stays; its next use reacquires the ordinal.
    """

    from .agent.agent_session_host import current_registry
    from .knowledge.base import active_generation_pins

    registry = current_registry()
    account = current_task_account()
    if registry is not None:
        bound = _LANE.get()
        if bound is not None and bound[0] is registry.lanes:
            registry.lanes.release(bound[1])
        if account is not None and 1 in registry.lanes.leased():
            # The mandatory lane rides whoever leases ordinal 1 (see
            # `bind_llm_worker`), so a pool that opens while someone else holds
            # it silently loses invariant I1 -- every worker would compete for
            # free slots with the task's reservation sitting unredeemed. It
            # cannot happen today (the run thread just handed 1 back and phases
            # release their ordinals), which is exactly why it would be a quiet
            # regression rather than a failure.
            current_reporter().debug(
                "llm-lane-one-busy", {"leased": sorted(registry.lanes.leased())}
            )
    return LlmWorkerContext(
        reporter=current_reporter(),
        pins=active_generation_pins(),
        registry=registry,
        account=current_task_account(),
    )


def bind_llm_worker(context: LlmWorkerContext) -> None:
    """``ThreadPoolExecutor`` initializer: runs once per worker thread.

    Rebinds everything the run's code reads off the ambient context --
    reporter, generation pins, session registry -- and leases this worker a
    lane ordinal from the run's pool. Eager on purpose: the leased *set* is
    then a deterministic 1..N per phase, whichever thread gets which number.
    """

    from .agent.agent_session_host import bind_agent_session_registry
    from .knowledge.base import bind_generation_pins

    bind_reporter(context.reporter)
    bind_generation_pins(context.pins)
    bind_agent_session_registry(context.registry)
    _TASK_ACCOUNT.set(context.account)
    if context.registry is not None:
        ordinal = context.registry.lanes.lease(threading.get_ident())
        bind_lane_ordinal(context.registry.lanes, ordinal)
        context.note_lease(ordinal)
        if ordinal == 1 and context.account is not None:
            # This worker inherits the task's mandatory lane for the phase:
            # its calls redeem the reservation instead of competing for free
            # slots (the run thread that normally holds lane 1 is parked on
            # this pool's futures and makes no calls meanwhile). The SAME
            # claim set as everywhere else, so the redeemed counters stay
            # whole across carriers.
            _SLOT_CLAIM.set(context.account.claims)
