"""Per-run context and lane ordinals (docs/task-parallelism-plan.md W1).

Three pieces of state used to be process-global -- the generation pin, the
agent session registry, the lane counter -- and each assumed one run per
process. These tests pin the new contract: context-scoped isolation between
concurrent runs, explicit propagation into thread pools (a fresh worker has
an *empty* Context), and lane ordinals that belong to the run so sequential
pools land on the same conversations and pseudo hosts.

The pin/read isolation test lives with the knowledge tests
(`test_llm_knowledge_base.py`); what is here needs no store.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from types import SimpleNamespace

import pytest

from finesub.llm.agent.agent_session_host import (
    AgentSessionHost,
    agent_session_scope,
    current_registry,
)
from finesub.llm.knowledge.base import active_generation_pins, bind_generation_pins
from finesub.llm.run_context import (
    LaneOrdinalPool,
    bind_llm_worker,
    lane_ordinal_for_thread,
    llm_worker_context,
)


@pytest.fixture(autouse=True)
def _clear_pins_after():
    """Pins bound on the test thread's context outlive the test otherwise."""

    yield
    bind_generation_pins({})


def _fake_host(builds: list[str], label: str) -> SimpleNamespace:
    builds.append(label)
    return SimpleNamespace(
        label=label,
        close=lambda **kwargs: None,
        usage_totals=lambda: {},
        native_search=False,
    )


def test_concurrent_scopes_are_isolated_per_context() -> None:
    """Two runs on two threads own separate registries; one closing does not
    close the other's (W1 acceptance 2/3)."""

    seen: dict[str, object] = {}
    a_open = threading.Event()
    a_closed = threading.Event()
    b_checked = threading.Event()

    def run_a() -> None:
        with agent_session_scope() as registry:
            seen["a"] = registry
            a_open.set()
            assert b_checked.wait(10)
        a_closed.set()

    def run_b() -> None:
        assert a_open.wait(10)
        with agent_session_scope() as registry:
            seen["b"] = registry
            assert registry is not seen["a"]
            assert current_registry() is registry
            b_checked.set()
            assert a_closed.wait(10)
            # Run A ended; this run's registry must still accept hosts.
            builds: list[str] = []
            registry.host_for(("T", "m", 1, "pseudo-conversational"),
                              lambda: _fake_host(builds, "h"))
            assert builds == ["h"]

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(10)
    thread_b.join(10)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert current_registry() is None  # the main thread never entered a scope


def test_pool_workers_inherit_pins_and_registry_concurrently() -> None:
    """W1 acceptance 5/6: workers read the run's pin and registry, and the
    binding survives many work items running concurrently in one pool."""

    bind_generation_pins({"root-x": 7})
    with agent_session_scope() as registry:
        context = llm_worker_context()
        barrier = threading.Barrier(3, timeout=10)
        seen: list[tuple[dict[str, int], object]] = []
        lock = threading.Lock()

        def probe(_item: int) -> None:
            barrier.wait()  # all three workers inside their contexts at once
            with lock:
                seen.append((active_generation_pins(), current_registry()))

        try:
            with cf.ThreadPoolExecutor(
                max_workers=3, initializer=bind_llm_worker, initargs=(context,)
            ) as pool:
                list(pool.map(probe, range(3)))
        finally:
            context.release_lanes()
        assert len(seen) == 3
        for pins, worker_registry in seen:
            assert pins == {"root-x": 7}
            assert worker_registry is registry
        assert registry.lanes.leased() == set()  # released back to the run


def test_sequential_pools_reuse_the_same_ordinals_and_hosts() -> None:
    """W1 acceptance 7: the same logical lane ordinal in two sequential pools
    lands on the same pseudo host; a second run builds its own."""

    def run_phase(registry, builds: list[str]) -> set[int]:
        context = llm_worker_context()
        lanes: set[int] = set()
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=10)

        def work(_item: int) -> None:
            barrier.wait()
            lane = lane_ordinal_for_thread(registry.lanes)
            with lock:
                lanes.add(lane)
            registry.host_for(
                ("T", "m", lane, "pseudo-conversational"),
                lambda: _fake_host(builds, f"lane{lane}"),
            )

        try:
            with cf.ThreadPoolExecutor(
                max_workers=2, initializer=bind_llm_worker, initargs=(context,)
            ) as pool:
                list(pool.map(work, range(2)))
        finally:
            context.release_lanes()
        return lanes

    with agent_session_scope() as registry:
        builds: list[str] = []
        first = run_phase(registry, builds)
        second = run_phase(registry, builds)
        assert first == second == {1, 2}
        # Phase two reused phase one's hosts: no new factory calls.
        assert sorted(builds) == ["lane1", "lane2"]

    # A different run: same ordinals, different registry, fresh hosts.
    with agent_session_scope() as other:
        other_builds: list[str] = []
        assert run_phase(other, other_builds) == {1, 2}
        assert sorted(other_builds) == ["lane1", "lane2"]


def test_the_run_threads_lane_is_lent_to_the_pool_and_taken_back() -> None:
    """The serial thread's ordinal (the research lane) is exactly the one a
    pool worker should reuse -- lent at snapshot, reacquired on next use --
    so a run's host count stays at N, not N+1."""

    with agent_session_scope() as registry:
        mine = lane_ordinal_for_thread(registry.lanes)
        assert mine == 1
        context = llm_worker_context()  # lends ordinal 1 to the phase
        worker_lanes: set[int] = set()
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=10)

        def work(_item: int) -> None:
            barrier.wait()
            with lock:
                worker_lanes.add(lane_ordinal_for_thread(registry.lanes))

        try:
            with cf.ThreadPoolExecutor(
                max_workers=2, initializer=bind_llm_worker, initargs=(context,)
            ) as pool:
                list(pool.map(work, range(2)))
        finally:
            context.release_lanes()
        assert worker_lanes == {1, 2}  # the lent ordinal was reused
        assert lane_ordinal_for_thread(registry.lanes) == 1  # and comes back


def test_a_reused_thread_re_leases_from_the_new_runs_pool() -> None:
    """A batch worker thread serves one run after another: its lane binding
    names the old run's pool and must not leak into the new one."""

    pool_a = LaneOrdinalPool()
    pool_b = LaneOrdinalPool()
    assert lane_ordinal_for_thread(pool_a) == 1
    assert lane_ordinal_for_thread(pool_b) == 1  # fresh lease, no crosstalk
    assert pool_a.leased() == {1} and pool_b.leased() == {1}


# --- W4: the agent slot budget and the mandatory-lane reservation -----------


def _budget(limit: int):
    from finesub.llm.agent.local_agent import AgentSlotBudget

    return AgentSlotBudget(limit)


def test_the_mandatory_lane_redeems_across_windows_and_never_blocks() -> None:
    """W4 acceptance: the reservation is a task-lifetime state machine --
    redeem on call start, swing back on call end -- so the SECOND and later
    mandatory calls still cannot starve once other callers fill the pool."""

    from finesub.llm.run_context import llm_task_slots

    budget = _budget(3)
    with llm_task_slots([budget]):
        assert budget.snapshot() == {"limit": 3, "held": 0, "reserved": 1, "free": 2}
        # Two untracked callers (another task's fan-out) take every free slot.
        held = [threading.Event(), threading.Event()]
        release = threading.Event()

        def occupy(index: int) -> None:
            with budget:  # fresh thread: no claim bound -> free slot
                held[index].set()
                release.wait(10)

        occupiers = [threading.Thread(target=occupy, args=(i,)) for i in range(2)]
        for thread in occupiers:
            thread.start()
        assert all(event.wait(10) for event in held)
        assert budget.free() == 0
        # Window one and window two of the mandatory lane: both redeem
        # instantly even though free == 0 (this thread carries the claim).
        for _window in range(2):
            with budget:
                assert budget.snapshot()["reserved"] == 0
                assert budget.snapshot()["held"] == 3
            assert budget.snapshot()["reserved"] == 1  # the swing back
        release.set()
        for thread in occupiers:
            thread.join(10)
    assert budget.snapshot() == {"limit": 3, "held": 0, "reserved": 0, "free": 3}


def test_untracked_callers_cannot_eat_the_reservation() -> None:
    """Invariant I1: free excludes reserved, so an optional/untracked caller
    waits while the mandatory lane sails through."""

    from finesub.llm.run_context import llm_task_slots

    budget = _budget(1)
    with llm_task_slots([budget]):
        entered = threading.Event()
        done = threading.Event()

        def untracked() -> None:
            with budget:  # must wait: the only slot is reserved
                entered.set()
            done.set()

        waiter = threading.Thread(target=untracked)
        waiter.start()
        assert not entered.wait(0.3)  # blocked on the reservation
        with budget:  # the mandatory call redeems instantly
            assert budget.snapshot()["held"] == 1
        # After the run releases the reservation the waiter gets in.
    assert done.wait(10)


def test_a_second_covered_call_falls_back_to_free_when_the_host_holds_the_reservation() -> None:
    """Pseudo tier: the long-lived host consumed the reservation; a covered
    call arriving meanwhile takes a free slot instead of corrupting the
    books (mixed-mode safety)."""

    from finesub.llm.run_context import llm_task_slots

    budget = _budget(2)
    with llm_task_slots([budget]):
        with budget:  # the host's long enter: redeems
            assert budget.snapshot() == {"limit": 2, "held": 1, "reserved": 0, "free": 1}
            with budget:  # covered again, no reservation left -> free slot
                assert budget.snapshot()["held"] == 2
            assert budget.snapshot()["held"] == 1
        # The host's exit swings back: the task keeps its backstop.
        assert budget.snapshot()["reserved"] == 1


def test_task_teardown_settles_a_reservation_a_hung_host_still_holds() -> None:
    """A pseudo CLI can outlive the close grace: the task then ends while its
    reservation is redeemed. Teardown must not raise, and the host's eventual
    exit must release the slot to free instead of resurrecting the dead
    task's reservation (this used to crash the run's teardown and then leak
    one reserved slot for the life of the process)."""

    from finesub.llm.run_context import bind_slot_claim, llm_task_slots

    budget = _budget(2)
    entered = threading.Event()
    task_closed = threading.Event()
    host_claim: list = []

    def hung_host() -> None:
        bind_slot_claim(host_claim[0])  # the supervisor rebinding the claim set
        with budget:  # redeems the reservation and holds past task end
            entered.set()
            task_closed.wait(10)

    with llm_task_slots([budget]) as account:
        host_claim.append(account.claims)
        supervisor = threading.Thread(target=hung_host)
        supervisor.start()
        assert entered.wait(10)
        assert budget.snapshot() == {"limit": 2, "held": 1, "reserved": 0, "free": 1}
    # llm_task_slots exited while the host still holds the redeemed slot:
    # no exception, and the books show the hold without a reservation.
    assert budget.snapshot() == {"limit": 2, "held": 1, "reserved": 0, "free": 1}
    task_closed.set()
    supervisor.join(10)
    # The late exit released to free -- nothing leaked into `reserved`.
    assert budget.snapshot() == {"limit": 2, "held": 0, "reserved": 0, "free": 2}


def test_claim_cap_decays_as_agent_tasks_crowd_the_pool() -> None:
    """W4 step 3: `1 + floor(free / A)` -- busy batches degrade every task
    to its single guaranteed lane, which is the intended behavior."""

    from finesub.llm.run_context import TaskSlotAccount

    budget = _budget(4)
    first = TaskSlotAccount([budget])
    assert first.open(timeout=1)
    try:
        assert first.claim_cap() == 1 + 3 // 1  # alone: full fan-out
        second = TaskSlotAccount([budget])
        assert second.open(timeout=1)
        try:
            assert first.claim_cap() == 1 + 2 // 2  # crowded: fair share
            third = TaskSlotAccount([budget])
            assert third.open(timeout=1)
            try:
                assert first.claim_cap() == 1  # saturated: serial on the lane
            finally:
                third.close()
        finally:
            second.close()
    finally:
        first.close()
    assert budget.snapshot()["reserved"] == 0


def test_max_parallel_rides_the_execution_settings_into_every_driver_config() -> None:
    """W4: `[llm] local_agent_max_parallel` is the one configuration surface
    for the physical ceiling."""

    from finesub.llm.routing.execution_policy import ExecutionSettings

    settings = ExecutionSettings(local_agent_max_parallel=2)
    assert settings.codex_driver_config(model="m").max_parallel == 2
    assert settings.claude_code_driver_config(model="m").max_parallel == 2
    assert settings.agy_driver_config(model="m").max_parallel == 2
    assert settings.dsh_driver_config(model="m").max_parallel == 2


def test_mcp_spec_carries_the_pins_captured_at_construction(tmp_path) -> None:
    """W1 acceptance 4: the MCP server spec is built on the supervisor
    thread, whose Context is empty -- the run's pin must have been written
    down when the host was created."""

    driver = SimpleNamespace(
        config=SimpleNamespace(
            next_task_wait_seconds=25.0,
            conversation_ttl_seconds=0.0,
            mcp_page_chars=0,
            mcp_block_files=False,
            model="fake-model",
        ),
        driver_id="fake",
    )
    bind_generation_pins({"C:/kb-root-of-this-run": 5})
    host = AgentSessionHost(
        driver,
        root=tmp_path / "assignment",
        execution_identity={},
        task_timeout_seconds=5.0,
        label="pin-capture",
    )
    spec_env: dict[str, str] = {}

    def fresh_thread() -> None:
        spec_env.update(host._mcp_server_spec("session-1")["env"])

    worker = threading.Thread(target=fresh_thread)
    worker.start()
    worker.join(10)
    assert spec_env["FINESUB_MCP_KNOWLEDGE_ROOT"] == "C:/kb-root-of-this-run"


def test_every_chain_budget_carries_its_own_reservation() -> None:
    """Reviewer 2026-08-30 P1-1: a chain mixing two vendors reserves on BOTH
    pools, and a call landing on the second vendor redeems that pool's claim
    -- not the first's, and not nothing."""

    from finesub.llm.run_context import llm_task_slots

    first, second = _budget(2), _budget(2)
    with llm_task_slots([first, second]) as account:
        assert first.snapshot()["reserved"] == 1
        assert second.snapshot()["reserved"] == 1
        # The call routed to the SECOND vendor: its claim redeems there.
        with second:
            assert second.snapshot() == {"limit": 2, "held": 1, "reserved": 0, "free": 1}
            assert first.snapshot()["reserved"] == 1  # untouched
        assert second.snapshot()["reserved"] == 1  # swung back
        # claim_cap follows the tightest pool.
        assert account.claim_cap() == 1 + min(first.free(), second.free())
    assert first.snapshot()["reserved"] == 0
    assert second.snapshot()["reserved"] == 0
