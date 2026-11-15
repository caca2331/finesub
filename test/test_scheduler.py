from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import unittest.mock

import pytest

from finesub import scheduler
from finesub.scheduler import BatchItem, IntakePoll, run_batch

# The option surface (manifest rows, item building, the CLI) lives with the
# pipeline now; the runner below knows nothing about it. Imported lazily in the
# tests that need it -- it pulls the ASR stack.


def _item(label: str, stages: dict) -> BatchItem:
    return BatchItem(label=label, stages=stages, payload=label)


def test_items_flow_through_all_bins_and_chain_payloads(tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def stage(name: str):
        def fn(payload):
            with lock:
                calls.append((name, payload))
            return payload + f"->{name}"

        return fn

    items = [
        _item(f"i{n}", {"download": stage("download"), "asr": stage("asr"), "llm": stage("llm")})
        for n in range(5)
    ]
    results = run_batch(items, status_path=tmp_path / "status.jsonl")

    assert [r.status for r in results] == ["done"] * 5
    assert results[0].payload == "i0->download->asr->llm"
    # every item hit every bin exactly once
    for n in range(5):
        assert sum(1 for name, p in calls if p.startswith(f"i{n}")) == 3


def test_missing_stages_pass_through(tmp_path) -> None:
    seen: list[str] = []
    items = [
        _item("local", {"asr": lambda p: (seen.append(p), p)[1]}),  # no download/llm
        _item("raw-only", {"download": lambda p: p, "asr": lambda p: p}),
    ]
    results = run_batch(items)
    assert [r.status for r in results] == ["done", "done"]
    assert seen == ["local"]


def test_a_group_consumes_in_item_order_despite_out_of_order_upstream() -> None:
    # Plan W5: submission order is a GROUP guarantee now. Item 0's download
    # blocks until item 2's completes, so upstream order is 1, 2, 0 — the
    # shared group must still run 0, 1, 2.
    item2_done = threading.Event()
    llm_order: list[str] = []

    def record(payload):
        llm_order.append(payload)
        return payload

    def slow_download(payload):
        item2_done.wait(timeout=10)
        return payload

    def download2(payload):
        item2_done.set()
        return payload

    items = [
        BatchItem("i0", {"download": slow_download, "llm": record}, payload="i0", group="g"),
        BatchItem("i1", {"download": lambda p: p, "llm": record}, payload="i1", group="g"),
        BatchItem("i2", {"download": download2, "llm": record}, payload="i2", group="g"),
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        status_path = Path(tmp) / "status.jsonl"
        results = run_batch(items, workers={"download": 2}, status_path=status_path)
        events = [
            json.loads(line)
            for line in status_path.read_text(encoding="utf-8").splitlines()
        ]
    assert [r.status for r in results] == ["done"] * 3
    assert llm_order == ["i0", "i1", "i2"]
    # The actual dispatch order is on file (plan W5: 可复现性从顺序固定降级为顺序可查).
    scheduled = [e for e in events if e.get("status") == "llm-scheduled"]
    assert [e["label"] for e in scheduled] == ["i0", "i1", "i2"]
    assert [e["order"] for e in scheduled] == [1, 2, 3]
    assert all(e["group"] == "g" for e in scheduled)


def test_the_scheduler_starts_the_longest_ready_item_first() -> None:
    """Plan W5: between groups the pick is LPT on the item's cost; within a
    group, submission order rules; a running group blocks its siblings."""

    sched = scheduler._LlmScheduler(["", "", "", "g", "g"])
    for index, cost in ((0, 10.0), (1, 30.0), (2, 20.0), (4, 99.0)):
        sched.submit(index, cost)
    # Item 3 (group g, earlier sibling) is not ready: item 4 must wait for it
    # no matter its cost, so the solo items go LPT: 30, 20, 10.
    first = sched.next()
    second = sched.next()
    third = sched.next()
    assert (first[0], second[0], third[0]) == (1, 2, 0)
    assert (first[1], second[1], third[1]) == (1, 2, 3)  # dispatch sequence
    assert first[2] == 30.0
    for index in (0, 1, 2):
        sched.finished(index)
    sched.submit(3, 1.0)
    assert sched.next()[0] == 3  # the group's head, not its long tail
    sched.finished(3)
    assert sched.next()[0] == 4
    sched.finished(4)
    assert sched.next() is None


def test_max_parallel_tasks_bounds_llm_concurrency() -> None:
    """W4 acceptance: N pure-API tasks never exceed the admission knob --
    agent-only accounting would let them all start at once."""

    lock = threading.Lock()
    active = 0
    peak = 0

    def llm(payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return payload

    items = [_item(f"i{n}", {"asr": lambda p: p, "llm": llm}) for n in range(6)]
    results = run_batch(items, workers={"llm": 2})
    assert [r.status for r in results] == ["done"] * 6
    assert peak <= 2
    assert peak == 2  # and the knob actually opens past one


def test_groups_stay_serial_even_with_spare_llm_workers() -> None:
    """Plan W5: 组内串行 is a correctness rule, not a scheduling preference."""

    lock = threading.Lock()
    active_in_group = 0
    peak_in_group = 0

    def grouped_llm(payload):
        nonlocal active_in_group, peak_in_group
        with lock:
            active_in_group += 1
            peak_in_group = max(peak_in_group, active_in_group)
        time.sleep(0.05)
        with lock:
            active_in_group -= 1
        return payload

    items = [
        BatchItem(f"g{n}", {"asr": lambda p: p, "llm": grouped_llm}, payload=f"g{n}", group="one")
        for n in range(4)
    ]
    results = run_batch(items, workers={"llm": 3})
    assert [r.status for r in results] == ["done"] * 4
    assert peak_in_group == 1


def test_failed_item_skips_downstream_and_isolates(tmp_path) -> None:
    llm_seen: list[str] = []

    def boom(payload):
        raise RuntimeError("asr exploded")

    def llm(payload):
        llm_seen.append(payload)
        return payload

    items = [
        _item("ok0", {"asr": lambda p: p, "llm": llm}),
        _item("bad", {"asr": boom, "llm": llm}),
        _item("ok2", {"asr": lambda p: p, "llm": llm}),
    ]
    status_path = tmp_path / "status.jsonl"
    # retry_failed=0: this test is about isolation, not about the retry.
    results = run_batch(items, status_path=status_path, retry_failed=0)

    assert results[0].status == "done"
    assert results[1].status == "failed"
    assert results[1].failed_stage == "asr"
    assert "asr exploded" in results[1].error
    assert results[2].status == "done"
    assert sorted(llm_seen) == ["ok0", "ok2"]  # ordered gate advanced past the failure

    events = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    failed = [e for e in events if e["status"] == "failed"]
    assert failed and failed[0]["label"] == "bad" and failed[0]["stage"] == "asr"
    item_events = [e for e in events if e["stage"] == "item"]
    assert {e["label"]: e["status"] for e in item_events} == {
        "ok0": "done",
        "bad": "failed",
        "ok2": "done",
    }


def test_asr_queue_backpressure_bounds_download_lead() -> None:
    release_asr = threading.Event()
    downloads_done = 0
    lock = threading.Lock()

    def download(payload):
        nonlocal downloads_done
        with lock:
            downloads_done += 1
        return payload

    def blocked_asr(payload):
        release_asr.wait(timeout=10)
        return payload

    items = [_item(f"i{n}", {"download": download, "asr": blocked_asr}) for n in range(12)]
    thread_result: list = []
    runner = threading.Thread(
        target=lambda: thread_result.extend(
            run_batch(items, workers={"download": 2}, asr_queue_size=2)
        ),
        daemon=True,
    )
    runner.start()

    # Wait for the download lead to stabilise, then check it stayed bounded:
    # queue capacity (2) + 1 in the asr worker + up to 2 blocked in put().
    cap = 2 + 1 + 2
    last = -1
    for _ in range(100):
        time.sleep(0.02)
        with lock:
            current = downloads_done
        if current == last:
            break
        last = current
    assert last <= cap, f"downloads ran {last} ahead despite backpressure"

    release_asr.set()
    runner.join(timeout=10)
    assert not runner.is_alive()
    assert [r.status for r in thread_result] == ["done"] * 12


def test_stop_event_skips_not_started_items() -> None:
    stop = threading.Event()
    started = threading.Event()

    def first(payload):
        started.set()
        time.sleep(0.05)
        return payload

    def others(payload):
        return payload

    items = [_item("i0", {"asr": first})] + [
        _item(f"i{n}", {"asr": others}) for n in range(1, 6)
    ]

    def trigger():
        started.wait(timeout=10)
        stop.set()

    threading.Thread(target=trigger, daemon=True).start()
    results = run_batch(items, stop_event=stop)
    assert results[0].status == "done"  # in-flight item finishes
    assert any(r.status == "skipped" for r in results[1:])


def test_run_batch_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="workers"):
        run_batch([], workers={"asr": 0})


# --- manifest / option merging ---------------------------------------------------
def test_appended_items_join_a_running_batch(tmp_path) -> None:
    """A batch with an intake grows mid-run and still terminates: the intake
    hands over one late item, the engine runs it, and the batch exits once a
    poll finds nothing new with everything finished."""

    polls = {"count": 0}

    def intake():
        polls["count"] += 1
        if polls["count"] == 1:
            return IntakePoll(
                items=(_item("late", {"asr": lambda p: p, "llm": lambda p: p}),)
            )
        return IntakePoll()

    items = [_item("early", {"asr": lambda p: p, "llm": lambda p: p})]
    status_path = tmp_path / "status.jsonl"
    results = run_batch(
        items, status_path=status_path, intake=intake, intake_poll_seconds=0.5
    )
    assert [r.label for r in results] == ["early", "late"]
    assert [r.status for r in results] == ["done", "done"]
    events = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    added = [e for e in events if e.get("status") == "added"]
    assert [e["label"] for e in added] == ["late"]
    item_events = [e for e in events if e["stage"] == "item"]
    assert {e["label"] for e in item_events} == {"early", "late"}


def test_an_empty_intake_batch_still_terminates() -> None:
    results = run_batch(
        [_item("only", {"llm": lambda p: p})],
        intake=lambda: IntakePoll(),
        intake_poll_seconds=0.5,
    )
    assert [r.status for r in results] == ["done"]


def test_an_intake_that_never_settles_still_lets_a_drained_batch_end() -> None:
    """The other side of it: waiting must be bounded, or a tail nobody ever
    finishes would hang the batch instead of leaving the row to the next run."""

    from finesub.scheduler import INTAKE_UNSETTLED_POLL_LIMIT

    polls = {"n": 0}

    def intake():
        polls["n"] += 1
        return IntakePoll(settled=False, reason="manifest ends mid-line")

    results = run_batch(
        [_item("only", {"llm": lambda p: p})],
        intake=intake,
        intake_poll_seconds=0.5,
    )
    assert [r.status for r in results] == ["done"]
    assert polls["n"] >= INTAKE_UNSETTLED_POLL_LIMIT


def test_a_second_interrupt_leaves_instead_of_waiting() -> None:
    """The first Ctrl-C stops taking new work and lets the in-flight stage
    finish; a second one has to get out. Swallowing every interrupt left no way
    to end a run whose current stage takes hours -- and since a single
    foreground source also comes through here, that is the ordinary
    `finesub one.wav` (reviewer 2026-08-30 P1)."""

    interrupts = {"n": 0}
    released = threading.Event()

    def slow(payload):
        released.wait(10)  # still "running" when both interrupts land
        return payload

    real_join = threading.Thread.join

    def interrupting_join(self, timeout=None):
        if interrupts["n"] < 2:
            interrupts["n"] += 1
            raise KeyboardInterrupt
        released.set()
        return real_join(self, timeout)

    try:
        with pytest.raises(KeyboardInterrupt):
            with unittest.mock.patch.object(
                threading.Thread, "join", interrupting_join
            ):
                run_batch([_item("slow", {"asr": slow})])
    finally:
        released.set()  # let the daemon worker go even if the assertion fails
    assert interrupts["n"] == 2  # the first was absorbed, the second was not


def test_a_failed_stage_is_retried_before_the_item_is_given_up_on(tmp_path, monkeypatch) -> None:
    """The stage, not the item: whatever ran before it is already on disk and
    would be skipped on existence, so a retry costs only the failed part."""

    monkeypatch.setattr(scheduler, "RETRY_BACKOFF_SECONDS", 0.0)
    attempts = {"n": 0}

    def flaky(payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("the network blinked")
        return payload

    status_path = tmp_path / "status.jsonl"
    results = run_batch(
        [_item("flaky", {"asr": flaky, "llm": lambda p: p})], status_path=status_path
    )

    assert results[0].status == "done" and attempts["n"] == 2
    events = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    retries = [e for e in events if e["status"] == "retry"]
    assert len(retries) == 1 and "the network blinked" in retries[0]["error"]
    assert not [e for e in events if e["status"] == "failed"]


def test_retries_are_bounded_and_then_the_item_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(scheduler, "RETRY_BACKOFF_SECONDS", 0.0)
    attempts = {"n": 0}

    def always(payload):
        attempts["n"] += 1
        raise RuntimeError("deterministic")

    results = run_batch([_item("bad", {"asr": always})], retry_failed=2)
    assert attempts["n"] == 3  # the first go plus two retries
    assert results[0].status == "failed"


def test_a_ctrl_c_during_the_retry_backoff_starts_no_retry(monkeypatch) -> None:
    """One Ctrl-C starts no new work, and a retry is new work: the stage may
    be a long ASR, or an LLM stage that spends quota."""

    monkeypatch.setattr(scheduler, "RETRY_BACKOFF_SECONDS", 0.01)

    class InterruptedDuringBackoff(threading.Event):
        def wait(self, timeout=None):
            if timeout == scheduler.RETRY_BACKOFF_SECONDS:
                self.set()
            return super().wait(timeout)

    attempts = {"n": 0}

    def flaky(payload):
        attempts["n"] += 1
        raise ConnectionError("the network blinked")

    results = run_batch(
        [_item("flaky", {"asr": flaky})],
        stop_event=InterruptedDuringBackoff(),
        retry_failed=2,
    )

    assert attempts["n"] == 1
    assert results[0].status == "failed"


def test_a_status_log_that_cannot_be_written_does_not_wedge_the_batch(
    tmp_path, capsys
) -> None:
    """The log is a record, not the work. A full disk used to end the worker
    thread on its first event -- the asr bin has one worker, so the batch then
    waited forever."""

    status_path = tmp_path / "status.jsonl"
    status_path.mkdir()  # opening it for append fails, as a full disk would
    items = [
        _item(f"i{n}", {"download": lambda p: p, "asr": lambda p: p, "llm": lambda p: p})
        for n in range(3)
    ]
    finished: list = []
    runner = threading.Thread(
        target=lambda: finished.append(run_batch(items, status_path=status_path)),
        daemon=True,
    )
    runner.start()
    runner.join(timeout=20)

    assert finished, "the batch wedged"
    assert [result.status for result in finished[0]] == ["done"] * 3
    # Said once, not once per event.
    assert capsys.readouterr().err.count("could not write the batch status log") == 1


def test_a_worker_survives_something_that_is_not_an_exception(tmp_path) -> None:
    """A `SystemExit` from deep in a library used to end the worker THREAD.
    The asr bin has one worker, so its queue then filled and the batch wedged
    -- an item may fail, a bin may not lose its worker."""

    def exits(payload):
        raise SystemExit(3)

    results = run_batch(
        [_item("bad", {"asr": exits}), _item("after", {"asr": lambda p: p})],
        retry_failed=0,
    )
    assert results[0].status == "failed" and "SystemExit" in results[0].error
    # ...and the batch is not cancelled by it: `sys.exit()` on a worker thread
    # ends that thread in plain Python, so it must not read as "stop everything"
    # either (reviewer 2026-08-31 P1).
    assert results[1].status == "done"


def test_priority_beats_size_between_groups() -> None:
    """The case it exists for: something urgent joins a queue that already has
    hours of work in it."""

    sched = scheduler._LlmScheduler(["", "", ""], priorities=[0, 0, 5])
    sched.submit(0, cost=100.0)  # the long one
    sched.submit(1, cost=10.0)
    sched.submit(2, cost=1.0)  # tiny, but urgent
    assert sched.next()[0] == 2
    assert sched.next()[0] == 0  # then LPT as before
    assert sched.next()[0] == 1


def test_an_urgent_item_jumps_the_asr_queue(tmp_path) -> None:
    """Priority on the llm bin alone would not be felt: ASR is the long pole,
    and its queue is where a late arrival waits."""

    started: list[str] = []
    release = threading.Event()

    def slow(payload):
        started.append(payload)
        release.wait(10)
        return payload

    def quick(payload):
        started.append(payload)
        return payload

    items = [_item("first", {"asr": slow})]
    items += [_item(f"plain{n}", {"asr": quick}) for n in range(3)]
    items.append(BatchItem(label="urgent", stages={"asr": quick}, payload="urgent", priority=9))

    def unblock():
        while len(started) < 1:
            time.sleep(0.01)
        time.sleep(0.2)  # let the rest queue up behind it
        release.set()

    unblocker = threading.Thread(target=unblock)
    unblocker.start()
    run_batch(items, workers={"download": 1})
    unblocker.join()

    assert started[0] == "first"
    assert started[1] == "urgent", f"urgent did not jump the queue: {started}"


def test_a_bad_control_action_does_not_stop_the_intake() -> None:
    """The intake thread is the only one that can admit work or end a growing
    batch, so a typo in one action used to silently stop every later addition
    while the batch still ended "successfully"."""

    ran: list[str] = []
    polls = {"n": 0}

    def stage(payload):
        ran.append(payload)
        return payload

    def intake():
        polls["n"] += 1
        if polls["n"] == 1:
            # Same poll: if the action had escaped, the admission behind it
            # would never have happened and the thread would be gone.
            return IntakePoll(
                actions=({"item": "a", "priority": "high"},),
                items=(_item("late1", {"asr": stage}),),
            )
        if polls["n"] == 2:
            return IntakePoll(items=(_item("late2", {"asr": stage}),))
        return IntakePoll()

    items = [_item("a", {"asr": stage})]
    results = run_batch(items, workers={"asr": 1}, intake=intake, intake_poll_seconds=0.05)
    assert sorted(ran) == ["a", "late1", "late2"]
    assert [r.status for r in results] == ["done", "done", "done"]


def test_an_item_under_way_is_not_dropped() -> None:
    """`status` stays `pending` for the whole run, so a drop that only looked
    at it could withdraw an item three minutes into its download."""

    release = threading.Event()
    running = threading.Event()
    polls = {"n": 0}
    states: list[str] = []

    def slow(payload):
        running.set()
        release.wait(10)
        return payload

    def intake():
        running.wait(10)
        polls["n"] += 1
        if polls["n"] == 1:
            return IntakePoll(actions=({"item": "a", "drop": True},))
        release.set()
        return IntakePoll()

    items = [_item("a", {"download": slow})]

    def publish(batch_items, results):
        states.append(results[0].view_state)

    results = run_batch(
        items,
        workers={"download": 1},
        intake=intake,
        intake_poll_seconds=0.05,
        publish=publish,
    )
    assert results[0].status == "done" and not results[0].dropped
    assert "running" in states and states[-1] == "done"


def test_the_view_state_names_what_the_item_is_doing() -> None:
    result = scheduler.ItemResult(label="a")
    assert result.view_state == "queued"
    result.stage = "asr"
    assert result.view_state == "running"
    result.stage = ""
    result.status = "skipped"
    assert result.view_state == "skipped"
    result.dropped = True
    assert result.view_state == "dropped"


def test_the_final_view_is_published_even_when_the_run_is_interrupted() -> None:
    """The view is what a resume reads; it may not describe a run that has
    since moved on."""

    seen: list[list[str]] = []
    items = [_item(name, {"asr": lambda payload: payload}) for name in ("a", "b")]
    stop = threading.Event()
    stop.set()
    run_batch(
        items,
        workers={"asr": 1},
        stop_event=stop,
        publish=lambda batch_items, results: seen.append(
            [r.view_state for r in results]
        ),
    )
    assert seen and seen[-1] == ["skipped", "skipped"]


def test_the_cursor_does_not_move_past_a_view_that_failed_to_publish() -> None:
    """A task admitted but not recorded must be re-read, not retired: the view
    is what a resume reads, so the cursor may not outrun it."""

    committed: list[int] = []
    polls = {"n": 0}

    def intake():
        polls["n"] += 1
        if polls["n"] == 1:
            return IntakePoll(
                items=(_item("late", {"asr": lambda payload: payload}),),
                commit=lambda: committed.append(polls["n"]),
            )
        return IntakePoll(commit=lambda: committed.append(polls["n"]))

    def publish(batch_items, results):
        raise OSError("no space left on device")

    run_batch(
        [_item("a", {"asr": lambda payload: payload})],
        workers={"asr": 1},
        intake=intake,
        intake_poll_seconds=0.05,
        publish=publish,
    )
    # Nothing was ever retired: the view never recorded any of it, and a
    # cursor past a task the view has no sign of loses that task for good.
    assert committed == []


def _lane_item(label: str, lane, asr) -> BatchItem:
    return BatchItem(label=label, stages={"asr": asr}, payload=label, asr_lane=lane)


def test_cloud_items_do_not_wait_behind_local_asr() -> None:
    """A cloud transcription waits on the network, not the GPU: the network lane
    runs its two items side by side while the GPU lane is still busy."""

    release = threading.Event()
    both_cloud = threading.Barrier(2, timeout=5)
    finished: list[str] = []

    def local(payload):
        release.wait(10)
        finished.append(payload)
        return payload

    def cloud(payload):
        both_cloud.wait()  # breaks (and fails the item) unless both run at once
        finished.append(payload)
        if len(finished) == 2:
            release.set()
        return payload

    items = [
        _lane_item("local", lambda p: scheduler.GPU_LANE, local),
        _lane_item("c1", lambda p: scheduler.NETWORK_LANE, cloud),
        _lane_item("c2", lambda p: scheduler.NETWORK_LANE, cloud),
    ]
    results = run_batch(items, workers={"download": 1})
    assert [r.status for r in results] == ["done"] * 3
    assert finished[-1] == "local"


def test_a_lane_that_cannot_be_answered_does_not_sink_the_item() -> None:
    """It takes the GPU lane; the stage itself then says what is wrong."""

    def broken(payload):
        raise OSError("unreadable products")

    asr = lambda p: p  # noqa: E731
    items = [_lane_item("broken", broken, asr), _lane_item("odd", lambda p: "elsewhere", asr)]
    results = run_batch(items, workers={scheduler.NETWORK_WORKERS: 1})
    assert [r.status for r in results] == ["done", "done"]


def test_an_empty_batch_ends_even_with_a_queue_smaller_than_a_lane() -> None:
    """The sentinels of an empty batch go in before any worker runs; the network lane has two."""

    done: list = []
    runner = threading.Thread(target=lambda: done.append(run_batch([], asr_queue_size=1)), daemon=True)
    runner.start()
    runner.join(timeout=5)
    assert done == [[]]


def test_a_reprioritised_item_jumps_its_own_lane() -> None:
    release = threading.Event()
    started: list[str] = []

    def slow(payload):
        started.append(payload)
        release.wait(10)
        return payload

    def quick(payload):
        started.append(payload)
        return payload

    cloud = lambda p: scheduler.NETWORK_LANE  # noqa: E731
    items = [_lane_item("c-first", cloud, slow), _lane_item("c-second", cloud, slow)]
    items += [_lane_item(f"c{n}", cloud, quick) for n in range(3)]
    items.append(_lane_item("late", cloud, quick))
    polls = {"n": 0}

    def intake():
        polls["n"] += 1
        if polls["n"] == 1:
            while len(started) < 2:
                time.sleep(0.01)
            return IntakePoll(actions=({"item": "late", "priority": 5},))
        if polls["n"] == 2:
            release.set()
        return IntakePoll()

    run_batch(items, workers={"download": 1}, intake=intake, intake_poll_seconds=0.05)
    assert started[2] == "late", started


def test_a_growing_batch_with_cloud_items_still_ends() -> None:
    polls = {"n": 0}

    def intake():
        polls["n"] += 1
        if polls["n"] == 1:
            return IntakePoll(items=(_lane_item("late", lambda p: scheduler.NETWORK_LANE, lambda p: p),))
        return IntakePoll()

    items = [_lane_item("a", lambda p: scheduler.NETWORK_LANE, lambda p: p)]
    results = run_batch(items, intake=intake, intake_poll_seconds=0.05)
    assert [r.status for r in results] == ["done", "done"]
