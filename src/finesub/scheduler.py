"""Three-bin work runner: download -> asr -> llm.

The engine behind every multi-source form of the pipeline CLI, and the ONE
runner it uses -- a single source is one item with one worker per bin, not a
separate code path (owner 2026-08-30). Items are isolated: one failing item is
recorded and skipped downstream, the rest keep flowing. Resume is inherited
from the pipeline's exist-skip: rerunning the same manifest fast-forwards
finished stages.

Deliberately domain-free: an item is a label plus per-bin callables, and the
runner never learns what a stage does. Everything that knows about pipeline
options, manifests and media lives in `pipeline.py`, which builds the items.
The bins are the resource classes those stages contend for -- network, GPU,
API/agent -- which is why there are three of them and not one per pipeline
stage. The asr bin has two lanes for the same reason: an item transcribed by a
cloud service waits on the network, not the GPU, so it has its own queue and
workers (`NETWORK_LANE`, default two) instead of queueing behind local ASR.
The item says which lane it takes (`BatchItem.asr_lane`); events still say
`asr`.

llm concurrency (task-parallelism plan W4/W5): the llm worker count is the
admission knob (`--max-parallel-tasks`). The in-process rate limiter has
always been lock-protected (`ModelRateLimiter` holds an RLock around every
read-modify-write; only *cross-process* concurrency meters blind), and the
knowledge store takes concurrent writers (per-root single-writer queue + CAS,
plan W2), so nothing below this flag forbids >1. The default is 2 with windows
at 1 (owner 2026-08-30): a preference, not a calibrated constant -- agents
tolerate parallel traffic fine, but too-high settings burn the 5h subscription
quota mid-session and trade token efficiency for diminishing wall-clock gains;
the user picks per their quota pool (docs/manual/agent.md). Scheduling above 1:
items sharing a ``group`` run strictly in submission order (same-video
segments, same-corpus batches); between groups the longest ready item starts
first (LPT, keyed on the stable json's segment count). The actual order is
therefore no longer fixed -- it is *recorded*: every dispatch emits an
``llm-scheduled`` event into batch-status.jsonl.

A batch with an ``intake`` is a GROWING batch: items appended to its source
while it runs join the queue (see `IntakePoll` for what "nothing new" has to
mean before the batch may end).

Presentation is the caller's: `item_reporter` decides how one item speaks
(several items share a terminal, one does not) and `on_item_error` decides
what a failure looks like. The defaults are the multi-item ones.
"""

from __future__ import annotations

import itertools
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import jsonl
from .reporting import (
    LEVELS,
    FanOutReporter,
    NullReporter,
    Reporter,
    TerminalReporter,
    reporting_to,
)

BIN_ORDER = ("download", "asr", "llm")
#: Worse than any real item's `-priority`, so the asr queue hands the sentinels
#: out only once every real entry is gone.
_SENTINEL_PRIORITY = float("inf")
#: Between a failed stage and its retry. Long enough for a rate limit or a
#: flapping connection to matter, negligible next to a stage that just ran for
#: minutes. Waited on the stop event, so an interrupt does not sit through it.
RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_RETRY_FAILED = 1
#: The asr bin's lanes. `asr` in `workers` counts the GPU lane's workers,
#: `NETWORK_WORKERS` the network lane's.
GPU_LANE, NETWORK_LANE = "gpu", "network"
ASR_LANES = (GPU_LANE, NETWORK_LANE)
NETWORK_WORKERS = "asr-network"
DEFAULT_WORKERS = {"download": 2, "asr": 1, NETWORK_WORKERS: 2, "llm": 2}
DEFAULT_ASR_QUEUE_SIZE = 4
DEFAULT_BATCH_ROOT = Path("out") / "batch"
STATUS_FILENAME = "batch-status.jsonl"


# --- engine ------------------------------------------------------------------
@dataclass
class BatchItem:
    label: str
    stages: Mapping[str, Callable[[Any], Any]]  # bin name -> fn(payload)->payload
    payload: Any = None
    # Items sharing a non-empty group run their llm stage strictly in
    # submission order, one at a time (plan W5: 同视频分段 / 同语料批).
    group: str = ""
    # LPT sort key for the llm scheduler: the item's estimated cost, read
    # when its asr stage completes (segment count of the stable json). None
    # or a failing callable both mean 0 -- arrival order breaks the tie.
    llm_cost: Callable[[Any], float] | None = None
    # Higher goes first, in the asr queue and in the llm scheduler alike.
    # Default 0; negative is legal and means "after everything ordinary".
    priority: int = 0
    #: Which asr lane the item's asr stage takes (`ASR_LANES`), asked of its
    #: payload once it reaches the bin: a URL item only learns where its
    #: products are while downloading. None, a failing callable or an unknown
    #: answer mean the GPU lane -- the stage itself then says what is wrong.
    asr_lane: Callable[[Any], str] | None = None
    #: What this item was asked for, opaque to the runner and carried only so
    #: that whoever publishes a view of the run can write it back out. The
    #: runner never reads it -- it does not know what a manifest row is.
    row: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IntakePoll:
    """One look at a growing batch's source of items.

    ``settled`` is the part that matters: the batch ends on an empty poll, so
    "there is nothing more" and "I could not see everything right now" (a row
    still being written, a read that failed) must be different answers.
    """

    items: tuple[BatchItem, ...] = ()
    #: Changes to items already admitted -- `{"item": <label>, "priority": N}`
    #: and `{"item": <label>, "drop": True}`. Same poll as the new items, so a
    #: control channel needs no second clock and no second contract.
    actions: tuple[Mapping[str, Any], ...] = ()
    settled: bool = True
    reason: str = ""
    #: Called once this poll's contents have actually been acted on -- after
    #: the actions are applied and the items are admitted (and therefore
    #: published). A source that has to survive being killed advances its
    #: durable cursor here and NOT when it hands the lines over: a cursor moved
    #: at read time turns "the process died before this took effect" into "this
    #: was already done" (reviewer 2026-08-31 P1).
    commit: Callable[[], None] | None = None


# How many consecutive unsettled polls to wait through before ending a drained
# batch anyway. A row mid-write completes in one poll; a tail that never
# completes (an editor left open, a file gone unreadable) must not hold the
# batch open forever -- it is picked up by the next run, exactly like resume.
INTAKE_UNSETTLED_POLL_LIMIT = 3


@dataclass
class ItemResult:
    label: str
    status: str = "pending"  # pending -> done | failed | skipped
    failed_stage: str = ""
    error: str = ""
    payload: Any = None
    #: The bin currently executing this item, "" between bins and at rest.
    #: `status` alone cannot answer "is this under way": it stays `pending`
    #: from admission until the item is terminal, so a view built from it
    #: shows nothing running and a drop cannot tell a queued item from one
    #: three minutes into its download (reviewer 2026-08-31 P1).
    stage: str = ""
    #: Withdrawn by a control action, as opposed to skipped by an interrupt or
    #: by an upstream failure. Both end up `skipped`, and only one of them
    #: should come back when the run is resumed.
    dropped: bool = False

    @property
    def view_state(self) -> str:
        """One word for what this item is doing, for whoever publishes a view.

        The runner owns its own state machine, so the mapping from `status` +
        `stage` to a name lives here rather than in each front end.
        """

        if self.dropped:
            return "dropped"
        if self.status != "pending":
            return self.status
        return "running" if self.stage else "queued"


class _StatusLog:
    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._warned = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, **event: Any) -> None:
        """Append one event. A record, not the work: a failed write is dropped.

        Every worker emits outside its item's error handling, so an OSError
        here (a full disk, a viewer holding the file without write sharing)
        used to end the worker thread -- and the asr bin's only worker gone
        wedges the whole batch. Said once, not once per event.
        """

        if self._path is None:
            return
        event.setdefault(
            "ts", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        with self._lock:
            try:
                jsonl.append(self._path, event)
            except OSError as exc:
                if not self._warned:
                    self._warned = True
                    print(
                        f"Warning: could not write the batch status log "
                        f"{self._path} ({exc}); the batch goes on without it",
                        file=sys.stderr,
                    )


class _LlmScheduler:
    """Which ready item the llm bin runs next (task-parallelism plan W5).

    Replaces the strict submission-order gate: groups serialize internally in
    submission order (their knowledge accumulation must stay ordered), while
    BETWEEN groups the highest `priority` runs first and, at equal priority,
    the longest ready item (LPT -- long tails need short tasks to fill them).
    An ungrouped item is its own group. The order is no longer fixed, so every
    dispatch is recorded (`llm-scheduled` in batch-status.jsonl):
    reproducibility degrades from "fixed" to "on file".

    Priority exists for one case, and it is the case a growing batch creates:
    something urgent is appended to a manifest that already has hours of work
    queued. Without it the newcomer competes on size alone.
    """

    def __init__(
        self,
        groups: Sequence[str],
        *,
        closed: bool = True,
        priorities: Sequence[int] | None = None,
    ) -> None:
        self._cond = threading.Condition()
        self._groups = [group or f"~solo-{index}" for index, group in enumerate(groups)]
        self._priorities = list(priorities or [0] * len(self._groups))
        self._ready: dict[int, float] = {}
        self._dispatched: set[int] = set()
        self._finished: set[int] = set()
        self._running_groups: set[str] = set()
        self._sequence = 0
        # An open scheduler expects more items (a live intake): workers park
        # instead of exiting once everything currently known has dispatched.
        self._closed = bool(closed)

    def add(self, group: str, priority: int = 0) -> int:
        """Register one late item (live intake); returns its index."""

        with self._cond:
            index = len(self._groups)
            self._groups.append(group or f"~solo-{index}")
            self._priorities.append(int(priority))
            self._cond.notify_all()
            return index

    def set_priority(self, index: int, priority: int) -> None:
        """Re-rank an item that has not been dispatched yet.

        Not a preemption: whatever is running keeps running, and an item
        already handed to a worker keeps its place. Reordering the queue is
        what a person asking for "this one next" actually wants -- stopping
        work already paid for is not.
        """

        with self._cond:
            self._priorities[index] = int(priority)
            self._cond.notify_all()

    def dispatched(self, index: int) -> bool:
        with self._cond:
            return index in self._dispatched

    def close(self) -> None:
        """No more items will ever be added: idle workers may leave."""

        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def submit(self, index: int, cost: float = 0.0) -> None:
        with self._cond:
            self._ready[index] = float(cost)
            self._cond.notify_all()

    def finished(self, index: int) -> None:
        with self._cond:
            self._finished.add(index)
            self._running_groups.discard(self._groups[index])
            self._cond.notify_all()

    def _eligible(self, index: int) -> bool:
        group = self._groups[index]
        if group in self._running_groups:
            return False
        # 组内提交序: every earlier member must have finished its llm stage
        # (or been skipped through it) before the next may start.
        return all(
            other in self._finished
            for other in range(index)
            if self._groups[other] == group
        )

    def next(self) -> Optional[tuple[int, int, float]]:
        """(index, dispatch sequence, cost), or None when everything ran."""

        with self._cond:
            while True:
                if self._closed and len(self._dispatched) >= len(self._groups):
                    return None
                candidates = [
                    index
                    for index in self._ready
                    if index not in self._dispatched and self._eligible(index)
                ]
                if candidates:
                    # Priority, then LPT between groups; submission order
                    # breaks the remaining ties.
                    best = max(
                        candidates,
                        key=lambda index: (
                            self._priorities[index],
                            self._ready[index],
                            -index,
                        ),
                    )
                    self._dispatched.add(best)
                    self._running_groups.add(self._groups[best])
                    self._sequence += 1
                    return best, self._sequence, self._ready[best]
                self._cond.wait()


# Knowledge write-path warnings that must reach batch-status.jsonl: under
# task-level parallelism a typed conflict is the batch-visible signal that one
# task's proposals lost a race to another's (task-parallelism plan W2), and a
# per-item terminal line scrolls away while the status log is the record.
KNOWLEDGE_STATUS_CODES = (
    "knowledge-apply-conflict",
    "knowledge-apply-rolled-back",
    "knowledge-locked",
)


class _KnowledgeStatusTee(NullReporter):
    """Mirror the knowledge conflict warnings of one item into the status log."""

    def __init__(self, status: "_StatusLog", index: int, label: str, stage: str) -> None:
        self._status = status
        self._index = index
        self._label = label
        self._stage = stage

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        if code in KNOWLEDGE_STATUS_CODES:
            self._status.emit(
                item=self._index,
                label=self._label,
                stage=self._stage,
                status="knowledge-conflict",
                code=code,
                message=message,
            )


def default_item_reporter(label: str, log_level: str) -> TerminalReporter:
    """A renderer that says which item it is speaking for.

    Line mode, never redrawn in place: several items share this terminal, and a
    progress line rewritten by whichever one spoke last leaves neither
    readable. A one-item run passes its own factory instead and keeps the
    in-place progress it always had.
    """

    return TerminalReporter(
        sys.stderr,
        level=log_level if log_level in LEVELS else "normal",
        isatty=False,
        prefix=f"[{label}] ",
    )


def run_batch(
    items: Sequence[BatchItem],
    *,
    workers: Mapping[str, int] | None = None,
    asr_queue_size: int | None = None,
    status_path: str | Path | None = None,
    stop_event: threading.Event | None = None,
    log_level: str = "normal",
    intake: Callable[[], IntakePoll] | None = None,
    intake_poll_seconds: float = 5.0,
    retry_failed: int | None = None,
    item_reporter: Callable[[str], Reporter] | None = None,
    on_item_error: Callable[[ItemResult, BaseException], None] | None = None,
    publish: Callable[[Sequence[BatchItem], Sequence[ItemResult]], None] | None = None,
) -> list[ItemResult]:
    """Run items through the three bins; returns one ItemResult per item.

    ``intake`` makes the batch a growing one: it is polled every
    ``intake_poll_seconds`` for items to add mid-run (the manifest CLI feeds
    appended manifest lines through it). The batch ends once every known item
    is finished AND a *settled* poll at that moment returns nothing -- a row
    appended after that final check is picked up by the next run, exactly like
    resume. Without ``intake`` the behavior is unchanged.

    ``retry_failed`` retries a FAILED STAGE that many times before the item is
    given up on (None = ``DEFAULT_RETRY_FAILED``). The stage, not the item: whatever ran before it
    already left its artifacts and would be skipped on existence, so a retry
    costs only the part that failed. ``asr_queue_size`` None = ``DEFAULT_ASR_QUEUE_SIZE``.

    ``item_reporter`` builds one item's reporter from its label (default: the
    prefixed line-mode one several items can share); ``on_item_error`` is
    called with the failed result and the exception (default: one stderr
    line). Both exist so a one-item run can keep its own presentation --
    in-place progress, a traceback in the run log -- while going through this
    same runner."""

    attempts = 1 + max(0, DEFAULT_RETRY_FAILED if retry_failed is None else int(retry_failed))
    counts = dict(DEFAULT_WORKERS)
    counts.update(workers or {})
    for name in (*BIN_ORDER, NETWORK_WORKERS):
        if counts.get(name, 0) < 1:
            raise ValueError(f"workers[{name!r}] must be >= 1")

    items = list(items)
    results = [ItemResult(label=item.label, payload=item.payload) for item in items]
    status = _StatusLog(Path(status_path) if status_path else None)
    stop = stop_event or threading.Event()

    def emit(**event: Any) -> None:
        """Record the event AND refresh the published view.

        Together on purpose: the view is a rendering of the same transitions
        the log records, so anything worth a line is worth a refresh, and no
        state change can update one without the other.
        """

        status.emit(**event)
        _publish()

    def _publish() -> bool:
        """Whether this run's state is now on disk. A view is not the work --
        a failure to write one never stops a batch -- but it IS what a resume
        reads, so an intake source may not retire a line the view has not
        recorded (see the intake loop)."""

        if publish is None:
            return True
        try:
            publish(items, results)
        except Exception as exc:  # noqa: BLE001 -- see above
            print(f"[batch] could not publish the queue view: {exc}", file=sys.stderr)
            return False
        return True

    make_reporter = item_reporter or (
        lambda label: default_item_reporter(label, log_level)
    )

    download_q: queue.Queue = queue.Queue()
    lanes = _AsrLanes(
        {GPU_LANE: counts["asr"], NETWORK_LANE: counts[NETWORK_WORKERS]}, asr_queue_size, stop
    )
    scheduler = _LlmScheduler(
        [item.group for item in items],
        closed=intake is None,
        priorities=[item.priority for item in items],
    )
    forward_lock = threading.Lock()
    forwarded_to_asr = 0
    completed_items = 0
    intake_closed = intake is None
    asr_sentinels_sent = False

    def _llm_cost(index: int) -> float:
        item, result = items[index], results[index]
        if item.llm_cost is None or result.status in ("failed", "skipped"):
            return 0.0
        try:
            return float(item.llm_cost(result.payload))
        except Exception:  # noqa: BLE001 -- a sort key must never sink an item
            return 0.0

    def _run_stage(bin_name: str, index: int) -> None:
        item, result = items[index], results[index]
        fn = item.stages.get(bin_name)
        if fn is None:
            return
        # Claiming the item and reading its verdict happen together: a drop
        # arriving between the two would be told "still queued" while this
        # thread was already starting the stage.
        with forward_lock:
            if result.status in ("failed", "skipped"):
                return  # upstream (or a control drop) already resolved it
            interrupted = stop.is_set()
            if interrupted:
                result.status = "skipped"
            else:
                result.stage = bin_name
        if interrupted:
            emit(item=index, label=item.label, stage=bin_name, status="skipped")
            return
        emit(item=index, label=item.label, stage=bin_name, status="started")
        for attempt in range(1, attempts + 1):
            try:
                # Bound here rather than by each caller: reference_ingest builds
                # its own items and calls run_batch directly, so a binding that
                # lived in `_build_item` reached only the manifest CLI -- and
                # everything the pipeline reports, warnings included, went to a
                # reporter that shows nothing.
                with reporting_to(
                    FanOutReporter(
                        make_reporter(item.label),
                        _KnowledgeStatusTee(status, index, item.label, bin_name),
                    )
                ):
                    result.payload = fn(result.payload)
                break
            # BaseException, not Exception: something that is not an Exception
            # (a SystemExit from deep in a library, say) used to end the worker
            # THREAD instead of the item -- and the asr bin has exactly one
            # worker, so its queue then filled and the whole batch wedged. An
            # item may fail; a bin may not lose its worker.
            #
            # And it is the ITEM that fails, not the batch: `sys.exit()` on a
            # worker thread ends that thread and nothing else in plain Python,
            # so treating it as a global stop would let one library call cancel
            # everything else (reviewer 2026-08-31 P1). A real interrupt
            # reaches the main thread, which is where the batch is stopped.
            except BaseException as exc:  # noqa: BLE001 -- see above
                if attempt < attempts and not stop.is_set():
                    # Retrying the STAGE, not the item: the stages before it
                    # already produced their artifacts and would be skipped on
                    # existence anyway, so this is the cheap half of a rerun.
                    emit(
                        item=index,
                        label=item.label,
                        stage=bin_name,
                        status="retry",
                        attempt=attempt,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    print(
                        f"[batch] {item.label}: {bin_name} failed "
                        f"({type(exc).__name__}: {exc}); retrying "
                        f"({attempt}/{attempts - 1})",
                        file=sys.stderr,
                    )
                    if not stop.wait(RETRY_BACKOFF_SECONDS):  # Ctrl-C meanwhile: no retry
                        continue
                result.status = "failed"
                result.stage = ""
                result.failed_stage = bin_name
                result.error = f"{type(exc).__name__}: {exc}"
                emit(
                    item=index,
                    label=item.label,
                    stage=bin_name,
                    status="failed",
                    error=result.error,
                )
                if on_item_error is not None:
                    on_item_error(result, exc)
                else:
                    print(
                        f"[batch] {item.label}: {bin_name} failed: {result.error}",
                        file=sys.stderr,
                    )
                return
        result.stage = ""
        emit(item=index, label=item.label, stage=bin_name, status="done")

    def _maybe_close_asr_locked() -> bool:
        """Whether the caller must now send the asr sentinels, decided here.

        Decided under `forward_lock` (the flag makes it once-only), sent
        outside it: `_AsrLanes.put` waits for room in a bounded queue, and every
        stage start needs that same lock -- so putting from in here would hold
        the batch's bookkeeping still for as long as the asr bin is busy.
        """

        nonlocal asr_sentinels_sent
        if intake_closed and not asr_sentinels_sent and forwarded_to_asr == len(items):
            asr_sentinels_sent = True
            return True
        return False

    def _send_asr_sentinels(send: bool) -> None:
        if send:
            lanes.close()

    def _lane(index: int) -> str:
        item, result = items[index], results[index]
        if item.asr_lane is None or result.status in ("failed", "skipped"):
            return GPU_LANE
        try:
            lane = item.asr_lane(result.payload)
        except Exception:  # noqa: BLE001 -- choosing a queue must never sink an item
            return GPU_LANE
        return lane if lane in ASR_LANES else GPU_LANE

    def _forward_to_asr(index: int) -> None:
        nonlocal forwarded_to_asr
        lanes.put(_lane(index), -items[index].priority, index)
        with forward_lock:
            forwarded_to_asr += 1
            sentinels = _maybe_close_asr_locked()
        _send_asr_sentinels(sentinels)

    def _download_worker() -> None:
        while True:
            index = download_q.get()
            if index is None:
                return
            _run_stage("download", index)
            _forward_to_asr(index)

    def _asr_worker(lane: str) -> None:
        while True:
            _priority, _arrival, index = lanes.queues[lane].get()
            if index is None:
                return
            _run_stage("asr", index)
            scheduler.submit(index, _llm_cost(index))

    def _llm_worker() -> None:
        nonlocal completed_items
        while True:
            slot = scheduler.next()
            if slot is None:
                return
            index, sequence, cost = slot
            try:
                # The dispatch order is the record (plan W5); inside the `try`
                # so a failed write cannot leave the item's group unfinished.
                emit(
                    item=index,
                    label=items[index].label,
                    stage="llm",
                    status="llm-scheduled",
                    order=sequence,
                    cost=cost,
                    **({"group": items[index].group} if items[index].group else {}),
                )
                _run_stage("llm", index)
            finally:
                scheduler.finished(index)
            result = results[index]
            if result.status == "pending":
                result.status = "done"
            emit(
                item=index,
                label=items[index].label,
                stage="item",
                status=result.status,
                **({"error": result.error} if result.error else {}),
            )
            with forward_lock:
                completed_items += 1

    def _close_intake() -> None:
        nonlocal intake_closed
        with forward_lock:
            if intake_closed:
                return
            intake_closed = True
            sentinels = _maybe_close_asr_locked()
        _send_asr_sentinels(sentinels)
        for _ in range(counts["download"]):
            download_q.put(None)
        scheduler.close()

    def _apply_actions(actions: Sequence[Mapping[str, Any]]) -> None:
        """`{"item": <label>, "priority": N}` / `{"item": <label>, "drop": True}`.

        Addressed by label because that is what the published view shows and
        what a person types; an index would be an internal number that shifts
        as a growing batch admits more. Anything already dispatched is left
        alone and said so -- see `_LlmScheduler.set_priority`.
        """

        for action in actions:
            # One bad action is one bad action. This runs on the intake
            # thread -- the only thread that can admit work or end a growing
            # batch -- so letting `{"priority": "high"}` out of here killed
            # the thread and silently stopped every later addition, in a batch
            # that still ended "successfully" (reviewer 2026-08-31 P1).
            try:
                _apply_action(action)
            except Exception as exc:  # noqa: BLE001 -- see above
                print(f"[batch] control action skipped: {exc}", file=sys.stderr)

    def _apply_action(action: Mapping[str, Any]) -> None:
        label = str(action.get("item") or "")
        with forward_lock:
            matches = [i for i, item in enumerate(items) if item.label == label]
        if not matches:
            print(f"[batch] control: no item named {label!r}", file=sys.stderr)
            return
        index = matches[0]
        if action.get("drop"):
            with forward_lock:
                busy = (
                    results[index].status != "pending"
                    or results[index].stage
                    or scheduler.dispatched(index)
                )
                if not busy:
                    results[index].status = "skipped"
                    results[index].dropped = True
            if busy:
                print(
                    f"[batch] control: {label} is already under way, not dropped",
                    file=sys.stderr,
                )
                return
            emit(item=index, label=label, stage="control", status="dropped")
            print(f"[batch] control: {label} dropped", file=sys.stderr)
            return
        if "priority" in action:
            priority = int(action["priority"])
            items[index].priority = priority
            scheduler.set_priority(index, priority)
            lanes.rekey(lambda index: -items[index].priority)
            emit(
                item=index,
                label=label,
                stage="control",
                status="reprioritised",
                priority=priority,
            )
            print(f"[batch] control: {label} priority -> {priority}", file=sys.stderr)

    def _admit(new_items: Sequence[BatchItem]) -> None:
        for item in new_items:
            with forward_lock:
                index = len(items)
                items.append(item)
                results.append(ItemResult(label=item.label, payload=item.payload))
            scheduler.add(item.group, item.priority)
            emit(
                item=index,
                label=item.label,
                stage="intake",
                status="added",
                **({"group": item.group} if item.group else {}),
            )
            download_q.put(index)

    def _intake_loop() -> None:
        # The one thread that decides the batch is over: every known item is
        # terminal AND a SETTLED poll at that very moment brings nothing new. A
        # row appended after that check belongs to the next run (resume picks
        # it up), which keeps the exit decision race-free without a stop file.
        # An unsettled poll (a row mid-write, a failed read) is not evidence of
        # "nothing more" -- but it cannot hold a drained batch open forever
        # either, hence the streak limit.
        unsettled = 0
        while True:
            if stop.is_set():
                _close_intake()
                return
            try:
                poll = intake()
                if not isinstance(poll, IntakePoll):
                    raise TypeError(
                        f"intake must return IntakePoll, got {type(poll).__name__}"
                    )
            except Exception as exc:  # noqa: BLE001 -- a bad poll must not kill the batch
                print(f"[batch] intake poll failed: {exc}", file=sys.stderr)
                poll = IntakePoll(settled=False, reason=f"{type(exc).__name__}: {exc}")
            if not poll.settled:
                unsettled += 1
                print(
                    f"[batch] intake poll inconclusive ({poll.reason or 'unknown'}); "
                    f"waiting ({unsettled}/{INTAKE_UNSETTLED_POLL_LIMIT})",
                    file=sys.stderr,
                )
            else:
                unsettled = 0
            if poll.actions:
                _apply_actions(poll.actions)
            if poll.items:
                _admit(poll.items)
            if poll.commit is not None:
                # Retire lines only once what they did is on disk. A commit is
                # offered whenever the source still has unretired ground, not
                # only on the poll that read it -- so this is also the retry
                # for a publish that failed a moment ago, and it is why the
                # test is not "did THIS poll bring anything" (that let the
                # next, empty poll retire the failed one: reviewer 2026-08-31
                # P1). A cursor past a task the view never recorded loses it
                # for good; re-publishing is the cheap direction.
                if _publish():
                    try:
                        poll.commit()
                    except Exception as exc:  # noqa: BLE001 -- a cursor is not the work
                        print(
                            f"[batch] intake could not record its cursor: {exc}",
                            file=sys.stderr,
                        )
            if not poll.items and (
                poll.settled or unsettled >= INTAKE_UNSETTLED_POLL_LIMIT
            ):
                with forward_lock:
                    drained = completed_items == len(items)
                if drained:
                    if not poll.settled:
                        print(
                            "[batch] giving up on the incomplete manifest tail; "
                            "rerun to pick it up",
                            file=sys.stderr,
                        )
                    _close_intake()
                    return
            stop.wait(max(0.5, float(intake_poll_seconds)))

    if publish is not None:
        emit(stage="run", status="started", items=len(items))
    for index in range(len(items)):
        download_q.put(index)
    if intake is None:
        for _ in range(counts["download"]):
            download_q.put(None)
        if len(items) == 0:
            _send_asr_sentinels(True)

    def _intake_thread() -> None:
        # Whatever happens in there, the batch must still be able to end: the
        # intake thread is the only one that can close a growing batch, so an
        # unexpected death would leave every other worker parked forever.
        try:
            _intake_loop()
        finally:
            _close_intake()

    threads = (
        [threading.Thread(target=_download_worker, daemon=True) for _ in range(counts["download"])]
        + [
            threading.Thread(target=_asr_worker, args=(lane,), daemon=True)
            for lane in ASR_LANES
            for _ in range(lanes.workers[lane])
        ]
        + [threading.Thread(target=_llm_worker, daemon=True) for _ in range(counts["llm"])]
        + (
            [threading.Thread(target=_intake_thread, daemon=True)]
            if intake is not None
            else []
        )
    )
    for thread in threads:
        thread.start()
    try:
        _join(threads, stop)
    finally:
        # The view is what a resume reads, so it may not be left describing a
        # run that has since moved on -- including the runs that end by
        # interrupt, which are exactly the ones somebody comes back to.
        _publish()
    return results


class _AsrLanes:
    """The asr bin's queues, one per lane, and the lock that lets a re-key hold them still.

    A priority queue per lane, so an urgent late arrival does not sit behind
    the ASR of everything already queued -- that is the long pole, and the
    reason priority on the llm bin alone would not be felt. Entries are
    `(-priority, arrival, index | None)`: the sentinel carries the worst
    priority and the largest arrival, so it is always taken last.
    """

    def __init__(self, workers: Mapping[str, int], size: int | None, stop: threading.Event) -> None:
        self.workers = dict(workers)
        size = DEFAULT_ASR_QUEUE_SIZE if size is None else size
        # Never smaller than the lane's worker count: closing an empty batch puts
        # every sentinel before any worker has started to take them.
        self.queues: dict[str, queue.PriorityQueue] = {
            lane: queue.PriorityQueue(maxsize=max(1, int(size), self.workers[lane])) for lane in ASR_LANES
        }
        self._arrivals = itertools.count()
        self._stop = stop
        # Guards the queues' ORDER, not their contents: only a re-key needs a
        # queue to hold still, and only against producers.
        self._lock = threading.Lock()

    def put(self, lane: str, priority: float, index: int | None) -> None:
        """Enqueue without ever holding the lock while waiting for room.

        The lock exists so a re-key (below) can drain and refill the queue with
        nobody adding in between; blocking inside it would make an urgent
        item's reorder wait out whichever ASR is running. Back pressure is
        unchanged -- the queue is still bounded, this just waits outside.
        """

        entry = (priority, next(self._arrivals), index)
        asr_q = self.queues[lane]
        while True:
            with self._lock:
                try:
                    asr_q.put_nowait(entry)
                    return
                except queue.Full:
                    pass
            if self._stop.wait(0.05):
                # Winding down: take the slot as soon as one frees so the
                # sentinels behind this entry are still reachable.
                with self._lock:
                    asr_q.put(entry)
                    return

    def close(self) -> None:
        """One sentinel per worker, on every lane."""

        for lane in ASR_LANES:
            for _ in range(self.workers[lane]):
                self.put(lane, _SENTINEL_PRIORITY, None)

    def rekey(self, priority_of: Callable[[int], float]) -> None:
        """Re-sort the queues after a priority change.

        A PriorityQueue orders by the key an entry was enqueued WITH, so an
        item already waiting keeps the rank it had when it arrived -- which is
        every item worth reprioritising, since the whole point is that it is
        already in line. Nothing is lost or gained: what comes out goes back
        in, so `put_nowait` cannot overflow a queue we just emptied.
        """

        with self._lock:
            for asr_q in self.queues.values():
                drained: list[tuple[float, int, int | None]] = []
                while True:
                    try:
                        drained.append(asr_q.get_nowait())
                    except queue.Empty:
                        break
                for _priority, arrival, index in drained:
                    asr_q.put_nowait(
                        (_SENTINEL_PRIORITY if index is None else priority_of(index), arrival, index)
                    )


def _join(threads: Sequence[threading.Thread], stop: threading.Event) -> None:
    for thread in threads:
        while thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except KeyboardInterrupt:
                if stop.is_set():
                    # The second one means it. Swallowing every interrupt left
                    # no way out of a run whose current stage takes hours --
                    # and since a single foreground source also comes through
                    # here, that is the ordinary `finesub one.wav` (reviewer
                    # 2026-08-30 P1). The workers are daemons: re-raising ends
                    # the process, and the exist-skip resume picks the run up.
                    print(
                        "[batch] second interrupt: leaving the in-flight "
                        "stage behind (rerun to resume)",
                        file=sys.stderr,
                    )
                    raise
                stop.set()
                print(
                    "[batch] interrupt: finishing in-flight stages, "
                    "skipping the rest (rerun to resume; interrupt again to "
                    "stop now)",
                    file=sys.stderr,
                )
