"""Parallel-safe admission control (docs/llm_harness_behavior.md).

The limiter instance is shared across threads (``lru_cache``); before P7a its
acquire was "check headroom → sleep → record", so concurrent callers saw the
same headroom and all fired at once. These tests pin the ticket model: slot
computation and accounting are one atomic step under the lock, bookings are
FIFO per bucket, settle targets its own reservation, and the daily strike
gate only counts attempts that departed after the previous strike.
"""

from __future__ import annotations

import threading

import pytest

from finesub.llm.routing.config import ModelEndpoint
from finesub.llm.exchange_log import ExchangeLogger
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.session_checkpoint import SessionCheckpointStore

# effective_rpm = floor(5 * 0.9) = 4, effective_tpm = 225_000
ENDPOINT = ModelEndpoint("GEMINI_FREE", "gemini/gemini-3.5-flash")


def _limiter(tmp_path) -> ModelRateLimiter:
    return ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)


def test_concurrent_reservations_never_exceed_either_window(tmp_path) -> None:
    """Twelve threads reserving at the same instant book disjoint slots."""

    limiter = _limiter(tmp_path)
    window = limiter.policy.window_seconds
    limits = limiter.limits_for(ENDPOINT)
    tickets = []
    guard = threading.Lock()

    def grab() -> None:
        ticket = limiter.reserve(ENDPOINT, 100_000, now_func=lambda: 0.0)
        with guard:
            tickets.append(ticket)

    threads = [threading.Thread(target=grab) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    departs = sorted(ticket.depart_at for ticket in tickets)
    # RPM: no window of `effective_rpm` departures shorter than the window.
    r = limits.effective_rpm
    for i in range(len(departs) - r):
        assert departs[i + r] >= departs[i] + window
    # TPM: 100k each against a 225k cap -> at most two departures per window.
    for i in range(len(departs) - 2):
        assert departs[i + 2] >= departs[i] + window


def test_bookings_are_fifo_within_a_bucket(tmp_path) -> None:
    """A cheap later reservation never departs before an expensive earlier one."""

    limiter = _limiter(tmp_path)
    first = limiter.reserve(ENDPOINT, 200_000, now_func=lambda: 0.0)
    second = limiter.reserve(ENDPOINT, 200_000, now_func=lambda: 0.0)
    third = limiter.reserve(ENDPOINT, 1, now_func=lambda: 0.0)
    assert first.depart_at == 0.0
    assert second.depart_at >= first.depart_at + limiter.policy.window_seconds
    assert third.depart_at >= second.depart_at


def test_settle_by_ticket_adjusts_its_own_event_not_the_newest(tmp_path) -> None:
    limiter = _limiter(tmp_path)
    window = limiter.policy.window_seconds
    early = limiter.reserve(ENDPOINT, 200_000, now_func=lambda: 0.0)
    late = limiter.reserve(ENDPOINT, 200_000, now_func=lambda: 0.0)
    assert late.depart_at == pytest.approx(early.depart_at + window)

    # The early call turns out tiny; its own event shrinks, the late one keeps
    # its full reservation. The legacy last-event fallback would have shrunk
    # the *late* event instead.
    limiter.settle(
        ENDPOINT,
        actual_input_tokens=10_000,
        estimated_input_tokens=200_000,
        ticket=early,
    )
    probe_now = late.depart_at + 1.0  # early's event expired, late's active
    wait = limiter.wait_seconds(ENDPOINT, 100_000, now=probe_now)
    assert wait > 0.0, "the late 200k reservation must still bind"

    limiter.settle(
        ENDPOINT,
        actual_input_tokens=10_000,
        estimated_input_tokens=200_000,
        ticket=late,
    )
    assert limiter.wait_seconds(ENDPOINT, 100_000, now=probe_now) == 0.0


def test_serial_acquire_still_sleeps_to_the_same_slot(tmp_path) -> None:
    """Ticketed acquire keeps the serial contract: 5th call waits one window."""

    limiter = _limiter(tmp_path)
    limits = limiter.limits_for(ENDPOINT)
    clock = [0.0]
    sleeps: list[float] = []

    def now() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    for _ in range(limits.effective_rpm):
        limiter.acquire(ENDPOINT, 100, now_func=now, sleep_func=sleep)
    assert sleeps == []
    limiter.acquire(ENDPOINT, 100, now_func=now, sleep_func=sleep)
    assert sleeps == [pytest.approx(limiter.policy.window_seconds)]


def test_rpm_only_note_ignores_future_tpm_bookings(tmp_path) -> None:
    """A sticky retry waits for real RPM headroom, not sibling TPM bookings.

    Four near-TPM reservations book minutes into the future; a retry of the
    call that departed at t=0 comes back at t=1 needing only RPM. Before the
    2026-08-10 fix it inherited the generic booking slot and fired after the
    whole booking horizon (observed: 182s extra wait).
    """

    limiter = _limiter(tmp_path)
    for _ in range(4):
        limiter.reserve(ENDPOINT, 200_000, now_func=lambda: 0.0)
    note = limiter.reserve(ENDPOINT, 0, token_event=False, now_func=lambda: 1.0)
    assert note.depart_at == 1.0


def test_rpm_only_note_still_respects_rpm(tmp_path) -> None:
    """Notes are exempt from TPM, never from RPM (effective_rpm = 4 here)."""

    limiter = _limiter(tmp_path)
    window = limiter.policy.window_seconds
    limiter.reserve(ENDPOINT, 100, now_func=lambda: 0.0)
    departs = [
        limiter.reserve(
            ENDPOINT, 0, token_event=False, now_func=lambda: 1.0
        ).depart_at
        for _ in range(3)
    ]
    assert departs == [1.0, 1.0, 1.0]
    fifth = limiter.reserve(ENDPOINT, 0, token_event=False, now_func=lambda: 1.0)
    # The rpm-th prior request departed at t=0; its slot frees one window on.
    assert fifth.depart_at == pytest.approx(0.0 + window)


def test_daily_strike_gate_needs_departure_after_the_previous_strike(tmp_path) -> None:
    """N concurrent 429s from one quota exhaustion count as one observation."""

    limiter = _limiter(tmp_path)
    # Three in-flight calls that all departed before any strike existed.
    assert not limiter.note_daily_quota_hit(
        ENDPOINT, key_id="k", now=1.0, departed_at=0.0
    )
    assert not limiter.note_daily_quota_hit(
        ENDPOINT, key_id="k", now=2.0, departed_at=0.5
    )
    assert not limiter.note_daily_quota_hit(
        ENDPOINT, key_id="k", now=3.0, departed_at=0.9
    )
    assert not limiter.is_daily_exhausted(ENDPOINT, key_id="k")

    # Calls that departed after observing the newest strike add real evidence.
    assert not limiter.note_daily_quota_hit(
        ENDPOINT, key_id="k", now=10.0, departed_at=5.0
    )
    assert limiter.note_daily_quota_hit(
        ENDPOINT, key_id="k", now=20.0, departed_at=15.0
    )
    assert limiter.is_daily_exhausted(ENDPOINT, key_id="k")


def test_exchange_logger_concurrent_logs_never_collide(tmp_path) -> None:
    """The old len(glob)+1 handed two writers one number; the later write won."""

    logger = ExchangeLogger(tmp_path / "exchanges")
    paths: list = []
    guard = threading.Lock()

    def log(index: int) -> None:
        path = logger.log(f"call-{index}", response_text=f"reply {index}")
        with guard:
            paths.append(path)

    threads = [threading.Thread(target=log, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(paths)) == 16
    files = sorted((tmp_path / "exchanges").glob("*.md"))
    assert len(files) == 16
    indices = sorted(int(path.name.split("-", 1)[0]) for path in files)
    assert indices == list(range(1, 17))


def test_checkpoint_store_concurrent_commits_all_land(tmp_path) -> None:
    store = SessionCheckpointStore(tmp_path)

    def commit(index: int) -> None:
        store.commit(
            session="query",
            key=f"{index:04d}",
            input_hash=f"sha256:{index:064d}"[:71],
            content=f"content {index}",
        )

    threads = [threading.Thread(target=commit, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    reloaded = SessionCheckpointStore(tmp_path)
    for index in range(8):
        record = reloaded.get("query", f"{index:04d}", f"sha256:{index:064d}"[:71])
        assert record is not None and record.content == f"content {index}"


def test_a_cooldown_reports_when_the_key_becomes_retryable(tmp_path) -> None:
    """The caller has to be able to say how long the wait actually is."""

    from datetime import datetime, timedelta, timezone

    from finesub.llm.rate_limit import COMBO_COOLDOWN_SKIP_SECONDS

    limiter = ModelRateLimiter(state_path=tmp_path / ".state")
    assert limiter.combo_cooldown_retry_at(ENDPOINT, key_id="k1") is None

    # The stamp is stored at millisecond precision, so the lower bound has
    # to be floored the same way or it sits a few microseconds too high.
    before = datetime.now(timezone.utc).replace(microsecond=0)
    limiter.note_combo_exhausted(ENDPOINT, key_id="k1")
    retry_at = limiter.combo_cooldown_retry_at(ENDPOINT, key_id="k1")

    assert retry_at is not None
    window = timedelta(seconds=COMBO_COOLDOWN_SKIP_SECONDS)
    assert before + window <= retry_at <= datetime.now(timezone.utc) + window
    # Per (tier, model, key), like every other combo fact.
    assert limiter.combo_cooldown_retry_at(ENDPOINT, key_id="k2") is None


def test_the_two_reasons_a_key_is_skipped_are_reported_apart() -> None:
    """They differ by hours, and one sentence naming both hid that.

    Observed 2026-09-03: two keys twenty minutes into a back-off, the provider
    console showing five calls all day, and "daily-exhausted or in cooldown"
    read as a spent daily quota.
    """

    from datetime import datetime, timezone

    from finesub.llm.llm_runtime import describe_skipped_keys

    retry_at = datetime(2026, 9, 3, 9, 43, tzinfo=timezone.utc)

    cooling = describe_skipped_keys(
        "GEMINI_FREE", daily=0, cooldown=2, probing=0, cooldown_retry_at=retry_at
    )
    assert "transient cooldown" in cooling
    assert "09:43 UTC" in cooling
    assert "day" not in cooling

    spent = describe_skipped_keys("GEMINI_FREE", daily=2, cooldown=0, probing=0)
    assert "exhausted for the day" in spent
    assert "cooldown" not in spent

    both = describe_skipped_keys(
        "GEMINI_FREE", daily=1, cooldown=1, probing=0, cooldown_retry_at=retry_at
    )
    assert "exhausted for the day" in both and "transient cooldown" in both

    # A key held by someone else's probe is neither, and saying so keeps a
    # reader from waiting out a window that is already being tested.
    probing = describe_skipped_keys("GEMINI_FREE", daily=0, cooldown=0, probing=1)
    assert "already probing" in probing
