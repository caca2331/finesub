from __future__ import annotations

from pathlib import Path
import time

import pytest

from asr_playground.speech.runtime import stall_watchdog


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Record faulthandler calls instead of installing a real timer."""

    calls: dict[str, object] = {"scheduled": [], "cancelled": 0}

    def dump_later(interval, repeat=False, exit=False, file=None):
        calls["scheduled"].append((interval, repeat, exit, file))

    def cancel() -> None:
        calls["cancelled"] = int(calls["cancelled"]) + 1

    monkeypatch.setattr(stall_watchdog.faulthandler, "dump_traceback_later", dump_later)
    monkeypatch.setattr(
        stall_watchdog.faulthandler, "cancel_dump_traceback_later", cancel
    )
    monkeypatch.setattr(stall_watchdog, "_ARMED", None)
    monkeypatch.delenv(stall_watchdog.TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(stall_watchdog.LOG_ENV, raising=False)
    return calls


def test_disabled_without_env(recorded, monkeypatch: pytest.MonkeyPatch) -> None:
    watchdog = stall_watchdog.arm("vad-asr")
    try:
        assert not watchdog.active
        assert recorded["scheduled"] == []
    finally:
        watchdog.disarm()
    # A disabled handle must not cancel a timer it never installed.
    assert recorded["cancelled"] == 0


@pytest.mark.parametrize("value", ["", "   ", "0", "-5", "later"])
def test_non_positive_or_unparsable_interval_stays_off(
    recorded, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(stall_watchdog.TIMEOUT_ENV, value)
    watchdog = stall_watchdog.arm("vad-asr")
    assert not watchdog.active
    assert recorded["scheduled"] == []


def test_arm_writes_dumps_to_the_configured_log(
    recorded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "stall.log"
    monkeypatch.setenv(stall_watchdog.TIMEOUT_ENV, "30")
    monkeypatch.setenv(stall_watchdog.LOG_ENV, str(log))

    watchdog = stall_watchdog.arm("vad-asr")
    assert watchdog.active
    interval, repeat, exit_flag, sink = recorded["scheduled"][0]
    # repeat=False is load-bearing: a repeating timer walks other threads'
    # frames without the GIL forever, which crashed a healthy run with an
    # access violation. The kicker is what keeps it from ever firing.
    assert (interval, repeat, exit_flag) == (30.0, False, False)
    assert Path(sink.name) == log
    # The header lands before any stall, so a killed process still shows the
    # run started -- the whole point of not relying on buffered stdio.
    assert "stall watchdog armed: vad-asr" in log.read_text(encoding="utf-8")

    watchdog.disarm()
    assert recorded["cancelled"] == 1
    assert sink.closed
    assert not watchdog.active


def test_nested_arm_keeps_the_outermost_timeline(
    recorded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(stall_watchdog.TIMEOUT_ENV, "30")
    monkeypatch.setenv(stall_watchdog.LOG_ENV, str(tmp_path / "stall.log"))

    outer = stall_watchdog.arm("pipeline")
    inner = stall_watchdog.arm("vad-asr")
    try:
        assert outer.active and not inner.active
        assert len(recorded["scheduled"]) == 1
        # A stage finishing inside the probe must not cancel the probe's timer.
        inner.disarm()
        assert recorded["cancelled"] == 0
        assert outer.active
    finally:
        outer.disarm()
    assert recorded["cancelled"] == 1


def test_kicker_pushes_the_one_shot_timer_back(
    recorded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A tiny interval makes the kicker period ~2.5ms, so a live interpreter
    # re-arms many times over in well under a second.
    monkeypatch.setenv(stall_watchdog.TIMEOUT_ENV, "0.01")
    monkeypatch.setenv(stall_watchdog.LOG_ENV, str(tmp_path / "stall.log"))

    watchdog = stall_watchdog.arm("vad-asr")
    try:
        deadline = time.monotonic() + 2.0
        while len(recorded["scheduled"]) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(recorded["scheduled"]) >= 5
        # Every re-arm stays one-shot; none of them may set repeat.
        assert all(not repeat for _, repeat, _, _ in recorded["scheduled"])
    finally:
        watchdog.disarm()

    # Disarm stops the kicker, so re-arming ceases.
    settled = len(recorded["scheduled"])
    time.sleep(0.1)
    assert len(recorded["scheduled"]) == settled


def test_forced_dump_is_synchronous_and_needs_no_timer(
    recorded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "stall.log"
    monkeypatch.setenv(stall_watchdog.TIMEOUT_ENV, "30")
    monkeypatch.setenv(stall_watchdog.LOG_ENV, str(log))
    dumped: list[bool] = []
    monkeypatch.setattr(
        stall_watchdog.faulthandler,
        "dump_traceback",
        lambda file=None, all_threads=True: dumped.append(all_threads),
    )

    watchdog = stall_watchdog.arm("vad-asr")
    try:
        watchdog.dump_now()
    finally:
        watchdog.disarm()

    assert dumped == [True]
    assert "forced dump" in log.read_text(encoding="utf-8")

    # An inactive handle must stay a no-op rather than write to a closed sink.
    stall_watchdog.StallWatchdog().dump_now()
    assert dumped == [True]


def test_disarm_is_idempotent(
    recorded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(stall_watchdog.TIMEOUT_ENV, "30")
    monkeypatch.setenv(stall_watchdog.LOG_ENV, str(tmp_path / "stall.log"))

    watchdog = stall_watchdog.arm("vad-asr")
    watchdog.disarm()
    watchdog.disarm()
    assert recorded["cancelled"] == 1
