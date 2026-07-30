"""Opt-in stall diagnostics for long-running GPU stages.

A hung stage used to leave nothing behind: redirected stdio is block-buffered on
Windows, so killing the process discarded every line it had produced. The
``faulthandler`` timer lives on its own native thread, so it still dumps each
thread's Python stack while the GIL is held by a thread blocked inside a C call
-- exactly the state ``sys._current_frames`` cannot observe.

That same property makes it dangerous: the timer thread walks other threads'
frames **without holding the GIL**, and on 3.12 those frames are lazily
materialized, so racing a thread that is churning frames fast can read freed
memory. A repeating timer once crashed a healthy separation run with an access
violation inside python312.dll, mid-dump. So the timer here is **one-shot**, and
a Python-side kicker re-arms it well before it can fire:

* healthy process -> the kicker keeps running -> the timer never fires -> zero
  frame walking, zero risk;
* wedged process (the case worth diagnosing: no Python runs at all) -> the
  kicker is wedged too -> the timer fires exactly once and dumps.

A caller that detects a stall by other means and still has a working
interpreter should call ``dump_now`` instead: dumping from Python holds the GIL,
which blocks the other threads and makes the walk safe.

Off unless ``ASR_STALL_WATCHDOG_SEC`` names a positive interval, so production
runs pay nothing. ``ASR_STALL_WATCHDOG_LOG`` sends the dumps to a file that
survives a kill; without it they go to stderr.

Only the outermost ``arm`` in a process takes effect -- a probe that arms around
a whole pipeline keeps one continuous timeline instead of letting each stage
reset it.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
from typing import IO, Optional, Tuple

TIMEOUT_ENV = "ASR_STALL_WATCHDOG_SEC"
LOG_ENV = "ASR_STALL_WATCHDOG_LOG"

_LOCK = threading.Lock()
_ARMED: "StallWatchdog | None" = None


def _interval_seconds() -> float:
    raw = (os.environ.get(TIMEOUT_ENV) or "").strip()
    if not raw:
        return 0.0
    try:
        interval = float(raw)
    except ValueError:
        print(
            f"Warning: ignoring non-numeric {TIMEOUT_ENV}={raw!r}; "
            "stall watchdog stays off.",
            file=sys.stderr,
        )
        return 0.0
    return interval if interval > 0 else 0.0


def _open_sink() -> Tuple[IO[str], bool]:
    """Return the dump target and whether this module must close it."""

    path = (os.environ.get(LOG_ENV) or "").strip()
    if not path:
        return sys.stderr, False
    try:
        return open(path, "a", encoding="utf-8", buffering=1), True
    except OSError as exc:
        print(
            f"Warning: stall watchdog cannot write {path} ({exc}); using stderr.",
            file=sys.stderr,
        )
        return sys.stderr, False


# The kicker re-arms this many times per interval. Comfortably more than once,
# so ordinary scheduling jitter never fires the timer on a healthy process.
_KICKS_PER_INTERVAL = 4


class StallWatchdog:
    """Handle for one armed watchdog. ``disarm`` is idempotent and safe on the
    inactive handle that nested or disabled ``arm`` calls return."""

    def __init__(
        self,
        sink: Optional[IO[str]] = None,
        *,
        interval: float = 0.0,
        owns_sink: bool = False,
    ) -> None:
        self._sink = sink
        self._interval = interval
        self._owns_sink = owns_sink
        self._stop = threading.Event()
        self._kicker: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        return self._sink is not None

    def _rearm(self) -> None:
        faulthandler.dump_traceback_later(
            self._interval, repeat=False, exit=False, file=self._sink
        )

    def _kick_until_stopped(self) -> None:
        period = self._interval / _KICKS_PER_INTERVAL
        while not self._stop.wait(period):
            # Reaching here proves the interpreter is alive, so push the
            # one-shot timer back out of reach.
            self._rearm()

    def start_kicker(self) -> None:
        self._kicker = threading.Thread(
            target=self._kick_until_stopped,
            name="stall-watchdog-kicker",
            daemon=True,
        )
        self._kicker.start()

    def dump_now(self) -> None:
        """Dump every thread's stack synchronously (safe: this holds the GIL)."""

        if self._sink is None:
            return
        print(
            f"\n===== stall watchdog forced dump "
            f"({time.strftime('%Y-%m-%dT%H:%M:%S')}) =====",
            file=self._sink,
            flush=True,
        )
        faulthandler.dump_traceback(file=self._sink, all_threads=True)

    def disarm(self) -> None:
        global _ARMED
        with _LOCK:
            if _ARMED is not self:
                return
            _ARMED = None
            sink, self._sink = self._sink, None
        self._stop.set()
        kicker, self._kicker = self._kicker, None
        if kicker is not None:
            kicker.join(timeout=2.0)
        faulthandler.cancel_dump_traceback_later()
        if sink is not None and self._owns_sink:
            try:
                sink.close()
            except OSError:
                pass


def arm(label: str) -> StallWatchdog:
    """Start periodic all-thread stack dumps, or return an inactive handle."""

    global _ARMED
    interval = _interval_seconds()
    with _LOCK:
        if interval <= 0 or _ARMED is not None:
            return StallWatchdog()
        sink, owns_sink = _open_sink()
        watchdog = StallWatchdog(sink, interval=interval, owns_sink=owns_sink)
        _ARMED = watchdog

    try:
        sink.write(
            f"\n===== stall watchdog armed: {label} (fires after {interval:g}s "
            f"without a kick, pid={os.getpid()}, "
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}) =====\n"
        )
        sink.flush()
        watchdog._rearm()
        watchdog.start_kicker()
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        # A captured stderr without a real fd is the usual cause; a diagnostic
        # aid must never take the stage down with it.
        print(f"Warning: stall watchdog could not start ({exc}).", file=sys.stderr)
        watchdog.disarm()
        return StallWatchdog()

    print(
        f"Info: stall watchdog armed ({label}, dumps after {interval:g}s "
        "without interpreter progress).",
        file=sys.stderr,
    )
    return watchdog
