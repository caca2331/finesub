"""One progress line for the correction stage, counted as windows land.

Two things make this stage unlike the others, and both are why the counter is
an object rather than a `for` index:

* **The denominator grows.** A window whose output overran its envelope splits
  on the final attempt, and both halves rejoin the queue as fresh units. The
  serial driver grows its own list; the parallel one grows a queue *inside* a
  worker, leaving the number of futures unchanged. Counting futures would
  therefore report neither the right total nor the right granularity.
* **The counting happens in worker threads.** Like the separator's block
  progress, the reporter is captured here, on the thread that builds this --
  `current_reporter()` is thread-local, and the workers that call `unit_done`
  are not the thread that started the stage.
"""

from __future__ import annotations

import threading

from finesub.reporting import Reporter, current_reporter


STAGE = "translated-srt"


class WindowProgress:
    """Counts finished correction windows against a total that may grow."""

    def __init__(self, total: int) -> None:
        self._reporter: Reporter = current_reporter()
        self._lock = threading.Lock()
        self._total = max(0, int(total))
        self._planned = self._total
        self._done = 0

    def add_units(self, count: int) -> None:
        """A split put `count` more units into the queue."""

        if count <= 0:
            return
        with self._lock:
            self._total += count

    def unit_done(self, chunk_id: str) -> None:
        with self._lock:
            self._done += 1
            done, total = self._done, self._total
        self._reporter.progress(
            STAGE,
            completed=done,
            total=total,
            unit="windows",
            detail=chunk_id,
        )
        self._reporter.debug(
            "correction window done",
            {"chunk": chunk_id, "completed": done, "total": total},
        )

    @property
    def counts(self) -> tuple[int, int]:
        with self._lock:
            return self._done, self._total

    @property
    def splits(self) -> int:
        """Units the queue gained after planning, i.e. windows that split."""

        with self._lock:
            return self._total - self._planned
