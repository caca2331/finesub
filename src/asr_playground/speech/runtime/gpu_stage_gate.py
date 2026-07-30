"""Coordinate GPU model families that each consume most of a profile budget."""

from __future__ import annotations

import threading


class GpuStageLease:
    def __init__(self, gate: "GpuStageGate | None", family: str) -> None:
        self._gate = gate
        self._family = family

    def release(self) -> None:
        gate = self._gate
        if gate is None:
            return
        self._gate = None
        gate.release(self._family)


class GpuStageGate:
    """Allow concurrent work from one model family, but never mixed families."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._family: str | None = None
        self._active = 0
        self._next_family: str | None = None

    def acquire(self, family: str, *, enabled: bool = True) -> GpuStageLease:
        if not enabled:
            return GpuStageLease(None, family)
        with self._condition:
            while True:
                if self._family is None and (
                    self._next_family is None or self._next_family == family
                ):
                    self._family = family
                    self._next_family = None
                    self._active += 1
                    return GpuStageLease(self, family)
                if self._family == family and self._next_family is None:
                    self._active += 1
                    return GpuStageLease(self, family)
                if self._family != family and self._next_family is None:
                    self._next_family = family
                self._condition.wait()

    def release(self, family: str) -> None:
        with self._condition:
            if self._family != family or self._active <= 0:
                raise RuntimeError(f"GPU stage lease mismatch for {family!r}.")
            self._active -= 1
            if self._active == 0:
                self._family = None
                self._condition.notify_all()


GPU_STAGE_GATE = GpuStageGate()
