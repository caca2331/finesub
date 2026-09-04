"""NVML sampling, so a timing can say what the card was doing while it ran.

Two things this exists to catch:

* **Clock state.** `docs/plans/crispasr-followups.md` -> A3.1: if the SM clock does
  not hold P0 through the ASR stage, that is not merely baseline pollution but
  a real production loss, because chunked decode is exactly the bursty shape
  WDDM penalises.
* **Contention.** A number taken while the desktop was compositing a video is
  not a number about our code. The sampler records the *pre-existing* load so
  the report can flag itself rather than look clean.

NVML is polled from a thread rather than read around the workload, because the
interesting failure -- a clock that sags mid-run -- is invisible at the edges.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Sample:
    t: float
    sm_mhz: int
    graphics_mhz: int
    memory_mhz: int
    pstate: int
    util_gpu: int
    util_memory: int
    power_w: float
    temperature_c: int


@dataclass
class Trace:
    samples: list[Sample] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.samples)

    def summary(self) -> dict[str, object]:
        if not self.samples:
            return {"samples": 0}
        clocks = [s.sm_mhz for s in self.samples]
        memory = [s.memory_mhz for s in self.samples]
        utils = [s.util_gpu for s in self.samples]
        powers = [s.power_w for s in self.samples]
        pstates = [s.pstate for s in self.samples]
        top = max(clocks)
        return {
            "samples": len(self.samples),
            "seconds": self.samples[-1].t - self.samples[0].t,
            "sm_mhz_min": min(clocks),
            "sm_mhz_median": statistics.median(clocks),
            "sm_mhz_max": top,
            "memory_mhz_min": min(memory),
            "memory_mhz_median": statistics.median(memory),
            # The A3.1 verdict. Deliberately expressed in *clocks*, not in the
            # NVML performance-state label: this GeForce card reports P0 while
            # idle and P1 under a saturating CUDA load, so the label inverts the
            # question. The clock does not lie, so sagging is measured as time
            # spent meaningfully below the run's own ceiling.
            "fraction_clock_sagging": sum(1 for c in clocks if c < top * 0.95) / len(clocks),
            "pstates_seen": sorted(set(pstates)),
            "util_gpu_median": statistics.median(utils),
            "util_gpu_max": max(utils),
            "power_w_median": statistics.median(powers),
            "power_w_max": max(powers),
            "temperature_c_max": max(s.temperature_c for s in self.samples),
        }


class _Nvml:
    """Thin, lazily-initialised NVML handle. Absent NVML degrades to no trace."""

    def __init__(self) -> None:
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 - absence is a supported state
            self._nvml = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def sample(self, origin: float) -> Sample | None:
        if self._nvml is None:
            return None
        n, h = self._nvml, self._handle
        try:
            utilization = n.nvmlDeviceGetUtilizationRates(h)
            return Sample(
                t=time.perf_counter() - origin,
                sm_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                graphics_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_GRAPHICS),
                memory_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                pstate=int(n.nvmlDeviceGetPerformanceState(h)),
                util_gpu=utilization.gpu,
                util_memory=utilization.memory,
                power_w=n.nvmlDeviceGetPowerUsage(h) / 1000.0,
                temperature_c=n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            )
        except Exception:  # noqa: BLE001
            return None

    def processes(self) -> list[dict[str, object]]:
        if self._nvml is None:
            return []
        try:
            rows = []
            for proc in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                rows.append({"pid": proc.pid, "used_mb": (proc.usedGpuMemory or 0) // (1 << 20)})
            return rows
        except Exception:  # noqa: BLE001
            return []


_NVML = _Nvml()


class ClockSampler:
    """Context manager that traces the card for the duration of a block."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.trace = Trace()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ClockSampler":
        if not _NVML.available:
            return self
        origin = time.perf_counter()

        def loop() -> None:
            while not self._stop.is_set():
                sample = _NVML.sample(origin)
                if sample is not None:
                    self.trace.samples.append(sample)
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, name="nvml-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def idle_baseline(seconds: float = 2.0) -> dict[str, object]:
    """What the card is doing *before* we touch it.

    A run started against a busy desktop is not invalid, but it must not be
    reported as if the card were ours alone.
    """

    sampler = ClockSampler(interval_s=0.1)
    with sampler:
        time.sleep(seconds)
    summary = sampler.trace.summary()
    summary["compute_processes"] = _NVML.processes()
    summary["contended"] = float(summary.get("util_gpu_median") or 0) > 3.0
    return summary


def available() -> bool:
    return _NVML.available
