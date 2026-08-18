"""Miscellaneous runtime utility helpers."""

from __future__ import annotations

import sys
import threading
from typing import Optional

import torch

from ...reporting import current_reporter
from .device import cuda_usable


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    size = float(max(0, int(value)))
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"


def _resolve_cuda_device_index(device: Optional[str]) -> Optional[int]:
    if not cuda_usable():
        return None
    if not device:
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return None
    text = str(device).strip().lower()
    if text == "cuda":
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return None
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except (TypeError, ValueError):
            return None
    return None


def reset_peak_gpu_memory_stats_for_run(device: Optional[str]) -> None:
    idx = _resolve_cuda_device_index(device)
    if idx is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        pass


def _windows_working_set_bytes() -> Optional[tuple[int, int]]:
    """Return ``(current, process_lifetime_peak)`` working set, or ``None``."""

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.WinDLL("Kernel32.dll")
        psapi = ctypes.WinDLL("Psapi.dll")

        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE

        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL

        ok = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    except Exception:
        return None


def _peak_process_memory_bytes() -> Optional[int]:
    """Peak resident memory since the process started -- never resets."""

    if sys.platform.startswith("win"):
        counters = _windows_working_set_bytes()
        return None if counters is None else counters[1]
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return int(usage)
    return int(usage * 1024)


def _current_process_memory_bytes() -> Optional[int]:
    """Resident memory right now, or ``None`` where it is not cheaply readable."""

    if sys.platform.startswith("win"):
        counters = _windows_working_set_bytes()
        return None if counters is None else counters[0]
    try:
        # Linux: field 2 of statm is resident pages.
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        import os

        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


# Fast enough to catch the multi-second model-load transients that dominate the
# stage peaks, cheap enough to ignore (one PSAPI call per tick).
STAGE_MEMORY_SAMPLE_SEC = 0.25


class StageMemorySampler:
    """Peak resident memory for a single stage.

    ``PeakWorkingSetSize`` and POSIX ``ru_maxrss`` are process-lifetime peaks
    that no API resets, unlike ``torch.cuda.reset_peak_memory_stats``. In one
    process the second stage therefore inherits the first stage's spike, and in
    a batch run every later task keeps reporting the first task's number -- so a
    real regression is invisible and the budget warning becomes permanent noise.
    Sampling the *current* working set gives each stage its own figure.

    A spike shorter than the sample interval can be missed, so the result is a
    lower bound; ``print_peak_resource_usage`` keeps the process-lifetime peak
    alongside it whenever the two disagree.
    """

    def __init__(self, interval_sec: float = STAGE_MEMORY_SAMPLE_SEC) -> None:
        self._interval = max(0.01, float(interval_sec))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = _current_process_memory_bytes()

    def start(self) -> None:
        if self._thread is not None or self._peak is None:
            return  # already running, or a platform without a current-RSS source
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="stage-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def _observe(self) -> None:
        current = _current_process_memory_bytes()
        if current is not None and current > (self._peak or 0):
            self._peak = current

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval):
            self._observe()

    def stop(self) -> Optional[int]:
        """Stop sampling and return the stage peak, or ``None`` if unsupported."""

        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
            self._observe()
        return self._peak


def start_stage_memory_sampling(
    interval_sec: float = STAGE_MEMORY_SAMPLE_SEC,
) -> StageMemorySampler:
    sampler = StageMemorySampler(interval_sec)
    sampler.start()
    return sampler


def _peak_gpu_memory_bytes(device: Optional[str]) -> Optional[int]:
    idx = _resolve_cuda_device_index(device)
    if idx is None:
        return 0
    try:
        allocated = int(torch.cuda.max_memory_allocated(idx))
        reserved = int(torch.cuda.max_memory_reserved(idx))
    except Exception:
        return None
    return max(allocated, reserved)


def print_peak_resource_usage(
    device: Optional[str],
    profile=None,
    sampler: Optional[StageMemorySampler] = None,
) -> None:
    stage_peak = sampler.stop() if sampler is not None else None
    process_peak = _peak_process_memory_bytes()
    # Compare the stage against the budget; the process figure is contaminated
    # by every earlier stage and cannot fail a profile check on its own.
    peak_mem = stage_peak if stage_peak is not None else process_peak
    peak_gmem = _peak_gpu_memory_bytes(device)
    reporter = current_reporter()
    fields: dict[str, object] = {"peak_mem": _format_bytes(peak_mem)}
    if (
        stage_peak is not None
        and process_peak is not None
        and process_peak > stage_peak
    ):
        fields["peak_mem_process"] = _format_bytes(process_peak)
    fields["peak_gmem"] = _format_bytes(peak_gmem)
    if profile is not None:
        fields["ram_limit"] = _format_bytes(profile.ram_limit_bytes)
        fields["gpu_limit"] = _format_bytes(profile.gpu_limit_bytes)
    # Usage inside the budget is profiling, not progress: it belongs in
    # verbose and in the run metadata. Only going over earns a warning.
    reporter.debug("resource usage", fields)
    if profile is not None:
        try:
            from .resources import resource_limit_violations

            violations = resource_limit_violations(
                peak_gpu_bytes=peak_gmem,
                peak_ram_bytes=peak_mem,
                profile=profile,
            )
        except Exception:
            violations = []
        for violation in violations:
            reporter.warning("resource-budget", str(violation))
