"""Miscellaneous runtime utility helpers."""

from __future__ import annotations

import sys
from typing import Optional

import torch


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
    if not torch.cuda.is_available():
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


def _peak_process_memory_bytes() -> Optional[int]:
    if sys.platform.startswith("win"):
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

            process = get_current_process()
            ok = get_process_memory_info(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                return None
            return int(counters.PeakWorkingSetSize)
        except Exception:
            return None
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


def print_peak_resource_usage(device: Optional[str], profile=None) -> None:
    peak_mem = _peak_process_memory_bytes()
    peak_gmem = _peak_gpu_memory_bytes(device)
    print("Resource usage:")
    print(f"  peak_mem: {_format_bytes(peak_mem)}")
    print(f"  peak_gmem: {_format_bytes(peak_gmem)}")
    if profile is not None:
        print(f"  ram_limit: {_format_bytes(profile.ram_limit_bytes)}")
        print(f"  gpu_limit: {_format_bytes(profile.gpu_limit_bytes)}")
        try:
            from resource_profiles import resource_limit_violations

            violations = resource_limit_violations(
                peak_gpu_bytes=peak_gmem,
                peak_ram_bytes=peak_mem,
                profile=profile,
            )
        except Exception:
            violations = []
        for violation in violations:
            print(f"Warning: {violation}", file=sys.stderr)
