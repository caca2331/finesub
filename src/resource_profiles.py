"""Resource budget profiles for the production ASR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BYTES_PER_GIB = 1024**3
DEFAULT_GPU_BUDGET_GB = 8
GPU_SYSTEM_RESERVE_GB = 0.5
RAM_BUDGET_GB = 8

# Conservative separator sizing guard used by tests. The separator model has a
# fixed footprint plus roughly batch-proportional activation memory.
VOCAL_SEPARATOR_FIXED_GPU_GB = 2.0
VOCAL_SEPARATOR_PER_BATCH_GPU_GB = 1.0


@dataclass(frozen=True)
class ResourceProfile:
    gpu_budget_gb: int
    vocal_separation_batch_size: int
    ram_budget_gb: int = RAM_BUDGET_GB
    gpu_system_reserve_gb: float = GPU_SYSTEM_RESERVE_GB

    @property
    def usable_gpu_gb(self) -> float:
        return float(self.gpu_budget_gb) - float(self.gpu_system_reserve_gb)

    @property
    def gpu_limit_bytes(self) -> int:
        return int(self.usable_gpu_gb * BYTES_PER_GIB)

    @property
    def ram_limit_bytes(self) -> int:
        return int(float(self.ram_budget_gb) * BYTES_PER_GIB)

    @property
    def estimated_vocal_separator_gpu_gb(self) -> float:
        return (
            VOCAL_SEPARATOR_FIXED_GPU_GB
            + VOCAL_SEPARATOR_PER_BATCH_GPU_GB * self.vocal_separation_batch_size
        )


RESOURCE_PROFILES = {
    8: ResourceProfile(gpu_budget_gb=8, vocal_separation_batch_size=4),
    12: ResourceProfile(gpu_budget_gb=12, vocal_separation_batch_size=6),
    16: ResourceProfile(gpu_budget_gb=16, vocal_separation_batch_size=8),
}


def gpu_budget_choices() -> tuple[int, ...]:
    return tuple(sorted(RESOURCE_PROFILES))


def get_resource_profile(gpu_budget_gb: int | str | None = None) -> ResourceProfile:
    if gpu_budget_gb is None:
        gpu_budget_gb = DEFAULT_GPU_BUDGET_GB
    key = int(gpu_budget_gb)
    try:
        return RESOURCE_PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(str(item) for item in gpu_budget_choices())
        raise ValueError(f"Unsupported GPU budget: {gpu_budget_gb}. Use one of: {choices}.") from exc


def resource_limit_violations(
    *,
    peak_gpu_bytes: Optional[int],
    peak_ram_bytes: Optional[int],
    profile: ResourceProfile,
) -> list[str]:
    violations: list[str] = []
    if peak_gpu_bytes is not None and peak_gpu_bytes > profile.gpu_limit_bytes:
        violations.append(
            f"peak_gmem exceeds profile limit ({peak_gpu_bytes} > {profile.gpu_limit_bytes})"
        )
    if peak_ram_bytes is not None and peak_ram_bytes > profile.ram_limit_bytes:
        violations.append(
            f"peak_mem exceeds profile limit ({peak_ram_bytes} > {profile.ram_limit_bytes})"
        )
    return violations
