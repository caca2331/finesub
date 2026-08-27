"""Which GPUs this machine has, asked once and never on the caller's thread.

The launcher process has no torch -- it is the worker's runtime that carries
it -- so the list comes from the driver's own `nvidia-smi`. That is a
subprocess, which on a cold driver can take a second or two, and nothing in the
interface may wait on it: the probe runs on a background thread and callers
read whatever the last one produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import threading
from typing import Literal

from finesub_bootstrap.system_tools import no_window


PROBE_TIMEOUT_SECONDS = 5
ProbeState = Literal["scanning", "ready", "unavailable"]


@dataclass(frozen=True, slots=True)
class Gpu:
    index: int
    name: str
    memory_mb: int


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    state: ProbeState
    devices: tuple[Gpu, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "memory_mb": device.memory_mb,
                }
                for device in self.devices
            ],
        }

    def find(self, index: int) -> Gpu | None:
        return next(
            (device for device in self.devices if device.index == index), None
        )


def parse_nvidia_smi(output: str) -> tuple[Gpu, ...]:
    """Read `index, name, memory.total` CSV rows, skipping anything unusable."""

    devices: list[Gpu] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            index = int(fields[0])
            memory_mb = int(float(fields[2]))
        except ValueError:
            # Header rows and driver error text land here.
            continue
        name = fields[1]
        if not name:
            continue
        devices.append(Gpu(index=index, name=name, memory_mb=memory_mb))
    return tuple(sorted(devices, key=lambda device: device.index))


def query_gpus() -> tuple[Gpu, ...]:
    """Run the probe. Blocking -- `GpuProbe` is what keeps it off the UI."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_SECONDS,
            # Without this the packaged app flashes a console window at every
            # probe: it is built --windowed and owns no console of its own.
            creationflags=no_window(),
        )
    except (OSError, subprocess.SubprocessError):
        # No NVIDIA driver, or it did not answer in time. Either way the
        # answer is "nothing to choose from".
        return ()
    if completed.returncode != 0:
        return ()
    return parse_nvidia_smi(completed.stdout)


class GpuProbe:
    """The last known GPU list, refreshed in the background on request."""

    def __init__(self, query=query_gpus) -> None:
        self._query = query
        self._lock = threading.Lock()
        self._snapshot = GpuSnapshot(state="scanning")
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Kick off a probe if none is running. Returns immediately."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._snapshot = GpuSnapshot(
                state="scanning", devices=self._snapshot.devices
            )
            self._thread = threading.Thread(
                target=self._run,
                name="gpu-probe",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        try:
            devices = self._query()
        except Exception:
            devices = ()
        with self._lock:
            self._snapshot = GpuSnapshot(
                state="ready" if devices else "unavailable",
                devices=tuple(devices),
            )

    def snapshot(self) -> GpuSnapshot:
        with self._lock:
            return self._snapshot
