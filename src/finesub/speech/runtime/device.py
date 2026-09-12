"""Whether this machine's GPU can run the stack, decided in one place.

Two different situations end in the same CPU fallback: there is no usable CUDA
at all (no driver, no card, ``CUDA_VISIBLE_DEVICES`` emptied), and there is a
card whose architecture the installed PyTorch ships no kernels for. The second
one is why this module exists -- torch reports ``is_available() == True`` for a
GTX 1060, and the mismatch only surfaces at the first real op as ``no kernel
image is available for execution on the device``, which names neither the card
nor the fix.

The check is deliberately derived from ``torch.cuda.get_arch_list()`` rather
than a hard-coded floor: the supported set is a property of the installed wheel,
so it moves on its own when the torch pin moves.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

from ...reporting import current_reporter

# Spelled as product lines because that is what a user can actually check on
# their own machine. sm_75 is the oldest consumer architecture in the cu128
# wheel's kernel list; sm_70 covers the datacenter TITAN V / V100. Keep in sync
# with README.md's hardware table.
SUPPORTED_GPU_HINT = "GTX 1650/1660 or RTX 20/30/40/50 series and newer"

#: The oldest architecture included in the patched CTranslate2 wheel. Unlike
#: the PyTorch floor above, CT2 does not expose its compiled architecture list,
#: so this moves only when that wheel is rebuilt.
CT2_MIN_COMPUTE_CAPABILITY = (7, 0)


def _arch_number(entry: str) -> Optional[int]:
    """``sm_86`` / ``compute_90a`` -> 86 / 90, the way torch itself reads them.

    Mirrors ``torch.cuda._extract_arch_version``: the number is
    ``major * 10 + minor``, and a trailing ``a``/``f`` on entries like
    ``sm_90a`` marks a variant rather than part of the number.
    """

    parts = entry.split("_", maxsplit=2)
    if len(parts) < 2:
        return None
    base = parts[1].removesuffix("a").removesuffix("f")
    return int(base) if base.isdigit() else None


def _capability_number(capability: tuple[int, int]) -> int:
    """Convert ``(major, minor)`` to the integer used by CUDA arch names."""

    return capability[0] * 10 + capability[1]


def _build_has_kernels_for(capability: tuple[int, int]) -> bool:
    """Can the installed torch actually run on a card of this capability?

    Only the *lower* bound of the build's arch list is treated as disqualifying,
    and membership is never required. Two traps drive that:

    - Requiring an exact ``sm_XY`` match would be plainly wrong. The cu128 list
      has no ``sm_89``, yet every RTX 40-series card runs on it, because a cubin
      is binary-compatible with later minor revisions of the same major version.
      Pushing those to CPU would be a far worse bug than the one this guard is
      here for.
    - Above the top of the list, a card may still work -- via a ``compute_XY``
      PTX entry the driver JITs forward, or that same in-major compatibility. So
      let torch try. A wrong fallback costs a silent 10x slowdown; a wrong
      attempt costs a loud error, which is the better way to be wrong.

    Below the minimum there is no such escape hatch (cubins are not backward
    compatible), which is exactly the GTX 10-series case. The bound therefore
    matches the condition torch warns about in ``torch.cuda._check_capability``.
    """

    numbers = [
        number
        for number in (_arch_number(entry) for entry in torch.cuda.get_arch_list())
        if number is not None
    ]
    if not numbers:
        # Nothing to compare against (ROCm, or a build that reports no list);
        # torch skips its own check here too, so assume the card is fine.
        return True
    return capability[0] * 10 + capability[1] >= min(numbers)


def cuda_unusable_reason() -> Optional[str]:
    """A reason CUDA cannot be used here, phrased for a user, or None if it can."""

    if not torch.cuda.is_available():
        return "it is unavailable"
    try:
        capability = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - driver-level failure
        return f"the CUDA device could not be queried ({exc})"
    if _build_has_kernels_for(capability):
        return None
    # Only now is the device name worth the call: this runs on every stage.
    major, minor = capability
    return (
        f"{torch.cuda.get_device_name()} (compute capability {major}.{minor}) is "
        f"older than anything this PyTorch build has kernels for; it needs "
        f"{SUPPORTED_GPU_HINT}"
    )


def cuda_usable() -> bool:
    """True when CUDA is present *and* this build has kernels for the card.

    The replacement for a bare ``torch.cuda.is_available()`` anywhere the answer
    decides whether real work goes to the GPU.
    """

    return cuda_unusable_reason() is None


def resolve_asr_device(requested_device: str, *, gpu_allowed: bool = True) -> str:
    """Resolve the CTranslate2 device independently from torch placement."""

    device = str(requested_device or "cuda")
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        return device
    if normalized != "cuda":
        raise ValueError(
            f"--device accepts 'cpu' or 'cuda', not {device!r}; select a "
            "specific GPU with the CUDA_VISIBLE_DEVICES environment variable."
        )
    if not gpu_allowed:
        return "cpu"
    reason = ct2_cuda_unusable_reason()
    if reason is None:
        return "cuda"
    current_reporter().warning(
        "cpu-fallback",
        f"CUDA requested for ASR but {reason}; falling back to CPU.",
        impact="速度会显著下降",
    )
    return "cpu"


def free_vram_gib() -> Optional[float]:
    """Return currently free usable CUDA memory in GiB."""

    if not cuda_usable():
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - driver-level failure
        return None
    return float(free_bytes) / float(1024**3)


def total_vram_gib() -> Optional[float]:
    """Return total usable CUDA memory in GiB."""

    if not cuda_usable():
        return None
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # pragma: no cover - driver-level failure
        return None
    return float(properties.total_memory) / float(1024**3)


def ct2_cuda_unusable_reason() -> Optional[str]:
    """Return why the installed CTranslate2 build cannot use CUDA."""

    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - dependency gate
        return f"CTranslate2 could not be imported ({exc})"
    try:
        if int(ctranslate2.get_cuda_device_count()) > 0:
            return _ct2_architecture_unusable_reason()
    except Exception as exc:  # pragma: no cover - driver-level failure
        return f"CTranslate2 could not be queried ({exc})"
    return (
        "this CTranslate2 build reports no CUDA device (a CPU-only wheel, or a "
        "driver it cannot see); see docs/manual/ct2-wheel.md"
    )


def _ct2_architecture_unusable_reason() -> Optional[str]:
    try:
        if not torch.cuda.is_available():
            return None
        capability = torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - driver-level failure
        return None
    if _capability_number(capability) >= _capability_number(
        CT2_MIN_COMPUTE_CAPABILITY
    ):
        return None
    major, minor = capability
    return (
        f"this CTranslate2 build has no kernels for compute capability "
        f"{major}.{minor}; it needs {SUPPORTED_GPU_HINT}"
    )


def _ct2_supports_mps() -> bool:
    """Whether the installed CTranslate2 wheel exposes the Metal backend.

    The repo ships the patched CT2 wheel used for the WT backend on Windows/CUDA,
    and the current macOS build here does not include Metal linkage. On that
    stack, ``device='mps'`` fails during model init with
    ``ValueError: unsupported device mps``. The check reads the native library's
    linkage rather than relying on a bare ``torch.backends.mps.is_available()``
    probe, so a future MPS-capable wheel is accepted without changing the call
    site.
    """

    override = os.environ.get("FINESUB_MPS_SUPPORTED", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True

    try:
        import ctranslate2
    except Exception:  # pragma: no cover - dependency gate
        return False

    package_root = Path(ctranslate2.__file__).resolve().parent
    dylib_dir = package_root / ".dylibs"
    if not dylib_dir.exists():
        return False
    for dylib in sorted(dylib_dir.glob("libctranslate2*.dylib")):
        try:
            result = subprocess.run(
                ["otool", "-L", str(dylib)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError:
            return False
        if "Metal.framework" in result.stdout or "MetalPerformanceShaders" in result.stdout:
            return True
    return False


def cuda_device_present() -> bool:
    """Whether the CUDA driver reports a device, regardless of torch kernels."""

    try:
        return int(torch.cuda.device_count()) > 0
    except Exception:  # pragma: no cover - driver-level failure
        return False


def mps_usable_reason() -> Optional[str]:
    """Whether this legacy CT2-facing placement can use MPS.

    ``mlx-refine`` and PyTorch auxiliary models do not call this probe: MLX
    manages its own device and the auxiliary stages query torch MPS directly.
    The patched CTranslate2 backend still has no Metal device.
    """

    if sys.platform == "darwin":
        return (
            "this project intentionally keeps macOS on the CPU fallback because the "
            "current CTranslate2/faster-whisper stack does not support Metal"
        )

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None:
        return "Metal Performance Shaders are unavailable on this build"
    try:
        if not mps.is_available():
            return "Metal Performance Shaders are unavailable"
        if not mps.is_built():
            return "the Metal backend was not built into this PyTorch install"
    except Exception as exc:  # pragma: no cover - platform-specific probe failure
        return f"MPS could not be queried ({exc})"
    if not _ct2_supports_mps():
        return (
            "the installed CTranslate2/faster-whisper stack does not expose the "
            "Metal backend; Apple Silicon MPS is unsupported here"
        )
    return None


def mps_usable() -> bool:
    """True when Apple's Metal Performance Shaders backend is available."""

    return mps_usable_reason() is None


def resolve_device(requested_device: str, *, context: str = "VAD-ASR") -> str:
    """Resolve legacy CT2-oriented placement with CUDA-first fallback.

    Apple MLX and auxiliary PyTorch placement are resolved by their backend
    owners; this function retains CPU fallback for the CT2 path.
    """

    device = str(requested_device or "cuda")
    normalized = device.strip().lower()

    if normalized == "cpu":
        return "cpu"
    if normalized in {"mps", "metal"}:
        reason = mps_usable_reason()
        if reason is None:
            return "mps"

        current_reporter().warning(
            "cpu-fallback",
            f"MPS requested for {context} but {reason}; falling back to CPU.",
            impact="速度会显著下降",
        )
        return "cpu"
    if normalized.startswith("mps"):
        raise ValueError(
            f"--device accepts 'cpu', 'cuda', or 'mps', not {device!r}; "
            "Apple Silicon uses the MPS backend directly."
        )
    if not normalized.startswith("cuda"):
        return device
    if normalized != "cuda":
        # Refused rather than passed through. Nothing here plumbs a device
        # index: the capability check below reads the *current* device, the
        # backend hands the string to CTranslate2 which rejects anything but
        # "cuda" (it takes the index as a separate device_index), and every CLI
        # advertises cpu/cuda. Accepting "cuda:1" would mean falling back to CPU
        # on the wrong card's verdict, then failing with an opaque ValueError
        # from inside faster-whisper.
        raise ValueError(
            f"--device accepts 'cpu', 'cuda', or 'mps', not {device!r}; select a "
            f"specific GPU with the CUDA_VISIBLE_DEVICES environment variable."
        )
    if sys.platform == "darwin":
        current_reporter().warning(
            "cpu-fallback",
            f"CUDA requested for {context} on macOS, but CTranslate2 does not support Metal; falling back to CPU. The mlx-refine backend manages Apple GPU execution separately.",
            impact="速度会显著下降",
        )
        return "cpu"
    reason = cuda_unusable_reason()
    if reason is None:
        return device
    current_reporter().warning(
        "cpu-fallback",
        f"CUDA requested for {context} but {reason}; falling back to CPU.",
        impact="速度会显著下降",
    )
    return "cpu"
