"""ASR engine selection and the small pool contract used by the pipeline."""

from __future__ import annotations

import platform
from typing import Any, Protocol

from ..runtime import hf_weights

AUTO = "auto"
FW_REFINE = "fw-refine"
MLX_REFINE = "mlx-refine"
BACKEND_CHOICES = (AUTO, FW_REFINE, MLX_REFINE)
MLX_DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


class AsrModel(Protocol):
    supports_beam: bool

    def transcribe_wt(self, audio, **options: Any) -> dict[str, object]: ...


class AsrModelPool(Protocol):
    def warm(self) -> None: ...

    def lease(self): ...

    def close(self) -> None: ...


def resolve_backend(
    requested: str | None,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Resolve ``auto`` without importing either inference runtime."""

    value = str(requested or AUTO).strip().lower()
    if value not in BACKEND_CHOICES:
        choices = ", ".join(BACKEND_CHOICES)
        raise ValueError(
            f"unknown ASR backend {requested!r}; expected one of: {choices}"
        )
    if value != AUTO:
        return value
    host_system = (system or platform.system()).strip().lower()
    host_machine = (machine or platform.machine()).strip().lower()
    if host_system == "darwin" and host_machine in {"arm64", "aarch64"}:
        return MLX_REFINE
    return FW_REFINE


def resolve_model_name(model_name: str, backend: str, *, default_model: str) -> str:
    """Map the public logical default to the engine-specific checkpoint."""

    value = str(model_name)
    if backend != MLX_REFINE:
        return value
    if value == default_model:
        return MLX_DEFAULT_MODEL
    if value.endswith("-ct2") or "faster-whisper" in value.lower():
        raise ValueError(
            f"model {value!r} is a CTranslate2 checkpoint and cannot be loaded by "
            "the mlx-refine backend; provide an MLX Whisper repository or local path"
        )
    return value


def build_pool(
    backend: str,
    model_name: str,
    *,
    device: str,
    size: int,
    refine_sec: float,
    load: hf_weights.HfLoad = hf_weights.UNMANAGED,
) -> AsrModelPool:
    if backend == MLX_REFINE:
        from .mlx_refine_backend import MlxRefineModelPool

        return MlxRefineModelPool(
            model_name,
            size=size,
            refine_sec=refine_sec,
            load=load,
        )
    if backend == FW_REFINE:
        from .fw_refine_backend import FwRefineModelPool

        return FwRefineModelPool(
            model_name,
            device=device,
            size=size,
            refine_sec=refine_sec,
            load=load,
        )
    raise ValueError(f"resolved ASR backend required, got {backend!r}")


def batch_decoder(backend: str):
    """Return the optional engine batch function used by DecodePrefetch."""

    if backend == FW_REFINE:
        from .fw_refine_backend import transcribe_batch

        return transcribe_batch
    return None
