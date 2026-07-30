"""Whisper-timestamped model lifecycle and task-local pooling."""

from __future__ import annotations

import contextlib
import os
import sys
import threading

import torch


_WHISPER_MODEL_LOAD_LOCK = threading.Lock()


def _disable_whisper_sdpa_processwide() -> None:
    """Keep attention weights available when independent WT models run in threads."""

    from whisper.model import MultiHeadAttention

    # whisper-timestamped wraps every decode in disable_sdpa(), whose
    # save/restore of this class-global flag races across threads. WT always
    # needs the non-SDPA path to collect attention weights, so False is the
    # correct process-wide steady state as well as the thread-safe one.
    MultiHeadAttention.use_sdpa = False


def read_shared_checkpoint(model_name: str, *, device: str):
    """Read an official model's checkpoint once, for the whole pool to build from.

    ``whisper.load_model`` re-reads and re-SHA256s the 1.6GB file on every call,
    so a second instance pays the whole cost again. Returns ``None`` -- meaning
    "use the stock loader" -- for CPU runs and for anything that is not an
    official model name, so local ``.pt`` paths and HuggingFace identifiers keep
    whisper-timestamped's own resolution.
    """

    if not str(device).strip().lower().startswith("cuda"):
        return None
    try:
        from whisper import _ALIGNMENT_HEADS, _MODELS, _download
    except ImportError:
        return None
    if model_name not in _MODELS:
        return None

    # Mirrors whisper.load_model's own default; whisper-timestamped passes no
    # download_root for official names.
    default_cache = os.path.join(os.path.expanduser("~"), ".cache")
    root = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache), "whisper")
    try:
        path = _download(_MODELS[model_name], root, False)
        with open(path, "rb") as handle:
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(
            f"Warning: could not pre-read the {model_name} checkpoint ({exc}); "
            "falling back to the stock whisper loader.",
            file=sys.stderr,
        )
        return None
    return checkpoint, _ALIGNMENT_HEADS[model_name]


def _build_half_precision_model(checkpoint, alignment_heads, *, device):
    """Build one FP16 model without ever materializing the FP32 one.

    The published checkpoints are already FP16, but ``Whisper(dims)`` builds in
    the default dtype, so the stock path holds an FP32 model (3.2GB) next to the
    checkpoint (1.6GB) and then allocates a second FP16 copy in ``.half()``.
    Constructing straight into FP16 removes both extra copies. Mutating the
    global default dtype is safe here: every build runs under
    ``_WHISPER_MODEL_LOAD_LOCK``, and this path is CUDA-only, where the GPU
    stage gate already keeps the separator family from running alongside.
    """

    from whisper.model import ModelDimensions, Whisper

    dims = ModelDimensions(**checkpoint["dims"])
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = Whisper(dims)
    finally:
        torch.set_default_dtype(previous_dtype)

    # strict=True on purpose: a persistent buffer missing from the checkpoint
    # must fail loudly rather than leave the model holding FP16-computed values.
    model.load_state_dict(checkpoint["model_state_dict"])
    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)
    # Whisper intentionally computes LayerNorm in FP32.
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()
    return model.to(device)


def load_whisper_model_serialized(
    whisper,
    model_name: str,
    *,
    device: str,
    checkpoint=None,
):
    """Load one Whisper model at a time, without serializing inference."""

    with _WHISPER_MODEL_LOAD_LOCK:
        _disable_whisper_sdpa_processwide()
        if not str(device).strip().lower().startswith("cuda"):
            return whisper.load_model(model_name, device=device)
        if checkpoint is not None:
            return _build_half_precision_model(*checkpoint, device=device)

        # Fallback for non-official names: build and cast on the CPU so the FP32
        # model and its FP16 replacement never coexist on the GPU.
        model = whisper.load_model(model_name, device="cpu").half()
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()
        return model.to(device)


class WtModelPool:
    """Task-internal pool of independent WT models (docs/wt-parallelism.md).

    One model per concurrent shard, each touched by exactly one thread. Models
    are never shared: whisper-timestamped registers forward hooks during a call
    and carries process-level configuration, so two threads in one model would
    corrupt each other. Loading stays serialized by the existing lock; only
    inference overlaps.

    ``warm`` builds every instance up front from one shared checkpoint;
    ``lease`` still builds on demand for any model that warm-up did not cover.
    """

    def __init__(self, whisper, model_name: str, *, device: str, size: int):
        self._whisper = whisper
        self._model_name = model_name
        self._device = device
        self._size = max(1, int(size))
        self._idle: list = []
        self._loaded = 0
        self._condition = threading.Condition()

    @contextlib.contextmanager
    def lease(self):
        model = self._acquire()
        try:
            yield model
        finally:
            self._release(model)

    def _acquire(self, checkpoint=None):
        with self._condition:
            while True:
                if self._idle:
                    return self._idle.pop()
                if self._loaded < self._size:
                    self._loaded += 1
                    break
                self._condition.wait()
        try:
            return load_whisper_model_serialized(
                self._whisper,
                self._model_name,
                device=self._device,
                checkpoint=checkpoint,
            )
        except BaseException:
            with self._condition:
                self._loaded -= 1
                self._condition.notify()
            raise

    def _release(self, model) -> None:
        with self._condition:
            self._idle.append(model)
            self._condition.notify()

    def warm(self) -> None:
        """Build every instance before timed shard work begins.

        The pool is sized from the shard plan, so all ``size`` models get used
        anyway; loading lazily only moved one build onto the second shard's
        critical path, which docs/wt-parallelism.md measures as a real loss
        ("模型加载错峰"). Building them together also lets a single checkpoint
        serve them all -- it is dropped as soon as the last model is built, so
        it costs peak RAM during warm-up rather than resident RAM afterwards.
        """

        checkpoint = read_shared_checkpoint(self._model_name, device=self._device)
        models = []
        try:
            for _ in range(self._size):
                models.append(self._acquire(checkpoint))
        finally:
            checkpoint = None
            for model in models:
                self._release(model)

    @property
    def loaded(self) -> int:
        return self._loaded

    def close(self) -> None:
        with self._condition:
            models, self._idle, self._loaded = self._idle, [], 0
        models.clear()
