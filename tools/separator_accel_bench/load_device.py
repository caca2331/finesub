"""Is load_model's cost device-neutral? Build the same model on CPU and on CUDA.

Whatever is device-neutral could run before the GPU stage gate is taken; what
is not, cannot. Construction only -- nothing is executed on CPU here, so this
does not go near the CPU-decode deadlock recorded for the ASR stack.
"""

from __future__ import annotations

import logging
import tempfile
import time

import torch

from finesub.paths import resolve_separator_model_dir

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def build(force_cpu: bool) -> float:
    from audio_separator.separator import Separator

    original = torch.cuda.is_available
    if force_cpu:
        torch.cuda.is_available = lambda: False
    try:
        separator = Separator(
            output_dir=tempfile.mkdtemp(prefix="loadsplit_"),
            output_format="flac",
            output_single_stem="Vocals",
            model_file_dir=str(resolve_separator_model_dir()),
            mdxc_params={"batch_size": 1},
            log_level=logging.ERROR,
        )
        started = time.perf_counter()
        separator.load_model(MODEL)
        if not force_cpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        device = next(separator.model_instance.model_run.parameters()).device
        print(f"  load_model on {str(device):>5}: {elapsed:6.2f}s")
        del separator
        return elapsed
    finally:
        torch.cuda.is_available = original


print("BS-Roformer load_model, same checkpoint, two devices")
cpu = build(force_cpu=True)
cuda = build(force_cpu=False)
print(f"\n  device-neutral share ≈ {100 * min(cpu, cuda) / cuda:.0f}% "
      f"（CPU {cpu:.2f}s vs CUDA {cuda:.2f}s）")
