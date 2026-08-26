"""How much of load_model is device-neutral, and how much is the GPU transfer?

Only the device-neutral part could in principle happen before the GPU stage
gate is taken. Compiled artefacts cannot -- they are built for one device -- so
this checkpoint-and-construct cost is the whole of the opportunity.
"""

from __future__ import annotations

import time

import torch

from finesub.paths import resolve_separator_model_dir

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
path = resolve_separator_model_dir() / MODEL
print(f"checkpoint: {path.stat().st_size / 2**20:.0f} MiB")

# Cold-ish disk read is what a first run pays; the OS cache makes a repeat
# cheap, so report both.
for label in ("first", "repeat"):
    started = time.perf_counter()
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    read = time.perf_counter() - started
    tensors = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    n = sum(1 for v in tensors.values() if torch.is_tensor(v))
    total_bytes = sum(v.numel() * v.element_size() for v in tensors.values() if torch.is_tensor(v))
    print(f"  torch.load -> CPU ({label}): {read:.2f}s  "
          f"{n} tensors, {total_bytes / 2**20:.0f} MiB")
    if label == "first":
        del state, tensors

torch.cuda.init()
torch.cuda.synchronize()
started = time.perf_counter()
moved = {k: (v.to("cuda", non_blocking=False) if torch.is_tensor(v) else v)
         for k, v in tensors.items()}
torch.cuda.synchronize()
print(f"  .to('cuda') of the same tensors: {time.perf_counter() - started:.2f}s")

started = time.perf_counter()
pinned = {k: (v.pin_memory() if torch.is_tensor(v) else v) for k, v in tensors.items()}
print(f"  pin_memory() on CPU side:        {time.perf_counter() - started:.2f}s")
torch.cuda.synchronize()
started = time.perf_counter()
for k, v in pinned.items():
    if torch.is_tensor(v):
        v.to("cuda", non_blocking=True)
torch.cuda.synchronize()
print(f"  pinned .to('cuda', non_blocking): {time.perf_counter() - started:.2f}s")
del moved
