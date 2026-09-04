"""Where the compiled forward's time actually goes, and against which ceiling.

E13 showed the separation stage's remaining cost is overwhelmingly the forward
(91s of 115.6s on the 2015s material). Before spending anything on a different
model, this answers two questions the queue was guessing at:

1. **What is the forward made of?** Attention, projection/FFN GEMMs, and
   everything else, as a share of both FLOPs and measured kernel time. This
   prices every "replace the attention" proposal -- Windowed Sink Attention's
   44.5x is a reduction in *attention* FLOPs, which is worth nothing here if
   attention is a small slice.

2. **Is it FLOP-bound or not?** Achieved TFLOP/s against the card's dense FP16
   tensor ceiling says whether there is headroom a better kernel could take.

FLOPs come from ``FlopCounterMode`` on the eager model -- exact, not derived by
hand, and identical maths to the compiled path. Time comes from
``torch.profiler`` kernel self-time, taken on whichever backend is asked for, so
the AOTI arm is measured as it actually ships.

    python -m tools.separator_accel_bench.roofline --backend aoti
    python -m tools.separator_accel_bench.roofline --backend eager --iters 6
    python -m tools.separator_accel_bench.roofline --flops-only     # no GPU work

Conclusions land in ``docs/separator-optimization.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.speech.preprocessing.separator import accel, separation  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "aoti", "jit"), default="aoti")
    parser.add_argument("--iters", type=int, default=6, help="profiled forwards")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--flops-only", action="store_true",
                        help="analytic pass only; still loads the model, no profiling")
    parser.add_argument("--result", type=Path, default=None)
    return parser.parse_args()


def _build_model(backend: str) -> tuple[Any, Any, str]:
    """A production-configured model, still eager. Accel is installed later.

    Order matters: AOTI runners do not dispatch through ATen, so
    ``FlopCounterMode`` sees an empty graph once they are in place. Count first,
    install second -- the maths is identical either way.
    """

    tmpdir = tempfile.TemporaryDirectory(prefix="roofline_")
    separator = separation._build_separator(tmpdir.name, "flac", 1)
    model_instance = separator.model_instance
    separation._warm_up_shared_roformer(model_instance, use_amp=True)
    return model_instance, tmpdir, backend


def _install(model_instance: Any, backend: str) -> str:
    if backend == "eager":
        return "eager"
    paths = accel.resolve_accel_paths(separation.MODEL_NAME)
    result = accel.apply_acceleration(model_instance, backend, paths)
    if result.effective != backend:
        print(f"! requested {backend}, got {result.effective}: {result.fallback_reason}")
    if result.effective == "jit":
        separation._warm_up_shared_roformer(model_instance, use_amp=True)
    return result.effective


def _chunk(model_instance: Any) -> torch.Tensor:
    config = model_instance.model_data_cfgdict
    hop = getattr(config.model, "stft_hop_length", None) or config.audio.hop_length
    frames = int(hop) * (int(config.inference.dim_t) - 1)
    channels = int(getattr(model_instance.model_run, "audio_channels", 2))
    device = next(model_instance.model_run.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(0)
    audio = torch.randn(1, channels, frames, generator=generator) * 0.05
    return audio.to(device)


#: Which module a flop belongs to, from its position in the module tree. The
#: model is `layers[i]` = (time transformer, freq transformer), each a ModuleList
#: of Attention/FeedForward; band_split and mask_estimators sit outside.
def _bucket(module_path: str) -> str:
    lowered = module_path.lower()
    if "band_split" in lowered:
        return "band_split"
    if "mask_estimator" in lowered:
        return "mask_estimator"
    if "layers" not in lowered:
        return "other"
    return "transformer"


def _attention_share(counter_counts: dict[str, dict[Any, int]]) -> tuple[int, int]:
    """(attention flops, total flops), read off the counter's ``Global`` row.

    Attention is whatever the dispatcher saw as an SDPA op; every other flop --
    the QKV/out projections and the feed-forwards -- lands in the remainder.
    """

    attention = total = 0
    for op, value in counter_counts.get("Global", {}).items():
        total += value
        name = getattr(op, "__name__", str(op))
        if "scaled_dot_product" in name or "sdpa" in name:
            attention += value
    return attention, total


def _profile(model: Any, audio: torch.Tensor, *, iters: int, warmup: int) -> dict[str, float]:
    from torch.profiler import ProfilerActivity, profile

    wall_ms = 0.0
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
        for _ in range(warmup):
            model(audio)
        torch.cuda.synchronize()
        # Wall time first, un-instrumented: the profiler perturbs it, and the gap
        # between wall and summed kernel time is the launch/idle overhead that
        # decides whether a faster kernel would even show up.
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            model(audio)
        end.record()
        torch.cuda.synchronize()
        wall_ms = start.elapsed_time(end) / iters
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(audio)
            torch.cuda.synchronize()

    # Device rows only. A CPU op that launched work carries the same device time
    # as the kernel it launched, so counting both doubles every launcher.
    kernels: dict[str, float] = defaultdict(float)
    skipped: dict[str, float] = defaultdict(float)
    for event in prof.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        self_us = getattr(event, "self_device_time_total", 0.0)
        if self_us <= 0:
            continue
        per_forward_ms = self_us / 1000.0 / iters
        if _is_graph_launcher(event.key):
            skipped[event.key] += per_forward_ms
            continue
        kernels[event.key] += per_forward_ms
    kernels["__wall__"] = wall_ms
    kernels["__graph_launch__"] = sum(skipped.values())
    return dict(kernels)


#: Rows that are a *container* for kernels rather than a kernel. When Inductor
#: replays a compiled region as a CUDA graph, the profiler emits one device row
#: for the graph launch carrying the whole region's device time **and** the
#: individual kernels inside it. Counting both double-counts everything in the
#: graph: on torch 2.9 that made the summed kernel time 972ms against a 533ms
#: forward wall, i.e. a nonsensical -82% "idle" gap. These are recorded
#: separately rather than dropped silently, so the split stays auditable.
_GRAPH_LAUNCH_MARKERS = ("CompiledFxGraph", "cudaGraphLaunch", "CUDAGraph")


def _is_graph_launcher(name: str) -> bool:
    return any(marker in name for marker in _GRAPH_LAUNCH_MARKERS)


#: Kernel-name fragments, most specific first. AOTI fuses into `triton_*`
#: kernels whose names spell out the ops they fused, so the same table reads
#: both arms; anything unmatched is reported verbatim so a miss is visible.
_CATEGORIES = (
    # `triton_tem_` is a GEMM *template*, not an elementwise kernel: with
    # max_autotune on, Inductor lowers the matmuls to these and fuses the
    # pointwise epilogue into them, so cuBLAS all but disappears from the
    # profile. Lumping them with `triton_poi_`/`triton_per_` would report the
    # matmul share as ~0 and make any precision argument nonsense. Both must
    # come before the name-fragment rules below, which would otherwise claim
    # `triton_tem_fused_addmm_gelu_view_18` for "gemm" and
    # `triton_poi_fused_mm_mul_...` for it too.
    ("gemm (triton template)", ("triton_tem_",)),
    ("pointwise / reduction", ("triton_poi_", "triton_per_", "triton_red_",
                               "triton_")),
    ("attention", ("cudnn_attention", "fmha", "flash", "mha_", "attention_kernel",
                   "efficient_attention", "scaled_dot_product", "sdpa")),
    ("gemm (cublas)", ("gemm", "cutlass", "cublas", "sm90", "sm100", "sm120",
                       "ampere", "nvjet", "wgrad", "implicit", "gemv", "matmul",
                       "addmm", "_mm_")),
    ("fft", ("fft", "stft")),
    ("copy/layout", ("copy", "transpose", "permute", "contiguous", "cat_", "catarray",
                     "vectorized_elementwise_kernel", "unrolled_elementwise",
                     "elementwise_kernel", "memcpy", "memset")),
)


def _categorize(kernels: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for name, ms in kernels.items():
        lowered = name.lower()
        for label, fragments in _CATEGORIES:
            if any(fragment in lowered for fragment in fragments):
                totals[label] += ms
                break
        else:
            totals[f"other: {name[:48]}"] += ms
    return dict(totals)


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 2

    requested = "eager" if args.flops_only else args.backend
    model_instance, tmpdir, _ = _build_model(requested)
    model = model_instance.model_run
    audio = _chunk(model_instance)
    device_name = torch.cuda.get_device_name(0)
    print(f"{device_name}  torch {torch.__version__}  backend={requested}")
    print(f"chunk {tuple(audio.shape)}  "
          f"{audio.shape[-1] / model_instance.sample_rate:.2f}s of audio")

    report: dict[str, Any] = {
        "device": device_name,
        "torch": torch.__version__,
        "backend": requested,
        "chunk_frames": int(audio.shape[-1]),
    }

    # FLOPs are counted on the eager math; the compiled arms run the same graph.
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False, depth=None)
    eager_model = model
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
        with counter:
            eager_model(audio)
    attention_flops, total_flops = _attention_share(counter.flop_counts)
    buckets: dict[str, int] = defaultdict(int)
    for module_path, ops in counter.flop_counts.items():
        if module_path == "Global":
            continue
        buckets[_bucket(module_path)] += sum(ops.values())

    print(f"\nFLOPs per chunk: {total_flops / 1e12:.3f} TFLOP")
    if total_flops:
        print(f"  attention (SDPA)      {attention_flops / 1e12:8.3f} TFLOP  "
              f"{attention_flops / total_flops * 100:5.1f}%")
        print(f"  everything else       {(total_flops - attention_flops) / 1e12:8.3f} TFLOP  "
              f"{(total_flops - attention_flops) / total_flops * 100:5.1f}%")
    report["flops_total"] = total_flops
    report["flops_attention"] = attention_flops
    report["flops_by_bucket"] = dict(buckets)

    if not args.flops_only:
        effective = _install(model_instance, requested)
        report["backend"] = effective
        model = model_instance.model_run
        print(f"\nbackend in place: {effective}")
        kernels = _profile(model, audio, iters=args.iters, warmup=args.warmup)
        wall_ms = kernels.pop("__wall__")
        graph_ms = kernels.pop("__graph_launch__", 0.0)
        per_forward_ms = sum(kernels.values())
        print(f"\nforward wall {wall_ms:.1f}ms   summed kernel time {per_forward_ms:.1f}ms"
              f"   gap {wall_ms - per_forward_ms:+.1f}ms "
              f"({(wall_ms - per_forward_ms) / wall_ms * 100:.0f}% idle/launch)")
        if graph_ms:
            print(f"  (excluded {graph_ms:.1f}ms of CUDA-graph launcher rows: they "
                  f"carry their own children's time -- see _GRAPH_LAUNCH_MARKERS)")
        for label, ms in sorted(_categorize(kernels).items(), key=lambda kv: -kv[1]):
            print(f"  {label:<24} {ms:8.2f}ms  {ms / per_forward_ms * 100:5.1f}%")
        print("\ntop kernels by self time:")
        for name, ms in sorted(kernels.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {ms:7.2f}ms  {name[:78]}")
        if wall_ms > 0:
            print(f"\nachieved {total_flops / (wall_ms / 1000) / 1e12:.1f} TFLOP/s "
                  f"on wall, {total_flops / (per_forward_ms / 1000) / 1e12:.1f} "
                  f"TFLOP/s on kernel time")
        report["wall_ms_per_forward"] = wall_ms
        report["kernel_ms_per_forward"] = per_forward_ms
        report["kernels"] = kernels

    if args.result:
        args.result.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n-> {args.result}")
    tmpdir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
