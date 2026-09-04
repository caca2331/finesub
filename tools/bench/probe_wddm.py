"""A3.1 -- does this card hold P0 through bursty, chunk-shaped GPU work?

`docs/plans/crispasr-followups.md` -> A3 makes this the *first* measurement, because
its answer either pollutes every other baseline or is itself a production loss.
The external number that motivated it -- 1400 ms cold vs 185 ms warm per chunk,
8x -- came from a mobile GPU, where the driver's power management is far more
aggressive. On a desktop card the effect is expected to be an order of
magnitude smaller or absent. Either answer is useful; an unmeasured assumption
is not.

The workload deliberately imitates the *shape* the driver penalises rather than
any particular model: short kernel groups separated by host-side gaps, which is
what chunked ASR decoding and chunked separator inference look like from the
scheduler's side. A sustained variant runs the identical kernels back to back,
so the difference between the two isolates the gap, not the arithmetic.

Run this in a fresh process -- "cold" is a property of the process and the
card, and it cannot be recovered once something else has warmed either.

    python -m tools.bench.probe_wddm --chunks 200 --gap-ms 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from finesub.speech.runtime.device import cuda_usable

from tools.bench import gpu


def _chunk_workload(device: torch.device, size: int, inner: int) -> callable:
    """Build a closure whose one call is one 'chunk' of decode-shaped work."""

    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)

    def run() -> None:
        acc = a
        for _ in range(inner):
            acc = torch.matmul(acc, b)
            acc = torch.nn.functional.gelu(acc)
        torch.cuda.synchronize()

    return run


def _burst_series(work, *, chunks: int, gap_ms: float) -> list[float]:
    times: list[float] = []
    gap_s = gap_ms / 1000.0
    for _ in range(chunks):
        started = time.perf_counter()
        work()
        times.append((time.perf_counter() - started) * 1000.0)
        if gap_s:
            time.sleep(gap_s)
    return times


def _window(values: list[float], start: int, count: int) -> float:
    piece = values[start : start + count]
    return statistics.median(piece) if piece else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--gap-ms", type=float, default=40.0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--inner", type=int, default=24)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # `cuda_usable`, not `torch.cuda.is_available()`: the bare call reports a
    # card too old for the installed torch as usable, and every "can this
    # machine use the GPU" decision in this project goes through one place
    # (CLAUDE.md -> GPU). A probe is not exempt -- it would just mint a
    # confusing crash instead of a clear refusal.
    if not cuda_usable():
        print("FAIL: no usable CUDA device; this probe measures a GPU power state")
        return 2

    device = torch.device("cuda")
    report: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "chunks": args.chunks,
        "gap_ms": args.gap_ms,
        "matmul": {"size": args.size, "inner": args.inner, "dtype": "float16"},
    }

    print("sampling idle baseline (2 s) -- a busy desktop is context, not an excuse ...")
    report["idle_baseline"] = gpu.idle_baseline(2.0)
    print(json.dumps(report["idle_baseline"], indent=2, default=str))

    # Allocation and the first kernel launch cost far more than a chunk; they
    # are setup, not the thing being measured, so they happen before the trace.
    work = _chunk_workload(device, args.size, args.inner)
    work()

    print(f"\nbursty: {args.chunks} chunks with a {args.gap_ms:.0f} ms host gap ...")
    sampler = gpu.ClockSampler(interval_s=0.05)
    with sampler:
        bursty = _burst_series(work, chunks=args.chunks, gap_ms=args.gap_ms)
    report["bursty"] = {
        "per_chunk_ms": bursty,
        "first5_median_ms": _window(bursty, 0, 5),
        "first20_median_ms": _window(bursty, 0, 20),
        "last20_median_ms": _window(bursty, max(0, len(bursty) - 20), 20),
        "overall_median_ms": statistics.median(bursty),
        "clocks": sampler.trace.summary(),
    }

    # Let the card fall back toward whatever its resting state is, so the
    # sustained run is not simply inheriting the bursty run's warmth.
    print("cooling 15 s ...")
    cooldown = gpu.ClockSampler(interval_s=0.25)
    with cooldown:
        time.sleep(15.0)
    report["cooldown_clocks"] = cooldown.trace.summary()

    print(f"sustained: {args.chunks} identical chunks, no gap ...")
    sampler2 = gpu.ClockSampler(interval_s=0.05)
    with sampler2:
        sustained = _burst_series(work, chunks=args.chunks, gap_ms=0.0)
    report["sustained"] = {
        "per_chunk_ms": sustained,
        "first5_median_ms": _window(sustained, 0, 5),
        "first20_median_ms": _window(sustained, 0, 20),
        "last20_median_ms": _window(sustained, max(0, len(sustained) - 20), 20),
        "overall_median_ms": statistics.median(sustained),
        "clocks": sampler2.trace.summary(),
    }

    burst_warmup = report["bursty"]["first5_median_ms"] / report["bursty"]["last20_median_ms"]
    burst_vs_sustained = (
        report["bursty"]["last20_median_ms"] / report["sustained"]["last20_median_ms"]
    )
    report["verdict"] = {
        "bursty_warmup_ratio": burst_warmup,
        "bursty_over_sustained_steady": burst_vs_sustained,
    }

    print("\n" + "=" * 78)
    print("A3.1 VERDICT")
    print("=" * 78)
    for name in ("bursty", "sustained"):
        block = report[name]
        print(
            f"  {name:<10} first5 {block['first5_median_ms']:7.2f} ms | "
            f"first20 {block['first20_median_ms']:7.2f} ms | "
            f"last20 {block['last20_median_ms']:7.2f} ms"
        )
        clocks = block["clocks"]
        print(
            f"             SM {clocks.get('sm_mhz_min')}-{clocks.get('sm_mhz_max')} MHz "
            f"(median {clocks.get('sm_mhz_median')}), "
            f"mem {clocks.get('memory_mhz_min')} MHz, "
            f"sagging {clocks.get('fraction_clock_sagging'):.3f}, "
            f"power {clocks.get('power_w_median')} W"
        )
    print(f"\n  warm-up penalty (bursty first5 / last20) : {burst_warmup:.3f}x")
    print(f"  steady bursty / steady sustained         : {burst_vs_sustained:.3f}x")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
