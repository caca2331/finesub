"""P7 -- is a stage compute-bound or launch-bound? Ask the card, not the clock.

`docs/plans/crispasr-followups.md` -> P7: before another round of separator graph
work, split host-side launch time from GPU execute time, because the same
trick is worth 9-13x in one regime and 0.3% in the other. The separator half
was already measured GPU-bound (`wall - sum(kernel) = +5.9 ms / 4%`); the open
half was the Whisper decoder.

A torch profiler cannot answer it: the decode runs inside CTranslate2's C++,
invisible to torch's tracer. What *is* visible is the card's own utilisation.
A stage that keeps the GPU near 100% is compute-bound; a stage that leaves it
idle between short kernels while Python and C++ prepare the next launch is
launch-bound, and that idle time is exactly what batching reclaims.

Run the sampler alongside a real stage:

    python -m tools.bench.probe_occupancy --seconds 90 --out tmp/bench/occ.csv

then summarise with `--summarise tmp/bench/occ.csv`.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from tools.bench import gpu


def _sample(seconds: float, interval: float, out: Path) -> int:
    if not gpu.available():
        print("FAIL: NVML unavailable; this probe measures GPU occupancy")
        return 2
    sampler = gpu.ClockSampler(interval_s=interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    with sampler:
        time.sleep(seconds)
    rows = ["t,sm_mhz,util_gpu,util_memory,power_w,temperature_c"]
    for s in sampler.trace.samples:
        rows.append(
            f"{s.t:.3f},{s.sm_mhz},{s.util_gpu},{s.util_memory},{s.power_w:.1f},{s.temperature_c}"
        )
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(sampler.trace.samples)} samples to {out}")
    return 0


def _summarise(path: Path, *, idle_below: int, busy_above: int) -> int:
    lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]
    if not lines:
        print("FAIL: no samples")
        return 2
    utils = [int(line.split(",")[2]) for line in lines]
    powers = [float(line.split(",")[4]) for line in lines]
    total = len(utils)

    print(f"samples            {total}")
    print(f"util median        {statistics.median(utils):.0f}%")
    print(f"util mean          {statistics.fmean(utils):.1f}%")
    print(f"power median       {statistics.median(powers):.0f} W")
    print(f"time at <={idle_below}%       {100 * sum(1 for u in utils if u <= idle_below) / total:.1f}%")
    print(f"time at >={busy_above}%       {100 * sum(1 for u in utils if u >= busy_above) / total:.1f}%")
    print()
    print("histogram (10% buckets):")
    for low in range(0, 100, 10):
        count = sum(1 for u in utils if low <= u < low + 10)
        bar = "#" * round(60 * count / total)
        print(f"  {low:3d}-{low + 9:3d}%  {count:5d}  {bar}")
    count = sum(1 for u in utils if u >= 100)
    print(f"     100%  {count:5d}  {'#' * round(60 * count / total)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("tmp/bench/occupancy.csv"))
    parser.add_argument("--summarise", type=Path, default=None)
    parser.add_argument("--idle-below", type=int, default=30)
    parser.add_argument("--busy-above", type=int, default=90)
    args = parser.parse_args()

    if args.summarise is not None:
        return _summarise(
            args.summarise, idle_below=args.idle_below, busy_above=args.busy_above
        )
    return _sample(args.seconds, args.interval, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
