"""Find *sentence-scale* collapses, not isolated zero-duration particles.

A one-mora particle landing on a single 80 ms cell is a resolution artefact and harmless.
A whole utterance packed into a fraction of its window is a real alignment failure: every
downstream consumer (segment_split's DP, premerge, SRT cue timing) would put the subtitle in
the wrong place. These are different phenomena and need different metrics.

Per segment:

- `span_ratio`  aligned span / window duration — how much of the window the words claim
- `stack_chars` longest run of consecutive characters packed into < `--stack-sec`
- `peak_density` max characters per second over a 1 s sliding window (natural JA speech
  runs roughly 6-12 char/s, so 25+ means text is stacked on top of itself)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .run_align import char_time_map


def segment_metrics(seg: dict, key: str = "words", stack_sec: float = 0.5) -> dict | None:
    chars = char_time_map(seg[key], "start", "end", "word")
    if not chars:
        return None
    window = seg["end"] - seg["start"]
    span = chars[-1][2] - chars[0][1]

    best_run = 0
    lo = 0
    for hi in range(len(chars)):
        while lo < hi and chars[hi][2] - chars[lo][1] > stack_sec:
            lo += 1
        best_run = max(best_run, hi - lo + 1)

    peak = 0.0
    for i, (_c, start, _e) in enumerate(chars):
        j = i
        while j < len(chars) and chars[j][1] < start + 1.0:
            j += 1
        peak = max(peak, float(j - i))

    return {
        "index": seg.get("index", -1),
        "chars": len(chars),
        "window": round(window, 2),
        "span_ratio": round(span / window, 3) if window > 0 else 0.0,
        "stack_chars": best_run,
        "peak_density": peak,
        "text": seg.get("text", "")[:44],
    }


def scan(path: str, key: str, stack_sec: float) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for seg in data["segments"]:
        m = segment_metrics(seg, key, stack_sec)
        if m and m["chars"] >= 6:  # too short to say anything about stacking
            rows.append(m)
    return rows


def report(label: str, rows: list[dict], span_min: float, density_max: float) -> None:
    if not rows:
        print(f"{label:44s} (no segments)")
        return
    flagged = [r for r in rows if r["span_ratio"] < span_min or r["peak_density"] > density_max]
    print(
        f"{label:44s} n={len(rows):4d} span_ratio p10={_p(rows, 'span_ratio', .1):.2f} "
        f"p50={_p(rows, 'span_ratio', .5):.2f} | peak_density p50={_p(rows, 'peak_density', .5):.0f} "
        f"p90={_p(rows, 'peak_density', .9):.0f} max={max(r['peak_density'] for r in rows):.0f} "
        f"| flagged {len(flagged)} ({100 * len(flagged) / len(rows):.0f}%)"
    )


def _p(rows: list[dict], key: str, q: float) -> float:
    v = sorted(r[key] for r in rows)
    return v[min(len(v) - 1, int(q * len(v)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", nargs="+", required=True, help="run_align.py outputs to scan")
    ap.add_argument("--baseline-too", action="store_true", help="also scan the Whisper-DTW timings")
    ap.add_argument("--stack-sec", type=float, default=0.5)
    ap.add_argument("--span-min", type=float, default=0.5, help="flag below this span ratio")
    ap.add_argument("--density-max", type=float, default=25.0, help="flag above this char/s")
    ap.add_argument("--worst", type=int, default=8)
    args = ap.parse_args()

    for path in args.align:
        name = Path(path).stem
        rows = scan(path, "words", args.stack_sec)
        report(f"{name} [qwen]", rows, args.span_min, args.density_max)
        if args.baseline_too:
            brows = scan(path, "baseline_words", args.stack_sec)
            report(f"{name} [whisper-dtw]", brows, args.span_min, args.density_max)
        worst = sorted(rows, key=lambda r: (r["span_ratio"], -r["peak_density"]))[: args.worst]
        for r in worst:
            print(
                f"    seg {r['index']:>4} chars={r['chars']:>3} window={r['window']:>5.2f}s "
                f"span_ratio={r['span_ratio']:.2f} peak={r['peak_density']:.0f}c/s  {r['text']}"
            )


if __name__ == "__main__":
    main()
