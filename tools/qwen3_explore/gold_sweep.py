"""Sweep splitter parameters against the **human gold set** instead of the mechanical metric.

Why this exists: `bench.py --cv` selects on `acceptability.py`, and the batch-1 gold set showed
that metric disagreeing in *sign* with human judgement on the one comparison that is clean
(whisper-prod vs whisper-split: mechanical says the tuned splitter is far better, gold says it is
slightly worse on both axes). So parameter selection needs a second opinion from the ruler that
can see semantics.

**This is a falsification tool, not a tuner.** Fitting on the gold set overfits immediately, so it
is used the other way round: fit on the evaluation set *deliberately* to get an **optimistic upper
bound** on the parameter family. If the best config found this way still cannot reach the
whisper-prod baseline, no amount of honest tuning will either, and the family is refuted without
needing more gold data. A config that *does* win here has proved nothing yet — re-validate it on
windows the fit never saw, with `--windows` (that is how `asr_boundary_bonus` was settled: fit on
batch 1, confirmed on batches 2+3, FINDINGS §3.2).

Prefer `--by-index` for anything comparing arms that share a word stream: time anchoring hands the
arm that supplied the substrate a free hit and is sensitive to cue re-timing (gold doc §2.1).

    python -m tools.qwen3_explore.gold_sweep --by-index            # baseline vs current Params
    python -m tools.qwen3_explore.gold_sweep --by-index --grid 'base=0.05,0.3;fragment_penalty=0.5,3'
    python -m tools.qwen3_explore.gold_sweep --by-index --windows yui-660,kaguya60,yingtao
    python -m tools.qwen3_explore.gold_sweep --arm qwen --grid ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.segmentation_gold.gold import (  # noqa: E402
    _cuts, cuts_to_indices, evaluate, evaluate_by_index, load_golds, load_words,
)

from .common import baseline_aligned, qwen_raw, vad_json  # noqa: E402
from .qwen_split import Params, segment  # noqa: E402

GOLD_DIR = Path(__file__).resolve().parents[1] / "segmentation_gold" / "labels"
TAU = 0.30


def _cues(raw: dict) -> list[dict]:
    return [{"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text") or ""}
            for s in raw["segments"] if str(s.get("text") or "").strip()]


def run_arm(golds: list[dict], arm: str, p: Params, cache: dict, by_index: bool = False) -> dict:
    """Re-segment each clip in-process and score it against its own gold windows, then pool.

    Scoring is strictly per clip: clip timelines all start at 0, so pooling cuts across clips
    before `evaluate` would score one clip's cuts against another clip's window wherever the
    windows overlap in absolute time (they do). Under `by_index` the anchor is the substrate word
    junction instead of time, which removes the substrate/re-timing advantage documented in
    docs/segmentation-gold.md §2.1 — at the cost of needing `k` on every gold item.
    """
    total: dict = {}
    for clip in sorted({g["clip"] for g in golds}):
        mine = [g for g in golds if g["clip"] == clip]
        key = (clip, arm, p)
        if key not in cache:
            vad, raw = cache[("src", clip, arm)]
            non_speech = [(float(a), float(b)) for a, b in vad["non_speech"]]
            if arm == "whisper-prod":
                out = raw
            else:
                out = segment(json.loads(json.dumps(raw)), non_speech, float(vad["duration"]), p,
                              words_carry_punct=arm.startswith("whisper"))
            cues = _cues(out)
            cache[key] = (_cuts(cues), cuts_to_indices(cues, cache[("words", clip)]))
        cuts, idx = cache[key]
        e = evaluate_by_index(mine, idx) if by_index else evaluate(mine, cuts, TAU)
        for k in ("must", "must_hit", "declared", "presumed", "in_window", "skipped"):
            total[k] = total.get(k, 0) + e[k]
    judged = total["in_window"] - total["skipped"]
    total["recall"] = total["must_hit"] / total["must"] if total["must"] else float("nan")
    total["viol_per100"] = 100 * (total["declared"] + total["presumed"]) / judged if judged else 0.0
    return total


def loss(e: dict, w_miss: float) -> float:
    """Missed `must` in percentage points at weight `w_miss`, violations per 100 cues at 1.

    Stated judgement, not a measurement — same discipline as `bench.objective`. A missed `must` is
    an under-split: unrecoverable downstream. A violation is an over-split, which `premerge`
    profile 3 can rejoin *if* the gap is under the merge threshold. Hence w_miss > 1, and
    `--weight-scan` shows how much of the answer is this number.
    """
    return w_miss * 100 * (1 - e["recall"]) + e["viol_per100"]


def fmt(label: str, e: dict, w_miss: float) -> str:
    return (f"{label:<26s} {100*e['recall']:6.1f}% ({e['must_hit']:2d}/{e['must']:2d}) "
            f"{e['in_window']:5d} {e['declared']:6d} {e['presumed']:6d} {e['viol_per100']:9.1f} "
            f"{loss(e, w_miss):8.1f}")


HEAD = (f"{'配置':<26s} {'必切召回':>16s} {'刀数':>5s} {'人标':>6s} {'推定':>6s} "
        f"{'违反/百条':>9s} {'损失':>8s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="whisper-split", choices=("whisper-split", "qwen"))
    ap.add_argument("--grid", help="FIELD=v1,v2[;FIELD=...]")
    ap.add_argument("--weight-miss", type=float, default=2.0)
    ap.add_argument("--by-index", action="store_true",
                    help="按底本词序号锚定（无容差，不受重定时/底本优势影响）")
    ap.add_argument("--windows", help="只用文件名含该子串的 gold 窗口，逗号分隔（用于拟合/留出分组）")
    args = ap.parse_args()

    golds = load_golds(GOLD_DIR)
    if args.windows:
        pats = args.windows.split(",")
        golds = [g for g in golds
                 if any(s in f"{g['clip']}-{int(g['window'][0])}-{int(g['window'][1])}" for s in pats)]
        if not golds:
            raise SystemExit(f"--windows {args.windows} 没匹配到任何窗口")
    clips = sorted({g["clip"] for g in golds})
    anchor = "词序号（精确）" if args.by_index else f"时间 τ={TAU}"
    print(f"gold 窗口 {len(golds)} 个 / clip {clips} / 锚定 {anchor} / 漏切权重 {args.weight_miss}\n")

    cache: dict = {}
    for clip in clips:
        vad = json.loads(vad_json(clip).read_text(encoding="utf-8"))
        cache[("words", clip)] = load_words(baseline_aligned(clip))
        for arm in ("whisper-prod", args.arm):
            src = baseline_aligned(clip) if arm.startswith("whisper") else qwen_raw(clip)
            cache[("src", clip, arm)] = (vad, json.loads(src.read_text(encoding="utf-8")))

    run = lambda arm, p: run_arm(golds, arm, p, cache, args.by_index)  # noqa: E731
    print(HEAD)
    base_e = run("whisper-prod", Params())
    print(fmt("whisper-prod (基线)", base_e, args.weight_miss))
    print(fmt(f"{args.arm} 当前 Params()", run(args.arm, Params()), args.weight_miss))

    if not args.grid:
        return

    grid = [(f, [float(v) for v in vs.split(",")])
            for f, vs in (part.split("=", 1) for part in args.grid.split(";"))]
    fields = [f for f, _ in grid]
    configs: list[tuple[float, ...]] = [()]
    for _, vals in grid:
        configs = [c + (v,) for c in configs for v in vals]

    rows = []
    for cfg in configs:
        p = replace(Params(), **dict(zip(fields, cfg)))
        rows.append(("/".join(f"{v:g}" for v in cfg), run(args.arm, p)))

    print(f"\n== 在 gold 上直接拟合 {len(configs)} 组（{'/'.join(fields)}）——这是**乐观上界**，不是选参")
    print(HEAD)
    for label, e in sorted(rows, key=lambda r: loss(r[1], args.weight_miss)):
        print(fmt(label, e, args.weight_miss))

    print("\n权重敏感性（各权重下的最优配置；若一路不变，说明结论不是权重选出来的）")
    for w in (1.0, 2.0, 3.0, 5.0, 10.0):
        best = min(rows, key=lambda r: loss(r[1], w))
        beat = "胜过基线" if loss(best[1], w) < loss(base_e, w) else "**仍不及基线**"
        print(f"  w={w:<5g} 最优 {best[0]:<20s} {beat}")


if __name__ == "__main__":
    main()
