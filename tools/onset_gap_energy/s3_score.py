"""Step 3: score forward-search detectors against the gold onsets.

Reports abs-error stats plus paired win/tie/loss against the production baseline,
separately for the target subset (segment-boundary + after-gap) and for the
`word_onset` rows where production is already right -- those are the damage check.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, load_candidates, summarize
from detectors import LastGapExit, OnsetPeak, SilenceExit, keep
from features import compute_tracks


def paired(base: np.ndarray, new: np.ndarray, tol: float = 0.02) -> str:
    d = np.abs(base) - np.abs(new)
    win = int(np.sum(d > tol))
    loss = int(np.sum(d < -tol))
    tie = len(d) - win - loss
    return f"W/T/L={win}/{tie}/{loss}"


def evaluate(name, det, cands, tr, limit_of) -> dict:
    errs, base = [], []
    for c in cands:
        pred = det(tr, c.plain_start, limit_of(c))
        errs.append(c.onset - pred)
        base.append(c.error)
    return {"name": name, "err": np.array(errs), "base": np.array(base)}


def report(title, res_list) -> None:
    print(f"### {title}")
    for r in res_list:
        line = summarize(r["err"], r["name"])
        if r["name"] != "baseline":
            line += "  " + paired(r["base"], r["err"])
        print(line)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cands = load_candidates(Path(args.gold))
    tr = compute_tracks(Path(args.audio), Path(args.cache) if args.cache else None)

    target = [c for c in cands if c.position in TARGET_SUBSETS]
    safe = [c for c in cands if c.label == "word_onset"]      # production already correct
    midph = [c for c in cands if c.position == "mid-phrase"]

    # A production heuristic has no block; the only bound it can know is the next
    # word's start from its own output. Use a generous fixed window instead.
    limit_of = lambda c: None

    variants = [
        ("baseline", keep),
        ("silence d=5 k=3", SilenceExit(delta_db=5, sustain=3)),
        ("silence d=8 k=3", SilenceExit(delta_db=8, sustain=3)),
        ("silence d=12 k=3", SilenceExit(delta_db=12, sustain=3)),
        ("silence d=8 k=5", SilenceExit(delta_db=8, sustain=5)),
        ("silence d=8 k=3 W=.5", SilenceExit(delta_db=8, sustain=3, window=0.5)),
        ("lastgap d=8 g=.06", LastGapExit(delta_db=8, min_gap=0.06)),
        ("lastgap d=8 g=.12", LastGapExit(delta_db=8, min_gap=0.12)),
        ("onsetpeak z=2", OnsetPeak(z=2.0)),
        ("onsetpeak z=3", OnsetPeak(z=3.0)),
    ]

    for title, subset in (("TARGET: segment-boundary + after-gap (n=21)", target),
                          ("DAMAGE: word_onset rows, production already right", safe),
                          ("mid-phrase (not a fix target; reference only)", midph)):
        report(title, [evaluate(n, d, subset, tr, limit_of) for n, d in variants])

    if not args.sweep:
        return

    print("### parameter sweep on the target subset (min_move=0.05, window=1.0)")
    print(f"{'delta':>6} {'sust':>5} {'back':>6} | {'med':>6} {'p90':>6} {'max':>6} {'>0.1':>5} | "
          f"{'W/T/L target':>14} | {'damage med':>10} {'dmg loss':>8}")
    for delta, sustain, backoff in itertools.product((4, 6, 8, 10, 14), (2, 3, 5), (0.0, 0.02, 0.05)):
        det = SilenceExit(delta_db=delta, sustain=sustain, backoff=backoff)
        rt = evaluate("x", det, target, tr, limit_of)
        rs = evaluate("x", det, safe, tr, limit_of)
        a = np.abs(rt["err"])
        ds = np.abs(rs["err"])
        dl = int(np.sum(np.abs(rs["base"]) - ds < -0.02))
        print(f"{delta:>6} {sustain:>5} {backoff:>6.2f} | {np.median(a):>6.3f} "
              f"{np.quantile(a,0.9):>6.3f} {a.max():>6.3f} {np.mean(a>0.1):>5.0%} | "
              f"{paired(rt['base'], rt['err']):>14} | {np.median(ds):>10.3f} {dl:>8d}")


if __name__ == "__main__":
    main()
