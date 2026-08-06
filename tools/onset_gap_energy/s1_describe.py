"""Step 1: what does the production start error actually look like, and how much
of the search window is silence according to the production VAD track?

No audio is touched here -- only the gold table plus the VAD interval list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, Candidate, load_candidates, load_vad, summarize


def vad_state_at(non_speech, t: float) -> bool:
    """True if t falls inside a VAD non-speech interval."""
    for a, b in non_speech:
        if a <= t < b:
            return True
        if a > t:
            break
    return False


def silence_after(non_speech, t: float) -> float:
    """Length of contiguous non-speech starting at t (0 if t is speech)."""
    for a, b in non_speech:
        if a <= t < b:
            return b - t
        if a > t:
            break
    return 0.0


def next_speech_onset(speech, t: float) -> float | None:
    for a, b in speech:
        if a >= t:
            return a
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--vad", required=True)
    args = ap.parse_args()

    cands = load_candidates(Path(args.gold))
    speech, non_speech, duration = load_vad(Path(args.vad))

    print(f"VAD: {len(speech)} speech / {len(non_speech)} non-speech over {duration:.2f}s")
    ns_len = np.array([b - a for a, b in non_speech])
    print(f"non-speech length: med={np.median(ns_len):.3f} p10={np.quantile(ns_len,0.1):.3f} "
          f"min={ns_len.min():.3f} max={ns_len.max():.3f}")
    print()

    print("=== baseline: production start error by position x label ===")
    for pos in ("segment-boundary", "after-gap", "mid-phrase"):
        sub = [c for c in cands if c.position == pos]
        print(summarize([c.error for c in sub], pos))
        for lab in ("filled_pause", "partial", "word_onset"):
            s2 = [c for c in sub if c.label == lab]
            if s2:
                print("   " + summarize([c.error for c in s2], lab))
    print()
    target = [c for c in cands if c.position in TARGET_SUBSETS]
    print(summarize([c.error for c in target], "TARGET (sb+ag)"))
    print()

    print("=== per-candidate view of the target subset ===")
    hdr = (f"{'#':>3} {'pos':<17} {'label':<13} {'plain':>8} {'onset':>8} {'err':>7} "
           f"{'blkend':>8} {'gap':>6} {'vadNS@plain':>11} {'silLen':>7} {'nextSpOnset':>11} {'sp-onset err':>12}")
    print(hdr)
    rows = []
    for c in sorted(target, key=lambda c: c.plain_start):
        ns = vad_state_at(non_speech, c.plain_start)
        sil = silence_after(non_speech, c.plain_start)
        sp = next_speech_onset(speech, c.plain_start)
        sp_err = (sp - c.onset) if sp is not None else float("nan")
        rows.append((c, ns, sil, sp, sp_err))
        print(f"{c.index:>3} {c.position:<17} {c.label:<13} {c.plain_start:>8.3f} {c.onset:>8.3f} "
              f"{c.error:>+7.3f} {c.end:>8.3f} {c.preceding_gap:>6.3f} {str(ns):>11} {sil:>7.3f} "
              f"{(sp if sp is not None else float('nan')):>11.3f} {sp_err:>+12.3f}")
    print()

    print("=== trivial oracles on the target subset ===")
    print(summarize([c.error for c in target], "as-is (production)"))
    print(summarize([c.onset - c.end for c in target], "snap to block end"))
    print(summarize([c.onset - (c.start + c.duration / 2) for c in target], "snap to block mid"))
    # VAD-driven: if the plain start sits in non-speech, jump to the next speech onset
    vad_pred = []
    for c, ns, sil, sp, _ in rows:
        pred = sp if (ns and sp is not None) else c.plain_start
        vad_pred.append(c.onset - pred)
    print(summarize(vad_pred, "VAD next-speech (gated)"))
    vad_all = []
    for c, ns, sil, sp, _ in rows:
        pred = sp if sp is not None and sp <= c.end + 0.2 else c.plain_start
        vad_all.append(c.onset - pred)
    print(summarize(vad_all, "VAD next-speech (bounded)"))


if __name__ == "__main__":
    main()
