"""Step 9: does a better noise floor fix the separated-vocals failure?

Scored the same way as everything else: the human timeline decides recall, the
annotations decide filled-pause rejection, and the noisy file shows whether the
floor now sits where the residual noise can be rejected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402
from backends import SILERO_HOP_SEC, silero_probs, total  # noqa: E402
from energy_sweep import compute_tracks  # noqa: E402
from floor_variants import VARIANTS, floor_from_targets  # noqa: E402
from refs import load_pause_ref, load_valid_words, load_word_srt  # noqa: E402
from score import score  # noqa: E402


def speech_with_floor(tr, floor_np):
    """Re-run the production scoring stage against a replacement floor."""
    from asr_playground.speech.preprocessing import energy as E

    floor_t = torch.from_numpy(floor_np.astype(np.float32))
    raw = E._score_to_non_speech_intervals(
        tr.energy_db, floor_t, tr.frame_dbfs, tr.frame_starts, tr.frame_ends,
        tr.duration, enter_margin_db=6.0, weighted=True)
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]





def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--noisy", required=True)
    args = ap.parse_args()

    from asr_playground.speech.preprocessing import energy as E
    cache = Path(args.cache_dir)
    pause = load_pause_ref(Path(args.gold))
    hw = load_word_srt(Path(args.word_srt))

    ctx = []
    for c in args.clips:
        a = Path(args.root) / c / f"{c}-vocal.flac"
        tr = compute_tracks(a)
        ctx.append((c, tr, silero_probs(a, cache / f"silero-{c}-vocal.npz"),
                    load_valid_words(Path(args.root) / c / f"{c}-stable.json")[0]))
    na = Path(args.noisy)
    ntr = compute_tracks(na)
    nprobs = silero_probs(na, cache / f"silero-{na.stem}.npz")

    kw = dict(gate=E.NOISE_TRACK_GATE_DB, follow=E.NOISE_TRACK_FOLLOW_ALPHA,
              rise=E.NOISE_TRACK_RISE_ALPHA, blend=E.NOISE_LOCAL_BLEND)

    def floor_for(tr, spec):
        return floor_from_targets(tr.energy_db.numpy().astype(np.float64),
                                  tr.frame_starts.numpy().astype(np.float64),
                                  tr.duration, spec, **kw)

    def noise_over_floor(tr, probs, floor):
        e = tr.energy_db.numpy().astype(np.float64)
        n = min(len(e), int(len(probs) * SILERO_HOP_SEC / 0.01))
        idx = (np.arange(n) * 0.01 / SILERO_HOP_SEC).astype(int).clip(0, len(probs) - 1)
        snr = (e - floor)[:n][probs[idx] < 0.2]
        return float(np.quantile(snr, 0.75)), float(np.quantile(snr, 0.90))

    print(f"{'floor variant':<28} | {'CLEAN lost':>10} {'recall':>8} {'clipH':>6} {'speech':>7} "
          f"{'pause':>6} {'humanLost':>9} | {'NOISY p75':>9} {'p90':>6} {'speech':>7} {'ghost':>6}")
    for spec in VARIANTS:
        rows = []
        for c, tr, probs, w in ctx:
            sp = speech_with_floor(tr, floor_for(tr, spec))
            rows.append(score(sp, w, tr.duration))
        c0, tr0, p0, _ = ctx[0]
        h = score(speech_with_floor(tr0, floor_for(tr0, spec)), hw, tr0.duration, pause)
        nfl = floor_for(ntr, spec)
        nsp = speech_with_floor(ntr, nfl)
        p75, p90 = noise_over_floor(ntr, nprobs, nfl)
        ghost = 0
        for s, e in nsp:
            i0 = int(s / SILERO_HOP_SEC)
            i1 = max(i0 + 1, int(e / SILERO_HOP_SEC))
            if float(nprobs[i0:min(i1, len(nprobs))].max()) < 0.5:
                ghost += 1
        rec = 1 - sum(r.word_sec_lost for r in rows) / sum(r.word_sec_total for r in rows)
        sf = sum(r.speech_frac * r.duration for r in rows) / sum(r.duration for r in rows)
        print(f"{spec.name:<28} | {sum(r.words_lost for r in rows):>10d} {rec:>8.3%} "
              f"{sum(r.clipped_head for r in rows):>6d} {sf:>7.1%} {h.pause_excluded:>6.1%} "
              f"{h.words_lost:>9d} | {p75:>9.1f} {p90:>6.1f} "
              f"{total(nsp)/ntr.duration:>7.1%} {ghost:>6d}")


if __name__ == "__main__":
    main()
