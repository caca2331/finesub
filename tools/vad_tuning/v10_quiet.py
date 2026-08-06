"""Step 10: does the new noise floor cost quiet speech?

The floor fix can only hurt in one direction: it raises the floor, and everything
within `enter_margin` of the floor is called non-speech. So the thing to measure is
not the average word, it is the *quietest* words -- the ones whose margin was small
to begin with.

Three probes:

  cohort recall   words ranked by their own median energy; recall reported for the
                  quietest decile and quietest 5% separately from the whole set.
  headroom        per-word SNR over the floor, before and after. A word that moves
                  from >6 dB to <6 dB is one the detector can now swallow.
  window sparsity how many anchor windows have so few non-silent frames that the
                  percentile could land inside speech instead of background. That is
                  the failure mode the exclusion rule creates, so it is measured
                  rather than assumed.

Run over every clip that has both separated vocals and a stable.json, plus the noisy
file, so the answer does not rest on the six clips the rule was designed against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402
from energy_sweep import compute_tracks  # noqa: E402
from floor_variants import FloorSpec, TrackerSpec, floor_with_tracker  # noqa: E402
from refs import Word, covered, load_valid_words, load_word_srt  # noqa: E402
from score import score  # noqa: E402

ENERGY_HOP = 0.01


def legacy_floor(tr) -> np.ndarray:
    """The floor as it was before 2026-08-04: percentile over every frame, and the
    tracker free to sink below it."""
    from asr_playground.speech.preprocessing import energy as E

    return floor_with_tracker(
        tr.energy_db.numpy().astype(np.float64),
        tr.frame_starts.numpy().astype(np.float64),
        tr.duration,
        FloorSpec("legacy", percentile=E.NOISE_INIT_PERCENTILE,
                  window_sec=E.NOISE_LOCAL_WINDOW_SEC, hop_sec=E.NOISE_LOCAL_HOP_SEC),
        TrackerSpec("legacy", blend=E.NOISE_LOCAL_BLEND,
                    follow=E.NOISE_TRACK_FOLLOW_ALPHA, rise=E.NOISE_TRACK_RISE_ALPHA,
                    gate=E.NOISE_TRACK_GATE_DB),
    )


def speech_with_floor(tr, floor_np: np.ndarray):
    from asr_playground.speech.preprocessing import energy as E

    floor_t = torch.from_numpy(np.asarray(floor_np, dtype=np.float32))
    raw = E._score_to_non_speech_intervals(
        tr.energy_db, floor_t, tr.frame_dbfs, tr.frame_starts, tr.frame_ends,
        tr.duration, enter_margin_db=6.0, weighted=True)
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def _span(w: Word, n: int):
    a = min(max(int(w.start / ENERGY_HOP), 0), n - 1)
    b = min(max(a + 1, int(w.end / ENERGY_HOP)), n)
    return a, b


def word_levels(words: Sequence[Word], energy_db: np.ndarray) -> np.ndarray:
    out = np.empty(len(words), dtype=np.float64)
    for k, w in enumerate(words):
        a, b = _span(w, len(energy_db))
        out[k] = float(np.median(energy_db[a:b]))
    return out


def word_snr(words: Sequence[Word], energy_db: np.ndarray, floor: np.ndarray) -> np.ndarray:
    out = np.empty(len(words), dtype=np.float64)
    for k, w in enumerate(words):
        a, b = _span(w, len(energy_db))
        out[k] = float(np.median(energy_db[a:b] - floor[a:b]))
    return out


def cohort_recall(speech, words: Sequence[Word]):
    lost = 0
    sec_lost = sec_tot = 0.0
    for w in words:
        d = w.end - w.start
        if d <= 0:
            continue
        miss = d - covered(speech, w.start, w.end)
        sec_tot += d
        sec_lost += miss
        if miss / d >= 0.9:
            lost += 1
    return lost, (1.0 - sec_lost / sec_tot) if sec_tot else 1.0


def gated_track(tr, speech_old):
    """Simulate a separator that emits true digital silence outside speech.

    Rule 1 keeps the percentile away from the separator's silence, but that means
    the frames it does see are increasingly *speech* as the separation gets cleaner.
    Pushed to the limit -- nothing but speech survives -- the 5th percentile would
    land inside speech and the floor would rise into it. This builds that limit case
    from real audio instead of arguing about it.
    """
    e = tr.energy_db.numpy().astype(np.float64).copy()
    keep = np.zeros(len(e), dtype=bool)
    for s, t in speech_old:
        keep[int(s / ENERGY_HOP):min(int(t / ENERGY_HOP) + 1, len(e))] = True
    e[~keep] = -100.0
    return e


def floor_for_track(tr, energy_np: np.ndarray) -> np.ndarray:
    from asr_playground.speech.preprocessing import energy as E

    return E.estimate_noise_floor_db_local(
        torch.from_numpy(energy_np.astype(np.float32)), tr.frame_starts, tr.duration,
        local_window_sec=E.NOISE_LOCAL_WINDOW_SEC, local_hop_sec=E.NOISE_LOCAL_HOP_SEC,
        local_percentile=E.NOISE_INIT_PERCENTILE, track_gate_db=E.NOISE_TRACK_GATE_DB,
        follow_alpha=E.NOISE_TRACK_FOLLOW_ALPHA, rise_alpha=E.NOISE_TRACK_RISE_ALPHA,
        local_blend=E.NOISE_LOCAL_BLEND,
    ).numpy().astype(np.float64)


def speech_with_energy(tr, energy_np: np.ndarray, floor_np: np.ndarray):
    from asr_playground.speech.preprocessing import energy as E

    raw = E._score_to_non_speech_intervals(
        torch.from_numpy(energy_np.astype(np.float32)),
        torch.from_numpy(np.asarray(floor_np, dtype=np.float32)),
        tr.frame_dbfs, tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=True)
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def window_sparsity(tr):
    """Share of non-silent frames per anchor window, and what the percentile lands on."""
    from asr_playground.speech.preprocessing import energy as E
    from floor_variants import DEGENERATE_DB, _window_bounds

    e = tr.energy_db.numpy().astype(np.float64)
    starts = tr.frame_starts.numpy().astype(np.float64)
    _a, i0, i1 = _window_bounds(starts, tr.duration,
                                E.NOISE_LOCAL_HOP_SEC, E.NOISE_LOCAL_WINDOW_SEC)
    fracs, gaps = [], []
    for a, b in zip(i0, i1):
        seg = e[a:b]
        if seg.size == 0:
            continue
        voiced = seg[seg > DEGENERATE_DB]
        fracs.append(voiced.size / seg.size)
        if voiced.size:
            gaps.append(float(np.quantile(voiced, 0.05)) - float(np.quantile(seg, 0.05)))
    return np.array(fracs), np.array(gaps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH",
                    help="clip name = path to the separated vocals")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--word-srt", default=None, help="human timeline (BV1cqLR6hEp3)")
    ap.add_argument("--human-clip", default=None)
    ap.add_argument("--quiet-pct", type=float, default=10.0)
    ap.add_argument("--gate-stress", action="store_true",
                    help="also score a synthetic perfectly-separated version")
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)

    hdr = (f"{'clip':<26} | {'words':>6} {'lostOld':>7} {'lostNew':>7} "
           f"{'recOld':>8} {'recNew':>8} | {'QUIET lostO':>11} {'lostN':>6} "
           f"{'recO':>8} {'recN':>8} | {'spOld':>6} {'spNew':>6}")
    print(hdr)
    print("-" * len(hdr))

    agg = dict(words=0, lo=0, ln=0, sl_o=0.0, sl_n=0.0, st=0.0,
               qw=0, qlo=0, qln=0, qsl_o=0.0, qsl_n=0.0, qst=0.0)
    sparse_rows = []
    snr_moves: List[tuple] = []
    gate_rows: List[tuple] = []

    for name, path in clips.items():
        tr = compute_tracks(Path(path))
        fl_new = tr.noise_floor.numpy().astype(np.float64)
        fl_old = legacy_floor(tr)
        sp_new = speech_with_floor(tr, fl_new)
        sp_old = speech_with_floor(tr, fl_old)

        e = tr.energy_db.numpy().astype(np.float64)
        fr, gp = window_sparsity(tr)
        sparse_rows.append((name, float(np.median(fr)), float(fr.min()),
                            float((fr < 0.10).mean()),
                            float(np.median(gp)) if gp.size else 0.0,
                            float(np.quantile(gp, 0.95)) if gp.size else 0.0,
                            float(np.median(fl_new - fl_old)),
                            float(np.quantile(fl_new - fl_old, 0.95))))

        st = stables.get(name)
        if not st:
            print(f"{name:<26} | (no stable.json) "
                  f"speech {len(sp_old)}->{len(sp_new)} intervals")
            continue
        words, _ = load_valid_words(Path(st))
        if not words:
            continue
        lv = word_levels(words, e)
        cut = np.quantile(lv, args.quiet_pct / 100.0)
        quiet = [w for w, v in zip(words, lv) if v <= cut]

        lo, ro = cohort_recall(sp_old, words)
        ln, rn = cohort_recall(sp_new, words)
        qlo, qro = cohort_recall(sp_old, quiet)
        qln, qrn = cohort_recall(sp_new, quiet)

        so = sum(b - a for a, b in sp_old) / tr.duration
        sn = sum(b - a for a, b in sp_new) / tr.duration
        print(f"{name:<26} | {len(words):>6d} {lo:>7d} {ln:>7d} {ro:>8.3%} {rn:>8.3%} "
              f"| {qlo:>11d} {qln:>6d} {qro:>8.3%} {qrn:>8.3%} | {so:>6.1%} {sn:>6.1%}")

        agg["words"] += len(words)
        agg["lo"] += lo
        agg["ln"] += ln
        agg["qw"] += len(quiet)
        agg["qlo"] += qlo
        agg["qln"] += qln
        for ws, key_t, key_o, key_n in ((words, "st", "sl_o", "sl_n"),
                                        (quiet, "qst", "qsl_o", "qsl_n")):
            for w in ws:
                d = w.end - w.start
                agg[key_t] += d
                agg[key_o] += d - covered(sp_old, w.start, w.end)
                agg[key_n] += d - covered(sp_new, w.start, w.end)

        s_old = word_snr(words, e, fl_old)
        s_new = word_snr(words, e, fl_new)
        crossed = int(((s_old > 6.0) & (s_new <= 6.0)).sum())
        snr_moves.append((name, float(np.quantile(s_old, 0.01)), float(np.quantile(s_new, 0.01)),
                          float(np.quantile(s_old, 0.05)), float(np.quantile(s_new, 0.05)),
                          crossed, len(words)))

        if args.gate_stress:
            ge = gated_track(tr, sp_old)
            gsp = speech_with_energy(tr, ge, floor_for_track(tr, ge))
            gl, gr = cohort_recall(gsp, words)
            gql, gqr = cohort_recall(gsp, quiet)
            gate_rows.append((name, float((ge <= -99.0).mean()), gl, gr, gql, gqr,
                              sum(b - a for a, b in gsp) / tr.duration))

    print("-" * len(hdr))
    print(f"{'TOTAL':<26} | {agg['words']:>6d} {agg['lo']:>7d} {agg['ln']:>7d} "
          f"{1 - agg['sl_o']/agg['st']:>8.3%} {1 - agg['sl_n']/agg['st']:>8.3%} "
          f"| {agg['qlo']:>11d} {agg['qln']:>6d} "
          f"{1 - agg['qsl_o']/agg['qst']:>8.3%} {1 - agg['qsl_n']/agg['qst']:>8.3%} |"
          f"  (quiet cohort n={agg['qw']})")

    print()
    print(f"{'clip':<26} {'voiced% med':>11} {'min':>6} {'<10%':>6} "
          f"{'p5 gap med':>10} {'p95':>7} | {'floor rise med':>14} {'p95':>7}")
    for r in sparse_rows:
        print(f"{r[0]:<26} {r[1]:>11.1%} {r[2]:>6.1%} {r[3]:>6.1%} "
              f"{r[4]:>10.1f} {r[5]:>7.1f} | {r[6]:>14.1f} {r[7]:>7.1f}")

    if snr_moves:
        print()
        print(f"{'clip':<26} {'word SNR p1 old':>15} {'new':>7} "
              f"{'p5 old':>7} {'new':>7} {'crossed<6dB':>12}")
        for r in snr_moves:
            print(f"{r[0]:<26} {r[1]:>15.1f} {r[2]:>7.1f} {r[3]:>7.1f} {r[4]:>7.1f} "
                  f"{r[5]:>7d}/{r[6]:<4d}")

    if gate_rows:
        print()
        print("synthetic 'perfect separation' (non-speech replaced by digital silence)")
        print(f"{'clip':<26} {'silent%':>8} {'lost':>6} {'recall':>8} "
              f"{'QUIET lost':>10} {'recall':>8} {'speech':>7}")
        for r in gate_rows:
            print(f"{r[0]:<26} {r[1]:>8.1%} {r[2]:>6d} {r[3]:>8.3%} "
                  f"{r[4]:>10d} {r[5]:>8.3%} {r[6]:>7.1%}")

    if args.word_srt and args.human_clip:
        print()
        tr = compute_tracks(Path(clips[args.human_clip]))
        hw = load_word_srt(Path(args.word_srt))
        e = tr.energy_db.numpy().astype(np.float64)
        lv = word_levels(hw, e)
        quiet = [w for w, v in zip(hw, lv) if v <= np.quantile(lv, args.quiet_pct / 100.0)]
        for label, fl in (("old", legacy_floor(tr)),
                          ("new", tr.noise_floor.numpy().astype(np.float64))):
            sp = speech_with_floor(tr, fl)
            s = score(sp, hw, tr.duration)
            ql, qr = cohort_recall(sp, quiet)
            print(f"human timeline {label}: lost={s.words_lost} recall={s.word_recall:.3%} "
                  f"| quietest {args.quiet_pct:.0f}% (n={len(quiet)}) "
                  f"lost={ql} recall={qr:.3%}")


if __name__ == "__main__":
    main()
