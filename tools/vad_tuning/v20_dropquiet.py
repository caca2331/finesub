"""Step 20: the synthesis the re-baselined comparison points at.

Re-running the rejected candidates against the new metrics (step 19) does not
resurrect any of them, but it shows *why* they lose, and the reason is a clean
trade rather than incompetence:

  production accumulator   absorbs filled pauses well (pause_excl 50%) but opens a
                           lot of small intervals on background (282 empty)
  viterbi / rolling floors  refuse to open on background (60-138 empty) but let the
                           filled pauses through (pause_excl 28-31%)

The two failure modes are anti-correlated across the whole design space and nothing
tested does both. So try bolting the second property onto the first directly: keep
the accumulator, then drop whole speech intervals that are short *and* barely clear
the noise floor. That is a claim about background, made only where the evidence is
weakest, and it cannot touch a filled pause sitting inside a longer interval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import floor_lab as FL  # noqa: E402
import scorers as SC  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from precision import noise_intervals, tightness, word_map_from  # noqa: E402
from refs import load_pause_ref, load_word_srt  # noqa: E402
from score import score  # noqa: E402
from v18_precision import build  # noqa: E402

HOP = 0.01


def drop_quiet(speech: Sequence[Tuple[float, float]], energy_db: np.ndarray,
               floor: np.ndarray, max_snr: float, max_sec: float,
               stat: str = "median"):
    """Drop whole intervals that never rise above the background.

    `stat` matters more than the threshold. The median over an interval is the wrong
    summary: a 0.2 s word inside a 1.5 s otherwise-quiet interval barely moves it, so
    a median rule drops the interval and takes a perfectly loud word with it --
    measured, it cost ってる at 22.8 dB and サンドローネ at 14.8-15.9 dB. A high
    quantile asks the question actually intended: did anything in here ever sound
    like speech?
    """
    out, dropped, sec = [], 0, 0.0
    for s, t in speech:
        a = min(max(int(s / HOP), 0), len(energy_db) - 1)
        b = min(max(a + 1, int(t / HOP)), len(energy_db))
        band = energy_db[a:b] - floor[a:b]
        snr = float(np.median(band) if stat == "median"
                    else np.quantile(band, 0.90) if stat == "p90"
                    else band.max())
        if (t - s) <= max_sec and snr <= max_snr:
            dropped += 1
            sec += t - s
            continue
        out.append((s, t))
    return out, dropped, sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--asr", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)

    cfg = [("B floor gate (current)", None)]
    for m, d in ((14.0, 2.0),):
        cfg.append((f"median snr<={m:.0f} dur<={d:.0f}s", (m, d, "median")))
    for m in (8.0, 10.0, 12.0, 14.0, 16.0):
        cfg.append((f"p90 snr<={m:.0f} dur<=2s", (m, 2.0, "p90")))
    for m in (10.0, 12.0, 14.0):
        cfg.append((f"p90 snr<={m:.0f} no dur cap", (m, 1e9, "p90")))
    for m in (12.0, 16.0):
        cfg.append((f"peak snr<={m:.0f} no dur cap", (m, 1e9, "max")))

    tot = {c[0]: [0, 0.0, 0.0, 0, 0.0] for c in cfg}
    guard: Dict[str, list] = {}
    cur, prod = FL.shipped(), SC.production(merge_gap_ms=100.0)
    ann_tr = None
    for name, path in clips.items():
        tr = cached_tracks(Path(path), Path(args.cache_dir))
        wm = word_map_from(asr[name])
        e = tr.energy_db.numpy().astype(np.float64)
        stn = tr.frame_starts.numpy().astype(np.float64)
        fl = cur(e, stn, tr.duration)
        base = build(tr, cur, prod, 40.0, 140.0)
        for label, p in cfg:
            sp, dr, ds = (base, 0, 0.0) if p is None else drop_quiet(base, e, fl, *p)
            v = noise_intervals(sp, wm)
            t = tot[label]
            t[0] += v.empty_intervals
            t[1] += v.unvoiced_sec
            t[2] += v.speech_sec
            t[3] += dr
            t[4] += ds
            if name == args.annotated:
                guard[label] = sp
        if name == args.annotated:
            ann_tr = tr
        print(f"  done {name}", file=sys.stderr, flush=True)

    hw = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))
    hdr = (f"{'arm':<28} | {'emptyInt':>8} {'unvoic%':>8} {'speech':>7} {'dropped':>10} "
           f"| {'lost':>4} {'onsetX':>7} {'pauseX':>7} {'cutOn':>5}")
    print(hdr)
    print("-" * len(hdr))
    for label, _ in cfg:
        ei, us, ss, dr, ds = tot[label]
        sp = guard[label]
        s = score(sp, hw, ann_tr.duration, pause)
        t = tightness(sp, hw)
        print(f"{label:<28} | {ei:>8d} {us/ss:>8.1%} {ss:>6.0f}s {dr:>5d}/{ds:>4.0f}s "
              f"| {s.words_lost:>4d} {s.onset_excluded:>7.1%} {s.pause_excluded:>7.1%} "
              f"{t.clipped_onsets:>5d}")


if __name__ == "__main__":
    main()
