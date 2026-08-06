"""Step 18: tighten the boundaries, and stop opening intervals on background.

Two sweeps, each answering one question with the reference that can actually answer
it:

  --mode tight   the hand-corrected clip only. How much dead audio sits at each
                 interval edge, and what it costs to cut it: the 25 annotated real
                 word onsets must stay covered (onset_excl = 0) and no human word
                 may be lost.
  --mode noise   any clip with an ASR run. How many speech-seconds went to intervals
                 that produced no word at all, using the union of several runs'
                 words as the speech-presence map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def build(tr, floor, scorer, pad_left: float, pad_right: float):
    from asr_playground.speech.preprocessing import energy as E

    e = tr.energy_db.numpy().astype(np.float64)
    dbfs = tr.frame_dbfs.numpy().astype(np.float64)
    st = tr.frame_starts.numpy().astype(np.float64)
    en = tr.frame_ends.numpy().astype(np.float64)
    raw = scorer(e, floor(e, st, tr.duration), dbfs, st, en, tr.duration)
    saved = (E.NEGATIVE_PAD_LEFT_MS, E.NEGATIVE_PAD_RIGHT_MS)
    try:
        E.NEGATIVE_PAD_LEFT_MS, E.NEGATIVE_PAD_RIGHT_MS = pad_left, pad_right
        ns = E._apply_negative_padding(raw, tr.duration)
    finally:
        E.NEGATIVE_PAD_LEFT_MS, E.NEGATIVE_PAD_RIGHT_MS = saved
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def plan(kind: str):
    """(label, floor, scorer, padL, padR)."""
    cur_floor = FL.shipped()
    leg = FL.legacy()
    cur_score = SC.prod_minrun(min_speech_frames=8, merge_gap_ms=60.0)
    old_score = SC.production(merge_gap_ms=100.0)
    out = [("pre-branch", leg, old_score, 40.0, 140.0),
           ("current", cur_floor, cur_score, 40.0, 140.0)]
    if kind == "tight":
        # The right shrink is the lead-in knob: it decides how early a speech
        # interval may start. The left shrink is the run-out knob.
        for pr in (120.0, 100.0, 80.0, 60.0, 40.0):
            out.append((f"padR {pr:.0f}", cur_floor, cur_score, 40.0, pr))
        for pl in (60.0, 80.0, 120.0, 160.0):
            out.append((f"padL {pl:.0f}", cur_floor, cur_score, pl, 140.0))
        out.append(("padL 120 / padR 80", cur_floor, cur_score, 120.0, 80.0))
    else:
        # Noise rejection is the enter margin and the floor, not the padding.
        for m in (6.5, 7.0, 8.0, 9.0, 10.0, 12.0):
            out.append((f"margin {m:.0f}", cur_floor,
                        SC.prod_minrun(margin=m, min_speech_frames=8,
                                       merge_gap_ms=60.0), 40.0, 140.0))
        # MIN_NON_SPEECH_MS is in here as a control: it looks like a big win on
        # empty intervals and is not one -- it merges the noise into a neighbouring
        # interval that does contain a word, and the seconds still reach the decoder.
        for ms in (500.0, 600.0):
            out.append((f"minNonSpeech {ms:.0f}", cur_floor,
                        SC.prod_minrun(min_speech_frames=8, merge_gap_ms=60.0,
                                       min_non_speech_ms=ms), 40.0, 140.0))
        # The other entry condition: the absolute dBFS gate. Raising it lets louder
        # frames still count as quiet, which is the opposite lever from the margin.
        for ab in (-26.0, -22.0):
            out.append((f"absEnter {ab:.0f}", cur_floor,
                        SC.prod_minrun(min_speech_frames=8, merge_gap_ms=60.0,
                                       abs_enter=ab), 40.0, 140.0))
        out.append(("margin 9 + absEnter -26", cur_floor,
                    SC.prod_minrun(margin=9.0, min_speech_frames=8,
                                   merge_gap_ms=60.0, abs_enter=-26.0), 40.0, 140.0))
        out.append(("no minrun (margin 6)", cur_floor,
                    SC.production(merge_gap_ms=60.0), 40.0, 140.0))
        out.append(("no minrun, margin 7", cur_floor,
                    SC.production(margin=7.0, merge_gap_ms=60.0), 40.0, 140.0))
        out.append(("minrun6 (looser)", cur_floor,
                    SC.prod_minrun(min_speech_frames=6, merge_gap_ms=60.0),
                    40.0, 140.0))
        out.append(("legacy floor, margin 8", leg,
                    SC.prod_minrun(margin=8.0, min_speech_frames=8,
                                   merge_gap_ms=60.0), 40.0, 140.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("tight", "noise", "guard"))
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--asr", action="append", default=[], metavar="NAME=PATH",
                    help="aligned/stable json feeding the speech-presence map")
    ap.add_argument("--word-srt")
    ap.add_argument("--gold")
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)
    cache = Path(args.cache_dir)
    arms = plan("tight" if args.mode == "tight" else "noise")

    if args.mode in ("tight", "guard"):
        hw = load_word_srt(Path(args.word_srt))
        pause = load_pause_ref(Path(args.gold)) if args.gold else None
        name, path = next(iter(clips.items()))
        tr = cached_tracks(Path(path), cache)
        print(f"{name}: {len(hw)} hand-corrected words, {tr.duration:.0f}s\n")
        print(f"{'arm':<26} {'speech':>7} {'lost':>5} {'onsetExcl':>9} {'pauseExcl':>9} "
              f"| dead audio at interval edges")
        for label, fl, sc, pl, pr in arms:
            sp = build(tr, fl, sc, pl, pr)
            s = score(sp, hw, tr.duration, pause)
            t = tightness(sp, hw)
            print(f"{label:<26} {s.speech_frac:>7.1%} {s.words_lost:>5d} "
                  f"{s.onset_excluded:>9.1%} {s.pause_excluded:>9.1%} | "
                  f"{t.line('')[26:]}")
        return

    print(f"{'arm':<26} " + "  ".join(f"{n[:9]:>18}" for n in clips) +
          f" | {'emptyInt':>8} {'emptySec':>9} {'unvoiced':>9} {'speech':>8}")
    print(f"{'':<26} " + "  ".join(f"{'empty/unvoiced':>18}" for _ in clips))
    rows: Dict[str, list] = {a[0]: [] for a in arms}
    tot = {a[0]: [0.0, 0.0, 0, 0.0] for a in arms}
    for name, path in clips.items():
        tr = cached_tracks(Path(path), cache)
        wm = word_map_from(asr[name])
        for label, fl, sc, pl, pr in arms:
            sp = build(tr, fl, sc, pl, pr)
            v = noise_intervals(sp, wm)
            rows[label].append(f"{v.empty_frac:>8.1%}/{v.unvoiced_frac:<9.1%}")
            tot[label][0] += v.empty_sec
            tot[label][1] += v.speech_sec
            tot[label][2] += v.empty_intervals
            tot[label][3] += v.unvoiced_sec
        print(f"  done {name} ({len(wm)} words in map)", file=sys.stderr, flush=True)
    for label, *_ in arms:
        es, ss, ni, us = tot[label]
        print(f"{label:<26} " + "  ".join(f"{x:>18}" for x in rows[label]) +
              f" | {ni:>8d} {es:>8.0f}s {us:>8.0f}s {ss:>7.0f}s")


if __name__ == "__main__":
    main()
