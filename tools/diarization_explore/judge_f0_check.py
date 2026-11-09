"""Is a cross-sex wrong move of an audio judge really wrong, or is the ruler wrong? Pitch as a third, independent witness.

    python tools/diarization_explore/judge_f0_check.py [--rows out.csv]

The ruler (row truth) and the acoustic layer are not independent -- the ruler was labelled with the acoustic layer in
view -- so "acoustic right, judge wrong" can be the ruler's error. Pitch is independent of both. For each window of
`judge_moves_readout.RUNS` the vocal track the judge heard is read, and every row's median F0 (pYIN, voiced frames
only) is taken. Each person's pitch is the median of their row medians over the window's long (>= REF_SECONDS),
acoustic-right rows. A row then sides with whichever of two people its F0 is nearer in log pitch; it is "unclear" with
fewer than MIN_VOICED voiced frames, or when it lies within MARGIN octaves of the midpoint between the two.

- moves: every cross-sex wrong move of an audio run (unique window, row, new person): does pitch side with the ruler's
  person or with the judge's?
- control: the rule's own error on rows like these, from short (< 2.5 s) acoustic-right rows no audio run moved,
  each against the most-speaking person of the other sex: how often pitch sides with the ruler there.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as M  # noqa: E402

SR = 16000
REF_SECONDS, REF_MAX = 1.5, 40
CONTROL_SECONDS, CONTROL_MAX = 2.5, 80
MIN_VOICED, MARGIN = 5, 0.1


def vocal(key):
    for folder, *_ in M.RUNS.values():
        path = M.SPK / folder / key / "input" / f"{key}-vocal.ogg"
        if path.exists():
            return path
    raise FileNotFoundError(key)


class Pitch:
    def __init__(self, key):
        import librosa
        self.librosa = librosa
        self.wave, _ = librosa.load(str(vocal(key)), sr=SR, mono=True)
        self.cache = {}

    def row(self, seg):
        """(median log2 F0 of the voiced frames, voiced frame count) of one row."""
        span = (float(seg["start"]), float(seg["end"]))
        if span not in self.cache:
            clip = self.wave[int(span[0] * SR):int(span[1] * SR)]
            if len(clip) < 2048:
                self.cache[span] = (None, 0)
            else:
                f0, voiced, prob = self.librosa.pyin(clip, fmin=65, fmax=600, sr=SR, frame_length=1024)
                good = f0[voiced & (prob > 0.5)]
                self.cache[span] = (float(np.median(np.log2(good))) if len(good) else None, int(len(good)))
        return self.cache[span]


def side(f, voiced, ruler, judge):
    if f is None or voiced < MIN_VOICED or ruler is None or judge is None:
        return "unclear"
    if abs(f - (ruler + judge) / 2) < MARGIN:
        return "unclear"
    return "ruler" if abs(f - ruler) < abs(f - judge) else "judge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="write every checked move to this CSV")
    args = ap.parse_args()
    wins, results = M.load(None)
    moves = defaultdict(set)  # (key, row, new person) -> runs
    for (name, key), (rows, _, _) in results.items():
        if M.RUNS[name][2] != "audio":
            continue
        for i, r in rows.items():
            if r["obvious"]:
                moves[key, i, r["new"]].add(name)
    moved_any = {(name_key[1], i) for name_key, (rows, _, _) in results.items()
                 if M.RUNS[name_key[0]][2] == "audio" for i in rows}
    out, control = [], Counter()
    for key in sorted({k for k, _, _ in moves} | set(M.DEV) | set(M.HELDOUT)):
        if key not in wins:
            continue
        win, pitch = wins[key], Pitch(key)
        sexes = M.GENDER.get(win.window.ident, {})
        refs = defaultdict(list)
        for i in win.scored:
            if win.seconds(i) >= REF_SECONDS and win.right(win.acoustic, win.amap, i):
                refs[win.truth[i].person].append(i)
        level = {}
        for person, rows in refs.items():
            values = [pitch.row(win.segs[i]) for i in sorted(rows, key=win.seconds, reverse=True)[:REF_MAX]]
            values = [f for f, n in values if f is not None and n >= MIN_VOICED]
            level[person] = float(np.median(values)) if values else None
        top = {sex: max((p for p in win.weight if sexes.get(p) == sex), key=win.weight.get, default=None)
               for sex in ("m", "f")}
        picked = [i for i in win.scored if win.seconds(i) < CONTROL_SECONDS and (key, i) not in moved_any
                  and win.right(win.acoustic, win.amap, i) and sexes.get(win.truth[i].person) in ("m", "f")]
        for i in picked[::max(1, len(picked) // CONTROL_MAX)][:CONTROL_MAX]:
            person = win.truth[i].person
            other = top["f" if sexes[person] == "m" else "m"]
            control[side(*pitch.row(win.segs[i]), level.get(person), level.get(other))] += 1
        for (k, i, new), runs in sorted(moves.items()):
            if k != key:
                continue
            person = win.truth[i].person
            f, n = pitch.row(win.segs[i])
            verdict = side(f, n, level.get(person), level.get(new))
            out.append(dict(window=key, row=i, seconds=round(win.seconds(i), 2), text=win.segs[i].get("text", ""),
                            ruler=person, judge=new, runs=len(runs), f0=round(2 ** f) if f is not None else "",
                            voiced=n, ruler_f0=round(2 ** level[person]) if level.get(person) else "",
                            judge_f0=round(2 ** level[new]) if level.get(new) else "", verdict=verdict))
        print(f"{key}: person F0 " + ", ".join(f"{p} {round(2 ** v)}" for p, v in level.items() if v is not None))
    total = sum(control.values())
    print(f"control (short acoustic-right rows no audio judge moved, vs the top person of the other sex), n {total}: "
          + ", ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in control.most_common()))
    tally = Counter(r["verdict"] for r in out)
    print(f"cross-sex wrong moves, n {len(out)} (unique window, row, new person): "
          + ", ".join(f"{k} {v}" for k, v in tally.most_common()))
    for r in out:
        print(f"  {r['window'][:7]} {r['row']:4} {r['seconds']:5}s runs {r['runs']} | {r['ruler']} {r['ruler_f0']}Hz"
              f" -> {r['judge']} {r['judge_f0']}Hz | row {r['f0']}Hz ({r['voiced']} voiced) => {r['verdict']} | {r['text'][:24]}")
    if args.rows:
        with open(args.rows, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
