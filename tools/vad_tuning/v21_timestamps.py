"""Does a tighter interval head make the word timestamps better?

The whole onset study started from word starts being systematically early at pauses.
`NEGATIVE_PAD_RIGHT_MS` decides how much silence a speech interval is allowed to open
with, so shrinking it is the crudest possible way to ask whether that lead-in is what
the decoder was spreading the first word over.

Measured against the 1231 hand-corrected timestamps -- the only reference in the repo
whose timings are not themselves a VAD's output. A word counts as matched when the
same text appears within `--tol` of the annotated position; unmatched words are
reported separately, because a configuration that simply drops the hard ones would
otherwise look like it improved the timings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from refs import Word, load_word_srt  # noqa: E402


def asr_words(path: Path) -> List[Word]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = d["segments"] if isinstance(d, dict) else d
    out = []
    for s in segs:
        for w in (s.get("words") or []):
            t = str(w.get("word", "")).strip()
            if t and float(w["end"]) > float(w["start"]):
                out.append(Word(float(w["start"]), float(w["end"]), t))
    return sorted(out, key=lambda w: w.start)


def match(human: List[Word], got: List[Word], tol: float):
    """Closest same-text ASR word within tol of each annotated word."""
    by_text: Dict[str, List[Word]] = {}
    for w in got:
        by_text.setdefault(w.text, []).append(w)
    dstart, dend, matched, missed = [], [], [], []
    for h in human:
        cand = by_text.get(h.text.strip())
        if not cand:
            missed.append(h)
            continue
        best = min(cand, key=lambda w: abs(w.start - h.start))
        if abs(best.start - h.start) > tol:
            missed.append(h)
            continue
        dstart.append(best.start - h.start)
        dend.append(best.end - h.end)
        matched.append(h)
    return np.array(dstart), np.array(dend), matched, missed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.30,
                    help="silence before a word that makes it an after-gap onset")
    args = ap.parse_args()

    human = [w for w in load_word_srt(Path(args.word_srt))
             if w.text.strip() and "[*]" not in w.text]
    print(f"reference: {len(human)} hand-corrected words "
          f"(matched by identical text within +/-{args.tol:.1f}s)\n")
    # The onset study was about word starts *after a pause*; mid-phrase words are
    # pinned by their neighbours and hide the effect in any aggregate.
    after_gap = set()
    for i, w in enumerate(human):
        if i == 0 or (w.start - human[i - 1].end) >= args.gap:
            after_gap.add(id(w))
    print(f"of which {len(after_gap)} are after-gap onsets "
          f"(>= {args.gap:.2f}s of silence before)\n")

    for cohort, keep in (("ALL WORDS", None), ("AFTER-GAP ONSETS", after_gap)):
        sub = human if keep is None else [w for w in human if id(w) in keep]
        hdr = (f"{cohort:<20} {'matched':>8} {'|dStart| med':>12} {'p90':>7} "
               f"{'bias':>7} {'early>50ms':>11} {'late>50ms':>10} | "
               f"{'|dEnd| med':>10} {'p90':>7}")
        print(hdr)
        print("-" * len(hdr))
        for spec in args.arm:
            label, path = spec.split("=", 1)
            ds, de, m, _miss = match(sub, asr_words(Path(path)), args.tol)
            if ds.size == 0:
                print(f"{label:<20} no matches")
                continue
            a = np.abs(ds)
            print(f"{label:<20} {len(m):>8d} {np.median(a):>12.3f} "
                  f"{np.quantile(a, .9):>7.3f} {np.median(ds):>+7.3f} "
                  f"{float((ds < -0.05).mean()):>11.1%} {float((ds > 0.05).mean()):>10.1%} | "
                  f"{np.median(np.abs(de)):>10.3f} {np.quantile(np.abs(de), .9):>7.3f}")
        print()


if __name__ == "__main__":
    main()
