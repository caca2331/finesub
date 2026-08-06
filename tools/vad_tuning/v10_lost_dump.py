"""What exactly does the new floor swallow?

Counting lost words says how much; this says what. A regression made of fillers and
one-frame fragments is a different thing from one made of sentence pieces, and the
tolerance the user set is stated in those terms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from energy_sweep import compute_tracks  # noqa: E402
from refs import covered, load_valid_words  # noqa: E402
from v10_quiet import ENERGY_HOP, _span, legacy_floor, speech_with_floor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--stable", required=True)
    ap.add_argument("--context", type=int, default=2, help="neighbour words to show")
    args = ap.parse_args()

    tr = compute_tracks(Path(args.audio))
    e = tr.energy_db.numpy().astype(np.float64)
    fl_new = tr.noise_floor.numpy().astype(np.float64)
    fl_old = legacy_floor(tr)
    sp_old = speech_with_floor(tr, fl_old)
    sp_new = speech_with_floor(tr, fl_new)
    words, _ = load_valid_words(Path(args.stable))

    rows = []
    for k, w in enumerate(words):
        d = w.end - w.start
        if d <= 0:
            continue
        mo = (d - covered(sp_old, w.start, w.end)) / d
        mn = (d - covered(sp_new, w.start, w.end)) / d
        if mo < 0.9 <= mn:
            a, b = _span(w, len(e))
            ctx = "".join(x.text for x in words[max(0, k - args.context):
                                               k + args.context + 1])
            rows.append((w.start, d, w.text, float(np.median(e[a:b])),
                         float(np.median(e[a:b] - fl_old[a:b])),
                         float(np.median(e[a:b] - fl_new[a:b])), ctx))

    print(f"newly lost: {len(rows)} of {len(words)} valid words")
    print(f"{'t':>9} {'dur':>5} {'word':<10} {'dB':>7} {'snrOld':>7} {'snrNew':>7}  context")
    for r in sorted(rows, key=lambda x: -x[1])[:40]:
        print(f"{r[0]:>9.2f} {r[1]:>5.2f} {r[2][:10]:<10} {r[3]:>7.1f} {r[4]:>7.1f} "
              f"{r[5]:>7.1f}  {r[6][:34]}")
    if rows:
        dur = np.array([r[1] for r in rows])
        print(f"\nduration: median {np.median(dur):.2f}s  <=0.2s {(dur <= 0.2).mean():.0%}  "
              f"<=0.1s {(dur <= 0.1).mean():.0%}")


if __name__ == "__main__":
    main()
