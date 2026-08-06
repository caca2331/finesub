"""Final check of whatever is currently in production, through run_vad_file.

Every sweep in this directory runs a shortcut; this one runs the streamed entrypoint
the pipeline actually calls, so the numbers quoted in FINDINGS are the shipped ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from refs import covered, load_pause_ref, load_valid_words, load_word_srt  # noqa: E402
from score import score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--word-srt")
    ap.add_argument("--human-clip")
    ap.add_argument("--gold")
    ap.add_argument("--quiet-pct", type=float, default=10.0)
    args = ap.parse_args()

    import numpy as np

    from asr_playground.speech.preprocessing import energy as E
    from energy_sweep import compute_tracks
    from v10_quiet import word_levels

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)
    tot = dict(lost=0, sl=0.0, st=0.0, qlost=0, qsl=0.0, qst=0.0, sp=0.0, dur=0.0)

    print(f"{'clip':<26} {'words':>6} {'lost':>5} {'recall':>8} {'Qlost':>5} "
          f"{'Qrecall':>8} {'speech':>7} {'n':>5}")
    for name, path in clips.items():
        items, _m, _d, _t = E.run_vad_file(Path(path), params=E.vad_params())
        sp = [(float(i["start"]), float(i["end"])) for i in items]
        tr = compute_tracks(Path(path))
        dur = tr.duration
        tot["sp"] += sum(b - a for a, b in sp)
        tot["dur"] += dur
        st = stables.get(name)
        if not st:
            continue
        words, _ = load_valid_words(Path(st))
        lv = word_levels(words, tr.energy_db.numpy().astype(float))
        cut = np.quantile(lv, args.quiet_pct / 100.0)
        quiet = [w for w, v in zip(words, lv) if v <= cut]
        s = score(sp, words, dur)
        ql = 0
        for w in quiet:
            d = w.end - w.start
            miss = d - covered(sp, w.start, w.end)
            tot["qst"] += d
            tot["qsl"] += miss
            if miss / d >= 0.9:
                ql += 1
        tot["qlost"] += ql
        tot["lost"] += s.words_lost
        tot["sl"] += s.word_sec_lost
        tot["st"] += s.word_sec_total
        qrec = 1 - sum(w.end - w.start - covered(sp, w.start, w.end)
                       for w in quiet) / sum(w.end - w.start for w in quiet)
        print(f"{name:<26} {len(words):>6d} {s.words_lost:>5d} {s.word_recall:>8.3%} "
              f"{ql:>5d} {qrec:>8.3%} {s.speech_frac:>7.1%} {len(sp):>5d}")

    print(f"{'TOTAL':<26} {'':>6} {tot['lost']:>5d} {1 - tot['sl']/tot['st']:>8.3%} "
          f"{tot['qlost']:>5d} {1 - tot['qsl']/tot['qst']:>8.3%} "
          f"{tot['sp']/tot['dur']:>7.1%}")

    if args.word_srt and args.human_clip:
        items, _m, _d, _t = E.run_vad_file(Path(clips[args.human_clip]),
                                           params=E.vad_params())
        sp = [(float(i["start"]), float(i["end"])) for i in items]
        tr = compute_tracks(Path(clips[args.human_clip]))
        s = score(sp, load_word_srt(Path(args.word_srt)), tr.duration,
                  load_pause_ref(Path(args.gold)) if args.gold else None)
        print(s.line("human timeline"))


if __name__ == "__main__":
    main()
