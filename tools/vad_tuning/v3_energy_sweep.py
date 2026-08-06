"""Can the production detector be improved on its own terms?

One knob at a time around the shipped operating point, then a focused grid on the
two that move the needle. The constraint is fixed: word recall must not drop.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from energy_sweep import DEFAULTS, compute_tracks, speech_from_tracks, verify
from refs import load_pause_ref, load_word_srt
from score import score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    audio = Path(args.audio)
    if not verify(audio):
        raise SystemExit("sweep shortcut diverged from production; aborting")

    tr = compute_tracks(audio)
    words = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))

    def run(name, **consts):
        sp = speech_from_tracks(tr, consts)
        print(score(sp, words, tr.duration, pause).line(name))

    print()
    run("production default")
    print()

    print("--- MIN_NON_SPEECH_MS (how long a quiet run must be to count) ---")
    for v in (150, 200, 250, 300, 400, 600):
        run(f"  min_ns={v}", MIN_NON_SPEECH_MS=float(v))
    print()

    print("--- MERGE_GAP_MS (how easily speech interrupts a quiet run) ---")
    for v in (50, 75, 100, 150, 250):
        run(f"  merge={v}", MERGE_GAP_MS=float(v))
    print()

    print("--- NEGATIVE_PAD_RIGHT_MS (how early speech is allowed to start) ---")
    for v in (60, 100, 140, 200):
        run(f"  padR={v}", NEGATIVE_PAD_RIGHT_MS=float(v))
    print()

    print("--- NEGATIVE_PAD_LEFT_MS ---")
    for v in (0, 40, 80):
        run(f"  padL={v}", NEGATIVE_PAD_LEFT_MS=float(v))
    print()

    print("--- ABS_NON_SPEECH_MAX_DBFS_ENTER (absolute quiet gate) ---")
    for v in (-36, -33, -30, -27):
        run(f"  absEnter={v}", ABS_NON_SPEECH_MAX_DBFS_ENTER=float(v))
    print()

    if args.grid:
        print("--- grid: min_ns x merge, padR fixed at production 140 ---")
        for m, g in itertools.product((200, 250, 300, 400), (50, 75, 100, 150)):
            run(f"  min={m} merge={g}", MIN_NON_SPEECH_MS=float(m), MERGE_GAP_MS=float(g))
        print()
        print("--- grid: best-looking min_ns x padR ---")
        for m, p in itertools.product((200, 250, 400), (60, 100, 140)):
            run(f"  min={m} padR={p}", MIN_NON_SPEECH_MS=float(m),
                NEGATIVE_PAD_RIGHT_MS=float(p))


if __name__ == "__main__":
    main()
