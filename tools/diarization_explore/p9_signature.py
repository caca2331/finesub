"""P9 follow-up (docs/plans/argon-audio-judge-plan.md): do the audio judge's wrong moves carry a signature production can
see, apart from its fixes?

    python tools/diarization_explore/p9_signature.py

For every move of the audio runs in `p9_gates.FAMILIES` (held-out and merged windows), features computable without the
truth: short (F1b's), outside the cell, sandwiched (the nearest rows before and after carry the row's own acoustic
lead track), isolated (neither neighbour answered the same track), echo (a stretch of at least ECHO characters of the
row also stands in the previous row), cross-band (`p9_gates.Levels`). Prints the share of wrong moves, cross-sex
wrong moves and fixes carrying each feature and each pair of features.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as M  # noqa: E402
import p9_gates as G  # noqa: E402

from finesub.llm.stages.speaker_judge.prompt import cell_tracks  # noqa: E402

ECHO = 3


def lead(win, i):
    labels = win.acoustic.get(i) or []
    return labels[0] if labels else None


def bare(text):
    return re.sub(r"[\W_]", "", text or "")


def features(win, lv, i, track, tracks):
    row = win.rows[i]
    text, prev = bare(win.segs[i].get("text")), bare(win.segs[i - 1].get("text")) if i else ""
    out = set()
    if M.short(row):
        out.add("short")
    if track not in cell_tracks(row, marks=True):
        out.add("outside")
    if i and i + 1 < len(win.segs) and lead(win, i - 1) == lead(win, i) == lead(win, i + 1):
        out.add("sandwiched")
    if tracks.get(i - 1) != track and tracks.get(i + 1) != track:
        out.add("isolated")
    if len(text) >= ECHO and any(text[k:k + ECHO] in prev for k in range(len(text) - ECHO + 1)):
        out.add("echo")
    if lv.cross(i, track):
        out.add("cross-band")
    return out


def main():
    wins, results = M.load(None)
    names = {n for f in G.FAMILIES.values() for n in f}
    for group, keys in G.GROUPS:
        count = Counter()
        for key in keys:
            win, lv = wins[key], G.Levels(wins[key])
            for name in names:
                if (name, key) not in results:
                    continue
                rows, _, payload = results[name, key]
                tracks = G.answers(payload, win)
                for i, r in rows.items():
                    if r["kind"] not in ("wrong", "fix") or i not in tracks:
                        continue
                    kind = "x-wrong" if r["obvious"] else r["kind"]
                    f = features(win, lv, i, tracks[i], tracks)
                    count[kind, "all"] += 1
                    for a in f:
                        count[kind, a] += 1
                    for a, b in combinations(sorted(f), 2):
                        count[kind, f"{a}+{b}"] += 1
                    for a, b, c in combinations(sorted(f), 3):
                        count[kind, f"{a}+{b}+{c}"] += 1
        kinds = ("x-wrong", "wrong", "fix")
        print(f"\n=== {group}: moves onto a known track, pooled over runs; n " +
              ", ".join(f"{k} {count[k, 'all']}" for k in kinds) + " (wrong = same-sex wrong)")
        feats = sorted({f for (_, f) in count if f != "all"}, key=lambda f: (f.count("+"), f))
        for f in feats:
            print(f"  {f:34} " + "  ".join(f"{k} {count[k, f]:4} ({100 * count[k, f] / max(count[k, 'all'], 1):3.0f}%)"
                                         for k in kinds))


if __name__ == "__main__":
    main()
