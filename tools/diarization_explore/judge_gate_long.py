"""Candidate one of `docs/reports/2026-11-14-audio-judge-overlap.md`: gate more than F1b, offline (owner 2026-11-14).

    python tools/diarization_explore/judge_gate_long.py

Replays every registered audio judge run (the pool of `judge_moves_overlap.py`: not the silence placeholder, not
P13, production's two runs as their " raw" answers) under each gate of `judge_moves_readout.gated`, on the plain hina
windows (dev, held-out) and the merged-track windows, and sums per group:

- moved rows: wrong moves, fixes and net (fixes - wrong), as `judge_moves_readout.py` counts them;
- whole window: rows right after the judge minus rows right before it, every label refitted to people per window --
  the reading that also counts the rows a merged track's reassignment turns right (findings 「比裁判时别只数被挪动的行」).

Gates: none; short (F1b, production: a move onto an existing track outside the cell on a short row is not taken);
shortiso (F1b, and on longer rows also when isolated: neither neighbour moved onto the same track); shortnbr (the
same, but neither neighbour ends on that track, moved or acoustic); isolated (F1c);
cell (F1: every move outside the cell). The readout's short row is under 1.5 s or at most 4 characters; production's
F1b looks at seconds only. The text runs of the same windows, ungated, are listed for reference.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as R  # noqa: E402

GATES = (None, "short", "shortiso", "shortnbr", "isolated", "cell")


def pool_of(name):
    media = R.RUNS[name][2]
    if media == "text":
        return "text"
    if "silence" in name or "p13" in name or name in ("Hyd dev r1", "38F dev r1"):
        return None
    return "audio"


def main():
    table = {}
    for gate in GATES:
        wins, results = R.load(gate)
        for (name, key), (rows, correct, _) in results.items():
            pool = pool_of(name)
            if pool is None or (pool == "text" and gate is not None):
                continue
            group = next(g for g, keys in R.GROUPS if key in keys)
            kinds = Counter(r["kind"] for r in rows.values())
            t = table.setdefault((pool, gate, group), Counter())
            t.update(runs=1, wrong=kinds["wrong"], fix=kinds["fix"],
                     whole=sum(correct.values()) - wins[key].a_right, scored=len(wins[key].scored))
    for pool, gates in (("audio", GATES), ("text", (None,))):
        print(f"=== {pool} runs" + ("" if pool == "audio" else " (ungated, reference)"))
        for group, _ in R.GROUPS:
            for gate in gates:
                t = table.get((pool, gate, group))
                if not t:
                    continue
                n = t["runs"]
                print(f"  {group:9} gate {gate or 'none':9} {n:3} run-windows | moved rows: wrong {t['wrong']:4}"
                      f"  fix {t['fix']:4}  net {t['fix'] - t['wrong']:+5} ({(t['fix'] - t['wrong']) / n:+.1f} per run-window)"
                      f" | whole window {t['whole']:+5} ({t['whole'] / n:+.1f} per run-window)")


if __name__ == "__main__":
    main()
