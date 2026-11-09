"""T2 split (report `docs/reports/2026-09-26-ts-diar-models.md`, 补充（2026-10-13）; closed plan `git show bff4b478:docs/plans/asr-recall-timing-plan.md`): where an integrated model's labels, put on production's rows
(`row_truth_cross.TRANSFER`), go wrong -- next to a change of speaker or inside a turn.

Main rows of one person (the row truth's `one` / `multi` with a main-口径 person, prio main), the many-to-one mapping
fitted per window as the cross readout does. A row is "at a change" when the nearest truth row before or after it is
another person. A wrong row at a change is "timing" when the model has a segment of the right person (mapped) within
±NEAR s of the row -- it wrote that person there, the edges just missed -- else "missing" (it wrote no such
segment: not transcribed, or merged into the neighbour's).

    python -m tools.ts_diar_explore.speaker_transfer
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diarization_explore"))
import row_truth_cross as rc  # noqa: E402
import row_truth_external as ext  # noqa: E402
import row_truth_score as rs  # noqa: E402
import window_judge_r9 as r9  # noqa: E402

NEAR = 1.0


def split(w, arm):
    """Counts for one window; `arm` an integrated model (its labels put on production's rows) or one of ours
    (production's own cells: no segments, so no timing / missing split)."""
    stable = rs.stable(w.ident)
    times = {i: (float(stable[i]["start"]), float(stable[i]["end"])) for i in w.rows}
    if arm in rc.OURS:
        segments, cells = [], rc.cells_of(w, arm)
    else:
        found = [o for o in ext.outputs() if o[1] == w.ident and o[2] == arm]
        if not found:
            return None
        segments = ext.rows(w.ident, arm, found[0][3])[0]
        cells = {}
        for i in w.rows:
            a, b = times[i]
            best = max(((min(b, s["end"]) - max(a, s["start"]), s["speaker"]) for s in segments if s["speaker"]),
                       default=(0.0, None))
            cells[i] = [best[1]] if best[0] > 0 else []
    mapping = rs.fit(w, cells)
    verdicts, _ = rs.score_window(w, cells, mapping)
    people = [i for i in w.rows if w.truth[i].person]
    out = Counter()
    for k, i in enumerate(people):
        row = w.truth[i]
        if i not in verdicts or row.prio != "main":
            continue
        change = any(w.truth[people[j]].person != row.person for j in (k - 1, k + 1) if 0 <= j < len(people))
        where = "change" if change else "turn"
        out[f"{where} n"] += 1
        out[f"{where} right"] += verdicts[i]["right"]
        if change and not verdicts[i]["right"]:
            a, b = times[i]
            near = any(mapping.get(s["speaker"]) == row.person and s["end"] > a - NEAR and s["start"] < b + NEAR
                       for s in segments)
            out["timing" if near else "missing"] += 1
    return out


def main():
    rs.rt.ready()
    fresh = [w for w in rs.windows("score") if w.group in r9.FRESH]
    print(f"{len(fresh)} new windows; main rows, many-to-one per window")
    for arm in ("acoustic", "sol", *ext.MODELS):
        total = Counter()
        n = 0
        for w in fresh:
            got = split(w, arm)
            if got is not None:
                total += got
                n += 1
        if not n:
            continue
        print(f"  {rc.NAMES[arm]:10s} ({n} windows): at a change {total['change right']}/{total['change n']} right "
              f"({total['change right'] / max(total['change n'], 1):.1%}), inside a turn {total['turn right']}/{total['turn n']} "
              f"({total['turn right'] / max(total['turn n'], 1):.1%})"
              + ("" if arm in rc.OURS else f"; wrong at a change: timing {total['timing']}, missing {total['missing']}"))


if __name__ == "__main__":
    main()
