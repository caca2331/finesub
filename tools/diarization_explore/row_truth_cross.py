"""Cross-route readouts (the closed plan's 跨路线比较, `git show 423458df:docs/plans/asr-row-truth-plan.md`; readout in findings 「换成一体的转写 + 分人模型」): our route (production's ASR rows with the
acoustic layer, the Sol and Luna judges, and round 9's agy 3.7 Flash judges on raw ASR, text and audio, all scored by the
main line's row truth) next to the integrated models (MOSS Pro, 35t at 300 s and 540 s, MAI2, each scored by its own
text-only ruler from `row_truth_external.py`, frozen in `data/diarization/row-truth/external-v4-text-high/`).

Every route has its own rows, so row counts do not compare; per window and route:

  edit cost / min     main rows' edit cost (owner's prices, as `row_truth_score.row_cost`), the merge price of extra
                      labels and the ASR-merge price (3 for every person beyond the first in a row the ruler split
                      between people; owner 2026-10-13: it counts), per minute of the scoring window's time (the same
                      denominator for every route; owner 2026-10-13: per minute) -- the readout the plan decides on;
                      also without the deletes of extra second labels (owner D4: both shown, a conclusion only when
                      they agree; the integrated models write one label per row)
  right (time)        duration-weighted accuracy of main rows: a `one` row whole, a `multi` row of two parts or more
                      split by its parts' characters (`row_truth_external.credit`)
  coverage            the time of the rows the ruler took for speech (one or several people, main and side; union,
                      clipped to the window) that falls inside the reference's speech time, over that speech time
                      (union of reference lines; the reference's ends run late and 魔裁's text boxes outlast the speech,
                      so it is approximate)
  halluc / h          rows the ruler calls hallucinated (text-only rulers only approximate it)
  overlap right       the accuracy above on rows that overlap another row in time (MOSS Pro, 35t cut per speaker)

The many-to-one mapping (`row_truth_score.fit`) is fitted per window on that route's own rows. Routes are compared only
on windows all of them have; for ours the ruler saw the acoustic labels, not Sol's or Luna's, so the comparison as a
judge's product is unfavourable to the judges; the merged windows (manosaba-b, hina-71) were in the judges' prompt
tuning, which favours them.

    python tools/diarization_explore/row_truth_cross.py [--windows]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import row_truth_external as ext  # noqa: E402
import row_truth_score as rs  # noqa: E402

rt, r9 = ext.rt, ext.r9
#: round 9's agy 3.7 Flash judges reading raw ASR: their answers move rows of production's acoustic cells
AGY = {"agy-text": "text-raw", "agy-audio": "audio-raw"}
OURS = ("acoustic", "sol", "luna", *AGY)
FROZEN = rs.DATA / "external-v4-text-high"  # the integrated models' frozen rulers
#: an integrated model's speaker labels on production's rows (each row takes the label it overlaps most), scored by
#: the main line's row truth: can its diarization stand in for ours on our rows (closed recall-timing plan, T2: `git show bff4b478:docs/plans/asr-recall-timing-plan.md`)
TRANSFER = {f"{arm}>prod": arm for arm in ext.MODELS}
#: an integrated model with round 9's judge (agy text-only) on its own rows (`row_truth_external_judge.py`)
JUDGED = {"mosspro+judge": ("mosspro", rs.DATA / "external-judge-high" / "text")}  # frozen copies
ROUTES = (*OURS, *ext.MODELS, *JUDGED, *TRANSFER)
NAMES = {"acoustic": "acoustic", "sol": "Sol", "luna": "Luna", "agy-text": "agy text", "agy-audio": "agy audio", **ext.MODELS,
         **{k: f"{ext.MODELS[a]}>rows" for k, a in TRANSFER.items()}, "mosspro+judge": "MOSS+judge"}
SEED, REPS = 11, 5000


def merged(intervals):
    """Sorted, non-overlapping intervals covering the same time."""
    out = []
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def length(intervals):
    return sum(b - a for a, b in intervals)


def overlap(xs, ys):
    """Time two merged interval lists share."""
    total, j = 0.0, 0
    for a, b in xs:
        while j < len(ys) and ys[j][1] <= a:
            j += 1
        k = j
        while k < len(ys) and ys[k][0] < b:
            total += min(b, ys[k][1]) - max(a, ys[k][0])
            k += 1
    return total


def score_rows(truth, raw, rows, cells, times, overlapping=frozenset()):
    """One route on one window -> (dict of sums: cost, merge, delete_second, asr_merge, counted / right seconds, halluc,
    overlap; [(start, end)] of the rows taken for speech)."""
    window = types.SimpleNamespace(rows=rows, truth=truth)
    mapping = rs.fit(window, cells)
    verdicts, extra = rs.score_window(window, cells, mapping)
    out, speech = defaultdict(float), []
    for i in rows:
        row = truth[i]
        out["halluc"] += row.cls == "halluc"
        if row.cls in ("one", "multi"):
            speech.append(times[i])
        if row.cls == "multi" and len(row.present) > 1:
            out["asr_merge"] += rs.ASR_MERGE * (len(row.present) - 1)
        if row.prio != "main" or i not in verdicts:
            continue
        out["cost"] += sum(verdicts[i]["cost"].values())
        out["delete_second"] += verdicts[i]["cost"]["delete second"]
        label = (cells.get(i) or [None])[0]
        counted, right = ext.credit(row, raw[i], row.text, mapping.get(label) if label else None, times[i][1] - times[i][0])
        out["counted"] += counted
        out["right"] += right
        if i in overlapping:
            out["overlap_counted"] += counted
            out["overlap_right"] += right
    out["merge"] = sum(extra.values())
    return out, speech


_picks = {}


def cells_of(w, candidate):
    """{row: [labels]}: production's cells for the acoustic layer, Sol and Luna; for an agy judge the acoustic cells
    with its moves applied (as `row_truth_review_video.judge_cells`)."""
    if candidate not in AGY:
        return rs.candidate_cells(w.ident, candidate)
    out = dict(rs.candidate_cells(w.ident, "acoustic"))
    if candidate not in _picks:
        _picks[candidate] = r9.picks_of(AGY[candidate], 1)
    for (ident, a, i), pick in _picks[candidate].items():
        if ident == w.ident and a == w.rows[0] and (re.fullmatch(r"s\d+", pick) or pick.startswith("@")):
            out[i] = [pick]
    return out


def transferred(w, arm):
    """Production's rows labelled by an integrated model's output over this window (None when there is none)."""
    found = [o for o in ext.outputs() if o[1] == w.ident and o[2] == arm]
    if not found:
        return None
    segments = ext.rows(w.ident, arm, found[0][3])[0]
    stable = rs.stable(w.ident)
    cells = {}
    for i in w.rows:
        a, b = float(stable[i]["start"]), float(stable[i]["end"])
        best = max(((min(b, s["end"]) - max(a, s["start"]), s["speaker"]) for s in segments if s["speaker"]),
                   default=(0.0, None))
        cells[i] = [best[1]] if best[0] > 0 else []
    return ours(w, cells)


def ours(w, candidate):
    """Our route on one scoring window; `candidate` names one of OURS, or is its cells."""
    found = rs.rulers()
    raw = {}
    for key, kid, lo, hi, source in rt.windows():
        if kid == w.ident and source == "accept":
            raw.update({i: found[key][f"q{i}"] for i in range(lo, hi)})
    segments = rs.stable(w.ident)
    times = {i: (float(segments[i]["start"]), float(segments[i]["end"])) for i in w.rows}
    cells = candidate if isinstance(candidate, dict) else cells_of(w, candidate)
    return score_rows(w.truth, raw, w.rows, cells, times)


def external(ident, arm, picks=None):
    """An integrated model on one scoring window, from its text-only ruler (None when not built); `picks` {id: answer}
    from a judge replace a judged row's label when they name a label (`s3`) or a new one (`@名字`)."""
    path = FROZEN / f"{ident}-{arm}.json"
    if not path.exists():
        return None
    ruler, truth, shown = ext.ruler_rows(path)
    rows = sorted(truth)
    cells = {i: [shown[i]["label"]] if shown[i]["label"] else [] for i in rows}
    for i in rows:
        pick = str(((picks or {}).get(f"q{i}") or {}).get("track", "")).strip()
        if re.fullmatch(r"s\d+", pick) or pick.startswith("@"):
            cells[i] = [pick]
    times = {i: (shown[i]["start"], shown[i]["end"]) for i in rows}
    every = [shown[i] for i in sorted(shown)]
    hit = ext.overlaps(every)
    overlapping = frozenset(int(every[k]["id"][1:]) for k in hit)
    return score_rows(truth, {i: ruler["rows"][f"q{i}"] for i in rows}, rows, cells, times, overlapping)


def collect():
    """{window key: (window, hours, reference speech seconds, {route: sums})}; `covered` in each route's sums."""
    out = {}
    for w in rs.windows("score"):
        segments = rs.stable(w.ident)
        t0, t1 = float(segments[w.rows[0]]["start"]), float(segments[w.rows[-1]]["end"])
        clip = lambda spans: merged((max(a, t0), min(b, t1)) for a, b in spans)  # noqa: E731
        reference = clip((a, b) for a, b, *_ in rt.reference_lines(w.ident) if b > t0 and a < t1)
        got = {c: ours(w, c) for c in OURS}
        for arm in ext.MODELS:
            result = external(w.ident, arm)
            if result is not None:
                got[arm] = result
        for key, (arm, folder) in JUDGED.items():
            path = folder / f"{w.ident}-{arm}.json"
            if path.exists():
                got[key] = external(w.ident, arm, json.loads(path.read_text(encoding="utf-8"))["answer"])
        for key, arm in TRANSFER.items():
            result = transferred(w, arm)
            if result is not None:
                got[key] = result
        for route, (sums, speech) in got.items():
            sums["covered"] = overlap(clip(speech), reference)
        out[w.key] = (w, (t1 - t0) / 3600, length(reference), {r: sums for r, (sums, _) in got.items()})
    return out


NOTES = """Read with: our routes are scored by rulers built with audio and the acoustic labels (prompts v2/v3), so as a judge's
product Sol and Luna are measured against a ruler anchored on the acoustic layer (unfavourable to them); the merged windows
(manosaba-b, hina-71) were in the judges' prompt tuning (favourable to them). The integrated models' rulers are text-only
with the model's own labels and give no person from a label alone, so unreferenced short reactions drop out of their
scoring. Their hallucination counts are a text-only approximation, ours come from audio: not comparable. 35t windows ran
on 2026-10-06 (manosaba-b) and 2026-10-12/13 (the rest); the server model may have changed in between.
Intervals resample windows; with fewer than 8 windows they are close to the per-window range, and the rulers' own rerun
noise is not in them. `k/n` counts the windows where the route is cheaper."""


def table(name, data, keys):
    windows = [k for k in keys if k in data]
    routes = [r for r in ROUTES if all(r in data[k][3] for k in windows)]
    missing = [r for r in ROUTES if r not in routes]
    hours = sum(data[k][1] for k in windows)
    speech = sum(data[k][2] for k in windows)
    print(f"\n{name}: {len(windows)} windows, {hours * 60:.0f} min"
          + (f"; left out (not on every window): {', '.join(NAMES[r] for r in missing)}" if missing else ""))
    for route in routes:
        s = {f: sum(data[k][3][route][f] for k in windows) for f in
             ("cost", "merge", "delete_second", "asr_merge", "counted", "right", "halluc", "covered",
              "overlap_counted", "overlap_right")}
        cost = s["cost"] + s["merge"] + s["asr_merge"]
        minutes = hours * 60
        line = (f"  {NAMES[route]:10s} edit cost {cost / minutes:5.1f}/min (without second-label deletes "
                f"{(cost - s['delete_second']) / minutes:.1f}; of it merge {s['merge'] / minutes:.1f}, "
                f"ASR merge {s['asr_merge'] / minutes:.1f}, second-label deletes {s['delete_second'] / minutes:.1f}), "
                f"right {100 * s['right'] / max(s['counted'], 1e-9):5.1f}% of time, "
                f"coverage {100 * s['covered'] / max(speech, 1e-9):3.0f}%, halluc {s['halluc'] / minutes:.2f}/min")
        if s["overlap_counted"]:
            line += f", overlapping rows right {100 * s['overlap_right'] / s['overlap_counted']:.1f}% ({s['overlap_counted'] / 60:.1f} min)"
        for base in ("acoustic", "sol"):
            if route not in OURS and len(windows) > 1:
                d = diff(data, windows, route, base)
                k = diff(data, windows, route, base, skip_second=True)
                line += (f"\n             vs {NAMES[base]}: edit cost {d[0]:+.1f}/min [{d[1]:+.1f}, {d[2]:+.1f}] {d[3]}, "
                         f"without second-label deletes {k[0]:+.1f}/min [{k[1]:+.1f}, {k[2]:+.1f}] {k[3]}")
        print(line)


def diff(data, windows, route, base, skip_second=False):
    """Edit cost per minute, route minus base: point, 95% interval from resampling windows, and `k/n` windows where the
    route is cheaper. Each comparison draws from its own seed, so adding a route moves no other interval."""
    cost = lambda k, r: (data[k][3][r]["cost"] + data[k][3][r]["merge"] + data[k][3][r]["asr_merge"]  # noqa: E731
                         - (data[k][3][r]["delete_second"] if skip_second else 0))
    d = np.array([cost(k, route) - cost(k, base) for k in windows])
    h = np.array([data[k][1] * 60 for k in windows])
    rng = np.random.default_rng([SEED, zlib.crc32(f"{route} {base} {skip_second} {windows}".encode())])
    draws = rng.integers(0, len(windows), (REPS, len(windows)))
    sims = d[draws].sum(1) / h[draws].sum(1)
    return d.sum() / h.sum(), np.percentile(sims, 2.5), np.percentile(sims, 97.5), f"{int((d < 0).sum())}/{len(windows)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", action="store_true", help="also one table per window")
    args = parser.parse_args()
    rs.rt.ready()
    data = collect()
    print(NOTES)
    group_merged = lambda k: data[k][0].group not in r9.FRESH  # noqa: E731
    fresh = [k for k in data if not group_merged(k)]
    show = lambda name, keys: table(name, data, keys)  # noqa: E731
    show("9 new windows", fresh)
    show("  dev, 3 new windows", [k for k in fresh if data[k][0].dev])
    show("  held-out, 6 windows", [k for k in data if not data[k][0].dev])
    show("  hina normal, 8 windows", [k for k in data if data[k][0].group == "hina 普通"])
    show("merged, 2 windows (describe only)", [k for k in data if group_merged(k)])
    if args.windows:
        for k in data:
            show(f"window {k} ({data[k][0].group})", [k])


if __name__ == "__main__":
    main()
