"""Row truth, phase 1 readouts (the closed plan's「校验」「建尺」, `git show 423458df:docs/plans/asr-row-truth-plan.md`): what
the built rulers say, before anything was frozen.

    python tools/diarization_explore/row_truth_phase1.py owner        # the owner's 23 rows: each ruler vs the ear
    python tools/diarization_explore/row_truth_phase1.py consistency  # two runs of one window: row agreement, src flips
    python tools/diarization_explore/row_truth_phase1.py anchoring    # labels vs no labels, against the rerun baseline
    python tools/diarization_explore/row_truth_phase1.py readings     # readout one / two under either ruler (simple rule)

A ruler "follows the acoustic layer" on a row when it names the person the row's leading track stands for. A track
stands for the person the labelled first run gives most of the track's `one` rows in that window: a property of the
acoustic layer, read off many rows, so one wrong row barely moves it and every run is measured against the same map.
Runs that are not there yet are skipped.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import row_truth as rt  # noqa: E402
from judge_review_video import OWNER, UNSURE  # noqa: E402

LABELLED, AGAIN, BARE = "labels-medium-r1", "labels-medium-r2", "nolabels-medium-r1"
SCORED = ("one", "multi")


def answer(run, key):
    path = rt.OUT / run / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"] if path.exists() else None


def lead(cell):
    found = re.match(r"s(\d+)", cell or "")
    return f"s{found.group(1)}" if found else None


_windows = None


def windows():
    global _windows
    if _windows is None:
        _windows = {key: (ident, a, b, source) for key, ident, a, b, source in rt.windows()}
    return _windows


_leads = {}


def leads(key):
    """{row id: leading track} of the labels the ruler was shown."""
    if key not in _leads:
        ident, a, b, source = windows()[key]
        cells = rt.rows_and_cells(ident, source)[1]
        _leads[key] = {f"q{i}": lead(cells[i]) for i in range(a, b)}
    return _leads[key]


def track_people(key):
    """{track: person} by the labelled first run's `one` rows."""
    rows = answer(LABELLED, key) or {}
    votes = defaultdict(Counter)
    for q, track in leads(key).items():
        if track and q in rows and rows[q]["cls"] == "one":
            votes[track][rows[q]["who"][0]] += 1
    return {t: max(c.items(), key=lambda kv: (kv[1], kv[0]))[0] for t, c in votes.items()}


def follows(key, q, row):
    person = track_people(key).get(leads(key).get(q))
    return person is not None and row["cls"] in SCORED and person in row["who"]


def same(x, y):
    return x["cls"] == y["cls"] and set(x["who"]) == set(y["who"])


def show(row):
    if row is None:
        return "-"
    return f"{row['cls']} {'+'.join(row['who']) or '·'}" + (f" ({row['note'][:40]})" if row.get("note") else "")


# --- readouts ------------------------------------------------------------------------------------------

def owner(*runs):
    """Plan「校验」: agreement = the ruler's people share one with what the owner heard; a non-scored class disagrees.
    The old ruler had 17 of the 22 it covered. Runs: folder names under OUT (default the phase-1 three)."""
    runs = runs or (LABELLED, AGAIN, BARE)
    tally, wrong_acoustic, follow = Counter(), 0, Counter()
    for (ident, row), heard in sorted(OWNER.items()):
        key = next((k for k, (i, a, b, _) in windows().items() if i == ident and a <= row < b), None)
        q = f"q{row}"
        track = leads(key).get(q) if key else None
        person = track_people(key).get(track) if key else None
        acoustic_wrong = person is not None and person not in heard  # unknown until the labelled run is there
        wrong_acoustic += acoustic_wrong
        cells = []
        for run in runs:
            rows = answer(run, key) if key else None
            if rows is None:
                cells.append(f"{run}: -")
                continue
            r = rows[q]
            agree = r["cls"] in SCORED and bool(set(r["who"]) & heard)
            tally[run] += agree
            tally[run, "n"] += 1
            if acoustic_wrong:
                follow[run] += follows(key, q, r)
                follow[run, "n"] += 1
            cells.append(f"{run} {'✓' if agree else '✗'} {show(r)}")
        print(f"{ident} q{row}{' (owner unsure)' if (ident, row) in UNSURE else ''}: owner {sorted(heard)}; "
              f"acoustic {track}={person}{' WRONG' if acoustic_wrong else ''}\n    " + "\n    ".join(cells))
    print(f"\nagree with the owner (old ruler 17/22): " + ", ".join(f"{run} {tally[run]}/{tally[run, 'n']}" for run in runs))
    print(f"acoustic layer wrong on {wrong_acoustic} of {len(OWNER)} (track -> person by the labelled run); "
          f"rulers following it there: " + ", ".join(f"{run} {follow[run]}/{follow[run, 'n']}" for run in runs))


def pairs(first, second):
    for key in windows():
        x, y = answer(first, key), answer(second, key)
        if x and y:
            yield key, x, y


def consistency():
    """Plan「建尺」自洽: rows agree when class and people match; also src flips, and agreement on the rows where the
    first run does not follow the acoustic layer (copying it twice inflates the plain rate)."""
    for key, x, y in pairs(LABELLED, AGAIN):
        n = len(x)
        agree = sum(same(x[q], y[q]) for q in x)
        flips = sum(x[q]["src"] != y[q]["src"] for q in x)
        free = [q for q in x if not follows(key, q, x[q])]
        print(f"{key}: {agree}/{n} rows agree ({agree / n:.1%}); src flips {flips}; on the {len(free)} rows the first run "
              f"does not follow the acoustic layer {sum(same(x[q], y[q]) for q in free) / max(1, len(free)):.1%}")
        for q in x:
            if not same(x[q], y[q]):
                print(f"    {q}: {show(x[q])}  |  {show(y[q])}")


def anchoring():
    """Plan「建尺」锚定: on rows two runs disagree on, how often only the first names the acoustic layer's person.
    The rerun pair (both labelled) is the noise baseline; labelled vs bare is the anchoring plus that noise."""
    for name, first, second in (("rerun baseline", LABELLED, AGAIN), ("labels vs none", LABELLED, BARE)):
        total = Counter()
        for key, x, y in pairs(first, second):
            split = [q for q in x if not same(x[q], y[q])]
            only_first = sum(follows(key, q, x[q]) and not follows(key, q, y[q]) for q in split)
            only_second = sum(follows(key, q, y[q]) and not follows(key, q, x[q]) for q in split)
            total.update(rows=len(x), split=len(split), first=only_first, second=only_second)
            print(f"{name} {key}: {len(split)}/{len(x)} rows differ; only {first} follows acoustic {only_first}, "
                  f"only {second} {only_second}")
        if total["rows"]:
            print(f"  {name} total: {total['split']}/{total['rows']} differ; only first follows {total['first']}, "
                  f"only second {total['second']}")


def person_of(row, text):
    """The row's scored person: a `one` row's person; a `multi` row's part with the most characters (None when there
    is one part or a tie)."""
    if row["cls"] == "one":
        return row["who"][0]
    if row["cls"] != "multi" or len(row["parts"]) < 2:
        return None
    starts, cursor = [], 0
    for part in row["parts"]:
        cursor = text.find(part["from"], cursor)
        starts.append(cursor)
        cursor += 1
    sizes = [end - start for start, end in zip(starts, [*starts[1:], len(text)])]
    best = max(sizes)
    if sizes.count(best) > 1 or len(row["parts"][sizes.index(best)]["who"]) != 1:
        return None
    return row["parts"][sizes.index(best)]["who"][0]


def readings():
    """Plan「建尺」锚定对读数的影响: readout one (acoustic) and two (Sol's judge) on the scoring window's rows, each label
    mapped many-to-one to the ruler's person with the most of its rows; two rulers, the same rows (scored by both)."""
    import window_judge_accept as acc
    import window_partition_score as ps
    import window_judge_r9 as r9
    chosen, _ = acc.picks(lambda ident: acc.aligned_stable(ident, "sol"))
    loaded = {ident: r9.load(ident) for ident, _, _ in r9.windows()}
    for ident, a, b in r9.windows():
        keys = [k for k, (i, lo, hi, s) in windows().items() if i == ident and s == "accept" and lo < b and hi > a]
        rulers = {run: {} for run in (LABELLED, AGAIN, BARE)}
        for run in rulers:
            for key in keys:
                rulers[run].update(answer(run, key) or {})
        rulers = {run: rows for run, rows in rulers.items() if rows}
        if len(rulers) < 2:
            continue
        m = loaded[ident]
        texts = {i: str(m.stable[i].get("text", "")).strip() for i in range(a, b)}
        truth = {run: {i: person_of(rows[f"q{i}"], texts[i]) for i in range(a, b) if f"q{i}" in rows}
                 for run, rows in rulers.items()}
        common = [i for i in range(a, b) if all(t.get(i) for t in truth.values())]
        labels = {"acoustic": ps.final_labels(None, m, ident, a, b, {}), "sol": ps.final_labels(None, m, ident, a, b, chosen)}
        shown = {i: lead(c) for i, c in enumerate(rt.rows_and_cells(ident, "accept")[1])}
        differ = sum(labels["acoustic"][i] != shown[i] for i in range(a, b))
        print(f"{ident} [{a}, {b}): {len(common)} rows every ruler scores; acoustic labels unlike the ones the ruler saw: {differ}")
        for run, t in truth.items():
            for subset, rows in (("all", common), ("not back-channel", [i for i in common if not r9.backchannel(m, i)])):
                rates = {}
                for name, lab in labels.items():
                    found = ps.match(rows, lab, {i: {t[i]} for i in rows}, one_to_one=False)
                    rates[name] = sum(found.get(lab[i]) == t[i] for i in rows) / max(1, len(rows))
                print(f"  {run:20s} {subset:17s} ({len(rows)}): acoustic {rates['acoustic']:.1%}, sol {rates['sol']:.1%}, "
                      f"sol - acoustic {100 * (rates['sol'] - rates['acoustic']):+.1f}")


def split(*runs):
    """Readout one (acoustic, as production outputs it: `*` fills included) and two (Sol's judge) on each scoring
    window, per ruler: all scored rows, `main` rows, `side` rows. Labels map many-to-one to people once, on all scored
    rows, and every group shares that map."""
    import window_judge_accept as acc
    import window_partition_score as ps
    import window_judge_r9 as r9
    chosen, _ = acc.picks(lambda ident: acc.aligned_stable(ident, "sol"))
    loaded = {ident: r9.load(ident) for ident, _, _ in r9.windows()}
    for ident, a, b in r9.windows():
        m = loaded[ident]
        keys = [k for k, (i, lo, hi, s) in windows().items() if i == ident and s == "accept" and lo < b and hi > a]
        cells = rt.rows_and_cells(ident, "accept")[1]
        acoustic = {i: lead(cells[i]) for i in range(a, b)}
        moved = ps.final_labels(None, m, ident, a, b, chosen)
        sol = {i: moved[i] if (ident, a, i) in chosen else acoustic[i] for i in range(a, b)}
        texts = {i: str(m.stable[i].get("text", "")).strip() for i in range(a, b)}
        for run in runs:
            rows = {}
            for key in keys:
                rows.update(answer(run, key) or {})
            if not all(f"q{i}" in rows for i in range(a, b)):
                continue
            truth = {i: person_of(rows[f"q{i}"], texts[i]) for i in range(a, b)}
            scored = [i for i in range(a, b) if truth[i]]
            maps = {name: ps.match(scored, lab, {i: {truth[i]} for i in scored}, one_to_one=False)
                    for name, lab in (("acoustic", acoustic), ("sol", sol))}
            cells_out = []
            for group, sub in (("all", scored), ("main", [i for i in scored if rows[f"q{i}"].get("prio", "main") == "main"]),
                               ("side", [i for i in scored if rows[f"q{i}"].get("prio") == "side"])):
                rate = {name: sum(maps[name].get(lab[i]) == truth[i] for i in sub) / max(1, len(sub))
                        for name, lab in (("acoustic", acoustic), ("sol", sol))}
                cells_out.append(f"{group}({len(sub)}) ac {rate['acoustic']:.1%} sol {rate['sol']:.1%} "
                                 f"d {100 * (rate['sol'] - rate['acoustic']):+.1f}")
            classes = Counter(rows[f"q{i}"]["cls"] for i in range(a, b))
            print(f"{ident:11s} {run:22s} " + " | ".join(cells_out) + f" | bg {classes['bg']} not scored {b - a - len(scored)}")


def pair(first, second):
    """Two runs of the same prompt (e.g. medium vs high): per window, rows with the same class and people, `prio`
    agreement on the rows both put in one / multi, and the differing rows."""
    for key, x, y in pairs(first, second):
        n = len(x)
        agree = sum(same(x[q], y[q]) for q in x)
        both = [q for q in x if x[q]["cls"] in SCORED and y[q]["cls"] in SCORED]
        prio = sum(x[q].get("prio") == y[q].get("prio") for q in both)
        side = (sum(x[q].get("prio") == "side" for q in both), sum(y[q].get("prio") == "side" for q in both))
        print(f"{key}: class+people {agree}/{n} ({agree / n:.1%}); prio {prio}/{len(both)} ({prio / max(1, len(both)):.1%}); "
              f"side rows {side[0]} vs {side[1]}; classes {dict(Counter(r['cls'] for r in x.values()))} vs "
              f"{dict(Counter(r['cls'] for r in y.values()))}")
        for q in x:
            if not same(x[q], y[q]) or (q in both and x[q].get("prio") != y[q].get("prio")):
                print(f"    {q}: {show(x[q])} [{x[q].get('prio')}]  |  {show(y[q])} [{y[q].get('prio')}]")


if __name__ == "__main__":
    rt.ready()
    {"owner": owner, "consistency": consistency, "anchoring": anchoring, "readings": readings, "pair": pair, "split": split}[sys.argv[1]](*sys.argv[2:])
