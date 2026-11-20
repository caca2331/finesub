"""What the speaker judge's wrong moves and fixes look like, overlapped speech first (owner 2026-11-14).

    python tools/diarization_explore/judge_moves_overlap.py [--thr 0.2] [--rows out.csv]

Reads the same runs and scoring as `judge_moves_readout.py` (no gate; the production runs as their " raw" variant,
what the model answered before F1b) on the five plain hina windows (dev and held-out; the merged-track windows are
left out: their moves are reassignments of a merged track, another error class). Pools:

- audio: audio runs that heard the recording and answered in one step (not the silence placeholder, not P13's
  text-draft-then-audio patches, not the F1b-gated copies);
- text: every text run of the same windows (the control: a text judge cannot hear an overlap, so what overlap
  explains in its errors is the rows, not the hearing).

Overlap per row, two independent sources:

- diar: the acoustic layer's own timeline, as production stores it in each row (`speaker.coverage`: share of the
  row each track covers; tracks may overlap). Tracks map to people by the acoustic mapping of the window (every
  track to the person with most of its rows); a row is overlapped when two different people each cover at least
  THR of it. `target audible`: the person the judge moved the row to covers at least THR;
- simultaneous (`sim_ovl`, the strict reading): the same timeline read frame by frame (`TIMELINES`, the acceptance
  run's `-diar.json`, which reproduces the stored coverage): two different people active at once over at least THR
  of the row; `target in sim`: the judge's new person is one of the people in those frames. `diar` alone also
  counts a row whose first half is one person and second half another;
- MAI2: MAI2's diarized transcript of the same file (`judge_moves_readout.MAI2`), two of its speakers each covering
  at least THR of the row. Its speakers split hina's cast by sex only, so a same-sex overlap mostly reads as none.

Rates: per run and row, the share of acoustic-right rows moved wrong (and of acoustic-wrong rows fixed), split by
overlapped or not, pooled over runs; plus each feature's share among wrong moves, fixes and all rows. The ruler is
3.8 Flash with audio and may itself miss on overlapped rows; a wrong move on such a row may be the ruler's miss.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as R  # noqa: E402

WINDOWS = (*R.DEV, *R.HELDOUT)
FILLER = re.compile(r"^(うん|ああ|あー|はい|ええ|へえ|へー|ほう|おお|おー|なるほど|そう|そうそう|そうね|そうだね|あ|え|お|ん|ね|わ|うわ|やば|すご|確かに)+[。、！？!?〜ー…]*$")


def pool_of(name):
    folder, model, media, prompt, shaped, keys = R.RUNS[name]
    if media == "text":
        return "text"
    if "silence" in name or "p13" in name or name in ("Hyd dev r1", "38F dev r1"):
        return None
    return "audio"


TIMELINES = R.S.r9.MAIN / "tmp" / "diarization-retrack" / "accept"
FRAME = 0.02


class Rows:
    """Per window: each scored row's features."""

    def __init__(self, win, thr, mai2):
        self.win, self.thr = win, thr
        self.mai2 = mai2
        ident = win.key.rsplit("-c", 1)[0]
        payload = json.loads((TIMELINES / ident / f"{ident}-diar.json").read_text(encoding="utf-8"))
        self.timeline = sorted(payload["tracks"], key=lambda t: t["start"])

    def simultaneous(self, i):
        """(share of the row where two different people are active at once, the people active in those frames)."""
        s, e = float(self.win.segs[i]["start"]), float(self.win.segs[i]["end"])
        spans = [(max(s, t["start"]), min(e, t["end"]), self.win.amap.get(f"s{t['track']}", f"?s{t['track']}"))
                 for t in self.timeline if t["start"] < e and t["end"] > s]
        n = max(int(round((e - s) / FRAME)), 1)
        both, who = 0, set()
        for k in range(n):
            x = s + (k + 0.5) * FRAME
            here = {p for lo, hi, p in spans if lo <= x < hi}
            if len(here) >= 2:
                both += 1
                who |= here
        return both / n, who

    def people_cov(self, i):
        cov = (self.win.segs[i].get("speaker") or {}).get("coverage") or {}
        out = Counter()
        for track, share in cov.items():
            out[self.win.amap.get(f"s{track}", f"?s{track}")] += share
        return out

    def diar_overlap(self, i):
        return sum(share >= self.thr for share in self.people_cov(i).values()) >= 2

    def audible(self, i, person):
        return self.people_cov(i).get(person, 0.0) >= self.thr

    def mai2_overlap(self, i):
        if self.mai2 is None:
            return None
        segs, offset = self.mai2
        s, e = float(self.win.segs[i]["start"]), float(self.win.segs[i]["end"])
        span = max(e - s, 1e-6)
        over = Counter()
        for g in segs:
            o = min(e, g["end"] + offset) - max(s, g["start"] + offset)
            if o > 0:
                over[g["speaker"]] += o / span
        return sum(v >= self.thr for v in over.values()) >= 2

    def neighbours(self, i):
        scored = self.win.scored
        before = [j for j in scored if j < i][-1:]
        after = [j for j in scored if j > i][:1]
        return [self.win.truth[j].person for j in before + after]

    def features(self, i, new=None):
        win = self.win
        text = re.sub(r"[\W_]", "", win.truth[i].text)
        nb = self.neighbours(i)
        own = win.truth[i].person
        out = dict(
            seconds=win.seconds(i), chars=len(text),
            short=R.short(win.rows[i]), filler=bool(FILLER.match(text)),
            diar_ovl=self.diar_overlap(i), mai2_ovl=self.mai2_overlap(i),
            sim_ovl=self.simultaneous(i)[0] >= self.thr,
            low=bool(win.rows[i].low), tracks2=len(win.rows[i].tracks) > 1,
            turn_edge=any(p != own for p in nb),
        )
        if new is not None:
            out.update(target_audible=self.audible(i, new), target_adjacent=new in nb,
                       target_in_sim=new in self.simultaneous(i)[1],
                       cross_sex=win.obvious(i, new))
        return out


def new_track_cov(win, i, label):
    """The share of the row the judge's chosen track covers (None for a new `@` label): what production could see."""
    if not label.startswith("s"):
        return None
    return round(((win.segs[i].get("speaker") or {}).get("coverage") or {}).get(label[1:], 0.0), 3)


CATEGORIES = ("A overlap, moved to a voice in it", "B overlap, moved to someone else",
              "C no overlap, new person audible", "D no overlap, new person silent")


def category(f):
    if f["sim_ovl"]:
        return CATEGORIES[0] if f["target_in_sim"] else CATEGORIES[1]
    return CATEGORIES[2] if f["target_audible"] else CATEGORIES[3]


def breakdown(moves, feats):
    """Audio pool: wrong moves and fixes by category (instances, unique rows), per model; then the share of moves
    that were right under features production can see."""
    print("\n##### audio pool: categories (instances / unique rows)")
    for kind in ("wrong", "fix"):
        items = moves["audio", kind]
        inst = Counter(category(f) for *_, f, _ in items)
        uniq = Counter(category(f) for f in {(k, i): f for k, i, _, f, _ in items}.values())
        print(f"  {kind}: " + "  ".join(f"{c[:1]} {inst[c]} / {uniq[c]}" for c in CATEGORIES)
              + f"  (total {len(items)} / {sum(uniq.values())})")
    print("  " + "; ".join(CATEGORIES))
    for model in sorted({R.RUNS[name][1] for _, _, name, _, _ in moves["audio", "wrong"]}):
        items = [(k, i, f) for k, i, name, f, _ in moves["audio", "wrong"] if R.RUNS[name][1] == model]
        runs = {name for _, _, name, _, _ in moves["audio", "wrong"] if R.RUNS[name][1] == model}
        uniq = Counter(category(f) for f in {(k, i): f for k, i, f in items}.values())
        print(f"  {model}, wrong, unique rows ({len(runs)} runs): " + "  ".join(f"{c[:1]} {uniq[c]}" for c in CATEGORIES))
    print("  share of moves that were right (audio instances), by what production can see:")
    every = [(kind, f, r, feats[k].win, i) for kind in ("wrong", "fix") for k, i, _, f, r in moves["audio", kind]]
    tests = (("low mark", lambda f, r, w, i: f["low"]), ("no low mark", lambda f, r, w, i: not f["low"]),
             ("new track in the cell", lambda f, r, w, i: not r["outside"]),
             ("new track outside the cell", lambda f, r, w, i: r["outside"]),
             ("  outside and short (F1b gates)", lambda f, r, w, i: r["outside"] and f["short"]),
             ("  outside, not short (F1b lets through)", lambda f, r, w, i: r["outside"] and not f["short"]),
             ("short row", lambda f, r, w, i: f["short"]), ("not short", lambda f, r, w, i: not f["short"]),
             ("simultaneous overlap", lambda f, r, w, i: f["sim_ovl"]),
             ("no simultaneous overlap", lambda f, r, w, i: not f["sim_ovl"]))
    for label, test in tests:
        hit = [kind for kind, f, r, w, i in every if test(f, r, w, i)]
        fix, wrong = hit.count("fix"), hit.count("wrong")
        print(f"    {label:40} fix {fix:4}  wrong {wrong:4}  right {pct(fix, fix + wrong)}")


def mai2_segments(ident):
    path = R.MAI2 / f"{ident}-mix.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["segments"], payload.get("offset") or 0.0


def pct(n, d):
    return f"{100 * n / d:5.1f}% ({n}/{d})" if d else "   -   (0/0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.2, help="coverage share that counts a person as speaking in a row")
    ap.add_argument("--rows", help="write every wrong move and fix with its features to this CSV")
    args = ap.parse_args()
    out_rows = Path(args.rows).resolve() if args.rows else None
    wins, results = R.load(None)
    feats = {key: Rows(wins[key], args.thr, mai2_segments(key.rsplit("-c", 1)[0])) for key in WINDOWS if key in wins}

    # base rows
    base = {key: {i: f.features(i) for i in f.win.scored} for key, f in feats.items()}
    a_right = {key: {i for i in f.win.scored if f.win.right(f.win.acoustic, f.win.amap, i)} for key, f in feats.items()}

    print(f"THR {args.thr}: a person speaks in a row when their tracks cover at least that share of it")
    print("=== all scored rows (one person, main), per window: overlapped by diar / by MAI2")
    for key, rows in base.items():
        n = len(rows)
        print(f"  {key}: {n} rows, acoustic right {len(a_right[key])}; diar overlap {pct(sum(r['diar_ovl'] for r in rows.values()), n)}"
              f"  MAI2 overlap {pct(sum(bool(r['mai2_ovl']) for r in rows.values()), n)}")

    # moves per pool
    moves = defaultdict(list)          # (pool, kind) -> [(key, i, name, feats)]
    runs_per_pool = defaultdict(lambda: defaultdict(set))  # pool -> key -> run names that read it
    for (name, key), (rows, _, _) in results.items():
        pool = pool_of(name)
        if pool is None or key not in feats:
            continue
        runs_per_pool[pool][key].add(name)
        for i, r in rows.items():
            if r["kind"] in ("wrong", "fix"):
                moves[pool, r["kind"]].append((key, i, name, feats[key].features(i, r["new"]), r))

    for pool in ("audio", "text"):
        print(f"\n##### {pool} pool: {sum(len(v) for v in runs_per_pool[pool].values())} run-windows "
              f"({', '.join(f'{k} {len(v)}' for k, v in runs_per_pool[pool].items())})")
        # risk: per run-row
        for source in ("sim_ovl", "diar_ovl", "mai2_ovl"):
            tally = Counter()
            for key, names in runs_per_pool[pool].items():
                for name in names:
                    rows = results[name, key][0]
                    for i in feats[key].win.scored:
                        ovl = bool(base[key][i][source])
                        right = i in a_right[key]
                        kind = rows.get(i, {}).get("kind")
                        tally["right", ovl] += right
                        tally["wrong", ovl] += right and kind == "wrong"
                        tally["awrong", ovl] += not right
                        tally["fix", ovl] += (not right) and kind == "fix"
            print(f"  [{source}] moved wrong, of acoustic-right rows: overlapped {pct(tally['wrong', True], tally['right', True])}"
                  f"  not {pct(tally['wrong', False], tally['right', False])}")
            print(f"  [{source}] fixed, of acoustic-wrong rows:       overlapped {pct(tally['fix', True], tally['awrong', True])}"
                  f"  not {pct(tally['fix', False], tally['awrong', False])}")

        # feature shares
        names = ("sim_ovl", "diar_ovl", "mai2_ovl", "short", "filler", "low", "tracks2", "turn_edge",
                 "target_audible", "target_in_sim", "target_adjacent", "cross_sex")
        print(f"  feature share         {'wrong (inst)':>20} {'wrong (uniq)':>20} {'fix (inst)':>20} {'fix (uniq)':>20}"
              f" {'a-right rows':>20} {'a-wrong rows':>20}")
        groups = {}
        for kind in ("wrong", "fix"):
            items = moves[pool, kind]
            uniq = {(k, i): f for k, i, _, f, _ in items}
            groups[kind] = ([f for *_, f, _ in items], list(uniq.values()))
        pools_base = {
            "a-right": [base[k][i] for k in base for i in a_right[k]],
            "a-wrong": [base[k][i] for k in base for i in base[k] if i not in a_right[k]],
        }
        for feat in names:
            cells = []
            for kind in ("wrong", "fix"):
                for lst in groups[kind]:
                    cells.append(pct(sum(bool(f[feat]) for f in lst), len(lst)))
            for lst in pools_base.values():
                cells.append(pct(sum(bool(f[feat]) for f in lst), len(lst)) if feat in lst[0] else "-")
            print(f"  {feat:20} " + " ".join(f"{c:>20}" for c in cells))
        for kind in ("wrong", "fix"):
            secs = sorted(f["seconds"] for f in groups[kind][1])
            if secs:
                print(f"  {kind} (uniq) seconds median {secs[len(secs) // 2]:.2f}")
        secs = sorted(f["seconds"] for f in pools_base["a-right"])
        print(f"  a-right rows seconds median {secs[len(secs) // 2]:.2f}")

        # the overlap x target cross-tab for wrong moves
        for kind in ("wrong", "fix"):
            c = Counter((f["sim_ovl"], f["target_in_sim"]) for f in groups[kind][1])
            print(f"  {kind} (uniq) simultaneous overlap x target among the overlapping: "
                  + "  ".join(f"ovl={o!s:5} aud={a!s:5} {c[o, a]}" for o in (True, False) for a in (True, False)))

    breakdown(moves, feats)

    if out_rows:
        with open(out_rows, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            cols = ("seconds", "chars", "short", "filler", "sim_ovl", "diar_ovl", "mai2_ovl", "low", "tracks2",
                    "turn_edge", "target_audible", "target_in_sim", "target_adjacent", "cross_sex")
            w.writerow(["pool", "kind", "window", "row", "run", *cols, "outside_cell", "new_track_cov", "truth", "new_person", "new_label",
                        "coverage", "text", "why"])
            for (pool, kind), items in sorted(moves.items()):
                for key, i, name, f, r in items:
                    win = feats[key].win
                    w.writerow([pool, kind, key, win.ids[i], name,
                                *[round(f[c], 2) if isinstance(f[c], float) else f[c] for c in cols],
                                int(r["outside"]), new_track_cov(win, i, r["new_label"]),
                                win.truth[i].person, r["new"], r["new_label"], json.dumps(dict(feats[key].people_cov(i)), ensure_ascii=False),
                                win.truth[i].text, r["why"]])


if __name__ == "__main__":
    main()
