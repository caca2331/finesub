"""How often does correction merge lines of two different people? Offline, on existing correction outputs
(docs/reports/2026-10-17-three-pass.md). `python tools/diarization_explore/merge_audit.py [N]`: N > 0 also prints N examples.

For every output line of an `-annotated.csv` that merges several source lines, each seam between consecutive
sources is judged against ruler 1's row truth: the speaker ending the earlier line vs the one starting the later.
Seams touching a line without clean truth (overlap, background, other) are left out.

Usage: `merge_audit.py` (counts), `merge_audit.py N` (up to N examples), `merge_audit.py rules` (what a turn marker would block), `merge_audit.py compare <run folder>...` (runs side by side, by marker).
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
sys.path.insert(0, str(MAIN / "tools" / "diarization_explore"))
import row_split as R  # noqa: E402

items, _ = R.load("1")
truth = {(it["ident"], it["i"]): it["truth"] for it in items}
acoustic = {}


def rows_of(ident):
    if ident not in acoustic:
        p = MAIN / "data/diarization/row-truth/products/acoustic" / ident / f"{ident}-stable.json"
        segs = json.loads(p.read_text(encoding="utf-8"))["segments"]
        acoustic[ident] = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(segs)}
    return acoustic[ident]


def ends(t):
    """(speaker starting the line, speaker ending it), or None when the truth is not clean."""
    if t is None:
        return None
    if t["kind"] == "single":
        p = t["people"]
        return (p[0], p[-1]) if len(p) == 1 else None
    if t["kind"] == "split":
        return (t["people"][0], t["people"][-1])
    return None


IDS = {}


def positions(cell):
    out = []
    for part in str(cell).split(","):
        lo, _, hi = part.strip().partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return [IDS.get(n, n) for n in out]


def audit(name, ident, stable, annotated):
    segs = json.loads(Path(stable).read_text(encoding="utf-8"))["segments"]
    IDS.clear()
    IDS.update({int(s["id"]): k for k, s in enumerate(segs, start=1) if s.get("id") is not None})
    index = rows_of(ident)
    src = [truth.get((ident, index.get((round(s["start"], 3), round(s["end"], 3))))) for s in segs]
    with open(annotated, encoding="utf-8") as f:
        rows = [r for r in csv.reader((x for x in f if not x.startswith("#")), delimiter="|") if r and r[0] == "sub"]
    merged_with = {}
    for r in rows:
        pos = positions(r[1])
        for a, b in zip(pos, pos[1:]):
            merged_with[a] = b
    c = Counter()
    for n in range(1, len(segs)):
        ea, eb = ends(src[n - 1]), ends(src[n])
        if ea is None or eb is None:
            continue
        cross = ea[1] != eb[0]
        merged = merged_with.get(n) == n + 1
        c[("cross" if cross else "same", "merged" if merged else "kept")] += 1
    seconds = sum(float(s["end"]) - float(s["start"]) for s, t in zip(segs, src) if t is not None)
    return c, seconds


def runs():
    d6 = MAIN / "tmp/row-split/d6"
    for model, folder in (("flash-lite", "runs"), ("gemini 3.7 flash", "runs-gemini-free-3_7-flash"),
                          ("agy 3.8 Flash", "runs-local-agy-media-gemini-3_8-flash")):
        for run in sorted((d6 / folder / "orig").glob("*")):
            key = run.name
            csvs = list(run.glob("*-annotated.csv"))
            if csvs:
                yield f"D6 {model}", key.rsplit("-c", 1)[0], d6 / "orig" / key / f"{key}-stable.json", csvs[0]
    for run in sorted((MAIN / "tmp/diarization-retrack/r9/correction").glob("*")):
        csvs = list(run.glob("*-annotated.csv"))
        if csvs:
            yield "r9 gemini flash", run.name.rsplit("-", 1)[0], run / f"{run.name}-stable.json", csvs[0]
    for label, folder in (("accept GPT/Sol?", "accept-judge"), ("accept Luna", "accept-judge-luna")):
        run = MAIN / "out" / folder / "hina-05"
        if (run / "hina-05-annotated.csv").exists():
            yield f"full-file {label}", "hina-05", run / "hina-05-stable.json", run / "hina-05-annotated.csv"


total = defaultdict(Counter)
hours = defaultdict(float)
for name, ident, stable, annotated in runs():
    c, s = audit(name, ident, stable, annotated)
    total[name] += c
    hours[name] += s / 3600
    print(f"{name:24} {ident:12} cross merged {c['cross', 'merged']:3}/{c['cross', 'merged'] + c['cross', 'kept']:3}"
          f"  same merged {c['same', 'merged']:3}/{c['same', 'merged'] + c['same', 'kept']:4}")
print()
for name, c in total.items():
    cm, ck, sm, sk = c["cross", "merged"], c["cross", "kept"], c["same", "merged"], c["same", "kept"]
    print(f"{name:24} {hours[name]:.2f} h | cross-speaker seams merged {cm}/{cm + ck} ({cm / max(1, cm + ck):.1%}), "
          f"{cm / hours[name]:.1f}/h | same-speaker seams merged {sm}/{sm + sk} ({sm / max(1, sm + sk):.1%})")


def chars(text):
    import re
    return len(re.sub(r"[\s、。！？!?…ー〜~,.\-「」『』（）()]", "", text))


def examples(wanted, limit=400):
    out = Counter()
    shown = 0
    for name, ident, stable, annotated in runs():
        if name not in wanted:
            continue
        segs = json.loads(Path(stable).read_text(encoding="utf-8"))["segments"]
        IDS.clear()
        IDS.update({int(s["id"]): k for k, s in enumerate(segs, start=1) if s.get("id") is not None})
        index = rows_of(ident)
        src = [truth.get((ident, index.get((round(s["start"], 3), round(s["end"], 3))))) for s in segs]
        with open(annotated, encoding="utf-8") as f:
            rows = [r for r in csv.reader((x for x in f if not x.startswith("#")), delimiter="|") if r and r[0] == "sub"]
        for r in rows:
            pos = positions(r[1])
            for a, b in zip(pos, pos[1:]):
                if b > len(src):
                    continue
                ea, eb = ends(src[a - 1]), ends(src[b - 1])
                if ea is None or eb is None or ea[1] == eb[0]:
                    continue
                left, right = segs[a - 1]["text"], segs[b - 1]["text"]
                short = min(chars(left), chars(right))
                kind = "short side <=4 chars" if short <= 4 else "both sides >4 chars"
                out[name, kind] += 1
                if shown < limit and short > 4:
                    shown += 1
                    print(f"{name} {ident}: [{ea[1]}] {left}  ||  [{eb[0]}] {right}  ->  {r[4]}")
    print(out)


def rules():
    """Which merges a turn marker would block: seams by the acoustic main tracks of their two lines.

    `differ` is rule B (main tracks differ), `differ(?)` and `sure-differ` its two parts; rule A is `sure-differ` alone
    (neither side has a `low` mark or a second track)."""
    def tier(a, b):
        if a.get("track") is None or b.get("track") is None or a.get("block", 0) != b.get("block", 0):
            return "none"
        if a["track"] == b["track"]:
            return "agree"
        sure = lambda s: not s.get("review") and len(s.get("tracks") or [s["track"]]) == 1
        return "sure-differ" if sure(a) and sure(b) else "differ(?)"
    res = defaultdict(Counter)
    for name, ident, stable, annotated in runs():
        segs = json.loads(Path(stable).read_text(encoding="utf-8"))["segments"]
        IDS.clear()
        IDS.update({int(s["id"]): k for k, s in enumerate(segs, start=1) if s.get("id") is not None})
        index = rows_of(ident)
        path = MAIN / "data/diarization/row-truth/products/acoustic" / ident / f"{ident}-stable.json"
        acoustic_segs = json.loads(path.read_text(encoding="utf-8"))["segments"]
        keys = [index.get((round(s["start"], 3), round(s["end"], 3))) for s in segs]
        src = [truth.get((ident, k)) for k in keys]
        spk = [(acoustic_segs[k].get("speaker") or {}) if k is not None else {} for k in keys]
        with open(annotated, encoding="utf-8") as f:
            rows = [r for r in csv.reader((x for x in f if not x.startswith("#")), delimiter="|") if r and r[0] == "sub"]
        merged_with = {}
        for r in rows:
            pos = positions(r[1])
            for a, b in zip(pos, pos[1:]):
                merged_with[a] = b
        for n in range(1, len(segs)):
            ea, eb = ends(src[n - 1]), ends(src[n])
            if ea is None or eb is None:
                continue
            res[name][("cross" if ea[1] != eb[0] else "same", "merged" if merged_with.get(n) == n + 1 else "kept",
                       tier(spk[n - 1], spk[n]))] += 1
    for name, c in res.items():
        print(name)
        for t in ("cross", "same"):
            for m in ("merged", "kept"):
                print(f"   {t:5} {m:6}", dict(sorted((k[2], v) for k, v in c.items() if k[:2] == (t, m))))


def compare(folders):
    """Per window, for each run folder (`<folder>/<window key>/*-annotated.csv`, D6 windows): seams with clean truth
    by cross / same speaker x marked / unmarked (rule B: main tracks differ) x merged. Only windows every folder ran."""
    d6 = MAIN / "tmp/row-split/d6"
    folders = [Path(f) for f in folders]
    keys = sorted(set.intersection(*({p.name for p in f.iterdir() if list(p.glob("*-annotated.csv"))} for f in folders)))
    totals = [Counter() for _ in folders]
    for key in keys:
        ident = key.rsplit("-c", 1)[0]
        stable = d6 / "orig" / key / f"{key}-stable.json"
        segs = json.loads(stable.read_text(encoding="utf-8"))["segments"]
        index = rows_of(ident)
        src = [truth.get((ident, index.get((round(s["start"], 3), round(s["end"], 3))))) for s in segs]
        marked = [False] + [a.get("speaker", {}).get("track") is not None and b.get("speaker", {}).get("track") is not None
                            and a["speaker"].get("block", 0) == b["speaker"].get("block", 0)
                            and a["speaker"]["track"] != b["speaker"]["track"] for a, b in zip(segs, segs[1:])]
        line = [f"{key:14}"]
        for folder, total in zip(folders, totals):
            with open(next((folder / key).glob("*-annotated.csv")), encoding="utf-8") as f:
                rows = [r for r in csv.reader((x for x in f if not x.startswith("#")), delimiter="|") if r and r[0] == "sub"]
            merged_with = {}
            for r in rows:
                pos = [int(p) for p in r[1].split(",")]
                for a, b in zip(pos, pos[1:]):
                    merged_with[a] = b
            c = Counter()
            for n in range(1, len(segs)):
                ea, eb = ends(src[n - 1]), ends(src[n])
                if ea is None or eb is None:
                    continue
                c["cross" if ea[1] != eb[0] else "same", "marked" if marked[n] else "unmarked",
                  "merged" if merged_with.get(n) == n + 1 else "kept"] += 1
            total.update(c)
            line.append(" ".join(f"{t[0]}{m[0]}={c[t, m, 'merged']}" for t in ("cross", "same") for m in ("marked", "unmarked")))
        print(" | ".join(line))
    print("total          | " + " | ".join(
        " ".join(f"{t[0]}{m[0]}={c[t, m, 'merged']}" for t in ("cross", "same") for m in ("marked", "unmarked"))
        for c in totals))
    print("(cm = cross-speaker seams merged where marked, cu = unmarked, sm / su = same speaker; one column group per folder)")


if len(sys.argv) > 1 and sys.argv[1] == "rules":
    rules()
elif len(sys.argv) > 2 and sys.argv[1] == "compare":
    compare(sys.argv[2:])
elif len(sys.argv) > 1:
    examples({"D6 gemini 3.7 flash", "D6 agy 3.8 Flash", "r9 gemini flash"}, limit=int(sys.argv[1]))
