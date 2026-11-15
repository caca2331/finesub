"""Acoustic row splitting readouts (docs/diarization-findings.md 6.5).

Two rulers, frozen inputs: (1) the row truth of the speaker judge's acceptance windows (`row_truth.py`), (2) the hina
Japanese fansub text truth (`edit_judge_truth.py`). A one-person row's tier comes from two LLM annotators' labels in
`data/diarization/row-split/labels/` (MERGE / FILLER -> minor, KEEP / UNSURE -> severe; KEEP if either says KEEP or UNSURE). The labelling
prompt is `row_split_label_task.md`; a variant that proposes a one-person cut nobody labelled prints the count, and
`candidates` writes those cuts for the next labelling round.

    python tools/diarization_explore/row_split.py cache 1      # build a ruler's cache (one ruler per process:
    python tools/diarization_explore/row_split.py cache 2      #   the rulers' loaders cannot share a process)
    python tools/diarization_explore/row_split.py score [--grid]
    python tools/diarization_explore/row_split.py candidates   # unlabelled one-person cuts -> tmp/row-split/
    python tools/diarization_explore/row_split.py rebuild      # the plan's rebuild check on ruler 1's files
    python tools/diarization_explore/row_split.py parity       # the production module against the settled arm
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import row_split_lib as L  # noqa: E402

LABELS = L.MAIN / "data" / "diarization" / "row-split" / "labels"
CACHE = L.MAIN / "tmp" / "row-split"


def load(r):
    path = CACHE / f"ruler{r}.pkl"
    if not path.exists():
        sys.exit(f"no cache for ruler {r}: run `row_split.py cache {r}` first")
    return pickle.loads(path.read_bytes())


def build(r):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = L.ruler1() if r == "1" else L.ruler2()
    (CACHE / f"ruler{r}.pkl").write_bytes(pickle.dumps(out))
    print(f"ruler {r}: {len(out[0])} rows, {out[1] / 3600:.2f} h")


def _read(pattern):
    out = {}
    for path in sorted(glob.glob(str(LABELS / pattern))):
        out.update({json.loads(x)["id"]: json.loads(x)["label"] for x in open(path, encoding="utf-8") if x.strip()})
    return out


_A, _B = None, None


def label(key):
    global _A, _B
    if _A is None:
        _A, _B = _read("labels-A*.jsonl"), _read("labels-B*.jsonl")
    got = {_A.get(key), _B.get(key)} - {None}
    if not got:
        return None
    # UNSURE counts as KEEP: a cut nobody could call is not one correction is known to repair
    return "KEEP" if got & {"KEEP", "UNSURE"} else "FILLER" if "FILLER" in got else "MERGE"


def key_of(r, it, cut):
    return f"r{r}|{it['key']}|{it['i']}|{cut}"


def grade(r, it, cut, missing):
    if it["truth"]["kind"] == "single":
        lab = label(key_of(r, it, cut))
        if lab is None:
            missing.append(key_of(r, it, cut))
            return L.grade(it, cut)
        return ("severe", "single:keep") if lab == "KEEP" else ("minor", "single:" + lab.lower())
    return L.grade(it, cut)


def cut_of(it, src, post, T, C):
    p = L.propose(it["seg"], it[src], T, C)
    if not p:
        return None
    if post == "none":
        return p[0]
    if post == "place":
        return L.place(it["seg"], p, widen=0)
    if post == "gate2":
        return L.gate2(it["seg"], p[0], T, C)
    raise ValueError(post)


#: (name, source, post, CLEAN, DIRTY, T, C): the variants the plan's readout table reports
HEADLINE = (
    ("slots + sentence-end gate (round 1)", "slots", "gate2", 0.8, 0.2, 0.3, 4),
    ("slots, old floors", "slots", "none", 0.8, 0.2, 0.3, 4),
    ("slots, new floors", "slots", "none", 0.7, 0.2, 0.15, 4),
    ("slots, new floors + placement", "slots", "place", 0.7, 0.2, 0.15, 4),
    ("vbx tracks, new floors", "tracks", "none", 0.7, 0.2, 0.15, 4),
)
GRID = list(itertools.product(("none", "place"), (0.8, 0.7, 0.6), (0.2, 0.3), (0.15, 0.3), (2, 3, 4)))


def readout(data, src, post, clean, dirty, T, C):
    L.CLEAN, L.DIRTY = clean, dirty
    tot, missing, hours = Counter(), [], 0.0
    for r, (items, sec) in data.items():
        hours += sec / 3600
        for it in items:
            cut = cut_of(it, src, post, T, C)
            if not cut:
                continue
            tier = grade(r, it, cut, missing)[0]
            kind = {"single": "one", "split": "two"}.get(it["truth"]["kind"], "other")
            tot[(kind, tier)] += 1
    return tot, len(missing), hours


def fmt(t, missing, hours):
    return (f"two: exact {t[('two', 'exact')]:2d} minor {t[('two', 'minor')]:2d} severe {t[('two', 'severe')]} | "
            f"one: minor {t[('one', 'minor')]:3d} severe {t[('one', 'severe')]} | "
            f"{sum(t.values()) / hours:4.1f} cuts/h | unlabelled {missing}")


ACCEPT = L.MAIN / "tmp" / "diarization-retrack" / "accept"


def _row_key(seg):
    return (round(float(seg["start"]), 3), round(float(seg["end"]), 3),
            tuple(str(w.get("word", "")) for w in seg.get("words") or []))


def rebuild(idents):
    """The plan's rebuild check: (1) the production DP re-run on aligned reproduces aligned's rows; (2) aligned's rows
    against stable's, by words (the difference is stabilization's)."""
    from finesub.speech.postprocessing import segmentation as seg
    from finesub.speech.recognition import segments as segment_ops
    for ident in idents:
        folder = ACCEPT / ident
        aligned = json.loads((folder / f"{ident}-aligned.json").read_text(encoding="utf-8"))
        vad = json.loads((folder / f"{ident}-vad.json").read_text(encoding="utf-8"))
        stable = json.loads((folder / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]
        meta = aligned["metadata"]["asr_align"]["segment_split"]
        params = seg.split_params_for_length_scale(float(meta["length_scale"]))
        redone = seg.split_segments(seg.regroup_by_whisper_segments(aligned["segments"]), vad["segments"], params=params)
        redone, _ = segment_ops.normalize_segment_timeline(redone, audio_end=vad.get("audio_duration"))
        a_keys = [_row_key(s) for s in aligned["segments"]]
        r_keys = [_row_key(s) for s in redone]
        same_rows = sum(1 for x, y in zip(a_keys, r_keys) if x == y)
        a_words = [k[2] for k in a_keys]
        s_words = [_row_key(s)[2] for s in stable]
        common = len(set(a_words) & set(s_words))
        print(f"{ident}: aligned {len(a_keys)} rows, re-run {len(r_keys)} rows, identical {same_rows}; "
              f"stable {len(s_words)} rows, word-identical to an aligned row {common}")


RULER2_RUNS = L.MAIN / "out" / "refined-2609"
_STREAMS = {}


def stream_of(r, it):
    """The production DP's scores over this item's stable file (ruler 1: the acceptance run's VAD; ruler 2: P2's)."""
    key = (r, it["ident"])
    if key not in _STREAMS:
        folder = ACCEPT / it["ident"] if r == "1" else RULER2_RUNS / it["ident"]
        vad = json.loads((folder / f"{it['ident']}-vad.json").read_text(encoding="utf-8"))["segments"]
        aligned = folder / f"{it['ident']}-aligned.json"
        scale = 1.0
        if aligned.exists():
            meta = json.loads(aligned.read_text(encoding="utf-8")).get("metadata", {})
            scale = float(meta.get("asr_align", {}).get("segment_split", {}).get("length_scale", 1.0))
        _STREAMS[key] = L.Stream(it["stable"], vad, scale)
    return _STREAMS[key]


NEW_FLOORS = dict(clean=0.7, dirty=0.2, T=0.15, C=4)
#: the DP boundary-score gate: drops cuts on a bare boundary (B = 3.0); B only takes a few discrete values
GATE_B = 2.5


def phase1_cut(r, it, arm, lam=None):
    """Cut word index or None. Arms: base (slots, new floors), A (base + DP placement), B (slots with minimal floors,
    gate on the DP margin <= lam, cut by A), vbxdiff (base, halves on different vbx tracks)."""
    L.CLEAN, L.DIRTY = NEW_FLOORS["clean"], NEW_FLOORS["dirty"]
    if arm == "B":
        p = L.propose(it["seg"], it["slots"], 0.0, 1)
        if not p:
            return None
        st = stream_of(r, it)
        m = L.dp_margin(st, it["i"], p, len(it["seg"]["words"]))
        if m is None or m > lam:
            return None
        return L.dp_place(st, it["i"], p)
    p = L.propose(it["seg"], it["slots"], NEW_FLOORS["T"], NEW_FLOORS["C"])
    if not p:
        return None
    if arm == "base":
        return p[0]
    if arm == "A":
        return L.dp_place(stream_of(r, it), it["i"], p)
    if arm == "vbxdiff":
        a, b = L.halves_tracks(it["seg"], p[0], it["tracks"])
        return p[0] if a is not None and b is not None and a != b else None
    if arm in ("A+gates", "final"):
        st = stream_of(r, it)
        cut = L.dp_place(st, it["i"], p)
        if not cut:
            return None
        if arm == "final":
            cut = L.glue_left_word(st, it["i"], cut)
        if st.boundary(it["i"], cut)[0].b > GATE_B:
            return None
        # A+gates checks the halves at the placed cut (as in the first readout); final checks them at the acoustic cut,
        # because the vbx tracks lag like the slots do and the gate asks "two people in this row", not "where"
        a, b = L.halves_tracks(it["seg"], cut if arm == "A+gates" else p[0], it["tracks"])
        return cut if a is not None and b is not None and a != b else None
    raise ValueError(arm)


def tally(data, fn):
    tot, missing, halves = Counter(), [], Counter()
    for r, (items, _) in data.items():
        for it in items:
            cut = fn(r, it)
            if not cut:
                continue
            tier = grade(r, it, cut, missing)[0]
            kind = {"single": "one", "split": "two"}.get(it["truth"]["kind"], "other")
            tot[(kind, tier)] += 1
            if kind == "other":
                continue
            a, b = L.halves_tracks(it["seg"], cut, it["tracks"])
            same = "same" if a == b else "diff"
            if r == "1" and kind == "two" and tier != "severe":
                parts = it["raw"].get("parts") or []
                nm = it["names"]
                ok = len(parts) >= 2 and a in nm and b in nm and bool(nm[a] & set(parts[0]["who"])) and bool(nm[b] & set(parts[-1]["who"]))
                halves[("r1 two right-person", ok)] += 1
            halves[(f"r{r} {kind}", same)] += 1
    return tot, missing, halves


def fmt_halves(h):
    keys = sorted(h)
    return "; ".join(f"{k[0]} {k[1]}={h[k]}" for k in keys)


def phase1(data, lams):
    hours = sum(sec for _, sec in data.values()) / 3600
    rows = {}
    for arm in ("base", "A", "vbxdiff", "A+gates", "final"):
        t, miss, h = tally(data, lambda r, it, arm=arm: phase1_cut(r, it, arm))
        rows[arm] = (t, miss, h)
        print(f"{arm:8s} {fmt(t, len(miss), hours)}\n         halves: {fmt_halves(h)}")
    # lambda: on ruler 1 only, the largest with one-person minor <= 1.25 x the base's ruler-1 count
    r1 = {"1": data["1"]}
    n1 = tally(r1, lambda r, it: phase1_cut(r, it, "base"))[0][("one", "minor")]
    print(f"ruler-1 base one-person minor n1 = {n1}; cap {1.25 * n1:.1f}")
    best = None
    for lam in lams:
        t1 = tally(r1, lambda r, it, lam=lam: phase1_cut(r, it, "B", lam))[0]
        print(f"  lambda {lam:+.2f} ruler 1: {fmt(t1, 0, data['1'][1] / 3600)}")
        if t1[("one", "minor")] <= 1.25 * n1:
            best = lam
    if best is None:
        print("no lambda meets the cap")
        return
    r2 = {"2": data["2"]}
    t, miss, h = tally(r2, lambda r, it: phase1_cut(r, it, "B", best))
    print(f"B at lambda {best:+.2f}, read on ruler 2: {fmt(t, len(miss), data['2'][1] / 3600)}\n         halves: {fmt_halves(h)}")
    t, miss, h = tally(data, lambda r, it: phase1_cut(r, it, "B", best))
    print(f"B at lambda {best:+.2f}, both rulers: {fmt(t, len(miss), hours)}\n         halves: {fmt_halves(h)}")


_GA, _GB = None, None


def grammar(key):
    """BREAK / CLEAN for a one-person cut (BREAK when either annotator says BREAK or UNSURE), None when unlabelled."""
    global _GA, _GB
    if _GA is None:
        _GA, _GB = _read("grammar-A*.jsonl"), _read("grammar-B*.jsonl")
    got = {_GA.get(key), _GB.get(key)} - {None}
    if not got:
        return None
    return "BREAK" if got & {"BREAK", "UNSURE"} else "CLEAN"


def grammar_breaks(data, arm):
    """Cuts a reader sees broken when nothing merges them back: one-person cuts labelled BREAK, plus two-person cuts
    with any characters on the wrong side (every tier but exact). Returns (breaks, unlabelled, hours)."""
    breaks, missing, hours = Counter(), 0, 0.0
    for r, (items, sec) in data.items():
        hours += sec / 3600
        for it in items:
            cut = phase1_cut(r, it, arm)
            if not cut:
                continue
            kind = it["truth"]["kind"]
            if kind == "single":
                g = grammar(key_of(r, it, cut))
                if g is None:
                    missing += 1
                elif g == "BREAK":
                    breaks["one"] += 1
            elif kind == "split" and grade(r, it, cut, [])[0] != "exact":
                breaks["two"] += 1
    return breaks, missing, hours


def parity(data):
    """The production module (`finesub.speech.diarization.row_split`) against the settled arm, row by row: on each
    ruler fed the inputs the readout used, and on ruler 1 also end to end from the stable, timeline and evidence."""
    from finesub.speech.diarization import row_split as P
    from finesub.speech.diarization.evidence import ClusterEvidence
    from finesub.speech.diarization.segments import TrackSegment
    from finesub.speech.postprocessing import segmentation
    for r, (items, _) in data.items():
        by_file = {}
        for it in items:
            by_file.setdefault(it["ident"], []).append(it)
        same = differ = 0
        for ident, its in by_file.items():
            it0 = its[0]
            st = stream_of(r, it0)
            folder = ACCEPT / ident if r == "1" else RULER2_RUNS / ident
            intervals = json.loads((folder / f"{ident}-vad.json").read_text(encoding="utf-8"))["segments"]
            bounds = segmentation.row_boundaries(it0["stable"], intervals, params=st.params)
            seg = lambda xs: [TrackSegment(int(t), float(a), float(b)) for t, a, b in xs]  # noqa: E731
            got = {c.row: c.word for c in P.find_cuts(it0["stable"], slots=seg(it0["slots"]), tracks=seg(it0["tracks"]),
                                                      boundaries=bounds)}
            if r == "1":
                payload = json.loads(Path(L.MAIN / "data" / "diarization" / "row-truth" / "products" / "acoustic" / ident /
                                          f"{ident}-stable.json").read_text(encoding="utf-8"))
                diar = json.loads((folder / f"{ident}-diar.json").read_text(encoding="utf-8"))
                tracks = [TrackSegment(int(t["track"]), float(t["start"]), float(t["end"])) for t in diar["tracks"]]
                ev = ClusterEvidence.load(folder / f"{ident}-diar-evidence.npz")
                whole = {c.row: c.word for c in P.cuts_for(payload, tracks, ev)}
                rows = {it["i"] for it in its}
                if {k: v for k, v in whole.items() if k in rows} != {k: v for k, v in got.items() if k in rows}:
                    print(f"  r1 {ident}: end to end differs from the readout's inputs")
            for it in its:
                want = phase1_cut(r, it, "final")
                if (want or None) == got.get(it["i"]):
                    same += 1
                else:
                    differ += 1
                    print(f"  r{r} {ident} row {it['i']}: readout {want}, production {got.get(it['i'])}")
        print(f"ruler {r}: {same} rows agree, {differ} differ")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebuild")
    sub.add_parser("phase1")
    sub.add_parser("grammar")
    c = sub.add_parser("cache")
    c.add_argument("ruler", choices=("1", "2"))
    s = sub.add_parser("score")
    s.add_argument("--grid", action="store_true")
    sub.add_parser("candidates")
    sub.add_parser("parity")
    a = ap.parse_args()
    if a.cmd == "cache":
        return build(a.ruler)
    if a.cmd == "rebuild":
        return rebuild(sorted({it["ident"] for it in load("1")[0]}))
    data = {r: load(r) for r in ("1", "2")}
    two = sum(L.counts(items)[0] for items, _ in data.values())
    hours = sum(sec for _, sec in data.values()) / 3600
    if a.cmd == "parity":
        return parity(data)
    if a.cmd == "phase1":
        return phase1(data, [x / 4 for x in range(-12, 13)])
    if a.cmd == "grammar":
        for arm in ("base", "A", "vbxdiff", "A+gates", "final"):
            b, miss, h = grammar_breaks(data, arm)
            print(f"{arm:8s} grammar-breaking cuts: one-person {b['one']}, two-person {b['two']}, "
                  f"{(b['one'] + b['two']) / h:.2f}/h ({h:.1f} h) | unlabelled {miss}")
        return
    if a.cmd == "score":
        print(f"{hours:.1f} h, {two} two-person-in-turn rows")
        for name, src, post, clean, dirty, T, C in HEADLINE:
            print(f"{name:38s} {fmt(*readout(data, src, post, clean, dirty, T, C))}")
        if a.grid:
            for post, clean, dirty, T, C in GRID:
                print(f"slots clean {clean} dirty {dirty} T {T} C {C} {post:5s} {fmt(*readout(data, 'slots', post, clean, dirty, T, C))}")
        return
    # candidates: every unlabelled one-person cut of the grid and the headline variants
    new = {}
    variants = [(src, post, cl, d, T, C) for _, src, post, cl, d, T, C in HEADLINE] + \
               [("slots", post, cl, d, T, C) for post, cl, d, T, C in GRID]
    for src, post, clean, dirty, T, C in variants:
        L.CLEAN, L.DIRTY = clean, dirty
        for r, (items, _) in data.items():
            for it in items:
                if it["truth"]["kind"] != "single":
                    continue
                cut = cut_of(it, src, post, T, C)
                if not cut or label(key_of(r, it, cut)) is not None or key_of(r, it, cut) in new:
                    continue
                ws = [str(w.get("word", "")) for w in it["seg"]["words"]]
                st, i = it["stable"], it["i"]
                ctx = lambda j: str(st[j].get("text", "")) if 0 <= j < len(st) else ""  # noqa: E731
                new[key_of(r, it, cut)] = dict(id=key_of(r, it, cut), prev=ctx(i - 1), left="".join(ws[:cut]),
                                               right="".join(ws[cut:]), next=ctx(i + 1))
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "candidates.jsonl"
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in new.values()), encoding="utf-8")
    print(f"{len(new)} unlabelled cuts -> {path}")


if __name__ == "__main__":
    main()
