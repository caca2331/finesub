"""Segment-edge timing by text anchoring (report `docs/reports/2026-09-26-ts-diar-models.md`, 补充（2026-10-13）; closed plan `git show bff4b478:docs/plans/asr-recall-timing-plan.md`, 时间戳 T0 / T1).

`timing.py` pairs edges by time alone, which flatters dense arms. Here each segment of a route is anchored on
its own text: its first ANCHOR characters (as hiragana) are looked up among the reference's word characters
within SEARCH s of the segment's start, the nearest exact occurrence wins, and the reference's time is the
start of the word holding the first character; the end likewise with the last characters and the word's end.
Segments whose head or tail is not found are left out and counted (`found`).

References: the human word timing of the gold clip (`timing.FIXED`), production's words, MAI2's words.
T0 checks that production's words stand in for the human ones: the same route's errors against both must agree.

    python -m tools.ts_diar_explore.seg_timing t0
    python -m tools.ts_diar_explore.seg_timing t1 [--items hina-25 ...]
"""
from __future__ import annotations

import argparse

import numpy as np

from . import recall as R
from . import timing as T

ANCHOR = 4
SEARCH = 10.0
COARSE = 2.0
PAUSE = 0.3


def ref_chars(words) -> tuple[str, np.ndarray, np.ndarray]:
    """The words' text as hiragana; for every character its word's start and end."""
    text, starts, ends = [], [], []
    for w in sorted(words, key=lambda w: w["start"]):
        h = R.hira(w.get("text", w.get("word", "")))
        text.append(h)
        starts += [w["start"]] * len(h)
        ends += [w["end"]] * len(h)
    return "".join(text), np.array(starts), np.array(ends)


def _find(text: str, needle: str, times: np.ndarray, at: float) -> int | None:
    best, k = None, text.find(needle)
    while k >= 0:
        if abs(times[k] - at) <= SEARCH and (best is None or abs(times[k] - at) < abs(times[best] - at)):
            best = k
        k = text.find(needle, k + 1)
    return best


def _walk(starts, ends, k: int, step: int) -> int:
    """From character k, back (step -1) or on (+1) through the reference's words while the gap between words stays
    under PAUSE: the opening (closing) of the phrase the anchor sits in."""
    while 0 <= k + step < len(starts):
        gap = starts[k] - ends[k + step] if step < 0 else starts[k + step] - ends[k]
        if gap >= PAUSE:
            break
        k += step
    return k


def anchored(segments, ref) -> dict:
    """Signed start and end errors (route − reference) of the segments that anchor within ±SEARCH s; + = route late.
    `on` / `off` against the anchored character's word, `on_walk` / `off_walk` against the phrase it opens or closes.
    Only the walked median means anything: a route's segment often opens mid-phrase (no pause before it, e.g. at a
    change of speaker), where walking back overshoots by seconds; the median's shift tells whether the route leaves
    out leading fillers (it would look early against the anchored word). Errors beyond ±COARSE are coarse: counted
    in `coarse`, left out of the arrays."""
    text, starts, ends = ref
    out = {k: [] for k in ("on", "off", "on_walk", "off_walk")}
    n, coarse = 0, 0
    for s in segments:
        h = R.hira(s.get("text", ""))
        if len(h) < ANCHOR:
            continue
        n += 1
        i = _find(text, h[:ANCHOR], starts, s["start"])
        j = _find(text, h[-ANCHOR:], ends, s["end"])
        for k, edge, key, ref_t in ((i, s["start"], "on", starts), (j, s["end"], "off", ends)):
            if k is None:
                continue
            k = k if key == "on" else k + ANCHOR - 1
            d = edge - ref_t[k]
            if abs(d) > COARSE:
                coarse += 1
                continue
            out[key].append(d)
            w = _walk(starts, ends, k, -1 if key == "on" else 1)
            out[f"{key}_walk"].append(edge - ref_t[w])
    return {"n": n, "coarse": coarse, **{k: np.array(v) for k, v in out.items()}}


EVENT_GAP = 1.0


def events(units) -> tuple[list[float], list[float]]:
    """The reference's phrase onsets with >= EVENT_GAP of silence before, offsets with as much after."""
    u = sorted(units, key=lambda x: x["start"])
    on = [x["start"] for i, x in enumerate(u) if i == 0 or x["start"] - max(y["end"] for y in u[max(0, i - 20):i]) >= EVENT_GAP]
    off = [x["end"] for i, x in enumerate(u) if i + 1 == len(u) or u[i + 1]["start"] - x["end"] >= EVENT_GAP]
    return on, off


def common(route_units, ref_units, segments: bool = False) -> dict:
    """Primary T1: the reference's events, each paired one-to-one with the route's phrase edges within
    `timing.MAX_D` (`timing.stats`); signed errors route − reference. A segment route (`segments`, MOSS Pro) offers
    every segment edge: each is an edge it reports, and the neighbour is mostly someone else speaking, so the
    >= PHRASE_GAP rule meant for word timing would drop real ones."""
    on, off = events(ref_units)
    h_on, h_off = T.edges([x for x in route_units if R.S.norm(x.get("text", ""))], phrase=not segments)
    return {"on": T.stats(on, h_on), "off": T.stats(off, h_off)}


def show_common(name: str, r: dict) -> None:
    for k, label in (("on", "start"), ("off", "end")):
        s = r[k]
        print(f"  {name:28s} {label:5s} {s['signed']:+.3f} | {s['p50']:.3f} / {s['p90']:.3f} | "
              f"<=0.3 {s['w03']:5.1%}  unmatched {s['unmatched']:5.1%} of {s['n']}")


def summary(d: np.ndarray) -> str:
    if not len(d):
        return "—"
    a = np.abs(d)
    return (f"{np.median(d):+.3f} | {np.median(a):.3f} / {np.quantile(a, .9):.3f} | "
            f">0.3 {(a > 0.3).mean():5.1%} >0.5 {(a > 0.5).mean():5.1%}")


def show(name: str, r: dict) -> None:
    print(f"  {name:28s} start {summary(r['on'])}  (found {len(r['on'])}/{r['n']}, coarse > {COARSE} s in all {r['coarse']})")
    print(f"  {'':28s} start, phrase opening: median {np.median(r['on_walk']):+.3f}" if len(r['on_walk']) else "")
    print(f"  {'':28s} end   {summary(r['off'])}  (found {len(r['off'])}/{r['n']})")
    print(f"  {'':28s} end, phrase closing: median {np.median(r['off_walk']):+.3f}" if len(r['off_walk']) else "")


def t0() -> None:
    item = "gold-BV1cqLR6hEp3"
    human = ref_chars(T.srt(T.FIXED))
    prod = ref_chars(R.source(item, "prod")["units"])
    words = T.srt(T.FIXED)
    print(f"{item}, common events (reference phrase edges with >= {EVENT_GAP} s silence): signed | p50 / p90")
    for route in ("mosspro", "mai2", "g35t", "prod"):
        src = R.source(item, route)
        if src is None:
            continue
        units = src["lines"] if route == "mosspro" else src["units"]
        show_common(f"{route} vs human", common(units, words, route == "mosspro"))
        if route != "prod":
            show_common(f"{route} vs production", common(units, R.source(item, "prod")["units"], route == "mosspro"))
    print(f"\n{item}, text anchoring: signed median | |d| p50 / p90 | shares; + = route late")
    for route in ("mosspro", "mai2", "g35t", "prod"):
        src = R.source(item, route)
        if src is None:
            continue
        segs = src["lines"]
        show(f"{route} vs human", anchored(segs, human))
        if route != "prod":
            show(f"{route} vs production", anchored(segs, prod))


def t1(items) -> None:
    print(f"common events (production's phrase edges with >= {EVENT_GAP} s silence): signed | p50 / p90; + = route late")
    for item in items:
        prod = R.source(item, "prod")["units"]
        print(item)
        for route in ("mosspro", "mai2", "g35t"):
            src = R.source(item, route)
            if src is None:
                continue
            show_common(f"{route} vs production", common(src["lines"] if route == "mosspro" else src["units"], prod, route == "mosspro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["t0", "t1"])
    ap.add_argument("--items", nargs="*", default=list(R.ITEMS))
    args = ap.parse_args()
    R.ready()
    t0() if args.cmd == "t0" else t1(args.items)


if __name__ == "__main__":
    main()
