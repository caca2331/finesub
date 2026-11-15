"""ASR recall of the integrated models next to production (report `docs/reports/2026-09-26-ts-diar-models.md`, 补充（2026-10-13）; closed plan `git show bff4b478:docs/plans/asr-recall-timing-plan.md`, 召回).

R1 finds candidate misses mechanically, per route, against every other source of text on the same audio:

  whole    the route has no text within ±PAD of an interval (>= MIN_RUN) where another source has some;
           word-timed sources count only their words' time, MOSS Pro (segments only) its segments', the fansub's
           lines shrunk by REF_SHRINK at both ends (their edges carry lead-in)
  partial  the route has text there, but aligning its characters with another source's over the same time
           leaves >= MIN_CHARS of the other source's characters with nothing (or less than half as much) opposite;
           both sides are compared as hiragana (pykakasi) so kanji / kana spellings of the same word line up
  short    a word-timed source's stand-alone phrase of <= SHORT characters (silence >= PHRASE_GAP either side) that
           the route does not write within ±SHORT_NEAR s: a word-timed route has no such characters there, a
           segment route no short segment (<= 2 x SHORT characters) holding them -- a long segment would match
           うん or はい by chance, and MOSS writes another person's interjection as a segment of its own
           (MOSS Pro's stand-alone short segments are a source too, checked against the word-timed routes)
  long     the route is silent for >= LONG_GAP where the other sources write >= LONG_CHARS characters, any route,
           production included (judged at a few points, see the plan); candidates inside go there, not to the items

Candidates of all routes within JOIN of each other are merged into one item, judged once for every route (R2),
up to MAX_ITEM long. Character times inside a unit are spread evenly over it.

    python -m tools.ts_diar_explore.recall candidates [--items hina-25 ...]   # R1, writes WORK (tmp/asr-recall-timing/)
    python -m tools.ts_diar_explore.recall fansub                             # C: each route against the fansub
"""
from __future__ import annotations

import argparse
import difflib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import arms as A
from . import score as S
from .common import MAIN, frozen_inputs

#: the candidates and the judge's batches and answers the report's readouts come from, frozen 2026-10-13 (hashes in
#: recall_frozen.sha256; data/ is never cleaned); new candidates, runs and clips go to WORK, never here
FROZEN = MAIN / "data" / "ts-diar" / "recall-timing"
WORK = MAIN / "tmp" / "asr-recall-timing"


def ready():
    """Before reading anything: production's stables and its ids.json, the inputs outside data/, are the frozen ones."""
    frozen_inputs(Path(__file__).with_name("recall_frozen.sha256"), FROZEN)
ITEMS = ("hina-14", "hina-25", "hina-27", "hina-67", "hina-70", "hina-71", "hina-72", "hina-73",
         "onetonton-01", "onetonton-05")
ROUTES = {"prod": "prod", "mosspro": "mosspro", "mai2": "mai2-diar", "g35t": "g35t"}

STEP = 0.02
PAD = 0.3
MIN_RUN = 0.3
JOIN = 0.5
REF_SHRINK = 0.3
MIN_CHARS = 3
CHUNK = 20.0
MARGIN = 5.0
LONG_GAP = 10.0
SHORT = 4
SHORT_NEAR = 1.0
PHRASE_GAP = 0.3
MAX_ITEM = 15.0
WORD_TIMED = ("prod", "mai2", "g35t")
LONG_CHARS = 5
GROUP = 1.0


@lru_cache(maxsize=None)
def _kakasi():
    import pykakasi
    return pykakasi.kakasi()


def hira(text: str) -> str:
    return S.norm("".join(x["hira"] for x in _kakasi().convert(S.norm(text))))


def _stable(item: str) -> Path:
    if item.startswith("onetonton-"):
        return MAIN / "out" / item / f"{item}-stable.json"
    return A.production_stable(item)


def source(item: str, name: str) -> dict | None:
    """{"units": [{start, end, text}], "spans": speech intervals the source claims} or None when absent.
    Units are words where the source times them, else its segments."""
    if name == "prod":
        stable = json.loads(_stable(item).read_text(encoding="utf-8"))["segments"]
        units = [{"start": w["start"], "end": w["end"], "text": w["word"]} for s in stable for w in s.get("words", [])]
        lines = [{"start": s["start"], "end": s["end"], "text": s.get("text", "")} for s in stable]
        return {"units": units, "spans": units, "lines": lines}
    if name == "ref":
        if not item.startswith("hina-"):
            return None
        lines = S.reference(item)  # text only where the fansub keeps Japanese lines; else timing support alone
        spans = [{"start": x["start"] + REF_SHRINK, "end": x["end"] - REF_SHRINK} for x in lines
                 if x["end"] - x["start"] > 2 * REF_SHRINK]
        return {"units": [x for x in lines if x["text"]], "spans": spans, "lines": lines}
    arm = ROUTES[name]
    if not A.exists(item, arm):
        return None
    out = A.load(item, arm)
    units = out["words"] if out.get("words") else out["segments"]
    units = [u for u in units if S.norm(u.get("text", ""))]
    return {"units": units, "spans": units, "lines": out["segments"]}


def sources(item: str) -> dict[str, dict]:
    return {n: s for n in (*ROUTES, "ref") if (s := source(item, n)) is not None}


def _mask(spans, n: int, pad: float = 0.0) -> np.ndarray:
    m = np.zeros(n, bool)
    for u in spans:
        a, b = int(max(0.0, u["start"] - pad) / STEP), int(np.ceil((u["end"] + pad) / STEP))
        if b > a:
            m[a:min(b, n)] = True
    return m


def _runs(m: np.ndarray, min_len: float) -> list[tuple[float, float]]:
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) * STEP >= min_len:
                out.append((i * STEP, j * STEP))
            i = j
        else:
            i += 1
    return out


def _join(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def chars(units) -> tuple[str, np.ndarray]:
    """The units' text as hiragana, each character with a time. Units closer than GROUP are read as one text, so
    a compound split into word units keeps its reading (原神 is はらかみ… only when read whole); a source
    character's time is spread evenly over its unit, a reading's over the source characters it came from."""
    text, times = [], []
    group: list[tuple[str, float]] = []

    def flush():
        if not group:
            return
        orig = "".join(ch for ch, _ in group)
        at = 0
        for tok in _kakasi().convert(orig):
            n = len(tok["orig"])
            src = [tm for _, tm in group[at:at + n]] or [group[min(at, len(group) - 1)][1]]
            at += n
            h = S.norm(tok["hira"])
            for k, ch in enumerate(h):
                text.append(ch)
                times.append(src[min(len(src) - 1, k * len(src) // len(h))])
        group.clear()

    last_end = None
    for u in sorted(units, key=lambda u: u["start"]):
        s = S.norm(u.get("text", ""))
        if not s:
            continue
        if last_end is not None and u["start"] - last_end > GROUP:
            flush()
        step = (u["end"] - u["start"]) / len(s)
        group += [(ch, u["start"] + step * (k + 0.5)) for k, ch in enumerate(s)]
        last_end = u["end"]
    flush()
    return "".join(text), np.array(times)


def whole(src: dict[str, dict], route: str, n: int) -> list[tuple[float, float]]:
    others = np.zeros(n, bool)
    for name, s in src.items():
        if name != route:
            others |= _mask(s["spans"], n)
    return _join(_runs(others & ~_mask(src[route]["spans"], n, PAD), MIN_RUN), JOIN)


def partial(src: dict[str, dict], route: str, duration: float) -> list[tuple[float, float, str]]:
    """Runs of another source's characters with nothing opposite in the route's text, where the route has text."""
    rt, rtimes = chars(src[route]["units"])
    covered = _mask(src[route]["spans"], int(duration / STEP) + 1, PAD)
    out = []
    for name, s in src.items():
        if name == route:
            continue
        st, stimes = chars(s["units"])
        for c0 in np.arange(0.0, duration, CHUNK):
            si = np.flatnonzero((stimes >= c0) & (stimes < c0 + CHUNK))
            if len(si) < MIN_CHARS:
                continue
            ri = np.flatnonzero((rtimes >= c0 - MARGIN) & (rtimes < c0 + CHUNK + MARGIN))
            a_txt = st[si[0]:si[-1] + 1]
            b_txt = rt[ri[0]:ri[-1] + 1] if len(ri) else ""
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_txt, b_txt, autojunk=False).get_opcodes():
                if tag not in ("delete", "replace") or i2 - i1 < MIN_CHARS or (tag == "replace" and j2 - j1 >= (i2 - i1) / 2):
                    continue
                t = stimes[si[0] + i1:si[0] + i2]
                a, b = float(t.min()), float(t.max())
                mid = int((a + b) / 2 / STEP)
                if mid < len(covered) and covered[mid]:  # where the route is silent, `whole` has it
                    out.append((a, b, name))
    return out


def phrases(units) -> list[dict]:
    """Word units grouped at silences >= PHRASE_GAP."""
    out = []
    for u in sorted(units, key=lambda u: u["start"]):
        if out and u["start"] - out[-1]["end"] < PHRASE_GAP:
            out[-1]["end"] = max(out[-1]["end"], u["end"])
            out[-1]["text"] += u.get("text", "")
        else:
            out.append({"start": u["start"], "end": u["end"], "text": u.get("text", "")})
    return out


def short(src: dict[str, dict], route: str) -> list[tuple[float, float]]:
    out = []
    mine = sorted(src[route]["units"], key=lambda u: u["start"])
    starts = np.array([u["start"] for u in mine])
    for name in (*WORD_TIMED, "mosspro"):
        if name == route or name not in src or (name == "mosspro" and route not in WORD_TIMED):
            continue
        for ph in (phrases(src[name]["units"]) if name in WORD_TIMED else src[name]["units"]):
            h = hira(ph["text"])
            if not h or len(h) > SHORT:
                continue
            lo, hi = ph["start"] - SHORT_NEAR, ph["end"] + SHORT_NEAR
            near = [u for u in mine[max(0, int(np.searchsorted(starts, lo - 60))):] if u["start"] < hi and u["end"] > lo]
            if route in WORD_TIMED:
                found = h in hira("".join(u.get("text", "") for u in near))
            else:
                found = any(h in (g := hira(u.get("text", ""))) and len(g) <= 2 * SHORT for u in near)
            if not found:
                out.append((ph["start"], ph["end"]))
    return out


def text_in(units, a: float, b: float) -> str:
    return "".join(u.get("text", "") for u in sorted(units, key=lambda u: u["start"]) if u["end"] > a and u["start"] < b)


def long_gaps(src: dict[str, dict], route: str, n: int) -> list[dict]:
    """Stretches of >= LONG_GAP where the route writes nothing and the other sources write >= LONG_CHARS characters;
    `dense` when production writes >= 1 character a second there (the case MOSS Pro's silent drops showed)."""
    out = []
    for a, b in _runs(~_mask(src[route]["spans"], n), LONG_GAP):
        others = {name: len(hira(text_in(s["units"], a, b))) for name, s in src.items() if name != route}
        if sum(others.values()) >= LONG_CHARS:
            out.append({"start": a, "end": b, "chars": others,
                        "dense": route != "prod" and others.get("prod", 0) >= b - a})
    return out


def candidates(item: str, src: dict | None = None) -> dict:
    """R1 over the item's sources (or `src`, an edited copy of them)."""
    src = src or sources(item)
    duration = max(u["end"] for s in src.values() for u in s["units"])
    n = int(duration / STEP) + 1
    found, gaps = [], {}
    for route in ROUTES:
        if route not in src:
            continue
        gaps[route] = long_gaps(src, route, n)
        inside = lambda a, b: any(g["start"] <= a and b <= g["end"] for g in gaps[route])  # noqa: E731
        found += [(a, b, route, kind) for kind, spans in (
            ("whole", whole(src, route, n)), ("partial", [(a, b) for a, b, _ in partial(src, route, duration)]),
            ("short", short(src, route))) for a, b in spans if not inside(a, b)]
    items = []
    for a, b, route, kind in sorted(found):
        if items and a - items[-1]["end"] <= JOIN and max(b, items[-1]["end"]) - items[-1]["start"] <= MAX_ITEM:
            it = items[-1]
            it["end"] = max(it["end"], b)
        else:
            it = {"start": a, "end": b, "flags": {}}
            items.append(it)
        it["flags"].setdefault(route, set()).add(kind)
    for it in items:
        it["flags"] = {r: sorted(k) for r, k in it["flags"].items()}
        it["text"] = {name: text_in(s["lines"], it["start"] - 0.5, it["end"] + 0.5) for name, s in src.items()}
    return {"item": item, "duration": duration, "routes": [r for r in ROUTES if r in src], "items": items, "long": gaps}


FANSUB_JA = ("hina-67", "hina-70", "hina-71", "hina-72", "hina-73")


def fansub_recall(item: str) -> dict:
    """C: share of the fansub's Japanese characters each route matches (both as hiragana, CHUNK-s blocks of the
    fansub against the route's text ±MARGIN), and the fansub runs >= MIN_CHARS the route leaves unmatched (also per
    block, `blocks`, for intervals that resample blocks)."""
    src = sources(item)
    ft, ftimes = chars(src["ref"]["units"])
    out = {}
    for route in ROUTES:
        if route not in src:
            continue
        rt, rtimes = chars(src[route]["units"])
        hit, runs, run_s = 0, 0, 0.0
        blocks = []
        for c0 in np.arange(0.0, float(ftimes.max()) + CHUNK, CHUNK):
            blocks.append(runs)
            si = np.flatnonzero((ftimes >= c0) & (ftimes < c0 + CHUNK))
            if not len(si):
                continue
            ri = np.flatnonzero((rtimes >= c0 - MARGIN) & (rtimes < c0 + CHUNK + MARGIN))
            a_txt = ft[si[0]:si[-1] + 1]
            b_txt = rt[ri[0]:ri[-1] + 1] if len(ri) else ""
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_txt, b_txt, autojunk=False).get_opcodes():
                if tag == "equal":
                    hit += i2 - i1
                elif i2 - i1 >= MIN_CHARS and (tag == "delete" or j2 - j1 < (i2 - i1) / 2):
                    runs += 1
                    t = ftimes[si[0] + i1:si[0] + i2]
                    run_s += float(t.max() - t.min())
        ends = blocks[1:] + [runs]
        out[route] = {"recall": hit / len(ft), "runs": runs, "run_s": run_s,
                      "blocks": np.array(ends) - np.array(blocks)}
    return {"chars": len(ft), "minutes": float(ftimes.max()) / 60, "routes": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "fansub"])
    ap.add_argument("--items", nargs="*", default=list(ITEMS))
    args = ap.parse_args()
    ready()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.cmd == "fansub":
        for item in [i for i in args.items if i in FANSUB_JA]:
            got = fansub_recall(item)
            print(f"{item}: fansub {got['chars']} chars, {got['minutes']:.0f} min; "
                  + "; ".join(f"{r} {v['recall']:.1%} ({v['runs'] / got['minutes']:.2f} runs/min, {v['run_s'] / got['minutes']:.1f} s/min)"
                              for r, v in got["routes"].items()))
        return
    for item in args.items:
        got = candidates(item)
        (WORK / f"{item}-candidates.json").write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
        minutes = got["duration"] / 60
        clip = sum(it["end"] - it["start"] + 2 for it in got["items"])
        per = {r: {k: sum(1 for it in got["items"] if k in it["flags"].get(r, ())) for k in ("whole", "partial", "short")}
               for r in got["routes"]}
        print(f"{item}: {minutes:.0f} min, {len(got['items'])} items, clip {clip:.0f} s (~{clip * 32 / 1000:.0f}k audio tokens); "
              + "; ".join(f"{r} {v['whole']}w/{v['partial']}p/{v['short']}s" for r, v in per.items())
              + "; long gaps " + ", ".join(f"{r} {len(v)} ({sum(g['end'] - g['start'] for g in v):.0f} s)" for r, v in got["long"].items() if v))


if __name__ == "__main__":
    main()
