"""The rescue's pure decisions, on segment dicts (`{"start", "end", "speaker", "text"}`, seconds): no model, no GPU.

Why each rule exists and where its thresholds came from: `docs/joint-asr.md`「本地 MOSS 的救援」.
"""

from __future__ import annotations

import difflib
import itertools
import re
import unicodedata
from typing import Callable, Mapping, Sequence

import numpy as np

from ....subtitles.metrics import weighted_char_count
from ..mosspro import LATIN, NON_LATIN
from ..stitch import norm

#: A hole in the transcript this long (s), with GATE_VOICED s of Silero speech in it, is talk the pass skipped.
GAP = 30.0
GATE_VOICED = 3.0
#: Holes from SHORT_GAP s up to GAP are filled too, with SHORT_LEAD s of audio either side for context.
SHORT_GAP = 3.0
SHORT_LEAD = 10.0
#: A gap window's segment saying what an existing one says (ANCHOR_SIMILAR of its letters and digits, in order) is
#: an anchor; the window's clock is off when the lower median of the anchors' offsets is over ANCHOR_TOL s.
ANCHOR_TOL = 1.0
ANCHOR_SIMILAR = 0.7
#: A segment running on more than TAIL_MIN s past its last speech ends there, but no earlier than its text takes
#: at 1 / TAIL_SLACK of the source's median speaking rate.
TAIL_MIN = 1.0
TAIL_SLACK = 3.0
#: Two labels talking over each other for more than this share of the smaller one's time are two people.
MERGE_OVERLAP = 0.1
#: A segment repeating its last LOOP_SPAN tokens with a period of at most LOOP_PERIOD is looping.
LOOP_SPAN = 48
LOOP_PERIOD = 12
#: Wrong language: how much text a test needs, and the shares below which it is not the source language.
LANG_WORDS = 30
LANG_LETTERS = 40
EN_SHARE = 0.15
KANA_SHARE = 0.2
EN_COMMON = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be been it its this that these those i "
    "you he she we they me him her us them my your his our their not no do does did have has had will would can "
    "could what who how why when where there here so if just like yeah okay oh all about up out".split())


def wrong_language(text: str, language: str | None) -> bool | None:
    """Whether `text` is not in `language`; None while it is too short to tell. English is told from other
    Latin-script languages by its common words, Japanese from Chinese by kana, the rest only by script (as
    `mosspro.wrong_script` checks MOSS Pro). A language it cannot test, or none, passes. Simplified against
    traditional Chinese is not a language error."""
    lang = (language or "").lower()
    if lang not in LATIN | NON_LATIN:
        return False
    if lang == "en":
        words = re.findall(r"[a-z']+", text.lower())
        if len(words) < LANG_WORDS:
            return None
        return sum(w in EN_COMMON for w in words) / len(words) < EN_SHARE
    # units: a run of Latin letters is one (a word), any other letter one each -- so a few English terms in Chinese
    # or Japanese talk do not outweigh the characters around them, as they would counted letter by letter
    names = [unicodedata.name(ch, "") for ch in re.sub(r"[A-Za-zÀ-ɏ]+", "A", text)
             if unicodedata.category(ch)[0] == "L"]
    if len(names) < LANG_LETTERS:
        return None
    latin = sum("LATIN" in n for n in names) / len(names)
    kana = sum("HIRAGANA" in n or "KATAKANA" in n for n in names) / len(names)
    if lang == "ja":
        return kana < KANA_SHARE
    if lang == "zh":
        return kana >= KANA_SHARE or latin > 0.5
    return latin < 0.5 if lang in LATIN else latin > 0.5


def periodic(body: Sequence[int]) -> bool:
    """Whether the last LOOP_SPAN tokens repeat with a period of at most LOOP_PERIOD."""
    if len(body) < LOOP_SPAN:
        return False
    tail = body[-LOOP_SPAN:]
    return any(all(tail[i] == tail[i - p] for i in range(p, LOOP_SPAN)) for p in range(1, LOOP_PERIOD + 1))


def gaps(segs: Sequence[dict], dur: float, least: float = GAP) -> list[tuple[float, float]]:
    """Stretches of at least `least` s that no segment covers, up to `dur`."""
    out, reach = [], 0.0
    for s in sorted(segs, key=lambda s: s["start"]):
        if s["start"] - reach >= least:
            out.append((reach, s["start"]))
        reach = max(reach, s["end"])
    if dur - reach >= least:
        out.append((reach, dur))
    return out


def anchors(extra: Sequence[dict], segs: Sequence[dict], lo: float, hi: float) -> list[tuple[dict, dict]]:
    """(window segment, existing segment) pairs saying the same thing among the existing segments inside the
    window [lo, hi], the nearest in time of those: what the window may have heard again."""
    inside = [y for y in segs if y["end"] > lo and y["start"] < hi and len(norm(y["text"])) >= 2]
    out = []
    for x in extra:
        t = norm(x["text"])
        if len(t) < 2:
            continue
        same = [y for y in inside if difflib.SequenceMatcher(None, t, norm(y["text"])).ratio() >= ANCHOR_SIMILAR]
        if same:
            out.append((x, min(same, key=lambda y: abs(y["start"] - x["start"]))))
    return out


def kept_from_window(extra: list[dict], segs: Sequence[dict], lo: float, hi: float, a: float, b: float,
                     short: bool) -> tuple[list[dict], int, float]:
    """What a gap window over [lo, hi] for the gap [a, b] adds: nothing when its clock is off (`anchors`), else
    its segments but those that are an existing segment heard again, inside the gap (a short gap's: centred in
    it, its long lead-in hands it the neighbours). Also the anchor count and offset, for the event."""
    pairs = anchors(extra, segs, lo, hi)
    late = sorted(abs(x["start"] - y["start"]) for x, y in pairs)
    off = late[(len(late) - 1) // 2] if late else 0.0  # the lower median: one chance match of two rejects nothing
    if off > ANCHOR_TOL:
        return [], len(pairs), off
    seen = {id(x) for x, y in pairs if abs(x["start"] - y["start"]) <= ANCHOR_TOL}
    kept = [s for s in extra if id(s) not in seen
            and (a <= (s["start"] + s["end"]) / 2 < b if short else s["start"] < b and s["end"] > a)]
    return kept, len(pairs), off


def trim_tails(segs: Sequence[dict], spans: Sequence[tuple[float, float]]) -> tuple[list[dict], list[list[float]]]:
    """`segs` with late ends pulled in to their last Silero speech (`spans`), and [start, old end, new end] of
    each. Silero misses lines under game music, whose length is normal for their text: hence the rate floor."""
    def last_speech(s):
        return max((min(e, s["end"]) for a, e in spans if e > s["start"] and a < s["end"]), default=None)

    rows = [(s, last_speech(s), weighted_char_count(s["text"].strip())) for s in segs]
    rates = [w / (s["end"] - s["start"]) for s, last, w in rows
             if last is not None and s["end"] - last < 0.5 and s["end"] - s["start"] > 0.5 and w > 0]
    if not rates:
        return list(segs), []
    rate = float(np.median(rates))
    out, trimmed = [], []
    for s, last, w in rows:
        end = (s["end"] if last is None or s["end"] - last <= TAIL_MIN
               else min(s["end"], max(last, s["start"] + TAIL_SLACK * w / rate)))
        if end < s["end"] - 0.05:
            trimmed.append([round(s["start"], 2), round(s["end"], 2), round(end, 2)])
            s = {**s, "end": round(end, 2)}
        out.append(s)
    return out, trimmed


def merge_labels(segs: list[dict], centroids: Callable[[list[dict]], Mapping[str, np.ndarray]],
                 theta: float) -> list[tuple[str, str, float]]:
    """Relabels `segs` in place and returns the merges (absorbed, kept, cosine): repeatedly, of the labels'
    voiceprint `centroids` (unit vectors, recomputed after each merge), the closest pair at cosine >= `theta` goes
    into the one with more speech -- unless the two talk over each other for more than MERGE_OVERLAP of the smaller
    one's time (one person does not overlap themselves)."""
    talk: dict[str, float] = {}
    for s in segs:
        talk[s["speaker"]] = talk.get(s["speaker"], 0.0) + s["end"] - s["start"]

    def overlap(a: str, b: str) -> float:
        xa = [(s["start"], s["end"]) for s in segs if s["speaker"] == a]
        xb = [(s["start"], s["end"]) for s in segs if s["speaker"] == b]
        return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in xa for s2, e2 in xb)

    merges = []
    while True:
        cents = centroids(segs)
        best = None
        for a, b in itertools.combinations(sorted(cents), 2):
            sim = float(cents[a] @ cents[b])
            if sim >= theta and (best is None or sim > best[2]) \
                    and overlap(a, b) <= MERGE_OVERLAP * min(talk[a], talk[b]):
                best = (a, b, sim)
        if best is None:
            return merges
        a, b, sim = best
        keep, gone = (a, b) if talk[a] >= talk[b] else (b, a)
        for s in segs:
            if s["speaker"] == gone:
                s["speaker"] = keep
        talk[keep] += talk.pop(gone)
        merges.append((gone, keep, round(sim, 3)))
