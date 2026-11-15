"""Acoustic row splitting (docs/diarization-findings.md 6.5): proposals, text gates, rulers, three-tier grader.

Library only; the commands are in `row_split.py`.
Tokenizing needs sudachipy + sudachidict_core, which no extra lists on its own: `pip install sudachipy sudachidict_core`
(`pip install -e ".[kb-phonetic]"` brings both too).

Tiers (owner 2026-10-14: by whether correction can repair it). A one-person row's tier comes from the frozen labels
(`row_split.py`); the text rule below is only the fallback for a cut nobody labelled:
  exact   two-person row, cut exactly at the change (punctuation, whitespace, long vowel not counted)
  minor   two-person row with <= MINOR_CHARS chars or only function words / interjections on the wrong side;
          a three-person row cut at one of its changes; a simultaneous row;
          one-person row cut mid-sentence (correction merges it back) or with one half only interjection / filler
  severe  two-person row with a content chunk (> MINOR_CHARS chars) on the wrong side;
          one-person row cut at a sentence boundary (both halves stand alone: correction keeps them apart)
  other   rows the ruler does not score (bg, filler, unclear, ...)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
# Rulers, labels and the acceptance runs' products live in the main checkout, not in a worktree.
MAIN = Path(*WT.parts[: WT.parts.index(".worktrees")]) if ".worktrees" in WT.parts else WT
for p in (HERE, WT / "examples", WT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CLEAN, DIRTY = 0.8, 0.2
MINOR_CHARS = 3
END_PUNCT = set("。！？!?…")
FINAL = set("よねなさかぞわ")
SKIPPABLE = set("ー!?！？、。…") | FINAL

_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from sudachipy import dictionary
        _TOK = dictionary.Dictionary().create()
    return _TOK


def chars(text):
    return len(re.sub(r"[\W_ー]", "", text))


def morphs(text):
    return [m for m in tok().tokenize(text) if m.part_of_speech()[0] != "空白"] if text else []


# --- acoustic proposal --------------------------------------------------------------------------------------

def word_cover(words, segs):
    out = []
    for w in words:
        s, e = float(w["start"]), float(w["end"])
        if e - s < 0.02:
            m = (s + e) / 2
            s, e = m - 0.01, m + 0.01
        d = Counter()
        for t, a, b in segs:
            if b > s and a < e:
                d[t] += min(b, e) - max(a, s)
        out.append({k: v / (e - s) for k, v in d.items()})
    return out


def clean_label(cov):
    if not cov:
        return None
    k, v = max(cov.items(), key=lambda kv: kv[1])
    return k if v >= CLEAN and all(x <= DIRTY for t, x in cov.items() if t != k) else None


_INDEX = {}


def _near(segs, lo, hi):
    import bisect
    key = id(segs)
    if key not in _INDEX:
        order = sorted(segs, key=lambda s: s[1])
        _INDEX[key] = (segs, order, [s[1] for s in order], max((b - a for _, a, b in segs), default=0.0))
    _, order, starts, longest = _INDEX[key]
    i = bisect.bisect_left(starts, lo - longest)
    j = bisect.bisect_right(starts, hi)
    return [s for s in order[i:j] if s[2] > lo and s[1] < hi]


def labels(row, segs):
    words = row.get("words") or []
    lo, hi = float(row["start"]) - 0.5, float(row["end"]) + 0.5
    near = _near(segs, lo, hi)
    return [clean_label(c) for c in word_cover(words, near)], near


def propose(row, segs, T, C):
    """(cut word index, A, B, valid cut range) or None."""
    words = row.get("words") or []
    if len(words) < 2:
        return None
    labs, near = labels(row, segs)
    dur = [float(w["end"]) - float(w["start"]) for w in words]
    clean = [(i, k) for i, k in enumerate(labs) if k is not None]
    if not clean:
        return None
    first = clean[0][1]
    change = next((n for n, (_, k) in enumerate(clean) if k != first), None)
    if change is None:
        return None
    A, B = first, clean[change][1]
    left, right = clean[:change], clean[change:]
    if any(k != B for _, k in right):
        return None
    last_a, first_b = left[-1][0], right[0][0]
    if sum(dur[i] for i, _ in left) < T or sum(dur[i] for i, _ in right) < T:
        return None
    onset = float(words[first_b]["start"])
    for t, a, b in near:
        if t == B and a <= float(words[first_b]["end"]) and b >= float(words[first_b]["start"]):
            onset = a
            break
    cut = min(range(last_a + 1, first_b + 1), key=lambda b: abs(float(words[b]["start"]) - onset))
    text = [str(w.get("word", "")) for w in words]
    if chars("".join(text[:cut])) < C or chars("".join(text[cut:])) < C:
        return None
    return cut, A, B, (last_a + 1, first_b)


# --- text checks --------------------------------------------------------------------------------------------

def ends_sentence(text):
    ms = morphs(text)
    if not ms:
        return False
    last = ms[-1]
    if last.part_of_speech()[0] == "補助記号":
        if last.surface() and last.surface()[-1] in END_PUNCT:
            return True
        ms = ms[:-1]
        if not ms:
            return False
        last = ms[-1]
    pos = last.part_of_speech()
    if pos[0] == "感動詞":
        return True
    if pos[0] == "助詞" and pos[1] == "終助詞":
        return True
    return pos[0] in ("助動詞", "動詞", "形容詞") and pos[5].startswith("終止形")


def starts_dependent(text):
    ms = [m for m in morphs(text) if m.part_of_speech()[0] != "補助記号"]
    return not ms or ms[0].part_of_speech()[0] in ("助詞", "助動詞", "接尾辞")


def only_interjection(text):
    ms = [m for m in morphs(text) if m.part_of_speech()[0] != "補助記号"]
    return bool(ms) and all(m.part_of_speech()[0] in ("感動詞", "フィラー") for m in ms)


def only_function(text):
    ms = [m for m in morphs(text) if m.part_of_speech()[0] != "補助記号"]
    return all(m.part_of_speech()[0] in ("助詞", "助動詞", "感動詞", "フィラー", "接尾辞") for m in ms)


def word_starts(words):
    s = [0]
    for w in words:
        s.append(s[-1] + len(w))
    return s


def gate(row, cut):
    """Snap to a morpheme start within the two tokens around `cut` whose left text ends a sentence; None if none."""
    words = [str(w.get("word", "")) for w in row["words"]]
    text = "".join(words)
    st = word_starts(words)
    lo, hi = st[max(0, cut - 1)], st[min(len(words), cut + 1)]
    ms = list(tok().tokenize(text))
    cands = sorted({m.begin() for m in ms if lo <= m.begin() <= hi and 0 < m.begin() < len(text)},
                   key=lambda b: abs(b - st[cut]))
    for b in cands:
        if ends_sentence(text[:b]):
            return min(range(1, len(words)), key=lambda i: abs(st[i] - b))
    return None


def gate2(row, cut, T, C):
    g = gate(row, cut)
    if g is None:
        return None
    words = [str(w.get("word", "")) for w in row["words"]]
    i = g
    while i < len(words):
        ms = morphs("".join(words[i:]))
        if not ms:
            break
        pos = ms[0].part_of_speech()
        if (pos[0] == "助詞" and pos[1] == "終助詞") or pos[0] == "補助記号" or set(words[i]) <= SKIPPABLE:
            i += 1
            continue
        break
    if i >= len(words) or starts_dependent("".join(words[i:])):
        return None
    dur = lambda ws: sum(float(w["end"]) - float(w["start"]) for w in ws)  # noqa: E731
    if chars("".join(words[:i])) < C or chars("".join(words[i:])) < C:
        return None
    if dur(row["words"][:i]) < T or dur(row["words"][i:]) < T:
        return None
    return i


# --- rulers -------------------------------------------------------------------------------------------------

def track_names(block):
    out = {}
    for line in block:
        m = re.match(r"\s*s(\d+)\s*=\s*(.+)", line)
        if m:
            names = re.sub(r"（.*?）|\(.*?\)", "", m.group(2))
            out[int(m.group(1))] = {n.strip() for n in names.split("+") if n.strip()}
    return out


def slot_segs(grid, step):
    from finesub.speech.diarization.segments import segments_from_activity
    return [(s.track, s.start, s.end) for s in segments_from_activity(grid.tolist(), frame_seconds=step)]


def ruler1():
    """Row truth: items with truth {cls, cuts (char positions) or None, parts}."""
    import row_truth as rt
    data = MAIN / "data" / "diarization" / "row-truth"
    rulers = {}
    for folder in ("dev-v2-high", "heldout-v3-high"):
        for p in sorted((data / folder).glob("*.json")):
            rulers[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    items, seconds, cache = [], 0.0, {}
    for key, ident, lo, hi, source in rt.windows():
        if source != "accept" or key not in rulers:
            continue
        if ident not in cache:
            stable = json.loads((data / "products" / "acoustic" / ident / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]
            base = MAIN / "tmp" / "diarization-retrack" / "accept" / ident
            diar = json.loads((base / f"{ident}-diar.json").read_text(encoding="utf-8"))
            tracks = [(int(t["track"]), float(t["start"]), float(t["end"])) for t in diar["tracks"]]
            slots = slot_segs(np.load(base / f"{ident}-diar-evidence.npz")["slot_grid"], 0.02)
            cache[ident] = (stable, tracks, slots)
        stable, tracks, slots = cache[ident]
        names = track_names(rulers[key].get("tracks") or [])
        seconds += float(stable[hi - 1]["end"]) - float(stable[lo]["start"])
        for i in range(lo, hi):
            row = rulers[key]["rows"][f"q{i}"]
            items.append(dict(ruler=1, key=key, ident=ident, i=i, seg=stable[i], tracks=tracks, slots=slots,
                              stable=stable, names=names, truth=_truth1(row, stable[i]), raw=row))
    return items, seconds


def _truth1(row, seg):
    parts = row.get("parts") or []
    cls = row["cls"]
    if cls == "one":
        return dict(kind="single", people=row["who"])
    if cls == "multi":
        if len(parts) < 2:
            return dict(kind="together", people=row["who"])
        text = "".join(str(w.get("word", "")) for w in seg.get("words") or [])
        cuts, pos = [], 0
        for p in parts[1:]:
            j = text.find(p["from"], max(1, pos))
            if j < 0:
                j = text.find(p["from"][:2], max(1, pos))
            if j > 0:
                cuts.append((j, j))
                pos = j + 1
        return dict(kind="split", cuts=cuts, people=[p["who"] for p in parts], nparts=len(parts))
    return dict(kind="other:" + cls)


def ruler2():
    import edit_judge_truth as ej
    import p2_features as pf
    items, seconds = [], 0.0
    for ident in ej.HINA:
        t = ej.load(ident)
        z = np.load(pf.CACHE / f"{ident}.npz")
        tracks = slot_segs(z["native"], pf.STEP)
        slots = slot_segs((z["probs"] > 0.5).astype(np.int8), float(z["frame_seconds"]))
        seconds += float(t.stable[-1]["end"]) - float(t.stable[0]["start"])
        for i, info in t.rows.items():
            seg = t.stable[i]
            kind = info.get("kind")
            if kind == "single":
                truth = dict(kind="single", people=info.get("people"))
            elif kind == "split":
                words = [str(w.get("word", "")) for w in seg.get("words") or []]
                st = word_starts(words)
                lo, hi = info["split"]
                truth = dict(kind="split", cuts=[(st[lo], st[hi])], people=info.get("people"), nparts=2)
            else:
                truth = dict(kind="other:" + str(kind))
            items.append(dict(ruler=2, key=ident, ident=ident, i=i, seg=seg, tracks=tracks, slots=slots,
                              stable=t.stable, names={}, truth=truth, raw=info))
    return items, seconds


# --- grading ------------------------------------------------------------------------------------------------

def grade(item, cut):
    """(tier, detail)."""
    words = [str(w.get("word", "")) for w in item["seg"]["words"]]
    text = "".join(words)
    pos = word_starts(words)[cut]
    left, right = text[:pos], text[pos:]
    t = item["truth"]
    kind = t["kind"]
    if kind.startswith("other:"):
        return "other", kind
    if kind == "together":
        return "minor", "simultaneous"
    if kind == "single":
        if only_interjection(left) or only_interjection(right):
            return "minor", "single:interjection half"
        if ends_sentence(left) and not starts_dependent(right):
            return "severe", "single:sentence boundary"
        return "minor", "single:mid-sentence"
    best = None
    for lo, hi in t["cuts"]:
        if lo <= pos <= hi:
            seg_text = ""
        else:
            a, b = (pos, lo) if pos < lo else (hi, pos)
            seg_text = text[a:b]
        n = chars(seg_text)
        if best is None or n < best[0]:
            best = (n, seg_text)
    if best is None:
        return "other", "split:truth cut not found"
    n, seg_text = best
    extra = t.get("nparts", 2) > 2
    if n == 0:
        return ("minor", "split:exact, 3+ people") if extra else ("exact", "split:exact")
    if n <= MINOR_CHARS or only_function(seg_text):
        return "minor", f"split:off {n} ({seg_text})"
    return "severe", f"split:off {n} ({seg_text})"


TIERS = ("exact", "minor", "severe", "other")


def summarize(results, seconds, positives, singles):
    """results: [(tier, detail)]."""
    c = Counter(t for t, _ in results)
    split_minor = sum(1 for t, d in results if t == "minor" and not d.startswith("single"))
    single_minor = sum(1 for t, d in results if t == "minor" and d.startswith("single"))
    sev_single = sum(1 for t, d in results if t == "severe" and d.startswith("single"))
    return (f"n={sum(c.values())} ({sum(c.values()) / (seconds / 3600):.1f}/h): exact {c['exact']}, "
            f"minor {c['minor']} (two-person {split_minor}, one-person {single_minor}), "
            f"severe {c['severe']} (one-person {sev_single}, two-person {c['severe'] - sev_single}), other {c['other']}; "
            f"two-person rows improved {c['exact'] + split_minor}/{positives}; severe per 1000 one-person rows "
            f"{1000 * sev_single / singles:.2f}")


def counts(items):
    pos = sum(1 for it in items if it["truth"]["kind"] == "split")
    single = sum(1 for it in items if it["truth"]["kind"] == "single")
    return pos, single


# --- in-context gate ----------------------------------------------------------------------------------------

def _final_pos(m):
    pos = m.part_of_speech()
    if pos[0] == "感動詞":
        return True
    if pos[0] == "助詞" and pos[1] == "終助詞":
        return True
    return pos[0] in ("助動詞", "動詞", "形容詞") and pos[5].startswith("終止形")


def gate3(row, cut, T, C, skip_finals=True):
    """Like gate2, but morphemes come from the whole row (context decides 終止形 vs 連体形). The cut snaps to a
    morpheme boundary within the two tokens around `cut` where the left ends a sentence (or punctuation) and the
    right starts with an independent word; sentence-final particles at the right's start are moved left first."""
    words = [str(w.get("word", "")) for w in row["words"]]
    text = "".join(words)
    st = word_starts(words)
    ms = [m for m in tok().tokenize(text)]
    lo, hi = st[max(0, cut - 1)], st[min(len(words), cut + 1)]
    bounds = sorted({m.begin() for m in ms if lo <= m.begin() <= hi and 0 < m.begin() < len(text)},
                    key=lambda b: abs(b - st[cut]))
    for b in bounds:
        if skip_finals:
            while True:
                nxt = [m for m in ms if m.begin() == b]
                if not nxt or nxt[0].end() <= b:
                    break
                p = nxt[0].part_of_speech()
                if (p[0] == "助詞" and p[1] == "終助詞") or p[0] in ("補助記号", "空白") or set(nxt[0].surface()) <= SKIPPABLE:
                    b = nxt[0].end()
                    continue
                break
        if b >= len(text):
            continue
        before = [m for m in ms if m.end() <= b and m.part_of_speech()[0] not in ("空白",)]
        after = [m for m in ms if m.begin() >= b and m.part_of_speech()[0] not in ("空白", "補助記号")]
        if not before or not after:
            continue
        last = before[-1]
        ok_left = (last.part_of_speech()[0] == "補助記号" and last.surface()[-1:] in END_PUNCT | set("、,")) or _final_pos(last) or \
            (last.part_of_speech()[0] == "補助記号" and len(before) > 1 and _final_pos(before[-2]))
        if not ok_left or after[0].part_of_speech()[0] in ("助詞", "助動詞", "接尾辞"):
            continue
        i = min(range(1, len(words)), key=lambda k: abs(st[k] - b))
        dur = lambda ws: sum(float(w["end"]) - float(w["start"]) for w in ws)  # noqa: E731
        if chars("".join(words[:i])) < C or chars("".join(words[i:])) < C:
            continue
        if dur(row["words"][:i]) < T or dur(row["words"][i:]) < T:
            continue
        return i
    return None


def place(row, p, widen=1):
    """Placement only (never rejects): within the acoustic range of valid cuts widened by `widen` tokens, the morpheme
    boundary whose left ends a sentence and whose right starts an independent word (whole-row parse), nearest to the
    acoustic cut; sentence-final particles at the right's start move left. Else the acoustic cut."""
    cut, _, _, (a, b) = p
    words = [str(w.get("word", "")) for w in row["words"]]
    text = "".join(words)
    st = word_starts(words)
    ms = list(tok().tokenize(text))
    lo, hi = st[max(1, a - widen)], st[min(len(words) - 1, b + widen)]
    cands = sorted({m.begin() for m in ms if lo <= m.begin() <= hi and 0 < m.begin() < len(text)},
                   key=lambda x: abs(x - st[cut]))
    for x in cands:
        while True:
            nxt = [m for m in ms if m.begin() == x]
            if not nxt or nxt[0].end() <= x:
                break
            pp = nxt[0].part_of_speech()
            if (pp[0] == "助詞" and pp[1] == "終助詞") or pp[0] in ("補助記号", "空白") or set(nxt[0].surface()) <= SKIPPABLE:
                x = nxt[0].end()
                continue
            break
        if x >= len(text):
            continue
        before = [m for m in ms if m.end() <= x and m.part_of_speech()[0] != "空白"]
        after = [m for m in ms if m.begin() >= x and m.part_of_speech()[0] not in ("空白", "補助記号")]
        if not before or not after or after[0].part_of_speech()[0] in ("助詞", "助動詞", "接尾辞"):
            continue
        last = before[-1]
        ok = _final_pos(last) or (last.part_of_speech()[0] == "補助記号" and (last.surface()[-1:] in END_PUNCT | set("、,")
                                                                             or (len(before) > 1 and _final_pos(before[-2]))))
        if ok and x in st:
            return st.index(x)
        if ok:
            return min(range(1, len(words)), key=lambda k: abs(st[k] - x))
    return cut


# --- the production DP's scores on a stable file (plan: 方案 A、B) --------------------------------------------

class Stream:
    """The segmentation DP's boundary and piece scores over one stable file, computed the way `split_segments` does
    (regroup to Whisper segments, per-segment `adjust_words` + `score_boundaries`, seams scored fresh with the bonus)
    and indexed back to the stable rows: row r's word w is stream word `offset[r] + w`."""

    def __init__(self, stable, vad_segments, length_scale=1.0):
        from finesub.speech.postprocessing import segmentation as seg
        self.seg = seg
        self.params = seg.split_params_for_length_scale(float(length_scale))
        spans = sorted((float(i["start"]), float(i["end"])) for i in vad_segments
                       if i.get("start") is not None and i.get("end") is not None and float(i["end"]) > float(i["start"]))
        zones = seg.build_zones(spans)
        rows = [seg._ensure_minimum_word(s) for s in stable]
        self.offset, n = [], 0
        for r in rows:
            self.offset.append(n)
            n += len(r.get("words") or [])
        groups = seg.regroup_by_whisper_segments(rows)
        self.adj, self.bounds, self.seam = [], [], []
        for g in groups:
            ga = seg.adjust_words(g.get("words") or [], spans, zones)
            if not ga:
                raise ValueError("adjust_words returned nothing for a segment")
            if self.adj:
                s0 = seg.boundary_score(self.adj[-1], ga[0], spans, self.params)
                self.bounds.append(seg.Boundary(False, s0.g, s0.t, s0.b - self.params.whisper_segment_bonus, s0.non_vad_gap))
                self.seam.append(True)
            if len(ga) >= 2:
                self.bounds.extend(seg.score_boundaries(ga, spans, self.params))
                self.seam.extend([False] * (len(ga) - 1))
            self.adj.extend(ga)
        if len(self.adj) != n:
            raise ValueError(f"stream has {len(self.adj)} words, stable rows {n}")
        from finesub.subtitles.metrics import weighted_char_count
        self.prefix = [0.0]
        for w in self.adj:
            self.prefix.append(self.prefix[-1] + weighted_char_count(w.text) + (0.5 if w.space_before else 0.0))

    def boundary(self, r, cut):
        """The Boundary between row r's words cut-1 and cut, and whether it is a Whisper seam."""
        k = self.offset[r] + cut - 1
        return self.bounds[k], self.seam[k]

    def shape(self, r, cut):
        """B(k) without the text term and without the seam bonus: the VAD-gap / word-pause part, plus base."""
        b, seam = self.boundary(r, cut)
        p = self.params
        return (b.b + (p.whisper_segment_bonus if seam else 0.0) - p.base) / p.a - b.t + p.base

    def piece(self, r, lo, hi):
        a, b = self.offset[r] + lo, self.offset[r] + hi
        return self.seg.piece_score(self.prefix[b] - self.prefix[a], self.adj[b - 1].end - self.adj[a].start, self.params)


def dp_place(stream, r, p):
    """Scheme A: within the acoustic range, the non-banned boundary with the lowest B(k) (seam bonus included), ties to
    the acoustic cut, then left. The acoustic cut when nothing in range qualifies; None when that one is banned too."""
    cut, _, _, (lo, hi) = p
    cands = []
    for c in range(max(1, lo), hi + 1):
        b, _ = stream.boundary(r, c)
        if not b.banned:
            cands.append((b.b, abs(c - cut), c))
    if cands:
        return min(cands)[2]
    return None if stream.boundary(r, cut)[0].banned else cut


def dp_margin(stream, r, p, n_words):
    """Scheme B's gate value: min over the non-banned cuts in range of P(left) + P(right) - P(row) + shape(k)."""
    _, _, _, (lo, hi) = p
    whole = stream.piece(r, 0, n_words)
    vals = [stream.piece(r, 0, c) + stream.piece(r, c, n_words) - whole + stream.shape(r, c)
            for c in range(max(1, lo), hi + 1) if not stream.boundary(r, c)[0].banned]
    return min(vals) if vals else None


def halves_tracks(row, cut, tracks):
    """(left track, right track): each half's majority clean vbx track (None when a half has no clean word)."""
    words = row["words"]
    labs, _ = labels(row, tracks)
    def major(xs):
        c = Counter(k for k in xs if k is not None)
        return c.most_common(1)[0][0] if c else None
    return major(labs[:cut]), major(labs[cut:])


#: B(k) of a bare boundary: CJK word boundary, no punctuation, no space, no pause (t = 1, non-VAD-gap penalty 1, base 1)
BARE = 3.0


def glue_left_word(stream, r, cut):
    """The segmentation DP's gap-word glue, applied to the word just left of the cut: a break before it and none after
    it (`B < BARE` on its left, `>= BARE` on its right) means it is the first word of the right half -- the
    diarizer's change point lags, so the second speaker's first word often lands on the first speaker's side."""
    if cut >= 2:
        before, _ = stream.boundary(r, cut - 1)
        after, _ = stream.boundary(r, cut)
        if not before.banned and before.b < BARE <= after.b:
            return cut - 1
    return cut
