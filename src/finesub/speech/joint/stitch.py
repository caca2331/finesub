"""Stitching overlapping windows of one provider back into one answer.

Units are dicts with `start`, `end`, `text`, `speaker` (absolute seconds).

* Hand-over: window k keeps the units whose midpoint falls before its hand-over
  point with window k+1 -- a gap no unit of either window spans, near the
  middle of their overlap (or near where window k stopped, if it stopped
  early).
* Speakers: labels are only consistent within one window. In each overlap the
  two windows' labels are linked one-to-one, greedily by co-speaking seconds;
  an unlinked label is a new person.
* Word units keep the providers' output order: window k's kept units, then
  window k+1's, never sorted (a turn is a speaker's run in output order, and a
  global sort by start would interleave two overlapping speakers' words).
  Segment units (MOSS Pro) are cut into differently shaped segments by each
  window, so each seam is reconciled by text and the result sorted by start --
  a segment is a turn of its own, so the sort interleaves nothing.

Ported from the experiment's `tools/ts_diar_explore/episode.py`; the seam
reconciliation was smoke-tested there, not run in the readouts (their sources
were all <= 60 min, one MOSS Pro call).
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Sequence

HANDOVER_SLACK = 15.0
#: share of a segment's characters that must appear (in order) on the other side for it to count as already kept
COVER_SHARE = 0.8
#: seconds two windows' timestamps for the same utterance may disagree by
SEAM_JITTER = 0.3
#: an event starting this close to where the right window begins (or ending this close to where the left one
#: ends) is taken from the other window, which heard all of it
EVENT_EDGE = 5.0

Part = tuple[float, float, list[dict]]  # (window start, window end, its units in output order)


def windows(lo: float, hi: float, size: float | None, overlap: float) -> list[tuple[float, float]]:
    if size is None or hi - lo <= size:
        return [(lo, hi)]
    out, start = [], lo
    while True:
        end = min(hi, start + size)
        out.append((start, end))
        if end >= hi:
            return out
        start = end - overlap


def norm(text: str) -> str:
    """Letters and digits only, NFKC: what two transcriptions of one utterance share."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def overlap(x: dict, y: dict) -> float:
    return max(0.0, min(x["end"], y["end"]) - max(x["start"], y["start"]))


def by_label(units: Sequence[dict], lo: float, hi: float) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for u in units:
        a, b = max(u["start"], lo), min(u["end"], hi)
        if u.get("speaker") is not None and b > a:
            out[u["speaker"]].append((a, b))
    return out


def _co_seconds(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b)


def link(prev: dict[str, list], cur: dict[str, list]) -> list[tuple[str, str]]:
    """(prev label, cur label) pairs, one-to-one, greedy by co-speaking seconds, largest first."""
    pairs = sorted(((_co_seconds(prev[p], cur[c]), i, j, p, c)
                    for i, p in enumerate(prev) for j, c in enumerate(cur)), key=lambda x: (-x[0], x[1], x[2]))
    out, used_p, used_c = [], set(), set()
    for seconds, _, _, p, c in pairs:
        if seconds > 0 and p not in used_p and c not in used_c:
            out.append((p, c))
            used_p.add(p)
            used_c.add(c)
    return out


def stitch(parts: Sequence[Part], *, segments: bool) -> list[dict]:
    """The windows' units as one list: output order for words, start order for segments."""

    if not parts:
        return []
    if len(parts) == 1:  # one window: its labels are already one namespace
        units = [dict(u) for u in parts[0][2]]
        return sorted(units, key=lambda u: u["start"]) if segments else units
    cuts = []
    for k in range(len(parts) - 1):
        units, nxt = parts[k][2], parts[k + 1][2]
        last = max((u["end"] for u in units), default=parts[k][0])
        mid = (parts[k + 1][0] + parts[k][1]) / 2
        # A window that stopped early hands over where it stopped (at the latest).
        cuts.append(handover(list(units) + list(nxt), parts[k + 1][0], max(parts[k + 1][0], min(parts[k][1], last)),
                             max(parts[k + 1][0], min(mid, last))))
    wins, prev = [], []
    for k, (start, _, units) in enumerate(parts):
        own = [dict(u) for u in units]
        for u in own:
            u["speaker"] = f"w{k}:{u['speaker']}" if u.get("speaker") is not None else None
        if k:
            lo, hi = start, parts[k - 1][1]
            mapping = {c: p for p, c in link(by_label(prev, lo, hi), by_label(own, lo, hi))}
            for u in own:  # an unlinked label is a new person and keeps its window-local name
                u["speaker"] = mapping.get(u["speaker"], u["speaker"])
        wins.append(own)
        prev = own
    kept = [[u for u in own if (cuts[k - 1] if k else float("-inf")) <= (u["start"] + u["end"]) / 2
             < (cuts[k] if k + 1 < len(parts) else float("inf"))] for k, own in enumerate(wins)]
    if not segments:
        return [u for side in kept for u in side]
    for k in range(len(parts) - 1):
        reconcile(kept[k], kept[k + 1], wins[k], wins[k + 1], cuts[k], (parts[k + 1][0], parts[k][1]))
    return sorted((u for side in kept for u in side), key=lambda u: u["start"])


def handover(units: Sequence[dict], lo: float, hi: float, target: float) -> float:
    """Where two overlapping windows hand over: in a gap no unit of either window spans, within
    HANDOVER_SLACK of `target` and inside [lo, hi], closest to `target`. Cutting where a unit crosses would
    split one sentence by two different segment shapes, and each window's copy can then fall on the
    discarded side. Without a gap near `target` (continuous speech), `target` itself."""
    lo, hi = max(lo, target - HANDOVER_SLACK), min(hi, target + HANDOVER_SLACK)
    spans = sorted((u["start"], u["end"]) for u in units if u["end"] > lo and u["start"] < hi)
    gaps, reach = [], lo
    for a, b in spans:
        if a >= reach:  # a zero-width gap (units that only touch) is still a point nothing crosses
            gaps.append((reach, a))
        reach = max(reach, b)
    if reach < hi:
        gaps.append((reach, hi))
    if not gaps:
        return target
    return min((min(max(target, a), b) for a, b in gaps), key=lambda t: abs(t - target))


def _lcs(a: str, b: str) -> int:
    """Length of the longest common subsequence."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        for j, other in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ch == other else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _slice_in(o: dict, lo: float, hi: float) -> str:
    """The part of `o`'s normalised text that falls in [lo, hi), cut in proportion to time."""
    text, dur = norm(o["text"]), max(o["end"] - o["start"], 1e-3)
    a, b = max(o["start"], lo), min(o["end"], hi)
    if b <= a:
        return ""
    return text[round(len(text) * (a - o["start"]) / dur):round(len(text) * (b - o["start"]) / dur)]


def _covered(unit: dict, others: Sequence[dict]) -> bool:
    """Whether the same utterance as `unit` is already among `others`. Only what the others say inside
    `unit`'s own time span (± SEAM_JITTER) counts, so a separate utterance with the same words never makes
    it a copy. Covered when COVER_SHARE of its characters appear in order in that time-ordered text, or one
    other segment spans most of it in time and shares at least half its characters there."""
    text = norm(unit["text"])
    if not text:
        return True
    lo, hi = unit["start"] - SEAM_JITTER, unit["end"] + SEAM_JITTER
    near = sorted((o for o in others if o["end"] > lo and o["start"] < hi), key=lambda o: o["start"])
    if _lcs(text, "".join(_slice_in(o, lo, hi) for o in near)) >= COVER_SHARE * len(text):
        return True
    span = max(unit["end"] - unit["start"], 1e-3)
    return any(overlap(unit, o) >= 0.5 * span and _lcs(text, _slice_in(o, lo, hi)) >= 0.5 * len(text) for o in near)


def _events(left: Sequence[dict], right: Sequence[dict]) -> list[tuple[list[dict], list[dict]]]:
    """The two windows' segments grouped into utterance events: a left and a right segment are one event when
    one lies mostly inside the other (>= half of the shorter, both widened by SEAM_JITTER); events chain."""
    units = [(0, u) for u in left] + [(1, u) for u in right]
    parent = list(range(len(units)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def padded(u: dict) -> dict:
        return {"start": u["start"] - SEAM_JITTER, "end": u["end"] + SEAM_JITTER}

    for i, (si, ui) in enumerate(units):
        for j in range(i + 1, len(units)):
            sj, uj = units[j]
            short = min(ui["end"] - ui["start"], uj["end"] - uj["start"]) + 2 * SEAM_JITTER
            if si != sj and overlap(padded(ui), padded(uj)) >= 0.5 * short:
                parent[root(i)] = root(j)
    groups: dict[int, tuple[list, list]] = {}
    for i, (side, u) in enumerate(units):
        groups.setdefault(root(i), ([], []))[side].append(u)
    return list(groups.values())


def reconcile(left: list[dict], right: list[dict], left_all: Sequence[dict], right_all: Sequence[dict], cut: float,
              region: tuple[float, float]) -> None:
    """Make a seam keep each utterance once, in place. In the windows' overlap `region` each event is taken
    whole from one window -- a window's own segments neither repeat nor miss speech: the left window for an
    event near the region's start, the right for one near its end, else the side of the cut its middle is on.
    An event chained across the whole region falls back to matching by text (`_text_reconcile`)."""
    lo, hi = region

    def in_region(u: dict) -> bool:
        return u["end"] > lo and u["start"] < hi

    for ls, rs in _events([u for u in left_all if in_region(u)], [u for u in right_all if in_region(u)]):
        start, end = min(u["start"] for u in ls + rs), max(u["end"] for u in ls + rs)
        near_lo, near_hi = start < lo + EVENT_EDGE, end > hi - EVENT_EDGE
        if near_lo and near_hi and ls and rs:
            _text_reconcile(left, right, ls, rs, cut)
            continue
        take_left = bool(ls) and (not rs or near_lo or (not near_hi and (start + end) / 2 < cut))
        ids = {id(u) for u in ls + rs}
        left[:] = [u for u in left if id(u) not in ids]
        right[:] = [u for u in right if id(u) not in ids]
        (left if take_left else right).extend(ls if take_left else rs)


def _text_reconcile(left: list[dict], right: list[dict], ls: list[dict], rs: list[dict], cut: float) -> None:
    """One long event: within HANDOVER_SLACK of the cut, a kept right segment whose utterance the kept left
    side already has is dropped, then a dropped segment whose utterance neither kept side has is restored."""

    def near(u: dict) -> bool:
        return u["end"] > cut - HANDOVER_SLACK and u["start"] < cut + HANDOVER_SLACK

    in_rs = {id(u) for u in rs}
    for u in [u for u in right if id(u) in in_rs and near(u)]:
        if _covered(u, left):
            right.remove(u)
    ids = {id(u) for u in left + right}
    dropped = [(u, left) for u in ls if near(u) and id(u) not in ids] + [(u, right) for u in rs if near(u) and id(u) not in ids]
    dropped.sort(key=lambda x: x[0]["start"])
    for u, side in dropped:
        if not _covered(u, left + right):
            side.append(u)
