"""Runs a model arm (`arms.ARMS`: adapter, window geometry) over an episode and stitches windowed output back
into one arm. Windowed arms' speaker labels are only consistent within one window. Stitching:

* text: window k keeps the tokens whose midpoint falls before the middle of its overlap with k+1 (or before
  where it stopped, if it stopped early);
* speakers: in each overlap the two windows' labels are linked greedily one-to-one by co-speaking seconds
  (`linking.link`); an unlinked label becomes a new person. `--no-link` keeps window-local labels.

Windows are cut from the vocal track (production's ASR input) unless `--mix`. Each window's parsed output
is cached, so a rerun only calls for the windows still missing -- a quota hit mid-episode resumes.
`--span A B` runs over [A, B) s of the episode only (times stay absolute; outputs get an `-sA-B` suffix), so a
representative stretch can be compared across arms without paying for the whole episode.

    python -m tools.ts_diar_explore.episode <item> <arm> [--span A B] [--mix] [--no-link]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import LCSseq

from . import linking
from .arms import ARMS, stem
from .common import EPISODES, RUNS, TMP, clip, probe_duration, save, source_audio, vocal_track, words_to_segments
from .score import norm

HANDOVER_SLACK = 15.0
#: share of a segment's characters that must appear (in order) on the other side for it to count as already kept
COVER_SHARE = 0.8
#: seconds two windows' timestamps for the same utterance may disagree by
SEAM_JITTER = 0.3
#: an event starting this close to where the right window begins (or ending this close to where the left one ends)
#: is taken from the other window, which heard all of it
EVENT_EDGE = 5.0
MOSS_PY = TMP / "moss" / ".venv" / "Scripts" / "python.exe"


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


def window_audio(item: str, src: Path, tag: str, start: float, end: float) -> Path:
    return clip(src, TMP / "clips" / item / f"{tag}-{start:.0f}-{end - start:.0f}.wav", start, end - start)


def run_moss(item: str, name: str, wins: list[tuple[float, float]], src: Path, tag: str) -> list[Path]:
    """One model load for every missing window; outputs renamed to the w<k>.json layout."""
    outdir = EPISODES / item / name
    outs = [outdir / f"w{k:02d}.json" for k in range(len(wins))]
    for k, (s, e) in enumerate(wins):  # a killed run leaves finished windows under raw/
        raw = outdir / "raw" / f"{window_audio(item, src, tag, s, e).stem}.json"
        if not outs[k].exists() and raw.exists():
            raw.replace(outs[k])
    todo = [(k, s, e) for k, (s, e) in enumerate(wins) if not outs[k].exists()]
    if todo:
        args = [f"{window_audio(item, src, tag, s, e)}@{s}" for _, s, e in todo]
        subprocess.run([str(MOSS_PY), "-m", "tools.ts_diar_explore.moss_run", str(outdir / "raw"), *args,
                        "--sdpa", "efficient"], check=True, encoding="utf-8", errors="replace")
        for k, s, e in todo:
            stem = window_audio(item, src, tag, s, e).stem
            (outdir / "raw" / f"{stem}.json").replace(outs[k])
    return outs


def run_window(item: str, arm: str, name: str, k: int, start: float, end: float, src: Path, tag: str) -> Path:
    module, extra = ARMS[arm].module, list(ARMS[arm].extra)
    out = EPISODES / item / name / f"w{k:02d}.json"
    if out.exists():
        return out
    audio = window_audio(item, src, tag, start, end)
    if module == "g35t" and item.startswith("ali-"):
        extra = [*extra, "--lang", "zh-CN"]  # the adapter defaults to ja-JP
    cmd = [sys.executable, "-m", f"tools.ts_diar_explore.{module}", str(audio), str(out), "--offset", str(start), *extra]
    subprocess.run(cmd, check=True, encoding="utf-8", errors="replace")
    return out


def stitch(parts: list[tuple[float, float, dict]], *, do_link: bool) -> dict:
    cuts = []
    for k in range(len(parts) - 1):
        units = parts[k][2]["words"] or parts[k][2]["segments"]
        nxt = parts[k + 1][2]["words"] or parts[k + 1][2]["segments"]
        last = max((u["end"] for u in units), default=parts[k][0])
        mid = (parts[k + 1][0] + parts[k][1]) / 2
        # A window that stopped early hands over where it stopped (at the latest).
        cuts.append(handover(units + nxt, parts[k + 1][0], max(parts[k + 1][0], min(parts[k][1], last)),
                             max(parts[k + 1][0], min(mid, last))))
    wins, prev_words = [], []
    for k, (start, _, arm) in enumerate(parts):
        # Units are words where the model times them, else its segments (MOSS).
        words = [dict(w) for w in (arm["words"] or arm["segments"])]
        for w in words:
            w["speaker"] = f"w{k}:{w['speaker']}" if w["speaker"] is not None else None
        if k and do_link:
            lo, hi = start, parts[k - 1][1]
            mapping = {c: p for p, c in linking.link(linking.by_label(prev_words, lo, hi), linking.by_label(words, lo, hi))}
            for w in words:  # an unlinked label is a new person and keeps its window-local name
                w["speaker"] = mapping.get(w["speaker"], w["speaker"])
        wins.append(words)
        prev_words = words
    # Each unit goes to the side of the handover its midpoint is on.
    kept = [[w for w in words if (cuts[k - 1] if k else float("-inf")) <= (w["start"] + w["end"]) / 2
             < (cuts[k] if k + 1 < len(parts) else float("inf"))] for k, words in enumerate(wins)]
    word_arm = any(p[2].get("words") for p in parts)
    if not word_arm:  # segments are long and cut differently per window: reconcile each seam by text
        for k in range(len(parts) - 1):
            reconcile(kept[k], kept[k + 1], wins[k], wins[k + 1], cuts[k], (parts[k + 1][0], parts[k][1]))
    words_out = sorted((w for side in kept for w in side), key=lambda w: w["start"])
    if not word_arm:  # a word arm whose first window came back empty is still one
        return {"words": None, "segments": words_out}
    return {"words": words_out, "segments": words_to_segments(words_out)}


def _slice_in(o: dict, lo: float, hi: float) -> str:
    """The part of `o`'s normalised text that falls in [lo, hi), cut in proportion to time."""
    text, dur = norm(o["text"]), max(o["end"] - o["start"], 1e-3)
    a, b = max(o["start"], lo), min(o["end"], hi)
    if b <= a:
        return ""
    return text[round(len(text) * (a - o["start"]) / dur):round(len(text) * (b - o["start"]) / dur)]


def _covered(unit: dict, others: list[dict]) -> bool:
    """Whether the same utterance as `unit` is already among `others`. Only what the others say inside `unit`'s own
    time span (± SEAM_JITTER) counts, so a neighbouring, separate utterance with the same words ("はい", "はい")
    never makes it a copy. Covered when at least COVER_SHARE of its characters appear in order in that time-ordered
    text (a subsequence: another speaker's segment can sit between two pieces of one sentence), or one other
    segment spans most of it in time and shares at least half its characters there (the same line transcribed a
    little differently by the two windows)."""
    text = norm(unit["text"])
    if not text:
        return True
    lo, hi = unit["start"] - SEAM_JITTER, unit["end"] + SEAM_JITTER
    near = sorted((o for o in others if o["end"] > lo and o["start"] < hi), key=lambda o: o["start"])
    if LCSseq.similarity(text, "".join(_slice_in(o, lo, hi) for o in near)) >= COVER_SHARE * len(text):
        return True
    span = max(unit["end"] - unit["start"], 1e-3)
    return any(linking.overlap(unit, o) >= 0.5 * span and LCSseq.similarity(text, _slice_in(o, lo, hi)) >= 0.5 * len(text)
               for o in near)


def _events(left: list[dict], right: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    """Group the two windows' segments into utterance events: a left and a right segment are the same event when
    one lies mostly inside the other (>= half of the shorter, both widened by SEAM_JITTER), and events chain
    through such pairs. Each group is
    (its left segments, its right segments); a segment only one window has is a group of one."""
    units = [(0, u) for u in left] + [(1, u) for u in right]
    parent = list(range(len(units)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def padded(u):  # the two windows may time the same utterance up to SEAM_JITTER apart
        return {"start": u["start"] - SEAM_JITTER, "end": u["end"] + SEAM_JITTER}

    for i, (si, ui) in enumerate(units):
        for j in range(i + 1, len(units)):
            sj, uj = units[j]
            short = min(ui["end"] - ui["start"], uj["end"] - uj["start"]) + 2 * SEAM_JITTER
            if si != sj and linking.overlap(padded(ui), padded(uj)) >= 0.5 * short:
                parent[root(i)] = root(j)
    groups: dict[int, tuple[list, list]] = {}
    for i, (side, u) in enumerate(units):
        groups.setdefault(root(i), ([], []))[side].append(u)
    return list(groups.values())


def reconcile(left: list[dict], right: list[dict], left_all: list[dict], right_all: list[dict], cut: float,
              region: tuple[float, float]) -> None:
    """Make a seam keep each utterance once, in place. Segment arms cut the same speech into differently shaped
    segments per window, so the midpoint rule alone can drop both copies or keep both. In the windows' overlap
    `region`, segments are grouped into events (`_events`) and each event is taken whole from one window --
    a window's own segments neither repeat nor miss speech, so taking all of one side's is exact: the left window
    for an event near the region's start (the right window begins there), the right for one near its end, else
    the side of the cut the event's middle is on. A segment only one window has is kept. An event chained across
    the whole region (continuous overlapping speech) falls back to matching by text (`_text_reconcile`)."""
    lo, hi = region
    in_region = lambda u: u["end"] > lo and u["start"] < hi  # noqa: E731
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
    """Fallback for one long event: within HANDOVER_SLACK of the cut, a kept right segment whose utterance the kept
    left side already has is dropped, then a dropped segment whose utterance neither kept side has is restored."""
    near = lambda u: u["end"] > cut - HANDOVER_SLACK and u["start"] < cut + HANDOVER_SLACK  # noqa: E731
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


def handover(units: list[dict], lo: float, hi: float, target: float) -> float:
    """Where two overlapping windows hand over: in a gap that no unit of either window spans, within
    HANDOVER_SLACK of `target` and inside [lo, hi], closest to `target`. Cutting where a unit crosses would split
    the same sentence by two different segment shapes -- each window's copy can then fall on the side that is
    discarded (MOSS segments differ in length between windows). The slack keeps the cut away from window
    edges, where models transcribe worst; without a gap near `target` (continuous speech), `target` itself."""
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


def tail_drops(item: str, parts: list[tuple[float, float, dict]], min_speech: float = 10.0) -> list[dict]:
    """Windows whose output ends while production's VAD still hears >= `min_speech` s of speech after it."""
    vad_path = RUNS / item / f"{item}-vad.json"
    if not vad_path.exists():
        return []
    vad = json.loads(vad_path.read_text(encoding="utf-8"))["segments"]
    out = []
    for k, (s, e, arm) in enumerate(parts):
        units = arm["words"] or arm["segments"]
        last = max((u["end"] for u in units), default=s)
        speech = sum(max(0.0, min(v["end"], e) - max(v["start"], last)) for v in vad)
        if speech >= min_speech:
            out.append({"window": k, "last_word": round(last, 1), "window_end": round(e, 1),
                        "vad_speech_after": round(speech, 1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("item")
    ap.add_argument("arm", choices=sorted(n for n, x in ARMS.items() if x.kind == "model"))
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--no-link", action="store_true")
    ap.add_argument("--span", nargs=2, type=float, metavar=("A", "B"))
    ap.add_argument("--rep", type=int, default=1, help="repeat number: 2 and up call the model again into a separate output")
    a = ap.parse_args()
    src = source_audio(a.item) if a.mix else vocal_track(a.item)
    tag = "mix" if a.mix else "vocal"
    arm = ARMS[a.arm]
    duration = probe_duration(src)
    lo, hi = (a.span[0], min(a.span[1], duration)) if a.span else (0.0, duration)
    base = stem(a.arm, span=(lo, hi) if a.span else None, tag=tag, rep=a.rep)
    wins = windows(lo, hi, arm.window, arm.overlap)
    paths = (run_moss(a.item, base, wins, src, tag) if arm.module == "moss_run"
             else [run_window(a.item, a.arm, base, k, s, e, src, tag) for k, (s, e) in enumerate(wins)])
    parts = [(s, e, json.loads(p.read_text(encoding="utf-8"))) for (s, e), p in zip(wins, paths)]
    merged = stitch(parts, do_link=not a.no_link)
    dropped = tail_drops(a.item, parts)
    name = base + ("-nolink" if a.no_link else "")
    out = EPISODES / a.item / f"{name}.json"
    save({"model": a.arm, "config": {"windows": wins, "overlap": arm.overlap, "linked": not a.no_link, "input": tag,
                                     "span": [lo, hi], "rep": a.rep},
          "audio": str(src), "offset": 0.0, "duration": duration,
          "wall": round(sum(p[2].get("wall", 0) for p in parts), 2), "tail_drops": dropped, **merged}, out)
    units = merged["words"] or merged["segments"]
    labels = Counter(w["speaker"] for w in units)
    print(f"{a.item} {name}: {len(wins)} windows, {len(units)} units, {len(labels)} labels, "
          f"tail drops {dropped} -> {out}")


if __name__ == "__main__":
    main()
