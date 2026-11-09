"""Readout of the three-pass acceptance runs (docs/reports/2026-10-17-three-pass.md 「验收的指标与门槛」).

    python tools/diarization_explore/accept_3pass_readout.py <run folder>...   # e.g. main-r1 base-judge-r1

Run folders are under tmp/spk-merge/accept/ (`accept_3pass.py`); only the windows every folder finished are read.
Per window and folder, against ruler 1's row truth (seams whose two sides have one clean speaker each):

- merges: seams merged in the final `-annotated.csv`, cross-speaker / same-speaker, out of how many;
- dropped: rows no final line covers whose truth is a main or side speaker's speech (`one` / `multi`), each with who
  dropped it -- pass 1 (`drop` in the judge's sidecar) or the correction -- for a reader to judge which are fillers;
- over 7 s: final lines of two or more sources whose span (first source's start to last source's end) passes 7 s;
- the judge's turns: same-speaker seams the judged speakers call a change, cross-speaker seams they call a change
  and the ones they miss;
- tokens per pass (from each answered call's record): cached input, uncached input, visible output, thinking.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[2]
MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
ACCEPT = MAIN / "tmp/spk-merge/accept"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(WORKTREE / "src"))
import row_split as R  # noqa: E402

from finesub.speech.diarization import judgments as jd  # noqa: E402
from finesub.speech.diarization.product import product  # noqa: E402

items, _ = R.load("1")
TRUTH = {(it["ident"], it["i"]): it["truth"] for it in items}
_INDEX: dict[str, dict] = {}


def truth_rows(ident, segs):
    if ident not in _INDEX:
        p = MAIN / "data/diarization/row-truth/products/acoustic" / ident / f"{ident}-stable.json"
        rows = json.loads(p.read_text(encoding="utf-8"))["segments"]
        _INDEX[ident] = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(rows)}
    index = _INDEX[ident]
    return [TRUTH.get((ident, index.get((round(s["start"], 3), round(s["end"], 3))))) for s in segs]


def ends(t):
    """(speaker starting the row, speaker ending it) when each is one clean person, else None."""
    if t is None:
        return None
    if t["kind"] == "single":
        return (t["people"][0], t["people"][0]) if len(t["people"]) == 1 else None
    if t["kind"] == "split":
        first, last = t["people"][0], t["people"][-1]
        return (first[0], last[0]) if len(first) == 1 and len(last) == 1 else None
    return None


def annotated(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.reader((x for x in f if not x.startswith("#")), delimiter="|") if r and r[0] == "sub"]


def tokens(folder):
    out = {}
    for line in (folder / "llm" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        kind, payload = record["kind"], record["payload"]
        usages = []
        if kind == "speaker_judge_response":
            usages, name = [payload.get("usage") or {}], "judge"
        elif kind == "proofread_response":
            usages, name = [payload.get("usage") or {}], "proofread"
        elif kind == "correction_window_response":
            # one record per answered call, so a run resumed after an interruption counts every call; the
            # end-of-run token_distribution_report holds only the resumed run's own calls
            usages, name = [payload.get("usage") or {}], "correction"
        for usage in usages:
            c = out.setdefault(name, Counter())
            c["cached"] += usage.get("cached_input_tokens", 0)
            c["uncached"] += usage.get("uncached_input_tokens", 0)
            c["output"] += usage.get("output_tokens", 0)
            c["thinking"] += usage.get("thinking_tokens", 0)
            c["calls"] += 1
    return out


def read(folder, key, segs, src):
    ids = jd.segment_ids(segs)
    at = {key_: n for n, key_ in enumerate(ids)}
    rows = annotated(next(folder.glob(f"{key}-annotated.csv")))
    merged, covered, long = {}, set(), []
    for r in rows:
        # a split row's second half (`12b`) and an inserted line (`12i`) stand at their row's position
        pos = [at[p.rstrip("bi")] for p in r[1].split(",") if p.rstrip("bi") in at]
        covered.update(pos)
        for a, b in zip(pos, pos[1:]):
            if a != b:
                merged[a] = b
        if len(pos) > 1:
            span = float(segs[pos[-1]]["end"]) - float(segs[pos[0]]["start"])
            if span > 7.0:
                long.append((r[1], round(span, 1), r[5]))
    c = Counter()
    for n in range(1, len(segs)):
        ea, eb = ends(src[n - 1]), ends(src[n])
        if ea is None or eb is None:
            continue
        c["cross" if ea[1] != eb[0] else "same", "merged" if merged.get(n - 1) == n else "kept"] += 1
    acts = jd.read_acts(folder / "input" / f"{key}-stable.json", segs, shape=jd.SHAPE_TRACKED,
                        warn=lambda m: None) if (folder / "input" / f"{key}-speaker-judge.json").exists() else None
    pass1 = set(acts.dropped) if acts else set()
    dropped = []
    for n, seg in enumerate(segs):
        t = src[n]
        if n in covered or t is None or not (t["kind"] in ("single", "split", "together")):
            continue
        dropped.append((ids[n], "pass1" if ids[n] in pass1 else "correction", round(float(seg["end"]) - float(seg["start"]), 2),
                        seg.get("text", "")))
    return c, long, dropped


def turns(folder, key, segs, src):
    stable = folder / "input" / f"{key}-stable.json"
    judged = jd.read_judgments(stable, segs, warn=lambda m: None)
    speakers = product(segs, judged).speakers() or {}
    ids = jd.segment_ids(segs)
    from finesub.speech.diarization.labels import different_speakers
    c = Counter()
    for n in range(1, len(segs)):
        ea, eb = ends(src[n - 1]), ends(src[n])
        left = f"{ids[n - 1]}b" if f"{ids[n - 1]}b" in speakers else ids[n - 1]  # a split row ends on its half
        if ea is None or eb is None or left not in speakers or ids[n] not in speakers:
            continue
        flagged = different_speakers(speakers[left], speakers[ids[n]])
        c["cross" if ea[1] != eb[0] else "same", "flagged" if flagged else "unflagged"] += 1
    return c


def main(names):
    folders = [ACCEPT / name for name in names]
    keys = sorted(set.intersection(*({p.name for p in f.iterdir() if (p / f"{p.name}-annotated.csv").exists()} for f in folders)))
    for key in keys:
        ident = key.rsplit("-c", 1)[0]
        segs = json.loads((folders[0] / key / "input" / f"{key}-stable.json").read_text(encoding="utf-8"))["segments"]
        src = truth_rows(ident, segs)
        print(f"=== {key}: {len(segs)} rows, {sum(t is not None for t in src)} with truth")
        for name, folder in zip(names, folders):
            c, long, dropped = read(folder / key, key, segs, src)
            t = turns(folder / key, key, segs, src)
            print(f"  {name:16} cross merged {c['cross', 'merged']}/{c['cross', 'merged'] + c['cross', 'kept']}"
                  f"  same merged {c['same', 'merged']}/{c['same', 'merged'] + c['same', 'kept']}"
                  f"  | judge: same flagged {t['same', 'flagged']}/{t['same', 'flagged'] + t['same', 'unflagged']}"
                  f"  cross flagged {t['cross', 'flagged']}/{t['cross', 'flagged'] + t['cross', 'unflagged']}"
                  f"  | over 7 s {len(long)}  | dropped speech {len(dropped)}")
            for item in long:
                print(f"      over 7 s: {item}")
            for item in dropped:
                print(f"      dropped: {item}")
            for phase, u in tokens(folder / key).items():
                print(f"      tokens {phase:10} calls {u['calls']}  cached {u['cached']:>8}  uncached {u['uncached']:>8}"
                      f"  output {u['output']:>7}  thinking {u['thinking']:>7}")


if __name__ == "__main__":
    main(sys.argv[1:])
