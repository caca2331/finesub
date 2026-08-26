"""Count subtitle cues that land where the human timeline says nobody spoke.

A cue whose span barely touches any human cue is the visible end of the
leakage story: VAD admitted non-speech, Whisper wrote something for it. Judged
against the human-timed subtitle, so it is independent of every arm.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Filled in by tools/separator_rate/_data.py's contract: the checkout that holds
# the gitignored assets/ data/ out/ trees (a worktree does not have its own).
DATA_ROOT = _Path(_os.environ.get("FINESUB_DATA_ROOT", ".")).resolve()
WORK = _Path(_os.environ.get("FINESUB_RATE_WORK", "out/separator-rate"))

import re
from pathlib import Path

import numpy as np

_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
GRID = 0.01
GOLD_DIR = DATA_ROOT / ("data/manually-refined-subs/四月一日")
GOLD = {
    "BV1kYLR6AEXv": "看立绘.srt",
    "BV1UBjq6fEgb": "新皮肤(2).srt",
    "BV1ySjz6FEzD": "看莉奈娅pv.srt",
    "BV1cqLR6hEp3": "最后的遗产(1).srt",
    "BV1dwjP6LECU": "布伦妮pv(1).srt",
}
RATES = ["32000", "29400", "22050", "16000"]
OVERLAP_FLOOR = 0.25  # a cue counts as grounded if a quarter of it is inside gold


def raw_srt(material: str, rate: str) -> Path:
    if material == "BV1cqLR6hEp3" and rate in ("44100", "22050", "16000"):
        folder, stem = {"44100": ("pipeA", "A"), "22050": ("pipeC", "C"),
                        "16000": ("pipeB", "B")}[rate]
        return WORK / (f"{folder}/{stem}-raw.srt")
    return WORK / (f"down/{material}-{rate}/{material}-raw.srt")


def blocks(path: Path):
    out = []
    for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [x for x in chunk.splitlines() if x.strip()]
        if len(lines) < 2:
            continue
        found = _TIME_RE.findall(lines[1])
        if len(found) != 2:
            continue

        def sec(item):
            h, m, s, ms = (int(v) for v in item)
            return h * 3600 + m * 60 + s + ms / 1000.0

        out.append((sec(found[0]), sec(found[1]), " ".join(lines[2:])))
    return out


def mask(spans, cells):
    out = np.zeros(cells, dtype=bool)
    for start, end, *_ in spans:
        lo = max(0, int(start / GRID))
        hi = min(cells, int(round(end / GRID)))
        if hi > lo:
            out[lo:hi] = True
    return out


print(f"{'material':>14} | " + " ".join(f"{r:>12}" for r in ["44100", "null"] + RATES))
print("-" * 92)
totals = {}
for material in GOLD:
    gold = blocks(GOLD_DIR / GOLD[material])
    cells = int(max(end for _, end, _ in gold) / GRID) + 12000
    gold_mask = mask(gold, cells)
    cells_out = []
    for label in ["44100", "null"] + RATES:
        path = (WORK / (f"null/{material}/{material}-raw.srt")
                if label == "null" else raw_srt(material, label))
        if not path.exists():
            cells_out.append(f"{'--':>12}")
            continue
        cues = blocks(path)
        ghosts = []
        for start, end, text in cues:
            lo, hi = max(0, int(start / GRID)), min(cells, int(round(end / GRID)))
            if hi <= lo:
                continue
            if gold_mask[lo:hi].mean() < OVERLAP_FLOOR:
                ghosts.append((start, end, text))
        totals.setdefault(label, [0, 0])
        totals[label][0] += len(ghosts)
        totals[label][1] += len(cues)
        cells_out.append(f"{len(ghosts):>4}/{len(cues):<4}  ")
        if material == "BV1cqLR6hEp3" and label in ("44100", "22050"):
            Path(f"tmp/ghosts-{label}.txt").write_text(
                "\n".join(f"{s:7.1f}-{e:7.1f}  {t}" for s, e, t in ghosts),
                encoding="utf-8",
            )
    print(f"{material:>14} | " + " ".join(cells_out))
print("-" * 92)
print(f"{'total':>14} | " + " ".join(
    f"{totals[k][0]:>4}/{totals[k][1]:<4}  " if k in totals else f"{'--':>12}"
    for k in ["44100", "null"] + RATES))
print(f"\ncues whose span is <{OVERLAP_FLOOR:.0%} inside the human timeline")
