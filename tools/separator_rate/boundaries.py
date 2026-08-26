"""How close does each arm's cue timing sit to the human-timed subtitle?

For every human cue boundary, the distance to the nearest boundary the arm
produced. Independent of the words, so it separates "when did someone speak"
from "what did they say" -- the two came apart in the 95-127s window.
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
GOLD_DIR = DATA_ROOT / ("data/manually-refined-subs/四月一日")
GOLD = {
    "BV1kYLR6AEXv": "看立绘.srt",
    "BV1UBjq6fEgb": "新皮肤(2).srt",
    "BV1ySjz6FEzD": "看莉奈娅pv.srt",
    "BV1cqLR6hEp3": "最后的遗产(1).srt",
    "BV1dwjP6LECU": "布伦妮pv(1).srt",
}
ARMS = ["44100", "null", "32000", "29400", "22050", "16000"]


def raw_srt(material: str, rate: str) -> Path:
    if rate == "null":
        return WORK / (f"null/{material}/{material}-raw.srt")
    if material == "BV1cqLR6hEp3" and rate in ("44100", "22050", "16000"):
        folder, stem = {"44100": ("pipeA", "A"), "22050": ("pipeC", "C"),
                        "16000": ("pipeB", "B")}[rate]
        return WORK / (f"{folder}/{stem}-raw.srt")
    return WORK / (f"down/{material}-{rate}/{material}-raw.srt")


def boundaries(path: Path):
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

        out.extend([sec(found[0]), sec(found[1])])
    return np.array(sorted(out))


print("distance from each human cue boundary to the nearest boundary the arm produced")
print(f"{'material':>14} | " + " ".join(f"{a:>16}" for a in ARMS))
print(f"{'':>14} | " + " ".join(f"{'median   <0.3s':>16}" for _ in ARMS))
print("-" * 96)
pool = {arm: [] for arm in ARMS}
for material, name in GOLD.items():
    gold = boundaries(GOLD_DIR / name)
    cells = []
    for arm in ARMS:
        path = raw_srt(material, arm)
        if not path.exists():
            cells.append(f"{'--':>16}")
            continue
        arm_bounds = boundaries(path)
        idx = np.searchsorted(arm_bounds, gold)
        left = arm_bounds[np.clip(idx - 1, 0, len(arm_bounds) - 1)]
        right = arm_bounds[np.clip(idx, 0, len(arm_bounds) - 1)]
        dist = np.minimum(np.abs(gold - left), np.abs(gold - right))
        pool[arm].append(dist)
        cells.append(f"{np.median(dist):>7.3f}s {100 * (dist < 0.3).mean():>5.1f}%")
    print(f"{material:>14} | " + " ".join(cells))
print("-" * 96)
cells = []
for arm in ARMS:
    if not pool[arm]:
        cells.append(f"{'--':>16}")
        continue
    dist = np.concatenate(pool[arm])
    cells.append(f"{np.median(dist):>7.3f}s {100 * (dist < 0.3).mean():>5.1f}%")
print(f"{'all':>14} | " + " ".join(cells))
