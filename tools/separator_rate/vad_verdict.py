"""The VAD verdict across all five materials, refereed by the human timelines."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Filled in by tools/separator_rate/_data.py's contract: the checkout that holds
# the gitignored assets/ data/ out/ trees (a worktree does not have its own).
DATA_ROOT = _Path(_os.environ.get("FINESUB_DATA_ROOT", ".")).resolve()
WORK = _Path(_os.environ.get("FINESUB_RATE_WORK", "out/separator-rate"))

import json
import re
from pathlib import Path

import numpy as np

_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
GRID = 0.01
GOLD_DIR = DATA_ROOT / ("data/manually-refined-subs/四月一日")

# Matched by duration; every one of the five is covered.
GOLD = {
    "BV1kYLR6AEXv": "看立绘.srt",
    "BV1UBjq6fEgb": "新皮肤(2).srt",
    "BV1ySjz6FEzD": "看莉奈娅pv.srt",
    "BV1cqLR6hEp3": "最后的遗产(1).srt",
    "BV1dwjP6LECU": "布伦妮pv(1).srt",
}
RATES = ["32000", "29400", "22050", "16000"]
MATERIALS = list(GOLD)


def aligned(material: str, rate: str) -> Path:
    if material == "BV1cqLR6hEp3" and rate in ("44100", "22050", "16000"):
        stem = {"44100": ("pipeA", "A"), "22050": ("pipeC", "C"),
                "16000": ("pipeB", "B")}[rate]
        return WORK / (f"{stem[0]}/{stem[1]}-aligned.json")
    return WORK / (f"down/{material}-{rate}/{material}-aligned.json")


def null_path(material: str) -> Path:
    return WORK / (f"null/{material}/{material}-aligned.json")


def srt_spans(path: Path):
    spans = []
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

        spans.append((sec(found[0]), sec(found[1])))
    return sorted(spans)


def vad_spans(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        (float(x["start"]), float(x["end"]))
        for x in data["vad_timeline"]["intervals"]
    )


def mask(spans, cells):
    out = np.zeros(cells, dtype=bool)
    for start, end in spans:
        lo = max(0, int(start / GRID))
        hi = min(cells, int(round(end / GRID)))
        if hi > lo:
            out[lo:hi] = True
    return out


def long_runs(flags, threshold=1.0):
    idx = np.flatnonzero(np.diff(np.concatenate(([0], flags.view(np.int8), [0]))))
    return sum(1 for a, b in zip(idx[::2], idx[1::2]) if (b - a) * GRID > threshold)


rows = {}
print(f"{'material':>14} {'arm':>10} | {'ivals':>5} {'cov':>7} | "
      f"{'missed':>7} {'real%':>6} {'>1s':>4} | {'extra':>7} {'real%':>6} {'>1s':>4}")
print("-" * 92)
for material in MATERIALS:
    gold = srt_spans(GOLD_DIR / GOLD[material])
    cells = int(max(end for _, end in gold) / GRID) + 12000
    gold_mask = mask(gold, cells)
    base = vad_spans(aligned(material, "44100"))
    base_mask = mask(base, cells)
    hit = 100 * (base_mask & gold_mask).sum() / base_mask.sum()
    print(f"{material:>14} {'44100':>10} | {len(base):>5} {base_mask.sum() * GRID:>6.1f}s | "
          f"{'--':>7} {'--':>6} {'--':>4} | {'--':>7} {'--':>6} {'--':>4}   "
          f"(baseline VAD inside gold: {hit:.1f}%)")
    for label, path in [("null", null_path(material))] + [
        (rate, aligned(material, rate)) for rate in RATES
    ]:
        if not path.exists():
            print(f"{'':>14} {label:>10} |  (missing)")
            continue
        arm = vad_spans(path)
        arm_mask = mask(arm, cells)
        missed = base_mask & ~arm_mask
        extra = arm_mask & ~base_mask
        m_real = 100 * (missed & gold_mask).sum() / max(1, missed.sum())
        e_real = 100 * (extra & gold_mask).sum() / max(1, extra.sum())
        rows.setdefault(label, []).append(
            (missed.sum() * GRID, m_real, extra.sum() * GRID, e_real)
        )
        print(f"{'':>14} {label:>10} | {len(arm):>5} {arm_mask.sum() * GRID:>6.1f}s | "
              f"{missed.sum() * GRID:>6.1f}s {m_real:>5.1f}% {long_runs(missed):>4} | "
              f"{extra.sum() * GRID:>6.1f}s {e_real:>5.1f}% {long_runs(extra):>4}")
    print()

print(f"{'totals over 5 materials':>26} | {'missed':>8} {'real%':>6} | {'extra':>8} {'real%':>6}")
print("-" * 62)
for label, values in rows.items():
    miss = sum(v[0] for v in values)
    extra = sum(v[2] for v in values)
    mr = sum(v[0] * v[1] for v in values) / max(1e-9, miss)
    er = sum(v[2] * v[3] for v in values) / max(1e-9, extra)
    print(f"{label:>26} | {miss:>7.1f}s {mr:>5.1f}% | {extra:>7.1f}s {er:>5.1f}%")
