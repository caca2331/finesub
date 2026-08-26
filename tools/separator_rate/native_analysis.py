"""The rerun under the condition production actually runs: native-rate stereo.

Reports the requested 44100 vs 22050 comparison with its null control, and --
from data already on disk -- what the stale 16 kHz mono .ogg cache was costing
at 44.1 kHz, which is the question the correction opened up.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Filled in by tools/separator_rate/_data.py's contract: the checkout that holds
# the gitignored assets/ data/ out/ trees (a worktree does not have its own).
DATA_ROOT = _Path(_os.environ.get("FINESUB_DATA_ROOT", ".")).resolve()
WORK = _Path(_os.environ.get("FINESUB_RATE_WORK", "out/separator-rate"))

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
GRID = 0.01
REF = DATA_ROOT / "out" / "reference"
GOLD_DIR = DATA_ROOT / ("data/manually-refined-subs/四月一日")
GOLD = {
    "BV1kYLR6AEXv": "看立绘.srt",
    "BV1UBjq6fEgb": "新皮肤(2).srt",
    "BV1ySjz6FEzD": "看莉奈娅pv.srt",
    "BV1cqLR6hEp3": "最后的遗产(1).srt",
    "BV1dwjP6LECU": "布伦妮pv(1).srt",
}
MATERIALS = list(GOLD)


def native(material: str, arm: str, suffix: str) -> Path:
    return WORK / (f"native-down/{material}-{arm}/{material}-{suffix}")


def ogg(material: str, arm: str, suffix: str) -> Path:
    if arm == "null":
        return WORK / (f"null/{material}/{material}-{suffix}")
    if material == "BV1cqLR6hEp3":
        folder, stem = {"44100": ("pipeA", "A"), "22050": ("pipeC", "C")}[arm]
        return WORK / (f"{folder}/{stem}-{suffix}")
    return WORK / (f"down/{material}-{arm}/{material}-{suffix}")


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


def norm(items):
    text = unicodedata.normalize("NFKC", "".join(t for *_, t in items))
    return re.sub(r"[\s、。，．,.!?！？「」『』…・ー―-]+", "", text)


def lev(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def vad(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted((float(x["start"]), float(x["end"]))
                  for x in data["vad_timeline"]["intervals"])


def mask(spans, cells):
    out = np.zeros(cells, dtype=bool)
    for start, end, *_ in spans:
        lo, hi = max(0, int(start / GRID)), min(cells, int(round(end / GRID)))
        if hi > lo:
            out[lo:hi] = True
    return out


def bounds(items):
    return np.array(sorted([v for s, e, *_ in items for v in (s, e)]))


print("=== 1. 原生输入：44100 / 22050 / 空白对照（Δ CER vs 原生 44100）")
print(f"{'material':>14} | {'null':>8} {'22050':>8}")
print("-" * 36)
acc = {"null": [], "22050": []}
for material in MATERIALS:
    ref = norm(blocks(REF / material / f"{material}-corrected.srt"))
    base = lev(ref, norm(blocks(native(material, "44100", "raw.srt")))) / len(ref)
    row = {}
    for arm in ("null", "22050"):
        row[arm] = lev(ref, norm(blocks(native(material, arm, "raw.srt")))) / len(ref) - base
        acc[arm].append(row[arm])
    print(f"{material:>14} | {row['null']:>+8.3f} {row['22050']:>+8.3f}")
print("-" * 36)
print(f"{'mean':>14} | " + " ".join(
    f"{sum(acc[a]) / len(acc[a]):>+8.3f}" for a in ("null", "22050")))
print(f"{'positive':>14} | " + " ".join(
    f"{sum(1 for v in acc[a] if v > 0)}/5".rjust(8) for a in ("null", "22050")))

print("\n=== 2. 旧 ogg 缓存 vs 原生输入，都在 44100（CER vs 同一参照）")
print(f"{'material':>14} | {'ogg 44100':>10} {'原生 44100':>11} {'差':>8} | "
      f"{'ogg null':>9} {'原生 null':>10}")
print("-" * 70)
deltas = []
for material in MATERIALS:
    ref = norm(blocks(REF / material / f"{material}-corrected.srt"))
    o = lev(ref, norm(blocks(ogg(material, "44100", "raw.srt")))) / len(ref)
    n = lev(ref, norm(blocks(native(material, "44100", "raw.srt")))) / len(ref)
    on = lev(ref, norm(blocks(ogg(material, "null", "raw.srt")))) / len(ref)
    nn = lev(ref, norm(blocks(native(material, "null", "raw.srt")))) / len(ref)
    deltas.append(n - o)
    print(f"{material:>14} | {o:>10.3f} {n:>11.3f} {n - o:>+8.3f} | "
          f"{on:>9.3f} {nn:>10.3f}")
print("-" * 70)
print(f"{'mean Δ':>14} | {'':>10} {'':>11} {sum(deltas) / len(deltas):>+8.3f}")

print("\n=== 3. 原生输入下的 VAD 裁定（人工时间轴为裁判）")
print(f"{'material':>14} {'arm':>8} | {'missed':>7} {'real%':>6} | {'extra':>7} {'real%':>6} "
      f"| {'边界中位':>8}")
print("-" * 76)
pool = {"null": [], "22050": []}
bpool = {"44100": [], "null": [], "22050": []}
for material in MATERIALS:
    gold = blocks(GOLD_DIR / GOLD[material])
    cells = int(max(e for _, e, _ in gold) / GRID) + 12000
    gold_mask = mask(gold, cells)
    gb = bounds(gold)
    base_mask = mask(vad(native(material, "44100", "aligned.json")), cells)
    for arm in ("44100", "null", "22050"):
        ab = bounds(blocks(native(material, arm, "raw.srt")))
        idx = np.searchsorted(ab, gb)
        dist = np.minimum(np.abs(gb - ab[np.clip(idx - 1, 0, len(ab) - 1)]),
                          np.abs(gb - ab[np.clip(idx, 0, len(ab) - 1)]))
        bpool[arm].append(dist)
        if arm == "44100":
            print(f"{material:>14} {arm:>8} | {'--':>7} {'--':>6} | {'--':>7} {'--':>6} "
                  f"| {np.median(dist):>7.3f}s")
            continue
        arm_mask = mask(vad(native(material, arm, "aligned.json")), cells)
        missed, extra = base_mask & ~arm_mask, arm_mask & ~base_mask
        mr = 100 * (missed & gold_mask).sum() / max(1, missed.sum())
        er = 100 * (extra & gold_mask).sum() / max(1, extra.sum())
        pool[arm].append((missed.sum() * GRID, mr, extra.sum() * GRID, er))
        print(f"{'':>14} {arm:>8} | {missed.sum() * GRID:>6.1f}s {mr:>5.1f}% | "
              f"{extra.sum() * GRID:>6.1f}s {er:>5.1f}% | {np.median(dist):>7.3f}s")
print("-" * 76)
for arm in ("null", "22050"):
    miss = sum(v[0] for v in pool[arm])
    ex = sum(v[2] for v in pool[arm])
    mr = sum(v[0] * v[1] for v in pool[arm]) / max(1e-9, miss)
    er = sum(v[2] * v[3] for v in pool[arm]) / max(1e-9, ex)
    print(f"{'total':>14} {arm:>8} | {miss:>6.1f}s {mr:>5.1f}% | {ex:>6.1f}s {er:>5.1f}%")
print()
for arm in ("44100", "null", "22050"):
    d = np.concatenate(bpool[arm])
    print(f"  边界对齐 {arm:>6}: 中位 {np.median(d):.3f}s, <0.3s {100 * (d < 0.3).mean():.1f}%")
