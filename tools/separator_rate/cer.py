"""CER penalty of every arm against the 44.1k baseline, with the null control."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Filled in by tools/separator_rate/_data.py's contract: the checkout that holds
# the gitignored assets/ data/ out/ trees (a worktree does not have its own).
DATA_ROOT = _Path(_os.environ.get("FINESUB_DATA_ROOT", ".")).resolve()
WORK = _Path(_os.environ.get("FINESUB_RATE_WORK", "out/separator-rate"))

import re
import unicodedata
from pathlib import Path

_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
REF = DATA_ROOT / "out" / "reference"
RATES = ["32000", "29400", "22050", "16000"]
MATERIALS = ["BV1cqLR6hEp3", "BV1kYLR6AEXv", "BV1UBjq6fEgb", "BV1ySjz6FEzD", "BV1dwjP6LECU"]


def raw_srt(material: str, rate: str) -> Path:
    if material == "BV1cqLR6hEp3" and rate in ("44100", "22050", "16000"):
        folder, stem = {"44100": ("pipeA", "A"), "22050": ("pipeC", "C"),
                        "16000": ("pipeB", "B")}[rate]
        return WORK / (f"{folder}/{stem}-raw.srt")
    return WORK / (f"down/{material}-{rate}/{material}-raw.srt")


def cues(path):
    out = []
    for chunk in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8").strip()):
        lines = [x for x in chunk.splitlines() if x.strip()]
        if len(lines) >= 2 and len(_TIME_RE.findall(lines[1])) == 2:
            out.append(" ".join(lines[2:]))
    return out


def norm(parts):
    text = unicodedata.normalize("NFKC", "".join(parts))
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


print("Δ CER vs the 44.1k baseline arm (per-material reference)")
print(f"{'material':>14} | {'null':>7} {'32000':>7} {'22050':>7} {'16000':>7}")
print("-" * 52)
acc = {"null": [], **{rate: [] for rate in RATES}}
for material in MATERIALS:
    ref = norm(cues(REF / material / f"{material}-corrected.srt"))
    base = lev(ref, norm(cues(raw_srt(material, "44100")))) / len(ref)
    row = {"null": lev(ref, norm(cues(
        WORK / f"null/{material}/{material}-raw.srt"))) / len(ref) - base}
    for rate in RATES:
        path = raw_srt(material, rate)
        row[rate] = (lev(ref, norm(cues(path))) / len(ref) - base
                     if path.exists() else None)
    for key, value in row.items():
        if value is not None:
            acc[key].append(value)
    cells = " ".join(f"{row[k]:>+7.3f}" if row[k] is not None else f"{'--':>7}"
                     for k in ["null"] + RATES)
    print(f"{material:>14} | {cells}")
print("-" * 52)
mean = " ".join(f"{sum(acc[k]) / len(acc[k]):>+7.3f}" if acc[k] else f"{'--':>7}"
                for k in ["null"] + RATES)
sign = " ".join(f"{sum(1 for v in acc[k] if v > 0)}/{len(acc[k]):>5}" if acc[k]
                else f"{'--':>7}" for k in ["null"] + RATES)
print(f"{'mean':>14} | {mean}")
print(f"{'positive':>14} | {sign}")
