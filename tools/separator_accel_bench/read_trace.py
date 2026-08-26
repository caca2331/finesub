"""Read a utilisation trace against the run's own phase boundaries.

The separator's log gives one hard timestamp -- when load_model finished -- and
the tool reports every phase duration after that, so the whole timeline can be
reconstructed from a single anchor.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Where the benchmark wrote its result JSONs and traces.

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

label, anchor_str = sys.argv[1], sys.argv[2]
OUT = _Path(_os.environ.get("FINESUB_ACCEL_WORK", "out/separator-accel-bench"))
ct = json.loads((OUT / f"{label}.json").read_text(encoding="utf-8"))["compile_timing"]

rows = []
for line in (OUT / f"{label}-gpu.csv").read_text(encoding="utf-8").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3 or not parts[1].endswith("%"):
        continue
    stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
    rows.append((stamp, int(parts[1].rstrip(" %")), int(parts[2].split()[0])))
if not rows:
    raise SystemExit("no samples parsed")

anchor = datetime.strptime(anchor_str, "%H:%M:%S,%f").replace(
    year=rows[0][0].year, month=rows[0][0].month, day=rows[0][0].day)
times = np.array([(s - anchor).total_seconds() for s, _, _ in rows])
util = np.array([u for _, u, _ in rows])
mem = np.array([m for _, _, m in rows])

# Sequential from the load_model anchor.
wrap = ct["compile_wrapper_sec"] + ct["artifact_load_and_inject_sec"]
w1 = ct["warmup_first_forward_sec"]
w2 = ct["warmup_reused_forward_sec"] or 0.0
phases = [
    ("包装/产物加载", 0.0, wrap),
    ("预热 forward #1（含恢复）", wrap, wrap + w1),
    ("预热 forward #2（纯计算）", wrap + w1, wrap + w1 + w2),
    ("分离阶段（含首块）", wrap + w1 + w2, times.max()),
]
print(f"\n### {label}   （anchor = load_model 完成 {anchor_str}）")
print(f"{'阶段':<26}{'时长':>8}{'GPU 均值':>10}{'GPU 中位':>10}{'>50% 采样':>11}{'显存增量':>10}")
print("-" * 76)
base_mem = mem[times < 0].max() if (times < 0).any() else mem[0]
for name, lo, hi in phases:
    sel = (times >= lo) & (times < hi)
    if sel.sum() == 0:
        print(f"{name:<26}{hi - lo:>7.1f}s   (无采样)")
        continue
    print(f"{name:<26}{hi - lo:>7.1f}s{util[sel].mean():>9.0f}%"
          f"{np.median(util[sel]):>9.0f}%{100 * (util[sel] > 50).mean():>10.0f}%"
          f"{mem[sel].max() - base_mem:>8} MiB")
