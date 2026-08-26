"""Headline throughput (uninstrumented) with the fixed costs pulled out."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Where the benchmark wrote its result JSONs and traces.

import json
from pathlib import Path

OUT = _Path(_os.environ.get("FINESUB_ACCEL_WORK", "out/separator-accel-bench"))
GROUPS = [
    ("270s 素材 / 4GB profile / 1 worker", 269.93,
     [("eager", "c-short-eager", "short-eager"),
      ("torch.compile（热缓存）", "c-short-jit", "short-jit-warm"),
      ("AOTI", "c-short-aoti", "short-aoti")]),
    ("2015s 素材 / 8GB profile / 2 worker", 2014.75,
     [("eager", "c-long-eager", "long-eager"),
      ("torch.compile（热缓存）", "c-long-jit", "long-jit-warm"),
      ("AOTI", "c-long-aoti", "long-aoti")]),
]


def load(name):
    path = OUT / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing benchmark result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


for title, duration, arms in GROUPS:
    print(f"\n### {title}")
    header = (f"{'档':<22} {'wall':>8} {'相对eager':>9} {'实时倍率':>8} | "
              f"{'模型加载':>8} {'编译/恢复':>9} {'去掉固定开销':>12} {'相对eager':>9} | "
              f"{'forward中位':>11}")
    print(header)
    print("-" * (len(header) - 16))
    rows = []
    for label, clean_name, timed_name in arms:
        clean, timed = load(clean_name), load(timed_name)
        wall = clean["elapsed_sec"]
        build = (clean.get("compile_timing") or {}).get("separator_build_sec") or 0.0
        tc = timed.get("compile_timing", {})
        restore = tc.get("compile_or_restore_total_estimate_sec") or 0.0
        sep = (tc.get("phase_forward_summaries") or {}).get("separation") or {}
        rows.append((label, wall, build, restore, sep.get("median_sec", 0.0)))
    base_wall = rows[0][1]
    base_net = rows[0][1] - rows[0][2] - rows[0][3]
    for label, wall, build, restore, median in rows:
        net = wall - build - restore
        print(f"{label:<22} {wall:>7.2f}s {base_wall / wall:>8.3f}x "
              f"{duration / wall:>7.1f}x | {build:>7.2f}s {restore:>8.2f}s "
              f"{net:>11.2f}s {base_net / net:>8.3f}x | {median * 1000:>10.1f}ms")

print("\n口径：wall 取**无 forward 仪器**的运行（每次 forward 一个 CUDA 同步会把 H2D 与"
      "\n      CPU overlap-add 串行化，短素材上值 ~3.3s）。「模型加载」= Separator 构造 +"
      "\n      load_model，就地实测。「编译/恢复」取自同配置的带仪器运行，AOTI 是产物加载与"
      "\n      常量注入，torch.compile 是热缓存下每进程仍要重建的 dynamo/guard。"
      "\n      「去掉固定开销」= wall 减这两项，即这一档真正的稳态吞吐。")
