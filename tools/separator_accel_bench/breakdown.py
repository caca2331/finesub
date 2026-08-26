"""Break each arm's wall time into load / compile-restore / forward / rest."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# Where the benchmark wrote its result JSONs and traces.

import json
from pathlib import Path

OUT = _Path(_os.environ.get("FINESUB_ACCEL_WORK", "out/separator-accel-bench"))
GROUPS = [
    ("270s / 4GB profile", ["short-eager", "short-jit-warm", "short-aoti",
                            "short-aoti-prod", "short-jit-cold"]),
    ("2015s / 8GB profile", ["long-eager", "long-jit-warm", "long-aoti",
                             "long-aoti-prod", "long-jit-cold"]),
]
LABEL = {
    "short-eager": "eager", "short-jit-warm": "torch.compile 热",
    "short-aoti": "AOTI", "short-aoti-prod": "AOTI（生产路径）",
    "short-jit-cold": "(torch.compile 冷)",
    "long-eager": "eager", "long-jit-warm": "torch.compile 热",
    "long-aoti": "AOTI", "long-aoti-prod": "AOTI（生产路径）",
    "long-jit-cold": "(torch.compile 冷)",
}


def load(name: str):
    path = OUT / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing benchmark result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


for title, names in GROUPS:
    rows = [(name, load(name)) for name in names]
    base = next((d for n, d in rows if n.endswith("eager")), None)
    workers = rows[0][1]["separator"].get("effective") or 1
    print(f"\n### {title}")
    print(f"（worker={workers}，torch {rows[0][1]['torch']}，"
          f"cuda {rows[0][1]['cuda']}，{rows[0][1].get('gpu')}）\n")
    header = (f"{'档':<20} {'wall':>8} {'相对eager':>9} | {'模型加载':>8} "
              f"{'编译/恢复':>9} {'forward墙钟':>11} {'其余':>8} | "
              f"{'块数':>5} {'forward中位':>11} {'peak resv':>9}")
    print(header)
    print("-" * (len(header) - 8))
    for name, d in rows:
        ct = d.get("compile_timing", {})
        wall = d["elapsed_sec"]
        build = ct.get("separator_build_sec") or 0.0
        restore = ct.get("compile_or_restore_total_estimate_sec")
        if restore is None:
            restore = (ct.get("compile_wrapper_sec", 0.0)
                       + ct.get("artifact_load_and_inject_sec", 0.0))
        sep = (ct.get("phase_forward_summaries") or {}).get("separation") or {}
        fwd_total = sep.get("total_sec", 0.0)
        count = sep.get("count", 0)
        median = sep.get("median_sec", 0.0)
        # forward runs concurrently across workers, so the summed CPU figure is
        # not what occupies the wall clock -- divide before subtracting.
        per_worker = fwd_total / max(1, workers)
        rest = wall - build - restore - per_worker
        rel = f"{base['elapsed_sec'] / wall:.3f}x" if base else "--"
        peak = d.get("peak_reserved_bytes", 0) / 2**30
        print(f"{LABEL[name]:<20} {wall:>7.2f}s {rel:>9} | {build:>7.2f}s "
              f"{restore:>8.2f}s {per_worker:>10.2f}s {rest:>7.2f}s | "
              f"{count:>5} {median * 1000:>10.1f}ms {peak:>8.2f}G")

print("\n口径：wall 已扣除 --probe-compile-timing 的额外探针。"
      "「编译/恢复」= 包装或产物加载 + 预热首次与复用之差 + 分离首块与稳态之差。"
      "\n「forward墙钟」= 各 worker forward 之和 / worker 数（forward 并发，累加值不占墙钟）。"
      "「其余」= wall 减去前三项：解码、CPU overlap-add、写盘。「块数」为全部 worker 合计。")
