"""A1 decode-batch sweep: speed and the pre-registered acceptance metrics.

Runs the vad-asr stage on finished products (their `-vocal.ogg` and
`-vad.json`, so neither the separator nor VAD is re-run) once per batch size,
then scores every arm against the B=1 arm of the same file with the metrics of
docs/bench-baselines.md 第十 (10.2/10.3): segment text similarity via
`tools/wt_refine_validation.run.edit_similarity` over `match_stable` pairs,
word-start deltas on text-identical words, the rescue counters, and
`asr_align_sec`.

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_asr_decode_batch \\
        --arms 1,4,8,16 --gpu-tier standard --out tmp/bench/a1 \\
        out/ts-baseline-20260831-hq/yingtao out/ts-baseline-20260831-hq/yui ...

Arm order alternates per file (P4: alternate order, medians), the referee is
off so the numbers are the ASR pass alone, and every product's arms are
written beside each other under --out for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finesub.reporting import NullReporter, reporting_to  # noqa: E402
from finesub.speech.recognition import vad_asr_stage  # noqa: E402
from tools.wt_refine_validation.artifact_survey import match_stable  # noqa: E402
from tools.wt_refine_validation.run import edit_similarity  # noqa: E402

RESCUE_KEYS = ("dropped_groups", "abnormal_groups", "beam_rescue_attempted")
PREFETCH_KEYS = ("prefetch_hits", "prefetch_misses", "prefetch_wasted", "prefetch_too_long", "prefetch_decoded")


def run_arm(product: Path, batch: int, out_dir: Path, gpu_tier: str, model: str) -> dict:
    stem = product.name
    vocal = product / f"{stem}-vocal.ogg"
    output = out_dir / f"{stem}-b{batch}-aligned.json"
    vad_copy = out_dir / f"{stem}-b{batch}-aligned-vad.json"
    shutil.copy(product / f"{stem}-vad.json", vad_copy)
    if output.exists():
        output.unlink()
    began = time.perf_counter()
    with reporting_to(NullReporter()):
        vad_asr_stage.run_vad_asr(
            str(vocal),
            output_path=str(output),
            model_name=model,
            device="cuda",
            gpu_tier=gpu_tier,
            asr_decode_batch=batch,
            qwen_verify="off",
        )
    wall = time.perf_counter() - began
    data = json.loads(output.read_text(encoding="utf-8"))
    meta = data["metadata"]["asr_align"]
    phases = meta.get("asr_phases", {})
    recovery = meta.get("recovery", {})
    return {
        "batch": batch,
        "wall_sec": round(wall, 2),
        "asr_align_sec": round(float(meta["timing"]["asr_align_sec"]), 2),
        "decode_sec": round(float(phases.get("asr.decode", {}).get("inclusive_s", 0.0)), 2),
        "decode_batch_sec": round(float(phases.get("asr.decode_batch", {}).get("inclusive_s", 0.0)), 2),
        "prefetch_sec": round(float(phases.get("asr.prefetch", {}).get("inclusive_s", 0.0)), 2),
        "segments": len(data["segments"]),
        "recovery": {k: int(recovery.get(k, 0)) for k in RESCUE_KEYS + PREFETCH_KEYS},
        "output": str(output),
    }


def score(base_path: str, arm_path: str) -> dict:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))["segments"]
    arm = json.loads(Path(arm_path).read_text(encoding="utf-8"))["segments"]
    sims: list[float] = []
    deltas: list[float] = []
    unmatched = 0
    for seg in base:
        match = match_stable(seg, arm)
        if match is None:
            unmatched += 1
            sims.append(0.0)
            continue
        sim = edit_similarity(str(seg.get("text", "")), str(match.get("text", "")))
        sims.append(1.0 if sim is None else float(sim))
        if sim == 1.0:
            for wa, wb in zip(seg.get("words", []), match.get("words", [])):
                if wa.get("word") == wb.get("word"):
                    deltas.append((float(wb["start"]) - float(wa["start"])) * 1000.0)
    return {
        "segments": len(base),
        "unmatched": unmatched,
        "sims": sims,
        "deltas": deltas,
    }


def summarise(sims: list[float], deltas: list[float]) -> dict:
    low = sum(1 for s in sims if s < 0.98)
    out = {
        "sim_median": round(statistics.median(sims), 4) if sims else None,
        "below_098_pct": round(100.0 * low / len(sims), 3) if sims else None,
        "words": len(deltas),
    }
    if deltas:
        over = sum(1 for d in deltas if abs(d) > 20.0)
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        out.update(
            {
                "over_20ms_pct": round(100.0 * over / len(deltas), 3),
                "mean_ms": round(statistics.mean(deltas), 3),
                "pos_neg_ratio": round(pos / neg, 3) if neg else None,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("products", nargs="+", help="out/<stem> directories with -vocal.ogg and -vad.json")
    parser.add_argument("--arms", default="1,4,8,16")
    parser.add_argument("--gpu-tier", default="standard")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--out", default="tmp/bench/a1")
    args = parser.parse_args()
    arms = [int(x) for x in args.arms.split(",")]
    if 1 not in arms:
        arms = [1] + arms
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"arms": arms, "gpu_tier": args.gpu_tier, "files": {}}
    pooled: dict[int, dict] = {b: {"sims": [], "deltas": [], "unmatched": 0, "segments": 0} for b in arms if b != 1}
    for index, item in enumerate(args.products):
        product = Path(item)
        stem = product.name
        order = list(arms) if index % 2 == 0 else list(reversed(arms))
        runs = {}
        for batch in order:
            runs[batch] = run_arm(product, batch, out_dir, args.gpu_tier, args.model)
            r = runs[batch]
            print(
                f"{stem:14} B={batch:2} align {r['asr_align_sec']:6.1f}s decode {r['decode_sec']:5.1f} "
                f"batch {r['decode_batch_sec']:4.1f} prefetch {r['prefetch_sec']:4.1f} | "
                f"{ {k: v for k, v in r['recovery'].items() if v} }",
                flush=True,
            )
        base = runs[1]
        entry = {"runs": runs, "scores": {}}
        for batch in arms:
            if batch == 1:
                continue
            s = score(base["output"], runs[batch]["output"])
            entry["scores"][batch] = {
                **summarise(s["sims"], s["deltas"]),
                "unmatched": s["unmatched"],
                "speedup": round(base["asr_align_sec"] / runs[batch]["asr_align_sec"], 3),
            }
            pooled[batch]["sims"] += s["sims"]
            pooled[batch]["deltas"] += s["deltas"]
            pooled[batch]["unmatched"] += s["unmatched"]
            pooled[batch]["segments"] += s["segments"]
            print(f"   B={batch:2} vs B=1: {entry['scores'][batch]}", flush=True)
        report["files"][stem] = entry
    print("\n== pooled over files ==")
    summary = {}
    for batch, pool in pooled.items():
        speeds = [report["files"][f]["scores"][batch]["speedup"] for f in report["files"]]
        rescue = {
            k: (
                sum(report["files"][f]["runs"][batch]["recovery"][k] for f in report["files"]),
                sum(report["files"][f]["runs"][1]["recovery"][k] for f in report["files"]),
            )
            for k in RESCUE_KEYS
        }
        total_base = sum(report["files"][f]["runs"][1]["asr_align_sec"] for f in report["files"])
        total_arm = sum(report["files"][f]["runs"][batch]["asr_align_sec"] for f in report["files"])
        summary[batch] = {
            **summarise(pool["sims"], pool["deltas"]),
            "unmatched": pool["unmatched"],
            "segments": pool["segments"],
            "speedup_median": round(statistics.median(speeds), 3),
            "speedup_total": round(total_base / total_arm, 3),
            "rescue(arm, base)": rescue,
        }
        print(f"B={batch:2}: {summary[batch]}", flush=True)
    report["summary"] = summary
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
