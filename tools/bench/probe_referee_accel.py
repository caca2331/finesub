"""Qwen referee efficiency: timing sweep and the decision-level acceptance check.

Five sub-commands (docs/bench-baselines.md 二十一、22.3); `recheck <out/stem>...`
prices a full re-check of every Whisper segment and `calls <out/stem>` shows
the compiled path over consecutive batches (21.6); `bs <out/stem>` sweeps the batch size (22.3):

    # timing + VRAM peaks: batched eager vs compiled, short and long clips
    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_referee_accel sweep \\
        out/yingtao/yingtao-vocal.ogg out/yingtao/yingtao-vad.json

    # acceptance: recompute evidence for finished products and compare the
    # three quantities stabilize reads (veto / phrase ghost / gap recovery)
    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_referee_accel agreement \\
        out/ts-baseline-20260831 out/ts-baseline-20260831-hq

Discipline (bench-baselines.md P4): check `nvidia-smi` first -- an 83% external
load inflated the first sweep 1.5-2.5x; absolute numbers sit beside ratios.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.speech.postprocessing import stabilization as st  # noqa: E402
from finesub.speech.runtime import phase_timing  # noqa: E402
from finesub.speech.verification import qwen_referee as qr  # noqa: E402
from finesub.text import normalized_compact  # noqa: E402


def _peak(label: str) -> None:
    import torch

    print(f"   peak VRAM after {label}: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    torch.cuda.reset_peak_memory_stats()


def _run(label, referee, batch):
    stats: dict = {}
    with phase_timing.collect(into=stats):
        began = time.perf_counter()
        out = referee.transcribe_batch(batch)
        elapsed = time.perf_counter() - began
    phases = {k: round(v.inclusive_s, 2) for k, v in stats.items()}
    shapes = [len(b) for b in qr.plan_batches([len(c) for c in batch])]
    print(f"{label}: {elapsed:6.2f}s phases={phases} batches={shapes}", flush=True)
    _peak(label)
    return out


def sweep(args) -> int:
    vad = json.load(open(args.vad, encoding="utf-8"))["segments"]
    reader = qr._SpanReader(str(args.audio))
    picks = [vad[int((i + 0.5) * len(vad) / 16)] for i in range(16)]
    short = [reader.read(float(p["start"]) - 0.1, float(p["end"]) + 0.1) for p in picks]
    mid8 = [reader.read(float(vad[i * 25 + 5]["start"]), float(vad[i * 25 + 5]["start"]) + 15.0) for i in range(8)]
    long8 = [reader.read(float(vad[i * 25 + 5]["start"]), float(vad[i * 25 + 5]["start"]) + 30.0) for i in range(8)]
    sets = {"16 short": short, "8x15s": mid8, "8x30s": long8}

    eager = qr.QwenReferee(device=args.device, accel="off")
    eager.warm()
    _peak("load")
    base = {name: _run(f"eager {name}", eager, clips) for name, clips in sets.items()}
    eager.close()

    compiled = qr.QwenReferee(device=args.device, accel="on", vram_budget_gib=args.vram)
    compiled.warm()
    _peak("load")
    for name, clips in sets.items():
        _run(f"compiled {name} (first)", compiled, clips)
    for name, clips in sets.items():
        out = _run(f"compiled {name} (again)", compiled, clips)
        same = sum(a == b for a, b in zip(base[name], out))
        print(f"   compiled==eager {name}: {same}/{len(out)}")
    compiled.close()
    return 0


def _decisions(segment, text):
    compact = normalized_compact(text or "")
    phrase = st._closing_phrase_of(str(segment.get("text") or ""))
    return bool(compact), (phrase is not None and phrase not in compact)


def agreement(args) -> int:
    dirs = sorted(
        {os.path.dirname(p) for root in args.roots for p in glob.glob(os.path.join(root, "*", "*-vocal.ogg"))}
    )
    print(len(dirs), "products")
    referees = {
        "batched-eager": qr.QwenReferee(device=args.device, accel="off"),
        "compiled": qr.QwenReferee(device=args.device, accel="on", vram_budget_gib=args.vram),
    }
    if args.eager_only:
        referees.pop("compiled")
    keys = ("suspects", "text", "veto", "ghost", "lang", "gaps", "gap_same")
    totals = {name: dict.fromkeys(keys, 0) for name in referees}
    for d in dirs:
        stem = os.path.basename(d)
        aligned = json.load(open(os.path.join(d, f"{stem}-aligned.json"), encoding="utf-8"))
        segs = aligned["segments"]
        vad = json.load(open(os.path.join(d, f"{stem}-vad.json"), encoding="utf-8"))["segments"]
        suspects = qr.collect_suspect_indices(segs)
        gaps = qr.collect_gaps(vad, segs)
        recorded = aligned["metadata"]["asr_align"].get("qwen_verify", {})
        base_gaps = {(g["start"], g["end"]) for g in recorded.get(qr.GAP_RECOVERY_KEY, [])}
        if not suspects and not gaps:
            continue
        reader = qr._SpanReader(os.path.join(d, f"{stem}-vocal.ogg"))
        clips = [
            reader.read(segs[i]["start"] - qr.SEGMENT_PAD_SEC, segs[i]["end"] + qr.SEGMENT_PAD_SEC)
            for i in suspects
        ]
        clips += [reader.read(s, e) for s, e in gaps]
        usable = [i for i, c in enumerate(clips) if len(c) >= int(0.05 * qr.TARGET_SR)]
        for name, referee in referees.items():
            replies = referee.transcribe_batch([clips[i] for i in usable])
            results = [("", None)] * len(clips)
            for pos, reply in zip(usable, replies):
                results[pos] = reply
            tot = totals[name]
            for pos, i in enumerate(suspects):
                base = segs[i].get(qr.VERIFY_KEY)
                if not isinstance(base, dict):
                    continue
                tot["suspects"] += 1
                text, lang = results[pos]
                base_text = base.get("text") or ""
                tot["text"] += normalized_compact(text) == normalized_compact(base_text)
                bd, nd = _decisions(segs[i], base_text), _decisions(segs[i], text)
                tot["veto"] += bd[0] == nd[0]
                tot["ghost"] += bd[1] == nd[1]
                tot["lang"] += base.get("language") == lang or not (
                    normalized_compact(text) or normalized_compact(base_text)
                )
                if bd != nd:
                    print(
                        f"   DECISION FLIP [{name}] {stem} seg {i} {segs[i]['start']:.1f}s "
                        f"whisper={segs[i]['text'][:30]!r} base={base_text!r} new={text!r}"
                    )
            for pos, (s, e) in enumerate(gaps):
                text, _ = results[len(suspects) + pos]
                tot["gaps"] += 1
                tot["gap_same"] += bool(normalized_compact(text)) == ((round(s, 3), round(e, 3)) in base_gaps)
            print(f"{stem:14} {name:14} suspects={len(suspects):2} gaps={len(gaps):2}", flush=True)
    for name, tot in totals.items():
        n = tot["suspects"]
        print(
            f"== {name}: suspects {n}  text {tot['text']}/{n}  veto {tot['veto']}/{n}  "
            f"ghost {tot['ghost']}/{n}  lang {tot['lang']}/{n}  gaps {tot['gap_same']}/{tot['gaps']}"
        )
    for referee in referees.values():
        referee.close()
    return 0


def recheck(args) -> int:
    """Every Whisper segment of a finished product through the referee: what a
    full re-check would cost (bench-baselines.md 21.6). Eager at B=8/16, then
    the compiled path unless --eager-only."""

    import torch

    for d in args.products:
        stem = os.path.basename(os.path.normpath(d))
        aligned = json.load(open(os.path.join(d, f"{stem}-aligned.json"), encoding="utf-8"))
        segs = aligned["segments"]
        align_sec = aligned["metadata"]["asr_align"]["timing"]["asr_align_sec"]
        minutes = max(s["end"] for s in segs) / 60.0
        reader = qr._SpanReader(os.path.join(d, f"{stem}-vocal.ogg"))
        clips = [reader.read(s["start"] - qr.SEGMENT_PAD_SEC, s["end"] + qr.SEGMENT_PAD_SEC) for s in segs]
        clips = [c for c in clips if len(c) >= int(0.05 * qr.TARGET_SR)]
        speech = sum(len(c) for c in clips) / qr.TARGET_SR
        print(f"== {stem}: {len(clips)} segments, {speech:.0f} s of clip audio, {minutes:.1f} min, whisper align {align_sec:.1f} s")
        arms = [("eager B=8", "off", 8), ("eager B=16", "off", 16)]
        if not args.eager_only:
            arms.append(("compiled B=8", "on", 8))
        for label, accel, batch in arms:
            qr.BATCH_CLIPS = batch
            referee = qr.QwenReferee(device=args.device, accel=accel, vram_budget_gib=args.vram)
            referee.warm()
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            out = referee.transcribe_batch(clips)
            if accel == "on":  # second pass: without the compile
                began = time.perf_counter()
                out = referee.transcribe_batch(clips)
            elapsed = time.perf_counter() - began
            empty = sum(1 for text, _ in out if not text.strip())
            print(
                f"   {label:14} {elapsed:6.1f} s -> {elapsed / minutes:5.2f} s/min of file, "
                f"{elapsed / (speech / 60):5.2f} s/min of speech | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | empty {empty}",
                flush=True,
            )
            referee.close()
    return 0


def calls(args) -> int:
    """The compiled path over many consecutive generate calls: 12 different
    batches, then the same batch 6 times. Per-call time separates a periodic
    re-record from a per-length recompile (bench-baselines.md 21.6)."""

    import torch

    d = os.path.normpath(args.product)
    stem = os.path.basename(d)
    segs = json.load(open(os.path.join(d, f"{stem}-aligned.json"), encoding="utf-8"))["segments"]
    reader = qr._SpanReader(os.path.join(d, f"{stem}-vocal.ogg"))
    clips = sorted(
        (c for c in (reader.read(s["start"] - 0.1, s["end"] + 0.1) for s in segs) if len(c) >= 800),
        key=len,
    )
    size = args.batch
    batches = [clips[i : i + size] for i in range(0, 12 * size, size)]
    referee = qr.QwenReferee(device=args.device, accel="on", vram_budget_gib=args.vram)
    referee.warm()

    def one(label, batch):
        torch.cuda.synchronize()
        began = time.perf_counter()
        referee.transcribe_batch(batch)
        torch.cuda.synchronize()
        print(
            f"{label:26} {time.perf_counter() - began:6.2f}s  alloc {torch.cuda.memory_allocated() / 2**30:5.2f} GiB  "
            f"reserved {torch.cuda.memory_reserved() / 2**30:5.2f} GiB  longest {max(len(c) for c in batch) / qr.TARGET_SR:4.1f}s",
            flush=True,
        )

    for i, batch in enumerate(batches):
        one(f"different batch #{i}", batch)
    for i in range(6):
        one(f"same batch (#3) rep {i}", batches[3])
    referee.close()
    return 0


def batch_sizes(args) -> int:
    """Batch size vs throughput and peak VRAM, eager and compiled, on
    production-shaped short clips and on 15 s clips (bench-baselines.md 22.3)."""

    import gc

    import torch

    d = os.path.normpath(args.product)
    stem = os.path.basename(d)
    segs = json.load(open(os.path.join(d, f"{stem}-aligned.json"), encoding="utf-8"))["segments"]
    reader = qr._SpanReader(os.path.join(d, f"{stem}-vocal.ogg"))
    short = [c for c in (reader.read(s["start"] - 0.1, s["end"] + 0.1) for s in segs[:64]) if len(c) >= 800]
    mid = [reader.read(float(segs[i * 6]["start"]), float(segs[i * 6]["start"]) + 15.0) for i in range(32)]
    print(f"short: {len(short)} clips, {sum(len(c) for c in short) / qr.TARGET_SR:.0f} s; mid: {len(mid)} x 15 s")
    for accel in ("off", "on"):
        for size in (4, 8, 16, 32):
            qr.BATCH_CLIPS = size
            qr.COMPILED_BATCH_CLIPS = size
            referee = qr.QwenReferee(device=args.device, accel=accel, vram_budget_gib=args.vram)
            referee.warm()
            cells = []
            for label, clips in (("short", short), ("15s", mid)):
                referee.transcribe_batch(clips[:size])  # warm-up / compile this shape
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.perf_counter()
                referee.transcribe_batch(clips)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                audio = sum(len(c) for c in clips) / qr.TARGET_SR
                cells.append(
                    f"{label}: {elapsed:6.2f}s ({elapsed / (audio / 60):5.2f} s/min) "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                )
            print(f"accel={accel:3} B={size:2} | " + " | ".join(cells), flush=True)
            referee.close()
            gc.collect()
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vram", type=float, default=6.5, help="budget handed to the compiled referee")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("recheck")
    r.add_argument("products", nargs="+", help="out/<stem> directories with -aligned.json and -vocal.ogg")
    r.add_argument("--eager-only", action="store_true")
    r.set_defaults(run=recheck)
    c = sub.add_parser("calls")
    c.add_argument("product")
    c.add_argument("--batch", type=int, default=8)
    c.set_defaults(run=calls)
    b = sub.add_parser("bs")
    b.add_argument("product")
    b.set_defaults(run=batch_sizes)
    s = sub.add_parser("sweep")
    s.add_argument("audio", type=Path)
    s.add_argument("vad", type=Path)
    s.set_defaults(run=sweep)
    a = sub.add_parser("agreement")
    a.add_argument("roots", nargs="+")
    a.add_argument("--eager-only", action="store_true")
    a.set_defaults(run=agreement)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
