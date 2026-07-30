"""Stratified sample of boundaries for human semantic adjudication.

The acceptability metrics are purely mechanical — a buried silence, a run of tight cues, a cue
whose first token is in `lexicon.CANNOT_START`. They cannot see the one thing
docs/merge-calibration.md actually cares about ("前句已完整再起信息，即便 gap≈0 也分开"). Every
number in FINDINGS is a count of mechanical verdicts, so **the verdicts themselves have to
be checked against semantics** — otherwise the whole tuning story is optimising a proxy that may
be blind to an entire error class, or flagging things that are fine.

Four strata, one per verdict the objective actually prices:

    欠切 / 不可救过切 / 词中切   the three error classes counted in `bench.objective`
    自由刀                       the cuts the metric *certifies* as free — the stratum that would
                                 expose a blind spot rather than a false positive

Segmentation is recomputed in-process from `Params()`, exactly as `bench.py` does, so the sample
always matches the parameters currently in the tree. (It used to read `*-QS.json` off disk, which
silently sampled whatever the last CLI run happened to leave there.)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from subtitle_metrics import weighted_char_count  # noqa: E402

from .acceptability import (  # noqa: E402
    GAP_MERGE_CONSIDER,
    GAP_MERGE_REFUSE,
    MERGE_CHAR_ABS,
    MERGE_SPAN_ABS,
    SENTENCE_MARKS,
    buried_silences,
    first_token,
    vad_gaps_within,
)
from .bench import ARMS  # noqa: E402
from .common import ALL_CLIPS, baseline_aligned, qwen_raw, vad_json  # noqa: E402
from .lexicon import lexicon_for  # noqa: E402
from .qwen_split import Params, segment  # noqa: E402


def collect(clip: str, arm: str, p: Params):
    vad = json.loads(vad_json(clip).read_text(encoding="utf-8"))
    ns = [(float(a), float(b)) for a, b in vad["non_speech"]]
    whisper = arm.startswith("whisper")
    src = baseline_aligned(clip) if whisper else qwen_raw(clip)
    data = json.loads(src.read_text(encoding="utf-8"))
    if arm != "whisper-prod":
        data = segment(data, ns, float(vad["duration"]), p, words_carry_punct=whisper)
    segs = [s for s in data["segments"] if str(s.get("text") or "").strip()]
    cannot_start = lexicon_for(whisper)

    pools = {"欠切": [], "不可救过切": [], "词中切": [], "自由刀": []}
    for i, s in enumerate(segs):
        if first_token(s) in cannot_start:
            pools["词中切"].append({"clip": clip, "kind": "词中切", "i": i})
            continue  # mirrors analyse(): a fragment is not also scored as an under-split
        for g in buried_silences(ns, s["start"], s["end"]):
            if g >= GAP_MERGE_REFUSE:
                pools["欠切"].append({"clip": clip, "kind": "欠切", "silence": round(g, 2), "i": i})

    def close(r: list[int]) -> None:
        if len(r) >= 3:
            pools["不可救过切"].append({"clip": clip, "kind": "不可救过切", "i": r[0], "n": len(r)})

    run = [0]
    for i, (a, b) in enumerate(zip(segs, segs[1:]), start=1):
        gap = max(0.0, b["start"] - a["end"])
        gap = max(gap, max(vad_gaps_within(ns, a["end"] - 0.05, b["start"] + 0.05) or [0.0]))
        if gap >= GAP_MERGE_CONSIDER:
            pools["自由刀"].append({"clip": clip, "kind": "自由刀", "gap": round(gap, 2), "i": i})
            close(run)
            run = [i]
        else:
            run.append(i)
    close(run)

    # analyse() only counts a run as unrecoverable when merge could actually have joined it.
    pools["不可救过切"] = [r for r in pools["不可救过切"] if _joinable(segs[r["i"] : r["i"] + r["n"]])]
    return segs, pools


def _joinable(group: list[dict]) -> bool:
    span = group[-1]["end"] - group[0]["start"]
    chars = sum(weighted_char_count(s["text"]) for s in group)
    complete = all(str(s["text"]).strip()[-1:] in SENTENCE_MARKS for s in group[:-1])
    return span <= MERGE_SPAN_ABS and chars <= MERGE_CHAR_ABS and not complete


def render(segs, item, n_item: int) -> str:
    i = item["i"]
    ctx = lambda k: segs[k]["text"] if 0 <= k < len(segs) else "—"  # noqa: E731
    head = f"### {n_item}. [{item['clip']}] {item['kind']}"
    if item["kind"] == "欠切":
        s = segs[i]
        return (
            f"{head} — 条内埋着 {item['silence']}s 静音\n"
            f"- 前: {ctx(i - 1)}\n- **本条**: {s['text']}\n- 后: {ctx(i + 1)}\n"
            f"- 判断：这条内部该不该切开？"
        )
    if item["kind"] == "不可救过切":
        grp = segs[i : i + item["n"]]
        lines = "\n".join(f"  {k + 1}) {g['text']}" for k, g in enumerate(grp))
        return (
            f"{head} — {item['n']} 条紧邻\n- 前: {ctx(i - 1)}\n{lines}\n- 后: {ctx(i + item['n'])}\n"
            f"- 判断：这 {item['n']} 条本该是几条？"
        )
    if item["kind"] == "词中切":
        return (
            f"{head} — 首词「{first_token(segs[i])}」\n"
            f"- 左: {ctx(i - 1)}\n- 右: {ctx(i)}\n"
            f"- 判断：这刀是否切断了一个词/紧密短语？"
        )
    return (
        f"{head} — 刀口 gap {item['gap']}s\n- 左: {ctx(i - 1)}\n- 右: {ctx(i)}\n"
        f"- 判断：在这里切开合适吗？"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="qwen", choices=ARMS)
    ap.add_argument("--clips", default=",".join(ALL_CLIPS))
    ap.add_argument("--per-stratum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pools: dict[str, list] = {"欠切": [], "不可救过切": [], "词中切": [], "自由刀": []}
    store = {}
    for clip in args.clips.split(","):
        segs, got = collect(clip, args.arm, Params())
        store[clip] = segs
        for k, v in got.items():
            pools[k] += v

    lines = [f"# 切点语义裁决抽样 — arm={args.arm} seed={args.seed}\n",
             "机械判定按分层抽样，逐条给语义裁决。判断的是语义，不是机械规则是否被正确执行。\n"]
    n = 0
    for kind, pool in pools.items():
        rng.shuffle(pool)
        picked = pool[: args.per_stratum]
        lines.append(f"\n## {kind}（总体 {len(pool)}，抽 {len(picked)}）\n")
        for item in picked:
            n += 1
            lines.append(render(store[item["clip"]], item, n))
            lines.append("")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"{args.out} [{args.arm}]: " + ", ".join(f"{k}池={len(v)}" for k, v in pools.items()))


if __name__ == "__main__":
    main()
