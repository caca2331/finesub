"""Decompose an arm's difference from production into **merges** and **extra cuts**, by gold label.

**Note (2026-07-29)**: the merge column's question is settled — production's DP is now global and
merges ASR seams itself, and the separate `premerge` pass was deleted (0 merges on the bed). The
`生产+premerge` arm is gone with it; "生产" here is whatever `segment_split` currently does.

`gold_sweep` reports recall and violations; those two numbers hide the fact that a global DP does
two independent things to production's cue set, with opposite signs:

    并掉 (merge)   production cut here, the arm does not  — the only thing production cannot do at
                   all -- when this was written `split_segments` only split, never joined, and
                   the `premerge` pass merged 1 segment across the whole 11-clip bed
    多切 (extra)   the arm cuts here, production does not — mostly *not* a merge side effect

Reporting them pooled is how "全局 DP 不值得做" got asserted from the cutting axis alone. Keep them
separate: the merge column is what decides whether a merge pass is worth building, and it is only
credible while its `毁 must` count is 0.

`--why` additionally decomposes each extra cut's boundary cost. That is how the dominant cause was
found: at a 0.55-0.66 s VAD gap `qwen_split`'s clipped cubic gap discount returns -2.74..-3.00
where production's square returns -0.62..-0.86, and `a*(t+g)+base` goes **negative** — the DP is
paid to cut and the piece score (worst case ~2.87) cannot veto it. A cut cost should never be
negative; `g_floor=-1.0` is the smallest change that restores it.

    python -m tools.qwen3_explore.merge_audit
    python -m tools.qwen3_explore.merge_audit --why
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import segment_split as prod_split  # noqa: E402

from tools.segmentation_gold.gold import (  # noqa: E402
    cuts_to_indices, evaluate_by_index, load_golds, load_words,
)

from .common import baseline_aligned, vad_json  # noqa: E402
from .gold_sweep import GOLD_DIR, _cues  # noqa: E402
from .lexicon import CANNOT_START_WHISPER  # noqa: E402
from .qwen_split import CLAUSE, PUNCT_STRIP, SENTENCE, Params, g_score, segment, vad_gap_at  # noqa: E402

# Batch 1 is the only set any parameter was ever fitted on; everything else is held out.
FIT = {"BV1nxje63ERi-115-235", "BV1UBjq6fEgb-46-166", "yui-660-780"}

ARMS: dict[str, dict] = {
    "b4/base1.0": dict(asr_boundary_bonus=4.0, base=1.0),
    "b4/base1.0/floor-1": dict(asr_boundary_bonus=4.0, base=1.0, g_floor=-1.0),
    "现行 Params()": {},
}


def boundary_cost(words: list[dict], k: int, non_speech, p: Params) -> dict:
    """Reproduce `split_block`'s per-boundary cost for one junction, term by term."""
    left, right = words[k], words[k + 1]
    pause = max(0.0, right["start"] - left["end"])
    vad = vad_gap_at(non_speech, (left["end"] + right["start"]) / 2)
    if vad > 0 and pause <= 1e-6:
        vad *= p.vad_gap_trust_no_pause
    g_eff = vad if vad > 0 else pause * p.word_pause_trust
    tail = left.get("trailing_punct", "") or left["word"][-1:]
    if any(c in SENTENCE for c in tail):
        t = p.punct_sentence
    elif any(c in CLAUSE for c in tail):
        t = p.punct_clause
    else:
        t = 1.0
    if right["word"].strip(PUNCT_STRIP) in CANNOT_START_WHISPER:
        t += p.fragment_penalty
    cost = p.a * (t + g_score(g_eff, p)) + p.base + (p.no_gap_penalty if g_eff <= 1e-6 else 0.0)
    return {"pause": pause, "vad": vad, "g_eff": g_eff, "g": g_score(g_eff, p),
            "g_prod": prod_split.g_score(g_eff, prod_split.DEFAULT_SPLIT_PARAMS),
            "t": t, "cost": cost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--why", action="store_true", help="逐条打印「多切」的代价构成")
    ap.add_argument("--include-fit", action="store_true", help="连拟合集（批 1）一起算")
    args = ap.parse_args()

    golds = [g for g in load_golds(GOLD_DIR)
             if args.include_fit
             or f"{g['clip']}-{int(g['window'][0])}-{int(g['window'][1])}" not in FIT]
    tot: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    why: list[tuple] = []

    for g in golds:
        clip = g["clip"]
        vad = json.loads(vad_json(clip).read_text(encoding="utf-8"))
        raw = json.loads(baseline_aligned(clip).read_text(encoding="utf-8"))
        words = load_words(baseline_aligned(clip))
        non_speech = [(float(a), float(b)) for a, b in vad["non_speech"]]
        lab = {it["k"]: it["label"] for it in g["items"] if isinstance(it.get("k"), int)}
        lo, hi = min(lab), max(lab)

        def win(cues) -> set[int]:
            return {k for k in cuts_to_indices(cues, words) if lo <= k <= hi}

        base_cues = _cues(raw)
        prod = win(base_cues)
        arms = {"生产": prod}
        for name, kw in ARMS.items():
            p = replace(Params(), **kw)
            out = segment(json.loads(json.dumps(raw)), non_speech, float(vad["duration"]), p,
                          words_carry_punct=True)
            arms[name] = win(_cues(out))
            if args.why and name.endswith("base1.0"):
                for k in sorted(arms[name] - prod):
                    why.append((clip, lab.get(k, "表外"), boundary_cost(words, k, non_speech, p)))

        for name, cuts in arms.items():
            e = evaluate_by_index([g], sorted(cuts))
            for key in ("must", "must_hit", "declared", "presumed", "in_window", "skipped"):
                tot[name][key] += e[key]
            if name == "生产":
                continue
            for k in prod - cuts:
                tot[name]["并"] += 1
                tot[name]["并_" + lab.get(k, "表外")] += 1
            for k in cuts - prod:
                tot[name]["切"] += 1
                tot[name]["切_" + lab.get(k, "表外")] += 1

    scope = "全部窗口" if args.include_fit else "留出（批 2+3）"
    print(f"gold {len(golds)} 窗 / {scope} / 词序号锚定\n")
    print(f"{'臂':22} {'必切召回':>16} {'刀':>5} {'违反':>5} {'/百条':>7}"
          f"   {'并掉(毁must/消违反)':>20} {'多切(得must/增违反)':>20}")
    for name, t in tot.items():
        judged = t["in_window"] - t["skipped"]
        viol = t["declared"] + t["presumed"]
        d = ""
        if "并" in t or "切" in t:
            d = (f"   {t['并']:>4} ({t['并_must']} / {t['并_never'] + t['并_表外']})"
                 f"          {t['切']:>4} ({t['切_must']} / {t['切_never'] + t['切_表外']})")
        print(f"{name:22} {100 * t['must_hit'] / t['must']:5.1f}% ({t['must_hit']:3d}/{t['must']}) "
              f"{t['in_window']:>5} {viol:>5} {100 * viol / judged if judged else 0:>7.1f}{d}")

    if why:
        print(f"\n「多切」的代价构成（b4/base1.0）——代价为负 = DP 因下刀而获利，形状项否决不了：")
        print(f"{'clip':14} {'停顿':>6} {'VAD':>5} {'qwen g':>7} {'生产 g':>7} {'t':>5} {'刀代价':>7} {'金标签':>6}")
        for clip, l, c in sorted(why, key=lambda r: r[2]["cost"]):
            print(f"{clip:14} {c['pause']:>6.2f} {c['vad']:>5.2f} {c['g']:>7.2f} {c['g_prod']:>7.2f} "
                  f"{c['t']:>5.1f} {c['cost']:>7.2f} {l:>6}")
        neg = sum(1 for _, _, c in why if c["cost"] < 0)
        print(f"\n共 {len(why)} 处，其中代价为负 {neg} 处")


if __name__ == "__main__":
    main()
