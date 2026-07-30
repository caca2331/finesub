"""Isolate *scope* — per-segment vs global DP — with everything else held fixed.

**SUPERSEDED (2026-07-29): the conclusion shipped.** Production `split_segments` is now
globally scoped with an ASR-seam bonus and the word-pause term (docs/segment_split.md).
Everything below describes the *pre-migration* production this driver was measuring against;
its `_global` arm is essentially what production does today.

Goal 2.2 step 2 asks whether running one DP over the whole word stream beats production's
per-segment splitting. Every earlier attempt answered a different question: the `qwen_split` arm
changes the scope *and* the per-cut cost (`base` 1.0 -> 0.05), the gap discount shape, six extra
boundary terms and the post-processing — and those constants were selected by `bench.py --cv` on
the *mechanical* metric, which FINDINGS §0 records as disagreeing in sign with human judgement.

So this driver does not reimplement anything. It calls production's own `split_segments` twice on
the same words with the same `DEFAULT_SPLIT_PARAMS`, and changes exactly one thing:

    per-segment  the ASR's own segments, i.e. production's actual output (control)
    global       one segment holding every word in the range, so the DP may cut anywhere

`--bonus` adds the one term concatenation throws away — "the ASR started a segment here" — still
inside production's own cost model. That makes it the only arm where a `bonus` reading is about
*scope*, and not about `qwen_split`'s separately-tuned constants.

**Plain `--bonus` never wins** (14 windows / 224 `must`): per-segment 95.1% / 31.6 violations per
100, global saturates at bonus>=8 on 94.2% / 32.8 — worse on both axes — and its merges keep
destroying `must` positions (68 at bonus=0, still 2 at saturation). `--g-floor` (clipping
production's unclipped discount, which returns -18 at a 3 s gap and so makes any real pause an
unconditional cut) changes almost nothing.

`--fix` adds the two defects that gap explains, both found by reasoning about *what orders the
boundaries* rather than by sweeping. Bonus is subtracted uniformly, so it cannot reorder anything;
which ASR boundary gets merged is decided entirely by `t + g_score(g) + no_gap_penalty*[g==0]`.

    (A) unban   `adjust_words` bans a cut between a gap word and the side it is anchored to — a
                rule written for boundaries *interior* to a segment. When the boundary IS the ASR's
                segment start, per-segment scope cut there structurally and the ban never applied;
                global scope turns it into `b = inf`, uncuttable at any bonus. 6 of the 8 merges
                surviving bonus=30 are banned boundaries, and both `must` destructions among them.
    (B) 词停     `interval_gap_between` returns 0 for every boundary inside one VAD interval,
                collapsing 183 of the gold windows' ASR boundaries (42 of them `must`) into a
                single tie broken only by `t_score`'s six discrete values. Word-level pause does
                discriminate there — `must` median +0.195 s vs presumed-never +0.000 s, only 9% of
                which exceed 0.05 s — and production discards it. (`qwen_split.word_pause_trust`.)

Each halves the damage alone (3 -> 1 and 3 -> 2 `must` destroyed); together 3 -> 0:

    逐段（生产）         95.1% (213/224)  546 刀  31.6/百
    全局 b=5             93.8% (210)      535     31.9   并 23 (毁 3)
    全局 b=5 +A+B        95.5% (214)      543     31.2   并 13 (毁 0)
    全局 b=4 +A+B        95.1% (213)      527     29.6   并 27 (毁 1, 消 20)

So the merge capability production lacks (`premerge` merges 1 segment across the 11-clip bed) is
harvestable inside production's own cost model — but the margins are ~1 `must` on 224 and the
pooled set includes the batch-1 windows some constants were fitted on. Directional, not an
acceptance test. (Batch 3's `ok` collapse was repaired and the numbers rerun: every arm's
violation count fell by the same amount and no arm-to-arm delta moved, so that class of labelling
error is neutral to comparisons.)

`dp_split` is O(n^2), so each gold window is run over a padded time range rather than the whole
clip; `PAD_SEC` of context on each side keeps edge effects out of the scored region.

    python -m tools.qwen3_explore.scope_control
    python -m tools.qwen3_explore.scope_control --bonus 4,5 --fix
    python -m tools.qwen3_explore.scope_control --bonus 3,5,8 --g-floor=-1.0
    python -m tools.qwen3_explore.scope_control --pad 40 --per-window
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from segment_split import (  # noqa: E402
    DEFAULT_SPLIT_PARAMS, Boundary, _build_piece_segment, adjust_words, build_zones, dp_split,
    interval_gap_between, score_boundaries, split_segments, t_score,
)
from segment_split import g_score as prod_g_score  # noqa: E402

from tools.segmentation_gold.gold import (  # noqa: E402
    cuts_to_indices, evaluate_by_index, load_golds, load_words,
)

from .common import baseline_aligned, vad_json  # noqa: E402
from .gold_sweep import GOLD_DIR, _cues  # noqa: E402

PAD_SEC = 30.0


def words_in_range(raw: dict, lo: float, hi: float) -> list[dict]:
    out = []
    for seg in raw["segments"]:
        for w in seg.get("words") or []:
            if lo <= float(w["start"]) and float(w["end"]) <= hi:
                out.append(w)
    out.sort(key=lambda w: (float(w["start"]), float(w["end"])))
    return out


def segments_in_range(raw: dict, lo: float, hi: float) -> list[dict]:
    """Production's own segments, keeping only those fully inside the padded range.

    Partially overlapping segments are dropped rather than truncated: truncating would invent a
    cue boundary that production never produced, which is the very thing being measured.
    """
    out = []
    for seg in raw["segments"]:
        ws = seg.get("words") or []
        if not ws:
            continue
        if lo <= float(ws[0]["start"]) and float(ws[-1]["end"]) <= hi:
            out.append(seg)
    return sorted(out, key=lambda s: float(s["start"]))


def as_one_segment(words: list[dict]) -> dict:
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": "".join(str(w.get("word", "")) for w in words),
        "words": words,
    }


_BOUNDARY_MARK = "__asr_boundary__"


def global_with_bonus(segs: list[dict], speech: list[dict], bonus: float,
                      params=DEFAULT_SPLIT_PARAMS, word_pause_trust: float = 0.0) -> list[dict]:
    """Production's cost model, per segment; only the **DP** is global, minus `bonus` at junctions.

    The earlier version of this function ran `adjust_words` + `score_boundaries` over the
    *concatenated* stream, and that was wrong. Two things change when those passes see one long
    list instead of one segment at a time:

    - `adjust_words` pass 2 treats the ends of the word list as separators when deciding text-glue
      runs, so a run touching a segment edge gets different anchors globally than per segment;
    - a global sort by (start, end) reorders words wherever segments overlap in time (49 places in
      the 11-clip bed, see docs/vad-asr.md).

    Measured fallout: 17 of 25 297 intra-segment boundaries got a different cost, several flipping
    between `inf` (banned) and a large negative, which is where the "banned blocks 8 production
    cuts" and "the shape term adds 11 cuts" readings came from. Both were artefacts.

    Computing costs per segment the way production does, and letting only `dp_split` span the whole
    window, gives **exact equivalence**: at `bonus=1e6` this reproduces `split_segments(segs)` cut
    for cut (0 missing, 0 extra). It still differs from the stored artefact by 7 cuts over the 14
    gold windows, but that is the **test bed**, not this function: the 8 `BV*` clips carry zero
    `splitted_before` tags and only `asr_align` metadata, i.e. `split_segments` never ran on them
    (`split-explorer-8bv-20260718/` is that experiment's intermediate output, not a pipeline
    artefact; only `yingtao`/`yui`/`kaguya60` are full-pipeline). All 7 land on `BV*` clips and are
    the *first* application. A third pass adds 0 — the function is idempotent. **Baseline = the
    re-run**, never the stored artefact.

    `word_pause_trust` is the one addition that survives, and its name is wrong: values from 0.1 to
    1.0 give bit-identical results, and dropping the discount entirely changes nothing. The whole
    effect is **waiving `no_gap_penalty`**. `interval_gap_between` reports 0 for every junction
    inside a single VAD interval, so production charges the no-gap penalty there — but the
    aligner's *word-level* timestamps do report a pause at many of them (82 of the 183 such ASR
    boundaries in the gold windows, 24 of them `must`). The penalty's condition, not its
    magnitude, is what is wrong. Applied to junction costs here; applying it to intra-segment
    boundaries improves the plain per-segment path too (FINDINGS §3.1).
    """
    spans = sorted((float(i["start"]), float(i["end"])) for i in speech
                   if float(i["end"]) > float(i["start"]))
    zones = build_zones(spans)
    adj: list = []
    bounds: list = []
    for seg in segs:
        part = adjust_words(seg.get("words") or [], spans, zones)
        if not part:
            continue
        if adj:  # junction between two ASR segments: scored fresh, never banned
            left, right = adj[-1], part[0]
            gap = interval_gap_between(spans, left.anchor, right.anchor)
            t = t_score(left.text, right.text, right.space_before)
            b = params.a * (t + prod_g_score(gap, params)) + params.base
            if gap <= 0:
                pause = max(0.0, float(right.source["start"]) - float(left.source["end"]))
                if word_pause_trust and pause > 0:
                    b = params.a * (t + prod_g_score(pause * word_pause_trust, params)) + params.base
                else:
                    b += params.non_vad_gap_penalty
            bounds.append(Boundary(False, gap, t, b - bonus, gap <= 0))
        if len(part) >= 2:
            bounds.extend(score_boundaries(part, spans, params))
        adj.extend(part)
    if len(adj) < 2:
        return list(segs)
    return [{"start": adj[a].start, "end": adj[b - 1].end,
             "text": "".join(adj[k].text for k in range(a, b))}
            for a, b in dp_split(adj, bounds, params).pieces]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=PAD_SEC)
    ap.add_argument("--per-window", action="store_true", help="逐窗口打印，不只汇总")
    ap.add_argument("--fix", action="store_true", help="再加一组带「词级停顿」的臂")
    ap.add_argument("--trust", type=float, default=0.25, help="B 的词级停顿折算系数")
    ap.add_argument("--bonus", default="",
                    help="逗号分隔：再加若干「全局 + ASR 边界 bonus」臂（生产参数不变）")
    args = ap.parse_args()
    bonuses = [float(x) for x in args.bonus.split(",") if x.strip()]

    golds = load_golds(GOLD_DIR)
    total: dict[str, dict] = {}
    rows = []
    for g in sorted(golds, key=lambda g: (g["clip"], g["window"][0])):
        clip = g["clip"]
        name = f"{clip}-{int(g['window'][0])}-{int(g['window'][1])}"
        raw = json.loads(baseline_aligned(clip).read_text(encoding="utf-8"))
        vad = json.loads(vad_json(clip).read_text(encoding="utf-8"))
        speech = [{"start": float(a), "end": float(b)} for a, b in vad["speech"]]
        words = load_words(baseline_aligned(clip))
        lo, hi = float(g["window"][0]) - args.pad, float(g["window"][1]) + args.pad

        ws = words_in_range(raw, lo, hi)
        if len(ws) < 2:
            continue
        segs = segments_in_range(raw, lo, hi)
        arms = {
            "逐段（生产原样）": segs,
            # Self-check on the harness: production's aligned output already went through
            # `split_segments`, so re-running it per segment must reproduce that output. If this
            # arm drifts from the first one, the difference below is not attributable to scope.
            # `split_segments` is not idempotent (+7 cuts over the 14 windows), and the global arm
            # inherits that, so this — not the stored artefact — is the baseline to compare against.
            "逐段（重跑，基线）": split_segments(segs, speech, params=DEFAULT_SPLIT_PARAMS),
        }
        for bo in bonuses:
            arms[f"全局 + b={bo:g}"] = global_with_bonus(segs, speech, bo)
            if args.fix:
                arms[f"全局 + b={bo:g} +词停"] = global_with_bonus(
                    segs, speech, bo, word_pause_trust=args.trust)

        prod_cuts = set(cuts_to_indices(_cues({"segments": segs}), words))
        lab = {it["k"]: it["label"] for it in g["items"] if isinstance(it.get("k"), int)}
        klo, khi = min(lab), max(lab)
        line = [name, len(ws)]
        for tag, arm in arms.items():
            cuts = cuts_to_indices(_cues({"segments": arm}), words)
            e = evaluate_by_index([g], cuts)
            t = total.setdefault(tag, {})
            for k in ("must", "must_hit", "declared", "presumed", "in_window", "skipped"):
                t[k] = t.get(k, 0) + e[k]
            inw = {k for k in cuts if klo <= k <= khi}
            pin = {k for k in prod_cuts if klo <= k <= khi}
            for k in pin - inw:
                t["并"] = t.get("并", 0) + 1
                t["并_" + lab.get(k, "表外")] = t.get("并_" + lab.get(k, "表外"), 0) + 1
            for k in inw - pin:
                t["切"] = t.get("切", 0) + 1
                t["切_" + lab.get(k, "表外")] = t.get("切_" + lab.get(k, "表外"), 0) + 1
            line += [e["must_hit"], e["must"], e["in_window"]]
        rows.append(line)

    if args.per_window:
        print(f"{'窗口':26} {'词数':>5} {'逐段 命中/must':>13} {'刀':>4} | "
              f"{'自检':>6} {'刀':>4} | {'全局 命中/must':>13} {'刀':>4}")
        for r in rows:
            print(f"{r[0]:26} {r[1]:>5} {r[2]:>8}/{r[3]:<4} {r[4]:>4} | "
                  f"{r[5]:>6} {r[7]:>4} | {r[8]:>8}/{r[9]:<4} {r[10]:>4}")
        print()

    print(f"{'臂':<22} {'必切召回':>16} {'刀数':>6} {'违反':>6} {'/百条':>7}"
          f"   {'并掉(毁must/消违反)':>18} {'多切(得must/增违反)':>18}")
    for tag, t in total.items():
        judged = t["in_window"] - t["skipped"]
        viol = t["declared"] + t["presumed"]
        d = ""
        if t.get("并") or t.get("切"):
            d = (f"   {t.get('并', 0):>4} ({t.get('并_must', 0)} / "
                 f"{t.get('并_never', 0) + t.get('并_表外', 0)})"
                 f"        {t.get('切', 0):>4} ({t.get('切_must', 0)} / "
                 f"{t.get('切_never', 0) + t.get('切_表外', 0)})")
        print(f"{tag:<22} {100*t['must_hit']/t['must']:6.1f}% ({t['must_hit']:3d}/{t['must']:3d}) "
              f"{t['in_window']:>6} {viol:>6} {100*viol/judged if judged else 0:>7.1f}{d}")


if __name__ == "__main__":
    main()
