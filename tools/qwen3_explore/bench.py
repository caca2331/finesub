"""Run both arms over all 11 clips and report the §4.6 table; also the leave-one-clip-out CV.

Everything runs in-process from cached artifacts (`*-Q-rescued-raw.json` + the Whisper baseline
`*-aligned.json` + the shared VAD), so a full sweep is pure CPU — no ASR, no GPU. This replaces
the ad-hoc loops the earlier numbers were produced by, which is why those runs were not
reproducible from the repo.

    python -m tools.qwen3_explore.bench                  # the comparison table
    python -m tools.qwen3_explore.bench --per-clip       # + one row per clip
    python -m tools.qwen3_explore.bench --cv fragment_penalty=0,0.5,1.0,1.5,2.0,3.0
    python -m tools.qwen3_explore.bench --cv 'word_pause_trust=0.25,0.5;no_gap_penalty=0.2,0.5'
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .acceptability import analyse
from .common import ALL_CLIPS, NEW_CLIPS, TUNING_CLIPS, baseline_aligned, qwen_raw, vad_json
from .qwen_split import Params, segment

# Three arms, because "whisper" is ambiguous and the ambiguity mattered: the earlier §4.6 table
# compared Qwen-with-a-purpose-built-splitter against Whisper's *production* segmentation, which
# no one had optimised. `whisper-split` is the fair arm — same words, same splitter, same VAD.
ARMS = ("whisper-prod", "whisper-split", "qwen")


def _load(clip: str) -> tuple[dict, list[tuple[float, float]], float]:
    vad = json.loads(vad_json(clip).read_text(encoding="utf-8"))
    return vad, [(float(a), float(b)) for a, b in vad["non_speech"]], float(vad["duration"])


def score(clip: str, arm: str, p: Params, snap: bool = True) -> dict:
    _, non_speech, duration = _load(clip)
    src = baseline_aligned(clip) if arm.startswith("whisper") else qwen_raw(clip)
    raw = json.loads(src.read_text(encoding="utf-8"))
    whisper = arm.startswith("whisper")
    if arm != "whisper-prod":
        # `asr_boundary_bonus` is meaningless on the Qwen arm — its segments are decode windows,
        # not sentence hypotheses — so it never reaches it, whatever the CLI says.
        raw = segment(raw, non_speech, duration, p if whisper else replace(p, asr_boundary_bonus=0.0),
                      words_carry_punct=whisper, snap=snap)
    return analyse(raw, non_speech, f"{clip}/{arm}", whisper_words=whisper)


def aggregate(rows: list[dict]) -> dict:
    """Per-100-cue rates over a group of clips, pooled (not averaged over clips)."""
    n = sum(r["cues"] for r in rows)
    out = {"cues": n}
    # All rates per 100 cues — arms differ in cue count by up to 30%, so raw counts are not
    # comparable across them (`turn_missed` in particular rises simply because cues are longer).
    for k in ("under_hard", "over_unrecoverable", "midword", "turn_missed"):
        out[k] = 100 * sum(r[k] for r in rows) / n
    out["shape_mean"] = sum(r["shape_mean"] * r["cues"] for r in rows) / n
    out["tier_bad"] = sum(r["tier_bad"] * r["cues"] for r in rows) / n
    return out


HEAD = f"{'组/臂':24s} {'cues':>5s} | {'欠切':>6s} {'不可救过切':>10s} {'词中切':>7s} {'转折':>6s} {'形态分':>7s} {'不行%':>6s}"


def _row(label: str, a: dict) -> str:
    return (
        f"{label:24s} {a['cues']:5d} | {a['under_hard']:6.2f} {a['over_unrecoverable']:10.2f} "
        f"{a['midword']:7.2f} {a['turn_missed']:6.2f} {a['shape_mean']:7.3f} {a['tier_bad']:6.1f}"
    )


def run_table(p: Params, per_clip: bool, snap: bool = True) -> None:
    results = {(c, a): score(c, a, p, snap) for c in ALL_CLIPS for a in ARMS}
    if per_clip:
        print(HEAD)
        for c in ALL_CLIPS:
            for a in ARMS:
                print(_row(f"{c}/{a}", aggregate([results[(c, a)]])))
        print()
    print(HEAD)
    for name, clips in (("调参集", TUNING_CLIPS), ("新来源", NEW_CLIPS)):
        for a in ARMS:
            print(_row(f"{name} {a}", aggregate([results[(c, a)] for c in clips])))


def objective(rows: list[dict], w_mid: float) -> float:
    """Unrecoverable errors at weight 1, mid-word cuts at `w_mid`, shape as a tiny tie-break.

    The weight is a *stated judgement*, not a measurement: an under-split or a 3-piece over-split
    survives to the final subtitle, while a mid-word cut is what `premerge` profile 3 exists to
    rejoin. It has to be explicit, because a purely lexicographic objective (unrecoverable first,
    mid-word only as tie-break) trivially selects `fragment_penalty=0` — mid-word cuts then have
    no price at all. `--weight-scan` reports how sensitive the winner is to this number.

    Deliberately excludes `turn_missed`: it is built from the same word list a turn bonus would
    have used, and putting it in the objective is how that term once passed CV by circular
    reasoning.
    """
    a = aggregate(rows)
    return a["under_hard"] + a["over_unrecoverable"] + w_mid * a["midword"] + 0.01 * a["shape_mean"]


def run_cv(spec: str, w_mid: float) -> None:
    """`spec` is `FIELD=v1,v2[;FIELD=...]` — one field, or a joint grid over several."""
    grid = [(f, [float(v) for v in vs.split(",")]) for f, vs in
            (part.split("=", 1) for part in spec.split(";"))]
    configs: list[tuple[float, ...]] = [()]
    for _, vals in grid:
        configs = [c + (v,) for c in configs for v in vals]
    fields = [f for f, _ in grid]

    def params(cfg: tuple[float, ...]) -> Params:
        return replace(Params(), **dict(zip(fields, cfg)))

    def show(cfg: tuple[float, ...]) -> str:
        return "/".join(f"{v:g}" for v in cfg)

    cache = {(c, cfg): score(c, "qwen", params(cfg)) for c in ALL_CLIPS for cfg in configs}
    print(f"== 留一交叉验证：{'/'.join(fields)}，{len(configs)} 组（8 折，qwen 臂，词中切权重 {w_mid}）\n")
    picks: Counter = Counter()
    for holdout in TUNING_CLIPS:
        train = [c for c in TUNING_CLIPS if c != holdout]
        best = min(configs, key=lambda cfg: objective([cache[(c, cfg)] for c in train], w_mid))
        held = aggregate([cache[(holdout, best)]])
        picks[show(best)] += 1
        print(
            f"  holdout={holdout:14s} best={show(best):<14s} -> 欠切{held['under_hard']:5.2f} "
            f"不可救{held['over_unrecoverable']:5.2f} 词中切{held['midword']:5.2f} "
            f"形态{held['shape_mean']:.3f}"
        )
    print(f"\n  各折选中频次: {dict(picks)}")

    # How much of the answer is the weight? If the winner is flat over a wide band, the weight is
    # not doing the work; if it swings, the "CV selected it" claim is really "I picked a weight".
    print(f"\n  权重敏感性（调参集全 8 clip 选出 / 新来源自选，后者仅作参考——它是留出集）:")
    for w in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        t = min(configs, key=lambda cfg: objective([cache[(c, cfg)] for c in TUNING_CLIPS], w))
        n = min(configs, key=lambda cfg: objective([cache[(c, cfg)] for c in NEW_CLIPS], w))
        print(f"    w={w:<5g} 调参集 {show(t):<14s} 新来源 {show(n)}")

    print(f"\n== 全集扫描（按调参集目标排序；新来源为留出集）")
    print(f"  {'/'.join(fields):>22s} | {'欠切':>6s} {'不可救':>7s} {'词中切':>7s} {'形态分':>7s} || 新来源同列")
    for cfg in sorted(configs, key=lambda cfg: objective([cache[(c, cfg)] for c in TUNING_CLIPS], w_mid)):
        t = aggregate([cache[(c, cfg)] for c in TUNING_CLIPS])
        n = aggregate([cache[(c, cfg)] for c in NEW_CLIPS])
        print(
            f"  {show(cfg):>22s} | {t['under_hard']:6.2f} {t['over_unrecoverable']:7.2f} "
            f"{t['midword']:7.2f} {t['shape_mean']:7.3f} || {n['under_hard']:6.2f} "
            f"{n['over_unrecoverable']:7.2f} {n['midword']:7.2f} {n['shape_mean']:7.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", action="store_true")
    ap.add_argument("--cv", help="FIELD=v1,v2[;FIELD=...] leave-one-clip-out over a Params grid")
    ap.add_argument("--weight-mid", type=float, default=0.2, help="objective weight on 词中切")
    ap.add_argument("--asr-boundary-bonus", type=float, default=Params.asr_boundary_bonus,
                    help="whisper 臂：把源 ASR 分段当软先验（qwen 臂应保持 0，见 qwen_split.Params）")
    ap.add_argument("--no-snap", action="store_true",
                    help="关掉 snap_cues_to_speech；它会把刀口从静音中点挪到边缘（qwen_split.segment 文档）")
    args = ap.parse_args()

    p = replace(Params(), asr_boundary_bonus=args.asr_boundary_bonus)
    if args.cv:
        run_cv(args.cv, args.weight_mid)
    else:
        run_table(p, args.per_clip, snap=not args.no_snap)


if __name__ == "__main__":
    main()
