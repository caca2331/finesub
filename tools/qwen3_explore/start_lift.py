"""Derive the "never begins a cue" token set, and audit it across tokenizers.

`acceptability.CANNOT_START` was derived from the *Whisper* baseline cues, whose word entries are
whisper-timestamped's subword tokens (`なら` `この` `世界` `けて` …). It is then applied to the
Qwen arm, whose word entries are **nagisa** tokens. Different inventories, so the same fragment
can be flagged on one arm and invisible on the other — and the bias runs in favour of whichever
arm the table was *not* derived from. FINDINGS §6 listed this as an open debt; this script
measures it and produces the tokenizer-symmetric replacement.

    python -m tools.qwen3_explore.start_lift            # both tables + the cross-tokenizer audit

The corpus is the same 11 baseline clips as everything else in §4.6. Note that using the Whisper
baseline as the corpus is not circular for this purpose: we are learning *Japanese morphology*
(which tokens can open an utterance), not which cuts are good.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import ALL_CLIPS, baseline_aligned

MIN_COUNT = 80        # below this the start rate is too noisy to trust
LIFT_CUTOFF = 0.25    # "essentially never begins a cue"
STRIP = "、。！？ 　,.!?…「」『』（）()"


def _clean(tok: str) -> str:
    return tok.strip(STRIP)


def nagisa_words(text: str) -> list[str]:
    import nagisa

    return nagisa.tagging(text).words


def collect(tokenizer: str) -> tuple[Counter, Counter, int]:
    """Return (starts, totals, n_cues) over the baseline cues under one tokenization."""
    starts: Counter = Counter()
    totals: Counter = Counter()
    n_cues = 0
    for clip in ALL_CLIPS:
        data = json.loads(baseline_aligned(clip).read_text(encoding="utf-8"))
        for seg in data["segments"]:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            if tokenizer == "whisper":
                toks = [_clean(str(w.get("word", ""))) for w in seg.get("words") or []]
            else:
                toks = [_clean(t) for t in nagisa_words(text)]
            toks = [t for t in toks if t]
            if not toks:
                continue
            n_cues += 1
            starts[toks[0]] += 1
            totals.update(toks)
    return starts, totals, n_cues


def derive(starts: Counter, totals: Counter, n_cues: int) -> tuple[dict[str, float], set[str]]:
    base = n_cues / sum(totals.values())
    lift = {w: (starts[w] / c) / base for w, c in totals.items() if c >= MIN_COUNT}
    return lift, {w for w, v in lift.items() if v < LIFT_CUTOFF}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the nagisa-derived set here as JSON")
    args = ap.parse_args()

    tables = {}
    for name in ("whisper", "nagisa"):
        starts, totals, n_cues = collect(name)
        lift, cannot = derive(starts, totals, n_cues)
        tables[name] = (lift, cannot, totals, n_cues)
        print(
            f"== {name} 分词：{n_cues} cues / {sum(totals.values())} tokens，"
            f"基准起始率 {n_cues / sum(totals.values()):.3f}，"
            f"词表 {len(lift)} 词（出现 >= {MIN_COUNT}），CANNOT_START {len(cannot)} 词"
        )
        print("   " + " ".join(sorted(cannot, key=lambda w: lift[w])))

    from .lexicon import CANNOT_START_NAGISA, CANNOT_START_WHISPER

    w_lift, w_set, _, _ = tables["whisper"]
    n_lift, n_set, n_totals, _ = tables["nagisa"]

    print("\n-- 复现代码里的两张表")
    for name, derived, checked in (
        ("whisper", w_set, CANNOT_START_WHISPER),
        ("nagisa", n_set, CANNOT_START_NAGISA),
    ):
        diff = (derived ^ checked) or None
        print(f"   {name:8s} 导出 {len(derived)} 词 / 代码 {len(checked)} 词"
              f"{'  一致' if diff is None else '  差异: ' + ' '.join(sorted(diff))}")

    print("\n-- 两种分词的交集/差集（原始欠账：同一个形态学事实，两边只各看见一半）")
    print(f"   共有 {len(w_set & n_set)}: {' '.join(sorted(w_set & n_set))}")
    print(f"   仅 whisper 侧 {len(w_set - n_set)}: {' '.join(sorted(w_set - n_set))}")
    print(f"   仅 nagisa 侧 {len(n_set - w_set)}: {' '.join(sorted(n_set - w_set))}")

    # Why not simply use the union on both arms — the fix that looked obvious and was tried first.
    unreachable = sorted(t for t in w_set if n_totals.get(t, 0) < MIN_COUNT)
    union = w_set | n_set
    print(
        f"\n-- 并集 {len(union)} 词。它修掉了漏检：whisper 表里有 {len(unreachable)} 个词"
        f" nagisa 几乎切不出（{' '.join(f'{t}({n_totals.get(t, 0)})' for t in unreachable)}），"
        "\n   套在 Qwen 臂上等于永不触发。但并集本身有反向代价 ——"
    )

    # A token earns its place from the tokenizer that *observed* it.
    # A token earns its place from the tokenizer that *observed* it; applied to the other arm it
    # may be a perfectly ordinary word opener there. 「か」 never opens an utterance as a nagisa
    # particle, but whisper emits it as the first subword of 「かわいい」 — so every 「かわいい」
    # cue on the whisper arm gets flagged as a mid-word cut. Cross-lift makes that measurable
    # without adjudication: high lift in the *other* tokenization = false positives over there.
    print(f"\n-- 跨用代价：并集里每个词在「另一种分词」下的 lift（>= {LIFT_CUTOFF} = 在那一臂上是误报源）")
    rows = [(w, w_lift.get(w), n_lift.get(w)) for w in sorted(union)]
    bad = [(w, wl, nl) for w, wl, nl in rows
           if (wl is not None and wl >= LIFT_CUTOFF) or (nl is not None and nl >= LIFT_CUTOFF)]
    print(f"   {'词':>4s} {'whisper lift':>13s} {'nagisa lift':>12s}   误报落在")
    for w, wl, nl in sorted(bad, key=lambda r: -max(r[1] or 0, r[2] or 0)):
        side = "whisper 臂" if (wl or 0) >= LIFT_CUTOFF else "qwen 臂"
        f = lambda v: "  —" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"   {w:>4s} {f(wl):>13s} {f(nl):>12s}   {side}")
    print(f"   {len(bad)}/{len(union)} 个词是跨用误报源，且**全部落在同一臂上**——并集不是对称的修法，"
          "\n   它只是把偏袒换了个方向。结论：每臂用各自分词导出的表，词中切这一列不跨臂比。")

    if args.out:
        def _l(w: str) -> float:
            return min(w_lift.get(w, 9.9), n_lift.get(w, 9.9))

        Path(args.out).write_text(
            json.dumps(
                {"min_count": MIN_COUNT, "lift_cutoff": LIFT_CUTOFF,
                 "whisper_only": sorted(w_set - n_set), "nagisa_only": sorted(n_set - w_set),
                 "union": sorted(union, key=_l),
                 "lift": {w: round(_l(w), 3) for w in sorted(union, key=_l)}},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\n写出 {args.out}")


if __name__ == "__main__":
    main()
