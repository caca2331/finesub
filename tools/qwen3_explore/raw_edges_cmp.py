"""原生对照：两个 whisper 实现的词时间戳，按「段首 / 段尾 / 段内」分开比。

回答：`aligner_three` 里 wt~ow 只差 0.015 s，是方法本来就一样，还是工程抹平了？
——都不是。那个数字是重定时之后量的，段边界的分歧被洗掉了（donor 的段结构与 baseline
不重合，段边界处取到的其实是 donor 的段内时间）。

三个必须守住的口径（各错过一次）：

1. **参数与生产一致**。`refine_whisper_precision` 生产用 1.0，是 wt 自带的边界精修。
   最初把它关成 0 再去量段边界 = 关掉被测机制本身，段首分歧虚高一倍（0.360 vs 0.180）。
   它只动边界：同一次解码下段内移动中位恰好 0.000 s，解码结果逐词不变。
2. **「与生产一致」指有效行为，不是名义参数值**。`asr_align.py` 没有设
   `condition_on_previous_text`，名义上取 whisper 的默认 `True`（`asr_wt.py` 里那个
   `False` 属于独立工具，无人 import）。但生产**不做整片解码**——`build_combined_audio`
   按 VAD 分组拼音频、`group_target_sec=30 s`，而 whisper 的内部窗口也是 30 s，每次
   `transcribe` 基本只有一个窗口，「上一段文本」不存在，**有效等价于 `False`**。
   本对照因此用 `False` 整片跑；照抄名义值会累积几分钟的条件漂移（两边相似度从
   0.88-0.98 掉到 0.52-0.60），那是生产从不会有的状态。
3. **两侧必须同时判为该类型**。生产的 segment 是 VAD 区间分组，原生的是解码器分段，
   不是同一种东西；只按一侧判定会混进「归属不一致」的位置（中位 0.205 s），
   曾把段首的工程移动量虚报成 0.330 s（实际 0.025 s）。

实测（214/225/2081 个位置，文本相似度 0.882-0.984）：

    位置       wt(refine=0) vs ow   wt(refine=1.0) vs ow   生产工程移动量
    段首词·头   0.360 s              0.180 s                0.025 s
    段尾词·尾   0.080 s              0.020 s                0.115 s   <- 能量补齐
    段内词      0.020 s              0.020 s                0.010 s

结论：两个实现只在段内与段尾一致；「segment 从哪里开始」始终分歧，即便带 refine 也是段内
的 9 倍。生产在 refine 之上的额外工程只动段尾。0.015 s 是**段内**噪声地板，不能外推到切点。

    python tools/qwen3_explore/raw_edges.py        # 解码 wt refine=0 / refine=1
    python tools/qwen3_explore/ow_words.py         # 解码 ow
    python -m tools.qwen3_explore.raw_edges_cmp
"""

from __future__ import annotations

import collections
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .common import baseline_aligned  # noqa: E402

STRIP = "。．｡.!！?？…‥、，,､；;：:「」『』（）()　 \t\n・-–—"
CLIPS = ("BV1kYLR6AEXv", "BV1UBjq6fEgb", "BV1ySjz6FEzD")


def strip(s: str) -> str:
    return "".join(c for c in s if c not in STRIP)


def flat(segs) -> list[tuple]:
    """(text, start, end, 是段首, 是段尾)；去标点后为空的词丢弃。"""
    out = []
    for s in segs:
        ws = s.get("words") or []
        for wi, w in enumerate(ws):
            t = strip(w.get("word") or w.get("text", ""))
            if t:
                out.append((t, float(w["start"]), float(w["end"]), wi == 0, wi == len(ws) - 1))
    return out


def pairs(a_words, b_words):
    sm = difflib.SequenceMatcher(None, [x[0] for x in a_words], [x[0] for x in b_words],
                                 autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            yield a_words[i + k], b_words[j + k]


def run(get_a, get_b, title: str, extra: bool = False) -> None:
    buckets: dict[str, list[float]] = collections.defaultdict(list)
    sim = []
    for clip in CLIPS:
        a, b = get_a(clip), get_b(clip)
        sim.append(difflib.SequenceMatcher(None, "".join(x[0] for x in a),
                                           "".join(x[0] for x in b), autojunk=False).ratio())
        for x, y in pairs(a, b):
            # 两侧必须同时判为该类型，否则比的不是同一种东西
            if x[3] and y[3]:
                buckets["段首词·头"].append(abs(x[1] - y[1]))
            if x[4] and y[4]:
                buckets["段尾词·尾"].append(abs(x[2] - y[2]))
            if not x[3] and not y[3]:
                buckets["段内词·头"].append(abs(x[1] - y[1]))
            if not x[4] and not y[4]:
                buckets["段内词·尾"].append(abs(x[2] - y[2]))
            if extra and x[3] != y[3]:
                buckets["段首归属不一致"].append(abs(x[1] - y[1]))

    print(f"\n{title}   文本相似度 {min(sim):.3f}-{max(sim):.3f}")
    print(f"{'位置':16} {'n':>6} {'中位':>8} {'p90':>8} {'>0.1s':>7} {'>0.3s':>7} {'最大':>7}")
    for name, vals in buckets.items():
        v = sorted(vals)
        q = lambda f: v[min(int(len(v) * f), len(v) - 1)]  # noqa: E731
        print(f"{name:16} {len(v):>6} {q(.5):>8.3f} {q(.9):>8.3f} "
              f"{sum(x > 0.1 for x in v) / len(v):>6.1%} "
              f"{sum(x > 0.3 for x in v) / len(v):>6.1%} {v[-1]:>7.2f}")


def raw4(clip: str, key: str) -> list[tuple]:
    return flat(json.loads(Path(f"out/qwen-explore/{clip}-raw4.json").read_text(encoding="utf-8"))[key])


def ow(clip: str) -> list[tuple]:
    return flat(json.loads(Path(f"out/qwen-explore/{clip}-raw2.json").read_text(encoding="utf-8"))["ow"])


def prod(clip: str) -> list[tuple]:
    segs = json.loads(baseline_aligned(clip).read_text(encoding="utf-8"))["segments"]
    return flat([s for s in segs if s.get("words")])


def main() -> None:
    run(lambda c: raw4(c, "wt_r0"), ow, "wt(refine=0，非生产值) vs ow")
    run(lambda c: raw4(c, "wt_r1"), ow, "wt(refine=1.0，生产值) vs ow")
    run(lambda c: raw4(c, "wt_r0"), lambda c: raw4(c, "wt_r1"),
        "refine 自己移动了多少（同一次解码，逐词相同）")
    run(prod, lambda c: raw4(c, "wt_r1"),
        "生产 aligned vs 原生 wt(refine=1.0)——工程移动了多少", extra=True)


if __name__ == "__main__":
    main()
