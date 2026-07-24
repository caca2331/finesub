# split explorer

`docs/segment_split.md` 的离线调参工具，**生产切分模块
`src/segment_split.py` 的薄封装**：读现成 aligned/stable JSON + 一次性缓存
的 VAD interval（不重跑 ASR），对全部 segment 跑 DP 打分切分，输出切分
报告（逐刀 g/T/B）、before/after 统计和可选 SRT。所有打分常量
（`SplitParams` 全字段）走 CLI 参数，调参即重跑（纯 CPU，秒级）。

VAD cache 同时写 `intervals` 与标准 `segments[{start,end}]`，所以也可直接作为
`asr_align.py` 的输入；一份 cache 即可复现 ASR 基线与后续离线切分。

```powershell
# 首次运行会从音频算一遍 VAD 并缓存到 <audio>.vadcache.json
python -m tools.split_explorer out/yui-exp/yui-cov2-stable.json \
  --audio out/yui-exp/yui-vocal.flac --srt out/yui-exp/yui-split.srt

# 调参示例；--seg N 打印某 segment 的全部边界打分
python -m tools.split_explorer ... --base 1.5 --g-knee 0.4 --no-gap-penalty 1 --seg 123

# 批量比较任意数量的真实 ASR stable 变体（条件参数可重复）
python -m tools.split_explorer.report run `
  --condition "baseline=-stable-a.json" `
  --condition "candidate=-stable-b.json"
```

## Gap policy 真实 ASR 实验

合成静音会改变 Whisper 解码、regroup、coverage rescue 和 fallback 控制流，不能
从另一种 gap 的 ASR 结果可靠回放。`asr_gap` 因此只支持真实模型运行；它在进程内
临时替换 gap policy，结束后恢复生产模块，不修改 `src/`：

```powershell
python -m tools.split_explorer.asr_gap run/BV-vad.json `
  --audio BV-vocal.flac --gap 0.2 --real-gap-max 0.5 `
  --output run/BV-aligned-real05-gap02.json --language ja

# 动态静音：min(0.1 + 0.2 * 原始 gap, 0.8)，仍至多保留 0.7s 原声
python -m tools.split_explorer.asr_gap run/BV-vad.json `
  --audio BV-vocal.flac --adaptive-gap 0.1 0.2 0.8 --real-gap-max 0.7 `
  --output run/BV-aligned-adaptive-gap.json --language ja
```

2026-07 的 8-BV 调参结论见
[`results/2026-07-18-8bv-gap-study.md`](results/2026-07-18-8bv-gap-study.md)。
原始音频、aligned/stable JSON 和逐刀长报告仍放 `out/`，不进入 Git。

打分公式与 gap word 调整逻辑的唯一实现在 `src/segment_split.py`
（规范见 `docs/segment_split.md`）；本工具不含独立打分代码，
不存在与主程序失同步的问题。

## 维护策略

**按需维护。** 不是生产 pipeline 的组成部分；修改 stable schema 或
VAD/ASR 算法时不要顺带更新此工具。
