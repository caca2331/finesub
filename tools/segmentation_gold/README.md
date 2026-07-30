# segmentation_gold — 分割点金标准

人工标注的「必切 / 禁切 / 宜切」标签集，用来给**任何**分割器打分，并反过来校准机械指标。

**标注规范、判据、打分口径、批次管理全部在 [`docs/segmentation-gold.md`](../../docs/segmentation-gold.md)。**
本文只列命令。改标签词表或候选位规则之前必须先读那份文档——它们承载着分母。

```bash
# 1. 冻结一个窗口的标注工作表（工作表生成后不要再改，标签按 i 引用）
python -m tools.segmentation_gold.gold prepare --clip yui \
    --words ../asr-playground/out/yui-exp/yui-split-aligned.json \
    --vad out/qwen-explore/yui-vad.json --window 660,780 \
    --reference ../asr-playground/out/yui/yui-corrected.srt \
    --out tools/segmentation_gold/worksheets/ws-yui.md

# 2. 人工标注 -> labels/<clip>-<start>-<end>.json，然后校验
python -m tools.segmentation_gold.gold validate

# 3. 给任意分割结果打分（srt 或 {segments:[{start,end,text}]} json 都行）
python -m tools.segmentation_gold.gold score --seg out/qwen-explore/yui-QS.json --clip yui
```

目录：`worksheets/` 冻结的工作表（标注输入）· `labels/` 标签（成果，必须跟踪）。

维护策略：**按需**。它不属于生产链路，默认测试套件不收集它的测试。但 `labels/` 是**手工
产出的长期资产**，不是可再生的中间产物——不要放进 `out/`（gitignore），不要随便重标。
