# onset_gap_energy — 词起点声学修正探索

**维护策略：按需**。这是一次探索的现场，不是生产代码，也不进默认测试。别在无关改动里
顺手更新它；要用结论直接读 [`FINDINGS.md`](FINDINGS.md)。

## 这里在回答什么

生产（WT 关闭 disfluency）会把语气停顿后的词起点提早 0.3s 量级。本目录探索：
**能不能只靠 VAD 区间或帧级能量，把段首、以及 gap 之后的词起点挪到正确位置。**

答案在 `FINDINGS.md`：gap 后词可以（中位 0.291 → 0.095，配对 6 胜 0 负），
段首基本不行（浊音语气词没有静音可找）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `common.py` | 读金标准 / VAD，统一的误差汇总 |
| `features.py` | 帧级轨（10 ms hop）：能量、低/高频带、频谱通量、onset 包络；npz 缓存 |
| `gap_onset.py` | **候选启发式本体**，四个门控参数都写了存在理由 |
| `detectors.py` | 被否定的几种前向搜索（首个能量上升 / 最后一段静音 / onset 峰） |
| `s1`–`s10` | 按顺序的实验脚本，每个文件头说明它回答哪一问 |

## 数据依赖

金标准 `tools/wt_refine_validation/disfluency_gold.json` 进 git；音频与产物
（`out/reference/<id>/<id>-vocal.flac`、`<id>-stable.json`、`out/qwen-explore/<id>-vad.json`）
是本机的，全部通过命令行参数传入。见 `docs/data-index.md`。

## 与既有工作的关系

`exp/seg-start-onset`（v1–v34）研究的是**换掉 wt / 用 ow/fw 时**的段首漂移，
用的是 wt 作伪金标准。本轮不同：目标是**今日生产（WT）**的残余误差，金标准是人工标注。
两边都否定了频谱平坦度与 cross-attention onset 一类的信号，结论一致。
