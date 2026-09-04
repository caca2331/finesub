# 数据与基线索引

本文回答一个问题：**"那份数据/那个基线在哪，能不能再用"**——但只回答**规则**那一半。

分三类：**跟踪的标注数据**（进 git，可复现评测）、**未跟踪的本地素材**（gitignore，
机器上才有）、**只存在于文档里的基线数字**（没有原始数据，重跑要重新测）。
每条注明**能否重新生成**——这是判断它值不值得保护的关键。

⚠ **具体素材不写进本文。** BV 号、主播名、第三方内容、本机路径一律记在本地索引里：
本文随公开快照发布（`publish-main.ps1` 的 `$PrivatePaths` 不含 `docs/data-index.md`），
而那些东西没有公开的用处，只有公开的代价。同样的理由已经让
`agent-tasks/run-audit/evals` 进了 `$PrivatePaths`。

| 找什么 | 去哪 |
| --- | --- |
| **本地素材、产物、基线的逐条清单**（含怎么重建） | **`data/index.md`**（本地，不进 git） |
| 原始媒体逐条清单 | `assets/index.md`（本地） |
| 人工精修字幕对照组：清单、口径、读数 | `data/manually-refined-subs/<系列>/` 各自的 `README.md` / `analysis.md`（本地）；判读规则见 [`knowledge.md`](knowledge.md)「精修字幕怎么读」 |
| 跟踪的标注数据的规范与打分口径 | 各自 `tools/*/README.md` 与下方第一节点名的文档 |

**`data/` 与 `out/`、`tmp/` 的区别**：后两者是产物目录，随时可能被整目录删掉重跑；
`data/` 放**输入与参考资料**——源音频、人工标注、精修字幕，**不会被重跑清理**。
人工产出的东西放这里，不要放 `out/`。

---

## 一、跟踪的标注数据（人工产出，丢了要重标）

标注本身进 git，所以 clean checkout 拿得到；但**重新标注或改判定口径所需的输入只在本机**
（位置见 `data/index.md`）。

### VAD 争议片段听审 · `tools/vad_tuning/step0_labels/`

**144 个片段的人工标注**（真语音/听不清/语气词/幻觉/噪声抖动）＋全特征 join。
-45 峰值底线、ghost-drop、voicing 门控 cap 等判据的标定依据；采集协议见
`tools/vad_tuning/v26_step0.py` 与 FINDINGS 附录 T。音频切片可由 v26 重生成。


### 词起点边界标注 · `tools/wt_refine_validation/disfluency_gold.json`

**61 个人工标注块**，标注者给出每个 disfluency 块之后那个词的**真实起点**。
按位置分 `segment-boundary`(11) / `after-gap`(10) / `mid-phrase`(40)，
按判定分 `word_onset` / `partial` / `filled_pause`。

- 规范、位置优先级与已知歧义：[`../tools/wt_refine_validation/README.md`](../tools/wt_refine_validation/README.md)
- 生成脚本：`build_disfluency_gold.py`（三方对齐：普通 run + disfluency run + 人工修正 SRT）
- 已用它得出的结论：[`wt-refine-port.md`](wt-refine-port.md) 的「词起点边界准确度」
- ⚠️ **不适合无保留地做跨模型比较**：标注是在一次 turbo 系运行的词级输出上修改的，
  同模型的 arm 天然占便宜。理由与正确用法见该 README。

### 分割点金标准 · `tools/segmentation_gold/labels/`

**14 个标注窗口 / 1205 条标签**（must 224 / ok 178 / never 760 / unknown 43，每条带
`why`），人工标注必切/禁切/宜切。含 `substrate_sha` 锁定底稿，worksheet 保留标注过程。
规范与打分口径见 [`segmentation-gold.md`](segmentation-gold.md)。

⚠ **底本是冻结的，不要「顺手重跑一下」。** 标签用词索引 `k` 指向具体词流，
重跑 ASR 会让 1205 条全部失效——不是过期，是**指向错位**。§8 记着的 `ok` 档塌缩
说明重标不便宜也不安全。底本位置与已清理的可再生产物见 `data/index.md`。

⚠ **底本「旧」在哪、什么时候不能用它**：产于 2026-07-18，之后 ASR 链路 26 次提交。
它们**没跑过 `split_segments`**（`metadata.asr_align` 无 `segment_split` 键）——这对
分割研究反而是优点（段起点即接缝本体，`bench-baselines.md` 19.7 靠的就是这点），
但没有 `whisper_segment_start` 词标记、没有 `qwen_verify` 证据、没有电平档位标记。
**要量今天的管线行为，用当前管线基线（`data/index.md` 第二节），不要用这批。**

### 异常 group 语料 · `tools/wt_refine_validation/manifest.json`

**13 个 group**（3 个 control + 10 个历史异常倾向），从生产证据里挑出来的。
不是人工标注，是**筛选结果**——可以按同样标准重新筛，但那批具体样本的连续性会断。
用途与结论见 [`wt-refine-validation.md`](wt-refine-validation.md)。

### 知识库样例 · `examples/knowledge/`

跟踪的迷你知识库样本，不是活的 `knowledge/` 树。见 [`knowledge.md`](knowledge.md)。

---

## 二、未跟踪的本地素材

`assets/`、`data/`、`out/`、`tmp/` 全部不进 git，**逐条清单在 `data/index.md`**。
本文只留一条判断规则：产物目录（`out/`、`tmp/`）随时可以整个删掉重跑，删之前把结论
写进对应的 owner 文档；`data/` 与 `assets/` 是输入，删了就要重标或重下。

## 三、只存在于文档里的基线（无原始数据）

这些是实测结论，**没有保存中间数据**；要复核就得按文档记的方法重跑。

| 基线 | 文档 | 可复现性 |
| --- | --- | --- |
| GPU 档位显存标定（复测口径为 entry/standard/high 三档） | [`gpu-profiles.md`](gpu-profiles.md) | 换卡必须重测（机器特性）；5070 Ti 复测记于 2026-09-01 一节。后加的 `standard_large_vram` 与 `cpu` 不需要各自标定，理由在该文开头 |
| BS-Roformer 推理效率 E0–E16（AMP / 编译路径 / worker 阶梯 / torch 2.11 迁移 / roofline / demix runner） | [`separator-optimization.md`](separator-optimization.md) | 产物已删，**素材与工具可重建**——见下。**分三段读**：E0–E10 取自 torch 2.9.0；E11–E13 在生产钉版 2.11.0 上，仍是 RTX 5060 Ti；**E14 起换机（RTX 5070 Ti），绝对时间不可跨段比较** |
| 块产物固定 FLAC 的四 run 对照、以及交付形态（16k 单声道 / 档位）的取舍实测（2026-08-18） | [`separator-optimization.md`](separator-optimization.md)「块产物固定为 FLAC」 | 素材可再生，命令见 `data/index.md` 第四节；四 run 共约 1 分钟 |
| WT 分片并发曲线、损失分解 | [`wt-parallelism.md`](wt-parallelism.md) | **实现已删**，只作历史 |
| fw-refine vs wt 迁移验收（5 素材 / 50.6 分钟） | [`wt-refine-port.md`](wt-refine-port.md) | 产物在 `out/acceptance/`，可复核 |
| batch size × 模型 × beam 的成本矩阵 | [`wt-refine-port.md`](wt-refine-port.md) | 需重跑；口径见文中「口径边界」 |
| beam 不抑制幻觉（310 窗口） | [`wt-refine-port.md`](wt-refine-port.md) | 需重跑 |
| large-v3 vs turbo 异常率（310 窗口） | [`wt-refine-port.md`](wt-refine-port.md) | 需重跑；**未记 per-window 配对**，做不了配对检验 |
| 人声分离占语音段 72% | [`wt-refine-port.md`](wt-refine-port.md) | 单素材单次，被游戏负载影响过——绝对值不可信，比例可参考 |
| 救援阶梯取舍（2h12m 素材） | [`asr-align.md`](asr-align.md) | 需重跑 |
| 语言票翻转重解：根因实验、采纳判据三例、referee 常驻显存与开销（2026-08-19） | [`asr-align.md`](asr-align.md)「语言票翻转重解」 | 输入是 `out/reference/` 的 8 份产物（仍在，清单在本地 `data/index.md` 第三节），**实验脚本未存档**；相似度绝对值须用实现内置的 `SequenceMatcher.ratio` 重测。**一个真外语负例都没有**——标定所需的素材清单在该节「待标定」。方案原稿与逐轮编年在本地 `archive/speech-quality-plan.md` |
| 精修合并软门槛标定 | [`merge-calibration.md`](merge-calibration.md) | 需重跑 |


## 使用前必读的两条

1. **口径优先于数字。** 每份基线都在文档里写了口径边界（样本量、是否单次、是否有噪声基线、
   有没有已知偏向）。拿数字之前先读那一段——本项目已经出现过多次"数字对但口径不可比"的情况
   （例如冷启动 vs 预热的耗时、标注锚定造成的模型偏向）。
2. **A/B 需要独立输出目录。** pipeline 的 stage 跳过只看**文件存在性**，不校验内容。
   拿旧产物跑新代码会静默复用，看起来"跑过了"其实没有。
