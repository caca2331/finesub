# 数据与基线索引

本文回答一个问题：**"那份数据/那个基线在哪，能不能再用"**。

分三类：**跟踪的标注数据**（进 git，可复现评测）、**未跟踪的本地素材**（gitignore，机器上才有）、
**只存在于文档里的基线数字**（没有原始数据，重跑要重新测）。

每条注明**能否重新生成**——这是判断它值不值得保护的关键。

---

## 一、跟踪的标注数据（人工产出，丢了要重标）

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
- **原始标注与派生所需的输入在 `data/disfluency-gold/BV1cqLR6hEp3/`**（本地，不跟踪，约 428 KB）：
  人工修正的 `-fixed.srt`、disfluency run 的 `stable.json` 与 word SRT、普通 run 的 word SRT。
  已验证可从这些源重新派生出与跟踪的 JSON **完全一致**的结果：
  ```bash
  python tools/wt_refine_validation/build_disfluency_gold.py     data/disfluency-gold/BV1cqLR6hEp3 --clip BV1cqLR6hEp3 -o tmp/regen.json
  ```
  ⚠️ **`data/` 不进 git，所以 clean checkout 只有派生产物 `disfluency_gold.json`。**
  重新标注或改判定口径需要这台机器上的原始文件。
- ⚠️ **不适合无保留地做跨模型比较**：标注是在一次 turbo 系运行的词级输出上修改的，
  同模型的 arm 天然占便宜。理由与正确用法见该 README。

### 分割点金标准 · `tools/segmentation_gold/labels/`

**14 个标注窗口**，人工标注必切/禁切/宜切。含 `substrate_sha` 锁定底稿，
worksheet 保留标注过程。规范与打分口径见 [`segmentation-gold.md`](segmentation-gold.md)。

### 异常 group 语料 · `tools/wt_refine_validation/manifest.json`

**13 个 group**（3 个 control + 10 个历史异常倾向），从生产证据里挑出来的。
不是人工标注，是**筛选结果**——可以按同样标准重新筛，但那批具体样本的连续性会断。
用途与结论见 [`wt-refine-validation.md`](wt-refine-validation.md)。

### 知识库样例 · `examples/knowledge/`

跟踪的迷你知识库样本，不是活的 `knowledge/` 树。见 [`knowledge.md`](knowledge.md)。

---

## 二、未跟踪的本地素材（gitignore，只在这台机器上）

`assets/`、`data/`、`out/`、`tmp/` 全部不进 git。以下是被文档和实验反复引用的：

**`data/` 与 `out/`、`tmp/` 的区别**：后两者是产物目录，随时可能被整目录删掉重跑；
`data/` 放**输入与参考资料**——源音频、人工标注、精修字幕，**不会被重跑清理**。
人工产出的东西放这里，不要放 `out/`。

| 位置 | 内容 | 谁在用 |
| --- | --- | --- |
| `assets/` | **全部原始媒体**（约 3.4 GB，含 `bilibili/` 下 8 个 reference 素材） | 逐条清单与关联产物位置见 **[`../assets/index.md`](../assets/index.md)** |
| `out/qwen-explore/*-vad.json` | 11 个 clip 的旧版 VAD 轨 | 310/405 窗口 sweep 的输入（beam 对照、模型对照、分组统计）；对应标注 `tools/wt_refine_validation/window_sweep_labels_20260804.json` |
| `out/qwen-explore-vadv2/` | 同 11 clip 的 **2026-08-05 改版 VAD 轨** + 能量 npz + pause_hints + 400 窗 sweep dump | VAD 改版后复测与词首修正标定的输入；对应标注 `window_sweep_labels_20260805_vadv2.json`（跟踪） |
| `out/reference/<id>/` | reference ingest 全套产物 | 精修对照、知识库、词起点标注的底稿 |
| `out/acceptance/<clip>/` | wt vs fw-refine 迁移验收产物 | [`wt-refine-port.md`](wt-refine-port.md) 的「迁移验收」；两侧 aligned/stable/srt + stderr 日志都在 |
| `data/disfluency-gold/BV1cqLR6hEp3/` | 词起点标注原件 + 三份源 run | `build_disfluency_gold.py`，见上 |

**这些都不可再生**：`assets/` 是外部素材，`out/` 是长时间累积的运行产物。
它们是上面那些结论的原始证据——文档里的数字全部由它们算出。

原始媒体已于 2026-08-02 从 `out/reference/<id>/` 集中到 `assets/bilibili/`。
**副作用：`llm-reference-ingest` 重跑会认为媒体缺失并重新下载**（它按 `<id>.ogg` 是否存在判断），
要避免就把需要的 `.ogg` 拷回对应 `out/reference/<id>/`。

---

## 三、只存在于文档里的基线（无原始数据）

这些是实测结论，**没有保存中间数据**；要复核就得按文档记的方法重跑。

| 基线 | 文档 | 可复现性 |
| --- | --- | --- |
| 4/8/12/16GB profile 显存标定 | [`gpu-profiles.md`](gpu-profiles.md) | 换卡必须重测（机器特性） |
| BS-Roformer 推理效率 E0–E11（AMP / 编译路径 / worker 阶梯 / torch 2.11 迁移） | [`separator-optimization.md`](separator-optimization.md) | 产物已删，**素材与工具可重建**——见下。注意 E0–E10 取自 torch 2.9.0，只有 E11 在生产钉版 2.11.0 上重取 |
| WT 分片并发曲线、损失分解 | [`wt-parallelism.md`](wt-parallelism.md) | **实现已删**，只作历史 |
| fw-refine vs wt 迁移验收（5 素材 / 50.6 分钟） | [`wt-refine-port.md`](wt-refine-port.md) | 产物在 `out/acceptance/`，可复核 |
| batch size × 模型 × beam 的成本矩阵 | [`wt-refine-port.md`](wt-refine-port.md) | 需重跑；口径见文中「口径边界」 |
| beam 不抑制幻觉（310 窗口） | [`wt-refine-port.md`](wt-refine-port.md) | 需重跑 |
| large-v3 vs turbo 异常率（310 窗口） | [`wt-refine-port.md`](wt-refine-port.md) | 需重跑；**未记 per-window 配对**，做不了配对检验 |
| 人声分离占语音段 72% | [`wt-refine-port.md`](wt-refine-port.md) | 单素材单次，被游戏负载影响过——绝对值不可信，比例可参考 |
| 救援阶梯取舍（2h12m 素材） | [`asr-align.md`](asr-align.md) | 需重跑 |
| 精修合并软门槛标定 | [`merge-calibration.md`](merge-calibration.md) | 需重跑 |

### 分离器优化的素材与产物（2026-08-03 清理）

`out/separator-opt/` 已整目录删除（约 2GB：4 份 AOTI package、几十份对照 FLAC 与逐次
JSON）。所有结论都已落到 [`separator-optimization.md`](separator-optimization.md)，
剩余待探索项的预期收益不足以让它长期占盘。重跑需要的三样东西：

| 需要什么 | 从哪来 | 代价 |
| --- | --- | --- |
| 质量素材 270.016s | `out/reference/BV1kYLR6AEXv/BV1kYLR6AEXv-source.wav` | 仍在；丢了可从 `assets/bilibili/` 重新 ingest |
| 性能素材 700.032s | `out/mt8g-stress/clip700.ogg` | 仍在 |
| 长素材 1400.072s（E7 的 worker 阶梯要它——700 秒会被时长阶梯封顶在 3 个 worker） | 由 clip700 自拼两遍：`ffmpeg -f concat -safe 0 -i concat.txt -c:a flac tmp/clip1400.flac`，`concat.txt` 是同一个 `clip700.ogg` 写两行 `file '...'` | 约 1 分钟 |
| 4 份无权重 AOTI package | `python -m tools.separator_aoti OUTPUT_DIR`（默认即最终配置） | 约 30 秒，需钉版的 Torch 2.11 + Triton 3.6 环境与 MSVC。生产会在 `cache/separator-accel/<key>/aoti/` 自己建一份，这条只在需要建**变体**时用 |

基准工具本身（`tools/separator_benchmark.py`、`tools/separator_aoti.py`）进了 git，
实验开关和协议见文档的「固定协议」。

---

## 使用前必读的两条

1. **口径优先于数字。** 每份基线都在文档里写了口径边界（样本量、是否单次、是否有噪声基线、
   有没有已知偏向）。拿数字之前先读那一段——本项目已经出现过多次"数字对但口径不可比"的情况
   （例如冷启动 vs 预热的耗时、标注锚定造成的模型偏向）。
2. **A/B 需要独立输出目录。** pipeline 的 stage 跳过只看**文件存在性**，不校验内容。
   拿旧产物跑新代码会静默复用，看起来"跑过了"其实没有。
