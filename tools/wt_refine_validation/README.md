# WT refine 小验证集

该验证集专门用于 one-pass refine 信号、局部修正与隔离重解研究，不是生产回归测试，也不替代人工
字幕审核。选样冻结在 `manifest.json`：3 个普通 group、10 个已有历史异常证据的 group。异常样本来自
`out/mt8g-stress/stress.log` 中实际触发过生产隔离的窗口，参考文本使用同源人工纠正 SRT。

manifest 不复制大音频，只保存相对 corpus root 的来源、生产 VAD group selector、历史问题和路由。
runner 会重新执行当前 VAD/grouping，并要求 selector 精确命中已有 interval；这既复现生产输入，也会在
VAD 参数漂移时显式失败，而不是悄悄换样本。

默认研究原则：

- refine 只产生可审计事件和局部边界候选，不静默删文本；
- 局部、确定且隔离重解能提高质量时，优先在 ASR controller 处理；
- 质量近似但 refine/ASR 处理能减少重解音频秒数时，同样优先；
- 需要改变分组或比较邻段的事件，只记录为 deferred，不在本工具中模拟 FineSub 决策；
- 所有重试保留原候选、事件、处理音频秒数和结果，不把“检测命中”等同于“质量提升”。

信号分两档：`collect_refine_signals` 只使用已经返回的 compact DTW path，收集
`alignment_stack`、`long_token_span`、`decoder_repetition`、`unfinished` 和
`zero_duration_chunk_tail`；
`collect_attention_signals` 才额外要求 CT2 返回后处理 attention，供 disfluency 与边界不确定性研究。
验证 runner 两档都开，生产接入时可只开低成本 path 信号。

大音频根目录通过 `--corpus-root` 显式指定，例如原始开发仓库；生成结果只能写入 `out/`：

```powershell
python -m tools.wt_refine_validation.run `
  --corpus-root C:/Users/Carl/Documents/Carl/projects/asr-playground `
  --model C:/Users/Carl/Documents/Carl/models/faster-whisper-large-v3-turbo `
  --ct2-python C:/Users/Carl/Documents/Carl/projects/CTranslate2/python/build/wt-refine-runtime-wide `
  --ct2-bin C:/Users/Carl/Documents/Carl/projects/CTranslate2/install-cu-wide/bin `
  --cuda-bin "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin" `
  --output out/wt-refine-validation/full.json
```

结果逐 case 保存 baseline、结构化事件、研究路由、历史 target 的隔离 probe、人工参考代理、覆盖与
`compute_sec_per_audio_sec`；顶层 `summary` 汇总分桶路由、信号命中 case 数和 policy 重试音频比例。

## 生产产物离线信号审计（`artifact_survey.py`）

不依赖 GPU/patched CT2 的只读第二入口：扫描已完成 run 的 aligned/stable JSON，把最终输出里的
`alignment_events[]` 与词级判定、已知幻觉短语、stable 丢弃/打标做段级对照。口径限制与
2026-08-04 的结论见 [`docs/wt-refine-validation.md`](../../docs/wt-refine-validation.md)
「生产产物离线复核」一节——它只看得到救援后的存活解码，测不出 decode-time recall。

```powershell
python -m tools.wt_refine_validation.artifact_survey out/acceptance --report tmp/signal-survey.json
```

## 生产窗口覆盖率对比（`window_sweep.py` + `window_score.py`）

`window_sweep.py`（GPU）把保存的 VAD 轨按当前生产规则重新分组，对每个 ≤30s 窗口做一次
生产配置 greedy 解码（无救援），逐窗 dump 词级规则命中、path 事件、覆盖率；
`window_score.py`（离线）在 dump 上计算每个现有规则与信号候选判定器的命中，配合人工
裁决标注输出覆盖率/误报对照表。2026-08-04 的 405 窗口结果与结论见
[`docs/wt-refine-validation.md`](../../docs/wt-refine-validation.md)「生产窗口覆盖率对比」；
裁决标注入库为 `window_sweep_labels_20260804.json`（window id 绑定当次分组，分组参数漂移
后需重扫重标）。2026-08-05 VAD 改版后的重扫标注在
`window_sweep_labels_20260805_vadv2.json`（重叠映射 + 新窗补裁决；旧异常未复现的窗按
新解码改判 benign，见文件 `_meta`），对应 `out/qwen-explore-vadv2/` 的轨与 sweep。

```powershell
python -m tools.wt_refine_validation.window_sweep `
  --vad-dir out/qwen-explore --corpus-root . `
  --model C:/Users/Carl/Documents/Carl/models/faster-whisper-large-v3-turbo `
  --output tmp/window-sweep.jsonl
python -m tools.wt_refine_validation.window_score tmp/window-sweep.jsonl `
  --labels tools/wt_refine_validation/window_sweep_labels_20260804.json
```

## disfluency 人工标注集（`disfluency_gold.json`）

`BV1cqLR6hEp3`（532s，日语）上对 WT `detect_disfluencies` 的 61 个候选块的人工标注，由
`build_disfluency_gold.py` 从「同一 clip 的普通 run + disfluency run + 人工修正词级 SRT」三方对齐生成。

派生所需的输入在 **`data/disfluency-gold/BV1cqLR6hEp3/`**（本地保留，不进 git）：

```bash
python tools/wt_refine_validation/build_disfluency_gold.py   data/disfluency-gold/BV1cqLR6hEp3 --clip BV1cqLR6hEp3   -o tools/wt_refine_validation/disfluency_gold.json
```

只有派生产物 `disfluency_gold.json` 进 git；原件留在本机的 `data/`（输入与参考资料目录，
不会被重跑清理，与产物目录 `out/`、`tmp/` 分开）。

标注**不是删/留二元**：每个块都是一个区间，后词的真实起首落在其中某处，修正把它放在——

| 标签 | n | 含义 | 块时长中位 | 普通 run 词首相对金标准 |
| --- | --- | --- | --- | --- |
| `filled_pause` | 32 | 起点在块尾：整块是语气停顿 | 334 ms | **偏早 273 ms** |
| `partial` | 4 | 起点在块内：部分停顿、部分词首 | 302 ms | 偏早 184 ms |
| `word_onset` | 25 | 起点在块首：整块是后词起首 | 160 ms | **0 ms（本来就对）** |

因此每条的金标准值是 `onset`（绝对时间）与 `onset_fraction`（0..1），既可做「是不是真语气词」的
分类，也可做「词首究竟在哪」的回归。

### 位置是最强的判别特征

位置相对生产分段判定：块本身是某 segment 的首词或末词、或其后词开启新 segment，记
`segment-boundary`；否则前置 gap ≥ 50 ms 记 `after-gap`。

| 位置 | `filled_pause` | `partial` | `word_onset` | 检测有效率 |
| --- | --- | --- | --- | --- |
| segment-boundary | 7 | 2 | 2 | **73%** |
| after-gap | 8 | 2 | 0 | **90%** |
| mid-phrase | 17 | 0 | 23 | **42%** |

**在 segment 边界与 gap 之后，`detect_disfluencies` 基本可信；在短语中部近乎抛硬币。** 这是
handoff P1「给句首 `disfluency_candidate` 增加声学门控」最直接的依据——单靠位置就能吃掉大部分收益，
声学门控只需处理剩余部分。块时长同向可分（真语气词 334 ms vs 误切 160 ms），前词以句末标点收尾
在真语气词侧也显著更多。

### 使用优先级（标注者指定）

1. **`segment-boundary` 是重点**——句首时间戳直接决定切点不确定性，评估与门控应以该子集为准；
2. `after-gap` 次之；
3. `mid-phrase` 仅作参考：标注者认为这些块**由后词吸收问题不大**，不必当作必须修正的目标。

### 已知歧义

`mid-phrase` 的 23 个 `word_onset` 里，后词多为 `の`/`を`/`に`/`が` 等助词，标注者观察到这些块
**实际上往往是前词的尾音**被切成了后词起首。约定统一把块归给后词，该歧义逐条记在
`annotator_note`；按上面的优先级，该子集本就不作为修正目标，歧义不影响重点结论。

另含 8 处人工时间戳修正（4 处句末提前结束，延长 115–395 ms；3 处词首偏晚，回退 65–172 ms；
1 处保留块收窄）与 1 处坍缩摊平（`クロンビーナー!` + `ん` → 5 个词）。

### 已用它测出的结论

2026-08-02 用本数据集给 fw-refine 的词起点打分（结果表在
[`docs/wt-refine-port.md`](../../docs/wt-refine-port.md) 的「词起点边界准确度」一节）：
fw-refine 在 `segment-boundary` 上与今日生产（WT 关闭 disfluency）**打平**，
两者都把 filled pause 吞进后词、起点提早 0.26–0.33s；只有 `after-gap` 有稳定改善。

⚠️ **本数据集不适合无保留地做跨模型比较。** 标注是在一次 turbo 系运行
（WT backend + `large-v3-turbo`）的词级输出上修改而成，块首块尾都是那次运行的词边界，
同模型的 arm 天然占便宜——25 个 `word_onset` 行的「真实起点」直接等于该运行的块首，
turbo 在其上 19 胜 2 负，足以主导全局数字。跨模型比较应只看 `segment-boundary`
（必要时再交叉 `filled_pause`），并明说残留偏向。

评分口径三条要点：**逐条配对的胜/负/平比中位数更可信**（本例中位数从 217ms 降到 68ms，
配对却是 14/16/29）；**「WT 开启 disfluency」不是独立 arm**——`filled_pause` 行的标注起点
就取自该运行，误差恒为 0；**全局中位数会被 40 个 mid-phrase 行主导**，而标注者已声明该子集
不是修正目标。
