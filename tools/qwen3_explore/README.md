# qwen3_explore — Qwen3-ASR / Qwen3-ForcedAligner 探索工具

一次性的探索脚本（按需维护，不进生产链路，默认测试套件不收集）。
**结论见 [FINDINGS.md](FINDINGS.md)**（只留结论与判据；推理过程与被推翻的中间结论
已精简，在 git 历史里）。本文只讲怎么跑。

## 环境

两个环境，分工是刻意的：

- **`qwen-asr`**（新建）：跑 Qwen 两个模型。`transformers>=5.13` 与生产 `asr` 环境的
  4.53.3 不兼容（同环境有 pyannote-audio / funasr / speechbrain / whisper-timestamped）。
- **生产 `asr`**：跑 `vad_energy` / `segment_split` / `subtitle_metrics` 等生产模块，以及
  需要 whisper 作第二验证器的脚本。

```powershell
conda create -n qwen-asr python=3.12 -y
conda activate qwen-asr
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
pip install "transformers>=5.13" accelerate soundfile librosa jiwer nagisa pykakasi
```

`nagisa` 是硬依赖：`split_words_for_alignment` 在 `language=Japanese` 时强制走它。
`pykakasi` 只给 `phonetic.py` 取假名读音。模型首次自动下载到 `~/.cache/huggingface`，
`Qwen/Qwen3-ASR-1.7B-hf` + `Qwen/Qwen3-ForcedAligner-0.6B-hf` 合计约 6 GB。

约定：`ENV_Q` = `C:/Users/Carl/miniconda3/envs/qwen-asr/python.exe`，
`ENV_A` = `C:/Users/Carl/miniconda3/envs/asr/python.exe`。

## 主链路（当前推荐的跑法）

```bash
# 1. 共享 VAD 语音区间 —— 两臂必须用同一份，否则比的是 VAD 噪声        [ENV_A]
$ENV_A -m tools.qwen3_explore.dump_vad --audio X-vocal.flac --out out/qwen-explore/X-vad.json

# 2. Qwen ASR + 对齐 + 救援阶梯（有界三级 / token 预算 / 覆盖率只算 >=1s 区间）  [ENV_Q]
$ENV_Q -m tools.qwen3_explore.rescued_asr --audio X-vocal.flac \
    --vad out/qwen-explore/X-vad.json --out out/qwen-explore/X-Q-raw.json --language ja

# 3. 分句：单次全局 DP                                              [ENV_A]
$ENV_A -m tools.qwen3_explore.qwen_split --raw out/qwen-explore/X-Q-raw.json \
    --vad out/qwen-explore/X-vad.json --out out/qwen-explore/X-QS.json
#   加 --words-carry-punct 可直接吃 whisper 的 aligned.json（对照臂用）

# 4. 评分：merge 契约口径的可接受性                                   [ENV_A]
$ENV_A -m tools.qwen3_explore.acceptability --vad out/qwen-explore/X-vad.json \
    --arm whisper=X-aligned.json --arm qwen=out/qwen-explore/X-QS.json
```

**调分句参数不需要重跑 ASR**：`X-Q-raw.json` 缓存了文本与词级时间戳，第 3–4 步纯 CPU。
只有改解码窗口大小或救援阶梯才要重跑第 2 步。

## 全量复跑（机械指标的所有数字）

第 1–2 步的产物齐了以后，整个对照与调参都在 `bench.py` 里，纯 CPU、几秒一轮：

```bash
$ENV_A -m tools.qwen3_explore.bench                 # 三臂 × 两组 的对照表
$ENV_A -m tools.qwen3_explore.bench --per-clip      # 加每个 clip 一行
$ENV_A -m tools.qwen3_explore.bench --cv 'word_pause_trust=0.25,0.5;no_gap_penalty=0.2,0.5;fragment_penalty=1,1.5,2'
$ENV_Q -m tools.qwen3_explore.start_lift --out out/qwen-explore/cannot-start.json  # 重导词表
```

三条臂：`whisper-prod`（生产原样）、`whisper-split`（whisper 词流走同一分句器）、`qwen`。
测试床（11 个 clip 及其基线路径）写在 `common.py`，基线目录可用 `QWEN_EXPLORE_BASELINE` 覆盖。

## 脚本索引

**主链路**
`dump_vad` 共享 VAD · `rescued_asr` ASR+对齐+救援 · `qwen_split` 分句 · `acceptability` 评分

**对照与调参**
- `bench` 全量对照表 + 留一交叉验证（机械口径；`--asr-boundary-bonus` / `--no-snap`）
- `gold_sweep` 按**人工金标准**扫参（`docs/segmentation-gold.md` 的标注集）。定位是
  **证伪工具不是调参器**：窗口太少，故意在评测集上拟合取乐观上界——连这样都追不平基线，
  就不必先攒更多 gold 也能否掉一个参数族
- `start_lift` 由基线 cue 导「不能起头的词」表（`lexicon.py` 的来源，需 nagisa → `ENV_Q`）

**对齐器对比**（目标：wt vs Qwen ForcedAligner，FINDINGS §4）
- `aligner_agree` 同一份文本上两个对齐器的边界一致性。按**剥标点后的字符偏移**锚定
  （两边分词不同且 nagisa 剥标点，直接比会漏掉全部带标点的 segment）。
  `--tail` 段尾专项（给 Qwen 补上 whisper 侧的能量扩展）·
  `--dump --ab` 导出盲听 A/B（共同起点、各自停在一个候选时刻）·
  `--punct-only` 只取分句器真正下刀的那类位置 · `--dump` 需 `qwen-asr` 环境

**评估与诊断**
- `vad_health` VAD 是否在漏语音：两臂「词落静音」率 + 分离后底噪电平 + 按分钟的时间线
  （`--levels`/`--timeline` 需 librosa → `ENV_Q`；这是定位 `yingtao` 根因的那把尺）
- `sample_boundaries` 按机械判定分层抽样，供人工语义裁决（判据校准的唯一手段）。
  `--arm` 选臂，分句在进程内按当前 `Params()` 重算，所以抽到的永远是当前配置的切分
- `segmentation_report` 按 `docs/segment_split.md` 的理想/可接受带打分（`--lexical-only` 只算实词条）
- `collapse_scan` 句子级坍缩扫描（`span_ratio` / 1 s 字符密度）
- `align_diag` 零时长词归因：同一音频 × 文本可信/可疑/词序打乱/完全错四条件（`--dump-dir` 可喂给 `collapse_scan`）
- `spotcheck` + `verify_whisper` 边界抽验：按各自时间戳切音频再转录，**两个验证器分两个环境跑**（避免与被测对齐器同源）
- `boundary_signals` / `punct_quality` 各边界信号的可重建率与精确率
- `recall_diff` 双向内容差 · `phonetic` 分歧的音位口径拆分 · `compare` / `compare_align` / `adjudicate` 早期 ASR 对照

**早期实验入口**（结论已并入 FINDINGS 或代码注释，脚本保留以便复现）
`run_asr`（`--window segment|full|group:N`，含 context 热词）· `run_align` · `apply_split`（走生产 `segment_split`）

## 复现要点

- 参数默认值都是 **8 折留一 CV 选出**并在 3 个未见来源上验证过的（`bench.py --cv` 可复跑），
  改之前先读 `qwen_split.py` 里各参数的注释——有几个看起来显然该加的特征实测是负收益，理由写在定义处
  或循环论证。
- `acceptability` 的判据来自 `docs/merge-calibration.md` 的代价非对称，不是我发明的尺子；
  但它是**纯机械**的，看不见语义。两轮共 96 条人工裁决给出的精确率是 58–67%，
  且「自由刀」那一档有 ~8% 漏报（原始裁决记录见 git 历史）。
  **分句质量的最终判据是语义，机械指标只是筛选器**——动了参数就该重抽一次
  `sample_boundaries`，别只看数字变好。
- **§4.6 的三臂对照表已被人工金标准反号，别单独引用它下结论**：同词流同 VAD 只换分句器，
  机械说 whisper-split 大胜生产，人工 gold 给出相反符号（FINDINGS §0 第 3 条、§3）。
  要评分句质量走 `gold_sweep.py`（按 `docs/segmentation-gold.md` 的标注集打分），
  `bench.py` 只用来看机械形态与复现历史数字。
- **词中切指标与 `fragment_penalty` 共用 `lexicon.py` 的词表，且每臂一张**（whisper 子词
  23 词 / nagisa 词素 24 词）。改词表会同时动尺子和被测对象，必须重跑 `bench --cv`。
  **词中切这一列不可跨臂比较**，理由与两次修错的经过都在 `lexicon.py` 文首。
- 产物都在 `out/qwen-explore/`（未跟踪）。
