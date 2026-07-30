# vad-asr

`vad-asr` 是生成 `*-aligned.json` 的组合阶段：先运行 `vad-energy`，再运行 `asr-align`，
并把 VAD 能量聚合到最终 ASR segment。实现位于
`src/asr_playground/speech/recognition/stage.py`，薄 CLI 入口位于
`src/asr_playground/speech/recognition/cli/vad_asr.py`。

流式 VAD 检测独立在 `src/asr_playground/speech/preprocessing/vad.py`；WT 模型的
串行加载、延迟创建和池化生命周期独立在
`src/asr_playground/speech/runtime/model_pool.py`。recognition stage
只负责把两者与分片识别、分段及 aligned JSON 产物编排起来。

## CLI

```powershell
vad-asr out/input/input-vocal.ogg \
  --output out/input/input-aligned.json \
  --model large-v3-turbo \
  --language ja \
  --gpu-budget-gb 8
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | vocal audio |
| `--output` | `<input>-aligned.json` | aligned JSON 路径 |
| `--model` | `large-v3-turbo` | Whisper 模型 |
| `--device` | `cuda` | 无 CUDA 时告警并回退 CPU |
| `--gpu-budget-gb` | `4` | 选择 4/8/12/16 GiB 资源档并写入 metadata |
| `--wt-workers` | profile 默认 | **开发/不安全 benchmark 覆盖**；生产不应传入，可超过 profile 并改变 aligned 产物 |
| `--language` | 自动检测 | Whisper 语言覆盖 |
| `--gap` | `0.3` 秒 | ASR 合批组尾静音时长（inter-interval 静音为自适应，不受此参数控制） |

生产中通常由 `asr-pipeline --stage aligned`（或更下游 stage）调用 `run_vad_asr()`，随后由
[`asr-stabilize`](asr-stabilize.md) 从 aligned 生成 stable；stage 间直接调用函数，不使用 subprocess。
生产调用不直接指定 `--wt-workers`：单文件由 GPU profile 给出上限，batch 则在文件级并发、
每任务固定 1 个 WT worker。显式开发覆盖会向 stderr 告警；小于 1 的值直接报错。

## 数据流

```text
normalized vocal audio
  -> streamed vad-energy（语音 interval + VadEnergyTrack）
  -> asr-align（regroup / fallback / 覆盖率救援 / recall / 尾词能量延长）
  -> 重叠收回 + 零时长段延长（`src/asr_playground/speech/recognition/segments.py`，见下）
  -> 全局 DP 分句（segment_split，docs/segment_split.md；可切可并）
  -> 按最终 segment 时间范围聚合 VAD weighted energy
  -> *-aligned.json
```

零时长 segment（映射单调钳位塌缩或 whisper 自身的零时长词）会被延长
`0.01` 秒——下游消费者（`asr_playground.subtitles.rendering`、LLM 层入口）都会静默过滤 `end <= start`
的条目，不延长其文本会在所有路径中丢失。延长允许挤占后一段：被挤占段的
起点（及受影响词的起点）后延，连锁情形按时间顺序依次解决。

若 ASR/rescue 产出“segment 有文本但 `words` 为空”，分句入口会用整段文本和
segment 的 start/end 合成一条 word，并标记 `synthetic_from_segment: true`。它只用于
保证依赖 word 时间戳的后处理不丢文本，不表示真实 forced alignment；该段因此不会被内部切分。
发生数量写入 `metadata.asr_align.segment_split.synthetic_word_segments` 并输出 warning，
便于确认生产数据中是否真实出现。

### 已知缺陷：`*-aligned.json` 里的 segment 会时间重叠

`*-aligned.json` **不保证 segment 互不重叠**。11 个 clip 的测试床上共 49 处，
全部是词级重叠（段字段忠实跟随词，`段end − 末词end = +0.00`），两种形态：

- **零宽感叹词嵌在长 segment 里**（`[405.3, 405.3] ん`）——`asr_playground.subtitles.rendering` 的
  `end <= start` 过滤本来就会丢掉它们；
- **幻觉长段吞掉真台词**（`[47.5, 76.6] おぉぉぉぉぉ` 里裹着 `[54.3, 56.1]` 的真台词）——
  这类**能活到成品**：`asr_stabilize` profile 0 之后仍剩 43 处，最大重叠 27 s。

根因未定位（`extend_last_word_end_with_energy` 的 `next_word_start` 只取**同一 VAD interval
内**的下一个 ASR 段，是嫌疑之一，但只解释得了 ≤1.0 s 的那 29 处；另外 20 处最大到 27.8 s，
量级远超该函数的上限，来自别处）。`src/asr_playground/speech/recognition/transcribe.py` 是高风险核心，
未在此改动。

**不变式改由 `asr_playground.subtitles.rendering.resolve_overlaps` 兜底**：
SRT 要求 cue 有序不重叠，
渲染时截断**较早**那条 cue 的 end（而不是后移较晚那条的 start——后者会把真台词推过自己的
终点直接删掉，而元凶恰恰是左边那条幻觉长段）；两条同起点时改为后移较晚那条。
文字永不改动，只缩短显示时长，且缩短量就是重叠量。发生时向 stderr 打 `Warning:`
——不变式在这里恢复，但成因在上游，不要当成已修。

能量在所有 ASR 边界处理完成后计算，不会把一个 VAD interval 的单值复制给其下多个 aligned
segment。聚合公式为：

```text
10 * log10(
  sum(overlap_seconds * 10^(frame_db / 10))
  / sum(overlap_seconds)
)
```

## Aligned JSON

每个正常 segment 包含：

- `start` / `end` / `text` / `lang`
- `words[]` 及可选 word `confidence`
- 可选 segment `confidence`、`no_speech_prob`
- `vad_weighted_energy_db`：最终 segment 在 normalized vocal VAD 能量轨上的功率均值 dB

所有浮点输出按当前 `asr_playground.speech.recognition.transcribe.ROUND_DIGITS`
保留 3 位；例外是 `no_speech_prob`
（`ROUND_DIGITS_BY_KEY`）保留 6 位——它按 log 尺度消费，常见取值 1e-4~1e-2，3 位
会把小概率坍缩成 0.0。aligned schema 不含 VAD 置信度字段。

字段语义边界：

- segment `confidence` / `no_speech_prob` 是 Whisper 在**合批拼接时间轴**（interval +
  至多 0.7 秒保留 gap 音频 + 0.3 秒合成静音，见 asr-align 文档）上算出的，其 30 秒
  窗口是拼接产物；`no_speech_prob` 的分布与常规整轨 Whisper 用法系统性不同，
  按常规语义调阈值会失准。
- whisper 的分段会被全局 DP 重新划分（`docs/segment_split.md`）：一个 whisper 段可被切成
  多条，相邻 whisper 段也可被合成一条，**输出段与 whisper 段没有包含关系**。片段的
  `confidence` / `no_speech_prob` / `lang` 按各来源段贡献的**词数加权**继承（单来源时即
  原样继承；语义被稀释，逐片阈值判断需留意）；`vad_weighted_energy_db` 在分句后按片段
  自身边界计算，无此稀释。参数快照在 `metadata.asr_align.segment_split`。
- 每条 whisper 源段的首词带 `whisper_segment_start: true`；起点不是 whisper 边界的片段
  带段级 tag `mid_segment_start`。前者是还原原始 ASR 分段的唯一依据。
- `metadata.vad.segment_energy` 中 `"audio": "normalized_vocal"` 是生产管线的假设声明
  （pipeline 先做人声分离）；直接对未分离音频运行 `vad_asr` 时该声明不代表实际输入，
  此时能量值不适合与正常产物横向比较。

`metadata.vad.segment_energy` 记录：

```json
{
  "field": "vad_weighted_energy_db",
  "source": "adaptive_weighted_spectral_energy",
  "aggregation": "overlap_weighted_power_mean_db",
  "frame_ms": 25.0,
  "hop_ms": 10.0,
  "audio": "normalized_vocal"
}
```

若 energy mode 不是 `weighted`，或 segment 与能量轨没有有效重叠，则省略该字段，不写默认值。

## 资源与失败行为

- VAD 在 CPU 上流式执行，仅保留小型整段能量轨。
- Whisper 默认使用 CUDA；CPU 仅为回退路径。
- ASR 音频由 `AudioBlockLoader` 以 600 秒 core + 10 秒 pad 流式读取。
- 空 VAD 输出仍生成合法的 `{"segments": [], "metadata": ...}`。
- `metadata.asr_align.timing` 保留 loading/energy/noise/VAD、Whisper load、
  alignment 和 VAD-ASR total 秒数；task report 默认只展示 ASR total。
- 每次运行结束仍打印 VAD、Whisper 和资源峰值统计。

## 验证

```powershell
python -m pytest -q \
  test/test_vad_segment_energy.py \
  test/test_vad_streaming.py \
  test/test_asr_and_text_utils.py \
  test/test_pipeline_refactor.py
```

