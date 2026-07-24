# vad-asr

`vad-asr` 是生成 `*-aligned.json` 的组合阶段：先运行 `vad-energy`，再运行 `asr-align`，
并把 VAD 能量聚合到最终 ASR segment。源码与 CLI 入口位于 `src/vad_asr.py`。

## CLI

```powershell
python src/vad_asr.py out/input/input-vocal.ogg \
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
| `--gpu-budget-gb` | `8` | 选择 8/12/16 GiB 资源档并写入 metadata |
| `--language` | 自动检测 | Whisper 语言覆盖 |
| `--gap` | `0.3` 秒 | ASR 合批组尾静音时长（inter-interval 静音为自适应，不受此参数控制） |

生产中通常由 `pipeline.py --stage aligned`（或更下游 stage）调用 `run_vad_asr()`，随后由
[`asr-stabilize`](asr-stabilize.md) 从 aligned 生成 stable；stage 间直接调用函数，不使用 subprocess。

## 数据流

```text
normalized vocal audio
  -> streamed vad-energy（语音 interval + VadEnergyTrack）
  -> asr-align（regroup / fallback / 覆盖率救援 / recall / 尾词能量延长）
  -> 重叠收回 + 零时长段延长（见下）
  -> 超长段 DP 切分（segment_split，docs/segment_split.md）
  -> 按最终 segment 时间范围聚合 VAD weighted energy
  -> *-aligned.json
```

零时长 segment（映射单调钳位塌缩或 whisper 自身的零时长词）会被延长
`0.01` 秒——下游消费者（`to_srt`、LLM 层入口）都会静默过滤 `end <= start`
的条目，不延长其文本会在所有路径中丢失。延长允许挤占后一段：被挤占段的
起点（及受影响词的起点）后延，连锁情形按时间顺序依次解决。

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

所有浮点输出按当前 `asr_align.ROUND_DIGITS` 保留 3 位；例外是 `no_speech_prob`
（`ROUND_DIGITS_BY_KEY`）保留 6 位——它按 log 尺度消费，常见取值 1e-4~1e-2，3 位
会把小概率坍缩成 0.0。aligned schema 不含 VAD 置信度字段。

字段语义边界：

- segment `confidence` / `no_speech_prob` 是 Whisper 在**合批拼接时间轴**（interval +
  至多 0.7 秒保留 gap 音频 + 0.3 秒合成静音，见 asr-align 文档）上算出的，其 30 秒
  窗口是拼接产物；`no_speech_prob` 的分布与常规整轨 Whisper 用法系统性不同，
  按常规语义调阈值会失准。
- 超长 whisper 段会被 DP 切分（`docs/segment_split.md`）为多个 segment：切分
  片段的 `confidence` / `no_speech_prob` / `lang` **继承来源段整体值**（语义
  被稀释，逐片阈值判断需留意）；`vad_weighted_energy_db` 在切分后按片段
  自身边界计算，无此稀释。切分参数快照在 `metadata.asr_align.segment_split`。
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
- 每次运行结束打印 VAD、Whisper 和资源峰值统计。

## 验证

```powershell
python -m pytest -q \
  test/test_vad_segment_energy.py \
  test/test_vad_streaming.py \
  test/test_asr_and_text_utils.py \
  test/test_pipeline_refactor.py
```

