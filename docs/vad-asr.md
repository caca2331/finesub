# vad-asr

`vad-asr` 是生成 `*-aligned.json` 的组合阶段：先运行 `vad-energy`，再运行 `asr-align`，
并把 VAD 能量聚合到最终 ASR segment。实现位于
`src/asr_playground/speech/recognition/stage.py`，薄 CLI 入口位于
`src/asr_playground/speech/recognition/cli/vad_asr.py`。

流式 VAD 检测独立在 `src/asr_playground/speech/preprocessing/vad.py`；模型的加载、
生命周期与 patched CT2 适配层在
`src/asr_playground/speech/recognition/fw_refine_backend.py`。recognition stage
只负责把两者与识别、分段及 aligned JSON 产物编排起来。

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
| `--language` | 自动检测 | Whisper 语言覆盖 |
| `--gap` | `0.3` 秒 | ASR 合批组尾静音时长（inter-interval 静音为自适应，不受此参数控制） |
| `--vad-silero-assist` | 关 | **opt-in** 两信号后置组合（energy AND silero）：(1) voicing 门控 cap——floor 压至滚动最小锚+10dB,仅在 silero voicing（右向膨胀 0.3s,不向前）处生效,解禁被 creep 压掉的响句,silero 失灵=回落原行为,只增不减;(2) ghost-drop——silero peak<0.3 且峰值≤0dB 且 ≤12s 的区间整段丢弃;(3) 无声 span carve——区间内无证据（无 voicing、<0dB）的前缀/尾部/桥接修剪切分;(4) 接缝恢复——被合并吞掉的基础检测 gap 按原边界还原,除非缝内有 ≥-5dB 的捞回内容。概率搭车在 energy 的流式 block 上算（见下「资源与失败行为」）,60min 素材约 +3s(CPU)/+1s(CUDA)。适用于分离残留噪声素材;干净素材不建议开（打包扰动白付）。标定与验收：FINDINGS 附录 V2/W/Z 及后续。统计写入 aligned metadata `vad.silero_assist`;`asr-pipeline` 同名 flag 透传 |

生产中通常由 `asr-pipeline --stage aligned`（或更下游 stage）调用 `run_vad_asr()`，随后由
[`asr-stabilize`](asr-stabilize.md) 从 aligned 生成 stable；stage 间直接调用函数，不使用 subprocess。
**ASR 固定单 worker**：2026-08-02 移除了单文件分片与 `--wt-workers` 开关，GPU profile 现在只
决定人声分离的实例数。理由与回溯点见 [`wt-parallelism.md`](wt-parallelism.md)。

`fw-refine` 是唯一 backend：正常 greedy 和 beam=5 timestamp span 都在同一次 CT2 decode 中完成
WT-compatible word refine。非单温度/多 hypothesis 等非主契约，或 compact trace 无法与最终 segments
核对时，才退回 faster-whisper teacher-force alignment；该 fallback 固定已有 tokens，不重新搜索文本。
构造模型时会校验 patched CT2 API，缺少扩展即失败，不静默切换算法。

## 数据流

```text
normalized vocal audio
  -> streamed vad-energy（语音 interval + VadEnergyTrack + pause_hints）
  -> asr-align（regroup / fallback / 覆盖率救援 / recall / 尾词能量延长；detect_disfluencies 开）
  -> 词首修正（`src/asr_playground/speech/recognition/word_starts.py`：`[*]` 块四规则
     + VAD interval / pause_hint 锚点 clamp，docs/asr-align.md「词首修正」）
  -> 幽灵重复段清理 + 重叠收回 + 零时长段延长（`src/asr_playground/speech/recognition/segments.py`，见下）
  -> 全局 DP 分句（segment_split，docs/segment_split.md；可切可并）
  -> 按最终 segment 时间范围聚合 VAD weighted energy
  -> 第二模型校验证据（`speech/verification/qwen_referee.py`，见下；--qwen-verify）
  -> *-aligned.json
```

**第二模型校验证据**（2026-08-05，`--qwen-verify {auto,on,off}`，默认 auto=装了
`qwen-asr` 就跑）：Whisper 池释放后加载 Qwen3-ASR-0.6B（bf16 峰值 ~1.5GB，所有 GPU
档位可容纳；单任务只加载一次，批量推理一次调用），对三类嫌疑段（整段收尾套话、
CJK 主导 run 里的 Latin 段、stabilize 噪声腿将标记丢弃的段）±0.1s 重认，证据写进
段级 `qwen_verify: {text, language}`；≥3s 的未覆盖 VAD 区间同批重认，听到语音的记入
`metadata.asr_align.qwen_verify.qwen_gap_recoveries`（仅证据，不插入字幕流）。
决策全部留给下游（stabilize 消费，见 docs/asr-stabilize.md）。67 clip 标定与
已知弱点（喊叫盲区）见 docs/wt-refine-handoff.md P1。
**安装**：包含在 `[asr]` 内（`transformers>=5.13,<6`，pip 增量 ~100MB；模型
Qwen3-ASR-0.6B-hf 首次运行时下载至 HF 缓存 ~1.5GB，与分离器模型同模式）。并入
`[asr]` 而非可选 extra 的理由：`--qwen-verify` 默认启用，若依赖可选则同一条命令
在不同安装上产出不同 stable，破坏再现性。模型用 `-hf` 权重经原生 transformers
推理；曾先实现过 `qwen-asr` 包 + 非-hf 权重的路线，因其 pin 死 transformers、
拖 gradio/flask/nagisa 无用负载、且需要 pyproject 无法表达的 `--no-deps` 安装而
弃用——同权重，输出逐字 parity 已验证（见 pyproject 注释）。

幽灵重复段清理（2026-08-04 新增，`drop_ghost_duplicate_segments`）在零时长延长**之前**
执行，三个条件缺一不可才删除：

1. **跨度**：整段 ≤0.1s 且归一化文本 ≥2 字（≥20 字/秒，不可能的语速）；
2. **解码证据**：段上带 `zero_duration_chunk_tail` 或 `alignment_stack` 事件——decoder
   自己报告了 chunk 尾挤压。没有这一条时，时间被量化压扁的**真实急促复读**（连喊两声
   `おい!`、歌词复唱）也会满足跨度+重复条件；全产物扫描（170 份 JSON / 5 万段）实测
   wt 时代产物里正是这类形态构成了多数命中，事件门控把它们全部挡下；
3. **重复来源**：归一化文本是 ±3s 内某个**非幽灵**段文本的子串（幽灵之间不能互证）。

不满足任一条件的微跨度段照常保留、走后续零时长延长与既有异常阶梯。删除明细
（时间+文本）写入 `metadata.asr_align.ghost_duplicate_segments_dropped` 并逐条输出
Warning，产物内可审计。验证记录见 docs/wt-refine-validation.md「已纳入生产」。

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
- `words[]` 及可选 word `confidence`；被 disfluency 块修正过起点的词带
  `disfluency_span: [块首, 块尾]` 与 `disfluency_action`
  （`merge`/`merge_short`/`delete`/`leading_*`，词级字段可穿过 DP 分句，
  见 docs/asr-align.md「词首修正」）。`[*]` 本身不进产物
- 可选 segment `confidence`、`no_speech_prob`
- 可选 `alignment_events[]`：`fw-refine` 默认收集的 path 观测与 disfluency 候选
  （`detect_disfluencies` 已默认开启）；时间已映射回原音轨，DP 分句只归属一条输出
  segment，stable 阶段原样保留，FineSub 当前不解析
- `vad_weighted_energy_db`：最终 segment 在 normalized vocal VAD 能量轨上的功率均值 dB

词首修正的动作计数写入 `metadata.asr_align.word_start_correction`
（`merge`/`delete`/`clamp_interval`/`clamp_hint` 等 → 次数）。

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
- `--vad-silero-assist` 开启时，silero 概率**搭车**在 energy 的流式 block 上算
  （`SileroProbCollector` 实现 `energy.WaveformObserver`），不再二次解码/归一化，
  也不再常驻整段波形。详见 [`vad-energy.md`](vad-energy.md#waveformobserver-钩子)。
- Whisper 默认使用 CUDA；CPU 仅为回退路径。
- ASR 音频由 `AudioBlockLoader` 以 600 秒 core + 10 秒 pad 流式读取。
- 空 VAD 输出仍生成合法的 `{"segments": [], "metadata": ...}`。
- `metadata.asr_align.timing` 保留 loading/energy/noise/VAD、Whisper load、
  alignment 和 VAD-ASR total 秒数；task report 默认只展示 ASR total。
  开 `--vad-silero-assist` 时另有 `silero_probs_sec`（概率，已含在 `vad_sec` 内，
  单列出来避免被 VAD 总时长吞掉）与 `silero_assist_sec`（VAD 之后的判据部分）。
- 每次运行结束仍打印 VAD、Whisper 和资源峰值统计。

## 验证

```powershell
python -m pytest -q \
  test/test_vad_segment_energy.py \
  test/test_vad_streaming.py \
  test/test_vad_silero_ghost.py \
  test/test_vad_carve_hints.py \
  test/test_asr_and_text_utils.py \
  test/test_pipeline_refactor.py
# silero 概率与 WaveformObserver 搭车（需加载模型）
python -m pytest -q test/test_vad_silero_probs.py --run-heavy-resource
```

