# vad-energy

`vad-energy` 是面向人声分离音频的 CPU 能量 VAD。源码与 CLI 入口均在
`src/asr_playground/speech/preprocessing/energy.py`，安装项目后使用 `vad-energy`
命令。

## 用途与边界

- 根据归一化后的人声音频估计非语音区间，再取补集得到供 ASR 使用的语音区间。
- 生产流水线不通过 subprocess 调用 CLI；`vad-asr` 直接调用本模块的 Python API。
- 该模块不执行 Whisper，也不生成带文字的字幕。

## CLI

```powershell
vad-energy out/input/input-vocal.ogg \
  -o out/input/input-vad_energy.srt \
  --energy-mode weighted
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 输入音频，生产路径通常为 vocal OGG（.flac 为 legacy fallback） |
| `-o`, `--output` | `<input>-vad_energy.srt` | 调试 SRT 输出 |
| `-e`, `--energy-mode` | `weighted` | `weighted` 为自适应频谱加权；`none` 使用 frame dBFS |
| `--snr` | 内置阈值 | 覆盖非语音判定的 SNR margin |

当前 CLI 将检测到的非语音区间反转为语音区间，再以文本 `""` 写入 SRT；该 SRT
主要用于检查边界，不是最终字幕。Python API 中
`detect_non_speech_intervals_file()` 返回非语音区间，而 `run_vad_file()` 返回语音区间，
两者语义不同。

## 处理流程

1. 流式读取、转单声道并重采样到 16 kHz。
2. 做 DC/高通处理、局部 RMS 轻量归一化（120s 窗 / 1s anchor 线性插值，增益钳在
   −4…+6 dB），最后把样本 clamp 到 ±`NORM_PEAK_LIMIT`(0.98)。
3. 以 25 ms frame、10 ms hop 计算 frame dBFS 与自适应频谱加权能量。
4. 估计 120 秒局部噪声地板，通过能量 margin 和绝对 dBFS gate 累积非语音分数。
   语音样帧的扣分**先记账**：连续 `SPEECH_EXIT_MIN_RUN_FRAMES`(4) 帧才入账,
   更短的毛刺被赦免——无词区抖动连跑中位 2–5 帧,真词 17–24 帧（FINDINGS Z）。
   记账中的帧不推进区间 `end`。
5. 合并、裁剪非语音区间。被负 padding 取消的 ≥40ms 原始 gap 位置记入
   `pause_hints`（gap 右缘前 40ms,写进 vad metadata）,供未来分句当信号。
6. 语音区间的**绝对峰值底线**（`MIN_SPEECH_PEAK_DB = -45`）：峰值加权能量从未达到
   -45 dB 的语音区间并回非语音。相对判据信任 noise floor，而 floor 塌进数字静音时
   -70 dB 的底噪扰动会带着 20+ dB 的"伪 SNR"开出区间——那种响度不可能是语音。
   标定与词守卫见 `tools/vad_tuning/FINDINGS.md` 附录 X。
7. -45 判据的 **partial apply**（`_carve_low_peak_speech`）：区间内 sub--45 的
   头部前缀/尾部/内部桥接段被修剪或切开（保留 0.14s/0.04s 接缝余量）,
   处理"噪声起头接正常语音"与"扰动桥接两句"的情形。
8. 再取补集形成语音 interval。

流式路径的输出由 `test/test_vad_streaming.py` 守护，要求与整段载入路径逐位一致。

## 为何流式（RAM）

整段载入路径会把归一化副本与帧能量一起堆在内存里，VAD 阶段 RAM 随时长近似线性增长；
约 **4h+** 开始超过 Whisper 主导峰值，**~8h** 会顶穿 8GB 档预算。流式路径（600s 核 +
对称 90s context；read 边界 snap 到 1s RMS 锚点栅格）只常驻一个核块 + 整段小能量轨，
8h 量级约 **0.3GB**。全局耦合步骤（DC 均值、RMS 窗口和、谱追踪、噪声地板、打分）走与
内存路径共用的确定性分块归约，故任意时长仍可 `torch.equal`。峰值 clamp 是逐样本的，
天然分块无关。

### 峰值 clamp 为什么不是全局缩放（2026-08-05 变更）

原实现是 `x * (0.98 / 全局峰值)`。它的问题是**由单个最响样本决定整轨位移**：
kaguya60 的 0.00029% 样本把 60 分钟降了 2.18 dB，mia 的 0.00004%（约 25 个样本）
把 108 分钟降了 4.12 dB；7 条真实分离人声里 4 条触发。而 VAD 判据里有一批**绝对
dBFS 阈值**（`ABS_NON_SPEECH_MAX_DBFS_ENTER/EXIT`、`MIN_SPEECH_PEAK_DB`，以及
silero assist 的 `CARVE_CEILING_DB`/`SEAM_LOUD_KEEP_DB`），它们能跨文件通用正是
靠上一步把局部 RMS 对到 −24 dBFS——全局缩放随后又把这个标定按文件拆掉一部分。

改成逐样本 clamp 后（以两文件实测，weighted 轨越阈帧数）：

| | 全局缩放 | clamp |
| --- | --- | --- |
| 受影响帧 | 全部有信号帧（11.7 万 / 54.6 万） | 179 / 4292 |
| 越 −30/−28/−45/0 dB 阈值 | 1960/1873/1607/2261 · 20116/20467/16565/39559 | 0/0/0/0 · 1/2/0/1 |

同时试过"局部限幅 + 线性过渡"（±10/30/100 ms ramp）：**每项指标都不如 clamp，且
ramp 越宽越差**——把峰值压下去所需的 −4 dB 会盖在整个 ramp 上，连累窗内正常语音。
且 `x·min(1, 0.98/|x|) ≡ clamp(x, ±0.98)`，clamp 就是该族在 ramp→0 的退化成员，
所以这是单调族的最优端点，不是二选一。

**输出影响**：限幅触发的文件上 segment 会变（kaguya60 752→753 个、语音 +4.8s；
mia 2561→2522 个、+46.1s），不触发的文件逐字节不变。旧产物重跑即可。
另外 clamp 与"完全去掉 clamp"在实测两文件上 interval 与 `pause_hints` **全等**，
保留它只是以零成本守住 |x| ≤ 0.98 这个不变量。

## WaveformObserver 钩子

归一化后的波形本来就在流式 pass 里生成了一遍，`observer=` 让第二个信号搭车读到它，
不必再解码/重采样/归一化一轮。`_streamed_frame_tracks` / `detect_non_speech_intervals_file` /
`run_vad_file` / `vad.detect_segments` 都透传该参数；**不传时本模块的每个输出值逐字节不变**
（`test_vad_silero_probs.py::test_no_observer_leaves_the_tracks_untouched`）。

协议（`energy.WaveformObserver`）：

- `feed(block)` 按顺序收到每个 core block，不重不漏，因此所有 block 拼接 == 整段归一化波形；
  block 拿到的是**归一化全部做完之后**（含 clamp）的切片；
- `reset()` 在首块之前调用一次。实现上仍应做成"清空重来"而非假定只调一次；
- block 是整秒的 16 kHz 音频，`STREAM_CORE_SEC * TARGET_SR` 能被 512 整除，所以 hop 能整除
  它的观察者天然帧对齐。

唯一的生产使用者是 opt-in 的 silero assist（`silero_ghost.SileroProbCollector`）。

## Python API

- `run_vad_file(...) -> (speech_items, metadata, duration_sec, energy_track)`：语音
  interval 加帧能量轨，一次分析同时产出；`vad-asr` 用它聚合最终 segment 能量。
- `detect_non_speech_intervals_file(...) -> (non_speech, duration_sec, energy_track)`：
  非语音区间版本，同样附带能量轨。
- `VadEnergyTrack`：保存 `energy_db` 与 `hop_sec` / `frame_sec` / `energy_mode`。帧时间
  按 `i * hop_sec` 均匀栅格在聚合时用 float64 现算——不存 float32 帧时间张量，
  避免约 2.3 小时以后帧时间被量化（毫秒级）导致短 segment 边界帧选择漂移。
- `aggregate_segment_weighted_energy_db(...)`：仅对 `weighted` track 生效。将逐帧 dB
  转回线性功率，按 frame 与 segment 的重叠时长加权平均，再转回 dB。

该能量基于 normalized vocal 音频和自适应频谱权重，不是原音频 dBFS，也不是置信度。
不同文件只有在 VAD 参数和预处理一致时才适合横向比较。

## 验证

```powershell
python -m pytest -q test/test_vad_streaming.py test/test_vad_segment_energy.py   test/test_intervals.py test/test_vad_low_peak_absorb.py test/test_vad_carve_hints.py
# WaveformObserver 与 silero 概率（需加载模型）
python -m pytest -q test/test_vad_silero_probs.py --run-heavy-resource
```

