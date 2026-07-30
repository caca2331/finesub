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
2. 做 DC/高通处理、局部 RMS 轻量归一化和全局 peak limit。
3. 以 25 ms frame、10 ms hop 计算 frame dBFS 与自适应频谱加权能量。
4. 估计 120 秒局部噪声地板，通过能量 margin 和绝对 dBFS gate 累积非语音分数。
5. 合并、裁剪非语音区间，再取补集形成语音 interval。

流式路径的输出由 `test/test_vad_streaming.py` 守护，要求与整段载入路径逐位一致。

## 为何流式（RAM）

整段载入路径会把归一化副本与帧能量一起堆在内存里，VAD 阶段 RAM 随时长近似线性增长；
约 **4h+** 开始超过 Whisper 主导峰值，**~8h** 会顶穿 8GB 档预算。流式路径（600s 核 +
对称 90s context；read 边界 snap 到 1s RMS 锚点栅格）只常驻一个核块 + 整段小能量轨，
8h 量级约 **0.3GB**。全局耦合步骤（DC 均值、RMS 窗口和、峰值限幅、谱追踪、噪声地板、
打分）走与内存路径共用的确定性分块归约，故任意时长仍可 `torch.equal`。

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
python -m pytest -q test/test_vad_streaming.py test/test_vad_segment_energy.py
```

