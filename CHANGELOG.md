# Changelog

## [Unreleased]

## [0.1.1] - 2026-07-27

Prompt version: `zh-subtitle-correction-csv-v65`。

### 新增

- 新增 Windows Desktop 应用，提供任务管理、资源管理、运行时设置与日志查看等桌面工作流。
- Pipeline 新增 `--name`；视频任务默认使用高质量多模态模型，并在不可用时自动降级。

### 变更

- Prompt 升级到 `zh-subtitle-correction-csv-v65`：删除 CapableA、BasicC 及 BasicC 的 JSONL 输出支线，现行变体为 capableB/C + basicA/B。
- 单窗口 query、correction 与 fast round 1 的目标字幕序号每窗重置为 `1..N`，只读前文按时间顺序编号为 `1-M..0`；harness 校验后映射回稳定源序号，oneshot、replay、benchmark 与任务反馈同步采用该契约。
- 改进搜索证据包和研究阶段，减少冗余上下文并提高可用证据密度。
- 完善 ASR 语言历史、分组 checkpoint 与 Whisper fallback，扩展救援阶梯和尾段回交策略。
- ASR 依赖更新至 PyTorch 2.8 系列。

### 修复

- Pipeline 与最终 SRT 输出改为原子写入，避免中断时留下不完整文件。
- 修复 clip 预取线程安全问题，并补充 Pipeline 失败诊断信息。

## [0.1.0] - 2026-07-23

首个 beta 版本。Prompt version: `zh-subtitle-correction-csv-v63`。

### 功能

- 本地长音频转字幕完整流水线：人声分离 → VAD + Whisper ASR 对齐 → ASR 稳定化 → SRT 输出
- 实验性 LLM 纠错与翻译后处理（Gemini），支持 6 档 preset（route × level）、fast 模式、多轮搜索调查
- 批量运行（三阶段流水线并行，单项失败隔离，断点续跑）
- 本地知识库（自动采集/统一更新/精修对照）
- URL 输入支持（yt-dlp 下载 + 自动 ID 映射）
- 词级字幕输出
- SRT 后处理（繁简转换、短轴延长、标点清理）
- GPU 显存档位（8/12/16GB）自适应
- 流式 VAD 与流式 ASR 对齐（内存上界恒定，支持任意时长音频）
- LLM session resume + 纠错窗口中途 resume
- Prompt 变体系统（capableA/B/C + basicA/B/C）

### 依赖

- Python >= 3.12
- 核心流水线：torch~=2.9.0, torchaudio~=2.9.0, numpy, soundfile, whisper-timestamped, onnxruntime, audio-separator, numba
- LLM harness：httpx, yt-dlp, opencc-python-reimplemented
- 开发：pytest, pytest-xdist, sudachipy, sudachidict_core

### 已知限制

- LLM 纠错翻译层为实验性功能，默认不包含在生产 stage 中（需 `--stage translated-srt` 或 `final-srt` 显式启用）
- 仅提供 Windows 预编译 token counter 二进制；Linux/macOS 回退到免费 countTokens API 或启发式
- 无 GPU 时回退 CPU（速度显著下降）
