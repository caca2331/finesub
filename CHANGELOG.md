# Changelog

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
