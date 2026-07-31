# Changelog

## [Unreleased]

### 修复

- Desktop 的 API Key 设置改用现行 CLI provider pool 变量，Windows AI runtime
  lock 重新对齐 Torch/Torchaudio 2.8；新增跨 Desktop/CLI 契约测试，避免两侧再次漂移。
- Desktop 完成页只展示实际存在的产物并暴露 run metadata；WebView 重载会恢复仍在
  运行的任务，翻译模式明确提示 Gemini 媒体片段上传。
- Desktop 默认 LLM level 和后处理 profile 范围与生产 pipeline 对齐；新增独立
  `desktop/VERSION` 作为桌面发布版本的单一来源。

## [0.2.0] - 2026-07-30

### 新增

- ASR 新增单文件 Whisper Timestamped 分片、分组 checkpoint、运行时 metadata、
  GPU stage gate、stall watchdog 与资源用量记录；4/8/12/16GB profile 会在文件内部
  分配 separator 和 WT 并发。
- 字幕分句改为全局 DP，并新增分割点金标准、标注工具和系统化评测资料。
- LLM harness 新增可配置 API key pool、sticky retry 后的组合冷却，以及更完整的
  token budget、任务报告和搜索证据处理。
- 新增 `config.example.toml`、桌面 launcher 资源配置和 Windows token counter 更新。

### 变更

- 生产代码重组到 `asr_playground` namespace，明确 media、speech、subtitles 和
  workflows 边界；命令行入口和打包清单同步迁移。
- Batch 从文件级 ASR 并发改为单文件独占 profile、文件内分片并发，避免两层并发相乘。
- 桌面应用同步外观控制与 UI 刷新，并在构建 bootstrap 时优先使用 conda env-root Python。
- OpenCC 转简加载器与本地 token counter 改为跨调用复用；ASR 模型从共享 checkpoint
  直接构建 FP16 实例，降低重复加载开销。

### 修复

- 修复 wheel 漏装 `batch`、`gpu_stage_gate`、`run_metadata`、`segment_split` 和
  `wt_shard` 顶层模块，以及 license metadata 无法在声明的 setuptools 下限构建的问题；
  增加顶层源码与 packaging 清单一致性测试。
- ASR checkpoint schema 升至 v2；旧 partial 明确失效并从头重跑，sharded merge
  遇到缺失 interval ownership 的结果会显式报错，不再静默丢字幕。
- `segment_split` 对有文本但没有 word timestamps 的 segment 合成一条带来源标记的
  segment-span word；无法安全归一化时保留原输入，不再从全局 DP 输出中消失。
- Reference ingest 在迁移到统一 batch workflow 前，先与普通 batch 一样固定每任务
  `wt_workers=1`，避免文件级并发与 shard 并发相乘。
- Pipeline 的 LLM round 汇总和 task report 现在遵循显式 `task_artifact_dir`；batch
  同一 logical run 的后续 pass 会继承已执行 stage，stage metadata 不再混出
  `reused` 加旧 `elapsed_sec` 的矛盾记录。
- 字幕渲染在时间轴后处理前修复 cue 重叠，保证不丢文字。
- 修复 separator dotted 临时文件识别、Gemini key 全部跳过时的错误类型，以及
  flash gap/end pad、RPM 失败计数等重试与边界问题。

### 变更

- `--wt-workers` 明确为开发/不安全 benchmark 覆盖参数；生产调度继续由 GPU profile
  和 batch runner 决定。
- 补齐 run metadata、WT sharding、segment split 和 packaging 测试的 pytest 域 marker。

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
