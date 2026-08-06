# Changelog

## [Unreleased]

## [0.3.0] - 2026-08-05

### 移除

- **`whisper-timestamped` backend 整个移除**，`fw-refine` 成为唯一 ASR backend。
  同时删除：`--asr-backend` 开关（pipeline 与 vad-asr）、`asr-wt` 命令、`WtModelPool`、
  `naive_approach` 退避梯子、`asr_transcribe_seed` / `whisper_timestamped_mode` metadata
  （fw-refine 从不读 seed，那条记录一直是假的），以及 `whisper-timestamped`、`dtw-python`
  两个依赖。checkpoint fingerprint 的 `asr_backend` 字段随之删除（只剩一个 backend）。
  理由：为优化要改 refine 内部逻辑，维持两套行为对齐的成本不划算；迁移验收（5 素材 /
  50.6 分钟）显示 fw-refine 快 3.19×、内容量差 ≤1%、救援活动在每个素材上都更少。
  回溯点：`dev` 的 `1fcc4e1`。
  **`asr-refine` extra 因此从可选变为必需**。

### 新增

- **`--qwen-verify {auto,on,off}`（默认 auto）：第二模型校验证据。** Whisper 池释放后加载
  Qwen3-ASR-0.6B，对三类嫌疑段（整段收尾套话、CJK 主导 run 里的 Latin 段、噪声腿将丢弃的段）
  重认，证据写进段级 `qwen_verify`，决策留给 stabilize。`[asr]` 因此新增
  `transformers>=5.13,<6`（pip 增量约 100MB，模型首次运行下载约 1.5GB 至 HF 缓存）。
  并入 `[asr]` 而非可选 extra，是为了让同一条命令在任何安装上产出相同的 stable。
- **词首修正**（`speech/recognition/word_starts.py`）：`[*]` disfluency 块按能量门决定
  融合/删除，再对首词做 VAD interval 与 pause_hint 锚点 clamp。gold 上词首 |err| 中位
  41→18ms。
- **LLM 窗口质量护栏 `max_window_subtitle_tokens`**（`ModelLimits` 默认 10,000，config.toml
  `[chunking]` 可覆盖，`0` 关闭）：单窗 `<asr_result>` 的 token 上限，独立于输出系数——
  窗口过长时翻译质量会掉，哪怕输出装得下。窗口数估算与真实 countTokens 校验两处生效，
  超限走既有的 k+1 重排；快速模式的 auto 判定也以它为第三道门（快速窗口就是全片，
  最容易撞上）。
- `energy.WaveformObserver` 钩子：让第二个信号搭车读取 VAD 已经算好的归一化 block，
  不必再解码/重采样/归一化一轮。不传时 energy 模块的每个输出值逐字节不变。
- `fw-refine` 的 multi-audio batch 设计与本机实测落入 `docs/wt-refine-port.md`：CT2 的
  `real_audio_frames` 改为逐样本、split-encode 批模式、确定性契约，以及模型 × beam × GPU profile
  的 batch size 档位表。迁移本身值 6.3×、batch 再叠 1.8×，故 batch 不阻塞 P0。
- 新增显式 opt-in 的 patched CT2 `fw-refine` checkpoint：greedy/beam=5 以 1-pass
  winner trace 对齐 WT refine，默认收集低成本 path 信号；修复后的 disfluency 保持显式
  开关，启用后同样以 `alignment_events` 透传到 aligned/stable 产物。FineSub 暂不消费。
- `fw_refine_backend.transcribe_batch()`：一次批量解码若干 ≤30s 窗口。split-encode（逐窗口
  encode、只批 decoder），再把每条结果回放进普通 `transcribe()`，因此 segment/词/事件的组装
  只有一份实现。24 个真实生产窗口上文本一致 22/24（18 条连词级时间逐位相同），加速 1.73×。
  调用方的组批策略尚未实现。

### 变更

- **VAD 归一化的峰值限幅由全局缩放改为逐样本 clamp——会改变 VAD 输出。** 旧实现是
  `x * (0.98 / 全局峰值)`，让**单个最响样本**决定施加到整轨的位移：kaguya60 的 0.00029%
  样本把 60 分钟压低 2.18 dB，mia 的 0.00004%（约 25 个样本）把 108 分钟压低 4.12 dB；
  7 条真实分离人声里 4 条触发。而 VAD 判据里有一批绝对 dBFS 阈值，它们能跨文件通用正是
  靠上一步把局部 RMS 对到 −24 dBFS，全局缩放随后又按文件拆掉一部分标定。改为削掉越界样本
  后，weighted 轨上受影响帧从「全部有信号帧」降到 179/4292，越阈帧从约 2000/20000 降到
  0/1。**限幅曾触发的文件重跑会得到不同 segment**（kaguya60 752→753、语音 +4.8s；
  mia 2561→2522、+46.1s），未触发的文件逐字节不变；旧产物重跑即可。同时也删掉了流式
  路径中专为施加全局标量而存在的第二遍 pass。详见 `docs/vad-energy.md`。
- **ASR 固定单 worker，移除单文件分片设计**。`sharding.py`、`ResourceProfile.wt_instances`、
  `--wt-workers`（pipeline 与 vad-asr 两处）、interval ownership 标记与合并、shard partial
  一并删除；GPU profile 现在只决定人声分离实例数，metadata 不再记录 asr workers。
  实测 worker=3 相对 worker=1 在 wt 上仅 1.40×、fw-refine 上仅 1.20×，代价是显存 2.4→6.5 GB；
  换到 fw-refine 后 ASR 已非瓶颈（人声分离占语音段 72%）。回溯点：`dev` 的 `1fcc4e1`。
- **ASR 分组改为按合成后的音频长度规划**：组尾垫料（至多 0.7 秒原始音频 + `--gap` 秒静音）
  此前不计入分组长度，于是按 30 秒规划的组实际可达 31 秒、溢出编码窗口。11 个真实 clip 上
  超窗分组 102 → 56，总组数 388 → 405。**这会改变分组边界，因而改变 ASR 输出**（11 个 clip
  中 9 个分组不同）；旧产物不会自动失效，需要重跑才能得到新分组。`combined_group_duration()`
  语义不变（auto language 短组启发式仍按「说了多少话」判断）。
- `asr-pipeline` 新增开发用 `--asr-backend {wt,fw-refine}`（默认 `wt`，与 `vad-asr` 一致）。
  面向一般用户的文档不介绍该开关。
- ASR checkpoint fingerprint 新增 `asr_backend` 且**无默认值**：两个 backend 对同一段音频给出
  不同词级时间，中断后换 backend 续跑此前会静默复用另一侧的 partial，把两种输出缝进同一份产物。
- `faster-whisper` 与 `ctranslate2` 移出 `asr`，独立为 **`asr-refine`** extra 并精确钉版
  （1.2.1 / 4.8.1，均为当前最新且互相兼容）。fw-refine 不在 `asr-pipeline` 的可达路径上，
  普通用户不必装一份链了 CUDA 的 CTranslate2；同时也消除了 `asr` 与 desktop runtime lock
  之间「lock 缺这两个包」的静默不一致。fw-refine 继承 faster-whisper 内部实现并读取 CT2
  解码轨迹，小版本变动可能悄然改变输出；升级顺序固定为先 faster-whisper 后 CT2——CT2 的
  可选范围由 fw 声明的 `>=4.0,<5` 决定。

### 性能

- **`--vad-silero-assist` 从「比 VAD 本身还贵 4 倍」降到几乎免费**：60 分钟分离人声上
  该 opt-in 此前在 17 秒的 VAD 阶段之上再花 72 秒，现在 CPU 约 3 秒、CUDA 约 1 秒。
  silero 的逐帧 JIT 调用（每 32ms 一次，一小时 112,500 次派发）改为：帧间独立的
  STFT+encoder 整批计算，LSTMCell 的权重驱动全序列 `nn.LSTM`；概率再搭车在 VAD 自己的
  流式 block 上算，省掉第二遍解码与整段波形常驻（0.23 GB/小时）。与逐帧实现的最大概率
  差 1.4e-05，判据阈值零翻转。CUDA 路径显式关闭 TF32（开启会让 112,500 帧中 16 帧越过
  `CAP_SIL_THR`）。

### 修复

- `fw-refine` 在 CTranslate2 缺少该设备的矩阵后端时，给出指向构建要求的可读错误，而不是从库
  深处抛出的 `No SGEMM backend on CPU`。该情况无法从 `get_supported_compute_types()` 查出
  （只有 CUDA 后端的构建对 CPU 仍报 `float32`），只能在首次 encode 处补上下文。
- `fw-refine` 的词切分不再丢弃永远无法解析的 `U+FFFD` token。真实幻觉撞上解码上限时会在半个
  字符处截断，旧实现一直等待补全、循环结束时静默丢弃该 token，使词分组与 one-pass 解码轨迹错位并
  抛 `ValueError`；310 个真实生产窗口里有 6 次触发。同时补上 fw-refine 的退避链——任何失败改退到
  该后端自己的 teacher-force 对齐（`naive_approach` 是 whisper-timestamped 的选项，对 fw-refine 会
  原样重放同一次调用），两次都失败才丢弃该 group，不再终止整个 run。
- recall 临时组的 complement 切片现在继承源 interval 的 shard 归属。此前它新建裸 dict，
  分片合并时以「missing interval ownership」中止**整个 run**；触发条件是 workers ≥ 2 且
  某个 block 命中 recall（≥5s 未覆盖），与 ASR backend 无关。
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
