# MLX refine 后端

在 Darwin/arm64 上，`--asr-backend auto` 选择 `mlx-refine`；非 Apple 平台的代码路由仍选择
`fw-refine`，但本次生产验收只覆盖 Windows/CUDA，Linux 尚未验证。显式指定的后端始终优先。逻辑默认模型只在 MLX 后端下映射为
`mlx-community/whisper-large-v3-turbo`。CTranslate2 模型仓库会被明确拒绝，不会被误当作
MLX checkpoint。

## 可复现契约

受支持的版本组合不可拆分：

- `mlx-whisper==0.4.3`
- `mlx==0.32.2`
- 模型 revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`
- FineSub MLX trace contract version 1
- `src/finesub_bootstrap/pylock.macos-arm64-py312.toml`

模型 manifest 固定并校验 `weights.safetensors`、`config.json` 和仓库 README 的哈希。启动时同时
检查软件包版本和 adapter 使用的 decoder 内部 hook；不受支持的组合会在模型加载前立即停止。
升级时必须整组更新以上五项，并重新运行 trace fixtures、完整测试和 Apple Silicon parity 验收。

## 解码与降级行为

在零温度、单窗口解码中，FineSub 记录每个选中 token、对应 log probability，以及选定
alignment head 的 cross-QK 行；同时保留终止 EOT（或达到解码上限的 unfinished tail）和真实音频
帧边界。随后把 trace 转换为 patched CTranslate2 使用的同一套 `TimestampSpan`/`AlignedSpan`
契约，复用分词、confidence、DTW、边界修复和 `alignment_events` 逻辑。

如果 segment/token 对账或 alignment 失败，仅该窗口改用 mlx-whisper 的 teacher-force word
timestamp。metadata 会记录 one-pass 和 teacher-force 窗口数、命中率和降级原因。由于
mlx-whisper 0.4.3 没有 beam search，低覆盖率救援会记录跳过 beam，然后直接进入现有的
peel/split 救援阶梯。

## 上游源码与许可证

`mlx_refine_backend.py` 中的窄 adapter 面向 MIT 许可证的 `mlx-whisper` 0.4.3 实现。它适配
`mlx_whisper/decoding.py`、`mlx_whisper/transcribe.py` 和 `mlx_whisper/timing.py` 的对象契约；
不跟踪上游 `main`，也没有复制完整 seek loop。上游版权声明保留在 adapter 模块中。

## Apple Silicon 验收契约

发版 parity 运行必须记录文本差异、逐词边界误差、语音/词覆盖率、异常次数、耗时、峰值 RAM 和
MLX 内存。在 decoded tokens 相同时，至少 95% 的词边界须位于 patched CT2 基准 ±20 ms 内，
全部位于 ±60 ms 内；每个 fixture 的词数和语音覆盖率差异都须在 1% 内。所有 fallback 或 rescue
都必须出现在 metadata 中。

使用以下命令生成机器可读的比较结果：

```console
python -m tools.wt_refine_validation.compare_backends ct2-aligned.json mlx-aligned.json -o mlx-parity.json
```

## 2026-09-07 真实视频冒烟测试

输入为 `/Users/<user>/sample-video.mp4`（29.513 秒，H.264 1288×720 + AAC 双声道）。本次运行
刻意隔离新 ASR 后端，关闭人声分离、Silero 辅助、Qwen 校验、语言重解、知识检索和 grounded
web search。可复现命令如下：

```console
PYTHONPATH=src python -m finesub.pipeline \
  /Users/<user>/sample-video.mp4 \
  --no-separate --no-vad-silero-assist \
  --asr-backend auto --qwen-verify off --lang-redecode off \
  --stage final-srt --test-profile --knowledge none --llm-retrieval none \
  --log-level verbose \
  -o /tmp/finesub-sample-video.srt
```

`auto` 选择了 `mlx-refine` 和固定 revision 的 large-v3-turbo。同进程指标补跑识别出 2 个语音
区间，生成 8 segments / 85 words，音频覆盖率 98.69%，总计 11.231 秒。模型加载 0.948 秒，
ASR alignment 9.197 秒。一个解码窗口使用 one-pass trace；另一个报告
`span-count-mismatch` 并正确进入 teacher-force，因此该 fixture 的 one-pass 命中率为 50%。
MLX 峰值内存为 2,180,607,202 bytes；pipeline 资源采样器报告进程峰值 1.24 GiB。
`/usr/bin/time -l` 独立记录 maximum RSS 1,326,120,960 bytes、macOS peak memory footprint
3,044,170,496 bytes。这为 fallback 路径提供了正向覆盖，但该 mismatch 仍须作为 trace adapter
parity fixture 继续调查，不能算作 one-pass 通过。

Gemini free pool 在 checkout `.env` 中配置为
`GEMINI_FREE={"main":"<AI Studio key>"}`。该文件被 Git 忽略且权限设为 `0600`。macOS 无法使用
项目仅限 Windows 的 DPAPI 保护，因此当前实现以明文保存该值；不得提交、复制或附带此文件。
非持久运行应只在启动进程的环境变量中传入同一变量。test profile 的纠错调用认证成功
（`HTTP 200`，5.228 秒，无重试）；复用 ASR 产物后，pipeline 在 9.662 秒内完成
`final-srt`。

运行中确认并修正了三个配置问题：

1. `--log-level debug` 无效。详细日志应使用 `--log-level verbose`，另外两个可选值为
   `quiet` 和 `normal`。
2. `--no-separate` 不会关闭默认 Silero 后处理。在没有 `silero_vad` 的隔离环境中还须传
   `--no-vad-silero-assist`；正常的完整 `[asr]` 安装可保持该后处理开启。
3. 默认/local retrieval 使可选 Gemma grounded-search endpoint 反复返回 HTTP 500。当验收目标是
   Gemini 纠错翻译而不是 web search 可用性时，应使用 `--llm-retrieval none`。这不表示 Gemini
   key 无效：两次 generation 调用都返回 HTTP 200。

本次产物有意放在仓库外：`/tmp/finesub-sample-video-aligned.json`、
`/tmp/finesub-sample-video-raw.srt` 和 `/tmp/finesub-sample-video.srt`。

## 2026-09-07 FasterWhisper CPU 降级检查

同一输入在没有 Metal/CUDA 的情况下，使用精确固定的 stock CPU 依赖
`faster-whisper==1.2.1` 和 `ctranslate2==4.8.1` 重跑。这两个包不属于 Darwin/arm64 的
MLX-only 依赖选择；需要可选紧急路径的机器须在 Python 3.12 环境中额外安装：

```console
python -m pip install "faster-whisper==1.2.1" "ctranslate2==4.8.1"
```

运行时须同时明确指定后端和资源档位：

```console
PYTHONPATH=src python -m finesub.pipeline \
  /Users/<user>/sample-video.mp4 \
  --no-separate --no-vad-silero-assist \
  --asr-backend fw-refine --device cpu --gpu-tier cpu \
  --qwen-verify off --lang-redecode off --stage raw-srt \
  --log-level verbose \
  -o /tmp/finesub-fwcpu481-sample-video.srt
```

已发布的 patched CTranslate2 wheel 仅支持 Windows/CUDA。在 macOS 上，`fw-refine` pool 会按预期
警告并退回 stock FasterWhisper teacher-force word timestamps，全程 GPU 内存为零。同进程指标中，
模型加载 4.662 秒、ASR alignment 18.707 秒；MLX 分别为 0.948 秒和 9.197 秒。CPU alignment
慢 2.034×；完整 VAD/ASR 阶段为 24.671 秒，MLX 为 11.231 秒，慢 2.197×。

decoded text、segment 数、word 数和有序词序列完全相同（8 segments / 85 words）。170 个配对
起止边界中，96.47% 位于 20 ms 内，平均绝对误差 4.882 ms。最大误差是第 7 段 CPU
word/segment start 提前 240 ms。word-duration coverage 为 24.215 秒，MLX 为 23.905 秒，
差异 1.297%。第 1–6 段边界相同；第 7 段相对 MLX 为 −240/+10 ms，第 8 段为 −20/0 ms。

这证明 CPU-only 降级路径可用，但不代表完整 refine parity。stock FasterWhisper 保留了全部 85 个
word confidence，却不产生 `alignment_events`；当前 generic adapter 因上游 segment 不提供聚合值，
把 segment confidence 写为零。metadata 在 stock fallback 后仍保留请求的 `fw-refine` 后端名，
当前运行时警告是 patched 实现未生效的证据。要求 FineSub refine events 或非零 segment confidence
的消费者不得把此路径视为 patched CT2 或 MLX 的等效实现。

机器可读的耗时比较为 `/tmp/finesub-mlx-vs-fwcpu481-parity.json`，CPU aligned 产物为
`/tmp/finesub-fwcpu481-sample-video-aligned.json`。

### 相同方法的完整比较

下表两行都来自新进程，使用相同的缓存输入/模型文件，并用 `/usr/bin/time -l` 包裹完整
`raw-srt` 命令。这补齐了首次冒烟记录缺少的进程内存、confidence、event 和 phase 字段。

| 指标 | MLX refine | FasterWhisper CPU | CPU 相对 MLX |
| --- | ---: | ---: | ---: |
| 系统 wall time | 13.22 s | 26.57 s | 2.010× |
| 模型加载 | 0.948 s | 4.662 s | 4.918× |
| ASR alignment | 9.197 s | 18.707 s | 2.034× |
| VAD/ASR 阶段总计 | 11.231 s | 24.671 s | 2.197× |
| maximum RSS | 1,326,120,960 B | 4,121,214,976 B | 3.108× |
| macOS peak memory footprint | 3,044,170,496 B | 7,640,075,584 B | 2.510× |
| 引擎报告的 MLX 峰值 | 2,180,607,202 B | n/a | — |
| pipeline 采样进程峰值 | 1.24 GiB | 3.84 GiB | 3.097× |
| decoded segments / words | 8 / 85 | 8 / 85 | 相同 |
| 非零 segment confidence | 8 / 8 | 0 / 8 | 降级 |
| segment confidence 均值 | 0.7290 | 0.0000 | 降级 |
| 非零 word confidence | 85 / 85 | 85 / 85 | 覆盖相同 |
| word confidence 均值 | 0.8690 | 0.8363 | −0.0327 |
| alignment events | 2 | 0 | 降级 |
| one-pass / teacher-force 窗口 | 1 / 1 | 0 / 1 stock path | 降级 |

文本与有序词序列完全相同。170 个配对词边界的平均绝对误差为 4.882 ms，96.47% 位于 20 ms 内，
最大误差 240 ms。word-duration coverage：MLX 23.905 秒，CPU 24.215 秒（+1.297%）。因此补跑
通过文本相同、词数相同、95% 位于 20 ms 内以及结构异常检查，但仍未通过全部位于 60 ms 内和覆盖率
差异不超过 1% 两项门槛。完整补跑比较结果为
`/tmp/finesub-mlx-vs-fwcpu481-rerun-parity.json`。

## 2026-09-08 macOS 标准流程

上面的隔离冒烟测试刻意关闭了多项生产阶段。第二次测试对同一段 29.513 秒输入启用标准人声分离、
Silero 辅助、自动 MLX 路由、异常结果救援、Qwen 校验、稳定化和最终 Gemini 纠错：

```console
FINESUB_CHECKOUT_DATA=0 PYTHONPATH=src python -m finesub.pipeline \
  /Users/<user>/sample-video.mp4 \
  --stage final-srt --asr-backend auto \
  --knowledge none --llm-retrieval none --log-level verbose \
  -o /tmp/finesub-macos-standard.srt
```

关闭 knowledge 和 retrieval 只是为了避免本次兼容测试读写无关的本地知识，以及依赖可选的
grounded-search endpoint；不会改变人声分离、VAD、ASR、稳定化或字幕时间轴。

第一次完整尝试发现的两个设备路由缺陷现已由测试守护：

1. `audio-separator` 已能检测 Apple Silicon，但 FineSub 过去会在 CUDA 不可用时把它覆盖为 CPU。
   现在 Darwin/arm64 保留 MPS/CoreML；显式 `--device cpu` 与 `--gpu-tier cpu` 仍然优先。
2. 解析后的 MLX 设备字符串曾进入 Silero，形成 `torch.device("mlx")`。现在 MLX ASR 会把
   PyTorch 辅助模型映射至 MPS（MPS 不可用时为 CPU），Silero 也不再套用 CTranslate2 专属的
   MPS 拒绝策略。

修复后的全新运行记录如下：

| 指标 | 结果 |
| --- | ---: |
| 输入时长 | 29.513 s |
| 人声分离 | 77.006 s，MPS + CoreML |
| Silero probability pass | 1.458 s，MPS |
| VAD 语音覆盖 | 27.180 s / 29.513 s（92.1%），8 intervals |
| ASR 后端 | `mlx-refine`，固定 revision |
| MLX one-pass / teacher-force 窗口 | 4 / 1（one-pass 80%） |
| fallback 原因 | `span-count-mismatch` ×1 |
| ASR 阶段 | 30.060 s |
| Qwen 校验 | 5.228 s，恢复 1 个 gap |
| 最终 LLM 阶段 | 26.031 s |
| 完整流程 | 133.251 s |
| 最终字幕 | 8 个有序 cue，无空段或倒序时间轴 |

fallback 已记录在 aligned metadata 中，没有导致中止或静默丢段。Gemini 一次临时 HTTP 503 被自动
重试，下一次请求成功。重复完整流程得到完全相同的 cue 时间轴；最终阶段是生成式模型，译文措辞按预期
存在变化。仓库文档没有保存本地用户名、API key、输入原标题或主机名。
