# finesub

本项目用于在本地把长音频转成字幕。默认生产流程是：

```text
原始音频 -> 人声分离 -> VAD + ASR 对齐 JSON -> raw SRT
```

最常用入口是 `src/pipeline.py`。第一次使用时，从下面的“快速开始”走即可。

**许可**：代码与文档默认 [MIT](LICENSE)；`src/llm/prompt_templates/` 下的 prompt 明文为 [CC BY-SA 4.0](src/llm/prompt_templates/LICENSE.md)。

## LLM 纠错与翻译后处理

默认生产流程停在 raw SRT。LLM 纠错翻译需显式启用（`--stage translated-srt` 或 `final-srt`）。

默认 dry-run（只生成计划和 prompt，不调用生成 API）：

```powershell
python -m llm.correction_translation out/input/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts
```

真实调用需加 `--execute`（`.env` 中须有 Gemini API key）：

```powershell
python -m llm.correction_translation out/input/input-stable.json --audio data/input.wav -o out/input/input.srt --execute
```

注意：`--audio` 应指向原始音频，不是人声分离后的 `*-vocal.flac`。

**路线与档位**：`--route text|mm` × `--level low|med|high`（默认 `mm med`）。短音频可用 `--fast auto|on|off`（默认 auto）。`--output-scale` 调整窗口大小。用户额外信息通过 `--extra-info` / `--extra-info-file` 提供。

**知识库**（三态 `--knowledge none|collect|update`，默认 `none`）：

```powershell
python -m llm.correction_translation out/input/input-stable.json --audio data/input.wav -o out/input/input.srt --execute --knowledge update
```

知识更新也可事后独立运行：

```powershell
python -m llm.knowledge.update out/input/input.srt --execute                      # 无精修
python -m llm.knowledge.update out/input/input.srt --execute --refined-srt 精修.srt  # 有精修
```

**参考素材批量导入**（默认全执行；`--dry-run` 只打印计划）：

```powershell
python -m llm.reference_ingest --index data/reference/batch1 --gpu-budget-gb 8
python -m llm.reference_ingest --task “out/refined-ep12.srt|https://www.bilibili.com/video/BVxxxx|备注|mm-med|--language en”
```

任务是竖线分隔的一行 `srt|media|note|preset|args`。`note` 写法建议：提到关键专名时用标准写法或知识库别名（会自动匹配预注入）；贴上相关网页链接（最多 8 个，供背景调查提取）；一行背景即可。

**SRT 后处理**：最终 `*.srt` 会经过后处理（繁简转换、短轴延长、标点清理）。`--postprocess-profile -1` 不修改时间与文本，仅规范化渲染。

完整运行时行为、preset 定义、搜索代理、token 预算等见 [`docs/llm_harness_behavior.md`](docs/llm_harness_behavior.md)；知识库详见 [`docs/knowledge.md`](docs/knowledge.md)。

## 运行环境

分两段，依赖不同：

**`*-stable.json` 之前（人声分离 + VAD-ASR + ASR 稳定化）**

- 推荐：NVIDIA GPU，至少 8GB 显存。
- 推荐：至少 8GB 空余系统内存。
- CUDA 不可用时会回退 CPU，并在 stderr 输出 `Warning:`；CPU 路径会慢很多。

**LLM harness 阶段（自 `*-stable.json` 起）**

- 无需 GPU / 显存；约 **4GB** 系统内存即可。
- 需要 PATH 上的 **ffmpeg** 与 **ffprobe**（窗口音频剪辑与时长探测）。
- API key 配在 `.env`（Gemini / Exa / Tavily 等）。申请步骤与字段说明见 [`docs/manual/env.md`](docs/manual/env.md)。

依赖由 `pyproject.toml` 的 optional extras 管理，不使用 `requirements.txt`。URL 输入另需 `pip install yt-dlp`（可选，仅传 URL 时需要）。

## 安装

需要 **Python 3.12+**。

ASR 流水线（至 stable.json / raw SRT）：

```powershell
pip install -e ".[asr]"
```

仅 LLM harness（已有 `*-stable.json` 时）：

```powershell
pip install -e ".[harness]"
```

完整运行时 + 测试：

```powershell
pip install -e ".[asr,harness,dev]"
```

## 推荐目录

```text
data/   原始音频
out/    生成的 FLAC / JSON / SRT
tmp/    临时文件
```

不要把新的音频、字幕、JSON 或媒体产物放在仓库根目录。

## 快速开始

把音频放到 `data/input.wav`，然后运行（默认停在 raw SRT，不调用 LLM）：

```powershell
python src/pipeline.py data/input.wav --model large-v3-turbo --language en --gpu-budget-gb 8
```

不传 `-o` 时，输出为 `out/input/input.srt`，该输入的全部 artifact 都归到 `out/input/` 下（见下文“输出文件”）。`input` 也可以是 URL；此时 URL→id 映射缓存在 `data/reference/url-map.json`，下载/抽取的 `<video-id>.mp4` / `<video-id>.ogg` 放在本次输出 artifact 目录（默认 `out/<video-id>/`），默认输出到 `out/<video-id>/<video-id>.srt`。

继续跑 LLM 纠错翻译和最终后处理：

```powershell
python src/pipeline.py data/input.wav --stage final-srt --model large-v3-turbo --language en --gpu-budget-gb 8
```

`--stage` 可选 `vocal` / `aligned` / `stable` / `raw-srt` / `translated-srt` / `final-srt`；默认是 `raw-srt`。其中 aligned 是未经稳定化的 VAD-ASR 结果，stable 由独立 ASR 稳定化 stage 生成。默认 `--asr-stabilize-profile 0`；完整规则见 [`docs/asr-stabilize.md`](docs/asr-stabilize.md)。每个 stage 都会优先复用已有 artifact，因此可以从中途 resume。

LLM stage 的路线/档位可通过 `--llm-route` / `--llm-level` / `--llm-fast` / `--llm-output-scale` / `--llm-video` 透传（默认 mm/med/auto/1.0，同 `python -m llm.correction_translation` 的对应参数；`--llm-video` 仅 mm-high）。URL 输入跑 mm-high 且未显式传 `--llm-video` 时，会下载一份 `<video-id>.mp4` 并同时作为 pipeline 音频源和 LLM 视频源。

生成词级字幕：

```powershell
python src/pipeline.py data/input.wav --word --model large-v3-turbo --language en --gpu-budget-gb 8
```

词级字幕同样按默认 `raw-srt` stage 输出到 `out/input/input-raw.srt`。

如果不知道语言，可以省略 `--language`，让模型自动判断。

### 批量运行

多个输入用 `src/batch.py` 批量跑：三阶段流水线并行（下载 ×2 → ASR ×1 → LLM ×1），
LLM 按投喂顺序串行消费（知识累积顺序可复现），单项失败不影响其余项，重跑同一批即续跑：

```powershell
# 直接给若干 URL/路径（全局参数作为每项默认值）
python src/batch.py https://example.com/v1 data/b.wav --stage final-srt --language ja

# 或用 JSONL manifest，每行一项，字段覆盖全局默认
# {"source": "https://example.com/v1", "language": "ja", "extra_info": "备注"}
# {"source": "data/b.wav", "stage": "raw-srt"}
python src/batch.py --manifest tasks.jsonl --knowledge update
```

每项产物仍归各自 `out/<stem>/`；批次事件流写入 `out/batch/<batch-id>/batch-status.jsonl`。
`llm.reference_ingest` 的多任务批（index.csv）也跑在同一套流水线上。

## 输出文件

不传 `-o` 时，默认输出路径为 `out/<输入名>/<输入名>.srt`，一次运行的**全部 artifact 都归到 `out/<输入名>/` 一个目录**下。以输入 `data/input.wav` 跑到 `--stage final-srt` 为例，主要产物：

| 文件 | 说明 |
| --- | --- |
| `input-vocal.flac` | ① 人声分离 |
| `input-aligned.json` | ② VAD 分段 + Whisper 对齐（未稳定化） |
| `input-stable.json` | ③ ASR 稳定化结果 |
| `input-raw.srt` | ④ 由 stable.json 直转的原始 SRT |
| `input.llm-artifacts/input-research-context.json` | LLM 背景调查结果，存在即跳过研究轮 |
| `input-translated.srt` | LLM 纠错+翻译后的中文字幕（未后处理） |
| `input-corrected.srt` | 纠错后「原文」SRT（分析 ASR 误听用） |
| `input-annotated.csv` | 9 列完整标注：`type|position|duration|gap|corrected|translation|conf|char_count|note` |
| `input.srt` | 最终 SRT（对 translated 做后处理） |
| `input.llm-artifacts/` | LLM 任务 artifact 目录（`task-report.md`、`exchanges/`、resume 缓存等） |

显式传 `-o` 时，所有 artifact 名从该 SRT 路径推导，不会自动加子目录。每一步执行前会先检查默认输出是否已存在，存在即复用（只按文件是否存在判断，不校验内容/参数是否匹配这次运行）；如需强制重跑，删除该步骤及下游产物。

完整产物树、`.llm-artifacts/` 内部结构、跳过规则和纠错窗口中途 resume 的细节见 [README_DEV.md「产物清单与路径」](README_DEV.md#产物清单与路径)与「Pipeline 复用规则」。显存档位（`--gpu-budget-gb` 支持 `8`/`12`/`16`，只影响人声分离 batch）见 README_DEV.md「资源约束」。

## 文档

- [`docs/manual/env.md`](docs/manual/env.md)：`.env` / API key 申请与配置。
- [`README_DEV.md`](README_DEV.md)：开发者说明（项目地图、分步调试、资源约束、产物清单、复用规则）。
- [`docs/asr-stabilize.md`](docs/asr-stabilize.md)：ASR 稳定化 profiles 与规则。
- [`docs/llm_harness_behavior.md`](docs/llm_harness_behavior.md)：LLM harness 运行时行为。
- [`docs/knowledge.md`](docs/knowledge.md)：知识库结构与更新流程。
- [`docs/testing.md`](docs/testing.md)：测试命令与标记。
- [`examples/knowledge/`](examples/knowledge/)：知识库样板条目（主仓追踪的迷你骨架）。
