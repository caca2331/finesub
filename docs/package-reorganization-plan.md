# 第 10 步：生产代码 package 与模块边界重整计划

> ⚠️ 本文是**迁移计划记录**，不是现状描述。其中的 `speech/recognition/sharding.py` 与
> `speech/runtime/model_pool.py` 已于 2026-08-02 随单文件分片与 whisper-timestamped backend
> 一并删除；现状以 `CLAUDE.md` 的 Architecture map 为准。

状态：进行中（Phase 3 部分完成，见 §7.0）；计划正文仍保留迁移前基线描述。已根据 `step10-package-reorganization-plan-review.md` 修订
范围：生产代码的 speech、LLM、media、subtitles、workflow 与 packaging；不重构
desktop，但同步更新其受 package 迁移影响的 import/源码探测
性质：结构性重构；除修复由迁移直接暴露的 packaging/import 问题外，不改变算法、模型参数、输出格式或运行语义

## 1. 背景

当前生产代码同时存在以下结构问题：

- `src/` 下有大量顶层 `py-modules`，依靠 `pyproject.toml` 手工列举，新增模块容易漏装。
- 声学/ASR 代码平铺在顶层，`asr_align.py` 已承担识别、时间映射、恢复、checkpoint、sharding 等多项职责。
- `acoustic` 不能准确概括 VAD、ASR、字幕稳定化和分段，因此决定使用更常见的 umbrella term：`speech`。
- `alignment` 不能准确概括当前 `asr_align.py`。该文件主体是在 VAD window 上执行 ASR recognition，并将结果映射回原时间轴，不是通常意义上的 forced alignment。
- `src/llm` 当前有 48 个受版本控制的 Python 文件、约 2.31 万物理行，其中 37 个文件平铺在 `llm` 顶层。
- `llm` 目录内混入了通用媒体下载、FFmpeg、SRT 数据模型等非 LLM 专属能力。
- `reference_ingest.py` 同时编排媒体、speech、LLM、knowledge 和 batch，属于应用 workflow，而不是 LLM library。
- 多个大文件已经混合独立职责：
  - `llm/web_search.py`：约 1,816 行。
  - `llm/stages/correction_loop.py`：约 1,791 行。
  - `llm/knowledge/base.py`：约 1,298 行。
  - `llm/csv_utils.py`：约 1,222 行。
  - `llm/research.py`：约 1,107 行。
  - `llm/search_loop.py`：约 1,075 行。
- 当前内部依赖有边界泄漏，例如 prompt artifact 代码导入 correction loop 私有函数，provider client 反向 re-export exchange metadata。

本计划的目标不是简单增加目录，而是建立稳定的领域边界和单向依赖，并将 setuptools 从手工模块清单迁移到标准 package discovery。

## 2. 已确定的命名决策

### 2.1 顶层 Python package

采用：

```text
src/asr_playground/
```

Python distribution 名仍为 `finesub`；import package 名使用 `asr_playground`。

理由：

- 避免继续向 Python 顶层 namespace 注入 `pipeline`、`batch`、`asr_align` 等宽泛模块名。
- 使用 package discovery 后，新增内部模块不再要求维护 `py-modules` 白名单。
- `prompt_templates`、model catalog 等 package data 可以与代码一起定位。
- 为 speech、LLM、media、subtitles、workflows 提供明确的共同根。

### 2.2 语音处理总目录

采用：

```text
asr_playground/speech/
```

不采用：

- `acoustic`：通常更偏声学特征、声学模型和波形层能力，不能自然覆盖字幕分段与稳定化。
- `audio`：范围太宽，会混入音乐、媒体 I/O、编码等非 speech 能力。
- `transcription`：适合识别服务名称，不适合作为 VAD、分离、运行时和后处理的总目录。

### 2.3 ASR 主体目录

采用：

```text
speech/recognition/
```

当前不建立 `speech/alignment/`。

只有未来出现以下独立能力时才建立 alignment package：

- 已知 transcript 与 audio 的 forced alignment。
- 独立的词级或音素级 aligner。
- 不依赖 ASR decode 的时间轴对齐服务。

当前 `asr_align.py` 中少量时间映射逻辑使用 `timestamp_mapping.py` 表达，不将整个识别引擎命名为 alignment。

### 2.4 LLM 总目录

保留：

```text
asr_playground/llm/
```

`llm` 对 provider 调用、prompt、correction、research、retrieval 和 knowledge update 是准确且常见的名称，无需改成 `ai` 或其他宽泛术语。

## 3. 目标目录

计划完成后的目标结构：

```text
src/
  asr_playground/
    __init__.py
    pipeline.py
    batch.py
    paths.py
    text.py

    workflows/
      __init__.py
      reference_ingest.py

    run/
      __init__.py
      context.py
      stage_events.py
      report.py

    media/
      __init__.py
      source.py
      ffmpeg.py
      clips.py
      audio.py

    subtitles/
      __init__.py
      model.py
      rendering.py
      alignment.py
      metrics.py
      postprocess.py

    speech/
      __init__.py

      preprocessing/
        __init__.py
        audio.py
        separation.py
        vad.py
        energy.py

      recognition/
        __init__.py
        transcribe.py
        stage.py
        decoder.py
        windows.py
        timestamp_mapping.py
        segments.py
        recovery.py
        checkpoint.py
        sharding.py
        cli/
          __init__.py
          align.py
          vad_asr.py
          wt.py

      postprocessing/
        __init__.py
        stabilization.py
        segmentation.py

      runtime/
        __init__.py
        resources.py
        gpu_stage_gate.py
        model_pool.py
        resource_usage.py

    llm/
      __init__.py
      config.py
      profiles.py
      output_tags.py
      content_filter.py

      providers/
        __init__.py
        client.py
        gemini.py
        catalog.py
        model_catalog.psv
        models.py
        rate_limit.py

      prompting/
        __init__.py
        loader.py
        context.py
        correction.py
        research.py
        knowledge.py
        variants.py
        artifacts.py
        templates/
          *.md
          LICENSE.md

      tokens/
        __init__.py
        counter.py
        budget.py
        truncate.py
        injection.py
        measure.py

      correction/
        __init__.py
        service.py
        planning.py
        chunking.py
        window_loop.py
        fast_session.py
        output_protocol.py
        contracts.py
        clip_prefetch.py

      research/
        __init__.py
        session.py
        search_loop.py

      retrieval/
        __init__.py
        client.py
        models.py
        render.py
        providers/
          __init__.py
          exa.py
          gemma.py
          tavily.py
          duckduckgo.py

      knowledge/
        __init__.py
        schema.py
        storage.py
        repository.py
        artifacts.py
        entries.py
        feedback.py
        materials.py
        mistakes.py
        update.py

      artifacts/
        __init__.py
        exchange_log.py
        exchange_metadata.py
        task_report.py
        session_checkpoint.py
```

这是最终边界设计，不要求在一个提交中一次拆到该粒度。迁移分成“建立 package”“机械移动”“大文件拆分”三类阶段，任何阶段都必须保持可运行。

## 4. 边界与依赖规则

### 4.1 顶层依赖方向

允许的主要依赖方向：

```text
pipeline / batch / workflows
        ↓
speech + llm
        ↓
media + subtitles + run
```

具体规则：

- `media` 不得导入 `speech`、`llm`、`pipeline` 或 `workflows`。
- `subtitles` 不得导入 `speech`、`llm`、`pipeline` 或 `workflows`。
- `speech` 可以导入 `media`、`subtitles`、`text` 和通用 `run`，不得导入 `llm`。
- `llm` 可以导入 `media`、`subtitles`、`text` 和通用 `run`，不得导入
  `speech`；GPU profile 等调度信息由 workflow/pipeline 注入，不以
  `llm → speech.runtime` 的反向依赖取得。
- `pipeline`、`batch` 和 `workflows` 负责组合 speech 与 LLM。
- `reference_ingest` 不得重新实现普通 batch 已有的 download、ASR、LLM correction stage。
- 通用运行 metadata 放在 `asr_playground/run`；LLM exchange 和 research session artifact 放在 `llm/artifacts`。
- extras 分层是硬约束：`asr_playground`、`media`、`subtitles`、`run`、`llm`
  及这些 package 的 `__init__.py` 不得在模块顶层导入 `torch`、`torchaudio`、
  `numpy`、`numba`、`audio_separator` 或 `whisper_timestamped`。ASR 重依赖只允许
  出现在 `speech` 内；确需跨层的可选能力使用窄接口和函数内延迟导入。

### 4.2 LLM 内部依赖方向

目标依赖方向：

```text
correction ─┐
research ───┼→ providers
knowledge ──┘

correction → research → retrieval

correction / research / knowledge
        ↓
prompting + tokens + artifacts
        ↓
config / profiles / output_tags / content_filter
```

`tokens` 保持 provider-neutral：本地/启发式 counter 与协议留在 `tokens`，
Gemini `countTokens` callable/API key resolver 由 provider 层在组装时注入，不允许
`tokens` 导入 provider 私有函数。`config.py` 中的 `ModelLimits`、
`RateLimitPolicy`、`ModelEndpoint`、`RoleModelConfig` 等纯 dataclass 继续作为
最底层类型/常量；`providers/models.py` 只存 provider 专属 endpoint-chain/tier 定义。

约束：

- `providers` 不得反向导入 correction、research、knowledge 或 artifacts。
- `prompting` 不得导入 workflow orchestration。
- `retrieval` 只负责调用搜索供应商、标准化结果和渲染，不决定研究轮次。
- `research` 决定搜索目标、轮次和证据完整性，不包含具体 Exa/Tavily/DDG HTTP 实现。
- `knowledge` 的 repository/storage 层不得导入 provider client；只有 `knowledge/update.py` 可以调用 LLM。
- 禁止通过 `__init__.py` 大量 re-export 来伪装旧路径。
- 禁止一个 package 导入另一个 package 的下划线私有符号。
- `llm/__init__.py` 清空现有 config re-export；调用方从真实定义模块导入。

### 4.3 对外稳定边界

本项目不要求维护旧的内部 Python import 路径，但本结构重构默认保持：

- 现有 console script 名称。
- CLI 参数及默认值。
- 输入输出文件格式。
- artifact 路径语义；已确认要修复的 custom artifact directory 漏洞除外。
- ASR、LLM、knowledge 的运行顺序。
- profile、模型和 decoding 参数。
- 测试 fixture 格式。

checkpoint 兼容性按单独决定处理：旧 schema 明确失效并从头重跑，不在本结构计划中实现 migration。

## 5. 当前文件迁移映射

### 5.1 顶层、workflow 与 run

| 当前文件 | 第一阶段目标 | 后续目标/说明 |
|---|---|---|
| `src/pipeline.py` | `asr_playground/pipeline.py` | 保持跨 speech/LLM 编排 |
| `src/batch.py` | `asr_playground/batch.py` | 保持通用三 bin scheduler |
| `src/run_metadata.py` | `asr_playground/run/report.py` | 与 RunContext/StageRecorder 改造协调 |
| `src/llm/reference_ingest.py` | `asr_playground/workflows/reference_ingest.py` | 重构为普通 batch pipeline + refined subtitle post task |
| repo-root 路径解析 | `asr_playground/paths.py` | 集中解析 `.env`、`config.toml`、`.state`、tokcount 与 knowledge root |
| `src/utils/text.py` | `asr_playground/text.py` | 纯 stdlib、跨 speech/subtitles 使用的文本工具 |

reference ingest 的 post task 必须与对应任务的 LLM correction 保持同一串行调度单元：

```text
correction(i)
→ refined knowledge update(i)
→ correction(i+1)
```

不得因为增加独立 post worker 而改变知识累积顺序。

### 5.2 media 与 subtitles

| 当前文件 | 目标文件 | 说明 |
|---|---|---|
| `src/llm/media_source.py` | `asr_playground/media/source.py` | 下载、URL map、媒体选择 |
| `src/llm/ffmpeg_clips.py` | `asr_playground/media/ffmpeg.py` | FFmpeg/ffprobe 命令 |
| `src/llm/audio_clips.py` | `asr_playground/media/clips.py` | clip 范围与抽取 |
| `src/utils/audio.py` | `asr_playground/speech/preprocessing/audio.py` | torch/torchaudio/numpy 重依赖，不进入 harness 公共层 |
| 从 `utils.audio` 提取轻量 duration probe | `asr_playground/media/audio.py` | 只用 stdlib/ffprobe；供 harness 使用，不加载 torch |
| `src/utils/others.py` | `asr_playground/speech/runtime/resource_usage.py` | GPU/RAM 运行统计；移除 `utils` 聚合 re-export |
| `src/llm/srt_utils.py` | `asr_playground/subtitles/model.py` | SrtSegment、parse、validate |
| `src/to_srt.py` | `asr_playground/subtitles/rendering.py` | JSON/SRT rendering 与 CLI |
| `src/llm/srt_alignment.py` | `asr_playground/subtitles/alignment.py` | 只保留 SRT-to-SRT 配对和完整 diff；token 截断移到 LLM 调用方 |
| `src/subtitle_metrics.py`、`src/llm/subtitle_metrics.py` | `asr_playground/subtitles/metrics.py` | 去除重复定义/re-export |
| `src/llm/srt_postprocess.py` | `asr_playground/subtitles/postprocess.py` | 繁简、标点、时长后处理 |

`llm/clip_prefetch.py` 不整体移入 media：prefetch scheduler 属于 correction，底层 clip 操作调用 `media.clips`。

`subtitles.alignment.render_alignment_diff` 不再依赖 `llm.token_truncate`：
公共层返回完整文本，knowledge/prompt 注入侧在原来的调用边界执行同等 token 截断。

### 5.3 speech

| 当前文件/职责 | 第一阶段目标 | 拆分后目标 |
|---|---|---|
| `vocal_separation.py` | `speech/preprocessing/separation.py` | 第三方 adapter 也收敛于此 |
| `vad_energy.py` | `speech/preprocessing/energy.py` | 保持纯能量工具与 CLI |
| `vad_asr.py` 中 VAD 检测 | `speech/preprocessing/vad.py` | 与识别编排分离 |
| `vad_asr.py` 主编排 | `speech/recognition/stage.py` | VAD + model pool + aligned JSON stage |
| `WtModelPool` | `speech/runtime/model_pool.py` | 模型生命周期与并发 |
| `resource_profiles.py` | `speech/runtime/resources.py` | GPU/worker profile |
| `gpu_stage_gate.py` | `speech/runtime/gpu_stage_gate.py` | GPU family gate |
| `asr_wt.py` | `speech/recognition/cli/wt.py` | 直接 Whisper CLI wrapper |
| `asr_stabilize.py` | `speech/postprocessing/stabilization.py` | 稳定化 |
| `segment_split.py` | `speech/postprocessing/segmentation.py` | 字幕 segment DP |
| `wt_shard.py` | `speech/recognition/sharding.py` | 与 `asr_align.py` 内 shard 执行逻辑合并，不是纯移动 |
| `asr_align.py` | `speech/recognition/transcribe.py` | 首先整体移动，之后按下表拆分 |

`asr_align.py` 后续拆分：

| 当前职责 | 目标文件 |
|---|---|
| VAD interval 归一化、group/window 构造、combined audio | `recognition/windows.py` |
| Whisper transcribe kwargs、candidate decode、naive fallback | `recognition/decoder.py` |
| combined time 映射回原 interval、word spacing annotation | `recognition/timestamp_mapping.py` |
| coverage shortfall、异常 interval isolate、recall/rescue | `recognition/recovery.py` |
| checkpoint identity、load/write/clear | `recognition/checkpoint.py` |
| shard tag、partial path、merge、stale sweep | `recognition/sharding.py` |
| `align_segments`/`align_segments_sharded` 对外 service | `recognition/transcribe.py` |
| overlap clamp、zero-length 修复、drop empty | `recognition/segments.py`；属于识别输出时间轴自洽，不进入 profile stabilization |

拆分时不建立泛化的 `alignment/core.py`。

三个现有 CLI 明确分开：`cli/align.py` 保留 `asr-align` 参数表，
`cli/vad_asr.py` 保留 `vad-asr` 参数表，`cli/wt.py` 保留 `asr-wt` 参数表；
它们调用 `transcribe.py`/`stage.py` 的 service，不共享一个 `main()`。

### 5.4 LLM providers 与 tokens

| 当前文件/职责 | 目标文件 |
|---|---|
| `llm/client.py` 中 provider-neutral role client | `llm/providers/client.py` |
| `llm/llm_runtime.py` 中 Gemini REST、key retry | `llm/providers/gemini.py` |
| `llm/model_catalog.py`、PSV | `llm/providers/catalog.py`、相邻 PSV |
| `llm/rate_limit.py` | `llm/providers/rate_limit.py` |
| `ModelLimits`、`RateLimitPolicy`、`ModelEndpoint`、`RoleModelConfig` | 留在 `llm/config.py` |
| provider endpoint-chain/tier 专属定义 | `llm/providers/models.py` |
| token counter protocol、Gemini/local/fallback counter | `llm/tokens/counter.py` |
| CorrectionBudget 与预算验证 | `llm/tokens/budget.py` |
| `llm/token_truncate.py` | `llm/tokens/truncate.py` |
| `llm/injection_budget.py` | `llm/tokens/injection.py` |
| `llm/token_measure.py` | `llm/tokens/measure.py` |

`client.py` 当前对 exchange metadata 的尾部 re-export 应移除；调用方直接导入 `llm.artifacts.exchange_metadata`。

### 5.5 LLM prompting

| 当前文件/职责 | 目标文件 |
|---|---|
| prompt template 加载和通用 fragment | `llm/prompting/loader.py` |
| `PROMPT_VERSION`、`PROMPT_TEMPLATE_DIR`、template loading | `llm/prompting/loader.py`；继续承担 resume cache 失效契约 |
| `ContextPack` | `llm/prompting/context.py` |
| correction prompt builders | `llm/prompting/correction.py` |
| research/search prompt builders | `llm/prompting/research.py` |
| knowledge update prompt builders | `llm/prompting/knowledge.py` |
| `llm/prompt_variants.py` | `llm/prompting/variants.py` |
| `llm/prompt_artifacts.py` | `llm/prompting/artifacts.py` |
| `llm/prompt_templates/` | `llm/prompting/templates/` |

需要消除 `prompt_artifacts.py` 对 correction loop 私有 `_window_audio_label` 的导入。该函数改为 correction 内公开 rendering/model helper，再由 prompting 依赖稳定接口。

`prompt_compose.py` 拆成 `loader.py`（版本、resource 定位、template loader）和
`correction.py`（correction/query/fast composers）；`prompts.py` 中的 `ContextPack`
进入 `context.py`，其余 builders 按 correction/research/knowledge 归属移动。

### 5.6 LLM correction

| 当前文件 | 第一阶段目标 |
|---|---|
| `llm/correction_translation.py` orchestration | `llm/correction/service.py` |
| `llm/stages/plan.py` | `llm/correction/planning.py` |
| `llm/chunking.py` | `llm/correction/chunking.py` |
| `llm/stages/correction_loop.py` | `llm/correction/window_loop.py` |
| `llm/stages/fast_session.py` | `llm/correction/fast_session.py` |
| `llm/csv_utils.py` | `llm/correction/output_protocol.py` |
| `llm/session_contract.py` | `llm/correction/contracts.py` |
| `llm/clip_prefetch.py` | `llm/correction/clip_prefetch.py` |

`stages` package在完成迁移后删除。它当前只包含 correction-specific stage，名称过于宽泛。

`correction_translation.py` 的 CLI 解析可以：

- 暂时保留在 `correction/service.py`；或
- 在最终阶段提取薄 CLI module。

不得让 CLI 参数处理重新渗透进 correction algorithm。

第二轮拆分候选：

```text
window_loop.py
query_round.py
validation.py
retry.py
rendering.py
```

拆分依据是职责与依赖，不按固定行数切割。

### 5.7 LLM research 与 retrieval

| 当前文件/职责 | 第一阶段目标 | 后续目标 |
|---|---|---|
| `llm/research.py` | `llm/research/session.py` | research session 与 parsing |
| `llm/search_loop.py` | `llm/research/search_loop.py` | contract/progress/evidence loop |
| `llm/web_search.py` | `llm/retrieval/client.py` | 再拆 models/render/providers |

职责边界：

- research 决定为何搜索、何时继续、何时完成。
- retrieval 执行 search/extract、管理 provider fallback、标准化结果。
- provider-specific HTTP 和解析逻辑进入 `retrieval/providers/`。

`web_search.py` 的后续拆分顺序：

1. 移出 request/result dataclass 到 `retrieval/models.py`。
2. 移出纯 rendering/metadata helper 到 `retrieval/render.py`。
3. 分别提取 Exa、Gemma、Tavily、DuckDuckGo provider。
4. `retrieval/client.py` 只保留 pool/fallback orchestration 和稳定公开接口。

### 5.8 LLM knowledge 与 artifacts

现有 `knowledge/` 边界保留。`knowledge/base.py` 后续拆分：

| 职责 | 目标文件 |
|---|---|
| category、operation、dataclass、验证 | `knowledge/schema.py` |
| Markdown 路径、读写、index | `knowledge/storage.py` |
| embedded git 操作 | `knowledge/repository.py` |
| task artifact ledger | `knowledge/artifacts.py` |

已有 `entries.py`、`feedback.py`、`materials.py`、`mistakes.py`、`update.py` 保持其业务名称。

LLM 执行 artifact：

| 当前文件 | 目标文件 |
|---|---|
| `llm/exchange_log.py` | `llm/artifacts/exchange_log.py` |
| `llm/exchange_metadata.py` | `llm/artifacts/exchange_metadata.py` |
| `llm/task_report.py` | `llm/artifacts/task_report.py` |
| `llm/session_checkpoint.py` | `llm/artifacts/session_checkpoint.py` |

通用 pipeline timing/status 不放入 `llm/artifacts`，而由 `asr_playground/run` 负责。

## 6. Packaging 迁移

### 6.1 setuptools

删除手工维护的：

```toml
py-modules = [...]
packages = [...]
```

改为 package discovery，示意：

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["asr_playground*"]
```

package data 必须显式覆盖：

- `asr_playground.llm.prompting/templates/*.md`
- `asr_playground.llm.prompting/templates/LICENSE.md`
- `asr_playground.llm.providers/model_catalog.psv`

是否使用 `include-package-data` 或 `[tool.setuptools.package-data]` 由实现者选择，但 isolated wheel test 必须验证文件实际存在并可读取。

`src/tools/gemini-token-counter/` 是 Go 源码，不是 Python package；package discovery
必须确认不会把它装入 wheel。本步骤不顺带移动该工具或已构建的 `bin/`。

### 6.2 console scripts

保留现有命令名，更新 import target：

| 命令 | 目标 |
|---|---|
| `asr-pipeline` | `asr_playground.pipeline:main` |
| `asr-align` | `asr_playground.speech.recognition.cli.align:main` |
| `asr-stabilize` | `asr_playground.speech.postprocessing.stabilization:main` |
| `asr-wt` | `asr_playground.speech.recognition.cli.wt:main` |
| `vad-asr` | `asr_playground.speech.recognition.cli.vad_asr:main` |
| `vad-energy` | `asr_playground.speech.preprocessing.energy:main` |
| `vocal-separation` | `asr_playground.speech.preprocessing.separation:main` |
| `to-srt` | `asr_playground.subtitles.rendering:main` |
| `llm-correct-translate` | `asr_playground.llm.correction.service:main` |
| `llm-knowledge-update` | `asr_playground.llm.knowledge.update:main` |
| `llm-reference-ingest` | `asr_playground.workflows.reference_ingest:main` |
| `llm-token-compare` | `asr_playground.llm.tokens.measure:main` |

`asr-align` 与 `vad-asr` 的输入和参数表不同，固定使用两个薄 wrapper，不再保留
“共用一个 `main()`”的可选项。`batch` 当前没有 console script，这是有意保持现有
公开入口；迁移后仓库内调用改为 `python -m asr_playground.batch`，不在结构重构中
擅自新增 `asr-batch` 命令。

### 6.3 运行时路径契约

禁止继续用脆弱的 `Path(__file__).parents[N]` 推断安装后的仓库根来寻找 package resource。

prompt template 和 model catalog 使用：

```python
importlib.resources.files(...)
```

新增 `asr_playground/paths.py` 作为唯一 repo/runtime path resolver。优先级固定为：

```text
函数显式参数
→ 对应 FINESUB_* 环境变量
→ 源码 checkout 标记发现
→ 该资源定义的安全 fallback / 明确报错
```

迁移点：

| 当前位置 | 资源 | 新契约 |
|---|---|---|
| `llm_runtime.py` | `.env` | `FINESUB_ENV_FILE` 或 checkout root；不存在只使用 process env |
| `llm/api_keys.py` | `config.toml` | `FINESUB_CONFIG_FILE` 或 checkout root；不存在使用 provider/pool 默认值 |
| `rate_limit.py`、`web_search.py` | `.state/` | `FINESUB_STATE_DIR` 或 checkout root；wheel 无 root 时使用明确的用户 state 目录 |
| `token_budget.py` | `bin/.../tokcount.exe` | `GEMINI_TOKEN_COUNTER_EXE` 优先，其次 checkout root；不存在保持现有 counter fallback |
| `knowledge/base.py` | knowledge root | 显式参数/`FINESUB_KNOWLEDGE_ROOT`/checkout root；无法确定时在实际启用 knowledge 时明确报错，禁止 import-time 报错或静默新建 |
| package templates/catalog | wheel package data | 只用 `importlib.resources`，不经过 project-root resolver |

`src/` 内除 `paths.py` 外禁止出现用于 repo root 定位的 `parents[N]`。artifact root
仍由 `RunContext` 从输出路径或显式参数解析，不与 project root 混为一谈。
`.state` 的 wheel fallback 使用稳定的用户 state 目录
（Windows `LOCALAPPDATA`、Unix `XDG_STATE_HOME`、最后才是用户目录下的
`.finesub/state`），不新增 `platformdirs` 依赖。`DEFAULT_KNOWLEDGE_ROOT` 改为惰性
resolver/调用参数 `None`，避免模块 import 时绑定错误 checkout。

wheel 支持面明确为：import、全部 CLI 参数解析、无 API/无模型轻量路径，以及在调用者
显式提供所需 runtime 路径/环境变量时的对应功能。源码 checkout 仍是当前完整生产运行的
基准模式；本步骤不把 `.env`、`config.toml`、`.state`、knowledge 或 `bin/` 打进 wheel。

## 7. 分阶段实施

### 7.0 执行进度（2026-07-29）

| 范围 | 状态 | Commit |
|---|---|---|
| 计划进入 `docs/` | 完成 | `3a02019` |
| `asr_playground` namespace / package discovery | 完成 | `70d87ec` |
| runtime path resolver | 完成 | `a7a71f3` |
| media / subtitles / workflows 公共层 | 完成 | `9e95030` |
| speech 目录与 CLI 迁移，移除旧 `utils` package | 完成 | `17e43b6` |
| VAD preprocessing 与 WT model pool | 完成 | `c882cc1` |
| recognition segment timeline normalization | 完成 | `006f9a9` |
| checkpoint 与 sharding execution | 完成 | `e23b79f` |
| windows / timestamp mapping / decoder / recovery | 待实施 | — |
| LLM package、reference ingest service 化、最终清理 | 待实施 | — |

上述“完成”均表示对应批次已通过默认完整测试；不表示整个 Phase 3 或第 10 步已经完成。
`transcribe.py` 仍保留 service、window、decoder、timestamp mapping 和 recovery，后续按
下面的 Phase 3 顺序逐批拆分。

### Phase 0：建立基线

动作：

- 记录当前 `dev` commit：`b5eea87`（若执行前 tip 变化则重新记录）。
- 确认工作区干净。
- 运行非 heavy 完整测试。
- 构建当前 wheel。此前漏装的 `batch`、`gpu_stage_gate`、`run_metadata`、
  `segment_split`、`wt_shard` 已在 `b5eea87` 修复并有 packaging 回归测试，
  因此 Phase 1 的基线是健康 wheel，不再把漏装列为已知失败。
- 保存主要 CLI `--help` 输出。
- 对典型 fixture 保存结构性输出摘要，而不是重新生成模型结果。
- 保存迁移清单：当前 56 个 test 文件、23 处字符串 monkeypatch target、
  25 个导入生产模块的 tools 文件，以及 14 个含当前路径的主文档
  （含 README/README_DEV/CLAUDE）+ CHANGELOG。实现时重新扫描，不把这些数量当永久常量。

基线命令至少包括：

```text
python -m compileall -q src test
python -m pytest -q
```

验收：

- 基线结果和已知 warning 被记录（当前：`811 passed, 1 skipped`；wheel/CLI smoke 通过）。
- `test/test_packaging.py` 在迁移到 package discovery 时同步改写，而不是删除保护。

### Phase 1：建立 `asr_playground` package 与 packaging

动作：

- 创建 package root。
- 先整体移动 `src/*.py` 顶层模块，尽量不拆文件；`src/llm` 在 Phase 4 前、
  `src/utils` 在 Phase 2/3 完成前可由 transitional package discovery 暂时收录，
  但最终 discovery 只允许 `asr_playground*`。
- 更新所有生产和测试 import。
- 同步更新 desktop 对旧布局的 4 处硬引用：
  `backend/worker/main.py` import、`backend/runtime/environment.py` source path、
  `backend/launcher/main.py` 的两处源码探测；不重构 desktop 内部结构。
- 更新 console scripts。
- 改用 package discovery。
- 配置 prompt templates 与 model catalog package data。
- 使用 `importlib.resources` 读取 package resource。
- 建立 `paths.py` 并迁移 §6.3 的 6 类 runtime path。
- 在开始跨域移动前建立第一版 AST import-boundary 测试；后续 Phase 逐步加规则。
- 盘点并迁移测试、desktop、tools 中的直接 import、字符串 monkeypatch 目标和
  `sys.path.insert(..., "src")` 引用。

约束：

- 不在本阶段改变 ASR/LLM 算法。
- 不保留大批旧模块 wrapper。
- 若需要短期 wrapper 维持单个 CLI，必须有删除阶段和测试。

验收：

- 源码运行测试通过。
- isolated wheel 中能导入所有生产模块。
- 所有 console script 的 `--help` 可执行。
- prompt templates、LICENSE 和 PSV 能从安装后的 wheel 读取。
- `src` 顶层不再依赖手工 `py-modules`。
- `desktop/backend/tests/test_launcher_paths.py`、
  `test_runtime_environment.py`、`desktop/scripts/tests/test_package_bootstrap.py` 通过。
- `src/` 内除 `paths.py` 外无 repo-root `parents[N]` 推断。

### Phase 2：移出公共 media、subtitles 与 workflow

动作：

- 机械移动 media/subtitle 模块。
- 将 torch 音频工具放入 speech，公共 `media.audio` 只保留轻量 duration probe。
- 将 SRT alignment 的 token truncation 提升到 LLM 调用方，公共 subtitles 层只返回完整 diff。
- 移动 reference ingest 到 workflows。
- 更新 import，不改变行为。
- 删除 `llm` 对通用媒体/SRT ownership。

验收：

- `media` 和 `subtitles` 无上层反向依赖。
- batch/pipeline 不再导入 `llm.media_source`。
- reference ingest 仍保持现有调度顺序。
- SRT parse/render、clip、download 相关测试全部通过。
- harness-only import 不加载 ASR 重依赖。
- session_replay 测试与 correction dry-run 通过。

### Phase 3：整理 speech package

动作：

- 先将现有文件整体映射到 preprocessing、recognition、postprocessing、runtime。
- 将 `vad_asr.py` 的 VAD、model pool、recognition orchestration 拆到对应层。
- 将 `asr_align.py` 按 windows/decoder/timestamp/recovery/checkpoint/sharding 拆分。
- 保持 `align_segments`/transcribe service 的稳定公共入口。

建议提交顺序：

1. runtime 与 preprocessing。
2. postprocessing。
3. sharding/checkpoint。
4. timestamp mapping/windows。
5. decoder/recovery。
6. 最终删除原 `asr_align.py`/`vad_asr.py`。

验收：

- recognition 不导入 LLM。
- 单 shard 与多 shard 输出在固定 fixture 上结构一致。
- checkpoint 新 schema 行为保持单独决定的“旧格式失效、重新运行”。
- profile 与 worker 默认值不改变。
- ASR/VAD scoped suite 和完整测试通过。

### Phase 4：机械整理 LLM package

动作：

- 建立 providers、prompting、tokens、correction、research、retrieval、artifacts。
- 将剩余 `src/llm/` 整体迁入 `asr_playground/llm/`，随后在该 package 内机械整理。
- 首先整体移动大文件，不立刻拆函数。
- 保留 knowledge package，先只更新外部 import。
- 删除通用 `stages` 名称，将其内容归入 correction。
- 消除私有跨 package import 和 provider 反向 re-export。
- 清空 `llm/__init__.py` 的 config re-export，调用方改为真实模块路径。
- 将 provider 的 Gemini countTokens callable 注入 token counter，不保留
  `token_budget → client._first_gemini_api_key`。

验收：

- `llm` 顶层只保留少数跨域模块。
- provider 层不依赖 workflow/artifact。
- correction、research、knowledge 通过明确的 provider client 调用模型。
- LLM scoped suite和完整测试通过。
- prompt artifact、exchange log、session resume 输出不变。
- `python -m pytest -q tools/session_replay/test_session_replay.py` 通过。
- `python -m tools.session_replay correction --dry-run` 通过。

### Phase 5：拆分 LLM 大文件

每次只拆一个大文件，并独立验收：

1. `web_search.py` → retrieval models/render/providers。
2. `correction_loop.py` → window/query/validation/retry/rendering。
3. `csv_utils.py` → correction output protocol 的 parse/validate/merge/render。
4. `knowledge/base.py` → schema/storage/repository/artifacts。
5. 评估 `research.py` 与 `search_loop.py` 是否仍需拆分。

拆分准则：

- 每个模块有一个可描述的主要职责。
- 依赖方向变清晰。
- 不为追求低行数创建只有一两个薄函数的模块。
- 不通过 `__init__.py` re-export 大量私有实现。

验收：

- 每个拆分提交独立测试通过。
- 没有新增 import cycle。
- 没有测试为了迁就结构而降低断言强度。
- 每次影响 correction/research/retrieval 注入接口的提交都运行
  `tools/session_replay/test_session_replay.py` 与 correction dry-run。

### Phase 6：整合已批准的相关重构

`b5eea87` 已修复 custom artifact directory 和 batch metadata 的具体行为缺陷；
本阶段只要求新结构保持这些 regression tests。尚待实现的结构工作：

- `RunContext + StageRecorder`。
- vocal separation 第三方 adapter。
- reference ingest 复用 batch pipeline + refined subtitle post task。

落点：

- 通用 run context：`asr_playground/run/`。
- LLM exchange/report：`asr_playground/llm/artifacts/`。
- separation adapter：`asr_playground/speech/preprocessing/separation.py`。
- reference ingest：`asr_playground/workflows/reference_ingest.py`。

验收：

- 一个 logical run 只有一个 RunContext。
- artifact root 在 run 开始时解析一次。
- reference post task 与 correction 保持原有顺序。
- batch ASR 并发继续由普通 batch scheduler 管理。
- 顺序回归测试使用 fake stages 断言
  `correction(i) → refined update(i) → correction(i+1)`；不得只靠 worker 数间接推断。

### Phase 7：文档与清理

动作：

- 更新 `CLAUDE.md` 架构图。
- 更新 README_DEV、testing、pipeline/ASR/LLM 相关文档中的模块路径。
- 更新 changelog。
- 删除旧目录、空 package、临时 wrapper 和无效 import alias。
- 搜索所有旧路径引用，包括直接 import、字符串 monkeypatch target 和
  `sys.path.insert(..., "src")`。

验收：

```text
rg "llm\\.(stages|media_source|ffmpeg_clips|audio_clips|srt_utils|srt_alignment)" src test tools desktop docs
rg "import (asr_align|vad_asr|wt_shard|segment_split|to_srt|subtitle_metrics)" src test tools desktop docs
rg "from (resource_profiles|gpu_stage_gate|run_metadata|utils|to_srt|subtitle_metrics)" src test tools desktop docs
rg "monkeypatch\\.setattr\\([\"'](llm|asr_align|vad_asr)" test tools desktop
rg "sys\\.path\\.insert.*[\"']src[\"']" test tools desktop
```

只允许在明确标为历史记录的文档中出现旧路径。

## 8. 测试与验证

### 8.1 每个 Phase 的最低测试

```text
python -m compileall -q src test
python -m pytest -q
```

根据范围附加：

```text
python -m pytest -q -m asr
python -m pytest -q -m pipeline
python -m pytest -q -m llm
```

完整 `pytest -q` 是强制门槛；scoped suite 只用于快速定位，不作为完成定义的独立证据。
marker 映射随新测试文件同步维护。
Phase 1–5 只要修改 session_replay 所依赖的 import 或注入点，还必须运行：

```text
python -m pytest -q tools/session_replay/test_session_replay.py
python -m tools.session_replay correction --dry-run
```

### 8.2 Isolated wheel smoke

必须在不暴露仓库 `src` 的两个隔离环境中：

1. 构建 wheel。
2. 创建 ASR virtualenv，只安装 wheel `[asr]`，导入 pipeline/speech 并执行 ASR CLI `--help`。
3. 创建 harness-only virtualenv，只安装 wheel `[harness]`，导入 LLM/media/subtitles，
   执行 LLM CLI `--help`。
4. harness-only import 后断言 `sys.modules` 中没有 `torch`、`torchaudio`、
   `numpy`、`numba`、`audio_separator` 或 `whisper_timestamped`。
5. 在两个环境中读取各自需要的 package data。
6. 用轻量 fixture 运行不需要模型/API 的 parse/render/planning 路径。

ASR 环境至少验证：

```python
import asr_playground.pipeline
import asr_playground.batch
import asr_playground.speech.recognition.transcribe
```

harness-only 环境至少验证：

```python
import asr_playground.llm.correction
import asr_playground.llm.knowledge.update
import asr_playground.media
import asr_playground.subtitles
```

### 8.3 Import 边界检查

增加轻量静态检查或测试，禁止：

- `media` 导入 `speech`/`llm`。
- `subtitles` 导入 `speech`/`llm`。
- `speech` 导入 `llm`。
- `providers` 导入 correction/research/knowledge/artifacts。
- 跨 package 导入下划线私有符号。
- `tokens` 导入 provider client 或 API-key 私有函数。
- 非 speech package 顶层导入 ASR 重依赖。
- 非 `paths.py` 模块用 `parents[N]` 推导 repo root。
- `__init__.py` 用批量 `from .x import ...` re-export 伪装旧路径；明确 package
  门面若确有必要必须逐项白名单。

实现可使用简单 AST 测试，不强制新增第三方 import-linter 依赖。
第一版检查在 Phase 1 建立，不能等移动结束后才补。

### 8.4 行为一致性

本重构不运行重模型来逐提交验证数值，但必须用固定 fixture 验证：

- SRT parse/render byte output。
- correction CSV parse/validate/merge。
- window planning。
- timestamp mapping。
- shard planning/merge。
- metadata JSON schema。
- task report rendering。
- knowledge storage/update 的无 API 路径。
- CLI 参数和默认值。
- reference ingest correction/refined-update 的跨任务顺序。

重资源真实模型验证按项目规则单独批准执行。

## 9. 提交与 review 策略

不创建一个包含全部移动的超大提交。建议：

1. `package: introduce asr_playground namespace and wheel discovery`
2. `refactor: move shared media and subtitle modules`
3. `refactor: move reference ingest to workflows`
4. `refactor: organize speech runtime and preprocessing`
5. `refactor: split recognition checkpoint and sharding`
6. `refactor: split recognition windows and timestamp mapping`
7. `refactor: split recognition decoder and recovery`
8. `refactor: organize llm providers and tokens`
9. `refactor: organize llm prompting and correction`
10. `refactor: organize llm research retrieval and artifacts`
11. `refactor: split retrieval providers`
12. `refactor: split correction output and loop`
13. `refactor: split knowledge repository`
14. `docs: update architecture and module references`

每个提交要求：

- 目标职责单一。
- 无不相关格式化。
- 测试通过。
- 删除旧文件与新增文件同时出现，便于 git 检测 rename。
- 不夹带算法调参或 prompt 内容变化。
- 大文件使用 `git mv`；同一提交允许必要的 import-only 修改以保持可运行，
  但不得夹带函数重写。用 `git diff --summary` 和
  `git log --follow <new-path>` 验证历史连续性。

如果实现过程中发现行为修复，应单独提交，并说明它属于已知缺陷修复还是新发现问题。

## 10. 风险与缓解

### 10.1 Package data 漏装

风险：

- prompt template 或 PSV 在源码运行正常，但 wheel 中缺失。

缓解：

- isolated wheel resource test。
- 使用 `importlib.resources`。
- 不依赖当前工作目录。

### 10.2 大规模 import 修改掩盖循环依赖

风险：

- `__init__.py` re-export 形成隐式 cycle。

缓解：

- `__init__.py` 保持最小。
- 生产代码从真实定义模块导入。
- 每个 Phase 做 cold import smoke。

### 10.3 CLI 入口变化

风险：

- console script target 移动后无法启动。

缓解：

- 保留命令名。
- 对每个入口执行 isolated `--help`。
- CLI wrapper 与业务 service 分离。

### 10.4 默认路径变化

风险：

- `Path(__file__).parents[N]` 因 package 层级变化而指向错误位置。

缓解：

- package data 使用 `importlib.resources`。
- `.env`、`config.toml`、`.state`、tokcount、knowledge root 统一走 `paths.py` 契约。
- 对 env file、state dir、tokcount、knowledge root、artifact root 增加路径测试。
- knowledge root 无法确定时明确失败，避免在错误位置创建并 auto-commit。

### 10.5 重构与功能修复互相污染

风险：

- checkpoint、metadata、reference ingest 等已知修复与文件移动混在一起，难以 review。

缓解：

- 机械移动与行为修改分开提交。
- 每个行为修复保留针对性 regression test。
- 以 Phase 6 的接口落点协调，不要求同时实现。

### 10.6 大文件拆得过细

风险：

- 模块数量增加但依赖不清晰，阅读时需要跨多个薄文件跳转。

缓解：

- 先整体移动，后按实际职责拆。
- 不使用行数作为唯一拆分标准。
- 一个模块只有在拥有独立数据模型、I/O 边界、算法阶段或测试边界时才拆出。

## 11. 非目标

本计划明确不处理：

- desktop 内部结构重构；但 Phase 1 必须同步修复它对旧 `src` 布局的 4 处引用并跑 3 个测试。
- `tools/qwen3_explore`、`split_explorer` 等实验工具的 import 迁移；它们会因旧路径删除
  而失效，明确记录为已知结果。`tools/session_replay` 是 prompt 迭代验收协议，
  作为例外必须随 LLM Phase 迁移并通过测试/dry-run。
- ASR 模型、decode 参数或数值算法调优。
- prompt 文本、知识库 schema 或搜索策略调整。
- 旧 checkpoint migration；旧 schema 由单独修复明确失效。
- 为旧内部 Python import 路径提供长期兼容 wrapper。
- 新增通用 plugin/framework 或依赖注入框架。
- 为追求“架构纯洁”替换已经清晰的小函数和 dataclass。

## 12. 完成定义

第 10 步完成需要同时满足：

- 所有生产代码位于 `src/asr_playground/` package 下。
- `speech`、`llm`、`media`、`subtitles`、`workflows` 边界符合本计划。
- 当前 `asr_align.py` 职责被拆到 recognition 下，不再存在误导性的总 alignment 模块。
- `llm` 顶层不再平铺 correction、research、retrieval、provider、token、artifact 实现。
- reference ingest 位于 workflows，并复用普通 batch pipeline。
- setuptools 使用 package discovery，不再依赖手工 `py-modules` 清单。
- wheel 包含全部 Python 模块、prompt templates、LICENSE 和 model catalog。
- 所有 console scripts 可从 isolated wheel 启动。
- harness-only wheel import 不加载 ASR 重依赖。
- 非 heavy 完整测试通过。
- 无新增 import cycle 或跨层私有依赖。
- desktop 的 package-layout 引用与 session_replay 已同步。
- 文档和架构图更新到新路径。
- 工作区不存在旧路径 wrapper、重复实现或过渡文件。

## 13. Review 后的定夺

- 接受：`speech`/`recognition` 命名、公共 media/subtitles、LLM 七个子域和
  workflows 方向。
- 接受并补齐：desktop 最小随行修改、runtime path resolver、utils 落点、
  harness extras 边界、session_replay 验收、reference ingest 顺序测试。
- 定死：`asr-align`/`vad-asr`/`asr-wt` 三个 CLI wrapper；overlap/zero-length
  helper 留在 recognition；基础 model/config dataclass 留在 `llm.config`。
- 定死：tokens 通过注入调用 provider countTokens；subtitles alignment 返回完整
  diff，token 截断由 LLM 调用方负责；`PROMPT_VERSION` 和 `ContextPack` 的稳定落点
  分别是 prompting loader/context。
- 不采用“纯移动提交完全不得改 import”的原建议：这会制造不可运行的中间提交。
  改为 `git mv + 必要 import-only 修改`，语义拆分另交，并显式验证 rename history。
- reviewer 所述 5 个 wheel 漏装模块已在本计划执行前的 `b5eea87` 修复，不再是
  Phase 0 已知失败。
