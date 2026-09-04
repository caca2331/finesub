# finesub 开发者 / Agent 说明

这份文档面向维护者和 LLM coding agent。目标是减少误改、误跑重资源任务，以及把生产入口和实验入口区分清楚。

## 当前项目地图

生产主路径：

```text
src/finesub/pipeline.py
```

生产流水线：

```text
source audio
  -> src/finesub/speech/preprocessing/separator/separation.py
  -> src/finesub/speech/recognition/vad_asr_stage.py
  -> src/finesub/speech/postprocessing/stabilization.py
  -> src/finesub/subtitles/rendering.py（默认产出 *-raw.srt）
```

关键模块：

- `src/finesub/pipeline.py`：命令行前端与唯一入口（`python -m finesub.pipeline`）——选项面、manifest、item 构造、结果呈现；一个源与 N 个源同路。
- `src/finesub/stages.py`：生产编排本体 `run_pipeline`，负责中间文件路径、跳过已有输出、stage-based resume。默认跑到 `raw-srt`；`translated-srt` / `final-srt` 才进入 LLM 纠错翻译和 SRT 后处理。
- `src/finesub/scheduler.py`：download / ASR / LLM 三 bin 引擎（领域无关；item 与选项面在 `pipeline.py`）。
- `src/finesub/speech/preprocessing/separator/separation.py`：人声分离，使用 `audio-separator`。
- `src/finesub/speech/preprocessing/vad.py`：流式 VAD 检测与能量轨迹。
- `src/finesub/speech/recognition/vad_asr_stage.py`：组合 VAD 与 Whisper recognition，输出未稳定化的 `*-aligned.json`。
- `src/finesub/speech/recognition/transcribe.py`：recognition service（单 worker——单文件分片已于 2026-08-02 移除，见 `docs/wt-parallelism.md`）；仍包含待拆分的 windows、decoder、timestamp mapping 和 recovery。
- `src/finesub/speech/recognition/checkpoint.py`：ASR partial identity、schema、原子写入与清理。
- `src/finesub/speech/recognition/segments.py`：识别输出的重叠收回、零时长修复与空段过滤。
- `src/finesub/speech/postprocessing/stabilization.py`：独立 ASR 稳定化 stage，按 profile 从 aligned 生成 stable。
- `src/finesub/subtitles/rendering.py`：stable JSON 转 SRT。
- `src/finesub/speech/recognition/fw_refine_backend.py`：patched CT2 适配层、模型池与批解码 driver。
- `src/finesub/speech/runtime/resources.py`：五个 GPU 档位 `cpu`/`entry`/`standard`/`standard_large_vram`/`high`（默认 `auto`，读驱动定档）、每档的空闲显存要求、WT/Separator 实例数与资源上限检查。
- `src/finesub/speech/preprocessing/energy.py`：VAD-energy 核心算法，体积较大，修改需谨慎。
- `src/finesub/speech/preprocessing/audio.py` / `spectral.py`：前者只管「把音频从磁盘取成
  波形」（解码、切片、重采样、并声道），后者是加权频谱能量本身（滤波器组 + numba/torch 帧循环）。
  **`spectral.py` 不是 VAD 私有的**：`recognition/transcribe.py` 也用 `weighted_spectral_energy_db`
  做逐段能量，这正是它没有并进 `energy.py` 的原因。
- `src/finesub/speech/preprocessing/separator/`：分离器自成一包——`separation.py`（阶段本体）、
  `accel.py`（选档与编译缓存）、`separator_aoti.py`（AOTI 包的构建与加载）。三个模块只互相依赖。
- `src/finesub/media/`：下载/URL 选择、ffmpeg/ffprobe 和 clip 提取；公共轻量层，不依赖 speech/LLM。
- `src/finesub/subtitles/`：SRT model、alignment、metrics、postprocess 和 rendering；公共轻量层。
- `src/finesub/workflows/reference_ingest.py`：跨 batch/media/speech/LLM/knowledge 的参考素材导入 workflow。
- `legacy/`：superseded 脚本（旧 `main.py`/`vad.py`/`align.py` 等，以及从 `src/` 移入的 `to_toon.py`、`rms.py`）。**本地目录，gitignore，不随仓库分发**；不被当前流水线使用，避免在此基础上开发。
- `src/finesub/llm/`：实验性 LLM 工具，包括两轮背景调查、纠错翻译 prompt/窗口规划/API harness；不是默认生产流水线的一部分。
  - `routing/`：一次调用选谁来答——`profiles.py` / `config.py` / `model_catalog.psv` + `model_catalog.py` / `model_routes.toml` + `model_routes.py` / `execution_policy.py` / `model_router.py` / `capabilities.py` / `api_keys.py`。层次单向（事实→组合→执行身份→逐调用计划），由 `test/test_import_boundaries.py` 固化。
  - `agent/`：本机 agent 后端（`local_agent.py` 与八个 `agent_*.py`）。模块名保留 `agent_` 前缀——`finesub_bootstrap.shell` 用字符串 `python -m finesub.llm.agent.agent_cleanup` 调它们。
  - `routing/profiles.py`：五条正交开关轴（`--correction-media` 与 `--planning-media` 各取 `text|audio|video` × `--retrieval none|local|native` × `--difficulty quality|intermediate|efficiency` × `--continuity serial|parallel`，输出公式 `k × c × csv_tokens`；旧 route/level preset 已退役，换算表见 `docs/llm_harness_routing.md`）。
  - `routing/api_keys.py`：读取 `.env` 的命名 key，并按 `config.toml` 统一解析 provider 开关与 pool。
  - `llm_runtime.py`：生成调用底层封装（原 `llm.py`）。
  - `research.py`：两轮背景调查 + 多轮搜索 loop（`run_research_stage`，原 `stages/research_stage.py` 已并入此文件）。
  - `stages/`：`plan.py`（窗口规划、fast 模式判定）、`fast_session.py`（融合会话）、`correction/`（纠错窗口循环：run/serial/parallel/attempts/query_round/context/commit/metadata）。
  - `knowledge/`：知识库子包——`base.py`（原 `knowledge_base.py`）、`update.py`（统一知识更新入口，原 `knowledge_update.py`，CLI 现为 `python -m finesub.llm.knowledge.update`）、`style.py`（`--style` 的选取与注入渲染；2026-09-02 取代了 `mistakes.py` 那两个 markdown 台账）、`entries.py`、`feedback.py`、`materials.py`。
  - `--llm-media video` 的视频剪辑经 `--video` 接入。
- `src/finesub/llm/prompt_templates/`：LLM 后处理 prompt/harness 模板，随主仓库版本化；prompt 迭代只改模板文件。v7 起纠错侧为骨架 + fragment 组装，选择逻辑在 `src/finesub/llm/prompt_compose.py`（按 preset 挑 fragment），组装参考见 `docs/llm_prompts.md`。
- `src/finesub/llm/routing/model_catalog.psv`：pipe-delimited 模型事实表——能力位、**限额（rpm/tpm/rpd/tpd，限流器直接读它）**、端点方言与 URL、thinking 映射、`token_scale`。事实在这里，组合（模型组/任务组/预设）在同目录的 `model_routes.toml`；旧的「role binding 写在 Python 里」已被 model-routing v2 取代，见 [`docs/manual/model-routing.md`](docs/manual/model-routing.md)。
- `knowledge/`：本地知识库数据（不是 `finesub/llm/knowledge/` 代码包）。结构、更新流程见 [`docs/knowledge.md`](docs/knowledge.md)。主 git 不追踪；目录内有独立 git 仓库，知识更新 apply 后自动 commit。主仓样板见 [`examples/knowledge/`](examples/knowledge/)。
- `docs/llm_design_notes.md`：LLM 纠错与翻译后处理层架构意图与设计决策。
- `docs/llm_harness_behavior.md`：当前 LLM harness 的窗口拆分、重试、拼接和 prompt 输入行为。
- `docs/knowledge.md`：知识库全套（结构、`--knowledge` 三态、统一知识更新、mistake 台账、`reference_ingest`）。
- `docs/llm_prompts.md`：prompt/fragment 组装参考。
- `docs/testing.md`：测试命令与域标记。
- `docs/vad-energy.md` / `asr-align.md` / `vad-asr.md` / `asr-stabilize.md`：stable 之前四个独立工具的运行时行为、接口和产物语义。
- `src/finesub_bootstrap/`：终端用户装机层（`AppPaths` 布局、校验下载、安全解压、
  uv 托管运行环境 + 跨进程安装锁），desktop 与 CLI 壳共用，禁止反向依赖
  `desktop`；测试在 `desktop/backend/tests`（desktop CI 是唯一 Windows lane）。
- `cli/`：可发布的 `finesub` CLI 壳（独立 pyproject；wheel = 薄启动器 +
  `_vendor` 源码快照，唯一入口 `finesub`）。构建 `cli/scripts/build-wheel.ps1`，
  版本取仓库根 `VERSION`；用法见 `cli/README.md`。
- `tools/`：独立开发工具，全部**按需维护**——不随主程序改动自动更新，测试不进默认套件。
  **总索引在 [`tools/README.md`](tools/README.md)**（2026-09-03 新增）：13 个子目录按
  活跃 / 一次性实验 / 跑不起来分三类，外加「找 X 去哪」一张表；活跃的四个是
  `bench/`、`session_replay/`、`segmentation_gold/`、`wt_refine_port/`，而 `tokcount/`
  住在这里但其实是生产组件。

## 分步调试

生产时优先使用 `python -m finesub.pipeline`。需要定位问题时，可以分步运行。

1. 人声分离：

```powershell
python -m finesub.speech.preprocessing.separator.separation data/input.wav -o out/input-vocal.flac --gpu-tier standard
```

CUDA 分离固定启用 AMP；共享模型预热使用相同精度，避免产生一次额外的 FP32 activation
峰值。FP32 只由开发基准工具生成对照，不作为生产 CLI 开关。标定与否决实验见
[`docs/separator-optimization.md`](docs/separator-optimization.md)。

2. VAD + ASR 对齐：

```powershell
python -m finesub.speech.recognition.cli.vad_asr out/input-vocal.flac --output out/input-aligned.json --model large-v3-turbo --language en --gpu-tier standard
```

3. ASR 稳定化：

```powershell
python -m finesub.speech.postprocessing.stabilization out/input-aligned.json -o out/input-stable.json --profile 0
```

4. 导出 SRT：

```powershell
python -m finesub.subtitles.rendering out/input-stable.json -o out/input.srt
```

### 开发专用的 ASR 开关（不写进面向用户的 README）

- **模型**：默认 `large-v3-turbo`，`--model large-v3` 可切换。**目前仅供开发比对**：实测
  large-v3 的生产异常率与 turbo 持平（41 vs 45 / 310 窗口）、文本语义互有胜负，而解码成本是
  3.4×，没有证据支持在生产里换用。依据见 [`docs/wt-refine-port.md`](docs/wt-refine-port.md)。
- **backend**：只有 `fw-refine`（打过补丁的 CTranslate2 一遍式 WT refine）。
  `whisper-timestamped` 已于 2026-08-02 移除——为优化要改 refine 内部逻辑，维护两套的成本
  不划算；迁移验收见 [`docs/wt-refine-port.md`](docs/wt-refine-port.md)。
  它需要 `tools/wt_refine_port/ct2-patches/` 那套补丁编译出的 CT2
  （`RefinedWhisperModel.__init__` 会在 stock CT2 上直接报错）。

```powershell
pip install -e ".[asr]"
```

`faster-whisper` 与 `ctranslate2` 在 `asr` extra 里**精确钉版**（1.2.1 / 4.8.1）。fw-refine
继承 faster-whisper 的内部实现并读取 CT2 的解码轨迹，任一侧的小版本变动都可能悄悄改变输出——
升级时先 faster-whisper 后 CT2（后者的可选范围由前者声明），并重跑输出一致性验证。

### 分离器的工作采样率（开发专用，默认不动）

`--separator-rate {44100,32000,22050}`（`finesub.pipeline` 与
`finesub.speech.preprocessing.separator.separation` 都有），默认 **44100**，也就是 BS-Roformer
自己的率、它的权重唯一训练过的那一个。

它省时间的机制只有一条：块是**固定 352800 个采样点**（`stft_hop_length × (dim_t − 1)`，与采样率
无关），所以一块覆盖 `352800 / rate` 秒，降率就按比例减少块数。**没有任何算子级收益**——稳态下
每块成本三档相同。长素材上 22050 约 1.5×、32000 约 1.26×，短素材更少（固定开销摊不开）。

**两档都不推荐**，理由是收益不成比例：分离只占 raw-srt 阶段约四成、完整 run 约一成，
省下的部分折合完整 run 约 4%，而变动落在一个所有下游都依赖的阶段上。22050 在生产输入上
复验后词准确率代价已不可判定（+0.014，符号 3/5，与「换个 worker 数」区分不开），但它会让
VAD 多 admit 约 2.7 倍时间，那一条还没有论证过无害。**16000 不在选项里**：它会整段抹掉人声，
结果随素材而异，最差一例丢掉 476 个有声秒里的 98 个。全部依据见
[`docs/separator-optimization.md`](docs/separator-optimization.md) 的 E12 与
[`tools/separator_rate/`](tools/separator_rate/README.md)。

**换档必须删产物。** 分离阶段按**存在性**跳过，文件名不带采样率，所以已有的 `<stem>-vocal.*`
会被原样复用——换档前删掉它和它的全部下游。run metadata 的 `workers.vocal_separation.sample_rate`
记着某份产物是用哪一档做的。

### 分离器的编译加速

分离阶段会自动选一档后端。run metadata 记三个字段：`accel_requested` 是选档结果，
`accel` 是 `apply_acceleration` 报告的实际生效后端，两者不同时 `accel_fallback_reason`
记降级原因（安装失败与首次 forward 失败都走这里）：

| 档 | 条件 | 2.11 实测（2015s / 2 worker） |
| --- | --- | --- |
| `aoti` | 本机能建包（需 MSVC）或已有包 | 1.895× |
| `jit` | 只有 triton，且**输入 ≥ 600 秒** | 1.381× |
| `eager` | 其余 | 1.000× |

两档都需要 `triton`（Windows 上由 `triton-windows` 提供，自带 TinyCC，**不需要 Visual
Studio**）。AOTI 额外需要 MSVC 编 C++ wrapper，由 `vswhere` 定位。**这次探测在选档之前
做**：没有编译器的机器直接落到 `jit`，不会先宣告一次「即将编译约 90 秒」再降级。首次
构建约 90 秒，会在 stderr 说明。

JIT 有时长门槛而 AOTI 没有，是因为每进程准备成本差一个量级（约 35s vs 2s）；实测回本点
约 800 秒，取 600 秒是为了与 `block_seconds` 常数一致。

AOTI 包默认对两个 Transformer 轴开 `max_autotune`（Triton GEMM 模板 + epilogue 融合）：
长素材 +3.7%，但**首次 forward 多付 0.55 秒**，所以不到约 6.4 分钟的输入是净亏一点。
**不对两个 band 模块开**——那会把它们的校验误差放大 5×/163×，端到端掉 3.4dB SI-SDR。
开关是 `build_packages(max_autotune=...)`，数据见 `docs/separator-optimization.md` E14。

`torch.compile` 是惰性的，真正的编译发生在第一次 forward。所以 `jit` 装好后会**再做一次
warm-up** 把编译提前到任何 block 开跑之前；这次失败（2026-08-20 实例：托管 inductor 目录里
某个 Triton kernel 缺 `.json`）会**原地还原**为 eager（不重载权重）、删掉托管的
`inductor/` 缓存、写 probe `jit=unavailable` 让下次运行直接 eager（显存不足除外），发
`separator-jit-failed` 警告并继续本次任务。安装中途失败同样回滚干净（AOTI 加载亦然），不会留下半编译的模型。
想重新启用就删 `<key>/` 目录，与 AOTI 的 probe 一样。

产物全部在 `cache/separator-accel/<key>/`（不是从 checkout 运行时退到
`~/.cache/audio-separator/accel/`；设了 `FINESUB_MODEL_DIR`——桌面端 worker 会设——
则优先落到 `<FINESUB_MODEL_DIR>/audio-separator/accel/`，分离器模型权重同理落
`<FINESUB_MODEL_DIR>/audio-separator/`，避免写进随版本更替的 app 目录或用户 home），
`<key>` 由 `BUILD_FORMAT`、torch 版本、CUDA、GPU 架构、**卡型**和
checkpoint 组成——**换任意一项即换目录，这就是失效机制**，不需要额外的比对代码。
卡型不能省：`max_autotune` 是在当前卡上实测选 kernel 的，同架构不同卡（sm_120 从
5060 Ti 到 5090）赢的 tile 不一样，只按架构分目录会静默地把旧卡的包喂给新卡。
`BUILD_FORMAT` 是给「key 本身分辨不出来」的构建配置变化留的手动闸：改了 inductor 开关或
target 集合就把它加一，旧目录随之整体作废（2026-08-27 因 `max_autotune` 从 `1` 升到 `2`）。
构建或加载失败会写进同目录的 `probe.json`，从而不会每次运行都重付一遍构建；删掉整个
`cache/separator-accel/` 就能让它重试。

```bash
rm -rf cache/separator-accel        # 重置全部加速状态（包、JIT 缓存、探测结果）
```

```bash
FINESUB_SEPARATOR_ACCEL=off ...     # 本次运行强制 eager，用于排查
```

**任何一步失败都降级到 eager 并在 stderr 说明**，绝不让加速不可用变成分离失败。已知的一处
粗糙：同一次运行内 AOTI 失败只退到 eager，不退到 JIT——要退需要把时长传进
`apply_acceleration`。因为选档前已经探过编译器，最常见的「无 MSVC」根本走不到这里，剩下的
路径下一次运行就会靠 `probe.json` 落到 JIT。

首次构建**在已经加载好的那个模型上做**，不另开一个。实测（sm_120 / 2.11）峰值 reserved
2.12GiB，而另建一份是 2.88GiB——4GB 档放不下，且构建期 OOM 会被记成「这台机器建不了包」。

JIT 档会改两个进程级设置且**不恢复**：`TORCHINDUCTOR_CACHE_DIR`（仅当进程启动时没设过）和
`torch._dynamo.config.enable_cpp_symbolic_shape_guards`。`torch.compile` 是惰性的——不到第一次
forward 什么都不编——所以恢复现场只会赶在它被读到之前把值改回去。进程内没有第二个
`torch.compile` 使用者，成立的前提就是这一条。

### torch 版本范围

**`torch==2.11.0` + `torchaudio==2.11.0` + `torchvision==0.26.0` +
`triton-windows==3.6.0.post26`，全部精确钉版，要动一起动。**

原先钉死 2.8.x 的理由（「2.9 把解码路由到 torchcodec，打断这套栈的 ffmpeg 后端」）已经
不成立：解码现在只走 soundfile，soundfile 打不开的容器由 ffmpeg 先转一份无损 FLAC
（`ensure_decodable_input`），`torchaudio.load/info/save` 全部删除。torchaudio 的接触面
只剩 `functional` 里的 `resample`、`highpass_biquad`、`lowpass_biquad` 三个纯 DSP 函数。

**为什么是精确钉版而不是范围**：四样东西跟 torch 绑定，其中两样不会自己拦住你——
`torchvision` 由 `audio-separator → onnx2torch-py313` 传递引入；`triton` 完全不声明 torch
约束（映射见下表）；patched CT2 则通过它运行时加载的 cuBLAS SONAME 绑定 CUDA 大版本。
范围写法会让解析器有机会配错，而配错的症状是运行时那种难读的 dlopen 失败。

**为什么停在 2.11**：

| torch | torchaudio | CUDA index | triton | 实测 |
| --- | --- | --- | --- | --- |
| 2.9.0 | 2.9.0 | cu128 | 3.5.0 | 基准语料所在版本 |
| 2.10.0 | 2.10.0 | cu128 | 3.6.0 | ✅ 全链路 |
| **2.11.0** | **2.11.0（最后一个）** | **cu128（最后一个）** | 3.6.0 | ✅ 全链路 |
| 2.12.1 | ❌ 不存在 | cu130 | 3.7.1 | ⚠️ 能跑但改了转写 |

2.12 起 torchaudio 没有配套版本，且迁到 cu130——那里 torch 自带 `cublas64_13.dll`，而
patched CT2 运行时找的是 `cublas64_12.dll`（详见
[`docs/ct2-distribution.md`](docs/ct2-distribution.md)）。

**2026-08-03 实测**（60 秒真实素材，`.mp4` 入、走完整 speech 链路，与 2.9.0 基线对比）：

| torch | ASR 段数 | ASR 文本 | 分离 max abs err / SNR |
| --- | ---: | --- | --- |
| 2.10.0 | 13 = 13 | 0 处差异 | 2.75e-4 / 74.42dB |
| 2.11.0 | 13 = 13 | **0 处差异** | 2.75e-4 / **74.42dB** |
| 2.12.1 | 13 = 13 | **1 处差异**（`などくらい`→`謎くらい`） | 2.14e-4 / 72.61dB |

⚠️ **2.11 相对 2.9 不是逐位相同**（2.8↔2.9 曾经是）。2.75e-4 优于 E0 的验收基线
（AMP vs FP32：max 4.27e-4 / SI-SDR 71.4dB），且真实语音段边界只差一两个 20ms VAD 帧，
唯一较大的变化是一个 `ご視聴ありがとうございました` 幻觉段的跨度从 5.0s 缩到 0.56s。
**但这意味着 `separator-optimization.md` 里 E0–E10 的数字是 2.9.0 上取的，不精确适用于
2.11**——要让文档与生产一致，需在 2.11 上重跑关键几行。

**安装必须指向 download.pytorch.org**：CUDA 构建带 `+cu128` local label，PyPI 上可能是
不带 CUDA 的构建，装错了 GPU 路径会静默失效。

**待办**：2.12 及以上要等 torchaudio 的替代方案（那三个 DSP 函数自实现并不难），或等
CT2 用 CUDA 13 重编。

### patched CT2 的分发方案

`pip install` **拿不到**打过补丁的 CTranslate2：PyPI 上只有 stock 版，它满足 `==4.8.1` 却跑不了
fw-refine（目前靠 `RefinedWhisperModel.__init__` 在构造时报错兜住）。方案如下。

**wheel 发到 GitHub Release，不进仓库。** 仓里已有 `bin/windows-amd64/tokcount.exe`（17.9 MB）
的先例——它同时也发 Release 供打包前端下载，两条路各有其用——但 CT2 wheel 不该照办：
它约 17 MB（内含未压缩 79 MB 的 DLL），且每次 CT2/CUDA/补丁
变动都要重编。git 会按内容去重，**同一份 wheel 提交多次只占一份**；但每个**不同**版本都会在
公开仓库里永久留下一个 blob，而 clone 的代价落到所有人头上——包括只用 LLM 层、根本不装
ASR 的人。Release 资产不进 clone，且给出稳定 URL。

**用独立 tag**，形如 `ct2-<上游版本>+finesub<产品版本>`，例如 `ct2-4.8.1+finesub0.4.0`。
label 记的是**引入该 wheel 的那次 finesub 发布**（便于追溯是哪一版换了二进制），但不等于每发
一次产品版本就重发 wheel——没换二进制就继续用旧 label。完整口径见
[`docs/ct2-distribution.md`](docs/ct2-distribution.md)，这里不重复。

打包与发布见 [`docs/ct2-distribution.md`](docs/ct2-distribution.md)，用户侧安装见
[`docs/manual/ct2-wheel.md`](docs/manual/ct2-wheel.md)，补丁与 CMake 构建标志见
[`tools/wt_refine_port/ct2-patches/README.md`](tools/wt_refine_port/ct2-patches/README.md)。

**当前状态（2026-08-03）：wheel 已做，还没发 Release。** 产物在本机
`CTranslate2/python/dist/`，装完不再需要 `sys.path` 注入或 `os.add_dll_directory()`。
发 Release、以及定下 `cublas64_12.dll` 的来源（目前靠先 `import torch` 隐式带入），
是分发仅剩的两件事。

## 开发原则

- 不要维护 `requirements.txt` 或 `requirements-dev.txt`；依赖只放在 `pyproject.toml`（安装 extras 见 README.md「安装」）。
- 根目录不放新的音频、字幕、JSON 或媒体产物；使用 `data/`、`out/`、`tmp/`。
- 默认不要改 VAD/ASR 参数。若必须改，说明**对输出质量的影响**，并补测试或实验记录。
  （注意口径：要说明的是质量，不是逐位一致性——见下「一致性是证明，不是及格线」。）
- **选项的默认值住在后端，前端只在少数情况下覆盖。** 解析顺序（高优先级在前）：

  ```text
  命令行参数  →  项目配置文件  →  全局配置文件  →  前端默认值  →  后端默认值
  ```

  推论，按重要性排序：

  - **后端是唯一的真相源，但「后端的哪里」必须挑明。** 本项目的调度层
    （`pipeline.py` 的 `main`）**显式传每一个 kwarg**——`vad_silero_assist=args.vad_silero_assist`
    这种。所以 argparse 传 `None` 时，`run_pipeline` 签名上的默认值**根本不会生效**，
    Python 只在实参缺席时才用签名默认。两种可行写法，**必须二选一并写明选了哪个**：

    1. **调度层剔除值为 `None` 的 kwarg**，让签名默认真正生效——此时真相源是签名字面量；
    2. **后端形参接受 `None`，在函数内部用共享 resolver / 常数解析**——
       此时真相源是**那个 resolver 与常数**，不是签名字面量。

    **本项目现有的正确样例走的是 ②**，照它写：
    `vad_asr_stage.resolve_split_params(explicit)`（docstring：显式值 > `config.toml` 的
    `[segmentation] length_scale` > 标定后的代码默认值）、
    `pipeline.resolve_knowledge_switch(knowledge, llm_difficulty)`（docstring：
    「The one rule three front ends share」）。**新选项优先用 ②**——它把整条优先级链
    收在一个可单测的纯函数里，而 ① 只解决签名那一层。
  - **argparse 不许再写一份默认值。** CLI 的 `default` 应当是 `None` / 不给，
    让「用户没说」这件事**可区分**地传到后端 resolver 去。在 `store_true` 上写死
    `True`/`False` 等于在 argparse 里复制了一份后端默认值，而且顺手废掉了关闭形态——
    布尔开关用 `argparse.BooleanOptionalAction`（给出 `--no-<flag>`）或三值 `auto|on|off`。
  - **前端默认值是显式的少数例外，每个都要有理由，并且要说明它在链上的真实位置。**
    例：桌面 `TaskRequest.knowledge` 默认 `'update'`（注释写明「知识库才是让后续任务
    变好的东西」），后端是 `None` → `resolve_knowledge_switch` → `'collect'`。
    这类覆盖应当登记成一张**带理由的豁免表**，而不是散落成第三份副本。

    ⚠ **一个前端默认值只有在「不传就是不传」时才真的位于链上的第四层。**
    桌面现在是 `knowledge=request.knowledge` **显式传下去**（`worker/main.py:276`），
    而 `TaskRequest.knowledge` 有默认 `'update'`——于是这个「前端默认值」实际进的是
    **第一层（等同用户在命令行敲了它）**，会**盖过**项目/全局配置。
    与本契约相反。要让它落在第四层，前端必须把「用户没选」传成 `None`，
    由后端 resolver 在配置之后再补上前端偏好。登记豁免表时要写清它当前在**哪一层**。
  - 「哪个层级说了话」必须可判定：中间层一律用 `None`/缺省表示「没说」，
    不要用哨兵值（`-1`、`""`）混进真实取值域。

  ⚠ **上面那条链是目标契约，不是现状。**（2026-08-31 复测，勿按理想状态读代码）

  | 差距 | 现状 |
  | --- | --- |
  | ~~后端两处真相源~~ | ✅ **已消除**（2026-08-31）。argparse 对每一个 `run_pipeline` 参数都传 `None`，`_defaults_from_args` 把 `None` 的键**整个丢掉**，于是签名成为唯一真相源。裸跑一次只写下 `stage` 一个键（它是「按 `--llm-correct-translate` 推导」的规则，不是值）。⚠ 走的是机制 ①（调度层剔除 `None`），不是 ②：真相源是**签名字面量**，不是 resolver |
  | 桌面是第三层 | `TaskRequest` 又一份默认；17 个共有字段里 16 项值一致、`knowledge` 是刻意分歧。**这一条没变**——桌面显式传值，仍从第一层进来 |
  | ~~行里的第三份~~ | ✅ **已消除**（2026-09-01，评审提出）。URL 分支要在 `run_pipeline`之前知道 `llm_media` 等值（它们决定下载什么），过去把默认值又写了一遍（`opts.get(...) or "audio"`，共 12 处）。现在统一走 `pipeline.opt(opts, key)`，回落读签名。⚠ 这类副本**不在原棘轮视野内**——它扫的是 argparse ↔ 签名，从不看行 |
| 守卫 | `test_option_defaults.py`：静态棘轮（`_ARGPARSE_CARRIES_A_COPY`，**现已清空**，新增一个副本即红）+ 行为扫描（每个未给的选项必须**不在**行里，而不是以 `None` 出现在行里）。⚠ 棘轮曾有一个盲区：它按同名相交，而 `--model`/`--gap`/`--separator-rate` 两侧拼写不同（`_ROW_ALIASES`），因此从未被比较过——其中两个一直带着重复默认值。2026-08-31 已让它跟随别名 |
  | 「项目 / 全局」两级配置**不存在逐键覆盖** | `finesub.paths.resolve_config_file` 取 `_checkout_data_root() or _packaged_user_data() or _managed_user_data()` 的**第一个命中**，整份用它。所以现在只有**一份**生效的 `config.toml`，不是两级合并 |
  | 配置层只对少数选项存在 | `pipeline.py` **自己完全不读 `config.toml`**；读配置的是各 stage 的 resolver（`resolve_split_params` 等）。所以链条中间那两层目前只对「有 resolver 的那几个选项」生效 |
  | 前端默认值不在第四层 | 见上一条 ⚠：桌面显式传值，实际落在第一层 |
- **一致性是一张证明，不是及格线——但这条有作用域。**

  **适用面：有意改变数值路径或实现形态的质量改动**（换 checkpoint、换采样率、
  编译/量化路径、组批、算子替换……）。这类改动**非逐位一致本身不是否决理由**：
  逐位一致只是一张低成本的无回归证明，拿到就免评估，**没拿到不代表变差**。
  所以「这个改动会让下游 VAD 边界不再逐段相等」是**定价**，不是否决。

  **但不适用于下面两类，它们仍按精确一致验收：**

  1. **声称「语义不变」的优化。** 一个改动如果宣称它不改变结果，就必须以
     **确定性字段逐位一致**为目标。例：救援重解复用 encoder 输出——同一份音频、
     同一个 group，`segments` 与确定性 metadata 应当逐位相同（timing、资源峰值这些
     运行观测字段除外，它们本来就会变）。这里的「不一致」是 bug，不是定价。
  2. **correctness contract**：确定性、幂等、序列化往返、resume/replay 复放、
     流式与整段等价。例：`split_params_for_length_scale(1.0)` 保证逐位不变、
     分句幂等、`session_replay` 的复放校验。这些的验收标准就是精确一致。

  ⚠ **「结果高质量且合理」不能单独充当验收标准。** 它是**目标**，不是判据。
  一旦放弃一致性证明，就**必须写明替代指标与门槛**
  （词准确率、人工时间轴对照、按响度分层的 SI-SDR、幻觉/复读率……），
  而不是事后说一句「听起来没问题」。**看完数字再判断好坏 = 没有验收。**

  **但闸门在「接受 / 翻默认值」这一步，不在「写代码」这一步。** 用
  [env 闸门 + keep-gated-not-revert](#开发原则) 把两件事拆开：新路径接线进去、
  **默认关闭**、A/B 用一个开关切——不翻默认就没有回归面，门槛可以并行准备。
  这样既不会出现「跑完再说服自己」的事后判断，也不会因为门槛没定就冻结实施。
  （反面参照：一个**零实测**的改动和一个**有 24 窗口对照**的改动不该走同一道闸门；
  本项目既有验收惯例见 `docs/wt-refine-validation.md`——真实语料上量、调阈值、
  报 FP/FN，并对没标定的地方老实写「尚未标定阈值」。）
- 生产入口应调用函数，不要用 subprocess 拼 CLI。
- LLM 后处理默认只生成计划和 prompt；真实生成 API 调用必须显式 opt-in，且默认测试不得联网或消耗 Gemini quota（窗口规划的 token 计数按 **本地 tokenizer 二进制 → 免费 `countTokens` 端点 → 启发式** 三级 fallback（`default_token_counter()`；本地二进制源码在 `tools/tokcount/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入依赖，离线且与 API 逐字一致，详见其 README；测试中一律注入 fake counter）。联网检索全部由本地检索代理（`finesub/llm/web_search.py`）执行，纠错/调查模型不直接启用 google_search 工具；provider 优先级、key pool、引导语等细节见 [`docs/llm_harness_routing.md`](docs/llm_harness_routing.md)。**例外**：`finesub.workflows.reference_ingest` 是用户主动发起的端到端工具，默认全执行（下载/GPU/LLM/知识库写入），`--dry-run` 才只打印计划。
- LLM 采样默认显式传 `temperature=1.0`；validation/parse retry 每失败一次下一次 logical attempt 降 `0.01` 并更换 `seed`，成功后的下一独立窗口/轮次恢复 attempt 0。`top_p` / `top_k` 不显式设置。
- 知识库更新走统一入口 `python -m finesub.llm.knowledge.update` / `run_knowledge_update`；三态开关 `--knowledge none|collect|update`、`--refined-srt` 精修对照模式、mistake 台账维护、`reference_ingest` 批量导入等完整行为见 [`docs/knowledge.md`](docs/knowledge.md)。
- URL 媒体下载逻辑在 `src/finesub/media/source.py`；主 pipeline 和 reference-ingest workflow 共享。URL→id 映射缓存在参考数据根下的 `url-map.json`，**抓来的标题另存同目录的 `url-info.json`**（url→`{title}`；分开放是因为前者是承重的——它决定产物路径与「重跑不上网」——后者只是个随时可删的便利缓存——**删它就是「重新抓一次」的显式动作**，因为「问过了，没有」也会被记进去，否则抓不到标题的 URL 每次重跑都要再探一次网。标题走 `extra_info`，**且只在运行会跑到纠错/翻译阶段时才抓**（`run_full_correction` 是它唯一的消费者），见 `docs/plans/crispasr-followups.md` P9）（`finesub.paths.resolve_reference_data_root()`：仓库形态是 `data/reference/`，装好的前端是 `user-data/reference/`——不能按当前工作目录解析，那在打包形态下是下次更新就被替换掉的源码快照）；下载的源视频与抽取音频放在对应 artifact 目录（pipeline 默认 `out/<video-id>/`，reference ingest 默认 `out/reference/<video-id>/`）。每窗音频剪辑在任务自己的 `<stem>.llm-artifacts/clips/`，随任务清理一并消失。

## 资源约束

**`*-stable.json` 之前（人声分离、VAD-ASR、ASR 稳定化等）**

- NVIDIA GPU，至少 4GB 显存，且**计算能力不低于装好的 torch wheel 的 kernel 列表下限**。判定
  不是硬编码的下限，而是拿 `torch.cuda.get_device_capability()` 去比
  `torch.cuda.get_arch_list()` 的最小值（见 `speech/runtime/device.py`）——torch pin 一动，
  支持范围自己跟着动。当前 cu128 构建为 `sm_70/75/80/86/90/100/120`，torch 自己的
  `CUDA_ARCHES_SUPPORTED` 也写着 cu128 = sm_70–120（cu126 = sm_50–90，Pascal 在 cu126 上是
  支持的）。用户可读的型号表在 README.md。
  ⚠️ **判定只看下限，不查是否在列表里，也不管上限**。cu128 列表里**没有 sm_89，而 RTX 40 系正是
  sm_89** —— cubin 在同一 major 内向前兼容，40 系用 sm_86 的 cubin 跑，所以「必须在列表里」会把
  整个 40 系踹到 CPU，比它要防的问题严重得多（这个 bug 在本分支上出现过一次）。高于上限同理放行：
  可能靠 PTX JIT 或同 major 兼容跑得起来，错误回退是静默 10× 变慢，错误尝试只是一个响亮的报错。
- 老卡为什么需要这个判定：torch 在架构不匹配时**只 `warnings.warn`**（`torch.cuda._check_capability`，
  经 `_lazy_call` 在 CUDA 初始化时才跑，那时 `is_available()` 已经返回 True 了），不会拒绝。所以
  驱动够新的老卡会一路开到第一次真算才炸 `no kernel image is available`。驱动太老的老卡则走另一条
  路：CUDA runtime 报设备数 0，`is_available()` 直接 False，现成分支就回退了（0.3.2 的一例 1060
  现场日志属于这条）。
- 至少 8GB 空余系统内存。
- CPU 回退只是兜底；发生回退时必须向 stderr 输出 `Warning:`。**回退有两个原因**：压根没有
  CUDA，以及有卡但这个 torch 构建没有它的 kernel（`is_available()` 为 True，老卡直到第一次
  真算才炸 `no kernel image is available`）。两者都由 `runtime/device.py` 的
  `resolve_device()` / `cuda_usable()` 统一判定——**不要在别处再写
  `torch.cuda.is_available()` 来决定活干在哪**，第三方库（audio-separator）自己那份判断要
  显式掰正，见 `separation.py` 的 `_build_separator`。
- CPU 回退上 ASR 按整机线程数并行，不需要旋钮。前提是 patched CT2 wheel 的 CPU GEMM 后端是
  **oneDNN**——`wtrefine1` 用的 Ruy 会在模型析构时死锁（0.3.2 的现场故障），中途试过的 MKL 正确
  但多吃 3.4 GB 内存且默认只在 Intel CPU 上启用。动 wheel 之前先读
  `tools/wt_refine_port/ct2-patches/README.md` 的后端对照表与最小验收：这三种失败都只在真解码
  时才暴露，`get_supported_compute_types("cpu")` 一律报 `float32`。

**LLM harness 阶段（自 stable.json 起）**

- 无需 GPU / 显存；约 4GB 系统内存。
- ffmpeg + ffprobe 在 PATH 上（窗口剪辑与时长探测）。
- 安装：`pip install -e ".[harness]"`（不含 torch / whisper 等 ASR 栈）。

档位按**显卡等级**命名，各带一个显存要求，要求的是**空闲**显存而非卡的容量（提示、违规阈值、裁判余量都用它）。卡的容量只在 `auto` 定档时出现一瞬：减去 `reserve_for_capacity()`（`容量/8 + 0.5`，在 4/8/12GB 上正好等于 1/1.5/2）后匹配。固定映射由最大窗口显存实测验证：

| 档位 | 需要空闲显存 | 内存要求 | WT 实例数 | Separator 实例数 | Separator BS | `auto` 选中它的卡 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cpu` | 不用显卡（`gpu=False`） | 8GB | 1 | 1 | 1 | **机器上没有 CUDA 设备** |
| `entry` | 3GiB | 8GB | 1 | 1 | 1 | <8GB；**有卡但这个 build 用不了也落这里**（留给 CT2，见 `device.py`） |
| `standard` | 6.5GiB | 8GB | 1 | 2 | 1 | 8–11GB |
| `standard_large_vram` | 10GiB | 8GB | 1 | **2**（与 `standard` 相同） | 1 | 12–23GB |
| `high` | 10GiB | 8GB | 1 | 3 | 1 | ≥24GB |

⚠ **`standard_large_vram` 的 worker 数与 `standard` 相同，多的是显存预算**——两个数回答的是
两个问题：worker 数是吞吐取舍（E7 实测两个是峰值），显存是**裁判花的预算**（编译解码步要
Whisper 池之外 3.5 GiB，`COMPILE_MIN_VRAM_GIB`）。旧表把两者绑死，12–16GB 卡只能在
「两个 worker 但预算 6.5」和「预算 10 但第三个更慢的 worker」之间二选一。
⚠ **`cpu` 是策略档不是能力档**：它说「这次不用显卡」，与「问过说不行」是两件事——
`--gpu-tier cpu --device cuda` 直接报错。用户向说明在 `docs/manual/resources.md`。

`auto` 在 `standard_large_vram` 处封顶（`AUTO_TIER_CEILING`，2026-09-02 从 `standard` 上调）：`high` 的第三个分离 worker 实测比第二个慢（E7），所以「卡更大」不该换来「跑更慢」，但多出来的显存该给裁判用。≥24GB 才放行，`high` 始终可手填。

16GB / 4 实例的 `max` 档在 0.5.0 移除：E7 的 worker 阶梯把吞吐峰值定在两个 worker，3/4 个反而更慢，而 4 worker 只在音频 ≥15 分钟时才够得着。见 `docs/gpu-profiles.md`。

默认档位是 `auto`：读整卡显存、先四舍五入到整 GiB（驱动报的从不是标称值，16GB 卡实测报 15.92）再向下取档，**没有可用 CUDA 设备时落 `cpu`**（2026-09-02 起；此前是 `entry`，那让一台没有显卡的机器在记录里写着 3GiB 的显存预算）。分离器实例数按档位是 1/1/2/2/3，**WT 不随档位变——ASR 恒定单 worker**
（2026-08-02 起，见本节开头的警示）。下面那条 `large-v3-turbo` 1/2/3/4 实例的曲线
是 WT 曾按档位缩放时的记录，**只作历史依据**，不再描述现行映射：
1/2/3/4 实例实测峰值分别为 2.17/4.29/6.01/8.25GiB。本机 3 实例吞吐最好只作为硬件特例记录。人声分离的
BS-Roformer 在当前 `audio-separator` 实现里不消费 `batch_size`；
620 秒最大读窗实测 bs=1/2/4 都是 2.86GiB、耗时差不超过 0.4 秒，因此所有档位取
语义最明确的 bs=1。并发分离任务共享同一个 `model_run`，wrapper 状态各自独立；
2 实例吞吐为单实例的 1.121×、显存 4.06GiB，3 实例为 1.109×、5.22GiB，
且输出 hash 均与单实例一致。完整环境、窗口口径、逐点数据及结论见
[`docs/gpu-profiles.md`](docs/gpu-profiles.md)。
**ASR 固定单 worker**（2026-08-02 移除单文件分片）。当初的分片实现只换到 1.1–1.2× 端到端加速，却带来 interval ownership、shard partial、跨 shard 语言历史分歧等一批复杂度，而显存随实例数线性增长；换到 fw-refine 后 ASR 已不再是瓶颈（人声分离占语音段 72%），这笔交易更不划算。
完整设计、标定数据与踩过的坑见 [`docs/wt-parallelism.md`](docs/wt-parallelism.md)，实现回溯点是 `dev` 的 `1fcc4e1`。吞吐要再优化的话方向是**单 worker + 批解码**，见 [`docs/wt-refine-port.md`](docs/wt-refine-port.md)。

单项管线在同一进程内顺序调用各阶段（无 subprocess）。进程级 GPU model-family
gate 允许多个 separator 或多个 WT 同类任务并行，但不会让两个模型族跨任务同时驻留。
并发分离只共享 Roformer `model_run`（权重和预热后的模型级缓存），每个音频块仍有
独立 `Separator` / `model_instance` wrapper、输入输出路径和 source cache；空输出重试
也重新取得干净 wrapper。不同的 600 秒 core + pad 块并行完成后由主线程按序裁剪拼接，
实际 worker 数不超过文件块数，单块短音频不创建线程池；全局加权限流防止多个 batch
task 把实例数相乘。非 CUDA 后端保持顺序独立模型。

ASR 对齐阶段不再把整段音频读进内存，而是用 `AudioBlockLoader`（600s 核 + 10s pad）
按块从磁盘流式取音频，RAM 不再随时长在 Whisper 之外线性增长；此路径与独立 CLI
`finesub.speech.recognition.transcribe.main` 一致，输出逐字节等价（见
`test_pipeline_refactor.py` 的等价性守卫测试）。
4GB 默认档位的实测峰值由 `test/test_resource_budget_pipeline.py` 守护
（`heavy_resource`，默认 skip，需 `--run-heavy-resource`）。

**VAD 也是流式的**（`finesub.speech.preprocessing.energy.run_vad_file` /
`_streamed_frame_tracks`，600s 核 + 90s
context）：帧局部量（帧 dBFS、band power）按块算，全部有状态/全局步骤（DC 均值、RMS 窗口和、
峰值限幅、自适应谱追踪器、噪声地板、打分）在确定性的分块归约上全局跑一次，输出与整段载入路径
**逐位一致**（任意时长；守卫测试 `test_vad_streaming.py`，含重采样栅格与限幅触发用例）、RAM
上界为一个核块 + 整段小能量轨（8h 音频 ≈ 0.3GB）。为此 DC 均值与 RMS 窗口和的归约顺序被定义为
固定网格上的 float64 分块求和（旧实现是整段 float32 `torch.mean`/`cumsum`）——相对旧版产物有
浮点 ulp 级差异（长音频下新实现更精确），远低于任何可感知阈值。

VAD 非语音打分（`_score_to_non_speech_intervals`）是全 VAD 里唯一没走 numba 的逐帧循环：
循环前把能量/噪声/时间张量一次性转成 numpy、循环内索引 numpy 标量而非 `tensor.item()`，实测
303s 音频 13× 提速且分段逐位不变（守卫测试见 `test_intervals.py`）。

## 运行时路径解析契约

（从 2026-07 的 package 重整计划提取——那份计划已迁 `docs/archive/`，但下面这条规则至今
成立，且此前没有任何被跟踪文档收录它。）

**`src/finesub/paths.py` 是唯一的仓库/运行时路径 resolver。** 每个资源的解析顺序固定：

```text
函数显式参数 → 对应 FINESUB_* 环境变量 → 源码 checkout 标记发现 → 该资源的安全 fallback 或明确报错
```

| 资源 | 环境变量 | 找不到时 |
| --- | --- | --- |
| `.env` | `FINESUB_ENV_FILE` | 只用进程环境变量 |
| `config.toml` | `FINESUB_CONFIG_FILE` | 用 provider/pool 默认值 |
| `.state/` | `FINESUB_STATE_DIR` | 稳定的用户 state 目录（Windows `LOCALAPPDATA`、Unix `XDG_STATE_HOME`、最后 `~/.finesub/state`）；不为此引入 `platformdirs` |
| 本地 tokenizer | `GEMINI_TOKEN_COUNTER_EXE` | checkout 的 `bin/`，再退到免费 `countTokens` 端点（名字与候选路径由 `finesub_bootstrap/token_counter.py` 单点定义） |
| knowledge root | `FINESUB_KNOWLEDGE_ROOT` | **实际启用 knowledge 时**才明确报错；禁止 import 期报错或静默新建 |

**`src/` 里除 `paths.py` 外不得出现用于定位仓库根的 `Path(__file__).parents[N]`。** 层数是
隐式契约：模块一搬就静默指错，且不会有任何东西报错——2026-08 把 `llm` 移进 `finesub` 时，
一处 `parents[2]` 正是这样从仓库根变成了 `src/`，而它所在的测试标着
`requires_main_checkout`、在 worktree 里从头到尾没跑过。`test_import_boundaries` 守着这条。

一处**与该契约的现存偏离**，记录在案而不是假装不存在：`model_catalog.psv` 与
`model_routes.toml` 用 `Path(__file__).with_name()` 读，不走 `importlib.resources`。它定位的
是包内同级文件而非仓库根，在 wheel 里照样能用（两者都在 package-data 里），只有 zip import
会坏——目前没有这种装法。

## 产物清单与路径

缺省输出路径为 `out/<stem>/<stem>.srt`（`default_output_path`，不传 `-o` 时），一次运行的全部 artifact 都从最终 SRT 路径推导、归到 `out/<stem>/` 一个目录；URL 输入使用 `video-id` 作为 stem，并把下载/抽取媒体放在同一 artifact 目录；显式传 `-o` 时按该路径同级推导、不加子目录。以 stem=`input`、跑到 `final-srt` 为例：

> 这一节讲的是**管线自己**（`python -m finesub.pipeline` / 仓库开发版）。推导规则对前端完全相同，区别只在
> 谁来定 `-o`：桌面总是把运行放进 `tasks/<task-id>/`；`finesub` 只在用户**没给** `-o` 时补一个，
> 给了就原样透传、运行就发生在用户的目录里（产物可见地增长，失败也留在那儿）。
> 跑完的处置是前端的事，管线自己从不删东西、也不记录任务：桌面会按
> `finesub_bootstrap/artifacts.py` 清理自己的任务目录，`finesub` 不动用户目录、只把
> `RECORD_SUFFIXES` 那两个文件抄进任务目录；任务记录见 `task_index.py`。

```text
out/input/
├── input-vocal.ogg              # 人声分离 (vocal_separation)：16 kHz 单声道 Vorbis
│                                #   `--no-separate` 时同规格、同路径，由源音转码而来；
│                                #   分别在 metadata 里记为 executed / skipped
├── input-vad.json               # VAD 阶段产物：interval + vad_meta + timing（契约见 docs/vad-asr.md）
├── input-vad-energy.npz         # 上一行的帧级能量轨（几十万帧，故不进 JSON）
├── input-aligned.json           # VAD 能量分段 + Whisper 对齐原始结果 (vad_asr)
├── input-aligned.partial.json   # ASR 断点续跑缓存；仅在 VAD-ASR 未跑完时存在，成功后删除
├── input-stable.json            # ASR 稳定化结果 (asr_stabilize)
├── input-raw.srt                # stable.json 原文；按最终 profile 的时间轴步骤延长短轴（不改文字）
├── input-translated.srt         # LLM 纠错+翻译中文字幕（未后处理）
├── input-corrected.srt          # 纠错后「原文」SRT
├── input-annotated.csv          # 9 列完整标注 CSV：type|position|duration|gap|corrected|translation|conf|char_count|note
├── input.srt                    # 最终 SRT（translated 后处理后）
├── input-metadata.json          # pipeline 元数据：核心阶段耗时、worker、LLM logical-round 耗时
└── input.llm-artifacts/         # task artifact 目录（默认 = 输出去后缀 + .llm-artifacts）
    ├── input-research-context.json  # 背景调查结果(research + context_pack)，存在即跳过研究轮
    ├── input-research-context-cNN.json  # 超长素材分块调查的逐块产物（按块输入哈希重放；见 docs/llm_harness_research.md）
    ├── task-artifacts.jsonl     #   结构化事件流：research_*/search_loop_round/correction_*
    │                            #   /content_filter_{ladder,blacklist}/token_distribution_report/final_srt …
    ├── session-checkpoints.jsonl #   已验证 LLM session 输出：research/query/search-judge/fast 的细粒度 resume
    ├── correction-window-plan.json # 只存 source-id 边界；clip/预算/前文均为派生态，恢复时按当前包络重算
    ├── correction-windows.jsonl #   纠错窗口 resume 缓存：每个成功窗口一行，供中途 resume 复用
    ├── exchanges/               #   每次 LLM API 交互一个 markdown（含 API call trace / reasoning 摘录）
    ├── task-report.md           #   运行时汇总（API 计数、token、fallback/warning 等；注入/thinking 见 docs/llm_harness_behavior.md）
    └── knowledge-update-{chunks.jsonl,harness-notes-NN.md}  # 仅 --knowledge update：apply ledger / 精修模式 harness notes
```

**人声分离的交付分两个模式，由输出后缀决定**（`separation.output_mode_for`）：`.ogg` 是
管线用的 ASR 交付——**16 kHz 单声道** Vorbis（`compression_level=0.2`），因为读它的每一个
下游（energy VAD、whisper、Qwen 裁判）第一件事都是下混加重采样到这个规格，交付 44.1 kHz
立体声等于把四分之三的码率花在下一阶段会扔掉的样本上；`.flac` 是无损交付，保持模型自己的
采样率与声道数，供试听、测量和实验。其它后缀直接报错。两者都由同一条无损合并链产出，ASR
交付只是在合并完成后多一次「下混 → 重采样 → 编码」的流式扫描。别的模块（`tools/`、人工
排查）想要整轨无损，直接调 `python -m finesub.speech.preprocessing.separator.separation -o x.flac`。

产物树里没写、但可能出现的一个文件：**`<源媒体 stem>-decoded.flac`**。源媒体一律原样
交给各阶段——本地视频和 URL 视频都不再预先抽音轨——只有 soundfile 打不开容器时（视频、
部分容器）才由分离阶段转一份**无损**音频（保持采样率与声道数，只换容器）；成功的阶段跑完当场删掉它，所以
它只在**跑挂了的任务**里留下来。名字随源媒体而不是随字幕，`-o` 改名或 URL 输入时两个
stem 并不相同，因而**推不出来**——所以解码一成功就把实际路径记进 `*-metadata.json` 的
`scratch_files`，`cleanup_intermediate` 读它来删（`REMOVABLE_SUFFIXES` 里的
`-decoded.flac` 只兜同 stem 那一种）。

`*-metadata.json` 与其他任务产物同级，不依赖 LLM stage。写入是「临时文件 + `os.replace`」；
Windows 上目标被别的进程打开时 replace 会 `PermissionError`，`update_run_metadata` 按
`REPLACE_RETRY_DELAYS_SEC` 重试约 2.4s（杀软扫描、索引器这类瞬时占用足够恢复），仍失败则抛
`RunMetadataLocked`，错误里列出可能的持锁者（另一个 finesub 实例 / 杀软或索引器 / 编辑器 /
无写权限）。不吞错：sidecar 持久写不进去多半意味着整个输出目录有问题。只追踪下载、人声分离、
VAD-ASR、LLM harness 四个有分析价值的大阶段及 pipeline 总耗时；ASR stabilization、
SRT 导出/后处理和普通文件 I/O 不单列，其耗时自然包含在总耗时中。worker 字段区分
batch pool、人声分离的 profile limit/effective workers，以及单文件 WT 的
requested/profile limit/effective workers。LLM logical round 聚合该轮全部 endpoint
fallback、失败 attempt 和 validation/format retry；逐 provider attempt 明细仍只在
`exchanges/` 保存。

`input-aligned.json` 的每个 ASR segment 保留 Whisper 来源段的 `confidence` 和
`no_speech_prob`，每个 `words[]` 项保留 Whisper 的 `confidence`。VAD 的旧段级
`conf` / `vad_conf` 不进入 aligned schema。segment 与 Whisper 来源段一一对应（不再在
VAD interval 边界切开）；异常重复清洗合成 word 时取各来源 word
confidence 的最小值。`python -m finesub.speech.recognition.cli.vad_asr` 另按最终 segment 边界写入
`vad_weighted_energy_db`；定义与 metadata 见 [`docs/vad-asr.md`](docs/vad-asr.md)。
旧产物或上游未返回相应指标时字段可缺省。stable 默认经 profile 0 清理/标记；完整
profiles、`tags` 与指标定义见 [`docs/asr-stabilize.md`](docs/asr-stabilize.md)。

不在该目录下的：URL→id 映射与抓来的标题在参考数据根（仓库形态 `data/reference/url-map.json` 与 `url-info.json`，装好的前端在 `user-data/reference/`）；窗口媒体剪辑在 `<stem>.llm-artifacts/clips/<chunk_id>.aac`（`--llm-media video` 的纠错轮另有 `<chunk_id>.mp4`）——它在 artifact 目录里面，所以是随任务整删的；`--knowledge update` 时知识库写入 `knowledge/`（独立内嵌 git 仓库，自动提交，非主仓库跟踪）。批量运行（`python -m finesub.pipeline` 给多个输入或 `--manifest`、reference-ingest 多任务）另在 `out/batch/<batch-id>/` 下放四样东西：`batch-status.jsonl`（事件流，每行 `{item,label,stage,status,error?,ts}`）、`queue.jsonl`（运行器发布的现状，行本身是合法 manifest 行，带 `_state`/`_stage`）、`control.jsonl`（用户只追加的控制面）与 `.control-cursor`（控制面消费到哪，指令生效后才推进）；运行期间还持有 `.batch.lock`（判活与互斥，进程怎么死都由 OS 释放）。批次指针记在数据根的 `batches.json`（键是 `(cwd, batch_id)`，`--resume-batch` 唯一查的地方，不存任何任务）。每项的产物位置不变，仍归各自 `out/<stem>/` 或 `out/reference/<id>/`，重跑同一批即按上面的存在性跳过规则续跑；契约细节见 [`docs/manual/batch.md`](docs/manual/batch.md)（用户向）；dev 侧这四件加注册表的实现与取舍在 `src/finesub/batch_state.py` 的模块 docstring（2026-08-31 从 `pipeline.py` 拆出），[`docs/batch-scheduler.md`](docs/batch-scheduler.md) 讲的是三 bin 的并发语义、不是这几个文件。独立实验 CLI `finesub.llm.correction_translation --prompt-dir <dir>`（默认 dry-run）另把 `plan.json`/`research-round{1,2}.txt`/`correction-NNNN[-query].txt` 写到 `--prompt-dir`，与生产 pipeline 的产物集不同。

## Pipeline 复用规则

`src/finesub/pipeline.py` 每一步都会检查默认输出是否存在：

- `*-vocal.ogg`（管线交付）或 `*-vocal.flac`（无损交付）任一存在则跳过人声分离。判定走
  `PipelinePaths.resolve_vocal_audio()`，与下游读取用的是同一个解析——只认 `.ogg` 会让手上
  已有无损轨的运行白跑一遍最贵的 GPU 阶段。
  `--no-separate`（输入已是纯人声）跳过的是**分离本身**，不是这份产物：同规格的
  `-vocal.ogg` 照样落在同一路径，由源音转码而来（`separation.encode_asr_delivery`）。
  所以复用判定、resume 与每一个读它的下游都不需要「没有人声轨」这个分支；
  区别只记在 `run-metadata.json` 的 `timing.stages.vocal_separation.status`
  （`executed` / `skipped` / `reused`，第三个才表示这一趟什么都没跑）。
- `*-vad.json`（+ 同名 `-vad-energy.npz`）存在、且**音频身份**对得上，就跳过 VAD 前缀直接
  进 Whisper。`--vad-silero-assist` **不参与这个判定**：它记在产物的 provenance 里，
  不匹配只 warning、照常复用。对不上则重算并覆写——判据与失效理由见
  [`docs/vad-asr.md`](docs/vad-asr.md)「VAD 阶段产物」。删 `*-aligned.json` 重跑识别/分句/
  复核时，这一段不会陪跑。
- `*-aligned.json` 存在则跳过 VAD-ASR；stable 缺失时可直接从 aligned 运行 ASR 稳定化。
- **ASR 断点续跑**（`finesub.speech.recognition.transcribe.align_segments`，
  长音频崩溃后不必从头再来）：每处理完一个
  alignment group 就原子写 `*-aligned.partial.json`，内含已完成 segments、区间游标、
  `prev_tail_segments` 与 auto-language history——即 group 边界上的完整状态。重启时按指纹
  （model / language / gap_sec / 音频路径+大小+mtime / 区间摘要 / `ASR_CHECKPOINT_VERSION`）
  校验，一致才续跑，否则整份丢弃重跑。checkpoint 的身份、读写与清理由
  `src/finesub/speech/recognition/checkpoint.py` 统一负责。当前 schema 为 v2；v1/缺版本的旧 partial
  不迁移，直接从头重跑。partial 只是缓存不是产物：损坏或过期一律当作不存在，
  文件名与 `*-aligned.json` 区分开，不会被"存在即跳过"误判；跑完即删除。
- **对齐降级**：一遍式路径要把词分组与解码轨迹一一配对，退化音频（长幻觉重复串撞上解码上限）
  会打破这个前提。`_transcribe_with_teacher_force_fallback` 捕获后以 teacher-force 对齐重试同一
  group（该路径不做这种配对）；再失败则该 group 按静音丢弃并打 `Warning:`，保证长音频不会因
  单个 group 崩掉。正常 group 不受影响。
- `*-stable.json` 存在则跳过 ASR 稳定化及其上游；不会为了补档而重新生成缺失的 aligned。
- **特殊**：显式目标为 `aligned` 时，stable 不能代替 aligned；aligned 缺失仍会运行人声分离和 VAD-ASR。
- `*-raw.srt` 存在则跳过 raw SRT 导出。
- LLM stage 会复用 artifact 目录内的 `*-research-context.json`（兼容旧的 run 根目录位置）、`*-translated.srt`、最终 `*.srt` 和 task artifact 目录；如果 translated 已存在但 final 不存在，只跑 SRT 后处理。
- **LLM session resume**（默认开启，需 task artifact 目录）：research R1/R2、每轮 search judge、fast round 1 和逐窗 query 的 parser 验证成功输出写入 `<artifact_dir>/session-checkpoints.jsonl`。重启后 harness 先重建本地确定性状态（搜索/网页提取、媒体剪辑和上传允许重做），到同一 LLM 边界时按“精确 messages + PROMPT_VERSION + 调用配置 + 媒体/任务身份”命中旧响应，并用当前 parser 再验证后复用；输入或契约变化自动失效。research 的完整 `*-research-context.json` 仍可整阶段复用。
- **纠错窗口中途 resume**：每个成功窗口另写 `<artifact_dir>/correction-windows.jsonl`；命中后整窗回放，连该窗 query、搜索、剪辑上传和纠错调用都跳过，从第一个未完成窗口继续 live。缓存按 task fingerprint + 每窗 input_hash 匹配；持久化窗口计划只复用 source-id 边界，恢复时按当前媒体/profile 重算 clip 与预算，pending 叶超出当前输入/输出/质量上限会在发车前递归拆半并写 `window_refit_report`。research 窗口笔记按 source-id 区间重映射，不要求新旧几何一致。`--no-resume` 同时关闭两种 resume ledger 的读写，但不删除已有文件，也不改变 pipeline 对完整 stage 输出文件的存在性复用。

stage 级跳过当前只检查文件存在，不校验内容和参数一致性（research planning metadata 与两种 resume 缓存例外，带 fingerprint/input_hash 校验）。后续如果增强，应优先加：

- FLAC 可读性检查。
- JSON schema / metadata 检查。
- 输出参数 fingerprint。
- 参数不一致时 warning 或强制重跑选项。

### 复用的依据是任务身份，不是质量判断

**Resume 延续的是现有任务的产物与历史选择；当前参数不追溯重定义已经完成的阶段。**

- resume 前后的参数**都是用户主动的选择**。新参数只作用于**尚未产出**的部分。
- 需要一整套全新的参数快照时，**用户应当新建一个任务**，而不是指望 resume 悄悄把
  旧阶段按新参数重做。
- **系统不因参数不同就自动推翻用户选择 resume 的那份任务状态。**
  产物失效只有三个理由：**损坏**、**身份不匹配**、**契约无法读取**。
- ⚠ **但依赖的「形状」变了，就得重跑被依赖的那一段。**
  形状变的判据是 owner 的原话：**直接 resume 会出错，或者某些数据拿不到**——
  结构不兼容、游标指向一个不再存在的东西、契约版本变了解析不了。
  **仅仅是「参数不同、假如重算内容会不一样」不算形状变。**

  | 情形 | 形状变了吗 | 处置 |
  | --- | --- | --- |
  | ASR partial 的区间游标 vs 新的 `gap_sec`/区间摘要 | ✅ 变了——游标指进的是**另一张区间表**，续跑会错位 | 丢弃 partial 重跑 |
  | checkpoint schema v1 → v2 | ✅ 变了——**读不了** | 丢弃 |
  | LLM 缓存响应 vs 新的 `PROMPT_VERSION` / 契约 | ✅ 变了——现 parser **验不过** | 失效 |
  | `*-vad.json` 是在 assist 关闭时产出的 | ❌ 没变——文件完整、下游照常消费得了 | **复用 + warning** |
  | 上游产物被重新生成，而下游那份**完整且仍可消费** | ❌ 没变 | **复用**——不要级联 |
  | 上游重新生成后，**挂在它身上的进行中缓存**（ASR partial 的音频身份 / 区间摘要）对不上 | ✅ 变了——游标与 prev_tail 是对着旧音频算的 | 丢弃那份缓存 |

  ⚠ **没有独立的「级联重跑」规则。** 上游被删/重算**本身**不使下游失效——
  管线是按需驱动的：下游产物存在就跳过，上游根本不会被要求重算。
  真正会被级联打断的只有**挂在具体上游身份上的进行中缓存**，而那已经是形状变了。

- **两种指纹要分开，别混成一个。**

  | | 用途 | 不匹配时 |
  | --- | --- | --- |
  | **provenance fingerprint** | 记录这份产物是用什么参数产出的 | **warning**，照常复用 |
  | **compatibility key** | 只放「不匹配就会报错或数据拿不到」的东西（schema 版本、游标所依赖的区间摘要、音频身份、契约版本） | **失效**，丢弃重算 |

  运行参数（`--vad-silero-assist`、批大小、`compute_type` 这类）默认属于**前者**。
  只有当它确实会让续跑报错或取不到数据时，才有资格进后者——而且要在代码里写明理由。
  **产物应当记录自己是用什么参数产出的**，这样参数对不上可以如实 warning，
  不必在「静默复用一份会误导人的产物」和「替用户推翻他的 resume」之间二选一。

所以上面四条增强项按这个口径分两类，**不要一律加**：

| 增强项 | 判定 |
| --- | --- |
| FLAC 可读性检查 | ✅ 值得——冲的是**损坏** |
| JSON schema / metadata 检查 | ✅ 值得——冲的是**契约无法读取** |
| 输出参数 fingerprint | ✅ 值得——但**作为 provenance** 记录并 warning；只有「不匹配就报错/取不到数据」的字段才升级成 compatibility key |
| 参数不一致时 warning 或强制重跑 | ⚠ **warning 可以**；**自动强制重跑不行**——那正是「系统替用户推翻他选择的 resume」 |

✅ **这条曾经的差距已经消除**（2026-08-29 落地）：

`*-vad.json` 前缀的复用键一度**含 `vad_silero_assist`**，翻一次开关就会把旧前缀判成
不匹配并重算覆写。现在它记在产物的 `provenance` 块里：**不匹配只 warning，产物照常
复用**；复用键只剩音频身份（文件名 + 字节数 + mtime）。要全量按新参数生成，新建任务。

判据仍然是那条：它不匹配既不会让续跑报错、也不会让数据拿不到，所以它是
**provenance fingerprint，不是 compatibility key**。

⚠ 迁移时的一个陷阱值得记下来：读取端比较 `source` 必须**按子集比**，不能整字典相等——
后者会让每一份升级前写下的 prefix 在首次运行时因「旧 4 键 vs 新 3 键」被作废，
正好是这次改动要消除的行为。`test_vad_prefix_*` 里有一条契约测试钉着它。

---

这与 [开发原则](#开发原则)「一致性是一张证明，不是及格线」是两条不同的规则，别混：
那条讲的是**新改动怎么验收**，这条讲的是**已有产物为什么复用**。

## 测试规则

默认全量（2 worker 并行），**约 1.5 分钟**：

```powershell
python -m pytest -q
```

默认单测不得加载 Whisper/audio-separator 模型、处理大音频、显著占用显存，或联网/消耗 Gemini quota。重资源测试标记 `@pytest.mark.heavy_resource`，只有显式传 `--run-heavy-resource` 才运行，且只有用户明确要求时才应主动跑。

全量既然是分钟级，**日常不必按域挑着跑**：改文件时跑对应单文件，准备提交时跑全量。
若你看到的全量是几十分钟，先确认**仓库根的 `conftest.py`** 还在——它在无法创建符号链接的
机器上关掉 pytest 的 `<prefix>current` 便利链接，少了它每个用例要多付两到三次
0.2 秒的必然失败。原委、写法约定与域标记的覆盖缺口见 [`docs/testing.md`](docs/testing.md)。

常规验证：

```powershell
python -m compileall -q src test
python -m pytest -q
python -m finesub.pipeline --help
python -m finesub.speech.recognition.cli.vad_asr --help
python -m finesub.speech.preprocessing.separator.separation --help
python -m finesub.subtitles.rendering --help
python -m finesub.llm.correction_translation --help
python -m finesub.llm.knowledge.update --help
```

## Agent 工作清单

开始改动前：

```powershell
git status --short
rg --files
```

改动入口或依赖时，同时检查：

```powershell
pyproject.toml
README.md
README_DEV.md
test/
```

改动 pipeline 行为时，至少更新：

```text
test/test_pipeline_refactor.py
README.md
README_DEV.md
```

提交前：

```powershell
python -m compileall -q src test
python -m pytest -q
git status --short
```

不要自动运行完整音频 pipeline 或 heavy-resource 测试，除非用户明确要求。

## 已知改进方向

**Pipeline / ASR**

- 为 pipeline 的已有输出复用增加完整性和参数一致性校验。
- 将资源上限从运行后 warning 升级为可选强失败。
- 把档位标定扩展到更多 GPU 型号；当前五档映射的最大窗口实测只在 RTX 5060 Ti 上做过，2026-09-01 在 RTX 5070 Ti 上复测过预算（见 `docs/gpu-profiles.md`）。尤其缺 ≥24GB 卡的 worker sweep——`auto` 在那里放行 `high`，而现有实测说三个 worker 更慢。
- 继续拆分 `src/finesub/speech/recognition/transcribe.py` 和
  `src/finesub/speech/preprocessing/energy.py` 的算法、I/O、CLI 边界。
- `segmentation-split.md` 待细化：跨语料泛化、合成词切点罚、纯虚构词归属、recall 救援交互、beam 信任折扣（5 项评审点）。

**LLM harness**

- 滑动窗（RPM/TPM）仍为进程内内存态；跨进程持久化为后续项（`docs/llm_harness_routing.md`）。
- Shared Context（跨窗口共享上下文）暂不做，触发条件见 `docs/llm_design_notes.md`。
- `tools/session_replay/run.py` argparse help 仍写 "basic->basicA"，应为 "basic->basicB"。

**知识库**

- 知识库真相源已是 SQLite（`docs/knowledge.md`「存储形态」）；共享/在线托管按 `docs/plans/knowledge-node-plan.md` §6。
- 精选维护任务、翻译风格注入统一机制、子词条拆分自动化——见 `docs/knowledge.md` 遗留开放项。
- `docs/plans/knowledge-node-plan.md`：node 模型、检索分级、三层信号与共享库——§8 与二次设计 §11 均已落地（2026-08-28）；原打分方案已归档。

**文档**

- 继续完善 `docs/llm_design_notes.md` 中的 LLM 纠错、翻译、知识库和 prompt/harness 自我迭代后处理层。
- 文档迁入本地 `docs/archive/` / `docs/report/` 前：提取非过时有用信息到仍在追踪的 docs（见 `CLAUDE.md` Archive extraction）。
