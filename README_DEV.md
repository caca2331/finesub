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

- `src/finesub/pipeline.py`：生产编排入口，负责中间文件路径、跳过已有输出、stage-based resume。默认跑到 `raw-srt`；`translated-srt` / `final-srt` 才进入 LLM 纠错翻译和 SRT 后处理。
- `src/finesub/batch.py`：download / ASR / LLM 三 bin 批处理引擎。
- `src/finesub/speech/preprocessing/separator/separation.py`：人声分离，使用 `audio-separator`。
- `src/finesub/speech/preprocessing/vad.py`：流式 VAD 检测与能量轨迹。
- `src/finesub/speech/recognition/vad_asr_stage.py`：组合 VAD 与 Whisper recognition，输出未稳定化的 `*-aligned.json`。
- `src/finesub/speech/recognition/transcribe.py`：recognition service（单 worker——单文件分片已于 2026-08-02 移除，见 `docs/wt-parallelism.md`）；仍包含待拆分的 windows、decoder、timestamp mapping 和 recovery。
- `src/finesub/speech/recognition/checkpoint.py`：ASR partial identity、schema、原子写入与清理。
- `src/finesub/speech/recognition/segments.py`：识别输出的重叠收回、零时长修复与空段过滤。
- `src/finesub/speech/postprocessing/stabilization.py`：独立 ASR 稳定化 stage，按 profile 从 aligned 生成 stable。
- `src/finesub/subtitles/rendering.py`：stable JSON 转 SRT。
- `src/finesub/speech/recognition/fw_refine_backend.py`：patched CT2 适配层、模型池与批解码 driver。
- `src/finesub/speech/runtime/resources.py`：4/8/12/16GB 显存档位、1GB 系统预留、WT/Separator 实例数与资源上限检查。
- `src/finesub/speech/preprocessing/energy.py`：VAD-energy 核心算法，体积较大，修改需谨慎。
- `src/finesub/speech/preprocessing/audio.py` / `spectral.py`：前者只管「把音频从磁盘取成
  波形」（解码、切片、重采样、并声道），后者是加权频谱能量本身（滤波器组 + numba/torch 帧循环）。
  **`spectral.py` 不是 VAD 私有的**：`recognition/transcribe.py` 也用 `weighted_spectral_energy_db`
  做逐段能量，这正是它没有并进 `energy.py` 的原因。
- `src/finesub/speech/preprocessing/separator/`：分离器自成一包——`separation.py`（阶段本体）、
  `accel.py`（选档与编译缓存）、`separator_aoti.py`（AOTI 包的构建与加载）。三者只互相说话。
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
  - `knowledge/`：知识库子包——`base.py`（原 `knowledge_base.py`）、`update.py`（统一知识更新入口，原 `knowledge_update.py`，CLI 现为 `python -m finesub.llm.knowledge.update`）、`mistakes.py`（原 `common_mistakes.py`）、`entries.py`、`feedback.py`、`materials.py`。
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
  版本取 `desktop/VERSION`；用法见 `cli/README.md`。
- `tools/`：独立开发工具，全部**按需维护**——不随主程序改动自动更新，测试不进默认套件，
  具体规则见各自 README：`session_replay/`（冻结注入重打 session，`python -m
  tools.session_replay`）、`asr-confidence-explorer/`（手工分析快照）。

## 分步调试

生产时优先使用 `python -m finesub.pipeline`。需要定位问题时，可以分步运行。

1. 人声分离：

```powershell
python -m finesub.speech.preprocessing.separator.separation data/input.wav -o out/input-vocal.flac --gpu-budget-gb 8
```

CUDA 分离固定启用 AMP；共享模型预热使用相同精度，避免产生一次额外的 FP32 activation
峰值。FP32 只由开发基准工具生成对照，不作为生产 CLI 开关。标定与否决实验见
[`docs/separator-optimization.md`](docs/separator-optimization.md)。

2. VAD + ASR 对齐：

```powershell
python -m finesub.speech.recognition.cli.vad_asr out/input-vocal.flac --output out/input-aligned.json --model large-v3-turbo --language en --gpu-budget-gb 8
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
随素材开盲盒，最差一例丢掉 476 个有声秒里的 98 个。全部依据见
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
`<key>` 由 torch 版本、CUDA、GPU 架构和
checkpoint 组成——**换任意一项即换目录，这就是失效机制**，不需要额外的比对代码。
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
- 默认不要改 VAD/ASR 参数。若必须改，说明对输出一致性的影响，并补测试或实验记录。
- 生产入口应调用函数，不要用 subprocess 拼 CLI。
- LLM 后处理默认只生成计划和 prompt；真实生成 API 调用必须显式 opt-in，且默认测试不得联网或消耗 Gemini quota（窗口规划的 token 计数按 **本地 tokenizer 二进制 → 免费 `countTokens` 端点 → 启发式** 三级 fallback（`default_token_counter()`；本地二进制源码在 `tools/tokcount/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入依赖，离线且与 API 逐字一致，详见其 README；测试中一律注入 fake counter）。联网检索全部由本地检索代理（`finesub/llm/web_search.py`）执行，纠错/调查模型不直接启用 google_search 工具；provider 优先级、key pool、引导语等细节见 [`docs/llm_harness_routing.md`](docs/llm_harness_routing.md)。**例外**：`finesub.workflows.reference_ingest` 是用户主动发起的端到端工具，默认全执行（下载/GPU/LLM/知识库写入），`--dry-run` 才只打印计划。
- LLM 采样默认显式传 `temperature=1.0`；validation/parse retry 每失败一次下一次 logical attempt 降 `0.01` 并更换 `seed`，成功后的下一独立窗口/轮次恢复 attempt 0。`top_p` / `top_k` 不显式设置。
- 知识库更新走统一入口 `python -m finesub.llm.knowledge.update` / `run_knowledge_update`；三态开关 `--knowledge none|collect|update`、`--refined-srt` 精修对照模式、mistake 台账维护、`reference_ingest` 批量导入等完整行为见 [`docs/knowledge.md`](docs/knowledge.md)。
- URL 媒体下载逻辑在 `src/finesub/media/source.py`；主 pipeline 和 reference-ingest workflow 共享。URL→id 映射缓存在参考数据根下的 `url-map.json`（`finesub.paths.resolve_reference_data_root()`：仓库形态是 `data/reference/`，装好的前端是 `user-data/reference/`——不能按当前工作目录解析，那在打包形态下是下次更新就被替换掉的源码快照）；下载的源视频与抽取音频放在对应 artifact 目录（pipeline 默认 `out/<video-id>/`，reference ingest 默认 `out/reference/<video-id>/`）。每窗音频剪辑在任务自己的 `<stem>.llm-artifacts/clips/`，随任务清理一并消失。

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

显存档位按整卡容量命名，每档固定给系统预留 1GiB。固定映射由最大窗口显存实测验证：

| 档位 | pipeline 可用 | WT 实例数 | Separator 实例数 | Separator BS |
| ---: | ---: | ---: | ---: | ---: |
| 4GB | 3GiB | 1 | 1 | 1 |
| 8GB | 7GiB | 2 | 2 | 1 |
| 12GB | 11GiB | 3 | 3 | 1 |
| 16GB | 15GiB | 4 | 4 | 1 |

默认 profile 为 4GB。实例数按硬件档位固定递增，而不是按本机局部吞吐最优点截断：
4GB 为 1，每增加 4GB，WT 和 separator 各增加 1 个实例。`large-v3-turbo`
1/2/3/4 实例实测峰值分别为 2.17/4.29/6.01/8.25GiB，均落在对应档位扣除
1GiB 后的预算内。本机 3 实例吞吐最好只作为硬件特例记录。人声分离的
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

不在该目录下的：URL→id 映射在参考数据根（仓库形态 `data/reference/url-map.json`，装好的前端在 `user-data/reference/`）；窗口媒体剪辑在 `<stem>.llm-artifacts/clips/<chunk_id>.aac`（`--llm-media video` 的纠错轮另有 `<chunk_id>.mp4`）——它在 artifact 目录里面，所以是随任务整删的；`--knowledge update` 时知识库写入 `knowledge/`（独立内嵌 git 仓库，自动提交，非主仓库跟踪）。批量运行（`python -m finesub.batch`、reference-ingest 多任务）另在 `out/batch/<batch-id>/batch-status.jsonl` 记录事件流（每行 `{item,label,stage,status,error?,ts}`）；每项的产物位置不变，仍归各自 `out/<stem>/` 或 `out/reference/<id>/`，重跑同一批即按上面的存在性跳过规则续跑。独立实验 CLI `finesub.llm.correction_translation --prompt-dir <dir>`（默认 dry-run）另把 `plan.json`/`research-round{1,2}.txt`/`correction-NNNN[-query].txt` 写到 `--prompt-dir`，与生产 pipeline 的产物集不同。

## Pipeline 复用规则

`src/finesub/pipeline.py` 每一步都会检查默认输出是否存在：

- `*-vocal.ogg`（管线交付）或 `*-vocal.flac`（无损交付）任一存在则跳过人声分离。判定走
  `PipelinePaths.resolve_vocal_audio()`，与下游读取用的是同一个解析——只认 `.ogg` 会让手上
  已有无损轨的运行白跑一遍最贵的 GPU 阶段。
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
- 把 profile 标定扩展到更多 GPU 型号；当前 4/8/12/16GB 映射只在 RTX 5060 Ti 上完成最大窗口实测。
- 继续拆分 `src/finesub/speech/recognition/transcribe.py` 和
  `src/finesub/speech/preprocessing/energy.py` 的算法、I/O、CLI 边界。
- `segmentation-split.md` 待细化：跨语料泛化、合成词切点罚、纯虚构词归属、recall 救援交互、beam 信任折扣（5 项评审点）。

**LLM harness**

- 滑动窗（RPM/TPM）仍为进程内内存态；跨进程持久化为后续项（`docs/llm_harness_routing.md`）。
- Shared Context（跨窗口共享上下文）暂不做，触发条件见 `docs/llm_design_notes.md`。
- `tools/session_replay/run.py` argparse help 仍写 "basic->basicA"，应为 "basic->basicB"。

**知识库**

- 内嵌 git 是过渡方案，未来替换为在线托管（`docs/knowledge.md`）。
- 精选维护任务、翻译风格注入统一机制、子词条拆分自动化——见 `docs/knowledge.md` 遗留开放项。
- `docs/knowledge-node-plan.md`：node 模型、误听反查检索、三层信号与共享库设计稿，待实施（原打分方案已归档）。

**文档**

- 继续完善 `docs/llm_design_notes.md` 中的 LLM 纠错、翻译、知识库和 prompt/harness 自我迭代后处理层。
- 文档迁入本地 `docs/archive/` / `docs/report/` 前：提取非过时有用信息到仍在追踪的 docs（见 `CLAUDE.md` Archive extraction）。
