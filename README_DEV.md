# finesub 开发者 / Agent 说明

这份文档面向维护者和 LLM coding agent。目标是减少误改、误跑重资源任务，以及把生产入口和实验入口区分清楚。

## 当前项目地图

生产主路径：

```text
src/asr_playground/pipeline.py
```

生产流水线：

```text
source audio
  -> src/asr_playground/speech/preprocessing/separation.py
  -> src/asr_playground/speech/recognition/stage.py
  -> src/asr_playground/speech/postprocessing/stabilization.py
  -> src/asr_playground/subtitles/rendering.py（默认产出 *-raw.srt）
```

关键模块：

- `src/asr_playground/pipeline.py`：生产编排入口，负责中间文件路径、跳过已有输出、stage-based resume。默认跑到 `raw-srt`；`translated-srt` / `final-srt` 才进入 LLM 纠错翻译和 SRT 后处理。
- `src/asr_playground/batch.py`：download / ASR / LLM 三 bin 批处理引擎。
- `src/asr_playground/speech/preprocessing/separation.py`：人声分离，使用 `audio-separator`。
- `src/asr_playground/speech/preprocessing/vad.py`：流式 VAD 检测与能量轨迹。
- `src/asr_playground/speech/recognition/stage.py`：组合 VAD 与 Whisper recognition，输出未稳定化的 `*-aligned.json`。
- `src/asr_playground/speech/recognition/transcribe.py`：单 shard recognition service；仍包含待拆分的 windows、decoder、timestamp mapping 和 recovery。
- `src/asr_playground/speech/recognition/checkpoint.py`：ASR partial identity、schema、原子写入与清理。
- `src/asr_playground/speech/recognition/sharding.py`：WT shard 规划、并发执行和 interval ownership 合并。
- `src/asr_playground/speech/recognition/segments.py`：识别输出的重叠收回、零时长修复与空段过滤。
- `src/asr_playground/speech/postprocessing/stabilization.py`：独立 ASR 稳定化 stage，按 profile 从 aligned 生成 stable。
- `src/asr_playground/subtitles/rendering.py`：stable JSON 转 SRT。
- `src/asr_playground/speech/runtime/model_pool.py`：WT 模型串行加载、按 shard 延迟创建与复用。
- `src/asr_playground/speech/runtime/resources.py`：4/8/12/16GB 显存档位、1GB 系统预留、WT/Separator 实例数与资源上限检查。
- `src/asr_playground/speech/preprocessing/energy.py`：VAD-energy 核心算法，体积较大，修改需谨慎。
- `src/asr_playground/media/`：下载/URL 选择、ffmpeg/ffprobe 和 clip 提取；公共轻量层，不依赖 speech/LLM。
- `src/asr_playground/subtitles/`：SRT model、alignment、metrics、postprocess 和 rendering；公共轻量层。
- `src/asr_playground/workflows/reference_ingest.py`：跨 batch/media/speech/LLM/knowledge 的参考素材导入 workflow。
- `legacy/`：superseded 脚本（旧 `main.py`/`vad.py`/`align.py` 等，以及从 `src/` 移入的 `to_toon.py`、`rms.py`）。不被当前流水线使用，避免在此基础上开发。
- `src/llm/`：实验性 LLM 工具，包括两轮背景调查、纠错翻译 prompt/窗口规划/API harness；不是默认生产流水线的一部分。
  - `profiles.py`：翻译路线/档位 preset（`--route text|mm × --level low|med|high`，输出公式 `k × c × csv_tokens`）。
  - `api_keys.py`：读取 `.env` 的命名 key，并按 `config.toml` 统一解析 provider 开关与 pool。
  - `llm_runtime.py`：生成调用底层封装（原 `llm.py`）。
  - `research.py`：两轮背景调查 + 多轮搜索 loop（`run_research_stage`，原 `stages/research_stage.py` 已并入此文件）。
  - `stages/`：`plan.py`（窗口规划、fast 模式判定）、`fast_session.py`（融合会话）、`correction_loop.py`（纠错窗口循环）。
  - `knowledge/`：知识库子包——`base.py`（原 `knowledge_base.py`）、`update.py`（统一知识更新入口，原 `knowledge_update.py`，CLI 现为 `python -m llm.knowledge.update`）、`mistakes.py`（原 `common_mistakes.py`）、`entries.py`、`feedback.py`、`materials.py`。
  - mm-high 视频剪辑经 `--video` 接入。
- `src/llm/prompt_templates/`：LLM 后处理 prompt/harness 模板，随主仓库版本化；prompt 迭代只改模板文件。v7 起纠错侧为骨架 + fragment 组装，选择逻辑在 `src/llm/prompt_compose.py`（按 preset 挑 fragment），组装参考见 `docs/llm_prompts.md`。
- `src/llm/model_catalog.psv`：pipe-delimited 模型/provider tier 事实表；运行期 role binding 和限速逻辑仍在现有 Python 配置中，catalog 仅用于元信息、exchange 和非思考模型显式 reasoning prompt 判断。
- `knowledge/`：本地知识库数据（不是 `llm/knowledge/` 代码包）。结构、更新流程见 [`docs/knowledge.md`](docs/knowledge.md)。主 git 不追踪；目录内有独立 git 仓库，知识更新 apply 后自动 commit。主仓样板见 [`examples/knowledge/`](examples/knowledge/)。
- `docs/llm_design_notes.md`：LLM 纠错与翻译后处理层架构意图与设计决策。
- `docs/llm_harness_behavior.md`：当前 LLM harness 的窗口拆分、重试、拼接和 prompt 输入行为。
- `docs/knowledge.md`：知识库全套（结构、`--knowledge` 三态、统一知识更新、mistake 台账、`reference_ingest`）。
- `docs/llm_prompts.md`：prompt/fragment 组装参考。
- `docs/testing.md`：测试命令与域标记。
- `docs/vad-energy.md` / `asr-align.md` / `vad-asr.md` / `asr-stabilize.md`：stable 之前四个独立工具的运行时行为、接口和产物语义。
- `tools/`：独立开发工具，全部**按需维护**——不随主程序改动自动更新，测试不进默认套件，
  具体规则见各自 README：`session_replay/`（冻结注入重打 session，`python -m
  tools.session_replay`）、`asr-confidence-explorer/`（手工分析快照）。

## 分步调试

生产时优先使用 `asr-pipeline`。需要定位问题时，可以分步运行。

1. 人声分离：

```powershell
vocal-separation data/input.wav -o out/input-vocal.flac --gpu-budget-gb 8
```

2. VAD + ASR 对齐：

```powershell
vad-asr out/input-vocal.flac --output out/input-aligned.json --model large-v3-turbo --language en --gpu-budget-gb 8
```

3. ASR 稳定化：

```powershell
asr-stabilize out/input-aligned.json -o out/input-stable.json --profile 0
```

4. 导出 SRT：

```powershell
to-srt out/input-stable.json -o out/input.srt
```

## 开发原则

- 不要维护 `requirements.txt` 或 `requirements-dev.txt`；依赖只放在 `pyproject.toml`（安装 extras 见 README.md「安装」）。
- 根目录不放新的音频、字幕、JSON 或媒体产物；使用 `data/`、`out/`、`tmp/`。
- 默认不要改 VAD/ASR 参数。若必须改，说明对输出一致性的影响，并补测试或实验记录。
- 生产入口应调用函数，不要用 subprocess 拼 CLI。
- LLM 后处理默认只生成计划和 prompt；真实生成 API 调用必须显式 opt-in，且默认测试不得联网或消耗 Gemini quota（窗口规划的 token 计数按 **本地 tokenizer 二进制 → 免费 `countTokens` 端点 → 启发式** 三级 fallback（`default_token_counter()`；本地二进制源码在 `src/tools/gemini-token-counter/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入依赖，离线且与 API 逐字一致，详见其 README；测试中一律注入 fake counter）。联网检索全部由本地检索代理（`llm/web_search.py`）执行，纠错/调查模型不直接启用 google_search 工具；provider 优先级、key pool、引导语等细节见 [`docs/llm_harness_behavior.md`](docs/llm_harness_behavior.md)。**例外**：`asr_playground.workflows.reference_ingest` 是用户主动发起的端到端工具，默认全执行（下载/GPU/LLM/知识库写入），`--dry-run` 才只打印计划。
- LLM 采样默认显式传 `temperature=1.0`；validation/parse retry 每失败一次下一次 logical attempt 降 `0.01` 并更换 `seed`，成功后的下一独立窗口/轮次恢复 attempt 0。`top_p` / `top_k` 不显式设置。
- 知识库更新走统一入口 `python -m llm.knowledge.update` / `run_knowledge_update`；三态开关 `--knowledge none|collect|update`、`--refined-srt` 精修对照模式、mistake 台账维护、`reference_ingest` 批量导入等完整行为见 [`docs/knowledge.md`](docs/knowledge.md)。
- URL 媒体下载逻辑在 `src/asr_playground/media/source.py`；主 pipeline 和 reference-ingest workflow 共享。URL→id 映射缓存在 `data/reference/url-map.json`；下载的源视频与抽取音频放在对应 artifact 目录（pipeline 默认 `out/<video-id>/`，reference ingest 默认 `out/reference/<video-id>/`）。每窗音频剪辑在 `tmp/llm-audio-clips/<stem>/`。

## 资源约束

**`*-stable.json` 之前（人声分离、VAD-ASR、ASR 稳定化等）**

- NVIDIA GPU，至少 4GB 显存。
- 至少 8GB 空余系统内存。
- CPU 回退只是兜底；发生回退时必须向 stderr 输出 `Warning:`。

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
单文件 WT 分片**已实现并标定**（`src/asr_playground/speech/recognition/sharding.py` 负责规划与
分片执行，`transcribe.py` 注入单 shard recognition service，
生产缺省取 profile 上限；`--wt-workers N` 仅为 DEV/UNSAFE benchmark 覆盖）。
**实测加速 1.1–1.2×**——对齐阶段 97.9% 的时间在
`whisper.transcribe` 内（管线自身 Python 仅 2.1%），并发扩展性本身没问题（完美配平应得
1.45×），差距来自 shard 配平失准与模型加载错峰，修法是动态派发；`workers=1` 与分片前产物
逐字节一致，多 worker 首轮实测也一致。并发只在单文件路径做，**batch 内每任务恒 1 worker**
（batch 的 asr bin 本就按 profile 并行跑多个文件，两侧模型实例总数相同、显存包络不变）。
WT 并发吞吐实测本机 2 实例 1.49×、3 实例见顶，**该曲线是机器特性，换卡须重测**。
语义 group 分片、扩容门槛、模型池、checkpoint、profile 影响产物的新耦合、已搁置的
跨 worker recall 与跨任务共享池，以及「修掉原生 whisper 段首漂移后改用 faster-whisper」
的备选路线，见 [`docs/wt-parallelism.md`](docs/wt-parallelism.md)。

单项管线在同一进程内顺序调用各阶段（无 subprocess）。进程级 GPU model-family
gate 允许多个 separator 或多个 WT 同类任务并行，但不会让两个模型族跨任务同时驻留。
并发分离只共享 Roformer `model_run`（权重和预热后的模型级缓存），每个音频块仍有
独立 `Separator` / `model_instance` wrapper、输入输出路径和 source cache；空输出重试
也重新取得干净 wrapper。不同的 600 秒 core + pad 块并行完成后由主线程按序裁剪拼接，
实际 worker 数不超过文件块数，单块短音频不创建线程池；全局加权限流防止多个 batch
task 把实例数相乘。非 CUDA 后端保持顺序独立模型。

ASR 对齐阶段不再把整段音频读进内存，而是用 `AudioBlockLoader`（600s 核 + 10s pad）
按块从磁盘流式取音频，RAM 不再随时长在 Whisper 之外线性增长；此路径与独立 CLI
`asr_playground.speech.recognition.transcribe.main` 一致，输出逐字节等价（见
`test_pipeline_refactor.py` 的等价性守卫测试）。
4GB 默认档位的实测峰值由 `test/test_resource_budget_pipeline.py` 守护
（`heavy_resource`，默认 skip，需 `--run-heavy-resource`）。

**VAD 也是流式的**（`asr_playground.speech.preprocessing.energy.run_vad_file` /
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

## 产物清单与路径

缺省输出路径为 `out/<stem>/<stem>.srt`（`default_output_path`，不传 `-o` 时），一次运行的全部 artifact 都从最终 SRT 路径推导、归到 `out/<stem>/` 一个目录；URL 输入使用 `video-id` 作为 stem，并把下载/抽取媒体放在同一 artifact 目录；显式传 `-o` 时按该路径同级推导、不加子目录。以 stem=`input`、跑到 `final-srt` 为例：

```text
out/input/
├── input-vocal.flac              # 人声分离 (vocal_separation)
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
    ├── correction-windows.jsonl #   纠错窗口 resume 缓存：每个成功窗口一行，供中途 resume 复用
    ├── exchanges/               #   每次 LLM API 交互一个 markdown（含 API call trace / reasoning 摘录）
    ├── task-report.md           #   运行时汇总（API 计数、token、fallback/warning 等；注入/thinking 见 docs/llm_harness_behavior.md）
    └── knowledge-update-{chunks.jsonl,harness-notes-NN.md}  # 仅 --knowledge update：apply ledger / 精修模式 harness notes
```

`*-metadata.json` 与其他任务产物同级，不依赖 LLM stage。只追踪下载、人声分离、
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
confidence 的最小值。`vad-asr` 另按最终 segment 边界写入
`vad_weighted_energy_db`；定义与 metadata 见 [`docs/vad-asr.md`](docs/vad-asr.md)。
旧产物或上游未返回相应指标时字段可缺省。stable 默认经 profile 0 清理/标记；完整
profiles、`tags` 与指标定义见 [`docs/asr-stabilize.md`](docs/asr-stabilize.md)。

不在该目录下的：URL→id 映射在 `data/reference/url-map.json`；窗口媒体剪辑在 `tmp/llm-audio-clips/<stable-json-stem>/<chunk_id>.aac`（mm-high 纠错轮另有 `<chunk_id>.mp4`）；`--knowledge update` 时知识库写入 `knowledge/`（独立内嵌 git 仓库，自动提交，非主仓库跟踪）。批量运行（`python -m asr_playground.batch`、reference-ingest 多任务）另在 `out/batch/<batch-id>/batch-status.jsonl` 记录事件流（每行 `{item,label,stage,status,error?,ts}`）；每项的产物位置不变，仍归各自 `out/<stem>/` 或 `out/reference/<id>/`，重跑同一批即按上面的存在性跳过规则续跑。独立实验 CLI `llm.correction_translation --prompt-dir <dir>`（默认 dry-run）另把 `plan.json`/`research-round{1,2}.txt`/`correction-NNNN[-query].txt` 写到 `--prompt-dir`，与生产 pipeline 的产物集不同。

## Pipeline 复用规则

`src/asr_playground/pipeline.py` 每一步都会检查默认输出是否存在：

- `*-vocal.flac` 存在则跳过人声分离。
- `*-aligned.json` 存在则跳过 VAD-ASR；stable 缺失时可直接从 aligned 运行 ASR 稳定化。
- **ASR 断点续跑**（`asr_playground.speech.recognition.transcribe.align_segments`，
  长音频崩溃后不必从头再来）：每处理完一个
  alignment group 就原子写 `*-aligned.partial.json`，内含已完成 segments、区间游标、
  `prev_tail_segments` 与 auto-language history——即 group 边界上的完整状态。重启时按指纹
  （model / language / gap_sec / 音频路径+大小+mtime / 区间摘要 / `ASR_CHECKPOINT_VERSION`）
  校验，一致才续跑，否则整份丢弃重跑。checkpoint 的身份、读写与清理由
  `src/asr_playground/speech/recognition/checkpoint.py` 统一负责。当前 schema 为 v2；v1/缺版本的旧 partial
  不迁移，直接从头重跑。partial 只是缓存不是产物：损坏或过期一律当作不存在，
  文件名与 `*-aligned.json` 区分开，不会被"存在即跳过"误判；跑完即删除。
- **whisper-timestamped 对齐降级**：该库 efficient 路径断言"whisper segment 数 == 词级对齐
  segment 数"，退化音频（长幻觉重复串）会打破前提并抛裸 `AssertionError` 直接终止整个 run。
  `_transcribe_with_naive_fallback` 捕获它并以 `naive_approach=True` 重试同一 group（该路径是
  库在 beam search / temperature fallback 时自己强制走的，非新路径）；naive 再失败则该 group
  按静音丢弃并打 `Warning:`，保证长音频不会因单个 group 崩掉。正常 group 不受影响。
- 同进程内若有多个 VAD-ASR worker，Whisper **模型加载串行、推理仍可并行**；加载锁只覆盖
  `load_model`，避免多个模型的瞬时加载峰值叠加，不把整段 ASR 锁成串行。
- `*-stable.json` 存在则跳过 ASR 稳定化及其上游；不会为了补档而重新生成缺失的 aligned。
- **特殊**：显式目标为 `aligned` 时，stable 不能代替 aligned；aligned 缺失仍会运行人声分离和 VAD-ASR。
- `*-raw.srt` 存在则跳过 raw SRT 导出。
- LLM stage 会复用 artifact 目录内的 `*-research-context.json`（兼容旧的 run 根目录位置）、`*-translated.srt`、最终 `*.srt` 和 task artifact 目录；如果 translated 已存在但 final 不存在，只跑 SRT 后处理。
- **LLM session resume**（默认开启，需 task artifact 目录）：research R1/R2、每轮 search judge、fast round 1 和逐窗 query 的 parser 验证成功输出写入 `<artifact_dir>/session-checkpoints.jsonl`。重启后 harness 先重建本地确定性状态（搜索/网页提取、媒体剪辑和上传允许重做），到同一 LLM 边界时按“精确 messages + PROMPT_VERSION + 调用配置 + 媒体/任务身份”命中旧响应，并用当前 parser 再验证后复用；输入或契约变化自动失效。research 的完整 `*-research-context.json` 仍可整阶段复用。
- **纠错窗口中途 resume**：每个成功窗口另写 `<artifact_dir>/correction-windows.jsonl`；命中后整窗回放，连该窗 query、搜索、剪辑上传和纠错调用都跳过，从第一个未完成窗口继续 live。缓存按 task fingerprint + 每窗 input_hash 匹配。拆窗时 in-iteration 产生的第一个半窗可能重算（其余半窗和整窗复用）。`--no-resume` 同时关闭两种 resume ledger 的读写，但不删除已有文件，也不改变 pipeline 对完整 stage 输出文件的存在性复用。

stage 级跳过当前只检查文件存在，不校验内容和参数一致性（research planning metadata 与两种 resume 缓存例外，带 fingerprint/input_hash 校验）。后续如果增强，应优先加：

- FLAC 可读性检查。
- JSON schema / metadata 检查。
- 输出参数 fingerprint。
- 参数不一致时 warning 或强制重跑选项。

## 测试规则

默认全量（2 worker 并行）：

```powershell
python -m pytest -q
```

默认单测不得加载 Whisper/audio-separator 模型、处理大音频、显著占用显存，或联网/消耗 Gemini quota。重资源测试标记 `@pytest.mark.heavy_resource`，只有显式传 `--run-heavy-resource` 才运行，且只有用户明确要求时才应主动跑。

按域标记（`llm`/`pipeline`/`asr`）、按改动路径选测试文件、`slow` 标记等完整对照见 [`docs/testing.md`](docs/testing.md)。

常规验证：

```powershell
python -m compileall -q src test
python -m pytest -q
asr-pipeline --help
vad-asr --help
vocal-separation --help
to-srt --help
python -m llm.correction_translation --help
python -m llm.knowledge.update --help
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
- 继续拆分 `src/asr_playground/speech/recognition/transcribe.py` 和
  `src/asr_playground/speech/preprocessing/energy.py` 的算法、I/O、CLI 边界。
- `segment_split.md` 待细化：跨语料泛化、合成词切点罚、纯虚构词归属、recall 救援交互、beam 信任折扣（5 项评审点）。

**LLM harness**

- 滑动窗（RPM/TPM）仍为进程内内存态；跨进程持久化为后续项（`docs/llm_harness_behavior.md`）。
- Shared Context（跨窗口共享上下文）暂不做，触发条件见 `docs/llm_design_notes.md`。
- `tools/session_replay/run.py` argparse help 仍写 "basic->basicA"，应为 "basic->basicB"。

**知识库**

- 内嵌 git 是过渡方案，未来替换为在线托管（`docs/knowledge.md`）。
- 精选维护任务、翻译风格注入统一机制、子词条拆分自动化——见 `docs/knowledge.md` 遗留开放项。
- `docs/kb_entry_scoring_plan.md`：条目评分设计定稿，待实施。

**文档**

- 继续完善 `docs/llm_design_notes.md` 中的 LLM 纠错、翻译、知识库和 prompt/harness 自我迭代后处理层。
- 文档迁入本地 `docs/archive/` / `docs/report/` 前：提取非过时有用信息到仍在追踪的 docs（见 `CLAUDE.md` Archive extraction）。
