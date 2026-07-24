# finesub 开发者 / Agent 说明

这份文档面向维护者和 LLM coding agent。目标是减少误改、误跑重资源任务，以及把生产入口和实验入口区分清楚。

## 当前项目地图

生产主路径：

```text
src/pipeline.py
```

生产流水线：

```text
source audio
  -> src/vocal_separation.py
  -> src/vad_asr.py
  -> src/asr_stabilize.py
  -> src/to_srt.py（默认产出 *-raw.srt）
```

关键模块：

- `pipeline.py`：生产编排入口，负责中间文件路径、跳过已有输出、stage-based resume。默认跑到 `raw-srt`；`translated-srt` / `final-srt` 才进入 LLM 纠错翻译和 SRT 后处理。
- `vocal_separation.py`：人声分离，使用 `audio-separator`。
- `vad_asr.py`：VAD-energy + Whisper alignment，输出未稳定化的 `*-aligned.json`。
- `asr_stabilize.py`：独立 ASR 稳定化 stage，按 profile 从 aligned 生成 stable。
- `to_srt.py`：JSON 转 SRT。
- `resource_profiles.py`：8/12/16GB 显存档位、0.5GB 系统预留、资源上限检查。
- `asr_align.py` / `vad_energy.py`：核心算法模块，体积较大，修改需谨慎。
- `legacy/`：superseded 脚本（旧 `main.py`/`vad.py`/`align.py` 等，以及从 `src/` 移入的 `to_toon.py`、`rms.py`）。不被当前流水线使用，避免在此基础上开发。
- `llm/`：实验性 LLM 工具，包括两轮背景调查、纠错翻译 prompt/窗口规划/API harness；不是默认生产流水线的一部分。
  - `profiles.py`：翻译路线/档位 preset（`--route text|mm × --level low|med|high`，输出公式 `k × c × csv_tokens`）。
  - `llm_runtime.py`：生成调用底层封装（原 `llm.py`）。
  - `research.py`：两轮背景调查 + 多轮搜索 loop（`run_research_stage`，原 `stages/research_stage.py` 已并入此文件）。
  - `stages/`：`plan.py`（窗口规划、fast 模式判定）、`fast_session.py`（融合会话）、`correction_loop.py`（纠错窗口循环）。
  - `knowledge/`：知识库子包——`base.py`（原 `knowledge_base.py`）、`update.py`（统一知识更新入口，原 `knowledge_update.py`，CLI 现为 `python -m llm.knowledge.update`）、`mistakes.py`（原 `common_mistakes.py`）、`entries.py`、`feedback.py`、`materials.py`。
  - mm-high 视频剪辑经 `--video` 接入。
- `llm/prompt_templates/`：LLM 后处理 prompt/harness 模板，随主仓库版本化；prompt 迭代只改模板文件。v7 起纠错侧为骨架 + fragment 组装，选择逻辑在 `llm/prompt_compose.py`（按 preset 挑 fragment），组装参考见 `docs/llm_prompts.md`。
- `llm/model_catalog.psv`：pipe-delimited 模型/provider tier 事实表；运行期 role binding 和限速逻辑仍在现有 Python 配置中，catalog 仅用于元信息、exchange 和非思考模型显式 reasoning prompt 判断。
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

生产时优先使用 `pipeline.py`。需要定位问题时，可以分步运行。

1. 人声分离：

```powershell
python src/vocal_separation.py data/input.wav -o out/input-vocal.flac --gpu-budget-gb 8
```

2. VAD + ASR 对齐：

```powershell
python src/vad_asr.py out/input-vocal.flac --output out/input-aligned.json --model large-v3-turbo --language en --gpu-budget-gb 8
```

3. ASR 稳定化：

```powershell
python src/asr_stabilize.py out/input-aligned.json -o out/input-stable.json --profile 0
```

4. 导出 SRT：

```powershell
python src/to_srt.py out/input-stable.json -o out/input.srt
```

## 开发原则

- 不要维护 `requirements.txt` 或 `requirements-dev.txt`；依赖只放在 `pyproject.toml`（安装 extras 见 README.md「安装」）。
- 根目录不放新的音频、字幕、JSON 或媒体产物；使用 `data/`、`out/`、`tmp/`。
- 默认不要改 VAD/ASR 参数。若必须改，说明对输出一致性的影响，并补测试或实验记录。
- 生产入口应调用函数，不要用 subprocess 拼 CLI。
- LLM 后处理默认只生成计划和 prompt；真实生成 API 调用必须显式 opt-in，且默认测试不得联网或消耗 Gemini quota（窗口规划的 token 计数按 **本地 tokenizer 二进制 → 免费 `countTokens` 端点 → 启发式** 三级 fallback（`default_token_counter()`；本地二进制源码在 `src/tools/gemini-token-counter/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入依赖，离线且与 API 逐字一致，详见其 README；测试中一律注入 fake counter）。联网检索全部由本地检索代理（`llm/web_search.py`）执行，纠错/调查模型不直接启用 google_search 工具；provider 优先级、key pool、引导语等细节见 [`docs/llm_harness_behavior.md`](docs/llm_harness_behavior.md)。**例外**：`llm.reference_ingest` 是用户主动发起的端到端工具，默认全执行（下载/GPU/LLM/知识库写入），`--dry-run` 才只打印计划。
- LLM 采样默认显式传 `temperature=1.0`；validation/parse retry 每失败一次下一次 logical attempt 降 `0.01` 并更换 `seed`，成功后的下一独立窗口/轮次恢复 attempt 0。`top_p` / `top_k` 不显式设置。
- 知识库更新走统一入口 `python -m llm.knowledge.update` / `run_knowledge_update`；三态开关 `--knowledge none|collect|update`、`--refined-srt` 精修对照模式、mistake 台账维护、`reference_ingest` 批量导入等完整行为见 [`docs/knowledge.md`](docs/knowledge.md)。
- URL 媒体下载逻辑在 `llm/media_source.py`；主 `pipeline.py` 和 `llm.reference_ingest` 共享。URL→id 映射缓存在 `data/reference/url-map.json`；下载的源视频与抽取音频放在对应 artifact 目录（pipeline 默认 `out/<video-id>/`，reference ingest 默认 `out/reference/<video-id>/`）。每窗音频剪辑在 `tmp/llm-audio-clips/<stem>/`。

## 资源约束

**`*-stable.json` 之前（人声分离、VAD-ASR、ASR 稳定化等）**

- NVIDIA GPU，至少 8GB 显存。
- 至少 8GB 空余系统内存。
- CPU 回退只是兜底；发生回退时必须向 stderr 输出 `Warning:`。

**LLM harness 阶段（自 stable.json 起）**

- 无需 GPU / 显存；约 4GB 系统内存。
- ffmpeg + ffprobe 在 PATH 上（窗口剪辑与时长探测）。
- 安装：`pip install -e ".[harness]"`（不含 torch / whisper 等 ASR 栈）。

显存档位（仅 ASR 阶段人声分离 batch）：

```text
8GB  -> vocal separation batch 4
12GB -> vocal separation batch 6
16GB -> vocal separation batch 8
```

所有档位额外预留 0.5GB 显存给系统。当前只有人声分离 batch 随档位变化；`vad_asr.py` 不随档位调整 Whisper 分组、VAD 能量线程或 ASR 上下文参数。

管线在同一进程内顺序调用各阶段（无 subprocess），因此两块 GPU 模型不会同时驻留：
`run_vocal_separation` 在返回前 `del separator` + `torch.cuda.empty_cache()`，人声分离显存
在 Whisper 加载前即释放。ASR 对齐阶段不再把整段音频读进内存，而是用 `AudioBlockLoader`
（600s 核 + 10s pad）按块从磁盘流式取音频，RAM 不再随时长在 Whisper 之外线性增长；此路径
与独立 CLI `asr_align.main` 一致，输出逐字节等价（见 `test_pipeline_refactor.py` 的等价性
守卫测试）。人声分离在空输出重试路径（`_build_separator` 重建前）也先释放旧 separator，避免
两份模型短暂共存。8GB 档位的实测峰值由 `test/test_resource_budget_pipeline.py` 守护
（`heavy_resource`，默认 skip，需 `--run-heavy-resource`）。

**VAD 也是流式的**（`vad_energy.run_vad_file` / `_streamed_frame_tracks`，600s 核 + 90s
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
├── input-stable.json            # ASR 稳定化结果 (asr_stabilize)
├── input-raw.srt                # 由 stable.json 直转的原始 SRT (to_srt)
├── input-translated.srt         # LLM 纠错+翻译中文字幕（未后处理）
├── input-corrected.srt          # 纠错后「原文」SRT
├── input-annotated.csv          # 9 列完整标注 CSV：type|position|duration|gap|corrected|translation|conf|char_count|note
├── input.srt                    # 最终 SRT（translated 后处理后）
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

`input-aligned.json` 的每个 ASR segment 保留 Whisper 来源段的 `confidence` 和
`no_speech_prob`，每个 `words[]` 项保留 Whisper 的 `confidence`。VAD 的旧段级
`conf` / `vad_conf` 不进入 aligned schema。segment 与 Whisper 来源段一一对应（不再在
VAD interval 边界切开）；异常重复清洗合成 word 时取各来源 word
confidence 的最小值。`vad-asr` 另按最终 segment 边界写入
`vad_weighted_energy_db`；定义与 metadata 见 [`docs/vad-asr.md`](docs/vad-asr.md)。
旧产物或上游未返回相应指标时字段可缺省。stable 默认经 profile 0 清理/标记；完整
profiles、`tags` 与指标定义见 [`docs/asr-stabilize.md`](docs/asr-stabilize.md)。

不在该目录下的：URL→id 映射在 `data/reference/url-map.json`；窗口媒体剪辑在 `tmp/llm-audio-clips/<stable-json-stem>/<chunk_id>.aac`（mm-high 纠错轮另有 `<chunk_id>.mp4`）；`--knowledge update` 时知识库写入 `knowledge/`（独立内嵌 git 仓库，自动提交，非主仓库跟踪）。批量运行（`python src/batch.py`、`llm.reference_ingest` 多任务）另在 `out/batch/<batch-id>/batch-status.jsonl` 记录事件流（每行 `{item,label,stage,status,error?,ts}`）；每项的产物位置不变，仍归各自 `out/<stem>/` 或 `out/reference/<id>/`，重跑同一批即按上面的存在性跳过规则续跑。独立实验 CLI `llm.correction_translation --prompt-dir <dir>`（默认 dry-run）另把 `plan.json`/`research-round{1,2}.txt`/`correction-NNNN[-query].txt` 写到 `--prompt-dir`，与生产 pipeline 的产物集不同。

## Pipeline 复用规则

`pipeline.py` 每一步都会检查默认输出是否存在：

- `*-vocal.flac` 存在则跳过人声分离。
- `*-aligned.json` 存在则跳过 VAD-ASR；stable 缺失时可直接从 aligned 运行 ASR 稳定化。
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
python src/pipeline.py --help
python src/vad_asr.py --help
python src/vocal_separation.py --help
python src/to_srt.py --help
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
- 为 8/12/16GB 档位补 heavy-resource 实测基准。
- 继续拆分 `asr_align.py` 和 `vad_energy.py` 的算法、I/O、CLI 边界。
- premerge E1/E2 过拟合风险：待 held-out 语料验证后固化；非日语启用前先离线审计（`docs/llm_design_notes.md` M.7）。
- `segment_split.md` 待细化：常量校准、合成词切点罚、纯虚构词归属、recall 救援交互、beam 信任折扣、confidence 继承语义（6 项评审点）。

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
