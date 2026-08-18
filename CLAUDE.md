# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` / `README_DEV.md` and `docs/` are written in Chinese and are the canonical docs.
This file keeps only always-needed facts plus an index — read the referenced doc when a task
touches its area.

## What this project is

Local pipeline that converts long-form audio into subtitles:

```text
source audio -> vocal separation -> VAD + ASR alignment -> ASR stabilization -> SRT
```

Production entrypoint: `python -m finesub.pipeline` (the packaged CLI's `finesub`
runs the same module). An experimental LLM
correction/translation post-processing layer lives in `src/finesub/llm/` but is **not** part of the default production flow
(default `--stage` stops at `raw-srt`; `translated-srt`/`final-srt` opt in).

## Commands

```powershell
pip install -e ".[asr]"              # GPU ASR stack incl. qwen-verify referee (patched CTranslate2, see README_DEV)
pip install -e ".[harness]"          # LLM layer only (~4GB RAM, ffmpeg)
pip install -e ".[asr,harness,dev]"  # full runtime + pytest

# Full pipeline (default output out/<stem>/<stem>.srt; artifacts grouped under out/<stem>/)
python -m finesub.pipeline data/input.wav --model large-v3-turbo --language en --gpu-budget-gb 8

# Batch over many sources: download(x2) -> asr(x1, each file uses the full profile) -> llm(x1, ordered);
# per-item failure isolation; events at out/batch/<id>/batch-status.jsonl
python -m finesub.batch --manifest tasks.jsonl   # or positional URLs/paths

# LLM correction/translation — DRY-RUN BY DEFAULT (plans + prompts only).
# --execute calls Gemini APIs and consumes quota: never add it unless the user asks.
python -m finesub.llm.correction_translation out/input/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts

# Dev tools under tools/ (maintained ON DEMAND only — never update them as a side
# effect of other changes; their tests are not collected by the default suite):
# Replay a harness session with frozen upstream injections (correction R2 default;
# reuses search+extract body; docs/session_replay.md). Default calls the API.
python -m tools.session_replay correction --dry-run --label dry1

# Run 产物/知识库审计（离线只读）。入口：.claude/skills/run-audit/SKILL.md
# （成品/知识/schema + harness 时间线）。纠错 prompt 迭代与验收协议不在本 skill——
# 见 docs/prompt-iterate.md + tools/session_replay。脚本只是确定性第一遍扫描。

# Tests (lightweight only — no model loads/GPU; docs/testing.md for markers & scoping)
python -m pytest -q
# Pre-commit sanity
python -m compileall -q src test; python -m pytest -q; git status --short
```

Heavy-resource tests (`--run-heavy-resource`) and full production runs: only when the user
explicitly asks. No linter/formatter is configured.

## Key facts & guardrails

- **Dry-run is the LLM default**; only `--execute` spends Gemini generation quota. Exception:
  `finesub.workflows.reference_ingest` executes everything by default (user invoking it is the opt-in).
  The Gemini `countTokens` endpoint is completely free (auth-only key) — planning can call it
  freely; with the local tokenizer binary even that is offline (a checkout runs
  `bin/windows-amd64/tokcount.exe` if present — untracked since 0.4.0, build or unzip it per
  `tools/tokcount/README.md`; packaged front ends fetch the `tokcount` manifest resource,
  and fall back to the endpoint if they cannot).
- **Standing test authorization (owner, 2026-08-13):** tests may use the configured
  `gemini-3.1-flash-lite` / `gemini-3.5-flash-lite` APIs, the GPT-5.6 Luna local Agent, and
  configured search APIs without asking first, provided usage stays within a normal test-sized
  range. Ask the owner before consuming any other model, Agent, paid/quota-bearing service, or
  heavy compute resource. This authorization permits `--execute` only for those scoped tests;
  it does not change the product CLI's dry-run default.
- `gemini/gemini-3.1-flash-lite` supports native thinking on both Gemini Free and Paid; the prior `supports_reasoning=false` catalog entry was a verified misclassification (2026-07-12).
- **Don't change VAD/ASR parameters** (`src/finesub/speech/recognition/transcribe.py`
  and `src/finesub/speech/preprocessing/energy.py` are the high-risk core);
  any change needs a stated output-consistency impact + tests or an experiment record.
- GPU (>=4GB VRAM; 4/8/12/16GB profiles) is expected; CPU fallback must print `Warning:` to
  stderr. Don't assume GPU is unavailable without checking. **All of "can this machine use the
  GPU" lives in `speech/runtime/device.py`** (`resolve_device` / `cuda_usable`): it also catches
  a card too old for the installed torch, which `torch.cuda.is_available()` reports as usable —
  never decide device placement from a bare `is_available()`. User-facing supported-card table
  is in README.md. The CPU fallback only works because the patched CT2 wheel uses **oneDNN** for
  CPU GEMM — Ruy deadlocks in the model destructor, MKL costs 3.4GB extra and is Intel-only, and
  all three failures surface only when something actually decodes. Read
  `tools/wt_refine_port/ct2-patches/README.md` before touching the wheel.
- **Artifacts** go in one of four ignored directories, never loose at the repo root: `assets/`
  (raw media — see `assets/index.md` for what each one is and where its derived files live),
  `data/` (hand annotations, refined subtitles, caches — **never cleared by a rerun**),
  `out/` (generated), `tmp/` (scratch). Pipeline stages skip on **existence only** (no content
  validation) — to force a rerun, delete the stage output and everything downstream.
  *Runtime state is a separate thing and does sit at the root* when you run from a checkout:
  `.state`, `.env` / `.env.lock`, `agent-sessions.jsonl`, `.task-activity/`, `cache/`. That is
  the checkout standing in for `%LOCALAPPDATA%\FineSub` (see the resources doc); it is ignored
  too, but do not go looking for it under `out/`.
- Dependencies live only in `pyproject.toml`; never create `requirements.txt`. Production
  entrypoints call functions directly (no subprocess).
- The knowledge base `knowledge/` is NOT tracked by the main repo (own embedded git, auto-commits
  on apply). Prompt templates `src/finesub/llm/prompt_templates/*.md` ARE tracked — prompt text is never
  hardcoded in Python.
- **Running from a checkout uses the checkout's own data** (`knowledge/`, `.env`, `.state`) and
  never touches `%LOCALAPPDATA%`; `FINESUB_CHECKOUT_DATA=0` opts out. A **git worktree resolves
  to the main checkout**, and knowledge auto-apply inside one is **skipped with a warning** unless
  `FINESUB_KNOWLEDGE_WRITE=1` — ask the developer before running anything that would update the
  main repo's knowledge base from a worktree.
- No backward-compatibility burden: interfaces change directly; stale artifacts just rerun
  (PROMPT_VERSION bumps invalidate resume caches/research contexts by design).
- When changing pipeline or LLM behavior, update the affected tests AND the owning doc
  (see index below) in the same change.
- **Archive extraction**: when moving a doc into local-only `docs/archive/` or `docs/report/`
  (or deleting a tracked draft), skim it for non-obsolete facts that tracked docs still need;
  promote those into the owning persistent doc (or a new one under `docs/`) and fix dangling
  links in the same change. Do not leave the only copy of current behavior inside archive.
- **Git / public release (orphan `main`)**: local long-lived branch is `dev` (full history;
  do not push to the public remote). Public GitHub (`origin`, product name finesub) only
  carries `main`: an orphan line of release snapshots so intermediate commits stay private.
  Publishing (release or plain sync) goes through `scripts/publish-main.ps1`: it snapshots
  `dev`'s tree onto `main`'s tip, pushes that commit to the throwaway `ci-gate` branch, and
  fast-forwards `main` only once CI is green — **never force-push `main`**, fix on `dev` and
  rerun the script instead.
  Tagging and the rest of a release: the `release` skill. Never merge orphan `main` back into
  `dev`. Back up full history via a private remote or bundle — the public repo is not a
  history backup.
  Worktrees must therefore branch from local HEAD, not `origin/main` — the default `fresh`
  would base new work on an orphan release snapshot. Set `worktree.baseRef: "head"` in each
  checkout's `.claude/settings.json`; **that file cannot be committed** (`.gitignore` keeps all
  of `.claude/` out except the run-audit skill), so a fresh clone starts on the wrong default
  until someone sets it. The setting only accepts `fresh`/`head` and so cannot name `dev`
  outright: be on `dev` when creating a worktree, or pass an explicit base to
  `git worktree add`. Existing worktrees are sibling directories
  `../asr-playground-<topic>`, not `.claude/worktrees/`.

## Architecture map

模块 → 谁拥有它的文档 → **动它之前必须知道的**。描述性细节都在 owner 文档里，这张表不复述；
它只回答「我要改这里，该读哪一份、有什么是不能碰的」。带 ✱ 的约束由测试固化，改错会红。

### 管线骨架

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `pipeline.py` | `README_DEV.md` 产物清单与路径 | 所有 artifact 路径从最终 SRT 派生；阶段按**存在性**跳过（不校验内容）——要重跑就删该阶段产物**及其下游**。LLM 阶段靠 `--stage` opt-in |
| `batch.py` | `docs/wt-parallelism.md` | 三 bin（download×2 / asr×1 / llm×1）。**asr 并发恒为 1**，每个文件独占整个 profile；LLM 任务池也恒为 1（任务内 `continuity=parallel` 才并发，走 ticketed 限流器） |
| `run_metadata.py` | `README_DEV.md` | 计时/worker sidecar；分离器 model batch 恒为 1 |
| `reporting.py` | `docs/cli-bootstrap-logging-download-plan.md` | 事件契约与 quiet/normal/verbose |

### speech（高危区）

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `speech/recognition/transcribe.py`、`speech/preprocessing/energy.py` | `docs/asr-align.md`、`docs/vad-energy.md` | **两个高危核心**。改参数要有输出一致性论证 + 测试，或一份实验记录（见上「Key facts」） |
| `speech/runtime/device.py` | 上方 Key facts 的 GPU 条 | 「这台机器能不能用 GPU」**只在这里**。永远不要用裸 `torch.cuda.is_available()` 决定设备 |
| `speech/preprocessing/separator/` | `docs/separator-optimization.md` | 三模块同住（stage + 编译缓存 `accel` + AOTI package）。已做过的实验别重做。交付形态由输出后缀二选一（`.ogg` = 16k 单声道 ASR 轨 / `.flac` = 无损），其余后缀报错 |
| `speech/preprocessing/spectral.py` | `docs/vad-energy.md` | 加权能量信号**同时**被 VAD 与 `recognition/transcribe.py` 读；`audio.py` 只管解码与切片 |
| `speech/recognition/vad_asr_stage.py` | `docs/vad-asr.md` | 写 `*-aligned.json`——**字段契约在那份文档里**，含全局 DP 重分句与 `whisper_segment_start` 标记 |
| `speech/postprocessing/segmentation.py` | `docs/segmentation-split.md` | 全局 DP 打分（可切可并）；幂等要求 |
| `speech/postprocessing/stabilization.py` | `docs/asr-stabilize.md` | 写 `*-stable.json`；profile 与 resume 规则 |
| `speech/recognition/word_starts.py` | `docs/asr-align.md`「词首修正」 | `[*]` 块解析 + VAD 锚定的词首 clamp |
| `speech/recognition/{segments,checkpoint}.py` | `docs/vad-asr.md` | 时间轴自洽；checkpoint 是**可丢弃**的 ASR partial |
| `speech/verification/qwen_referee.py` | `docs/vad-asr.md` | `--qwen-verify` 的第二模型证据，stabilize 消费它 |
| `speech/runtime/{resources,gpu_stage_gate}.py` | `docs/gpu-profiles.md` | GPU 档位只决定**分离器实例数**——**ASR 永远单 worker**；分离器与 WT 模型族不共存 |
| `speech/recognition/fw_refine_backend.py` | `docs/wt-refine-handoff.md` | 模型池 + patched-CT2 adapter。换 wheel 前先读 `tools/wt_refine_port/ct2-patches/README.md` |

✱ `speech` 不得 import `llm`。

### 无依赖的公共层

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `media/` | — | URL/下载选择、ffmpeg/ffprobe、剪辑。✱ 无 speech / LLM 依赖 |
| `subtitles/` | `docs/segmentation-split.md`（分句口径） | SRT 模型、对齐、指标、后处理、渲染。✱ 无 speech / LLM 依赖 |
| `config.py` | `docs/manual/model-routing.md` | 共享 `config.toml` 的**定位/解析/记忆化**，仅此而已。stdlib-only 且与领域无关——各域校验自己那张表。保留注释的**写入器**是 `finesub_bootstrap/config_file.py` |
| `paths.py` | `README_DEV.md`「运行时路径解析契约」 | 唯一的仓库/运行时路径 resolver。✱ `src/` 里别处不得用 `parents[N]` 找仓库根 |
| `workflows/reference_ingest.py` | `docs/knowledge.md` | **默认全执行**（用户主动发起即 opt-in），与 LLM 层 dry-run 默认相反 |

### LLM harness（`llm/`）

| 子域 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `routing/` | `docs/manual/model-routing.md`（用户）、`docs/llm_design_notes.md`（取舍） | ✱ 层次单向：catalog → routes → execution_policy → model_router → client，`config` 只被读。**policy 只是 backend 闸门**，agent 靠成为模型组成员参与。`model_catalog.psv` / `model_routes.toml` 与代码同目录、进 package-data |
| `client.py` · `llm_runtime.py` · `provider_transports.py` · `rate_limit.py` · `content_filter.py` | `docs/llm_harness_behavior.md`、`docs/provider-adapters.md` | Gemini REST + OpenAI-compat/Anthropic 纯文本 adapter；RPM/TPM 限流；PROHIBITED_CONTENT 阶梯 |
| `agent/` | `docs/llm_local_agent.md`（**§12.1 是「什么已接线」的唯一入口**） | ✱ 模块名保留 `agent_` 前缀——`finesub_bootstrap.shell` 用**字符串** `python -m finesub.llm.agent.agent_cleanup` 调它们，改名不炸 import、只在运行时炸。durable task 运行时是**尚未接进生产调用点的地基** |
| `stages/correction/` | `docs/prompt-iterate.md` | 八模块：`run` 规划+收尾 · `serial`/`parallel` 两 driver · `attempts` 每窗重试拆分 · `query_round` · `context` · `commit` 断点与失效判据 · `metadata`。**测试替身放做名字查找的那个模块**，不要放包 `__init__`（`test/conftest.py` 的 `setattr_correction`） |
| `chunking.py` · `token_budget.py` · `token_truncate.py` | `docs/llm_harness_behavior.md` | 三级 token 计数：本地二进制 → 免费 `countTokens` → 启发式上界 |
| `prompts.py` · `prompt_compose.py` · `prompt_variants.py` · `prompt_templates/` | `docs/llm_prompts.md` | ✱ **prompt 文本从不硬编码在 Python 里**；合并阈值**只在** `prompt_constants.py` 写一次；示例由 `example_builder.py` 生成，golden 快照按 `PROMPT_VERSION` 锁 |
| `output_tags.py` · `session_contract.py` · `output_protocol.py` | `docs/llm_harness_behavior.md` | 每会话输出契约的唯一真相，生产与 replay 校验共用；纠错的 CSV 轮不在 `session_contract` 里，归 `output_protocol` |
| `research.py` · `search_loop.py` · `web_search.py` | `docs/llm_harness_research.md` | 联网检索一律由本地检索代理执行（Exa → Gemma4 grounded → Tavily → DDG），纠错/调查模型不直开 `google_search` |
| `knowledge/` | `docs/knowledge.md` | ✱ 不依赖 `stages/`（单向：stages 读写知识库）。`knowledge/` 本身不被主仓跟踪，自带内嵌 git |
| `prompt_artifacts.py` · `task_report.py` · `exchange_log.py` | `README_DEV.md` 产物清单 | 产物与可读日志 |

### 装机与前端

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `finesub_bootstrap/` | `desktop/README_DEV.md`（维护）、`docs/manual/resources.md`（用户） | ✱ **不得 import 主包**（`shell.py` 里唯一一处是函数内延迟 import）。`secrets.py` 与 `token_counter.py` ✱ **stdlib-only**——纯 `[harness]` 装机与薄 CLI 的 3.10 都会 import 它们；包 `__init__` 必须保持 import-free。`secrets.py` 是本项目**唯一**的 `.env` 解析/写入器 |
| `desktop/` | `desktop/README_DEV.md` | 三个目录根、bridge、jobs 四模块、样式表导入顺序即层叠顺序 |
| `cli/` | `cli/README.md` | 薄 launcher + `_vendor` 源码快照；唯一入口是 `finesub`。构建清单有离线守卫 |
| `tools/` | 各自的 README | **只按需维护**——不要作为其他改动的副作用去更新它们。例外：改名/移动类改动必须同步 `tools/session_replay` |
| `legacy/` | — | 本地 gitignored 目录，不随仓库发布；不要在它上面建东西 |

## Docs index (read on demand)

按「任务涉及什么」分段；段内顺序即阅读顺序。

**怎么读这堆文档**

| Doc | Read when the task involves |
| --- | --- |
| `docs/README.md` | 这堆文档怎么分：`manual/` 是使用者，`docs/` 根是开发者；分界是读者不是主题 |
| `CHANGELOG.md` | 面向用户的变更记录，含破坏性变更。改了用户看得见的行为就要写一条 |

**面向使用者**

| Doc | Read when the task involves |
| --- | --- |
| `README.md` | User-facing usage: install, quickstart, common entrypoints, note-writing tips |
| `docs/manual/env.md` | `.env` / Gemini (AI Studio) / Exa / Tavily API key setup |
| `docs/manual/repo-install.md` | 仓库（源码）安装全步骤：uv 默认 / pip 替代（torch 必须走 cu128 索引的坑在此）。README 只留 Desktop + 托管 CLI |
| `docs/manual/resources.md` | **面向用户**：数据落在哪（user-data 统一在 `%LOCALAPPDATA%`、大文件默认随安装目录）、`finesub relocate` 搬盘与共用、注册脚本、卸载三档、缓存为何单独删没用、仓库/worktree 模式 |
| `docs/manual/ct2-wheel.md` | **面向用户**：patched CTranslate2 怎么装、怎么自检、装错了什么症状 |
| `docs/manual/model-routing.md` | **面向用户**：一次调用怎么定下来（会话→任务组→预设格子→模型组→逐候选过滤）、两个媒体开关 / difficulty / 思考旋钮的意义与出厂实况、catalog 各列、接自己的 provider/模型/模型组/预设、启动告警怎么读、改什么会作废 checkpoint |
| `docs/manual/agent.md` | **面向用户**：用本机 Codex / Claude Code / Antigravity 订阅代替 Gemini 额度——三个 execution_policy 档位怎么选、要装什么、agy 下为何不建议开视频多模态、失败会怎样、`finesub agent-clean` 与搬盘/卸载。少术语，不讲实现 |
| `examples/knowledge/` | Tracked mini knowledge-base samples (not the live `knowledge/` tree) |

**开发与维护总入口**

| Doc | Read when the task involves |
| --- | --- |
| `README_DEV.md` | Dev principles, resource constraints, **canonical artifact tree**, reuse/resume rules, agent checklist |
| `desktop/README_DEV.md` | 桌面端维护者文档：架构与 bridge、三个目录根、依赖与 pylock 再生成、外部工具（manifest 而非 lock）、包内命令行、开发/测试/构建、签名发布与更新演练。`desktop/README.md` 只留用户向内容 |
| `docs/testing.md` | Test markers, scoped commands, which tests cover which paths |
| `docs/data-index.md` | **数据与基线索引**：跟踪的人工标注（词起点边界、分割点金标准、异常 group 语料）、未跟踪但被反复引用的本地素材（`assets/`、`out/qwen-explore` 的 VAD 轨、`out/acceptance` 验收产物），以及只存在于文档里的实测基线及其可复现性。找"那份数据在哪"先看这里 |

**语音链路（VAD / ASR / 分句 / 分离器）**

| Doc | Read when the task involves |
| --- | --- |
| `docs/vad-energy.md` | energy VAD 本体：处理流程（含 exit-run 累加器、-45 峰值底线与其 partial carve、pause hints）、流式=内存契约、峰值 clamp 为何不是全局缩放（2026-08-05，含实测与被否决的线性过渡限幅）、`WaveformObserver` 搭车钩子（第二个信号复用已归一化 block）、Python API。opt-in `--vad-silero-assist` 见 vad-asr.md |
| `docs/vad-asr.md` | `vad-asr` 组合阶段（`speech/recognition/vad_asr_stage.py`）：CLI（含 `--vad-silero-assist`）、VAD→ASR 数据流、**aligned JSON 字段契约**、资源与失败行为 |
| `docs/asr-align.md` | VAD interval -> aligned ASR：解码配置、词级映射、异常救援阶梯（greedy -> 异常 interval 隔离）与其取舍依据、覆盖率救援、输出字段语义（含 `confidence` 不作质量指标的说明） |
| `docs/asr-stabilize.md` | aligned → stable ASR profiles, metrics, tags, CLI, and resume rules (profile 3 pre-merge was removed 2026-07-29 — that section records why) |
| `docs/segmentation-split.md` | 字幕分句规范：全局 DP 打分（可切可并，ASR 接缝带 bonus）、gap word 调整、字段继承与幂等（生产 `src/finesub/speech/postprocessing/segmentation.py`；`tools/split_explorer` 为调参薄封装） |
| `docs/segmentation-gold.md` | **分割点金标准**：人工标必切/禁切/宜切的规范与判据、时间轴锚定、完整标注窗口契约、打分口径（`tools/segmentation_gold/`）。审计分割质量、或要动机械指标时先读 |
| `docs/gpu-profiles.md` | 4/8/12/16GB GPU profile mapping, maximum-window benchmark data and concurrency rationale |
| `docs/separator-optimization.md` | **BS-Roformer 推理效率探索（E0–E11）**：生产已采纳的 AMP + 同精度预热与**已进生产**的编译路径（regional AOTInductor 1.895× / JIT 1.381×，档位选择见 README_DEV「分离器的编译加速」）、块产物为何固定 FLAC 与**交付两模式**（16k 单声道 ogg / 无损 flac，2026-08-18，含实测与接缝论证）；已否决的 `inference_mode`/延后 cache 清理；无权重 package 的常量烘焙缺陷与交叉校验、worker 阶梯实测；E11 记录迁到 torch 2.11 后 `emulate_precision_casts` 的方向反转。想动分离器性能或并发数之前先读，避免重复已做过的实验 |
| `docs/wt-parallelism.md` | **单文件 WT 分片，已于 2026-08-02 移除**（回溯点 `dev` 的 `1fcc4e1`）。仍然成立的部分：align 时间 97.9% 在 whisper.transcribe 内、语义分组边界、checkpoint、intra-op 线程预算、2026-07-29 双 shard 冻结的根因（未读取的 capture pipe 造成 stdio 背压——长任务绝不要走它）、以及开发用 stall watchdog（`ASR_STALL_WATCHDOG_SEC`）及其 GIL 隐患 |
| `docs/wt-refine-handoff.md` | **CT2 WT refine 研究交接入口**（已合入 dev）：目的、过程、1-pass/2-pass 与性能结论、patch-series 交付决策、研究脚本 pointer、切默认 backend 前的剩余待办 |
| `docs/wt-refine-port.md` / `docs/wt-refine-validation.md` | WT refine → FW/CT2 的详细算法契约、multi-audio batch 设计与档位表、beam/模型/边界的质量实测，以及 13-group 信号/局部隔离验证结果；先从 handoff 导航 |

**LLM harness**

| Doc | Read when the task involves |
| --- | --- |
| `docs/llm_harness_behavior.md` | **Canonical LLM runtime behavior**（总入口，文首有拆分导航）：开关轴、fast 模式、输入输出、窗口拆分与调用形态、prompt 信息、注入上限、artifact、最终 SRT 后处理、重试与拼接、知识库更新 |
| `docs/llm_harness_routing.md` | 路由的 **dev 侧**：模型事实/池/路由链、thinking 档位换算、模型配置与限流。面向使用者的同一主题在 `manual/model-routing.md` |
| `docs/llm_harness_research.md` | **本地检索代理与背景调查**：Exa → Gemma4 grounded → Tavily → DDG 的降级链、按轴退化的 r1/r2、会话级轮结构 |
| `docs/llm_local_agent.md` | **Agent 执行后端唯一入口**（§1–§15）：三家 one-shot transport 的当前契约、durable task 协议（`agent-task-v3`：租约靠工作续期、blocked 出口、retrieval 三态）、订阅额度耗尽的 tier 冻结（§11.1）、会话记录的三处落点（§14.3）。**三种会话形态各差什么、接线在哪、为何押后一律看 §12.1**，不要从别处推断。文首有拆分导航；实施编年在本地 `docs/archive/agent_backend_implementation_log.md` |
| `docs/llm_agent_tool_protocol.md` | **把 agent 后端改成「调工具」而不是「收表格」的方案**（未实施，**实施前有四项待定见 §5**——其中「并行窗口下 agy 的配置作用域」是硬冲突，会改变它的实施形态）：今天为何**一次往返都没有**、由此长出的五处赘生物、目标形态（协议不动、新增 harness 自己的 MCP server、driver 收缩）、**为什么是 MCP 不是放行接口脚本**，以及三家 CLI 的实测代价（Claude Code 最干净；agy 需要一行全局权限、隔离并不因此变好；Codex 未测且可能翻转选型）。动 agent 传输前先读 |
| `docs/llm_local_agent_experiments.md` | Agent 长驻会话的**准则与实测**（§1–§3）：会话复用 A/B（agy 小任务净亏 46%）、缓存写入门槛的成因与生产尺寸复测、Claude Code 的 n=1 反向信号。要动复用默认值、或想知道某个数字怎么量出来的，看这份 |
| `docs/llm_local_agent_runtime.md` | Agent 的**执行环境卫生**（§1–§2）：episode 分两档落在哪、capsule 是一次性 episode 不是持久 store、滚动上限 20 与清理。排查现场残留、动清理命令或搬盘时读 |
| `docs/llm_local_agent_agy.md` | **agy 专属**（原第 16 节，现自成一篇、从 §1 起编）：catalog 行、音频必须容器化、视频分辨率不可调与 token 公式失真、`view_file` 准入硬门、原生搜索的第二个 project。只跟 agy 打交道时读 |
| `docs/llm_prompts.md` | Prompt templates/fragments, prompt_compose assembly table, PROMPT_VERSION semantics |
| `docs/llm_design_notes.md` | Architecture intent, durable model-routing decisions & rationale, current budget-formula derivation, knowledge-update decision ledger, deferred designs |
| `docs/llm_followups.md` | **LLM 尚未完成的实验/设计唯一入口**：P6 开关组合与输出系数标定、none/native 逐窗选词条实验及重启条件、P7d parallel 并发上限标定、P8 超长素材分块调查的 go/no-go 与实现约束，以及模型路由/Agent 遗留（fast 内部预算、自定义 endpoint 媒体、Agent worker 重构、per-provider token 汇总）。不要从已归档的历史计划推断现状 |
| `docs/knowledge.md` | Everything knowledge-base: structure, `--knowledge` tri-state, feedback v2, unified update, mistake ledger, reference_ingest |
| `docs/provider-adapters.md` | 自定义 provider（OpenAI-compat/Anthropic 纯文本 adapter）：供应商行为差异调研表与 adapter 契约（usage/截断/拒答/失败分类归一化、D18 无采样参数、key 命名） |
| `docs/prompt-iterate.md` | **纠错 prompt 迭代方法论**（长期）：定位（prompt/harness 迭代唯一机制，不碰知识库更新）、四变体 capableB/C + basicA/B、session_replay 协议（`--model`/`--variant`）、prompt 原则、失效模式（含 singles 残留案例）、产物命名。已完成 run 的离线诊断仍走 `.claude/skills/run-audit`；二者分工见该 skill 文首 |
| `docs/session_replay.md` | Prompt-iteration replay: 6 sessions (correction R2 + query/research-r1/r2/search-judge/fast-round1), 各轮 fixture/validation 契约、补中间态落盘、变体仅 correction 支持 |
| `docs/merge-calibration.md` | 精修标定的合并软门槛与模型边界（默认不并、gap/字数先验、flash-lite thinking=0）；现行变体契约仍以 prompt-iterate §4 为准 |
| `docs/kb_entry_scoring_plan.md` | 知识库词条打分方案（设计稿）；未落地前只作参考 |

**分发、前端与跨前端**

| Doc | Read when the task involves |
| --- | --- |
| `docs/cli-bootstrap-logging-download-plan.md` | **CLI/日志/下载（P1–P5 已实施；发布验收未做）**：`finesub.reporting` 的事件契约与 quiet/normal/verbose、首次大文件目录选择（仅 CLI，`setup --dirs-only`）、`download_routes` 地区解析、模型与依赖的加速接线。`download-sources.json`/`model-manifest.json`/cn lock 已填入实测值；非大陆机器地区解析为 global、一律走官方源。待做的发布验收（全量摘要比对、大陆实机安装演练）见该文档 §5 |
| `docs/ct2-distribution.md` | **面向维护者**：patched CT2 的打包与分发——自包含 wheel 怎么做（DLL 进包目录，免 `add_dll_directory`）、为什么发 Release 不进仓库、只有 direct reference 能排除 stock（`==4.8.1` 两个都收）、`cublas64_12.dll` 来源等未决项 |
| `docs/cross-frontend-lease.md` | **跨前端租约（活动集合、task-id 与 output 工作区互斥已实现，owner/pid 元数据待实现）**：稳定 user-data 闸门、每 reader/worker 独立租约、CLI/桌面 worker 共用的强制 sidecar、崩溃判活与仍缺的精确归属提示。动多前端并发语义前先读 |

**计划与决策**

| Doc | Read when the task involves |
| --- | --- |
| `docs/refactor-followups.md` | 2026-08 结构重构**已全部完成**后剩下的：挂发版流程的 tokcount 三步（必须按序）、发版验收要算进改名、判定押后的七项与其理由、明确不做的五条，以及**做同类改动前先读的四条教训**（筛选面要比改写面宽 / worktree+editable 假绿 / 绿≠对 / 改写与校验共用启发式会一起瞎）。计划正文在本地 `docs/archive/` |

两处此前在索引里零命中、要靠翻代码才找得到的：

| 找什么 | 去哪 |
| --- | --- |
| **纠错窗口的 CSV 输出契约**（列名、列数、按位置解析、`note` 里的 `\|` 怎么保住） | `docs/llm_harness_behavior.md`「输出协议」一节 + `src/finesub/llm/output_protocol.py`（header 常量的唯一定义处）。产物侧 `<stem>-annotated.csv` 的字段含义见 `README_DEV.md` 的产物树 |
| **哪些产物是记录、哪些可删、谁来删** | `desktop/README_DEV.md`「完成后清理中间产物」及其后两段（含**故意不删**的两类：URL 输入下载的源媒体、`-annotated.csv`/`-corrected.srt`）；实现在 `finesub_bootstrap/artifacts.py` |

`docs/archive/` 与 `docs/report/` 为本地笔记（gitignore），不随仓库发布；迁入前按上方
**Archive extraction** 规则抽非过时信息。
