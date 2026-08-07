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

Production entrypoint: `asr-pipeline` / `asr_playground.pipeline`. An experimental LLM
correction/translation post-processing layer lives in `src/llm/` but is **not** part of the default production flow
(default `--stage` stops at `raw-srt`; `translated-srt`/`final-srt` opt in).

## Commands

```powershell
pip install -e ".[asr]"              # GPU ASR stack incl. qwen-verify referee (patched CTranslate2, see README_DEV)
pip install -e ".[harness]"          # LLM layer only (~4GB RAM, ffmpeg)
pip install -e ".[asr,harness,dev]"  # full runtime + pytest

# Full pipeline (default output out/<stem>/<stem>.srt; artifacts grouped under out/<stem>/)
asr-pipeline data/input.wav --model large-v3-turbo --language en --gpu-budget-gb 8

# Batch over many sources: download(x2) -> asr(x1, each file uses the full profile) -> llm(x1, ordered);
# per-item failure isolation; events at out/batch/<id>/batch-status.jsonl
python -m asr_playground.batch --manifest tasks.jsonl   # or positional URLs/paths

# LLM correction/translation — DRY-RUN BY DEFAULT (plans + prompts only).
# --execute calls Gemini APIs and consumes quota: never add it unless the user asks.
python -m llm.correction_translation out/input/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts

# Dev tools under tools/ (maintained ON DEMAND only — never update them as a side
# effect of other changes; their tests are not collected by the default suite):
# Replay a harness session with frozen upstream injections (correction R2 default;
# reuses search+extract body; docs/session_replay.md). Default calls the API.
python -m tools.session_replay correction --dry-run --label dry1

# Run 产物/知识库审计（离线只读）。入口：.claude/skills/run-audit/SKILL.md
# （成品/知识/schema + harness 时间线）。纠错 prompt 迭代与验收协议不在本 skill——
# 见 docs/tools/prompt-iterate.md + tools/session_replay。脚本只是确定性第一遍扫描。
python .claude/skills/run-audit/scripts/extract_digest.py out/reference/<id> --refined <精修.srt>
python .claude/skills/run-audit/scripts/extract_digest.py --kb knowledge

# Tests (lightweight only — no model loads/GPU; docs/testing.md for markers & scoping)
python -m pytest -q
# Pre-commit sanity
python -m compileall -q src test; python -m pytest -q; git status --short
```

Heavy-resource tests (`--run-heavy-resource`) and full production runs: only when the user
explicitly asks. No linter/formatter is configured.

## Key facts & guardrails

- **Dry-run is the LLM default**; only `--execute` spends Gemini generation quota. Exception:
  `asr_playground.workflows.reference_ingest` executes everything by default (user invoking it is the opt-in).
  The Gemini `countTokens` endpoint is completely free (auth-only key) — planning can call it
  freely; with the local tokenizer binary (`bin/windows-amd64/tokcount.exe`) even that is offline.
- `gemini/gemini-3.1-flash-lite` supports native thinking on both Gemini Free and Paid; the prior `supports_reasoning=false` catalog entry was a verified misclassification (2026-07-12).
- **Don't change VAD/ASR parameters** (`src/asr_playground/speech/recognition/transcribe.py`
  and `src/asr_playground/speech/preprocessing/energy.py` are the high-risk core);
  any change needs a stated output-consistency impact + tests or an experiment record.
- GPU (>=4GB VRAM; 4/8/12/16GB profiles) is expected; CPU fallback must print `Warning:` to
  stderr. Don't assume GPU is unavailable without checking.
- Artifacts never land at the repo root: `assets/` (raw media — see `assets/index.md` for what
  each one is and where its derived files live), `data/` (hand annotations, refined subtitles,
  caches — **never cleared by a rerun**), `out/` (generated), `tmp/` (scratch). All ignored. Pipeline stages skip on **existence only** (no content validation) — to force a
  rerun, delete the stage output and everything downstream.
- Dependencies live only in `pyproject.toml`; never create `requirements.txt`. Production
  entrypoints call functions directly (no subprocess).
- The knowledge base `knowledge/` is NOT tracked by the main repo (own embedded git, auto-commits
  on apply). Prompt templates `src/llm/prompt_templates/*.md` ARE tracked — prompt text is never
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
  First publish / later releases: from a clean `dev` tip, `git checkout --orphan main` (first
  time only) or on existing `main` replace the tree with `dev`'s (`git read-tree -u --reset
  dev` / equivalent), commit one snapshot, `git tag vX.Y.Z`, `git push origin main --tags`.
  Never merge orphan `main` back into `dev`. Back up full history via a private remote or
  bundle — the public repo is not a history backup.

## Architecture map

- `src/asr_playground/pipeline.py` — orchestrator; derives artifact paths from the output SRT path, skips stages
  whose outputs exist, accepts URLs as input (yt-dlp). LLM stages opt-in via `--stage` and pass
  `--llm-*` / `--knowledge` / `--refined-srt` through.
- `src/asr_playground/batch.py` — three-bin batch runner (download×2/asr×1/llm×1 pools, ordered llm bin, per-item
  failure isolation) + manifest CLI; reference-ingest multi-task runs use the same engine.
  asr concurrency is 1 so each file takes the whole profile's separator width. ASR itself is
  single-worker (intra-file sharding was removed 2026-08-02; docs/wt-parallelism.md records the
  design and why it went). llm concurrency stays 1 by design
  (unlocked in-process rate limiter; knowledge auto-apply commits to the embedded git).
- `src/asr_playground/speech/preprocessing/separation.py` /
  `src/asr_playground/speech/recognition/stage.py` /
  `src/asr_playground/speech/postprocessing/stabilization.py` /
  `src/asr_playground/subtitles/rendering.py` — pipeline stages;
  concurrent CUDA separator blocks share one warmed Roformer `model_run` but keep per-block
  wrappers and are merged in order; separator and WT model families do not co-reside;
  `src/asr_playground/speech/recognition/stage.py` writes `*-aligned.json` (incl. global-DP
  re-segmentation via `src/asr_playground/speech/postprocessing/segmentation.py`,
  docs/segment_split.md; ASR-native segment starts are marked
  word-level `whisper_segment_start`), then ASR stabilization writes `*-stable.json`
  (profile 0 = cleanup → noise tags → drop; docs/asr-stabilize.md);
  `src/asr_playground/speech/recognition/transcribe.py` +
  `src/asr_playground/speech/preprocessing/energy.py` are the large core algorithm modules
  (high-risk); `src/asr_playground/speech/recognition/segments.py` keeps recognition output
  timelines self-consistent; `src/asr_playground/speech/recognition/word_starts.py` resolves
  `[*]` disfluency blocks and applies the VAD-anchored word-start clamps (docs/asr-align.md
  「词首修正」); `src/asr_playground/speech/verification/qwen_referee.py` produces
  second-model evidence at the stage tail (--qwen-verify; stabilize consumes it,
  docs/vad-asr.md); `src/asr_playground/speech/recognition/checkpoint.py`
  owns disposable ASR partials; `src/asr_playground/speech/runtime/resources.py` defines GPU
  budget tiers (separator instances only — **ASR is always one worker**),
  `src/asr_playground/speech/recognition/fw_refine_backend.py` owns the model pool and the
  patched-CT2 adapter,
  `src/asr_playground/speech/runtime/gpu_stage_gate.py` serializes separator/WT model
  families, and `src/asr_playground/run_metadata.py` owns pipeline timing/worker
  sidecars (separator instance counts; separator model batch stays 1).
- `src/asr_playground/media/` — lightweight media ownership: URL/download selection,
  ffmpeg/ffprobe operations and clip extraction. It has no speech or LLM dependency.
- `src/asr_playground/subtitles/` — shared SRT model, alignment, metrics, postprocessing and
  rendering. It has no speech or LLM dependency.
- `src/asr_playground/workflows/reference_ingest.py` — end-to-end reference ingest
  workflow; executes by default and combines batch, media, speech, LLM and knowledge update.
- `src/llm/` — the LLM harness:
  - `profiles.py` (6 route/level presets) · `config.py` (roles, limits, budgets) ·
    `model_catalog.psv` (model/tier facts; its `capability` column drives the correction
    prompt's capability tier — assembled per answering endpoint: capable=judgment merge,
    basic=conservative 1:1 merge fragments) · `api_keys.py` (`config.toml` provider switches
    and named pools) · `client.py` + `llm_runtime.py` + `rate_limit.py` +
    `content_filter.py` (Gemini REST calls via httpx, endpoint chains, RPM/TPM limiting,
    PROHIBITED_CONTENT ladder)
  - `chunking.py` (window planning) · `token_budget.py`/`token_truncate.py` (3-tier counter:
    local binary → countTokens → heuristic upper bound)
  - `output_tags.py` (tag block extraction incl. nesting-aware `find_top_level_tag_blocks`) +
    `session_contract.py` (per-session output contract: top-level nonempty/present blocks;
    single source of truth shared by production stages and the replay validators; correction's
    CSV round is out — validated by `csv_utils`)
  - `research.py` (2-round research + `run_research_stage`) · `search_loop.py` (multi-round
    judge) · `web_search.py` (local search agent: Exa → Gemma4 grounded → Tavily → DDG)
  - `stages/` (`plan.py` fast-mode decision, `fast_session.py`, `correction_loop.py` window
    execution/retries/resume) · `correction_translation.py` (CLI + orchestration)
  - `prompts.py` + `prompt_compose.py` + `prompt_variants.py` + `prompt_templates/`
    (message builders, fragment assembly, PROMPT_VERSION; four named prompt variants
    capableB/C + basicA/B, tier picks default — capable→capableC, basic→basicB —
    overridable via `--variant`, see docs/tools/prompt-iterate.md)
  - `knowledge/` (base, update, materials, feedback, entries, mistakes — knowledge base +
    unified post-task update; CLI `python -m llm.knowledge.update`)
  - `prompt_artifacts.py` (dry-run prompt assembly shared with `--prompt-dir`) ·
    `task_report.py` / `exchange_log.py` / `exchange_metadata.py` (artifacts & readable logs)
- `src/finesub_bootstrap/` — shared end-user provisioning layer. `paths.py` owns the three
  roots (data root `%LOCALAPPDATA%\FineSub` for `user-data`; install root for `runtime`;
  big-data root for `models`/`cache`/`tasks`, defaulting to the install root and recorded in
  `locations.json`) plus packaged-install detection; `locks.py`/`fsops.py` are the shared
  advisory lock and the link-safe, crash-safe directory operations; `shell.py` is the `finesub`
  subcommands themselves, shared by the CLI wheel and the desktop package's own command line;
  `secrets.py` is the `.env` key-protection layer (DPAPI-bound envelope encryption, in-place
  `fs$…` value replacement) and the project's **only** `.env` parser/writer — do not grow
  another; `migrations/` holds one-way `user-data` fixes recorded by id, run by both front
  ends at startup and never fatal. Also verified downloads, safe zip extraction and the
  uv-managed runtime.
  Used by the desktop app (`desktop/backend`) and the CLI shell; never imports from `desktop`.
  Needs `pydantic`+`httpx` — except `secrets.py`, which stays stdlib-only because
  `llm.llm_runtime` imports it from plain `[harness]` installs (the package `__init__` must
  stay import-free; `test/test_secrets.py` guards both). Tests live in `desktop/backend/tests`
  (desktop CI); `secrets` is tested in the root suite.
- `cli/` — the publishable `finesub` CLI shell (own pyproject; wheel = thin launcher +
  vendored source snapshot under `finesub_cli/_vendor`, only entry point is `finesub`;
  subcommands come from `finesub_bootstrap.shell`, this layer only supplies `FINESUB_HOME`,
  the vendored sources and uv).
  Provisions `%LOCALAPPDATA%\FineSub` via `finesub_bootstrap` (uv from its own wheel
  dependency — pin must match runtime-manifest, test-enforced), shares user-data with the
  installed desktop app. Build: `cli/scripts/build-wheel.ps1` (version from `desktop/VERSION`).
  Tests in `cli/tests` (desktop CI).
- `tools/` — standalone dev tools, maintained ON DEMAND only (each has a README with its
  policy): `session_replay/` (freeze upstream
  injections, re-hit a session), `asr-confidence-explorer/` (manual analysis snapshot).
  Their tests live next to them and are not collected by the default suite.
- `legacy/` — superseded scripts; don't build on it. Shared text helpers live in
  `src/asr_playground/text.py`; heavy audio/resource helpers live under `speech/`.

## Docs index (read on demand)

| Doc | Read when the task involves |
| --- | --- |
| `README.md` | User-facing usage: install, quickstart, common entrypoints, note-writing tips |
| `docs/manual/env.md` | `.env` / Gemini (AI Studio) / Exa / Tavily API key setup |
| `docs/manual/repo-install.md` | 仓库（源码）安装全步骤：uv 默认 / pip 替代（torch 必须走 cu128 索引的坑在此）。README 只留 Desktop + 托管 CLI |
| `docs/manual/resources.md` | **面向用户**：数据落在哪（user-data 统一在 `%LOCALAPPDATA%`、大文件默认随安装目录）、`finesub relocate` 搬盘与共用、注册脚本、卸载三档、缓存为何单独删没用、仓库/worktree 模式 |
| `docs/manual/ct2-wheel.md` | **面向用户**：patched CTranslate2 怎么装、怎么自检、装错了什么症状 |
| `docs/ct2-distribution.md` | **面向维护者**：patched CT2 的打包与分发——自包含 wheel 怎么做（DLL 进包目录，免 `add_dll_directory`）、为什么发 Release 不进仓库、只有 direct reference 能排除 stock（`==4.8.1` 两个都收）、`cublas64_12.dll` 来源等未决项 |
| `README_DEV.md` | Dev principles, resource constraints, **canonical artifact tree**, reuse/resume rules, agent checklist |
| `desktop/README_DEV.md` | 桌面端维护者文档：架构与 bridge、三个目录根、依赖与 pylock 再生成、外部工具（manifest 而非 lock）、包内命令行、开发/测试/构建、签名发布与更新演练。`desktop/README.md` 只留用户向内容 |
| `docs/llm_harness_behavior.md` | **Canonical LLM runtime behavior**: presets, fast mode, search agent, window/token budgets, injection limits, output contract, retries/stitching, resume, rate limits, artifacts |
| `docs/knowledge.md` | Everything knowledge-base: structure, `--knowledge` tri-state, feedback v2, unified update, mistake ledger, reference_ingest |
| `examples/knowledge/` | Tracked mini knowledge-base samples (not the live `knowledge/` tree) |
| `docs/llm_prompts.md` | Prompt templates/fragments, prompt_compose assembly table, PROMPT_VERSION semantics |
| `docs/llm_design_notes.md` | Architecture intent, design decisions & rationale (budget formula derivation, knowledge-update decision ledger), deferred designs |
| `docs/data-index.md` | **数据与基线索引**：跟踪的人工标注（词起点边界、分割点金标准、异常 group 语料）、未跟踪但被反复引用的本地素材（`assets/`、`out/qwen-explore` 的 VAD 轨、`out/acceptance` 验收产物），以及只存在于文档里的实测基线及其可复现性。找"那份数据在哪"先看这里 |
| `docs/testing.md` | Test markers, scoped commands, which tests cover which paths |
| `docs/gpu-profiles.md` | 4/8/12/16GB GPU profile mapping, maximum-window benchmark data and concurrency rationale |
| `docs/separator-optimization.md` | **BS-Roformer 推理效率探索（E0–E11）**：生产已采纳的 AMP + 同精度预热与**已进生产**的编译路径（regional AOTInductor 1.895× / JIT 1.381×，档位选择见 README_DEV「分离器的编译加速」）；已否决的 `inference_mode`/延后 cache 清理；无权重 package 的常量烘焙缺陷与交叉校验、worker 阶梯实测；E11 记录迁到 torch 2.11 后 `emulate_precision_casts` 的方向反转。想动分离器性能或并发数之前先读，避免重复已做过的实验 |
| `docs/wt-parallelism.md` | **单文件 WT 分片，已于 2026-08-02 移除**（回溯点 `dev` 的 `1fcc4e1`）。仍然成立的部分：align 时间 97.9% 在 whisper.transcribe 内、语义分组边界、checkpoint、intra-op 线程预算、2026-07-29 双 shard 冻结的根因（未读取的 capture pipe 造成 stdio 背压——长任务绝不要走它）、以及开发用 stall watchdog（`ASR_STALL_WATCHDOG_SEC`）及其 GIL 隐患 |
| `docs/wt-refine-handoff.md` | **CT2 WT refine 研究交接入口**（已合入 dev）：目的、过程、1-pass/2-pass 与性能结论、patch-series 交付决策、研究脚本 pointer、切默认 backend 前的剩余待办 |
| `docs/wt-refine-port.md` / `docs/wt-refine-validation.md` | WT refine → FW/CT2 的详细算法契约、multi-audio batch 设计与档位表、beam/模型/边界的质量实测，以及 13-group 信号/局部隔离验证结果；先从 handoff 导航 |
| `docs/vad-energy.md` | energy VAD 本体：处理流程（含 exit-run 累加器、-45 峰值底线与其 partial carve、pause hints）、流式=内存契约、峰值 clamp 为何不是全局缩放（2026-08-05，含实测与被否决的线性过渡限幅）、`WaveformObserver` 搭车钩子（第二个信号复用已归一化 block）、Python API。opt-in `--vad-silero-assist` 见 vad-asr.md |
| `docs/vad-asr.md` | `vad-asr` 组合阶段（`speech/recognition/stage.py`）：CLI（含 `--vad-silero-assist`）、VAD→ASR 数据流、**aligned JSON 字段契约**、资源与失败行为 |
| `docs/asr-align.md` | VAD interval -> aligned ASR：解码配置、词级映射、异常救援阶梯（greedy -> 异常 interval 隔离）与其取舍依据、覆盖率救援、输出字段语义（含 `confidence` 不作质量指标的说明） |
| `docs/asr-stabilize.md` | aligned → stable ASR profiles, metrics, tags, CLI, and resume rules (profile 3 pre-merge was removed 2026-07-29 — that section records why) |
| `docs/session_replay.md` | Prompt-iteration replay: 6 sessions (correction R2 + query/research-r1/r2/search-judge/fast-round1), 各轮 fixture/validation 契约、补中间态落盘、变体仅 correction 支持 |
| `docs/segment_split.md` | 字幕分句规范：全局 DP 打分（可切可并，ASR 接缝带 bonus）、gap word 调整、字段继承与幂等（生产 `src/asr_playground/speech/postprocessing/segmentation.py`；`tools/split_explorer` 为调参薄封装） |
| `docs/tools/prompt-iterate.md` | **纠错 prompt 迭代方法论**（长期）：定位（prompt/harness 迭代唯一机制，不碰知识库更新）、四变体 capableB/C + basicA/B、session_replay 协议（`--model`/`--variant`）、prompt 原则、失效模式（含 singles 残留案例）、产物命名。已完成 run 的离线诊断仍走 `.claude/skills/run-audit`；二者分工见该 skill 文首 |
| `docs/segmentation-gold.md` | **分割点金标准**：人工标必切/禁切/宜切的规范与判据、时间轴锚定、完整标注窗口契约、打分口径（`tools/segmentation_gold/`）。审计分割质量、或要动机械指标时先读 |
| `docs/merge-calibration.md` | 精修标定的合并软门槛与模型边界（默认不并、gap/字数先验、flash-lite thinking=0）；现行变体契约仍以 prompt-iterate §4 为准 |
| `docs/package-reorganization-plan.md` | 第 10 步结构重构：`asr_playground` namespace、speech/LLM/media/subtitles/workflows 边界、分阶段迁移与验收 |

`docs/archive/` 与 `docs/report/` 为本地笔记（gitignore），不随仓库发布；迁入前按上方
**Archive extraction** 规则抽非过时信息。
