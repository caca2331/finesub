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
python -m finesub.pipeline data/input.wav --model large-v3-turbo --language en --gpu-tier standard

# Same entry point, many sources: download(x2) -> asr(x1, each file uses the full
# profile) -> llm(x2); per-item failure isolation; events at out/batch/<id>/batch-status.jsonl,
# live queue/control plane beside it, `--resume-batch` to continue an unfinished one
python -m finesub.pipeline --manifest tasks.jsonl   # or several positional URLs/paths

# LLM correction/translation — DRY-RUN BY DEFAULT (plans + prompts only).
# --execute calls Gemini APIs and consumes quota: never add it unless the user asks.
python -m finesub.llm.correction_translation out/input/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts

# Dev tools under tools/ (maintained on demand only — never update them as a side
# effect of other changes; their tests are not collected by the default suite):
# Replay a harness session with frozen upstream injections (correction R2 default;
# reuses search+extract body; docs/session_replay.md). Default calls the API.
python -m tools.session_replay correction --dry-run --label dry1

# Run 产物/知识库审计（离线只读）。入口：agent-tasks/run-audit/SKILL.md
# （成品/知识/schema + harness 时间线）。纠错 prompt 迭代与验收协议不在本 skill——
# 见 docs/prompt-iterate.md + tools/session_replay。脚本只是确定性第一遍扫描。

# Tests (lightweight only — no model loads/GPU; docs/testing.md for markers & scoping)
python -m pytest -q
# 结构守卫（跟着上面一起跑，不必单独记）：导入方向 / 选项默认值 / 文档链接与节号 /
# 单函数 <=300 行的棘轮 / 文档枚举的集合与代码一致 / manual 的中文正文用全角标点
# ——见 docs/testing.md「结构守卫」
# Pre-commit sanity
python -m compileall -q src test; python -m pytest -q; git status --short
```

Heavy-resource tests (`--run-heavy-resource`) and full production runs: only when the user
explicitly asks. No linter/formatter is configured.

## Key facts & guardrails

- **Dry-run is the LLM default**; only `--execute` spends Gemini generation quota. Exception:
  `finesub.workflows.reference_ingest` executes everything by default (user invoking it is the opt-in).
  The Gemini `countTokens` endpoint is free (auth-only key), so planning can call it freely; with
  the local tokenizer binary (`tools/tokcount/`, packaged as `bin/windows-amd64/tokcount.exe` in a
  checkout — untracked since 0.4.0, build or unzip it per its README) even that is offline;
  packaged front ends fetch it as a manifest resource and fall back to the endpoint if they cannot.
- **Standing test authorization (owner, 2026-08-13):** tests may use the configured
  `gemini-3.1-flash-lite` / `gemini-3.5-flash-lite` APIs, the GPT-5.6 Luna local Agent, and
  configured search APIs without asking first, provided usage stays within a normal test-sized
  range. Ask the owner before consuming any other model, Agent, paid/quota-bearing service, or
  heavy compute resource. This authorization permits `--execute` only for those scoped tests;
  it does not change the product CLI's dry-run default.
- `gemini/gemini-3.1-flash-lite` supports native thinking on both Gemini Free and Paid; the prior `supports_reasoning=false` catalog entry was a verified misclassification (2026-07-12).
- **Don't change VAD/ASR parameters** (`src/finesub/speech/recognition/transcribe.py`
  and `src/finesub/speech/preprocessing/energy.py` are the high-risk core);
  any change needs a stated **quality** impact + tests or an experiment record.
- **Non-bit-exactness is not by itself a veto** — for a change that deliberately alters
  the numeric path or implementation form (new checkpoint, sample rate, compile/quant
  path, batching). There, byte/segment equality is an optional *low-cost no-regression
  proof*: take it when you get it, **price** around it when you don't. **Two carve-outs
  still demand exact equality**: (a) an optimization that *claims semantic invariance*
  (deterministic fields must match bit-for-bit — timing/resource observations excepted);
  (b) correctness contracts — determinism, idempotence, serialization round-trips,
  resume/replay, streaming-vs-whole equivalence. And "high-quality and reasonable" is
  the *goal*, never a standalone acceptance criterion: dropping the consistency proof
  obliges you to **write down the substitute metric and threshold** — judging quality after
  seeing the numbers is not acceptance. That gate sits at **accepting / flipping the
  default**, not at writing the code: land the new path behind a gate, default off, and
  prepare the threshold in parallel. Owner doc: `README_DEV.md` → 开发原则.
- **Option defaults belong in the backend**; front ends override only as explicit,
  justified, registered exceptions. Target precedence: **CLI args → project config →
  global config → front-end default → backend default**. argparse must not carry a
  second copy of a default — pass `None`/absent and resolve in a shared backend resolver
  (`resolve_split_params` / `resolve_knowledge_switch` are the house pattern), and use
  `BooleanOptionalAction` / `auto|on|off` for switches. ⚠ **The chain's two middle layers
  are still the target, not today's behaviour** — there is **no per-key project/global
  config merge** (one `config.toml` wins whole) and front-end defaults have no layer of
  their own (a front end preferring X passes X explicitly, which enters at the *args*
  level and outranks config — the desktop did exactly that until it left in 0.5.0). The
  **CLI half is done** (2026-08-31): argparse passes `None` for every `run_pipeline`
  parameter and the runner drops unset keys, so the signature is the single source of
  truth. `test_option_defaults.py` holds the ratchet (empty; one new copy turns it red)
  plus two **strict xfails** marking the two missing layers — implementing either makes
  them XPASS, which is the reminder to delete the marker and update the divergence table.
  `README_DEV.md` → 开发原则 has the chain, the two admissible mechanisms, and that table.
- **Resume continues a task's own artifacts and past choices**; current parameters do not
  retroactively redefine finished stages. Want a clean parameter snapshot → start a new
  task. Artifacts expire only on **corruption, identity mismatch, or an unreadable
  contract** — never merely because a parameter differs. Invalidate only on a **shape**
  change: resuming straight through would **error out or leave data unreachable** (a cursor
  indexing a different interval list, an unreadable schema, a contract the current parser
  rejects). "A parameter differs, so recomputing would differ" is **not** a shape change,
  and there is **no separate cascade rule** — regenerating an upstream does not by itself
  expire a complete, still-consumable downstream (the pipeline is demand-driven; it would
  not have rerun that upstream at all). Keep the two fingerprints apart: a **provenance
  fingerprint** records what an artifact was produced with and only *warns* on mismatch;
  a **compatibility key** holds solely what errors or loses data when it mismatches.
  Runtime parameters default to provenance — `vad_silero_assist` was migrated out of the
  `*-vad.json` compatibility key in 2026-08 and is the worked example. Owner doc:
  `README_DEV.md` → 复用的依据是任务身份.
- GPU (>=4GB VRAM; five tiers `cpu`/`entry`/`standard`/`standard_large_vram`/`high`, default `auto`) is expected; CPU fallback must print `Warning:` to
  stderr. Don't assume GPU is unavailable without checking. **All of "can this machine use the
  GPU" lives in `speech/runtime/device.py`** (`resolve_device` / `cuda_usable`): it also catches
  a card too old for the installed torch, which `torch.cuda.is_available()` reports as usable —
  never decide device placement from a bare `is_available()`. User-facing supported-card table
  and fallback behaviour are in `docs/manual/resources.md` (README.md keeps one line). CPU fallback works only because the patched CT2 wheel uses **oneDNN** for
  CPU GEMM — Ruy deadlocks in the model destructor, MKL costs 3.4GB extra and is Intel-only,
  and all three failures surface only when something actually decodes. Read
  `tools/wt_refine_port/ct2-patches/README.md` before touching the wheel.
- **Artifacts** go in one of four ignored directories, never scattered at the repo root:
  `assets/` (raw media, see `assets/index.md`), `data/` (hand annotations, refined subtitles,
  caches — **never cleared by a rerun**), `out/` (generated), `tmp/` (scratch). Pipeline stages
  skip on **existence only** (no content validation) — to force a rerun, delete the stage output
  and everything downstream. Running from a checkout also puts runtime state at the root
  (`.state`, `.env` / `.env.lock`, `agent-sessions.jsonl`, `.task-activity/`, `cache/`); that is
  the checkout standing in for `%LOCALAPPDATA%\FineSub` (see the resources doc). It is ignored,
  but do not look for it under `out/`.
- Dependencies live only in `pyproject.toml`; never create `requirements.txt`. Production
  entrypoints call functions directly (no subprocess).
- The knowledge base `knowledge/` is NOT tracked by the main repo. Since 2026-08-22 its source of
  truth is `knowledge/knowledge.sqlite` (versioned rows, one revision per apply); `knowledge/rendered/`
  is a derived but **editable** markdown projection (edits are harvested at the next correction
  run — `docs/manual/knowledge.md`), and the old per-entry markdown + embedded git is an archive
  that gets imported once and never read again. Prompt templates `src/finesub/llm/prompt_templates/*.md` ARE tracked — prompt text is never
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
  Publishing goes through `scripts/publish-main.ps1` — it snapshots `dev`'s tree onto `main`,
  pushes to the throwaway `ci-gate` branch, and moves `main` only once CI is green. **`main` is
  releases and checkpoints only** (owner 2026-09-04): a checkpoint (the default message, nothing
  tagged) is a rung the next publish **replaces** (`--force-with-lease` on exactly that commit),
  so at most one untagged commit ever sits on top of the last tag. Anything a tag points at is
  never rewritten; a red gate is fixed on `dev` and re-run, never by hand on `main`. The snapshot is `dev`'s tree
  minus `$PrivatePaths` (`.claude/`, `docs/archive/`, `docs/report/`, less `$PublicExceptions`),
  so **un-ignoring anything means adding it to `$PrivatePaths` in the same change**
  (enforced by `test/test_publish_filter.py`). CI runs on the filtered tree, so a public tree
  that needs a stripped file goes red on the gate rather than after publication.
  Tagging and the rest of a release: the `release` skill. Never merge orphan `main` back into
  `dev`; back up full history via a private remote or bundle — the public repo is not a backup.
  Worktrees branch from local HEAD, not `origin/main`: `worktree.baseRef: "head"` lives in
  `.claude/settings.json` (tracked on `dev`, stripped from public snapshots). The setting only
  accepts `fresh`/`head`, so be on `dev` when creating a worktree, or pass an explicit base.
  Worktrees are sibling directories `../asr-playground-<topic>`, not `.claude/worktrees/`.

## Architecture map

模块 → 谁拥有它的文档 → **动它之前必须知道的**。描述性细节都在 owner 文档里，这张表不复述；
它只回答「我要改这里，该读哪一份、有什么是不能碰的」。带 ✱ 的约束由测试固化，改错会红。

### 管线骨架

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `pipeline.py` | `README_DEV.md` 产物清单与路径 | **唯一入口与前端**：一个源与 N 个源走同一条路（N=1 只是呈现档位）。选项面只有一份——manifest 行可覆盖的键由 `run_pipeline` 签名导出，别再手抄第二份清单。✱ **它是 `-m` 的目标，所以 src 里任何东西都不得 import `finesub.pipeline`**（那会让模块体执行两遍、出现两份 `run_pipeline`）；要引擎就 import `stages`。✱ **argparse 对每个 `run_pipeline` 参数一律传 `None`**，未给的键被整个丢掉、由签名回答——别再在命令行侧写第二份默认值（`test_option_defaults.py` 的棘轮已清空，加一个就红）。批次的磁盘状态在 `batch_state.py`，不在这 |
| `stages.py` | `README_DEV.md` 产物清单与路径 | 转换本身（`run_pipeline`）：所有 artifact 路径从最终 SRT 派生；阶段按**存在性**跳过（不校验内容）——要重跑就删该阶段产物**及其下游**。LLM 阶段靠 `--stage` opt-in。与 `llm/stages/` 不是一回事，后者是 LLM 纠错内部的阶段 |
| `batch_state.py` | `docs/batch-scheduler.md`、`README_DEV.md` 产物清单（四件+注册表）、`docs/manual/batch.md`（用户向契约） | 一个批次在磁盘上的状态：`out/batch/<id>/` 四件（`queue.jsonl` 运行器发布 / `control.jsonl` 用户追加 / `.control-cursor` / `.batch.lock`）加数据根的 `batches.json`（键 `(cwd, batch_id)`）。续跑**保留批次身份**，且那是唯一一处「显式命令行选项盖过行里记的值」的地方。✱ **它不认识任务是什么**——没有音频、模型、阶段；`control_intake` 要认合法选项，所以那一步由调用方以 `admit` 钩子给。分层 `pipeline → batch_state → scheduler`，方向由类型定死（它造 `BatchItem`、返回 `IntakePoll`），所以 `DEFAULT_BATCH_ROOT`/`STATUS_FILENAME` 留在 scheduler |
| `scheduler.py` | `docs/batch-scheduler.md` | 三 bin（download×2 / asr×1 / llm×2）的**领域无关**引擎——不认识管线选项，item 由 `pipeline.py` 构造。**asr 并发恒为 1**，每个文件独占整个 profile；呈现（每项 reporter、失败怎么显示）由调用方给钩子。`ItemResult.stage`/`view_state` 是「这项在干什么」的唯一真相（`status` 到终态前一直是 pending）；`IntakePoll.commit` 在指令**生效之后**才被调，持久游标靠它 |
| `run_metadata.py` | `README_DEV.md` | 计时/worker sidecar；分离器 model batch 恒为 1 |
| `reporting.py` | `docs/reporting.md` | 八个事件的契约与 quiet/normal/verbose。✱ **`stage_started` 的 `reused` 只有一个含义：这一趟什么都没跑**——产物侧的状态词表是三个（`executed`/`reused`/`skipped`），这里**没有**对应的第三态，唯一能跳过的阶段照样产出它的产物、只是换了条路，换哪条由 `detail` 说（别反过来把 `skipped` 折进 `reused`）。✱ 管线代码**不得裸 `print`**（`llm/` 由 AST 守卫钉着，产物正文用 `# product output` 逐处放行）；线程池必须带 `initializer=bind_reporter` |

### speech（高危区）

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `speech/recognition/transcribe.py`、`speech/preprocessing/energy.py` | `docs/asr-align.md`、`docs/vad-energy.md` | **两个高危核心**。改参数要有**质量**论证 + 测试，或一份实验记录；放弃一致性证明时须**预先**写明替代指标与门槛（见上「Key facts」）。组批预取层 `DecodePrefetch` 只在 `decode_batch>1` 时介入、只批单窗 group、循环本身不动；**默认 1**——12 份产物实测 1.06×，不到预注册的 1.15×（`bench-baselines.md` 二十二） |
| `speech/runtime/device.py` | 上方 Key facts 的 GPU 条 | 「这台机器能不能用 GPU」**只在这里**，「这张卡多大」（`total_vram_gib`）也是。永远不要用裸 `torch.cuda.is_available()` 决定设备 |
| `speech/preprocessing/separator/` | `docs/separator-optimization.md` | 四模块同住（stage + 编译缓存 `accel` + AOTI package + `demix` 块推理）。已做过的实验别重做。交付形态由输出后缀二选一（`.ogg` = 16k 单声道 ASR 轨 / `.flac` = 无损），其余后缀报错。✱ **`-vocal.ogg` 有两个生产者**：分离，以及 `--no-separate` 走的 `encode_asr_delivery`（输入已是纯人声时由源音转码，同规格同路径）——所以下游、存在性跳过与 resume 都不需要「没有人声轨」这个分支；两者的临时解码文件都是**成功才删、失败保留**（`ensure_decodable_input` 的契约），差别只记在 metadata 的 `status`。`demix` 取代了 audio-separator 的文件到文件入口——**改它之前先读它的模块 docstring**，那层外壳还替我们做过 autocast 与 mono→stereo |
| `speech/preprocessing/energy.py` 的**绝对 dBFS 两档** | `docs/vad-energy.md` 第 6b 步 | 与 `MIN_SPEECH_PEAK_DB` 是**两个量**：那个是自适应加权 `energy_db` 的峰值，这一对是真 `frame_dbfs` 的**峰值与功率均值同时**低于门限（实测两种峰值中位差 30.8 dB，别合成一个常量）。丢弃档 −60/−70 **默认开、区间不进解码器**；可疑档 −35/−45 **只打 `vad_level_tier` 标记**，是否变成推理跟随 `--qwen-verify`。✱ 两个条件缺一不可（耳语动态范围被压扁，单看峰值会吃掉真耳语），且三个 interval 生产者都必须调用——两条都有测试钉着 |
| `speech/preprocessing/spectral.py` | `docs/vad-energy.md` | 加权能量信号**同时**被 VAD 与 `recognition/transcribe.py` 读；`audio.py` 只管解码与切片 |
| `speech/recognition/vad_asr_stage.py` | `docs/vad-asr.md` | 写 `*-aligned.json`——**字段契约在那份文档里**，含全局 DP 重分句与 `whisper_segment_start` 标记 |
| `speech/postprocessing/segmentation.py` | `docs/segmentation-split.md` | 全局 DP 打分（可切可并）；幂等要求 |
| `speech/postprocessing/stabilization.py` | `docs/asr-stabilize.md` | 写 `*-stable.json`；profile 与 resume 规则 |
| `speech/recognition/word_starts.py` | `docs/asr-align.md`「词首修正」 | `[*]` 块解析 + VAD 锚定的词首 clamp |
| `speech/recognition/{segments,checkpoint}.py` | `docs/vad-asr.md` | 时间轴自洽；checkpoint 是**可丢弃**的 ASR partial |
| `speech/verification/qwen_referee.py` | `docs/vad-asr.md` | `--qwen-verify` 的第二模型证据，stabilize 消费它。保持三类嫌疑与 segment ±0.1s 探测跨度。解码是 **launch-bound**（每步 60 ms、GPU 核 5 ms），所以 `transcribe_batch` 成批（`plan_batches`：≤16 条且 `条数×最长`≤120 s，显存由此封顶）。pacing ≥150 s 且 VRAM ≥3.5 GiB 时走 `qwen_decode.FixedShapeDecoder` 的定形编译解码步（每步 65→9 ms；一张 graph 覆盖所有长度，编译批 16 条）。✱ **别换回 transformers 的 `cache_implementation="static"` 自动编译**：它的 2-D mask 每 token 长一列、每个长度录一张 CUDA graph（全片复核 11 GiB、不比 eager 快，`bench-baselines.md` 21.6）；Windows 两个 inductor 开关与分离器同源；`close()` 必须走 decoder 的重置（Dynamo 代码缓存 + graph 池），否则每实例留 3 GiB。✱ 两条路都不 bit-exact，验收看 stabilize 决策一致率；改 batch/cache 常量前先读 `bench-baselines.md` 二十一 |
| `speech/recognition/lang_redecode.py` | `docs/asr-align.md`「语言票翻转重解」 | `--lang-redecode` 的解码循环内强制语言重解（默认 auto）。采纳判据与历史回滚有测试固化；**阈值未标定、负例为零**，改它们前先按该节「待标定」补真外语素材 |
| `speech/recognition/lang_audit.py` | `docs/asr-align.md`「全局语言审计」 | 上一条的判据是**相对的**（比滚动众数），众数整体错时结构性失明；本模块是那个**外部锚**（抽样交 Qwen 重认、比时长加权众数）。**判据无阈值、只报警不动手**。四格模式的唯一真相是 `resolve_mode()`；随 `--lang-redecode on`，**默认 `auto` 不开**（一轮 16.9 s）。✱ **抽样账本是 run 状态**：随 ASR checkpoint 走（`lang_observations`），且**记的是发出去的那个语言**（重解被采纳后要改写）——两条都由测试固化，动它之前先读 `docs/asr-align.md`「两条 resume 契约」 |
| `speech/runtime/{resources,gpu_stage_gate}.py` | `docs/gpu-profiles.md` | GPU 档位只决定**分离器实例数**与语言裁判放哪（`referee_device`）——**ASR 永远单 worker**；分离器与 WT 模型族不共存。✱ 档位是**显卡等级的名字**（`entry`/`standard`/`standard_large_vram`/`high`），外加一个**策略档 `cpu`**（`TierSpec.gpu=False`，「这次不用 GPU」，与「问过说不行」是两件事；`auto` 只在**驱动报不出 CUDA 设备**时选它——卡老到本 build 没 kernel 仍落 `entry`，因为 CT2 可能还跑得动。✱ 策略必须真的到得了权重：分离器不走 `resolve_device`，`run_vocal_separation` 把「`--device` 的请求 ∧ `profile.gpu` ∧ `cuda_usable()`」折叠一次再传下去（`--device cpu` 是对整条任务的承诺，分离是它第一个也是最重的阶段）。✱ `--gpu-tier cpu --device cuda` 报错，两个前端各有一处检查点——CLI 在 `pipeline.py` 的 `_pipeline_kwargs`、桌面在 worker 调 `run_pipeline` 之前；两处的共同前提是 `device` 能取 `None`，否则分不清显式与默认，会把裸 `--gpu-tier cpu` 也拒掉），各带**一个**显存要求，要的是**空闲**量（3/6.5/10/10 GiB）而非卡的容量——把后者当要求正是「我的卡有 8GB 所以装得下」变成 OOM 的那条路。卡的容量只在 `tier_for_vram` 里出现一瞬（减 `reserve_for_capacity()` = `容量/8 + 0.5` 后匹配），之后没有任何地方见到它——那条曲线是**选来正好穿过档位表**的（4/8/12GB → 3/6.5/10），所以匹配用 `<=`。默认 `auto` 读驱动定档，**先四舍五入到整 GiB、减预留、再取能满足的最大档**（驱动从不报标称值，直接向下取会把每张卡降一级），**最后在 `standard_large_vram` 处封顶**——`high` 的第三个分离 worker 实测更慢，只有卡报 ≥24GB 才放行；**无显卡落 `cpu`**，而「有卡但本 build 认不出」与「显存不足」落 `entry`（前者留给 CT2，后者只告警不降档）。✱ **`standard_large_vram` 与 `high` 的 `usable_gib` 相同（10.0），所以表的顺序有语义**：`tier_for_vram` 取最后一个放得下的档，前者必须排在 `high` 之前（测试钉着非递减排序）。它存在是因为**实例数与显存回答的是两个问题**——实例数是吞吐取舍（两个是峰值），显存是裁判花的预算（编译解码步要 Whisper 池之外 3.5 GiB）；旧表把两者绑死，12–16GB 卡只能在「两个 worker 但预算 6.5」和「预算 10 但第三个更慢的 worker」之间二选一。**没有更宽的档**：E7 实测两个 worker 就是吞吐峰值，3/4 个更慢 |
| `speech/runtime/cuda_libs.py` | `docs/ct2-distribution.md` 的 cuBLAS 条 | CT2 按名字加载 cuBLAS，这里在建模型前把它的目录加进搜索路径。✱ **不得 import torch**（会把刚去掉的导入顺序依赖写回来）；调用必须在 `super().__init__` 之前，源码守卫钉着 |
| `speech/recognition/fw_refine_backend.py` | `docs/wt-refine-handoff.md` | 模型池 + patched-CT2 adapter。换 wheel 前先读 `tools/wt_refine_port/ct2-patches/README.md` |

✱ `speech` 不得 import `llm`。

### 无依赖的公共层

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `media/` | — | URL/下载选择、ffmpeg/ffprobe、剪辑。✱ 无 speech / LLM 依赖 |
| `subtitles/` | `docs/segmentation-split.md`（分句口径） | SRT 模型、对齐、指标、后处理、渲染。✱ 无 speech / LLM 依赖 |
| `config.py` | `docs/manual/model-routing.md` | 共享 `config.toml` 的**定位/解析/记忆化**，仅此而已。stdlib-only 且与领域无关——各域校验自己那张表。**没有写入器**：`config.toml` 只有手改一条路（桌面端的保留注释写入器 2026-09-04 随它删了，owner 决定；要 `finesub config set` 时从 `0.5.0pre` 拿回来重做） |
| `paths.py` | `README_DEV.md`「运行时路径解析契约」 | 唯一的仓库/运行时路径 resolver。✱ `src/` 里别处不得用 `parents[N]` 找仓库根 |
| `workflows/reference_ingest.py` | `docs/knowledge.md` | **默认全执行**（用户主动发起即 opt-in），与 LLM 层 dry-run 默认相反 |

### LLM harness（`llm/`）

| 子域 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `routing/` | `docs/manual/model-routing.md`（用户）、`docs/llm_design_notes.md`（取舍） | ✱ 层次单向：catalog → routes → execution_policy → model_router → client，`config` 只被读。**policy 只是 backend 闸门**，agent 靠成为模型组成员参与。`model_catalog.psv` / `model_routes.toml` 与代码同目录、进 package-data。✱ **两个「组内最小」不是一回事**：`group_planning_envelope` 是规划包络（输入已扣掉输出，供 `planning_limits_for`），`group_declared_minima` 是 catalog 两列的原值（供 `check_model_group_windows` 的窗口下限闸门，在 ASR 之前跑）。混用会把「总量受限但形状健康」的模型误判——owner 2026-09-03 为此裁定 haiku 放行，`docs/plans/field-feedback-batch-plan.md` §2.1.1 |
| `client.py` · `llm_runtime.py` · `provider_transports.py` · `http.py` · `media_upload.py` · `rate_limit.py` · `content_filter.py` | `docs/llm_harness_behavior.md`、`docs/provider-adapters.md`、`docs/manual/model-routing.md`（`[llm] proxy`） | Gemini REST + OpenAI-compat/Anthropic 纯文本 adapter；RPM/TPM 限流；PROHIBITED_CONTENT 阶梯。✱ **`media_upload.py` 是 `UploadedFileRef` 与 Files API 上传的唯一去处**（2026-09-03 从 `client.py` 拆出）：方向单向 `client → media_upload`，上传侧**不得认识模型选择**（RoleClient / transport / 谁来应答）。⚠ 它**确实**读两处 routing，且只读这两处：谁的 key 上传（`routing.api_keys`——Files 对象只属于创建它的那把 key，别的 key 403，而急切上传发生在还没有候选之前）、以及某个 policy 意味着哪种载体（`window_media_ref`）。⚠ 拆完之后**测试替身要跟着搬**——`window_media_ref`/`upload_gemini_file` 在 `media_upload` 的全局里查名字，而 `RoleClient._uploaded_media_ref` 在 `client` 的全局里查，patch 错模块不会报错、只会失灵。✱ **HTTP 客户端只在 `http.py` 造**——`llm/` 里别处不得出现 `httpx.Client`（默认参数里也不行，源码守卫钉着），否则 `[llm] proxy` 到不了那个出口。代理**只管 API 不管下载**：写 `os.environ` 会被下载族的路由发现读到，所以是显式工厂 |
| `agent/` | `docs/llm_local_agent.md`（**§12.1 是「什么已接线」的唯一入口**） | ✱ 模块名保留 `agent_` 前缀——`finesub_bootstrap.shell` 用**字符串** `python -m finesub.llm.agent.agent_cleanup` 调它们，改名不炸 import、只在运行时炸。传输由会话档位派生、**没有配置开关**（形态见 `docs/llm_agent_tool_protocol.md` §1，driver 接线见 §6）。`agent_mcp_server.py` 由 CLI 作为子进程拉起；新 validator 必须注册在 `agent_validators.py`、参数必须可 JSON 化 |
| `stages/correction/` | `docs/prompt-iterate.md` | 八模块：`run` 规划+收尾 · `serial`/`parallel` 两 driver · `attempts` 每窗重试拆分 · `query_round` · `context` · `commit` 断点与失效判据 · `metadata`。**测试替身放做名字查找的那个模块**，不要放包 `__init__`（`test/conftest.py` 的 `setattr_correction`） |
| `chunking.py` · `token_budget.py` · `token_truncate.py` | `docs/llm_harness_behavior.md` | 三级 token 计数：本地二进制 → 免费 `countTokens` → 启发式上界 |
| `prompts.py` · `prompt_compose.py` · `prompt_variants.py` · `prompt_templates/` | `docs/llm_prompts.md` | ✱ **prompt 文本从不硬编码在 Python 里**；合并阈值**只在** `prompt_constants.py` 写一次；示例由 `example_builder.py` 生成，golden 快照按 `PROMPT_VERSION` 锁 |
| `output_tags.py` · `session_contract.py` · `output_protocol.py` | `docs/llm_harness_behavior.md` | 每会话输出契约的唯一真相，生产与 replay 校验共用；纠错的 CSV 轮不在 `session_contract` 里，归 `output_protocol` |
| `research.py` · `search_loop.py` · `web_search.py` | `docs/llm_harness_research.md` | 联网检索一律由本地检索代理执行（Exa → Gemma4 grounded → Tavily，**无免 key 兜底**），纠错/调查模型不直开 `google_search` |
| `knowledge/` | `docs/knowledge.md` | ✱ 不依赖 `stages/`（单向：stages 读写知识库）。`knowledge/` 本身不被主仓跟踪。子包 `knowledge/node/` 是真相源（SQLite 版本行 store、preset、三投影、无损导入、overlay+CAS apply 引擎、`signals.py` 事件/证据遥测；`docs/plans/knowledge-node-plan.md` §8 已全部落地）；`base.py` 只剩读 API 门面、写锁与 task artifacts；`maintain.py` 是人工维护 CLI（`python -m finesub.llm.knowledge`，含 `ingest` 材料吸收；2026-09-01 从 `node/cli.py` 上移）；`report.py` 只读信号报告；子包 `knowledge/share/` 是共享协议（bundle/snapshot 交换 + 不可信边界、sync 合并、stdlib server——local_id 不上线、维护者 verdict 才入库）。同名 subject（两文件同 H1）按 **mtime → 长度 → 路径** 排序，落选的由 phase-b 并进胜出者的「待归类」并退休——⚠ mtime 打平是常态（clone/解压抹平时间戳），实际多是长度在决定。✱ **迁移遇到不合语法的部分不加特判去对齐**：导入器原样保存（未知小节的行当 `note`、保留原名，parity 仍逐字节），phase-b 泊进「待归类」，再由 `ingest`/`repair` 蒸馏（owner 2026-09-01）；pull 没合上的字段落 `share-conflicts.jsonl`，`share conflicts [--repair]` 消费它——身份是**分歧本身**，`dismissed` 粘住而 `resolved` 不粘。✱ 模型看到的一律是 prompt 投影（注入面裸行、任务后更新带 `@k` 句柄）；人读/可编辑面是 `rendered/` 缓存（markdown bullet 行，回写剥前缀）；事件/证据永不进 prompt。行文法是 `[标记] 行体`（冒号无语法作用；**在允许术语行的节里**行体 ≥4 段竖线即术语行，只收 note 的节里竖线只是字符——2026-09-02）；一行 = 一个 node，无续行、无转义换行；`preview="full"|"partial"` 与 mode 正交，按**这一面能不能写**分档——完整预览（用户、以及知识更新任务，agent 前端即 `kb_tools=propose`）渲染空节、preset 生成的说明注释与 core 空槽，部分预览（其余只读 LLM 任务，含它们的 `kb_read`）三样都不渲染 |
| `prompt_artifacts.py` · `task_report.py` · `exchange_log.py` | `README_DEV.md` 产物清单 | 产物与可读日志 |

### 装机与前端

| 模块 | Owner 文档 | 动它之前 |
| --- | --- | --- |
| `finesub_bootstrap/` | `README_DEV.md`「任务目录的清理与保留」（产物清单）、`docs/ct2-distribution.md`「锁的重建」（两份 pylock）、`docs/manual/resources.md`（用户）、`docs/download-routes.md`（下载族） | ✱ **不得 import 主包**（`shell.py` 里唯一一处是函数内延迟 import）。`secrets.py` 与 `token_counter.py` ✱ **stdlib-only**——纯 `[harness]` 装机与薄 CLI 的 3.10 都会 import 它们；包 `__init__` 必须保持 import-free。`secrets.py` 是本项目**唯一**的 `.env` 解析/写入器 |
| `cli/` | `cli/README.md` | 薄 launcher + `_vendor` 源码快照；唯一入口是 `finesub`。构建清单有离线守卫。新版本提醒接在这里（`finesub_bootstrap/update_check.py`，只有它才是 PyPI 上那个 `finesub` 发行版）：查 PyPI `info.version`、**只认正式版**，非正式落法（`0.5.0rc1` / GitHub prerelease）因此天然不被推荐 |
| `tools/` | `tools/README.md`（总索引）+ 各自的 README | **只按需维护**——不要作为其他改动的副作用去更新它们。例外：改名/移动类改动必须同步 `tools/session_replay`。⚠ 13 个子目录只有 4 个活跃（`bench`/`session_replay`/`segmentation_gold`/`wt_refine_port`），`tokcount` 是生产组件而非工具，`split_explorer` 与三个散落文件已跑不起来或零引用——**动之前先看总索引那三类**；这里的 16 个 `test_*.py` 默认永不执行（`testpaths` 不含 `tools/`） |
| `legacy/` | — | 本地 gitignored 目录，不随仓库发布；不要在它上面建东西 |

## Docs index (read on demand)

完整清单与状态在 `docs/README.md`（文档地图）；下表只留 agent 做任务时的**判断提示**：
唯一入口、状态异常、踩坑。按「任务涉及什么」分段；段内顺序即阅读顺序。

**怎么读这堆文档**：分界是读者不是主题（`manual/` 使用者 / `docs/` 根开发者），
见 `docs/README.md`。改了用户看得见的行为，就在 `CHANGELOG.md` 写一条。

**面向使用者**（用户向入口是 `README.md`；manual 主题见 `docs/README.md`）：
`docs/manual/env.md` `docs/manual/repo-install.md` `docs/manual/resources.md` `docs/manual/batch.md`
`docs/manual/ct2-wheel.md` `docs/manual/models.md`（模型选择：ASR 三档 Whisper——**large-v3 无优势、日语微调实测打平**、不可换的分离器/第二模型、各 LLM 后端印象）
`docs/manual/model-routing.md` `docs/manual/agent.md`
`docs/manual/knowledge.md`（知识库用户向：改内容的四条路、三档 dry-run、共享与冲突）
`docs/manual/outputs.md`（产物与 `--stage` 六档、`-annotated.csv` 九列、想重跑该删什么）
`docs/manual/tuning.md`（用户向旋钮总表：识别侧与 LLM 侧各一张，含默认值与「改完删什么」）
`docs/manual/troubleshooting.md`（症状 → 那一页的总目录；⚠ 它只做路由，别把内容搬进去）
——用户向描述以 manual 为准，别拿 dev 侧细节去纠正它。⚠ **加了用户能给的选项就要在
`tuning.md` 落一行**（或说明为什么它不该出现在用户面前）：2026-09-02 审计时，54 个 CLI
选项里有 20 个在全部用户向文档里 0 次出现，唯一入口是英文 `--help`。

**agent 任务说明**：`agent-tasks/`（一个子目录一件事，主文件 `SKILL.md`）。⚠ **不再依赖 harness 的 skill 注入**（2026-09-01）——没有谁会自动把它们塞进上下文，手上的任务对得上就自己整份读完再动手；清单与写作约定在 `agent-tasks/README.md`，用户向的能力清单在 `docs/manual/agent-tasks.md`。

**开发与维护总入口**：`README_DEV.md`（dev principles、canonical artifact tree、
reuse/resume 规则、agent checklist）`docs/testing.md`
（markers、scoped commands、覆盖）`docs/data-index.md`（数据与基线索引的规则半边；逐条清单在本地 `data/index.md`，找数据先看这两份）
`docs/bench-discipline.md`（一个数字算数的六个条件；**动性能前先读这份**，它短）
`docs/bench-baselines.md`（换机后的本机基线与实验记录；节号从「二」起，**一律不重编号**）。

**语音链路（VAD / ASR / 分句 / 分离器）**：`docs/vad-energy.md` `docs/vad-asr.md`
`docs/asr-align.md` `docs/asr-stabilize.md` `docs/segmentation-split.md`
`docs/segmentation-gold.md` `docs/gpu-profiles.md` `docs/separator-optimization.md`
`docs/batch-scheduler.md`（三 bin/队列面/失败隔离的 owner）
`docs/speech-followups.md`（**speech 侧未完成工作的唯一入口**：五项没开工 + 两项默认关着 + 各批「不要做什么」
摘要 + 散在别处的未做项。⚠ 它只是索引，状态的真相源是 `docs/plans/crispasr-followups.md`
的「状态总览」表——改状态先改那份）
`docs/wt-parallelism.md` `docs/wt-refine-handoff.md`（入口/结论）
`docs/wt-refine-port.md`（算法契约）`docs/wt-refine-validation.md`（质量实测）
`docs/gemini35-transcribe.md`（外部转录模型 `gemini-3.5-transcribe` 的调用规则，调查/仲裁用，非生产链路）
- 单文件 WT 分片**已移除**（回溯点 `dev` 的 `1fcc4e1`），别按 wt-parallelism 实现
- 动分离器性能/并发前先读 separator-optimization，避免重复已做过的实验
- 审计分割质量、动机械指标前先读 segmentation-gold

**LLM harness**：`docs/llm_harness_behavior.md` `docs/llm_harness_routing.md`（路由 dev 侧）
`docs/llm_harness_research.md` `docs/llm_local_agent.md` `docs/llm_agent_tool_protocol.md`
`docs/llm_local_agent_experiments.md`（会话复用实测）`docs/llm_local_agent_runtime.md`（现场与清理）
`docs/llm_local_agent_agy.md`（agy 专属）`docs/llm_prompts.md`
`docs/llm_design_notes.md`（架构意图与决策台账）`docs/llm_followups.md`
`docs/plans/conversational-live-test-plan.md` `docs/knowledge.md` `docs/provider-adapters.md`
`docs/prompt-iterate.md`（迭代方法论）`docs/session_replay.md`（replay 工具契约）
`docs/merge-calibration.md`（合并门槛标定）`docs/plans/knowledge-node-plan.md`
- 运行时总入口 llm_harness_behavior；**接线唯一入口 `docs/llm_local_agent.md` §12.1**，别从别处推断
- 未完成实验/设计唯一入口 llm_followups；别从归档历史推断现状
- **任务级并行（W1–W7）已于 2026-08-30 全部落地**：现行为在 llm_harness_behavior
  「任务级并行与 agent 槽位预算」与 llm_harness_research「超长素材分块调查」；设计依据与
  验收记录在本地 `docs/archive/task-parallelism-plan.md`；暂缓项只剩 conversational 按需并发
  （P7d 裁定不做，B' 回喂与研究块间并行已实施），在 llm_followups
- 动 agent 传输前先读 llm_agent_tool_protocol（两条总原则：不碰用户全局设置、
  不为过度保守的安全策略加机制）
- `docs/plans/knowledge-node-plan.md`：§8 与 §11 均已落地，读它是为取舍依据与 owner 锚点（§11.1），别当未竟计划
- 行文法 v3（`[标记] 行体`、preset v2、从 md 重导，2026-08-29）已全部落地：现行为在
  knowledge.md，**owner 决定的 17 条锚点与取舍依据在 llm_design_notes.md 的
  「知识库行文法 v3 的决策记录」**——动这些结论前先看那节

**分发、前端与跨前端**：`docs/reporting.md` `docs/download-routes.md`
`docs/ct2-distribution.md` `docs/cross-frontend-lease.md`
- reporting / download-routes 是 owner 文档，契约细节以它们为准
- 动多前端并发语义前先读 cross-frontend-lease（已完成；§4 记着三条不做的事及其理由）

**计划与决策**

- `docs/plans/stage-device-plan.md`——逐阶段设备解析 + 拆 `entry`/CPU。**已全部实施**。⚠ 动这一族代码的两条验收手法（进程内劈开 torch 与 CT2、`CUDA_VISIBLE_DEVICES=-1` 无卡回归）2026-09-03 起在 `docs/testing.md`「设备解析」一节——**那是纪律，按那份跑**；本文只留取舍依据：读它是为
  取舍依据（尤其「CT2 不能用 CUDA 就回退」为什么与 `device.py` 的哲学冲突），别据此推断代码里
  有什么。已核的现状在它的第 2 节，带锚点

- `docs/plans/docs-reorg-plan.md`——**`docs/` 自身的整理方案：2026-09-03 已全部执行**（此后只作取舍依据，现行组织看 `docs/README.md` 的地图）。
  读它是为了别重复那份盘点：索引已全覆盖、已死段落是有意保留的、以及为什么**不**做全面子目录化
  与重编号（全仓约 1100 处 `§N` 引用，523 处在 `.py` 注释里）。裁决三条：`crispasr-followups.md`
  **不改名不瘦身不归档**（改名要碰 27 个文件、19 个是代码），speech 侧另建薄的
  `speech-followups.md`，以及建 `docs/plans/` 收已结案的计划稿。动 `docs/` 结构前先看它的
  「明确不做」与第 5 节

- `docs/plans/refactor-followups.md`——2026-08 重构**已全部完成**（含挂发版的 tokcount 三步，
  0.4.0 时做完）。现在剩的是**押后项的理由**与做同类改动前先读的**五条教训**（第五条 2026-09-01 补：**守卫的扫描面本身就是守卫的一部分**，一天内同形撞三次）。
  计划正文在本地 `docs/archive/`
- `docs/plans/crispasr-followups.md`——2026-08-29 外部对照（CrispASR）后的待办计划，六批
  （守卫 / 测量 / 加速 / 契约 / 新能力 / diarization 工程层），**2026-08-31 合入 `dev`
  时大部分已完成或结案**。⚠ **现行状态只看它的「状态总览」表**，别从批次名推断：
  **五项没开工**（A6 的 CT2 wheel、P16 文本侧 LID、分离器耳语救援接线、
  P11–P15 diarization 代码，加 2026-09-02 新增的「第二模型否决救回的文本是错的」——
  那一项的**判据一半 2026-09-03 已结案**（49 段听审 41 对 / 8 错，默认行为不动），
  留在表里的是它的**镜像一半**；它不在原 28 项里，最容易被按旧数字漏掉）；**另有两项交付了但默认关着、翻默认还欠证据**：
  A1 的组批（门槛未过，`bench-baselines.md` 二十二）与 P9 的 `--asr-context`（差一次逐段
  听审）——这两项最容易被当成做完了，因为功能就在代码里。其余是**已完成**或**已结案不做**
  （P18 是后者）。清单在 `docs/speech-followups.md`。
  ⚠ 每批都带「不要做什么」，动手前先读那一条；证据在 `bench-baselines.md`，
  对照过程在本地 `docs/report/`
- `docs/plans/translation-style-plan.md`——把 `common-mistake` / `good-example` 两个翻译台账折成
  一类知识库条目（category `style`）。**主体已落地（2026-09-02）**：一个 style = 一个 entry，
  约定是它的行；取用 `--style`、收录只在 refined 档、上限在 preset 的节上。**现行行为看
  `docs/knowledge.md` 的「style 类别」**，这份计划留的是**取舍依据与 owner 决定台账**。
  ⚠ **六步都已结案**（2026-09-02）：§4 的实验判定不做、N/L 已拍定（每节 20 行 × 200 字 +
  整条 6000 token 兜底），存量迁移做完（27 条范例迁入、27 条错误库判定不迁）。没做的只剩
  共享那一次跨库验收，和 §6 的两条未决：style 没有 `landed` 信号、删除权与证据
- `docs/plans/model-window-limits-plan.md`——窗口限额三档化：catalog 加 `context_window` 列记厂商值，
  删掉 `routing/config.py` 里把 `DEFAULT_LIMITS.prompt_input_limit` / `output_limit` **当上限用**
  的那两处 `min(...)`（文中代称 `HARNESS_INPUT_CAP` / `HARNESS_OUTPUT_CAP`，⚠ **代码里没有这两个
  标识符，grep 不到**）以及 `context_limit` / `safety_margin` 两个字段，`derive` 只剩两行算术。**已实施（2026-09-03）**，现行行为以代码与 `manual/model-routing.md` 的
  列说明为准，读这份是为取舍依据。§5 是验收（audio 与低清 video
  必须逐窗相等），§6 是为什么 P6 标定这次不阻塞、以及**关掉质量护栏后它就阻塞了**，
  §7 是一份独立的 catalog 可疑值审计（13 个 194000 里哪些是厂商值、哪些是默认值误填、哪个是 owner 有意保留的速率近似）
- `docs/plans/desktop-split-plan.md`——0.5.0 把 `desktop/` 移出本仓的实施计划。
  **阶段 A、锚点 `0.5.0pre` 与阶段 B 均已完成（2026-09-03）**，只剩阶段 C 发版；此后读它是为
  取舍依据与实施记录（§9），别据此推断仓库里还有 `desktop/`。剥离前的最后一份在公开 `main`
  的 tag `0.5.0pre`。仍然有效的三条：两份 pylock 与 `runtime-manifest.json` 住在
  `src/finesub_bootstrap/`，`VERSION` 在仓库根；`ci.yml` 的 `windows` job 是**唯一** Windows
  lane（`cli/tests`、DPAPI、junction、wheel 构建只在那里真跑）；`[runtime]` extra 只为编译锁
  而存在。✱ 运行时 marker 记的是锁的**内容**摘要（`lock_content_digest`，去注释、去行尾），
  `_LEGACY_LOCK_FILE_DIGESTS` 替 0.4.x 安装接住整文件哈希——**下次真正重新生成锁时删掉它**。
  §8 第 5 条已裁：`config_file.py` 删（2026-09-04）

- `docs/plans/field-feedback-batch-plan.md`——一轮用户反馈带出的五项。**已全部实施**
  （2026-09-03 当天起草、两轮复审、落地；§9 是实施记录，含四处偏离计划的地方）。⚠ 两处最容易做错：**§2 的闸门必须扫 `model_groups` 而不是 catalog**
  （`gemini-free-gemma-4-31b` 的 `max_input_tokens=16000`，但它只是 grounded search 的 target、
  不在任何组里，扫 catalog 会把它判死），**§3 的「文本描述」是状态附带的一句话、不是 prompt
  正文**（正文在 `exchanges/`，日志再来一份就是几十 MB）。§1 附带查出一个独立缺陷：
  `is_mirror_failure` 不认 401/403（⚠ 准确说是**跨进程**那一半：下载在子进程里，异常只剩文本，
  进程内本该由 `httpx.HTTPError` 分支救下）。§6 明确不做五条，§7 六条 owner 决定（**无未决**），
  §8 两轮复审记录。⚠ **§2.1.1 是最容易做错的一节**：闸门比 catalog 的 `max_input/max_output`
  两列，**不比 `group_planning_envelope` 的规划包络**（owner 裁定 haiku 放行——总量受限但形状健康）

索引里没写、要翻代码才找得到的两处：

| 找什么 | 去哪 |
| --- | --- |
| **纠错窗口的 CSV 输出契约**（列名、列数、按位置解析、`note` 里的 `\|` 怎么保住） | `docs/llm_harness_behavior.md`「输出协议」一节 + `src/finesub/llm/output_protocol.py`（header 常量的唯一定义处）。产物侧 `<stem>-annotated.csv` 的字段含义见 `README_DEV.md` 的产物树 |
| **哪些产物是记录、哪些可删、谁来删** | `README_DEV.md`「任务目录的清理与保留」（含**故意不删**的两类：URL 输入下载的源媒体、`-annotated.csv`/`-corrected.srt`，以及「清理今天没有调用者」）；清单在 `finesub_bootstrap/artifacts.py` |

`docs/archive/` 与 `docs/report/` 为本地笔记：**在 `dev` 上被跟踪**（worktree 与 clone 都拿得到），
但由 `scripts/publish-main.ps1` 从每次公开快照里剥掉，因此不随仓库发布、也不进索引；
迁入前按上方 **Archive extraction** 规则抽非过时信息。
