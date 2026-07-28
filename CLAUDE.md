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

Production entrypoint: `src/pipeline.py`. An experimental LLM correction/translation
post-processing layer lives in `src/llm/` but is **not** part of the default production flow
(default `--stage` stops at `raw-srt`; `translated-srt`/`final-srt` opt in).

## Commands

```powershell
pip install -e ".[asr]"              # GPU ASR stack
pip install -e ".[harness]"          # LLM layer only (~4GB RAM, ffmpeg)
pip install -e ".[asr,harness,dev]"  # full runtime + pytest

# Full pipeline (default output out/<stem>/<stem>.srt; artifacts grouped under out/<stem>/)
python src/pipeline.py data/input.wav --model large-v3-turbo --language en --gpu-budget-gb 8

# Batch over many sources: download(x2) -> asr(x1) -> llm(x1, ordered) stage-parallel;
# per-item failure isolation; events at out/batch/<id>/batch-status.jsonl
python src/batch.py --manifest tasks.jsonl   # or positional URLs/paths

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
  `llm.reference_ingest` executes everything by default (user invoking it is the opt-in).
  The Gemini `countTokens` endpoint is completely free (auth-only key) — planning can call it
  freely; with the local tokenizer binary (`bin/windows-amd64/tokcount.exe`) even that is offline.
- `gemini/gemini-3.1-flash-lite` supports native thinking on both Gemini Free and Paid; the prior `supports_reasoning=false` catalog entry was a verified misclassification (2026-07-12).
- **Don't change VAD/ASR parameters** (`asr_align.py`, `vad_energy.py` are the high-risk core);
  any change needs a stated output-consistency impact + tests or an experiment record.
- GPU (>=8GB VRAM) is expected in this dev environment; CPU fallback must print `Warning:` to
  stderr. Don't assume GPU is unavailable without checking.
- Artifacts never land at the repo root: `data/` (source audio), `out/` (generated), `tmp/`
  (scratch). Pipeline stages skip on **existence only** (no content validation) — to force a
  rerun, delete the stage output and everything downstream.
- Dependencies live only in `pyproject.toml`; never create `requirements.txt`. Production
  entrypoints call functions directly (no subprocess).
- The knowledge base `knowledge/` is NOT tracked by the main repo (own embedded git, auto-commits
  on apply). Prompt templates `src/llm/prompt_templates/*.md` ARE tracked — prompt text is never
  hardcoded in Python.
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

- `pipeline.py` — orchestrator; derives artifact paths from the output SRT path, skips stages
  whose outputs exist, accepts URLs as input (yt-dlp). LLM stages opt-in via `--stage` and pass
  `--llm-*` / `--knowledge` / `--refined-srt` through.
- `batch.py` — three-bin batch runner (download×2/asr×1/llm×1 pools, ordered llm bin, per-item
  failure isolation) + manifest CLI; `llm.reference_ingest` multi-task runs use the same engine.
  llm concurrency stays 1 by design (unlocked in-process rate limiter; knowledge auto-apply
  commits to the embedded git).
- `vocal_separation.py` / `vad_asr.py` / `asr_stabilize.py` / `to_srt.py` — pipeline stages;
  `vad_asr.py` writes `*-aligned.json` (incl. DP splitting of over-long segments via
  `segment_split.py`, docs/segment_split.md; split-produced segments carry a
  `splitted_before` tag), then ASR stabilization writes `*-stable.json` (profile 0 =
  cleanup → deterministic pre-merge via `premerge.py` → noise tags → drop;
  docs/asr-stabilize.md); `asr_align.py` + `vad_energy.py` are the large core
  algorithm modules (high-risk); `resource_profiles.py` — GPU budget tiers (only
  vocal-separation batch scales).
- `src/llm/` — the LLM harness:
  - `profiles.py` (6 route/level presets) · `config.py` (roles, limits, budgets) ·
    `model_catalog.psv` (model/tier facts; its `capability` column drives the correction
    prompt's capability tier — assembled per answering endpoint: capable=judgment merge,
    basic=conservative 1:1 merge fragments) · `client.py` + `llm_runtime.py` + `rate_limit.py`
    + `content_filter.py` (Gemini REST calls via httpx, endpoint chains, RPM/TPM limiting,
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
  - `reference_ingest.py` (end-to-end ingest, executes by default) · `prompt_artifacts.py`
    (dry-run prompt assembly shared with `--prompt-dir`) ·
    `task_report.py` / `exchange_log.py` / `exchange_metadata.py` (artifacts & readable logs)
- `tools/` — standalone dev tools, maintained ON DEMAND only (each has a README with its
  policy): `session_replay/` (freeze upstream
  injections, re-hit a session), `asr-confidence-explorer/` (manual analysis snapshot).
  Their tests live next to them and are not collected by the default suite.
- `legacy/` — superseded scripts; don't build on it. `utils/` — shared helpers.

## Docs index (read on demand)

| Doc | Read when the task involves |
| --- | --- |
| `README.md` | User-facing usage: install, quickstart, common entrypoints, note-writing tips |
| `docs/manual/env.md` | `.env` / Gemini (AI Studio) / Exa / Tavily API key setup |
| `README_DEV.md` | Dev principles, resource constraints, **canonical artifact tree**, reuse/resume rules, agent checklist |
| `docs/llm_harness_behavior.md` | **Canonical LLM runtime behavior**: presets, fast mode, search agent, window/token budgets, injection limits, output contract, retries/stitching, resume, rate limits, artifacts |
| `docs/knowledge.md` | Everything knowledge-base: structure, `--knowledge` tri-state, feedback v2, unified update, mistake ledger, reference_ingest |
| `examples/knowledge/` | Tracked mini knowledge-base samples (not the live `knowledge/` tree) |
| `docs/llm_prompts.md` | Prompt templates/fragments, prompt_compose assembly table, PROMPT_VERSION semantics |
| `docs/llm_design_notes.md` | Architecture intent, design decisions & rationale (budget formula derivation, knowledge-update decision ledger), deferred designs |
| `docs/testing.md` | Test markers, scoped commands, which tests cover which paths |
| `docs/asr-align.md` | VAD interval -> aligned ASR：解码配置、词级映射、异常救援阶梯（greedy -> 异常 interval 隔离）与其取舍依据、覆盖率救援、输出字段语义（含 `confidence` 不作质量指标的说明） |
| `docs/asr-stabilize.md` | aligned → stable ASR profiles (incl. profile 3 deterministic pre-merge), metrics, tags, CLI, and resume rules |
| `docs/session_replay.md` | Prompt-iteration replay: 6 sessions (correction R2 + query/research-r1/r2/search-judge/fast-round1), 各轮 fixture/validation 契约、补中间态落盘、变体仅 correction 支持 |
| `docs/segment_split.md` | 超长 segment 切分规范：DP 打分、gap word 调整（生产 `segment_split.py`；`tools/split_explorer` 为调参薄封装） |
| `docs/tools/prompt-iterate.md` | **纠错 prompt 迭代方法论**（长期）：定位（prompt/harness 迭代唯一机制，不碰知识库更新）、四变体 capableB/C + basicA/B、session_replay 协议（`--model`/`--variant`）、prompt 原则、失效模式（含 singles 残留案例）、产物命名。已完成 run 的离线诊断仍走 `.claude/skills/run-audit`；二者分工见该 skill 文首 |
| `docs/merge-calibration.md` | 精修标定的合并软门槛与模型边界（默认不并、gap/字数先验、flash-lite thinking=0）；现行变体契约仍以 prompt-iterate §4 为准 |

`docs/archive/` 与 `docs/report/` 为本地笔记（gitignore），不随仓库发布；迁入前按上方
**Archive extraction** 规则抽非过时信息。
