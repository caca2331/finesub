# 测试分类与按需运行

默认 `pytest -q` 跑全量；准确数量以 `pytest --collect-only -q` 为准。测试通过 `pytest-xdist` 以 **2 worker** 并行（`pyproject.toml` 的 `addopts = "-n 2"`）。单 worker：`pytest -q -n 0`。

## 日常命令

| 命令 | 用途 |
|------|------|
| `pytest -q` | 全量回归（默认 `-n 2`） |
| `pytest -q -m llm` | 只跑 LLM harness |
| `pytest -q -m pipeline` | 只跑管线编排 / GPU 配置 |
| `pytest -q -m asr` | 只跑 ASR/VAD/对齐/文本工具 |
| `pytest -q -m "not slow"` | 跳过慢用例的快速回归 |
| `pytest -q --run-heavy-resource` | 含重资源测试（若有） |

## 何时跑哪种

- 改 `src/llm/` → `-m llm`（或下表对应文件）
- 改 `pipeline.py` / `vad_asr.py` / `asr_stabilize.py` / `vocal_separation.py` → `-m pipeline`，稳定化纯逻辑也跑 `-m asr`
- 改 `asr_align.py` / `vad_energy.py` / `utils/text.py` → `-m asr`
- 大功能完成或合并前 → 全量 `pytest -q`

## 路径 → 测试文件

| 改动位置 | 推荐命令 |
|----------|----------|
| `src/llm/client.py`, `llm_runtime.py`, `rate_limit.py`, `config.py`, `content_filter.py` | `pytest -q test/test_llm_client.py test/test_llm_config_and_budget.py test/test_llm_content_filter.py` |
| `src/llm/search_loop.py`, `research.py` | `pytest -q test/test_llm_search_loop.py test/test_llm_research.py` |
| `src/llm/correction_translation.py`, `stages/` | `pytest -q test/test_llm_correction_translation.py test/test_llm_fast_mode.py test/test_llm_text_route.py test/test_llm_video_route.py` |
| `src/llm/knowledge/` | `pytest -q test/test_llm_knowledge_base.py test/test_llm_knowledge_materials.py test/test_llm_knowledge_update.py test/test_llm_common_mistakes.py` |
| `src/llm/web_search.py` | `pytest -q test/test_llm_web_search.py test/test_llm_web_search_urls.py` |
| `src/llm/chunking.py`, `prompt_templates/`, `prompt_artifacts.py` | `pytest -q test/test_llm_srt_and_chunking.py test/test_llm_prompts.py test/test_llm_prompt_compose.py` |
| `src/subtitle_metrics.py`, `src/llm/subtitle_metrics.py`, `csv_utils.py`, `srt_utils.py`, `srt_postprocess.py`, `task_report.py` | `pytest -q test/test_llm_subtitle_metrics.py test/test_llm_csv_utils.py test/test_llm_srt_and_chunking.py test/test_llm_srt_postprocess.py test/test_llm_task_report.py` |
| `src/pipeline.py`, `src/batch.py`, `vad_asr.py`, `asr_stabilize.py`, `vocal_separation.py` | `pytest -q test/test_asr_stabilize.py test/test_pipeline_refactor.py test/test_batch_runner.py` |
| 显存/内存预算实测（真跑分离+ASR，重资源） | `pytest -q test/test_resource_budget_pipeline.py --run-heavy-resource -n 0` |
| `src/asr_align.py`, `vad_energy.py`, `utils/text.py` | `pytest -q test/test_asr_and_text_utils.py test/test_vad_streaming.py test/test_vad_segment_energy.py` |
| `src/resource_profiles.py` | `pytest -q test/test_resource_profiles.py` |

也可用域标记代替显式文件列表，例如 `pytest -q -m llm -m "not slow"`。

## 标记说明

| 标记 | 含义 |
|------|------|
| `llm` | `test_llm_*.py`（conftest 自动） |
| `pipeline` | `test_pipeline_refactor.py`, `test_batch_runner.py`, `test_resource_profiles.py`, `test_resource_budget_pipeline.py` |
| `asr` | `test_asr_stabilize.py`, `test_asr_and_text_utils.py`, `test_intervals.py`, `test_srt_rendering.py`, `test_vad_streaming.py`, `test_vad_segment_energy.py` |
| `slow` | 实测慢用例（~0.5s+）；`search_loop` 与长 VAD streaming 守卫等 |
| `heavy_resource` | GPU/大模型/大音频；默认 skip，需 `--run-heavy-resource` |

## 备注

- PowerShell 下不要用 `test/test_llm_*.py` glob（不展开）；用显式文件列表或 `-m llm`。
- `test_pipeline_refactor.py`、`test_intervals.py`、`test_vad_streaming.py` 收集时会 import torch，collect 略慢属正常。
- 默认测试套件不生成 prompt preview。重打工具（`tools/session_replay`）按需维护，
  其测试不被默认收集，需要时显式运行 `python -m pytest tools/session_replay -n 0`。
- 默认单测不得加载 Whisper/audio-separator、处理大音频或消耗 Gemini quota。
- `test_resource_budget_pipeline.py`：合成短音频上真跑「分离 → VAD-ASR」两段，断言实测 peak
  显存/内存不超过所选 profile 上限（默认 8GB）。需 CUDA（无 CUDA 自动 skip）与已缓存的
  audio-separator 模型；`RESOURCE_TEST_SECONDS` 可加长音频以压 RAM 路径，参数化 `gpu_budget_gb`
  可在 12/16GB 机器上验证对应档位。
