# `tools/` —— 这里面有什么，哪些还在用

13 个子目录、4 个散落文件，**不是一个「工具箱」而是三类东西混住**：还在用的工具、
做完就留在原地的一次性实验、以及两处已经跑不起来的。这份索引存在的唯一目的是让你
**30 秒内判断该进哪个目录**，而不是挨个打开 README。

⚠ **`tools/` 是按需维护的**（`CLAUDE.md`）：不要作为其它改动的副作用去更新它们。
唯一例外是改名/移动类改动——那必须同步 `tools/session_replay`。

## 找什么去哪

| 我要… | 去 |
| --- | --- |
| 测性能、跑探针、复算某个基准数字 | [`bench/`](bench/README.md)（先读 `docs/bench-discipline.md`） |
| 重放一次 LLM 会话、迭代 prompt | [`session_replay/`](session_replay/README.md) |
| 看/改分割金标准的人工标注 | [`segmentation_gold/`](segmentation_gold/README.md) |
| 动 patched CTranslate2 的补丁或重编 wheel | [`wt_refine_port/`](wt_refine_port/README.md) |
| 本地 token 计数器（**生产组件**，不是工具） | [`tokcount/`](tokcount/README.md) |
| 复算纠错窗口的丢弃比例（`MAX_DISCARD_RATIO` 的标定） | `discard_ratio_scan.py`（读归档 `correction-windows.jsonl`；记录在 `docs/bench-baselines.md` 二十五） |
| **调 VAD 参数** | ⚠ 生产实现是 `src/finesub/speech/preprocessing/energy.py`，规范是 `docs/vad-energy.md`。`vad_tuning/` 是**已结束的探索**，读它的 `FINDINGS.md` 是为取舍依据，不要照它改代码 |

## 三类目录

### 活跃（还在用，改动它们要负责）

| 目录 | 文件 | 谁在引用 |
| --- | ---: | --- |
| `bench/` | 42 | `docs/bench-baselines.md` 20+ 处 |
| `session_replay/` | 20 | `CLAUDE.md`、`agent-tasks/run-audit/`、`docs/session_replay.md` |
| `segmentation_gold/` | 31 | `docs/segmentation-gold.md`（**人工标注不可再生**） |
| `wt_refine_port/` | 24 | `CLAUDE.md`、`docs/ct2-distribution.md`（CT2 补丁基线） |

⚠ **判「还在不在用」看「谁在引用」，不要看改动日期。** 这里原本有一列 `git log -1 -- <dir>`
的日期，2026-09-03 删掉了：它把「补一份 README」和「跑了一轮实验」记成同一件事，
建索引当天就有两行失真——一个需要脚注才能读对的数字，不如没有。

**错位一个**：`tokcount/`（5 文件）是 Go 写的本地 token 计数器，被
`llm/token_budget.py` **在生产路径上依赖**，不是工具。它住在 `tools/` 是历史原因；
搬它要同时改 `runtime-manifest.json` 与打包清单，所以先记在这里，不顺手动。

### 一次性实验（结论已沉淀，代码留作回溯）

读它们是为**当时怎么判的**；现行行为一律以 `src/` 与对应的 owner 文档为准。

| 目录 | 文件 | 结论落在哪 |
| --- | ---: | --- |
| `vad_tuning/` | 59 | `FINDINGS.md`（附录号被 `silero_ghost.py` 的 docstring 引用，**不能重编号**）；生产形态见 `docs/vad-energy.md` |
| `qwen3_explore/` | 39 | `FINDINGS.md`；被 `docs/segmentation-gold.md` 引为机械指标精确率的原始裁决 |
| `separator_rate/` | 19 | `docs/separator-optimization.md`（E12：降采样率的代价） |
| `escape_density/` | 2 | `docs/plans/nonoka-downstream-findings-plan.md` 第十二节的两条阈值——⚠ 转义那条最后**没有留下任何阈值**，这两个脚本记的是为什么；`finesub.text.looks_escaped` 的注释引它。⚠ `corpus_baseline.py` 读的是**跑过任务的那个 checkout** 的 `out/`，worktree 里要把路径当参数传 |
| `onset_gap_energy/` | 17 | 无外部引用——探索留档 |
| `separator_accel_bench/` | 14 | `docs/separator-optimization.md` |
| `wt_refine_validation/` | 13 | `docs/wt-refine-validation.md` |
| `asr-confidence-explorer/` | 3 | 无外部引用（目录名用连字符，其余全用下划线） |

### 跑不起来的

- `split_explorer/`（8 文件）：它自己的 README 第 11 行写着「当前暂不可运行」——
  生产切分器改过，这个薄封装的 import 没跟。结论在 `docs/segmentation-split.md`，
  下次真要用它得先迁 import。
- 散落在一级目录的四个文件：`compare_vad_srt.py` + `test_compare_vad_srt.py`
  （互相引用之外**零引用**）、`separator_benchmark.py`（零引用，结论已进
  `docs/separator-optimization.md`）。两个例外是活的：`separator_aoti.py`
  （`README_DEV.md` 与 `docs/separator-optimization.md` 引着）与
  `discard_ratio_scan.py`（`output_protocol.py` 的门槛注释、
  `docs/llm_harness_behavior.md` 与 `docs/bench-baselines.md` 二十五都引着）。

## ⚠ 两件容易踩的事

1. **这里的 16 个 `test_*.py` 默认永远不跑。** `pyproject.toml` 的 `testpaths` 只含
   `test/`。要跑就显式给路径：`python -m pytest tools/bench -q`。
   同时 `test/test_import_boundaries.py` 又把 `tools/` 当源码树扫描——**对测试套件而言
   `tools/` 既是「要检查的源码」又是「不运行的测试」**，这是有意的取舍，不是漏配。
2. **素材不在仓库里。** 这些脚本大多读 `data/` 与 `assets/`，两个目录都不入库，
   逐条清单见 `docs/data-index.md`（规则半边入库，清单半边只在本地）。

## 加新东西时

- 一次性实验：建子目录 + 一份 README，**首行写状态**（活跃 / 一次性实验 / 已结束），
  并在上面的表里加一行。做完之后结论要蒸馏进 `docs/` 的 owner 文档——留在这里的
  `FINDINGS.md` 是过程，不是规范。
- 运维脚本（发布、签名之类）进 `scripts/`，不进这里。
