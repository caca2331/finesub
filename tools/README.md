# `tools/` —— 这里面有什么，哪些还在用

**不是一个「工具箱」而是三类东西混住**：还在用的工具、做完就留在原地的一次性实验、
以及跑不起来的。这份索引存在的唯一目的是让你
**30 秒内判断该进哪个目录**，而不是挨个打开 README。

⚠ **`tools/` 是按需维护的**（`AGENTS.md`）：不要作为其它改动的副作用去更新它们。
唯一例外是改名/移动类改动——那必须同步 `tools/session_replay`。

## 找什么去哪

| 我要… | 去 |
| --- | --- |
| 测性能、跑探针、复算某个基准数字 | [`bench/`](bench/README.md)（先读 `docs/bench-discipline.md`） |
| 重放一次 LLM 会话、迭代 prompt | [`session_replay/`](session_replay/README.md) |
| 看/改分割金标准的人工标注 | [`segmentation_gold/`](segmentation_gold/README.md) |
| 动 patched CTranslate2 的补丁或重编 wheel | [`wt_refine_port/`](wt_refine_port/README.md)（补丁在 `ct2-patches/`，CI 构建脚本在 `ct2-build/`） |
| 在 A 卡 / Mac 等硬件上做社区验收、交回记录 | [`hw_acceptance/`](hw_acceptance/README.md) |
| 重新导出 **DiariZen** 后端的产物（分割 + 声纹 + PLDA） | [`diarizen_export/`](diarizen_export/README.md)（⚠ **不要 freeze**：折 BN 会让答案移动 4.5e-4~7.3e-4） |
| 把 `-diar.json` 里记下的每轨试听片段切出来 | `diarizen_export/cut_samples.py`（十秒一段、每轨一份；**听一遍比任何 prompt 都快**。目录是 `<stem>-speaker-samples/`，⚠ 不是 `--speaker-tracks` 的 `<stem>-speakers/`。仅 diarizen 的时间线有） |
| 重新导出说话人模型的三个 TorchScript 件 | [`sortformer_export/`](sortformer_export/README.md)（**唯一需要 nemo 的地方**；导完必须跑计划 §5a 的 G1–G4） |
| 本地 token 计数器（**生产组件**，不是工具） | [`tokcount/`](tokcount/README.md) |
| 跑 / 打分任务式流程的验收臂 | `bench/run_flow_ab.py`（交错跑四臂，每次 run 独立目录 + `run.json`）、`bench/score_flow_ab.py`（对精修字幕的相似度/对上率 + 调用与 token） |
| 看每次 LLM 调用想了多少、找思考为 0 的调用（「模型回答的退化」的依据） | `thinking_scan.py`（读 `task-artifacts.jsonl`，按路由汇总；`--zero` 逐条列出并对照耗时、输入、可见输出） |
| 复算纠错窗口的丢弃比例（保留数下限 `KEPT_FLOOR` 的依据） | `discard_ratio_scan.py`（读归档 `correction-windows.jsonl`；记录在 `docs/bench-baselines.md` 二十五） |
| **调 VAD 参数** | ⚠ 生产实现是 `src/finesub/speech/preprocessing/energy.py`，规范是 `docs/vad-energy.md`。`vad_tuning/` 是**已结束的探索**，读它的 `FINDINGS.md` 是为取舍依据，不要照它改代码 |

## 三类目录

### 活跃（还在用，改动它们要负责）

文件数是 2026-09-24 手数的读数，只说规模，不随改动维护。

| 目录 | 文件 | 谁在引用 |
| --- | ---: | --- |
| `bench/` | 44 | `docs/bench-baselines.md` 20+ 处 |
| `session_replay/` | 15 | `AGENTS.md`、`agent-tasks/run-audit/`、`docs/session_replay.md` |
| `segmentation_gold/` | 31 | `docs/segmentation-gold.md`（**人工标注不可再生**） |
| `wt_refine_port/` | 31 | `AGENTS.md`、`docs/architecture.md`、`docs/ct2-distribution.md`（CT2 补丁基线与 CI 构建脚本）、`.github/workflows/build-ct2.yml` |
| `kb_phonetic_eval/` | 6（2026-10-23 数） | `docs/knowledge.md`（「精修字幕怎么读」的词条瘦身一段）与 `llm/kb_phonetic_hints.py` 的注释：音近提示的离线重放与评估，`FINESUB_KB_SLIM` 翻默认要看的读数 |
| `hw_acceptance/` | 4 | `docs/manual/resources.md`、`docs/asr-backends.md`（社区验收套件） |
| `diarizen_export/` | 33 | [`diarizen_export/README.md`](diarizen_export/README.md)、`docs/diarization-findings.md`：DiariZen 后端的 TorchScript 产出方式与分层校验器（`check_s1`–`s4`、`check_port`、`noise_arm`，⚠ **只报不判**：换产物形态是 A 类，要刻画抖动而不是消除它）；测量床（单人证据、角色命名 `roles_bed`+`roles_run`、人数上限、跨集声纹、轨身份、原生边界、行内换人 `turn_shape_bed` / `turn_signal_bed` / `turn_llm_bed` / `turn_cut_bed` / 判官 `turn_output_judge`），各自的结论在 findings 对应节；`cut_samples` 按需把生产选好的试听段切成音频；⭐ `verify_guards` 把每个守卫声称能防的 bug 注回去，确认它真的红 |
| `sortformer_export/` | 2 | `docs/architecture.md`（`speech/diarization/` 一行）：说话人模型三个 TorchScript 件的产出方式。⚠ 里面有两份 NeMo 方法的逐字拷贝（TorchScript 拒绝原样编译），**只住在这里、管线看不到**；拷贝烤进去的假设都在导出前断言 |
| `diarization_explore/` | 183 | `docs/diarization-findings.md`（结论）、归档的过程记录（findings 第 13 节）：说话人实验与打分台。⭐ **先读 `eval_contract.py`**——它一处声明「这批素材的真值标的是人还是角色 / 是不是逐行 / 哪些行被计价 / 弃权算不算钱 / crossfit 留出多少」，`require_people_truth()` 会拦住在角色真值上问身份问题。findings 的每个数字都由这里的脚本产出，逐文件结果在 `results/`。三份 `test_*.py` 要显式跑，`testpaths` 不含 `tools/` |
| `ts_diar_explore/` | 33 | 维护者的本地笔记（结论不随公开快照发布）：自带时间戳 + 说话人的外部模型（MAI-Transcribe-2 / Gemini 3.5 Transcribe / MOSS-Transcribe-Diarize 开源与 Pro）评估台。`arms.py` 是所有臂的注册表（名字、产物位置、带什么、怎么跑），`episode.py` 按它运行并拼接（`--span` 只跑一段），`report.py` 出全部读数（配对均值），`merge_windows.py` 合窗，`followups.py` 是两个后续实验，`moss_serve.py` 是 MOSS 开源版经 vLLM / SGLang-Omni（WSL）推理的客户端；开源版的进程内快速解码与运行时救援已进生产（`--asr moss-local`，`speech/joint/moss_local/`），臂 `mossg` / `mossr` / `mossrv` 跑的是每条规则带开关的实验版，在维护者的归档里；`degen.py` 按生产 Whisper 对照找开源 MOSS 的退化段、只在干净处打分；指标在 `score.py` / `overlap.py` / `timing.py`。⚠ mai2 / 35t / Pro 有配额，调用都记在主 checkout 的 `tmp/ts-diar-quota/ledger.jsonl`，适配器到限自停；35t 与 Pro 的额度日由 owner 告知 |

⚠ **判「还在不在用」看「谁在引用」，不要看改动日期。** 这里原本有一列 `git log -1 -- <dir>`
的日期，2026-09-03 删掉了：它把「补一份 README」和「跑了一轮实验」记成同一件事，
建索引当天就有两行失真——一个需要脚注才能读对的数字，不如没有。

**错位一个**：`tokcount/` 是 Go 写的本地 token 计数器，被
`llm/token_budget.py` **在生产路径上依赖**，不是工具。它住在 `tools/` 是历史原因；
搬它要同时改 `runtime-manifest.json` 与打包清单，所以先记在这里，不顺手动。

### 一次性实验（结论已沉淀，代码留作回溯）

新的一次性实验不再放这里，放仓库根的 `explore/`（发布时剥离）；下表是之前留下、公开文档还在引用的。

⚠ **已经在 `tools/` 里的实验与测量床不挪进 `explore/`，脚本里写死的本机路径也不统一改**（owner 2026-11-08，第二次定）：
公开文档引用它们的结论、标注与复现命令（`vad_tuning/FINDINGS.md` 与人工标注、`qwen3_explore/FINDINGS.md`、
`diarization_explore/` 的打分台），挪进 `explore/` 会在发布时把这些依据一起剥掉；脚本按需维护，有人重新启用哪个，
再顺手把它的路径改成参数。2026-11 的全仓审计挪过一次、又撤回了，以后审计不再提这一条。

读它们是为**当时怎么判的**；现行行为一律以 `src/` 与对应的 owner 文档为准。

| 目录 | 文件 | 结论落在哪 |
| --- | ---: | --- |
| `vad_tuning/` | 59 | `FINDINGS.md`（附录号被 `silero_ghost.py` 的 docstring 引用，**不能重编号**）；生产形态见 `docs/vad-energy.md` |
| `qwen3_explore/` | 39 | `FINDINGS.md`；被 `docs/segmentation-gold.md` 引为机械指标精确率的原始裁决 |
| `separator_rate/` | 19 | `docs/separator-optimization.md`（E12：降采样率的代价） |
| `separator_accel_bench/` | 14 | `docs/separator-optimization.md` |
| `wt_refine_validation/` | 13 | `docs/wt-refine-validation.md` |
| `cf_share_poc/` | 5 | `deploy/share-worker/README.md`（引它的坑记）。实测结论：共享服务端的 `KnowledgeStore` **不改一行**就能跑在 Durable Object 的 SQLite 上（2026-09-09 实测，17 张表 + 一次提交 + 一次回滚）。⚠ 它复制那三个 store 文件而不入库副本——跑的必须是没改过的那份；⚠ 本地 miniflare 在**长路径**下 SQLite 起不来，症状是毫无细节的 `internal error`（README 记着这个坑和另外两个） |

### 跑不起来的

- `split_explorer/`：它自己的 README 写着「当前暂不可运行」——生产切分器改过，这个薄封装的
  import 没跟。它仍被引用：`docs/segmentation-split.md` 把它写成调参入口，`segmentation.py` 的注释
  也指向它。下次真要用它得先迁 import。
- 一级目录的散落文件都有人引用：`separator_benchmark.py`（`docs/separator-optimization.md` 的复现命令、
  `separator_rate/` 与 `separator_accel_bench/` 的 `run/*.sh` 都调它）、`separator_aoti.py`
  （`README_DEV.md` 与 `docs/separator-optimization.md` 引着）、`discard_ratio_scan.py`（`output_protocol.py`
  的门槛注释、`docs/llm_harness_behavior.md` 与 `docs/bench-baselines.md` 二十五都引着）。
  `compare_vad_srt.py` 与它的测试零引用，2026-09-24 删了，要回溯看 git 历史。

## ⚠ 两件容易踩的事

1. **这里的 `test_*.py` 默认永远不跑。** `pyproject.toml` 的 `testpaths` 只含
   `test/`。要跑就显式给路径：`python -m pytest tools/bench -q`。
   同时 `test/test_import_boundaries.py` 又把 `tools/` 当源码树扫描——**对测试套件而言
   `tools/` 既是「要检查的源码」又是「不运行的测试」**，这是有意的取舍，不是漏配。
2. **素材不在仓库里。** 这些脚本大多读 `data/` 与 `assets/`，两个目录都不入库，
   逐条清单见 `docs/data-index.md`（规则半边入库，清单半边只在本地）。

## 加新东西时

- 一次性实验不进这里，放仓库根的 `explore/<名字>/`（发布时剥离，规矩与清单在它自己的 README）；
  做完之后结论要蒸馏进 `docs/` 的 owner 文档。要让公开读者复现的实验，脚本才升格进这里，
  建子目录 + 一份 README，**首行写状态**，并在上面的表里加一行。
- 运维脚本（发布、签名之类）进 `scripts/`，不进这里。
