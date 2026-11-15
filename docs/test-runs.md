# 跑哪些测试：时机与命令

这份只讲**什么时候跑什么、命令怎么写**。测试为什么这样组织——域标记的划分、结构守卫、测试替身放哪、
仓库根的 `conftest.py`、设备解析与子进程弹窗的验证手法——在 [`testing.md`](testing.md)。

## 何时跑哪种

| 时机 | 跑什么 |
| --- | --- |
| 改完一个文件，还在改 | 它对应的测试文件，`-n 0`（秒级）；怎么找见[改动对应的测试](#改动对应的测试) |
| 一批改动做完（合并或交给 owner 之前），一批只跑一次；同一批里中间的提交只跑改到的文件对应的测试 | `python -m compileall -q src test && ruff check && python -m pytest -q && git status --short` |
| 只改了文档（改动全是 `src/`、`test/` 之外的 `.md`），提交前 | 文档守卫：`python -m pytest -q test/test_doc_links.py test/test_doc_facts.py test/test_doc_style.py test/test_publish_filter.py test/test_settings.py test/test_packaging.py test/test_llm_coldread_scripts.py`，再加 `git grep -l <被改文件名> -- test` 找到的测试；然后 `git status --short` |
| 增删改名 `src/finesub/` 顶层模块，或动发布契约（`build-wheel.ps1` 的清单） | 全量 **+ `python -m pytest -q cli/tests`**：根套件收集不到 `cli/tests`，而 wheel 的清单守卫在根套件的 `test_packaging.py`、壳的行为在 `cli/tests` |
| 动设备解析那一族（`speech/runtime/device.py`、`resources.py`、裁判放置、分离器设备） | 全量**两遍**：`pytest -q` 与 `CUDA_VISIBLE_DEVICES=-1 pytest -q`。为什么见 [testing.md「设备解析」](testing.md#设备解析两条只能这样验的手法) |
| 换 CT2 wheel / 动 VAD-ASR 核心 / 动分离器性能 | 全量 + [两条重资源测试](#ci-跑不到只能本机跑的)。重资源测试按 `AGENTS.md` 的护栏只在用户明确要求时跑，所以先问 |
| 改下载路径的镜像表、lock 生成器或模型清单 | 除回归测试外，再做一次实机验收（全量摘要比对、全新安装、破坏哈希、中断续传），做法与上次结果在 [download-routes.md「验证」](download-routes.md#9-验证) |
| 在 linked worktree 里动了 `requires_main_checkout` 覆盖的面 | 这类用例在 worktree 里一律 skip、在哪个分支上都是绿的。用逃生舱跑一遍：`FINESUB_TEST_ALLOW_WORKTREE=1 python -m pytest -q <file>`；其中 `test_paths` / `test_separator_accel` 本来就该失败（worktree 天然重定向到主 checkout），不是回归 |
| 发版 | 两个环境各跑一次全量（装了 `[asr]` 的与没装的**查的是相反的病**）+ `cli/tests` |

**别按域挑着跑来代替全量**：全量只比最大的域多二十来秒，挑选省下的时间抵不过挑错的风险。

## 改动对应的测试

| 改动 | 跑 |
| --- | --- |
| 一个模块 | 与它同名的测试文件：`test/test_<模块名>.py`，LLM 层是 `test/test_llm_<模块名>.py`，`finesub_bootstrap` 的在 `test/bootstrap/`。对不上名字时用 `git grep -l <模块名> -- test` 找 |
| `src/finesub/llm/` 下的多个文件 | `python -m pytest -q -m llm`（约 80 秒） |
| 语音链（VAD、解码、对齐、稳定化、分句）的多个文件 | `python -m pytest -q -m asr`（约 20 秒） |
| 管线编排、运行时与资源、配置、`finesub_bootstrap` 的多个文件 | `python -m pytest -q -m pipeline`（约 35 秒） |
| 信箱 Worker（`deploy/share-worker/src/index.js`） | 在 `deploy/share-worker` 下 `node --test "test/*.test.mjs"`（不到一秒，默认全量经 `test/test_llm_knowledge_share_worker.py` 也跑它；CI 的 ubuntu 任务为它装 Node 22，本机没有带 `node:sqlite` 的 Node 时那一条 skip）；部署前再对着 `wrangler dev` 跑 `e2e_check.py`，见 [Worker README 的「测试」](../deploy/share-worker/README.md#测试) |

三个域合起来就是全量；耗时是本机 2026-09-29 的读数（当时 `-n auto`，16 路）。4 路下 `-m llm` 约 130 秒（2026-10-24）。

## 命令

| 命令 | 用途 |
| --- | --- |
| `python -m pytest -q` | 全量。默认 `-n 4 --timeout=120`：每个 worker 都加载 CUDA 版 torch（约 1.3 GiB 已提交、0.5 GiB 常驻），8 个在本机和游戏同开时内存吃紧，4 个只慢约半分钟（owner 2026-10-24）；约 6200 例，本机 4 路约 180 秒、8 路约 150 秒、16 路约 145 秒，测试进程的内存峰值约 4.6 / 7.7 / 15 GiB（2026-10-24，顺带测时，各跑一次）。几十分钟的话先确认仓库根的 `conftest.py` 还在，见 [testing.md 的说明](testing.md#为什么仓库根有一个-conftestpy) |
| `python -m pytest -q -n 0 <file>` | 单 worker 跑单文件 |
| `--run-heavy-resource` | 加上重资源测试（GPU、大模型、大音频） |
| `--run-network-mock` | 加上绑本机端口的网络层证明测试（`[api] proxy` 路由、API 调用硬截止） |
| `--ignore=test/test_vad_streaming.py` | 临时跳过最慢的那个文件（它独占一个 worker 三十多秒）；比标记可靠，原因见 [testing.md「没有 slow 标记」](testing.md#没有-slow-标记) |
| `python -m pytest -q cli/tests` | 薄 CLI 的独立套件，根 `pytest -q` 收集不到 |
| `ruff check` | 静态检查，在仓库根跑（不到一秒）：只开 pyflakes 的 F 规则（未定义的名字、没用的 import 与变量、被覆盖的定义），范围 `src/` `test/` `cli/`，`tools/` 与 `explore/` 不查，配置在 `pyproject.toml` 的 `[tool.ruff]`。版本钉在 `[dev]` extra 里，CI 的 ubuntu 主作业跑同一个版本；`ruff check --fix` 自动删掉没用的 import，删之前先 `git grep` 一下那个名字是不是别处从这个模块 import 的或测试 monkeypatch 的目标 |
| `python -m pytest tools/session_replay -n 0` | 重打工具的测试，不进默认套件 |
| `python -m pytest tools/bench` | 离线探针的测试，不进默认套件；依赖在 `[bench]` extra（`pip install -e ".[asr,bench]"`），标准的 `[asr,harness,dev]` 会 `ImportError` |

⚠ PowerShell 不展开 `test/test_llm_*.py` 这样的 glob：用显式文件列表或 `-m llm`。

## CI 跑不到、只能本机跑的

CI 不装 `[asr]`、不跑重资源用例（`extras` 作业里的 CTranslate2 是 stock 的），下面两条在 CI 里永远不执行，CI 绿不代表它们验过：

- **换 CT2 wheel 之后**：`python -m pytest test/test_fw_refine.py --run-heavy-resource -q`。其中
  `test_a_cpu_model_decodes_and_then_shuts_down` 守的是 CPU GEMM 后端选错导致的死锁或 `No SGEMM backend on CPU`，
  两种都只在真解码时暴露。完整的换 wheel 验收清单见 `tools/wt_refine_port/ct2-patches/README.md`。
- **动显存 / 内存预算或分离器之后**：`python -m pytest -q test/test_resource_budget_pipeline.py --run-heavy-resource -n 0`。
  需要 CUDA、已缓存的 audio-separator 模型，VAD-ASR 那一半还需要 `assets/harvard.flac`（缺了会明确 skip）。
  它守什么、有哪些坑见 [testing.md「备注」](testing.md#备注)。
