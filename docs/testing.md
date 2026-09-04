# 测试分类与按需运行

默认 `pytest -q` 跑全量；准确数量以 `pytest --collect-only -q` 为准。测试通过 `pytest-xdist` 按 **CPU 核数**并行（`pyproject.toml` 的 `addopts = "-n auto --timeout=120"`）。单 worker：`pytest -q -n 0`。

**`--timeout=120` 卡的是「挂住」，不是「跑得慢」**：实测最慢的单例约 3.3 秒，两个数量级的余量；套件里有真起子进程（agent driver）和真起本地 HTTP server 的用例，没有超时就是一个没有任何输出的CI job。重资源用例自己放宽。

**全量约 1 分钟**（`-n auto`，约 2050 例；`-n 2` 约 2 分钟）。如果你看到的是几十分钟，先确认仓库根的
`conftest.py` 还在——见下方「为什么仓库根有一个 conftest.py」。

**`finesub_bootstrap` 的测试在根套件里**（`test/bootstrap/`）：装机层是 CLI 和桌面共用的，
改 `fsops.py` 应该在提交前那条 `pytest -q` 里就红，而不是等一个本地没人跑的套件。留在
`desktop/backend/tests` 的是**只有 Windows 能真跑**的那些（junction/robocopy、DPAPI、
要 `powershell.exe`）——搬过来只会在 Linux runner 上变成永久 skip，等于把真实执行降级。

## 日常命令

| 命令 | 用途 |
|------|------|
| `pytest -q` | 全量回归（默认 `-n auto`，约 1 分钟） |
| `pytest -q -m llm` | 只跑 LLM harness |
| `pytest -q -m pipeline` | 只跑管线编排 / GPU 配置 |
| `pytest -q -m asr` | 只跑 ASR/VAD/对齐/文本工具 |
| `pytest -q --run-heavy-resource` | 含重资源测试（若有） |
| `pytest -q --run-network-mock` | 含绑本机端口的路由证明测试（`[llm] proxy`） |

**不建议再按域挑着跑**：全量既然是分钟级，挑选的收益已经低于挑错的风险。日常节奏是
**改文件时跑对应的单文件（`-n 0`，秒级），准备提交时跑全量**。上面的域命令留给「只想看
某一域是否还绿」的场合——三个合起来现在确实盖得全（见「标记说明」）。

## 何时跑哪种

| 时机 | 跑什么 |
|------|--------|
| 改完一个文件，还在改 | 下表里对应的**单文件**，`-n 0`（秒级） |
| 一个改动自认为完成 | 全量 `pytest -q` |
| 提交前 | `python -m compileall -q src test; pytest -q; git status --short` |
| 换 CT2 wheel / 动 VAD-ASR 核心 / 动分离器性能 | 全量 + `pytest test/test_fw_refine.py --run-heavy-resource -q`（**CI 永远跑不到它**）+ `test_resource_budget_pipeline.py --run-heavy-resource -n 0` |
| 增删改名 `src/finesub/` 顶层模块，或动发布契约（`REQUIRED_APP_FILES`、`build-wheel.ps1` 的清单） | 全量 **+ `pytest -q desktop`**——根 `pytest -q` 收集不到 `desktop/backend/tests`，而那里的更新器/安装器夹具按名字列着必需文件：只跑根套件会全绿，桌面套件却是红的（2026-08-30 实测） |
| 发版 | 全量 + desktop CI（`desktop/backend/tests`、`cli/tests`、`desktop/scripts/tests`；装机层的平台无关部分已在全量里，两边不重复收集） |
| **动设备解析那一族**（`speech/runtime/device.py`、`resources.py`、裁判放置、分离器设备） | 全量 **两遍**：`pytest -q` 与 `CUDA_VISIBLE_DEVICES=-1 pytest -q`。见下 |

## 设备解析：两条只能这样验的手法

两条都来自 2026-08 的逐阶段设备解析（取舍依据在
[`plans/stage-device-plan.md`](plans/stage-device-plan.md)，那份计划两次写下「本机验不了」、
两次都太早）。它们是**测试纪律**，所以住在这里，不住在那份计划稿里。

**一、在进程内劈开 torch 与 CT2。** 「CT2 能用这张卡而 torch 不能」这种机器多数人没有，
但不必换机器：`cuda_usable()` 判负的条件是卡的 compute capability **低于**
`torch.cuda.get_arch_list()` 的最小值，而那个列表可以替换——

```python
monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_130"])
```

于是 torch 如实认为自己没有这张卡的 kernel，而 CTranslate2 自带 arch 列表与 CUDA 运行时，
**照样看得见真实设备**。这不是打桩：`ct2_cuda_unusable_reason()` 跑的是真查询，卡也是真的。
⚠ `TestTorchAndCt2Disagree` 的第一条测试专门守着**这个前提本身**（`cuda_usable()` 为假且
`ct2_cuda_unusable_reason()` 为 None）——前提一旦不成立，后面两条就不再是它们自称的东西。

**二、用 `CUDA_VISIBLE_DEVICES=-1` 跑一遍无卡回归。** 这是最接近 CI 的本机复现，**只跑有卡
那一次会漏掉 CI 会红的**。第一次这么跑就红了 9 条：一条是真 bug（`cuda_device_present()` 用了
`torch.cuda.is_available()`，而 `CUDA_VISIBLE_DEVICES=""` 时它答 True 但 `device_count()` 是 0
——一台零可见设备的机器因此被放进 GPU 档，改成 `device_count() > 0`），八条是测试假设了宿主
有卡（钉算术的测试不该依赖宿主硬件，已加 autouse fixture 把能力钉成 True）。

⚠ 顺带记一条两个库真实不一致的地方：`CUDA_VISIBLE_DEVICES=""`（空串）下 torch 的
`is_available()` 是 True、`device_count()` 是 0，而 CT2 报 1（它把空串读成「未设置」）。
这正是「档位问 torch、ASR 问 CT2」的又一个理由——谁也别替谁回答。

## 路径 → 测试文件

⚠ **`requires_main_checkout` 在 worktree 里一律 skip，所以那些用例在任何分支上都是绿的。**
⚠ 那个 `FINESUB_KNOWLEDGE_WRITE=1` **只配给测试用**：它解除的是产品对主库的 worktree 写保护，
测试能带着它是因为它们全用 tmp 根。**别带着它从 worktree 跑任何生产命令**——那道保护就是为了
不让 worktree 里的一次实验写进主 checkout 的知识库。
动到它们覆盖的面时，用逃生舱跑一遍：`FINESUB_TEST_ALLOW_WORKTREE=1 FINESUB_KNOWLEDGE_WRITE=1
python -m pytest -q <file>`。2026-09-02 就是这样才发现两条断言早已删除字段的死测试——它们在
`dev` 上是红的，而每个 worktree 里的「全量 passed」都没碰到。⚠ 逃生舱下会多出两条**本来就
该失败**的（`test_paths` / `test_separator_accel` 断言 checkout 根解析，worktree 天然重定向），
那不是回归。

| 改动位置 | 推荐命令 |
|----------|----------|
| `src/finesub/config.py`（共享 `config.toml` 的读取/缓存）、`src/finesub_bootstrap/config_file.py`（保留注释的写入器） | `pytest -q test/test_config.py test/test_config_file.py test/test_llm_api_keys.py` |
| `src/finesub/llm/routing/{api_keys,config}.py`, `client.py`, `llm_runtime.py`, `rate_limit.py`, `content_filter.py`, `config.toml` | `pytest -q test/test_llm_api_keys.py test/test_llm_client.py test/test_llm_config_and_budget.py test/test_llm_content_filter.py test/test_paths.py` |
| `src/finesub/llm/search_loop.py`, `research.py` | `pytest -q test/test_llm_search_loop.py test/test_llm_research.py` |
| `src/finesub/llm/correction_translation.py`, `stages/`（含 `stages/correction/` 八模块） | `pytest -q test/test_llm_correction_translation.py test/test_llm_fast_mode.py test/test_llm_text_route.py test/test_llm_video_route.py` |
| `src/finesub/llm/knowledge/` | `pytest -q test/test_llm_knowledge_base.py test/test_llm_knowledge_materials.py test/test_llm_knowledge_update.py test/test_llm_knowledge_style.py` |
| `src/finesub/llm/knowledge/node/`（node store：schema/版本行/CAS、preset、三投影、无损导入与 parity；`apply`/`envelope`/`draft` 为 overlay 归并 + 按实体 CAS 的 apply engine；`cli`/`edit`/`history` 为人工编辑面与 revert/restore） | `pytest -q test/test_llm_knowledge_node.py test/test_llm_knowledge_node_apply.py test/test_llm_knowledge_node_cli.py test/test_llm_knowledge_node_matching.py test/test_llm_agent_mcp_kb.py`；真实库实测用 `python -m finesub.llm.knowledge.node.migrate --source knowledge --store tmp/kb-shadow/kb.sqlite --report tmp/kb-shadow/report` |
| `src/finesub/llm/web_search.py` | `pytest -q test/test_llm_web_search.py test/test_llm_web_search_urls.py` |
| `src/finesub/llm/agent/{local_agent,agent_paths,agent_cleanup}.py`、Agent 路由/执行身份、bootstrap 路径与 shell | `pytest -q test/test_llm_local_agent.py test/test_llm_agent_paths.py test/test_llm_execution_policy.py test/test_llm_model_router.py cli/tests/test_cli_main.py test/bootstrap/test_paths.py test/bootstrap/test_shell_commands.py desktop/backend/tests/test_shell.py` |
| `src/finesub/llm/agent/{agent_task_runtime,agent_transports,agent_retrieval,agent_task_control}.py`（durable task 协议：租约/回收/blocked 出口/预算 ledger/会话谱系） | `pytest -q test/test_llm_agent_task_runtime.py test/test_llm_agent_transports.py test/test_llm_agent_retrieval.py test/test_llm_agent_task_control.py` |
| `src/finesub/llm/session_checkpoint.py`、`knowledge/snapshot.py` | `pytest -q test/test_llm_session_checkpoint.py test/test_llm_knowledge_snapshot.py` |
| `src/finesub/llm/agent/{agent_quota,agent_ping}.py`（订阅耗尽判据、**按额度池**冻结、`finesub agent-ping`） | `pytest -q test/test_llm_agent_quota.py` |
| `src/finesub/llm/routing/model_routes.{py,toml}`、`routing/model_catalog.{py,psv}`（target/模型组/预设/policy 的声明与校验、快速选模型、双 digest、出厂预设与下限告警） | `pytest -q test/test_llm_model_routes.py test/test_llm_config_and_budget.py` |
| `src/finesub/llm/chunking.py`, `prompt_templates/`, `prompt_artifacts.py` | `pytest -q test/test_llm_srt_and_chunking.py test/test_llm_prompts.py test/test_llm_prompt_compose.py` |
| `src/finesub/subtitles/{alignment,metrics,model,postprocess,rendering}.py`, `src/finesub/llm/output_protocol.py`, `task_report.py` | `pytest -q test/test_llm_srt_alignment.py test/test_llm_subtitle_metrics.py test/test_llm_output_protocol.py test/test_llm_srt_and_chunking.py test/test_llm_srt_postprocess.py test/test_srt_rendering.py test/test_llm_task_report.py` |
| `src/finesub/{pipeline,batch,paths,run_metadata}.py`, media/subtitles/speech import boundaries, speech stage/stabilization/separation/runtime gate, packaging | `pytest -q test/test_asr_stabilize.py test/test_pipeline_refactor.py test/test_batch_runner.py test/test_import_boundaries.py test/test_paths.py test/test_run_metadata.py test/test_vocal_separation_pool.py test/test_gpu_stage_gate.py test/test_packaging.py` |
| 显存/内存预算实测（真跑分离+ASR，重资源） | `pytest -q test/test_resource_budget_pipeline.py --run-heavy-resource -n 0` |
| `src/finesub/speech/recognition/{transcribe,checkpoint,segments}.py`, `src/finesub/speech/postprocessing/segmentation.py`, `src/finesub/speech/preprocessing/{vad,energy}.py`, `src/finesub/text.py` | `pytest -q test/test_asr_and_text_utils.py test/test_segment_split.py test/test_vad_streaming.py test/test_vad_segment_energy.py` |
| `src/finesub/speech/runtime/{resources,gpu_stage_gate}.py` | `pytest -q test/test_resource_profiles.py test/test_gpu_stage_gate.py` |
| `src/finesub/speech/recognition/word_starts.py`、`vad_asr_stage.py` 的 `vad_timeline` 产物 | `pytest -q test/test_word_starts.py test/test_pipeline_refactor.py test/test_vad_streaming.py` |
| 跨全树的源码守卫：`subprocess.run(text=True)` 必须带 `encoding=`（中文 Windows 上不带就崩，见 `CHANGELOG.md`） | `pytest -q test/test_subprocess_text_encoding.py` |
| `src/finesub/speech/recognition/vad_asr_stage.py` 的 VAD 阶段产物（`-vad.json` + 帧级 npz 的读写、失效判据、与 pipeline/清理的对应） | `pytest -q test/test_vad_prefix.py` |
| `src/finesub_bootstrap/fsops.py` 的发布性重命名（两档预算、两处故意不重试） | `pytest -q test/bootstrap/test_fsops.py test/bootstrap/test_downloader.py` |
| `src/finesub/reporting.py`、各阶段的上报点（契约见 [`reporting.md`](reporting.md)） | `pytest -q test/test_reporting.py test/test_pipeline_reporting_boundary.py` |
| `src/finesub_bootstrap/` 的下载族：`download_routes`/`download_sources`/`downloader`/`asset_resolve`/`model_fetch`/`model_ensure`/`hf_verify`/`model_manifest`（契约见 [`download-routes.md`](download-routes.md)） | `pytest -q test/bootstrap/test_download_routes.py test/bootstrap/test_downloader.py test/bootstrap/test_hf_verify.py test/bootstrap/test_model_ensure.py test/bootstrap/test_model_caches.py test/bootstrap/test_asset_resolve.py`，另加 `test/bootstrap/test_cn_lock.py test/bootstrap/test_runtime_regional_lock.py` |

也可用域标记代替显式文件列表，例如 `pytest -q -m llm`。

**下载路径的默认单测一律不联网**：fake HTTP、fake subprocess、小文件，不访问公共镜像也不加载
模型。真正的实机验收（全量摘要比对、全新安装、破坏哈希、中断续传）已于 2026-08-21 跑过一次，
逐项结果在 [`download-routes.md`](download-routes.md) §9；它不是回归套件的一部分，也不需要
每次发版重跑——除非改了镜像表、lock 生成器或模型清单。

## 标记说明

三个域标记在 `test/conftest.py` 的 `_PIPELINE_FILES` / `_ASR_FILES` 里按**文件**声明
（`test_llm_*.py` 由前缀自动进 `llm`）。

| 标记 | 含义 |
|------|------|
| `llm` | `test_llm_*.py`（conftest 按前缀自动加） |
| `pipeline` | 管线/批处理编排，以及一次运行所依赖的运行时与预置层：GPU 与线程预算、设备判定、分离器自身的管道、终端 reporting、config、secrets、state |
| `asr` | 音频 → 词 → 字幕这条路：VAD、解码、对齐、稳定化、分句，以及它们用的文本工具 |
| `heavy_resource` | GPU/大模型/大音频；默认 skip，需 `--run-heavy-resource` |
| `network_mock` | 起真实的本机 socket 去证明一条路由（目前只有 `[llm] proxy` 的转发证明）；默认 skip，需 `--run-network-mock`。快且不出机器，但不该让每次 `pytest -q` 都去绑端口 |
| `requires_main_checkout` | 契约依赖当前仓库就是主 checkout（例如真实 checkout 路径、知识库 apply/commit）；linked worktree 中自动 skip |

**三个域标记是一个划分**：每个测试文件恰好带一个，`-m "llm or pipeline or asr"` 收集到的
就是全量（与 `pytest -q` 一致）。这条由 `test_packaging.py::test_the_domain_markers_cover_every_test_file`
守住——新增文件没登记会直接失败，顺带也拦住改名后留下的死条目。加新文件时把它加进
`test/conftest.py` 对应的那个元组即可。

**登记用的是相对 `test/` 的 posix 路径**（`bootstrap/test_paths.py`），不是裸文件名：套件
有子目录之后，两个目录各有一个 `test_paths.py` 是常态，裸文件名会让其中一个默默继承另一个
的标记。守护测试同样按 `rglob` 递归收集——搬进子目录的文件不能就这样滑出划分。

`testpaths` 里唯一不在 `test/` 下的
`desktop/scripts/tests/test_desktop_dependencies.py` 够不着 `test/conftest.py`，所以它在文件
开头自带 `pytestmark = pytest.mark.pipeline`；同一条测试会检查 `testpaths` 是否又长出了没标
记的条目。

### 没有 `slow` 标记

`slow` 已于 2026-08-15 连同 36 处用法一起删除。它不是「不好用」，而是**和真实耗时反相关**：
被它标住的 36 例里有 31 例来自 `test_llm_search_loop.py`（模块级 `pytestmark`），单例全部
低于 0.27 秒；而实测最慢的三例——`test_llm_local_agent.py` 的两个子进程用例（各 3.3 秒）和
`test_state_store.py::test_concurrent_writers_do_not_lose_each_others_sections`（3.06 秒）——
一个都没被标住。`-m "not slow"` 于是恰好踢掉最快的那批、留下最慢的那批。

要省时间就按实测来（`-n 0` 全量 136 秒）：

| 位置 | 耗时 |
|------|------|
| `test_vad_streaming.py`（5 例） | 34.2 秒，占全量 25% |
| `test_llm_local_agent.py` | 次之，大量真起子进程的用例 |
| 其余单例 | 均 < 3.1 秒 |

`test_vad_streaming.py` 那 34 秒串在一个 worker 上，所以**加 worker 也压不到 35 秒以下**，除非拆那个文件。真想临时跳过，用
`--ignore=test/test_vad_streaming.py` 显式说，比维护一张会失准的标记表可靠。

## 测试替身放哪：一个名字可能有多个查找点

**`monkeypatch.setattr` 只改一个模块的绑定。** 一个名字被拆进多个模块后，patch 其中一个
不会炸——只是那条测试悄悄不再测另外几个。实测两例：`load_entry_texts` 在纠错子包的四个
模块里各绑一份、`PROMPT_VERSION` 被 `run`（窗口断点指纹）与 `query_round`（会话 checkpoint
哈希）各绑一份，只 patch 前者会让「contract 一 bump 全部断点作废」这条测试半真。

两条随之而来的规矩：

- **替身放在真正做名字查找的那个模块上**，不要放在包的 `__init__`——patch 一个 re-export
  等于没 patch，却能骗过存在性断言。纠错子包有现成的 `test/conftest.py::setattr_correction`：
  它 patch 所有绑定该名字的模块，且至少命中一个才通过。
- **把函数内的延迟 import 提到模块级之前，先看这个名字有没有被 patch。**
  `from .x import f` 会让 `x.f` 上的 patch 失效；要提就写成 `from . import x` + `x.f()`
  ——依赖照样写在文件顶部，名字仍在调用时解析。反过来，有些按名绑定是**故意**的
  （测试要替换的正是「这个模块的视图」），移动前先读注释。

拆模块或改名时这两条会一起出现，见 [`refactor-followups.md`](plans/refactor-followups.md) §5。

## 为什么仓库根有一个 conftest.py

`conftest.py`（**仓库根**，不是 `test/` 下那个）只做一件事：在**不能创建符号链接**的机器上
把 pytest 私有的 `_pytest.pathlib._force_symlink` 换成 no-op。

pytest 的 `tmp_path` / `tmp_path_factory.mktemp` 都走 `make_numbered_dir`，它会给最新的临时
目录挂一个 `<prefix>current` 便利链接——pytest 自己的 docstring 称之为 "best effort"，本仓库
没有任何代码读它。Windows 上没有 Developer Mode 或管理员权限时，背后的 `os.symlink` 必然抛
`WinError 1314`（pytest 吞掉继续），而**这次失败要 0.205 秒**，对比它装饰的那个 `os.mkdir`
只要 0.0003 秒。`test/conftest.py` 有两个 autouse fixture 各取一个临时目录，于是**每个用例
在跑第一行之前就付了两次**，再要 `tmp_path` 的还要付第三次——成本随用例数线性增长，而不是
随它们做的事增长。这就是「本机越来越慢、CI（Linux，符号链接正常）毫无感觉」的形状。

实测：全量套件 **38 分钟 → 1 分 21 秒**（当时的 `-n 2`，1893 passed / 21 skipped），只加这一个文件。

写法上的三个约定，改动它之前先读一遍：

- **探测而不是判平台**。能建符号链接的机器（Linux、CI、开了 Developer Mode 的 Windows）
  保留 pytest 文档化的行为；只有建不了的才跳过。谁哪天开了 Developer Mode，链接自己就回来，
  不需要有人记得回来删代码。探测本身每进程一次，约 0.28 秒。
- **`hasattr` 兜底**。`_force_symlink` 是私有 API，将来改名的代价应该是丢掉加速，
  而不是套件跑不起来。
- **放仓库根**。三个套件（`test/`、`desktop/backend/tests`、`cli/tests`）的 rootdir 都解析
  到仓库根，一处覆盖全部；放 `test/conftest.py` 只管一个。

## 结构守卫：六条自动化规则

这些测试不测行为，测**结构约束**，因为它们各自对应一次「什么都没报错，但事情已经错了」：

| 守什么 | 文件 | 出错时长什么样 |
| --- | --- | --- |
| 导入方向（media/subtitles 不向上依赖、`speech` 不依赖 `llm`、别处不得 import `finesub.pipeline`） | `test_import_boundaries.py` | import 照常成功，分层悄悄没了 |
| 选项默认值只有一份（argparse 不得抄后端的答案） | `test_option_defaults.py` | 两处默认值慢慢漂开，前端与后端行为不一致 |
| 文档链接与章节号落得到（含源码注释里的、含中文数字节号） | `test_doc_links.py` | 链接照常渲染，指向不存在的东西 |
| **单个函数不得超过 300 行，已超的不得再长** | `test_function_size.py` | 没有任何一刻需要有人判断「这个函数是不是太大了」——`run_search_loop` 就是这样长到 882 行的 |
| **文档枚举的集合必须与代码一致**（当前只守 GPU 档位名） | `test_doc_facts.py` | 代码加了两个档位，文档的表还是三行，而且这类错误已经复发过一次 |
| **中文正文用全角标点**（范围只有 `docs/manual/`） | `test_doc_style.py` | 文档照常渲染，但下一次内容改动的 diff 被标点噪声淹掉——2026-09-03 一次 700 行改写里真正的变动只有六个选项名 |

⚠ `test_function_size.py` 的 `_ALLOWED` 是**棘轮不是豁免名单**：每条带着测量当天的行数，
断言是 `<=`，掉回 300 行以下的必须删条目。要让某个函数变大，得手动改那个数字——
那正是这个决定该被看见的时刻。

⚠ `test_doc_facts.py` 只守**可枚举**的事实（一组名字），不守数字与描述。逼近「读懂句子」
的守卫误报率高到没人维护，那比没有守卫更糟。

⚠ `test_doc_style.py` 的范围**只有 `docs/manual/`**，因为只有那里归一化过。`docs/` 根与
仓库根还各有一百多处 CJK 相邻的半角（`CHANGELOG.md` 里还有不该改写的历史条目）。要扩大
范围，**先归一化再改 `_SCOPES`**，顺序反了就是一片红。它遮蔽代码块、行内代码、链接目标、
URL 与 HTML——遮蔽面和做归一化时用的是同一套，否则守卫会去管它没管过的地方。

## 备注

- PowerShell 下不要用 `test/test_llm_*.py` glob（不展开）；用显式文件列表或 `-m llm`。
- `test_pipeline_refactor.py`、`test_intervals.py`、`test_vad_streaming.py` 收集时会 import torch，collect 略慢属正常。
- 默认测试套件不生成 prompt preview。重打工具（`tools/session_replay`）按需维护，
  其测试不被默认收集，需要时显式运行 `python -m pytest tools/session_replay -n 0`。
- 离线探针（`tools/bench`）同理不被默认收集：`python -m pytest tools/bench`。
  它们的依赖在 **`[bench]` extra**（`onnx` / `sentencepiece` / `scikit-learn`；
  `onnxruntime` 已在 `[asr]` 里），装机是 `pip install -e ".[asr,bench]"`——
  标准的 `[asr,harness,dev]` 跑这些探针会直接 `ImportError`。
- 默认单测不得加载 Whisper/audio-separator、处理大音频或消耗 Gemini quota。
- `.github/workflows/desktop-ci.yml` 另有 Python 3.10 的薄 CLI job：构建并安装 wheel，在 managed
  runtime 不存在时运行 `finesub agent-clean`，守住 cleanup 的轻依赖/不 provisioning 契约。
- ⚠️ **`test_fw_refine.py` 整个模块在 CI 里从不执行**：开头两条
  `pytest.importorskip("scipy" / "faster_whisper")` 会整模块 skip，而 CI 只装
  `[harness,dev]`——`[asr]` 装不了，patched CTranslate2 wheel 是 Windows 专属的。
  在装了 `[asr]` 的机器上跑同一条命令则会执行（这就是为什么本机计数比 CI 多约 25 个）。
  **代价最大的一条在这里面**：`test_a_cpu_model_decodes_and_then_shuts_down`
  （`heavy_resource`，另需 `--run-heavy-resource`）守的是 CPU GEMM 后端选错导致的
  死锁/`No SGEMM backend on CPU`，两种都只在真解码时暴露。也就是说**换 CT2 wheel 后必须
  在本机手动跑它**，CI 绿不代表这条验过：

  ```powershell
  python -m pytest test/test_fw_refine.py --run-heavy-resource -q
  ```

  完整的换 wheel 验收清单见 `tools/wt_refine_port/ct2-patches/README.md`。
- linked worktree 的 `.git` 是指向主仓的文件。`test/conftest.py` 据此跳过
  `requires_main_checkout`，但显式验证 worktree 重定向/写保护的测试仍正常运行；回到主
  checkout 后这些用例自动恢复，不需要额外参数。
- `test_resource_budget_pipeline.py`：真跑「分离 → VAD-ASR」，断言实测 peak 显存/内存不超过
  所选档位上限；参数化跑遍 `entry`/`standard`/`high` 三档。需 CUDA（无 CUDA 自动 skip）与
  已缓存的 audio-separator 模型。
  - **分离那一半**用合成音频，一直有效。`RESOURCE_TEST_SECONDS` 可加长音频以压 RAM 路径——
    也是够到多 worker 的唯一办法（300 秒阶梯：>300s 才 2 个，>600s 才 3 个）。
  - **VAD-ASR 那一半需要真实语音**（`assets/harvard.flac`，`assets/` 不进仓库），缺失时
    明确 skip。合成音频是包络下的纯音，能骗过能量 VAD 但解码器转不出任何东西——2026-09-01
    实测三档 segments 全为 0。它此前「通过」，靠的是分离器泄漏的权重喂饱了那个
    `peak_gmem > 256MB` 守卫；守卫现已改成从产物验。
  - 同文件的 `test_the_separator_gives_its_weights_back` 是显存/内存泄漏回归：连跑三次分离，
    CUDA 与**主机侧** torch 张量残留都必须有界（只看 CUDA 会漏掉「把权重搬去内存」这种修法）。
