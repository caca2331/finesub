# docs/ 怎么分

一条约定，此前只存在于习惯里：

- **`docs/manual/`——写给使用者。** 装在哪、怎么配、怎么搬、出了问题看什么。可以提到
  模块名，但不要求读者读得懂代码；命令一律给能直接粘的形式。`README.md` 的「文档」一节
  链的就是这些。
- **`docs/` 根下——写给开发者和维护者。** 行为契约、算法、产物字段、实验记录、重构计划。
  默认读者能读源码，也会去读；术语不解释第二遍。

同一个话题两边都有，是正常的，不是重复：`manual/resources.md` 说「模型装在哪、怎么删」，
`README_DEV.md` 说「哪些产物是记录、哪些可删、谁来删」。**分界是读者，不是主题。**

⚠ 一条只能靠人守的纪律：**写「细节见 `X.md`」之前，先去 X 里确认那段真的在。**
`test_doc_links` 只验链接指向的文件存在，验不了它有没有讲那件事——2026-09-01 一次审计
逮到两处空头承诺（`download-routes.md` 说用户向说明在 `manual/resources.md`、
`manual/agent.md` 说 `config.toml` 的创建方法在 `manual/resources.md`，两者当时都不存在），
两处都是「甩出去但没人接住」。这没有便宜的自动判据可写——「X 里有没有讲 Y」是语义判断，
按关键词近似的规则假阳性高到没人会维护——所以它是改文档时的复核动作，不是测试。

## 文档地图

被跟踪的 `docs/*.md` 全量清单，按领域分组。每行：文档 + 一句话主题 + 状态标注。
状态标主类，两类兼有的写成 `A · 含 B`：

- **规范**——现行行为契约，改行为前先读。**项目「已完成」算规范，不算历史**
- **台账**——活的未完成项与决策记录，现状以它为准
- **设计稿**——未落地方案，只作参考
- **实验记录**——实测数据与探索过程，数字可能已过时
- **历史**——描述的东西**已从代码里移除**，只用于回溯

### 面向使用者

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `../README.md` | 用户入口：安装、快速上手、常见命令 | 规范 |
| `manual/env.md` | API key 与 `.env`：Gemini/Exa/Tavily、Windows DPAPI 加密、`keys`/`doctor` | 规范 |
| `manual/repo-install.md` | 源码安装全步骤：uv 默认 / pip 替代（torch 必须走 cu128 索引的坑） | 规范 |
| `manual/batch.md` | 一次跑多个输入：manifest 行的写法与三个非选项键、**运行期队列面与控制面**（加任务/插队/撤掉）、**`--resume-batch`**（含 7 天、活批、异目录三道拒绝与「显式选项盖过记录值」）、产物唯一性、失败/中断/重跑、事件流与逐项日志 | 规范 |
| `manual/resources.md` | 数据落在哪、`relocate` 搬盘与共用、卸载档位、缓存为何单独删没用、worktree 模式；显卡支持范围、`--gpu-tier` 五档与 CPU 回退 | 规范 |
| `manual/outputs.md` | 运行产物：`out/<名字>/` 里每个文件是什么、`--stage` 六个值各停在哪、`-annotated.csv` 九列怎么读（`conf` 是 LLM 自评，不是 ASR 置信度）、想重跑某一步该删什么 | 规范 |
| `manual/tuning.md` | 调参：字幕长短（`--split-length-scale` / `[segmentation]`）、识别侧与 LLM 侧各一张旋钮表（默认值 + 改完要删什么）、`--no-download-video`、一个耗时量级参考 | 规范 |
| `manual/troubleshooting.md` | 故障排查的总目录：按症状（回退 CPU、显存告警、空字幕、429、设置没生效…）指到那一页；末尾是提 issue 该带什么 | 规范 |
| `manual/ct2-wheel.md` | patched CTranslate2 怎么装、怎么自检、装错了什么症状 | 规范 |
| `manual/models.md` | 模型选择：ASR 三个可选 Whisper（默认 turbo / large-v3 无优势、日语微调实测打平）、不可换的分离器与第二模型、各 LLM 后端的使用印象 | 规范 |
| `manual/model-routing.md` | 一次调用怎么定下来：会话→任务组→预设格子→候选过滤、媒体/difficulty/思考旋钮、catalog、接自己的模型、启动告警、改什么会作废 checkpoint | 规范 |
| `manual/knowledge.md` | 知识库用户向：它记什么、在哪、改内容的四条路（含「改 rendered/ 要等下一次纠错运行才收割」）、三档 dry-run 的后果、共享与冲突、故障对照表 | 规范 |
| `manual/agent-tasks.md` | 让 agent 替你做的几件事：审 run、把资料收进知识库、打反馈包、发版；说什么话它就去读哪一份、为什么不需要安装配置 | 规范 |
| `manual/agent.md` | 用本机 Codex / Claude Code / agy 订阅或 DeepSeek Harness 代替 Gemini 额度；档位选择、失败行为、`agent-clean`、搬盘/卸载 | 规范 |

### 开发与维护总入口

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `../README_DEV.md` | 开发原则、资源约束、canonical artifact tree、reuse/resume 规则、agent checklist | 规范 |
| `testing.md` | test markers、常用命令、哪些测试盖哪些路径 | 规范 |
| `data-index.md` | 数据与基线索引的**规则那一半**：三类划分、跟踪标注的口径与警告、只存在于文档的实测基线。逐条清单（含 BV 号与本机路径）在本地 `data/index.md`，不进 git | 规范 |
| `bench-discipline.md` | **动性能前先读的那一份**（短）：一个数字算数的六个条件、`tools/bench/discipline.py` 的强制项、以及「本机不是干净测量环境」这条读数前提。2026-09-03 从 `bench-baselines.md` 第一节搬出 | 规范 |
| `bench-baselines.md` | 2026-08-27 换机后重取的本机基线与二十三节实验记录（编号从「二」起，一律不重编号）。哪些结论已经关闭也记在里面 | 实验记录 |

### 语音链路（VAD / ASR / 分句 / 分离器）

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `vad-energy.md` | energy VAD 本体：处理流程、流式=内存契约、峰值 clamp 取舍、`WaveformObserver` 钩子、Python API | 规范 |
| `vad-asr.md` | VAD→ASR 组合阶段：CLI（含 `--vad-silero-assist`）、数据流、aligned JSON 字段契约、失败行为 | 规范 |
| `asr-align.md` | interval→aligned ASR：解码配置、词级映射、异常救援阶梯、覆盖率救援、输出字段语义 | 规范 |
| `asr-stabilize.md` | aligned→stable：profiles/metrics/tags/CLI、resume 规则（profile 3 pre-merge 移除的缘由记录在内） | 规范 |
| `segmentation-split.md` | 分句规范：全局 DP 打分、gap 调整、字段继承与幂等 | 规范 |
| `segmentation-gold.md` | 分割点金标准：必切/禁切/宜切判据、时间轴锚定、打分口径。审计分割质量前先读 | 规范 |
| `gpu-profiles.md` | 五档映射（`cpu`/`entry`/`standard`/`standard_large_vram`/`high`，`cpu` 是策略档）、`auto` 定档与上限、最大窗口实测与并发依据 | 规范 · 含实测 |
| `separator-optimization.md` | BS-Roformer 推理效率探索（E0–E11）：已采纳 AMP+编译路径、交付两模式、已否决方案 | 实验记录 |
| `batch-scheduler.md` | 三 bin（download×2 / asr×1 / llm×2）的并发语义、llm bin 准入、运行期队列面与控制面、`--resume-batch` 与批次身份、产物唯一性、失败与中断语义。`scheduler.py` / `batch_state.py` 的 owner；2026-09-03 从 `wt-parallelism.md` 整节搬出 | 规范 |
| `speech-followups.md` | speech 侧未完成工作的索引（对称于 `llm_followups.md`）：五项没开工 + 两项交付了但默认关着（A1 组批、P9 `--asr-context`，翻默认各欠一份证据）、各批「不要做什么」摘要、散在别处的未做项。⚠ 只是索引，状态真相源是 `plans/crispasr-followups.md` 的状态总览表 | 台账 |
| `wt-parallelism.md` | 单文件 WT 分片（已移除）：回溯点、仍成立的结论（asr 恒 1 的并发实测依据、stdio 背压根因、intra-op 线程预算）。**现行 batch 契约已搬去 `batch-scheduler.md`** | 历史 |
| `wt-refine-handoff.md` | CT2 WT refine 研究交接入口（已合入 dev）：结论、交付决策、剩余待办 | 规范 |
| `wt-refine-port.md` | WT refine→FW/CT2 详细算法契约、multi-audio batch 设计与档位表、质量实测 | 规范 |
| `wt-refine-validation.md` | 13-group 信号与局部隔离验证结果 | 实验记录 |
| `gemini35-transcribe.md` | 外部转录模型 `gemini-3.5-transcribe` 的调用规则：两条通道各自的配额与撞限形态、互斥开关、按词数切窗、Live 的三条必须、拿它做仲裁的纪律。**不是生产管线的一部分**，是调查工具 | 规范 |

### LLM harness

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `llm_harness_behavior.md` | LLM 运行时 canonical 总入口（文首有拆分导航）：开关轴、窗口拆分与调用形态、SRT 后处理、知识库更新 | 规范 |
| `llm_harness_routing.md` | 路由 dev 侧：模型事实/池/路由链、thinking 档位换算、模型配置与限流。使用者向见 `manual/model-routing.md` | 规范 |
| `llm_harness_research.md` | 本地检索代理：Exa→Gemma4 grounded→Tavily 降级链、按轴退化的 r1/r2 | 规范 |
| `llm_local_agent.md` | Agent 执行后端唯一入口：三家 one-shot transport 契约、durable task 协议、tier 冻结、会话档位接线（§12.1） | 规范 |
| `llm_agent_tool_protocol.md` | agent 工具化协议现行规格：工具表与 request id、必读块台账、审计包、四家 driver 接线 | 规范 |
| `llm_local_agent_experiments.md` | 长驻会话准则与实测：会话复用 A/B、缓存写入门槛成因与复测、Claude Code 反向信号 | 实验记录 |
| `llm_local_agent_runtime.md` | 执行环境卫生：episode 落点、capsule 是一次性 episode、滚动上限 20 与清理 | 规范 |
| `llm_local_agent_agy.md` | agy 专属：catalog 行、音频必须容器化、视频分辨率不可调、`view_file` 准入硬门 | 规范 |
| `llm_prompts.md` | prompt 模板/碎片、`prompt_compose` 组装表、PROMPT_VERSION 语义 | 规范 |
| `llm_design_notes.md` | 架构意图、路由 durable 决策与理由、预算公式推导、知识更新决策台账、deferred designs | 规范 |
| `llm_followups.md` | LLM 未完成实验/设计唯一入口：按 harness / Agent / 知识库 / 前端分节，每条带触发条件或重启前提；已完成项不留正文（2026-09-02 整理前的原文在本地 `docs/archive/llm_followups-2026-09-02-before-tidy.md`） | 台账 |
| `knowledge.md` | 知识库一切：结构、`--knowledge` 三态、feedback v2、统一更新、mistake ledger | 规范 |
| `provider-adapters.md` | 自定义 provider（OpenAI-compat/Anthropic 纯文本）：差异调研表与 adapter 契约 | 规范 |
| `prompt-iterate.md` | 纠错 prompt 迭代方法论：定位与唯一机制、四变体、session_replay 协议、失效模式 | 规范 |
| `session_replay.md` | replay 协议：6 个 session 的 fixture/validation 契约、补中间态落盘 | 规范 |
| `merge-calibration.md` | 合并软门槛标定：默认不并、gap/字数先验、flash-lite thinking=0 | 实验记录 |

### 分发、前端与跨前端

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `reporting.md` | `finesub.reporting` owner 文档：八个事件契约、级别与 renderer、各阶段上报口径、LLM 段事件词表 | 规范 |
| `download-routes.md` | `finesub_bootstrap` 下载族 owner 文档：地区解析、镜像降级、cn lock 约束、marker 四态、续传 | 规范 |
| `ct2-distribution.md` | patched CT2 打包与分发（面向维护者）：自包含 wheel、为何发 Release、CUDA 12 边界 | 规范 |
| `cross-frontend-lease.md` | 跨前端租约（已完成）：user-data 闸门、独立租约、sidecar、崩溃判活；§4 是三条不做的事 | 规范 |

### `plans/`——已结案的计划稿

**分界线**：根目录放**契约与活台账**，`plans/` 放**已结案的计划**。所以
`llm_followups.md` 与 `speech-followups.md`（都是「还没做的是哪些」）留在根，
而 `refactor-followups.md` 名字像台账、记的却是押后理由与教训，进这里。

读它们是为**取舍依据**——「为什么这么做」「为什么不做 X」——现行行为一律以 owner 文档为准。
2026-09-03 从根目录整体搬入，正文未改。

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `plans/docs-reorg-plan.md` | `docs/` 自身怎么整理：三处体裁混住（`wt-parallelism` 里的现行 batch 契约、crispasr 状态总览 30 行里 24 行已结案、bench 219KB 混纪律与流水账）、两处标签失准、speech 侧缺唯一入口；五步方案 + 明确不做的五件事。**owner 2026-09-03 已拍板并全部执行**（不改名不归档、新建薄 `speech-followups.md`、建 `docs/plans/`）；此后只作取舍依据 | 台账 |
| `plans/crispasr-followups.md` | 2026-08-29 外部对照后的待办：六批（守卫 / 测量 / 加速 / 契约 / 新能力 / diarization 工程层）+ 明确不做的七条。**2026-08-31 起大部分已完成或结案，现行状态以它的「状态总览」表为准**——30 行里 23 行已结案，未结案 7 行（五行没开工 + 两行交付了但默认关着）| 台账 |
| `plans/refactor-followups.md` | 2026-08 重构完成后的剩余：押后的七项、不做的五条、四条教训（tokcount 三步已随 0.4.0 做完） | 台账 |
| `plans/stage-device-plan.md` | 逐阶段设备解析 + 拆分 `entry` 与 CPU 的方案：现状核查（ASR 拿 torch 的答案去问 CT2）、与 `device.py` 哲学冲突的那条回退、五步顺序。**已全部实施**；§6 记着两条验收手法（进程内劈开 torch 与 CT2、`CUDA_VISIBLE_DEVICES=-1` 无卡回归） | 台账 |
| `plans/translation-style-plan.md` | 翻译风格库：取舍依据与 owner 决定台账。**六步全部结案**（§4 的实验 2026-09-02 判定不做、N/L 已拍定；存量迁移已做完，27 条范例迁入、27 条错误库判定不迁）；剩的是共享那一次跨库验收与 §6 两条未决（style 没有 `landed` 信号、删除权与证据）。**现行行为在 `knowledge.md`** | 台账 |
| `plans/knowledge-node-plan.md` | 知识库 node 模型 / 检索分级 / 三层信号 / 共享库设计稿（取代已归档的打分方案；2026-08-28 的 kb-followups 迭代与 2026-08-29 的行文法 v3 重导都已实施，计划正文在本地 `docs/archive/`，取舍依据蒸馏进 `llm_design_notes.md`）。⚠ **§8 与 §11 都已全部落地**，读它是为取舍依据与 owner 锚点，不是未竟计划 | 台账 · 含设计稿 |
| `plans/conversational-live-test-plan.md` | conversational 首次真机实测：读数、查证到行的代码事实、四条取舍与六步计划 | 台账 |
| `plans/model-window-limits-plan.md` | 窗口限额三档化：catalog 加 `context_window`，删掉 `DEFAULT_LIMITS` 那两个当上限用的 `min(...)`（本文代称 `HARNESS_INPUT_CAP` / `HARNESS_OUTPUT_CAP`，**代码里没有这两个名字**）与 `context_limit` / `safety_margin` 两个字段，planner 只剩两行算术。§6 是风险与前置（为什么 P6 标定这次不阻塞），§7 是一份独立的 catalog 可疑值审计，§9 是实施记录（五条 owner 裁定 + 方案自己写错的一处 + 复审后追加的那处扩展） | 台账 |
| `plans/desktop-split-plan.md` | 0.5.0 把 `desktop/` 移出本仓：盘点结论（没有 Python 文件 `import desktop`，剥的是构建面）、两段执行顺序（阶段 A 把四份共享资产搬出 `desktop/`，已完成；旧路径一律作废）、§5 删目录后会红的十余处守卫与两处会丢东西的缺口（B0 的 78 条共享层测试，其中 6 条在函数体内 import 桌面；B3 的整条 Windows lane）、§7 明确不做、§8 四条已定加一条未决、§9 三轮复审与阶段 A / 锚点 / 阶段 B 的实施记录。**A、B 与锚点均已完成，只剩阶段 C 发版**（2026-09-03） | 台账 |
| `plans/field-feedback-batch-plan.md` | 一轮用户反馈带出的五项，互不依赖、可单独落地：§1 HF 镜像下 Xet 401 让 `cn` 装不上模型（附带查出 `is_mirror_failure` 不认 401，连回退官方源都不会发生）、§2 模型组窗口下限的 warning/退出（⚠ 扫 `model_groups` 而非 catalog，否则误伤只做 grounded search 的 `gemma-4-31b`）、§3 关键 API 交互进 run 日志（只写状态与一句话描述，正文留在 `exchanges/`）、§4 反馈打包 agent-task（两模式、去重台账、隐私边界）、§5 `--no-separate`。§6 明确不做五条，§7 owner 六条决定（**无未决**），§8 两轮复审记录六条，§9 实施记录（四处偏离计划 + 棘轮两次拦下都改成拆分）。⚠ §2.1.1 是最容易做错的一节：闸门比 catalog 的 `max_input/max_output` 两列，**不比 `group_planning_envelope` 的规划包络**——owner 裁定 haiku 放行，`context_window` 的总量约束不进闸门。**五项已全部实施**（2026-09-03 当天起草、复审、落地） | 台账 |

## 找东西

两份索引，两个读者。**本文件的地图**给做文档维护的人：全量清单 + 主题 + 状态。
**`CLAUDE.md` 的「Docs index」**给 agent：同一批文件名，按领域分组，加少量**判断提示**
（唯一入口、状态异常、踩坑）与「索引里没写、要翻代码才找得到」的几处。分界写死：
**描述只写一份，在这里；判断提示只写一份，在 CLAUDE.md。文件名两边都有**——
名字漏在哪一边，那一边的读者就看不见这份文档，所以 `test_doc_links.py` 两边都盯。

除本文件自身外，被跟踪的 `docs/` 下的每一份 md——根目录、`manual/`、`plans/`——都应当出现在上面的地图里（路径按相对 `docs/` 写，例如 `manual/env.md`、`plans/refactor-followups.md`）。`docs/archive/` 与
`docs/report/` 是本地笔记：在 `dev` 上被跟踪，但由 `scripts/publish-main.ps1` 从公开快照里
剥掉，所以不随仓库发布，也不进地图（`test_doc_links.py` 因此既不扫它们的链接、也不要求
它们进索引）；往里迁文档前先按 `CLAUDE.md` 的 **Archive extraction** 规则把仍然成立的
事实提回被跟踪的文档。
