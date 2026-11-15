# docs/ 怎么分

分界是读者：

- **`docs/manual/`——写给使用者。** 装在哪、怎么配、怎么搬、出了问题看什么。可以提到
  模块名，但不要求读者读得懂代码；命令一律给能直接粘的形式。`README.md` 的「文档」一节
  链的就是这些。
- **`docs/` 根下——写给开发者和维护者**：行为契约、算法、产物字段、实验记录。默认读者能读源码，也会去读；
  术语不解释第二遍。（常见的做法是另建 `docs/dev/` 放这些；本项目不搬目录，引用太多，搬了只是换名字。）

同一个话题两边都有，是正常的，不是重复：`manual/resources.md` 说「模型装在哪、怎么删」，
`README_DEV.md` 说「哪些产物是记录、哪些可删、谁来删」。**分界是读者，不是主题。**

## 写文档的规矩

1. **先写给人。** 结论、限制、要人决定的放前面；只供 agent 的执行细节放末尾，标题写
   「Agent 参考，人可跳过」，长了就拆出去。
2. **只写现在成立的事。** 失效的内容直接改写，不在旧正文后面追加更正；过程与流水不进正文，
   值得留的教训写明成立条件。
3. **按名字引用。** 引用其他章节用带标题的锚点链接，同一文档内也一样。旧编号保留，但引用要带
   标题；实验、阶段 ID 带名称（「E7（分离器 worker 数）」）。
4. **plan 开头是状态块。** 标题下第一块是引用块，以 `**状态**` 开头，至少回答：做到哪、验证到哪、
   还剩什么、要不要人决定。状态变了就原地改，不追加。
5. **甩出去之前确认接得住。** 写「细节见 X」之前，先去 X 里确认那段真的在。`test_doc_links`
   只验文件、锚点和 `X.md「标题」` 里的名字落得到，验不了那里讲没讲这件事——这是语义判断，按关键词近似的假阳性高到没人会
   维护，所以它是改文档时的复核动作，不是测试。
6. **定了的事写进决策记录。** LLM 层的写进 `llm_design_notes.md`，speech、评测与跨层的写进 `decisions.md`：
   定了什么、理由、重开条件。follow-up 文档（`llm_followups.md`、`speech-followups.md`）只放开着的工作，
   不做的条目只留一行、指到它的记录（owner 2026-11-08）。

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
| `../README.md` | 用户入口：安装、快速上手、常见命令。「文档」一节不再逐份列举 manual，改为指向下面那份索引（该节另有知识库样板与 dev 侧入口两条链接） | 规范 |
| `manual/index.md` | `manual/` 自己的索引，写给**使用者**：按任务分组（第一次上手 / 跑起来之后 / 模型、额度与硬件 / 出问题了 / 交给 agent 做）指到本节其余每一份（`test_doc_links` 盯着这条覆盖面）。与本地图的分工是读者不同——这张表是全量清单加维护状态，那份是「我现在要做 X，读哪一份」 | 规范 |
| `manual/env.md` | API key 与 `.env`：Gemini/Exa/Tavily、Windows DPAPI 加密、`keys`/`doctor` | 规范 |
| `manual/repo-install.md` | 源码安装全步骤：uv 默认 / pip 替代（torch 必须走 cu128 索引的坑） | 规范 |
| `manual/batch.md` | 一次跑多个输入：manifest 行的写法与三个非选项键、**运行期队列面与控制面**（加任务/插队/撤掉）、**`--resume-batch`**（含 7 天、活批、异目录三道拒绝与「显式选项盖过记录值」）、产物唯一性、失败/中断/重跑、事件流与逐项日志 | 规范 |
| `manual/resources.md` | 数据落在哪、`relocate` 搬盘与共用、从 0.4.x 桌面端迁移、`config.toml` 放哪与有哪些节、下载镜像与手动覆盖、卸载档位、缓存为何单独删没用、运行环境的核对与重建、worktree 模式；显卡支持范围、平台与支持等级（Linux / AMD / macOS 实验）、`--gpu-tier` 五档与 CPU 回退 | 规范 |
| `manual/outputs.md` | 运行产物：`out/<名字>/` 里每个文件是什么、`--stage` 六个值各停在哪、`-annotated.csv` 十列怎么读（含说话人的 `spk` 列）（`conf` 是 LLM 自评，不是 ASR 置信度）、想重跑某一步该删什么 | 规范 |
| `manual/joint-asr.md` | 一体转写用户向：`--asr` 的五个选择怎么选（对照表）、隐私与费用、额度与 key、`<名字>-cloud/` 缓存，本地 MOSS（什么卡、显存、第一次下载、语言、和 MOSS Pro 比），任务记着自己用的是谁、说话人、补漏转与复读（`--asr-repair`）、`--asr-aligner`、安装与现在的限制 | 规范 |
| `manual/chinese.md` | 中文素材用户向：LLM 阶段只纠错不翻译（自动按素材语言判）、两列（忠实转写与润色）各改什么、风格只用写给中文纠错的、`auto` 不校对、统一简体、推荐 `--asr moss-local --language zh`、之前开始的任务照旧翻译 | 规范 |
| `manual/diarization.md` | 说话人分离用户向：`--diarize` 的四档怎么开、首次自动下载、默认模型（可商用）的各组件来源与署名、GPU 与 CPU 的速度、多出来的产物、逐行置信标、LLM 说话人裁判（仍在开发中）、相关选项、研究用的两个后端 | 规范 |
| `manual/tuning.md` | 调参：字幕长短（`--split-length-scale` / `[segmentation]` / 风格的字幕长度档位；识别分句与 LLM 合并）、字幕停留时间（`[postprocess]`）、识别侧（含说话人分离）与 LLM 侧各一张旋钮表（默认值 + 改完要删什么）、`--no-download-video`、运行环境（`[runtime]`）、一个耗时量级参考 | 规范 |
| `manual/troubleshooting.md` | 故障排查的总目录：按症状（回退 CPU、显存告警、空字幕、429、设置没生效…）指到那一页；末尾是提 issue 该带什么 | 规范 |
| `manual/ct2-wheel.md` | patched CTranslate2 怎么装、怎么自检、装错了什么症状 | 规范 |
| `manual/models.md` | 模型选择：ASR 三个可选 Whisper（默认 turbo / large-v3 无优势、日语微调互有胜负但慢 2.4 倍）、不可换的分离器与第二模型、说话人分离用的模型、各 LLM 后端的使用印象 | 规范 |
| `manual/model-routing.md` | 一次调用怎么定下来：会话→任务组→预设格子→候选过滤、`[api] proxy`、媒体/difficulty/思考旋钮、预设带开关默认值、catalog、出厂预设（含快速粗纠 `agy-fast` / `fast`）、接自己的模型、启动告警、改什么会作废 checkpoint | 规范 |
| `manual/share-server.md` | 自建共享库：信箱 vs 整台服务端怎么选（对投稿者无差别）、Cloudflare 那条从 login 到 deploy 的步骤、签名密钥（`share keygen`，发布快照必须签）、三步收发循环与「审核不要自动化」、搬家备份只认 `share-server.sqlite` 一个文件、别人怎么接。⚠ 决定与操作在这份，端点契约与配额常量在 `deploy/` 下那两份 | 规范 |
| `manual/knowledge.md` | 知识库用户向：它记什么、在哪、改内容的四条路（含「改 rendered/ 要等下一次纠错运行启动时才收录」）、AI 改了你写过的行时怎么看与怎么认、要动用 AI 的命令默认停在第一步（三步：看问题 / `--execute` 看建议 / `--apply` 才写库）、共享与冲突、风格条目、故障对照表 | 规范 |
| `manual/agent-tasks.md` | 让 agent 替你做的几件事：审 run、把资料收进知识库、打反馈包；说什么话它就去读哪一份、为什么不需要安装配置 | 规范 |
| `manual/agent.md` | 用本机 Codex / Claude Code / agy / DeepSeek Harness / CodeBuddy 订阅代替 Gemini 额度（启用与安装五家共用一节，agy / dsh / CodeBuddy 的注意事项各有一节）；档位选择、失败行为、`agent-clean`、搬盘/卸载 | 规范 |

### 开发与维护总入口

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `../README_DEV.md` | 开发原则、资源约束、canonical artifact tree、reuse/resume 规则、agent checklist | 规范 |
| `architecture.md` | 架构地图：数据流与入口，每个模块的 owner 文档与「动它之前必须知道的」（✱ 标着有测试钉住的约束） | 规范 |
| `test-runs.md` | 跑哪些测试：什么时候跑什么、改动对应哪组测试、命令、CI 跑不到只能本机跑的两条 | 规范 |
| `testing.md` | 测试的组织方式与守卫：域标记的划分、结构守卫、测试替身、设备解析与子进程弹窗的验证手法 | 规范 |
| `data-index.md` | 数据与基线索引的**规则那一半**：三类划分、跟踪标注的口径与警告、只存在于文档的实测基线。逐条清单（含 BV 号与本机路径）在本地 `data/index.md`，不进 git | 规范 |
| `bench-discipline.md` | **动性能前先读的那一份**（短）：一个数字算数的六个条件、`tools/bench/discipline.py` 的强制项、以及「本机不是干净测量环境」这条读数前提。2026-09-03 从 `bench-baselines.md` [第一节 删掉这条规则](bench-baselines.md#一删掉这条规则别的信号能覆盖吗--不能)搬出 | 规范 |
| `decisions.md` | speech、评测尺子与跨层工程上已经定了的事，一条一节：定了什么、理由、重开条件（LLM 层的在 `llm_design_notes.md`）；两份 follow-up 的「明确不做」指到这里。专题文档自己的取舍一节（`asr-align.md`、`joint-asr.md`、`segmentation-gold.md` 等）留在原处 | 台账 |
| `bench-baselines.md` | 2026-08-27 换机后重取的本机基线与实验记录（第二到第二十八节，编号从「二」起，一律不重编号）。哪些结论已经关闭也记在里面 | 实验记录 |

### 语音链路（VAD / ASR / 分句 / 分离器）

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `vad-energy.md` | energy VAD 本体：处理流程、流式=内存契约、峰值 clamp 取舍、`WaveformObserver` 钩子、Python API | 规范 |
| `vad-asr.md` | VAD→ASR 组合阶段：CLI（含 `--vad-silero-assist`）、数据流、aligned JSON 字段契约、失败行为 | 规范 |
| `asr-align.md` | interval→aligned ASR：解码配置、词级映射、异常救援阶梯、覆盖率救援、语言票翻转重解与全局语言审计、输出字段语义、词首修正、组批预取（默认关） | 规范 |
| `asr-backends.md` | ASR 后端契约：解码选项只在 `asr_backend.DECODE_OPTIONS` 声明、后端拒绝未知选项、必须支持与挂能力标记两种身份、不可用时加载前报错；新后端（如 MLX）的入场条件 | 规范 |
| `joint-asr.md` | 一体转写（`--asr`，`speech/joint/`）的开发者文档：云端三家模型实测的行为、生产补了什么、设计取舍与理由（跳过本地阶段、按话轮分句、MOSS Pro 的 Silero 定时与续调补调、声纹重连、本地记账、说话人只用模型自己的标签、重叠规则只管有键的行）；本地 MOSS（为什么接与读数、开源 MOSS 的已知行为、显卡与进程状态、救援各条与阈值来历、解码器与显存、开发开关与等价重放）；改的时候容易踩的 | 规范 |
| `asr-stabilize.md` | aligned→stable：profiles/metrics/tags/CLI、时间轴格式契约、resume 规则（profile 3 pre-merge 移除的缘由记录在内） | 规范 |
| `segmentation-split.md` | 分句规范：全局 DP 打分、gap 调整、字段继承与幂等 | 规范 |
| `segmentation-gold.md` | 分割点金标准：必切/禁切/宜切判据、时间轴锚定、打分口径。审计分割质量前先读 | 规范 |
| `gpu-profiles.md` | 五档映射（`cpu`/`entry`/`standard`/`standard_large_vram`/`high`，`cpu` 是策略档）、厂商（AMD 走 `torch.cuda` 的三处差异、Apple 暂不用）、`auto` 定档与上限、最大窗口实测与并发依据 | 规范 · 含实测 |
| `separator-optimization.md` | BS-Roformer 推理效率探索（E0–E11）：已采纳 AMP+编译路径、交付两模式、已否决方案 | 实验记录 |
| `diarization-findings.md` | **说话人声学能力边界的真相源**，按主题组织：各后端（默认 Nemotron-3 + vbx，研究用的 DiariZen / Sortformer）的质量与人工代价、轨不等于人的四种失败、可复现性与数值敏感性、重叠、>4 人、「是不是同一个人」、分离与编码、LLM 层、尺子与测量纪律、撤回的读数。过程记录已归档（本地），第 13 节列了去哪找 | 规范 · 含实测 |
| `batch-scheduler.md` | 三 bin（download×2 / asr×1 / llm×2）的并发语义、llm bin 准入、运行期队列面与控制面、`--resume-batch` 与批次身份、产物唯一性、失败与中断语义。`scheduler.py` / `batch_state.py` 的 owner；2026-09-03 从 `wt-parallelism.md` 整节搬出 | 规范 |
| `speech-followups.md` | speech 侧未完成工作的索引（对称于 `llm_followups.md`）：四项没开工 + 两项交付了但默认关着（只出 `-raw.srt` 时的声学说话人分离、A1 组批，翻默认欠一份证据）、明确不做（一条一行，指到 `decisions.md`）、散在别处的未做项。⚠ **公开侧的状态以本文为准**（状态只写在这里、只改这里）；逐项证据、判据与展开在维护者的本地计划里（那里的状态格是注明日期的历史，不同步），不随公开快照发布 | 台账 |
| `wt-parallelism.md` | 单文件 WT 分片（已移除）：回溯点、仍成立的结论（asr 恒 1 的并发实测依据、stdio 背压根因、intra-op 线程预算）。**现行 batch 契约已搬去 `batch-scheduler.md`** | 历史 |
| `wt-refine-handoff.md` | CT2 WT refine 研究交接入口（已合入 dev）：一页结论、当前实际行为、研究过程、P0–P3 待办（算法契约在 `wt-refine-port.md`） | 台账 · 含实验记录 |
| `wt-refine-port.md` | WT refine→FW/CT2 详细算法契约、multi-audio batch 设计与档位表、质量实测 | 规范 |
| `wt-refine-validation.md` | 13-group 信号与局部隔离验证结果 | 实验记录 |
| `gemini35-transcribe.md` | 外部转录模型 `gemini-3.5-transcribe` 的调用规则：两条通道各自的配额与撞限形态、互斥开关、按词数切窗、Live 的三条必须、拿它做仲裁的纪律。**不是生产管线的一部分**，是调查工具 | 规范 |

### LLM harness

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `llm_harness_behavior.md` | LLM 运行时 canonical 总入口（文首有拆分导航）：开关轴、快速模式、窗口拆分与调用形态、Prompt 信息与注入上限、说话人裁判、SRT 后处理、重试与拼接、知识库更新、任务级并行与 agent 槽位预算 | 规范 |
| `llm_harness_routing.md` | 路由 dev 侧：模型事实/池/路由链、thinking 档位换算、模型配置与限流。使用者向见 `manual/model-routing.md` | 规范 |
| `llm_harness_research.md` | 本地检索代理（Exa→Gemma4 grounded→Tavily 降级链）、准备任务（一条对话，按轴收放请求通道；取代了旧的 r1/r2 两轮）、超长素材分块调查 | 规范 |
| `llm_local_agent.md` | Agent 执行后端唯一入口：五家 one-shot transport 契约、durable task 协议、tier 冻结、会话档位接线（[§12.1 会话形态](llm_local_agent.md#121-会话形态四档已决定什么还差什么接在哪)） | 规范 |
| `llm_agent_tool_protocol.md` | agent 工具化协议现行规格：工具表与 request id、必读块台账、审计包、五家 driver 接线 | 规范 |
| `llm_local_agent_experiments.md` | 长驻会话准则与实测：会话复用 A/B、缓存写入门槛成因与复测、Claude Code 反向信号 | 实验记录 |
| `llm_local_agent_runtime.md` | 执行环境卫生：episode 落点、capsule 是一次性 episode、滚动上限 20 与清理 | 规范 |
| `llm_local_agent_agy.md` | agy 专属：catalog 行、音频必须容器化、视频分辨率不可调、`view_file` 准入硬门 | 规范 |
| `llm_prompts.md` | prompt 模板/碎片、`prompt_compose` 组装表、PROMPT_VERSION 语义 | 规范 |
| `llm_design_notes.md` | LLM 层的架构意图与决策记录：路由 durable 决策与理由、预算公式推导、知识更新、行文法 v3、三个 pass 等各项「…的决策记录」，`llm_followups.md`「明确不做」各条的理由与重开条件，已退役的 capability tier，口语颗粒的取舍，实测沉淀的 harness 原则（speech、评测与跨层的在 `decisions.md`） | 规范 |
| `llm_followups.md` | LLM 未完成实验/设计唯一入口：按 harness / Agent / 知识库 / 前端分节，每条带触发条件或重启前提；已完成项不留正文（2026-09-02 整理前的原文在维护者的本地归档里） | 台账 |
| `knowledge.md` | 知识库一切：结构、`--knowledge` 三态、feedback v3、统一更新、风格条目、条目的长度与子条目；两个翻译台账已退役（留着退役记录） | 规范 |
| `provider-adapters.md` | 自定义 provider（OpenAI-compat/Anthropic 纯文本）：adapter 契约，与一份 2026-08-11 的供应商行为调研 | 规范 · 含实验记录 |
| `prompt-iterate.md` | 纠错 prompt 迭代方法论：定位与唯一机制、四变体、session_replay 协议、失效模式 | 规范 |
| `session_replay.md` | replay 协议：3 个 session（`correction` / `prepare-task` / `correction-request`）的 fixture/validation 契约、补中间态落盘 | 规范 |
| `merge-calibration.md` | 合并软门槛标定：默认不并、gap/字数先验、flash-lite thinking=0 | 实验记录 |

### 分发、前端与跨前端

| 文档 | 主题 | 状态 |
| --- | --- | --- |
| `reporting.md` | `finesub.reporting` owner 文档：八个事件契约、级别与 renderer、各阶段上报口径、LLM 段事件词表 | 规范 |
| `download-routes.md` | `finesub_bootstrap` 下载族 owner 文档：地区解析、镜像降级、cn lock 约束、marker 四态、续传 | 规范 |
| `ct2-distribution.md` | patched CT2 打包与分发（面向维护者）：自包含 wheel、为何发 Release、CI 多平台构建与发布（macOS CPU / Linux CUDA / Windows CUDA / ROCm）、资产不覆盖、CUDA 12 边界 | 规范 |
| `cross-frontend-lease.md` | 跨前端租约（已完成）：user-data 闸门、独立租约、sidecar、崩溃判活；[§4 已定的问题与边界](cross-frontend-lease.md#4-已定的问题与边界) 是三条已定的不做与一条并发说明 | 规范 |

### 私有笔记：`plans/`、`archive/`、`reports/`

这三个目录在 `dev` 上被跟踪（worktree 与 clone 都拿得到），但由 `scripts/publish-main.ps1` 的
`$PrivatePaths` 从每次公开快照里剥掉，公开 clone 里没有，**因此不进上面的地图**，公开文档也不点名、
不链接里面的文件（`test_doc_links.py` 既不扫它们的链接、也不要求它们进地图）。

- `plans/` 放进行中的计划与两份私有台账（外部对照的证据、重构押后项），索引在仓库根的 `AGENTS.local.md`。
  计划结案时先蒸馏，再删掉，原文从 git 历史取回；只有原文本身还要回查的（例如 owner 原话锚点仍是裁决依据）才挪进
  `archive/`，并在开头写明为什么留、现在该读哪里。
- `archive/` 是退出日常指导、但原文还有回查价值的旧材料，`reports/` 是调研与实验报告原件。

分界线：根目录放**契约与活台账**。所以 `llm_followups.md` 与 `speech-followups.md`（都是「还没做的是哪些」）
留在根、随公开快照发布。一份私有笔记只要还有公开侧要读的东西，就先按现有体裁送回家：现行行为进规范、
未决进台账、踩坑进规范的 ⚠，仍约束后续修改的取舍理由（为什么这么做、为什么不做 X 及其前提）进 owner 文档。往里迁文档前也一样，先把仍然成立的事实提回被跟踪的
公开文档，同一次修好断链。

## 找东西

两份索引，两个读者。**本文件的地图**是全量清单：每份被跟踪的 `docs/` 文档一行，写主题与状态，给做文档维护的人，
也是 agent 找冷门文档的地方。**仓库根的 `AGENTS.md`** 给 agent：只有按任务找入口的短表，加少量**判断提示**
（唯一入口、状态异常、踩坑）与「要翻代码才找得到」的几处。分界写死：**全量清单与描述只在这里，判断提示只在
`AGENTS.md`**；模块级的「动它之前」在 [`architecture.md`](architecture.md)。

除本文件自身外，被跟踪的 `docs/` 下的每一份 md——**根目录与 `manual/`**——都应当出现在上面的地图里（路径按相对
`docs/` 写，例如 `manual/env.md`），`test_doc_links.py` 盯着这一条。
