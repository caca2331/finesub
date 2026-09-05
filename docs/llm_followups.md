# LLM Harness 后续实验与未实施工作

本文只记录**仍需验证或尚未实施**的工作，按 harness / Agent / 知识库 / 前端分节，每条一行：
状态 + 一句话现状 + 去处。要展开的放在后面的同名小节。现行 runtime 行为以
[`llm_harness_behavior.md`](llm_harness_behavior.md) 为准，prompt 结构以
[`llm_prompts.md`](llm_prompts.md) 为准，Agent 接线以 [`llm_local_agent.md`](llm_local_agent.md)
§12.1 为准。

**已完成的项目不在这里留正文**：现行行为归 owner 文档，实施过程与当时的论证在本地
`docs/archive/llm_followups-2026-09-02-before-tidy.md`（2026-09-02 整理前的整份原文）。
本文以后新增一条要写清**触发条件**（什么时候该做）或**重启前提**（什么条件下再看），
没有的就写「记一笔」。

## harness：窗口、重试、预算、路由

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **「模型有没有真的收到窗口正文」缺一个独立信号** | 待触发（2026-09-03 记） | `MAX_DISCARD_RATIO` 拦的是「没读正文」这种失效，但它**只管未拆分的整窗**：叶子上丢弃比例没有分辨力（按同一条 2.3 倍标定，门槛会落到 100% 以上，退回既有的全-discard 判据；实测见 [`bench-baselines.md`](bench-baselines.md) 二十五）。所以叶子上那类失效今天拦不住。补它要的是**另一种信号**——回执里能证明模型读到了正文的东西（例如要求回引窗口首末源的片段，或让 agent 前端把「正文已投递」做成可校验的事实），不是另一个数字。⚠ **今天没有触发证据**：归档里 4 个真实叶子丢弃全是 0，那次 canary 也发生在整窗第一次尝试上。真出现一例叶子上的空读，再按这条做 |
| **P6：开关组合与输出系数标定** | 待实验 | 未标定组合可运行但打印 warning；efficiency 与 video 沿用旧系数。协议见下方同名小节 |
| **修复轮的效果数据** | 未量 | 契约在 [`llm_harness_behavior.md`](llm_harness_behavior.md)「重试与拼接」。⚠ **API 路是「通而未用」**（2026-09-01 核对）：REST 端点拿到的正是消息形态的修复轮，实跑也验过一次；但扫 55 份 run 的 `task-artifacts.jsonl`，修复轮只发生过 3 次、全在 `local_agent`。「通过率是否真的提高」因此仍未知——要立论先攒够 API 路的失败样本，别拿一次玩具调用当证据 |
| **两档重试在桌面 `TaskRequest` 上未暴露** | 待触发 | 桌面今天连第一档都不携带（全走默认）。若将来把任一档加进 `TaskRequest`，**两档必须同批**，不能只加一个 |
| **fast 会话内部预算未按组收缩** | 待触发 | 包络放不下融合窗时闸门按名拒绝；注入上限仍是绝对值。见下方同名小节 |
| **自定义 endpoint 的媒体支持** | 按计划推迟 | 文本方言声明媒体能力即报错；纯文本模型靠 `--correction-media text` 用在音视频素材上。见下方同名小节 |
| **`<reasoning>` 措辞按注入实况变化** | prompt 迭代项 | 2026-08-12 已中性化。可能有价值的形态是按**实际注入了什么**变（`retrieval=none` 要求写不确定处与候选读法，有 context pack 则要求交叉核对），与 `CORRECTION_SLOTS` 的注入实况原则一致。**必须走 [`prompt-iterate.md`](prompt-iterate.md) + `tools/session_replay` 的协议做**，不能凭感觉改 |
| **none/native 的逐窗选词条轮** | 暂缓；**可能已无对象** | 当前不保留逐窗轮，会话级 r1 一次选定。数据面改成「索引必读 + 自主 query」后「挑」这个动作消失，本条随之作废（[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §7）。重启前提与协议见下方同名小节 |
| **预设携带 difficulty/retrieval/knowledge/continuity 的默认值** | 记一笔（owner 2026-08-14：闲了再做） | 预设当前只绑「格子 → 模型组」。让预设带开关默认值合理（agy 档顺便把 retrieval 调成 local），但要先定它与 CLI 显式值的优先级、以及它算不算 routing digest 的一部分（会不会作废 checkpoint） |
| **catalog 只有两列窗口，没有「上下文」那一档** | **已实施（2026-09-03）** | 全文移到 [`model-window-limits-plan.md`](plans/model-window-limits-plan.md)：catalog 加 `context_window`，删掉 `DEFAULT_LIMITS` 那两个当上限用的 `min(...)`（本文代称 `HARNESS_INPUT_CAP` / `HARNESS_OUTPUT_CAP`，**代码里没有这两个名字**）与 `context_limit` / `safety_margin` 两个字段。⚠ 它同时带一份 catalog 可疑值审计（§7），那批是独立决定 |
| **桌面端 `efficiency` + `knowledge=update` 的硬冲突** | 待触发 | `TaskRequest.knowledge` 写死 `"update"` 是**有意的产品决定**（不交给 `resolve_knowledge_switch`——那会静默降成 collect）；而 LLM 层在 efficiency 下拒绝 update。今天到不了：前端不暴露 difficulty。等 UI 把它做成可选项时再加护栏 |

## Agent：会话、长驻 worker、工具协议

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **会话模式四档的对照 + 复用经济账重测** | 未做 | 四档 `api`/`per-window`（默认）/`resume`（实验开关）/`pseudo-conversational` 全部接线（[`llm_local_agent.md`](llm_local_agent.md) §12.1）。已有的只有：agy **小任务**复用净亏 46%（[`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) §3.1，成因 §3.2：缓存写入门槛约 1.6 万 token、跨 session 不继承）、生产尺寸窗能过门槛但不保证命中、Claude Code n=1 的反向信号（§3.4）。**默认值在有数据前不动**；重测协议 §3.5（三家各两臂 + 噪声基线、n≥5、判据先写死；开关 `agent_session_mode=resume`）。Codex 未测，注意它 cache write 恒 0 是显示缺陷、cache read 才准 |
| **agy 的 `--add-dir` 工作区读授权** | 等下游回复（2026-09-04 记） | 下游 `Ricori/nonoka-sub-x` 打了这条 patch，理由是 agy 1.1.20 起工作区外的读一律弹权限、而 headless 没有弹窗通道（失败形态见 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) §6.2）。⚠ **对我们不成立**：2026-09-03 agy **1.1.25** 真机实测，project 建在 `<domain>/.finesub-native`、读 `<domain>/call-1/block.txt`（父目录的兄弟子树）**成功且无询问**，而 `C:/Windows/win.ini` 被拒——`view_file` 的关比 project 目录宽，我们两条路径都在关内。**卡在一句话上**：下游作者说「之前没配置文件的话就读不了」，而「配置文件」指 project 记录（`~/.gemini/config/projects/<id>.json`）还是 `~/.gemini` 全局配置，两种读法的修法完全不同——前者我们按构造必然有一份（`_grant_permissions` 记录缺失即 fail closed），后者说明是开发机的历史授权兜住了我们。问清之前不动。逐条判定与要转达的建议在 [`nonoka-downstream-findings-plan.md`](plans/nonoka-downstream-findings-plan.md)「单列 0003」 |
| **`resume` 档在工具会话下未定义** | 未做 | 今天恒走 capsule 窄路。见下方同名小节 |
| **长会话 assignment 的簿记随 task 数二次增长** | 未做；今天够不着 | 见下方同名小节（含触发条件） |
| **Agent 工具化协议 D 步** | 未做 | 删 capsule 输入路径、`accepts_repair_context`、`repair_in_messages`、会话 handle 缓存——等全部 driver 在生产稳定后。见 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §7 |
| **数据面迁移（索引必读 + 自主 query）** | 明确暂缓，独立项目 | 量级与 B 步相当，禁止夹进控制面补丁；顺序与闸门见 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §7 |
| **动态 driver（进程外版本化协议）** | 剩余架构项 | 长驻 worker runtime 其余部分已完整；见 [`llm_local_agent.md`](llm_local_agent.md) §12 |
| **conversational 按需并发** | 暂缓（回流 2026-08-30） | 现强制 serial（`conversational-forced-serial` 告警）：并发度取决于谁 join 了、harness 分配不了，一人 join 时 fan-out 纯亏。放开时：默认仍 serial；显式 parallel 才允许，run 起步查队列注册 worker 数、只有 1 个就 warning；**不按注册数自动决定 continuity**（会把语义决定变成非确定） |
| **conversational 真机覆盖面只有一格** | 未做 | 2026-08-24 那次只验了**单 task、单 worker、无修复轮**；多 task 连续领取、修复轮往返、两个 worker 同时在场都没跑过。⚠ [`llm_local_agent.md`](llm_local_agent.md) §13 的措辞按**已跑到的部分**更新，别一次写满。背景与那次的读数在 [`conversational-live-test-plan.md`](plans/conversational-live-test-plan.md) |
| **`local_agent_timeout_seconds` 在 replay 路径上没进队列** | 未做，**是个待查 bug** | 临时配置写 3600、单独 `load_execution_settings()` 也读得到 3600，但跑起来的队列拿到默认 900（由租约 TTL `max(30min, wait)` 反推出来的）。**触发条件**：动 replay 或队列的 timeout 传递时先查这条；它不改变 2026-08-24 那次的结论，所以当时没拦。出处同上 |
| **`local_agent_timeout_seconds` 的默认值要重估** | 未做，**前提已满足** | 那次 48 分钟大半是截断重试，判不了 900 秒够不够；而两个时钟分开之后（该计划第 4 步，已做），这个值只再管「等人来领」那一段，量级判断要**重新起头**——上限 3600 一并重估。**重启前提**：计划第 1、2、4 步落地后再看一次真机耗时，三步都已落地，缺的只是一次真机跑 |
| **agy 原生搜索的 URL provenance** | 缺口，可能补不上 | `--retrieval native` 已通（[`llm_local_agent_agy.md`](llm_local_agent_agy.md) §5），但 agy 不报来源 URL：事件流零 http、只有查询词，`search_events[].urls` 恒空；Codex/Claude 有逐 call 的完整 provenance。补齐要么看 agy 是否在 `tool_info` 里带结果，要么改走 `read_url_content` 逐页取（慢且改变检索语义）。做 native/local 检索质量对照时必须把这条不对称算进去 |
| **agy 的质量/成本持续标定** | 进行中 | A1–A5 接入已完成（[`llm_local_agent_agy.md`](llm_local_agent_agy.md)），剩的是标定不是接入 |
| **Agent cache ping** | 明确暂缓 | 先测同类型连续 task 与 compact/重连后的 cached input/墙钟；task heartbeat 不承担 cache 保活。[`llm_local_agent.md`](llm_local_agent.md) §13 |
| **Agent 单任务模式** | 明确暂缓 | 不做产品入口，但作为 `session_scope=task` 全重放基线永久保留。[`llm_local_agent.md`](llm_local_agent.md) §13 |
| **Agent 直接编辑知识库** | 明确暂缓 | 当前只走 proposal/update；预留隔离 worktree 的 `staged-edit` 策略。[`llm_local_agent.md`](llm_local_agent.md) §13 |
| **完整知识只读 + 联网工具的外泄通道** | P2，记录不做 | 见下方同名小节（定级理由与重评条件） |

## 知识库

其余知识库遗留项（超限条目压缩、语义级输出校验、子词条拆分自动化、PROHIBITED_CONTENT 误杀）
在 [`knowledge.md`](knowledge.md) 末节，这里只放跨模块的一条。

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| ~~**style 条目上限 N/L 的标定**~~ | **结案不做**（owner 2026-09-02） | 值已定（20 行 × 200 字 + 6000 token 兜底），理由与「为什么不值得单独跑那一轮」在 [`translation-style-plan.md`](plans/translation-style-plan.md) §4/§6。⚠ 保留其中一条纪律：真要接一个会改变译文的开关时，门槛仍须**预先**写死 |
| **知识库遥测卷积（`events` 表）** | 有触发条件，未做 | 见下方同名小节 |

## 前端与分发

一条都不剩：打包 CLI 的知识库入口已于 2026-09-03 接上（`finesub knowledge` /
`knowledge-update` / `knowledge-share` 三条 `runtime_module` 转发，用户向说明在
[`manual/knowledge.md`](manual/knowledge.md)「下面的命令怎么敲」）。

## 明确不做（owner 裁定，别再当待办）

- **给 free Gemini 的 `max_input_tokens` 正名**（owner 2026-09-03）。free 填 194000 而同一个
  `api_model_id` 的 paid 行填 1048576，因为前者是按 `tpm=250000` 反推的实用上限而不是上下文窗口
  （194000 已占 TPM 的 78%）。owner 裁定**不为它改代码**——「这是个对真实行为的高效模拟」。
  相关事实备查：限流器**不拦**超额的单次请求（`rate_limit.py:565` 的 `and active_events`，窗口
  为空就放行），拦的是供应商 429。上下文在
  [`model-window-limits-plan.md`](plans/model-window-limits-plan.md) §4.1。
- **P7d：parallel 并发上限的标定实验**（owner 2026-08-30）。并发上限是用户偏好旋钮：各家 agent
  对并行流量本身宽容，没有可实验出来的「安全值」。选值指引与两条真实代价在 `manual/agent.md`
  「同时运行的 agent 数量」；出厂默认 windows 1 / tasks 2。**不得「每个 worker 绑一把 key」提速**。
- **conversational 的 worker 能力申报与逐 task 对账**（owner 2026-08-22）：那条路上跑的是用户自己
  正在用的 agent，能力与质量由用户负责，注册只申报 `kind`。
- **修复轮的廉价折中「只给错误、不给上一轮输出」**：便宜且可能对付机械违规，但那已是 prompt
  强化而非修复轮，要做先 A/B。**prompt 侧强化不作为主解**（owner 2026-08-15）：对「模型知道
  规则却漏执行」的强化已到尽头，真解是让它看见错误。
- **把 `TaskRequest.knowledge` 改成 `None` 交给 `resolve_knowledge_switch`**（reviewer 建议，
  2026-08-17 核实后拒绝）：那会把桌面端从「写知识库」静默降成「只采集」，与产品决定相反。
- **接 Gemini 的 agentic video**（2026-09-02 实测后不接）：它只存在于 `/v1beta/interactions`，
  `generateContent` 四种字段摆法全 400，所以是**接一条新 transport**（要接也在 `llm_runtime`，
  不在只做纯文本的 `provider_transports`）。默认流程收益为零——`processing` 在 audio 输入上被拒，
  而默认 `correction_media` 是 audio；对生产口径基线（static `fps=0.25`+`low`）也只省约 59%，
  不是宣传的 88%。两个硬阻塞与省不省钱无关：interactions **拒绝 `safety_settings`**（本仓库的
  `BLOCK_NONE` 放宽是有事故记录的），错误体没有 `quotaId`/`promptFeedback`（日封禁 strike 与
  `is_prompt_blocked` 都会瞎）。重开此事前先跑那份报告 §6.5 的预注册实验（真实纠错 prompt、
  n≥5、token 省 ≥40% 且覆盖率与时间戳不劣于 static）。证据在本地 `docs/report/2026-09-02-agentic-video-interactions.md`。

---

## P6：开关组合与输出系数标定

目标是校准 `profiles.py` 的输出系数和实际质量，不是再次调整开关结构。重点包括：

- efficiency：现在强制 basicB、thinking low，旧 text-low 测量条件已经变化；
- `media=video`：媒体成本与窗口几何仍沿用旧标定；
- `audio+native` 等可表达但未实测的组合；native 会在当前绑定组内过滤支持原生搜索的
  target（出厂 quality 由付费 3.7 Flash 接地），仍需单独验证质量与媒体成本；
- 其他 `capabilities.profile_warnings()` 标记的未标定向量。

实验纪律：

1. 每个对照臂至少 `n≥5`，并做同配置复跑作为噪声基线。
2. 固定素材、知识库快照、prompt version、endpoint/tier、output scale 与搜索轮数。
3. 同时报告产出质量、窗口数、输入/输出/thinking token、重试和实际墙钟；不要只看
   `token_distribution_report.output_budget` 的预测/实测比。
4. efficiency 与 video 完成重测后，才从 `RECALIBRATION_PENDING` 移除 warning；新组合只有
   质量和系数都稳定后才加入 `CALIBRATED_VECTORS`。

**已有一份观测，但不构成标定**（2026-09-04）：56 个纠错窗、14 天、按模型分桶量了
「回复正文 token ÷ 窗口 CSV token」，中位 **2.03**、p90 2.54，而预算的 c 是 4.5–5.0
——约 40% 的预算被用掉，且倍率随窗口增大而下降，全程 `output_limited = 0`。方法与
逐模型数字在本地 `docs/report/2026-09-04-workbuddy-driver.md` §3（**不随仓库发布**）。
⚠ 它只回答了「预算给多了多少」，没有回答「调小之后质量如何」——那正是上面这四条
纪律要的东西，所以**没有据此改系数**。

## fast 会话内部预算未按组收缩

`fast_session` 内部各上限是写死的绝对值：`FAST_ROUND2_INPUT_RESERVE_TOKENS` = 56k、词条块
28k、证据包 20k、notes 2k。组包络已接到 fast 判定闸门、纠错主循环、知识更新分块三处，但这些
注入上限本身不随包络收缩。2026-08-12 只做了让失败可读：绑定组的输入包络 ≤ 预留时，闸门直接说
「该组放不下融合窗，走常规多窗流程」（`--fast on` 仍报错而非静默降级）。

真要支持小上下文模型跑 fast，必须缩的是**注入上限本身**——那是质量旋钮，是标定活不是接线活；
按比例缩 56k 这个和数只会让闸门放行、第 2 轮照样溢出。

- **触发条件**：真有人要在小上下文模型上跑 fast。
- **一起核的东西**：`FAST_ROUND2_INPUT_RESERVE_TOKENS` 与它对应的那几个分项上限必须同时重算。

## 自定义 endpoint 的媒体支持

第一刀只做文本。媒体接线很深：Gemini Files API + `file_ref` 复用（预取线程、`file_ref_seed`、
查询轮与纠错轮共用一份剪辑），而 OpenAI/Anthropic 走 inline base64，限额与音频支持差异大。
catalog 里文本方言（`openai_compat`/`anthropic`）的行**声明媒体能力即报错**——两个文本传输展平
messages 时会丢掉附件，放行等于静默降质。

## none/native 是否需要逐窗选词条轮

当前裁决是**不保留**：有索引时由会话级 r1 一次挑词条；没有逐窗请求能力的路径强制全量透传。
给纠错终稿轮直接加索引已排除（请求到的词条本窗来不及用、挤压纠错输入预算、扩大输出契约与
replay 面）。

**重启前提**（须同时满足）：知识库词条数显著超过 r1 额度 12，使选择真正需要取舍；且至少有一个
多窗 local 运行的逐窗查询轮产生过非空 `requested_entries`。此前数据不满足（知识库 6 条、逐窗请求
全空，r1 召回在实验前就是 100%）。

**协议**：两份有精修的素材（专名均匀分布一份、关键专名集中在后段一份）；比较 `r1@lightweight`
与 `r1@general_capable`，触及上限的样本作废；令 `R` 为会话级 r1 注入集合、`U` 为逐窗实际
`injected_entries` 的并集，报告 `|R∩U|/|U|`、`R\U`、`U\R` 与 `hit_cap`；人工判 `U\R` 是否真影响
成品——两臂都漏关键条目是结构问题才考虑逐窗轮，只有 lightweight 漏就升级 r1 模型，漏项都无
作用则维持现状。`research-context.json.entry_selection` 可判样本资格。

## 未做：`resume` 档在工具会话下仍未定义

`resume` 今天恒走 capsule 窄路——三家都报 MCP 能力，若按「有 MCP 就工具会话」它会永远退化成
`per-window`，A/B 路径就没了。要把它挪到工具会话，先答两问：(a) 三家的 resume flag 与
per-invocation MCP 配置能否共存，没实测过；(b) 续上的会话里模型已有上窗的必读块，而台账每次
lease 清零，「台账说没读、模型说读过」要么多打几次 `read_context`（浪费但无害），要么需要一个
跨窗台账概念。

连带的一条：`resume` 的 handle 缓存仍挂在 `RoleClient` 上，没并入 run 作用域的会话注册表
（owner 决定 4 要求共用）。两件事一起做才划算。

## 未做：长会话 assignment 的簿记开销随 task 数二次增长

`AgentTaskRuntime` 每次操作都整份重写 durable state，而一条 run 的 task 行只增不减（终态的行
也留着——`_assignment_complete` 与 accept 重放都要读）。pseudo-conversational 把「一条 run 一个
assignment」变成常态之后，这条按 O(n²) 记账。

**实测（2026-08-22，本机）**：终态行约 2.7 KB，状态文件线性涨到 600 task 时 1.5 MB；耗时按每
100 task 累计 7.5 → 22.5 → 45 → 76 → 116 → 164 s。**换算成真实规模今天够不着**：一个 task
= 一次 `complete()`，现实区间 25 task ≈ 1.1 s、100 ≈ 7.1 s、200 ≈ 22.5 s；600 task 要几十到上百
小时的单个文件（batch 每个文件各自开 assignment，不累积）。不影响正确性——request 表撑满那条
硬失败已修掉。

要动就直接上「终态 task 行移出热状态」（落 `tasks/<id>/record.json`、热状态只留小桩）；只瘦身
终态行是把同一条曲线右移；真正去掉平方项要改成追加日志 + 周期快照，那会动到 exactly-once 所
依赖的「一次整文件原子写 + accept 的 WAL」，需要单独一轮设计与复审。

- **触发条件**：一条 run 的 task 数常态过三百，或簿记时间在 run 里可见。

## 完整知识只读 + 联网工具的外泄通道（P2，记录不做）

完整知识只读与 local/native 联网同时开启时，字幕或网页里的注入可以诱导 Agent 用**已获准**的
`knowledge.read()` 取内容，再塞进搜索 query 或 fetch URL 外发。不需要提权，所以「payload 不能
扩大权限」挡不住；provenance 校验和输出 validator 管的是流入，这条通道是流出。

**定级 P2、暂不做工程**：能力集不超过用户已接受的本地 coding agent 风险（后者能读 `.env`/密钥/
整个源码树），而知识库的资产是主播专名与档案——泄露是关于第三方的隐私问题，不是对用户自身的
入侵。两处真实差异记录在案：headless worker 连续 drain 队列时**无人在环**；输入（任意视频字幕 +
抓取的网页）按构造就不可信。

已做的两项：prompt 层的内容边界（`fragment_knowledge_structure_v1.md` 的「一律不收集」三条）；
搜索 loop 的 extract URL 收口成「只能选已展示过的 URL」（`retrieval=local` 的 ledger 把它做成
硬门）。**两条通道的差别是有没有接收端**：fetch 的目的地由模型指定、注入方看得见访问日志；
search query 只发往 Exa/Gemma4/Tavily，注入方没有读取途径——所以 query 不设限也不该设长度上限，
收口只针对 fetch。收口后残留的是「选哪一条」的 log₂(N) bit 级通道。

- **重评条件**：headless 多 Agent 真的上线时重看「无人在环」这一条；可选的服务端约束是 task 级
  知识读取 scope，或禁止「完整知识只读」与「任意网络工具」同时出现。

## 知识库遥测卷积（`events` 表）

**版本历史不是增长源，事件表才是。** 真库实测（2026-09-02，rev 53，5 天 30 个生产任务，文件
2.5 MB）：`events` 4581 行 / 848 KB（59%）、`evidence` 1274 行 / 342 KB（24%），四张
`*_versions` 合计 139 KB（10%），其中已关闭的旧版本行约 40 KB（3%）。所以
[`knowledge-node-plan.md`](plans/knowledge-node-plan.md) §2.1「初版不实现 GC」的决定维持——压版本行
最多省 3%，且它列的前置条件（`snapshot_pins`、revert 可达规则、dry-run 与显式确认）一个都没有。

`events` 每个生产任务 150–500 行、每行 185 字节，`exposed` 占 85%——量是「注入节点数 × 窗口数」，
库越大、任务越长它越多。三个读者都是全表扫：`report.py`、`share/exchange.py` 打包 bundle 时按
节点聚合、以及每次 `INSERT OR IGNORE` 撞的 `dedupe_key` 唯一索引。

**形态**（到触发条件再做，不是现在）：

1. 新表 `event_rollups`，主键 `(node_id, item_id, kind, opportunity, matcher)`，字段是任务数、
   窗口数、首末 task_id、首末 rev。§5.2 的误触发判据、report 的 per-matcher 转化率、bundle 的
   按节点计数，三个消费者要的都只是这些聚合，没有一个读单行 span。
2. 保留窗口：最近 K 个任务（建议 20）留原始行（逐窗视图与排查用），更早的折进 rollup 后删除。
3. **幂等性不能丢**：折掉的任务写进 `rolled_tasks(task_id)`，`signals.py` 的插入对已折任务直接
   跳过，否则重跑一次旧任务就双计。这是唯一要碰写路径的一处。
4. `evidence` 暂不动：它是 claim 级记录，bundle 逐行导出、共享 server 按 `(node, field,
   value_hash)` 查存在性、`verify.py` 也读它。若将来要收，`transcript` 类占 83%，可按同一键收成
   计数加末次日期，键要原样保留。
5. 顺手清 `migration_aux`（plan 写了「切换后可删」，20 KB）。
6. 落点：`maintain.py` 加子命令 `compact`，默认 dry-run 打印将折多少行、聚合前后是否一致，
   `--execute` 才写；写完 `VACUUM`。`report.py` 与 `exchange.py` 改为读 rollup 加原始行之和。
7. 验收：折叠前后 `report` 的聚合数字逐字节相等（声称语义不变的优化，按 README_DEV 必须
   bit-exact）；折两次等于折一次；已折任务重跑不改变任何计数。

- **触发条件**：`report` 或 `push` 单次超过 5 秒，或文件超过 50 MB——按当前速率约 1500 个任务
  之后。
- **不要做的两件事**：版本行 squash（前置条件未备、收益 3%）；为省空间改 `exposed` 的记账粒度
  （那是 §5.1 一致性度量的分母，改了报告就没法与旧数据比）。
