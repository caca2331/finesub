# Agent 工具化协议：agent 调工具取 task、读上下文、提交

**状态（2026-08-22）**：A/B/C 步已实施，三家 CLI 小链真机通过，Claude 生产单窗 canary 通过；
**两个过渡开关已撤**，传输由会话档位派生（[§1 档位与形态](#1-档位与形态)）：出厂默认 `per-window` 在报 MCP 能力的 driver 上
**就是工具会话**。本文是现行规格——
是什么、怎么接、哪里还没做。为什么这样定、七轮复审各改了什么，过程记录在维护者的本地归档里
（不随仓库发布）。
Agent 后端的其余契约仍以 [`llm_local_agent.md`](llm_local_agent.md) 为准。

两条总原则：**不碰用户的全局设置**（agy 的 `settings.json`、Codex 的 `config.toml`、Claude Code 的
用户级 settings 一行不写）；**不为过度保守的安全策略加机制**。

## 1. 档位与形态

配置面只有会话档位 `agent_session_mode`（按 cell 设，见 [`llm_local_agent.md`](llm_local_agent.md)
[§12.1 会话形态](llm_local_agent.md#121-会话形态四档已决定什么还差什么接在哪)）；传输由 **`agent_transports.agent_transport_for(档位, probe, 带不带媒体)`** 派生
（owner 2026-08-22 收敛，A/B/C 步实施期的两个 `[llm]` 过渡键已删）：

| 档 | 传输 | 说明 |
| --- | --- | --- |
| `api` | capsule（`client.py` 窄路，task scope） | 该档的定义：没有上一轮、全重放 |
| `per-window` | probe 报 `supports_mcp_config` → **工具会话**；否则 capsule 窄路 + 每 driver 一次 warning（`agent-transport-capsule`） | 工具会话的原生形态：一次 invocation 内交、被打回、再交。**出厂默认** |
| `resume` | capsule 窄路（跨窗续 handle） | 三家都报 MCP，按「有 MCP 就工具会话」它会永远退化成 `per-window`；工具会话 + resume 未验证 |
| `pseudo-conversational` | 工具会话（一次 run 一条会话，`agent_session_host.py`）；无 MCP **硬失败** | 已接线 2026-08-22，见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.3（本文 [§4 完成与终止契约](#4-完成与终止契约) 有它的完成谓词） |
| 任一档、调用带媒体 part | capsule | 工具协议文本专用 |

工具会话：每次调用建一个单 task assignment（root 在 driver episode 域旁 `assignments/<call-id>`），
agent 经 harness 自己的 MCP server 取 task / 读上下文 / 提交，第一档修复在 runtime 的
`submit → repairable` 循环里跑完（`complete(validator_spec=, max_repair_attempts=)` 把校验器按 id
交进去）；用尽时返回最后被拒输出并打 `repair_exhausted`，`attempts.py` 据此跳到链末，下一次调用即
第二档替换（harness 重路由）。工具会话永远是一次 CLI 调用 = task scope。dev-only 强制某一传输：
环境变量 `FINESUB_AGENT_TRANSPORT=capsule|tool-session`（档位定义性的规则——`api`/`resume`/媒体恒
capsule、pseudo 必须工具会话——不受它影响）。

**开场带上任务**（单 task 工具会话；2026-10-16 起默认开，dev 开关 `FINESUB_TOOL_SESSION_OPENING=0` 关掉、供 A/B）：
harness 在起 CLI 之前以 `worker-1` 的身份 claim 这唯一的 task（`client._claim_for_opening`，「harness 不持 lease」的第二个例外），
开场消息（`agent_tool_worker_opening_v1.md`）带 manifest 与 driver 能带的块，模型不调 `next_task` 直接读、交。
能带什么看 driver config 的 `mcp_opening`：
- `inline`（Codex、CodeBuddy，prompt 走 stdin）：协议与载荷的正文；`kb_index` 在内容的 digest 与 manifest 相等时也带。
  块按「各窗相同 → 每窗不同」放进去，放到渲染后的整条开场超过 driver 的 `opening_max_chars`（[通道上限](#通道上限超限的回复与送达核对)）为止，
  量的是开场作为消息字符串（转义算在内）；放不下的块标 `read: tool`（`kb_index` 调它自己的工具，协议与载荷用 `read_context` 按 ref 读），不按 push 记账。
- `agent_document`（agy，prompt 走 argv、放不下正文）：协议写进 agent 文档（见[各 driver 接线](#6-各-driver-接线)表下）；载荷与 `kb_index` 写成
  一次 `view_file` 读得完的几份（`<assignment>/control/opening/<task>/part-NN.md`），标 `read: file`，开场末尾点名全部份、要模型第一步并行读完；
  整条开场按 UTF-16 量会超过 30,000 时改写成「目录 + `part-01.md` to `part-NN.md`」，模型照样一步读完。
- `system_prompt`（Claude Code，默认；CLI 没有那两个开关时由 `opening_mode()` 退回 `inline`）：协议标 `read: system`，接进 Claude Code 的 system prompt（见[各 driver 接线](#6-各-driver-接线)表下）；载荷与 `kb_index` 同 `inline`。
- `fork`（Codex 开了父会话 fork、CLI 能隔离用户配置时，由 driver 的 `opening_mode()` 答）：协议标 `read: earlier`，放进 driver 建的父会话、子会话从它 fork（见[各 driver 接线](#6-各-driver-接线)表下）；载荷与 `kb_index` 同 `inline`。
- 空（dsh）：照旧 `next_task`。

开场里按「各窗相同 → 每窗不同」排：规则、协议、`kb_index`，然后 manifest 与载荷。带进去的块每个 attempt 都按 push 记一次台账
（premature stop 的 reset 会清台账，新 CLI 拿同一份开场），并经 `FINESUB_MCP_PREDELIVERED` 告诉 server。代领在 driver 排队等槽位之前，
所以 CLI 一起来 harness 就续一次租；排队比租期还久时 task 已回到队列，server 回「hold lapsed，重新调 `next_task`」，模型重领后拿到全部块，多一轮。每个 attempt 记账前先确认 lease 还在本 worker 手里，不在就重新代领；argv 的短 prompt 换成
不叫它领任务的 `AGENT_TASK_PROMPT_TOOL_SESSION_OPENING`；manifest 的 metadata 记 `tool_session_opening`，审计包分得出两种形态。
验收读数在 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) [3.11 单任务工具会话开场带上任务：验收与对 capsule](llm_local_agent_experiments.md#311-单任务工具会话开场带上任务验收与对-capsule2026-10-16)。

为什么这样定（owner 2026-10-16）：
- agy 用 agent 文档形态（协议即 system prompt，开场只点名载荷与 `kb_index` 的份）；开场直接给块的路径是它不灵时的退路，只省 `next_task` 一轮。
- `kb_index` 随开场交付：能内联的三家放在开场里，agy 写进份、与载荷同一步读。
- 否决过的做法：把协议与载荷合成一个文件给 agy（两块共 51 KB，超过 `view_file` 单次 46 KB，照样读两次；协议压到 32 KB 以下时可重看）；关掉 Codex 的 code mode（关了模型就没有工具）；在外壳里叫模型别重读或一次只读一页（重读是截断后的正确反应，外壳又是各 driver 共用的）；把超限的协议拆成「开头内联、其余分页」（一块两种来源，台账规则要改，开场带上任务之后用处不大）。

**身份**：档位进 `routing_identity_digest`（`presets.*.agent_session`），改档位作废 checkpoint；实际
走的传输**不进**身份（owner 显式决定 2026-08-22：同档位下两种传输消费同一 prompt、同一 validator，
产出等价；probe 是这台机器今天的样子不是契约），只记进 route decision trace 的 `agent_transport`。

**validator 跨进程**（`agent_validators.py`）：`submit` 由 CLI 拉起的 server 进程判，闭包过不去，
所以 `complete(validator_spec={"id", "params"})`，server 与 harness 都从 `VALIDATOR_BUILDERS` 按 id
解析；纠错窗口是 `correction-window` + 序列化窗口，variant / tier 进 task metadata。新 validator
必须注册在那张表、参数必须可 JSON 化。

**validator 看到的是 manifest 加 `observed`**（2026-10-01）：runtime 判 `submit` / `lint` 时把 task manifest 与
**它观察到作答的那个上下文做过的事**一起交给 validator（`AgentTaskRuntime.validator_view`），目前是 `observed.kb_reads`
——这个上下文里每次**成功**的 `kb_read`：解析到的条目、类别、有没有带 `sections`。MCP server 在 `_kb_read` 过了回复上限
（`_kb_finish`）之后才记一行（`record_kb_read`），撞上限报错的读不记；这一行不是 fail-soft 遥测，记不进就报工具错、让模型重读。
它和必读块的 `pulled` 是同一份**按上下文**的台账（见[必读块台账与 submit 门](#3-必读块台账与-submit-门runtime-协议-v4)）：
新 lease、`reset_conversation`、退役都清空，checkpoint 保留——旧对话里读过的 style 不算新对话读过。
用途是 fast `--style auto` 的「选中的那套真读过」核对（[知识库契约](knowledge.md)「style 类别」）：`correction-window`
的 params 带 `style_pick`（候选名 + rev）时，validator 还要求交付里有合法的 `<style_selection>`，且选中的
`style/<名>` 在台账里有一行不带 `sections` 的读取。harness 侧用**同一个判断**（`judge_style_pick`）判同一份台账：
会话结束后 `fold_task_ledgers` 把**最后一次 submit 判定时看到的那份**（`submit` 存在 task 行的快照，作答的上下文之后可能已被
reset 或在 assignment 会话里接着干别的）折进这次调用的 execution attempt（`kb_reads`，与代理检索的
`fold_proxied_retrieval` 同一处），所以两边对「合法窗口」的判断一致，会话内修复轮也用得上。一个判定若取决于
`observed`（读的核对没过），结果带 `observed_dependent`，runtime **不按候选指纹缓存**它——同一份答案读过之后要能通过。
不用 driver 的事件流：各 driver 规范化 harness 的 MCP 调用时只留工具名、不留参数，也分不出成功还是超限报错；
也不用知识库 `events` 表的 `exposed`（fail-soft 遥测，worktree 里关着）。

## 2. 生产工具表（唯一真相）

agy 的逐工具 `permissionGrants`、Codex 的 `enabled_tools`、Claude Code 的 `--allowed-tools` 与
readiness 校验都从这张表派生：

| 工具 | 作用 | readOnly | destructive | idempotent | openWorld | 何时暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `next_task` | 以本 worker claim / 续租；返回 manifest、**首轮直接带 `protocol` / `payload` 的整份正文**（按 push 记台账；整个回复超过 driver 通道的上限时内容外置成份、照样记账，见[通道上限](#通道上限超限的回复与送达核对)）、剩余修复轮数。**同一条 CLI 已为更早的 task 交付过、且仍在台账里的 protocol 不再重发**，只标 `read: seen`：pseudo-conversational 的 protocol 全程一份，原先每窗一份副本；载荷是任务本身，照发。两个条件缺一不可——台账在同一 CLI 内 reset 时清零，「交付过」记在 server 进程里、换新 CLI 就没了；开场交付过（`FINESUB_MCP_PREDELIVERED`）且在台账里的 protocol 与 `kb_index` 同样标 `read: seen`。**这个 task 的载荷已经交给过这条 CLI**（开场带了，或本进程为同一个 task 答过 `next_task`）且仍在台账里时，再调 `next_task` 不重发：已交付的块都标 `read: seen`，回复带一句「已经发过、照手上那份做；没有了就带 `{"resend": true}` 再调」（`fragment_next_task_already_sent_v1.md`），带上它才照旧全发；不重发时照样续租。后来的 task 载荷与先前的相同时，它第一次领照样拿到载荷。再调多半是模型手上还有任务（验收里 CodeBuddy 一窗因此多收 37 KB）；真丢了（compact 之后、回复没到）多一轮。`resend` 在参数表里、不写进工具描述，只由这句回复告诉模型。`read_context` 照旧可读。本进程没答过 `next_task` 时，其余工具按本 worker 持有的 lease 找回 task（开场代领、Claude Code 在 CLI 内重启的 MCP 进程），没有 lease 才要求先调 `next_task`。pseudo-conversational 会话里没活时**长轮询**（`FINESUB_MCP_WAIT_SECONDS`，默认 25s）后回 `still_waiting`（再问），seal 后回 `assignment_complete`（退出）；预算耗尽的 task 在被 harness 收回前不再发出 | false | false | false | false | 恒在 |
| `read_context(ref)` | 按 ref 读整块资源，回 `text` 与还欠的 `owed_blocks`，读到即记台账（回复超限时外置成份）；**只接受 manifest 点名的 ref**，错 ref 的报错列出合法 ref（模型会瞎猜，每猜一次一轮） | false | false | false | false | 恒在 |
| `read_spill(spill_id, part)` | 读一个外置回复的第 `part` 份，原样；只读本 task 的份（`meta.json` 记着 task），份的大小留在这家的回复上限之内 | true | false | true | false | 通道 `overflow = tool` 且有上限（Codex、Claude Code、CodeBuddy） |
| `batch(calls)` | 几次互不依赖的只读调用放进一轮：成员只有 `kb_index`、`kb_search`、`kb_read`、`kb_read_node`、`kb_validate`、`web_search`、`web_fetch`（`read_context` 记交付台账、`read_spill` 每份已贴着上限、`batch` 不能套 `batch`，放了只那一项报错）。按顺序逐个执行，每项走与单独调用同一条分派（授权、台账、frame log 照旧）；第 i 项的请求 id 是 `{请求 id}-b{i}`，重放回缓存的整份回复、一项都不重跑。回按顺序的列表，每项 `{tool, result}` 或 `{tool, error}`，整体过一次通道 | true | false | false | true | 只在 Codex（`ChannelLimits.batch_tool`：GPT-6 在 Codex 里不在一次回复里并行发调用） |
| `pull_status` | 本 context 还欠哪些必读块 | true | false | true | false | 恒在 |
| `submit(payload)` | validator → accepted / repairable / retired；**按内容幂等**（同一 context 内重投已判过的答案 = 重放首次判定、不烧修复预算），修复预算与线上 submit 次数都记在 **durable task 行**（server 进程内不计数），用尽 / 超 submit 上限 / 被退役后固定回「停止」 | false | false | false | false | 恒在 |
| `web_search` / `web_fetch` | 经 harness 本地检索代理，runtime 检索账本计费 | true | false | false | true | **暴露**：单 task 会话按 task `retrieval_mode=local`；pseudo-conversational 会话发车即暴露/授权全部六个（三家授权只能在 invocation 时定），因此该档下检索授权一变就**替换**会话（同 key 只有一条 CLI，见 `llm_local_agent.md` [§12.1.3 pseudo-conversational](llm_local_agent.md#1213-pseudo-conversational已接线2026-08-22形态如下)）。**调用时准入**：server 按当前 task 的 `retrieval_mode` 放行，非 `local` 的 task 调它报错。三态原样传到 manifest（`client._agent_task_inputs`）：**`local` = harness 执行检索**（这两个工具，受 runtime 台账预算约束），**`native` = provider 自己的搜索工具**（driver 按 `--tools` 授权 `WebSearch`/`WebFetch`，harness 不再提供 web 工具，预算降级为事后记账），`none` = 都不给。曾经把三态压成布尔、于是每个真值都变成 `local`——`retrieval=native` 因此在所有 agent 调用上跑的是 harness 代理，正是该开关声明自己绝不做的事。**出处**：`local` 下代理检索的 URL 只在 runtime 账本里，而下游（`knowledge.feedback.retrieval_urls_from_response`、`research._mark_unverified_sources`）只看 driver 归一化事件——收尾时由 `retrieval_search_events()` 折进事件面，否则「harness 亲自抓的页」反而一个来源都不记；`native` 下 driver 的归一化事件本身就带 URL。三种会话形态共用 `fold_proxied_retrieval()`（单任务、pseudo-conversational、conversational——后两者每个任务的事件面本来就是空的，不逐任务折就等于没有出处）；读不出的结果引用如实回报并告警，不静默缩短来源表 |
| `kb_index` / `kb_search` / `kb_read` / `kb_read_node` / `kb_validate` | 知识库只读五件套（`knowledge-node-plan.md` §4.3/§6.5）：索引、精确检索（surface/alias/misheard，假名归一）、按条目渲染（带 `@k` 句柄；预览档位随本任务的 `kb_tools` 走——`propose` 得完整预览（空节、收录纪律注释、core 空槽），`read` 得部分预览；`sections` 缩小范围——section tier 随行文法 v3 退休）、按句柄读单节点、提案无副作用预检（`kb_validate`：共享 translate+preview，**仅 `propose` 授权准入**（`KB_ENTITLEMENT_TOOLS`），即知识更新；prompt 的句柄表经 `metadata.kb_handle_bindings` 播种进会话，工具与提案块同一句柄空间）。**读哪个 rev/root 由 manifest 决定**（`metadata.knowledge_identity` "rev:N" + `metadata.kb_root`，spawn env root 只是同 run 缺省），回复带 `knowledge_read_rev` 与 `result_digest`（frame log 即审计）；`kb_read` / `kb_search` / `kb_read_node` 硬上限 `KB_REPLY_MAX_CHARS`（「一次给模型多少知识」，不是传输），超限报错提示缩小（`kb_read` 的报错列出该条目各节名与字数，供 `sections` 填）、无分页、`kb_search` 不做静默截断；`kb_index` 不受它管：它是必读块，整份多大由通道去送 | true | false | true | false | **暴露**：spawner 的 tools 列表，**逐个工具**（无列表时看 env root，有 root 就全给；单 task 工具会话现在总是给列表，`FINESUB_MCP_TOOLS` 与告诉 CLI 的一致）。**调用时准入 = 调用方显式授予**：stage 调用带 `kb_tools: read\|propose\|style`（授权矩阵：correction/research/query/fast=read，knowledge-update=propose，其余不授；**`style` 是窄授权**（2026-10-01）：没开知识库读取、只为 `--style auto` 选风格的 agentic fast 窗，以及没有词条索引、有可选候选的 agentic 准备任务——只有 `kb_read`、只解析 `style/<名字>`，不欠 `kb_index` 必读块，自带 root + rev 不靠 pin；每个授权能用哪些工具在 `agent_mcp_server.KB_ENTITLEMENT_TOOLS`（`read` 不含只给知识更新的 `kb_validate`），pseudo-conversational 会话开场暴露全部工具，所以逐调用再核一次），client 折进 manifest metadata，server 逐调用核——generation pin 只供缺省 root/rev，**有 pin 不等于有授权**；standalone 知识更新显式带 root+identity、无 pin 也获得完整绑定。**必读门**：获授权的 task 欠一个 `kb_index` 必读块（manifest 标 `read: tool`，worker 外壳写明这种块用 manifest 点名的工具取；agentic 准备任务因此不在 payload 里另贴索引，与 agentic 纠错窗一致），回复 digest 与 manifest 相等才记 pulled、submit 才放行；`propose` 下整条 `kb_read` 挂条目尺寸提示，manifest `metadata.kb_size_notice: off`（提案回喂）时不挂；`kb_validate` 需 `propose` 且把 fold rejection 计入结果（`ops_translated/ops_appliable/rejected`）；conversational 任务不挂块（其控制协议无 kb 工具） |

注解**如实**：Codex 在 `default_tools_approval_mode="auto"` 下放行只读、或既不破坏也不开放的工具
（0.160 源码 `requires_mcp_tool_approval`；测试逐个核对每个暴露给 Codex 的工具过得了），不伪装只读。`web_search`
（open-world）已实测可过 Codex `auto`。

**request id**（2026-08-22 起对 `submit` 只是审计字段，正确性由下一段的内容指纹承担）：MCP 调用
`H(assignment, worker, server_session_id, server_instance_id, JSON-RPC id)`；`server_session_id`
每次 CLI invocation 由 driver 经 `env` 传入，`server_instance_id` 由每个 MCP server 进程随机生成。后者不可
省：Claude Code 会在同一 CLI 内重启 MCP 子进程并把 JSON-RPC 计数归零。server 按
`request_id → {参数指纹, 回复}` 缓存——同一 live server 内同 id 同
参数返回缓存（修复计数不动两次），同 id 不同参数回 conflict。runtime 侧每个有记录的操作都带输入
指纹：同 id 同指纹 = 重放，不同指纹 = `AssignmentConflictError`；`record_pull` 只存指纹不存正文；
表按 assignment 512 条硬上限。harness 自己的收尾调用（reset / retire，每会话一次、无恢复方）用
`H(server_session_id, 操作名)`、不持久化，是「可恢复的调用者必须持久化请求序号」这条总规则的
显式例外：harness 进程崩溃时整个 run 由用户重跑、resume 靠窗口级 checkpoint，不存在重建这个 id 的消费者。

**`submit` 的内容幂等与两个计数**（[§7 未做与闸门](#7-未做与闸门) 收口，已实施 2026-08-22）：task 行按 lease 持有
`submissions: {H(task, lease, input_hash, candidate) → 首次判定}`、`submit_count`、`repair_attempts`，
三者在新 lease（claim / retire）时清零、`reset_conversation` 保留（同 lease）。每次 `submit`：
同 `request_id` 传输重放 → 两计数都不动；新 id 同指纹 → `submit_count += 1`、回首次判定并标
`replayed`，`repair_attempts` 不动；新指纹 → `submit_count += 1`，validator 判 repairable 再
`repair_attempts += 1`。欠块拒绝既不计预算也不缓存（它的判定取决于此后读了什么，不取决于答案）。
`submit_count > max_repair_attempts + 3`（`EXTRA_SUBMITS_PER_CONTEXT`）→ 同一次落盘内退役，
`protocol_violation=submit_cap`——这是「死不改口」会话的出口。`max_repair_attempts` 数的是修复轮，同 API 路径：
首答之外再修这么多轮，所以 `repair_rounds_remaining` 是 `max_repair_attempts − max(0, repair_attempts − 1)`（第一次被拒开出第 1 轮，
第 N+1 次被拒才用尽；owner 2026-10-31）。它与停止回复一律从 `task_record()` 派生，并在启动时对账：本 assignment 已完成即答「已 accepted，请退出」。

**补丁修复**（2026-10-23）：validator 判 repairable 时可以在回复里带 `repair_instruction`，server 原样转给模型；
纠错窗口的 validator 在答案大半是对的时候（门槛同 harness 的修复轮，见
[`llm_harness_behavior.md`「重试与拼接」](llm_harness_behavior.md#重试与拼接)）用它要一个只写要改的行的 `<patch>`；说话人裁判的
validator（`speaker-judge`）在 `<judge>` 里有能留下的东西时（门槛同 harness，见 [`llm_harness_behavior.md`「说话人裁判」](llm_harness_behavior.md#说话人裁判--diarize-judge-text--judge-audio)）要一个只写出错与缺的行的 `<patch>`（按 id 叠），否则只回错误、要整份。注册了
`merge` 的 validator（`agent_validators.VALIDATOR_MERGES`：`correction-window` 与 `speaker-judge`）收到补丁时，runtime 在
`submit` 与 `lint` 里先把它叠到**底稿**上，拼出整份再走上面的一切（内容指纹、欠块门、validator、`last_candidate`），
所以会话没 accept 就结束时交回的也是整份。拼出的整份仍被拒时，回复另带 `patch_merged`：修复说明里的「你刚才那份」指拼出的整份；validator 再要补丁时还带
`rows_to_fix`，即拼出的整份里含待修源的那几行（裁判是出错的那几行；不给整份：会话有自己交过的整份和补丁，缺的只是这几行现在的样子；缺的源没有行）。
同一份整份再次提交拿回的旧结论（replay）也把它重新定为底稿，不带 `rows_to_fix`（会话能从自己的底稿与补丁重新拼出那份整份）。底稿是这个 lease 在这个对话里最后一份被拒的整份答案（`last_candidate`，
台账里记 lease 与摘要；`reset_conversation` 清掉对话台账、新 lease 换代，都让旧底稿失效）。
补丁叠不上（某行读不出源序号或 id、源序号或 id 在窗外、和整份同时出现、纠错的补丁转义成 `\uXXXX`；列数、JSON 之类的错照样叠上，交校验器判）回 repairable、底稿不变、计一次修复、不进内容缓存；
没有底稿时回 repairable 要整份，不计修复。
会话开头的规则里讲 `repair_instruction`、`patch_merged`、`rows_to_fix` 的几句（`fragment_agent_tool_patch_repair_v1.md`）只在开场带着的任务收补丁时渲染（它的 validator 在 `VALIDATOR_MERGES` 里，`task_record` 的 `validator_id`）；准备任务这类修复轮只收整份的不带。还没领任务的 bootstrap 与 pseudo 会话照旧都带。

## 3. 必读块台账与 submit 门（runtime 协议 v4）

- task 声明 `required_blocks`（`kind` / `digest` / `ref` / `tool`），`@protocol` / `@context` 占位由
  runtime 在物化文档后填 ref 与 digest；身份是 `(kind, digest)`。
- 台账存在 context 里：assignment scope 在 conversation 记录上（reset / retire 整条替换即清零），
  task scope 在 task 行上（每次 lease 清零，`reset_conversation` 也清）。push 与 pull 都算。
- `submit` 前置门：欠块首拒一次列全并点名工具，每 context 一次补读机会（硬编码）；补读后仍欠任
  一块 = 会话不服从协议，**同一次落盘内退役**（`_retire_task_locked`：会话 reset、lease 撤销、重入
  队、`retirements+1`、存 `last_candidate`），返回 `retired`。
- 静态块规则保证的是「模型有机会看过」，不是「上下文里一定有」；compact 后重读靠 `read_context`。

### 3.1 工具参数的转义：入口修复 + 出口拒绝（2026-09-04）

antigravity-cli 1.1.24 **无法把非 ASCII 字符放进工具参数**：它把每个非 ASCII 码点写成 JSON
字符串**内部**的字面 `\uXXXX`，于是 `json.loads` 交回六个 ASCII 字符。注意它只对非 ASCII 做这
件事（引号与反斜杠没有加倍），所以这不是「被编码了两次的 JSON」，再解析一遍解决不了。

实测一次（下游提供的 `llm-artifacts`，本地复核）：一个 `submit` 载荷是 5991 个字符、全 ASCII、
815 个反斜杠后面全跟着 `u`，815 个转义全部解出非 ASCII；那一窗的字幕就以转义原样进了 SRT 与
`correction-windows.jsonl` 断点缓存，而 stage 全程报成功。

两层，职责不同——**这是有意的冗余，不是「保险起见都做」**：

| 层 | 位置 | 作用 |
| --- | --- | --- |
| **入口修复** | `agent_mcp_server._unescaped_arguments`，帧进来时、在请求指纹与任何 handler 读参数**之前** | 让运行照常成功。**绝不静默**：**每修复一帧**往 stderr 写一行、带转义个数，落在 capsule 的 `events/stderr.log`。⚠ 不是每进程一次——exchange 产物记的是 harness **收到**的东西，那已经是修复后的文本，所以这行是「发生过改写」的唯一记录；个数让人分得出整窗污染（几十个）与一次可疑改写（一个） |
| **出口拒绝** | `output_protocol._refused_reply`，在其余每一项检查**之前** | 不变量：坏内容永远到不了 SRT，也到不了断点缓存 |

判据是共用的一个（`finesub.text.looks_escaped`），**只剩两条**：**纯 ASCII** ∧
**至少一个转义解出非 ASCII**。解码在 `unescape_codepoints`：BMP 外字符的代理对会重新拼合，
落单代理直接放弃而不猜，⚠ 且**只解 U+007F 以上**——拼出 ASCII 字符的那个是模型自己敲的，
连它一起解就从「修复传输」变成了「编辑内容」。判据住在 `text.py` 是因为两侧都要它、
而两侧互相不能 import。

⚠ **曾经有过两条下限，都删了**（2026-09-04）：先是转义密度 ≥0.20，后是转义个数 ≥4。两条量的
都是「这次任务长什么样」而不是「故障长什么样」——密度量源语言（同一份完整污染，日→中 0.859、
英→中的字幕行低到 0.072），计数量窗口大小（单行窗口配两字译文只有两个转义，被静默放行）。
标定脚本是一次性实验，在 `explore/`，不随公开快照发布；读数就是上面这两个。

⚠ **两层的冗余有边界**：它们**共用同一个判据**，所以出口拒绝冗余的是**入口修复那条代码路径**
（回复没走过 `agent_mcp_server` 的传输面、replay 存档、或修复的递归漏掉某种参数嵌套），
**不是判据本身**。agy 若改成只转义一部分字符，回复不再是纯 ASCII，**两层一起漏**。
那个盲区是目前唯一真正的防线缺口，但它是**推测形态**——还没有观测到过一次部分转义。唯一能堵它的
办法是去掉「纯 ASCII」条件、改判转义密度，可阈值只有拿到部分转义的真实素材才标得出，而且因为
两层共用判据，它会同时改变入口修复的行为（修复一段只夹着几个字面转义的正常文本）；所以**暂缓到
生产里真正观测到一次部分转义再动**。

⚠ **信号早就有过一次，被扔掉了**：那次运行的 `char_count` 校验其实全数命中——15 行全部对不上、
比值全部约 3 倍（一个 CJK 字变 6 个 ASCII 字符、按 0.5 权重折算）——但它的处置是 `normalized`：
用被污染的计算值覆盖模型写的诚实值，然后放行。**「个别行对不上」normalize 是合理的，
「每行都对不上且同方向」不是**。后来加过一条窗口级 warning（不符行 ≥1/3 且比值中位数 >1）；
没有变体再让模型写 `char_count` 以后，这个信号连同逐行对照一起没了（2026-11-07 删掉），
这类事故现在只靠上面的出口拒绝（`output_protocol._refused_reply`）。

## 4. 完成与终止契约

| 情形 | 契约 |
| --- | --- |
| 完成 | **`accepted` 是唯一完成点**；最终 assistant 文本无产物语义。driver 监督循环接 `completion` 谓词（约每秒查）。**谓词按会话形态分两种**（第四轮修订 2026-08-22）：单 task 会话传 `task_record().status == "accepted"`；pseudo-conversational 会话传「assignment 已 seal 且全部 task 终态（或已 failed）」，单个 task accepted 不触发结束，CLI 继续 `next_task`。谓词为真之后：**单 task 会话默认立刻结束进程树**（`[llm] local_agent_exit_on_accept = auto`：补得了用量的 driver 才这样，Codex 与 dsh 不；`run(exit_on_completion=True)`，attempt 记 `ended_on_accept`）——MCP server 把让 assignment 完成的那条 `accepted` 回复记进帧日志后压住 `ACCEPTED_REPLY_HOLD_SECONDS`（10s）再发，CLI 收不到回复、就没机会再发一次「submitted」收尾请求（实测这一次在 agy 上约 0.9 万未缓存输入）；这是预期结束，照干净成功 settle，用量由流里逐请求的用量补全（[`llm_local_agent_runtime.md`「目标形态」](llm_local_agent_runtime.md#目标形态)）。设置关掉、以及 pseudo 会话，给 CLI `ACCEPTED_EXIT_GRACE_SECONDS`（15s）自行退出，否则回收进程树、**原始事件流保留**在 capsule（`raw_events_retained`）。两种结束都不要求最终消息 / 零退出码 / 干净事件流 |
| driver 在 accepted 之后出错 | 先读 runtime：accepted 就收下，driver 错误降级为 warning |
| 首次 premature stop（未提交即退出） | 持原 lease 调 `reset_conversation`（清台账），同 generation 起一个新 CLI 会话，不计替换；`AGENT_PREMATURE_STOP_RETRIES = 1`。driver 抛的错里只有 `LocalAgentRefusedCallError`（CLI 拒掉了模型的畸形调用、turn 因此空着结束，见[各 driver 接线](#6-各-driver-接线)的 agy 一行）也按 premature stop 走，其余照「第二次静默退出 / 修复耗尽 / driver 抛错后」那一行。**每一次**未提交的 attempt 都标 `premature_stop` 并带 `final_text_tail`（`no_submit_attempt`）——此前只有触发了重试的那几次被标，决定失败的最后一次反而是唯一没标的。文本尾巴让 artifact 自己就能分辨「没写出答案」与「写了没 submit」；完整现场（事件流、stderr、输入）在保留下来的 capsule 里，attempt 上的定位符指向它；会话宿主那条路同样带尾巴，并把 driver 的 `retained_unaccepted` 抄到 attempt 上 |
| 交过答案、被退回后离开（单 task 会话；pseudo 会话宿主照 [`llm_local_agent.md`「12.1.3 `pseudo-conversational`」](llm_local_agent.md#1213-pseudo-conversational已接线2026-08-22形态如下)的「CLI 中途退出」一条重开 CLI） | 带着文字退出，或 turn 空着结束（driver 抛 `LocalAgentEmptyTurnError`），两种一样：不重开 CLI，代退役 task，返回最后被拒的答案并打 `repair_exhausted`，修复轮数报实际用掉的。纠错据 `repair_exhausted` 把这条链按用完算，第二档还有名额时下一次调用是替换（重新路由，通常仍是同一目标）；校对与说话人裁判把轮数记进自己的预算，有剩就带着被拒的答案与错误再问。答案已经在 runtime 里，空着的最终文字只说明模型丢下了修复，不是目标出了故障；以前它按瞬时错误走路由链的下一个目标，链上只有一个目标时整窗没交（2026-11-05 Codex Luna）。没交过答案的空 turn 仍是 driver 错误，照下一行 |
| 第二次静默退出 / 修复耗尽 / driver 抛错后仍持 lease 且未 accepted | harness 代已退出的 CLI `retire_task`（「harness 不持 lease」的两个例外之一，另一个是开场代领，见[档位与形态](#1-档位与形态)），task 重入队、无主；耗尽返回 `repair_exhausted`，抛错照抛 |
| `retire_task` | CAS 用 lease 本身（`worker_id + lease_generation`），`request_id` 幂等；与迟到 `submit` 先落盘者赢；复用同一 worker id、起 fresh CLI |
| 重投已判过的答案 | 同 context 内重放首次判定、不烧预算（[§2 生产工具表（唯一真相）](#2-生产工具表唯一真相)「内容幂等」）；线上 submit 次数超 `max_repair_attempts + 3` → 退役（`submit_cap`） |
| 反复退役 | 每次退役都把 task 重新入队并**重置预算**，所以退役次数本身要封顶：`MAX_RETIREMENTS_PER_TASK = 2`，第二次退役直接把 task 判 `failed`（host 立刻抛错进第二档），否则不守协议的会话会一直重来到 per-task 期限为止 |

## 5. 审计包

每条工具会话写进 capsule 的 `audit/`——pseudo-conversational 会话一次调用服务多个 task，
所以按 task 分名 `audit-<task_id>/`（随 capsule 保留规则：成功可删、失败留到 `agent-clean`；**未被 accept 就干净退出的会话也算失败**——它的 capsule 从 2026-09-16 起不再按 transport 成功删除，审计包因此写得进去，归 `RETAINED_UNACCEPTED_EPISODES` 池滚动，见 `docs/llm_local_agent_runtime.md`「目标形态」）：
`manifest.json`、`blocks/<kind>.md`（必读块**正文**）、`blocks/<kind>.json`（**工具服务块**只有声明可留：
`kb_index` 由自己的工具取、`ref` 就是那个工具名而非产物引用，正文不落盘，凭据是声明里的 digest 加
拉取台账。审计包是全有或全无，所以早先把它当产物读会让**每条绑知识库的调用**整包写不成——
判据是 `is_artifact_reference()`，别再靠「遍历 required_blocks 就能读到正文」这个假设）、
`outcome.json`（最终 durable task record——
reset / retire **之后**重新读的、拉取台账、conversation epoch 与全部 `resets`、`error`）、
`artifact.txt`、`mcp-frames.jsonl`（server 记的每一帧）。先写临时目录再原子 rename；**写失败不删
assignment root**（它此时是唯一证据）并打 warning；driver 抛错路径也写。
「写失败」只指**有 capsule 可写却没写成**：一次**被 accept 的**干净调用里 driver 已经把 capsule
删了（那正是上面的保留规则），此时没有 locator、也没有证据要留，assignment root 照删——反过来把每
条成功调用的 root 都留下，是这条规则的反面。没被 accept 就干净退出的调用不在此列：它的 capsule
会被保留，审计包照写——但单 task 工具会话只对**最后一个** result 写一次，首次 premature stop 之后
重试时，前一个 capsule 留下却没有审计包（见 `docs/llm_local_agent_runtime.md`「目标形态」）。

## 6. 各 driver 接线

| driver | 声明 server | 授权 | 事件层 | 实测 |
| --- | --- | --- | --- | --- |
| Claude Code 2.1.231 | `--mcp-config` 内联 JSON（身份走 `env`，`cwd` 未见文档），去 `--safe-mode`（它关 MCP；managed hooks 的残余面按 owner 口径接受），`--strict-mcp-config` 已在 | `--tools ""` 精确移除全部内置工具，再用 `--allowed-tools` 逐个授权 `mcp__finesub__<tool>`；不开 `Read` | `tool_use` 按 `mcp__finesub__*` entitled | 整链通过（Haiku），也发非标准 `server/discover`（答 -32601） |
| Codex 0.147.0；复验 0.157.1 | `--config 'mcp_servers.finesub = {command, args, env, default_tools_approval_mode = "auto", enabled_tools, startup_timeout_sec = 30}'`（`_codex_mcp_server_override`），`--ignore-user-config` 下仍生效；另带 `--config features.code_mode.direct_only_tool_namespaces=["mcp__finesub"]`（[通道上限](#通道上限超限的回复与送达核对)）与 `--disable apps`，不让 ChatGPT 账号连的 apps（`codex_apps`）进工具表（2026-10-22，见 [`llm_local_agent.md`](llm_local_agent.md) [§14 当前实现与迁移注意事项](llm_local_agent.md#14-当前实现与迁移注意事项) 讲 Codex 隔离的那段） | `enabled_tools` | `mcp_tool_call` 按 `(server, tool)` 判 entitled，`command_execution` 照禁 | 整链通过（0.147.0 + gpt-5.6-luna）；`mcp list` 加载检查通过；server 进程在沙箱外。0.157.1 + gpt-6-luna 对桩 server 复验 `next_task → submit`（2026-09-26） |
| agy 1.1.18 | 第三、第四种 project `.finesub-tool-<slot>` / `.finesub-tool-native-<slot>`（一个进程同时最多持 `max_parallel` 个，一次调用持一个；每个 slot 另由域根下的锁文件 `.finesub-tool-slot-<n>.lock` 跨进程独占，别的进程持着的号跳过，所以号可能大于 `max_parallel`-1；**按本次是否 `retrieval=native` 二选一**，两份各写一次而非就地改写），**每次 invocation 前**原子重写其 `.agents/mcp_config.json`（身份走 `env`）与 `.agents/view_roots.json`（本次 assignment root：开场的份与外置回复的份都在它下面，交给 `view_file` 读——agy 的 MCP 回复超 ≈4k 字节即被它自己换成模型读不到的文件路径，见 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) [§5 MCP 工具结果的内联上限](llm_local_agent_agy.md#5-mcp-工具结果的内联上限约-4k-字节超过即外置成文件无预览2026-08-22)） | 只写 project **自己的**记录 `~/.gemini/config/projects/<id>.json` 的 `permissionGrants`（逐工具 `mcp(finesub/<tool>)`，另加 `read_file(*)`（agy 1.2.13 起 `view_file` 也要），native 变体再加 `read_url(*)`；缺记录 fail closed；路径是 agy 的实现细节，只定义在 `finesub_bootstrap/agy_records.py`）；guard 放行 `call_mcp_tool@finesub` 与 `view_roots.json` 所列根之下现有文件的 `view_file`，native 变体另放行 `search_web` / `read_url_content`（2026-08-30 补齐；此前研究轮在这条路上被静默拒绝，见 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) [§6.1 tool-session 也要一对 proj…](llm_local_agent_agy.md#61-tool-session-也要一对-project2026-08-30-补齐)）；agent 文档写 `tools: [view_file]`（native 再加两个检索工具）+ `inheritMcp: true`，由 `inheritMcp` 带上 `call_mcp_tool`；**不能**写 `mcpServers`（agy 会整份丢弃、静默退回默认 agent——08-21 到 10-06 的工具会话都是这样跑的），也不能把 `call_mcp_tool` 列进 `tools`（executor 建不起，`not found in registry`）。`_ensure_project` 的 `/hooks` 核对带 `--agent` 与独立 `--log-file`，日志出现 `Agent "<name>" not found`（或日志里没有 agy 的版本行、判不了）时工具 project 失败关闭；capsule project 与 probe 只发一次警告（`agent-cli-agent-document-ignored`；判不了时 `agent-cli-agent-document-unverifiable`）、照常运行（owner 2026-10-06：边界在 hook，agy 又是音视频窗口唯一的路）（agy 1.2.16，2026-10-06） | `call_mcp_tool` 按 `(ServerName, ToolName)` 判 entitled，别的 server、或 server 有但本次没给的工具（如没授检索的调用去调 `web_search`）都判违规。唯一的例外是模型的畸形调用：ToolName 是 server 根本没有的名字（`SERVER_TOOL_NAMES` 之外；实测是把 `call_mcp_tool` 又套进 ToolName），且 agy 自己拒掉了（那一步 ERROR、没有 DONE，没到 server）——不算违规，那几行记 `refused_call`、attempt 记 `refused_tool_calls`。agy 拒掉它时会结束整个 turn：会话已被 accept 就照收；没 submit 就抛 `LocalAgentRefusedCallError`、按 premature stop 再问一次（owner 2026-10-31，agy 1.4.11 + Gemini 3.8 Flash 实测） | 单槽与双槽并发都通过；hook 看得见 MCP 调用但授不了权 |
| dsh 0.1.5-rc.1（接线于 0.1.1-rc.2） | `--patch <capsule>/input/dsh-patch.yml`（写成 JSON——JSON 即 YAML，省掉 Windows 路径的转义坑），条目必须用 **`insert:` 列表**：裸条目是按 id 定向覆盖，profile 里没有 mcp-client 可覆盖，patch 引擎只 warn 就跳过——模型没工具而 harness 毫不知情。`serverName` 即 `finesub`，身份走 `env`，`failOnStartupError: true` | 无工具白名单；同一份 patch 按 id 把 plugin **关掉**（比白名单强：工具不注册），`DSH_PERMISSION_MODE=read-only` 兜底。留 `tool-fs`（读侧），`tool-web` 只在 native 轮留；0.1.5 起抓取后端 `web-fetch-http` 与它同进同退（`dsh-tool-web` 按开关注册 `web_fetch`、不看后端在不在，只关后端会留下一个一调就报错的工具） | **会话 transcript**（2026-08-30 接线）。stdout 仍只有答案，但 session-persistence 插件一直在写完整的 `tool/call` / `tool/result`；driver 用 patch 把它的 `root` 指进 capsule、`compression: none` 写成明文 JSONL（不共享用户 `$DSH_HOME/sessions`——那里已是 zstd，插件 README 明说一个 root 只能一种编码），`_dsh_session_rows` 读出来。⚠ 0.1.5 起文件名是 `session.v3.jsonl`，只认旧名时一行都读不到；现在认 `session.jsonl` 与 `session.v<N>.jsonl`。`web_fetch` 也记成检索行，`urls` 只取所抓页面（参数与结果首行 `Fetched <url>`），不收正文里的链接。`observes_tool_events` 已置 `True`。entitled 判定仍退化成「不存在的工具不可能被调用」 | 整链通过：玩具任务，以及 **270 条真实纠错窗口**（0.1.1-rc.2，v4-flash，782s，三次工具调用、零重试）。0.1.5-rc.1 上合成 native 窗口两次（2026-09-12）：搜索、抓取、`submit` 被接受，会话日志读回审计。一帧 `next_task` 就 55,096 B > 出厂 `maxInlineBytes` 50,000，`spill-policy: {}` 是前提而非保险。读数、限速端点的假阴性、thinking 为何只对自带路由生效，见 [`llm_local_agent.md`](llm_local_agent.md) [§12.1.0 dsh](llm_local_agent.md#1210-dsh第四家-driver只落在-per-window-一格2026-08-25-接线) |
| CodeBuddy 2.137.1 | `--mcp-config` 内联 JSON（身份走 `env`），`--strict-mcp-config` 已在。没有 `--safe-mode` 可去——这个分支根本没有那个 flag | **没有单独的授权面**：`--tools` 既是可用性边界也是唯一的授权，harness 的工具名直接写进它；`--allowedTools` 一律不发（variadic，放在 prompt 前会把 prompt 吃掉，而且它什么都不多给）。另外**必须**在环境里设 `CODEBUDDY_DEFER_TOOL_LOADING=0`：默认 MCP 工具是 deferred 的，写进 `--tools` 也看不见，模型拿到的是 `ToolSearch` | `tool_use` 按 `mcp__finesub__*` entitled（与 Claude Code 同一份归一化）；⚠ `system.init` 的 `tools` 报的是**注册表**不是本次可用集，所以公告集审计对这家关掉 | 整链通过（deepseek-v4-flash）：`next_task → submit`，且一帧 147,034 字符的 `next_task` 在 `MAX_MCP_OUTPUT_TOKENS=200000` 下完整送达——出厂上限下它会被换成一个文件路径，与 dsh 的 `spill-policy` 同形。读数与其余四条分支差异见 [`llm_local_agent.md`](llm_local_agent.md) [§12.1.5 codebuddy](llm_local_agent.md#1215-codebuddy第五家-driverclaude-code-的分支2026-09-04-接线) |

Codex 的 `fork` 形态（`CodexDriverConfig.parent_fork`，默认开）：每个协调域按父会话的整条命令行（模型、推理档、连接、`--search`、工具集、home）加父会话那条消息（协议正文连同 `codex_fork_parent_v1.md` 的措辞，措辞一改旧父会话就不再用）与 CLI 版本算一个键，`<域>/.codex-parents.json` 记键到父会话 id 与它那一回合的用量；一个键第一次出现只记下、这一窗不 fork（只用一次的协议——准备任务、知识库分块——建父会话纯属多花，owner 2026-10-17），第二次出现才在域锁（`.codex-parents.lock`）下建：非 ephemeral 的 `codex exec`，最多跑 180 秒（`CODEX_PARENT_TIMEOUT_SECONDS`：建的时候整个域的父会话锁都等着它），cwd 是 `<域>/.codex-parent-cwd`（建之前确认是空的：Codex 读 cwd 里的规则文件，而父会话那一回合每个子会话都继承），MCP server 用同一套工具、身份是不接任务的 `fork-parent`、根是空的 `<域>/.codex-parent-mcp`，唯一一条消息是协议（`codex_fork_parent_v1.md`，只回 READY）。每窗的工具会话跑 `codex exec --ephemeral … fork <父会话> -`：fork 沿用父会话的 cache key，第一次请求就读到工具定义、base instructions 与协议。提示词与开场一起走 stdin（argv 末尾必须是 `-`：同 `exec resume`，给了提示词参数 Codex 就丢掉管道里的 stdin，2026-10-17 验收里每窗因此改用工具把块全读了一遍）。子会话的 `turn.completed` 把父会话那一回合也算进去：建父会话的那一窗照记（那一回合是它的花费，`fork_parent_usage: included`），之后各窗扣掉（`subtracted`；扣成负数就照原值记 `mismatch_kept`）；审计的 `isolation.fork_parent` 记父会话 id 与是不是这一窗建的。表里记着的父会话，rollout 已不在它那个 home 的 `sessions/` 里就先重建再 fork；Codex 约一周后把旧 rollout 压成 `.jsonl.zst`，这也算不在，多付一次重建（压缩过的父会话能不能 fork 没测，不赌）。fork 本身失败照普通失败分类，不重建。父会话的回复只能是 READY、不能有工具调用或搜索（Codex 可恢复的 `error` 诊断项不算），否则算没建成。建不成的键在表里记下时间与原因，30 分钟内（`CODEX_PARENT_RETRY_AFTER_SECONDS`）不再建，这期间的窗照下一句不 fork。父会话在同一套 kill-on-close 进程树里跑，stderr 写进 `<域>/.codex-parent-logs/`。父会话建不起来时不 fork，协议作为开场前单独一条消息带上（`agent_protocol_earlier_v1.md`），开场里的 `read: earlier` 仍然成立。父会话建在这次调用所用的 home 里：凭据走环境变量的连接是 harness home（`<域>/.codex-home`）；ChatGPT 登录（或 `codex login` 存的 API key）的连接是用户自己的 home，每个键一条 `exec` 来源的会话，Codex Desktop 的会话列表不显示这类会话（owner 2026-10-17 确认），子会话照样带上用户的全局 `AGENTS.md` 与开头那句用户规则提示。父会话不能用 `codex archive` 藏起来：归档之后 fork 被拒（探针 2026-10-17）。所以用户的 home 里会慢慢攒下这些会话：每个键一条，协议正文、措辞、模型、推理档或 CLI 版本一变就是新键；纠错的协议带着任务自己的档案与风格块，说话人裁判的带着任务的全局摘要，所以实际上是每个任务、每种会话各一条（一个任务这种会话不到两窗时不建）；`agent-clean` 删掉域连同 `.codex-parents.json` 时不碰它们，之后同一个键再建一条。要清理，用户自己 `codex delete <id>`；让它们不进用户 home 的做法在 `llm_followups.md` 的「Codex 的全局 `AGENTS.md` 结构性挡住」。跨 run 沿用同一个父会话不会比重建更贵：缓存凉了，第一个子会话付一次前缀、写热缓存，重建则还要多付父会话那一回合。每个键一个父会话，是因为同一个键的子请求共用一个 cache key：OpenAI 说每个 key 每分钟超过约 15 次会溢出到别的机器，溢出只是没命中、不是失败；探针 35 秒内 36 次请求、验收里 3 个窗并行，都没看到溢出。开场 `earlier` 一档和父会话、回退消息里那几句说得很重（那段就是这个块、不要先调 `read_context`），是实测逼出来的：第一版措辞下 14 个 fork 窗有 3 个又用 `read_context` 把协议读了一遍（每次多约 1.5 万未缓存），模型不信协议已在前面；加重之后 13 窗 0 次（2026-10-17）。冷读里要把这几句改软，先拿这组数比。验收（2026-10-17，GPT-6 Luna，uchida 10 分钟 5 窗；未缓存 / 缓存读 / 输出）：凭据走环境变量的连接，fork 的窗每窗约 1.0–1.2 万 / 9.0 万 / 0.3–0.6 万，不 fork 时未缓存 2.5–3.8 万，串行、并行首次接受都是 5/5；ChatGPT 登录的连接 0.93–1.17 万 / 9.6–9.9 万 / 0.27–0.43 万，不 fork 时未缓存 2.75–3.17 万，5/5。

agy 的开场形态（`mcp_opening = agent_document`）：`_write_project_resources` 把这次调用的协议接在 agent 文档固定部分之后
（`AGY_AGENT_PROTOCOL_MARKER` 一行分界，`agy_tool_agent_protocol_v1.md` 那几句叫模型直接读载荷、不读工具定义文件），在算 digest 与
零 token 核对之前写好，所以核对的就是这次调用用的那份。`_hook_digest` 只算到分界：frontmatter 与固定说明变了照旧重新注册 project，
协议变了不注册（agy 每次调用都重读 agent 文档，2026-10-16 探针）；驱动身份里的 `tool_agent_document_sha256` 只取固定模板。

Claude Code 的 `system_prompt` 形态（`ClaudeCodeDriverConfig.mcp_opening`，默认）：driver 把协议套进 `claude_system_protocol_v1.md`（只有一句说明与 `## Protocol` 标题，没有逐调用的内容，
所以同一种协议各窗逐字节相同）写进 capsule 的 `input/system-protocol.md`，argv 加 `--append-system-prompt-file <它>` 与
`--exclude-dynamic-system-prompt-sections`。走文件是因为纠错协议约 4 万字节，超过 Windows 约 3.2 万字符的命令行上限。第二个开关把 cwd、
环境、memory 路径从 system prompt 挪进第一条 user 消息：Claude Code 的缓存断点在 system prompt 末尾，不挪的话每窗新建的 cwd 排在协议前面，
下一窗只读得到 CLI 自己 system prompt 的前段。cwd 仍每次新建（工具会话没有 `--safe-mode`，auto-memory 目录由 cwd 派生）。审计里
`input_preparation.system_protocol` 记文件的 sha256 与大小，capsule 的输入账本记一条 `harness_system_protocol`；config 选了这一档而 CLI 的
`--help` 里没有 `--append-system-prompt` 与 `--exclude-dynamic-system-prompt-sections` 时，这个 driver 的每次调用（capsule 调用也算）
都在 `isolation.system_protocol` 记 `unsupported_by_cli`（2.1.302 的 help 只把文件版写作
`--append-system-prompt[-file]`，所以只探不带 `-file` 的那个）。

### 通道上限：超限的回复与送达核对

每家 CLI 能原样送到模型的东西有上限，FineSub 先把能调宽的调宽（Codex 的 `tool_output_token_limit`、Claude Code 的
`MAX_MCP_OUTPUT_TOKENS` 与 `_meta["anthropic/maxResultSizeChars"]`、dsh 关掉 spill-policy），剩下的上限记在 driver config 的
`channel`（`agent/channels.py` 的 `ChannelLimits`），server 经 `FINESUB_MCP_CHANNEL` 拿到：

| driver | 成功 / 错误回复（UTF-8 字节） | 超过时 | 开场上限 |
| --- | --- | --- | --- |
| Codex | 700,000 / 700,000，量整个 CallToolResult 的紧凑 JSON（Codex 截在 `tool_output_token_limit` × 4 字节） | `read_spill` | 1,044,480 字符：`MAX_USER_INPUT_TEXT_CHARS`（1,048,576，超了整次请求被拒）减去留给提示参数与 `<stdin>` 外框的 4,096 |
| Claude Code、CodeBuddy | 400,000 / 10,000（`isError` 文本过约 1.1 万字符被砍中间） | `read_spill` | Claude Code 400,000 字符（2.1.302 + Haiku 5.5 实测一条 40 万字符的 stdin 整条送到）；CodeBuddy 150,000 字符（它的 stdin 没测过，取盖得住生产一窗的数） |
| agy | 3,600 / 3,600（实测 ≈4 KiB 以上被换成文件路径） | 写成份，`view_file` 读（每份 ≤ 40,000 字节 / 700 行，一次读得完） | 30,000 UTF-16 单元（整条命令行，Windows 上限 32,767） |
| dsh | 不限（spill-policy 关着，量不到） | — | 不限 |

- **每个回复**（成功的与 `isError` 的）在 server 的 `_call` 出口过 `fit_reply`：带一个回复 id（成功回复是 payload 的 `reply_id`，错误回复是文本开头的 `[reply r-…]`）；放得下原样发，
  放不下写进 `<assignment>/control/spill/<回复的请求 id>/part-NN.md` 与 `meta.json`。整个回复由一个通用渲染器写成份（每个字段一行 `@@ 路径 · 类型 · 大小`，
  字符串原样），模型拿到的桩留着控制字段（`status`、`task_id`、`message`、修复轮数、`owed_blocks`、`required_blocks`、`knowledge_read_rev`、`result_digest`），列出份
  （太长时写「目录 + 份名范围」）并带一句读法；控制字段自己就放不下时（agy 的 3,600 字节下一长串 `owed_blocks`），桩只留 `status` 与 `task_id`，其余都在份里（`fragment_reply_spill_files_v1.md` / `fragment_reply_spill_tool_v1.md`；有 `batch` 的 CLI 不能并行调用、`batch` 又不跑 `read_spill`，读法换成 `fragment_reply_spill_tool_serial_v1.md` 的一次一份）。重放同一个请求 id 回同一个桩、不重写文件。
  外置的必读块在外置那一刻按 push 记账；提示只出现在外置的那一次回复里，协议与外壳里没有有条件的话，各窗相同的前缀不变。
- **agy 的正文**走同一个写份的函数（`write_document_parts`）：工具会话的开场（见[档位与形态](#1-档位与形态)）与 capsule 调用的消息（`input/task/part-NN.md`，
  每条消息一行 `@@ message N · role`，修复轮接在最后：整份修复只接校验错误，补丁修复接上一份答案与修复说明（说明里含错误）；`input/messages.json` 只留作审计），argv 点名全部份，第一步连同媒体并行读完（`agy_first_step_v1.md`）。
- **Codex 直调**：带 MCP server 的调用多一个 `--config features.code_mode.direct_only_tool_namespaces=["mcp__finesub"]`，finesub 的工具成为顶层工具，
  不经 code mode 的 JS `exec`（exec 里的回复截在 4 万字节、砍中间）。它不在 `_ALLOWED_CONFIG_OVERRIDE_KEYS` 里（那是管配置给的 override 的，放进去就能被设成空）。
  每个 driver 第一次工具会话前带着它跑一次 `codex features list`：Codex 不认这个键（改名或删了，code mode 的配置拒收认不出的键）就报 `LocalAgentUnavailableError` 点名这个设置，
  不退回 code mode；`non_prefixed_mcp_tool_names` 开着时告警（工具名没了 `mcp__` 前缀，命名空间对不上）。
- **`read_spill`** 只读本 task 的份，而且只在本 session 仍持有这个 task 的租约（同一个 worker、同一代、没到期、task 还在做）时，读一份也给租约续期：租约失效或被接走就回「重新调 `next_task`」。
- **送达核对**（`agent/delivery.py`，记进 execution attempt 的 `delivery`）：frame log 每行带 server session，发出的一侧是这个 session 的每个回复（回复 id、发出的字节、
  Codex 调用来源：`_meta.callId` 是 `call_…` 为直调、`exec-…` 为 exec 里调的、外置的份）；收到的一侧在 driver 已有的记录里找回复 id：模型那边的上下文（`model`：Claude Code、CodeBuddy 的
  stream-json，agy 的 brain transcript，Codex 非 ephemeral 调用的 rollout，dsh 的会话日志）或只是 CLI 收到的副本（`cli`：Codex 的 `--json`）。带着 id 的那段不短于发出的才算送达
  （JSON 回复按 server 的写法重新量，CLI 重新序列化、加框架不会被判成没送到；砍中间的截断因此也抓得到；带 Codex `truncated output` 标记的记没送到；
  带着几个回复又被 Codex 标了截断的，几个都记没送到；没标截断、分不出各自多少的记 `unknown`）。外置的回复要份也读到才算送达：`read_spill` 的每一份都送达（回了错误的那一次不算读到），或者每个份文件都被文件工具打开过
  （记录里有一个字符串正好是那个路径，即文件工具的参数；列出路径的提示不算）。agy 点名给模型的任务份（capsule 的 `input/task/`、工具会话开场的份）同样核对有没有被打开。
  只有 `model` 级能判「没送到」，`cli` 与 `unknown` 只记；exit-on-accept 扣下的 `accepted` 记 `held`。模型的回复撞上输出上限（Claude Code / CodeBuddy 的
  `output token maximum` 错误、agy 的 cut-off 续写）每次调用都数，capsule 调用也算。按 driver 往 stderr 告警一次：有回复没送到或份没读、点名的任务份没打开、
  Codex 在 exec 里调了 finesub 工具（直调设置没生效）、撞上输出上限。
- **dev 开关**：`FINESUB_MCP_REPLY_LIMIT="成功,错误"` 压低两个回复上限、逼出外置（验收与单测用；低于约 1 KB 时 `read_spill` 自己的回复也会外置、连锁下去，生产的上限都在 10 KB 以上，碰不到）；`FINESUB_CODEX_FORK_ROLLOUT=1` 让 Codex 的 fork 子会话不带 `--ephemeral`、留下 rollout，
  送达核对因此有 `model` 级证据（生产里子会话照旧 ephemeral，只能按调用来源判直调有没有生效）。

`agent-clean --all-domains` 与 `uninstall --purge-big-data` 会按目录归属删掉 agy 为已删 domain 登记的
project 记录（目录不存在、记录不可解析一律跳过）。

## 7. 未做与闸门

- **D 步不做**（删 capsule 输入路径、`accepts_repair_context`、`repair_in_messages`、会话 handle 缓存；
  owner 2026-10-01 撤回）：`api` 档、`resume` 档与带媒体的调用只能走 capsule，它也是长期保留的全重放基线。
  理由与重开条件见 [`llm_design_notes.md`「工具协议 D 步：不删 capsule 输入路径」](llm_design_notes.md#工具协议-d-步不删-capsule-输入路径owner-2026-10-01-撤回)。
- **数据面迁移：本批次明确暂缓，作为独立项目**（设计在维护者的本地归档里，量级与 B
  相当）。今天工具会话的 payload 仍是 harness 组好的整段 prompt；runtime 内修复轮也仍不逐轮重算
  输入上限。不是只补几个 MCP 动词：爆炸半径横跨 research / fast / query round / correction
  serial+parallel、`session_contract.py`、prompt/replay、token budget 与 L1/L2 resume identity。

  独立项目按以下顺序做，禁止夹进控制面补丁：

  1. 先定 pull-aware L1 身份与 replay 格式：静态资源 digest 集合在调用前可知，实际 pull 序列在调用
     后才知道，必须先解决 lookup/commit 两阶段口径；否则 checkpoint 会把不同输入误认成同一次；
  2. 加 context-pack index / 段落读取与 knowledge index / entry query；关键词预匹配只落 shadow
     事件、不产建议（node 设计稿 v8 收敛：必读块只剩 `kb_index`，原 `suggested_refs` 必读块取消），
     先以 shadow/dual-read 对照现有整包注入；
  3. 把上一轮输出+校验错误改成带 attempt 序号的必读修复块；随后才删除 research / fast /
     query / correction 五处 `keep_entries` 契约与 serial/parallel transfer state；
  4. 最后替换 `_window_input_hash`：只纳入规划期可知的 knowledge snapshot/index + task 静态块
     digest，实际 pull 顺序与 attempt 修复块不得让已提交窗口随机失效。

  知识库那一半已经落地（现行为见 [`knowledge.md`](knowledge.md)「知识库定位与结构」一节）：
  `generation_rev` pinned read 回答上面第 1 条（pull 不改变身份），`kb_*` 工具与 shadow 匹配
  对应第 2 条；`keep_entries` 的删除只覆盖 agent 会话形态，REST 会话长期保留注入数据面。

  启动条件：当前控制面开关达到默认启用资格，或 owner 明确开独立数据面工作流。验收至少包括三家
  production-size canary、pull/replay 确定性、serial/parallel/fast/none/native 组合回归、漏读静态块
  fail-closed，以及质量/成本 A/B。未满足前，它是后续形态，不是当前 Agent 控制面的交付缺口。
  **轮次要压住**（owner 2026-10-01）：改成按需拉取后，每一段内容都可能多占一轮请求，正好与
  [`llm_followups.md`](llm_followups.md)「工具会话的轮次精简」的方向相反。真做时遵从那里的思路：拉取要能批量（一次请求
  取多个块或段），每一轮都要有理由；验收里加一条「每窗请求数不高于迁移前」。
- **agy 文件保底 adapter**：触发条件（免全局配置路径全失败）没出现，不先建。
- **已过闸门**：Codex `auto` 放行 open-world `web_search`；Claude Code `--tools ""` 真机确认只留下
  MCP 工具，已替换易陈旧的内置工具 denylist；Claude Haiku 真实 177 段、8 分 53 秒单窗最终
  `accepted`，无校验重试。
- **原「跨重连 exactly-once」闸门已收口（2026-08-22，见 [§2 生产工具表（唯一真相）](#2-生产工具表唯一真相)「内容幂等」）**：响应丢失后，没有一家 CLI
  保证跨重连复用原 JSON-RPC id（Codex 起 fresh CLI，Claude 在同一 CLI 内重启 MCP 且 id 归零，agy
  报 transient），且生产 canary 的首会话曾在生成完整答案后未 `submit`、靠一次 premature-stop 重试才
  成功。

  **owner 2026-08-22：这条闸门量错了对象。** 传输层 exactly-once 既做
  不到也不必要——durable 状态本来就是权威的，harness 侧「重启一个新 agent 从 checkpoint 接手」
  已经是现状（首次 premature stop → 持原 lease `reset_conversation` + 新 CLI；二次失败/耗尽 →
  `retire_task` CAS 重入队 + fresh CLI）。残余模糊只在 agent 自己那一侧：它重投时手上没有 durable
  状态。而**代价是有界的**——会重复的只有 `submit`（`next_task` 本就「claim 或 resume」，
  `record_pull` 是集合语义），后果是修复预算多记一次，不是产物丢失。

  三步收口都已实施，都不需要 CLI 厂商配合：(1) `submit` 去重键换成内容指纹
  `H(task, lease, input_hash, candidate)`；(2) server 启动对账；(3) `request_id` 降为审计字段。
  实施时多补了一条：修复预算原本是 server **进程内**计数，Claude 重启 MCP 子进程即归零，现与
  `submit_count` 一起落在 task 行。语义从「transport exactly-once（做不到）」变成「at-least-once +
  按内容 effectively-once」；「死不改口」的会话由独立的 `submit_count` 上限退役。

  canary 首会话「生成完答案却不 submit」是**另一件事**（模型行为，与 id 无关），仍由
  premature-stop 重试兜住，代价是重新生成一遍。
- 工具协议**文本专用**：带媒体部件的调用报错、留在 capsule 路径。

## 8. 测试口径与实测读数

- 测试模型：Codex → `gpt-6-luna`，Claude Code → Haiku；agy 可真跑整链，另两家按需。
- 真子进程测试覆盖「accepted 后宽限回收、原始流保留」与「accepted 后立刻结束、照干净成功删 capsule、按逐请求用量补全」；`test_llm_agent_tool_session*.py`、
  `test_llm_agent_task_runtime_v4.py` 是契约测试。
- 一次性缓存读数（一条工具会话 4–5 次往返，非 A/B）：Claude Haiku 73% prompt 走缓存读、Codex luna
  87%、agy ≈57–61%——多轮工具会话天然吃前缀缓存，不需要额外机制。首轮推块后 Claude 从 5 次工具
  调用降到 2 次。
