# Agent 工具化协议：agent 调工具取 task、读上下文、提交

**状态（2026-08-22）**：A/B/C 步已实施，三家 CLI 小链真机通过，Claude 生产单窗 canary 通过；
**两个过渡开关已撤**，传输由会话档位派生（§1）：出厂默认 `per-window` 在报 MCP 能力的 driver 上
**就是工具会话**。本文是现行规格——
是什么、怎么接、哪里还没做。为什么这样定、七轮复审各改了什么，见归档
`docs/archive/agent_tool_protocol_plan.md`（本地件，不随仓库发布）。
Agent 后端的其余契约仍以 [`llm_local_agent.md`](llm_local_agent.md) 为准。

两条总原则：**不碰用户的全局设置**（agy 的 `settings.json`、Codex 的 `config.toml`、Claude Code 的
用户级 settings 一行不写）；**不为过度保守的安全策略加机制**。

## 1. 档位与形态

配置面只有会话档位 `agent_session_mode`（按 cell 设，见 [`llm_local_agent.md`](llm_local_agent.md)
§12.1）；传输由 **`agent_transports.agent_transport_for(档位, probe, 带不带媒体)`** 派生
（owner 2026-08-22 收敛，A/B/C 步实施期的两个 `[llm]` 过渡键已删）：

| 档 | 传输 | 说明 |
| --- | --- | --- |
| `api` | capsule（`client.py` 窄路，task scope） | 该档的定义：没有上一轮、全重放 |
| `per-window` | probe 报 `supports_mcp_config` → **工具会话**；否则 capsule 窄路 + 每 driver 一次 warning（`agent-transport-capsule`） | 工具会话的原生形态：一次 invocation 内交、被打回、再交。**出厂默认** |
| `resume` | capsule 窄路（跨窗续 handle） | 三家都报 MCP，按「有 MCP 就工具会话」它会永远退化成 `per-window`；工具会话 + resume 未验证 |
| `pseudo-conversational` | 工具会话（一次 run 一条会话，`agent_session_host.py`）；无 MCP **硬失败** | 已接线 2026-08-22，见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.3（本文 §4 有它的完成谓词） |
| 任一档、调用带媒体 part | capsule | 工具协议文本专用 |

工具会话：每次调用建一个单 task assignment（root 在 driver episode 域旁 `assignments/<call-id>`），
agent 经 harness 自己的 MCP server 取 task / 读上下文 / 提交，第一档修复在 runtime 的
`submit → repairable` 循环里跑完（`complete(validator_spec=, max_repair_attempts=)` 把校验器按 id
交进去）；用尽时返回最后被拒输出并打 `repair_exhausted`，`attempts.py` 据此跳到链末，下一次调用即
第二档替换（harness 重路由）。工具会话永远是一次 CLI 调用 = task scope。dev-only 强制某一传输：
环境变量 `FINESUB_AGENT_TRANSPORT=capsule|tool-session`（档位定义性的规则——`api`/`resume`/媒体恒
capsule、pseudo 必须工具会话——不受它影响）。

**身份**：档位进 `routing_identity_digest`（`presets.*.agent_session`），改档位作废 checkpoint；实际
走的传输**不进**身份（owner 显式决定 2026-08-22：同档位下两种传输消费同一 prompt、同一 validator，
产出等价；probe 是这台机器今天的样子不是契约），只记进 route decision trace 的 `agent_transport`。

**validator 跨进程**（`agent_validators.py`）：`submit` 由 CLI 拉起的 server 进程判，闭包过不去，
所以 `complete(validator_spec={"id", "params"})`，server 与 harness 都从 `VALIDATOR_BUILDERS` 按 id
解析；纠错窗口是 `correction-window` + 序列化窗口，variant / tier 进 task metadata。新 validator
必须注册在那张表、参数必须可 JSON 化。

## 2. 生产工具表（唯一真相）

agy 的逐工具 `permissionGrants`、Codex 的 `enabled_tools`、Claude Code 的 `--allowed-tools` 与
readiness 校验都从这张表派生：

| 工具 | 作用 | readOnly | destructive | idempotent | openWorld | 何时暴露 |
| --- | --- | --- | --- | --- | --- | --- |
| `next_task` | 以本 worker claim / 续租；返回 manifest、**首轮直接带放得下的 `protocol` / `payload` 正文**（按 push 记台账；driver 的 `mcp_page_chars`（UTF-8 字节）限内联回复大小，放不下的块标 `read: paged` 留给分页拉取，agy 为 2800、其余 0 = 不限）、剩余修复轮数。pseudo-conversational 会话里没活时**长轮询**（`FINESUB_MCP_WAIT_SECONDS`，默认 25s）后回 `still_waiting`（再问），seal 后回 `assignment_complete`（退出）；预算耗尽的 task 在被 harness 收回前不再发出 | false | false | false | false | 恒在 |
| `read_context(ref, offset=0)` | 按 ref 读资源，**分页**：回 `text` / `offset` / `total_chars` / `next_offset`，页在换行处断；**读到最后一页才记台账**；**只接受 manifest 点名的 ref**，错 ref 的报错列出合法 ref（模型会瞎猜，每猜一次一轮） | false | false | false | false | 恒在 |
| `pull_status` | 本 context 还欠哪些必读块 | true | false | true | false | 恒在 |
| `submit(payload)` | validator → accepted / repairable / retired；**按内容幂等**（同一 context 内重投已判过的答案 = 重放首次判定、不烧修复预算），修复预算与线上 submit 次数都记在 **durable task 行**（server 进程内不计数），用尽 / 超 submit 上限 / 被退役后固定回「停止」 | false | false | false | false | 恒在 |
| `web_search` / `web_fetch` | 经 harness 本地检索代理，runtime 检索账本计费 | true | false | false | true | **暴露**：单 task 会话按 task `retrieval_mode=local`；pseudo-conversational 会话发车即暴露/授权全部六个（三家授权只能在 invocation 时定）。**调用时准入**：server 按当前 task 的 `retrieval_mode` 放行，非 `local` 的 task 调它报错（工具会话下 `native_search` 映射为它，CLI 原生搜索关闭） |

注解**如实**：Codex `auto` 放行只看 destructive / open-world（实测），不伪装只读。`web_search`
（open-world）已实测可过 Codex `auto`。

**request id**（2026-08-22 起对 `submit` 只是审计字段，正确性由下一段的内容指纹承担）：MCP 调用
`H(assignment, worker, server_session_id, server_instance_id, JSON-RPC id)`；`server_session_id`
每次 CLI invocation 由 driver 经 `env` 传入，`server_instance_id` 由每个 MCP server 进程随机生成。后者不可
省：Claude Code 会在同一 CLI 内重启 MCP 子进程并把 JSON-RPC 计数归零。server 按
`request_id → {参数指纹, 回复}` 缓存——同一 live server 内同 id 同
参数返回缓存（修复计数不动两次），同 id 不同参数回 conflict。runtime 侧每个有记录的操作都带输入
指纹：同 id 同指纹 = 重放，不同指纹 = `AssignmentConflictError`；`record_pull` 只存指纹不存正文；
表按 assignment 512 条硬上限。harness 自己的收尾调用（reset / retire，每会话一次、无恢复方）用
`H(server_session_id, 操作名)`，是归档件第 2 节总规则「持久化序号」的显式例外。

**`submit` 的内容幂等与两个计数**（§7 收口，已实施 2026-08-22）：task 行按 lease 持有
`submissions: {H(task, lease, input_hash, candidate) → 首次判定}`、`submit_count`、`repair_attempts`，
三者在新 lease（claim / retire）时清零、`reset_conversation` 保留（同 lease）。每次 `submit`：
同 `request_id` 传输重放 → 两计数都不动；新 id 同指纹 → `submit_count += 1`、回首次判定并标
`replayed`，`repair_attempts` 不动；新指纹 → `submit_count += 1`，validator 判 repairable 再
`repair_attempts += 1`。欠块拒绝既不计预算也不缓存（它的判定取决于此后读了什么，不取决于答案）。
`submit_count > max_repair_attempts + 3`（`EXTRA_SUBMITS_PER_CONTEXT`）→ 同一次落盘内退役，
`protocol_violation=submit_cap`——这是「死不改口」会话的出口。server 的 `repair_rounds_remaining`
与停止回复一律从 `task_record()` 派生，并在启动时对账：本 assignment 已完成即答「已 accepted，请退出」。

## 3. 必读块台账与 submit 门（runtime 协议 v4）

- task 声明 `required_blocks`（`kind` / `digest` / `ref` / `tool`），`@protocol` / `@context` 占位由
  runtime 在物化文档后填 ref 与 digest；身份是 `(kind, digest)`。
- 台账存在 context 里：assignment scope 在 conversation 记录上（reset / retire 整条替换即清零），
  task scope 在 task 行上（每次 lease 清零，`reset_conversation` 也清）。push 与 pull 都算。
- `submit` 前置门：欠块首拒一次列全并点名工具，每 context 一次补读机会（硬编码）；补读后仍欠任
  一块 = 会话不服从协议，**同一次落盘内退役**（`_retire_task_locked`：会话 reset、lease 撤销、重入
  队、`retirements+1`、存 `last_candidate`），返回 `retired`。
- 静态块规则保证的是「模型有机会看过」，不是「上下文里一定有」；compact 后重读靠 `read_context`。

## 4. 完成与终止契约

| 情形 | 契约 |
| --- | --- |
| 完成 | **`accepted` 是唯一完成点**；最终 assistant 文本无产物语义。driver 监督循环接 `completion` 谓词（约每秒查），为真后给 CLI `ACCEPTED_EXIT_GRACE_SECONDS`（15s）自行退出，否则回收进程树。**谓词按会话形态分两种**（第四轮修订 2026-08-22）：单 task 会话传 `task_record().status == "accepted"`；pseudo-conversational 会话传「assignment 已 seal 且全部 task 终态（或已 failed）」，单个 task accepted 不触发回收，CLI 继续 `next_task`；此时不要求最终消息 / 零退出码 / 干净事件流，**原始事件流保留**在 capsule（`raw_events_retained`） |
| driver 在 accepted 之后出错 | 先读 runtime：accepted 就收下，driver 错误降级为 warning |
| 首次 premature stop（未提交即退出） | 持原 lease 调 `reset_conversation`（清台账），同 generation 起一个新 CLI 会话，不计替换；`AGENT_PREMATURE_STOP_RETRIES = 1` |
| 第二次静默退出 / 修复耗尽 / driver 抛错后仍持 lease 且未 accepted | harness 代已退出的 CLI `retire_task`（「harness 不持 lease」的唯一例外），task 重入队、无主；耗尽返回 `repair_exhausted`，抛错照抛 |
| `retire_task` | CAS 用 lease 本身（`worker_id + lease_generation`），`request_id` 幂等；与迟到 `submit` 先落盘者赢；复用同一 worker id、起 fresh CLI |
| 重投已判过的答案 | 同 context 内重放首次判定、不烧预算（§2「内容幂等」）；线上 submit 次数超 `max_repair_attempts + 3` → 退役（`submit_cap`） |
| 反复退役 | 每次退役都把 task 重新入队并**重置预算**，所以退役次数本身要封顶：`MAX_RETIREMENTS_PER_TASK = 2`，第二次退役直接把 task 判 `failed`（host 立刻抛错进第二档），否则不守协议的会话会一直重来到 per-task 期限为止 |

## 5. 审计包

每条工具会话写进 capsule 的 `audit/`——pseudo-conversational 会话一次调用服务多个 task，
所以按 task 分名 `audit-<task_id>/`（随 capsule 保留规则：成功可删、失败留到 `agent-clean`）：
`manifest.json`、`blocks/<kind>.md`（必读块**正文**）、`outcome.json`（最终 durable task record——
reset / retire **之后**重新读的、拉取台账、conversation epoch 与全部 `resets`、`error`）、
`artifact.txt`、`mcp-frames.jsonl`（server 记的每一帧）。先写临时目录再原子 rename；**写失败不删
assignment root**（它此时是唯一证据）并打 warning；driver 抛错路径也写。
「写失败」只指**有 capsule 可写却没写成**：一次干净的调用里 driver 自己已经把 capsule 删了
（那正是上面的保留规则），此时没有 locator、也没有证据要留，assignment root 照删——反过来把每
条成功调用的 root 都留下，是这条规则的反面。

## 6. 各 driver 接线

| driver | 声明 server | 授权 | 事件层 | 实测 |
| --- | --- | --- | --- | --- |
| Claude Code 2.1.231 | `--mcp-config` 内联 JSON（身份走 `env`，`cwd` 未见文档），去 `--safe-mode`（它关 MCP；managed hooks 的残余面按 owner 口径接受），`--strict-mcp-config` 已在 | `--tools ""` 精确移除全部内置工具，再用 `--allowed-tools` 逐个授权 `mcp__finesub__<tool>`；不开 `Read` | `tool_use` 按 `mcp__finesub__*` entitled | 整链通过（Haiku），也发非标准 `server/discover`（答 -32601） |
| Codex 0.147.0 | `--config 'mcp_servers.finesub = {command, args, env, default_tools_approval_mode = "auto", enabled_tools, startup_timeout_sec = 30}'`（`_codex_mcp_server_override`），`--ignore-user-config` 下仍生效 | `enabled_tools` | `mcp_tool_call` 按 `(server, tool)` 判 entitled，`command_execution` 照禁 | 整链通过（gpt-5.6-luna）；`mcp list` 加载检查通过；server 进程在沙箱外 |
| agy 1.1.18 | 第三种 project `.finesub-tool-<slot>`（slot 0..`max_parallel`-1，driver 内信号量、一次调用持一个），**每次 invocation 前**原子重写其 `.agents/mcp_config.json`（身份走 `env`）与 `.agents/view_roots.json`（本次 assignment root，块作为文件交给 `view_file` 读——agy 的 MCP 回复超 ≈4k 字节即外置，见 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) §5） | 只写 project **自己的**记录 `~/.gemini/config/projects/<id>.json` 的 `permissionGrants`（逐工具 `mcp(finesub/<tool>)`，缺记录 fail closed；路径是 agy 的实现细节，只定义在 `finesub_bootstrap/agy_records.py`）；guard 放行 `call_mcp_tool@finesub` 与 `view_roots.json` 所列根之下现有文件的 `view_file`；agent 文档**必须写 `mcpServers: [finesub]`**，只写 `tools` 会空跑 | `call_mcp_tool` 按 `(ServerName, ToolName)` 判 entitled | 单槽与双槽并发都通过；hook 看得见 MCP 调用但授不了权 |
| dsh 0.1.1-rc.2 | `--patch <capsule>/input/dsh-patch.yml`（写成 JSON——JSON 即 YAML，省掉 Windows 路径的转义坑），条目必须用 **`insert:` 列表**：裸条目是按 id 定向覆盖，profile 里没有 mcp-client 可覆盖，patch 引擎只 warn 就跳过——模型没工具而 harness 毫不知情。`serverName` 即 `finesub`，身份走 `env`，`failOnStartupError: true` | 无工具白名单；同一份 patch 按 id 把 plugin **关掉**（比白名单强：工具不注册），`DSH_PERMISSION_MODE=read-only` 兜底。留 `tool-fs`（读侧），`tool-web` 只在 native 轮留 | **无**——headless 只打印最终答案。entitled 判定退化成「不存在的工具不可能被调用」；`observes_tool_events = False` 让 native 轮记 `native_search_unobserved` 而非谎称没搜 | 整链通过：玩具任务，以及 **270 条真实纠错窗口**（v4-flash，782s，三次工具调用、零重试）。一帧 `next_task` 就 55,096 B > 出厂 `maxInlineBytes` 50,000，`spill-policy: {}` 是前提而非保险。读数、限速端点的假阴性、thinking 为何只对自带路由生效，见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.0 |

`agent-clean --all-domains` 与 `uninstall --purge-big-data` 会按目录归属删掉 agy 为已删 domain 登记的
project 记录（目录不存在、记录不可解析一律跳过）。

## 7. 未做与闸门

- **D 步**（删 capsule 输入路径、`accepts_repair_context`、`repair_in_messages`、会话 handle 缓存）：
  等全部 driver 在生产稳定后再做。
- **数据面迁移：本批次明确暂缓，作为独立项目**（归档件第 2.3 节「B 之后的形态」，量级与 B
  相当）。今天工具会话的 payload 仍是 harness 组好的整段 prompt；runtime 内修复轮也仍不逐轮重算
  输入上限。不是只补几个 MCP 动词：爆炸半径横跨 research / fast / query round / correction
  serial+parallel、`session_contract.py`、prompt/replay、token budget 与 L1/L2 resume identity。

  独立项目按以下顺序做，禁止夹进控制面补丁：

  1. 先定 pull-aware L1 身份与 replay 格式：静态资源 digest 集合在调用前可知，实际 pull 序列在调用
     后才知道，必须先解决 lookup/commit 两阶段口径；否则 checkpoint 会把不同输入误认成同一次；
  2. 加 context-pack index / 段落读取与 knowledge index / entry query，关键词预匹配只产
     `suggested_refs`，先以 shadow/dual-read 对照现有整包注入；
  3. 把上一轮输出+校验错误改成带 attempt 序号的必读修复块；随后才删除 research / fast /
     query / correction 五处 `keep_entries` 契约与 serial/parallel transfer state；
  4. 最后替换 `_window_input_hash`：只纳入规划期可知的 knowledge snapshot/index + task 静态块
     digest，实际 pull 顺序与 attempt 修复块不得让已提交窗口随机失效。

  启动条件：当前控制面开关达到默认启用资格，或 owner 明确开独立数据面工作流。验收至少包括三家
  production-size canary、pull/replay 确定性、serial/parallel/fast/none/native 组合回归、漏读静态块
  fail-closed，以及质量/成本 A/B。未满足前，它是后续形态，不是当前 Agent 控制面的交付缺口。
- **agy 文件保底 adapter**：触发条件（免全局配置路径全失败）没出现，不先建。
- **已过闸门**：Codex `auto` 放行 open-world `web_search`；Claude Code `--tools ""` 真机确认只留下
  MCP 工具，已替换易陈旧的内置工具 denylist；Claude Haiku 真实 177 段、8 分 53 秒单窗最终
  `accepted`，无校验重试。
- **原「跨重连 exactly-once」闸门已收口（2026-08-22，见 §2「内容幂等」）**：响应丢失后，没有一家 CLI
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

- 测试模型：Codex → `gpt-5.6-luna`，Claude Code → Haiku；agy 可真跑整链，另两家按需。
- 真子进程测试覆盖「accepted 后宽限回收、原始流保留」；`test_llm_agent_tool_session*.py`、
  `test_llm_agent_task_runtime_v4.py` 是契约测试。
- 一次性缓存读数（一条工具会话 4–5 次往返，非 A/B）：Claude Haiku 73% prompt 走缓存读、Codex luna
  87%、agy ≈57–61%——多轮工具会话天然吃前缀缓存，不需要额外机制。首轮推块后 Claude 从 5 次工具
  调用降到 2 次。
