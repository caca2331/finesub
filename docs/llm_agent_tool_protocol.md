# Agent 工具化协议：让 agent 调工具，而不是收表格

**状态**：方向已定，**实施前有四项待定**（§5），未实施。spike 已完成（2026-08-17，
Claude Code 与 agy 实测，Codex 未测）。

本文是「把 agent 后端从单向投递改成工具调用」这件事的唯一入口：为什么要改、目标形态、
三家 CLI 各自的实测代价、**动手前必须先解决的四件事**、以及实施顺序。Agent 后端的现行契约仍以
[`llm_local_agent.md`](llm_local_agent.md) 为准，本文只讲**要改成什么**。

---

## 1. 背景：今天根本没有交互

生产路径上，agent 与 harness 之间**一次往返都没有**。它不是「没有工具的实时交互」，
而是一次进程、推进去、捞出来：

| 环节 | 今天的实现 |
| --- | --- |
| 取 task | 没有取。argv 末尾挂一句固定指令（`AGENT_TASK_PROMPT_STDIN_ONLY`），真正的消息作为一整块 JSON 从 stdin 灌进去 |
| 提交 | 没有提交。**最终 assistant 消息就是答案**；harness 解析事件流取出最后一段文本，再由 **harness 自己**写进 `staging/result.txt`（`local_agent.py`，`write_atomic(capsule.staging_result_path, ...)`） |
| 修复 | 没有会话内修复。进程退出 → harness 校验 → 组一次新调用 → **全新进程**，上一轮答案作为 stdin JSON 里的一个字段 |

旁边那个 capsule 目录（`input/messages.json`、`input/previous-output.txt`、
`input/validation-errors.txt`）是**取证，不是通道**：Claude Code 在 completion 下工具集为空，
连 `Read` 都没有，读不了那些文件。只有 `AGENT_TASK_PROMPT_READABLE_CAPSULE` 变体会告诉
读得了的 driver 文件存在。

所以整个协议是：**argv + stdin + 最终消息 + 退出码**。

这个形状不是权衡出来的，是**第一个后端决定的**——最早接的是 `codex exec`，一个 one-shot
批处理 CLI，后来两家照着它做。仓库里没有任何地方论证过这套文件协议更好。

### 1.1 由此长出来的赘生物

「harness 必须在发车前决定 agent 将会看到的一切」这一条，直接生出了下面每一处：

| 现象 | 根因 |
| --- | --- |
| `accepts_repair_context()`（逐 driver 的策略开关） | agent 没有任何办法开口要上一轮输出，只能由 harness 提前替它决定 |
| `supports_session_reuse` 去 grep 各家帮助文本 | 我们自己没有通道，想让第 N 次尝试连续，唯一杠杆是借厂商的 `--resume` |
| `repair_in_messages = repair_enabled and endpoint.backend != "local_agent"` | API 后端把修复当额外聊天轮，agent 后端当 stdin 载荷，于是组装层必须知道后端是谁 |
| `client._run_local_agent` 的 handle 穿线与 `(tier, model, chain)` 缓存 | 同一件事再上一层 |
| agy 的修复长期是盲重掷 | 它拒绝「新会话读自己上一轮输出」这个形态，而那是当前协议唯一能提供的形态 |

工具化之后这些**全部可删**。会话形态随之变成真正正交的一个轴：只决定「一个进程处理几个
task」和「provider 对话要不要复用吃缓存」，不再决定工作是怎么送进去的。

### 1.2 一个此前判断错误的前提

2026-08-16 曾判断「把生产接到 task runtime 需要把校验器下沉进 client，是跨层搬家」。
**这是错的**，也是当时选窄路径（会话内修复）的主要理由。runtime 早就留好了缝：

```python
Validator = Callable[[candidate, manifest], ValidationResult]
AgentTaskRuntime(validators={"correction-csv": ...})   # 调用方注册
task.spec.validator_id                                  # 每个 task 点名用哪个
```

且 `agent_task_runtime.py` 会拒绝 agent task 点名未知 validator。纠错阶段按 id 注册自己的
校验器即可，**校验逻辑一步都不用离开 `stages/correction/`**。

---

## 2. 目标形态

| 层 | 变化 |
| --- | --- |
| 协议（`AgentTaskRuntime`） | **不动**。assignment/task/lease/generation、fencing、WAL、`submit → repairable/accepted/blocked`、conversation lineage、知识快照都已就位并有 contract 测试 |
| 传输 | **新增** harness 自己的 MCP server，按 assignment + worker 限定作用域，暴露 `next_task` / `read_context` / `submit` / `knowledge.*` / `web.*` |
| driver | 收缩**输入摆放与结果扒取**：不再摆 `input/previous-output.txt`，不再从事件流里扒最终消息，不再有 `accepts_repair_context`。隔离参数、probe、事件解析、usage 提取、失败分类、进程树回收与超时**全部照旧** |
| 调用方（`client.complete`） | 签名不变。内部建单 task assignment、注册纠错 validator、跑 worker、取 accepted artifact |

### 2.1 最大的收益是「能拉」，不是「能修」

修复轮只是最容易量的那块。真正的代价是：**agent 无法索取，所以一切必须在发车前塞进 prompt。**

harness 为此长出了一整套注入预算机器——知识词条的三个上限（`KB_WINDOW_TOTAL_ENTRIES` 等）、
r1 那一整轮「先让模型挑 12 条词条」、三级 token 计数（`token_budget` / `token_truncate` /
`chunking`）、context pack 的整包注入与截断。**这些机械大半是「推」这个约束的产物，不是问题
本身固有的。**

目标形态在 [`llm_followups.md`](llm_followups.md)「Agent 长驻 worker 重构」里已有原话：
「Agent 完整读取知识库，只按 ref 读取 context pack，**不再反复注入筛选后的知识正文**」。
若成立，选词条轮可能整轮消失——而它现在既花 token、又占一次调用、还派生出「选漏了怎么办」
这一类至今未决的实验（见 `llm_followups.md`「none/native 是否需要逐窗选词条轮」）。

估算本方案价值时**不要只按「省掉修复轮重发整窗的 token」来算**。

**协议与传输分开**是本方案的核心：模块化的收益来自统一**动词**（一套 `next_task` /
`read_context` / `submit`、一个 validator 缝、一个修复循环）。这些动词走 MCP stdio 还是走
子进程，是一层很薄的按厂商适配。这比今天的分叉薄得多——今天 capsule 文件与 control CLI
表达的是**不同的能力**。

---

## 3. 为什么是 MCP，不是「放行一个接口脚本」

两条路都能让 agent 主动调用。差别在**能力**与**过滤**：

- MCP：agent 从头到尾**没有 exec 能力**。guard 写错也变不出代码执行，因为没有可达的执行工具。
- 脚本：先把 `Bash` / `run_command` 交给它，再用模式过滤。guard 是唯一那道墙。

失败模式不对称：MCP 的 guard 出错 → agent 用坏参数调我们的工具 → runtime 校验（JSON schema、
lease 栅栏、`input_hash`）→ **有界**；脚本的 guard 出错 → 任意命令、当前用户身份 → **无界**。
而这里的输入按构造不可信（任意字幕 + 抓来的网页），headless 又无人在环。

**最锋利的一条是 submit 的载荷没处放**。通过 shell 传候选答案只有两条路：当命令行参数
（则必须允许带通配的规则，且要押在厂商「匹配前有没有按 `;` `&&` `|` 切开」的实现细节上），
或让 agent 写文件再由脚本读（则要给它写权限）。MCP 完全绕开：载荷是结构化调用上一个带
schema 的字段，既不过 shell，也不需要写权限。

**但脚本路线不是没有道理**，两点要记在案：它不需要造新传输（`finesub agent-task` 已存在并
注册在命令表里），且 agy 的 `command()` 匹配比通常的 shell 前缀 glob 严得多——
逐 whitespace token 按锚定正则 `^(?:pattern)$` 求值（见 §4.2）。如果 Codex 也缺少按调用的
MCP 配置，三家里两家要额外注册，uniformity 的账要重算，届时应重估脚本路线。

---

## 4. 实测：三家 CLI 的代价

这个仓库对三家 CLI 的每条结论都是实测而非推断，本节同此。复现方式见 §8。

### 4.1 Claude Code（v2.1.231）——最干净

生产 argv 加 `--mcp-config <file>` 即可，`--strict-mcp-config` 本来就在 argv 里，语义正是
「只用 `--mcp-config` 给的，忽略其它一切 MCP 配置」。实测：

```text
init.tools       = ["mcp__finesub__submit_probe"]      ← 恰好一个，我们的
init.mcp_servers = [{"name":"finesub","status":"connected"}]
tool_use         → mcp__finesub__submit_probe
assistant        → 'ZG9uZQ-7731'                        ← 工具返回值里那个猜不到的 token
server saw       = initialize → notifications/initialized → tools/list → tools/call
```

也就是说 agent 手里**恰好一个工具，连 `Read` 都没有**——比今天 capsule 路线的暴露面还小。

两个必须处理的条件：

1. **MCP 工具必须显式进 `--allowed-tools`**。不放行时工具在 `init.tools` 里可见，但调用被
   权限拦下，`tools/call` 从未到达 server。白名单显式、可审计，是好性质。
2. **`--safe-mode` 必须去掉**——它明确关掉 MCP servers（加回去后 `mcp_servers=[]`、
   `tools=[]`、server 根本没启动）。代价比预想的小，逐字段对比只差两处：

   | 字段 | 无 safe-mode | 有 safe-mode |
   | --- | --- | --- |
   | `skills` / `plugins` / `slash_commands` | `[]` | `[]` |
   | `memory_paths` | **出现** | 不出现 |
   | `agents` | 6 条 | 5 条 |

   即 skills/plugins/slash_commands 靠 `--setting-sources ""` 已经挡住，不依赖 `--safe-mode`。
   多出来的 `memory_paths` 是会被自动读进上下文的记忆目录，属内容注入面，但它**从 cwd 派生**
   （`~/.claude/projects/<cwd-slug>/memory/`），而 cwd 本就由 driver 控制、capsule 本就一次性
   ——每 episode 一个新 cwd，该目录即为空。

   **未验证**：hooks。init 事件不报它，`--setting-sources ""` 理论上覆盖。

### 4.2 agy（Antigravity CLI）——需要一行全局权限

| | 作用域 | 实测 |
| --- | --- | --- |
| MCP server 声明 | **按 project**：`.agents/mcp_config.json` | 全局 `mcp_config.json` 保持 0 字节时，server 仍被拉起并走完握手 |
| 调用权限 | **只能全局**：`~/.gemini/antigravity-cli/settings.json` 的 `permissions.allow` | 官方文档明写权限三列表在 global settings |

`.agents/` 正是 driver 已经在写的目录（hooks.json / guard 脚本 / agent.md），所以 server 声明
不增加新的落点。全局足迹只剩**一行**：`mcp(finesub/submit)`。它的性质可接受——一次性注册而
非每次运行改动、内容恒定（两个实例写同一行不冲突）、粒度精确到单 server 单工具、用户可审计。
另有 `mcp(server/*)` 与 `mcp(*)` 两档更宽的，我们只用最窄那档。

三条必须知道的：

1. **PreToolUse hook 只能拒，不能批。** 实测一个发 `{"decision":"allow"}` 的 guard，调用照样得到
   `User denied permission for mcp(finesub/submit_probe)`。权限是 hook **之上**的一层。
2. **工具面不收缩。** 与 Claude Code 不同，agy 恒定宣告全部约 55 个工具（含 `run_command`、
   `write_to_file`、`browser_*`、`invoke_subagent`），MCP 经由通用的 `call_mcp_tool` 派发。
   **隔离仍然 100% 依赖 PreToolUse hook**，与今天相同，没有变好。
3. **权限匹配规则**（官方文档）：优先级 `Deny > Ask > Allow`；`command()` 逐 whitespace token
   按锚定正则 `^(?:pattern)$` 求值；项目目录内的文件读写默认放行（Workspaces are Auto-Allowed）。
   server 条目支持 `disabledTools` 进一步收窄暴露面。

**全局权限缺失时必须 fail closed**：与 driver 现有的 readiness 判据一致——发车前查不到该
allow 规则就不把这条 fact 加进生产 target，而不是跑到一半才发现工具调不动。

agy 还会在标准 MCP 握手之前发非标准的 `server/discover`，以及 `notifications/roots/list_changed`。
生产 server 必须对未知方法返回 JSON-RPC 错误而不是崩溃。

### 4.3 Codex——未测

配额耗尽，押后。预期可行（该 CLI 有 MCP 支持），但**在实测之前不写进结论**。它是唯一还能
改变架构选型的变量：若它也缺少按调用的 MCP 配置，见 §3 末段。

### 4.4 依赖

手写 JSON-RPC（`initialize` / `notifications/initialized` / `tools/list` / `tools/call`）即可满足
三家，**不需要引入 MCP SDK**。生产实现应保持零新依赖。

---

## 5. 实施前必须解决的四件事

spike 证明了「可达」，没有证明「可实施」。下面四条在动手之前必须有答案，前两条尤其——
它们能改变形态，不是收尾细节。

### 5.1 MCP server 的进程归属

按 MCP 的机制，**server 是 agent CLI 依 `mcp_config` 拉起的子进程，不是 harness 的子进程**。
所以它怎么触达 harness 那边的 `AgentTaskRuntime`？

好消息是这条能成立：runtime 本来就是**落盘且多进程安全**的——`control/index.json` 是唯一
reader 入口、状态不可变 + WAL、`holding_lock` 守护、全调用带 lease generation 栅栏。一个独立
进程打开同一个 root 就能服务，不需要另造 IPC。

**待定**：server 从哪里得知该打开哪个 root（见 5.2），以及它以什么身份取 lease——它代表的是
那个 agent worker，还是一个代理身份。后者影响 fencing 的语义。

### 5.2 并行窗口下的作用域（**agy 上是硬冲突**）

`parallel_windows` 默认 **4**：四个窗口并发 = 四个 CLI = 四个 server 实例，每个都得知道自己
服务哪个 assignment/worker/root。

- **Claude Code 没问题**：`--mcp-config` 是按调用给的文件，每次写一份不同的、带 `env` 的配置
  即可（`env` / `cwd` 都是受支持的 server 条目属性）。
- **agy 不行**：`.agents/mcp_config.json` 是**按 project 的静态文件**，同一 project 下四个并发
  调用共用一份，没有 per-invocation 的注入点。

三条候选出路，都未验证：每个并发窗口一个独立 project（则 project 注册变成热路径上的动作，
而它现在是一次性的）；server 从 cwd 反推身份（agy 的 cwd 是 driver 控制的）；或 agy 档在
`continuity=parallel` 下暂时不走 MCP。**选哪条会改变 agy 的实施形态，必须先定。**

### 5.3 终止契约

有了 `submit()` 之后，「这次调用结束了」由什么判定？三种情况都要有答案：正常提交后退出；
提交后继续说话；**退出但从未提交**。第三种 runtime 已有 `premature_stop_attempts` 的概念，
但当前 driver 的完成判据是「进程退出 + 最终消息非空」，两者必须合并成一套，否则会出现
「runtime 认为 task 还在 leased，而进程已经没了」的悬挂。

### 5.4 取证迁移

今天的审计资产是 capsule：`input/messages.json`、`staging/result.txt`、
`events/agent-events.jsonl`，[`llm_local_agent.md`](llm_local_agent.md) §11 明确要求它们保留。
若 §6 的 B 步把 capsule 缩到只剩 prompt，这些的替代品必须先定——runtime 侧已有 artifact 与
WAL，但**产物形状与保留期不同**（capsule 成功即删、失败留存等 `agent-clean`）。不要在迁移中
静默丢掉取证面。

---

## 6. 实施顺序

**A. 生产走 task runtime（仍用现有 capsule 传输）** —— 即 [`llm_local_agent.md`](llm_local_agent.md)
§12 第 3 步。注册 validator、建单 task assignment、`HeadlessTaskWorker` 跑。行为中性、可测。

> **A 的主要工作是一个设计决定，不是接线**：repair 的记账。现行契约是「修复轮不算独立
> attempt，`--max-retries-per-window` 记账、checkpoint 身份、artifact schema 全不变」
> （见 [`llm_harness_behavior.md`](llm_harness_behavior.md)「重试与拼接」），而 runtime 有自己的
> `max_repair_attempts`（默认 5）。两套预算必须先合并成一套。
>
> **「行为中性」需要论证，不能假定**：三层恢复身份（`PROMPT_VERSION` / `WINDOW_INVALIDATION_INPUTS`
> / 路由 digest）里，重试记账换了位置会不会让已提交窗口失效？A 的验收里必须包含「同素材
> resume 仍命中已提交窗口」这一条。

**B. Claude Code 一家切 MCP** —— 闸门最干净、基线已实测为空。agent 真的调 `submit()`，
capsule 缩到只剩 prompt。driver 侧另需两件：去掉 `--safe-mode` 并补 per-episode cwd、
把 MCP 工具名加进 `--allowed-tools`。

> **B 必须带一个逐 driver 的回退开关**，而不是等 D 一次性硬切。它动的是每一个纠错窗口，
> 失败模式是整个 run 死；capsule 传输在 D 之前不得删除。

**C. Codex（先补 spike）与 agy。** agy 另需一次性的全局权限注册，应挂在装机/首次运行流程上
并对用户可见。

**D. 删除**：capsule 输入路径、`accepts_repair_context`、`repair_in_messages`、
`client._run_local_agent` 与会话 handle 缓存、以及 `supports_session_reuse` 作为修复闸门的角色
（它退回成纯缓存优化）。

**A 单独做完并不兑现本文的主张**——A 之后 agent 仍然不调工具，只是修复循环搬进了 runtime。
最小的有意义单位是 **A + B**。

---

## 7. 未决与风险

- **Codex 未测**（§4.3），且它可能翻转选型。
- **Claude Code 的 hooks 维度未验证**（§4.1）。
- **agy 的隔离没有因此变好**：工具面不收缩，边界仍只有 hook（§4.2）。工具化对 agy 的收益是
  协议统一，不是暴露面缩小。
- **模型对自己工具面的自述不可信**：实测中 `init.tools = []` 时模型仍声称「我有 file/search/shell
  工具」。权威是 `init.tools` 与 server 侧日志，**不要用「问 agent 有什么工具」来验证隔离**。
- **排期判断（2026-08-17）**：A 单独在一个发版窗口内可行；**A + B 很紧**；A–D 不可能。
  且本方案动的是纠错主路径，而桌面端的更新演练**覆盖不到它**——风险要靠别的方式兜。
- 本文所有性能/质量收益均**未测**。工具化省掉的是修复轮重发整窗的 token，实际幅度要在真实
  运行上量（同素材同窗口，工具化 vs 今天的 attempt 数与墙钟）。

---

## 8. 复现

spike 用两个文件：一个零依赖的最小 MCP stdio server（对未知方法返回 -32601，并把每一帧记进
日志，以便区分「客户端从未询问」与「我们答错了」），一个用**生产 argv** 驱动 CLI 的 runner。
两者未入库（属一次性验证）。要重跑时的要点：

- prompt 必须走 stdin 或放在所有变参 flag 之后——`--mcp-config`、`--allowed-tools`、
  `--disallowed-tools` 都是变参，会吞掉后面的位置参数（driver 里那些 flag 逐一 comma-join
  就是为了这个）。
- 判据取三处，缺一不可：`system.init` 的 `tools`/`mcp_servers`、server 侧收到的方法序列、
  以及模型是否复述出**只可能来自工具返回值**的一个不可猜 token。
- agy 需要先注册 project 并显式 `--project <id>`，否则 `.agents/` 下的 hook 与 mcp_config 都不加载。
- 动到 `~/.gemini/` 下任何文件前先备份，跑完还原（用户的 `mcp_config.json` 原为 0 字节、
  `settings.json` 原无 `permissions` 键）。
