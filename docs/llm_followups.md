# LLM Harness 后续实验与未实施工作

本文只记录仍需验证或尚未实施的工作。现行 runtime 行为以
[`llm_harness_behavior.md`](llm_harness_behavior.md) 为准，prompt 结构以
[`llm_prompts.md`](llm_prompts.md) 为准；已经完成的开关化、示例 builder 与并行执行
不在这里重复。

## 状态总览

| 项目 | 状态 | 当前行为 |
| --- | --- | --- |
| P6：开关组合与输出系数标定 | 待实验 | 未标定组合可运行但打印 warning；efficiency 与 video 沿用旧系数并提示需重标定 |
| none/native 的逐窗选词条价值 | 暂缓；纳入独立数据面迁移 | 当前不保留逐窗选词条轮；会话级 r1 一次选定并全程透传。数据面项目改成「索引必读 + 自主 query」后，「挑词条」动作本身消失，本行随之作废。该项目的启动条件、阶段与验收见 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §7 |
| P7d：parallel 并发上限标定 | 待真实运行 | `--parallel-windows` 默认 4 |
| P8：超长素材分块调查 | 未实施，先做质量 A/B | 研究 transcript 仍作为一个整体；调用前硬查 194,000 token |
| fast 会话内部预算未按组收缩 | 待触发 | 包络放不下融合窗时闸门按名拒绝；注入上限仍是绝对值 |
| 自定义 endpoint 的媒体支持 | 按计划推迟 | 文本方言声明媒体能力即报错；纯文本模型靠 `--correction-media text` 用在音视频素材上 |
| Agent 工具化协议（agent 调工具取 task/提交） | **A、B、C 步已实施，三家 CLI 整链真机通过，Claude 生产单窗通过；2026-08-22 起两个过渡开关已删、`per-window` 在报 MCP 的 driver 上默认即工具会话；D 未做** | open-world search 与 Claude 精确工具集闸门已过；跨重连 exactly-once 闸门未过，故不翻默认——**owner 2026-08-22 判定该闸门量错了对象，已改为「submit 按内容指纹去重 + 预算落盘 + server 启动对账」（同日实施），见 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §2「内容幂等」**。现行形态、三家接线、审计包与未做项见 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md)。两档重试由 runtime 内第一档修复 + harness 外层替换共同实现 |
| Agent 长驻 worker 架构 | runtime 已完整，生产调用点已接（工具会话，由档位派生） | `agent-task-v4`（v3 + 拉取台账/`retire_task`/去重指纹）：多 worker 分桶、fencing/WAL、固定知识快照、conversational CLI、task/assignment 两档 headless worker 与 retrieval 三态账本（local 硬预算 / native 软记录）均已落地。driver 级 `max_parallel`、`conversation_ttl_seconds` 与 stall watchdog（默认关，先收集 `max_event_gap_seconds`）也已落地。默认保持关闭是已验收决策；剩余架构项是动态 driver。见 [`llm_local_agent.md`](llm_local_agent.md) §3–§12 |
| **agy `view_file` 读取边界** | 已实施并真机验证 | 原生 `--sandbox` 无效；生产 driver 显式绑定受控 project，并在每次发车前验证 deny-by-default `PreToolUse` hook。已实测拒绝 cwd 外路径、junction 逃逸与非白名单工具，hook 漂移 fail closed。见 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) §4 |
| **单 agent session 连续处理多 harness session（G-B）** | 已实施；**agy 上实测复用净亏，默认保持 task** | assignment scope 的 checkpoint identity 已按准入门 D 的 C 案收敛：只含 harness 自知的 durable digests 与窗内 repair history；provider epoch/handle/turn lineage 仅作可丢的运行记录。三家 driver 各自的 resume 已接，handle 丢失/漂移或超 TTL 由 `reset_conversation` 递增 epoch 后全重放。2026-08-14 agy A/B（n=5+噪声基线，按 agy 自己的 `gen_metadata` 逐次账本重算）：复用贵 **46%**、线性增长、墙钟无优势、几乎零缓存命中——见 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) §3.1（注意 `result` 事件的 token 是**会话累计**口径，第一版据此得出的 3.3× 已作废）。**成因已查明**（§3.2 顶部有全貌表）：agy **能**缓存，写入门槛约 1.6 万 token、写后隔 1–2 次请求可读；会话内 resume（含跨进程）继承，**跨 session 不继承**。生产尺寸纠错窗（前缀 19.6k）实测过门槛并命中 16.3k，但过门槛**不保证**命中。此前全部小任务测量都落在不缓存区间，**46% 那个数只对小任务成立**。**待办：用生产尺寸窗口重测复用经济账**（协议见 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) §3.5：三家各两臂 + 噪声基线、n≥5、判据先写死；生产路径的实验开关已接线——`agent_session_mode=resume`，2026-08-19）；门槛只框到 10.7k–16.8k，未精确定位。**Claude Code 有反向的 n=1 信号**（`llm_local_agent_experiments.md` §3.4：Haiku resume 命中整段前缀，三轮便宜 36%；fresh 每轮白写 6.6k 缓存），但不足以改默认，需补 n≥5 正式 A/B；**Codex 未测**，跑它时注意 cache write 恒 0 是显示缺陷、cache read 才准 |
| **agy（Antigravity）多模态 agent** | A1–A5 已实施 | 3.7 Flash 与 Opus 4.6 两个生产 target（分属两个额度池）、driver、受控 project/hook、单帧音频容器、0.25fps 视频、高分辨率规划包络、`agy` 预设与惰性 API 上传均已落地并真机 smoke。剩余是质量/成本持续标定，不再是接入阻塞。见 [`llm_local_agent.md`](llm_local_agent.md) §2 与 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) 全篇 |
| **agy 原生搜索的 URL provenance** | 打通了，但证据薄一档 | `search_web` + `read_url_content` 已由第二个 project 授权，`--retrieval native` 可落到 `local-agy-native-gemini-3_7-flash`，真机验证过（见 [`llm_local_agent_agy.md`](llm_local_agent_agy.md) §5）。**剩下的缺口是 agy 不报来源 URL**：整条事件流零 http，只有查询词，所以 `search_events[].urls` 恒为空。Codex/Claude 两家有逐 call 的完整 URL provenance，agy 只能证明查了什么、不能证明看了哪些页。想补齐得看 agy 是否愿意在 `tool_info` 里带结果，或改走 `read_url_content` 逐页取（会显著变慢且改变检索语义）。做 native/local 检索质量对照时必须把这条不对称算进去。 |
| 预设携带 difficulty/retrieval/knowledge/continuity 的默认值 | 记一笔，暂不做 | 预设当前只绑「格子 → 模型组」，其余开关全部来自 CLI/`config.toml` 顶层。让预设带上这几个开关的默认值是合理的（"agy 档顺便把 retrieval 调成 local"），但要先想清楚它与 CLI 显式值的优先级、以及它算不算 routing digest 的一部分（会不会作废 checkpoint）。owner 2026-08-14：以后闲了再做 |
| Agent 六阶段实施的四道准入门 | durable core 已实现并有 contract 测试 | 全调用 fencing、队列终止/WAL、固定 commit/tree 知识快照，以及仅由 harness-known state 构造的 conversation identity 均已落代码；多 worker 扩展仍须保持同一契约。见 [`llm_local_agent.md`](llm_local_agent.md) §4/§7/§8 与 §12 第 0 步 |
| **解析 Gemini 的 groundingMetadata** | 已实施（2026-08-15） | `retrieval=native` 走 gemini_rest 时，应答里的 `groundingMetadata`（模型跑过的 query + 接地页面 URI）现在被解析成检索 ledger 的同一形状，挂在应答的那次 execution attempt 上。**每次调用一行、不按 query 拆**：Gemini 只说跑了哪些 query、哪些页面接地了答案，从不说哪个页面回答了哪个 query。**因此「native 不可审计」这条结论要重写**——它此前有一半是我们没读。Gemma4 搜索代理与这里共用同一组读取函数（`web_search.gemini_grounding_*`）。agy 仍然零 URL，见上一行 |
| **桌面端 `TaskRequest` 的开关取值对不上 LLM 层** | 已修（2026-08-17 改 `models.py`；**shell.py 那半漏到 2026-08-24 才发现**） | `llm_difficulty` 换成 LLM 层的 `quality/intermediate/efficiency`，默认 `quality`；旧词表在**读入侧**由 `LLMDifficulty` 别名自带的 `BeforeValidator` 映射过来，否则改名会把用户的历史静默清空。**2026-08-24 补记**：那次改名漏了 `finesub_bootstrap/shell.py` 的 `_recorded_request`，它的取值表一直停在 `high/med/minimum`，于是 CLI 跑的 `efficiency` 被记成 `high`、回读折算成 `quality`——桌面重试跑的是另一档难度。`--llm-media` 的缺省也一并错成 `video`（管线是 `audio`）。两处已改，并补了「省略开关时按管线缺省记录」的用例；这是改名类改动**筛选面比改写面窄**的又一例。`knowledge` 的写死 `"update"` **保持不动**——它是有意的产品决定。详见下方同名小节 |
| **validation 失败改为修复轮** | 已实施（2026-08-15）；**会话内修复已接线（2026-08-17）**；**两档重试已实施（2026-08-19，harness 层）** | 同窗重试带上一轮输出 + 每条校验错误：无状态端点收 assistant/user 两轮，agent 复用同一会话续问。契约与三处例外见 [`llm_harness_behavior.md`](llm_harness_behavior.md)「重试与拼接」。一个窗口的整条重试链共用一个 `repair_session_key`，`client.complete` 据此以 `session_scope=assignment` 续用会话——agy 因此不再走 declined 分支。**这不是长驻 worker 的接线**，跨窗复用默认关闭（`agent_session_mode=resume` 实验开关除外），旋钮读取点见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.1。两档重试（`--max-retries-per-window` 会话内修复 + `--max-replacements-per-window` 最大替换次数，总数取乘积）已落在 `attempts.py`；runtime 侧对齐仍挂 A 步。详见下方同名小节 |
| **纠错窗校验放过「几乎全 discard」的输出** | 未做（2026-08-22 canary 发现） | 一行 `sub` + 其余全部 `discard|<id>` 的 CSV 通过了结构校验，成品只剩一条；应加「discard 比例 / 覆盖率」门槛，让看不到正文的 agent 输出在 submit 处被打回而不是变成成品 |
| 完整知识只读 + 联网工具的外泄通道 | P2，记录不做 | 已加 prompt 层内容边界 + extract URL 只能选已展示过的（query 无接收端故不设限）——`retrieval=local` 的 ledger 把这条做成了硬门（fetch 的 URL 必须来自本 task 已完成的 search）；多 Agent headless 上线时重评 |
| task report 的 per-provider token 汇总 | 已实施（2026-08-15） | `task-report.md` 的「Provider Token Totals」按 **(provider tier, model)** 汇总调用数与 uncached/cached input、visible/thinking output。tier 从 `route_decision` 反查答题候选，因为同一模型在免费档与付费档是两笔账。与「Session Token Totals」互不推导：一个会话可能跨档 fallback，一个档服务多个会话 |
| **Agent 会话模式四档的行为/性能/质量对照** | 四档全部接线（2026-08-22），对照观察未做 | 四档 `api`/`per-window`（默认）/`resume`（实验开关）/`pseudo-conversational`（一次 run 一条长驻 CLI）；任务组级、difficulty 向上继承，见 [`llm_local_agent.md`](llm_local_agent.md) §12.1 与 §14.2。已有的只有 agy 小任务复用净亏 + 生产窗能命中缓存、以及 Claude Code 的 n=1 正向信号；**默认值在有数据前不动**，重测协议见 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) §3.5 |
| **Agent 会话档位收敛与 pseudo/conversational 接线** | **已完成（2026-08-22）**，真机验收通过（`llm_local_agent_experiments.md` §3.6/§3.7） | 现行契约在 owner 文档（`llm_local_agent.md` §11/§12.1，`llm_agent_tool_protocol.md` §2/§5/§7，`manual/model-routing.md`「会话档位」）。真正剩下的两条——`resume` 在工具会话下未定义（连带 handle 缓存并入注册表）、长会话簿记的二次增长——见下方同名小节；worker 能力对账 owner 明确不做 |
| Agent cache ping | 明确暂缓 | 先测同类型连续 task 与 compact/重连后的 cached input/墙钟；task heartbeat 不承担 cache 保活。见 [`llm_local_agent.md`](llm_local_agent.md) §13 |
| Agent 单任务模式 | 明确暂缓 | 主路径为长驻 worker；当前 per-session completion 只作迁移兼容，接口保留 one-shot transport。见 [`llm_local_agent.md`](llm_local_agent.md) §13 |
| Agent 直接编辑知识库 | 明确暂缓 | 当前只走 proposal/update；预留隔离 worktree 的 `staged-edit` 策略。见 [`llm_local_agent.md`](llm_local_agent.md) §13 |

## 桌面端 `TaskRequest` 的开关取值对不上 LLM 层

Reviewer 于 2026-08-15 指出，核实后比描述的更严重。

**取值词表曾经是退役预设时代的**：`desktop/backend/common/models.py` 里
`llm_difficulty: Literal["high", "med", "minimum"] = "high"`（两处，请求体与默认体各一），
而 `finesub.llm.routing.profiles.DIFFICULTY` 是 `("quality", "intermediate", "efficiency")`。
`desktop/backend/worker/main.py` 把它**原样直传** `run_pipeline`，中间没有任何翻译。

后果不是「行为不一致」，是**直接失败**：

```python
>>> resolve_profile("audio", "local", "high")
ValueError: Unknown difficulty 'high'; expected one of ('quality', 'intermediate', 'efficiency')
```

也就是说桌面端只要 `--stage` 走到 `translated-srt`/`final-srt`，就会在 profile 解析处抛错。
ASR-only 的运行不受影响（LLM 阶段不启动，这个值没人读），所以问题一直没暴露。

**顺带一条**：`TaskRequest.knowledge` 默认写死 `"update"`，绕过了
`resolve_knowledge_switch` 的 None 解析。它本身是有意的产品决定（注释写明「默认写库让后续任务
变好」），但意味着桌面端拿不到「efficiency 自动解析为 none」那条规则——修好 difficulty 词表
之后，`--llm-difficulty efficiency` + `knowledge=update` 会撞上 LLM 层的硬报错。两处要一起想。

**已修（2026-08-17）**：两个模型的 `llm_difficulty` 都改用 `LLMDifficulty`
（`quality/intermediate/efficiency`，`TaskRequest` 默认 `quality`），前端三处写死的
`"high"` 与 `types.ts` 的 union 同步。

**读入侧保留旧词表的映射,这不是疏忽**。`jobs/history.py:83` 用
`JobSnapshot.model_validate` 解每条记录，外面是 `except Exception: continue`——「一条读不懂
的记录不该让用户丢掉其余历史」。而 `JobSnapshot.request` 的类型就是 `TaskRequest`，盘上每
一条改名之前的记录都带着 `llm_difficulty: "high"`。**只换 Literal 会让用户升级后打开应用看到
一个空白的任务列表，且没有任何报错**。所以 `LLMDifficulty` 是个 `Annotated[..., BeforeValidator]`
别名，把三个退役词映射过去；写入侧不再产生旧词，等盘上不可能还有这种记录时整块删掉即可。
放在类型别名上而不是各模型各加一个 validator，是为了让以后新增的模型自动继承。

这是 CLAUDE.md「不留向后兼容」的既定例外：`migrations/__init__.py` 开篇写明个人数据
（知识库、API key、任务历史）无法重新生成，是唯一需要搬运而不是重跑的东西。

**Reviewer 建议里没有采纳的一条**：把 `TaskRequest.knowledge` 改成 `| None = None` 交给
`resolve_knowledge_switch`。核实后那会**静默降级产品行为**——`pipeline.py:149` 的未设值解析
是 `"none" if efficiency else "collect"`，**不是 `"update"`**，于是桌面端会从「写知识库」变成
「只采集不写」，与同一份文档里记着的产品决定相反。`knowledge` 维持写死 `"update"`。

因此 `efficiency + update` 的硬冲突仍然存在，但**今天到不了**：前端不暴露 difficulty 选择
（三处写死一个值）。等哪天 UI 真的把它做成可选项，再在那时加护栏。

守卫测试：`test_the_difficulty_is_the_word_the_llm_layer_actually_accepts`（对着
`profiles.DIFFICULTY` 断言，不复述字面量）与 `test_history_written_before_the_rename_still_loads`。
桌面套件在 `desktop/backend/tests`，用 `.venv-desktop` 跑，不进根套件。

## validation 失败改为修复轮（而不是盲重掷）

**动机**：2026-08-15 的 605 条素材（BV1ojjc6MEAs）窗口 0001 连续 5 次失败、第 6 次才过，
六次里**没有一次是内容质量问题**，全是输出契约的机械违规，而且正确内容每次都已经在响应里
（详见 [`prompt-iterate.md`](prompt-iterate.md) §5 的案例）。「删掉第 99 行」「补上
id 47」「把 32 挪到 33 前面」都是模型一眼能改的，盲重掷却烧掉 47 分钟和整个免费档日额度，
最后还以一个与真因无关的 `ProviderUnavailableError` 崩掉整个 run。

**已实施**（2026-08-15）。行为契约写在
[`llm_harness_behavior.md`](llm_harness_behavior.md)「重试与拼接」，不在这里重复。当时三个
待定设计点的结论：

1. **agy 必须复用同一 session** —— 按此实现：`AgyLocalAgentDriver.accepts_repair_context`
   只在 `session_scope=assignment` 且有 conversation handle 时为真，否则该次重试保持盲掷
   并记 `declined_by_driver`。复用会话时上一轮输出已在上下文里，prompt 只指向
   `validation-errors.txt`。
   注意 §15.5.1 那条「agy 复用净亏 46%」**不适用于这里**：那次 A/B 测的是**互相独立的任务**
   之间复用，修复轮是同一份内容的追问，前缀天然命中且不需要重发正文，是完全不同的形状。
2. **修复轮不算独立 attempt** —— 它就是原来那次重试，只是 prompt 里多了东西。
   `max_retries_per_window` 的记账、checkpoint 身份、artifact schema 全部不变，也因此不需要
   动 PROMPT_VERSION（attempt 0 的 prompt 一字未改）。
3. **agy 读不到 capsule 里没被点名的文件** —— `_argv` 现在在检测到
   `input/validation-errors.txt` 时把绝对路径写进 prompt。

**会话内修复已接线（2026-08-17）**。此前生产调用点一律逐调用新会话，agy 因此**永远走
declined 分支**、实际仍是盲重试。现在 `attempts.py` 给每个窗口传一个
`repair_session_key`（`correction-<chunk_id>`），`client.complete` 用它在这条重试链内续用
同一个 agent 会话：第 0 次尝试开会话并记下 handle，之后每次修复以
`session_scope=assignment` 带着 handle 回到同一会话——上一轮输出已在上下文里，只递
validation errors。

三条边界，别读岔：

1. **不是长驻 worker 的接线**。durable task runtime 仍未进生产调用点；`agent_session_mode`
   旋钮自 2026-08-19 起由这条窄路读取（四档，见
   [`llm_local_agent.md`](llm_local_agent.md) §12.1.1），出厂默认 `per-window` 正是本条行为。
2. **跨窗复用默认仍然关闭**。key 按窗口取，新窗口一定开新会话；显式设 `resume` 才跨窗续用
   （实验开关）。agy 那个「复用净亏 46%」测的是
   **互相独立的任务**之间复用，与本条无关——同一份内容的追问前缀天然命中。
3. **能力不足的 driver 不受影响**。`session_scope=assignment` 在 `supports_session_reuse=false`
   的 driver 上是**发车前硬失败**而非降级，所以先探能力，探不到就保持全重放。会话变冷
   （TTL、compact、CLI 被杀）时回落一次全重放，即改动前的行为。

**没做、也有意不做的**：会话不可复用时「只给错误、不给上一轮输出」的廉价折中。它便宜
（几十 token）且可能对付得了机械违规，但那已经是 prompt 强化而不是修复轮，属于下面这条
判断的范围，要做得先 A/B。

**prompt 侧只能缓解，不作为主解**（owner 2026-08-15）：`<void>` 撤回机制的措辞可以再强调，
但历史上对这类「模型知道规则却漏执行」的强化已经做到尽头，收益递减。真解是让它看见错误。

**还没有的是效果数据**：这次改动的收益要在真实运行上量（同素材同窗口，修复轮 vs 盲重掷的
attempt 数与墙钟），目前只有单元测试保证信息确实送到了。

### 两档重试：会话内修复，超了就换个 agent 接手（owner 定，2026-08-19；**同日已在 harness 重试语义层实施**）

**实施状态（2026-08-19）**：落在 `attempts.py` 的 `run_window_attempts`——一条按乘积展开的
线性 attempt 序列，链边界处丢弃修复上下文（agent 因链首无修复上下文而开新会话，无状态端点即
盲重掷），`correction_window_retry` artifact 多一个 `replacement` 字段标记边界。旋钮
`--max-retries-per-window`（第一档，默认 5，帮助文本已改写语义）与
`--max-replacements-per-window`（第二档，默认 **1**）暴露在 `pipeline.py` /
`correction_translation.py` / `batch.py` 三个 argparse 与 `batch.py` 的 opts 白名单；桌面
`TaskRequest` 今天**连第一档都不携带**（全走默认），所以第二档也未加——若将来把任一档加进
TaskRequest，两档必须同批。逐轮重路由天然保留（每轮仍走 `client.complete` 的候选循环），
第一档耗尽换人时的重路由因此自动成立。A 步已把第一档接进每次调用新建的 single-task
assignment：runtime 返回 `repair_exhausted` 后，`attempts.py` 直接越过本链剩余 attempt，下一次
外层调用就是 fresh assignment 上的第二档替换。当前形态不借用 `blocked_requeues`，也不需要在
`_give_up` 前额外插一次 `reset_conversation`。compact 预测触发只与未来跨 task 长驻会话有关，未做。

一个窗口现在有**两个**计数器：`--max-retries-per-window` 管同一条链内的修复，
`--max-replacements-per-window` 管 fresh replacement。旧实现只有前者，退化会话会吃完全部预算；
两档拆分正是为了让链内预算耗尽后能丢掉那条会话与修复历史重新发车。

**定下来的形状是两档**：

| 档 | 计什么 | 超了怎么办 |
| --- | --- | --- |
| 第一档 | **同一个 agent 会话内**的修复次数 | 退役这条会话，换一个 fresh agent 接手同一个 task（带完整上下文重放，不带那条坏会话的历史） |
| 第二档 | **最大替换次数**（换了几个 agent） | 放弃该窗口，走今天的放弃路径 |

**两档在没有会话的形态上同样成立，只是"退役"退的不是会话**（2026-08-19 复审更正：本节初稿写
「两档只适用于有会话的档位、`api` 天然只有一层」，与本节下面「在无状态端点上两档同样成立」
自相矛盾，也与实施不符——`attempts.py` 无条件按乘积展开，与后端类型和
[`llm_local_agent.md`](llm_local_agent.md) §12.1 的档位都无关）。**第一档从来不是"会话内"独有
的东西**：它真正的定义是**带修复上下文**（上一轮输出 + 逐条校验错误）的重试，这在无状态端点上
照样成立——那里它表现为追加的 assistant/user 两轮。有会话时它**额外**享受"续同一条会话"这个
更强的形态，如此而已。第二档同理：丢掉修复上下文，有会话就换一条新会话，没有就是盲重掷。

**唯一真正退化的是 agy**：它 decline 掉不在自己会话里的修复上下文
（`accepts_repair_context`），所以在 `api` 档上它的第一档确实空转成盲重掷。那是**一家 driver
的限制**，不是 `api` 这一档的定义——Codex / Claude Code / 任何 REST 后端在 `api` 档下的第一档
都是真修复轮。

runtime 的 `reset_conversation` 仍负责 transport 失败、TTL 过期等机械原因下的会话退役与全重放；
这些原因不消耗 harness 的替换预算。修复预算耗尽走上面的 `repair_exhausted` 出口，语义与机械退役
分账。预测会话将超过 compact 阈值时主动退役仍只是长驻 `resume` / conversational 的候选优化，
在正式会话收益 A/B 之前不实施。

**旋钮形态：两个独立旋钮，总调用数是乘积**（owner 定 2026-08-19）。理由是**两档的单价差一个
量级**，所以"最坏情况 `(档1+1) × (档2+1)` 次调用"这个数严重高估真实成本：

- **第一档便宜**——但要说准它便宜在哪。**不是**"只发 delta"：现行 assignment 传输的所谓 delta
  （`_assignment_delta_messages`）每轮仍重发 `<session_protocol>` + `<run_context>` + 整个 task
  manifest，修复轮也走这条路；它相对全重放**只省下 bootstrap**。上一轮输出从来不在 messages 里
  （走 driver 的 `previous_output=` 形参，agy 的 `_argv` 因此只点名 errors 文件），所以它不算
  delta 的节省。第一档真正的便宜来自：前缀落在同一条会话里、可能被缓存命中，以及不必重开会话。**缓存那一半的证据目前很薄**——[`llm_local_agent_experiments.md`](llm_local_agent_experiments.md)
  §3.4 的 Claude Code 是 **n=1**，而 §3.1 的 agy 那组几乎零命中；把它当通则会高估第一档的便宜。
  即便如此，第一档仍明显比第二档便宜，因为——
- **第二档是真花钱的**：退役会话意味着完整上下文重放，新会话从零写缓存。

  > 重发本身是**要拆掉的形态**，不是要接受的成本：见
  > [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §7——protocol、知识库、research
  > 全局背景与主 payload 都改成 agent 自取之后，"每轮重发多少"这个决定不复存在。所以第一档的
  > 单价在目标形态下只会更低，两档的比价不会因此翻转。

因此第二档保持"最大替换次数"的字面意思，作为一等旋钮暴露，不从总预算导出。

**实施结果与剩余暴露面**：

1. `--max-retries-per-window` 已明确表示第一档，默认 5；`--max-replacements-per-window` 表示第二档，
   默认 1，帮助文本与 `CHANGELOG.md` 已同步；
2. 三个 argparse 与 `batch.py` opts 白名单已暴露两档；
3. 桌面 `TaskRequest` 仍未暴露这两个高级旋钮，继续使用默认值。将来若开放，必须同批加入，不能只加
   其中一个。

**与 `blocked_requeues` 的关系已经定清**：输入阻塞的回队列仍归 runtime；修复预算耗尽返回
`repair_exhausted`，由 harness 外层开始 replacement。两类原因不共用计数器，也不会把模型会话退化
误记成输入阻塞。

**在无状态端点上两档同样成立，只是名字不同**：API 后端没有会话，第一档是"带修复上下文的同窗
重试"（追加 assistant/user 两轮），第二档是**丢掉修复上下文的盲重掷**——也就是 2026-08-15
之前的行为。所以这套设计在两种传输上都有一致的读法，不是只为 agent 造的。

**与 agent 工具化协议的关系**：这两档落在 harness 的重试语义里，与 agent 用什么传输无关，
所以 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §1 的 A 步是它的**载体**：runtime 的
`max_repair_attempts` 是第一档，第二档是它外面的 harness replacement 循环。两边已经接通。

## Agent 会话档位收敛与 pseudo/conversational 接线（**已完成** 2026-08-22）

七件事全部实施并真机验收（agy 37f，读数 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md)
§3.6/§3.7）。**现行契约不在本文**，按主题去 owner 文档读：四档形态与传输派生
[`llm_local_agent.md`](llm_local_agent.md) §12.1，pseudo 的会话生命周期 §12.1.3，conversational
§12.1.4，probe 分级 §11，内容幂等与审计包 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md)
§2/§5/§7，用户侧四档 [`manual/model-routing.md`](manual/model-routing.md)「会话档位」。实施前的定向、
七件事的原始论证与逐轮拍板留在本地 `docs/archive/agent_session_tiers_plan.md`。

**已明确不做**：conversational 的 worker 能力申报与逐 task 对账（owner 2026-08-22——那条路上跑的是
用户自己正在用的 agent，能力与质量由用户负责；注册只申报 `kind`）。理由见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.4，别再当待办。

**conversational 的首次真机实测（2026-08-23）已跑通**：单窗 303 源、走生产 validator 通过、harness token 全 0。由此暴露的收尾项（bootstrap 的参数与措辞、`agent-task lint`、清场留墓碑与关键文件落进 artifact、lease TTL、两句反过度思考的 prompt）单独立在
[`conversational-live-test-plan.md`](conversational-live-test-plan.md)，不在本文展开。

**固定窗的 merge/drop 金标准已按时间对齐（2026-08-25 修复）**：金标准按 286 源标注、当前窗 303 源，
`benchmark.py` 曾直接拒绝执行；现在先试精确匹配、失败则按时间把判断搬到新边界上
（`tools/session_replay/alignment.py`），新切分多出来的边界归「未审」两不计分，报告首行打覆盖率
（本窗 223/302 边界）。离线质量打分因此**恢复可用**，但读数是下界，详见
[`prompt-iterate.md`](prompt-iterate.md) §5。

下面两条是这个项目**真正剩下的**。

### 未做：`resume` 档在工具会话下仍未定义

`resume` 今天恒走 capsule 窄路——三家都报 MCP 能力，若按「有 MCP 就工具会话」它会永远退化成
`per-window`，A/B 路径就没了。要把它挪到工具会话，先答两问：(a) 三家的 resume flag 与
per-invocation MCP 配置能否共存，没实测过；(b) 续上的会话里模型已有上窗的必读块，而台账每次
lease 清零，「台账说没读、模型说读过」要么多打几次 `read_context`（浪费但无害），要么需要一个
跨窗台账概念。

连带的一条：`resume` 的 handle 缓存仍挂在 `RoleClient` 上，没并入 run 作用域的会话注册表
（owner 决定 4 要求共用）。两件事一起做才划算。

### 未做：长会话 assignment 的簿记开销随 task 数二次增长

`AgentTaskRuntime` 每次操作都整份重写 durable state，而一条 run 的 task 行只增不减（终态的行也
留着——`_assignment_complete` 与 accept 重放都要读）。pseudo-conversational 把「一条 run 一个
assignment」变成常态之后，这条以前不显眼的性质开始按 O(n²) 记账。

**实测（2026-08-22，本机，每 task 一次 add/claim/pull/submit）**：终态行约 2.7 KB（`spec` 占 1.3 KB），
状态文件线性涨到 600 task 时 1.5 MB；耗时按每 100 task 累计 7.5 → 22.5 → 45 → 76 → 116 → 164 s。

**换算成真实规模，这条今天够不着**：一个 task = 一次 `complete()`，即每窗 1（纠错）+ 每 base
chunk 1（`retrieval=local` 的查询轮）+ 整条 run 约 5–6（两轮调查、搜索 judge、知识更新）。窗口数
约为每小时口语素材 2–4 窗（kaguya60：60 分钟 553 段，压到 4000 token/窗才切出 4 窗，默认 10,000
约 2 窗）。现实区间是 25 task ≈ 1.1 s、50 ≈ 2.6 s、100 ≈ 7.1 s、200 ≈ 22.5 s；600 task 要一个
几十到上百小时的**单个文件**才够得着（batch 每个文件各自开 assignment，不累积）。

不影响正确性——request 表撑满那条**硬失败**已经修掉（accept 的重放答案挪到 task 行、其余按 task
归档，agent 与 external 两条 accept 都是）。要动的话直接上「终态 task 行移出热状态」（落
`tasks/<id>/record.json`、热状态只留小桩）；只瘦身终态行是把同一条曲线右移一点、不改性质；真正
去掉平方项要改成追加日志 + 周期快照，那会动到当前 exactly-once 所依赖的「一次整文件原子写 +
accept 的 WAL」，需要单独一轮设计与复审。触发条件：一条 run 的 task 数常态过三百，或簿记时间在
run 里可见。

## P6：开关组合与输出系数标定

目标是校准 `profiles.py` 的输出系数和实际质量，不是再次调整开关结构。重点包括：

- efficiency：现在强制 basicB、thinking low，旧 text-low 测量条件已经变化；
- `media=video`：媒体成本与窗口几何仍沿用旧标定；可见 reasoning 措辞现已中性化，不再作为分档变量；
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

## none/native 是否需要逐窗选词条轮

> **这个实验有一个会让它整体作废的上位改动**（2026-08-19）：知识库改成「索引必读 + 自主 query」
> 之后（[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) §7（设计见归档 `archive/agent_tool_protocol_plan.md` §2.3.1）），**没有「挑」这个动作
> 了**——本节问的「要不要多一轮来挑词条」也就失去对象，连带 `<keep_entries>` 与三个注入上限
> 一起消失。所以：在工具化协议的 B 步落地之前，本节的重启前提照旧有效；一旦落地，**先确认本节
> 还有没有对象再开工**，不要照着下面的协议跑一个已经不存在的问题。

当前裁决是**不保留**：有索引时由会话级 r1 一次挑词条；没有逐窗请求能力的路径强制
全量透传当前集合。给纠错终稿轮直接加索引已经排除，因为请求到的词条本窗来不及使用、
会挤压最紧张的纠错输入预算，还会扩大输出契约和 replay 面。

### 重新启动实验的前提

以下两项必须同时满足：

1. 知识库词条数显著超过 r1 当前额度 12，使选择真正需要取舍；
2. 至少有一个多窗 local 运行的逐窗查询轮产生非空 `requested_entries`，证明增量能力确实
   被使用过。

此前数据不满足：知识库只有 6 条，而已检查的多窗 local 运行逐窗请求全部为空，所以
会话级 r1 对当时分母的召回在实验前就已经是 100%，继续调用没有信息价值。

### 实验协议

1. 选两份有精修字幕的素材：专名均匀分布一份、关键专名集中在后段一份。
2. 比较 `r1@lightweight` 与 `r1@general_capable`；触及 12 条上限的样本作废。
3. 令 `R` 为会话级 r1 注入集合，`U` 为 local 逐窗实际 `injected_entries` 的并集；报告
   `|R∩U|/|U|`、`R\U`、`U\R` 与 `hit_cap`。
4. 人工判断 `U\R` 中每条是否真的影响精修成品：
   - 两个 r1 臂都漏关键条目：结构问题，再考虑逐窗选词条轮；
   - 只有 lightweight 漏：升级 r1 模型，不增加轮次；
   - 漏项都无实际作用：维持当前无逐窗轮。

可用 `research-context.json.entry_selection` 判断样本是否具备实验资格，无需重新推断当时
的额度与索引规模。

## P7d：parallel 并发上限标定

并行执行、限流器取号、固定词条屏障、drain-then-raise、熔断和 resume 已经实现；这里只
标定安全并发值。媒体调用会钉同一把 key，因此不能把 worker 数当作 key 数横向扩展。

至少比较 `parallel-windows=1/2/4`，在同一素材和同一 key/pool 下记录：

- 纠错阶段墙钟、API 时间之和与实际发车时间线；
- RPM/TPM 等待、429/5xx/timeout、sticky retry 和 endpoint fallback；
- 熔断、拆窗、content-filter ladder、成功缓存数；
- serial 与 parallel 的字幕质量差异，尤其专名一致性和跨窗指代；
- RPD 消耗和同一媒体文件项目钉 key 后的风控表现。

默认值 4 在标定完成前保持保守 warning。不得通过“每个 worker 绑一把 key”提速：媒体文件
项目隔离要求调用固定 key，多把免费 key 同时活跃也违背现有风控策略。

## P8：超长素材分块调查

### 是否值得做

这不是输入溢出补丁：按现有实测，194k 硬上限大约要 10–13 小时素材才会撞到。目标是
验证研究质量是否在更早的长度开始下降，尤其 r1 是否遗漏后半段的检索选题。先做同素材
“全量调查 vs 分块调查”的 A/B；每臂 `n≥5` 并做同配置噪声基线。测不出稳定质量收益就
不实现。

候选软阈值为 30k–60k transcript tokens（约 2.5–5 小时），最终数值必须来自实验。若
实施，新增 `[chunking].research_transcript_max_tokens`，`0` 表示关闭；阈值以下必须沿用
当前单次路径。

### 预期形态

仅首要支持 `retrieval=local`：

1. 按整数个调查窗口分块，绝不切开窗口；各块笔记继续绑定 stable source-id 区间，
   下游纠错无需与调查共用窗口几何。
2. 每块独立执行完整 r1 + search loop + r2；只切 r2 会让同样过长的 r1 继续漏检索选题。
3. `window_contexts` 按互不重叠的 source-id 区间合并；各块 `general_context` 按稳定块 id 合并，
   再用同一份合并结果替换各块全局部分。
4. 合并说明必须告诉纠错模型如何处理不同块对同一专名的冲突：优先证据更充分者，并在
   note 标注。先不增加额外 LLM 归并轮。
5. 合并后的全局块必须有独立 token 上限和 warning；不能套用知识词条 dict 的
   `injection_block_token_limit`。

`retrieval=none` 可退化为每块 r1 选词条后做确定性并集；是否采用需和上一节的逐窗选词条
实验一起判断。`retrieval=native` 暂不进入 P8，因为它的收益机制是模型内搜索，先在 P6
观察长输入退化。

### Resume 与确定性约束

- 每块 research context 单独落盘；r1/r2 checkpoint key 使用块 id，不再写死 `main`。
- 所有块完成后才生成最终 context pack，再开始纠错；不能边调查边纠错。
- 拼合必须是纯函数：块顺序、key 和说明文字固定。这是**可复现性**要求，不再是恢复
  正确性要求——context pack 自 2026-08-12 起不进纠错窗口指纹（它只影响尚未执行的
  窗口），所以非确定拼合不会再让窗口缓存失效，但会让两次运行的注入无从比对。

## `client.py` 的 Files API 上传该拆出去（2026-08-21 押后）

上传（`upload_gemini_file` / `_upload_gemini_file_rest` / `_wait_for_media_tokens` 与 2026-08-21
加的重试 helper）和模型调用没有关系，`client.py` 已过 2400 行。没在加重试时顺手拆，是因为
`UploadedFileRef` 定义在 `client.py` 且被 agent 侧共用：新模块 import 它、`client.py` 再 import
新模块就是循环；要先把 `UploadedFileRef` 挪走，再改 7 个测试文件与
`tools/session_replay/sessions/correction.py` 的导入。单独做，纯搬动一个 commit。

## 模型路由 v2 的遗留（折自已归档的实施方案）

v2 的里程碑 1–5 已实施（事实归 catalog、编排归 config、任务组/模型组/预设、三类 endpoint
的纯文本 adapter、difficulty 三档改名）。下面是当时明确留下的部分，每条都记了**为什么没做**
与**什么条件下再做**——不要从已归档的方案正文推断现状。

### fast 会话内部预算未按组收缩

`fast_session` 内部各上限是写死的绝对值：`FAST_ROUND2_INPUT_RESERVE_TOKENS` = 56k、词条块
28k、证据包 20k、notes 2k。组包络（D13）现已接到 **fast 判定闸门、纠错主循环、知识更新分块**
三处，但这些注入上限本身不随包络收缩。

2026-08-12 只做了**让失败可读**：绑定组的输入包络 ≤ 预留时，闸门直接说「该组放不下融合窗，
走常规多窗流程」，而不是给出负数预算（`--fast on` 仍报错而非静默降级）。

真要支持小上下文模型跑 fast，必须缩的是**注入上限本身**——那是质量旋钮（模型能看到多少
证据），是标定活不是接线活；按比例缩 56k 这个和数只会让闸门放行、第 2 轮照样溢出。

- **触发条件**：真有人要在小上下文模型上跑 fast 时再做。
- **一起核的东西**：`FAST_ROUND2_INPUT_RESERVE_TOKENS` 与它对应的那几个分项上限必须同时
  重算，否则闸门与实际注入会再次脱节。

### 自定义 endpoint 的媒体支持（文本 adapter 之后的「第二刀」）

第一刀只做文本。媒体接线很深：Gemini Files API + `file_ref` 复用（预取线程、`file_ref_seed`、
查询轮与纠错轮共用一份剪辑），而 OpenAI/Anthropic 走 inline base64，限额与音频支持差异大。

现状：catalog 里文本方言（`openai_compat`/`anthropic`）的行**声明媒体能力即报错**——两个文本
传输展平 messages 时会丢掉附件，放行等于静默降质。纯文本强模型用在有音视频素材上的正规路径
是 `--correction-media text`（查询轮继续吃剪辑）。

### Agent 长驻 worker 重构

方向已于 2026-08-12 定案，不再按旧“第 1/2/3 档”划分：

- `headless` 与 `conversational` 共用 `AgentTaskRuntime`、task/goal/lease、工具预算、checkpoint、
  validator 和 `submit → repair → accepted` 协议；
- conversation 可以连续处理多个独立 harness task；模型结束一个 turn 不构成 task 完成；
- Agent 完整读取知识库，只按 ref 读取 context pack，不再反复注入筛选后的知识正文；
- `retrieval=native` 使用 Agent 自带搜索并校验 provenance；`local` 开放 harness-owned 搜索工具并
  由服务端按 API 同口径限额；
- compact/重连从 harness durable `control/index.json` rehydrate；ready task 由 submit 夹带，API
  依赖由 runtime waiter 唤醒；动态 driver 走进程外版本化协议；多 Agent 通过单 lease + generation
  + TTL/heartbeat 调度；
- 当前 Codex per-session completion 保留为迁移兼容，按
  [`llm_local_agent.md`](llm_local_agent.md) §12 的六步顺序逐段替换。

单任务 Agent 不做产品入口，但作为 `session_scope=task` 全重放基线永久保留。仍暂缓的只有 provider
cache ping 与第二种知识写入（隔离 worktree direct edit）。完整授权 prompt、early-stop、compact、
安全和模块边界见 [`llm_local_agent.md`](llm_local_agent.md)。

#### 六阶段实施前的四道准入门（2026-08-12 复审提出）

内容**就地写在 [`llm_local_agent.md`](llm_local_agent.md) 各自章节**，并已列为 §12 的第 0 步——
那份文档是 Agent 接入形态的唯一入口，条件放在这里等于允许按唯一入口实施的人整批漏掉。
这里只留状态摘要：

| 门 | 位置 | 一句话 |
| --- | --- | --- |
| A 全调用 fencing | §4 | 每个有副作用的 control/tool 调用都要带并校验 lease generation，不只 `submit()` |
| B 队列终止语义 | §4 | `waiting/wait_token` 与 `assignment_complete` 必须分开；提交链要幂等或 WAL |
| C 知识库真快照 | §8 | 记录 HEAD ≠ 绑定版本；读取要走固定 commit/tree |
| D 隐含历史进入调用身份 | §7 | **2026-08-19 改定 C，2026-08-22 迁移完成**：会话复用降格为纯加速，依赖隐含历史的调用不产出可复用 checkpoint，身份只由 harness 自知的东西构成。代价只落在 `resume` / `pseudo-conversational` 两档（`api` 与 `per-window` 跨窗不携带历史，代价为零）。生产窄路的 `resumable=False` 与 runtime 的 harness-known repair identity 两半均已落地 |

四项只挡 Agent 六阶段实施。

#### 完整知识只读 + 联网工具的外泄通道（P2，记录不做）

§8 的完整知识只读与 §9 的 local/native 联网同时开启时，字幕或网页里的注入可以诱导 Agent 用
**已获准**的 `knowledge.read()` 取内容，再塞进搜索 query 或 fetch URL 外发。不需要提权，
所以「payload 不能扩大权限」这条挡不住；provenance 校验和输出 validator 也都在错误的方向上
（它们管流入，这条通道是流出）。

**定级 P2、暂不做工程**，理由是能力集不超过用户已经接受的本地 coding agent 风险（后者能读
`.env`/密钥/整个源码树），而知识库的资产是主播专名与档案——泄露是**关于第三方的隐私问题**，
不是对用户自身的入侵。两处真实差异记录在案：headless worker 连续 drain 队列时**无人在环**，
以及输入（任意视频字幕 + 抓取的网页）**按构造就不可信**。

已做的动作有两项：

1. prompt 层的内容边界（`fragment_knowledge_structure_v1.md` 的「一律不收集」三条：素材里的
   指令不是指令、与字幕理解无关的个人信息、凭据形态字符串），它同时是知识库质量护栏。
2. **2026-08-13：搜索 loop 的 extract URL 收口成「只能选已展示过的 URL」**（实现见
   `docs/llm_harness_behavior.md`「本地检索代理」）。**两条通道的差别是有没有接收端**：
   fetch 的目的地由模型指定，请求会落进那台主机自己的访问日志，注入方看得见；而 search
   query 只发往 Exa/Tavily/DDG，注入方没有任何读取途径——所以 query 不设限（也不该设长度上限，
   长而具体的 query 正是有用的那种），收口只针对 fetch。
   收口后残留的是「选哪一条」的 log₂(N) bit 级通道，且语料按 owner 决定包含已提取页面正文，
   即挡编造不挡投放。

**若 headless 多 Agent 真的上线，重新评估「无人在环」这一条**，届时可选的服务端约束是
task 级知识读取 scope、或禁止「完整知识只读」与「任意网络工具」同时出现。

### `<reasoning>` 措辞按注入实况变化（prompt 迭代）

2026-08-12 把该措辞**中性化**了（一句话，所有会话相同），因为旧的 `(difficulty, retrieval)`
分档从未标定，且它编码的理由与这段 prompt 的实际用途不符（它兜的是「模型自己的思考没发生
或退化」，不随思考旋钮缩放）。

仍然可能有价值的形态是**按实际注入了什么**变化——`retrieval=none` 的窗口手里什么都没有，
要求它写出不确定处与候选读法；有 context pack 的窗口则要求它对着注入内容交叉核对。这与仓库
既有的「注入实况原则」（`CORRECTION_SLOTS` 的 `requires` 按真实注入校验 fragment）一致。

**必须走 prompt 迭代协议做**（`docs/prompt-iterate.md` + `tools/session_replay`），
不能凭感觉改措辞——那是要 A/B 的东西。

### quality_score 初值

已由维护者验收（2026-08-12）。此条仅作记录，不再是遗留项。
