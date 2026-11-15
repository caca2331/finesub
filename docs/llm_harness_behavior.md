# LLM 纠错翻译 Harness 行为说明

> **前向提示（2026-08-19）**：本文描述的是**今天**的形态。知识库若按
> [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门)（设计在维护者的本地归档里）改成「索引必读 + 词条自主
> query」，这里的预注入名额、`<keep_entries>` 透传链与那组注入上限会一并消失。**不要据此实施
> 新功能**，先确认那个改动落地没有。

状态：实验性实现。默认只生成计划和中文 prompt；真实 API 调用必须显式使用 `--execute`。


> **本文拆出了两块**（2026-08-16）：dev 侧的模型事实/池/路由链/思考档位/限流去了
> [`llm_harness_routing.md`](llm_harness_routing.md)，本地检索代理与背景调查去了
> [`llm_harness_research.md`](llm_harness_research.md)。其余仍在本文。

## 开关轴（--media / --retrieval / --difficulty / --continuity）

正交开关轴取代了原先的严格 6 档 preset（`--route`/`--level` 已退役；设计意图见
`docs/llm_design_notes.md`，未完成的标定见 [`llm_followups.md`](llm_followups.md)，实现在
`src/finesub/llm/routing/profiles.py`）。
**媒体只有一个开关**（2026-08 曾按 model-routing v2 D20 拆成 `--correction-media` / `--planning-media` 两个按任务的开关；
查询轮并入窗口任务后只剩纠错窗读媒体，2026-09-09 合回一个；代码里的字段仍叫 `correction_media`）。

| 轴 | 取值 | 含义 |
| --- | --- | --- |
| `--media` | `text` / `audio` / `video` | **纠错窗**（请求回合与交付轮同一条对话）看到什么；阶梯，video 蕴含 audio；窗口几何（剪辑占的输入 token）、纠错 prompt 的媒体措辞与 fragment 选择读它，输出系数不读 |
| `--retrieval` | `none` / `local` / `native` | `local` = 整套 harness 注入（准备任务的回填 + 每窗请求回合 + 本地搜索代理）；`native` = 模型自带搜索工具；`none` = 无检索 |
| `--difficulty` | `quality` / `intermediate` / `efficiency` | 按**想要什么**命名（2026-08-12 由 high/med/minimum 改名，与 thinking 档位区分开），是定位、不写死模型能力：选该格的 prompt 变体与思考旋钮，预设还可按档位绑不同模型组，纠错任务还按它定走几个 pass。前两档给 capable 模型用，都能走[三个 pass](#三个-pass)：`quality` 默认开说话人裁判、`auto` 下跑 pass 3，`intermediate` 两样都不；`efficiency` 是最省的可用形态：单 pass、只用出厂默认风格、不能开说话人裁判（owner 2026-10-30）。不写时是 `intermediate`（owner 2026-11-08）。它不钉别的开关 |
| `--continuity` | `serial` / `parallel` | 窗口连续性（v75）：`serial`（默认）保留窗口间链式上下文（advice 台账、词条透传链）；`parallel` 放弃它们、并投纠错窗换墙钟——两阶段一屏障（全部请求回合并投 → 会话级词条集一次定死 → 全部纠错窗的交付轮并投 → 按 chunk_id 有序合并），配 `--parallel-windows`（默认 **1**，owner 2026-08-30 由 4 改：任务内并行有质量与 token 效率代价，任务间并行没有——见下方「任务级并行与 agent 槽位预算」；预设可带它的默认值，出厂的 `agy-fast` / `fast` 写 4，只在 parallel 下生效）。prompt 侧 `<previous_advice>`/`<next_advice>`/`<keep_entries>` 整体撤除；失败 drain-then-raise（所有跑完的窗都进缓存后才抛错）、同批 **3 次会话链耗尽**即熔断（计的是**链**不是窗口，2026-08-19；它并不保证一个批次待在日额度以内，为什么这样定见下方「重试与拼接」）；缓存记录带 `continuity` 与去词条核心哈希，**只允许 serial→parallel 方向复用**；key 的用法与 serial 同构（2026-08-19 起并行不再额外钉 key，只有带媒体的调用钉一把，见 [`llm_harness_routing.md`](llm_harness_routing.md)） |

`--knowledge {none,read,update}` 是独立的任务级三态开关，不属于
`TranslationProfile` 的四轴向量；它控制知识输入、反馈采集与任务后更新，详见
[`knowledge.md`](knowledge.md)。

输出系数按开关合成：`c = 3.5 + 1.0(retrieval≠none)`。媒体不进系数（owner 2026-09-30）：剪辑只占输入，
实测输出长度主要随模型与推理强度变化。
**difficulty 不再进系数**（model-routing v2：efficiency 取消 -1.5 减免，text/none/efficiency 的 c 从实测的
2.0 提到 3.5、窗口容量降至 57%）。系数不按开关组合逐个标定，也不在运行时提示（owner 2026-09-30）。
difficulty 在 profile 里只选 prompt 变体 + thinking level；纠错任务另按它定走几个 pass（见[三个 pass](#三个-pass)）。
「开关高于实际可用媒体」是入口处的配置错误（缺 `--audio`/`--video` 直接报错），不做静默降级；
运行期仅剩一级 video→audio 安全网（见「client 层」）。

退役 preset 的换算（仅用于读旧产物；`finesub.llm.routing.profiles.parse_profile_id` 仍能解析这些旧名）：

| 旧 preset | media | retrieval | difficulty | c |
| --- | --- | --- | --- | ---: |
| text-low | text | none | efficiency | 3.5（原实测 2.0，model-routing v2 取消该档减免） |
| text-med | text | none | quality | 3.5 |
| text-high | text | native | quality | 4.5 |
| mm-low | text | local | quality | 4.5 |
| mm-med（2026-11-08 之前两个入口的默认） | audio | local | quality | 4.5 |
| mm-high | video | local | quality | 4.5 |

- 纠错窗归 `correction-mm` / `correction-text` 任务组（由 `correction_media` 选；role 只是产物标签）。**`retrieval=local`** 的定义特征是
  harness 侧外部注入：准备任务与每个纠错窗的请求通道里都有本地搜索（`media=text` 时
  无媒体但保留注入）。**`native`** 不用本地搜索代理：准备任务的交付调用与纠错调用自身开模型的
  搜索工具（Gemini 3 免费层 grounding 429）。**`none`** 两者都没有，准备任务只整理素材与词条。
  逐档的请求通道见下方「准备任务与请求通道」。native search 是**按调用请求的能力**而非角色（D4，v1 的独立 native 链已删）：任务组不变，`client.complete(..., native_search=True)` 在**当前绑定的模型组内**按 target 是否带搜索工具过滤（模型 `supports_native_search` 为真时带它那个 tier 的工具，非 native 调用不发）。出厂预设由付费档接地；免费档 3.x 完全不能联网，唯一能联网的免费模型 2.5 Flash 低于纠错/知识下限，因此只作为打包的 `gemini-native-search` 组供显式绑定。过滤后为空即报错，不做静默降级。
- 预期输出估算 `k × c × csv_tokens`（k 为 `--output-scale`，默认 1.0；调大 k 切出更小窗口）；常规窗口须满足 `≤ 0.9 × 65536 − 5000 = 53982`。
- **insert/插轴已于 v63 全面废弃**（所有路线、所有变体均不再注入或接受 insert 行；校验层视 insert 为结构性错误、同窗重试）。
- 纠错调用 thinking：来自预设级思考旋钮（出厂实况见「LLM thinking effort」的表）。
- **没有会话要求写推理**：纠错窗 v107 起，其它会话（准备任务、知识类、校验与审核）2026-11-08 起，只交规定的块。可见推理块原本是给
  「模型自己的思考没发生或退化时」兜底，但 capable 模型真零思考时思考并没有挪进这一块
  （[`llm_followups.md`](llm_followups.md)「模型回答的退化」），而在 Claude 上明着要求写推理会被以
  `reasoning_extraction` 拒答（准备任务在 Sonnet 5.5 直连 API 上整段被拒，提到 `<reasoning>` 的几处全删掉才正常作答）。
  模型自己写了 `<reasoning>` 也不算错：解析只认要读的块。校验与共享审核原来写在推理块里的核查结论挪进它们打印的 JSON 字段：
  校验判 `unverifiable` 时 `note` 写查过的方向，审核有说法没核实到时 `reason` 末尾写查过的方向。
- `basicA` 交付终稿时先写 `<decisions>` 决定清单（v109）：不超过 8 行结论（专名与称呼定成什么、哪些区间识别不可靠、
  哪些行要丢），重交被退回的输出时先逐条列出这一轮要改的问题和它违反的输出规则；只写结论，不写推导过程，不写
  字幕行草稿。它守两件实测过的事：3.5 Flash Lite 在 low 档思考 token 为 0，没有这一块时整份重问几乎逐字重交同一份错的
  输出，有了它复述规则后一轮就改对（2026-11-05 的 A/B）；弱模型拿计划代替产出（round-46），所以限 8 行、禁止逐行预演。
  缺块不校验、不重试，块的内容不参与解析。

### 会话清单（触发谓词 → 路由链）

一次 run 最多存在 **4 类** LLM 会话（2026-09-09 起的形态：
旧的研究两轮、搜索 loop、每窗查询轮与 fast 融合轮都已删除）。下表的「触发」是代码里的真实谓词而非经验描述；
会话按 **任务组 × difficulty** 选模（v2：预设把格子绑到模型组，未绑格难度回落，见 [`llm_harness_routing.md`「模型事实、池与路由链」](llm_harness_routing.md#模型事实池与路由链)），
thinking 与变体来自格子，输出契约见 `finesub.llm.session_contract.SESSION_CONTRACTS`。

| 会话 | 触发谓词 | 任务组 | 媒体 | 次数 | 输出校验 |
| --- | --- | --- | --- | --- | --- |
| 准备任务（一条对话：请求批 → 回填 → 交付，或直接交付） | 非 fast 恒跑（`_prepare_stage_runs`） | `research`；`retrieval=native` 时按能力过滤组内成员 | 纯文本 | 每 run 1 条对话（1–2 次调用 + 解析重试）；超长素材按块各 1 条 | `prepare_task_request` / `prepare_task_delivery` 契约，按回复里有无 `<context_pack>` 分流 |
| 纠错窗（含请求回合） | 恒跑 | `correction-mm` / `correction-text`（按 `correction_media`） | 按 `correction_media`：`≥audio` 带本窗剪辑（`video` 带 mp4） | 每窗 1 条对话：`retrieval=local` 且策略 batched 时首条消息带请求通道（直接交付即 attempt 0，否则回填后交付），其余档位首条消息就是交付轮；+ 校验重试 | 请求回合 `correction_request` 契约；交付 `output_protocol`（不进 SESSION_CONTRACTS） |
| fast 单窗 | fast 启用 | 同「纠错窗」 | 同上（整段剪辑） | 1 条对话（唯一窗口，同样带请求通道；不跑准备任务） | 同上 |
| 统一知识更新 | `--knowledge update`（且 `--execute`） | `knowledge` | 纯文本 | 每 chunk 1 次（`knowledge-update-chunk<NN>`） | 自有 parser |

会话选模型的三条规则（v2）：

- **任务组由会话（与 media 开关）决定，模型组由预设的格子绑定决定**。角色
  （`audio_multimodal` / `general_capable`）只剩 artifact 标签。`retrieval=native` 是 per-call 能力（D4）：
  在选中的模型组内按 `supports_native_search`（且该 target 声明了搜索工具）过滤，
  **过滤后为空即报错**，不做静默降级，也没有独立的 native 链。出厂预设可用：付费
  3.8 Flash 的 target 直接声明 `google_search`（非 native 调用不发该工具），免费 3.x 不能
  联网，因此纯免费档跑 `--retrieval native` 会在启动校验处失败——要么开付费档，要么显式绑定
  打包的 `gemini-native-search` 组（含免费 2.5 Flash，但它低于纠错/知识下限）。
- **native 检索的出处记录**（2026-08-15）：Gemini 应答里的 `groundingMetadata` 会被解析成
  检索 ledger 的同一形状（`{tool, query, queries, urls}`），挂在应答的那次 execution
  attempt 的 `search_events` 上，因此 **task artifacts** 里能看到模型跑了哪些 query、
  哪些页面接地了答案。**exchange 里看不到**：`## Execution Attempts` 表是固定列的，
  `search_events` 不在其中——三家 agent 的检索事件同样只落 artifacts，这里保持一致。
  **每次调用一行、不按 query 拆**：Gemini 从不说哪个页面回答了哪个
  query，拆开等于替它做一个它没做的归属。三家本地 agent 里 Codex/Claude 有逐 call 的完整
  URL provenance，**agy 只报 query、不报 URL**——做 native/local 对照时这条不对称必须算
  进去，`research._mark_unverified_sources` 也只对 agy 这类零 URL 的后端加
  `verified: false` 标记。
- **prompt 变体来自格子（可被模型组条目覆盖）**。`correction-*` 的 quality、intermediate 格是 capableC、
  efficiency 格是 basicA——旧的「按应答模型档位 + difficulty 上限」推导
  （`effective_tier`）已删除：组内退让不再换 prompt，链内不再有静默降质；
  `retrieval=native` + `difficulty=quality` 也拿 capableC。逐窗缓存记录实际使用的变体名。
- **agent 靠成为模型组的成员参与，policy 只是 backend 闸门**（2026-08-14）。此前 policy 会按
  任务组把 agent 组 prepend 到每个格子上；现在它只能否掉某个组已经列出的 backend，永远不能加上
  一个组没列的。出厂绑定 agent 的预设是 `agy-hybrid`（`agy-capable` / `agy-basic`，免费 API 打头）
  与 `agy`（`agy-only-*`，名单里没有 API 成员）；Codex/Claude
  的 target 仍然声明着，用户列进自己的 `[llm.model_groups]`、或把格子直接绑到 target id 上
  （快速选模型）即可用。带剪辑的纠错窗（含 fast 单窗）仍会在能力过滤时跳过纯文本的
  agent 候选（Codex `supports_audio=false`，agy 的 Claude 同样是纯文本）。

### 准备任务与请求通道（按轴拆分）

准备阶段是**一条逻辑对话**（`stages/prepare_task.py`，方案 §10.3）：首条消息就是完整任务与交付契约，模型要么直接交付
`<context_pack>`（一次调用），要么先提**一批**请求；harness 执行后把结果作为下一条 user 消息回填（模型自己的请求回复作为
assistant 消息放回历史，REST 适配器因此仍只发一段合法历史），第二条回复必须交付——再提请求按解析失败处理、走同一重试预算，
**不会有第二次回填**（§2.2：有界、无依赖猜测）。能力只改**请求通道**，不删任务。下表的「有知识库索引」= `knowledge_index_available()`
= `--knowledge read|update` **且** 索引非空；`--knowledge` 默认为 `read`，所以默认 run 在库非空时走「是」行，空库或 `--knowledge none` 时走「否」行：

| retrieval | 有知识库索引 | 请求通道里有什么 | harness 回填 | 交付 |
| --- | --- | --- | --- | --- |
| `local` | 是 | `search_queries` + `requested_entries`/`keep_entries` | 一轮本地搜索 + 查 KB | `<context_pack>` |
| `local` | 否 | 仅 `search_queries` | 一轮本地搜索 | `<context_pack>` |
| `native` | 是 | 仅 `requested_entries`/`keep_entries` | 查 KB | `<context_pack>`；交付调用自身开原生搜索工具 |
| `native` | 否 | 无请求通道 | — | 一次交付（原生搜索） |
| `none` | 是 | 仅 `requested_entries`/`keep_entries` | 查 KB | `<context_pack>` |
| `none` | 否 | 无请求通道（prompt 要求 `sources` 写 `[]`、自行推断的译名在 `note` / `rule` 标「未核实」，仍未覆盖的疑点进 `uncertainties`） | — | 一次交付 |

`<analysis_notes>` 只在请求回复里要求（≤1500 token，它是回填后交付的起点，不注入纠错窗）；直接交付的回复只要 `<context_pack>`（与授权里的随后几块）。
`--extra-info` 里的 URL（≤8 个，`retrieval=local`）在首条消息前预提取成 `<note_url_extracts>`；备注里命中 index key/alias 的词条预注入（≤8 条）并可被 `keep_entries` 保留，直接交付时同样进种子。

**窗口怎么取缺的输入**由 `task_graph` 一处回答：`window_requests` = 有东西可请求（`retrieval=local` 的 harness 搜索，
**或**知识开启且索引非空），与执行策略无关；`style_pick` = fast 窗口要先选 `--style auto` 的风格（fast、auto、目录里有
`default_style` 之外的候选，由编排层回答，2026-10-01）。`window_flow` 在两者之上给出 `request_channel` = batched 且
（`window_requests` 或 `style_pick`）——harness 的请求回合——与 `agent_fetches` = agentic 且（`window_requests` 或 `style_pick`）
——执行者自己的工具会话。`window_requests` 本身就是「丢掉的词条能不能再要回来」：透传链（`CorrectionRun.window_requests`
直接取它，不从通道布尔量反推——只为选风格开的通道要不回词条）、准备任务的下游额度读它；fast 闸门读「`window_requests` 或
有风格目录」（一律按 batched 形态估、刻意不把路由拉进 planner），运行与 dry-run 读 `window_flow`。

**无请求能力的窗强制保留词条**：`window_requests` 为假时（没有可请求的东西），准备任务与纠错窗
都不再被要求输出 `<keep_entries>`（prompt 里根本没有这个块），harness 直接透传当前全集，透传上限也从
`KB_TRANSFER_MAX_ENTRIES`(8) 提到 `KB_WINDOW_TOTAL_ENTRIES`(12)——给「下一窗再补」留的格子在这条路线上纯属白丢词条。
同理准备任务的请求额度在这些档位下用 12 而非 8（`finesub.llm.prompts.r1_request_cap`）。2026-09-30 起 none/native
有可读知识库时也能请求（见下方「纠错窗口调用形态」），所以这一格与 local 一样是 8 + `<keep_entries>`（owner 2026-09-30）。
⚠ **agentic 是例外，在任何档位上都要 `<keep_entries>`**：harness 没有注入过任何东西可以透传——
执行者是自己用 `kb_read` 读的库——所以不由它点名，后续窗口就什么都拿不到。前提仍是索引可读：
`--knowledge none` 下没有这个块，harness 也不会去解析它（KB 关闭不读正文）。准备任务的 `emits_keep` 因此是
`(window_requests or 准备任务自己是 agentic) and 索引可读`。

⚠ **改这一族的准入与可见集判据之前**，先读这四条（2026-09-09 分支复审留下的）：

1. **复用一个判据就复用整条。** 同一个错在这一族犯了五次：往条件里加一个 `or`（`or agentic`、
   `or run.agent_fetches`），删掉的是它今天为假所靠的隐含前提——比如「KB 关着就没有可见集」。
   扩一个判据之前先问**它今天为假靠的是谁**；**挪一个判断的位置和改它的内容一样危险**
   （第五次是改了它与 fast 早退的先后，`git diff` 里看不出）。
2. **通则不能替代例外条款。** 驳回一条引用了某节计划的复审意见之前，先去读那一节。
3. **修复自己会长出缺陷**，而且集中在新加的机制里：三十条修复里有七条是上一次修复引入的。
   补丁比原代码更值得一条回归测试。
4. **被点名的症状往往只是一格。** 照症状打补丁，会把同一个洞留在没被点名的档位上。

**纠错窗口也是一个任务**（§10.4，形态见下方「纠错窗口调用形态」）：`request_channel` 为真时首条消息就是纠错契约加
「获取缺失输入」一节（`request_prompts.py` 叠在纠错 prompt 之上），模型要么直接交付终稿
（作为 attempt 0 验收，不再调用），要么一批请求；harness 查库（+ local 下一轮搜索）后回填，交付轮与修复轮都在这条对话里。
请求回合跑在**纠错 cell** 上（同一模型组、同一媒体）。`native` 的窗两轮都设 `native_search=True`（与准备任务一致，owner 2026-09-30）。

### 开关组合 → 实际会跑的会话

一次 run 的会话集合只由 fast 与 `--knowledge update` 决定（`task_graph.plan_task_graph`：普通 = `prepare` + `translate`，
fast = 单个 `whole` 的 `translate`，update 追加 `knowledge_update`）；`retrieval` × 有无索引只改各任务的**工具授权**
（请求通道里有什么），media 开关只决定纠错窗挂不挂剪辑，`difficulty`/`continuity` 只改 prompt 与调度，都不增删会话：

| retrieval | 索引 | 准备任务的请求通道 | 纠错窗（batched）的请求通道 |
| --- | --- | --- | --- |
| `local` | 有 | queries + entries | notes + entries + queries |
| `local` | 无 | 仅 queries | notes + queries |
| `native` | 有 | 仅 entries（原生搜索） | notes + entries（原生搜索） |
| `native` | 无 | 无（一次交付，原生搜索） | —（纠错调用自身原生搜索） |
| `none` | 有 | 仅 entries | notes + entries |
| `none` | 无 | 无（一次交付，无外部证据） | — |

`retrieval=none` + `--knowledge none` 下准备任务照跑（旧流程的「整个研究阶段不跑」这一行已不存在）：它只整理素材与用户信息，
`sources` 留空。统一知识更新与本表正交：只看 `--knowledge update`，在任务结束后按 chunk 跑。

执行策略按任务解析（§10.5，`task_graph.resolve_execution_strategy`）：**agentic** 要两个条件同时成立——任务的链全是
`local_agent`，**且这个任务的 agent 调用真的会拿到 harness 工具**（`capabilities.chain_serves_tool_session`）。满足时准备任务与窗口
都不再由 harness 代取，prompt 改为「用授权的工具（`web_search`/`web_fetch`、`kb_*`）自己取、一次回复直接交付」。任一不成立则为
**batched**（上面的请求回合）。⚠ 第二个条件是必要的：传输由**会话档位、是否带媒体、driver 认不认 MCP** 三者派生
（[Agent 工具协议](llm_agent_tool_protocol.md) [§1 档位与形态](llm_agent_tool_protocol.md#1-档位与形态)），三条里任一不成立就走 capsule、那里没有工具——只看 backend 就选 agentic，
会让 harness 收起自己的请求回合、再叫一个没有工具的模型自己去搜，`retrieval=local` 就此静默变成 `none`。所以：
`correction_media != text` 的窗口任务永远是 batched；`api`/`resume` 两个档位永远是 batched；dev-only 的 `FINESUB_AGENT_TRANSPORT=capsule` 钉住的 `per-window` 也是；`per-window` 还要该 target 的 CLI
真的接受 per-invocation MCP server（`supports_mcp_config`，探一次 `--help`，按 target 记忆化，探不到就当作不支持）。
链上**每个**可能应答的 agent 都要满足——回退到链尾的那个也会读到同一句「你自己取」。
`FINESUB_TASK_STRATEGY=batched|agentic` 可钉死（开发用，优先于下面两条），`--test-profile` 为 batched。策略、来源
（`correction_chain` / `test_profile` / `env_pin` / `override`）与任务列表记在 `task_graph` artifact。

## 快速模式（--fast auto|on|off，默认 auto）

短输入把整段作为**单个窗口任务**处理：不跑准备任务（没有 context pack），首条消息就是纠错契约加请求通道
（`request_channel` 为真时：local，或知识开启且索引可读），按 `correction_media` 附加无媒体、音频或音视频整段剪辑；知识开启且索引可读时请求通道里可请求词条；
`--extra-info` 作为 `general_context.user_notes` 直接进窗口 prompt（`stages/plan.fast_context_pack`，没有别的环节消化它）。
`retrieval=native|none` 的 fast run 同样是单窗任务；没有可读知识库时没有请求通道。

auto 判定三个条件都过才启用（结果与数值写入 `fast_decision` artifact）：

- 输出：`k × c × 全量 csv_tokens ≤ 0.8 × 65536 − 10000 = 42428`；
- 输入：首条消息（countTokens，按请求通道形态组装）+ 剪辑媒体 token ≤ `prompt_input_limit − 56000`
  （`FAST_FILL_INPUT_RESERVE_TOKENS`，为回填轮的搜索结果与词条注入留余量；包络由绑定组的 catalog 行算出，免费 Gemini 上仍是 194000）；
- 质量护栏：整段 `<asr_result>` ≤ `max_window_subtitle_tokens`（默认 10,000，见「窗口拆分」）。
  快速窗口按定义就是全片，是最容易撞上这条的路径，所以它与两个预算条件同级参与判定。

⚠ 输入条件一律按 batched 的请求通道形态组装首条消息。agentic 执行者（全 agent 链）的整段窗实际发的是「自己取」那段
说明、且不带知识库索引，所以它的首条消息被**多算**：实测多 956 字符（空库）、约 2.8k（50 条索引）、约 6.7k（200 条），
对 138k 预算约 2%。多算的方向是安全的（退回常规多窗流程），要算准就得把路由与 driver 探测拉进 planner，所以不改
（2026-09-10 量过，owner 2026-09-30 结案）。只在全 agent + 纯文本 + `--fast on`、输入正好卡在线上几 k 之内时会报错。

`--fast on` 不满足即报错退出；`--fast off` 强制常规多窗流程。快速模式不写 `*-research-context.json`（没有准备任务）；
`--context-file` 传入的现成上下文照常注入。

## 输入与输出

输入以 `*-stable.json` 为字幕源。纠错翻译阶段会保留下列中间 artifact：

- `*-raw.srt`：从 `stable.json` 渲染出的完整 ASR 原文 SRT，便于人工排查；若请求的最终 postprocess profile 包含时间轴步骤（当前为 0/1），仅同步执行重叠修复 + 末端延长/闪轴闭合（profile `4 → 1`），文字保持原样。
- `*-corrected.srt`：从模型输出的 `corrected_text` 列渲染出的纠错后原文 SRT，便于分析 ASR 修正和误听模式。
- `*-translated.srt`：模型输出的中文字幕直出版本，尚未做最终 SRT 后处理。
- `*.srt`：最终中文字幕 SRT；默认由 `*-translated.srt` 经过 profile 0 后处理得到。中文素材只纠错（[只纠错](#只纠错)），`-corrected.srt` 是忠实转写，`-translated.srt` 与最终 `*.srt` 是基础润色版。
- `*-research-context.json`：准备任务的产物（context pack + 请求/交付两条回复的输出（含 analysis_notes）+ planning 元数据 + token 报告），写在 task artifact 目录下（如 `input.llm-artifacts/input-research-context.json`）；旧 run 若仍在 SRT 同级，首次触达时会迁入 artifact 目录。

纠错窗口输入格式是：

```text
global_id|local_start|duration|gap|text
```

时间单位为秒，展示到 0.1；`local_start` 以本窗口**剪辑音频的 0 秒**为基准，每条 CSV 记录占一个物理行。传给多模态纠错模型的音频是原始音频（不是人声分离后的 `*-vocal.ogg`；旧产物可能为 `*-vocal.flac` 兜底格式）按窗口裁剪出的 mono-16k AAC 片段（ffmpeg：`-c:a aac -ac 1 -ar 16000 -b:a 32k`）：范围是窗口首条字幕头到末条字幕尾（云端任务里另一个人的行比最后一行结束得晚时，到那一行的结束：`line_groups.run_end`，窗口的 `end` 同此），两端各加 5s padding；含全局第一条字幕的窗口头部 padding 为 60s（clamp 到 0），含全局最后一条字幕的窗口尾部 padding 为 60s（clamp 到音频末尾）。剪辑写到任务 artifact 目录下的 `clips/<chunk_id>.aac`（每次覆盖；随 artifact 目录一并清理），每个执行窗口（含 `-a`/`-b` 半窗）单独上传一次并在请求回合/交付轮/同窗重试间复用。窗口规划完成后即在后台线程预切首窗；进入窗口 *i* 时预切 *i+1*，切完即上传 Gemini，主循环仅在取当前窗 file ref 时等待就绪。`*-translated.srt` 时间轴由本地按源序号从原始 `stable.json` 回填，因此保留原始高精度。最终中文字幕 SRT 使用 `translation` 列再做后处理；`-corrected.srt` 使用 `corrected_text` 列。

默认命令只写出计划和 prompt（准备任务的首条消息 + 每窗口纠错 prompt）：

```powershell
python -m finesub.llm.correction_translation out/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts
```

执行 API 时默认串联：准备任务 → 纠错窗口循环（`--execute` 必须提供 `--audio`；旧的 `--audio-file-id` / `--upload-audio` 已随按窗剪辑上传移除）：

```powershell
python -m finesub.llm.correction_translation out/input-stable.json --audio data/input.wav -o out/input.srt --execute
```

这个 CLI 与管线的 `translated-srt` / `final-srt` 阶段、reference ingest 走**同一个任务**
（`run_correction_task`）：同样按存在性跳过已有的 `-translated.srt` / 最终 SRT（欠着的后处理与
`--knowledge update` 照做），同样把 `llm_harness` 计时写进 run metadata、在会话关闭后写任务报告。
要重跑就删产物，与管线一致。

相关 CLI 参数：

- `--media {text,audio,video}` / `--retrieval {none,local,native}` / `--difficulty {quality,intermediate,efficiency}`：开关轴（两个入口没写时取激活预设的默认值，出厂 `default` 下是 `audio/local/intermediate`；管线 CLI 的 `--llm-media video` 在纯音频输入上自动降到 audio 并提示）。`difficulty=efficiency` 不钉别的开关：只写它时在出厂 `default` 预设下是 `audio/local`、知识库照常读（`update` 也可以）；要纯文本、不检索就用 `fast` / `agy-fast` 预设（它们在自己的 `defaults` 里写了 `media = "text"`、`retrieval = "none"`）或自己写开关。它的风格只用出厂默认（`knowledge/style.py` 的 `FACTORY_STYLE_NAME`）：配置里的 `style` 与知识库的 `default_style` 是用户的默认，在这一档不生效；命令行点名风格（含 `auto`）报错；续跑照任务记下的风格；不写回风格（显式 `--style-mode update` 报错）。**difficulty 做的事**（model-routing v2）：选任务组格子的 prompt 变体（出厂：纠错 quality、intermediate→capableC，efficiency→basicA）和该格的思考旋钮；纠错任务按它定能不能走三个 pass、`auto` 下跑不跑 pass 3（见[三个 pass](#三个-pass)）。它不从应答 endpoint 读 capability tier（该列已删），也不影响窗口几何。预设可以**按档位绑不同模型组**，出厂纠错就是这样（quality、intermediate=capable 组，efficiency=两个 lite、思考 low），所以「切到 efficiency 继续」是用户显式发起的降档，而非链内静默降质。（生产 CLI **没有** `--variant`：变体由任务组格子决定；按名覆盖只存在于 `tools/session_replay`，见 docs/prompt-iterate.md。）
- `--parallel-windows N`：`continuity=parallel` 的最大并投窗口数；没写时取激活预设的 `parallel_windows`，否则 1（2026-08-30 起；任务级并行是首选的并行面）；串行模式忽略。
- `--output-scale K`：输出估算系数 k（默认 1.0）；调大切出更小窗口。
- `--fast {auto,on,off}`：快速模式开关（默认 auto）。
- `--video PATH`：源视频文件；`--media video` 时 `--execute` 必填（见「视频」）。
- `--extra-info` / `--extra-info-file`：用户提供的额外信息（来源 URL、内容说明、额外要求），注入准备任务，其中要纠错照做的要求由准备任务转述给纠错窗口（`user_requests` 与专属背景）；fast 模式下作为窗口的 `user_notes` 直接注入。
- `--context-file <research-context.json>`：复用已有准备结果，跳过准备任务。
- `--research-only`：只跑准备任务并写出 `research-context.json`，不进入纠错；和其他用法一样要 `--execute` 才调用模型，不加时是只出计划与 prompt 的 dry-run（冲突检查照旧）。快速模式没有准备任务，所以这时 `--fast auto` 落成 `off`，`--fast on` 报冲突（`fast_mode_for_research_only`）。
- `--extra-style`：注入纠错 system prompt 的特殊翻译风格段落（默认为空）；只纠错的任务不用它，告警一次。

## 纠错输入的段序列

`chunking.load_segments_from_stable_json` 是纯加载：拿到的 stable JSON 已经是最终段序列，
源序号按位置顺次编号。曾经在这之前还有一道确定性预合并（`src/premerge.py` / stabilize
profile 3），2026-07-29 随 `segment_split` 迁到全局 DP 一并删除——分句器自己决定 ASR 段接缝
的去留，词中切断的碎片不再产生，预合并因此无对象可并。

**为什么这么定**（M.1–M.10 十条决策，含被证伪的假设与教训）见
[`decisions.md`「确定性预合并与切点锚定的决策记录」](decisions.md#确定性预合并与切点锚定的决策记录2026-07)
——决策台账只在那一处，这里不复述。

## 窗口拆分

Harness 采用"先估算窗口数、再均匀放置分割点"的规划方式：

1. 对全量 CSV 做一次 countTokens，按每行字符占比折算每段文本 token；每段媒体 token 按 `媒体速率 × 到下一段开始的时间跨度` 折算（`media=audio|video` 含音频 32 tok/s，`media=video` 再加画面 17.75 tok/s；`media=text` 为 0），得到每段的规划质量（mass）与前缀和。
2. 由输出约束（每窗字幕 token ≤ `窗口输出预算 / (k × c)`，其中窗口输出预算 = `0.9 × 65536 − 5000 = 53982`，k 为 `--output-scale`、c 为开关合成的输出系数；默认 `audio/local/intermediate` 算出约 11,996，实际由下面第 4 步的质量护栏 10,000 定）与输入约束（字幕+媒体+上下文 ≤ `prompt_input_limit`，免费 Gemini 上是 `194000`；含重叠与 padding 的固定加成）估算窗口数。规划时固定预留 `72000` tokens 的上下文额度（`WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS`），覆盖纠错调用中窗口 CSV/媒体之外的全部内容：静态 system prompt（实测 ~4k）+ user 脚手架 + 调查 context pack + serial advice 台账（≤8k）+ 请求回合 notes + 搜索结果块（≤20k）+ 知识库词条块（≤28k），最坏合计约 69k；窗口实际由输出公式或质量护栏限死，加大 reserve 几乎不改变窗口数。
3. 在均匀 mass 目标点附近（半径约 `0.4·n/k`，由近及远）snap 到合适边界；每个规划窗口再用真实 countTokens 预算校验，任一窗口超限则 `k+1` 全局重排（保持均匀），上限 `k0+16` 后报错；单段放不下直接报错。发生过重排（输入超预算导致窗口缩小）时写入 `window_plan_report` artifact（`estimated_windows`/`planned_windows`/`replan_attempts`/最后一次超限错误，分 research/correction 两个 phase），task report 渲染为独立 "Window Planning" 小节。
4. **质量护栏 `max_window_subtitle_tokens`**（`ModelLimits` 默认 10,000；纠错格绑定组的 catalog 模型可以各自声明 `window_subtitle_tokens`，组内取最小、没写的按默认计；config.toml `[chunking]` 写了就压过它们，`0` 关闭，非法/负值硬报错；输出预留按组值算，显式值不进预留）：单窗 `<asr_result>` CSV（正文 + 重叠行）的 token 上限，独立于输出系数——窗口太长时翻译质量会掉，哪怕输出装得下。默认为什么是 10,000、缩到 5,000 换来什么，见 [`llm_design_notes.md`「纠错窗口字幕上限的决策记录」](llm_design_notes.md#纠错窗口字幕上限的决策记录2026-11-0708)。两处生效：第 2 步的窗口数估算取它与输出约束的较小者，第 3 步的真实 countTokens 校验后再硬查一次，超限走上面同一条 `k+1` 重排路径。
5. research 与 correction 可以采用不同窗口几何；research 笔记按 source-id 区间重映射。
   `research-context.json` 仍带完整 `planning` 元数据供审计，几何不再参与复用比较。

每个窗口包含：

- 本窗口的输入源序号范围与重叠源序号（物理上重复包含上一窗口尾部；模型照样回答这些行，但拼接时重叠区留上一窗口的行，见[重试与拼接](#重试与拼接)的拼接规则）。
- 重叠条数 = `边界前 30s 内开始的字幕条数`（v13 起纯内容驱动、无下限），并 clamp 到上一窗口长度以内。边界落在 >30s 空档处时重叠为 0——这是正确行为：拼接不该跨大空白，连续性由只读前文块负责。
- 只读前文 `preceding_segments`：窗口开始前最多 10 条 raw ASR 字幕（`PRECEDING_CONTEXT_MAX_SEGMENTS`，固定条数、**不设 gap-stop**——大空档后恰是冷启动风险最高处，负时间戳让模型自行判断新旧）。第一个窗口为空；-a 半窗继承父窗口的、-b 半窗回看父窗口自身尾部。纯输入、不参与翻译、不影响窗口位置与 clip 范围。
- 音频剪辑区间 `clip_start`/`clip_end`（见上文 padding 规则）。
- 预算估算：字幕文本 token、剪辑音频 token（含 padding）、预计输出 token、总量和计数来源。

边界选择规则：

- 优先在句号、问号、感叹号、省略号等自然句末截断，其次是较长静音（`even_sentence_or_gap_boundary`）。
- snap 半径内没有合适边界时在均匀目标点强切，标记 `forced_even_boundary`。
- **被盖住的行之前不切**（云端任务）：一行开始时另一个键的行还没说完（[Prompt 信息](#prompt-信息)的「同时」），它之前不是候选切点，上一行以句末标点结尾也不行（`is_reasonable_boundary(covered=)`）；全被排除时才强切。纠错窗、对半拆开的窗（`split_window_in_half`）与说话人裁判的半窗（`speaker_judge/windows.halves`）三处共用。没有键的任务没有被盖住的行，切点不变。

预算规则：

- 输入 prompt 上限：`prompt_input_limit`——由绑定组的 catalog 行算出
  （`min(max_input_tokens, context_window − 预留)` 取组内最小；预留自 2026-09-04 起是**预计输出**
  `output_scale × 输出系数 × 每窗字幕上限`，不再是声明的输出上限），免费 Gemini 上是 `194000`。
- **不再有单独的「模型上下文规划上限」**（2026-09-03 删）：包络已经从上下文里扣掉了输出，
  `输入 + 预期输出 ≤ context_window` 按构造成立，那道 `256000` 的检查永远不会响。
- API 输出上限：`output_limit`——组内最小的 `max_output_tokens`，免费 Gemini 上是 `65536`。
- 文本 token 计数按 **本地 tokenizer 二进制 → `countTokens` API → 启发式** 三级 fallback（`default_token_counter()`，逐 sha 缓存）：
  - 首选本地 `tokcount`（Go，go-sentencepiece 编码，Gemma 3 词表内嵌在二进制里，首次运行也不联网；源码在 `tools/tokcount/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入 pyproject 依赖）。Python 进程内所有 counter 实例按 binary 共享一个 lazy 启动的 stdio server；默认空闲 300 秒自动退出，下次精确计数透明重启，避免逐次加载词表。重启之后仍起不来时，本进程 30 s 内不再启动它、直接走下一级，再失败间隔翻倍、封顶 10 分钟。离线、免配额；词表是 genai 本地 tokenizer 给 `gemini-2.5-flash` 用的那份，实测与 3.1-flash-lite 的 `countTokens` 相差**恒定 +1 token**（API 的 `contents` 外壳），Harness 已加回该 offset 使二者逐字一致。
  - 本地 binary 可执行时，截断/注入预算跳过启发式预检，直接使用常驻 server 的精确结果。本地 binary 不可用时才启用 heuristic fast path：明显低于上限则直接返回估算，接近或超过上限时进入 `countTokens` API 精确计数；API 再失败才回落启发式 counter（`HeuristicTokenCounter`，按字符类别加权求和：数字/拉丁/CJK/谚文/全角标点/其他文字/空格/ASCII 符号/其他，权重经实测拟合为**上界**——对每个测试类别 heuristic ≥ real，对实际喂给模型的字幕 CSV 最紧约 +1~8%）。旧版启发式因对 CJK 混合文本低估 25-40% 被弃用。
  - `execution_policy=agent-only` 时，counter chain 明确移除 `countTokens` HTTP backend，只允许本地 binary → 启发式；“不调用 provider API”覆盖预算估算辅助路径。调用方注入 client 时以该 client 的实际 execution settings 为准，不读取全局 policy 覆盖它。
  - `countTokens` 端点**完全免费**：不消耗任何生成配额、不计费、无实际速率约束，`.env` key 只用于鉴权。因此即便回落到 API 也不烧 quota。
  - `countTokens` 的时限是 connect 5 s、读 15 s；连接阶段失败隔 1 s 再试一次，其它失败不重试。它的 host 判定连不上之后（[连不上的 host](#连不上的-host)）不再发请求、直接落到启发式，只按探测间隔放一次探测。
  - 落到启发式时每个进程打一次 `token-count-heuristic` Warning，写明前面各级为什么失败、估算偏大约 1–8%（窗口会略小）。
  - 本地二进制在位时**默认 dry-run 无需联网/无需 key**；缺二进制才回落到 countTokens 端点。
  - **二进制从哪来**：源码 checkout 跑 `bin/` 里那份（不被跟踪：新 clone 没有它，回落到 `countTokens`，直到在 `tools/tokcount/` 编译或解开 Release 资产）；发布版 CLI 不带 `bin/`，改由 `runtime-manifest.json` 的 `tokcount` 资源下载到 `runtime/tokcount/<版本>/`，前端用 `GEMINI_TOKEN_COUNTER_EXE` 指名注入（`finesub_bootstrap.environment.token_counter_overrides`，解析顺序：用户显式设的环境变量 → 系统已有的 → 托管的）。它是**唯一一个装不上也照跑**的托管工具，不拿它当门槛：CLI 在 LLM 阶段任务前尽力拉一次、失败只警告。发布与版本规则见 `tools/tokcount/README.md`。
- 每轮候选规划需要 `k` 次 token 计数校验，通常一轮即收敛。
- 基于 token 上限的文本截断走 `finesub/llm/token_truncate.py::truncate_to_token_window`（插值+二分搜索最接近上限的安全切片，按切片长度缓存计数，只需个位数次 counter 调用；`keep="head"` 保留前缀/截尾部（默认），`keep="tail"` 保留后缀/截前缀；可选回退到自然句末边界）。两个默认开启的快速开关：本地 binary 不可用时，`lazy` 先用启发式 upper-bound 预检，`估算 × 1.02`（`lazy_safety_factor`，额外保险）≤ 上限则原样返回、零 API 计数；本地 binary 可用时直接精确计数；`quick` 把截断搜索的命中窗口放宽到 0.95/50（更少计数次数）。通用 `cap_tokens`、注入预算和累计 advice ledger 使用同一分流；显式传 `gold_ratio`/`abs_slack` 时 `quick` 不覆盖。
- 音频 token 本地按 Gemini 官方口径 `32 tok/s` 乘以**剪辑时长（含 padding）**估算；`media=video` 另加 `71 tok/frame × 0.25 fps = 17.75 tok/s`。由于每次调用只附本窗剪辑，估算口径与 provider 实际计费一致。
- 纠错输出估算：`k × c × csv_asr_result_tokens`（k = `--output-scale`，c = 3.5 + 1.0(retrieval≠none)，media、difficulty、continuity 都不改变 c；替代旧的 `csv × 5 + 10000` 启发式）。
- 窗口规划要求该估算 ≤ 窗口输出预算 `0.9 × 65536 − 5000 = 53982`（快速模式收紧为 `0.8 × 65536 − 10000 = 42428`）；真实 API 请求仍使用 `65536`。

## 纠错窗口调用形态（由 retrieval 与知识库决定）

- **有请求通道（`request_channel`）**：一个窗口是**一条对话**（`stages/correction/request_turn.py` 开场，`attempts.py` 收尾）。首条消息 =
  纠错契约 + 「获取缺失输入」一节：模型读完本窗输入后要么直接交付终稿（作为 attempt 0 验收，不再调用），要么只输出
  `<window_notes>`（≤800 token）、可选 `<requested_entries>`/`<keep_entries>`（新请求 ≤8、与透传合计 ≤12）
  与 `<search_queries>`（≤8 条，可为空块）。harness 查库 + 一轮本地搜索后把词条全文与 `<search_results>` 作为下一条 user
  消息回填（模型的请求回复作 assistant 消息放回历史，剪辑跟在最后一条 user 消息上），交付轮与修复轮都在这条对话里；回填只有
  一次，再请求按格式错误重试。请求回合 best-effort：调用异常按空产物继续——**要选风格时除外**（见下一条）。产物 `RequestTurnProduct`（搜索结果 + window_notes +
  请求/保留的词条 key + 请求回复原文）按 base 窗口 id 缓存：同窗口的 validation 重试和 `-a`/`-b` 拆分半窗复用第一次的结果，
  不重复调用或重搜。`continuity=parallel` 下请求回合在前置 pass 并发跑（`previous_advice` 为空），屏障后再并投交付轮。
  `retrieval=none|native` 时（2026-09-30 起，有可读知识库才有）同一条路只请求词条：通道里没有搜索规则，请求回复以
  `<requested_entries>` 或 `<keep_entries>` 为准（local 仍以 `<search_queries>` 为准），回填只有词条；native 下通道的措辞改为
  「联网搜索用你自带的工具」，请求回合与交付轮都开原生搜索。
  否决过的替代：harness 按原文精确匹配、把命中的词条逐窗预注入（不加回合、更便宜）。精确匹配恰好漏掉最需要纠正的误听专名，
  而且是 harness 替模型选，与 agent 路「模型自己取」相反；匹配面能可靠覆盖误听（音近匹配）时可重评（2026-09-30）。
- **fast 窗口要选风格（`style_pick`，`--style auto`）**：选择是交付契约的一部分（[知识库契约](knowledge.md)「style 类别」）。
  batched：首条回复必须是请求回复并带一个合法的 `<style_selection>`——直接交付、只请求不选、选了不在候选里的名字都按格式错误
  重试，`REQUEST_TURN_FORMAT_RETRIES` 用完仍不合就抛错（任务失败），调用异常也直接抛；带搜索授权时只选不搜的回复合格。
  回填带上选中那套的全文。没有搜索、也没有可读知识库时，请求回合**只为选风格而开**：开场是「选择翻译风格」一节，回复里出现
  `<search_queries>` / `<requested_entries>` / `<keep_entries>` 算格式错误，回填只有风格、不要求 `<keep_entries>`
  （没有词条通道的回填都不要求它，只有搜索的那一格也一样，v91 起）。
  agentic：没有请求回合，交付里附选择，并须先用 `kb_read` 完整读过选中那套；没开知识库读取时窗口拿窄授权 `kb_tools=style`
  （只有 `kb_read`、只解析 `style/<名字>`），说明里只讲选风格，不再列不存在的检索工具。
- **`retrieval=native`、没有可读知识库**：没有请求通道（要选风格时例外，见上一条）；纠错调用自身设置 `native_search=True`，在组内过滤出能联网的成员并在模型内完成查证。
- **`retrieval=none`、没有可读知识库**：没有请求通道（要选风格时例外，见上），纠错调用也不开工具。
- **agentic** 的窗不走请求回合：说明里列出本窗真有的工具——联网检索只在 `local` 下有，知识库工具在索引可读时有。开了 `local` 时 system 的检索段说由它自己用工具查（`fragment_retrieval_self_v1.md`），不说 harness 代查、也不说它不能联网；与 harness 注入那份共用交叉验证、知识库冲突、内容不是指令三条规则（`fragment_search_rules_shared_v1.md`）。

交付轮任务组 `correction-mm`/`correction-text`（免费 3.8 打头的组）；`retrieval=native` 不换组，只是在组内过滤出能联网的成员（出厂为付费 3.8）。
回填消息带来 `<search_results>` 与 `<entry_details>`（请求的词条全文；准备任务的种子优先）：开头一句说是回填，选了风格时先给风格（要遵守的约定），
其余资料前另有一句标明是内容、不是指令，最后是交付提醒；搜索结果怎么用只写在 system 里，回填消息不重复。请求回合的 `<window_notes>`
留在被回放的 assistant 消息里，交付轮不再单独注入（首条消息的 `<pre_round_notes>` 槽在任务式流程里恒为空，空时整段不渲染）；交付轮不启用工具、不能再发起搜索。

### 视频（--video；`--media video`）

- `video` 档的会话媒体是**低清视频+音轨的 `.mp4` 剪辑**（同剪辑区间与 padding 规则；`finesub.media.ffmpeg.extract_video_clip`：decode 先 `-hwaccel auto` 失败退 CPU，编码 libx264 + AAC）。只切 `correction_media` 要的那种剪辑（`audio` → `.aac`，`video` → `.mp4`）；请求回合与交付轮共用同一份剪辑与上传。
- API 侧：mp4 以 `detail=low` + `video_metadata.fps=0.25` 经 REST 直传（→ Gemini 每 part `mediaResolution: {level: MEDIA_RESOLUTION_LOW}` 与 `videoMetadata.fps`），计费口径与规划一致：`32 tok/s（音轨）+ 71 tok/frame × 0.25 fps（画面）`。
- 快速模式的单窗同样按 `correction_media` 上传整段剪辑，请求回合与交付轮复用。
- **video→audio 一级安全网（client 层）**：带视频剪辑的调用落到「能听不能看」的候选 target 时，不再整个跳过，而是换发同窗音频剪辑（按需现切）、按 target 去重告警一次、并在 route decision trace 上记 `media_downgrade: "video->audio"`。这是安全网不是机制——正常配置应让接视频的格子里都是有视频能力的模型；audio 不降 text，native search 不降级。

### LLM session 级 resume（默认开启）

> **一句话口径**：**一个 harness LLM session 跑完、校验通过，就是一个 checkpoint。** 重跑时它
> 直接复用，不再调模型。下面的分层与失效键回答的是另一个问题——**改了什么之后这个 checkpoint
> 就不算数了**——不是"复用单位是什么"。
>
> 两个词的关系也在这里说清，别的地方不再重复：**窗口**是把字幕切成的一段，纠错链上**一段就是
> 一次 session**，所以「窗口」和「session」基本同义；**stage** 是流水线的一大步（调查 / 纠错
> / …），一个 stage 里装着好几个 session（比如准备阶段 = 请求回复 + 搜索 + 交付回复），它们各自存档，合起来
> 产出这个 stage 的产物。

传入 task artifact 目录且 `resume=True` 时，生产 harness 会把以下经当前 parser/contract 验证成功的原始响应追加到 `<artifact_dir>/session-checkpoints.jsonl`：准备任务的请求回复与交付回复（session `prepare-t1` / `prepare-t2`，超长素材按块 key），以及纠错窗口的请求回合（session `request`，key 为 chunk id；直接交付的回复同样入账，回放时作为 attempt 0 再过 validator）。ledger 为 append-only JSONL；每条 committed 记录包含 `schema_version`、稳定 session/key、`input_hash`、`content_hash`、原始 `content` 和模型 metadata。截断行、坏 JSON、未知 schema/status 或 content hash 不符的记录加载时直接忽略；读写走 `finesub.jsonl`（只按 `\n` 切行、追加前补齐崩溃留下的半行），除了没写完的最后一行，坏行打一行 `Warning:`。

**一条例外（准入门 D 的 C 案，2026-08-19，docs/llm_local_agent.md §7）**：依赖隐含 provider
历史的调用——`agent_session_mode=resume` 下跨窗继承了 conversation handle 的那次 agent 调用——
被打上 `LLMCallResult.resumable=False`，**不写进本账本**：它的 hash 不覆盖那段只有厂商知道的
历史，两次不同输入会被认成同一次。代价只有「中断后那一次调用重发」，已提交的窗口/stage 不受
影响。

`input_hash` 覆盖精确组装后的 messages、`PROMPT_VERSION`、角色/输出上限/thinking 等调用配置，以及 messages 外的调用状态。**它一个字段都不放松**：docs/llm_local_agent.md §4 让它做 `submit()` 的 compare-and-swap 令牌，§12 让它成为未来统一 task runtime 的唯一提交口径。它只服务于“外层 stage/window 尚未提交”时的单次调用恢复：重启从确定性本地代码重建状态，到同一边界且 hash 精确命中才取旧响应，并再次走**当前** parser/contract；复验失败即 live 重打。这里的精确性是为了证明中断点能有效重建，不会向上推翻已经提交的 research context 或纠错窗口，也不要求整条任务保持同一模型/prompt。请求回合命中只省模型调用，关联搜索仍会重新执行。

因此细粒度可恢复边界是：准备任务的请求回复后与交付后；每窗请求回合后。纠错交付轮使用下述 `correction-windows.jsonl` 整窗提交缓存；知识更新另有 `knowledge-update-chunks.jsonl`。`--no-resume`（`resume=False`）使 session 与 correction-window 两种 ledger 都不读也不写，但不删除旧文件，也不关闭 pipeline 对完整 `*-research-context.json`/SRT 文件的 stage 级存在性复用。

### 纠错窗口中途 resume（默认开启）

成功且未截断的执行叶会把原始响应追加到
`<artifact_dir>/correction-windows.jsonl`。记录包含 chunk/source id、clip、continuity、
`task_fingerprint`（现为 stable 源指纹）、完整 `input_hash`、不含词条签名的 `input_hash_core`、实际注入词条 keys、
响应、实际 `variant` / `difficulty`、答它的 prompt 版本 `prompt_version`（分组窗是 `GROUPED_PROMPT_VERSION`，其余窗是 `PROMPT_VERSION`）与兼容用 capability tier 元数据；拆分父记录另写
`split_into`，恢复时先重建同一棵 `-a/-b` 叶，因而
首次拆出的前半窗也可复用。任何缓存响应都要再过当前 variant-aware validator（`ResumeLedger.replay`：串行、并行、复用计划的预拆与三个 pass 的规划共用这一个检查）。

- `task_fingerprint` 是一份**显式 include 列表** `WINDOW_INVALIDATION_INPUTS`：`test_profile`、源 `*-stable.json` 解析后的 id/时间/文本序列（排版或无关 JSON 字段变化不算源变化）、源媒体的 path+size（**不含 mtime**：重下载同一份音频不是内容变化）。
  ⚠ **翻译风格不在里面**（2026-09-02）：`extra_style`（自由文本）与 `--style`（具名条目）中途改了都不作废已完成的窗。后果是产物前半段一个口吻、后半段另一个——owner 判定可接受，不值得为它重跑整批。这条推翻的是 `commit.py` 注释里曾明写的失败模式，理由在影响面：风格条目有界、形状固定、只碰风格，与知识库词条正文同一档。没有被分类的字段进不了这个列表，`_task_fingerprint` 会断言 payload 与列表一致——漏维护的失败方向是「多重跑一个窗口」，不是「错误复用」。
- ⚠ **prompt 版本不在里面**（owner 2026-11-08）：升 `PROMPT_VERSION` 或 `GROUPED_PROMPT_VERSION` 不作废未完成任务已提交的窗口，窗口计划与并行的会话固定集也照用。每窗记着答它的版本，续跑时版本不同的窗照样回放，阶段结束时告警一次（`correction-older-prompt-windows`：各版本分开计数，只数真正回放了的窗；补救是删掉 `correction-windows.jsonl`，当前版本答过的窗也会跟着重做）。仍会重做的只有两种窗：**渲染出的输入变了的**——窗口身份哈希 `input_hash_core` 覆盖渲染出的输入 CSV（列、`gap` 的含义、窗口内的局部编号），所以改了输入渲染的 bump（例如 csv-v111 换 `gap` 的方向）照样重做全部窗口，只改指令措辞的才沿用；以及**当前校验器不再放行的**——回放校验连同质量门槛（保留比例、整窗错位、按比例的长度门槛）一起跑，门槛后来收紧了，旧窗就重做。记录里点名的 variant 或档已不存在（如已删的 `capableB`）同样算校验不过、重做，不报错。输出列的含义一变就换一个 header 名：回放逐字校验 header，所以换了名的旧窗重做；同名换了含义的列会被当成旧义照用，所以不设手工升的兼容常量（owner 2026-11-08）。
- 因此**不在**失效键里的有：prompt 版本（上一条）、execution identity（模型、预设、模型组、thinking、execution policy、agent driver/effort/超时；docs/llm_local_agent.md §11）、difficulty 与 variant（逐窗记录实际值，按记录里的 variant 回放校验）、知识库词条与常见错误正文、`task_update_feedback`、context pack，以及几何类旋钮（`correction_media`、`retrieval`、`--output-scale`、窗口上限、组包络）。
- 回放要求 `input_hash_core`（当前窗口 + 只读前文的 id/时间/文本，当前窗口另含换人标记）匹配，并把原始响应按记录中的实际 variant 再过当前 validator。完整 `input_hash`、注入词条 keys 与 `knowledge_version`（知识库嵌套 git HEAD，docs/llm_local_agent.md §8）留作审计。
- 只缓存"成功且非 output_limited"的窗口（与既有提交门槛一致）；`-a`/`-b` 半窗各自按 chunk id 缓存，拆分父记录保存 `split_into`，恢复时会重建并复用完整拆分树。
- **窗口计划持久化**：非 fast 且 resume 开启时，规划结果写入
  `<artifact_dir>/correction-window-plan.json`（含 `source_fingerprint` 与审计用 `geometry`）。复用门只看
  L2 `task_fingerprint`；计划的语义只是 chunk id 与 source-id 边界，文件里**只有**
  `chunk_id`/`segment_ids`/`overlap_ids`/`boundary_reason`（schema v2）。clip 区间、token 预算和
  只读前文都是派生态，恢复时按当前 profile、组包络、媒体时长与当前段列表重算——落盘的旧包络
  数字没有机会混进新一轮。
- 复用计划先展开既有 `split_into` 树、验证可回放叶，再对 pending 叶检查输入包络、输出 CSV
  上限和质量护栏；不合身则在发车前递归拆半，并先写同一套 `split_into` 记录。发生预拆时打印
  stderr 告警并写 `window_refit_report`；单段仍放不下会以包含 source ids、估算值、限额和失败面
  的明确错误终止。拆分只会让计划更细；若换大模型后希望合并成更少窗口，需显式删除
  `correction-window-plan.json`（这也放弃原 chunk id 对缓存的寻址）。
  ⚠ **预拆和重试拆分花的是同一份预算**：refit 也要过 `WindowGeometry.may_split`，深度到顶
  就以同一种错误终止，不会再拆。（它以前是**不问的**，所以复用计划每续跑一次就能再对半一层、
  一路越过上限——2026-09-03 修。）
  ⚠ **上限管的是「能不能新拆」，不是「能不能回放」**：上限从 2 降到 1 之前记下的 `0001-a-a`
  仍会被 `expand_cached_splits` 完整重建并回放——已完成的产物不因今天换了个参数而作废
  （`README_DEV.md`「复用的依据是任务身份」）。被拒的只是**从它再拆一次**。
- **serial**：接受同源且 core hash 匹配的记录，并按窗口顺序回放；
  回放重新执行 `merge → advice/transfer commit`，所以后续 live 窗看到的台账与首次运行一致。
- **parallel**：缓存可以是任意窗口子集。先回放可用叶，再对 pending 叶并投请求回合；屏障
  使用首次运行持久化在同一 JSONL、同一源指纹下的 `parallel_entry_set`，不会用
  “剩余窗口的请求”重新缩小会话固定集。key 固定，正文每次从当前 KB 重渲染。第二阶段只
  投 pending 纠错窗；所有在途窗排空、成功窗写缓存后才抛首个错误，最终结果按窗口顺序合并。
  同批累计 3 个失败会阻止尚未开始的窗口继续发车。
- **模式切换**：serial / parallel 双向都复用已完成窗口，结果可以合法混合两种执行方式。
  `parallel -> serial` 会**打印告警**：parallel 窗从不写 advice，所以恢复后的 serial 台账比首次
  运行短一截，后续 live 窗看到的东西因此不同——允许，但不静默。
- **research context（L3）**：planning metadata 全部落盘作审计；`research_reuse_key` 只比较
  stable 源、`extra_info_hash`、`test_profile` 与
  `research_semantics_id`（当前为 retrieval）。媒体/窗口几何、模型、知识版本、difficulty、continuity
  均不作废已提交调查；`prompt_version` 也只作审计、不告警——它是纠错的 `PROMPT_VERSION`，不是准备任务的，
  超长素材分块调查的逐块产物同样不按它重跑。`planning_metadata` 的每个字段都必须显式归入 gate 或 audit-only 两组之一
  （`RESEARCH_REUSE_GATES` / `RESEARCH_REUSE_AUDIT_ONLY`，测试断言两组之并等于全部字段）——
  这是 L2 `_task_fingerprint` 那条断言在 L3 的对应物：漏分类会在测试里炸，而不是在字幕里。
  损坏/不兼容文件在覆盖前复制为同目录 `*.invalid[.N]`，备份失败则拒绝覆盖。fast context 规则不变。

## Prompt 信息

纠错窗口 prompt 会给模型这些信息：

- `<asr_result>`：本窗口需要处理的直接类 CSV 文本块（时间以剪辑 0 秒为基准），header 为 `local_id|start|duration|gap|text`。`gap` 是上一行结束到本行开始的秒数（与上一行重叠时为 0，首行留空），与 `turn` 列一样说本行与上一行之间的接缝（csv-v111 起；此前是本行结束到下一行开始，prompt 要反复教模型「与前一句的间隔看前一行的 gap」，末行恒为 0）。只读前文与知识更新的 `<raw_csv>` 同义。每个执行窗口（含 fast 单窗）都把目标行重编号为 `1..N`。
- **换人标记 `turn`（v95）**：任务有说话人标签（`--diarize`）时，`<asr_result>` 的 header 是 `local_id|start|duration|gap|turn|text`，与上一行主轨不同的行在 `turn` 填 `换人`，其余留空；同一任务每个窗口都有这一列，没有标签的任务没有，prompt 与 v93 逐字相同。列说明写明「空只表示没判出换人、不保证是同一个人」；合并规则多一句「标了换人的行不得与上一行合并，合并行里除第一行外任何一行标了都不行」，「合并行检查」多一句标了就拆开，最后提醒里也有一句。单 pass 的 capableC（v99）不同，在认人那一节说「换人」是声学层给的线索（云端任务说是转写模型给的），没标的接缝不说明是同一个人，认人仍以台词为准（[`llm_prompts.md`「单 pass 的拼法」](llm_prompts.md#单-pass-的拼法)）。
  - **`同时` 与 `换回`（v103，三个 pass 的 pass 2 是 3pass-v11）**：云端任务的行带分组键 `line_speaker`（`subtitles/line_groups.py`；同一个键的行不重叠，不同键的可以，见 [`vad-asr.md`](vad-asr.md)「已知缺陷：`*-aligned.json` 里的 segment 会时间重叠」），两个人同时说话时各自的行在时间上重叠。换人的接缝在这种任务里再分两种（`chunking.turn_seams`，只看键与时间，不看接缝从哪来）：本行开始时，此前键与它不同的某一行还没说完（**被盖住**，`line_groups.covered`），填 `同时`；没被盖住、往前找到同一个键的上一行 p 不是紧挨着的上一行，而 p 与本行之间的每一行都是在 p 还没说完时开始的（被插话的人在插话完了接着说），填 `换回`。回到前面说过的任何一个人都算的宽定义不用：两人一问一答时几乎每次换人都会是它。两个值与 `换人` 一样隔开两行（`Seam.separates`），合并规则与校验都按 `换人` 处理。带键的任务**每一窗**都在列说明里解释这两个值（`同时`：也是换人，这一行开始时另一个人还在说；`换回`：也是换人，是被插话之前那个人接着说，与插话之前那一行中间隔着别人的行，不能合并），单 pass 的合并规则、合并行检查与最后提醒写成「`turn` 列标了「换人」「同时」或「换回」的行」——窗里有没有出现都写，一个任务只有一份系统 prompt，前缀缓存不被拆成几份（owner 2026-10-31）。没有键的任务两个值都不会出现，prompt 与 v97 逐字相同。本地任务唯一的键是叠拉道（坍缩行压到下一行上的那一段，`overlap_lane`，见 [`asr-stabilize.md`](asr-stabilize.md)「坍缩行」）：有说话人标签时它压住的那一行标 `同时`（capableC 讲线索从哪来时仍说声学层：`prompt_compose` 的 `cloud` 只认说话人键）；「带键」按窗判，所以只有含叠拉道行的窗解释这两个值，这样的任务有两份系统 prompt（叠拉每小时只有几处，多暖一份前缀缓存的代价小，不为它把「带键」改成按任务判）。理由见 [`llm_design_notes.md`](llm_design_notes.md)「`同时` / `换回` 的决策记录」。
  - 判定：`labels.different_speakers`——两行都有主轨、在同一 block、主轨不同，`?` 不减弱它；行是 loader 留下的行、按阅读顺序相邻（按开始时间排，同时开始的按键排）。主轨取 `judgments.correction_speakers`：声学标签（云端任务是转写模型的说话人），裁判结果有效时套上裁判的改动，与 `spk` 列同源。`同时` / `换回` 再按键与时间从 `换人` 里分出来，盖过标签层给的接缝；键不同且时间重叠的两行（`line_groups.at_once`）不管标签怎么说都是 `换人`，于是裁判把附和改到长句那条轨上时，它照样标 `同时`（单 pass 与三个 pass 的 pass 1 同一条，owner 2026-10-31）。
  - 写在 `SubtitleSegment.seam` 上（`chunking.Seam`；没有标签的任务全为 `None`）：`load_segments_from_stable_json` 读行时按当时磁盘上的标签与裁判结果填 `换人`、`同时`、`换回` 或空，dry-run、fast、session replay 都经它；正式运行里裁判在排窗之后才跑，所以裁判之后由 `load_product_segments` 读成品层的行、`chunking.product_windows` 按排好的窗重拼，接缝跟着重填。
  - 三个 pass 的 pass 2（何时走、续跑与记录见[三个 pass](#三个-pass)）：`grouping.py` 按 pass 1 的 `join` / `may_join` / `drop` 把成品层的行分组（判决挂在裁判层的行上，`grouping.line_acts` 换到成品层：整行与前半段用原行的，后半段与补出的行没有、各自单独成一条，补出的行之后那一行只留 `drop`——不隔着它 join、不标可并；排不进空当、还不是一行的补行（`deferred`），它的人与前后任一行不是同一个人时，后面那一行同样只留 `drop`（并起来会把这次换人藏掉），同一个人或分不出时照 pass 1 的判；join 接的是上一条保留下来的行，中间隔着 drop 的行时，夹在中间的补行同样算跨过），分组后的一行取第一行的 id，`Grouping` 记它代表的原始行；第三种取值 `Seam.MAY`（`可并`）标 pass 1 判为同一个人接着说、pass 2 可以并的接缝。脚本把关：join 不跨两个人、不跨窗口的第一行（第一行被丢时由下一条留下的行顶上），并完超过一般门槛的秒数或合并源上限的 join 改标可并，两行合起来超过绝对门槛的秒数或换了人的不标可并（两道挡按任务的长度比例，见[合并门槛与任务的长度比例](#合并门槛与任务的长度比例)）。`execute_correction_windows(grouping=)` 收到的窗口已是 pass 2 的（分组后的行，或回退单 pass 的窗），产物（`-annotated.csv` 的 `position`、`spk` 列、SRT）按原始行写回（分组窗的行在提交时就展开回原始行，拼接都按原始行做：后面接一个回退窗时，它重做的重叠行按原始行顶替）；pass 1 丢掉的行与模型自己 discard 的行一样不出现在产物里，核心行全被丢的窗口不跑；只读前文也按分组读。分组后的窗口带 `SubtitleWindow.grouped`，prompt 换成三个 pass 的 pass 2 那套（[`llm_prompts.md`](llm_prompts.md) 的「三个 pass 的 pass 2」：只并标着可并的行，输出六列、不写 `#` 注释，自己的版本号），窗口的第一行没有上一行可并，那一格留空；对半拆开的窗口与 agent 侧的校验器都带着这个标志。pass 1 缺了的窗整窗回退（`grouping.pass2_windows`）：纠错窗的核心行里有一行在 `read_acts` 给的「没有 pass 1 的行」里（读不到可用的 sidecar 时是所有行），这一窗原样用单 pass 的 prompt，它的行的 pass 1 回答一概不用。只有纠错窗口自己的行渲染这一列，只读前文、规划预算、知识库素材都不带（规划不随标签变）。
  - 三个 pass 的 pass 3（校对，`execute_correction_windows(proofread=True)`；开关与续跑见[三个 pass](#三个-pass)）：窗口拼完之后先把 pass 2 的产物写成 `-pass2-annotated.csv`、`-pass2-translated.srt`、`-pass2-corrected.srt` 与后处理过的 `-pass2.srt`，再校对，最终产物（`-annotated.csv`、三份 SRT）带上它的改动。校对读观众看到的版本（「读不完」的门槛 9 字/秒取的是 Netflix 简体中文的成人读速，只管 10 字以上的行）：每行在屏幕上的秒数取 `-pass2.srt` 的（补过尾之后；行数对不上就不校），按这次的纠错窗分（一行归第一个含它第一个源的窗，即拼接时留下的那一版；补出的行跟着上一行），窗前窗后各带两行只读的行；每窗还看到准备任务给它的专属背景（按源序号选，与纠错窗选法相同，`ContextPack.window_context_for_source_ids`）；任务有转述的 `user_requests`、或这一窗的专属背景里有写明「出自用户」的要求时，多一段用户要求（分全片适用的与只关这一段的；照要求改写的译文可以变长，读速超线也不按那条改短）；每窗还看到它纠错时注入过的知识库词条里的专名与术语行（这次运行提交与回放的窗口记录的 `injected_entries`，`--no-resume` 时也有；拆开的半窗合在一起；按那一窗纠错时的知识库版本读，别的任务之后改了知识库，已答好的窗不因此重问；读不到知识库就不带、告警）。每窗一次调用（`correction-text` 组，纠错那档思考；`continuity` 为 serial 时一窗接一窗，否则按窗口并行度），回答只写改了的行，五列 `序号|类别|纠错文|译文|理由`（`stages/proofread/answers.parse_changes`：序号在窗内、不重复，类别 1–6，纠错文与译文非空；正好五格，哪一列里多出半角 `|` 都整份不收——纠错文或译文里的 `|` 会把后面的列整体挪位、写进成品也查不出，prompt 本来就不许理由用 `|`，输入里的 `|` 也写成全角；哪一行里又写了 `<proofread>` 就整份不收——分不清答案从哪开始，那一行之前的草稿会被当成答案）；读不了就带上原因重问，首次之后的轮数与纠错窗口的修复轮同一个上限（`max_retries_per_window`，默认 3；agent 会话在一次调用里做的修复轮、临时错误后的重试都从这里扣），额度类错误停掉整个 pass 3。写回只换纠错文、译文与随译文重算的字数，源、类型、时间、把握与备注不动，所以行数、`position` 与时间轴都不变，SRT 重新后处理。每窗答好的回答按它拿到的输入的哈希存进任务目录的 `proofread.json`（`_input_digest`：窗里与前后只读的行、标题、准备任务的整份通用结论、转述的用户要求（有要求时）、这一窗的专属背景（有时）、风格、它的知识库词条，不含 prompt 的措辞），重跑只问没答好或输入变了的窗（前后的只读行变了也算）；只改了模板措辞或升了 `proofread-v*` 时已答好的窗照用，回答记着答它的版本；照用的回答不再按现行的 `parse_changes` 重验（存的是读出来的改动，窗与行数都在哈希里，序号不会越界；以后收紧了解析，要旧回答重问就删掉 `proofread.json`），沿用了别的版本的窗告警一次（`proofread-older-prompt-windows`，补救是删掉 `proofread.json`，当前版本答过的窗也会重问）；没答好的窗保留 pass 2 的行，整体记 `failed`，其余窗的改动照用；pass 3 整个跑不起来（例如对不上行数）时最终产物就是 pass 2 的，`proofread.json` 也记成 `failed`，不留上一次的 `done`。没有任务目录时这些文件写进由输出名推出的 `.llm-artifacts` 目录，与窗口的切片同处。纠错文与译文里的换行按 annotated 的写法（`\n`）发出去、读回时还原。改动与理由记进 `proofread` artifact（`done` / `failed`）。agent 路径注册了同一套检查的 `proofread-window` 校验器。
  - 校验：合并行里相继的两个源之间（后一个源本身算在内）有任何一行标了换人（`同时`、`换回` 同样算），就把这一行的源序号记进 `CsvValidationResult.turn_violation_rows`（不进 `errors`、不影响 `ok`；校验之后映射回源序号）。现在合并源必须连续，这就是看后一个源自己的标记；以后为时间重叠的字幕放开跳行合并时，这条只会更严，不会放过换人。它自己不发修复请求：重试循环照收这次输出、告警 `correction-turn-merge`，窗口记录的 `turn_violations` 按源序号留下是哪几行，修复轮的「顺带」说明写出那一行标的是哪个值（owner 2026-10-16：生产里跨人合并不算需要修复的错，标记只靠 prompt 起作用；10-15 起是修复轮、重试用完才接受）。分组后的窗口（三个 pass 的 pass 2）换成「后一个源没标可并」就记，说明与告警也说没标可并。这一窗若因结构上的错反正要走修复轮，这几行以「顺带」的说明（`turn_notes`，模型的本地序号）附在校验错误后面一起发回，让模型在同一次重交里拆开（owner 2026-10-16：修复轮本来就要重交整窗，顺带改掉代价很小；没有别的错时不为它单发一轮）。agent 侧的 `correction-window` 校验器（lint 与 MCP 提交）同样不把它当错误，只在一次提交因别的错被退回时把同样的说明附上；缓存回放只看 `ok`，照常复用。
- `<preceding_context>`（v13）：窗口前最多 10 条 **raw ASR** 只读前文，与 `<asr_result>` 并列、使用同一时间基准；按时间顺序编号为 `1-M..0`，最近前文恒为 0。0/负数不属于输出范围，误引用会命中未知序号校验整窗重试。目标语侧连续性由 advice 台账与 context_pack 承载；advice 禁止携带无窗口命名空间的局部序号。请求回合与交付轮是同一条对话，前文块在首条消息里。
- 通用背景（`general_context` JSON；分块调查时按本窗所在的块选，见 [`llm_harness_research.md`「超长素材分块调查」](llm_harness_research.md#超长素材分块调查2026-08-30-落地plan-w7--p8)）与按 source-id 覆盖关系选出的本窗口背景，来自背景调查。通用背景里的 `user_requests` 是准备任务从 `--extra-info` 转述的用户要求（分块调查时各块的按原文去重求并集，每窗都给，不占通用背景的 20k），专属背景里写明「出自用户」的也是；这一窗的背景里有这两种之一时，纠错 user 模板多一段（`fragment_user_requests_v1.md`）：总体上照做，写法与格式要求用在 `translation`（只纠错是 `polished`）；用户给的原文（例如歌词）只作依据，与录音里实际的不一致时内容照实际、格式照用户；上下文明显需要时可以变通；内容照实际写了、或做了变通的在 note 写一句，照要求改写法本身不写；翻译任务带了 `--extra-style` 时，其中的风格要求与它冲突听它（owner 2026-11-08）。两块放在直接 ASR 块之前以提高 input cache 复用效率。旧 payload 中的 `audio_file`、`chunk_id`、`segment_range`、`boundary_reason`、`overlap_source_ids` 和 token budget 均不再发给模型。
- `<previous_advice>`：仅 `continuity=serial`；此前所有成功窗口 `<next_advice>` 的累积台账，
  请求回合和交付轮都能看到，整体≤8000 token，超限从最旧条目截断。parallel 不注入该块。
- `<entry_details>`：仅知识开启；来自准备任务的种子、local 请求回合的请求，或 parallel 屏障固定集的
  本地词条全文，可为空。
- `<search_results>`：仅 `retrieval=local` 的请求回合产生（fast 单窗同样），由本地搜索代理渲染、只随回填消息注入；首条消息里没有搜索结果块
  （只有 replay 工具冻结了搜索结果、重打交付轮时才渲染，`fragment_search_results_input_v1.md`）。
  请求回合的 `<window_notes>` 留在被回放的 assistant 消息里；首条消息的 `<pre_round_notes>` 槽恒为空、整段不渲染（只有 replay 工具重打交付轮时还填它，才出现）。
- 原生搜索：`retrieval=native` 不注入本地 `<search_results>`，而由纠错调用自身启用工具。
- **外来文字里的闭合标签**：网页摘录与 URL 提取（`web_search` 渲染每个查询 / URL 的那一段时）、模型读到的知识库条目（`render_subject` 的 prompt 投影；共享拉来的、从素材或网页收进来的条目都是别人写的字）、`knowledge ingest` 的素材、共享审核里贡献者的内容，嵌进 prompt 前把其中每个 `</名字` 换成全角的 `＜/名字`（`output_tags.neutralize_closing_tags`），这样它关不掉自己所在的块、冒充后面的 prompt。没有闭合标签的内容逐字不变，所以不升 `PROMPT_VERSION`；用户看的条目（human 投影、`knowledge/rendered/`）不改。agent 工具会话里的检索结果是 JSON 工具返回、不进标签块，不经过这一步。
- **任务 recap**：自 v9 起，每种 session 的 user prompt 末尾（payload 之后）都有一段 2-3 行的静态"最后提醒"，重申任务目标与输出格式关键约束（Gemini 长上下文最佳实践：指令重申放在大段 context 之后）。纠错与准备任务的 user 模板都有；回填消息末尾另有一句交付提醒。

模型随后输出 translated 终稿。tag parser 只认第一级同名块。输出结构按 variant 区分：所有变体都不写 `duration`、`gap`、`char_count` 三列；basicA（v104 起）使用带 `start` 的七列 CSV，capableC（v99 起）与三个 pass 的 pass 2（`capableC-grouped`）使用六列 CSV：

生产由任务组×difficulty 格子显式给出 variant（出厂纠错：quality、intermediate→capableC，
efficiency→basicA）；provider fallback 不再改变 prompt。纠错 live、窗口 resume cache
与 session replay 共用 variant-aware validator，每个变体只交一个 `<translated>`：basicA（v101 起）一行对一个源、只接回词中切断，带 `start` 的七列契约（v104 起）；capableC 与分组版六列；各变体绑定各自的 CSV 列布局。旧记录没有 variant
时才用 capability tier 作兼容回落；新记录逐窗保存实际 variant 与 difficulty。

```text
type|position|start|corrected_text|translation|conf|note
type|position|corrected_text|translation|conf|note
```

- `type`：留空或 `sub` 表示默认行为（拼合/纠错/翻译，`position` 填源序号）。~~`insert` 插轴已于 v63 全面废弃~~（旧模板已移至 `legacy/`，gitignored 不随仓库分发）。多个源片段合并时源序号用英文逗号连接，例如 `sub|3,4|good morning|你好|high|ASR 错分`（六列）。
- `start`：basicA 以 CSV 列携带；单源抄输入 start，合并行抄首源 start。解析只校验存在和数值类型，最终时间轴仍按映射后的稳定源序号回填（合并行从首源开始，到末源的结束；云端任务里合并源中键与末源不同的某一行结束得更晚时到那一行的结束，`line_groups.run_end`——只有违规跨 `同时` 的合并才会这样；分组键取首源的）；抄值准确率仅作能力观测。
- capable 一族（单 pass 的 capableC 与 pass 2 的 `capableC-grouped`）不写行前的 `#` 推理注释（owner 2026-10-23：中档能力以上的模型用不着辅助列与注释，`note` 留着主要为可解释）：合并超过一般长度要特批的理由，写在该行 `note` 里，用一句短结论；`note` 照旧只写短结论和检查项，没事可写就留空。单 pass 的 capableC 的合并行 note 另写为什么是同一个人接着说（对应 pass 1 的 `why`）。没有 prompt 要 `#` 行了，模型若还写，validator 照旧跳过（不算错），不进入 SRT。名字里的 C 原本表示「带注释」，名字没改（`prompt_variants.py` 写明）。
- `gap`：输入 ASR CSV 的 gap 是上一行结束到本行开始的秒数（csv-v111 起；与上一行重叠记 0，首行留空，见[Prompt 信息](#prompt-信息)）。输出里没有这一列，`duration` 也没有。
- `conf`（v39）：`high`（very certain）/`median`（likely correct）/`low`（better to manually check）三档自评信心；旧缓存中的 1–9 数字仍会兼容映射为三档。这一列有用（2026-10-30 离线读数：三个 pass 的成品里，看不到这一列的评委列出的实质错误落在 `median` / `low` 行的比例是 `high` 行的 4–8 倍——`high` 2.5%、`median` 11.5%、`low` 17.6%，四个模型方向一致；标了的行占 7.7%，装着三成的错），所以三个 pass 的校对把标了 `low` / `median` 的行的备注一律当猜测，补改也要过听错那道门（proofread-v5）。`char_count`：模型不写这一列，按长度规则旁边写的加权算法（拉丁/数字/标点/空格=0.5，其余可见字符=1）自己估。本地的加权字数都按译文当场算，用在合并行的绝对门槛（见[重试与拼接](#重试与拼接)「当前本地校验」）、pacing 打分（`score_translated_segments`）、校对 prompt 的「字/秒」列，和写 `<stem>-annotated.csv` 时的 `char_count` 列（给人看，没有代码读这一列；校对与知识更新都不看它）。模型以前写这一列时，本地拿它和复算值对照、不一致写 warning，另有一条「整窗系统性少报」的告警；没有变体再写这一列以后，这套对照与带辅助列的十列 / 九列契约于 2026-11-07 一起删掉。那条整窗告警防的传输事故（校验器看到的文本不是模型写的）另有整份拒收兜底：回复是纯 ASCII、却带着解码成非 ASCII 的字面转义时直接判失败（`output_protocol._refused_reply`）。统一公式由 `finesub.subtitles.metrics.weighted_char_count` 定义，也用于通用 SRT 行宽汇总；它只衡量字幕显示长度，与 token 预算及 ASR 异常检测用的 `finesub.text.count_word_units` 相互独立。`note`：自由注记，是最后一列；prompt 要求文本中的 `|` 写成全角 `｜`，解析器不宽容 `note` 里的半角 `|`（见[重试与拼接](#重试与拼接)「当前本地校验」的列数一条）。
- 统一入口是 `output_protocol.validate_correction_window_output`：它先按 variant 校验窗口局部 CSV，再把有效 `position` 与 discard 序号映射回稳定源序号。parser 对 type/note 宽松，`conf` 非法只告警不失败（仅供参考，从不单独判行失败）；漂移行靠数格子拦住，格数与 header 不同的行都判错。结构性错误（列数不符；未知/乱序/重复源序号，含 discard 与普通行之间的冲突；意外 start 列；insert 行；空文本）判失败触发重试。⚠ **不要再加「译文行数＝输入条数」断言**：把多个源合并到一行是合法输出，而行数对得上也抓不到语义错位（反馈包 #22 就是结构全对、内容串行）——这里保的是**结构**（每个源序号都被交代），内容对不对要靠别的信号。
- **保留数下限 `KEPT_FLOOR`（owner 2026-09-30）**：一个窗口里没被 `discard` 的源序号数 ≤ `min(10, 总数×0.5−10)` 判失败，进修复轮。它不是新规矩，是把「全部 discard → 无有效行」这条既有判据从 100% 边界挪开——**coverage 只问源序号有没有被交代，不问窗口有没有产出字幕**。起因是 2026-08-22 canary：一行 `sub` + 其余全 `discard` 通过了全部结构校验，成品只剩一条。
  读法：40 条以上的窗口是「最多只留 10 条就拒」；不到 40 条时门槛随总数降，20 条及以下不再触发（只剩「全部丢弃」那条）。
  它取代了 2026-09-03 的 50% 丢弃比例上限，那条有两个问题：没有小样本保护（一段大半是歌的 1 分钟短片，6 句里丢 4 句是对的），也用不到拆分后的半窗上（归档里最坏的正确半窗丢 43.8%，见 `bench-baselines.md` [二十五 纠错窗口的丢弃比例](bench-baselines.md#二十五纠错窗口的丢弃比例05-的标定以及为什么它不能用在拆分叶子上)）。绝对下限没有这两个问题，所以**整窗和半窗一律适用**（owner 2026-09-30 确认）。正确输出里丢得最多的整窗也只丢 21.9%，离这条线很远，所以它是错误探测器而不是质量旋钮。
  ⚠ 仍是一处**写明的缺口**：没读正文、但留下超过下限条数的回复照样通过；补它要的是**另一种信号**（模型到底有没有收到窗口正文），已记在 `llm_followups.md`。
  这一条同样管着 `agent-task lint`（同一个 validator，同样的范围），所以看不到正文的 agent 在提交前就现形。
- **行尾 `<void>` 自弃标记（v12 起）**：模型写完一行才发现不对（时长失控、分组/取舍错误）时，可在行尾追加 `<void>` 废弃整行并另起重写。解析时带标记的行在一切结构检查**之前**剥离（内容再破也不报错），其源序号可被后续行重新使用；数量计入 `CsvValidationResult.voided_rows` 并写进 `correction_window_response` artifact（用于观测模型是否真的使用该通道）。全部行都自弃且无其他有效行时按"无有效行"判失败重试。
- 只有 `translation`（纠错 SRT 另用 `corrected_text`）进入 SRT；`type`/`duration`/`gap`/`conf`/`char_count`/`note` 留存在 `<stem>-annotated.csv`（10 列，含 `spk`；`duration`、`gap`、`char_count` 在那里本地算出，`gap` 与输入同向：上一行结束到本行开始）。行时间轴按源序号从 `*-stable.json` 回填。知识更新阶段另会 overlay 最终 SRT 时间轴，生成含 start/end、不带 `gap` 的 8 列 `<final_csv>`。
  insert 的发射/去重/合并代码已于 2026-08-07 删除（两个生产调用点早已 `allow_insert=False`，整条路径不可达）；`KIND_INSERT` 常量保留，因为知识素材仍要从旧 `annotated.csv` 里读回该类型。

如果某段高度疑似 ASR 幻觉（含套话式幻觉）或无意义重复：所有变体都须在 `<translated>` 以 `discard|<局部序号>` 显式丢弃，不能静默漏掉（否则 coverage validation 失败）。v12 起 prompt 侧取向为**拿不准时保留并在 note 标记「疑似幻觉」**（人工删一条错留的幻觉比重听补一句被误删的台词便宜）。v16 起幻觉判定/保守保留/丢弃与插轴取舍收拢为独立的「幻觉与丢弃」fragment（`$hallucination_block`，取舍子句仅音频路线注入），套话幻觉另有独立反例（输入/输出块形态）；处置措辞仍按模态参数化（`$hallucination_handling`/`$noisy_span_handling`：音频路线以重听裁决，纯文本路线默认保留、禁止凭空"还原"台词），按类别特征描述（套话语域＋上下文脱节＋无对话区间）而非具体短语黑名单。单 pass 的 capableC（v99）没有这一节：整行丢弃按 pass 1 的 `drop` 判据（单独成行的垫词、确信的套话幻觉），一行里无意义的复读按契约「字幕节奏与排版要求」第 4 条；判据的保留清单只管认人这一步，纠错时口语颗粒压缩后不剩任何信息的照翻译目标第 4 条丢，套话只在判据的三点同时满足时丢（别处的「高度疑似幻觉」不放宽它）。

每行首先是字幕显示单元/时间单元，不是逐词语义对齐单元。`translation` 可以为了中文语序和阅读节奏，在同一连续语义单元内相对 `corrected_text` 前后错位；勿为对齐强行并成长字幕。basicA 只接回词中切断的两源；pass 2 的三源例外仅限 filler 三明治，禁止四源以上；单 pass 的 capableC 一条最多并 4 行，每处接缝都要是它判成同一个人接着说的。

`continuity=serial` 时，`</translated>` 后要求一个可为空的
`<next_advice>...</next_advice>`：只写本窗新增/修正的术语、说话人状态、未决指代等，
每窗≤800 token。Harness 按窗口 id 入账并注入后续窗口，台账整体≤8000 token；漏块按空
建议处理。`continuity=parallel` 时 system/user 都不提该块，也不解析或累积 advice。

`retrieval=local` 的纠错调用不开工具，查证由同一对话里的请求回合 + 本地搜索代理完成；
`retrieval=native` 的纠错调用自身启用原生搜索；`none` 两者都没有。所有本地 query、provider
和结果元数据都会写入 task artifact。

当且仅当采集反馈时（`--knowledge update`，或参考资料导入），纠错 prompt 会额外要求最后输出
`<task_update_feedback>`（v3：`knowledge_hints` + `asr_corrections` + `uncertainties`）。
它前面的尾块按实况变化：serial+local 可有 `<next_advice>`、`<keep_entries>`；serial 非 local
只有 advice；parallel 两者都没有。feedback 的 `source_ids` 使用本窗口正局部序号，落盘前
映射回稳定源序号；解析失败只告警不重试，也不进入最终 SRT。准备任务的交付同样采集
`research_task_feedback`。该档位纳入纠错 resume 的 task fingerprint。

## 合并门槛与任务的长度比例

几行并成一条字幕的上限按**任务的长度比例**缩放（owner 2026-11-04）。比例依次取：显式的 `--split-length-scale`、这次所用风格的
字幕长度档位（`[字幕长度] 偏短 / 标准 / 偏长` = 0.8 / 1.0 / 1.25，解析见 [`knowledge.md`「style 类别」](knowledge.md#style-类别2026-09-02)
的字幕长度档位一条）、`config.toml` 的 `[segmentation] length_scale`、1.0；范围检查同 `settings.SPLIT_LENGTH_SCALE`（0.6–1.6）。
ASR 分句读同一个设置，但看不到风格，所以两边的比例可以不同：偏短的风格只能让 LLM 少并，切不得比识别更碎。

- **哪些数跟着缩放**（`prompt_constants.Thresholds.at`，比例 1.0 时与 v105 逐位相同）：一般门槛（20 字 / 4 秒）、绝对门槛（36 字 / 7 秒）
  与 filler 三明治（16 字 / 4 秒）的字数与秒数；字数四舍五入（half-up），秒数取到 0.1，整数秒照写成 `5`。三档的值：

  | 门槛 | 偏短 0.8 | 标准 1.0 | 偏长 1.25 |
  | --- | --- | --- | --- |
  | 一般门槛 | 16 字 / 3.2 秒 | 20 字 / 4 秒 | 25 字 / 5 秒 |
  | 绝对门槛 | 29 字 / 5.6 秒 | 36 字 / 7 秒 | 45 字 / 8.8 秒 |
  | 三明治 | 13 字 / 3.2 秒 | 16 字 / 4 秒 | 20 字 / 5 秒 |

  不缩放的：垫词本身多短（`filler_max_chars`）、`basicA` 接回词中切断的两个数（`splice_*`）、一条最多并几个源（`fused_max_sources`）——
  它们说「这是什么」，不是「多长算太长」；校对的读速线（读得多快是生理）；单源行（本来就不受门槛约束）；`output_protocol` 只记录的
  节奏评分；efficiency 的 `basicA`（它渲染的数里没有一个缩放，变体的 `follows_length_scale=False`，按标准档渲染）。
- **模型看到的数与校验拦的数是同一组**：比例是窗口的字段 `SubtitleWindow.length_scale`，每个渲染与校验点按它取
  `Thresholds.at(window.length_scale)`——prompt 的门槛参数与示例的 `{thr:}`、单 pass 的形状判据、pass 2 之前分组的两道挡
  （`grouping.group_lines(thresholds=)`）、单 pass `capableC` 的绝对门槛校验（首次校验、续跑重放、并行与串行的缓存校验、agent 的
  `submit` 校验、补丁修复），以及 `tools/session_replay`（按回放那次任务记下的比例，没记的按 1.0）。字段默认 `None`，渲染与校验一进来
  就取门槛，`None` 报错、不静默按 1.0：窗口在风格定下之前就切好了，比例由执行路径在 `resolve_correction_plan` 返回之后统一写进每个
  窗（新计划、续跑重建、fast 的 `windows_override` 都经过这里），之后切半、分组、`product_windows` 从上游窗继承；dry-run 由
  `build_prompt_artifacts` 写进它自己切的窗。agent manifest 的 `serialize_window` 带着它；窗口的输入哈希含比例，比例不同的窗不当同一窗重放。
- **示例**：示例输入的每行时长按比例缩放，行间停顿不缩放（停顿规则不缩放），带数的判语由 `{calc}` / `{thr:}` 算出；前文按它最后一行的
  结尾、每个块按它第一行的开始锚定，两边仍在原处相接（`example_builder.scaled_rows`、`_scaled_example`）。字数不能缩放，所以示例里的合并行
  按偏短档写，测试在三档下逐条核对合并行都在该档的一般门槛内。
- **pass 1 的 prompt 三档通用**：说话人裁判的形状一节照标准档写「接完大约 4 秒以内」（`prompt_constants.PASS1`、`shape_criteria("pass1")`），
  比例不同也逐字相同，所以 `-speaker-judge.json` 换了档位照样复用。只有脚本在 pass 2 之前的两道挡按任务的比例：并完越过一般门槛秒数的
  `join` 改标可并、两行合起来越过绝对门槛秒数的不标可并。偏短时 pass 1 照旧按 4 秒接，3.2–4 秒的由挡改成可并、交 pass 2 按偏短的
  门槛决定；偏长时更长的句子本来就标可并，由 pass 2 按 5 秒去并。
- **什么时候定、记在哪**：风格定下之后、pass 1 分组之前（`length_tier.task_length_scale`，`correction_translation._with_length_scale`），
  档位从实际注入的那一版风格读；与三个 pass 的选择记在一起：`correction-mode.json` 的 `length_scale` 与 `length_scale_source`
  （`explicit` / `style` / `config` / `default`）。单 pass 的任务同样记。`--research-only` 用不到门槛，不记。
  - **续跑**：记下的比例压过一切，`config.toml` 或风格的档位改了也照记录；显式给了不同的 `--split-length-scale` 告警
    （`length-scale-recorded`，provenance 只告警、不失效）。理由：续跑时分组按 pass 1 的结果加门槛重新算、已交付的窗重放时还会再校验，
    比例一变分组就变，已交付的窗对不上被重做，或一个任务混着两种比例。要换比例就删 LLM 产物重跑（`--no-resume` 只对还没交付成品的任务起作用：成品在就照样复用）。记录里没有比例的
    （比例出现之前写的）重新解析：那次改动把比例写进了每窗的输入哈希（`scale=`），比例出现之前交付的窗对不上、本来就重做。
  - **dry-run**：风格可能还没定。点名或只有 `default_style` 时用它的档位；`auto` 有候选时按其余各层解析，`plan.json` 的 `length_scale`
    写明比例、来源与「风格未定，档位未计入」。
  - **fast 窗口自选风格**时档位不参与，选中的档位比例不同只告警（`style-length-tier-window-pick`）。
- **入口**：`python -m finesub.pipeline` 与 `python -m finesub.llm.correction_translation` 都有 `--split-length-scale`（argparse 传 `None`），
  管线把它原样传给 LLM 层；没给时 LLM 层自己按上面的次序解析。

## 注入上限

下列上限由 Harness 在运行时强制（常量定义见 `src/finesub/llm/routing/config.py` 与 `src/finesub/llm/web_search.py`）；全部以 **token** 计（用任务的 token counter 计数，截断走 `token_truncate.cap_tokens`）。超出 prompt 侧上限时截断或丢弃；超出输入硬上限时报错。

统一注入预算公式（搜索结果、深度提取、知识库词条共用，`injection_block_token_limit`）：**整块 ≤ 该轮单位上限 × 2000 + 4000 token**；块尾追加"注入预算说明"列出受影响条目，`RenderedBlock.report()` 全量记入 artifact。块内怎么分，两类不同：

- **搜索 / 提取结果**：单 section（单 query 结果 / 单 URL）≤ 4000 token，按优先序装填，装不下的截断或连同其后全部丢弃（先到先得）。搜索循环拿截断 / 丢弃名单决定事实优先级，所以这一类不改。
- **知识库词条**（`render_knowledge_entries_block`，2026-10-02）：每条先按所属类别 preset 的 `max_entry_tokens`（三个类别都是 6000）封顶；知识更新预取（上限 60k）里有两类条目先分、每条要么整条进要么不进、合计最多半块——本次更新的 style 条目，以及超过软阈值（5000）、被要求压缩的条目（这类不受 6000 封顶：看不全就压不了）；装不下的退回共享分配、照常截，也不挂压缩提示（`docs/knowledge.md`「条目的长度」）；其余条目「最长的先让」——求一个水位，短于它的整条进，长于它的截到它，所以块满时只削最长的几条。截断**按行**：条目头（H1、简介、`档案`）与元数据节总在，其余各节按顺序逐行装，装不下的那一行及其后的都不进；条目头之和都装不下时按优先序从尾部整条丢弃。块末说明写到「哪个条目的哪一节少了几行」，report 另记每条被削的原因（`cap` 超过单条上限 / `block` 块满被分）。**不变式：块永远不超过整块上限**。被截掉的行在知识更新里不分句柄（`retain_injected_handles`），曝光只记真正注入的行（`signals.injected_node_ids`）。纠错 system prompt 里的 style 块按同一上限逐套截断。
- **词条瘦身**（开发者开关 `FINESUB_KB_SLIM=on`，默认关）：serial 窗口透传来的词条整条渲染超过 2000 token 时，改为
  条目头 + 术语名字索引 + 本窗原样说到的整行，另加一块读音相近术语的提示；预算渲染照旧作用在瘦身后的正文上。batched 窗口的
  请求回合可以把精简过的透传条目要回全文，回填里给全文。闸门、提示的匹配、
  依赖与读数见 [知识库契约「精修字幕怎么读」](knowledge.md#精修字幕怎么读)那一节里与术语预注入放在一起的「词条瘦身与音近提示」一段。

| 项目 | 上限 | 说明 |
| --- | --- | --- |
| 窗口规划 reserve | 72000 tokens | 规划每窗输入预算时扣除，覆盖窗口 CSV/媒体之外的全部内容（静态 prompt、context pack、建议台账、请求回合 notes、搜索块、词条块），最坏合计约 69k。research 与 correction 两侧必须同值（窗口 id 一致性）。 |
| 准备任务搜索 query | 8..16 条 | 本地执行硬上限；`min(16, 8 + sqrt(原始段数)//10)`（`research_search_query_limit`）。模型多出的 query 丢弃。 |
| 纠错窗口搜索 query | 8 条 | 每窗请求回合 `<search_queries>` 最多执行这么多条（Exa → Gemma4 → Tavily），再回填给交付轮。 |
| 搜索/提取结果渲染 | 单 section 4000 token；整块 `该轮query上限×2000+4000` token | 统一预算公式。软上限：单条 snippet/answer 600 token、单 URL 提取内容 1800 token（section 内部的排版控制）。 |
| 知识库词条注入 | 单词条按类别 `max_entry_tokens`（6000）；整块 `条数上限×2000+4000` token，块内最长的先让 | 准备任务的 request≤8、keep≤8、keep-first 合计≤12（整块≤28k；下游无请求能力的档位 request 直接 12）；请求回合新请求与透传同样合计≤12；本地预注入按自己的上限。 |
| 本地关键词预注入 | 8 条 | 用户备注与 index key/alias 的 casefold 子串匹配，按频次排序；仅知识开启时注入准备任务的首条消息，直接交付时也留在种子里。旧的 text 直注纠错窗路径已删除。 |
| `--extra-info` URL 预提取 | 8 个 URL | 从 `--extra-info` 去重后的 HTTP(S) 链接；`retrieval=local` 时在准备任务首条消息前 deep extract，注入 `<note_url_extracts>`（`<search_results>` 文本），块预算 `8×2000+4000`。 |
| `analysis_notes`（准备任务的请求回复） | 1500 token | 解析后 Harness 截断；只是回填后交付的起点，不注入纠错窗。 |
| `window_notes`（纠错请求回合） | 800 token | 请求回合的预搜索分析；截断后进产物记录，交付轮通过被回放的请求回复看到它。 |
| `next_advice` | 800 token/窗；台账整体 8000 token | 仅 `continuity=serial`；按窗口 id 累积并注入后续窗口（含拆分叶）。parallel 完全撤除。 |
| Prompt 输入硬上限 | `prompt_input_limit`（免费 Gemini 上 194000） | ✱ **不是常量**（2026-09-03 起）：由绑定组的 catalog 行算出，`min(max_input_tokens, context_window − 预留)`（预留=预计输出，2026-09-04）。准备任务与纠错窗调用 API 前走 countTokens；超出即硬错误（无 map/reduce）。 |
| 快速模式回填 reserve | 56000 tokens | 快速窗首条消息的输入门槛 = `prompt_input_limit` − 56000（`FAST_FILL_INPUT_RESERVE_TOKENS`），为回填轮的搜索结果（≤20k）+ 词条（≤28k）留余量。这几个上限是绝对值、不随组包络收缩，包络放不下时闸门按名拒绝；不为小上下文模型缩它们（owner 2026-09-30：这类模型已基本没有）。 |
| 纠错窗输出上限 | `output_limit`（免费 Gemini 上 65536） | 请求回合与交付轮共用（Gemini 3.x 上 thinking 与可见输出竞争同一预算）。 |
| 准备任务输出 | **无 harness 上限**（请求打满答题模型自己的 `max_output_tokens`） | 请求回复与交付回复都是。`SESSION_OUTPUT_MAX_TOKENS`（32000，× `--output-scale`）自 2026-09-04 起只是**上下文预留**，不再当上限发出去。 |
| 知识库更新与维护命令输出 | **无 harness 上限**（同准备任务） | 知识库更新，以及维护命令 `knowledge repair` / `ingest` / `verify`、`knowledge.share conflicts --repair` / `review`，都只传 `SESSION_OUTPUT_MAX_TOKENS` 作预留。维护命令整份读回复，没有更小的单位可重试，所以一条回复碰到它请求的上限（REST 路径，输出 + 思考距上限不到 100）时整份拒收、不解析也不写库（`OutputCutOff`）：`repair` / `conflicts --repair` 一次处理多个条目时，前面已处理完的照常生效，命令列出它们后报错退出；`ingest` / `verify` 报错退出；`review` 把它当这一条回复不可用、释放 lease 后接着处理下一条。本地 agent 路径不发上限，不做这个判断。 |

## 三个 pass

开了说话人裁判的任务按三个 pass 分工走（只有 `quality` 与 `intermediate` 档能开），不开裁判的任务一律单 pass，有没有声学标签都一样。跑到 LLM 阶段的 `quality` 档默认开说话人裁判（见[说话人裁判](#说话人裁判--diarize-judge-text--judge-audio)），所以 `quality` 默认就走；默认档位 `intermediate` 不开说话人分离，默认是单 pass（owner 2026-11-08 把默认档位从 `quality` 改成它）。代码注释里的 owner D1、D3、D4、D9 等与它们的理由在 [`llm_design_notes.md`](llm_design_notes.md)「三个 pass 的决策记录」。

- **形状**：pass 1 是说话人裁判多答字幕的形状（每行 `join` / `may_join` / `drop`，见[说话人裁判](#说话人裁判--diarize-judge-text--judge-audio)）；pass 2 是纠错窗口，读按 pass 1 分组之后的行、只在标了「可并」的接缝上并（[Prompt 信息](#prompt-信息)里分组那一条）；pass 3 在窗口拼接与最终 SRT 的时间轴后处理之后校对观众看到的字幕，只改文字、不改划分（[Prompt 信息](#prompt-信息)里 pass 3 那一条）。
- **哪些任务走三个 pass**（`stages/three_pass.choose`，owner D1、2026-10-30）：开了说话人裁判（`--diarize judge-text` / `judge-audio`）就走，pass 1 是带轨号的裁判；档位要是 `quality` 或 `intermediate`（`correction_mode.THREE_PASS_DIFFICULTIES`：前两档给 capable 模型用），纠错那一格可能回答的每个模型的 prompt 集都要有分组版（只有 capableC 有，叫 `capableC-grouped`，窗口记录的 `variant` 写这个名字，`prompt_variants.has_grouped_variant`），否则报错（下一条）。声学说话人分离没跑成、行上没有标签时也记三个 pass，裁判以「没有说话人标签」跳过，各窗按单 pass 的 prompt 跑，pass 3 照档位。
  不开裁判的新任务一律单 pass，`correction-mode.json` 的理由记 `speaker judge off`：行上有声学标签（`--diarize acoustics`，或目录里留着以前写的 `-diar-rows.json`，`read_rows` 照样叠上标签）时，单 pass 的 capableC 自己认人、定形状，`turn` 列标着声学的「换人」（[单 pass 的拼法](llm_prompts.md#单-pass-的拼法)）。以前只有声学标签的任务也记三个 pass；这样记下的任务续跑时没开裁判，就没有 pass 1 可跑：模式照旧，还没纠完的窗按单 pass 的 prompt 纠错，告警一次（`three-pass`，`three_pass.pass1_shape`）。
- **开了裁判就要分组版**（owner 2026-10-30：开了裁判就按三个 pass 分工走，不再有「裁判 + 单 pass」）：`efficiency`、或纠错那一格没有分组版时开了裁判，开跑前报错（`three_pass.refuse_judge`），提示改用 `--diarize acoustics` / `off`，或换前两档、用有分组版的 variant。管线只在这次跑到 LLM 阶段（`--stage` 到 `translated-srt` 及以后）时查，在说话人分离之前，免得白花分离的 GPU 时间与模型下载；停在 `raw-srt` 的运行（含批处理里 ASR 那半）不查，也不因此加载纠错的路由层。单独跑纠错模块（`--speaker-judge`）时在入口查（`run_correction_task`），在已有成品的捷径与准备任务之前，`task_choice` 再查一次。续跑放行：任务记下的裁判与这次的相同（`task_diarize` 交回的，或照原命令又写了一遍），就照记下的走；额度用完后加 `--llm-difficulty efficiency` 续跑的就是这种。新任务、`--no-resume`、显式写的裁判与记录不同的续跑报错；走法在准备任务之后才记（`record_choice`），准备任务里停下的任务没有记录，按新任务查。已有成品的任务也一样：成品交回与给成品补跑裁判（`_judge_owed`）都在这道检查之后。
  `--diarize off` 不加新的值：纠错请求只收到 `speaker_judge`，标签有没有按数据推断，与纠错的换人列同一个推断。
- **pass 3 的开关**：`--llm-proofread auto|on|off`（单独跑纠错模块是 `--proofread`），默认 `auto` 看档位：`quality` 跑，其余不跑，不论走了几个 pass（owner D2，`correction_mode.PROOFREAD_DIFFICULTIES`）；只纠错的任务（中文素材，[只纠错](#只纠错)）不跑；`on` 各档都跑，`off` 不跑。`correction-mode.json` 的 `mode` 仍是 `single` / `three-pass`，跑没跑 pass 3 看它的 `proofread`。
- **任务的选择只定一次**（「resume 延续任务自己的产物与当时的选择」）：要跑窗口之前写进任务目录的 `correction-mode.json`（`mode`、`proofread`、理由、裁判的档、任务的长度比例与只纠错的模式，比例见[合并门槛与任务的长度比例](#合并门槛与任务的长度比例)，模式见[只纠错](#只纠错)），续跑照它走，开关或预设后来变了也不重推；目录里已有窗口计划或窗口缓存、却没有这个文件的任务，是三个 pass 之前开始的，按单 pass（pass 3 按 `on` / 不跑）。`--no-resume` 重新选。dry-run 与 `research` 模式不写它。
  记成三个 pass 的任务续跑时换了难度档或 variant、那一格不再有分组版的 prompt：模式照旧，窗口缓存里已提交的分组窗照常回放，其余窗按单 pass 的 prompt 纠错、告警一次（`three-pass`），不在 `grouped_variant` 上报错。
- **云端任务**：行上总有说话人（转写模型的 `line_speaker`，见 [`manual/joint-asr.md`](manual/joint-asr.md)），照上面的条件走。pass 1 的 `join` 与 `可并` 不跨「时间重叠且键不同」的两行（`three_pass._different` 对它们也为真，标签说是同一个人也一样），组仍是连续的一段；分组后的行的接缝再过一次 `turn_seams`，`同时` / `换回` 由 `group_lines` 给出，分组后的一行的结束取它的行里最晚的另一键结束（`run_end`）。pass 2 读到的时间就是分组后的行的；拼回原始行（`expand_sources`）时不重算，只在 pass 1 把两个键并成一组、第一个键之后的行又在第二个键的行还没说完时开始、pass 2 还跨换人并了它们这种组合里，合并行比按原始行算的早结束一点（`grouping.expand_sources` 的说明）。
- **pass 1 缺窗**：裁判是可选步骤，失败只告警；pass 1 没有回答的窗（单窗回答不合法、额度停下、声学标签分了多个 block 时裁判整个跳过、没有标签）按这次的纠错窗整窗回退到单 pass 的 prompt（`grouping.pass2_windows`），任务仍记三个 pass，pass 3 照跑。续跑时裁判的窗是上一次切的、join 跨过这次纠错窗的边界时在边界处断开、告警一次（`three-pass`），不重判。核心行全被 pass 1 丢掉的窗不跑，进度与完成判断按去掉它之后的窗口列表算。
- **窗口记录与缓存**：每个提交的窗口记录都带 `prompt_version`（分组窗 `GROUPED_PROMPT_VERSION`，回退窗与单 pass 的窗 `PROMPT_VERSION`，各自对照自己的常量）；三个 pass 时另记回退窗的 `pass1: missing`、分组窗里按原始行数超出合并源软上限的行（`over_cap_rows`：校验器看的是分组后的行，它的软上限警告按分组行数）。分组窗另记退化的迹象（`degeneration`，`degeneration.pass2_signals`）：本窗自己的行里标了可并的接缝数与回答并掉的数（`may_seams` / `may_merged`），以及触发的信号（`signals`：可并接缝不少于 30 处而并掉的不超过 5% 记 `unmerged`）。现在只记录、不重试；触发 `unmerged` 的窗在纠错阶段的摘要里列一项「可并几乎没并 N 窗（窗号）」，是提示不是 Warning。判据与去处见 [`llm_followups.md`](llm_followups.md)「模型回答的退化」。分组窗的输入哈希另带一个固定的分组标记，同一块的分组窗与回退窗永不互相回放；两个版本号都不进哈希，升版不作废已交付的窗（见[纠错窗口中途 resume](#纠错窗口中途-resume默认开启)）。分组窗的行带着分组（拼起来的文字、可并），所以 pass 1 补判或重判之后那一窗自然重做；单 pass 窗的哈希不带标记。
- **pass 3 的续跑**：任务记下的选择里开着 pass 3、上一次 pass 3 没做完（记了 `failed`：有窗没答好，或整个跑不起来、成品是 pass 2 的；或补跑到一半被打断，状态停在空），续跑不走「成品已在」的捷径（`_finished_output`），重新走一遍窗口：不调裁判（pass 1 当时缺的窗照旧留在单 pass 的结果），窗口全从缓存回放、不调模型，pass 3 只问 `proofread.json` 里没答好或内容变了的窗。`--no-resume` 时这条捷径照旧（成品在就复用），pass 3 一旦重跑就问每一窗、不用 `proofread.json` 里的回答。`--llm-proofread off` 的任务不写 `proofread.json`，所以不会被当作欠着 pass 3。
- **任务完成后补跑裁判**（`_judge_finished_task`）：三个 pass 的任务按 pass 1 的形状判（sidecar 仍是 pass 1 的，单元缺 `acts` 就算没判），只重渲 `-annotated.csv` 的 `spk` 列，不重新分组、不重跑 pass 2 与 pass 3；分组变了要重跑就删产物。
- **边角情形的处理理念**（owner 2026-10-17）：越难碰到、后果越轻（少量质量损失、多花一些 token、某几窗重做）的组合，越不为它写专门的规则，而是让下面几条通则自然给出结果，并在这里写明：
  - 产物在就读：`--diarize off` 不跑说话人分离和裁判，但目录里留着的 `-diar-rows.json`、`-speaker-judge.json` 照样喂给 LLM 阶段（单 pass 的 `turn` 列照样标出换人）；要回到不带说话人的纠错就删掉它们。
  - 续跑照任务的记录走，显式写的开关优先；改了开关让窗口的输入变了（标签出现或消失、裁判换了档），那几窗按新输入重做，是多花的 token，不是错误。
  - 可选步骤失败只告警，续跑时再试：说话人分离当时失败、这次成了，已交付的窗会因多出换人列而重问。
  - 不丢不可复现的东西：只差 pass 3 的续跑不调裁判，pass 1 当时缺的窗留在单 pass 的结果。
  - 真要专门处理的只有会丢数据、坏续跑、或在常见路径上悄悄花钱的情形。
- **快速模式**（owner D3）：三个 pass 照走，pass 1、pass 2、pass 3 各是整段一个窗。要跑 pass 3 时快速模式的判定多一条：pass 3 的输入估计（它的 system prompt 与素材说明，加 3 倍字幕 token：每行有识别原文、纠错文与译文）不超过 pass 3 那组的输入上限，否则不走快速模式（`--fast on` 报错）；`fast_decision` 记下这两个数。
- **dry-run**：pass 1 还没跑，pass 2 的 prompt 预览用分组版的 prompt 套在原始行上（没有「可并」），不出 basic 档那份；`plan.json` 的 `correction_mode` 写明选择、pass 1 的形状与「pass 2 的输入要等 pass 1」；行上没有裁判能用的标签（`three_pass.pass1_groups`，裁判会跳过）时预览单 pass 的 prompt，`pass2_input` 改说各窗按单 pass 跑；裁判的预览按 pass 1 的形状出。
- **记录**：`correction_mode` artifact 记选择与 pass 1 留下的（形状、窗数、回退单 pass 的窗、被跳过的窗、并了几行、丢了几行、被脚本按住的 join 按原因计数）；task report 的「Passes」一节读它与 `proofread` artifact，「Speaker Judge」一节在三个 pass 时多列各类 act 的个数，并逐行列出 pass 1 丢掉的行与理由。
  说话人裁判（不论是不是 pass 1）与 pass 3 开跑时各在任务目录写一份输入（`speaker-judge-input.json`：标题、每窗的概要、候选人物、模态、形状；`proofread-input.json`：每窗的行与源序号、只读的前后行、准备任务的 context pack 与源序、风格与知识库的名词），供 [`session_replay.md`](session_replay.md) 的 `speaker-judge` 与 `proofread` 重建 prompt；运行本身不读它们。

## 只纠错

中文素材跑到 LLM 阶段时只纠错、不翻译（owner 2026-11-05）：每行写两列，`corrected_text` 是忠实转写，`polished` 是在它上面的基础润色，成品字幕用润色版（owner 2026-11-06）。没有开关，由素材的语言决定；目标语言写死成简体中文，中文译成中文没有意义。粤语（`yue`）仍走翻译，口语转书面中文本身就是翻译。

- **怎么判**（`three_pass.task_correct_only`，任务开工的第一步，知识库、风格、已有成品的捷径与计划都跟着它）：素材的语言取转写时告诉 ASR 的语言（`asr_align.language`），没有就按段的 `lang` 以字数加权投票（`recall_repair.episode_language`，与召回修补同一个读法），是 `zh` 就只纠错。判不出（没告诉语言、段上也没有 `lang`）按翻译走，打一行 `Warning:`（分类 `correct-only`），提示转写时加 `--language zh`。读不了 stable 时按翻译、不告警，读它的那一步自己会报错。
- **记一次**：写进 `correction-mode.json` 的 `correct_only` 与 `source_language`（可选字段，`MODE_VERSION` 不升），续跑照记录走，素材后来重转成别的语言也一样；记录里没有这两个字段的任务、以及有窗口计划或窗口缓存却没有记录的任务，是这之前开始的，按翻译续跑。`--no-resume` 重新判；判成只纠错时，按翻译做过的准备任务产物不复用（复用键 `research_semantics_id` 带上模式，翻译任务的键不变），任务里按翻译记下的 `auto` 风格选择（快速模式的窗口挑的）点名没写 `[任务] 中文纠错` 的风格，不注入、告警一次。dry-run 不写记录，但 `plan.json` 的 `correction_mode` 写明两个字段，预览的就是只纠错的 prompt。任务的 profile 带着它（`TranslationProfile.correct_only`），`profile_id` 只在为真时多一段 `,correct_only=true`，翻译任务的 id 不变。
- **prompt 与输出**：纠错用三套派生的 prompt 集 `capableC-polish`、`basicA-polish`、`capableC-grouped-polish`（`prompt_variants.correct_only_name`）。client 选候选时把每个候选的变体名映射成派生名（`RoleClient.complete(correct_only=)`，按 target 覆盖的变体也映射），之后的窗口记录、窗口缓存、agent 的 metadata 与补丁修复轮都按这个名字解析，所以窗口缓存的身份自带模式。翻译任务的 prompt 逐字不变（快照钉着）。模型输出的第二列表头叫 `polished`（模型面前不出现「翻译」），解析后放在 `translation` 里（[`llm_prompts.md`「只纠错」](llm_prompts.md#只纠错)），之后的字数规范化、SRT 渲染与 `-annotated.csv` 看到的形状与翻译任务相同。整行只有垫词的行 `polished` 留空（校验放行，字数与长合并按 `corrected_text` 算）：它进 `-corrected.srt` 与 `-annotated.csv`，不进 `-translated.srt` 与最终 `*.srt`（`run._write_products`；从盘上收尾时 `_finished_keys` 也跳过它）。`discard` 只给套话幻觉；三个 pass 的 pass 1 也不再丢整行垫词的行。
- **两列各改什么**：`corrected_text` 只改听错的字词（同音、近音、专名）并补齐标点，垫词与口吃照原话录下（整行垫词也照写）。`polished` 是字幕用的那一版（owner 2026-11-06「再加顺语法」）：先照抄 corrected，再只做三种局部编辑——删垫词（拿不准是垫词还是应答、语气的留着）、口吃重复压成一次（整行是实词或应答的重复也压成一次，不留空）、修明显不通的口语病句（量词、语序、多字漏字）；每处只改出问题的几个字，没有这些问题两列相同；不换同义词、不另写一句、不补没说出的内容、带意思的字词不删，方言与台湾腔照留，拿不准是不是病句就不改。写成「照抄再局部编辑」而不是一列「基础润色版」加禁令，是因为后者在 quality 档的一份讲座里，一个窗口越写越偏，后半段改成书面与文学腔、近六成的行写进 corrected 没有的字，加了「不重写整句」的反例也没收住；同一段原话放在下一窗开头就照原样写（Codex Sol 的诊断：第二列被当成另写一句成品字幕，长输出里漂移）。拆成两列是为了要忠实原文的人在 corrected 里拿得到，模型也更不容易在 corrected 里顺手润色。`corrected_text` 里说对了的字照原话写：不顺语法、不为通顺补字删字（「的」「就」「了」；有音频时 ASR 漏转、听得出的字照听到的补）、不改量词与语序、不换同义词；分不清时看改法，换成同音、近音的字是纠错，为了通顺加字、删字、调语序是局部编辑、只在 `polished` 里做（单列只纠错、不写明这一条时，模型会把讲座口语顺成书面语：一份 30 分钟的讲座里 efficiency 判为改写润色的 33 处；两列时把这串禁令缩成一句「照录」，quality 也把润色式的改动带进 corrected：同一份讲座上读者判的改坏 9 → 27（各一次运行），多是删「的」「就」「他」、换「里边」→「里面」、改量词），夹在中文里的外语照原样写；繁体字改成简体，字形之外照原文的写法：语气词不换字（喔、欸、捏不改成哦、诶、呢；留不留照口语颗粒判定），台湾腔、方言的说法不换成大陆或书面的（空白、掰掰不改成空格、拜拜），听错的字照常改（不写明时模型会统一写法：一份台湾直播 30 分钟里，efficiency 把「喔」改成「哦」83 次）。
- **交付前机械转简**（owner 2026-11-06，`stages.correction.run._write_products`）：两列写进产物之前都过一遍 opencc t2s。prompt 要求简体，但模型会照抄手上拿到的繁体（一份讲座里准备任务把人名写成繁体，efficiency 跟着写了 28 处）；t2s 只换字形，不换说法（「萤幕」「喔」不变）。
- **风格只用写给它的**（owner 2026-11-05，`correction_translation._correct_only_style`）：档案节写了 `[任务] 中文纠错` 的风格（`style.CORRECT_ONLY_TASK`）整份注入，风格块的抬头说用户写的要求（例如加注）只用在 `polished`、优先于对它的限制与「只输出这些话」，`corrected_text` 照原话写；`--style auto` 只在这些风格里挑（`_auto_style_catalog`），挑选的措辞走槽。没写这个标记的风格不注入：命令行点名的只读它的 `[字幕长度]`（`StyleSelection.tier_names`），`config.toml` 点名的一概不读，各告警一次（`correct-only`）。出厂默认风格不注入（它的条目除 `[不加注]`、`[语气词]` 外都是译法，这两条纠错目标里已有同义规则），中文纠错也没有自己的出厂风格。不写回风格（`--style-mode update` 按 `read`，告警）；`--extra-style` 不用，告警。efficiency 照旧不用用户风格。
- **不读写知识库**（owner 2026-11-06，`correction_translation._with_task_mode`）：知识库的条目与知识更新都是按翻译写的，中文任务该存什么还没设计，所以任务按 `--knowledge none` 跑：不读条目、不注入索引、不要模型交知识反馈（参考资料导入的采集开关也关掉），不做任务后的更新，`--refined-srt` 也不用。默认的 `read` 不声不响地关掉，用户或预设给了 `read` / `update`、给了 `--refined-srt` 的告警一次（`correct-only`）。单独跑的知识更新（`finesub.llm.knowledge.update`）与参考字幕收录遇到记录为只纠错的任务也跳过、告警（`knowledge-correct-only-skipped`）。风格不受影响（`--knowledge none` 不管风格）。
- **pass 3**：`--llm-proofread auto` 不跑；显式 `on` 照跑现有的校对 prompt（只读进字幕的行，`polished` 留空的行原样留着）（它写的是日语原文对中文译文，效果没量过），告警一次。
- **规则里举的例子是中文的**（owner 2026-11-06）：认人与定形状的判据（停在哪类连接处算没说完、单独成行的垫词、附和、转场、语气词）、合并行 note 的例子、basicA 的词中切断例、分组版三明治中间的垫词，以及幻觉套话清单与它的边界（换成中文 Whisper 常见的套话：「谢谢观看」「请不吝点赞 订阅 转发 打赏支持明镜与点点栏目」「字幕由Amara.org社区提供」之类），都按模式换；纠错目标里写给日语的几句也换：专名被转成「音似的假名」改成「音似的别的汉字」，「生僻假名串」改成「生僻字串」，「修正口语停顿和重复」（纯文本版是「无意义重复」）改成说话人自己的停顿、重复不算 ASR 的错（corrected_text 照录）；准备任务里说专名被转成假名的两句也换。示范也是中文素材的（owner 2026-11-06），见 [`llm_prompts.md`「只纠错」](llm_prompts.md#只纠错)。
- **别的会话**：准备任务说到翻译的几句（角色、输出交给谁、「不做的事」、挑风格那一项）与定名一项按模式换：`entities` 与 `terminology` 的 `zh` 填字幕里该怎么写，中文专名写对汉字，说出的外语照原样，不换成中文名。pass 1 定形状一节里「下一步看什么」一句与判据的举例按模式换字句。说话人裁判的其余部分不变。
- **已知局限**（不改）：纠错 prompt 里说知识库的几句（词条定名、音近提示）也开了槽，但任务不读知识库，用不上。
- **识别这一侧**：Whisper 的 `zh` 不分繁简，stabilize 最后一步把普通话的段统一写成简体，不论跑不跑 LLM（[`asr-stabilize.md`「繁简」](asr-stabilize.md#繁简)）。

## 说话人裁判（--diarize judge-text / judge-audio）

实验与验收的结论在 `docs/diarization-findings.md`「裁判放在哪」「生产裁判的验收」。用户向说明在 [`docs/manual/diarization.md`](manual/diarization.md)「LLM 说话人裁判」。

- **开关与模态**：pipeline 的 `--diarize` 四档（`stages.DIARIZE_LEVELS`），`judge-text` / `judge-audio` 让 LLM 阶段的请求带 `speaker_judge="text"` / `"audio"`；单独跑纠错模块用它自己的 `--speaker-judge off|text|audio`。
  不写 `--diarize` 时由 `stages.default_diarize` 定（owner 2026-10-17）：跑到 LLM 阶段的 `quality` 档（`correction_mode.DEFAULT_JUDGE_DIFFICULTIES`）按纠错的模态取 `judge-text` / `judge-audio`，其余 `off`——`intermediate` 能走三个 pass，但默认不开裁判；任务用的档由 `stages.task_diarize` 读回：`correction-mode.json` 记着这个任务的 LLM 阶段用的裁判（`llm/correction_mode.py` 的 `PassChoice.speaker_judge`，与走法同时记，在窗口开跑之前；这个模块不 import LLM 层的别的模块，语音流水线读它不拖进纠错层），续跑照它走；没有这一项、而窗口或成品已在的任务是默认开之前开始的，取 `off`，否则窗口的输入多出换人一列、已交付的窗全部重问；`--no-resume` 重新选。批处理的一项先跑到 raw-srt、再跑 LLM 阶段，默认按整项的目标阶段定（`run_pipeline` 的 `_diarize_default`），否则前一半不跑说话人分离。没有只分出一个人就降档的一步：声学层会把一个人分成几条轨，见 `docs/llm_followups.md`「说话人分离的 `auto`」。
  模态只看这一个值，不看纠错的 `correction_media`：`audio` 时裁判调用走 `correction-mm` 组（`run.judge_group`），纠错照自己的组。开跑前用 `usable_endpoints` 判一次这组里有没有能听音频的模型——选路时没有能听的候选只会以「没有路由」失败、到不了降级——没有就整次按纯文本判、告警一次（`task.judge_media`）。
  目标阶段低于 `translated-srt`、或说话人分离没跑成时，pipeline 往 stderr 打一行（`_warn_if_the_judge_cannot_run`）。
  开了裁判就按三个 pass 分工走，裁判是 pass 1；`efficiency` 或没有分组版的格子开裁判在开跑前报错，续跑照任务记下的裁判放行，见[三个 pass](#三个-pass)「开了裁判就要分组版」。
- **位置**：`correction_translation._run_task` 里，准备任务之后、纠错窗口之前（`stages/speaker_judge/task.judge_task`）。路线一里两边的判决互不读取，只共用窗口边界：
  纠错窗口计划由 `resolve_correction_plan` 一次运行解析一次（读计划文件，指纹对得上就沿用；读不到就新算并写回；fast 模式用 `FastDecision.window`），交给裁判与纠错。
  ⚠ 别在同一次运行里再解析一次：第二次会读到第一次刚写的计划文件、当成复用，纠错账本就会拿已丢失的计划续跑。`mode == "research"` 不跑裁判。
- **读的行**：stable 的行一律经 `subtitles.stable_rows.read_rows` 读，标签来自旁边的 `-diar-rows.json`——开了 `--diarize` 且任务选了拆行时，被切开的行读作两行（见 [`docs/manual/diarization.md`](manual/diarization.md)「拆开两个人的行」）。这是**裁判层**：裁判判的、纠错窗口规划的、sidecar 的 `rows_hash` 与判决里的行 id 都是它。裁判之后读的是**成品层**（`speech/diarization/product.py`）：裁判层套上带音频的 pass 1 切开、补上的行，后半段记 `<原号>b`（owner 2026-10-16）、补出的行记 `<锚定行号>i`，每行显式带 `id`，与裁判层逐行有映射（来自哪一行、是整行、前半、后半还是补出的）；纠错（pass 2 的输入、`-annotated.csv`、知识库收割的 ASR 一侧）、`spk` 列与分轨导出读它，`-raw.srt` 仍是裁判层。纠错窗口在裁判层规划（计划文件里没有后半段与补出的行），裁判之后每窗把自己的行展开成成品层（`chunking.product_windows`）：切出、补出的行紧跟在原行后面、留在它的窗里，不会是窗的第一条核心行；窗的音频片段放宽到装得下它的每一行（补了半句的行可能提前开始，窗末补出的行可能超出片段原来的余量）。同一份 sidecar 总给出同样的成品层，续跑逐行一致；判决不再成立时成品层就是裁判层。
- **窗口**：一个纠错窗口的核心行（重叠区之后的行）就是一个裁判窗（owner 2026-10-09）；窗 id 就是纠错窗口的 `chunk_id`。前后带不判的上下文，长度同纠错跨切点带的重叠（owner 2026-10-09）：切点前 30 秒内开始的行（`chunking.overlap_count_for_boundary`，长停顿之后为 0），窗后镜像：本窗最后一行之后 30 秒内开始的行（长停顿之前为 0）。纠错的 prompt 不带后文块：窗尾那段下一窗会当重叠区再答一遍，但拼接用的是本窗那份（core 归属，见[重试与拼接](#重试与拼接)的拼接规则），所以纠错切点前的几行最终没看过后文；裁判一行只判一次，带上这段后文。两边不一致是有意留的：纠错补后文要改它的输入，记在 [`llm_design_notes.md`「纠错窗口拼接改成 core 归属的决策记录」](llm_design_notes.md#纠错窗口拼接改成-core-归属的决策记录2026-11-08)的「不改成「窗口尾部带只读后文、重叠区不再重答」」一行。窗前只取重叠，不取纠错另带的最多 10 行只读前文（`preceding_context_for_boundary`，长停顿处也不停），所以长停顿之后开始的裁判窗没有前文（owner 2026-10-09：上下文同纠错的重叠那一部分）。行数由行的时间算，sidecar 不记。
  第一次切好的窗列表写进 `<stem>-speaker-judge.json`，此后以它为准：纠错之后重新规划不重切、不重判；只有 sidecar 不在、版本不是 2、或行的 id / 时间 / 文字变了（重跑过 ASR）才按当前计划整份重切。
- **放不下时对半拆一次**（owner 2026-10-09：跟着纠错窗的分割，正常不特判；兜底照纠错的拆窗，见[重试与拼接](#重试与拼接)）。纠错规划已按纠错的输出公式卡过窗的大小，裁判每行的输出比纠错少（GPT-6.1 Sol 约 35 token / 行含思考，纠错约 90 / 源段），所以裁判不另估输出。三种情形对半拆：
  - API 用量贴着输出上限（`is_likely_output_limited`，主信号）：不再重试，立刻拆。agent 的用量是整个会话累计的，不拿来比。
  - 重试用完、最后一次回答 `<judge>` 开了没闭合（`answers.cut_off`，兜底，同纠错只看最后一次；agent 通道只有它）。
  - 开跑前按裁判自己那组的限额（`planning_limits_for`）估输入超了（主要防 `judge-audio`：整窗录音与样本）。
  切点取离中间最近的合理边界（`windows.halves`，规则同 `chunking.split_window_in_half`，不带重叠：半窗前后的上下文按同一规则取）。被拆的窗在自己那一项里记 `parts`（窗列表从不增项，并行写入靠下标），半窗 id 是 `窗id-a` / `窗id-b`，各有完整的重试次数。**只拆一层**：半窗再撞上限、输入还超、或重试用完时回答仍开着，就记为失败、退回声学标签（纠错在这里停掉整个任务；裁判是可选的，不拖垮纠错）。不记行数上限，下一个窗照常整窗发。
- **人物候选**：开了裁判时准备任务多交一个 `general_context.speakers`（`fragment_prepare_speakers_{duty,field}_v1.md`，要求轻：只凭标题、简介与台词直接看得出的，不另外检索；关着时 prompt 逐字不变）。它不进纠错窗口（`ContextPack.general_text_for` 去掉，分块调查的每块里也去掉）。候选顺序：`speakers` → 知识库主播（`roster.roster_entries`）→ `entities` 里的人物，按名字原文去重（`speakers` 里的人在 `entities` 里也有时，prompt 要它照抄 `entities` 的 `name`，同一人才不会列两遍），最多 30 个；整份候选直接进裁判 prompt，裁判可以写名单外认得出的人（owner 2026-10-09）。分块调查时候选读全部块，概要按裁判窗选块、本块在前。准备任务的复用不看开没开裁判：已有的调查结果没有 `speakers` 就只用另外两类候选。
- **短行闸门**（带音频，owner 2026-11-02）：不到 `SHORT_ROW_SECONDS`（1.5 秒）的行，判决挪到这一行格子（`cell_tracks`，与 prompt 显示的同一份）里没有的已有轨时改作 keep，不进 `results`，记进这个单元的 `gated`（行、轨、理由；`run._short_row_gate`）。新建的轨、接着用的轨（`carried`）、切行、补字、插行不挡，只有被挡的那一行切开、后半段又写被挡的那条轨时，这次切行进 `ignored`（同一次听错，owner 2026-11-02）；切行的「后半段与前半段同一人」按挡之后的判决比；纯文本不挡。只看时长、不看字数：字数跟语言走，而试过的数据里字数条件从没单独起作用。旧单元续跑时照旧沿用，不补挡。
- **每窗一次调用**（owner 2026-10-09；准备任务已经列出全片的人，不另挑人）：纠错同一个思考档，组按裁判的模态。
  回答读不了走修复轮，做法同纠错（owner 2026-10-31，见[重试与拼接](#重试与拼接)「修复轮只交补丁」）：下一次调用带上这份回答与每条错误（`answers.survey` 逐行读，一次列全：不是 JSON、不是对象、id 不在窗里或重复、轨号不在可答的里、缺行；同一类错并成一行、点名它的行，最多点 10 个、余下说还有几个，补丁自己叠不上的错同样并）。补丁轮要两样都有：有要重写的 id（读得出 id 的坏行或缺行；读不出 id 的坏行，它的 id 算缺行，补丁照样会要），也有能留下的（至少一行能读）。这时只要出错与缺的那几行，交一个 `<patch>`（`speaker_judge_repair_patch_v1.md`：一个 id 一行，写法同 `<judge>`；写到的行最后挂着的轨或新标签有底稿 `<tracks>` 里没有的，在 `<patch>` 之外另写一个只含这些的 `<tracks>`，按轨号或标签并进底稿），harness 按 id 换掉底稿里的行、整份再读（`patch.py`），下游只见到整份。补丁用不上（没有 `<patch>`、某行读不出 id 或 id 在窗外、与 `<judge>` 同时出现）时下一轮修同一份底稿、补丁自己的错排最前；回了整份 `<judge>` 的照整份读；拼出来仍不过的，下一轮底稿是拼出的整份。`<judge>` 闭合了却缺其中一样（一行都读不了，或错的只有窗外的 id、读不出 id 的坏行）时，同纠错补丁帮不上时的做法，带着回答与错误要整份重写（`speaker_judge_repair_whole_v1.md`：照 system 指令的格式整份重交，列出的每一类错都改到所有出这类错的行（不只是点名的那几行），每个 ▶ 行一行 JSON 再写 `<tracks>`；不用通用的 `fragment_repair_round_v1.md`，它说的 header 与列数裁判没有）；没有闭合的 `<judge>`（空回答、被截断、根本没写）时照旧原样重问整份、不带修复上下文。
  三种传输同纠错：API 是同一段对话追加两轮，capsule 的 agent 回到写出这份回答的会话（每单元一条链，`repair_session_key` 为 `speaker_judge-<单元 id>`），工具会话的 `submit` 用同一个读法校验（`agent_validators` 的 `speaker-judge`），被拒时在会话里修、同样收补丁，由 runtime 叠到底稿上。
  修复轮就是重试：每单元（窗或半窗）连同修复轮最多 4 次调用（`MAX_ATTEMPTS`），工具会话在一次调用里跑掉的修复轮也扣：它拿到的修复次数是这一单元在这次调用之后剩下的调用数，会话里首答之外最多再修这么多轮，与 API 路径交的答案一样多，用完记为失败、退回声学标签，续跑再判。补丁拼出的回答不做「没思考再问一次」。sidecar 条目记 `repairs`（harness 发出的修复轮，加工具会话没交上时报回的轮数；工具会话里修好的不报，同纠错的计数）。单独不合法的 split / fill / insert 照旧放进 `ignored`，不算回答读不了。
  额度类错误（按路由的 `classify_failure` 判为配额或限速：429、402 等；或来自至少两个单元、连续 3 次不花输出 token 的空回答）停掉整个裁判，没判的窗记为待判，把额度留给纠错；其他错误只让那一窗失败。窗口并行度跟纠错窗口同一个设置。
- **跨单元的新建轨**（owner 2026-10-09）：`continuity == serial` 时按顺序判，上一个单元（窗，或被拆开的窗的前一半）新建或接着用的轨，连同裁判给的名字、目前为止的例句与声音样本，进下一个单元的画像；裁判答这条轨号就算并进去，sidecar 记成 `carried: {window: 单元 id, label, of}`（不记号码：号码由读的一侧按单元顺序编；`of` 是源单元当时那份判决的摘要 `judgments.verdict_of`，源单元不再成立、或被重判成另一份判决时这条结果作废，也不再传给下一个单元；作废的那些行回到声学标签，接力的单元本身不因此重判）。`parallel` 时各单元各开新轨，不合并；同一个 `@甲` 在两个半窗里也是两条轨。
- **三个 pass 的 pass 1**（`run_speaker_judge(shape=...)`；何时走见[三个 pass](#三个-pass)）：裁判 prompt 多「定字幕的形状」一节，每行多带 `时长|gap`（gap 与纠错窗同义：上一行结束到本行开始，重叠时为 0，speaker-judge-v10 起钳 0），回答里每行可多一个 `join` / `may_join` / `drop`（`answers.parse_acts`：窗口第一行的 join / may_join、一行写了几个的，放进 `ignored`，不算数）。带音频的裁判多一个 `split`（这一行从开头到换人处的原文，加后半段的人 `track2`）：`answers.split_at` 把切点落到最近的词边界（两边去掉空格与标点、NFKC、不分大小写之后，值必须是这一行的开头、后面还剩字）；这一行 drop 了、后半段与前半段是同一个人、对不上开头、整行或只有一个词的，不切（词缺时间的也不切），记进 `ignored`；读回时切点超出那一行的词数（词变了、文字没变）的不读。
  带音频的裁判还能补上 ASR 漏掉的话（owner 2026-10-16：第一期开着收数据）：`fill` 是这一行补全后的整行（同一个人紧接着这一行之前或之后说的半句），`answers.fill_ends` 要求原有的字完整地留在值里、头尾加了字（比较方式同 `split`），这一行 drop 了、这一行被切开了（一行只取 `split`）、改了原有的字或没加字的，不补；`insert` 是这一行之后整段漏掉的一句（起止秒、原文、轨号或 `@标签`；重叠时 ASR 写下了附和、盖掉了主要的声音也算），换回源时间、夹在这一行开始到下一行结束之间（文件最后一行之后 5 秒以内），没给结束秒或结束不在开始之后的按字数估（每秒 7 字、0.8–5 秒）；起止、原文或轨号读不出的放进 `ignored`，回答照常成立。
  补上的字没有逐词时间，放在哪由读的一侧按全文件算（相邻的行可能在别的单元里，`judgments.placed`）：补了半句的行按加的字数（每秒 7 字）往旁边的静音里延，不压到上一行（它延过之后的结尾）与下一行；然后补出的行放进这一行结束到下一行开始之间剩下的空当，长度取听到的长度（至少 0.5 秒），截到下一行开始。成品现在不重叠，空当不到 0.5 秒的补行先不显示（`Inserted.shown` 为 None，即 `deferred`），听到的起止照样留着，等以后支持字幕重叠时用；原有的行不挪、不缩。drop 的行也算相邻的行。显示的那些进成品层（见上面「读的行」）：补了半句的行文字换成补全后的、起止换成延过的，去掉逐词数据（它们拼不出补上的字）；补出的行排在它锚定的那一行之后。
  sidecar 每个单元多存 `acts`（行 id → `{act, why}`）与 `ignored`，带音频的还有 `fills`（行、补全后的原文、头尾各加了几个字）、`inserts`（跟在哪一行后面、原文、听到的源时间起止，轨照判决结果的写法；`read_judgments` 给出放好时间的 `Judgments.fills` / `Judgments.inserts`，`inserts` 的新标签与后半段的一样编号，也随串行接力传给下一个单元）与 `splits`（行、原文、`at` 即后半段的第一个词、`from`，后半段的人照判决结果的写法：`track` / `label` / `carried`；判决仍挂在切开之前的那一行上，算前半段的，`read_judgments` 把后半段的轨给在 `Judgments.halves`，后半段新开的标签排在本单元判决结果的标签之后编号）；以前只答形状的 pass 1 写下的单元标 `trackless`、没有 `results`：读判决的一侧跳过这类单元（它们不挪任何人，不算失效），裁判把它们当作没判。要形状的读者（`holds` / `covers` 的 `shape`、`read_acts`）把缺 `acts` 的单元当作没判，单 pass 照旧复用带 `acts` 的单元。`speaker_judge_report` 记各类 act 的个数、逐行列出丢掉的行与理由（`dropped`）、没有 pass 1 的裁判窗（`without_pass1`；纠错按 `read_acts` 给的行换算到自己的窗上），带音频时还有补了半句的行数、补出的行数与其中先不显示的个数（`fills` / `inserts` / `deferred`）。
- **云端任务的标签**：行带 `line_speaker` 时（`speaker_judge/prompt.label_source`；本地的叠拉道不算），prompt 里讲「轨号从哪来」的几处换成转写模型的说话人：常把一个人拆到几条轨、错多在换人前后的几行、标签多数时候是对的；不讲 `??` / `s5+s4`，也不讲样本里混着并在同一条轨的别人（`SOURCE_WORDS`；形状一节的「轨号相同本身不算同一个人」同样换了理由）。本地任务的措辞逐字不变（`speaker-judge-v8` 起）。声音样本不挑与别的键时间重叠的行（`line_groups.overlapping`：被盖住或盖住别人），免得一段样本里有两个人；整段录音的终点与纠错窗口同一规则（`run_end`）；半窗的切点不落在被盖住的行之前（[窗口拆分](#窗口拆分)）。
- **续跑**：每判完一个单元立即写 sidecar。重判条件只有失败、待判、身份不对（单元内行、它们的整个 `speaker` 对象、画像用到的轨）、结果读不了（缺 `results` 或形状不对；三个 pass 还要 `acts`：`judgments.holds`）。裁判自己的 prompt 版本（`run.PROMPT_VERSION`，独立于纠错的 `PROMPT_VERSION`）随每个判决记下，升级时不重判仍对得上的单元（owner 2026-10-09：可复用的窗不因 prompt 升级丢掉）；换模态续跑时已判的单元也照旧沿用、只告警。要按新 prompt 重判就删 sidecar。
  **已完成的任务**（成品在）：sidecar 已覆盖全部窗口（不看版本）时照旧走 `reused` 捷径；没覆盖时走完整的任务边界，复用准备任务的上下文，只补裁判（窗取存下的纠错计划的边界，不看它的指纹——那是纠错实际跑过的窗；计划不在就新算一份、不写回；排窗失败只告警），
  再按现有 `-annotated.csv` 的 `position` 列重渲 `spk` 列（`refresh_annotated_speakers`），不重跑纠错、不碰 SRT；CSV 不在时只告警。
- **dry-run**：`--prompt-dir` 多出 `speaker-judge-0001.txt`（候选只有知识库的，准备任务还没跑），`plan.json` 的 `speaker_judge` 给窗数与输入 token；用 dry-run 自己算的计划，不写计划文件。
- **读回**：`speech/diarization/judgments.read_judgments` 是唯一的读取函数（`spk` 列、分轨导出共用），读的一侧同样逐层校验，见其模块 docstring。分轨导出在 raw-srt 阶段与 LLM 阶段之后各跑一次，每次先清掉上次导出的文件。
- **开发开关**：`FINESUB_SPEAKER_JUDGE_MARKS=0` 让裁判看不到置信标（「裁判放在哪」那组实验的输入），默认看得到（写成 `s3（低）`、`s3（低，备选 s5）`、`s4（补）`，与量过置信标对裁判有没有帮助的那组实验相同）。

## 任务 Artifact 记录

显式指定 `--task-artifact-dir` 时，`task-artifacts.jsonl`、`exchanges/`、
`task-report.md` 和 pipeline 的 LLM round 汇总全部使用该目录；不会再回退扫描或写入
默认 `<stem>.llm-artifacts`。其中 `task-artifacts.jsonl` 会记录：

- `fast_decision`：快速模式判定（mode/enabled/reason 与输出、输入两侧的估算值和预算）。
- `task_graph`：紧随 `fast_decision` 的一条**目标任务图**记录（2026-09-09 起）：执行策略（`batched`/`agentic`）与其来源（`correction_chain`／`test_profile`／`override`）、能力授权（search＝`--retrieval`、knowledge、style）以及任务列表（普通＝`prepare`+`translate`，fast＝单个 `whole` 的 `translate`，`--knowledge update` 追加 `knowledge_update`）。它是审计面，不是 resume 依据。⚠ 记的那个 `strategy` 是**纠错链**的：策略按任务解析，准备任务读的是研究链，两条链不同时（例如研究钉 agent、纠错混 API）这一条不代表准备任务实际走了哪条——准备任务自己的那次解析记在 `research-context.json` 的 `rounds.task_flow`。
- `prepare_task_request` / `prepare_task_response`：准备任务的请求回复与交付回复（模型、usage、解析错误，以及 `emits_queries`/`emits_entries`/`native_search`/`agentic` 等形态字段；token 报告计入 `phase: research`）；直接交付时只有后者。
- `research_search_results` / `correction_search_results`：本地搜索的 query 列表、每条 query 的 provider/条数/URL/错误，以及注入文本长度。
- `correction_query_response`：纠错窗请求回合的回复（模型、usage、解析错误、提取出的 query 与 window_notes；`delivered` 标记直接交付、`task_flow="batched"`）。⚠ **`delivered=true` 的那条不是一次独立调用**：同一个回复接着就是这一窗的 attempt 0，由下面的 `correction_window_response` 记录，所以它不进会话统计、不出 token 行、也不再写一份 exchange（记录本身留着，它是「请求通道开着但没被用」的审计面）；`correction_query_call_error`：请求回合调用异常（best-effort，不中断任务）；`correction_query_format_retry`：请求回合的格式重试。三个名字沿用旧查询轮的产物契约（bench 打分与已有 run 的读法不分叉）。
- `window`：窗口 id、源序号范围、起止时间、重叠源序号、只读前文序号（`preceding_source_ids`，v13）、音频剪辑区间（`clip_start`/`clip_end`）、边界原因和规划预算。
- `request`：请求输出上限、消息文本字符数和请求文本哈希（fingerprint），以及是否附加 Gemini 文件。真实 token 数以 provider usage 和 token 分布报告为准，request 侧不再做本地估算。
- `provider`：从 raw response 中提取的 `usageMetadata` / `usage`、模型版本、response id 和 prompt feedback 等 provider 元数据；如果 provider 没返回 usage，该字段为空。
- `response`：响应文本长度和哈希，以及提取出的 `next_advice`。完整模型文本仍保留在 `response_content`，供失败分析和任务中知识更新使用。
- `correction_window_retry` 会额外记录失败窗口、重试窗口和（拆分时的）后半窗口。
- `exchanges/` 子目录：每次 LLM API 交互一个可读 markdown 文件。串行阶段（准备任务的各条回复、知识更新）为 `NNN-<call>.md` 按调用顺序编号；纠错窗（请求回合 + 各次交付 attempt）为 `NNN-MM-<call>.md`——`NNN` 是**调度时**按窗口顺序预领的 block 号、`MM` 是该窗内的调用序号，因此并行模式下两次相同的 run 产出完全相同的文件名（完成顺序不影响编号；缓存回放的窗口不占号）。文件顶部先渲染 API Calls 表，逐行列出本 logical attempt 内所有底层 `completion()` 调用（provider tier、model、api key name、同 key+model 的 call #（按调用结束的顺序编号，没发出的备份也占一个号）、return code、发起/返回时间、耗时、备份请求——原请求写备份在第几秒启动（之后可能还要等限流）、备份写明自己是备份、收到响应头与第一段可见正文的时刻——从发起算的秒数，没到就空着、失败时留下的半截正文文件）；模型 fallback 或同 key retry 的失败尝试也会保留，最后一行通常是成功调用。随后是去重后的元数据头（attempt、finish/校验状态、`input_tokens`、`output_tokens_breakdown`、逐输入块 token 等；provider/model/key 只在 API Calls 表出现）、逐 message 的请求全文（音频附件以 `[附件文件: ...]` 标注），以及完整模型响应全文（含 `<reasoning>` 等标签，不再单独摘录）。覆盖准备任务、请求回合、交付轮（含重试/拆分）和统一知识更新调用（逐 chunk 一个文件）；供知识库更新、prompt 迭代等后续任务直接提取信息。
  元数据头包含**`prompt_version`**（当前 `PROMPT_VERSION`，便于对照契约）、**输入分块 token 统计**（`exchange_metadata.py` 的 `*_input_components`，用任务 counter 对各标签块计数，0 值省略）和合并 token 行（`uncached / cached / total` input，有缓存写入时后面再跟 `cache write N`；`visible / thinking / total` output）：准备任务 `transcript/extra_info/note_url_extract/<每个可匹配类别>_index/preinjected_entry/knowledge_injection/search_injection`；请求回合与交付轮 `csv/media/knowledge(context pack)/entry_details/advice_ledger/pre_round_notes/preceding_context/各类别 index/search_injection/expected_output`；⚠ **index 那几个键是按可匹配类别派生的**（`exchange_metadata.index_block_tokens`，2026-09-07 前写死为 `streamer_index_tokens` / `common_index_tokens` 两个），所以键的**数量随类别集合变**，读日志时别假设正好两个；知识更新 `window_packs/kb_entries/prompt_tokens_estimate`。纠错 validation failed 时，header 会保留错误理由并从错误文本中摘出 row/source id 位置摘要。
- 检索/词条注入类 artifact 均带 `render_report`（included/truncated/dropped 与块 token 数）；请求回合响应另记 `requested_entries/injected_entries/missing_entries`；本地预注入记 `knowledge_preinjection`（匹配词、频次、来源 phase）。
- `content_filter_ladder` / `content_filter_blacklist`：Gemini PROHIBITED_CONTENT 阶梯恢复——某次调用被拦后按「URL leave-one-out → 丢全部 URL → 丢全部检索注入」重建 prompt（不占 validation/parse 重试预算）；定位到的毒块按 `content_hash` 写入黑名单，同任务后续窗口/轮次 render 前预剔除；resume 启动时从 artifact 加载。知识更新与请求回合无检索注入可丢时只做原样重试一次。
- `window_plan_report`：仅当规划因输入超预算发生 `k+1` 重排时写入（research/correction 各自 phase），task report 渲染成 "Window Planning" 小节。
- `token_distribution_report`：每阶段一条（`phase: research` 在准备任务后写入并同时并入 `research-context.json` 的 `token_report`；`phase: correction` 在最终 SRT 写出后写入）。`rows` 逐调用记录 `call/chunk_id/attempt/model/finish_reason` 和 token 分布（`prompt_text_tokens` / `prompt_audio_tokens` / `thinking_tokens` / `output_tokens` / `total_tokens`，来自 Gemini REST `usageMetadata` 的 `promptTokensDetails` 模态拆分与 `thoughtsTokenCount`）；`totals` 为各项求和加 `call_count`，供调参参考。
- `api_call`：本地非 LLM API 调用计数（如 `gemini_file_upload`、`web_extract`），供 `task-report.md` 汇总。
- `../<stem>-metadata.json`：与主产物同级的 pipeline metadata；只记录下载、人声分离、VAD-ASR、LLM harness 和 pipeline 总耗时、相关 worker，以及从本次 resolved artifact 目录汇总的 LLM logical-round 耗时。同一 batch logical run 的后续 pass 保留前一 pass 已执行的 stage；stage snapshot 整条替换，不会出现 `reused` 携带旧执行耗时。单轮跨度覆盖该轮全部失败 attempt、endpoint fallback 与 validation/format retry；底层 API 行仍以 `exchanges/` 为明细来源。轻量的 stabilize/SRT 导出/后处理不单列。
- `task-report.md`：任务完成后给用户阅读的**运行时摘要**——输出路径、上述核心阶段/总耗时和 worker、LLM logical-round 耗时/API attempt/retry、API 调用计数、三本 token 账——分阶段（准备、说话人裁判、纠错、校对、知识更新）、分会话、**按 (provider tier, model) 汇总的 token 成本视图**。三本账是同一份逐调用记账（`token_books.py`）的三种分组，合计相同：每次调用记一次，没答成却花了 token 的 attempt（fallback 前面的目标、续接失败重建的会话、整次失败的调用）记在它所属的会话与它自己的目标下，`pseudo-conversational` 的会话总账（`agent-session-usage.json`）单列成会话行；`token_distribution_report` 不再是报告的来源。仍记不进的是 provider 没报用量的花费：API 上中途断掉的流、被替下的备份请求，agent 超时或卡住被结束的调用（attempt 行没有用量）；Gemini API 上被内容过滤挡下、在写任何记录之前就抛出的调用（`GeminiPromptBlockedError` 不带 attempt）；`pseudo-conversational` 会话的 CLI 抛错时那条会话的用量（宿主只从成功返回的结果读用量）。成本视图里 cached、cache write 与 uncached input 分列，因为三者计价不同，CLI 不单列思考的目标思考一格写 `n/a`（可见输出含思考）；tier 从 `route_decision` 里答题的那个候选反查，因为同一个模型在免费档和付费档是两笔账，本地 agent 档则按订阅计费、其行只用于缓存与上下文核算；**既反查不出答题者、表里计的 token 又全为零的调用不进该表**（判的是数值不是键在不在：response 记录的 `usage` 恒为补零的全键 dict）——它不是一行账，只会渲染成 `unknown / unknown` 加一次调用和一串零，读起来像「认不出的 provider」，实为「失败在任何 provider 上报之前」，会话表里那一行照留）、fallback 与疑似 IP/代理风控 warning、**所有失败调用的逐条列表**（任何 `*_call_error`，不只下面两类各有专节的——否则 agent 未 submit 这类失败会让本节只剩一句「No fallback was recorded」）、Gemini File 403 提示、重试/拆窗、SRT 后处理与知识库更新摘要。注入上限与 thinking effort 的静态说明见上文「注入上限」「LLM thinking effort」，不在此文件重复。

## 最终 SRT 后处理

当前支持 profile `-1`、`0`、`1`、`2`、`3`、`4`，其他值会直接报错。默认 profile `0` 按顺序执行 profile `3 → 4 → 1 → 2`，从 `*-translated.srt` 生成最终 `*.srt`：

- profile `1`（时长）：每条字幕末端先固定后延 `0.3s`（不得越过下一条开始）；随后把剩余小于 `0.3s` 的短闪轴空隙闭合到下一条开始。
- profile `2`（标点）：中文逗号、中文句号和中文全角空格替换为英文空格；每行首尾 whitespace trim；不修改时间轴。
- profile `3`（繁简）：整篇 opencc t2s 试转，字符差异率超过阈值才判定为繁体并整体转简。
- profile `4`（重叠）：检测相邻字幕重叠（前一条结束晚于后一条开始），把前一条的结束提前到后一条开始，并向 stderr 打一条 `Warning:` 报告条数与首个实例。重叠是上游时间轴缺陷，不是渲染选择，所以必须可见。乱序输入（后一条开始早于前一条开始）会把该条压成零时长而不是让结束早于开始。
- profile `0`（默认）：`3`（繁简）→ `4`（重叠）→ `1`（时长）→ `2`（标点）。`4` 必须排在 `1` 前面：`1` 的「不得越过下一条开始」会把重叠的字幕**截短**并计进 `duration_extended`，先解重叠才能让那个 cap 退化成 no-op。

`--postprocess-profile -1` 不做时间轴或文本清理；实现仍会解析并重新渲染 SRT，因此字幕时间与文本语义保持不变，但不保证字节级原样复制。

`*-raw.srt` 跑同一套时间轴策略 `4 → 1`（见「产物」一节），文字保持 ASR 原样。顺序常量
`TIMELINE_POSTPROCESS_PROFILES` 是唯一来源，profile `0` 与 raw 导出共用，不会各写各的。

`final_srt` artifact 与 `task-report.md` 会记录请求的 `profile`、实际 `applied_profiles`、重叠修复条数 `overlaps_fixed`、末端延长条数 `duration_extended`、闪轴闭合条数 `flash_extended`、标点替换数与 trim 行数。

## 连不上的 host

被墙、代理没开或挂了时，连接阶段的失败每次都要等满 connect 时限（每个解析出的地址各一次），以前每个请求都重付
一遍、最后只报 "timed out"。`api_http.HostBreaker`（进程级，按 `(host, 路由)` 记，不写 `.state`）让同一个进程
只付一次，并且说出原因：

- **什么算连不上**：只认连接阶段的异常类型 `httpx.ConnectError`、`ConnectTimeout`、`ProxyError`（沿 `__cause__`
  / `__context__` 找），不看文本。DNS 投毒、SYN 被丢、SNI 之后被重置、代理拒绝都落在这三类。拿到任何响应
  （含 4xx / 5xx）都算连上；读超时、`PoolTimeout`、`DeadlineExceeded` 不算连接失败。
- **路由**：`[api] proxy` 设了就是它，否则是 `urllib.request.getproxies()`（环境变量，Windows 上没有环境变量时读
  系统代理，httpx 也这么取），都没有就是直连。换了代理就是另一条路由，立即重新尝试。
- **Warning**：一串失败的第一次就打 `host-connect-failed`，只报事实：host、路由、异常正文前 120 字。一串失败
  持续满 30 s 判定连不上，打 `host-unreachable` 并给处理建议；两次失败相隔超过 60 s 重新起算。拿到响应时打
  `host-reachable`。没有绑定 reporter 的进程打到 stderr，不会静默。
- **判定之后**：可选环节（countTokens、检索的 Exa / Tavily / Gemma4）除探测外不再发请求，直接用替代品（启发式
  计数、下一家检索，检索记 `SearchFallbackEvent(reason="host_unreachable")`）。探测间隔从 30 s 起翻倍、封顶
  10 分钟，同一 host 同一时刻最多一个（名额由拿到探测的那次调用在结束时让出，不论怎样结束；别的请求的成败不让出它），探测本身不重试。必需环节（生成调用、媒体上传）没有替代品，每次调用仍试，
  但只试一次、不重试；正在退避的调用在下一次重试前发现已判定就停。以连接失败告终时抛 `HostUnreachableError`
  （`route_failure_kind="unavailable"`），候选链换下一个 provider；那一次拿到了响应（429、5xx、400）就照常分类、
  记配额与换 key。
- **恢复**：任何请求拿到响应（5xx 也算，包括被 `raise_for_status` 抛成异常的）就清掉判定，不用重启：生成调用下一次就恢复，可选环节最迟在下一次探测时恢复。
  发出时间早于最后一次成功的请求，后来失败了也不算（并发时晚到的旧失败不覆盖新的成功）。
- **生成调用里的分类**：连接阶段失败记 `NO_RESPONSE_CONNECT`，不计入连续超时、不触发连续超时中止、不置
  sticky 耗尽、不启动 combo 冷却（冷却管配额与过载，写进 `.state` 跨进程生效，拿它记「连不上」会让下一个进程
  也只看到「cooldown until …」）；combo 在 PROBE 阶段时释放探测名额，不记耗尽。对冲的两条请求里任一条拿到
  响应头就算连上。
- **记在哪个 host 上**：请求实际发往的那个。Gemini 生成调用设了 `GEMINI_BASE_URL` 就记在镜像上，所以 Google 被判定
  连不上时，经镜像的调用照常重试；上传与 countTokens 不走镜像，记在 Google 上。
- **候选链**：同一条链里一个 `(host, 路由)` 连不上之后，后面同 host 的候选直接跳过（决策记
  `reason="host_unreachable"`），判定之前也一样，所以一条链对一个连不上的 host 只付一次等待。连接失败时不丢
  已上传的媒体引用。
- **各处的 connect 时限**：countTokens 5 s、检索的 Exa / Tavily 10 s，其余（生成调用、Gemma4、上传，以及云端转写）共用 `api_http.CONNECT_TIMEOUT_SECONDS` = 15 s；下载侧另有一份同值的（[`download-routes.md`「路线解析」](download-routes.md#1-路线解析)）。connect 时限只管服务端开口之前的那段（TCP、代理的 CONNECT 回话、TLS），Google 的任何回答（429、5xx、长时间思考）都在它之后，截不到。
- **代理不回 CONNECT**：经代理（`http://` 或 `https://`）访问 HTTPS 时，先发 `CONNECT` 等代理回话，httpcore 等这个回话用的是读超时，代理接了连接却一直不回（卡住了，或先连被墙的上游再回话）时，以前要等满读超时（生成调用是硬截止），结束时是 `ReadTimeout`、不算连接失败。`api_http_client` 与下载侧的 `create_client` 给每个 HTTPS 请求挂一个 `TunnelWatch`（`finesub_bootstrap.http_client`，httpcore 的 `trace` 钩子）：只有等 CONNECT 回话这一段用 connect 时限，隧道建好后读取时限照旧，超时算连接阶段失败，Warning 写「代理没有回应 CONNECT」。它认的是 CONNECT 请求本身，不是事件先后（`https://` 代理在 CONNECT 之前还有一次自己的 TLS 握手）；按 httpcore 的事件名工作，事件名改了就失效、退回旧行为，有一条真 socket 的测试钉着。
- 不按 IP 或地区预判：要请求第三方服务，挂了 TUN 或系统代理时会误判；连接阶段失败才是直接证据。
- **实测（2026-11-09，本机）**：没有 tokcount，DNS 指向丢 SYN 的地址，或代理接了连接却不回 CONNECT：前三次计数各约 11 s
  （connect 5 s + 间隔 1 s + 5 s），第 33 s 判定，之后每次 0 s。countTokens 真机 6.5 万 / 19.5 万 / 39 万 token 各 0.3–0.9 s，
  读超时 15 s 够用。生成调用被墙时的等待没实测，按时限推算：丢 SYN 时每次尝试 15 s × 解析出的地址数，第 30 s 之后的那次失败即判定。
  取舍与理由在 [`llm_design_notes.md`「连不上的 host 与离线 tokcount 的决策记录」](llm_design_notes.md#连不上的-host-与离线-tokcount-的决策记录2026-11-09)。

## 重试与拼接

每个纠错窗口的目标是一次 API 交互完整成功。

**API 调用的两段超时**（`llm_runtime.api_timeouts`，只管 REST 生成调用；本地 agent 用
`[llm] local_agent_timeout_seconds`，上传/countTokens/检索各有自己的）：

- 所有 REST 生成调用（Gemini、OpenAI 兼容、Anthropic，以及检索里的 Gemma4）都走 SSE 流式，
  但流式只让人看见答案在来，看不出剩下的还要多久；思考中的模型在出正文之前可能一个字节都
  不发。所以时限随**这次允许的输出量**变：硬超时 = 180 s + `maxOutputTokens` ÷ 25 tok/s（按
  最低可接受的速度写完整个上限），软超时 = 硬超时 × 0.5；没有给上限的调用按 65536 算。64k
  输出的硬超时约 46 分钟、软超时约 23 分钟，8k 是 8 分钟与 4 分钟。三个数可在 `config.toml`
  的 `[llm]` 下覆盖：`api_timeout_overhead_seconds`（≥0）、`api_timeout_floor_tokens_per_second`
  （≥1）、`api_soft_timeout_fraction`（0.01–1）。检索里的 Gemma4 没有输出公式，硬超时固定 600 s，
  软超时按同一个比例。
- 用的是**实际发出的 `maxOutputTokens`**（已按上下文剩余截过，思考 token 也算在里面），**不是**
  规划用的输出估计（`output_reserve`）——答案本来就允许超过估计，按估计定时限会把这些答案在
  生成途中切断。
- **软超时按静默算，到点发一个备份请求，原请求不断**（`http.hedge_on_silence`，owner
  2026-10-23）：从请求发出算起，收到任何字节（响应头、保活注释、思考摘要、正文）都重新计时；
  连续这么久什么都没收到，就发 `llm-api-soft-timeout` warning，写明原请求走到哪了（还没收到
  响应头 / 已连上、还没出正文 / 正文已出多少字），并在同一个 key 上另发一个相同的备份请求
  （照常占这个 key 的 RPM 与 TPM）。两条谁先拿到合法回答就用谁，另一条立刻断开、记
  `SUPERSEDED`，答中的那条总记在最后；有过备份的成功调用再发 `llm-api-backup-outcome`，写明是
  哪一条答的；原请求在备份等限流时就答中的，备份不再发出，但它的限流预约与 `SUPERSEDED`
  一行照样留着。attempt 上原请求记 `backup_sent_sec`（备份启动的时刻），备份记
  `backup`（exchange 文件 API Calls 表的「backup request」列）。最多两条：备份自己静默不再发
  第三条。两条都失败时这一对算一次尝试，按备份的错误往下走（退避重试、换 key、连续超时
  计数）——它是对这个端点更新的判断。不在静默时直接断开，是因为静默分不清模型在思考还是
  连接死了，断开要从零重新生成；多开一条几乎不花钱，只有原请求其实在正常思考、最后也会
  回来时才多付一份。⚠ 被断开的那条拿不到 usage，它已花的 token 不进任何本地用量统计
  （会话 token 表、`task-report.md` 的成本视图），所以本地数字可能比服务商的账单少
  （owner 2026-10-23）。
- **硬超时**断开这一条请求并记 `NO_RESPONSE_TIMEOUT`，每条请求从各自发出算。不设响应头超时：
  Gemini 的 OpenAI 兼容端点在 72 s 的思考里连响应头都不发（2026-10-23 实测），没有响应头也只
  算静默。这几条取舍的理由见
  [`llm_design_notes.md`「API 调用改流式的决策记录」](llm_design_notes.md#api-调用改流式的决策记录2026-10-2223)。
- ⚠ **硬超时是整次调用的截止，不是 httpx 的 timeout。** 后者只限单次读写的等待，服务端隔一会儿
  发一个字节（分块传输的 body 慢慢往外滴）就永远触发不了，调用会一直占着窗口槽位、连续超时退出
  也不会发生。所以 `http.stream_within_deadline` 自己守截止，三件事：
  1. 请求在工作线程里跑，调用方只等到截止——哪怕工作线程卡在谁也打断不了的 DNS 解析里，调用方
     也准点返回。等待按 0.5 s 分片（Windows 上带超时的 `join` 打断不了），所以 Ctrl+C 立即生效，
     并且同样拆掉在途请求；
  2. 到点**拆掉**请求而不是丢下：对 socket 做 `shutdown`（能唤醒另一个线程里阻塞的读，Linux 上
     光 `close` 不行），再关 client。socket 通过 httpcore 的 `trace` 扩展在每一步产出时拿到——
     先是 TCP 的，再是替换它的 TLS 的（`wrap_socket` 会让原 socket 失效，只 shutdown 它等于
     什么都没做），所以还没收到响应时也拆得掉。⚠ TLS 握手进行中那一小段没有可拆的 socket，
     由 connect 超时（≤15 s）兜住，调用方不受影响；
  3. 同一个 `trace` 在「开始连接 / 连接完成 / 开始 TLS / 开始发请求头」这几步检查截止，**过了截止
     就拒绝往下走**——卡在 DNS 里的工作线程回来后也发不出请求，不会在没人等的时候触发一次远端生成。

  是否超期一律按**单调时钟**判断，不看计时器有没有跑过：逐次读超时与截止共用同一个预算，谁先触发
  是调度噪声（Windows 上线程唤醒约晚 15 ms），异常类型不能取决于它。只要截止决定了结局，调用方
  拿到的就是 `DeadlineExceeded`——`httpx.TimeoutException` 的子类、文本带 "timed out"，重试与
  分类规则照旧把它当超时；截止之前的真失败（例如 15 s 的连接超时）原样抛出。httpx 的逐次读超时
  设成与截止相同，只是兜底。证明在 `test/test_llm_http_deadline.py`（真 socket 的几条要
  `--run-network-mock`）。
- connect 阶段单独 15 s，不随输出量放大（以前是 45 s，被墙时按每个解析出的地址各等一次，第一条 Warning 要等约 169 s）。连接阶段失败（host 连不上）不算超时，规则见[连不上的 host](#连不上的-host)。
- **流的另外两种失败**：流没有终止标记就结束（连接中途断开、提前正常关闭，或服务端在同一条流
  里从头重新生成——已收到的文本（含思考摘要）开头被重复满 32 字符，Gemma4 实测这样做过）记 `STREAM_INTERRUPTED`，可重试、不计入
  连续超时；HTTP 200 之后流里来的报错事件按它自己的语义码分类（Anthropic `overloaded_error` 即
  529、额度用尽即 429 …），与同样的 HTTP 错误一样处理。
- **失败请求收到的正文留一段时间**：失败的 attempt 记 `partial_chars`（这次丢掉的正文字数）；
  收到过正文的，还把那段正文连同模型、key、返回码与原因写进日志目录下的 `api-partial/`
  （`http.keep_partial_text`），文件名记在 attempt 的 `partial_text_file`（API Calls 表的最后
  一列）。只留最近 20 份，与本地 agent 的失败现场同一个界（`RETAINED_FAILED_EPISODES`），每写
  一份按修改时间删掉更旧的。被另一条先答而断开的不算失败，不留。这些文件不进反馈包：正文是
  用户的素材，反馈包只收运行日志（owner 2026-10-23）。
- **流拼回非流式的形状**（`llm/streaming.py`，每种方言一个拼装器），所以下游读的字段不变。两处
  例外是故意的：Gemini 因 SAFETY / PROHIBITED_CONTENT / BLOCKLIST / SPII / RECITATION 中途
  停下时，已到的正文丢掉，和非流式一样交出空回答——前三种照常走被拦截的处理，SPII /
  RECITATION 按空回答处理；
  `groundingMetadata` 取最后一份（实测只在最后一块出现一次），后一份丢了前一份的来源时发
  `llm-stream-grounding-replaced` 警告。
- 连续两次以硬超时收场的尝试（一对请求算一次）仍按原规则立即抛出
  （`CONSECUTIVE_TIMEOUT_ABORT_COUNT`）。一次尝试最长约软超时 + 硬超时（备份在原请求静默满软
  超时时才发，再跑满自己的硬超时；原请求先出一阵字再卡住时接近两个硬超时），所以一个窗口最坏
  占用约三到四个硬超时，另加退避——纠错窗口一次全静默的尝试约 70 分钟。

**媒体上传有自己的重试预算，与模型调用的分开。** `RoleClient(max_retries=...)` 从拿到 media
ref 之后才开始计，所以上传本身的一次网络中断（2026-08-20 实例：finalize 步
`httpx.ReadError [WinError 10054]`，经 `http_proxy`）曾直接终止整个 LLM 阶段。现在
`_upload_gemini_file_rest` 内部：

- 最多 3 次尝试，退避 1s / 3s 加 ≤0.5s jitter；`429`/`503` 等带 `Retry-After` 时用它——只认
  秒数形式且 ≤30s，HTTP-date 形式按固定退避。**每次尝试新建 resumable session 整体重传**——中断后的旧 session 状态不明，也没有
  offset 查询；不做断点续传。文件内容只读一次。
- 可重试判定**按异常类型与状态码，不看错误文本**，与模型调用、云端 ASR 共用
  `api_http.is_retryable_http_error`（规则见 [`llm_harness_routing.md`「生成 API 速率限制」](llm_harness_routing.md#生成-api-速率限制finesubllmrate_limitpy)）：
  没收到响应的传输错误与 408/409/425/429/5xx 重试；其余 4xx、缺 `x-goog-upload-url`、本地文件错误立即失败。
- finalize 成功之后只重试轮询：state GET 与 `countTokens` 各自按单次请求计 3 次，正常的
  `PROCESSING` 等待不消耗它；不会重新上传造成远端重复文件。`countTokens` 的 400 是「尚未采样」
  的就绪信号，照旧等待；其余非 200 走同一套分类（429/5xx 重试，401/403 立即失败，不会被当成
  没就绪等满 5 分钟）。
- 超时是 `httpx.Timeout(connect=15, read=600, write=600, pool=45)`，是每个网络操作的上限，不是整次
  上传的总时限。connect 原是 45 s，实测连不上时静默 169 s（每个解析出的地址各等一次），降到 15 s。
  上传的 host 判定连不上之后每次上传只试一次，见[连不上的 host](#连不上的-host)。
- 上传跑在 `WindowClipPrefetcher` 线程里。prefetcher 关闭时先 set 一个 `threading.Event`，
  上传在每次尝试之间、退避等待与轮询间隔里都检查它，收到即抛 `UploadCancelled`；**已经在线上的那个请求
  不能被打断**，仍可能跑满自己的超时——取消只保证不进入下一次尝试。
- 最终失败抛出的是最后一次的原始异常（不包一层），日志 `gemini-upload-failed` 带尝试次数。

- 不做多轮续写。
- 输出上限判定用两个信号：usage 计数（输出+thinking token >= `65536 - 100`，主信号）与 `<translated>` 开标签无闭标签（兜底）；finish reason（`MAX_TOKENS` 等）只记录、不参与判断，原因见下一条。
- **agent 传输的 usage 不参与这个判定**：CLI `result` 事件里的 usage 是跨工具轮累计的整个会话，不是单轮输出，和单轮上限比会把完整的结果误判成截断、白拆一次窗口。产物记 `basis: agent_session_cumulative_usage_not_comparable`；这条路径上的截断只认显式信号（错误结果、输出校验失败）。agy 撞上单次输出上限时自己在同一次调用里让模型续写，续完的答案从它的 transcript 取回、按普通回复走校验与修复轮，不算错误结果，也不拆窗（[`llm_local_agent_agy.md`「轮内重试之后的答案」](llm_local_agent_agy.md#8-轮内重试之后的答案撞输出上限断流空回复2026-10-13)）。
- 拆分判据（2026-08-08 修正）：**主信号是 usage 计数**（`output_tokens_plus_thinking_tokens`）。`finish_reason` 记录进产物但**不参与判断**——`46206b1` 有意降级它，因为 flash 会在输出完整时误报 `length`，害得生产窗口 0001 把一个通过校验的完整结果白拆一次。`<translated>` 开标签无闭标签这条内容启发式作为**兜底**：仅当 usage 缺失或低报（导致主信号为假）**且窗口重试已用尽**时才触发，避免「只是需要再试一次」的窗口被提前拆开。此前它完全没有接线，于是截断的回复表现为普通校验失败：同一个超长窗口再发 5 次，然后以 `RuntimeError: Window NNNN failed validation` 杀掉整个任务。满足以上任一即把当前窗口**对半拆分**重试：在窗口中间附近选择合适边界（句末 > 长静音 > 片段边界）拆成两半，两半之间保留与正常窗口同规则的动态重叠（切点前 30s 内条数，纯内容驱动，稀疏处可为 0），先处理前半，再处理后半；每个半窗有自己的音频剪辑与上传，-a 继承父窗口的只读前文、-b 回看父窗口尾部。
- 子窗口 chunk id 是 `父id-a` / `父id-b`；两个子窗口按窗口 id 继承父窗口的 window context。serial 下前半的 `<next_advice>` 传给后半；parallel 没有 advice，叶窗口可独立执行。总调用次数仍受两档重试预算约束（见下）；单片段窗口无法拆分时同窗口重试。⚠ **只拆一层**（`WindowGeometry.MAX_SPLITS = 1`；owner 2026-09-02 定的 2，**2026-09-03 改为 1**）：半窗还不成就不是尺寸问题了，再拆只是拿配额换同一个失败——此时抛 `RuntimeError` 停掉该 task。所以生产能产生的 chunk id 只有 `父id-a` / `父id-b` 两层，**不会出现 `0001-a-a`**。深度直接从 chunk id 数 `-` 得出，不另记计数器（id 就是血缘，计数器会与它漂移）——读法与上限是两件事，`split_depth` 对一个 `0001-a-a` 仍答 2，只是没人再造得出它。
- CSV 格式错误、未知源序号、重复源序号或源序号乱序默认同窗口重试，重试后仍失败则报错。
- **同窗口重试是修复轮，不是盲重掷**（2026-08-15）：`reason=validation_same_window`
  的下一次 attempt 会带上**上一轮的输出**和**校验器给出的每一条错误**。表示形态由
  transport 决定——无状态端点收到追加的 `assistant`（上轮输出）+ `user`（错误清单与修复说明）两轮；
  本地 agent 收到的是 capsule 输入（`input/previous-output.txt`、`input/validation-errors.txt`、
  `input/repair-instruction.txt`）。
  修复轮**不额外占预算**：它就是原来那次重试，只是 prompt 里多了东西，所以
  `--max-retries-per-window` 与 checkpoint 身份都不变。
- **修复轮只交补丁**（2026-10-23）：上一份答案大半是对的时候，修复说明（`fragment_repair_patch_v1.md`）
  要模型只交一个 `<patch>` 块，写要改的源序号的行（格式同 `<translated>`）；harness 把它叠到上一份答案里
  通过校验的行上（`correction_patch.py`：补丁里出现的源序号，底稿里碰到它的整行连同其上的 `#` 注释换掉），
  拼出整份再走同一个校验，下游只见到整份。同时满足才用补丁，否则照旧要整份重写（`fragment_repair_round_v1.md`）：
  上一份有一个 `<translated>` 块；每条错误都归属到某一行、缺源、跨换人合并或整窗错位（错位那一段的行当作没通过，送回去改）；通过校验的行覆盖窗口至少一半的源
  （`PATCH_MIN_COVERAGE`，owner 2026-10-23）；纠错这一格没有 `variant_overrides`
  （下一次可能落到写另一种行格式的候选）。补丁用不上（没有 `<patch>`、某行读不出源序号或序号在窗外、与 `<translated>` 同时出现、转义成 `\uXXXX`）时，
  下一轮修的还是同一份底稿，补丁自己的错误排在最前；回了整份 `<translated>` 的照整份收；拼出来仍不过的，下一轮的
  底稿是拼出的整份。重放 A/B 里四个模型补丁版的修过率都不低于整份重写，含思考的输出 token 降到 3%–11%、耗时降到
  21%–63%，整份重写还会顺手改写大量本来没错的行（较弱的模型在这些行里改出过能过校验的错），补丁版不碰它们
  （读数在维护者的本地报告里）。agent 默认档的工具会话在会话里修（`submit → repairable`），同样只交补丁（runtime 在 `submit` 里拼，见
  [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§2 生产工具表（唯一真相）](llm_agent_tool_protocol.md#2-生产工具表唯一真相)「补丁修复」）。
  说话人裁判（含三个 pass 的 pass 1）同一套做法（owner 2026-10-31）：`<judge>` 里有能留下的行就只要出错与缺的行的 `<patch>`，按 id 叠（裁判的行逐行独立，没有覆盖门槛）；闭合了却没有能留下的，带着错误要整份重写；没有闭合的 `<judge>` 才原样重问；
  见[说话人裁判](#说话人裁判--diarize-judge-text--judge-audio)「每窗一次调用」。
- 三种情况下不带修复上下文，退回原来的盲重试：
  - **窗口变了**——拆半窗或截断重试，上一轮的输出讲的是别的行；
  - **装不下**——附加后超出该候选的 `max_input_tokens` 时逐候选丢弃（决策记
    `repair_context: dropped_input_limit`），因为丢掉辅助也比丢掉这次尝试好；
  - **agy 且没有会话复用**——见下。
  产物里 `correction_window_response.repair_round` 记录这次 attempt 有没有带，`repair_mode`（`patch` / `full`）
  记怎么要的，补丁轮再记 `patch_reply`（`patch` / `full` / `invalid`）、补丁单元数与换掉的底稿单元数，
  `response_content` 存拼出的整份；`correction_window_retry.repair_context` 与 `repair_mode` 记录下一次会不会带、
  怎么要。**exchange 渲染的 message
  列表是基础 prompt**——修复轮是在 client 里按后端追加的，不经过调用方组装的那份，所以
  不在下方 message 里；exchange 头因此只在带了时写一行 `repair_round`，指向上一 attempt
  的 exchange（那里有上一轮输出全文与 `## Validation` 里的错误原文）；exchange 的回复是模型的原样回复（补丁）。
  ⚠ `repair_round` 只说明调用方发了修复上下文，不说明 driver 用上了：被拒收时它仍记 true，
  审计要同时看 `execution_attempts[0].repair_context`（`declined_by_driver` 即实为盲重掷）。
  **效果（2026-09-30 结案）**：当时归档里 10 次修复轮全部一次通过；其中 6 次是在上一轮真实内容上修的
  （另 4 次上一轮只有一个时间戳或被 driver 拒收，等同盲重试），这 6 次保住了上一轮 99.5%–100% 的字幕行，
  没有一次整窗重写，唯一的回归是一次在 `corrected_text` 里各丢了一个假名、译文未变。所以按「保住已经对的
  内容」判，修复轮优于整窗重来。
  盲重试的通过率没有对照数据，「通过率更高」这一点没有立论。
- **agy 只在写出这份答案的那个会话里修**（owner 决定 2026-08-15）。新开会话要把整窗连同
  上一轮输出重发一遍，而那第二份对模型不是同一个东西：在自己的会话里它是「刚写的」，
  换个会话就成了「别人给的一段文本」，代价还落在 agy 最贵的媒体窗上。所以
  `session_scope=assignment` 且有 conversation handle 才接修复上下文，否则该次重试保持
  盲掷（执行记录写 `repair_context: declined_by_driver`）。复用会话时上一轮输出已在
  上下文里，整份修复的任务份末尾因此只接校验错误（`@@ harness validation errors`），不为第二份副本付钱；补丁修复的会话里只有补丁，
  底稿是拼出的整份，所以任务份末尾依次接上一份答案（`@@ your previous answer`）与修复说明（`@@ harness repair instruction`）。
  argv 不点名修复文件，只说这些在任务份的末尾（任务份怎么写见 [通道上限：超限的回复与送达核对](llm_agent_tool_protocol.md#通道上限超限的回复与送达核对)）。
- **会话内修复（2026-08-17）**：一个窗口的整条重试链共用一个
  `repair_session_key`（`correction-<chunk_id>`，由 `attempts.py` 传入）。第 0 次尝试开
  会话并记下 handle，之后每次修复带着它以 `session_scope=assignment` 回到同一会话，
  于是上面那条对 agy 的限制自然满足，其他 agent 也省掉整窗重发。**只在窗口内**——新
  窗口一定开新会话，跨窗复用仍然关闭（那是另一回事：跨窗复用在 agy 上实测净亏，见
  [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) [§3.1 agy 上的 A/B](llm_local_agent_experiments.md#31-agy-上的-ab复用是净亏2026-08-14)；重测协议在
  [`llm_followups.md`](llm_followups.md)）。
  `supports_session_reuse=false` 的 driver 拿到 `assignment` 是**发车前硬失败**而不是
  降级，所以先探能力，探不到就全重放；会话变冷（TTL / compact / CLI 被杀）则回落一次
  全重放，即本条改动前的行为。**额度耗尽与策略拒绝不走这条回落**——重建会话补不回
  额度，重试只会白发一次车，还会盖掉配额账本用来判冻结的两次连续失败之一。会话缓存
  按 `(provider_tier, model, chain)` 索引，与 driver 缓存同口径：session id 属于签发它
  的那个 CLI，而后续 attempt 是独立路由的，可能落到另一家或同模型的另一个档。这**不是**长驻 worker 的接线；`agent_session_mode`
  旋钮自 2026-08-19 起由这条路读取（四档：`api`/`per-window` 默认/`resume` 实验开关/
  `pseudo-conversational` 拒绝），本条描述的正是默认档 `per-window`，见
  [`llm_local_agent.md`](llm_local_agent.md) [§12.1.1 档位是唯一配置面](llm_local_agent.md#1211-档位是唯一配置面传输由它派生2026-08-22-起-per-window-默认即工具会话)。
- **重试预算是两档**（2026-08-19）：
  `--max-retries-per-window`（默认 3）是**一条会话链内**的修复次数；链用尽后
  `--max-replacements-per-window`（默认 2）次把窗口交给**全新会话**——修复上下文丢弃，
  agent 开新会话、无状态端点盲重掷，每次替换照常重新路由。一个窗口的调用总数是两个
  (n+1) 的乘积（出厂默认 4×3=12 次上界；典型路径「一两次修复就过」不变）。
  默认值从 5 / 1 改成 3 / 2（owner 2026-11-05），总数不变、多一次换会话：全部记录过的运行里，第一档修复第 1 轮
  129 / 166 通过、第 2 轮 15 / 38、第 3–5 轮合计 3 / 57；接连失败的修复 107 / 139 次与上一轮的错误一字不差，
  卡住时换全新会话才打断得了（读数与脚本在维护者的本地记录里）。pass 3 校对只有第一档，每窗因此从最多 6 次调用变成 4 次。
  `correction_window_retry.replacement` 记录下一次是否为替换。**两个独立旋钮而不是一个
  总预算**（owner 2026-08-19）：两档的单价差一个量级——第一档省的是不重开会话与可能的
  前缀缓存命中（不是「只发 delta」，assignment 每轮仍重发 protocol + run_context + 整份
  manifest），第二档是完整重放加冷缓存——所以「最坏 (档1+1)×(档2+1) 次」这个数严重高估
  真实成本，第二档保持「最大替换次数」的字面意思作为一等旋钮。原始论证在维护者的本地归档里。
  两档都登记在 `finesub.settings`（`[llm] max_retries_per_window` / `max_replacements_per_window`），命令行值优先。
  ⚠ 将来别的前端要带这两档，**两档必须同批**，不能只加一个。
- **回答报了没思考，再问一次**（owner 2026-10-17，`degeneration.thinking_of`）：纠错窗、说话人裁判与 pass 3 的校对，
  一份已经通过校验的回答，若这次调用要了思考却报上来 0，先不收，开**新会话**再问一次，两份里留一份。最明显的质量滑坡是模型
  完全不想（读数与判法的理由在 [`llm_followups.md`「模型回答的退化」](llm_followups.md#模型回答的退化)）。
  - 修复轮里补丁拼出的答案不判（owner 2026-10-24）：补丁只是几行、不是全量答案，不想是正常的，拼出的整份照样走完整校验。
  - 「要了思考」看送出的值（模型目录映射之后；agy 看实际送出的 `--effort`）：off / none / minimal / low 或者不送思考参数时，0 不算。
    映射之前的 low 不豁免，映射到 medium 的就是要它想。
  - 「报上来 0」要用量在、且这条路由会报思考：Gemini API 恒报（0 时省略字段）；本地 agent 看 driver 在调用记录上打的
    `thinking_reported`（Codex 报，agy 只对 Gemini 模型报，Claude Code、CodeBuddy、dsh 不报）；别的 API 看用量里有没有
    `reasoning_tokens`。用量全空的是缺账，不算。续着旧会话的调用：Codex、agy、CodeBuddy 的 driver 已把之前各轮减掉
    （`resumed_usage: subtracted`），agy 另用本轮步骤自己的思考（`turn_reasoning_output_tokens`）；减不了的
    （`cumulative`、`mismatch_kept`）用整条会话的累计——累计只会多算，它是 0 这一轮也是 0，大于 0 就当想过、不重问。现在按整次调用的合计判，不单看写答案的那一步。
  - 重问在新会话里（`fresh_session`）、不带修复上下文，**在两档预算之外**，最后一次尝试通过的窗也有；每窗最多一次。`resume` 档里
    这一问就是续会话链的轮换点：旧会话退役，之后的窗接在新会话上。
  - 留哪份（`degeneration.pick_after_retry`）：重问想了（或这次送的档不要求想）且通过校验，留它；想了却没通过，留原来那份；
    也没想（或判不了），比校验错误数，相同留原来那份。重问调用报错、批次熔断、额度停下（这一份照记，停照停），也留原来那份。
  - 记录：纠错窗的窗口记录、裁判与校对窗的 sidecar 条目多一个 `thinking`（两份各自的思考量与档位、`chosen`，重问出错时的
    `retry_error`）；纠错阶段的摘要里列一项「没思考、重答 N 窗（窗号）」。
- **思考漏进回答，取它末尾的答案**（owner 2026-10-17）：回答以 `thought\n` 开头时（agy 偶尔把推理当回答交出来），这三处都先剥掉
  推理、取最后一个完整的答案块再照常校验，校验不过才走修复轮；交换记录留原样，窗口记录记 `thought_stripped`。怎么认答案块见
  [`llm_local_agent_agy.md`](llm_local_agent_agy.md#7-答案通道本身不可靠约四分之一的回答是畸形的2026-09-02) 里 thought 泄漏那一条。
- **并行熔断按「会话链耗尽」计数，不按窗口**（2026-08-19）：两档之后一个窗口要烧满
  12 次调用才算失败，按窗口计会让熔断的价码随第二档翻倍（实测 4 lane 24→48 次）。
  按链计与改动前同价——链耗尽正是熔断一直在计的东西，两档之前「一个窗口失败」就等于
  一条链耗尽。同一个计数器也收「窗口彻底失败」，因为最后一条链走的是抛错路径、到不了
  链边界的钩子；熔断跳闸后在飞的窗在下一次 attempt 边界停手。
- **熔断不保证批次待在日额度以内，owner 2026-08-19 判定不必修**。实测（4 lane，
  第一档 5）：齐步 26 次调用、错峰 34–36 次；关掉第二档则两种调度都是 24 次。
  地板 24 = `lane 数 × 链长`，**不可约**——每条 lane 的首链都在任何证据出现之前就花掉了；
  多出来的部分是「信号还没到阈值时合法启动的替换链」，那些调用在发出的当下都有依据
  （彼时只有一个窗口失败，而第二档的定义就是「这个窗口的会话退化了」）。
  **超额不是失败模式，是一次路由事件**：多出来的调用换到 PerDay 429，而 429 要连续
  `DAILY_STRIKE_COUNT` 次才锁档（并发同时撞上的多个 429 由 `departed_at` 闸门判为**一次**
  观测），锁上之后该候选按 `daily_exhausted` 跳过，落到下一个免费 flash、直至付费尾。
  净代价是「在注定失败的批次上多几次失败尝试」加「该模型当日免费额度提前用完」。
  key 的用法与 serial 一致（2026-08-19 改判，并行不再额外钉 key，见
  [`llm_harness_routing.md`](llm_harness_routing.md)）：配额错误先在同一把 key 上原地
  sticky 重试，预算花光才轮到下一把，所以「免费档耗尽」不等于「这条链到此为止」。
  带媒体的调用仍钉一把（文件属于上传那把 key 的 project）。**被否的更紧方案**：替换链发起前预留配额能把错峰压到 ~20，但代价是
  「两个窗口各自只需要一次替换」就熔断掉一个本来能跑完的批次——拿真失败换一次配额事件，
  方向反了。真要卡死上限，杠杆是 `--parallel-windows`（2 lane 实测 24 次）或调小第一档。
- **只有输出上限（usage 主信号）、或重试用尽后 `<translated>` 明显截断（兜底）才对半拆分**；
  格式错误一律同窗口重试。Gemini `503 high demand` 只做同请求退避重试，失败后记
  `correction_window_call_error` 并停止——**不拆窗口**（那不是窗口太大）。重试预算内触发的
  拆分：前半窗继续用剩余预算、后半窗排队为新单元；在最后一次重试上触发的拆分则两个半窗
  都作为全新单元重新入队（各自完整重试预算，串行模式下重走 resume 检查），而不是以
  "failed validation" 终止。
- **采样参数**：生成调用默认显式传 `temperature=1.0`；validation/parse retry 的第 N 次
  logical attempt 用 `temperature=max(0, 1.0 - 0.01×N)`。只有**原样重问**（N ≥ 1 且不是修复轮：
  解析失败重问、替换轮）在末尾 user message 追加 `(seed=N)` 文本提示（没有哪家原生 seed 我们在用）；
  首轮与修复轮不带——修复轮靠自己那一轮已经和上一次不同，而首轮带了 seed，修复轮重发的首条 user
  就和发过的那条对不上，提供方的前缀缓存断在那里（[`provider-adapters.md`「缓存」](provider-adapters.md#缓存)）。
  成功后的下一个独立窗口/轮次从 attempt 0 恢复。`top_p` / `top_k` 不显式设置，保留 provider 默认。
- **Content filter 阶梯**（独立于上述 validation 重试）：`finish_reason=content_filter` 且空输出时，调用侧按注入 unit（URL 提取 / query 结果）逐级丢弃重建 prompt 再调；同任务黑名单跨窗口生效。全部丢弃后仍被拦则报错（疑似源文本触发）。准备任务（首条消息的 URL 预提取、回填消息的搜索块）、纠错请求回合与交付轮、知识更新均接入（知识更新与无检索注入可丢的请求回合仅原样重试）。

当前本地校验包括：

- 必须有且仅有一个 `<translated>...</translated>` 块。
- 列**按位置严格解析**，列数不符即结构性错误，**不做启发式重排**。basic 档（basicA，v104 起）7 列（`type|position|start|corrected_text|translation|conf|note`），`start` 须为数字；capableC 与三个 pass 的 pass 2（`capableC-grouped`）是 6 列（`type|position|corrected_text|translation|conf|note`）。带辅助列 `duration|gap|char_count` 的 10 列 / 9 列契约已删（2026-11-07，basicA 自 v104、capable 一族自 v99 起都不写），模型还写这几列的行照样逐行点名多出的列。
- 不带辅助列的契约**数格子**（owner 2026-10-23）：capable 一族 `position` 之后没有须为数字的格（basicA 只有 `start` 一格），多写一格的行会整体错位却照样通过（`sub|1|1.0|テスト|测试|high|` 会把 `1.0` 读成纠错原文、日文读成译文），所以每个 `sub` 行的格数必须与 header 相同，多一格少一格都判错、进修复轮，`note` 里不许有半角 `|`，行尾多一个 `|` 也不豁免（豁免的话，`note` 留空的错位行正好能过，而 `note` 留空最常见）。与哪一列在不在无关，以后去掉 `conf` 也照样成立。prompt 本来就要求文本里的 `|` 写成全角，这只是把要求变成校验；2026-10-23 读本机 1442 份旧回答，`note` 正文里真带半角 `|` 的 0 行。`discard|` 行不数格子（理由照旧可以附在后面）。行里带着 `duration|gap|char_count`、或连 `start` 一起写时，这一行的那条报错改成点名多出的列（仍是这一行的一条，修复轮照样只退这一行）；意外 `start` 列的检查对所有不带 `start` 的契约都跑；basicA 的行多写了这三列时同样点名。
  旧的 3/5/7 列容忍已删除（2026-08-07）。此前解析器靠单元格内容猜布局，多一列的行会被整体左移一格，把**未翻译的原文**写进 `translation` 且不报错。唯二的消费者都不涉及 `data/` 里不可再生的人工标注：`annotated.csv` 由纠错运行自己产出、可重跑，`tools/session_replay` 按约定为按需维护。
- 本地校验不拒 `<translated>` 每条 `sub` 引用的源序号数量，超过软上限只告警（按变体：capableC 4 个、其余 2 个，`CorrectionVariant.max_merged_sources`；三个 pass 的 `over_cap_rows` 按原始行数另记）；仍校验源序号存在、唯一、相邻且顺序正确。prompt 照各自的上限写（basicA：只接回词中切断的两源；pass 2：两源，三源仅限 filler 三明治；capableC：最多 4 源），不向模型暴露这一放宽。
- **合并行的绝对门槛（只有单 pass 的 capableC，owner Q13 2026-10-30）**：并了两个以上源的行，跨度（首源开始到末源结束，按窗口显示的 0.1 秒取整）或译文的加权字数（本地按译文算）超过窗口那一档的绝对门槛（标准档 7 秒 / 36，见[合并门槛与任务的长度比例](#合并门槛与任务的长度比例)），这一行校验不过、进修复轮（补丁修复轮只退这一行）；单个源本身超过的照收。basicA 不查；三个 pass 靠 `grouping.py` 的脚本护栏挡 pass 1 越限的 join（静默改成可并），pass 2 也不查。
- 源序号必须属于当前窗口，且每个源序号最多出现一次。
- **整窗错位**（`serial_shift.py`，2026-11-05）：模型中途漏掉或并掉一个源、编号却照常往下数，之后每行写的是后面某个源的话，
  窗口结尾缺几个源；修复轮只补结尾那几个源就能过，前面的错位原样交付（3.5 Flash Lite 最常见，efficiency 的生产纠错组就是它）。
  所以逐行比纠错原文与源文本：规范化（NFKC、小写、片假名转平假名、去标点空白）后用 `difflib` 的 ratio，整窗里最像的别的源
  至少 0.5、并且比自己那几个源高出 0.15，这一行算跑偏（整窗找，因为漏掉一长段时偏移可达上百个源）；**连续 5 行**跑偏、
  而且像的是至少 5 个不同的源，就判错位（一个反复出现的词改成 ASR 在别处的写法，几行都像同一个源，不是错位）。加一条窗口级
  错误，写明从哪个源起、那一行写的是哪个源的话、起点那个源自己的话是什么（不要它就写 `discard`，不能略过），结尾若缺源也是它的后果。只比结构校验过了的 `sub` 行；单独一行跑偏
  （假名改汉字、听错的词改掉）不算；和自己那几个源已经足够像的行（相似度 0.85 以上）不找。
  从第一段跑偏的起点到最后一段的终点都送回去：两段隔得很远时中间那段也一起重做，多半是整份重写（少见）。修复轮改不好、
  用尽重试的，和别的校验失败一样让运行停在这一窗（`RuntimeError`），续跑时只重做这一窗。一份回答平均多花约 20 ms，真实回答里最长约 0.5 s
  （构造的最坏情形：340 行英文、每行都大改，约 1.7 s）。
  读数（2026-11-05，日语约 20 种模型 1893 份、英语与中文 Flash Lite 55 与 33 份真实回答）：判出的 42 份逐份核对都是真错位，
  其余 1939 份没有误报；在没错位的回答里人造的六种错位上（含跳过十几到几十个源的），5–10 行的段检出 92%–99%，20 行以上
  全部检出，合法的合并与丢弃不误报。3.5 Flash Lite 的修复轮在 18 个真实错位现场、各跑两遍的 36 条链里，35 条 5 轮内交出不错位
  的回答（22 条第一轮）；原来对的行几乎没动（改了的多是顺手改正原来的错）。判不了的：纠错原文与源不是同一种语言的（曾见中文素材被写成日语），以及错进一串
  几乎一样的行里的；并了三四个源的行成串错开时，只拿单个源比，相似度可能不到 0.5，会漏。会误报的：同一段内容在窗口里出现两次
  （副歌、固定的口播），ASR 把其中一次听坏、模型照另一次连着改对 5 行以上，就像是错位：多一次修复轮，模型坚持改对的写法就用尽重试、运行停在这一窗；已有数据里没见过。决定与理由见 [`llm_design_notes.md`](llm_design_notes.md)「纠错整窗错位的决策记录」。
- **占位答案**（`output_protocol._with_placeholder`，2026-11-09，移植自 Nonoka X 的引擎补丁 0009）：结构全过、却不是字幕的回答整窗退回——
  5 行及以上的 `sub` 行每行的纠错原文和译文（只纠错时是 `polished`）都一样、而窗口的源不全是同一句，或过半的 `sub` 行有一格是占位词（`text`、`test`、
  `placeholder`、`todo`、`tbd`，不分大小写）。Nonoka X 见过 agy 的真答案连续被工具守卫拒掉之后（守卫改成按字节读载荷之前的事，见
  [`llm_local_agent_agy.md`「4 准入硬门：`view_file` 的读取边界」](llm_local_agent_agy.md#4-准入硬门view_file-的读取边界)），模型交了 52 行 `text|text` 的试探答案，逐行校验全过，那一窗字幕全成了 "text"。
  这条错误不归属任何一行，所以修复轮整窗重写、不走补丁；续跑回放走同一个校验，缓存里这样的窗会重做。单独一行短句、
  不到 5 行的窗口里重复的「うん」不受影响。读数（2026-11-09）：本机 `data/` `out/` `tmp/` 里 2409 份不重复的真实纠错回答，一份都不拒。
  源本身整窗都是同一句（反复的口号）时不判整窗同句，所以剩下会误报的只有源各不相同、却该写成同一句的 5 行以上的窗，难以想象。
- 同一行和不同行的源序号都必须保持源时间顺序。
- 空 `<translated></translated>` 合法，表示当前窗口全部丢弃。

拼接规则（每个源序号用它所在核心区的那个窗口的行；`output_protocol.merge_translated_csv_windows`，在 `CorrectionRun.commit_window` 里逐窗调用；决策与理由见 [`llm_design_notes.md`「纠错窗口拼接改成 core 归属的决策记录」](llm_design_notes.md#纠错窗口拼接改成-core-归属的决策记录2026-11-08)）：

- 所有窗口输出先解析为带源序号的字幕片段。窗口 = 重叠前缀（`overlap_segments`，上一窗口的尾部）+ 核心区；三个 pass 的 pass 2 的分组窗先展开回原行再拼。
- 重叠区留上一窗口的行；新窗口对重叠区的回答（包括它在重叠区写的 `discard`）不用，下面闭包改归新窗口的序号除外。接缝因此落在切点，切点由窗口规划对齐到句末或停顿。
- 新窗口核心区的行替换旧结果里落在核心区的序号；核心区里新窗口有意丢弃的序号保持丢弃。
- 新窗口把重叠区末行和核心区首行并成一行时，这一行整行用新窗口的，它带走的重叠序号改归新窗口；接着闭包：凡与改归的序号相交、整行落在新窗口内的旧行与新行，它们的序号一起改归，直到没有窗口内的行被切开。行永远不切开。
- 闭包碰到伸出新窗口起点之外的旧行（上一窗口把重叠起点前后并成了一行）时，那行旧行整行保留，与新行共有的序号重复出现一次，换来不丢字幕（切行就要丢一半文字）。
- **每窗拼接后核对覆盖**（`output_protocol.window_merge_gaps`）：有过行的每个序号恰好一行；可以没有的只有两种：窗口内新窗口没有回答的序号，和只有新窗口在重叠区回答过的序号（上一窗口把它丢弃了）。丢了或重复了就打一行 `Warning:`，点名窗口与源段号，**不中止任务**（窗口都已付过费，手补一行比重跑便宜）。前面的窗已经报过的重复不再重报。上面那种整行保留造成的重复会在这里报出来。
- 加入新窗口字幕片段后按原始时间轴排序并重新编号。

## 知识库更新行为

知识库结构、任务反馈采集与统一知识更新的完整行为见 [`knowledge.md`](knowledge.md)；本文其余章节仅涉及采集开关对纠错 prompt/resume fingerprint 的影响。

## 任务级并行与 agent 槽位预算（2026-08-30 落地）

多个纠错 run 可同进程并行（设计与验收记录在维护者的本地归档里）。
现行行为：

**三个旋钮，语义分开。** `[llm] local_agent_max_parallel`（默认 4）是**物理上限**——本机 +
订阅同时活跃的 agent CLI 进程数，**每个 vendor 一份**进程级预算（`AgentSlotBudget`，按
driver_id 共享——同订阅的不同模型共用一池；agy 的 tool-slot project 按 domain root 共享，
两个并发 run 拿到不同的 `.finesub-tool-<slot>`，不在同一个进程里也一样：每个 slot 由锁文件跨进程独占）。`finesub.pipeline --max-parallel-tasks`
（默认 2）是**准入上限**——同时活跃的任务数，纯 API 任务同样受它约束。
`--llm-parallel-windows`（默认 1）是**单任务的意愿上限**。三者都是上限不是配额：约束是
瞬时槽位占用，三者都由用户覆盖。**它们不是彼此独立的**：保底格在**任务起步**就预留，所以
`local_agent_max_parallel` 小于 `--max-parallel-tasks` 时，第二个任务会卡在起步的 `reserve`
上等到第一个任务整体结束——连它本可以先跑的纯 API 阶段也一起等。要并行多个任务，物理上限
必须不低于准入上限。

**保底 lane（不变式 I1）。** 路由链可达 agent 后端的任务起步时在**每个可达 vendor 的预算**里
各 `reserve` 一格（`TaskSlotAccount` + `TaskSlotClaims`，按 catalog 过近似判定需求；调用落到
哪个池按次决定，只保首池会让路由到第二家 vendor 的调用没有保底——reviewer 2026-08-30 P1；
纯 API/测试档不占）。必得 lane 的每次调用在**本池的** claim 上锁内一步兑现（reserved→held）、
调用结束摆回（held→reserved），**任何时刻都不会被别的任务的可选扇出饿死**——可选调用只吃
`free = limit - held - reserved`。pseudo-conversational 的长驻 host 在创建线程捕获 claim 集、
由 supervisor 一次 enter 消费保底，CLI 退出才摆回；task 结束时保底仍被 host 占着的话由那次
enter 的退出直接释放回 free（不炸收尾、不漏 reserved）。兑现从不等待：等待意味着账面与池
脱节，直接报错。

**动态扇出。** 并行纠错的每个阶段取 `want = min(1 + ⌊free/A⌋, parallel_windows, 待跑数)`
（`A` = 进程内有 agent 需求的活跃任务数）。批繁忙时 `claim_cap→1`，每个任务自动退化为
单 lane 串行——语义仍是 `continuity=parallel`（advice 链已撤），只是顺序执行；`continuity`
永远不由分配器改写。

**lane = run 发放的逻辑 ordinal**（`run_context.LaneOrdinalPool`）。lane 是对话身份：`resume`
会话与 pseudo host 都按它键。ordinal 属于 run 而非线程/池——查询与纠错两个先后线程池领到
同一组 1..N、命中同一批会话；串行段（research/知识更新）的 lane 在阶段期间借给池、之后取回。
pin（知识库快照）、session registry、reporter、lane 同走线程池 initializer
（`run_context.bind_llm_worker`）进 worker——ContextVar 不会自己进新线程。

**conversational 强制 serial。** 纠错格绑定 conversational 后端时 `continuity=parallel` 被
强制改写为 serial 并告警（`conversational-forced-serial`）：一个人的 agent 是一条队列，扇出
只会剥掉 advice 台账而不省墙钟。它不占 driver 槽（不进预算）。超长素材的分块调查不在任务内并行面里：
块一律串行，后块要前块定下的名字（见 [`llm_harness_research.md`「超长素材分块调查」](llm_harness_research.md#超长素材分块调查2026-08-30-落地plan-w7--p8)）。

**知识库侧**：并发 run 各读各的 pin、写路径进 per-root 单写者队列、真并发冲突按类型化解，
见 [`knowledge.md`](knowledge.md)。**batch 侧**：组串行 / LPT / 调度序记录，见
[`batch-scheduler.md`](batch-scheduler.md)「llm bin 的并发语义」。**上限怎么定**：owner
2026-08-30 裁定不做标定实验——并发上限是用户偏好（各家 agent 对并行流量宽容；代价是 5h
额度被快速耗尽导致 session 中途白费，以及墙钟收益边际递减、token 效率与上下文损失边际
递增），选值指引在 [`manual/agent.md`](manual/agent.md)；出厂默认 windows 1 / tasks 2。
