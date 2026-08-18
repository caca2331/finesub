# 本地检索代理与背景调查

> 从 [`llm_harness_behavior.md`](llm_harness_behavior.md) 拆出（2026-08-16，原文 930 行）。
> 那份仍是 harness 运行时行为的总入口；本文是它的一块，单独成篇是因为读者不同。

## 本地检索代理

gemini-3.x 免费层级不开放 `google_search` grounding（实测立即 429），所以纠错/调查模型不直接启用联网工具；检索由 `finesub/llm/web_search.py` 在本地执行，含 **search**（网页搜索）与 **extract**（单 URL 深度整页提取）两类。Gemma4 fallback 使用 `gemma-4-31b-it` + 通用 Search grounding 免费配额（约 1500 RPD），区别于 Gemini 3 免费层级专用 grounding 0 RPD：

- search 顺序：Exa（`type:"deep"`，`contents.highlights.query` + summary，`x-api-key`）→ Gemma4 grounded（`GEMINI_FREE`，默认 `<|think|>` + 中等深度；若接地 metadata 为空，会用同一请求去掉可见 thinking token 自动重试一次）→ Tavily（`auto_parameters` + `include_answer=advanced` + `max_results=10`，Bearer）→ 免 key 的 DuckDuckGo HTML 兜底。
- extract 顺序：Exa `/contents`（summary + highlights）→ Gemma4 grounded（prompt 中先把 URL 百分号转义还原为标准字符）→ Tavily `/extract`（`chunks_per_source=5`）→ 暂无本地兜底（全部失败则返回错误结果）。本地 search/extract 之后可接开源本地检索 MCP，目前 DuckDuckGo 仅作 search 兜底。
- key pool：`.env` 的 `GEMINI_FREE` / `GEMINI_PAID` / `EXA_KEYS` /
  `TAVILY_KEYS` 只保存 `{name:key,...}`；根目录 `config.toml` 的 `[pools]` 按名字
  筛选和重排，[`config.example.toml`](../config.example.toml) 为模板。空/缺失 pool
  默认取 Gemini Free 前 2 把、Exa/Tavily 前 3 把，Gemini Paid 默认全取且无推荐
  上限；显式 pool 超过推荐数只告警、不截断。`[providers]` 可关闭 Exa、Gemma4
  grounded、Tavily 或 DuckDuckGo；Gemma4 复用选定的 Gemini Free pool。未配置 key
  或被关闭的 provider 静默跳过（不产生 fallback 事件）。
- key 不可用判定：Exa 遇 401/402/403/429、Gemma4 遇 401/403/429、Tavily 遇 401/403/429/432/433 视为该 key 不可用，在 runtime state 目录中按 provider 锁定 24h 并选 pool 内下一 key 重试；源码 checkout 默认 `<root>/.state`，可用 `FINESUB_STATE_DIR` 覆盖，wheel 无 checkout 时使用用户 state 目录。某 provider 全 key 锁定/失败即回退下一 provider。Gemma4 如果没有返回 usable `groundingMetadata.groundingChunks`，也视为 provider 失败并继续 fallback。全部失败则该 query/URL 记为"搜索失败/提取失败"，流程继续，不中断任务。
- **引导语（guided query）**：search query 与 extract URL 均可带一句话引导语——对 search 映射到 Exa 的 `highlights.query` 与 Gemma4 的 `search_goal`（Tavily search/DDG 无对应即忽略），对 extract 映射到 Exa `highlights.query` / Gemma4 `extract_goal` / Tavily extract `query`；只影响网页重点提取方向，不改变搜索关键词。
- `search_many`/`extract_many` 按 `(query|url, 引导语)` 去重、按上限截断（背景研究第 0 轮 `min(20, 8 + sqrt(raw字幕片段数)//10)` 条、loop 追加轮为其一半（向上取整）/ 每纠错窗口 8 条；extract 仅 loop 追加轮可发起，见下），并做 1.5s 限速（Exa 的 10 qps 限制远宽于此）。
- Gemma4 search 单次 pass 最多接收 8 条 query；若 pending query 更多，会按 8 条一批自动分批调用。Gemma4 grounded REST 调用单独使用 1200s timeout（Exa/Tavily/DDG 保持通用 timeout）；多次真实测试中，触达最大输出时耗时可到约 900s。
- Gemma4 search 单 query 注入量级（2026-07-10 真实 8-query smoke）：按 `render_search_results([result])` 计，观测范围约 283–1079 tokens，中位数约 1062 tokens。Google grounding 的 redirect URL 较长，title+URL token 往往占主要部分；预算估算可先按 300–1200 tokens/query 记，source-heavy 查询留到 1500 tokens/query 更稳。
- 结果按 query 分组渲染（provider、标题、URL、摘要，含长度截断）后注入 `<search_results>` 块；query section 用 `--- query: ... ---`，深度提取 URL section 用 `--- 深度提取 url: ... ---`，避免 Markdown `###` 标题与正文错位。图片及其 URL 在清洗阶段丢弃。背景调查启用多轮 loop 时，Round 2 注入的是整理后的 Evidence Pack 而非原始结果。
- **extract 目标只能来自已展示给模型的 URL**（2026-08-13）。search loop 维护一张「见过的 URL」表，语料是模型确实看到过的全部文本：background、搜索结果条目的 URL 与摘要、注入的知识库 index 与词条正文、**以及已提取页面的正文**（owner 决定收，见下）。词条会引用来源，漏收它会让拒绝理由「该 URL 未在此前的输入中出现过」对一条 harness 自己递过去的链接说假话；词条是 harness 自有资产，威胁面严格小于已收的页面正文。模型自己写的文本（progress 台账、evidence pack）永远不是来源，否则它能自造目的地。模型写的 URL 不直接发出去，而是经 `normalize_url_key` 查表、命中后**发表里那条原文**——没有任何一个模型写的字节到达网络。查不到就不执行，写进该轮 `rejected_extract_urls` 并在下一轮的 `<previous_search_request>` 快照里标注理由（静默丢弃会让 judge 每轮重发同一个 URL 直到额度耗尽）。
  - 归一化只折叠「同一个链接的不同写法」：http/https 互认、host 小写 + 去尾点 + IDN↔punycode、去默认端口、去 fragment、HTML 实体解码、percent-encoding 只对 RFC 3986 unreserved 字符规范化。**query string 保留**（能携带载荷的正是它）：参数顺序与全部 reserved 转义（`%2F`/`%3D`/`%26` 等）原样保留，只折叠 unreserved 转义（`?a=%7Ex` 与 `?a=~x` 指向同一资源），因此签名类参数也不会被改坏。因为最终发的是表里的原文，宽松匹配的后果只是「取到另一条已见 URL」，不是「到达新目的地」。
  - 这道闸挡的是**编造的目的地**，不是被选中的目的地：语料含已提取页面正文，攻击者可在自己页面里投放链接让模型看见。残留通道是「选哪一条、什么顺序」，每次约 log₂(N) bit，且 extract 有额度上限。search query 不设此限——它只发往 Exa/Tavily/DDG，注入攻击者没有接收端。

### 已验证的 API 响应结构（Exa/Tavily 2026-07-04，Gemma4 2026-07-10 实测）

四个接口均用真实 key 打通（HTTP 200），响应字段与 `web_search.py` 的解析一致；清洗后只保留文本，`image`/`images` 一律丢弃。

| 接口 | 顶层键 | `results[0]` 字段 | 解析取用 |
| --- | --- | --- | --- |
| Exa `/search`（`type:"deep"`） | `requestId, resolvedSearchType, results, searchTime, costDollars` | `id, title, url, highlights[], summary, image` | `title` / `url` / `summary`+`highlights`（`_exa_snippet`） |
| Exa `/contents`（extract） | `requestId, results, statuses, costDollars, searchTime` | `id, title, url, author, highlights[], summary, image` | `title` / `summary`+`highlights` |
| Tavily `/search`（`include_answer=advanced`） | `query, follow_up_questions, answer, images, results, auto_parameters, response_time, request_id` | `title, url, content, score, raw_content` | `title` / `url` / `content`，顶层 `answer` | 
| Gemma4 grounded `generateContent` | `candidates, usageMetadata` | `groundingMetadata.groundingChunks[]`, `groundingMetadata.groundingSupports[]`, content JSON block | `groundingChunks.web.title/uri` 作来源；`groundingSupports.segment.text` + `groundingChunkIndices` 作支持片段；content JSON 仅用于按 query/URL 分组和摘要 |
| Tavily `/extract`（`chunks_per_source=5`） | `results, failed_results, response_time, request_id` | `url, title, raw_content, images` | `raw_content`（回退 `content`）；`failed_results` 用于报错 |

要点：`type:"deep"` 被 Exa 接受（返回 `resolvedSearchType`）；`include_answer=advanced` 实测返回非空 `answer`；guided query 经 `highlights.query` 生效（Exa summary/highlights 会围绕引导语聚焦）。Gemma4 返回必须含 grounding chunks，否则不把未接地模型正文注入。Exa 接口返回 `costDollars`，属计费/额度调用。

## 背景调查（按轴退化的 r1 / r2）

r1、r2 都是独立 API 调用（非多轮对话），角色 `general_capable`。共同输入是带窗口标记
的紧凑字幕文本（`源序号|文本`，边界处插入 `--- window N ---`）和用户额外信息；是否
运行、输出哪些块由 `retrieval` 与知识索引实况决定，和 `media` 无关。`retrieval=native`
时只有 r2 开原生搜索工具；其余研究调用不启用模型工具。

- **r1**：`retrieval=local` 时必跑；`native|none` 仅在有可读知识索引时跑。知识开启时
  注入 streamer/common index，并可按用户备注的 key/alias casefold 子串匹配预注入最多
  8 条词条。仅在 r2 会运行时输出 `<analysis_notes>`（≤1500 token）；仅在有知识输入时
  输出 `<requested_entries>` / `<keep_entries>`；仅 `local` 输出搜索 query/contract。
  request/keep canonicalize 后各≤8、合计≤12，keep-first，超限从 request 尾部丢弃。
- **取数**：仅 `local` 执行 query；单轮时直接调用本地检索代理，多轮时运行下述 search
  loop。知识请求由 harness 从本地 KB 加载。`extra_info` 中最多 8 个 URL 也只在 local
  路径于 r1 前做深度提取。
- **r2**：`local|native` 运行，`none` 跳过。local 消费 r1 notes、知识词条与本地搜索结果
  或 Evidence Pack；native 消费知识词条并由本次调用的原生搜索补足背景。两者都输出
  `<context_pack>` JSON：`general_context` 注入所有窗口；harness 把模型按窗口 id 给出的
  `window_contexts` 绑定为 `{window_id, first_source_id, last_source_id, context}` 区间后落盘。
  纠错时按正文 source-id 覆盖匹配：优先拼接完全落入当前窗的旧笔记，否则取包含当前窗的
  最小旧笔记；横切旧区间则不注入。拼接上限 8,000 token，截断写
  `window_context_truncated` artifact（上限常量与其他注入上限一起在 `config.py`）。
  笔记没有区间的 context（模型刚吐出、尚未绑定的中间态，或落盘早于区间寻址的旧产物）
  **不可注入**，因此读取时直接判为不兼容：走 `*.invalid[.N]` 备份并重跑调查，而不是
  只拿 `general_context` 交差——那等于把用户已经付过配额的逐窗笔记静默丢掉。
  绑定时窗口 id 不在计划里的笔记无处安放、只能丢弃，此时落 `window_context_bind_report`
  并向 stderr 告警：r2 把 id 系统性写错会丢光逐窗笔记，而产物看上去完全正常（已绑定、
  0 条），上面那道「未绑定即不兼容」的闸恰恰看不见这种情况。

因此 `media=text` 也可以有完整背景调查；旧的“文本路线不调查、关键词条直塞纠错窗”
路径已经删除。`retrieval=none` 只可能跑一次挑词条的 r1，不生成 context pack。

### 多轮搜索 loop（默认开启，最多 3 轮）

`finesub/llm/search_loop.py` 实现可插拔的多轮搜索：只替换发起侧 prompt 片段（`fragment_search_queries_output_v1` → `fragment_search_contract_output_v1`）和消费侧片段（`fragment_search_results_usage_v1` → `fragment_evidence_pack_usage_v1`），其余调用面不动。当前只有背景调查接入；窗口查询轮保持单轮（少一轮调用，兼作效果对比组）。

三个数据结构：

- **Research Contract**（Round 1 输出的 `<research_contract>` JSON）：`goal`（这次调查要什么）、`facts[]`（`{id, fact, priority 1-5, done_when, hints}`，≤12 条）、`out_of_scope[]`。priority 由 harness 机械维护：某 fact 被追加轮 query（`F1|query` 前缀标记）覆盖过一轮则 -1（最低 0）。**priority 只是提示，降到 0 仍可查**，是否继续由 loop 模型自行判断；防死磕的硬边界只有最大轮数。
- **Research Progress**（harness 维护的累积台账）：loop 模型每轮输出 `<progress_update>` 增量（≤2000 token，按 fact id 记 confirmed/partial/not_found/dead_end + 结论 + 来源，另有"新发现/死胡同"），harness 以 `## 搜索轮 N` 头拼接。只在 loop 内部流转。
- **Evidence Pack**（`<evidence_pack>` markdown，最终注入 Round 2）：`## 结论` / `## 关键证据摘录` / `## 未解决` 三节；harness 统一在头部注入"由搜索代理整理、仍需交叉验证"声明。（实验模板 `search_loop_v2.md` 另有第四节 `## ASR 误听候选`——疑似误听→候选对应表，仅为线索不构成纠错指令。v2 只能经 `tools/session_replay --loop-version v2` 触发：生产调用方 `research.py` / `stages/fast_session.py` 均不传 `loop_version`，走默认 v1 三节。）

流程：Round 1 产出 contract + 第 0 轮 query → 本地执行 → 每轮搜索后按本次 run 的 difficulty 选择 `search_judge` cell（纯文本；解析重试沿用同一 difficulty，输出上限 32,768=SESSION_OUTPUT_MAX_TOKENS）做筛选、去重、抽证据，输出 progress 增量并决定"继续检索"或"生成 Evidence Pack"。judge 输入在 `<knowledge_entries>` 前明确列出上一调用的 requested/kept entry 原名；在 `<search_results>` 紧前注入 `<previous_search_request>`，其中是发起时的 contract 快照和经过 cap/去重后实际执行的 query/extract，另保留 `<current_research_contract>` 供下一步判断。继续检索时可同时给出 `<search_queries>`（`fact_id|query` 前缀，可带 ` >> 引导语`）与可选的 `<extract_urls>`（对已出现在结果中的 URL 发起深度整页提取，可带 ` >> 引导语`；**只能从已展示给模型的 URL 中挑选**，见下）——**extract 仅此评审代理可发起**（主调查/纠错查询轮不发起）；两者跨轮去重，**去重键是「请求 + 引导语」，且解析层、loop 层、`search_many`/`extract_many` 三层一致**——引导语改变 provider 的重点提取方向，所以同一 URL/query 换个提取重点是新请求而非重复（2026-08-13 前三层都按 URL/query 单独去重，模型换角度的第二次请求会被静默吞掉；其中客户端那层最隐蔽：loop 已扣额度并在快照里报告「已执行」，请求却从未发出）；合计计入追加轮上限（=第 0 轮的一半），其中每条 query 计 1、每 2 条 extract URL 计 1（预算按半单位计：query 2 半单位、URL 1 半单位）。不在已展示 URL 白名单中的 extract 请求会被拒绝、写入该轮的 `rejected_extract_urls`，并照常消耗 1 个半单位，避免用大量伪造 URL 绕过快照上限；畸形或越界端口 URL 直接视为不匹配。深度提取结果与搜索结果**合并为一个预算块**渲染进下一轮（单 section 4k token、整块 `本轮cap×2k+4k` token，块尾附"注入预算说明"列出被截断/丢弃的条目）。priority 递减在渲染**之后**执行：只有结果**完整进入**渲染块的 query 才会使其 facts 的 priority -1；渲染 section 的标签就是请求身份（`query`／`query >> 引导语`，extract 同理），递减因此按标签而非位置对齐，同一 query 配不同引导语不会互相顶替。被截断/丢弃的 query 视同未执行（模型可在后续轮重发，不算重复）。非末轮的 judge 还可输出可选 `<requested_entries>` 块（每行一个知识库 key/别名，独立上限 = 追加轮 query 上限、不占检索用量）：harness 解析后把词条全文按预算渲染注入下一轮 `<knowledge_entries>`，跨轮按主 key 去重；两份知识库 index 注入每个非末轮（末轮置空并禁止请求）。轮次提示来自模板 fragment（`fragment_search_loop_{continue,final}_notice_v1`，Python 只按是否末轮选并填剩余轮数）：非末轮提示"仍有 priority ≥ 2 且 partial/not_found 的 fact 时默认继续检索、不要过早收尾"，到达最后一轮时强制要求输出 pack、不得再输出 query/extract/词条请求。若 judge 在**非末轮**就产出 pack 而累积台账里仍有 priority ≥ 2 未决 fact（按每 fact 最新状态判定），harness 落一条 `premature_evidence_pack` 告警 artifact（仅告警、不阻断收尾）。Progress/Evidence Pack prompt 各带 one-shot，并要求重要 confirmed/partial fact 尽可能保留多条支持、补充或冲突证据；只有原结果中的逐字内容可标为引文，否则必须标为摘要。loop 模型调用异常、解析重试耗尽（默认 max_parse_retries=5）、或末轮拒不输出 pack 时降级回退：用 Progress 台账 + 全部原始搜索结果拼一个降级 pack，不让任务失败。contract（含递减后的 priority）、逐轮元数据与 evidence pack 摘要持久化到 `research-context.json` 的 `search_loop` 字段。

v1 厚度要求（prompt 层）：`<progress_update>` 每条 fact 可含 2-3 句（核心判断 + 相关上下文：读音变体、关联实体、出现场景、交叉印证或冲突），软上限约 2000 token。`<evidence_pack>` 的 `## 结论` **必须逐条覆盖 Contract 每一个 fact**（含 priority 0），每条至少 2 句（核心事实 + 相关上下文），至多 4 句；`## 关键证据摘录` 定位为下游交叉验证的唯一依据，每个独立来源单独成条、不合并，每个 confirmed/partial fact 尽量 2 条以上。harness 在 pack 提取后做逐 fact 覆盖软校验：`## 结论` 中缺失的 contract fact id 记一条 `evidence_pack_missing_facts` 告警 artifact（仅告警、不阻断）。one-shot 仅示范格式与信息密度下限，不限制条数与篇幅。

**v2 变体**（`search_loop_v2.md` + `search_loop_user_v2.md`，`build_search_loop_v2_messages`；生产尚未接入，仅 session_replay `--loop-version v2` 可用）：取消 v1 的"继续检索 OR 收尾"二选一，改为每轮**必须**输出完整 `<evidence_pack>`（在上一轮 pack 基础上更新），可选输出 `<search_queries>`/`<extract_urls>`；harness 以"无 query"作为终止信号。取消 `<progress_update>`，跨轮连续性由 `<previous_evidence_pack>`（上一轮的完整 pack）承载。结构更简、省约 400 output tokens/轮、pack 始终可用（无降级路径）；厚度与 v1 持平（flash-lite thinking=0 下证据摘录仍为 1 条/fact 的天花板）。

预算与失败处理：

- 每轮输入必须满足 `prompt_input_limit = 194000`；超限直接报错，提示先切分音频。不做 map/reduce。
- 模型输出标签块/JSON 解析失败时同请求最多再试 5 次（`max_parse_retries=5`，共 6 次调用），仍失败则任务失败（`<context_pack>` 缺失时会尝试直接解析裸 JSON 兜底）。
- 每轮响应、usage token 计数和解析错误会写入 task artifact（如指定 `--task-artifact-dir`）。
