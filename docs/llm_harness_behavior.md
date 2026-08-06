# LLM 纠错翻译 Harness 行为说明

状态：实验性实现。默认只生成计划和中文 prompt；真实 API 调用必须显式使用 `--execute`。

## 翻译路线与档位（--route / --level）

严格 6 档 preset（不提供自由开关组合；设计意图与决策记录见 `docs/llm_design_notes.md`，实现在 `src/llm/profiles.py`）：

| preset | 媒体 | 检索 | 纠错角色 | 输出系数 c |
| --- | --- | --- | --- | ---: |
| text-low | 无 | 无（低思考） | `audio_multimodal`（3.6 优先） | 2.0 |
| text-med | 无 | 无 | `audio_multimodal`（3.6 优先） | 3.5 |
| text-high | 无 | 模型内建搜索工具 | `internet_capable` | 4.5 |
| mm-low | 无 | harness 外部注入 | `audio_multimodal`（3.6 优先） | 4.5 |
| mm-med（默认） | 音频 | harness 外部注入 | `audio_multimodal`（3.6 优先） | 5.0 |
| mm-high | 音频+视频 | harness 外部注入 | `audio_multimodal`（3.6 优先） | 6.0 |

- **mm 路线**的定义特征是 harness 侧外部注入：两轮背景调查、每窗查询轮、本地搜索代理（mm-low 无媒体但保留注入）。**text 路线**不跑任何 harness 检索/调查（`--context-file`/`--research-only` 传入即报错）；text-high 只启用纠错模型自带搜索工具（Gemini 3 免费层 grounding 429，需为 `internet_capable` 角色配置支持原生搜索的模型）。
- 预期输出估算 `k × c × csv_tokens`（k 为 `--output-scale`，默认 1.0；调大 k 切出更小窗口）；常规窗口须满足 `≤ 0.9 × 65536 − 5000 = 53982`。
- **insert/插轴已于 v63 全面废弃**（所有路线、所有变体均不再注入或接受 insert 行；校验层视 insert 为结构性错误、同窗重试）。
- 纠错调用 thinking：text-low per-call 覆盖为 low，其余全部 medium（见「LLM thinking effort」）。

## 快速模式（--fast auto|on|off，默认 auto）

短输入把整段作为**单个融合窗口**处理：调查 Round 1 与查询轮合并为快速第 1 轮（`fast_round1`，mm 路线：音频/音视频整段剪辑 + 全量 CSV + 两份知识库 index + 本地预注入词条，输出 `<analysis_notes>`（≤2000 token）→ `<requested_entries>`（新请求）→ `<keep_entries>`（保留预注入）→ 搜索 query/contract），随后可选多轮搜索 loop（`--fast-search-rounds`，默认 2 轮，`--research-search-rounds` 不适用于快速运行），产物（evidence pack / 原始结果 + notes + 合并后的条目详情）直接种子进唯一纠错窗口——第 1 轮上传的剪辑也被纠错轮复用，不重复上传。text 路线快速 = 单窗直接纠错（无注入）。

auto 判定三个条件都过才启用（结果与数值写入 `fast_decision` artifact）：

- 输出：`k × c × 全量 csv_tokens ≤ 0.8 × 65536 − 10000 = 42428`；
- 输入：第 1 轮 prompt 文本（countTokens）+ 剪辑媒体 token ≤ `194000 − 56000`（预留第 2 轮注入空间）；
- 质量护栏：整段 `<asr_result>` ≤ `max_window_subtitle_tokens`（默认 10,000，见「窗口拆分」）。
  快速窗口按定义就是全片，是最容易撞上这条的路径，所以它与两个预算条件同级参与判定。

`--fast on` 不满足即报错退出；`--fast off` 强制常规多窗流程。快速会话产物写入同一个 `*-research-context.json`（带 `"mode": "fast"` 标记，位于 artifact 目录），`--context-file` 复用与常规调查一致；resume 指纹额外并入种子注入内容的哈希。

## 输入与输出

输入以 `*-stable.json` 为字幕源。纠错翻译阶段会保留下列中间 artifact：

- `*-raw.srt`：从 `stable.json` 渲染出的完整 ASR 原文 SRT，便于人工排查；若请求的最终 postprocess profile 包含时间轴步骤（当前为 0/1），仅同步执行重叠修复 + 末端延长/闪轴闭合（profile `4 → 1`），文字保持原样。
- `*-corrected.srt`：从模型输出的 `corrected_text` 列渲染出的纠错后原文 SRT，便于分析 ASR 修正和误听模式。
- `*-translated.srt`：模型输出的中文字幕直出版本，尚未做最终 SRT 后处理。
- `*.srt`：最终中文字幕 SRT；默认由 `*-translated.srt` 经过 profile 0 后处理得到。
- `*-research-context.json`：两轮背景调查的产物（context pack + 轮次输出（含 analysis_notes/contract）+ 多轮搜索 loop 元数据 + token 报告），写在 task artifact 目录下（如 `input.llm-artifacts/input-research-context.json`）；旧 run 若仍在 SRT 同级，首次触达时会迁入 artifact 目录。

纠错窗口输入格式是：

```text
global_id|local_start|duration|gap|text
```

时间单位为秒，展示到 0.1；`local_start` 以本窗口**剪辑音频的 0 秒**为基准，每条 CSV 记录占一个物理行。传给多模态纠错模型的音频是原始音频（不是人声分离后的 `*-vocal.ogg`；旧产物可能为 `*-vocal.flac` 兜底格式）按窗口裁剪出的 mono-16k AAC 片段（ffmpeg：`-c:a aac -ac 1 -ar 16000 -b:a 32k`）：范围是窗口首条字幕头到末条字幕尾，两端各加 5s padding；含全局第一条字幕的窗口头部 padding 为 60s（clamp 到 0），含全局最后一条字幕的窗口尾部 padding 为 60s（clamp 到音频末尾）。剪辑写到 `tmp/llm-audio-clips/<stem>/<chunk_id>.aac`（每次覆盖），每个执行窗口（含 `-a`/`-b` 半窗）单独上传一次并在查询轮/纠错轮/同窗重试间复用。窗口规划完成后即在后台线程预切首窗；进入窗口 *i* 时预切 *i+1*，切完即上传 Gemini，主循环仅在取当前窗 file ref 时等待就绪。`*-translated.srt` 时间轴由本地按源序号从原始 `stable.json` 回填，因此保留原始高精度。最终中文字幕 SRT 使用 `translation` 列再做后处理；`-corrected.srt` 使用 `corrected_text` 列。

默认命令只写出计划和 prompt（两轮调查 prompt + 每窗口纠错 prompt）：

```powershell
python -m llm.correction_translation out/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts
```

执行 API 时默认串联：背景调查 Round 1 → Round 2 → 纠错窗口循环（`--execute` 必须提供 `--audio`；旧的 `--audio-file-id` / `--upload-audio` 已随按窗剪辑上传移除）：

```powershell
python -m llm.correction_translation out/input-stable.json --audio data/input.wav -o out/input.srt --execute
```

相关 CLI 参数：

- `--route {text,mm}` / `--level {low,med,high}`：翻译路线与档位（本模块默认 mm/med；`pipeline.py` 默认 mm/high，音频输入自动降级 med）。
- `--output-scale K`：输出估算系数 k（默认 1.0）；调大切出更小窗口。
- `--fast {auto,on,off}` / `--fast-search-rounds N`：快速模式开关与其搜索轮数（默认 auto / 2）。
- `--video PATH`：源视频文件，仅 mm-high；`--execute` 时必填（见「mm-high 视频」）。
- `--extra-info` / `--extra-info-file`：用户提供的额外信息（来源 URL、内容说明、额外要求），注入两轮调查。
- `--context-file <research-context.json>`：复用已有调查结果，跳过两轮调查。
- `--research-only`：只跑两轮调查并写出 `research-context.json`，不进入纠错。
- `--extra-style`：注入纠错 system prompt 的特殊翻译风格段落（默认为空）。
- `--no-web-search`：关闭本地搜索代理；调查 Round 2 拿不到搜索结果，纠错窗口查询轮整体跳过（多轮搜索 loop 也随之关闭）。
- `--research-search-rounds`：背景调查的总搜索轮数（含第 0 轮，默认 3）。>1 时启用多轮搜索 loop（Research Contract / Evidence Pack，见下文）；设为 1 恢复旧的单轮搜索。窗口查询轮始终单轮，不受此参数影响。

## 本地检索代理

gemini-3.x 免费层级不开放 `google_search` grounding（实测立即 429），所以纠错/调查模型不直接启用联网工具；检索由 `llm/web_search.py` 在本地执行，含 **search**（网页搜索）与 **extract**（单 URL 深度整页提取）两类。Gemma4 fallback 使用 `gemma-4-31b-it` + 通用 Search grounding 免费配额（约 1500 RPD），区别于 Gemini 3 免费层级专用 grounding 0 RPD：

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
- `search_many`/`extract_many` 会去重、按上限截断（背景研究第 0 轮 `min(20, 8 + sqrt(raw字幕片段数)//10)` 条、loop 追加轮为其一半（向上取整）/ 每纠错窗口 8 条；extract 仅 loop 追加轮可发起，见下），并做 1.5s 限速（Exa 的 10 qps 限制远宽于此）。
- Gemma4 search 单次 pass 最多接收 8 条 query；若 pending query 更多，会按 8 条一批自动分批调用。Gemma4 grounded REST 调用单独使用 1200s timeout（Exa/Tavily/DDG 保持通用 timeout）；多次真实测试中，触达最大输出时耗时可到约 900s。
- Gemma4 search 单 query 注入量级（2026-07-10 真实 8-query smoke）：按 `render_search_results([result])` 计，观测范围约 283–1079 tokens，中位数约 1062 tokens。Google grounding 的 redirect URL 较长，title+URL token 往往占主要部分；预算估算可先按 300–1200 tokens/query 记，source-heavy 查询留到 1500 tokens/query 更稳。
- 结果按 query 分组渲染（provider、标题、URL、摘要，含长度截断）后注入 `<search_results>` 块；query section 用 `--- query: ... ---`，深度提取 URL section 用 `--- 深度提取 url: ... ---`，避免 Markdown `###` 标题与正文错位。图片及其 URL 在清洗阶段丢弃。背景调查启用多轮 loop 时，Round 2 注入的是整理后的 Evidence Pack 而非原始结果。

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

## 背景调查（两轮）

两轮都是独立 API 调用（非多轮对话），模型角色 `general_capable`，不启用任何工具（走 Gemini REST 直连，free key 失败可回退 paid key）。共同输入是带窗口标记的紧凑字幕文本（`源序号|文本`，窗口边界处插入 `--- window N ---`）和用户额外信息。窗口边界来自本地 chunk planner，先于调查完成规划。

- Round 1 额外注入 streamer/common 两份知识库 index 全文，以及 **harness 本地预注入的知识库条目**：对用户备注（`extra_info`/note）做 key+alias 的 casefold 子串匹配（`knowledge.base.match_index_keywords`，别名去重到条目、按出现频次排序，最多 8 条，1 字符词跳过），命中条目全文按预算渲染注入 `<preinjected_entries>`。职责分步：先做中轻量分析并输出 `<analysis_notes>`（≤1500 token，写给 Round 2 的要点，写于搜索前，未证实判断须标"待定"；缺失按空处理不重试）；再用 `<requested_entries>` 请求尚未注入的新词条（按重要性排序）、用 `<keep_entries>` 保留后续仍需的预注入词条；最后提出搜索 query（`<search_queries>`，每行一条，上限为动态第 0 轮上限）。request/keep 分别保留原始输出供审计，canonicalize 后各自最多 8 条、合计最多 12 条，harness keep-first 合并，超限从 request 尾部丢弃；keep 只能命中本轮实际渲染（included/truncated、非 dropped）的预注入条目。启用多轮搜索时还必须输出 `<research_contract>` 块（缺失触发解析重试）。
- 两轮之间 harness 用本地搜索代理执行 query（单轮）或运行多轮搜索 loop（默认），结果写入 `research_search_results` artifact（多轮时逐轮细节在 `search_loop_round` artifact）。
- Round 2 额外注入 Round 1 索取条目的文件全文（按注入预算渲染：单条 4k token、整块 `query上限×2k+4k` token）、Round 1 的 `<round1_notes>`（即 analysis_notes，注明写于搜索前）和搜索产物（单轮为原始 `<search_results>`；多轮为 Evidence Pack，消费规则片段随之替换）。职责：全局/分窗口理解、实体术语抽取（含推荐译名）、ASR 误听风险识别；不能再发起搜索，未覆盖疑点写入 `uncertainties`。输出包裹在 `<context_pack>` 标签块中的 JSON：`general_context`（注入所有窗口）和 `window_contexts`（按窗口 id 对齐注入）。
- 文本路线没有背景调查：预注入条目改为直接进入每个纠错窗口的 `<entry_details>`（并计入 resume 指纹）。

### 多轮搜索 loop（默认开启，最多 3 轮）

`llm/search_loop.py` 实现可插拔的多轮搜索：只替换发起侧 prompt 片段（`fragment_search_queries_output_v1` → `fragment_search_contract_output_v1`）和消费侧片段（`fragment_search_results_usage_v1` → `fragment_evidence_pack_usage_v1`），其余调用面不动。当前只有背景调查接入；窗口查询轮保持单轮（少一轮调用，兼作效果对比组）。

三个数据结构：

- **Research Contract**（Round 1 输出的 `<research_contract>` JSON）：`goal`（这次调查要什么）、`facts[]`（`{id, fact, priority 1-5, done_when, hints}`，≤12 条）、`out_of_scope[]`。priority 由 harness 机械维护：某 fact 被追加轮 query（`F1|query` 前缀标记）覆盖过一轮则 -1（最低 0）。**priority 只是提示，降到 0 仍可查**，是否继续由 loop 模型自行判断；防死磕的硬边界只有最大轮数。
- **Research Progress**（harness 维护的累积台账）：loop 模型每轮输出 `<progress_update>` 增量（≤2000 token，按 fact id 记 confirmed/partial/not_found/dead_end + 结论 + 来源，另有"新发现/死胡同"），harness 以 `## 搜索轮 N` 头拼接。只在 loop 内部流转。
- **Evidence Pack**（`<evidence_pack>` markdown，最终注入 Round 2）：`## 结论` / `## 关键证据摘录` / `## 未解决` 三节；harness 统一在头部注入"由搜索代理整理、仍需交叉验证"声明。（实验模板 `search_loop_v2.md` 另有第四节 `## ASR 误听候选`——疑似误听→候选对应表，仅为线索不构成纠错指令。v2 只能经 `tools/session_replay --loop-version v2` 触发：生产调用方 `research.py` / `stages/fast_session.py` 均不传 `loop_version`，走默认 v1 三节。）

流程：Round 1 产出 contract + 第 0 轮 query → 本地执行 → 每轮搜索后调一次 `lightweight`（纯文本；per-call thinking 覆盖为 medium：`thinking_level=medium`，输出上限 32,768=SESSION_OUTPUT_MAX_TOKENS）做筛选、去重、抽证据，输出 progress 增量并决定"继续检索"或"生成 Evidence Pack"。judge 输入在 `<knowledge_entries>` 前明确列出上一调用的 requested/kept entry 原名；在 `<search_results>` 紧前注入 `<previous_search_request>`，其中是发起时的 contract 快照和经过 cap/去重后实际执行的 query/extract，另保留 `<current_research_contract>` 供下一步判断。继续检索时可同时给出 `<search_queries>`（`fact_id|query` 前缀，可带 ` >> 引导语`）与可选的 `<extract_urls>`（对已出现在结果中的 URL 发起深度整页提取，可带 ` >> 引导语`）——**extract 仅此评审代理可发起**（主调查/纠错查询轮不发起）；两者跨轮归一化去重，合计计入追加轮上限（=第 0 轮的一半），其中每条 query 计 1、每 2 条 extract URL 计 1（预算按半单位计：query 2 半单位、URL 1 半单位）。深度提取结果与搜索结果**合并为一个预算块**渲染进下一轮（单 section 4k token、整块 `本轮cap×2k+4k` token，块尾附"注入预算说明"列出被截断/丢弃的条目）。priority 递减在渲染**之后**执行：只有结果**完整进入**渲染块的 query 才会使其 facts 的 priority -1，被截断/丢弃的 query 视同未执行（模型可在后续轮重发，不算重复）。非末轮的 judge 还可输出可选 `<requested_entries>` 块（每行一个知识库 key/别名，独立上限 = 追加轮 query 上限、不占检索用量）：harness 解析后把词条全文按预算渲染注入下一轮 `<knowledge_entries>`，跨轮按主 key 去重；两份知识库 index 注入每个非末轮（末轮置空并禁止请求）。轮次提示来自模板 fragment（`fragment_search_loop_{continue,final}_notice_v1`，Python 只按是否末轮选并填剩余轮数）：非末轮提示"仍有 priority ≥ 2 且 partial/not_found 的 fact 时默认继续检索、不要过早收尾"，到达最后一轮时强制要求输出 pack、不得再输出 query/extract/词条请求。若 judge 在**非末轮**就产出 pack 而累积台账里仍有 priority ≥ 2 未决 fact（按每 fact 最新状态判定），harness 落一条 `premature_evidence_pack` 告警 artifact（仅告警、不阻断收尾）。Progress/Evidence Pack prompt 各带 one-shot，并要求重要 confirmed/partial fact 尽可能保留多条支持、补充或冲突证据；只有原结果中的逐字内容可标为引文，否则必须标为摘要。loop 模型调用异常、解析重试耗尽（默认 max_parse_retries=5）、或末轮拒不输出 pack 时降级回退：用 Progress 台账 + 全部原始搜索结果拼一个降级 pack，不让任务失败。contract（含递减后的 priority）、逐轮元数据与 evidence pack 摘要持久化到 `research-context.json` 的 `search_loop` 字段。

v1 厚度要求（prompt 层）：`<progress_update>` 每条 fact 可含 2-3 句（核心判断 + 相关上下文：读音变体、关联实体、出现场景、交叉印证或冲突），软上限约 2000 token。`<evidence_pack>` 的 `## 结论` **必须逐条覆盖 Contract 每一个 fact**（含 priority 0），每条至少 2 句（核心事实 + 相关上下文），至多 4 句；`## 关键证据摘录` 定位为下游交叉验证的唯一依据，每个独立来源单独成条、不合并，每个 confirmed/partial fact 尽量 2 条以上。harness 在 pack 提取后做逐 fact 覆盖软校验：`## 结论` 中缺失的 contract fact id 记一条 `evidence_pack_missing_facts` 告警 artifact（仅告警、不阻断）。one-shot 仅示范格式与信息密度下限，不限制条数与篇幅。

**v2 变体**（`search_loop_v2.md` + `search_loop_user_v2.md`，`build_search_loop_v2_messages`；生产尚未接入，仅 session_replay `--loop-version v2` 可用）：取消 v1 的"继续检索 OR 收尾"二选一，改为每轮**必须**输出完整 `<evidence_pack>`（在上一轮 pack 基础上更新），可选输出 `<search_queries>`/`<extract_urls>`；harness 以"无 query"作为终止信号。取消 `<progress_update>`，跨轮连续性由 `<previous_evidence_pack>`（上一轮的完整 pack）承载。结构更简、省约 400 output tokens/轮、pack 始终可用（无降级路径）；厚度与 v1 持平（flash-lite thinking=0 下证据摘录仍为 1 条/fact 的天花板）。

预算与失败处理：

- 每轮输入必须满足 `prompt_input_limit = 194000`；超限直接报错，提示先切分音频。不做 map/reduce。
- 模型输出标签块/JSON 解析失败时同请求最多再试 5 次（`max_parse_retries=5`，共 6 次调用），仍失败则任务失败（`<context_pack>` 缺失时会尝试直接解析裸 JSON 兜底）。
- 每轮响应、usage token 计数和解析错误会写入 task artifact（如指定 `--task-artifact-dir`）。

## 纠错输入的确定性预合并（**已删除**，2026-07-29）

原 `src/premerge.py` / stabilize profile 3 已随 `segment_split` 迁到全局 DP 一并删除：
分句器现在自己决定每个 ASR 段接缝的去留，词中切断的碎片不再产生，预合并因此无对象可并
（9 clip 测试床实测 0 次合并，对照旧逐段 split 输出 1 次）。删除范围与历史结论见
`docs/asr-stabilize.md`「Profile 3」节。

对 LLM 侧的影响：`chunking.load_segments_from_stable_json` 仍是纯加载，拿到的 stable JSON
已是最终段序列，源序号按位置顺次编号——这一点没变，变的只是"最终"由分句器而非预合并决定。

## 窗口拆分

Harness 采用"先估算窗口数、再均匀放置分割点"的规划方式：

1. 对全量 CSV 做一次 countTokens，按每行字符占比折算每段文本 token；每段媒体 token 按 `媒体速率 × 到下一段开始的时间跨度` 折算（音频 32 tok/s，mm-high 另加视频 17.75 tok/s；text 路线与 mm-low 速率为 0），得到每段的规划质量（mass）与前缀和。
2. 由输出约束（每窗字幕 token ≤ `窗口输出预算 / (k × c)`，其中窗口输出预算 = `0.9 × 65536 − 5000 = 53982`，k/c 为 preset 的 `--output-scale`/输出系数，mm-med 默认上限约 10,796）与输入约束（字幕+媒体+上下文 ≤ `194000`，含重叠与 padding 的固定加成）估算窗口数 `k`。规划时固定预留 `72000` tokens 的上下文额度（`WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS`），覆盖纠错调用中窗口 CSV/媒体之外的全部内容：静态 system prompt（实测 ~4k）+ user 脚手架 + 调查 context pack + 累积建议台账（≤8k）+ 查询轮 notes + 搜索结果块（≤20k）+ 知识库词条块（≤28k），最坏合计约 69k；窗口实际由输出公式限死，加大 reserve 几乎不改变窗口数。
3. 在均匀 mass 目标点附近（半径约 `0.4·n/k`，由近及远）snap 到合适边界；每个规划窗口再用真实 countTokens 预算校验，任一窗口超限则 `k+1` 全局重排（保持均匀），上限 `k0+16` 后报错；单段放不下直接报错。发生过重排（输入超预算导致窗口缩小）时写入 `window_plan_report` artifact（`estimated_windows`/`planned_windows`/`replan_attempts`/最后一次超限错误，分 research/correction 两个 phase），task report 渲染为独立 "Window Planning" 小节。
4. **质量护栏 `max_window_subtitle_tokens`**（`ModelLimits` 默认 10,000；config.toml `[chunking]` 可覆盖，`0` 关闭，非法/负值硬报错）：单窗 `<asr_result>` CSV（正文 + 重叠行）的 token 上限，独立于输出系数——窗口太长时翻译质量会掉，哪怕输出装得下。两处生效：第 2 步的窗口数估算取它与输出约束的较小者，第 3 步的真实 countTokens 校验后再硬查一次，超限走上面同一条 `k+1` 重排路径。
5. research 侧与 correction 侧的规划参数（reserve、profile、护栏值）必须一致，否则窗口 id 分歧、`window_contexts` 错位；`research-context.json` 因此带 `planning` 元数据（prompt_version/reserve/profile_id/max_window_subtitle_tokens），复用时不匹配会警告并重跑调查。护栏值记录的是**解析后的生效值**——"未配置"（取默认 10,000）与"显式 0"（关闭）是两种不同的切分，都记成 0 会让前者复用后者的 context。

每个窗口包含：

- 本窗口的输入源序号范围与重叠源序号（物理上重复包含上一窗口尾部）。
- 重叠条数 = `边界前 30s 内开始的字幕条数`（v13 起纯内容驱动、无下限），并 clamp 到上一窗口长度以内。边界落在 >30s 空档处时重叠为 0——这是正确行为：拼接不该跨大空白，连续性由只读前文块负责。
- 只读前文 `preceding_segments`：窗口开始前最多 10 条 raw ASR 字幕（`PRECEDING_CONTEXT_MAX_SEGMENTS`，固定条数、**不设 gap-stop**——大空档后恰是冷启动风险最高处，负时间戳让模型自行判断新旧）。第一个窗口为空；-a 半窗继承父窗口的、-b 半窗回看父窗口自身尾部。纯输入、不参与翻译、不影响窗口位置与 clip 范围。
- 音频剪辑区间 `clip_start`/`clip_end`（见上文 padding 规则）。
- 预算估算：字幕文本 token、剪辑音频 token（含 padding）、预计输出 token、总量和计数来源。

边界选择规则：

- 优先在句号、问号、感叹号、省略号等自然句末截断，其次是较长静音（`even_sentence_or_gap_boundary`）。
- snap 半径内没有合适边界时在均匀目标点强切，标记 `forced_even_boundary`。

预算规则：

- 免费层级输入 prompt 安全上限：`194000` tokens。
- 模型上下文规划上限：`256000` tokens。
- API 输出上限固定为 `65536`。
- 文本 token 计数按 **本地 tokenizer 二进制 → `countTokens` API → 启发式** 三级 fallback（`default_token_counter()`，逐 sha 缓存）：
  - 首选本地 `gemini-token-counter`（Go/`google.golang.org/genai/tokenizer`，源码在 `src/tools/gemini-token-counter/`，预编译产物 `bin/windows-amd64/tokcount.exe`，不列入 pyproject 依赖）。Python 进程内所有 counter 实例按 binary/model 共享一个 lazy 启动的 stdio server；默认空闲 300 秒自动退出，下次精确计数透明重启，避免逐次初始化 tokenizer。离线、免配额；它用 `gemini-2.5-flash` 词表，实测与 3.1-flash-lite 的 `countTokens` 相差**恒定 +1 token**（API 的 `contents` 外壳），Harness 已加回该 offset 使二者逐字一致。
  - 本地 binary 可执行时，截断/注入预算跳过启发式预检，直接使用常驻 server 的精确结果。本地 binary 不可用时才启用 heuristic fast path：明显低于上限则直接返回估算，接近或超过上限时进入 `countTokens` API 精确计数；API 再失败才回落启发式 counter（`HeuristicTokenCounter`，按字符类别加权求和：数字/拉丁/CJK/谚文/全角标点/其他文字/空格/ASCII 符号/其他，权重经实测拟合为**上界**——对每个测试类别 heuristic ≥ real，对实际喂给模型的字幕 CSV 最紧约 +1~8%）。旧版启发式因对 CJK 混合文本低估 25-40% 被弃用。
  - `countTokens` 端点**完全免费**：不消耗任何生成配额、不计费、无实际速率约束，`.env` key 只用于鉴权。因此即便回落到 API 也不烧 quota。
  - 本地二进制在位时**默认 dry-run 无需联网/无需 key**；缺二进制才回落到 countTokens 端点。
- 每轮候选规划需要 `k` 次 token 计数校验，通常一轮即收敛。
- 基于 token 上限的文本截断走 `llm/token_truncate.py::truncate_to_token_window`（插值+二分搜索最接近上限的安全切片，按切片长度缓存计数，只需个位数次 counter 调用；`keep="head"` 保留前缀/截尾部（默认），`keep="tail"` 保留后缀/截前缀；可选回退到自然句末边界）。两个默认开启的快速开关：本地 binary 不可用时，`lazy` 先用启发式 upper-bound 预检，`估算 × 1.02`（`lazy_safety_factor`，额外保险）≤ 上限则原样返回、零 API 计数；本地 binary 可用时直接精确计数；`quick` 把截断搜索的命中窗口放宽到 0.95/50（更少计数次数）。通用 `cap_tokens`、注入预算和累计 advice ledger 使用同一分流；显式传 `gold_ratio`/`abs_slack` 时 `quick` 不覆盖。
- 音频 token 本地按 Gemini 官方口径 `32 tok/s` 乘以**剪辑时长（含 padding）**估算；mm-high 另加视频 `71 tok/frame × 0.25 fps = 17.75 tok/s`。由于每次调用只附本窗剪辑，估算口径与 provider 实际计费一致（旧的"整文件计费"问题随按窗剪辑上传消除）。
- 纠错输出估算：`k × c × csv_asr_result_tokens`（k = `--output-scale`，c 为 preset 输出系数，见「翻译路线与档位」；替代旧的 `csv × 5 + 10000` 启发式）。
- 窗口规划要求该估算 ≤ 窗口输出预算 `0.9 × 65536 − 5000 = 53982`（快速模式收紧为 `0.8 × 65536 − 10000 = 42428`）；真实 API 请求仍使用 `65536`。

## 纠错窗口两步流程（仅 mm 路线）

mm 路线每个纠错窗口拆成两次 API 调用（text 路线没有查询轮，只有单次纠错调用；快速模式的查询轮由快速第 1 轮的种子产物取代）：

1. 查询轮（纠错 r1）：`lightweight_multimodal`（与 search-loop 的 `lightweight` 共用 3.5-flash-lite 优先链；thinkingLevel medium，输出上限 32768=SESSION_OUTPUT_MAX_TOKENS；mm-low 无音频附件但仍走本角色）。输入与纠错轮基本一致（音频、当前窗口 CSV、通用/窗口背景、累积建议台账），**另注入两份知识库 index 与已透传词条全文（v17，`<carried_entries>`，勿重复请求）**。职责分步：先以 `<reasoning>` 块开头（v17 全局必须），再做中轻量分析并输出 `<window_notes>` 块（≤800 token，写给纠错轮；须注明写于搜索前、未证实候选标"待定"）；可选输出 `<requested_entries>` 块（每行一个 index 中的 key/别名，新请求上限 8 条、与透传合计 ≤12 且透传优先；harness 解析为 canonical key，与透传集合并后统一按预算渲染注入纠错轮）；再输出 `<search_queries>` 块（上限 8 条，可为空块）。该轮 best-effort：调用异常、格式错误或输出为空都按"无 query / 无 notes / 无词条"处理并留 artifact，不阻塞纠错。
2. 纠错轮：按 preset 选角色（非 text-high 一律 `audio_multimodal` 的 3.6 优先链；text-high `internet_capable`），user prompt 注入查询轮换来的 `<search_results>`、`<entry_details>`（查询轮请求的词条全文；fast/text 路线的全局注入优先）和 `<query_round_notes>`（查询轮的 window_notes，标注"写于搜索前、仅供参考、需交叉验证"）；模型不启用工具、不能再发起搜索。

查询轮产物（`QueryRoundProduct`：搜索结果 + window_notes + entry_details）按 base 窗口 id 缓存：同窗口的 validation 重试和 `-a`/`-b` 拆分半窗复用第一次的结果，不重复调用查询轮或搜索代理。

### mm-high 视频（--video）

- 纠错轮媒体从 `.aac` 换成**低清视频+音轨的 `.mp4` 剪辑**（同剪辑区间与 padding 规则；`asr_playground.media.ffmpeg.extract_video_clip`：decode 先 `-hwaccel auto` 失败退 CPU，编码 libx264 + AAC）。后台预切/上传流水线跟随纠错媒体（mp4）。
- 查询轮**仍是纯音频**：`.aac` 在查询轮实际运行时按需现切（每 base 窗口最多一次），不给 lite 模型喂视频 token。
- API 侧：mp4 以 `detail=low` + `video_metadata.fps=0.25` 经 REST 直传（→ Gemini 每 part `mediaResolution: {level: MEDIA_RESOLUTION_LOW}` 与 `videoMetadata.fps`），计费口径与规划一致：`32 tok/s（音轨）+ 71 tok/frame × 0.25 fps（画面）`。
- 快速模式下第 1 轮直接上传 mp4（融合轮与纠错轮同媒体），纠错窗口复用该上传。

### LLM session 级 resume（默认开启）

传入 task artifact 目录且 `resume=True` 时，生产 harness 会把以下经当前 parser/contract 验证成功的原始响应追加到 `<artifact_dir>/session-checkpoints.jsonl`：research R1、research R2、search loop 的每轮 judge、fast round 1，以及普通纠错窗口的 query 轮。ledger 为 append-only JSONL；每条 committed 记录包含 `schema_version`、稳定 session/key、`input_hash`、`content_hash`、原始 `content` 和模型 metadata。截断行、坏 JSON、未知 schema/status 或 content hash 不符的记录加载时直接忽略。

`input_hash` 覆盖精确组装后的 messages、`PROMPT_VERSION`、角色/输出上限/thinking 等调用配置，以及 messages 外的任务身份（fast 的窗口/profile/备注/知识/web/media planning metadata；query 的 correction task fingerprint）。重启仍从确定性本地代码重建状态；搜索、网页提取、媒体剪辑和上传允许重做。到同一 LLM 边界时只有 hash 精确命中才取出旧响应，并再次走**当前** parser/contract；复验失败即 live 重打，不迁移旧的 parsed Python 对象。query/search judge 命中只省模型调用，关联搜索仍会重新执行。

因此细粒度可恢复边界是：research R1 后、每个 search judge 后、research R2 后；fast round 1 后及其每个 search judge 后；普通逐窗 query 后。correction R2 使用下述 `correction-windows.jsonl` 整窗提交缓存；知识更新另有 `knowledge-update-chunks.jsonl`。`--no-resume`（`resume=False`）使 session 与 correction-window 两种 ledger 都不读也不写，但不删除旧文件，也不关闭 pipeline 对完整 `*-research-context.json`/SRT 文件的 stage 级存在性复用。

### 纠错窗口中途 resume（默认开启）

纠错窗口按顺序处理，天然形成"已完成前缀"。启用 resume 且有 task artifact 目录时，每个成功且未截断的窗口把原始模型响应追加到 `<artifact_dir>/correction-windows.jsonl`（`{chunk_id, source_ids, clip_start, input_hash, task_fingerprint, content}`）。崩溃/退出后重跑 `execute_correction_windows` 时，在每个窗口循环顶部先查缓存：命中（`task_fingerprint` 与本次一致、`input_hash` 与当前窗口输入一致、且缓存内容仍能通过校验）则**回放**——直接走既有 `validate → merge → advice` 路径重建下游状态（v13 起前文块是规划期输入，无需从输出重建），跳过音频剪辑上传、查询轮和纠错调用，`i += 1` 继续；未命中则照常 live 调用并在成功后写缓存。因回放严格按记录顺序、逐字复用旧输出，后续 live 窗口看到的 advice 台账与首次运行完全一致，结果确定性等价。

- `task_fingerprint` = sha256(PROMPT_VERSION + extra_style + common_mistakes + context_pack + test_profile + task_update_feedback + profile_id + output_scale + 源媒体文件身份 + 全局 entry_details + 快速种子哈希)：任一全局参数（含 route/level/k、源媒体路径/大小/修改时间、快速/文本路线注入内容）变化即整体失效重算。`input_hash` = sha256(该窗口实际渲染出的 `preceding_context` + `asr_result` + 本窗最终注入的词条 keys/正文)：捕获局部序号布局、时间/文本、只读前文、透传词条及查询轮新索取词条；词条正文变化会使对应窗口失效。缓存记录另保存稳定 `source_ids`，回放时仍先按局部序号重新校验并映射到当前窗口的稳定序号。
- 只缓存"成功且非 output_limited"的窗口（与既有提交门槛一致），所以 `-a`/`-b` 半窗各自按其 chunk id 缓存复用；已知小限制：某窗口在一次迭代内因截断而拆分时，拆出的**第一个**半窗会在 resume 时 live 重算（其余半窗与整窗都复用），拆分罕见、至多多一次调用。
- research 的 `*-research-context.json` 另以 planning metadata 严格复用：含源 stable JSON 内容 hash、profile/output scale、备注 hash、可见知识输入 hash、web/search rounds、反馈采集档位和音频时长；fast context 还含窗口内容及媒体文件身份。任一不一致即重跑。

## Prompt 信息

纠错窗口 prompt 会给模型这些信息：

- `<asr_result>`：本窗口需要处理的直接类 CSV 文本块（时间以剪辑 0 秒为基准），header 为 `local_id|start|duration|gap|text`。纠错、query 与 fast 的每个执行窗口都把目标行重编号为 `1..N`。
- `<preceding_context>`（v13）：窗口前最多 10 条 **raw ASR** 只读前文，与 `<asr_result>` 并列、使用同一时间基准；按时间顺序编号为 `1-M..0`，最近前文恒为 0。0/负数不属于输出范围，误引用会命中未知序号校验整窗重试。目标语侧连续性由 advice 台账与 context_pack 承载；advice 禁止携带无窗口命名空间的局部序号。查询轮暂不注入前文块。
- 通用背景（`general_context` JSON）与本窗口专属背景（`window_contexts` 中对应条目），来自背景调查，放在直接 ASR 块之前以提高 input cache 复用效率。旧 payload 中的 `audio_file`、`chunk_id`、`segment_range`、`boundary_reason`、`overlap_source_ids` 和 token budget 均不再发给模型。
- `<previous_advice>`：此前**所有**成功窗口 `<next_advice>` 的累积台账（harness 维护，按 `[window N]` 标注逐条拼接，空条目跳过），查询轮和纠错轮都能看到；注入时整体上限 8000 token，超出从最旧窗口起截断（keep-tail）。
- `<entry_details>`：本地知识库词条全文——查询轮请求的、或（fast/text 路线）全局预注入的；可为空。
- `<query_round_notes>`：查询轮的轻量分析要点（写于搜索前，仅供参考，可为空）。
- `<search_results>`：查询轮 + 本地搜索代理换来的本窗口搜索结果（可为空；块尾可能带"注入预算说明"）。
- **任务 recap**：自 v9 起，每种 session 的 user prompt 末尾（payload 之后）都有一段 2-3 行的静态"最后提醒"，重申任务目标与输出格式关键约束（Gemini 长上下文最佳实践：指令重申放在大段 context 之后）。六个 user 模板均有：纠错、查询轮、调查 R1/R2、loop judge、快速 R1。

模型随后输出 translated 终稿；basicA 还会先逐源完成 singles。tag parser 只认第一级同名块。输出结构按 variant 区分：capableB/C 使用下述九列 CSV；BasicA/B 使用带 header 的十列 CSV：

生产默认映射为 `CAPABLE → capableC`、`BASIC → basicB`；其余变体仅在显式选择时使用。
生产 live、纠错窗口 resume cache 与 session replay 都通过同一个 variant-aware validator
按**实际回答端点**的 tier 解析：capableC 不要求 `<singles>`，basicB 继承 capableB 合并并带 `start`（basicA 仍要求完整
`<singles>`），并同时绑定各自的 CSV 列布局。provider fallback 改变
tier 时，validator 会跟随实际收到的 prompt variant，不沿用首选端点的契约。

```text
type|position|duration|gap|corrected_text|translation|conf|char_count|note
type|position|start|duration|gap|corrected_text|translation|conf|char_count|note
# 单源本身越过硬门槛，仍须如实输出
sub|1|2.5|4.6|5.6|...|...|high|13|
```

- `type`：留空或 `sub` 表示默认行为（拼合/纠错/翻译，`position` 填源序号）。~~`insert` 插轴已于 v63 全面废弃~~（旧模板已移至 `legacy/`，gitignored 不随仓库分发）。多个源片段合并时源序号用英文逗号连接，例如 `sub|3,4|1.9|0.2|good morning|你好|high|2|ASR 错分`。
- `start`：BasicA/B 以 CSV 列携带；单源抄输入 start，合并行抄首源 start。解析只校验存在和数值类型，最终时间轴仍按映射后的稳定源序号回填；抄值准确率仅作能力观测。`duration`/`gap` 同样是引导字段而非可信时间源。
- capableC 的局部推理使用目标行正上方的 `#` 注释；普通单源在界内前不输出。validator 只计数，不进入 SRT。
- `gap`（v37）：**本条结束后到下一条开始**的间隔秒数（与输入 ASR CSV 的 gap 同义），绝不是本条到前一句的距离；判断是否与前一句合并时须读取前一行 gap。引导用列，解析后丢弃。
- `conf`（v39）：`high`（very certain）/`median`（likely correct）/`low`（better to manually check）三档自评信心；旧缓存中的 1–9 数字仍会兼容映射为三档。`char_count`：独立加权译文字数列，位于 note 左侧；本地按“拉丁/数字/标点/空格=0.5，其余可见字符=1”复算并规范化，模型值不一致时把 warning 写入窗口 artifact。统一公式由 `asr_playground.subtitles.metrics.weighted_char_count` 定义，并同时用于 pacing、annotated CSV 与通用 SRT 行长 warning；它只衡量字幕显示长度，与 token 预算及 ASR 异常检测用的 `asr_playground.text.count_word_units` 相互独立。`note`：自由注记，是最后一列；prompt 要求文本中的 `|` 写成全角 `｜`，解析器仍宽容旧输出在末列使用半角分隔符。
- 统一入口是 `csv_utils.validate_correction_window_output`：它先按 variant 校验窗口局部 CSV，再把有效 `position` 与 discard 序号映射回稳定源序号。CSV parser 对 type/conf/note 宽松；v39 行的 char_count 格式会校验。结构性错误（未知/乱序/重复源序号，含 discard 与普通行之间的冲突；意外 start 列；insert 时间不可解析；空文本；缺时长列）仍判失败触发重试；底层 parser 仍兼容旧的 3 列 `source_ids|corrected|translation`（按 `sub` 处理，免时长列）。
- **行尾 `<void>` 自弃标记（v12 起）**：模型写完一行才发现不对（时长失控、分组/取舍错误）时，可在行尾追加 `<void>` 废弃整行并另起重写。解析时带标记的行在一切结构检查**之前**剥离（内容再破也不报错），其源序号可被后续行重新使用；数量计入 `CsvValidationResult.voided_rows` 并写进 `correction_window_response` artifact（用于观测模型是否真的使用该通道）。全部行都自弃且无其他有效行时按"无有效行"判失败重试。
- 只有 `translation`（纠错 SRT 另用 `corrected_text`）进入 SRT；`type`/`duration`/`gap`/`conf`/`char_count`/`note` 和 insert 行全部留存在 `<stem>-annotated.csv`（9 列）。默认行时间轴按源序号从 `*-stable.json` 回填；insert 行自带时间轴，跨重叠窗口按开始时间就近去重（新窗口优先，~1s 容差），且不挤占默认行。知识更新阶段另会 overlay 最终 SRT 时间轴，生成含 start/end 的 10 列 `<final_csv>`。

如果某段高度疑似 ASR 幻觉（含套话式幻觉）、无意义重复或非主播有效内容：basicA 仍须在 `<singles>` 输出并标注；所有变体都须在 `<translated>` 以 `discard|<局部序号>` 显式丢弃，不能静默漏掉（否则 coverage validation 失败）。v12 起 prompt 侧取向为**拿不准时保留并在 note 标记「疑似幻觉」**（人工删一条错留的幻觉比重听补一句被误删的台词便宜）。v16 起幻觉判定/保守保留/丢弃与插轴取舍收拢为独立的「幻觉与丢弃」fragment（`$hallucination_block`，取舍子句仅音频路线注入），套话幻觉另有独立反例（输入/输出块形态）；处置措辞仍按模态参数化（`$hallucination_handling`/`$noisy_span_handling`：音频路线以重听裁决，纯文本路线默认保留、禁止凭空"还原"台词），按类别特征描述（套话语域＋上下文脱节＋无对话区间）而非具体短语黑名单。

每行首先是字幕显示单元/时间单元，不是逐词语义对齐单元。`translation` 可以为了中文语序和阅读节奏，在同一连续语义单元内相对 `corrected_text` 前后错位；多数源保持独立，勿为对齐强行并成长字幕。少数三源例外仅限同一句连续三切，禁止四源以上。

`</translated>` 之后模型必须输出一个 `<next_advice>...</next_advice>` 块（可为空），内容是本窗口**新增或修正**的简短建议（新术语定名、说话人状态、未解决指代等，上限 800 token/窗；不复述台账已有条目，失效条目写"修正 [window N] 的某条"）。Harness 把它 append 进累积台账并注入后续所有窗口的 prompt（`-a`/`-b` 半窗同样入账，前半的建议自然传给后半），同时留存为 artifact；台账注入时整体上限 8000 token，超出丢弃最旧窗口的建议（prompt 提示模型可重申仍然有效的长期建议）。模型漏输出该块时按空建议处理，不触发重试。

纠错调用不启用任何工具；对专名/术语的查证由前置查询轮 + 本地搜索代理完成，prompt 要求模型对注入结果先交叉验证再采信。查询轮的 query、搜索 provider 和结果元数据会记入 task artifact。

当且仅当 `--knowledge collect/update` 时，纠错 prompt 会额外注入任务反馈采集要求（v3 schema：`knowledge_hints` + `asr_corrections` + `uncertainties`）。模型需要按 `<next_advice>` → `<keep_entries>` → `<task_update_feedback>` 的顺序，最后输出一个短 feedback JSON 块。该反馈的 `source_ids` 使用本窗口正局部序号，harness 在 artifact 落盘前映射回稳定源序号；`confidence` 仍为 1–9，与字幕行的 high/median/low conf 相互独立。Harness 不会把该块拼进最终 SRT，只作为 `correction_window_task_feedback` artifact 留存。research 末轮同样采集并写 `research_task_feedback` artifact：普通 round 2 直接引用多窗口 transcript 的稳定源序号，fast round 1 使用单窗口局部序号并在落盘前映射。解析失败只告警不重试。该档位纳入纠错 resume 的 task fingerprint。

## 注入上限

下列上限由 Harness 在运行时强制（常量定义见 `src/llm/config.py` 与 `src/llm/web_search.py`）；全部以 **token** 计（用任务的 token counter 计数，截断走 `token_truncate.cap_tokens`）。超出 prompt 侧上限时截断或丢弃；超出输入硬上限时报错。

统一注入预算公式（搜索结果、深度提取、知识库词条共用，`injection_block_token_limit`）：**单 section（单 query 结果 / 单 URL / 单词条）≤ 4000 token；整块 ≤ 该轮单位上限 × 2000 + 4000 token**。块按优先序装填，装不下的 section 截断或整体丢弃，块尾追加"注入预算说明"列出受影响条目；`RenderedBlock.report()` 全量记入 artifact。

| 项目 | 上限 | 说明 |
| --- | --- | --- |
| 窗口规划 reserve | 72000 tokens | 规划每窗输入预算时扣除，覆盖窗口 CSV/媒体之外的全部内容（静态 prompt、context pack、建议台账、查询轮 notes、搜索块、词条块），最坏合计约 69k。research 与 correction 两侧必须同值（窗口 id 一致性）。 |
| 背景调查搜索 query | 8..16 条 | round 0 本地执行硬上限；`min(16, 8 + sqrt(原始段数)//10)`。模型多出的 query 丢弃。搜索 loop 后续轮每轮上限为 round 0 的一半。 |
| 纠错窗口搜索 query | 8 条 | 每窗查询轮 `<search_queries>` 最多执行这么多条（Exa → Gemma4 → Tavily → DuckDuckGo），再进入纠错调用。 |
| 搜索/提取结果渲染 | 单 section 4000 token；整块 `该轮query上限×2000+4000` token | 统一预算公式。软上限：单条 snippet/answer 600 token、单 URL 提取内容 1800 token（section 内部的排版控制）。loop 内搜索+提取合并为一个块；因块超限被截断/丢弃的 query **不递减** fact priority，可在后续轮重发。 |
| 知识库词条注入 | 单词条 4000 token；整块 `条数上限×2000+4000` token | 调查/Fast R1 的 request≤8、keep≤8、keep-first 合计≤12（整块≤28k）；查询轮新请求与透传同样合计≤12；loop 非末轮词条请求和本地预注入各自按该轮上限。 |
| 本地关键词预注入 | 8 条 | 用户备注与 index key/alias 的 casefold 子串匹配，按频次排序；注入调查/快速 R1（text 路线注入纠错窗口）。 |
| `--extra-info` URL 预提取 | 8 个 URL | 从 `--extra-info` 去重后的 HTTP(S) 链接；调查 round 1 前 deep extract，注入 `<note_url_extracts>`（`<search_results>` 文本），块预算 `8×2000+4000`。 |
| `analysis_notes`（调查 R1） | 1500 token | 解析后 Harness 截断；仅作 round 2 与搜索 loop 背景。 |
| `analysis_notes`（快速第 1 轮） | 2000 token | 快速模式下兼任纠错轮的主要背景，上限更宽。 |
| `evidence_pack`（搜索 loop） | 20000 token | 多轮搜索最终产物；`search_rounds > 1` 时在调查 round 2 替换原始搜索结果。 |
| `progress_update`（搜索 loop） | 2000 token | 每轮 loop judge 调用后追加的增量台账条目。 |
| `window_notes`（纠错查询轮） | 800 token | 轻量多模态查询轮可选预搜索分析；以 advisory 文本注入纠错 prompt。 |
| `next_advice` | 800 token/窗；台账整体 8000 token | 每窗增量建议；Harness 按窗口 id 累积并注入后续窗口（含 `-a`/`-b` 半窗）。注入时台账超限从最旧窗口截断（keep-tail）。 |
| Prompt 输入硬上限 | 194000 tokens | 调查两轮调用 API 前走 countTokens；超出即硬错误（无 map/reduce）。 |
| 快速 round-2 reserve | 56000 tokens | 快速 round 1 的输入门槛 = 194000 − 56000，为纠错窗的种子注入（搜索/evidence ≤20k + 词条 ≤28k + notes 2k）留余量。 |
| 默认 LLM 输出上限 | 65536 tokens | 调查轮与纠错窗口共用（Gemini 3.x 上 thinking 与可见输出竞争同一预算）。 |
| 纠错查询轮输出 | 32768 tokens（SESSION_OUTPUT_MAX_TOKENS，v17 起所有非纠错 session 共用该默认） | 搜索 query + 词条请求的多模态调用。 |
| 搜索 loop judge 输出 | 32,768 tokens（SESSION_OUTPUT_MAX_TOKENS） | 容纳 progress 增量、后续 query/词条请求或完整 evidence pack。 |

## LLM thinking effort

Gemini 3.x 通过 `thinkingConfig.thinkingLevel` 控制思考深度：`client.complete` 把角色的 `thinking_level` 直接传入 Gemini REST `generationConfig.thinkingConfig.thinkingLevel`；`thinking_level` 为空时才按 `thinking_budget` 折算（≤800→low，>800→high）。per-call 的 `thinking_budget` / `thinking_level` 覆盖目前仅 text-low 纠错窗口使用。

2026-07-12 使用 `GEMINI_FREE` 直连实测确认 `gemini/gemini-3.1-flash-lite` 支持原生 thinking（`thinkingLevel=medium` 返回 `thoughtsTokenCount`）；此前 catalog 中将其 `supports_reasoning=false` 视为能力事实是误判，免费与付费条目现均为 `true`。

按 token 数计的 `thinking_budget`（供不支持 thinkingLevel 的模型）不再单独维护，一律由 level 按 API 输出上限的比例派生（`config.thinking_budget_for_level`）：low/medium/high = 20%/40%/60% × 65,536 → 13,107 / 26,214 / 39,321。

总原则：除 text-low（低思考）外，所有调用一律 medium，除非另有说明。

| 调用点 | 角色 | thinking_level | thinking_budget（派生） | per-call 覆盖 |
| --- | --- | --- | ---: | --- |
| 调查 round 1/2 · 快速第 1 轮 | `general_capable` | medium | 26,214 | 无 |
| 搜索 loop judge | `lightweight` | medium | 26,214 | 无 |
| 纠错查询轮（纠错 r1） | `lightweight_multimodal` | medium | 26,214 | 无 |
| 纠错窗口 / fast 纠错步（纠错 r2） | `audio_multimodal` | medium | 26,214 | 无 |
| 纠错窗口（text-low） | `audio_multimodal` | medium | 26,214 | **low / 13,107** |
| 纠错窗口（text-high） | `internet_capable`（原生搜索工具） | medium | 26,214 | 无 |
| 统一知识更新 | `general_capable` | medium | 26,214 | 无 |

## 模型配置、速率限制与显式 reasoning

`src/llm/model_catalog.psv` 是 pipe-delimited 模型/provider tier 事实表，列为
`provider_tier|model|litellm_model|max_input_tokens|max_output_tokens|supports_video_audio|supports_native_search|supports_reasoning|rpm|tpm|rpd|tpd|is_free|capability`。`provider_tier` 与 `.env` entry 名一致（`GEMINI_FREE`、`GEMINI_PAID` 等）。**`tpm`/`tpd` 仅指输入 token**（不含输出/thinking）；`rpd`/`tpd` 列仅供人工参考，运行时**不预追踪**日额度。

### 生成 API 速率限制（`llm/rate_limit.py`）

- **限流桶**：`(provider_tier, litellm_model, key_id)`；限额来自 catalog 对应行 × **0.9** 安全系数。每把 key 独立计数（Gemini 免费层 RPM/TPM 是 per-project）。
- **主动追踪**（**61s** 滑动窗）：**RPM**（请求次数）与 **TPM input**（输入 token 预扣/结算）；输出 token 不影响 TPM 等待。每个实际 HTTP 尝试（含 sticky 失败重试）都记入 RPM；首次 attempt 走 `acquire`（RPM+TPM 预扣），后续 sticky 重试走 `note_request`（只记 RPM、等 RPM 空位）。`settle` 在成功后按应答 key 校正 TPM。client 级不再做限流。
- **错误分类**（`client.classify_quota_error`，按**结构化 `quotaId`** 而非 `retryDelay` 提示）：`...PerDay...` → `DAILY`、`...PerMinute...` → `PER_MINUTE`、其余 429/限流 → `OTHER_RATE`。**不用 retry 提示判日限**——Gemini 对真·日耗尽也返回 ~20–60s 的通用退避，提示区分不了日/分钟（旧 `has_short_retry_hint` 门控正因此把真日限误当瞬时，已删）。
- **日封禁需 strike 确认（per-key）**：单发的 `PerDay` 不立即封。`rate_limit.note_daily_quota_hit` 累计 strike，**连续 ≥3 次**（`DAILY_STRIKE_COUNT`；成功即 `reset_daily_strikes` 清零）才写入 `.state` 的 `llm_rate_limit.daily_exhausted`，在 **Pacific 日历日**内跳过该 key。不再要求首末跨度（旧 `DAILY_STRIKE_SPAN_SECONDS` 已删）。strike/`daily_exhausted` 以 **`(tier, model, key_id)`** 记账（照 exa `ApiKeyPool` 模式）：named key 用其名称，匿名 key 用 `sha256:<前12位hex>`——**.state 不明文存 key 原值**。一把 key 的日封不连带同 endpoint 的其他 key。免费档 `PerDay` 信号会 flicker，故不凭一次就封整天；strike 与 `daily_exhausted` 均落 `.state` 跨进程可见。`PerMinute`/普通 429、临时 5xx 退避重试或回退下一 endpoint；生成请求单次 timeout 为 15 分钟，**sticky retry budget 为 3**（观察发现哪怕 5xx 也会占用日额度，故从 7 下调并拉长退避），但连续两次 timeout 会提前抛出原始 timeout failure；参数/鉴权等不可重试 4xx 立即上抛。
- **429/可重试错误退避公式**：`sleep = min(max(4×2^attempt, parse_retry_after_seconds(exc)), 300) + 1`（基数 2026-07-29 由 0.5 改为 4，同上：少烧额度、拉长间隔）。`parse_retry_after_seconds`（`rate_limit.py`）解析主流 provider 的等待提示（Gemini `retryDelay`/`"Please retry in Xs"`、OpenAI/Anthropic `Retry-After`/`retry_after`、通用 `"wait Xs"`/`"try again in Xs"`），无提示时取 0；上限 300s 防止 provider 返回异常大值。
- **Endpoint 链**：每个 `LLMRole` 在 `config.py` 配置有序 `endpoint_chain`。分档：`audio_multimodal`（纠错窗 / fast 纠错步）优先 `FREE+3.6-flash`；`general_capable`（research r1/r2、fast r1、知识更新等）为 `FREE+3.5-flash → FREE+3.6-flash → FREE+3.5-flash-lite → …`；`lightweight_multimodal`（纠错 r1）与 `lightweight`（search-loop 查询 judge）共用 `FREE+3.5-flash-lite` 优先链。原生搜索角色只用 catalog 标 `supports_native_search=true` 的 2.5 Flash free/paid 链。
- **`chat_complete` 的 key 处理（sticky + per-key daily）**：每次调用指定单一
  `provider_tier`（或 `profile`），先按 `config.toml` 解析该 tier 的 pool；被关闭或
  无 key 的 tier 在 endpoint chain 中跳过。同 tier 下按 pool 顺序取 key，**跳过已
  daily-exhausted 的 key**，**429/限流在同一把 key 上原地重试**（不首撞即换），
  PerDay 429 同时喂给该 key 的 strike gate（gate 确认即锁该 key 并立即轮换），仅当
  该 key 的重试预算全花在限流上才轮到下一把；media 调用因上传文件项目隔离钉选定
  Gemini pool 的第一把 key。pool 不改变 LLM concurrency=1。
- **组合临时冷却（`combo_cooldowns`）**：`(tier, model, key_id)` 在一次调用内耗尽 sticky
  retry（可重试错误）后进入冷却：**0–20 分钟** skip（立即换链上下一组合，不干等）、**20–120
  分钟** probe（sticky retry=0，成功清除、失败重置起点）、**≥120 分钟**自动清除。持久化于
  `.state`，与 daily-exhausted 独立。
- **尚未做（后续）**：滑动窗（RPM/TPM）仍为进程内内存态（进程退出即重置）；跨进程持久化滑动窗为后续项。
- **`countTokens`** 与文件上传不走生成限流器；`test_profile` 禁用 limiter 以免拖慢单测。

v17 起所有 prompt 模板都要求**回复以一个 `<reasoning>...</reasoning>` 块开头**（措辞按调用的 thinking 深度分 low/medium/high 三档；纠错/查询轮按 profile 取档，research/loop/知识更新默认 medium）。运行时不再改写 message；缺块不校验、不重试，`<reasoning>` 不参与业务解析。

### 纠错 prompt 的 capability tier（随实际 endpoint 解析）

catalog 的 `capability` 列派生纠错 prompt 的能力分层：`tier_for_capability`（`config.py`，
阈值 ≥6）把 3.5/3.0-flash 判为 `capable`，flash-lite/2.5-flash/gemma 判为 `basic`。纠错调用点
把消息以**工厂函数**（`TieredMessages`）传给 `LiteLLMRoleClient.complete`，endpoint 循环在
选定即将应答的模型时按其 tier **惰性组装并按 tier 记忆化** prompt——因此跨 tier 回退
（3.5-flash 限流落到 flash-lite）时，弱模型收到的是为它组装的版本，一致性由同一 `endpoint`
同时决定「模型」与「prompt 版本」的结构保证。查不到 catalog 条目的 endpoint 默认按
`capable` 处理。实际应答的 tier 经 `LLMCallResult.capability_tier` 回传，进 token 报表行与
exchange 元数据；调用方用它重建实际发出的消息写产物。其余角色（查询轮、research、
search loop、知识更新）仍传固定消息列表，行为不变。

tier 只选择默认命名变体：capable 档为 capableC，basic 档为 basicB；两者都使用
`fragment_merge_rules_nosingles_v1` 的判断型合并规则，capableC 额外要求决策点前置局部
reasoning，basicB 额外带 start 列。保守 1:1 的 basicA 和无局部 reasoning 的 capableB
仅供显式对照。tier 无关的产出纪律（gap 方向、char_count 列纪律、列核对）抽在
`fragment_translated_common_v1`，四个变体共用。`test_profile` 的纠错 test_endpoint 为
flash-lite，默认覆盖 basicB prompt。`--prompt-dir` dry-run 常规窗口按 capable 组装，另落
`correction-0001-basic-tier.txt`（首窗 basic 变体）供 prompt 迭代检查。决策摘录见
`llm_design_notes.md`。

## 任务 Artifact 记录

显式指定 `--task-artifact-dir` 时，`task-artifacts.jsonl`、`exchanges/`、
`task-report.md` 和 pipeline 的 LLM round 汇总全部使用该目录；不会再回退扫描或写入
默认 `<stem>.llm-artifacts`。其中 `task-artifacts.jsonl` 会记录：

- `fast_decision`：快速模式判定（mode/enabled/reason 与输出、输入两侧的估算值和预算）。
- `research_round1_response` / `research_round2_response`：两轮调查的响应、usage token 计数和解析错误（如有）；快速模式下第 1 轮为 `fast_round1_response`（token 报告计入 `phase: research`）。
- `research_search_results` / `correction_search_results`：本地搜索的 query 列表、每条 query 的 provider/条数/URL/错误，以及注入文本长度（多轮 loop 时该摘要只记 loop 概要，细节见 `search_loop_round`）。
- `search_loop_round`：多轮搜索 loop 的逐轮记录——每个搜索轮的 query 与 provider 元数据、每次 loop 模型调用的响应/usage/是否产出 evidence pack/解析错误。
- `correction_query_response`：查询轮响应（模型、usage、解析错误、提取出的 query 与 window_notes）；`correction_query_call_error`：查询轮调用异常（best-effort，不中断任务）。
- `window`：窗口 id、源序号范围、起止时间、重叠源序号、只读前文序号（`preceding_source_ids`，v13）、音频剪辑区间（`clip_start`/`clip_end`）、边界原因和规划预算。
- `request`：请求输出上限、消息文本字符数和请求文本哈希（fingerprint），以及是否附加 Gemini 文件。真实 token 数以 provider usage 和 token 分布报告为准，request 侧不再做本地估算。
- `provider`：从 raw response 中提取的 `usageMetadata` / `usage`、模型版本、response id 和 prompt feedback 等 provider 元数据；如果 provider 没返回 usage，该字段为空。
- `response`：响应文本长度和哈希，以及提取出的 `next_advice`。完整模型文本仍保留在 `response_content`，供失败分析和任务中知识更新使用。
- `correction_window_retry` 会额外记录失败窗口、重试窗口和（拆分时的）后半窗口。
- `exchanges/` 子目录：每次 LLM API 交互一个可读 markdown 文件（`NNN-<call>.md`，按调用顺序编号）。文件顶部先渲染 API Calls 表，逐行列出本 logical attempt 内所有底层 `completion()` 调用（provider tier、model、api key name、同 key+model 的 call #、return code、发起/返回时间、耗时）；模型 fallback 或同 key retry 的失败尝试也会保留，最后一行通常是成功调用。随后是去重后的元数据头（attempt、finish/校验状态、`input_tokens`、`output_tokens_breakdown`、逐输入块 token 等；provider/model/key 只在 API Calls 表出现）、逐 message 的请求全文（音频附件以 `[附件文件: ...]` 标注），以及完整模型响应全文（含 `<reasoning>` 等标签，不再单独摘录）。覆盖调查两轮、查询轮、纠错轮（含重试/拆分）和统一知识更新调用（逐 chunk 一个文件）；供知识库更新、prompt 迭代等后续任务直接提取信息。
  元数据头包含**`prompt_version`**（当前 `PROMPT_VERSION`，便于对照契约）、**输入分块 token 统计**（`exchange_metadata.py` 的 `*_input_components`，用任务 counter 对各标签块计数，0 值省略）和合并 token 行（`uncached / cached / total` input，`visible / thinking / total` output）：调查轮 `transcript/extra_info/note_url_extract/streamer_index/common_index/preinjected_entry/round1_notes/knowledge_injection/search_injection`；loop judge `background/contract/executed_queries/progress/两份 index/knowledge_injection/search_injection`；查询轮与纠错轮 `csv/audio/knowledge(context pack)/entry_details/advice_ledger/pre_round_notes/preceding_context/（查询轮的两份 index）/search_injection/expected_output`；知识更新 `window_packs/kb_entries/prompt_tokens_estimate`。纠错 validation failed 时，header 会保留错误理由并从错误文本中摘出 row/source id 位置摘要。
- 检索/词条注入类 artifact 均带 `render_report`（included/truncated/dropped 与块 token 数）；查询轮响应另记 `requested_entries/injected_entries/missing_entries`；loop 词条请求单独记一条含相同字段的记录；本地预注入记 `knowledge_preinjection`（匹配词、频次、来源 phase）。
- `content_filter_ladder` / `content_filter_blacklist`：Gemini PROHIBITED_CONTENT 阶梯恢复——某次调用被拦后按「URL leave-one-out → 丢全部 URL → 丢全部检索注入」重建 prompt（不占 validation/parse 重试预算）；定位到的毒块按 `content_hash` 写入黑名单，同任务后续窗口/轮次 render 前预剔除；resume 启动时从 artifact 加载。知识更新与查询轮无检索注入可丢时只做原样重试一次。
- `window_plan_report`：仅当规划因输入超预算发生 `k+1` 重排时写入（research/correction 各自 phase），task report 渲染成 "Window Planning" 小节。
- `token_distribution_report`：每阶段一条（`phase: research` 在两轮调查后写入并同时并入 `research-context.json` 的 `token_report`；`phase: correction` 在最终 SRT 写出后写入）。`rows` 逐调用记录 `call/chunk_id/attempt/model/finish_reason` 和 token 分布（`prompt_text_tokens` / `prompt_audio_tokens` / `thinking_tokens` / `output_tokens` / `total_tokens`，来自 Gemini REST `usageMetadata` 的 `promptTokensDetails` 模态拆分与 `thoughtsTokenCount`）；`totals` 为各项求和加 `call_count`，供调参参考。
- `api_call`：本地非 LLM API 调用计数（如 `gemini_file_upload`、`web_extract`），供 `task-report.md` 汇总。
- `../<stem>-metadata.json`：与主产物同级的 pipeline metadata；只记录下载、人声分离、VAD-ASR、LLM harness 和 pipeline 总耗时、相关 worker，以及从本次 resolved artifact 目录汇总的 LLM logical-round 耗时。同一 batch logical run 的后续 pass 保留前一 pass 已执行的 stage；stage snapshot 整条替换，不会出现 `reused` 携带旧执行耗时。单轮跨度覆盖该轮全部失败 attempt、endpoint fallback 与 validation/format retry；底层 API 行仍以 `exchanges/` 为明细来源。轻量的 stabilize/SRT 导出/后处理不单列。
- `task-report.md`：任务完成后给用户阅读的**运行时摘要**——输出路径、上述核心阶段/总耗时和 worker、LLM logical-round 耗时/API attempt/retry、API 调用计数、分阶段/会话 token 用量、fallback 与疑似 IP/代理风控 warning、Gemini File 403 提示、重试/拆窗、SRT 后处理与知识库更新摘要。注入上限与 thinking effort 的静态说明见上文「注入上限」「LLM thinking effort」，不在此文件重复。

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

## 重试与拼接

每个纠错窗口的目标是一次 API 交互完整成功。

- 不做多轮续写。
- 输出上限判定有三个信号：finish reason（`MAX_TOKENS` 等）、usage 计数（输出+thinking token >= `65536 - 100`）、`<translated>` 开标签无闭标签。
- 只有当单次输出明确到达输出上限，或 `<translated>` 明显被截断时，才把当前窗口**对半拆分**重试：在窗口中间附近选择合适边界（句末 > 长静音 > 片段边界）拆成两半，两半之间保留与正常窗口同规则的动态重叠（切点前 30s 内条数，纯内容驱动，稀疏处可为 0），先处理前半，再处理后半；每个半窗有自己的音频剪辑与上传，-a 继承父窗口的只读前文、-b 回看父窗口尾部。
- 子窗口 chunk id 是 `父id-a` / `父id-b`；两个子窗口按窗口 id 继承父窗口的 window context；前半的 `<next_advice>` 传给后半。拆分可递归（`0001-a-a` 等），总重试次数仍受 `--max-retries-per-window` 约束；单片段窗口无法拆分时同窗口重试。
- CSV 格式错误、未知源序号、重复源序号或源序号乱序默认同窗口重试，重试后仍失败则报错。
- 默认每个窗口最多做 5 次纠错重试（初次调用之外），可通过 `--max-retries-per-window` 调整。
- **Content filter 阶梯**（独立于上述 validation 重试）：`finish_reason=content_filter` 且空输出时，调用侧按注入 unit（URL 提取 / query 结果 / Evidence Pack 来源）逐级丢弃重建 prompt 再调；同任务黑名单跨窗口生效。全部丢弃后仍被拦则报错（疑似源文本触发）。调查 R1/R2、fast R1、search loop、查询轮、纠错轮、知识更新均接入（知识更新/查询轮仅原样重试）。

当前本地校验包括：

- 必须有且仅有一个 `<translated>...</translated>` 块。
- 每行至少 5 列；现行契约为 9 列（`type|position|duration|gap|corrected_text|translation|conf|char_count|note`），`duration`/`gap`/`char_count` 须为数字。仍兼容无 gap 的 7 列旧行，以及旧 3 列 `source_ids|corrected|translation`。
- 本地校验暂不限制 `<translated>` 每条 `sub` 引用的源序号数量；仍校验源序号存在、唯一且顺序正确。prompt 要求通常使用单源或两源，只在少数同一句连续三切时允许三源，并禁止四源以上；不向模型暴露这一临时校验放宽。
- 源序号必须属于当前窗口，且每个源序号最多出现一次。
- 同一行和不同行的源序号都必须保持源时间顺序。
- 空 `<translated></translated>` 合法，表示当前窗口全部丢弃。

拼接规则（重叠区倾向采用最新窗口译文，且对跨界 merge 安全）：

- 所有窗口输出先解析为带源序号的字幕片段。
- 旧结果中**完全落在**当前窗口源序号集内的行被替换为新窗口输出（newest wins）。
- 旧结果中**跨越重叠边界的 merge 行**（含当前窗口之外的源序号，如 `79,80,81` 只有 81 在重叠区）整行保留，并"认领"其落在当前窗口内的序号；与认领序号冲突的新行被丢弃，其因此丢失的其他序号再从被替换的旧行回填，保证不产生覆盖空洞。
- 新窗口有意丢弃的序号保持丢弃（回填只针对"因冲突丢失"的序号）；已知边界情形：双跨界 merge 行回填时可能连带恢复个别被有意丢弃的序号。
- 加入新窗口字幕片段后按原始时间轴排序并重新编号。

## 原始 Prompt 关键细节覆盖

- 背景调查阶段负责总结内容、搜集剧情/设定/主播信息、提取术语和可能误听。
- 纠错阶段结合音频修正 Whisper ASR 的口语停顿、重复、错语言、专有名词误听，以及音似假名/汉字/英文。
- 输出只保留主播说话部分。
- 可合并短间隔且语义连贯的条目；最终时间轴由本地按源序号回填。
- 多源合并后超过 16 字默认不并；只有不可分的极少数情况可放宽，但不得超过 36 字。
- 可删去非关键填充词，但保留承载情绪、犹豫、转折或角色特征的语气词。
- 不擅自添加原文没有的人称代词。
- 适度意译，保留原文语气、情感和句式；少量语序调整可以，但避免大幅重排。
- 可使用轻小说感和可爱口语风格，如 “ちゃん” -> “酱”；“やばい”不确定时可译为“牙白”，但不能滥用。
- 特殊翻译风格通过 `--extra-style` 作为独立段落注入 system prompt；通用风格常驻纠错模板。

## 知识库更新行为

知识库结构、任务反馈采集与统一知识更新的完整行为见 [`knowledge.md`](knowledge.md)；本文其余章节仅涉及采集开关对纠错 prompt/resume fingerprint 的影响。

## 本轮实测后的关键行为

- 类 CSV 输入显著减少时间戳和编号 token；纠错 prompt 不再要求模型生成 SRT 时间轴。
- 音频 token 直接按 Gemini 官方 `32 tok/s` 乘剪辑时长计算，不调用 API 计音频 token；按窗剪辑上传后计费口径与估算一致（不再整文件计费）。
- 剪辑用 ffmpeg 写 mono-16k 裸 AAC（`.aac`）；视觉多模态 opt-in 见 `src/asr_playground/media/ffmpeg.py`（视频 decode 先 `-hwaccel auto`，失败 CPU；编码 libx264 + AAC in `.mp4`）。
- 所有生成类调用（含音频多模态）统一走 Gemini REST 直连（`client.complete` →
  `llm.chat_complete` → `_gemini_generate_content`），free tier 失败可遍历到 paid
  tier；两者均服从 `config.toml` 的 provider 开关与 pool 顺序。音频以 Gemini Files
  API 的 `fileData.fileUri` 引用注入（`upload_gemini_file` 走 REST 上传，拿到 URI 后
  直接传入 generateContent body）。`countTokens` 与文件上传两个辅助路径同样是直连
  REST（`x-goog-api-key` header，错误输出不得包含 key），使用第一个启用且有 key 的
  Gemini pool（Free 优先，随后 Paid）的第一把 key。
- 采样参数：生成调用默认显式传 `temperature=1.0`；validation/parse retry 的第 N 次 logical attempt 使用 `temperature=max(0, 1.0 - 0.01×N)` 并在末尾 user message 追加 `(seed=N)` 文本提示（Gemini REST 无原生 seed 参数），成功后的下一个独立窗口/轮次从 attempt 0 恢复。`top_p` / `top_k` 不显式设置，保留 provider 默认。
- gemini-3.x 的 thinking 统一用 `thinkingLevel`：各角色默认 `thinking_level=medium`，`thinking_budget` 由 level 按 20/40/60% × 输出上限派生（medium=26,214，见「LLM thinking effort」表）；搜索 loop judge 每次调用使用 medium/26,214，输出上限 32,768（SESSION_OUTPUT_MAX_TOKENS）。Gemini REST 响应为原始 JSON dict，`client.complete` 内直接存为 `raw_response`，token 分布从 `usageMetadata.promptTokensDetails`（模态拆分）/ `thoughtsTokenCount` / `candidatesTokenCount` 提取。
- 联网检索全部由本地检索代理（search：Exa → Gemma4 → Tavily → DuckDuckGo；extract：Exa → Gemma4 → Tavily）执行，纠错/调查模型不再启用 google_search 工具（gemini-3.x 免费层级会立即 429）。
- Gemini `503 high demand` 只做同请求退避重试；失败后记录 `correction_window_call_error` 并停止，不拆分窗口。
- 只有输出上限或 `<translated>` 明显截断才对半拆分；格式错误默认同窗口重试。
- 短片段通常只做两源以内的少量合并；少数三源例外仅限同一句连续三切，禁止四源及以上。幻觉可丢。Harness 不提供合并 hint；字数与 note 检查表见 prompt。
