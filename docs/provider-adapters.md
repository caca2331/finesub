# 自定义 provider 与文本 adapter

> **要接一个自己的 endpoint，先去** [`manual/model-routing.md` §5「接自己的模型」](manual/model-routing.md#5-接自己的模型)：
> 该写哪几列、key 放哪、怎么绑格子、第一版的三条硬限制，都在那儿。**本文不讲怎么配**——
> 它是实现契约与供应商行为差异的调研台账，写给要改 adapter、或要判断「这家为什么这么回」
> 的人。两份是同一件事的两半，分界是读者不是主题。

状态：已实现第一刀（纯文本）。实现在 `src/finesub/llm/provider_transports.py`（两个薄传输）+
`llm_runtime.chat_complete` 的按 provider 分派；用户声明入口是数据根的 `model_catalog.toml`
（事实）+ `config.toml` 的 `[llm.model_groups]` / `[llm.presets]`（编排）——样例见
`config.example.toml`，有一致性测试防漂移。
不用 litellm（D19：会丢 `thinkingLevel`、结构化 `quotaId`、cached-token usage 三处精度）。

## 供应商行为调研（2026-08-11，Sonnet 调研代理产出，已按实现取舍裁剪）

关键事实（★ = 已在实现中消化）：

| 主题 | OpenAI 官方 | DeepSeek | Anthropic |
| --- | --- | --- | --- |
| ★ 模型名 | — | **`deepseek-chat`/`deepseek-reasoner` 已于 2026-07-24 停用**。现为 `deepseek-flash`（V4.1 Flash，不带版本号）与 `deepseek-v4-pro` + thinking 开关：2026-09-12 实测传 `deepseek-v4.1-flash` 返回 `INVALID_REQUEST`，报错原文列出的受支持名只有这两个，所以旧的 `deepseek-v4-flash` 也不在其中 | — |
| ★ reasoning 模型 × 采样参数 | 传 temperature 等 → **400** | 静默忽略 | 新模型（Opus 5/Sonnet 5 等）整体移除采样参数，传了 400 |
| ★ 思考深度 | `reasoning_effort`（none…max，按模型） | `thinking:{type}` 或 `reasoning:{effort}` 两种方言 | `thinking:{type:"enabled",budget_tokens}`（另有 `output_config.effort`） |
| ★ 截断 | `finish_reason="length"` | 同左；另有特有值 `insufficient_system_resource`（资源不足中断，宜重试） | `stop_reason="max_tokens"` |
| ★ 拒答 | `content_filter` | `content_filter` | **`refusal`**（HTTP 200 + stop_reason，可能已部分输出） |
| ★ 限流/余额 | 429；余额尽也是 **429 `insufficient_quota`** | 429；余额尽是 **402** | 429 `rate_limit_error`；过载 **529 `overloaded_error`** |
| ★ cached usage | `prompt_tokens_details.cached_tokens` | 扁平 `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` | 顶层 `cache_read_input_tokens`/`cache_creation_input_tokens` |
| ★ 思考输出 | 不返回内容，只计数（`completion_tokens_details.reasoning_tokens`） | `message.reasoning_content`（vLLM 已改名 `reasoning`） | `thinking` content block；token 并入 output 不单列 |
| ★ 原生 seed | Chat Completions 有（Responses API 无；确定性弱） | 仅第三方提及，未一手确认 | 无 |
| ★ Messages API | — | — | `x-api-key` + `anthropic-version`；`system` 是顶层参数；`max_tokens` 必填；同角色连续消息服务端自动合并，首条须 user |
| 方言差异 | — | 与 OpenAI 一致（Bearer / `/chat/completions` / `max_tokens`） | — |
| vLLM/Ollama | 未知参数**可能 400 而非忽略**（vLLM 有社区实例）；Ollama 本地默认无认证但要求占位 key | | |

未确认项（用前实测）：DeepSeek 384K max output 的准确性、DeepSeek 错误体字段名、
`reasoning_tokens` 计费口径、OpenAI `xhigh`/`max` 按型号可用性。

## 实现契约（对应 §10.1 七项）

- **归一化目标**：两个传输都输出 OpenAI-ish 形状（`choices[0].message.content` +
  `finish_reason` + 嵌套 `usage` details）——harness 下游（`extract_token_distribution` /
  `_response_finish_reason` / `is_prompt_blocked`）本就双兼容 Gemini 与该形状。
  Anthropic `end_turn→stop`、`max_tokens→length`，`refusal` 原样透传并被
  `is_prompt_blocked` 识别；DeepSeek 扁平 cache/reasoning 字段折进嵌套 details；
  `reasoning_content`/`reasoning`（原始思维链）一律剥除，绝不当可见输出。
- **D18**：非 Gemini 一律**不传采样参数**；重掷扰动是 prompt 末尾 `(seed=N)` 文本
  （provider 无关，`chat_complete` 在分派前统一追加；只在原样重问时追加，见[缓存](#缓存)）。
- **thinking**：抽象档位来自预设级思考旋钮（`[presets.<id>.thinking]`，难度回落、缺省
  medium）；模型在 catalog 的 `thinking` 列声明自己的映射。**缺省是恒等**
  （`true`，这一列留空时也是它）——三家的档位词本来就同名：
  Gemini `thinkingLevel`、OpenAI 系 `reasoning_effort`、Anthropic `output_config.effort`
  都取 low/medium/high（前两者另有 xhigh/minimal 等扩展值）。端点用别的词才写
  `high映射,med映射,low映射` 三个取值覆盖；`false` = 不发思考参数（对未知字段会 400 的
  自建端点的声明式出口）。client 逐候选换算，产物记录映射后实际发出的值。
- **流式**：两个传输都发 `stream: true`（OpenAI 兼容另带 `stream_options.include_usage`，否则流里
  没有用量），由 `streaming.py` 的拼装器拼回上面的形状。终止标记：OpenAI 兼容是 `[DONE]`，或已见
  `finish_reason` 且已拿到 usage；Anthropic 是 `message_stop`；没等到就当截断。用量取全流最后
  一个非空 `usage`（OpenAI 放在 choices 为空的块、OpenRouter 放在结束块、Gemini 兼容端点每块
  累计）；Anthropic 是 `message_start` 的输入用量加 `message_delta` 的累计输出。只有正文增量算
  可见输出（`first_output_sec`、`partial_chars` 与失败时留下的半截正文只算它）：
  `reasoning_content` / `reasoning`、Anthropic 的 `thinking_delta`（默认 `display: "omitted"`
  时是一个空串）只是让软超时的静默重新计时的字节。
- **失败分类**：非 2xx 抛 `RuntimeError("HTTP <code> <body>")` 走现有关键词分类；流里的报错事件
  （Anthropic `event: error`、OpenAI / OpenRouter 的顶层 `error`）抛同样文本形状的
  `ProviderStreamError`，状态码取语义码：Anthropic 按错误类型（`overloaded_error` 529、
  `rate_limit_error` 429、`api_error` 500 …），OpenAI 系按数字 `code` 或 `insufficient_quota`
  之类的字符串，认不出按 500；
  新增两条映射——**402 → quota**（DeepSeek 余额尽）与 **429 `insufficient_quota` → quota**
  （OpenAI 账户额度尽），组内直接推进而不是原地重试。**日封禁不适用**：strike 依赖
  Gemini 结构化 `quotaId`，其他供应商的 429 归 rate_limit，撞满就退避重试。
- **联网搜索**（catalog 计划 D9，2026-11-09）：只在 `retrieval=native` 的调用上发。Anthropic 带
  `{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}`（基础版：带动态过滤的
  `web_search_20260209` 只有较新的模型认，Vertex 与 Azure 上的 Claude 也只有基础版）；拼装器另留
  `server_tool_use`（搜索词）与 `web_search_tool_result`（结果 URL）块，按 `tool_use_id` 记成一次搜索
  一行的检索记录，正文去掉紧挨在每次搜索前面的那段开场白，`usage.server_tool_use.web_search_requests`
  记进调用记录。`pause_turn`（服务端循环跑得久时 API 暂停了它，官方不给同步调用的轮数上限；超过 `max_uses` 的搜索则是
  `max_uses_exceeded` 错误结果、模型照常作答）不续接：抛 `SearchTurnPaused`（不带状态码、`route_failure_kind =
  "transient"`），不原地重试、不进冷却（冷却后的探测撞上它也算服务端答了、冷却结束），候选链换下一个成员——续接要每段各算硬截止、静默备份会重跑
  并重扣搜索费、合计输出会被误判成截断。回复以 `stop_reason = "tool_use"` 结束或带名为
  `web_search` 的客户端 `tool_use` 块，说明端点没在服务端搜，抛 400 的 `ProviderStreamError`，
  这次调用失败、不换成员。OpenAI 兼容方言没有通用开关：模型的 `search_request` 浅合并进请求体顶层，
  加载时拒绝与传输自己写的键同名；检索记录只记一行「开了搜索」。开了搜索的调用不进估算校准的样本：
  返回的输入 token 含搜到的内容。两处已知的不精确：`web_search_requests` 只记在作答的那次尝试上，被静默备份
  替下的请求和因 `pause_turn` 失败的尝试搜过也不计（少计，不影响路由）；主请求触发上面的 400、而静默备份
  正在跑并以硬超时结束时，抛出的是备份的超时，判为 TIMEOUT，会换成员——只在主请求先静默到触发备份时出现，
  后果只是换成员，不会不搜就作答。
- **usage**：报告与校准以 API 返回值为准（缺失记 0 交由现有口径处理）。
- **媒体**：第一刀不做（`supports_audio/video=false` 固定）；纯文本强模型用在音视频素材
  上的路径是 `--media text`。
- **key**：`FINESUB_KEY_<PROVIDER_TIER>`（可用 catalog 的 `key_env` 键覆盖），走 `secrets.py`
  同一条加密 `.env` 路径，不另开明文存放点。
- **声明位置（2026-08-12 起）**：provider 方言与 base_url 是**事实**，写在数据根的
  `model_catalog.toml` 的模型表上（`provider_kind` / `base_url` / `key_env` 三个键），一个表即声明
  provider + fact + target；`config.toml` 只留模型组与预设。旧的 `[llm.providers]` /
  `[llm.models]` 会报错并指向新位置。

## 缓存

各家都给最近见过的前缀打折，而一条 run 的调用共享很长的前缀：相邻两个纠错窗逐字相同的开头约占七成（system 加全局
context pack，断在 user 里本窗的字幕条数那句）。这里写的是为了让折扣落下来做的几件事，实现都在 `prompt_cache.py`；其余取舍的理由见 [`llm_design_notes.md`「付费 API 前缀缓存的决策记录」](llm_design_notes.md#付费-api-前缀缓存的决策记录2026-10-24)。
2026-10-24 在 OpenRouter、DeepSeek、Anthropic、Gemini 上各跑过一次，读数见下文[验收读数](#验收读数)。

现行行为（2026-10-24）：

- **记账**：用量分未缓存、缓存读、缓存写三份（`extract_token_distribution` 的 `cache_write_input_tokens`），任务报告的
  Provider 表多一列「of which cache write」。Anthropic 与 OpenAI 新模型的写入比普通输入贵，以前它被折进总数里看不见。
  只有 API 传输报写入：本地 agent 的写入照旧算在未缓存里（CodeBuddy 报的「写入」其实是上游没命中的部分，没有写入溢价；agent 档按订阅计费）。
- **OpenAI 兼容**：每次带 `prompt_cache_key`，值是 system 的哈希：同一份 system 的调用（一条 run 的各窗、同 prompt 的不同 run）
  路由到同一台机器；OpenRouter 也认它作会话键，把同一会话粘在一家上游。有的自建端点见到不认识的字段回 400（FastAPI 一类回 422），
  所以端点回 400 或 422 且报错点名这个字段时，当次去掉它重发一次，本进程里对这个 `base_url` 不再发（被拒的请求不计费）。不另设目录列（owner 2026-10-24：
  目录的列已经够多，能自动判断就自动判断）。
- **Anthropic**：system 后面一个 5 分钟断点，各窗、各 run 都读它。⚠ 跨窗只盖得住 system（约占一窗输入的六成）：全局 context pack
  在 user 消息开头、不在块边界上，Anthropic 读不到它；自动前缀缓存的几家能读到 system 加 pack（约七成）。调用方声明还有下一轮时（纠错窗的请求轮、准备任务带索取的 t1，
  `RoleClient.complete(followup_expected=True)`），最后一条消息后面再加一个；对话的后一轮（fill、t2、修复轮）在开场末尾（第一条回复
  之前的那条消息）打同一个断点去读它——Anthropic 只从断点往前找条目，后一轮自己不打就读不到。首轮不为修复轮预先写：写一次是 1.25 倍、
  读一次 0.1 倍，下一轮出现的概率要超过约 28% 才回本，而本机 1036 次纠错回复里修复轮约占 13%；第一次修复时才写，需要修一次的窗口约一半
  还要再修，第二次就读得到。带 seed 的原样重问不打尾部断点：下一轮重发的开场不带 seed，那里写下的条目谁也读不到。
- **并行先暖**：两种付费方言上，同一前缀（端点、key、模型、system 哈希；缓存按账户分）冷的时候，第一个请求先发，同时到的请求等它结束再一起发；
  成功之后 5 分钟（Anthropic 缓存的默认寿命，三家里最短）内算热，不再等。串行从不等；并行每个前缀多等一次调用的时间。
  Gemini 不走这道闸：用它的主要是免费档，命中不省钱，等只费时间。没有前缀缓存的自建 OpenAI 兼容端点也会走闸，开跑时白等一次调用，不为它分辨。
- **seed**：只在原样重问时追加（[`llm_harness_behavior.md`「采样参数」](llm_harness_behavior.md)）。首轮带了 seed，修复轮重发的首条
  user 就和发过的那条对不上，缓存断在那里。

各家机制里决定上面做法的几条（2026-10-24 查的官方文档；价格与门槛常变，动手前再查）：

- [DeepSeek](https://api-docs.deepseek.com/guides/kv_cache)：默认自动、存在磁盘上，几小时到几天；命中价约为未命中的 2%
  （[价格](https://api-docs.deepseek.com/quick_start/pricing)）；前缀要先被识别、建缓存要几秒，所以头一两个共享前缀的请求不中。
- [Anthropic](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching)：不打断点就不缓存，最多 4 个；
  写 1.25 倍（1 小时档 2 倍）、读 0.1 倍；第一个响应**开始返回后**条目才可读，同时发出的请求各写各的。
- [OpenAI](https://developers.openai.com/api/docs/guides/prompt-caching)：1024 token 起自动缓存，`prompt_cache_key` 只影响路由、
  不保证命中；Responses 的 `previous_response_id` 照样把整条链按输入计费，所以不用它。
- [OpenRouter](https://openrouter.ai/docs/features/prompt-caching)：缓存随上游；命中之后把同一会话粘在那家上游，前提是它的缓存读
  比普通输入便宜。免费模型输入价为 0，文档没提缓存；实测免费模型照样报缓存读（见下文验收读数）。

没做的：Anthropic 上给 context pack 单独打断点（验收那次 pack 约占一窗输入的 7%，省约 6%、只对 Anthropic 有用，代价是渲染层在 pack 边界把 user 消息切成两块；pack 占比超过两成时再议）；Gemini 的 `cachedContents`（按存储时长另计费，隐式缓存够用时不加）；把字幕条数那句挪出提醒段让前缀再长约 6%（改提示位置，
要做质量对照，收益不够）；价格进目录、报告直接出金额（价格变得快，进目录就要有人维护）。

### 验收读数

2026-10-24，每家一次，日语直播素材、纯文本。「第 2 窗起」= 首个纠错窗之后的纠错调用里，缓存读 token 之和 ÷ 输入 token 之和。

| 端点 | 跑法 | 第 2 窗起 | 说明 |
| --- | --- | ---: | --- |
| OpenRouter 免费模型（`nvidia/nemotron-3-ultra-550b-a55b:free`） | 改后串行，10 窗 | 40% | `prompt_cache_key` 不被拒；10 窗里 3 窗首轮读 0，命中不保证 |
| DeepSeek `deepseek-flash` | 改后并行 4 路，5 窗 | 85% | 先暖闸门生效：首窗读 0，后发的两窗各读 13,952；占比被截断后的原样重问抬高，各窗第一次调用约 58% |
| Anthropic `claude-haiku-5-5` | 改后串行，只跑完首窗 | 54%（首窗拆出的两半） | t1 写、t2 整段读回；system 断点被之后每次调用读到；写入进了报告 |
| Gemini `gemini-3.5-flash-lite` 免费档 | 改前、改后串行各一次，3 窗 | 45% → 61% | 改后修复轮读到首轮输入的 95%；改后紧接改前跑，首轮命中可能沾了改前的缓存 |

- 各家各窗第一次调用读到的都约等于 system 那么多（约占一窗输入的 52–58%）：那次 context pack 只有约 2,500 字符，自动缓存比 Anthropic 多读的很少。
- DeepSeek V4.1 Flash 与 Haiku 5.5 不发思考参数时默认思考，两次都把 32k 输出上限用在了思考上（Anthropic 那次因此只跑完首窗）；目录还没有「关掉思考」的写法，见 [`llm_followups.md`「目录没法声明「关掉思考」」](llm_followups.md)。
