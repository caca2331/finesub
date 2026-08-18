# LLM 路由的 dev 侧：模型事实、池、思考档位与限流

> 从 [`llm_harness_behavior.md`](llm_harness_behavior.md) 拆出（2026-08-16，原文 930 行）。
> 那份仍是 harness 运行时行为的总入口；本文是它的一块，单独成篇是因为读者不同。

> **面向使用者的同一主题**在 [`manual/model-routing.md`](manual/model-routing.md)：
> 那份讲「你要改哪几行」，本文讲「运行时按什么规则挑、限流怎么算、思考档位怎么换算」。

## 模型事实、池与路由链

模型运行配置分成两份随包发布的数据：`model_catalog.psv` 是 provider tier + endpoint 维度的
运行事实（`quality_score` 为咨询性字段；v2 已删除 `correction_prompt_tier`——变体归属在
任务组格子上），`model_routes.toml` 显式声明 execution target、**模型组、任务组与预设**
（v1 的 pool/chain 已由它们取代）。同一名义模型的 free/paid 行不合并，因为能力、context、
quota 或工具开放情况可以按 tier 不同。

**7 个任务组 × 3 档 difficulty = 21 个格子**；预设把格子绑到有序模型组，格子携带默认
变体与 thinking 档位，模型组条目可选覆盖变体（prompt 迭代用）。**格子支持难度回落**
（2026-08-11）：没绑的 intermediate/efficiency 沿 efficiency→intermediate→quality 在**同一
预设内**向上找（difficulty 只换 prompt，不悄悄换别人的组）；整组一格都没绑才跨到 default
预设。因此 default 只需绑每组的 quality（loader 强制）——现行 9 个绑定：7×quality +
correction-mm/intermediate + correction-text/intermediate（correction 的 intermediate 显式
降到 lite 组，是"免费额度尽→切档继续"的出口；efficiency 继承它）。
有意变更（相对 v1 链）：correction 的 lite 移入 intermediate 格；knowledge 用自己的窄组；research
**去掉了 lite 兜底**、lightweight 只留 3.5-lite（免费档撞额度时这些任务停下等待或走付费尾巴
[3.6]，而不是降质）。组内 fallback 统一为 `unavailable/quota/rate_limit/timeout/transient`
（D8）。新增 catalog fact 不会自动上线，必须显式进入 target、模型组和预设绑定。
绑定期告警在启动校验时按**当前激活的预设**打印：**下限分数只查 quality 格**（floor 守的是
任务的标称质量；correction/knowledge 70，其余 50——另外两档本身就是声明过的降档，不告警）、
规划包络与 -mm 格媒体能力查全部格子；默认预设零告警有测试保护。下限的「参考模型」是**派生
的**（default 预设**该格**所绑模型组的首个成员，所以 correction 的 intermediate 格参考的是
3.5 Flash Lite），不在声明里写死。

**用户自定义（model-routing v2，`config.toml [llm.*]`，样例在 `config.example.toml`，有一致性测试）**：
`[llm.providers.<id>]`（kind=`openai_compat|anthropic` + base_url；key 按 `FINESUB_KEY_<ID>`
放 `.env`，同一条加密路径）、`[llm.models.<id>]`（成为 self-reported fact + 自动 target；
**`max_input_tokens` 必填**——按它切窗，填错每窗都炸；其余限额缺省按付费档，输出上限缺省
65536）、`[llm.model_groups.<id>]`、`[llm.presets.<id>]`（未绑格子回落 default 同名格，
test_target 可省继承 default 的），`[llm] preset = "<id>"` 选激活预设。
**同名即覆盖**（2026-08-12 统一）：用户的模型组/预设用打包已有的 id 就替换掉那一个，
与 catalog 按 `fact_id` 覆盖同一套心智模型；覆盖 `default` 预设也允许，其完整性校验
（每个任务组的 quality 格必须绑上、test_target 必须可达）跑在合并之后。**test_target 可达性
只查本次能选到的预设**（激活的那个 + default）：查每一个声明过的预设，会让一个打包预设因为你
替换了 `default` 而报错——哪怕你从没选过它。代价是拼错 id
不再被拦，因此每次覆盖会在启动校验处打印一行 `Note: config.toml 覆盖了打包声明的 …`。
**格子可以直接绑 target id**（快速选模型），loader 自动包成单成员组 `target:<id>`；同名时
模型组优先，且声明的组不得用 `target:` 前缀。
注意 `config.toml` 文件本身**不**跨目录合并：按「checkout 根 → user-data」取第一个存在的。全部命名表，无
`[[数组表]]`，所以 `config_file.py` 的标量写入器照常可写——D11 预告的 round-trip TOML
放宽实际不需要，暂不引入。行为要点：**自定义 HTTP endpoint 只做纯文本**（媒体由打包 Gemini REST 或本地 Agy 承担，
配 `--correction-media text` 使用）；**非 Gemini 不传采样参数**（D18，重掷靠 prompt 尾部
seed 文本）；**日封禁不适用**（strike 依赖 Gemini `quotaId`；其他家 429 归 rate_limit，
402/`insufficient_quota` 归 quota 直接推进组内下一位）；**未知限额不 fail-closed**
（self_reported fact 放行，artifact 标注来源）；**规划包络取组内最低**（D13：纠错窗口与知识更新分块都按
`min(max_input)`/`min(max_output)` 规划、并按组内最大 `token_scale` 换算，包络由谁决定在绑定期告警里点名；fast 会话内部的固定上限仍是遗留项）；**改未被激活
预设引用的模型/组只动 advisory digest**，也不影响新调用的路由身份。供应商行为差异与
adapter 契约见 `docs/provider-adapters.md`。

每次调用先按 provider 是否启用、日额度、输入模态和 native-search 能力预过滤；随后按候选的
**变体**（格子默认或条目覆盖）惰性组装实际 messages，再以该 target 的
`max_input_tokens`/`max_output_tokens` 做硬检查。远端失败被分类为
`unavailable/quota/rate_limit/timeout/transient/permanent`，按统一 fallback 集合退让；
确定性 4xx 默认立即上抛。未知 runtime fact 在生产路径 fail-closed，测试 fixture 必须显式声明
`unverified=True`。`test_profile` 由**预设级** `test_target` 保证只打桩（v1 的链级 test target 上移）。

每次成功或失败都会在 task artifact/exchange metadata 记录 `route_decision`：policy、chain、
`routing_identity_digest` 与 `advisory_digest`、完整 effective chain/fact snapshot，以及每个候选的
accepted/skipped、原因和执行结果（含 video→audio 安全网触发时的 `media_downgrade` 标记）。
因此不仅能看到实际调用了谁，也能重建其他 target 为什么未被选择。

**配置 digest 已拆成两个**：`routing_identity_digest` 只覆盖会改变
「这次调用可能选到谁、用什么 prompt、按什么预算切窗」的配置——打包声明按**解析后的结构**
（可达 target、执行 profile、格子变体、policy）而非文件字节参与，加上除咨询字段外的 runtime
facts 与用户侧的激活预设/绑定/思考旋钮/模型组——它进入调用路由与尚未提交的 session 调用身份（L1），
**不进入已提交 research/window/stage 的失效键**（L2/L3），这正是「换模型组继续跑」能用的原因
（docs/llm_local_agent.md §11）；`advisory_digest` 覆盖咨询性字段（显示名、
`quality_score`、`floor_score`、预设名），只进 artifact。

`config.toml [llm].execution_policy` 可选：

policy 是**闸门**：它按 backend 过滤绑定的模型组，不改变顺序，也不引入组外的 target。

- `agent-text-preferred`（默认）：不拦任何 backend。名字里的 "preferred" 指的就是"不拦
  agent"——谁在前完全由模型组的成员顺序决定（`agy` 预设是 free API 在前、agent 在后）；
- `api-only`：去掉 local-agent backend，剩下 Gemini 与用户自声明的 API provider；
- `agent-only`：只保留 local-agent backend。配一个全是 API 模型的预设会得到空链并在启动校验处
  报错——这是正确行为，policy 不再凭空变出模型。

组内顺序是 fallback 链而非质量排序；只有声明为 unavailable/timeout/transient 等错误才继续组内
fallback。provider tier（`LOCAL_CODEX`/`LOCAL_CLAUDE`/`LOCAL_AGY`）决定用哪个 CLI driver，只装了
部分 CLI 的机器由 readiness 预筛自动跳过另一家。

两个 agent 策略下，窗口剪辑先只以**本地路径**引用，不预先上传 Gemini Files——`agent-only` 因此
根本不需要 Gemini key。只有当某个 API 候选真的要回答时才惰性上传，且缓存在 **client 实例**上按
`(provider_tier, 剪辑绝对路径)` 记账：同一个窗口的查询轮、纠错轮与每次重试共用一次上传（Files
对象 48 小时过期，调用失败时会丢弃缓存条目，下次重传）。

媒体规划包络按**声明的**候选链算，不看 agy 是否真的装了：只要路由里有 `video_high_resolution_only`
的成员，视频就按高分辨率档（269 tok/帧，对低档 71 的 3.79×）切窗。方向是保守的，但在没装 agy
的机器上选了 agent 策略，视频窗口会白白变小。
  fast/correction 的可直接调用 stage 入口也会在剪辑或 Gemini Files 上传前按实际 routed plan
  做媒体能力 preflight，因此不会先产生 provider 辅助请求再失败。
  该 policy 还会从 Harness local-retrieval fallback 移除 Gemma4 grounded，避免通过搜索子系统
  绕行 Gemini `generateContent`；Exa/Tavily/DDG 仍按其独立 provider 配置工作。

本地 agent 的完整设计与未实现项见 [`docs/llm_local_agent.md`](llm_local_agent.md)。Codex、Claude
Code 与 Agy 共用 one-shot transport：每次调用在专用 parent 下原子建立 episode，以供应商各自的结构化事件
和严格 terminal schema 执行；Harness 从终态 assistant message 写 staging，业务 contract validator
通过后才提交。completion 禁止搜索/tool 事件；native target 允许联网，但模型本轮未调用搜索只记录
`native_search_not_used` note，不再判 transport 违规。Windows suspended 创建后先绑定 kill-on-close
Job Object，POSIX 使用独立 process group。

当前 CLI sandbox 不提供全盘读取隔离；Codex 使用进程级只读 sandbox，Claude Code 使用具名工具拒绝
列表，并对事件流里实际发起的 `tool_use` 做审计（初始化工具集里出现未授权名字只打告警，不再判
违规——否则一次 CLI 升级就是 permanent 失败、整条 API 链不再尝试）。agy 的读边界是它那个受控
project 根，也就是**整个 episode parent**，而不是本次 capsule：并发调用之间、以及与留存的失败
现场之间，读权限是共享的（都是本 harness 自己的数据）。子进程环境按 allowlist 裁剪，其中**不含
`HTTP(S)_PROXY`/`NO_PROXY`/`NODE_EXTRA_CA_CERTS`**——代理后面的机器上三家 CLI 都会连不上，且失败
会归类成 unavailable/transient，不容易看出根因。
不能忽略 user config/rules 的旧 CLI 默认不可用，只能以明确的
`local_agent_allow_unisolated_user_config=true` 做实验。API 与 agent 均写 `execution_attempts`；API 另
保留 `api_attempts` 兼容字段，agent native-search attempt 内保留紧凑 query/URL events。policy、
`routing_identity_digest`、capsule schema、driver/protocol/config digest、toolset、sandbox 和 agent
profile 进入 resume execution identity。若调用方注入自定义 client/router，fingerprint 使用该实际
client 的 execution identity，不读取全局 policy 代替。

transport 成功后只删除本次原子创建并记下的 episode；transport 失败，或成功后的清理失败时保留
有界现场并写稳定 evidence locator。运行时不扫描、认领或清扫别的 episode。打包安装的 parent 是
当前大文件根下 `agent-capsules`；源码 checkout 使用仓库外
`%TEMP%\finesub-agent-runtime\<完整域 digest>`。清理、搬迁和卸载语义见
[`docs/manual/resources.md`](manual/resources.md)。

**打包 Gemini 这条链是 REST 直连**：所有生成类调用（含音频多模态）走
`client.complete` → `llm.chat_complete` → `_gemini_generate_content`，free tier 失败可
遍历到 paid tier，两者都服从 `config.toml` 的 provider 开关与 pool 顺序。音频以 Files API
的 `fileData.fileUri` 引用注入（`upload_gemini_file` 走 REST 上传，拿到 URI 直接进
generateContent body）。`countTokens` 与文件上传这两条辅助路径同样是直连 REST
（`x-goog-api-key` header，**错误输出不得包含 key**），用第一个启用且有 key 的 Gemini
pool（Free 优先，其次 Paid）的第一把 key。

## LLM thinking effort

思考深度由两层组成（2026-08-11 设计）：

1. **预设级思考旋钮**（`[presets.<id>.thinking]`，键 `"任务组/难度"`，值抽象档位
   low/medium/high）：与模型组绑定同一套档位回落（同预设内 efficiency→intermediate→quality，
   整组未填才跨到 default 预设），**全部不填时取 medium**。旧的 per-call 覆盖（最省档纠错、
   search judge 常量）已随之移除，旋钮是唯一来源。打包 default 的取值见下方「出厂旋钮实况」。
   注意回落方向**朝上**：research/knowledge 的 intermediate 格没有写死，正是靠这条继承了
   quality 的 high。
2. **模型级 thinking 映射**（catalog / `[llm.models]` 的 `thinking` 列，默认恒等）：

   | 取值 | 含义 |
   | --- | --- |
   | `true`（打包行全是它；`[llm.models]` 不写该字段时的默认） | **恒等映射**：抽象档位原样发出 |
   | `false` | 该模型完全不发思考参数 |
   | `高映射,中映射,低映射` | 三个供应商取值，覆盖恒等 |

   恒等之所以是合理默认而非巧合：三种方言的档位词本来就同名——Gemini
   `thinkingConfig.thinkingLevel` = low/medium/high、OpenAI 系 `reasoning_effort` =
   low/medium/high（另有 `xhigh`/`minimal` 等扩展值）、Anthropic `output_config.effort` =
   low/medium/high（另有 `xhigh`）。只有当某端点的词不同（如 `max,default,minimal`）才需要
   写显式映射。`false` 是对"未知字段可能 400"的自建端点的声明式出口。映射在 client 逐候选
   套用（含本地 Codex driver），产物记录的是**映射后实际发出**的值。本地 driver 的
   `[llm].local_agent_reasoning_effort` 默认为空；非空时是刻意压过所有模型映射的全局兼容覆盖。

Gemini 3.x 通过 `thinkingConfig.thinkingLevel` 控制思考深度；`thinking_level` 为空时才按
`thinking_budget` 折算（≤800→low，>800→high）。

2026-07-12 使用 `GEMINI_FREE` 直连实测确认 `gemini/gemini-3.1-flash-lite` 支持原生 thinking（`thinkingLevel=medium` 返回 `thoughtsTokenCount`）；此前 catalog 将其标为不支持是误判（该结论在旧 `supports_reasoning` 列时代得出，现映射列同样适用）。

按 token 数计的 `thinking_budget`（供不支持 thinkingLevel 的模型）不再单独维护，一律由 level 按 API 输出上限的比例派生（`config.thinking_budget_for_level`）：low/medium/high = 20%/40%/60% × 65,536 → 13,107 / 26,214 / 39,321。

出厂旋钮实况（2026-08-12 调整）：

| 任务组 | quality | intermediate | efficiency |
| --- | --- | --- | --- |
| correction-mm / correction-text | medium | medium | **low** |
| planning-mm / planning-text / search_judge | medium | medium | **low** |
| research / knowledge | **high** | **high**（继承） | **low** |

即：**efficiency 一律 low（13,107 budget）；research 与 knowledge 一路到 intermediate 都是
high（39,321），其余 medium（26,214）**。抬高这两组的理由是它们的产物被下游全量复用——
research 写的 context pack 每个窗口都读，knowledge 直接写库；换档换的是模型，不该顺带砍掉
推理，所以它们的 intermediate 格**不写**、靠向上回落继承 quality 的 high。任何一格都可在
用户预设的 `[llm.presets.<id>.thinking]` 覆盖。

## 模型配置、速率限制与显式 reasoning

`model_catalog.psv` 是 pipe-delimited 模型事实表，**一行一个可调用的 (provider, 模型)**，
列为 `fact_id|provider_tier|provider_kind|base_url|key_env|display_name|api_model_id|max_input_tokens|max_output_tokens|supports_audio|supports_video|supports_native_search|thinking|token_scale|rpm|tpm|rpd|tpd|is_free|quality_score`。
`provider_tier` 与 `.env` entry 名一致（`GEMINI_FREE`、`GEMINI_PAID`，或自定义 provider id）。
**`tpm`/`tpd` 仅指输入 token**（不含输出/thinking）；`rpd`/`tpd` 列仅供人工参考，运行时
**不预追踪**日额度。

**两份 catalog（2026-08-12）**：随包发布的默认表 + 数据根下同名的覆盖表，按 `fact_id` 合并、
后者胜（顺序保持，覆盖不会把模型挪出它在组内的位置），覆盖行标 `self_reported`。数据根的解析
与 `.env`/`config.toml` 同一条（`finesub.paths.resolve_model_catalog_override`，可用
`FINESUB_MODEL_CATALOG` 指定）：装好的前端是 `user-data`，仓库版没有单独 user-data，就是
checkout 根。**表头声明列**（必填 `fact_id`/`provider_tier`/`api_model_id`/`max_input_tokens`，
未知列名报错；旧名 `litellm_model`/`model` 报错时直接给出新名对照），列内留空取默认值：输出 65536、无媒体、thinking 恒等、`token_scale=1.0`、
RPM 100、TPM 4M、日限额无限、`quality_score=50`、`provider_kind` 按打包 tier 推断否则
`openai_compat`。
覆盖是整行替换而非补丁。

**`[llm.providers]` / `[llm.models]` 已从 `config.toml` 移除**：端点方言与 URL 是事实，进
catalog 列；`config.toml` 只留编排（`[llm.model_groups]`、`[llm.presets]`、`[llm].preset`
与执行策略）。写了旧表会直接报错并指向新位置，不会被忽略。

**`token_scale`（每 fact，默认 1.0）**：本地三级计数器用 Gemini 词表，别家分词不同，估算有
系统偏差。系数乘在**本地估算**上，只作用于三处——候选的 `max_input_tokens` 硬检查、限流的 TPM 预留，
以及**窗口规划的输入包络**（规划按局部估算口径计数、`max_input_tokens` 按供应商口径，系数是
两者的换算；规划取组内最大系数且不低于 1.0，否则规划说装得下、发车却判超限，每窗都被跳过）；
**报告与成本核算一律用 API 返回值**，不乘系数。每次调用把「返回 input tokens /
未缩放的本地估算」记进 route decision 的 `estimate_calibration`，task report 的
「Token Estimate Calibration」按 fact 给中位数、样本数与建议值（样本 <5 或偏离 <10% 不给
建议）。**不自动写回**：它改窗口几何、进 routing 指纹，采纳是一次显式配置变更。

### 生成 API 速率限制（`finesub/llm/rate_limit.py`）

- **限流桶**：`(provider_tier, api_model_id, key_id)`；限额来自 catalog 对应行 × **0.9** 安全系数。每把 key 独立计数（Gemini 免费层 RPM/TPM 是 per-project）。
- **主动追踪**（**61s** 滑动窗）：**RPM**（请求次数）与 **TPM input**（输入 token 预扣/结算）；输出 token 不影响 TPM 等待。每个实际 HTTP 尝试（含 sticky 失败重试）都记入 RPM；首次 attempt 走 `acquire`（RPM+TPM 预扣），后续 sticky 重试走 `note_request`（只记 RPM）。P7a 起为**取号制**：`reserve` 在同一把进程内锁里算出发车时刻并当场入账，调用者在锁外等待；带 TPM 的新调用按 bucket FIFO。RPM-only retry 使用独立 `_rpm_only_slot`：只计算候选时刻已经发车的 RPM 请求，不受其他窗口未来 TPM 预约或 FIFO horizon 拖延。`settle` 按本次票据校正自己的 TPM 预扣。日封禁 strike 只计「发车晚于上一 strike」的尝试；client 级不再另做限流。
- **是否限流/是否可重试先看状态码**（2026-08-08 修）：`is_quota_or_rate_limit_error` /
  `is_retryable_provider_error` 先取 `provider_status_code(exc)`（依次为
  `exc.response.status_code`、`GeminiAPIError.status_code`，都没有才从消息里按**词边界**
  抓一个 4xx/5xx），只有**取不到状态码**时才回落到文本匹配。此前文本匹配是主信号且**无
  边界**，而 `GeminiAPIError` 的字符串里嵌着完整 JSON 错误体：一个确定性的 400
  `The input token count (215000) exceeds the maximum` 就因为 `215000` 里含 `500` 被判成
  可重试，同理任何 id/计数里的 `4290` 会被当成 `429`。后果是白白冷却一把 key 并原地重试
  一个重试一万次也不会变的参数错误。
- **错误分类**（`client.classify_quota_error`，按**结构化 `quotaId`** 而非 `retryDelay` 提示）：`...PerDay...` → `DAILY`、`...PerMinute...` → `PER_MINUTE`、其余 429/限流 → `OTHER_RATE`。**不用 retry 提示判日限**——Gemini 对真·日耗尽也返回 ~20–60s 的通用退避，提示区分不了日/分钟（旧 `has_short_retry_hint` 门控正因此把真日限误当瞬时，已删）。
- **日封禁需 strike 确认（per-key）**：单发的 `PerDay` 不立即封。`rate_limit.note_daily_quota_hit` 累计 strike，**连续 ≥3 次**（`DAILY_STRIKE_COUNT`；成功即 `reset_daily_strikes` 清零）才写入 `.state` 的 `llm_rate_limit.daily_exhausted`，在 **Pacific 日历日**内跳过该 key。不再要求首末跨度（旧 `DAILY_STRIKE_SPAN_SECONDS` 已删），但**跨 Pacific 日的旧 strike 会被丢弃**（2026-08-08）——此前时间戳只写不读、streak 只在成功或封禁时清零，导致「周一两次 flicker + 周四一次 429」在全新日额度上凑满 3 次而误封。strike **按 HTTP 尝试计数**（含同一逻辑调用的 sticky 重试）是有意为之：尝试之间的等待是 `max(指数退避, provider retryDelay)`，真·PerDay 429 带的 retryDelay 很长，所以三次尝试跨越的是有意义的时间段，而非一阵突发。strike/`daily_exhausted` 以 **`(tier, model, key_id)`** 记账（照 exa `ApiKeyPool` 模式）：named key 用其名称，匿名 key 用 `sha256:<前12位hex>`——**.state 不明文存 key 原值**。一把 key 的日封不连带同 endpoint 的其他 key。免费档 `PerDay` 信号会 flicker，故不凭一次就封整天；strike 与 `daily_exhausted` 均落 `.state` 跨进程可见——写入时按 key **合并**而非整段覆盖（2026-08-08 修）：limiter 是 `lru_cache` 的、内存副本在 `__init__` 时读取一次，整段写回会抹掉另一个 FineSub 进程期间写下的日封禁；本进程**主动清除**的项不会被磁盘副本复活。`PerMinute`/普通 429、临时 5xx 退避重试或回退下一 endpoint；生成请求单次 timeout 为 15 分钟，**sticky retry budget 为 3**（观察发现哪怕 5xx 也会占用日额度，故从 7 下调并拉长退避），但连续两次 timeout 会提前抛出原始 timeout failure；参数/鉴权等不可重试 4xx 立即上抛。
- **429/可重试错误退避公式**：`sleep = min(max(4×2^attempt, parse_retry_after_seconds(exc)), 300) + 1`（基数 2026-07-29 由 0.5 改为 4，同上：少烧额度、拉长间隔）。`parse_retry_after_seconds`（`rate_limit.py`）解析主流 provider 的等待提示（Gemini `retryDelay`/`"Please retry in Xs"`、OpenAI/Anthropic `Retry-After`/`retry_after`、通用 `"wait Xs"`/`"try again in Xs"`），无提示时取 0；上限 300s 防止 provider 返回异常大值。
- **Endpoint 链**：`LLMRole` 仍是调用点兼容 API，但实际有序 target 来自 `model_routes.toml` 的
  route chain。`audio_multimodal`（纠错窗 / fast 纠错步）API 基链为
  `FREE+3.7-flash → FREE+3.6-flash → FREE+3.5-flash → PAID+3.7-flash`；
  `general_capable`（research r1/r2、fast r1、知识更新等）为
  `FREE+3.6-flash → FREE+3.5-flash → FREE+3.7-flash → PAID+3.7-flash`；
  `lightweight_multimodal` 与 `lightweight` 共用 3.5 Flash Lite 的 free→paid 池。原生搜索按调用在
  当前绑定组内过滤；显式 opt-in 的 `gemini-native-search` 是免费 2.5 Flash → 付费 3.7 Flash。
  execution policy 可在基础链前叠加 agent pool。
  启动能力校验在 fast 规划之后按**本次实际调用形态**检查：local 常规流要求 planning 任务组的
  查询轮链，local fast 改为要求 correction 任务组的 media-bearing fast R1 链；不会同时要求两条
  互斥链。
- **`chat_complete` 的 key 处理（sticky + per-key daily）**：每次调用指定单一
  `provider_tier`（或 `profile`），先按 `config.toml` 解析该 tier 的 pool；被关闭或
  无 key 的 tier 在 endpoint chain 中跳过。同 tier 下按 pool 顺序取 key，**跳过已
  daily-exhausted 的 key**，**429/限流在同一把 key 上原地重试**（不首撞即换），
  PerDay 429 同时喂给该 key 的 strike gate（gate 确认即锁该 key 并立即轮换），仅当
  该 key 的重试预算全花在限流上才轮到下一把；media 调用因上传文件项目隔离钉选定
  Gemini pool 的第一把 key。batch 的 LLM task pool 仍为 1；单个 task 内只有
  `continuity=parallel` 会并发查询轮/纠错窗，并因媒体上传项目隔离固定到池中第一把 key。
- **组合临时冷却（`combo_cooldowns`）**：`(tier, model, key_id)` 在一次调用内耗尽 sticky
  retry（可重试错误）后进入冷却：**0–20 分钟** skip（立即换链上下一组合，不干等）、**20–120
  分钟** probe（sticky retry=0，成功清除、失败重置起点）、**≥120 分钟**自动清除。持久化于
  `.state`，与 daily-exhausted 独立。probe 是**每个冷却窗口一次**，不是每个并发窗口一次：
  读相位是纯读，`continuity=parallel` 下所有并发调用会同时读到 probe，所以由
  `claim_combo_probe()` 原子认领，认领失败者按 skip 处理（与 daily strike 的 `departed_at`
  闸门同理——把「N 个并发观测同一件事」压成一次）。认领只存在于进程内、按窗口起点计，
  失败重置窗口即重新开放；探针不落地则该窗口内不再探测（保守方向）。
- **尚未做（后续）**：滑动窗（RPM/TPM）仍为进程内内存态（进程退出即重置）；跨进程持久化滑动窗为后续项。
- **`countTokens`** 与文件上传不走生成限流器；`test_profile` 禁用 limiter 以免拖慢单测。

v17 起所有 prompt 模板都要求**回复以一个 `<reasoning>...</reasoning>` 块开头**。v76 起可见
措辞不再随 thinking 档缩放；BASIC 纠错变体仍使用受限的 8 行版本。运行时不再改写 message；
缺块不校验、不重试，`<reasoning>` 不参与业务解析。

### 纠错 prompt variant（由任务组格子决定）

catalog 不再拥有 correction prompt tier。`model_routes.toml` 的任务组格子直接声明 variant；
`role_config_for` 在调用前解析该格，`RoleClient.complete` 对整条候选链使用同一个显式
variant。这样 provider fallback 只换执行 target，不会在调用中途静默换 prompt 契约；要降档
必须由用户显式切 difficulty。实际 variant 进入 exchange、窗口缓存与任务产物。

capableC 与 basicB 都使用
`fragment_merge_rules_nosingles_v1` 的判断型合并规则，capableC 额外要求决策点前置局部
reasoning，basicB 额外带 start 列。保守 1:1 的 basicA 和无局部 reasoning 的 capableB
仅供显式对照。variant 无关的产出纪律（gap 方向、char_count 列纪律、列核对）抽在
`fragment_translated_common_v1`，四个变体共用。`--prompt-dir` 另落
`correction-0001-basic-tier.txt` 作为 basic 对照。旧缓存或工具未显式传 variant 时仍可按
capability tier 选择 capableC/basicB，这是兼容路径，不是生产路由规则。
