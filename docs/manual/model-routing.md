# 模型路由配置手册

**本章回答：一次 run 的每一次 LLM 调用会发给谁，使用什么 prompt、多少思考深度、是否携带
音视频；以及接入自有模型时需要修改哪几处。**

面向使用者。实现细节与设计取舍见 `docs/llm_harness_behavior.md`（运行时行为）与
`docs/provider-adapters.md`（自定义 endpoint 的供应商差异）。

---

## 1. 一次调用是怎么定下来的

一次 run 会运行多类 LLM 会话（背景调查、每窗查询轮、纠错窗、知识更新等）。选模型分四步，
每一步只回答一个问题：

```text
会话(harness 决定)
  └─→ 任务组          "这是哪一类活"      7 个,固定
        └─→ 格子       任务组 × difficulty  21 格,携带 prompt 变体
              └─→ 模型组  预设把格子绑到它    有序列表,可以自行组合
                    └─→ 逐候选过滤 → 真正应答的那个
```

- **任务组**由 harness 定义，不能新增：`correction-mm` / `correction-text` /
  `planning-mm` / `planning-text` / `research` / `search_judge` / `knowledge`。
  带 `-mm` / `-text` 的两对由两个媒体开关直接选中（见 §2）。
- **预设**是一张「格子 → 模型组」的绑定表。随包发布一个 `default`；可以在 `config.toml` 中编写
  自己的预设，只需覆盖需要调整的几格。
- **模型组**是一个**有序**的模型列表。前面的先试，失败（额度/限流/超时/不可用/瞬时错误）
  才退到下一个。顺序就是优先级，没有打分选模、没有随机负载均衡。
- **逐候选过滤**发生在发车前：provider 是否启用、当日额度是否已打满、模型能否处理本次携带的
  音视频、能否联网搜索、prompt 估算长度与请求输出上限是否超过其自身限额。过滤掉就试下一个；
  **全部被过滤时报错**，不会静默降级。

> 新增一行模型事实**不会自动上线**。必须显式进入模型组、再被某个预设绑定，才会被调用。

## 2. 开关与旋钮

命令行开关在命令行上以 `--llm-` 开头（下表写的是全名）,`finesub` 与
`python -m finesub.pipeline` 一致：

| 开关 | 取值 | 作用 |
| --- | --- | --- |
| `--llm-media` | text / audio / video | 便捷写法，一次设定下面两个 |
| `--llm-correction-media` | text / audio / video | **纠错窗**吃什么；同时选中 `correction-mm` 或 `correction-text` |
| `--llm-planning-media` | text / audio / video | **每窗查询轮**吃什么；同时选中 `planning-mm` 或 `planning-text` |
| `--llm-retrieval` | none / local / native | `local` = harness 自行检索（背景调查 + 每窗查询轮 + 本地搜索代理）;`native` = 模型自带联网搜索（检索质量更好，推荐优先使用，见下）;`none` = 不检索 |
| `--llm-difficulty` | quality / intermediate / efficiency | 见下 |
| `--llm-continuity` | serial / parallel | 窗口之间是否串联上下文；`parallel` 放弃串联换并发 |
| `--knowledge` | none / collect / update | 知识库读写档位，独立于上述各轴；它是少数**不带** `--llm-` 前缀的选项 |

### 走代理：`[llm] proxy`

所有 LLM 层的 API 请求（Gemini REST、OpenAI 兼容 / Anthropic 端点、媒体上传、
Exa / Gemma4 / Tavily 检索、免费的 `countTokens`）共用一条代理设置：

```toml
[llm]
proxy = "http://127.0.0.1:7890"   # 或 socks5://...
```

三种状态：

| 写法 | 含义 |
| --- | --- |
| 不写这一行 | 沿用环境变量 `HTTPS_PROXY` / `HTTP_PROXY`(0.5.0 之前就是这个行为) |
| `proxy = "<url>"` | 走这个代理，**并忽略环境变量**——文件比环境更具体，不做合并 |
| `proxy = "direct"` | 不走代理，**并忽略环境变量**。给「整机挂着代理，但 LLM 调用要直连」用 |

**范围仅限 API 调用。** 下载（模型、ffmpeg、yt-dlp、媒体源）有自己的一套路由发现
（系统代理 + 环境变量 + 本地端口探活），不受这一项影响。

> ⚠️ **开启代理不仅改变线路，也会改变失败的表现。** Gemini 会对机房 IP 与共享出口 IP 评分，
> 判定可疑时直接拒答。因此「检索正常、纠错却持续失败」是该设置下的一种正常现象，并非 bug——
> 可先更换出口，再排查其他问题。

> 本机 agent(Codex / Claude Code / agy)**不受这一项影响**：它们是各自登录、自己发请求的
> 外部程序，harness 拉起它们时不传递任何代理变量。这是有意的——它们有自己的网络配置与
> 风控，替它们改线路可能让发车失败被误判成额度耗尽。**要给它们配代理，按各自 CLI 的文档
> 独立设置即可**，那条路完全通，不需要 finesub 代劳。

### 检索开关：优先 `native`

**模型支持联网搜索时，优先 `--llm-retrieval native`。** 它的检索质量明显好过 harness
自行检索：同一素材、同一模型各取 3 个样本对照，native 认出的背景实体并集是 local 的 1.6 倍
(33 vs 21)，专名也更准——一个角色的全名 local 给错、native 给对。差别在时序：`local` 的
搜索词要在见到内容之前一次性定好，`native` 却能边读边追问。

代价与注意：

- **需要能联网的模型**，见 §4 末尾那条。纯免费 Gemini 档不行。
- **`native` 下出处能否核对，取决于后端类型。** agy 这类后端只回报搜索词、不回报来源 URL，模型
  并不掌握真实 URL。实测中，模型会编造形似真实来源的链接填入 `sources`(YouTube video id 被写成
  `genshin_last_legacy`、`4gamer.net/games/000/G000000`)，且每次编造的内容都不同——此类后端的
  `sources` 会被自动标记为未核实。Claude Code / Codex 这类会回报工具事件的后端则不同：harness
  从事件中收取每次搜索的真实 URL，产物中有完整记录。
- `local` 的出处始终可查：harness 亲自执行检索，URL 直接写入产物。
- 如何选择：**追求覆盖面使用 `native`，需要控制预算与 provider 时使用 `local`**（每次调用的查询
  数、抓取数与墙钟均由 runtime 台账强制；`native` 只能事后记账，无法预先限制搜索次数）。

### 两个媒体开关

这两个开关决定的是「**哪些任务使用媒体**」，而非「本次是否携带媒体」——后者由是否传入
`--audio` / `--llm-video` 文件决定。若开关要求的媒体类型多于实际提供的文件，属于**配置错误**，
会直接报错，不会静默降级。

常见组合：

- `--llm-media video`：查询轮和纠错窗共用同一份视频剪辑，只切一次、只传一次。
- `--llm-correction-media text --llm-planning-media video`：纠错交给纯文本强模型（比如自接的
  DeepSeek/Claude），查询轮仍然看视频。**这是让纯文本模型处理带音视频素材的正规路径**。
- `--llm-media text`：完全不切片、不上传。

> 查询轮读视频比读音频贵：视频在低分辨率下约 `71 tokens/帧 × 0.25 fps ≈ 17.75 tok/s`,
> 叠在音频约 32 tok/s 上，媒体 token 约 **+55%**。换来的是与纠错窗共用一份剪辑、少一次上传。
> 是否值得由预设决定，harness 不替用户判断。

### difficulty 只做两件事

选择**该格的 prompt 变体**和**该格的思考档位**。它不再影响窗口如何切分，也不从应答模型反推
prompt 档位。

三档按**预期目标**命名，而非按输入难度：`quality`（要质量）/ `intermediate`（折中）/
`efficiency`（要省）。

**预设可以按档位绑定不同的模型组**，出厂就是这么配的：纠错的 `quality` 绑 capable 组、
`intermediate` 绑两个 lite。因此：

- 免费档 capable 的日额度(20 RPD)耗尽时，run 会**停下**而不是偷偷换成 lite;
- 两条出路：等额度恢复后 resume，或显式 `--llm-difficulty intermediate` 继续。
- 切档**不会丢失已完成窗口**：窗口指纹不含 difficulty，窗口计划已落盘。产物中会出现前后不同
  变体的混合，这是显式发起的，逐窗都有记录。

`efficiency` 还会把纠错与查询轮的媒体都钉为 `text`、检索钉为 `none`，并要求
**`--knowledge none`**（不读索引、不注入词条、不做任务后更新）。若发生冲突会直接报错，
不会静默降级。

### 思考旋钮

抽象档位 low / medium / high，写在预设上，按 `"任务组/难度"` 索引：

```toml
[llm.presets.my.thinking]
"correction-text/quality" = "high"
```

未填的格子会先在同一预设内向上查找更高档位(efficiency→intermediate→quality)，整组未填才回落到
default 预设，最终回退为 **medium**。

回落方向**朝上**，所以想让某格保持 medium 而它上一格是 high 时，必须显式写出来。

**出厂实况**（21 格全解析，格式为 `思考档 / 模型组 / prompt 变体`）:

| 任务组 | quality | intermediate | efficiency |
| --- | --- | --- | --- |
| correction-mm | medium / correction-capable / capableC | medium / correction-basic / basicB | **low** / correction-basic / basicB |
| correction-text | medium / correction-capable / capableC | medium / correction-basic / basicB | **low** / correction-basic / basicB |
| planning-mm | medium / lightweight-default / 单模板 | medium / 同左 | **low** / 同左 |
| planning-text | medium / lightweight-default / 单模板 | medium / 同左 | **low** / 同左 |
| search_judge | medium / lightweight-default / 单模板 | medium / 同左 | **low** / 同左 |
| research | **high** / research-default / 单模板 | **high** / 同左 | **low** / 同左 |
| knowledge | **high** / knowledge-capable / 单模板 | **high** / 同左 | **low** / 同左 |

两条规律：**efficiency 一律 low**（最省的形态）;**research 与 knowledge 到 intermediate 都是
high**。它们的产物被下游全量复用（research 写的背景包每个窗口都读，knowledge 直接写库），换档
换的是模型，不应顺带砍掉推理。intermediate 那一格是**继承**来的，不是写死的：回落朝上。

抽象档位到供应商参数的换算见 §3 的 `thinking` 列。

### 会话档位（只对 agent 生效）

一格绑到本机 agent(Codex / Claude Code / Antigravity)时，`agent_session` 决定**同一条 run
内这些调用如何共用会话**。与 thinking 一样写在预设上、按 `"任务组/难度"` 索引，未填则继承
`per-window`:

```toml
[llm.presets.my.agent_session]
"correction-text/quality" = "pseudo-conversational"
```

| 档位 | 一次调用是什么 | 什么时候选它 |
| --- | --- | --- |
| `api` | 每次调用一个全新会话，整包 prompt 重发 | 想要每窗完全独立、可复现 |
| `per-window`（默认） | 一个窗口连同它的修复轮共用一次会话 | 出厂默认，不必动 |
| `resume` | 整条 run 一条会话，跨窗口续上下文 | 想让模型记住前面窗口 |
| `pseudo-conversational` | 整条 run **一个常驻 CLI**：它做完一个窗口就向 FineSub 要下一个 | 想吃满会话缓存、少付冷启动 |

三点要知道：

- `pseudo-conversational` **要求 CLI 支持 per-invocation MCP server**（三家均支持）。探测不到时
  直接报错停止，不会静默切换档位，因为选择该档即表示要求此形态。
- 带音视频的调用**任何档位下都单独走一次会话**，不进常驻 CLI（工具协议是纯文本的）。
- 该档位下 token 用量按**会话**记账（CLI 一次调用只报一次总账），所以任务报告里 agent 那行的
  「调用次数」按窗口算、token 按会话算，两列本就不是一回事。

## 3. 模型事实表(catalog)

**事实进 catalog，编排进 config.toml。** 一行一个**可调用的 （provider， 模型）**：能做什么、
限额多少、走哪个端点、思考参数如何表达。同一模型在免费档和付费档是**两行**，因为供应商在不同
档位上可能真的阉割能力和额度，这里不去重。

**两份文件，同一格式**：

| 文件 | 位置 | 作用 |
| --- | --- | --- |
| 默认 catalog | 随代码发布(`src/finesub/llm/routing/model_catalog.psv`) | 维护者实测的事实 |
| 你的 catalog | 数据根目录下的 `model_catalog.psv`，与 `config.toml` 同级 | 覆盖同名 `fact_id`、追加新 id。⚠ **覆盖是整行替换**：没写的列退回默认值，不保留出厂值 |

数据根即 `.env` / `config.toml` 所在目录：装好的 CLI 是 `user-data`，仓库版没有单独的
user-data，即 checkout 根。你的行会标为 `self_reported`，产物中与实测事实区分开。

> **贯穿两个文件的规则：同名时，本地声明覆盖打包内容。** catalog 中同 `fact_id` 覆盖对应行，
> `config.toml` 中同名的模型组 / 预设覆盖打包的对应项（包括 `default` 预设本身）。代价是拼错的
> id 不会被拦截，因此每次覆盖都会在启动时打印一行 `Note: config.toml 覆盖了打包声明的 …`;
> 若出现不认识的名字，即为拼写错误。
>
> `config.toml` 本身**不**跨目录合并：按「checkout 根 → user-data」取第一个存在的文件。

**表头声明有哪几列**，覆盖文件只写需要覆盖的列即可；列留空取默认值：

| 列 | 必填 | 留空时 |
| --- | --- | --- |
| `fact_id` | ✅ | — 这行的名字，模型组按它引用 |
| `provider_tier` | ✅ | — 与 `.env` 条目名一致（`GEMINI_FREE` / `GEMINI_PAID` / 自己的 provider id）。本地 agent 用 `LOCAL_CODEX` / `LOCAL_CLAUDE` / `LOCAL_AGY` / `LOCAL_DSH` / `LOCAL_WORKBUDDY`：它们不读 key，而是**决定用哪个 CLI driver**(`LOCAL_DSH` 的 `api_model_id` 要写成 `<dsh provider>/<model>`，例如 `deepseek-official/deepseek-v4-flash`，因为 dsh 从插件配置选模型而不是命令行；`LOCAL_WORKBUDDY` 写模型 id 本身，取值以服务端认的那张清单为准，且**每行都要写自己的 `quota_pool`**——那个账号是按模型线分别计额度的，见 [`agent.md`](agent.md) 4.2) |
| `api_model_id` | ✅ | — 发给供应商的真实模型名 |
| `max_input_tokens` | ✅ | — **这个 API 能接受的最大输入**，没有乐观默认。它和 `context_window` 一起决定窗口怎么切，见下一行 |
| `provider_kind` | | 打包 tier 按其方言推断，否则 `openai_compat`。取 `gemini` / `local_agent` / `openai_compat` / `anthropic` |
| `base_url` | 文本方言必填 | 两个文本方言必须给；`gemini`/`local_agent` 不能给（传输自带端点） |
| `key_env` | | `FINESUB_KEY_<PROVIDER_TIER>`;key 本身放 `.env` |
| `display_name` | | 取 `api_model_id`。给人看的显示名 |
| `max_output_tokens` | | 65536 |
| `context_window` | | **留空 = 这一行不声明联合约束**，存成两者之和，于是它恒不起作用。Gemini 分别公布输入与输出两个上限，所以**免费档留空**；**付费档与 agy 仍然填了**（`ctx = max_input = 1048576`，包络 983040）——该决定基于按单池规划的考虑，而非假设两个公布值能够同时用尽。**单池供应商必须填**（Codex、Anthropic：答案和提示词花的是同一份预算）。窗口的输入包络取 `min(max_input_tokens, context_window − 输出上限)`，因此填写该列后，能放入一窗的输入会比 `max_input_tokens` 略小，差额正是预留给答案的空间。⚠ 若填写值小于 `max_input_tokens` 或 `max_output_tokens`，会**直接报错**，不会被静默截断 |
| `supports_audio` / `supports_video` / `supports_native_search` | | `false`。能力位，用于逐候选过滤。两个自定义 HTTP 文本方言(`openai_compat`/`anthropic`)**不允许**声明音视频；打包的 Gemini REST 与本地 Agy target 可以声明媒体能力 |
| `video_high_resolution_only` | | `false`。为 `true` 时，该模型参与的媒体格强制按高分辨率视频费率计帧（见 `docs/llm_local_agent_agy.md` §3）；出厂仅 agy 的 Gemini 行开启 |
| `thinking` | | `true`（恒等映射），见下 |
| `token_scale` | | 1.0，见下 |
| `rpm` / `tpm` / `rpd` / `tpd` | | 100 / 4M / 无限 / 无限；`tpm`/`tpd` 只算输入 token,`-1` = 无限 |
| `is_free` | | `false` |
| `hint_output_ceiling` | | `false`。开关：告诉 agent「你每轮输出上限是本行的 `max_output_tokens`，快到时先调一次工具再继续」。**只对被实测证明受益的模型开**——说错的数比不说更糟，所以数字永远取本行的 `max_output_tokens`，不另填。出厂只有 `local-workbuddy-glm-5_3-flash` 开着 |
| `fallback_model` | | 空。**这一列会花钱**：填了之后，本行的模型答不了时（额度耗尽，或重试一次仍然过载）由 CLI 自己把这次 session 交给这里写的模型跑完，并在结束时打一条 warning。只对有免费/付费孪生的行才有意义；出厂只有 `local-workbuddy-hy3`（→ `hy3-x`）与 `local-workbuddy-hy4`（→ `hy4-preview-x`）写了。留空 = 答不了就报错，这是每一行的默认 |
| `quality_score` | | 0–100，**纯咨询性**：只用来生成告警，绝不参与任何路由决策。**留空 = 没有判断**，按 100 算（不触发任何质量下限），并在日志里记一条 note 说明这件事——留空不是「判断为差」，想让下限对它生效就填个数 |
| — | | **规划包络与请求上限是两个数**（2026-09-04）：包络按「预计输出」从 `context_window` 里扣（纠错窗是 `output_scale × 输出系数 × 每窗字幕上限`，非纠错轮是 `output_scale × 32000`），而每次调用**仍然按答题模型自己的 `max_output_tokens` 打满地请求**。预留按估计放开输入，请求按上限保住答案余量 |
| `quota_pool` | | 取 `provider_tier`。**只对本地 agent 有意义**：一个订阅额度用完时，同池的 target 会一起被停用（见 [`agent.md`](agent.md) §6）。写入该列才启用分组；出厂由 Antigravity(`AGY_GEMINI` / `AGY_ANTHROPIC`)与 WorkBuddy(每行一个，实测该账号按模型线分别计额度)配置了，因为其单个 CLI 背后是两份独立计量的额度 |

### 窗口太小的模型会被拦下

绑定进模型组的成员，如果**最大输入 < 192,000** 或**最大输出 < 64,000**，启动时会告警一次
（照跑，但窗口会被切得更碎，纠错质量和合并判断都受影响）；低于 **96,000 / 32,000** 会**直接
停下来**，且发生在识别开始之前——避免在跑完数十分钟 ASR 后才暴露配置问题。

⚠ 比的是 `max_input_tokens` 与 `max_output_tokens` **这两列本身**，不是扣掉输出之后的规划包络。
所以 Claude Haiku 4.5(200000 进 / 64000 出，上下文窗口 200000)是**放行**的：它两边的量都够，
只是总量有限，而"总量有限"是正常模型的形态，不是配置错误。

只检查**当前 preset 用得到**的组；没被任何组引用的行（例如只做联网检索的 `gemma-4-31b`）不看。

覆盖是**整行替换**而不是打补丁：覆盖一行打包模型时，未写的列取**默认值**，不是打包行的原值。
只想改一个限额时，把要保留的列一并写上。

### `token_scale` 列

本地那套三级 token 计数器用的是 Gemini 词表，别家模型分词不同，估算会有系统偏差。
`token_scale` 就是这一行的修正系数，乘在**本地估算**上，只影响两件事：窗口能否装下的硬检查，
以及限流的 TPM 预留。

**报告和成本核算永远以 API 返回值为准**，不用估算值，也不乘这个系数。task report 里有
「Token Estimate Calibration」一节，按 fact 给出「返回 input tokens / 本地估算」的中位数与
样本数；样本够多且偏离当前系数够远时给一句建议值。**不会自动写回**：这个系数会改变**后续**
窗口预算，而且样本本身有偏（密集 CSV 与自然语言背景包差很多）。已落盘的窗口计划续跑时继续
定位已完成窗口；未完成窗口若不再合身，会在发车前自动拆小。新任务直接按新系数规划。

它的危险方向是**调小**：估算是安全边界。因此窗口规划取组内**最大**的系数、且不低于 1.0。
调大会让窗口相应变小（规划与发车两端口径一致），调小则不会放宽任何边界。

### `thinking` 列

| 取值 | 含义 |
| --- | --- |
| `true` | **恒等映射**（默认）：抽象档位原样发出 |
| `false` | 这个模型完全不发思考参数（**包括全局覆盖也不发**） |
| `高映射,中映射,低映射` | 三个供应商取值，覆盖恒等 |

恒等是合理默认而非巧合，三家的档位词本来就同名：

| 抽象档位 | Gemini `thinkingConfig.thinkingLevel` | OpenAI 兼容 `reasoning_effort` | Anthropic `output_config.effort` |
| --- | --- | --- | --- |
| high | `high` | `high` | `high` |
| medium | `medium` | `medium` | `medium` |
| low | `low` | `low` | `low` |

只有当你的端点用别的词（例如 `max,default,minimal`）才需要写显式映射；自建端点对未知字段直接
400 时写 `false`。**第二种用法是给爱想的模型封顶**：出厂的 3.8 Flash 三行写的是
`medium,medium,low`——它在同一个抽象档位上比 3.7 多想约 1.5 倍，而思考与字幕正文共用同一份
输出上限，所以 high 被压回 medium;medium 那一格**故意保持 medium 不再往下**，因为实测把它降到
`low` 之后，五个纠错窗里有四个一个思考 token 都不出——`low` 是把要不要想交给模型，不是少想一点。
映射同样用于本地 headless agent target:Codex 走 `model_reasoning_effort`,
Claude Code 走 `--effort`,dsh 写进它模型插件的 `reasoningEffort`。DeepSeek 那个插件只有
off/low/high/max，没有 `medium`，所以 FineSub 会把 `medium` 译成 `high`、`xhigh` 译成 `max`。
产物里记录的是**映射后实际发出**的值。`[llm].local_agent_reasoning_effort` 默认留空，只有
显式非空时才全局覆盖每个模型自己的映射；但**覆盖不了 `thinking = false`**：那不是「用默认档」,
而是「这个模型根本不收这个参数」，硬塞会在发车前被 CLI 拒绝，而那种硬失败会被额度探测误判成
订阅耗尽。

## 4. 出厂预设长什么样

| 任务组 | 下限 | quality 格参考模型 | quality 绑定的模型组 | 组成员（有序） |
| --- | ---: | --- | --- | --- |
| correction-mm | 70 | 3.7 Flash | `correction-capable` | 免费 3.7 Flash(75) → 免费 3.8 Flash(75) → 免费 3.6 Flash(75) → 免费 3.5 Flash(70) → 付费 3.7 Flash(75) → 付费 3.8 Flash(75) |
| correction-text | 70 | 3.7 Flash | `correction-capable` | 同上 |
| planning-mm | 50 | 3.5 Flash Lite | `lightweight-default` | 免费 3.5 Lite(60) → 付费 3.5 Lite(60) |
| planning-text | 50 | 3.5 Flash Lite | `lightweight-default` | 同上 |
| research | 50 | 3.6 Flash | `research-default` | 免费 3.6 Flash(75) → 免费 3.5 Flash(70) → 免费 3.7 Flash(75) → 免费 3.8 Flash(75) → 付费 3.7 Flash(75) → 付费 3.8 Flash(75) |
| search_judge | 50 | 3.5 Flash Lite | `lightweight-default` | 同 planning |
| knowledge | 70 | 3.6 Flash | `knowledge-capable` | 同 research |

「参考模型」是**派生**的，且**按格**：该格所绑模型组的第一个成员。所以纠错的 `intermediate` /
`efficiency` 格绑 `correction-basic`（免费 3.5 Lite → 付费 3.5 Lite），参考模型就是
**3.5 Flash Lite**，而非 quality 格的 3.7 Flash。参考模型仅用于展示，换代时只需修改一处真值。
下限告警本身只查 quality 格（另外两档本来就是声明过的降档）。

其余任务组只绑 quality，低档自动向上继承。

### 另外两个出厂预设：`agy-hybrid` 与 `agy`

两个都把 Antigravity 排进模型组，是 agent 模式的入口（详见 [`agent.md`](agent.md)）;
差别只有一处——**免费 API 在不在名单里**。

`[llm] preset = "agy-hybrid"`：免费 Gemini 优先，agy 作为后备；免费额度用尽后才消耗订阅。

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-capable` | 免费 3.7 Flash → 免费 3.8 Flash → agy Opus 4.6 → agy Gemini 3.7 Flash | 纠错的 quality + intermediate、knowledge/quality |
| `agy-basic` | 免费 3.5 Flash → agy Opus 4.6 → agy Gemini 3.7 Flash | research、planning、search_judge 的 quality |

`[llm] preset = "agy"`：同样的格子，但名单中**没有任何 API 成员**。适用于已付费订阅、不愿再
管理 API key 的情况。

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-only-capable` | agy Opus 4.6 → agy Gemini 3.7 Flash | 纠错的 quality + intermediate、knowledge/quality |
| `agy-only-basic` | agy Opus 4.6 → agy Gemini 3.7 Flash | research、planning、search_judge 的 quality |

> **与 `execution_policy = "agent-only"` 有何区别？** 两条路径可能产生相同的调用，但语义不同：
> policy 是**全局后端闸门**，只能禁用一个后端、不能新增成员，且对**任何**激活的预设生效；预设
> 描述的是「这份名单中有哪些成员」。需要临时将某次运行限制为纯 agent 时使用 policy；需要长期按
> 订阅运行时使用 `agy` 预设。两者叠加使用也没有问题。
>
> ⚠ `agy` 预设的 `test_target` 是 agy 自身的 Gemini 前端，**不是**默认预设使用的免费
> Gemini Lite:`--test-profile` 会将整轮运行中的每一次调用都固定到 test target 上；若沿用默认
> 值，相当于让「纯 agy」运行的测试全部打到 API 上。

> Opus 4.6 一行的思考档位**不可调**：agy 将档位编码在模型名中，仅其 Gemini 系列区分
> high/medium/low。因此该行 `thinking` 填写 `false`,`[llm].local_agent_reasoning_effort`
> 对它也不生效（携带该参数时 agy 会直接拒绝请求）。

### dsh 的两个组：`dsh-capable` / `dsh-basic`

**没有预设会自动绑它们**，要用就在 `[llm.bindings]` 里显式绑：

| 模型组 | 成员（有序） |
| --- | --- |
| `dsh-capable` | DeepSeek-V4-Pro → V4-Flash → V4-Pro（带搜索） |
| `dsh-basic` | DeepSeek-V4-Flash → V4-Pro（带搜索） |

> **只能绑定到按窗口执行的格子。** dsh 仅从命令行接收任务，整窗字幕无法放入命令行，因此它只支持
> 工具会话；绑定到 `api` 或 `resume` 档位会直接报错。同时它不支持音视频，也无法报告 token 用量。
> 安装方式、key 位置及「搜索使用另一把 key」的说明见 [`agent.md`](agent.md) 4.1。

**接入自己的 endpoint**：出厂仅包含 dsh 自带的 `deepseek-official` 路由，其他 provider 只存在于
**你自己的** `~/.dsh/settings.yaml` 中，随包发布对其他人没有意义。因此，自建网关通过 override
catalog 接入：在数据根目录（源码 checkout 即仓库根）的 `model_catalog.psv` 中写一行，该行会
**自动**成为可绑定的 target，无需修改其他文件。

```text
fact_id|provider_tier|api_model_id|max_input_tokens|supports_native_search
my-dsh-gateway|LOCAL_DSH|my-provider/my-model|194000|false
```

`api_model_id` 的 `my-provider` 必须是在 `~/.dsh/settings.yaml` 的 `llm-pi-ai.providers` 下
声明过的名字。`supports_native_search` 决定这一行拿到带搜索还是不带搜索的那档，一行只能得一个
target；两档都要等于两个模型，得写两行不同的 `api_model_id`。

**接入自己的网关时，请将 `thinking` 设为 `false`**，随后在 `settings.yaml` 中为对应 provider 配置
`reasoning`。原因是 dsh 的配置覆盖采用整键替换：若 FineSub 代为写入该项，会把该 provider 的
`baseURL` 与 key 一并覆盖，因此 FineSub 不修改此项，档位设置留给你自行完成。

免费 API 在前是有意的，两边的额度都不浪费。Opus 4.6 是**纯文本**，排在媒体成员之前同样有意：
文字任务优先给它，带剪辑的任务在能力过滤时自动跳过它、落到 agy 前置的那个 Gemini。

前面几张表没写全的一点：每组末尾都还有一个 `local-agy-native-gemini-3_7-flash`——同一个
模型，只是额外授权了 agy 自带的搜索工具，**只有 `--llm-retrieval native` 会选中它**（普通
调用在它前面就已经落到未授权那一档）。注意 agy 只报搜索词、不报来源 URL,Codex/Claude
那两档才有完整出处。

两条要事先知道的代价：

- **免费档的知识更新为尽力而为(best-effort)。** `knowledge` 与 `research` 均以免费 capable 模型
  优先，而免费档仅有 **20 RPD**。带调查的 run 会先消耗 research 配额，之后再分 chunk 执行知识
  更新，免费档很容易在知识更新阶段耗尽配额，且**没有 lite 后备**。这是「不静默降质」的代价，并非
  bug。
- **`--llm-retrieval native` 需要具备联网能力的模型。** 出厂由付费 3.7 / 3.8 Flash 提供联网；免费
  3.x 完全无法联网，纯免费档运行 native 会在启动校验时报错。此时可开启付费档，或显式绑定打包的
  `gemini-native-search` 组（含免费 2.5 Flash，但其 50 分低于纠错/知识的下限 70）。

## 5. 接自己的模型

两个文件，两件事。完整样例见 `config.example.toml`。

> **配置正确但仍出现异常时，原因可能在另一半。** 本节说明*需要配置什么*；各家 API 的实际差异
> (429 表示限流还是余额不足、思考深度参数的名称、拒答的表现形式、reasoning token 的计量方式)
> 已在 [`docs/provider-adapters.md`](../provider-adapters.md) 的调研表中逐家列出，并标注了哪些已
> 被实现处理。接入非 OpenAI 兼容端点前，建议先浏览该表。

**① 事实**——在数据根目录建 `model_catalog.psv`（与 `config.toml` 同级）。**新增**一行时
只需要写你关心的列（其余取默认值）；**覆盖出厂的某一行**时要把整行抄过去再改，因为同名
`fact_id` 是整行替换、不是按列合并：

```text
fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|max_output_tokens|quality_score
ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|32768|60
```

这一行同时做了三件事：声明 `deepseek` 这个 provider（方言 + 端点）、成为一条 `self_reported`
事实、并自动得到一个同名 target 供模型组引用。

**② API key**——不写在上面任何一处：按 `key_env`（缺省 `FINESUB_KEY_DEEPSEEK`）放进 `.env`,
走和其它 key 完全相同的加密路径（见 [`env.md`](env.md)）。

**③ 编排**——在 `config.toml` 里组队、绑格子。全是**具名表**（没有 `[[数组表]]`），所以设置
面板的写入器和手动编辑可以共存。用打包已有的名字就是覆盖它（例如重写
`[llm.model_groups.lightweight-default]` 换掉出厂的轻量组，或整个重写
`[llm.presets.default]`）；覆盖会在启动时列出来：

```toml
[llm]
preset = "my-deepseek"          # 选激活预设

[llm.model_groups.my-corr]      # 有序模型组,先试前者
targets = ["ds-flash", "gemini-paid-3_7-flash"]

[llm.presets.my-deepseek]
name = "DeepSeek 纠错"
[llm.presets.my-deepseek.bindings]
"correction-text/quality" = "my-corr"
```

未绑定的格子会先在你的预设内向上回落到更高难度，整组未绑才回落到 default 预设；因此只需编写
关心的几格。`test_target` 未指定时继承 default 预设的取值。

### 快速选模型：格子可以直接绑一个模型

当需要指定「这一格固定使用某个模型、不做回退」时，无需先创建只含一个成员的模型组，直接将格子
绑定到 **target 名**即可：

```toml
[llm.presets.my-deepseek.bindings]
"correction-text/quality" = "my-corr"        # 一个模型组
"research/quality"        = "ds-flash"       # 直接一个模型
"knowledge/quality"       = "local-codex-completion-gpt-5_6-luna"
```

FineSub 会自动把它包成一个单成员模型组（内部名 `target:<名字>`）。这也是使用 Codex / Claude
Code / WorkBuddy 的办法：它们的 target 都已声明好，出厂预设只是没绑。可用的 target 名见
`model_routes.toml` 的 `[targets.*]` 段（随代码发布，与 catalog 同目录）:`local-codex-*` /
`local-claude-*` 各有两个变体，`completion` 与 `native` 的区别在于是否允许模型使用自己的搜索
工具，后者走带 `web-search` 的 execution profile。`local-workbuddy-*` 与 `local-dsh-*` 没有
`completion` 中缀（`local-workbuddy-hy3` / `local-workbuddy-native-hy3`），因为这两个
tier 会给用户自己加的 catalog 行**自动生成同形的 target**。

**同名时模型组优先**，因此即使 catalog 中新增了同名模型行，也不会在不知情的情况下改变你的绑定。

### 优先模型：一行说「就用它」，不改预设

当希望指定某个模型、但不想为此编写整套预设时，可使用该表。取值可以是**模型组**或**单个 target**:

```toml
[llm.preferred_targets]
default  = "my-corr"          # 所有格子换绑到 my-corr 组
research = "ds-flash"         # 这个任务组钉单模型(按任务组的写法覆盖 default)
```

语义为**整体换绑，不保留回退链**：被固定的模型组/模型无法处理的调用（例如纯文本 target 遇到带
音视频的窗口，或要求联网检索的任务）会**明确失败**，而不是静默切换到其他模型——既然指定了该
模型，失败也应清晰可见。若因此导致某个 `-mm` 格子无可用模型，启动时会给出警告。

- 值先查模型组、再查 target；同名同时存在时报二义性错误。任务组名必须是真的
  (`correction-mm` / `correction-text` / `planning-mm` / `planning-text` /
  `research` / `search_judge` / `knowledge`)，写错直接报错并列出可选值。
- **对激活预设与 default 预设均生效**——未绑定的格子会回落到 default 预设，若优先模型恰好在这些
  格子上无法使用，则属于较难发现的问题。
- 修改该表会使未完成进度失效（见 §7），因为「由谁应答」本身就是路由身份的一部分。

**不改文件的运行时版本**：`--llm-model [任务组=]值`（可重复；裸值 = default）,
`finesub.pipeline` / `finesub.llm.correction_translation` 与
`finesub.llm.knowledge` 的 `repair`/`verify` 都收。语义与上表相同，只对本次进程生效，
config.toml 不动；优先级按 `命令行[任务组] > 命令行[default] > 配置[任务组] > 配置[default]`
逐格解析——裸写 `--llm-model X` 会盖过配置里的按任务组指定（这一次就是全用 X）,
配置里没被命令行碰到的任务组照旧生效。

与上一节的区别：**绑定 target 名**表示「这一格仅使用它，不做回退」；**优先模型**表示「让它排在
首位，回退仍然保留」。

### 第一版的三条硬限制

1. **自定义 endpoint 只做纯文本。** 媒体由打包的 Gemini REST 或本地 Agy target 承担。想让纯
   文本强模型处理带音视频的素材，用 `--llm-correction-media text`（查询轮照常吃剪辑）。绑错不会
   静默丢音频，带媒体的调用会在能力过滤时跳过它。
2. **日封禁机制不适用于非 Gemini 服务。** 日额度封禁依赖 Gemini 的结构化 `quotaId` 判定；其他
   服务返回的 429 默认按限流处理，达到上限后会持续退避重试（若错误文本明确指向日配额，仍归入日
   封禁）。DeepSeek 的 402 余额不足与 OpenAI 的 `insufficient_quota` 为例外：二者归入 quota 类，
   直接推进到组内下一个候选。
3. **第三方模型进 `knowledge` 组要谨慎。** 知识更新的产物会自动 apply 进知识库，未标定 /
   provenance 不可审计的 target **不得作为知识更新的唯一证据来源**。

## 6. 启动时那几条告警怎么读

绑定期告警按**当前激活的预设**打印（出厂预设零告警，有测试保护）:

| 告警 | 触发 | 该怎么办 |
| --- | --- | --- |
| `成员 X 的 quality_score=N 低于下限 M` | quality 格的组里有成员低于任务组下限 | 确认是有意放的。另外两档本身就是声明过的降档，不告警 |
| `成员 X 未声明 quality_score，按 100 计` | 这是 **note 不是告警**（只进日志，`--verbose` 才上屏）| 不必处理。想让质量下限对它生效，就在 catalog 行里填个分 |
| `规划包络由 X 决定 (…tokens…)，窗口数约 ×N` | 组内最小的输入包络/输出上限低于免费 Gemini 基线 | 窗口按**组内最低**规划（谁应答还没定，必须保守）。×N 是窗口数放大倍数。⚠ 2026-09 起**反过来也成立了**：harness 那道 194,000 的硬上限已经删掉，组内全是大窗口模型就真的会切出更大的窗。真正还在限制每窗字幕量的是质量护栏 `[chunking] max_window_subtitle_tokens`（默认 10000），把它设成 0 会连这道也去掉——那时窗口大小随模型的输出上限走，进入未标定区（`docs/llm_followups.md` 的 P6） |
| `组内没有成员支持音频` | `-mm` 格里一个能听的都没有 | 带媒体的调用会在能力过滤后无候选可用；换模型，或把该任务改成 `text` |
| `N 个成员里只有 1 个支持音频` | 该格能服务媒体调用，但只有一个候选 | 不是错误：纯文本成员排在前面是刻意的（文字任务优先给它）。但媒体调用的链长为 1，那一个失败就整轮失败，想要冗余就再加一个能听的成员 |
| `target X 不支持视频，按 video->audio 降一级` | 运行期安全网触发 | 正常配置不该看到它。说明该格里混进了看不了视频的模型 |

## 7. 什么会作废已完成的进度

判据并非「配置是否一字不差」，而是**已提交的结果能否在相同来源与同一套窗口边界下继续成立**。
若可以，则继续沿用，新配置仅作用于尚未完成的部分；因此单条任务允许混合使用模型、思考档位与
prompt 变体，逐窗记录会保留实际 `variant` / `difficulty` / `knowledge_version` 备查。

| 改动或状态 | 已完成窗口 | 已完成 research context |
| --- | --- | --- |
| 换 `--model`、预设、**模型组**、thinking、execution policy、agent 参数 | 复用 | 复用 |
| 改 `agent_session` 档位 | 复用 | 复用 |
| 改 `difficulty` 到 `intermediate` | 复用 | 复用 |
| 改 `difficulty` 到 `efficiency` | 复用；pending 窗按新预算预拆 | retrieval 随档位变为 `none` 时重跑 |
| 改 `continuity` | 复用（`parallel→serial` 会告警：advice 台账从空重建） | 复用 |
| 改知识库内容、`--knowledge` | 复用 | 复用 |
| 改 `quality_score`、`floor_score`、显示名、预设名、注释 | 复用 | 复用 |
| 重新下载同一份音视频（大小不变） | 复用 | 复用 |
| 改媒体开关、`--output-scale`、`token_scale`、窗口上限，或换成**上下文更小的模型组** | 复用；pending 窗按当前包络预拆 | 复用（若同时改 retrieval，则重跑） |
| 改 prompt version、任务备注/风格、`--research-search-rounds`、`--test-profile` | ❌ | ❌ |
| `*-stable.json` 内容变化 | ❌ | ❌ |
| 计划/JSON 损坏、schema 不兼容、存档响应过不了当前 parser | 该层重做；被覆盖的 context 先备份为 `*.invalid[.N]` | 同左 |
| `--no-resume` 或删除产物 | 按要求重新执行 | 同左 |

窗口计划作为恢复地址，不会因几何类参数变化而自动更换。恢复时会按当前媒体/profile/模型包络
重新计算 clip 与预算，仅对未完成且不再适配的叶子节点递归拆半；已完成窗口保持不变。拆分不会使
窗口变大：更换为更大模型后，如需合成更少、更大的窗口，可显式删除 artifact 目录中的
`correction-window-plan.json`——这会同时放弃原 chunk id 对窗口缓存的寻址。

`efficiency` 会连带把纠错媒体钉成 `text`、检索钉成 `none`（见 §2）。前者只触发
pending 窗口预算体检；后者改变调查语义，所以已有 research context 会重跑。只想在额度耗尽后
换便宜模型并最大限度复用两层产物，仍优先用 **`intermediate`**：它只换 prompt 变体和思考档位，
窗口与调查都原样保留；`efficiency` 则保留已完成窗口、重做调查并体检 pending 窗。

再往下还有一层更细的 session checkpoint（一次尚未提交的调用）。它要求精确一致，是「这个断点
能不能重建」的结构判断，不是要求整条任务保持同一模型。`agent_session` 档位就在这一层生效：
改它只作废尚未提交的调用断点，不影响已提交的窗口与 research context。

预设**在 run 内不会重读**：启动时快照一次；运行中途修改设置面板不会影响正在进行的 run，将于
下次生效。
