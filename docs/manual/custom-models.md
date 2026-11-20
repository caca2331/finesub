# 接自己的模型

出厂的模型不够用、想接自己的 API 端点（DeepSeek、OpenRouter、公司内部的 OpenAI 兼容服务等），或者想改出厂模型的某项事实（限额、输出上限）时读这一份。只用出厂的模型、只想换预设或挑一个出厂模型，看[模型路由配置](model-routing.md)就够了。本篇沿用那里开头说的几个词：任务组、格子、模型名（target）、模型组。

## 模型事实表（catalog）

**模型本身的事实写进 `model_catalog.toml`（下称 catalog），谁做什么写进 `config.toml`。** catalog 里一个 TOML 表对应一个**可调用的「服务商 × 模型」**：
能做什么、限额多少、走哪个端点、思考参数如何表达、备注。同一模型在免费档和付费档是**两个表**，
因为供应商在不同档位上可能真的阉割能力和额度，这里不去重。

**两份文件，同一格式**：

| 文件 | 位置 | 作用 |
| --- | --- | --- |
| 默认 catalog | 随程序发布，安装版本机看不到；各出厂模型的原值在 [GitHub 上的这份文件](https://github.com/caca2331/finesub/blob/main/src/finesub/llm/routing/model_catalog.toml) | 维护者实测的事实 |
| 你的 catalog | `user-data` 目录下的 `model_catalog.toml`，与 `config.toml` 同级 | 给出厂模型打补丁、追加新模型 |

```toml
# 给出厂模型打补丁：只写要改的键，其余沿用出厂值
[models.local-codebuddy-hy4flash-free]
fallback_model = ""        # "" = 取下表「不写时」那一列的默认（这里是不自动扣费）

# 新模型：写齐必填键
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-flash"
max_input_tokens = 128000
capability = "low"
note = "自己的 DeepSeek key"
```

- **打补丁**：同名的模型只改你写的键。写 `""` 是「取默认值」（下表最后一列），**删掉这个键**才是沿用
  出厂值；`""` 也是去掉一项出厂值的唯一写法（比如让 `capability` 回到「没有判断」）。
- **值带类型**：数写成整数，开关写成 `true` / `false`；写成字符串（`"100"`、`"true"`）会报错。
  `0` 和 `false` 是值，不是「没写」。必填键不能写 `""`。
- 模型名里有 `.` 时要加引号：`[models."my-model-1.5"]`。
- 旧格式的 `model_catalog.psv`（2026-11-09 之前）会被拒绝，并提示转换命令
  `python -m finesub.llm.routing.migrate_catalog`（要源码运行，安装版没有对应命令）。转换按旧文件的意思写出
  `model_catalog.toml`，不覆盖已有文件，旧文件改名留作备份；还会列出你 `config.toml` 里改了名或删掉的模型名。

`user-data` 目录即 `.env` / `config.toml` 所在目录，在哪见[资源与大文件「两类数据，两个去处」](resources.md#两类数据两个去处)。你新加的模型、以及你改了事实的出厂模型在任务产物里记作「自报」（`self_reported`），
与维护者实测的事实区分开；只加了 `note` / `avoid` 的不算。

> **贯穿两个文件的规则：同名时，本地声明覆盖打包内容。** catalog 中同名的模型按键打补丁，
> `config.toml` 中同名的模型组 / 预设整个替换打包的对应项（包括 `default` 预设本身）。代价是拼错的
> id 不会被拦截，因此每次覆盖都会在启动时打印一行 `Note: config.toml 覆盖了打包声明的 …`;
> 若出现不认识的名字，即为拼写错误。
>
> `config.toml` 本身**不**跨目录合并：从源码仓库运行时用仓库根目录那份，否则用 `user-data` 那份，只读一份。

各键的类型与默认值（「不写时」也是写 `""` 时取的值）：

| 键 | 必填 | 不写时 |
| --- | --- | --- |
| 表名 `[models.<名字>]` | ✅ | — 模型的名字，也是可以直接绑进格子、写进模型组的模型名（target） |
| `provider_tier` | ✅ | — 服务名。Gemini 两档是 `GEMINI_FREE` / `GEMINI_PAID`，key 就是 `.env` 里同名的那两个变量；自己接的服务起个名字（如下例的 `deepseek`），它的 key 在 `.env` 里叫什么由 `key_env` 定。本地 agent 用 `LOCAL_CODEX` / `LOCAL_CLAUDE` / `LOCAL_AGY` / `LOCAL_DSH` / `LOCAL_CODEBUDDY`：它们不读 key，而是**决定调用哪个命令行工具**（`LOCAL_DSH` 的 `api_model_id` 要写成 `<dsh provider>/<model>`，例如 `deepseek-official/deepseek-flash`，因为 dsh 从插件配置选模型而不是命令行；`LOCAL_CODEBUDDY` 写模型 id 本身，取值以服务端认的那张清单为准，且**每行都必须写 `quota_pool`**（留空会直接报错，见 [`agent.md`](agent.md) 4.2）） |
| `api_model_id` | ✅ | — 字符串，发给供应商的真实模型名 |
| `max_input_tokens` | ✅ | — 整数，**这个 API 能接受的最大输入**，没有乐观默认。它和 `context_window` 一起决定窗口怎么切，见下一行 |
| `provider_kind` | | 接口格式。出厂的几个 `provider_tier` 自己知道是哪种，其余默认 `openai_compat`。取 `gemini` / `local_agent` / `openai_compat` / `anthropic` |
| `base_url` | `openai_compat` / `anthropic` 必填 | 这两种接口必须给；`gemini` / `local_agent` 不能给（地址是固定的） |
| `key_env` | | `FINESUB_KEY_` 加大写的服务名（`deepseek` → `FINESUB_KEY_DEEPSEEK`）。自己接的服务的 key 在 `.env` 里用这个变量名，写成 `FINESUB_KEY_DEEPSEEK=sk-...` |
| `display_name` | | 取 `api_model_id`。给人看的显示名 |
| `max_output_tokens` | | 65536 |
| `context_window` | | **不写 = 输入与输出是两份独立的上限**（Gemini 的形态）。**输入与输出共用一个上限的服务必须填**（Codex、Anthropic：答案和提示词花的是同一份预算）；填了之后一窗能放的输入会比 `max_input_tokens` 略小，差额留给答案。⚠ 填得比 `max_input_tokens` 或 `max_output_tokens` 小会**直接报错**。出厂各行为什么这样填，见 [`llm_harness_routing.md`](../llm_harness_routing.md)「模型配置、速率限制与显式 reasoning」（开发者参考） |
| `supports_audio` / `supports_video` | | `false`。能不能听音频、看视频；做不了的调用会跳过它，交给组里下一个。`openai_compat` / `anthropic` 两种接口**不允许**声明音视频；只有 Gemini API 与 agy 的行可以 |
| `supports_native_search` | | Gemini 两档、五个本地 agent 与 `anthropic` 接口的端点为 `true`；`openai_compat` 端点写了 `search_request` 才为 `true`；会话 agent 为 `false`。这个模型能不能用它那家的搜索工具；只有 `--llm-retrieval native` 的调用才会真的把工具交给它，见[检索开关](model-routing.md#检索开关优先-native)。出厂只有免费档的 Gemini 3.x 写了 `false`。⚠ dsh 的搜索要绑 DeepSeek 的 API key，每次搜索扣少量费用（[`agent.md`](agent.md) 4.1）。自定义端点见[自己的端点联网搜索](#自己的端点联网搜索) |
| `search_request` | | 空。只给 `openai_compat` 端点：这家 API 开搜索要加的请求字段，写成内联表，`--llm-retrieval native` 的调用才并进请求体，见[自己的端点联网搜索](#自己的端点联网搜索) |
| `video_high_resolution_only` | | `false`。为 `true` 时，该模型参与的媒体格按高分辨率视频的费率算 token（窗口切得更小）；出厂只为 agy 的 Gemini 开 |
| `thinking` | | `true`（high / medium / low 原样发出），见下 |
| `token_scale` | | 1.0，见下 |
| `rpm` / `tpm` / `rpd` / `tpd` | | 限速：每分钟请求数、每分钟 token 数、每天请求数、每天 token 数。本地 agent 与会话 agent 的 `rpm` / `tpm` 不限，其余 100 / 4M；`rpd` / `tpd` 无限。`tpm`/`tpd` 只算输入 token，`-1` = 无限 |
| `is_free` | | `false`。给人看的注解，**不参与路由**；它说免费、却和一条付费行写了同一个 `quota_pool` 时告警（两行里必有一行的额度记错了） |
| `hint_output_ceiling` | | `false`。开关：在开场说明里告诉 agent 它每轮的输出上限（本行的 `max_output_tokens`）。只对实测受益的模型开，出厂没有一行开着 |
| `fallback_model` | | 空。**这一列会花钱**：填了之后，本行的模型答不了时（额度耗尽，或重试一次仍然过载）由 CLI 自己把这次会话交给这里写的模型跑完，并在结束时打一条 warning。只对有免费/付费孪生的行才有意义；出厂只有 `local-codebuddy-hy4flash-free`（→ `hy4flash`）写了。留空 = 答不了就报错，这是每一行的默认；写成本行自己的模型名等于没写（CLI 会跳过它） |
| `capability` | | 能力档，四档之一：`very-low`（极低）/ `low`（低）/ `medium`（中）/ `high`（高），写成字符串（`"medium"`）。只用来生成告警，不参与路由。**不写 = 没有判断**，按「高」算并在日志里记一条 note；想让能力下限对它生效就填一档。也收 0–100 的整数分数，读成它够得着的最高一档：50 起是低、70 起是中、90 起是高，不到 50 是极低（写成字符串的 `"77"` 会报错） |
| `avoid` | | 空。不推荐用于哪些任务组、为什么：`avoid = { knowledge = "原因" }`（组名是 `correction-mm` / `correction-text` / `research` / `knowledge`，拼错加载报错）。把这个模型绑进这些组的**任何一档**格子时告警一次，带上原因；不参与路由。出厂 GPT-6 Luna 与 Haiku 写了 `knowledge` |
| `note` | | 空。给人看的备注，程序不读 |
| `quota_pool` | | 取 `provider_tier`。**只对本地 agent 有意义**：一个订阅额度用完时，同池的模型会一起被停用（见 [`agent.md`](agent.md) [§6 异常处理](agent.md#6-异常处理)）。三种写法——**留空** = 该 tier 一份额度；**`per-model`** = 这一行自己一份（池名自动派生，所以新增一条独立计量的线不用起名字）；**写一个名字** = 明确和写了同一个名字的行共用。出厂 agy 的行写了 `AGY_GEMINI` / `AGY_ANTHROPIC` / `AGY_ARGON`（Gemini 4 Argon 单独一份），CodeBuddy 的免费线写 `per-model`、付费行写 `CODEBUDDY_CREDITS`（它们花的是同一份积分）。⚠ **这两个 tier 上不写会直接报错**：一个 CLI 背后有好几份分开计量的额度，「按 tier」在那里不是答案，猜哪边都会把某类行记错——报错文本会告诉你该写哪个 |
| `prepare_chunk_tokens` / `window_subtitle_tokens` / `knowledge_csv_tokens` | | 50000 / 10000 / 100000。三个**质量上限**，见下面的[三个质量上限](#三个质量上限) |

### 三个质量上限

一次任务分块、切窗口时各有一道上限（单位 token），写在模型上，也可以在 `config.toml` 的 `[chunking]` 里一次性压过所有模型：

| 上限 | 看哪一格绑的组 | catalog 里的键（按模型） | `config.toml` `[chunking]` 的键 | 出厂值 |
| --- | --- | --- | --- | --- |
| 准备任务一块转写多长就分块 | `research` | `prepare_chunk_tokens` | `research_transcript_max_tokens` | 50000 |
| 一个纠错窗口的字幕多长 | 纠错（准备任务规划的是同一批窗口，也看它） | `window_subtitle_tokens` | `max_window_subtitle_tokens` | 10000 |
| 知识更新一块的三份 CSV 合计多长 | `knowledge` | `knowledge_csv_tokens` | 没有 | 100000 |

- 只影响规划（窗口与块怎么切），不影响调用本身。catalog 里只收 ≥1 的整数，0 与负数加载即报错。
- 一个组按**组内最小值**规划，没写的成员按出厂值参与比较——**想调大，组里每个成员都要写**；组内不一致时启动告警一次，点名是谁定的。
- `config.toml` 里写了就压过模型的值。想关掉纠错窗口的字幕上限，写 `max_window_subtitle_tokens = 0`：窗口只随模型的输出上限变大，
  这道上限原本挡住的「窗口太大、纠错变粗」就没人管了。

### 窗口太小的模型会被拦下

绑定进模型组的成员，如果**最大输入 < 192,000** 或**最大输出 < 64,000**，启动时会告警一次
（照跑，但窗口会被切得更碎，纠错质量和合并判断都受影响）；低于 **96,000 / 32,000** 会**直接
停下来**，且发生在识别开始之前——避免在跑完数十分钟 ASR 后才暴露配置问题。

⚠ 比的是 `max_input_tokens` 与 `max_output_tokens` **这两列本身**，不是扣掉输出之后一个窗口能放进的输入量。
所以一个 200000 进 / 64000 出、上下文窗口 200000 的模型是**放行**的：它两边的量都够，
只是总量有限，而"总量有限"是正常模型的形态，不是配置错误。

只检查**当前 preset 用得到**的组；没被任何组引用的行（例如只做联网检索的 `gemma-4-31b`）不看。

### `token_scale`

别家模型分词与 Gemini 不同时，乘在本地 token 估算上的修正系数；只会让窗口变小，不会放宽任何边界（窗口按组内最大的系数切，且不低于 1.0）。
任务报告的「Token Estimate Calibration」一节给建议值，不会自动写回：要采纳就自己改这一格。报告与成本核算一律用 API 返回的数，不乘它。
（推导见 [`llm_harness_routing.md`](../llm_harness_routing.md)「模型配置、速率限制与显式 reasoning」，开发者参考。）

### `thinking`

| 取值 | 含义 |
| --- | --- |
| `true` | **原样发出**（默认）：high / medium / low 不换词 |
| `false` | 这个模型完全不发思考参数（**包括全局覆盖也不发**） |
| `["高映射", "中映射", "低映射"]` | 三档各换成这家服务认的词 |
| `[高, 中, 低]` 三个整数 | 按 token 预算控制思考的模型（Gemini 2.5）：直接就是三档的思考预算，如 `[24576, 16384, 8192]` |

三家（Gemini、OpenAI 兼容、Anthropic）的档位词本来就是 high / medium / low，所以默认原样发出就对。
自己的端点用别的词（例如 `max,default,minimal`）时写三个映射值；端点对未知字段直接报 400 时写 `false`；
写三个整数时不要超过该模型的思考预算上限，超了 API 会拒绝整次调用。
本地 agent 也按这一列发：Codex 走 `model_reasoning_effort`，Claude Code 走 `--effort`；产物里记的是映射后实际发出的值。
`[llm].local_agent_reasoning_effort` 默认留空，写了就全局覆盖每个模型自己的映射，但盖不过 `thinking = false`。
出厂各模型写了什么映射、为什么，见 [`llm_harness_routing.md`](../llm_harness_routing.md)「LLM thinking effort」（开发者参考）。

## 接自己的端点


两个文件，两件事。带注释的完整样例见 [GitHub 上的 `config.example.toml`](https://github.com/caca2331/finesub/blob/main/config.example.toml)（安装版本机没有这个文件）。

> **配置正确但仍出现异常时，原因可能在另一半。** 本节说明*需要配置什么*；各家 API 的实际差异
> (429 表示限流还是余额不足、思考深度参数的名称、拒答的表现形式、reasoning token 的计量方式)
> 已在 [`docs/provider-adapters.md`](../provider-adapters.md) 的调研表中逐家列出，并标注了哪些已
> 被实现处理（开发者参考）。接入非 OpenAI 兼容端点前，建议先浏览该表。

**① 事实**——在 `user-data` 目录建 `model_catalog.toml`（与 `config.toml` 同级）。**新模型**写齐必填键、
再写你关心的键（其余取默认值）；**改出厂的某个模型**只写要改的键（见[模型事实表](#模型事实表catalog)）：

```toml
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-flash"
max_input_tokens = 128000
max_output_tokens = 32768
capability = "medium"
```

这个表同时做了三件事：声明 `deepseek` 这家服务（接口格式 + 地址）、记成一条「自报」的事实、并自动得到一个同名的模型名供模型组引用。

**② API key**——不写在上面任何一处：按 `key_env`（缺省 `FINESUB_KEY_DEEPSEEK`）放进 `.env`，即 `FINESUB_KEY_DEEPSEEK=sk-...`，
走和其它 key 完全相同的加密路径（见 [`env.md`](env.md)）。

**③ 编排**——在 `config.toml` 里组队、绑格子。每个模型组、预设都写成 `[llm.model_groups.<名字>]`、`[llm.presets.<名字>]` 这样带名字的表。用打包已有的名字就是覆盖它（例如重写
`[llm.model_groups.correction-basic]` 换掉出厂 efficiency 格的纠错组，或整个重写
`[llm.presets.default]`）；覆盖会在启动时列出来：

```toml
[llm]
preset = "my-deepseek"          # 选激活预设

[llm.model_groups.my-corr]      # 有序模型组,先试前者
targets = ["ds-flash", "gemini-paid-3_8-flash"]

[llm.presets.my-deepseek]
name = "DeepSeek 纠错"
[llm.presets.my-deepseek.bindings]
"correction-text/quality" = "my-corr"
[llm.presets.my-deepseek.defaults]
media = "text"                  # 纠错只给文字，才走上面绑的 correction-text
```

未绑定的格子会先在你的预设内向上回落到更高难度，整组未绑才回落到 default 预设；因此只需编写
关心的几格。最后两行不能省：自己接的端点只读文字，不写时纠错用不上你的模型（见[模型路由「媒体开关」](model-routing.md#媒体开关---llm-media)）。
同理，例子只绑了纠错：准备任务（`research`）与知识更新（`knowledge`）两组没绑，仍用 default 预设的 Gemini；要全换就把 `research/quality`、`knowledge/quality` 也绑上。`test_target`（加 `--test-profile` 这个测试开关时，所有调用改用的那个便宜模型）未指定时继承 default 预设的取值。

### 自己的端点联网搜索

`--llm-retrieval native` 时，你接的 API 也能用供应商自己的联网搜索，按接口格式分两种：

- **`anthropic` 接口**：不用写什么，默认就会带上 Anthropic 服务端的 `web_search` 工具，**每次调用最多搜 5 次**。
  检索记录里有每次搜索的词和它找到的网页。⚠ 官方按次收费，每千次搜索约 10 美元，搜到的内容另按输入 token 计。
  超过 5 次的搜索会被拒，模型拿已搜到的作答。服务端的搜索循环跑得久时 API 偶尔会在作答前暂停它（`pause_turn`），
  FineSub 不续接，直接把这次调用交给组里的下一个模型。
  第三方的 Anthropic 兼容端点：DeepSeek 的 anthropic 接口（`https://api.deepseek.com/anthropic`）文档里示范了这个工具，
  示例用的是 `deepseek-v4-pro`，DeepSeek 另按每千次成功搜索 0.75 美元收费；Kimi、智谱的 anthropic 接口文档没提它。
  不跑这个工具的端点要么直接报错，要么把工具原样交回，FineSub 都会让这次调用失败并提示你写
  `supports_native_search = false`，不会不搜就作答。
- **`openai_compat` 接口**：各家开搜索的写法不一样，在模型表上写 `search_request`，这次调用要搜时才并进请求体：

  ```toml
  [models.qwen-max]
  provider_tier = "dashscope"
  provider_kind = "openai_compat"
  base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_model_id = "qwen-max"
  max_input_tokens = 128000
  search_request = { enable_search = true }
  ```

  几家的写法（2026-11-09 按各家官方文档核过；收费是文档上的价，另算搜到内容的 token）：

  | 供应商 | `search_request` | 说明 |
  | --- | --- | --- |
  | 阿里云百炼 / DashScope | `{ enable_search = true }` | 要强制搜索加 `search_options = { forced_search = true }`。只对支持的模型有效（同平台上的第三方模型不一定认）；北京地域每千次 3–4 元 |
  | OpenRouter | `{ tools = [{ type = "openrouter:web_search" }] }` | 模型自己决定搜不搜、可搜多次（Beta）；旧写法 `{ plugins = [{ id = "web" }] }` 已弃用。按搜索引擎收费，默认 Exa 每次 0.007 美元 |
  | 智谱 BigModel | `{ tools = [{ type = "web_search", web_search = { enable = true, search_engine = "search_pro" } }] }` | 引擎要写明（不写时默认哪个，文档没说）；`search_std` / `search_pro` 每次 0.01 / 0.03 元 |
  | OpenAI 官方 | `{ web_search_options = {} }` | 只有搜索专用模型（现为 `gpt-5-search-api`）认，`api_model_id` 要写成它；普通模型在这个接口上不能搜。每千次 10 美元 |
  | 火山方舟 / 豆包 | `{ web_search_options = {} }` | 要先开通「豆包搜索 Custom 版」、用支持的模型；每次 0.02 元 |
  | 百度千帆 | `{ web_search = { enable = true } }` | 限文档列出的 ERNIE 模型，ERNIE 不支持强制搜索 |
  | 腾讯混元（TokenHub） | `{ web_search_options = { enable = true } }` | 旧混元平台是 `{ enable_search = true }`，随旧平台下线；TokenHub 每千次 7 / 12 元（轻量 / 标准） |

  这个接口上不能搜的：DeepSeek 官方（Chat Completions 没有搜索开关；要搜就用它的 anthropic 接口，按上一条接成
  `anthropic` 接口）、xAI Grok（旧的 `search_parameters` 已移除，带上反而报错）、Kimi（`$web_search` 要多走一轮工具调用）。
  `search_request` 不能写 `model`、`messages`、`stream`、`stream_options`、
  `max_tokens`、`reasoning_effort`、`prompt_cache_key`（这些由 FineSub 自己写）。没写 `search_request` 的
  `openai_compat` 模型不能搜：没写开法就发不出搜索请求。检索记录只记「开了搜索」，看不到搜到了什么网页，
  因为各家回复格式不同。OpenAI 官方的搜索模型每次调用都会搜，这个开关管不到。

### 自己接的端点现在的三条限制

1. **只做纯文本。** 音视频只能交给出厂的 Gemini API 或 agy 的模型；带音视频的素材要加 `--llm-media text` 才交得到它，见[模型路由「媒体开关」](model-routing.md#媒体开关---llm-media)。
2. **「当天额度用完就停用到第二天」多数只对 Gemini 生效。** 别家服务额度用完时，多数按限流处理、反复等待重试；
   报错原文明说是每日额度（`per day`、`daily quota` 之类）时，照样停用到第二天。DeepSeek 的余额不足（402）与 OpenAI 的 `insufficient_quota` 会直接换组里下一个模型。
3. **第三方模型进 `knowledge` 组要谨慎。** 知识更新的结果会直接写进知识库，而没实测过的模型写得对不对没人核过。
   要用它，先在一两集上跑 `finesub knowledge-update … --execute --no-apply`，看过它打算写的内容再绑进 `knowledge` 那几格。
