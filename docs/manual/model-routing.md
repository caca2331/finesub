# 模型路由配置手册

**这一章回答：这次 run 的每一次 LLM 调用，到底会打给谁，用什么 prompt、多深的思考、吃不吃
音视频；以及你想换成自己的模型时，改哪几行。**

面向使用者。实现细节与设计取舍见 `docs/llm_harness_behavior.md`（运行时行为）与
`docs/provider-adapters.md`（自定义 endpoint 的供应商差异）。

---

## 1. 一次调用是怎么定下来的

一次 run 会跑好几类 LLM 会话（背景调查、每窗查询轮、纠错窗、知识更新……）。选模型分四步，
每一步只回答一个问题：

```text
会话（harness 决定）
  └─→ 任务组          "这是哪一类活"      7 个，固定
        └─→ 格子       任务组 × difficulty  21 格，携带 prompt 变体
              └─→ 模型组  预设把格子绑到它    有序列表，你可以自己组
                    └─→ 逐候选过滤 → 真正应答的那个
```

- **任务组**是 harness 定的，你不能新增：`correction-mm` / `correction-text` /
  `planning-mm` / `planning-text` / `research` / `search_judge` / `knowledge`。
  带 `-mm` / `-text` 的两对，由下面两个媒体开关**直接选中**（见 §2）。
- **预设**是一张「格子 → 模型组」的绑定表。随包发布一个 `default`；你可以在
  `config.toml` 里写自己的，只覆盖在意的几格。
- **模型组**是一个**有序**的模型列表。前面的先试，失败（额度/限流/超时/不可用/瞬时错误）
  才退到下一个。顺序就是优先级，没有打分自动选模、没有随机负载均衡。
- **逐候选过滤**发生在发车前：provider 有没有开、今天额度是不是已经打满、这个模型能不能
  吃这次要带的音视频、能不能联网搜索、prompt 估算长度和请求的输出上限是否超过它自己的限额。
  过滤掉就试下一个；**全被过滤掉就报错**，不会偷偷降级。

> 新加一行模型事实**不会自动上线**。必须显式进模型组、再被某个预设绑上，才会被调用。

## 2. 开关与旋钮

命令行开关（`python -m finesub.pipeline` 上加 `--llm-` 前缀，如 `--llm-correction-media`）：

| 开关 | 取值 | 管什么 |
| --- | --- | --- |
| `--media` | text / audio / video | 便捷写法，一次设定下面两个 |
| `--correction-media` | text / audio / video | **纠错窗**吃什么。同时选中 `correction-mm` 还是 `correction-text` |
| `--planning-media` | text / audio / video | **每窗查询轮**吃什么。同时选中 `planning-mm` 还是 `planning-text` |
| `--retrieval` | none / local / native | `local` = harness 自己查（背景调查 + 每窗查询轮 + 本地搜索代理）；`native` = 模型自带联网搜索（**检索质量更好，模型能接得上，推荐优先用**，见下）；`none` = 都不查 |
| `--difficulty` | quality / intermediate / efficiency | 见下 |
| `--continuity` | serial / parallel | 窗口之间是否串联上下文；`parallel` 放弃串联换并发 |
| `--knowledge` | none / collect / update | 知识库读写档位，独立于上面几轴 |

### 检索开关：优先 `native`

**模型接得上联网搜索时，优先 `--retrieval native`。** 它的检索质量明显好于 harness 自己查：
同一素材、同一模型、各 3 个样本的对照里，native 认出的背景实体并集是 local 的 1.6 倍
（33 vs 21），且在专名上更准——一个角色的全名 local 给错、native 给对。原因不难理解：
`local` 的 query 是在看到内容之前一次性定好的，`native` 可以边读边追问。

代价与注意：

- **它需要能联网的模型**，见 §4 末尾那条。纯免费 Gemini 档不行。
- **`native` 下的出处不可信，别拿来核对。** agy 这类后端只回报搜索 query、不回报来源 URL，
  于是模型手里没有真 URL——实测它会把 `sources` 填上长得很像真的编造链接（YouTube video id
  写成 `genshin_last_legacy`、`4gamer.net/games/000/G000000`），且每次编的都不一样。
  要可核对的检索台账就用 `local`：它注入的每条 URL 都在产物里。
- 两者不是「谁全面替代谁」：**要覆盖面选 `native`，要可追溯选 `local`。**

### 两个媒体开关

它们回答的是「**哪个任务用媒体**」，不是「这次有没有媒体」——后者由你给不给
`--audio` / `--llm-video` 文件决定。开关要得比手上的文件多，是**配置错误会报错**，不会
悄悄降级。

常见组合：

- `--media video`：查询轮和纠错窗共用同一份视频剪辑，只切一次、只传一次。
- `--correction-media text --planning-media video`：纠错交给纯文本强模型（比如你自己接的
  DeepSeek/Claude），查询轮仍然看视频。**这是让纯文本模型用在有音视频素材上的正规路径**。
- `--media text`：完全不切片、不上传。

> 查询轮读视频比读音频贵：视频在低分辨率下约 `71 tokens/帧 × 0.25 fps ≈ 17.75 tok/s`，
> 叠在音频的约 32 tok/s 上，媒体 token 大约 **+55%**。换来的是与纠错窗共用一份剪辑、
> 少一次上传。值不值由你按预设决定，harness 不替你判断。

### difficulty 只做两件事

选**该格的 prompt 变体** + **该格的思考档**。它不再影响窗口怎么切，也不再从应答模型
反推 prompt 档位。

三个档位是按**你想要什么**命名的，不是按输入有多难：`quality`（要质量）/
`intermediate`（折中）/ `efficiency`（要省）。

而**预设可以按档位绑不同的模型组**，出厂就是这么配的：纠错的 `quality` 绑 capable 组、
`intermediate` 绑两个 lite。所以：

- 免费档 capable 的日额度（20 RPD）耗尽时，run 会**停下**而不是偷偷换成 lite；
- 你有两条出路：等额度恢复后 resume，或者显式 `--difficulty intermediate` 继续。
- 切档**不会丢掉已完成的窗口**：窗口指纹里没有 difficulty，窗口计划也落了盘。产物里会
  出现前后不同变体的混合，这是你显式发起的，逐窗都有记录。

`efficiency` 额外钉死 `correction_media=text`、`planning_media=text`、`retrieval=none`，
并且 **`--knowledge` 必须是 `none`**（不读索引、不注入词条、不做任务后更新）。冲突即报错，
不会悄悄降级。

### 思考旋钮

抽象档位 low / medium / high，写在预设上，按 `"任务组/难度"` 索引：

```toml
[llm.presets.my.thinking]
"correction-text/quality" = "high"
```

不填的格子先在同一预设内向更高档位找（efficiency→intermediate→quality），整组都没填才
回落到 default 预设，最后兜底是 **medium**。

回落方向是**朝上**的，所以想让某格保持 medium 而它上面那格是 high，必须显式写出来。

**出厂实况**（21 格全解析，格式为 `思考档 / 模型组 / prompt 变体`）：

| 任务组 | quality | intermediate | efficiency |
| --- | --- | --- | --- |
| correction-mm | medium / correction-capable / capableC | medium / correction-basic / basicB | **low** / correction-basic / basicB |
| correction-text | medium / correction-capable / capableC | medium / correction-basic / basicB | **low** / correction-basic / basicB |
| planning-mm | medium / lightweight-default / 单模板 | medium / 同左 | **low** / 同左 |
| planning-text | medium / lightweight-default / 单模板 | medium / 同左 | **low** / 同左 |
| search_judge | medium / lightweight-default / 单模板 | medium / 同左 | **low** / 同左 |
| research | **high** / research-default / 单模板 | **high** / 同左 | **low** / 同左 |
| knowledge | **high** / knowledge-capable / 单模板 | **high** / 同左 | **low** / 同左 |

两条规律：**efficiency 一律 low**（那是最省的形态）；**research 与 knowledge 一路想到
intermediate 都是 high**——它们的产物被下游全量复用（research 写的背景包每个窗口都读，
knowledge 直接写库），换档换的是模型，不该顺带砍掉推理。（intermediate 那一格是**继承**
来的，不是写死的：回落朝上。）

抽象档位到供应商参数的换算，见 §3 的 `thinking` 列。

## 3. 模型事实表（catalog）

**事实进 catalog，编排进 config.toml。** 一行一个**可调用的 (provider, 模型)**：它能做什么、
限额多少、走哪个端点、思考参数怎么说。同一个模型在免费档和付费档是**两行**——供应商在不同
档位上可能真的阉割能力和额度，这里不做去重。

**两份文件，同一个格式**：

| 文件 | 位置 | 作用 |
| --- | --- | --- |
| 默认 catalog | 随代码发布（`src/finesub/llm/routing/model_catalog.psv`） | 维护者实测的事实 |
| 你的 catalog | 数据根目录下的 `model_catalog.psv`，与 `config.toml` 同级 | 覆盖同名 `fact_id`、追加新 id |

数据根就是 `.env` / `config.toml` 所在的那个目录：装好的桌面端/CLI 是 `user-data`，仓库版没有
单独的 user-data，就是 checkout 根。你的行会被标成 `self_reported`，产物里与实测事实区分开。

> **一条贯穿两个文件的规则：同名就是「我的赢」。** catalog 里同 `fact_id` 覆盖那一行，
> `config.toml` 里同名的模型组 / 预设覆盖打包的那一个（包括 `default` 预设本身）。
> 代价是拼错 id 不会被拦下——所以每次覆盖都会在启动时打印一行
> `Note: config.toml 覆盖了打包声明的 …`，看到不认识的名字就是拼错了。
>
> `config.toml` 本身**不**跨目录合并：它按「checkout 根 → user-data」取第一个存在的文件。

**表头声明有哪几列**，所以覆盖文件只写你要说的那几列就行；列里留空取默认值：

| 列 | 必填 | 留空时 |
| --- | --- | --- |
| `fact_id` | ✅ | — 这行的名字，模型组按它引用 |
| `provider_tier` | ✅ | — 与 `.env` 条目名一致（`GEMINI_FREE` / `GEMINI_PAID` / 你自己的 provider id）。本地 agent 用 `LOCAL_CODEX` / `LOCAL_CLAUDE` / `LOCAL_AGY`：它们不读 key，而是**决定用哪个 CLI driver** |
| `api_model_id` | ✅ | — 发给供应商的真实模型名 |
| `max_input_tokens` | ✅ | — 上下文窗口。**窗口按它切**，填错每一窗都会炸，所以它没有乐观默认。注意它只往下起作用：harness 自己有一道 194,000 的上限，填得更大不会切出更大的窗（见下） |
| `provider_kind` | | 打包 tier 按其方言推断，否则 `openai_compat`。取 `gemini` / `local_agent` / `openai_compat` / `anthropic` |
| `base_url` | 文本方言必填 | 两个文本方言必须给；`gemini`/`local_agent` 不能给（传输自带端点） |
| `key_env` | | `FINESUB_KEY_<PROVIDER_TIER>`；key 本身放 `.env` |
| `display_name` | | 取 `api_model_id`。给人看的显示名 |
| `max_output_tokens` | | 65536 |
| `supports_audio` / `supports_video` / `supports_native_search` | | `false`。能力位，用于逐候选过滤。两个自定义 HTTP 文本方言（`openai_compat`/`anthropic`）**不允许**声明音视频；打包的 Gemini REST 与本地 Agy target 可以声明媒体能力 |
| `thinking` | | `true`（恒等映射），见下 |
| `token_scale` | | 1.0，见下 |
| `rpm` / `tpm` / `rpd` / `tpd` | | 100 / 4M / 无限 / 无限；`tpm`/`tpd` 只算输入 token，`-1` = 无限 |
| `is_free` | | `false` |
| `quality_score` | | 50。0–100，**纯咨询性**：只用来生成告警，绝不参与任何路由决策 |
| `quota_pool` | | 取 `provider_tier`。**只对本地 agent 有意义**：一个订阅额度用完时，同池的 target 会一起被停用 5 小时。写了它才分家——出厂只有 Antigravity 分了（`AGY_GEMINI` / `AGY_ANTHROPIC`），因为它一个 CLI 后面是两份分开计量的额度 |

覆盖是**整行替换**而不是打补丁：覆盖一行打包模型时，你没写的列取的是**默认值**，不是打包
行的原值。想只改一个限额，把要保留的列一并写上。

### `token_scale` 列

本地那套三级 token 计数器用的是 Gemini 的词表，别家模型的分词不同，估算会有系统偏差。
`token_scale` 就是这一行的修正系数，乘在**本地估算**上——只影响两件事：窗口能不能装下的
硬检查，以及限流的 TPM 预留。

**报告和成本核算永远以 API 返回值为准**，不用估算值，也不乘这个系数。task report 里有一节
「Token Estimate Calibration」，按 fact 给出「返回 input tokens / 本地估算」的中位数与样本数；
样本够多且偏离当前系数够远时给一句建议值。**不会自动写回**：这个系数会改变**后续**窗口预算，
而且样本本身有偏（密集 CSV 与自然语言背景包差很多）。已落盘的窗口计划续跑时继续定位已完成
窗口；未完成窗口若不再合身，会在发车前自动拆小。新任务直接按新系数规划。

它的危险方向是**调小**：估算是安全边界。因此窗口规划取组内**最大**的系数、且不低于 1.0——
调大会让窗口相应变小（规划与发车两端口径一致），调小则不会放宽任何边界。

### `thinking` 列

| 取值 | 含义 |
| --- | --- |
| `true` | **恒等映射**（默认）：抽象档位原样发出 |
| `false` | 这个模型完全不发思考参数（**包括全局覆盖也不发**） |
| `高映射,中映射,低映射` | 三个供应商取值，覆盖恒等 |

恒等是合理默认而非巧合——三家的档位词本来就同名：

| 抽象档位 | Gemini `thinkingConfig.thinkingLevel` | OpenAI 兼容 `reasoning_effort` | Anthropic `output_config.effort` |
| --- | --- | --- | --- |
| high | `high` | `high` | `high` |
| medium | `medium` | `medium` | `medium` |
| low | `low` | `low` | `low` |

只有当你的端点用别的词（例如 `max,default,minimal`）才需要写显式映射；自建端点如果对
未知字段直接 400，写 `false`。映射同样用于本地 headless agent target（Codex 走
`model_reasoning_effort`，Claude Code 走 `--effort`）；产物里记录的是
**映射后实际发出**的值。`[llm].local_agent_reasoning_effort` 默认留空，只有显式非空时才全局覆盖
每个模型自己的映射——但**覆盖不了 `thinking = false`**：那不是"用默认档"，是"这个模型根本不收
这个参数"，硬塞会在发车前被 CLI 拒绝，而那种硬失败会被额度探测误判成订阅耗尽。

## 4. 出厂预设长什么样

| 任务组 | 下限 | quality 格参考模型 | quality 绑定的模型组 | 组成员（有序） |
| --- | ---: | --- | --- | --- |
| correction-mm | 70 | 3.7 Flash | `correction-capable` | 免费 3.7 Flash(75) → 免费 3.6 Flash(75) → 免费 3.5 Flash(70) → 付费 3.7 Flash(75) |
| correction-text | 70 | 3.7 Flash | `correction-capable` | 同上 |
| planning-mm | 50 | 3.5 Flash Lite | `lightweight-default` | 免费 3.5 Lite(60) → 付费 3.5 Lite(60) |
| planning-text | 50 | 3.5 Flash Lite | `lightweight-default` | 同上 |
| research | 50 | 3.6 Flash | `research-default` | 免费 3.6 Flash(75) → 免费 3.5 Flash(70) → 免费 3.7 Flash(75) → 付费 3.7 Flash(75) |
| search_judge | 50 | 3.5 Flash Lite | `lightweight-default` | 同 planning |
| knowledge | 70 | 3.6 Flash | `knowledge-capable` | 同 research |

「参考模型」是**派生**的，且**按格**：该格所绑模型组的第一个成员。所以纠错的
`intermediate` / `efficiency` 格绑 `correction-basic`（免费 3.5 Lite → 付费 3.5 Lite），
它们的参考模型就是 **3.5 Flash Lite**，而不是 quality 格的 3.7 Flash。
它只写给人看——换代时只有一处真值要改。下限告警本身只查 quality 格（另外两档本来就是声明过
的降档）。

其余任务组只绑 quality，低档自动向上继承。

### 另一个出厂预设：`agy`

`[llm] preset = "agy"` 会换成把 Antigravity 排进去的两个组。它是 agent 模式的入口，
详见 [`agent.md`](agent.md)：

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-capable` | 免费 3.7 Flash → agy Opus 4.6 → agy Gemini 3.7 Flash | 纠错的 quality + intermediate、knowledge/quality |
| `agy-basic` | 免费 3.5 Flash → agy Opus 4.6 → agy Gemini 3.7 Flash | research、planning、search_judge 的 quality |

> Opus 4.6 那一行的思考档位**不可调**：agy 把档位烘进了模型名，只有它的 Gemini 系列
> 分 high/medium/low。所以该行 `thinking` 填 `false`，`[llm].local_agent_reasoning_effort`
> 对它也不生效（带上去 agy 会直接拒绝发车）。

免费 API 在前是有意的：两边的额度都不浪费。Opus 4.6 是**纯文本**，排在媒体成员之前也是有意
的——文字活优先给它，带剪辑的活在能力过滤时自动跳过它、落到 agy 前置的那个 Gemini。

两个组的末尾还各有一个 `local-agy-native-gemini-3_7-flash`：同一个模型、但授权了 agy 自己的
搜索工具，**只有 `--retrieval native` 会用到它**（普通调用在它之前就选中了没授权的那一档）。
注意 agy 只报搜索了什么词、不报来源 URL，Codex/Claude 那两档才有完整出处。

两条要事先知道的代价：

- **免费档的知识更新是 best-effort**。`knowledge` 与 `research` 都以免费 capable 打头，而它
  只有 **20 RPD**。带调查的 run 先烧 research、跑完再分 chunk 做知识更新，免费档很容易在
  知识更新阶段撞墙——而它**没有 lite 兜底**。这是「不静默降质」的代价，不是 bug。
- **`--retrieval native` 需要能联网的模型**。出厂由付费 3.7 Flash 接地；免费 3.x 完全不能
  联网。纯免费档跑 native 会在启动校验处报错，要么开付费档，要么显式绑定打包的
  `gemini-native-search` 组（含免费 2.5 Flash，但它 50 分，低于纠错/知识的下限 70）。

## 5. 接自己的模型

两个文件，两件事。完整样例见 `config.example.toml`。

> **配好了却行为古怪，另一半在这里。** 本节讲*你要写什么*；各家 API 的实际差异——429 到底
> 是限流还是余额尽、思考深度那个参数叫什么名、拒答长什么样、reasoning token 算在哪——在
> [`docs/provider-adapters.md`](../provider-adapters.md) 的调研表里逐家列着，并标明了哪些
> 已被实现消化。接非 OpenAI 兼容的端点之前值得先扫一眼那张表。

**① 事实**——在数据根目录建 `model_catalog.psv`（与 `config.toml` 同级），写你要说的那几列：

```text
fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|max_output_tokens|quality_score
ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|32768|60
```

这一行同时做了三件事：声明了 `deepseek` 这个 provider（方言 + 端点）、成为一条
`self_reported` 事实、并自动得到一个同名 target 供模型组引用。

**② API key**——不写在上面任何一处：按 `key_env`（缺省 `FINESUB_KEY_DEEPSEEK`）放进 `.env`，
走和其它 key 完全相同的加密路径。

**③ 编排**——在 `config.toml` 里组队、绑格子。全是**具名表**（没有 `[[数组表]]`），所以设置
面板的写入器和你手动编辑可以共存。用打包已有的名字就是覆盖它（例如重写
`[llm.model_groups.lightweight-default]` 换掉出厂的轻量组，或整个重写
`[llm.presets.default]`）；覆盖会在启动时列出来：

```toml
[llm]
preset = "my-deepseek"          # 选激活预设

[llm.model_groups.my-corr]      # 有序模型组，先试前者
targets = ["ds-flash", "gemini-paid-3_7-flash"]

[llm.presets.my-deepseek]
name = "DeepSeek 纠错"
[llm.presets.my-deepseek.bindings]
"correction-text/quality" = "my-corr"
```

**API key 不写在 config.toml 里**：按 `key_env`（缺省 `FINESUB_KEY_<PROVIDER_ID>`）放进
`.env`，走和其它 key 完全相同的加密路径。

未绑定的格子先在你的预设内向更高难度回落，整组没绑才回落到 default 预设——你只需要写
在意的那几格。`test_target` 不写就继承 default 的。

### 快速选模型：格子可以直接绑一个模型

只想说「这一格就用它，不要回退」时，不必先写一个只有一个成员的模型组——把格子直接绑到
**target 名**上就行：

```toml
[llm.presets.my-deepseek.bindings]
"correction-text/quality" = "my-corr"        # 一个模型组
"research/quality"        = "ds-flash"       # 直接一个模型
"knowledge/quality"       = "local-codex-completion-gpt-5_6-luna"
```

FineSub 会自动把它包成一个单成员模型组（内部叫 `target:<名字>`）。这也是用 Codex / Claude Code
的办法：它们的 target 都声明好了，出厂预设只是没绑而已。可用的名字见
`model_catalog.psv` 的 `fact_id` 列（`local-codex-*` / `local-claude-*` 各有 `completion` 和
`native` 两个变体，后者才允许模型用自己的搜索工具）。

**同名时模型组优先**，所以别人往 catalog 里加一行同名模型不会悄悄改掉你的绑定。

### 第一版的三条硬限制

1. **自定义 endpoint 只做纯文本。** 媒体由打包的 Gemini REST 或本地 Agy target 承担。想让纯文本强模型处理有音视频的
   素材，用 `--correction-media text`（查询轮照常吃剪辑）。绑错了不会静默丢音频——带媒体的
   调用会在能力过滤时跳过它。
2. **日封禁不适用于非 Gemini。** 日额度封禁靠 Gemini 结构化 `quotaId` 判定；别家的 429
   一律归 rate_limit，撞满了会一直退避重试（DeepSeek 的 402 余额不足和 OpenAI 的
   `insufficient_quota` 例外，它们归 quota，直接推进到组内下一位）。
3. **第三方模型进 `knowledge` 组要谨慎。** 知识更新的产物会自动 apply 进知识库，而未标定 /
   provenance 不可审计的 target **不得作为知识更新的唯一证据来源**。

## 6. 启动时那几条告警怎么读

绑定期告警按**当前激活的预设**打印（出厂预设零告警，有测试保护）：

| 告警 | 触发 | 该怎么办 |
| --- | --- | --- |
| `成员 X 的 quality_score=N 低于下限 M` | quality 格的组里有成员低于任务组下限 | 确认你是有意放的。另外两档本身就是声明过的降档，不告警 |
| `规划包络由 X 决定 (…tokens…)，窗口数约 ×N` | 组内最小的上下文/输出上限低于免费 Gemini 基线 | 窗口按**组内最低**规划（谁应答还没定，必须保守）。×N 是窗口数放大倍数。反过来不成立：组内全是大窗口模型也**不会**切出更大的窗，194,000 是 harness 的硬上限，要动它得先重标定输出系数与窗口预算（`docs/llm_followups.md` 的 P6/P8） |
| `组内没有成员支持音频` | `-mm` 格里一个能听的都没有 | 带媒体的调用会在能力过滤后无候选可用；换模型，或把该任务改成 `text` |
| `N 个成员里只有 1 个支持音频` | 该格能服务媒体调用，但只有一个候选 | 不是错误：纯文本成员排在前面是刻意的（文字活优先给它）。但媒体调用的链长为 1，那一个失败就整轮失败——想要冗余就再加一个能听的成员 |
| `target X 不支持视频，按 video->audio 降一级` | 运行期安全网触发 | 正常配置不该看到它。说明该格里混进了看不了视频的模型 |

## 7. 什么会作废已完成的进度

判据不是“配置是否一字不差”，而是**这条已提交的结果还能不能按同一个源和同一套窗口边界来解释**。
能，就继续用，新配置只管尚未完成的部分；因此一条任务允许混合模型、思考档位和 prompt 变体，
逐窗记录保留实际 `variant` / `difficulty` / `knowledge_version` 备查。

| 改动或状态 | 已完成窗口 | 已完成 research context |
| --- | --- | --- |
| 换 `--model`、预设、**模型组**、thinking、execution policy、agent 参数 | 复用 | 复用 |
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
| `--no-resume` 或删除产物 | 按你的要求重做 | 同左 |

窗口计划是恢复地址，不因几何旋钮变化自动换掉。恢复时会按当前媒体/profile/模型包络重算 clip 与
预算，只对未完成且不合身的叶子递归拆半；已完成窗口不动。拆分不能把窗口变大：换了更大模型后若想
合成更少、更大的窗口，显式删除 artifact 目录里的 `correction-window-plan.json`，这也会放弃原
chunk id 对窗口缓存的寻址。

`efficiency` 会**连带钉住** `correction_media=text` 和 `retrieval=none`（见 §3）。前者只触发 pending
窗口预算体检；后者改变调查语义，所以已有 research context 会重跑。只想在额度耗尽后换便宜模型并
最大限度复用两层产物，仍优先用 **`intermediate`**——它只换 prompt 变体和思考档位，
窗口与调查都原样保留；`efficiency` 则会保留已完成窗口、重做调查并体检 pending 窗。

再往下还有一层更细的 session checkpoint（一次尚未提交的调用）。它要求精确一致——那是“这个断点能
不能重建”的结构判断，不是要求整条任务保持同一模型。

预设在 run 内**不重读**：启动时快照一次，跑到一半改设置面板不影响正在跑的 run，下次生效。
