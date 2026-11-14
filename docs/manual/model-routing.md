# 模型路由配置手册

**本章回答：一次 run 的每一次 LLM 调用会发给谁，使用什么 prompt、多少思考深度、是否携带
音视频；以及接入自有模型时需要修改哪几处。**

面向使用者。实现细节与设计取舍见 `docs/llm_harness_behavior.md`（运行时行为）与
`docs/provider-adapters.md`（自定义 endpoint 的供应商差异）。

---

## 1. 一次调用是怎么定下来的

一次 run 会运行多类 LLM 会话（准备任务、纠错窗、知识更新等）。选模型分四步，
每一步只回答一个问题：

```text
会话(harness 决定)
  └─→ 任务组          "这是哪一类活"      4 个,固定
        └─→ 格子       任务组 × difficulty  12 格,携带 prompt 变体
              └─→ 模型组  预设把格子绑到它    有序列表,可以自行组合
                    └─→ 逐候选过滤 → 真正应答的那个
```

- **任务组**由 harness 定义，不能新增：`correction-mm` / `correction-text` /
  `research` / `knowledge`。带 `-mm` / `-text` 的一对由媒体开关直接选中（见 [§2 开关与旋钮](#2-开关与旋钮)）。
- **预设**是一张「格子 → 模型组」的绑定表。随包发布一个 `default`；可以在 `config.toml` 中编写
  自己的预设，只需覆盖需要调整的几格。
- **模型组**是一个**有序**的模型列表。前面的先试，失败（额度/限流/超时/不可用/瞬时错误）
  才退到下一个。顺序就是优先级，没有打分选模、没有随机负载均衡。
- **逐候选过滤**发生在发车前：provider 是否启用、当日额度是否已打满、模型能否处理本次携带的
  音视频、能否联网搜索、prompt 估算长度与请求输出上限是否超过其自身限额。过滤掉就试下一个；
  **全部被过滤时报错**，不会静默降级。

> 新增一个模型**不会自动上线**。必须显式进入模型组、再被某个预设绑定，才会被调用。

## 2. 开关与旋钮

命令行开关在命令行上以 `--llm-` 开头（下表写的是全名）,`finesub` 与
`python -m finesub.pipeline` 一致：

| 开关 | 取值 | 作用 |
| --- | --- | --- |
| `--llm-media` | text / audio / video | **纠错窗**吃什么（阶梯，video 蕴含 audio）；同时选中 `correction-mm` 或 `correction-text`。`text` 也是让纯文本强模型处理音视频素材的方式 |
| `--llm-retrieval` | none / local / native | `local` = harness 自行检索（准备任务的回填 + 每窗请求回合 + 本地搜索代理）;`native` = 模型自带联网搜索（检索质量更好，推荐优先使用，见下）;`none` = 不检索 |
| `--llm-difficulty` | quality / intermediate / efficiency | 见下 |
| `--llm-continuity` | serial / parallel | 窗口之间是否串联上下文；`parallel` 放弃串联换并发 |
| `--knowledge` | none / read / update | 知识库档位：不用 / 只读（默认）/ 读并在任务后更新；独立于上述各轴，是少数**不带** `--llm-` 前缀的选项 |

这五个开关没在命令行上写时，取激活预设的默认值（见下面的[预设带开关默认值](#预设带开关默认值)），再取 `default` 预设的默认值，最后取内置默认。

### 走代理：`[api] proxy`

所有远程 API 请求共用一条代理设置：LLM 层的（Gemini REST、OpenAI 兼容 / Anthropic 端点、媒体上传、
Exa / Gemma4 / Tavily 检索、免费的 `countTokens`），以及云端转写的（MAI2 经 OpenRouter 或 Azure、
Gemini 3.5 Transcribe、MOSS Pro，见 [`joint-asr.md`](joint-asr.md)）：

```toml
[api]
proxy = "http://127.0.0.1:7890"   # 代理软件的 HTTP 端口
```

旧名字 `[llm] proxy` 照样读；两个都写时 `[api]` 的为准。只支持 HTTP 代理：Clash、v2rayN 等都提供 HTTP 端口，填它；
写 `socks5://` 会直接报错。

三种状态：

| 写法 | 含义 |
| --- | --- |
| 不写这一行 | 沿用环境变量 `HTTPS_PROXY` / `HTTP_PROXY`(0.5.0 之前就是这个行为) |
| `proxy = "<url>"` | 走这个代理，**并忽略环境变量**——文件比环境更具体，不做合并 |
| `proxy = "direct"` | 不走代理，**并忽略环境变量**。给「整机挂着代理，但 API 调用要直连」用 |

**范围仅限 API 调用。** 下载（模型、ffmpeg、yt-dlp、媒体源）有自己的一套路由发现
（系统代理 + 环境变量 + 本地端口探活），不受这一项影响。

改了这一行，**之后发出的请求**就会用新值——运行中途改也一样；已经在路上的那次请求不受影响。

**连不上时会看到什么。** 某个 API 地址连不上（被墙、代理没开或挂了）时，第一次失败就打一条
`Warning: 连不上 <地址>（直连 / 经代理 …）：<错误>`；连续 30 秒都连不上，再打一条「判定为不可达」并说该怎么办。
之后同一次运行里：token 计数改用本地估算（窗口略小，不影响正确性），联网检索跳过这家，生成与上传每次只试一次、
失败就换下一个候选或报错，不再一遍遍等超时。开了代理或网络恢复之后不用重启：生成调用下一次就恢复，计数与检索
最迟 10 分钟内自己恢复。要的处理通常就是开代理，或在这里设 `proxy`。

> ⚠️ **开启代理不仅改变线路，也会改变失败的表现。** Gemini 会对机房 IP 与共享出口 IP 评分，
> 判定可疑时直接拒答。因此「检索正常、纠错却持续失败」是该设置下的一种正常现象，并非 bug——
> 可先更换出口，再排查其他问题。

> 本机 agent(Codex / Claude Code / agy)**不受这一项影响**：它们是各自登录、自己发请求的
> 外部程序，`[api] proxy` 不会传给它们。这是有意的——它们有自己的网络配置与风控，替它们改线路
> 可能让发车失败被误判成额度耗尽。系统里本来就设着的 `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` /
> `NO_PROXY`（以及证书相关的变量）会原样传给它们；**要给它们单独配代理，按各自 CLI 的文档设置即可**，
> 不需要 finesub 代劳。

### 检索开关：优先 `native`

**模型支持联网搜索时，优先 `--llm-retrieval native`。** 它的检索质量明显好过 harness
自行检索：同一素材、同一模型各取 3 个样本对照，native 认出的背景实体并集是 local 的 1.6 倍
(33 vs 21)，专名也更准——一个角色的全名 local 给错、native 给对。差别在时序：`local` 的
搜索词要在见到内容之前一次性定好，`native` 却能边读边追问。

代价与注意：

- **需要能联网的模型**，见 [§4 出厂预设长什么样](#4-出厂预设长什么样) 末尾那条。纯免费 Gemini 档不行。
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

- `--llm-media video`：每个窗口切一份低清视频剪辑，请求回合与交付轮共用它，只切一次、只传一次。
- `--llm-media text`：完全不切片、不上传。**这也是让纯文本强模型（比如自接的 DeepSeek/Claude）
  处理带音视频素材的路径**——窗口只拿字幕文本。

> 读视频比读音频贵：视频在低分辨率下约 `71 tokens/帧 × 0.25 fps ≈ 17.75 tok/s`,
> 叠在音频约 32 tok/s 上，媒体 token 约 **+55%**。
> 是否值得由预设决定，harness 不替用户判断。

### difficulty 是定位

**怎么选**：

- **`intermediate`（默认）**：多数素材用它。强模型一步纠错翻译，不开说话人裁判、不校对。
- **`quality`**：多人对谈、常有人插话、要分清谁在说，或想要成品后再校对一遍时用。它就是 `intermediate` 加上说话人裁判、
  [三个 pass](diarization.md#三个-pass) 与校对；在已有的日语直播素材上，按价格加权约是 `intermediate` 的 1.4–2.9 倍
  （agy 上的 Gemini 3.8 Flash），第一次跑会下载说话人模型（约 450 MB）。
- **`efficiency`**：capable 的免费额度用完，或要大量、便宜地跑时用。换成两个 lite 模型，只接回被切断的词、不合并短行
  （ASR 切得太碎的行会照样一闪而过），只用出厂默认风格，不能开说话人裁判。

三档按**预期目标**命名，而非按输入难度。档位是定位，不写死模型能力：预设照样可以把任何模型组绑到任何格子上。
它选**该格的 prompt 变体**和**该格的思考档位**，并决定纠错怎么走；不影响窗口如何切分，也不从应答模型反推 prompt 档位。

| | `quality` | `intermediate`（默认） | `efficiency` |
| --- | --- | --- | --- |
| 给什么模型（出厂绑定） | capable 组 | capable 组 | 两个 lite，思考 low |
| 纠错的 prompt | capableC | capableC | basicA（一行对一个输入行，只接回被切断的词） |
| 不写 `--diarize` 时开说话人裁判 | 开（跑到 LLM 阶段时） | 不开 | 不开；显式开报错 |
| 三个 pass | 跟着说话人裁判：开了就走 | 写了 `--diarize judge-text` / `judge-audio` 才走 | 不走，一律单 pass |
| `--llm-proofread auto` 跑 pass 3 | 跑 | 不跑 | 不跑 |
| 风格 | 照常 | 照常 | 只用出厂默认风格 |

三个 pass、说话人裁判与 pass 3 见 [`diarization.md`](diarization.md)「三个 pass」；`efficiency` 的风格见
[`knowledge.md`](knowledge.md)。

**预设可以按档位绑定不同的模型组**，出厂就是这么配的：纠错的 `quality` 与 `intermediate` 绑 capable 组、
`efficiency` 绑两个 lite。因此：

- 免费档 capable 的日额度(20 RPD)耗尽、又没配付费 key 时，run 会**停下**而不是偷偷换成 lite（配了付费 key 就轮到组尾的付费成员）;
- 两条出路：等额度恢复后 resume，或显式 `--llm-difficulty efficiency` 继续（`intermediate` 绑的也是 capable 组，
  换过去额度一样不够）。开着说话人裁判的任务这样续跑也不报错：续跑照任务记下的裁判走。
- 切档**不会丢失已完成窗口**：窗口指纹不含 difficulty，窗口计划已落盘。产物中会出现前后不同
  变体的混合，这是显式发起的，逐窗都有记录。

`efficiency` 不钉别的开关：只写它时媒体、检索取预设或内置默认（出厂 `default` 下是音频 + `local` 检索，
`local` 要搜索 key），知识库照常读，也可以 `update`。要纯文本、不检索，用[快速粗纠](#快速粗纠agy-fast-与-fast)的两个预设，
或自己写 `--llm-media text --llm-retrieval none`。

### 思考旋钮

抽象档位 low / medium / high，写在预设上，按 `"任务组/难度"` 索引：

```toml
[llm.presets.my.thinking]
"correction-text/quality" = "high"
```

未填的格子会先在同一预设内向上查找更高档位(efficiency→intermediate→quality)，整组未填才回落到
default 预设，最终回退为 **medium**。

回落方向**朝上**，所以想让某格保持 medium 而它上一格是 high 时，必须显式写出来。

**出厂实况**（12 格全解析，格式为 `思考档 / 模型组 / prompt 变体`）:

| 任务组 | quality | intermediate | efficiency |
| --- | --- | --- | --- |
| correction-mm | medium / correction-capable / capableC | medium / correction-capable / capableC | **low** / correction-basic / basicA |
| correction-text | medium / correction-capable / capableC | medium / correction-capable / capableC | **low** / correction-basic / basicA |
| research | **high** / research-default / 单模板 | **high** / 同左 | **low** / 同左 |
| knowledge | **high** / knowledge-capable / 单模板 | **high** / 同左 | **low** / 同左 |

两条规律：**efficiency 一律 low**（最省的形态）;**research 与 knowledge 到 intermediate 都是
high**。它们的产物被下游全量复用（research 写的背景包每个窗口都读，knowledge 直接写库），换档
换的是模型，不应顺带砍掉推理。intermediate 那一格是**继承**来的，不是写死的：回落朝上。

抽象档位到供应商参数的换算见 [§3 模型事实表(catalog)](#3-模型事实表catalog) 的 `thinking` 列。

### 会话档位（只对 agent 生效）

一格绑到本机 agent(Codex / Claude Code / Antigravity)时，`agent_session` 决定**同一条 run
内这些调用如何共用会话**。与 thinking 一样写在预设上、按 `"任务组/难度"` 索引，未填则继承
`per-window`:

```toml
[llm.presets.my.agent_session]
"correction-text/quality" = "api"
```

| 档位 | 一次调用是什么 | 什么时候选它 |
| --- | --- | --- |
| `api` | 每次调用一个全新会话，整包 prompt 重发 | 想要每窗完全独立、可复现 |
| `per-window`（默认） | 一个窗口连同它的修复轮共用一次会话 | 出厂默认，不必动 |
| `resume`（实验） | 整条 run 一条会话，跨窗口续上下文 | 只用于实验：历史没有上限，长素材会撑满上下文；CodeBuddy 2.156.0 及更早的版本续会话吃不到历史缓存，2.164.0 起 hy3 实测已正常 |
| `pseudo-conversational`（已冻结） | 整条 run **一个常驻 CLI**：它做完一个窗口就向 FineSub 要下一个 | 不建议：保留现状、不再改进，试跑里比 `resume` 贵 |

三点要知道：

- `pseudo-conversational` **要求 CLI 支持 per-invocation MCP server**（五家都支持）。探测不到时
  直接报错停止，不会静默切换档位，因为选择该档即表示要求此形态。
- 带音视频的调用**任何档位下都单独走一次会话**，不进常驻 CLI（工具协议是纯文本的）。
- 该档位下 token 用量按**会话**记账（CLI 一次调用只报一次总账），所以任务报告里 agent 那行的
  「调用次数」按窗口算、token 按会话算，两列本就不是一回事。

### 预设带开关默认值

预设除了绑模型，还可以给上表那五个开关写默认值。例如一份只绑纯文本模型的预设，顺便把媒体定成文字：

```toml
[llm.presets.my-deepseek.defaults]
media = "text"
retrieval = "none"
```

键是 `media` / `retrieval` / `difficulty` / `continuity` / `knowledge`，取值与对应的命令行开关相同；另有一个整数键
`parallel_windows`（≥ 1，对应 `--llm-parallel-windows`），只在最后的 `continuity` 是 `parallel` 时生效，否则按 1 算、出处里写明。
写错键、写错值，或者同一份预设里自相矛盾（既写 `parallel_windows` 又写 `continuity = "serial"`），启动时直接报错。

一个开关取什么，按这个顺序找，找到就停：

1. 命令行上写了的值；
2. 激活预设的 `defaults`；
3. `default` 预设的 `defaults`（出厂的 `default` 一项都没写；出厂预设里写了默认值的是 `agy`、`agy-fast`、`fast`，见下面各节）；
4. 内置默认（audio / local / intermediate / serial / read）。

预设给的值和在命令行上手写同一个值，效果完全一样，作废已完成进度的范围也一样（[§7 什么会作废已完成的进度](#7-什么会作废已完成的进度)）。
每次纠错任务都会在产物目录写一份 `resolved-switches.json`，记下五个开关最后取了什么、各自来自命令行、哪个预设还是内置默认；
加 `--verbose` 时，取自预设的开关也会打印出来。

`finesub` 的任务记录里，这五个开关只记命令行上写了的，另有一个 `resolved_switches` 字段指向上面那份文件（没进纠错的 run 没有这份文件）。

参考资料导入（`reference_ingest`）不读这里的默认值：它的 `index.csv` 每行自带一个「预设」（`mm-med`、`text` 等），那是导入流程自己的一套，
媒体与检索由它定；路由预设对导入只管选模型。

## 3. 模型事实表(catalog)

**事实进 catalog，编排进 config.toml。** 一个 TOML 表对应一个**可调用的 （provider， 模型）**：
能做什么、限额多少、走哪个端点、思考参数如何表达、备注。同一模型在免费档和付费档是**两个表**，
因为供应商在不同档位上可能真的阉割能力和额度，这里不去重。

**两份文件，同一格式**：

| 文件 | 位置 | 作用 |
| --- | --- | --- |
| 默认 catalog | 随代码发布(`src/finesub/llm/routing/model_catalog.toml`) | 维护者实测的事实 |
| 你的 catalog | 数据根目录下的 `model_catalog.toml`，与 `config.toml` 同级 | 给出厂模型打补丁、追加新模型 |

```toml
# 给出厂模型打补丁：只写要改的键，其余沿用出厂值
[models.local-codebuddy-hy3]
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
  `python -m finesub.llm.routing.migrate_catalog`。它按旧文件的意思转换（旧的覆盖行是整行替换，
  没写的列在新文件里写成 `""`；只有 `supports_native_search` 不写，沿用出厂值，没写过的取新默认；`openai_compat` 行写过的 `true` 丢掉，它在旧版里本来就没搜过），逐项核对后才写出 `model_catalog.toml`，不覆盖已有文件；
  写出后旧文件改名成 `model_catalog.psv.bak` 留作备份（已有就用 `model_catalog.psv.2.bak`、`.3.bak`…）；还会列出你 `config.toml` 里改了名或删掉的 target 名。

数据根即 `.env` / `config.toml` 所在目录：装好的 CLI 是 `user-data`，仓库版没有单独的
user-data，即 checkout 根。你新加的模型、以及你改了事实的出厂模型会标为 `self_reported`，产物中与
实测事实区分开；只加了 `note` / `avoid` 的不标。

> **贯穿两个文件的规则：同名时，本地声明覆盖打包内容。** catalog 中同名的模型按键打补丁，
> `config.toml` 中同名的模型组 / 预设整个替换打包的对应项（包括 `default` 预设本身）。代价是拼错的
> id 不会被拦截，因此每次覆盖都会在启动时打印一行 `Note: config.toml 覆盖了打包声明的 …`;
> 若出现不认识的名字，即为拼写错误。
>
> `config.toml` 本身**不**跨目录合并：按「checkout 根 → user-data」取第一个存在的文件。

各键的类型与默认值（「不写时」也是写 `""` 时取的值）：

| 键 | 必填 | 不写时 |
| --- | --- | --- |
| 表名 `[models.<名字>]` | ✅ | — 模型的名字，也是它的 target 名（本地 agent 与自定义端点），模型组按它引用 |
| `provider_tier` | ✅ | — 与 `.env` 条目名一致（`GEMINI_FREE` / `GEMINI_PAID` / 自己的 provider id）。本地 agent 用 `LOCAL_CODEX` / `LOCAL_CLAUDE` / `LOCAL_AGY` / `LOCAL_DSH` / `LOCAL_CODEBUDDY`：它们不读 key，而是**决定用哪个 CLI driver**(`LOCAL_DSH` 的 `api_model_id` 要写成 `<dsh provider>/<model>`，例如 `deepseek-official/deepseek-flash`，因为 dsh 从插件配置选模型而不是命令行；`LOCAL_CODEBUDDY` 写模型 id 本身，取值以服务端认的那张清单为准，且**每行都必须写 `quota_pool`**（留空会直接报错）——那个账号背后不止一份额度：一条免费线一份、付费的共用一份积分，见 [`agent.md`](agent.md) 4.2) |
| `api_model_id` | ✅ | — 字符串，发给供应商的真实模型名 |
| `max_input_tokens` | ✅ | — 整数，**这个 API 能接受的最大输入**，没有乐观默认。它和 `context_window` 一起决定窗口怎么切，见下一行 |
| `provider_kind` | | 打包 tier 按其方言推断，否则 `openai_compat`。取 `gemini` / `local_agent` / `openai_compat` / `anthropic` |
| `base_url` | 文本方言必填 | 两个文本方言必须给；`gemini`/`local_agent` 不能给（传输自带端点） |
| `key_env` | | `FINESUB_KEY_<PROVIDER_TIER>`;key 本身放 `.env` |
| `display_name` | | 取 `api_model_id`。给人看的显示名 |
| `max_output_tokens` | | 65536 |
| `context_window` | | **不写 = 这个模型不声明联合约束**，存成两者之和，于是它恒不起作用。Gemini 分别公布输入与输出两个上限，所以**免费档留空**；**付费档与 agy 仍然填了**（`ctx = max_input = 1048576`，包络 983040）——该决定基于按单池规划的考虑，而非假设两个公布值能够同时用尽。**单池供应商必须填**（Codex、Anthropic：答案和提示词花的是同一份预算）。窗口的输入包络取 `min(max_input_tokens, context_window − 输出上限)`，因此填写该列后，能放入一窗的输入会比 `max_input_tokens` 略小，差额正是预留给答案的空间。⚠ 若填写值小于 `max_input_tokens` 或 `max_output_tokens`，会**直接报错**，不会被静默截断 |
| `supports_audio` / `supports_video` | | `false`。能力位，用于逐候选过滤。两个自定义 HTTP 文本方言(`openai_compat`/`anthropic`)**不允许**声明音视频；打包的 Gemini REST 与本地 Agy target 可以声明媒体能力 |
| `supports_native_search` | | Gemini 两档、五个本地 agent 与 `anthropic` 方言的端点为 `true`；`openai_compat` 端点写了 `search_request` 才为 `true`；会话 agent 为 `false`。这个模型能不能用它那家的搜索工具；只有 `--llm-retrieval native` 的调用才会真的把工具交给它，见[检索开关](#检索开关优先-native)。出厂只有免费档的 Gemini 3.x 写了 `false`。⚠ dsh 的搜索要绑 DeepSeek 的 API key，每次搜索扣少量费用（[`agent.md`](agent.md) 4.1）。自定义端点见[自己的端点联网搜索](#自己的端点联网搜索) |
| `search_request` | | 空。只给 `openai_compat` 端点：这家 API 开搜索要加的请求字段，写成内联表，`--llm-retrieval native` 的调用才并进请求体，见[自己的端点联网搜索](#自己的端点联网搜索) |
| `video_high_resolution_only` | | `false`。为 `true` 时，该模型参与的媒体格强制按高分辨率视频费率计帧（见 `docs/llm_local_agent_agy.md` [§3 视频](../llm_local_agent_agy.md#3-视频分辨率不可调现有-token-公式失真)）；出厂仅 agy 的 Gemini 行开启 |
| `thinking` | | `true`（恒等映射），见下 |
| `token_scale` | | 1.0，见下 |
| `rpm` / `tpm` / `rpd` / `tpd` | | 本地 agent 与会话 agent 的 `rpm` / `tpm` 不限，其余 100 / 4M；`rpd` / `tpd` 无限。`tpm`/`tpd` 只算输入 token,`-1` = 无限 |
| `is_free` | | `false`。这一列**不参与任何路由决策**，只是给人看的注解——唯一用到它的是一条告警：如果它说是免费的、却和一条付费行写了同一个 `quota_pool`，那两行里必有一行的额度记错了地方 |
| `hint_output_ceiling` | | `false`。开关：告诉 agent「你每轮输出上限是本行的 `max_output_tokens`，快到时先调一次工具再继续」。**只对被实测证明受益的模型开**——说错的数比不说更糟，所以数字永远取本行的 `max_output_tokens`，不另填。出厂只有 `local-codebuddy-glm-5_3-flash` 开着 |
| `fallback_model` | | 空。**这一列会花钱**：填了之后，本行的模型答不了时（额度耗尽，或重试一次仍然过载）由 CLI 自己把这次 session 交给这里写的模型跑完，并在结束时打一条 warning。只对有免费/付费孪生的行才有意义；出厂只有 `local-codebuddy-hy3`（→ `hy3-x`）写了。留空 = 答不了就报错，这是每一行的默认；写成本行自己的模型名等于没写（CLI 会跳过它） |
| `capability` | | 能力档，四档之一：`very-low`（极低）/ `low`（低）/ `medium`（中）/ `high`（高）。**纯咨询性**：只用来生成告警，绝不参与任何路由决策。写成字符串（`"medium"`）；也可以写 0–100 的整数分数，读成它够得着的最高一档：50 起是低、70 起是中、90 起是高，不到 50 是极低（`"77"` 这样写成字符串的分数会报错）。**不写 = 没有判断**，按「高」算（不触发任何能力下限），并在日志里记一条 note 说明这件事——不写不是「判断为差」，想让下限对它生效就填一档 |
| `avoid` | | 空。不推荐用于哪些任务组、为什么：`avoid = { knowledge = "原因" }`（组名是 `correction-mm` / `correction-text` / `research` / `knowledge`，拼错加载报错）。把这个模型绑进这些组的**任何一档**格子时告警一次，带上原因；不参与路由。出厂 GPT-6 Luna 与 Haiku 写了 `knowledge` |
| `note` | | 空。给人看的备注，程序不读 |
| — | | **规划包络与请求上限是两个数**（2026-09-04）：包络按「预计输出」从 `context_window` 里扣（纠错窗是 `output_scale × 输出系数 × 每窗字幕上限`，非纠错轮是 `output_scale × 32000`），而每次调用**仍然按答题模型自己的 `max_output_tokens` 打满地请求**。预留按估计放开输入，请求按上限保住答案余量 |
| `quota_pool` | | 取 `provider_tier`。**只对本地 agent 有意义**：一个订阅额度用完时，同池的 target 会一起被停用（见 [`agent.md`](agent.md) [§6 异常处理](agent.md#6-异常处理)）。三种写法——**留空** = 该 tier 一份额度；**`per-model`** = 这一行自己一份（池名自动派生，所以新增一条独立计量的线不用起名字）；**写一个名字** = 明确和写了同一个名字的行共用。出厂 Antigravity 写了 `AGY_GEMINI` / `AGY_ANTHROPIC` / `AGY_ARGON`（Gemini 4 Argon 单独一份），CodeBuddy 的免费线写 `per-model`、付费行写 `CODEBUDDY_CREDITS`（它们花的是同一份积分）。⚠ **这两个 tier 上不写会直接报错**：一个 CLI 背后有好几份分开计量的额度，「按 tier」在那里不是答案，猜哪边都会把某类行记错——报错文本会告诉你该写哪个 |
| `prepare_chunk_tokens` / `window_subtitle_tokens` / `knowledge_csv_tokens` | | 50000 / 10000 / 100000。三个**质量上限**，单位 token：准备任务一块转写多长就分块、一个纠错窗口的字幕多长、知识更新一块的三份 CSV 合计多长。只影响规划（窗口与块怎么切），不影响调用本身。只收 ≥1 的整数，0 与负数加载即报错（想关掉上限用 `config.toml`）。一个组按**组内最小值**规划，没写的成员按默认值参与比较——**想调大，组里每个成员都要写**；组内不一致时启动告警一次，点名是谁定的。`config.toml` `[chunking]` 的 `research_transcript_max_tokens` / `max_window_subtitle_tokens` 写了就压过模型的值；三个上限各看自己任务那一格绑的组：准备任务看 `research` 格，纠错窗口看纠错格（准备任务规划的是同一批窗口，也看它），知识更新看 `knowledge` 格 |

### 窗口太小的模型会被拦下

绑定进模型组的成员，如果**最大输入 < 192,000** 或**最大输出 < 64,000**，启动时会告警一次
（照跑，但窗口会被切得更碎，纠错质量和合并判断都受影响）；低于 **96,000 / 32,000** 会**直接
停下来**，且发生在识别开始之前——避免在跑完数十分钟 ASR 后才暴露配置问题。

⚠ 比的是 `max_input_tokens` 与 `max_output_tokens` **这两列本身**，不是扣掉输出之后的规划包络。
所以一个 200000 进 / 64000 出、上下文窗口 200000 的模型是**放行**的：它两边的量都够，
只是总量有限，而"总量有限"是正常模型的形态，不是配置错误。

只检查**当前 preset 用得到**的组；没被任何组引用的行（例如只做联网检索的 `gemma-4-31b`）不看。

### `token_scale`

本地那套三级 token 计数器用的是 Gemini 词表，别家模型分词不同，估算会有系统偏差。
`token_scale` 就是这个模型的修正系数，乘在**本地估算**上，只影响两件事：窗口能否装下的硬检查，
以及限流的 TPM 预留。

**报告和成本核算永远以 API 返回值为准**，不用估算值，也不乘这个系数。task report 里有
「Token Estimate Calibration」一节，按 fact 给出「返回 input tokens / 本地估算」的中位数与
样本数；样本够多且偏离当前系数够远时给一句建议值。**不会自动写回**：这个系数会改变**后续**
窗口预算，而且样本本身有偏（密集 CSV 与自然语言背景包差很多）。已落盘的窗口计划续跑时继续
定位已完成窗口；未完成窗口若不再合身，会在发车前自动拆小。新任务直接按新系数规划。

它的危险方向是**调小**：估算是安全边界。因此窗口规划取组内**最大**的系数、且不低于 1.0。
调大会让窗口相应变小（规划与发车两端口径一致），调小则不会放宽任何边界。

### `thinking`

| 取值 | 含义 |
| --- | --- |
| `true` | **恒等映射**（默认）：抽象档位原样发出 |
| `false` | 这个模型完全不发思考参数（**包括全局覆盖也不发**） |
| `["高映射", "中映射", "低映射"]` | 三个供应商取值，覆盖恒等 |
| `[高, 中, 低]` 三个整数 | 按 token 预算控制思考的模型（Gemini 2.5）：直接就是三档的思考预算，如 `[24576, 16384, 8192]` |

恒等是合理默认而非巧合，三家的档位词本来就同名：

| 抽象档位 | Gemini `thinkingConfig.thinkingLevel` | OpenAI 兼容 `reasoning_effort` | Anthropic `output_config.effort` |
| --- | --- | --- | --- |
| high | `high` | `high` | `high` |
| medium | `medium` | `medium` | `medium` |
| low | `low` | `low` | `low` |

只有当你的端点用别的词（例如 `max,default,minimal`）才需要写显式映射；自建端点对未知字段直接
400 时写 `false`。**第二种用法是给爱想的模型封顶**：出厂的 3.8 Flash 三行（agy 的 Gemini 4 Argon 照它）写的是
`medium,medium,low`——它在同一个抽象档位上比它之前的 3.7 Flash 多想约 1.5 倍，而思考与字幕正文共用同一份
输出上限，所以 high 被压回 medium;medium 那一格**故意保持 medium 不再往下**，因为实测把它降到
`low` 之后，五个纠错窗里有四个一个思考 token 都不出——`low` 是把要不要想交给模型，不是少想一点。
**按预算控制思考的模型写三个整数**：Gemini 2.5 没有档位词，只收 token 预算。不写时预算按输出上限的比例推出（20% / 40% /
60%，65536 下即 13107 / 26214 / 39321），可 2.5 Flash 的上限是 24576，超了 API 直接拒绝整次调用。所以出厂的 2.5 Flash 写
`24576,16384,8192`——按同一比例缩到它的上限之内。自己接 2.5 系模型时照此写，数字不超过该模型的上限。
映射同样用于本地 headless agent target:Codex 走 `model_reasoning_effort`,
Claude Code 走 `--effort`,dsh 写进它模型插件的 `reasoningEffort`。DeepSeek 那个插件只有
off/low/high/max，没有 `medium`，所以 FineSub 会把 `medium` 译成 `high`、`xhigh` 译成 `max`。
产物里记录的是**映射后实际发出**的值。`[llm].local_agent_reasoning_effort` 默认留空，只有
显式非空时才全局覆盖每个模型自己的映射；但**覆盖不了 `thinking = false`**：那不是「用默认档」,
而是「这个模型根本不收这个参数」，硬塞会在发车前被 CLI 拒绝，而那种硬失败会被额度探测误判成
订阅耗尽。

## 4. 出厂预设长什么样

| 任务组 | 能力下限 | quality 格参考模型 | quality 绑定的模型组 | 组成员（有序，括号里是能力档） |
| --- | --- | --- | --- | --- |
| correction-mm | 中 | 3.8 Flash | `correction-capable` | 免费 3.8 Flash（中）→ 免费 3.6 Flash（中）→ 付费 3.8 Flash（中） |
| correction-text | 中 | 3.8 Flash | `correction-capable` | 同上 |
| research | 低 | 3.6 Flash | `research-default` | 免费 3.6 Flash（中）→ 免费 3.8 Flash（中）→ 付费 3.8 Flash（中） |
| knowledge | 中 | 3.6 Flash | `knowledge-capable` | 同 research |

「参考模型」是**派生**的，且**按格**：该格所绑模型组的第一个成员。所以纠错的 `efficiency`
格绑 `correction-basic`（免费 3.5 Lite → 付费 3.5 Lite），参考模型就是
**3.5 Flash Lite**，而非 quality、intermediate 格的 3.8 Flash。参考模型仅用于展示，换代时只需修改一处真值。

下限告警本身只查 quality 格（档位只是定位，quality 以下的格绑什么由预设自己定）。

其余任务组只绑 quality，低档自动向上继承；纠错三格都写了出来：capable、capable、lite。

两条要事先知道的代价：

- **免费档的知识更新为尽力而为(best-effort)。** `knowledge` 与 `research` 均以免费 capable 模型
  优先，而免费档仅有 **20 RPD**。带调查的 run 会先消耗 research 配额，之后再分 chunk 执行知识
  更新，免费档很容易在知识更新阶段耗尽配额，且**没有 lite 后备**。这是「不静默降质」的代价，并非
  bug。
- **`--llm-retrieval native` 需要具备联网能力的模型。** 出厂由付费档提供联网；免费
  3.x 完全无法联网，纯免费档运行 native 会在启动校验时报错。此时可开启付费档，或显式绑定打包的
  `gemini-native-search` 组（含免费 2.5 Flash，但它的能力档「低」够不着纠错/知识要的「中」）。

### agent 的两个出厂预设：`agy-hybrid` 与 `agy`

两个都把 Antigravity 排进模型组，是 agent 模式的入口（详见 [`agent.md`](agent.md)）;
差别只有一处——**免费 API 在不在名单里**。

`[llm] preset = "agy-hybrid"`：免费 Gemini 优先，agy 作为后备；免费额度用尽后才消耗订阅。

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-capable` | 免费 3.8 Flash → agy Sonnet 5.5 → agy Gemini 3.8 Flash → agy Gemini 4 Argon | 纠错的 quality + intermediate、knowledge/quality |
| `agy-basic` | 免费 3.6 Flash → agy Sonnet 5.5 → agy Gemini 3.8 Flash → agy Gemini 4 Argon | research 的 quality |

免费 API 在前是有意的，两边的额度都不浪费。Sonnet 5.5 是**纯文本**，排在媒体成员之前同样有意：
文字任务优先给它，带剪辑的任务在能力过滤时自动跳过它、落到 agy 前置的那个 Gemini。

`--llm-retrieval native` 时，组里每个成员都用 agy 自带的搜索工具（每个模型只有一个 target，
搜不搜由这次调用定）。注意 agy 只报搜索词、不报来源 URL，Codex/Claude 那两家才有完整出处。

`[llm] preset = "agy"`：同样的格子，但名单中**没有任何 API 成员**。适用于已付费订阅、不愿再
管理 API key 的情况。它带一个开关默认值 `retrieval = "native"`，联网搜索用 agy 自带的工具，
因为 `local` 检索要 Exa / Gemini Free / Tavily 的 key。代价是只报搜索词、不报来源 URL；要换回就在命令行写
`--llm-retrieval local`。

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-only-capable` | agy Sonnet 5.5 → agy Gemini 3.8 Flash → agy Gemini 4 Argon | 纠错的 quality + intermediate、knowledge/quality |
| `agy-only-basic` | agy Sonnet 5.5 → agy Gemini 3.8 Flash → agy Gemini 4 Argon | research 的 quality |

> **与 `execution_policy = "agent-only"` 有何区别？** 两条路径可能产生相同的调用，但语义不同：
> policy 是**全局后端闸门**，只能禁用一个后端、不能新增成员，且对**任何**激活的预设生效；预设
> 描述的是「这份名单中有哪些成员」。需要临时将某次运行限制为纯 agent 时使用 policy；需要长期按
> 订阅运行时使用 `agy` 预设。两者叠加使用也没有问题。
>
> ⚠ `agy` 预设的 `test_target` 是 agy 自身的 Gemini 前端，**不是**默认预设使用的免费
> Gemini Lite:`--test-profile` 会将整轮运行中的每一次调用都固定到 test target 上；若沿用默认
> 值，相当于让「纯 agy」运行的测试全部打到 API 上。

> agy 的 Sonnet 5.5、Opus 5.5 按 high/medium/low 原样接收思考档位（`thinking` 为默认映射）；Gemini 4 Argon 与 3.8 Flash 一样封在 `medium,medium,low`。
> agy 没有 `xhigh`：`[llm].local_agent_reasoning_effort = "xhigh"` 对 agy 的各行按 `high` 发。
> Gemini 4 Argon 在每个出厂 agy 组里紧跟 3.8 Flash，当它的替补（额度另算，3.8 那份用完时还能接）。
> Opus 5.5 不在任何出厂模型组里，要用就 `--llm-model local-agy-opus-5_5` 点名，或列进自己的模型组。

### 快速粗纠：`agy-fast` 与 `fast`

给很长的输入做一遍不过度思考的简单纠错，意思大致对、方便找亮点即可。用 `--llm-preset agy-fast` 或 `--llm-preset fast` 选，
两者的开关默认值相同：`difficulty = "efficiency"`（低思考、单 pass、只用出厂默认风格；照常读知识库）、
`media = "text"`、`retrieval = "none"`（纯文本、不检索：`efficiency` 本身不钉这两个，所以预设自己写）、
`continuity = "parallel"`、`parallel_windows = 4`（4 个窗口同时跑）。命令行写了的照样优先。
在它们下面改用 `--llm-difficulty quality` 时也不搜索，而不是落到要 key 的 `local`；要 agy 的完整质量档（含自带搜索），
用 `--llm-preset agy`。`agy-fast` 只给纯文本的纠错格绑了 `agy-only-fast`：在它下面加 `--llm-media audio`，
纠错按「未绑的格向上回落」落到 `intermediate` 那格的 `agy-only-capable`（agy Sonnet 5.5 打头），仍不转到 API。

| 预设 | 给谁 | efficiency 格 | 其余格 |
| --- | --- | --- | --- |
| `agy-fast` | 有 agy 订阅的人 | 纠错与准备任务都只用 agy Gemini 3.8 Flash，它的额度用完时接 agy Gemini 4 Argon（`agy-only-fast`）；两份都用完就停下报错，**不会转到 API** | 与 `agy` 相同 |
| `fast` | 只用 API 的人 | 纠错落到两个 3.5 Flash Lite（免费在前、付费在后）；准备任务与 `default` 相同（免费 3.6 / 3.8 Flash，再接付费） | 与 `default` 相同 |

4 个窗口是上限，实际并发还受 `[llm].local_agent_max_parallel`（agent，默认 4）或 API 的每分钟请求限速约束。

### dsh 的两个组：`dsh-capable` / `dsh-basic`

**没有预设会自动绑它们**，要用就在自己预设的 `[llm.presets.<名>.bindings]` 里显式绑：

| 模型组 | 成员（有序） |
| --- | --- |
| `dsh-capable` | DeepSeek-V4.1-Flash → V4-Pro |
| `dsh-basic` | DeepSeek-V4.1-Flash |

> **只能绑定到按窗口执行的格子。** dsh 仅从命令行接收任务，整窗字幕无法放入命令行，因此它只支持
> 工具会话；绑定到 `api` 或 `resume` 档位会直接报错。同时它不支持音视频，也无法报告 token 用量。
> 安装方式、key 位置及「搜索使用另一把 key」的说明见 [`agent.md`](agent.md) 4.1。

**接入自己的 endpoint**：出厂仅包含 dsh 自带的 `deepseek-official` 路由，其他 provider 只存在于
**你自己的** `~/.dsh/settings.yaml` 中，随包发布对其他人没有意义。因此，自建网关通过 override
catalog 接入：在数据根目录（源码 checkout 即仓库根）的 `model_catalog.toml` 中写一个表，它会
**自动**成为同名的可绑定 target，无需修改其他文件。

```toml
[models.my-dsh-gateway]
provider_tier = "LOCAL_DSH"
api_model_id = "my-provider/my-model"
max_input_tokens = 194000
supports_native_search = false   # 不写就是 true：native 检索时会用 dsh 的搜索（要 DeepSeek key，扣少量费用）
```

`api_model_id` 的 `my-provider` 必须是在 `~/.dsh/settings.yaml` 的 `llm-pi-ai.providers` 下
声明过的名字。Codex、Claude Code、agy、CodeBuddy 的新模型也一样，写一个表就能用（agy 与 CodeBuddy
还要写 `quota_pool`）。

**接入自己的网关时，请将 `thinking` 设为 `false`**，随后在 `settings.yaml` 中为对应 provider 配置
`reasoning`。原因是 dsh 的配置覆盖采用整键替换：若 FineSub 代为写入该项，会把该 provider 的
`baseURL` 与 key 一并覆盖，因此 FineSub 不修改此项，档位设置留给你自行完成。

### CodeBuddy 的两个组：`codebuddy-capable` / `codebuddy-basic`

同样**没有预设会自动绑它们**，要用就在自己预设的 `[llm.presets.<名>.bindings]` 里显式绑：

| 模型组 | 成员（有序） |
| --- | --- |
| `codebuddy-capable` | GLM-5.3-Flash → DeepSeek-V4.1-Flash → V4-Pro |
| `codebuddy-basic` | hy3 → GLM-5.3-Flash → DeepSeek-V4.1-Flash |

> **两个组都会扣积分。** `codebuddy-capable` 的成员全是付费线；`codebuddy-basic` 从免费的 hy3 开始，但 hy3
> 出厂带着付费孪生（免费额度用完时由 CLI 切到 `hy3-x` 计费），后面的 GLM、DeepSeek 也是付费线。只想用免费
> 额度，就不绑组，直接绑 `local-codebuddy-hy3`，再把它的 `fallback_model` 清空（做法见
> [`agent.md`](agent.md) 4.2）。
>
> GLM-5.3-Flash 排第一，因为它是 [`agent.md`](agent.md) 4.2 推荐的那一行，这样绑组和绑单个模型从同一个模型开始。
> 它的单次输出上限只有 32,000，所以绑这两个组会看到「最大输出 32000 < 64000」的窗口告警，原因同 4.2。
> 免费线 hy3 与付费线各记各的额度，hy3 用完只让链条往下走一格。模型清单跟着账号走：你的账号够不到其中某个
> 模型时，调用会直接报错并贴出服务端原话（安装、登录与站点的说明也在 4.2）。

## 5. 接自己的模型

两个文件，两件事。完整样例见 `config.example.toml`。

> **配置正确但仍出现异常时，原因可能在另一半。** 本节说明*需要配置什么*；各家 API 的实际差异
> (429 表示限流还是余额不足、思考深度参数的名称、拒答的表现形式、reasoning token 的计量方式)
> 已在 [`docs/provider-adapters.md`](../provider-adapters.md) 的调研表中逐家列出，并标注了哪些已
> 被实现处理。接入非 OpenAI 兼容端点前，建议先浏览该表。

**① 事实**——在数据根目录建 `model_catalog.toml`（与 `config.toml` 同级）。**新模型**写齐必填键、
再写你关心的键（其余取默认值）；**改出厂的某个模型**只写要改的键（见[模型事实表](#3-模型事实表catalog)）：

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

这个表同时做了三件事：声明 `deepseek` 这个 provider（方言 + 端点）、成为一条 `self_reported`
事实、并自动得到一个同名 target 供模型组引用。

**② API key**——不写在上面任何一处：按 `key_env`（缺省 `FINESUB_KEY_DEEPSEEK`）放进 `.env`,
走和其它 key 完全相同的加密路径（见 [`env.md`](env.md)）。

**③ 编排**——在 `config.toml` 里组队、绑格子。全是**具名表**（没有 `[[数组表]]`），所以设置
面板的写入器和手动编辑可以共存。用打包已有的名字就是覆盖它（例如重写
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
"knowledge/quality"       = "local-codex-gpt-6_1-sol"
```

FineSub 会自动把它包成一个单成员模型组（内部名 `target:<名字>`）。这也是使用 Codex / Claude
Code / CodeBuddy 的办法：出厂预设只是没绑。可用的 target 名：本地 agent 与自定义端点的每个模型都有
一个与模型同名的 target（出厂的和你加的一样，如 `local-codex-gpt-6_1-sol`、`local-claude-sonnet`），
Gemini API 与 `conversational-agent` 写在 `model_routes.toml` 的 `[targets.*]` 段（随代码发布，与 catalog
同目录）。搜不搜不看名字：模型 `supports_native_search` 为真、这次运行又是 `--llm-retrieval native` 时，
同一个 target 就会用它那家的搜索工具。

**同名时模型组优先**，因此即使 catalog 中新增了同名模型，也不会在不知情的情况下改变你的绑定。

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

- 值先查模型组、再查 target；模型组与按模型名长出的 target 同名时用模型组（同上面预设绑定），与出厂手写的 target（Gemini API 的几个、`conversational-agent`）同名才报二义性错误。任务组名必须是真的
  (`correction-mm` / `correction-text` / `research` / `knowledge`)，写错直接报错并列出可选值。
- **对激活预设与 default 预设均生效**——未绑定的格子会回落到 default 预设，若优先模型恰好在这些
  格子上无法使用，则属于较难发现的问题。
- 修改该表会使未完成进度失效（见 [§7 什么会作废已完成的进度](#7-什么会作废已完成的进度)），因为「由谁应答」本身就是路由身份的一部分。

**命令行选预设**：`--llm-preset <名>` 只对这次运行生效，优先于 `config.toml` 的 `[llm] preset`；不存在的名字直接报错并列出可选的。
`finesub.pipeline`（整批一个；manifest 行里不能写）、`finesub.llm.correction_translation`、`finesub.llm.knowledge` 的
`repair`/`verify`/`ingest`、`finesub.llm.knowledge.update`、参考资料导入都收。换预设与在配置里换一样：只作废还没交付的那次调用的断点。

**不改文件的运行时版本**：`--llm-model [任务组=]值`（可重复；裸值 = default；作用在选好的预设之上），收它的入口与上面
`--llm-preset` 相同（两者共用一份定义）。语义与上表相同，只对本次进程生效，
config.toml 不动；优先级按 `命令行[任务组] > 命令行[default] > 配置[任务组] > 配置[default]`
逐格解析——裸写 `--llm-model X` 会盖过配置里的按任务组指定（这一次就是全用 X）,
配置里没被命令行碰到的任务组照旧生效。

与上一节的区别：**绑定 target 名**表示「这一格仅使用它，不做回退」；**优先模型**表示「让它排在
首位，回退仍然保留」。

### 自己的端点联网搜索

`--llm-retrieval native` 时，你接的 API 也能用供应商自己的联网搜索，按方言分两种：

- **`anthropic` 方言**：不用写什么，默认就会带上 Anthropic 服务端的 `web_search` 工具，**每次调用最多搜 5 次**。
  检索记录里有每次搜索的词和它找到的网页。⚠ 官方按次收费，每千次搜索约 10 美元，搜到的内容另按输入 token 计。
  超过 5 次的搜索会被拒，模型拿已搜到的作答。服务端的搜索循环跑得久时 API 偶尔会在作答前暂停它（`pause_turn`），
  FineSub 不续接，直接把这次调用交给组里的下一个模型。
  第三方的 Anthropic 兼容端点：DeepSeek 的 anthropic 接口（`https://api.deepseek.com/anthropic`）文档里示范了这个工具，
  示例用的是 `deepseek-v4-pro`，DeepSeek 另按每千次成功搜索 0.75 美元收费；Kimi、智谱的 anthropic 接口文档没提它。
  不跑这个工具的端点要么直接报错，要么把工具原样交回，FineSub 都会让这次调用失败并提示你写
  `supports_native_search = false`，不会不搜就作答。
- **`openai_compat` 方言**：各家开搜索的写法不一样，在模型表上写 `search_request`，这次调用要搜时才并进请求体：

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
  `anthropic` 方言）、xAI Grok（旧的 `search_parameters` 已移除，带上反而报错）、Kimi（`$web_search` 要多走一轮工具调用）。
  `search_request` 不能写 `model`、`messages`、`stream`、`stream_options`、
  `max_tokens`、`reasoning_effort`、`prompt_cache_key`（这些由 FineSub 自己写）。没写 `search_request` 的
  `openai_compat` 模型不能搜：没写开法就发不出搜索请求。检索记录只记「开了搜索」，看不到搜到了什么网页，
  因为各家回复格式不同。OpenAI 官方的搜索模型每次调用都会搜，这个开关管不到。

### 第一版的三条硬限制

1. **自定义 endpoint 只做纯文本。** 媒体由打包的 Gemini REST 或本地 Agy target 承担。想让纯
   文本强模型处理带音视频的素材，用 `--llm-media text`。绑错不会
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
| `成员 X 的能力档「低」低于下限「中」` | quality 格的组里有成员低于任务组的能力下限 | 确认是有意放的。另外两档只是定位，绑什么由预设自己定，不告警 |
| `成员 X 未声明能力档，按「高」计` | 这是 **note 不是告警**（只进日志，`--verbose` 才上屏）| 不必处理。想让能力下限对它生效，就在 catalog 行里填一档 |
| `模型组 X 的成员 … 不一致，按其中最小的 N 切分（由 Y 决定…）` | 组里成员写的三个质量上限不同（没写的按默认计），这个组按最小的切分 | 想用大的，给组里每个成员都写上 |
| `规划包络由 X 决定 (…tokens…)，窗口数约 ×N` | 组内最小的输入包络/输出上限低于免费 Gemini 基线 | 窗口按**组内最低**规划（谁应答还没定，必须保守）。×N 是窗口数放大倍数。⚠ 2026-09 起**反过来也成立了**：harness 那道 194,000 的硬上限已经删掉，组内全是大窗口模型就真的会切出更大的窗。真正还在限制每窗字幕量的是质量护栏 `[chunking] max_window_subtitle_tokens`（默认 10000），把它设成 0 会连这道也去掉——那时窗口大小只随模型的输出上限与输出系数走，护栏原本挡住的质量风险就没人管了 |
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
| 改 `difficulty` 到 `efficiency` | 复用 | 复用 |
| 改 `continuity` | 复用（`parallel→serial` 会告警：advice 台账从空重建） | 复用 |
| 改知识库内容、`--knowledge` | 复用 | 复用 |
| 改能力档 `capability`、任务组的 `min_capability`、显示名、预设名、注释 | 复用 | 复用 |
| 改 catalog 的三个质量上限 | 复用（窗口切法存在任务里，续跑沿用）；调小 `window_subtitle_tokens` 时，没跑完的窗按新上限预拆，同下面「窗口上限」那行 | 整份准备结果复用（不看分块上限）；分块的准备任务没跑完时改 `prepare_chunk_tokens`，范围变了的块重跑 |
| 改预设的开关默认值（`defaults`） | 与在命令行上改同一个开关相同（见上面各行） | 同左 |
| 重新下载同一份音视频（大小不变） | 复用 | 复用 |
| 改媒体开关、`--output-scale`、`token_scale`、窗口上限，或换成**上下文更小的模型组** | 复用；pending 窗按当前包络预拆 | 复用（若同时改 retrieval，则重跑） |
| 改 prompt version、任务备注/风格、`--test-profile` | ❌ | ❌ |
| `*-stable.json` 内容变化 | ❌ | ❌ |
| 计划/JSON 损坏、schema 不兼容、存档响应过不了当前 parser | 该层重做；被覆盖的 context 先备份为 `*.invalid[.N]` | 同左 |
| `--no-resume` 或删除产物 | 按要求重新执行 | 同左 |

窗口计划作为恢复地址，不会因几何类参数变化而自动更换。恢复时会按当前媒体/profile/模型包络
重新计算 clip 与预算，仅对未完成且不再适配的叶子节点递归拆半；已完成窗口保持不变。拆分不会使
窗口变大：更换为更大模型后，如需合成更少、更大的窗口，可显式删除 artifact 目录中的
`correction-window-plan.json`——这会同时放弃原 chunk id 对窗口缓存的寻址。

切档只换 prompt 变体、思考档位与（按出厂绑定）模型组，媒体与检索不随档位变，所以额度耗尽后显式切到
`efficiency` 时窗口与调查都原样保留。任务记成三个 pass 的，没做完的窗按单 pass 的 prompt 纠错、告警一次。

再往下还有一层更细的 session checkpoint（一次尚未提交的调用）。它要求精确一致，是「这个断点
能不能重建」的结构判断，不是要求整条任务保持同一模型。`agent_session` 档位就在这一层生效：
改它只作废尚未提交的调用断点，不影响已提交的窗口与 research context。

预设**在 run 内不会重读**：启动时快照一次；运行中途修改设置面板不会影响正在进行的 run，将于
下次生效。
