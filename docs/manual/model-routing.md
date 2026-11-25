# 模型路由配置手册

**本章回答：一次运行里的每一次 LLM 调用会发给哪个模型，给它哪一套说明（prompt）、多少思考深度、是否携带
音视频；怎么换预设、挑一个出厂模型。** 接自己的 API 端点见[接自己的模型](custom-models.md)。

本章常用的几个词：**任务组**是这次调用做的哪一类活（纠错、准备任务、知识更新）；**难度档**是 `--llm-difficulty` 的三档；
**格子**是「任务组 × 难度档」的一格（如「纠错 / quality」），预设按格子指定用哪个模型；**模型名**（配置里叫 target）是可以直接写进格子的单个模型，
出厂有哪些见[出厂的模型名](#出厂的模型名target)；**模型组**是几个模型名排成的有序列表，前一个答不了才轮到下一个；
**窗口**是纠错时分批送给模型的一段字幕，一个窗口的结果存下了就算完成，续跑不再重做。

---

## 常用配方

下面几条给出现成写法，细节在后面各节。下文说的 `config.toml` 装好时并不存在，没有就新建（放在哪、怎么建见[资源与大文件「设置文件 `config.toml`」](resources.md#设置文件-configtoml)）。

- **只用 Antigravity（下文简称 agy）的订阅、不配 API key**：`config.toml` 写 `[llm]` 下 `preset = "agy"`；想先用 Gemini 免费额度、用完再接订阅，写
  `preset = "agy-hybrid"`（见 [agent 的两个出厂预设](#agent-的两个出厂预设agy-hybrid-与-agy)）。
- **只这一次换预设或模型**：命令行加 `--llm-preset agy`，或 `--llm-model local-codex-gpt-6_2-sol --llm-media text`（写成
  `--llm-model research=<模型名>` 只换准备任务这一个任务组），不改文件（见[优先模型](#优先模型一行说就用它不改预设)末尾）。
  换成只读文字的模型时要带 `--llm-media text`，不带会怎样见[媒体开关](#媒体开关---llm-media)。
- **某一格固定用一个模型、不回退**：在自己预设的 `bindings` 里把这一格直接写成模型名，例如
  `"knowledge/quality" = "local-codex-gpt-6_2-sol"`（见[快速选模型](#快速选模型格子可以直接绑一个模型)）。
- **不写整套预设，只说「就用它」**：`[llm.preferred_targets]` 下写 `default = "<模型组或模型名>"`，所有格子换绑过去、不再回退到原来的组
  （见[优先模型](#优先模型一行说就用它不改预设)）。
- **很长的素材快速过一遍**：`--llm-preset agy-fast`（有 agy 订阅）或 `--llm-preset fast`（只用 API）。这是「快速粗纠」：
  便宜档、纯文本、不检索、4 个窗口并行（见[快速粗纠](#快速粗纠agy-fast-与-fast)）。
- **免费额度用完了**：额度恢复后重跑同一条命令接着做，或加 `--llm-difficulty efficiency` 换便宜档续跑，已完成的窗口不丢
  （见 [difficulty 是定位](#difficulty-是定位)）。
- **API 要走代理**：`config.toml` 写 `[api]` 下 `proxy = "http://127.0.0.1:7890"`（见[走代理](#走代理api-proxy)）。
- **接自己的 OpenAI 兼容端点**：在 `model_catalog.toml` 写一个 `[models.<名字>]` 表、key 放进 `.env`，再在 `config.toml`
  里把它绑进预设（见[接自己的模型](custom-models.md)）。

## 1. 一次调用是怎么定下来的

一次运行会发出好几类 LLM 调用（准备任务、纠错窗口、知识更新等）。选模型分四步，
每一步只回答一个问题：

```text
这次调用(FineSub 按任务定)
  └─→ 任务组          "这是哪一类活"      4 个,固定
        └─→ 格子       任务组 × 难度档      12 格,各带一套 prompt 变体
              └─→ 模型组  预设把格子绑到它    有序列表,可以自行组合
                    └─→ 逐个筛掉做不了的 → 真正应答的那个
```

prompt 变体是给模型的说明的不同写法，如给强模型的一套与给 lite 模型的一套。

- **任务组**是固定的四个，不能新增：`correction-mm` / `correction-text`（纠错，带不带音视频）/
  `research`（准备任务：纠错前通读整段素材、查背景）/ `knowledge`（知识更新）。带 `-mm` / `-text` 的一对由媒体开关直接选中（见 [§2 开关与旋钮](#2-开关与旋钮)）。
- **预设**是一张「格子 → 模型组」的绑定表。随包发布一个 `default`；可以在 `config.toml` 中编写
  自己的预设，只需覆盖需要调整的几格。
- **模型组**是一个**有序**的模型列表。前面的先试，失败（额度/限流/超时/不可用/瞬时错误）
  才退到下一个。顺序就是优先级，没有打分选模、没有随机负载均衡。
- **逐个筛掉做不了的**发生在发出请求之前：这家服务（配了 key、装了 CLI）能不能用、当日额度是否已打满、模型能否处理本次携带的
  音视频、能否联网搜索、prompt 估算长度与请求输出上限是否超过其自身限额。筛掉就试下一个；
  **全部被过滤时报错**，不会静默降级。

> 新增一个模型**不会自动上线**。必须显式进入模型组、再被某个预设绑定，才会被调用。

## 2. 开关与旋钮

命令行开关在命令行上以 `--llm-` 开头（下表写的是全名）,`finesub` 与
`python -m finesub.pipeline` 一致：

| 开关 | 取值 | 作用 |
| --- | --- | --- |
| `--llm-media` | text / audio / video | **纠错窗**吃什么（阶梯，video 蕴含 audio）；同时选中 `correction-mm` 或 `correction-text`。`text` 也是让纯文本强模型处理音视频素材的方式 |
| `--llm-retrieval` | none / local / native | `local` = FineSub 自己用搜索 API 检索，把结果交给模型；`native` = 模型自带联网搜索（检索质量更好，模型能联网时优先用；纯免费 Gemini 档不行，见下）；`none` = 不检索 |
| `--llm-difficulty` | quality / intermediate / efficiency | 见下 |
| `--llm-continuity` | serial / parallel | 窗口之间是否串联上下文；`parallel` 放弃串联换并发 |
| `--knowledge` | none / read / update | 知识库档位：不用 / 只读（默认）/ 读并在任务后更新；独立于上述各轴，是少数**不带** `--llm-` 前缀的选项 |

这五个开关没在命令行上写时，取激活预设的默认值（见下面的[预设带开关默认值](#预设带开关默认值)），再取 `default` 预设的默认值，最后取内置默认。

### 走代理：`[api] proxy`

所有远程 API 请求共用一条代理设置：LLM 层的（Gemini REST、OpenAI 兼容 / Anthropic 端点、媒体上传、
Exa / Gemma4 / Tavily 检索、免费的 `countTokens`），以及云端转写的（MAI2 经 Vercel、OpenRouter 或 Azure、
ElevenLabs Scribe、Gemini 3.5 Transcribe、MOSS Pro，见 [`asr-backend-choice.md`](asr-backend-choice.md)）：

```toml
[api]
proxy = "http://127.0.0.1:7890"   # 代理软件的 HTTP 端口
```

旧名字 `[llm] proxy` 照样读；两个都写时 `[api]` 的为准。只支持 HTTP 代理：Clash、v2rayN 等都提供 HTTP 端口，填它；
写 `socks5://` 会直接报错。

三种状态：

| 写法 | 含义 |
| --- | --- |
| 不写这一行 | 沿用环境变量 `HTTPS_PROXY` / `HTTP_PROXY` |
| `proxy = "<url>"` | 走这个代理，**并忽略环境变量**——文件比环境更具体，不做合并 |
| `proxy = "direct"` | 不走代理，**并忽略环境变量**。给「整机挂着代理，但 API 调用要直连」用 |

**范围仅限 API 调用。** 下载（模型、ffmpeg、yt-dlp、媒体源）有自己的一套路由发现
（系统代理 + 环境变量 + 本地端口探活），不受这一项影响。

改了这一行，**之后发出的请求**就会用新值——运行中途改也一样；已经在路上的那次请求不受影响。

**连不上时会看到什么。** 某个 API 地址连不上（被墙、代理没开或挂了）时，第一次失败就打一条
`Warning: 连不上 <地址>（直连 / 经代理 …）：<错误>`；连续 30 秒都连不上，再打一条「判定为不可达」并说该怎么办。之后同一次运行里：
token 计数改用本地估算（窗口略小，不影响正确性），联网检索跳过这一家，生成与上传每次只试一次、失败就换下一个候选或报错，
不再一遍遍等超时。开了代理或网络恢复之后不用重启：生成调用下一次就恢复，计数与检索最迟 10 分钟内自己恢复。
要的处理通常就是开代理，或在这里设 `proxy`。（判定与恢复的细节见
[`llm_harness_behavior.md`](../llm_harness_behavior.md)「连不上的 host」，开发者参考，可跳过。）

> ⚠️ **开启代理不仅改变线路，也会改变失败的表现。** Gemini 会对机房 IP 与共享出口 IP 评分，
> 判定可疑时直接拒答。因此「检索正常、纠错却持续失败」是该设置下的一种正常现象，并非 bug——
> 可先更换出口，再排查其他问题。

> 本机 agent（Codex / Claude Code / agy）不走这个代理，见 [`agent.md`](agent.md)「3. 需要装什么」。

### 检索开关：优先 `native`

**模型支持联网搜索时，优先 `--llm-retrieval native`。** 它的检索质量明显好过 FineSub
自行检索（`local`）：同一素材、同一模型各跑 3 次对照，三次结果合并去重后，native 认出的背景实体是 33 个、local 是 21 个（约 1.6 倍；不是单次多认出六成），专名也更准。
原因是 `local` 的搜索词要在见到内容之前一次性定好，`native` 却能边读边追问。

代价与注意：

- **需要能联网的模型**，见 [`models.md`](models.md)「免费档的两个代价」。纯免费 Gemini 档不行。
- **`native` 下出处能否核对，取决于后端类型。** agy 只报搜索词、不报来源网址，它给的出处会被标成未核实；
  其余几家（Codex、Claude Code、dsh、CodeBuddy）会记下每次搜索的真实网址。要核对出处，用 `local` 或这几家。
- `local` 的出处始终可查：FineSub 亲自执行检索，网址直接写入产物。
- 如何选择：**追求覆盖面使用 `native`，需要控制预算、限定用哪家搜索服务时使用 `local`**（每次调用的查询
  数、抓取数与实际耗时都有硬上限；`native` 只能事后记账，无法预先限制搜索次数）。

### 媒体开关 `--llm-media`

`--llm-media` 决定的是纠错窗口带哪种媒体给模型。媒体从哪来不用另给：主命令（`finesub` 与 `python -m finesub.pipeline`）的输入本身就是音频来源。

- 选 `video` 要有画面：输入是视频就用它；输入是音频时用 `--llm-video <视频文件>` 另给，不给就降成 `audio` 并提示一行
  （URL 输入加了 `--no-download-video` 时不降而是报错，见[调参「输入与下载」](tuning.md#输入与下载)）。
- 从源码单独跑 LLM 层（`python -m finesub.llm.correction_translation`，安装版没有对应命令）时，音视频要用它自己的 `--audio` / `--video` 给；
  开关要的媒体多于给了的文件时直接报错，不会降级。

常见组合：

- `--llm-media video`：每个窗口切一份低清视频剪辑，同一个窗口的几次请求共用它，只切一次、只传一次。
- `--llm-media text`：完全不切片、不上传，窗口只拿字幕文本。**只读文字的模型（[出厂的模型名](#出厂的模型名target)里「音视频」一列写「否」的，
  以及所有自己接的端点）要靠它才接得到纠错**：媒体开关决定纠错走 `correction-mm` 还是 `correction-text` 格，出厂默认 `audio`，走 `correction-mm`。不写时：
  - 只把这个模型绑在 `correction-text` 格、`correction-mm` 没绑：回落到 `default` 预设的 Gemini（没配 Gemini key 就报错），你的模型用不上纠错；
  - 用 `--llm-model` 或 `[llm.preferred_targets]` 钉了它（不保留回退链，见[优先模型](#优先模型一行说就用它不改预设)）：纠错报错停下；
  - 它在一个模型组里：带音视频的调用跳过它、交给组里下一个成员，不会静默丢掉音频；
  - `execution_policy = "agent-only"`（[agent.md「执行策略」](agent.md#5-进阶执行策略-execution_policy)）时：上面回落到的 Gemini API 被屏蔽，带音视频的调用没有 agent 能接（只有 agy 的 Gemini 能听）就直接报错。

  预设绑的都是只读文字的模型时，在它的 `defaults` 里写 `media = "text"`（见[预设带开关默认值](#预设带开关默认值)），等于每次都加 `--llm-media text`。

> 读视频比读音频贵：开视频比只给音频多花约 **55%** 的媒体 token。
> 是否值得由预设决定，FineSub 不替用户判断。

### difficulty 是定位

**怎么选**：

- **`intermediate`（默认）**：多数素材用它。强模型一步纠错翻译，不开说话人裁判；每个窗口交卷前在同一段对话里再校对一遍（[窗内校对](tuning.md#llm-这一侧)），成品出来后不另校对。要分清谁在说、又想省一点时，加 `--diarize judge-text`（或 `judge-audio`）：
  仍是一步纠错，同一次调用顺带认人（[一步纠错兼认人](diarization.md#一步纠错兼认人)），连同窗内校对，花费比 `quality`（含成品后的校对）低约两成（在 Claude Opus 上量的，窗内校对读到前缀缓存时）。
- **`quality`**：多人对谈、常有人插话、要分清谁在说，或想要成品后再校对一遍时用。它在 `intermediate` 之上加说话人裁判
  （这一档默认开，见[说话人分离「怎么开」](diarization.md#怎么开)）、纠错走[三个 pass](diarization.md#三个-pass)、成品出来后再校对一遍（不做窗内校对）；在已有的日语直播素材上，按价格加权约是 `intermediate` 的 1.4–2.9 倍
  （agy 上的 Gemini 3.8 Flash；`intermediate` 还没有窗内校对时量的，现在的差距会小一些），第一次跑会下载说话人模型（约 450 MB；`--asr` 交给一体转写模型的任务不下，见[转写后端「说话人」](asr-backend-choice.md#说话人)）。
- **`efficiency`**：capable 组（出厂的强模型组：Gemini 4 Hydrogen、3.8 / 3.6 Flash）的免费额度用完，或要大量、便宜地跑时用。
  换成两个 lite 模型（Gemini 3.5 Flash Lite 的免费与付费两档），只接回被切断的词、不合并短行
  （ASR 切得太碎的行会照样一闪而过），只用出厂默认风格，不能开说话人裁判。

三档按**预期目标**命名，而非按输入难度。档位是定位，不写死模型能力：预设照样可以把任何模型组绑到任何格子上。
它选**该格的 prompt 变体**和**该格的思考档位**，并决定纠错怎么走；不影响窗口如何切分，也不从应答模型反推 prompt 档位。

| | `quality` | `intermediate`（默认） | `efficiency` |
| --- | --- | --- | --- |
| 给什么模型（出厂绑定） | capable 组 | capable 组 | 两个 lite，思考 low |
| 纠错的 prompt | 强模型那套 | 强模型那套 | lite 那套（一行对一个输入行，只接回被切断的词） |
| 不写 `--diarize` 时开说话人裁判 | 开（跑到 LLM 阶段时） | 不开 | 不开；显式开报错 |
| 纠错走三个 pass | 跟着说话人裁判：开了就走 | 不走：开了裁判是一步纠错兼认人（中文只纠错的任务也是，开跑时提示一行）；一体转写的任务例外，照样走三个 pass | 不走，一律一步纠错 |
| `--llm-proofread auto` 时校对（`on` 时三档都校对） | 校对 | 不校对 | 不校对 |
| `--llm-proofread-inline auto` 时窗内校对 | 不校对 | 校对（一步纠错的窗口；中文只纠错的任务与换了 prompt 的格子不做，见[调参](tuning.md#llm-这一侧)） | 不校对 |
| 风格 | 照常 | 照常 | 只用出厂默认风格，见[`knowledge.md`「风格条目」](knowledge.md#风格条目style记住一套翻译习惯) |

说话人裁判、三个 pass 与校对的细节见 [`diarization.md`](diarization.md)「[三个 pass](diarization.md#三个-pass)」。`intermediate` 开了裁判时，一体转写的任务照样走三个 pass；中文只纠错的素材也一步纠错兼认人，但量过的中文素材上认人不如三个 pass，开跑时提示一行，要更稳用 `quality`。

**预设可以按档位绑定不同的模型组**，出厂就是这么配的：纠错的 `quality` 与 `intermediate` 绑 capable 组、
`efficiency` 绑两个 lite。因此：

- 免费档 capable 的日额度（每个模型每天 20 次请求）耗尽、又没配付费 key 时，运行会**停下**而不是偷偷换成 lite（配了付费 key 就轮到组尾的付费成员）；
- 两条出路：等额度恢复后续跑，或显式 `--llm-difficulty efficiency` 继续（`intermediate` 绑的也是 capable 组，
  换过去额度一样不够）。`efficiency` 只把纠错换成 lite：准备任务与知识更新在三档都用各自 quality 那一格的组（同一批免费强模型），额度在准备任务里就用完、或带着 `--knowledge update` 时，换过去也接不住，只能等额度或配付费 key。开着说话人裁判的任务这样续跑也不报错：续跑照任务记下的裁判走。
  例外：额度在准备任务里就用完的任务，纠错怎么走还没记下来，这时命令里仍写着 `--diarize judge-text` / `judge-audio` 会按新任务报错；去掉它再续跑。
- 切档**不会丢失已完成窗口**。产物中会出现前后不同变体的混合，这是显式发起的，逐窗都有记录。

`efficiency` 不钉别的开关：只写它时媒体、检索取预设或内置默认（出厂 `default` 下是音频 + `local` 检索，
`local` 要 Exa、Tavily 的 key 或一把 `GEMINI_FREE`（用 Gemma 4 搜），哪样都没有就不检索，见[`env.md`「最小可用组合」](env.md#最小可用组合)），知识库照常读，也可以 `update`。要纯文本、不检索，用[快速粗纠](#快速粗纠agy-fast-与-fast)的两个预设，
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

**出厂的 12 格**（格式为 `思考档 / 模型组 / prompt 变体`）：

| 任务组 | quality | intermediate | efficiency |
| --- | --- | --- | --- |
| correction-mm | medium / correction-capable / 强模型那套 | medium / correction-capable / 强模型那套 | **low** / correction-basic / lite 那套 |
| correction-text | medium / correction-capable / 强模型那套 | medium / correction-capable / 强模型那套 | **low** / correction-basic / lite 那套 |
| research | **high** / research-default / 只有一套 | **high** / 同左 | **low** / 同左 |
| knowledge | **high** / knowledge-capable / 只有一套 | **high** / 同左 | **low** / 同左 |

两条规律：**efficiency 一律 low**（最省的形态）；**准备任务（`research`）与知识更新（`knowledge`）到 intermediate 都是
high**。它们的结果被后面全部复用（准备任务写的背景每个纠错窗口都读，知识更新直接写库），换档
换的是模型，不应顺带砍掉思考深度。intermediate 那一格是**继承**来的，不是写死的：回落朝上。

low / medium / high 到各家参数的换算见[接自己的模型「模型事实表」](custom-models.md#模型事实表catalog)的 `thinking` 一行。

### 会话档位（只对 agent 生效）

一格绑到本机 agent（Codex / Claude Code / agy 等）时，`agent_session` 决定**同一次运行
里这些调用如何共用会话**。与 thinking 一样写在预设上、按 `"任务组/难度"` 索引，未填则继承
`per-window`：

```toml
[llm.presets.my.agent_session]
"correction-text/quality" = "api"
```

| 档位 | 一次调用是什么 | 什么时候选它 |
| --- | --- | --- |
| `api` | 每次调用一个全新会话，整包 prompt 重发 | 想要每窗完全独立、可复现 |
| `per-window`（默认） | 一个窗口连同它的格式修复重试共用一次会话 | 出厂默认，不必动 |
| `resume`（实验） | 整次运行一条会话，跨窗口续上下文 | 只用于实验：历史没有上限，长素材会撑满上下文 |
| `pseudo-conversational`（已冻结） | 整次运行**一个常驻 CLI**：它做完一个窗口就向 FineSub 要下一个 | 不建议：保留现状、不再改进，试跑里比 `resume` 贵 |

- 带音视频的调用**在任何档位下都单独开一次会话**，不进跨窗口的会话或常驻 CLI。
- `pseudo-conversational` 下 token 按整个会话记一笔：任务报告里 agent 那一行的调用次数按窗口算、token 按会话算，逐窗看到 0 token 不等于没花；
  CodeBuddy 的 `hy4flash-free` 在这一档里自动转了付费时，记账也分不出这一次花的是免费额度还是积分。

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
加 `--verbose` 时，取自预设的开关会打印出来；事后想查，看任务的 `<名字>.llm-artifacts/resolved-switches.json`（见[运行产物](outputs.md#一次运行留下什么)）。

### 快速选模型：格子可以直接绑一个模型

当需要指定「这一格固定使用某个模型、不做回退」时，无需先创建只含一个成员的模型组，直接将格子
绑定到**模型名**即可：

```toml
[llm.presets.my-deepseek.bindings]
"correction-text/quality" = "my-corr"        # 一个模型组
"research/quality"        = "ds-flash"       # 直接一个模型
"knowledge/quality"       = "local-codex-gpt-6_2-sol"
```

FineSub 会自动把它包成一个单成员模型组。这也是使用 Codex / Claude
Code / CodeBuddy 的办法：出厂预设只是没绑。能写的名字见下面的[出厂的模型名](#出厂的模型名target)；
你在 `model_catalog.toml` 里自己加的模型也按表名得到同名的模型名（见[接自己的模型](custom-models.md)）。
搜不搜不看名字：模型能联网搜索、这次运行又是 `--llm-retrieval native` 时，同一个模型名就会用它那家的搜索工具。
例子里绑的都是只读文字的模型，所以这份预设还要像上一节那样写 `[llm.presets.my-deepseek.defaults]` 的 `media = "text"`（为什么见[媒体开关](#媒体开关---llm-media)）。

**同名规则**：一个名字既是模型组又是模型名时用模型组，所以以后新增了同名的模型，也不会悄悄改变你的绑定。
只有一处例外：在 `[llm.preferred_targets]` 与 `--llm-model` 里写的名字，既是一个模型组、又是出厂的 `gemini-` 开头的几个模型名或
`conversational-agent` 之一时，报「名字有歧义」的错。

### 出厂的模型名（target）

可以写进格子、模型组、`--llm-model` 与 `[llm.preferred_targets]` 的名字。「要什么」一列是这个模型能用的前提；
各模型好不好用见[模型选择「各后端的使用体验」](models.md#各后端的使用体验)。

「能力档」是出厂给每个模型标的档（极低 / 低 / 中 / 高，一格里有几个模型时按名字的顺序列），只用来在启动时提醒你某一格绑了低于下限的模型
（见[启动时那几条告警怎么读](#6-启动时那几条告警怎么读)），不参与选模型。「音视频」一列写「能」的才听得了音频、看得了视频，其余只读文字。

| 模型名 | 是什么 | 能力档 | 音视频 | 要什么 |
| --- | --- | --- | --- | --- |
| `gemini-free-3_8-flash`、`gemini-free-3_6-flash` | Gemini 3.8 / 3.6 Flash 的免费档，出厂纠错、准备任务与知识更新的主力 | 中 | 能 | `.env` 里的 `GEMINI_FREE` key（[环境变量与 API Key](env.md#geminigoogle-ai-studio)） |
| `gemini-free-4-hydrogen` | Gemini 4 Hydrogen 的免费档（对标 Flash 系列，限额与 3.8 Flash 相同、额度另算），出厂排在纠错组（含 `agy-capable`）的首位、准备任务与知识更新组的 3.8 Flash 之后 | 中 | 能 | 同上 |
| `gemini-free-3_5-flash-lite`、`gemini-free-3_1-flash-lite` | 免费档的 lite 模型，前者是 `efficiency` 档的纠错模型 | 低 | 能 | 同上 |
| `gemini-free-2_5-flash` | 免费档里能用 `native` 联网搜索的 Flash；出厂的 `gemini-native-search` 组用它 | 低 | 能 | 同上 |
| `gemini-free-gemma-4-31b` | Gemma 4 31B，出厂只在 `local` 检索里用；输入上限只有 16000，绑进纠错等格子会在启动时报错 | 极低 | 否 | 同上 |
| `gemini-paid-3_8-flash`、`gemini-paid-3_5-flash-lite` | 同名模型的付费档，出厂排在各组末尾 | 中 / 低 | 能 | `.env` 里的 `GEMINI_PAID` key |
| `local-codex-gpt-6-luna`、`local-codex-gpt-6_1-terra`、`local-codex-gpt-6_2-sol` | 经 Codex CLI 调 GPT-6 Luna / GPT-6.1 Terra / GPT-6.2 Sol | 中 / 中 / 高 | 否 | 装好并登录 Codex CLI（[本机 agent](agent.md)） |
| `local-claude-opus`、`local-claude-sonnet`、`local-claude-haiku` | 经 Claude Code 调 Opus / Sonnet / Haiku 的当前最新版 | 高 / 高 / 中 | 否 | 装好并登录 Claude Code |
| `local-agy-gemini-3_8-flash`、`local-agy-gemini-4-hydrogen`、`local-agy-gemini-4-neon` | 经 agy 调 Gemini 3.8 Flash / Gemini 4 Hydrogen / Gemini 4 Neon；出厂 agy 组用 Hydrogen，Neon 紧跟它当故障替补，3.8 Flash 不在出厂组里 | 中 / 中 / 高 | 能 | 装好并登录 agy（[agent.md「需要装什么」](agent.md#3-需要装什么)） |
| `local-agy-sonnet-5_5`、`local-agy-opus-5_5` | 经 agy 调 Claude Sonnet 5.5 / Opus 5.5；Opus 不在任何出厂模型组里 | 高 | 否 | 同上 |
| `local-dsh-deepseek-v4_1-flash`、`local-dsh-deepseek-v4-pro` | 经 dsh 调 DeepSeek V4.1 Flash / V4 Pro，只能绑按窗口跑的格子 | 中 | 否 | 装好 dsh、配好 DeepSeek key（[agent.md 4.1](agent.md#41-dshdeepseek-harness只能按窗口跑)） |
| `local-codebuddy-hy4flash-free`、`local-codebuddy-hy4flash`、`local-codebuddy-deepseek-v4_1-flash` | 经 CodeBuddy 调混元 4 Flash（免费线 / 付费）/ DeepSeek V4.1 Flash；只有 `hy4flash-free` 免费（活动到 2026-12-01），它的当日免费额度用完默认转付费的 `hy4flash`（关法见 [agent.md 4.2](agent.md#42-codebuddy模型清单跟着账号走)） | 低 / 低 / 中 | 否 | 装好并登录 CodeBuddy Code CLI（[agent.md 4.2](agent.md#42-codebuddy模型清单跟着账号走)） |
| `conversational-agent` | 交给一个已经开着的 agent 会话（已冻结） | 中 | 否 | 见 [agent.md 5.1](agent.md#51-让正在使用的-agent-直接处理任务conversational) |

各模型的输入、输出上限与思考参数的出厂原值，以及怎么改，见[接自己的模型「模型事实表」](custom-models.md#模型事实表catalog)。

### 优先模型：一行说「就用它」，不改预设

当希望指定某个模型、但不想为此编写整套预设时，可使用该表。取值可以是**模型组**或**单个模型名**：

```toml
[llm.preferred_targets]
default  = "my-corr"          # 所有格子换绑到 my-corr 组
research = "ds-flash"         # 这个任务组钉单模型(按任务组的写法覆盖 default)
```

语义为**整体换绑，不保留回退链**：被固定的模型组/模型无法处理的调用（例如纯文本模型遇到带
音视频的窗口，或要求联网检索的任务）会**明确失败**，而不是静默切换到其他模型——既然指定了该
模型，失败也应清晰可见。若因此导致某个 `-mm` 格子无可用模型，启动时会给出警告。

- 值先查模型组、再查模型名，同名时按上面的[同名规则](#快速选模型格子可以直接绑一个模型)。任务组名必须是真的
  (`correction-mm` / `correction-text` / `research` / `knowledge`)，写错直接报错并列出可选值。
- **对激活预设与 default 预设均生效**——未绑定的格子会回落到 default 预设，若优先模型恰好在这些
  格子上无法使用，则属于较难发现的问题。
- 修改该表与换预设一样，已完成的窗口照用（见 [§7 什么会作废已完成的进度](#7-什么会作废已完成的进度)）。

**命令行选预设**：`--llm-preset <名>` 只对这次运行生效，优先于 `config.toml` 的 `[llm] preset`；不存在的名字直接报错并列出可选的。
主命令 `finesub`（整批一个；manifest 行里不能写）、`finesub knowledge` 的 `repair` / `verify` / `ingest`、`finesub knowledge-update` 都收；
从源码运行时，单独跑 LLM 层（`python -m finesub.llm.correction_translation`）与参考资料导入也收。换预设与在配置里换一样：只作废还没交付的那次调用的断点。

**不改文件的运行时版本**：`--llm-model [任务组=]值`（可重复；不写 `任务组=` 时等于上表的 `default` 一行，换所有任务组；作用在选好的预设之上），
能写它的命令与上面 `--llm-preset` 一样。语义与上表相同，只对本次进程生效，
config.toml 不动；优先级按 `命令行[任务组] > 命令行[default] > 配置[任务组] > 配置[default]`
逐格解析：

- 裸写 `--llm-model X`（即命令行的 default）会盖过配置里的按任务组指定，这一次所有格子都用 X；
- 只写 `--llm-model research=X` 这样按任务组点名时，只换点到的任务组，配置里其余任务组的指定照旧生效。

与上一节的区别：**格子绑模型名**写在某个预设里、只换那一格；**优先模型**不改预设，一行换掉所有格（或一个任务组的各格）。
两者都不保留原来的回退链；值写模型组时，按组内顺序回退。

## 3. 模型事实表(catalog)

出厂模型的能力档与能不能听音视频在上面的[出厂的模型名](#出厂的模型名target)；限额与思考参数的原值、怎么给它们打补丁，见[接自己的模型](custom-models.md#模型事实表catalog)。

## 4. 出厂预设长什么样

| 任务组 | 能力下限 | quality 绑定的模型组 | 组成员（有序，括号里是能力档） |
| --- | --- | --- | --- |
| correction-mm | 中 | `correction-capable` | 免费 Gemini 4 Hydrogen（中）→ 免费 3.8 Flash（中）→ 免费 3.6 Flash（中）→ 付费 3.8 Flash（中） |
| correction-text | 中 | `correction-capable` | 同上 |
| research | 低 | `research-default` | 免费 3.6 Flash（中）→ 免费 3.8 Flash（中）→ 免费 Gemini 4 Hydrogen（中）→ 付费 3.8 Flash（中） |
| knowledge | 中 | `knowledge-capable` | 同 research |

其余任务组只绑 quality，低档自动向上继承；纠错三格都写了出来：capable、capable、lite。

免费档有两个要事先知道的代价（知识更新是尽力而为、`--llm-retrieval native` 要能联网的模型），见
[`models.md`](models.md)「免费档的两个代价」。

### agent 的两个出厂预设：`agy-hybrid` 与 `agy`

两个都把 agy 排进模型组，是 agent 模式的入口（详见 [`agent.md`](agent.md)）;
差别只有一处——**免费 API 在不在名单里**。

`[llm] preset = "agy-hybrid"`：免费 Gemini 优先，agy 作为后备；免费额度用尽后才消耗订阅。

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-capable` | 免费 Gemini 4 Hydrogen → 免费 3.8 Flash → agy Sonnet 5.5 → agy Gemini 4 Hydrogen → agy Gemini 4 Neon | 纠错的 quality + intermediate、knowledge/quality |
| `agy-basic` | 免费 3.6 Flash → agy Sonnet 5.5 → agy Gemini 4 Hydrogen → agy Gemini 4 Neon | research 的 quality |

免费 API 在前是有意的，两边的额度都不浪费。Sonnet 5.5 是**纯文本**，排在媒体成员之前同样有意：
文字任务优先给它，带剪辑的任务在能力过滤时自动跳过它、落到 agy 前置的那个 Gemini。

`--llm-retrieval native` 时，组里每个成员都用 agy 自带的搜索工具（出处核对不了，见[检索开关](#检索开关优先-native)）。

`[llm] preset = "agy"`：同样的格子，但名单中**没有任何 API 成员**。适用于已付费订阅、不愿再
管理 API key 的情况。它带一个开关默认值 `retrieval = "native"`，联网搜索用 agy 自带的工具，
因为 `local` 检索要 Exa / Gemini Free / Tavily 的 key。代价是出处核对不了（见[检索开关](#检索开关优先-native)）；要换回就在命令行写
`--llm-retrieval local`。

| 模型组 | 成员（有序） | 绑定的格子 |
| --- | --- | --- |
| `agy-only-capable` | agy Sonnet 5.5 → agy Gemini 4 Hydrogen → agy Gemini 4 Neon | 纠错的 quality + intermediate、knowledge/quality |
| `agy-only-basic` | agy Sonnet 5.5 → agy Gemini 4 Hydrogen → agy Gemini 4 Neon | research 的 quality |

> **与 `execution_policy = "agent-only"` 有何区别？** 两条路径可能产生相同的调用，但语义不同：
> policy 是一道**对所有预设都起作用的开关**，只能把某一类后端整个关掉、不能新增成员，且对**任何**激活的预设生效；预设
> 描述的是「这份名单中有哪些成员」。需要临时将某次运行限制为纯 agent 时使用 policy；需要长期按
> 订阅运行时使用 `agy` 预设。两者叠加使用也没有问题。

> agy 的 Sonnet 5.5、Opus 5.5 按 high/medium/low 原样接收思考档位（`thinking` 为默认映射）；Gemini 4 Hydrogen、Gemini 4 Neon 与 3.8 Flash 一样封在 `medium,medium,low`。
> `[llm].local_agent_reasoning_effort = "xhigh"` 对 agy 的各行按 `high` 发。
> Gemini 4 Neon 在每个出厂 agy 组里紧跟 Gemini 4 Hydrogen，接 Hydrogen 的超时、不可用、限速与其他临时失败。两者同一份额度：额度用完时 agy 的报错不明说，FineSub 要等 Hydrogen 与 Neon 通常各失败一次、再探一次才停用这份额度。
> Opus 5.5 不在任何出厂模型组里，要用就 `--llm-model local-agy-opus-5_5` 点名，或列进自己的模型组。

### 快速粗纠：`agy-fast` 与 `fast`

给很长的输入做一遍不过度思考的简单纠错，意思大致对、方便找亮点即可。它和「快速模式」（`--llm-fast`：短素材整段放进一个窗口，见[调参](tuning.md)）是两回事。用 `--llm-preset agy-fast` 或 `--llm-preset fast` 选，
两者的开关默认值相同：`difficulty = "efficiency"`（低思考、一步纠错、只用出厂默认风格；照常读知识库）、
`media = "text"`、`retrieval = "none"`（纯文本、不检索：`efficiency` 本身不钉这两个，所以预设自己写）、
`continuity = "parallel"`、`parallel_windows = 4`（4 个窗口同时跑）。命令行写了的照样优先，例如：

- 在它们下面加 `--llm-difficulty quality`：仍不搜索（不会落到要 key 的 `local`）。要 agy 的完整质量档（含自带搜索），改用 `--llm-preset agy`。
- 在 `agy-fast` 下面加 `--llm-media audio`：纠错改由 `agy-only-capable` 做（带音频的窗口由其中的 agy Gemini 接），仍不转到 API。

| 预设 | 给谁 | efficiency 格 | 其余格 |
| --- | --- | --- | --- |
| `agy-fast` | 有 agy 订阅的人 | 纠错与准备任务都只用 agy Gemini 4 Hydrogen，它出故障时接 agy Gemini 4 Neon（`agy-only-fast`）；Gemini 那份额度用完就停下报错，**不会转到 API** | 与 `agy` 相同 |
| `fast` | 只用 API 的人 | 纠错落到两个 3.5 Flash Lite（免费在前、付费在后）；准备任务与 `default` 相同（免费 3.6 / 3.8 Flash 与 Gemini 4 Hydrogen，再接付费） | 与 `default` 相同 |

4 个窗口是上限，实际并发还受 `[llm].local_agent_max_parallel`（agent，默认 4）或 API 的每分钟请求限速约束。

### dsh 的两个组：`dsh-capable` / `dsh-basic`

**没有预设会自动绑它们**，要用就在自己预设的 `[llm.presets.<名>.bindings]` 里显式绑：

| 模型组 | 成员（有序） |
| --- | --- |
| `dsh-capable` | DeepSeek-V4.1-Flash → V4-Pro |
| `dsh-basic` | DeepSeek-V4.1-Flash |

> **只能绑到按窗口执行的格子**，也不支持音视频；原因、安装与 key 的说明，以及经 dsh 接自己的网关怎么登记，见
> [`agent.md`「dsh」](agent.md#41-dshdeepseek-harness只能按窗口跑)。

### CodeBuddy 的两个组：`codebuddy-capable` / `codebuddy-basic`

同样**没有预设会自动绑它们**，要用就在自己预设的 `[llm.presets.<名>.bindings]` 里显式绑：

| 模型组 | 成员（有序） |
| --- | --- |
| `codebuddy-capable` | DeepSeek-V4.1-Flash |
| `codebuddy-basic` | hy4flash-free → hy4flash → DeepSeek-V4.1-Flash |

> **两个组都会扣积分**：`codebuddy-capable` 是付费模型；`codebuddy-basic` 打头的 hy4flash-free 免费额度用完默认转付费的 hy4flash，后面的成员也是付费模型。
> 只想用免费额度，就不绑组，直接绑 `local-codebuddy-hy4flash-free` 并关掉它的自动转付费。混元 4 Flash 能力判为「低」，所以只在 basic 组里。
> 免费活动 2026-12-01 结束；之后免费线若从账号的模型表里消失，`codebuddy-basic` 会在第一个成员上报错停下、不往下走，要改绑 `local-codebuddy-hy4flash`。
> 额度、费用、怎么关自动转付费与那条输出上限告警，见 [`agent.md`「CodeBuddy」](agent.md#42-codebuddy模型清单跟着账号走)。

## 5. 接自己的模型

已挪到单独一份：[接自己的模型](custom-models.md)——在 `model_catalog.toml` 里声明自己的 API 端点，再把它绑进预设。只想在出厂模型里挑一个用，见上面的[快速选模型](#快速选模型格子可以直接绑一个模型)与[优先模型](#优先模型一行说就用它不改预设)。

## 6. 启动时那几条告警怎么读

这几条告警在启动时按**当前激活的预设**检查后打印（出厂预设一条都不会触发）：

| 告警 | 触发 | 该怎么办 |
| --- | --- | --- |
| `成员 X 的能力档「低」低于下限「中」` | quality 那一格的组里有成员低于这个任务组要的能力档（见[出厂预设长什么样](#4-出厂预设长什么样)的「能力下限」列） | 确认是有意放的。另外两档只是定位，绑什么由预设自己定，不告警 |
| `成员 X 未声明能力档，按「高」计` | 这只是一条记录，**不是告警**（只进日志，`--verbose` 才上屏）| 不必处理。想让能力下限对它生效，就在 `model_catalog.toml` 里给它填 `capability` |
| `模型组 X 的成员 … 不一致，按其中最小的 N 切分（由 Y 决定…）` | 组里成员写的三个质量上限不同（没写的按默认计），这个组按最小的切分 | 想用大的，给组里每个成员都写上 |
| `规划包络由 X 决定 (…tokens…)，窗口数约 ×N` | 告警里的「规划包络」指一个窗口最多能放进多少输入；组里有成员能放进的量或输出上限比免费 Gemini 小 | 窗口按**组内最低**规划（谁应答还没定，必须保守），×N 是窗口数放大倍数。组里有小上下文模型时窗口会变多；想要大窗口，组里别放小模型 |
| `组内没有成员支持音频` | `-mm` 格里一个能听的都没有 | 带媒体的调用会在能力过滤后无候选可用；换模型，或把该任务改成 `text` |
| `target X 不支持视频，按 video->audio 降一级` | 运行中途的兜底检查触发 | 正常配置不该看到它。说明该格里混进了看不了视频的模型 |

## 7. 什么会作废已完成的进度

判据并非「配置是否一字不差」，而是**已完成的窗口在同样的输入、同一套窗口切法下是否仍然成立**。
若成立，则继续沿用，新配置仅作用于尚未完成的部分；因此一个任务里可以混着不同的模型、思考档位与
prompt 变体，每个窗口实际用了哪一套都记在任务产物里备查。

| 改动或状态 | 已完成窗口 | 已完成的准备任务 |
| --- | --- | --- |
| 换 `--llm-model`、预设（`--llm-preset`）、**模型组**、thinking、执行策略（`execution_policy`）、agent 参数 | 复用 | 复用 |
| 换识别用的 Whisper（`--model`） | 不受影响：识别产物还在时不重新识别，这个改动不生效；删掉识别产物让它生效时，按下面「`*-stable.json` 内容变化」那行 | 同左 |
| 改 `agent_session` 档位 | 复用 | 复用 |
| 改 `difficulty` 到 `intermediate` | 复用 | 复用 |
| 改 `difficulty` 到 `efficiency` | 复用 | 复用 |
| 改 `continuity` | 复用（`parallel→serial` 会告警：窗口之间往下传的提示从空开始重新积累） | 复用 |
| 改知识库内容、`--knowledge` | 复用 | 复用 |
| 改能力档 `capability`、任务组的 `min_capability`、显示名、预设名、注释 | 复用 | 复用 |
| 改 `model_catalog.toml` 里的三个质量上限 | 复用（窗口切法存在任务里，续跑沿用）；调小 `window_subtitle_tokens` 时，没跑完的窗按新上限预拆，同下面「窗口上限」那行 | 整份准备结果复用（不看分块上限）；分块的准备任务没跑完时改 `prepare_chunk_tokens`，范围变了的块重跑 |
| 改预设的开关默认值（`defaults`） | 与在命令行上改同一个开关相同（见上面各行） | 同左 |
| 重新下载同一份音视频（大小不变） | 复用 | 复用 |
| 改媒体开关、`--output-scale`、`token_scale`、窗口上限，或换成**上下文更小的模型组** | 复用；没跑完的窗口按现在的上限先切小 | 复用（若同时改 retrieval，则重跑） |
| 软件更新换了 prompt 版本、中途改风格（`--style`、`--extra-style`） | 复用（前后口吻可能不一；版本不同时告警一次） | 复用 |
| 改任务备注（`--extra-info`） | 复用 | ❌ |
| 改 `--test-profile`（测试开关：所有调用改用预设里那个便宜的测试模型） | ❌ | ❌ |
| `*-stable.json` 内容变化 | ❌ | ❌ |
| 窗口计划或 JSON 产物损坏、格式与当前版本不兼容、存下的模型回答当前版本读不了 | 这一层重做；被覆盖的旧文件先备份为 `*.invalid[.N]` | 同左 |
| `--no-resume` 或删除产物 | 按要求重新执行 | 同左 |

续跑时窗口只会拆小、不会合大：换成更大的模型后想要更少、更大的窗口，删掉
`<名字>.llm-artifacts\correction-window-plan.json`（已完成的窗口要重做）。

切档只换 prompt 变体、思考档位与（按出厂绑定）模型组，媒体与检索不随档位变，所以额度耗尽后显式切到
`efficiency` 时窗口与准备任务的结果都原样保留。任务记成纠错走三个 pass 的，没做完的窗按一步纠错的 prompt 纠错、告警一次。

预设只在启动时读一次：运行中途改了，下次运行才生效。

---

实现细节与设计取舍（开发者参考，可跳过）：运行时行为见 [`llm_harness_behavior.md`](../llm_harness_behavior.md)，
自定义端点的供应商差异见 [`provider-adapters.md`](../provider-adapters.md)。
