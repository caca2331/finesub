# 用本机 AI 订阅运行字幕处理（agent 模式）

FineSub 默认通过 Gemini 的 API 额度完成纠错和翻译。如果本机已安装并登录 **Codex CLI**、
**Claude Code** 或 **Antigravity**（下文简称 agy），可以让 FineSub 改用这些工具执行，消耗已有
订阅。Gemini 额度还用不用看预设：`agy` 只用订阅，`agy-hybrid` 先用 Gemini 免费额度、用完再接着用订阅（见下文）。

这就是 agent 模式。换的是执行纠错翻译的模型，字幕质量跟着所选模型走（各模型的使用体验见[模型选择](models.md#各后端的使用体验)）。

---

## 1. 适用场景

**适合：**

- Gemini 免费额度不够用，而本机正好有 Codex / Claude Code / agy 订阅。
- 素材多、运行时间长，不想受配额限制。
- 不方便配置 Gemini API key（`agy` 预设完全不需要它）。

**不适合：**

- 只是偶尔处理一两个短视频，默认的 API 路径更省事。

---

## 2. 启用方式

在 `config.toml` 中加一行（文件位置和创建方法见 [`resources.md`](resources.md)）:

```toml
[llm]
preset = "agy-hybrid"
```

这里配置的是「预设」，不是执行策略（见 [§5 进阶：执行策略 `execution_policy`](#5-进阶执行策略-execution_policy)，一般不用动）。预设决定每类任务由谁执行；出厂
预设中只有 `agy-hybrid`、`agy` 与 `agy-fast` 把本机工具排了进去（`agy-fast` 是长输入的快速粗纠：便宜、并行、纯文本地过一遍，见 [`model-routing.md`](model-routing.md)「快速粗纠」）。装好并登录 agy 后，写入这一行即完成启用。

`agy-hybrid` 先用 Gemini 免费额度，用尽或失败后才轮到 agy 订阅；组里谁先谁后、为什么这样排，见
[`model-routing.md`「agent 的两个出厂预设」](model-routing.md#agent-的两个出厂预设agy-hybrid-与-agy)。

**不想用 API、只走订阅**就改写 `preset = "agy"`：同样的格子（任务组 × 难度档的一格，如「纠错 / quality」，预设按格子指定模型，见[模型路由](model-routing.md)开头），但名单里没有任何 API 成员
（连 `--test-profile` 这个测试开关换上的便宜模型也是 agy 的）。

**只装了 Codex 或 Claude Code**（未装 agy）时，出厂预设中没有它们的绑定，需要自行指定
模型。照抄以下配置即可：

```toml
[llm]
preset = "my-agent"

[llm.presets.my-agent]
name = "本机 Codex"

[llm.presets.my-agent.bindings]
"correction-text/quality" = "local-codex-gpt-6-luna"
"research/quality"        = "local-codex-gpt-6-luna"
"knowledge/quality"       = "local-codex-gpt-6_2-sol"   # Luna 不推荐用于知识更新（见 models.md）

[llm.presets.my-agent.defaults]
media = "text"
```

换成 Claude Code 时三行都换，写 `local-claude-sonnet`（knowledge 那格别用 Haiku，理由同 Luna，Sonnet 在 Claude Code 上也没测过知识更新；另有 `local-claude-opus`、`local-claude-haiku`，都指
Claude Code 当前的最新版）。加 `--llm-retrieval native` 时模型会用自己的搜索工具，模型名不用换。
示例只绑了 `/quality` 那几格：没绑的 `intermediate`、`efficiency` 格会在同一预设里向上找，用到同一个模型，所以默认档位不用另写。
这两家不能听音频，所以示例用预设默认值把媒体固定成文字（等于每次都加 `--llm-media text`；不写会怎样见[模型路由「媒体开关」](model-routing.md#媒体开关---llm-media)）。想确保一次都不走 API，在 `[llm]` 下写 `execution_policy = "agent-only"`（见[执行策略](#5-进阶执行策略-execution_policy)），这时带音视频的调用会直接报错。如需按难度分档、或让失败回退到 Gemini 等更细的控制，见
[`model-routing.md`](model-routing.md)。

---

## 3. 需要装什么

按已有订阅选择安装，不需要都装（工具名链到各自的官方安装说明）：

| 工具 | 负责 | 登录方式 |
| --- | --- | --- |
| agy（[Antigravity](https://antigravity.google) 的命令行工具） | 文字任务 + 带音频/视频的任务，出厂 `agy` / `agy-hybrid` 预设使用 | 在 agy 里登录 Google 账号 |
| [Codex CLI](https://github.com/openai/codex) | 文字任务（纠错、翻译、准备任务），需自行配置绑定；出厂的 GPT-6.2 Sol 要 0.169.1 或更新 | `codex` 自带登录；用 ChatGPT 账号的，再给 FineSub [单独登录一次](#codex-用-chatgpt-账号时给-finesub-单独登录一次) |
| [Claude Code](https://code.claude.com/docs/en/setup) | 同上 | `claude /login` |
| dsh(DeepSeek Harness) | 只做文字任务，且只能按窗口跑（见 4.1） | 环境变量或 `~/.dsh/.env` 里的 `DEEPSEEK_API_KEY` |
| CodeBuddy Code CLI | 文字任务，需自行配置绑定（见 4.2） | 独立安装 CLI，在 CLI 中登录 |

**agy 装好了没有**：agy 是 Antigravity 的命令行工具，终端里 `agy --version` 能打出版本号就是装好了（FineSub 验证过的是 `1.5.7`；
更旧的读不到任务文件与音视频，调用会失败，每次运行提醒一次，请先更新 agy）；再运行 `finesub agent-ping --tier LOCAL_AGY`，它向 agy 的每个模型发一句极短的请求，
能答就说明登录与额度都没问题（见[检查额度是否耗尽](#7-检查额度是否耗尽)）。agy 花的是登录的 Google 账号的 Antigravity 额度（分几份计见[§7](#7-检查额度是否耗尽)）；本项目用的是 Pro 与 Ultra 订阅，免费账号能不能在命令行里用、够跑多少，没试过。

> **都能联网**（`--llm-retrieval native`，使用模型自带的搜索工具）。Codex / Claude Code /
> CodeBuddy 的格子照常绑，加上 `--llm-retrieval native` 就会用自己的搜索工具，模型名不用换（见 [`model-routing.md`](model-routing.md) 的
> 「快速选模型」）；`agy` / `agy-hybrid` 预设开箱即用。agy 搜到的出处核对不了，见[`model-routing.md`「检索开关」](model-routing.md#检索开关优先-native)。
>
> 提示：`config.toml` 的 `[api] proxy` **不传给这些工具**：它们各自登录、自己发请求，要给它们单独配代理，按各自 CLI 的文档设置。
> 系统里本来就设着的代理与证书相关的环境变量（`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、
> `NO_PROXY`、`SSL_CERT_FILE` 等）会原样传给它们。FineSub 自己用的密钥（如 `OPENAI_API_KEY`）不会传；
> 你在 Codex 配置里给 provider 点名的变量会交给 Codex。
>
> Codex 走的是**你在 Codex 自己的配置里选的 provider**（`~/.codex/config.toml` 的 `model_provider`，
> 例如本机代理）。provider 指向本机地址而那个端口没在监听时（代理没开），FineSub 会在调用前把 Codex
> 标为不可用、改用别的后端，并说明原因；不会悄悄改走直连登录号。

### Codex 用 ChatGPT 账号时，给 FineSub 单独登录一次

```powershell
finesub agent-codex-login
```

从源码运行 FineSub 的（仓库里 `python -m finesub.pipeline`），在仓库根运行 `python -m finesub.llm.agent.codex_login`，下面的 `--status`、`--logout` 也加在它后面：
仓库里没有 `finesub` 命令，另装的 `finesub` 登进的是安装版自己的目录，源码运行的调用用不上。

运行后照 Codex 的提示在浏览器里授权（没有浏览器时加 `--device-auth`，改用设备码），一次就够。

- **这是另一份登录**：存在 FineSub 自己的目录里（从源码运行时是仓库根下的 `.agent-codex-home`，安装版在 `user-data` 下），
  和你自己 `~/.codex` 的登录互不影响：在这里登录、退出，不会把你自己的 Codex 登出。
- **为什么要它**：Codex 每次调用都会读它所在目录里的 `AGENTS.md`，没有开关。不登录也能跑，只是 FineSub 的 Codex 调用会从你自己的 `~/.codex` 发出，
  你的个人规则（`~/.codex/AGENTS.md`）会进 FineSub 的调用，每次运行提醒一次；FineSub 在开场请模型把它放在一边、只按任务回答，但这只是请求。
- **查看与撤销**：`finesub agent-codex-login --status` 看登没登录、目录在哪；`--logout` 退出。`finesub agent-ping --tier LOCAL_CODEX` 的每一行也注明这次用的是哪个目录。
  登录过期或被撤销时，Codex 调用的报错与每次运行的提醒都会给出这条命令。FineSub 查不出这份登录的状态时（`codex login status` 没答上来），这次照样从你自己的 `~/.codex` 跑，并提醒一次原因。
- **不需要登录的情形**：Codex 的连接靠环境变量里的 key（你在 Codex 配置里接了别的 provider、它点名了 key 变量）时，FineSub 的调用本来就不碰你的规则，命令会直接说不需要。
- **目录里会攒下什么**：FineSub 的 Codex 会话记录（每个任务每种会话约一条，用来让后面的窗口读到缓存）。`agent-clean` 不删它们；
  要清就把这个目录里的 `sessions` 删掉，或整个目录删掉再登录一次。安装版的 `finesub uninstall` 默认保留这个目录，加 `--purge-user-data` 才删；仓库里的那个目录要删就自己删。

其余几家的个人规则文件都不会进 FineSub 的调用：Claude Code、dsh、agy（`~/.gemini/GEMINI.md`）与 CodeBuddy（`~/.codebuddy/CODEBUDDY.md`）。
CodeBuddy 要 2.178.1 或更新才挡得住；更旧的版本照样读这个文件，FineSub 每次运行提醒一次更新 CLI，并在开场请模型把它放在一边。

---

## 4. agy：不建议开视频

agy 是几家里面唯一支持音频和视频的后端，但**建议只用音频**。开视频后，相同长度的素材会消耗
明显更多的输入 token，切出的片段更小、调用次数更多；而字幕纠错真正依赖的信息绝大部分在
**声音**里，画面带来的提升有限。除非素材必须依靠画面分辨（屏幕上出现的人名、专有名词字牌
之类），否则默认的音频模式已足够。

---

## 4.1 dsh(DeepSeek Harness)：只能按窗口跑

`npm install -g @deepseek-ai/dsh`（需要 Node ≥ 24.9）。FineSub 验证过的是 `0.2.2-rc.1`（npm 的 `latest` 标签）；更旧的版本每次运行会提醒一次「比验证过的版本旧」，
而且可能跑不起来（它们独有的插件会被当成未经核对的关掉），请升级。它与另外几家有几处实质差异，安装前请确认：

- **只能绑到「按窗口」的格子。** FineSub 只从命令行给 dsh 任务，一整窗字幕无法塞进一条命令行，因此
  FineSub 仅使用它的工具协议：模型自行取任务、读正文、提交答案。绑定到 `api` 或 `resume`
  档位会**直接报错**（错误信息会说明原因），不会静默降级。
- **不支持音视频。** 纠错窗需要听音频时不要绑定 dsh。
- **无法报告 token 用量。** dsh 运行完毕只打印最终答案，不输出统计信息，因此任务报告中 dsh
  行的 token 数为空。这不是 bug，而是它没有可报告的数据。
- **不能续会话。** 每次调用都是全新会话。
- **不要接限速的端点。** 一整窗字幕是数百行的任务，慢端点会在
  `[llm].local_agent_timeout_seconds`（出厂 28 分钟）上超时；且 dsh 失败时不打印任何内容，看起来
  就像「dsh 不可用」。实测（2026-08，用的是上一代 DeepSeek V4 Flash；出厂的 dsh 模型现在是 V4.1 Flash）：正常速率下处理一窗 270 条约 13 分钟；在限速到
  ≈5 tok/s 的免费预览端点上，同一窗两次均超时、无任何输出。

**key 放在环境变量 `DEEPSEEK_API_KEY` 里**（设成用户环境变量即可），FineSub 每次调用 dsh 都会把它传过去。
也可以写进 `~/.dsh/.env`（Windows 下为 `C:\Users\<你>\.dsh\.env`）：

```text
DEEPSEEK_API_KEY=sk-...
```

FineSub 用的是 dsh 的 `headless` 配置，它不读 dsh 网页界面存的 key（`~/.dsh/.credentials.yaml`），只读环境变量和
`.env`；网页界面里能用、FineSub 报 `MISSING_CREDENTIAL`，多半就是这个原因。

**联网搜索也用这把 key。** dsh 的 `web_search` 由自带的 DeepSeek 搜索插件提供，读的同样是 `DEEPSEEK_API_KEY`；
模型接的是别家网关时（见下），搜索仍走这把 DeepSeek key。⚠ 每次搜索会在这把 key 上扣少量费用；没有有效的
DeepSeek key 则无法搜索。同一档里模型还能用 `web_fetch` 抓取网页正文，这一项不需要 key。两项都只在 native 检索
（`--llm-retrieval native`）时开放。

**接入自己的网关**：dsh 支持 OpenAI 兼容的自定义 provider，在 `~/.dsh/profiles/headless/cordis.patch.yml` 的
`llm-pi-ai` 条目里声明（写法见 dsh 的 `dsh-llm-pi-ai` 说明），它的 key 写进 `~/.dsh/.env`、名字与声明里的 `apiKeyEnv` 一致
（FineSub 只把 `DEEPSEEK_API_KEY` 这一个环境变量传给 dsh）。出厂只带 dsh 自带的 `deepseek-official` 这一路，你声明的别家服务只有你自己有，
所以要在 FineSub 这边登记：在 `user-data` 目录（与 `.env`、`config.toml` 同一处，见[资源与大文件「两类数据，两个去处」](resources.md#两类数据两个去处)）的
`model_catalog.toml`（模型事实表：每个模型能做什么、限额多少）里写一个表，它会**自动**成为同名的模型名、可以直接绑，
无需修改其他文件（表的写法见[接自己的模型](custom-models.md)）：

```toml
[models.my-dsh-gateway]
provider_tier = "LOCAL_DSH"
api_model_id = "my-provider/my-model"
max_input_tokens = 194000
supports_native_search = false   # `LOCAL_DSH` 这一行不写就是 true：native 检索时会用 dsh 的搜索（要 DeepSeek key，扣少量费用）
```

- `api_model_id` 的 `my-provider` 必须是在上面那个 `llm-pi-ai` 条目的 `providers` 下声明过的名字。
- **思考档**照这一行的 `thinking` 发，和别的模型一样：不写就把 high / medium / low 原样发出，这家服务用别的词就写三个映射值
  （见[接自己的模型「`thinking`」](custom-models.md#thinking)）。dsh 只收这个模型声明过的档：在 `llm-pi-ai` 条目里给模型写
  `reasoningEfforts`，列出 FineSub 会发的那几档。没写的话，dsh 认为它不支持思考档，FineSub 会不带思考档重跑这次调用、
  用你在 dsh 里给这家服务设的默认档，并提醒一次；同一次运行里之后的调用直接不带。想一直用 dsh 里的设置，把 `thinking` 写成 `false`。
- Codex、Claude Code、agy、CodeBuddy 的新模型也一样，写一个表就能用（agy 与 CodeBuddy 还要写 `quota_pool`）。

---

## 4.2 CodeBuddy：模型清单跟着账号走

**安装与登录**：`npm install -g @tencent-ai/codebuddy-code` 装独立的 CodeBuddy Code CLI（WorkBuddy 桌面端自带的那个不算），
再运行 `codebuddy` 登录：中国站选 **Chinese Site**，国际站选 **International Site**。出厂的三个模型是中国站的，
国际站用户要按自己账号能用的模型在 `model_catalog.toml` 里加行（见[接自己的模型](custom-models.md)）。要固定站点，
设环境变量 `CODEBUDDY_INTERNET_ENVIRONMENT`（中国站 `internal`、国际站 `external`），在那个站点登录，再从带着这个变量的终端运行 FineSub。
FineSub 验证过的 CLI 是 `2.164.0`，更旧的照常能用，每次运行提醒一次。登录失效时 FineSub 换组里下一个模型，并提示你重新运行 `codebuddy` 登录。

**能用哪些模型看账号，不看版本**：`codebuddy --help` 列的那张表不算数；随便给一个不存在的 `--model`，报错里列的才是你这个账号能用的。
绑了账号够不到的模型，FineSub 直接报错并贴出服务端原话。

**选哪个**：出厂预设没有绑它，照下面建一个自己的预设（模型名见 [`model-routing.md`「出厂的模型名」](model-routing.md#出厂的模型名target)；
想失败时在组内往下换，就绑出厂的两个组，见同一份的「CodeBuddy 的两个组」）：

```toml
[llm]
preset = "my-codebuddy"

[llm.presets.my-codebuddy]
name = "本机 CodeBuddy"

[llm.presets.my-codebuddy.defaults]
media = "text"

[llm.presets.my-codebuddy.bindings]
"correction-text/quality" = "local-codebuddy-deepseek-v4_1-flash"
"research/quality"        = "local-codebuddy-deepseek-v4_1-flash"
```

**目前推荐 `deepseek-v4.1-flash`**：出厂三行里唯一能力判为「中」、够得上纠错门槛的一行，单次输出上限也最大（128,000）；出厂已把它的纠错格压到 `low`，
不会陷进长时间思考。`hy4flash`（腾讯混元 4 Flash）更便宜，但能力判为「低」，绑进纠错的 `quality` 格会看到能力低于下限的告警，适合调查这类门槛低的任务；
2026-12-01 之前它有一条免费线 `hy4flash-free`。

**额度与费用：一个登录背后有两份额度。** 免费的 `hy4flash-free` **自己一份每日额度**；`hy4flash`（每次调用按 x0.03 的倍率扣积分）、
`deepseek-v4.1-flash`（x0.11）**花同一份积分余额**（倍率是 2026-11-16 厂商配置里的值）。
所以 `hy4flash-free` 用光只影响它自己，积分用光时那两个一起停、报错。

- ⚠ **`hy4flash-free` 的当日免费额度用光后，默认自动转付费、任务不中断**：CLI 把这次会话交给 `hy4flash` 跑完，跑完打一条提示（这条提示等于一张小额账单）；
  持续限流时也可能这样切。不想自动扣费，在 `user-data` 目录（与 `.env`、`config.toml` 同一处，见[资源与大文件「两类数据，两个去处」](resources.md#两类数据两个去处)）的 `model_catalog.toml` 里写这两行
  （`""` 表示「没有付费版可转」，这个模型的其余设置照出厂）：

  ```toml
  [models.local-codebuddy-hy4flash-free]
  fallback_model = ""
  ```

  之后 `hy4flash-free` 用光就停用它两小时、换组里下一个模型，绝不扣积分（但出厂的 `codebuddy-basic` 组里下一个就是付费的 `hy4flash`，
  只想用免费额度就直接绑这一行、不绑组）。CLI 太旧、不支持自动转付费时会提醒一次，`hy4flash-free` 就只用免费额度，
  用光后每个任务多一次失败的调用，不影响结果。
- ⚠ **免费活动 2026-12-01 结束**。之后如果这个名字从你账号的模型表里消失，绑着它（或绑着 `codebuddy-basic` 组）的调用会以「账号够不到这个模型」报错，
  **不会**自动换到付费的 `hy4flash`；到时把绑定改成 `local-codebuddy-hy4flash`，或等 FineSub 更新出厂的组。

**会遇到的告警与限制**：

- DeepSeek 别用最高思考档：`high` 下会一直想、不交答案，撞上 `[llm].local_agent_timeout_seconds`（出厂 28 分钟）；
  出厂的纠错格已压到 `low`。`hy4flash` 只有 `high` 一档，三格都是它。想全局压档写 `[llm].local_agent_reasoning_effort = "low"`。
- 不支持音视频；联网只能搜索、不能抓取网页正文，搜到的来源网址照常记录。

这些行为的实测与设计取舍见 [`llm_local_agent.md`「12.1.5 codebuddy」](../llm_local_agent.md#1215-codebuddy第五家-driverclaude-code-的分支2026-09-04-接线)（开发者参考，可跳过）。

---

## 5. 进阶：执行策略 `execution_policy`

`execution_policy` 是一道**闸门**，只能限制后端，不能新增后端：

| 取值 | 含义 | 使用场景 |
| --- | --- | --- |
| `agent-text-preferred` | 出厂默认，不限制任何后端，按预设顺序执行 | 默认使用 |
| `api-only` | 屏蔽全部本机工具，只走 API | 临时只用 API，又不想改预设 |
| `agent-only` | 屏蔽全部 API，只走本机工具 | 确保不消耗任何 Gemini 额度 |

由于它只能限制，设置它本身不会启用本机工具：出厂 `default` 预设中没有任何本机工具，
`agent-text-preferred` 也不会改变这一点；`agent-only` 搭配 `default` 预设会因无可用后端而直接
报错。要使用本机工具，先按 [§2 启用方式](#2-启用方式) 配置预设。

**同时运行的 agent 数量**：`config.toml` 的 `[llm].local_agent_max_parallel`（默认 4，下限 1）
限制本机同时活跃的 agent CLI 进程数——这是整台机器与订阅的物理上限，对所有本机工具共用，
与「一个任务开几个并行窗口」(`--llm-parallel-windows`)和「批处理同时跑几个任务」
(`--max-parallel-tasks`)是三个不同的旋钮。

怎么调看你的流量池，没有一个能测出来的常数。各家 agent 对并发的容忍度其实很高，并发高
不会被封；真正的代价是两条——**调太高时，任务一多 5 小时额度会被快速耗尽，大量会话
执行到一半断掉，已花的 token 全部白费**；而省下的时间边际递减，token 效率（缓存命中、窗口之间往下传的提示）
和上下文完整性受到的损失却边际递增。出厂默认每个任务 1 个窗口、同时 2 个任务：任务之间并行
（互不共享上下文，没有质量代价），任务之内保持串行（保留窗口之间往下传的提示与缓存收益）；额度富余
再逐档往上调。

**这三个旋钮不是彼此独立的**：每个任务在**起步**就占住一格保底 agent 槽，所以
`local_agent_max_parallel` 一定要不低于 `--max-parallel-tasks`，否则多出来的任务会卡在起步处
等前一个任务整体跑完（连它本可以先做的纯 API 阶段也一起等），看起来像卡死。

**交稿之后提前结束**：`config.toml` 的 `[llm].local_agent_exit_on_accept`（默认 `auto`）让 agent 交稿被接受后立刻结束，
不再为一句「已提交」多发一次模型请求；提前结束时用量按它每次请求自己报的数补全。`auto` 只对补得了用量的 agy、CodeBuddy、
Claude Code 这样做；Codex 只在最后报一次用量，所以默认不提前结束——`on` 让它也提前结束，代价是这次调用的用量记为未知、
不进任务报告的 token 合计；`off` 一律不提前结束。

---

## 5.1 让正在使用的 agent 直接处理任务(conversational)

> 这条路已冻结（2026-10-01）：照常可用，但不再改进。

如果本机已有一个正在运行的 Claude Code / Codex 或其他 agent 会话，可以让 FineSub 不再启动新
的 CLI，而是把纠错任务交给该会话：在路由表中将某个任务组绑定到 `conversational-agent` 模型组
（该组只能独占使用）。运行到第一个交给它的调用时，FineSub 会打印一条提示；在你的 agent 里执行
`finesub agent-join`，把打印出的说明交给它即可。只有纯文本调用会走这条路，所以整次运行要用
`--llm-media text`（出厂默认 `audio`），或只绑定不带媒体的格子。

- **到场与掉线**：提示出现后要在一小时内执行 `finesub agent-join`，否则报错停止。接手之后 agent 不要长时间一条
  `finesub agent-task` 命令都不执行：静默超过 `[llm].local_agent_timeout_seconds` 加 2 分钟（默认 30 分钟）就当它离开了，
  任务退回去重新等。这时再执行一次 `finesub agent-task status` 就能取回，写好的答案照样提交，做过的不丢；会话断了就重新
  `finesub agent-join`。要更长的余量，调大 `local_agent_timeout_seconds`。同一个任务被接手又放下第 3 次之后不再等，报错说明原因。
- **一条回复写不完**：答案通过文件提交（`finesub agent-task submit --text-file <文件>`），可以分几次写进同一个文件再提交；
  仍嫌大就调小 `config.toml` 的 `[chunking] max_window_subtitle_tokens`（默认 10000），窗口切得更碎。

如需免去守候，**分两步运行**：同一命令执行两次，第一次停在语音部分，第二次继续到纠错。

```powershell
# 第一步:完成耗时的语音部分,不涉及 agent
finesub data\input.wav --language ja --stage raw-srt

# 第二步:人在电脑前时,原样重跑第一步的命令,只改 --stage 并加上 --llm-media
finesub data\input.wav --language ja --stage final-srt --llm-media text
```

第二步根据产物是否已存在跳过语音部分，只执行纠错，提示几秒内出现。`--llm-media text` 不可
省略：出厂默认是 `audio`，纠错窗会带音频，而带媒体的调用不会交给 agent（见上文）。其余开关
（`--language`、`--extra-info` 等）两次需保持一致。

## 6. 异常处理

- **工具未安装 / 未登录 / 版本缺少能力**：默认执行策略下切换到预设中的下一个（`agy-hybrid` 预设里就是
  Gemini）继续运行，并打印**一行** `Warning:` 说明原因：未安装、安装但探测失败、或安装但缺少
  本次调用所需能力（同类原因每次运行仅提醒一次）。`agent-only` 下停止报错。
- **工具升级后，FineSub 用来让它不读你的个人配置、只用本次允许的工具的那些设置变了**：命令行打印一行 `Warning:` 提醒，但**任务照常完成**。
  可忽略该提醒，也可将 CLI 版本和这条提醒告知维护者复核。
- **运行中途工具崩溃**：该小段会自动重试，最多重试若干次；仍失败则明确报错，不会将失败视为成功。
- **额度耗尽**：同一份额度连续失败两次后自动探测一次，探测也失败则**停用 2 小时**（打印一行
  提醒），期间不再使用；任意一次成功调用立即解除，`finesub quota thaw agent` 也能提前放回（见[额度与解冻](env.md#额度与解冻)）。该判断可能误判：探测使用同一模型、同一设置，
  因此「额度耗尽」与「模型配置错误」在探测看来相同。提醒中会包含实际报错原文，通常先读那一句
  即可区分。

任何情况下，字幕的正确性不取决于使用哪条路径：所有结果都要通过同一套校验才会写入产物。

---

## 7. 检查额度是否耗尽

```powershell
finesub agent-ping
```

该命令向每个已安装的模型发送一句极短的请求，检查能否正常应答。这几家 CLI 要么没有查询额度的
命令，要么只报整个额度窗口或额度桶还剩多少、说不出某个模型此刻能不能答，所以该命令实际发送一次调用，
本身会消耗少量额度。加 `--tier` 只探测一个，
名称可写工具名(`LOCAL_CODEX`)或额度池名(`AGY_ANTHROPIC`)。

输出直接引用 CLI 的原话，例如额度耗尽时 Codex 会提示「You've hit your usage limit... try
again at ...」。FineSub 不解释该信息的含义，由用户自行判断。

agy 的 Gemini、Claude（Sonnet、Opus 共用一份）与 Gemini 4 Argon 分开计量，FineSub 分三份记录（表格中会标注归属）。Argon 在 Pro 订阅上是单独一份、只按周计；在 Ultra 订阅上从 Gemini 那份里花、另有自己的上限，FineSub 照样单独记它：Gemini 那份用完时，Argon 最多再失败几次才被停用。
Claude 那份耗尽不影响带音视频的任务继续使用 agy 的 Gemini。CodeBuddy 的两份额度见 [4.2](#42-codebuddy模型清单跟着账号走)；
`agent-ping` 探它的三个付费模型时各扣一点积分（一句话的量），探 `hy4flash-free` 不会转付费。

## 8. FineSub 创建的会话

agy 与 dsh 每次调用都会在自己那边留下会话记录。Codex、Claude Code、CodeBuddy 按窗口跑时
（出厂的 `per-window`）FineSub 让它们不留，只有实验性的 `resume`、`pseudo-conversational` 档位才留；
`config.toml` 里写了 `[llm] local_agent_keep_episodes = true` 时 Codex 也留。例外是 Codex 给同一种会话的后续窗口留的那条共用开头（每个任务每种会话约一条），它留在 FineSub 自己的 Codex 目录里（用 ChatGPT 账号时是[单独登录](#codex-用-chatgpt-账号时给-finesub-单独登录一次)的那个目录），不进你自己的 Codex 历史。要识别 CLI 记录里哪些会话由 FineSub 产生，查看与 `config.toml` 同目录的
`agent-sessions.jsonl`：每行一次调用，记录时间、工具、模型、用途，以及**这个工具自己的会话 id**。
凭该 id 可在 CLI 自己的记录中对应。

失败的调用和上述小探测也会记录，这正是出问题时最需要查阅的。清理和搬盘操作不会改动该文件。

## 9. 磁盘上留下的内容

调用失败时，FineSub 会把现场（发送的内容、返回的事件、错误日志）保留在磁盘上，便于事后排查。
成功的调用不留任何内容。

**最多保留最近 20 个**，新增时删除最旧的，因此不会持续增长。单个现场通常几十 KB，最多几 MB
（带音视频的任务会保存一份剪辑副本）。立即清空全部现场：

```powershell
finesub agent-clean
```

该命令只清理当前这套安装的现场。若在多个位置安装过 FineSub 并希望全部清理，加 `--all-domains`：它先列出要删的目录，
要你输入 `DELETE` 确认；在脚本里定期清理、没人输入时，写成 `finesub agent-clean --all-domains --force`。
`--all-domains` **无法证明其他位置的任务已停止**，先停掉所有 FineSub 任务再用。

其他相关行为：

- **排查时想把每次调用都留下**：`config.toml` 写 `[llm] local_agent_keep_episodes = true`（默认 `false`）。每次调用的现场
  （输入、事件流、agent 与 FineSub 工具之间往来的每条消息）一律留下，成功的也不删、不按上面的名额清理；Codex 同时保留它自己的会话记录。占用的磁盘不封顶，查完关掉，再用 `agent-clean` 清。
- `finesub relocate <目录>` 搬移大文件时，这些现场会一并搬移。
- `finesub uninstall` 对它们的处理与模型相同：默认一起删；大文件目录搬到别处或与别的安装共用时默认保留，加 `--purge-big-data` 才删。
- 有任务正在运行时，清理和卸载都会拒绝执行，并提示先等待任务结束。

---

## 10. 相关文档

- API key 配置：[`docs/manual/env.md`](env.md)
- 数据存放与搬盘：[`docs/manual/resources.md`](resources.md)
- 一次调用如何选择模型：[`docs/manual/model-routing.md`](model-routing.md)
