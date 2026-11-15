# 用本机 AI 订阅运行字幕处理（agent 模式）

FineSub 默认通过 Gemini 的 API 额度完成纠错和翻译。如果本机已安装并登录 **Codex CLI**、
**Claude Code** 或 **Antigravity**（下文简称 agy），可以让 FineSub 改用这些工具执行，消耗已有
订阅，不再占用 Gemini 的免费或付费额度。

这就是 agent 模式。它不改变字幕质量，只改变执行路径。

---

## 1. 适用场景

**适合：**

- Gemini 免费额度不够用，而本机正好有 Codex / Claude Code / Antigravity 订阅。
- 素材多、运行时间长，不想受配额限制。
- 不方便配置 Gemini API key(agent 模式的一种档位完全不需要它)。

**不适合：**

- 只是偶尔处理一两个短视频，默认的 API 路径更省事。

---

## 2. 启用方式

在 `config.toml` 中加一行（文件位置和创建方法见 [`resources.md`](resources.md)）:

```toml
[llm]
preset = "agy-hybrid"
```

这里配置的是「预设」，不是「档位」（档位见 [§5 进阶：三个「档位」的作用](#5-进阶三个档位的作用)，一般不用动）。预设决定每类任务由谁执行；出厂
预设中只有 `agy-hybrid`、`agy` 与 `agy-fast` 把本机工具排了进去（`agy-fast` 是长输入的快速粗纠，见 [`model-routing.md`](model-routing.md)「快速粗纠」）。安装 Antigravity 后，写入这一行即完成启用。

组内顺序：先使用 Gemini 免费额度，用尽或失败后才轮到 Antigravity 订阅，两边额度都不浪费。
文字类任务优先交给 agy 的 Sonnet 5.5；带音频或视频的任务自动由 agy 的 Gemini 承担。

**不想用 API、只走订阅**就改写 `preset = "agy"`：同样的格子，但名单里没有任何 API 成员
（连 `--test-profile` 的 test target 也是 agy 自己的）。两个预设的完整名单见
[`model-routing.md`](model-routing.md) 第 4 节。

**只装了 Codex 或 Claude Code**（未装 Antigravity）时，出厂预设中没有它们的绑定，需要自行指定
模型。照抄以下配置即可：

```toml
[llm]
preset = "my-agent"

[llm.presets.my-agent]
name = "本机 Codex"

[llm.presets.my-agent.bindings]
"correction-text/quality" = "local-codex-gpt-6-luna"
"research/quality"        = "local-codex-gpt-6-luna"
```

换成 Claude Code 时写 `local-claude-sonnet`（另有 `local-claude-opus`、`local-claude-haiku`，都指
Claude Code 当前的最新版）。加 `--llm-retrieval native` 时模型会用自己的搜索工具，target 名不用换。
处理带音视频的素材时加 `--llm-media text`（这两家不能听音频）。如需按难度分档、或让失败回退到 Gemini 等更细的控制，见
[`model-routing.md`](model-routing.md)。

---

## 3. 需要装什么

按已有订阅选择安装，不需要都装：

| 工具 | 负责 | 登录方式 |
| --- | --- | --- |
| Antigravity(agy) | 文字任务 + 带音频/视频的任务，出厂 `agy` / `agy-hybrid` 预设使用 | agy 自带登录 |
| Codex CLI | 文字任务（纠错、翻译、调查），需自行配置绑定 | `codex` 自带登录 |
| Claude Code | 同上 | `claude /login` |
| dsh(DeepSeek Harness) | 只做文字任务，且只能按窗口跑（见 4.1） | `~/.dsh/.credentials.yaml` 中的 API key |
| CodeBuddy Code CLI | 文字任务，需自行配置绑定（见 4.2） | 独立安装 CLI，在 CLI 中登录 |

> **都能联网**（`--llm-retrieval native`，使用模型自带的搜索工具）。Codex / Claude Code /
> CodeBuddy 把格子绑到名字带 `native` 的 target 即可（见 [`model-routing.md`](model-routing.md) 的
> 「快速选模型」）;`agy` / `agy-hybrid` 预设开箱即用。
>
> **Antigravity 不回报来源 URL，只报告搜索词**；其余几家会记录每个来源 URL。如果需要核对「这条
> 修改依据哪个页面」，使用 `--llm-retrieval local`（由 FineSub 自行搜索，把带出处的结果交给模型）,
> 或改用 Codex / Claude Code 两种后端。
>
> 提示：本机上网要走代理时，FineSub 会把代理与证书相关的环境变量（`HTTP_PROXY`、`HTTPS_PROXY`、
> `NO_PROXY`、`SSL_CERT_FILE` 等）原样传给这些工具。FineSub 自己用的密钥（如 `OPENAI_API_KEY`）不会传；
> 你在 Codex 配置里给 provider 点名的变量会交给 Codex。
>
> Codex 走的是**你在 Codex 自己的配置里选的 provider**（`~/.codex/config.toml` 的 `model_provider`，
> 例如本机代理）。provider 指向本机地址而那个端口没在监听时（代理没开），FineSub 会在调用前把 Codex
> 标为不可用、改用别的后端，并说明原因；不会悄悄改走直连登录号。

---

## 4. agy(Antigravity)：不建议开视频

agy 是几家里面唯一支持音频和视频的后端，但**建议只用音频**。开视频后，相同长度的素材会消耗
明显更多的输入 token，切出的片段更小、调用次数更多；而字幕纠错真正依赖的信息绝大部分在
**声音**里，画面带来的提升有限。除非素材必须依靠画面分辨（屏幕上出现的人名、专有名词字牌
之类），否则默认的音频模式已足够。

---

## 4.1 dsh(DeepSeek Harness)：只能按窗口跑

`npm install -g @deepseek-ai/dsh`（需要 Node ≥ 22.19）。FineSub 验证过的是 `0.1.5-rc.1`；更旧的版本照常能用，
但每次运行会提醒一次「比验证过的版本旧」。它与另外几家有几处实质差异，安装前请确认：

- **只能绑到「按窗口」的格子。** dsh 只从命令行接收任务，一整窗字幕无法塞进一条命令行，因此
  FineSub 仅使用它的工具协议：模型自行取任务、读正文、提交答案。绑定到 `api` 或 `resume`
  档位会**直接报错**（错误信息会说明原因），不会静默降级。
- **不支持音视频。** 纠错窗需要听音频时不要绑定 dsh。
- **无法报告 token 用量。** dsh 运行完毕只打印最终答案，不输出统计信息，因此任务报告中 dsh
  行的 token 数为空。这不是 bug，而是它没有可报告的数据。
- **不能续会话。** 每次调用都是全新会话。
- **不要接限速的端点。** 一整窗字幕是数百行的任务，慢端点会在
  `[llm].local_agent_timeout_seconds`（出厂 28 分钟）上超时；且 dsh 失败时不打印任何内容，看起来
  就像「dsh 不可用」。实测：正常速率的 DeepSeek-V4-Flash 处理一窗 270 条约 13 分钟；在限速到
  ≈5 tok/s 的免费预览端点上，同一窗两次均超时、无任何输出。

**key 存放位置**：`~/.dsh/.credentials.yaml`(Windows 下为 `C:\Users\<你>\.dsh\`):

```yaml
version: 1
refs:
  deepseek: sk-...
```

再在 `~/.dsh/settings.yaml` 中将 provider 指向该 ref:

```yaml
llm-deepseek:
  apiKeyEnv: deepseek        # 这是 ref 的名字,不是环境变量的值
```

**联网搜索使用另一把 key。** dsh 的 `web_search` 由自带的 DeepSeek 搜索插件提供，该插件读取
**自己的**凭据（缺省 ref 名 `DEEPSEEK_API_KEY`），与运行模型的 key 相互独立。因此可以用别家的
key 运行模型、仅用 DeepSeek 的 key 做搜索。⚠ 每次搜索会在这把 DeepSeek key 上扣少量费用；没有有效的
DeepSeek key 则无法搜索。同一档里模型还能用 `web_fetch` 抓取网页正文，这一项不需要 key，同样只在 native 检索时
开放。使用 `--llm-retrieval native` 时需填写两个 ref:

```yaml
version: 1
refs:
  deepseek: sk-...           # 模型使用(也可指向别家)
  DEEPSEEK_API_KEY: sk-...   # 搜索插件使用
```

**接入自己的网关**：dsh 支持 OpenAI 兼容的自定义 provider，在 `settings.yaml` 的
`llm-pi-ai.providers` 下声明；FineSub 侧用一条 catalog 行的 `provider/model` 指过去即可
（见 [`model-routing.md`](model-routing.md)）。

---

## 4.2 CodeBuddy：模型清单跟着账号走

先独立安装官方 CodeBuddy Code CLI：`npm install -g @tencent-ai/codebuddy-code`，再运行 `codebuddy`
完成登录。中国站选择 **Chinese Site**，国际站选择 **International Site**；两站共用 CLI 协议，但登录态、
模型与套餐权益必须对应所选站点。FineSub 沿用 CLI 的登录态，保留 `CODEBUDDY_INTERNET_ENVIRONMENT`
站点配置，不读取 WorkBuddy 桌面端凭据。出厂模型行是中国站模型，国际站用户须按账号可用清单添加模型行。
显式选择站点时，该环境变量中国站用 `internal`，国际站用 `external`；切换后在对应站点登录，
并从继承新环境的终端或应用进程运行 FineSub。
FineSub 验证过的是 `2.164.0`；更旧的版本照常能用，但每次运行会提醒一次版本比验证过的旧
（`… is older than the pinned 2.164.0`）。
登录失效时（CLI 报 `Authentication required` 或 `Please use /login`），FineSub 当作未登录处理：按
[6. 异常处理](#6-异常处理) 换下一个候选，并在提示里让你重新运行 `codebuddy` 登录，不当作临时故障反复重试。
出厂预设没有绑定它，按下面建一个自己的预设即可（`local-codebuddy-` 开头的 target 名见
[`model-routing.md`](model-routing.md) 的「快速选模型」；想要失败时在组内往下换，就绑出厂的两个组，见同一份的
「CodeBuddy 的两个组」）:

```toml
[llm]
preset = "my-codebuddy"

[llm.presets.my-codebuddy]
name = "本机 CodeBuddy"

[llm.presets.my-codebuddy.bindings]
"correction-text/quality" = "local-codebuddy-glm-5_3-flash"
"research/quality"        = "local-codebuddy-glm-5_3-flash"
```

**目前推荐 `glm-5.3-flash`**：四行里它是实测下来最省心的一档——纠错档位已经调到不会陷进
思考，质量在同一窗对照里没有可见差距。它**不是最便宜的那个**：单次调用的积分倍率 x0.06，
`deepseek-v4.1-flash` 是 x0.03；也不是窗口最大的，单次输出上限 32,000 比别人小。省心换的
就是这两样：想要更便宜或更大的窗口就换 `hy3`（免费，64,000）或 `deepseek-v4.1-flash`
（128,000）。`deepseek-v4-pro` 目前全面不如 Flash，出厂仍保留，以免以后升级又能用了再加回来。

安装前请确认这几点：

- **能用哪些模型由账号决定，不由版本决定。** `codebuddy --help` 印的那张模型表可能一个都用不了；
  以服务端为准——随便给一个不存在的 `--model`，报错信息会把**你这个账号真正能用的**清单列出来。
  FineSub 打包了其中四行（`hy3` / `glm-5.3-flash` / `deepseek-v4.1-flash` / `deepseek-v4-pro`）；如果你的套餐是别的
  模型，按 [`model-routing.md`](model-routing.md) 的「5. 接自己的模型」写进 `config.toml`，target
  会自动生成。绑了一个账号够不到的模型，FineSub 会**直接报错并把服务端那句原话贴给你**，不会当
  成「登录过期」或临时故障反复重试。
- **DeepSeek 与 GLM 两系不要用最高思考档。** 实测（2026-09-04，一整窗 79 条真实字幕）：
  `deepseek-v4-flash`（当时的 Flash；现在的 V4.1 在 2026-09-30 的 `high` 档实测里同样不收敛）与 `glm-5.3-flash` 在 `high` 档下都会陷进不收敛的思考——拿到正文后
  连续输出几万到十几万字符的推理、始终不交答案，直到撞上 `[llm].local_agent_timeout_seconds`
  （出厂 28 分钟）。同一窗压到 `low` 就一次跑通，质量还相当好。混元系（`hy3`）在 `high`
  下正常。想全局压档用 `[llm].local_agent_reasoning_effort = "low"`。
  ⚠ 这不是速度问题：三个模型的纯输出速率实测相当。
- **`glm-5.3-flash` 会被额外告知它自己的输出上限。** catalog 里有一列开关
  `hint_output_ceiling`（见 [`model-routing.md`](model-routing.md)），开着时 FineSub 会在任务
  说明里加一句（英文）：每轮输出上限约为这一行的 `max_output_tokens`（32,000），快到时先调一次 `pull_status` 再接着写。出厂只对这一行开，
  因为实测只有它受益：同一窗它不加这句会超时，加了就一次跑完；而当时的 `deepseek-v4-flash`
  加不加、写什么数都没变化。⚠ 那组对照是在**高思考档**下做的，而纠错档现在已经降到
  `low`、本来就不会超时——所以这个开关在出厂配置下更像一层保险，它自己的增量收益没有
  单独测过。自己加模型行时默认关着。
- **绑 `glm-5.3-flash` 会看到一条窗口告警，那是实话。** 它的单次输出上限是 32,000，正好贴着
  FineSub 允许的最低值，所以绑进模型组时会提示「最大输出 32000 < 64000」。它照常能跑，只是
  一窗能塞的字幕比别的模型少；不想看到这条就换 `hy3`（64,000）或 `deepseek-v4.1-flash`（128,000）。
  `deepseek-v4-pro` 的上限是 50,000，同样会告警。
- **不支持音视频。** 纠错窗需要听音频时不要绑定它。
- **联网只有搜索，没有抓取网页。** 它自带 `WebSearch` 与 `WebFetch` 两个工具，但后者在无人值守
  下会被它自己的权限层拒绝，所以 FineSub 只授权前者。搜索结果的来源 URL 照常记录。
- **它读得到你的规则文件。** 这个 CLI 没有「忽略用户配置」的开关（Claude Code 的
  `--safe-mode` 在这里不存在）。FineSub 每次调用都在自己新建的空目录里跑，所以**项目里**的
  `CODEBUDDY.md` / `AGENTS.md` 不会进来；但你的**用户级** `~/.codebuddy/CODEBUDDY.md` 照样会进来。
  FineSub 在开场说明里请模型把这类注入的规则放在一边、只按任务的要求回答，这是软约束；任务报告里
  会如实记成 `inherited`。Codex（`~/.codex/AGENTS.md`）与 Antigravity（`~/.gemini/GEMINI.md`）同样如此。
- **一个登录背后有两份额度，不是一份、也不是按行一份。** 免费线 `hy3` **自己一份每日额度**；
  另外三行（`glm-5.3-flash`、`deepseek-v4.1-flash`、`deepseek-v4-pro`）连同 `hy3` 的付费孪生，**花的是同一份
  积分余额**。所以 FineSub 记两个额度池：`hy3` 用光只影响那一条；积分用光则同时影响那三行。
  链条往下走一格即可。
- **`hy3` 每天有免费额度；用光之后会自动接着用付费额度，不停下来。**
  这一行出厂就在 catalog 的 `fallback_model` 列里写了自己的付费孪生（`hy3` → `hy3-x`），
  当天免费额度用光时由 CLI 把这次会话交给付费线跑完，
  **任务不中断**；跑完会打一条提示告诉你这次切过。免费线是 x0.00，付费线是 x0.05 ——
  也就是说这条提示等于一张小额账单。
  ⚠ **不想自动扣费就把这个模型的 `fallback_model` 清空。** 它只从 catalog 读，不在
  `config.toml` 里：在数据根目录（源码 checkout 即仓库根）的 `model_catalog.toml` 里写这两行，
  其余的键沿用出厂值（`""` 表示取默认值，这里就是「没有付费孪生」）：

  ```toml
  [models.local-codebuddy-hy3]
  fallback_model = ""
  ```

  ⚠ **免费额度用完这件事，现在会被说出来**（2026-09-05 起）：孪生一开口，FineSub 就知道那条
  免费线今天连挂两次了，于是发一次极短的探测去确认，确认属实就**记下来并告警一次**（`agent-ping`
  之后也报得出）。**任务照常继续走孪生**——填了这一列就是这个意思，它不会因此停下。这一行真正
  走不通、被换掉，只有积分也空了那一次。（探测答得出，就把刚记下的连挂次数清零、不冻结：切换也可能
  只是赶上了持续限流，而不是额度用尽。）

  只想关掉自动扣费就写上面那一行。清空后行为变成另一种：**只用免费额度，用完就冻结这条线两小时、
  换模型组里下一个候选，绝不扣积分**。两种模式就是这一列填不填的区别。

  ⚠ **有一种情况认不出来**：CLI 悄悄切到孪生、还没来得及产出任何回答就失败了（比如积分也正好
  空了）。这时事件流里没有任何模型名，FineSub 分不清这次花的是免费额度还是积分。处理分两种：
  两份额度都还在，就**两边都不记、只告警**；只剩一份还在，就拿那一份**去实测一次**——它答得出
  就什么都不冻。宁可多问一句，也不凭猜把一条还活着的线冻掉。
  ⚠ 抄过去的那份不会跟着升级走：以后出厂行改了别的列（窗口、思考档），你的覆盖行还是旧
  的。改法的完整说明见 [`model-routing.md`](model-routing.md) [§5](model-routing.md#5-接自己的模型)「接自己的模型」。
  ⚠ 它对**持续的限流**也会切：CLI 先用原模型重试一次，还不行才换人。所以偶发的高峰期
  也可能让你看到那条提示。
  ⚠ 需要独立 CodeBuddy CLI 足够新（CLI 得认识 `--fallback-model`）。太旧的话 FineSub 会提醒你
  一句，并把这一行当纯免费线用——不会因为这个选项跑不起来。**但有一处已知的粗糙**：路由是按
  catalog 里写没写孪生来判断这一行还有没有额度可用的，它不知道你这台机器上的 CLI 其实用不了
  孪生。于是免费额度用光之后，这一行每次仍会被试一下、失败、再往下走一格，直到当天额度恢复。
  代价是每个任务多一次失败的调用，不影响结果；不想要就把 `fallback_model` 清空，或升级独立 CodeBuddy CLI。
- **另外三行本来就在扣积分**，没有免费孪生可切：`glm-5.3-flash` x0.06、
  `deepseek-v4.1-flash` x0.03、`deepseek-v4-pro` x0.51。它们的额度用光就是报错。

---

## 5. 进阶：三个「档位」的作用

`execution_policy` 是一道**闸门**，只能限制后端，不能新增后端：

| 档位 | 含义 | 使用场景 |
| --- | --- | --- |
| `agent-text-preferred` | 出厂默认，不限制任何后端，按预设顺序执行 | 默认使用 |
| `api-only` | 屏蔽全部本机工具，只走 API | 临时只用 API，又不想改预设 |
| `agent-only` | 屏蔽全部 API，只走本机工具 | 确保不消耗任何 Gemini 额度 |

由于它只能限制，设置档位本身不会启用本机工具：出厂 `default` 预设中没有任何本机工具，
`agent-text-preferred` 也不会改变这一点；`agent-only` 搭配 `default` 预设会因无可用后端而直接
报错。要使用本机工具，先按 [§2 启用方式](#2-启用方式) 配置预设。

**同时运行的 agent 数量**：`config.toml` 的 `[llm].local_agent_max_parallel`（默认 4，下限 1）
限制本机同时活跃的 agent CLI 进程数——这是整台机器与订阅的物理上限，对所有本机工具共用，
与「一个任务开几个并行窗口」(`--llm-parallel-windows`)和「批处理同时跑几个任务」
(`--max-parallel-tasks`)是三个不同的旋钮。

怎么调看你的流量池，没有一个能测出来的常数。各家 agent 对并发的容忍度其实很高，并发高
不会被封；真正的代价是两条——**调太高时，任务一多 5 小时额度会被快速耗尽，大量 session
执行到一半断掉，已花的 token 全部白费**；而省下的时间边际递减，token 效率（缓存命中、advice
台账）和上下文完整性受到的损失却边际递增。出厂默认 windows 1 / tasks 2：任务之间并行
（互不共享上下文，没有质量代价），任务之内保持串行（保留 advice 台账与缓存收益）；额度富余
再逐档往上调。

**这三个旋钮不是彼此独立的**：每个任务在**起步**就占住一格保底 agent 槽，所以
`local_agent_max_parallel` 一定要不低于 `--max-parallel-tasks`，否则多出来的任务会卡在起步处
等前一个任务整体跑完（连它本可以先做的纯 API 阶段也一起等），看起来像卡死。

---

## 5.1 让正在使用的 agent 直接处理任务(conversational)

> 这条路已冻结（2026-10-01）：保留现状、照常可用，但不再扩展或改进。

如果本机已有一个正在运行的 Claude Code / Codex 或其他 agent 会话，可以让 FineSub 不再启动新
的 CLI，而是把纠错任务交给该会话：在路由表中将某个任务组绑定到 `conversational-agent` 模型组
（该组只能独占使用）。运行时 FineSub 会打印一条提示：

```text
finesub agent-join
```

在该 agent 中执行此命令，并把打印出的说明交给 agent;agent 将通过 `finesub agent-task` 取任务、
读文件、提交。FineSub 不会向 agent 注入任何内容，也不会修改其权限。**无需带参数**，它会自动
找到正在等待的 run；若有多个 run 在等待，会列出供选择。

**等待上限为一小时**，超时则报错停止；该限制固定，没有开关（若无法及时到场，建议使用下面的
分两步运行方式）。agent 接手后该表即停止，可慢慢处理而不被打断；若中途断线，重新执行
`finesub agent-join` 即可接上，等待时长从头计算。

真正限制「慢慢处理」的是另一条规则：**不要长时间完全不执行任何 `finesub agent-task` 命令**。
默认静默上限 30 分钟，超过即视为 agent 离开，任务退回队列重新等待。agent 再次执行
`finesub agent-task status` 即可取回任务，已写好的答案原样提交，已完成的工作不会丢失。如需
放宽，调整 `config.toml` 的 `[llm].local_agent_timeout_seconds`（默认 1680 秒 = 28 分钟，无
上限）：该值表示「一次调用最多运行多久」，静默容忍度跟随其变化，始终比它多两分钟。

同一个反复接手又离开的 agent 会在第 3 次之后被放弃，报错会明确指出该原因。

**只有纯文本调用会走这条路**，带音视频的调用不会交给它。因此**不要将带媒体的格子绑定到它**
（`--llm-media audio` 下的纠错窗即属此类，而 `audio` 是出厂默认）：此类调用在该格找不到
可用模型时，会直接报错停止，而不是丢弃媒体后发送纯文本。整条 run 使用 `--llm-media text`，或
只绑定不带媒体的格子。

**何时需要在场**：提示并非在运行开始时出现，而是在**第一个真正交给你的调用**时才打印。若运行
完整管线，前面的分离、VAD/ASR、稳定化阶段都需先完成；加上等待上限一小时，实际情形是「先跑
一小时，然后在某个不确定的时刻于一小时内到场」。错过提示也不要紧，直接执行 `finesub agent-join`
即可接上（无需记住任何目录名）；但人不在场时，那一小时等待仍会走完。

如需免去守候，**分两步运行**：同一命令执行两次，第一次停在语音部分，第二次继续到纠错。

```powershell
# 第一步:完成耗时的语音部分,不涉及 agent
finesub data\input.wav --language ja --stage raw-srt

# 第二步:人在电脑前时,原样重跑第一步的命令,只改 --stage 并加上 --llm-media
finesub data\input.wav --language ja --stage final-srt --llm-media text
```

第二步根据产物是否已存在跳过语音部分，只执行纠错，提示几秒内出现。`--llm-media text` 不可
省略：出厂默认是 `audio`，纠错窗会带音频，而带媒体的调用不会交给 agent（见上一段）。其余开关
（`--language`、`--extra-info` 等）两次需保持一致。

**如果窗口大到 agent 一条回复写不完**：答案通过文件提交，可以分几次写入同一文件后再提交，不必
挤在一条回复里。若仍嫌大，调小 `config.toml` 的 `[chunking] max_window_subtitle_tokens`
（默认 10000），窗口会被切得更碎、每个更短。

## 6. 异常处理

- **工具未安装 / 未登录 / 版本缺少能力**：默认档位下切换到预设中的下一个（`agy-hybrid` 预设里就是
  Gemini）继续运行，并打印**一行** `Warning:` 说明原因：未安装、安装但探测失败、或安装但缺少
  本次调用所需能力（同类原因每次运行仅提醒一次）。`agent-only` 下停止报错。
- **工具版本过新，工具隔离契约发生变化**：命令行打印一行 `Warning:` 提醒，但**任务照常完成**。
  可忽略该提醒，也可将 CLI 版本和 warning 告知维护者复核。
- **运行中途工具崩溃**：该小段会自动重试，最多重试若干次；仍失败则明确报错，不会将失败视为成功。
- **额度耗尽**：同一份额度连续失败两次后自动探测一次，探测也失败则**停用 2 小时**（打印一行
  提醒），期间不再使用；任意一次成功调用立即解除。该判断可能误判：探测使用同一模型、同一设置，
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

Antigravity 的 Gemini、Claude（Sonnet、Opus 共用一份）与 Gemini 4 Argon 分开计量，FineSub 分三份记录（表格中会标注归属）。Argon 在 Pro 订阅上是单独一份、只按周计；在 Ultra 订阅上从 Gemini 那份里花、另有自己的上限，FineSub 照样单独记它：Gemini 那份用完时，Argon 最多再失败几次才被停用。
Claude 那份耗尽不影响带音视频的任务继续使用 agy 的 Gemini。CodeBuddy 同理，但分法不同：**一条免费线
一份，付费的共用一份积分**，所以 `hy3` 用光不影响 `glm-5.3-flash` / `deepseek-v4.1-flash` /
`deepseek-v4-pro`，而那三行付费是一起空的。

⚠ **有一个已知的盲区**：如果你在 `config.toml` 里把会话档位设成了 `pseudo-conversational`
（出厂默认不是它），那么「这次是免费线还是付费孪生答的」在那一档识别不出来——那种模式下一个
长驻 CLI 顺序处理多个任务，模型归属只在整条会话结束时才确定。上面这些按应答者分账的行为在
那一档不成立。出厂默认 `per-window` 不受影响。

⚠ **探测一次只问一个模型，且绝不会走付费孪生**。这两件事都是有意的：逐模型探才答得出「哪个
模型能答」（合并成按池探就得挑一个代表，而代表能答不等于另外两个能答，模型可能被账号收回或
改名）；而带着 `fallback_model` 去探，免费线用光时会被 CLI 切到付费孪生、**答成功**——工具于是
报「可用」，既答错了又扣了积分。所以探测发车前会把那一列摘掉。⚠ 但它不等于「探测不花钱」：
`glm-5.3-flash` / `deepseek-v4.1-flash` / `deepseek-v4-pro` 三行本来就在扣积分，探它们就是在调
计费模型，只是一句话的量。

## 8. FineSub 创建的 session

agy 与 dsh 每次调用都会在自己那边留下会话记录。Codex、Claude Code、CodeBuddy 按窗口跑时
（出厂的 `per-window`）FineSub 让它们不留，只有实验性的 `resume`、`pseudo-conversational` 档位才留；
开了 `[llm] local_agent_keep_episodes` 时 Codex 也留。要识别 CLI 记录里哪些会话由 FineSub 产生，查看与 `config.toml` 同目录的
`agent-sessions.jsonl`：每行一次调用，记录时间、工具、模型、用途，以及**对端的 session id**。
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

该命令只清理当前这套安装的现场。若在多个位置安装过 FineSub 并希望全部清理，加 `--all-domains`。
该命令**无法证明其他位置的任务已停止**，仅在确认没有任务运行时使用。

其他相关行为：

- `finesub relocate <目录>` 搬移大文件时，这些现场会一并搬移。
- `finesub uninstall` 默认**保留**它们（与模型、成品字幕一致），加 `--purge-big-data` 才会删除。
- 有任务正在运行时，清理和卸载都会拒绝执行，并提示先等待任务结束。

---

## 10. 相关文档

- API key 配置：[`docs/manual/env.md`](env.md)
- 数据存放与搬盘：[`docs/manual/resources.md`](resources.md)
- 一次调用如何选择模型：[`docs/manual/model-routing.md`](model-routing.md)
