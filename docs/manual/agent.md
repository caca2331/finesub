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

这里配置的是「预设」，不是「档位」（档位见 §5，一般不用动）。预设决定每类任务由谁执行；出厂
预设中只有 `agy-hybrid` 与 `agy` 把本机工具排了进去。安装 Antigravity 后，写入这一行即完成启用。

组内顺序：先使用 Gemini 免费额度，用尽或失败后才轮到 Antigravity 订阅，两边额度都不浪费。
文字类任务优先交给 Opus 4.6；带音频或视频的任务自动由 agy 的 Gemini 承担。

**不想用 API、只走订阅**就改写 `preset = "agy"`：同样的格子，但名单里没有任何 API 成员
（连 `--test-profile` 的 test target 也是 agy 自己的）。两个预设的完整名单见
[`model-routing.md`](model-routing.md) 第 4 节。

**只装了 Codex 或 Claude Code**（未装 Antigravity）时，出厂预设中没有它们的绑定，需要自行指定
模型。照抄以下配置即可：

```toml
[llm]
preset = "my-agent"

[llm.presets.my-agent.bindings]
"correction-text/quality" = "local-codex-completion-gpt-5_6-luna"
"research/quality"        = "local-codex-native-gpt-5_6-luna"
```

把 `local-codex-` 换成 `local-claude-` 即为 Claude Code。target 名带 `native` 的变体才允许
模型使用自己的搜索工具。处理带音视频的素材时加 `--llm-correction-media text`（这两家不能听
音频）。如需按难度分档、或让失败回退到 Gemini 等更细的控制，见
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

> **三家都能联网**（`--llm-retrieval native`，使用模型自带的搜索工具）。Codex 和 Claude Code 把
> 格子绑到名字带 `native` 的 target 即可（见 [`model-routing.md`](model-routing.md) 的
> 「快速选模型」）;`agy` / `agy-hybrid` 预设开箱即用。
>
> **Antigravity 不回报来源 URL，只报告搜索词**；另两家会记录每个来源 URL。如果需要核对「这条
> 修改依据哪个页面」，使用 `--llm-retrieval local`（由 FineSub 自行搜索，把带出处的结果交给模型）,
> 或改用 Codex / Claude Code 两种后端。
>
> 提示：如果本机上网要走公司代理，这些工具在 FineSub 中可能无法联网，因为 FineSub 出于安全
> 考虑不会把代理设置传给它们。此时请使用 `api-only` 档位。

---

## 4. agy(Antigravity)：不建议开视频

agy 是三家里面唯一支持音频和视频的后端，但**建议只用音频**。开视频后，相同长度的素材会消耗
明显更多的输入 token，切出的片段更小、调用次数更多；而字幕纠错真正依赖的信息绝大部分在
**声音**里，画面带来的提升有限。除非素材必须依靠画面分辨（屏幕上出现的人名、专有名词字牌
之类），否则默认的音频模式已足够。

---

## 4.1 dsh(DeepSeek Harness)：只能按窗口跑

`npm install -g @deepseek-ai/dsh`（需要 Node ≥ 22.19）。它与另外三家有几处实质差异，安装前
请确认：

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
key 运行模型、仅用 DeepSeek 的 key 做搜索。搜索本身不额外收费，但没有有效的 DeepSeek key 则
无法搜索。使用 `--llm-retrieval native` 时需填写两个 ref:

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

## 5. 进阶：三个「档位」的作用

`execution_policy` 是一道**闸门**，只能限制后端，不能新增后端：

| 档位 | 含义 | 使用场景 |
| --- | --- | --- |
| `agent-text-preferred` | 出厂默认，不限制任何后端，按预设顺序执行 | 默认使用 |
| `api-only` | 屏蔽全部本机工具，只走 API | 临时只用 API，又不想改预设 |
| `agent-only` | 屏蔽全部 API，只走本机工具 | 确保不消耗任何 Gemini 额度 |

由于它只能限制，设置档位本身不会启用本机工具：出厂 `default` 预设中没有任何本机工具，
`agent-text-preferred` 也不会改变这一点；`agent-only` 搭配 `default` 预设会因无可用后端而直接
报错。要使用本机工具，先按 §2 配置预设。

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
（`--llm-media audio` 下的纠错窗与查询轮即属此类，而 `audio` 是出厂默认）：此类调用在该格找不到
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

该命令向每个已安装的模型发送一句极短的请求，检查能否正常应答。三个工具都**没有**查询额度的
命令，实际发送一次调用是唯一可靠的办法，因此该命令本身会消耗少量额度。加 `--tier` 只探测一个，
名称可写工具名(`LOCAL_CODEX`)或额度池名(`AGY_ANTHROPIC`)。

输出直接引用 CLI 的原话，例如额度耗尽时 Codex 会提示「You've hit your usage limit... try
again at ...」。FineSub 不解释该信息的含义，由用户自行判断。

Antigravity 的 Gemini 和 Opus 是**两份独立计量的额度**，FineSub 分开记录（表格中会标注归属）。
Opus 耗尽不影响带音视频的任务继续使用 agy 的 Gemini。

## 8. FineSub 创建的 session

三个工具都会保留各自的会话记录。要识别其中哪些由 FineSub 产生，查看与 `config.toml` 同目录的
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
