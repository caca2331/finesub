# 用本机的 AI 订阅跑字幕（agent 模式）

FineSub 默认用 Gemini 的 API 额度做纠错和翻译。如果你本来就装了 **Codex CLI**、**Claude Code**
或 **Antigravity**（下称 agy）并且已经登录，可以让 FineSub 改用它们来干活——用的是你已经付过钱
的那份订阅，不再消耗 Gemini 的免费/付费额度。

这就是 agent 模式。它不会让字幕变好或变差，只是换了一条出活的路。

---

## 1. 什么时候值得开

**适合：**

- Gemini 免费额度不够用，而你手上正好有 Codex / Claude Code / Antigravity 的订阅。
- 素材多、跑得久，不想盯着配额。
- 不方便配 Gemini API key（agent 模式的一种档位完全不需要它）。

**不适合：**

- 只是偶尔跑一两个短视频——默认的 API 路子更省事。

---

## 2. 怎么开

在 `config.toml` 里加这一行（这个文件在哪、怎么建，见
[`resources.md`](resources.md)）：

```toml
[llm]
preset = "agy"
```

**要开的是「预设」，不是「档位」**（档位是另一回事，见 §5，一般不用动）。预设决定每种活找谁
干；出厂只有 `agy` 这一个预设把本机工具排进去了。装了 Antigravity 的话，写上这一行就开好了。

它是这么排的：先用 Gemini 免费额度，撞墙了才轮到你的 Antigravity 订阅——两边额度都不浪费。
文字活优先给 Opus 4.6，带音频/视频的活自动交给 agy 那边的 Gemini。

**只装了 Codex 或 Claude Code**（没装 Antigravity）的话，出厂预设里没有它们的位置，需要自己
指定一下用哪个模型。照抄这段就能开：

```toml
[llm]
preset = "my-agent"

[llm.presets.my-agent.bindings]
"correction-text/quality" = "local-codex-completion-gpt-5_6-luna"
"research/quality"        = "local-codex-native-gpt-5_6-luna"
```

把 `local-codex-` 换成 `local-claude-` 就是 Claude Code。名字里带 `native` 的那个才允许模型用
自己的搜索工具。有音视频的素材请加 `--llm-correction-media text`（这两家不能听音频）。想改得更细
——比如按难度分档、或让它撞墙后回退到 Gemini——见 [`model-routing.md`](model-routing.md)。

---

## 3. 需要装什么

按你有的订阅装就行，**不需要都装**：

| 工具 | 负责 | 登录方式 |
| --- | --- | --- |
| Antigravity（agy） | 文字活 + 带音频/视频的活，出厂 `agy` 预设用的就是它 | agy 自带的登录 |
| Codex CLI | 文字活（纠错、翻译、调查），需自己配置才会用上 | `codex` 自带的登录 |
| Claude Code | 同上 | `claude /login` |
| dsh（DeepSeek Harness） | 只做文字活，且**只能按窗口跑**（见 4.1） | `~/.dsh/.credentials.yaml` 里的 API key |

> **三家都能联网**（`--retrieval native`，让模型用自己的搜索工具）。Codex 和 Claude Code 把格子
> 绑到名字里带 `native` 的 target 即可（见 [`model-routing.md`](model-routing.md) 的「快速选模型」）；
> `agy` 预设开箱即用。
>
> 但 **Antigravity 不会告诉你它看了哪些网页**——它只报搜索了什么词。另两家会把每个来源 URL 都
> 记下来。如果你在意"这条修改是根据哪个页面来的"，用 `--retrieval local`（FineSub 自己去搜、
> 把带出处的结果喂给模型），或者用 Codex / Claude Code 那两档。

> 提示：如果你的电脑上网要走公司代理，这几个工具在 FineSub 里可能连不上网——FineSub 出于安全
> 考虑不会把代理设置传给它们。这种情况下用 `api-only`。

---

## 4. agy（Antigravity）：不建议开视频

agy 是三个里唯一能听音频、看画面的，但**建议只用音频**。开了视频之后，同样长的素材要多花
非常多的输入 token，而且切出来的片段会明显变小、总次数变多；字幕纠错真正用得上的信息绝大
部分在**声音**里，画面带来的提升很小。除非你的素材必须靠画面才能分辨（屏幕上出现的人名、
专有名词字牌之类），否则默认的音频就够。

---

## 4.1 dsh（DeepSeek Harness）：只能按窗口跑

`npm install -g @deepseek-ai/dsh`（要 Node ≥ 22.19）。它和另外三家有几处**实打实的不同**，
装之前先看清楚：

- **只能绑到「按窗口」的格子。** dsh 只从命令行接任务，而一整窗字幕塞不进一条命令行，所以
  FineSub 只用它的工具协议：模型自己来取任务、读正文、交答案。把它绑到 `api` 或 `resume`
  档位会**直接报错**（错误信息会说明原因），不会静默降级。
- **不带音视频。** 纠错窗要听音频就别绑给它。
- **报不出 token 用量。** dsh 跑完只打印最终答案，不吐任何统计，所以任务报告里 dsh 那行的
  token 数是空的——这不是 bug，是它没有可报的东西。
- **不能续会话。** 每次调用都是全新会话。
- **别接限速的端点。** 一整窗字幕是几百行的活，慢端点会在
  `[llm].local_agent_timeout_seconds`（出厂 28 分钟）上超时，而且 dsh 失败时什么都不打印，
  看起来就像「dsh 不能用」。实测：正常速率的 DeepSeek-V4-Flash 一窗 270 条约 13 分钟跑完；
  一条被限到 ≈5 tok/s 的免费预览端点上同一窗两次都超时、一个字都没出来。

**key 放哪**：`~/.dsh/.credentials.yaml`（Windows 下即 `C:\Users\<你>\.dsh\`）：

```yaml
version: 1
refs:
  deepseek: sk-...
```

再在 `~/.dsh/settings.yaml` 里把 provider 指过去：

```yaml
llm-deepseek:
  apiKeyEnv: deepseek        # 这是 ref 的名字，不是环境变量的值
```

**联网搜索用的是另一把 key。** dsh 的 `web_search` 由它自带的 DeepSeek 搜索插件提供，
而那个插件读**自己的**凭据（缺省 ref 名 `DEEPSEEK_API_KEY`），和跑模型用的 key 是分开的。
所以可以**拿别家的 key 跑模型、只拿 DeepSeek 的 key 用来搜索**——搜索本身不额外收费，但没有
一把有效的 DeepSeek key 就搜不了。要用 `--retrieval native` 就把两个 ref 都填上：

```yaml
version: 1
refs:
  deepseek: sk-...           # 模型用（也可以指向别家）
  DEEPSEEK_API_KEY: sk-...   # 搜索插件用
```

**接自己的网关**：dsh 支持 OpenAI 兼容的自定义 provider，写在 `settings.yaml` 的
`llm-pi-ai.providers` 下，FineSub 这边用一条 catalog 行的 `provider/model` 指过去即可
（见 [`model-routing.md`](model-routing.md)）。

---

## 5. 进阶：那三个「档位」是干什么的

`execution_policy` 是一道**闸门**，只能往外拦，不能往里加：

| 档位 | 意思 | 什么时候用 |
| --- | --- | --- |
| `agent-text-preferred` | 出厂默认，谁都不拦，顺序按预设排的来 | 就用这个 |
| `api-only` | 把本机工具全拦掉，只走 API | 临时想只用 API，又不想改预设 |
| `agent-only` | 把 API 全拦掉，只走本机工具 | 想确保一分 Gemini 额度都不花 |

因为它只能往外拦，**光设档位不会让本机工具跑起来**：出厂 `default` 预设里一个本机工具都没有，
`agent-text-preferred` 也变不出来；`agent-only` 配 `default` 预设更会因为一个能用的都不剩而直接
报错。要用本机工具，先按 §2 把预设配好。

---

## 5.1 让你正在用的 agent 直接来干活（conversational）

如果你手头就开着一个 Claude Code / Codex / 别的 agent 会话，可以不让 FineSub 再起一个 CLI，而是把
纠错任务排队交给它：在路由表里把某个任务组绑到 `conversational-agent` 这个模型组（它只能独占一组）。
运行时 FineSub 会提示一条：

```text
finesub agent-join
```

在你的 agent 里执行这条命令，把它打印出来的说明交给 agent，agent 就会用 `finesub agent-task`
取任务、读文件、提交；FineSub 不往你的 agent 里注入任何东西，也不改它的权限。**不用带参数**——
它自己会找到那个在等的 run；同时有多个在等就会列出来让你指名。

**等人来领**有一小时的上限，等不到就报错停下；这个是固定的，没有开关（真要晚点再到场，用
下面的分两步跑，比让 run 空等强）。你的 agent 一接手这个表就停，它慢慢做不会被打断；中途掉线、
你重新 `finesub agent-join` 接上，这一小时也会重新从头算。

真正管着「慢慢做」的是另一条：**别长时间完全不发任何 `finesub agent-task` 命令**。默认允许
静默 30 分钟，超过就当你走人、任务退回队列重等——你的 agent 再 `finesub agent-task status`
就能把它领回来，已经写好的答案原样提交即可，做过的活不白费。要放宽就调
`config.toml` 的 `[llm].local_agent_timeout_seconds`（默认 1680 秒 = 28 分钟，没有上限）：
它说的是「一次调用最多跑多久」，静默容忍度跟着它走，永远比它多两分钟。

同一个反复接了又走的 agent 会在第 3 次之后被放弃，那时报错会直说是这个原因。

**只有纯文本调用会走这条路**，带音视频的调用不会交给它——所以**别把带媒体的格子绑到它**（`--llm-media audio` 下的纠错窗与查询轮
就是，而 `audio` 正是出厂默认）：那种调用在这一格找不到能用的模型，会直接报错停下，而不是悄悄把
媒体丢掉发纯文本。整条 run 用 `--llm-media text`，或者只把不带媒体的格子绑过来。

**什么时候要守着**：那行提示不是运行一开始就出现的，而是等到**第一个真要交给你的调用**才打——
全管线跑的话，前面的分离、VAD/ASR、稳定化都得先跑完。加上等人只等一小时，就成了「先跑一小时，
然后要在某个说不准的时刻一小时内到场」。提示本身错过了也不要紧，直接跑 `finesub agent-join`
就能接上（它不需要你记住任何目录名）；但人不在场那一小时还是会走完。

想省这份守候，**分两步跑**：同一条命令跑两次，第一次停在语音部分，第二次再往下走到纠错。

```powershell
# 第一步：把耗时的语音部分跑完，不涉及你的 agent
finesub data\input.wav --language ja --stage raw-srt

# 第二步：人在电脑前的时候，把第一步的命令原样再跑一遍，只改 --stage、加上 --llm-media
finesub data\input.wav --language ja --stage final-srt --llm-media text
```

第二步靠产物是否已存在跳过语音部分，只跑纠错，提示几秒内就出来。`--llm-media text` 不能省——
出厂默认是 `audio`，那样纠错窗会带音频，而带媒体的调用根本不会交给你的 agent（见上一段）。
其余开关（`--language`、`--extra-info` 等）两次要一致。

**如果一个窗口大到你的 agent 一条回复写不完**：答案是通过文件提交的，可以分几次写进同一个文件再
提交，不必挤在一条回复里。仍嫌大就把 `config.toml` 的 `[chunking] max_window_subtitle_tokens`
调小（默认 10000），窗口会切得更碎、每个更短。

## 6. 出问题会怎么样

- **工具没装 / 没登录 / 版本缺能力**：默认档位下换成预设里的下一个（`agy` 预设里就是 Gemini）
  继续跑，并打**一行** `Warning:` 说明是哪一类——没装、装了但探测失败、还是装了但缺这次调用要
  的能力（同一类每次运行只提醒一次）。`agent-only` 停下报错。
- **工具版本太新，工具隔离契约发生变化**：会在命令行打一行 `Warning:` 提醒，但**任务照常完成**。
  看到这行可以忽略，也可以把 CLI 版本和 warning 告诉维护者复核。
- **跑到一半工具崩了**：那一小段会自动重来，最多重试若干次；实在过不去就明确报错，不会假装成功。
- **额度用完了**：同一份额度连着失败两次就自动探测一次，探测也失败就把它**停用 5 小时**（会打一行
  提醒），期间不再碰它；任何一次成功调用立刻解除。这个判断**有可能猜错**——探测用的是同一个模型、
  同一套设置，所以「额度用完了」和「这个模型压根配错了」在它看来一样。提醒里会贴出实际的报错原文，
  先读那一句通常就能分清。

任何时候，字幕的正确性都不取决于用了哪条路——所有结果都要过同一套校验才会被写进产物。

---

## 7. 想知道是不是额度用完了

```powershell
finesub agent-ping
```

它给每个装好的模型发一句极短的话，看它答不答得上来。三个工具都**没有**查额度的命令，真发一次
调用是唯一诚实的办法——也因此这条命令本身会花掉一点点额度。加 `--tier` 只探一个，名字写工具
（`LOCAL_CODEX`）或额度池（`AGY_ANTHROPIC`）都行。

输出直接贴 CLI 自己的原话，比如额度用完时 Codex 会说「You've hit your usage limit... try again
at ...」。FineSub 不去猜这句话什么意思，你自己看就行。

Antigravity 的 Gemini 和 Opus 是**两份分开算的额度**，FineSub 也分开记（表格里会标出是哪一份）：
Opus 用完了不影响带音视频的活继续走 agy 的 Gemini。

## 8. 哪些 session 是 FineSub 建的

三个工具都会在自己那边留会话记录。想知道其中哪些是 FineSub 跑出来的，看跟 `config.toml` 放在
一起的 `agent-sessions.jsonl`：每行一次调用，记着时间、哪个工具、哪个模型、干什么用的，以及
**对方那边的 session id**——拿这个 id 就能在 CLI 自己的记录里对上号。

失败的调用和上面那种小探测也会记进去，恰恰是出问题时最需要查的。清理和搬盘都不会动它。

## 9. 磁盘上留下的东西

一次调用失败时，FineSub 会把现场（发出去的内容、返回的事件、错误日志）留在磁盘上，方便事后
查。成功的调用不留任何东西。

**最多留最近 20 个**，再有新的就把最旧的删掉，所以它不会一直涨。一个现场通常几十 KB，最多
几 MB（带音视频的活会存一份剪辑副本）。要立刻清掉全部：

```powershell
finesub agent-clean
```

它只清理本机当前这套安装的现场。如果你在多个地方装过 FineSub、想全清掉，加 `--all-domains`——
这条命令**没法证明别处的任务已经停了**，所以只在确定没有任务在跑的时候用。

其他相关行为：

- `finesub relocate <目录>` 搬大文件时，这些现场会跟着一起搬。
- `finesub uninstall` 默认**保留**它们（和模型、成品字幕一样），加 `--purge-big-data` 才一起删。
- 有任务正在跑的时候，清理和卸载都会拒绝执行并提示你先等它结束。

---

## 10. 相关文档

- API key 配置：[`docs/manual/env.md`](env.md)
- 数据都放在哪、怎么搬盘：[`docs/manual/resources.md`](resources.md)
- 一次调用到底怎么选模型：[`docs/manual/model-routing.md`](model-routing.md)
