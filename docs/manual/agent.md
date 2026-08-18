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
自己的搜索工具。有音视频的素材请加 `--correction-media text`（这两家不能听音频）。想改得更细
——比如按难度分档、或让它撞墙后回退到 Gemini——见 [`model-routing.md`](model-routing.md)。

---

## 3. 需要装什么

按你有的订阅装就行，**不需要三个都装**：

| 工具 | 负责 | 登录方式 |
| --- | --- | --- |
| Antigravity（agy） | 文字活 + 带音频/视频的活，出厂 `agy` 预设用的就是它 | agy 自带的登录 |
| Codex CLI | 文字活（纠错、翻译、调查），需自己配置才会用上 | `codex` 自带的登录 |
| Claude Code | 同上 | `claude /login` |

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

## 6. 出问题会怎么样

- **工具没装 / 没登录**：默认档位下悄悄换成预设里的下一个（`agy` 预设里就是 Gemini）继续跑；
  `agent-only` 停下报错。
- **工具版本太新，多了几个 FineSub 不认识的功能**：会在命令行打一行 `Warning:` 提醒，但**任务照常
  完成**。看到这行可以忽略，也可以告诉维护者更新一下名单。
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
