# finesub

把长音频变成精修级中文字幕。

```text
音频/视频 → 人声分离 → 语音检测（VAD）+ 语音识别（ASR）→ 稳定化 → 大模型（LLM）纠错翻译 → 成品 SRT
```

## 效果

以主播游戏实况（日语）为例：

**时间轴**——原生 Whisper（语音识别模型）输出 vs 稳定化后、还没纠错的原文字幕（下称 raw 轴）：

```
# 原生 Whisper（幻觉 + 碎片 + 超长句）
00:00:24.500 --> 00:00:27.800  ご視聴ありがとうござい          ← ASR 幻觉
00:00:27.800 --> 00:00:29.100  あの、なんだっけ?              
00:00:29.100 --> 00:00:30.200  メミじゃなくて、                ← 碎片
00:00:30.200 --> 00:00:31.500  ユメじゃなくて                  ← 碎片
00:00:34.800 --> 00:00:42.000  じゃあなんかがなんかして 最後の遺産が時が来てそれを得て  ← 超长句

# 稳定化后（幻觉丢弃、碎片合并、超长句拆分、时间精准）
00:00:27.800 --> 00:00:29.000  あの、なんだっけ?             
00:00:29.600 --> 00:00:31.200  メミじゃなくて、ユメじゃなくて
00:00:34.800 --> 00:00:37.850  じゃあなんかがなんかして
00:00:38.144 --> 00:00:42.067  最後の遺産が時が来てそれを得て
```

**纠错翻译**——结合音频/画面语境：

| raw ASR                 | 纠错翻译后             |
| ----------------------- | ----------------- |
| `ほんまに?新書に変わってる。あ、ほんとだ。` | 真的吗？换成新衣服了。啊，真的耶。 |
| `ネジがやっぱ分かりやすいな`         | 发条果然很明显呢。         |
| `ごめんごめん 怖どらないで`         | 抱歉抱歉，别害怕。         |

核心特色：

1. **精准时间轴**——BGM、静默段不出幽灵字幕，尽量不漏真实语音，时间边界贴着说话的起止。
2. **越用越准的纠错翻译**——LLM 结合音频、画面纠错并译成自然中文；加 `--knowledge update` 时把主播术语、角色名写回本地知识库，之后的任务自动用上。

如果觉得好用，欢迎点个 [Star](https://github.com/caca2331/finesub) ⭐

## 快速开始

### 命令行 CLI（推荐）

用 [uv](https://docs.astral.sh/uv/) 安装：

```powershell
winget install astral-sh.uv  # 没有 uv 的话先安装 uv（装完需开新终端再跑下一条）
```

```powershell
uv tool install finesub
```

⚠ 用 `uv tool install` 装的安装版**只支持 Windows x64**；Linux / macOS 请走[源码安装](docs/manual/repo-install.md)。

安装的是一个轻量外壳：首次运行时自动装好隔离的 Python 运行环境（无需预装 Python）与 FFmpeg，模型按需下载，都放在 `%LOCALAPPDATA%\FineSub` 下；设置（`config.toml`）、API Key（`.env`）与知识库在其中的 `user-data` 里。C 盘放不下、要搬盘见[资源与大文件](docs/manual/resources.md)，其余子命令与卸载见 [cli/README.md](cli/README.md)。以后升级跑 `uv tool upgrade finesub`。

### 源码安装

想改代码或跑开发版的话，clone 仓库后按 [仓库安装](docs/manual/repo-install.md) 装好依赖（含 ASR 必需的补丁版 CTranslate2），入口是 `python -m finesub.pipeline`，参数与下文的 `finesub` 完全相同。

### 一条命令出字幕

要得到中文字幕（下面的 `--stage final-srt`），先配好一把 Gemini 免费 key，或配好本机 agent（见下面列表里的 agent 一条）。
在 `%LOCALAPPDATA%\FineSub\user-data\.env` 里写（没有这个文件就新建；第一次运行任务之前连 `user-data` 目录也还没有，一起建上；第二行的 Exa key 可选，有它联网查资料更好）：

```text
GEMINI_FREE={"main":"AIza..."}
EXA_KEYS={"exa1":"exa-..."}
```

两家 key 去哪申请、怎么确认配上了，见 [环境配置](docs/manual/env.md)。

```powershell
# 音频、视频输入都可
finesub <输入名>.mp4 --language ja --extra-info "主播四月一日，原神直播切片" --stage final-srt --knowledge update

# URL 也行
finesub "https://www.bilibili.com/video/BVxxxx" --stage final-srt --name "四月一看PV"
```

其中：

- 输入（文件或 URL）写在 `finesub` 后面第一个；写在选项后面时，产物落到当前目录的 `out\` 下、不进下文的任务目录；
- 不传 `--language` 时本机 Whisper 自动检测语言；换用别的转写后端（`--asr`）或日语微调模型时请给上，见 [转写后端](docs/manual/asr-backend-choice.md)；
- 纠错默认只给模型听音频；要它也看画面（名牌、对话框），加 `--llm-media video`（更费额度）。听音频、看画面都只有出厂的 Gemini API 与 agy 的 Gemini 做得到，别的模型只读文字，见[模型选择](docs/manual/models.md#各后端的使用体验)；
- `--extra-info` 提供背景信息（主播名、游戏名、关键专名等），能显著提升纠错准确率，非必须。写进去的要求（某种写法、要放进字幕的歌词等）会被照做。只对这一次的翻译风格要求写进 `--extra-style`，如 `--extra-style "人名保留日文原文"`（例子与它和风格条目的关系见[知识库](docs/manual/knowledge.md#风格条目style记住一套翻译习惯)）。
- 不传 `--stage` 则默认停在 raw SRT（ASR 结果；不用云端转写时不调 API）；加 `--stage final-srt` 跑 LLM 纠错翻译。`--stage` 依次是 `vocal` → `aligned` → `stable` → `raw-srt` → `translated-srt` → `final-srt`，各停在哪见 [运行产物](docs/manual/outputs.md#--stage指定停止的阶段)。这一步需要配置 API 或 agent，二选一或组合：
  - API：需要配好 Gemini API key（写进上面 `user-data` 下的 `.env`）；推荐再配上 Exa API key；都是免费的，见 [环境配置](docs/manual/env.md)。免费额度一天大约够纠 2–3 小时素材（默认档，不含 `--knowledge update` 的知识更新，它也占同样的次数；估算的依据与别的档见[同一页](docs/manual/env.md#geminigoogle-ai-studio)），用完就停、第二天续跑。连不上 Google 的网络（如中国大陆）要先设代理，见[走代理](docs/manual/model-routing.md#走代理api-proxy)。
  - agent：使用 **Antigravity（下称 agy，用它的命令行工具）/ Codex CLI / Claude Code** 已有的订阅额度运行。其中 agy 提供了现成预设，且是本机 agent 里唯一能听音频的（Gemini API 也能听）。配置与细节见 [本机 Agent 后端](docs/manual/agent.md)。
- LLM 阶段默认是 `--llm-difficulty intermediate`：强模型一步纠错翻译，适合多数素材。多人对谈、要分清谁在说时加 `--llm-difficulty quality`（更贵，成品字幕不标人名，它多做的几步见 [说话人分离](docs/manual/diarization.md)）；免费额度用完或要大量便宜跑时用 `efficiency`。见 [模型路由](docs/manual/model-routing.md#difficulty-是定位)。
- 知识库（主播术语、角色名等）**默认读取但不写入**（`--knowledge read`）：已有内容会自动注入，本次任务不改动它。传 `--knowledge update` 才在纠错后把本次的发现写回；传 `--knowledge none` 则完全不读也不写。
- 传 `--name` 以指定和覆盖输入名。
- `--style <名字>` 让译文沿用知识库里记好的一套翻译口味（某字幕组的用词与断句习惯）；不传时怎么挑、怎么建一套，见 [知识库](docs/manual/knowledge.md#风格条目style记住一套翻译习惯)。

运行结束后，在任务目录里查看字幕：安装版跑单个输入时是 `%LOCALAPPDATA%\FineSub\tasks\<输入名>-<时间>-<后缀>\`（跑完会打印；用 `finesub relocate` 搬过盘的，看 `finesub doctor` 里 `tasks` 那一行），批量运行与源码运行是当前目录下的 `out/<输入名>/`。字幕文件：`<输入名>.srt`（成品）与 `<输入名>-raw.srt`（未经纠错的原文）。其余文件的含义与可删除范围，见 [运行产物](docs/manual/outputs.md)。

中途停下（额度用完、出错、Ctrl+C）时，原样再跑同一条命令就接着上次做，已完成的部分不重做：安装版按输入文件（或 URL）认任务，能接着做时开头打印 `Continuing task <任务名>`。中途停下时不打印任务目录，它就是 `tasks` 下以 `<输入名>-` 开头、时间最近的那个。

**第一次用，先拿一段几分钟的视频或音频照上面的命令试跑一次**，确认 key、网络都通了再处理长素材。成功的样子：任务目录里有 `<输入名>.srt`，打开是中文。
finesub 没有只处理一段时间的选项；手头只有整场直播时，先用 ffmpeg 截出前 5 分钟（本机要装有 ffmpeg，如 `winget install Gyan.FFmpeg`；安装版自带的那份不在命令行能直接用的路径上）：

```powershell
ffmpeg -i 整场.mp4 -t 300 -c copy 试跑.mp4
```

没成功时：提示没有可用的 API key，看[环境配置](docs/manual/env.md)；连不上 Google，看[走代理](docs/manual/model-routing.md#走代理api-proxy)；别的现象按[故障排查](docs/manual/troubleshooting.md)的症状表找。

### 批量运行

同一条命令传入多个输入（或 `--manifest tasks.jsonl`：一个每行写一项任务的清单文件）即为批量处理：三段流水线并行推进，单项失败不影响其他任务，运行期间可增删任务、调整优先级，中断后 `finesub --resume-batch` 续跑——细节见 [一次跑多个输入](docs/manual/batch.md)。

## 它做了什么

1. **人声分离**——去掉 BGM 和音效，仅保留人声。
2. **VAD + ASR 对齐**——VAD（语音检测）找出哪段有人说话，再跑 Whisper，输出带时间戳的逐句转写。
3. **ASR 稳定化**——去噪、合并碎片、丢弃幻觉，输出干净且时间精准的 raw 轴。
4. **LLM 纠错翻译**——结合音频/画面语境纠正误听、翻译成中文、进一步合并和丢弃，输出成品字幕。
    - 多模态纠错：结合音频/画面纠正 ASR 误听（专名、同音词、口误）
    - 翻译成自然中文（而非生硬的机翻风格）
    - 合并碎片成完整句（严守时长/字数门槛）
    - 丢弃复读幻觉、套话、无意义填充词
    - 输出置信度标注，低置信行建议人工核对

> LLM 怎么选模型、难度档与额度，见 [模型路由配置](docs/manual/model-routing.md)；知识库见 [知识库](docs/manual/knowledge.md)。

## 环境要求

| 阶段         | 需要                                 |
| ---------- | ---------------------------------- |
| 人声分离 + ASR | NVIDIA、AMD 或 Intel 显卡（Intel 为实验级，见下；没有也能跑，回退 CPU）、≥8GB 内存 |
| LLM 纠错翻译   | 无需 GPU；≥4GB 内存；ffmpeg（安装版自动提供；源码运行要自己装） |

N 卡须为 **RTX 20 系或更新**（GTX 1660 / 1650，和部分其他卡亦可）。显存 ≥4GB；≥8GB 更佳。驱动要 570 或更新（`nvidia-smi` 右上角的 CUDA Version 显示 12.8 或更高即可），不用另装 CUDA。WSL2 + N 卡是官方支持，A 卡、Intel 卡、原生 Linux 与 macOS 是实验支持，这几种都只走源码装法，各自支持到什么程度见 [平台与支持等级](docs/manual/resources.md)；完整型号表、各档位的显存要求与不支持时的处理方式也在那份。

URL 输入用 [yt-dlp](https://github.com/yt-dlp/yt-dlp) 下载，它支持的站点都能直接给链接（实测过 YouTube 与 B 站）；安装版第一次给 URL 时自动下载它。下载失败时的办法见 [故障排查](docs/manual/troubleshooting.md)。

## 反馈问题

结果不对、想发给开发者时，打一个反馈包：

- **用 `uv tool install` 装的**：把这次任务的目录（跑完时打印的那个，见上文「一条命令出字幕」）和 `%LOCALAPPDATA%\FineSub\user-data\logs\` 一起压缩成 zip。
- **源码安装的**：把这次的 `out/<输入名>/`（在你运行命令的那个目录下）和仓库根目录下的 `logs/` 一起压缩成 zip。手边有 Claude Code、Codex 这类编码 agent 的，也可以在仓库目录里对它说「把这次出问题的任务打个包」，它把字幕、参数、LLM 问答记录、运行日志与人声轨打成桌面上的一个 zip，只生成文件、不代为发送。包里有什么、怎么只贡献精修过的字幕而不带音频，见 [让 AI agent 替你做的几件事](docs/manual/agent-tasks.md#把出问题的那次打包好方便你发给开发者)。

打好的包通过你和开发者平常的联系方式（社交软件等）发过去；没有别的联系方式，就在 [GitHub Issues](https://github.com/caca2331/finesub/issues) 新开一个、附上它。⚠ 包里有这次的人声音频和本机路径（含用户名），发到公开的 Issue 之前想清楚；附件太大传不上时，先在 Issue 里把问题写清楚。

## 相关项目

- [Nonoka Sub X](https://github.com/Ricori/nonoka-sub-x/)——第三方图形工作站，带 UI 界面和更多编辑功能。以 finesub 的算法引擎为转写与翻译内核，外加多轨时间轴、波形、ASS 实时渲染与视频压制。
- [Yanami Sub](https://github.com/tuzibuqiahuluobo/yanami-sub)——第三方 Windows 桌面字幕工作台，基于 finesub，提供图形界面、批处理、资源管理、知识库与应用内更新。
- [audio-overlap-removal](https://github.com/caca2331/audio-overlap-removal)——从混合音频里剔除一条**已知**的参考音轨，尤其是带人声的；参考轨和混合里的那份可以时间线不一致（暂停、跳转、变速、有损编码都处理）。可以使 ASR 由此不识别媒体音轨，而专注识别主要内容。
- [flowocr](https://github.com/caca2331/FlowOCR)——读出视频画面上出现过的所有文字（字幕、对话框、名牌、标注），每条带精确到帧的起止时间、按位置自动分轨，产出 SRT 或原位叠加的 ASS；有原文剧本时可把字幕对上剧本。与 finesub 的语音字幕互补。

## 文档

面向使用者的文档都在 [使用手册](docs/manual/index.md)，那份索引按任务分组：安装与 API key、运行产物、调参、批量、知识库、模型与额度、显卡与搬盘、故障排查。另有 [知识库样板](examples/knowledge/)——迷你骨架条目。

如需了解实现层：[开发者说明](README_DEV.md) 为入口，`docs/` 根下的其余文件面向开发者（约定见 [docs/README.md](docs/README.md)）。

---

代码 [GPL-3.0-or-later](LICENSE)（自 0.5.0 起；0.4.x 及更早的发行版仍按 MIT 授权，不受本次变更影响）；`src/finesub/llm/prompt_templates/` 下的 prompt 明文 [CC BY-SA 4.0](src/finesub/llm/prompt_templates/LICENSE.md)。
