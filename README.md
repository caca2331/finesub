# finesub

把长音频变成精修级中文字幕。

```text
音频/视频 → 人声分离 → VAD + ASR → 稳定化 → LLM 纠错翻译 → 成品 SRT
```

## 效果

以主播游戏实况（日语）为例：

**时间轴**——原生 Whisper 输出 vs 稳定化后的 raw 轴：

```
# 原生 Whisper（幻觉 + 磎片 + 超长句）
00:00:24.500 --> 00:00:27.800  ご視聴ありがとうござい          ← ASR幻觉
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

1. **精准时间轴**——稳定化后的 raw 轴低幻觉、高召回：BGM/静默段不会产生幽灵字幕，真实语音不会被吞，时间边界精确到帧。
2. **翻译harness**——高度优化的prompt，包含自维护的知识库，主播常用术语、角色名、游戏专名会自动积累并应用到后续窗口，越跑越准。

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

装的是一个轻量壳：首次运行自动装好隔离的 Python 运行环境（Python 都无需预装）和
FFmpeg，模型按需下载；一切落在 `%LOCALAPPDATA%\FineSub` 下（大文件可以用
`finesub relocate` 搬到别的盘），`finesub uninstall` 即可卸载干净。设置、API Key
和知识库与 Desktop 共用同一份。子命令与细节见 [cli/README.md](cli/README.md)，
数据位置见 [docs/manual/resources.md](docs/manual/resources.md)。

一条命令出字幕：

```powershell
# 音频，视频输入都可
finesub <输入名>.mp4 --language ja --extra-info "主播四月一日，原神直播切片" --stage final-srt --knowledge update

# URL 也行
finesub "https://www.bilibili.com/video/BVxxxx" --stage final-srt --name "四月一看PV"
```

其中：
- 不传`--language`时自动检测语言；
- `--extra-info`提供背景信息（主播名、游戏名、关键专名等），能显著提升纠错准确率，非必须。
- 不传 `--stage` 则默认停在 raw SRT（ASR结果，不调 API）；加 `--stage final-srt` 跑 LLM 纠错翻译。这一步需要配置 API 或 agent，二选一或组合：
  - API：需要配好 Gemini API key——Desktop 在设置页填，CLI/源码写 `.env`；推荐再配上 Exa API key；都是免费的，见 [环境配置](docs/manual/env.md)。
  - agent：用 **Antigravity CLI / Codex CLI / Claude Code** 已有的订阅额度来跑。其中 Antigravity
    已提供现成预设，且它是唯一支持音频多模态的。配置与细节见 [本机 Agent 后端](docs/manual/agent.md)。
- 知识库（主播术语、角色名等）**默认读取但不写入**（`--knowledge collect`）：已有内容会自动注入，本次任务不改动它。传 `--knowledge update` 才在纠错后把本次的发现写回；传 `--knowledge none` 则完全不读也不写。
- 传 `--name` 以指定和覆盖输入名。
- 显存够的话可以额外传 `--gpu-budget-gb 8`，语音识别阶段会并行提速。如果卡比较好可传12或16，但边际收益有限。

跑完后去 `out/<输入名>/` 里找字幕：`<输入名>.srt`（成品）和 `<输入名>-raw.srt`（未纠错原文）。

### Windows Desktop App（功能少于 CLI）
Windows 用户可以使用 [FineSub Desktop](desktop/README.md) 图形客户端来创建任务、管理资源和查看日志；它复用同一套 pipeline，不取代命令行。从 [Releases](https://github.com/caca2331/finesub/releases) 下载 `FineSub-Desktop-<版本>-Setup.exe` 安装；或下载 `finesub-full-<版本>-win-x64.zip` 解压即用（portable，不写注册表）。两种形式与 CLI 共用同一份设置、API Key 和知识库（`%LOCALAPPDATA%\FineSub\user-data`）；模型与缓存默认跟着安装目录，可搬到别的盘，见 [docs/manual/resources.md](docs/manual/resources.md)。
>**⚠️ 桌面端只覆盖单个任务的常用路径，功能少于 CLI**（例如没有批量处理）：遇到问题建议改用 CLI；不熟悉命令行的话，可以让 AI agent 辅助你使用。

### 源码安装
开发者要用仓库开发版、或想复用已有 Python/pip 环境的话，见 [仓库安装](docs/manual/repo-install.md)（uv 与 pip 两种流程；本页命令把 `finesub` 换成 `python -m finesub.pipeline` 即可）。

## 它做了什么

1. **人声分离**——去掉 BGM 和音效，只留人声。
2. **VAD + ASR 对齐**——切分语音段、跑 Whisper、输出带时间戳的逐句转写。
3. **ASR 稳定化**——去噪、合并碎片、丢弃幻觉，输出干净且时间精准的 raw 轴。
4. **LLM 纠错翻译**——结合音频/画面语境纠正误听、翻译成中文、进一步合并和丢弃，输出成品字幕。
- 多模态纠错：结合音频/画面纠正 ASR 误听（专名、同音词、口误）
- 翻译成自然中文（不是机翻味）
- 合并碎片成完整句（严守时长/字数门槛）
- 丢弃复读幻觉、套话、无意义填充词
- 输出置信度标注，低置信行建议人工核对
- 自动积累知识库：主播术语、角色名、常用表达会写入本地知识库，下次跑同一主播时自动注入，越用越准

> LLM路线/档位、知识库、搜索代理、token 预算等细节见 [LLM Harness 行为](docs/llm_harness_behavior.md) 和 [知识库说明](docs/knowledge.md)。

## 批量运行

```powershell
# 多个输入
finesub batch a.wav b.mp4 --stage final-srt --language ja

# JSONL manifest
finesub batch --manifest tasks.jsonl --knowledge update
```

（源码安装对应 `python -m finesub.batch`。）

单项失败不影响其余，重跑即续跑。

## 输出文件

以 `data/input.mp4` 跑到 `--stage final-srt` 为例：


| 文件              | 说明                   |
| --------------- | -------------------- |
| `input-raw.srt` | 未纠错原文 SRT            |
| `input.srt`     | **成品 SRT**（纠错翻译+后处理） |


全部产物归到 `out/input/` 一个目录下。完整产物树见 [README_DEV.md](README_DEV.md)。

## 环境要求


| 阶段         | 需要                            |
| ---------- | ----------------------------- |
| 人声分离 + ASR | NVIDIA 显卡（见下表）、≥8GB 内存   |
| LLM 纠错翻译   | 无需 GPU；≥4GB 内存；ffmpeg（Desktop/CLI 自动提供；源码安装需自备并加入 PATH） |

**显卡支持范围**

| | 型号 |
| --- | --- |
| 支持 | RTX 50 / 40 / 30 / 20 系（含 Ti、SUPER、笔记本版），GTX 1660、GTX 1650，以及数据中心的 V100 / A100 / H100 —— 显存需 ≥4GB |
| 不支持 | GTX 10 系及更早（1080 / 1070 / 1060 / 1050、GTX 9 系等），以及 AMD、Intel 核显 |

不支持的显卡**不会报错，而是自动回退 CPU** 并在 stderr 打一条 `Warning:`。回退能出正确
字幕，但慢很多（人声分离尤其慢），长音频不建议这么跑。显存 4GB 起够用；更大显存只在人声
分离阶段换来并行提速，边际收益有限（见 `--gpu-budget-gb`）。

URL 输入 Desktop/CLI 开箱即用；源码安装另需 `uv pip install yt-dlp`。

## 文档

面向使用者：

- [环境配置](docs/manual/env.md)——API key 配置
- [资源与大文件](docs/manual/resources.md)——数据装在哪、怎么搬盘、怎么删干净
- [仓库安装](docs/manual/repo-install.md)——源码安装完整步骤（uv 默认 / pip 替代）
- [patched CTranslate2](docs/manual/ct2-wheel.md)——ASR 必需的补丁版 CT2（源码安装用）
- [模型路由配置](docs/manual/model-routing.md)——哪个任务用哪些模型、各开关与旋钮的意义、接自己的 API endpoint
- [本机 Agent 后端](docs/manual/agent.md)——用本机 Codex / Claude Code / Antigravity 订阅代替 API 额度
- [知识库样板](examples/knowledge/)——迷你骨架条目

想读到实现层：[开发者说明](README_DEV.md) 是入口，`docs/` 根下的其余文件都是给开发者的
（约定见 [docs/README.md](docs/README.md)）。

---

代码 [MIT](LICENSE)；`src/finesub/llm/prompt_templates/` 下的 prompt 明文 [CC BY-SA 4.0](src/finesub/llm/prompt_templates/LICENSE.md)。
