# finesub

把长音频变成精修级中文字幕。

```text
音频/视频 → 人声分离 → VAD + ASR → 稳定化 → LLM 纠错翻译 → 成品 SRT
```

## 效果

以主播游戏实况（日语）为例：

**时间轴**——原生 Whisper 输出 vs 稳定化后的 raw 轴：

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

1. **精准时间轴**——稳定化后的 raw 轴低幻觉、高召回：BGM/静默段不会产生幽灵字幕，真实语音不会被遗漏，时间边界精确到帧。
2. **翻译 harness**——高度优化的 prompt，包含自维护的知识库，主播常用术语、角色名、游戏专名会自动积累并应用到后续窗口，准确度随使用不断提升。

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

安装的是一个轻量外壳：首次运行时会自动安装隔离的 Python 运行环境（无需预装 Python）与 FFmpeg，模型按需下载；所有数据存放于 `%LOCALAPPDATA%\FineSub` 下（大文件可通过 `finesub relocate` 迁移到其他磁盘），执行 `finesub uninstall` 即可完整卸载。设置、API Key 和知识库放在 `user-data` 下，一处配置、处处生效。子命令与细节见 [cli/README.md](cli/README.md)，数据位置见 [docs/manual/resources.md](docs/manual/resources.md)。

### 源码安装

想改代码或跑开发版的话，clone 仓库后按 [仓库安装](docs/manual/repo-install.md) 装好依赖（含 ASR 必需的补丁版 CTranslate2），入口是 `python -m finesub.pipeline`，参数与下文的 `finesub` 完全相同。

### 一条命令出字幕

```powershell
# 音频、视频输入都可
finesub <输入名>.mp4 --language ja --extra-info "主播四月一日，原神直播切片" --stage final-srt --knowledge update

# URL 也行
finesub "https://www.bilibili.com/video/BVxxxx" --stage final-srt --name "四月一看PV"
```

其中：

- 不传 `--language` 时自动检测语言；
- `--extra-info` 提供背景信息（主播名、游戏名、关键专名等），能显著提升纠错准确率，非必须。
- 不传 `--stage` 则默认停在 raw SRT（ASR 结果，不调 API）；加 `--stage final-srt` 跑 LLM 纠错翻译。这一步需要配置 API 或 agent，二选一或组合：
  - API：需要配好 Gemini API key（写进 `.env`）；推荐再配上 Exa API key；都是免费的，见 [环境配置](docs/manual/env.md)。
  - agent：使用 **Antigravity CLI / Codex CLI / Claude Code** 已有的订阅额度运行。其中 Antigravity 提供了现成预设，且是唯一支持音频多模态的后端。配置与细节见 [本机 Agent 后端](docs/manual/agent.md)。
- 知识库（主播术语、角色名等）**默认读取但不写入**（`--knowledge collect`）：已有内容会自动注入，本次任务不改动它。传 `--knowledge update` 才在纠错后把本次的发现写回；传 `--knowledge none` 则完全不读也不写。
- 传 `--name` 以指定和覆盖输入名。
- `--style <名字>` 让译文沿用知识库里记好的一套翻译口味（某字幕组的用词与断句习惯）；不传则用库里的 `default_style`（如果有）。怎么建一套、怎么让它边翻边学（`--style-mode`），见 [知识库](docs/manual/knowledge.md)。

运行结束后，在 `out/<输入名>/` 目录中查看字幕：`<输入名>.srt`（成品）与 `<输入名>-raw.srt`（未经纠错的原文）。其余文件的含义与可删除范围，见 [运行产物](docs/manual/outputs.md)。

### 批量运行

同一条命令传入多个输入（或 `--manifest tasks.jsonl`）即为批量处理：三段流水线并行推进，单项失败不影响其他任务，运行期间可增删任务、调整优先级，中断后 `finesub --resume-batch` 续跑——细节见 [一次跑多个输入](docs/manual/batch.md)。

## 它做了什么

1. **人声分离**——去掉 BGM 和音效，仅保留人声。
2. **VAD + ASR 对齐**——切分语音段、跑 Whisper、输出带时间戳的逐句转写。
3. **ASR 稳定化**——去噪、合并碎片、丢弃幻觉，输出干净且时间精准的 raw 轴。
4. **LLM 纠错翻译**——结合音频/画面语境纠正误听、翻译成中文、进一步合并和丢弃，输出成品字幕。
    - 多模态纠错：结合音频/画面纠正 ASR 误听（专名、同音词、口误）
    - 翻译成自然中文（而非生硬的机翻风格）
    - 合并碎片成完整句（严守时长/字数门槛）
    - 丢弃复读幻觉、套话、无意义填充词
    - 输出置信度标注，低置信行建议人工核对
    - 自动积累知识库：主播术语、角色名、常用表达会写入本地知识库，下次处理同一主播时自动注入，随使用不断变准

> LLM 路线/档位、知识库、搜索代理、token 预算等细节见 [LLM Harness 行为](docs/llm_harness_behavior.md) 和 [知识库说明](docs/knowledge.md)。

## 输出文件

以 `data/input.mp4` 跑到 `--stage final-srt` 为例：

| 文件              | 说明                   |
| --------------- | -------------------- |
| `input-raw.srt` | 未纠错原文 SRT            |
| `input.srt`     | **成品 SRT**（纠错翻译+后处理） |

全部产物归到 `out/input/` 一个目录下。完整产物树见 [README_DEV.md](README_DEV.md)。

## 环境要求

| 阶段         | 需要                                 |
| ---------- | ---------------------------------- |
| 人声分离 + ASR | NVIDIA 显卡（见下）、≥8GB 内存              |
| LLM 纠错翻译   | 无需 GPU；≥4GB 内存；ffmpeg（托管 CLI 自动提供） |

显卡须为 **RTX 20 系或更新**（GTX 1660 / 1650，和部分其他卡亦可）。显存 ≥4GB；≥8GB 更佳。完整型号表、各档位的显存要求与不支持时的处理方式，见 [显卡支持范围与档位](docs/manual/resources.md)。

URL 输入开箱即用。

## 相关项目

- [Nonoka Sub X](https://github.com/Ricori/nonoka-sub-x/)——第三方图形工作站，带 UI 界面和更多编辑功能。以 finesub 的算法引擎为转写与翻译内核，外加多轨时间轴、波形、ASS 实时渲染与视频压制。
- [audio-overlap-removal](https://github.com/caca2331/audio-overlap-removal)——从混合音频里剔除一条**已知**的参考音轨，尤其是带人声的；参考轨和混合里的那份可以时间线不一致（暂停、跳转、变速、有损编码都处理）。可以使 ASR 由此不识别媒体音轨，而专注识别主要内容。

## 文档

面向使用者：

- [环境配置](docs/manual/env.md)——API key 配置
- [运行产物](docs/manual/outputs.md)——每个产物文件是什么、`--stage` 停在哪个阶段、低置信度行在哪里查看
- [调参](docs/manual/tuning.md)——字幕长短、识别与 LLM 两侧的参数、大致耗时
- [故障排查](docs/manual/troubleshooting.md)——按症状定位对应文档
- [资源与大文件](docs/manual/resources.md)——数据存放位置、迁移与删除方式；显卡支持范围、档位与 CPU 回退
- [模型选择](docs/manual/models.md)——ASR 可选用的 Whisper、分离器与第二模型的作用、各 LLM 后端的实际体验
- [模型路由配置](docs/manual/model-routing.md)——各任务使用的模型、开关与参数的含义、接入自有 API endpoint
- [本机 Agent 后端](docs/manual/agent.md)——用本机 Codex / Claude Code / Antigravity 订阅代替 API 额度
- [知识库样板](examples/knowledge/)——迷你骨架条目

如需了解实现层：[开发者说明](README_DEV.md) 为入口，`docs/` 根下的其余文件面向开发者（约定见 [docs/README.md](docs/README.md)）。

---

代码 [GPL-3.0-or-later](LICENSE)（自 0.5.0 起；0.4.x 及更早的发行版仍按 MIT 授权，不受本次变更影响）；`src/finesub/llm/prompt_templates/` 下的 prompt 明文 [CC BY-SA 4.0](src/finesub/llm/prompt_templates/LICENSE.md)。
