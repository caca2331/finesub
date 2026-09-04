# FineSub Desktop

FineSub Desktop 是 FineSub 的可选 Windows 客户端：用图形界面创建任务、管理资源、
查看日志。它跑的是同一套 pipeline（`src/finesub/pipeline.py`，在隔离的
worker 进程里），**不取代命令行**——同一台机器上装了 CLI 的话，两边共用设置、
API Key 和知识库。

## 安装

从 [Releases](https://github.com/caca2331/finesub/releases) 取任一种：

- `FineSub-Desktop-<版本>-Setup.exe` —— 安装器，写开始菜单与卸载项。
- `finesub-full-<版本>-win-x64.zip` —— 解压即用，不写注册表。

首次运行会自动下载并安装隔离的 Python 3.12 运行环境与 AI 依赖（约 5 GB）、
FFmpeg，模型按需下载。这一步在应用内有进度与日志；装不上时可以暂停后重试。

## 数据放在哪

- **设置、API Key、知识库、任务历史** → `%LOCALAPPDATA%\FineSub\user-data`。
  安装版、便携版和 pip 安装的 CLI **共用这一份**，所以换个入口不会变成另一套知识库。
- **运行环境、模型、下载缓存、任务产物** → 默认跟着安装目录走，可以整体搬到别的盘，
  也可以让多个安装共用一份，不必重复下载。

搬盘、共用、卸载时删哪些，见 [`docs/manual/resources.md`](../docs/manual/resources.md)。
API Key 的配置见 [`docs/manual/env.md`](../docs/manual/env.md)。

## 界面能设什么

桌面端覆盖单个任务的常用路径。一次任务里能在界面上选的:

| 界面上有 | 对应的命令行选项 |
| --- | --- |
| 输入文件 / URL、输出名 | 输入、`--name` |
| 跑到哪一步 | `--stage` |
| 识别模型 | `--model` |
| 语言 | `--language` |
| 处理设备(自动 / 显卡 / CPU)、用哪张卡 | `--device`(选卡是桌面独有,命令行用 `CUDA_VISIBLE_DEVICES`) |
| 显卡档位 | `--gpu-tier` |
| 背景信息、翻译风格补充 | `--extra-info`、`--extra-style` |
| 知识库开关 | `--knowledge` |
| LLM 的媒体 / 检索 / 难度 / 快速模式 / 输出系数 | `--llm-media`、`--llm-retrieval`、`--llm-difficulty`、`--llm-fast`、`--llm-output-scale` |
| 稳定化档位、后处理档位、词级 SRT | `--asr-stabilize-profile`、`--postprocess-profile`、`--word` |
| 跑完清理中间产物 | (命令行不清,产物留在原地) |
| 设置页:字幕长度偏好 | `--split-length-scale` / `config.toml` 的 `[segmentation] length_scale` |

⚠ **有三处默认值和命令行不一样**,同一个文件两边跑结果可能不同:

| | 桌面端 | 命令行 |
| --- | --- | --- |
| 知识库 | `update`(跑完把本次发现写回知识库) | `collect`(只读不写) |
| LLM 看什么 | `video`(有视频就给画面) | `audio` |
| 跑完的中间产物 | 可勾选自动清理 | 一律保留 |

**只有命令行才有的**(要用就开 `finesub.cmd`,见下一节):

- **批量**:多个输入、`--manifest`、`--resume-batch`([`docs/manual/batch.md`](../docs/manual/batch.md))
- **翻译风格库**:`--style` / `--style-mode`,以及把人工精修喂回去的 `--refined-srt`
- **识别侧的细调**:`--qwen-verify`、`--asr-context`、`--lang-redecode`、`--vad-silero-assist`、
  `--asr-decode-batch`、`--gap`、`--separator-rate`([`docs/manual/tuning.md`](../docs/manual/tuning.md))
- **LLM 侧的细调**:`--llm-continuity` / `--llm-parallel-windows`(窗口并发)、`--llm-video`、
  `--extra-info-file`、`--knowledge-root`、两个重试预算
- **URL 只下音频**:`--no-download-video`

## 命令行

包根附带 `finesub.cmd`——**子命令与 pip 安装的 `finesub` 完全一致**，直接驱动它
所在的那份安装（同样的运行环境、模型、知识库）。适合批量处理、脚本调用，或者图形
界面起不来的时候。资源的安装与修复仍然只在应用内做。

```powershell
.\finesub.cmd doctor                 # 运行环境状态与各路径
.\finesub.cmd agent-clean            # 清理当前域保留的本地 Agent 失败现场
.\finesub.cmd input.mp4 --language ja
.\finesub.cmd relocate D:\FineSub    # 把模型/缓存/任务产物/Agent 现场搬到别的盘
```

## 更新

应用内检查并安装：小版本只换应用层（重启生效），大版本换整个安装（需要先退出
FineSub，由随包发布的 updater 完成）。个人数据、模型、缓存、任务产物都会保留。
也可以到 Release 页手动下载。

---

维护者请看 [README_DEV.md](README_DEV.md)：架构、依赖与 lock、外部工具、开发环境、
测试、构建与签名发布。
