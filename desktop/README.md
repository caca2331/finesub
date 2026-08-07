# FineSub Desktop

FineSub Desktop 是 FineSub 的可选 Windows 客户端：用图形界面创建任务、管理资源、
查看日志。它跑的是同一套 pipeline（`src/asr_playground/pipeline.py`，在隔离的
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

## 命令行

包根附带 `finesub.cmd`——**子命令与 pip 安装的 `finesub` 完全一致**，直接驱动它
所在的那份安装（同样的运行环境、模型、知识库）。适合批量处理、脚本调用，或者图形
界面起不来的时候。资源的安装与修复仍然只在应用内做。

```powershell
.\finesub.cmd doctor                 # 运行环境状态与各路径
.\finesub.cmd input.mp4 --language ja
.\finesub.cmd relocate D:\FineSub    # 把模型/缓存/任务产物搬到别的盘
```

## 更新

应用内检查并安装：小版本只换应用层（重启生效），大版本换整个安装（需要先退出
FineSub，由随包发布的 updater 完成）。个人数据、模型、缓存、任务产物都会保留。
也可以到 Release 页手动下载。

---

维护者请看 [README_DEV.md](README_DEV.md)：架构、依赖与 lock、外部工具、开发环境、
测试、构建与签名发布。
