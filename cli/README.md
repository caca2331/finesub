# FineSub CLI

FineSub 的命令行发行版：把长音频转成字幕（人声分离 → VAD+ASR 对齐 → 稳定化 →
SRT）。安装的是一个**轻量壳**——首次运行时它会在 `%LOCALAPPDATA%\FineSub` 下
自动装好隔离的 Python 3.12 运行环境（含锁定的 AI 依赖）和 FFmpeg，模型按需下载
到同一目录。装过 FineSub Desktop（安装器版）的机器还会共享它的设置与 API Key。

## 安装

已发布在 [PyPI](https://pypi.org/project/finesub/)：

```powershell
uv tool install finesub            # 升级：uv tool upgrade finesub
```

没有 uv 的机器可用一条命令（[cli/install.ps1](install.ps1)：先装 uv 再装
finesub，重跑即升级）：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/caca2331/finesub/main/cli/install.ps1 | iex"
```

不想用 uv 的话 `pipx install finesub` / 自建 venv `pip install finesub`
也行，壳很轻。

## 使用

参数与仓库开发版的 `asr-pipeline` 完全一致：

```powershell
finesub input.wav --model large-v3-turbo --language en --gpu-budget-gb 8
finesub batch --manifest tasks.jsonl
finesub setup        # 只预装运行环境，不跑任务
finesub doctor       # 查看运行环境状态与各路径（跑完整健康探针，需数秒）
finesub uninstall    # 删除运行环境/模型/缓存；个人数据需 --purge-user-data
```

普通命令的健康检查是瞬时的文件系统检查（必需包目录 + CT2 补丁标签），启动接近
零开销；怀疑环境坏了就跑 `doctor`——它永远做完整 import 探针。

彻底移除：`finesub uninstall` 之后 `uv tool uninstall finesub`。

环境变量 `FINESUB_HOME` 可改数据根（默认 `%LOCALAPPDATA%\FineSub`；小 C 盘用户
可指到别的盘）。指向 FineSub Desktop 的安装目录即可复用它已下载的运行环境和模型。
本地 token 计数器（`tokcount`）不随 wheel 分发，计数自动退到免费的 countTokens
接口；要用本地版就设 `GEMINI_TOKEN_COUNTER_EXE`。

## 构建（维护者）

wheel 由 `cli/scripts/build-wheel.ps1` 产出：staging 目录里放入本包源码 +
`_vendor`（`src/asr_playground`、`src/llm`、`src/finesub_bootstrap` 快照、
`pylock.win-py312.toml`、`runtime-manifest.json`），版本号取自 `desktop/VERSION`
（CLI 与桌面同版本、同 tag、同 Release）。构建机需要 `python -m build`。

```powershell
.\cli\scripts\build-wheel.ps1
```

uv 钉版必须与 `desktop/resources/runtime-manifest.json` 一致，由
`desktop/scripts/tests/test_desktop_dependencies.py` 强制。
