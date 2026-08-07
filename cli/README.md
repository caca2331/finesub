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
finesub relocate D:\FineSub   # 把模型/缓存/任务产物搬到别的盘（运行环境留在原处）
finesub uninstall    # 删除运行环境/模型/缓存；成品字幕与个人数据分别需
                     # --purge-tasks / --purge-user-data
```

普通命令的健康检查是瞬时的文件系统检查（必需包目录 + CT2 补丁标签），启动接近
零开销；怀疑环境坏了就跑 `doctor`——它永远做完整 import 探针。

彻底移除：`finesub uninstall` 之后 `uv tool uninstall finesub`。

**个人数据（设置、API Key、知识库、任务历史）永远在
`%LOCALAPPDATA%\FineSub\user-data`**，与桌面端共用同一份——换个入口不会变成另一个
知识库。大文件（运行环境、模型、缓存、任务产物）默认装在 `FINESUB_HOME` 下
（默认也是 `%LOCALAPPDATA%\FineSub`）。

小 C 盘用户有两条路：装之前设 `FINESUB_HOME` 指到别的盘（连运行环境一起过去，推荐），
或者装完用 `finesub relocate D:\FineSub` 搬走模型/缓存/任务产物——后者会让缓存与运行环境
分处两盘、失去硬链接共享，反而多占约 5 GB，命令会当场提示。详见
[`docs/manual/resources.md`](../docs/manual/resources.md)。

（桌面版包根另有 `finesub.cmd`，子命令与这里同源，直接驱动它所在的那份安装；
只有它装不了资源——那仍归应用内的资源面板。）
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
