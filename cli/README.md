# FineSub CLI

FineSub 的命令行发行版：把长音频转成字幕（人声分离 → VAD+ASR 对齐 → 稳定化 →
SRT）。安装的是一个**轻量壳**——首次运行时它会在 `%LOCALAPPDATA%\FineSub` 下
自动装好隔离的 Python 3.12 运行环境（含锁定的 AI 依赖）和 FFmpeg，模型按需下载
到同一目录。装过 FineSub Desktop（安装器版）的机器还会共享它的设置与 API Key。

用安装脚本装的话，装完会问一次模型和缓存放哪（回车用默认位置）；那一步只登记位置、
不下载东西，所以安装仍是几秒钟。详见
[数据放在哪](../docs/manual/resources.md)。

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

参数与仓库开发版的 `python -m finesub.pipeline` 完全一致：

```powershell
finesub input.wav --model large-v3-turbo --language en --gpu-budget-gb 8
finesub batch --manifest tasks.jsonl
finesub setup        # 只预装运行环境，不跑任务
finesub setup --dirs-only          # 只登记大文件位置，不下载任何东西
finesub setup --data-dir D:\FineSub --dirs-only   # 非交互场景直接指定
finesub doctor       # 查看运行环境状态与各路径（跑完整健康探针，需数秒）
finesub agent-clean  # 清理当前域保留的本地 Agent 失败现场（不安装/启动运行环境）
finesub relocate D:\FineSub   # 把模型/缓存/任务产物/Agent 现场搬到别的盘（运行环境留在原处）
finesub uninstall    # 删除运行环境/模型/缓存；成品字幕与个人数据分别需
                     # --purge-tasks / --purge-user-data
```

普通命令的健康检查是瞬时的文件系统检查（必需包目录 + CT2 补丁标签），启动接近
零开销；怀疑环境坏了就跑 `doctor`——它永远做完整 import 探针。

## 成品放在哪

每次任务都记入和桌面版共用的 `user-data\tasks.json`。没有 `-o` 时，任务在
`tasks\<任务名>-<时间>-<随机后缀>\` 下进行；显式 `-o` 时则在指定位置运行，并在任务目录
保留续跑所需的记录，所以换个入口仍看得见同一批任务。

**重跑同一个源文件会接着上次继续**，不会新开一个任务——真能复用时打印 `Continuing task ...`，
只是归到同一个任务、但那个位置没有可复用产物时（比如换了 `-o`），会明说这次从头跑。
这正是调参重跑便宜的原因：人声分离和 ASR 的产物还在，管线会跳过它们。反过来说，
**换了 `--model`、窗口预算、prompt/预设等参数重跑，已完成的阶段和已提交的 LLM 窗口仍然会被跳过**；
新参数只用于尚未完成的部分，结果允许带有可审计的混合配置。FineSub 不会为了让整条任务的模型、
窗口或 prompt 看起来一致而丢掉有效结果。自动重做只发生在无法从断点安全恢复时，例如源
`*-stable.json` 已变化、恢复计划/JSON 损坏或 schema 不兼容、已存响应不能通过当前结构校验。
要主动全量重来，仍需删除对应产物/断点或换一个源路径。

**加 `-o` 就在你指定的位置跑**，产物照常一个个出现在那里（人声 → 对齐数据 → 字幕），
中途失败也留在那儿。你的目录 FineSub 不动一个文件。跑完只是把 `-stable.json` 和
（有 LLM 阶段时的）`-annotated.csv` **抄一份**进任务目录 —— 这两个再生不了，而任务记录
需要它们，下次续跑也读它们。

源文件必须写在最前面（`finesub 输入 [选项...]`）。写在选项后面的话 FineSub 认不出哪个
是输入，会退回旧行为——产物落在当前目录的 `out\` 下，并打印一行提示。

运行时输出可以调：`--log-level quiet`（只留告警和最终路径）／`normal`（默认：阶段、
进度、阶段摘要）／`verbose`（再加耗时、资源占用和逐步的算法恢复细节）。也可以用
环境变量 `FINESUB_LOG_LEVEL` 设默认值。

彻底移除：`finesub uninstall` 之后 `uv tool uninstall finesub`。

**个人数据（设置、API Key、知识库、任务历史）永远在
`%LOCALAPPDATA%\FineSub\user-data`**，与桌面端共用同一份——换个入口不会变成另一个
知识库。大文件（运行环境、模型、缓存、任务产物、Agent 失败现场）默认装在 `FINESUB_HOME` 下
（默认也是 `%LOCALAPPDATA%\FineSub`）。

小 C 盘用户有两条路：装之前设 `FINESUB_HOME` 指到别的盘（连运行环境一起过去，推荐），
或者装完用 `finesub relocate D:\FineSub` 搬走模型/缓存/任务产物/Agent 现场——后者会让缓存与运行环境
分处两盘、失去硬链接共享，反而多占约 5 GB，命令会当场提示。详见
[`docs/manual/resources.md`](../docs/manual/resources.md)。

（桌面版包根另有 `finesub.cmd`，子命令与这里同源，直接驱动它所在的那份安装；
只有它装不了资源——那仍归应用内的资源面板。）
本地 token 计数器（`tokcount`，约 9MB）不随 wheel 分发，而是在第一次跑 LLM 阶段
（`--stage translated-srt|final-srt`）时按需下载——有了它，规划与预算全在本地算，
dry-run 不需要联网也不需要 key。装不上不影响任务：计数退到免费的 countTokens 接口，
只是每次多一个网络往返。已经有自己那份的话设 `GEMINI_TOKEN_COUNTER_EXE`（或让
`tokcount` 在 PATH 上），就不会再下一份。

## 构建（维护者）

wheel 由 `cli/scripts/build-wheel.ps1` 产出：staging 目录里放入本包源码 +
`_vendor`（`src/finesub`、`src/llm`、`src/finesub_bootstrap` 快照、
`pylock.win-py312.toml`、`runtime-manifest.json`），版本号取自 `desktop/VERSION`
（CLI 与桌面同版本、同 tag、同 Release）。构建机需要 `python -m build`。

```powershell
.\cli\scripts\build-wheel.ps1
```

uv 钉版必须与 `desktop/resources/runtime-manifest.json` 一致，由
`desktop/scripts/tests/test_desktop_dependencies.py` 强制。
