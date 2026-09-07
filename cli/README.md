# FineSub CLI

FineSub 的命令行发行版：把长音频转成字幕（人声分离 → VAD+ASR 对齐 → 稳定化 →
SRT）。安装的是一个**轻量壳**——首次运行时它会在 Windows 的
`%LOCALAPPDATA%\FineSub` 或 macOS 的 `~/Library/Application Support/FineSub` 下自动装好
隔离的 Python 3.12 运行环境（含平台专用、锁定的 AI 依赖）和 FFmpeg，模型按需下载到同一目录。

当前验收平台为 Windows/NVIDIA CUDA 与 Apple Silicon macOS。macOS 自动使用 `mlx-refine`；
Linux 尚未完成 patched CTranslate2 分发和端到端验证，不能因 CLI 可启动就视为受支持。

用安装脚本装的话，装完会问一次模型和缓存放哪（回车用默认位置）；那一步只登记位置、
不下载东西，所以安装仍是几秒钟。详见
[数据放在哪](../docs/manual/resources.md)。

## 安装

已发布在 [PyPI](https://pypi.org/project/finesub/)：

```powershell
uv tool install finesub            # 升级：uv tool upgrade finesub
```

Apple Silicon macOS 在 Terminal 中使用相同命令；首次运行会按
`pylock.macos-arm64-py312.toml` 建立 MLX 运行环境。仓库分支尚未发布到 PyPI 时，请按
[`docs/manual/repo-install.md`](../docs/manual/repo-install.md) 从源码安装。

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
finesub input.wav --model large-v3-turbo --language en --gpu-tier standard
finesub a.wav b.mp4                # 多个输入就是批量，没有单独的子命令
finesub --manifest tasks.jsonl     # 每行一项，可逐项覆盖任何选项
finesub --resume-batch             # 接着跑上一个没跑完的批
finesub setup        # 只预装运行环境，不跑任务
finesub setup --dirs-only          # 只登记大文件位置，不下载任何东西
finesub setup --data-dir D:\FineSub --dirs-only   # 非交互场景直接指定
finesub doctor       # 查看运行环境状态与各路径（跑完整健康探针，需数秒）
finesub agent-clean  # 清理当前域保留的本地 Agent 失败现场（不安装/启动运行环境）
finesub agent-ping   # 逐个问一遍装好的 agent CLI 还答不答话（会花一点额度）
finesub agent-join   # 打印一段提示词，让你自己的 agent 接手一次等待中的运行
finesub agent-task   # 查看/操纵长驻 agent 任务（status、next-task、submit…）
finesub keys         # 查看 API key（默认打码）；换机器/重装前用 --out 导出明文
finesub relocate D:\FineSub   # 把模型/缓存/任务产物/Agent 现场搬到别的盘（运行环境留在原处）
finesub uninstall    # 删除运行环境/模型/缓存；成品字幕与个人数据分别需
                     # --purge-tasks / --purge-user-data
```

普通命令的健康检查是瞬时的文件系统检查（必需包目录 + CT2 补丁标签），启动接近
零开销；怀疑环境坏了就跑 `doctor`，它会做完整的 import 探针。

## 新版本提醒

命令跑完后，若 PyPI 上有更新的**正式版**，会在 stderr 打一行提示并给出升级命令
（`uv tool upgrade finesub`）。它只提醒，从不自动升级。

几条纪律，免得它碍事：

- **不阻塞**：查询在后台线程里跟命令并行跑，命令结束后最多再等 0.2 秒；没查完就用上次的答案。
- **失败完全静默**：没网、PyPI 不通、返回体不对——一个字都不打。
- **每天最多查一次**，结果缓存在共享数据目录的 `update-check.json`。
- **装完第一次运行不提醒**（那次只把缓存种下）；`setup` / `uninstall` / `--help` 也不提醒。
- **只在交互终端提醒**，stderr 被重定向或 `CI` 环境变量存在时不打。
- **只认正式版**：`0.5.0rc1` 这类预发布版本号不会被推荐。

关掉：环境变量 `FINESUB_NO_UPDATE_CHECK=1`，或在 `config.toml` 里写

```toml
[cli]
update_check = false
```

## 成品放在哪

每次任务都记入 `user-data\tasks.json`——**批量运行除外**：一次给几个输入、或用
`--manifest` / `--resume-batch` 时，这次运行不属于任何单个任务，产物按 `out/<名字>/` 落在**当前
目录**、不进 tasks（启动时会说一句），细节见 [`docs/manual/batch.md`](../docs/manual/batch.md)。
没有 `-o` 时，任务在
`tasks\<任务名>-<时间>-<随机后缀>\` 下进行；显式 `-o` 时则在指定位置运行，并在任务目录
保留续跑所需的记录，所以换个入口仍看得见同一批任务。

**重跑同一个源文件会接着上次继续**，不会新开一个任务。能复用时打印
`Continuing task ...`；归到同一个任务、但那个位置没有可复用产物时（比如换了 `-o`），
会明说这次从头跑。要复用的产物**超过 7 天没动**时，那行会加上 `(last run N days ago)`——
只是告诉你一声，不拦（这里是你自己指名了源文件；会拦的是不带 id 的 `--resume-batch`，
它挑的是你没点名的那一批，见 `docs/manual/batch.md`）。这正是调参重跑便宜的原因：人声分离和 ASR 的产物还在，管线会跳过它们。
反过来，**换了 `--model`、窗口预算、prompt/预设等参数重跑，已完成的阶段和已提交的 LLM 窗口
仍然会被跳过**；新参数只用于尚未完成的部分，结果允许混合配置（逐窗可审计）。FineSub 不会
为了让整条任务的模型、窗口或 prompt 看起来一致而丢掉有效结果。自动重做只发生在无法从断点
安全恢复时，例如源 `*-stable.json` 已变化、恢复计划/JSON 损坏或 schema 不兼容、已存响应
不能通过当前结构校验。要主动全量重来，仍需删除对应产物/断点，或换一个源路径。

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
`%LOCALAPPDATA%\FineSub\user-data`**，源码运行也读同一份——换个入口不会变成另一个
知识库。大文件（运行环境、模型、缓存、任务产物、Agent 失败现场）默认装在 `FINESUB_HOME` 下
（默认也是 `%LOCALAPPDATA%\FineSub`）。

小 C 盘用户有两条路：装之前设 `FINESUB_HOME` 指到别的盘（连运行环境一起过去，推荐），
或者装完用 `finesub relocate D:\FineSub` 搬走模型/缓存/任务产物/Agent 现场——后者会让缓存与运行环境
分处两盘、失去硬链接共享，反而多占约 5 GB，命令会当场提示。详见
[`docs/manual/resources.md`](../docs/manual/resources.md)。

本地 token 计数器（`tokcount`，约 9MB）不随 wheel 分发，而是在第一次跑 LLM 阶段
（`--stage translated-srt|final-srt`）时按需下载——有了它，规划与预算全在本地算，
dry-run 不需要联网也不需要 key。装不上不影响任务：计数退到免费的 countTokens 接口，
只是每次多一个网络往返。已经有自己那份的话设 `GEMINI_TOKEN_COUNTER_EXE`（或让
`tokcount` 在 PATH 上），就不会再下一份。

## 构建（维护者）

wheel 由 `cli/scripts/build-wheel.ps1` 产出：staging 目录里放入本包源码 +
`_vendor`（`src/finesub`、`src/llm`、`src/finesub_bootstrap` 快照、Windows 两份 pylock、
`pylock.macos-arm64-py312.toml`、`runtime-manifest.json`），版本号取自仓库根 `VERSION`
（版本号只有这一份）。构建机需要 `python -m build`。

```powershell
.\cli\scripts\build-wheel.ps1
```

uv 钉版必须与 `src/finesub_bootstrap/runtime-manifest.json` 一致，由
`test/test_packaging.py` 强制。
