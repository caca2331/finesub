# FineSub CLI

FineSub 的命令行发行版：把长音频转成字幕（人声分离 → VAD+ASR 对齐 → 稳定化 →
SRT）。安装的是一个**轻量壳**——首次运行时它会在 `%LOCALAPPDATA%\FineSub` 下
自动装好隔离的 Python 3.12 运行环境（含锁定的 AI 依赖）和 FFmpeg，模型按需下载
到同一目录。

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
finesub input.mp4 --language ja --stage final-srt   # 日语视频出中文字幕（要先配好 API key，见首页）
finesub input.wav --model large-v3-turbo --language en --gpu-tier standard   # 指定识别模型、语言与显卡档位
finesub a.wav b.mp4                # 多个输入就是批量，没有单独的子命令
finesub --manifest tasks.jsonl     # 每行一项，可逐项覆盖任何选项
finesub --resume-batch             # 接着跑上一个没跑完的批
finesub setup        # 只预装运行环境，不跑任务
finesub setup --dirs-only          # 只登记大文件位置，不下载任何东西
finesub setup --data-dir D:\FineSub --dirs-only   # 非交互场景直接指定
finesub setup --repair             # 按依赖锁重建运行环境，装完逐个文件核对内容
finesub doctor       # 查看运行环境状态与各路径（跑完整健康探针，需数秒）
finesub doctor --verify-hashes     # 连运行环境里每个文件的内容一起核对（约二十秒）
finesub keys         # 查看 API key（默认打码）；换机器/重装前用 --out 导出明文
finesub quota        # 列出因额度用完被搁置的云端转写 key；thaw <名字> / --all 提前放回
finesub relocate D:\FineSub   # 把模型/缓存/任务产物/Agent 现场搬到别的盘（运行环境留在原处）
finesub uninstall    # 删除运行环境/模型/缓存；成品字幕与个人数据分别需
                     # --purge-tasks / --purge-user-data
```

用本机 agent（Codex、Claude Code、agy 等，见[本机 Agent 后端](../docs/manual/agent.md)）跑 LLM 阶段时才用得上的几条：

```powershell
finesub agent-clean  # 删掉本机 agent 调用失败时留下的排查文件（只清这套安装的）；不安装、不启动运行环境，环境缺失或损坏时也能用
finesub agent-ping   # 逐个问一遍装好的 agent CLI 还答不答话（会花一点额度）
finesub agent-join   # 打印一段提示词，让你自己的 agent 接手一次等待中的运行
finesub agent-task   # 查看/操纵长驻 agent 任务（status、next-task、submit…）
```

怀疑环境坏了就跑 `finesub doctor`，它会把运行环境完整检查一遍。

命令跑完后，若 PyPI 上有更新的**正式版**，会在 stderr 打一行升级命令，从不自动升级；
怎么关见 [新版本提醒](../docs/manual/resources.md#新版本提醒只有安装版有)。

## 成品放在哪

每次任务都记入 `user-data\tasks.json`——**批量运行除外**：一次给几个输入、或用
`--manifest` / `--resume-batch` 时，这次运行不属于任何单个任务，产物按 `out/<名字>/` 落在**当前
目录**、不进 tasks（启动时会说一句），细节见 [`docs/manual/batch.md`](../docs/manual/batch.md)。
没有 `-o` 时，任务在
`tasks\<任务名>-<时间>-<随机后缀>\` 下进行；显式 `-o` 时则在指定位置运行，并在任务目录
保留续跑所需的记录，所以换个入口仍看得见同一批任务。

**重跑同一个源文件会接着上次继续**，不会新开一个任务。能复用时打印
`Continuing task ...`；归到同一个任务、但那个位置没有可复用产物时（比如换了 `-o`），
会明说这次从头跑。要复用的产物**超过 7 天没动**时，那行会加上 `(last run N days ago)`，
只是告诉你一声，不拦。中断后续跑不重做已完成的阶段和已提交的 LLM 窗口（LLM 分批送给模型的一段字幕，结果已经存下）；
反过来，改了参数直接重跑，已完成的部分**不会**按新参数重做，要先删对应产物，见
[重跑某一阶段](../docs/manual/outputs.md#重跑某一阶段需删除的产物)。

**加 `-o` 就在你指定的位置跑**，产物照常一个个出现在那里（人声 → 对齐数据 → 字幕），
中途失败也留在那儿。FineSub 不把它们挪走、也不删；每次跑完（失败、中断也一样）把其中几份**抄一份**进任务目录、
盖掉上次抄的：`-stable.json`、`-annotated.csv`，以及跑过说话人分离时的 `-diar.json`、`-diar-rows.json`、`-speaker-judge.json`。
这几份重做要花大量时间或额度，重做的结果也不会和原来一模一样，所以任务目录里留一份。

用过 `-o` 的任务，再跑时读哪一份、要重做时删哪一份：

- **再跑时给同一个 `-o`**：只读那个位置的文件，任务目录里抄的那份不读。要重做某一阶段，就删 `-o` 位置里的产物；
  任务目录里的不用管，这次跑完会被新的盖掉。
- **再跑时不给 `-o`**：在任务目录里跑，那里只有抄过去的几份。识别从 `-stable.json` 接着来、不重跑；
  纠错要重跑、花额度（只有 `-annotated.csv` 不够跳过纠错）。

源文件必须写在最前面（`finesub 输入 [选项...]`）。写在选项后面的话 FineSub 认不出哪个
是输入，会退回旧行为——产物落在当前目录的 `out\` 下，并打印一行提示。

运行时输出可以调：`--log-level quiet`（只留告警和最终路径）／`normal`（默认：阶段、
进度、阶段摘要）／`verbose`（再加耗时、资源占用和逐步的算法恢复细节）。也可以用
环境变量 `FINESUB_LOG_LEVEL` 设默认值。

**个人数据（设置、API Key、知识库、任务历史）在 `%LOCALAPPDATA%\FineSub\user-data`**，
安装版的各个入口都读这一份；从源码仓库运行时默认用仓库自己的数据，见
[`docs/manual/resources.md`](../docs/manual/resources.md)「从仓库源码运行时」。大文件（运行环境、模型、缓存、
任务产物、Agent 失败现场）默认也在 `%LOCALAPPDATA%\FineSub`；C 盘小时怎么放到别的盘，见同一份的
「第一次安装时选盘（仅安装版）」与「搬到别的盘」。

## 卸载

`finesub uninstall` 要在 `uv tool uninstall finesub` 之前跑：程序卸掉之后就没有这个命令了。

```powershell
# 只卸程序：成品字幕与个人数据（设置、API Key、知识库、任务历史）留着
finesub uninstall
uv tool uninstall finesub

# 连成品字幕与个人数据一起删
finesub uninstall --purge-tasks --purge-user-data
uv tool uninstall finesub
```

大文件搬过盘（`finesub relocate`）的，模型与缓存默认也留着，再加 `--purge-big-data` 才删；各项删什么、留什么见
[资源与大文件「清理」](../docs/manual/resources.md#清理)。

## 构建（维护者参考，用户可跳过）

wheel 由 `cli/scripts/build-wheel.ps1` 产出：staging 目录里放入本包源码 +
`_vendor`（`src/finesub`、`src/llm`、`src/finesub_bootstrap` 快照、
`pylock.win-py312.toml`、`runtime-manifest.json`），版本号取自仓库根 `VERSION`
（版本号只有这一份）。构建机需要 `python -m build`。

```powershell
.\cli\scripts\build-wheel.ps1
```

uv 钉版必须与 `src/finesub_bootstrap/runtime-manifest.json` 一致，由
`test/test_packaging.py` 强制。

普通命令启动时的健康检查只看文件系统（必需包目录 + CT2 补丁标签），接近零开销；
`finesub doctor` 才做完整的 import 探针。

新版本提醒在 `finesub_bootstrap/update_check.py`，它什么时候查、什么时候提醒见
[资源与大文件「新版本提醒」](../docs/manual/resources.md#新版本提醒只有安装版有)。
