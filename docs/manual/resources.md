# 资源与大文件：位置、迁移与删除

FineSub 安装完成后约占十几个 GB，其中绝大部分为可再生内容（Python 运行环境、AI 依赖、模型权重
与下载缓存）。本文介绍各类数据的位置、迁移方式，以及卸载时的删除范围。

## 两类数据，两个去处

| 数据 | 位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| 个人数据 `user-data` | `%LOCALAPPDATA%\FineSub\user-data` | 十几 MB | 设置(`config.toml`)、API Key(`.env`)、自定义模型表（可选的 `model_catalog.toml`，见[接自己的模型](custom-models.md)）、知识库、任务历史、日志。**安装版的各个入口共用这一份**，知识库不会因使用入口不同而各自独立；从源码仓库运行时默认用仓库自己的数据（见文末「从仓库源码运行时」） |
| 大文件 `models` / `cache` / `tasks` / `agent-capsules` | 默认在**安装目录**下，可整体搬走 | 约 12 GB | 模型权重、下载缓存、任务产物，以及本地 Agent 失败现场（有界的文本与 JSONL） |
| 运行环境 `runtime` | **永远**在安装目录下 | 约 5 GB | Python 3.12 + 锁定的 AI 依赖 |

**安装目录**是环境变量 `FINESUB_HOME` 指的目录，没设时就是 `%LOCALAPPDATA%\FineSub`。个人数据 `user-data` 不跟它走，永远在 `%LOCALAPPDATA%\FineSub\user-data`。

`finesub uninstall` 默认只删可再生的运行环境、模型与下载缓存，个人数据与成品字幕留着（见[清理](#清理)）。直接删掉整个 `%LOCALAPPDATA%\FineSub` 文件夹则会把 `user-data` 里的 API Key、知识库一起删掉。

### 为什么 `runtime` 不能单独搬

运行环境与下载缓存在同一个盘上时共用一份数据；只把大文件搬走、运行环境留在原盘，会多占约 5 GB。
所以 C 盘小时，最好在**第一次安装之前**把安装目录整个放到别的盘（见下一节）；已经装好了再搬，见[搬到别的盘](#搬到别的盘)。

## 第一次安装时选盘（仅安装版）

**想连运行环境一起放到 D 盘**，在装之前做：

1. 在 PowerShell 里把 `FINESUB_HOME` 设成用户环境变量，然后**开一个新终端**（旧终端看不到它）：

   ```powershell
   [Environment]::SetEnvironmentVariable("FINESUB_HOME", "D:\FineSub", "User")
   ```

2. 照常安装（`uv tool install finesub` 或安装脚本）。安装脚本问大文件放哪时直接回车，默认就是 `D:\FineSub`。

之后运行环境、模型、缓存与任务产物都在 `D:\FineSub`，FineSub 在 C 盘上的数据只剩十几 MB 的 `user-data`。⚠ 这个环境变量以后要一直留着：
删了它，FineSub 会回到 `%LOCALAPPDATA%\FineSub` 找安装目录，在 C 盘重装一份运行环境。

**只想把大文件放到别的盘**（运行环境留在 C 盘）：用安装脚本装 `finesub` 时，会在装完后问一次大文件放哪：

```text
FineSub 将在这里保存模型、下载缓存和任务产物,建议预留至少 20 GB。
直接回车使用:C:\Users\<你>\AppData\Local\FineSub
也可以输入其他绝对路径,例如 D:\FineSub
大文件位置:
```

回车就是用默认位置。这一步只登记位置、**不下载任何东西**，安装仍只要几秒钟；模型和运行环境
在第一次真正处理文件时才下载。

不方便交互时（CI、脚本、重定向的终端）不会等待输入，直接用默认位置，也可以事先指定：

```powershell
finesub setup --dirs-only --data-dir D:\FineSub   # 只登记位置
$env:FINESUB_BIG_DATA_DIR = "D:\FineSub"          # 或者用环境变量
```

`--data-dir` 与 `FINESUB_BIG_DATA_DIR` 只管大文件，运行环境仍在安装目录；它们仅在首次安装时生效，不会
迁移已存在的数据。安装后如需更改位置，请使用下文的 `finesub relocate`。

⚠ **Windows 用户名是中文时**，默认位置整条路径都含中文。系统代码页不是 UTF-8 的中文 Windows 上，
部分底层库会因此读错路径、报「文件不存在」。FineSub 已尽量绕开这个问题；首次安装遇到这种路径时会
提示一次。想彻底避开，就按本节开头的步骤把 `FINESUB_HOME` 设到纯英文路径（例如 `D:\FineSub`）——
运行环境跟着它走，只搬大文件目录不够。

## 搬到别的盘

已经装好之后，`finesub relocate` 把大文件（models / cache / tasks / agent-capsules）搬到别处，运行环境（约 5 GB）留在安装目录：

```powershell
finesub relocate --show              # 先看看现在在哪、各占多少
finesub relocate D:\FineSub          # 把 models/cache/tasks/agent-capsules 搬过去
finesub relocate --reset             # 搬回安装目录
```

运行环境也想离开 C 盘，只能在新位置重装一份：

1. `finesub relocate D:\FineSub` 先把大文件搬走；
2. `finesub uninstall`：大文件已搬走，默认留着，删掉的只是 C 盘的运行环境（个人数据与成品字幕也留着）；
3. 按[第一次安装时选盘](#第一次安装时选盘仅安装版)的第 1 步把 `FINESUB_HOME` 设成 `D:\FineSub`，开新终端；
4. 下次运行时 FineSub 在 `D:\FineSub` 下重装运行环境。

迁移前要求没有任务在跑、没有安装在进行；被拒绝时会说明占用方是哪个前端或任务（也可看 `finesub doctor` 的
`activity` 行）。

**中途断电或强制终止也不会丢数据**：程序会提示「上次搬迁未完成」，再跑一次 `finesub relocate` 就能完成。

**两个安装可共用一份数据**：在第二个安装中同样执行一次 `finesub relocate D:\FineSub` 即可。
若目标已是完整的数据目录，则仅登记、不复制。同一台机器上有两个安装时（例如两个
`FINESUB_HOME`），可借此避免重复下载两份模型。

**你自己做的目录联接(junction)会原样保留**：如果你用 `mklink /J` 把 `models` 或它下面的某个
子目录指到了别处，搬迁只会把这个链接搬过去、重新指向同一个位置，不会把目标里的数据复制一份
到新盘，也不会去动那个位置。它不属于 FineSub，可能还有别的程序在用。卸载同理。

### 用资源管理器搬也可以

- **移动大文件目录（如 `D:\FineSub`）** → 在新位置双击其中的 `register-location.cmd` 即可完成
  位置登记。该脚本与 `.finesub-store.json` 标记随目录一同迁移，无需使用命令行。
- ⚠ **别把整个 `%LOCALAPPDATA%\FineSub` 挪走**：个人数据 `user-data` 只在那里找，挪走后设置、API Key 与知识库都读不到；
  运行环境要换盘按上一节的步骤重装。

登记文件位于 `%LOCALAPPDATA%\FineSub\locations.json`，内容仅为若干路径记录，即使损坏也无碍：
找不到记录时自动回退到安装目录，不会报错。

## 从 0.4.x 桌面端迁移

桌面端自 0.5.0 起不再随本仓发版，也不会收到应用内更新提示（它只认带签名清单的 Release，
0.5.0 起没有，会安静地跳过）。留在 0.4.2 继续用没有问题；要换到 CLI 的话：

1. 装 CLI：`uv tool install finesub`（没有 uv 先 `winget install astral-sh.uv`）。
2. **个人数据不用导**：设置、API Key、知识库、任务历史都在 `%LOCALAPPDATA%\FineSub\user-data`，
   两端共用，装完直接就是原来的那一套（`.env` 里的密钥绑定 Windows 账户，同一台机器照常可读）。
3. **模型和缓存多半自动接上**：桌面端把自己的大文件目录登记在共用的 `locations.json` 里，CLI 读
   同一份记录。⚠ 但这个目录通常就是桌面端的**安装目录**，用桌面卸载器卸载会连模型一起删——
   所以**卸桌面之前**先 `finesub relocate D:\FineSub` 把 models / cache / tasks 搬到一个干净目录
   （跨盘是复制→校验→删源，中断不丢，见上节）。不搬也行，代价是重下约 12 GB。
4. **Python 运行环境（约 5 GB）不复用**：CLI 首次运行会在 `FINESUB_HOME\runtime` 下自己装一份，
   桌面那份随桌面卸载。
5. CLI 启动时若报「记录的数据目录已不存在」，把旧安装目录里的 `models`、`cache`、`tasks` 三个
   目录挪进一个新文件夹，双击里面的 `register-location.cmd`（或 `finesub relocate <该文件夹>`）
   即可——它只认「只含 FineSub 大文件的目录」，不接受整个旧安装目录（那里还有应用和运行环境）。

## 设置文件 `config.toml`

`.env` 存 API Key，`config.toml` 存其余设置——用哪些供应商、模型预设、分句参数之类。
它在**用户数据目录**里（就是上表 `user-data` 那一行的位置），和 `.env`、知识库同级。

**这个文件可选，默认没有**：没有它时一切用默认值；写了的只改写到的那几项，没写的照旧用默认值；命令行给了同一项以命令行为准。
多数人用不着它，要用时（如 [`agent.md`](agent.md) 里的 `preset = "agy"`、[接自己的模型](custom-models.md)）照下面新建。

### 新建

1. 打开用户数据目录（上表那个位置）。
2. 新建一个纯文本文件，文件名**精确**是 `config.toml`。
   ⚠ Windows 记事本默认会加 `.txt` 后缀，存成 `config.toml.txt` 就不会被读到——保存时把
   「保存类型」选成「所有文件」，或存好后在资源管理器里打开「文件扩展名」显示确认一遍。
3. 写你要改的那几行，例如：

```toml
[providers]
gemini_free = true
```

⚠ 请使用 UTF-8 编码（记事本默认）。修改后无需重启，下次运行即生效。

上面这个例子只演示写法（写的就是默认值，写了等于没改）。带注释的完整模板是源码仓库里的
[`config.example.toml`](https://github.com/caca2331/finesub/blob/main/config.example.toml)，不随安装包发布；
各个键的写法去下表那一行指的文档查，那里都有示例。

### 有哪些节


| 节 | 管什么 | 细节在哪 |
| --- | --- | --- |
| `[providers]` | 哪几家供应商可用（Gemini 免费/付费、Exa、Tavily） | [`env.md`](env.md) |
| `[pools]` | 同一家配多把 key 时的池子 | [`env.md`](env.md) |
| `[api]` | `proxy`：远程 API 请求（LLM 层与云端转写）走的代理 | [`model-routing.md`](model-routing.md) |
| `[asr]` | `backend`：本机转写还是哪家一体转写；`aligner`：MOSS 的长段与 35t 补回的文字用本机对齐模型对时间（默认开，有 GPU 时 MOSS 两家再由 Whisper 修一遍）；`repair`：云端漏转与复读的补法；`rows`、`overlap_cue`：`moss-diarization-aligner` 的行从哪家来、插话切开的行合不合成一条；`moss_pro_paid`：MOSS Pro 池里最后一把 key 超出免费额度后付费 | [`asr-backend-choice.md`](asr-backend-choice.md) |
| `[llm]` | 预设、优先模型、思考档位、风格默认值、本机 agent 的几项 | [`model-routing.md`](model-routing.md)、[`agent.md`](agent.md)、[`knowledge.md`](knowledge.md) |
| `[chunking]` | 准备任务分块、纠错窗口的字幕上限 | [接自己的模型「三个质量上限」](custom-models.md#三个质量上限) |
| `[segmentation]` | `length_scale`：字幕长短偏好 | [`tuning.md`](tuning.md) |
| `[vad]` | `silero_assist`：语音检测的二次校正开关 | [`tuning.md`](tuning.md) |
| `[separator]` | `enabled`：要不要跑人声分离（输入已是纯人声时设 `false`） | [`tuning.md`](tuning.md) |
| `[postprocess]` | `end_extend_seconds` / `flash_gap_seconds`：字幕结尾延长多少、多短的空隙直接合上 | [`tuning.md`](tuning.md) |
| `[share]` | `remote`：共享知识库的地址（不写就用内置的）；`signing_key`：拉你自己配的那个库时只认这把公钥签的快照（不写就记住第一次见到的；内置库始终用内置公钥） | [`knowledge.md`](knowledge.md) |
| `[cli]` | `update_check`：新版本提醒 | 本页「新版本提醒」 |
| `[runtime]` | `integrity_check`：什么时候核对运行环境；`auto_repair`：核对不上时是否自动重建 | 本页「运行环境的核对与重建」 |

**没有对应键的选项**：`--language`、`--stage`、`--model`（换 Whisper）在 `config.toml` 里没有键，只能写在命令行上；
批量时也可以写进 manifest 的每一行（见[一次跑多个输入「manifest 长什么样」](batch.md#manifest-长什么样)）。
`--knowledge`、`--llm-media`、`--llm-retrieval`、`--llm-difficulty`、`--llm-continuity` 这五个开关可以写成预设的开关默认值，
见[模型路由「预设带开关默认值」](model-routing.md#预设带开关默认值)。

## 下载走哪条线：镜像与手动覆盖

装模型和运行环境要下几个 GB。FineSub 会**自动**判断从官方源还是国内镜像取，判断结果只影响
「从哪儿取」，不影响取到什么——每个文件都有固定的哈希校验，镜像换不了内容。

看当前判断：

```powershell
finesub doctor
```

里面有一行：

```text
download   cn (自动检测，缓存)  pypi=…  huggingface=…  github=…
```

- 第一段是判定结果(`cn` / `global`)和它的来源（自动检测 / 你设的环境变量）。
- 后面每一类要么是正在用的镜像入口，要么是「官方源」，要么是「已停用（连续失败）」——某类
  镜像连续失败几次后会**自动**放弃、回到官方源，只影响那一类，不影响其它。

### 什么时候需要手动覆盖

使用 VPN 或公司代理时，自动检测的结果可能与实际出口不一致（例如判定为 `cn` 却无法连接镜像，或
判定为 `global` 却走了较慢的线路）。此时可通过环境变量**强制**指定区域，该设置始终优先于自动
检测：

| 环境变量 | 作用 |
| --- | --- |
| `FINESUB_DOWNLOAD_REGION` | `cn` / `global` / `auto`（默认）。非法值当没设 |
| `FINESUB_PYPI_INDEX` | 换 pypi 入口；**设成空值 = 这一类不用镜像，回官方源** |
| `FINESUB_HF_ENDPOINT` | 同上，模型权重 |
| `FINESUB_GITHUB_FILE_PROXY` | 同上，GitHub 上的固定文件 |

自己设了 `HF_ENDPOINT` 而那个网关支持 Hugging Face 的 Xet 传输时，设 `HF_HUB_DISABLE_XET=0`（不设时走镜像就关掉 Xet，公共镜像不代理它）。

```powershell
# 在这个终端里强制走官方源
$env:FINESUB_DOWNLOAD_REGION = "global"
finesub doctor
```

这样设只在当前这个 PowerShell 窗口里有效，但这个窗口里之后的每次运行都受影响；关掉窗口，或
`Remove-Item Env:FINESUB_DOWNLOAD_REGION`，就回到自动选择。设完再跑一次 `doctor` 确认那一行变了。要长期生效就设成系统环境变量。

## 清理

```powershell
finesub uninstall                    # 删运行环境、模型、下载缓存
finesub uninstall --purge-tasks      # 连成品字幕一起删
finesub uninstall --purge-user-data  # 连设置、API Key、知识库一起删
```

按「是否可再生」区分：运行环境、模型与缓存删除后会自动重新下载；成品字幕与个人数据删除后无法
恢复，因此需要显式指定。

**若大文件目录已迁移或与其他安装共用**，默认不会被删除（其他安装可能仍在使用）；如需删除，须
添加 `--purge-big-data`。

### 本地 Agent 失败现场

本地 agent 调用失败时留下的排查文件（下称现场：发出去的内容、返回的事件、错误日志；最多 20 个，没交出答案就退出的另留 10 个）放在大文件目录的 `agent-capsules` 下，
跟着 `finesub relocate` 搬，卸载规则与其他大文件相同。怎么清见[本机 Agent 后端](agent.md)「9. 磁盘上留下的内容」；
日志里每个现场都附一段 JSON 写的定位信息；把它原样贴进 `finesub agent-clean --locate '<那段 JSON>'`，会打印这个现场现在所在的目录（搬过盘也找得到），不删任何东西。

### 关于下载缓存

`cache\uv` 占数个 GB，但**单独删它几乎不省空间**：运行环境与它共用一份数据，要回收得**先删运行环境、再清缓存**。

下载目录里偶尔留下的 `<文件名>.part.bad` 是一次没通过校验的下载（之后已重新下载），**可随时删除**。

## 运行环境的核对与重建

运行环境（`runtime\python`）由 FineSub 管理：装什么、装哪个版本，由程序自带的依赖锁决定。FineSub 会
把它和锁逐个文件对照，对不上就按锁重建——不需要重新下载时几分钟就好。

**什么时候核对**（`config.toml` 的 `[runtime] integrity_check`）：

| 值 | 什么时候 |
| --- | --- |
| `auto`（默认） | 新装或升级 FineSub 之后，直到成功跑完一次转写之前，每次转写前都核对；转写失败之后也核对一次 |
| `on` | 每次转写前 |
| `off` | 不自动核对 |

核对只看文件在不在、包的版本对不对，以及同一个包有没有装了两份（中断的升级或手动 `pip install` 常留下这种残留），约几秒钟。`finesub doctor` 随时可以手动核对；加
`--verify-hashes` 还会逐个文件核对内容（约二十秒）。

**对不上时**：FineSub 先说明哪里不对、重建会去掉什么，再按锁重建。

- 文件缺失**多半是杀毒软件隔离了它们**：请把运行环境目录与下载缓存目录（`<FINESUB_HOME>\runtime`、`<FINESUB_HOME>\cache`，
  重建后仍失败的报错里会写出完整路径）加进杀毒软件的白名单，否则重建时还会再被隔离。
- 有别的 FineSub 任务正在运行时不会重建（它正在用这些文件），等它结束后再运行一次即可。
- `finesub setup --repair` 可以随时手动按锁重建，并在装完后逐个文件核对内容。

**直接改过运行环境的**（往里面装包、升级或卸载某个包），这些改动在重建时会被去掉，重建说明里会列出来。
需要额外的 Python 包时，把它们装在运行环境之外的目录，再用环境变量 `FINESUB_EXTRA_PYTHONPATH` 指过去
（多个目录用 `;` 分隔）：转写时它们排在 FineSub 自己的模块之后，重建也不影响它们。想完全自己管理运行
环境的，在 `config.toml` 里写：

```toml
[runtime]
auto_repair = false   # 对不上时只提示，不自动重建（包括升级 FineSub 带来的重建；第一次安装不受影响）
```

## 显卡支持范围与档位

### 哪些显卡能用

| | 型号 |
| --- | --- |
| NVIDIA | RTX 50 / 40 / 30 / 20 系（含 Ti、SUPER、笔记本版），GTX 1660、GTX 1650，以及数据中心的 V100 / A100 / H100 —— 显存需 ≥4GB |
| AMD（实验） | 装了 ROCm 版 PyTorch 的 Radeon RX 6800 / 6900、RX 7000 / 9000 系与 Strix 系列 APU（RX 6700 XT 等要靠 `HSA_OVERRIDE_GFX_VERSION`，见下文「AMD 显卡」）。Windows 上只覆盖 RX 7000 / 9000 系与 Strix APU，且只能源码装：识别靠 ROCm 版补丁 wheel 上卡（没在实卡上跑过），换回 `cu128` 那份时识别回到 CPU、其余阶段照常用卡；安装版装的是 CUDA 版 PyTorch，在 A 卡上整条走 CPU。见下方等级表 |
| Intel（实验） | 装了 XPU 版 PyTorch 的 Arc A / B 系独显与带 Arc 核显的 Core Ultra（PyTorch 的 Intel GPU 支持列表；Iris Xe、UHD 等老核显不在其中。需 Python 3.12，见下文「Intel 显卡」）。Intel 平台 `auto` 档位一律封顶 `standard`（核显的「显存」是共享内存，Arc 独显也没在 `standard` 以上标定过）；识别仍在 CPU |
| 不支持 | GTX 10 系及更早（1080 / 1070 / 1060 / 1050、GTX 9 系等），以及 Apple 的 GPU（Mac 上目前只走 CPU） |

### 平台与支持等级

| 平台 + 硬件 | 等级 | 装法 |
| --- | --- | --- |
| Windows + NVIDIA | 官方支持 | 安装版或源码 |
| Windows + CPU | 官方支持 | 安装版或源码 |
| Linux（WSL2）+ NVIDIA | 官方支持 | 源码 |
| Linux（原生）+ NVIDIA | 实验 | 源码 |
| Linux + AMD | 实验 | 源码 |
| Windows + AMD | 实验 | 源码，用 `[asr-win-rocm]`。AMD 的这条线是预览版：限 RX 7000 / 9000 系与 Strix APU、Windows 11、驱动 26.2.2、Python 3.12——见 [`repo-install.md`](repo-install.md)「Windows 上的 A 卡」 |
| Windows + Intel | 实验 | 源码，用 `[asr-win-xpu]`。需 Python 3.12：补丁版 CTranslate2 只有 cp312 的 Windows 轮子（XPU 的 torch 轮子本身另有 cp313/cp314）。上卡的只有分离、Silero VAD 辅助与语言重解用的 Qwen 裁判，`--qwen-verify` 的尾段校验随识别留在 CPU——见 [`repo-install.md`](repo-install.md)「Intel 显卡」 |
| macOS（Apple Silicon）+ CPU | 实验 | 源码 |

每个组合都有能装的 patched CTranslate2（[`ct2-wheel.md`](ct2-wheel.md)）。⚠ 两个 AMD 的 wheel 没有在任何一张 A 卡上跑过；
识别出问题时的退路见 [`repo-install.md`](repo-install.md)「其他平台」。

- **官方支持**：维护者在自己的机器上验收过。
- **社区验收**：有贡献者按[社区验收套件](../../tools/hw_acceptance/README.md)（要源码仓库）交回过记录，维护者看过。
  文档会写明最近一次验收时的版本；版本更新后不会自动降级。
- **实验**：能装能跑，但没人验收过。

⚠ **等级只是提示，不改变任何行为**：哪个阶段用显卡、什么默认开，与等级无关。它唯一的作用是：
某个阶段跑在未经验收的加速器上时，运行时会输出一条 `unverified-accelerator` 警告，说明是哪个阶段、
哪张卡，并告诉你两件事——结果若有异常，加 `--device cpu` 退回 CPU；以及去哪里反馈。

非 Windows 平台只支持源码装法（[`repo-install.md`](repo-install.md)），安装版只支持 Windows。

### AMD 显卡

AMD 卡通过 ROCm 版 PyTorch 使用，它沿用 `cuda` 这个设备名，所以 `--device cuda` 同样表示用 AMD 卡。

- **哪些卡有内核**：AMD 的 GPU 内核按架构（`gfx` 目标）分别编译，不能跨目标运行。所以判断的是
  「这张卡的目标在不在 PyTorch 的构建列表里」，不在就回退 CPU 并说明原因。RX 6700 XT 这类不在列表里、
  但与列表中某个目标接近的卡，可以按惯例设置 `HSA_OVERRIDE_GFX_VERSION`（例如 `10.3.0`），
  此时判断的就是覆盖后的目标。
- **识别**：需要打过补丁的 CTranslate2 的 ROCm 版（见 [`ct2-wheel.md`](ct2-wheel.md)），它的构建目标是
  RX 6800 / 6900、RX 7000 / 9000 系与 Strix 系列 APU。装的是 CUDA 版或 CPU 版时，识别回退 CPU，其余阶段照常用卡。
- **有核显又有独显**：程序不会自己换卡，用的是 0 号设备（常常是核显）。若有显存更大的卡，运行时会提示一次
  `amd-device-choice`，告诉你该设置 `HIP_VISIBLE_DEVICES=<编号>`；设置后程序只看得见那张卡。已经设置了
  `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、`ROCR_VISIBLE_DEVICES` 或 `GPU_DEVICE_ORDINAL` 中任一个时不提示。
- **第二模型校验**：显卡不原生支持 bfloat16（RDNA3 之前）时改用 float32；编译加速在 AMD 卡上默认不开。
- **人声分离**：Windows ROCm GPU 使用普通注意力路径，避开 SDPA 的 HIP 内核错误，可能更慢、占用更多显存。
  Linux 与显式 CPU 模式保持原有路径；当前版本尚未在实卡上复验。

### Intel 显卡

Intel 卡走 XPU 版 PyTorch，设备名就是它自己的 `xpu`：`--device xpu`。只有 Intel 卡、却按习惯写了
`--device cuda` 时，程序会落到这张卡上，不告警——机器上能加速的卡就这一张，把活放上去才是请求的本意。

- **哪些阶段用卡**：人声分离、Silero VAD 辅助（`--vad-silero-assist`，默认开）与语言重解（`--lang-redecode`，
  默认 `auto`，只在语言自动识别的那类运行里生效）用的 Qwen 裁判，和 NVIDIA / AMD 一样。分离器在 Intel 卡上
  不用 NVIDIA 上的编译加速：那条路径没在 Intel 卡上跑过。
  **识别仍在 CPU 上**：CTranslate2 没有 Intel GPU 后端，这一阶段会输出一条 `cpu-fallback` 警告说明原因，
  其余阶段照常用卡——不是整条管线退回 CPU。
- **用哪张卡**：程序用 0 号 Intel 设备，不会自己换卡。有两张 Intel 卡（例如核显加 Arc 独显）时，用
  `ZE_AFFINITY_MASK=<编号>` 只留下要用的那张。
- **`--qwen-verify` 的尾段二次校验不跟卡**：它按解析后的识别设备派生，Intel 上就是 CPU + float32——换设备会改一条
  数值路径，而「CPU-float32 对 GPU-bf16」这一对还没量过（见 [`asr-align.md`](../asr-align.md)「待标定」），所以这一步留在 CPU。
- **档位**：Intel 平台一律封顶（`gpu_device() == "xpu"` 就压到 `standard`，不分核显/独显）：核显的「显存」是
  共享系统内存，而 Arc 独显也没有在 `standard` 以上标定过，按它定档会把分离 worker 开多；`--gpu-tier` 显式指定
  不受此限。
- **装法与前提**：要 **Python 3.12** 的原因只有一个——补丁版 CTranslate2 只有 cp312 的 Windows 轮子
  （XPU 的 torch 轮子本身另有 cp313/cp314）。步骤见 [`repo-install.md`](repo-install.md)「Intel 显卡」。
- **说话人分离与强制对齐**：`--diarize` 在 Intel 上仍跑 CPU——这条链路只在 CUDA 上验收过，换到没验收过的后端等于换算法，
  会输出 `diarization-on-cpu` 告警；`--asr-aligner` 的强制对齐同理留在 CPU，并输出 `cpu-fallback` 告警。
  其余阶段照常用卡。见[说话人分离](diarization.md)。

不支持的显卡**不会报错，而是自动回退到 CPU**，并在 stderr 输出一条 `Warning:`。回退后仍能生成
正确的字幕，但速度明显变慢（人声分离尤甚），长音频不建议采用此方式；如需从一开始就避免这些警告，
可直接显式指定 `--device cpu`。回退并非整体发生，各阶段的回退情况见下文「这张卡能不能用不是
一个问题，是两个」。

显存自 4GB 起即可使用（`entry` 档要求 3GB **空闲**显存）；更大的显存仅在人声分离阶段带来并行
加速，边际收益有限——档位默认自动检测，详见下文。

### 显卡档位与 CPU 回退

`--gpu-tier` 决定人声分离同时处理几段音频（段数越多越占显存）、以及第二模型校验能用多少显存，
**默认 `auto`，一般不用管**。五档：

| 档位 | 需要**空闲**显存 | 内存 | 分离同时处理几段 | `auto` 在多大的卡上选它 |
| --- | ---: | ---: | ---: | ---: |
| `cpu` | 不用显卡 | 8GB | 1 | 机器上没有显卡 |
| `entry` | 3GB | 8GB | 1 | <8GB |
| `standard` | 6.5GB | 8GB | 2 | 8–11GB |
| `standard_large_vram` | 10GB | 8GB | **2** | 12–23GB |
| `high` | 10GB | 8GB | 3 | ≥24GB |

显存那一栏是**空闲**量，不是卡的容量：桌面和驱动本来就占着一部分。

**Intel 显卡上 `auto` 封顶 `standard`**：Intel 平台一律封顶（`gpu_device() == "xpu"` 就压，不分核显/独显）——
核显的「显存」是共享系统内存，驱动报出的总量不是可独占的容量；而 Arc 独显也没有在 `standard` 以上标定过，
按它定档会把分离 worker 开多。显式 `--gpu-tier` 不受此限。

**`cpu` 档**表示「本次不使用显卡」，没有显卡的机器自动选它；手动指定的常见场景是把整张卡让给别的程序。
⚠ **`--gpu-tier cpu --device cuda` 会直接报错**（二者矛盾）；「GPU 档位 + `--device cpu`」是正常组合，也表示本次不用显卡。

**`standard_large_vram` 与 `standard` 的区别只在给第二模型校验的显存**：主模型换成 `large-v3`、日语微调这类大模型时，
校验的快速版本要它。手动指定 `high` 始终允许，但通常不会更快。

**`entry` 档表示「基础 GPU 加速」，并非「弱卡专用」。** 以下三种情况都会选择该档，但结果不同
（无显卡的机器不在此列——它会选择 `cpu` 档）:

| 情况 | 会怎样 |
| --- | --- |
| 显卡支持、空闲显存 ≥3GB | 拿到基础 GPU 加速，正常跑 |
| 当前 PyTorch 版本不支持该显卡 | **人声分离、第二模型校验、silero 辅助回退到 CPU** 并输出 `Warning:`；识别不随之回退，由 CTranslate2 自行判断（见下） |
| 显卡可用但空闲显存不足 | **不会自动切换 CPU**（那会改变产物切分），而是输出警告说明差距，随后继续运行 |

空闲显存略低于要求的卡多数素材照样跑得完。真遇到显存不足而中止（这种中止程序拦不住），关掉占显存的程序、降低档位，
或显式指定 `--device cpu`。

### 「这张卡能不能用」不是一个问题，是两个

人声分离和第二模型校验用 PyTorch，**识别用 CTranslate2**，两边各自判断能不能用这张卡，所以可能只有一部分阶段回退 CPU，
回退时都会说明原因。CTranslate2 连 CPU 后端都没有时，任务**直接报错**并指向 [`ct2-wheel.md`](ct2-wheel.md)。

**GTX 10 系及更早的老显卡仍可完整运行**：三个语音阶段都在 CPU 上跑，结果正确，只是慢；不想看那些回退警告，
就从一开始加 `--device cpu`。**比编译范围更新的卡**（如 RTX 50 系）不会被拦。

## 新版本提醒（只有安装版有）

`finesub` 命令运行结束后，若 PyPI 存在更新的**正式版**（`0.5.0rc1` 这类预发布版不算），会在 stderr 输出一行提示，并给出
`uv tool upgrade finesub`。该机制仅提醒、从不自动升级：

- 每天至多联网查一次；查询与命令在后台并行，命令结束后最多再等 0.2 秒，没查完就用上次的结果。失败（没网、PyPI 不通）时完全静默。
- 安装后的第一次运行照样查、把结果存下，但不提醒。
- 只在交互终端里查和提醒：输出被重定向、设了 `CI` 环境变量，或跑的是 `setup`、`uninstall`、`--help` 时都不查。

检查结果缓存在共享数据根目录的
`update-check.json`（与 `tasks/` 的索引同级），**可随时删除**——删除后仅会使下一次重新检查。

关掉：设环境变量 `FINESUB_NO_UPDATE_CHECK=1`，或在 `config.toml` 里写：

```toml
[cli]
update_check = false
```

## 从仓库源码运行时

从源码 checkout 运行时（运行的代码本身就在这个 checkout 里），FineSub 使用**仓库自身的数据**(`<repo>/knowledge`、`<repo>/.env`、
`<repo>/.state`)，完全不访问 `%LOCALAPPDATA%`，开发过程中的产物不会混入日常使用的知识库。安装版的 `finesub`
在某个 checkout 目录里运行不算：它照常用安装自己的数据。
如需仓库运行同样接入共享数据，请设置 `FINESUB_CHECKOUT_DATA=0`。

源码运行用的是你自己装的 Python 环境，不做[运行环境的核对与重建](#运行环境的核对与重建)里的核对，也不会重建
它。装在中文路径下时 FineSub 绕开读错路径的那几处处理同样生效，但[第一次安装时选盘](#第一次安装时选盘仅安装版)里那条
提示只有安装版才打。

源码运行时本地 agent 的失败现场放在 `%TEMP%\finesub-agent-runtime\` 下（不进仓库），用
`python -m finesub.llm.agent.agent_cleanup` 清。
