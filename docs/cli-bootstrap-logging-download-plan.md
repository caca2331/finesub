# CLI 首次初始化、pipeline 日志与依赖/模型下载加速方案

状态：P1–P5 已实施。`model-manifest.json` 与 `download-sources.json` 都已填入**经实测**的值
（模型摘要来自官方源重下比对与 Hub 元数据 API；镜像候选做过可用性与内容校验）。§5.4 的前提
已实测成立（见该节）。

**仍未做**：发布验收——全量 sha256 比对（抽查只覆盖 12 个小 wheel 的完整摘要）、`cn`/`global`
各一次全新安装、断网恢复、中途切源、破坏哈希，且需要在大陆出口测。非大陆机器行为与实施前完全
一致（`auto` 解析为 `global`，cn lock 与镜像一个都用不到）。

本文收敛三个相互关联的用户体验改动：CLI 首次初始化时选择大文件位置、降低
pipeline 默认日志噪声、以及按出口 IP 为依赖包和模型选择适合中国大陆的公共下载入口。
它是实施边界和验收契约，不描述已经存在的行为；现状仍以
[`manual/resources.md`](manual/resources.md)、[`desktop/README_DEV.md`](../desktop/README_DEV.md)
和代码为准。

## 1. 目标与边界

### 1.1 要解决的问题

1. CLI 第一次真正下载资源之前，用户能决定 `models` / `cache` / `tasks` 放在哪个磁盘，
   而不是下载完成后才知道要用 `finesub relocate`。
2. 默认日志只保留当前阶段、可理解的进度、影响结果的警告和最终产物；算法调查所需的
   细节仍完整可查，但不再占满终端和 Desktop 日志抽屉。
3. `auto` 模式根据当前出口 IP 选择 global 或 cn 下载路线。第一阶段不部署 FineSub
   自有镜像，只利用经过验证的公共入口加速依赖包和模型。

### 1.2 下载加速的明确范围

| 下载类别 | 本方案是否处理 | 说明 |
| --- | --- | --- |
| `uv tool install finesub` 使用的 PyPI 包 | 是 | `cli/install.ps1` 在 cn 路线下给 uv 传大陆 PyPI index |
| managed Python 与 `pylock` 中的 Python/AI 依赖 | 是 | 包括普通 PyPI wheel；Torch、patched CT2 等绝对 URL 单独处理 |
| BS-Roformer、faster-whisper、Qwen referee 模型 | 是 | Hugging Face 走可配置 endpoint；固定 GitHub 模型文件走校验下载器 |
| uv / FFmpeg / Git / yt-dlp / tokcount 外部工具 | 否 | 仍用 `runtime-manifest.json` 的原地址；不是本轮目标。`githubFileProxy` 只作用于 `model_fetch` 的固定模型文件，manifest 资源不走 route——tokcount 的 Release 地址同理 |
| FineSub Desktop 更新包 | 否 | 仍走签名 GitHub Release |
| 用户输入的 YouTube/Bilibili/其他媒体 URL | 否 | yt-dlp 的站点选择、代理和限速语义不变 |
| Gemini、Exa、Tavily 等 API | 否 | 不是资源下载，不受地区路线影响 |

“不部署其他源”意味着：不建设对象存储、反向代理、CDN 或 geo 服务。公共镜像只是候选
下载入口，不成为新的可信根；锁文件哈希、固定资源 SHA-256 和模型清单仍决定内容是否可用。

## 2. 总体形态

三个改动共用 bootstrap 层，但不互相阻塞：

```text
CLI 首次命令
  -> 首次目录选择（仅 CLI wheel 前端 + TTY；先记录，后下载）
  -> 下载路线解析（forced > cached IP result > live IP result > global）
  -> runtime / 模型下载环境
  -> pipeline structured reporter
       -> CLI 单行进度
       -> Desktop structured progress
       -> 完整 task log
```

目录事实仍由 `finesub_bootstrap.paths` 拥有，网络事实由新的 bootstrap 网络模块拥有，
pipeline 只接收 reporter，不读取安装器配置。`speech`、`media`、`llm` 的依赖边界不改变。

## 3. CLI 首次选择大文件位置

### 3.1 触发时机

规范触发点是首次执行会写入或下载资源的命令：

- `finesub setup`；
- 第一次 pipeline 或 batch 运行。

`--help`、`doctor`、`keys`、`uninstall` 不弹窗。直接运行 `uv tool install finesub` 不保证有
交互终端，因此不把 wheel 安装过程本身作为唯一触发点。

触发只属于 CLI wheel 前端。`Shell` 是两个前端共用的（`cli/src/finesub_cli/main.py` 与
`finesub_bootstrap.shell.package_shell`），把提示写进 `Shell.ensure_ready()` 会让桌面包自带的
命令行也弹窗，而桌面端的大文件位置本来由应用自己的界面决定。提示以回调形式由前端注入：
CLI wheel 传入，`package_shell`（`can_provision=False`）不传，`Shell` 只负责在 `ensure_store()`
之前调用它。

只有同时满足以下条件才算“新安装”：

- `locations.json` 没有 `bigData` 记录；
- 默认根下没有可采用的 `.finesub-store.json`；
- 默认根下没有已有的 `models`、`cache` 或 `tasks`；
- 默认根下没有 `runtime`。

最后一条是必要的：`runtime` 存在说明这台机器已经装好过一次，此时 `locations.json` 缺失属于记录
损坏而不是新装，不该借机改问用户。整体上，升级旧版本、修复损坏记录或第二个前端接入已有数据时
都不会突然询问。

桌面端不在本轮范围：它的大文件位置仍由安装器决定、事后用 `finesub relocate` 调整，首次启动向导
不加选盘步骤。

`setup` 因此拆成两档。今天的 `setup` 就是 `ensure_ready()`——安装 ffmpeg 并下载构建整个 Python
运行环境（数 GB、数分钟）；而 `cli/install.ps1` 今天是个秒级脚本，在它末尾接上完整 `setup` 会把
一行安装命令变成数分钟、数 GB 的过程。所以：

- `finesub setup --dirs-only`：只解析并登记大文件位置（首次时含交互选择），建目录、写 marker 和
  注册脚本，**不下载任何东西**。`install.ps1` 末尾调的是这一档，安装仍是秒级，而用户确实是在
  安装期间选的盘。
- `finesub setup`（不带参数）：语义不变，仍是完整 provisioning；它自己也走同一套首次选择逻辑。

`irm ... | iex` 这条安装路径的标准输入是否可交互必须实测（`[Console]::IsInputRedirected`），不能
假定它有 TTY；探测为不可交互时按非 TTY 规则走默认位置，不阻塞安装。

### 3.2 交互与非交互契约

建议提示：

```text
FineSub 将在这里保存模型、下载缓存和任务产物，建议预留至少 20 GB。
直接回车使用：C:\Users\<name>\AppData\Local\FineSub
也可以输入其他绝对路径，例如 D:\FineSub
大文件位置：
```

- TTY：空输入采用默认路径；自定义路径复用 `relocate` 的目标校验。**采用默认也要写进记录**——
  否则安装脚本问过一次、什么都没记，第一次真正运行又问一遍。
- 非 TTY / CI：绝不读取 stdin，直接沿用默认位置并打印一行提示。
- 自动化：增加 `finesub setup --data-dir <absolute-path>`；也接受
  `FINESUB_BIG_DATA_DIR` 作为首次初始化覆盖。`--data-dir` 与 `--dirs-only` 正交，可以同时给。
- 首次初始化的选择优先级：`--data-dir` > 环境变量 > 交互输入 > 默认路径。
- 已有有效记录时不再进入首次选择流程；若仍传 `--data-dir`，返回用法错误并提示改用
  `finesub relocate`，不暗中搬文件。

20 GB 是按模型约 3 GB、运行环境约 5 GB、跨盘时再多占约 5 GB 加上任务产物估的；15 GB 在跨盘场景
下已经贴边。

选择发生在 `Shell.ensure_ready()` 为默认路径调用 `ensure_store()` 之前。目标通过现有的绝对路径、
安装目录嵌套、个人数据目录嵌套、非空陌生目录和可写性检查后，先构造候选 `AppPaths` 并调用
`ensure_store()` 创建 marker、注册脚本和原子位置记录，再开始任何下载。随后重新
`load_app_paths()`：若并发的另一个首次进程抢先登记了有效位置，本进程沿用锁内最终记录，不继续
使用已经落败的候选路径。

当前 `runtime` 永远留在安装根。若自定义大文件目录与 runtime 跨盘，uv cache 无法与环境硬链接，
提示必须继续说明可能多占约 5 GB，并引导真正想释放系统盘的 CLI 用户在安装前设置
`FINESUB_HOME`。这段跨盘提示与 `Shell.relocate` 已有的同一段警告共用一处文案，不要复制第二份。
本轮不扩展为搬迁 runtime；那是独立的 `relocate --all` 设计。

## 4. Pipeline 日志方案

### 4.1 现状判断

默认输出混合了四种受众：用户进度、输出质量警告、性能诊断和算法逐步决策，信息密度不合理。
最明显的噪声是：

- audio-separator 每处理一个外部分块都会创建自己的 tqdm；并发分块会交错输出多条进度条。
- ASR 每个 group 都打印一次 `group ASR`，虽然消息带 `progress_pentile`，并没有只在 pentile
  变化时输出。
- temporary recall、短 group 语言复用、beam/coverage rescue 的每一步都直接写 stderr。
- 每个阶段分别打印 `Wrote`、多行 `Timing` 和多行 `Resource usage`，pipeline 末尾又打印完成
  路径与总耗时。
- Desktop 将 stdout/stderr 转成有上限的 UI 事件；噪声可能挤掉更早的关键上下文，不能把这个
  有限事件缓冲区继续当作完整任务日志的来源。
- `EventLogWriter` 只按 `\n` 切分，`\r` 不产生事件。所以第三方 tqdm 今天的表现不是“成百上千条
  日志”，而是把整条进度条累进一个无上限的缓冲区，直到该 block 结束才吐出一条几 MB 的日志行——
  内存增长，外加一条巨行落进 `task-log.txt`。这使“禁用第三方 tqdm”成为本节优先级最高的一条。

### 4.2 统一 reporter

在 `finesub` 增加轻量、无 UI 依赖的 reporter 契约：

```text
planned(stages)
stage_started(stage, reused, detail)
progress(stage, completed, total, unit, detail)
summary(stage, metrics)
warning(code, message, impact, action)
debug(message, fields)
completed(output, elapsed_sec)
failed(stage, message)
```

`planned` 是 `[n/N]` 的来源：分母由本次运行自己的计划决定，renderer 不去猜。`failed` 也是必需
的——`warning` 描述“结果受影响但还在跑”，运行终止今天只靠异常加 traceback，renderer 无从把它
渲染成阶段列表里的一格。异常本身仍照常向上抛，`failed` 只负责上报。

阶段中文名以桌面端 `desktop/frontend/lib/translations.ts` 的 `stages` 为准（人声分离／语音识别／
字幕稳定化／原始字幕／纠错翻译／最终字幕），CLI renderer 复用同一套：同一次运行被两个前端叫成
两个名字是支持成本，不是风格问题。

`summary` 收的是带标签的数字而不是一句话：只有 stage 知道那 7 段叫“噪声片段”，只有 renderer 知道
这次运行有没有终端可以重画。零值和 `None` 由 renderer 丢弃，stage 照常把计数报全。

`run_pipeline()` 从**调用方的绑定**里取 reporter（`reporting_to(...)`），不另加参数：调用树深处的
stage 本来就是这样读的，同一件事两套机制迟早会分叉。CLI 和 Desktop 各自提供 renderer；没人绑定
就什么都不显示。`on_stage` 直接删掉，不留迁移期适配器：生产调用点只有
`desktop/backend/worker/main.py` 一处，其余都是测试，而本仓库不保留兼容层。

batch 是唯一需要行前缀的场景：多个任务共用一个终端，可原地重画的进度行会被最后说话的那个任务
覆盖掉，两边都读不成。所以 batch 给每个任务一个带 `[<source>] ` 前缀、且强制行模式
（`isatty=False`）的 renderer。

日志级别：

| 级别 | 默认显示内容 |
| --- | --- |
| `quiet` | 影响运行的 warning/error、最终路径 |
| `normal`（默认） | 阶段、节流进度、阶段摘要、warning/error、最终路径与总耗时 |
| `verbose` | 再加 timing、正常资源占用、worker/backend、逐次 rescue 和语言选择 |

完整 debug 信息始终可以写任务日志，不要求用户为了保留诊断信息而使用 verbose UI。

### 4.3 各阶段调整

**Separator**

- 第三方降噪**不在分离器适配层，而是整次运行**（`reporting.quieted_libraries`，由前端在
  `main()` 里进入）。设计稿原先只说 tqdm，实测证明这判断不完整：3.1 分钟素材的 45 行默认输出里
  **25 行来自 audio-separator 的 logging**（版本横幅、主机 OS 与 CPU 型号、它打开的每个文件），
  而 tqdm 那一路除了分离器还有 transformers 在校验时画的权重加载条。
- 两条通道都要管，但方式不同：**logging 抬到 WARNING 而不是静音**（分离器的
  `CUDAExecutionProvider not available` 要留），**tqdm 直接禁用**（进度条写进日志文件永远只是
  噪声，何况 pipeline 自己会画）。`verbose` 两者都不动。
- logger 名按**模块**而非包命名（`separator` / `common_separator` / `mdxc_separator`，不是
  `audio_separator`）；而且 `Separator.__init__` 有自己的 `log_level` 参数，会覆盖构造前设好的
  级别，所以必须在构造时直接告诉它。这两点是实跑才发现的——单测用了和代码同一个错误猜测。
- 默认只维护一条 FineSub 进度：`完成 block / 总 block`、并发数、已用时间。
- 并发 future 完成时更新计数，不等按输出顺序 merge 后才更新。
- block 内没有可用的稳定 callback 时，不伪造平滑百分比；长 block 期间用同一行 heartbeat
  更新已用时间，而不是不断新增行。

**VAD / ASR**

- group 进度只在越过下一个 5% 档位时上报；恢复 checkpoint 时立即上报当前起点。节流放在**调用点**
  而不是只靠 renderer：renderer 只管终端，而桌面端会把每次上报变成一个事件，事件量的上界必须在
  源头就有。计数单位是 interval 而不是 group——group 总数要等分组跑完才知道，分母不能是它。
- temporary recall、short-language reuse、rescue ladder 步骤进入 verbose/debug。
- normal 在阶段结束输出汇总计数：总 groups、temporary recall 次数、beam rescue 尝试/接受、
  隔离异常 interval 数、真正丢弃的 groups。
- 会改变质量且用户可能需要处理的事件仍立即 warning，例如 group 最终丢弃、CPU 回退、模型
  验证被禁用；正常且已恢复的内部尝试不逐条警告。
- 每个 `main()` 都要绑定 renderer。线程局部的默认 reporter 是静默的——这对库是对的，但
  `speech/recognition/transcribe.py`、`vad_asr_stage.py` 这些模块同时是独立开发 CLI，不绑定就会把它们
  自己的 CPU 回退告警吞掉。`reporting.terminal_reporter()` 是这个绑定的单一入口
  （输出到 stderr，级别可用 `FINESUB_LOG_LEVEL` 覆盖）。

**稳定化 / SRT / LLM**

- 稳定化保留一行摘要，只展示非零的 removed/dropped/tag 计数。
- 中间产物不再逐个 `Wrote`；reused 也由 stage 行表达。最终交付物只打印一次。
- timing 和低于预算的资源占用写 metadata + verbose；接近预算或越界才在 normal warning。
  **四个阶段都要计时**，走 `pipeline._record_stage_time` 一个入口：verbose 承诺了 timing，而
  只有部分阶段报会让人靠总耗时做减法。复用的阶段记为「无耗时」而不是 0——「没跑」和「瞬间跑完」
  是两回事，metadata 会被工具读去做统计。
- LLM normal 显示窗口完成数、重试和最终摘要；prompt、搜索轮和 exchange 详情留在 artifacts。

**Desktop 完整日志**

- worker/manager 收到 log event 时同步追加 `task-log.txt.part`，任务结束后原子改名；失败和被杀
  也尽量保留 part。
- UI 事件仍可设上限并合并同一阶段的 progress，但文件日志不经过该上限。
- verbose 细节走单独的 `debug` 事件：只落文件、不进 UI 事件队列。诊断一次桌面端运行不该要求先在
  CLI 上复现，但逐 group 的救援细节会把抽屉那几百条预算一次花光。
- carriage-return 进度只更新一个 structured progress event。writer 的行缓冲必须有上限：遇 `\r`
  就地覆盖或截断，不允许把一整段进度条攒成一行，也不允许缓冲区无界增长。

### 4.4b 落盘的 verbose 日志（2026-08-17 实施）

终端之外，每次运行还往 `<user-data>/logs/run-<时间戳>-<输入名>.log` 写一份**恒定 verbose**
的记录，与 `--log-level` 无关：终端保持好读，决策细节留在盘上，用户报障时直接把文件发来即可。

三个刻意的选择：

- **一次运行一个文件，不是共享一个文件。** 桌面 worker 与 CLI 可以同时在跑
  （`cross-frontend-lease.md`），分文件既不用加锁也不会把两次运行交错在一起。
- **不放在 `out/<stem>/`。** 日志必须在输出目录确定之前就开始记（参数解析、设备解析、下载
  路线判定），而且**启动就失败的运行根本没有输出目录**——那恰恰是最需要日志的一次。
- **`quieted_libraries` 仍然收终端档位。** `verbose` 捆绑了两件事：我们自己的 debug 行，以及
  放行第三方 logging 与 tqdm。文件要前者，绝不能要后者——tqdm 靠 `
` 高频重画，倒进文件就是
  几十 MB 的噪声。这是本节唯一容易写错的地方，有守卫测试
  （`test_the_log_file_does_not_drag_tqdm_and_library_logging_along`）。

**已知覆盖缺口：LLM 段几乎为空。** 3.1 分钟素材实跑（2026-08-17）：纯 ASR 4.4 KB，内容正是
救援阶梯与逐阶段计时；而带纠错翻译的两次（API+local 7m55s、agy+native 2m10s）**LLM 那一段只有
进出两行**。根因是 `src/finesub/llm/` 里 `current_reporter()` **零处**、裸 `print` **88 处**，
整层绕过 reporter。细节没丢（在 `task-report.md`／`correction-windows.jsonl`／`exchanges/`），
但用户发来的那个文件恰恰缺了最需要的部分——重试、配额、校验失败全在 LLM 段。补齐＝把那 88 处
接进 reporter，即 [`refactor-followups.md`](refactor-followups.md) 判定押后的那一项。

实现是 `reporting.FileReporter`（恒定 verbose、按十分位节流 progress、单锁串行化多线程写入）
与 `reporting.FanOutReporter`（终端 + 文件，任一渲染器抛异常不影响运行）。清理沿用
`finesub_bootstrap.logs.prune` 的「保留最新 100 个」，与桌面的 install/session 日志共用预算。

### 4.4 默认输出

3.1 分钟素材的**实测输出**（重定向到文件，即行模式；TTY 上同一条进度原地刷新）：

```text
[1/4] 人声分离
[1/4] 人声分离     0% · 0/1 blocks · 00:00
[1/4] 人声分离     100% · 1/1 blocks · 00:17
[2/4] 语音识别
[2/4] 语音识别     0% · 0/35 intervals
[2/4] 语音识别     37% · 13/35 intervals · 1 groups
[2/4] 语音识别     100% · 35/35 intervals · 5 groups
      语音识别摘要：区间 35，异常隔离 3
[3/4] 字幕稳定化
      字幕稳定化摘要：段 79 -> 79，标记 23
[4/4] 原始字幕

完成：...\demo-raw.srt
总耗时：1m 15s
```

同一次运行改造前是 45 行、其中 25 行是 audio-separator 的 INFO；现在是 20 行，其中 4 行第三方
且全部是真警告。

同一条进度在 TTY 上原地刷新；非 TTY/文件重定向时只在跨 10% 或阶段变化时新增一行，保证 CI
日志可读。`[n/N]` 的分母按本次运行实际计划的阶段数算——`--stage` 与已存在的上游产物都会改变
它，不是固定的 4。

## 5. 依赖包与模型的大陆下载路线

### 5.1 路线解析

新增 `finesub_bootstrap.download_routes`，解析结果只有 `cn` 或 `global`，并记录来源：

```text
FINESUB_DOWNLOAD_REGION=cn|global|auto
  > 近期缓存的 auto 结果
  > 公共 IP country endpoint
  > global（超时、离线、响应非法）
```

- 默认 `auto`；显式环境变量永远优先，便于 VPN、公司代理和故障排查。
- 不部署自有 geo 服务。选择一个主、一个备的公共 country endpoint，单个连接超时不超过
  1.5 秒，总预算不超过 3 秒。
- 请求走 `network_routes()` 的首选路由，所以地区代表下载出口而不是本机物理位置。这里有一个
  已知近似：`network_routes()` 返回的是“代理路由 + 直连兜底”的列表，`downloader` 逐条尝试，
  探测走了代理而实际下载落到直连是可能的。判定以首选路由为准；某次下载实际走了另一条路由时，
  不拿它的成败去更新缓存结论。
- 只缓存 `region`、判定时间和 endpoint 名称，不保存 IP；TTL 24 小时。
- cache 放在 `%LOCALAPPDATA%\FineSub` 的小状态文件，而不是可能尚未选择的 big-data root。
- `finesub doctor` 显示 `download   cn (自动检测，缓存)  pypi=…  huggingface=…  github=…`，
  每类要么是已配置的入口、要么是「官方源」、要么是「已停用（连续失败）」；不打印 IP。
- `download-sources.json` 随包发布，**已填入经可用性实测的候选**（见 §5.2 的表）。非大陆出口的
  机器不受影响：`auto` 解析为 `global`，一个都不会用到。某个源真的坏掉时，按类的连续失败计数会
  在本机把它停用并回落官方源。`FINESUB_DOWNLOAD_SOURCES` 可指向另一张表，供验收演练或分支自定义。

公共 endpoint 和镜像地址放在随版本发布的 `download-sources.json`，不散落在 Python 代码里。
环境变量提供紧急覆盖：

```text
FINESUB_PYPI_INDEX
FINESUB_HF_ENDPOINT
FINESUB_GITHUB_FILE_PROXY
```

空值表示禁用该类 cn 加速并回到官方源。第一版不增加 Desktop 设置项。

`download-sources.json` 随版本发布，意味着某个公共入口挂掉要等下一次发版才能换，而环境变量覆盖
只对读文档的人有效。所以每类资源自带本机降级：连续失败达到阈值后，把“本机禁用该类 cn 加速”写进
与 region 缓存同一个状态文件（同一套 TTL 口径），后续运行直接走官方源。

### 5.2 依赖包与 managed runtime

先看字节分布，它决定本节的优先级。当前 `desktop/runtime/pylock.win-py312.toml` 里：

| 来源 | 体量 |
| --- | --- |
| `files.pythonhosted.org` 的全部 wheel | 约 257 MB |
| `download-r2.pytorch.org` 的 torch / torchaudio / torchvision cu128 | 约 2.5–3 GB（lock 里没有 size 字段） |
| 自建 GitHub Release 上的 patched CT2 | 数十 MB |

也就是说普通 PyPI wheel 占依赖下载不到十分之一。**torch 三件套的镜像可行性是本节的前置判定**：
它恰好最适合“只改 artifact URL、包名/版本/文件名/SHA-256 逐项一致”这套约束——多个高校镜像镜的
就是 `download.pytorch.org/whl/cu128` 的同名文件，可以逐项比对哈希。torch 通不过发布验收时，
只为 257 MB 维护第二份 lock 加一道 CI 门禁并不划算，整份 cn 依赖 lock 应当降级为可选，而不是
照做一遍拿到一个用户无感的加速。

patched CT2 与 §5.4 的 BS-Roformer 是同一类东西：固定 URL、已知 SHA-256、托管在 GitHub。两者
共用 `FINESUB_GITHUB_FILE_PROXY` 一个机制，不要一处写“保留原地址”、另一处单独设计代理。

cn 路线的候选 PyPI index 优先采用有公开运维主体和 HTTPS 的高校/云厂商镜像。发布前必须用实际
Windows runtime lock 做完整安装演练，候选未通过就不启用。

**候选可用性实测（2026-08-10，出口在境外，因此只验存在与内容、不测速）：**

| 候选 | 结果 |
| --- | --- |
| SJTU `pytorch-wheels/cu128` | **三个 torch wheel 全部存在**，长度与官方源一致，首尾各 1 MB 逐字节一致 |
| TUNA `pytorch-wheels/cu128` | 404（该布局下没有这个路径，需另找或放弃） |
| SJTU / TUNA / BFSU / Aliyun PyPI | 均 200 |
| `hf-mirror.com` | 200，能按钉住的 revision 取到文件 |
| `ghproxy.net` / `gh-proxy.com` | 200，分离器 yaml **摘要与官方源一致** |

torch 是 §5.2 定的硬门槛，SJTU 通过意味着 cn lock 值得做（2.75 GB 的 torch 本体在镜像上存在
且内容一致）。**首尾比对不等于全量校验**：完整 sha256 要在发布验收里对着 lock 里已有的摘要跑
一次全量下载。TUNA 的 PyPI 可用但 pytorch-wheels 路径不可用，因此这两类要分开选源。

分两条路径：

1. `cli/install.ps1` 只认显式的 `FINESUB_PYPI_INDEX`。这一步跑在 FineSub 装好**之前**，自动地区
   判定就在它即将安装的那份 Python 里，此处用不上；在 PowerShell 里重写一份判定，等于为一个还
   没有合格候选的入口维护第二套实现。从第一条 `finesub` 命令起，路线正常解析。
2. `RuntimeEnvironment.install()` 根据路线选择 global 或 cn runtime lock。**marker 始终哈希
   canonical lock**：两份 lock 只差在谁发货、文件与哈希完全相同，若按实际安装用的那份计算，
   跨地区就会被判成「依赖变了」而重建一个本来就正确的 5 GB 环境。

不能只设置 `UV_DEFAULT_INDEX`：当前 `pylock.win-py312.toml` 已锁定 wheel 的绝对 URL，uv 会按
这些 URL 下载。构建时从 canonical global lock 生成 `pylock.win-py312.cn.toml`：

- 只改 artifact URL，不重新解析版本；
- 包名、版本、marker、文件名和 SHA-256 必须逐项与 global lock 一致；
- 普通 `files.pythonhosted.org` URL 改成已验证镜像的等价文件地址；
- Torch 三件套优先于普通 PyPI wheel 处理；patched CT2 走与 §5.4 相同的 GitHub file proxy。
  两者都只有在存在通过验证的公共加速入口时才改，否则保留原地址——允许“部分加速”而不是降低
  可重复性，但“只有普通 wheel 被加速”这种组合按上面的前置判定不算达标；
- CI 比较两个 lock 的语义清单和哈希，任何依赖漂移都失败。生成器
  `desktop/scripts/make_cn_lock.py` 自带这道门：生成后立刻自检，不等价就删掉产物并报错，
  不留下一份「看起来生成成功了」的 lock。测试对**入库的那两份文件**再跑一次同样的比对，所以
  重新生成后漂移了也走不到用户机器上。
- **`pylock.win-py312.cn.toml` 已生成入库**（2026-08-10）：170 个 PyPI wheel 改写到 TUNA、
  3 个 torch wheel 改写到 SJTU、patched CT2 保留自建 GitHub release（没有合格镜像，属于设计允许
  的「部分加速」）。抽查 12 个小 wheel **摘要与 lock 完全一致**、17 个大 wheel 长度一致，零失败。
  两个打包脚本都要带上它——CLI wheel 是逐文件 vendoring 的，漏了它 CN 机器永远找不到、整条路径
  变成 wheel 里的死重量。

managed Python 本体只有在找到兼容 uv 路径契约、可验证且稳定的公共镜像后才设置
`UV_PYTHON_INSTALL_MIRROR`。没有合格候选时继续官方源，不把 runtime 安装和依赖 wheel 加速
绑在一起。

安装失败策略：cn URL 的连接错误、404、408、429 或 5xx 可以整次重试 global lock；磁盘错误和
解压错误不伪装成网络故障。切换 lock 前保留 uv 自己可安全复用的 cache。

**哈希不符要按「谁发的货」分方向**（这条是实施时修正的，原稿把它和磁盘错误归为一类）：cn lock
保留 canonical 的 sha256，所以 uv 会校验；镜像发错字节时哈希不符，而这恰恰是官方源能救的情况
——镜像陈旧或损坏、官方源是好的。把它判为不可重试等于把唯一的补救路径堵死，让一个过期镜像变成
硬失败。所以：**从 cn lock 安装时哈希不符 → 回退 canonical 并降级该镜像；从 canonical 安装时
哈希不符 → 直接失败**（那时它确实说明文件本身有问题，换一台主机也没用）。

### 5.3 Hugging Face 模型

faster-whisper 与 Qwen referee 共用 `HF_ENDPOINT`。它必须设在
`RuntimeEnvironment.worker_context()`——那里已经在设 `HF_HOME`/`TORCH_HOME`/`FINESUB_MODEL_DIR`，
而且这是两个前端唯一的公共通路：CLI 根本没有预取阶段，模型是 pipeline 跑起来之后惰性下载的，
只在 Desktop 的 prefetch 里设置等于漏掉一半用户。取值：

- global：不设置，使用 `https://huggingface.co`；
- cn：设置为 `download-sources.json` 中经演练的公共 Hugging Face mirror；
- 用户已经显式设置 `HF_ENDPOINT` 时不覆盖。

为避免公共镜像把可变的 `main` 解析成不同内容，三个生产模型都固定 revision。发布时生成
`model-manifest.json`，记录 repo、revision、必需相对路径、大小和 SHA-256。与 `download-sources.json`
一样，**随包发布的那份是空的**：真实大小与摘要要等发布验收才有，而编造的哈希比没有哈希更糟——
一个永远过不了的校验最后会被关掉，而不是被修好。未登记的模型走今天的老路（由拥有它的库自己
下载，不声称有额外校验）。通过公共镜像新下载
的 snapshot 在标记 ready 前校验一次，并写入带 manifest hash 的 verified marker；已有的官方
HF cache 仍按当前复用规则读取，不强制重下。

HF mirror 不可用时的回退不能只在同一进程里晚改环境变量，因为 huggingface_hub 可能已在
import 时缓存 endpoint。预取按 endpoint 启动独立子进程，且**粒度是每个模型一个子进程**：现在
三个模型在同一个进程里顺序下载，整批重试会让已经拿到的 whisper 陪着失败的 qwen 重下 1.6 GB。
cn 子进程失败且错误可重试时，只为这一个模型再以 global 环境启动一次。

pipeline 的惰性下载复用同一个 helper，然后用 `local_files_only`/已缓存目录加载，避免模型 loader
各自实现一套回退。**这个 helper 不能留在 `desktop/backend/worker/prefetch.py`**：`finesub`
不允许 import `desktop`。它落在 `finesub_bootstrap`，Desktop 的 prefetch 入口和 pipeline 都从那里
调用，前者只保留自己的进度上报。

verified marker 与共享缓存的关系要一并定清楚：`existing_hf_home()` 会在机器已有官方 HF 缓存时
整体切到 `~/.cache/huggingface`，此时“这个 snapshot 是不是经镜像下的”没有任何记录。至少要写明
marker 落在哪个缓存根、与已有官方缓存如何共存，以及断点续传的 snapshot 必须在打 marker 前完成
一次完整校验。

### 5.3a 两个已知缺口（评审记录，未实施）

分离器已经两端统一（`separation.place_separator_files()` 在建 Separator 前调用，桌面 prefetch
复用同一入口），HF 这一半还差两步：

1. **CLI 没有按模型的镜像回退。** `worker_context()` 会注入 `HF_ENDPOINT`，但
   `fetch_with_fallback()` 的唯一生产调用方仍是桌面 prefetch。CLI 在 pipeline 跑起来之后惰性
   下载，镜像临时故障时不会切官方源、也不会累计失败次数。要补需要一个两端共用的「确保模型就位」
   步骤（每模型一子进程 + 回退），挂在 `Shell.ensure_ready()` 与桌面资源面板同一层——这会把
   CLI 的首次体验从「跑起来再下」变成「先下再跑」，是行为变更而不是补丁。
2. **Whisper/Qwen 没有按 manifest 校验。** `missing_pipeline_models()` 只看 snapshot 目录存在、
   无 `.incomplete`、且非空，注释里明确接受「两个文件之间中断」这个窗口。manifest 里的大小与
   SHA-256 目前只用于分离器。要补必须同时做 verified marker：下载后校验一次、写 marker，之后
   只查 marker——每次状态轮询都去哈希 3 GB 不可行，而 UI 会频繁轮询。

### 5.4 BS-Roformer 固定模型

audio-separator 当前自己从 GitHub 下载 checkpoint，既不受 `HF_ENDPOINT` 控制，也无法复用
FineSub 的地区回退。把“下载”和“加载”拆开：

- 在 `model-manifest.json` 登记 `SEPARATOR_CHECKPOINT` **及其配套模型配置文件**的官方 URL、大小
  和 SHA-256——`_build_separator` 读的 `model_data_cfgdict` 来自随 ckpt 一起下载的配置，只登记
  ckpt 一个文件不足以让“加载不再联网”成立；
- cn 路线可给这个**固定文件**配置公共 GitHub file proxy URL；
- 用 `finesub_bootstrap.downloader` 下载、断点续传和校验后放入现有 separator model dir；
- `Separator.load_model()` 只加载已经存在且校验过的文件，不再负责首次联网；
- proxy 失败回退官方 URL；哈希不符时隔离 `.bad`，重新从官方源完整下载，不拼接两个来源的
  未验证字节。

**前提已实测成立，但需要的是三个文件而不是一个**（2026-08-10，audio-separator 0.44.3；
把 `requests.get` 换成一调用就抛异常，再走一次 `load_model()`）：

| 文件 | 大小 | 来源 | 可否钉哈希 |
| --- | --- | --- | --- |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 639 MB | GitHub release（不可变） | 可 |
| `model_bs_roformer_ep_317_sdr_12.9755.yaml` | 2.3 KB | 同上，模型配置 | 可 |
| `download_checks.json` | 28 KB | **`raw.githubusercontent.com` 的 `main` 分支** | **不可** |

三者齐备时 `load_model()` **零网络调用**，所以「先放好、再加载」成立。两点必须记住：

1. `download_checks.json` 是 `list_supported_model_files()` 取的**上游模型索引**，走
   `download_file_if_not_exists`——本地有就不下载。它是这条链路上唯一真正来自
   `raw.githubusercontent.com` 的文件，也就是大陆最难连的那个；预取必须把它一起放好，否则
   ckpt 和 yaml 都在也照样卡住。
2. 它在上游是**可变的**（`main` 分支的文件列表），所以 manifest **不能钉它的 SHA-256**——钉了
   会在上游更新当天全体校验失败。ckpt 与 yaml 是 release 资产、不可变，钉哈希是对的。这个文件
   按「取一次、不校验」处理，cn 路线下同样可以走 GitHub file proxy。

另外 `run_vocal_separation` 会并发创建多个 Separator、每个都调用一次 `load_model()`；由于文件
已在本地，这只是重复的本地读取，不会放大成并发网络请求。

public GitHub proxy 只有在连续可用性演练和 HTTPS 检查通过后才进入默认配置。没有合格候选时，
该模型仍走官方源；不为了做到“所有模型都加速”而引入不可审计的 URL 重写站。

### 5.5 可观测性与隐私

normal 日志每类下载只输出一次选择结果，例如：

```text
下载路线：中国大陆（自动检测，缓存 24h）
Python 依赖：TUNA PyPI；Torch / patched CT2：官方源
模型：HF mirror；BS-Roformer：官方源
```

切换和回退要说明资源类别与原因，但不打印完整代理凭据、IP、带 token URL 或 HF token。
`doctor` 只做轻量配置/缓存报告，不通过下载大文件验证镜像。

### 5.6 外部工具的钉法：钉不住的东西怎么还能校验（2026-08-18 实施）

`runtime-manifest.json` 的五个资产默认按 `url` + `size` + `sha256` 钉死，`downloader` 先查
大小再查摘要。**ffmpeg 是唯一的例外**，它带 `digest_from: "github-release-api"` 而不带
`size`/`sha256`。

原因和 §5.4 里 `download_checks.json` 那条是同一个：上游的字节会动。BtbN/FFmpeg-Builds 的
`autobuild-<日期>` tag 只保留几天的滚动窗口（2026-08-18 实测：manifest 里钉的
2026-07-24 那个已 404，仓库里最老的只剩 08-14），而 `latest` tag 下的资产**每次构建都被删了
重传**——包括 `n9.0` / `n8.1` 这两个发布分支的（三个 win64-lgpl 资产的 `created_at` 都是当天）。
所以这个位置上「钉一个哈希」不是选项。

但 §5.4 当年只能得出「取一次、不校验」，这里可以做得更好：**GitHub 的 release API 会给出每个
资产的 `digest`**，于是可以把「哪个版本」和「哪些字节」分开——

- 装机时先问 `/repos/<owner>/<repo>/releases/tags/<tag>`，拿到当前资产的 `size` 与
  `sha256`（`asset_resolve.resolve_asset`）；
- 把它变成一个普通的 pinned `DownloadAsset` 再交给 `downloader`，下游的续传边界、进度总量、
  大小与摘要校验全都不需要知道「有会动的资产」这回事；
- 答案走 API 自己的 TLS 连接，不是文件传输那条。要伪造得同时改掉 API 响应——而 runtime 资产
  本来也不走大陆文件镜像（只有 `model_fetch` 走）。

**换来的与放弃的**：完整性和续传安全都保住了，放弃的是**可复现性**——隔一天装的两台机器会拿到
不同的 ffmpeg 构建，而且都不是我们测过的那个。这笔交易只对「接口面极小且稳定」的工具成立：
管线对 ffmpeg 只用 `-i` / `-ss` / `-t` 和 `ffprobe -show_entries`。凡是我们依赖其细节行为的
东西一律继续钉死，`test_runtime_manifest_pins_every_asset_it_can` 把「例外只有 ffmpeg」钉成
红线——再加一个名字进去，得先让那条测试变红。

三个连带的点：

1. **`version` 是静态标签而不是构建号**（`n9.0-latest`）。`status()` 比的是
   installed == spec.version，所以装过一次就不会每天重下；要把所有人推到新的 ffmpeg，改这个
   标签，用户看到的是 `outdated`（提示升级，不阻塞）。
2. **续传要能识别「目标变了」**。`.part` 旁边多一个 `.part.expect` 记着这份部分下载是冲着哪个
   摘要去的；不一致（pin 被 bump，或上游重编了）就直接丢掉重来，而不是把新构建的尾巴接到旧
   构建的头上——那种文件只能在下载完之后校验失败。没有 `.expect` 的 `.part`（旧版本留下的）
   同样丢弃：`.part` 是缓存产物，丢它只损失已经花掉的字节，信它可能白花一整次。
3. **`download_checks.json` 仍然不能用这套**：它是 `raw.githubusercontent.com` 上 `main` 分支的
   文件，不是 release 资产，没有 API 会告诉你它此刻的摘要。§5.4 的「取一次、不校验」对它依然
   是唯一可行的处理。

失败面比钉死的多一处：`api.github.com` 得能连上（未认证按 IP 每小时 60 次；一次装机花一次，
所以只有共用出口才可能撞到，撞到时报的是「稍后重试」而不是下载损坏）。这台机器本来就要从
`github.com` 下这个文件，所以不算引入新的可达性依赖。

## 6. 预计改动位置

| 区域 | 主要改动 |
| --- | --- |
| `src/finesub_bootstrap/paths.py` | 首次位置状态读取；继续复用原子记录和锁 |
| `src/finesub_bootstrap/shell.py` | 首次 TTY 交互、`setup --data-dir`、route 摘要与 doctor |
| `cli/src/finesub_cli/main.py`、`cli/install.ps1` | 首次 setup 接线、CLI wheel 安装的 PyPI route |
| `src/finesub_bootstrap/download_routes.py`（新） | IP 判定、TTL cache、override、公共源注册表 |
| `src/finesub_bootstrap/environment.py` | cn/global lock、uv/Python 下载环境、整次 fallback；`worker_context()` 注入 `HF_ENDPOINT` |
| `src/finesub_bootstrap/model_fetch.py`（新） | endpoint 选择、按模型的 route/fallback、固定文件的取回与校验；两个前端共用 |
| `src/finesub_bootstrap/model_manifest.py`（新） | 模型清单：repo/revision/文件大小与 SHA-256，含「无摘要=取一次不校验」 |
| `desktop/backend/worker/prefetch.py` | 每模型一子进程、钉 revision；分离器文件的放置交给分离阶段 |
| `src/finesub/speech/preprocessing/separator/separation.py` | `place_separator_files()`（两端共用的固定模型放置）、block progress |
| `src/finesub/reporting.py`（新） | pipeline 事件契约、CLI renderer、第三方 logging/tqdm 降噪（`quieted_libraries`，整次运行而非单阶段） |
| `src/finesub/pipeline.py` 与各 stage | 直接 print 迁移为 stage/progress/summary/warning |
| `src/finesub/batch.py` | 多任务交错输出改走 reporter |
| `desktop/backend/worker`、`jobs/manager.py` | Desktop renderer、progress 合并、完整日志流式落盘 |
| `desktop/backend/worker/protocol.py` | `EventLogWriter` 的 `\r` 处理与行缓冲上限 |
| `desktop/runtime/`、`desktop/resources/` | cn lock、download/model source manifests |

实现时同步更新 `README.md`、`cli/README.md`、`manual/resources.md` 和
`desktop/README_DEV.md`；本设计稿不提前把未实施行为写进用户手册。

## 7. 测试与验收

### 7.1 首次目录

- TTY 空输入、自定义路径、非法路径、不可写路径。
- 非 TTY 永不等待输入；`--data-dir` 可自动化。
- `setup --dirs-only` 登记完位置就返回，不触发 ffmpeg 或 runtime 下载；不带参数的 `setup`
  行为不变。
- 已有 install/store/locations 不再询问；`runtime` 已存在但 `locations.json` 缺失（记录损坏的
  老安装）同样不询问。
- 桌面包自带的命令行在 TTY 下也不弹提示。
- 两个首次进程竞争时只采用一处位置，且下载开始前已经登记。
- 跨盘警告与 `FINESUB_HOME` 指引存在。

### 7.2 日志

- fake separator 验证默认输出没有第三方 tqdm，多 worker 只更新一条 progress。
- ASR 100 个 group 的 normal 进度不超过 21 次，verbose 仍保留逐 group 诊断。
- 正常 stage 不重复打印中间 `Wrote`、Timing 和 Resource usage。
- 质量相关 warning 不因 quiet/normal 重构丢失。
- Desktop UI event 超限不会截断磁盘 task log；失败 traceback 在文件中完整存在。
- writer 行缓冲有上限：持续 `\r` 输出既不攒成一条巨行，也不让缓冲区无界增长。
- golden logs 覆盖 clean、resume、异常 rescue、batch 交错和非 TTY 五种场景。

### 7.3 下载路线

- forced/cached/live/failure 四种地区判定；代理出口按代理结果判定。
- geo 超时总预算受控且回到 global；不记录 IP。
- cn dependency lock 与 global lock 的包/version/hash/marker 完全相同。
- cn 依赖失败可整次回退 global；哈希失败不会静默接受或错误续传。
- HF 子进程收到正确 endpoint；cn 失败后 global 重试；显式 `HF_ENDPOINT` 不被覆盖。
- 三个模型只有在 snapshot/model manifest 校验通过后才报告 ready。
- 连续失败达到阈值后本机禁用该类 cn 加速，并在状态文件里留下依据。
- ffmpeg/git/yt-dlp/tokcount、Desktop update、yt-dlp 媒体下载和 API 请求不受 route 影响。
  其中只有 tokcount 拉不到也不影响任务（退回 `countTokens` 端点），其余仍是硬依赖。

默认单测使用 fake HTTP、fake subprocess 和小文件，不访问公共镜像、不加载模型。发布前另跑一次
Windows 实机验收：全新目录分别强制 `cn` 和 `global`，安装 runtime、预取三个模型、断网恢复、
中途切源和哈希破坏各一次；记录结果后才能更新默认 `download-sources.json`。

这次实机验收必须**同时记录两条路线的实测耗时**——只验证“镜像能下下来”不足以判断它值不值得当
默认：公共镜像限速或拥塞时可能比官方源更慢，而验收里没有任何判据能发现这件事。cn 明显不快于
global 的资源类别不进默认配置。

## 8. 实施顺序

1. **P1 日志地基**：reporter、separator tqdm 静音、ASR 节流、Desktop 完整日志。
2. **P2 首次目录**：TTY/非 TTY 流程、`setup --data-dir`、`setup --dirs-only`、安装脚本接线。
3. **P3 路线路由**：IP 判定、cache、override、doctor，不改变任何下载源。
4. **P4 模型加速**：固定 revisions、model manifest、HF route、separator 校验预取。
5. **P5 依赖加速**：先验证 torch 三件套的镜像候选，再验证公共 PyPI mirror；两者都过才生成
   cn lock 并接上 global fallback。torch 不过则本步降级为可选——不为 257 MB 引入第二份 lock
   加一道 CI 门禁。

模型排在依赖前面是按字节量决定的（见 §5.2 的体量表）：模型约 3 GB，且不需要第二份 lock 和 CI
门禁，做完就有用户可感知的收益；依赖那边工程量最大的部分恰好服务于最小的 257 MB，值不值得做
要等 torch 的镜像验收结果。

P3 与后两步分开是刻意的：地区判断正确不代表镜像可用。某一类资源没有通过发布验收时，只让该类
继续走官方源，其他已验证类别仍可加速；不设置“一开全开”的总闸门。
