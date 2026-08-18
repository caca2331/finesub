# FineSub Desktop：开发与发布

面向维护者。用户向的说明（怎么装、数据在哪）在 [README.md](README.md)。

Desktop 使用独立于 Python 包版本的发布版本；唯一版本源是 `desktop/VERSION`。
构建脚本、CI、launcher、前端 package 和 Windows 版本资源必须与该文件保持一致。

## 架构

```text
Next.js static UI
        │ pywebview API
DesktopBridge
        ├── JobManager ── isolated worker ── existing FineSub pipeline
        ├── ResourceManager ── uv / Python 3.12 / FFmpeg
        ├── SettingsStore ── local API keys
        └── signed GitHub Release check + in-app install
```

`backend/jobs/` 四个模块，按「一次任务的哪一段」分：`manager.py` 是 `JobManager` 本体
（锁、当前快照、reader 线程、历史的读写）；`launch.py` 是进程存在之前就定下来的事
（`WorkerLaunchContext`、任务命名、显卡可见性、`spawn_worker`）；`history.py` 是记录本身
与共享索引的判据（`JobSnapshot` 及其纯函数）；`task_log.py` 是 `task-log.txt` 的落盘。

**样式表按分节拆进 `app/styles/`**，`app/globals.css` 只剩导入清单。**导入顺序就是层叠
顺序**（`@import` 在构建期被原样内联）：同优先级的规则谁在后面谁赢，所以 `dark.css`
必须排在被它覆盖的组件规则之后——`[data-theme="dark"] .custom-select-trigger` 与
`.appearance-item .custom-select-trigger` 优先级相同，把暗色规则挪到组件旁边就会翻盘。
读样式的测试走 `test/stylesheet.ts` / `test_window_config.py::_stylesheet`：它们解析
导入清单而不是遍历目录，所以没人导入的文件不会偷偷继续满足断言；反过来
`test/stylesheet.test.ts` 断言目录里的文件与清单一一对应——**新加一个 `.css` 忘了写
`@import`，是「组件悄悄没样式」而不是任何人看得见的报错**。

**界面文案一种语言一个文件**：`lib/translations.zh.ts`（加新键的地方，定义结构）+
`lib/translations.en.ts`，`lib/translations.ts` 只做类型与组装。英文那份标注为
`Translations`（由中文那份推导、把字面量放宽成 `string`），所以**漏一个英文键是编译错误**，
不再是界面上一块空白。

装机层原语（`AppPaths` 目录布局、校验下载、安全解压、uv 托管 Python 运行环境
`RuntimeEnvironment`、跨进程锁 `locks.py`、链接安全与崩溃安全的目录操作 `fsops.py`）
不在 `desktop/` 下，而在共享包
`src/finesub_bootstrap/`——它同时服务桌面端与 CLI 壳，禁止反向依赖 `desktop`。

目录分三个根，按数据行为而不是按谁写的划分：**数据根**
`%LOCALAPPDATA%\FineSub`（`user-data`：设置、API Key、知识库、任务历史；三种安装形式共用同一份，
所以一个用户只有一个知识库）、**安装根**（应用自身与 `runtime/`，版本绑定、从不共享）、
**大数据根**（`models`/`cache`/`tasks`/`agent-capsules`，默认等于安装根，可被 `finesub relocate` 指到别处并被
多个安装共用，位置记在数据根的 `locations.json`）。用户手工搬动目录后靠就近自检与
`register-location.cmd` 自愈；细节见 `docs/manual/resources.md`。
其测试仍在 `desktop/backend/tests`（desktop CI 是唯一的 Windows lane）；
PyInstaller 构建会把该包 stage 进 `--paths`。

**安装模式**：Inno 安装器在 `{app}` 写入 `installed.marker`（只有安装器会写；
更新载荷不含它，full updater 的 preserved 清单保它不丢）。它**不再决定个人数据的
位置**——安装版、便携版、CLI 现在共用 `%LOCALAPPDATA%\FineSub\user-data`，所以
一个用户只有一份知识库和一套 API Key；marker 目前没有运行时消费者，保留它是为了
「这份拷贝是安装器装的」这个事实本身（卸载器要用）。可重建数据（runtime/models/
cache）与任务产物（tasks）两种模式都在 exe 旁，可用 `finesub relocate` 搬走或
在多个安装间共用。卸载器显式删除运行环境/模型/缓存（Inno 默认不删运行期生成物），
并分别询问是否删除成品字幕与 `%LOCALAPPDATA%\FineSub`——按能否再生分档，与
`finesub uninstall` 同一套语义。便携版旧包内的 `user-data` 由迁移
`0002-user-data-to-managed-location` 自动搬到新位置，两处都有时不合并、持续告警。

任务恢复会复用原 task ID、请求、输出路径和历史记录。现有 pipeline 会跳过已经
完成的中间产物，并从同一个 LLM artifact 目录读取 session/window checkpoint。
恢复判据以产物是否已提交、源 `*-stable.json` 是否仍一致、存档是否可解析并能通过当前结构校验为准；
模型、prompt、预设、窗口预算或 continuity 改动只影响未完成部分，不会为了统一整条任务的执行配置
而使已完成窗口失效。最终任务因此可能混合多种配置，逐窗 metadata 保留实际来源。只有源变化、
计划/JSON/schema 损坏或响应无法重新校验等无法有效恢复的情况才自动重做。

**「完成后清理中间产物」**（`TaskRequest.cleanup_intermediate`，默认关）只在整条
pipeline 返回、字幕发布成功之后执行——中途失败会带着异常跳过清理，恢复所需的一切
原样留着。因此 LLM artifact 目录也在清理范围内：checkpoint 的存在意义是熬过一次
**失败**，而清理只发生在没有失败的时候。唯一无条件保留的是 `*-stable.json`，后面
每个阶段都读它，删了就要从音频重做分离与识别。

**同样保留、但机制不同的**：URL 输入下载到 `out/<stem>/` 的源媒体
（`<stem>.mp4`/`.ogg`，往往是清理后目录里最大的一块）与纠错翻译的
`-annotated.csv`/`-corrected.srt`。它们不在 `PipelinePaths` 里，
`_cleanup_intermediate_outputs` 看不见——这是**当前有意接受的结果**
（2026-08-08 确认）：annotated 要留作对照，源媒体是该文件在本机的唯一副本。
若以后要清它们，得先让这两类进入 `PipelinePaths` 或另给清理函数一份路径。

**任务进度**由 pipeline 的 `on_stage` 回调驱动（`run_pipeline` 的可选参数，CLI 不传
即完全不变）。上报点在 `run_pipeline` 各阶段的分支上，**不在 `_use_or_create` 里**：
整段跳过的阶段走 `elif` 兜底、根本不进那个 helper，LLM 段也不经过它，而那两类恰恰是
进度显示最容易出错的地方。复用已有产物的阶段以 `reused` 上报，界面显示「已有结果」。

**处理设备**：多卡时由设置页选择，选择随 `TaskRequest.gpu_index` 传下去，在
`jobs/launch.py` 的 `device_environment` 里翻译成 `CUDA_VISIBLE_DEVICES` +
`CUDA_DEVICE_ORDER=PCI_BUS_ID`（由同文件的 `spawn_worker` 在起进程时调用）。
放在 spawn 处有两个原因：`worker_env` 会被 `refresh_worker_context` 整体重建，而
retry/resume 重放的是 JobManager 存档里的请求、根本不经过 bridge。显卡清单由
`resources/gpus.py` 的后台探测（`nvidia-smi`）提供，界面只读快照、从不等待。

**安装日志**：`ResourceInstallManager` 的内存快照只留最后 100 行供界面轮询，完整输出由
`resources/install_log.py` 逐行写进 `user-data\logs\`，保留最近 100 份。

更新走签名的 GitHub Release，可在应用内直接下载安装：app 增量替换版本指针（重启
生效），full 包交给随包发布的独立 updater 替换整个安装（需退出 FineSub）。打开
Release 页面手动下载仍然保留为退路。

## 依赖

Python extras 位于根目录 `pyproject.toml`：

- `desktop`：桌面运行依赖，包括 pywebview、Pydantic、HTTP、版本及签名校验。
- `dev`：测试和桌面构建依赖，包括 Pillow、PyInstaller 和 hooks。
- `asr` / `harness`：原 FineSub pipeline；不属于桌面壳本身。

前端使用 Node.js 22，依赖由 `frontend/package-lock.json` 锁定。发布包使用
Windows 自带的 `Microsoft YaHei UI`、`Segoe UI`、`Cascadia Mono` 和
`Consolas`，不携带 Web Font。

最终用户的 Windows/Python 3.12/CUDA 12.8 AI 环境锁在：

```text
runtime/pylock.win-py312.toml
```

更新 AI 依赖后，在仓库根目录重新生成：

```powershell
uv pip compile pyproject.toml `
  --extra asr `
  --extra harness `
  --extra desktop-worker `
  --python-platform x86_64-pc-windows-msvc `
  --python-version 3.12 `
  --torch-backend cu128 `
  --format pylock.toml `
  --output-file desktop/runtime/pylock.win-py312.toml
```

改完 canonical lock **必须重新生成地区 lock**，否则两者会漂移：

```powershell
python -m desktop.scripts.make_cn_lock desktop/runtime/pylock.win-py312.toml `
  --output desktop/runtime/pylock.win-py312.cn.toml
```

它只改 artifact URL（镜像地址取自 `download-sources.json`），生成后立刻自检包名/版本/
marker/文件名/摘要是否与 canonical 逐项一致，不一致就删产物报错；测试对入库的两份文件
再跑一次同样的比对。**运行时 marker 只哈希 canonical lock**——两份 lock 描述的是同一批
文件、只差谁发货，按实际安装用的那份算会让跨地区被判成「依赖变了」而重建整个环境。

## 外部工具：托管资源，不进 lock

`desktop/resources/runtime-manifest.json` 声明外部工具（url + size + sha256 +
required_files），`ResourceManager` 通用地下载/校验/版本化/原子切换。**不进
`pylock.win-py312.toml`**：运行时 marker 含 lock 的哈希，改 lock 会触发整个 Python
环境重建（数 GB），而改 manifest 不碰运行时——对 yt-dlp 这种要跟版本的工具，差别是
3MB 对上数 GB。

| 工具 | 注入 | 何时装 | 复用系统已有 |
| --- | --- | --- | --- |
| ffmpeg | PATH ← `bin/` | setup 时 | ✅ 探测 + 编解码能力校验 |
| git (MinGit) | PATH ← `cmd/` | `--knowledge update` 时 | ✅ `which` + `--version` |
| yt-dlp | **PYTHONPATH ← 解压根** | URL 输入时 | ❌ 见下 |
| tokcount | `GEMINI_TOKEN_COUNTER_EXE` | LLM 阶段（CLI）／资源页手动（桌面） | ✅ 环境变量 + `which` + 数一次 |

tokcount 是本地 token 计数器（Go，源码 `tools/tokcount/`，构建与发布
见其 README）。它按**名字**注入而不是进 PATH：管线 `finesub.llm.token_budget` 先读
`GEMINI_TOKEN_COUNTER_EXE`，指名道姓才不会被机器上别的同名程序截胡。

**它是唯一一个「装不上也照跑」的工具**，两端都不拿它当门槛：没有它时 token 计数退到
免费的 `countTokens` 接口，只是从离线变成每次要一个网络往返、且需要 key。所以它既不在
`ALWAYS_REQUIRED`、也不在 `required_capabilities`，而是走
`preferred_capabilities`：CLI 在 LLM 阶段的任务前尽力拉一次，失败只打一行 stderr 就继续；
桌面端把它列为资源页的可选行（同「模型权重」那一档），由用户点。两端的解析顺序都是
「用户显式设的环境变量 → 系统已有的 → 托管的」，见 `environment.token_counter_overrides`。

yt-dlp 走 PYTHONPATH 是因为管线 `import yt_dlp` 用 Python API，不是调可执行文件。
也正因如此它**无法复用系统安装**：管线跑在托管运行时的解释器里，看不见用户的
site-packages。它的强制依赖为零，裸解压 wheel 即可 import（不装 `[default]` extra
的代价是部分站点降级：br 压缩、部分直播/加密流、YouTube 的 JS challenge）。

**git 与 yt-dlp 在桌面端是必需项**（2026-08-08 起进 `ALWAYS_REQUIRED`）。两者合计
约 40MB，而桌面端本就有意收掉这类自由度——「能不能粘链接取决于你装了什么」是比
多下一次更差的产品。缺任一项时新建任务页显示「请先下载资源」并跳资源页。

### 任务记录：`user-data/tasks.json`，两端共写

文件协议在 `finesub_bootstrap/task_index.py`：读、按 id 合并、原子写，写之前**先拿锁再重读**
（另一个前端可能刚追加过，直接写出自己的内存视图会静默吞掉它的任务）。桌面的
`_persist_history` 和 CLI 的运行记录都走它，所以只有一份实现。
普通读取遇到不可解析或结构无效的索引会降级为空历史，不影响实际任务；下一次写入会先在同目录
保留原始字节为 `tasks.json.invalid`（若已存在则追加数字），再建立新索引。备份失败时拒绝覆盖
原文件。备份只用于人工恢复，不会被程序自动合并。

索引还有第二个用途：桌面任务跑完把字幕发布到用户素材旁边（`<stem>-raw.srt` 等），而那个
名字可能已经被占了。**判据是「这条路径出现在索引任一条目的 outputs 里」**——是我们写过的
就覆盖（重跑同一个文件必须落回同名，否则一个视频攒一堆副本），不是就改落
`<stem>-raw.finesub.srt`（再冲突则 `.finesub.2.`… 递增）。worker 直接读索引而不是信任
`TaskRequest`：那是前端 payload 且整份持久化，加个「这文件是我们的」字段等于让前端可以
认领磁盘上任意路径。索引读不出来时判定全为「别人的」，宁可多写一个文件也不覆盖。

历史页每条任务有「清理中间产物」，走 `JobManager.delete_intermediates` →
`artifacts.cleanup_intermediate`，**失败/中断的任务同样开放**——恰恰是它们会留下人声分离
音频和解码副本（跑挂的任务根本没走到自己的清理）。一条例外：**未完成的任务保留
`*.llm-artifacts/`**，那是「继续」要读的断点，腾空间不该悄悄把已经花掉的 LLM 调用作废。

两端**跑在哪**不同，这点别弄混：桌面总把运行放进 `tasks/<task-id>/`，跑完按
`cleanup_intermediate` 清理自己的任务目录；`finesub` 只在用户没给 `-o` 时才这么做，给了
`-o` 就在用户目录里跑（产物可见地增长、失败也留在那儿、我们一个文件都不删），跑完把
`artifacts.RECORD_SUFFIXES` 那两个抄进任务目录当记录。所以任务目录对桌面是**原件**、对
`-o` 模式是**记录**。

匹配用输入的**解析后绝对路径**（URL 保持原样），并跳过 state 为 `running` 的条目 ——
续跑一个正在跑的任务等于两个进程写同一批产物。

条目里的 `request` 必须是完整、可由桌面 `TaskRequest` 校验的重试配置。桌面直接保存请求；CLI
保存该 schema 能表达的全部有效配置，且写入 **CLI 的实际默认值**（尤其 `device=cuda`、
`knowledge=none`），不能让桌面用自己不同的默认值补空缺。CLI 独有、`TaskRequest` 尚不能表达的
开关仍不会由桌面重放。

**索引不截断**：条目里没有 events(已完成任务写入前会剥掉),一条几百字节,一万条也就几 MB、
读一次几十毫秒,没有理由丢弃用户跑过的东西。截断只发生在渲染:`JobManager.history()` 取最近
`HISTORY_RENDER_LIMIT`(100)条。上限曾经在存储层,那意味着**任一前端**跑一次就可能把别人更早的
任务挤出记录 —— 而 CLI 恰恰要靠它找到值得续跑的旧任务。

CLI 在**开跑前**就写一条 `running`,跑完改成 `completed`/`failed`,Ctrl-C 走 `finally` 改成
`interrupted`。每个 CLI/worker 运行实例都强制持有
`<user-data>/.task-activity/<随机 id>.lock`；桌面 `JobManager` 还会在启动 worker 前先取得一份，
在闸门内用 `load_app_paths()` 重新解析最终 tasks 根，再把同一结果传给 worker。这样中断搬迁时
两端都会回退到 `migratingFrom/tasks`，应用空闲时被 CLI 搬迁后也不会让任务日志仍写回旧根；
前端此前缓存的“复用 ASR”请求也会从旧托管根重定向到新根。每个 reader 独立拥有这份交接租约，
新任务不会提前关闭仍在排空日志的上一 reader 租约。
空闲时的任务列表、重试请求、“打开任务目录”和“打开输出”也走同一刷新入口；解析根、更新磁盘
历史、创建和调用 Explorer 都在短期活动租约内，不能重新创建已搬空的旧根。历史合并遇到相同
`updated_at` 时保留磁盘记录，因为迁移只改路径表示而不伪造一次任务更新时间。
磁盘刷新会把其他前端的 `running` 放进任务列表，但绝不替换本窗口可操作的 `_snapshot`；后者只
代表本窗口实际拥有的 worker。多次 A→B→A 搬迁产生的旧 UI 路径别名会压缩到最新根且保持无环。
这条所有权判断看本地快照的生命周期，不看旧 `_process` 是否仍在退场：本地非 `running` 永远
不能被磁盘刷新提升为外部 `running`。worker 终态用事件产生时间更新历史，reader 延迟消费或
进程稍后退出都不再用消费时刻“刷新”旧世代；Cancel 和 Shutdown 也携带当前本地世代，在同一
历史文件锁内避让已取得相同 task-id 的后继任务。
注册前短暂取得稳定路径上的 `<user-data>/.task-activity.lock`。`relocate`/迁移持有同一闸门完成整次搬迁并逐个
检查活动租约，因此 A 先结束不会掩盖仍在运行的 B，新任务也不能插进「检查→搬迁」窗口。
`<tasks>/.active.lock` 仅为旧版本消费者保留为 best-effort 信号。每个任务另有
`<tasks>/.task-<task-id 的 sha256>.lock`:CLI 在规划后、写
`running` 前强制取得它,桌面 worker 也在进管线前强制取得它。同一 task_id 因此只有一个写入者,
另有 `<tasks>/.workspace-<output 父目录的 sha256>.lock` 保护实际产物目录：CLI/worker 固定先取
task-id 锁、再取 workspace 锁，因此新 task_id 复用旧 ASR 时也不能和旧任务的重试并发写。
不同任务且不同工作区仍可并行。进程崩溃后 sidecar 保留而 OS 释放字节锁,下一次才能安全续跑;旧版本留下、
没有 sidecar 的 `running` 条目无法判活,保守地不自动认作中断。更丰富的 owner/pid 信息仍见
`docs/cross-frontend-lease.md`。

⚠️ **不要往条目里加字段,也不要用 `Literal` 之外的取值**（`working`、`in-progress` 这类都不行）。`JobSnapshot`/`TaskRequest` 都是 `extra="forbid"`，老版本读到
不认识的字段会整条丢弃，而它下次写入只写回加载到的内容 —— 于是历史被静默清空。顶层加 key
是安全的（读取只取 `tasks`），条目里加不安全。用户完全可能只升级一端（`uv tool upgrade
finesub` 不动桌面），所以这不是理论风险。

### 「已装但版本旧」：`state="outdated"`

`ResourceStatus.state` 有第五个取值 `outdated`：磁盘上装着一份**能用**的、但不是
manifest 现在指定的版本。它与 `missing` 分开，是因为合并之后 bump 一次版本就等于对
所有人说「你从没装过这个」——而 git/yt-dlp 成为必需项之后，那会直接挡住每个人的任务，
包括从不粘链接的人。yt-dlp 需要周期性跟版本，这个代价会反复出现。

判据在 `ResourceManager.status()`：pointer 指向的版本目录里 required_files 齐全就算
「装着」，与 `spec.version` 不等则 `outdated`，并把磁盘上那个版本填进
`installed_version`（`version` 仍是目标版本，UI 好把升级的两端都显示出来）。pointer
指向的目录已经残缺则仍是 `missing` —— 那时没有可用副本，`outdated` 会是个假承诺。

**该问的问题是 `ResourceStatus.usable`（= ready 或 outdated），不是 `state == "ready"`。**
后者多断言了「而且是最新版」，那是另一回事，任务并不关心。后端 `ensure()` /
`task_ready()` 与前端 `blockingResources()` / `isUsable()` 都走这个口径；就绪计数把
outdated 算作就绪，「所需空间」不计它（旧的那份还能用，没人在等磁盘）。资源页给它
单独一档：标签「可更新」、按钮「更新」、版本行显示「目标版本：X（当前：Y）」。
`install()` 对 outdated 不会短路，照常下载并切换 pointer，也就是升级。

`capability_requirements`（`finesub_bootstrap/capabilities.py`）仍在，CLI 依旧按请求
实际用到的能力做按需校验；桌面端它现在只会点到 `ALWAYS_REQUIRED` 已覆盖的资源。

就绪计数与"所需空间"只统计必需项——可选项是模型权重（下节）与 tokcount。两者可选的
理由不同：模型权重是任务自己会下，tokcount 是任务根本不需要。

### 模型权重：资源面板里的第五行（`models`）

三个模型（BS-Roformer 分离器、faster-whisper large-v3-turbo CT2、Qwen3-ASR referee）
合并为一行。它**不进 manifest**：没有一个是我们下载的，各自由 audio-separator /
faster-whisper / huggingface_hub 拉取，所以和 `uv` 一样在 `DesktopResourceService`
里单开分支（`status` / `install` / `locations`）。

- **永远 optional**。缺模型的任务照样能开始、自己下——这一行只是让用户提前付这笔钱，
  而不是给「能不能开始」加一道 3 GB 的门。
- **Qwen referee 在列**，因为 `qwen_verify` 默认 `"auto"` = 「transformers 5.x 能导入
  就跑」，而托管运行时一定有。CLI 的 flag 看起来像 opt-in，桌面端实际不是。
- **检测走缓存复用规则**（`finesub_bootstrap/model_caches.missing_pipeline_models`）：
  已经躺在 `~/.cache/huggingface` 或 `~/.cache/audio-separator` 里的权重算已就绪，
  否则会提出重下三个 GB。HF 侧要求没有 `.incomplete` blob，并且至少有一个非空的
  snapshot 修订目录；只有仓库目录、空修订或明确的半截 blob 都不算就绪。预取进程即使
  以 0 退出也会再走一次这个检查，仍缺文件则安装失败，不能让资源管理器误报完成。
  **已接受的残余窗口**：文件是逐个下载的，中断恰好落在两个文件之间时没有任何
  `.incomplete` 痕迹，会被误判为完整——封死它需要拿远端文件清单（即解析 manifest）。
  代价只是行显示已就绪而首个任务补下剩余部分，即这行资源出现之前的原有行为。
- **分离器三个文件另有一条 manifest 化的取件路径**（`model_fetch.fetch_fixed_files`，
  CLI 侧在每次分离前也走）：带摘要的文件校验通过后在旁边留一枚
  `<文件名>.verified`（记 size + mtime_ns + 校验时用的 sha256），下次只要三项都对得上
  就跳过全量重算——639 MB 的 checkpoint 此前是**每次分离都从磁盘重新哈希一遍**。戳只是
  优化，从不充当判据：戳缺失、对不上、或 manifest 改了摘要，一律回落全量校验；没有摘要
  的文件（moving branch 上的上游索引）不写戳。老安装第一次全量校验通过时补上戳。
- **下载在托管解释器里跑**：`desktop.backend.worker.prefetch`，由
  `resources/model_prefetch.py` 拉起并把 stdout 翻成现成的 stage/log 回调。
  **没有字节级进度**——三个下载器三种进度口径，合并出来的百分比会是我们维护的数字
  而不是测出来的数字；界面走既有的不确定态进度条 + 阶段文字。
  Whisper 的仓库 id 与轻量缓存检测共用 `model_caches.WHISPER_REPO_ID`；不能把 friendly
  alias 交给 faster-whisper 私有映射后，再在另一处手写一个可能漂移的缓存名。referee
  仍直接取其生产模块的默认模型名。
- **暂停与退出**：launcher 以轮询方式检查暂停，不依赖子进程恰好打印一行；真正退出
  应用时 `ResourceInstallManager.shutdown()` 会向活动安装发暂停并等待 worker 收尾，
  避免 Windows 留下继续下载或占用 GPU、但已没有界面能控制的孤儿进程。
- **依赖 `uv`**：没有托管运行时就没有解释器可跑 prefetch，此时 `status()` 返回
  `blocked_by="uv"`，前端据此禁用按钮。这是 `ResourceStatus` 上第一个依赖字段。

⚠️ 用伪造任务预热**不可行**：`QwenReferee` 懒加载，合成音频不产生可疑段，referee
构造了却从不加载，那 1.5 GB 一个字节都不会下。

缺必需组件时，新建任务页不再显示「开始生成」，而是「请先下载资源」并跳转资源页。
判据是 `blockingResources()`（只看非 optional），所以**模型缺失不拦任务** —— 它自己
会下，这也是模型行与 git/yt-dlp 的分界：能自动获取的不拦，不能的拦。

## 应用日志

`user-data/logs/` 下三种文件，共用「保留最新 100 个」这一条规则
（`finesub_bootstrap/logs.py`——2026-08-17 从 `desktop/backend/common/` 搬过去的，因为管线
那侧也要写日志而 `src/finesub` 不能反向依赖 desktop；各前缀一起排序，谁也不因为前缀受保护）：

- `install-<资源>-<时间戳>.log` —— 单次资源安装的完整记录（`install_log.py`）。
- `session-<时间戳>.log` —— 一次应用运行（`launcher/session_log.py`）：启动、版本与
  安装根、窗口关闭、以及未捕获异常的 traceback。刻意做得很薄，**不是**通用日志设施；
  逐任务的记录归 worker 的事件日志，逐安装的归上面那个。
- `run-<时间戳>-<输入名>.log` —— 一次管线运行的 verbose 记录（`finesub.pipeline`）。桌面端
  的任务也会写它，因为它由管线自己产生，与前端无关。**它吃掉的是同一份 100 个的预算**。

逐任务的 `task-log.txt` 落在任务目录里，由 `jobs/task_log.py` 的 `TaskLog` **边收边写**
（`task-log.txt.part`，任务结束时原子改名）。它不再由结束时的快照重建：抽屉的事件队列
有上限，噪声大的运行会把自己的开头挤出唯一一份能在关掉应用后留存的记录，而被杀死的
worker 则什么都不留。verbose 细节走单独的 `debug` 事件——**只落文件、不进 UI 事件队列**，
因为逐 group 的救援细节会把抽屉那几百条预算一次花光。改名发生在终止事件那一刻（UI 就是
在那时翻出 running 并允许「打开文件夹」），之后到达的行追加到已改名的文件。

它存在的理由是应用可能无声消失（托盘退出、关窗、GUI 线程崩溃、未捕获异常都是同一个
结果：没进程、没提示、磁盘上什么都没有）。`main()` 在**任何东西可能失败之前**就开日志，
异常处兜 `BaseException`（Windows 关机送的是 `KeyboardInterrupt`）。最小化到托盘不写
「window closed」—— 所以一份没有结束行的 session 日志本身就是答案。

⚠️ 复用系统 ffmpeg 意味着行为依赖用户机器。解析到的路径与版本会写进 run metadata
的 `tools.ffmpeg`，转码差异可追溯。

`[desktop-worker]` 是这份 lock 独有的 extra：worker 在托管运行时里跑，需要
`[asr]`+`[harness]` 之外的 pydantic，以及**打过补丁的 CTranslate2**——后者以带
sha256 的 direct URL 锁定，因为原版能装上却跑不了 fw-refine（`docs/ct2-distribution.md`）。

`pylock.toml` 为每个分发包记录来源、平台 wheel 和 SHA-256，避免 PyPI 与
PyTorch 索引混用时降低 uv 的依赖混淆防护。运行环境只按该 lock 安装第三方依赖；
worker 通过 `PYTHONPATH` 直接运行当前版本随包发布的 FineSub 源码。lock 变化会使
runtime marker 失效并触发环境重建，普通应用更新不会无故重装数 GB AI 环境。

`status()` 的健康检查是**纯文件系统**的（site-packages 里的必需包目录 +
ctranslate2 dist-info 的补丁标签），瞬时完成、不起子进程——bridge 线程每次 poll
都会调它，import 探针（加载 torch 全栈，秒级）只在 install 校验 staging 时跑一次。
包内部深层损坏是目录检查的盲区，诊断入口（`finesub doctor` /
`status(force_probe=True)`）显式跑真探针兜底。

新环境先建在 `runtime/python.staging`，校验通过、写完 marker 后才**改名**就位
（旧环境先退到 `python.previous`，失败即回滚）。Windows 上这一步是整个安装最脆的
地方：目录改名在树内还有句柄时会被拒（刚写完的数 GB 文件正被杀软或网盘同步扫描），
目标名被占用时也是同一个"拒绝访问"——`MOVEFILE_REPLACE_EXISTING` 对目录无效。
因此换名带退避重试，目标名的判定用 `os.path.lexists`（`Path.exists()` 会跟随链接，
把指向别处的 junction 当成不存在），清理旧目录时链接只删链接本身、绝不递归进它指向
的目录。仍然失败时报错会说明是占用并给出处置建议，而不是抛原始 `WinError 5`。
此时 staging **保留**：它已经装完并校验过，重试只需再做一次改名，不必重装数 GB；
没写 marker 的残留 staging 则一律重建。

### 包内命令行

包根附带 `finesub.cmd` + `finesub.py`：前者在自己旁边找托管解释器，后者按 `app/current.json`
定位当前应用源码并交给 `finesub_bootstrap.shell`——**与 pip 安装的 `finesub` 是同一套子命令**，
因此知识库、`.env`、模型缓存、限流状态的落点与应用内启动完全一致（不再需要用户自己拼
`runtime\python -m finesub.pipeline`，那正是知识库曾经落进 `app/versions/<版本>` 的原因）。
两个 exe 都是 `--windowed`、拿不到控制台，所以入口走托管解释器而不是再打一个 console exe。
它 `can_provision=False`：自己就跑在托管运行时上，装不了也换不了这个运行时，缺资源时指回应用内的
资源面板，而不是给一条走不通的 `finesub setup`。

**子命令与帮助都出自一张表**：`finesub_bootstrap.shell.COMMANDS`（name → handler /
runtime 模块 → 帮助行 → 该命令对哪个前端可见）。`dispatch` 按它派发，两个前端的
`--help` 按 `render_usage(前端)` 渲染；帮助里的非命令行（默认的 `finesub <input>` 管线
分支、CLI 的 `Environment:` 段、包内的「跑的是这个安装」说明）各前端自己拼。可见性**不是
权限**：包内命令行照样派发 `setup`/`uninstall`，只是不列——装/卸留在 app 本体。
`test/bootstrap/test_shell_commands.py` 把「每个前端渲染出的帮助 = 表按该前端过滤后的
集合」钉死；此前两份 USAGE 手写，CLI 那份漏了 `agent-task`、包内那份漏了 `keys`。
包内命令行现在**先定位应用源码再打印帮助**（帮助文本来自共享表）：找不到源码的安装本来
也跑不了任何命令，说这句比给一份不存在的安装列命令有用。

## 开发

要求：

- Windows x64
- Python 3.12
- Node.js 22
- Edge WebView2

```powershell
# 默认安装桌面 UI、测试、构建，以及完整锁定的 ASR/LLM Pipeline 依赖
.\desktop\scripts\setup-dev.ps1

# 仅开发 UI、不运行字幕任务时，可显式安装轻量环境
.\desktop\scripts\setup-dev.ps1 -DesktopOnly

.\desktop\scripts\run-dev.ps1
```

API Key 保存在 FineSub Desktop 用户数据目录的 `.env` 中，不会返回给前端，但
当前仍是本机明文文件。Desktop 的 Gemini、Exa、Tavily 字段分别注入 CLI 的
`GEMINI_FREE`、`EXA_KEYS`、`TAVILY_KEYS`；Gemini 用于翻译，Exa/Tavily
仅在启用网页搜索时使用。旧版 Desktop 保存的三个单 Key 变量会在首次读取时
一次性迁移并改写。raw SRT 全程本地处理；纠错翻译会按所选 LLM profile 向
Gemini 上传必要的音频或视频片段。

## 测试

只有 `desktop/scripts/tests/test_desktop_dependencies.py` 在根项目的默认
`testpaths` 里——它守的是 `pyproject.toml` 的 extras 与 lock 之间的契约，而破坏
该契约的改动发生在仓库根，不在 `desktop/` 下（desktop CI 只在 `main` 上跑，等到
那时才发现就晚了）。其余桌面测试要么依赖 `[desktop]`、要么调用 `powershell.exe`，
根 CI（ubuntu + `[harness,dev]`）两样都没有，所以仍要显式运行：

```powershell
python -m compileall -q desktop
python -m pytest -q -n 0 desktop/backend/tests desktop/scripts/tests cli/tests

Push-Location desktop/frontend
npm test
npm run typecheck
npm run build
Pop-Location

python desktop/scripts/verify_static_export.py desktop/frontend/out/index.html
```

`.github/workflows/desktop-ci.yml` 在 Windows 上执行同样的后端、前端和
PyInstaller bootstrap smoke build。根项目原有 CI 不承担桌面验证。

## 构建

`desktop/scripts/` 的命名是有约定的，加脚本时照它走：**`.ps1` 用连字符**
（`build-release.ps1`，人从命令行调的入口，名字不必是合法标识符）、**`.py` 用下划线**
（`build_release.py`，会被 `import` 或 `python -m` 的模块——连字符的 `.py` 只能当脚本
执行，一旦有人想 import 就得改名）。同名的两半是同一件事的两层：`.ps1` 负责参数与
环境，真正的逻辑在同名 `.py` 里。

正式 bootstrap 需要非示例 launcher 配置和可信 Ed25519 公钥：

```powershell
Copy-Item desktop/resources/launcher.example.json `
  desktop/resources/launcher.json
# 创建被 desktop/.gitignore 忽略的：
# desktop/resources/trusted-update-keys.json

.\desktop\scripts\build-bootstrap.ps1
```

未显式传入 `-Version` 时，构建脚本会读取 `desktop/VERSION`；发布自动化如需
显式传值，也应先从该文件读取，避免生成版本不一致的资源。

bootstrap 产出 `FineSub Desktop.exe` 和 `updater/FineSub Desktop Updater.exe`
两个 PyInstaller 目标，不会生成 `FineSub.exe` 兼容副本。updater 是必需的：full
更新要替换正在运行的安装，执行替换的进程不能是被替换的那个——缺了它
`_install_full` 会停在 "Installed updater runtime is missing"，只有 app 增量能装。

updater 是 windowed 构建（无控制台），所以未捕获异常会变成 PyInstaller 的模态
traceback 弹窗，而此时 FineSub 已经退出、没人会去点它。`updater_main.main()`
因此兜住所有异常，把 traceback 写到 `<request>.error.txt` 并以 1 退出。

### 中断安全（2026-08-08）

替换过程有一个「安装根里没有任何可执行文件」的空窗期。围绕它的三条约束：

- **回滚兜 `BaseException`**。Windows 关机送的是 `KeyboardInterrupt`，`except Exception`
  漏掉它，而那正是空窗期最可能被打断的方式。回滚顺序是**先把程序文件搬回来**再清理，
  且每步独立兜底——清理失败不能成为程序文件回不去的原因。
- **启动即自愈**。`updates/recovery.recover_interrupted_update()` 在
  `create_application()` 最早期运行：安装根没有 `FineSub Desktop.exe` 时，自动把
  `.update/backup-*` 搬回来并告知用户。自动而非询问，因为此时没有界面可供询问。
- **备份只在安装可启动时才删**。旧代码在**下一次**更新开始时无条件清 `backup-*`，
  而那份备份可能是唯一一份能用的安装。`discard_backups()` 自己会复查这一点。

`<request>.error.txt` 由 `recovery.take_update_error_reports()` 读取并改名归档，
读一次即止——此前全仓没有任何读取方。`wait_for_parent` 的上限是 1 小时而不是 2 分钟：
界面承诺「退出后自动完成」且不带期限，用户很自然会先把手上的转写跑完再退出。

使用 Inno Setup 6 生成手动安装器：

```powershell
.\desktop\scripts\build-installer.ps1 `
  -ApplicationDirectory ".\dist\bootstrap\FineSub Desktop.dist"
```

## 发布（签名更新）

发布私钥必须位于仓库之外。公钥以 `desktop/resources/trusted-update-keys.json`
随包发布（该文件被 gitignore，构建时准备）。

更新检查读的是 **GitHub Releases 列表里最新一个带签名 manifest 的 release**，
不是 `/releases/latest`——这个仓库还发 CLI 快照和 patched CT2 wheel，仓库级的
"latest" 会被它们顶掉（`is_desktop_release()`）。所以一个 release 要被桌面版
认作更新，必须同时带 `update-manifest.json` 和 `update-manifest.sig`。

CLI 与桌面**共用一个版本号、一个 tag、一个 Release**，由
`test_the_cli_and_the_desktop_app_ship_one_version_number` 强制。更新服务按
`v{manifest.version}` 解析 release，版本号分叉会指向不存在或没有桌面资产的 tag。
（`v0.3.0` 是这条契约成立之前发的 CLI-only release，所以联合发布线从 0.3.1 起。）

⚠️ **「更新之后数据还在」测不出来。** 保留名单的内容有测试钉住
（`updater_main.py` 的 `preserved`），但整条链路——下载签名 manifest、更新器原地
替换整棵树、用户数据幸存——要私钥和一个**已经发布过的**旧版本，本地构造不出来。
所以**动过保留名单或数据布局的版本，发版时必须演练一次 app 增量和一次 full**。
演练步骤在发布 skill 的「验证收尾」里。

```powershell
# 1. 产出 app/full 包 + 签名 manifest（版本号取自 desktop/VERSION）
.\desktop\scripts\build-release.ps1 `
  -Version (Get-Content desktop\VERSION -Raw).Trim() `
  -KeyId finesub-release-2026 `
  -PrivateKeyPath <仓库外的 .pem>

# 2. Inno 安装器（README 引导新用户从 Release 下载它；full 包兼作 portable 下载）
.\desktop\scripts\build-installer.ps1 `
  -ApplicationDirectory ".\dist\bootstrap\FineSub Desktop.dist"

# 3. 构建 CLI wheel（与桌面同版本同 Release；见 cli/README.md）
.\cli\scripts\build-wheel.ps1 -Version $Version

# 4. 建 Release：前四个桌面资产缺一不可；Setup 与 CLI wheel 是面向新用户的
#    下载入口（根 README 指向它们），一并上传
gh release create "v$Version" `
  dist\release\update-manifest.json `
  dist\release\update-manifest.sig `
  "dist\release\finesub-app-$Version-win-x64.zip" `
  "dist\release\finesub-full-$Version-win-x64.zip" `
  "dist\installer\FineSub-Desktop-$Version-Setup.exe" `
  "dist\cli\finesub-$Version-py3-none-any.whl"

# 5. 同一个 wheel 发 PyPI（`uv tool install finesub` 的来源；token 存仓库外，
#    定期轮换）。版本号不可重传——传错只能 yank。
uv publish "dist\cli\finesub-$Version-py3-none-any.whl" --token <pypi-token> `
  "dist\cli\finesub-$Version-py3-none-any.whl"
```

`-SupportedFrom` 默认为空 = 所有旧版本都拿 full 包。下一次发布时才把可以走 app
增量的版本列进去（如 `-SupportedFrom 0.3.1`）。0.2.7 及更早一律 full。

⚠️ **资产要一次传齐**：`is_desktop_release()` 要求两个 manifest 资产同时存在，
分批上传期间的 release 会被跳过（有测试覆盖），但先建 draft 再发布最稳妥。

应用内安装的接线：`install_update(kind, version)` 起一个后台线程跑
`GitHubUpdateService.install()`，前端轮询 `get_update_install()` 拿进度快照
（`UpdateInstallManager`，与运行时资源下载同一套形状）。bridge 调用必须立即返回
——pywebview 在绘制窗口的线程上派发它们，而 full 包是几百 MB。

前端传的 `kind` 只是它从 `check_updates` 看到的值；`install()` 会从签名 manifest
重新推导并拒绝不一致，所以过期页面无法诱导后端装错载荷。
