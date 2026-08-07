# Changelog

## [Unreleased] — 0.3.1

### 桌面窗口：原生缩放、标题栏行为与主题动效

无边框窗口重新可以用鼠标拖边缩放：加回 `WS_THICKFRAME` 并用 `WM_NCHITTEST`
子类化自己判定边框与标题栏区域，安装在 WinForms 的 UI 线程上。pywebview 的启动
回调早于 WinForms 窗口存在，所以这些工作等 `shown` 之后再做；等待超时只打
`Warning:` 并继续绑定文件拖放，不再让整个回调抛异常。最小尺寸 900×620 → 720×520。
标题栏双击最大化/还原，最大化按钮改为切换。

原生边框与标题栏配色跟随应用主题（新增 bridge 方法 `set_window_chrome`，前端在
每次外观变更后回传当前 `--app-bg`/`--text`）；窗口刚创建、前端还没画出来的那几帧
按 Windows 的浅色/深色设置取色，默认的「跟随系统」主题因此不会闪烁另一半配色。

动效：侧栏切换的移动指示条、主题色切换的 View Transition、任务高级选项的展开。
全部是 CSS/浏览器原生能力，没有引入动画库；设置页可整体关闭，也尊重系统的
`prefers-reduced-motion`。设置页的「查看文档」改为应用内对话框。

安装器：简体中文不再取决于编译机上有没有装社区译本（Inno Setup 官方不含简中，
此前的探测在 CI 上一直落空，发出去的其实是英文界面）——译本随仓库分发。英文保留，
按系统 UI 语言自动选择，不再弹语言选择框。

### `.env` 密钥保护：绑定 Windows 账户

`.env` 里的 API key 不再以明文落盘：随机主密钥经 DPAPI（绑定当前 Windows 账户，带
应用 entropy）包裹后存在文件首部的 `FINESUB_KEYRING` 行，每个密钥值原地替换为
`fs$…` 密文（信封加密，纯 stdlib）。变量名、命名 key 的显示名、注释与换行逐字节
保留，`cat .env` 仍能与 `config.toml` 的 `[pools]` 对照。转换由启动迁移
`0004-protect-env-keys` 完成（源码 checkout 则在首次读取时兜底），失败一律退回明文
加 `Warning:`，绝不阻塞运行。

对外措辞统一为**绑定当前 Windows 账户的保护 + 防泛扫混淆**：它防误传文件、通用
扫盘、DPAPI 批量收割、同机其他账户，不防以当前用户身份执行的代码（程序必须能无
口令自解密）。密文随 `.env` 拿到别的机器上表现为「未配置」且**一个字节都不动**；
只有在用户显式重填/删除全部不可解密的值之后，保护才会以新机器的账户自动重建。

导出通道：`finesub keys`（默认掩码，`--reveal` 输出可直接粘回 `.env` 的明文，
`--out FILE` 落盘并警告）、桌面端设置页「显示已保存的密钥」（新增 bridge 方法
`reveal_api_keys`）、`finesub doctor` 新增 `env-keys` 状态行。换机、重装 Windows
之前先导出。

过渡开关 `FINESUB_ENV_PROTECT=0` 暂停自动加密（解密照常、迁移保持未完成），给
仍直接读 `.env` 的旧代码（未收敛的 worktree）留缓冲；收敛后移除变量即自动转换。

配套：`finesub_bootstrap/secrets.py` 成为全项目唯一的 `.env` 解析/写入层（
`llm_runtime` 与桌面 SettingsStore 的两个自建解析器删除；桌面写侧从「白名单整文件
重写」改为逐行保留式更新）；`.env-sample` 删去死配置 `EXA_POOL`/`TAVILY_POOL`；
`.gitignore` 补 `.env.*`。

CLI 与桌面从这一版起**共用一个版本号、一个 tag、一个 GitHub Release**（由
`test_the_cli_and_the_desktop_app_ship_one_version_number` 强制）。`v0.3.0` 是这条
契约成立之前发的 CLI-only release，所以联合发布线从 0.3.1 起。

### 数据布局：一个用户一份数据

三种安装形式（桌面安装版、便携版、pip 安装的 CLI）的**个人数据统一到
`%LOCALAPPDATA%\FineSub\user-data`**——此前便携版把它放在包内，于是同一个用户可能有三个知识库，
还不知道应用在读哪一个。大文件（`models`/`cache`/`tasks`）默认仍在安装目录下，可以用
`finesub relocate <目录>` 整体搬到别的盘，也可以让两个安装指向同一处、不重复下载几个 GB；
位置记在数据根的 `locations.json`，找不到就自动回落，不报错。用资源管理器整个搬走安装目录能被
自动认出来；只搬大文件目录的话，到新位置双击里面的 `register-location.cmd` 即可。

`runtime` 永远留在安装目录下：它与版本绑定，而且 uv 在同盘时用**硬链接**把 wheel 从缓存链进
环境（实测 `torch_cpu.dll` 一份数据两个路径），把两者分到不同磁盘会让硬链接退化成复制、
总占用**反而多出约 5 GB**。装环境前会比较两者所在磁盘并在不同磁盘时告警。同理，单独清
`cache\uv` 几乎不释放空间——先删运行环境再清缓存才有效。

`tasks` 从 `user-data` 里挪了出去（每任务几十 MB、无上限增长），并进了更新器的保留名单；
任务历史改存相对路径，所以手工搬动文件夹之后"打开文件夹"不再全部失效。
卸载改成三档：默认删可再生的运行环境/模型/缓存，成品字幕与个人数据分别要
`--purge-tasks`、`--purge-user-data`；大文件目录一旦搬走或共用，默认也不删（另一个安装多半还在用）。

**从仓库源码运行仍然用仓库自己的数据**（`knowledge/`、`.env`、`.state`），
`FINESUB_CHECKOUT_DATA=0` 可退出；git worktree 解析到主仓，且 worktree 内的知识库自动更新默认
跳过，需要 `FINESUB_KNOWLEDGE_WRITE=1` 才写。

### 修复：并发写坏共享数据的三处

个人数据共享之后，几个此前不可能发生的竞争变成了可能，一并补上跨进程锁：数据迁移
（锚在 user-data **外面**，因为迁移要搬的正是它）、知识库 auto-apply（锚在
`<knowledge_root>.lock`，即知识库目录的兄弟文件——放里面会被 auto-commit 收编，锚在安装根则三个
前端锁的是三个不同文件）、以及共享下载缓存里同一个压缩包的并发下载（此前会互相写坏
`.part`，报出看起来像被投毒的 SHA-256 不匹配）。任务历史改成"锁 + 重读 + 按 id 合并"回写，
不再是内存快照整体覆盖。知识库拿不到锁的行为与"仓库脏"一致：跳过、保留提案、不推进 ledger、
只打 warning。

### 修复：装完 2.8 GB 依赖后倒在最后一步改名

有用户在装 Python 运行环境时拿到 `[WinError 5] 拒绝访问`：依赖全部装完、校验也过了，
只差把 `runtime/python.staging` 改名就位。Windows 上目录改名在树内还有句柄时会被拒
（刚写完的数 GB 文件正被杀软或网盘同步扫描），目标名被占用时也是同一个错误——
`MOVEFILE_REPLACE_EXISTING` 对目录无效。改名现在带退避重试；目标名的判定改用
`os.path.lexists`（`Path.exists()` 会跟随链接，把指向别处的 junction 当成不存在）；
清理旧目录时链接只删链接本身，不再递归进它指向的目录。仍然失败时报错说明是占用并给出
处置建议，而不是抛原始 `WinError 5`；且**保留已装好的 staging**——它已通过校验，重试
只需再做一次改名，不必重装数 GB。

### 修复：知识库被写进应用目录，会被下次更新删掉

桌面端装不上时，用户会直接用包内解释器跑 pipeline。这条路绕开启动器注入的
`FINESUB_KNOWLEDGE_ROOT`，而发行包的 `app/versions/<版本>` 同样带 `pyproject.toml` +
`src/asr_playground`，于是被当成源码 checkout，知识库落在了 `app/versions/<版本>/knowledge`。
更新器的保留名单只有 `user-data`/`models`/`runtime`/`cache`，`app/` 整体替换——数据会静默消失。
checkout 探测现在显式排除这种布局，并按安装布局反推数据根（安装版走
`%LOCALAPPDATA%\FineSub\user-data`、便携版走包内 `user-data`），与启动器注入的位置一致；
`.env`、`config.toml`、限流状态文件同理。已经写错位置的那份由新的数据迁移搬回来。

### 桌面包自带命令行

包根新增 `finesub.cmd` + `finesub.py`，与 pip 安装的 `finesub` **同源子命令**
（实现下沉到 `finesub_bootstrap/shell.py`，两个前端共用），直接驱动它所在的那份安装。
用户和 agent 不必再自己拼 `runtime\python -m asr_playground.pipeline`——那正是上一条的
成因。它不负责装资源：自己就跑在托管运行时上，缺资源时指回应用内的资源面板。

### 用户数据迁移机制

新增 `finesub_bootstrap/migrations/`：按 id 记账（`user-data/.migrations.json`）而非版本区间，
因为桌面与 CLI 共享同一棵 `user-data` 且各自跳版本。启动器与命令行都会在读用户数据之前跑一次；
失败只记日志、下次重试，绝不影响启动。首个迁移把 `app/versions/*/knowledge` 搬进
`user-data/knowledge`；两边都有知识库时不自动合并，持续告警直到人工处理。

### 首次任务提示

首次任务要按需下载模型权重（合计约 3.4 GB）并预热分离器的编译路径，这些都发生在任何
进度出现之前，不说明就像卡住了。新建任务页（选好输入后）与处理页各提示一次。判据是
「没有已完成的任务」而非「历史为空」——只失败过的机器同样还没缓存任何权重。

### 修复：安装 Python 环境时界面假死

`status()` 里的运行时体检会**同步**起一个 Python 去 import torch + 整条解码链——
实测 **14.7 秒**（热缓存，冷启动更久）。而它跑在 pywebview 绘制窗口的那个线程上，
`get_bootstrap_state` 和每次资源轮询都会调。表现就是"点安装卡住、点暂停后又显示成功"
（那次点击触发了新一轮，此时探针刚好返回）。是上一轮把 `REQUIRED_RUNTIME_IMPORTS`
从 4 个扩到 8 个引入的。

`install()` 本来就在写 marker 之前验证过，而 marker 绑定 lock 哈希——所以 `status()`
再跑一遍是在 UI 线程上重复证明已证之事。改为纯文件系统检查（site-packages 下的包目录
是否还在，CT2 的补丁版从 dist-info 目录名读），进程零开销；导入探针保留在安装时。

### 模型优先复用本机已有缓存

`FINESUB_MODEL_DIR` 一旦设置就完全接管，从不看常见缓存目录，于是已经下过的权重被再下
一遍。改为"先找再下"（`finesub_bootstrap/model_caches.py`），两种粒度：

- **分离器**：精确到文件——检查 `~/.cache/audio-separator` 里有没有那个 ckpt。
- **Hugging Face**：只有一个内容寻址的缓存根、无法搜索多个，所以是**按缓存整体**判断：
  常规根里已有本管线用到的任一仓库就整体复用（包括之后新下的）。里面若只有别人的模型
  则不动它——往别人的缓存里下载既意外、卸载也清不掉。显式设了 `HF_HOME` 一律不猜。

编译加速产物（accel）**不跟着走**：它绑定单一 torch 构建与 GPU，写进共享缓存会留下
无人能归属、也无人能清理的文件。为此新增 `managed_separator_model_dir()`。

### 修复：拖拽时的卡顿

`dragenter`/`dragover` 的 Python 回调是空的，只为 `preventDefault` 而存在——而
pywebview 生成的监听器同步执行 `preventDefault` 后，仍会把整个 DragEvent（含
dataTransfer）序列化过桥调用它，`dragover` 每秒几十次。加 `debounce=500` 压掉这些无用
往返；`preventDefault` 不受影响，drop 仍然即时。

### 修复：LLM 阶段因缺 tzdata 直接崩溃

`llm/rate_limit.py` 在**模块导入时**构造 `ZoneInfo("America/Los_Angeles")`（对齐 Gemini
的日配额窗口），而 Windows 的 Python 不自带 tz 数据库。`tzdata` 从来没有出现在
`pyproject.toml` 里——只在 `.github/workflows/ci.yml` 有一行临时 `pip install`，那正是
同一个 bug 的补丁。任何按 extras 安装的环境（包括桌面托管运行时）一进 LLM 阶段就
`ZoneInfoNotFoundError`。已加进 `[harness]`（限 Windows）并重编 lock；CI 的临时行删除。

### 任务目录与日志

- 任务 id 由裸 uuid 改为 `<stem>-YYMMDD-HHMM-<6位hex>`，`user-data/tasks` 下终于能看出
  哪个目录对应哪次任务。stem 取自输出名称，没设则取源文件名。
- 任务结束后（成功/失败/取消）自动把日志写到该任务目录的 `task-log.txt`——日志抽屉
  是有上限的环形缓冲且随应用关闭消失，而值得上报的失败往往过后才被注意到。
- 「复制日志」改为「导出日志」（下载为文件），历史页右上角新增「打开任务目录」。

### 桌面任务表单调整

- 输出结果收敛为两项：原始字幕（无 LLM 处理）/ 最终字幕（须 LLM 处理）。
- 补充信息移到基础设置并给出实例文案；处理设备移入高级设置。
- 输出名称去掉常驻说明，改为**填错才提示**（与 `TaskRequest.validate_name` 同规则）。
- 知识库默认改为「自动更新」。配套修正一处会误伤的门禁：
  `required_capabilities` 现在要求 stage 真的进到 LLM 才需要 git——否则默认设置下
  每个纯转写任务都会被要求下载一个根本不会运行的 git。
- 运行日志显式可选中，并新增「复制日志」按钮。
- 资源磁盘估算修正：`uv` 那一栏此前标 24.5MB（uv 二进制），但安装它会拉取整个
  `pylock.win-py312.toml`——**实测 torch 一个就 2.56 GiB，合计约 2.83 GiB**，低估了两个
  数量级。另在提示里补上模型权重的按需下载估算（约 3.4 GB），并把「模型如何管理」
  的说明改为逐个列出权重与体积。

### 桌面支持 URL 输入

- DropZone 增加链接输入（文件选择 / 拖放 / 粘贴链接三选一）。管线一直支持 URL，
  桌面此前单方面砍掉了这个入口。链接与文件走同一条状态路径，yt-dlp 按需拉取。
- 前端的 URL 判定 `isUrlSource` 与后端 `finesub_bootstrap.capabilities.is_url`
  规则一致——不一致就会出现"UI 收下了、后端拒绝"的输入。

### 外部工具改为托管资源（git / yt-dlp），并复用系统已有的

- **manifest 新增 git（MinGit）与 yt-dlp（PyPI wheel）**，`ResourceManager` 一行未改。
  不进 lock：运行时 marker 含 lock 哈希，改 lock 会触发数 GB 的环境重建，而改 manifest
  不碰运行时。注入方式按性质分——git 走 PATH，yt-dlp 走 **PYTHONPATH**（管线是
  `import yt_dlp`，不是调可执行文件）。
- **懒装**：git 只在 `--knowledge update` 时装，yt-dlp 只在 URL 输入时装。规则收在
  `finesub_bootstrap/capabilities.py`，桌面（读 TaskRequest）与 CLI（读命令行）共用，
  避免两个入口对"这次运行需要什么"产生分歧。
- **复用系统已有的 ffmpeg / git**：照 `_find_system_python` 的模式——`which` 找到后
  实际执行校验（ffmpeg 还要查必需编解码器，缺了就退回托管副本，否则会在管线中段才炸）。
  一台已有 ffmpeg 的机器因此省掉 146MB。探测可注入，否则测试结果会取决于跑测试的机器。
  yt-dlp 无法这样复用：托管解释器看不见用户的 site-packages。
- `task_ready` 改为按请求校验，错误信息说明缺的是哪个工具（此前无论缺什么都说
  "请先安装 Python 运行环境和 FFmpeg"）。
- CLI 从 manifest 取全部资源（此前硬编码只取 ffmpeg），`doctor` 统一报告三者状态与来源。
- run metadata 新增 `tools.ffmpeg`：复用系统版本让行为依赖用户机器，路径与版本要可追溯。
- `ResourceStatus` 新增 `optional`：按需工具在资源面板里列出（否则"缺 git"的报错会把
  用户指向一个找不到 git 的面板），但不计入就绪数与所需空间，缺失时也不显示告警图标。

### 知识库：git 缺失不再让任务失败

- **`_run_git` 不再抛裸异常。** 本项目不安装 git，所以「没有 git」是常态；此前
  `subprocess.run(["git", ...])` 会抛 `FileNotFoundError`，在字幕已经落盘之后把整个
  任务带崩。改为返回 `returncode=127` 的合成结果，调用方现有的「非零即失败」处理原样生效。
- **前置拦截，不浪费配额。** `run_knowledge_update` 在 `execute and apply` 时先查 git，
  不可用就返回 `skipped: "git_unavailable"`，**一次 API 调用都不发**（此前要先花钱生成提案、
  走到 apply 才发现装不了）。
- **三条失败路径统一降级为 warning，且都不推进 chunk ledger**，所以修好后重跑会完整重做：
  git 缺失 / 仓库脏或在别的分支 / 文件已改但 commit 失败。最后一条原先在改成 warning 后
  会继续走到 `_append_chunk_ledger`——等于把「没提交」记成「已完成」，那批改动将永远不被记录；
  现已阻止。

### 桌面任务控件

- 新增 `name`（对齐 CLI 的 `--name`，产出 `out/<name>/<name>.srt`，带路径分隔符校验）、
  `extra_info`、`knowledge`（暂只暴露 none/update）、`cleanup_intermediate` 四个控件；
  移除联网检索开关（保持默认开启）。
- **中间产物默认不再清理。** 此前无条件删除，连 `stable.json` 和整个 LLM artifact 目录
  一起删——后果是重跑要从头做分离与识别，纠错翻译也没了输入和 checkpoint。现在改为可选，
  且**即便勾选也始终保留 `stable.json` 与 artifact 目录**：它们相对人声音频很小，却决定了
  重跑是否廉价。

### 桌面自动更新（此前从未跑通）

- **更新检查不再打 `/releases/latest`。** 那是仓库级的，而本仓库还发 CLI 快照与
  patched CT2 wheel——发完 wheel 之后 "latest" 就指向它，更新检查必然抛
  "missing the signed update manifest"。改为列举 releases、取最新一个**真正带签名
  manifest** 的（`is_desktop_release()`），签名与 tag 校验仍在其后兜底。顺带覆盖
  资产分批上传期间的半成品 release。
- **签名发布流程从未被执行过**：v0.2.7 只发了 portable zip，`build_release.py`
  产出的 `update-manifest.json` / `.sig` 一次都没上传。`desktop/README.md` 新增
  发布 runbook，说明四个资产缺一不可。
- `build-release.ps1` 的陈旧默认值（`-SupportedFrom 0.2.3`，一个从未发过签名
  manifest 的版本）改为空 = 所有旧版本拿 full 包，增量必须显式声明。
- **应用内一键更新接线完成。** 新增 `install_update` / `get_update_install` 两个 bridge
  方法与 `UpdateInstallManager`：安装跑在后台线程，前端轮询进度快照（与运行时资源下载
  同一套形状）。bridge 调用必须立即返回——pywebview 在绘制窗口的线程上派发它们，而 full
  包是几百 MB。设置页显示下载进度，完成后按 app/full 分别提示"重启"或"退出以完成更新"。
- **独立 updater 恢复构建。** `desktop/FineSubUpdater.py` 一直在仓库里，但
  `build-bootstrap.ps1` 不构建它——而 `_install_full` 要求
  `<root>/updater/FineSub Desktop Updater.exe` 存在，否则抛 "Installed updater runtime
  is missing"。也就是说 full 更新在任何真实安装上都不可能成功，而按上面的策略 0.2.7 →
  0.3.1 恰恰只能走 full。已补回第二个 PyInstaller 目标并实跑构建验证。
- **updater 失败不再挂住进程。** 它是 windowed 构建（无控制台），未捕获异常会变成
  PyInstaller 的模态 traceback 弹窗——而此时 FineSub 已退出，没人会去点。实测确认：修复前
  一个坏请求会让进程一直活着。现在兜住异常、写 `<request>.error.txt`、以 1 退出。

### 托管运行时

- **lock 重建。** `desktop/runtime/pylock.win-py312.toml` 上次对齐是
  2026-07-31，`[asr]` 还是 whisper-timestamped 时代；fw-refine 迁移之后它**一个解码器
  都不含**（faster-whisper / ctranslate2 / transformers / silero-vad / triton-windows
  全缺），装得上、跑不了。现按今天的 `[asr]+[harness]+[desktop-worker]` 重新生成：
  torch 2.8→2.11.0+cu128，包数 75→88。
- **补丁版 CTranslate2 进 lock。** `[desktop-worker]` 用 direct reference 锁到 release
  wheel（带 sha256）。桌面运行时只有 win_amd64/cp312/cu128 一个组合，所以 `[asr]` 那边
  规避的平台钉死在这里是零成本的。开发机与端用户安装都走同一份 lock，两边自动拿到
  补丁版。
- **运行时校验补全。** `REQUIRED_RUNTIME_IMPORTS` 原本只查分离器一侧，加入
  `faster_whisper` / `ctranslate2` / `silero_vad` / `transformers`，并校验 CT2 的
  `__version__` 带 `wtrefine`——原版能 import、能满足 `==4.8.1`，只是跑不了 fw-refine。
  探针脚本抽成 `runtime_probe_source()` 以便直接测试。
- **lock 漂移进默认测试。** `desktop/scripts/tests` 加入根 `testpaths`：契约由仓库根的
  `pyproject.toml` 打破，而 desktop CI 只在 `main` 上跑。断言从"torch 版本对不对"改为
  "三个 extra 的每个直接依赖都在 lock 里且版本相容"，拿旧 lock 验过会报 10 条。
- 桌面任务恒开 `vad_silero_assist`（CLI 仍是 opt-in）：桌面任务必经分离器，而流式化后
  它的边际成本约 1s。
- 后台控制台隐藏、真实系统托盘、角色主题与 Yanami 主题、深色完成页、纯字幕产物发布、
  本地视频先转码再 ASR、分离器/RoFormer 运行时依赖校验、实时日志跟随（PR #7）。
- 清理：`setup-dev.ps1` 的 `-IncludePipeline` 已成空开关，删除；
  `media/source.py` 的 `ensure_aac_audio` 死别名删除；测试里 8 处
  `whisper_timestamped` 桩模块 monkeypatch 与 `HEAVY_IMPORTS` 条目删除（生产代码早已
  零引用）。

## [0.3.0] - 2026-08-05

### 移除

- **`whisper-timestamped` backend 整个移除**，`fw-refine` 成为唯一 ASR backend。
  同时删除：`--asr-backend` 开关（pipeline 与 vad-asr）、`asr-wt` 命令、`WtModelPool`、
  `naive_approach` 退避梯子、`asr_transcribe_seed` / `whisper_timestamped_mode` metadata
  （fw-refine 从不读 seed，那条记录一直是假的），以及 `whisper-timestamped`、`dtw-python`
  两个依赖。checkpoint fingerprint 的 `asr_backend` 字段随之删除（只剩一个 backend）。
  理由：为优化要改 refine 内部逻辑，维持两套行为对齐的成本不划算；迁移验收（5 素材 /
  50.6 分钟）显示 fw-refine 快 3.19×、内容量差 ≤1%、救援活动在每个素材上都更少。
  回溯点：`dev` 的 `1fcc4e1`。
  **`asr-refine` extra 因此从可选变为必需**。

### 新增

- **`--qwen-verify {auto,on,off}`（默认 auto）：第二模型校验证据。** Whisper 池释放后加载
  Qwen3-ASR-0.6B，对三类嫌疑段（整段收尾套话、CJK 主导 run 里的 Latin 段、噪声腿将丢弃的段）
  重认，证据写进段级 `qwen_verify`，决策留给 stabilize。`[asr]` 因此新增
  `transformers>=5.13,<6`（pip 增量约 100MB，模型首次运行下载约 1.5GB 至 HF 缓存）。
  并入 `[asr]` 而非可选 extra，是为了让同一条命令在任何安装上产出相同的 stable。
- **词首修正**（`speech/recognition/word_starts.py`）：`[*]` disfluency 块按能量门决定
  融合/删除，再对首词做 VAD interval 与 pause_hint 锚点 clamp。gold 上词首 |err| 中位
  41→18ms。
- **LLM 窗口质量护栏 `max_window_subtitle_tokens`**（`ModelLimits` 默认 10,000，config.toml
  `[chunking]` 可覆盖，`0` 关闭）：单窗 `<asr_result>` 的 token 上限，独立于输出系数——
  窗口过长时翻译质量会掉，哪怕输出装得下。窗口数估算与真实 countTokens 校验两处生效，
  超限走既有的 k+1 重排；快速模式的 auto 判定也以它为第三道门（快速窗口就是全片，
  最容易撞上）。
- `energy.WaveformObserver` 钩子：让第二个信号搭车读取 VAD 已经算好的归一化 block，
  不必再解码/重采样/归一化一轮。不传时 energy 模块的每个输出值逐字节不变。
- `fw-refine` 的 multi-audio batch 设计与本机实测落入 `docs/wt-refine-port.md`：CT2 的
  `real_audio_frames` 改为逐样本、split-encode 批模式、确定性契约，以及模型 × beam × GPU profile
  的 batch size 档位表。迁移本身值 6.3×、batch 再叠 1.8×，故 batch 不阻塞 P0。
- 新增显式 opt-in 的 patched CT2 `fw-refine` checkpoint：greedy/beam=5 以 1-pass
  winner trace 对齐 WT refine，默认收集低成本 path 信号；修复后的 disfluency 保持显式
  开关，启用后同样以 `alignment_events` 透传到 aligned/stable 产物。FineSub 暂不消费。
- `fw_refine_backend.transcribe_batch()`：一次批量解码若干 ≤30s 窗口。split-encode（逐窗口
  encode、只批 decoder），再把每条结果回放进普通 `transcribe()`，因此 segment/词/事件的组装
  只有一份实现。24 个真实生产窗口上文本一致 22/24（18 条连词级时间逐位相同），加速 1.73×。
  调用方的组批策略尚未实现。

### 变更

- **VAD 归一化的峰值限幅由全局缩放改为逐样本 clamp——会改变 VAD 输出。** 旧实现是
  `x * (0.98 / 全局峰值)`，让**单个最响样本**决定施加到整轨的位移：kaguya60 的 0.00029%
  样本把 60 分钟压低 2.18 dB，mia 的 0.00004%（约 25 个样本）把 108 分钟压低 4.12 dB；
  7 条真实分离人声里 4 条触发。而 VAD 判据里有一批绝对 dBFS 阈值，它们能跨文件通用正是
  靠上一步把局部 RMS 对到 −24 dBFS，全局缩放随后又按文件拆掉一部分标定。改为削掉越界样本
  后，weighted 轨上受影响帧从「全部有信号帧」降到 179/4292，越阈帧从约 2000/20000 降到
  0/1。**限幅曾触发的文件重跑会得到不同 segment**（kaguya60 752→753、语音 +4.8s；
  mia 2561→2522、+46.1s），未触发的文件逐字节不变；旧产物重跑即可。同时也删掉了流式
  路径中专为施加全局标量而存在的第二遍 pass。详见 `docs/vad-energy.md`。
- **ASR 固定单 worker，移除单文件分片设计**。`sharding.py`、`ResourceProfile.wt_instances`、
  `--wt-workers`（pipeline 与 vad-asr 两处）、interval ownership 标记与合并、shard partial
  一并删除；GPU profile 现在只决定人声分离实例数，metadata 不再记录 asr workers。
  实测 worker=3 相对 worker=1 在 wt 上仅 1.40×、fw-refine 上仅 1.20×，代价是显存 2.4→6.5 GB；
  换到 fw-refine 后 ASR 已非瓶颈（人声分离占语音段 72%）。回溯点：`dev` 的 `1fcc4e1`。
- **ASR 分组改为按合成后的音频长度规划**：组尾垫料（至多 0.7 秒原始音频 + `--gap` 秒静音）
  此前不计入分组长度，于是按 30 秒规划的组实际可达 31 秒、溢出编码窗口。11 个真实 clip 上
  超窗分组 102 → 56，总组数 388 → 405。**这会改变分组边界，因而改变 ASR 输出**（11 个 clip
  中 9 个分组不同）；旧产物不会自动失效，需要重跑才能得到新分组。`combined_group_duration()`
  语义不变（auto language 短组启发式仍按「说了多少话」判断）。
- `asr-pipeline` 新增开发用 `--asr-backend {wt,fw-refine}`（默认 `wt`，与 `vad-asr` 一致）。
  面向一般用户的文档不介绍该开关。
- ASR checkpoint fingerprint 新增 `asr_backend` 且**无默认值**：两个 backend 对同一段音频给出
  不同词级时间，中断后换 backend 续跑此前会静默复用另一侧的 partial，把两种输出缝进同一份产物。
- `faster-whisper` 与 `ctranslate2` 移出 `asr`，独立为 **`asr-refine`** extra 并精确钉版
  （1.2.1 / 4.8.1，均为当前最新且互相兼容）。fw-refine 不在 `asr-pipeline` 的可达路径上，
  普通用户不必装一份链了 CUDA 的 CTranslate2；同时也消除了 `asr` 与 desktop runtime lock
  之间「lock 缺这两个包」的静默不一致。fw-refine 继承 faster-whisper 内部实现并读取 CT2
  解码轨迹，小版本变动可能悄然改变输出；升级顺序固定为先 faster-whisper 后 CT2——CT2 的
  可选范围由 fw 声明的 `>=4.0,<5` 决定。

### 性能

- **`--vad-silero-assist` 从「比 VAD 本身还贵 4 倍」降到几乎免费**：60 分钟分离人声上
  该 opt-in 此前在 17 秒的 VAD 阶段之上再花 72 秒，现在 CPU 约 3 秒、CUDA 约 1 秒。
  silero 的逐帧 JIT 调用（每 32ms 一次，一小时 112,500 次派发）改为：帧间独立的
  STFT+encoder 整批计算，LSTMCell 的权重驱动全序列 `nn.LSTM`；概率再搭车在 VAD 自己的
  流式 block 上算，省掉第二遍解码与整段波形常驻（0.23 GB/小时）。与逐帧实现的最大概率
  差 1.4e-05，判据阈值零翻转。CUDA 路径显式关闭 TF32（开启会让 112,500 帧中 16 帧越过
  `CAP_SIL_THR`）。

### 修复

- `fw-refine` 在 CTranslate2 缺少该设备的矩阵后端时，给出指向构建要求的可读错误，而不是从库
  深处抛出的 `No SGEMM backend on CPU`。该情况无法从 `get_supported_compute_types()` 查出
  （只有 CUDA 后端的构建对 CPU 仍报 `float32`），只能在首次 encode 处补上下文。
- `fw-refine` 的词切分不再丢弃永远无法解析的 `U+FFFD` token。真实幻觉撞上解码上限时会在半个
  字符处截断，旧实现一直等待补全、循环结束时静默丢弃该 token，使词分组与 one-pass 解码轨迹错位并
  抛 `ValueError`；310 个真实生产窗口里有 6 次触发。同时补上 fw-refine 的退避链——任何失败改退到
  该后端自己的 teacher-force 对齐（`naive_approach` 是 whisper-timestamped 的选项，对 fw-refine 会
  原样重放同一次调用），两次都失败才丢弃该 group，不再终止整个 run。
- recall 临时组的 complement 切片现在继承源 interval 的 shard 归属。此前它新建裸 dict，
  分片合并时以「missing interval ownership」中止**整个 run**；触发条件是 workers ≥ 2 且
  某个 block 命中 recall（≥5s 未覆盖），与 ASR backend 无关。
- Desktop 的 API Key 设置改用现行 CLI provider pool 变量，Windows AI runtime
  lock 重新对齐 Torch/Torchaudio 2.8；新增跨 Desktop/CLI 契约测试，避免两侧再次漂移。
- Desktop 完成页只展示实际存在的产物并暴露 run metadata；WebView 重载会恢复仍在
  运行的任务，翻译模式明确提示 Gemini 媒体片段上传。
- Desktop 默认 LLM level 和后处理 profile 范围与生产 pipeline 对齐；新增独立
  `desktop/VERSION` 作为桌面发布版本的单一来源。

## [0.2.0] - 2026-07-30

### 新增

- ASR 新增单文件 Whisper Timestamped 分片、分组 checkpoint、运行时 metadata、
  GPU stage gate、stall watchdog 与资源用量记录；4/8/12/16GB profile 会在文件内部
  分配 separator 和 WT 并发。
- 字幕分句改为全局 DP，并新增分割点金标准、标注工具和系统化评测资料。
- LLM harness 新增可配置 API key pool、sticky retry 后的组合冷却，以及更完整的
  token budget、任务报告和搜索证据处理。
- 新增 `config.example.toml`、桌面 launcher 资源配置和 Windows token counter 更新。

### 变更

- 生产代码重组到 `asr_playground` namespace，明确 media、speech、subtitles 和
  workflows 边界；命令行入口和打包清单同步迁移。
- Batch 从文件级 ASR 并发改为单文件独占 profile、文件内分片并发，避免两层并发相乘。
- 桌面应用同步外观控制与 UI 刷新，并在构建 bootstrap 时优先使用 conda env-root Python。
- OpenCC 转简加载器与本地 token counter 改为跨调用复用；ASR 模型从共享 checkpoint
  直接构建 FP16 实例，降低重复加载开销。

### 修复

- 修复 wheel 漏装 `batch`、`gpu_stage_gate`、`run_metadata`、`segment_split` 和
  `wt_shard` 顶层模块，以及 license metadata 无法在声明的 setuptools 下限构建的问题；
  增加顶层源码与 packaging 清单一致性测试。
- ASR checkpoint schema 升至 v2；旧 partial 明确失效并从头重跑，sharded merge
  遇到缺失 interval ownership 的结果会显式报错，不再静默丢字幕。
- `segment_split` 对有文本但没有 word timestamps 的 segment 合成一条带来源标记的
  segment-span word；无法安全归一化时保留原输入，不再从全局 DP 输出中消失。
- Reference ingest 在迁移到统一 batch workflow 前，先与普通 batch 一样固定每任务
  `wt_workers=1`，避免文件级并发与 shard 并发相乘。
- Pipeline 的 LLM round 汇总和 task report 现在遵循显式 `task_artifact_dir`；batch
  同一 logical run 的后续 pass 会继承已执行 stage，stage metadata 不再混出
  `reused` 加旧 `elapsed_sec` 的矛盾记录。
- 字幕渲染在时间轴后处理前修复 cue 重叠，保证不丢文字。
- 修复 separator dotted 临时文件识别、Gemini key 全部跳过时的错误类型，以及
  flash gap/end pad、RPM 失败计数等重试与边界问题。

### 变更

- `--wt-workers` 明确为开发/不安全 benchmark 覆盖参数；生产调度继续由 GPU profile
  和 batch runner 决定。
- 补齐 run metadata、WT sharding、segment split 和 packaging 测试的 pytest 域 marker。

## [0.1.1] - 2026-07-27

Prompt version: `zh-subtitle-correction-csv-v65`。

### 新增

- 新增 Windows Desktop 应用，提供任务管理、资源管理、运行时设置与日志查看等桌面工作流。
- Pipeline 新增 `--name`；视频任务默认使用高质量多模态模型，并在不可用时自动降级。

### 变更

- Prompt 升级到 `zh-subtitle-correction-csv-v65`：删除 CapableA、BasicC 及 BasicC 的 JSONL 输出支线，现行变体为 capableB/C + basicA/B。
- 单窗口 query、correction 与 fast round 1 的目标字幕序号每窗重置为 `1..N`，只读前文按时间顺序编号为 `1-M..0`；harness 校验后映射回稳定源序号，oneshot、replay、benchmark 与任务反馈同步采用该契约。
- 改进搜索证据包和研究阶段，减少冗余上下文并提高可用证据密度。
- 完善 ASR 语言历史、分组 checkpoint 与 Whisper fallback，扩展救援阶梯和尾段回交策略。
- ASR 依赖更新至 PyTorch 2.8 系列。

### 修复

- Pipeline 与最终 SRT 输出改为原子写入，避免中断时留下不完整文件。
- 修复 clip 预取线程安全问题，并补充 Pipeline 失败诊断信息。

## [0.1.0] - 2026-07-23

首个 beta 版本。Prompt version: `zh-subtitle-correction-csv-v63`。

### 功能

- 本地长音频转字幕完整流水线：人声分离 → VAD + Whisper ASR 对齐 → ASR 稳定化 → SRT 输出
- 实验性 LLM 纠错与翻译后处理（Gemini），支持 6 档 preset（route × level）、fast 模式、多轮搜索调查
- 批量运行（三阶段流水线并行，单项失败隔离，断点续跑）
- 本地知识库（自动采集/统一更新/精修对照）
- URL 输入支持（yt-dlp 下载 + 自动 ID 映射）
- 词级字幕输出
- SRT 后处理（繁简转换、短轴延长、标点清理）
- GPU 显存档位（8/12/16GB）自适应
- 流式 VAD 与流式 ASR 对齐（内存上界恒定，支持任意时长音频）
- LLM session resume + 纠错窗口中途 resume
- Prompt 变体系统（capableA/B/C + basicA/B/C）

### 依赖

- Python >= 3.12
- 核心流水线：torch~=2.9.0, torchaudio~=2.9.0, numpy, soundfile, whisper-timestamped, onnxruntime, audio-separator, numba
- LLM harness：httpx, yt-dlp, opencc-python-reimplemented
- 开发：pytest, pytest-xdist, sudachipy, sudachidict_core

### 已知限制

- LLM 纠错翻译层为实验性功能，默认不包含在生产 stage 中（需 `--stage translated-srt` 或 `final-srt` 显式启用）
- 仅提供 Windows 预编译 token counter 二进制；Linux/macOS 回退到免费 countTokens API 或启发式
- 无 GPU 时回退 CPU（速度显著下降）
