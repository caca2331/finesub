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

装机层原语（`AppPaths` 目录布局、校验下载、安全解压、uv 托管 Python 运行环境
`RuntimeEnvironment`、跨进程锁 `locks.py`、链接安全与崩溃安全的目录操作 `fsops.py`）
不在 `desktop/` 下，而在共享包
`src/finesub_bootstrap/`——它同时服务桌面端与 CLI 壳，禁止反向依赖 `desktop`。

目录分三个根，按数据行为而不是按谁写的划分：**数据根**
`%LOCALAPPDATA%\FineSub`（`user-data`：设置、API Key、知识库、任务历史；三种安装形式共用同一份，
所以一个用户只有一个知识库）、**安装根**（应用自身与 `runtime/`，版本绑定、从不共享）、
**大数据根**（`models`/`cache`/`tasks`，默认等于安装根，可被 `finesub relocate` 指到别处并被
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

yt-dlp 走 PYTHONPATH 是因为管线 `import yt_dlp` 用 Python API，不是调可执行文件。
也正因如此它**无法复用系统安装**：管线跑在托管运行时的解释器里，看不见用户的
site-packages。它的强制依赖为零，裸解压 wheel 即可 import（不装 `[default]` extra
的代价是部分站点降级：br 压缩、部分直播/加密流、YouTube 的 JS challenge）。

git 与 yt-dlp 在资源面板里**列出但标记为可选**（`ResourceStatus.optional`）：
不列会让"缺 git"的报错把用户指向一个找不到 git 的面板，而计入必需又会让一台完好的
机器显示"2/4 就绪"。就绪计数与"所需空间"只统计必需项。

git 与 yt-dlp **不进 `task_ready()`** —— 按请求实际用到的能力校验
（`finesub_bootstrap/capabilities.py`，桌面与 CLI 共用同一套规则）。

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
`runtime\python -m asr_playground.pipeline`，那正是知识库曾经落进 `app/versions/<版本>` 的原因）。
两个 exe 都是 `--windowed`、拿不到控制台，所以入口走托管解释器而不是再打一个 console exe。
它 `can_provision=False`：自己就跑在托管运行时上，装不了也换不了这个运行时，缺资源时指回应用内的
资源面板，而不是给一条走不通的 `finesub setup`。

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
