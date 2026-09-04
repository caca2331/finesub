# 剥离桌面端（0.5.0）实施计划

状态：**阶段 A、锚点 `0.5.0pre` 与阶段 B 均已完成**（2026-09-03；起草、三轮复审、A1–A4、
发布锚点与 B0–B7 都落在同一天，见 §9）。**下一步是阶段 C**（§6）；owner 定的是发版前
另找人审计一遍，所以 B 合入 `dev` 后先停在那里。
0.5.0 把 `desktop/` 从本仓移出；剥离之前的那一份
以 `0.5.0pre` 的形态留在公开 `main` 上，供桌面端维护者据此迁移到自己的仓库。
`0.5.0pre` **不是正式发版**（一个带标 commit 即可，不发 PyPI、不建六资产），
0.5.0 本身照常走 `agent-tasks/release/SKILL.md`。

本文是取舍依据与执行顺序。现行行为的真相源始终是代码与 owner 文档，不是这份计划。

---

## 1. 盘点结论：剥离动的不是代码，是构建面

`desktop/` **之外**有 115 个被跟踪文件提到它（不含本文；全仓连 `desktop/` 自身一起数是
219，那个数字对本计划没有意义）。但**没有任何 Python 文件 `import desktop`**——
`test/test_paths.py:401` 的守卫一直钉着这个方向。`src/finesub_bootstrap/` 的 30 个模块
逐个查过使用面，**没有一个会因为剥离而变成不可达**，所以共享装机层整包留下。

⚠ 但「不可达」不等于「还有生产使用者」：`config_file.py` 的 `update_config_file`
**唯一的生产调用者是 `desktop/backend/settings/store.py`**，CLI 侧没有写入者。剥离后它
只剩 `test/test_config_file.py` 养着，而 `CLAUDE.md` 还把它写成「保留注释的写入器」。
这不是删不删的问题，是 CLI 该不该长出一个写配置的命令——列为 §8 的未决项，不在本计划里裁。

真正的耦合只有三类：

1. **`desktop/` 里住着 CLI 也在用的四份资产**（§3，阶段 A 解决）；
2. **构建与发布基础设施**：两条 workflow、两个 ps1、`pyproject.toml` 的三处（§5）；
3. **守卫的扫描面**：十余处按路径写死 `desktop/`，删目录即红（§5 B2）。

## 2. 为什么阶段 A 必须在 `0.5.0pre` 之前

`0.5.0pre` 是交给下游维护者的锚点。如果四份共享资产还留在 `desktop/` 里，
那个快照就把「谁拥有托管运行时的锁」这个问题也一并交出去了——维护者会 fork 到一份
**我们仍在用**的 pylock，两边此后各改各的，而 CLI 这边还得在剥离提交里再搬一次。

阶段 A 先做，锚点交出去的边界就已经是对的：`desktop/` 里剩下的每一个文件都确实是桌面的。

⚠ **这对下游是一处行为改变，要写进 `0.5.0pre` 的 tag/release 说明**：从这一版起
`desktop/` 不再自带 pylock 与 runtime-manifest，桌面端要么消费
`finesub_bootstrap` 里的那一份，要么在自己的仓库里另存一份副本。

---

## 3. 阶段 A：把共享资产搬出 `desktop/`（`0.5.0pre` 之前）

### A0 现状：四份资产与它们的消费者

| 资产 | 仓外消费者 |
| --- | --- |
| `desktop/VERSION` | `cli/scripts/build-wheel.ps1:12`（wheel 版本默认值）、`.github/workflows/release.yml:155,157`、`.github/workflows/desktop-ci.yml:65`、根 `pyproject.toml:7`（见 A1：那里是**另一份**静态副本） |
| `desktop/runtime/pylock.win-py312.toml` | `build-wheel.ps1:85`、`src/finesub_bootstrap/shell.py:1881`、`test/bootstrap/test_runtime_environment.py:888`、`desktop/scripts/tests/test_desktop_dependencies.py:164`（**在根 `testpaths` 里**） |
| `desktop/runtime/pylock.win-py312.cn.toml` | `build-wheel.ps1:92`；生成器是 `desktop/scripts/make_cn_lock.py`（见 A4） |
| `desktop/resources/runtime-manifest.json` | `build-wheel.ps1:96`、`shell.py:1863`、`scripts/check-pinned-urls.ps1:53`、`test/bootstrap/test_resource_manager.py:205`、`test_desktop_dependencies.py:217`（**在根 `testpaths` 里**）、`cli/tests/test_cli_main.py:340`（夹具从这里读真 manifest 再 monkeypatch `_VENDOR`） |

⚠ 表里带「在根 `testpaths` 里」的两处最容易漏：`test_desktop_dependencies.py` 虽然住在
`desktop/scripts/tests/`，却被根 `pyproject.toml:184` 点名收集——**阶段 A 落地当天
`pytest -q` 就会红**，而不是等桌面 CI。

桌面侧的消费者（同一提交里一起改，因为 A 落地时 `desktop/` 还在）：
`desktop/backend/launcher/main.py`、`desktop/scripts/build-installer.ps1`、
`build-bootstrap.ps1`、`make_cn_lock.py`、`setup-dev.ps1`，以及 8 份桌面测试
（`test_cn_lock.py`、`test_shell.py`、`test_desktop_resources.py`、`test_launcher_paths.py`、
`test_installer_definition.py`、`test_package_bootstrap.py`、`test_build_bootstrap.py`、
`test_dev_script_security.py`）。

还有一批**按路径提到这几份资产的注释与文档**，同批改掉，否则读者会被指去一个空目录：
`cli/README.md:140`、`cli/pyproject.toml:30`、根 `pyproject.toml:95,102,176`、
`docs/download-routes.md:60,71`、`docs/ct2-distribution.md:79`、
`tools/tokcount/README.md:109`、`agent-tasks/release/SKILL.md:22,38,66`。

### A1 版本号 → 仓库根 `VERSION`

**新家：`VERSION`（仓库根，纯文本一行）；根 `pyproject.toml` 同时改成 `dynamic` 读它。**

⚠ **版本号现在就是两份，不是一份**：根 `pyproject.toml:7` 有静态 `version = "0.4.2"`，
`desktop/VERSION` 有同一个值，靠 `test_desktop_dependencies.py:248` 那条
「一个版本号」钉着相等。只把 `desktop/VERSION` 搬到根 `VERSION` 会变成**三份**——
所以 A1 必须连根 `pyproject.toml` 一起改：`dynamic = ["version"]` +
`[tool.setuptools.dynamic] version = {file = "VERSION"}`，和 `cli/pyproject.toml:62`
同一种写法。那条测试随之从「两份相等」改写成「只有一份」。

理由：

- `cli/pyproject.toml:62` 已经是 `version = {file = "VERSION"}`，读的是**打包暂存区根部**
  的 `VERSION`，由 `build-wheel.ps1` 用 `WriteAllText` 现写。也就是说「版本号写在一个叫
  `VERSION` 的文件里」已经是既定形态，`desktop/VERSION` 只是那个值的**默认来源**。
  搬到仓库根等于让来源与产物同名同形，`build-wheel.ps1` 只改一行路径。
- 剥离之后这个仓库只产出一个带版本的东西（PyPI 上的 `finesub` wheel），而
  `CHANGELOG.md` 与 git tag 本来就是仓库级的。版本号是**这个仓库这一次快照**的版本，
  不是某个子目录的属性。
- 不选 `cli/VERSION`：`build-wheel.ps1` 只把 `cli/` 下的 `pyproject.toml`、`MANIFEST.in`、
  `README.md` 三个文件拷进暂存区，一个被跟踪的 `cli/VERSION` 不会被拷、却和生成出来的
  同名文件长得一模一样——两份同名文件不同语义，是给未来的自己埋雷。

改动：`build-wheel.ps1:12`、`release.yml:155,157`、`desktop-ci.yml:65`、
根 `pyproject.toml:7`、`cli/pyproject.toml:7` 的注释、`README_DEV.md:73`、
`cli/README.md:133`；桌面侧四处（`build-bootstrap.ps1:17`、`build-installer.ps1:14`、
`test_build_bootstrap.py:67,111`、`test_desktop_dependencies.py:248,258`）。

### A2 三份运行时资产 → `src/finesub_bootstrap/`

**新家（平铺，不新建子目录）：**

```text
src/finesub_bootstrap/runtime-manifest.json
src/finesub_bootstrap/pylock.win-py312.toml
src/finesub_bootstrap/pylock.win-py312.cn.toml
```

理由：

- 该目录**已经**平铺住着两份同类数据文件：`download-sources.json`（配 `download_sources.py`）
  与 `model-manifest.json`（配 `model_manifest.py`）。第三、四、五份放同一层是既有惯例，
  不是新发明。`runtime-manifest.json` 配 `resources.py`/`system_tools.py`，
  两份 pylock 配 `environment.py` 的 `RuntimeEnvironment`。
- **打包路径自动通了**：`build-wheel.ps1` 的 `Copy-PythonTree` 拷贝除 `.pyc` 外的一切文件，
  `cli/MANIFEST.in` 又是 `graft src/finesub_cli/_vendor`。所以搬进去之后，
  `build-wheel.ps1` 里那**三条 `Copy-Item` 可以整段删掉**——资产随包一起进 wheel。
  这是本次搬迁唯一能把构建脚本改短的选择。
- 不选仓库根 `runtime/`：`.gitignore:61` 就忽略着 `/runtime/`（那是桌面 launcher 从
  checkout 跑时的状态目录）。搬进去会被**静默忽略**，且这个坑不会有任何报错。
- 不选 `cli/runtime/`：`src/finesub` 与 `finesub_bootstrap` 在源码 checkout 里也要读它们
  （`shell.py` 的 `package_shell`），而 `src/` 反过来依赖 `cli/` 是新的方向依赖。

考虑过但未采纳的备选：`src/finesub_bootstrap/runtime/` 子目录，三份同住。
好处是「这三份是一套、一起进出」这层语义有落点；代价是与既有两份 JSON 的平铺惯例不一致，
且要多一条 package-data glob。选平铺，因为惯例一致的收益更实在（复审已同意，§8 第 2 条）。

### A3 顺带：让包自己解析这三份文件

`cli/src/finesub_cli/main.py:98,112` 现在**按路径**指名 `_VENDOR / "runtime-manifest.json"`
与 `_VENDOR / "pylock.win-py312.toml"`；`shell.py:1863,1881` 同理指名 `desktop/...`。
搬迁之后两处都应改成由 `finesub_bootstrap` 相对 `__file__` 解析，参数保留但带默认值。

这不是顺手重构：`CHANGELOG.md:1320` 记着这几份文件「被外部按路径引用」正是它们难搬的原因，
而搬完还留着按路径引用，就等于把同一个坑挪了个地方。区域锁「在标准锁旁边找」的规则不变。

⚠ 连带要重写的夹具：`cli/tests/test_cli_main.py:340` 的 `_vendored` 现在**从
`desktop/resources/` 读真 manifest**、写进临时 `_vendor` 再 monkeypatch `cli._VENDOR`。
解析方式一改，这个夹具的两条假设（资产在 `_vendor` 根、路径由调用方给）都不成立。

### A4 cn 锁的生成器跟着锁走

`desktop/scripts/make_cn_lock.py` 是 `pylock.win-py312.cn.toml` 的**生成器**——
`download_sources.py` 的 `torchMirror` 注释写明它「在构建时被 `make_cn_lock.py` 读」。
锁归了 `finesub_bootstrap`，生成它的脚本却留在 `desktop/` 等着被删，是把一份留下来的
产物和它唯一的再生手段拆开。

**落点：`scripts/`，并给它加一个 `__init__.py`。**

⚠ 这不是纯搬文件：`desktop/scripts/` **是个包**（有 `__init__.py`），所以
`make_cn_lock.py` 的 docstring 里写的用法是 `python -m desktop.scripts.make_cn_lock`，
`test_cn_lock.py` 也是 `from desktop.scripts.make_cn_lock import ...`。而 `scripts/`
今天只有两份 `.ps1`、**没有任何 Python、不是包**。所以搬过去必须二选一：

- **把 `scripts/` 变成包**（加 `__init__.py`，用法改 `python -m scripts.make_cn_lock`）——**选这个**；
- 或让测试改成按路径 `import`——多一层 `importlib` 胶水，只为省一个空文件。

不选 `tools/`：那里本来就是 Python 且有 README 惯例，但 `tools/` 的约定是
「只按需维护、测试不进默认套件」（`pyproject.toml` 的 `testpaths` 不含它），
与本步「让这 15 条进根套件」直接相抵触。

同批要改的三处：`make_cn_lock.py` 的 docstring 用法行（`:15`）、
`test_cn_lock.py` 的 import、以及 `docs/download-routes.md:71` 里那句
「由 `desktop/scripts/make_cn_lock.py` 生成」。

它的 15 条测试 `desktop/backend/tests/test_cn_lock.py` 是**纯逻辑、无平台依赖**
（只在字符串里出现 `sys_platform`），一并搬进 `test/bootstrap/`，从此进根套件——
别忘了 B0 那条标记表登记。

### A5 阶段 A 的验收

```powershell
python -m compileall -q src test
python -m pytest -q
python -m pytest -q desktop          # 根套件收集不到，必跑
python -m pytest -q cli/tests
.\cli\scripts\build-wheel.ps1        # wheel 里三份资产的路径变了，清单断言要同步
.\scripts\check-pinned-urls.ps1
```

⚠ `scripts/check-pinned-urls.ps1:51-53` 的三条路径要一起改，否则它会安静地扫一个空集合
（脚本对不存在的路径不报错——这正是 `docs/plans/refactor-followups.md` 第五条教训
「守卫的扫描面本身就是守卫的一部分」的同形复发）。

⚠ 根 `pyproject.toml` 的 `[tool.setuptools.package-data]` 里**现在就没有**
`finesub_bootstrap` 的条目（那两份既有 JSON 也不在）。本仓只做 `pip install -e .`，
所以一直没暴露。要不要顺手补一条属于**本计划之外**的独立判断，不在这里夹带。

---

## 4. 锚点 `0.5.0pre`

> **已执行（2026-09-03）**：`main` = `3d5993c1`，tag `0.5.0pre`，GitHub prerelease。
> 过程与途中抓到的东西见 §9。

阶段 A 全绿之后，按 `scripts/publish-main.ps1` 把 `dev` 快照推上去，打 `0.5.0pre` 标。
**不走 `release` skill 的完整流程**：不发 PyPI、不建六资产、不签更新 manifest。
`finesub_bootstrap/update_check.py` 只认正式版，`0.5.0pre` 天然不会被推荐给用户。

⚠ **「不走 release 流程」不等于「可以直接推 `dev`」**（owner 2026-09-03）：这个锚点是
**公开** remote 上的东西，本地 only 的被跟踪文件一样要剥掉。剥离由
`publish-main.ps1` 的 `$PrivatePaths` 完成，今天是六项——`.claude`、`docs/archive`、
`docs/report`、`agent-tasks/release`、`agent-tasks/desktop-portable`、
`agent-tasks/run-audit/evals`。所以**必须走这个脚本**（它同时会把快照推 `ci-gate`、
CI 绿了才快进 `main`），而不是 `git push origin dev:main` 之类的手工捷径。

顺带：`agent-tasks/desktop-portable` 此刻还在 `$PrivatePaths` 里，所以那份 skill
**本来就不在公开快照里**——`0.5.0pre` 交给下游的树里没有它，桌面维护者要它得另外给。
B2 删这一项是在剥离之后（那时它连本地都没了），不影响本锚点。

release 说明里写清三件事：

1. 这是给桌面端维护者的迁移锚点，不是可安装的版本；
2. §2 那条「`desktop/` 不再自带 pylock 与 runtime-manifest」；
3. **快照里的两条 workflow 不能直接拿走**：`desktop-ci.yml` 与 `release.yml` 此刻仍是
   桌面与 CLI 合体的，会去找 `cli/`、根 `pyproject.toml`、`test/` —— 下游 fork 之后
   得自己拆出桌面那半边。

---

## 5. 阶段 B：删除 `desktop/`

> **已执行（2026-09-03）**：分支 `refactor/desktop-split-phase-b`。过程、四处偏离本节的地方
> 与一条还没兑现的验收见 §9「阶段 B 实施记录」。下文是执行前的计划，按原样保留。

### B0 先抢救 `desktop/backend/tests` 里的共享层测试

⚠ **这一步必须在 B1 之前，否则 `git rm -r desktop` 会连同 78 条测共享层的用例一起删掉。**
`docs/testing.md:10-13` 与 `finesub_bootstrap/__init__.py:15` 都写着「留在
`desktop/backend/tests` 的是只有 Windows 能真跑的那些」——**对其中两份不成立**：

| 文件 | 条数 | 实情 |
| --- | --- | --- |
| `test_shell.py` | 51 | 模块头只 import `finesub_bootstrap` 的 `shell`/`environment`/`paths`/`resources`/`locks`/`system_tools`，**零平台标记**。与 `test/bootstrap/test_shell_{activity,commands,first_run}.py` 无重名、不重叠 |
| `test_model_fetch.py` | 27 | 模块头只 import `finesub_bootstrap` 的 `download_routes`/`model_fetch`/`http_client`/`model_manifest`，**零平台标记**。根套件**没有**对应文件（`test_model_caches` / `test_model_ensure` 是别的模块） |
| `test_fsops.py` | 4 | 真 Windows-only：docstring 明说只留 junction 语义那半边，平台无关的已在 `test/bootstrap/test_fsops.py` |

⚠ **「模块头只 import 共享层」不等于整份都能原样搬**：78 条里有 **6 条在函数体内**
import `desktop`，`ast.walk` 才看得见（`test/test_import_boundaries.py:43` 的
`_every_import` 正是为这种漏网写的）。逐条处置：

| 位置 | 测什么 | 怎么处置 |
| --- | --- | --- |
| `test_shell.py:333,431,463` | 函数内 `from desktop.backend.common.models import TaskRequest`——CLI 写的历史能被桌面的 `TaskRequest` 回放 | **改写成直接断言记录下来的字典，不要删。**`test/bootstrap/test_task_output.py` 覆盖不到：它的 docstring 说自己只管命名与放置规则，桌面那半边在 `test_shared_index_contract.py`（随 `desktop/` 走）。这三条测的是 `_recorded_request` 把**非默认**开关如实记下来的契约，docstring 明说它被 `--llm-difficulty efficiency` 记成 `high`、回放成 `quality` 咬过一次——去掉 `TaskRequest` 只是去掉回放那一端，被咬的那一端还在 |
| `test_shell.py:1196,1224` | `package_shell` | 随 B5 一起删 |
| `test_model_fetch.py:546` | 函数内 `from desktop.backend.resources.model_prefetch import ModelPrefetchFailed`——跨进程只传回消息时的判定 | 用普通异常构造同样的消息即可保留 |

所以账是 **72 条直接搬 + 6 条要处理**，不是「78 条原样搬」。

前两份搬进 `test/bootstrap/`；`test_fsops.py` 连同其余真 Windows-only 用例，去处见 B3 的新 lane。

⚠ **搬完要登记标记表**：`test/conftest.py:22-45` 的 `_PIPELINE_FILES` 逐个文件列名，
而 `test/test_packaging.py:87` 的 `test_the_domain_markers_cover_every_test_file` 要求
根套件收集到的每个文件**恰好有一个域标记**——新文件不登记就红，报错是
「collected by the root suite but declares no domain marker」。`test_cn_lock.py`（A4 搬的）
同理。

搬完先跑一遍 `python -m pytest -q`，确认这 72 条在根套件里是真跑而不是 skip，再做 B1。

（顺带已核实：根 `test/conftest.py` 的 autouse 夹具已覆盖 `desktop/backend/tests/conftest.py`
做的三项环境隔离，搬过来不缺夹具。）

### B1 删目录

`git rm -r desktop`（192 个被跟踪文件）。`.claude/launch.json` 三个配置全是
`desktop/frontend` 的 npm dev server，一并删。

### B2 会变红的守卫与测试

| 位置 | 处理 |
| --- | --- |
| `pyproject.toml:184` `testpaths` | 去掉 `desktop/scripts/tests/test_desktop_dependencies.py` |
| `desktop/scripts/tests/test_desktop_dependencies.py` | 9 条里 **4 条是 CLI 契约**，迁进 `test/`：uv 钉版一致、AI 运行时锁与 extras 一致、一个版本号、CLI 只暴露 launcher 入口。其余 5 条（trust anchor、两个 packager、release 增量默认）随桌面走 |
| `test/conftest.py:184` | `_marker_key` 里为那个域外文件写的特判，随 `testpaths` 一起删 |
| `test/test_packaging.py:148` | 读 `desktop/backend/updates/installer.py` 校验更新契约——桌面更新器没了，这一半删；wheel 清单那一半留 |
| `test/test_packaging.py:182` | 文档清单去掉两份 desktop README |
| `test/test_paths.py:401` | 「不得 import desktop」变成永真。删，或改写成「vendored 树自洽」 |
| `test/test_import_boundaries.py:40` | `IMPORTING_TREES` 去掉 `"desktop"`，**同一行加上 `"scripts"`**——A4 之后 `make_cn_lock.py` 是那里唯一的 Python，不加就成了守卫扫描面之外的一棵树（正是 §9 引的那条教训）。另两个源码守卫不受影响：`test_doc_links` 的 `SOURCE_ROOTS` 已含 `scripts/`，而 `make_cn_lock.py` 不碰 `subprocess` 与 `rmtree` |
| `test/test_import_boundaries.py:250` | rmtree 守卫的 `roots` 去掉 `desktop/backend` |
| `test/test_subprocess_text_encoding.py:38` | 扫描面去掉 `desktop/backend` |
| `test/test_doc_links.py:301` | `SOURCE_ROOTS` 去掉 `"desktop/"` |
| `test/test_doc_links.py:421` | ⚠ 它**也**校验 `docs/archive/` 与 `docs/report/` 内部的 md 链接。`docs/archive/cli-bootstrap-logging-download-plan.md` 两处 `../../desktop/README_DEV.md`、`docs/manual/troubleshooting.md:74` 的 `../../desktop/README.md` 会全部悬空 |
| `test/bootstrap/` 五份 | 夹具按 `desktop/runtime/pylock...` 造目录：`test_runtime_environment.py:30,34,47,114,139,888`、`test_runtime_regional_lock.py:20,36`、`test_resource_manager.py:37,205`、`test_shell_commands.py:99`。⚠ **其中大部分在阶段 A 就要改**，这里只剩残余 |
| `scripts/publish-main.ps1:29` | `$RequiredWorkflows` 去掉 `"Desktop CI"` |
| `scripts/publish-main.ps1:40` | `$PrivatePaths` 去掉 `agent-tasks/desktop-portable`，否则每次发布都告警「这条保护不了任何东西」 |

### B3 CI 与发布

⚠ **删掉 `desktop-ci.yml` 会让整条 Windows lane 消失，不只是那个 py310 job。**
`ci.yml` 只有一个 `test` job，`runs-on: ubuntu-latest`。`desktop-ci.yml` 的两个 job
都是 `windows-latest`，而**属于 CLI 的东西全在里面**：

| 现在跑在哪 | 内容 | 删了会怎样 |
| --- | --- | --- |
| `desktop-ci.yml` 的 `thin-cli-py310` | 3.10 建 wheel、装、无托管运行时时跑 `finesub agent-clean` | 薄 CLI 的轻依赖 / 不 provisioning 契约裸奔（`docs/testing.md:238` 把它写成守卫） |
| `desktop` job：`pytest -q -n 0 cli/tests` | CLI 壳的全部测试 | **`cli/tests` 从此没有任何 CI 执行** |
| `desktop` job：`pytest -q -n 0 test/test_secrets.py` | 真 DPAPI 的信封加密用例，`skipif(os.name != "nt")` | 保护用户磁盘上 API key 的那条用例**在 CI 里一次也跑不到**（workflow 里的注释自己写着这句） |
| `desktop` job：`build-wheel.ps1` (3.12) | wheel 构建本身 | 发布用的构建脚本不再被验证 |

**所以 B3 不是「迁一个 job」，是「新建一条 Windows lane」**：在 `ci.yml` 里加一个
`windows-latest` job，承接上表前四行，外加 B0 第三行那些真 Windows-only 的 `fsops`
junction 用例。做完再删 `desktop-ci.yml`。
- **`.github/workflows/release.yml`** 大改：约 20 处桌面步骤（`.venv-desktop`、
  Next.js 静态导出、`build-bootstrap.ps1`、Inno Setup 安装器、Ed25519 签名、
  `verify_release_key`、`update-manifest.json`）全部移除，只留 PyPI wheel
  与 CT2 wheel 资产那条线。
- **`scripts/check-pinned-urls.ps1`**：阶段 A 已改过路径，这里再核一遍没有残留。

### B4 `pyproject.toml`

- 删 `[desktop]` extra（pywebview / pystray / Pillow / cryptography）。
- ⚠ **`[desktop-worker]` 不能无脑删**：它里面那条 patched CT2 wheel 的 direct reference
  是喂给 pylock 的，而 pylock 留下来给 CLI 用。改名（如 `runtime-worker`）并把
  第 91-102 行那段解释三个环境的注释重写；`docs/ct2-distribution.md:111` 同步。
  改名时**同步 `test_windows_ai_runtime_lock_matches_the_pipeline_extras`**——它在
  `test_desktop_dependencies.py:174,181,201` 三处**按字面量**写着 `desktop-worker`
  （`181` 那处是 `for extra in ("asr", "harness", "desktop-worker")`，改漏了就是静默少校验一个 extra）。
- `testpaths` 见 B2；`[project.scripts]` 那段「为什么故意没有」的注释里提到桌面，措辞更新。

### B5 死代码

- **`src/finesub_bootstrap/shell.py:1849` `package_shell()`**：唯一调用者是
  `desktop/assets/package-cli/finesub.py`，桌面走了就零引用。删。
  - **`application_source`（`shell.py:1967`）一起删**：全仓唯一调用者就是
    `package_shell:1860`（桌面自己那个 `_application_source` 与
    `resolve_application_source` 是另外两个函数，随 `desktop/` 走）。
  - **`load_app_paths` 留**：除定义处外，`desktop/` 之外有 4 处在用
    （`cli/src/finesub_cli/main.py`、`src/finesub/paths.py`、`shell.py` 自己、
    `test/bootstrap/test_paths.py`）。
  - `desktop/backend/tests/test_shell.py` 里针对 `package_shell` 的用例（B0 搬过来的
    那批）随之删，别搬进根套件又立刻变红。
- **`src/finesub_bootstrap/locks.py:195`** 的前端标签 `{"desktop": "桌面端", ...}`：
  lease 协议本身要留（`docs/cross-frontend-lease.md` 的机制对任何第二前端都成立），
  但这张表可以只留 `cli` 与兜底。

### B6 文档

必改（公开树）：`README.md`（88-90 行整节，另 61 / 78 / 161 / 185 行的顺带提及）、
`README_DEV.md:69-73`、`CLAUDE.md`（架构表的 `desktop/` 行、`finesub_bootstrap/` 行的
owner 文档、Docs index、末尾那张表）、`docs/README.md:12,56`、
`cli/README.md:133,140`、`docs/manual/troubleshooting.md:74`、`docs/manual/resources.md`、
`docs/testing.md`（10-13 行、命令表 38-39 行、结构守卫 238 行）、
`docs/cross-frontend-lease.md`（整份前提是「两个前端」）、`docs/reporting.md`、
`docs/download-routes.md`、`docs/ct2-distribution.md`、`config.example.toml:5`、
`CHANGELOG.md` 加一条。

⚠ **按 `CLAUDE.md` 的 Archive extraction 规则，删 `desktop/README_DEV.md` 之前要先捞内容**。
至少三处它是**唯一落点**：

1. 「哪些产物是记录、哪些可删、谁来删」（含**故意不删**的两类：URL 输入下载的源媒体、
   `-annotated.csv` / `-corrected.srt`）——`CLAUDE.md` 末尾那张表就指着它，
   实现在 `finesub_bootstrap/artifacts.py`。
2. pylock 的维护流程（怎么重建、`make_cn_lock.py` 的角色）——阶段 A 之后锁与生成器
   都归我们（A2、A4），这份流程必须跟着它们走。
3. 「`.ps1` 用连字符（入口）/ `.py` 用下划线（可 import 模块）」的命名约定。

去处：1 与 3 并进 `README_DEV.md`；2 并进 `docs/ct2-distribution.md`——它已经在讲
patched CT2 wheel 怎么进两份 pylock（`:79`、`:111`），锁的重建流程落在同一份文档里，
才不会出现「wheel 怎么换」和「锁怎么重建」分家的第二次。

### B7 agent-tasks

- **`agent-tasks/desktop-portable/`** 整个删（含 `scripts/rebuild-portable.ps1`），
  同步 `agent-tasks/README.md` 与 `docs/manual/agent-tasks.md`。
- **`agent-tasks/release/SKILL.md`** 大改（36 处提及）：六资产、签名、增量更新演练、
  `desktop/VERSION` 盖章全部重写为「一个 PyPI wheel + CT2 wheel 资产」。

### B8 阶段 B 的验收

```powershell
python -m compileall -q src test
python -m pytest -q                  # B0 搬进来的 78 条要在这里真跑
python -m pytest -q cli/tests
.\cli\scripts\build-wheel.ps1
git grep -in "desktop" -- . ":!CHANGELOG.md" ":!docs/archive/" ":!docs/report/"
```

外加一条只有 CI 能答的：**B3 的新 Windows job 必须绿过一次**，且它的日志里能看到
`cli/tests` 与 `test/test_secrets.py` 的 DPAPI 用例是 **passed 而不是 skipped**
（这正是它们此前唯一的执行现场）。

最后那条 grep 的期望不是零——`locks.py` 的兼容读取、`cross-frontend-lease.md` 里
「曾经有过第二个前端」这类历史陈述都可以留——但每一条都要看过并有理由。

---

## 6. 阶段 C：0.5.0 正式发版

照 `agent-tasks/release/SKILL.md`（B7 已重写过的那版）走。⚠ `memory` 里
`release-0.4.0-status` / `update-chain-facts` 记着 0.4.0 漏发 CT2 wheel 的教训，
仍然适用。

---

## 7. 明确不做

- **不把 `desktop/` 的历史迁进新仓库**。锚点 `0.5.0pre` 是个可 clone 的快照，
  下游要历史自己从公开 `main` 取；本仓的 `dev` 全量历史不推公开 remote。
- **不删 `finesub_bootstrap` 的任何模块**。逐模块查过，剥离不会让任何一个变成不可达
  （`config_file.py` 失去生产调用者是另一回事，见 §1 的 ⚠ 与 §8 第 5 条）。
- **不动 lease 机制**。`docs/cross-frontend-lease.md` 的协议对任何第二前端成立，
  少一个前端不是删机制的理由。
- **不顺手补 `finesub_bootstrap` 的 package-data**（见 A5 的第二条 ⚠）。
- **不在本计划里裁 `update_config_file` 的存废**（§8 第 5 条）。

---

## 8. 未决与已定

前四条经 2026-09-03 复审（记录见 §9）：1–4 已定；第 5 条是复审新提出的，owner 2026-09-04 裁定。

1. **版本号落仓库根 `VERSION`**（A1）。**已定**——复审同意不选 `cli/VERSION`
   （撞名是真坑），并指出必须连根 `pyproject.toml` 一起改成 `dynamic`，否则
   「一个版本号」根本没做到（版本号现在就是两份）。A1 已按此重写。
2. **三份运行时资产平铺进 `src/finesub_bootstrap/`**（A2）。**已定**——复审逐条核过
   `Copy-PythonTree` 只跳 `__pycache__`/`tests`/点目录、`MANIFEST.in` 的 graft
   覆盖整个 `_vendor`，确认三条 `Copy-Item` 可删；两份 pylock 各 58KB，进包不算负担。
3. **阶段 A 整体先于 `0.5.0pre`**（§2，owner 已定）。**已定**，并补上了 §4 第 3 条
   （快照里的两条 workflow 是合体的，下游不能直接用）。
4. **`desktop/README_DEV.md` 三段内容的去处**（B6）。**已定**：1 与 3 进
   `README_DEV.md`，2 进 `docs/ct2-distribution.md`，且必须连 `make_cn_lock.py` 与
   `test_cn_lock.py` 一起搬（A4）。
5. **`update_config_file` 失去唯一生产调用者之后怎么办**（§1 的 ⚠）。**已定（owner，
   2026-09-04）：删。** 本项目不会再做第一方桌面端，CLI 的 `config.toml` 一直只有手改一条路，
   Nonoka 用自己的快照；`config_file.py` 与 `test_config_file.py` 连同文档里的四处提及一起
   移除。将来真要 `finesub config set`，从 `0.5.0pre` 拿回来重做——到时要的形状多半也不同。

---

## 9. 实施与复审记录

### 阶段 B 实施记录（2026-09-03，分支 `refactor/desktop-split-phase-b`）

按 §5 的 B0–B7 顺序做完，一个 commit。逐项对照计划：

| 项 | 做了什么 | 与计划的出入 |
| --- | --- | --- |
| B0 | 三份搬进 `test/bootstrap/`：`test_shell.py`（47 条）、`test_model_fetch.py`（27 条）、`test_fsops_links.py`（原 `test_fsops.py` 的 4 条，加了 `skipif(os.name != "nt")`）。六处函数内 import 桌面：三条 `TaskRequest` 回放改成直接断言记录下来的字典（`_recorded_request` 写的键与 `model_dump()` 一模一样，所以是等价替换）；两条 `package_shell` 随 B5 删；`ModelPrefetchFailed` 换成测试内的 `RuntimeError` 子类。三份都登记进 `test/conftest.py` | 计划算 72 + 6，实际 78 − 4（`package_shell` 两条与 `can_provision` 两条一起走）= 74 条进根套件 |
| B1 | `git rm -r desktop`，连同 `.claude/launch.json`、`agent-tasks/desktop-portable/`、`.github/workflows/desktop-ci.yml` | — |
| B2 | 表里十四处全改。`test_desktop_dependencies.py` 的 4 条 CLI 契约进了 `test/test_packaging.py`（其中「一个版本号」改写成 `test_the_version_number_has_one_source`：只剩根 `VERSION` 一处，lockstep 从 9 位归 1 位）；`test_paths.py` 的「不得 import desktop」直接删；`IMPORTING_TREES` 去 `desktop` 时 `scripts` 已在里面；`test_subprocess_text_encoding` 的扫描面用 `cli/src` 顶替了 `desktop/backend`（守卫的扫描面是守卫的一部分，少一棵树不能只是删） | — |
| B3 | `ci.yml` 新增 `windows` job（`cli/tests`、按名字跑 `test_secrets.py` + `test_fsops_links.py`、构建 wheel）与 `thin-cli-py310` job（从 `desktop-ci.yml` 原样搬来）。`release.yml` 重写成 wheel-only：`plan → build → github-release → pypi`，`sign` job、`supported_from` 输入、六资产校验、签名密钥全部移除；`plan` 只认 `CI` 一条 workflow。`publish-main.ps1` 的 `$RequiredWorkflows` 只剩 `CI`，`$PrivatePaths` 去掉 `desktop-portable` | B8 的「新 Windows job 必须绿过一次」已兑现（2026-09-04，run `33835921988`，`main` = `09a93a03`）：`cli/tests` 29 passed，`test_secrets.py` + `test_fsops_links.py` **41 passed、零 skip**，wheel 1.56 MB。重写后的 `release.yml` 也跑过一次 dry run（run `33836142090`，`dry_run_ref: main`）：`plan` 与 `build` 绿，两个发布 job 按 `dry_run` 跳过 |
| B4 | 删 `[desktop]`；`[desktop-worker]` 改名 `[runtime]`，注释重写；`[dev]` 去掉 Pillow / pyinstaller / hooks（三者只服务桌面构建） | 第一版留着锁的旧头部注释（marker 记的是整文件 sha256，改一字节就是几 GB 重建）。owner 问「会不会一直传下去」后改成**内容摘要**（`lock_content_digest`，去注释、去行尾——顺手修掉了 autocrlf 不同的 checkout 会互相触发重建的问题），`_LEGACY_LOCK_FILE_DIGESTS` 接住 0.4.x 安装的整文件哈希（两种行尾各一条），两份锁的头部随即改成 `--extra runtime`。那个常量在下次真正重新生成锁时删 |
| B5 | 删 `package_shell`、`application_source`、`PACKAGE_FRONT_END`/`CLI_FRONT_END`、`Command.shown_in`（`render_usage()` 不再收前端参数，`cli/main.py` 同步）。**额外删了 `can_provision`**：全仓只有 `package_shell` 传过 `False`，留着就是四个永远走不到的分支和一条「去桌面端装」的死提示 | 第一版为 0.4.x 互操作保留了 `locks.py` 的 `"desktop": "桌面端"` 标签与 `_recorded_request` 里只有桌面才设的三个字段。owner 问过之后核了 `0.5.0pre` 里的 `TaskRequest`：三个字段都有默认值（`False` / `None` / `""`），`extra="forbid"` 只拒**多出来**的键，所以 CLI 不写它们对 0.4.x 读者无害——三个字段删了；标签表只留 `cli`，0.4.x 的租约落到兜底、原样打出 `desktop`，看得懂。`task_index.py` 那条「别加字段」的规则仍成立 |
| B6 | 必改清单全过了一遍，外加 manual 里九处「桌面端在设置页填」类的说法。`desktop/README_DEV.md` 三段唯一落点按 §8 第 4 条去处：「清理与保留」+ 脚本命名进 `README_DEV.md`，pylock 重建进 `ct2-distribution.md`「锁的重建」。`cross-frontend-lease.md` 只在开头加一条状态注，正文当设计记录保留 | `CLAUDE.md` 的 Key facts 里「桌面默认值进 args 层」那句改成过去式，`test_option_defaults.py` 的两条 strict xfail 不动——第二层缺的仍是缺的 |
| B7 | `desktop-portable/` 删；`release/SKILL.md` 整份重写（两处版本位、两个环境全绿、四个 job、密钥退场） | — |

B8 验收：`compileall` 过；根套件 ci-venv 3883 passed / 50 skipped / 2 xfailed，miniconda 3913 passed / 44 skipped / 2 xfailed；`cli/tests` 29 passed；`build-wheel.ps1` 出 `finesub-0.4.2-py3-none-any.whl`（330 个文件，零个 desktop 路径，两份运行时资产在 `_vendor/src/finesub_bootstrap/`）；最后那条 grep 逐条看过——留下的全是三类：历史陈述（「桌面曾…」）、取舍依据里的先例、以及上面两处**故意**为 0.4.x 互操作留的。

⚠ 一条计划没写、做的时候才看见的：`test_a_front_end_without_a_prompt_is_never_asked` 之前靠 `can_provision=False` 表达「桌面包」，删掉那个开关后它测的仍然是「没给 prompt 就静默用默认位置」，行为没变、只是不再有第二个前端来命名它。

### 锚点 `0.5.0pre` 发布记录（2026-09-03）

`main` = `3d5993c1`（parent `8a33092a`，快进、无 force-push），tag `0.5.0pre` 为 annotated
并已推送，GitHub 上是 **prerelease** 而非正式 release。剥离面按 §4 的六项执行，58 个文件，
校验过公开树里一条私有路径都没有；`desktop/` 里的 pylock 与 runtime-manifest 确认已搬空。

**没有 bump 版本号**：快照里的 `VERSION` 仍是 `0.4.2`。理由写进了 release 说明——版本号
只在真发版时才动，它同时钉着 9 处 lockstep 版本位，其中 `package.json` 的 semver 与
Windows 版本资源的纯数字字段都容不下 `0.5.0pre` 这种写法。tag 名标的是「0.5.0 之前的那
一份」，不是树里的版本号。tag 不带 `v` 前缀，与 `v0.4.2` 那一列正式版分开；`release.yml`
是 `workflow_dispatch` only，所以任何 tag 都不会自动触发发版流程。

⚠ **第一次闸门是红的，而红的原因与剥离无关**——`dev` 上积压着四条**只有装了 `[asr]`
才绿**的测试（`249bf326` 修）。这是本次最值得记的一条：

| 发现 | 处置 |
| --- | --- |
| **三条不是「忘了跳过」，是测试真的去做了那件事**：`test_vocal_separation_pool` 的两条在夹具装假替身**之前**就捕获了要包装的 `_acquire_separator`，于是 recorder 包住真的那个，每跑一次构造一次真权重；`test_decode_prefetch` 的顺序基线写成 `decode_batch=4` 且不给假 decoder，紧邻的姊妹测试用的是 `1` | 捕获挪到夹具之后；基线改回 `1` |
| `test_vad_stage_guards` 确实该跳过，但**整模块 import 会把同文件另外 12 条不需要 `[asr]` 的设备解析测试一起拖红** | 改成 fixture 按需 import，那 12 条因此**第一次真正在 CI 上跑起来** |
| **本机检查清单只放了 miniconda**。既有 memory 只记了「ci-venv 会把 ASR 测试整片 skip 成假绿」，于是它被当成不可信环境跳过——但**反方向同样成立**：miniconda 的 `[asr]` 会盖住只有装了 extra 才绿的测试 | 推公开快照前按 `agent-tasks/release/SKILL.md` 第 2 步**两个环境都跑**（它本来就两个都列了） |

闸门本身表现完全符合设计：CI 红的那一轮 `main` 纹丝未动，`ci-gate` 留在原地等修复；
`Desktop CI` 两轮都是绿的，所以 Windows lane 没有第二个问题。

### 阶段 A 实施记录（2026-09-03，分支 `refactor/desktop-split-phase-a`）

A1–A4 全部落地，三套测试与两个脚本验收通过：root 3848 passed / `desktop` 393 passed /
`cli/tests` 29 passed；`build-wheel.ps1` 产出 `finesub-0.4.2`，三份资产在
`_vendor/src/finesub_bootstrap/` 下；`check-pinned-urls.ps1` 仍找到全部 13 条 URL。

计划之外的四处，记在这里：

| 发现 | 处置 |
| --- | --- |
| **版本号还有第三个读者**：`test_release_defaults_do_not_promise_deltas_from_unreleased_versions` 也从根 `pyproject.toml` 读 `project.version`（不是 A0 表里那个 `:7`），改 `dynamic` 后 `KeyError` | 一并改成读根 `VERSION` |
| **A3 不能对桌面端一视同仁**：launcher 是**单独冻结**的，它自己的 `finesub_bootstrap` 与它正在安装的 app source 是两个版本，`__file__` 在那里会指错树 | 参数保留、默认值只给 CLI 用；桌面端（launcher 与 `package_shell`）显式传 `app_source/src/finesub_bootstrap/...`，两处都写了为什么 |
| **`IMPORTING_TREES` 加 `"scripts"` 提前到 A4**：计划把它排在 B2，但守卫的扫描面应该在那棵树长出来的那一刻就覆盖它 | A4 一并加；B2 那一行只剩删 `"desktop"` |
| **`cli/tests` 的 `_vendored` 夹具变简单了**，不再伪造 manifest 与 lock（它们现在从真包读） | 顺带让那几条测试校验的是真资产 |

⚠ 本机 miniconda 没装 `build`，`build-wheel.ps1` 的默认 `-Python python` 也解析到
Windows Store 的解释器。验收时用了一个一次性 venv（`python -m venv` + `pip install build`）
并显式传 `-Python`。这不是本次改动引入的，但下次跑同一条验收会再撞一次。

### 第三轮复审

**第二轮修订新加的断言全部核实成立，无新错误**（`desktop/scripts/__init__.py` 确实存在、
`_every_import` 的位置与注释原话、`_PIPELINE_FILES` 与域标记守卫的报错文案逐字一致、
`load_app_paths` 的 4 处与 72+6 的账）。并确认 A4 的选择可行：仓库根有 `conftest.py`
而没有 `__init__.py`，pytest 会把根目录放进 `sys.path`——`desktop.scripts` 今天就是靠
同一机制被 import 的，`scripts.make_cn_lock` 会一样工作。

两处补进正文：

| 补什么 | 落点 |
| --- | --- |
| 那三条 `TaskRequest` 用例的处置写死为「改写成直接断言记录下来的字典」，不留「或删」——`test_task_output.py` 的 docstring 自己说桌面那半边在 `test_shared_index_contract.py`（随 `desktop/` 走），所以**没覆盖** | B0 处置表 |
| `scripts/` 一旦有 Python 就该进 `IMPORTING_TREES`，与删 `"desktop"` 同一行 | B2 那一行 |

另加 owner 决定一条：**`0.5.0pre` 不走 release 流程，但仍要剥掉本地 only 的被跟踪文件**
——写进 §4。

### 第二轮复审

重读全文、重点核对第一轮修订新加的断言。**新加的断言绝大多数经核实成立**
（115/219 的两个口径、两个 job 都是 `windows-latest`、DPAPI 的 `skipif` 与 workflow
里那句注释、`desktop-worker` 的三处字面量、`test_cn_lock.py` 只在字符串里出现
`sys_platform`，以及根 conftest 的 autouse 夹具已覆盖桌面 conftest 的三项环境隔离）。
四处修正 + 一处拍板，均已并入：

| 问题 | 落点 |
| --- | --- |
| B0「只 import `finesub_bootstrap`」只看了模块头：`test_shell.py:333,431,463` 有三条**函数体内** import 桌面的 `TaskRequest`，`test_model_fetch.py:546` 一条 import `ModelPrefetchFailed` | B0 加逐条处置表，账改成 **72 条直接搬 + 6 条要处理** |
| B0 漏了一步：搬进 `test/bootstrap/` 的文件要登记进 `test/conftest.py` 的标记表，否则 `test_packaging.py:87` 直接红 | B0 末尾的 ⚠ |
| B5 的「`load_app_paths` 仓外 10 个文件」是把 `desktop/` 也数进去了，实际 4 处 | B5 更正（结论「留」不变） |
| A4 的落点没法直接搬：`desktop/scripts/` 是包，而 `scripts/` 没有任何 Python | A4 拍板「`scripts/` 加 `__init__.py`」，并写清为什么不选 `tools/` |

**这一轮的教训与第一轮同形**：两次都是「只看了模块头/只看了一侧的扫描面」。
`test/test_import_boundaries.py:43` 的 `_every_import` 用 `ast.walk` 而不是 `tree.body`，
注释里写着「rename 漏掉的那个 import 就坐在函数体里」——本文第一轮恰好又踩了一次。

### 第一轮复审

一轮外部复审，逐条核对了本文对仓库现状的断言。**方向确认**（剥的是构建面不是代码，
A0 资产表与 B2 守卫表的行号基本属实），但指出五处实质缺口——按初稿执行会在阶段 B 丢测试、
丢 CI，阶段 A 落地当天根套件就红。五处全部核实成立，已并入正文：

| 缺口 | 落点 |
| --- | --- |
| Windows lane 会整条消失，不只 thin-cli-py310：`cli/tests`、`test_secrets` 的 DPAPI 用例、3.12 的 wheel 构建都只跑在 `desktop-ci.yml` 里 | B3 重写成「新建一条 Windows lane」 |
| `desktop/backend/tests` 里有 78 条测共享层的用例（`test_shell.py` 51、`test_model_fetch.py` 27，零平台标记），`git rm -r desktop` 会一起删 | 新增 B0，且**必须在 B1 之前** |
| A1 漏了根 `pyproject.toml:7` 已有静态 `version`，版本号现在就是两份 | A1 补「连根 pyproject 一起改 dynamic」 |
| `make_cn_lock.py` 会被删，而它生成的 cn 锁留下来了 | 新增 A4，脚本与它的 15 条测试一起搬 |
| §1「没有桌面专属模块」不完全对：`config_file.py` 的 `update_config_file` 失去唯一生产调用者 | §1 的 ⚠ + §8 第 5 条 未决项 |

另更正三处数字与遗漏的改动点：文件计数的口径（115 是 `desktop/` 之外，219 是全仓）、
`release.yml` 是 155 与 157 两处、「7 份桌面测试」实列 8 个名字；以及
`test_desktop_dependencies.py:164,217`、`test_cli_main.py:340`、`desktop-ci.yml:65`、
`test_windows_ai_runtime_lock` 的三处 `desktop-worker` 字面量、`application_source`
的唯一调用者，均已补进对应小节。

复审同时确认了两处本文原先只是推断的事实：`Copy-PythonTree` 递归拷贝且只跳
`__pycache__`/`tests`/点目录，`MANIFEST.in` 的 graft 覆盖整个 `_vendor`——A2 那三条
`Copy-Item` 确实可删。
