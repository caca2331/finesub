# 测试：组织方式与守卫

这份讲测试为什么这样组织、守着什么。**什么时候跑什么、命令怎么写**在 [`test-runs.md`](test-runs.md)。

**`--timeout=120` 卡的是「挂住」，不是「跑得慢」**（`pyproject.toml` 的 `addopts = "-n 4 --timeout=120"`）：实测最慢的单例约 3.3 秒，两个数量级的余量；套件里有真起子进程（agent driver）和真起本地 HTTP server 的用例，没有超时就是一个没有任何输出的CI job。重资源用例自己放宽。

**`finesub_bootstrap` 的测试在根套件里**（`test/bootstrap/`）：装机层是 CLI 的地基，
改 `fsops.py` 应该在全量那条 `pytest -q` 里就红，而不是等一个本地没人跑的套件。其中
**只有 Windows 能真跑**的几处（`test/bootstrap/test_fsops_links.py` 的 junction/robocopy、
`test/test_secrets.py` 的 DPAPI、`test/test_native_paths.py::test_the_crt_locale_switches_to_utf8`
的 CRT locale 切换）在 Linux runner 上是 skip，所以 `ci.yml` 的 `windows` job
**按名字**再跑它们一次——skip 不是 pass，那个 job 就是为了让它们真的执行一回。locale 那条在
维护者本机（代码页已是 65001）上是平凡通过，只有代码页不是 UTF-8 的 runner 才真测到切换。

**下载路径的默认单测一律不联网**：fake HTTP、fake subprocess、小文件，不访问公共镜像也不加载
模型。实机验收何时要重做见 [test-runs.md「何时跑哪种」](test-runs.md#何时跑哪种)。

## 设备解析：两条只能这样验的手法

两条都来自 2026-08 的逐阶段设备解析（取舍依据在维护者的本地计划里，不随公开快照发布；
那份计划两次写下「本机验不了」、两次都太早）。它们是**测试纪律**，所以住在这里，不住在计划稿里。

**一、在进程内劈开 torch 与 CT2。** 「CT2 能用这张卡而 torch 不能」这种机器多数人没有，
但不必换机器：`cuda_usable()` 判负的条件是卡的 compute capability **低于**
`torch.cuda.get_arch_list()` 的最小值，而那个列表可以替换——

```python
monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_130"])
```

于是 torch 如实认为自己没有这张卡的 kernel，而 CTranslate2 自带 arch 列表与 CUDA 运行时，
**照样看得见真实设备**。这不是打桩：`ct2_cuda_unusable_reason()` 跑的是真查询，卡也是真的。
⚠ `TestTorchAndCt2Disagree` 的第一条测试专门守着**这个前提本身**（`cuda_usable()` 为假且
`ct2_cuda_unusable_reason()` 为 None）——前提一旦不成立，后面两条就不再是它们自称的东西。

**二、用 `CUDA_VISIBLE_DEVICES=-1` 跑一遍无卡回归。** 这是最接近 CI 的本机复现，**只跑有卡
那一次会漏掉 CI 会红的**。第一次这么跑就红了 9 条：一条是真 bug（`cuda_device_present()` 用了
`torch.cuda.is_available()`，而 `CUDA_VISIBLE_DEVICES=""` 时它答 True 但 `device_count()` 是 0
——一台零可见设备的机器因此被放进 GPU 档，改成 `device_count() > 0`），八条是测试假设了宿主
有卡（钉算术的测试不该依赖宿主硬件，已加 autouse fixture 把能力钉成 True）。

⚠ 顺带记一条两个库真实不一致的地方：`CUDA_VISIBLE_DEVICES=""`（空串）下 torch 的
`is_available()` 是 True、`device_count()` 是 0，而 CT2 报 1（它把空串读成「未设置」）。
这正是「档位问 torch、ASR 问 CT2」的又一个理由——谁也别替谁回答。

## 标记说明

三个域标记在 `test/conftest.py` 的 `_PIPELINE_FILES` / `_ASR_FILES` 里按**文件**声明
（`test_llm_*.py` 由前缀自动进 `llm`）。

| 标记 | 含义 |
|------|------|
| `llm` | `test_llm_*.py`（conftest 按前缀自动加） |
| `pipeline` | 管线/批处理编排，以及一次运行所依赖的运行时与预置层：GPU 与线程预算、设备判定、分离器自身的管道、终端 reporting、config、secrets、state |
| `asr` | 音频 → 词 → 字幕这条路：VAD、解码、对齐、稳定化、分句，以及它们用的文本工具 |
| `heavy_resource` | GPU/大模型/大音频；默认 skip，需 `--run-heavy-resource` |
| `network_mock` | 起真实的本机 socket 去证明一条路由或一个网络层契约（`[api] proxy` 的转发证明、API 调用硬截止对滴流/无响应服务端成立）；默认 skip，需 `--run-network-mock`。快且不出机器，但不该让每次 `pytest -q` 都去绑端口 |
| `requires_main_checkout` | 契约依赖当前仓库就是主 checkout（真实 checkout 路径的解析）；linked worktree 中自动 skip，所以这些用例在任何分支上都是绿的。现在只剩三条（`test_paths`、`test_separator_accel`、`test_llm_config_and_budget` 里探测本地 tokenizer 的那条）；知识库的测试不在其中——worktree 写保护只挡**共享的**知识库，测试用的 tmp 根照常可写。逃生舱见 [`test-runs.md`](test-runs.md#何时跑哪种) |

**三个域标记是一个划分**：每个测试文件恰好带一个，`-m "llm or pipeline or asr"` 收集到的
就是全量（与 `pytest -q` 一致）。这条由 `test_packaging.py::test_the_domain_markers_cover_every_test_file`
守住——新增文件没登记会直接失败，顺带也拦住改名后留下的死条目。加新文件时把它加进
`test/conftest.py` 对应的那个元组即可。

**登记用的是相对 `test/` 的 posix 路径**（`bootstrap/test_paths.py`），不是裸文件名：套件
有子目录之后，两个目录各有一个 `test_paths.py` 是常态，裸文件名会让其中一个默默继承另一个
的标记。守护测试同样按 `rglob` 递归收集——搬进子目录的文件不能就这样滑出划分。

`testpaths` 今天只有 `test/`。同一条测试仍会检查 `testpaths` 有没有长出 `test/` 之外的条目——
那样的文件够不着 `test/conftest.py`，必须在文件开头自带 `pytestmark`，否则划分又有了洞
（0.5.0 之前 `desktop/scripts/tests/test_desktop_dependencies.py` 就是这样挂在外面的）。

### 没有 `slow` 标记

`slow` 已于 2026-08-15 连同 36 处用法一起删除。它不是「不好用」，而是**和真实耗时反相关**：
被它标住的 36 例里有 31 例来自当时的 `test_llm_search_loop.py`（模块级 `pytestmark`；那个文件随
搜索 loop 于 2026-09-09 一起删了，所以这段只能回溯到 `f5ba89df` 的父提交去核），单例全部
低于 0.27 秒；而实测最慢的三例——`test_llm_local_agent.py` 的两个子进程用例（各 3.3 秒）和
`test_state_store.py::test_concurrent_writers_do_not_lose_each_others_sections`（3.06 秒）——
一个都没被标住。`-m "not slow"` 于是恰好踢掉最快的那批、留下最慢的那批。

要省时间就按实测来（`-n 0` 全量 136 秒）：

| 位置 | 耗时 |
|------|------|
| `test_vad_streaming.py`（5 例） | 34.2 秒，占全量 25% |
| `test_llm_local_agent.py` | 次之，大量真起子进程的用例 |
| 其余单例 | 均 < 3.1 秒 |

`test_vad_streaming.py` 那 34 秒串在一个 worker 上，所以**加 worker 也压不到 35 秒以下**，除非拆那个文件。真想临时跳过，用
`--ignore=test/test_vad_streaming.py` 显式说，比维护一张会失准的标记表可靠。

## 测试替身放哪：一个名字可能有多个查找点

**`monkeypatch.setattr` 只改一个模块的绑定。** 一个名字被拆进多个模块后，patch 其中一个
不会炸——只是那条测试悄悄不再测另外几个。实测两例：`load_entry_texts` 在纠错子包的五个
模块里各绑一份、`PROMPT_VERSION` 被 `commit`（窗口记录的版本）与 `request_turn`（会话 checkpoint
哈希）各绑一份，只 patch 前者会让「bump 之后窗口照用、未提交的调用重发」这条测试半真。

两条随之而来的规矩：

- **替身放在真正做名字查找的那个模块上**，不要放在包的 `__init__`——patch 一个 re-export
  等于没 patch，却能骗过存在性断言。纠错子包有现成的 `test/conftest.py::setattr_correction`：
  它 patch 所有绑定该名字的模块，且至少命中一个才通过。
- **把函数内的延迟 import 提到模块级之前，先看这个名字有没有被 patch。**
  `from .x import f` 会让 `x.f` 上的 patch 失效；要提就写成 `from . import x` + `x.f()`
  ——依赖照样写在文件顶部，名字仍在调用时解析。反过来，有些按名绑定是**故意**的
  （测试要替换的正是「这个模块的视图」），移动前先读注释。

拆模块或改名时这两条会一起出现。

## 为什么仓库根有一个 conftest.py

`conftest.py`（**仓库根**，不是 `test/` 下那个）只做一件事：在**不能创建符号链接**的机器上
把 pytest 私有的 `_pytest.pathlib._force_symlink` 换成 no-op。

pytest 的 `tmp_path` / `tmp_path_factory.mktemp` 都走 `make_numbered_dir`，它会给最新的临时
目录挂一个 `<prefix>current` 便利链接——pytest 自己的 docstring 称之为 "best effort"，本仓库
没有任何代码读它。Windows 上没有 Developer Mode 或管理员权限时，背后的 `os.symlink` 必然抛
`WinError 1314`（pytest 吞掉继续），而**这次失败要 0.205 秒**，对比它装饰的那个 `os.mkdir`
只要 0.0003 秒。`test/conftest.py` 有两个 autouse fixture 各取一个临时目录，于是**每个用例
在跑第一行之前就付了两次**，再要 `tmp_path` 的还要付第三次——成本随用例数线性增长，而不是
随它们做的事增长。这就是「本机越来越慢、CI（Linux，符号链接正常）毫无感觉」的形状。

实测：全量套件 **38 分钟 → 1 分 21 秒**（当时的 `-n 2`，1893 passed / 21 skipped），只加这一个文件。

写法上的三个约定，改动它之前先读一遍：

- **探测而不是判平台**。能建符号链接的机器（Linux、CI、开了 Developer Mode 的 Windows）
  保留 pytest 文档化的行为；只有建不了的才跳过。谁哪天开了 Developer Mode，链接自己就回来，
  不需要有人记得回来删代码。探测本身每进程一次，约 0.28 秒。
- **`hasattr` 兜底**。`_force_symlink` 是私有 API，将来改名的代价应该是丢掉加速，
  而不是套件跑不起来。
- **放仓库根**。两个套件（`test/`、`cli/tests`）的 rootdir 都解析到仓库根，一处覆盖全部；
  放 `test/conftest.py` 只管一个。

## 结构守卫：自动化规则

这些测试不测行为，测**结构约束**，因为它们各自对应一次「什么都没报错，但事情已经错了」：

| 守什么 | 文件 | 出错时长什么样 |
| --- | --- | --- |
| 导入方向（media/subtitles 不向上依赖、`speech` 不依赖 `llm`、别处不得 import `finesub.pipeline`） | `test_import_boundaries.py` | import 照常成功，分层悄悄没了 |
| **「这台机器有什么 GPU」只在 `speech/runtime/device.py` 问 torch**（`is_available`、`get_device_*`、`torch.version.cuda/hip`、`torch.backends.mps`、`torch.xpu` 等；用卡的调用如 `empty_cache` 不算） | `test_import_boundaries.py` 的 `test_device_probes_live_in_one_module` | 一处裸 `is_available()` 把活放上没有内核的卡；AMD 卡被当成 NVIDIA |
| **ASR 解码选项只在 `asr_backend.DECODE_OPTIONS` 声明**，`transcribe.py` 不得内联拼选项名；每个注册的后端正好接受它能力对应的那一组 | `test_asr_backend_contract.py`（含一个故意写坏的第二后端，证明守卫会红） | 加了选项，只实现了 CT2 的后端悄悄忽略它，在没人跑 CI 的硬件上走另一条路 |
| **prompt 文字不写在渲染 prompt 的 Python 里**（名字带 `prompt` 的模块与 `shape_criteria.py`；够一个分句长的中日文字面量，文档字符串、报错与给用户的提示不算） | `test_llm_prompt_literals.py`（参数层的 dict 逐个点名豁免，此外一句也不许有） | 迭代 prompt 要同时改模板和 Python，冷读者照文档去看模板会漏掉这些句子——加守卫时已攒了 60 多句 |
| **`python -m finesub.pipeline --help` 与参数错误不 import torch** | `test_import_boundaries.py` 的 `test_the_pipeline_cli_answers_help_and_argument_errors_without_torch`（新解释器里跑） | 顶层多一个 import 就让每次 `--help`、每个打错的选项多付一秒多的 torch 加载，没有任何报错 |
| **`torch.jit.load` 只在 `speech/runtime/native_paths.py` 调**，别处用它的 `load_torchscript`（传文件对象） | `test_import_boundaries.py` 的 `test_torchscript_loads_go_through_one_helper` | UTF-8 代码页的机器上一切正常；中文 Windows（GBK）上装在中文路径下时，libtorch 的窄字符 `fopen` 报「文件不存在」——0.5.1 的 silero 报错就是这样 |
| **依赖锁里的环境标记只能是目标平台（64 位 Windows 上的 CPython）恒为真的那几种** | `test/bootstrap/test_runtime_health.py` 的 `test_every_lock_marker_is_true_on_the_target_platform` | 运行环境的快检不求值标记：一个在目标平台为假的标记会让那个包被当成「整个包不见了」，重建又装不上它，于是每次运行都停在重建上 |
| 选项默认值只有一份（argparse 不得抄后端的答案） | `test_option_defaults.py` | 两处默认值慢慢漂开，前端与后端行为不一致 |
| 文档链接、章节号与 `X.md「标题」` 落得到（含源码注释里的、含中文数字节号） | `test_doc_links.py` | 链接照常渲染，指向不存在的东西 |
| **单个函数不得超过 300 行，已超的不得再长** | `test_function_size.py` | 没有任何一刻需要有人判断「这个函数是不是太大了」——`run_search_loop` 就是这样长到 882 行的 |
| **文档枚举的集合必须与代码一致**（当前只守 GPU 档位名） | `test_doc_facts.py` | 代码加了两个档位，文档的表还是三行，而且这类错误已经复发过一次 |
| **中文正文用全角标点**（范围只有 `docs/manual/`） | `test_doc_style.py` | 文档照常渲染，但下一次内容改动的 diff 被标点噪声淹掉——2026-09-03 一次 700 行改写里真正的变动只有六个选项名 |
| **只有注册表和两个结构化表的主人直接读 `config.toml`**（`settings.py`、`api_keys.py`、`model_routes.py`） | `test_settings.py` | 又多出一种读法：`[chunking]` 就是这样把 `true` 读成 1、`3.7` 读成 3 的 |
| **每个声明过的配置键都写进它的文档**；说明在 `docs/manual/` 下的，还要进 `config.example.toml` 与 `resources.md` 的节表 | `test_settings.py` | 键能用，但除了读代码没人知道——启用这条时一次抓到 10 处 |
| **捕获文本输出的子进程必须写明 `encoding=`** | `test_subprocess_text_encoding.py` | 英文与 UTF-8 机器上一切正常；中文 Windows 上第一个非 ASCII 字节就在读线程里崩，报错不指向真因 |
| **子进程必须带 `creationflags=no_window()`**，须共用终端的逐个列进 `SHARES_THE_TERMINAL` | `test_subprocess_console_window.py` | 在终端里和本机 agent 宿主里都看不出来；只有 FineSub 被无窗口拉起时，每个子进程闪一个黑框（见下一节） |
| **公开文档不得点名发布时剥离的文件、不得带维护者本机的路径**；`.gitignore` 放行的路径必须进 `$PrivatePaths` | `test_publish_filter.py` | 公开仓库里的读者拿到一个打不开的文件名或别人机器上的路径；清理前攒了 140 多处 |
| **测试不得在 checkout 与开发者的家目录里留下东西**（只含 `.batch.lock` 的 `out/batch/<id>/`、空的 `logs/run-*.log`、指向 pytest 临时目录的 `batches.json` 行；checkout 根上新出现的 `.state*`、`.env.lock`、`.task-activity*`；`data/reference/url-info.json` 里保留域名（`example.com`、`*.test` 等）的条目） | `test/conftest.py`：`managed_data_root` 把状态文件、`.env`、agent 活动租约、`HOME` 与 agy 在家目录下的两个目录指进临时目录（Windows 的 `USERPROFILE` 不整体改：碰模型的测试从家目录的模型缓存读权重，空家目录会让它们去下载；剩下没隔离的只是读家目录里的缓存与 CLI 配置，没出过问题，先把那两条测试改成不依赖缓存的代价不值，owner 2026-11-07 不做），`_checkout_paths_in_tmp` 改写模块自己绑定的批次根、日志目录与 `data/reference`，会话结束时再查一遍（不是单条测试：整轮跑完才判，泄漏了就让整轮失败；`test_suite_isolation.py` 钉住这两半） | 每跑一次全量多一批垃圾；注册表被测试行灌满，真批次的 `--resume-batch` 要在里面找——修复前攒了 34 个目录、101 个日志、50 行；限流账本记上测试 key 的冷却、`url-info.json` 混进假 URL，key 池测试读到开发者真的 `.env` |

⚠ `test_function_size.py` 的 `_ALLOWED` 是**棘轮不是豁免名单**：每条记着函数现在的行数，
长了红，短了没把数字调低也红（留着的余量会被下一次改动悄悄用掉），掉回 300 行以下的必须删条目。
要让某个函数变大，得手动改那个数字——那正是这个决定该被看见的时刻。

⚠ `test_doc_facts.py` 只守**可枚举**的事实（一组名字），不守数字与描述。逼近「读懂句子」
的守卫误报率高到没人维护，那比没有守卫更糟。

⚠ `test_doc_style.py` 的范围**只有 `docs/manual/`**，因为只有那里归一化过。`docs/` 根与
仓库根还各有一百多处 CJK 相邻的半角（`CHANGELOG.md` 里还有不该改写的历史条目）。要扩大
范围，**先归一化再改 `_SCOPES`**，顺序反了就是一片红。它遮蔽代码块、行内代码、链接目标、
URL 与 HTML——遮蔽面和做归一化时用的是同一套，否则守卫会去管它没管过的地方。

## Windows 子进程不弹窗：为什么要这个 flag、怎么验

**机制。** Windows 让控制台子进程沿用父进程的 console。父进程**没有** console 时（被 GUI
宿主、`pythonw`、计划任务无窗口拉起），每个控制台子进程都会新开一个**可见**的，活多久闪多久。
`finesub_bootstrap.processes.no_window()`（`CREATE_NO_WINDOW`）让子进程拿到隐藏 console，
它再起的进程沿用这个隐藏 console——所以 agy 的 hook、厂商 CLI 内部起的进程这些我们传不了 flag
的，也一并安静。agent CLI 的启动原本就带 `creationflags`，但只有进程组和挂起，照样闪；守卫因此
判**值**（跟随同一作用域里的一次局部赋值），不只看有没有这个关键字。

**实测（2026-09-16）。** 同一段操作（agy `--version`/`--help`/`/hooks` ×2、`ffmpeg -version`），
在无 console 的 FineSub 进程里：修复前 6 个可见窗口，修复后 0 个。修复后又在无 console 下跑了一次
真实的 Codex 调用：CLI 正常启动、正常输出事件流，窗口数为 0。

**怎么复现**——三步，缺一步就测不出来：

1. **FineSub 那一层必须真的没有 console。** 用 `subprocess.Popen([...], creationflags=subprocess.DETACHED_PROCESS)`
   拉起跑被测代码的 Python（进程里 `kernel32.GetConsoleCP()` 为 0 即确认）。⚠ 在终端里、在
   Claude Code 或 CodeBuddy 的 shell 里直接跑**看不出问题**：它们都给了 console（agent 宿主给的是
   隐藏的，code page 非 0、`GetConsoleWindow()` 为 0），不带 flag 的子进程也不会弹窗。
2. **用 WinEvent 钩子数窗口，不要轮询。** `SetWinEventHook(EVENT_OBJECT_CREATE, EVENT_OBJECT_SHOW, …,
   WINEVENT_OUTOFCONTEXT)` 加消息循环，只数类名为 `ConsoleWindowClass` 的 `EVENT_OBJECT_SHOW`。
   ⚠ 每 20 ms `EnumWindows` 轮询一次会漏掉：`agy --version` 这种进程几毫秒就结束，第一版监视就是
   这样一个都没抓到的。先用一次 `CREATE_NEW_CONSOLE` 起的 `cmd /c exit` 自测，确认钩子能数到 1。
3. **确认 import 的是被测代码。** 在 worktree 里，不经 pytest 的 `python -c` 会 import 主 checkout
   的 editable 安装；被测脚本里显式 `sys.path.insert` worktree 的 `src` 并打印模块的 `__file__`。
   这次就有一轮因此测到了未修的代码。

**守卫的已知盲区**（今天仓库里都不存在，记下而不处理）：`**kwargs` 拆包按「已抑制」放行；
`from subprocess import run` 这种直接导入看不见；经参数注入的进程工厂（`environment.py` 的
`process_factory`）不在扫描面里，那一处靠调用点自己传 flag。

## 备注

- `test_pipeline_refactor.py`、`test_intervals.py`、`test_vad_streaming.py` 收集时会 import torch，collect 略慢属正常。
- 默认测试套件不生成 prompt preview。重打工具（`tools/session_replay`）与离线探针（`tools/bench`）的测试
  不被默认收集，怎么跑见 [`test-runs.md`](test-runs.md#命令)。
- 默认单测不得加载 Whisper/audio-separator、处理大音频或消耗 Gemini quota。
- `.github/workflows/ci.yml` 的根套件跑两遍 Linux：`test` 只装 `[harness,dev]` 加 CPU torch，缺可选包的用例在这里
  skip，模块级 import 了可选包的改动也在这里红；`extras` 再加上能装成 CPU wheel 的可选包（`[cloud]`、`[kb-phonetic]`、
  读音库、faster-whisper 与 stock CTranslate2、librosa、safetensors），让那些用例真跑，`-rs` 列出仍然 skip 的。
  CPU torch 与 `[asr]` 的钉版一致（`test_packaging` 核对）。另有 macOS 一个 job 与两个 Windows job：`windows`
  （`cli/tests`、按名字跑 DPAPI、junction 与 CRT locale 三处、构建 wheel）与 `thin-cli-py310`（Python 3.10 构建并安装
  wheel，在 managed runtime 不存在时运行 `finesub agent-clean`，守住 cleanup 的轻依赖/不 provisioning 契约）。
- ⚠️ **CI 里的 CTranslate2 是 stock 的**：`extras` 跑 `test_fw_refine.py` 的单元用例，但 patched wheel 不在 CI 里，
  也不跑重资源用例。**代价最大的一条因此不在 CI 里**：`test_a_cpu_model_decodes_and_then_shuts_down`
  （`heavy_resource`，另需 `--run-heavy-resource`）守的是 CPU GEMM 后端选错导致的
  死锁/`No SGEMM backend on CPU`，两种都只在真解码时暴露。也就是说**换 CT2 wheel 后必须
  在本机手动跑它**，命令见 [test-runs.md「CI 跑不到、只能本机跑的」](test-runs.md#ci-跑不到只能本机跑的)。
- linked worktree 的 `.git` 是指向主仓的文件。`test/conftest.py` 据此跳过
  `requires_main_checkout`，但显式验证 worktree 重定向/写保护的测试仍正常运行；回到主
  checkout 后这些用例自动恢复，不需要额外参数。
- `test_resource_budget_pipeline.py`：真跑「分离 → VAD-ASR」，断言实测 peak 显存/内存不超过
  所选档位上限；参数化跑遍 `entry`/`standard`/`high` 三档。需 CUDA（无 CUDA 自动 skip）与
  已缓存的 audio-separator 模型。
  - **分离那一半**用合成音频，一直有效。`RESOURCE_TEST_SECONDS` 可加长音频以压 RAM 路径——
    也是够到多 worker 的唯一办法（300 秒阶梯：>300s 才 2 个，>600s 才 3 个）。
  - **VAD-ASR 那一半需要真实语音**（`assets/harvard.flac`，`assets/` 不进仓库），缺失时
    明确 skip。合成音频是包络下的纯音，能骗过能量 VAD 但解码器转不出任何东西——2026-09-01
    实测三档 segments 全为 0。它此前「通过」，靠的是分离器泄漏的权重喂饱了那个
    `peak_gmem > 256MB` 守卫；守卫现已改成从产物验。
  - 同文件的 `test_the_separator_gives_its_weights_back` 是显存/内存泄漏回归：连跑三次分离，
    CUDA 与**主机侧** torch 张量残留都必须有界（只看 CUDA 会漏掉「把权重搬去内存」这种修法）。
