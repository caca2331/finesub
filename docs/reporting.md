# 上报与日志：`finesub.reporting` 的契约

`src/finesub/reporting.py` 的 owner 文档。谁在什么时候说话、说给谁听、以什么级别落到哪个前端。

管线代码**从不直接 `print`**：所有面向人的输出都过 reporter。这不是风格约定——batch、
CLI 终端、落盘 run 日志是三个不同的渲染面（桌面 worker 曾是第四个），绕过 reporter 的一行
`print` 只会到达其中一个，而且通常不是需要它的那个。

历史与取舍（为什么是这八个事件、为什么砍掉 `on_stage`）在本地
`docs/archive/cli-bootstrap-logging-download-plan.md` §4——它不随仓库发布。

## 1. 事件契约

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

几条不显然的：

- **`planned` 是 `[n/N]` 的唯一来源**。分母由本次运行自己的计划决定，renderer 不去猜——
  `--stage` 与已存在的上游产物都会改变它，它不是固定的 4。
- **`failed` 是必需的，不能靠异常代替**。`warning` 说的是「结果受影响但还在跑」；运行终止
  如果只有异常加 traceback，renderer 无从把它渲染成阶段列表里的一格。异常照常向上抛，
  `failed` 只负责上报。
- **`stage_started` 的 `reused` 只有一个含义：这一趟什么都没跑。** 产物侧的状态词表是三个
  （`executed` / `reused` / `skipped`，见 `README_DEV.md` 的复用规则），而这里**没有**对应的
  第三态，这是有意的：唯一能被跳过的阶段（`--no-separate` 的人声分离）照样产出它的产物，
  只是换了条更便宜的路，所以它「在跑」这件事没有变。**换的是哪条路由 `detail` 说**——
  2026-09-03 加这个开关时评估过给 `reused` 加第三个取值，结论是那会让四个 renderer 各长一个
  它们并不需要区分的分支，而 `detail` 本来就是为「这一格里正在发生什么」准备的。
  ⚠ 所以别反过来把 `skipped` 折进 `reused`：那会让终端显示「已有结果，跳过」，而实际上
  这一趟确实转码了一份新产物。
- **`summary` 收带标签的数字，不收一句话**。只有 stage 知道那 7 段叫「噪声片段」，只有
  renderer 知道这次运行有没有终端可以重画。零值与 `None` 由 renderer 丢弃，stage 照常报全。
- **`stage` 参数收的是 key，不是中文**。中文由 renderer 经 `STAGE_LABELS` 映射；传中文标签
  会让 `[n/N]` 前缀查不到。阶段名的唯一真相是 `STAGE_LABELS`——同一次运行在两个渲染面叫成
  两个名字是支持成本，不是风格问题。
  `STAGE_LABELS` 里另有三个 key 是 runner 的 **bin**（`download`/`asr`/`llm`）：失败发生在哪个
  bin 是 runner 唯一知道的粒度，`failed()` 用它上报。传一个表里没有的 key 不会报错，只会把英文
  原样打出来。

## 2. 谁绑定，谁渲染

`run_pipeline()` 从**调用方的绑定**里取 reporter（`reporting_to(...)`），不另加参数：调用树
深处的 stage 本来就是这样读的。线程局部的默认 reporter 是 `NullReporter`——这对库是对的，
但 `speech/recognition/transcribe.py`、`vad_asr_stage.py` 这些**同时是独立开发 CLI** 的模块
不绑定就会把自己的 CPU 回退告警吞掉。`reporting.terminal_reporter()` 是这个绑定的单一入口
（输出到 stderr，级别可用 `FINESUB_LOG_LEVEL` 覆盖）。

**每个 `main()` 都要绑定 renderer。**

| Renderer | 用在哪 |
| --- | --- |
| `TerminalReporter` | CLI 终端。TTY 上同一条进度原地重画，非 TTY 只在跨 10% 或阶段变化时新增一行 |
| `FileReporter` | 落盘 run 日志，**恒定 verbose**，见 §6。全仓只有 `pipeline.py`（前台单源那条路）一处构造 |
| `FanOutReporter` | 终端 + 文件。任一 renderer 抛异常不影响运行——上报是旁白，不是工作 |

**多源批是唯一需要行前缀的场景**：多个任务共用一个终端，可原地重画的进度行会被最后说话的
那个任务覆盖掉，两边都读不成。所以 batch 给每个任务一个带 `[<source>] ` 前缀、且强制行模式
（`isatty=False`）的 renderer。

## 3. 三个级别

| 级别 | 默认显示 |
| --- | --- |
| `quiet` | 影响运行的 warning/error、最终路径 |
| `normal`（默认） | 阶段、节流进度、阶段摘要、warning/error、最终路径与总耗时 |
| `verbose` | 再加 timing、正常资源占用、worker/backend、逐次 rescue 与语言选择 |

完整 debug 信息**始终**写进 run 日志，不要求用户为了保留诊断而使用 verbose UI。

## 4. 各阶段的上报口径

**分离器**

- 级别的来源只有一处：`reporting.resolve_log_level` —— 显式 `--log-level` > `FINESUB_LOG_LEVEL`
  > `normal`。前端要用两次（自己建 reporter、再交给 `quieted_libraries`），各解析一次就会出现
  「环境变量在某条路上失效」（2026-08-30 实测：统一入口把未指定提前解析成 `normal`，环境变量
  对所有路径都失灵了）。
- 第三方降噪**不在分离器适配层，而是整次运行**（`reporting.quieted_libraries`，由前端在
  `main()` 里进入）。两条通道两种处理：**logging 抬到 WARNING 而不是静音**（分离器的
  `CUDAExecutionProvider not available` 要留），**tqdm 直接禁用**（进度条写进日志文件永远
  只是噪声，何况 pipeline 自己会画）。`verbose` 两者都不动。
- logger 名按**模块**而非包命名（`separator` / `common_separator` / `mdxc_separator`，不是
  `audio_separator`），而且 `Separator.__init__` 的 `log_level` 参数会覆盖构造前设好的级别，
  所以必须在构造时直接告诉它。这两点是实跑才发现的——单测曾用了和代码同一个错误猜测。
- 默认只维护一条 FineSub 进度：完成 block / 总 block、并发数、已用时间。并发 future 完成时
  就更新计数，不等按输出顺序 merge。block 内没有稳定 callback 时**不伪造平滑百分比**，用同
  一行 heartbeat 更新已用时间。
- 编译后端降级是 warning 不是 failed（任务还在跑，只是慢）。四个 code 对应四个失败点：
  `separator-compile-unavailable`（AOTI 建包）、`separator-package-unusable`（AOTI 加载）、
  `separator-jit-unavailable`（JIT 安装）、`separator-jit-failed`（JIT 首次 forward，已原地
  还原）。后者的 `action` 给出删 accel 目录重新启用的方法；显存不足那次不给（没写 probe）。
- `gpu-vram-short`：所选档位要的空闲显存比驱动当下报的多。**只报不改**——降档会连带改变
  分块规划、进而改变产物边界，而 `auto` 读容量正是为了让档位稳定；何况"不够"不等于
  "跑不了"（`entry` 实测峰值 2.26GiB，要求写的是 3GiB）。两个 GPU 阶段各报一次，
  判据与位置对 `auto` 和手填一视同仁。已在最小档时 `action` 不会建议再降档。

**VAD / ASR**

- group 进度只在越过下一个 5% 档位时上报；恢复 checkpoint 时立即上报当前起点。节流放在
  **调用点**而不是只靠 renderer——renderer 只管终端，而一个把每次上报变成事件的渲染面
  （桌面 worker 曾是）要求事件量的上界在源头就有。**计数单位是 interval 而不是 group**：group 总数要等分组跑完
  才知道，分母不能是它。
- **尾部第二模型校验按 clip 报进度，挂在 `aligned` 名下**（2026-09-04）。它是该阶段的尾巴，
  不是新阶段——`aligned` 走到 100% 之后冒出第二个阶段名会被读成新阶段开始。同名下把计数
  从 0 重开是**已被预期的情形**，不是打擦边球：两个 renderer 的去重都键在 `(step, total)`
  上，注释写明就是为了「分母中途变化」的阶段（`FileReporter.progress`）。
  **`0/total` 在模型加载之前就发**：加载与第一次 `generate` 正是慢的那段，观察者首先要知道
  「在跑什么、有多少」，其次才是数字在动。**按批而不是按 clip 报**：一次 `generate` 从外面
  不可中断，逐 clip 计数是假的，而批也正好是上面那条「事件量上界在调用点」要的东西。
  没有可查的 clip 时一条都不发（`0/0` 会是唯一一条永远不动的进度线）。
  在此之前这里是全流程最长的一段静默——受限显卡上 197 s 的阶段里有 168 s 零事件，
  而这种静默与进程挂掉在观感上完全一样。
- temporary recall、short-language reuse、rescue ladder 步骤进 verbose/debug。
- normal 在阶段结束输出汇总计数：总 groups、temporary recall 次数、beam rescue 尝试/接受、
  隔离异常 interval 数、真正丢弃的 groups。
- 会改变质量且用户可能需要处理的事件仍立即 warning（group 最终丢弃、CPU 回退、模型验证被
  禁用）；正常且已恢复的内部尝试不逐条警告。

**稳定化 / SRT**

- 稳定化保留一行摘要，只展示非零的 removed/dropped/tag 计数。
- 中间产物不再逐个 `Wrote`；reused 由 stage 行表达。最终交付物只打印一次。
- timing 与低于预算的资源占用写 metadata + verbose；接近预算或越界才在 normal warning。
  **四个阶段都要计时**，走 `pipeline._record_stage_time` 一个入口：verbose 承诺了 timing，
  只有部分阶段报会让人靠总耗时做减法。复用的阶段记为「无耗时」而不是 0——「没跑」和
  「瞬间跑完」是两回事，metadata 会被工具读去做统计。

## 5. LLM 段

单列有两个原因：主体是**新增上报点**而不是转换现有 `print`，以及它有三个线程池
（`parallel.py` 的纠错与查询轮、`clip_prefetch.py` 的剪辑预取）而 reporter 绑定是线程局部的。

**阶段边界不归这一层**：`stages.py` 已为每个阶段（含 `translated-srt`）发过 `stage_started`，
再发一次会重开阶段行并重置进度节流。

| 事件 | 时机 / 位置 | 级别 |
| --- | --- | --- |
| `debug("correction plan", {windows, reused_plan, driver})` | `run.py` 规划与 refit 之后 | debug |
| `progress("translated-srt", …, unit="windows", detail=chunk_id)` | 每个窗口单元完成（`stages/correction/progress.py`）。单位用英文，与既有阶段的 `blocks`／`intervals` 一致 | normal |
| `debug("correction window attempt", …)` | `attempts.py` 每次重试 | debug |
| `debug("correction window validation failed", …)` | 进入修复轮时 | debug |
| `debug("llm api call", {model, tier, key, code, sec, n, why?})` | `llm_runtime._record_api_attempt`——**每次 REST 调用一条**，成功与失败同一处。`chat_complete` 是三种 transport 的唯一收口，所以这一个点覆盖全部 API 后端。`why` 只在失败时出现，写的是**端点自己回的那句话**（截 200 字符）——裸一个 `429` 分不出「这把 key 今天用完了」和「慢一点」 | debug |
| `debug("agent call", {driver, model, code, sec, episode})` | `local_agent._report_attempt`，由 `_run_episode` 的 `finish_attempt` 调用——**每次本地 agent CLI 调用一条**，同样成功与失败同一处 | debug |
| `debug("web search request", {provider, status, for, sec, retry})` | `web_search.WebSearchClient._report_request`，由 `_try_pool` 调用——检索与抽取、逐条与成批的**唯一**公共路径 | debug |
| `debug("rate limit wait", {scope, seconds})` | 上报点在 `rate_limit.py`：`scope` 只有它内部知道，且能同时盖到 `llm_runtime.py` 两个调用点 | debug |
| `debug("research round", …)` | `research.py` / `search_loop.py` 每轮 | debug |
| `warning(…)` | **举例，不是词表**：知识库仓库不可用、worktree/缺 git 跳过知识库更新、key 池超建议值、target 不支持视频而降级 | normal |
| `gemini-upload-retry` / `gemini-upload-failed` | Files API 上传的重试（与模型调用的重试预算分开）：每次重试一条，写阶段（`upload`/`state_poll`/`token_poll`）、attempt/max 与等待秒数；只在重试过之后仍失败才发 `failed` 那条。正文只写异常类型或 HTTP 状态，**不写** resumable session URL（它是 capability token）与 key | normal |
| 内容过滤阶梯 | 判据看 `LadderOutcome`：`level <= 0` 且 `dropped_units` 空＝纯重试通过发 `debug`；**丢了注入单元**（证据/词条，不是源字幕）发 `warning` | 两者皆有 |
| `summary("translated-srt", …)` | 阶段收尾。报窗口、拆窗、调用、重试、修复轮、内容过滤恢复次数与按 tier 的调用数 | normal |

✱ **这三条只写状态与一句话描述，永不写 prompt 或响应正文。** 而且那句描述是
**对方给的**（HTTP 状态、端点回的错误正文、provider 抛的异常、agent 进程的退出码），
不是我们替它拟的措辞——日志的用处是复现对方说了什么，不是复述我们的理解。 全文已经按次写在
`<stem>.llm-artifacts/exchanges/`，日志再来一份会把一次长任务撑到几十 MB——而这个文件存在的
理由正是「出问题时用户直接把它发过来」。检索那条带 query 本身（本来就短，去掉它这行等于没说），
超过 120 字符截断。两边对得上不靠文件名：exchange 正文渲染的就是 `api_attempts` 这份清单。

⚠ **不写 key、也不写带凭证的 URL。** key 只写 label；Files API 的 resumable session URL
不写（见上表）；而端点回的那句话在进日志之前先过 `reporting.redact_credentials`，把 URL 里的
userinfo 剥掉——`[llm] proxy` 与自定义 `base_url` 是用户自己的地址，`https://user:token@host`
是写它的常见形式，而 httpx 的错误消息会把请求 URL 原样引上。这跟
`finesub_bootstrap.shell._safe_host` 为 `doctor` 做的是同一件事、同一个理由：**这份文件的用途
就是被贴出去**。三条都有测试钉着。

`summary` **是收尾一行，不是审计凭据**——逐窗真相在 `correction-windows.jsonl` 与 exchange
日志里，这一行不追求与它们逐项对齐。

**不设 `window_dropped` / `window_degraded`**：今天没有「丢一窗继续跑」这条路径——重试耗尽即
`raise`，parallel 侧 drain-then-raise 整次失败，由 `pipeline` 已有的 `failed` 表达。要引入它是
**产品决定**，不该夹带在词表里。（降级交付确实存在——内容过滤丢注入单元后仍交付，由上表那行的
`warning` 表达。）

### 5.1 三条约定

1. **分母会中途变大，且两个 driver 的形态不同。** 来源是 `attempts.py` 的 `add_tail` /
   `restart_halves`（截断拆窗），**不是** `run.py` 的 `_refit_pending`（它在 driver 之前只跑
   一次）。serial 直接改外层列表；parallel 只动 slot 内部的 `pending_units`，`future_map` 条数
   恒定。**所以分母由一个共享计数器维护（完成单元数／已知单元数），不能从 future 数推。**
   实现是 `stages/correction/progress.py`：构造时捕获 reporter + 锁，`commit_window` 是唯一的
   记数点，拆窗处 `add_units`。
2. **报点在工作线程，两种手段都要。** 进度计数器在构造时（主线程）捕获 `current_reporter()`
   并自带锁；触发点是 `add_done_callback`，跑在完成的那个线程里。池另外绑
   `initializer=bind_reporter` 供体内其它上报用。✱ **AST 守卫：`llm/` 下的 `ThreadPoolExecutor(`
   必须带 `initializer`，无豁免。** 范围先只到 `llm/`——`speech/` 那两个池（`energy.py`、
   `spectral.py`）今天没有上报点，等它们要说话再扩。
3. **事件量上界在源头（窗口数）。** 实测每次运行 1–8 窗，所以不额外节流。✱ **两个 renderer
   的进度去重键都把 `total` 算进去**（`FileReporter._last_step` 是 `(step, total)`，
   `TerminalReporter._last_tenth` 是 `(tenth, total)`）——**曾经只比十分位、且只在
   `stage_started` 里复位，分母增大时会吞掉整窗事件**，而本段明令不发 `stage_started`。
   这是所有阶段共有的缺陷，纠错只是第一个分母会动的阶段。

### 5.2 没做主的产品决定

不是遗漏，是没人拍板：

- ~~**多源批要不要也写 run 日志。**~~ 已做（2026-08-31）：每项一份，文件在该项第一次说话时才
  开、随本次运行一起关（没跑起来的项不留空文件），同名 basename 加数字后缀而不是往同一个文件里
  追加。多源批正是没人逐行盯着的那种运行，最需要这份文件。

## 6. 落盘的 run 日志

终端之外，每次运行往 `<user-data>/logs/run-<时间戳>-<输入名>.log` 写一份**恒定 verbose** 的
记录，与 `--log-level` 无关：终端保持好读，决策细节留在盘上，用户报障时直接把文件发来即可。

三个刻意的选择：

- **一次运行一个文件，不是共享一个文件。** 两个 CLI 运行可以同时在跑
  （[`cross-frontend-lease.md`](cross-frontend-lease.md)），分文件既不用加锁也不会把两次运行
  交错在一起。
- **不放在 `out/<stem>/`。** 日志必须在输出目录确定之前就开始记（参数解析、设备解析、下载
  路线判定），而且**启动就失败的运行根本没有输出目录**——那恰恰是最需要日志的一次。
- ✱ **`quieted_libraries` 仍然收终端档位。** `verbose` 捆绑了两件事：我们自己的 debug 行，
  以及放行第三方 logging 与 tqdm。文件要前者，**绝不能要后者**——tqdm 靠 `\r` 高频重画，倒进
  文件就是几十 MB 的噪声。这是本节唯一容易写错的地方，有守卫测试
  `test_pipeline_log_shape.py::test_the_log_file_does_not_drag_tqdm_and_library_logging_along`。

实现是 `FileReporter`（恒定 verbose、按十分位节流 progress、单锁串行化多线程写入）与
`FanOutReporter`（终端 + 文件）。清理沿用 `finesub_bootstrap.logs.prune` 的「保留最新 100 个」，
与 install/session 日志共用预算。

**产物正文不是日志。** `llm/` 下 3 处是产物正文而非日志（dry-run 的 prompt、`token_measure`
的 TSV 表），用守卫的 `# product output` 行内标记逐个放行——比整模块豁免窄，且每次使用都是
可 grep 的显式动作。

## 7. 默认输出长什么样

3.1 分钟素材的实测输出（重定向到文件，即行模式；TTY 上同一条进度原地刷新）：

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

同一次运行改造前是 45 行、其中 25 行来自 audio-separator 的 INFO；现在 20 行，其中 4 行第三方
且全部是真警告。

## 8. 验证

```powershell
python -m pytest -q test/test_reporting.py test/test_pipeline_reporting_boundary.py
```

golden logs 覆盖 clean、resume、异常 rescue、batch 交错和非 TTY 五种场景；另有守卫钉着
「`llm/` 下不得有裸 `print`」与「线程池必须带 `initializer`」。
