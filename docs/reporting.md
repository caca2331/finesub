# 上报与日志：`finesub.reporting` 的契约

`src/finesub/reporting.py` 的 owner 文档。谁在什么时候说话、说给谁听、以什么级别落到哪个前端。

管线代码**从不直接 `print`**：所有面向人的输出都过 reporter。这不是风格约定——桌面 worker、
batch、CLI 终端、落盘 run 日志是四个不同的渲染面，绕过 reporter 的一行 `print` 只会到达其中
一个，而且通常不是需要它的那个。

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
- **`summary` 收带标签的数字，不收一句话**。只有 stage 知道那 7 段叫「噪声片段」，只有
  renderer 知道这次运行有没有终端可以重画。零值与 `None` 由 renderer 丢弃，stage 照常报全。
- **`stage` 参数收的是 key，不是中文**。中文由 renderer 经 `STAGE_LABELS` 映射；传中文标签
  会让 `[n/N]` 前缀查不到。阶段名以桌面端 `desktop/frontend/lib/translations.ts` 的 `stages`
  为准，CLI renderer 复用同一套——同一次运行被两个前端叫成两个名字是支持成本，不是风格问题。

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
| `FileReporter` | 落盘 run 日志，**恒定 verbose**，见 §5。全仓只有 `pipeline.py` 一处构造 |
| `FanOutReporter` | 终端 + 文件。任一 renderer 抛异常不影响运行——上报是旁白，不是工作 |
| `WorkerReporter`（`desktop/backend/worker/main.py`） | 桌面。转成 UI 事件 + `task-log.txt` |

**batch 是唯一需要行前缀的场景**：多个任务共用一个终端，可原地重画的进度行会被最后说话的
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

**VAD / ASR**

- group 进度只在越过下一个 5% 档位时上报；恢复 checkpoint 时立即上报当前起点。节流放在
  **调用点**而不是只靠 renderer——renderer 只管终端，而桌面端会把每次上报变成一个事件，
  事件量的上界必须在源头就有。**计数单位是 interval 而不是 group**：group 总数要等分组跑完
  才知道，分母不能是它。
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

**阶段边界不归这一层**：`pipeline.py` 已为每个阶段（含 `translated-srt`）发过 `stage_started`，
再发一次会重开阶段行并重置进度节流。

| 事件 | 时机 / 位置 | 级别 |
| --- | --- | --- |
| `debug("correction plan", {windows, reused_plan, driver})` | `run.py` 规划与 refit 之后 | debug |
| `progress("translated-srt", …, unit="windows", detail=chunk_id)` | 每个窗口单元完成（`stages/correction/progress.py`）。单位用英文，与既有阶段的 `blocks`／`intervals` 一致 | normal |
| `debug("correction window attempt", …)` | `attempts.py` 每次重试 | debug |
| `debug("correction window validation failed", …)` | 进入修复轮时 | debug |
| `debug("rate limit wait", {scope, seconds})` | 上报点在 `rate_limit.py`：`scope` 只有它内部知道，且能同时盖到 `llm_runtime.py` 两个调用点 | debug |
| `debug("research round", …)` | `research.py` / `search_loop.py` 每轮 | debug |
| `warning(…)` | **举例，不是词表**：知识库仓库不可用、worktree/缺 git 跳过知识库更新、key 池超建议值、target 不支持视频而降级 | normal |
| `gemini-upload-retry` / `gemini-upload-failed` | Files API 上传的重试（与模型调用的重试预算分开）：每次重试一条，写阶段（`upload`/`state_poll`/`token_poll`）、attempt/max 与等待秒数；只在重试过之后仍失败才发 `failed` 那条。正文只写异常类型或 HTTP 状态，**不写** resumable session URL（它是 capability token）与 key | normal |
| 内容过滤阶梯 | 判据看 `LadderOutcome`：`level <= 0` 且 `dropped_units` 空＝纯重试通过发 `debug`；**丢了注入单元**（证据/词条，不是源字幕）发 `warning` | 两者皆有 |
| `summary("translated-srt", …)` | 阶段收尾。报窗口、拆窗、调用、重试、修复轮、内容过滤恢复次数与按 tier 的调用数 | normal |

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

### 5.2 两条没做主的产品决定

都不是遗漏，是没人拍板：

- **桌面要不要显示逐窗进度。** `WorkerReporter.progress()` 至今是空实现，注释写明是刻意的
  （UI 显示阶段，逐条计数会变成任务日志里的几百行）。改它属桌面侧。
- **batch 要不要也写 run 日志。** 今天没有——`FileReporter` 全仓只有 `pipeline.py` 一处构造，
  而 batch 正是 LLM 任务排队跑的那一档。

## 6. 落盘的 run 日志

终端之外，每次运行往 `<user-data>/logs/run-<时间戳>-<输入名>.log` 写一份**恒定 verbose** 的
记录，与 `--log-level` 无关：终端保持好读，决策细节留在盘上，用户报障时直接把文件发来即可。

三个刻意的选择：

- **一次运行一个文件，不是共享一个文件。** 桌面 worker 与 CLI 可以同时在跑
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
与桌面的 install/session 日志共用预算。

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
