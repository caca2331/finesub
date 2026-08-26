# conversational 首次真机实测：观察与收尾计划

> 2026-08-24。范围限于 `agent_session_mode` 之外的那一档——**conversational**（独立 backend
> `conversational_agent`，绑模型组启用，见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.4）。
> 接线本身已在 dev 的 `ad2cd6d`（2026-08-22 合并，conversational 那半是 `b062015`）；本文只记
> **第一次真机跑通之后**要收的尾。四档旋钮、pseudo 那一档与 task 协议本体不在范围内。

## 1. 这次实测跑了什么

测试床用 `tools.session_replay correction`（固定窗 `BV1ojjc6MEAs-0001`，303 源，`media=text`，
fixture 冻结所以不打任何搜索/生成 API），把 `correction-text/quality` 这一格绑到
`conversational-agent`。配置写在 scratchpad 的临时 `FINESUB_CONFIG_FILE`（新建 preset `conv-test`，
`local_agent_timeout_seconds = 3600`，即允许的上限），**checkout 自身没有 config.toml，未被改动**。

一个真实的 agent 通过 `finesub agent-join` 拿到 bootstrap、按 `finesub agent-task` 协议领活交活。

| 项 | 结果 |
| --- | --- |
| 入队 → accepted | 2026-08-23 22:12:22 → 23:00:08（48 分钟） |
| model | `conversational-agent`（该格只有这一个 endpoint，不存在回退） |
| harness token | 全 0（会话由对方自己的额度承担） |
| validation | `validation_ok: true`，走生产同一套 `validate_correction_window_output` |
| 产出 | 292 行 CSV，覆盖 source id 1–303，15 处两源合并，无 discard |
| 产物 | `out/prompt-iterate/BV1ojjc6MEAs-0001/conv-live/` |

**结论：这条路是通的**，协议、文件交接、validator、清场都按设计走完了一遍。下面全部是"通了之后
才看得见"的问题。

## 2. 观察

### 2.1 代码事实（本次为查证问题而确认，均已核对到行）

- `char_count` **由 harness 重算并覆盖**：`output_protocol.py` 的 `_normalized_char_count` 无条件
  用 `weighted_char_count(translation)` 替换该列，偏差超过 `2 + 20%` 只记一条 **warning**。
- 合并的 20 字 / 4 秒门槛（`HARD_MERGE_CHARS` / `ABSOLUTE_MERGE_CHARS`）在 validator 里
  **零引用**——它是 prompt 层的软门槛（[`merge-calibration.md`](merge-calibration.md)），不是机器闸门。
- 加权字数的实现权重：所有 Unicode `P*` / `N*` / 空格 / 拉丁字母 = 0.5，其余可见字符 = 1
  （`subtitles/metrics.py`）。也就是中文标点、`—`、`……`、`・` 都是 0.5，而 prompt fragment
  只写了「拉丁字母、数字和标点」。
- 租约 TTL 30 分钟（`DEFAULT_LEASE_TTL_SECONDS`），而 bootstrap 写着「不需要心跳……只要不长时间
  完全静默」——对会话式 agent 来说，一次长生成就是完全静默。
- `checkpoint_progress` 把任意 JSON 持久化进 `task["progress"]`（`agent_task_runtime.py:2042`），
  过期换 CLI 也能捞回来；它同时续租。
- 清场：`ConversationalQueue.close()` seal 后只等 `CONVERSATIONAL_SEAL_GRACE_SECONDS = 5.0` 秒就
  删整棵树，而 conversational worker 的 `await-next-task` **一轮 28 分钟**
  （`CONVERSATIONAL_WATCH_SECONDS`）。`keep_evidence` 只在 run 失败时为真。
- 窗口几何的现成旋钮：`[chunking] max_window_subtitle_tokens`（config.toml，默认 10k，`0` 关闭）。
- `submit` 的必需参数是 `--worker/--request-id/--task/--lease-generation/--input-hash`，bootstrap
  只提了后两个。
- bootstrap 硬编码 `finesub agent-task`；源码 checkout 没有这个可执行文件。
- 入队提示走 `current_reporter().warning()`，没有绑定 reporter 的环境（如 replay）会被吞掉——
  而这行是唯一告知 assignment root 的渠道。未绑定时 `current_reporter()` 返回 `_NULL`
  （`reporting.py:137`），且 `_announce()` 是**先**置 `_announced = True` **再**发
  （`agent_session_host.py:932`），所以吞一次就是永久吞。**已修（2026-08-24）**：只有 reporter
  确实送得出去才置位（`reporting.reporter_delivers`）——单纯把置位挪到 `warning()` 之后**不解决
  问题**，null reporter 是正常返回的 no-op，"发过了"和"被吞了"在调用侧一模一样。另外
  `finesub agent-join` 不带参数就能找到这棵树，提示丢了也还有救。
- **外层调用 deadline 不随活性延长**：`deadline = started + self.wait_seconds`
  （`agent_session_host.py:1017`），claim 之后一动不动。也就是说 `local_agent_timeout_seconds`
  同时承担「等人来领」和「总墙钟」两件事——一个已领活、正在 `checkpoint_progress`、明显活着的
  agent 照样会在到点被 withdraw。这次跑通只是因为把它顶到了上限 3600。**已修（2026-08-24）**，
  见下面第 4 步。
- **assignment root 不在 `finesub agent-clean` 的射程内**：默认 root 是
  `<temp>/finesub-agent-runtime/conversational/`（`client.py:1581` 硬编码），绕开了 `agent_paths`
  的按域分区；而 `agent_cleanup.py:50` 只清 `resolve_agent_episode_location().parent`，即
  `<temp>/finesub-agent-runtime/<域 digest>/`——那是它的**兄弟目录**，只有 `--all-domains`
  连根删才会带上。run 失败时 `keep_evidence=True`，整篇字幕正文就永久留在那儿。
  [`llm_local_agent_runtime.md`](llm_local_agent_runtime.md) 是「执行环境卫生」的 owner 文档，
  对这个现场只字未提。**已修（2026-08-24）**：落点归位到 `agent_paths`，该文档 §1 也补了一节。

### 2.2 owner 观察

- 那个 agent 在思考里**逐字计算译文长度**以求 `char_count` 精确，输出因此剧增。
- 48 分钟里的大部分不是排队，是**两次超出模型单次输出上限**后手动接续。
- 成功后现场被删，关键文件应当像 API 交互那样落进任务 artifact 目录。

### 2.3 agent 侧反馈（对方自述，按影响排序）

1. 长输出被反复截断，根因是试图在单条回复里写完终稿；
2. 首次 submit 撞 `StaleLeaseError`，答案已落盘，重领后原样提交即成功；
3. 手算加权字数有 73 处偏差，且「标点」边界不明确（破折号、`・`、省略号）；
4. 自写的校验脚本本身出过两次错（把 `<reasoning>` 内的文本当字幕行；统计变量 bug 造成假警报）；
5. `gap` 的方向性（本行 gap 指向下一句）在快速产出时容易写反；
6. 提交文件必须是**一个 JSON 字符串**而非对象，容易踩。

## 3. 已定的取舍（owner，2026-08-23/24）

1. **`char_count` / `gap` 继续由模型填**。让模型填这些可事后算出的量，初衷是逼它对长度与合并
   形成判断——所以**不能**告诉它"反正会被重算"。要加的是**反过度思考**：这几列是估算，写下判断
   即可，不要为小数位反复核算或逐字计数。**适用于所有 agent 模式与 API 后端**，不是 conversational 专属。
2. **写文件的分块策略不在 bootstrap 里硬性规定**。单轮输出上限因 harness 与模型而异，harness 侧
   既探不到也不该猜；引导对方按自身情况决定（甚至什么都不说）比规定一个数字更好——对方的 harness
   多半有 memory，跑过一次就知道下次怎么做。harness 要保证的是**失败可检出、进度可恢复**，不是
   替它规划回复长度。
3. **不为 conversational 开窗口几何特例**。先用第 1 条的 prompt 抑制过度思考；仍不够时，依次考虑
   调小现成的 `max_window_subtitle_tokens`、或在 prompt 层引导"想一段写一段"。窗口几何是规划期的
   全局量、还进 `_window_input_hash`，不值得为一条路引入"按谁来答决定"的耦合。
4. **成功清场要留证据**：关键文件进任务 artifact 目录，与 API 交互形状一致。
5. **取舍 2 的修订（owner 2026-08-25）**：conversational 档下**要**在 bootstrap 里给分块节奏，
   而且给具体数量——**每 80–100 条 raw 为一段**，想一段写一段。原来的"不规定分块写法"是在
   「harness 探不到对方的单次上限」这个前提下定的；第二次真机实测把前提推翻了：撞上限的地方是
   **thinking 途中**，中断处既不在文件里也不在对方上下文里，代价是整段重来，而且它自己事前
   也判断不出来。给一个偏保守的节奏比让每个 agent 自己试错便宜。**只改推进节奏，不改窗口几何**
   （那仍是取舍 3：不为这条路开窗口特例）。

## 4. 计划

按依赖排；1 与 2 是同一件事的两半，分开做没意义。

**但下一次真机实测卡在别处**：第 3 步的"保 sealed index"（5 秒 grace vs 28 分钟一轮 = **必然**
撞空）、第 6 步的 `_announce` 与发现模式，都是几行的确定性缺陷，且直接决定第二次实测能不能顺利
起跑。建议它们跟第 1 步一起走，第 2 步紧随；第 3 步的 artifact 拷贝那半、第 5 步照常排后。

> **2026-08-24：第 1–4、6 步全部落地。** 现状写在 [`llm_local_agent.md`](llm_local_agent.md)
> §12.1.4 与 [`llm_local_agent_runtime.md`](llm_local_agent_runtime.md) §1，测试在
> `test_llm_agent_conversational.py`（22 个用例）。
> **第二次真机实测（2026-08-25，v78c）**：单窗 303 源、走生产 validator **一次过**（0 修复轮）、
> harness token 全 0，产出 279 行、24 处合并、无 discard；只有 4 行 char_count 越容差且全是低估
> ——对方**不再逐字数数**了，而这正是第一次实测里最贵的那个行为。清场按新规矩留下了墓碑
> （只剩当前代 state）与 `evidence/call-0001/` 四份文件，共 198 KB。
> 两个新问题记在下面 §6。
>
> **第 5 步定案（2026-08-25）**：三次进共享输出契约的试作全部退回 v77；那句话改为**只按传输追加**
> ——conversational 路的纠错 task 才拿得到（`fragment_conversational_correction_effort_v1.md`，
> 见 [`llm_local_agent.md`](llm_local_agent.md) §12.1.4）。理由是它要防的行为（逐字数数、另写脚本
> 验算）只有带工具的 agent 才付得起，而放进共享契约会让 REST 模型平白多想 55%。
>
> **三次试作的经过**：v78（含糊版"是估算"）在 3.5-flash-lite 上把 validation-ok 从
> 3/10 打到 1/10；改到带思考的 agy 3.7-flash、措辞收紧成明确禁令的 v78b，则把 **thinking 翻了
> 一倍**（16.0k → 32.4k，每臂 n=7），方向与意图相反。都已回退到 v77，两条负结果与"下一次该怎么
> 试"记在 [`prompt-iterate.md`](prompt-iterate.md) §5。

### 第 1 步：bootstrap 模板改正确 —— **已做（2026-08-24）**

落点 `prompt_templates/agent_worker_bootstrap_v1.md` + `agent_transports.conversational_bootstrap`。
不动 `PROMPT_VERSION`（该模板只服务 agent 端，不进纠错 prompt 的版本）。

- `submit` / `checkpoint-progress` 的必需参数写全；
- 命令名按**实际入口**生成，而不是硬编码 `finesub agent-task`；
- 提交格式给一行具体示例（JSON 字符串 vs 对象）；
- 点明两件对方无从得知的事：**每条控制命令都会续租**（长时间不发命令则租约会过期，过期后
  `status` → 重领 → 原样重交即可），以及 **`checkpoint-progress` 可以存任意 JSON 进度**，
  换 CLI 也能捞回；
- 按取舍 2：**不规定**分块阈值或写法，只说明"答案通过文件提交，可以分多次写好再一次提交"这一
  形状是允许的。

验收：现有 `test_llm_agent_conversational.py` 的脚本化宿主用例照旧通过；模板渲染出的命令行在源码
checkout 下可直接执行（这次的踩坑点）。

### 第 2 步：`agent-task lint` 与 `submit --text-file` —— **已做（2026-08-24）**

- `lint`：用同一套 validator 校验一份候选答案，**不动任何状态**，返回与 submit 相同的
  `validation_errors`。这是整套里唯一不依赖对端自律的一环——截断、缺列、覆盖不全在提交前就现形，
  重试只重做出问题的那一块。headless / pseudo 同样受益。
- `submit --text-file`：直接读原文，免掉把几十 KB 答案转义成 JSON 字符串这一步。

验收：新增用例覆盖"lint 报错 → 修正 → submit 通过"，以及 `--text-file` 与 `--json-file` 等价。

### 第 3 步：清场留墓碑 + 关键文件进 artifact —— **已做（2026-08-24）**

落点 `ConversationalQueue.close()` + `write_agent_session_usage` 那个 seam（artifact 目录在那里
已是已知量）。

- accepted 后把 protocol / context / answer 与一份控制摘要（worker id、领取次数、修复轮数、
  lease reset 次数、时间）拷进 `<artifact_dir>/agent-conversational/<task_id>/`；
- 删掉正文，但**保留 sealed 的 `control/index.json`**，让仍停在 `await-next-task` 的 worker 能读到
  `assignment_complete` 而不是撞上空目录（现状 5 秒 grace vs 28 分钟一轮，等于必然撞空）；
- ~~顺手把默认 root 从 `client.py` 硬编码的 `<temp>/finesub-agent-runtime/conversational/` 挪进
  `agent_paths` 拥有的按域分区~~ **已做（2026-08-24）**：落点是
  `agent_paths.conversational_assignment_parent()`，普通 `agent-clean` 现在够得着；
  [`llm_local_agent_runtime.md`](llm_local_agent_runtime.md) §1 补了这个现场的位置与留存规则。

验收：新增用例——worker 停在 watcher 里、run 成功清场后仍能拿到 `assignment_complete`；artifact
目录下拿得到三份文件与摘要；失败留下的 root 落在当前域的 parent 下，普通 `agent-clean` 能清掉。

### 第 4 步：把两个时钟分开 —— **已做（2026-08-24）**

原本只写「把 lease TTL 提到调用期限量级」（`start_assignment(lease_ttl_seconds=…)` 传参即可，
理由是这条路上"过期回队"换不来任何东西——调用本身在 deadline 就死了，回队只让对方白做一轮）。
查证 §2.1 之后这条要扩成两半，因为**外层 deadline 根本不看活性**：

- **claim 前**：`local_agent_timeout_seconds` 管「等不等得到人来领」，语义不变，手册里也是这么
  讲的；
- **claim 后**：改由租约活性兜底（每条控制命令续租，第 1 步正要把这点写进 bootstrap）。一个在
  `checkpoint_progress` 的 agent 不该因为总墙钟到点被 withdraw，而一个死掉的 worker 仍会在一个
  TTL 内被回收。

这样 TTL 该定多大也就清楚了：**覆盖一次长生成的静默**，而不是覆盖整通调用。两者一起改，单独
调哪一个都会留下另一半的坑。

### 第 5 步：prompt 两句（单独走 prompt-iterate）—— **2026-08-24 试作，未采纳**

落点输出契约 fragment。动 `PROMPT_VERSION`，按 [`prompt-iterate.md`](prompt-iterate.md) 在固定窗上
做对照，臂间 n≥5 并加一次同配置复跑作噪声基线。

- **反过度思考**：`char_count` / `gap` / `duration` 是给自己判断长度与合并用的估算，写下判断即可，
  不要为小数位反复核算、逐字计数或另写校验；
- **把「标点」写实**：现有 fragment 说的「拉丁字母、数字和标点」那半**不能丢**——它们确实都是
  0.5。要补的是中西文标点与空格同样 0.5（含 `—`、`……`、`・`、`、`、`%`）。
  但**别写成「全部标点」**：实现的分界是 Unicode 类别，`P*` 计 0.5 而 `S*` 计 1.0，于是
  `+` `=` `→` `♪` `$` 都是 1.0——而多数人会把它们叫标点。写实就得举出这组反例，否则含糊照旧，
  而规则含糊本身就是过度思考的诱因。

看点不只是质量不掉，还有**输出/thinking 是否真的降下来**——那正是可量的部分。

**实测结果：两轮都没兑现，改动已回退。**

**第二轮（2026-08-25，按 owner 意见改到 agy 3.7-flash api 档、措辞收紧成明确禁令）**：每臂 n=7，
thinking 均值 16,025 → **32,412（2.02×）**，中位 13.6k → 31.2k，区间几乎不重叠；可见输出与
质量闸门无差别。**禁止模型在思考里推演某个量，反而让它在那个量上想得更多**——"不要想 X"把 X
抬成了显著话题。要减少这里的思考，得让它**不值得想**（降精度要求或不让模型填），而不是禁止想；
但那与取舍 1「继续由模型填以逼出长度判断」冲突，所以这条路先封闭。

**第一轮（2026-08-24，3.5-flash-lite，含糊版"是估算"）：** 固定窗 + `3.5-flash-lite` 各 10 次尝试：
v77 得 3/10（刚好达标），v78 得 1/10；两臂输出 token 几乎相同（128.2k vs 129.6k），
所以**省不出输出**，而结构合规率掉到验收线以下——失败里压倒多数是 `Row N has too few fields`。
可信的机制是：告诉模型这几列「不必仔细」，它连把这几列**写出来**都变松了。详见
[`prompt-iterate.md`](prompt-iterate.md) §5 的条目（含重启这条改动的拆分方式）。
两个附带观察：一是 lite 写的 `char_count` 系统性偏大，说明 0.5 加权规则在它身上本来就没落地；
二是**这个模型 `thinking_tokens` 恒为 0**，所以"过度思考是否减少"在 lite 上根本量不到——
那半要么上带思考的 3.7-flash（超出常规测试授权，需另行确认），要么就在 conversational 真机上看。

### 第 6 步：`agent-join` 的无参发现模式，与两个小项 —— **已做（2026-08-24）**

`root` 现在是必需位置参数（`agent_join.py:24`）。给它一个**无参模式**：扫 conversational parent
下有 `control/index.json` 且未 sealed 的树，唯一就直接 join，多个就列出来让人挑。一步解掉三件事：

- 手册里那段「目录名每次都不一样，所以没法提前 join、只能守着」的根因——人可以先跑起来，等
  想起来再 `finesub agent-join`；
- 提示被吞（无 reporter 环境）之后无从找回；
- 目录名与 assignment id 是两个不同的 `conv-<hex>`（`client.py:1584` 与
  `agent_session_host.py:891` 各 uuid 一次）——统一成同一个之后，发现模式还能直接把 assignment id
  报出来。

剩下两个小项：

- `_announce()` 先置 `_announced` 再发 warning（§2.1），顺序反过来；要么在 replay 这类入口绑一个
  reporter，要么让未绑定时的 warning 落 stderr——**别在这里裸 `print`**，AST 守卫钉着；
- [`manual/agent.md`](manual/agent.md) §5.1 补一句：窗口大到撞上你 agent 的输出上限时，可调小
  `[chunking] max_window_subtitle_tokens`。

## 5. 明确不做

- **validator 强制 `char_count` 或合并门槛**。软门槛是标定过的结论，变硬是另一个决定。
- **worker 能力对账**（owner 2026-08-22 已定不做）。这次的产出质量反过来印证了它：能力由用户
  负责，validator 兜底就够。
- **按后端/档位决定窗口几何**（取舍 3）。
- **默认把这条路的变体换成不带 reasoning 的 basicB**。它确实能砍掉最长的连续输出块，但推理先行是
  capableC 的设计核心，属于质量决定，要试就进 prompt-iterate 的对照，不在本计划内。

## 6. 未决

- **真机实测的覆盖面**：这次只验了单 task、单 worker、无修复轮。多 task 连续领取、修复轮往返、
  两个 worker 同时在场都还没跑过。[`llm_local_agent.md`](llm_local_agent.md) §13 的措辞按已跑到的
  部分更新，别一次写满。
- **bootstrap 曾谎称"每一条控制命令都会续租"（2026-08-25 已修）**：实际只有带
  `--task`/`--lease-generation` 的命令续租（`lint`/`submit`/`checkpoint-progress`/`next-task`/
  `web-*`），而 `status`/`rehydrate`/`await-next-task` **不续**——偏偏 bootstrap 教的第一步就是
  `status`。第二次实测里对方埋头干了 22.5 分钟、租约只剩 7.5 分钟，是人工 `checkpoint-progress`
  救回来的。模板已改成点名哪些续、哪些不续，并建议"边写边 lint"（既查错又续租）。
- **`local_agent_timeout_seconds` 在 replay 路径上没被带进队列**：临时配置写了 3600，单独
  `load_execution_settings()` 也读得到 3600，但跑起来的队列拿到的是默认 900（由租约 TTL
  `max(30min, wait)` = 30min 反推）。不影响本次结论，但要查。
  **顺带验证了两个时钟**：wait 是 900 秒而这次调用活了 56 分钟才收到答案——换成旧的单一墙钟，
  它早在 15 分钟前就把对方撤单了。
- **`local_agent_timeout_seconds` 的默认值**：这次 48 分钟里大部分是截断重试，不能据此判断 900 秒
  够不够。但真正要问的不是"默认值够不够大"——第 4 步把两个时钟分开之后，这个值只再管「等人来领」
  那一段，量级判断因此要重新起头。等第 1、2、4 步落地后再看一次真机耗时；上限 3600 秒同样重估。
