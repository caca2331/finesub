# Agent 执行环境：episode 落点与 capsule 生命周期

> 本文从 [`llm_local_agent.md`](llm_local_agent.md) 拆出（2026-08-16，原文 2,198 行），
> **章节号已按本文重编、从 1 起**——每份文档独立编号，才能在中间插入新节。跨文档引用
> 一律写成 `` `文件名` §N `` 的形式，由 `test_doc_links.py` 守着：指不到的章节号会红。
>
> 本文是拆分前第 15 节的**运维半边**；同一节的准则与实测在
> [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md)。

## 1 执行环境卫生：episode 落在哪（G-C，已实施）

episode 的落点分两档（**开头就分，别只读这一段就接路径**）：

- **打包安装**：capsule 根成为大文件根下的第四个目录，与 `models`/`cache`/`tasks` 平级
  （`paths.BIG_DATA_NAMES` 加一项），随 `relocate` 搬、随默认档卸载；
- **仓库 / worktree**：**不能**用大文件根——那里的大文件根就是 checkout 本身，而 Codex 会自动
  吃掉 cwd 之上的 `AGENTS.md`（§2 门二实测）。**定为 `%TEMP%/finesub-agent-runtime/<域
  digest>/`**，不随打包安装的 relocate 移动。理由见下方「为什么 checkout 档留在 TEMP」；
- 两档都由 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施) 那个统一的 location resolver 解析，调用方不自己拼路径。

**选哪一档的判据已写死。** 不能留给调用者猜，因为有一个配置会让两个显而易见的
判据给出不同答案：`FINESUB_CHECKOUT_DATA=0`（`resources.md` 记载的"让仓库运行接到共享数据"）。
此时按"是不是源码 checkout"判会选 `machine_temp`，但该档的 anchor 是"该 checkout 的 `.state`
文件"——而这个开关下 `.state` 已经解析到托管位置，anchor 不存在；按"数据根在哪"判则会选
`managed_big_data`。**定为按数据根判**（与 `resolve_state_file()` 同源），anchor 随之取实际解析
到的协调根；这样 `FINESUB_CHECKOUT_DATA=0` 的 checkout 与打包安装落进同一档，符合该开关"接到
共享数据"的本意。

打包安装那一档由此自动获得三件事：

- `finesub relocate` 一并搬走，不用单独处理；
- 卸载**按大文件目录的既有规则走，不做特例**：大文件根在默认位置且未共用时，随默认档一起删
  （与 `cache` 同级——证据不是成品，`--purge-tasks` 之前就该没了）；**已搬走或与别的安装共用
  时不删**，与 `models`/`cache`/`tasks` 一致，要 `--purge-big-data`。
  不特例的理由：打包档的协调域是**所有安装共用的那一份 `user-data`**（`resources.md`），所以
  共用的大文件根里躺着的是**别的安装的失败现场**；因为本安装要卸载就把它删掉，是在毁别人的证据。
  代价要说清楚：**"卸载即清干净"在这两种情况下不成立**，capsule（含字幕正文）会留下。因此
  **卸载输出必须点名它还在哪**，并指向 `agent-clean`——这也正是那条显式清理命令不可省的原因；
- `locations.json` 记录位置，用户"简单地清理"就是删这一个目录。

配套：resources.md **两节都要写**——打包安装那节写大文件根下的第四个目录，「从仓库源码运行时」
那节写 `%TEMP%/finesub-agent-runtime/<域 digest>/`；都要说明位置、量级、留存规则（见下）以及
清理命令怎么用。

**清理命令已新造。** 共享实现是 `finesub.llm.agent.agent_cleanup`；打包入口是 `finesub agent-clean`，checkout
入口是 `python -m finesub.llm.agent.agent_cleanup`。它不走 managed runtime provisioning，因此运行环境缺失或损坏时仍可用。

按仓库既有形状定：

| 位 | 定法 |
| --- | --- |
| 共享实现 | `python -m finesub.llm.agent.agent_cleanup`——**定案，见下方「resolver 归属」**。它要用 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施) 的 resolver，而 resolver 要用 `finesub.paths` 的 checkout 检测；把它搬进 `finesub_bootstrap` 会逼出第二个检测器 |
| 打包 CLI | 加 `finesub agent-clean`，**但不能走 `run_in_runtime()`**——它第一句就是 `ensure_ready()`，会为了删几个文本文件先装运行环境、下 ffmpeg，而且运行环境损坏时清理反而用不了（那恰恰是有人要清理的典型时刻）。改成在 **shell 自己的解释器**上以子进程跑 vendored 的 `finesub.llm.agent.agent_cleanup`（不 import，保持分层；不 provisioning） |
| checkout | `python -m finesub.llm.agent.agent_cleanup`，与 `python -m finesub.llm.correction_translation` 同级。**仓库根包不再声明 `[project.scripts]`**（2026-08：对外只留 CLI wheel 的 `finesub`），所以 checkout 一律走 `python -m` |
| 作用域 | 默认**只清本域**；`--all-domains` 才扫未分区旧根与其他域，见下 |

**子进程找不到 vendored 模块——必须显式给 `PYTHONPATH`，而且是两条路径。** 薄 CLI 只是把
`_vendor/src` 插进**当前进程**的 `sys.path`（`_ensure_vendor_on_path()`），新起的 Python 子进程
不继承它。直接复用 runtime 已有的组合规则（`environment.py` 里 `worker_context` 那段），别自己发明：

```text
PYTHONPATH = <app_source>  ;  <app_source>/src  ;  <原 PYTHONPATH>
<shell 自己的 python> -m <cleanup 模块> ...
```

两条都要：包在 `<app_source>/src/<pkg>`，只给 `<app_source>` 找不到。也不能指望 cwd 恰好在
vendor 树里。

**Python 3.10 是硬约束。** 薄 CLI 是 `requires-python >=3.10`，主应用是 `>=3.12`；用 shell 自己
的解释器，就意味着 cleanup 模块**及其全部传递导入**受 3.10 约束。已知地雷：
`finesub.config` 用了 **`tomllib`（3.11+）**。resolver 真正需要的 `finesub.paths`
只 import stdlib，所以路是通的。

**`-m` 必然先执行包的 `__init__`——上一版写的"走独立入口而不经包顶层"是错的，没有那条路。**

**定案：清空 `finesub/llm/__init__.py` 的 re-export，resolver 与 cleanup 都留在 `llm`。** 加一条
import-free 守卫测试（照 `test_secrets.py` 守 `finesub_bootstrap.__init__` 的写法）。

为什么不是"搬进 `finesub_bootstrap`"：那条路要求 resolver 能做 checkout 检测，而检测在
`finesub.paths`（更高层，bootstrap 不该反向 import）。上一版写的"在该模块内用几行 stdlib
自己判"**是错的，而且错得很典型**——主 checkout 与 worktree 归一正是 domain identity 的地基，
run 和 cleanup 各有一套检测，任何细微分叉都会让 cleanup 算出不同的 digest，**恰好打碎这几轮
建立起来的那个不变量**。这就是本节开头那句"两处各写一遍就会漂移"的同一个错误，换了个地方。
要么把 checkout 检测整体下沉、让 `finesub.paths` 委托它（结构方案，代价大得多），
要么根本不要第二个检测器——取后者。

代价核对过：`finesub/llm/__init__.py` 现在只做 re-export，**全仓库没有任何地方 import 这些符号**
（只有 `from finesub.llm import <子模块>`，与空 `__init__` 无冲突），没有兼容负担。清空之后
`-m` 的包顶层就是空的，3.10 契约只落在 cleanup 模块自己的传递导入上。

**import 契约**：cleanup 模块与它的**全部传递导入**只许 stdlib 与路径解析；不得拉进 httpx /
torch / provider 客户端 / `finesub.config`。清理**不能依赖 managed runtime 是否健康**，
这是它的定义属性。

（顺带更正：`finesub_bootstrap` **不是** stdlib-only 包——它自己的说明写明整体需要 `pydantic`
与 `httpx`，`secrets.py` 是刻意的例外。准确的说法是：它的 `__init__` 必须 import-free，而
cleanup 用到的那几个具体模块——`paths`、`locks`——必须维持 3.10 与轻依赖契约。）

**验收（已进 CI）**：`ci.yml` 的 `thin-cli-py310` job **在 Python 3.10 上装薄 CLI wheel，在没有
managed runtime 的情况下执行 `finesub agent-clean`**，断言它能跑、不 provisioning。3.12 上跑的
测试**抓不到** `tomllib` 这类问题，只有它能。

（另一条路是把薄 CLI 的下限提到 3.12。**不建议**：薄 CLI 的职责就是在用户**已有**的解释器上
把 managed runtime 装起来，抬下限正好砍掉它存在的理由。）

**`--all-domains` 无法自证前提，就不要假装能。** 各域用各自的 `activity_root`，旧版本更是完全
不发租约，所以这条命令**证明不了"所有 agent 已停"**。定为**危险维护操作**：先打印将要删除的
完整清单，再要求交互确认或显式 `--force`；文档写明**停掉 agent 是操作者的责任**。不引入所谓
机器级 gate——它照样覆盖不了旧版本，只会制造"已经验证过"的错觉。默认的本域 cleanup 不受影响，
它只删自己那棵子树。

**不能只做成 `uninstall` 的一个选项**：计划已经定死 checkout 不走 uninstall。

**历史实现（G-D 后已退场）。** 此前把锁推迟的理由——"锁需要对手方，而 cleanup/relocate 还不
存在"——是错的：对手方一直都在，就是**另一个正在跑 agent 调用的 FineSub 进程**，而且
`finesub relocate` 早已存在并且已经持 activity barrier + install lock。实测无锁时 6 线程同时
认领同一个未标记根，**每次都产生 2 个不同的 store_id**，落败方从此认为自己不拥有那个本就属于
它的根。

G-D 之前 claim / prune / create 各自整段跑在 capsule root 锁里；以下只记录为何那版锁必须这样放，
**不是当前运行时契约**：

- **锁文件不在 capsule 根内**，否则它会被它本该排除的那两个操作（relocate 搬走、cleanup 删掉）
  连带处理。它是 **parent 的兄弟文件**（`<parent>.lock`），只由 parent 推导。
  （第一版放在机器本地 state 文件旁并按 root digest 命名，那是错的：state 文件**按 checkout**
  走而 capsule 根是**机器级**的，等于一棵共享树配了每个 checkout 一把互不相识的锁；而且每见到
  一个新 root 就往仓库根目录扔一个文件。已由 `9902c71` 修复。）
- **加锁顺序**：activity barrier 与 install lock 永远在外层，capsule 锁最内层、不等待任何外层
  锁。`finesub relocate` 要动 capsule 根时按同样顺序补取这把锁即可，不会与 agent 调用死锁。

G-D 已把这把锁、逐次 `rmtree` 复验与 24h/50 个保留策略全部删除。当前只有用户发起的 root-wide
cleanup/relocate/uninstall 取 activity barrier；普通调用只删除自己原子创建并记下的精确路径。

**看上去的冲突，以及它为什么不是冲突。**
`_reject_repository_or_reparse_root()` 拒绝把 runtime 根放在任何仓库里（向上找到 `.git` 就抛），
而**仓库/worktree 模式下大文件根就是 checkout 本身**（`checkout_data_enabled()` 默认 True）。

这个守卫**不是在防 agent 读到仓库**。driver 自己就记着 `"read_isolation": False`，
followups 里也早写明「只读 sandbox 挡写不挡读」：agent 能不能读到仓库与 capsule 放哪儿无关，
放 `%TEMP%` 一样读得到。写则由机制挡住（Codex 只读 sandbox / Claude Code 拒绝全部写工具），
同样不靠位置。

它真正兜的是**我们自己的递归删除**。当时 `AgentCapsuleManager.prune()` 对 `runtime_root` 下的
**任意**子目录动手——不校验 manifest、不校验 id 形状——只要早于 24h 且不在最新 N 个之内就
`shutil.rmtree`；`runtime_root` 一旦被指到有内容的地方，清理就会删掉别人的东西。reparse point
那半是同一类事故：`rmtree` 跟着 junction 走出预期目录。（下面第 1 条已修；整套判定按 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施)
还会整体退场。）

历史过渡顺序如下；这些 prune 所有权判据已随 G-D 删除：

1. ~~prune 只删自己认得的 episode~~ **已做**：识别条件是"名字符合 `<task>-<32 位 hex>`
   **且**（有我们写的 manifest **或** 五个 capsule 子目录齐全）"，后半段是为了让 `create` 崩在
   半路留下的骨架仍可回收；同时跳过 reparse point，避免 junction 把删除重定向出去。误配
   `runtime_root` 现在最坏是"什么都不删"。
2. **位置守卫改成"只放行 canonical 路径"，不是原样保留**——但它**不是**可有可无的防御
   纵深（我一度这么写，错了）：实测证明它是目前唯一挡住 Codex 吃掉用户 `AGENTS.md` /
   `CLAUDE.md` 的东西，见 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施) 门二。`.git` 祖先检查会拒掉 checkout 模式
   下的大文件根（那里的大文件根就是 checkout 本身），所以必须明确：由 `paths.py` 推导出的
   `<big-data>/agent-capsules` 放行，调用方自定义的仓库内路径照旧拒绝。判据是"这个路径是不是我们
   算出来的"，不是"它在不在仓库里"。reparse 那半原样保留——它防的是 rmtree 走偏，仍然值钱。
   仍然拒绝任意仓库路径的理由：capsule 根是 agent 的 cwd，落在用户仓库里会让供应商那些"从 cwd
   往上找项目文件"的机制（CLAUDE.md 自动发现、`--skip-git-repo-check` 想跳过的那类检查）指向用户
   的源码树。这与"能不能读"无关（读本来就没隔离），是"默认锚点指向哪里"的问题。
3. **checkout 模式能否用大文件根，取决于 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施) 门二——而门二已实测未过。** 打包安装的大文件根
   不在仓库里，放行；**仓库/worktree 模式改用 `%TEMP%/finesub-agent-runtime/<域 digest>/`**
   （本节开头已定，此处不再是二选一），因为那里的大文件根就是 checkout 本身，而 Codex 会自动
   吃掉 cwd 之上的 `AGENTS.md`。第 2 步的 canonical 放行只解决"我们自己的 rmtree"，解决不了这条。

第 1 条曾作为过渡修复单独落地，现已由 one-shot 删除语义取代。

**全局配置隔离。** 目标是 agent 的**行为**只由本次任务决定，不继承这台机器的个性化配置（规则文件、MCP、插件、
hooks）；**连到哪、用哪个号**则跟随用户，Codex 的做法见 [`llm_local_agent.md`](llm_local_agent.md)
[§14 当前实现与迁移注意事项](llm_local_agent.md#14-当前实现与迁移注意事项)。用户级的**全局规则文件**只有 Claude Code
（`--setting-sources ""`）与 dsh（关掉 `agent-instructions`）挡得住；Codex 在凭据走环境变量、开了 harness home 时也挡得住，
其余情况下的 Codex 与 CodeBuddy、agy 照样注入，靠开场兜底句（2026-09-30 实测）。当前实测
（Claude Code，`--safe-mode` + `--setting-sources ""`）：`skills: []`、`plugins: []`、
`mcp_servers: []`、`slash_commands: []`；**`CLAUDE.md` 自动发现已于 2026-08-13 实测确认关闭**
（工作目录的与上层的都不加载，见 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施) 门二的表）。**仍需确认**：`agents` 字段仍列出内建 agent
（`Task*` 已被拒所以调不动，但值得确认无旁路）。Codex 已测——它**会**加载，见门二；agy 在
显式 `--new-project`/`--project` 下也会加载 `AGENTS.md`，未绑定 project 时则不可靠。

**任务级说明文件。** owner 提议（2026-08-13）：在 agent 的工作目录里放一份专门写的
`CLAUDE.md` / `AGENT.md`，用供应商原生的机制交付任务约定，而不是把所有约束堆进 prompt 前缀。

**实测结论：三家行为不一致，这条路走不成统一机制。** Codex 会读工作目录里的
`AGENTS.md`/`CLAUDE.md`（实测命中），Claude Code 在 `--safe-mode` + `--setting-sources ""` 下
**两者都不读**（工作目录的和上层的都不读）；agy 只有绑定 project 后读 `AGENTS.md`，不读
`CLAUDE.md`。所以采用供应商原生规则会让实际约束分叉。**结论：当前 one-shot 维持 prompt 前缀；
长驻 runtime 改用 §7 的显式 control namespace + `read_context`，不要把正确性建立在规则发现上。**

### conversational 的 assignment 现场（2026-08-24）

一次绑到 `conversational-agent` 的 run 还会留下第三类现场：assignment 树。落点是
`agent_paths.conversational_assignment_parent()`，即 **episode parent 下的 `conversational/`**，
每个 run 一棵 `conv-<hex>`（目录名就是 assignment id）。

- **装的是什么**：这条 run 交给对方 agent 的全部字幕正文、协议与控制台账。与 capsule 同一
  证据等级，因此同一条规则——**成功即删**（`ConversationalQueue.close()`），失败或整条 run 抛错
  才整棵留下。
- **谁来清**：普通 `finesub agent-clean`。它删的是当前协调域的 episode parent，`conversational/`
  在那底下，所以一条命令连 capsule 带 assignment 一起走。
  **这正是 2026-08-24 修掉的问题**：此前落点硬编码在 `client.py` 的
  `<temp>/finesub-agent-runtime/conversational/`，是域目录的**兄弟**而不是子目录，普通
  `agent-clean` 够不着，只有 `--all-domains` 连根删才清得掉——失败留下的整篇正文因此永久堆积。
- **搬盘/卸载**：跟着 episode parent 走，没有单独的语义。

## 2 capsule 是一次性 episode，不是持久 store（G-D，已实施）

**为什么不扫描、不认领、不 prune。** 曾经有过一套所有权机制（marker → store_id → 按内容认领
→ 跨进程锁），它一路长出来是因为一个前提：

> `prune()` 删的是**扫描发现的**东西，不是**自己创建并记下路径的**东西。

只要如此，就必须在盘上证明所有权，而证明要写、要读、要跨进程一致。换成「只删自己这次原子
创建并记下路径的那一个目录」之后，整个证明链就不需要了——运行时不扫描、不认领、不清扫别人的
episode，清理是一条显式的用户命令。**这是当前契约，不要再往回长。**（那五轮的经过留在维护者的本地归档里。）

### 事实基线

- **capsule 在调用期间是有用的**：Codex 的 cwd 与 `--cd` 指向它，stdout/stderr 泵进它，
  `_normalize()` 从 `raw.jsonl` 读事件，非零退出分类读 `stderr.log`，结果先落
  `staging/result.txt`。**要退掉的是它「长期、共享、可扫描」的身份，不是目录本身。**
- **它和 `exchanges/` 只在成功路径上重复**。`exchange_logger.log(...)` 在
  `client.complete(...)` 返回**之后**才写（`stages/correction/attempts.py`），所以
  **transport 一失败就没有 exchange**，raw events 与 stderr 只存在于 capsule 里。这正是
  「成功即删、失败留存」的真正理由——不是「capsule 是副本」。
- `ExchangeLogger` 本身是可选的：没有 `task_artifact_dir` 的直接库调用不产生 exchange。
  这与 API 路径一致，不额外承诺。

### 目标形态

```text
AgentEpisode
  ├─ mkdtemp 在专用 parent root 下原子创建唯一目录
  ├─ 调用期间存放 input / 事件 / stderr / staging 结果
  ├─ transport 成功 → 删除自己创建并记下的那一条精确路径
  │    └─ 例外：harness 会话（带 completion 谓词）而 runtime 始终没 accept
  │             → 标记后留下，进独立的滚动池（见下文）
  └─ transport 失败或进程被强杀 → 原样留下，那是唯一的失败现场
```

**「成功」的边界定义为 transport 成功**：进程正常退出、事件完成规范化、无 transport policy
violation、提取到有界的最终文本。**不是**「字幕通过了业务 validator」。返回了格式错误的 CSV
仍算 transport 成功——exchange **将会**记下完整应答，harness 自己去 repair，capsule 不必留。
这样 episode 的生命周期不会渗进 correction / research / knowledge 各个上层调用点。

**一个刻意接受的审计空窗**：删除发生在 driver 返回之前，而 exchange 在 `client.complete()`
返回之后才写。进程正好崩在这两点之间，两边都不会留下东西。接受它，因为 API 路径本来就是同样
的保证，而 capsule 没有任何真实的恢复逻辑值得为它建一套事务提交协议。**但不能写成「exchange
已经记下」**——落盘发生在删除之后，不是之前。

**harness 会话下「exchange 会记下完整应答」这个前提不成立**（2026-09-16）。上面「capsule
不必留」的理由是 exchange 会记下完整应答；对 completion 传输成立，因为最终文本就是产物。tool
session（以及会话宿主的 CLI）的最终文本**没有产物语义**（产物只经 `submit` 交付），于是一个
**没被 accept 就干净退出**的会话：transport 成功 → capsule 按上面的规则删除；harness 随后才判
失败 → 失败轮不写 exchange。CLI 留下的一切两边都没有——实测 11.5k output token 零字节留存，
「根本没写出答案」与「写了但没调 `submit`」无从区分，而两者修法相反。两处收口：

- **attempt 上的文本尾巴**：no-submit 的 execution attempt 自带 `final_text_tail`（末
  `NO_SUBMIT_TEXT_TAIL_CHARS` 字符）与 `final_text_chars`，从 driver 返回的内存结果上取，不依赖
  capsule 是否还在（`agent_session_host.no_submit_attempt`）。
- **capsule 本身也留**（owner 2026-09-16）：尾巴只说明回复怎么收的尾，事件流、stderr、它读到的
  输入只在 capsule 里。（回复**为空**的会话不走这里：`_run_episode` 对空回复抛
  `_empty_answer_error`，本来就按 transport 失败保留、进失败池。）driver 在删除点
  （`_settle_episode`）判断：带 completion 谓词、而谓词不成立 → 写 `UNACCEPTED_EPISODE_MARKER`、
  保留、定位符留在 attempt 上。这类 episode **单独计数**：`RETAINED_UNACCEPTED_EPISODES = 10`，
  与 transport 失败的 `RETAINED_FAILED_EPISODES = 20` 各自按 mtime 滚动，互不挤占。判断留在 driver
  内（谓词是 harness 本来就传进来的），上层调用点不感知——本节不让 episode 生命周期渗进上层的
  那条约束仍然成立。
- **审计包只进最后一个 capsule**：tool session 在循环结束后对最后一个 result 写一次审计包。第一次
  premature stop 之后重试时，前一个 capsule 带标记留下，但里面**没有审计包**——只有事件流、stderr
  与输入；它那一轮的拉取台账随 `reset_conversation` 清掉，只在最后一个审计包的 `resets` 里留痕。

**谓词的含义是「CLI 可以走了」，不完全等于「工作被 accept 了」。** 单 task 会话两者重合；会话宿主
（pseudo-conversational）的谓词有两处偏差，保留判断照读不误：

- **空闲时自己离开**：CLI 在没有未完成 task、assignment 尚未 seal 时干净退出（状态 `waiting`），
  谓词为假，于是即使它服务过的 task 全都 accept 了，capsule 也会被标成 unaccepted 留下。代价只是
  这 10 个名额里的一个。
- **harness 主动结束**：`_end_session`（replacement round / task timeout / close）先置 `_dead`，
  谓词因此恒真。通常 CLI 不会自己走（它停在长轮询上，或正在干活）→ 宽限后回收 → 原始流随 capsule
  留在失败池；只有 CLI 在宽限期内恰好自己干净退出，才按已完成删掉——哪怕当时有 task 没 accept。
  罕见；`close(keep_evidence=True)` 保住的是 assignment 树，不是这个 capsule。

要让两者严格一致，得给 driver 单独传一个「是否已 accept」的谓词，与回收用的谓词分开——今天没做，
上面两处偏差的代价不值得为它改 driver 接口。

**判断前必须再问一次 runtime。** 监督循环只在 CLI 活着时约每秒轮询一次谓词，进程退出后不再查。
一个 submit 完一秒内就退出的 CLI，被 accept 了却从没被轮询观察到；若只凭轮询结果判断，这次成功
调用会被当成 unaccepted 留下。多常发生取决于 CLI：2026-09-16 实测 CodeBuddy 10 次调用都在 accept
之后 4.6–11.7s 才退出，轮询次次赶上——所以对它这是少见情形，但补查只多读一次状态，换来判断不依赖
这个时序。谓词本身抛异常按「未 accept」处理：多留一份证据只费磁盘，少留一份则丢掉诊断。

**要全部留下时**：`[llm] local_agent_keep_episodes = true`（默认 `false`，只为排查）。打开后 driver 不删任何 episode、也不跑滚动清理，pseudo 会话关闭时也不删 assignment 树；Codex 不再带 `--ephemeral`，它自己的 rollout（`~/.codex/sessions`；用 harness 自己的 Codex home 时在 `<域>/.codex-home/sessions`，见 [当前实现与迁移注意事项](llm_local_agent.md#14-当前实现与迁移注意事项)；这家唯一的逐请求账本）随之留下。不进执行身份，改它不作废 checkpoint。磁盘不封顶，查完关掉再 `agent-clean`。

**accept 了的会话，不管轮询看没看到，干净退出就删 capsule**（2026-09-30）。轮询看到 accept 时
（`done_by_runtime`），driver 不再要求退出码与最终文本，但仍走同一个删除点（`_runtime_completed_result`
→ `_settle_episode`）；只有宽限期满被回收（`reclaimed_after_completion`）、或原始流读不了
（`stream_partial`）的才留下，原始流（`raw_events_retained`）在 capsule 里，进失败池。此前这条路直接
返回、从不删除，CodeBuddy 10/10 走它，一条长 run 的成功会话就能把真正的 transport 失败按 mtime 挤出那
20 个名额。

**单 task 工具会话默认 accept 后立刻结束 CLI**（`[llm] local_agent_exit_on_accept`，`auto | on | off`，默认 `auto`，
2026-09-30）。`auto` 只对能从逐请求事件补全用量的 driver 打开（agy、CodeBuddy、Claude Code）；Codex 只在
`turn.completed` 报一次用量、提前结束就整次未知，dsh 的补全没验证，二者在 `auto` 下不提前结束（owner 2026-09-30），
`on` 强制打开。按 driver 的取舍在 `ExecutionSettings.exit_on_accept_for`。这是预期的结束，不是回收：attempt 记 `ended_on_accept`，被杀的退出码与被截断的最后
半行都不算数（读事件流时丢掉末尾不完整的一行），照常 settle、删 capsule（`local_agent_keep_episodes`
打开时照常留下）。只有工具会话入口（`RoleClient._run_agent_tool_session`）按设置传
`run(exit_on_completion=True)`；pseudo-conversational 宿主从不传——它的谓词是整个 assignment 结束，
单个 accept 之后 CLI 还要回去 `next_task`。为了让 CLI 来不及发「收尾」那次请求，MCP server 把让
assignment 完成的那条 `accepted` 回复先记进帧日志、再压住 `ACCEPTED_REPLY_HOLD_SECONDS`（10s）才发，
driver 在下一次轮询（约每秒）看到 accept 就结束进程树；没人结束它，回复照发，只多花这段时间。
关掉设置即回到宽限期行为。

**提前结束时用量怎么来**：CLI 的总账在它最后的终止事件里，提前结束就没有。driver 改用流里每次请求
自己的用量补全，标出来源（尽力而为：读不出就记成未知，attempt 上一条 `usage_unknown_after_accept`
warning，调用照常成功，不等待、不重试）：

| CLI | 补全办法 | `usage.source` |
| --- | --- | --- |
| Claude Code / CodeBuddy | 每条 `assistant` 事件的 `message.usage`，按 `message.id` 去重（同一请求拆成多条事件时逐字段取最大）后相加 | `claude_code_message_events` / `codebuddy_message_events` |
| agy | 每个 `step_update`（state `DONE`）的 `usage`，按 `step_index` 去重后相加；只算本轮，不是会话累计 | `agy_step_events` |
| Codex | `exec --json` 只在 `turn.completed` 给一次用量，提前结束拿不到 | 未知（`unavailable`） |

各来源的 `input_tokens` 口径与同家 `result` 一致（Claude Code、agy 不含缓存，CodeBuddy 含），换算见
`client._INPUT_EXCLUDES_CACHE`。
实测（2026-09-30，各 n=1）：agy 3.7 Flash 两次不提前结束的工具会话
（7 次、5 次请求），各步用量逐字段相加与 `result.usage` 完全相等；省下的收尾请求在 agy 上是一次约
0.9 万未缓存 + 1.6 万缓存读取，CodeBuddy hy3 上约 1.1 万输入（其中缓存写入 815）。⚠ agy 自己的
逐请求账本（`conversations/<id>.db` 的 `gen_metadata`）被提前结束时会缺最后一次（交稿那次）请求的
记录，查账以事件流为准。
缓存写入：Claude Code 报的是 Anthropic 真正的写入，单列成 `cache_write_input_tokens`；
CodeBuddy 报的「写入」就是它没命中的那部分，算进未缓存。

**续会话的用量**：修复轮续进原会话、`resume` 档一条会话接着答，都是 `session_scope="assignment"` 的续接。
Codex（`turn.completed`）、agy（`result`）、CodeBuddy（`result`）这时报的都是整条会话到这一轮为止的累计
（Codex 2026-10-17 对照 rollout；agy 3.8 Flash 与 CodeBuddy hy3 2026-11-08 三轮探针：第 2、3 轮的输出与思考在第 1 轮之上往上加）。
逐次相加会把前面的轮重算：一窗的修复轮把第 1 轮再记一遍（2026-11-08 的一次 agy 纠错：68,682 + 126,849，实际 126,849）。
driver 基类记着每条会话到目前为止的总账（`LocalAgentDriver._own_session_usage`），只记这一轮自己的（usage 里 `resumed_usage: subtracted`）；
提前结束时按逐次请求补出的用量（agy 的 `step_update`、CodeBuddy 的 `assistant`）本来就只算这一轮，照记并加进总账。
这个 driver 实例没见过那条会话起头时照原值记 `cumulative`，扣成负数时照原值记 `mismatch_kept`。Claude Code 的 `result` 逐轮报 token
（Haiku，2026-11-09 同样的三轮探针：输出 601 → 4 → 5），不扣；它同一事件里的 `total_cost_usd` 却是整条会话累计的，只留在 attempt 的 usage 里，没有地方读它。

### 可以删掉的

runtime root marker、`store_id`、adoption predicate、`_is_own_episode`、capsule root 锁、
自动 prune 的保留策略、删除前的 ownership 复验、marker schema 升级路径。约 200 行生产代码，
以及多于此的测试。**不损失任何已存在的恢复能力——那项能力本来就没实现。**

### 精简后仍须保留

- parent root 必须是 FineSub 明确管理的专用目录（不是任意传入路径）；
- episode 用 `mkdtemp` / 强随机名**原子**创建；
- 删除时只删**本次创建后记下的那一条精确路径**，并确认它仍在 parent 之下；
- 绝不把任意用户目录当成「清理整个根」的对象；
- **用户发起的 cleanup / uninstall 删除整个 canonical root 之前，仍需 activity barrier**，
  否则会删掉正在运行的 episode。扫描没有消失，它被移到了唯一一个用户发起、可以保守行事的入口。

措辞更正：删除的依据是「**记下了自己原子创建的精确路径**」，不是「持有句柄」——我们持有的是
`Path`，不是能对抗目录被替换的 OS directory handle。这里不打算构建对抗同一用户恶意换目录的
安全模型。

### 两道硬门（均已解决）

**它们挡的不是同一件事**：门一（activity lease）曾挡 G-D 本身；门二（checkout cwd 隔离）只挡
G-C 的搬家。当前实现已让 `LocalAgentDriver.run()` 全程发租约并在租约内重解析 location；checkout
则固定使用仓库外的机器临时目录。

**门一的改造前问题：activity lease 覆盖不到直接调用的入口。** cleanup/uninstall 取 activity
barrier 之后删除整个 canonical root，但**目前只有 `finesub` shell 发布租约**
（`grep` 确认：`src/finesub/llm/` 与 `src/finesub/` 里没有任何发布点）。而
`python -m finesub.llm.correction_translation`、`python -m finesub.pipeline`、
`python -m finesub.llm.knowledge.update` 都是可以直接运行的入口，
`LocalAgentDriver.run()` 自己也不发租约。于是：

```text
直接跑 python -m finesub.llm.correction_translation → agent 正在用某个 episode
另一个终端跑 cleanup → barrier 判定空闲 → 删掉整个 parent
```

二选一，**取前者**：让 `LocalAgentDriver.run()` 在整个调用期间发布同一协调域的 activity
lease。它不重新引入 marker/ownership——租约说的是"有人在跑"，不是"这目录归谁"，而且顺带让
`relocate` 也不会在 agent 跑到一半时搬走 parent。

**但只发租约不够：拿到租约之后必须重新解析 parent。** driver 被 `LLMClient` 按
`(tier, model)` 缓存，而 capsule manager 在 driver **构造时**就把 root 解析并存下来了，于是：

```text
driver 构造并缓存旧 parent → relocate 完成 → run() 取得租约
→ 用缓存的旧路径建 episode，把刚被搬走的目录重新造回来
```

在门口等待时同理：run 在 activity gate 上排队，relocate 先完成，run 拿到租约后用的仍是**排队
之前**解析的路径。这个竞态仓库里已经解过一次——`shell.py` 取得 activity lease 之后立刻重跑
`load_app_paths()`，注释写的就是「a relocation that already owned the gate may have completed
while this run waited … so the run cannot recreate the just-moved old tree」。照抄那个顺序：

```text
解析稳定 coordination root
→ 注册 activity lease
→ 重新解析当前 big-data / episode parent
→ mkdtemp
→ 运行、清理
→ 释放 lease
```

因此 **canonical parent 不能缓存在 driver / manager 上**，每次 `run()` 在租约内重新解析；
只有显式注入的测试 root 例外（它本来就不参与搬迁）。

**还有一个更根本的前提：一个 episode parent 必须恰好对应一个协调域。** 当前已由完整域 digest
分区和统一 resolver 保证。历史上它曾分成两件状态不同的事：

- **capsule lock 的域错配——已由 `9902c71` 修复**（锁改为 parent 的兄弟文件，由 parent 单独
  推导，所有域取同一把）。下面这段实测是修复前的记录，保留是因为它说明了错在哪；
- **activity lease 与 cleanup 的多域问题——已由按域分区解决**；每个域只操作自己的 parent，
  默认 cleanup 的 barrier 与删除范围因此一致。

修复前的实测（`18d0ff0`）：`%TEMP%/finesub-agent-runtime` 是机器级
共享的，协调状态却不是：打包安装用共享 `user-data`，checkout 用自己仓库的 `.state`，两个独立
clone 各有各的。实测同一个 parent 在三个协调域下解析出**三个不同的锁文件**：

```text
shared parent: %TEMP%\finesub-agent-runtime
  checkout-A   -> ...\checkout-A\.state.agent-capsules-581f9646b719.lock
  checkout-B   -> ...\checkout-B\.state.agent-capsules-581f9646b719.lock
  packaged     -> ...\packaged\.state.agent-capsules-581f9646b719.lock
distinct locks for ONE shared parent: 3   => 互斥不成立
```

当时第 4 轮"修好"的认领竞态，换成两个 checkout 仍然可复现（后果有界：一方被判 unavailable 并
回退 API 链，不丢数据）；`9902c71` 之后这一条已经不成立了。但 **lease 那一侧仍然成立**——
"每次调用都发 lease" 只有在下述不变量成立时才有效：

> **一个 episode parent ↔ 一个 coordination root，且所有使用或删除该 parent 的进程都用这个 root。**

**取分区方案**：按协调域给 parent 开子目录（`<temp>/finesub-agent-runtime/<domain-digest>/`），
不变量就由构造成立——每个域只看得见也只删得掉自己那棵子树，不需要再造一个机器全局锁，也不用
指望别人跟你用同一把锁。

**统一入口**：所有路径由同一个解析器给出，`run()`、cleanup、`relocate` 都只经它，不各自拼：

```text
AgentEpisodeLocation(
    parent,                  # 本域的 episode parent
    domain_identity_anchor,  # 算 digest 的锚点，**可以是文件**
    activity_root,           # 发 lease 的位置，**必须是目录**
    locator_kind,            # managed_big_data | machine_temp
    location_identity,       # 域 digest，用于跨搬迁重新解析
)
```

**锚点与 activity root 必须拆开——把它们当成一个东西会直接崩。** checkout 的 `.state` 是一个
**JSON 文件**（`finesub/paths.py`），而 activity lease 会在给定 root **下面**建
`.task-activity.lock` 与 `.task-activity/<lease>.lock`（`activity_gate_path()` 就是
`coordination_root / ACTIVITY_GATE_NAME`）。把 `.state` 当 activity root 的后果是：文件已存在
时报 `NotADirectoryError`；文件还不存在时更糟——先被建成一个名叫 `.state` 的**目录**，之后状态
文件写不进去。

各档的取值：

| 档 | `activity_root`（目录） | `domain_identity_anchor` |
| --- | --- | --- |
| 打包安装 | `user-data` | `user-data` |
| 仓库 / worktree | **主 checkout 根**（与 `user-data` 一一对应的那个位置） | 该 checkout 的 `.state` 文件 |

仓库档要给 `.gitignore` 补 `.task-activity.lock` 与 `.task-activity/`——现在只忽略了
`.state` / `.state.lock` / `.state.tmp`。

**域 digest 是删除隔离边界，它的规范化必须是契约，不是命名细节。** 整个隔离证明依赖"同一协调域
必定算出同一 digest"；Windows 上同一位置有盘符大小写、路径大小写、junction/符号链接、UNC 等多
种写法，两个进程算出不同 digest 就又退回"一棵树、多套锁"。定死：

```text
location_identity = sha256(locator_kind + "\0" + canonical_domain_path).hexdigest()
```

`canonical_domain_path` 至少要求：绝对化并 `resolve()`（穿透 junction/符号链接）、Windows 下
case-fold、分隔符与 UNC 形式归一。digest 保留完整 SHA-256，不截断。

**验收矩阵**（这几条必须有测试，否则隔离只是声称的）：

- 主 checkout 与它的 worktree → **同一** identity（worktree 本就解析到主 checkout）；
- 同一路径的大小写 / 分隔符 / junction 变体 → **同一** identity；
- 两个独立 clone → **不同** identity；
- 任一域执行 cleanup → **看不到也删不掉**另一域的 episode。

**门二：checkout 模式下 canonical parent 仍与执行隔离冲突。** G-C 要把 capsule 放进
`<big-data>/agent-capsules`，而仓库/worktree 模式的大文件根**就是主 checkout**。[§1 执行环境卫生](#1-执行环境卫生episode-落在哪g-c已实施) 自己写了
拒绝任意仓库路径的理由是 **cwd 锚点**——供应商那些"从 cwd 往上找项目文件"的机制会指向用户源码
树。**这个理由对 canonical 路径同样成立**："它是 canonical" 只证明目录归 FineSub 管，不证明供应
商不会向上发现 `CLAUDE.md` / 项目规则。而各 driver 的规则隔离，下面这张表就是实测结果
（Claude Code 两档都不加载，Codex 两档都加载；agy 仅在显式 project 下加载 `AGENTS.md`）。

**2026-08-13 实测（该测的已经测了，结论是硬的）**：在一个带 `.git` 的目录树里放哨兵口令，
用各 driver 的**真实 argv**、cwd 指向下层 episode 目录，问模型口令是什么。

| | cwd **之上**的规则 | cwd **本身**的规则 |
| --- | --- | --- |
| Claude Code（`--safe-mode` + `--setting-sources ""`） | **不加载** | **不加载** |
| Codex（`--ignore-user-config` + `--ignore-rules`） | **加载** | **加载** |
| agy（显式 `--new-project` / 后续 `--project`） | **加载 `AGENTS.md`** | **加载 `AGENTS.md`**；`CLAUDE.md` 不加载 |

Codex 是**自动注入**，不是模型自己去读：`--ignore-rules` 确实在 argv 里，事件流中没有任何工具
调用，第一条 `agent_message` 直接吐出哨兵口令。那两个 flag 连**用户级**的全局 `$CODEX_HOME/AGENTS.md`
也管不到（2026-09-30 实测）：`--ignore-rules` 管的是执行策略 `.rules`，不是指令文件。Claude Code 侧确认不是假阴性：`is_error=false`、`terminal_reason=
completed`、答案就是 `NONE`。

**所以"逐 driver 实测"这条分支已经走完并且失败了**，剩下的不是二选一：

- **checkout / worktree 模式必须使用仓库外的专用 parent**。canonical allowlist 不足以放行——
  它只证明目录归 FineSub 管，不改变 Codex 会向上吃 `AGENTS.md` 的事实。

  **落点定为 `%TEMP%/finesub-agent-runtime/<域 digest>/`，即 checkout 档就留在 TEMP。**
  三个候选各自的问题：

  | 候选 | 问题 |
  | --- | --- |
  | checkout 的兄弟目录 | checkout 一搬家就成孤儿。⚠ 当时还有第二条理由「worktree 本来就是兄弟目录（`../asr-playground-<topic>`）、再塞一个进去很吵」——2026-09-04 worktree 搬进内嵌 `.worktrees/` 后它不成立了；第一条足够，落点不变 |
  | 本机 cache/state | 违反「仓库版只用自己的数据、绝不碰 `%LOCALAPPDATA%`」这条已写进 AGENTS.md 与 resources.md 的用户契约，而 capsule 装的是任务正文 |
  | **TEMP + 域分区** | **无** |

  关键在于 **G-C 的搬迁/卸载目标对 checkout 本来就不适用**：随 `relocate` 搬、随
  `finesub uninstall` 卸载，这两件事都是**打包安装**的生命周期。仓库版根本不跑
  `finesub uninstall`，开发者的清理方式是删目录或跑显式 cleanup。把 checkout 档硬塞进大文件根
  的形状，买不到它们，却要付一个契约违背或一堆孤儿目录。

  **但"写进 resources.md"不在此列——那一条对两档都成立。** `docs/manual/resources.md` 本来就有
  「从仓库源码运行时」一节；既然 `machine_temp` 是终态，就必须在那一节写明位置、"失败才留存"
  以及显式 cleanup 怎么用。否则正好重演本节开头那句抱怨：**用户不知道任务正文的副本落在哪里**。

  因此 `locator_kind` 里的 `legacy_temp` 应改名为 **`machine_temp`**：它不是过渡态，是 checkout
  档的**终态**。

- **路径层级只在这里定义一次**（别处不要再推导，上一版就是因为两处各写一遍而多出一层 digest）：

  ```text
  managed_big_data:
    parent  = <当前 big-data> / agent-capsules
    episode = parent / <episode_id>

  machine_temp:
    parent  = %TEMP% / finesub-agent-runtime / <location_identity>
    episode = parent / <episode_id>
  ```

  即 **digest 已经包含在 `machine_temp` 的 parent 里**，`<parent>/<digest>/<episode>` 是错的。

- **升级后旧根的处理：不迁移、不自动删。** 现有 episode 直接躺在**未分区**的
  `%TEMP%/finesub-agent-runtime/<episode>`（新布局多了一层 `<location_identity>`）。规则定死：

  - **按域的 cleanup 只看自己那棵子树**，绝不枚举未分区旧根、也绝不碰别的域的子树——旧根可能
    正被另一个协调域或旧版本使用，靠某一个 checkout 的 activity barrier 判它空闲是不成立的；
  - 未分区旧根交给**显式的一次性全量 cleanup**：用户主动执行、列清楚要删什么、要求所有 agent
    已停止。这正是 [§2 capsule 是一次性 episode](#2-capsule-是一次性-episode不是持久-storeg-d已实施) 那条规则的应用——**扫描可以存在，但必须是用户发起的**；
  - 不做静默迁移。旧现场都是小文本，留着不影响正确性。

- **留存规则的准确说法**（用户文档按这句写，别写成"只有失败会留"）：
  **transport 失败，或成功后的目录清理失败时，才会留存。** 第二种是本节自己定的规则造成的——
  成功后 `rmtree` 输给杀软或占用时，结果仍判成功、落 `episode_cleanup_failed`、目录留着等显式
  清理。漏掉它，用户看到一个"成功却留了现场"的目录就会以为任务失败了。
- **`_reject_repository_or_reparse_root` 已在 G-D 之后保留**。我在 15.6 里一度把它降级成
  belt-and-braces，那是错的：在 Codex 上它是目前唯一挡住"用户仓库的项目规则被注入进纠错调用"
  的东西。G-D 拆所有权机制时不能顺手把它拆掉。
- agy 已跑该实验；结论同样要求 checkout/worktree 的执行根在仓库外，并要求 driver 每次显式绑定
  它自己生成的 project。漏传 `--project` 不仅改变规则发现，还会让 `llm_local_agent_agy.md` [§4 准入硬门：view_file 的读取边界](llm_local_agent_agy.md#4-准入硬门view_file-的读取边界) 的 hooks 消失。

顺带一提，这不只是"洁癖"：用户仓库的 `AGENTS.md` 一旦进入上下文，既是 prompt 污染（项目规则
混进字幕任务），也是注入面（任何能写到那份文件的人就能给 agent 下指令）。

### 已实施的返回契约

- **成功路径只返回 `episode_id`，不返回 `AgentExecutionResult.capsule`**：目录已经删除，暴露其中
  path 只会制造悬空对象。生产代码在 `client.py` 记录这个 id。
- **失败现场必须可定位，而绝对路径不是"可定位"**：失败之后 `relocate` 会把整个
  `agent-capsules` 搬走，当时记下的绝对路径立刻失效。但**只记 store 相对路径也不够**——它只在
  打包安装那一档成立，`%TEMP%` 过渡期和 checkout 的仓库外 parent 都不在 `locations.json` 里。
  locator 必须自带**位置类型**，由上面那个 resolver 打开：

  ```text
  { locator_kind:      managed_big_data | machine_temp,
    location_identity: <域 digest>,
    episode_id:        ...,
    absolute_at_write: ...   # 仅供即时诊断，明确标注不稳定 }
  ```

  **不存 `relative_path`**：它可以由上面那张层级表派生，而冗余正是上一版两处层级写法打架的根源。
  resolver 拿前三项重新算出 `parent / episode_id`：`managed_big_data` 按当前 `locations.json`
  解析（搬迁中沿用 `migratingFrom` 回退），`machine_temp` 按 `location_identity` 解析。
  再给一个面向用户的「打开/清理失败现场」入口。

### 成功之后删除失败，算什么

Windows 上 `rmtree` 会因为杀毒软件、短暂占用等失败。这时**结果已经是好的**，所以不能：

- 把一个有效结果改判成 transport failure；
- 触发下一候选 fallback——那是**重复消费一次配额**，为了删不掉一个临时目录；
- 用 `ignore_errors=True` 糊过去，假装删掉了。

正确语义：**保留成功结果**，落一条 `episode_cleanup_failed` warning 加上面那个 evidence
locator，交给显式 cleanup 以后处理。删不掉临时文件是卫生问题，不是正确性问题，不该让它改变
调用结果。

### 两个必须落地、否则得不偿失的前提

1. **不要直接用 `tempfile.TemporaryDirectory`**：它在正常和异常退出时**都会**清理
   （实测异常路径同样删除），与「失败留存」正相反。用 `mkdtemp` + 成功路径显式 `rmtree`。
2. **清理入口必须真的做出来**，不能停在文档里。否则只是把「自动清理的复杂度」换成「永不清理」，
   而残留正是失败现场——数量受现有事件/stderr/result 上限约束，内容是文本与 JSONL，
   攒到用户执行 cleanup 是可以接受的，前提是那条命令存在。

### 滚动上限：最多留 20 个（2026-08-15）

上面第 2 条的「攒到用户 cleanup」在有 cleanup 命令之后仍然是无界的：单个现场几十 KB 到几 MB
（agy 媒体失败还存一份剪辑副本），失败率高的长任务能攒出可观的量，而**旧现场没人看**——要查的
永远是刚发生那次。因此 `AgentCapsuleManager.prune` 在**建新 episode 时**按 mtime 只保留最近
`RETAINED_FAILED_EPISODES = 20` 个。放在创建前而不是调用结束后：域只在这一刻增长，先剪再建也
保证崩在半路时不会留下 N+1。

**第二个池（2026-09-16）**：harness 会话干净退出却没被 accept 的现场带 `UNACCEPTED_EPISODE_MARKER`，
由同一次 `prune` 单独计数，只留最近 `RETAINED_UNACCEPTED_EPISODES = 10` 个。分开数是因为合在一起
时，一批此类会话会把 transport 失败全挤出去（反之亦然）。下面两道闸门对两个池同样适用。为什么
这类现场要留、为什么判断前要再问一次 runtime，见上文「目标形态」。

**绝不能删掉正在跑的 capsule**，两道彼此独立的闸门都为此：

1. **最近 N 个永不是候选**。并发只有 `max_parallel`（每 driver 4）× 进程数，正常远小于 20。
2. **年龄闸门**：只删早于 `max(2 × timeout_seconds, 3600s)` 的。调用在 `timeout_seconds` 必被
   掐断，所以更老的必然没有活主人。这道闸门是为「一个域内同时在飞超过 N 次」那种非正常情况
   兜底的，让上一条从「几乎总是对」变成「一定对」。

**点开头的条目一律跳过**：`.conversations` 是 assignment scope 的工作目录，`.agents` 是 agy 的
受控 project（它的 hook 就是读取边界）——删掉这两个是搞坏正在跑的 run，而不是回收证据。

整个 prune 是 best-effort：返回删了什么、不抛异常。剪不掉最多是占磁盘，让调用失败则是毁掉这次
运行——与上面「删不掉临时文件是卫生问题，不是正确性问题」同一条原则。`finesub agent-clean`
仍然是「现在就全清掉」的显式入口，两者不冲突。

### 实施顺序记录

实施时 G-D 排在 G-C（capsule 根挪进大文件根）之前：先定「一次性 episode」，parent root 搬家
便只需替换 resolver 结果，没有把 marker/锁/认领那套带入终态。

---
