# 知识库

> **2026-08-22 起知识库的真相源是 SQLite**（`<knowledge_root>/knowledge.sqlite`；设计取舍在维护者的
> 本地计划里，不随公开快照发布）。本文其余部分
> 仍按「条目 / 小节 / 行」描述知识库，因为模型看到的渲染形态没变；凡提到 markdown 文件的地方，指的是
> `rendered/` 下的**派生缓存**（同时是可编辑投影，见「存储形态」）。2026-08-26 起 agent 工具会话有
> `kb_index/search/read/read_node/kb_validate` 五件套（授权 = 调用方显式 `kb_tools` 授予、manifest
> 携带；`kb_index` 是必读门），harness 精确扫描以 shadow 事件落账；`keep_entries` 经 owner 决定
> **保留**，预注入与透传链照旧是两形态共同的数据面。2026-08-27 起有
> 字段级事件与 claim 证据：`exposed`/`landed` 事件、精修对照的确证/推翻回写、只读报告
> `python -m finesub.llm.knowledge.report`（见下「信号与证据」节）；同日落地共享的
> 首个可运行形态（`knowledge/share/`，见下「共享」节）。

存放公开网络中很少存在、难以进入 LLM 语料的知识（主播设定/经历、社区常用梗与人物、常见翻译错误台账），供背景调查、纠错窗口和知识更新流程注入。能直接在网络上简单搜到的知识，或 LLM 已知的大众知识，无需收集。

本文是知识库相关行为的唯一权威文档。相关文档：

- 纠错/调查/资料注入等运行时 harness 行为：[`llm_harness_behavior.md`](llm_harness_behavior.md)
- 架构意图与设计决策：[`llm_design_notes.md`](llm_design_notes.md)

涉及模块速查：

- `src/finesub/llm/knowledge/base.py` — index 加载、`match_index_keywords` 本地别名匹配。
- `src/finesub/llm/knowledge/update.py` — 统一知识更新入口 `run_knowledge_update`，CLI 为 `python -m finesub.llm.knowledge.update`。
- `src/finesub/llm/knowledge/materials.py` — 按窗口/全局分组构造统一知识更新的输入材料。
- `src/finesub/llm/knowledge/feedback.py` — `<task_update_feedback>` 的聚合与 hint 归并。
- `src/finesub/llm/knowledge/entries.py` — 知识条目预取（`<kb_entries>` 块）。
- `src/finesub/workflows/reference_ingest.py` — 参考素材导入 workflow 入口。

## 知识库定位与结构

**存储形态（2026-08-22 起）**：`<knowledge_root>/knowledge.sqlite` 是唯一真相（`src/finesub/llm/knowledge/node/`：
identity/version 两层表、pinned read、每次 apply 一个事务 = 一个 `rev`，按实体 CAS，版本行只增不删）。
一次 run 读一个 rev：`correction_translation.run_correction_task` 全程持有 `base.pinned_generation_rev`
（准备任务/纠错窗口（含请求回合）的默认读全部钉在 run 开始时的 rev），任务后知识更新每 chunk 显式传
`working_rev` 覆盖（read-your-writes）。pin 是 **run 作用域**（ContextVar，2026-08-30 起，
任务级并行，总览见 [`llm_harness_behavior.md`](llm_harness_behavior.md)「任务级并行」节）：
同 root 的并发 run 各读各的 rev、
互不等待；run 自己开的线程池经 `run_context.bind_llm_worker` 把 pin 带进 worker，跨进程边界
（MCP server）则在创建时写死（spawn env 带 root、任务 manifest 带 rev），不从线程状态推。
并行纠错在**屏障处重钉**（`repin_generation_rev`）：词条集定死的那一点把 pin 移到
当前 rev——快照语义从「一 run 一版」变「一阶段一版」，期间别的 task 提交的词条对纠错阶段可见，
同阶段所有窗口仍读同一版；resume 身份本就不含词条正文，缓存不受影响。
`<knowledge_root>/rendered/{streamer,common}/*.md` 与 `index.md` 是每次写入后重生成的**派生缓存**（store 是真相，冲突时 store 赢），
给人看、给文件视图与编辑器用。**文件名由条目名换算**（subject 的 `surface`；2026-10-05 起）：条目名里的
`\ / : * ? " < >` 换成全角同形字（`Re:ゼロ` → `Re：ゼロ.md`，`Fate/stay night` → `Fate／stay night.md`），
换完仍当不了文件名的（`.`、`..`、Windows 设备名 `CON`、`NUL`、`COM1` 等、超过 240 字节）、同一目录里已被占用的
（类别的 `index.md`，或绕过查重的旧数据里只差全角半角的两个名字）退回 `<local id>.md`；文件里的 H1 永远是真名，
收割按清单把文件对回条目 id、不看文件名。两个只差全角半角的名字在库里本来就是同一个名字（查重做 NFKC 折叠），
所以换算不会让两个活条目撞名。条目名本身只禁这些：竖线 `|`（index 行与术语行用它分列）、控制字符与换行类字符、首尾空白、以「类别名/」开头（那是「类别/名字」写法，这样的名字按名字永远查不到），
外加不超过 100 字。规则只有一处（`node/model.py` 的 `entry_name_problem`），所有设名的写入都经
`validate_payload(strict=True)` 问它——新建、`rename_entry`、改 H1、push 准入、pull 准入；条目下的各行
不受限。渲染时再问一遍：规则之前落库的名字不合规就退回 `<local id>.md`，且写出的路径必须仍在
`rendered/` 之内。「类别/名字」的写法（`style/某字幕组`）只在斜杠前是已知类别时才算指定类别，否则整串是名字。human 投影的条目行带 `- ` 前缀（markdown 渲染态下每条一行，兼容在渲染态
编辑的 md 编辑器；回写解析剥掉前缀，没打前缀的新行也照收）。**模型注入面一律走 prompt 投影**（裸行、
不带 bullet）：`load_entry_texts`（research/搜索/纠错注入）与 agent 快照读的都是
`entry_injection_text`（prompt 无句柄），任务后更新才用带 `@k` 句柄的 `entry_prompt_text`——
`entry_text`（human）只喂 rendered/、`show` 与编辑回路，绝不进 prompt。自 §11.3 起 rendered/
也是**可编辑投影**：每文件的基线（rev+hash）记在
`rendered/.manifest.json`，refresh 只重写内容变化的文件、**绝不覆盖用户改过的文件**（dirty 保留）；
纠错 run 启动时（`--knowledge` 非 none）自动收割干净的用户编辑为 `kind=user` revision
（note `rendered-edit:<文件>`，作者是人、与触发时机无关），**启动时绝不自动跑 LLM 修复**；
worktree 闸门与跨进程写锁照常生效。失败分两档（2026-09-03）：**行级**看不懂的（空标记、缺别名列的
三段术语行、该节收不了的行体，以及 strict preset 下整节不被允许）**原样泊进该 preset 的「待归类」**，
文件其余编辑照常落库——泊进去的行按构造就是 `staging-line` 候选，`repair` 接得住（O3「确定式先行、
LLM 只碰规则标出的判断题」，O4 的兜底由此闭环，而且闭环那一步是人按的）；**文件级**坏掉的（首行不是
H1、未闭合的 HTML 注释、多于一行的 intro、意外的元数据行、改名撞车）仍是**逐文件告警跳过、原样保留**——
结构坏了就无从知道哪一行本该是什么，而改名撞车是人才能拍的板。泊进暂存区是 `on_invalid="stage"`，
**只有收割走这条**：`knowledge edit` 保持 `reject`，人就在编辑器前面，报错指着他刚写的那行更有用。
⚠ **暂存区自己不参与这一趟**：它的行按定义已经泊过了，再扫一遍等于每次收割都把同一行退休重建
（node 身份换新，`repair` 对它做过的裁定跟着丢，还多一条没有语义的修订）。写进暂存区的行一律
按 note 收——那一节声明了 `term`，不这样的话三段行在那里同样是格式错。编辑面仍是 ops、冲突时 store 赢——
markdown 编辑只是一种录入方式。结构化人工编辑走 `python -m finesub.llm.knowledge`（实现在 `knowledge/maintain.py`；
2026-09-01 从 `node/cli.py` 上移改名——`node/` 只留引擎，`node.cli` 这个叫法有歧义）：`show`/`log`/`refresh` 只读；`edit <entry>` 把 human 投影吐进 `$EDITOR`（或 `--file`），
改完用导入器同款解析器 diff 成引擎操作、走共享 apply engine 的一个 `kind=user` revision——纯移动的行
（节内重排、跨节搬家）保持节点身份，别名列 diff 双向同步 `items`，误听描述 add-only。
**只差首尾空白的改动不是编辑**：回写逐行 trim 后再 diff、字段也按 trim 比较，所以多打一个空格这类改动
（行首行尾、术语某一列的两侧、简介）不出操作、不写修订，也就不算人改版（见下「人改版注记」）。

### 人改版注记（human 投影，2026-10-01）

模型可以改写、退休人改过的行，不设闸门（owner 2026-10-01）；代价由可见性兜。`entry_text`（human 投影的
唯一出口：`rendered/` 与 `show`）在一行的**现值 ≠ 最近一次人认可的值**时，在那一行下面渲染一行 HTML
注释、给出那个值；被退休的行在原节里留一条同形注释（原节已不渲染时挂在正文末尾、写明原节名；只留在退休那一刻
还挂着它的条目里——挂在多个条目下的行每个都留，移到别的条目之后才退休的只留在新条目）。经跨条目 `move_line` 挪来的行（多半是合并时），注记跟着节点到新条目，并写明「迁自「原条目」」；原条目里没被挪走的行随原条目退役，不留注释（owner 2026-10-02：整条附上太重）。prompt 投影
（`entry_injection_text` / `entry_prompt_text`）不带注记，模型永远看不到人改版。实现在 `node/human_marks.py`。

- **按行、按值判**：一行 = 小节里的一个子节点连同它的别名 item，值取 human 投影里这一行的文本（含别名列）、
  trim 后比较。误听 item 与不渲染的旧字段不算——注记给不出、也抄不回它们。
- **别名按集合、只看少没少**（owner 2026-10-01）：人认可的别名都还在（现值的别名是它的超集）就不算变化，所以模型
  只追加别名、或别名只是换了顺序，都不出注记；删掉或改掉任何一个人认可的别名才算。其余部分（去掉别名列的行文、
  H1、简介）仍按 trim 后的值精确比较。值存成结构化的（行文、去掉别名列的行文、别名列表），比较时不回头解析拼好的
  行——描述里可能带竖线。行的别名列与条目头的别名两处都这样；ack 认可的值同样这样比。「人改版」的判定不变：人自己
  追加别名仍算一次人改。注记里给出的仍是人认可的那一版完整的值。
- **条目头算一个单位**（owner 2026-10-01）：H1（`surface`）、简介（`intro`）与条目层别名（subject 自己的 aliases
  item，渲染在档案的 `[别名]` 槽）合起来相当于一行术语——本名、描述、别名列——判定、ack 与不出注记的情形全照行的
  规则。别名只在那个槽真的渲染时算进值：preset 没有别名角色（如 `style`）、档案不在渲染的小节里、或档案里已有一行
  带 `[别名]` 标记（核心空槽于是不渲染）时，值里没有别名这一项，条目别名的增删都不出注记，注记里也不写别名。只认这三处：`updated_date` 每次写入都刷，`category`、重分类理由、`section_order` 与派生的检索字段也都不是
  条目头渲染出的东西，算进来就会每次更新都出注记。注记挂在简介下面（没有简介就在 H1 下面）。
- **人认可**有两种：一是 `kind=user` 修订（edit、CLI、`rendered/` 收割）且这一行的值真的变了——`share mark` /
  `unmark` 也是 `user` 修订，但只改可见性、不算；二是一次 ack。`harness`、`pull`、`share`、`import`、`revert`、
  `restore` 都不算（owner 2026-10-01）：revert 回到人改版时现值等于人认可的值，不出注记；回到一个从没被人认可过
  的值就出注记——包括 revert 掉人自己新建的行，注释留在原节、ack 才清。人自己删掉的行不留注释。只改
  `canonical_id` / `accepted_rev` 的 pull 不改值，既不刷新人认可的值、也不让注记出现。
- **注记不会自己消失**：人再改这一行，或 ack 认可现值，才覆盖它；模型再改一次，注记照旧、只是现值换了。
  模型或 pull 把值改回人认可的值时，两者没有差异，注记随之不出现。
- **形态**：`<!-- 人改版 r41（行 3f2a9c1b）：<那一版的行> -->`；条目头写
  `（条目头 …）：H1「…」；简介「…」；别名「…」`；由 ack 得来的写 `人认可版`，退休行加「已被删除」。
  `行 …` / `条目头 …` 是节点 id 的前 8 位，`ack` 用它指行（human 投影里没有别的句柄）。值里的 `-->` 与 `--!>` 渲染成
  `--\>` / `--!\>`，注释里写明抄回时删掉反斜杠。整行注释在回写解析时本来就剥掉（`strip_comments`），
  所以注记不会被当成编辑收回；要恢复人改版，把注释里的行抄回正文即可。
- **ack**：`knowledge ack <条目> [行号]`，只收带注记的行，不给行号就是该条目所有带注记的行（含退休行，认可的
  是「这行没了」）。记进 `line_acks` 表（schema v6）：不是 revision（版本表一行不动），也不进 `evidence`——
  那是随共享 bundle 导出、进报告的 claim 出处，ack 只是本地 human 投影的记账。它按 store rev 排序：认可的是
  ack 时那个 rev 的现值，判注记时从版本表按那个 rev 现算（`value` 列只作记录、不读回，所以值的存法变了旧 ack 照样能用）。

### style 类别（2026-09-02）

第三个 category `style`，与 streamer / common 的区别只有两条，其余（版本行、apply 引擎、
三投影、`rendered/` 回写、共享）完全共用：

- **不在匹配面上。** 常量分两组（`node/presets.py`，2026-09-07 从 `node/model.py` 搬来）：
  `MATCHABLE_CATEGORIES` 是能被文本匹配到的
  那些（`resolve()` 默认扫它、index 由它建、关键词预注入读它、`create_entry` 的全局查重按它
  判），`STANDALONE_CATEGORIES` 只能被显式点名，`CATEGORIES` 是两者之和——⚠ 但前两者**不再是
  字面量**：`CATEGORIES` 是唯一的有序声明，两组由**各自 preset 里的 `matchable` 字段**派生，所以
  「这个类别进不进匹配面」只写在它自己的 TOML 里；
  `base.KNOWLEDGE_CATEGORIES` 是前者的**别名**，不是第二份字面量。`style` 属于后者：它的取用
  方式是 `--style <名字>`，进匹配面不但会挨文本误命中，`create_entry` 的**全局**查重还会拒绝
  一条与某个专名条目同名的 style——而 style 通常就以它服务的字幕组或主播命名。代价是名字
  唯一性得自己管（两条同名 style 照样撞，那是它自己命名空间内的事）——`repo.resolve_in(name, category)`
  按命名空间查重（可匹配类别共用一个，standalone 各管各的），`create_entry` 与改名冲突检查都走它；
  人读/CLI 面走 `repo.resolve_qualified()`。standalone 类别**不出 index**（索引就是匹配面），
  但照常渲染进 `rendered/style/` 并可回写。
  ⚠ **两个命名空间意味着同名是常态**（style 通常以它服务的字幕组/主播命名），所以每条
  「名字→条目」的路都要面对它，这是 2026-09-02 复审一次抓出四处的地方：
  - **人的那条路**：`resolve_qualified()` —— `style/某字幕组` 显式指定类别（`/` 在 key 里
    非法，所以不会歧义），裸名字只命中一处就用它、命中多处**抛 `AmbiguousName`**。
    `show`/`edit`/`retire`/`repair`/`ingest`/`share mark`/`share push` 全部走它，
    没有"优先某一边"这种默认——静默选一边就会退役、编辑或推送用户没指的那个条目。
    ⚠ CLI 的 `retire` 必须**先解析再造提案**并带上 category：提案按名字解析，裸名字会回到
    可匹配集。
  - **模型那条路**：`subject_ref` 遇到不在 `allow_categories` 里的显式 category **直接拒**，
    不降级成无类别查找——降级会把"写进 style"变成"写进任何叫这个名字的条目"。
  - 改名与新建共用 `resolve_in`：跨命名空间重名合法，同命名空间重名拒绝。
- **`verify` 全节 `none`，`share_inherit` 全节 `local`。** style 没有外部真相（一条约定的
  正确性不取决于任何可检索的事实），也不是默认外发的东西。

**接线**（2026-09-02，方案第 3、5 步）：

- **三档开关 `--style-mode`**（与 `--knowledge` 同形，2026-09-02）：`none` 不用 /
  `read` 只注入 / `update` 还把本次学到的写回去。未给读 `[llm] style_mode`，再未给是 **`read`**
  ——注入本来就是 style 存在的理由，写回改的是存量、要人开口。⚠ `update` 搭的是任务后知识
  更新那班车，缺 `--knowledge update` 或 `--refined-srt` 时它写不成，此时**出 warning 而不是
  静默降级**（`style-update-inert`）。
- **`--style auto`（2026-09-09；强制选与 fast 全格 2026-10-01；同日翻成默认，见下「取用」）**：意图不是名字——
  模型拿到本 rev 全部 style 条目的**描述目录**（名字｜intro｜作者｜适用范围，`style.load_style_catalog`），按「适用范围」
  与用户本次要求选**一套**，回复里带一个 `<style_selection>` JSON 块；harness 把它变成 `StyleChoice`（selected / none）。
  - **必须选，没有回退**（owner 2026-10-01）：`default_style` 本身是候选，「对不上专门的那几套」由模型明确选它来表达。
    缺块、块不是合法 JSON、名字不在候选里、或库里有 `default_style` 时留空，都是**提问那一阶段的违约**，走该阶段自己的重试 /
    修复，用完**整个任务失败**（`StyleCatalog.choose` / `decide` 抛 `StyleContractError`，不落到任何默认值）。库里没有
    `default_style` 时，明确留空是合法的「本次不用用户的风格」（注入出厂默认，见下）。
  - **谁来选**：普通模式是准备任务——校验在 `parse_prepare_reply` 里，走解析重试；检查点回放到不合契约的旧回复记
    `session_checkpoint_invalid` 并重新调用。fast 是整段窗口自己（`task_graph.WindowFlow.style_pick`：fast、auto、目录里有
    `default_style` 之外的候选，与执行策略无关）：
    - batched：首条回复必须是请求回合并带合法选择，直接交付或只请求不选都按格式错误重试，用完抛错。只有 style 可请求时
      （`retrieval` 不是 local、又没有词条索引）请求回合只为选风格而开：开场只讲选风格，回复里出现搜索或词条请求算格式错误，
      回填只带选中那套的全文、不要求 `<keep_entries>`。
    - agentic：交付里附选择，而且必须先在本窗的工具会话里用 `kb_read` **完整、成功**读过选中那套（不带 `sections`、没撞回复
      上限）。核对用 harness 侧的 `kb_read` 台账，在窗口 validator 里判，runtime 与 harness 同一个判断（见
      [Agent 工具协议「生产工具表」](llm_agent_tool_protocol.md#2-生产工具表唯一真相)），走修复轮。第一份通过校验的选择钉住全任务：窗口被劈成
      -a/-b 时，后半窗不再展示目录、直接拿到那套全文。
    - `style_pick` 只开通道、不改 `window_requests`：只为选风格开的通道要不回丢掉的词条，透传链照旧不剪
      （`CorrectionRun.window_requests` 直接取 `WindowFlow.window_requests`）。
  - **先读记录**：编排层（`correction_translation._apply_auto_style`）在两种模式下都先读这个任务已有的选择，有就照它办，
    续跑不重选。`research-context.json` 里有 `style_selection` 就以它为准——编排层每次钉住时抄进产物的那份副本压不过它，
    所以准备阶段重跑选出新的一套时，注入与写权限都跟新选择走；context 里没有时才读最新的 `style_selection` 产物（fast
    窗口带 `chunk_id` 的选择，或下面「没有可选」的记录）。写权限（`_writable_style_names`）读的是同一条规则。没有记录、且目录里没有
    `default_style` 之外的候选时谁也不问：直接记下「没有可选的 style」——有 `default_style` 就选它（注入、可写），零套库只用
    出厂默认——不告警、不多调用，结果与不给 `--style` 时相同，也写进 `style_selection` 产物，续跑看到的是同一个决定。
  - **续跑按记录的那一版渲染**：选择带着当时的 rev（`StyleChoice.rev`），续跑渲染的是那一版正文，条目在两趟之间被改写或
    退休也照样渲染得出；记录的 rev 与当前库都渲染不出才降级成「没选中」并告警。⚠ rev 只在同一份 store 内有意义：`knowledge/`
    重建过或换了 `--knowledge-root`，就退化成按当前库渲染——判定与渲染必须返回同一个 rev（`_renderable_style_rev`）。
    ⚠ 读与写判在不同 revision 上：译文可以从历史渲染退休条目，知识更新却写当前 head，所以回写资格一律按当前候选判
    （`_writable_style_names`）。
  - **超长素材分块准备时按票数选**（owner 2026-10-01）：每块的准备任务都看到目录、按强制选交付一个选择，作为这一块的一票
    （选「无」的块——只在库里没有 `default_style` 时可能——也算一票），得票最多的胜出，写进 `research-context.json` 的
    `style_selection`，下游与单块准备同一处。平局确定式地破、不随机，依次比：使用率（下面「使用率」，多者胜；读不到就跳过）→
    条目在本次目录修订（`catalog.rev`）时最近一次被改动的 rev（新者胜）→ 名字的码点序；「无」与某一套打平时那一套胜。理由里写明
    各套的票数、投票的块和破平局用到的数（`style.tally_style_votes`）。续跑读持久化的结果、不重算；没带选票的旧分块产物（这一块
    当时没投票）在 `--style auto` 下重跑那一块，带票的照常回放。
  - **使用率**：任务的风格一确定就记一条 `exposed` 事件，`opportunity="style_injected"`（`node.signals.book_style_injected`）：
    编排层钉住时（auto 的选择或记录、显式 `--style`、静态的 `default_style`）、fast batched 请求回合选定时、fast agentic 第一份
    合法选择钉住时。一个任务对同一套只记一次。使用率 = 记过它的不同 `task_id` 数（`style_usage_counts`）；只 `kb_read` 读过
    候选、没注入的不算（那些读记的是 `opportunity="context"`）。fail-soft 遥测，worktree 里随知识库信号开关关着。
  - 选择窗口启动前钉住，`--style-mode update` 的写入对象是选中的那套（包括明确选中的 `default_style`）。取舍依据在维护者的本地
    计划里（不随公开快照发布）。
  ⚠ **读权限**：`kb_read` 接受 `style/<名>`。拿到 `kb_tools=read` 的任务能读任何一套 style 的正文——只是读，写权限仍只给
  选中的那套，因此不分（owner 2026-09-30）。没有开知识库读取、只为选风格的 agentic fast 窗，以及没有词条索引的 agentic
  准备任务（`retrieval=local` 时，prompt 让它可以细看候选）拿的是**窄授权** `kb_tools=style`：只暴露 `kb_read`，只解析
  `style/<名字>`——只开 style 不因此获得普通知识库读取权。
- **取用**：`--style <名字>[,…]` 或 `--style auto`（`python -m finesub.pipeline` 与
  `python -m finesub.llm.correction_translation` 都有），未给读 `[llm] style`，**再未给就是 `auto`**（owner 2026-10-01：
  所有格子都能选之后翻成默认；风格仍默认开、默认只读，owner 2026-09-02）。库里没有可选的 style 时 `auto` 不问模型、不多调用：
  有 `default_style`（`DEFAULT_STYLE_NAME`）就用它，没有就只用出厂默认（见上「先读记录」）。
  ⚠ **`difficulty=efficiency` 只用出厂默认风格**（owner 2026-10-30）：该档不注入用户的风格、
  不问模型、不写回。配置里的 `style` 与库里的 `default_style` 是用户的默认，在该档不生效、也不报错；命令行点名一套
  或 `--style auto` 报错，提示换到 intermediate、续跑去掉 `--style` 照旧用记下的风格；`--style ""` 与 `--style-mode none` 照常接受；显式 `--style-mode update`
  报错，来自 `[llm] style_mode` 的 `update` 降为 `read`（那是给别的档的常设默认）。不点名时是**不发起选择的 `auto`**
  （`StyleSelection.ask=False`）：沿用任务已有的选择记录；没有记录时把「不用用户的风格」当成一条选择记录写下来，与「没有可选的
  style」那条捷径同一处、同样不告警。所以换难度档续跑不会悄悄换风格：额度用完换到 efficiency 续跑的任务照旧用它记下的风格。
  同一个理由，显式点名的一套（或显式 `--style ""`）也记成任务的选择（`intent="named"`），之后不点名的续跑沿用它；
  点名本身仍以用户为准，不读记录。`--research-only` 也在返回前把这次的选择定下并记录。`reference_ingest` 解析时不传 difficulty、不受这两条限制（交给纠错那半的请求带
  `style_efficiency_cap=False`）：它是写回流程，知识库那半在任何档都更新，`auto` 时写的是纠错那半记下的那套。
  ⚠ 静态的 `default_style` **库里没有就当没有**（新装的库本来就没有），而用户**点名**的名字指不到
  仍然报错：默默不用你要的那套，看起来和「这套风格没起作用」一模一样；
  解析在 `knowledge/style.py`：**`resolve_style_selection` 是决定这一切的那个函数**（三级取值、默认 `auto`、efficiency 只用出厂默认、以及 `writable` 从 mode 派生都在它里面），`resolve_style_names` 只做「参数 → 配置 → 空」那一层。argparse 一律传 `None`。选中的条目
  以 **prompt 投影**注入纠错 system prompt 里原先放「常见翻译错误对照」的那个槽
  （`style_block`）。名字指不到、指到别的类别、或同名歧义都**报错**而不是静默跳过——
  静默的样子和「这套风格没起作用」一模一样。
- **出厂默认风格**（owner 2026-10-31，`style.FACTORY_STYLE_NAME`）：软件自带的一套，以 style 条目的 prompt 投影写成模板
  `prompt_templates/style_factory_default_v1.md`，不进库、`--style` 点不了。纠错与校对 prompt 里原先散着的口味要求（语感、
  不加注、语气词、意译尺度、语序、口语译法、圈内借词）挪进了它，通用规则只留机制。选择定下来、却没有用户的风格时
  （`--style ""`、`--style-mode none`、auto 选「无」、零套库、efficiency），`render_style_block` 注入它；选择还没定（fast 窗口
  要自己选、dry-run 预览这种选择）时不注入任何风格块，pass 3 用窗口实际选中的那套（`CorrectionRun.picked_style_block`）。agentic 窗口选风格与译文是同一次交付，所以目录里没有 `default_style`（「无」是合法答案）时，选择提示末尾附出厂默认全文，写成「`selected` 填空字符串时按下面这套」（`fragment_style_no_match_factory_v1.md`；没有 `default_style` 时，选了候选外的名字的报错也说可以填空字符串）；之后的半窗由 `_pin_style_pick` 钉住。
  用户的风格在时它不注入：拷贝，不叠层。
  - 模板只有约定、正例、反例三节：没有标题与档案（会话靠 `<style name="出厂默认">` 认它，档案写的是维护者看的说明，抄进用户的风格又会让 auto 按它选歪），`knowledge new style` 把这三节抄进新条目。
  - `--style A,B` 时两套里一字不差的行只渲染一次（`_dedupe_stacked_lines`），改过的行照常各自渲染。
  - **已有风格的迁移**（`knowledge adopt-base-style`，`node/base_style.py`）：一次性 LLM 任务，一套风格一次调用，把出厂默认里
    这套风格没说到、又不冲突的约定加成同一条目的新一版。外层模板 `kb_base_style_v1.md` 自成一份、不套 `kb_repair_v1.md`：
    repair 的收录标准、writing 规则与完整提案语法都与「把通用的话抄进条目」相反。写库前在提案层（`translate` 之前）过筛：
    只收 `append_lines`、条目按句柄或 key 都要解析到这一套、小节只能是约定 / 正例 / 反例，逐行再查标记是出厂默认那一节里的、例子的约定在同一批里（否则引擎按孤例整批回滚）。回答里另有 `<skipped>` 块，逐行写
    没加的出厂默认行与原因，CLI 打印。
  - **待迁移名单**是 `meta` 表的 `factory_style_pending` 键（`local_id` 的 JSON 列表），**不升 schema**：升了的话，worktree 里一次
    只读打开就会把主仓的库升上去，`dev` 代码与退回旧版的用户都打不开它；旧代码不认识这个键，照常打开。新代码打开库时键还
    不存在，就在一个 IMMEDIATE 事务里把活着的 style 记进去（`seed_pending`，在自动导入之后；空库记空，并发的首次打开只写一次）。
    `--apply` 之后划掉那一套（一行没加也划），回滚、回答缺 `<knowledge_proposals>` 或 `<skipped>` 块、有提案被拒收时不划；`new` 建的不在名单上。纠错读到名单上的风格时告警一次
    （`knowledge-style-not-migrated`，`style.load_style_entries` 里）。记不上、不告警的三种情形写在这里、不另处理：迁移那一版被
    `knowledge revert` 退掉之后；风格从 share 拉进来；新代码第一次打开时它还不在库里。
- **收录**：`refined_aligned` 的知识更新任务把选中的 style 条目**钉进**每一块的
  `<kb_entries>`（不参与打分排序，它是任务的目标），prompt 的「翻译风格条目」一节讲成类
  做法、标记配对、上限与组合、以及「只动本次注入的那条 entry 的行」。
- **字幕长度档位**（owner 2026-11-04）：「档案」节的非 core 标记 `[字幕长度]`，行体只能是 `偏短` / `标准` / `偏长`
  （preset 的 `values`，比例在 `length_tier.TIER_RATIOS`：0.8 / 1.0 / 1.25，测试把两处的名字钉成一样）。它定 LLM 合并门槛的
  比例，在 `--split-length-scale` 之下、`config.toml` 之上；门槛怎么算、哪些缩放、比例怎么记与续跑见
  [`llm_harness_behavior.md`「合并门槛与任务的长度比例」](llm_harness_behavior.md#合并门槛与任务的长度比例)。用户向说明在
  [`manual/knowledge.md`「字幕长度档位」](manual/knowledge.md#字幕长度档位)。
- **任务标记**（owner 2026-11-05）：「档案」节的非 core 标记 `[任务]`，自由文本；写了 `中文纠错`（`style.CORRECT_ONLY_TASK`，可与别的并写，`for_correct_only` 按「、，, /」切开认）的风格，中文素材只纠错的任务整份注入，没写的只用于翻译（[`llm_harness_behavior.md`「只纠错」](llm_harness_behavior.md#只纠错)）。用户向说明在 [`manual/knowledge.md`「中文纠错用的风格」](manual/knowledge.md#中文纠错用的风格)。
  - **只由人写**（owner 2026-11-07）：翻译任务的知识更新没有理由改它，中文任务又不跑知识更新。preset 给这个标记设了 `model_hidden`，模型看的完整预览（`mode="prompt"`：知识更新、`ingest` 与 `repair`、共享冲突修复、agent 的 propose 读取）都不列它的说明，`rendered/` 照列；只有知识更新对它的新增、改写、删除、挪动一律跳过（`node.proposals.hidden_label_gate`，与 `[字幕长度]` 的闸门一起挂在 `_style_line_gate` 上，`HIDDEN_LABEL_HELD`），不算可改正的拒收、不回喂；按用户的话维护风格的 `ingest`、`repair` 不带闸门，和 `[字幕长度]` 一样。prompt 里不讲它；条目里已有的那一行模型看得到，知识更新里改不了。
  - **写入检查**：`importer.classify_line` 对带 `values` 的标记查行体，不在表里就是 `invalid`，于是走各条写路径已有的出口——
    模型提案（`append_lines` 与 `update`）跳过并带原因回喂，`knowledge edit` 报错，`rendered/` 收割把那一行泊进「待归类」。
  - **解析**：`style.profile_fields(text).length` 只读档案节（写在约定里的同名行是模型读的规则，不是档位）。
    `length_tier.styles_tier` 读任务级选定的那几套在注入那一版（`StyleSelection.rev`，`auto`、点名、只有 `default_style`
    都一样；选中那版渲染不了、退到当前版时读当前版）里的档位：叠用取点名顺序里第一套写了的，后面写了不同的告警
    （`style-length-tier-stacked`）；不认识的值告警、不参与（`style-length-tier-unknown`，库里只可能是绕过写入检查的旧数据）。
  - **fast 窗口自选风格**（`window_picks`）时档位不参与：窗口的 prompt 在选风格之前就写好了门槛数。选中那套的档位比例与任务的
    不同时告警一次（`length_tier.warn_window_pick`，`style-length-tier-window-pick`）。owner 2026-11-04 认定这种情况少见，不按窗
    重算分组去让它生效。
  - **dry-run** 时点名或只有 `default_style` 照常读档位；`auto` 有候选时风格未定、档位不计，`plan.json` 的 `length_scale` 带
    `note: 风格未定，档位未计入`。
  - **谁改档位**（owner 2026-11-05）：只按用户的话改。维护风格的流程（`ingest`、`repair`、人改）照用户说的写或改，档案节的
    标记说明告诉写风格的模型「要长句、短句写这一行，不写成约定」。任务后的知识更新（带 `--refined-srt`，与 `reference_ingest`）
    不主动加、改、删 `[字幕长度]`：refined 模板第 7 条只照用户在 `<task_summary>` 里的明确要求办：点名一档写那一档，要更长、更短在现有档位上
    挪一档（没有这一行按标准算，到头不动），要取消就删；精修稿看着更长或更短不算要求（2026-11-05 定向冷读后补的换档与删除）。
    **代码也拦**（`length_tier.tier_line_gate`，经 `apply_model_proposals(line_gate=)` 挂在逐行翻译上）：`run_knowledge_update`
    的 `user_summary` 为假（用户没给 `--task-summary`：任务用的是默认摘要，或 `reference_ingest` 自己写的摘要）时，新增、改写、
    删除、挪动 style 的 `[字幕长度]` 行的提案都跳过（`TIER_HELD`），它们不算可改正的拒收（`_is_rejection`），不触发提案回喂轮；
    为真时由模型判断那段话算不算要求，代码只拦挪动与写到档案节以外（`TIER_PROFILE_ONLY`，回喂给模型改）。回喂轮自己的落库也带
    同一个 gate。`ingest`、`repair` 与人改不带它。
    早先按实测学档位的那条路（精修与这次交付每行长度的比较给模型，由它决定写哪档）已删：一份精修稿的统计噪声大（换了精修者、
    双语稿、夹注释行都会让它偏），而档位改的是这套风格以后每个任务的合并上限。
  - `knowledge new style` 不抄档案节，新风格没有档位；`adopt-base-style` 只收出厂默认自己的标记，也写不进档位。
- **上限**在 preset 的节上（`max_lines` / `max_body_chars`）：apply 引擎在每条写路径上校验
  （模型提案与人改 `rendered/` 走同一处），节说明里渲染给模型看。三节各 **20 行 × 200 字
  （含 `[标记]`）**，owner 2026-09-02 拍的粗值、随时可改，不是标定值。⚠ 注入面付的是条目里
  **实际有的东西**，不是上限：一条真实条目（4 条约定 + 7 个例子）渲染出来约 600 字。
- **整条的注入上限**：preset 顶层的 `max_entry_tokens`（三个类别都是 6000）。逐节上限只管
  「一行多长、一节多少行」，三节都逼近上限时整条仍会很大。⚠ 它**不是写入规则**（owner
  2026-10-02）：超了照写，人改与模型写入都一样；注入时按行截到上限——纠错 system prompt 里的
  style 块逐套截，`<kb_entries>` 等注入块按「统一注入预算」截（[`llm_harness_behavior.md`](llm_harness_behavior.md)
  「注入上限」一节）。此前（2026-09-02 起）超过上限的 style 只准改短或删行，那条规则已删：拒绝写入会让新证据在
  有人压缩之前一直丢掉，压缩改由更新任务看到的提示来推动（见「条目的长度」）。
- **配对是可校验的**：`正例`/`反例` 的 preset 带 `labels_from = "约定"`，带标记的例子必须
  指向一条真实存在的约定——加孤例会被拒，删掉约定却留着配对的例子同样会被拒（无标记的
  例子合法，只是不参与配对）。
- **不进 resume 失效键**：`--style` 与 `extra_style` 都不作废已完成的窗（见
  [`llm_harness_behavior.md`](llm_harness_behavior.md) 的失效键一节）。

五节：`档案`（`[本名]` 即 `--style` 点名的串 / `[作者]` / `[适用范围]`）、`约定`、`正例`、
`反例`、`待归类`。一个 node 装不下结构（行不能挂子行），所以**规则与它的例子靠标记配对**：
`[引用标注]` 的约定、正例、反例是同一条规则的三面。「同一节内标记唯一」是 apply 引擎已有的
校验，于是「一条约定至多一个正例、至多一个反例」不用写代码就成立，孤例也可机器判定。
建条目：`python -m finesub.llm.knowledge new style <名字> --intro …`。

**行文法 v3**（2026-08-29）：每行是「可选 `[标记]` + 行体」。方括号是唯一的消歧符，
**冒号没有任何语法作用**；标记名不需要词表（`labels` 是登记表，只决定完整预览留不留空槽以及
`role`/`share`/`verify`），同一小节内标记唯一。行体形态自定：≥4 段竖线是**术语行**
`源语言|中文定名|别名|一句话描述`（别名列从 items 渲染、顿号分隔、无别名留空；desc 居末可含竖线），
恰三列报格式错误，其余是自由句；行体 kind 不在该节 `body_kinds` 内同样报错，绝不静默降级。
⚠ **别名共用本行的中文定名**（owner 2026-09-07）：判据是「这些叫法能不能共用一个译名」——拼写/假名/大小写变体共用，留在别名列；本名与固定绰号各有稳定译法的**拆成同一条目内的两条术语行**（不另建条目）并把绰号从本名那行的别名列摘掉，否则原文出现它一次会注入**两条译名不同的行**（`test_a_nickname_left_in_the_alias_column_injects_two_renderings` 钉着）；只因说话人/场景/双关才改译法的不拆行，条件写进第四列——术语命中只注入 `format_line` 出来的那一行，写在别处跟不过来。条目层面用于「找到谁」的别名不受此限。**四列格式不变**：`alias→译名` 字段消不掉语境判断，却要牵动解析、编辑回写、共享与迁移。
⚠ **common 的自由节收两种行体**（owner 2026-09-07，`term`+`note`）：术语行装需要独立匹配/定名/别名的专名，自由句装帮助理解与翻译但不值得独立成词的简短事实；排除流水账、攻略、无关百科与风格指令（归 style）。纯扩张，无存量迁移。⚠⚠ **代价是竖线的歧义搬进了散文真正会去的那些节**：混合节里 3 段竖线的自由句整条被拒、4 段起被当成术语行存下并进匹配面（`test_prose_with_pipes_is_read_as_a_term_in_a_mixed_section` 钉着），契约里让自由句改用「、」或斜杠；根治要动行文法，未做。⚠ 另一个连带：streamer→common 现在整条都装得下、什么都不泊，「节名有但不收这种行体」那条泊靠分支已无法由出厂 preset 造出，测试用`_preset_folder` 接缝覆盖。
⚠ **那两条竖线规则只在允许术语行的节里生效**（2026-09-02）：`body_kinds` 只有 `note` 的节，
行体一律是自由句，竖线只是字符。字幕的多人行本来就用 ` | ` 连接，按全局规则判会让真内容
写不进去，而它拦住的那个错误（在散文节里写术语行）那一节本来也承载不了。
⚠ **一行就是一个 node**，没有续行也没有转义换行：写入路径按 `splitlines()` 切，投影一行一个
bullet。要多行就拆成多行（各自是独立 node，有自己的版本、证据与共享位）；一句字幕内部真有
换行，按素材原样写 `\N`——那是内容，不是语法。
`invalid` 是通用出口：写入路径响亮拒绝并进 repair 轮；导入路径按上面那两档分流——行级泊进
「待归类」再由 repair 接手，文件级让文件保持 dirty。
**预览分两档，与 `mode`（human/prompt）正交**：完整预览（`preview="full"`）渲染全部小节
（空的也渲染）、每节的用途/收录纪律/标记说明（HTML 注释，收割时剥离；**登记的非 core 标记也带 note**——
只有 core 才占空槽，但两者都得让模型知道该往里写什么）与 core 标记的空槽；
部分预览（`preview="partial"`）三样都不渲染。prompt 投影不论哪一档都不带 `## 元数据`（更新日期对纠错与知识更新都没用，模型也不能改它；只读面 2026-10-19、可写面 2026-11-06），
只有条目带重分类理由时，完整预览留这一节、只写理由那一行（可写会话可能再挪这个条目）；人读的 `rendered/` 照旧带。分档看的是**这一面能不能写**，不是它是哪个
工具：人读的 `rendered/` 与 `show`/`edit` 回路、以及知识更新任务（agent 前端即
`kb_tools=propose`）得完整预览；纠错翻译等只读任务的注入面与它们的 `kb_read`
（`kb_tools=read`）得部分预览——脚手架是给写的人看的，只读任务拿到也用不上。
**自由节的那份注释只渲染一次**：`strict_sections = false` 的 preset（`common`）把每个未登记
节名都解析到同一个 `default_section`，逐节渲染会把同一段话在「角色」「地区」「组织」下各印
一遍。它本来就不是在讲某一节，而是在讲「任何自由节能收什么」，所以提到全部小节之前渲染一次、
用一句 `以下自由节…` 划定范围（`render.default_section_comment`）。**具名节仍各自带自己的那份**
——那里的文字确实是节局部的。`sections=` 过滤掉全部自由节时整块不渲染。
**空槽就是提醒本身**，不产生信号也不落库；`new`/`retire` 复用
`create_entry`/`retire_entry` 语义；`revert <rev>` 是 §2.5 的补偿事务（全有或全无，被更晚触碰的实体阻塞
时拒绝并列出阻塞者）；`restore <local_id>` 把退役节点在**同一 id** 下复活并带回同一 rev 关闭的边/items/
links；`phase-b` 是归档形态 → v3 的确定式转换：fact 字段命中标记表转
`[标记] 值`、`其他` 大杂烩按 `；` 拆行、空值退役、relation → 术语体、event 连同「重要经历」退休、
`说话风格`/`喜好 / 特点` 并入「特点」、别名行退役为 items 投影、占位符 item（老库英文列的 `—`）
退役、折叠后等于表层的读音不登记为别名；**v1 按冒号误切成 fact 的散文整行复原**（不去归一那个
本就不该存在的分隔符——旧 sep 归一在书名号里的冒号上改过内容，已永久删除）。判断题一律停在
`待归类` 或留成候选，`--report DIR` 落 JSON 与**逐行对照表**（Phase B 之后 legacy 投影不再适用，
对照表是唯一的对照面）；
`repair` 是判断题的 LLM 半（§11.3/O3，`node/repair.py` + `kb_repair_v1.md`）：把候选（`staging-line` /
`unnamed-term` / `episodic-desc` / `duplicate-term` / `chapter-alias` / `orphan-section`，带 `@c` 句柄）连同条目 prompt
投影交给会话，产出
`<knowledge_proposals>` 走同一条 validate→apply 通道，另输出 `<candidate_verdicts>`
（`propose|dismiss|needs_human` 逐条裁定）——默认 dry-run 只渲染 prompt，`--execute`
跑会话但**只展示提案**，`--apply` 才落库并把裁定写进候选决策账本（`candidate_decisions` 表，
kb-followups A6）：`dismiss`→resolved、`needs_human`→pending_human（必须写明缺什么证据——missing 缺失时回退 reason、
两者皆空保持未决；进 report 的「待人工裁定」小节，人工用 `candidates --resolve <key>` 关闭）、
`propose` 仅当 **apply 后重扫不再产出该候选**才 resolved（完成断言是扫描本身，部分落地的修复保持
未决重入），已决候选不再重入下一轮，候选内容变化时旧决定自动作废重新可见；`verify` 是校验任务（§11.4/O9，`knowledge/verify.py` +
`kb_verify_v1.md`）：目标集合由 preset 的 `verify` 策略决定（label → section → `unknown_label_verify`，
**fail-closed 在 none**）——人际关系/特点/直播内容/待归类一律不外发，未登记的标记（`[中之人]` 这类）
永远不外发；tentative 也不占坑，`--execute` 跑 native-search 会话、`--apply` 把结论落成
`evidence_kind=external` 行（confirmed 必须带 URL，查不到记 `unverifiable` 终态防重扫，`note` 写查过哪些方向，终端打印）。核到什么程度算 `confirmed`：term 以源词、中文定名与身份为准，描述细节查不到不影响，描述被明确反驳判 `refuted`；档案 fact 核那一行的事实（owner 2026-10-19 选的宽松口径：全部核实才算的话，网上查不全描述的术语大多会永久记成 `unverifiable`，tentative 也升不了正式；代价是 `confirmed` 也替没核实的描述背书）。
`repair`/`verify` 都收 `--llm-model [任务组=]值`（可重复；值是模型组或 route target，对应格
**整体换绑、无回退**——verify 强制 native search，钉单模型要选 `supports_native_search` 为真的模型如
`local-claude-opus`，不能搜的会响亮失败）做**运行时**模型指定——按
preferred_targets 语义装进进程级路由 overlay，仅本进程、不落 config.toml。维护会话
（repair/verify/共享 review/材料吸收/共享冲突修复）按会话注入（2026-10-04，冷读 Q7）：判断标准
`fragment_kb_judgment_v1.md`（收录标准、同一实体判定、证据纪律）都给；写条目的规则
`fragment_kb_writing_v1.md`（有无参照物、行的归属、描述纪律、标记命名）只给写条目的修复、材料吸收与共享冲突修复。
共享冲突修复也不拿整份 op 契约，只拿它的闸门放得过的那几种（见下面「pull 冲突台账」一段）；修复、材料吸收与任务后的知识更新
仍拿整份 op 契约（owner 2026-10-04：它们的闸门放行全部 op）。
**判断标准第 0 条是收录标准**（2026-09-01）：知识库只为「让 ASR 听对」（源语言表层、读音、易错写法）
与「让字幕写对」（中文定名、别名、语体/自称）两件事服务，两者都不服务的不收；preset 各节
注释里建议的标记就是它的具体化。⚠ 同一条标准下，**有没有 ASR 文本作参照物决定能写多硬**：
任务后的知识更新手上有那次的 ASR 与精修对照，可以写「误听: xxx」；材料吸收看**手上这份材料**：
同时带着「机器听成什么」与「正确的是什么」的（如旧台账的 `原文片段 + 正确译文 + 说明`）同样是对照物，可以写误听，
**没有 ASR 文本的材料不写误听**。`--prompt` 只偏向收什么、怎么写，越不过这条标准。
对话式编辑（用户直接让
agent 更新知识库）走同一套 `edit`/`apply` 命令。写命令都走既有跨进程写锁。

旧形态（每条目一个 markdown 文件 + 内嵌 git）在第一次打开知识库根时**一次性无损导入**（`KnowledgeRepo.open`：
发现没有 `knowledge.sqlite` 而有 `streamer/`/`common/` 的 `*.md` 就导入，打 `knowledge-migrated` 告警）；
原 markdown 目录与 `.git` 保留为历史档案，不再被读写。index 行里有而条目文件里没有的别名/简介会并入条目
（无损）；只有 index 行没有文件的条目导入为「index-only」subject。导入是确定性的（UUIDv5），重复导入得到
同一套 id。（`translation/` 下那两个 markdown 台账 2026-09-02 退役，见下。）
也不再有 git 历史。

**git 不再是知识库更新的前置条件**：`required_capabilities` 不再为 `--knowledge update` 索要 git，
`finesub doctor` 把 git 列为可选。**写入失败一律降级为 warning，绝不失败整个任务**：字幕是主产物，
知识库是副产物。两条路径**不推进 chunk ledger**，重跑会完整重做：①另一个进程持有写锁（跳过应用，
提案保留）；②store 提交后、ledger 写入前崩溃——`revisions.note` 记着原始提案文本的哈希，下次运行据此识别为
`recovered_after_commit`。整批提案叠到库上校验不过时，只拒收出问题的那几条、其余照常提交（见下面
「整批校验不过时的拒收」）；查不出是哪条提案造成的才整体回滚（`knowledge-apply-rolled-back` 告警），
这一块什么都没写进库，但 ledger **照常推进**：重跑同一个任务不会重做这一块。

**知识库根的解析顺序**（`finesub.paths.resolve_knowledge_root`）：显式
`--knowledge-root` → `FINESUB_KNOWLEDGE_ROOT` → **源码 checkout 的 `<repo>/knowledge`** →
发行包自身的 user-data → 本机受管安装的 user-data（`%LOCALAPPDATA%\FineSub\user-data`）。
绝不静默落到 CWD。

- **checkout 优先是默认值**：几乎所有从仓库发起的运行都是开发，而真实 API key 就在
  `<repo>/.env`；把它们导向共享知识库会让开发噪声与真实条目在提交流里交织，而分裂只需事后
  合并一次。`FINESUB_CHECKOUT_DATA=0` 显式退出。只在**代码本身在某个 checkout 里**时才用（`paths.own_checkout_root`），这时先按当前目录、再按代码位置找
  checkout（两个独立 clone 共用一份 editable 安装时，用的是当前目录那个）；安装版（打包启动器的运行环境、wheel）在 checkout 目录里运行不用它的数据，否则任何长得像 checkout 的目录都能塞给它
  `.env`、`config.toml`、编译缓存与 token 计数器（审计 X-3，owner 2026-11-07）。**git worktree 解析到主仓**（`.git` 文件里的
  `gitdir:` 上溯三段）。worktree 里对**共享知识库**（主 checkout 的与安装的那份，
  `paths.shared_knowledge_roots`）的自动写入一律**跳过并告警**，除非 `FINESUB_KNOWLEDGE_WRITE=1`：
  任务后更新、运行开始时的 rendered/ 收割、以及命中/曝光/落地信号事件（agent 侧的曝光由 harness
  在任务清单里写 `kb_signals: "off"` 转告 MCP server）。判据看**写到哪个库**，不看根是怎么给的——
  显式 `--knowledge-root` 指向主仓也照拦；指向别的目录（测试的 tmp 根）照写。闸门只有一处
  （`knowledge/base.py` 的 `worktree_blocks_knowledge_write`）；信号写入只问闸门、不拿写锁。
  `maintain`、`share` 这类由用户亲手发起的命令不问它。
- CLI 壳正常会注入 `FINESUB_KNOWLEDGE_ROOT`；后两档是给「绕过启动器、直接用包内
  解释器跑 pipeline」兜底：按模块所在的 `app/versions/<版本>` 布局反推安装根
  （`finesub_bootstrap.paths.packaged_app_root`）。发行包同样带 `pyproject.toml` +
  `src/finesub`，所以 checkout 探测**显式排除**这种布局——否则知识库会写进
  `app/versions/<版本>/knowledge`，而 `app/` 会被下次更新整体替换，数据静默消失。
- `.env`、`config.toml`、限流状态文件按同样顺序解析。三种终端用户形态（安装版/便携版/CLI）的
  user-data 现在是**同一个目录**，所以同一个用户只有一份知识库。

**跨进程写锁**：知识库 auto-apply 在 `<knowledge_root>.lock` 上排他（知识库目录的兄弟文件——
放在目录里会被 auto-commit 收编，锚在任何安装根下则三个前端锁的是三个不同文件）。等待超时的
行为与"仓库脏"一致：跳过应用、保留提案、不推进 ledger、只打 warning。**同进程的多个写者共享
这把锁**（引用计数，2026-08-30，任务级并行 W2）：
退让语义只对别的进程生效，进程内的并行 task 改为在 apply 的 **per-root 有序单写者队列**
（`node/apply.py` `_FifoLock`，按到达序交接）里等待毫秒级的写事务，不再白丢一次生成。

**并发冲突按类型化解，不整包回滚**（2026-08-30）：唯一性输家若冲突对方是
**read_rev 之后**才落库的（作者当时不可能看见——真并发竞争），只丢冲突 op 及其依赖闭包：
relabel 回退到 stored 标记（同 op 的其它改动照常提交）、新建同名节点连 items/memberships/links
一起丢、并发移动落到刚被占用的小节只丢那条 move。作者当时可见的冲突（用户编辑面依赖被明确
告知）与批内自撞是作者错误，交给整批校验：知识更新只拒收写重的那条（见下一段），其余调用方整包回滚。全部 op 都被化解/拒绝时**不再空转一个 revision**（`rev=None`）——
两遍化解都算：进队列前那遍，以及**事务内**对着权威状态重做的那遍（竞争者恰好落在 preview 与
BEGIN IMMEDIATE 之间时只有后者能看见，2026-08-30 复审补上，否则会提交一个零行 revision 并
报成功）。出处记账（`book_revision_evidence`）也在**同一个事务内**完成——它是自动提交连接上的
逐行 INSERT，放在提交之后等于在单写者队列之外逐行抢写锁（并发 apply 下实测 `database is
locked`），且是唯一可能在 revision 已提交之后失败的写。建表/迁移（`init_schema`）同样走
`BEGIN IMMEDIATE`：延迟事务先拿读锁再升级，两个线程同时开库就是互等的升级死锁，SQLite 对它
立刻报 `database is locked` 而不理会 `busy_timeout`。
冲突从 `result.conflicts` 升为用户可见面：task report 的知识小节逐条列出、
`knowledge-apply-conflict` 告警、batch 的 batch-status.jsonl 收 `knowledge-conflict` 事件。

**整批校验不过时的拒收**（owner 2026-11-09）：翻译层把提案译成写库操作时已经逐条拒收一部分（条目不存在、
本节已有同名标记等）；剩下的操作叠到库上之后，引擎再查一遍整批的一致性（`apply.validate_overlay`）。知识更新
（主轮与提案回喂轮）在这一步**只拒收出问题的操作**，原因写成 `整批校验：<问题原文>`，其余照常提交；被拒收的照
「提案回喂」进修复轮。其余调用方——`knowledge edit`、维护会话、`adopt-base-style` 迁移、`knowledge verify` 与共享
冲突的修复——仍整批回滚：迁移依赖它（例子的约定要和例子在同一批里），人点名的一批改动半提交会让人意外
（`apply_envelope(invalid_ops="rollback"|"reject")`，默认回滚）。
- **追到操作**：每个问题由查出它的检查归给本批写它的操作；一个实体被同一批的几个操作写过（建、再改）时全部拒收，
  问题是它们合起来造成的，挑一个是猜。拒收后从头重建、再查，直到干净。
- **同一节同名标记**归给本批**最后**写进这一节的那一行（按操作序号；新建或改了标记的节点、把它放进来的归属边才算，
  只改正文、原地不动的库存行仍算库里原有）：先写的胜，库里原有的总胜过本批的。以前按节点 id 排序点名，id 是随机的，一半时候点名模型自己那行、一半时候点名
  `create_entry` 补的那行；回滚原因现在也指向真正改动的那行。
- **追不到的**整批回滚，原因只列追不到的问题：问题实体只是被读进来、本批没写（库里原有的两行同名、库里原有的
  超长行），或拒收不出新的操作。事务内与并发写者竞争之后剩下的问题也照旧回滚（那是并发造成的，不是作者的错）。
- **连带**：依赖被拒收的新建行的操作（行的误听、别名，挂在新条目下的行）随之作废，原因是
  `随被拒收的那一行一起作废：<根原因>`，不写翻译层内部的 `@newN` 句柄。
- **只剩改日期时不提交**：翻译层给每个被触及的条目补一个改 `updated_date` 的操作；实质操作全被拒收时不留一个
  「只改了日期」的修订。`create_entry` 补的 `[本名]` 行是实际数据，不算记账。
- 翻译层能查出的就在翻译层拒收：`create_entry`、`split_entry` 补的 `[本名]` 行登记进同一批的标记账，同一批里再写
  一行 `[本名]` 在翻译时就被拒（「标记 [本名] 在本节已存在——改用 update」），不必等到整批校验。

| 整批校验的问题 | 归给谁 |
| --- | --- |
| 节点载荷不合法 | 写这个节点的操作 |
| 同一节同名标记 | 本批最后写进这一节的那一行；都是库里原有的就追不到 |
| 小节行数超限 | 本批加进或挪进这一节的归属边 |
| 归属边重复、父或子不在、小节或行体不合模版 | 这条归属边的操作；对面是库里原有的那条时取本批这条；只读载入的边指向本批退役的节点时归那个节点的操作 |
| 链接、条目项的主人不在 | 这条链接、条目项的操作 |
| 重分类装不下、重分类到不存在的类别、带子条目的条目改类别 | 条目节点上的操作 |
| 别名项落在没有别名字段的类别上 | 这条别名项的操作 |
| style 例子的 `labels_from` 标记不在 | 例子是本批写的或挪进来的就取它；否则取本批退役、改标记、挪出或删了归属边的、读取版本上带这个标记的约定行；都没有就追不到 |
| 每行字数超限 | 本批写的或挪进来的就取它；库里原有的超长行追不到 |

加一种整批校验问题时，在 `_overlay_problems` 里写明它归给哪些操作，或有意留空（留空就整批回滚），并补进这张表。
`kb_validate`（agent 交提案前自检）只预览、不提交，照旧返回全部问题。

**提案回喂（B'，2026-08-31；被拒的 op，2026-10-23）**：提案里有没写进去的，harness 追加**一轮**修复调用，
只让模型补这几条（其余已落库，不重交）。这一轮的回答读不了时，同主轮一样带上原因重问、同一个上限；内容上仍只回喂一轮，它又被拒的照旧记在报告里。两种原因各占一节、有才出现：
- **并发被丢**（B'）：告诉模型「你读的是 rev X、现在是 rev Y、你的这几条因为什么被丢」，别人已写等价内容就跳过。
  只有真有**外来** revision 才回喂：判据是 rev 里除主 apply 自己的提交外还有没有别人的（光看
  `current_rev > read_rev` 不行——主 apply 只要提交就自增 rev，自撞也会被误当并发）。
- **被拒收**（owner 2026-10-23）：apply 层拒掉、记录里带条目名的 op（op 不认识、父节点找不到、行文法不合、改名撞了已有的键、
  整批校验查出的问题等），按拒收原因改正后重提。结果其实已经对了的不回喂：行或别名本来就有、`create_entry` 的条目已经存在、改前改后一样。
  不带条目名的拒收（句柄找不到、行不是 JSON）没有条目可投影，也不回喂，只在 apply 报告里。
两节都**逐条附上被丢提案的原文**（只附这一条出自的那份提案——按报告记录里的提案序号 `proposal` 找，`update` 只给句柄、不带条目名，按条目找会附错——同条目已落库的不附；原始 JSON，它自带 `content` 与模型当时给的 `reason`，所以证据也在里面），
并把相关条目在**当前 rev** 的 prompt 投影连同**该版本重新分配的句柄**一起给它（输出空块是正常结果）。
回喂有自己的 system 头（`knowledge_proposal_repair_system_v1.md`，owner 2026-11-06）：输入只有这两样，原提案的 `reason` 就是证据，
所以知识更新的窗口包与 raw/final 的输入说明、置信门槛与子条目规则都不给它；结构说明与输出格式与知识更新共用。
回喂轮不再注入本轮材料——看不到证据还要「补充」等于凭空捏造。第二个 envelope 以当前 rev 为
`knowledge_read_rev`、绑新句柄，正是冲突第一次没通过的 CAS 前提。
**整个环节是尽力而为**：主提案已经落库，回喂失败/被拒/返回空都只打 `knowledge-proposal-repair-failed`
告警，运行照常成功（失败轮的 exchange 也照记——验证没过的那轮恰恰是要回看的那轮）；一块只回喂一轮
（这一轮再被拒的只进报告），最多带 `MAX_REPAIR_ENTRIES`(6) 个条目（并发被丢的优先），超过说明该重跑整块。
修复轮的 apply 同样只拒收出问题的操作。整包回滚（问题追不到本批的哪条提案）不走这条路，只打告警。

**迁移**（`finesub_bootstrap/migrations/`，按 id 记账于 `user-data/.migrations.json`，启动器与
命令行都会在读用户数据前跑一次，跨进程加锁，失败只告警并在下次重试）：
`0001` 把 `app/versions/*/knowledge` 搬回 `user-data/knowledge`，`0002` 把便携包内的整个
`user-data` 搬到 `%LOCALAPPDATA%`。两处都有时**不自动合并**，持续告警直到人工处理。

目录结构：

```text
knowledge/
  knowledge.sqlite      # 真相源（revisions / nodes / node_versions / items / memberships / links / evidence / events …）
  rendered/
    streamer/{index.md,<key>.md}   # 派生缓存（可编辑投影），写入后重生成、手改在 run 前收割
    common/{index.md,<key>.md}
  streamer/ common/     # 旧 markdown 形态，导入后只作历史档案
```

index 每条目一行、四字段（v14）：`key [类型] | 其他语言本名 | 别名 | 一句简介`（`[类型]` 仅 common 使用，如 `[游戏]`、`[梗]`）。key = 条目 H1 = 文件名 = **源语言本名**；「其他语言本名」是正式名的中/英等写法；「别名」收窄为昵称、简称、全称、俗称（误听变体不进 index）。三列 + key 都参与备注关键词匹配与按名索取。整个 index 行由渲染层从 subject 派生——key/本名/简介来自 payload，别名列来自检索索引项 `items(aliases)`（经 `add_item`、term 行第三列（别名列）或归档导入期的旧五段行登记）；proposal 不携带 index 字段。旧 3 字段行仍可解析。

条目结构（行文法 v3，两类共通）：H1 → 一句话描述 → `## 档案` → 若干二级节 → `## 元数据`（永远最后，apply 层自动维护 `最近更新日期`，模型不可修改；prompt 投影不带这一节，见上面预览分两档那段）。新条目由 `create_entry` 按 preset 骨架建档，只物化**有内容**的行（身份标记由 preset 的 `role` 指出，不是硬编码的 `本名`）。preset 是领域 schema：`src/finesub/llm/knowledge/node/presets/{streamer,common}.toml`，定义固定节、每节的 `body_kinds`、标记登记表（`core` 决定完整预览留不留空槽）、`purpose`/`exclude` 收录纪律与 `share`/`verify` 策略；section 没有 tier——空节与空槽在完整预览里全渲染，注入面全不渲染。

- streamer 六节：`档案`（`[本名]` `[别名]` `[人设]` `[外观]`）、`直播内容`、`频道用语`（术语行，`[自称]`）、`特点`（`[语体]` `[音色]`；旧 `说话风格` 与 `喜好 / 特点` 并入）、`人际关系`（术语行）、`待归类`（暂存区，非空即出信号）。旧 `重要经历` 与 `event` kind 已退休。
- common 分类节由模型自由分组命名（持续更新的游戏建议按大版本开二级节，节内可用 `###` 三级目录细分人名/地名/其他专有名词；预设骨架含 档案+待归类+元数据——`待归类` 2026-09-03 补上，收割手改时行级泊位需要它，否则整份编辑只能被拒）；分类行**固定四列文法**（行文法 v3，见上）`源语言|中文定名|别名|一句话描述`——别名恒在第三列、由 `items(aliases)` 渲染（无别名留空、顿号分隔），特殊读音写进别名（假名写法即参与匹配），描述居末**可含竖线**，恰三列报格式错误；描述内可带「误听: xxx」作**录入语法**（自动转为 misheard item 并从描述中剥离；misheard 只供匹配器消费，不再渲染）。旧五段行仅归档导入期可解析（别名列与读音都归入 items）。streamer 另有 `频道用语` 节收频道自造词（粉丝统称、吉祥物名等），行文法同上。碎知识（只有一两行可说、附属既有主题的专名/梗）作为行进母条目分类节，不建独立文件。
- 尺寸纪律：见下「条目的长度」与「子条目」。
- **条目的长度**（2026-10-02）：每个类别的 preset 声明三件事——`max_entry_tokens`（6000，注入时的截断上限，不是写入规则）、`soft_entry_tokens`（5000，过了就提示压缩）、`splittable`（能否按节拆子条目：`common` 能，`streamer` 与 style 不能）。超过软阈值的条目，在**能整条改写它的面**上顶部挂一行提示——知识更新的预取、agent 在 `propose` 授权下整条 `kb_read` 的回复；纠错窗口不挂（不能写），候选修复、材料吸收与提案回喂也不挂（提案回喂同样持有 `propose`，它的 agent 任务带 `kb_size_notice: off`，`kb_read` 因此也不挂）（只许改输入点名的内容，挂上只会被忽略或诱导越权；2026-10-02 的 prompt 冷读里 5 个读者全部忽略了它，owner 2026-10-03），提案回喂里超长的条目照样整条给出。提示只给一个百分比「建议至少删减 X%，删得更多也可以」，X 由程序按压到软阈值的八成换算（5000 → 20%、6000 → 35%、8000 → 50%，`entry_size.compression_percent`），不写目标长度也不写 token 数——给了目标模型会停在目标上，它对长度的感知也不准；超过 6000 时另说一句纠错时超出的部分会被截掉。要压缩的条目在知识更新预取里与钉选的 style 一样先整条分配、不截到 6000（看不全就压不了），合计超过半块时排在后面的退回共享分配、照常截，这次也不挂提示。⚠ 所以一条要压缩的条目**自己就超过半块**（预取 60k 的一半，约 3 万 token）时，知识更新永远整条看不到它、也就永远不提示——正常增长到不了这一步（过 5000 就开始提示，此后每次更新都整条可见），只有模型长期不理提示或一次性导入超长条目才会；agent 可以分节读，其余只靠报告。知识库报告（`python -m finesub.llm.knowledge.report`）另列超过 6000 的条目与纠错注入时会少掉的节，模型一直不压缩时人看得见。为什么不是写入规则：拒绝写入会让新证据在有人压缩之前一直丢掉，而人改也会被拦（owner 2026-10-02）。为什么先压缩、满足判据才拆：拆分只要一行 JSON、压缩要逐行改，常驻的拆分选项会压过压缩，所以`split_entry` 只在提示建议时出现。提示文字在 `fragment_entry_size_notice_v1.md` / `fragment_entry_size_over_cap_v1.md`。
- **子条目**（2026-10-02，`node/subentries.py`）：`splittable` 的类别（今天只有 `common`）可以把一节整节拆成子条目——模型的 `split_entry`、人手的 `python -m finesub.llm.knowledge split <条目> <节> --intro …`，走同一个引擎操作。子条目名由系统派生为 `母条目·节名`；类别、`entry_type`、共享设置（visibility）继承母条目，那一节的行原样挪过去（节点、历史、证据与人改版注记都跟着走），母子关系是一条 `part_of` 链接（子 → 母），母条目的 `section_order` 去掉这一节、子条目只有 `[档案, 节]`。两端头部各写一行「子条目：…」/「属于：…」——prompt 投影是普通行，`rendered/` 里是整行注释（H1 与第一节之间多一行普通文字会让收割失败）。子条目只要求身份槽，不再给别名、收录范围留空槽。规则：`档案` 不能拆、子条目不能再拆、`streamer` 与 style 不能拆（主播条目按名字整条注入，拆出去的节会离开默认注入；style 按名字静态选取），母条目与子条目都不能改类别；母条目改名（模型或人改 H1）时，名字仍是派生名的子条目跟着改，撞上别的条目则保留旧名；母条目退役时链接随之关闭，`restore` 带回母条目时只在子条目还在、同类别、自己不是母条目时恢复链接。索引里子条目紧跟母条目；选中母条目不会自动带上子条目（带上就等于没拆）——术语行另有按术语注入的路，名字出现在用户备注里时照样够得着；知识更新按反馈的 `sub` 记分时，那一行若已在子条目里，就给子条目记分。拆分说明只在需要时出现：可拆条目超过软阈值时，提示里多一段拆分判据（不是条目头、约 2500 token 以上——在满块的分配水位之上，单独成条后在满块里也能整条进——以术语行为主，附逐节大小与术语行占比；不做成工具：API 前端没有工具可调，agent 要多一次往返，说明本身只有几行），同一次的操作说明才带上 `split_entry` 那一行；agent 整条 `kb_read` 时这行随回复给出。`part_of` 随共享导出。已知边界：单独一节就超过 6000 时，拆出的子条目仍超限，而子条目不能再拆——出路是拆之前先把那一节的行分到几个新节，再逐节拆。不做按命中裁剪（只注入和窗口文本命中的行）：node 设计稿 v8 已取消。⚠ 共享设置显式继承：不挂父节点的新 subject 默认是 `local`，随后每一行挪进来又会被降成 `local`、只降不升——拆一个已共享的条目就等于把那一节悄悄撤出共享。已知边界：改名连带撞名时子条目保留旧名、告警`knowledge-subentry-rename-kept`，母条目照常改名（为罕见的撞名回滚一次无关的改名不划算）；同一批里先 `split_entry`、再给母条目改名，子条目按旧名派生（下一次改名会带过去）；用 CLI `split` 拆出的子条目，母条目之后被模型改名时子条目的 H1 会挂人改版注记（那是人那次操作写下的值），`knowledge ack` 即可。

**改模版是一次迁移，账记在 `test/data/preset_migrations.json`**（2026-09-07）：每个类别一条
`{digest, migration}`，`digest` 是 `preset_digest()` 对**解析后形状**的哈希（改注释不算改模版），
`migration` 用一句话说明存量条目怎么办（不需要动就写 `none — <为什么>`）。守卫在
`test_llm_knowledge_presets.py`，改了 preset 而没动这份账就红。⚠ **触发条件是内容不是 `version` 号**
——删一节、改一个标记名、收紧一个 `max_lines`，`version` 照样是 2 而存量已经不合规。
⚠ 守卫拦不住「只改 digest 不想迁移」，这是有意的：没有守卫能判断人有没有真想过，它能做的是让这次
改动无法在没人写下那句话的情况下落地。存量不合规的**发现**面是 `scan` 的 `orphan-section` 候选
（报告，不拒绝写入——拒绝会把条目锁死在它最差的形态）。

**preset TOML 是结构的唯一真相源，模版与 prompt 都是它的投影。** 完整预览里每节的
用途/收录纪律注释、以及 prompt fragment `fragment_knowledge_structure_v1.md` 的结构一节，
都由 `render_structure_spec()` 从 preset 生成，不手写——三处并存（preset / 代码常量 /
手写 fragment）曾经真的漂移过（fragment 教三列术语行、解析器要四列）。**类别的一句话说明
（`blurb`）与「进不进匹配面」（`matchable`）2026-09-07 也归了 preset**，此前前者是 `render.py`
里一份写死的三行字典、后者是 `model.py` 里一个写死的元组。方向只能是
**preset → 模版**：反过来从模版 markdown 解析结构，等于给注释另造一套迷你语法，回到
「从文本猜结构」。两个特设标记靠 `role` 数据化（`identity` = 承载 native_names、
`aliases` = 承载别名索引），index 派生、`create_entry` 骨架、编辑面同步一律**按 role 查**，
代码里不再匹配字面量 `本名`/`别名`；只剩 `元数据` 与 `最近更新日期` 两个真常量，它们是
harness 自有结构，preset 声明它们会被 loader 拒绝。机制在代码（`classify_line` 的算法），
策略在数据（`body_kinds` 这一节允许什么行体）。

**加一个类别就是两步**：往 `node/presets/` 放一份 TOML，再在 `node/presets.py` 的
`CATEGORIES` 里加一个名字（`matchable` 决定它进不进匹配面，`blurb` 是它的一句话说明）。
实现上三处接线自动跟上，不需要在别处加分支：**索引**（只含 matchable 的那些，含 exchange
里的逐类别计数键）、**模板**（维护会话收到的结构表，含 standalone）、**store**（preset 自己
加载，带自己的节与上限）。端到端验收是 `test_llm_knowledge_presets.py` 的
`test_a_new_category_reaches_every_face_without_editing_them`——它用一个**不进产品**的测试类别
（`seiyuu`）走完整条路，所以验收本身不会变成一次产品决定。⚠ 新类别**也要在迁移账里占一行**
（`migration` 写 `none — 新类别，没有存量条目` 之类），守卫按 `set(ledger) == set(CATEGORIES)`
卡；上面那条迁移账讲的是改**已有** preset 内容时怎么办。

⚠ 那三处是**代码接线**，不要和 prompt 里的三套类别集合混成一句。prompt 里是索引面（matchable）、
模板面（有精修的知识更新含 style；无精修与提案回喂只给可写的类别，开头的类别清单仍列全）、**可写面**（本次会话的 `allow_categories`，由
`render_knowledge_ops_contract` 渲染成枚举）。可写面跟模板面在同一条 prompt 里只隔几段：漏掉它
会让模型读得到某个类别的模版、却被同一条 prompt 告知它不存在。可匹配类别共用一个命名空间，
所以 `recategorize` 的查重必须排除自身，否则每次搬迁都被判成与自己重名。

⚠ **动这一族的接线之前还有四条**（2026-09-07 实施时踩到的）：

- **严格审核槽读 preset 的 `verify_for`，不按类别名判断。** 旧的 `_slot_for` 只认 `streamer`
  进身份严格槽，于是**加一个类别反而让审核变松**。改成读 preset 之后，`common` 的身份字段也
  顺带收紧进了严格槽。
- **`validate_overlay` 的问题要说清归给哪些操作。** 知识更新按它把出问题的操作拒收、交修复轮，其余提交；
  归不到本批任何操作的问题仍让整批回滚，其余调用方更是一律整批回滚（「整批校验不过时的拒收」）。在那里 append
  一条 problem，对维护会话等于让同一批的所有改动陪葬，对知识更新也至少丢掉那条操作。所以重分类装不下的行是**泊进目标类别的暂存节**
  （`apply._park_if_homeless`，报告里记 `parked` + 一条 warning），`_check_recategorized` 只做兜底；
  判据只问「这次移动有没有把它弄坏」，本来就不合自己 preset 的旧行原样放行。
- **删 `if/else` 分支时，`else` 会静默改挂到上一个 `if` 上**，`git diff` 看不出来——删 search
  loop v2 时就这样红了 11 个测试。
- **`tools/session_replay` 的四个会话适配器随这一族一起改**，而它们的测试不在默认套件里：
  改完手动跑 `python -m pytest tools/session_replay/test_session_replay.py`。

**preset loader 严格校验，不做任何兜底转换**——下列一律报错拒绝加载：重复标记名（含
NFKC + 去空白后同名）、重复 `role`、`core`/`allow_custom_labels`/`staging` 非布尔（不接受
字符串——`bool("false")` 是 `True`，一个笔误就能把标记升成 core）、`share`/`verify`/
`body_kinds` 取值越界、未知键（拼错的键必须炸）、声明禁止的 section、`core = true` 却没写
`note`（空槽没有说明等于没有脚手架）、标记的 `values` 不是非空字符串的列表或有重复、`model_hidden` 非布尔或设在 core 标记上。

标记可以带 `values`：这个标记下的行体只能是表里的值（style 的 `[字幕长度]`，见上「style 类别」的字幕长度档位），
`classify_line` 把别的行体判成 `invalid`；节说明给模型渲染出这几个值。`values` 只在设了时进 `preset_digest`，
没设的标记 digest 不变。标记可以设 `model_hidden = true`：只由人写的标记（style 的 `[任务]`），模型看的完整预览不列它的说明；它不改条目合不合规，所以不进 digest。

**共享策略按 section 决定**（`share_inherit`），单个标记可用 `share` 覆盖；未登记的标记走
`unknown_label_share`，两份 preset 都是 `local`——**fail-closed，不随本节默认自动继承**。
按 kind 判定的旧口径已废：人际关系改成术语行之后，按 kind 判会让它随 term 自动继承 shareable。
外部验证策略同构（`verify` → section → `unknown_label_verify`，fail-closed 在 `none`）。

**导入 parity 是真闸门**：`KnowledgeRepo.open` 的 auto-import 也跑 `check_parity`，失败即
抛 `ImportParityError` 并**删掉半成品库**（旧行为只打一条 warning 就带着可能损坏的库继续）。
Phase B 之后 legacy 投影回放不再适用，替代对照面是该步的**逐行新旧对照表**（`--report DIR`）。

翻译风格**是**一个知识类别（2026-09-02 起，见上「style 类别」）：一套口味是一个条目，
`--style` 点名注入。`--extra-style` 仍在，是这一次临时加的一段自由文本，与 style 条目注入
同一位置。prompt/harness 模板在 `src/finesub/llm/prompt_templates/`，由主 git 追踪。

## 两个翻译台账（已退役，2026-09-02）

`knowledge/translation/` 下的 `common-mistake.md` 与 `good-example.md` 不再存在于知识库里。
一条翻译约定现在是 `style` 条目里的一行（见上「style 类别」）：模型的写入块
`<mistake_proposals>`、`## 精选` 的注入、以及 `mistakes.py` 整个模块都已删除。

两份原文搬到本地 `data/legacy-translation-ledgers/`（同目录的 README 记着各自的下场）：

- **`good-example.md`（27 条）已迁移**：蒸馏成 `style/default_style` 的 5 条约定 + 5 个配对
  正例（rev 53）。
- **`common-mistake.md`（27 条）判定不迁**（owner 2026-09-02）：18/27 的说明点名「误听/同音」，
  整个库是同一课重复十几遍，而现行纠错 prompt 早写着这一课；17/27 来自 prompt v16/v17（当时
  已是 v82）。唯一耐久的残渣是专名与它们的误听形态，那是词条材料——对真 `原神` 条目跑过 dry run
  （14 条提案）但没有 apply，材料留在那个目录里，改主意时用 `ingest` 即可。

## 任务反馈采集（--knowledge）

**三态开关**（`--knowledge none|read|update`，`finesub.llm.correction_translation` 与管线 CLI 同名透传，默认 `read`——读取并注入，不采集、不写回；`difficulty=efficiency` 下一样，`update` 也照常可用（owner 2026-10-24、10-30 放开，此前该档拒绝 `update`）；`read` 原名 `collect` 且会采集反馈，owner 2026-10-01 改为只读不采）。开关化重构起（v73/v74）三态同时门控**读取**：`none` 是真正的不用知识库，`read`/`update` 才读——索引注入、词条请求/透传、text 路线预注入全部随读取门控走同一谓词：

- `none`：**不读取、不注入**、不采集、不更新。准备任务与纠错窗请求回合的 prompt 里不出现索引与词条相关内容。
- `read`：读取并注入知识库（索引、词条）；不输出反馈块，不写回（默认）。
- `update`：`read` 的全部 + 纠错各窗口与准备任务的交付回复额外输出 `<task_update_feedback>` v3 JSON 块（`knowledge_hints`（category/entry/可选 sub/direction/focus/reason/source_ids/confidence 1-9；`entry` 永远填主词条，`sub` 为该条目里某一行的行首字段、表示行级更新——指的是一行，不是拆出去的子条目）+ `asr_corrections` + `uncertainties`），harness 分别留存为 `correction_window_task_feedback` / `research_task_feedback` artifact。纠错窗口（含 fast 单窗）的模型输出使用本窗口 `1..N` 局部 `source_ids`，harness 在落盘前映射回稳定源序号；准备任务的多窗口反馈原本就使用稳定源序号。解析失败只告警、不重试；`category` 的枚举拼接误写（如 `streamer|game_lore`，纠错阶段刚输出完 `|` 分隔 CSV 时高发）取 `|` 前段救回而非整条丢弃。切换 `--knowledge` 只影响未完成窗口，已提交窗口不会为补齐反馈而重算。
  任务结束后执行统一知识更新；配合 `--refined-srt` 走精修对照模式。
- 反馈只在 `update` 下采集，例外是参考资料导入：它第一步用 `read` 加内部的采集开关（`CorrectionRequest.collect_feedback`）跑纠错，
  第二步的知识更新读这些反馈。

采集点与落盘 artifact：

| 采集点 | 触发条件 | artifact kind |
| --- | --- | --- |
| 每个纠错窗口 | `--knowledge update`（及参考资料导入） | `correction_window_task_feedback` |
| research 末轮（round 2；fast mode 为 fused round 1） | `--knowledge update`（及参考资料导入） | `research_task_feedback` |

两者都是 `<task_update_feedback>` 标签包裹的 v3 JSON：`knowledge_hints`（每条含 `category`/`entry`/可选 `sub`/`direction`/`focus`/`reason`/`source_ids`/`confidence` 1-9）+ `asr_corrections` + `uncertainties`。落盘 artifact 内的 `source_ids` 一律是稳定源序号，不暴露窗口局部编号；这些 artifact 是统一知识更新阶段读取的原始素材，聚合逻辑见下文「输入材料」。

artifact 里的 `retrieval_urls`（这一轮检索实际用到的来源）由 harness 在落盘时从 agent 传输的事件里回填（`knowledge.feedback.retrieval_urls_from_response`；REST 路径为空，它的接地搜索证据走执行记录那一条），**不解析模型输出**：让模型自己抄 URL 时它会漏掉。

## 统一知识更新（knowledge update）

更新走单一入口 `finesub.llm.knowledge.update`（旧 `task_auto` + `post_task` 双路径已删除），实现在 `finesub/llm/knowledge/update.py`（`run_knowledge_update`）。

**证据模式**（各一段独立 system prompt）：

- `artifacts_only`（无精修）：证据 = 按窗口分组的 raw/final CSV + context + feedback；写入标准从严（宁缺毋滥）。prompt **不含**风格一节，选中的 style 条目也**不注入**——它看到的是机器自己的 raw/final，让它改风格就是模型把自己的习惯学回去。
- `refined_aligned`（`--refined-srt`）：精修行按窗口时间切成 `<refined_csv>`（`start|end|text`，harness 先按 start 重排；index 错乱/注释性重叠字幕因此可容忍），是最高优先级证据；**这一档才把 style 条目钉进 `<kb_entries>`**（至多一套，见上「style 类别」）。精修噪音（非音频注释、拆合行、时间偏移、与 final 不一一对应）由 prompt 明示，不做 harness 侧对齐健康检查，也不再生成/注入 `alignment-report.md`（`src/finesub/subtitles/alignment.py` 保留给人工使用）。

中文素材只纠错的任务不读写知识库（owner 2026-11-06，[`llm_harness_behavior.md`「只纠错」](llm_harness_behavior.md#只纠错)）：任务按 `--knowledge none` 跑；单独跑的知识更新与参考字幕收录遇到记录为只纠错的任务也跳过（`skipped: correct_only`）。中文任务该往库里存什么还没设计（[`llm_followups.md`](llm_followups.md) 只纠错一行）。风格不受影响：写了 `[任务] 中文纠错` 的风格照常注入。

### 精修字幕怎么读

**精修字幕是最高优先级证据，但不是金标。** 它是另一个人在另一套约束下做出的成品，
与机器输出的差异里，只有一部分是「机器错了」。凡是从精修对照里提炼结论的地方
（`refined_aligned` 的 prompt、`refined_alignment_evidence` 回写、
以及任何拿精修当评测参照的实验）都按这七条判读——**前五条管内容，后两条管时间轴**：

1. **断句是精修者的自由。** 同一段语音，机器一行、精修两行，或反过来，都正常；
   拆合本身不是错误信号。**行数差 ≠ 质量差。**
2. **填充词的取舍没有定规。** 语气词、应答词、赞助播报这类非内容行，精修可能保留、
   可能并进相邻行、也可能整条删掉，同一位译者在不同素材上都不一致。
   **「精修里没有」不能推出「机器多译了」**，反过来也不行。
3. **译文带译者偏好。** 意译、口语化重写、换称呼、补主语、加引号都是正常操作；
   **差异不等于机器译错**。只有「歪曲原意/丢失信息」才是错误，「换了个说法」不是。
   连术语一致性都不能假定：同一译者对同一昵称在不同素材里译法可以不同。
4. **精修会插入与语音无关的行。** 译注、梗解释、屏幕文字、他人台词的转写。
   它们常常**压在有语音的时间上**，所以「时间有重叠」不足以判定对应关系；
   要文本上也确实对应才算一对。
5. **精修本身也会错。** 别字、漏译、听错都出现过。它是一份高质量参照，不是判决。
6. **时间不是真值，尤其结束时间。** 精修轴按阅读体验调：**结束时间通常被人为延后**（留读秒），
   开始时间也会提前一点点。此外**压制/重编码会带来整条轴的微小整体漂移**。
   ⇒ 拿它对时间下结论之前，先量一遍整体偏移并说明用的是哪一端；**秒级以下的时间判据不要建立在它上面**。
7. **先问它是不是基于 finesub 修的。** 两种来源差别很大，而**默认是后者**（除非 owner 明说）：
   - **基于我们的输出修的**：精修者只动文本与断句，**轴原样不动**——`0.4.2实操` 那组实测
     1:1 行的起点差中位数 **5 ms**，多数素材 100% 精确命中。按时间 1:1 对齐基本可信。
   - **不是基于我们输出的**：行与行只是恰好落在相近的时间上，**「时间重叠」完全不构成对应关系**
     （与第 4 条同一个陷阱，原因不同），再叠上第 6 条的延后与漂移。
   ⚠ **下面那个 74% 的对齐率来自前一种**，不要搬到后一种上去；后一种的对齐率没量过。

前五条合起来只说一件事：**「精修行数 ≠ 机器行数」和「精修改了 ≠ 机器错了」都不是证据。**
要从一处精修差异得出结论，必须先判定它属于**歪曲原意 / 丢失信息**（错误），还是
**换个说法**（偏好）——只有前者能进错误台账。量级上，**在基于我们输出修的那种里**
（`0.4.2实操`，见第 7 条）能按时间 1:1 对上的机器行约七到八成，而这些对上的行里逐字未改的
只有三成上下：**大多数差异是偏好。**

各对照组的清单、口径与逐项读数在 `data/manually-refined-subs/<系列>/` 各自的
`README.md` / `analysis.md`（本地，含第三方真实素材，不随仓库发布）。

**输入材料**（`src/finesub/llm/knowledge/materials.py`）按 stitch 后实际归属分组，分两层：

- 每个纠错窗口一个窗口包：
  - `<context_slice>`：该窗口的背景调查专属背景（`window_contexts` 里按源序号区间选出的那几条；通用背景在全局块）。
  - `<feedback_slice>`：该窗口纠错调用产出的 `<task_update_feedback>`。
  - `<raw_csv>`：表头行 `# id|start|duration|gap|text`（全局秒；gap 是上一行结束到本行开始，首行留空）——ASR 原始输入，按纠错读的行（三个 pass 里带音频的裁判切开、补上之后的成品层）。裁判听录音补上字的行（补了半句的、补出的整行）文本以 `〔听〕` 开头，这一块有这种行时输入说明多一句：它与成品的差异不当作 ASR 听错。
  - `<final_csv>`：8 列 `type|position|start|end|corrected|translation|conf|note`（不带加权字数与 `gap`：知识更新没有规则读它们，停顿从开始、结束秒看得出）；由 `*-annotated.csv` 按序号 1:1 overlay 后处理 final SRT 的时间与 translation，`corrected` 不 overlay。
  - 可选 `<refined_csv>`：仅 `refined_aligned` 模式，见上文。
- 全局块（跨所有窗口共享一份）：
  - `<general_context>`：背景调查的通用背景，连同准备任务转述的用户要求（`user_requests`）；分块调查时是整片视图：全片名字表在前，各块按块序拼，超 20k 从尾部整块丢。
  - `<research_feedback>`：research 末轮（或 fast round 1）的 `<task_update_feedback>`。
  - `<aggregated_feedback>`：所有窗口 feedback 的聚合摘要，聚合逻辑在 `finesub/llm/knowledge/feedback.py`。
  - `<kb_entries>`：由 feedback hints 频率排序 top 20 预取（research hints ×2 加权，别名归并；每条按类别 `max_entry_tokens` 封顶、整块 ≤60k、钉选的 style 先分、其余最长的先让；前序块已更新的条目标注提示），预取逻辑在 `finesub/llm/knowledge/entries.py`。正文按 prompt 投影渲染（节点带 `@k` 句柄；v79 起行内不再渲染 membership 句柄——没有模型 op 消费它）。
  - 不再注入 `<common_mistakes>` / `<good_examples>`（台账维护与跨任务查重留给独立维护模块）。

**分块与幂等**：三块 CSV 合计超 100k token 时按窗口边界顺序切块（`knowledge` 格绑定组的 catalog 模型可以各自声明 `knowledge_csv_tokens`，组内取最小、没写的按默认计）；组装后整块超 194k 输入硬限再按窗口对半拆（单窗口仍超限则报错）。每块：取当前 `working_rev` 渲染 `<kb_entries>`（prompt 投影，每个节点带 `<!-- @k数字 -->` 句柄，句柄绑定到模型看到的版本）→ 调用 `general_capable` → 校验提案：与落库用同一个解析，`<knowledge_proposals>` 块里有一行不是 JSON 对象（`#` 注释行除外）就算失败，没有这个块时把整段回答当块逐行查；读不了就带上原因重问（`knowledge_update_retry_v1.md`），首次之后的轮数与纠错窗口的修复轮同一个上限（`max_retries_per_window`，默认 3），用完仍读不了就报错；带上坏回答的重问会超过这次规划的输入上限时（回答被输出上限截断、而 prompt 本来就接近上限），这一次改成原样重发首轮→ 先向 `<artifact_dir>/knowledge-update-chunks.jsonl` 写 intent（材料/proposal hash + `rev_before`）→ 提案翻译成节点操作、整批归并成 overlay、校验、**按实体 CAS 一次**、二次校验 → 一个 SQLite 事务 = 一个 `rev`（`revisions.proposal_hash` 是 envelope 哈希，`note` 记原始提案文本哈希）→ 重生成 `rendered/` → 写 applied ledger（含 `rev_after`）→ 下一块以新 `working_rev` 重新渲染词条块（read-your-writes）。stale 的意图按类型处理：标量/整行改写跳过、仅新增的行与别名合并、`remove`/`retire` 拒绝，都记进 apply report 的 `conflicts`。重跑按实际材料 hash（并记录逐窗口 material hash，允许后来改变分块边界）识别已 apply 内容；材料的渲染变了（例如 csv-v113 给 `<raw_csv>` 加了表头行）窗口哈希也跟着变，升级前已写入知识库的任务单独再跑一次知识更新会认不出已写入的窗口、再写一次——整任务重跑时纠错本来就因 `PROMPT_VERSION` 重做，触发面窄，不为此改哈希；prompt、模型、difficulty、任务说明和 token 上限变化不能导致重复写库。⚠ **本次可写的 style 条目算进这两个 hash**——同一素材换一套风格是另一个问题（「这份素材对 B 说明了什么」），不带它就会被判 already_applied、新那套一条提案都收不到；没有可写 style 时这个键整个不写，风格出现之前的 ledger 照旧认得。`task_fingerprint` 只作生成审计，不是幂等门槛；`--no-resume` 才会按用户要求强制重跑。

proposal schema（`MODEL_OPS` 共**十二种** op，v91 起加了 `set_intro`，2026-10-02 加了 `split_entry`；下面列出 prompt 里写明的那些，`remove_item` 不在 prompt 的列表里、只在下面的 apply 说明里，`split_entry` 那一行只在需要时出现（见上「子条目」）；下面写作 `<可写类别>` 的地方，prompt 里渲染的是**本次会话被授权写入的类别**——知识更新轮是可匹配的那些，两个 repair 会话是全部，由 `render_knowledge_ops_contract` 从同一个 `allow_categories` 生成，v85 起不再是模板里手写的一对，每行一条；`@k数字` 是 `<kb_entries>` 渲染里的句柄（档案里 `[别名]` 行的是 `@a数字`），只在本次输出内有效）：

```json
{"op":"append_lines","entry":"@k1 或 源语言key","category":"<可写类别>","section":"目标小节名","content":"一行或多行","reason":"…"}
{"op":"update","id":"@k12","line":"该行的完整新内容","reason":"…"}
{"op":"remove","id":"@k12","reason":"…"}
{"op":"create_entry","category":"<可写类别>","entry":"源语言key","entry_type":"游戏|动画|社区|其他（仅 common）","intro":"一句简介","aliases":["…"],"reason":"为何不并入已有词条（必填）"}
{"op":"retire_entry","entry":"@k1 或 key","merged_into":"@k2 或 key","reason":"…"}
{"op":"rename_entry","entry":"@k1 或 key","new_key":"新源语言key","reason":"…"}
{"op":"set_intro","entry":"@k1 或 key","intro":"新的一句简介","reason":"…"}
{"op":"move_line","id":"@k12","section":"目标小节名","to_entry":"（挪到另一个条目时：H1 句柄或 key）","entry":"（仅当这行同时挂在两个条目下时才需要）","reason":"…"}
{"op":"move_line","entry":"@k1 或 key","from_section":"要整节挪走的小节名","to_entry":"目标条目","section":"（可选）","reason":"…"}
{"op":"recategorize","entry":"@k1 或 key","category":"<可写类别>","to_category":"<可写类别>","reason":"必填"}
{"op":"split_entry","entry":"@k1 或 key","section":"要拆出去的小节名","intro":"子条目的一句简介","reason":"…"}
{"op":"add_item","id":"@k12 或 H1 句柄","field":"misheard|aliases","value":"…","reason":"…"}
```

`add_item` 填 H1 句柄时登记在条目本身：`misheard` 是条目名常被听错的写法（匹配器照样索引条目上的 items，2026-10-05 冷读第二轮 P175，owner 定），`aliases` 与对 `[别名]` 行 `@a` 句柄的 `add_item` 是同一件事。

apply 层（`finesub/llm/knowledge/node/proposals.py` 翻译 → `node/apply.py` 引擎）：`append_lines` 每行按行文法 v3 分类（剥 `[标记]` → 行体形态定 term/note），**标记在该节内唯一**（同名标记改用 `update`）；描述里的「误听: …」与别名列自动登记为检索索引项（`items`）后**从存储中剥离**（别名列渲染自 items、不存 payload；误听散文不留在 desc）；`add_item`/`remove_item`（aliases）落 items 即在渲染列可见，无需回写文本；空行体被拒（稀疏=缺席，plan §11.2）；`update` 只能整行替换且不能改变行体类型；`create_entry` 按 preset 的 `role` 找身份标记建 subject + 一行 `[本名]`（只物化有内容的行，空槽只存在于完整预览）；`retire_entry` 退役 subject 并**级联**关闭其归属边、items 与既有 links（同一 rev；`knowledge restore` 整体带回），另可写 `supersedes` link；`rename_entry` 改 surface（alias/繁简归一命中既有词条时拒绝）；`set_intro`（v91，owner 2026-10-01）整句替换简介，与 `rename_entry` 一样是条目级操作（`require_entry_level`：只在会话可写的类别上生效，知识更新的写入范围是可匹配类别，所以 style 条目的简介不经它改），没变就跳过。**简介的输入约束**（`proposals.intro_problem`，`create_entry` 同样适用）：单行（带换行就拒——`parse_entry_text` 把 H1 与第一个小节之间多于一行当错误，收割会从此失败，索引行也会被截断）、trim 后非空、不以 `#` 开头。H1 仍只走 `rename_entry`，条目别名在引擎里走 `add_item`/`remove_item`。**条目头的 `[别名]` 行**（owner 2026-10-03/04，冷读 Q5）在完整预览的 prompt 投影里带 `@a` 句柄（独立前缀，不占 `@k` 编号；共享冲突修复不给，它的闸门拒条目别名）：`update` 整行改写按集合差分（与人在 `rendered/` 改这一行、与术语行别名列同一套切分——顿号、半角或全角逗号都分隔，名字本身含分隔符时渲染与书写都用「」括起来（`base.join_names` / `_split_names`，index 的本名与别名列同此），`—` 这类没有字词的占位符不算别名——和同一套规范化差分；必须保留 `[别名]` 标记，行体留空即清空），`remove` 清空，`add_item(aliases)` 加一个，其余 op 指向它都拒；差分基准是模型读到的那一行（读取版本上渲染出的别名，不含 tentative），所以改写与清空都碰不到 tentative 别名，也碰不到同一批前面的 op 刚加的。术语行的别名列同样经 `update` 整行改写按差分增删，基准也是渲染出的那一列（不含 tentative；人在 `rendered/` 改术语行同此），与 tentative 别名同名的新增算没变、不发 `add_item`，它仍是 tentative（`importer.alias_row_delta`，四条路共用）；模型改了人认可过的简介，条目头照常出人改版注记。**两个「挪」（2026-09-07）**：`move_line` 把一行挪到另一小节——同一条目内，或带 `to_entry` 挪到另一个条目；给 `entry` + `from_section`（不给 `id`）则把一整节挪到 `to_entry`，挪不了的行（目标节已有同一个 `[标记]` 等）逐行跳过（2026-10-02，引擎的 `move_membership` 本来就能换父条目；目标条目要在本次可写类别里、目标小节要收得下这种行体、且没有同一个 `[标记]` 的行（按本批状态逐条拒，不让整块回滚），已挂在目标条目下的行不能再挪；合并条目时先这样把要留的行挪过去再 `retire_entry`，节点连同历史、证据与人改版注记一起走，不要在目标条目里重抄）（引擎的 `move_membership` 一直都在，只是没进模型词表，于是唯一的搬法是删了重打——那会换掉节点，历史与证据一起断），`recategorize` 把整个条目挪到另一类别，**`reason` 必填**并落进条目元数据、**只在完整预览里渲染**（注入面看不到；§0 第 13 条）。⚠ 目标类别的模版装不下的行**自动泊进目标类别的暂存区**，移动照常完成（`apply._park_if_homeless`，报告里 `parked` 逐行列出；`_check_recategorized` 退居兜底，只有目标 preset 根本没有暂存节时才拒绝，两者都兜住裸 `update` 改 `payload.category` 那条路）——判据是**节名存在**且**那一节收这种行体**：非严格类别（`common`）接受任何节名，所以只查节名等于不查，而各节收的行体不同（固定的「档案」节只收自由句，自由节收术语行与自由句）。⚠ 为什么是泊不是拒：拒绝是一条整批校验的问题，除知识更新外的调用方会因此**整批回滚**，一条挪不动的行会把同批其他改动全带走；知识更新虽然只拒收那个重分类并交修复轮，模型要的移动还是丢了，而暂存区本来就是接这种行的。规矩沿用 `edit.py` 的 `stage`/`reject` 分界：没人在看就泊，作者就在跟前才拒。⚠ 但只问「这次移动有没有把它弄坏」：本来就不合自己 preset 的行（legacy 导入的一堆）原样放行，否则最该修的条目反而挪不动。⚠ 改类别同时**重跑一遍共享判据**：新类别不登记的标记按 `unknown_label_share` fail-closed 降回 `local`（`streamer` 的 `[人设]` 到了 `common` 就是这样），只降不升；⚠ 泊靠对**同批后续提案可见**：`apply.parked_section()` 是唯一判据，`_park_if_homeless` 与翻译层的 `_Context.placement_of` 共用——否则「先 recategorize 再 update 那一行」这个契约自己教的顺序会被按旧小节拒掉。⚠ `move_line` 判**目标节收不收这种行体**（不只是节名在不在），不合就按 op 跳过——归位是泊靠之后的常规动作，而各节收的行体不同；⚠ 它的句柄指的是**行**，一行可以挂在两个条目下、两处渲染出同一个句柄，所以多归属时必须用 `entry` 指明动哪一个，指不明就拒绝、不取第一个；⚠ 可匹配类别**共用一个命名空间**，所以查重必须排除自身，否则每一次可匹配类别之间的搬迁都会被判成「与自己重名」。「新」条目 key 经简繁归一（NFKC+casefold+opencc t2s）命中既有 key/本名/别名时**拒绝新建**（不再隐式重定向）。被触及的 subject 自动刷新 `最近更新日期`；index 由 subject payload 与 `items(aliases)` 派生（别名列不存 payload 副本）。非法 proposal 跳过并记录 apply report。模型照抄整行时带进来的句柄注释（`<!-- @k… -->`、`<!-- @a… -->`）在翻译层剥掉，不会存进描述或别名（2026-10-05）。post-task 更新调用固定走真实 `GENERAL_CAPABLE` 链（5 端点 fallback 链），test_profile 不降级。agent 前端（`kb_propose` 工具）到数据面迁移的 4c 步才启用，届时同一引擎消费 accepted envelope（`node/envelope.py` / `node/draft.py`）。

局部检索匹配：背景调查 Round 1 与快速第 1 轮对用户备注（`extra_info`/note）做 key+alias 的 casefold 子串匹配（`finesub/llm/knowledge/base.py` 的 `match_index_keywords`，别名去重到条目、按出现频次排序，最多 8 条，1 字符词跳过），命中条目全文按预算渲染预注入；text 路线没有背景调查，v17 起预注入条目作为**首窗口的透传 seed**（不再恒注入每窗）。

**小词条也参与匹配了**（2026-08-31）。上面那条只问「这个**条目**被点名了吗」，
而条目名是作品名与主播名——**很少有人把它说出口**。引擎其实早就存在：
`node/matching.py` 的 `ExactIndex` 索引 subject surface **加上** term surface 与 items
（别名、误听，含片假名→平假名折叠），纠错阶段每窗都在扫，但**跑在 shadow 里**——
只记 `matched` 事件，对注入毫无影响。

促成解除 shadow 的是那本账本自己的数（20 个真实任务 / 351 次命中）：

| | |
| --- | --- |
| 命中的是 term / subject | **347 / 4** |
| 小词条命中里「母条目从未被点名」的 | **301（87%）** |
| 母条目也命中、可去重的 | 46（13%） |
| 命中来源 surface / item（别名·误听） | 204 / **147（42%）** |
| 每任务命中数 | 中位 8、最大 59，只涉及 42 个不同节点 |

⚠ **这不说明条目级匹配坏了**——匹配到的 subject 是「原神」「水月りうむ」，
而 term 是「ルミ」「ドラキナ」「シグリット」这些角色名。条目名本来就不是嘴里常说的东西。

`match_terms()` / `render_term_matches()`（同一份 `base.py`）：
**命中一行只注入那一行 + 它的母条目名与节名**（`- [原神 / 空月之歌] ルミ|露米|Lumi|…`），
不再拖整篇进来；母条目已在条目级命中的直接跳过（它的正文里本来就有这些行）。
上限 `KB_TERM_PREINJECT_MAX_HITS = 24`，是照上表的中位 8 / 最大 59 定的，不是拍的。
命中明细进 run report 的 `term_matches`，好让 A/B 读得到注入了什么。

**词条瘦身与音近提示（开发者开关 `FINESUB_KB_SLIM=on`，默认关）**：只作用于 serial 的纠错窗口，对象是窗口透传来的词条
（agent 形态与 batched 形态都是）。先把这些词条整条渲染一遍：不超过 2000 token 就原样注入、不加提示；超过才瘦身并加提示。
瘦身在请求回合之前算好，请求回合与交付看到的是同一块。

- **batched 窗口**：首条 user 消息里的词条块说明它经过精简，并写明要看某个术语被省掉的描述时，把它所属条目的 key 写进 `<requested_entries>`；
  请求回合要了的透传条目在回填里给全文，回填的标签也改说「请求的是全文、其余与首条消息一样是精简过的」。请求回合新要的条目照旧是全文。
- **agent 窗口**：词条块的说明同样写明经过精简，不提读回全文（实验里 agent 从没读回过，不去诱导多一轮）。

- **瘦身**：本窗注入的每个条目留条目头（H1、简介、`档案`）；其余各节的术语行里，本窗 ASR 原样说到本名或别名的留整行，
  被提示的交给提示块，其他只留名字（本名、中文定名、别名，去掉描述）；散文行与元数据节拿掉。
- **提示**：本窗每一行的读音（pyopenjtalk 取假名，去长音、促音、浊点，转罗马音）与本窗条目的术语比对：罗马音相似度 ≥ 95
  （行比名字长时用 `partial_ratio`，短时用整串 `ratio`；名字至少 3 拍、6 个罗马字母），或原样命中已登记的误听写法，而这一行
  没有原样写出该术语的，每个术语一行「行号们｜条目 / 节｜术语行」（表头 `rows|entry|写法|定名|别名|描述`），最多 24 个术语，
  列在本窗 ASR 之前（`fragment_kb_phonetic_hints_v1.md`）。不做噪声标定。
- 读的是本次运行钉住的那个修订，术语表按库与修订缓存；提示行进窗口的词条签名（续跑认得出内容变了）。每窗的读数（闸门是否触发、
  整条与瘦身后的 token、提示到的术语、请求回合要回全文的条目 `whole_on_request`）记在 `correction_window_response` 的 `kb_slim` 里。
- parallel 整个会话共用一块定死的词条块，照旧整条、不加提示。
- 读音依赖在 `kb-phonetic` extra 里（`pip install -e ".[kb-phonetic]"`，pyopenjtalk-plus 连同 sudachi 词典约 330 MB），
  不在 `harness` 与运行时锁里；没装时开关打一条 `Warning:`、整条注入。开关关着时纠错 prompt 逐字节不变。
- 实现在 `llm/kb_phonetic_hints.py`，接线在 `stages/correction/serial.py` 的 `_carried_view` 与 `_window_entries`；离线重放与评估在
  `tools/kb_phonetic_eval/`。
- **为什么默认关、翻默认要看什么**（2026-10-16 的两组 A/B，只量专名有没有写对，对称谓、措辞的影响没量）：
  - 准确率两种形态都没变差：agent 形态（WorkBuddy deepseek-v4.1-flash，两组素材各一遍）与 API batched 形态（免费 Gemini 3.8、3.7、3.6 Flash 各一遍）。
  - 省多少取决于条目描述的分量：描述丰富的库（原神，词条块 4547 token）每窗总输入少约 8%；描述只一句的库（2013 token）不到 1%，
    提示说明与提示块把删描述省下的吃掉了；不到 2000 token 的块不触发闸门，原样注入。
  - 请求回合没被「可以要回全文」引出来（72 个 batched 窗一次也没开）；agent 也没读回过整条。
  - 代价是那份读音依赖：翻默认要把 `kb-phonetic` 并进 `harness` 与运行时锁，已有安装会整个重建运行环境。
  - 还开着的见 [`llm_followups.md`](llm_followups.md)「知识库」一节的词条瘦身那一行。

词条透传链（v17；v18 起 prompt 明示词条 key = 条目 H1 / index 主 key）：research round 2 输出 `<keep_entries>`（≤8，选自其注入词条；持久化于 research-context.json 顶层 `keep_entries`）种子首窗口；每个纠错窗口的纠错轮再输出 `<keep_entries>`（选自本窗实际注入集合，canonical 化、截 8）决定链条续/断。

**以上只适用于窗口能把词条再要回来的档位**（`task_graph.window_requests`：`retrieval=local`，或 2026-09-30 起任一 retrieval 只要知识库可读）。其余档位没有每窗请求，丢掉的词条无从要回，因此准备任务与纠错窗都不再被要求输出 `<keep_entries>`（prompt 无此块），harness 直接透传当前全集，上限用 `KB_WINDOW_TOTAL_ENTRIES`(12)；准备任务也因此拿满 12 条请求额度而非按每窗增量定的 8（`finesub.llm.prompts.r1_request_cap`）。逐档请求通道见 `docs/llm_harness_behavior.md`「准备任务与请求通道」。透传词条全文注入下一窗请求回合的 `<carried_entries>`（勿重复请求）并自动进入其交付轮 entry_details；请求回合新请求上限 8、与透传合计 ≤12（超出裁新请求，预算渲染透传优先完整）。透传 keys+正文 hash 仍记录在逐窗完整 `input_hash` 供审计，cache 另记 `keep_entries`/`injected_entries` 供回放续链；已提交窗口的复用只校验 stable 源对应的 core hash，不因词条正文后来变化而重做。

使用示例：

```powershell
# 主任务内一条龙（采集 + 更新）
python -m finesub.llm.correction_translation out/input/input-stable.json `
  --audio data/input.wav -o out/input/input.srt --execute `
  --knowledge update

# 只读，事后独立更新（位置参数 = 标准 final SRT，其余路径按 stem 派生）；
# read 不采集反馈，事后的更新只以 raw/final CSV 与 context 为证据，会提示没有反馈
python -m finesub.llm.correction_translation ... --execute --knowledge read
python -m finesub.llm.knowledge.update out/input/input.srt --execute

# 有精修 SRT（精修对照模式）
python -m finesub.llm.knowledge.update out/input/input.srt --execute --refined-srt data/manually-refined-subs/精修.srt

# 只看 prompt（不调模型）/ 只生成不写库
python -m finesub.llm.knowledge.update out/input/input.srt --prompt-dir out/ku-prompts
python -m finesub.llm.knowledge.update out/input/input.srt --execute --no-apply
```

**降级行为**：独立 CLI 找不到 feedback artifact 时警告并降级继续（词条块为空、证据只剩 CSV/context）；找不到窗口元数据时全部行落入单一 fallback 窗口。

**路径派生与覆盖**：`--stable-json`/`--annotated-csv`/`--research-context`/`--artifact-dir` 可覆盖派生路径（`reference_ingest` 因 stable 与 final SRT stem 不同而使用）。默认 `*-research-context.json` 在 artifact 目录下。

### style 约定的精修对照标记（v91）

owner 2026-10-01 定、实现在 `knowledge/style_marks.py`。只在带精修字幕（`refined_aligned`）、且钉了一套
可写 style 的知识更新里做：执行者在同一次调用里，对那套 style「约定」节（preset 里正例/反例的
`labels_from` 指向的那一节）的每一行判一种——精修遵守了 `confirmed`、精修推翻了 `refuted`、用不上的
不标；正例、反例的行不标。契约在 `fragment_knowledge_style_marks_v1.md`，只在这一格并进 system 的输出
说明，交付是 `<knowledge_proposals>` 之后的一个 `<style_marks>` 块，每行
`句柄|confirmed 或 refuted|精修行的开始时间|一句理由`（全角竖线也认）。没有精修字幕或没钉 style 时 prompt
里没有这段，冲突修复轮也没有。

harness 逐条核对，核不上的丢弃并计数（`chunk_result["style_marks"]`，也写进 apply report 产物），
从不让整次更新失败：句柄必须解析到本次渲染的、钉住那套 style「约定」节里的行；引用的开始时间必须与
**本次调用看到的那组精修行**（按窗口分组，`split_refined_by_window`）里某一行渲染出的 `start` 串
（`_format_csv_seconds`，0.1 秒）完全相同——不按整份 SRT 核，那会放过从别的组碰巧对上的时间。核过的
记进 `evidence`：`(约定节点, "payload.text", 那一版行文的 hash)`，`evidence_kind="judged"`，`verdict`
用 `confirmed` / `refuted`，引用的时间进 `span`。`span` 在去重键里，所以同一任务对同一精修行的同一种
标记只记一条，不同精修行各记一条。它和确定式回写一样只在 `--execute` 且写库时记，是 fail-soft 的遥测。

## 参考素材导入（reference_ingest）

`python -m finesub.workflows.reference_ingest` 接受一组任务，每个任务是竖线分隔的一行
`srt | media | note | preset | args`，端到端执行。任务来源二选一或并用：`--index <目录>`
读取 `<目录>/index.csv`（每行一个任务，`#` 注释与空行跳过），`--task "<行>"`（可重复）单条传入。

**字段**：

- `srt`（必填）：精修 SRT。批量模式下不含 `.srt` 后缀的裸名解析为 `<index目录>/<名>.srt`，否则按路径；单条模式必须是路径。
- `media`：视频/音频 URL（走 yt-dlp）或本地文件。批量模式裸名（无后缀）在 index 目录里 glob 同名文件，否则按路径/URL；本地媒体跳过下载。
- `note`：注入 research 的 `extra_info`（不能含 `|`）。
- `preset`：reference-ingest 自己的一组设置捆绑（`PRESETS`），留空 = `mm-med`。这些名字为兼容既有 index 保留，但全部使用当前 difficulty 默认值 `intermediate`：`mm-med`（audio/local + `test_profile`，全角色 gemini-3.5-flash-lite，便宜，适合知识/prompt 迭代）、`prod`（audio/local，真实模型）、`text`（text/none）、`text-high`（text/native）、`mm-low`（text/local）、`mm-high`（video/local + `test_profile`）。
- `args`：像 CLI flag 一样解析并**覆盖 preset**（行内优先）：`--media/--retrieval/--difficulty/--fast/--output-scale/--video/--model/--language/--gpu-tier/--test-profile/--no-test-profile`。`media=video` 需要视频：本地视频作 media 直接用；URL media 会下载一份视频到 artifact 目录（默认 `out/reference/<id>/<id>.mp4`，优先 720p、并选最低 fps，因 LLM 只按 detail=low/0.25fps 采样），该 mp4 同时就是 pipeline 的输入（不再预先抽音轨）；或用 `--video` 显式指定；三者皆无（且无 media）则报错。

处理步骤：

1. URL：URL→id 映射缓存于 `data/reference/url-map.json`；下载媒体放在本次 artifact 目录。普通 URL 保留 yt-dlp 给的音频容器 `<id>.<ext>`（不重编码，已存在则跳过）；mm/high URL 下载 `<id>.mp4`，该视频直接作为 pipeline 输入。任何情况下都不在人声分离前做有损转码——分离阶段只在 soundfile 打不开容器时解一份无损 FLAC，跑完即删。本地 media：直接使用，`<id>` 取文件名 stem。媒体守卫：视频的音频流远短于视频流（断流 resume 损坏后 merger 静默截断）会直接报错。
2. 完整 pipeline（人声分离 → VAD+ASR → raw SRT，`stages.run_pipeline` 函数直调，按阶段存在跳过）。
3. LLM 纠错翻译（`run_correction_task`，`knowledge="read"` 加采集反馈：research 或复用已有 `research-context.json` → 纠错窗口（各窗口输出 `<task_update_feedback>`）→ SRT 后处理；`out/reference/<id>/<id>.srt` 已存在则整步跳过，`*-translated.srt` 作为模型直出保留）。
4. **统一知识更新（refined_aligned 模式）**：`run_knowledge_update(refined_srt=精修SRT, stable_json=..., artifact_dir=..., style_names=...)`——精修行按窗口切成 `<refined_csv>` 注入，apply `<knowledge_proposals>`（风格约定与词条更新同一个块）；块级 apply ledger 使重跑免重复写库。不再生成时间对齐报告。

注意：该工具**默认全执行**（下载、GPU pipeline、Gemini 配额、知识库写入），偏离 repo 的 `--execute` 惯例——用户主动发起即视为授权；`--dry-run` 只打印每个任务的解析后计划；`--no-apply` 照常跑完全流程但知识更新只生成不写库（proposals 留在 exchanges 供人工审阅，不写 chunk ledger）。`--model`/`--language`/`--gpu-tier` 是全局默认，可被行内 `args` 覆盖。

**执行模型（三 bin 流水线）**：任务跑在 `src/finesub/scheduler.py` 的通用三 bin 引擎上——下载（×2 并行）→
ASR（×1，单文件独占整个 profile）→ LLM（纠错 + 知识更新为一个不可拆单元）。后面任务的下载/ASR
与前面任务的 LLM 重叠执行。LLM 并发自 2026-08-30 起由 `--max-parallel-tasks` 准入（task-parallelism
默认 2，属用户偏好、不做标定，见 `manual/agent.md`）：同 `group` 的任务严格提交序串行
（批内知识累积保持有序——后一个任务的纠错能用上前一个任务刚提交的词条），组间按 LPT 派发、
实际顺序记录在 batch-status 的 `llm-scheduled` 事件里。**reference_ingest 自 2026-08-30 起不带
`group`**（owner）：它的任务与其它批一样可以重叠，放弃的正是上面那条批内累积顺序——素材通常
互不相关，而写库自 W2 起支持并发写。限流器进程内本就带锁（`ModelRateLimiter` 的 RLock，只有跨进程会盲计）；
知识库进程内并发走 per-root 单写者队列 + CAS，跨进程仍是抢不到写锁就跳过并告警、
提案保留、重跑捡回（见 `batch-scheduler.md`）。**单任务失败被隔离**（记录 stage +
error，跳过其下游阶段，继续其余任务；结束时汇总、exit code 非 0），不再首败全停；失败任务可用
`--task` 单条重跑，靠各级存在跳过廉价续跑。多任务批（或显式 `--batch-id`）在
`out/batch/<batch-id>/batch-status.jsonl` 记录事件流，batch-id 默认 `reference-<时间戳>`。
执行前仍先整批校验精修 SRT 存在与行格式（投喂前失败要趁早，投喂后失败要隔离）。

```powershell
# 批量：目录内 index.csv，行如 `clipA|clipA|备注|prod|--media video`
python -m finesub.workflows.reference_ingest --index data/reference/batch1 --gpu-tier standard
# 单条：
python -m finesub.workflows.reference_ingest --task "out/refined-ep12.srt|https://www.bilibili.com/video/BVxxxx|备注|mm-med|--language en"
```

## 信号与证据（事件表 / evidence / 报告）

2026-08-27 落地（`src/finesub/llm/knowledge/node/signals.py`）。四级口径
`matched → exposed → landed → confirmed/refuted`，全部是**幂等落账的遥测**：dedupe_key 去重
（resume/replay 不重复计数）、fail-soft（永不因此让 run 失败）、且**永不回流进任何 prompt**。

- **`matched`**（shadow 扫描，4a 起）：纠错窗口定稿后按 pinned rev 对 raw 文本全库精确匹配，
  `stages/correction/run.py::_shadow_scan_windows`。
- **`exposed`**：节点内容真的到了模型面前。REST 前端 = 该窗口 prompt 实际携带的条目整包
  （`attempts.py` 提交点逐窗落账）；agent 前端 = `kb_read`/`kb_read_node` 实际返回的节点
  （MCP server 在回复通过容量检查后落账，身份用调用方经 extras 传的
  `kb_signal_task`/`kb_signal_window`——纠错窗（含请求回合）带 run task + chunk id，准备任务带
  task；无上游身份退 `assignment_id:task_id`，绝不落 runtime 的常量 "call"）。窗口 raw 文本
  命中该节点 misheard item 的曝光记 `opportunity=correction`——只有这些进误触发分母；agent 路
  径由 report 侧 join 派生同一口径（同 (node, task, window) 有 misheard matched 即计入）。
  `opportunity=style_injected` 是另一种：任务的风格确定时对那套 style 记一条（`subject_id` = `node_id` = 该条目的
  subject，无 window，每个任务每套只一条），是 style 使用率的唯一来源（见上「style 类别」）。
- **`landed`**：run 收尾时逐窗对照——节点定名（surface/`zh`）出现在 corrected 文本而 raw 里没有。
  措辞是「一致」不是「使用」：不度量因果。
- **claim 证据（`evidence` 表）**：附着在 `(node_id, field_path, value_hash)`，无节点级分数。
  verdict 三值 `confirmed / refuted / unverifiable`（后者是校验任务的终态，防止每次重扫都重试）；
  `evidence_kind` 词表（§11.4 信源三分类）：`refined`＝精修对照回写（`refined_alignment_evidence`，
  `update.py` 在 refined_aligned 模式且 execute+apply 时先于 LLM chunk 运行：raw 命中 misheard、
  `zh` 进了产出、精修保留 → `confirmed`；精修推翻 → `refuted`）；`external`＝维护者亲核的 URL
  （share 批准落账）；`user`＝人的 apply（edit/CLI/rendered 收割，`apply_envelope` 收尾对该
  revision 落下的每个语义字段与 item 自动记 `confirmed`——本地 owner 即本地真相，但共享侧不冒充
  external）；`transcript`＝harness apply 的 revision 级素材出处（同一钩子；hint 级 source_ids
  仍在产物里，细化待做）；`judged`＝模型对照精修对 style 约定的判断（见上「style 约定的精修对照标记」）——
  它是本地模型的意见，不进 claim 的计数与状态、不随共享 bundle 导出（`share/exchange.py` 显式排除），
  只在报告里单列。claim 状态词汇（unverified/corroborated/contested/suspect/
  evidence-stale/unverifiable）由 report **现算不入库**。

**报告**：`python -m finesub.llm.knowledge.report`（只读、不调模型；`--subject`/`--rev`/
`--min-exposures`）——按节点/field 列各级计数与最近印证日期、从未命中的 item、高误触发候选
（§5.2 准入：≥N 个 correction 曝光窗口且从未 landed，建议 `exact_enabled=false`）、按 matcher
的转化率。证据按 `(node, field_path, value_hash)` 聚合：只有与当前字段值相符的行贴在当前值上，
旧值的证据标 `[stale value]` 单列（item 改写不继承旧 claim 的确证/推翻）。另有一节
「style 约定的精修对照」：每套 style 的使用率（纠错侧在任务的风格选择钉住时记的
`exposed`、`opportunity="style_injected"`，按不同 `task_id` 数；一条都没记过时显示「未记」），以及
「约定」节逐行的 `judged` 遵守 / 推翻次数（贴在当前行文上，改写前旧值上的另列）。**使用率与这些次数
只进报告，不驱动任何自动退休或删除**（owner 2026-10-01）：style 的约定「这次用不上」是常态，低使用率
更不能当删除理由；词条的 `exposed` / `landed` 也一样只给人看。

## 维护面全清单

十一种形态、两个 CLI，外加两处没有入口的。散在各节里找不全，所以在这里列一次——**这张表只说
「有哪些、谁触发、写到哪」，每一件的判据与取舍仍在它自己那节**。

⚠ 用户向的那一半（`ingest`/`edit`/`show`/`share pull`/`share conflicts`/`share review`
——用户会主动对自己的知识库做的事）在 [`manual/knowledge.md`](manual/knowledge.md)，
按「动作与后果」写。这里不重复那些，只补它不该承载的：dry-run 档位的机制、产物落点、owner 模块。

| 任务 | 谁触发 | 入口 | 档位 | 落到哪 |
| --- | --- | --- | --- | --- |
| **一次性迁移**（0.4.x markdown → SQLite） | 首次打开旧知识库，自动 | `KnowledgeRepo.open` | 无（parity 闸门不过即中止并删半成品库） | 一个 `import` revision；未知小节原样保留、`degraded_sections` 记账 |
| **同名条目折叠** | 迁移后，随 phase-b | 同下 | 同下 | 落选条目的行进胜出者「待归类」并退休（判据 mtime→长度→路径） |
| **确定式形状转换 Phase B** | 迁移后一次 | `knowledge phase-b` | 默认打印 plan，`--execute` 才动 | 一个 revision + `write_report` 的 plan 报告 |
| **整理候选的 LLM 判定** | phase-b 后 / scan 有候选 | `knowledge repair` | 三档：渲染 → `--execute` → `--apply` | 提案走 `apply_model_proposals`；裁定进候选台账（`candidate_decisions`） |
| **材料吸收** | 用户给一份材料 | `knowledge ingest` | 同上 | 同上（无候选台账） |
| **已有风格叠加出厂默认** | 新代码第一次打开库时记下的风格；用到时告警 | `knowledge adopt-base-style` | 三档：渲染 → `--execute` → `--apply` | 同一条目的新一版；`meta` 的待迁移名单划掉这一套（见「style 类别」的出厂默认风格） |
| **缺证 claim 校验** | 需要外部印证时 | `knowledge verify` | 三档：渲染 → `--execute` → `--apply` | `evidence` 行（confirmed 必须带 URL） |
| **候选人工裁定** | 人读 `pending_human` 队列 | `knowledge candidates` | 直接写 | 候选台账 |
| **认可模型改过的人改行** | 人看到人改版注记、接受现值 | `knowledge ack <条目> [行号]` | 直接写 | `line_acks` 表（见「人改版注记」） |
| **共享库维护者审核** | 队列有投稿 | `share review` | 只读 peek → `--execute` → `--post` | 服务端 verdict（按队列项版本 CAS） |
| **pull 合并冲突** | pull 后有未决字段 | `share conflicts [--repair]` | 列表 → `--repair` → `--execute` → `--apply` | 知识库根 `share-conflicts.jsonl`；`dismissed` 粘、`resolved` 不粘 |

两处**没有独立入口**，都是有意的：

- **提案回喂（B' 与被拒的 op）**——嵌在知识更新流程里自动跑，失败只打 `knowledge-proposal-repair-failed`
  告警、不阻断（主提案已落库）。产物在 task artifact 的 `knowledge_proposal_repair`。
  它没有命令是因为它没有「什么时候该跑」这个问题：apply 报出冲突或拒收的那一刻就是唯一时机。

⚠ **不要**再包一层「知识库维护总 CLI」去转发这两个入口。上面十一件里十件已经在同一个
`python -m finesub.llm.knowledge` 下，另一半在 `share`——两者的分界是「本地库 / 与他人交换」，
是真实的边界，不是历史遗留。转发层只会变成第五份选项面副本。

## 材料吸收（`ingest`）

一份材料（用户的笔记、存下来的网页、别处的术语表）加一句可选交代 → 蒸馏成对**一个**条目的
提案 → 走 `apply_model_proposals` 这条唯一写路径。三档与其他 LLM 入口一致：不带旗标只渲染
prompt，`--execute` 跑会话，`--apply` 才落库。任务文档 `agent-tasks/knowledge-ingest/`。

```powershell
python -m finesub.llm.knowledge ingest --subject 星野灯 --material notes.txt
python -m finesub.llm.knowledge ingest --subject 星野灯 --material - --prompt "重点收自称"
```

它与 `repair` 共用判断标准与写入路径，**区别只在输入**：`repair` 处理扫描器给出的候选，
`ingest` 处理外面来的一段文本（material 模式 2026-09-01 从 `repair` 分出来，两个输入合在一个
命令里让两边都难描述）。⚠ 素材按**不可信外部文本**处理，模板里写明其中像指令的句子不是给
模型的。⚠ **一次一个条目**，由 `--subject` 点名；材料讲的条目不存在就先 `new`——把材料路由到
正确的（或多个）条目是还没定的判断题，不猜。⚠ 不抓 URL，网页自己先存成文件。

**这也是迁移读不懂时的去处**（owner 2026-09-01）：机械导入遇到不合语法的部分**不加特判去
对齐**，而是原样保存 → phase-b 泊进「待归类」→ 由本任务或 `repair` 蒸馏。导入器因此是无损
档案员：未知小节的行原样进来当 `note`、保留原小节名，legacy 投影仍逐字节还原，parity 闸门
保持满强度（实测见本地 `data/index.md`「三、未跟踪的本地素材」里的第三方迁移样本）。

## 共享（knowledge/share/）

共享协议的首个可运行形态（2026-08-27）。现行的使用面与契约以本节和两份部署 README
（`deploy/share-server/README.md`、`deploy/share-worker/README.md`）为准；设计取舍与实施记录在维护者的本地计划里，**不随公开快照发布**：

**服务端有两种形态**，客户端命令与协议完全一样：整台服务端（自托管，`deploy/share-server/`），
或者公网上只放一个**信箱 + 快照 CDN**、审核仍在本机（Cloudflare Worker，`deploy/share-worker/`
——那个 Worker 刻意不理解 bundle 语义，判定全部由本机 `ShareService.push()` 做，所以安全模型
与自托管相同）。信箱形态多一条维护者命令 `share relay`，见下。

⚠ **想把整个服务端托管到 Cloudflare，别用 D1**（2026-09-09 定案时核过）：approve 要在**一个事务里**
做 verdict CAS + bundle apply + 追哈希链，D1 没有交互式事务；它的 Worker API 又是异步的，而
`knowledge/` 里有 149 处同步的 `conn.execute`。对口的是 **Durable Object 的 SQLite**（`sql.exec()` 同步返回），
而且要 Workers Paid——Free 每请求只有 10 ms CPU。今天的信箱形态这两样都不需要：真相库仍是维护者本机的一个文件。

```powershell
# 服务端（本机模拟 / 轻量服务器；TLS 与域名归 Caddy/Tunnel，不在这层；
# systemd/Caddy/litestream 模板与部署步骤见 deploy/share-server/README.md）
# 口令只从 --maintainer-token-file 或 FINESUB_SHARE_MAINTAINER_TOKEN 读，不收命令行参数；
# --signing-key-file 可选（签快照，见 docs/manual/share-server.md「签名密钥」）
python -m finesub.llm.knowledge.share.server --root D:\share-data --port 8787 --maintainer-token-file <文件>

# 信箱形态的维护者循环（deploy/share-worker/README.md）：收投稿 → 本机审核 → 送回裁定与快照。
# --remote 同样可省（走内置信箱）；口令建议用 -file 形式，别进 shell 历史。
# 顺带清理队列存档：裁定超 90 天的项，被拒/过期整行删、已批准只清 bundle 正文
# （--purge-days N，0 关闭；⚠ 已批准的行不能删，digest 的跨贡献者计票要数它们）
python -m finesub.llm.knowledge.share relay `
  --server-root D:\share-data --mailbox-token-file <存放 Worker MAINTAINER_TOKEN 的文件> `
  --signing-key-file <share keygen 写出的私钥>

# 维护者审核（本机跑，key 不上服务器；默认 dry-run 只读 peek + 门槛预检，
# --execute 逐项租一并跑审核会话（耗配额），--post 才把 verdict + 外部印证回写——一次一项
# （要 --queue-id），提交前在终端确认，--yes 免问；未过 §6.3 门槛的批准会被服务端拒绝，
# 除非 --override-thresholds "理由"）。审核会话的回复解析不了（缺 <review_verdict> 块、JSON 坏、结论不在
# approve/approve_tentative/reject 里）只算这一项失败：打出原因、释放租约，其余项照常。待核实的 claim
# 在 prompt 里按短句柄 @v1… 列出，外部印证写句柄 + URL，由 harness 映射回 (node, field_path, value_hash)；
# 句柄对不上的那行丢弃并在终端列出。verdict 的 reason 写裁定依据，有 claim 查了却没核实到时末尾写查过哪些方向
python -m finesub.llm.knowledge.share review --remote … --maintainer-token-file … --queue-id N

# 客户端（--root 缺省为运行时知识库根；--remote 缺省为内置信箱地址，见 share/mailbox.py：
# 优先级是 命令行 > config.toml 的 [share] remote > 内置。内置那份以异或+base64 存放，
# ⚠ 只为躲开扫仓库找 workers.dev 的爬虫，**不是安全措施**——runtime 解得出，谁都解得出）
python -m finesub.llm.knowledge.share register --remote http://127.0.0.1:8787
python -m finesub.llm.knowledge.share mark 原神            # 显式勾选；默认只 subject/term
python -m finesub.llm.knowledge.share mark 原神 --kinds note --match 生日  # 单条说明行精确勾选（按标记匹配）
python -m finesub.llm.knowledge.share unmark 原神          # 撤销勾选（回 local）
python -m finesub.llm.knowledge.share push 原神 --remote …  # 进服务端审核队列
python -m finesub.llm.knowledge.share status --remote … --queue-id N  # 批准后回填 canonical id
python -m finesub.llm.knowledge.share pull --remote …       # 校验链 + 防回滚 + 合并
python -m finesub.llm.knowledge.share conflicts             # pull 没能自动合并的字段（未决的）
python -m finesub.llm.knowledge.share conflicts --repair              # 只渲染判定 prompt（dry-run）
python -m finesub.llm.knowledge.share conflicts --repair --execute --apply  # 跑会话并按常规写路径落库
```

**库的身份是 corpus，不是地址（2026-09-09）**：服务端建库时铸一个 `corpus_id`，它是哈希链的**种子**
（创世条目 = `H(corpus_id, 空库 digest)`），所以身份**可验证**而非自称——两个都从空库起的服务端不再撞
身份（旧的创世是 `H("", …)`，人人相同），服务端也无法在两次 pull 之间改口而不断链。客户端的**全部本地
状态都以它为键**：token、防回滚锚点（`server_rev`/`chain_hash`）、`sync_state` 的三方合并基线，以及
`share-pushes.jsonl` / `share-conflicts.jsonl` 两本台账。于是换地址（workers.dev → VPS、主站 → 镜像）
不再让任何一样失效——⚠ **基线是最贵的那样**：丢了不报错，只是把下一次 pull 变成满屏幻影冲突。
地址仍然只是地址：`--remote <url>` 用法不变，CLI 每次先问一次免认证的 `GET /identity`，并记住
`share:url:<url>` → corpus 的映射。⚠⚠ **corpus id 是公开的**（每份 snapshot 都带着它），所以
**身份不等于授权**：identity 只回答「这是哪个库」，绝不回答「这个地址可以代表它」。后一个问题由
那张映射表回答，而它只为用户**主动指过**的地址而写——闸门只问一句「我是不是正要把一个本地已有
状态的 corpus 交给这个地址」：

- **地址与 corpus 的对应关系没变** → 照常。
- 一个**没用过的地址**报一个本地**没有**任何状态的 corpus → 新关系，TOFU（与第一次 pull 同级）。
- 其余一律**拒绝**，要显式 `--accept-move`（register / push / status / pull 四个命令都有），
  两种情形各有各的理由：本地**已有**该 corpus 的状态时，那是冒用——⚠ 没有这道闸门，一个只是抄了
  公开 corpus id 的镜像就能拿到真服务端发给你的贡献者 token；**用惯的地址改口**报另一个 corpus 时
  没有凭据可泄露，但你并没有要求这次更换，照单全收等于让下一次 pull 把陌生人的库并进本地
  （两条都是 2026-09-09 复审发现）。

⚠ **canonical id 按 corpus 隔离**：它们也是公开的（每份快照里都有），所以**每一处**按 canonical id
找本地行的地方都必须限定在本 remote 内，共四处——节点走本 remote 的 `sync_state` 行；item 限定在
`local_of` 解析出的**属主节点**下；membership 限定在解析出的 **parent** 下；redirect 的改名
`UPDATE` 限定在本 remote 拥有的节点上（那一条最直接：改名就是接管）。曾经这四处都是无归属的全库
查找，任何一个新库首次 pull 就能凭 id 改写、退休**别的库**拉来的条目——不需要冒用身份，TOFU 一次
即可（2026-09-09 两轮复审）。推上去的条目由 `share status` 回填时一并写 `sync_state`，所以不需要
那条全库兜底。⚠ 反过来也要成立：两个库**各自 mint 出同一个 canonical id** 是常态（各发各的），
隔离后它们各自落到各自的本地行，谁也不会吞掉谁——有测试钉着。
**库里挪了的行**（2026-10-02，`sync._apply_membership`）：挪动保留 canonical membership id、换父条目，所以按
「(id, 新 parent)」找不到本地行。拉取找到本地那一行（id 相同、子节点与它现在的父条目**都**在本 remote 的 `local_of` 里——
同样的隔离理由）就跟着挪，挪走再挪回也跟得上。例外三条：本地挪过的行不跟（它现在的父条目不是库里最后给的那个——最近一次拉取
写入的版本的父条目，没有就取第一个版本的；所以只换过节或顺序、被拉取认领的本地行照常跟随），只在库里**也**挪过、挪到别处时
摘要记一句 `local kept: N line(s) currently moved both here and in the library`（三方比较：库里只是还留着旧位置——推送只加不减——
不算分歧）；目标父条目在本地已退役时不挪也不建（合并后退役的条目不会把行收回去）；行活在用户自己的本地条目下时不另建。
推送一侧只加不减，见 `docs/llm_followups.md`「共享推送只加不减」。
⚠ 纯本地的 `share conflicts --remote` 走离线解析（认已知地址，或直接给 corpus id），对从没打过交道的
地址**报错而不是回「没有冲突」**。snapshot schema 因此升到 2，**链已存在却没有 `corpus_id` 的旧服务端
库拒绝启动**——重新 seed 等于重算客户端已经锚定的每一个链接。

要点：**local_id 不上线**（bundle 全部走 handle / canonical id，服务端拒收违例；canonical 锚点
必须在服务端现行 rev 真实存在，新实体 id 一律服务端分配）；**§6.3 门槛是服务端 gate**——payload 按
**语义组**出 claim（`payload:core` 兜底盖住全部会改语义的字段，没有 zh 的 term 照样有 claim、改任何
字段都作废旧印证；term body / subject intro 单独成 `payload:body`/`payload:intro` 走人工槽，subject
身份组走外部印证槽）加每 item 一条 claim，未过门槛批准会被拒，维护者 override 须留因；**贡献者随包
自述的证据摘要一律不作数**（只作审核参考），能过门槛的只有维护者流程给出的、精确匹配入队冻结 claim
的外部出处（关系/经历槽两路都无法核实，批准恒为显式 override）；verdict 的 **merge 映射有结构校验**
（键必须是 bundle 节点、双方 kind 一致、目标互不重复、重映射的 membership 不自环不成环、指纹提示
之外的目标须显式 override）；租出的队列项除指纹提示 `merge_hints`（handle → 表面形相同的既有 canonical id，`_check_merge`
读的就是它）外另带 `merge_candidates`（canonical id → 那个节点的 kind、表面形、中文定名、描述或简介与所属条目），审核 prompt
把内容写在提示旁，让审核者比内容而不是比 id；旧服务端没有这个字段时只列 id（新字段要重新部署服务端才有）；匿名注册与队列有硬配额（每贡献者
pending ≤5、全局 ≤200、30 天过期——过期项由同内容重推自动复活重新入队）；共享文本双向
按不可信输入处理（保留标签剥除、标签形态转义、控制字符删除、长度上限），且服务端在**入队前**
再清洗一遍（绕过 CLI 也过不去）并有 bundle 数量与 HTTP body 上限；**任何内容都要维护者 verdict
才入服务端库**（approve/reject/merge，按队列项版本 CAS，租约防两端并发覆盖）；push 幂等键在发送
**之前**落本地 `share-pushes.jsonl`，响应丢失后的重试复用同一 key、服务端校验同 contributor 同
内容，绝不产生第二个队列项；`status` 只有推送者本人（或 maintainer）可查；pull 先逐项验 snapshot
哈希链（rev 严格递增 + 全链 hash 转移）与防回滚锚点，再对 content 做与 push 同源的**结构准入**
（集合与元素类型、canonical id 语法、kind/field/rel 白名单、live payload 的 shape+key 语法、条目名合不合规——
链只证明服务器说过，不证明内容安全，恶意服务端与恶意贡献者同权），全部通过才写库；**谁发布的**
另由签名回答：维护者本机的 `ShareService` 持 Ed25519 私钥（`share keygen`，`share/signing.py`）签快照头
`(corpus_id, server_rev, chain_hash)`，客户端定信任的次序是：内置信箱的地址只认 `BUILTIN_SIGNING_KEY`；
别的地址认 `[share] signing_key`（它指的是用户自己的库，不作用于内置库），没配就钉首次签名快照的钥匙
（`mailbox.trusted_signing_key`、`sync.check_signature`），钉住之后未签名或
换钥匙的快照一律拒收（`--accept-new-key` 换钉），所以拿到信箱口令或 Worker 的人发不出会被接受的快照；
从没签过名的库照常可拉（自建且未设钥匙）；标量
three-way 合并、冲突本地保留并**落进冲突台账**（下一段）；共享策略按 **(section, label)** 解析，标记可用 `share` 覆盖本节默认，
未登记标记走 `unknown_label_share`（**fail-closed 在 local**——可能是「中之人」这类）；`update`/移动之后
**重算有效策略并只降不升**（改成表外标记、移进 local 节、换成 `share=local` 的标记都会降级；反方向仍需显式 `mark`）；
`unmark` 随时撤销并向下级联。
可共享集合始终保持祖先闭包：`mark` 子节点自动带上父链，local subject 下的 shareable 子节点不会
被推送，服务端也拒收无 subject 路径的孤儿节点。客户端写命令（含 `push` 的幂等 intent 记录）与
知识维护 CLI 共用同一把跨进程写锁。审核 LLM 与 API key 不在服务器上。

**pull 冲突台账（2026-09-01）**：合并冲突不再只是 `pull` 打完就没的一行。每条未决字段
（标量三方冲突、以及从不自动合并的散文字段）作为**记录**写进知识库根的
`share-conflicts.jsonl`——`remote`/`canonical_id`/`local_id`/字段/本地值/远端值/基线值/
`had_base`/看到它的 rev。⚠ **没有 `end` 式的完整字段就没法事后回到现场**：合并器保留本地、
丢弃远端，远端值不落库，所以下次 pull 会**再报一次同一条冲突**（`sync` 模块 docstring 记着
这个再报语义）。台账因此按「**这场分歧本身**」定身份 `(remote, canonical_id, field, 本地值,
远端值)`：重复上报折叠到同一行，任一侧一动就是**新的一条**、不继承旧裁定。追加式、同 id
最后一行生效，与 `share-pushes.jsonl` 同一套写法。

裁定两种，差别正来自「会被永远再报」：`dismissed`（本地是对的，别再问了）**粘住**，
后续 pull 不重开；`resolved`（我改掉了）**不粘**——同一条再出现说明没改成，会带
`reopened_after` 重开。`share conflicts` 列未决项；`--repair` 是三档，与维护 CLI 的 `repair`
/ `share review` 同形：不带旗标只列，`--repair` 渲染 prompt（**dry-run，连 client 都不构造**），
`--repair --execute` 跑会话，加 `--apply` 才经 `apply_model_proposals` 这条唯一写路径落库。
会话看到的是条目**当前**的 prompt 投影（`@k` 句柄）与三种结局
`keep_local`/`updated`/`needs_human`。⚠ **写入面被限制在本轮冲突涉及的节点**（`restrict_to_conflicted_nodes`）：判断本地值对不对要看邻居，所以读得宽；而被判断的远端值来自另一个贡献者、是不可信输入，所以写得窄，越界提案整条丢弃并报警。按**节点**不按字段——`zh` 冲突的正当修法就是重写整行术语行，必然带着该行其它列。
⚠ **简介与 H1 的冲突交给会话**（owner 2026-10-02：「原则上知识库的所有问题都应该能被 AI 自动解决。」）：会话用
`set_intro` 答简介、用 `rename_entry` 答 H1（`ENTRY_FIELD_OPS`），写入面同样收窄——一条 `intro`（`surface`）
冲突只让那个条目的 `set_intro`（`rename_entry`）通过（按 H1 句柄认），条目的别名、其它字段与另一个操作照旧
碰不到。改名照常过查重、简繁归一与命名空间规则；被拒时那条冲突保持未决，拒绝原因记进会话产物里那行
booking 的 `skipped`。改名会改 `rendered/` 的文件名，会话应用后的那次刷新删掉旧文件、写出新文件并更新
manifest。改名的查重排除被改名的条目自己（`resolve_in(..., exclude=)`，与人工改 H1 同一条规则），所以远端把本地的某个别名定为 H1、或两边只差繁简 / 大小写 / 全半角时都能改成；旧 H1 不会自动留作别名（引擎只改 `surface`；完整 op 契约提示模型旧名仍可能出现时用 `add_item` 留作别名，owner 2026-10-02「模型自会判断」；冲突修复轮的写入面不放行条目的 `add_item`，它的 op 小节也就不给这句，那里只能改名）。**`--repair --execute` 开会话前先收割 `rendered/`**（owner 2026-10-02；与纠错开跑前同一个入口 `harvest_rendered_edits_at_run_start`：worktree 闸门、跨进程写锁、逐文件隔离、解析失败的文件原样保留并告警，不让会话失败）：收割按条目整份 diff，没收割的旧文件会把会话刚改好的 H1 等字段改回旧值、下次拉取又报同一条冲突。dry-run（不带 `--execute`）只列出待收割的文件，不写库。`updated` 的结案判据（重读 `payload[field]` 与本地值比）对两者一样成立。**仍不进会话、只能人工的**
（`human_only`）是会话没有 op 能兑现的：本地节点或它所属的条目已退役（没有条目可给会话看）、以及简介与 H1
以外的条目层字段（没有 op；送进去只会得到「模型说改了、op 被跳过、行永远 open」的死循环）。它们都在列表里标出，出口是`--dismiss`/`--resolve`；
这些缺口记在 [`llm_followups.md`](llm_followups.md)「知识库里仍只能人工的环节」。⚠ **`updated` 只有在字段真的动了才结案**——apply 回滚或
提案被引擎跳过时该行保持未决，判据是重读那个字段而不是模型的说法（与 `node/repair.py` 用扫描
复核候选是同一条纪律）。⚠ prompt 用**自己的模板** `share_conflict_repair_v1.md`，不复用知识更新的回喂轮
`knowledge_proposal_repair_v1.md`：后者开口就是「你上一轮的提案已经提交」，而 pull 冲突里两侧
都不是本会话写的，讲错故事换来的是自信的错答案。op 也只给闸门放得过的那几种（2026-10-04，冷读 Q7）：`fragment_share_conflict_ops_v1.md` 的
`update`/`remove`/`add_item` 加本轮有冲突的条目字段说明（带 `set_intro`/`rename_entry` 的行格式），不给整份 op 契约。

**tentative 分发与 digest（§11.5）**：verdict 第三值 `approve_tentative` 让**纯新建 term** 的 bundle
免门槛入库（schema v3 的 `maturity` 列，node 与 item 两层；含 canonical 锚定节点的 payload 变更或
非 term 新节点一律 400——tentative 只收新建，不收变更）。tentative 内容进 snapshot（含在 content
digest 里，随 pull 落到接收端），但**退出一切 model-facing 面**：prompt/human 投影、index、
`kb_search`（`ExactIndex.build` 默认排除）；只有影子扫描（`include_tentative=True`）消费它采
matched/精修证据——分发的唯一效果就是攒佐证，攒够后服务端把 maturity 翻回 normal、下次 pull 跟进。
`POST /digest`（maintainer）一次跑齐：过期清扫、跨贡献者**参考**计票（每 claim 每有被批准记录的
贡献者 1 票；匿名 token 是写入凭证不是证据身份，计票**不满足任何门槛**）、维护者工作单（含
tentative 资格与阻塞原因）、**晋升**与**提名退役**（都按**当前** claim 的 (node, field_path,
value_hash) 判定，node 与 item 各自独立：当前 core/item claim 有 confirmed 证据 → digest 单独一个
revision+chain entry 翻回 normal；90 天无当前佐证 → 提名退役，只提名不删除；旧 hash、无关字段的
历史确认两边都不算数），以及自动 tentative 路——服务端 `--auto-tentative` 门控、**默认关**（关时
审核会话的 approve_tentative 是唯一入口）。tentative 的另两条晋升路在 verdict 里：全门槛 approve
命中同一 canonical 节点即晋升该节点；item 去重命中即晋升该 item。校验任务的 verdict 三值
confirmed/refuted/unverifiable——refuted 同样必须带 URL（「查到反证」≠「查不到」），三者都终结该
hash 的重扫（改值换 hash 后自动重新入列）。

- **结构性方向**（2026-08-22 设计稿，在维护者的本地计划里，**不随公开快照发布**）：带稳定 id 的 node 模型取代「key = 文件名、子词条 = 行首字段」、误听反查/音近匹配进预注入、hit/landed/confirmed 三层事件取代打分、共享库按 node 拉/推/审。设计稿与其二次设计已于 2026-08-28 全部落地，现行行为以本文上方各节为准。

- **两个翻译台账已折进 style 条目（2026-09-02）**：现行为在上「style 类别」与「两个翻译台账（已退役，2026-09-02）」，取舍依据在维护者的本地计划里（不随公开快照发布）。上限不另做标定（owner 2026-09-02）：每节 20 行 × 200 字，整条 6000 token 兜底，随时可改。存量已迁完（27 条范例蒸馏进 `default_style`，27 条错误库判定不迁）。删除权：模型可以改写人改内容，由 human 投影逐行给出人改版来兜（见上「人改版注记」）；`landed` 信号即「style 约定的精修对照标记」（见上同名一节），使用率只进报告，不驱动删除。
