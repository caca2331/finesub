# 知识库

> **2026-08-22 起知识库的真相源是 SQLite**（`<knowledge_root>/knowledge.sqlite`，设计见
> [`knowledge-node-plan.md`](plans/knowledge-node-plan.md)，已完成其 §8 第 3 步「切换真相源」）。本文其余部分
> 仍按「条目 / 小节 / 行」描述知识库，因为模型看到的渲染形态没变；凡提到 markdown 文件的地方，指的是
> `rendered/` 下的**派生缓存**（自 §11.3 起同时是可编辑投影，见「存储形态」）。2026-08-26 起 node 计划 §8 第 4 步也已落地：agent 工具会话有
> `kb_index/search/read/read_node/kb_validate` 五件套（授权 = 调用方显式 `kb_tools` 授予、manifest
> 携带；`kb_index` 是必读门），harness 精确扫描以 shadow 事件落账；`keep_entries` 经 owner 决定
> **保留**（4d 关闭），预注入与透传链照旧是两形态共同的数据面。§8 第 5 步（2026-08-27）落地
> 字段级事件与 claim 证据：`exposed`/`landed` 事件、精修对照的确证/推翻回写、只读报告
> `python -m finesub.llm.knowledge.report`（见下「信号与证据」节）；§8 第 6 步同日落地共享的
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
一次 run 读一个 rev：`run_full_correction` 与模块 CLI 全程持有 `base.pinned_generation_rev`
（research/纠错窗口/查询轮的默认读全部钉在 run 开始时的 rev），任务后知识更新每 chunk 显式传
`working_rev` 覆盖（read-your-writes）。pin 是 **run 作用域**（ContextVar，2026-08-30 起，
任务级并行 W1，总览见 [`llm_harness_behavior.md`](llm_harness_behavior.md)「任务级并行」节）：
同 root 的并发 run 各读各的 rev、
互不等待；run 自己开的线程池经 `run_context.bind_llm_worker` 把 pin 带进 worker，跨进程边界
（MCP server）则在创建时写死（spawn env 带 root、任务 manifest 带 rev），不从线程状态推。
并行纠错在**屏障处重钉**（plan W3，`repin_generation_rev`）：词条集定死的那一点把 pin 移到
当前 rev——快照语义从「一 run 一版」变「一阶段一版」，期间别的 task 提交的词条对纠错阶段可见，
同阶段所有窗口仍读同一版；resume 身份本就不含词条正文，缓存不受影响。
`<knowledge_root>/rendered/{streamer,common}/*.md` 与 `index.md` 是每次写入后重生成的**派生缓存**（store 是真相，冲突时 store 赢），
给人看、给桌面文件视图用。human 投影的条目行带 `- ` 前缀（markdown 渲染态下每条一行，兼容在渲染态
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

### style 类别（2026-09-02）

第三个 category `style`，与 streamer / common 的区别只有两条，其余（版本行、apply 引擎、
三投影、`rendered/` 回写、共享）完全共用：

- **不在匹配面上。** 常量分两组（`node/model.py`）：`MATCHABLE_CATEGORIES` 是能被文本匹配到的
  那些（`resolve()` 默认扫它、index 由它建、关键词预注入读它、`create_entry` 的全局查重按它
  判），`STANDALONE_CATEGORIES` 只能被显式点名，`CATEGORIES` 是两者之和；
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
- **取用**：`--style <名字>[,…]`（`python -m finesub.pipeline` 与
  `python -m finesub.llm.correction_translation` 都有），未给读 `[llm] style`，**再未给用
  `default_style`**（`DEFAULT_STYLE_NAME`，owner 2026-09-02：风格默认开、默认只读）——
  ⚠ **`difficulty=efficiency` 例外**：该档下**未给**的 `--style-mode` 解析成 `none`，与
  `resolve_knowledge_switch` 对 knowledge 的处理同一条规则（最便宜的形态不读任何东西）；
  显式点名仍然有效，这只决定「未给」的含义。
  ⚠ 这个隐式默认**库里没有就当没有**（新装的库本来就没有），而用户**点名**的名字指不到
  仍然报错：默默不用你要的那套，看起来和「这套风格没起作用」一模一样；
  解析在 `knowledge/style.py`：**`resolve_style_selection` 是决定这一切的那个函数**（三级取值、隐式默认、efficiency 例外、以及 `writable` 从 mode 派生都在它里面），`resolve_style_names` 只做「参数 → 配置 → 空」那一层。argparse 一律传 `None`。选中的条目
  以 **prompt 投影**注入纠错 system prompt 里原先放「常见翻译错误对照」的那个槽
  （`style_block`）。名字指不到、指到别的类别、或同名歧义都**报错**而不是静默跳过——
  静默的样子和「这套风格没起作用」一模一样。
- **收录**：`refined_aligned` 的知识更新任务把选中的 style 条目**钉进**每一块的
  `<kb_entries>`（不参与打分排序，它是任务的目标），prompt 的「翻译风格条目」一节讲成类
  做法、标记配对、上限与组合、以及「只动本次注入的那条 entry 的行」。
- **上限**在 preset 的节上（`max_lines` / `max_body_chars`）：apply 引擎在每条写路径上校验
  （模型提案与人改 `rendered/` 走同一处），节说明里渲染给模型看。三节各 **20 行 × 200 字
  （含 `[标记]`）**，owner 2026-09-02 拍的粗值、随时可改，不是标定值。⚠ 注入面付的是条目里
  **实际有的东西**，不是上限：一条真实条目（4 条约定 + 7 个例子）渲染出来约 600 字。
- **整条的兜底**：preset 顶层的 `max_entry_tokens`（style = 6000）。逐节上限只管「一行多长、
  一节多少行」，三节都逼近上限时整条仍会很大（实测约 1 万 token）。⚠ **超了不是拒绝写入**——
  修复超大条目本身就是一次写入，一律拒绝会把它锁死在最差状态；规则是**单调**的：超预算时
  只准改短或删行，不准新增，也不准把某一行改长（`_check_entry_budget`）。同时条目在注入面
  顶部挂一行提示，让下一次更新或 repair 先做压缩。计数用无依赖的启发式上界
  （`HeuristicTokenCounter`）——它跑在 apply 校验里，不能联网也不能起子进程，高估是安全方向。
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
部分预览（`preview="partial"`）三样都不渲染。分档看的是**这一面能不能写**，不是它是哪个
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
`unnamed-term` / `episodic-desc` / `duplicate-term` / `chapter-alias`，带 `@c` 句柄）连同条目 prompt
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
`evidence_kind=external` 行（confirmed 必须带 URL，查不到记 `unverifiable` 终态防重扫）。
`repair`/`verify` 都收 `--llm-model [任务组=]值`（可重复；值是模型组或 route target，对应格
**整体换绑、无回退**——verify 强制 native search，钉单模型要选 native 形态如
`local-claude-native-opus-5`，completion 形态会响亮失败）做**运行时**模型指定——按
preferred_targets 语义装进进程级路由 overlay，仅本进程、不落 config.toml。三个会话
（repair/verify/共享 review/材料吸收）共用判断标准 `fragment_kb_judgment_v1.md`。
**第 0 条是收录标准**（2026-09-01）：知识库只为「让 ASR 听对」（源语言表层、读音、易错写法）
与「让字幕写对」（中文定名、别名、语体/自称）两件事服务，两者都不服务的不收；preset 各节
注释里建议的标记就是它的具体化。⚠ 同一条标准下，**有没有 ASR 文本作参照物决定能写多硬**：
任务后的知识更新手上有那次的 ASR 与精修对照，可以写「误听: xxx」；**材料吸收没有对照物，
因此不写误听**。`--prompt` 只偏向收什么、怎么写，越不过这条标准。
对话式编辑（用户直接让
agent 更新知识库）走同一套 `edit`/`apply` 命令（⚠ 早前这里写的 `.claude/skills/kb-edit`**从未存在**，2026-09-01 更正）。写命令都走既有跨进程写锁。

旧形态（每条目一个 markdown 文件 + 内嵌 git）在第一次打开知识库根时**一次性无损导入**（`KnowledgeRepo.open`：
发现没有 `knowledge.sqlite` 而有 `streamer/`/`common/` 的 `*.md` 就导入，打 `knowledge-migrated` 告警）；
原 markdown 目录与 `.git` 保留为历史档案，不再被读写。index 行里有而条目文件里没有的别名/简介会并入条目
（无损）；只有 index 行没有文件的条目导入为「index-only」subject。导入是确定性的（UUIDv5），重复导入得到
同一套 id。（`translation/` 下那两个 markdown 台账 2026-09-02 退役，见下。）
也不再有 git 历史。

**git 不再是知识库更新的前置条件**：`required_capabilities` 不再为 `--knowledge update` 索要 git，
`finesub doctor` 把 git 列为可选。**写入失败一律降级为 warning，绝不失败整个任务**：字幕是主产物，
知识库是副产物。三条路径都**不推进 chunk ledger**，重跑会完整重做：①另一个进程持有写锁（跳过应用，
提案保留）；②整批提案在 overlay 校验阶段就不合法（整体回滚，`knowledge-apply-rolled-back` 告警）；
③store 提交后、ledger 写入前崩溃——`revisions.note` 记着原始提案文本的哈希，下次运行据此识别为
`recovered_after_commit`。

**知识库根的解析顺序**（`finesub.paths.resolve_knowledge_root`）：显式
`--knowledge-root` → `FINESUB_KNOWLEDGE_ROOT` → **源码 checkout 的 `<repo>/knowledge`** →
发行包自身的 user-data → 本机受管安装的 user-data（`%LOCALAPPDATA%\FineSub\user-data`）。
绝不静默落到 CWD。

- **checkout 优先是默认值**：几乎所有从仓库发起的运行都是开发，而真实 API key 就在
  `<repo>/.env`；把它们导向共享知识库会让开发噪声与真实条目在提交流里交织，而分裂只需事后
  合并一次。`FINESUB_CHECKOUT_DATA=0` 显式退出。**git worktree 解析到主仓**（`.git` 文件里的
  `gitdir:` 上溯三段），且 worktree 内的 auto-apply 默认**跳过并告警**，除非
  `FINESUB_KNOWLEDGE_WRITE=1`。
- 桌面端与 CLI 壳正常都会注入 `FINESUB_KNOWLEDGE_ROOT`；后两档是给「绕过启动器、直接用包内
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

**并发冲突按类型化解，不整包回滚**（plan W2，2026-08-30）：唯一性输家若冲突对方是
**read_rev 之后**才落库的（作者当时不可能看见——真并发竞争），只丢冲突 op 及其依赖闭包：
relabel 回退到 stored 标记（同 op 的其它改动照常提交）、新建同名节点连 items/memberships/links
一起丢、并发移动落到刚被占用的小节只丢那条 move。作者当时可见的冲突（用户编辑面依赖被明确
告知）与批内自撞仍整包回滚。全部 op 都被化解/拒绝时**不再空转一个 revision**（`rev=None`）——
两遍化解都算：进队列前那遍，以及**事务内**对着权威状态重做的那遍（竞争者恰好落在 preview 与
BEGIN IMMEDIATE 之间时只有后者能看见，2026-08-30 复审补上，否则会提交一个零行 revision 并
报成功）。出处记账（`book_revision_evidence`）也在**同一个事务内**完成——它是自动提交连接上的
逐行 INSERT，放在提交之后等于在单写者队列之外逐行抢写锁（并发 apply 下实测 `database is
locked`），且是唯一可能在 revision 已提交之后失败的写。建表/迁移（`init_schema`）同样走
`BEGIN IMMEDIATE`：延迟事务先拿读锁再升级，两个线程同时开库就是互等的升级死锁，SQLite 对它
立刻报 `database is locked` 而不理会 `busy_timeout`。
冲突从 `result.conflicts` 升为用户可见面：task report 的知识小节逐条列出、
`knowledge-apply-conflict` 告警、batch 的 batch-status.jsonl 收 `knowledge-conflict` 事件。

**冲突回喂（B'，2026-08-31）**：真并发丢了提案之后，harness 追加**一轮**修复调用——告诉模型
「你读的是 rev X、现在是 rev Y、你的这几条因为什么被丢」，**逐条附上被丢提案的原文**（原始
JSON，它自带 `content` 与模型当时给的 `reason`，所以证据也在里面），并把相关条目在**当前 rev**
的 prompt 投影连同**该版本重新分配的句柄**一起给它，只让它重判这几条（别人已写等价内容就跳过，输出空块
是正常结果）。回喂轮不再注入本轮材料——看不到证据还要「补充」等于凭空捏造。第二个 envelope
以当前 rev 为 `knowledge_read_rev`、绑新句柄，正是它第一次没通过的 CAS 前提。
**整个环节是尽力而为**：主提案已经落库，回喂失败/被拒/返回空都只打 `knowledge-conflict-repair-failed`
告警，运行照常成功（失败轮的 exchange 也照记——验证没过的那轮恰恰是要回看的那轮）；一次冲突只
回喂一轮，最多带 `MAX_REPAIR_ENTRIES`(6) 个条目，超过说明该重跑整块。
只有真有**外来** revision 才回喂：判据是 rev 里除主 apply 自己的提交外还有没有别人的（光看
`current_rev > read_rev` 不行——主 apply 只要提交就自增 rev，自撞也会被误当并发）。
整包回滚（作者当时可见的占用、批内自撞）不走这条路——那是作者错误，不是竞争。

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

条目结构（行文法 v3，两类共通）：H1 → 一句话描述 → `## 档案` → 若干二级节 → `## 元数据`（永远最后，apply 层自动维护 `最近更新日期`，模型不可修改）。新条目由 `create_entry` 按 preset 骨架建档，只物化**有内容**的行（身份标记由 preset 的 `role` 指出，不是硬编码的 `本名`）。preset 是领域 schema：`src/finesub/llm/knowledge/node/presets/{streamer,common}.toml`，定义固定节、每节的 `body_kinds`、标记登记表（`core` 决定完整预览留不留空槽）、`purpose`/`exclude` 收录纪律与 `share`/`verify` 策略；section 没有 tier——空节与空槽在完整预览里全渲染，注入面全不渲染。

- streamer 六节：`档案`（`[本名]` `[别名]` `[人设]` `[外观]`）、`直播内容`、`频道用语`（术语行，`[自称]`）、`特点`（`[语体]` `[音色]`；旧 `说话风格` 与 `喜好 / 特点` 并入）、`人际关系`（术语行）、`待归类`（暂存区，非空即出信号）。旧 `重要经历` 与 `event` kind 已退休。
- common 分类节由模型自由分组命名（持续更新的游戏建议按大版本开二级节，节内可用 `###` 三级目录细分人名/地名/其他专有名词；预设骨架含 档案+待归类+元数据——`待归类` 2026-09-03 补上，收割手改时行级泊位需要它，否则整份编辑只能被拒）；分类行**固定四列文法**（行文法 v3，见上）`源语言|中文定名|别名|一句话描述`——别名恒在第三列、由 `items(aliases)` 渲染（无别名留空、顿号分隔），特殊读音写进别名（假名写法即参与匹配），描述居末**可含竖线**，恰三列报格式错误；描述内可带「误听: xxx」作**录入语法**（自动转为 misheard item 并从描述中剥离；misheard 只供匹配器消费，不再渲染）。旧五段行仅归档导入期可解析（别名列与读音都归入 items）。streamer 另有 `频道用语` 节收频道自造词（粉丝统称、吉祥物名等），行文法同上。碎知识（只有一两行可说、附属既有主题的专名/梗）作为行进母条目分类节，不建独立文件。
- 尺寸纪律：单条目软上限 ~3.5k token（注入 4k 截断留余量）；分类养肥后拆子条目（如 `原神·角色`），拆分由人工确认。

**preset TOML 是结构的唯一真相源，模版与 prompt 都是它的投影。** 完整预览里每节的
用途/收录纪律注释、以及 prompt fragment `fragment_knowledge_structure_v1.md` 的结构一节，
都由 `render_structure_spec()` 从 preset 生成，不手写——三处并存（preset / 代码常量 /
手写 fragment）曾经真的漂移过（fragment 教三列术语行、解析器要四列）。方向只能是
**preset → 模版**：反过来从模版 markdown 解析结构，等于给注释另造一套迷你语法，回到
「从文本猜结构」。两个特设标记靠 `role` 数据化（`identity` = 承载 native_names、
`aliases` = 承载别名索引），index 派生、`create_entry` 骨架、编辑面同步一律**按 role 查**，
代码里不再匹配字面量 `本名`/`别名`；只剩 `元数据` 与 `最近更新日期` 两个真常量，它们是
harness 自有结构，preset 声明它们会被 loader 拒绝。机制在代码（`classify_line` 的算法），
策略在数据（`body_kinds` 这一节允许什么行体）。

**preset loader 严格校验，不做任何兜底转换**——下列一律报错拒绝加载：重复标记名（含
NFKC + 去空白后同名）、重复 `role`、`core`/`allow_custom_labels`/`staging` 非布尔（不接受
字符串——`bool("false")` 是 `True`，一个笔误就能把标记升成 core）、`share`/`verify`/
`body_kinds` 取值越界、未知键（拼错的键必须炸）、声明禁止的 section、`core = true` 却没写
`note`（空槽没有说明等于没有脚手架）。

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
  整个库是同一课重复十几遍，而现行纠错 prompt 早写着这一课；17/27 来自 prompt v16/v17（当前
  v82）。唯一耐久的残渣是专名与它们的误听形态，那是词条材料——对真 `原神` 条目跑过 dry run
  （14 条提案）但没有 apply，材料留在那个目录里，改主意时用 `ingest` 即可。

## 任务反馈采集（--knowledge）

**三态开关**（`--knowledge none|collect|update`，`finesub.llm.correction_translation` 与管线 CLI 同名透传，默认 `collect`——读取并注入、但不写回；`difficulty=efficiency` 下该默认解析为 `none`）。开关化重构起（v73/v74）三态同时门控**读取**：`none` 是真正的不用知识库，`collect`/`update` 才读——索引注入、词条请求/透传、text 路线预注入全部随读取门控走同一谓词：

- `none`：**不读取、不注入**、不采集、不更新（默认）。研究轮/查询轮/fast r1 的 prompt 里不出现索引与词条相关内容。
- `collect`：读取并注入知识库（索引、词条、featured mistakes）；纠错各窗口与 research 末轮（round 2；fast mode 为 fused round 1）额外输出 `<task_update_feedback>` v3 JSON 块（`knowledge_hints`（category/entry/可选 sub/direction/focus/reason/source_ids/confidence 1-9；`entry` 永远填主词条，`sub` 为母词条内子词条的行首字段、表示行级更新）+ `asr_corrections` + `uncertainties`），harness 分别留存为 `correction_window_task_feedback` / `research_task_feedback` artifact。纠错窗口与 fast round 1 的模型输出使用本窗口 `1..N` 局部 `source_ids`，harness 在落盘前映射回稳定源序号；普通 research round 2 的多窗口反馈原本就使用稳定源序号。解析失败只告警、不重试；`category` 的枚举拼接误写（如 `streamer|game_lore`，纠错阶段刚输出完 `|` 分隔 CSV 时高发）取 `|` 前段救回而非整条丢弃。切换 `--knowledge` 只影响未完成窗口，已提交窗口不会为补齐反馈而重算。
- `update`：`collect` 的全部行为 + 任务结束后执行统一知识更新；配合 `--refined-srt` 走精修对照模式。

采集点与落盘 artifact：

| 采集点 | 触发条件 | artifact kind |
| --- | --- | --- |
| 每个纠错窗口 | `--knowledge collect/update` | `correction_window_task_feedback` |
| research 末轮（round 2；fast mode 为 fused round 1） | `--knowledge collect/update` | `research_task_feedback` |

两者都是 `<task_update_feedback>` 标签包裹的 v3 JSON：`knowledge_hints`（每条含 `category`/`entry`/可选 `sub`/`direction`/`focus`/`reason`/`source_ids`/`confidence` 1-9）+ `asr_corrections` + `uncertainties`。落盘 artifact 内的 `source_ids` 一律是稳定源序号，不暴露窗口局部编号；这些 artifact 是统一知识更新阶段读取的原始素材，聚合逻辑见下文「输入材料」。

## 统一知识更新（knowledge update）

更新走单一入口 `finesub.llm.knowledge.update`（旧 `task_auto` + `post_task` 双路径已删除），实现在 `finesub/llm/knowledge/update.py`（`run_knowledge_update`）。

**证据模式**（各一段独立 system prompt）：

- `artifacts_only`（无精修）：证据 = 按窗口分组的 raw/final CSV + context + feedback；写入标准从严（宁缺毋滥）。prompt **不含**风格一节，选中的 style 条目也**不注入**——它看到的是机器自己的 raw/final，让它改风格就是模型把自己的习惯学回去。
- `refined_aligned`（`--refined-srt`）：精修行按窗口时间切成 `<refined_csv>`（`start|end|text`，harness 先按 start 重排；index 错乱/注释性重叠字幕因此可容忍），是最高优先级证据；**这一档才把 style 条目钉进 `<kb_entries>`**（至多一套，见上「style 类别」）。精修噪音（非音频注释、拆合行、时间偏移、与 final 不一一对应）由 prompt 明示，不做 harness 侧对齐健康检查，也不再生成/注入 `alignment-report.md`（`src/finesub/subtitles/alignment.py` 保留给人工使用）。

### 精修字幕怎么读

**精修字幕是最高优先级证据，但不是金标。** 它是另一个人在另一套约束下做出的成品，
与机器输出的差异里，只有一部分是「机器错了」。凡是从精修对照里提炼结论的地方
（`refined_aligned` 的 prompt、mistake/example 台账、`refined_alignment_evidence` 回写、
以及任何拿精修当评测参照的实验）都按这五条判读：

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

这五条合起来只说一件事：**「精修行数 ≠ 机器行数」和「精修改了 ≠ 机器错了」都不是证据。**
要从一处精修差异得出结论，必须先判定它属于**歪曲原意 / 丢失信息**（错误），还是
**换个说法**（偏好）——只有前者能进错误台账。量级上，实测过的对照组里能按时间 1:1
对上的机器行约七到八成，而这些对上的行里逐字未改的只有三成上下：**大多数差异是偏好。**

各对照组的清单、口径与逐项读数在 `data/manually-refined-subs/<系列>/` 各自的
`README.md` / `analysis.md`（本地，含第三方真实素材，不随仓库发布）。

**输入材料**（`src/finesub/llm/knowledge/materials.py`）按 stitch 后实际归属分组，分两层：

- 每个纠错窗口一个窗口包：
  - `<context_slice>`：该窗口的背景调查 context（`general_context` + `window_contexts` 对应条目）。
  - `<feedback_slice>`：该窗口纠错调用产出的 `<task_update_feedback>`。
  - `<raw_csv>`：`源序号|开始|时长|gap|文本`（全局秒）——ASR 原始输入。
  - `<final_csv>`：10 列 `type|position|start|end|gap|corrected|translation|conf|char_count|note`；由 `*-annotated.csv` 按序号 1:1 overlay 后处理 final SRT 的时间与 translation，`corrected` 不 overlay。
  - 可选 `<refined_csv>`：仅 `refined_aligned` 模式，见上文。
- 全局块（跨所有窗口共享一份）：
  - `<general_context>`：背景调查的全局摘要。
  - `<research_feedback>`：research 末轮（或 fast round 1）的 `<task_update_feedback>`。
  - `<aggregated_feedback>`：所有窗口 feedback 的聚合摘要，聚合逻辑在 `finesub/llm/knowledge/feedback.py`。
  - `<kb_entries>`：由 feedback hints 频率排序 top 20 预取（research hints ×2 加权，别名归并；≤4k token/条、整块 ≤40k；前序块已更新的条目标注提示），预取逻辑在 `finesub/llm/knowledge/entries.py`。正文按 prompt 投影渲染（节点带 `@k` 句柄；v79 起行内不再渲染 membership 句柄——没有模型 op 消费它）。
  - 不再注入 `<common_mistakes>` / `<good_examples>`（台账维护与跨任务查重留给独立维护模块）。

**分块与幂等**：三块 CSV 合计超 100k token 时按窗口边界顺序切块；组装后整块超 194k 输入硬限再按窗口对半拆（单窗口仍超限则报错）。每块：取当前 `working_rev` 渲染 `<kb_entries>`（prompt 投影，每个节点带 `<!-- @k数字 -->` 句柄，句柄绑定到模型看到的版本）→ 调用 `general_capable` → 校验 `<knowledge_proposals>` JSONL 语法（失败则采样重试，默认共 2 次）→ 先向 `<artifact_dir>/knowledge-update-chunks.jsonl` 写 intent（材料/proposal hash + `rev_before`）→ 提案翻译成节点操作、整批归并成 overlay、校验、**按实体 CAS 一次**、二次校验 → 一个 SQLite 事务 = 一个 `rev`（`revisions.proposal_hash` 是 envelope 哈希，`note` 记原始提案文本哈希）→ 重生成 `rendered/` → 写 applied ledger（含 `rev_after`）→ 下一块以新 `working_rev` 重新渲染词条块（read-your-writes）。stale 的意图按类型处理：标量/整行改写跳过、仅新增的行与别名合并、`remove`/`retire` 拒绝，都记进 apply report 的 `conflicts`。重跑按实际材料 hash（并记录逐窗口 material hash，允许后来改变分块边界）识别已 apply 内容；prompt、模型、difficulty、任务说明和 token 上限变化不能导致重复写库。⚠ **本次可写的 style 条目算进这两个 hash**——同一素材换一套风格是另一个问题（「这份素材对 B 说明了什么」），不带它就会被判 already_applied、新那套一条提案都收不到；没有可写 style 时这个键整个不写，风格出现之前的 ledger 照旧认得。`task_fingerprint` 只作生成审计，不是幂等门槛；`--no-resume` 才会按用户要求强制重跑。

proposal schema（v78 起七种 op，每行一条；`@k数字` 是 `<kb_entries>` 渲染里的句柄，只在本次输出内有效）：

```json
{"op":"append_lines","entry":"@k1 或 源语言key","category":"streamer|common","section":"目标小节名","content":"一行或多行","reason":"…"}
{"op":"update","id":"@k12","line":"该行的完整新内容","reason":"…"}
{"op":"remove","id":"@k12","reason":"…"}
{"op":"create_entry","category":"streamer|common","entry":"源语言key","entry_type":"游戏|动画|社区|其他（仅 common）","intro":"一句简介","aliases":["…"],"reason":"为何不并入已有词条（必填）"}
{"op":"retire_entry","entry":"@k1 或 key","merged_into":"@k2 或 key","reason":"…"}
{"op":"rename_entry","entry":"@k1 或 key","new_key":"新源语言key","reason":"…"}
{"op":"add_item","id":"@k12","field":"misheard|aliases","value":"…","reason":"…"}
```

apply 层（`finesub/llm/knowledge/node/proposals.py` 翻译 → `node/apply.py` 引擎）：`append_lines` 每行按行文法 v3 分类（剥 `[标记]` → 行体形态定 term/note），**标记在该节内唯一**（同名标记改用 `update`）；描述里的「误听: …」与别名列自动登记为检索索引项（`items`）后**从存储中剥离**（别名列渲染自 items、不存 payload；误听散文不留在 desc）；`add_item`/`remove_item`（aliases）落 items 即在渲染列可见，无需回写文本；空行体被拒（稀疏=缺席，plan §11.2）；`update` 只能整行替换且不能改变行体类型；`create_entry` 按 preset 的 `role` 找身份标记建 subject + 一行 `[本名]`（只物化有内容的行，空槽只存在于完整预览）；`retire_entry` 退役 subject 并**级联**关闭其归属边、items 与既有 links（同一 rev；`knowledge restore` 整体带回），另可写 `supersedes` link；`rename_entry` 改 surface（alias/繁简归一命中既有词条时拒绝）。「新」条目 key 经简繁归一（NFKC+casefold+opencc t2s）命中既有 key/本名/别名时**拒绝新建**（不再隐式重定向）。被触及的 subject 自动刷新 `最近更新日期`；index 由 subject payload 与 `items(aliases)` 派生（别名列不存 payload 副本）。非法 proposal 跳过并记录 apply report。post-task 更新调用固定走真实 `GENERAL_CAPABLE` 链（5 端点 fallback 链），test_profile 不降级。agent 前端（`kb_propose` 工具）到数据面迁移的 4c 步才启用，届时同一引擎消费 accepted envelope（`node/envelope.py` / `node/draft.py`）。

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

词条透传链（v17；v18 起 prompt 明示词条 key = 条目 H1 / index 主 key）：research round 2 输出 `<keep_entries>`（≤8，选自其注入词条；持久化于 research-context.json 顶层 `keep_entries`）种子首窗口；每个纠错窗口的纠错轮再输出 `<keep_entries>`（选自本窗实际注入集合，canonical 化、截 8）决定链条续/断。

**以上只适用于 `retrieval=local`。** 其余档位没有每窗查询轮，丢掉的词条无从要回，因此 r2 与纠错窗都不再被要求输出 `<keep_entries>`（prompt 无此块），harness 直接透传当前全集，上限用 `KB_WINDOW_TOTAL_ENTRIES`(12)；会话级 r1 也因此拿满 12 条请求额度而非按每窗增量定的 8（`finesub.llm.prompts.r1_request_cap`）。逐档轮结构见 `docs/llm_harness_research.md`「会话级轮结构」。透传词条全文注入下一窗查询轮的 `<carried_entries>`（勿重复请求）并自动进入其纠错轮 entry_details；查询轮新请求上限 8、与透传合计 ≤12（超出裁新请求，预算渲染透传优先完整）。透传 keys+正文 hash 仍记录在逐窗完整 `input_hash` 供审计，cache 另记 `keep_entries`/`injected_entries` 供回放续链；已提交窗口的复用只校验 stable 源对应的 core hash，不因词条正文后来变化而重做。search loop 各轮只读可见 research r1 请求的词条（persistent base，与 loop 自请求去重），无中继权。

使用示例：

```powershell
# 主任务内一条龙（采集 + 更新）
python -m finesub.llm.correction_translation out/input/input-stable.json `
  --audio data/input.wav -o out/input/input.srt --execute `
  --knowledge update

# 只采集反馈，事后独立更新（位置参数 = 标准 final SRT，其余路径按 stem 派生）
python -m finesub.llm.correction_translation ... --execute --knowledge collect
python -m finesub.llm.knowledge.update out/input/input.srt --execute

# 有精修 SRT（精修对照模式，额外维护 mistake 台账）
python -m finesub.llm.knowledge.update out/input/input.srt --execute --refined-srt data/manually-refined-subs/精修.srt

# 只看 prompt（不调模型）/ 只生成不写库
python -m finesub.llm.knowledge.update out/input/input.srt --prompt-dir out/ku-prompts
python -m finesub.llm.knowledge.update out/input/input.srt --execute --no-apply
```

**降级行为**：独立 CLI 找不到 feedback artifact 时警告并降级继续（词条块为空、证据只剩 CSV/context）；找不到窗口元数据时全部行落入单一 fallback 窗口。

**路径派生与覆盖**：`--stable-json`/`--annotated-csv`/`--research-context`/`--artifact-dir` 可覆盖派生路径（`reference_ingest` 因 stable 与 final SRT stem 不同而使用）。默认 `*-research-context.json` 在 artifact 目录下。

## 参考素材导入（reference_ingest）

`python -m finesub.workflows.reference_ingest` 接受一组任务，每个任务是竖线分隔的一行
`srt | media | note | preset | args`，端到端执行。任务来源二选一或并用：`--index <目录>`
读取 `<目录>/index.csv`（每行一个任务，`#` 注释与空行跳过），`--task "<行>"`（可重复）单条传入。

**字段**：

- `srt`（必填）：精修 SRT。批量模式下不含 `.srt` 后缀的裸名解析为 `<index目录>/<名>.srt`，否则按路径；单条模式必须是路径。
- `media`：视频/音频 URL（走 yt-dlp）或本地文件。批量模式裸名（无后缀）在 index 目录里 glob 同名文件，否则按路径/URL；本地媒体跳过下载。
- `note`：注入 research 的 `extra_info`（不能含 `|`）。
- `preset`：reference-ingest 自己的一组设置捆绑（`PRESETS`），留空 = `mm-med`。这些名字为兼容既有 index 保留，但全部使用当前 difficulty 默认值 `quality`：`mm-med`（audio/local + `test_profile`，全角色 gemini-3.5-flash-lite，便宜，适合知识/prompt 迭代）、`prod`（audio/local，真实模型）、`text`（text/none）、`text-high`（text/native）、`mm-low`（text/local）、`mm-high`（video/local + `test_profile`）。
- `args`：像 CLI flag 一样解析并**覆盖 preset**（行内优先）：`--media/--retrieval/--difficulty/--fast/--output-scale/--video/--model/--language/--gpu-tier/--test-profile/--no-test-profile`。`media=video` 需要视频：本地视频作 media 直接用；URL media 会下载一份视频到 artifact 目录（默认 `out/reference/<id>/<id>.mp4`，优先 720p、并选最低 fps，因 LLM 只按 detail=low/0.25fps 采样），该 mp4 同时就是 pipeline 的输入（不再预先抽音轨）；或用 `--video` 显式指定；三者皆无（且无 media）则报错。

处理步骤：

1. URL：URL→id 映射缓存于 `data/reference/url-map.json`；下载媒体放在本次 artifact 目录。普通 URL 保留 yt-dlp 给的音频容器 `<id>.<ext>`（不重编码，已存在则跳过）；mm/high URL 下载 `<id>.mp4`，该视频直接作为 pipeline 输入。任何情况下都不在人声分离前做有损转码——分离阶段只在 soundfile 打不开容器时解一份无损 FLAC，跑完即删。本地 media：直接使用，`<id>` 取文件名 stem。媒体守卫：视频的音频流远短于视频流（断流 resume 损坏后 merger 静默截断）会直接报错。
2. 完整 pipeline（人声分离 → VAD+ASR → raw SRT，`stages.run_pipeline` 函数直调，按阶段存在跳过）。
3. LLM 纠错翻译（`run_full_correction(knowledge="collect")`：research 或复用已有 `research-context.json` → 纠错窗口（各窗口输出 `<task_update_feedback>`）→ SRT 后处理；`out/reference/<id>/<id>.srt` 已存在则整步跳过，`*-translated.srt` 作为模型直出保留）。
4. **统一知识更新（refined_aligned 模式）**：`run_knowledge_update(refined_srt=精修SRT, stable_json=..., artifact_dir=..., style_names=...)`——精修行按窗口切成 `<refined_csv>` 注入，apply `<knowledge_proposals>`（风格约定与词条更新同一个块）；块级 apply ledger 使重跑免重复写库。不再生成时间对齐报告。

注意：该工具**默认全执行**（下载、GPU pipeline、Gemini 配额、知识库写入），偏离 repo 的 `--execute` 惯例——用户主动发起即视为授权；`--dry-run` 只打印每个任务的解析后计划；`--no-apply` 照常跑完全流程但知识更新只生成不写库（proposals 留在 exchanges 供人工审阅，不写 chunk ledger）。`--model`/`--language`/`--gpu-tier` 是全局默认，可被行内 `args` 覆盖。

**执行模型（三 bin 流水线）**：任务跑在 `src/finesub/scheduler.py` 的通用三 bin 引擎上——下载（×2 并行）→
ASR（×1，单文件独占整个 profile）→ LLM（纠错 + 知识更新为一个不可拆单元）。后面任务的下载/ASR
与前面任务的 LLM 重叠执行。LLM 并发自 2026-08-30 起由 `--max-parallel-tasks` 准入（task-parallelism
plan W5；默认 2，属用户偏好、不做标定，见 `manual/agent.md`）：同 `group` 的任务严格提交序串行
（批内知识累积保持有序——后一个任务的纠错能用上前一个任务刚提交的词条），组间按 LPT 派发、
实际顺序记录在 batch-status 的 `llm-scheduled` 事件里。**reference_ingest 自 2026-08-30 起不带
`group`**（owner）：它的任务与其它批一样可以重叠，放弃的正是上面那条批内累积顺序——素材通常
互不相关，而写库自 W2 起支持并发写。限流器进程内本就带锁（`ModelRateLimiter` 的 RLock，只有跨进程会盲计）；
知识库进程内并发走 per-root 单写者队列 + CAS（plan W2），跨进程仍是抢不到写锁就跳过并告警、
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

node 计划 §5 的落地（2026-08-27，`src/finesub/llm/knowledge/node/signals.py`）。四级口径
`matched → exposed → landed → confirmed/refuted`，全部是**幂等落账的遥测**：dedupe_key 去重
（resume/replay 不重复计数）、fail-soft（永不因此让 run 失败）、且**永不回流进任何 prompt**。

- **`matched`**（shadow 扫描，4a 起）：纠错窗口定稿后按 pinned rev 对 raw 文本全库精确匹配，
  `stages/correction/run.py::_shadow_scan_windows`。
- **`exposed`**：节点内容真的到了模型面前。REST 前端 = 该窗口 prompt 实际携带的条目整包
  （`attempts.py` 提交点逐窗落账）；agent 前端 = `kb_read`/`kb_read_node` 实际返回的节点
  （MCP server 在回复通过容量检查后落账，身份用调用方经 extras 传的
  `kb_signal_task`/`kb_signal_window`——纠错窗/查询轮带 run task + chunk id，research/fast 带
  task；无上游身份退 `assignment_id:task_id`，绝不落 runtime 的常量 "call"）。窗口 raw 文本
  命中该节点 misheard item 的曝光记 `opportunity=correction`——只有这些进误触发分母；agent 路
  径由 report 侧 join 派生同一口径（同 (node, task, window) 有 misheard matched 即计入）。
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
  仍在产物里，细化待做）。claim 状态词汇（unverified/corroborated/contested/suspect/
  evidence-stale/unverifiable）由 report **现算不入库**。

**报告**：`python -m finesub.llm.knowledge.report`（只读、不调模型；`--subject`/`--rev`/
`--min-exposures`）——按节点/field 列各级计数与最近印证日期、从未命中的 item、高误触发候选
（§5.2 准入：≥N 个 correction 曝光窗口且从未 landed，建议 `exact_enabled=false`）、按 matcher
的转化率。证据按 `(node, field_path, value_hash)` 聚合：只有与当前字段值相符的行贴在当前值上，
旧值的证据标 `[stale value]` 单列（item 改写不继承旧 claim 的确证/推翻）。

## 维护面全清单

九种形态、两个 CLI，外加两处没有入口的。散在各节里找不全，所以在这里列一次——**这张表只说
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
| **缺证 claim 校验** | 需要外部印证时 | `knowledge verify` | 三档：渲染 → `--execute` → `--apply` | `evidence` 行（confirmed 必须带 URL） |
| **候选人工裁定** | 人读 `pending_human` 队列 | `knowledge candidates` | 直接写 | 候选台账 |
| **共享库维护者审核** | 队列有投稿 | `share review` | 只读 peek → `--execute` → `--post` | 服务端 verdict（按队列项版本 CAS） |
| **pull 合并冲突** | pull 后有未决字段 | `share conflicts [--repair]` | 列表 → `--repair` → `--execute` → `--apply` | 知识库根 `share-conflicts.jsonl`；`dismissed` 粘、`resolved` 不粘 |

两处**没有独立入口**，都是有意的：

- **并发写冲突回喂（B'）**——嵌在知识更新流程里自动跑，失败只打 `knowledge-conflict-repair-failed`
  告警、不阻断（主提案已落库）。产物在 task artifact 的 `knowledge_conflict_repair`。
  它没有命令是因为它没有「什么时候该跑」这个问题：CAS 冲突发生的那一刻就是唯一时机。

⚠ **不要**再包一层「知识库维护总 CLI」去转发这两个入口。上面九件里八件已经在同一个
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
保持满强度（实测见本地 `data/index.md` 的第三方迁移样本一节）。

## 共享（knowledge/share/）

node 计划 §6 的首个可运行形态（2026-08-27）。协议细节与实施记录以
[`knowledge-node-plan.md`](plans/knowledge-node-plan.md) §6 / §8 第 6 步为准；这里只记使用面：

```powershell
# 服务端（本机模拟 / 轻量服务器；TLS 与域名归 Caddy/Tunnel，不在这层；
# systemd/Caddy/litestream 模板与部署步骤见 deploy/share-server/README.md）
python -m finesub.llm.knowledge.share.server --root D:\share-data --port 8787

# 维护者审核（本机跑，key 不上服务器；默认 dry-run 只读 peek + 门槛预检，
# --execute 逐项租一并跑审核会话（耗配额），--post 才把 verdict + 外部印证回写；
# 未过 §6.3 门槛的批准会被服务端拒绝，除非 --override-thresholds "理由"）
python -m finesub.llm.knowledge.share review --remote … --maintainer-token …

# 客户端（--root 缺省为运行时知识库根）
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

要点：**local_id 不上线**（bundle 全部走 handle / canonical id，服务端拒收违例；canonical 锚点
必须在服务端现行 rev 真实存在，新实体 id 一律服务端分配）；**§6.3 门槛是服务端 gate**——payload 按
**语义组**出 claim（`payload:core` 兜底盖住全部会改语义的字段，没有 zh 的 term 照样有 claim、改任何
字段都作废旧印证；term body / subject intro 单独成 `payload:body`/`payload:intro` 走人工槽，subject
身份组走外部印证槽）加每 item 一条 claim，未过门槛批准会被拒，维护者 override 须留因；**贡献者随包
自述的证据摘要一律不作数**（只作审核参考），能过门槛的只有维护者流程给出的、精确匹配入队冻结 claim
的外部出处（关系/经历槽两路都无法核实，批准恒为显式 override）；verdict 的 **merge 映射有结构校验**
（键必须是 bundle 节点、双方 kind 一致、目标互不重复、重映射的 membership 不自环不成环、指纹提示
之外的目标须显式 override）；匿名注册与队列有硬配额（每贡献者
pending ≤5、全局 ≤200、30 天过期——过期项由同内容重推自动复活重新入队）；共享文本双向
按不可信输入处理（保留标签剥除、标签形态转义、控制字符删除、长度上限），且服务端在**入队前**
再清洗一遍（绕过 CLI 也过不去）并有 bundle 数量与 HTTP body 上限；**任何内容都要维护者 verdict
才入服务端库**（approve/reject/merge，按队列项版本 CAS，租约防两端并发覆盖）；push 幂等键在发送
**之前**落本地 `share-pushes.jsonl`，响应丢失后的重试复用同一 key、服务端校验同 contributor 同
内容，绝不产生第二个队列项；`status` 只有推送者本人（或 maintainer）可查；pull 先逐项验 snapshot
哈希链（rev 严格递增 + 全链 hash 转移）与防回滚锚点，再对 content 做与 push 同源的**结构准入**
（集合与元素类型、canonical id 语法、kind/field/rel 白名单、live payload 的 shape+key 语法——
链只证明服务器说过，不证明内容安全，恶意服务端与恶意贡献者同权），全部通过才写库；标量
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
⚠ **两类冲突不进会话，只能人工**（`human_only`）：本地条目已退役（没有条目可判）、以及**条目自身的字段**（`intro`/`surface`——提案 schema 刻意不给模型改条目正文的 op，见
`proposals.py` 的 `use rename_entry / append_lines for subjects`；送进去只会得到「模型说改了、op 被跳过、行永远 open」的死循环）。两者都在列表里标出，出口是`--dismiss`/`--resolve`。⚠ **`updated` 只有在字段真的动了才结案**——apply 回滚或
提案被引擎跳过时该行保持未决，判据是重读那个字段而不是模型的说法（与 `node/repair.py` 用扫描
复核候选是同一条纪律）。⚠ prompt 用**自己的模板** `share_conflict_repair_v1.md`，不复用 B' 的
`knowledge_conflict_repair_v1.md`：后者开口就是「你上一轮的提案被丢弃」，而 pull 冲突里两侧
都不是本会话写的，讲错故事换来的是自信的错答案。

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

- **结构性方向**（2026-08-22 设计稿 [`knowledge-node-plan.md`](plans/knowledge-node-plan.md)）：带稳定 id 的 node 模型取代「key = 文件名、子词条 = 行首字段」、误听反查/音近匹配进预注入、hit/landed/confirmed 三层事件取代打分、共享库按 node 拉/推/审。下列各项若与之重叠，以该设计稿为准。

- **kb_entries 超限条目的 prompt 压缩**（pending feature）：`<kb_entries>` 预取每条 ≤4k token，超限条目被截断注入。v78 起没有整节覆盖的操作，截断不再有数据丢失风险（`update`/`remove` 只作用于模型看得见的句柄）；剩下的是被截掉的尾部对模型不可见。方向仍是让条目不再超限（agent 前端由 `kb_read` 的 `sections` 参数自然规避——`tier` 随行文法 v3 退休；REST 注入的 `by-hit` 裁剪方案已在 node 设计稿 v8 取消）。
- ~~**两个翻译台账折成知识库条目**~~ **已完成（2026-09-02）**：现行为在上「style 类别」与「两个翻译台账（已退役）」，取舍在 [`translation-style-plan.md`](plans/translation-style-plan.md)。上限 N/L 的标定实验 owner 2026-09-02 结案不做（值已拍定：每节 20 行 × 200 字 + 整条 6000 token 兜底，随时可改）；存量迁移也已做完（27 条范例迁入 `default_style`，27 条错误库判定不迁）。剩下的两条未决——style 没有 `landed` 信号、删除权与证据——在那份计划的第 6 节。
- 语义级输出校验（harness 侧）：元话语关键词进字幕文本列、单条字幕时间跨度超阈值、长时段无 sub/insert 覆盖的空档统计告警——现有校验只查格式形状，三个 test run 证明「格式合规 ≠ 内容合规」。
- 用更长、多窗口素材复测 feedback schema 偏差与 `<reasoning>` 块合法化后的行为（2026-07 的三个 test run 每个只有 1 个窗口，样本太小）。
- **子词条拆分的自动化**：专名误听/误译个例已由 v14 分类行的「误听: xxx」标记承接；剩余问题是分类节超限时的子词条拆分（如 `原神·角色`）当前靠模型在 reason 里建议 + 人工执行，是否要 apply 层自动检测超限并提示，观察后再定。
- **PROHIBITED_CONTENT 误杀**：已落地 content filter 阶梯（`src/finesub/llm/content_filter.py`；调查/fast/loop/查询/纠错/知识更新均接入，任务级黑名单 resume 可见）。仍可能被源文本本身误杀——阶梯耗尽后的报错即该情形；积累样本后再看是否要做源文本侧降级。
