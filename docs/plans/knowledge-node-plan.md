# 知识库 node 模型、检索与共享 —— 设计稿

状态：设计稿 v7（2026-08-22；v1→v2 补存储真相/证据粒度/迁移切换，v2→v3 补 pinned snapshot、可落库
schema、检索状态机、曝光分母、三投影 parity、共享证据摘要，v3→v4 补 identity/version 两层主键、CAS 写入
与双 rev、事件 trace、确定性迁移 id、restore 语义、共享子实体身份，v4→v5 按「agent 为主、兼顾免费往返」重排检索前端与提案承载，v5→v6 把 apply 从 MCP submit 移回父 harness、overlay 归并按实体 CAS、数据面按协议顺序分 4a–4d 上线、工具预算/审计/授权矩阵，v6→v7 补 `knowledge_read_rev`、自包含 envelope、durable draft、内容 digest 必读块、统一 apply engine、初版不做 GC，v7→v8（2026-08-26）按过度设计复审收敛：模糊匹配 v1 不实现、harness 扫描降为 shadow 度量、必读块只剩 `kb_index`、取消分页游标与 §4.4 的 REST 注入改造、durable draft 押后为升级路径、§5 重定位为共享前置、人工编辑 CLI 提前；同日 v8.1 字段收敛：aliases 单一真相在 items、preset 由 category 派生、names{}/style/双类型 target 删稿就实、verbatim 冗余挂入第二次迁移清单）。
本文**取代**原 `kb_entry_scoring_plan.md`（已归档到 `docs/archive/`，理由见 §1.2）。**实施进度**：§8 第 1–6 步
已全部落地（shadow 包 → 契约夹具 → 2026-08-22 切换真相源为 SQLite → 2026-08-26 人工编辑面、shadow 扫描、
`kb_*` 五件套与必读门、`kb_validate`；4d 以「保留 `keep_entries`」关闭 → 2026-08-27 第 5 步字段级
事件/evidence/report 与第 6 步共享首个可运行形态），现行行为以 [`knowledge.md`](../knowledge.md) 为准；
实施记录在 §7.1 末尾与 §8 各步条目内。仍开放：§8 第 7 步的可选项、§6 的部署件与审核 LLM 半边
（见第 6 步「未做」）、4b/4c 工具面的三家 production-size canary。
2026-08-27 基于真实库拆分预览的审查与 owner 多轮讨论定稿了**二次设计**（条目结构收敛、编辑面、
置信度、共享 digest）——已于 2026-08-28 **全部实施**（进度注记在 §11 各小节末尾），并对实库
完成二次迁移与 repair；owner 原话锚点在 §11.1，实现与其冲突时以锚点为准。

## 1. 动机

### 1.1 三个问题，一个根因

1. **词条打分没有好信号**：原方案靠事后 LLM 打分，行级身份用 `(section, 行首字段, hash)` 对账，
   对 streamer 的 prose 行任何改写都等于删行重建、历史归零；分数是 EMA 有损值，跨用户不可合并。
2. **检索想不到去加载**：预注入只对用户备注做 key/alias 子串匹配；最能把模型带到正确词条的线索
   ——ASR 误听变体——写在行描述的「误听: xxx」里，**不进索引、不可检索**。
3. **共享**（拉取/推送/审核）需要的交换单位、合并规则、审核依据，在「key = H1 = 文件名、子词条 =
   行首字段」的平面 markdown 上都不存在。

根因相同：**知识库的最小单位没有稳定身份**。本方案把「小词条」升格为带 id 的节点（node），
「大词条」降级为渲染时拼出来的视图。prompt 看到的东西不变。

### 1.2 原打分方案为何归档

原方案（`archive/kb_entry_scoring_plan.md`）的内部逻辑自洽，但：

- 主信号选反：把确定性命中扫描排除在外（其 §12），而那是零 API 成本、不依赖模型自律的信号。
- 0/1/2 量表混了「本次落地」与「全局价值」两个维度，后者单次任务里模型没有依据。
- served 口径绑定注入形态（其文首已承认 pull 形态下要重定义）。
- 单用户下分数只服务人工报告，成本收益不成立；共享场景下跨用户聚合才是价值所在，而 EMA 不可聚合。

仍然成立并被本文继承的：**分数/事件永不回流到模型可见的请求决策**（可见性边界）、「落地痕迹」而非
因果帮助的度量语义、宽容丢弃不拒绝重做的校验态度。

### 1.3 保留的既有判断

- **大块注入更 robust**：模型看到整包相关术语才容易意识到误听。这是关于*注入*的论断，不是关于
  *存储*的——node 存储 + 渲染时按 subject 拼块，模型看到的仍是大块。精确匹配引擎仍是必需的，但角色
  收窄（v8）：它是 `kb_search` 的后端与 REST 路径的既有召回，不再由 harness 主动把匹配结果推进注入或
  必读台账——agent 看到「残表」不必想到「散兵」，把可疑 token 交给 `kb_search` 就能命中 misheard 项；
  重名消歧由模型在上下文里做，比 harness 全局规则可靠（§4.1–4.3）。
- **主形态是 agent，兼顾免费 REST 往返**（owner 2026-08-22 澄清）：立项时面向按往返计费的免费额度，
  现已演化为以本地 agent（Claude Code / Codex / agy，工具协议见 `llm_agent_tool_protocol.md`）为主。
  因此检索引擎的**第一消费者是工具**（`kb_*`，§4.3），REST 路径的预注入是同一引擎的第二个前端；
  所有 harness 侧分级仍零往返，REST 模型侧仍是「预注入 + 查询轮」两跳以内。
- **模板提升主播词条质量**：固定槽位是「收集清单」，比自由生长好。自由度留在存储层，纪律放在
  preset 层（§3）。

### 1.4 现有数据里已经出现的症状（迁移时要面对）

- `common/原神.md`：`ユムカ竜` 在两个 section 各一行，正文与误听信息不同；误听写法不统一
  （「误听: …」与「常见 ASR 误听为……」并存）。
- `common/绝区零.md`：`ロスカリファ` 两行**中文定名与英文名互相冲突**（罗斯凯利法/Roscalifa vs
  罗斯卡里法/Roscaelifer）。`append_lines` 只做节内去重，跨节重复无人管。
- 这说明「按 surface 自动合并」不可行，导入必须无损并把冲突交给人（§7）；也说明 parity 不可能是
  byte 级（§7.2）。

## 2. 数据模型

> **行文法与 preset 已被取代（2026-08-29）**：本节描述的 kind 集合（fact/event/relation）、
> 三/四段术语行、`line_form` 单选与 `tier` 都已退休。现行文法是 `[标记] 行体`，preset 是 v2
> （`body_kinds` / `labels` / `purpose` / `exclude` / `share` / `verify`），见
> [`knowledge.md`](../knowledge.md)（现行为）与 [`llm_design_notes.md`](../llm_design_notes.md)
> 的「知识库行文法 v3 的决策记录」（取舍依据；计划正文在本地
> `docs/archive/kb-line-grammar-plan.md`）。本节其余部分（版本行 store、
> overlay+CAS、items、信号、共享协议）仍然成立。

### 2.1 schema（SQLite，可直接落库）

每种实体分 **identity 表 + version 表** 两层。identity 表只存不变量；version 表采用版本行
`valid_from_rev` / `valid_to_rev`（开区间，NULL = 现行），**主键是 `(id, valid_from_rev)`**。更新 = 关闭旧行
+ 插入新行；删除 = 关闭旧行（tombstone）。任何读取都带 pinned `rev`（§2.5）。会变的属性（含 `visibility`）
一律放 version 表，identity 表不放任何可变字段。

```text
revisions           (rev PK 单调递增, created_at, kind: harness|user|import|revert|restore|pull,
                     task_id, proposal_hash, base_rev, note)

nodes               (local_id PK, kind, created_rev)                       # 不变量
node_versions       (local_id, valid_from_rev, valid_to_rev, payload JSON, canonical_id?,
                     visibility: local|shareable, accepted_rev?,  PRIMARY KEY(local_id, valid_from_rev))

items               (item_id PK, local_id, field: aliases|misheard, created_rev)
item_versions       (item_id, valid_from_rev, valid_to_rev, value, exact_enabled, fuzzy_enabled,
                     requires_subject_context, min_mora, canonical_item_id?, accepted_rev?,
                     PRIMARY KEY(item_id, valid_from_rev))

memberships         (membership_id PK, created_rev)
membership_versions (membership_id, valid_from_rev, valid_to_rev, parent_id, child_id, section, order_key,
                     canonical_membership_id?, accepted_rev?,  PRIMARY KEY(membership_id, valid_from_rev))

links               (link_id PK, created_rev)
link_versions       (link_id, valid_from_rev, valid_to_rev, source_id, rel: see_also|supersedes, target_id,
                     accepted_rev?,  PRIMARY KEY(link_id, valid_from_rev))

redirects           (old_canonical_id PK, new_canonical_id, learned_rev)
sync_state          (remote, canonical_id, local_id, last_server_rev, last_pulled_payload_hash,
                     PRIMARY KEY(remote, canonical_id))                                          # §6.2 的 base

evidence            (evidence_id PK, dedupe_key TEXT NOT NULL UNIQUE, node_id, field_path, value_hash,
                     verdict: confirmed|refuted, evidence_kind, source_ref?, task_id, span, algo_version)
                       # dedupe_key = sha256(canonical JSON of {task_id,node_id,field_path,value_hash,verdict,span,algo_version})
events              (event_id PK, dedupe_key TEXT NOT NULL UNIQUE, trace_id, parent_event_id?,
                     kind: matched|selected|exposed|landed, opportunity: correction|context,
                     task_id, window_id?, subject_id, node_id?, item_id?, matcher?, rev, span?, algo_version)
                       # dedupe_key = sha256(canonical JSON of 全部键列，NULL 写成显式 null)；SQLite UNIQUE 对 NULL 不去重，故不用多列 UNIQUE

migration_aux       (local_id PK, legacy_raw, source_path, source_line)                        # 仅迁移期，切换后可删
```

`accepted_rev` 在四张 version 表上都有，membership/link 同样能进 accepted 基线。

**历史版本保留 / GC**：旧 checkpoint 会按其 pinned rev 读旧版本行，`revert <rev>` 也要旧值，而
`revisions` 只存元数据、没有 inverse payload——所以**初版不实现 GC**，版本行只增不删。将来若需要，先补
`snapshot_pins` 表（显式登记活跃 checkpoint，不靠扫 artifact 目录）、revision delta 或周期性完整快照、
「删除后哪些 rev 不再可 read/revert」的明确规则、dry-run 报告与显式确认，再谈回收。

**kind 与 payload**（discriminated union）：

```text
subject   {surface, reading, category, entry_type?, updated_date, intro: prose,
           section_order[], native_names[]}       # preset 由 category 派生（preset_for_category），不落 payload
term      {surface, zh, reading, desc: 一句话, body?: prose, alias_text(verbatim 过渡)}
fact      {field, value}                          # 档案/直播内容/喜好 的 `字段: 值`
event     {occurred_at: date, description}        # 重要经历
relation  {target: 文本, description}             # 人际关系；指向另一 subject 的图边一律走 links，不做双类型 target
note      {text}                                  # 不符合所在 section 行文法的自由行（现有 streamer 词条大量存在；说话风格的整体描述也在此）
```

`note` 是实施时从真实数据逼出来的：现有 `说话风格`/`喜好`/`直播内容` 里大半是无 `字段:` 的自由句。
它保证导入无损；preset 的 `kinds` 决定每个 section 允不允许它。

- `aliases` / `misheard` 不在 payload 里，只在 `items` / `item_versions`（§2.1 顶部）。**items 是匹配与
  索引的唯一真相**（v8.1）：索引行的别名列从 items 渲染，payload 不存副本；term 5 段行仍显示
  `alias_text`（verbatim 渲染源），与 items **双向同步**——行更新按新旧列 diff 增删 items，
  `add_item`/`remove_item`（aliases）反向改写列文本；diff 与引擎重复判定共用 `_match_normalize`
  归一化身份（NFKC 等价改写不动 items）——直到第二次迁移删除 `alias_text`；misheard 是
  add-only 精度缓存（§4.1），行改写不回收既有项。结构化 `names{zh,en,…}` 与 `subject.style` 不做——前者等共享真需要英文名时加字段，
  后者的整体描述以 note 子节点承载（行级身份，合并粒度比单块 prose 细）。
- **归属只由 `memberships` 表达**，section 与顺序属于边。一个 term 可以有多条 membership（多归属）；
  `move` 操作针对 membership 而非 node（§6.5）。
- `valid`（时效）放进 payload：`{kind: date | version, from, to}`，`to` 为开区间；版本号比较由 subject 的
  preset 定义（游戏按大版本）。
- `field_path` 语法：`payload.zh`、`items/<item_id>`、`payload.body/<claim_id>`（prose 按句切 claim，
  只在需要时才切）。

### 2.2 身份：local_id 与 canonical_id

- `local_id`（ulid）：创建后永不改变，只保证**本地**稳定。
- `canonical_id`：拉取/审核后从共享库获得；本地可为空。
- `redirects`：服务端合并后的重定向表，客户端拉取时跟随。
- `entity_fingerprint`（head 归一化：NFKC + casefold + opencc t2s + 假名归一）：**只用于产生合并候选**，
  不自动决定身份，不落库（派生）。

两个用户独立创建同一实体必然得到不同 local_id，这是正常状态，由 §6.2 的合并流程收敛。

### 2.3 三个投影（渲染）

| 投影 | 用途 | 形态 |
| --- | --- | --- |
| `render_legacy(rev)` | **只在 shadow 迁移期**用于 parity（§7.2） | 尽量还原原 markdown；误听等结构化字段按导入时保留的原文片段回填 |
| `render_human(rev)` | `<knowledge_root>/rendered/` **可编辑投影**（人读/人改面，编辑由 run-start 收割入库；§11.3/O4-O5） | 规范化写法，**不带 handle**；misheard 不渲染（matcher-only） |
| `render_prompt(rev, call)` | 注入 prompt | 在 human 视图上叠加本次调用的短 handle（`@k12`、item `@i3`；membership `@m4` 自 v8 **停渲染**——现行 op 集合不含 membership 操作，渲染它只是逐行噪音 token，op 集合真的包含 `move_membership` 类操作时再恢复），harness 持有 handle → `(id, expected_valid_from_rev)` 映射（§2.5 CAS 用）；模型永远不看到 ulid |

渲染规则（三投影共用）：

| 节点 | 渲染 |
| --- | --- |
| `subject` | H1 + intro + 按 preset/section 顺序的 `## 节`，空槽保留 |
| `term` | **固定四列** `源语言\|中文定名\|别名\|一句话描述`（kb-followups A1）：别名列从 alias items 渲染、恒在（空即留空）；desc 居末可含竖线；misheard items **不渲染**（matcher-only）。旧五段语法仅存于归档导入（`grammar="archive"`） |
| `fact` / `event` / `relation` | `字段: 值` / `日期: 事件` / `对象 \| 描述` |

`N| ` 行号渲染、`edit_lines` 行号快照、`line_editable`、「被截断条目禁用 edit」整套机制在切换时删除。
index 文件也是派生物：每个顶层 subject 一行，格式沿用今天的四字段。

### 2.4 存储：SQLite 为真相，JSONL 为交换格式

- 本地真相：`<knowledge_root>/knowledge.sqlite`（§2.1 全部表）。理由：事务、索引、图关系、崩溃恢复、
  pinned read 都是实打实的需求；既然 git 退场，人类可读文件的好处已不足以抵消自己实现这些的成本。
- markdown（`rendered/`）是派生的**可编辑投影**（`render_human`）：人工可以直接改文件，改动在下次
  带写权限的运行开始时按 manifest 记录的行语法收割入库（§11.3）；结构化编辑仍可走薄 CLI
  （`knowledge …`）。store 是唯一真相，冲突时 store 赢。
- JSONL 只用于导入/导出与共享交换（一行一个 node / item / membership / link / evidence 记录）。
  「JSONL 天然 merge 幂等」不成立——更新一行仍是整行冲突。
- 知识库根的解析顺序、跨进程写锁、worktree 告警等（`knowledge.md`）照旧；写锁改为 SQLite 连接级 +
  既有文件锁双保险。

### 2.5 版本、pinned read 与事务（替代内嵌 git 承担的语义）

内嵌 git 今天承担的不只是历史：apply 后统一提交、commit 后 / ledger 前的崩溃恢复、不可变 snapshot
identity、`main/unverified` 的人工可靠锚点、回滚与审计。停止写 git 之前这些语义必须有新归宿：

| git 语义 | 新归宿 |
| --- | --- |
| 一次 apply 一个 commit | 一次 apply 一个 SQLite 事务，分配新 `rev`；`revisions` 记 task_id、proposal hash、`base_rev`（提案生成时读的 rev） |
| 不可变 snapshot（长任务中途库被别人更新） | **pinned read**：任务启动时取 `generation_rev`，research / correction 全程所有查询带 `valid_from_rev ≤ generation_rev < valid_to_rev`。版本行保证 `read_at(rev)` 无需长期持有读事务，也不受后续 rev 影响 |
| 下一 chunk 要看到上一 chunk 的写入（今天 `update.py` 重载词条块的语义） | **两个 rev**：`generation_rev` 在 research/correction 全程固定；post-task 知识更新用 `working_rev`，每个 chunk 的事务成功后前进到新 rev，下一 chunk 以它为 pinned rev 组装材料。read-your-writes 由此保住 |
| **stale write**（提案基于 rev 10 生成，库已到 rev 12） | handle 映射保存模型实际看到的版本：`@k12 → (local_id, expected_valid_from_rev)`。apply **按实体归并、每实体只对外部基线 CAS 一次**：先把整批 op 应用到事务内的 draft overlay（同一批对 `@k12` 的两次 update 合成一个最终 payload；`create` 得到 `@new1` 类 draft handle 供同批后续 op 引用；validator 对 overlay 校验而非只看库的旧状态），再对每个稳定 id 做一次 `… WHERE id=? AND valid_to_rev IS NULL AND valid_from_rev=expected`，写一个新版本行。item/membership/link 各按自身 identity 同样处理。不符时按实体的最终意图处理：标量/prose 变更 → 记 conflict、跳过（不覆盖）；仅新增 item → 对现行版本重新去重后合并；membership/link → 对现行版本重新校验后处理；`retire`/`remove_*` → **拒绝**。**只读依赖也在事务内查存活**：`add_item` 的 owner、link 两端、membership 的 parent/child 自身没有写 op、不进上面的 CAS——它们必须在提交时**仍然存活**，被并发 retire 的引用连同其依赖 op 一起丢弃记 conflict（只为挂在它下面而建的新节点级联丢弃）；存活但版本前移的引用允许（增量合并语义）。conflict 进 apply report，与今天「非法 proposal 跳过记 report」同一路径 |
| resume 的 core hash | `generation_rev` + 被注入节点/item 的 `(id, valid_from_rev)` 集合 digest |
| commit 后 / ledger 前崩溃恢复 | chunk ledger 记 `rev`；重跑时比对 `revisions.proposal_hash`，与今天的 intent/HEAD 对账等价 |
| `main`（可靠锚点）/ `unverified` | **行级 `accepted_rev`**（四张 version 表都有）：harness 写入的版本行 `accepted_rev = NULL`（draft）；人工核定把当时的现行版本标 `accepted_rev = rev`。一个 accepted 节点被模型改了字段 → 旧版本行仍带 `accepted_rev`（accepted 基线），新行是 draft；`read_accepted(rev)` 取每个 id 在 `≤ rev` 的最新 accepted 行。运行时默认读现行（含 draft，= 今天读 `unverified`） |
| 回滚 | `knowledge revert <rev>` = **产生一个新的补偿事务**（kind=revert，把 `rev` 的改动反向应用为新版本行），`rev` 只增不减 |
| 撤销 retire | 三档：自动流程里 retire 单调、不复活；用户显式 `knowledge restore <id>` → **同一 local_id** 新建恢复版本（kind=restore，保住已有 evidence/events/membership）；节点已在共享服务端 canonical-retire 的 → 才用新 local_id + `supersedes` link，不擅自复活全局实体 |
| 审计 | `evidence` + `events` + `revisions` |

内嵌 git 仓库在切换后保留为历史档案，不再写入。

## 3. preset：纪律放在这一层

> **行文法与 preset 已被取代（2026-08-29）**：本节描述的 kind 集合（fact/event/relation）、
> 三/四段术语行、`line_form` 单选与 `tier` 都已退休。现行文法是 `[标记] 行体`，preset 是 v2
> （`body_kinds` / `labels` / `purpose` / `exclude` / `share` / `verify`），见
> [`knowledge.md`](../knowledge.md)（现行为）与 [`llm_design_notes.md`](../llm_design_notes.md)
> 的「知识库行文法 v3 的决策记录」（取舍依据；计划正文在本地
> `docs/archive/kb-line-grammar-plan.md`）。本节其余部分（版本行 store、
> overlay+CAS、items、信号、共享协议）仍然成立。

数据模型「什么都放得下」；preset 回答「对某一类节点，什么该填、填成什么样」。preset 是渲染器、
validator、审核器共同读取的**领域 schema**，不是 prompt 素材：独立为带版本的
`src/finesub/llm/knowledge/presets/*.toml`（package-data，主 git 跟踪），prompt 只是它的一个消费者。
新增一类 subject（企划/团体）加一个文件。

### 3.1 streamer（强 preset）

```text
sections（固定顺序，禁止新建 section）:
  简介        subject.intro: prose，1 段，必填
  档案        children: fact；固定字段 本名/别名/人设/其他
  直播内容    children: fact
  说话风格    children: note（整体描述，逐行）+ term（口癖，可带 misheard）
  喜好 / 特点 children: fact
  重要经历    children: event；绝对日期、升序（validator 重排）
  人际关系    children: relation；target 可 link 到另一 subject
tiers:
  core     = 简介 / 档案 / 说话风格
  optional = 其余
share_inherit:                       # §6.4
  term / fact(非真实姓名) = inherit
  relation / fact(真实姓名类字段) / prose = local
```

- 主播 subject 的各方面是可枚举的，固定槽位 = 收集清单。
- 同槽对同槽，跨用户合并是槽级的；审核规则也写到槽级（§6.3）。
- prose 只留「简介」一处（说话风格的整体描述是 note 子节点：行级身份、合并粒度更细）——prose 的
  跨用户合并没有好办法，把爆炸半径压到最小。
- `tiers` 服务 agent 的 `kb_read(tier=core|all)`：写入侧对 core 层有体积上限（validator 拒绝超限写入），
  保证 core 永远整块拿得动——避免 subject 养大后从「整块可用」突变成「超限不可用」。

### 3.2 common（弱 preset）

`档案` 固定；其余 section 自由命名、允许新建（持续更新的游戏按大版本开 section）；子节点默认
`term`。body 非空的 `term` 可按需挂轻 preset（游戏角色：所属/元素/武器/口癖）。`share_inherit`：
term = inherit，prose = local。

### 3.3 执行层三处

1. **apply 校验**：streamer 的新建 section 直接拒；payload 不合 kind 的跳过记 report（沿用非法
   proposal 路径）；日期升序由 validator 在事务内重排。
2. **渲染按 preset 固定顺序**，空槽保留给模型看（今天的行为，保住）。
3. **审核/合并按槽位**（§6）。

## 4. 检索：一个引擎、两个前端，harness 侧零往返

### 4.1 索引项与策略

可匹配项 = `subject/term` 的 `surface`、`items(aliases|misheard)` 的每一行。定位（v8）：`misheard`
items 是**已确认修正的精度缓存 + 检索语料 + 词条内容**，不以「记全误听」为目标——召回由模型 +
`kb_index` + `kb_search` 承担，清单缺一条只是少一条缓存，不是漏一个洞。

```text
exact_enabled              默认 true（surface / alias / misheard 全部）
fuzzy_enabled / min_mora   schema 字段保留，v1 不实现任何模糊 matcher（重启条件见 §8 第 7 步）
requires_subject_context   默认 false；只约束 REST 预注入的自动选择——agent 路径没有「自动打开」这个动作
```

### 4.2 状态机（度量口径；harness 匹配只作 shadow）

```text
matched  →  selected  →  exposed  →  landed
```

1. **全库精确匹配照跑，但只落事件**（`matched`，shadow）：raw ASR 文本（假名归一后）对所有
   `exact_enabled` 项走倒排索引。v8 起匹配结果**不自动打开 subject、不进注入、不进必读台账**——
   已确认映射（raw 里只有 `ユプカ竜`、没有「原神」）的命中价值由模型侧 `kb_search` 承接；shadow 事件
   积累「scan 命中、模型没读该 subject、精修证实确实漏改」的对照证据，数据说话之前不恢复强制曝光。
2. **REST 候选集维持现状**：用户备注子串匹配（今天的 `match_index_keywords`）选 subject 整包注入。
3. **模糊音近 v1 不实现**（§4.1）；音近判断由模型在上下文里做。
4. **`selected`**：REST = 备注命中的 subject；agent 路径没有 harness selected——模型自选要读什么。
5. **`exposed`**：REST = 渲染块真正进入 prompt（预算裁剪后仍在）；agent = 工具实际返回的节点。
6. 可选（最后做）：BM25 / 本地 embedding 对 `body`/`intro`——给「主播聊到某事件但没提名字」用；不引入 API。

**事件是平表**（v8）：每级各落一条 `events`（带 matcher、window_id、subject_id、item_id），靠
`dedupe_key` 幂等——**`rev` 进事件身份**：同一窗口对更高 rev 重扫（语料变了）是新事实，同 rev 的
resume/replay 才去重；`trace_id`/`parent_event_id` 列保留在 schema，v1 不维护谱系——matcher 只剩 exact
一种，报告按 `(item_id, task, window)` join 计数即可，不需要链级归因。whole-pack 曝光时块内实际渲染的
节点记 `exposed(opportunity=context)`，misheard 命中对应的曝光记 `opportunity=correction`，只有后者进
误触发分母（§5.2）。

### 4.3 一个引擎、两个前端

§4.1–4.2 是引擎；两种形态各有一个前端，**agent 前端是主**。REST 前端不冻结、也不强制双覆盖：共享契约
（proposal schema、PROMPT_VERSION）改动时测当次主要前端即可，模型对载体差异足够 robust（owner
2026-08-26）；`session_replay` 仍以 REST 会话为夹具基础。

| | agent 前端（主） | REST 前端（兼顾免费往返） |
| --- | --- | --- |
| 进会话的东西 | `kb_index`（顶层 subject 一行一条，**唯一的必读块**）；可选一行来自用户备注匹配的软提示（不进台账、不 gate submit） | 备注匹配选中的 subject 整包预注入（今天的形态，不改造） |
| 进一步检索 | agent 自己调 `kb_search(text, subject?)`（v1 仅 exact）、`kb_read(subject, sections?)`、`kb_read_node(@k)`；重名/歧义由模型在返回的候选列表里消歧 | 查询轮按名索取（今天的形态） |
| `exposed` | 工具实际返回的节点 = exposed | 渲染块进 prompt = exposed |
| 预算 | **硬上限，无分页**：每次返回 ≤ `max_tokens`（复用现有 `retrieval_budget`），超限直接报错并提示缩小范围（`sections`）；响应带 `returned_digest`。库大到硬上限装不下一个条目那天再谈分页 | 今天的注入预算，不改 |

**必读门只覆盖 `kb_index`**（v8；digest 门语义保留，`suggested_refs` 必读块取消——恢复条件见 §4.2 第 1 条）：

```text
required_block {ref: kb_index, knowledge_read_rev, full_content_digest}
```

只有 index 内容的 digest 等于 `full_content_digest` 才标记 pulled；「调用过一次」不放行。`exposed`
只在 pulled 时记，门与分母口径一致。

**审计**：所有动态 `kb_*` 调用记 `(tool, normalized_args, knowledge_read_rev, result_digest)` 进审计包与
replay fixture；**不进** resume identity（§2.5），但没有它就无法解释某次 agent 实际看过哪些知识。

**授权矩阵**（manifest / entitlement 层；**授权是调用方的显式授予**——各 stage 调用时带
`kb_tools: read|propose`，client 折进 manifest metadata，server 逐调用核 manifest；generation pin
只提供缺省 root/rev，**有 pin 不等于有授权**，无授予的任务（如 search judge）既无工具也无必读门。
standalone 知识更新（reference_ingest 等，无 pin）显式带 `knowledge_root + knowledge_identity` 获得
完整绑定）：

| 任务 | `kb_index/search/read/read_node` | `kb_validate`（无副作用预检，§6.5） | web |
| --- | --- | --- | --- |
| research / correction / fast / query | ✓ | ✗ | 按既有策略 |
| knowledge-update | ✓ | ✓ | ✗ |
| review（共享审核） | ✓ | ✗ | ✓ |
| 其他 | ✗ | ✗ | — |

**任何任务都没有 `kb_apply`**：apply 只在父 harness（§6.5）。

**工具读哪个 rev** 由任务 manifest 的 `knowledge_read_rev` 决定，不写死：research / correction / fast /
query / review 为 `generation_rev`；knowledge-update 的第 N 个 chunk 为当时的 `working_rev`，chunk apply
成功后**下一个 agent task** 用新 `working_rev`——每个 chunk 是独立 agent task，同一会话内不改 pinned rev
（否则 handle 与 replay 语义都坏）。pull 序列不改变会话身份——这正是 `llm_agent_tool_protocol.md` §7
押后数据面迁移时列的第一个前置（「静态 digest 调用前可知、pull 序列调用后才知」）：resume identity =
`knowledge_read_rev` + index digest + 任务静态块 digest，实际 pull 顺序不纳入。
本方案即该独立项目的知识库那一半；`keep_entries` 链与透传 state 的删除按该文 §7 规定的顺序放在**最后**
（§8 第 4d 步），且只发生在 **agent 会话形态**——REST 会话没有工具可替代，长期保留注入与 transfer
数据面，两个数据面并存是接受的终局。先有 pull 工具并 dual-read，再删旧路径，绝不留下「旧预注入已删、
新工具未通」的中间版本。
`matched/selected` 看的是文本，`exposed` 按前端各有定义，四级口径在两种形态下可比。

### 4.4 REST 前端的注入（维持现状）

v8 取消原 `whole`/`by-hit` 注入改造：为一条兼顾路径新做 section 级裁剪不值。REST 预注入维持今天的
整包形态与 `RenderedBlock.truncated` 守卫。（§3.1 的 tiers 已随行文法 v3 退休，`kb_read`
只按 `sections` 取。）

## 5. 信号：证据附着在 claim 上

**定位（v8）**：本节是共享（§6）的前置——claim 级证据、value_hash、脱敏摘要的粒度都是为跨用户聚合
与审核设计的，单用户本地不为它买单（这正是 §1.2 归档打分方案的理由）。单用户期只落 §4.2 的平事件表
与 exposed/landed 计数（给 4a 的 shadow 对照用）；本节其余机制（evidence 落账、`refined_aligned` 回写、
report CLI）紧贴第 6 步之前实施（§8 第 5 步）。

### 5.1 四级事件与三层证据

| 层 | 事件 | 能证明 | 不能证明 |
| --- | --- | --- | --- |
| 命中 `matched` | 某索引项在 raw 文本里触发 | 这条项在真实素材里会被触发 | 触发得对不对 |
| 曝光 `exposed` | 节点内容真的到了模型面前（§4.2 第 5 条） | 模型有机会用它 | 模型用没用 |
| 一致 `landed` | corrected 文本出现该节点定名，且 raw 对应位置不是它 | **输出与该节点一致**（发生了一次与之相符的修正） | 模型是否真的用了它、改得对不对 |
| 准确 `confirmed` / `refuted` | 精修 SRT 保留/推翻了这次修正；用户手动确认；独立用户收敛；带出处的外部印证 | 这一个 claim 可信 / 不可信 | 同节点其他 claim |

`landed` 的措辞刻意是「一致」而非「使用」，与 §1.2 继承的「不度量因果」一致。

### 5.2 分母：只有 exposed 才能算

「matched 但未 landed」**不是**误触发：可能命中的是 surface 本来无需修正、subject 被预算丢弃模型没见到、
节点相关但本次输出不涉及该 claim、pull 形态下模型没去读该 subject。因此：

- `false-injection rate` 的分母是 `opportunity=correction` 的 `exposed`，不是 `matched`，也不是
  whole-pack 里顺带曝光的 `context` 事件。
- **高误触发报告的准入**：某 `misheard` item 已记 `exposed`（correction）、raw 中存在它对应的修正
  机会（该 span 在 corrected 里被改成了别的东西或没改）、且长期不 `landed`——三条同时满足才进报告；
  人工置 `exact_enabled=false`。
- `landed rate` 同样按 `exposed` 算；按 matcher 分列。

### 5.3 证据粒度比 node 小

精修保留了「ユプカ竜 → ユムカ竜」只能确认 `items/<misheard item> → surface` 这一个 claim，不能顺带确认
该节点的中文定名、英文名、正文；反过来一次 `refuted` 也不该拖累整个节点。因此：

- `evidence` 附着在 `(node_id, field_path, value_hash)`（§2.1）。
- **不存 node 级 `confidence` 标量**（那会重新引入有损、不可合并的分数）。可信度是查询：某 field 的
  confirmed/refuted 计数、独立来源数、最近印证日期。
- 过时性用「最近印证日期」表达：主播节点一年没被任何任务印证，审核/报告侧标灰。

### 5.4 幂等与边界

- 事件与证据靠 `dedupe_key`（规范化 JSON 哈希，§2.1）去重；evidence 的键含 `value_hash`，events 的键含
  `matcher`；resume/replay 不重复计数；matcher 算法改版用 `algo_version` 隔开。
- **可见性边界**：`events`/`evidence`/派生计数永不进入纠错、查询、research 的 prompt。`matched` 本身是
  客观事实（不是模型评分），允许用它排注入优先级，没有「低分→不被请求→永无翻身」的死循环。
- `refined_aligned` 推翻的修正 = 最强负面证据：同时喂 mistake 台账与该 claim 的 `refuted`。今天该模式
  只产提案不回写，是现成缺口。
- LLM 事后打分只保留给扫描判不了的 prose（可选、二元「一致/不一致」、盲评）。

### 5.5 人工报告

`python -m finesub.llm.knowledge.report`（只读，不调模型）：按节点/field 列 matched/exposed/landed/
confirmed/refuted 计数、最近印证日期、从未命中标记；按 §5.2 准入的高误触发项单独一节；按 matcher 的各级
转化率；`--subject` 过滤。

## 6. 共享库

**服务端形态（owner 2026-08-26）**：终态是一台轻量 Linux 服务器——单 Python 服务（三类端点：
`GET /snapshot` 版本化整包拉取、`POST /push` 收 bundle 进审核队列、维护者专用的队列/verdict 面）+
服务器上一个 SQLite（复用 `node/` 的 schema 加队列/贡献者/redirects 几张表，§6 的 validator 与合并
候选代码原样复用）+ Caddy/systemd/litestream。**LLM 审核不在服务器上跑**：维护者本机拉队列、本地
validator+LLM、回写 verdict——API key 不离开维护者机器，服务器无密钥可泄。身份用服务器自发的匿名
token（不强制 GitHub/邮箱；代价是 `independent_contributor_count` 可刷，但任何内容都要维护者批准
才进库，刷数只能污染排序信号，被滥用再加验证门槛）。**过渡**：开发测试期用本机 + Cloudflare
Tunnel（或 Oracle/GCP 免费档 VM）——与终态同构，迁移 = 拷 SQLite + 同一份 systemd unit；serverless
与 GitHub 推送通道都不用（运行时形态不同 / 强制提交者持有 GitHub 账号）。GitHub 只作公开 snapshot
镜像与备份（拉取匿名）。实现共享协议时的三条加固要求（复审 2026-08-26）：**snapshot 完整性与
防回滚**（snapshot 带签名/哈希链，客户端拒绝 server rev 倒退）、**push 幂等键**（客户端生成、
server 去重，重传不产生第二份队列项）、**审核队列的 lease 与 verdict CAS**（维护者拉取带租约，
verdict 按队列项版本 CAS，防两端并发覆盖）。**代码落点**：客户端半边（交换格式、canonical id 工具、`sync_state` 合并、
推送打包）落 `src/finesub/llm/knowledge/share/`；server 是同一协议的薄壳 `share/server.py`
（**stdlib-only**，三个 JSON 端点 + 队列表，`python -m finesub.llm.knowledge.share.server --root
<仓库外数据目录>` 起本机实例；TLS/域名归 Caddy/Tunnel），测试进默认套件；真需要框架能力时再按
`desktop/`/`cli/` 先例提为顶层 `server/` 根，协议代码不动。

### 6.1 交换单位 = node（及其 item / membership / link / 证据摘要）

- **拉取**：按 subject 拉节点集。本地已有同 `canonical_id` → 按字段策略合并（§6.2）；新 id 直接进；
  `links` 指向本地不存在的 id 保留为悬挂引用；`redirects` 跟随。
- **推送**：用户显式勾选（§6.4）。推送包带节点/item/membership/link 的现行版本，与**claim 级证据摘要**
  （脱敏，不带 task_id、span、原文）：

  ```text
  claim_summary (field_path, value_hash, evidence_kind, source_refs[], confirmed_count, refuted_count,
                 exposed_count, landed_count)
  ```

  服务端再按提交者聚合出 `independent_contributor_count`。只有 URL 列表不够——审核器要知道某来源支持的
  是 `zh` 还是某段 body。
- **审核**：validator（preset 槽级规则、§6.3 门槛、§6.4 内容边界）先跑，不经 LLM；审核 LLM 只回答
  `approve | merge_into:<canonical_id> | reject:<reason>`。队列按「独立贡献者数 × landed 率」排序。
- **质量信号 = 跨用户聚合**：被 5 个用户 exposed 40 次、landed 0 次，比任何 EMA 有说服力。

### 6.2 合并策略（字段级）

| 字段类型 | 策略 |
| --- | --- |
| 标量（zh、reading、desc） | three-way（base = 上次拉取版本）；冲突进审核 |
| 集合（items、memberships、links） | 按 **canonical 子实体 id** 并集 + tombstone（见下）；无冲突概念 |
| prose（intro、style、term.body） | 人工或 LLM 合并，默认保留本地并标记待合并 |

⚠ **「标记待合并」之后是什么，这里当时没写；2026-09-01 补上了**：未决字段落进知识库根的
`share-conflicts.jsonl`（身份是**分歧本身**，`dismissed` 粘、`resolved` 不粘），`share conflicts [--repair]` 消费它。行为见 [`knowledge.md`](../knowledge.md)。
⚠ 一处**表里没料到的收口**：上面写「人工或 LLM 合并」，但 `intro`/`surface` 这类**条目自身
payload** 的冲突只能人工——模型侧 op schema 刻意不给「改已有条目正文」的能力
（`proposals.py`: `use rename_entry / append_lines for subjects`），送进 LLM 会话只会得到
「模型说改了、op 被跳过、行永远 open」的死循环。所以 LLM 那一路只覆盖**行级**冲突。
| `retire` / tombstone | 单调：一旦 canonical-retire 不可复活（复活 = 新节点 + `supersedes` link；本地 restore 见 §2.5） |

three-way 的 base 来自 `sync_state(remote, canonical_id, local_id, last_server_rev, last_pulled_payload_hash)`
（§2.1）——没有它就没有「上次拉取版本」。

**子实体身份**：本地 `item_id` / `membership_id` 不能直接跨用户并集——两个用户独立加同一 alias 会得到两个
local id，A 的 tombstone 也删不掉 B 的那一项。协议补：

- 服务端按 `(canonical_node_id, field, normalized_value)` 给 item 产生合并候选，确认后分配
  `canonical_item_id`；membership 按 `(canonical_parent, canonical_child, section)`；本地 version 表的
  `canonical_*_id` 列在拉取时回填。
- 上传包里 membership/link 的 parent/child/target、claim_summary 的 `items/<id>` 一律转成 canonical id
  或 **bundle-local handle**（服务端一次性映射），**不得传客户端 local_id**。
- tombstone 以 canonical 子实体 id 传播；本地尚无 canonical id 的项不参与跨用户删除。

服务端按 `entity_fingerprint` 提示节点级 `merge_into` 候选，审核确认后旧 `canonical_id` 写入 `redirects`。

### 6.3 审核门槛按槽位

| 槽位/类型 | 通过要求 |
| --- | --- |
| common `term`（定名/别名/误听） | 独立来源 ≥ 2，或一次精修印证，或带出处的外部印证 |
| streamer `档案` | 带出处的外部印证（改本名/人设影响全库） |
| streamer `人际关系` / `重要经历` | 独立来源 ≥ 2 或精修印证 |
| prose（简介/说话风格） | 人工 |

「外部印证」只有在 claim_summary 里有具体 `source_refs` 支持该 `field_path` 后才算证据；审核 LLM 的结论
本身不是独立来源。审核会话按 agent 形态跑（带原生 web 工具），`source_refs` 由它逐 claim 产出并落
`evidence(evidence_kind=external)`；REST 路径的审核只做 validator，不做外部印证。

### 6.4 推送范围与内容边界

主播信息按 owner 判定视为公开（知识库本身是公网总结）；但节点内容可能来自直播口述、用户备注或模型推断，
这些不是公开出处。因此：

- `visibility` 默认 `local`；推送是用户对 subject 的显式勾选。勾选后**按 preset 的 `share_inherit`
  继承**（§3.1/§3.2）：term 定名/误听类新节点继承 `shareable`；relation、真实姓名类 fact、prose 新建时
  仍为 `local`，需再次确认——对未来未知内容不做一揽子授权。
- 推送时 `source_refs` 为空的 claim 照常推，只是在审核队列里排后。
- **共享正文按不可信输入处理**：长度上限、剥离 harness 保留标签（`output_tags` 全表）、转义控制字符与
  协议标签——防止知识正文变成 prompt injection。拉取侧同样过一遍。

### 6.5 proposal 契约随之改变

今天的六种 op 改为按 handle 的操作（harness 把 `@k` / `@i` / `@m` 映射回 id）：

```json
{"op":"create","kind":"term","parent":"@k3","section":"角色","payload":{…},"reason":"…"}
{"op":"update","id":"@k12","set":{"payload.zh":"…","payload.body":"…"},"reason":"…"}
{"op":"add_item","id":"@k12","field":"misheard","value":"…","reason":"…"}
{"op":"remove_item","item":"@i3","reason":"…"}
{"op":"add_membership","id":"@k12","parent":"@k3","section":"…","reason":"…"}
{"op":"move_membership","membership":"@m4","parent":"@k3","section":"…","reason":"…"}
{"op":"remove_membership","membership":"@m4","reason":"…"}
{"op":"link","id":"@k12","rel":"see_also","target":"@k7","reason":"…"}
{"op":"retire","id":"@k12","merged_into":"@k7","reason":"…"}
```

`set` 只接受标量与 prose 字段；集合字段只能 add/remove。多归属下 `move` 必须指定 membership。
`retire` **级联**：同一 rev 内关闭该节点的全部归属边、items 与既有 links（`restore` 整体带回）；
同批新写的 `supersedes` link 除外。需要 `PROMPT_VERSION` bump。

**两个前端的承载方式**：

- agent 前端（主，v8 简化）：提案**整份一次交付**——submit 时作为 accepted artifact 交回（ops 格式与
  REST 输出块同构），不做逐 op `kb_propose`。提交前可调 `kb_validate`（无副作用预检 = apply engine 的
  `preview`，对整批 op 跑 overlay 校验；批内 `create` 用 `@new1` 类 draft handle 供后续 op 引用）。
  runtime 只维护一张**只增的 handle 绑定表**（随 `kb_read*` 返回登记，同 id 幂等、重连安全），submit 时
  装进 envelope 的 `handle_bindings[]`——没有 durable draft 要管。**`submit` 不产生知识库副作用**：
  agent runtime 的 JSON/WAL 与 `knowledge.sqlite` 不是同一事务域，且工具协议不保证跨重连 exactly-once，
  在 MCP `submit` 里写库必然留下崩溃窗口。apply 由**父 harness** 在收到 accepted artifact 后走今天的
  路径：chunk ledger + proposal hash + 一个 SQLite 事务 + 按实体 CAS（§2.5）。conflict 进 apply report；
  确需 agent 重提的另起 repair / follow-up task，不回到原会话。handle 由 `kb_read*` 返回。按 100k token
  切 chunk 的必要性随之下降，chunk 边界只剩「材料超输入上限」一种。
- REST 前端：仍是输出块 `<knowledge_proposals>` JSONL，语法校验失败采样重试（今天的形态）。

**两个前端复用同一个 apply engine**：

```text
parse ops → build overlay → validate overlay → CAS once per entity → 移除 stale 实体的意图
          → 对剩余 overlay 再跑一次 validator → 全部通过才写版本行，否则按依赖闭包跳过或整体回滚
```

第二次 validator 是必需的：被跳过的 node update 所依赖的 membership/link 若照常提交，会得到一个 submit 时
合法、落库后非法的部分结果。

**accepted proposal artifact 是自包含 envelope**（父 harness 不持有 MCP 会话内的 handle 表）：

```text
envelope {schema_version, task_id, assignment_id, context_epoch, knowledge_read_rev,
          ops[], handle_bindings[]  (@k/@i/@m → id + expected_valid_from_rev),
          draft_bindings[]  (@new1 → 草稿内临时身份), required_block_digests[],
          preset_version, validator_version, input_hash}
```

`proposal_hash` 覆盖整个 canonical envelope 而非只 `ops[]`。父 harness apply 前重新校验 envelope 与任务
manifest、`knowledge_read_rev`、授权域一致，防止跨任务 handle 重放。

**durable draft（押后的升级路径，v1 不接线）**：逐 op `kb_propose` 的完整语义已定稿并落为夹具
（`node/draft.py`）——draft 落 runtime durable state（与任务 WAL 同域）、`draft_op_id` 按 canonical op
fingerprint 去重（重连后重复 `create` 不得产生两个节点）、scope 为 `(task_id, context_epoch)`、
reset/requeue 时整份 draft 连同 handle map 一起清空、修改面 `kb_draft_status / kb_drop_proposal /
kb_reset_draft`。v8 判断它换来的增量（validator 错误当场返回、不整批重试）不值当下的接线成本——
一次交付 + `kb_validate` 自检 + repair task 兜底已覆盖同一失败面。**升级条件**：实测显示知识更新会话
因整批提案错误反复重试、浪费显著时再接线；届时沿用本契约，不重新设计。

## 7. 迁移：无损导入，冲突交给人

### 7.1 导入（第一遍不做任何合并）

导入器 `python -m finesub.llm.knowledge.migrate`：

1. 解析现有 `knowledge/{streamer,common}/*.md`：H1 → subject；`## 节` → membership.section；5 段行 →
   `term`（行首字段 → surface，第 2 段 → names.zh，第 3 段 → aliases items，第 4 段 → reading，描述里
   的误听变体 → misheard items，其余 → desc/body）；streamer 的 `日期: 事件` → event、`对象 | 描述` →
   relation、`字段: 值` → fact。每行一个节点，重复 surface 也各建各的。
2. **确定性 id**：迁移期 local/item/membership id 用 UUIDv5（namespace = 迁移命名空间，name =
   `category / 相对路径 / section / 行序 / 行首字段`），并落 `migration-id-map.jsonl`。两次 shadow import
   结果可比、映射表与 merge-candidates 稳定、中断重跑同一套身份、shadow 验收的库与最终切换库是同一身份集。
   正式运行中新建节点才用 ULID。
3. 输入锁定：导入记录源 `knowledge/` 的内嵌 git commit hash（无 git 时记全树内容哈希）；切换时若源已变化
   则拒绝并要求重跑 shadow。
4. 每个节点的原行文本、源路径与行号进 `migration_aux`（§2.1，migration-only 表）供 `render_legacy`
   回填；切换后可删。
5. `translation/common-mistake.md` 与 `good-example.md` **不变**：它们是台账不是知识节点。
   ⚠ 2026-09-02 起这两份台账已退役（`knowledge.md` 的「两个翻译台账（已退役）」）：范例库
   迁进了 `style` 条目，错误库判定不迁。本条记的是当时的取舍，不是现状。

**实施记录（2026-08-22，§8 第 1–2 步已落地为 inactive shadow 包 `src/finesub/llm/knowledge/node/`）**：

- 入口 `python -m finesub.llm.knowledge.node.migrate --source <kb> --store <sqlite> --report <dir>`；
  只写目标 store 与报告目录。生产 harness 不读它。
- **verbatim 优先**：每个源行原样存 `migration_aux.legacy_raw`；`items`（alias / misheard）、`reading`、
  `native_names` 是从原文**抽取的派生索引**，不改写原文。`render_human` 在 v1 里**不做**误听写法规范化
  （§2.3 表里的「规范化写法」推迟到第二次迁移），所以 human 投影与源文件的差异只有白名单内的
  「标题后空行」一类。
- 布局标志：每个 section 的「标题后是否空行」、空 section 的空行归 separator、行间空行数
  （`blank_before`）、文件尾换行，全部记在 `migration_aux.layout`，`render_legacy` 据此回填。
- 真实库实测（6 subject / 122 node / 101 item）：legacy 投影 byte 相等；merge candidates 3 条
  （`クク竜`、`ユムカ竜` 互补，`ロスカリファ` 字段冲突）与 §1.4 一致；另发现源 `streamer/index.md`
  两条相对词条**过期**（词条有 `小唯`/`浅濑`，index 没有）——index 是派生物，parity 把它记为
  「source index stale」发现而非失败。
- 每个 section 的 `line_form` 决定分类器（fact / event / relation / term），不匹配的行落 `note`；
  strict preset 外的 section 直接拒绝导入（整个 import 事务回滚）。
- **§8 第 3 步已切换（2026-08-22）**：`KnowledgeRepo.open` 是唯一入口（无 store 且有旧 markdown → 一次性导入）；
  `base.py` 的读 API 全部走 store；`update.py` 每 chunk 取 `working_rev` 渲染带句柄的 `<kb_entries>`、
  提案经 `node/proposals.py` 翻译成引擎 op、`node/apply.py` 一个事务提交，ledger 记 `rev_before/after`，
  崩溃恢复靠 `revisions.note` 里的原始提案文本哈希；`snapshot.py`（agent 读 API）按 rev 钉读；
  `run.py` 的 knowledge 身份改为 `rev:N`；内嵌 git 全部删除（`mistakes.py` 的台账成普通文件）；
  `required_capabilities` 不再要 git；PROMPT_VERSION v78。**未做**：`keep_entries` 链与预注入形态照旧
  （agent 路径 4d 才删；REST 路径长期保留，§4.3）；`render_human` 仍不规范化误听写法。
- **人工编辑面已补（2026-08-26，§8 的 3/4 间插入项）**：`python -m finesub.llm.knowledge`
  （`node/{cli,edit,history}.py`）——`edit` 为 human 投影 round-trip diff → 共享 apply engine 的
  `kind=user` 事务（纯移动的行保持节点身份，别名列 diff 同步 items，档案 本名/别名 行的变化按
  新旧文本 delta 同步 `native_names`/`reading`/subject 别名 items）；`revert` 全有或全无、被后续
  触碰阻塞；`restore` 同 `local_id` 复活并带回同 rev 关闭的边/items/links；另有
  `show/log/new/retire/refresh`。store 补了 `revive_*` 原语。
- **复审修正（2026-08-26 第二轮）**：apply 的 op fold 改为原子（快照/回滚，半成品 create 不再随批
  提交）；只读依赖的存活检查进 `_drop_stale`（并发 retire 的 owner 不再收下 stale `add_item`）；
  `retire` 级联关闭 items/既有 links（`restore` 语义因此成立）；**pinned read 贯穿生产读取**——
  `base.pinned_generation_rev` 以 run 为界钉住默认读 rev（`run_full_correction` 包全程），知识更新每
  chunk 显式传 `working_rev` 覆盖；`add_item`/`remove_item`（aliases）反向改写 term 的 `alias_text`
  （双向同步闭环）；`matched` 事件的 `dedupe_key` 纳入 `rev`。
- **复审修正（2026-08-26 第三轮）**：模块 CLI（`correction_translation.main`）与库入口同样进 pin
  （knowledge 开关先于 pin 解析，`--knowledge none` 不开 store）；pin 附带**每 root 互斥**（RLock，
  同线程可重入、第二个 run 等待——生产本就单任务，锁是正确性兜底，防交错 pin 互读/交错退出恢复
  过期值。2026-08-30 起互斥已撤：pin 改为 run 作用域 ContextVar，并发 run 天然隔离，见
  `llm_harness_behavior.md`「任务级并行」节）；别名列 diff 与 items 增删按 `_match_normalize` **归一化身份**判定（`alias_delta`）——
  NFKC 等价改写不动 items，与 apply 引擎的重复判定同一口径；retire 级联与显式 `remove_item` 同批
  幂等（级联已关的 item 跳过），写入期 `NotFoundError` 转 rollback 而非裸抛。
- **4a 第二批（2026-08-26）**：shadow 扫描接进纠错窗口循环（`stages/correction/run.py`
  `_shadow_scan_windows`——引擎自此有生产调用方）；`kb_*` 只读四件套落进 `agent_mcp_server`
  （契约见 `llm_agent_tool_protocol.md` §2 工具表新行）。
- **4b 机制接线（2026-08-26）**：`base.active_generation_pins` + `kb_index_block_text`（server 与
  harness 共用一个组合器算 digest）；client `_agent_knowledge_binding` 在 pin 存在时给每个
  tool-session task 注入 metadata identity + `kb_index` 必读块，spawn env 带
  `FINESUB_MCP_KNOWLEDGE_ROOT`（per-call 与 session host 两条路）；server `kb_index` 只在回复
  digest 与 manifest 相等时 `record_pull`。conversational 不挂块。真机 canary 判定未跑。
- **4c 机制接线（2026-08-26）**：`complete(agent_task_extras=…)` 通道；update 每 chunk 携带
  working_rev identity / `kb_validate` 旗标 / prompt 句柄表；server `kb_validate`（translate+preview，
  只读）与 `HandleMap.seed`（sparse 编号防撞）。durable draft 仍是夹具。
- **复审修正（2026-08-26 第四轮）**：kb 授权改为**调用方显式授予**（`kb_tools: read|propose` →
  manifest metadata，server 核 manifest；pin 只供缺省 root/rev——search judge 之类无授予任务不再
  被强制进门）；standalone/reference 知识更新显式带 root+identity、无 pin 也获得完整绑定（root 进
  `metadata.kb_root`，spawn env 退为同 run 缺省）；`kb_index` 容量检查移到 `record_pull` **之前**
  （超限报错不再先结清必读台账，fail-closed 保住）；`kb_validate` 把 fold rejection 计入结果
  （`ops_translated`/`ops_appliable`/`rejected`，NFKC 重复别名不再假阳性通过）；`kb_search` 删掉
  静默前 50 截断（全部返回或撞回复上限报错提示缩小）。
- **agy 微型 canary 通过（2026-08-26，gemini-3.7-flash effort=low，一次过）**：真实 agy 工具会话，
  工具序列 `next_task → kb_index → kb_read → kb_search → kb_validate → pull_status → submit`——
  必读门在 `kb_index` 回复里当场结清（`owed_blocks: []`，digest 相等才记）；`kb_read` prompt 投影
  带 `@k` 无 `@m`；`kb_search` 从「残表」命中 misheard → 原神/`@k3`（与 `kb_read` 同一句柄空间）；
  `kb_validate` ops=2、无 problems、store rev 不动；submit 首答 accepted（repairs=0）。
  踩到一坑：agy 的 3.7-flash 硬性要求 `--effort`，直连驱动时 `reasoning_effort` 不能为空
  （生产路径由 routing/execution settings 供值，无此问题）。**这是机制级微型 canary**。
- **agy 真尺寸 canary 通过（2026-08-27，owner 授权，gemini-3.7-flash effort=low，两任务各一次过）**：
  真实库**副本**（6 subject / 四月一日词条 75 节点）+ 真实日语 ASR 材料（kaguya60 stable 120 行，
  植入一个真实 misheard「面言」）。任务 A（纠错窗 read 形态，`kb_tools=read` + 信号身份）：序列
  `next_task → kb_index → kb_search×3 → kb_read(core) → kb_read_node → submit`，门当场结清、
  repairs=0；submit 产出真实专名映射（含「みやよろー → 浅瀬みやこ」的误听链路），core 投影未撞
  24k 回复上限；曝光 8 条落账 `task=canary-prod window=w1`。任务 B（知识更新 propose 形态，真实
  `kb_handle_bindings`）：agent 自选真实句柄 `@k28` 构造 add_item+update 两行提案，`kb_validate`
  返回 `ops_translated=3 / appliable=3 / rejected=[]`；曝光 75 条（整包）落账 `window=chunk-1`；
  store rev 全程不动。**4b/4c 的 agy 真尺寸判定就此关闭**；codex/claude 两家等有动机再跑
  （工具协议同一套，driver 差异主要在传输层，已有各自的协议级测试覆盖）。
- 第 1 步的契约夹具也已落地（同包）：`envelope.py`（§6.5 自包含 envelope、整体 `proposal_hash`、
  `check_manifest` 防跨任务重放）、`draft.py`（durable draft：fingerprint 去重、`draft_op_id`、epoch 作用域、
  drop/reset、`to_envelope`）、`apply.py`（统一 apply engine：overlay 归并 → 校验 → 按实体 CAS 一次 →
  按类型丢 stale 意图 → 依赖闭包 → 二次校验 → 写版本行或整体回滚；`preview` 即 `kb_validate`）。
  两个前端切换时直接复用，不再各写一套。

### 7.2 parity：结构语义等价 + 已批准的规范化 diff

byte-equivalent 不可能：误听写法在原数据里不统一（§1.4），抽成结构化字段后重渲染必然规范化。parity 定义为：

- `render_legacy()` 与原 markdown 的 diff **只允许出现在白名单里**：误听描述的规范化、行尾空白、
  5 段行空字段的对齐。白名单每一类在迁移报告里逐行列出并人工批准。
- 结构语义等价：每个原行都能唯一映射到一个节点且字段逐一对得上（导入器输出映射表，测试据此断言）。
- golden 测试以 `render_human()` 为基线，**从切换那一刻起**锁定；切换前的 golden 仍是 markdown。

### 7.3 冲突报告与第二次迁移

`merge-candidates.jsonl`：同 subject 内 `entity_fingerprint` 相同的节点分三类——完全相同（可自动共用
一个节点，多 membership）、信息互补（人工确认后 union）、字段冲突（必须人工，如 `ロスカリファ`）。
去重作为**第二次独立迁移**在 parity 通过后执行。第二次迁移同时清理 verbatim 过渡冗余（v8.1 清单）：
desc 原文里的误听写法（`render_human` 规范化）、`fact/event/relation` 的 `sep`、term 的 `alias_text`、
payload 里残留的 `aliases`/`preset` 死数据。内嵌 git 仓库保留为历史档案；切换后不再写入。
第二次迁移的**扩展清单与条目结构定稿**（term 三段式、reading 列删除、误听退出 prompt、fact 拆分与
软词表、kind 再分类）见 §11——执行时以 §11.2 为准，本节的 v8.1 清单被它包含。

## 8. 实施顺序

切换读写真相源是**一次性**的：node 存储真正接管生产之前，现有 proposal 仍改 markdown，任何提前激活的
node 副本都会立刻过期——所以第 1–2 步只做 **inactive shadow store**，第 3 步一次切完。

1. **定稿契约**：§2.1 schema（identity/version 两层、`dedupe_key`）+ pinned read / 双 rev / overlay 归并与
   按实体 CAS / accepted 基线 / restore（§2.5）+ preset schema（§3）+ 三投影（§2.3）+ **accepted envelope 与
   draft epoch 契约**（§6.5，到 4c 才启用，但影响 handle、CAS 与 runtime schema，先写成夹具）。纯设计与
   测试夹具，无生产改动。
2. **只读 shadow importer / renderer**：确定性 id、输入锁定、无损迁移、`render_legacy` parity 与白名单
   报告、映射表、`merge-candidates.jsonl`。prompt 不动，golden 不变，可单独合入；生产仍读写 markdown。
3. **切换真相源**（一次提交）：node 写入 + handle proposal（§6.5 的 REST 输出块形态）+ SQLite 事务/rev/
   补偿 revert/restore + overlay 归并与按实体 CAS + `generation_rev` 接入 research/correction 全链、
   `working_rev` 接入知识更新 chunk 循环 + resume identity 改用 `generation_rev`（§2.5）+ 删除行号快照
   机制 + PROMPT_VERSION bump + 停止写 git + 事件表落地（平表，§4.2；此时只有 REST 预注入一个
   前端在产事件）。**保留**现有关键词预注入、查询轮、`keep_entries` 链与 agent 整段 prompt——这一步
   只换存储，不动数据面形态。
   验收：session_replay 全 6 会话 dry-run、知识更新 chunk ledger 的崩溃恢复测试、多 chunk read-your-writes、
   `knowledge revert`/`restore`、长任务中途并发写入的 pinned read + stale write CAS 测试、同批多 op 归并
   测试。确定性迁移 id（§7.1）**必须在此步之前**到位。
**3 与 4 之间（v8 插入，2026-08-26 已完成）：人工编辑面**——真相源已切换而编辑面缺位是当时唯一的可用性回退：
   `knowledge node` CLI（create/update/retire/add-item，薄封装 Transaction）+ `knowledge edit <subject>`
   （human 投影吐进 $EDITOR，改完用导入器重解析、diff 成 op、走 kind=user 的正常事务——解析器已被
   parity 验证）+ `knowledge log`（列 `revisions`）。

4. **检索引擎与数据面上线**，严格按 `llm_agent_tool_protocol.md` §7 的顺序，每一小步有独立闸门：
   - **4a 只读工具 + shadow 事件（2026-08-26 已实施）**：倒排索引与精确匹配引擎（§4.2，只落平事件，
     不产建议）；shadow 扫描接进纠错窗口循环（`_shadow_scan_windows`：窗口定稿后按 pinned rev 整批
     扫描，fail-soft、按 `(task, window, rev)` 幂等）；`kb_index / kb_search / kb_read / kb_read_node`
     已在 `agent_mcp_server` 实现（硬上限 `KB_REPLY_MAX_CHARS`、无分页、回复带 rev 与 `result_digest`
     随 frame log 审计、session 级 HandleMap；暴露 = spawn env `FINESUB_MCP_KNOWLEDGE_ROOT`，准入 =
     manifest `metadata.knowledge_identity`，读 rev 全由 manifest 决定）；prompt 投影已停渲染 `@m`
     （v79）。
   - **4b 启用 `kb_index` 必读门**（§4.3，fail-closed；**机制已接线 2026-08-26**）：generation pin
     存在时，tool-session 两条路（per-call 与 pseudo-conversational）自动给 spawn env 注入
     `FINESUB_MCP_KNOWLEDGE_ROOT`、给每个 task 的 manifest metadata 带 `knowledge_identity`（显式
     call kwarg 可覆盖，留给 4c 的 working_rev）并挂 `kb_index` 必读块——digest 相等才记 pulled，
     manifest 里该块标 `read: tool`，`read_context` 对它指路回工具。**conversational 例外**：其控制
     协议没有 kb 工具，不挂块（记 followup）。REST 前端不改造（v8 取消原 selected-set 与
     `whole`/`by-hit` 计划）。**判定（未跑）**：真机 canary 看 agent 是否按 owed 提示自寻
     `kb_index`（必要时 bootstrap 补一句）、门与审计如实落账。
   - **4c 启用 `kb_validate` + 整份提案 artifact**（§6.5；**机制已接线 2026-08-26**）：知识更新的
     提案本就整份交付（输出块），REST 照旧、apply 仍在父 harness；agent 化增量 = `complete` 的
     `agent_task_extras` 通道——update 每 chunk 传 `knowledge_identity=rev:<working_rev>`（覆盖 run
     pin）、`kb_validate` 准入旗标、prompt 的 `kb_handle_bindings`（manifest 携带、server 端
     `HandleMap.seed` 播种，工具与提案块共享一个句柄空间）；server `kb_validate` 跑共享
     translate+preview 无副作用预检，仅 knowledge-update 任务准入。durable draft 不接线
     （升级条件见 §6.5）。真机 canary 未跑。
   - **4d 已关闭：`keep_entries` 保留（owner 2026-08-26 定）**。这是一个被如实承认的 trade-off：
     parallel 形态在 v75 实质弃用它，证明「可弃」；但弃用不是免费的——API 形态下 keep 集出自
     正在纠错的 capable 模型（优于 35fl 查询轮的旁观选择），**同一窗口内 query→correction 的
     衔接**里它仍承担实质作用，per-session agent 里还买到会话惯性（少一轮 `kb_read` 自取）。
     删除的唯一动机是降复杂度，不构成充分理由——**没有充分理由就不删**。若将来出现动机，按
     验证阶梯走：离线 shadow 对照（exchange log 重放，量化模型 keep 集 vs harness 确定性携带集
     的分歧与 landed 占比）→ A/B（臂间 n≥5 + 噪声基线）→ 三家 production-size canary，全过才删。
     transfer state 与两形态的注入数据面照旧；原挂在 4d 的三家 production-size canary 要求改挂到
     4b/4c 工具面的真尺寸验收上。
   A/B（参照既有采样纪律：臂间 n≥5 + 同配置复跑）除最终字幕质量外单独测 retrieval recall、
   false-injection rate（分母 exposed）、refined 保留率。
5. **字段级事件 + evidence 落账**（§5，含 UNIQUE 键与 `refined_aligned` 回写）+ `report` CLI。
   **定位为第 6 步的前置**（v8）：紧贴共享实施之前做，单用户期只有 4a 的平事件计数。
   **已实施（2026-08-27，`node/signals.py` + `knowledge/report.py`）**：`exposed` 两处接线
   （REST = attempts 提交点按窗整包落账、misheard 命中分 `opportunity=correction`；agent =
   `kb_read`/`kb_read_node` 通过容量检查后落账）；`landed` 在 run 收尾逐窗对照（定名新出现于
   corrected；rendered 段按 `source_ids` 归窗）；evidence 表补 `created_at`（store schema v2,
   就地 ALTER 迁移，dedupe 键不含时间戳）；`refined_alignment_evidence` 在 refined_aligned 模式
   （execute+apply）先于 LLM chunk 确定性回写 confirmed/refuted——只确证/推翻 misheard item 那
   一个 claim（§5.3），mistake 台账仍归模型提案；report CLI 按 §5.5 出计数/最近印证/从未命中/
   高误触发准入（`--min-exposures`，默认 3 个 correction 曝光窗口）/按 matcher 转化。全部
   fail-soft + dedupe 幂等；事件永不进 prompt。
6. **共享**：canonical identity 与 redirects、**子实体 canonical id 与 bundle-local handle**、`sync_state`、
   字段级合并策略、`share_inherit` 与 `visibility`、claim 级证据摘要、不可信内容边界，然后才是服务端
   拉/推/审。这一步的身份协议不阻塞 1–5。
   **首个可运行形态已实施（2026-08-27，`knowledge/share/`：exchange/sync/client/cli/server）**：
   - **交换**（`exchange.py`）：push bundle 全部引用走 bundle-local handle 或 `c:<canonical>`——
     local_id 不上线（server 端 `validate_bundle` 拒收，带 handle_map 的未剥离 bundle 同样拒收）；
     claim 摘要脱敏聚合（kinds/source_refs/confirmed/refuted/exposed/landed 计数），指向 item 的
     claim 改写为 item handle、映射不到就丢弃。§6.4 边界 `sanitize_text` 推拉两向执行：保留标签
     整段剥除、其余标签形态转义为全角、控制字符删除、逐字段长度上限。
   - **服务端**（`server.py`，stdlib ThreadingHTTPServer + 同一 SQLite 复用 node schema 加
     share_queue/contributors/share_chain 三表）：`POST /register` 匿名 token；`GET /snapshot` 带
     完整 chain history（`chain_hash = sha256(prev + content_digest)`）；`POST /push` 按客户端
     幂等键去重（重传返回同一 queue item）；`GET /queue` 租约（token+到期，租内对第二个
     maintainer 会话不可见）+ 指纹 merge 提示；`POST /verdict` 按 `verdict_version` CAS——approve
     的 CAS、bundle apply、chain append 是**一个事务**（队列与语料同库正是为此）。`merge` 映射
     实现 §6.2 的 `merge_into`：重复节点挂到既有 canonical 节点（payload 不取，item/membership
     按归一化身份去重收敛）；无 merge 裁定绝不自动合并（§9）。审核 LLM 不在服务器上。
   - **拉取合并**（`sync.py`）：先验 chain 再查防回滚锚点（客户端每 remote 存
     `(server_rev, chain_hash)`，history 不含锚点或 rev 倒退即拒绝，任何写入之前）；标量 three-way
     （base = 上次拉取的远端 payload，从版本行历史按哈希找回；两侧都动 → conflict 记报告、本地
     保留，且因 base 不再可寻会**持续重报**直到人工收敛，绝不静默倒向远端）；prose 保本地标
     待合并；items/memberships 按 canonical 子实体 id 并集 + 单调 tombstone（本地删除在服务端
     retire 前不复活）；links 按 (source, rel, target) 自然键；redirects 先行、canonical 改写就地
     （不产版本行）。一次 pull 一个 `kind=pull` revision。
   - **客户端 CLI**（`cli.py`）：`mark`（显式勾选；默认只有 subject/term 继承 shareable，其余
     kind 要 `--kinds` 二次确认，§6.4）→ `push`（只打包 shareable 节点，本地留 push 记录）→
     `status`（批准后按 assigned 回填 canonical_*id 与 sync_state——推送者的身份闭环）→ `pull`。
   - **审核 LLM 半边已实施（2026-08-27，`share/review.py` + `prompt_templates/share_review_v1.md`）**：
     `share review --remote … --maintainer-token …`（维护者本机跑，key 不上服务器）——先出
     §6.3 门槛的确定性预检（精修印证/外部印证/独立来源三路，逐 claim 标「需外部印证」），
     **默认 dry-run** 只渲染审核 prompt；`--execute` 跑一次审核会话（请求 native web，
     agent 形态可逐 claim 产 URL 出处；REST 兜底自然给不出外部印证，门槛如实不过）；会话
     输出 `<review_verdict>` JSON（approve/reject + merge 映射 + external_evidence）；
     `--post` 才回写 verdict——批准事务里服务端把 external_evidence 经 handle→canonical
     重键落 `evidence(evidence_kind=external)`（映射不上的丢弃不猜）。bundle 正文按不可信
     输入对待写进了 prompt 纪律；LLM 结论不算独立来源。
   - **部署件已备（2026-08-27，`deploy/share-server/`）**：systemd 单元（hardened，token 走
     override 注入）、Caddyfile（TLS/暴露归 Caddy，服务只听 127.0.0.1）、litestream.yml、
     README（终态 VPS 步骤 + 过渡 Cloudflare Tunnel 同构迁移 + 备份/恢复与防回滚注意——
     旧备份覆盖新库会被客户端锚点拒绝，回滚须同步清客户端锚点）。
   - **未做**（后续）：GitHub snapshot 镜像（README 记了手动做法，未内建自动推送）、
     `independent_contributor_count` 聚合与队列排序。本机模拟测试（owner 同意的过渡）由
     默认测试套件的 HTTP 回环用例覆盖。
   - **复审修正（2026-08-27 第七轮，6 条）**：
     1. 门槛成为真 gate——claim 参照集改为 `bundle_claims`（对每个可共享标量/item/prose 生成，
        零证据的新内容如实显示未过门槛，而非空报告全过）；槽位按 subject category 判定（streamer
        档案 fact 只认外部出处、精修不放行；关系/经历只认精修/独立来源、外部不单独放行；外部路
        要求 `evidence_kind=external`，不再任意 `source_refs` 非空即过）；**服务端在 approve 时
        执行同一 gate**（逻辑在 stdlib 的 exchange.py，CLI 与 server 共用不漂移），未过门槛拒绝
        批准，除非维护者显式 `override` 并留因（记入 verdict note）。
     2. 审核证据必须匹配**入队冻结的 claim**——verdict 的 external_evidence 逐行对
        `bundle_claims + claim_summaries` 的 (node, field_path, value_hash) 精确匹配 + 严格
        `https?://` URL，不匹配拒绝整个 verdict；审核模型不能凭空造「已确认外部证据」。
     3. canonical 锚点必须真实存在——push 入队前逐个查 `canonical_id` 与 `c:` 引用在服务端现行
        rev 存在（否则 400）；`_apply_bundle` 对未知 canonical 改为报错（双保险），新实体 id
        一律服务端分配，客户端命名的 id 永不落地。
     4. 公网配额——注册数 ≤500、每贡献者 pending ≤5（幂等重试不受限）、全局 pending ≤200、
        pending 30 天惰性过期；Caddyfile 补 IP 限速插件注释，README 记两层防护。
     5. dry-run 改只读 `GET /queue/peek`（看不加锁）；lease 支持 `queue_id`/`limit` 定向单租；
        新增 `POST /release`；review CLI 逐项租一、失败释放、`--override-thresholds REASON`。
     6. evidence 去重键纳入 `evidence_kind + source_ref`——同一 claim 的两个 URL 是两条证据，
        重放同一来源仍去重。
   - **自查修正（2026-08-27，第七轮后）**：第七轮的门槛实现仍把 bundle 自带的
     `claim_summaries` 当放行依据——那是**贡献者自述**（客户端从自己本地 evidence 聚合），
     伪造 `evidence_kinds`/`source_refs` 即可绕过 gate。收紧为：**自述一律不满足门槛**，只作
     审核 prompt 里的参考信息（`self_reported_*` 字段，标注不可核实）；能满足门槛的只有走
     维护者流程的 verdict `external_evidence`（且仅对接受外部路的槽位——关系/经历两路服务端
     都无法核实，其批准恒为显式 override，等跨用户聚合落地后精修/独立来源两路才可能自动化）。
     连带收紧：verdict 证据的 allowed 集只认 `bundle_claims`（哈希来自 bundle 实际值），自述
     summaries 的任意三元组不再是可落账目标；merge 并入节点的 payload claim 免 gate（其标量
     根本不落库，items 仍计）。回归：伪造自述过不了 gate、伪造三元组整单拒绝。
   - **复审修正（2026-08-27 第五轮，8 条）**：
     1. agent 曝光身份——per-call 任务的 runtime task_id 恒为 "call"，曝光会跨窗口折叠：stage
        调用方经 `agent_task_extras` 显式带 `kb_signal_task`（run task）/`kb_signal_window`
        （chunk id），client 折进 manifest metadata（并从 run_kwargs 排除），server 曝光按它
        落账（无上游身份退 `assignment_id:task_id`）；agent 曝光的 correction 口径由 report
        侧 join 派生（同 (node, task, window) 有 misheard matched 即计入分母）。
     2. 服务端入队清洗——`push` 在任何持久化之前 `sanitize_bundle`（逐字段重建、只留已知键，
        绕过 CLI 的恶意 bundle 与审核队列之间不再有未清洗窗口）；claim summaries 逐字段校验
        （field_path/value_hash 形状、计数非负整数）；bundle 数量硬上限；HTTP body 2MB 上限
        且在服务锁**外**读取。
     3. 链路连通性——history 每项带 `content_digest`，`verify_snapshot` 逐项验证 rev 严格递增
        与 `chain_hash = H(prev, content_digest)` 全链转移；「锚点出现在列表里」不再放行断链
        伪造（trusted → disconnected → head 的构造被拒）。
     4. push 幂等闭环——CLI 发送**之前**把 `{idempotency_key, bundle_digest, handle maps}` 落
        `share-pushes.jsonl`（status=intent），响应丢失后的重试按内容摘要找回同一 key；server
        对已存在 key 校验 contributor 与内容摘要一致（不一致 409），重传不产生第二个队列项。
     5. report 证据按 `(node, field_path, value_hash)` 聚合，只有与**当前**字段值 digest 相符的
        行贴在当前值上，其余标 `[stale value]` 单列——item 改值后旧 claim 不再冒充新值的证据。
     6. share CLI 写命令（register/mark/unmark/status/pull）进 `knowledge_write_lock`；
        `apply_snapshot` 的 canonical→local 映射移进 BEGIN IMMEDIATE 事务内，并发 pull 不再
        各建一份同 canonical 节点。
     7. `share_inherit` 接入创建路径——apply 引擎 `create` 在共享 subject 下按 preset 继承
        （term → shareable；fact 第六轮收紧为 local，见下）；CLI 补 `unmark` 与
        `--match`（按标签文本选择单条 fact，不再一勾全 kind）。
     8. `GET /push/<id>` 加归属鉴权：contributor token 只能查自己的队列项，maintainer token
        全可见；客户端随请求带 token。
   - **复审修正（2026-08-27 第六轮，5 条）**：
     1. 清洗深递归——`sanitize_value` 对任意 JSON 深度递归（list 里的 mapping 不再漏）；且服务端
        入队按 kind 校验 payload **形状**（标量 / 标量列表 / 仅 `valid` 一层 mapping），未知嵌套
        结构直接拒收——审核 LLM 不该收到 schema 没数过的结构。
     2. `share_inherit` 收紧为 **term-only**：两个 preset 的 `fact` 改 `local`（新 fact 可能是
        本名，kind 级判断表达不了字段敏感性；每条 fact 都走显式 `mark --kinds fact --match`）。
     3. shareable 集合保持**祖先闭包**：`mark` 选中子节点自动带上父链到 subject；`unmark` 向下
        级联（含继承出的 term）；`build_push_bundle` 只沿 shareable 路径下探（local subject 下的
        shareable fact 不出门）；`validate_bundle` 校验 bundle 内每个无 canonical 的非 subject
        节点都有从 subject / `c:` 锚点出发的 membership 路径，孤儿拒收。
     4. push 幂等摘要排除 `base_rev`（无关本地 revision 不再作废 intent）；`push` 进 CLI 写锁
        （并发 push 对同内容收敛到同一 key）。
     5. fast 路径传 `kb_signal_window`（`_call_and_parse` 新参数）；report 的 matched/exposed
        join 键改 `(task, window, rev)`——同素材在新 rev 重跑不再把 rev1 的 matched 和 rev2 的
        exposure 拼成 correction opportunity。
7. 可选：`kb_search` 的模糊音近 matcher（**重启条件**：报告显示同一词条反复以**不同**表面形态误听、
   精确缓存反复 miss；实现时也只作为 agent 显式发起的检索模式，不进自动注入）；BM25/embedding 第六级
   检索；LLM 对 prose 的二元盲评；durable draft 接线（条件见 §6.5）；去重迁移第二遍（§7.3）可在 3 之后
   任意时点。

## 9. 明确不做

- 事件/证据/派生计数回流到模型 prompt。
- node 级 confidence 标量。
- 自动合并同 surface 节点（只产候选）。
- 自动删除任何节点（`retire` 只来自提案 + 人工/审核确认；tombstone 单调；revert 是补偿事务不是倒退；
  自动流程不复活，restore 只由用户显式发起）。
- stale 提案覆盖现行版本（apply 一律按实体 CAS，且先归并 overlay——逐 op CAS 会把同批前序 op 当 stale）；
  任务中途切换 `generation_rev`。
- 在 MCP `submit` 里写 `knowledge.sqlite`；任何任务拿到 `kb_apply`；`kb_read` 无上限返回。
- 旧预注入/transfer 先于 pull 工具删除。
- 初版做 GC；工具写死读 `generation_rev`（读哪个 rev 由 manifest 的 `knowledge_read_rev` 决定）；同一 agent 会话内改 pinned rev；父 harness 依赖 MCP 会话内 handle 表；必读门凭「调用过一次」放行。
- 上传包里出现客户端 local_id；多列 UNIQUE 含 NULL 列当去重键。
- 迁移期用随机 id。
- markdown 作为编辑面；JSONL 作为本地真相。
- 在内嵌 git 之上建共享同步。
- v1 实现任何模糊 matcher（schema 字段保留；重启条件见 §8 第 7 步）；用 `matched` 作误触发分母。
- harness 扫描结果自动进注入或必读台账（v8：只落 shadow 事件；恢复条件 = shadow 对照证明确有漏改）。
- `kb_*` 返回分页游标（硬上限 + 报错缩小范围；库大到 core 投影装不下再议）。
- v1 接线 durable draft / 逐 op `kb_propose`（升级条件见 §6.5）。
- prompt 投影渲染现行 op 集合消费不到的 handle（`@m`）。
- 强制双前端回归覆盖（共享契约改动测当次主要前端即可，§4.3）。
- payload 里存 items 的副本（索引行与 5 段行的别名列从 items 渲染；`alias_text` 仅 verbatim 过渡）。
- payload 同时存 `category` 与 `preset`（preset 由 category 派生）。
- 双类型 `relation.target`；结构化 `names{}`；`subject.style`（整体描述走 note 子节点）。
- byte-equivalent parity。
- 勾选 subject 后对 relation / 真实姓名 / prose 的新内容一揽子授权共享。
- 把渲染器或检索引擎绑死到预注入形态；为 REST 前端单独做一套检索逻辑。

## 10. 文档同步（实施时）

- `knowledge.md`：目录结构、条目结构、proposal schema、apply 层、局部检索匹配各节按本文重写；
  内嵌 git 一节改为「历史档案」；resume 的 snapshot identity 改述为 `generation_rev`。
- `llm_prompts.md`：preset 移出 `prompt_templates/`；新 proposal 块；PROMPT_VERSION。
- `llm_design_notes.md`：补记 node/item/membership、版本行与 pinned read、claim evidence、四级事件、
  preset 分层、SQLite 取舍。
- `llm_harness_behavior.md`：REST 预注入维持现状的说明 + shadow 事件；agent `kb_*` 工具与两种 `exposed` 的定义。
- `agent-tasks/run-audit/references/artifact-map.md`：新事件类型、evidence 与 report 产物。

## 11. 二次设计（2026-08-27 预览审查后定稿）

依据：真实库拆分预览的逐层核对（6 subject / 59 term / 33 fact / 23 note / 4 event / 3 relation /
101 items；prompt 投影、匹配语料、evidence 生产者与消费者全部实查），加 owner 多轮讨论。执行载体
是 §7.3 的第二次迁移 + 少量 preset/validator/投影改动 + 一次 schema 迁移（v3：`maturity` 列）。
**实施进度**：§11.2 条目结构已落地（2026-08-27——term 三/四段投影与解析、表面形统一 items、误听
录入语法+剥离、fact 严格校验与稀疏 scaffold、频道用语 preset、relation 入匹配语料、`second-pass`
CLI 确定式半 + 候选报告，PROMPT_VERSION v80；真库副本验证 103 变更/25 候选/幂等。**活库执行待
owner 确认**）。**其中 term 行文法与 PROMPT_VERSION 已被 kb-followups A1 迭代**（2026-08-28，计划正文在本地
`docs/archive/kb-followups-plan.md`）：三/四段投影升级为**固定四列** `源|中|别名|desc`（v2 语法，v81），
候选账本（candidate_decisions，schema v4/v5）接管人工裁定——本节其余记录仍为准。§11.3 编辑面已落地（同日——rendered/ per-file manifest、diff 面 refresh 与脏文件保护、
run 启动收割（worktree 闸门+写锁+逐文件隔离）、share approve 归因独立 `share` revision kind）；
§11.4 证据词汇已落地（同日——verdict+`unverifiable`、`apply_envelope` 收尾的 user/transcript
revision 级 provenance 记账、report 的 claim 状态现算）。§11.5 已落地（同日——schema v3 双层
`maturity`、`approve_tentative`（仅纯新建 term，changed-payload/非 term 拒）、tentative 全
model-facing 面过滤（投影/index/`kb_search`；影子扫描 `include_tentative=True` 采证）、snapshot/
pull 携带并跟随生命周期、`POST /digest`（过期清扫+参考计票+工作单+90 天提名退役+默认关的
auto-tentative 门）。**两处有意偏离 §11.5 原文**：a) 「每 digest 一个 revision/chain entry」未做——
verdict 的逐项 CAS/租约事务性保留，链长在贡献量成为问题前不值得为其重构；b) tentative 退场为
digest **提名+人工确认**而非自动 retire（守 §9「不自动删除」）。修复 prompt/skill 与 KB 校验任务同日落地（`node/repair.py`+`kb_repair_v1.md`（候选/素材两模式，
proposals 走 validate→apply 单一通道）、`knowledge/verify.py`+`kb_verify_v1.md`（confirmed 必须带
URL、unverifiable 终态）、三会话共用 `fragment_kb_judgment_v1.md`、对话式编辑走同一套 `edit`/`apply`
命令，⚠ 这里原写的 `.claude/skills/kb-edit` **从未存在**（2026-09-01 更正））。**§11 至此全部实施完毕**。活库的迁移已于 2026-08-29 随行文法 v3
执行完毕（Phase A–D，行文法 v3 重导；见 [`knowledge.md`](../knowledge.md)）；`second-pass`
CLI 的确定式半在那一版拆给了 `phase-b`，候选扫描留在 `scan.py`。共享 merge prompt 片段的
进一步合流仍待 owner。

### 11.1 Owner 原话锚点（防漂移）

> 以下 O# 是 owner 明确表达的判断与决定的精简转述。后续实现若与本节冲突，以本节为准或回头问 owner；
> 其余小节是围绕这些锚点展开的设计，可以演化，锚点不可以。

- **O1** 律にゃー（听众统称）、スタンドちゃん（人设支架）这类条目不适合归入 relation。
- **O2** term 五段式的读音列几乎派不上用处、浪费 token（本意只为特殊读音的汉字）；后续明确：
  **考虑直接删除 reading 列**，少数符合条件的另行记录。
- **O3** 老库迁移采用**确定式转化 + one-time LLM（或 agent-only）修复**；修复 prompt 可独立使用，
  泛化为「一段文本 → 本库标准」，乃至复用共享 merge prompt 的部分。
- **O4** 用户直接编辑渲染出的 kb 条目，最好能自动解析并更新；有问题时用 O3 的修复兜底。
- **O5** 投影的写入和检测到 diff 时的写回**只做 diff 的面**；LLM 任务启动时先自动应用用户编辑；
  修改历史区分用户编辑与模型编辑（后者如有必要再细分）。
- **O6** 常见模式：用户通过对话让 agent 执行更新、或基于用户提供的资料更新知识库——在 manual
  或 skill 里把相应 prompt 与 validate 等方法暴露给 agent（「执行时的 conversational agent」）；
  future：其他描述不够清晰的功能也可如此处理。
- **O7** 纠错 CSV 的 conf 是另一语义——**假设已有资料为真**，模型对本次纠错准确性的自信——与知识库
  无关；校准可以做，但不属知识库。
- **O8** 用户自己修改的条目应作为**高质量信源**。
- **O9** 每次修改最好带 source，分 **url / 用户提供 / 视频内信息**；url 类允许后续二次验证；可增设
  知识库校验任务（prompt），对缺乏验证的进行校验，无法校验则另行标注。
- **O10** 共享聚合可加权：单词条**每用户的置信度有上限，所有用户合计又有总上限**；或用校验任务核实。
- **O11** 服务器定期 **digest**（定时，或 diff 积累到一定数量，whichever comes first）；低置信条目
  四选项（攒队列 / 入库不分发 / 分发但标记 / 其他）中，采纳「分发但标记」的变体（11.5）。
- **O12** 自动进 tentative（信誉门槛路）**可以接线，但开关先不开**，仍走 LLM 审核。
- **O13** 误听发散且难录全（ルミ 四个误听形即证据）；两种污染要防：模糊/短词误匹配、同名词条跨大
  类别归属。注入选择通常靠任务 note 背景、出处 url 背景、raw 字幕语义即可判断大类别。
- **O14** 「茜特菈莉被误纠为夏洛特」是单次任务的偶然犯傻，写成警告或属建议过拟合。
- **O15** fact「其他」拆分有理，但粉丝名/额外设定等字段**稀疏**（有些主播没有）——不要固定槽位。
- **O16** kb_validate 工具化而提交独立的设计，经讨论确认**保留**（会话只提案、apply 引擎持写权）。

### 11.2 条目结构定稿（并入第二次迁移执行）

> **行文法已被 kb-followups A1 取代（2026-08-28）**：下条的「三段 + 别名按需后缀」演进为
> **固定四列** `源|中|别名|desc`（别名列恒在第三、desc 居末可含竖线、恰三段报错）——可选尾列
> 是「省略即删除」静默损坏的根源。items 统一、reading 删除等其余定稿不变。

- **term 三段式** `surface|zh|desc`：`reading` 列删除（O2），`alias_text` 列取消——一切**备选表面形**
  （别名、特殊读音、误听）统一归 items（§9「payload 不存 items 副本」的最终形态）。特殊读音存为
  alias item（ASR 常直接输出假名，作 item 才能被 exact 命中触发注入；短读音 `exact_enabled=false`
  防误报、留作未来模糊匹配语音键）。prompt 行 = 三段 + 别名**按需**后缀。
- **误听 items 只供 harness 消费，不渲染进 prompt**：定位是**事后观察日志**（非枚举，O13 的发散性
  正是"只录观测、靠重复度筛选"的理由）。消费 = `kb_search`（agent 用残破文本找到节点）、matched
  **shadow** 信号（misheard 命中 = 纠错机会，report 的 join 依赖它）、未来模糊匹配。**不喂任何
  自动注入**——v8 的"harness 扫描只作 shadow"决定（§1.4/§4.2/§9）不因本节重开；misheard-only
  窗口拿不到注入是已知代价，也是将来按 §9 恢复条件（shadow 对照证明确有漏改）走验证阶梯的动机，
  在那之前 REST 侧维持现有关键词预注入不变。收集渠道纯结构化（更新会话 `add_item` ops、用户编辑；
  可选：精修对照确定式产出候选）；修剪由冷数据报告提名。desc 中的误听散文与反误纠警告全部撤出；
  **复发性**误纠进错题账（重复才升精选；O14 的 n=1 不入库）。
- **fact**：禁空值（稀疏用**缺席**表达，O15；现库六条空 自称/口癖/语体 删除）；「档案.其他」拆为
  独立 fact（共享 claim 的 value_hash 粒度随之修对）；field 名保持自由字符串 + preset **软词表**
  （推荐 field 名列表，表外 warning 不 error——硬枚举正是「其他」大杂烩的成因），修复/ingest prompt
  以词表为归一化目标；field 名稳定关乎共享 claim 的 `field_path` 对齐。
- **relation**：validator 准入 = target 须为人/组织，频道名词一律 term；允许挂 alias/misheard items；
  `target` 纳入匹配语料（消灭"relation = 检索黑洞"）。
- **preset**：streamer 加「频道用语」section（`kinds=["term","note"]`，`line_form="term"`）；
  律にゃー/スタンドちゃん 等迁为 term。kind 翻转连带共享槽位变化，一律**脚本提候选、人工确认**。
- **subject**：删残留 `payload.aliases`；版本篇章名从 subject alias 摘除（独立 term 已覆盖）；
  term desc 中的版本性叙事移入 event 或删除。

### 11.3 投影与编辑面（O4/O5/O6）

- rendered/ 升格为**可编辑投影**，但编辑面仍是 ops、冲突时 store 赢（§9「markdown 不作编辑面」指
  的是真相地位，不禁止作录入面）。逐 subject 的 diff 回写已存在（`node/edit.py`）；补的是：
  per-file sidecar 记（生成 rev, 内容 hash），refresh 只写变更 subject、**脏文件不覆盖**（O5）。
- LLM 任务启动时在现有 knowledge auto-apply 检查点内（同写锁、同 worktree 闸门）收割干净编辑；
  解析失败仅告警跳过、逐文件隔离，**启动时绝不自动烧 LLM 修复**（dry-run 纪律）。
  ⚠ **2026-09-03 收窄**：只有**文件级**失败才整份跳过；**行级**看不懂的行原样泊进「待归类」、
  其余编辑照常落库，泊进去的行按构造就是 `staging-line` 候选，由既有 `repair` 接手——这才是
  O4「有问题时用 O3 的修复兜底」的落点，而「启动时不烧配额」一字未动。现行行为见
  [`knowledge.md`](../knowledge.md)。
- 归因：rendered 收割记 `revision_kind=user`（作者是人，与触发时机无关）；模型编辑要细分时从统一
  更新路径传 `revision_kind`；share approve 从复用 "import" 改为独立 kind。
- 第二次迁移形态（O3）：确定式规则先行，LLM 修复只碰规则标出的判断题，输出 ops 走 validate→apply
  单一通道（O16）；修复 prompt fragment 化，与 reference_ingest、share review 共享判断片段。
  另做 repo skill 把「文本→ops→validate→确认→apply」暴露给对话式 agent（O6）；泛化准入判据 =
  该功能有确定性 validator + dry-run + 幂等 apply。

### 11.4 置信度

- 纪律维持 §9：无统一置信分数（claim 状态 `unverified/corroborated/contested/suspect/evidence-stale`
  由 report 层从证据行推导，**不入库**）；模型自报永不获机器权重；证据不进生产 prompt——显式例外
  仅**人裁定的会话**（share review 属此类），例外判据写进 knowledge.md。
- evidence 表零列变更，只扩词汇：`verdict` 加 `unverifiable`（校验任务终态，防重试）；
  `evidence_kind` 定为 `refined / external / user / transcript`（O9 三分类的落点：url→external、
  用户→user、视频内→transcript+span）。`revision_kind=user` 的 apply 对触及 claim 自动记 user
  证据（O8；本地高信，共享侧不冒充 external）；知识更新 apply 把 hint 的 `source_ids` 翻译成
  transcript 证据行（现在这个出处在产物里蒸发）。
- KB 校验任务（O9）= share review 会话的本地化：同判断 fragment、native_search、dry-run 默认，
  消费 report 生成的缺证据清单；只对有外部路的槽跑网络校验，搜不到的槽标 `unverifiable`。
- conf 列校准分析归 prompt 迭代线（O7），结论决定该列去留。
- 行为闭环（refuted 降权、注入过滤）押后但**验收线定量**：≥5 真实 run + ≥2 精修对照落库后，先人工
  核 refuted 精确率（防"exposed 但该窗口没用上"的假阴性），再谈任何 gate。

### 11.5 共享 digest（O10/O11/O12）

- digest 定时或 diff 攒够触发；**每 digest 一个 revision + 一个 chain entry + 一个 snapshot**（链长
  与 pull 节奏自然化，GitHub 快照镜像变为"每 digest 推一次"）。跨贡献者聚合在 digest 分组时算：
  每 claim 每贡献者 1 票封顶、只计有被批准记录的贡献者——O10 的封顶加权取此最简形。**计票在可抗
  Sybil 的身份/信誉机制存在之前只作 review 参考，不满足任何门槛**（注册 token 是匿名自助的
  **写入凭证**，不是证据身份，两者必须分开；换号即可伪装独立来源）；关系/经历槽因此维持显式
  override，直到身份机制落地——届时总上限按槽位差异化（关系/经历可至满足门槛，term/档案 fact
  只到 review 参考）。
- 低置信 term claim 采纳「分发但标记」变体：`maturity=tentative` 入库并分发，接收端默认
  **shadow-only**——tentative（node 与 item 同律）从**一切 model-facing 面**排除：prompt 投影、
  `kb_search`、`kb_index` 都过滤（工具响应也是影响模型的通道，返回了就不是 shadow）；只有
  shadow matcher 消费它采 matched/精修对照证据。分发的唯一效果是采集佐证（佐证只能来自还没有
  该条目的人），错误知识不影响任何输出。否决记录：攒队列（30 天过期 + 配额饥饿掐死慢热佐证）、
  入库不分发（snapshot≠corpus 会复杂化 `verify_snapshot` 与防回滚）。tentative 长期无佐证由 digest
  **提名** retire、维护者工作单确认（维持 §9「不自动删除」）。
- schema/协议：`maturity` 列（normal/tentative）**同时落 `node_versions` 与 `item_versions`**——
  置信是 claim 级（11.4），node 级单列表达不了"已验证 term 新增一条低置信 misheard"：整节点标
  tentative 会连带隐藏已验证的 surface/zh，只标 node 又管不住新 item。配套约束：tentative **只用于
  新建实体**（新节点整棵 tentative、既有节点下的新 item 逐条 tentative）；对既有 normal 实体的
  payload 更新**不适用** tentative，一律走严格审核路——不存在"半 tentative 的 payload"。node 与
  item 的 tentative 语义一致 = 退出全部 model-facing 面、仅参与 shadow 采证（见上）。列进 content_digest、随 bundle/
  snapshot 传输；是生命周期状态**不是**置信标量，不违 §9。verdict 加 `approve_tentative`
  （仅 term 槽合法，槽位分流钉在 exchange.py 门槛报告，server 不长第二套）；自动 tentative 路
  （sanity + 信誉门槛免审）**接线但配置默认关**（O12）——关时审核会话是进 tentative 的唯一路由。
- 简化连带：lease/release 协议降级为 digest 审核工作单 + 整单锁（现实只有一个维护者）；LLM 审核
  只看危险槽 + 争议项 + merge 冲突。信任模型例外（开关开启后 term 可无 verdict 入库；缓解 = 清洗
  与锚照跑、信誉门槛、shadow-only、提名退场）须在 knowledge.md 共享节记为显式决定。
- `README_DEV.md` 产物树与「运行时路径解析契约」：`knowledge.sqlite` 与 `rendered/`。
