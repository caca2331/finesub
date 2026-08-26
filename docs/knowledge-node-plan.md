# 知识库 node 模型、检索与共享 —— 设计稿

状态：设计稿（2026-08-22 三轮讨论收敛），待实施。本文**取代**原
`kb_entry_scoring_plan.md`（已归档到 `docs/archive/`，理由见 §1.2）。今天的知识库形态仍以
[`knowledge.md`](knowledge.md) 为准；本文描述的是目标形态与迁移路径，**实施前不要据此改生产行为**。

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
  *存储*的——node 存储 + 渲染时按 pack 拼块，模型看到的仍是大块。
- **不增加 API 往返**：立项面向免费额度，按往返计费。所有检索分级都在 harness 内完成（§4），
  模型侧仍是「预注入 + 查询轮」两跳以内。
- **模板提升主播词条质量**：固定槽位是「收集清单」，比自由生长好。自由度留在存储层，纪律放在
  preset 层（§3）。

## 2. 数据模型：node

### 2.1 节点

```text
node:
  id          ulid                       # 稳定身份，跨用户不变
  kind        subject | term | fact | event | relation
  head:                                  # 结构化，全部进检索索引
    surface   源语言名（term/subject 必填）
    reading   读音（假名/拼音）
    names     {zh, en, …}                # 各语言定名
    aliases   [昵称/简称/全称/俗称]
    misheard  [ASR 误听变体]             # ← 误听反查索引（§4.1）
  body        可选；按 section 分组的 prose 段落
  children    [{section, id}]            # 子节点按 section 归组
  links       [{rel: see_also | part_of, id}]   # 跨 pack 多归属、平级互指
  inject      whole | by-hit             # 注入策略（§4.3）；默认 subject=whole，其余=by-hit
  preset      可选，preset 名（§3）
  valid       {from, to}                 # 可选：版本号或日期
  provenance  {origin: task|user|ingest|shared, confidence: 1-9, sources: n, last_confirmed: date}
  events      append-only (kind, task_id, date)       # served / hit / landed / confirmed / refuted
```

- **「大的小词条」** = body 非空的 `term`（游戏角色的剧情介绍、事件线、梗的出处）。今天它们被
  硬塞在一条 5 段行里（如 `原神.md` 的 `シトラリ`）。
- **主播** = 一个 `subject` 节点（body：简介 + 说话风格整体描述）+ 按 preset 槽位归组的
  `fact`/`event`/`relation`/`term` 子节点（档案字段、经历、关系、口癖）。
- **common 游戏** = 一个 `subject` 节点（body 很薄）+ 大量 `term` 子节点，按 section 分组；
  section 可以再是一个 `subject`（`原神 ⊃ 原神·挪德卡莱`），多层分类由此免费得到。
- 一个 `term` 可以是多个 subject 的 child（`part_of` 多归属）——解决今天跨节重复（`ユプカ竜`、
  `クク竜` 在 `原神.md` 两节各出现一次，`append_lines` 只做节内去重）。

### 2.2 渲染（prompt 不变）

`render(node)` 产出今天的 markdown：

| 节点 | 渲染 |
| --- | --- |
| `subject` | H1 + 简介 + 按 preset/section 顺序的 `## 节`，空槽保留 |
| body 为空的 `term` | 5 段行 `源语言\|中文定名\|别名/缩写\|特殊读音\|一句话描述`，误听变体渲染进描述的「误听: …」 |
| body 非空的 `term` | `### name` 小节：首行 5 段行，其后 body |
| `fact` / `event` / `relation` | 各自行文法（`字段: 值` / `日期: 事件` / `对象 \| 描述`） |

`N| ` 行号渲染与 `edit_lines` 的行号快照语义**整体消失**：模型的修改提案改为按 node id
（§6.2）。index 文件也是派生物：每个顶层 subject 一行，格式沿用今天的四字段。

### 2.3 存储

- 每个顶层 subject 一个目录或一个 JSONL（一行一个 node，children 在同文件内）；`links` 跨文件只存
  id。JSONL 天然按行 diff / merge / append 幂等，不需要解析 markdown 回写。
- markdown 是**派生的只读渲染缓存**（给人看、给 prompt 用），不是编辑面。人工编辑 JSONL 行，或走
  薄 CLI（`knowledge node edit <id>`）。不采用「markdown 每行带 id 注释」：prompt 渲染要剥、人工
  编辑容易弄丢。
- 本地历史不再依赖内嵌 git：`changes.jsonl`（谁、何时、改了哪个 id 的哪个字段）+ 节点 `events`。
  目标用户机器上没有 git 是常态，现有 git 路径只在开发机有效。内嵌 git 在迁移完成后移除。
- 知识库根的解析顺序、跨进程写锁、worktree 告警等（`knowledge.md`）照旧。

## 3. preset：纪律放在这一层

数据模型「什么都放得下」；preset 回答「对某一类节点，什么该填、填成什么样」。preset 是数据
（`prompt_templates/` 下，主 git 跟踪），不是代码；新增一类 subject（企划/团体）加一个 preset。

### 3.1 streamer（强 preset）

```text
sections（固定顺序，禁止新建 section）:
  简介        body: prose，1 段，必填
  档案        children: fact，行文法 `字段: 值`；固定字段 本名/别名/人设/其他
  直播内容    children: fact
  说话风格    body: prose（整体描述）+ children: term（口癖，可带 misheard）
  喜好 / 特点 children: fact
  重要经历    children: event，行文法 `日期: 事件`，绝对日期、升序（validator）
  人际关系    children: relation，行文法 `对象 | 描述`，对象可 link 到另一 subject
```

- 主播 subject 的各方面是可枚举的，固定槽位 = 收集清单（没有「人际关系」槽模型不会主动去总结
  联动搭档）。
- 同槽对同槽，跨用户合并是槽级的；审核规则也写到槽级（§5.3）。
- prose 只留「简介」与「说话风格」整体描述两处——prose 的跨用户合并没有好办法（只能 LLM 合并或
  人工二选一），所以把 prose 压到最少，让 LLM 合并的爆炸半径限定在这两段。

### 3.2 common（弱 preset）

`档案` 固定；其余 section 自由命名、允许新建（持续更新的游戏按大版本开 section）；子节点默认
`term`。body 非空的 `term` 可按需挂轻 preset（游戏角色：所属/元素/武器/口癖）。

### 3.3 执行层三处

1. **apply 校验**：streamer 的新建 section 直接拒；行文法不合的 event/relation 跳过记 report
   （沿用非法 proposal 路径）；日期升序由 validator 在 apply 后重排而非拒绝。
2. **渲染按 preset 固定顺序**，空槽保留给模型看（今天的行为，保住）。
3. **审核/合并按槽位**（§5）。

## 4. 检索：全部 harness 侧，零往返

### 4.1 分级

1. **误听反查**：raw ASR 文本 ∩ 全库 `misheard ∪ aliases ∪ surface`。专为 ASR 纠错设计，优先级最高。
2. **音近匹配**（日语尤其有效）：假名归一化（片假→平假、长音、小字、浊半浊）后编辑距离，或罗马音
   n-gram。ASR 错误基本是音近错误，比 BM25 贴题。
3. **用户备注子串**：今天的 `match_index_keywords`，保留。
4. 可选：BM25 / 本地 embedding 对 `body` 与 `names`——给「主播聊到某事件但没提名字」的 fact 类
   用；收益最不确定，最后做，且不引入任何 API 调用。

### 4.2 命中 → 注入

命中节点 → 提升其所属 subject（沿 `part_of` 上溯）→ 注入整个 subject 的渲染块。保留「大块注入」。

### 4.3 预算裁剪按 `inject`

- `whole`（subject=streamer 默认）：要么整块进，要么整块不进；不截尾——人称与关系互相解释，
  缺一节就失真。
- `by-hit`（common 默认）：超预算时按 section 裁掉**未命中**的 section，再裁未命中的 `term`；
  不再盲目截尾，`RenderedBlock.truncated` 守卫与「被截断条目禁用 edit」随之消失。

### 4.4 与 pull 形态的关系

`llm_agent_tool_protocol.md` §7 的数据面迁移（索引必读 + 自主 query）暂缓。本节的检索引擎同时是
未来 query 工具的后端；渲染器不得写死成只有预注入一条路。命中扫描看的是文本而非注入方式，两种
形态下口径一致。

## 5. 信号：抓取 ≠ 相关 ≠ 准确

三层事件各自只能证明自己那一层；分数是派生值，不单独存储。

| 层 | 事件 | 能证明 | 不能证明 |
| --- | --- | --- | --- |
| 检索命中 `hit` | 误听/别名/子串在 raw 文本里触发 | 这条 alias/misheard 在真实素材里会被触发 | 触发得对不对 |
| 落地 `landed` | corrected 文本出现该节点定名，且 raw 对应位置不是它（确实发生了一次修正） | 模型用了这条改了东西 | 改得对不对（模型与词条可一起错） |
| 准确 `confirmed` / `refuted` | 精修 SRT 保留/推翻了这次修正；用户手动确认；独立用户收敛；研究轮外部检索印证 | 内容可信 / 不可信 | — |

### 5.1 推论

- **命中与落地只排序审核队列、不作通过依据**；通过靠第三层。主播类记录尤其。
- **命中事件的真正用处是反向的**：某 `misheard` 命中很多次但从未落地 → 高误触发别名，该从节点摘掉。
  这就是原方案「预注入噪声不可见」的缺口，不需要 LLM 打分。
- **`refined_aligned` 推翻的修正 = 最强负面证据**：同时喂 mistake 台账与该节点 `provenance`。今天
  该模式只产提案、不回写被推翻节点，这是现成缺口。
- 过时性用 `provenance.last_confirmed` 表达：主播节点一年没被任何任务印证，审核/报告侧标灰。
- **可见性边界**：`events`/派生分数永不进入纠错、查询、research 的 prompt。但**命中本身是客观事实**
  （不是模型评分），用它排注入优先级没有「低分→不被请求→永无翻身」的死循环，允许。
- LLM 事后打分只保留给扫描判不了的 `fact`/prose（可选、二元「落地/未落地」、盲评）；不做 0/1/2。

### 5.2 人工报告

`python -m finesub.llm.knowledge.report`（只读，不调模型）：按节点列 hit/landed/confirmed/refuted
计数、最近印证日期、从未命中标记；高误触发别名单独一节；`--subject` 过滤。

### 5.3 审核门槛按槽位（共享侧）

| 槽位/类型 | 通过要求 |
| --- | --- |
| common `term`（定名/别名/误听） | 独立来源 ≥ 2，或一次精修印证，或审核 LLM 外部检索印证 |
| streamer `档案` | 审核 LLM 外部检索印证（改本名/人设影响全库） |
| streamer `人际关系` / `重要经历` | 独立来源 ≥ 2 或精修印证 |
| prose（简介/说话风格） | 人工 |

主播信息视为公开（知识库本身是公网总结），**不设推送门槛**；门槛差异只在准确性。

## 6. 共享库

### 6.1 交换单位 = node

- **拉取**：按 subject 拉节点集；本地已有同 id → 按字段三路合并（base = 上次拉取版本）；字段级
  冲突才交给 LLM；新 id 直接进；`links` 指向本地不存在的 id 保留为悬挂引用。
- **推送**：用户勾选 subject / 节点；剥掉 `events` 明细与 task_id，只带聚合计数
  （served n / hit m / landed k / confirmed c）。
- **审核**：结构化节点比 markdown diff 好审。审核 LLM 只回答
  `approve | merge_into:<id> | reject:<reason>`，外加 §5.3 的槽级规则（validator 部分不经 LLM）。
  队列按「独立贡献者数 × 落地率」排序，人工只看前几条。
- **质量信号 = 跨用户聚合**：被 5 个用户注入 40 次、落地 0 次，比任何 EMA 有说服力。
- **身份合并**：两个用户各建了 `ナド・クライ` / `ナドクライ`——服务端按 head 归一化（NFKC +
  casefold + opencc t2s + 假名归一）提示 `merge_into`，由审核确认，合并后旧 id 写入
  `links: supersedes`，客户端拉取时跟随重定向。

### 6.2 proposal 契约随之改变

今天的六种 op（`append_lines` / `edit_lines` / `replace_section` / `create_entry` /
`delete_entry` / `rename_entry`）改为按 id 的节点操作：

```json
{"op":"create","kind":"term","parent":"<subject id>","section":"角色","head":{…},"body":"…","reason":"…"}
{"op":"update","id":"<id>","set":{"head.misheard":["…"],"body":"…"},"reason":"…"}
{"op":"move","id":"<id>","parent":"<subject id>","section":"…","reason":"…"}
{"op":"link","id":"<id>","rel":"see_also|part_of","target":"<id>","reason":"…"}
{"op":"retire","id":"<id>","merged_into":"<id>","reason":"…"}
```

- 渲染块里每个节点带其 id（`<!-- id -->` 不进 prompt；prompt 里用短 handle `@k12` 映射到 id，
  harness 映射回），模型永远不看到 ulid。
- 行号快照、`line_editable`、「被截断条目禁用 edit」整套机制删除。
- 需要 `PROMPT_VERSION` bump；resume 缓存按既有规则失效。

## 7. 迁移（一次性，无兼容负担）

导入器 `python -m finesub.llm.knowledge.migrate`：

1. 解析现有 `knowledge/{streamer,common}/*.md`：H1 → subject；`## 节` → section；5 段行 → `term`
   （行首字段 → surface，第 2 段 → names.zh，第 3 段 → aliases，第 4 段 → reading，描述里的
   「误听: a、b」→ misheard，其余 → body 首段或空）；streamer 的 `日期: 事件` → event、
   `对象 | 描述` → relation、`字段: 值` → fact。
2. 同 subject 内跨节重复（head 归一化相同）在导入时合并为一个节点、多 section 归属。
3. `translation/common-mistake.md` 与 `good-example.md` **不变**：它们是台账不是知识节点。
4. 导入后 `render()` 全库与原 markdown 做 diff 人工核对一次；此后 markdown 只读。
5. 内嵌 git 仓库保留为历史档案，不再写入；`unverified`/`main` 分支语义终止。

## 8. 实施顺序

1. **node 模型 + 渲染器 + 导入器**：目标是渲染输出与今天的 markdown 在 golden 测试下等价
   （`test_llm_prompt_compose.py` 快照不变），prompt 不动，可单独合入。
2. **误听反查 + 假名音近匹配**进预注入（§4.1 前两级）——对纠错质量最直接的提升，也是验证
   node 模型价值的第一站。需要 A/B（参照 `llm-ab-sampling-discipline`：臂间 n≥5 + 同配置复跑）。
3. **事件落账**（hit/landed 确定性扫描 + `refined_aligned` 回写 confirmed/refuted）+ `report` CLI。
4. **proposal 契约改按 id**（§6.2），PROMPT_VERSION bump，删除行号快照机制。
5. **共享库**（服务端节点存储 + 拉/推/审）——把 1–4 的结构搬到网络上。
6. 可选：BM25/embedding 第四级检索；LLM 对 prose 的二元盲评。

1–3 不动 prompt 契约（2 只改 harness 侧预注入集合），可分三次提交；4 是一次 prompt 版本；5 独立项目。

## 9. 明确不做

- 分数/事件回流到模型 prompt。
- 自动删除任何节点（`retire` 只来自提案 + 人工/审核确认）。
- markdown 作为编辑面（见 §2.3）。
- 在内嵌 git 之上建共享同步。
- 在 `llm_agent_tool_protocol.md` §7 数据面迁移落地前，把渲染器绑死到预注入形态。

## 10. 文档同步（实施时）

- `knowledge.md`：目录结构、条目结构、proposal schema、apply 层、局部检索匹配各节按本文重写；
  内嵌 git 一节改为「历史档案」。
- `llm_prompts.md`：preset 文件改为 schema 形式；新 proposal 块；PROMPT_VERSION。
- `llm_design_notes.md`：补记 node 模型、三层信号、preset 分层的决策与理由。
- `llm_harness_behavior.md`：预注入集合的新来源（误听反查/音近）；截断守卫的删除。
- `.claude/skills/run-audit/references/artifact-map.md`：新事件类型与 report 产物。
- `CLAUDE.md` / `README_DEV.md` 索引行。
