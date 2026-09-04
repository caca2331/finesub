# `docs/` 整理方案（已拍板，待执行）

> 状态：**五步已全部执行（2026-09-03）**。本文此后是**取舍依据的记录**，不是待办；
> 现行的文档组织以 `docs/README.md` 的地图与 `CLAUDE.md` 的索引为准。
> 三个待定问题的答案在第 5 节，第 3 步按其中的裁决改写过——**「瘦身并改名」那套说法已作废**。
>
> 执行结果（顺序即复审建议、owner 采纳的那个）：
>
> | 步 | 做了什么 | commit |
> | --- | --- | --- |
> | 2 | 两处标签订正；conversational 的三条未决并入 `llm_followups.md` | `b4231d7e` |
> | 1 | 拆出 `batch-scheduler.md`（整节 152 行逐字搬出），4 处「会指错」的入链改指 | `75c6448f` |
> | 4 | 抽出 `bench-discipline.md`，`bench-baselines.md` 节号从「二」起、其余不动 | `0bd3861c` |
> | 5+3 | 建 `plans/` 搬 7 份；新建薄的 `speech-followups.md`；两条验收手法提升进 `testing.md` | 本次 |
>
> 与计划的两处出入，都记在对应步骤里：第 2 步搬的是**三条**不是四条（第四条标着已修，
> 是记录不是待办），且搬运时差点丢掉「两个时钟」的唯一一次实测印证；第 5 步实际搬了
> **7 份**（`docs-reorg-plan.md` 自己也在内）。

## 1. 盘点：动手之前是什么样

> 下表是 **2026-09-03 执行前**的快照，保留原样作为「改了什么」的对照。执行后：根目录 39、
> `plans/` 7、`manual/` 13，另新增 `batch-scheduler.md`、`bench-discipline.md`、
> `speech-followups.md`。现行清单以 `docs/README.md` 的地图为准。


| 目录 | 份数 | 体量 | 性质 |
| --- | ---: | ---: | --- |
| `docs/` 根 | 42 | 21200 行 / 1.8 MB | 契约 + 实验记录 + 计划，**混住** |
| `docs/manual/` | 13 | 156 KB | 用户向（2026-09-02 刚补过三章） |
| `docs/archive/` | 34 | 1.0 MB | 已归档，`publish-main.ps1` 从公开快照剥离 |
| `docs/report/` | 17 | 464 KB | 调研报告，同上 |

先说**健康的部分**，免得整理变成乱砍：

- **索引是完整的**：`docs/README.md` 的地图覆盖全部 42 + 13 份，没有孤儿文件，也没有指向已删
  文件的死行；`CLAUDE.md` 的索引同样全覆盖。最低入链数是 4，没有没人引用的文档。
- **已死的设计段落都挂了牌子**：`wt-parallelism.md` 开头、`gpu-profiles.md` 第 240 行附近、
  `knowledge.md` 的 `mistakes.py` 那句，都是「本节描述的设计已经不存在，但下面的实测仍有效」
  的显式标注。这是有意保留，不是烂账——**不要因为「提到了已删模块」就删段落**。
- 所以本方案里**没有「删除内容」这一项**。要动的是**体裁分离**与**标签订正**。

## 2. 问题与证据

### 2.1 一个文件里住着两种体裁（最该修）

| 文档 | 体量 | 问题 |
| --- | ---: | --- |
| `wt-parallelism.md` | 42 KB | 标题写着「已移除」，但「与 batch 模式的分工」一节是 `scheduler.py` 的**现行契约**（三 bin、llm bin 准入、队列/控制面、intake 三态、失败隔离）。而 `CLAUDE.md` 的架构表正是把 `scheduler.py` 的 owner 文档指向这里——**一个活模块的 owner 文档标着「历史」** |
| `crispasr-followups.md` | 107 KB | 状态总览 30 行里 **24 行已完成或结案**，逐项证据段落在 `bench-baselines.md` 里已按节号索引过一遍，这里是第二份 |
| `bench-baselines.md` | 219 KB / 3593 行 / 24 节 | 第一节「测量纪律」是动性能前人人要读的**稳定规范**，其余 23 节是按时间堆积的实验流水账（其中一节就 1000 行） |

### 2.2 地图标签与实际不符

| 文档 | 地图标 | 实际 |
| --- | --- | --- |
| `knowledge-node-plan.md` | 设计稿 | 第 8、11 节**全部落地**（`CLAUDE.md` 已在警告「别当未竟计划」，是地图没跟上） |
| `crispasr-followups.md` | 计划 | 状态总览 30 行里 24 行已结案（改「台账」，正文不动——第 5.1 节） |

### 2.3 「唯一入口」漏了活项

- `llm_followups.md` 自称 LLM 未完成工作的唯一入口，但 `conversational-live-test-plan.md`
  最后一节的 4 条未决只在那份计划里，其中一条是**待查 bug**（replay 路径上
  `local_agent_timeout_seconds` 没被带进队列：配置写 3600，跑起来拿到 900）。
- **speech 侧没有对应的唯一入口**：未做项散在 `crispasr-followups.md` 的状态总览、
  `bench-baselines.md` 的预注册门槛、`asr-align.md` 的「待标定」、`separator-optimization.md`。

### 2.4 分类：根目录 42 份平铺

契约（约 22 份）、实验记录（约 6 份）、已结案的计划（约 6 份）挤在同一层，靠文件名区分。

## 3. 五步方案

编号按「收益 ÷ 风险」排，**执行顺序不是这个**——见文首（第 2 步先做）。每步可独立提交、
独立回滚。

### 3.1 第一步：拆出 `docs/batch-scheduler.md`

把 `wt-parallelism.md` 的「与 batch 模式的分工」整节（含 llm bin 并发语义、运行期队列面、
intake 三态、追加 manifest 的退出判据）搬进新文档，作 `scheduler.py` / `batch_state.py` 的
owner 文档；`wt-parallelism.md` 只留单文件分片的历史与仍成立的三条结论（stdio 背压、
intra-op 线程预算、语义分组边界）。同步改 `CLAUDE.md` 架构表的 owner 指向、`docs/README.md`
的两行、以及 `manual/batch.md` 的 dev 侧指路。

- **成本**：低。入链 11 处（排除 `archive/`、`report/`）。
- ⚠ **风险不是「链接会断」，是「链接会指错」**（2026-09-03 复审补）。11 处里有 **4 处说的就是
  batch 内容**——`CLAUDE.md` 架构表的 owner 列、`knowledge.md` 的失败隔离那句、
  `llm_harness_behavior.md` 的「llm bin 的并发语义」、`docs/README.md` 的地图行。搬走之后它们
  照样解析得到，只是指向一份不再讲那件事的文档，**`test_doc_links` 抓不到**。
  所以搬之前先按「指的是哪一节」把入链分成两类：这 4 处直接改指新文档，旧文档只留一行兜底指针。
- **另一处要看一眼、但结论是不动**（2026-09-03 复审补）：`vad-asr.md` 第 41 行引
  `wt-parallelism.md` 是为了「asr 恒为 1 的理由」，那段在**「Worker 数量」**一节，不在要搬走的
  「与 batch 模式的分工」里——确认它留在原文档即可，**不必改指**。分类入链时按这个方式逐条问
  「它指的是哪一节」，而不是按文件名一刀切。
- **其余风险**：搬运时漏段。**验收**：搬走的小节标题在新文档里逐字保留（按标题引的地方只改
  路径）；`test_doc_links` 全绿；人工核对上面那 4 处。

### 3.2 第二步：订正标签 + 收拢 conversational 的活项

- 地图两行改状态（第 2.2 节）。
- `conversational-live-test-plan.md` 最后一节的 4 条搬进 `llm_followups.md` 的 Agent 分节，
  原处留指针。那条 timeout bug 单独成一行，写清触发条件。
- **成本**：半小时。**风险**：几乎没有。

### 3.3 第三步：新建一份薄的 `docs/speech-followups.md`（不动 crispasr）

**owner 2026-09-03 采纳复审的选法：不瘦身、不改名、不归档。** 原方案（把
`crispasr-followups.md` 瘦成台账、逐项实测叙述移进 `archive/`、顺带改名）作废，理由见第 5 节。

改为新建一份薄文档，只放四样东西：

1. **六个未做项**，每项一行：一句现状 + 卡在哪 + 指回 `crispasr-followups.md` 的行。
   A6 的 CT2 wheel、P16 文本侧 LID、分离器耳语救援接线、P11–P15 diarization 代码、
   2026-09-02 新增的「第二模型否决只问证据非空」，加 **A1**（组批本体已做但门槛未过、
   默认仍关，按「还没结案」算）。
   ⚠ **是六个不是「4+1」**——`CLAUDE.md` 原来也写着「四项」，2026-09-03 已一并订正。
   ⚠ **执行时又多出一个：七项。** 拿「交付了但默认关着、翻默认还欠证据」这条判据去对状态
   总览，P9（`--asr-context`，差一次逐段听审）与 A1 是同一形状，本计划和复审都漏了它。
   落地的 `speech-followups.md` 因此分两组写：五项没开工 + 两项默认关着。
2. **各批的「不要做什么」**：只留一句话摘要 + 指针，展开仍在 `crispasr-followups.md`。
3. **散在别处的 speech 未做项的指针**：`bench-baselines.md` 的预注册门槛、
   `asr-align.md`「语言票翻转重解」的待标定（阈值未标定、负例为零）、
   `separator-optimization.md` 的未采纳项。
4. 一句**真相源声明**（见下）。

它与 `llm_followups.md` 对称：那份是 LLM 侧的唯一入口，这份是 speech 侧的。

⚠ **两份文档的一致性靠一条约定，不靠人记性**——这是这个选法唯一的代价，必须写死：
**`crispasr-followups.md` 的「状态总览」表是状态的唯一真相源**，
`speech-followups.md` 是它的索引。改状态时**先改厚的那份**，再同步薄的；
薄文档里不写第二份证据、不写第二份判据，只写「还没做的是这些 + 去哪读」。

- **成本**：低。新建一份文档，不动任何既有文件的正文。
- **风险**：**零信息损失**（原方案的归档风险随裁决消失），剩下的只有两份文档漂移，由上面那条
  约定与「薄文档不承载证据」的形状压住。
- **为什么值得多这一份**：改名的代价是碰 27 个文件、其中 19 个是代码（`src/` 5、`test/` 2、
  `tools/bench/` 12 处注释引用它，有
  `test_every_section_reference_in_source_points_at_a_real_heading` 守着，改名会红），
  而且撞上 `CLAUDE.md` 的「`tools/` 只按需维护、不要作为其他改动的副作用去更新」。
  用一份薄文档换掉这些，划算。
- **验收**：六个未做项与七条「不做」的措辞在两份文档里都找得到；
  **`git diff -- docs/plans/crispasr-followups.md` 为空**（地图标签改的是 `docs/README.md`，不是它）。

### 3.4 第四步：抽出 `docs/bench-discipline.md`

把第一节「测量纪律」与本机基线表抽成独立文档（短、常读、动性能前的必读），
`bench-baselines.md` 保留为实验流水账。

- ⚠ **二~二十四节的编号一律不动**：全仓有 **24 处**中文数字引用——13 处在 md 里，
  **11 处在源码注释里**（`src/` 8、`test/` 1、`tools/bench/` 2，多数写成不带反引号的
  `docs/bench-baselines.md 二十二`）。**空档已经补上**（2026-09-03）：
  `test_doc_links` 的 `test_every_chinese_numeral_section_reference_lands` 两个面都扫，
  重编号会红而不是悄悄指错。规则不变——**仍然别重编号**，只是不再只靠人记得。
  ⚠ 第一版守卫只扫 md、且要求反引号，那 11 处源码引用全在扫描面之外
  （`refactor-followups.md` 第五条教训的又一次同形）；现在复用 `§N` 源码守卫的同一份文件集。
- **成本**：低（只抽第一节）。**验收**：`test_doc_links` 全绿。**它覆盖的是**「引用里写出了
  文档名」的那些；没写文档名的裸章节号（源码里常见）任何守卫都认不出来，那一类仍要人工看。

### 3.5 第五步：建 `docs/plans/`，搬 6 份已结案的计划

搬这六份：`stage-device-plan.md`、`translation-style-plan.md`、`knowledge-node-plan.md`、
`conversational-live-test-plan.md`、`refactor-followups.md`、**`crispasr-followups.md`**
（第 3 步裁决之后它原样保留，正好整份搬走）。根目录 42 → 37（`docs-reorg-plan.md` 自己也搬，
第 3 步新建的 `speech-followups.md` 留在根）。

**分界线**（这是新目录的判据，不是文件名手感）：**根目录放契约与活台账，`plans/` 放已结案的
计划稿。** `llm_followups.md` 与新的 `speech-followups.md` 是活台账——「还没做的是哪些」——
所以留在根；`refactor-followups.md` 名字像台账，但它记的是**押后不做的理由与教训**，属于结案
计划，进 `plans/`。

- **成本**：实测约 45 处链接改写（16 处入链 + 29 处外链），全部由 `test_doc_links` 兜底。
- ⚠ **裸路径提及不是链接，守卫不管**（2026-09-03 复审补）：`CLAUDE.md` 的索引、`README_DEV.md`、
  `agent-tasks/` 里有约 11 处直接写 `docs/<名字>.md` 而不是 markdown 链接。搬完要人工
  `grep -rn "docs/<名字>.md"` 一遍。
- 六份文档在 `docs/README.md` 地图与 `CLAUDE.md` 索引里的行都要改路径——
  `test_every_tracked_doc_appears_in_both_indexes` 会盯着，漏一份就红。
- ⚠ **`docs/plans/` 必须保持公开**：不能塞进 `archive/`，那会被 `publish-main.ps1` 剥掉，而
  这些计划是「为什么不做 X」的唯一出处。新目录不需要动 `$PrivatePaths`。
- **提升只剩一处**（`CLAUDE.md` 的 Archive extraction 规则原本针对归档；`plans/` 是公开且进
  索引的，所以「唯一副本」不再是问题——`translation-style-plan.md` 的 owner 决定台账原地不动
  即可）。仍要做的那处是**操作性规则**，与搬不搬无关：`stage-device-plan.md` 第 6 节的两条
  验收手法（进程内抬高 torch arch 下界劈开两个后端、`CUDA_VISIBLE_DEVICES=-1` 无卡回归）
  应落到 `testing.md`——「动这一族代码要跑无卡那次」是一条测试纪律，不该让它的家是一份计划稿。
  落好之后 `CLAUDE.md` 里那句改指 `testing.md`。

## 4. 明确不做

- **全面子目录化**（`docs/speech/` `docs/llm/` `docs/dist/`）。全仓有 **约 1100 处 `§N` 引用**
  （排除 `archive/`、`report/` 后；含它们 2016 处，其中 **523 处在 `.py` 源码注释里**）、
  相对链接密集，而 memory 里记着改名类改动的静默失效教训（`doc-guard-lessons`）。
  收益是「根目录可扫」，价格是一次高风险的全量改写——不值。第 5 步只搬 6 份是它的廉价版。
- **给任何文档重新编号**。理由同上，且中文数字编号的引用没有守卫。
- **删除已死设计的段落**。它们的实测面仍然有效，且都已显式标注（第 1 节）。
- **动 `docs/report/`**。它是调研原件，本来就不进索引、不随仓库发布。
- **本轮不往 `archive/` 移任何东西**（第 5.2 节）。第 3 步改成新建薄文档之后，没有段落需要
  移出去；将来真要瘦身时的判据也写在那一节。

## 5. 三个问题的答案（owner 2026-09-03）

### 5.1 Q2 —— **不改名、不瘦身，新建一份薄的**（owner 选「后者」）

`crispasr-followups.md` 原样保留，只把地图标签改成「台账」；speech 侧的唯一入口由新建的
`speech-followups.md` 承担（形状见第 3.3 节）。理由：改名要碰 27 个文件、其中 19 个是代码，
而收益只是名字更贴切。代价是多一份要维持一致的文档——用「厚的是真相源、薄的只放指针」这条
约定压住。

### 5.2 Q1 —— **本轮不归档任何段落**（owner 让我判，我的判断与理由）

Q2 定了不瘦身，归档面这个问题本身就消失了：没有段落要移出去。所以本轮的答案是**零归档**。

留给将来真要瘦身时的判据，我选**机械可核**的那条（复审的建议，我同意）：
**只归档在 `bench-baselines.md` 里逐节号有对应的实测叙述**。不用「有没有第二份副本」——
那要通读全文才能判，判错的方式是静默的（把唯一副本挪进公开快照剥掉的目录），而
「节号对得上」可以脚本核、错了会当场看出来。⚠ 判据只覆盖**证据**；**依据**（为什么不做 X、
每批的「不要做什么」）一律不归档，无论有没有第二份副本——它们正是别人来读这份文档的原因。

### 5.3 Q3 —— **做**（owner）

建 `docs/plans/`，搬六份，分界线见第 3.5 节。放在最后执行，与第 3 步合成一次提交。

## 6. 验收（每一步都要过）

- `python -m pytest test/test_doc_links.py -q`——六条守卫：相对链接、`§N` 落点、
  **中文数字节号落点**（2026-09-03 新增）、两份索引的全覆盖
  （`test_every_tracked_doc_appears_in_both_indexes`）、源码注释里的 `§N`
  （`test_every_section_reference_in_source_points_at_a_real_heading`——改文档名会在这里红）。
- 守卫抓不到、必须人工核的四样：**入链指的是哪一节**（第 3.1 节）、**裸路径提及**
  （第 3.5 节）、**搬走的段落有没有漏**、**两份 followups 有没有漂移**（第 3.3 节：
  `crispasr-followups.md` 的正文在第 3 步里应当逐字未改）。
- 搬走的小节标题逐字保留，按标题引用的地方只改路径。
- 用户向行为无变化，因此**不写 `CHANGELOG.md`**（`docs/manual/` 不在本方案范围内）。
