# 知识库词条质量评分机制 —— 设计方案

状态：设计定稿，待实施。本文是多轮讨论的收敛结果；打分语义、常数、对账机制均已与用户确认。
预计实施时 `PROMPT_VERSION` bump 至 `zh-subtitle-correction-csv-v17`（知识更新 prompt 新增输出块，属输出契约变化）。

> **注**：本文中所有 v17 版本引用为原始设计时的历史占位符。实际实施应 bump 至当前 `PROMPT_VERSION`（v63）之后的下一个版本。

## 1. 目标与动机

知识库目前只有写入闭环、没有使用反馈闭环，造成三个真实缺口：

1. **尺寸纪律无裁剪依据**：词条接近 3500 token 软上限时，「优先精简描述而不是删行」「拆分由人工确认」都缺少哪些内容从未帮上忙的证据。
2. **低价值词条无退出信号**：`delete_entry` 只服务「碎片并入」，从建档起就没用过的词条会永远留着。
3. **预注入噪声不可见**：按用户备注做别名子串匹配的预注入（`match_index_keywords`），哪些别名经常造成无关注入无从观测。

本机制让 post-task 知识更新模型对本次任务用过的词条/子词条打分，分数由 harness 侧维护，用于辅助模型与人工做裁剪、合并、删除决策。

**核心语义澄清**：打分度量的是「**落地痕迹**」（词条内容是否在本次任务的修正、译名、人称、语气处理中体现），不是因果帮助。因果不可观测（模型可能本来就会），但修剪场景需要的恰是「内容从未落地」这一信号，因果归因并非必需。打分规则的措辞按此语义撰写。

## 2. 术语与范围

- **词条**：`knowledge/{streamer,common}/*.md` 中的一个文件。
- **子词条**：词条正文中除 Markdown 标题行（`#`/`##`/`###`）、空行、`## 元数据` 节内容之外的所有行（H1 下的一句话描述行也算子词条）。元数据节由 harness 自动维护，不参与追踪。
- **served（被服务）**：一个词条在本次任务的任一环节被注入过 prompt——首轮按备注关键词预注入、research/fast round 1 的 `<requested_entries>` 兑现、纠错查询轮的词条请求兑现（含 text 路线逐窗注入的 `<entry_details>`）、search loop 的词条请求兑现。
- **注入状态**：`full`（完整注入）/ `truncated`（因预算被截断）/ `dropped`（因预算整体丢弃）。渲染时 harness 已知。
- **可见性边界（硬规则）**：分数只在 post-task 知识更新流程与人工工具中可见。**永不**进入纠错、查询、research、search loop 的 prompt，**永不**影响请求/注入决策——避免「低分→不被请求→永无翻身」的死循环。

## 3. 打分规则（模型侧）

### 3.1 词条级（必打）

- 范围：本次任务中注入状态为 `full` 的 served 词条**必须**打分；`truncated` 的**可以**打分（台账中标记 truncated）；`dropped` 的**不得**打分（模型没见过内容，打了也会被 harness 丢弃）。
- 量表：
  - `2`：内容在本次任务中有落地痕迹（定名/读音/误听模式/人物设定体现在修正或译文中）。
  - `1`：本次无落地痕迹，但内容对其他任务可能有用。
  - `0`：本次无落地痕迹，且内容低价值、对其他任务也大概率无用。
- **盲评**：打分请求所在的 prompt 中，served 词条不渲染任何历史分数注解。

### 3.2 子词条级（可选）

- 范围：仅限本次 `full` 注入的词条的行；`truncated` 词条禁止行级打分（与 `edit_lines` 的截断禁用同一逻辑——行号快照不完整）。
- 量表：`2`（该行内容明确落地）/ `0`（该行低价值、对其他任务也大概率无用）。无中间档。
- **稀疏输出**：只对有把握的行打分，其余不打（未打分的行由常数 b 做缓慢均值回归，见 §4）。这是刻意的：控制输出压力，避免逐行编造。
- 行引用方式：沿用 `<kb_entries>` 渲染中现成的 `N| ` 行号（与 `edit_lines` 同一快照语义）；harness 在渲染时建立 行号 → (section, dedup_token, 行 hash) 的映射。

### 3.3 输出块 `<entry_scores>`

知识更新输出新增一个块，作为最后一个块，位置在 `<knowledge_proposals>`（refined 模式在 `<mistake_proposals>`）之后。块内 JSONL，每行一个词条：

```json
{"category":"streamer","entry":"星野灯","score":2,"reason":"口癖『てぇてぇ』的固定译法在窗口3落地","lines":[{"line":12,"score":2},{"line":30,"score":0}]}
{"category":"common","entry":"原神","score":1,"reason":"注入了但本次视频不涉及"}
```

- `score` ∈ {0,1,2}；`lines[].score` ∈ {0,2}；`lines` 可省略。
- `reason` 建议填写（一句话，写明落地/未落地的证据位置），供台账审计；可省略。
- 解析失败或整块缺失：仅记录 warning，不重试、不影响主输出的 apply（与 `<task_update_feedback>` 同策略）。
- **多块（chunked）更新时只有第 1 块承担打分职责**：served 词条清单与打分规则 fragment 只注入首块，后续块不请求也不接受 `<entry_scores>`。

### 3.4 harness 侧校验（宽容丢弃，不拒绝重做）

- 词条不在本次 served 集合 → 整条丢弃并记录。
- `dropped` 状态词条的打分 → 丢弃。
- `truncated` 词条的 `lines` → 丢弃（词条分保留、标记 truncated）。
- 行号映射不到被追踪行（标题/空行/元数据节/越界）→ 该行条目丢弃。
- 同一词条重复出现 → 取第一条。
- 必打词条缺分 → 该词条本次不更新（视同未评估），记录 warning。

## 4. 分数模型与常数

### 4.1 词条级：warmup 均值→EMA 过渡 + 评估台账

```
初始：score = 1, n = 1（先验伪计数）, evals = []
每次评估（rating ∈ {0,1,2}）：
    n += 1
    score += (rating - score) / min(n, N)      # N = 8
    evals.append([task_id, rating, truncated?])  # 封顶保留最近 20 条
```

- n < N 时等价于普通累积均值（先验权重随样本增加自然稀释），n ≥ N 后渐近等价于 a≈0.875 的 EMA（兼顾近因性）。
- **选 N=8 的依据**：单人管线下一个词条的评估节奏约每月 1–4 次。原 a=0.9 方案连打 4 个 0 分后仍有 0.66，按月节奏要一年半才能确认死词条；N=8 + 先验下 4 个 0 分即降至 0.2，且 n 与台账在场、不会误杀。
- **台账是审计层**：EMA 分数是台账的缓存/展示值。分数支撑 delete/merge 提案时，人工可回溯每次评估对应的 task_id（进而查 exchange log 核实当次是否为无关 serve 造成的冤枉 0 分）；将来调 N 也可从台账重导出。
- 词条级不做未评估回归：没被 serve 的词条分数原地不动（niche 词条不因冷门受罚），过期性由 `last_eval` 日期表达。

### 4.2 子词条级：纯 EMA，三常数

| 场景 | 公式 | 常数 | 依据 |
| --- | --- | --- | --- |
| 被打分（rating ∈ {2,0}） | `score = a·score + (1-a)·rating` | **a = 0.7** | {2,0} 是强观点量表，单次打分应有可见位移（1→0.7 或 1→1.3）；无 n 做 warmup，0.9 的单次位移仅 0.1，配合 b 回归形同白打。三个 0 → 0.34。 |
| 所在词条本次 `full` 注入但该行未被打分 | `score = b·score + (1-b)·1` | **b = 0.98** | 刻意贴近 1：死行恰恰是不会再被打分的行（模型不再注意它们），b 是作用其上的唯一力量，回归快 = 销毁仅有的负面证据。0.98 下偏差半衰期约 34 次 serve（活跃词条约一年）；0.99 形同永不遗忘，0.95 几个月就洗白死行。误判 0 分的平反路径是下次真正落地时挨一个 2（0.34 → 0.84，两次回到中性以上）。 |
| 行内容发生实质变化（对账中 token 同、hash 异；每次变化触发一次） | `score = c·score + (1-c)·1` | **c = 0.5** | 内容改写后旧证据大半失效；0.8 保留太多（0.3 分死行改写后仅回到 0.44，仍背旧罪名），0.5 回到 0.65——接近重新做行但保留一点历史怀疑。 |

- 子词条初始 1 分；不追踪评估次数、不留台账（量大、且不直接支撑删除决策，有损压缩可接受）。
- 常数将来调整时直接改值、任分数自行回归（用户已确认接受）。

## 5. 存储：`knowledge/.meta/` 平行树

每词条一个 JSON：`knowledge/.meta/<category>/<key>.json`。选平行树而非同目录 `.meta.json`：现有注入、index 重建、词条枚举全部按 `*.md` glob，平行树保证零误伤。随嵌入式 git 一起自动提交。

```json
{
  "schema": 1,
  "file_hash": "sha1 of the entry .md content",
  "entry": {
    "score": 1.38,
    "n": 6,
    "last_eval": "2026-07-12",
    "evals": [["task-20260701-siyueyi", 2, false], ["task-20260712-xxx", 1, false]]
  },
  "lines": [
    {"section": "直播内容", "token": "常玩游戏", "hash": "sha1(整行)", "score": 1.12},
    {"section": "5.0 版本/人名", "token": "ブレンニ", "hash": "…", "score": 1.30}
  ]
}
```

- `token` 复用 `base.py` 现成的 `_line_dedup_token`（append 查重的行首字段），key 为 `(section, token)` 以降低跨节碰撞；同节同 token 的重复行按文件顺序对应。
- serve 统计（可选字段）：`served: {count, last_date}`——harness 记录 serve 事件时顺带累加，为 `--usage-report` 提供「从未被 serve」维度。不参与任何公式。

## 6. 对账机制（初始化 + 错位修复，统一路径）

**不做逐 op 钩子**。任何读写分数的场合先比对 `file_hash`，不一致即触发对账；apply 提交后、update 材料组装时、报告生成时都走同一过程。机器改（六种 op）与用户手改 .md 由此走完全相同的路径。

对账规则（幂等、确定性）：

1. 词条无 .meta → 建档：entry 初始 (score=1, n=1, evals=[])；逐行按 `(section, token)` 初始化 1 分。
2. `(section, token)` 在文件中、不在 .meta → 新行，初始 1 分。
3. 在 .meta、不在文件中 → 行已删，条目丢弃。
4. token 同、行 hash 同 → 保留原分。
5. token 同、行 hash 异 → 内容被改，触发一次 c 回拉，更新 hash。
6. 对账完成后更新 `file_hash`。

需要显式处理的只有跨文件身份（对账看不到）：

- `rename_entry` → .meta 文件跟随改名（apply 层 Phase E 附带动作）。
- `delete_entry` → .meta 文件删除。
- 孤儿 .meta（对应 .md 不存在，例如用户手动删词条）→ 对账扫描时删除并记录。

## 7. 管道改动

### 7.1 served 清单的采集与聚合

- **采集**：在每个词条渲染注入点（research round 1 预注入、round 2 entry_details、fast round 1、查询轮→纠错轮 entry_details、text 路线逐窗注入、search loop 词条注入）记录事件 `(round, category, key, status)`，落入 `task-artifacts.jsonl`（新事件类型 `kb_entry_served`）。渲染函数本来就知道 full/truncated/dropped 状态。
- **聚合**：知识更新（`llm/knowledge/materials.py` 组装材料时）从 artifacts 读取全部事件，按 (category, key) 归并；状态取最优（任一次 full 即 full；否则 truncated；全 dropped 即 dropped）。窗口重试造成的重复 serve 自然去重。
- 独立运行的 `python -m llm.knowledge.update <final.srt>` 从同一 artifacts 目录读取，无需额外状态。找不到 served 事件（旧任务产物）→ 跳过打分流程，只做原有更新（向后无负担，旧产物重跑即可）。

### 7.2 知识更新 prompt 改动

- **user 侧**新增一个块（仅首块注入）：

  ```
  本次任务实际注入过的知识库词条（打分范围；标注注入状态）：
  <scored_entries>
  --- streamer/星野灯 [full] ---
  1| # 星野灯
  2| …（与 <kb_entries> 同一渲染格式，行号供行级打分引用）
  --- common/原神 [truncated] ---
  …
  </scored_entries>
  ```

  与 `<kb_entries>`（hints 预取）的关系：两者集合不同（前者按 serve、后者按 hints 频率）。同一词条同时出现时只在 `<scored_entries>` 渲染全文，`<kb_entries>` 中以一行指针代替，避免双份 token。served 词条通常 ≤10 个，预算风险低；超预算时按「必打词条优先完整渲染」截断。
- **system 侧**新增 `fragment_entry_scores_v1.md`：打分语义（落地痕迹）、量表、必打范围、行级稀疏原则、`<entry_scores>` schema 与校验规则；由 `build_knowledge_update_messages` 在有 served 清单时拼入（refined / artifacts_only 共用）。
- **盲评实现**：`<scored_entries>` 渲染不带任何分数注解。**分数注解只渲染在 `<kb_entries>` 中「本次未 served」的预取词条**上（形如 `--- streamer/某某 [评估 8 次, 均值 0.4, 最近评估 2026-05-02] ---`），供模型在提 delete/merge 时援引；served 词条的清理决策留给人工报告或后续任务。
- user 模板「最后提醒」追加 `<entry_scores>` 的输出提醒（仅首块变体）。

### 7.3 apply 层

顺序：原有 proposals apply → commit → 解析 `<entry_scores>` → 校验（§3.4）→ 逐词条更新 .meta（词条公式 + 被打分行 a / 未打分行 b）→ 对账兜底 → 随同一 task 的 git commit 落库（或紧随其后的独立 `scores` commit，实施时取顺手者）。

注意顺序细节：本次 update 自己的 `delete_entry`/`rename_entry` 先于打分 apply——被删词条的分数随 .meta 删除（该打分丢弃并记录）；改名词条的打分按新 key 落账。

### 7.4 消费面

1. **更新 prompt 注解**（§7.2，未 served 词条）+ 更新原则新增一条：「评估次数 ≥5 且均值 <0.5 的词条、或长期 <0.5 的行，可在 reason 中援引统计提出合并/`delete_entry`/删行；删除词条仍受既有守卫与人工确认约束」。
2. **人工报告**：`python -m llm.knowledge.update --usage-report`（只读，不调模型）：按均值升序列出词条（均值/n/最近评估/最近 serve/从未 serve 标记），词条内列出低分行；支持 `--category` 过滤。
3. 明确**不做**的消费：分数不进入请求/注入排序，不自动删除任何内容。

## 8. 边界情况

| 情况 | 处理 |
| --- | --- |
| fast 模式（单窗口） | round 1 预注入 + requested 同样记 served 事件，无特殊逻辑 |
| text 路线（无查询轮，预注入直接进每窗 entry_details） | 每窗注入记事件，聚合去重 |
| 窗口重试 / -a/-b 拆分 | serve 事件重复，聚合层去重 |
| research context 复用（跳过调查轮重跑纠错） | 复用产生的注入照常记录于本次任务 artifacts |
| 任务中途失败、update 未跑 | 无打分发生；serve 统计字段可选累加（实施时若嫌麻烦可只在 update 成功路径累加） |
| 同一词条 full 与 truncated 各注入一次 | 状态取 full，行级打分允许 |
| update 模型对未 served 词条打分 | 丢弃（防拿 hints 预取词条凑数） |
| .md 被用户手改后行序/内容错位 | §6 对账按 (section, token, hash) 修复；改动行触发 c 回拉 |
| .meta 损坏/schema 不符 | 视同不存在，重新建档（分数丢失可接受——台账在 git 历史里仍可考古） |
| 旧任务产物（无 serve 事件） | 跳过打分，不报错 |

## 9. 测试计划

- 公式单测：词条 warmup→EMA 过渡（n<8 / n≥8）、台账封顶、truncated 标记；子词条 a/b/c 三路径。
- 对账单测：六条规则各一例 + 孤儿 .meta + 手改错位（token 同 hash 异）+ rename/delete 跟随。
- 解析/校验单测：`<entry_scores>` JSONL 宽容解析、§3.4 全部丢弃路径、多块只认首块。
- serve 采集/聚合：各注入点事件、重试去重、状态归并。
- prompt 组装：新 fragment 拼入两种模式、`<scored_entries>` 与 `<kb_entries>` 去重、盲评（served 词条无注解）、未 served 注解渲染、最后提醒更新；`test_llm_prompt_compose.py` 版本断言 → v17。
- `--usage-report` 冒烟。

## 10. 文档同步

- `docs/knowledge.md`：新章节「词条质量评分」（语义、公式、.meta、对账、消费、可见性边界）。
- `docs/llm_prompts.md`：fragment 清单 + `<entry_scores>` 输出块；PROMPT_VERSION v17。
- `docs/llm_design_notes.md`：设计决策补记（落地痕迹语义、盲评与消费分离、常数依据、台账 vs EMA 的取舍）。
- `.claude/skills/run-audit/references/artifact-map.md`：新增 `kb_entry_served` 事件与 `<entry_scores>` 产物说明。

## 11. 实施顺序与体量

1. `.meta` 读写 + 公式 + 对账（`knowledge/base.py` 或新 `knowledge/scores.py`；纯本地，先行落地，含 §9 前两组测试）。
2. serve 事件采集（各注入点）+ 聚合（materials）。
3. prompt：新 fragment、`<scored_entries>` 渲染、盲评/注解、最后提醒；PROMPT_VERSION → v17。
4. apply 层接线 + `<entry_scores>` 解析校验。
5. `--usage-report` CLI。
6. 文档 + 全量测试 + `session_replay --dry-run` 核查。

体量估计：约为 v16 批次的一半到同量级（新增一个独立模块 + 多点采集管道）；1、2 两步不动 prompt、可单独提交（无版本 bump），3–5 一起 bump v17。

## 12. 明确不纳入本方案

- 术语行确定性命中扫描（harness 直接检索五段行词项是否出现在 raw/final 中）——可作为后续零成本增强，与本机制正交。
- 专门的「维护模式 update session」（渲染全量注解、只做清理不打分）——先靠 `--usage-report` + 人工，观察需求再定。
- 分数参与注入预算排序或请求决策——违反可见性边界，永久排除在本方案之外。
