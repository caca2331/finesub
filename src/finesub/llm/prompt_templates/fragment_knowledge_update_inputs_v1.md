输入说明（字幕材料按纠错窗口分组，各窗口互不重叠）：
1. 每个 `--- window N ---` 包内依次是：
   - `<context_slice>`：背景调查对该窗口的摘要（可能为空）。
   - `<feedback_slice>`：纠错模型处理该窗口时现场提出的知识线索（`knowledge_hints`）、可复用 ASR 修正与存疑点（可能为空）。
   - `<raw_csv>`：该窗口的原始 ASR 字幕行，`id|start|duration|gap|text`（`id` 是源序号；`start` 是开始秒，时间为全局秒；`duration` 是时长；`gap` 是上一行结束到本行开始的秒数，首行留空）。
   - `<final_csv>`：机器最终成品行，`type|position|start|end|corrected|translation|conf|note`（`start`、`end` 是开始、结束秒）。`sub` 行的 `position` 是其合并的源序号，对应 `<raw_csv>` 的 `id`（一行可合并多行 raw）；`insert` 行是纠错模型补写的字幕，没有对应 raw 行；`translation` 与时间轴来自最终后处理字幕，`corrected` 保留纠错模型的原文修正。`conf` 是纠错模型对这一行的把握（`high` / `median` / `low`），`note` 是它写的简短说明（疑点、取舍）。$heard_bullet$refined_csv_bullet
2. 全局块：`<general_context>`（全局背景调查）、`<research_feedback>`（调查阶段的全局知识线索）、`<aggregated_feedback>`（各采集点存疑点与可复用修正的汇总）、`<kb_entries>`（按线索预取的知识库条目现有全文，条目间以 `--- 类别/key ---` 行分隔；H1 与已有内容行末尾的 `<!-- @k数字 -->`、档案里 `[别名]` 行的 `<!-- @a数字 -->` 是句柄，各能用于哪些 op 见下面的输出格式，都不属于内容）。
