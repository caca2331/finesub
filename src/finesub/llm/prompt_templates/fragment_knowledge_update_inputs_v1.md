输入说明（字幕材料按纠错窗口分组，各窗口互不重叠）：
1. 每个 `--- window N ---` 包内依次是：
   - `<context_slice>`：背景调查对该窗口的摘要（可能为空）。
   - `<feedback_slice>`：纠错模型处理该窗口时现场提出的知识线索（`knowledge_hints`）、可复用 ASR 修正与存疑点（可能为空）。
   - `<window_csv>`：该窗口的字幕行，原始 ASR 与机器最终成品放在同一行，`type|position|start|end|raw|corrected|translation|conf|note`（`start`、`end` 是开始、结束秒，时间为全局秒）。`sub` 行是成品行：`position` 是它合并的源序号（一行可合并多个源），`raw` 是这些源的原始 ASR 文本，合并了多个源时每段前面标 `⟦源序号⟧`；`corrected` 是纠错模型的原文修正，`translation` 与时间轴来自最终后处理字幕。`insert` 行是纠错模型补写的字幕，`position` 是 `开始秒,时长`，没有 `raw`。`discard` 行是纠错丢掉的源，`position` 是它的源序号，只有 `raw` 与它自己的时间，没有成品。`conf` 是纠错模型对这一行的把握（`high` / `median` / `low`），`note` 是它写的简短说明（疑点、取舍）。下文的「成品」指 `corrected` 与 `translation`。$heard_bullet$refined_csv_bullet
2. 全局块：`<general_context>`（全局背景调查）、`<research_feedback>`（调查阶段的全局知识线索）、`<aggregated_feedback>`（各采集点存疑点与可复用修正的汇总）、`<kb_entries>`（按线索预取的知识库条目现有全文，条目间以 `--- 类别/key ---` 行分隔；H1 与已有内容行末尾的 `<!-- @k数字 -->`、档案里 `[别名]` 行的 `<!-- @a数字 -->` 是句柄，各能用于哪些 op 见下面的输出格式，都不属于内容）。
3. 误听只登记确定的修正。`conf` 为 `median` / `low` 的行、`note` 写着推测的修正，以及背景调查或反馈里标着〔候选〕「待定」之类的对应，都还只是推定，没有独立的证据（如精修字幕）印证就不写成误听。
