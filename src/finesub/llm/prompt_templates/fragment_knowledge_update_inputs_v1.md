输入说明（字幕材料按纠错窗口分组，各窗口互不重叠）：
1. 每个 `--- window N ---` 包内依次是：
   - `<context_slice>`：背景调查对该窗口的摘要（可能为空）。
   - `<feedback_slice>`：纠错模型处理该窗口时现场提出的知识线索（`knowledge_hints`）、可复用 ASR 修正与存疑点（可能为空）。
   - `<raw_csv>`：该窗口的原始 ASR 字幕行，`源序号|开始秒|时长|gap|文本`（时间为全局秒）。
   - `<final_csv>`：机器最终成品行，`type|position|开始秒|结束秒|gap|corrected_text|translation|conf|char_count|note`。`sub` 行的 position 是其合并的源序号（一行可合并多行 raw）；`insert` 行是纠错模型补写的字幕，没有对应 raw 行；`translation` 与时间轴来自最终后处理字幕，`corrected_text` 保留纠错模型的原文修正，`char_count` 是独立的加权译文字数列。$refined_csv_bullet
2. 全局块：`<general_context>`（全局背景调查）、`<research_feedback>`（调查阶段的全局知识线索）、`<aggregated_feedback>`（各采集点存疑点与可复用修正的汇总）、`<kb_entries>`（按线索预取的知识库条目现有全文，条目间以 `--- 类别/key ---` 行分隔；H1 与每一行末尾的 `<!-- @k数字 -->` 是句柄、供 `update`/`remove`/`append_lines` 引用，不属于内容）。
