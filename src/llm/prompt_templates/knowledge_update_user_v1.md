请根据 system 指令，从以下任务材料中提炼知识库更新提案。

任务概要：
<task_summary>
$task_summary
</task_summary>

本次纠错使用的 prompt 版本（`add_mistake` 的 `prompt_version` 字段用它）：$task_prompt_version
$chunk_notice
知识库索引（新建词条前必须对照）：
<kb_index>
<streamer_index>
$streamer_index
</streamer_index>
<common_index>
$common_index
</common_index>
</kb_index>

全局背景调查（覆盖全部窗口，可能为空）：
<general_context>
$general_context
</general_context>

调查阶段的全局知识线索（可能为空）：
<research_feedback>
$research_feedback
</research_feedback>

各采集点存疑点与可复用 ASR 修正的汇总（可能为空）：
<aggregated_feedback>
$aggregated_feedback
</aggregated_feedback>

按线索预取的知识库条目现有全文（可能为空）：
<kb_entries>
$kb_entries
</kb_entries>

按纠错窗口分组的字幕材料：
$window_packs

最后提醒（读完以上全部输入后）：$final_reminder
