请根据 system 指令，先以一个 `<reasoning>` 块开头，再依次输出 <analysis_notes>$fast_entry_block_list，然后按搜索规则输出后续标签块。

用户提供的额外信息（可能为空）：
<extra_info>
$extra_info
</extra_info>

额外信息中 URL 的预提取内容（可能为空）：
<note_url_extracts>
$note_url_extracts
</note_url_extracts>

$fast_knowledge_inputs整段 ASR 类 CSV：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头，随后按 system 指令依次输出 `<analysis_notes>`$fast_entry_reminder 及搜索相关标签块$task_feedback_reminder；本轮只做分析与检索请求，不输出字幕；不能因输出较长省略必需标签，必须严格按要求格式完整输出；除上述块外不要输出任何其他文字。
