请根据 system 指令，先以一个 `<reasoning>` 块开头，再依次输出 <analysis_notes>、<requested_entries>、<keep_entries>，然后按搜索规则输出后续标签块。

用户提供的额外信息（可能为空）：
<extra_info>
$extra_info
</extra_info>

额外信息中 URL 的预提取内容（可能为空）：
<note_url_extracts>
$note_url_extracts
</note_url_extracts>

主播知识库索引：
<streamer_index>
$streamer_index
</streamer_index>

common 知识库索引：
<common_index>
$common_index
</common_index>

根据用户备注关键词由 harness 预注入的知识库条目全文（可能为空；若后续仍需其内容，请把对应 key 写入 `<keep_entries>`，不要重复 request）：
<preinjected_entries>
$preinjected_entries
</preinjected_entries>

整段 ASR 类 CSV：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头，随后按 system 指令依次输出 `<analysis_notes>`、`<requested_entries>`、`<keep_entries>` 及搜索相关标签块$task_feedback_reminder；本轮只做分析与检索请求，不输出字幕；不能因输出较长省略必需标签，必须严格按要求格式完整输出；除上述块外不要输出任何其他文字。
