请根据 system 指令完成第一轮背景调查：先做中轻量分析，再 request 尚未注入的词条、keep 仍需使用的预注入词条，并提出值得联网查证的搜索 query；按 system 指令输出 `<analysis_notes>`、`<requested_entries>`、`<keep_entries>` 及搜索相关标签块。

用户提供的额外信息（可能为空）：
<extra_info>
$extra_info
</extra_info>

用户备注中的 URL 经本地深度提取后的页面内容（去重后最多 8 个 URL；可能为空）：
<note_url_extracts>
$note_url_extracts
</note_url_extracts>

主播知识库索引：
<streamer_index>
$streamer_index
</streamer_index>

Common 知识库索引：
<common_index>
$common_index
</common_index>

根据用户备注中出现的关键词，由 harness 预注入的知识库条目全文（可能为空；若后续仍需其内容，请把对应 key 写入 `<keep_entries>`，不要重复 request）：
<preinjected_entries>
$preinjected_entries
</preinjected_entries>

带窗口标记的 ASR 字幕文本：
<transcript>
$transcript
</transcript>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头，随后按 system 指令依次输出 `<analysis_notes>`、`<requested_entries>`、`<keep_entries>` 及搜索相关标签块；本轮没有联网结果，未经证实的判断必须标注「待定」；不能因输出较长省略必需标签，必须严格按要求格式完整输出；除上述块外不要输出任何其他文字。
