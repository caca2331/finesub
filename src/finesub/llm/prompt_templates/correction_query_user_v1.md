请根据 system 指令，先以一个 `<reasoning>` 块开头做中轻量分析，再提出需要联网查证的搜索 query；随后依次输出一个 `<window_notes>` 块$entry_block_list和一个 `<search_queries>` 块。

$knowledge_index_block通用背景知识和术语（来自背景调查，覆盖全部窗口）：
$general_context_json

本窗口专属背景（来自背景调查，按窗口对齐；可能为空）：
<window_context>
$window_context
</window_context>

$knowledge_carried_block此前所有窗口的累积建议（按窗口标注，可能为空）：
<previous_advice>
$previous_advice
</previous_advice>

本窗口 ASR：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头，随后依次输出 `<window_notes>`（可为空块）$entry_block_reminder、`<search_queries>`（可为空块）；不纠错、不翻译、不输出字幕内容，除上述块外不要输出任何其他文字。
