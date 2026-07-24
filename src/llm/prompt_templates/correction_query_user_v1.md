请根据 system 指令，先以一个 `<reasoning>` 块开头做中轻量分析，再提出需要联网查证的搜索 query；随后依次输出一个 `<window_notes>` 块、可选的 `<requested_entries>` 块、一个 `<keep_entries>` 块和一个 `<search_queries>` 块。

主播知识库索引：
<streamer_index>
$streamer_index
</streamer_index>

common 知识库索引：
<common_index>
$common_index
</common_index>

通用背景知识和术语（来自背景调查，覆盖全部窗口）：
$general_context_json

本窗口专属背景（来自背景调查，按窗口对齐；可能为空）：
<window_context>
$window_context
</window_context>

已透传词条（此前环节确认对本任务持续有用，全文如下；harness 已自动注入本窗口的纠错调用，**不要在 `<requested_entries>` 中重复请求**；可能为空）：
<carried_entries>
$carried_entries
</carried_entries>

本窗口 `<requested_entries>` 新请求剩余额度：$remaining_entries 条。

此前所有窗口的累积建议（按窗口标注，可能为空）：
<previous_advice>
$previous_advice
</previous_advice>

本窗口 ASR：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头，随后依次输出 `<window_notes>`（可为空块）、可选的 `<requested_entries>`（勿重复请求已透传词条）、`<keep_entries>`（每行一个已透传词条 key；没有需保留的条目时输出空块）、`<search_queries>`（可为空块）；不纠错、不翻译、不输出字幕内容，除上述块外不要输出任何其他文字。
