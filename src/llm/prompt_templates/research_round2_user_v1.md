请根据 system 指令完成第二轮背景调查：理解整段内容，输出包裹在 `<context_pack>` 块中、可注入后续纠错翻译的紧凑 JSON context。

用户提供的额外信息（可能为空）：
<extra_info>
$extra_info
</extra_info>

第一轮调查的分析要点（写于搜索结果返回之前，其中标注"待定"的判断未经证实，须结合搜索结果与知识库交叉验证；可能为空）：
<round1_notes>
$round1_notes
</round1_notes>

第一轮索取的知识库条目详情：
<knowledge_entries>
$entry_details
</knowledge_entries>

本地搜索代理返回的搜索结果（按第一轮提出的 query 分组，可能为空）：
<search_results>
$search_results
</search_results>

带窗口标记的 ASR 字幕文本：
<transcript>
$transcript
</transcript>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头；随后输出有且仅有一个 `<context_pack>...</context_pack>` 块，内容为 system 规定结构的紧凑 JSON（`general_context` + 与窗口 id 对齐的 `window_contexts`），再输出 `<keep_entries>` 块（可为空）；不能再发起搜索$task_feedback_reminder；除上述块外不要输出任何其他文字。
