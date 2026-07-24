请根据 system 指令处理第 $round_index 轮搜索的结果（第 0 轮为主调查模型发起；总搜索轮数上限 $max_rounds 轮）。$round_notice

背景资料（可能为空）：
<background>
$background
</background>

当前 Research Contract（用于本轮判断下一步；`priority` 已由 harness 按搜索轮次递减）：
<current_research_contract>
$contract_json
</current_research_contract>

已执行过的搜索 query（不要重复提出）：
<executed_queries>
$executed_queries
</executed_queries>

上一轮你输出的 Evidence Pack（第 0 轮为空；本轮在此基础上更新）：
<previous_evidence_pack>
$previous_evidence_pack
</previous_evidence_pack>

本地知识库索引（末轮为空；可按 system 指令请求词条全文——词条是本地权威资料，比网页搜索更可靠；如果索引中有与未完成 fact 直接相关的条目，优先请求）：
<streamer_index>
$streamer_index
</streamer_index>
<common_index>
$common_index
</common_index>

上一调用输出的 entry 选择（保留原始名称与顺序；可能为空）：
<previous_requested_entries>
$previous_requested_entries
</previous_requested_entries>
<previous_kept_entries>
$previous_kept_entries
</previous_kept_entries>

上一轮请求/保留后实际可用的知识库词条全文（另含 R1 持续携带词条；可能为空）：
<knowledge_entries>
$knowledge_entries
</knowledge_entries>

产生下方结果的上一轮实际搜索请求（已经过 harness 的额度限制与跨轮去重；Contract 是发起请求时的快照）：
<previous_search_request>
<research_contract>
$previous_contract_json
</research_contract>
<search_queries>
$previous_search_queries
</search_queries>
<extract_urls>
$previous_extract_urls
</extract_urls>
</previous_search_request>

本轮新搜索结果：
<search_results>
$search_results
</search_results>

最后提醒（读完以上全部输入后）：先以 `<reasoning>` 块开头；如果仍需继续检索则输出 `<search_queries>`/`<extract_urls>`（可附 `<requested_entries>`），否则省略这些块；然后**必须**输出更新后的完整 `<evidence_pack>`（覆盖全部 fact）。末轮不得输出检索请求。不能因为输出较长而省略任何必需标签、fact 或固定章节；压缩措辞也必须严格按要求格式完整输出。块外不要输出任何其他文字。$round_notice
