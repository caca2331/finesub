获取缺失输入（同一任务内，至多一次）：
1. $acquire_clause读完本窗口输入后，若需要由 harness 代为取得的输入，这一条回复只输出 `<window_notes>`、下列请求块$style_block_clause，然后结束；harness 会执行全部请求，把结果作为下一条消息回填给你，你再在同一对话里按 system 与首条 user 消息的全部要求交付终稿。
2. 回填只有一次：一次提出全部相互独立的请求，不要分批、不要预留后续追问。回填之后不能再请求，未覆盖的疑点保守处理并写进 note。
3. $no_request_clause
请求回复的输出块：
<window_notes>
请求前对本窗口的简短要点（800 token 以内）：话题/剧情位置、说话状态、可疑 ASR 点及推断的正确候选。写于回填之前，未经证实的判断标「待定」；没有要点时输出空块。
</window_notes>
$entry_request_blocks$search_queries_rules
$knowledge_index_block