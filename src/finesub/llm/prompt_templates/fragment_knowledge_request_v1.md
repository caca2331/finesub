知识库词条请求规则（`<requested_entries>`，可选）：
1. 输入中有两份本地知识库索引（主播 / common）。词条 key = index 行首主 key = 条目 Markdown 文件的一级标题（`# 源语言本名`）。若本窗口涉及索引中的条目、且其正文对纠错或翻译有帮助，在 `<window_notes>` 之后输出一个 `<requested_entries>...</requested_entries>` 块：每行一个索引中的主 key 或别名。
2. 这些条目的全文会由 harness 注入随后的纠错调用（不注入本轮），与搜索结果分开计预算；索引已覆盖的内容不必再发搜索 query。
3. 新请求上限 $max_entries 条；user 输入中「已透传词条」无需也不应重复请求，新请求与透传词条合计不超过 $total_entries 条（超出时优先保留透传、丢弃新请求）。不需要时省略整个块。

词条透传（`<keep_entries>`）：
1. 在 `<requested_entries>` 之后、`<search_queries>` 之前，必须输出有且仅有一个 `<keep_entries>...</keep_entries>` 块（可为空块）。
2. 每行一个已透传词条（`<carried_entries>` 中实际可见的）的 key 或别名，表示该条目对后续窗口仍有价值、应继续透传。没有需保留的条目时输出空块。
3. 只能引用本轮 `<carried_entries>` 中实际出现的词条；不要把未注入的词条写进 keep。
