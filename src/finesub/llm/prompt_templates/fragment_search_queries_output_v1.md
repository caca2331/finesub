搜索 query 输出规则：
1. 必须输出有且仅有一个 `<search_queries>...</search_queries>` 块；块内每行一个 query，不要编号、引号、bullet、解释或 Markdown。
2. 最多 $max_queries 条，按重要性从高到低排列；超出的会被丢弃（行尾引导语不单独计数）。
3. 只搜索理解内容所必需、且你自身知识可能不足或过时的主题（如较新的游戏剧情/角色/系统、近期事件、社区梗、主播背景）；不要为你已确定的常识提交 query。$knowledge_query_note
4. 没有值得搜索的内容时输出空块 `<search_queries></search_queries>`。

$query_style
