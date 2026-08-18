
词条透传（`<keep_entries>`）：
词条 key = 知识库 index 行首主 key = 条目 Markdown 文件的一级标题（`# 源语言本名`）。在 `<context_pack>` 之后输出有且仅有一个 `<keep_entries>...</keep_entries>` 块（可为空块）：每行一个 `<knowledge_entries>` 中实际出现的词条 key（主 key 或别名），只写对后续纠错全程大概率持续有用的词条（主播本人、正在玩的游戏本体是典型），上限 $max_keep_entries 条、超出丢弃。被 keep 的词条会由 harness 自动注入后续窗口（无需窗口重新请求）；引用 `<knowledge_entries>` 之外的 key 会被忽略。
