词条透传（keep）：
词条 key = 知识库 index 行首主 key = 条目 Markdown 文件的一级标题（`# 源语言本名`）；每行写主 key 或 index 中的别名即可。
1. 在 `<next_advice>` 块之后，必须输出有且仅有一个 `<keep_entries>...</keep_entries>` 块（可为空块）：每行一个词条 key，只能引用本窗口「知识库条目详情」（`<entry_details>`）中实际出现的词条。
2. 只 keep 本窗口确实用到、且后续窗口大概率继续需要的词条（主播本人、正在玩的游戏本体是典型）；不再需要的词条不要写入，让它掉出透传链。拿不准就不写——后续窗口仍可自行请求。
3. 被 keep 的词条会由 harness 自动注入后续窗口（后续窗口无需重新请求）；上限 $max_keep_entries 条，超出丢弃；引用 `<entry_details>` 之外的 key 会被忽略。
