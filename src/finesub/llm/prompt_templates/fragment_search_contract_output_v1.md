搜索输出规则（多轮搜索模式）：
你的搜索 query 不是一次性的：本地搜索代理会先执行你的第 0 轮 query，之后一个轻量搜索代理会依据你制定的 Research Contract 评估结果并继续追加若干轮搜索，最终把整理好的证据汇总（Evidence Pack）注入后续调查。因此除 query 外，你还要输出一份 Research Contract，说清"到底要查什么、查到什么程度算完成"。

1. 必须输出有且仅有一个 `<research_contract>...</research_contract>` 块，块内是一个 JSON 对象（不要 Markdown 代码块）：
{
  "goal": "一句话说明这次调查到底要什么（服务于后续纠错翻译的哪个需求）",
  "facts": [
    {"id": "F1", "fact": "要查证的事实/问题", "priority": 5, "done_when": "什么算查到/确认", "hints": "候选写法、限定词、可能的信息来源"}
  ],
  "out_of_scope": ["明确不需要查的内容（已有背景覆盖、常识等）"]
}
2. `facts` 按重要性给 `priority`（1-5 整数，5 最高；搜索优先级越高）。每个 fact 的 `id` 用 F1、F2… 依次编号。fact 数量控制在 12 个以内，聚焦影响理解与翻译质量的内容。涉及 ASR 误听的 fact（如“某词的真实对应”），`hints` 中应尽量列出具体疑似误听文本及候选正确写法，供搜索 loop 逐条查证并在 Evidence Pack 中保留 ASR 原文与候选写法的对应。
3. 必须在 `<research_contract>` 之后输出有且仅有一个 `<search_queries>...</search_queries>` 块，作为第 0 轮搜索 query；块内每行一个 query，不要编号、引号、bullet、解释或 Markdown。优先覆盖高 priority 的 facts。
4. 第 0 轮最多 $max_queries 条，按重要性从高到低排列；超出的会被丢弃（行尾引导语不单独计数）。追加轮由轻量搜索代理提出，不需要你考虑。
5. 只搜索理解内容所必需、且你自身知识可能不足或过时的主题；不要为你已确定的常识提交 query。$knowledge_query_note
6. 没有值得搜索的内容时，`facts` 输出空数组、`<search_queries>` 输出空块。

$query_style
