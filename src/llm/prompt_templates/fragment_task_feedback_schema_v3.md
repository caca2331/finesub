`<task_update_feedback>` 内容必须是一个紧凑 JSON 对象（不要 Markdown 代码块），schema 如下：
{"knowledge_hints":[{"category":"streamer|common","entry":"主词条key（源语言本名）","sub":"子词条行首字段（可选）","direction":"new_entry|replace_section|append_lines","focus":"一句话：想更新/新增什么","reason":"证据来源","source_ids":["12","13"],"confidence":7}],"asr_corrections":["可复用的 ASR 误听修正模式"],"uncertainties":["仍不确定、需人工或后续核实的点"]}
字段规则：
1. `knowledge_hints` 是核心输出：你在本次输入中发现的、值得写入本地知识库的条目线索及更新方向。`entry` **永远填主词条**：优先用知识库索引中的既有 key 或别名；索引中没有合适主词条时写你建议的新主词条名（源语言本名），`direction` 填 `new_entry`。
2. `sub`（可选）：当线索针对主词条内的某个子词条（某一行）时，填该行的行首字段（源语言词项），如《原神》词条内的 `ブレンニ`。带 `sub` 的 hint 表示母词条的行级更新，**不要为 sub 建独立词条**；单个角色、单个梗、单场事件一律作为母词条的行、用 `sub` 表达。
3. `category` 只能取字面值 `streamer` 或 `common` 两者之一（游戏、动画、社区对象等一律归 `common`）。schema 中的 `a|b` 竖线表示枚举取值，不是拼接格式；非法拼接可能被纠正为 `|` 前段或整条丢弃，因此必须只写一个合法值。
4. `direction` 取值：`new_entry`=建议新建主词条；`replace_section`=条目某节已过时需整体改写；`append_lines`=条目某节需要新增行（术语、经历、特点、人际关系等增量信息）。
5. $source_ids_rule
6. `confidence`：任务反馈专用的 1-9 整数（9 最高），表示“该线索值得写入且内容属实”的把握；它与字幕行的 high/median/low conf 是两个独立字段。
7. 只记录可复用、可核实、来自本次输入证据的内容；拿不准的写进 `uncertainties`，不要编造 hint。没有内容的字段输出空数组，不要省略字段。
8. 纯粹的误听/读音修正只写 `asr_corrections`；仅当该词条本身需要新增或更新知识库内容（人物、事件、设定等长期知识）时才另开 `knowledge_hints`，不要为同一个修正两边重复填写。
