输出格式：
1. 知识更新输出一个 `<knowledge_proposals>...</knowledge_proposals>` 块，块内是 JSONL（每行一个 proposal，紧凑 JSON）；没有值得写入的知识时输出空块。
2. 不要 Markdown 代码块，不要解释性散文，不要声称已经写入知识库。
3. $reasoning_clause 除 `<reasoning>` 和本任务规定的标签块外，不要输出任何其他块或文字。

`<knowledge_proposals>` 按 op 分六种行（字段顺序不限）：
{"category":"streamer|common","entry":"源语言key","op":"append_lines","section":"目标小节名","content":"一行或多行新增内容","reason":"更新依据与证据来源"}
{"category":"streamer|common","entry":"源语言key","op":"edit_lines","edits":[{"action":"change|insert_after|remove","line":12,"content":"新行内容（remove 不填）"}],"reason":"…"}
{"category":"streamer|common","entry":"源语言key","op":"replace_section","section":"目标小节名","content":"小节完整新全文","reason":"…"}
{"category":"streamer|common","entry":"源语言key","op":"create_entry","entry_type":"游戏|动画|社区|其他（仅 common）","intro":"一句简介","aliases":["初始别名，可选"],"reason":"…"}
{"category":"streamer|common","entry":"源语言key","op":"delete_entry","reason":"内容已并入哪个词条的哪个分类"}
{"category":"streamer|common","entry":"旧key","op":"rename_entry","new_key":"新源语言key","reason":"…"}

说明：
- schema 中 `a|b` 竖线表示枚举取值（多选一），不是拼接格式：`category` 只能是 `streamer` 或 `common`，`op` 只能是上述六种之一，`edits[].action` 只能是 `change`/`insert_after`/`remove`，`create_entry` 的 `entry_type` 只能取 游戏/动画/社区/其他 四个字面值；`streamer|游戏` 这类拼接写法非法，自造枚举会被整条拒绝。
- `edit_lines` 的 `line` 指 `<kb_entries>` 中该条目行首 `N| ` 的行号；同一次输出内行号全部按注入快照理解，不要自行换算增删后的位置。第 1 行（H1）与 `元数据` 节不可编辑。未注入或被标注截断的条目不得使用 `edit_lines` 与 `replace_section`（整节替换会丢掉你没看到的尾部）。
- 针对 `元数据` 小节的任何修改都会被拒绝（harness 自动维护）；index 无需也无法通过 proposal 修改。
- `delete_entry` 仅用于「碎片词条并入大词条后删除原文件」：**必须先并入**（同一输出块内的 append_lines/replace_section，或此前已并入）——被删词条的名字在其他词条或本批 proposal 内容中检索不到时，删除会被拒绝。
