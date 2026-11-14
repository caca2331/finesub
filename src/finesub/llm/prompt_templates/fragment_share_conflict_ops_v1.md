$term_line_rules

`<knowledge_proposals>` 里改行用下面几种（字段顺序不限）：
{"op":"update","id":"@k12","line":"该行的完整新内容","reason":"…"}
{"op":"remove","id":"@k12","reason":"…"}
{"op":"add_item","id":"@k12","field":"misheard|aliases","value":"一个误听变体或别名","reason":"…"}

说明：
- 除上面规定的两个块外，不要输出任何其他块或文字；不要 Markdown 代码块。
- `update` 用新的整行替换该句柄对应的行（行的类型不能变：术语行仍是四列术语行，自由句仍是自由句）。`remove` 删除该行。行尾的 `<!-- @k… -->` 是句柄，不是内容，不要抄进 `line`。条目的 H1 句柄不能 `update`/`remove`/`add_item`。
- `add_item` 给某一行登记误听变体（`misheard`）或别名（`aliases`）作检索索引；写在描述里的「误听: xxx」会被自动登记，所以通常不必单独输出。
- $rejected_rows
