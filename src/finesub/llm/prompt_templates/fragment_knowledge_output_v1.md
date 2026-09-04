输出格式：
1. 知识更新输出一个 `<knowledge_proposals>...</knowledge_proposals>` 块，块内是 JSONL（每行一个 proposal，紧凑 JSON）；没有值得写入的知识时输出空块。
2. 不要 Markdown 代码块，不要解释性散文，不要声称已经写入知识库。
3. $reasoning_clause 除 `<reasoning>` 和本任务规定的标签块外，不要输出任何其他块或文字。

引用方式：`<kb_entries>` 里每个条目的 H1 与每一行末尾都带一个 `<!-- @k数字 -->` 注释，那是该条目/该行的**句柄**，只在本次输出内有效。修改或删除已有行、向已有条目追加，一律用句柄；`<kb_entries>` 里没有的条目用 `category` + `entry`（源语言 key）指名。句柄不是内容，不要把它写进 `content`/`line`。

术语行格式（固定四列）：`源语言|中文定名|别名|一句话描述`。
- 别名在**第三列**：多个用顿号「、」分隔，无别名时留空（`源|中||描述`）；别名本身不得含竖线。
- 描述在**最后一列**，可以含竖线；恰好三列的行会被整条拒绝（多半是漏了别名列）。
- `update` 术语行时别名列**双向同步**：照抄=不动，少列=删除未列出的别名，留空=清空全部别名——只改描述时把现有别名列原样照抄。
- 误听索引（misheard）**不渲染在行内**，update 时不必也无法复述；登记新误听用 `add_item` 或描述里的「误听: xxx」语法。

`<knowledge_proposals>` 按 op 分七种行（字段顺序不限）：
{"op":"append_lines","entry":"@k1 或 源语言key","category":"streamer|common（用 key 指名时必填）","section":"目标小节名","content":"一行或多行新增内容","reason":"更新依据与证据来源"}
{"op":"update","id":"@k12","line":"该行的完整新内容","reason":"…"}
{"op":"remove","id":"@k12","reason":"…"}
{"op":"create_entry","category":"streamer|common","entry":"源语言key","entry_type":"游戏|动画|社区|其他（仅 common）","intro":"一句简介","aliases":["初始别名，可选"],"reason":"为何不并入某已有词条（必填）"}
{"op":"retire_entry","entry":"@k1 或 源语言key","merged_into":"@k2 或 源语言key","reason":"内容已并入哪个词条的哪个分类"}
{"op":"rename_entry","entry":"@k1 或 源语言key","new_key":"新源语言key","reason":"…"}
{"op":"add_item","id":"@k12","field":"misheard|aliases","value":"一个误听变体或别名","reason":"…"}

说明：
- schema 中 `a|b` 竖线表示枚举取值（多选一），不是拼接格式：`category` 只能是 `streamer` 或 `common`，`op` 只能是上述七种之一，`create_entry` 的 `entry_type` 只能取 游戏/动画/社区/其他 四个字面值；`streamer|游戏` 这类拼接写法非法，自造枚举会被整条拒绝。
- `update` 用新的整行替换该句柄对应的行（行的类型不能变：术语行仍是四列术语行，`字段: 值` 行仍是该形态；要换类型就 `remove` 再 `append_lines`）。`remove` 删除该行。条目的 H1 句柄（`@k1` 这种出现在 `# 标题` 后面的）只能用于 `append_lines`/`retire_entry`/`rename_entry`，不能 `update`/`remove`。
- `append_lines` 的 `content` 每行一条；行首字段与该节现有行重复的会被自动跳过。`section` 必须是该条目允许的小节（streamer 固定七节，不可新建；common 除 `档案` 外可自由命名、缺节自动创建）。
- 针对 `元数据` 小节的任何修改都会被拒绝（harness 自动维护）；index 无需也无法通过 proposal 修改。
- `retire_entry` 仅用于「碎片词条并入大词条后退役」：**必须先并入**（同一输出块内的 `append_lines`，或此前已并入），并在 `merged_into` 指明并入了谁。
- `add_item` 给某一行登记误听变体（`misheard`）或别名（`aliases`）作检索索引；写在描述里的「误听: xxx」会被自动登记，所以通常不必单独输出。
- 被 harness 拒绝的行只记录在 apply 报告里，不影响其余行。

示例——条目摘录：

<kb_entries>
# 原神 <!-- @k1 -->
开放世界游戏。

## 角色

ヴィリーナ|维琳娜|Vilina|Ver.3.0 新登场代理人。 <!-- @k8 -->
ノルム|诺姆||补给站老板。 <!-- @k9 -->
</kb_entries>

对应的合法提案（覆盖常见情形）：

<knowledge_proposals>
{"op":"update","id":"@k8","line":"ヴィリーナ|维琳娜|Vilina|常驻代理人、维多利亚家政成员。","reason":"只改描述：别名列照抄 Vilina 不动"}
{"op":"update","id":"@k9","line":"ノルム|诺姆||补给站老板、道具商人。","reason":"描述内并列用顿号；该行本无别名，第三列保持留空"}
{"op":"add_item","id":"@k9","field":"misheard","value":"ノーム","reason":"多次误听为该形"}
{"op":"append_lines","entry":"@k1","section":"角色","content":"セス|赛斯|Seth|防卫军新人。","reason":"新登场角色，四列齐全"}
</knowledge_proposals>

反例（会被整条拒绝）：
{"op":"create_entry","category":"streamer|游戏","entry":"X","intro":"x","reason":"r"} ← 拼接枚举非法
{"op":"update","id":"@k8","line":"ヴィリーナ|维琳娜|新描述。","reason":"r"} ← 只有三列（别名列缺失）
