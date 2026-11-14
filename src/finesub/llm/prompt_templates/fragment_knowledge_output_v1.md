输出格式：
1. 知识更新输出一个 `<knowledge_proposals>...</knowledge_proposals>` 块，块内是 JSONL（每行一个 proposal，紧凑 JSON）；没有值得写入的知识时输出空块。
2. 不要 Markdown 代码块，不要解释性散文，不要声称已经写入知识库。
3. 除本任务规定的标签块外，不要输出任何其他块或文字。

引用方式：`<kb_entries>` 里每个条目的 H1 与每个已有内容行末尾都带一个 `<!-- @k数字 -->` 注释，那是该条目/该行的**句柄**（档案里的 `[别名]` 行是 `@a数字`；空槽没有句柄，要填就用 `append_lines` 向所属小节追加带该 `[标记]` 的整行），只在本次输出内有效。修改或删除已有行、向已有条目追加，一律用句柄；`<kb_entries>` 里没有的条目用 `category` + `entry`（源语言 key）指名。句柄不是内容，不要把它写进 `content`/`line`。

$term_line_rules
- 某个叫法（本名之外的固定绰号之类）有自己稳定的译法时，把它拆成**同一条目内的另一条术语行**，并从本行别名列里删掉。

`<knowledge_proposals>` 按 op 分以下几种行（字段顺序不限）：
{"op":"append_lines","entry":"@k1 或 源语言key","category":"$writable_categories（用 key 指名时必填）","section":"目标小节名","content":"一行或多行新增内容","reason":"更新依据与证据来源"}
{"op":"update","id":"@k12","line":"该行的完整新内容","reason":"…"}
{"op":"remove","id":"@k12","reason":"…"}
{"op":"create_entry","category":"$writable_categories","entry":"源语言key","entry_type":"游戏|动画|社区|其他（仅 common）","intro":"一句简介","aliases":["初始别名，可选"],"reason":"证据来源，以及为何不并入某已有词条（必填）"}
{"op":"retire_entry","entry":"@k1 或 源语言key","merged_into":"@k2 或 源语言key","reason":"内容已并入哪个词条的哪个分类"}
{"op":"rename_entry","entry":"@k1 或 源语言key","new_key":"新源语言key","reason":"…"}
{"op":"set_intro","entry":"@k1 或 源语言key","intro":"新的一句简介","reason":"…"}
{"op":"move_line","id":"@k12","section":"目标小节名","to_entry":"（挪到另一个条目时才填：目标条目的 H1 句柄或条目名；不填就是同一条目内换小节，填了且不给 section 就挪进同名小节）","entry":"（仅当这一行同时挂在两个条目下时才需要，用来指明动哪一个）","reason":"为什么这行属于那一节"}
{"op":"move_line","entry":"@k1 或 源语言key","from_section":"要整节挪走的小节名","to_entry":"目标条目的 H1 句柄或条目名","section":"（可选：落到目标条目的哪一节，不填就是同名小节）","reason":"…"}
{"op":"recategorize","entry":"@k1 或 源语言key","category":"$writable_categories（用 key 指名时必填）","to_category":"$writable_categories","reason":"为什么它属于新类别（必填）"}
$split_op_row{"op":"add_item","id":"行句柄 @k12、`[别名]` 行的 @a3，或条目的 H1 句柄 @k1","field":"misheard|aliases","value":"一个误听变体或别名","reason":"…"}

说明：
- schema 中 `a|b` 竖线表示枚举取值（多选一），不是拼接格式：`category` 只能取 `$writable_categories` 里的一个，`op` 只能是上述几种之一，`create_entry` 的 `entry_type` 只能取 游戏/动画/社区/其他 四个字面值；`$concat_counterexample` 这类拼接写法非法，自造枚举会被整条拒绝。
- `update` 用新的整行替换该句柄对应的行（行的类型不能变：术语行仍是四列术语行，自由句仍是自由句；要换类型就 `remove` 再 `append_lines`）。`remove` 删除该行。对 `[别名]` 行的 `@a` 句柄：`update` 整行改写这个条目的别名（保留 `[别名]` 标记，行体留空即清空），`remove` 清空，`add_item` 加一个（`value` 是一个别名，原样写，不用括；`field` 可省）；一批里对这一行只给一条，几条都以你读到的那一行为准，不会叠加（要加几个别名，就用一条 `update` 整行写全）。条目的 H1 句柄（`@k1` 这种出现在 `# 标题` 后面的）指整个条目：schema 里写「@k数字 或 源语言key」的条目字段（含 `merged_into`）和 `move_line` 的 `to_entry` 都可以填它；`add_item` 填它，是给条目本身登记误听（`field` 填 `misheard`，比如条目名常被听错的写法）或别名（`field` 填 `aliases`，与对 `@a` 句柄 `add_item` 是同一件事，二选一）。H1 句柄不能 `update`/`remove`。
- `append_lines` 的 `content` 每行一条；行首字段与该节现有行重复的会被自动跳过。`section` 必须是该条目允许的小节：固定小节的类别只能写入条目预览里列出的小节（当前没有内容的也可直接追加），自由命名的类别缺节自动创建。
- index 无需也无法通过 proposal 修改。
- `rename_entry` 只改条目名，旧名不会自动留下：旧名若仍可能出现在素材里，可以再用 `add_item`（`id` 填 H1 句柄、`field` 填 `aliases`）把它留作条目的别名。
- `set_intro` 整句替换条目的简介（H1 下面那一行），`entry` 可以用条目 H1 的句柄：只写一行、不以 `#` 开头，带换行或为空会被拒绝；条目名改用 `rename_entry`，条目的别名用 `add_item`。
- `retire_entry` 仅用于「碎片词条并入大词条后退役」：**必须先并入**，并在 `merged_into` 指明并入了谁。并入时，原词条里要保留的行用 `move_line`（带 `to_entry`）原样挪过去，必要时再 `update`（目标小节里已有同一个 `[标记]` 的行不要挪——会被拒——改用 `update` 改目标那一行）；不要在目标词条里用 `append_lines` 重抄一遍——重抄会换成新的行，这一行的历史、证据和人工改过的记录都会断掉。原词条里没有对应行的新内容，才用 `append_lines` 补。
- **放错地方就挪，不要删了重打。** `move_line` 把一行挪到同一条目的另一小节（带 `to_entry` 则挪到另一个条目），`recategorize` 把整个条目挪到另一类别；两者都保留这一行/这一条目的历史与证据，而「`remove` 再 `append_lines`」会换成一个新的东西，原来的记录就断了——只有要换行体类型时（见上面 `update` 一条）才这样做。`recategorize` 的 `reason` 必填，会记进条目的元数据；目标类别的模版放不下的行（节名没有，或那一节不收这种行体）会**自动挪进目标类别的暂存区**，移动本身照常完成——apply 报告里逐行列出，下一轮维护该把它们归位或删掉。想控制它们落在哪，就先用 `move_line`。
- `move_line` 的整节写法（给 `entry` + `from_section`、不给 `id`）把那一节的全部行挪到另一个条目：适合「这一块其实是另一个东西」——先 `create_entry` 新建条目（同一次输出里先建后挪即可），再整节或逐行挪过去，原条目必要时 `retire_entry`。不要删了在新条目里重写：那会换成新的行，历史、证据和人工改过的记录都会断掉；要改内容就挪过去之后逐行 `update`。目标小节里已有同一个 `[标记]` 的那一行会被单独跳过，其余照挪。
$split_op_note- `add_item` 给某一行（或用 H1 句柄给条目本身）登记误听变体（`misheard`）或别名（`aliases`）作检索索引；写在描述里的「误听: xxx」会被自动登记，所以通常不必单独输出。
- $rejected_rows

示例——条目摘录与本次的依据：

<kb_entries>
# 原神 <!-- @k1 -->
开放世界游戏。

## 角色

ヴィリーナ|维琳娜|Vilina|Ver.3.0 新登场代理人。 <!-- @k8 -->
ノルム|诺姆||补给站老板。 <!-- @k9 -->
</kb_entries>

依据：window 3 第 12–14 行主播说维琳娜已是常驻、属维多利亚家政；window 5 第 8 行说诺姆也卖道具；搜索「原神 赛斯」查到官网 3.1 版本公告，防卫军新人赛斯（Seth）登场。

对应的合法提案：

<knowledge_proposals>
{"op":"update","id":"@k8","line":"ヴィリーナ|维琳娜|Vilina|常驻代理人、维多利亚家政成员。","reason":"window 3 第 12–14 行：已常驻、属维多利亚家政"}
{"op":"update","id":"@k9","line":"ノルム|诺姆||补给站老板、道具商人。","reason":"window 5 第 8 行：也卖道具"}
{"op":"append_lines","entry":"@k1","section":"角色","content":"セス|赛斯|Seth|防卫军成员。","reason":"官网 3.1 版本公告（搜索「原神 赛斯」）"}
</knowledge_proposals>

示例里的写法：只改描述时别名列照抄（@k8 的 `Vilina`）；本无别名的行第三列保持留空（@k9）；描述内并列用顿号；新增术语行四列齐全。

反例（会被整条拒绝）：
{"op":"create_entry","category":"$concat_counterexample","entry":"X","intro":"x","reason":"r"} ← 拼接枚举非法
{"op":"update","id":"@k8","line":"ヴィリーナ|维琳娜|新描述。","reason":"r"} ← 只有三列（别名列缺失）
