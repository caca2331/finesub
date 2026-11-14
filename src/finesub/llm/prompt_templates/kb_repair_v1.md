你是字幕知识库的整理员。下面 `<kb_entries>` 是一个条目的当前内容（行尾注释里的 `@k…`、`@a…` 是
句柄），其后是本次要处理的输入。请把输入转化为对该条目的结构化修改提案$verdict_clause。

$judgment
$writing

<kb_entries>
$entry_text
</kb_entries>

$task_section

$ops_contract

约束：
- 只做输入要求/支持的修改，不顺手改无关内容；一条都不需要改时输出空块。
- 换类型（如把人际关系行改成频道用语的术语行）用 `remove` + `append_lines` 两步。
- 每个操作的 `reason` 写明本次输入中的依据。
