本次要处理的整理候选（逐条判断）：
$candidate_rows

候选裁定：除 `<knowledge_proposals>` 外，再输出一个 `<candidate_verdicts>` 块（JSONL），
对每条候选恰好一行：
{"candidate":"@c1","verdict":"$verdicts","reason":"…","missing":"仅 needs_human 时填：缺什么证据"}

- `propose`：你为该候选提出了修改——请在对应 proposal 的 `reason` 里写明候选句柄（如 @c1），便于人工追溯；
- `dismiss`：候选不成立，reason 写明依据；
- `needs_human`：不确定且没有可行的验证途径——reason 写判断过程，missing 写清缺什么证据，等待人工裁定；
- chapter-alias 类候选（删检索面，高风险）不要直接提删除操作，只在 dismiss/needs_human 之间裁定。
