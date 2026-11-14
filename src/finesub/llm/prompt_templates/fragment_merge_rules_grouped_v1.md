$merge_intro
短片段合并策略（仅作用于 `<translated>`）：
1. 直接在 `<translated>` 中逐条${work_verbs}并按需合并。终稿以合并后时长/字数为准；$merge_scope_clause
2. $merge_seams_rule$turn_rule
3. $may_merge_lead也不一定要并，看两件事：
   - 停顿：长的自然停顿（约 1 秒以上）不并。中短的停顿看情况：同一句话、同一个意思接着说下去的并（例如词或短语被切成两行，前一行停在明显没说完的连接处）；转折另起、情绪变了、换了话题、只是相关的不并。
   - 并完的长度：一般在 ${thr_hard_chars} 字、${thr_hard_seconds} 秒左右以内；同一句话确实连着时可以稍长，并把理由$approval_reason_slot。加权字数 >${thr_absolute_chars} 或跨度 >${thr_absolute_seconds} 秒是**绝对门槛**，任何情况都必须拒绝。一句话很长，就让它留成几条，不硬并成一条。
4. **合并行检查**：已合并行先检查末源是否更宜独立或留给后句；若是则 void 并缩窄范围。越过 ${thr_absolute_chars} 字/${thr_absolute_seconds} 秒绝对门槛时无条件拆回。$turn_check$merge_check_seams
