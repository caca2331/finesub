请严格根据 system 指令，输出当前窗口的最终简体中文字幕。

通用背景知识和术语（可能为空）：
$general_context_json

本窗口专属背景：
<window_context>
$window_context
</window_context>

知识库条目详情：
<entry_details>
$entry_details
</entry_details>

此前所有窗口的累积建议：
<previous_advice>
$previous_advice
</previous_advice>

前置轮分析要点（未经证实，须结合$verify_basis验证）：
<pre_round_notes>
$pre_round_notes
</pre_round_notes>

本窗口搜索结果：
<search_results>
$search_results
</search_results>

本窗口易忘要求：
1. `<asr_result>` 共有 $current_asr_row_count 条字幕（header 不计数）。直接输出一个 `<translated>` JSONL 终稿，不要 Markdown、SRT 时间戳或无标签散文。
2. translated **没有 header**；每个非空物理行必须是一个 JSON object。sub 固定包含 type/position/start/duration/gap/corrected_text/translation/conf/char_count/note；字段名不可改写或漏掉。
3. position 是 string；start 抄首源，duration 为合并跨度，gap 为末源到下一源的间隔。按 system 合并规则处理：$mid_reminder_merge_rule
4. 多数源保持独立。高度疑似幻觉用 type=`discard` object；拿不准保留并在 note 标「疑似幻觉」。发现对象错误时追加 `"void":true` 后重写。
5. 合并/discard/conf=low/越硬门槛时，先在目标对象正上方输出一条 `{"type":"reasoning","reasoning":"..."}`；普通单源在界内时不要输出 reasoning 对象。不要输出 `#` 注释行或同行 reasoning 字段。
6. `<preceding_context>` 只读，不得输出。

本窗口易忘要求（模态相关）：
$reminder_tail

只读前文 ASR：
<preceding_context>
$preceding_context_csv
</preceding_context>

本窗口待处理 ASR：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒：本窗口共有 **$current_asr_row_count 条输入字幕**。先写全局 `<reasoning>`，再写完整 `<translated>` JSONL。$merge_reminder必须覆盖每个源；不能省略、缩写或写占位说明。每个 translated 非空行都必须独立通过 JSON 解析，不要 header、数组、尾逗号或 `#`。其后输出 `<next_advice>`、`<keep_entries>`（可空，及如需的 `<task_update_feedback>`）。
