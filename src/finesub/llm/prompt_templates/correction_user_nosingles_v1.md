请严格根据 system 指令，输出当前窗口的最终简体中文字幕。

通用背景知识和术语（来自背景调查；可能为空）：
$general_context_json

$user_requests_block本窗口专属背景（来自背景调查，按窗口对齐；可能为空）：
<window_context>
$window_context
</window_context>

$entry_details_block$advice_input_block$pre_round_notes_block$search_results_block本窗口易忘要求（通用）：
1. $style_pick_reminder本窗口 `<asr_result>` **共有 $current_asr_row_count 条字幕**（首行 header 不计数）。终稿写在一个 `<translated>...</translated>` 块里，在其中完成$block_work_short。不要 Markdown 代码块、SRT 时间戳或无标签散文。
2. translated 第一行先原样输出 header：`$output_csv_header`；header 后的字幕行固定 $output_column_count 列。输入里的 `gap` 是上一行结束到本行开始的停顿；判断本行是否并进上一行时看本行的 gap。`conf` 只能是 `high`/`median`/`low`。按 system 的合并规则处理：$mid_reminder_merge_rule
3. $merge_habit_reminder高度疑似幻觉写 `discard|<源序号>` 显式丢弃（拿不准则保留并在 note 标「疑似幻觉」）。写完某行才发现不对时，行尾 `<void>` 废弃再重写。
4. `<asr_result>` 中每一行都是本次输出范围；只读 `<preceding_context>` 中的行一律不得输出。

本窗口易忘要求（模态相关）：
$reminder_tail

只读前文 ASR（不属于本窗口输出范围；首窗可为空）：
<preceding_context>
$preceding_context_csv
</preceding_context>

$overlap_candidates_block本窗口待处理 ASR：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒（读完以上全部输入后）：$style_pick_reminder本窗口共有 **$current_asr_row_count 条输入字幕**。交付终稿时先写 `<decisions>`（不超过 8 行结论，重交被退回的输出时先列这一轮要改的问题），再写 `<translated>` 完整终稿；translated 第一行必须是完整 $output_column_count 列 header。$merge_reminder不能因为输出长而省略、缩写或用省略号替代任何必需 header 或字幕行：translated 必须给出完整终稿，覆盖本窗每个源（高度疑似幻觉以 `discard|<源序号>` 显式丢弃），并严格保持标签、header、$output_column_count 列字段与顺序；「(此处省略…)」「(内容已生成…)」一类占位说明等同于空输出，整窗作废。每个字幕行首列 type 必须写 `sub`，不要留空。$trailing_blocks_reminder不要输出其他文字。
