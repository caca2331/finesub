请严格根据 system 指令，输出当前窗口的最终简体中文字幕。

通用背景知识和术语（来自背景调查；可能为空）：
$general_context_json

$user_requests_block本窗口专属背景（来自背景调查，按窗口对齐；可能为空）：
<window_context>
$window_context
</window_context>

$entry_details_block$advice_input_block$pre_round_notes_block$search_results_block本窗口易忘要求（通用）：
1. $style_pick_reminder本窗口 `<asr_result>` **共有 $current_asr_row_count 条字幕**（首行 header 不计数）。终稿写在一个 `<translated>...</translated>` 块里，在其中完成$block_work_short。不要 Markdown 代码块、SRT 时间戳或无标签散文。
2. translated 第一行先原样输出 header：`$output_csv_header`；header 后的字幕行固定 $output_column_count 列。输入里的 `gap` 是上一行结束到本行开始的停顿；判断本行是否并进上一行时看本行的 gap。`conf` 只能是 `high`/`median`/`low`。按 system 的合并规则处理（字数按加权字数算）：合并后一般在 ${thr_hard_chars} 字/${thr_hard_seconds} 秒左右以内，同一句话确实连着可以稍长；超 ${thr_absolute_chars} 字或超 ${thr_absolute_seconds} 秒即越绝对门槛，任何情况禁止（单源豁免）。
3. $mid_reminder_merge_rule合并后先检查尾部去留和自然断点并酌情缩窄。高度疑似幻觉写 `discard|<源序号>` 显式丢弃（拿不准则保留并在 note 标「疑似幻觉」）。写完某行才发现不对时，行尾 `<void>` 废弃再重写。
4. 不写逐行推理：每行的合并、丢弃取舍直接体现在输出行上；note 只写短结论（如「词中接回」「疑似幻觉」）；超过一般长度的合并行，note 的短结论里还要写明为什么可以稍长。
5. `<asr_result>` 中每一行都是本次输出范围；只读 `<preceding_context>` 中的行一律不得输出。

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

最后提醒（读完以上全部输入后）：$style_pick_reminder本窗口共有 **$current_asr_row_count 条输入字幕**。交付终稿时以 `<translated>` 开头，写完整终稿；translated 第一行必须是完整 $output_column_count 列 header。同一个人的同一句话被切开、中间没有长停顿时合并；并完一般在 ${thr_hard_chars} 字/${thr_hard_seconds} 秒左右，${thr_absolute_chars} 字/${thr_absolute_seconds} 秒为绝对门槛，合并行不得越过。$merge_reminder不能因为输出长而省略、缩写或用省略号替代任何必需 header 或字幕行：translated 必须给出完整终稿，覆盖本窗每个源（高度疑似幻觉以 `discard|<源序号>` 显式丢弃），并严格保持标签、header、$output_column_count 列字段与顺序；「(此处省略…)」「(内容已生成…)」一类占位说明等同于空输出，整窗作废。每个字幕行首列 type 必须写 `sub`，不要留空。$trailing_blocks_reminder不要输出其他文字。
