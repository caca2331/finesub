请严格根据 system 指令，输出当前窗口的最终简体中文字幕。

通用背景知识和术语（来自背景调查，覆盖全部窗口；可能为空）：
$general_context_json

本窗口专属背景（来自背景调查，按窗口对齐；可能为空）：
<window_context>
$window_context
</window_context>

知识库条目详情（前置轮索取、或按用户备注关键词预注入的本地知识库条目全文；可能为空）：
<entry_details>
$entry_details
</entry_details>

$advice_input_block前置轮对本窗口的分析要点（由前置模型在搜索结果返回之前写下，仅供参考；其中的候选和判断未经证实，须结合$verify_basis交叉验证后才能采信；与背景调查重复时以背景调查为准；可能为空）：
<pre_round_notes>
$pre_round_notes
</pre_round_notes>

本窗口搜索结果（本地搜索代理执行前置搜索 query 的返回，可能为空）：
<search_results>
$search_results
</search_results>

本窗口易忘要求（通用）：
1. 本窗口 `<asr_result>` **共有 $current_asr_row_count 条字幕**（首行 header 不计数）。直接输出一个 `<translated>...</translated>` 终稿，在其中完成纠错、翻译、合并与丢弃。不要 Markdown 代码块、SRT 时间戳、`plan|` 或无标签散文。
2. translated 第一行先原样输出 header：`$output_csv_header`；header 后的字幕行固定 $output_column_count 列。`gap` **只表示本行结束到下一行开始的间隔**，不是本行与前一句的间隔；判断“与前一句”是否合并时看前一行的 gap。`conf` 只能是 `high`/`median`/`low`；`char_count` 单独填写加权译文字数（如 `12.5`）。按 system 的合并规则处理：$mid_reminder_merge_rule
3. 可以合并语义连贯的短片段，但多数源应保持独立；合并后先检查尾部去留和自然断点并酌情缩窄。高度疑似幻觉写 `discard|<源序号>` 显式丢弃（拿不准则保留并在 note 标「疑似幻觉」）。写完某行才发现不对时，行尾 `<void>` 废弃再重写。
4. `<asr_result>` 中每一行都是本次输出范围；只读 `<preceding_context>` 中的行一律不得输出。

本窗口易忘要求（模态相关）：
$reminder_tail

只读前文 ASR（不属于本窗口输出范围；首窗可为空）：
<preceding_context>
$preceding_context_csv
</preceding_context>

本窗口待处理 ASR：
<asr_result>
$current_asr_csv
</asr_result>

最后提醒（读完以上全部输入后）：本窗口共有 **$current_asr_row_count 条输入字幕**。先以 `<reasoning>` 开头（只写跨行/全局判断，不逐行预演），再写 `<translated>` 完整终稿；translated 第一行必须是完整 $output_column_count 列 header。$merge_reminder不能因为输出长而省略、缩写或用省略号替代任何必需 header 或字幕行：translated 必须给出完整终稿，覆盖本窗每个源（疑似幻觉以 `discard|<源序号>` 显式丢弃），并严格保持标签、header、$output_column_count 列字段与顺序；”(此处省略…)””(内容已生成…)”一类占位说明等同于空输出，整窗作废。每行首列 type 必须写 `sub`，不要留空。$trailing_blocks_reminder不要输出 `plan|` 或其他文字。
