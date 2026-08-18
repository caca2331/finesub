输出解析契约（最高优先级）：
以下规则决定输出能否被机器解析；违反结构要求会使整个窗口重做。
不能因为预计输出较长而省略、缩写或用省略号代替任何必需标签或字幕行，必须严格按要求格式完整输出。尤其 `<singles>` 必须逐源完整覆盖，`<translated>` 必须给出完整终稿。
任何“(此处省略 N 行…)”“(内容已根据指令生成…)”一类的占位说明都等同于空输出，整窗作废重来——输出的价值只在完整的逐行内容本身，没有任何情况允许用描述代替字幕行。

1. 必须输出类 HTML 标签块；不要 Markdown 代码块，不要无标签散文。
2. 纠错结果分两个字幕块，且各有且仅有一个：
   - **`<singles>...</singles>`**（先写）：`<asr_result>` 每个源序号恰好一行的单源对照稿；行数必须与本窗口输入字幕条数完全相等。本地不用它拼 SRT。
   - **`<translated>...</translated>`**（后写）：在 singles 基础上完成合并、润色和丢弃后的终稿；本地只解析本块生成 SRT。
   两块的第一行都必须先原样输出下述 header；header 后才写字幕行。不要写 `plan|…`、说明行或 Markdown。
3. header 与字幕行固定 $output_column_count 列，顺序严格为：
   `$output_csv_header`
   header 不计入 singles 的字幕行数。
   - `type`：默认字幕必须写 `sub`，不要留空——首列缺失会使整行列位错移作废$insert_type_clause。
   - `position`：singles 必须是单个源序号，禁止合并；$translated_position_clause$insert_position_clause。$output_start_clause
   - `duration`：本条字幕跨度秒数，保留 1 位小数；singles 填单源时长，translated 填合并后跨度$insert_duration_clause。必须为数字，并与实际分组一致。
   - `gap`：**只表示本条结束后到下一条开始的间隔**，绝不表示本条与前一句的间隔。singles 直接抄该输入行的 gap；判断本行是否与前一句合并时，必须看前一行的 gap。translated 合并行填末源的尾部 gap，窗末可填 `0`。必须为数字。
   - `corrected_text`：该 position 范围的源语纠错结果，不得为空。
   - `translation`：简体中文译文，不得为空。
   - `conf`：只能填 `high`（very certain）、`median`（likely correct）、`low`（better to manually check）。
   - `char_count`：加权译文字数的独立列。$weighted_char_count_rule。写 `11` / `12.5`；不要加“译”或“字”。
   - `note`：不再重复字数。$singles_note_style，并以五选一结论收束：「宜与前一句合并」/「宜与前两句合并」/「视情况可向后合并」/「宜独立」/「宜丢弃」。$note_gap_clause translated 可写短结论和检查项$insert_note_clause，无事可留空。
4. **singles 覆盖**：每个 `<asr_result>` 源序号有且仅有一行，按输入顺序；最后一行的 position 必须等于本窗末源序号（可据此自查有没有写完）。即使拟在终稿丢弃也必须输出并在 note 标明。`<preceding_context>` 不属于本窗口，不得输出。
5. **translated 覆盖**：只能引用本窗源序号；同一源序号只能出现一次（void 后可重用）；各行按源顺序。每个源必须被某一行覆盖（单独成行或并入合并行）或以 `discard|<源序号>` 显式丢弃——不得静默省略。$translated_merge_rule终稿可润色 singles，但不得改变已确定的纠错事实。
6. 每条记录只能占一个物理行。corrected_text、translation、note 中的 `|` 改用全角 `｜`。
7. `conf`、`char_count`、`duration`、`gap` 必须符合各自格式；`type` 和 `position` 不得含 `|`。
8. 不要输出 SRT 编号或时间戳。duration、gap、char_count 写各自列，理由只写 note。
9. **`<void>` 仅用于 translated**：写完某行才发现时长、字数、分组或取舍错误时，在行尾追加 `<void>`；本地丢弃该行，源序号可重用。singles 禁止 `<void>`。
10. $reasoning_clause 规定顺序为：`<reasoning>` → `<singles>` → `<translated>` → 其它允许块。在 reasoning 里不要写尖括号标签字面量。reasoning 只写跨行/全局判断（专名统一、话题与高风险区间定位、整体分组与验证思路）；每条的取舍与合并判断写在该行 note 里即可，不要在 reasoning 里逐行预演或重复各行结论。thinking 内同理——不必逐条预演 singles/translated，逐行判断直接落到该行 note。
11. 最终 SRT 只使用 translated 的 translation；type/conf/char_count/note 与整个 singles 均不进入 SRT。

字幕节奏与排版要求（针对 translated）：
$pacing_merge_clause
3. translation 必须全程使用简体中文；corrected_text 保持源语言原样。
4. 高度疑似幻觉或无效内容：singles 仍须输出并以「宜丢弃」收束；translated 写 `discard|<源序号>` 显式丢弃。
