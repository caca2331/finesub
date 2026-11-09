输出解析契约（最高优先级）：
以下规则决定输出能否被机器解析；违反结构要求会使整个窗口重做。
不得因输出较长而省略、缩写或用省略号代替任何必需字幕行；必须按格式完整输出。尤其 `<translated>` 必须给出完整终稿。
任何「(此处省略 N 行…)」「(内容已根据指令生成…)」一类的占位说明都等同于空输出，整窗作废重来——没有任何情况允许用描述代替字幕行。

1. 必须输出类 HTML 标签块；不要 Markdown 代码块，不要无标签散文。
2. `<translated>...</translated>` 终稿块有且仅有一个（回复里其它块的顺序见本契约第 9 条）：直接在其中完成$block_work；本地只解析本块生成 SRT。块内第一行必须先原样输出 header，header 后才写字幕行和 `discard|` 辅助行；不要写说明行或 Markdown。
3. header 与字幕行固定 $output_column_count 列，顺序严格为：
   `$output_csv_header`
   header 不计入字幕行数$dropped_columns_clause。
   - `type`：默认字幕必须写 `sub`，不要留空——首列缺失会使整行列位错移作废$insert_type_clause。
   - `position`：$translated_position_clause$insert_position_clause。
$output_start_clause
$text_column_items
   - `conf`：只填 `high`、`median`、`low`，依次表示很有把握、大致正确、需人工核对。
   - `note`：可写短取舍结论和检查项$insert_note_clause，无事可留空。$note_gap_clause
4. **覆盖**：只能引用本窗源序号；同一源序号只能出现一次（void 后可重用）；各行按源顺序。每个源必须被某一行覆盖（单独成行或并入相邻合并行）或以 `discard|<源序号>` 显式丢弃——不得静默省略。$translated_merge_rule`<preceding_context>` 不属于本窗口，不得输出。
5. 每条记录只能占一个物理行。$text_cells 中的 `|` 改用全角 `｜`。
6. `start`、`conf` 必须符合各自格式；`type` 和 `position` 不得含 `|`。
7. 不要输出 SRT 编号或时间戳，也不要输出时长、间隔或字数的列；理由只写 note。
8. **`<void>`**：写完某行才发现时长、字数、分组或取舍错误时，在行尾追加 `<void>`；本地丢弃该行，源序号可重用。
9. 交付终稿的回复，规定顺序为：`<decisions>` → `<translated>` → 其它允许块。`<decisions>` 是本窗的决定清单，不超过 8 行，每行一条结论：专名与称呼定成什么、哪些区间识别不可靠、要丢哪些行（合成一条写）；重交上一轮被退回的输出时，先逐条列出这一轮要改掉的问题和它违反的输出规则。只写结论，不写推导过程，不写字幕行草稿，块内不写尖括号标签字面量；写完立即开始 `<translated>`。每行的取舍写在该行 note，不在清单里逐行列出。
10. 最终 SRT 只使用 translated 的 ${srt_column}；type/start/conf/note 均不进入 SRT。

字幕节奏与排版要求（针对 translated）：
$pacing_merge_clause
2. $script_rule
3. 高度疑似幻觉或无意义重复：写 `discard|<源序号>` 显式丢弃。
