输出解析契约（最高优先级）：
以下规则决定输出能否被机器解析；违反结构要求会使整个窗口重做。
不得因输出较长而省略、缩写或用省略号代替任何必需字幕行；必须按格式完整输出。尤其 `<translated>` 必须给出完整终稿。
任何「(此处省略 N 行…)」「(内容已根据指令生成…)」一类的占位说明都等同于空输出，整窗作废重来——没有任何情况允许用描述代替字幕行。

1. 必须输出类 HTML 标签块；不要 Markdown 代码块，不要无标签散文。
2. `<translated>...</translated>` 终稿块有且仅有一个（回复里其它块的顺序见本契约第 9 条）：直接在其中完成$block_work；本地只解析本块生成 SRT。块内第一行必须先原样输出 header，header 后才可写下述 $output_column_count 列字幕行和丢弃行（`discard|`）；不要写说明行或 Markdown。
3. header 与字幕行固定 $output_column_count 列，顺序严格为：
   `$output_csv_header`
   header 不计入字幕行数$dropped_columns_clause。
   - `type`：默认字幕必须写 `sub`，不要留空——首列缺失会使整行列位错移作废$insert_type_clause。
   - `position`：$translated_position_clause$insert_position_clause。
$output_start_clause
$text_column_items
   - `conf`：只填 `high`、`median`、`low`，依次表示很有把握、大致正确、需人工核对。
   - `note`：纯输出元数据——短结论（如"词中接回""口播碎片成一句"）和检查项$insert_note_clause，无事可留空。不要在 note 里写推理过程。$note_gap_clause
4. **覆盖**：只能引用本窗源序号；同一源序号只能出现一次（void 后可重用）；各行按源顺序。每个源必须被某一行覆盖（单独成行或并入相邻合并行）或以 `discard|<源序号>` 显式丢弃——不得静默省略。**长度**按两个量估：加权字数（$weighted_char_count_rule）与跨度（首源开始到末源结束，由输入的 `start`、`duration` 算出）。合并后加权字数一般在 ${thr_hard_chars} 字、跨度 ${thr_hard_seconds} 秒左右以内；同一个人的同一句话确实连着时可以稍长，并在该行 note 里用一句短结论写明理由（如「同一句，拆开会断在词中」）。加权字数 >${thr_absolute_chars} 或跨度 >${thr_absolute_seconds} 秒是**绝对门槛**，任何情况都必须拒绝。$sandwich_limit_clause`<preceding_context>` 不属于本窗口，不得输出。
5. 每条记录只能占一个物理行。$text_cells 中的 `|` 改用全角 `｜`。
6. `conf` 必须符合格式；`type` 和 `position` 不得含 `|`。
7. 不要输出 SRT 编号或时间戳，也不要输出时长、间隔或字数的列；理由只写 note。
8. **`<void>`**：写完某行才发现时长、字数、分组或取舍错误时，在行尾追加 `<void>`；本地丢弃该行，源序号可重用。
9. 交付终稿的回复，规定顺序为：`<translated>` → 其它允许块。`<translated>` 里不写逐行推理：逐行的疑点与取舍在内部思考里权衡，不写出来，结果直接体现在输出行上，note 只写短结论。
10. 最终 SRT 只使用 translated 的 ${srt_column}；type/conf/note 均不进入 SRT。

字幕节奏与排版要求（针对 translated）：
1. 拟合并时先估算合并后的字数和跨度：一般在 ${thr_hard_chars} 字/${thr_hard_seconds} 秒左右以内，同一句话确实连着可以稍长；字数 >${thr_absolute_chars} 或跨度 >${thr_absolute_seconds} 秒是**绝对门槛**，任何多源合并都不得越过；单个源自身超限时如实输出，不因该门槛丢弃。源数遵守 position 规则。
2. 多源 position 用英文逗号连接连续的源序号；合并行在时间轴上从首源开始到末源结束。
3. $script_rule
4. 高度疑似幻觉或无意义重复：写 `discard|<源序号>` 显式丢弃（可附简短理由，如 `discard|18|复读幻觉`）。
