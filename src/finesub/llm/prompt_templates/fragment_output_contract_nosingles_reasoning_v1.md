输出解析契约（最高优先级）：
以下规则决定输出能否被机器解析；违反结构要求会使整个窗口重做。
不能因为预计输出较长而省略、缩写或用省略号代替任何必需字幕行，必须严格按要求格式完整输出。尤其 `<translated>` 必须给出完整终稿。
任何"(此处省略 N 行…)"(内容已根据指令生成…)"一类的占位说明都等同于空输出，整窗作废重来——输出的价值只在完整的逐行内容本身，没有任何情况允许用描述代替字幕行。

1. 必须输出类 HTML 标签块；不要 Markdown 代码块，不要无标签散文。
2. 只输出一个 `<translated>...</translated>` 终稿块：直接在其中完成纠错、翻译、合并、润色与丢弃；本地只解析本块生成 SRT。块内第一行必须先原样输出 header，header 后才可写下述 $output_column_count 列字幕行和两种辅助行（`#` 注释与 `discard|`）；不要写 `plan|…`、说明行或 Markdown。
3. header 与字幕行固定 $output_column_count 列，顺序严格为：
   `$output_csv_header`
   header 不计入字幕行数。
   - `type`：默认字幕必须写 `sub`，不要留空——首列缺失会使整行列位错移作废$insert_type_clause。
   - `position`：$translated_position_clause$insert_position_clause。
$output_start_clause
   - `duration`：本条字幕跨度秒数，保留 1 位小数；填合并后跨度$insert_duration_clause。必须为数字，并与实际分组一致。
   - `gap`：**只表示本条结束后到下一条开始的间隔**，绝不表示本条与前一句的间隔。合并行填末源的尾部 gap，窗末可填 `0`。判断本行是否与前一句合并时，必须看前一行的 gap。必须为数字。
   - `corrected_text`：该 position 范围的源语纠错结果，不得为空。
   - `translation`：简体中文译文，不得为空。
   - `conf`：只能填 `high`（very certain）、`median`（likely correct）、`low`（better to manually check）。
   - `char_count`：加权译文字数的独立列。$weighted_char_count_rule。写 `11` / `12.5`；不要加"译"或"字"。
   - `note`：纯输出元数据——短结论（如"词中接回""口播碎片成一句"）和检查项$insert_note_clause，无事可留空。不要在 note 里写推理过程（推理由前置 reasoning 行承载）。$note_gap_clause
4. **覆盖**：只能引用本窗源序号；同一源序号只能出现一次（void 后可重用）；各行按源顺序。每个源必须被某一行覆盖（单独成行或并入相邻合并行）或以 `discard|<源序号>` 显式丢弃——不得静默省略。$translated_merge_rule`<preceding_context>` 不属于本窗口，不得输出。
5. 每条记录只能占一个物理行。corrected_text、translation、note 中的 `|` 改用全角 `｜`。
6. `conf`、`char_count`、`duration`、`gap` 必须符合各自格式；`type` 和 `position` 不得含 `|`。
7. 不要输出 SRT 编号或时间戳。duration、gap、char_count 写各自列，理由只写 note。
8. **`<void>`**：写完某行才发现时长、字数、分组或取舍错误时，在行尾追加 `<void>`；本地丢弃该行，源序号可重用。
9. **前置 reasoning 注释**（本变体核心机制）：
   格式：以 `#` 开头的注释行（如 `# gap=0.3s 同一枚举未完，合并后 3.2s/18字在界内`），紧接在其管辖的 sub 或 discard 行**正上方**。不写源序号——位置即归属。
   - **触发条件（门控）**：当且仅当该行满足以下之一时，**必须**前置一条 `#` 注释：
     (a) 合并 ≥2 个源；(b) discard 丢弃；(c) conf=low；(d) 越过硬门槛（字数 >${thr_hard_chars} 或跨度 >${thr_hard_seconds}s）。
   - **禁止条件**：纯单源、在界内（≤${thr_hard_chars} 字且 ≤${thr_hard_seconds}s）、conf=high 或 median 的行，**不得**前置 `#` 注释。
   - 注释写**局部决策推理**：综合该行及其前后各 1-2 行的上下文，分析合并/拆分/丢弃的依据（gap、语义连贯性、说话人、字数估算）。如有局部翻译取舍（惯用语、双关、ASR 撕坏需解读），也写在此处。
   - `#` 注释不计入字幕行、不进入 SRT；解析器直接跳过。一条注释只管辖紧随的一条 sub/discard 行。
10. $reasoning_clause 规定顺序为：`<reasoning>` → `<translated>` → 其它允许块。开头 `<reasoning>` 块只写**全局/跨行判断**（专名统一、称呼与语域、话题分段、高风险区间定位、整体验证思路）；局部逐行推理一律由 translated 内的 `#` 注释承载，不要在开头块里逐行预演。thinking 内同理——全局扫描在 thinking/开头块，逐行决策落到行间注释。
11. 最终 SRT 只使用 translated 的 translation；type/conf/char_count/note/`#` 注释均不进入 SRT。

字幕节奏与排版要求（针对 translated）：
$pacing_merge_clause
3. translation 必须全程使用简体中文；corrected_text 保持源语言原样。
4. 高度疑似幻觉或无效内容：写 `discard|<源序号>` 显式丢弃（前置 reasoning 行说明丢弃依据）。
