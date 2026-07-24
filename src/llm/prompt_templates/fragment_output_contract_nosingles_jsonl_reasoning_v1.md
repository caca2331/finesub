输出解析契约（最高优先级）：
以下规则决定输出能否被机器解析；违反结构要求会使整个窗口重做。
不能因为预计输出较长而省略、缩写或用占位说明代替任何必需字幕对象。`<translated>` 必须给出完整终稿。

1. 必须输出类 HTML 标签块；不要 Markdown 代码块，不要无标签散文。
2. 只输出一个 `<translated>...</translated>` 终稿块。块内是 **JSONL**：每个非空物理行必须是一个完整 JSON object；不要 header、JSON array、尾逗号、注释行或 `plan`。
3. 普通字幕对象必须按下列键名与顺序完整输出，所有键都必须存在：
   `{"type":"sub","position":"7","start":22.0,"duration":3.5,"gap":2.5,"corrected_text":"...","translation":"...","conf":"high","char_count":18.5,"note":""}`
   - `type`：普通字幕固定为 `sub`$insert_type_clause。
   - `position`：字符串；$translated_position_clause$insert_position_clause。
   - `start`：数字，直接抄该行首源在 `<asr_result>` 中的 start；合并行抄首源 start，保留 1 位小数。
   - `duration`：数字，本条字幕合并后的跨度秒数，保留 1 位小数$insert_duration_clause。
   - `gap`：数字，只表示本条结束后到下一条开始的间隔；合并行抄末源 gap，窗末可为 0。判断与前一句是否合并时看前一行 gap。
   - `corrected_text`：该 position 范围的源语纠错，不得为空。
   - `translation`：简体中文译文，不得为空。
   - `conf`：只能是 `high`、`median`、`low`。
   - `char_count`：JSON number。$weighted_char_count_rule。
   - `note`：JSON string，短结论和检查项$insert_note_clause；无事写空字符串。
4. 丢弃对象使用：`{"type":"discard","position":"18","note":"复读幻觉"}`。不得静默省略源。
5. **reasoning 门控**：合并 ≥2 源、discard、conf=low、或越过硬门槛的对象，必须在其正上方先输出一个独立对象：`{"type":"reasoning","reasoning":"局部证据"}`。reasoning 综合前后 1–2 行说明 gap、语义、说话人、字数/跨度；一条 reasoning 只管辖紧随的一条 sub/discard。纯单源、界内、conf=high/median 的对象前不得输出 reasoning。不要写 `#` 注释行，也不要把 reasoning 塞进字幕对象。
6. **覆盖**：只能引用本窗源序号；同一源只能出现一次；对象按源顺序。每个源必须被一个 sub 对象覆盖或被 discard 对象显式丢弃。`<preceding_context>` 不得输出。
7. JSON string 中的双引号、反斜杠和换行必须按 JSON 规则转义；一个对象只能占一个物理行。文本中的 `|` 无需改写。
8. 写完对象才发现错误时，给该对象追加 `"void":true`；本地忽略整个对象，源序号可在后续对象重用。
9. $reasoning_clause 规定顺序为 `<reasoning>` → `<translated>` → 其它允许块。开头 reasoning 只写全局判断；逐行判断写成对应字幕对象正上方的 type=reasoning 独立对象。
10. 最终 SRT 只使用 translation；start/duration/gap/type/conf/char_count/note 及 type=reasoning 对象均不进入 SRT，时间轴仍按 source id 回填。

字幕节奏与排版要求：
$pacing_merge_clause
3. translation 必须全程使用简体中文；corrected_text 保持源语言原样。
4. 高度疑似幻觉或无效内容使用 discard JSON object 显式丢弃。
