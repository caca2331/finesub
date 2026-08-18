1. CSV 的开始时间以本窗口第一条字幕为 0 秒基准，仅用于把握节奏；不要输出任何时间戳。
2. `<preceding_context>` 是窗口开始前的只读原始转写（未纠错、可能含误听）：只用于衔接语境；不要输出其中的源序号，也不要纠正或翻译它们。
3. 不要为了让 corrected_text 和 translation 逐行严格对齐而强行合并成长字幕；连续语义单元内允许 translation 为中文语序小幅前后错位。
4. translation 必须全程使用**简体**字形（不要出现「這/時/機/愛」等繁体字）；corrected_text 保持源语言原样，不受此限。
$next_advice_reminder_item
