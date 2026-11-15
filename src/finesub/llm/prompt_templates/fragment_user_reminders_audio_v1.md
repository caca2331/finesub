1. CSV 的开始时间以本窗口剪辑音频的 0 秒为基准；音频前后 padding 内的语音不属于本窗口输入，不要为其输出字幕。
2. `<preceding_context>` 是窗口开始前的只读原始转写（未纠错、可能含误听）：只用于衔接语境；不要输出其中的源序号，其语音即使落在剪辑开头可听，也不要为其输出字幕。
3. $reminder_alignment
4. $reminder_script$reminder_kana
$next_advice_reminder_item
