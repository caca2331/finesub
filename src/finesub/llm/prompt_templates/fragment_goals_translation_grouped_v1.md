${goals_title}：
${goals_first}${audio_fill_clause}${goals_first_tail}
2. ${goals_output}，不在单元格里写分析或 Markdown；拿不准的写进 note 列。$paren_rule
3. 默认逐源处理当前窗口输入。同一个人的同一句话被切开时，按下面「短片段合并策略」合并。高度疑似幻觉以 `discard|<源序号>` 显式丢弃。
4. 口语颗粒（语气词/填充词/结巴重复）${goals_grain_where}按三步判定：**是内容** → ${goals_grain_keep}；**是机械噪声** → ${goals_grain_noise}；压缩后**不剩任何信息**（纯占位的孤立垫词、无意义的感叹）→ ${goals_grain_empty}${grain_empty_audio_clause}。${goals_grain_rule}取舍按上述三步在 ${grain_target} 中直接处理。${goals_grain_note}
${goals_tail}${goals_style_conflict_lead}$style_conflict_rule
