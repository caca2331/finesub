窗口策略：
1. 当前窗口必须一次性完整输出；不要写“待续”“继续”等字样。
2. `<asr_result>` 中的每一行都属于本次输出范围，包括与上一窗口重叠的行。
3. `<preceding_context>` 是窗口之前最近若干条未纠错原始 ASR，只用于理解上文；时间列与本窗同一基准。$preceding_audibility_note
4. 前文行不属于输出范围：不得为其输出 JSON object、纠错翻译或并入 position；只能引用 `<asr_result>` 的源序号。
5. `<asr_result>` 是本次处理范围。短片段按合并规则处理；高度疑似幻觉用 type=`discard` 的 JSON object 显式丢弃。
6. 结合本窗口完整上下文给出最佳译文，不迁就此前窗口可能的译法。

只读前文示例：
<preceding_context>
41|-12.4|1.3|0.5|さっきのボス硬すぎだろ
42|-10.6|1.0|0.0|回復買ってから行くか
</preceding_context>
<asr_result>
43|0.9|1.1|0.5|よし 行くよ
44|2.5|1.6|0.4|今度こそ倒す
</asr_result>
正确做法：translated 从 position 43 开始；41/42 只用于确认语境。错误做法：输出 41/42、跨界合并或把前文专名当作已确认译名。
