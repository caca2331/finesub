---
id: speaker-turn
kind: mini
applies: [capableB, capableC, basicA, basicB]
teach: 相邻且相关但换人一问一答，不合并；无法确认是否同一人时按不同人处理
---
跨说话人反例（相邻且相关，但换人一问一答，不合并）：

<input>
question | 320.0 | 1.4 | これ絶対課金するわ
answer   | 321.6 | 1.6 | いやまだ我慢しなって
</input>

<output>
sub | question | AUTO | AUTO | これ絶対課金するわ | 这个我绝对要氪金 | high | AUTO | 下一句为他人应答，跨说话人不合并
sub | answer | AUTO | AUTO | いやまだ我慢しなって | 不行，还是再忍一忍吧 | high | AUTO |
</output>

对照要点：局部序号 {ref:question}、{ref:answer} gap 极小且话题连续，但音频里是两人一问一答，跨说话人即使相邻也不合并；无法确认是否同一人时同样按不同人处理、不合并。
