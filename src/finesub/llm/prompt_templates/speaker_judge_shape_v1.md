

## 定字幕的形状

除了轨号，每个 ▶ 行还可以带下面几个字段，只在为 true 时写进这一行的 JSON，用不上就不写。

`"join":true`：把这一行接到上一条保留下来的行后面，成为同一条字幕（上一行 drop 了，就接再上一条保留的行，说话人也跟那一行比）。这一段里第一个 ▶ 行不 join。
$join_criteria
- 写了 join 的行，`why` 写出为什么是同一个人接着说。

`"may_join":true`：这一行是同一个人接着上一行往下说，只是不在这一步并（上一行已经是完整的一句，或并起来会比短字幕长）。下一步纠错会${judge_next_step}决定并不并，但只在你标了 `may_join` 的地方并，所以认人这件事在这里做完。
$may_criteria
- 写了 `may_join` 的行，`why` 写出为什么是同一个人接着说。一行只写 `join` 与 `may_join` 中的一个。

`"drop":true`：这一行整行不进字幕。${judge_drop_kinds}
${drop_criteria}drop 的行照常答 `track`。

字幕行里的 `时长` 是这一行的秒数，`gap` 是上一行结束到这一行开始的秒数，与上一行重叠时记 0。
例（id 是虚构的，与本段无关）：
{"id":"qE","track":"keep","why":"一直在自言自语、没有在回应别人，是同一个人接着说：上一行以${seam_cue}收尾，gap 0.2 秒，并完约 2 秒","join":true}
{"id":"qF","track":"keep","why":"同一个人接着补充自己的上一句（中间没有人回答）：上一行已说完，gap 0.3 秒","may_join":true}
