输出：有且仅有一个 `<$tag>` 块，内容是 JSON 数组，每个要给结论的 id 一条结论：
<$tag>
[{"id": "某行的 id", "verdict": "confirmed", "url": "https://…"},
 {"id": "某行的 id", "verdict": "corrected", "set": {"zh": "官方写法", "desc": "改后的整段描述"}, "url": "https://…"},
 {"id": "某行的 id", "verdict": "not_found", "kind": "temporary", "note": "查过哪些方向"}]
</$tag>
