你是字幕知识库的校验员。下面列出**缺乏外部印证**的知识条目 claim（term 与档案 fact——只有这两类
有外部可核实路）。请逐条用联网搜索核实其内容是否属实、译名是否为官方或社区公认。

$judgment

校验对象（每条含 claim_id、条目行内容与所属条目）：
$claims_text

输出要求：先写简短的核查过程，然后输出**有且仅有一个** `<verify_results>` 块，内容为紧凑 JSON 数组，
每个校验对象一行结论：
<verify_results>
[{"claim_id": "c1", "verdict": "confirmed", "url": "https://…", "note": "官方页面确认译名"},
 {"claim_id": "c2", "verdict": "unverifiable", "note": "无公开来源"}]
</verify_results>

规则：
- `verdict` 三个值：`confirmed`（你亲自打开并核实过的 URL 支持该内容——必须给 `url`）、
  `refuted`（可靠来源**明确反驳**该内容——同样必须给 `url`，note 写明矛盾点；「查到反证」和
  「查不到」是两回事，不要混用）、`unverifiable`（查不到可靠来源；**不要**因此建议删除，
  它只是记为终态、下次不再重查）。
- 一个 URL 只支持它真正支持的那条 claim，不要复用到相邻条目。
- `url` 必须是**可长期复核的最终页面地址**（如 `https://<站点>/<条目路径>`）。搜索工具返回的
  跳转链接（例如 `vertexaisearch.cloud.google.com/grounding-api-redirect/...` 之类的重定向器）
  会过期、对将来的人毫无意义：请写你实际读到的那个页面的规范地址；若确实拿不到规范地址，
  就按 `unverifiable` 处理，不要拿跳转链接充数。
- 覆盖每一个 claim_id，不要遗漏、不要发明列表之外的 id。
