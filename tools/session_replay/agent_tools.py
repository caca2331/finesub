"""A search/extract tool the replayed model drives itself, run by the harness.

Why this exists. There are two ways a research round can reach the web, and
each gives up something the other keeps:

* ``retrieval=local`` -- the harness picks the queries up front, runs them, and
  injects the rendered results. Every URL is in the artifact, so the pack's
  ``sources`` can be checked line by line. But the model never gets to follow
  up on what it reads.
* ``retrieval=native`` -- the model searches for itself through agy's own
  ``search_web``. It reaches more, but agy reports the *query* and not the
  source URLs, so the model has no real URL to cite; measured on this run it
  filled the ``sources`` field with plausible-looking inventions instead
  (YouTube ids like ``genshin_last_legacy``, ``4gamer.net/games/000/G000000``),
  none of which recurred across samples.

This module is the third option: the model chooses the queries, the harness
executes them against the project's own Exa/Gemma4/Tavily/DDG chain, and the
results come back as the next turn's input. Model-driven reach, harness-owned
provenance.

The mechanics are dictated by the transport. agy is one-shot -- a prompt goes
in, one answer comes out, and there is no way to register a Python function as
an agy tool. So the "tool" is a protocol across repeated one-shot calls: the
reply may carry a ``<tool_calls>`` block, this module runs it, and the next
call carries the results. Every round is appended to ``agent-tool-calls.jsonl``
next to the replies, which is the ledger the native arm cannot produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

TOOL_BLOCK = "tool_calls"
TOOL_RESULT_BLOCK = "tool_results"
DEFAULT_MAX_ROUNDS = 4
DEFAULT_MAX_CALLS_PER_ROUND = 6
# Deep-extract returns whole pages; without a cap one fetch can outweigh the
# entire research prompt.
EXTRACT_CHAR_CAP = 6000
SNIPPET_CHAR_CAP = 600

TOOL_PROTOCOL_TEXT = f"""
你可以联网检索，但不要自己调用任何内置搜索工具——本次运行由 harness 代你执行检索，
这样每一条 query 与每一个 URL 都会被完整记录下来，你给出的出处才可核对。

需要检索时，在回复的**最后**输出一个 `<{TOOL_BLOCK}>` 块，每行一个请求：

```
<{TOOL_BLOCK}>
search|<搜索关键词>|<可选：希望从命中网页里重点提取什么>
fetch|<完整 URL>|<可选：希望从该页重点提取什么>
</{TOOL_BLOCK}>
```

规则：
1. 一轮最多 {DEFAULT_MAX_CALLS_PER_ROUND} 条请求；最多来回 {DEFAULT_MAX_ROUNDS} 轮。
2. 只有 `search` 结果里出现过的 URL 才能 `fetch`，不要凭印象写 URL。
3. 输出 `<{TOOL_BLOCK}>` 的那一轮**只写这个块**，不要同时给最终答案。
4. 检索够了就正常按 system 指令输出最终答案，并且**不要**再输出 `<{TOOL_BLOCK}>`。
5. `sources` 只能引用你在 `<{TOOL_RESULT_BLOCK}>` 里真实看到的 URL。没有就留空——
   编造出处比没有出处更糟。
""".strip()


@dataclass(frozen=True)
class ToolCall:
    kind: str  # "search" | "fetch"
    argument: str
    focus: str = ""


@dataclass
class ToolRound:
    index: int
    calls: List[ToolCall] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    rendered: str = ""

    def as_record(self) -> Dict[str, Any]:
        return {
            "round": self.index,
            "calls": [
                {"kind": c.kind, "argument": c.argument, "focus": c.focus}
                for c in self.calls
            ],
            "results": self.rows,
        }


def parse_tool_calls(
    content: str, *, max_calls: int = DEFAULT_MAX_CALLS_PER_ROUND
) -> List[ToolCall]:
    """Read the request block out of a reply, if it asked for one.

    Uses the production top-level tag parser, so a reply that merely *mentions*
    the block name in prose does not read as a request.
    """

    from finesub.llm.output_tags import find_top_level_tag_blocks

    blocks = find_top_level_tag_blocks(content or "", TOOL_BLOCK)
    if not blocks:
        return []
    calls: List[ToolCall] = []
    for line in blocks[0].splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        kind = parts[0].lower()
        if kind not in {"search", "fetch"} or len(parts) < 2 or not parts[1]:
            continue
        calls.append(
            ToolCall(kind=kind, argument=parts[1], focus=parts[2] if len(parts) > 2 else "")
        )
        if len(calls) >= max_calls:
            break
    return calls


def _render_search(result: Any) -> tuple[str, Dict[str, Any]]:
    items = list(getattr(result, "items", ()) or ())
    lines = [f"### search: {getattr(result, 'query', '')}"]
    if getattr(result, "error", ""):
        lines.append(f"（检索失败：{result.error}）")
    if getattr(result, "answer", ""):
        lines.append(f"摘要：{str(result.answer)[:SNIPPET_CHAR_CAP]}")
    for item in items:
        lines.append(
            f"- {item.title}\n  {item.url}\n  {str(item.snippet)[:SNIPPET_CHAR_CAP]}"
        )
    row = {
        "kind": "search",
        "query": getattr(result, "query", ""),
        "provider": getattr(result, "provider", ""),
        "error": getattr(result, "error", ""),
        "urls": [item.url for item in items],
    }
    return "\n".join(lines), row


def _render_extract(result: Any) -> tuple[str, Dict[str, Any]]:
    body = str(getattr(result, "content", "") or "")[:EXTRACT_CHAR_CAP]
    lines = [f"### fetch: {getattr(result, 'url', '')}"]
    if getattr(result, "error", ""):
        lines.append(f"（抓取失败：{result.error}）")
    if getattr(result, "title", ""):
        lines.append(f"标题：{result.title}")
    if body:
        lines.append(body)
    row = {
        "kind": "fetch",
        "url": getattr(result, "url", ""),
        "provider": getattr(result, "provider", ""),
        "error": getattr(result, "error", ""),
        "chars": len(body),
    }
    return "\n".join(lines), row


def execute_tool_calls(calls: Sequence[ToolCall], client: Any) -> ToolRound:
    """Run one round of requests against the project's own retrieval chain."""

    round_ = ToolRound(index=0, calls=list(calls))
    chunks: List[str] = []
    for call in calls:
        try:
            if call.kind == "search":
                text, row = _render_search(client.search(call.argument, call.focus))
            else:
                text, row = _render_extract(client.extract(call.argument, call.focus))
        except Exception as exc:  # a dead provider must not kill the arm
            text = f"### {call.kind}: {call.argument}\n（执行异常：{type(exc).__name__}: {exc}）"
            row = {
                "kind": call.kind,
                "argument": call.argument,
                "error": f"{type(exc).__name__}: {exc}",
            }
        chunks.append(text)
        round_.rows.append(row)
    round_.rendered = (
        f"<{TOOL_RESULT_BLOCK}>\n" + "\n\n".join(chunks) + f"\n</{TOOL_RESULT_BLOCK}>"
    )
    return round_


def append_ledger(out_dir: Path, rounds: Sequence[ToolRound], sample_index: int) -> None:
    path = Path(out_dir) / "agent-tool-calls.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for round_ in rounds:
            record = {"sample": sample_index, **round_.as_record()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def ledger_urls(rounds: Sequence[ToolRound]) -> set[str]:
    """Every URL the harness actually saw, for checking a reply's ``sources``."""

    urls: set[str] = set()
    for round_ in rounds:
        for row in round_.rows:
            urls.update(row.get("urls") or [])
            if row.get("url"):
                urls.add(str(row["url"]))
    return urls
