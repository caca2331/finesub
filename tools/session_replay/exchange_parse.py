"""Parse harness exchange markdown into request/response sections."""

from __future__ import annotations

import re
from typing import Dict

# Only harness-authored section headers — search/extract bodies often contain
# markdown ``##`` headings that must not truncate the user prompt.
_SECTION_RE = re.compile(
    r"^##\s+(?P<title>请求（[^）]+）|请求\([^)]+\)|模型响应|请求|响应)\s*$",
    re.MULTILINE,
)


def split_exchange_sections(text: str) -> Dict[str, str]:
    """Split an exchange file into ``system`` / ``user`` / ``response`` bodies.

    Section titles match ``ExchangeLogger`` output: ``请求（system）``,
    ``请求（user）``, ``模型响应``. Missing sections become empty strings.
    """

    matches = list(_SECTION_RE.finditer(text or ""))
    out: Dict[str, str] = {"system": "", "user": "", "response": ""}
    for idx, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = (text[start:end] or "").strip()
        title_lower = title.lower()
        if title.startswith("请求") and "system" in title_lower:
            out["system"] = body
        elif title.startswith("请求") and "user" in title_lower:
            out["user"] = body
        elif "模型响应" in title or title.strip() in {"响应", "response"}:
            out["response"] = body
    return out
