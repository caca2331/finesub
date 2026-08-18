"""Extraction and validation of tag-wrapped model output blocks.

Model outputs wrap their core payloads in HTML-like tags (``<translated>``,
``<search_queries>``, ``<context_pack>`` ...) instead of emitting a bare JSON
answer; extracting a tagged block tolerates surrounding prose and is far more
robust to format drift.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def find_tag_blocks(text: str, tag: str) -> List[str]:
    pattern = re.compile(
        rf"<{re.escape(tag)}\b[^>]*>(?P<body>.*?)</{re.escape(tag)}\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    return [match.group("body") for match in pattern.finditer(text or "")]


def extract_single_tag_block(text: str, tag: str, *, required: bool = True) -> str:
    """Return the body of the unique **top-level** ``<tag>`` block.

    Raises ``ValueError`` when the block is missing (if required) or duplicated.

    Top-level, not "anywhere": this is the same definition the session contract
    and :func:`missing_top_level_tags` use. They used to disagree — the gate
    checked top-level siblings while the payload readers took the first match
    at any depth — so a block the model buried inside ``<reasoning>`` would
    parse here and fail the contract, for the same reply.
    """

    blocks = find_top_level_tag_blocks(text, tag)
    if not blocks:
        if required:
            raise ValueError(f"Output is missing the <{tag}> block.")
        return ""
    if len(blocks) > 1:
        raise ValueError(f"Output must contain exactly one <{tag}> block, found {len(blocks)}.")
    return blocks[0].strip()


def looks_truncated_tag_block(text: str, tag: str) -> bool:
    """True when an opening ``<tag>`` exists without a matching closing tag."""

    lower = (text or "").lower()
    if f"<{tag.lower()}" not in lower:
        return False
    return not re.search(rf"</{re.escape(tag)}\s*>", text or "", flags=re.IGNORECASE)


_TAG_TOKEN_RE = re.compile(r"</?([A-Za-z_][\w-]*)\s*>")


def find_top_level_tag_blocks(text: str, tag: str) -> List[str]:
    """Return bodies of every top-level ``<tag>...</tag>`` (never nested).

    Model replies use sibling blocks only (``reasoning`` / ``window_notes`` /
    ``search_queries`` / ``translated`` / …). While any block is open, further
    opening tags are treated as prose mentions (including markdown `` `<tag>` ``)
    and not pushed, so a mid-``reasoning`` name-drop cannot poison the stack or
    steal a later sibling. Nesting itself is therefore tolerated — only the
    top-level siblings are extracted.

    An opening tag with no matching close anywhere after it never opens a
    frame: it is scaffolding (a stray ``<br>``, a void element the model
    invented), not a block. Pushing it used to leave the stack permanently
    dirty, which made *every* later top-level block invisible and threw away a
    well-formed payload. Truncation is still caught — an unclosed block of the
    wanted name simply yields nothing, which is the correct answer.
    """

    body = text or ""
    want = tag.lower()
    stack: List[str] = []
    content_start: int | None = None
    found: List[str] = []

    tokens = list(_TAG_TOKEN_RE.finditer(body))
    closes_after: Dict[str, List[int]] = {}
    for index, match in enumerate(tokens):
        if match.group(0).startswith("</"):
            closes_after.setdefault(match.group(1).lower(), []).append(index)

    def has_close_after(name: str, index: int) -> bool:
        return any(position > index for position in closes_after.get(name, ()))

    for index, match in enumerate(tokens):
        name = match.group(1).lower()
        closing = match.group(0).startswith("</")
        if closing:
            if name not in stack:
                continue
            while stack and stack[-1] != name:
                stack.pop()
            if not stack:
                continue
            stack.pop()
            if name == want and content_start is not None:
                found.append(body[content_start : match.start()].strip())
                content_start = None
            continue

        # Opening tag.
        if stack:
            # Nested / mid-prose mention: ignore so sibling schema stays intact.
            continue
        if not has_close_after(name, index):
            # Scaffolding, not a block -- never open a frame we cannot close.
            continue
        stack.append(name)
        if name == want:
            content_start = match.end()

    return found


def missing_top_level_tags(text: str, tags: List[str]) -> List[str]:
    """Return errors for required tags with no top-level (non-nested) block.

    Nesting per se is allowed (a ``<reasoning>`` block may mention other tag
    names); the failure this catches is a *missing* first-level block — which
    is what happens when the model swallows a sibling inside another block
    (e.g. ``<window_notes>`` ending up inside ``<search_queries>``).
    """

    errors: List[str] = []
    for tag in tags:
        if not find_top_level_tag_blocks(text, tag):
            errors.append(f"<{tag}> missing at top level (nested or absent)")
    return errors


def parse_line_items(body: str) -> List[str]:
    """Parse a one-item-per-line block: strip bullets/numbering, drop empties/dupes."""

    items: List[str] = []
    seen: set[str] = set()
    for raw_line in (body or "").splitlines():
        line = raw_line.strip()
        line = re.sub(r"^(?:[-*・•]|\d{1,3}[.)、])\s*", "", line).strip()
        line = line.strip("\"'“”")
        if not line or line in seen:
            continue
        seen.add(line)
        items.append(line)
    return items


GUIDED_QUERY_SEPARATOR = " >> "


def parse_guided_line_items(body: str) -> List[tuple[str, str]]:
    """Parse one-item-per-line blocks that may carry a ` >> guided` suffix.

    Each non-empty line is ``text`` or ``text >> one-sentence guided query``.
    Returns ``(text, guided_query)`` pairs (guided empty when absent);
    bullets/numbering/quotes are stripped like :func:`parse_line_items`.

    Dedup is by ``(text, guided)``: the guided query changes what the provider
    highlights or extracts, so the same page asked about twice with different
    focuses is two requests. Keying on ``text`` alone -- as this did until
    2026-08-13 -- dropped the second one here, before the search loop's own
    budget and dedup ever saw it.
    """

    items: List[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in (body or "").splitlines():
        line = raw_line.strip()
        line = re.sub(r"^(?:[-*・•]|\d{1,3}[.)、])\s*", "", line).strip()
        if GUIDED_QUERY_SEPARATOR in line:
            text, guided = line.split(GUIDED_QUERY_SEPARATOR, 1)
            text = text.strip().strip("\"'“”")
            guided = guided.strip().strip("\"'“”")
        else:
            text = line.strip("\"'“”")
            guided = ""
        key = (text, " ".join(guided.split()).lower())
        if not text or key in seen:
            continue
        seen.add(key)
        items.append((text, guided))
    return items


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and prose."""

    candidate = (text or "").strip()
    fence = CODE_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model output does not contain a JSON object.")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is not an object.")
    return parsed


def parse_json_tag_block(text: str, tag: str) -> Dict[str, Any]:
    """Parse the JSON object inside the unique ``<tag>`` block.

    Falls back to scanning the whole output for a JSON object when the tag is
    missing, so a model that forgets the wrapper doesn't hard-fail the round.
    """

    blocks = find_tag_blocks(text, tag)
    if len(blocks) > 1:
        raise ValueError(f"Output must contain exactly one <{tag}> block, found {len(blocks)}.")
    if blocks:
        return parse_json_object(blocks[0])
    return parse_json_object(text)
