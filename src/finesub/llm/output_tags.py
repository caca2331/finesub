"""Extraction and validation of tag-wrapped model output blocks.

Model outputs wrap their core payloads in HTML-like tags (``<translated>``,
``<search_queries>``, ``<context_pack>`` ...) instead of emitting a bare JSON
answer; extracting a tagged block tolerates surrounding prose and is far more
robust to format drift.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


_CLOSING_TAG_START_RE = re.compile(r"</(?=\s*[A-Za-z_])")


def neutralize_closing_tags(text: str) -> str:
    """Untrusted text as it goes into a tagged prompt block: every ``</name``
    becomes ``＜/name``, so nothing in it can close the block it sits in and
    pass what follows off as the prompt's own text.

    For text from outside the task -- web search and URL extracts, knowledge
    entries (a shared or ingested entry is someone else's text), reference
    material. Text without a closing tag comes back unchanged."""

    return _CLOSING_TAG_START_RE.sub("＜/", text)


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


_CLOSING_AT_END_RE = re.compile(r"</([A-Za-z_][\w-]*)\s*>\s*$")


def last_complete_answer(text: str, required: Sequence[str]) -> int | None:
    """Where the last complete answer in ``text`` starts, or ``None`` when there is none.

    For a reply that is prose with an answer at its end -- reasoning that mentions the answer's tags, drafts
    it, then writes it. The answer starts at the latest opening of a ``required`` tag from which every
    ``required`` tag is exactly one top-level block (`find_top_level_tag_blocks`): a draft before it would make
    a second block. Openings that stand alone on their line are tried first, the rest only when none of
    those gives a complete answer: a block opens on its own line, while a tag named in prose -- or in the
    answer's own rows, which the latest opening would otherwise cut the answer at -- sits inside a line.
    Sibling blocks written right before it, with only whitespace between, belong to it too (a
    ``<reasoning>`` the answer opened with), as long as taking one in adds a block the answer does not have
    and leaves the required ones single.

    A tag inside a block is ambiguous from tags alone: a mention in the answer's rows and an answer opened
    mid-line after a mention standing alone on its line look the same. The rule follows the format the
    prompts ask for (a block opens on its own line). The cut is validated as any reply, so what a wrong cut
    lets through is what the task's own check accepts from a reply that is just that suffix; anything else
    fails and goes to the repair round, where every leaked reply went before.
    """

    body = text or ""
    names = {tag.lower() for tag in required}
    if not names:
        return None

    def complete(start: int) -> bool:
        tail = body[start:]
        return all(len(find_top_level_tag_blocks(tail, tag)) == 1 for tag in names)

    openings = [match for match in _TAG_TOKEN_RE.finditer(body)
                if not match.group(0).startswith("</") and match.group(1).lower() in names]
    alone = [match for match in openings if _alone_on_line(body, match)]
    start = next((match.start() for tier in (alone, openings) for match in reversed(tier)
                  if complete(match.start())), None)
    if start is None:
        return None
    while (closing := _CLOSING_AT_END_RE.search(body, 0, start)) is not None:
        name = closing.group(1)
        opened = [match for match in re.finditer(rf"<{re.escape(name)}\s*>", body[: closing.start()], re.IGNORECASE)]
        if not opened:
            break
        candidate = opened[-1].start()
        before = len(find_top_level_tag_blocks(body[start:], name))
        if before or len(find_top_level_tag_blocks(body[candidate:], name)) != 1 or not complete(candidate):
            break
        start = candidate
    return start


def _alone_on_line(text: str, match: "re.Match[str]") -> bool:
    before = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
    end = text.find("\n", match.end())
    return not before.strip() and not text[match.end() : len(text) if end < 0 else end].strip()


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

    The tag is required. Scanning the whole reply for a JSON object instead
    would accept any stray object -- a draft inside leaked reasoning, an
    example the model quoted -- as the product, silently; none of the stored
    replies ever needed it (owner 2026-10-01). A missing tag is a parse failure
    and takes the caller's retry.
    """

    blocks = find_top_level_tag_blocks(text, tag)
    if len(blocks) != 1:
        raise ValueError(f"Output must contain exactly one <{tag}> block, found {len(blocks)}.")
    return parse_json_object(blocks[0])
