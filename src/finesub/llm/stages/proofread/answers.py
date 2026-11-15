"""Reading pass 3's reply: the `<proofread>` block, one changed line per row.

A reply that is not one is refused with a reason the retry hands back to the model; the agent path's validator
(`agent_validators`, `proofread-window`) refuses with the same reasons, so the two never disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...output_protocol import strip_leaked_thought
from ...text_cells import decode_text_cell

#: The categories the system prompt numbers under「改什么」.
CATEGORIES = range(1, 7)


class AnswerError(ValueError):
    """The reply did not answer in the agreed form."""


@dataclass(frozen=True)
class Change:
    """One line the model changed: its number in the window (1-based), the category, the line's corrected text
    and translation as they should read, and why."""

    position: int
    category: int
    corrected: str
    translation: str
    why: str


def answer_of(text: str) -> str:
    """The reply, or the answer at its end when it is leaked reasoning (`strip_leaked_thought`)."""
    return strip_leaked_thought(text, ("proofread",)) or text


def parse_changes(text: str, count: int) -> list[Change]:
    """The changed lines of a window of `count` lines. Each line of the block has exactly five fields: a `|` in
    any of them, the reason included, shifts the texts into the wrong columns with nothing to tell, so a line with
    more refuses the reply (the prompt forbids `|` in the reason; the input writes it full-width). The texts are
    read as the prompt wrote them (`\n` a line break). A number
    outside the window or given twice, a category outside 1-6, an empty corrected text or translation, or a line
    that names the block's own opening tag refuses the reply: the block cannot tell where a reply that opened it
    again meant its answer to start, and a draft above that line would be read as answer."""
    match = re.search(r"<proofread>(.*?)</proofread>", text or "", re.S)
    if match is None:
        raise AnswerError("没有闭合的 <proofread> 块")
    out: list[Change] = []
    seen: set[int] = set()
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if "<proofread>" in line.lower():
            raise AnswerError(f"这一行里又写了 <proofread>，分不清答案从哪开始：{line.strip()!r}")
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != 5:
            raise AnswerError(f"这一行不是五个字段（哪一列里都不要用半角 |）：{line.strip()!r}")
        number, category, corrected, translation, why = cells
        corrected, translation = decode_text_cell(corrected), decode_text_cell(translation)
        if not number.isascii() or not number.isdigit() or not 1 <= int(number) <= count:
            raise AnswerError(f"序号不在 1–{count} 里：{line.strip()!r}")
        if int(number) in seen:
            raise AnswerError(f"序号 {int(number)} 写了两次")
        if not category.isascii() or not category.isdigit() or int(category) not in CATEGORIES:
            raise AnswerError(f"类别不是 1–6：{line.strip()!r}")
        if not corrected.strip() or not translation.strip():
            raise AnswerError(f"纠错文或译文是空的：{line.strip()!r}")
        seen.add(int(number))
        out.append(Change(int(number), int(category), corrected, translation, why))
    return out
