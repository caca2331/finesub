"""How subtitle text sits in one `|`-separated cell of the CSV blocks, both ways.

One pair for every block the model reads or writes (`<asr_result>`, `<translated>`, the annotated CSV, pass 3's
rows): a `|` in the text becomes the full-width `｜`, so it cannot split the row, and a line break is written as
the two characters `\\n`. Reading back turns `\\n` into a line break again, so a literal `\\n` in the original
(a path, code) comes back as a break.
"""

from __future__ import annotations

import re


def encode_text_cell(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized.replace("|", "｜").replace("\n", r"\n")


def decode_text_cell(text: str) -> str:
    decoded = (text or "").strip().replace(r"\n", "\n")
    return re.sub(r"\n{2,}", "\n", decoded)
