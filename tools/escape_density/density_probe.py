"""How escape density varies with the source language, and why it is not a knob.

Builds fully-escaped replies in the real CSV shape and measures the density a
transport fault would produce. The point of the numbers is negative: the
density of a *corrupted* reply depends on how much ASCII the reply carried to
begin with, which is a property of the task's source language, not of the
fault. See this directory's README.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from finesub.llm.output_protocol import OUTPUT_CSV_HEADER  # noqa: E402
from finesub.text import _ESCAPED_CODEPOINT  # noqa: E402

ROWS_PER_WINDOW = 16
FILLER = (
    "and then we were talking about the schedule for next week which is "
    "still not decided but probably fine either way so please stay tuned "
)


def as_a_broken_cli_would_write(text: str) -> str:
    return "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in text)


def density(text: str) -> float:
    return len(_ESCAPED_CODEPOINT.findall(text)) * 6 / max(len(text), 1)


def reply(source: str, translation: str, note: str = "") -> str:
    body = "\n".join(
        f"sub|{index + 1}|2.4|0.3|{source}|{translation}|high|"
        f"{len(translation)}|{note}"
        for index in range(ROWS_PER_WINDOW)
    )
    return (
        f"<singles>\n{OUTPUT_CSV_HEADER}\n{body}\n</singles>\n"
        f"<translated>\n{OUTPUT_CSV_HEADER}\n{body}\n</translated>"
    )


SHAPES = {
    "ja->zh, incident-like": (
        "こんにちは皆さん今日もよろしくお願いします",
        "大家好今天也请多多关照",
        "",
    ),
    "en->zh, short source": ("Hi everyone", "大家好", ""),
    "en->zh, typical source": (
        "Hi everyone thanks for coming to the stream today",
        "大家好感谢今天来看直播",
        "",
    ),
    "en->zh, chinese note": (
        "Hi everyone thanks for coming to the stream today",
        "大家好感谢今天来看直播",
        "语气偏口语，保留感谢",
    ),
    "en->zh, long source + english note": (
        FILLER[:150].strip(),
        "大家好感谢今天来看直播",
        "kept the casual register and trimmed the repetition",
    ),
}


def main() -> None:
    print(f"{'shape':<40} {'density':>8}")
    for name, (source, translation, note) in SHAPES.items():
        text = as_a_broken_cli_would_write(reply(source, translation, note))
        print(f"{name:<40} {density(text):>8.3f}")

    # Bounded to shapes a subtitle line can actually have. An unbounded grid
    # bottoms out near 0.015, but only on a 640-character "line" translated
    # into two characters -- a corner that proves nothing, and quoting it
    # would be exactly the kind of over-claim this directory exists to avoid.
    lowest = (1.0, ())
    for source_length in (40, 80, 120, 160, 200):
        for output_chars in (4, 6, 8, 12):
            for note_length in (0, 40, 80):
                text = as_a_broken_cli_would_write(
                    reply(
                        (FILLER * 5)[:source_length].strip(),
                        "好的" * (output_chars // 2),
                        (FILLER * 5)[:note_length].strip(),
                    )
                )
                value = density(text)
                if value < lowest[0]:
                    lowest = (value, (source_length, output_chars, note_length))

    print()
    print(
        f"lowest over plausible line shapes        : {lowest[0]:.3f}  "
        f"(source {lowest[1][0]} chars, {lowest[1][1]} out, note {lowest[1][2]})"
    )
    print("a stray literal escape in a long reply   : ~0.003")
    print()
    print("The spread across shapes is the point, not the floor: one complete")
    print("corruption reads 0.859 or 0.072 depending on the source language,")
    print("so any threshold calibrated on one language pair is wrong on the")
    print("next. That is why the predicate counts escapes instead.")


if __name__ == "__main__":
    main()
