"""中文正文里的标点用全角，而且这条得有人盯着。

`docs/manual/` 在 2026-09-03 之前一路漂向半角（全角占比 36% → 28%），而仓库其余
中文文档是 100% / 98% 全角。漂移本身不贵，贵的是**修复它的那一趟**：一次 700 行的
改写里，真正的内容变动只有六个选项名，其余全是标点，审查时只能另写一个 token 差分器
才看得出来。这个守卫存在的理由就是不要再有第二次那样的 diff。

范围只有 `docs/manual/`，这是有意的：`docs/` 根与仓库根还各有一百多处 CJK 相邻的半角，
把它们一起归一化是另一个决定（`CHANGELOG.md` 里还有不该改写的历史条目）。想扩大范围，
先做归一化，再把 `_SCOPES` 加一项——顺序反了就是一片红。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: 归一化过、因此可以钉住的目录。
_SCOPES = ("docs/manual",)

_CJK = r"[一-鿿぀-ヿ]"

#: 遮蔽掉不该被这条规则管的东西：代码块、行内代码、链接目标、URL、HTML。
#: 与做归一化时用的是同一套遮蔽，否则守卫会去管它没归一化过的地方。
_MASKED = (
    re.compile(r"^```.*?^```", re.S | re.M),
    re.compile(r"\]\([^)]*\)"),
    re.compile(r"https?://\S+"),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"<[^>\n]+>"),
)

#: 半角标点，且两侧最近的非强调字符里有汉字——`**闸门**,只能` 这种形状里，
#: 紧邻位置上是 markdown 强调而不是汉字，所以不能只看相邻一格。
_HALF = re.compile(rf"{_CJK}[*_ ]{{0,3}}[,;:?!]|[,;:?!][*_ ]{{0,3}}{_CJK}")

_SKIP = "*_ \t"


def _mask(text: str) -> str:
    for pattern in _MASKED:
        text = pattern.sub(lambda m: "\x00" * len(m.group(0)), text)
    return text


def _documents() -> list[Path]:
    found: list[Path] = []
    for scope in _SCOPES:
        found.extend(sorted((REPOSITORY_ROOT / scope).glob("*.md")))
    return found


def test_the_scoped_documents_exist() -> None:
    """守卫扫不到东西时必须是红的，而不是安静地通过。

    `doc-guard-lessons` 里记着这一类：筛选面本身就是守卫的一部分，一个匹配到
    零个文件的正则永远是绿的。
    """

    assert _documents(), f"没有文档落在 {_SCOPES} 里"


@pytest.mark.parametrize(
    "path", _documents(), ids=lambda p: p.relative_to(REPOSITORY_ROOT).as_posix()
)
def test_chinese_prose_uses_full_width_punctuation(path: Path) -> None:
    masked = _mask(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for line_number, line in enumerate(masked.splitlines(), start=1):
        if _HALF.search(line):
            original = path.read_text(encoding="utf-8").splitlines()[line_number - 1]
            offenders.append(f"  {line_number}: {original.strip()[:90]}")
    assert offenders == [], (
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()} 的中文正文里有半角标点，"
        "改成 ，；：？！ ——代码、行内代码、链接与 URL 不在此列：\n"
        + "\n".join(offenders)
    )


def test_full_width_brackets_are_balanced() -> None:
    """成对转换才不会留下 `（…)` 这种半边。"""

    unbalanced = []
    for path in _documents():
        text = path.read_text(encoding="utf-8")
        if text.count("（") != text.count("）"):
            unbalanced.append(
                f"{path.name}: （{text.count('（')} vs ）{text.count('）')}"
            )
    assert unbalanced == [], "全角括号不配对：\n" + "\n".join(unbalanced)
