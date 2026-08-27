"""Every text-mode subprocess call must name the encoding it decodes with.

`text=True` alone decodes with `locale.getpreferredencoding(False)`, which on a
Chinese Windows install is cp936, not UTF-8. The decode runs inside
`subprocess`'s own reader thread, so a child that writes UTF-8 -- ffmpeg's
metadata lines, a git commit subject, a traceback naming a CJK path -- does not
produce mojibake a caller could shrug off. It raises `UnicodeDecodeError` from a
thread nobody wrapped, and the surviving traceback points at `_readerthread`
rather than at the call that decided how to decode.

A guard rather than per-call tests: the failure needs a non-UTF-8 ANSI code page
to reproduce, so CI on a UTF-8 runner is green no matter how many call sites
forget. What can be checked anywhere is the source.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

#: Trees that ship. The suites inside them are skipped below.
SCANNED_TREES = ("src", "desktop", "cli")

#: Test directories anywhere under those trees -- this suite is not the only
#: one, `desktop/backend/tests/` is another. Their children are fixtures whose
#: output the test itself wrote, so pinning the decoding would pin the fixture
#: rather than the product.
TEST_DIRECTORY_NAMES = frozenset({"test", "tests"})

#: The two keywords that put `subprocess` in text mode. `universal_newlines` is
#: the old spelling of `text` and decodes identically.
TEXT_MODE_KEYWORDS = frozenset({"text", "universal_newlines"})

#: Constructors that decode. `check_output` is here for the day someone uses
#: it: it is text-mode whenever either keyword is set, exactly like `run`.
DECODING_CALLS = frozenset({"run", "check_output", "Popen"})


def _is_subprocess_call(node: ast.Call) -> str | None:
    """The `subprocess.X` this call names, or None if it is something else.

    Attribute-only: every call site in the tree spells the module out, and a
    bare `run(...)` is far more likely to be an unrelated local helper than an
    imported `subprocess.run` -- guessing at those would cost false failures in
    exchange for a call shape nobody writes here.
    """

    function = node.func
    if not isinstance(function, ast.Attribute):
        return None
    if function.attr not in DECODING_CALLS:
        return None
    module = function.value
    if isinstance(module, ast.Name) and module.id == "subprocess":
        return function.attr
    return None


def _text_mode_without_encoding(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _is_subprocess_call(node)
        if name is None:
            continue
        keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
        if not keywords & TEXT_MODE_KEYWORDS:
            # Bytes mode. The caller decodes explicitly or does not decode at
            # all; either way no locale-dependent decode happens in here.
            continue
        if "encoding" in keywords:
            continue
        offenders.append((node.lineno, name))
    return offenders


def test_text_mode_subprocess_calls_name_an_encoding() -> None:
    offenders: list[str] = []
    for tree_name in SCANNED_TREES:
        root = REPO_ROOT / tree_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if TEST_DIRECTORY_NAMES & set(path.parts):
                continue
            for line, name in _text_mode_without_encoding(path):
                relative = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{relative}:{line} subprocess.{name}")
    assert not offenders, (
        "text-mode subprocess calls decode with the ANSI code page unless they "
        'say otherwise; add encoding="utf-8", errors="replace":\n  '
        + "\n  ".join(offenders)
    )
