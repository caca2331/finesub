"""A captured subprocess must not be decoded with the machine's code page.

`subprocess.run(..., text=True)` decodes with `locale.getencoding()`. On an
English install that is UTF-8 and everything works; on a zh-CN Windows install
it is cp936, and the first byte of UTF-8 the child emits raises
`UnicodeDecodeError` inside `subprocess`'s reader thread -- before the caller
sees an exit status, and with a traceback that names none of the real subject.

Three of these shipped: ffprobe reading a file whose tags are not ASCII killed
the pipeline at the duration probe, `_run_git` did the same for a knowledge
entry with a non-ASCII path, and the model downloader turned a hub error into a
decode error. All three are invisible to us -- CI and this repository's
machines run UTF-8 -- while Chinese Windows is the platform the docs name
first. A static check has no such blind spot.

The rule is on `encoding=`, not on `errors=`: a wrong code page is the bug, and
`errors="replace"` alone would decode cp936 bytes as UTF-8 just as wrongly, only
quietly. Callers that want bytes are unaffected -- this only fires once a call
has already asked for text.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: ⚠ `desktop/backend` is here because leaving it out is how this guard let one
#: through: `resources/gpus.py` called `subprocess.run(text=True)` for
#: `nvidia-smi` from the day the guard was written, and nothing was red --
#: the check was correct and simply never looked there. An outside
#: contributor reported it (PR #14). A guard's filter surface is part of the
#: guard; widening it costs nothing here because this test only reads files.
SOURCE_ROOTS = (
    Path(__file__).resolve().parents[1] / "src" / "finesub",
    Path(__file__).resolve().parents[1] / "src" / "finesub_bootstrap",
    Path(__file__).resolve().parents[1] / "desktop" / "backend",
)

#: The constructors that decode. `Popen` is here for the same reason as `run`:
#: it takes the same two keywords and hands back the same decoded streams.
CAPTURING_CALLS = {"run", "Popen", "check_output"}

#: Either keyword turns on decoding; `universal_newlines` is the old spelling
#: and still works, so a call using it would otherwise slip past.
TEXT_KEYWORDS = ("text", "universal_newlines")


def _asks_for_text(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg not in TEXT_KEYWORDS:
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and value.value is True:
            return True
    return False


def _is_subprocess_call(call: ast.Call) -> bool:
    target = call.func
    return (
        isinstance(target, ast.Attribute)
        and target.attr in CAPTURING_CALLS
        and isinstance(target.value, ast.Name)
        and target.value.id == "subprocess"
    )


def _offending_lines(source: Path) -> list[int]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_subprocess_call(node):
            continue
        if not _asks_for_text(node):
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        # A `**kwargs` splat reaches here as `arg is None`. The encoding may
        # well be in there, and guessing either way is worse than saying
        # nothing: no call site in this repository forwards subprocess
        # keywords, so this is a note about the future, not an exemption.
        if None in keywords or "encoding" in keywords:
            continue
        offenders.append(node.lineno)
    return offenders


def _sources() -> list[Path]:
    return sorted(
        path for root in SOURCE_ROOTS for path in root.rglob("*.py")
    )


@pytest.mark.parametrize("source", _sources(), ids=lambda path: path.name)
def test_captured_subprocess_output_pins_its_encoding(source: Path) -> None:
    offenders = _offending_lines(source)
    assert not offenders, (
        f"{source}: subprocess call(s) at line(s) "
        f"{', '.join(str(line) for line in offenders)} decode with the "
        "machine's code page. Pass encoding=\"utf-8\", errors=\"replace\" "
        "(media/ffmpeg.py's run_capture is the shared one for that package)."
    )


def test_the_check_can_still_see_an_offender(tmp_path: Path) -> None:
    """The guard's own smoke test: a rule that cannot fail protects nothing."""

    offender = tmp_path / "offender.py"
    offender.write_text(
        "import subprocess\n"
        "subprocess.run(['git'], capture_output=True, text=True)\n"
        "subprocess.run(['git'], capture_output=True, universal_newlines=True)\n"
        "subprocess.run(['git'], capture_output=True, text=True, "
        "encoding='utf-8')\n"
        "subprocess.run(['git'], capture_output=True)\n",
        encoding="utf-8",
    )
    assert _offending_lines(offender) == [2, 3]
