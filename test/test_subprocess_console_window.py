"""A background subprocess must not open a console window.

Windows gives a console child its parent's console. When FineSub itself has
none -- started windowless by a GUI host, `pythonw`, a scheduled task -- every
console child allocates a new, visible one: a black box that flashes for as
long as the child lives. Measured 2026-09-16 from a consoleless process: each
`agy` invocation and `ffmpeg -version` flashed one. `processes.no_window()`
gives the child a hidden console instead, which its own children inherit.

The check reads the *value*, not just the keyword: the agent CLI's spawn always
passed `creationflags` (process group + suspended) and flashed anyway.

How the flash was measured, and this guard's known blind spots (a `**kwargs`
splat, `from subprocess import run`, an injected process factory), are in
docs/testing.md, "Windows 子进程不弹窗".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: Same surface as `test_subprocess_text_encoding.py`, for the reason given there.
SOURCE_ROOTS = (
    REPO / "src" / "finesub",
    REPO / "src" / "finesub_bootstrap",
    REPO / "cli" / "src",
)
SPAWNING_CALLS = {"run", "Popen", "call", "check_call", "check_output"}
#: Either spelling suppresses the window.
SUPPRESSING_NAMES = ("no_window", "CREATE_NO_WINDOW")

#: Children that must share the user's terminal, keyed by file and enclosing
#: function. A hidden console of their own would swallow their output and the
#: terminal's Ctrl+C.
SHARES_THE_TERMINAL = {
    # `finesub <command>` runs the pipeline here; its progress is the output.
    ("src/finesub_bootstrap/shell.py", "run_in_runtime"),
    # `finesub agent-clean` prints its report straight to the terminal.
    ("src/finesub_bootstrap/shell.py", "agent_clean"),
    # $EDITOR may be a console editor (vi), which needs the terminal.
    ("src/finesub/llm/knowledge/maintain.py", "_spawn_editor"),
    # `finesub agent-codex-login`: `codex login` is interactive (browser or
    # device code) and `codex logout` reports to the person who ran it.
    ("src/finesub/llm/agent/codex_login.py", "main"),
}


def _is_subprocess_call(call: ast.Call) -> bool:
    target = call.func
    return (
        isinstance(target, ast.Attribute)
        and target.attr in SPAWNING_CALLS
        and isinstance(target.value, ast.Name)
        and target.value.id == "subprocess"
    )


def _flag_source(value: ast.expr, scope: ast.AST) -> str:
    """The flags expression, following one local `name = ...` in the same scope.

    The agent spawn builds its flags in a variable first; reading only the call
    would see `creationflags=creationflags` and learn nothing.
    """

    text = ast.unparse(value)
    if isinstance(value, ast.Name):
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == value.id
                for target in node.targets
            ):
                text += " " + ast.unparse(node.value)
    return text


def _suppresses_window(call: ast.Call, scope: ast.AST) -> bool:
    for keyword in call.keywords:
        if keyword.arg is None:
            # A `**kwargs` splat: nothing in this repository forwards subprocess
            # keywords, so treat it as unknown rather than guess.
            return True
        if keyword.arg == "creationflags":
            source = _flag_source(keyword.value, scope)
            return any(name in source for name in SUPPRESSING_NAMES)
    return False


def _offenders(source: Path, relative: str) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    found: list[str] = []

    def visit(node: ast.AST, function: str, scope: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function, scope = node.name, node
        if (
            isinstance(node, ast.Call)
            and _is_subprocess_call(node)
            and not _suppresses_window(node, scope)
            and (relative, function) not in SHARES_THE_TERMINAL
        ):
            found.append(f"{node.lineno} ({function or '<module>'})")
        for child in ast.iter_child_nodes(node):
            visit(child, function, scope)

    visit(tree, "", tree)
    return found


def _sources() -> list[Path]:
    return sorted(path for root in SOURCE_ROOTS for path in root.rglob("*.py"))


@pytest.mark.parametrize("source", _sources(), ids=lambda path: path.name)
def test_background_subprocesses_open_no_console_window(source: Path) -> None:
    relative = source.relative_to(REPO).as_posix()
    offenders = _offenders(source, relative)
    assert not offenders, (
        f"{relative}: subprocess call(s) at {', '.join(offenders)} can flash a "
        "console window. Pass creationflags=no_window() "
        "(finesub_bootstrap.processes), or, if the child must share the user's "
        "terminal, add it to SHARES_THE_TERMINAL with the reason."
    )


def test_every_terminal_exemption_still_names_a_spawn() -> None:
    """A stale exemption would silently cover whatever lands there next."""

    for relative, function in SHARES_THE_TERMINAL:
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        spawns = [
            node
            for definition in ast.walk(tree)
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
            and definition.name == function
            for node in ast.walk(definition)
            if isinstance(node, ast.Call) and _is_subprocess_call(node)
        ]
        assert spawns, f"{relative}:{function} no longer spawns anything"


def test_the_check_can_still_see_an_offender(tmp_path: Path) -> None:
    """The guard's own smoke test: a rule that cannot fail protects nothing."""

    offender = tmp_path / "offender.py"
    offender.write_text(
        "import subprocess\n"
        "subprocess.run(['git'])\n"
        "subprocess.Popen(['agy'], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)\n"
        "subprocess.run(['git'], creationflags=no_window())\n"
        "subprocess.run(['git'], creationflags=subprocess.CREATE_NO_WINDOW)\n"
        "def spawn():\n"
        "    flags = subprocess.CREATE_NEW_PROCESS_GROUP\n"
        "    subprocess.Popen(['agy'], creationflags=flags)\n"
        "def quiet():\n"
        "    flags = subprocess.CREATE_NEW_PROCESS_GROUP | no_window()\n"
        "    subprocess.Popen(['agy'], creationflags=flags)\n",
        encoding="utf-8",
    )
    assert _offenders(offender, "offender.py") == [
        "2 (<module>)",
        "3 (<module>)",
        "8 (spawn)",
    ]
