"""The command table is the single truth about what `finesub` can do.

Dispatch and help used to be independent lists -- an if-chain in the shell
and a hand-written USAGE string in the front end. They drifted exactly the way
that arrangement always drifts: the published CLI never mentioned `agent-task`.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from finesub_bootstrap import shell as shell_module
from finesub_bootstrap.shell import (
    AGENT_CLEANUP_MODULE,
    COMMANDS,
    COMMANDS_BY_NAME,
    PIPELINE_MODULE,
    Shell,
    render_usage,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Help lines that are not commands: the default branch (anything that is not a
#: subcommand runs the pipeline) has no name to dispatch on, so it takes part in
#: neither the table nor the comparison below.
NON_COMMAND_INVOCATIONS = {"<input>"}


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _advertised(help_text: str) -> set[str]:
    found = set(re.findall(r"^  finesub (\S+)", help_text, flags=re.MULTILINE))
    return found - NON_COMMAND_INVOCATIONS


def _expected() -> set[str]:
    return {command.name for command in COMMANDS}


def test_every_command_in_the_table_can_actually_be_dispatched() -> None:
    for command in COMMANDS:
        assert bool(command.method) != bool(command.runtime_module), (
            f"{command.name}: exactly one of method/runtime_module"
        )
        if command.method:
            assert callable(getattr(Shell, command.method, None)), (
                f"{command.name} names a handler `Shell` does not have"
            )
        if command.runtime_module:
            # A moved module leaves this string pointing at nothing, and
            # nothing fails until a user runs the command. `find_spec` answers
            # without executing the module, so naming a heavy module here
            # costs no torch import.
            assert importlib.util.find_spec(command.runtime_module) is not None, (
                f"{command.name} runs `python -m {command.runtime_module}`, "
                "which is not an importable module"
            )
        assert command.help, f"{command.name} is dispatched but never advertised"

    # `agent-clean` dispatches to a method that shells out instead of naming a
    # `runtime_module`, so it needs the same check by hand. So does the module
    # a bare `finesub <args...>` runs, which is not in the table at all.
    assert importlib.util.find_spec(AGENT_CLEANUP_MODULE) is not None
    assert importlib.util.find_spec(PIPELINE_MODULE) is not None


def test_the_published_cli_advertises_the_table_and_nothing_else() -> None:
    module = _load(
        REPOSITORY_ROOT / "cli" / "src" / "finesub_cli" / "main.py",
        "finesub_cli_main_under_test",
    )

    assert _advertised(module.usage()) == _expected()


def test_the_knowledge_commands_forward_their_module_and_arguments(monkeypatch) -> None:
    """The three knowledge entries are pure forwards, so what matters is that
    each reaches the right module with its arguments intact.

    `test_every_command_in_the_table_can_actually_be_dispatched` proves the
    modules exist; it cannot see a table row wired to the wrong one, and these
    three differ only by a suffix (`finesub.llm.knowledge` vs `.update` vs
    `.share`) -- the shape most likely to be copy-pasted wrong.

    ⚠ It goes through `dispatch`, not through `run_in_runtime` directly.
    Calling the stub itself would assert that the stub records what it was
    handed -- true of any stub, and blind to the two things that can actually
    break: the word not reaching the table at all (it falls through to the
    pipeline), and the subcommand being consumed instead of forwarded.
    """

    forwarded: list[tuple[str, list[str]]] = []
    shell = Shell.__new__(Shell)
    # `dispatch` applies pending migrations before anything else; neither that
    # nor the paths it needs is what this test is about.
    shell.paths = None
    monkeypatch.setattr(
        Shell,
        "run_in_runtime",
        lambda _self, module, arguments: (
            forwarded.append((module, list(arguments))) or 0
        ),
    )
    monkeypatch.setattr(shell_module, "apply_pending", lambda *a, **kw: None)
    monkeypatch.setattr(
        Shell,
        "run_pipeline",
        lambda _self, arguments: pytest.fail(f"not dispatched: {arguments}"),
    )

    invocations = [
        ["knowledge", "show", "星野灯"],
        ["knowledge-update", "out/x/x.srt", "--refined-srt", "mine.srt"],
        ["knowledge-share", "pull", "--remote", "u"],
    ]
    for arguments in invocations:
        assert shell.dispatch(arguments) == 0

    assert forwarded == [
        ("finesub.llm.knowledge", ["show", "星野灯"]),
        ("finesub.llm.knowledge.update", ["out/x/x.srt", "--refined-srt", "mine.srt"]),
        ("finesub.llm.knowledge.share", ["pull", "--remote", "u"]),
    ]
