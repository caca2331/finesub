"""The command table is the single truth about what `finesub` can do.

Dispatch and help used to be three independent lists -- an if-chain in the
shell and a hand-written USAGE string in each front end. They drifted exactly
the way that arrangement always drifts: the published CLI never mentioned
`agent-task`, and the desktop package's never mentioned `keys`.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

from finesub_bootstrap.shell import (
    AGENT_CLEANUP_MODULE,
    CLI_FRONT_END,
    COMMANDS,
    COMMANDS_BY_NAME,
    PACKAGE_FRONT_END,
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


def _expected(front_end: str) -> set[str]:
    return {
        command.name for command in COMMANDS if front_end in command.shown_in
    }


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
            # without executing the module, so naming `finesub.batch`
            # here costs no torch import.
            assert importlib.util.find_spec(command.runtime_module) is not None, (
                f"{command.name} runs `python -m {command.runtime_module}`, "
                "which is not an importable module"
            )
        assert command.help, f"{command.name} is dispatched but never advertised"

    # `agent-clean` dispatches to a method that shells out instead of naming a
    # `runtime_module`, so it needs the same check by hand.
    assert importlib.util.find_spec(AGENT_CLEANUP_MODULE) is not None


def test_the_published_cli_advertises_the_table_and_nothing_else() -> None:
    module = _load(
        REPOSITORY_ROOT / "cli" / "src" / "finesub_cli" / "main.py",
        "finesub_cli_main_under_test",
    )

    assert _advertised(module.usage()) == _expected(CLI_FRONT_END)


def test_the_package_command_line_advertises_its_own_subset() -> None:
    """A subset by choice, not by accident.

    Installing and removing an installation belong to the app, so the package's
    help leaves `setup` and `uninstall` out -- while still dispatching them.
    Everything else it must list, which is the half that had gone missing.
    """

    module = _load(
        REPOSITORY_ROOT / "desktop" / "assets" / "package-cli" / "finesub.py",
        "finesub_package_cli_under_test",
    )
    rendered = render_usage(PACKAGE_FRONT_END) + module.INSTALLATION_HELP

    assert _advertised(rendered) == _expected(PACKAGE_FRONT_END)
    assert _expected(PACKAGE_FRONT_END) < _expected(CLI_FRONT_END)
    assert {"setup", "uninstall"} == _expected(CLI_FRONT_END) - _expected(
        PACKAGE_FRONT_END
    )


def test_a_hidden_command_is_still_dispatched() -> None:
    """Visibility is a help concern; the package shell runs the whole table."""

    for name in ("setup", "uninstall"):
        assert name in COMMANDS_BY_NAME
        assert PACKAGE_FRONT_END not in COMMANDS_BY_NAME[name].shown_in
