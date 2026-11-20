"""FineSub's own Codex login: `finesub agent-codex-login` (`login_command`).

Codex reads `$CODEX_HOME/AGENTS.md` into every request and has no switch for
it, so a call run from the user's own home carries the user's personal rules.
A connection whose credentials ride the environment runs from an empty home
per coordination domain; a login Codex keeps in its home (ChatGPT, or an API
key stored by `codex login`) needs a home that holds that login and nothing
else. This is that home: one fixed place per front end
(`paths.resolve_codex_login_home`), logged into once by the user, separate
from their `~/.codex` -- FineSub never copies or links the user's login
(owner 2026-10-17, reconfirmed 2026-11-14).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from finesub import paths

from . import codex_connection

#: The installed front end's command (`finesub_bootstrap.shell`).
INSTALLED_COMMAND = "finesub agent-codex-login"
#: A checkout has no `finesub` on PATH (pyproject declares no scripts), and an
#: installed `finesub` run from inside one logs into its own `user-data`, not
#: the checkout's home: a checkout runs the module.
CHECKOUT_COMMAND = "python -m finesub.llm.agent.codex_login"
#: How long `codex login status` may take; it reads a file or the keyring.
STATUS_TIMEOUT_SECONDS = 30


def login_command() -> str:
    """The command that logs into the home this process's calls use."""

    checkout = paths.own_checkout_root() if paths.checkout_data_enabled() else None
    return CHECKOUT_COMMAND if checkout is not None else INSTALLED_COMMAND


@dataclass(frozen=True)
class LoginStatus:
    logged_in: bool
    #: Codex's own words ("Logged in using ChatGPT", "Not logged in"), or why
    #: it was not asked.
    detail: str
    #: False when `codex login status` could not be run or did not answer in
    #: time: whether there is a login is then unknown, not "no".
    answered: bool = True


def connection_arguments(connection: codex_connection.CodexConnection) -> list[str]:
    """The user's connection settings as `--config` pairs, so a login lands
    where the calls look for it (`cli_auth_credentials_store` among them)."""

    return [part for override in connection.overrides for part in ("--config", override)]


def environment(home: Path, base: Mapping[str, str]) -> dict[str, str]:
    return {**base, "CODEX_HOME": str(home)}


def status(
    command: Sequence[str],
    home: Path,
    connection: codex_connection.CodexConnection,
    base_env: Mapping[str, str],
) -> LoginStatus:
    """`codex login status` in ``home``; a home not made yet holds no login."""

    if not home.is_dir():
        return LoginStatus(False, "no login home yet")
    try:
        result = subprocess.run(
            [*command, *connection_arguments(connection), "login", "status"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=STATUS_TIMEOUT_SECONDS,
            check=False,
            env=environment(home, base_env),
            creationflags=_no_window(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LoginStatus(False, f"`codex login status` did not answer: {exc}", answered=False)
    said = [
        line.strip()
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.strip() and not line.startswith("WARNING:")
    ]
    return LoginStatus(result.returncode == 0, said[-1] if said else f"exit {result.returncode}")


def _no_window() -> int:
    from finesub_bootstrap.processes import no_window

    return no_window()


_OPENING = """\
FineSub's Codex calls use a login of their own, kept in
  {home}
It is separate from your own ~/.codex: logging in or out here changes nothing
for the Codex you use yourself.

Why: Codex reads the AGENTS.md in its home into every request. Without this
login FineSub still works, but its Codex calls run from your own Codex home and
carry your personal rules. With it they run from the home above, which holds
only the login.

To undo: {command} --logout
Codex's own login follows (browser or device code).
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=login_command(),
        description=(
            "Log FineSub's Codex calls in with a login of their own, separate "
            "from your ~/.codex, so your global AGENTS.md stays out of them. "
            "Any other argument goes to `codex login` unchanged "
            "(e.g. --device-auth, --with-api-key)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="show whether it is logged in, and where")
    mode.add_argument("--logout", action="store_true", help="remove FineSub's Codex login")
    args, passthrough = parser.parse_known_args(argv)

    from .local_agent import (
        CodexDriverConfig,
        LocalAgentError,
        _codex_login_home,
        _resolve_shell_free_command,
        _sanitized_environment,
    )

    try:
        connection = codex_connection.read_connection()
    except codex_connection.CodexConnectionError as exc:
        print(f"Cannot read your Codex connection settings: {exc}")
        return 2
    if connection.credentials_from_env:
        print(
            f"Your Codex provider ({connection.provider}) takes its key from the environment, "
            "so FineSub's Codex calls already run from a home of their own: no login needed."
        )
        return 0
    home = paths.resolve_codex_login_home()
    if home is None:
        print(
            "This FineSub has no data directory (FINESUB_CHECKOUT_DATA=0 outside an installed "
            "FineSub), so its Codex calls can only run from your own Codex home."
        )
        return 2
    command = _resolve_shell_free_command(CodexDriverConfig().command)
    if command is None:
        print("Codex is not installed, or not on PATH.")
        return 2
    base_env = _sanitized_environment()

    if args.status:
        state = status(command, home, connection, base_env)
        print(f"FineSub's Codex login home: {home}")
        verdict = "Logged in" if state.logged_in else "Not logged in" if state.answered else "Could not tell"
        print(f"{verdict} ({state.detail})")
        if not state.logged_in and state.answered:
            print(f"Run `{login_command()}` to log in.")
        return 0 if state.logged_in else 1
    if args.logout:
        if not home.is_dir():
            print(f"Nothing to log out of: {home} does not exist.")
            return 0
        return subprocess.run(
            [*command, *connection_arguments(connection), "logout"],
            env=environment(home, base_env),
            check=False,
        ).returncode

    try:
        # Made and checked the way a call checks it: a file Codex would read
        # into the calls is refused here too, before anyone logs in.
        _codex_login_home()
    except LocalAgentError as exc:
        print(str(exc))
        return 2
    print(_OPENING.format(home=home, command=login_command()))
    # The terminal is the child's: the login is interactive.
    code = subprocess.run(
        [*command, *connection_arguments(connection), "login", *passthrough],
        env=environment(home, base_env),
        check=False,
    ).returncode
    if code == 0:
        state = status(command, home, connection, base_env)
        print(
            f"\nFineSub's Codex calls now run from {home}."
            if state.logged_in
            else f"\nCodex still reports: {state.detail}."
        )
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
