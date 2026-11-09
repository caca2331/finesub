"""The Codex user's connection settings, carried past `--ignore-user-config`.

The Codex driver isolates the user's `config.toml` so their MCP servers,
plugins, hooks and model choices cannot change what a FineSub call does. The
same flag also dropped *where* the call goes: a user who routes Codex through
a local proxy (`model_provider = "cliproxyapi"` plus its provider table) had
every FineSub call fall back to the built-in `openai` provider and their
ChatGPT login, silently (2026-09-30: every FineSub session that day recorded
`model_provider = "openai"`, every Codex Desktop session `cliproxyapi`).

Connection and authentication are facts about the machine, not behaviour, so
they follow the user: the keys below are read from the user's own config and
handed back as `-c` overrides, which survive `--ignore-user-config` (measured
on 0.157.1). Everything else in that file stays ignored.

Secrets never reach the command line, where process lists show them: a
provider's `experimental_bearer_token` moves into the child's environment and
the table names it through `env_key` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import math
import os
from pathlib import Path
import re
import socket
import tomllib
from typing import Any, Mapping
from urllib.parse import urlsplit

#: Top-level keys that say where a call goes and which login it uses. Names
#: from the 0.157.1 binary's top-level config keys; only `model_provider` (with
#: its table) is measured to work through `-c`, the others are passed only
#: when the user set them.
CONNECTION_KEYS: tuple[str, ...] = (
    "model_provider",
    "openai_base_url",
    "chatgpt_base_url",
    "cli_auth_credentials_store",
    "forced_login_method",
    "forced_chatgpt_workspace_id",
)
#: Where a provider's literal bearer token goes instead of the command line.
PROVIDER_TOKEN_ENV = "FINESUB_CODEX_PROVIDER_TOKEN"
#: How long the pre-flight connect to a local provider may take, per address.
LOCAL_CONNECT_TIMEOUT_SECONDS = 1.0

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


class CodexConnectionError(ValueError):
    """The user's connection settings cannot be carried, or cannot work now."""


@dataclass(frozen=True)
class CodexConnection:
    """What one Codex call needs to reach the provider the user chose."""

    #: `-c` values, each `key=<TOML literal>`.
    overrides: tuple[str, ...] = ()
    #: Values this driver itself puts into the child's environment.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Environment names the user's provider reads (`env_key`,
    #: `env_http_headers`); the driver passes them through from its own.
    passthrough_env: tuple[str, ...] = ()
    provider: str = ""
    base_url: str = ""
    #: The top-level keys carried, for the audit record.
    keys: tuple[str, ...] = ()
    #: The provider authenticates with a variable in the child's environment
    #: (`env_key`, a moved bearer token included) rather than the login kept
    #: in `$CODEX_HOME/auth.json` or the keyring -- so a call can run from a
    #: home that is not the user's.
    credentials_from_env: bool = False

    def audit(self) -> dict[str, Any]:
        """The record's view of it: names and addresses, never a secret."""

        return {
            "carried_keys": list(self.keys),
            "model_provider": self.provider or "openai",
            "base_url": self.base_url,
        }


def codex_home(environ: Mapping[str, str] | None = None) -> Path:
    """`$CODEX_HOME`, else `~/.codex` -- the rule Codex itself follows."""

    env = os.environ if environ is None else environ
    configured = env.get("CODEX_HOME", "")
    return Path(configured) if configured else Path.home() / ".codex"


def read_connection(home: Path | None = None) -> CodexConnection:
    """The connection settings in `<home>/config.toml`; none when it is absent."""

    path = (home or codex_home()) / "config.toml"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return CodexConnection()
    except OSError as exc:
        raise CodexConnectionError(f"Cannot read Codex config {path}: {exc}") from exc
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CodexConnectionError(f"Codex config {path} is not valid TOML: {exc}") from exc
    return connection_from_config(document, source=str(path))


def connection_from_config(
    document: Mapping[str, Any], *, source: str = "config.toml"
) -> CodexConnection:
    """Build the overrides from a parsed `config.toml`."""

    # Top level only. A legacy `profile = "..."` selector is not read: Codex
    # 0.157.1 itself no longer supports it (profiles are now chosen per call
    # with `--profile <name>` and `<name>.config.toml`, which FineSub never
    # passes), so following it would read the file differently from the
    # user's own Codex.
    chosen = {key: document[key] for key in CONNECTION_KEYS if key in document}

    overrides = [f"{key}={toml_literal(value, where=f'{source}: {key}')}" for key, value in chosen.items()]
    env: dict[str, str] = {}
    passthrough: list[str] = []
    credentials_from_env = False
    provider = str(chosen.get("model_provider") or "")
    base_url = str(chosen.get("openai_base_url") or "") if provider in ("", "openai") else ""
    tables = document.get("model_providers") or {}
    table = tables.get(provider) if provider and isinstance(tables, Mapping) else None
    # No table: a provider Codex defines itself (openai, ollama, lmstudio, or
    # one added later) -- the name alone is carried and Codex judges it.
    if isinstance(table, Mapping):
        if not _BARE_KEY.fullmatch(provider):
            raise CodexConnectionError(f"{source}: unsupported provider name {provider!r}")
        table = dict(table)
        for literal in ("http_headers", "query_params"):
            if literal in table:
                raise CodexConnectionError(
                    f"{source}: [model_providers.{provider}] sets {literal}, which FineSub would "
                    "have to put on the command line; use env_http_headers instead"
                )
        token = table.pop("experimental_bearer_token", None)
        if token is not None:
            if "env_key" in table:
                raise CodexConnectionError(
                    f"{source}: [model_providers.{provider}] sets both experimental_bearer_token "
                    "and env_key; keep one"
                )
            if not isinstance(token, str):
                raise CodexConnectionError(
                    f"{source}: [model_providers.{provider}].experimental_bearer_token is not a string"
                )
            env[PROVIDER_TOKEN_ENV] = token
            table["env_key"] = PROVIDER_TOKEN_ENV
        else:
            env_key = table.get("env_key")
            if isinstance(env_key, str) and env_key:
                passthrough.append(env_key)
        # Even beside `requires_openai_auth = true`: calls from an empty home
        # with no `auth.json` authenticated on the env token alone (owner's
        # proxy, 0.160.0, 2026-10-16).
        credentials_from_env = bool(table.get("env_key"))
        headers = table.get("env_http_headers")
        if isinstance(headers, Mapping):
            passthrough.extend(str(name) for name in headers.values() if name)
        base_url = str(table.get("base_url") or "")
        overrides.append(
            f"model_providers.{provider}="
            + toml_literal(table, where=f"{source}: [model_providers.{provider}]")
        )
    return CodexConnection(
        overrides=tuple(overrides),
        env=env,
        passthrough_env=tuple(dict.fromkeys(passthrough)),
        provider=provider,
        base_url=base_url,
        keys=tuple(chosen),
        credentials_from_env=credentials_from_env,
    )


def toml_literal(value: Any, *, where: str = "value") -> str:
    """`value` as a TOML literal for `-c`.

    Only the shapes a provider table uses: strings, booleans, numbers, string
    arrays and inline tables of those. Anything else is refused rather than
    guessed at -- `-c` falls back to a raw string when its value does not
    parse, so a wrong literal would not fail, it would change type.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodexConnectionError(f"{where}: non-finite number is not supported")
        return repr(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise CodexConnectionError(f"{where}: only arrays of strings are supported")
        return "[" + ", ".join(_toml_string(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{_toml_key(str(key))} = {toml_literal(item, where=f'{where}.{key}')}"
            for key, item in value.items()
        )
        return "{" + items + "}"
    raise CodexConnectionError(f"{where}: unsupported value type {type(value).__name__}")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else _toml_string(key)


def _toml_string(text: str) -> str:
    out = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char == "\n":
            out.append("\\n")
        elif char == "\t":
            out.append("\\t")
        elif char == "\r":
            out.append("\\r")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


def check_ready(
    connection: CodexConnection,
    child_env: Mapping[str, str],
    *,
    withheld: frozenset[str] = frozenset(),
) -> None:
    """Refuse before spawning when the provider cannot work right now.

    Two cases that would otherwise reach Codex, fail after its own retries,
    count as an ordinary failure, fail the quota probe the same way and end in
    a two-hour freeze blamed on a spent subscription: a variable the provider
    reads is missing from the child's environment (stripped as one of
    FineSub's own credentials, or never set), and a provider on this machine
    whose port does not answer (the proxy is not running). Nothing here reads
    Codex's wording. The connect is a snapshot: a proxy that dies after it
    passes takes the ordinary path.
    """

    for name in (*connection.env, *connection.passthrough_env):
        if child_env.get(name):
            continue
        if name.upper() in withheld:
            raise CodexConnectionError(
                f"Your Codex provider reads its key from {name}, which is one of FineSub's own "
                "credentials and is never handed to an agent CLI; point your provider at a variable of its own"
            )
        raise CodexConnectionError(
            f"Your Codex config needs the environment variable {name}, which is not set"
        )
    target = local_endpoint(connection.base_url)
    if target is None:
        return
    host, port = target
    try:
        # Tries every address the name resolves to (`localhost` may give
        # `::1` before `127.0.0.1`) and raises only when all of them fail.
        with socket.create_connection((host, port), timeout=LOCAL_CONNECT_TIMEOUT_SECONDS):
            pass
    except OSError as exc:
        raise CodexConnectionError(
            f"The Codex provider you configured ({connection.provider or 'openai'} at "
            f"{connection.base_url}) is not answering: {exc}"
        ) from exc


def local_endpoint(base_url: str) -> tuple[str, int] | None:
    """`(host, port)` when `base_url` points at this machine, else None."""

    if not base_url:
        return None
    try:
        parts = urlsplit(base_url)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname or ""
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                return None
        except ValueError:
            return None
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return host, port
