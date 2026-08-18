"""Where the local tokenizer binary is called `tokcount`, and nothing else.

Three layers look for the same binary -- the LLM layer that runs it, the
provisioning layer that decides whether to download one, and the checkout path
resolver -- and each used to carry its own spelling of the name. They had
already drifted: two of them still searched for `gemini-token-counter`, the
Go module's old name, and one offered a `bin/gemini-token-counter` candidate
that has never existed in this repository.

Stdlib-only on purpose: `finesub.llm.token_budget` imports it, and that runs on
plain `[harness]` installs and on the thin CLI's own interpreter.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil


#: Where the LLM layer reads an explicitly chosen binary's path from. The
#: variable keeps its historical spelling: it is what the desktop app already
#: injects and what users have in their `.env`.
TOKEN_COUNTER_VARIABLE = "GEMINI_TOKEN_COUNTER_EXE"

#: The one name. `runtime-manifest.json` ships the resource under it, the
#: release archive contains `tokcount.exe`, and the Go module is `tokcount`.
EXECUTABLE_NAME = "tokcount"

#: What the pre-compiled binary is called inside a source checkout, relative to
#: the repository root.
CHECKOUT_RELATIVE_PATH = Path("bin") / "windows-amd64" / f"{EXECUTABLE_NAME}.exe"


def configured_path() -> str | None:
    """The binary the environment names, if it names one."""

    return os.environ.get(TOKEN_COUNTER_VARIABLE) or None


def find_on_path() -> str | None:
    """The binary this machine already has on PATH, if it has one."""

    return shutil.which(EXECUTABLE_NAME)
