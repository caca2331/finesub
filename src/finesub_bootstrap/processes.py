from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any

#: Windows searches the working directory for an executable before PATH unless
#: `NoDefaultCurrentDirectoryInExePath` is set. `shutil.which` honours the
#: variable only from 3.12 (the CLI also runs on 3.10 and 3.11), and only when
#: whoever called in set it (`Shell.dispatch` does; a front end calling
#: `run_in_runtime` directly may not), so `which` does not rely on it.
_WHICH_SEARCHES_CWD = os.name == "nt"


def no_window() -> int:
    """Creation flags for a child that must not open a console window.

    Windows gives a console child the console of its parent. A parent that has
    none -- anything started windowless, a GUI host, `pythonw`, a scheduled
    task -- makes each console child allocate a new one, and that one is
    *visible*: a black box that flashes for as long as the child runs. With
    this flag the child gets a hidden console instead, and every process *it*
    starts inherits that, so the whole tree below stays quiet too -- including
    what a vendor CLI spawns, which nothing here can pass flags to. Except a
    child the vendor starts detached: it has no console to inherit, and its own
    console children open visible ones. agy's update check is one (switched
    off by `AGY_QUIET_ENV` in `local_agent.py`).

    Measured 2026-09-16 with a consoleless FineSub process: every `agy`
    invocation and `ffmpeg -version` showed a window; with a console (even a
    hidden one) none did.

    Not for a child that must share the user's terminal -- the pipeline the
    `finesub` shell runs, an editor: a separate hidden console means it neither
    prints there nor receives the terminal's Ctrl+C.
    `test_subprocess_console_window.py` holds the list of those.
    """

    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def which(name: str) -> str | None:
    """`shutil.which` that never answers with a copy found only in the working directory.

    What the launcher finds it runs (the ffmpeg capability probe, the Python
    candidates), and the working directory may be a downloaded media folder.
    Where `shutil.which` still searches it first, a match there is looked up
    again along PATH alone; an entry naming that directory still counts, as
    it does on 3.12.
    """

    found = shutil.which(name)
    if found is None or not _WHICH_SEARCHES_CWD or Path(found).resolve().parent != Path.cwd().resolve():
        return found
    suffixes = [""] if Path(name).suffix else os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep)
    for entry in filter(None, os.environ.get("PATH", "").split(os.pathsep)):
        for suffix in suffixes:
            candidate = Path(entry) / f"{name}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


def terminate_process_tree(process: Any) -> None:
    pid = int(process.pid)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=no_window(),
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
