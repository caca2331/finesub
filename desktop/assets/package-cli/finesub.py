"""`finesub` for a FineSub Desktop package, run beside the executable.

Shipped at the root of the package and started by ``finesub.cmd`` through the
managed interpreter -- the app's own executables are windowed, so they cannot
serve a command line. Everything here is bootstrap: find the app sources this
install is running, put them on the path, hand over to the shared shell.

Resolving the active version duplicates a few lines of
``finesub_bootstrap.shell`` on purpose: that module is one of the sources this
has to find first.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

USAGE = """\
FineSub — local long-form audio to subtitles.

Usage:
  finesub <input> [pipeline options...]   Run the pipeline (asr-pipeline flags)
  finesub batch [batch options...]        Run the batch runner
  finesub doctor                          Show runtime status and paths
  finesub relocate [--show|<dir>]         Move models/downloads/subtitles to
                                          another directory

Runs against this installation: same runtime, models, settings and knowledge
base as the app. Installing or repairing resources stays in the app itself.
"""


def _application_source(root: Path) -> Path:
    pointer = root / "app" / "current.json"
    candidates = []
    if pointer.is_file():
        try:
            current = json.loads(pointer.read_text(encoding="utf-8")).get("current")
        except (OSError, ValueError, AttributeError):
            current = None
        if isinstance(current, str) and current:
            candidates.append(root / "app" / "versions" / current)
    candidates.append(root)
    for candidate in candidates:
        if (candidate / "src" / "asr_playground" / "pipeline.py").is_file():
            return candidate.resolve()
    raise SystemExit(f"No FineSub application source under {root}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(USAGE, end="")
        return 0 if arguments else 2
    root = Path(__file__).resolve().parent
    source = _application_source(root)
    for entry in (str(source / "src"), str(source)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from finesub_bootstrap.shell import package_shell

    return package_shell(root).dispatch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
