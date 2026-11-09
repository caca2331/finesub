"""The prompt cold-read scripts still run against the current `src/` and docs.

They live in `agent-tasks/prompt-coldread/`, outside the suite's reach, so a
renamed owner-doc heading or a changed signature in `src/` broke them silently
twice: `docbundle.py` lost a section, and `render.py` raised `TypeError` on a
keyword `_claim_for_opening` no longer takes. These two checks take seconds; a
full render stays with the task's own `check.py`. The task is stripped from the
public tree, where both skip.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "agent-tasks" / "prompt-coldread" / "scripts"

pytestmark = pytest.mark.skipif(not SCRIPTS.is_dir(), reason="maintainer-only task, not in the public tree")


def test_every_heading_the_doc_bundle_cuts_still_exists() -> None:
    spec = importlib.util.spec_from_file_location("coldread_docbundle", SCRIPTS / "docbundle.py")
    docbundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(docbundle)

    missing = []
    for doc, heading in {*docbundle.COMMON, *(pair for pairs in docbundle.FAMILIES.values() for pair in pairs)}:
        try:
            docbundle.section(doc, heading)
        except SystemExit as gone:
            missing.append(str(gone))
    assert not missing, f"{missing}: point docbundle.py at the new heading"


def test_the_renderer_still_builds_every_agent_session() -> None:
    # render.py patches Path.read_text, sockets and subprocess when imported: its own process.
    code = ("import render\n"
            "for shell in render.AGENT_SHELLS:\n"
            "    render.c3p_window_parts(shell)\n"
            "render.c2_window()\n")
    run = subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS, capture_output=True, text=True,
                         encoding="utf-8", timeout=100)
    assert run.returncode == 0, (run.stderr[-3000:] + "\nFix the call in render.py and log it as a "
                                 "signature adaptation (agent-tasks/prompt-coldread/SKILL.md)")
